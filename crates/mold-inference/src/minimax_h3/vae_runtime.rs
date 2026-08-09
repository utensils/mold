//! Opened-file-bound production runtime for the shared Comfy MiniMax H3 VAEs.
//!
//! This module deliberately does not register a family, advertise a
//! capability, discover a checkpoint, or bypass the H3 policy gate. A future
//! authorized backend loader supplies four explicit paths from one frozen
//! Comfy task manifest. Each source is opened and identity-fenced exactly once.
//! The two weight files are authenticated while being copied into one private,
//! bounded staging directory; Candle may reopen only those private copies.
//! Both CUDA-resident models finish construction before staging is removed.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use candle_core::{DType, Device};
use mold_candle::minimax_h3::{
    expected_visual_vae_parameter_bytes, inspect_audio_vae_config_bytes,
    inspect_safetensors_header, inspect_visual_vae_config_bytes,
    load_validated_audio_vae_with_observer, validate_audio_safetensors, validate_comfy_weight_file,
    AudioTensorLayout, AudioVae, AudioVaeLoadObserver, DecodeComputePolicy, MiniMaxH3VisualVae,
    VisualAttentionBackend, VisualVaeEvent, VisualVaeObserver,
};
use mold_core::manifest::{find_manifest, storage_path, ModelComponent, ModelManifest};
use mold_core::minimax_h3::{self as contract, ArtifactRole as ContractArtifactRole, Layout, Task};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

const LOAD_PLAN_SCHEMA_VERSION: u32 = 1;
const AUTHENTICATION_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const VISUAL_CONFIG_SOURCE_PATH: &str = "vae/config.json";
const AUDIO_CONFIG_SOURCE_PATH: &str = "audio_vae/config.json";
const COMFY_VISUAL_WEIGHT_SOURCE_PATH: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
const COMFY_AUDIO_WEIGHT_SOURCE_PATH: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum H3ComfyVaeArtifactRole {
    VisualConfig,
    VisualWeights,
    AudioConfig,
    AudioWeights,
}

impl H3ComfyVaeArtifactRole {
    pub(crate) const ALL: [Self; 4] = [
        Self::VisualConfig,
        Self::VisualWeights,
        Self::AudioConfig,
        Self::AudioWeights,
    ];

    const WEIGHTS: [Self; 2] = [Self::VisualWeights, Self::AudioWeights];

    const fn stable_id(self) -> &'static str {
        match self {
            Self::VisualConfig => "visual-config",
            Self::VisualWeights => "visual-vae-fp16",
            Self::AudioConfig => "audio-config",
            Self::AudioWeights => "audio-vae-folded-fp32",
        }
    }

    const fn is_config(self) -> bool {
        matches!(self, Self::VisualConfig | Self::AudioConfig)
    }

    pub(crate) const fn source_path(self) -> &'static str {
        match self {
            Self::VisualConfig => VISUAL_CONFIG_SOURCE_PATH,
            Self::VisualWeights => COMFY_VISUAL_WEIGHT_SOURCE_PATH,
            Self::AudioConfig => AUDIO_CONFIG_SOURCE_PATH,
            Self::AudioWeights => COMFY_AUDIO_WEIGHT_SOURCE_PATH,
        }
    }

    pub(crate) const fn manifest_component(self) -> ModelComponent {
        match self {
            Self::VisualConfig | Self::AudioConfig => ModelComponent::ModelConfig,
            Self::VisualWeights => ModelComponent::Vae,
            Self::AudioWeights => ModelComponent::AudioVae,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum H3ComfyVaeLoadPhase {
    Open,
    Authenticate,
    Stage,
    ValidateConfig,
    ValidateHeader,
    Construct,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct H3ComfyVaeLoadEvent {
    pub(crate) role: H3ComfyVaeArtifactRole,
    pub(crate) phase: H3ComfyVaeLoadPhase,
    pub(crate) completed: u64,
    pub(crate) total: u64,
}

pub(crate) trait H3ComfyVaeLoadObserver {
    /// Return `false` to cancel at this bounded checkpoint.
    fn checkpoint(&mut self, event: H3ComfyVaeLoadEvent) -> bool;
}

#[derive(Default)]
pub(crate) struct NoopH3ComfyVaeLoadObserver;

impl H3ComfyVaeLoadObserver for NoopH3ComfyVaeLoadObserver {
    fn checkpoint(&mut self, _event: H3ComfyVaeLoadEvent) -> bool {
        true
    }
}

#[derive(Debug, Error)]
pub(crate) enum H3ComfyVaeLoadError {
    #[error("invalid MiniMax H3 Comfy VAE load plan: {0}")]
    InvalidPlan(String),
    #[error("MiniMax H3 Comfy VAE loading was cancelled during {phase:?} for {role:?}")]
    Cancelled {
        role: H3ComfyVaeArtifactRole,
        phase: H3ComfyVaeLoadPhase,
    },
    #[error("MiniMax H3 {role:?} artifact authority failed: {message}")]
    Artifact {
        role: H3ComfyVaeArtifactRole,
        message: String,
    },
    #[error(
        "MiniMax H3 VAE staging needs {required_bytes} bytes but only {available_bytes} bytes are available at {path}"
    )]
    InsufficientStaging {
        path: PathBuf,
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("MiniMax H3 {role:?} validation failed: {message}")]
    Validation {
        role: H3ComfyVaeArtifactRole,
        message: String,
    },
    #[error("MiniMax H3 {role:?} construction failed: {message}")]
    Construction {
        role: H3ComfyVaeArtifactRole,
        message: String,
    },
}

type LoadResult<T> = std::result::Result<T, H3ComfyVaeLoadError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ComfyVaeArtifactPaths {
    pub(crate) visual_config: PathBuf,
    pub(crate) visual_weights: PathBuf,
    pub(crate) audio_config: PathBuf,
    pub(crate) audio_weights: PathBuf,
}

impl H3ComfyVaeArtifactPaths {
    fn get(&self, role: H3ComfyVaeArtifactRole) -> &Path {
        match role {
            H3ComfyVaeArtifactRole::VisualConfig => &self.visual_config,
            H3ComfyVaeArtifactRole::VisualWeights => &self.visual_weights,
            H3ComfyVaeArtifactRole::AudioConfig => &self.audio_config,
            H3ComfyVaeArtifactRole::AudioWeights => &self.audio_weights,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedArtifact {
    role: H3ComfyVaeArtifactRole,
    source_repository: String,
    source_revision: String,
    source_path: String,
    content_sha256: String,
    file_bytes: u64,
}

impl ExpectedArtifact {
    fn basename(&self) -> LoadResult<&str> {
        Path::new(&self.source_path)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                H3ComfyVaeLoadError::InvalidPlan(format!(
                    "artifact {:?} has no UTF-8 basename",
                    self.role
                ))
            })
    }

    fn validate(&self) -> LoadResult<()> {
        if self.source_repository.trim().is_empty()
            || self.source_revision.len() != 40
            || !self
                .source_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.source_path.trim().is_empty()
            || !valid_sha256(&self.content_sha256)
            || self.file_bytes == 0
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "artifact {:?} has an incomplete pinned identity",
                self.role
            )));
        }
        if self.role.is_config() && self.file_bytes > MAX_CONFIG_BYTES {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "artifact {:?} exceeds the {MAX_CONFIG_BYTES}-byte config bound",
                self.role
            )));
        }
        Ok(())
    }
}

/// Exact task, artifact, local-path, and staging authority for one cold load.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FrozenH3ComfyVaeLoadPlan {
    schema_version: u32,
    canonical_model: String,
    task: Task,
    layout: Layout,
    paths: H3ComfyVaeArtifactPaths,
    staging_root: PathBuf,
    artifacts: Vec<ExpectedArtifact>,
    artifact_plan_identity_sha256: String,
    identity_sha256: String,
    #[cfg(test)]
    synthetic: bool,
}

impl FrozenH3ComfyVaeLoadPlan {
    /// Resolve the exact private VAE component paths from Mold's hidden,
    /// source-pinned manifest and canonical storage authority. No caller-
    /// mutable `ModelPaths` value participates in this plan.
    pub(crate) fn from_hidden_storage(
        model: &str,
        models_root: &Path,
        staging_root: &Path,
    ) -> LoadResult<Self> {
        let models_root = validate_private_storage_directory(models_root, "models root")?;
        let staging_root = validate_private_storage_directory(staging_root, "staging root")?;
        let manifest = find_manifest(model).ok_or_else(|| {
            H3ComfyVaeLoadError::InvalidPlan(format!("missing manifest {model:?}"))
        })?;
        if !manifest.hidden {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "private H3 VAE storage resolution requires a hidden manifest".into(),
            ));
        }
        let path_for = |role: H3ComfyVaeArtifactRole| -> LoadResult<PathBuf> {
            let expected_path = role.source_path();
            let matches = manifest
                .files
                .iter()
                .filter(|file| {
                    file.hf_filename == expected_path && file.component == role.manifest_component()
                })
                .collect::<Vec<_>>();
            let [file] = matches.as_slice() else {
                return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                    "hidden manifest {model:?} requires exactly one {expected_path}"
                )));
            };
            let relative = storage_path(manifest, file);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(H3ComfyVaeLoadError::InvalidPlan(
                    "hidden manifest produced a non-canonical VAE storage path".into(),
                ));
            }
            Ok(models_root.join(relative))
        };
        Self::new(
            model,
            H3ComfyVaeArtifactPaths {
                visual_config: path_for(H3ComfyVaeArtifactRole::VisualConfig)?,
                visual_weights: path_for(H3ComfyVaeArtifactRole::VisualWeights)?,
                audio_config: path_for(H3ComfyVaeArtifactRole::AudioConfig)?,
                audio_weights: path_for(H3ComfyVaeArtifactRole::AudioWeights)?,
            },
            staging_root,
        )
    }

    pub(crate) fn new(
        model: &str,
        paths: H3ComfyVaeArtifactPaths,
        staging_root: impl Into<PathBuf>,
    ) -> LoadResult<Self> {
        if !matches!(model, contract::FL2VA_COMFY | contract::REF2VA_COMFY) {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "{model:?} is not an exact canonical Comfy MiniMax H3 task"
            )));
        }
        let manifest = find_manifest(model).ok_or_else(|| {
            H3ComfyVaeLoadError::InvalidPlan(format!("missing manifest {model:?}"))
        })?;
        let manifest_contract = contract::manifest_contract(manifest).ok_or_else(|| {
            H3ComfyVaeLoadError::InvalidPlan(format!("{model:?} lost its H3 manifest contract"))
        })?;
        if manifest_contract.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq
            || manifest_contract.manifest_name != model
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "the requested model is not the pinned Comfy H3 layout".into(),
            ));
        }
        let artifacts = production_artifacts(manifest)?;
        let mut plan = Self {
            schema_version: LOAD_PLAN_SCHEMA_VERSION,
            canonical_model: model.to_string(),
            task: manifest_contract.task,
            layout: manifest_contract.layout,
            paths,
            staging_root: staging_root.into(),
            artifacts,
            artifact_plan_identity_sha256: String::new(),
            identity_sha256: String::new(),
            #[cfg(test)]
            synthetic: false,
        };
        plan.validate_fields()?;
        plan.artifact_plan_identity_sha256 = artifact_plan_identity(
            &plan.canonical_model,
            plan.task,
            plan.layout,
            &plan.artifacts,
        );
        plan.identity_sha256 = load_plan_identity(&plan);
        plan.validate()?;
        Ok(plan)
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub(crate) fn artifact_plan_identity_sha256(&self) -> &str {
        &self.artifact_plan_identity_sha256
    }

    pub(crate) const fn task(&self) -> Task {
        self.task
    }

    pub(crate) fn canonical_model(&self) -> &str {
        &self.canonical_model
    }

    fn artifact(&self, role: H3ComfyVaeArtifactRole) -> LoadResult<&ExpectedArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.role == role)
            .ok_or_else(|| {
                H3ComfyVaeLoadError::InvalidPlan(format!("missing artifact role {role:?}"))
            })
    }

    fn validate_fields(&self) -> LoadResult<()> {
        if self.schema_version != LOAD_PLAN_SCHEMA_VERSION
            || self.canonical_model.trim().is_empty()
            || self.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "schema, model, or layout authority changed after freezing".into(),
            ));
        }
        let roles = self
            .artifacts
            .iter()
            .map(|artifact| artifact.role)
            .collect::<BTreeSet<_>>();
        if self.artifacts.len() != H3ComfyVaeArtifactRole::ALL.len()
            || roles != H3ComfyVaeArtifactRole::ALL.into_iter().collect()
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "the VAE plan requires exactly four unique artifact roles".into(),
            ));
        }
        for artifact in &self.artifacts {
            artifact.validate()?;
        }
        let paths = H3ComfyVaeArtifactRole::ALL
            .into_iter()
            .map(|role| self.paths.get(role).to_path_buf())
            .collect::<BTreeSet<_>>();
        if paths.len() != H3ComfyVaeArtifactRole::ALL.len()
            || paths.iter().any(|path| path.as_os_str().is_empty())
            || self.staging_root.as_os_str().is_empty()
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "artifact paths and staging root must be explicit and distinct".into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> LoadResult<()> {
        self.validate_fields()?;
        if !valid_sha256(&self.artifact_plan_identity_sha256)
            || self.artifact_plan_identity_sha256
                != artifact_plan_identity(
                    &self.canonical_model,
                    self.task,
                    self.layout,
                    &self.artifacts,
                )
            || !valid_sha256(&self.identity_sha256)
            || self.identity_sha256 != load_plan_identity(self)
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "load-plan identity changed after freezing".into(),
            ));
        }
        #[cfg(test)]
        if self.synthetic {
            return Ok(());
        }
        let manifest = find_manifest(&self.canonical_model).ok_or_else(|| {
            H3ComfyVaeLoadError::InvalidPlan("the frozen H3 manifest disappeared".into())
        })?;
        let manifest_contract = contract::manifest_contract(manifest).ok_or_else(|| {
            H3ComfyVaeLoadError::InvalidPlan("the frozen H3 contract disappeared".into())
        })?;
        if manifest_contract.task != self.task
            || manifest_contract.layout != self.layout
            || production_artifacts(manifest)? != self.artifacts
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "task or artifact authority changed after freezing".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn synthetic(
        paths: H3ComfyVaeArtifactPaths,
        staging_root: PathBuf,
        artifacts: Vec<ExpectedArtifact>,
    ) -> LoadResult<Self> {
        let mut plan = Self {
            schema_version: LOAD_PLAN_SCHEMA_VERSION,
            canonical_model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            layout: Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
            paths,
            staging_root,
            artifacts,
            artifact_plan_identity_sha256: String::new(),
            identity_sha256: String::new(),
            synthetic: true,
        };
        plan.validate_fields()?;
        plan.artifact_plan_identity_sha256 = artifact_plan_identity(
            &plan.canonical_model,
            plan.task,
            plan.layout,
            &plan.artifacts,
        );
        plan.identity_sha256 = load_plan_identity(&plan);
        plan.validate()?;
        Ok(plan)
    }

    fn requires_cuda(&self) -> bool {
        #[cfg(test)]
        if self.synthetic {
            return false;
        }
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ComfyVaeComponentMemory {
    pub(crate) artifact_file_bytes: u64,
    pub(crate) header_bytes: u64,
    pub(crate) encoded_tensor_bytes: u64,
    pub(crate) resident_device_weight_bytes: u64,
    pub(crate) effective_device_weight_bytes: u64,
    pub(crate) construction_staging_disk_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ComfyVaeRuntimeMemory {
    pub(crate) visual: H3ComfyVaeComponentMemory,
    pub(crate) audio: H3ComfyVaeComponentMemory,
    pub(crate) config_bytes: u64,
    pub(crate) peak_host_io_buffer_bytes: u64,
    pub(crate) peak_host_mapped_file_bytes: u64,
    pub(crate) peak_staging_disk_bytes: u64,
    pub(crate) resident_host_weight_bytes: u64,
    pub(crate) resident_device_weight_bytes: u64,
    pub(crate) effective_device_weight_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ComfyVaeOpenedArtifactFact {
    pub(crate) role: H3ComfyVaeArtifactRole,
    pub(crate) source_repository: String,
    pub(crate) source_revision: String,
    pub(crate) source_path: String,
    pub(crate) content_sha256: String,
    pub(crate) file_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ComponentLoadFacts {
    artifact_file_bytes: u64,
    header_bytes: u64,
    encoded_tensor_bytes: u64,
    resident_device_weight_bytes: u64,
    effective_device_weight_bytes: u64,
}

impl ComponentLoadFacts {
    fn validate(&self, expected: &ExpectedArtifact) -> LoadResult<()> {
        if self.artifact_file_bytes != expected.file_bytes
            || self.header_bytes == 0
            || self.header_bytes.checked_add(self.encoded_tensor_bytes)
                != Some(self.artifact_file_bytes)
            || self.resident_device_weight_bytes == 0
            || self.resident_device_weight_bytes > self.encoded_tensor_bytes
            || self.effective_device_weight_bytes != self.resident_device_weight_bytes
        {
            return Err(H3ComfyVaeLoadError::Validation {
                role: expected.role,
                message: "Comfy VAE file, header, resident, and effective byte accounting disagree"
                    .into(),
            });
        }
        Ok(())
    }

    fn memory(&self) -> H3ComfyVaeComponentMemory {
        H3ComfyVaeComponentMemory {
            artifact_file_bytes: self.artifact_file_bytes,
            header_bytes: self.header_bytes,
            encoded_tensor_bytes: self.encoded_tensor_bytes,
            resident_device_weight_bytes: self.resident_device_weight_bytes,
            effective_device_weight_bytes: self.effective_device_weight_bytes,
            construction_staging_disk_bytes: self.artifact_file_bytes,
        }
    }
}

/// Fully authenticated VAE source and exact pre-allocation memory authority.
///
/// The token intentionally does not implement `Clone`. It owns all four
/// opened source descriptors and both private authenticated staging files,
/// and the construction function consumes it exactly once.
pub(crate) struct H3AuthenticatedComfyVaeAuthority {
    plan: FrozenH3ComfyVaeLoadPlan,
    opened: Vec<OpenedArtifact>,
    staged: StagedWeights,
    visual_config: Vec<u8>,
    audio_config: Vec<u8>,
    visual_facts: ComponentLoadFacts,
    audio_facts: ComponentLoadFacts,
    memory: H3ComfyVaeRuntimeMemory,
    artifact_facts: Vec<H3ComfyVaeOpenedArtifactFact>,
    identity_sha256: String,
}

impl H3AuthenticatedComfyVaeAuthority {
    pub(crate) const fn task(&self) -> Task {
        self.plan.task
    }

    pub(crate) fn canonical_model(&self) -> &str {
        &self.plan.canonical_model
    }

    pub(crate) fn source_path(&self, role: H3ComfyVaeArtifactRole) -> LoadResult<&Path> {
        self.opened
            .iter()
            .find(|artifact| artifact.expected.role == role)
            .map(|artifact| artifact.canonical_path.as_path())
            .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan(format!("missing opened {role:?}")))
    }

    pub(crate) fn plan_identity_sha256(&self) -> &str {
        self.plan.identity_sha256()
    }

    pub(crate) fn artifact_plan_identity_sha256(&self) -> &str {
        self.plan.artifact_plan_identity_sha256()
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub(crate) fn memory(&self) -> &H3ComfyVaeRuntimeMemory {
        &self.memory
    }

    pub(crate) fn artifact_facts(&self) -> &[H3ComfyVaeOpenedArtifactFact] {
        &self.artifact_facts
    }

    pub(crate) fn validate(&self) -> LoadResult<()> {
        self.plan.validate()?;
        for artifact in &self.opened {
            artifact.validate_source("while authenticated VAE authority is retained")?;
        }
        self.staged
            .validate("while authenticated VAE authority is retained")?;
        if self.identity_sha256 != authenticated_vae_authority_identity(self) {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "authenticated VAE authority identity changed".into(),
            ));
        }
        Ok(())
    }
}

/// The source files remain open and shared-locked for the complete runtime
/// lifetime. Models are constructed from authenticated private copies, so path
/// replacement cannot redirect Candle to different bytes; the lease still
/// fails closed if the frozen installed-artifact identity later changes.
pub(crate) struct H3ComfyVaeArtifactLease {
    plan_identity_sha256: String,
    authority_identity_sha256: String,
    artifacts: Vec<OpenedArtifact>,
}

impl H3ComfyVaeArtifactLease {
    pub(crate) fn plan_identity_sha256(&self) -> &str {
        &self.plan_identity_sha256
    }

    pub(crate) fn authority_identity_sha256(&self) -> &str {
        &self.authority_identity_sha256
    }

    pub(crate) fn is_active(&self) -> bool {
        self.validate().is_ok()
    }

    pub(crate) fn validate(&self) -> LoadResult<()> {
        if !valid_sha256(&self.plan_identity_sha256)
            || !valid_sha256(&self.authority_identity_sha256)
            || self.authority_identity_sha256
                != artifact_lease_identity(&self.plan_identity_sha256, &self.artifacts)
        {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "VAE artifact-lease identity changed".into(),
            ));
        }
        for artifact in &self.artifacts {
            artifact.validate_source("while the VAE runtime lease is active")?;
        }
        Ok(())
    }
}

/// Field order is load-bearing: CUDA model allocations drop before the source
/// artifact lease. The private construction staging directory has already been
/// removed before this bundle is returned.
pub(crate) struct H3ComfyVaeRuntimeBundle<V = MiniMaxH3VisualVae, A = AudioVae> {
    pub(crate) visual_vae: V,
    pub(crate) audio_vae: A,
    memory: H3ComfyVaeRuntimeMemory,
    task: Task,
    canonical_model: String,
    artifact_plan_identity_sha256: String,
    plan_identity_sha256: String,
    #[cfg(feature = "h3-private-uat")]
    device: Device,
    pub(crate) artifact_lease: H3ComfyVaeArtifactLease,
}

impl<V, A> H3ComfyVaeRuntimeBundle<V, A> {
    pub(crate) fn memory(&self) -> &H3ComfyVaeRuntimeMemory {
        &self.memory
    }

    pub(crate) const fn task(&self) -> Task {
        self.task
    }

    pub(crate) fn canonical_model(&self) -> &str {
        &self.canonical_model
    }

    pub(crate) fn plan_identity_sha256(&self) -> &str {
        &self.plan_identity_sha256
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn artifact_plan_identity_sha256(&self) -> &str {
        &self.artifact_plan_identity_sha256
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn authority_identity_sha256(&self) -> &str {
        self.artifact_lease.authority_identity_sha256()
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn validate_authority(&self) -> LoadResult<()> {
        if self.plan_identity_sha256 != self.artifact_lease.plan_identity_sha256() {
            return Err(H3ComfyVaeLoadError::InvalidPlan(
                "runtime and artifact lease plan identities disagree".into(),
            ));
        }
        self.artifact_lease.validate()
    }
}

pub(crate) fn load_h3_comfy_vae_runtime(
    plan: &FrozenH3ComfyVaeLoadPlan,
    device: &Device,
    observer: &mut dyn H3ComfyVaeLoadObserver,
) -> LoadResult<H3ComfyVaeRuntimeBundle> {
    let authority = open_h3_comfy_vae_authority(plan, observer)?;
    load_h3_comfy_vae_runtime_from_authority(authority, device, observer)
}

/// Open, completely authenticate, privately stage, and inspect the exact VAE
/// artifacts before any visual/audio model tensor allocation occurs.
pub(crate) fn open_h3_comfy_vae_authority(
    plan: &FrozenH3ComfyVaeLoadPlan,
    observer: &mut dyn H3ComfyVaeLoadObserver,
) -> LoadResult<H3AuthenticatedComfyVaeAuthority> {
    plan.validate()?;
    let mut opened = Vec::with_capacity(H3ComfyVaeArtifactRole::ALL.len());
    for role in H3ComfyVaeArtifactRole::ALL {
        opened.push(OpenedArtifact::open(
            plan.artifact(role)?.clone(),
            plan.paths.get(role),
            observer,
        )?);
    }
    let visual_config = opened_for_mut(&mut opened, H3ComfyVaeArtifactRole::VisualConfig)?
        .read_authenticated_config(observer)?;
    let audio_config = opened_for_mut(&mut opened, H3ComfyVaeArtifactRole::AudioConfig)?
        .read_authenticated_config(observer)?;
    let required_staging_bytes = required_staging_bytes(plan)?;
    ensure_staging_capacity(plan, required_staging_bytes)?;
    let staged = StagedWeights::create(plan, &mut opened, observer)?;
    for artifact in &opened {
        artifact.validate_source("before VAE pre-allocation inspection")?;
    }
    let visual_facts = inspect_visual_component(
        staged.path(H3ComfyVaeArtifactRole::VisualWeights)?,
        &visual_config,
        observer,
    )?;
    visual_facts.validate(plan.artifact(H3ComfyVaeArtifactRole::VisualWeights)?)?;
    let audio_facts = inspect_audio_component(
        staged.path(H3ComfyVaeArtifactRole::AudioWeights)?,
        &audio_config,
        observer,
    )?;
    audio_facts.validate(plan.artifact(H3ComfyVaeArtifactRole::AudioWeights)?)?;
    for artifact in &opened {
        artifact.validate_source("after VAE pre-allocation inspection")?;
    }
    let memory = vae_runtime_memory(
        &visual_facts,
        &audio_facts,
        visual_config.len(),
        audio_config.len(),
        required_staging_bytes,
    )?;
    let artifact_facts = opened
        .iter()
        .map(|artifact| H3ComfyVaeOpenedArtifactFact {
            role: artifact.expected.role,
            source_repository: artifact.expected.source_repository.clone(),
            source_revision: artifact.expected.source_revision.clone(),
            source_path: artifact.expected.source_path.clone(),
            content_sha256: artifact.expected.content_sha256.clone(),
            file_bytes: artifact.expected.file_bytes,
        })
        .collect();
    let mut authority = H3AuthenticatedComfyVaeAuthority {
        plan: plan.clone(),
        opened,
        staged,
        visual_config,
        audio_config,
        visual_facts,
        audio_facts,
        memory,
        artifact_facts,
        identity_sha256: String::new(),
    };
    authority.identity_sha256 = authenticated_vae_authority_identity(&authority);
    authority.validate()?;
    Ok(authority)
}

/// Consume one authenticated, non-Clone VAE authority and construct both
/// runtime models on the selected device.
pub(crate) fn load_h3_comfy_vae_runtime_from_authority(
    authority: H3AuthenticatedComfyVaeAuthority,
    device: &Device,
    observer: &mut dyn H3ComfyVaeLoadObserver,
) -> LoadResult<H3ComfyVaeRuntimeBundle> {
    authority.validate()?;
    if authority.plan.requires_cuda() && !device.is_cuda() {
        return Err(H3ComfyVaeLoadError::InvalidPlan(
            "production Comfy VAE construction requires the frozen CUDA route".into(),
        ));
    }
    let H3AuthenticatedComfyVaeAuthority {
        plan,
        opened,
        staged,
        visual_config,
        audio_config,
        visual_facts,
        audio_facts,
        memory,
        artifact_facts: _,
        identity_sha256: _,
    } = authority;
    let mut factory = ProductionVaeFactory;
    staged.validate("before visual VAE construction")?;
    let (visual_vae, loaded_visual_facts) = factory
        .load_visual(
            staged.path(H3ComfyVaeArtifactRole::VisualWeights)?,
            &visual_config,
            device,
            observer,
        )
        .map_err(factory_error)?;
    staged.validate("after visual VAE construction")?;
    staged.validate("before audio VAE construction")?;
    let (audio_vae, loaded_audio_facts) = factory
        .load_audio(
            staged.path(H3ComfyVaeArtifactRole::AudioWeights)?,
            &audio_config,
            device,
            observer,
        )
        .map_err(factory_error)?;
    staged.validate("after audio VAE construction")?;
    if loaded_visual_facts != visual_facts || loaded_audio_facts != audio_facts {
        return Err(H3ComfyVaeLoadError::InvalidPlan(
            "constructed VAE facts differ from authenticated pre-allocation evidence".into(),
        ));
    }
    for artifact in &opened {
        artifact.validate_source("after VAE construction")?;
    }
    drop(staged);
    let authority_identity_sha256 = artifact_lease_identity(plan.identity_sha256(), &opened);
    let artifact_lease = H3ComfyVaeArtifactLease {
        plan_identity_sha256: plan.identity_sha256().into(),
        authority_identity_sha256,
        artifacts: opened,
    };
    artifact_lease.validate()?;
    Ok(H3ComfyVaeRuntimeBundle {
        visual_vae,
        audio_vae,
        memory,
        task: plan.task,
        canonical_model: plan.canonical_model,
        artifact_plan_identity_sha256: plan.artifact_plan_identity_sha256,
        plan_identity_sha256: plan.identity_sha256,
        #[cfg(feature = "h3-private-uat")]
        device: device.clone(),
        artifact_lease,
    })
}

fn required_staging_bytes(plan: &FrozenH3ComfyVaeLoadPlan) -> LoadResult<u64> {
    H3ComfyVaeArtifactRole::WEIGHTS
        .into_iter()
        .try_fold(0_u64, |total, role| {
            total
                .checked_add(plan.artifact(role)?.file_bytes)
                .ok_or_else(|| {
                    H3ComfyVaeLoadError::InvalidPlan("VAE staging byte count overflow".into())
                })
        })
}

fn ensure_staging_capacity(plan: &FrozenH3ComfyVaeLoadPlan, required_bytes: u64) -> LoadResult<()> {
    let available_bytes = fs2::available_space(&plan.staging_root).map_err(|error| {
        H3ComfyVaeLoadError::InvalidPlan(format!(
            "cannot inspect staging capacity at {}: {error}",
            plan.staging_root.display()
        ))
    })?;
    if available_bytes < required_bytes {
        return Err(H3ComfyVaeLoadError::InsufficientStaging {
            path: plan.staging_root.clone(),
            required_bytes,
            available_bytes,
        });
    }
    Ok(())
}

fn inspect_visual_component(
    weight_path: &Path,
    config_bytes: &[u8],
    observer: &mut dyn H3ComfyVaeLoadObserver,
) -> LoadResult<ComponentLoadFacts> {
    if !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::VisualConfig,
        H3ComfyVaeLoadPhase::ValidateConfig,
        0,
        1,
    ) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role: H3ComfyVaeArtifactRole::VisualConfig,
            phase: H3ComfyVaeLoadPhase::ValidateConfig,
        });
    }
    let config = inspect_visual_vae_config_bytes(config_bytes).map_err(|error| {
        H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::VisualConfig,
            message: error.to_string(),
        }
    })?;
    if !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::VisualConfig,
        H3ComfyVaeLoadPhase::ValidateConfig,
        1,
        1,
    ) || !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::VisualWeights,
        H3ComfyVaeLoadPhase::ValidateHeader,
        0,
        1,
    ) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            phase: H3ComfyVaeLoadPhase::ValidateHeader,
        });
    }
    let header = inspect_safetensors_header(weight_path).map_err(|error| {
        H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            message: error.to_string(),
        }
    })?;
    let validated = validate_comfy_weight_file(&config, weight_path).map_err(|error| {
        H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            message: error.to_string(),
        }
    })?;
    let resident_device_weight_bytes = expected_visual_vae_parameter_bytes(&config, DType::F16)
        .map_err(|error| H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            message: error.to_string(),
        })?;
    if !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::VisualWeights,
        H3ComfyVaeLoadPhase::ValidateHeader,
        1,
        1,
    ) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            phase: H3ComfyVaeLoadPhase::ValidateHeader,
        });
    }
    let header_bytes =
        header
            .header_len
            .checked_add(8)
            .ok_or_else(|| H3ComfyVaeLoadError::Validation {
                role: H3ComfyVaeArtifactRole::VisualWeights,
                message: "visual header byte overflow".into(),
            })?;
    let encoded_tensor_bytes = header.file_len.checked_sub(header_bytes).ok_or_else(|| {
        H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            message: "visual tensor byte underflow".into(),
        }
    })?;
    if validated.inspection().total_size != header.file_len {
        return Err(H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::VisualWeights,
            message: "visual validated file bytes differ from opened header".into(),
        });
    }
    Ok(ComponentLoadFacts {
        artifact_file_bytes: header.file_len,
        header_bytes,
        encoded_tensor_bytes,
        resident_device_weight_bytes,
        effective_device_weight_bytes: resident_device_weight_bytes,
    })
}

fn inspect_audio_component(
    weight_path: &Path,
    config_bytes: &[u8],
    observer: &mut dyn H3ComfyVaeLoadObserver,
) -> LoadResult<ComponentLoadFacts> {
    if !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::AudioConfig,
        H3ComfyVaeLoadPhase::ValidateConfig,
        0,
        1,
    ) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role: H3ComfyVaeArtifactRole::AudioConfig,
            phase: H3ComfyVaeLoadPhase::ValidateConfig,
        });
    }
    let config = inspect_audio_vae_config_bytes(config_bytes).map_err(|error| {
        H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::AudioConfig,
            message: error.to_string(),
        }
    })?;
    if !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::AudioConfig,
        H3ComfyVaeLoadPhase::ValidateConfig,
        1,
        1,
    ) || !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::AudioWeights,
        H3ComfyVaeLoadPhase::ValidateHeader,
        0,
        1,
    ) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role: H3ComfyVaeArtifactRole::AudioWeights,
            phase: H3ComfyVaeLoadPhase::ValidateHeader,
        });
    }
    let validated =
        validate_audio_safetensors(weight_path, &config, AudioTensorLayout::ComfyFolded).map_err(
            |error| H3ComfyVaeLoadError::Validation {
                role: H3ComfyVaeArtifactRole::AudioWeights,
                message: error.to_string(),
            },
        )?;
    let file_bytes = std::fs::metadata(weight_path)
        .map_err(|error| H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::AudioWeights,
            message: error.to_string(),
        })?
        .len();
    if !checkpoint(
        observer,
        H3ComfyVaeArtifactRole::AudioWeights,
        H3ComfyVaeLoadPhase::ValidateHeader,
        1,
        1,
    ) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role: H3ComfyVaeArtifactRole::AudioWeights,
            phase: H3ComfyVaeLoadPhase::ValidateHeader,
        });
    }
    let header_bytes = file_bytes
        .checked_sub(validated.tensor_bytes)
        .ok_or_else(|| H3ComfyVaeLoadError::Validation {
            role: H3ComfyVaeArtifactRole::AudioWeights,
            message: "audio header byte underflow".into(),
        })?;
    Ok(ComponentLoadFacts {
        artifact_file_bytes: file_bytes,
        header_bytes,
        encoded_tensor_bytes: validated.tensor_bytes,
        resident_device_weight_bytes: validated.tensor_bytes,
        effective_device_weight_bytes: validated.tensor_bytes,
    })
}

fn vae_runtime_memory(
    visual: &ComponentLoadFacts,
    audio: &ComponentLoadFacts,
    visual_config_bytes: usize,
    audio_config_bytes: usize,
    required_staging_bytes: u64,
) -> LoadResult<H3ComfyVaeRuntimeMemory> {
    let visual = visual.memory();
    let audio = audio.memory();
    let resident_device_weight_bytes = visual
        .resident_device_weight_bytes
        .checked_add(audio.resident_device_weight_bytes)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan("resident VAE byte overflow".into()))?;
    let effective_device_weight_bytes = visual
        .effective_device_weight_bytes
        .checked_add(audio.effective_device_weight_bytes)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan("effective VAE byte overflow".into()))?;
    let config_bytes = u64::try_from(visual_config_bytes)
        .ok()
        .and_then(|visual| {
            u64::try_from(audio_config_bytes)
                .ok()
                .and_then(|audio| visual.checked_add(audio))
        })
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan("VAE config byte overflow".into()))?;
    Ok(H3ComfyVaeRuntimeMemory {
        peak_host_mapped_file_bytes: visual.artifact_file_bytes.max(audio.artifact_file_bytes),
        visual,
        audio,
        config_bytes,
        peak_host_io_buffer_bytes: AUTHENTICATION_CHUNK_BYTES as u64,
        peak_staging_disk_bytes: required_staging_bytes,
        resident_host_weight_bytes: 0,
        resident_device_weight_bytes,
        effective_device_weight_bytes,
    })
}

fn authenticated_vae_authority_identity(authority: &H3AuthenticatedComfyVaeAuthority) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.comfy-vae-authenticated-open.v1\0");
    digest.update(authority.plan.identity_sha256().as_bytes());
    digest.update(artifact_lease_identity(
        authority.plan.identity_sha256(),
        &authority.opened,
    ));
    authority.staged.update_identity(&mut digest);
    for fact in &authority.artifact_facts {
        digest.update(fact.role.stable_id().as_bytes());
        digest.update([0]);
        digest.update(fact.content_sha256.as_bytes());
        digest.update(fact.file_bytes.to_le_bytes());
    }
    for value in [
        authority.memory.visual.artifact_file_bytes,
        authority.memory.visual.header_bytes,
        authority.memory.visual.encoded_tensor_bytes,
        authority.memory.visual.resident_device_weight_bytes,
        authority.memory.audio.artifact_file_bytes,
        authority.memory.audio.header_bytes,
        authority.memory.audio.encoded_tensor_bytes,
        authority.memory.audio.resident_device_weight_bytes,
        authority.memory.config_bytes,
        authority.memory.peak_host_io_buffer_bytes,
        authority.memory.peak_host_mapped_file_bytes,
        authority.memory.peak_staging_disk_bytes,
        authority.memory.resident_host_weight_bytes,
        authority.memory.resident_device_weight_bytes,
        authority.memory.effective_device_weight_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

trait VaeFactory {
    type Visual;
    type Audio;

    fn load_visual(
        &mut self,
        weight_path: &Path,
        config_bytes: &[u8],
        device: &Device,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> std::result::Result<(Self::Visual, ComponentLoadFacts), FactoryError>;

    fn load_audio(
        &mut self,
        weight_path: &Path,
        config_bytes: &[u8],
        device: &Device,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> std::result::Result<(Self::Audio, ComponentLoadFacts), FactoryError>;
}

enum FactoryError {
    Cancelled {
        role: H3ComfyVaeArtifactRole,
        phase: H3ComfyVaeLoadPhase,
    },
    Validation {
        role: H3ComfyVaeArtifactRole,
        message: String,
    },
    Construction {
        role: H3ComfyVaeArtifactRole,
        message: String,
    },
}

impl FactoryError {
    fn cancelled(role: H3ComfyVaeArtifactRole, phase: H3ComfyVaeLoadPhase) -> Self {
        Self::Cancelled { role, phase }
    }

    fn validation(role: H3ComfyVaeArtifactRole, error: impl ToString) -> Self {
        Self::Validation {
            role,
            message: error.to_string(),
        }
    }

    fn construction(role: H3ComfyVaeArtifactRole, error: impl ToString) -> Self {
        Self::Construction {
            role,
            message: error.to_string(),
        }
    }
}

struct ProductionVaeFactory;

impl VaeFactory for ProductionVaeFactory {
    type Visual = MiniMaxH3VisualVae;
    type Audio = AudioVae;

    fn load_visual(
        &mut self,
        weight_path: &Path,
        config_bytes: &[u8],
        device: &Device,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> std::result::Result<(Self::Visual, ComponentLoadFacts), FactoryError> {
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::VisualConfig,
            H3ComfyVaeLoadPhase::ValidateConfig,
            0,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::VisualConfig,
                H3ComfyVaeLoadPhase::ValidateConfig,
            ));
        }
        let config = inspect_visual_vae_config_bytes(config_bytes).map_err(|error| {
            FactoryError::validation(H3ComfyVaeArtifactRole::VisualConfig, error)
        })?;
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::VisualConfig,
            H3ComfyVaeLoadPhase::ValidateConfig,
            1,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::VisualConfig,
                H3ComfyVaeLoadPhase::ValidateConfig,
            ));
        }
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::VisualWeights,
            H3ComfyVaeLoadPhase::ValidateHeader,
            0,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::VisualWeights,
                H3ComfyVaeLoadPhase::ValidateHeader,
            ));
        }
        let header = inspect_safetensors_header(weight_path).map_err(|error| {
            FactoryError::validation(H3ComfyVaeArtifactRole::VisualWeights, error)
        })?;
        let validated = validate_comfy_weight_file(&config, weight_path).map_err(|error| {
            FactoryError::validation(H3ComfyVaeArtifactRole::VisualWeights, error)
        })?;
        let resident_device_weight_bytes = expected_visual_vae_parameter_bytes(&config, DType::F16)
            .map_err(|error| {
                FactoryError::validation(H3ComfyVaeArtifactRole::VisualWeights, error)
            })?;
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::VisualWeights,
            H3ComfyVaeLoadPhase::ValidateHeader,
            1,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::VisualWeights,
                H3ComfyVaeLoadPhase::ValidateHeader,
            ));
        }
        let mut visual_observer = VisualConstructionObserver {
            observer,
            cancelled: false,
        };
        let model = MiniMaxH3VisualVae::from_validated(
            config,
            &validated,
            device,
            DecodeComputePolicy::Reference,
            VisualAttentionBackend::Auto,
            &mut visual_observer,
        );
        let model = match model {
            Ok(model) => model,
            Err(_) if visual_observer.cancelled => {
                return Err(FactoryError::cancelled(
                    H3ComfyVaeArtifactRole::VisualWeights,
                    H3ComfyVaeLoadPhase::Construct,
                ));
            }
            Err(error) => {
                return Err(FactoryError::construction(
                    H3ComfyVaeArtifactRole::VisualWeights,
                    error,
                ));
            }
        };
        if model.stored_weight_dtype() != DType::F16 || model.decode_compute_dtype() != DType::F16 {
            return Err(FactoryError::construction(
                H3ComfyVaeArtifactRole::VisualWeights,
                "Comfy visual VAE did not preserve FP16 resident/reference execution",
            ));
        }
        let header_bytes = header.header_len.checked_add(8).ok_or_else(|| {
            FactoryError::validation(
                H3ComfyVaeArtifactRole::VisualWeights,
                "visual header byte overflow",
            )
        })?;
        let tensor_bytes = header.file_len.checked_sub(header_bytes).ok_or_else(|| {
            FactoryError::validation(
                H3ComfyVaeArtifactRole::VisualWeights,
                "visual tensor byte underflow",
            )
        })?;
        Ok((
            model,
            ComponentLoadFacts {
                artifact_file_bytes: header.file_len,
                header_bytes,
                encoded_tensor_bytes: tensor_bytes,
                resident_device_weight_bytes,
                effective_device_weight_bytes: resident_device_weight_bytes,
            },
        ))
    }

    fn load_audio(
        &mut self,
        weight_path: &Path,
        config_bytes: &[u8],
        device: &Device,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> std::result::Result<(Self::Audio, ComponentLoadFacts), FactoryError> {
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::AudioConfig,
            H3ComfyVaeLoadPhase::ValidateConfig,
            0,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::AudioConfig,
                H3ComfyVaeLoadPhase::ValidateConfig,
            ));
        }
        let config = inspect_audio_vae_config_bytes(config_bytes).map_err(|error| {
            FactoryError::validation(H3ComfyVaeArtifactRole::AudioConfig, error)
        })?;
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::AudioConfig,
            H3ComfyVaeLoadPhase::ValidateConfig,
            1,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::AudioConfig,
                H3ComfyVaeLoadPhase::ValidateConfig,
            ));
        }
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::AudioWeights,
            H3ComfyVaeLoadPhase::ValidateHeader,
            0,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::AudioWeights,
                H3ComfyVaeLoadPhase::ValidateHeader,
            ));
        }
        let validated =
            validate_audio_safetensors(weight_path, &config, AudioTensorLayout::ComfyFolded)
                .map_err(|error| {
                    FactoryError::validation(H3ComfyVaeArtifactRole::AudioWeights, error)
                })?;
        let file_bytes = std::fs::metadata(weight_path)
            .map_err(|error| FactoryError::validation(H3ComfyVaeArtifactRole::AudioWeights, error))?
            .len();
        if !checkpoint(
            observer,
            H3ComfyVaeArtifactRole::AudioWeights,
            H3ComfyVaeLoadPhase::ValidateHeader,
            1,
            1,
        ) {
            return Err(FactoryError::cancelled(
                H3ComfyVaeArtifactRole::AudioWeights,
                H3ComfyVaeLoadPhase::ValidateHeader,
            ));
        }
        let mut audio_observer = AudioConstructionObserver {
            observer,
            cancelled: false,
        };
        // SAFETY: `weight_path` is a read-only file inside the loader's private
        // staging directory. That directory remains owned until construction
        // completes, and this production path accepts CUDA only, so every
        // returned parameter tensor is copied into device storage before the
        // staging directory is removed.
        let model = unsafe {
            load_validated_audio_vae_with_observer(
                weight_path,
                config,
                AudioTensorLayout::ComfyFolded,
                DType::F32,
                device,
                &mut audio_observer,
            )
        };
        let model = match model {
            Ok(model) => model,
            Err(_) if audio_observer.cancelled => {
                return Err(FactoryError::cancelled(
                    H3ComfyVaeArtifactRole::AudioWeights,
                    H3ComfyVaeLoadPhase::Construct,
                ));
            }
            Err(error) => {
                return Err(FactoryError::construction(
                    H3ComfyVaeArtifactRole::AudioWeights,
                    error,
                ));
            }
        };
        let header_bytes = file_bytes
            .checked_sub(validated.tensor_bytes)
            .ok_or_else(|| {
                FactoryError::validation(
                    H3ComfyVaeArtifactRole::AudioWeights,
                    "audio header byte underflow",
                )
            })?;
        Ok((
            model,
            ComponentLoadFacts {
                artifact_file_bytes: file_bytes,
                header_bytes,
                encoded_tensor_bytes: validated.tensor_bytes,
                resident_device_weight_bytes: validated.tensor_bytes,
                effective_device_weight_bytes: validated.tensor_bytes,
            },
        ))
    }
}

struct VisualConstructionObserver<'a> {
    observer: &'a mut dyn H3ComfyVaeLoadObserver,
    cancelled: bool,
}

impl VisualVaeObserver for VisualConstructionObserver<'_> {
    fn checkpoint(&mut self, event: VisualVaeEvent) -> candle_core::Result<()> {
        if checkpoint(
            self.observer,
            H3ComfyVaeArtifactRole::VisualWeights,
            H3ComfyVaeLoadPhase::Construct,
            event.completed as u64,
            event.total.max(1) as u64,
        ) {
            Ok(())
        } else {
            self.cancelled = true;
            candle_core::bail!("MiniMax H3 visual VAE construction cancelled")
        }
    }
}

struct AudioConstructionObserver<'a> {
    observer: &'a mut dyn H3ComfyVaeLoadObserver,
    cancelled: bool,
}

impl AudioVaeLoadObserver for AudioConstructionObserver<'_> {
    fn checkpoint(&mut self, completed: usize, total: usize) -> candle_core::Result<()> {
        if checkpoint(
            self.observer,
            H3ComfyVaeArtifactRole::AudioWeights,
            H3ComfyVaeLoadPhase::Construct,
            completed as u64,
            total.max(1) as u64,
        ) {
            Ok(())
        } else {
            self.cancelled = true;
            candle_core::bail!("MiniMax H3 audio VAE construction cancelled")
        }
    }
}

fn load_with_factory<F: VaeFactory>(
    plan: &FrozenH3ComfyVaeLoadPlan,
    device: &Device,
    observer: &mut dyn H3ComfyVaeLoadObserver,
    factory: &mut F,
) -> LoadResult<H3ComfyVaeRuntimeBundle<F::Visual, F::Audio>> {
    plan.validate()?;
    if plan.requires_cuda() && !device.is_cuda() {
        return Err(H3ComfyVaeLoadError::InvalidPlan(
            "production Comfy VAE construction requires the frozen CUDA route".into(),
        ));
    }
    let mut opened = Vec::with_capacity(H3ComfyVaeArtifactRole::ALL.len());
    for role in H3ComfyVaeArtifactRole::ALL {
        opened.push(OpenedArtifact::open(
            plan.artifact(role)?.clone(),
            plan.paths.get(role),
            observer,
        )?);
    }

    let visual_config = opened_for_mut(&mut opened, H3ComfyVaeArtifactRole::VisualConfig)?
        .read_authenticated_config(observer)?;
    let audio_config = opened_for_mut(&mut opened, H3ComfyVaeArtifactRole::AudioConfig)?
        .read_authenticated_config(observer)?;
    let required_staging_bytes =
        H3ComfyVaeArtifactRole::WEIGHTS
            .into_iter()
            .try_fold(0_u64, |total, role| {
                total
                    .checked_add(plan.artifact(role)?.file_bytes)
                    .ok_or_else(|| {
                        H3ComfyVaeLoadError::InvalidPlan("VAE staging byte count overflow".into())
                    })
            })?;
    let available_bytes = fs2::available_space(&plan.staging_root).map_err(|error| {
        H3ComfyVaeLoadError::InvalidPlan(format!(
            "cannot inspect staging capacity at {}: {error}",
            plan.staging_root.display()
        ))
    })?;
    if available_bytes < required_staging_bytes {
        return Err(H3ComfyVaeLoadError::InsufficientStaging {
            path: plan.staging_root.clone(),
            required_bytes: required_staging_bytes,
            available_bytes,
        });
    }
    let staged = StagedWeights::create(plan, &mut opened, observer)?;
    for artifact in &opened {
        artifact.validate_source("before VAE construction")?;
    }

    let (visual_vae, visual_facts) = factory
        .load_visual(
            staged.path(H3ComfyVaeArtifactRole::VisualWeights)?,
            &visual_config,
            device,
            observer,
        )
        .map_err(factory_error)?;
    visual_facts.validate(plan.artifact(H3ComfyVaeArtifactRole::VisualWeights)?)?;
    let (audio_vae, audio_facts) = factory
        .load_audio(
            staged.path(H3ComfyVaeArtifactRole::AudioWeights)?,
            &audio_config,
            device,
            observer,
        )
        .map_err(factory_error)?;
    audio_facts.validate(plan.artifact(H3ComfyVaeArtifactRole::AudioWeights)?)?;
    for artifact in &opened {
        artifact.validate_source("after VAE construction")?;
    }

    let visual_memory = visual_facts.memory();
    let audio_memory = audio_facts.memory();
    let resident_device_weight_bytes = visual_memory
        .resident_device_weight_bytes
        .checked_add(audio_memory.resident_device_weight_bytes)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan("resident VAE byte overflow".into()))?;
    let effective_device_weight_bytes = visual_memory
        .effective_device_weight_bytes
        .checked_add(audio_memory.effective_device_weight_bytes)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan("effective VAE byte overflow".into()))?;
    let config_bytes = (visual_config.len() as u64)
        .checked_add(audio_config.len() as u64)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan("VAE config byte overflow".into()))?;
    let memory = H3ComfyVaeRuntimeMemory {
        peak_host_mapped_file_bytes: visual_memory
            .artifact_file_bytes
            .max(audio_memory.artifact_file_bytes),
        visual: visual_memory,
        audio: audio_memory,
        config_bytes,
        peak_host_io_buffer_bytes: AUTHENTICATION_CHUNK_BYTES as u64,
        peak_staging_disk_bytes: required_staging_bytes,
        resident_host_weight_bytes: 0,
        resident_device_weight_bytes,
        effective_device_weight_bytes,
    };
    drop(staged);

    let authority_identity_sha256 = artifact_lease_identity(plan.identity_sha256(), &opened);
    let artifact_lease = H3ComfyVaeArtifactLease {
        plan_identity_sha256: plan.identity_sha256().into(),
        authority_identity_sha256,
        artifacts: opened,
    };
    artifact_lease.validate()?;
    Ok(H3ComfyVaeRuntimeBundle {
        visual_vae,
        audio_vae,
        memory,
        task: plan.task,
        canonical_model: plan.canonical_model.clone(),
        artifact_plan_identity_sha256: plan.artifact_plan_identity_sha256.clone(),
        plan_identity_sha256: plan.identity_sha256.clone(),
        #[cfg(feature = "h3-private-uat")]
        device: device.clone(),
        artifact_lease,
    })
}

fn factory_error(error: FactoryError) -> H3ComfyVaeLoadError {
    match error {
        FactoryError::Cancelled { role, phase } => H3ComfyVaeLoadError::Cancelled { role, phase },
        FactoryError::Validation { role, message } => {
            H3ComfyVaeLoadError::Validation { role, message }
        }
        FactoryError::Construction { role, message } => {
            H3ComfyVaeLoadError::Construction { role, message }
        }
    }
}

fn checkpoint(
    observer: &mut dyn H3ComfyVaeLoadObserver,
    role: H3ComfyVaeArtifactRole,
    phase: H3ComfyVaeLoadPhase,
    completed: u64,
    total: u64,
) -> bool {
    observer.checkpoint(H3ComfyVaeLoadEvent {
        role,
        phase,
        completed,
        total,
    })
}

struct StagedWeights {
    _directory: tempfile::TempDir,
    visual: StagedWeight,
    audio: StagedWeight,
}

struct StagedWeight {
    role: H3ComfyVaeArtifactRole,
    path: PathBuf,
    identity: FileIdentity,
    file: File,
}

impl StagedWeights {
    fn create(
        plan: &FrozenH3ComfyVaeLoadPlan,
        opened: &mut [OpenedArtifact],
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> LoadResult<Self> {
        let directory = tempfile::Builder::new()
            .prefix(".mold-h3-vae-stage-")
            .tempdir_in(&plan.staging_root)
            .map_err(|error| {
                H3ComfyVaeLoadError::InvalidPlan(format!(
                    "cannot create private H3 VAE staging directory at {}: {error}",
                    plan.staging_root.display()
                ))
            })?;
        let visual = stage_weight(
            opened_for_mut(opened, H3ComfyVaeArtifactRole::VisualWeights)?,
            directory.path(),
            observer,
        )?;
        let audio = stage_weight(
            opened_for_mut(opened, H3ComfyVaeArtifactRole::AudioWeights)?,
            directory.path(),
            observer,
        )?;
        Ok(Self {
            _directory: directory,
            visual,
            audio,
        })
    }

    fn path(&self, role: H3ComfyVaeArtifactRole) -> LoadResult<&Path> {
        match role {
            H3ComfyVaeArtifactRole::VisualWeights => Ok(&self.visual.path),
            H3ComfyVaeArtifactRole::AudioWeights => Ok(&self.audio.path),
            _ => Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "artifact {role:?} is not a staged weight"
            ))),
        }
    }

    fn validate(&self, operation: &str) -> LoadResult<()> {
        self.visual.validate(operation)?;
        self.audio.validate(operation)
    }

    fn update_identity(&self, digest: &mut Sha256) {
        self.visual.update_identity(digest);
        self.audio.update_identity(digest);
    }
}

impl StagedWeight {
    fn validate(&self, operation: &str) -> LoadResult<()> {
        let descriptor_identity = file_identity(&self.file.metadata().map_err(|error| {
            artifact_error(
                self.role,
                format!("cannot stat retained staging descriptor {operation}: {error}"),
            )
        })?)?;
        let path_metadata = std::fs::symlink_metadata(&self.path).map_err(|error| {
            artifact_error(
                self.role,
                format!("cannot stat private staging path {operation}: {error}"),
            )
        })?;
        if path_metadata.file_type().is_symlink()
            || !path_metadata.file_type().is_file()
            || descriptor_identity != self.identity
            || file_identity(&path_metadata)? != self.identity
        {
            return Err(artifact_error(
                self.role,
                format!("retained staging descriptor or path changed {operation}"),
            ));
        }
        Ok(())
    }

    fn update_identity(&self, digest: &mut Sha256) {
        digest.update(self.role.stable_id().as_bytes());
        digest.update([0]);
        let path = self.path.as_os_str().as_encoded_bytes();
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path);
        digest.update(self.identity.len.to_le_bytes());
        #[cfg(unix)]
        for value in [
            self.identity.device,
            self.identity.inode,
            self.identity.modified_seconds as u64,
            self.identity.modified_nanoseconds as u64,
            self.identity.changed_seconds as u64,
            self.identity.changed_nanoseconds as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
    }
}

fn stage_weight(
    artifact: &mut OpenedArtifact,
    directory: &Path,
    observer: &mut dyn H3ComfyVaeLoadObserver,
) -> LoadResult<StagedWeight> {
    let role = artifact.expected.role;
    let target = directory.join(artifact.expected.basename()?);
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut output = options.open(&target).map_err(|error| {
        artifact_error(
            role,
            format!(
                "cannot create private staging file {}: {error}",
                target.display()
            ),
        )
    })?;
    artifact
        .file
        .seek(SeekFrom::Start(0))
        .map_err(|error| artifact_error(role, format!("cannot rewind opened source: {error}")))?;
    let total = artifact.expected.file_bytes;
    if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Stage, 0, total) {
        return Err(H3ComfyVaeLoadError::Cancelled {
            role,
            phase: H3ComfyVaeLoadPhase::Stage,
        });
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; AUTHENTICATION_CHUNK_BYTES];
    let mut completed = 0_u64;
    while completed < total {
        let remaining = usize::try_from((total - completed).min(buffer.len() as u64))
            .map_err(|_| artifact_error(role, "staging read bound overflows usize"))?;
        artifact
            .file
            .read_exact(&mut buffer[..remaining])
            .map_err(|error| artifact_error(role, format!("opened source truncated: {error}")))?;
        output
            .write_all(&buffer[..remaining])
            .map_err(|error| artifact_error(role, format!("staging write failed: {error}")))?;
        digest.update(&buffer[..remaining]);
        completed = completed
            .checked_add(remaining as u64)
            .ok_or_else(|| artifact_error(role, "staging progress overflow"))?;
        if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Stage, completed, total) {
            return Err(H3ComfyVaeLoadError::Cancelled {
                role,
                phase: H3ComfyVaeLoadPhase::Stage,
            });
        }
    }
    let mut extra = [0_u8; 1];
    if artifact
        .file
        .read(&mut extra)
        .map_err(|error| artifact_error(role, format!("cannot fence source length: {error}")))?
        != 0
    {
        return Err(artifact_error(role, "opened source grew while staging"));
    }
    output
        .flush()
        .map_err(|error| artifact_error(role, format!("cannot flush staging file: {error}")))?;
    let digest = format!("{:x}", digest.finalize());
    if !digest.eq_ignore_ascii_case(&artifact.expected.content_sha256) {
        return Err(artifact_error(
            role,
            format!(
                "content SHA-256 mismatch: expected {}, found {digest}",
                artifact.expected.content_sha256
            ),
        ));
    }
    let mut permissions = output
        .metadata()
        .map_err(|error| artifact_error(role, format!("cannot stat staging file: {error}")))?
        .permissions();
    permissions.set_readonly(true);
    output
        .set_permissions(permissions)
        .map_err(|error| artifact_error(role, format!("cannot seal staging file: {error}")))?;
    let identity = file_identity(&output.metadata().map_err(|error| {
        artifact_error(
            role,
            format!("cannot stat sealed staging descriptor: {error}"),
        )
    })?)?;
    if identity.len != total {
        return Err(artifact_error(role, "sealed staging length changed"));
    }
    artifact.validate_source("after authentication and staging")?;
    let staged = StagedWeight {
        role,
        path: target,
        identity,
        file: output,
    };
    staged.validate("after sealing")?;
    Ok(staged)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

struct OpenedArtifact {
    expected: ExpectedArtifact,
    requested_path: PathBuf,
    canonical_path: PathBuf,
    identity: FileIdentity,
    file: File,
}

impl OpenedArtifact {
    fn open(
        expected: ExpectedArtifact,
        requested_path: &Path,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> LoadResult<Self> {
        let role = expected.role;
        if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Open, 0, 1) {
            return Err(H3ComfyVaeLoadError::Cancelled {
                role,
                phase: H3ComfyVaeLoadPhase::Open,
            });
        }
        let requested_metadata = std::fs::symlink_metadata(requested_path).map_err(|error| {
            artifact_error(
                role,
                format!("cannot inspect {}: {error}", requested_path.display()),
            )
        })?;
        if requested_metadata.file_type().is_symlink() || !requested_metadata.file_type().is_file()
        {
            return Err(artifact_error(
                role,
                format!(
                    "source must be a non-symlink regular file: {}",
                    requested_path.display()
                ),
            ));
        }
        #[cfg(unix)]
        {
            // SAFETY: `geteuid` has no preconditions and only reads process
            // state.
            let effective_uid = unsafe { libc::geteuid() };
            if requested_metadata.uid() != effective_uid || requested_metadata.mode() & 0o022 != 0 {
                return Err(artifact_error(
                    role,
                    "source must be process-owned and not group/other writable",
                ));
            }
        }
        let canonical_path = std::fs::canonicalize(requested_path).map_err(|error| {
            artifact_error(
                role,
                format!("cannot canonicalize {}: {error}", requested_path.display()),
            )
        })?;
        let file = File::open(&canonical_path).map_err(|error| {
            artifact_error(
                role,
                format!("cannot open {}: {error}", canonical_path.display()),
            )
        })?;
        fs2::FileExt::lock_shared(&file).map_err(|error| {
            artifact_error(
                role,
                format!("cannot acquire immutable shared file lease: {error}"),
            )
        })?;
        let identity = file_identity(&file.metadata().map_err(|error| {
            artifact_error(role, format!("cannot stat opened descriptor: {error}"))
        })?)?;
        if identity.len != expected.file_bytes {
            return Err(artifact_error(
                role,
                format!(
                    "expected {} bytes, found {}",
                    expected.file_bytes, identity.len
                ),
            ));
        }
        let artifact = Self {
            expected,
            requested_path: requested_path.to_path_buf(),
            canonical_path,
            identity,
            file,
        };
        artifact.validate_source("immediately after opening")?;
        if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Open, 1, 1) {
            return Err(H3ComfyVaeLoadError::Cancelled {
                role,
                phase: H3ComfyVaeLoadPhase::Open,
            });
        }
        Ok(artifact)
    }

    fn read_authenticated_config(
        &mut self,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> LoadResult<Vec<u8>> {
        let role = self.expected.role;
        if !role.is_config() {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "artifact {role:?} is not a config"
            )));
        }
        let total = self.expected.file_bytes;
        let capacity = usize::try_from(total)
            .map_err(|_| artifact_error(role, "config size overflows usize"))?;
        let mut bytes = Vec::with_capacity(capacity);
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; AUTHENTICATION_CHUNK_BYTES.min(capacity.max(1))];
        self.file.seek(SeekFrom::Start(0)).map_err(|error| {
            artifact_error(role, format!("cannot rewind opened config: {error}"))
        })?;
        if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Authenticate, 0, total) {
            return Err(H3ComfyVaeLoadError::Cancelled {
                role,
                phase: H3ComfyVaeLoadPhase::Authenticate,
            });
        }
        while bytes.len() < capacity {
            let remaining = (capacity - bytes.len()).min(buffer.len());
            self.file
                .read_exact(&mut buffer[..remaining])
                .map_err(|error| artifact_error(role, format!("config truncated: {error}")))?;
            digest.update(&buffer[..remaining]);
            bytes.extend_from_slice(&buffer[..remaining]);
            if !checkpoint(
                observer,
                role,
                H3ComfyVaeLoadPhase::Authenticate,
                bytes.len() as u64,
                total,
            ) {
                return Err(H3ComfyVaeLoadError::Cancelled {
                    role,
                    phase: H3ComfyVaeLoadPhase::Authenticate,
                });
            }
        }
        let mut extra = [0_u8; 1];
        if self
            .file
            .read(&mut extra)
            .map_err(|error| artifact_error(role, format!("cannot fence config length: {error}")))?
            != 0
        {
            return Err(artifact_error(role, "config grew while authenticating"));
        }
        let observed = format!("{:x}", digest.finalize());
        if !observed.eq_ignore_ascii_case(&self.expected.content_sha256) {
            return Err(artifact_error(
                role,
                format!(
                    "content SHA-256 mismatch: expected {}, found {observed}",
                    self.expected.content_sha256
                ),
            ));
        }
        self.validate_source("after config authentication")?;
        Ok(bytes)
    }

    fn validate_source(&self, operation: &str) -> LoadResult<()> {
        let role = self.expected.role;
        let requested_metadata =
            std::fs::symlink_metadata(&self.requested_path).map_err(|error| {
                artifact_error(
                    role,
                    format!("source path disappeared {operation}: {error}"),
                )
            })?;
        if requested_metadata.file_type().is_symlink() || !requested_metadata.file_type().is_file()
        {
            return Err(artifact_error(
                role,
                format!("source path is no longer a regular file {operation}"),
            ));
        }
        let canonical = std::fs::canonicalize(&self.requested_path).map_err(|error| {
            artifact_error(
                role,
                format!("cannot canonicalize source {operation}: {error}"),
            )
        })?;
        let descriptor_identity = file_identity(&self.file.metadata().map_err(|error| {
            artifact_error(role, format!("cannot stat descriptor {operation}: {error}"))
        })?)?;
        let path_identity = file_identity(&std::fs::metadata(&canonical).map_err(|error| {
            artifact_error(
                role,
                format!("cannot stat source path {operation}: {error}"),
            )
        })?)?;
        if canonical != self.canonical_path
            || descriptor_identity != self.identity
            || path_identity != self.identity
        {
            return Err(artifact_error(
                role,
                format!("opened descriptor or source-path identity changed {operation}"),
            ));
        }
        Ok(())
    }
}

fn opened_for_mut(
    opened: &mut [OpenedArtifact],
    role: H3ComfyVaeArtifactRole,
) -> LoadResult<&mut OpenedArtifact> {
    opened
        .iter_mut()
        .find(|artifact| artifact.expected.role == role)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan(format!("missing opened {role:?}")))
}

fn file_identity(metadata: &std::fs::Metadata) -> LoadResult<FileIdentity> {
    if !metadata.file_type().is_file() {
        return Err(H3ComfyVaeLoadError::InvalidPlan(
            "opened VAE artifact is not a regular file".into(),
        ));
    }
    Ok(FileIdentity {
        len: metadata.len(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(unix)]
        modified_seconds: metadata.mtime(),
        #[cfg(unix)]
        modified_nanoseconds: metadata.mtime_nsec(),
        #[cfg(unix)]
        changed_seconds: metadata.ctime(),
        #[cfg(unix)]
        changed_nanoseconds: metadata.ctime_nsec(),
        #[cfg(not(unix))]
        modified: metadata.modified().ok(),
    })
}

fn validate_private_storage_directory(path: &Path, label: &str) -> LoadResult<PathBuf> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
            "private H3 {label} must be an absolute canonical path"
        )));
    }
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        H3ComfyVaeLoadError::InvalidPlan(format!(
            "cannot inspect private H3 {label} {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
            "private H3 {label} must be a real directory"
        )));
    }
    #[cfg(unix)]
    {
        // SAFETY: `geteuid` has no preconditions and only reads the process
        // effective user identifier.
        let effective_uid = unsafe { libc::geteuid() };
        if metadata.uid() != effective_uid || metadata.mode() & 0o022 != 0 {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "private H3 {label} must be process-owned and not group/other writable"
            )));
        }
    }
    #[cfg(not(unix))]
    return Err(H3ComfyVaeLoadError::InvalidPlan(
        "private H3 storage resolution currently requires Unix ownership semantics".into(),
    ));
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        H3ComfyVaeLoadError::InvalidPlan(format!("cannot canonicalize private H3 {label}: {error}"))
    })?;
    if canonical != path {
        return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
            "private H3 {label} must not contain aliases or symlink components"
        )));
    }
    Ok(canonical)
}

fn production_artifacts(manifest: &ModelManifest) -> LoadResult<Vec<ExpectedArtifact>> {
    [
        (
            H3ComfyVaeArtifactRole::VisualConfig,
            VISUAL_CONFIG_SOURCE_PATH,
            ModelComponent::ModelConfig,
            ContractArtifactRole::SharedConfig,
            "config",
        ),
        (
            H3ComfyVaeArtifactRole::VisualWeights,
            COMFY_VISUAL_WEIGHT_SOURCE_PATH,
            ModelComponent::Vae,
            ContractArtifactRole::VideoVae,
            "fp16",
        ),
        (
            H3ComfyVaeArtifactRole::AudioConfig,
            AUDIO_CONFIG_SOURCE_PATH,
            ModelComponent::ModelConfig,
            ContractArtifactRole::SharedConfig,
            "config",
        ),
        (
            H3ComfyVaeArtifactRole::AudioWeights,
            COMFY_AUDIO_WEIGHT_SOURCE_PATH,
            ModelComponent::AudioVae,
            ContractArtifactRole::AudioVae,
            "fp32",
        ),
    ]
    .into_iter()
    .map(|(role, source_path, component, contract_role, dtype)| {
        let matches = manifest
            .files
            .iter()
            .filter(|file| file.hf_filename == source_path && file.component == component)
            .collect::<Vec<_>>();
        let [file] = matches.as_slice() else {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "manifest {} requires exactly one {source_path}",
                manifest.name
            )));
        };
        let artifact_contract = contract::artifact_contract(manifest, file).ok_or_else(|| {
            H3ComfyVaeLoadError::InvalidPlan(format!(
                "manifest {} lost artifact contract {source_path}",
                manifest.name
            ))
        })?;
        if artifact_contract.role != contract_role || artifact_contract.dtype != dtype {
            return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
                "manifest {} changed role or dtype for {source_path}",
                manifest.name
            )));
        }
        Ok(ExpectedArtifact {
            role,
            source_repository: artifact_contract.identity.source_repo.to_string(),
            source_revision: artifact_contract.identity.source_revision.to_string(),
            source_path: artifact_contract.identity.source_path.to_string(),
            content_sha256: artifact_contract.identity.sha256.to_ascii_lowercase(),
            file_bytes: file.size_bytes,
        })
    })
    .collect()
}

/// Path-independent authority for the exact four-artifact Comfy VAE plan.
/// Admission and the opened-file runtime derive this independently from the
/// pinned manifest; local source/staging paths remain confined to the
/// attempt-specific load-plan identity below.
pub(crate) fn expected_h3_comfy_vae_artifact_plan_identity(model: &str) -> LoadResult<String> {
    let manifest = find_manifest(model)
        .ok_or_else(|| H3ComfyVaeLoadError::InvalidPlan(format!("missing manifest {model:?}")))?;
    let manifest_contract = contract::manifest_contract(manifest).ok_or_else(|| {
        H3ComfyVaeLoadError::InvalidPlan(format!("{model:?} lost its H3 manifest contract"))
    })?;
    if manifest_contract.manifest_name != model
        || manifest_contract.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq
    {
        return Err(H3ComfyVaeLoadError::InvalidPlan(format!(
            "{model:?} is not an exact canonical Comfy MiniMax H3 task"
        )));
    }
    let artifacts = production_artifacts(manifest)?;
    Ok(artifact_plan_identity(
        model,
        manifest_contract.task,
        manifest_contract.layout,
        &artifacts,
    ))
}

fn artifact_plan_identity(
    canonical_model: &str,
    task: Task,
    layout: Layout,
    artifacts: &[ExpectedArtifact],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.comfy-vae-artifact-plan.v1\0");
    digest.update(canonical_model.as_bytes());
    digest.update([0]);
    digest.update(match task {
        Task::Fl2va => b"fl2va".as_slice(),
        Task::Ref2va => b"ref2va".as_slice(),
    });
    digest.update([0]);
    digest.update(match layout {
        Layout::OfficialBf16 => b"official-bf16".as_slice(),
        Layout::ComfyPrunedInt8ConvrotNvfp4Awq => b"comfy-pruned-int8".as_slice(),
    });
    for artifact in artifacts {
        digest.update([0]);
        digest.update(artifact.role.stable_id().as_bytes());
        digest.update([0]);
        digest.update(artifact.source_repository.as_bytes());
        digest.update([0]);
        digest.update(artifact.source_revision.as_bytes());
        digest.update([0]);
        digest.update(artifact.source_path.as_bytes());
        digest.update([0]);
        digest.update(artifact.content_sha256.as_bytes());
        digest.update(artifact.file_bytes.to_le_bytes());
    }
    digest.update(b"visual-resident-fp16\0visual-reference-compute-fp16\0");
    digest.update(b"audio-resident-fp32\0audio-compute-fp32\0");
    format!("{:x}", digest.finalize())
}

fn artifact_error(role: H3ComfyVaeArtifactRole, message: impl Into<String>) -> H3ComfyVaeLoadError {
    H3ComfyVaeLoadError::Artifact {
        role,
        message: message.into(),
    }
}

fn load_plan_identity(plan: &FrozenH3ComfyVaeLoadPlan) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.comfy-vae-load-plan.v1\0");
    digest.update(plan.schema_version.to_le_bytes());
    digest.update(plan.canonical_model.as_bytes());
    digest.update([0]);
    digest.update(match plan.task {
        Task::Fl2va => b"fl2va".as_slice(),
        Task::Ref2va => b"ref2va".as_slice(),
    });
    digest.update([0]);
    digest.update(b"comfy-pruned-int8");
    for artifact in &plan.artifacts {
        digest.update([0]);
        digest.update(artifact.role.stable_id().as_bytes());
        digest.update([0]);
        digest.update(artifact.source_repository.as_bytes());
        digest.update([0]);
        digest.update(artifact.source_revision.as_bytes());
        digest.update([0]);
        digest.update(artifact.source_path.as_bytes());
        digest.update([0]);
        digest.update(artifact.content_sha256.as_bytes());
        digest.update(artifact.file_bytes.to_le_bytes());
        update_path_digest(&mut digest, plan.paths.get(artifact.role));
    }
    update_path_digest(&mut digest, &plan.staging_root);
    digest.update(b"visual-resident-fp16\0visual-reference-compute-fp16\0");
    digest.update(b"audio-resident-fp32\0audio-compute-fp32\0");
    #[cfg(test)]
    digest.update([u8::from(plan.synthetic)]);
    format!("{:x}", digest.finalize())
}

fn update_path_digest(digest: &mut Sha256, path: &Path) {
    let bytes = path.as_os_str().as_encoded_bytes();
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn artifact_lease_identity(plan_identity: &str, artifacts: &[OpenedArtifact]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.comfy-vae-artifact-lease.v1\0");
    digest.update(plan_identity.as_bytes());
    for artifact in artifacts {
        digest.update([0]);
        digest.update(artifact.expected.role.stable_id().as_bytes());
        digest.update([0]);
        digest.update(artifact.expected.content_sha256.as_bytes());
        update_path_digest(&mut digest, &artifact.canonical_path);
        update_file_identity(&mut digest, &artifact.identity);
    }
    format!("{:x}", digest.finalize())
}

fn update_file_identity(digest: &mut Sha256, identity: &FileIdentity) {
    digest.update(identity.len.to_le_bytes());
    #[cfg(unix)]
    for value in [
        identity.device,
        identity.inode,
        identity.modified_seconds as u64,
        identity.modified_nanoseconds as u64,
        identity.changed_seconds as u64,
        identity.changed_nanoseconds as u64,
    ] {
        digest.update(value.to_le_bytes());
    }
    #[cfg(not(unix))]
    digest.update(format!("{:?}", identity.modified).as_bytes());
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Debug)]
    struct FakeVisual(u8);

    impl FakeVisual {
        fn decode(&self, input: u8) -> u8 {
            self.0.wrapping_add(input)
        }
    }

    #[derive(Debug)]
    struct FakeAudio(u8);

    impl FakeAudio {
        fn decode(&self, input: u8) -> u8 {
            self.0.wrapping_mul(input)
        }
    }

    struct FakeFactory;

    impl VaeFactory for FakeFactory {
        type Visual = FakeVisual;
        type Audio = FakeAudio;

        fn load_visual(
            &mut self,
            weight_path: &Path,
            _config_bytes: &[u8],
            _device: &Device,
            observer: &mut dyn H3ComfyVaeLoadObserver,
        ) -> std::result::Result<(Self::Visual, ComponentLoadFacts), FactoryError> {
            fake_load(H3ComfyVaeArtifactRole::VisualWeights, weight_path, observer)
                .map(|(byte, facts)| (FakeVisual(byte), facts))
        }

        fn load_audio(
            &mut self,
            weight_path: &Path,
            _config_bytes: &[u8],
            _device: &Device,
            observer: &mut dyn H3ComfyVaeLoadObserver,
        ) -> std::result::Result<(Self::Audio, ComponentLoadFacts), FactoryError> {
            fake_load(H3ComfyVaeArtifactRole::AudioWeights, weight_path, observer)
                .map(|(byte, facts)| (FakeAudio(byte), facts))
        }
    }

    fn fake_load(
        role: H3ComfyVaeArtifactRole,
        path: &Path,
        observer: &mut dyn H3ComfyVaeLoadObserver,
    ) -> std::result::Result<(u8, ComponentLoadFacts), FactoryError> {
        if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Construct, 0, 1) {
            return Err(FactoryError::cancelled(
                role,
                H3ComfyVaeLoadPhase::Construct,
            ));
        }
        let bytes = std::fs::read(path).map_err(|error| FactoryError::construction(role, error))?;
        if bytes.len() < 2 {
            return Err(FactoryError::construction(
                role,
                "synthetic artifact is too short",
            ));
        }
        if !checkpoint(observer, role, H3ComfyVaeLoadPhase::Construct, 1, 1) {
            return Err(FactoryError::cancelled(
                role,
                H3ComfyVaeLoadPhase::Construct,
            ));
        }
        let tensor_bytes = bytes.len() as u64 - 1;
        Ok((
            bytes[1],
            ComponentLoadFacts {
                artifact_file_bytes: bytes.len() as u64,
                header_bytes: 1,
                encoded_tensor_bytes: tensor_bytes,
                resident_device_weight_bytes: tensor_bytes,
                effective_device_weight_bytes: tensor_bytes,
            },
        ))
    }

    #[derive(Default)]
    struct RecordingObserver {
        events: Vec<H3ComfyVaeLoadEvent>,
        cancel_stage_after: Option<u64>,
    }

    impl H3ComfyVaeLoadObserver for RecordingObserver {
        fn checkpoint(&mut self, event: H3ComfyVaeLoadEvent) -> bool {
            self.events.push(event);
            !matches!(
                self.cancel_stage_after,
                Some(limit)
                    if event.phase == H3ComfyVaeLoadPhase::Stage
                        && event.completed >= limit
            )
        }
    }

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn write_artifacts(
        source: &Path,
        visual_weight_bytes: &[u8],
    ) -> (H3ComfyVaeArtifactPaths, Vec<ExpectedArtifact>) {
        let values = [
            (
                H3ComfyVaeArtifactRole::VisualConfig,
                "visual.json",
                b"visual-config".as_slice(),
            ),
            (
                H3ComfyVaeArtifactRole::VisualWeights,
                "visual.safetensors",
                visual_weight_bytes,
            ),
            (
                H3ComfyVaeArtifactRole::AudioConfig,
                "audio.json",
                b"audio-config".as_slice(),
            ),
            (
                H3ComfyVaeArtifactRole::AudioWeights,
                "audio.safetensors",
                b"\x08audio".as_slice(),
            ),
        ];
        for (_, name, bytes) in values {
            std::fs::write(source.join(name), bytes).unwrap();
        }
        let paths = H3ComfyVaeArtifactPaths {
            visual_config: source.join("visual.json"),
            visual_weights: source.join("visual.safetensors"),
            audio_config: source.join("audio.json"),
            audio_weights: source.join("audio.safetensors"),
        };
        let artifacts = values
            .into_iter()
            .map(|(role, name, bytes)| ExpectedArtifact {
                role,
                source_repository: "synthetic/repository".into(),
                source_revision: "a".repeat(40),
                source_path: name.into(),
                content_sha256: sha(bytes),
                file_bytes: bytes.len() as u64,
            })
            .collect();
        (paths, artifacts)
    }

    #[test]
    fn production_plans_pin_exact_shared_comfy_artifacts_and_task() {
        let root = tempfile::tempdir().unwrap();
        let paths = H3ComfyVaeArtifactPaths {
            visual_config: root.path().join("visual-config"),
            visual_weights: root.path().join("visual-weights"),
            audio_config: root.path().join("audio-config"),
            audio_weights: root.path().join("audio-weights"),
        };
        let fl = FrozenH3ComfyVaeLoadPlan::new(contract::FL2VA_COMFY, paths.clone(), root.path())
            .unwrap();
        let reference =
            FrozenH3ComfyVaeLoadPlan::new(contract::REF2VA_COMFY, paths, root.path()).unwrap();
        assert_eq!(fl.task(), Task::Fl2va);
        assert_eq!(reference.task(), Task::Ref2va);
        assert_ne!(fl.identity_sha256(), reference.identity_sha256());
        assert_ne!(
            fl.artifact_plan_identity_sha256(),
            reference.artifact_plan_identity_sha256()
        );
        assert_eq!(
            fl.artifact_plan_identity_sha256(),
            expected_h3_comfy_vae_artifact_plan_identity(contract::FL2VA_COMFY).unwrap()
        );
        let visual = fl.artifact(H3ComfyVaeArtifactRole::VisualWeights).unwrap();
        assert_eq!(visual.file_bytes, 5_207_808_496);
        assert_eq!(
            visual.content_sha256,
            "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
        );
        let audio = fl.artifact(H3ComfyVaeArtifactRole::AudioWeights).unwrap();
        assert_eq!(audio.file_bytes, 605_254_808);
        assert_eq!(
            audio.content_sha256,
            "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
        );
        assert!(
            FrozenH3ComfyVaeLoadPlan::new("minimax-h3", fl.paths.clone(), root.path()).is_err()
        );
    }

    #[test]
    fn hidden_storage_plan_uses_manifest_paths_without_model_paths() {
        let models = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let models_root = models.path().canonicalize().unwrap();
        let staging_root = staging.path().canonicalize().unwrap();
        let plan = FrozenH3ComfyVaeLoadPlan::from_hidden_storage(
            contract::FL2VA_COMFY,
            &models_root,
            &staging_root,
        )
        .unwrap();
        let manifest = find_manifest(contract::FL2VA_COMFY).unwrap();
        for role in H3ComfyVaeArtifactRole::ALL {
            let source = plan.artifact(role).unwrap().source_path.as_str();
            let file = manifest
                .files
                .iter()
                .find(|file| file.hf_filename == source)
                .unwrap();
            assert_eq!(
                plan.paths.get(role),
                models_root.join(storage_path(manifest, file))
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn hidden_storage_plan_rejects_alias_and_writable_roots() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let parent = tempfile::tempdir().unwrap();
        let models = parent.path().join("models");
        let staging = parent.path().join("staging");
        std::fs::create_dir(&models).unwrap();
        std::fs::create_dir(&staging).unwrap();
        let models = models.canonicalize().unwrap();
        let staging = staging.canonicalize().unwrap();
        let alias = parent.path().join("models-alias");
        symlink(&models, &alias).unwrap();
        assert!(FrozenH3ComfyVaeLoadPlan::from_hidden_storage(
            contract::FL2VA_COMFY,
            &alias,
            &staging,
        )
        .is_err());

        std::fs::set_permissions(&models, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(FrozenH3ComfyVaeLoadPlan::from_hidden_storage(
            contract::FL2VA_COMFY,
            &models,
            &staging,
        )
        .is_err());
    }

    #[test]
    fn synthetic_loader_is_opened_file_bound_and_reports_exact_memory() {
        let source = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let (paths, artifacts) = write_artifacts(source.path(), b"\x08visual");
        let plan = FrozenH3ComfyVaeLoadPlan::synthetic(
            paths.clone(),
            staging.path().to_path_buf(),
            artifacts,
        )
        .unwrap();
        let mut observer = RecordingObserver::default();
        let bundle =
            load_with_factory(&plan, &Device::Cpu, &mut observer, &mut FakeFactory).unwrap();
        assert_eq!(bundle.visual_vae.decode(3), b'v'.wrapping_add(3));
        assert_eq!(bundle.audio_vae.decode(2), b'a'.wrapping_mul(2));
        assert_eq!(bundle.memory().resident_host_weight_bytes, 0);
        assert_eq!(bundle.memory().resident_device_weight_bytes, 11);
        assert_eq!(bundle.memory().effective_device_weight_bytes, 11);
        assert_eq!(bundle.memory().peak_staging_disk_bytes, 13);
        assert_eq!(bundle.memory().peak_host_io_buffer_bytes, 1024 * 1024);
        assert!(observer.events.iter().any(|event| {
            event.role == H3ComfyVaeArtifactRole::VisualWeights
                && event.phase == H3ComfyVaeLoadPhase::Stage
                && event.completed == event.total
        }));
        assert_eq!(std::fs::read_dir(staging.path()).unwrap().count(), 0);

        std::fs::write(&paths.visual_weights, b"replacement").unwrap();
        assert!(!bundle.artifact_lease.is_active());
        assert_eq!(bundle.visual_vae.decode(3), b'v'.wrapping_add(3));
    }

    #[test]
    fn cancellation_during_bounded_staging_removes_private_files() {
        let source = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let large = vec![7_u8; AUTHENTICATION_CHUNK_BYTES + 17];
        let (paths, artifacts) = write_artifacts(source.path(), &large);
        let plan =
            FrozenH3ComfyVaeLoadPlan::synthetic(paths, staging.path().to_path_buf(), artifacts)
                .unwrap();
        let mut observer = RecordingObserver {
            cancel_stage_after: Some(AUTHENTICATION_CHUNK_BYTES as u64),
            ..Default::default()
        };
        let error = load_with_factory(&plan, &Device::Cpu, &mut observer, &mut FakeFactory)
            .err()
            .unwrap();
        assert!(matches!(
            error,
            H3ComfyVaeLoadError::Cancelled {
                role: H3ComfyVaeArtifactRole::VisualWeights,
                phase: H3ComfyVaeLoadPhase::Stage,
            }
        ));
        assert_eq!(std::fs::read_dir(staging.path()).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn opened_vae_source_rejects_group_writable_artifacts() {
        use std::os::unix::fs::PermissionsExt;

        let source = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let (paths, artifacts) = write_artifacts(source.path(), b"\x08visual");
        std::fs::set_permissions(
            &paths.visual_weights,
            std::fs::Permissions::from_mode(0o666),
        )
        .unwrap();
        let plan =
            FrozenH3ComfyVaeLoadPlan::synthetic(paths, staging.path().to_path_buf(), artifacts)
                .unwrap();
        let error = load_with_factory(
            &plan,
            &Device::Cpu,
            &mut RecordingObserver::default(),
            &mut FakeFactory,
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            H3ComfyVaeLoadError::Artifact {
                role: H3ComfyVaeArtifactRole::VisualWeights,
                ..
            }
        ));
    }

    #[test]
    fn authenticated_staging_descriptor_rejects_path_replacement_before_allocation() {
        let source = tempfile::tempdir().unwrap();
        let staging_root = tempfile::tempdir().unwrap();
        let (paths, artifacts) = write_artifacts(source.path(), b"\x08visual");
        let plan = FrozenH3ComfyVaeLoadPlan::synthetic(
            paths,
            staging_root.path().to_path_buf(),
            artifacts,
        )
        .unwrap();
        let mut observer = RecordingObserver::default();
        let mut opened = H3ComfyVaeArtifactRole::WEIGHTS
            .into_iter()
            .map(|role| {
                OpenedArtifact::open(
                    plan.artifact(role).unwrap().clone(),
                    plan.paths.get(role),
                    &mut observer,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let staged = StagedWeights::create(&plan, &mut opened, &mut observer).unwrap();
        let target = staged
            .path(H3ComfyVaeArtifactRole::VisualWeights)
            .unwrap()
            .to_path_buf();
        let authenticated = target.with_extension("authenticated");
        std::fs::rename(&target, &authenticated).unwrap();
        std::fs::write(&target, b"replacement").unwrap();

        let error = staged.validate("before allocation").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("staging descriptor or path changed"),
            "{error}"
        );
    }

    struct ReplacingObserver {
        target: PathBuf,
        replacement_done: Arc<Mutex<bool>>,
    }

    impl H3ComfyVaeLoadObserver for ReplacingObserver {
        fn checkpoint(&mut self, event: H3ComfyVaeLoadEvent) -> bool {
            if event.role == H3ComfyVaeArtifactRole::VisualWeights
                && event.phase == H3ComfyVaeLoadPhase::Construct
                && !*self.replacement_done.lock().unwrap()
            {
                let replacement = self.target.with_extension("replacement");
                std::fs::write(&replacement, b"replacement").unwrap();
                std::fs::rename(replacement, &self.target).unwrap();
                *self.replacement_done.lock().unwrap() = true;
            }
            true
        }
    }

    #[test]
    fn source_replacement_during_construction_fails_the_lease_fence() {
        let source = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let (paths, artifacts) = write_artifacts(source.path(), b"\x08visual");
        let plan = FrozenH3ComfyVaeLoadPlan::synthetic(
            paths.clone(),
            staging.path().to_path_buf(),
            artifacts,
        )
        .unwrap();
        let replaced = Arc::new(Mutex::new(false));
        let mut observer = ReplacingObserver {
            target: paths.visual_weights,
            replacement_done: Arc::clone(&replaced),
        };
        let error = load_with_factory(&plan, &Device::Cpu, &mut observer, &mut FakeFactory)
            .err()
            .unwrap();
        assert!(*replaced.lock().unwrap());
        assert!(matches!(
            error,
            H3ComfyVaeLoadError::Artifact {
                role: H3ComfyVaeArtifactRole::VisualWeights,
                ..
            }
        ));
        assert_eq!(std::fs::read_dir(staging.path()).unwrap().count(), 0);
    }

    /// Opt-in private smoke for the exact released Comfy VAE pair. This test
    /// constructs CUDA models directly through this private module and proves
    /// the public capability authority remains disabled before and after.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires authorized private MiniMax H3 artifacts and a CUDA device"]
    fn private_released_artifacts_construct_without_public_activation() {
        let artifact_root = PathBuf::from(
            std::env::var("MOLD_H3_PRIVATE_VAE_ARTIFACT_ROOT")
                .expect("set MOLD_H3_PRIVATE_VAE_ARTIFACT_ROOT"),
        );
        let staging_root = PathBuf::from(
            std::env::var("MOLD_H3_PRIVATE_VAE_STAGING_ROOT")
                .expect("set MOLD_H3_PRIVATE_VAE_STAGING_ROOT"),
        );
        let paths = H3ComfyVaeArtifactPaths {
            visual_config: artifact_root.join("vae/config.json"),
            visual_weights: artifact_root.join(COMFY_VISUAL_WEIGHT_SOURCE_PATH),
            audio_config: artifact_root.join("audio_vae/config.json"),
            audio_weights: artifact_root.join(COMFY_AUDIO_WEIGHT_SOURCE_PATH),
        };
        let plan =
            FrozenH3ComfyVaeLoadPlan::new(contract::FL2VA_COMFY, paths, staging_root.clone())
                .expect("freeze private VAE load plan");
        let device = Device::new_cuda(0).expect("private VAE smoke requires CUDA device zero");
        let mut observer = RecordingObserver::default();
        assert!(!contract::capabilities(Task::Fl2va).runtime_available);
        assert!(contract::runnable_capability_contract_for_model(contract::FL2VA_COMFY).is_none());

        let started = std::time::Instant::now();
        let bundle = load_h3_comfy_vae_runtime(&plan, &device, &mut observer)
            .expect("construct exact private Comfy VAE runtime");
        bundle.validate_authority().expect("validate VAE authority");
        assert_eq!(bundle.task(), Task::Fl2va);
        assert_eq!(bundle.canonical_model(), contract::FL2VA_COMFY);
        assert_eq!(bundle.plan_identity_sha256(), plan.identity_sha256());
        assert_eq!(bundle.memory().visual.artifact_file_bytes, 5_207_808_496);
        assert_eq!(
            bundle.memory().visual.resident_device_weight_bytes,
            5_207_737_968
        );
        assert_eq!(bundle.memory().audio.artifact_file_bytes, 605_254_808);
        assert_eq!(bundle.memory().resident_host_weight_bytes, 0);
        assert!(observer.events.iter().any(|event| {
            event.role == H3ComfyVaeArtifactRole::VisualWeights
                && event.phase == H3ComfyVaeLoadPhase::Construct
                && event.completed == event.total
        }));
        assert!(observer.events.iter().any(|event| {
            event.role == H3ComfyVaeArtifactRole::AudioWeights
                && event.phase == H3ComfyVaeLoadPhase::Construct
                && event.completed == event.total
        }));
        assert!(!std::fs::read_dir(&staging_root)
            .expect("read private staging root")
            .flatten()
            .any(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with(".mold-h3-vae-stage-")));

        println!(
            "mold.h3.private-vae-smoke.v1 plan={} lease={} elapsed_ms={} visual_file={} visual_tensor={} audio_file={} audio_tensor={} resident_device={} effective_device={} public_runtime=false",
            bundle.plan_identity_sha256(),
            bundle.artifact_lease.authority_identity_sha256(),
            started.elapsed().as_millis(),
            bundle.memory().visual.artifact_file_bytes,
            bundle.memory().visual.encoded_tensor_bytes,
            bundle.memory().audio.artifact_file_bytes,
            bundle.memory().audio.encoded_tensor_bytes,
            bundle.memory().resident_device_weight_bytes,
            bundle.memory().effective_device_weight_bytes,
        );
        drop(bundle);

        assert!(!contract::capabilities(Task::Fl2va).runtime_available);
        assert!(contract::runnable_capability_contract_for_model(contract::FL2VA_COMFY).is_none());
    }
}
