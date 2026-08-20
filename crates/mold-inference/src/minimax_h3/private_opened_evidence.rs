//! Opaque opened-evidence preparation for private MiniMax H3 FL2VA UAT.
//!
//! This non-shipping module resolves component locations only from the hidden
//! manifest/storage contract, performs normal CPU endpoint preprocessing and
//! seed resolution, and derives the existing factory value records from
//! authenticated opened objects. It deliberately does not issue a scheduler
//! lease, clear an activation prerequisite, or expose a server/family bridge.

use std::fs::File;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::Device;
use mold_candle::minimax_h3::{
    H3AuthenticatedQwenNvfp4Authority, H3ComfyOpenedInt8Checkpoint, H3ComfyPublishedArtifact,
    H3QwenNvfp4RuntimePlacement, H3TransformerTask,
};
use mold_core::manifest::{find_manifest, storage_path, ModelComponent, ModelManifest};
use mold_core::minimax_h3::{self as contract, Layout, Mode, Task};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use mold_core::GenerateRequest;
#[cfg(feature = "mp4")]
use mold_core::GenerationReferenceKind;
use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use super::backend::prepare_fl2va_conditioner_input;
use super::pipeline::{
    collect_endpoint_bytes, prepare_request, H3EndpointAnchor, H3PipelineObserver,
    H3PreparedFl2VaRequest,
};
#[cfg(feature = "mp4")]
use super::private_qwen::prepare_ref2va_conditioner_input_for_admission;
use super::private_qwen_support::H3PrivateQwenSupport;
#[cfg(feature = "mp4")]
use super::private_server::H3PrivatePreparationCheckpoint;
use super::sampler::{H3DualSchedule, H3SamplerKind};
use super::vae_runtime::{
    FrozenH3ComfyVaeLoadPlan, H3AuthenticatedComfyVaeAuthority, H3ComfyVaeArtifactRole,
};
use crate::h3_factory::{
    expected_h3_factory_prepared_attempt_identity, expected_h3_factory_prepared_request_identity,
    expected_h3_factory_raw_checkpoint_identity, expected_h3_factory_reference_media_identity,
    expected_h3_factory_target_budget_identity, H3FactoryArtifactHostInput,
    H3FactoryArtifactHostRole, H3FactoryBlockMemoryInput, H3FactoryEndpointAnchor,
    H3FactoryEndpointInput, H3FactoryEndpointPreprocess, H3FactoryExecutionBudgetEchoInput,
    H3FactoryPreparedAttemptInput, H3FactoryPreparedRequestInput, H3FactoryPreparedRowsInput,
    H3FactoryRawCheckpointInput, H3FactoryTargetBudgetInput, H3FactoryTargetDenoiseCopyPolicy,
    H3FactoryTargetLoadDropPolicy, H3FactoryTurboAdapterAuthority,
};
#[cfg(feature = "mp4")]
use crate::h3_factory::{
    expected_h3_factory_reference_charges, H3FactoryReferenceInput, H3FactoryReferenceKind,
};
use crate::progress::ProgressReporter;

const FL2VA_TRANSFORMER_SOURCE: &str =
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
const QWEN_WEIGHT_SOURCE: &str = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
const FL2VA_TASK_CONFIG_SOURCE: &str = "transformer/config.json";

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateStorageRootIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    user: u32,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

impl H3PrivateStorageRootIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            user: metadata.uid(),
            #[cfg(unix)]
            mode: metadata.mode(),
            #[cfg(unix)]
            modified_seconds: metadata.mtime(),
            #[cfg(unix)]
            modified_nanoseconds: metadata.mtime_nsec(),
            #[cfg(unix)]
            changed_seconds: metadata.ctime(),
            #[cfg(unix)]
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    fn update_digest(&self, digest: &mut Sha256) {
        #[cfg(unix)]
        for value in [
            self.device,
            self.inode,
            u64::from(self.user),
            u64::from(self.mode),
            self.modified_seconds as u64,
            self.modified_nanoseconds as u64,
            self.changed_seconds as u64,
            self.changed_nanoseconds as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct H3PrivateOpenedFileIdentity {
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
}

impl H3PrivateOpenedFileIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
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
        }
    }
}

struct H3PrivateOpenedTaskConfigAuthority {
    path: PathBuf,
    file: File,
    identity: H3PrivateOpenedFileIdentity,
    content_sha256: String,
}

impl H3PrivateOpenedTaskConfigAuthority {
    fn open(path: &Path, expected_bytes: u64, expected_sha256: &str) -> Result<Self> {
        require_sha256(expected_sha256, "private H3 task config")?;
        let requested =
            std::fs::symlink_metadata(path).context("failed to inspect private H3 task config")?;
        if requested.file_type().is_symlink() || !requested.file_type().is_file() {
            bail!("private H3 task config must be a regular non-symlink file")
        }
        let path = std::fs::canonicalize(path)?;
        let file = open_regular_file_no_follow(&path)?;
        fs2::FileExt::lock_shared(&file)?;
        let identity = H3PrivateOpenedFileIdentity::from_metadata(&file.metadata()?);
        if identity != H3PrivateOpenedFileIdentity::from_metadata(&requested)
            || identity.len != expected_bytes
        {
            bail!("private H3 task config changed while its descriptor was opened")
        }
        let content_sha256 = sha256_open_file(&file)?;
        if content_sha256 != expected_sha256 {
            bail!("private H3 task config content differs from the hidden manifest")
        }
        let authority = Self {
            path,
            file,
            identity,
            content_sha256,
        };
        authority.revalidate()?;
        Ok(authority)
    }

    fn revalidate(&self) -> Result<()> {
        let descriptor = H3PrivateOpenedFileIdentity::from_metadata(&self.file.metadata()?);
        let current = open_regular_file_no_follow(&self.path)?;
        let path_identity = H3PrivateOpenedFileIdentity::from_metadata(&current.metadata()?);
        let content_sha256 = sha256_open_file(&self.file)?;
        if descriptor != self.identity
            || path_identity != self.identity
            || content_sha256 != self.content_sha256
        {
            bail!("private H3 task config changed after authentication")
        }
        Ok(())
    }

    fn file_bytes(&self) -> u64 {
        self.identity.len
    }
}

/// Exact hidden-manifest paths for the private Comfy FL2VA partition.
///
/// This value is intentionally non-Clone. It binds the canonical models-root
/// identity and never accepts `ModelPaths` or config-file overrides.
pub(crate) struct H3PrivateComfyStorageAuthority {
    models_root: PathBuf,
    root_identity: H3PrivateStorageRootIdentity,
    transformer: PathBuf,
    qwen_weights: PathBuf,
    task_config: PathBuf,
    identity_sha256: String,
}

/// Payload-free snapshot produced from the exact opened component objects
/// that remain owned by one prepared attempt. It is not executable authority;
/// the non-Clone activation token consumes this snapshot only while the
/// opened objects themselves are still available for the runtime owner.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateOpenedActivationFacts {
    pub(crate) identity_sha256: String,
    pub(crate) storage_identity_sha256: String,
    pub(crate) support_identity_sha256: String,
    pub(crate) transformer_checkpoint_identity_sha256: String,
    pub(crate) transformer_memory_identity_sha256: String,
    pub(crate) qwen_authority_identity_sha256: String,
    /// Exact descriptor/staging identity for this activation only. This value
    /// must never escape into canonical scheduler or execution identities.
    pub(crate) vae_attempt_open_identity_sha256: String,
    pub(crate) vae_artifact_validation_identity_sha256: String,
    pub(crate) vae_artifact_plan_identity_sha256: String,
}

impl H3PrivateComfyStorageAuthority {
    pub(crate) fn resolve(models_root: &Path) -> Result<Self> {
        let (models_root, root_identity) = validate_storage_root(models_root)?;
        let manifest = private_fl2va_manifest()?;
        let transformer = resolve_component(
            manifest,
            &models_root,
            FL2VA_TRANSFORMER_SOURCE,
            ModelComponent::Transformer,
        )?;
        let qwen_weights = resolve_component(
            manifest,
            &models_root,
            QWEN_WEIGHT_SOURCE,
            ModelComponent::TextEncoder,
        )?;
        let task_config = resolve_component(
            manifest,
            &models_root,
            FL2VA_TASK_CONFIG_SOURCE,
            ModelComponent::TaskConfig,
        )?;
        let mut authority = Self {
            models_root,
            root_identity,
            transformer,
            qwen_weights,
            task_config,
            identity_sha256: String::new(),
        };
        authority.identity_sha256 = storage_authority_identity(&authority);
        authority.validate()?;
        Ok(authority)
    }

    pub(crate) fn transformer_path(&self) -> &Path {
        &self.transformer
    }

    /// Canonical, validated models root this authority resolved under. The
    /// transformer-load phase uses it to resolve a manifest-selected Turbo
    /// adapter without re-deriving storage from config.
    pub(crate) fn models_root(&self) -> &Path {
        &self.models_root
    }

    pub(crate) fn qwen_weights_path(&self) -> &Path {
        &self.qwen_weights
    }

    pub(crate) fn task_config_path(&self) -> &Path {
        &self.task_config
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub(crate) fn vae_plan(&self, staging_root: &Path) -> Result<FrozenH3ComfyVaeLoadPlan> {
        self.validate()?;
        FrozenH3ComfyVaeLoadPlan::from_hidden_storage(
            contract::FL2VA_COMFY,
            &self.models_root,
            staging_root,
        )
        .map_err(Into::into)
    }

    fn open_task_config(&self) -> Result<H3PrivateOpenedTaskConfigAuthority> {
        self.validate()?;
        let manifest = private_fl2va_manifest()?;
        let matches = manifest
            .files
            .iter()
            .filter(|file| {
                file.hf_filename == FL2VA_TASK_CONFIG_SOURCE
                    && file.component == ModelComponent::TaskConfig
            })
            .collect::<Vec<_>>();
        let [file] = matches.as_slice() else {
            bail!("hidden MiniMax H3 manifest requires exactly one FL2VA task config")
        };
        if self.task_config != self.models_root.join(storage_path(manifest, file)) {
            bail!("private H3 task config path differs from hidden storage authority")
        }
        let expected_sha256 = file
            .sha256
            .ok_or_else(|| anyhow!("hidden MiniMax H3 task config has no digest"))?;
        let authority = H3PrivateOpenedTaskConfigAuthority::open(
            &self.task_config,
            file.size_bytes,
            expected_sha256,
        )?;
        if authority.path != self.task_config {
            bail!("private H3 task config resolves outside hidden storage authority")
        }
        Ok(authority)
    }

    fn vae_source_path(&self, role: H3ComfyVaeArtifactRole) -> Result<PathBuf> {
        resolve_component(
            private_fl2va_manifest()?,
            &self.models_root,
            role.source_path(),
            role.manifest_component(),
        )
    }

    pub(crate) fn validate_opened_components(
        &self,
        support: &H3PrivateQwenSupport,
        transformer: &H3ComfyOpenedInt8Checkpoint,
        qwen: &H3AuthenticatedQwenNvfp4Authority,
        vae: &H3AuthenticatedComfyVaeAuthority,
    ) -> Result<()> {
        self.validate()?;
        support.revalidate()?;
        support.validate_storage_root(&self.models_root)?;
        transformer.revalidate()?;
        qwen.revalidate()?;
        vae.validate()?;
        validate_fl2va_transformer_contract(transformer.candidate().artifact)?;
        validate_fl2va_vae_contract(vae.task(), vae.canonical_model())?;
        if transformer.source_path() != self.transformer || qwen.source_path() != self.qwen_weights
        {
            bail!(
                "private H3 opened component does not belong to the hidden FL2VA storage authority"
            )
        }
        for role in H3ComfyVaeArtifactRole::ALL {
            if vae.source_path(role)? != self.vae_source_path(role)? {
                bail!("private H3 opened VAE path differs from hidden FL2VA storage authority")
            }
        }
        Ok(())
    }

    pub(crate) fn opened_activation_facts(
        &self,
        support: &H3PrivateQwenSupport,
        transformer: &H3ComfyOpenedInt8Checkpoint,
        qwen: &H3AuthenticatedQwenNvfp4Authority,
        vae: &H3AuthenticatedComfyVaeAuthority,
    ) -> Result<H3PrivateOpenedActivationFacts> {
        self.validate_opened_components(support, transformer, qwen, vae)?;
        let mut facts = H3PrivateOpenedActivationFacts {
            identity_sha256: String::new(),
            storage_identity_sha256: self.identity_sha256.clone(),
            support_identity_sha256: support.support_identity_sha256().into(),
            transformer_checkpoint_identity_sha256: transformer.checkpoint_identity_sha256().into(),
            transformer_memory_identity_sha256: transformer
                .memory_evidence()
                .identity_sha256
                .clone(),
            qwen_authority_identity_sha256: qwen.identity_sha256().into(),
            vae_attempt_open_identity_sha256: vae.attempt_open_identity_sha256().into(),
            vae_artifact_validation_identity_sha256: vae
                .artifact_validation_identity_sha256()
                .into(),
            vae_artifact_plan_identity_sha256: vae.artifact_plan_identity_sha256().into(),
        };
        let mut digest = Sha256::new();
        digest.update(b"mold.minimax-h3.private-opened-activation-facts.v1\0");
        for value in [
            facts.storage_identity_sha256.as_str(),
            facts.support_identity_sha256.as_str(),
            facts.transformer_checkpoint_identity_sha256.as_str(),
            facts.transformer_memory_identity_sha256.as_str(),
            facts.qwen_authority_identity_sha256.as_str(),
            facts.vae_attempt_open_identity_sha256.as_str(),
            facts.vae_artifact_validation_identity_sha256.as_str(),
            facts.vae_artifact_plan_identity_sha256.as_str(),
        ] {
            require_sha256(value, "private H3 opened activation fact")?;
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        facts.identity_sha256 = format!("{:x}", digest.finalize());
        Ok(facts)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let (root, root_identity) = validate_storage_root(&self.models_root)?;
        if root != self.models_root
            || root_identity != self.root_identity
            || self.identity_sha256 != storage_authority_identity(self)
            || self.transformer
                != resolve_component(
                    private_fl2va_manifest()?,
                    &root,
                    FL2VA_TRANSFORMER_SOURCE,
                    ModelComponent::Transformer,
                )?
            || self.qwen_weights
                != resolve_component(
                    private_fl2va_manifest()?,
                    &root,
                    QWEN_WEIGHT_SOURCE,
                    ModelComponent::TextEncoder,
                )?
            || self.task_config
                != resolve_component(
                    private_fl2va_manifest()?,
                    &root,
                    FL2VA_TASK_CONFIG_SOURCE,
                    ModelComponent::TaskConfig,
                )?
        {
            bail!("private H3 hidden storage authority changed after resolution")
        }
        Ok(())
    }
}

fn private_fl2va_manifest() -> Result<&'static ModelManifest> {
    let manifest = find_manifest(contract::FL2VA_COMFY)
        .ok_or_else(|| anyhow!("missing MiniMax H3 FL2VA Comfy manifest"))?;
    let manifest_contract = contract::manifest_contract(manifest)
        .ok_or_else(|| anyhow!("MiniMax H3 manifest lost its contract"))?;
    if manifest.name != contract::FL2VA_COMFY
        || manifest.family != contract::FAMILY
        || manifest_contract.task != Task::Fl2va
        || manifest_contract.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq
        || manifest_contract.runtime_available
    {
        bail!("private H3 storage requires the exact inactive FL2VA Comfy manifest")
    }
    Ok(manifest)
}

fn resolve_component(
    manifest: &ModelManifest,
    models_root: &Path,
    source_path: &str,
    component: ModelComponent,
) -> Result<PathBuf> {
    let matches = manifest
        .files
        .iter()
        .filter(|file| file.hf_filename == source_path && file.component == component)
        .collect::<Vec<_>>();
    let [file] = matches.as_slice() else {
        bail!("hidden MiniMax H3 manifest requires exactly one {source_path}")
    };
    let relative = storage_path(manifest, file);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("hidden MiniMax H3 manifest produced an unsafe component path")
    }
    Ok(models_root.join(relative))
}

fn validate_storage_root(path: &Path) -> Result<(PathBuf, H3PrivateStorageRootIdentity)> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        bail!("private H3 models root must be an absolute canonical path")
    }
    let metadata =
        std::fs::symlink_metadata(path).context("failed to inspect private H3 models root")?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("private H3 models root must be a real non-symlink directory")
    }
    #[cfg(not(unix))]
    bail!("private H3 opened storage currently requires Unix ownership semantics");
    let canonical = std::fs::canonicalize(path)?;
    if canonical != path {
        bail!("private H3 models root must not contain aliases or symlink components")
    }
    Ok((
        canonical,
        H3PrivateStorageRootIdentity::from_metadata(&metadata),
    ))
}

fn storage_authority_identity(authority: &H3PrivateComfyStorageAuthority) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-hidden-storage.v1\0");
    authority.root_identity.update_digest(&mut digest);
    for path in [
        &authority.models_root,
        &authority.transformer,
        &authority.qwen_weights,
        &authority.task_config,
    ] {
        let bytes = path.as_os_str().as_encoded_bytes();
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

/// Independently qualified non-artifact runtime bounds. Artifact, geometry,
/// row, residency, and phase totals are never accepted through this type;
/// they are recomputed by the binder below.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateQualifiedRuntimeBounds {
    pub(crate) fixed_runtime_host_bytes: u64,
    pub(crate) fixed_runtime_device_bytes: u64,
    pub(crate) qwen_activation_workspace_bytes: u64,
    pub(crate) vae_construction_device_workspace_bytes: u64,
    pub(crate) condition_vae_workspace_device_bytes: u64,
    pub(crate) attention_workspace_device_bytes: u64,
    pub(crate) ffn_workspace_device_bytes: u64,
    pub(crate) decoder_tile_workspace_device_bytes: u64,
    pub(crate) audio_decode_workspace_device_bytes: u64,
    pub(crate) encoded_video_host_bytes_bound: u64,
    pub(crate) thumbnail_host_bytes_bound: u64,
    pub(crate) mux_output_host_bytes_bound: u64,
    pub(crate) aac_mux_staging_host_bytes: u64,
}

impl H3PrivateQualifiedRuntimeBounds {
    fn validate(&self) -> Result<()> {
        if [
            self.fixed_runtime_host_bytes,
            self.fixed_runtime_device_bytes,
            self.qwen_activation_workspace_bytes,
            self.vae_construction_device_workspace_bytes,
            self.condition_vae_workspace_device_bytes,
            self.attention_workspace_device_bytes,
            self.ffn_workspace_device_bytes,
            self.decoder_tile_workspace_device_bytes,
            self.audio_decode_workspace_device_bytes,
            self.encoded_video_host_bytes_bound,
            self.thumbnail_host_bytes_bound,
            self.mux_output_host_bytes_bound,
            self.aac_mux_staging_host_bytes,
        ]
        .contains(&0)
        {
            bail!("private H3 qualified runtime bounds must all be nonzero")
        }
        Ok(())
    }
}

/// Opaque, one-shot prepared attempt. The concrete normalized CPU tensors and
/// resolved seed stay owned here until the caller consumes the record.
pub(crate) struct H3PrivatePreparedFl2VaAttempt {
    prepared: H3PreparedFl2VaRequest,
    factory_attempt: H3FactoryPreparedAttemptInput,
    budget_echo: H3FactoryExecutionBudgetEchoInput,
    // Retain the authenticated descriptor so the artifact priced into the
    // prepared budget cannot be replaced before these inputs are consumed.
    _transformer_support: H3PrivateOpenedTaskConfigAuthority,
}

/// Payload-free result of exact CPU request preprocessing for scheduler
/// admission. The concrete normalized tensors remain local to this function
/// and are dropped before the value crosses the inference boundary.
pub(crate) struct H3PrivateFl2VaAdmissionPreparedRequest {
    pub(crate) request: H3FactoryPreparedRequestInput,
    pub(crate) seed: u64,
}

pub(crate) struct H3PrivatePreparedFl2VaFactoryInputs {
    pub(crate) prepared: H3PreparedFl2VaRequest,
    pub(crate) factory_attempt: H3FactoryPreparedAttemptInput,
    pub(crate) budget_echo: H3FactoryExecutionBudgetEchoInput,
    _transformer_support: H3PrivateOpenedTaskConfigAuthority,
}

/// Artifact-backed half of a consumed preparation. The concrete tensor request
/// is split out exactly once for `pipeline::execute_staged`; this retention
/// remains alive through mux and continues to pin the task-config descriptor
/// plus immutable attempt/budget identities.
pub(crate) struct H3PrivatePreparedFl2VaRetention {
    factory_attempt: H3FactoryPreparedAttemptInput,
    budget_echo: H3FactoryExecutionBudgetEchoInput,
    transformer_support: H3PrivateOpenedTaskConfigAuthority,
}

impl H3PrivatePreparedFl2VaAttempt {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        request: &GenerateRequest,
        execution_fingerprint: &str,
        storage: &H3PrivateComfyStorageAuthority,
        support: &H3PrivateQwenSupport,
        transformer: &H3ComfyOpenedInt8Checkpoint,
        qwen: &H3AuthenticatedQwenNvfp4Authority,
        vae: &H3AuthenticatedComfyVaeAuthority,
        bounds: &H3PrivateQualifiedRuntimeBounds,
        turbo: Option<&H3FactoryTurboAdapterAuthority>,
        progress: &ProgressReporter,
        observer: &mut dyn H3PipelineObserver,
    ) -> Result<Self> {
        require_sha256(execution_fingerprint, "private H3 execution fingerprint")?;
        bounds.validate()?;
        storage.validate_opened_components(support, transformer, qwen, vae)?;
        let transformer_support = storage.open_task_config()?;
        transformer_support.revalidate()?;
        let (prepared, factory_request) =
            prepare_private_fl2va_request_input(request, support, progress, observer)?;
        let raw_checkpoint = raw_checkpoint_input(transformer)?;
        let target_budget = build_canonical_private_fl2va_target_budget(
            &factory_request,
            &raw_checkpoint,
            support,
            qwen,
            vae,
            &transformer_support,
            bounds,
            turbo,
        )?;
        let mut factory_attempt = H3FactoryPreparedAttemptInput {
            identity_sha256: String::new(),
            execution_fingerprint: execution_fingerprint.to_owned(),
            request: factory_request,
            raw_checkpoint,
            target_budget,
        };
        factory_attempt.identity_sha256 =
            expected_h3_factory_prepared_attempt_identity(&factory_attempt);
        let budget_echo = H3FactoryExecutionBudgetEchoInput {
            prepared_attempt_identity_sha256: factory_attempt.identity_sha256.clone(),
            device_peak_bytes: factory_attempt.target_budget.predicted_device_peak_bytes,
            host_increment_bytes: factory_attempt.target_budget.predicted_host_increment_bytes,
        };
        Ok(Self {
            prepared,
            factory_attempt,
            budget_echo,
            _transformer_support: transformer_support,
        })
    }

    pub(crate) fn seed(&self) -> u64 {
        self.prepared.seed
    }

    pub(crate) fn identity_sha256(&self) -> &str {
        &self.factory_attempt.identity_sha256
    }

    pub(crate) fn into_factory_inputs(self) -> H3PrivatePreparedFl2VaFactoryInputs {
        H3PrivatePreparedFl2VaFactoryInputs {
            prepared: self.prepared,
            factory_attempt: self.factory_attempt,
            budget_echo: self.budget_echo,
            _transformer_support: self._transformer_support,
        }
    }
}

/// Derive the factory's ordered reference descriptors from the prepared shapes
/// and the facts the decoder reported.
///
/// Normalized geometry comes from the prepared shape (metadata-derived, the
/// same values the runtime will normalize to); native geometry comes from what
/// the decoder actually found, which is the authority for the retained-media
/// charge. The two retained-byte totals are left to the factory to re-derive —
/// this only reports the geometry they are computed from.
#[cfg(feature = "mp4")]
fn ref2va_factory_references(
    prepared: &[super::pipeline::ref2va::H3PreparedReference],
    decoded: &[super::pipeline::ref2va::H3DecodedReferenceFacts],
) -> Result<Vec<H3FactoryReferenceInput>> {
    prepared
        .iter()
        .zip(decoded)
        .map(|(reference, facts)| {
            let shape = &reference.shape;
            let kind = match reference.metadata.kind {
                GenerationReferenceKind::Image => H3FactoryReferenceKind::Image,
                GenerationReferenceKind::Video => H3FactoryReferenceKind::Video,
                GenerationReferenceKind::Audio => H3FactoryReferenceKind::Audio,
            };
            let mut input = H3FactoryReferenceInput {
                index: reference.metadata.index,
                kind,
                content_sha256: reference.metadata.sha256.clone(),
                preprocess_version: shape.version,
                normalized_width: shape.normalized_width,
                normalized_height: shape.normalized_height,
                normalized_video_frames: shape.normalized_video_frames,
                video_frames: shape.video_frames,
                qwen_video_frames: shape.qwen_video_frames,
                audio_samples_per_channel: shape.audio_samples_per_channel,
                native_width: facts.width,
                native_height: facts.height,
                native_audio_samples_per_channel: facts
                    .audio
                    .as_ref()
                    .map(|audio| audio.samples_per_channel),
                native_audio_channels: facts.audio.as_ref().map(|audio| audio.channels),
                visual_rows: 0,
                audio_rows: 0,
                qwen_vision_rows: 0,
                normalized_host_bytes: 0,
                native_host_bytes: 0,
            };
            // The factory owns this arithmetic; ask it rather than restate it,
            // so the builder and the validator agree by construction.
            let charges = expected_h3_factory_reference_charges(&input)?;
            input.visual_rows = charges.visual_rows;
            input.audio_rows = charges.audio_rows;
            input.qwen_vision_rows = charges.qwen_vision_rows;
            input.normalized_host_bytes = charges.normalized_host_bytes;
            input.native_host_bytes = charges.native_host_bytes;
            // The metadata-derived shape must agree with what the factory
            // derives; a disagreement means the two contracts have drifted.
            if shape.visual_rows != charges.visual_rows || shape.audio_rows != charges.audio_rows {
                bail!(
                    "private H3 reference {} prepared shape disagrees with the factory's rows",
                    reference.metadata.index
                )
            }
            Ok(input)
        })
        .collect()
}

/// Assemble the Ref2VA prepared-request input the frozen plan is built from.
#[cfg(feature = "mp4")]
fn ref2va_prepared_request_input(
    request: &GenerateRequest,
    prepared: &super::pipeline::ref2va::H3PreparedRef2VaRequest,
    references: Vec<H3FactoryReferenceInput>,
    qwen_output_text_rows: u64,
) -> Result<H3FactoryPreparedRequestInput> {
    let geometry = prepared.geometry();
    let condition_visual_rows = checked_sum(references.iter().map(|entry| entry.visual_rows))?;
    let condition_audio_rows = checked_sum(references.iter().map(|entry| entry.audio_rows))?;
    let qwen_vision_rows = checked_sum(references.iter().map(|entry| entry.qwen_vision_rows))?;
    let target_video_rows = u64::try_from(geometry.generated_video_rows)?;
    let target_audio_rows = u64::try_from(geometry.generated_audio_rows)?;
    let total_packed_rows = checked_sum([
        qwen_output_text_rows,
        condition_visual_rows,
        condition_audio_rows,
        target_video_rows,
        target_audio_rows,
    ])?;
    let schedule =
        H3DualSchedule::new_for_sampler(prepared.grid_points(), H3SamplerKind::ComfyResMultistep)?;
    let mut input = H3FactoryPreparedRequestInput {
        identity_sha256: String::new(),
        canonical_model: contract::REF2VA_COMFY.into(),
        task: Task::Ref2va,
        mode: Mode::ReferenceToAudioVideo,
        prompt_sha256: sha256(prepared.prompt().as_bytes()),
        seed: prepared.seed(),
        grid_points: u32::try_from(prepared.grid_points())?,
        denoise_forward_count: u32::try_from(schedule.counts().transformer_evaluations)?,
        guidance_f64_bits: request.guidance.to_bits(),
        strength_f64_bits: request.strength.to_bits(),
        batch_size: request.batch_size,
        width: u32::try_from(geometry.width)?,
        height: u32::try_from(geometry.height)?,
        frames: u32::try_from(geometry.frames)?,
        fps: contract::FIXED_FPS,
        synchronized_audio: true,
        mp4_output: true,
        video_latent_frames: u64::try_from(geometry.latent_frames)?,
        audio_latents_per_channel: u64::try_from(geometry.audio_latents_per_channel)?,
        audio_samples_per_channel: u64::try_from(geometry.audio_latents_per_channel)?
            .checked_mul(800)
            .ok_or_else(|| anyhow!("private H3 Ref2VA audio sample count overflow"))?,
        // Ref2VA conditions on references, so the endpoint fingerprint is the
        // fixed no-endpoint domain and the reference fingerprint is the live
        // one the staged set produced.
        conditioning_fingerprint: sha256(b"mold.minimax-h3.ref2va-no-endpoints.v1"),
        reference_fingerprint: prepared.reference_fingerprint().into(),
        endpoints: Vec::new(),
        references,
        rows: H3FactoryPreparedRowsInput {
            qwen_output_text_rows,
            qwen_vision_rows,
            condition_visual_rows,
            condition_audio_rows,
            target_video_rows,
            target_audio_rows,
            total_packed_rows,
        },
    };
    input.identity_sha256 = expected_h3_factory_prepared_request_identity(&input);
    Ok(input)
}

/// Build the Ref2VA admission prepared request.
///
/// Admission decodes and normalizes the ordered references through the SAME
/// [`H3ReferenceMediaAdapter`] the runtime uses — there is deliberately no
/// second decoder — because the exact Qwen row counts come from packing real
/// vision pads, and the target budget is sized from the retained media those
/// same steps produce. It runs entirely on the CPU and opens no CUDA device,
/// matching the FL2VA path's endpoint normalization.
///
/// The decoded media is dropped when this returns: only geometry, digests, and
/// row counts survive into the frozen plan, so no reference byte reaches the
/// request, the queue journal, or the gallery.
#[cfg(feature = "mp4")]
pub(crate) fn prepare_private_ref2va_admission_request(
    request: &GenerateRequest,
    references: &[crate::engine::GenerationReferenceBinding],
    support: &H3PrivateQwenSupport,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PrivateFl2VaAdmissionPreparedRequest> {
    if support.model() != contract::REF2VA_COMFY || support.task() != Task::Ref2va {
        bail!("private H3 Ref2VA admission has cross-task Qwen support")
    }
    let prepared = super::pipeline::ref2va::prepare_resolved_request(request, progress, observer)?;
    let prepared_references = prepared.references();
    if prepared_references.len() != references.len() {
        bail!("private H3 Ref2VA admission reference count differs from its staged bindings")
    }

    let mut media = super::reference_media::H3ReferenceMediaAdapter::default();
    let mut checkpoint = H3PrivatePreparationCheckpoint { progress };
    let mut decoded = Vec::with_capacity(references.len());
    for (reference, binding) in prepared_references.iter().zip(references) {
        decoded.push(media.decode_reference(reference, binding, &mut checkpoint)?);
    }
    let mut presentations = Vec::with_capacity(references.len());
    for (reference, facts) in prepared_references.iter().zip(&decoded) {
        presentations.push(media.preprocess_reference(reference, facts, &mut checkpoint)?);
    }

    let (conditioner, _) = prepare_ref2va_conditioner_input_for_admission(
        support.tokenizer(),
        prepared.prompt(),
        &presentations,
        &media,
    )?;
    let (_, qwen_output_text_rows) = conditioner.input_ids.dims2()?;

    let factory_references = ref2va_factory_references(prepared_references, &decoded)?;
    let factory_request = ref2va_prepared_request_input(
        request,
        &prepared,
        factory_references,
        u64::try_from(qwen_output_text_rows)?,
    )?;
    Ok(H3PrivateFl2VaAdmissionPreparedRequest {
        seed: prepared.seed(),
        request: factory_request,
    })
}

pub(crate) fn prepare_private_fl2va_admission_request(
    request: &GenerateRequest,
    support: &H3PrivateQwenSupport,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<H3PrivateFl2VaAdmissionPreparedRequest> {
    // Admission intentionally performs endpoint normalization and seeded
    // preparation before the allocation commit. The delegated pipeline path
    // may construct CPU-only Candle tensors, but it cannot open a CUDA device
    // or construct a CUDA tensor.
    let (prepared, request) =
        prepare_private_fl2va_request_input(request, support, progress, observer)?;
    Ok(H3PrivateFl2VaAdmissionPreparedRequest {
        seed: prepared.seed,
        request,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_private_fl2va_admission_attempt(
    execution_fingerprint: &str,
    request: H3FactoryPreparedRequestInput,
    storage: &H3PrivateComfyStorageAuthority,
    support: &H3PrivateQwenSupport,
    transformer: &H3ComfyOpenedInt8Checkpoint,
    qwen: &H3AuthenticatedQwenNvfp4Authority,
    vae: &H3AuthenticatedComfyVaeAuthority,
    bounds: &H3PrivateQualifiedRuntimeBounds,
    turbo: Option<&H3FactoryTurboAdapterAuthority>,
) -> Result<H3FactoryPreparedAttemptInput> {
    require_sha256(execution_fingerprint, "private H3 execution fingerprint")?;
    bounds.validate()?;
    storage.validate_opened_components(support, transformer, qwen, vae)?;
    let transformer_support = storage.open_task_config()?;
    transformer_support.revalidate()?;
    let raw_checkpoint = raw_checkpoint_input(transformer)?;
    let target_budget = build_canonical_private_fl2va_target_budget(
        &request,
        &raw_checkpoint,
        support,
        qwen,
        vae,
        &transformer_support,
        bounds,
        turbo,
    )?;
    let mut attempt = H3FactoryPreparedAttemptInput {
        identity_sha256: String::new(),
        execution_fingerprint: execution_fingerprint.into(),
        request,
        raw_checkpoint,
        target_budget,
    };
    attempt.identity_sha256 = expected_h3_factory_prepared_attempt_identity(&attempt);
    Ok(attempt)
}

fn prepare_private_fl2va_request_input(
    request: &GenerateRequest,
    support: &H3PrivateQwenSupport,
    progress: &ProgressReporter,
    observer: &mut dyn H3PipelineObserver,
) -> Result<(H3PreparedFl2VaRequest, H3FactoryPreparedRequestInput)> {
    let mode = contract::validate_request_contract(request, Task::Fl2va)
        .map_err(|error| anyhow!("{}: {}", error.code, error.message))?;
    let encoded_endpoints = collect_endpoint_bytes(request, mode)?
        .into_iter()
        .map(|(anchor, bytes)| (anchor, bytes.len() as u64, sha256(bytes)))
        .collect::<Vec<_>>();
    let prepared = prepare_request(request, progress, observer)?;
    let factory_request =
        prepared_request_input(request, mode, &prepared, &encoded_endpoints, support)?;
    Ok((prepared, factory_request))
}

impl H3PrivatePreparedFl2VaFactoryInputs {
    /// Revalidate the complete one-shot preparation without reopening any
    /// artifact path. The task-config descriptor remains shared-locked inside
    /// this value until terminal attempt cleanup.
    pub(crate) fn revalidate(&self) -> Result<()> {
        self._transformer_support.revalidate()?;
        require_sha256(
            &self.factory_attempt.identity_sha256,
            "private H3 prepared attempt",
        )?;
        require_sha256(
            &self.factory_attempt.target_budget.identity_sha256,
            "private H3 target budget",
        )?;
        if self.factory_attempt.request.canonical_model != contract::FL2VA_COMFY
            || self.factory_attempt.request.task != Task::Fl2va
            || self.factory_attempt.identity_sha256
                != expected_h3_factory_prepared_attempt_identity(&self.factory_attempt)
            || self.factory_attempt.target_budget.identity_sha256
                != expected_h3_factory_target_budget_identity(&self.factory_attempt.target_budget)
            || self.budget_echo.prepared_attempt_identity_sha256
                != self.factory_attempt.identity_sha256
            || self.budget_echo.device_peak_bytes
                != self
                    .factory_attempt
                    .target_budget
                    .predicted_device_peak_bytes
            || self.budget_echo.host_increment_bytes
                != self
                    .factory_attempt
                    .target_budget
                    .predicted_host_increment_bytes
        {
            bail!("private H3 prepared runtime input changed after opened-evidence binding")
        }
        validate_prepared_runtime_request(&self.prepared, &self.factory_attempt.request)?;
        Ok(())
    }

    pub(crate) fn prepared_attempt_identity_sha256(&self) -> &str {
        &self.factory_attempt.identity_sha256
    }

    pub(crate) fn target_budget_identity_sha256(&self) -> &str {
        &self.factory_attempt.target_budget.identity_sha256
    }

    pub(crate) fn execution_fingerprint(&self) -> &str {
        &self.factory_attempt.execution_fingerprint
    }

    pub(crate) fn predicted_device_peak_bytes(&self) -> u64 {
        self.factory_attempt
            .target_budget
            .predicted_device_peak_bytes
    }

    pub(crate) fn predicted_host_increment_bytes(&self) -> u64 {
        self.factory_attempt
            .target_budget
            .predicted_host_increment_bytes
    }

    pub(crate) fn factory_attempt_input(&self) -> &H3FactoryPreparedAttemptInput {
        &self.factory_attempt
    }

    pub(crate) fn prepared_request_input(&self) -> &H3FactoryPreparedRequestInput {
        &self.factory_attempt.request
    }

    pub(crate) fn budget_echo_input(&self) -> &H3FactoryExecutionBudgetEchoInput {
        &self.budget_echo
    }

    pub(crate) fn into_runtime_parts(
        self,
    ) -> (H3PreparedFl2VaRequest, H3PrivatePreparedFl2VaRetention) {
        (
            self.prepared,
            H3PrivatePreparedFl2VaRetention {
                factory_attempt: self.factory_attempt,
                budget_echo: self.budget_echo,
                transformer_support: self._transformer_support,
            },
        )
    }
}

impl H3PrivatePreparedFl2VaRetention {
    pub(crate) fn revalidate(&self) -> Result<()> {
        self.transformer_support.revalidate()?;
        if self.factory_attempt.identity_sha256
            != expected_h3_factory_prepared_attempt_identity(&self.factory_attempt)
            || self.factory_attempt.target_budget.identity_sha256
                != expected_h3_factory_target_budget_identity(&self.factory_attempt.target_budget)
            || self.budget_echo.prepared_attempt_identity_sha256
                != self.factory_attempt.identity_sha256
            || self.budget_echo.device_peak_bytes
                != self
                    .factory_attempt
                    .target_budget
                    .predicted_device_peak_bytes
            || self.budget_echo.host_increment_bytes
                != self
                    .factory_attempt
                    .target_budget
                    .predicted_host_increment_bytes
        {
            bail!("private H3 retained attempt or budget identity changed")
        }
        Ok(())
    }

    pub(crate) fn prepared_attempt_identity_sha256(&self) -> &str {
        &self.factory_attempt.identity_sha256
    }

    pub(crate) fn target_budget_identity_sha256(&self) -> &str {
        &self.factory_attempt.target_budget.identity_sha256
    }

    pub(crate) fn denoise_forward_count(&self) -> Result<usize> {
        usize::try_from(self.factory_attempt.request.denoise_forward_count)
            .map_err(|_| anyhow!("private H3 retained denoise count exceeds usize"))
    }

    /// The admitted packed-sequence length. It is the frozen ceiling the
    /// orchestrator checks its assembled sequence against, so it must come
    /// from the retained factory authority rather than be recomputed.
    pub(crate) fn total_packed_rows(&self) -> Result<usize> {
        usize::try_from(self.factory_attempt.request.rows.total_packed_rows)
            .map_err(|_| anyhow!("private H3 retained packed row count exceeds usize"))
    }
}

fn validate_prepared_runtime_request(
    prepared: &H3PreparedFl2VaRequest,
    frozen: &H3FactoryPreparedRequestInput,
) -> Result<()> {
    let geometry = &prepared.geometry;
    let counts =
        H3DualSchedule::new_for_sampler(prepared.grid_points, H3SamplerKind::ComfyResMultistep)?
            .counts();
    let mut current_endpoints = Vec::with_capacity(prepared.endpoints.len());
    for (prepared_endpoint, frozen_endpoint) in
        prepared.endpoints.iter().zip(frozen.endpoints.iter())
    {
        let pixels = prepared_endpoint.pixels.flatten_all()?.to_vec1::<u8>()?;
        let shape = prepared_endpoint
            .pixels
            .dims5()
            .map(|(b, c, t, h, w)| [b, c, t, h, w])?
            .map(u32::try_from)
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let shape: [u32; 5] = shape
            .try_into()
            .map_err(|_| anyhow!("private H3 prepared endpoint shape changed"))?;
        let current = H3FactoryEndpointInput {
            anchor: factory_anchor(prepared_endpoint.anchor),
            encoded_bytes: frozen_endpoint.encoded_bytes,
            encoded_content_sha256: frozen_endpoint.encoded_content_sha256.clone(),
            preprocess: frozen_endpoint.preprocess,
            normalized_shape: shape,
            normalized_cpu_bytes: pixels.len() as u64,
            normalized_cpu_content_sha256: sha256(&pixels),
        };
        if current.anchor != frozen_endpoint.anchor
            || current.normalized_shape != frozen_endpoint.normalized_shape
            || current.normalized_cpu_bytes != frozen_endpoint.normalized_cpu_bytes
            || current.normalized_cpu_content_sha256
                != frozen_endpoint.normalized_cpu_content_sha256
        {
            bail!("private H3 prepared endpoint changed after target-budget binding")
        }
        current_endpoints.push(current);
    }
    let condition_visual_rows = u64::try_from(geometry.condition_video_rows)?;
    let target_video_rows = u64::try_from(geometry.generated_video_rows)?;
    let target_audio_rows = u64::try_from(geometry.generated_audio_rows)?;
    let total_packed_rows = [
        frozen.rows.qwen_output_text_rows,
        condition_visual_rows,
        target_video_rows,
        target_audio_rows,
    ]
    .into_iter()
    .try_fold(0_u64, |sum, rows| sum.checked_add(rows))
    .ok_or_else(|| anyhow!("private H3 prepared row total overflow"))?;
    if prepared.endpoints.len() != frozen.endpoints.len()
        || sha256(prepared.prompt.as_bytes()) != frozen.prompt_sha256
        || prepared.seed != frozen.seed
        || prepared.grid_points != usize::try_from(frozen.grid_points)?
        || u32::try_from(counts.transformer_evaluations)? != frozen.denoise_forward_count
        || geometry.mode != frozen.mode
        || geometry.width != usize::try_from(frozen.width)?
        || geometry.height != usize::try_from(frozen.height)?
        || geometry.frames != usize::try_from(frozen.frames)?
        || geometry.latent_frames != usize::try_from(frozen.video_latent_frames)?
        || geometry.audio_latents_per_channel != usize::try_from(frozen.audio_latents_per_channel)?
        || frozen.audio_samples_per_channel
            != u64::try_from(geometry.audio_latents_per_channel)?
                .checked_mul(800)
                .ok_or_else(|| anyhow!("private H3 prepared audio sample count overflow"))?
        || frozen.rows.condition_visual_rows != condition_visual_rows
        || frozen.rows.condition_audio_rows != 0
        || frozen.rows.target_video_rows != target_video_rows
        || frozen.rows.target_audio_rows != target_audio_rows
        || frozen.rows.total_packed_rows != total_packed_rows
        || conditioning_fingerprint(&current_endpoints) != frozen.conditioning_fingerprint
    {
        bail!("private H3 prepared prompt, geometry, rows, or conditioning changed")
    }
    Ok(())
}

fn prepared_request_input(
    request: &GenerateRequest,
    mode: Mode,
    prepared: &H3PreparedFl2VaRequest,
    encoded: &[(H3EndpointAnchor, u64, String)],
    support: &H3PrivateQwenSupport,
) -> Result<H3FactoryPreparedRequestInput> {
    if support.model() != contract::FL2VA_COMFY || support.task() != Task::Fl2va {
        bail!("private H3 prepared request has cross-task Qwen support")
    }
    let (conditioner, _) = prepare_fl2va_conditioner_input(
        support.tokenizer(),
        &prepared.prompt,
        &prepared.endpoints,
        &Device::Cpu,
    )?;
    let (_, qwen_output_text_rows) = conditioner.input_ids.dims2()?;
    let qwen_vision_rows = conditioner
        .image
        .as_ref()
        .map(|image| image.pixel_values.dim(0))
        .transpose()?
        .unwrap_or(0);
    let mut endpoints = Vec::with_capacity(prepared.endpoints.len());
    for (index, endpoint) in prepared.endpoints.iter().enumerate() {
        let (encoded_anchor, encoded_bytes, encoded_sha256) = encoded
            .get(index)
            .ok_or_else(|| anyhow!("private H3 encoded endpoint vector is incomplete"))?;
        if *encoded_anchor != endpoint.anchor {
            bail!("private H3 encoded and normalized endpoint order differs")
        }
        let pixels = endpoint.pixels.flatten_all()?.to_vec1::<u8>()?;
        let normalized_shape = endpoint
            .pixels
            .dims5()
            .map(|(b, c, t, h, w)| [b, c, t, h, w])?
            .map(u32::try_from)
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let normalized_shape: [u32; 5] = normalized_shape
            .try_into()
            .map_err(|_| anyhow!("private H3 normalized endpoint shape changed"))?;
        endpoints.push(H3FactoryEndpointInput {
            anchor: factory_anchor(endpoint.anchor),
            encoded_bytes: *encoded_bytes,
            encoded_content_sha256: encoded_sha256.clone(),
            preprocess: H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1,
            normalized_shape,
            normalized_cpu_bytes: pixels.len() as u64,
            normalized_cpu_content_sha256: sha256(&pixels),
        });
    }
    if endpoints.len() != encoded.len() {
        bail!("private H3 endpoint preprocessing lost an encoded endpoint")
    }
    let geometry = &prepared.geometry;
    let qwen_output_text_rows = u64::try_from(qwen_output_text_rows)?;
    let qwen_vision_rows = u64::try_from(qwen_vision_rows)?;
    let condition_visual_rows = u64::try_from(geometry.condition_video_rows)?;
    let target_video_rows = u64::try_from(geometry.generated_video_rows)?;
    let target_audio_rows = u64::try_from(geometry.generated_audio_rows)?;
    let total_packed_rows = [
        qwen_output_text_rows,
        condition_visual_rows,
        target_video_rows,
        target_audio_rows,
    ]
    .into_iter()
    .try_fold(0_u64, |sum, value| sum.checked_add(value))
    .ok_or_else(|| anyhow!("private H3 packed row count overflow"))?;
    let conditioning_fingerprint = conditioning_fingerprint(&endpoints);
    let reference_fingerprint = sha256(b"mold.minimax-h3.fl2va-no-references.v1");
    let schedule =
        H3DualSchedule::new_for_sampler(prepared.grid_points, H3SamplerKind::ComfyResMultistep)?;
    let counts = schedule.counts();
    let mut input = H3FactoryPreparedRequestInput {
        identity_sha256: String::new(),
        canonical_model: contract::FL2VA_COMFY.into(),
        task: Task::Fl2va,
        mode,
        prompt_sha256: sha256(prepared.prompt.as_bytes()),
        seed: prepared.seed,
        grid_points: u32::try_from(prepared.grid_points)?,
        denoise_forward_count: u32::try_from(counts.transformer_evaluations)?,
        guidance_f64_bits: request.guidance.to_bits(),
        strength_f64_bits: request.strength.to_bits(),
        batch_size: request.batch_size,
        width: u32::try_from(geometry.width)?,
        height: u32::try_from(geometry.height)?,
        frames: u32::try_from(geometry.frames)?,
        fps: contract::FIXED_FPS,
        synchronized_audio: true,
        mp4_output: true,
        video_latent_frames: u64::try_from(geometry.latent_frames)?,
        audio_latents_per_channel: u64::try_from(geometry.audio_latents_per_channel)?,
        audio_samples_per_channel: u64::try_from(geometry.audio_latents_per_channel)?
            .checked_mul(800)
            .ok_or_else(|| anyhow!("private H3 audio sample count overflow"))?,
        conditioning_fingerprint,
        reference_fingerprint,
        endpoints,
        references: Vec::new(),
        rows: H3FactoryPreparedRowsInput {
            qwen_output_text_rows,
            qwen_vision_rows,
            condition_visual_rows,
            condition_audio_rows: 0,
            target_video_rows,
            target_audio_rows,
            total_packed_rows,
        },
    };
    input.identity_sha256 = expected_h3_factory_prepared_request_identity(&input);
    Ok(input)
}

fn raw_checkpoint_input(
    transformer: &H3ComfyOpenedInt8Checkpoint,
) -> Result<H3FactoryRawCheckpointInput> {
    transformer.revalidate()?;
    validate_fl2va_transformer_contract(transformer.candidate().artifact)?;
    let evidence = transformer.memory_evidence();
    if transformer.content_sha256()
        != H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot.content_sha256()
        || evidence.verified_file_bytes
            != H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot.file_bytes()
        || evidence.blocks.len() != 50
    {
        bail!("private H3 opened transformer evidence differs from the exact FL2VA artifact")
    }
    let blocks = evidence
        .blocks
        .iter()
        .map(|block| H3FactoryBlockMemoryInput {
            index: block.index,
            encoded_host_bytes: block.memory.encoded_host_bytes,
            protected_device_bytes: block.memory.protected_device_bytes,
            max_device_weight_staging_bytes: block.memory.max_device_weight_staging_bytes,
            max_host_read_staging_bytes: block.memory.max_host_read_staging_bytes,
            content_sha256: block.content_sha256.clone(),
        })
        .collect();
    let mut input = H3FactoryRawCheckpointInput {
        identity_sha256: String::new(),
        raw_content_sha256: transformer.content_sha256().into(),
        verified_file_bytes: evidence.verified_file_bytes,
        raw_header_identity_sha256: transformer.candidate().header_identity_sha256.clone(),
        // The parsed header stays resident for the whole stream lifetime; the
        // tensor payload does not (comfy_dit.rs:1373-1407 reads it through a
        // bounded buffer), so this is the checkpoint's only host residency.
        retained_header_host_bytes: evidence.header_bytes,
        opened_checkpoint_identity_sha256: transformer.checkpoint_identity_sha256().into(),
        quantization_policy_identity_sha256: transformer
            .candidate()
            .strategy
            .quantization_policy
            .policy_sha256
            .clone(),
        config_identity_sha256: transformer.config_identity_sha256(),
        fixed_transformer_encoded_host_bytes: evidence.fixed_encoded_host_bytes,
        fixed_transformer_protected_device_bytes: evidence.fixed_protected_device_bytes,
        fixed_transformer_max_host_read_staging_bytes: evidence.fixed_max_host_read_staging_bytes,
        fixed_transformer_max_device_weight_staging_bytes: evidence
            .fixed_max_device_weight_staging_bytes,
        blocks,
    };
    input.identity_sha256 = expected_h3_factory_raw_checkpoint_identity(&input);
    Ok(input)
}

fn validate_fl2va_transformer_contract(artifact: H3ComfyPublishedArtifact) -> Result<()> {
    if artifact != H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot
        || artifact.task() != H3TransformerTask::T2VaFl2Va
    {
        bail!("private H3 prepared attempt requires the exact FL2VA INT8 ConvRot transformer")
    }
    Ok(())
}

fn validate_fl2va_vae_contract(task: Task, canonical_model: &str) -> Result<()> {
    if task != Task::Fl2va || canonical_model != contract::FL2VA_COMFY {
        bail!("private H3 prepared attempt requires the exact FL2VA VAE authority")
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_canonical_private_fl2va_target_budget(
    request: &H3FactoryPreparedRequestInput,
    checkpoint: &H3FactoryRawCheckpointInput,
    support: &H3PrivateQwenSupport,
    qwen: &H3AuthenticatedQwenNvfp4Authority,
    vae: &H3AuthenticatedComfyVaeAuthority,
    transformer_support: &H3PrivateOpenedTaskConfigAuthority,
    bounds: &H3PrivateQualifiedRuntimeBounds,
    turbo: Option<&H3FactoryTurboAdapterAuthority>,
) -> Result<H3FactoryTargetBudgetInput> {
    transformer_support.revalidate()?;
    let mut artifacts = support
        .artifact_facts()
        .iter()
        .enumerate()
        .map(|(index, fact)| {
            Ok(H3FactoryArtifactHostInput {
                role: H3FactoryArtifactHostRole::Conditioner,
                index: u16::try_from(index)?,
                content_sha256: fact.content_sha256.clone(),
                bytes: fact.file_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    artifacts.push(H3FactoryArtifactHostInput {
        role: H3FactoryArtifactHostRole::Conditioner,
        index: u16::try_from(artifacts.len())?,
        content_sha256: qwen.artifact_identity_sha256().into(),
        bytes: qwen.artifact_file_bytes(),
    });
    artifacts.push(H3FactoryArtifactHostInput {
        role: H3FactoryArtifactHostRole::RawTransformerCheckpoint,
        index: 0,
        content_sha256: checkpoint.raw_content_sha256.clone(),
        bytes: checkpoint.verified_file_bytes,
    });
    artifacts.push(H3FactoryArtifactHostInput {
        role: H3FactoryArtifactHostRole::TransformerSupport,
        index: 0,
        content_sha256: transformer_support.content_sha256.clone(),
        bytes: transformer_support.file_bytes(),
    });
    let mut visual_index = 0_u16;
    let mut audio_index = 0_u16;
    for fact in vae.artifact_facts() {
        let (role, index) = match fact.role {
            H3ComfyVaeArtifactRole::VisualConfig | H3ComfyVaeArtifactRole::VisualWeights => {
                let index = visual_index;
                visual_index += 1;
                (H3FactoryArtifactHostRole::VisualVae, index)
            }
            H3ComfyVaeArtifactRole::AudioConfig | H3ComfyVaeArtifactRole::AudioWeights => {
                let index = audio_index;
                audio_index += 1;
                (H3FactoryArtifactHostRole::AudioVae, index)
            }
        };
        artifacts.push(H3FactoryArtifactHostInput {
            role,
            index,
            content_sha256: fact.content_sha256.clone(),
            bytes: fact.file_bytes,
        });
    }
    artifacts.sort_by_key(|artifact| (artifact.role, artifact.index));
    let artifact_host_bytes = checked_sum(artifacts.iter().map(|artifact| artifact.bytes))?;
    let qwen_memory = qwen.memory_facts();
    let qwen_output_state_device_bytes = request
        .rows
        .qwen_output_text_rows
        .checked_mul(5_120 * 2)
        .ok_or_else(|| anyhow!("private H3 Qwen output bytes overflow"))?;
    let (
        qwen_host_activation_bytes,
        qwen_host_output_state_bytes,
        qwen_device_parameter_bytes,
        qwen_activation_device_bytes,
        qwen_output_transfer_device_bytes,
    ) = match qwen.placement() {
        H3QwenNvfp4RuntimePlacement::Accelerated => (
            0,
            0,
            qwen_memory.device_resident_parameter_bytes,
            bounds.qwen_activation_workspace_bytes,
            0,
        ),
        H3QwenNvfp4RuntimePlacement::Cpu => (
            bounds.qwen_activation_workspace_bytes,
            qwen_output_state_device_bytes,
            0,
            0,
            qwen_output_state_device_bytes,
        ),
    };
    let qwen_host_parameter_bytes = qwen_memory.host_resident_parameter_bytes;
    let qwen_host_workspace_bytes = checked_sum([
        qwen_host_parameter_bytes,
        qwen_host_activation_bytes,
        qwen_host_output_state_bytes,
    ])?;
    // The NVFP4 loader reads one tensor at a time into an anonymous `Vec`
    // (`qwen_nvfp4.rs:820-856`) and `Tensor::from_raw_buffer` copies it again
    // (`qwen_nvfp4_runtime.rs:806-821`), so the largest tensor is live twice
    // while the parameters already read are resident. This transient was
    // measured but never budgeted before.
    let qwen_host_load_staging_bytes = qwen_memory
        .maximum_tensor_staging_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow!("private H3 Qwen host load staging overflow"))?;
    // Metadata each opened authority retains beside the payload it streams:
    // the Qwen's parsed raw header, the transformer's parsed safetensors
    // header, and the VAE authorities' two decoded config buffers.
    let qwen_retained_header_host_bytes = qwen_memory.retained_raw_header_bytes;
    let transformer_retained_header_host_bytes = checkpoint.retained_header_host_bytes;
    let condition_latent_backing_device_bytes = request
        .rows
        .condition_visual_rows
        .checked_mul(96 * 4)
        .ok_or_else(|| anyhow!("private H3 condition latent bytes overflow"))?;
    let target_video_latent_device_bytes = request
        .rows
        .target_video_rows
        .checked_mul(96 * 4)
        .ok_or_else(|| anyhow!("private H3 target video bytes overflow"))?;
    let target_audio_latent_device_bytes = request
        .rows
        .target_audio_rows
        .checked_mul(32 * 4)
        .ok_or_else(|| anyhow!("private H3 target audio bytes overflow"))?;
    let packed_video_state_device_bytes = condition_latent_backing_device_bytes
        .checked_add(target_video_latent_device_bytes)
        .ok_or_else(|| anyhow!("private H3 packed video bytes overflow"))?;
    let packed_audio_state_device_bytes = target_audio_latent_device_bytes;
    let packed_layout_device_bytes = request
        .rows
        .total_packed_rows
        .checked_mul(24)
        .ok_or_else(|| anyhow!("private H3 packed layout bytes overflow"))?;
    let denoise_tensor_copy_workspace_device_bytes = packed_video_state_device_bytes
        .checked_add(packed_audio_state_device_bytes)
        .and_then(|bytes| bytes.checked_mul(9))
        .ok_or_else(|| anyhow!("private H3 denoise copy bytes overflow"))?;
    let waveform_host_bytes = request
        .audio_samples_per_channel
        .checked_mul(u64::from(contract::AUDIO_CHANNELS))
        .and_then(|samples| samples.checked_mul(4))
        .ok_or_else(|| anyhow!("private H3 waveform bytes overflow"))?;
    let vae_memory = vae.memory();
    // Charged on either side of denoise but never across it: the runtime parks
    // both VAEs once conditions are encoded and reconstructs them after the
    // transformer is dropped. The visual-decode phase therefore also carries
    // the construction workspace that reload stages through.
    let retained_vaes = vae_memory.resident_device_weight_bytes;
    let max_device_weight_staging_bytes = checkpoint
        .blocks
        .iter()
        .map(|block| block.max_device_weight_staging_bytes)
        .max()
        .unwrap_or(0);
    let max_host_read_staging_bytes = checkpoint
        .blocks
        .iter()
        .map(|block| block.max_host_read_staging_bytes)
        .max()
        .unwrap_or(0);
    let streamed_block_device_overlap_bytes = checkpoint
        .blocks
        .iter()
        .map(|block| block.protected_device_bytes)
        .max()
        .unwrap_or(0);
    // The one live packed block, plus the tensor being read held twice: the
    // `Vec` from `read_tensor_bytes` and the `from_raw_buffer` CPU copy built
    // from it (`comfy_dit.rs:1373-1407`, `:1451-1462`). Both are alive until
    // the loaded tensor replaces them.
    let max_streamed_block_host_overlap_bytes = checkpoint
        .blocks
        .iter()
        .map(|block| {
            block
                .max_host_read_staging_bytes
                .checked_mul(2)
                .and_then(|staging| staging.checked_add(block.encoded_host_bytes))
                .ok_or_else(|| anyhow!("private H3 streamed host overlap overflow"))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    // One dense non-block tensor at a time reaches host memory during the fixed
    // transformer load and lands on the device before the next is read
    // (`comfy_dit.rs:1410-1447`): the read `Vec`, its `from_raw_buffer` CPU
    // copy, and the optional widened `to_dtype` result. Charging the SUM of
    // every fixed tensor's bytes treated device-resident weights as host
    // residency; `fixed_transformer_protected_device_bytes` already owns those.
    let fixed_transformer_load_host_staging_bytes = checked_sum([
        checkpoint.fixed_transformer_max_host_read_staging_bytes,
        checkpoint.fixed_transformer_max_host_read_staging_bytes,
        checkpoint.fixed_transformer_max_device_weight_staging_bytes,
    ])?;
    let fixed_transformer_load_device_staging_bytes =
        checkpoint.fixed_transformer_max_device_weight_staging_bytes;
    // Both terms come from the authority that declared the adapter, so the
    // builder and `validate_target_budget` cannot disagree about its cost.
    let turbo_adapter_device_bytes =
        turbo.map_or(0, H3FactoryTurboAdapterAuthority::resident_device_bytes);
    let turbo_adapter_device_staging_bytes =
        turbo.map_or(0, H3FactoryTurboAdapterAuthority::device_staging_peak_bytes);
    let turbo_adapter_host_staging_bytes =
        turbo.map_or(0, H3FactoryTurboAdapterAuthority::host_staging_peak_bytes);
    let protected_block_device_bytes = checked_sum(
        checkpoint
            .blocks
            .iter()
            .map(|block| block.protected_device_bytes),
    )?;
    let streamed_block_device_bytes = protected_block_device_bytes;
    let endpoint_encoded_host_bytes = checked_sum(
        request
            .endpoints
            .iter()
            .map(|endpoint| endpoint.encoded_bytes),
    )?;
    let normalized_endpoint_host_bytes = checked_sum(
        request
            .endpoints
            .iter()
            .map(|endpoint| endpoint.normalized_cpu_bytes),
    )?;
    let schedule_host_bytes = u64::from(request.grid_points) * 16;
    let packed_layout_host_bytes = request.rows.total_packed_rows * 24;
    let packed_layout_construction_staging_host_bytes = request.rows.total_packed_rows * 16;
    let packed_layout_freeze_staging_host_bytes = request.rows.total_packed_rows * 12;
    let text_modality_tags_host_bytes = request.rows.qwen_output_text_rows * 8;
    let noise_cpu_staging_host_bytes = condition_latent_backing_device_bytes
        .max(target_video_latent_device_bytes)
        .max(target_audio_latent_device_bytes);
    let condition_backing_host_bytes = condition_latent_backing_device_bytes;
    let condition_vae_workspace_device_bytes = if request.rows.condition_visual_rows == 0 {
        0
    } else {
        bounds.condition_vae_workspace_device_bytes
    };
    let vae_load_phase_device_bytes = checked_sum([
        bounds.fixed_runtime_device_bytes,
        retained_vaes,
        bounds.vae_construction_device_workspace_bytes,
    ])?;
    let qwen_encode_phase_device_bytes = match qwen.placement() {
        H3QwenNvfp4RuntimePlacement::Accelerated => checked_sum([
            bounds.fixed_runtime_device_bytes,
            retained_vaes,
            qwen_device_parameter_bytes,
            qwen_activation_device_bytes,
            qwen_output_state_device_bytes,
        ])?,
        H3QwenNvfp4RuntimePlacement::Cpu => {
            checked_sum([bounds.fixed_runtime_device_bytes, retained_vaes])?
        }
    };
    let qwen_transfer_phase_device_bytes = match qwen.placement() {
        H3QwenNvfp4RuntimePlacement::Accelerated => 0,
        H3QwenNvfp4RuntimePlacement::Cpu => checked_sum([
            bounds.fixed_runtime_device_bytes,
            retained_vaes,
            qwen_output_transfer_device_bytes,
        ])?,
    };
    let condition_encode_phase_device_bytes = checked_sum([
        bounds.fixed_runtime_device_bytes,
        retained_vaes,
        qwen_output_state_device_bytes,
        condition_vae_workspace_device_bytes,
        condition_latent_backing_device_bytes,
        packed_layout_device_bytes,
    ])?;
    let noise_allocation_phase_device_bytes = checked_sum([
        bounds.fixed_runtime_device_bytes,
        qwen_output_state_device_bytes,
        condition_latent_backing_device_bytes,
        condition_latent_backing_device_bytes,
        packed_layout_device_bytes,
        target_video_latent_device_bytes,
        target_video_latent_device_bytes,
        target_audio_latent_device_bytes,
        target_audio_latent_device_bytes,
        packed_video_state_device_bytes,
        packed_audio_state_device_bytes,
    ])?;
    // Shared with `validate_target_budget`; see `H3TransformerLoadDeviceTerms`.
    let transformer_load_phase_device_bytes =
        crate::h3_factory::transformer_load_phase_device_bytes(
            crate::h3_factory::H3TransformerLoadDeviceTerms {
                fixed_runtime_device_bytes: bounds.fixed_runtime_device_bytes,
                fixed_transformer_device_bytes: checkpoint.fixed_transformer_protected_device_bytes,
                qwen_output_state_device_bytes,
                condition_latent_backing_device_bytes,
                packed_layout_device_bytes,
                packed_video_state_device_bytes,
                packed_audio_state_device_bytes,
                // This path streams every block, so nothing is resident.
                resident_block_device_bytes: 0,
                fixed_transformer_load_device_staging_bytes,
                turbo_adapter_device_bytes,
                turbo_adapter_device_staging_bytes,
            },
        )?;
    let denoise_phase_device_bytes =
        crate::h3_factory::denoise_phase_device_bytes(crate::h3_factory::H3DenoiseDeviceTerms {
            fixed_runtime_device_bytes: bounds.fixed_runtime_device_bytes,
            fixed_transformer_device_bytes: checkpoint.fixed_transformer_protected_device_bytes,
            qwen_output_state_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
            packed_video_state_device_bytes,
            packed_audio_state_device_bytes,
            denoise_tensor_copy_workspace_device_bytes,
            denoise_transient_workspace_device_bytes:
                crate::h3_factory::denoise_transient_workspace_device_bytes(
                    bounds.attention_workspace_device_bytes,
                    bounds.ffn_workspace_device_bytes,
                ),
            denoise_hidden_activation_device_bytes:
                crate::h3_factory::denoise_hidden_activation_device_bytes(
                    request.rows.total_packed_rows,
                )?,
            resident_block_device_bytes: 0,
            streamed_block_device_overlap_bytes,
            prefetch_device_bytes: 0,
            max_device_weight_staging_bytes,
            turbo_adapter_device_bytes,
        })?;
    let visual_decode_phase_device_bytes = checked_sum([
        bounds.fixed_runtime_device_bytes,
        retained_vaes,
        bounds.vae_construction_device_workspace_bytes,
        packed_video_state_device_bytes,
        packed_audio_state_device_bytes,
        target_video_latent_device_bytes,
        target_audio_latent_device_bytes,
        bounds.decoder_tile_workspace_device_bytes,
    ])?;
    let audio_decode_phase_device_bytes = checked_sum([
        bounds.fixed_runtime_device_bytes,
        retained_vaes,
        packed_video_state_device_bytes,
        packed_audio_state_device_bytes,
        target_audio_latent_device_bytes,
        bounds.audio_decode_workspace_device_bytes,
        waveform_host_bytes,
    ])?;
    let waveform_transfer_phase_device_bytes = checked_sum([
        bounds.fixed_runtime_device_bytes,
        retained_vaes,
        waveform_host_bytes,
    ])?;
    let predicted_device_peak_bytes = [
        vae_load_phase_device_bytes,
        qwen_encode_phase_device_bytes,
        qwen_transfer_phase_device_bytes,
        condition_encode_phase_device_bytes,
        noise_allocation_phase_device_bytes,
        transformer_load_phase_device_bytes,
        denoise_phase_device_bytes,
        visual_decode_phase_device_bytes,
        audio_decode_phase_device_bytes,
        waveform_transfer_phase_device_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    // Host demand is a per-phase max over ANONYMOUS bytes, mirroring the device
    // peak above. Two classes are charged in no phase at all:
    //
    // * `artifact_host_bytes` is the sum of every artifact's FILE size. The
    //   Qwen (`qwen_nvfp4.rs:820-856`) and the transformer
    //   (`comfy_dit.rs:1373-1407`) stream through bounded `Vec`s with
    //   seek+read_exact, the VAEs mmap (`visual_weights.rs:178`,
    //   `audio_weights.rs:413`), and authentication hashes in 1 MiB chunks.
    //   Nothing holds a whole artifact in RAM, so charging ~42 GB of file bytes
    //   as anonymous demand repeats the #1108 LTX-2 mistake exactly.
    // * `vae_memory.peak_host_mapped_file_bytes` is a real mapping, but it is
    //   file-backed and reclaimable, and `MemAvailable` — the very quantity
    //   this prediction is compared against — already counts those pages as
    //   available. Same reasoning as `ltx2_cpu_gemma_streams_from_mmap`.
    //
    // `vae_memory.peak_staging_disk_bytes` stays excluded as before: it is
    // disk, fenced separately by `ensure_staging_capacity`.
    let attempt_host_bytes = checked_sum([
        bounds.fixed_runtime_host_bytes,
        endpoint_encoded_host_bytes,
        normalized_endpoint_host_bytes,
    ])?;
    // Retained metadata lives exactly as long as its own authority: the Qwen
    // header until the conditioner drops after encode, the transformer header
    // until the transformer drops after denoise, and the VAE configs until the
    // post-denoise reload authority is consumed before visual decode.
    let vae_retained_config_host_bytes = vae_memory.config_bytes;
    let qwen_alive_metadata_host_bytes = checked_sum([
        qwen_retained_header_host_bytes,
        transformer_retained_header_host_bytes,
        vae_retained_config_host_bytes,
    ])?;
    let transformer_alive_metadata_host_bytes = checked_sum([
        transformer_retained_header_host_bytes,
        vae_retained_config_host_bytes,
    ])?;
    let vae_load_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        qwen_alive_metadata_host_bytes,
        vae_memory.peak_host_io_buffer_bytes,
    ])?;
    let qwen_encode_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        qwen_alive_metadata_host_bytes,
        qwen_host_workspace_bytes,
        qwen_host_load_staging_bytes,
        text_modality_tags_host_bytes,
    ])?;
    let qwen_transfer_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        qwen_alive_metadata_host_bytes,
        qwen_host_workspace_bytes,
        text_modality_tags_host_bytes,
    ])?;
    let condition_encode_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        transformer_alive_metadata_host_bytes,
        condition_backing_host_bytes,
        packed_layout_host_bytes,
        packed_layout_construction_staging_host_bytes,
        packed_layout_freeze_staging_host_bytes,
        text_modality_tags_host_bytes,
        noise_cpu_staging_host_bytes,
    ])?;
    let noise_allocation_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        transformer_alive_metadata_host_bytes,
        condition_backing_host_bytes,
        packed_layout_host_bytes,
        text_modality_tags_host_bytes,
        schedule_host_bytes,
        noise_cpu_staging_host_bytes,
    ])?;
    let transformer_load_phase_host_bytes = crate::h3_factory::transformer_load_phase_host_bytes(
        crate::h3_factory::H3TransformerLoadHostTerms {
            attempt_host_bytes,
            transformer_alive_metadata_host_bytes,
            condition_backing_host_bytes,
            packed_layout_host_bytes,
            text_modality_tags_host_bytes,
            schedule_host_bytes,
            fixed_transformer_load_host_staging_bytes,
            turbo_adapter_host_staging_bytes,
        },
    )?;
    let denoise_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        transformer_alive_metadata_host_bytes,
        condition_backing_host_bytes,
        packed_layout_host_bytes,
        text_modality_tags_host_bytes,
        schedule_host_bytes,
        max_streamed_block_host_overlap_bytes,
    ])?;
    let visual_decode_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        vae_retained_config_host_bytes,
        packed_layout_host_bytes,
        vae_memory.peak_host_io_buffer_bytes,
        bounds.encoded_video_host_bytes_bound,
        bounds.thumbnail_host_bytes_bound,
    ])?;
    let audio_decode_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        packed_layout_host_bytes,
        bounds.encoded_video_host_bytes_bound,
        bounds.thumbnail_host_bytes_bound,
        waveform_host_bytes,
    ])?;
    let waveform_transfer_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        bounds.encoded_video_host_bytes_bound,
        bounds.thumbnail_host_bytes_bound,
        waveform_host_bytes,
    ])?;
    let mux_phase_host_bytes = checked_sum([
        attempt_host_bytes,
        bounds.encoded_video_host_bytes_bound,
        bounds.thumbnail_host_bytes_bound,
        waveform_host_bytes,
        bounds.mux_output_host_bytes_bound,
        bounds.aac_mux_staging_host_bytes,
    ])?;
    // Ref2VA's four leading phases. Everything above is task-neutral; these
    // and the load/drop order are the whole difference, and they must agree
    // exactly with `validate_target_budget`'s own derivation.
    let ref2va = request.task == Task::Ref2va;
    let reference_normalized_media_host_bytes = checked_sum(
        request
            .references
            .iter()
            .map(|reference| reference.normalized_host_bytes),
    )?;
    // Every reference's NATIVE decoded payload is held simultaneously: the
    // orchestrator decodes the whole ordered set before preprocessing any.
    let reference_native_media_host_bytes = checked_sum(
        request
            .references
            .iter()
            .map(|reference| reference.native_host_bytes),
    )?;
    // The transient on top of that is the one reference being materialized.
    let reference_decode_staging_host_bytes = request
        .references
        .iter()
        .map(|reference| reference.native_host_bytes)
        .max()
        .unwrap_or(0);
    // Preprocess writes one normalized payload while its native source is
    // still held; the transient is the largest normalized form.
    let reference_preprocess_staging_host_bytes = request
        .references
        .iter()
        .map(|reference| reference.normalized_host_bytes)
        .max()
        .unwrap_or(0);
    // The audio encoder has no FL2VA analogue; it borrows the decoder's
    // measured workspace, which runs the larger BigVGAN stack.
    let reference_audio_encode_workspace_device_bytes = if ref2va {
        bounds.audio_decode_workspace_device_bytes
    } else {
        0
    };
    let reference_decode_phase_host_bytes = if ref2va {
        checked_sum([
            attempt_host_bytes,
            qwen_alive_metadata_host_bytes,
            reference_native_media_host_bytes,
            reference_decode_staging_host_bytes,
        ])?
    } else {
        0
    };
    let reference_preprocess_phase_host_bytes = if ref2va {
        checked_sum([
            attempt_host_bytes,
            qwen_alive_metadata_host_bytes,
            reference_native_media_host_bytes,
            reference_normalized_media_host_bytes,
            reference_preprocess_staging_host_bytes,
        ])?
    } else {
        0
    };
    let reference_encode_phase_host_bytes = if ref2va {
        checked_sum([
            attempt_host_bytes,
            transformer_alive_metadata_host_bytes,
            reference_normalized_media_host_bytes,
            condition_backing_host_bytes,
            packed_layout_host_bytes,
            text_modality_tags_host_bytes,
        ])?
    } else {
        0
    };
    let reference_decode_phase_device_bytes = if ref2va {
        bounds.fixed_runtime_device_bytes
    } else {
        0
    };
    let reference_preprocess_phase_device_bytes = reference_decode_phase_device_bytes;
    let reference_visual_encode_phase_device_bytes = if ref2va {
        checked_sum([
            bounds.fixed_runtime_device_bytes,
            retained_vaes,
            qwen_output_state_device_bytes,
            bounds.condition_vae_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
        ])?
    } else {
        0
    };
    let reference_audio_encode_phase_device_bytes = if ref2va {
        checked_sum([
            bounds.fixed_runtime_device_bytes,
            retained_vaes,
            qwen_output_state_device_bytes,
            reference_audio_encode_workspace_device_bytes,
            condition_latent_backing_device_bytes,
            packed_layout_device_bytes,
        ])?
    } else {
        0
    };
    // Condition encode is not in Ref2VA's order at all.
    let condition_encode_phase_host_bytes = if ref2va {
        0
    } else {
        condition_encode_phase_host_bytes
    };
    let condition_encode_phase_device_bytes = if ref2va {
        0
    } else {
        condition_encode_phase_device_bytes
    };
    let predicted_host_increment_bytes = [
        reference_decode_phase_host_bytes,
        reference_preprocess_phase_host_bytes,
        reference_encode_phase_host_bytes,
        vae_load_phase_host_bytes,
        qwen_encode_phase_host_bytes,
        qwen_transfer_phase_host_bytes,
        condition_encode_phase_host_bytes,
        noise_allocation_phase_host_bytes,
        transformer_load_phase_host_bytes,
        denoise_phase_host_bytes,
        visual_decode_phase_host_bytes,
        audio_decode_phase_host_bytes,
        waveform_transfer_phase_host_bytes,
        mux_phase_host_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let predicted_device_peak_bytes = [
        predicted_device_peak_bytes,
        reference_decode_phase_device_bytes,
        reference_preprocess_phase_device_bytes,
        reference_visual_encode_phase_device_bytes,
        reference_audio_encode_phase_device_bytes,
        condition_encode_phase_device_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let mut budget = H3FactoryTargetBudgetInput {
        identity_sha256: String::new(),
        load_drop_policy: if ref2va {
            H3FactoryTargetLoadDropPolicy::DecodeReferencesPreprocessReferencesLoadVaesLoadQwenEncodeVisionTransferDropQwenEncodeVisualReferencesEncodeAudioReferencesParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux
        } else {
            H3FactoryTargetLoadDropPolicy::LoadVaesLoadQwenEncodeTransferDropQwenEncodeConditionsParkVaesAllocateNoiseLoadTransformerDenoiseDropTransformerReloadVaesDecodeVisualAudioDropVaesMux
        },
        artifacts,
        artifact_host_bytes,
        fixed_runtime_host_bytes: bounds.fixed_runtime_host_bytes,
        qwen_host_parameter_bytes,
        qwen_host_activation_bytes,
        qwen_host_output_state_bytes,
        qwen_host_workspace_bytes,
        condition_backing_host_bytes,
        endpoint_encoded_host_bytes,
        normalized_endpoint_host_bytes,
        schedule_host_bytes,
        packed_layout_host_bytes,
        packed_layout_construction_staging_host_bytes,
        packed_layout_freeze_staging_host_bytes,
        text_modality_tags_host_bytes,
        noise_cpu_staging_host_bytes,
        reference_media_identity_sha256: expected_h3_factory_reference_media_identity(
            &request.references,
        ),
        reference_normalized_media_host_bytes,
        reference_decode_staging_host_bytes,
        reference_preprocess_staging_host_bytes,
        reference_decode_phase_host_bytes,
        reference_preprocess_phase_host_bytes,
        reference_visual_encode_phase_host_bytes: reference_encode_phase_host_bytes,
        reference_audio_encode_phase_host_bytes: reference_encode_phase_host_bytes,
        reference_audio_encode_workspace_device_bytes,
        reference_decode_phase_device_bytes,
        reference_preprocess_phase_device_bytes,
        reference_visual_encode_phase_device_bytes,
        reference_audio_encode_phase_device_bytes,
        vae_peak_host_io_buffer_bytes: vae_memory.peak_host_io_buffer_bytes,
        vae_peak_host_mapped_file_bytes: vae_memory.peak_host_mapped_file_bytes,
        vae_peak_staging_disk_bytes: vae_memory.peak_staging_disk_bytes,
        max_host_read_staging_bytes,
        max_streamed_block_host_overlap_bytes,
        fixed_transformer_load_host_staging_bytes,
        encoded_video_host_bytes_bound: bounds.encoded_video_host_bytes_bound,
        thumbnail_host_bytes_bound: bounds.thumbnail_host_bytes_bound,
        waveform_host_bytes,
        mux_output_host_bytes_bound: bounds.mux_output_host_bytes_bound,
        aac_mux_staging_host_bytes: bounds.aac_mux_staging_host_bytes,
        qwen_host_load_staging_bytes,
        qwen_retained_header_host_bytes,
        transformer_retained_header_host_bytes,
        vae_retained_config_host_bytes,
        vae_load_phase_host_bytes,
        qwen_encode_phase_host_bytes,
        qwen_transfer_phase_host_bytes,
        condition_encode_phase_host_bytes,
        noise_allocation_phase_host_bytes,
        transformer_load_phase_host_bytes,
        denoise_phase_host_bytes,
        visual_decode_phase_host_bytes,
        audio_decode_phase_host_bytes,
        waveform_transfer_phase_host_bytes,
        mux_phase_host_bytes,
        predicted_host_increment_bytes,
        fixed_runtime_device_bytes: bounds.fixed_runtime_device_bytes,
        fixed_transformer_device_bytes: checkpoint.fixed_transformer_protected_device_bytes,
        visual_vae_resident_device_bytes: vae_memory.visual.resident_device_weight_bytes,
        audio_vae_resident_device_bytes: vae_memory.audio.resident_device_weight_bytes,
        attempt_resident_vae_device_bytes: retained_vaes,
        vae_construction_device_workspace_bytes: bounds.vae_construction_device_workspace_bytes,
        vae_memory_evidence_identity_sha256: vae.artifact_validation_identity_sha256().into(),
        qwen_device_parameter_bytes,
        qwen_activation_device_bytes,
        qwen_output_state_device_bytes,
        qwen_output_transfer_device_bytes,
        condition_vae_workspace_device_bytes,
        condition_latent_backing_device_bytes,
        target_video_latent_device_bytes,
        target_audio_latent_device_bytes,
        packed_layout_device_bytes,
        packed_video_state_device_bytes,
        packed_audio_state_device_bytes,
        denoise_copy_policy: H3FactoryTargetDenoiseCopyPolicy::CandleF32PairedResMultistepV2,
        denoise_tensor_copy_workspace_device_bytes,
        audio_waveform_device_bytes: waveform_host_bytes,
        attention_workspace_device_bytes: bounds.attention_workspace_device_bytes,
        ffn_workspace_device_bytes: bounds.ffn_workspace_device_bytes,
        decoder_tile_workspace_device_bytes: bounds.decoder_tile_workspace_device_bytes,
        audio_decode_workspace_device_bytes: bounds.audio_decode_workspace_device_bytes,
        resident_block_device_bytes: 0,
        streamed_block_device_bytes,
        prefetch_device_bytes: 0,
        dequantization_workspace_device_bytes: max_device_weight_staging_bytes,
        protected_block_device_bytes,
        streamed_block_device_overlap_bytes,
        max_device_weight_staging_bytes,
        fixed_transformer_load_device_staging_bytes,
        turbo_adapter_device_bytes,
        turbo_adapter_device_staging_bytes,
        turbo_adapter_host_staging_bytes,
        vae_load_phase_device_bytes,
        qwen_encode_phase_device_bytes,
        qwen_transfer_phase_device_bytes,
        condition_encode_phase_device_bytes,
        noise_allocation_phase_device_bytes,
        transformer_load_phase_device_bytes,
        denoise_phase_device_bytes,
        visual_decode_phase_device_bytes,
        audio_decode_phase_device_bytes,
        waveform_transfer_phase_device_bytes,
        mux_phase_device_bytes: 0,
        predicted_device_peak_bytes,
    };
    budget.identity_sha256 = expected_h3_factory_target_budget_identity(&budget);
    Ok(budget)
}

fn conditioning_fingerprint(endpoints: &[H3FactoryEndpointInput]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.fl2va-conditioning.v1\0");
    for endpoint in endpoints {
        digest.update(match endpoint.anchor {
            H3FactoryEndpointAnchor::First => b"first".as_slice(),
            H3FactoryEndpointAnchor::Last => b"last".as_slice(),
        });
        digest.update(endpoint.encoded_content_sha256.as_bytes());
        digest.update(endpoint.normalized_cpu_content_sha256.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn factory_anchor(anchor: H3EndpointAnchor) -> H3FactoryEndpointAnchor {
    match anchor {
        H3EndpointAnchor::First => H3FactoryEndpointAnchor::First,
        H3EndpointAnchor::Last => H3FactoryEndpointAnchor::Last,
    }
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(value))
        .ok_or_else(|| anyhow!("private H3 byte sum overflow"))
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("{label} must be a lowercase 64-character SHA-256 digest")
    }
    Ok(())
}

fn sha256(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::minimax_h3::pipeline::H3PreparedEndpoint;
    use candle_core::DType;

    fn prepared_runtime_pair() -> (H3PreparedFl2VaRequest, H3FactoryPreparedRequestInput) {
        let prompt = "bound prompt".to_string();
        let endpoint = H3PreparedEndpoint {
            anchor: H3EndpointAnchor::First,
            source_width: 64,
            source_height: 64,
            resize: super::super::pipeline::H3EndpointResize::Identity,
            pixels: candle_core::Tensor::zeros((1, 3, 1, 64, 64), DType::U8, &Device::Cpu).unwrap(),
        };
        let factory_endpoint = H3FactoryEndpointInput {
            anchor: H3FactoryEndpointAnchor::First,
            encoded_bytes: 7,
            encoded_content_sha256: sha256(b"encoded"),
            preprocess: H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1,
            normalized_shape: [1, 3, 1, 64, 64],
            normalized_cpu_bytes: 12_288,
            normalized_cpu_content_sha256: sha256(vec![0_u8; 12_288]),
        };
        let geometry = super::super::pipeline::H3Fl2VaGeometry {
            mode: Mode::FirstFrameToAudioVideo,
            width: 64,
            height: 64,
            frames: 33,
            latent_frames: 4,
            latent_width: 4,
            latent_height: 4,
            audio_latents_per_channel: 40,
            rows_per_video_frame: 4,
            condition_video_rows: 4,
            generated_video_rows: 16,
            generated_audio_rows: 80,
        };
        let grid_points = 5;
        let denoise_forward_count = u32::try_from(
            H3DualSchedule::new(grid_points)
                .unwrap()
                .counts()
                .transformer_evaluations,
        )
        .unwrap();
        let prepared = H3PreparedFl2VaRequest {
            geometry,
            endpoints: vec![endpoint],
            prompt: prompt.clone(),
            seed: 42,
            grid_points,
        };
        let current_endpoints = vec![factory_endpoint.clone()];
        let frozen = H3FactoryPreparedRequestInput {
            identity_sha256: sha256(b"request"),
            canonical_model: contract::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            mode: Mode::FirstFrameToAudioVideo,
            prompt_sha256: sha256(prompt.as_bytes()),
            seed: 42,
            grid_points: u32::try_from(grid_points).unwrap(),
            denoise_forward_count,
            guidance_f64_bits: 0,
            strength_f64_bits: 0,
            batch_size: 1,
            width: 64,
            height: 64,
            frames: 33,
            fps: contract::FIXED_FPS,
            synchronized_audio: true,
            mp4_output: true,
            video_latent_frames: 4,
            audio_latents_per_channel: 40,
            audio_samples_per_channel: 32_000,
            conditioning_fingerprint: conditioning_fingerprint(&current_endpoints),
            reference_fingerprint: sha256(b"mold.minimax-h3.fl2va-no-references.v1"),
            endpoints: vec![factory_endpoint],
            references: Vec::new(),
            rows: H3FactoryPreparedRowsInput {
                qwen_output_text_rows: 3,
                qwen_vision_rows: 64,
                condition_visual_rows: 4,
                condition_audio_rows: 0,
                target_video_rows: 16,
                target_audio_rows: 80,
                total_packed_rows: 103,
            },
        };
        (prepared, frozen)
    }

    #[test]
    fn opened_authorities_and_prepared_attempt_are_single_consumption() {
        trait AmbiguousIfClone<Marker> {
            fn assert_not_implemented() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        struct ImplementsClone;
        impl<T: Clone> AmbiguousIfClone<ImplementsClone> for T {}

        <H3PrivateComfyStorageAuthority as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3PrivateOpenedTaskConfigAuthority as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3ComfyOpenedInt8Checkpoint as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3AuthenticatedQwenNvfp4Authority as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3AuthenticatedComfyVaeAuthority as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3PrivatePreparedFl2VaAttempt as AmbiguousIfClone<_>>::assert_not_implemented();
        <H3PrivatePreparedFl2VaFactoryInputs as AmbiguousIfClone<_>>::assert_not_implemented();
    }

    #[test]
    fn opened_task_config_rejects_path_replacement_after_authentication() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.json");
        let bytes = br#"{"task":"fl2va"}"#;
        std::fs::write(&path, bytes).unwrap();
        let authority =
            H3PrivateOpenedTaskConfigAuthority::open(&path, bytes.len() as u64, &sha256(bytes))
                .unwrap();
        authority.revalidate().unwrap();

        std::fs::rename(&path, root.path().join("authenticated-config.json")).unwrap();
        std::fs::write(&path, bytes).unwrap();
        assert!(authority.revalidate().is_err());
    }

    #[test]
    fn opened_task_config_rejects_content_mutation_after_authentication() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.json");
        let bytes = b"task-config-a";
        std::fs::write(&path, bytes).unwrap();
        let authority =
            H3PrivateOpenedTaskConfigAuthority::open(&path, bytes.len() as u64, &sha256(bytes))
                .unwrap();

        std::fs::write(&path, b"task-config-b").unwrap();
        assert!(authority.revalidate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn opened_task_config_rejects_symlink_and_accepts_group_writable_sources() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.json");
        let alias = root.path().join("config-alias.json");
        let bytes = b"task-config";
        std::fs::write(&path, bytes).unwrap();
        symlink(&path, &alias).unwrap();
        assert!(H3PrivateOpenedTaskConfigAuthority::open(
            &alias,
            bytes.len() as u64,
            &sha256(bytes),
        )
        .is_err());

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert!(H3PrivateOpenedTaskConfigAuthority::open(
            &path,
            bytes.len() as u64,
            &sha256(bytes),
        )
        .is_ok());
    }

    #[test]
    fn hidden_storage_resolution_is_canonical_and_identity_bound() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path().canonicalize().unwrap();
        let authority = H3PrivateComfyStorageAuthority::resolve(&root).unwrap();
        authority.validate().unwrap();
        assert!(authority
            .transformer_path()
            .ends_with(FL2VA_TRANSFORMER_SOURCE));
        assert!(authority.qwen_weights_path().ends_with(QWEN_WEIGHT_SOURCE));
        assert!(authority
            .task_config_path()
            .ends_with(FL2VA_TASK_CONFIG_SOURCE));
        assert_eq!(authority.identity_sha256().len(), 64);
    }

    #[cfg(unix)]
    #[test]
    fn hidden_storage_rejects_symlink_and_accepts_group_writable_roots() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let parent = tempfile::tempdir().unwrap();
        let real = parent.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let real = real.canonicalize().unwrap();
        let alias = parent.path().join("alias");
        symlink(&real, &alias).unwrap();
        assert!(H3PrivateComfyStorageAuthority::resolve(&alias).is_err());

        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(H3PrivateComfyStorageAuthority::resolve(&real).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn hidden_storage_rejects_root_identity_replacement() {
        let parent = tempfile::tempdir().unwrap();
        let parent = parent.path().canonicalize().unwrap();
        let root = parent.join("models");
        std::fs::create_dir(&root).unwrap();
        let authority = H3PrivateComfyStorageAuthority::resolve(&root).unwrap();
        let authenticated = parent.join("authenticated-models");
        std::fs::rename(&root, authenticated).unwrap();
        std::fs::create_dir(&root).unwrap();

        let error = authority.validate().unwrap_err();
        assert!(error.to_string().contains("authority changed"), "{error}");
    }

    #[test]
    fn cross_task_transformer_and_vae_authorities_fail_closed() {
        assert!(validate_fl2va_transformer_contract(
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot
        )
        .is_err());
        assert!(validate_fl2va_vae_contract(Task::Ref2va, contract::REF2VA_COMFY).is_err());
        validate_fl2va_transformer_contract(H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot)
            .unwrap();
        validate_fl2va_vae_contract(Task::Fl2va, contract::FL2VA_COMFY).unwrap();
    }

    #[test]
    fn execution_fingerprint_must_be_a_digest() {
        assert!(require_sha256("not-a-digest", "execution").is_err());
        assert!(require_sha256(&"A".repeat(64), "execution").is_err());
        require_sha256(&sha256(b"execution"), "execution").unwrap();
    }

    #[test]
    fn normalized_digest_is_an_identity_axis() {
        let first = H3FactoryEndpointInput {
            anchor: H3FactoryEndpointAnchor::First,
            encoded_bytes: 3,
            encoded_content_sha256: sha256(b"raw-a"),
            preprocess: H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1,
            normalized_shape: [1, 3, 1, 64, 64],
            normalized_cpu_bytes: 12_288,
            normalized_cpu_content_sha256: sha256(b"normalized-a"),
        };
        let mut second = first.clone();
        second.normalized_cpu_content_sha256 = sha256(b"normalized-b");
        assert_ne!(
            conditioning_fingerprint(&[first]),
            conditioning_fingerprint(&[second])
        );
    }

    #[test]
    fn prepared_runtime_rejects_same_shape_prompt_and_pixel_mutation() {
        let (mut prepared, frozen) = prepared_runtime_pair();
        validate_prepared_runtime_request(&prepared, &frozen).unwrap();

        prepared.prompt = "mutated prompt".into();
        assert!(validate_prepared_runtime_request(&prepared, &frozen).is_err());

        let (mut prepared, frozen) = prepared_runtime_pair();
        prepared.endpoints[0].pixels =
            candle_core::Tensor::ones((1, 3, 1, 64, 64), DType::U8, &Device::Cpu).unwrap();
        assert!(validate_prepared_runtime_request(&prepared, &frozen).is_err());
    }
}
