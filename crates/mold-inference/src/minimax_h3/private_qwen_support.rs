//! Opened-file-bound Qwen support assets for MiniMax H3.
//!
//! Public `h3` and private `h3-private-uat` builds both resolve the five
//! lightweight conditioner files through the exact pinned manifest and
//! `storage_path` contract, then parse only bytes read from authenticated,
//! no-follow file descriptors. It does not register an engine, expose runtime
//! capability, download anything, or relax H3 execution admission.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use mold_candle::minimax_h3::{
    register_extra_special_tokens, validate_processor_assets, H3ConditionerConfig,
    H3_EXTRA_SPECIAL_TOKENS, H3_REGISTERED_MAX_TOKEN_ID, H3_REGISTERED_VOCABULARY_SIZE,
};
use mold_core::manifest::{find_manifest, storage_path, ModelComponent};
use mold_core::minimax_h3::{self as contract, ArtifactRole as ManifestArtifactRole, Layout, Task};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

/// Retained by every executable that reaches the support loader. Private UAT
/// keeps a release-rejectable marker; the public H3 runtime uses a distinct
/// non-private provenance marker for the same authenticated loader mechanics.
#[cfg(feature = "h3-private-uat")]
pub const H3_PRIVATE_QWEN_SUPPORT_CLAIM_MARKER: &str =
    "mold.minimax-h3.private-uat-qwen-support-loader.v1";

#[cfg(all(feature = "h3", not(feature = "h3-private-uat")))]
pub const H3_PRIVATE_QWEN_SUPPORT_CLAIM_MARKER: &str = "mold.minimax-h3.qwen-support-loader.v1";

const SUPPORT_IDENTITY_SCHEMA: &str = "mold.minimax-h3.private-qwen-support.v1";
const RELEASED_TOKENIZER_BASE_VOCAB_SIZE: usize = 151_643;
const RELEASED_TOKENIZER_ADDED_TOKEN_COUNT: usize = 26;
const RELEASED_TOKENIZER_VOCAB_SIZE: usize =
    RELEASED_TOKENIZER_BASE_VOCAB_SIZE + RELEASED_TOKENIZER_ADDED_TOKEN_COUNT;
const RELEASED_TOKENIZER_MAX_ID: u32 = 151_668;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SupportRole {
    ArchitectureConfig,
    Tokenizer,
    TokenizerConfig,
    ImageProcessorConfig,
    VideoProcessorConfig,
}

impl SupportRole {
    const ALL: [Self; 5] = [
        Self::ArchitectureConfig,
        Self::Tokenizer,
        Self::TokenizerConfig,
        Self::ImageProcessorConfig,
        Self::VideoProcessorConfig,
    ];

    const fn stable_id(self) -> &'static str {
        match self {
            Self::ArchitectureConfig => "architecture-config",
            Self::Tokenizer => "tokenizer",
            Self::TokenizerConfig => "tokenizer-config",
            Self::ImageProcessorConfig => "image-processor-config",
            Self::VideoProcessorConfig => "video-processor-config",
        }
    }

    const fn source_path(self) -> &'static str {
        match self {
            Self::ArchitectureConfig => "text_encoder/config.json",
            Self::Tokenizer => "processor/tokenizer.json",
            Self::TokenizerConfig => "processor/tokenizer_config.json",
            Self::ImageProcessorConfig => "processor/preprocessor_config.json",
            Self::VideoProcessorConfig => "processor/video_preprocessor_config.json",
        }
    }

    const fn component(self) -> ModelComponent {
        match self {
            Self::ArchitectureConfig => ModelComponent::ModelConfig,
            Self::Tokenizer
            | Self::TokenizerConfig
            | Self::ImageProcessorConfig
            | Self::VideoProcessorConfig => ModelComponent::Processor,
        }
    }

    const fn manifest_role(self) -> ManifestArtifactRole {
        match self {
            Self::ArchitectureConfig => ManifestArtifactRole::SharedConfig,
            Self::Tokenizer
            | Self::TokenizerConfig
            | Self::ImageProcessorConfig
            | Self::VideoProcessorConfig => ManifestArtifactRole::Processor,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SupportFileContract {
    role: SupportRole,
    relative_path: PathBuf,
    source_repo: String,
    source_revision: String,
    source_path: String,
    size_bytes: u64,
    sha256: String,
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

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
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
            #[cfg(not(unix))]
            modified: metadata.modified().ok(),
        }
    }
}

#[derive(Debug)]
struct OpenedSupportArtifact {
    role: SupportRole,
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    expected_sha256: String,
    /// Completed full re-hashes of the opened descriptor. Revalidation runs
    /// one full content re-hash per opened descriptor; afterwards the
    /// dev/inode/mtime/ctime opened-file identity is the change detector, the
    /// same metadata-identity trust model the qualification digest caches use.
    /// Revalidation is on the per-request admission path, so an unconditional
    /// re-hash turns these small files into a hashing hot loop.
    full_rehashes: std::sync::atomic::AtomicU64,
}

impl OpenedSupportArtifact {
    fn revalidate(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        let before = FileIdentity::from_metadata(
            &self
                .file
                .metadata()
                .with_context(|| format!("failed to stat opened {}", self.role.stable_id()))?,
        );
        if before != self.identity {
            bail!(
                "private H3 {} opened-file identity changed after authentication",
                self.role.stable_id()
            )
        }
        if self.full_rehashes.load(Ordering::Acquire) == 0 {
            let actual_sha256 = sha256_open_file(&self.file)
                .with_context(|| format!("failed to rehash opened {}", self.role.stable_id()))?;
            let after = FileIdentity::from_metadata(&self.file.metadata()?);
            if before != after || actual_sha256 != self.expected_sha256 {
                bail!(
                    "private H3 {} changed after authentication",
                    self.role.stable_id()
                )
            }
            self.full_rehashes.fetch_add(1, Ordering::AcqRel);
        }
        let current = open_regular_file_no_follow(&self.path).with_context(|| {
            format!(
                "failed to re-open private H3 {} through the manifest path",
                self.role.stable_id()
            )
        })?;
        if FileIdentity::from_metadata(&current.metadata()?) != self.identity {
            bail!(
                "private H3 {} manifest path no longer names the authenticated opened file",
                self.role.stable_id()
            )
        }
        Ok(())
    }
}

/// Immutable, parsed Qwen state plus the opened descriptors that established
/// its private qualification authority. Paths and raw bytes remain private;
/// the conditioner receives only the exact config, tokenizer, stable support
/// identity, and a revalidation operation.
pub(crate) struct H3PrivateQwenSupport {
    model: String,
    task: Task,
    conditioner_config: H3ConditionerConfig,
    tokenizer: Arc<Tokenizer>,
    support_identity_sha256: String,
    artifact_facts: Vec<H3PrivateQwenSupportArtifactFact>,
    artifacts: Vec<OpenedSupportArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateQwenSupportArtifactFact {
    pub(crate) role: &'static str,
    pub(crate) source_repository: String,
    pub(crate) source_revision: String,
    pub(crate) source_path: String,
    pub(crate) content_sha256: String,
    pub(crate) file_bytes: u64,
}

impl H3PrivateQwenSupport {
    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) const fn task(&self) -> Task {
        self.task
    }

    pub(crate) fn conditioner_config(&self) -> &H3ConditionerConfig {
        &self.conditioner_config
    }

    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub(crate) fn support_identity_sha256(&self) -> &str {
        &self.support_identity_sha256
    }

    pub(crate) fn artifact_facts(&self) -> &[H3PrivateQwenSupportArtifactFact] {
        &self.artifact_facts
    }

    pub(crate) fn validate_storage_root(&self, models_root: &Path) -> Result<()> {
        let contracts = production_support_contracts(&self.model)?;
        for artifact in &self.artifacts {
            let contract = contracts
                .iter()
                .find(|contract| contract.role == artifact.role)
                .ok_or_else(|| anyhow!("private H3 support storage binding omits a role"))?;
            if artifact.path != models_root.join(&contract.relative_path)
                || std::fs::canonicalize(&artifact.path)? != artifact.path
            {
                bail!("private H3 support descriptor is outside the hidden storage authority")
            }
        }
        Ok(())
    }

    /// Prove that every retained descriptor and its manifest path still name
    /// the authenticated opened files before conditioner construction. The
    /// first call fully re-hashes each descriptor's content; later calls rely
    /// on the opened-file metadata identity, which detects any write through
    /// the filesystem via ctime.
    pub(crate) fn revalidate(&self) -> Result<()> {
        for artifact in &self.artifacts {
            artifact.revalidate()?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn full_rehash_count(&self) -> u64 {
        self.artifacts
            .iter()
            .map(|artifact| {
                artifact
                    .full_rehashes
                    .load(std::sync::atomic::Ordering::Acquire)
            })
            .sum()
    }
}

/// Independently authenticate and parse the already-qualified lightweight
/// Qwen files for one exact hidden Comfy manifest.
///
/// The private artifact qualifier remains the campaign/host authorization
/// boundary. This loader intentionally repeats the small-file hashes so a
/// prior report cannot become a time-of-check/time-of-use bypass.
pub(crate) fn load_qualified_private_qwen_support(
    models_root: &Path,
    model: &str,
) -> Result<H3PrivateQwenSupport> {
    std::hint::black_box(H3_PRIVATE_QWEN_SUPPORT_CLAIM_MARKER);
    let contracts = production_support_contracts(model)?;
    load_support_from_contracts(models_root, model, contracts)
}

fn production_support_contracts(model: &str) -> Result<Vec<SupportFileContract>> {
    let task = match model {
        contract::FL2VA_COMFY => Task::Fl2va,
        contract::REF2VA_COMFY => Task::Ref2va,
        _ => bail!("private H3 Qwen support requires an exact Comfy manifest name"),
    };
    let manifest =
        find_manifest(model).ok_or_else(|| anyhow!("missing MiniMax H3 manifest for {model}"))?;
    let manifest_authority = contract::manifest_contract(manifest)
        .ok_or_else(|| anyhow!("MiniMax H3 manifest has no frozen contract"))?;
    if manifest.name != model
        || manifest.family != contract::FAMILY
        || manifest_authority.task != task
        || manifest_authority.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq
        || (!cfg!(feature = "h3") && manifest_authority.runtime_available)
    {
        bail!("private H3 Qwen support requires the exact Comfy manifest")
    }

    let mut contracts = Vec::with_capacity(SupportRole::ALL.len());
    for role in SupportRole::ALL {
        let matches = manifest
            .files
            .iter()
            .filter(|file| {
                file.hf_filename == role.source_path() && file.component == role.component()
            })
            .collect::<Vec<_>>();
        let [file] = matches.as_slice() else {
            bail!(
                "private H3 manifest must contain exactly one {}",
                role.source_path()
            )
        };
        let artifact_authority = contract::artifact_contract(manifest, file)
            .ok_or_else(|| anyhow!("{} has no frozen H3 artifact contract", role.source_path()))?;
        if artifact_authority.role != role.manifest_role()
            || artifact_authority.identity.source_repo != contract::OFFICIAL_REPO
            || artifact_authority.identity.source_revision != contract::OFFICIAL_REVISION
            || artifact_authority.identity.source_path != role.source_path()
            || !artifact_authority.compatible_tasks.contains(&Task::Fl2va)
            || !artifact_authority.compatible_tasks.contains(&Task::Ref2va)
        {
            bail!(
                "private H3 {} differs from the shared Qwen support contract",
                role.stable_id()
            )
        }
        let relative_path = storage_path(manifest, file);
        validate_relative_path(&relative_path)?;
        contracts.push(SupportFileContract {
            role,
            relative_path,
            source_repo: artifact_authority.identity.source_repo.to_owned(),
            source_revision: artifact_authority.identity.source_revision.to_owned(),
            source_path: artifact_authority.identity.source_path.to_owned(),
            size_bytes: file.size_bytes,
            sha256: artifact_authority.identity.sha256.to_owned(),
        });
    }
    validate_contract_set(&contracts)?;
    Ok(contracts)
}

fn load_support_from_contracts(
    models_root: &Path,
    model: &str,
    contracts: Vec<SupportFileContract>,
) -> Result<H3PrivateQwenSupport> {
    load_support_from_contracts_at_boundary(models_root, model, contracts, |_| Ok(()))
}

fn load_support_from_contracts_at_boundary(
    models_root: &Path,
    model: &str,
    contracts: Vec<SupportFileContract>,
    root_validated: impl FnOnce(&Path) -> Result<()>,
) -> Result<H3PrivateQwenSupport> {
    validate_contract_set(&contracts)?;
    let task = private_comfy_task(model)?;
    if !models_root.is_absolute() {
        bail!("private H3 models root must be absolute")
    }
    let root_metadata = std::fs::symlink_metadata(models_root)
        .context("failed to inspect the private H3 models root")?;
    if !root_metadata.file_type().is_dir() || root_metadata.file_type().is_symlink() {
        bail!("private H3 models root must be a real non-symlink directory")
    }
    let root_identity = FileIdentity::from_metadata(&root_metadata);
    root_validated(models_root)?;
    validate_models_root_identity(models_root, &root_identity)?;
    let root = models_root.to_path_buf();

    let support_identity_sha256 = support_identity(model, &contracts)?;
    let artifact_facts = contracts
        .iter()
        .map(|contract| H3PrivateQwenSupportArtifactFact {
            role: contract.role.stable_id(),
            source_repository: contract.source_repo.clone(),
            source_revision: contract.source_revision.clone(),
            source_path: contract.source_path.clone(),
            content_sha256: contract.sha256.clone(),
            file_bytes: contract.size_bytes,
        })
        .collect();
    let mut opened = Vec::with_capacity(contracts.len());
    let mut bytes_by_role = BTreeMap::new();
    for contract in &contracts {
        let (artifact, bytes) = authenticate_support_file(&root, contract)?;
        if bytes_by_role.insert(contract.role, bytes).is_some() {
            bail!(
                "private H3 support contract repeats {}",
                contract.role.stable_id()
            )
        }
        opened.push(artifact);
    }
    validate_models_root_identity(models_root, &root_identity)?;

    let bytes = |role: SupportRole| -> Result<&[u8]> {
        bytes_by_role
            .get(&role)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow!("private H3 support omits {}", role.stable_id()))
    };
    let conditioner_config =
        H3ConditionerConfig::from_json(bytes(SupportRole::ArchitectureConfig)?)?;
    validate_tokenizer_config(bytes(SupportRole::TokenizerConfig)?)?;
    validate_processor_assets(
        bytes(SupportRole::ImageProcessorConfig)?,
        bytes(SupportRole::VideoProcessorConfig)?,
    )?;
    let mut tokenizer = Tokenizer::from_bytes(bytes(SupportRole::Tokenizer)?)
        .map_err(|error| anyhow!("invalid pinned H3 tokenizer: {error}"))?;
    // Validate the file's own shape before registering, so a swapped
    // tokenizer.json is still caught, then register the configured extras and
    // validate the shape they produce.
    validate_released_tokenizer_shape(&tokenizer)?;
    register_extra_special_tokens(&mut tokenizer, bytes(SupportRole::TokenizerConfig)?)
        .map_err(|error| anyhow!("pinned H3 tokenizer special tokens: {error}"))?;
    validate_tokenizer(&tokenizer, &conditioner_config)?;

    let support = H3PrivateQwenSupport {
        model: model.to_owned(),
        task,
        conditioner_config,
        tokenizer: Arc::new(tokenizer),
        support_identity_sha256,
        artifact_facts,
        artifacts: opened,
    };
    support.revalidate()?;
    Ok(support)
}

fn private_comfy_task(model: &str) -> Result<Task> {
    let capability = contract::capability_contract_for_model(model)
        .ok_or_else(|| anyhow!("private H3 Qwen support model has no capability contract"))?;
    if capability.canonical_model != model
        || capability.layout != Layout::ComfyPrunedInt8ConvrotNvfp4Awq
    {
        bail!("private H3 Qwen support requires an exact Comfy canonical model")
    }
    Ok(capability.task)
}

fn validate_models_root_identity(path: &Path, expected: &FileIdentity) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .context("failed to revalidate the private H3 models root")?;
    if !metadata.file_type().is_dir()
        || metadata.file_type().is_symlink()
        || FileIdentity::from_metadata(&metadata) != *expected
    {
        bail!("private H3 models root changed after validation")
    }
    Ok(())
}

fn validate_contract_set(contracts: &[SupportFileContract]) -> Result<()> {
    if contracts.len() != SupportRole::ALL.len() {
        bail!(
            "private H3 Qwen support requires exactly {} files, found {}",
            SupportRole::ALL.len(),
            contracts.len()
        )
    }
    let mut roles = BTreeSet::new();
    for contract in contracts {
        if !roles.insert(contract.role) {
            bail!(
                "private H3 Qwen support repeats {}",
                contract.role.stable_id()
            )
        }
        validate_relative_path(&contract.relative_path)?;
        if contract.size_bytes == 0
            || contract.sha256.len() != 64
            || !contract
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!(
                "private H3 {} has an invalid pinned fingerprint",
                contract.role.stable_id()
            )
        }
    }
    if roles != SupportRole::ALL.into_iter().collect() {
        bail!("private H3 Qwen support role set differs from the runtime contract")
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("private H3 manifest produced an unsafe support path")
    }
    Ok(())
}

fn authenticate_support_file(
    root: &Path,
    contract: &SupportFileContract,
) -> Result<(OpenedSupportArtifact, Vec<u8>)> {
    validate_relative_path(&contract.relative_path)?;
    let path = root.join(&contract.relative_path);
    let mut file = open_regular_file_no_follow(&path).with_context(|| {
        format!(
            "failed to open private H3 {} through the manifest path {}",
            contract.role.stable_id(),
            contract.relative_path.display()
        )
    })?;
    let opened_metadata = file.metadata()?;
    let before = FileIdentity::from_metadata(&opened_metadata);
    if before.len != contract.size_bytes {
        bail!(
            "private H3 {} has {} bytes, expected {}",
            contract.role.stable_id(),
            before.len,
            contract.size_bytes
        )
    }
    let capacity = usize::try_from(contract.size_bytes)
        .context("private H3 support file is too large to buffer on this platform")?;
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read private H3 {}", contract.role.stable_id()))?;
    file.seek(SeekFrom::Start(0))?;
    let after = FileIdentity::from_metadata(&file.metadata()?);
    if bytes.len() != capacity || before != after {
        bail!(
            "private H3 {} changed while its opened descriptor was read",
            contract.role.stable_id()
        )
    }
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != contract.sha256 {
        bail!(
            "private H3 {} SHA-256 mismatch: expected {}, found {actual_sha256}",
            contract.role.stable_id(),
            contract.sha256
        )
    }
    let current = open_regular_file_no_follow(&path).with_context(|| {
        format!(
            "failed to re-open private H3 {} after authentication",
            contract.role.stable_id()
        )
    })?;
    if FileIdentity::from_metadata(&current.metadata()?) != before {
        bail!(
            "private H3 {} manifest path changed during authentication",
            contract.role.stable_id()
        )
    }
    Ok((
        OpenedSupportArtifact {
            role: contract.role,
            path,
            file,
            identity: before,
            expected_sha256: contract.sha256.clone(),
            full_rehashes: std::sync::atomic::AtomicU64::new(0),
        },
        bytes,
    ))
}

fn support_identity(model: &str, contracts: &[SupportFileContract]) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(SUPPORT_IDENTITY_SCHEMA.as_bytes());
    digest.update([0]);
    digest.update(model.as_bytes());
    for role in SupportRole::ALL {
        let contract = contracts
            .iter()
            .find(|contract| contract.role == role)
            .ok_or_else(|| anyhow!("private H3 support identity omits {}", role.stable_id()))?;
        for field in [
            contract.role.stable_id(),
            &contract.source_repo,
            &contract.source_revision,
            &contract.source_path,
            &contract.relative_path.to_string_lossy(),
            &contract.size_bytes.to_string(),
            &contract.sha256,
        ] {
            digest.update([0]);
            digest.update(field.as_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_tokenizer_config(bytes: &[u8]) -> Result<()> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("invalid pinned H3 tokenizer config JSON")?;
    let exact = value
        .get("tokenizer_class")
        .and_then(serde_json::Value::as_str)
        == Some("Qwen2Tokenizer")
        && value
            .get("add_bos_token")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        && value
            .get("bos_token")
            .is_some_and(serde_json::Value::is_null)
        && value
            .get("model_max_length")
            .and_then(serde_json::Value::as_u64)
            == Some(262_144)
        && value.get("pad_token").and_then(serde_json::Value::as_str) == Some("<|endoftext|>")
        && value.get("eos_token").and_then(serde_json::Value::as_str) == Some("<|im_end|>");
    if !exact {
        bail!("pinned H3 tokenizer config is not the released raw Qwen2 contract")
    }
    Ok(())
}

/// The shape of `tokenizer.json` as released, before the configured extra
/// special tokens are registered. Checked first so a swapped file is still
/// refused at its own fingerprint.
fn validate_released_tokenizer_shape(tokenizer: &Tokenizer) -> Result<()> {
    let base_vocab_size = tokenizer.get_vocab_size(false);
    let tokenizer_vocab_size = tokenizer.get_vocab_size(true);
    if base_vocab_size != RELEASED_TOKENIZER_BASE_VOCAB_SIZE
        || tokenizer_vocab_size != RELEASED_TOKENIZER_VOCAB_SIZE
    {
        bail!(
            "pinned H3 tokenizer has {base_vocab_size} base and {tokenizer_vocab_size} total entries, expected {RELEASED_TOKENIZER_BASE_VOCAB_SIZE} base and {RELEASED_TOKENIZER_VOCAB_SIZE} total"
        )
    }
    let max_token_id = tokenizer.get_vocab(true).values().copied().max();
    if max_token_id != Some(RELEASED_TOKENIZER_MAX_ID) {
        bail!(
            "pinned H3 tokenizer max ID {max_token_id:?} is not the released {RELEASED_TOKENIZER_MAX_ID}"
        )
    }
    Ok(())
}

/// The shape after `register_extra_special_tokens`, which is what the runtime
/// actually encodes with.
fn validate_tokenizer(tokenizer: &Tokenizer, config: &H3ConditionerConfig) -> Result<()> {
    let base_vocab_size = tokenizer.get_vocab_size(false);
    let tokenizer_vocab_size = tokenizer.get_vocab_size(true);
    if base_vocab_size != RELEASED_TOKENIZER_BASE_VOCAB_SIZE
        || tokenizer_vocab_size != H3_REGISTERED_VOCABULARY_SIZE
    {
        bail!(
            "registered H3 tokenizer has {base_vocab_size} base and {tokenizer_vocab_size} total entries, expected {RELEASED_TOKENIZER_BASE_VOCAB_SIZE} base and {H3_REGISTERED_VOCABULARY_SIZE} total"
        )
    }
    let max_token_id = tokenizer.get_vocab(true).values().copied().max();
    if max_token_id != Some(H3_REGISTERED_MAX_TOKEN_ID)
        || usize::try_from(H3_REGISTERED_MAX_TOKEN_ID)? >= config.text_config.vocab_size
    {
        bail!(
            "registered H3 tokenizer max ID {max_token_id:?} does not fit the released {}-row embedding table",
            config.text_config.vocab_size
        )
    }
    for (token, expected) in H3_EXTRA_SPECIAL_TOKENS {
        let found = tokenizer.token_to_id(token);
        if found != Some(expected) {
            bail!(
                "registered H3 tokenizer token {token} resolves to {found:?}, expected {expected}"
            )
        }
    }
    for (token, expected) in [
        ("<|vision_start|>", config.vision_start_token_id),
        ("<|vision_end|>", config.vision_end_token_id),
        ("<|image_pad|>", config.image_token_id),
        ("<|video_pad|>", config.video_token_id),
    ] {
        let found = tokenizer.token_to_id(token);
        if found != Some(expected) {
            bail!("pinned H3 tokenizer token {token} resolves to {found:?}, expected {expected}")
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_loader_marker_matches_the_compiled_runtime_scope() {
        #[cfg(feature = "h3-private-uat")]
        assert_eq!(
            H3_PRIVATE_QWEN_SUPPORT_CLAIM_MARKER,
            "mold.minimax-h3.private-uat-qwen-support-loader.v1"
        );
        #[cfg(all(feature = "h3", not(feature = "h3-private-uat")))]
        assert_eq!(
            H3_PRIVATE_QWEN_SUPPORT_CLAIM_MARKER,
            "mold.minimax-h3.qwen-support-loader.v1"
        );
    }

    const CONDITIONER_CONFIG: &str = r#"{
      "architectures":["Qwen3VLForConditionalGeneration"],
      "image_token_id":151655,"video_token_id":151656,
      "vision_start_token_id":151652,"vision_end_token_id":151653,
      "text_config":{"model_type":"qwen3_vl_text","vocab_size":151936,
        "hidden_size":5120,"intermediate_size":25600,"num_hidden_layers":64,
        "num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,
        "rms_norm_eps":0.000001,"rope_theta":5000000.0,
        "max_position_embeddings":262144,"hidden_act":"silu","attention_bias":false,
        "rope_scaling":{"rope_type":"default","mrope_interleaved":true,
          "mrope_section":[24,20,20]}},
      "vision_config":{"model_type":"qwen3_vl","depth":27,"hidden_size":1152,
        "intermediate_size":4304,"num_heads":16,"in_channels":3,"patch_size":16,
        "temporal_patch_size":2,"spatial_merge_size":2,"out_hidden_size":5120,
        "num_position_embeddings":2304,"deepstack_visual_indexes":[8,16,24],
        "hidden_act":"gelu_pytorch_tanh"}
    }"#;
    const TOKENIZER_CONFIG: &str = r#"{
      "tokenizer_class":"Qwen2Tokenizer","add_bos_token":false,"bos_token":null,
      "model_max_length":262144,"pad_token":"<|endoftext|>","eos_token":"<|im_end|>",
      "additional_special_tokens":[
        "<|im_start|>","<|im_end|>","<|object_ref_start|>","<|object_ref_end|>",
        "<|box_start|>","<|box_end|>","<|quad_start|>","<|quad_end|>",
        "<|vision_start|>","<|vision_end|>","<|vision_pad|>","<|image_pad|>",
        "<|video_pad|>","<d>","</d>","<|cutoff|>","<|lyrics_start|>",
        "<|lyrics_end|>","<|caption_start|>","<|caption_end|>"
      ]
    }"#;
    const IMAGE_PROCESSOR: &str = r#"{
      "size":{"shortest_edge":65536,"longest_edge":16777216},
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
      "processor_class":"Qwen3VLProcessor",
      "image_processor_type":"Qwen2VLImageProcessorFast"
    }"#;
    const VIDEO_PROCESSOR: &str = r#"{
      "size":{"shortest_edge":4096,"longest_edge":25165824},
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
      "processor_class":"Qwen3VLProcessor",
      "video_processor_type":"Qwen3VLVideoProcessor"
    }"#;

    /// The released `processor/tokenizer.json` added-token block, verbatim. The
    /// synthetic fixture mirrors it so "which of the configured special tokens
    /// are missing from the vocabulary" has the same answer here as on disk.
    const RELEASED_ADDED_TOKEN_CONTENTS: &[(u32, &str)] = &[
        (151_643, "<|endoftext|>"),
        (151_644, "<|im_start|>"),
        (151_645, "<|im_end|>"),
        (151_646, "<|object_ref_start|>"),
        (151_647, "<|object_ref_end|>"),
        (151_648, "<|box_start|>"),
        (151_649, "<|box_end|>"),
        (151_650, "<|quad_start|>"),
        (151_651, "<|quad_end|>"),
        (151_652, "<|vision_start|>"),
        (151_653, "<|vision_end|>"),
        (151_654, "<|vision_pad|>"),
        (151_655, "<|image_pad|>"),
        (151_656, "<|video_pad|>"),
        (151_657, "<tool_call>"),
        (151_658, "</tool_call>"),
        (151_659, "<|fim_prefix|>"),
        (151_660, "<|fim_middle|>"),
        (151_661, "<|fim_suffix|>"),
        (151_662, "<|fim_pad|>"),
        (151_663, "<|repo_name|>"),
        (151_664, "<|file_sep|>"),
        (151_665, "<tool_response>"),
        (151_666, "</tool_response>"),
        (151_667, "<think>"),
        (151_668, "</think>"),
    ];

    fn synthetic_tokenizer() -> Vec<u8> {
        let mut vocab = serde_json::Map::with_capacity(RELEASED_TOKENIZER_BASE_VOCAB_SIZE);
        for id in 0..u32::try_from(RELEASED_TOKENIZER_BASE_VOCAB_SIZE).unwrap() {
            vocab.insert(format!("token-{id}"), id.into());
        }
        let added_tokens = (u32::try_from(RELEASED_TOKENIZER_BASE_VOCAB_SIZE).unwrap()
            ..=RELEASED_TOKENIZER_MAX_ID)
            .map(|id| {
                let content = RELEASED_ADDED_TOKEN_CONTENTS
                    .iter()
                    .find(|(released_id, _)| *released_id == id)
                    .map(|(_, content)| (*content).to_owned())
                    .unwrap_or_else(|| format!("<|added_{id}|>"));
                serde_json::json!({
                    "id": id,
                    "content": content,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                })
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "token-0"
            }
        }))
        .unwrap()
    }

    fn synthetic_contracts(root: &Path) -> Vec<SupportFileContract> {
        let tokenizer = synthetic_tokenizer();
        let fixtures = [
            (
                SupportRole::ArchitectureConfig,
                CONDITIONER_CONFIG.as_bytes(),
            ),
            (SupportRole::Tokenizer, tokenizer.as_slice()),
            (SupportRole::TokenizerConfig, TOKENIZER_CONFIG.as_bytes()),
            (
                SupportRole::ImageProcessorConfig,
                IMAGE_PROCESSOR.as_bytes(),
            ),
            (
                SupportRole::VideoProcessorConfig,
                VIDEO_PROCESSOR.as_bytes(),
            ),
        ];
        fixtures
            .into_iter()
            .map(|(role, bytes)| {
                let relative_path = PathBuf::from("shared")
                    .join(contract::FAMILY)
                    .join(role.source_path());
                let path = root.join(&relative_path);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, bytes).unwrap();
                SupportFileContract {
                    role,
                    relative_path,
                    source_repo: contract::OFFICIAL_REPO.into(),
                    source_revision: contract::OFFICIAL_REVISION.into(),
                    source_path: role.source_path().into(),
                    size_bytes: bytes.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                }
            })
            .collect()
    }

    #[test]
    fn exact_hidden_comfy_manifests_resolve_the_same_qwen_support_set() {
        let fl2va = production_support_contracts(contract::FL2VA_COMFY).unwrap();
        let ref2va = production_support_contracts(contract::REF2VA_COMFY).unwrap();
        assert_eq!(fl2va, ref2va);
        assert_eq!(fl2va.len(), SupportRole::ALL.len());
        for contract in fl2va {
            assert_eq!(contract.source_repo, contract::OFFICIAL_REPO);
            assert!(contract
                .relative_path
                .starts_with(Path::new("shared").join(contract::FAMILY)));
        }
        assert!(production_support_contracts(contract::FL2VA_OFFICIAL).is_err());
    }

    /// MiniMax declares seven extra special tokens in `tokenizer_config.json`'s
    /// `additional_special_tokens` that carry no id in `tokenizer.json`; the
    /// released tokenizer stops at 151668 and HuggingFace assigns them
    /// 151669..=151675 at load time. Without that step `<d>` reaches Qwen as
    /// byte-level BPE pieces and no token delimits the dialogue span.
    /// See the official README ("we add several special tokens, such as `<d>`")
    /// and issue #1430.
    #[cfg(unix)]
    #[test]
    fn official_dialogue_special_tokens_register_at_their_released_ids() {
        let root = tempfile::tempdir().unwrap();
        let contracts = synthetic_contracts(root.path());
        let support =
            load_support_from_contracts(root.path(), contract::FL2VA_COMFY, contracts).unwrap();
        let tokenizer = support.tokenizer();

        for (content, expected) in [
            ("<d>", 151_669_u32),
            ("</d>", 151_670),
            ("<|cutoff|>", 151_671),
            ("<|lyrics_start|>", 151_672),
            ("<|lyrics_end|>", 151_673),
            ("<|caption_start|>", 151_674),
            ("<|caption_end|>", 151_675),
        ] {
            assert_eq!(
                tokenizer.token_to_id(content),
                Some(expected),
                "{content} must resolve to the released id {expected}"
            );
        }

        // The presentation encodes with `add_special_tokens = false`; that flag
        // governs BOS/EOS wrapping only, so the added-vocabulary trie must still
        // collapse a dialogue tag to one id inside ordinary prompt text.
        let encoded = tokenizer
            .encode("token-5 <d>token-7</d> token-9", false)
            .unwrap();
        assert!(
            encoded.get_ids().contains(&151_669) && encoded.get_ids().contains(&151_670),
            "dialogue tags must survive as single ids, got {:?}",
            encoded.get_ids()
        );
    }

    #[cfg(unix)]
    #[test]
    fn synthetic_support_load_is_parsed_and_bound_to_opened_files() {
        let root = tempfile::tempdir().unwrap();
        let mut contracts = synthetic_contracts(root.path());
        let expected_identity = support_identity(contract::FL2VA_COMFY, &contracts).unwrap();
        contracts.reverse();
        assert_eq!(
            support_identity(contract::FL2VA_COMFY, &contracts).unwrap(),
            expected_identity
        );
        let tokenizer_relative = contracts
            .iter()
            .find(|contract| contract.role == SupportRole::Tokenizer)
            .unwrap()
            .relative_path
            .clone();
        let support =
            load_support_from_contracts(root.path(), contract::FL2VA_COMFY, contracts).unwrap();
        assert_eq!(support.model(), contract::FL2VA_COMFY);
        assert_eq!(support.conditioner_config().text_config.hidden_size, 5_120);
        assert_eq!(
            support.tokenizer().get_vocab_size(true),
            H3_REGISTERED_VOCABULARY_SIZE
        );
        assert_eq!(support.support_identity_sha256().len(), 64);
        support.revalidate().unwrap();

        let tokenizer_path = root.path().join(tokenizer_relative);
        let original_path = tokenizer_path.with_extension("authenticated.json");
        std::fs::rename(&tokenizer_path, &original_path).unwrap();
        std::fs::copy(&original_path, &tokenizer_path).unwrap();
        let error = support.revalidate().unwrap_err();
        assert!(error.to_string().contains("private H3 tokenizer"));
    }

    #[cfg(unix)]
    #[test]
    fn revalidate_fully_rehashes_each_descriptor_exactly_once() {
        let root = tempfile::tempdir().unwrap();
        let contracts = synthetic_contracts(root.path());
        let tokenizer_relative = contracts
            .iter()
            .find(|contract| contract.role == SupportRole::Tokenizer)
            .unwrap()
            .relative_path
            .clone();
        let support =
            load_support_from_contracts(root.path(), contract::FL2VA_COMFY, contracts).unwrap();
        // Loading already ran the one full revalidation re-hash per artifact.
        let per_artifact_once = SupportRole::ALL.len() as u64;
        assert_eq!(support.full_rehash_count(), per_artifact_once);
        support.revalidate().unwrap();
        support.revalidate().unwrap();
        assert_eq!(support.full_rehash_count(), per_artifact_once);

        // The fast path still fails closed when the manifest path stops naming
        // the authenticated inode.
        let tokenizer_path = root.path().join(tokenizer_relative);
        let original_path = tokenizer_path.with_extension("swapped.json");
        std::fs::rename(&tokenizer_path, &original_path).unwrap();
        std::fs::copy(&original_path, &tokenizer_path).unwrap();
        let error = support.revalidate().unwrap_err();
        assert!(error.to_string().contains("private H3 tokenizer"));
    }

    #[test]
    fn support_file_hash_and_contract_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = PathBuf::from("support/tokenizer.json");
        std::fs::create_dir_all(root.path().join("support")).unwrap();
        std::fs::write(root.path().join(&relative_path), b"authenticated").unwrap();
        let contract = SupportFileContract {
            role: SupportRole::Tokenizer,
            relative_path,
            source_repo: contract::OFFICIAL_REPO.into(),
            source_revision: contract::OFFICIAL_REVISION.into(),
            source_path: SupportRole::Tokenizer.source_path().into(),
            size_bytes: 13,
            sha256: "0".repeat(64),
        };
        let root = root.path().canonicalize().unwrap();
        let error = authenticate_support_file(&root, &contract).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));

        let mut duplicate = vec![contract; SupportRole::ALL.len()];
        duplicate[0].sha256 = format!("{:x}", Sha256::digest(b"authenticated"));
        let error = validate_contract_set(&duplicate).unwrap_err();
        assert!(error.to_string().contains("repeats tokenizer"));
    }

    #[test]
    fn tokenizer_config_rejects_implicit_special_token_changes() {
        validate_tokenizer_config(TOKENIZER_CONFIG.as_bytes()).unwrap();
        let wrong = TOKENIZER_CONFIG.replace("false", "true");
        assert!(validate_tokenizer_config(wrong.as_bytes()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn models_root_swap_to_symlink_fails_at_the_validated_boundary() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        let models_root = workspace.path().join("models");
        std::fs::create_dir(&models_root).unwrap();
        let contracts = synthetic_contracts(&models_root);
        let authenticated_root = workspace.path().join("authenticated-models");
        let replacement_root = workspace.path().join("replacement-models");
        std::fs::create_dir(&replacement_root).unwrap();
        let result = load_support_from_contracts_at_boundary(
            &models_root,
            contract::FL2VA_COMFY,
            contracts,
            |validated_root| {
                std::fs::rename(validated_root, &authenticated_root)?;
                symlink(&replacement_root, validated_root)?;
                Ok(())
            },
        );
        let error = match result {
            Ok(_) => panic!("models-root replacement must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("models root changed"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn group_writable_models_root_is_accepted() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let contracts = synthetic_contracts(root.path());
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o770)).unwrap();

        load_support_from_contracts(root.path(), contract::FL2VA_COMFY, contracts).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn group_writable_support_file_is_accepted() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let contracts = synthetic_contracts(root.path());
        let tokenizer = contracts
            .iter()
            .find(|contract| contract.role == SupportRole::Tokenizer)
            .unwrap();
        std::fs::set_permissions(
            root.path().join(&tokenizer.relative_path),
            std::fs::Permissions::from_mode(0o666),
        )
        .unwrap();

        load_support_from_contracts(root.path(), contract::FL2VA_COMFY, contracts).unwrap();
    }

    #[test]
    #[ignore = "requires the authorized private H3 support artifacts"]
    fn authorized_private_support_loads_for_both_comfy_tasks() {
        let models_root = std::env::var_os("MOLD_H3_MODELS_ROOT")
            .map(PathBuf::from)
            .expect("set MOLD_H3_MODELS_ROOT to the authorized private models root");
        let mut identities = Vec::new();
        for model in [contract::FL2VA_COMFY, contract::REF2VA_COMFY] {
            let support = load_qualified_private_qwen_support(&models_root, model).unwrap();
            assert_eq!(support.model(), model);
            assert_eq!(support.conditioner_config().text_config.hidden_size, 5_120);
            assert_eq!(
                support.tokenizer().get_vocab_size(true),
                H3_REGISTERED_VOCABULARY_SIZE
            );
            // The released file itself must still be the released file.
            assert_eq!(
                support.tokenizer().get_vocab_size(false),
                RELEASED_TOKENIZER_BASE_VOCAB_SIZE
            );
            for (token, expected) in H3_EXTRA_SPECIAL_TOKENS {
                assert_eq!(
                    support.tokenizer().token_to_id(token),
                    Some(expected),
                    "{token} must resolve to {expected} on the real released tokenizer"
                );
            }
            assert_eq!(support.support_identity_sha256().len(), 64);
            support.revalidate().unwrap();
            identities.push(support.support_identity_sha256().to_owned());
        }
        assert_ne!(
            identities[0], identities[1],
            "task-specific private authorities must not share one runtime identity"
        );
    }
}
