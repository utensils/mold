//! Authorization-bound MiniMax H3 artifact qualification.
//!
//! This module is compiled only by the `h3-private-uat` development feature.
//! It authenticates and structurally inspects the exact private-host artifact
//! set without registering a model, constructing an engine, or changing the
//! ordinary product policy. Its claim marker is forbidden in published Mold
//! binaries by `scripts/verify-h3-release-exclusion.sh`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use mold_candle::minimax_h3::{
    inspect_h3_comfy_published_header, inspect_h3_qwen_nvfp4_awq_header,
    validate_audio_safetensors, validate_comfy_weight_file, validate_processor_assets,
    AudioTensorLayout, AudioVaeConfig, H3ComfyPublishedArtifact, H3ConditionerConfig,
    H3TransformerTask, MiniMaxH3VisualVaeConfig, H3_QWEN_NVFP4_AWQ_FILENAME,
    H3_QWEN_NVFP4_AWQ_FILE_BYTES, H3_QWEN_NVFP4_AWQ_HEADER_BYTES, H3_QWEN_NVFP4_AWQ_POLICY_SHA256,
    H3_QWEN_NVFP4_AWQ_SHA256, H3_QWEN_NVFP4_AWQ_STABLE_ID, H3_QWEN_NVFP4_AWQ_TENSOR_COUNT,
};
use mold_core::manifest::{find_manifest, storage_path, ModelComponent, ModelFile, ModelManifest};
use mold_core::minimax_h3::{self as contract, Task};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};

pub const H3_PRIVATE_UAT_CLAIM_MARKER: &str = "mold.minimax-h3.private-uat-artifact-reader.v1";
pub const H3_PRIVATE_AUTHORIZATION_SCOPE: &str = "private-h3-uat";

const MAX_PRIVATE_HEADER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_AUTHORIZATION_RECORD_BYTES: u64 = 64 * 1024;
const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";
const REVIEWED_AUTHORIZATION_EVIDENCE_SHA256: &str =
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d";
const QUALIFICATION_SCHEMA: &str = "mold.minimax-h3.private-artifact-qualification.v2";
const REQUIRED_AUTHORIZATION_SCOPES: [&str; 3] = [
    "checkpoint-execution",
    "fixture-capture",
    "generated-output-retention",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct H3ArtifactHashProgress {
    pub relative_path: String,
    pub artifact_bytes_verified: u64,
    pub artifact_bytes_total: u64,
    pub total_bytes_verified: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct H3QualifiedArtifact {
    pub relative_path: String,
    pub component: &'static str,
    pub source_revision: &'static str,
    pub size_bytes: u64,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header_identity_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
    pub structural_contract: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_identity_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct H3PrivateArtifactQualificationReport {
    pub schema: &'static str,
    pub claim_marker: &'static str,
    pub decision: &'static str,
    pub authorization_scope: &'static str,
    pub authorization_schema: &'static str,
    pub authorization_record_sha256: String,
    pub authorization_source_document_sha256: String,
    pub authorization_review_reference: String,
    pub canonical_model: String,
    pub task: &'static str,
    pub layout: &'static str,
    pub official_model_revision: &'static str,
    pub comfy_model_revision: &'static str,
    pub official_implementation_revision: &'static str,
    pub comfy_implementation_revision: &'static str,
    pub artifacts: Vec<H3QualifiedArtifact>,
    pub total_bytes: u64,
    pub qualification_identity_sha256: String,
    pub runtime_constructed: bool,
    pub generated_media: bool,
    pub public_activation: &'static str,
    pub remaining_release_requirements: Vec<&'static str>,
}

#[derive(Debug)]
struct StructuralInspection {
    contract: &'static str,
    header_identity_sha256: Option<String>,
    tensor_count: Option<usize>,
    policy_identity_sha256: Option<String>,
}

#[derive(Clone, Debug)]
struct ResolvedArtifact<'a> {
    file: &'a ModelFile,
    relative_path: PathBuf,
    canonical_path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct H3PrivateAuthorizationRecord {
    schema_version: String,
    family: String,
    decision: String,
    license_revision: String,
    license_sha256: String,
    approved_scopes: Vec<String>,
    source_document_path: PathBuf,
    source_document_sha256: String,
    review_reference: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProtectedPathKind {
    Directory,
    File,
}

#[derive(Clone, Debug)]
struct ProtectedPath {
    path: PathBuf,
    kind: ProtectedPathKind,
    identity: FileIdentity,
}

#[derive(Debug)]
struct ValidatedPrivateScope {
    models_root: PathBuf,
    authorization_record_sha256: String,
    authorization_source_document_sha256: String,
    authorization_review_reference: String,
    protected_paths: Vec<ProtectedPath>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ReleasedTransformerConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    num_attention_heads: usize,
    attention_head_dim: usize,
    hidden_size: usize,
    num_layers: usize,
    num_refiner_layers: usize,
    ffn_dim: usize,
    in_channels: usize,
    audio_in_channels: usize,
    patch_size: [usize; 3],
    text_dim: usize,
    freq_dim: usize,
    time_embed_hidden_dim: usize,
    time_embed_dim: usize,
    rope_freq_dim: usize,
    rope_theta: f64,
    norm_eps: f64,
    qk_norm_eps: f64,
    final_norm_eps: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleasedSchedulerConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    shift: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    owner: u32,
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
            mode: metadata.mode(),
            #[cfg(unix)]
            owner: metadata.uid(),
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

/// Authenticate and structurally inspect the exact Comfy deployment set.
///
/// The owner-only campaign layout, external authorization record, hidden
/// manifest, file sizes, full SHA-256 digests, and bounded headers are all
/// mandatory. This function never builds an inference engine and cannot change
/// `runtime_available`.
pub fn qualify_private_artifacts(
    models_root: &Path,
    model: &str,
    authorization_record: &Path,
    mut progress: impl FnMut(H3ArtifactHashProgress),
) -> Result<H3PrivateArtifactQualificationReport> {
    qualify_private_artifacts_with_control(models_root, model, authorization_record, |event| {
        progress(event);
        Ok(())
    })
}

/// Cancellable counterpart used by the singular private server attempt.
/// Returning an error from `progress` stops before the next bounded artifact
/// read, so preparation cannot outlive its scheduler cancellation scope.
pub(crate) fn qualify_private_artifacts_with_control(
    models_root: &Path,
    model: &str,
    authorization_record: &Path,
    mut progress: impl FnMut(H3ArtifactHashProgress) -> Result<()>,
) -> Result<H3PrivateArtifactQualificationReport> {
    let private_scope = validate_private_scope(models_root, authorization_record)?;
    let (manifest, task, published_transformer) = qualification_manifest(model)?;
    let root = private_scope.models_root.clone();
    let artifacts = manifest
        .files
        .iter()
        .map(|file| resolve_artifact(&root, manifest, file))
        .collect::<Result<Vec<_>>>()?;
    let total_bytes = artifacts.iter().try_fold(0_u64, |total, artifact| {
        total
            .checked_add(artifact.file.size_bytes)
            .ok_or_else(|| anyhow!("private H3 artifact byte count overflow"))
    })?;

    let mut verified_before = 0_u64;
    let mut authenticated_identities = BTreeMap::new();
    let mut qualified = Vec::with_capacity(artifacts.len());
    for artifact in &artifacts {
        let relative_path = portable_path(&artifact.relative_path)?;
        let expected_sha = artifact.file.sha256.ok_or_else(|| {
            anyhow!("private H3 manifest file {relative_path} has no pinned SHA-256")
        })?;
        let (actual_sha, authenticated_identity) = hash_exact_file(
            artifact,
            expected_sha,
            verified_before,
            total_bytes,
            &mut progress,
        )?;
        let structural = inspect_structure(artifact, task, published_transformer)?;
        require_artifact_identity(artifact, &authenticated_identity, "structural inspection")?;
        authenticated_identities.insert(artifact.relative_path.clone(), authenticated_identity);
        qualified.push(H3QualifiedArtifact {
            relative_path,
            component: component_id(artifact.file.component),
            source_revision: contract::repo_revision(&artifact.file.hf_repo).ok_or_else(|| {
                anyhow!(
                    "private H3 artifact source {} has no pinned revision",
                    artifact.file.hf_repo
                )
            })?,
            size_bytes: artifact.file.size_bytes,
            sha256: actual_sha,
            header_identity_sha256: structural.header_identity_sha256,
            tensor_count: structural.tensor_count,
            structural_contract: structural.contract,
            policy_identity_sha256: structural.policy_identity_sha256,
        });
        verified_before = verified_before
            .checked_add(artifact.file.size_bytes)
            .ok_or_else(|| anyhow!("private H3 verified byte count overflow"))?;
    }
    revalidate_artifact_set(&artifacts, &authenticated_identities)?;
    validate_support_assets(&artifacts, &authenticated_identities, task)?;
    revalidate_artifact_set(&artifacts, &authenticated_identities)?;
    revalidate_private_scope(&private_scope)?;

    let qualification_identity_sha256 = qualification_identity(
        model,
        task,
        &private_scope.authorization_record_sha256,
        &private_scope.authorization_source_document_sha256,
        &qualified,
    );
    Ok(H3PrivateArtifactQualificationReport {
        schema: QUALIFICATION_SCHEMA,
        claim_marker: H3_PRIVATE_UAT_CLAIM_MARKER,
        decision: "qualified-private-artifacts",
        authorization_scope: H3_PRIVATE_AUTHORIZATION_SCOPE,
        authorization_schema: AUTHORIZATION_SCHEMA,
        authorization_record_sha256: private_scope.authorization_record_sha256,
        authorization_source_document_sha256: private_scope.authorization_source_document_sha256,
        authorization_review_reference: private_scope.authorization_review_reference,
        canonical_model: model.to_string(),
        task: task_id(task),
        layout: "comfy-pruned-int8-convrot-plus-nvfp4-awq",
        official_model_revision: contract::OFFICIAL_REVISION,
        comfy_model_revision: contract::COMFY_REVISION,
        official_implementation_revision: contract::OFFICIAL_IMPLEMENTATION_REVISION,
        comfy_implementation_revision: contract::COMFY_IMPLEMENTATION_REVISION,
        artifacts: qualified,
        total_bytes,
        qualification_identity_sha256,
        runtime_constructed: false,
        generated_media: false,
        public_activation: "rejected",
        remaining_release_requirements: vec![
            "real-numerical-conformance",
            "private-cuda-runtime-qualification",
            "public-authorization-expansion",
            "shipping-runtime-review",
        ],
    })
}

fn validate_private_scope(
    models_root: &Path,
    authorization_record: &Path,
) -> Result<ValidatedPrivateScope> {
    validate_private_scope_against_evidence(
        models_root,
        authorization_record,
        REVIEWED_AUTHORIZATION_EVIDENCE_SHA256,
    )
}

fn validate_private_scope_against_evidence(
    models_root: &Path,
    authorization_record: &Path,
    reviewed_evidence_sha256: &str,
) -> Result<ValidatedPrivateScope> {
    let models_root = canonical_absolute_path(models_root, "private H3 models root")?;
    if models_root.file_name().and_then(|name| name.to_str()) != Some("models") {
        bail!("private H3 models root must be the campaign mold-home/models directory")
    }
    let mold_home = models_root
        .parent()
        .ok_or_else(|| anyhow!("private H3 models root has no Mold home parent"))?;
    if mold_home.file_name().and_then(|name| name.to_str()) != Some("mold-home") {
        bail!("private H3 models root must be nested directly under campaign mold-home")
    }
    let campaign_root = mold_home
        .parent()
        .ok_or_else(|| anyhow!("private H3 Mold home has no campaign parent"))?;
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("cannot resolve the Mold repository root"))?
        .canonicalize()
        .context("failed to canonicalize the Mold repository root")?;
    if campaign_root.starts_with(&repository_root) {
        bail!("private H3 qualification campaign must live outside the Mold repository")
    }
    let compliance_root = campaign_root.join("compliance");
    let authorization_record =
        canonical_absolute_path(authorization_record, "private H3 authorization record")?;
    if authorization_record.parent() != Some(compliance_root.as_path()) {
        bail!("private H3 authorization record must be a direct child of campaign compliance")
    }

    let mut protected_paths = Vec::new();
    let campaign = require_private_directory(campaign_root, None)?;
    let owner = campaign.owner();
    require_effective_process_owner(owner)?;
    protected_paths.push(campaign);
    protected_paths.push(require_private_directory(mold_home, Some(owner))?);
    protected_paths.push(require_private_directory(&models_root, Some(owner))?);
    protected_paths.push(require_private_directory(&compliance_root, Some(owner))?);
    let record_path = require_private_file(&authorization_record, owner)?;
    let record_bytes = read_protected_file(
        &record_path,
        MAX_AUTHORIZATION_RECORD_BYTES,
        "private H3 authorization record",
    )?;
    let authorization_record_sha256 = format!("{:x}", Sha256::digest(&record_bytes));
    let record: H3PrivateAuthorizationRecord = serde_json::from_slice(&record_bytes)
        .context("private H3 authorization record is not valid exact-schema JSON")?;
    validate_authorization_fields(&record, reviewed_evidence_sha256)?;

    let source_document = canonical_absolute_path(
        &record.source_document_path,
        "private H3 authorization source document",
    )?;
    if source_document == authorization_record
        || source_document.parent() != Some(compliance_root.as_path())
    {
        bail!(
            "private H3 authorization source document must be a distinct direct child of campaign compliance"
        )
    }
    let source_path = require_private_file(&source_document, owner)?;
    let source_document_sha256 =
        hash_protected_file(&source_path, "private H3 authorization source document")?;
    if source_document_sha256 != record.source_document_sha256 {
        bail!("private H3 authorization source document hash does not match its record")
    }

    protected_paths.push(record_path);
    protected_paths.push(source_path);
    Ok(ValidatedPrivateScope {
        models_root,
        authorization_record_sha256,
        authorization_source_document_sha256: source_document_sha256,
        authorization_review_reference: record.review_reference,
        protected_paths,
    })
}

fn validate_authorization_fields(
    record: &H3PrivateAuthorizationRecord,
    reviewed_evidence_sha256: &str,
) -> Result<()> {
    if !valid_lower_sha256(reviewed_evidence_sha256) {
        bail!("private H3 reviewed authorization evidence identity is invalid")
    }
    if record.schema_version != AUTHORIZATION_SCHEMA {
        bail!("private H3 authorization record schema is unsupported")
    }
    if record.family != contract::FAMILY || record.decision != "approved" {
        bail!("private H3 authorization record does not approve MiniMax H3")
    }
    if record.license_revision != contract::OFFICIAL_REVISION {
        bail!("private H3 authorization record covers a different license revision")
    }
    if record.license_sha256 != contract::LICENSE_SHA256 {
        bail!("private H3 authorization record covers different license bytes")
    }
    let actual_scopes = record
        .approved_scopes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected_scopes = REQUIRED_AUTHORIZATION_SCOPES
        .into_iter()
        .collect::<BTreeSet<_>>();
    if actual_scopes.len() != record.approved_scopes.len()
        || !expected_scopes.is_subset(&actual_scopes)
        || actual_scopes.iter().any(|scope| {
            scope.is_empty()
                || scope.len() > 128
                || scope.trim() != *scope
                || scope.chars().any(char::is_control)
        })
    {
        bail!("private H3 authorization record does not cover every reviewed activity")
    }
    if !valid_lower_sha256(&record.source_document_sha256) {
        bail!("private H3 authorization source document SHA-256 is not canonical")
    }
    if record.source_document_sha256 != reviewed_evidence_sha256 {
        bail!("private H3 authorization record does not bind the reviewed evidence")
    }
    let review_reference = record.review_reference.as_str();
    if review_reference.is_empty()
        || review_reference.len() > 256
        || review_reference.trim() != review_reference
        || review_reference.chars().any(char::is_control)
    {
        bail!("private H3 authorization record lacks a canonical external review reference")
    }
    Ok(())
}

fn canonical_absolute_path(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("{label} must be an absolute path")
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        bail!("{label} must not contain relative or parent components")
    }
    let requested_metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
    if requested_metadata.file_type().is_symlink() {
        bail!("{label} must not be a symlink")
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {label}"))?;
    if canonical != path {
        bail!("{label} must not contain aliases or symlink components")
    }
    Ok(canonical)
}

#[cfg(unix)]
fn require_effective_process_owner(owner: u32) -> Result<()> {
    if owner != effective_process_owner() {
        bail!("private H3 qualification campaign must be owned by the invoking process user")
    }
    Ok(())
}

#[cfg(unix)]
fn effective_process_owner() -> u32 {
    // SAFETY: `geteuid` has no preconditions, does not dereference pointers,
    // and only returns the effective uid maintained by the operating system.
    unsafe { libc::geteuid() }
}

#[cfg(not(unix))]
fn require_effective_process_owner(_owner: u32) -> Result<()> {
    bail!("private H3 qualification currently requires Unix ownership semantics")
}

fn require_private_directory(path: &Path, expected_owner: Option<u32>) -> Result<ProtectedPath> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect private directory {}", path.display()))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        bail!("private H3 qualification path must be a real directory")
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o7777 != 0o700 {
        bail!(
            "private H3 qualification directory {} must have mode 0700",
            path.display()
        )
    }
    #[cfg(unix)]
    if expected_owner.is_some_and(|owner| metadata.uid() != owner) {
        bail!("private H3 qualification directories must have one owner")
    }
    #[cfg(not(unix))]
    bail!("private H3 qualification currently requires Unix permission semantics");
    Ok(ProtectedPath {
        path: path.to_path_buf(),
        kind: ProtectedPathKind::Directory,
        identity: FileIdentity::from_metadata(&metadata),
    })
}

fn require_private_file(path: &Path, expected_owner: u32) -> Result<ProtectedPath> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect private file {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!("private H3 qualification evidence must be a regular non-symlink file")
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o7777 != 0o600 {
        bail!(
            "private H3 qualification evidence {} must have mode 0600",
            path.display()
        )
    }
    #[cfg(unix)]
    if metadata.uid() != expected_owner {
        bail!("private H3 qualification evidence must share the campaign owner")
    }
    #[cfg(not(unix))]
    bail!("private H3 qualification currently requires Unix permission semantics");
    Ok(ProtectedPath {
        path: path.to_path_buf(),
        kind: ProtectedPathKind::File,
        identity: FileIdentity::from_metadata(&metadata),
    })
}

impl ProtectedPath {
    #[cfg(unix)]
    fn owner(&self) -> u32 {
        self.identity.owner
    }

    #[cfg(not(unix))]
    fn owner(&self) -> u32 {
        0
    }
}

fn read_protected_file(path: &ProtectedPath, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let mut file = File::open(&path.path).with_context(|| format!("failed to open {label}"))?;
    if FileIdentity::from_metadata(&file.metadata()?) != path.identity {
        bail!("{label} changed while it was opened")
    }
    if path.identity.len > maximum {
        bail!("{label} exceeds the bounded reader size")
    }
    let capacity = usize::try_from(path.identity.len).context("private file is too large")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label}"))?;
    if bytes.len() != capacity || FileIdentity::from_metadata(&file.metadata()?) != path.identity {
        bail!("{label} changed during its read")
    }
    revalidate_protected_path(path)?;
    Ok(bytes)
}

fn hash_protected_file(path: &ProtectedPath, label: &str) -> Result<String> {
    let mut file = File::open(&path.path).with_context(|| format!("failed to open {label}"))?;
    if FileIdentity::from_metadata(&file.metadata()?) != path.identity {
        bail!("{label} changed while it was opened")
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    let mut read_bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {label}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        read_bytes = read_bytes
            .checked_add(read as u64)
            .ok_or_else(|| anyhow!("{label} byte count overflow"))?;
    }
    if read_bytes != path.identity.len
        || FileIdentity::from_metadata(&file.metadata()?) != path.identity
    {
        bail!("{label} changed during hashing")
    }
    revalidate_protected_path(path)?;
    Ok(format!("{:x}", digest.finalize()))
}

fn revalidate_private_scope(scope: &ValidatedPrivateScope) -> Result<()> {
    for path in &scope.protected_paths {
        revalidate_protected_path(path)?;
    }
    Ok(())
}

fn revalidate_protected_path(path: &ProtectedPath) -> Result<()> {
    let metadata = fs::symlink_metadata(&path.path)
        .with_context(|| format!("failed to revalidate private path {}", path.path.display()))?;
    let kind_matches = match path.kind {
        ProtectedPathKind::Directory => metadata.file_type().is_dir(),
        ProtectedPathKind::File => metadata.file_type().is_file(),
    };
    if !kind_matches
        || metadata.file_type().is_symlink()
        || FileIdentity::from_metadata(&metadata) != path.identity
    {
        bail!("private H3 authorization or campaign scope changed during qualification")
    }
    Ok(())
}

fn valid_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn qualification_manifest(
    model: &str,
) -> Result<(&'static ModelManifest, Task, H3ComfyPublishedArtifact)> {
    let (task, published) = match model {
        contract::FL2VA_COMFY => (
            Task::Fl2va,
            H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        ),
        contract::REF2VA_COMFY => (
            Task::Ref2va,
            H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
        ),
        _ => {
            bail!("private artifact qualification requires an exact Comfy H3 canonical model name")
        }
    };
    let manifest = find_manifest(model)
        .ok_or_else(|| anyhow!("missing hidden MiniMax H3 manifest for {model}"))?;
    if manifest.name != model || manifest.family != contract::FAMILY || !manifest.hidden {
        bail!("private H3 artifact qualification requires the exact hidden family manifest")
    }
    if contract::capabilities(task).runtime_available {
        bail!("private artifact qualification refuses an already-active public H3 runtime")
    }
    Ok((manifest, task, published))
}

fn resolve_artifact<'a>(
    root: &Path,
    manifest: &ModelManifest,
    file: &'a ModelFile,
) -> Result<ResolvedArtifact<'a>> {
    let relative_path = storage_path(manifest, file);
    if relative_path.as_os_str().is_empty()
        || relative_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        bail!("private H3 manifest produced an unsafe storage path")
    }
    let candidate = root.join(&relative_path);
    let metadata = fs::symlink_metadata(&candidate).with_context(|| {
        format!(
            "private H3 artifact is missing at {}",
            relative_path.display()
        )
    })?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!(
            "private H3 artifact {} must be a regular non-symlink file",
            relative_path.display()
        )
    }
    let canonical_path = candidate.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize private H3 artifact {}",
            relative_path.display()
        )
    })?;
    if !canonical_path.starts_with(root) || canonical_path != candidate {
        bail!(
            "private H3 artifact {} escapes or aliases the reviewed models root",
            relative_path.display()
        )
    }
    Ok(ResolvedArtifact {
        file,
        relative_path,
        canonical_path,
    })
}

fn hash_exact_file(
    artifact: &ResolvedArtifact<'_>,
    expected_sha: &str,
    verified_before: u64,
    total_bytes: u64,
    progress: &mut impl FnMut(H3ArtifactHashProgress) -> Result<()>,
) -> Result<(String, FileIdentity)> {
    let relative_path = portable_path(&artifact.relative_path)?;
    let before_path_metadata = fs::symlink_metadata(&artifact.canonical_path)
        .with_context(|| format!("failed to inspect private H3 artifact {relative_path}"))?;
    if !before_path_metadata.file_type().is_file() || before_path_metadata.file_type().is_symlink()
    {
        bail!("private H3 artifact {relative_path} is no longer a regular non-symlink file")
    }
    let before_path = FileIdentity::from_metadata(&before_path_metadata);
    let mut file = File::open(&artifact.canonical_path)
        .with_context(|| format!("failed to open private H3 artifact {relative_path}"))?;
    let before = FileIdentity::from_metadata(
        &file
            .metadata()
            .with_context(|| format!("failed to stat private H3 artifact {relative_path}"))?,
    );
    if before != before_path {
        bail!("private H3 artifact {relative_path} changed while it was opened")
    }
    if before.len != artifact.file.size_bytes {
        bail!(
            "private H3 artifact {relative_path} has {} bytes, expected {}",
            before.len,
            artifact.file.size_bytes
        )
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    let mut artifact_bytes_verified = 0_u64;
    progress(H3ArtifactHashProgress {
        relative_path: relative_path.clone(),
        artifact_bytes_verified,
        artifact_bytes_total: before.len,
        total_bytes_verified: verified_before,
        total_bytes,
    })?;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash private H3 artifact {relative_path}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        artifact_bytes_verified = artifact_bytes_verified
            .checked_add(read as u64)
            .ok_or_else(|| anyhow!("private H3 artifact hash byte count overflow"))?;
        let total_bytes_verified = verified_before
            .checked_add(artifact_bytes_verified)
            .ok_or_else(|| anyhow!("private H3 progress byte count overflow"))?;
        progress(H3ArtifactHashProgress {
            relative_path: relative_path.clone(),
            artifact_bytes_verified,
            artifact_bytes_total: before.len,
            total_bytes_verified,
            total_bytes,
        })?;
    }
    if artifact_bytes_verified != before.len {
        bail!("private H3 artifact {relative_path} changed length during hashing")
    }
    let after_open = FileIdentity::from_metadata(&file.metadata()?);
    let after_path_metadata = fs::symlink_metadata(&artifact.canonical_path)?;
    if !after_path_metadata.file_type().is_file() || after_path_metadata.file_type().is_symlink() {
        bail!("private H3 artifact {relative_path} was replaced during hashing")
    }
    let after_path = FileIdentity::from_metadata(&after_path_metadata);
    if before != after_open || before != after_path {
        bail!("private H3 artifact {relative_path} changed during hashing")
    }
    let actual = format!("{:x}", digest.finalize());
    if !actual.eq_ignore_ascii_case(expected_sha) {
        bail!(
            "private H3 artifact {relative_path} SHA-256 mismatch: expected {expected_sha}, found {actual}"
        )
    }
    Ok((actual, before))
}

fn require_artifact_identity(
    artifact: &ResolvedArtifact<'_>,
    expected: &FileIdentity,
    phase: &str,
) -> Result<()> {
    let relative_path = portable_path(&artifact.relative_path)?;
    let metadata = fs::symlink_metadata(&artifact.canonical_path).with_context(|| {
        format!("failed to revalidate private H3 artifact {relative_path} after {phase}")
    })?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        bail!("private H3 artifact {relative_path} was replaced during {phase}")
    }
    if &FileIdentity::from_metadata(&metadata) != expected {
        bail!("private H3 artifact {relative_path} changed during {phase}")
    }
    Ok(())
}

fn revalidate_artifact_set(
    artifacts: &[ResolvedArtifact<'_>],
    authenticated_identities: &BTreeMap<PathBuf, FileIdentity>,
) -> Result<()> {
    for artifact in artifacts {
        let expected = authenticated_identities
            .get(&artifact.relative_path)
            .ok_or_else(|| anyhow!("private H3 artifact set contains an unauthenticated path"))?;
        require_artifact_identity(artifact, expected, "qualification")?;
    }
    Ok(())
}

fn read_authenticated_artifact(
    artifact: &ResolvedArtifact<'_>,
    expected: &FileIdentity,
) -> Result<Vec<u8>> {
    require_artifact_identity(artifact, expected, "support-asset read")?;
    let relative_path = portable_path(&artifact.relative_path)?;
    let mut file = File::open(&artifact.canonical_path)
        .with_context(|| format!("failed to open private H3 support asset {relative_path}"))?;
    if FileIdentity::from_metadata(&file.metadata()?) != *expected {
        bail!("private H3 support asset {relative_path} changed while it was opened")
    }
    let capacity = usize::try_from(expected.len)
        .context("private H3 support asset is too large to buffer on this platform")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read private H3 support asset {relative_path}"))?;
    if bytes.len() != capacity || FileIdentity::from_metadata(&file.metadata()?) != *expected {
        bail!("private H3 support asset {relative_path} changed during its read")
    }
    require_artifact_identity(artifact, expected, "support-asset read")?;
    let expected_sha = artifact
        .file
        .sha256
        .ok_or_else(|| anyhow!("private H3 support asset has no pinned SHA-256"))?;
    let actual_sha = format!("{:x}", Sha256::digest(&bytes));
    if !actual_sha.eq_ignore_ascii_case(expected_sha) {
        bail!("private H3 support asset {relative_path} failed its post-read digest fence")
    }
    Ok(bytes)
}

fn inspect_structure(
    artifact: &ResolvedArtifact<'_>,
    task: Task,
    published_transformer: H3ComfyPublishedArtifact,
) -> Result<StructuralInspection> {
    match artifact.file.component {
        ModelComponent::Transformer => {
            let candidate = inspect_h3_comfy_published_header(
                &artifact.canonical_path,
                candle_task(task),
                published_transformer,
            )
            .map_err(|error| anyhow!(error))?;
            if !candidate
                .expected_content_sha256
                .eq_ignore_ascii_case(artifact.file.sha256.unwrap_or_default())
            {
                bail!("H3 transformer header authority disagrees with the manifest digest")
            }
            Ok(StructuralInspection {
                contract: "comfy-pruned-int8-convrot-header-v1",
                header_identity_sha256: Some(candidate.header_identity_sha256),
                tensor_count: Some(candidate.tensor_count),
                policy_identity_sha256: Some(candidate.strategy.quantization_policy.policy_sha256),
            })
        }
        ModelComponent::TextEncoder => inspect_qwen_nvfp4_awq(artifact),
        ModelComponent::Vae => {
            let validated = validate_comfy_weight_file(
                &MiniMaxH3VisualVaeConfig::default(),
                &artifact.canonical_path,
            )?;
            Ok(StructuralInspection {
                contract: "comfy-visual-vae-fp16-v1",
                header_identity_sha256: Some(validated.inspection().header_identity_sha256.clone()),
                tensor_count: Some(validated.inspection().tensor_count),
                policy_identity_sha256: None,
            })
        }
        ModelComponent::AudioVae => {
            let validated = validate_audio_safetensors(
                &artifact.canonical_path,
                &AudioVaeConfig::default(),
                AudioTensorLayout::ComfyFolded,
            )?;
            Ok(StructuralInspection {
                contract: "comfy-audio-vae-fp32-folded-v1",
                header_identity_sha256: Some(hash_safetensors_header(&artifact.canonical_path)?),
                tensor_count: Some(validated.tensor_count),
                policy_identity_sha256: None,
            })
        }
        ModelComponent::TaskConfig => Ok(StructuralInspection {
            contract: "official-task-config-pinned-v1",
            header_identity_sha256: None,
            tensor_count: None,
            policy_identity_sha256: None,
        }),
        ModelComponent::ModelConfig => Ok(StructuralInspection {
            contract: "official-component-config-pinned-v1",
            header_identity_sha256: None,
            tensor_count: None,
            policy_identity_sha256: None,
        }),
        ModelComponent::Processor => Ok(StructuralInspection {
            contract: "official-processor-asset-pinned-v1",
            header_identity_sha256: None,
            tensor_count: None,
            policy_identity_sha256: None,
        }),
        ModelComponent::VideoScheduler | ModelComponent::AudioScheduler => {
            Ok(StructuralInspection {
                contract: "official-dual-scheduler-config-pinned-v1",
                header_identity_sha256: None,
                tensor_count: None,
                policy_identity_sha256: None,
            })
        }
        other => bail!("unexpected private H3 manifest component {other:?}"),
    }
}

fn inspect_qwen_nvfp4_awq(artifact: &ResolvedArtifact<'_>) -> Result<StructuralInspection> {
    if !qwen_nvfp4_awq_filename_is_canonical(&artifact.file.hf_filename, &artifact.canonical_path) {
        bail!("private H3 Qwen NVFP4-AWQ artifact has a non-canonical filename")
    }
    if artifact.file.size_bytes != H3_QWEN_NVFP4_AWQ_FILE_BYTES
        || artifact.file.sha256 != Some(H3_QWEN_NVFP4_AWQ_SHA256)
    {
        bail!("private H3 Qwen NVFP4-AWQ manifest identity differs from the strict published contract")
    }
    let inspection = inspect_h3_qwen_nvfp4_awq_header(&artifact.canonical_path)
        .map_err(|error| anyhow!(error))?;
    if inspection.stable_id != H3_QWEN_NVFP4_AWQ_STABLE_ID
        || inspection.expected_artifact_sha256 != H3_QWEN_NVFP4_AWQ_SHA256
        || inspection.artifact_file_bytes != H3_QWEN_NVFP4_AWQ_FILE_BYTES
        || inspection.header_bytes != H3_QWEN_NVFP4_AWQ_HEADER_BYTES
        || inspection.tensor_count != H3_QWEN_NVFP4_AWQ_TENSOR_COUNT
        || inspection.policy_sha256 != H3_QWEN_NVFP4_AWQ_POLICY_SHA256
    {
        bail!("private H3 Qwen NVFP4-AWQ inspection differs from the strict published contract")
    }
    Ok(StructuralInspection {
        contract: H3_QWEN_NVFP4_AWQ_STABLE_ID,
        header_identity_sha256: Some(inspection.header_identity_sha256),
        tensor_count: Some(inspection.tensor_count),
        policy_identity_sha256: Some(inspection.policy_sha256),
    })
}

fn qwen_nvfp4_awq_filename_is_canonical(hf_filename: &str, canonical_path: &Path) -> bool {
    hf_filename == format!("text_encoders/{H3_QWEN_NVFP4_AWQ_FILENAME}")
        && canonical_path.file_name().and_then(|name| name.to_str())
            == Some(H3_QWEN_NVFP4_AWQ_FILENAME)
}

fn hash_safetensors_header(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_PRIVATE_HEADER_BYTES {
        bail!("private H3 safetensors header length {header_len} is invalid")
    }
    let mut header = vec![0_u8; usize::try_from(header_len)?];
    file.read_exact(&mut header)?;
    let mut digest = Sha256::new();
    digest.update(length_bytes);
    digest.update(header);
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_support_assets(
    artifacts: &[ResolvedArtifact<'_>],
    authenticated_identities: &BTreeMap<PathBuf, FileIdentity>,
    task: Task,
) -> Result<()> {
    let read = |filename: &str| -> Result<Vec<u8>> {
        let artifact = artifacts
            .iter()
            .find(|artifact| artifact.file.hf_filename == filename)
            .ok_or_else(|| anyhow!("private H3 manifest omits support asset {filename}"))?;
        let expected = authenticated_identities
            .get(&artifact.relative_path)
            .ok_or_else(|| anyhow!("private H3 support asset {filename} was not authenticated"))?;
        read_authenticated_artifact(artifact, expected)
    };

    H3ConditionerConfig::from_json(&read("text_encoder/config.json")?)?;
    validate_processor_assets(
        &read("processor/preprocessor_config.json")?,
        &read("processor/video_preprocessor_config.json")?,
    )?;
    Tokenizer::from_bytes(&read("processor/tokenizer.json")?)
        .map_err(|error| anyhow!("invalid pinned H3 tokenizer: {error}"))?;

    let visual_value: serde_json::Value = serde_json::from_slice(&read("vae/config.json")?)?;
    require_string_field(&visual_value, "_class_name", "AutoencoderKLMiniMaxH3")?;
    require_string_field(&visual_value, "_diffusers_version", "0.36.0.dev0")?;
    let visual: MiniMaxH3VisualVaeConfig = serde_json::from_value(visual_value)?;
    visual.validate_production_contract()?;
    if visual != MiniMaxH3VisualVaeConfig::default() {
        bail!("pinned H3 visual VAE config differs from Mold's production contract")
    }

    let task_directory = match task {
        Task::Fl2va => "transformer/config.json",
        Task::Ref2va => "transformer_ref/config.json",
    };
    validate_transformer_config(&read(task_directory)?)?;
    validate_scheduler_config(&read("scheduler/scheduler_config.json")?, 12.0)?;
    validate_scheduler_config(&read("audio_scheduler/scheduler_config.json")?, 3.0)?;
    Ok(())
}

fn validate_transformer_config(bytes: &[u8]) -> Result<()> {
    let config: ReleasedTransformerConfig = serde_json::from_slice(bytes)?;
    let expected = ReleasedTransformerConfig {
        class_name: "MiniMaxH3Transformer3DModel".into(),
        diffusers_version: "0.36.0.dev0".into(),
        num_attention_heads: 56,
        attention_head_dim: 128,
        hidden_size: 5_376,
        num_layers: 50,
        num_refiner_layers: 2,
        ffn_dim: 14_336,
        in_channels: 24,
        audio_in_channels: 32,
        patch_size: [1, 2, 2],
        text_dim: 5_120,
        freq_dim: 256,
        time_embed_hidden_dim: 5_376,
        time_embed_dim: 2_688,
        rope_freq_dim: 16,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        final_norm_eps: 1e-5,
    };
    if config != expected {
        bail!("pinned H3 transformer config differs from the released production contract")
    }
    Ok(())
}

fn validate_scheduler_config(bytes: &[u8], expected_shift: f64) -> Result<()> {
    let config: ReleasedSchedulerConfig = serde_json::from_slice(bytes)?;
    if config.class_name != "MiniMaxH3Scheduler"
        || config.diffusers_version != "0.36.0.dev0"
        || config.shift != expected_shift
    {
        bail!("pinned H3 scheduler config differs from the released dual-shift contract")
    }
    Ok(())
}

fn require_string_field(value: &serde_json::Value, field: &str, expected: &str) -> Result<()> {
    if value.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
        bail!("pinned H3 config field {field} does not match {expected:?}")
    }
    Ok(())
}

fn qualification_identity(
    model: &str,
    task: Task,
    authorization_record_sha256: &str,
    authorization_source_document_sha256: &str,
    artifacts: &[H3QualifiedArtifact],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.minimax-h3.private-artifact-qualification.v2\0");
    digest.update(model.as_bytes());
    digest.update(task_id(task).as_bytes());
    digest.update(contract::OFFICIAL_REVISION.as_bytes());
    digest.update(contract::COMFY_REVISION.as_bytes());
    digest.update(authorization_record_sha256.as_bytes());
    digest.update(authorization_source_document_sha256.as_bytes());
    for artifact in artifacts {
        digest.update(artifact.relative_path.as_bytes());
        digest.update(artifact.component.as_bytes());
        digest.update(artifact.source_revision.as_bytes());
        digest.update(artifact.size_bytes.to_le_bytes());
        digest.update(artifact.sha256.as_bytes());
        if let Some(header) = &artifact.header_identity_sha256 {
            digest.update(header.as_bytes());
        }
        if let Some(policy) = &artifact.policy_identity_sha256 {
            digest.update(policy.as_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn candle_task(task: Task) -> H3TransformerTask {
    match task {
        Task::Fl2va => H3TransformerTask::T2VaFl2Va,
        Task::Ref2va => H3TransformerTask::Ref2Va,
    }
}

const fn task_id(task: Task) -> &'static str {
    match task {
        Task::Fl2va => "fl2va",
        Task::Ref2va => "ref2va",
    }
}

const fn component_id(component: ModelComponent) -> &'static str {
    match component {
        ModelComponent::Transformer => "transformer",
        ModelComponent::TextEncoder => "conditioner",
        ModelComponent::Vae => "visual-vae",
        ModelComponent::AudioVae => "audio-vae",
        ModelComponent::TaskConfig => "task-config",
        ModelComponent::ModelConfig => "component-config",
        ModelComponent::Processor => "processor",
        ModelComponent::VideoScheduler => "video-scheduler",
        ModelComponent::AudioScheduler => "audio-scheduler",
        _ => "unexpected",
    }
}

fn portable_path(path: &Path) -> Result<String> {
    let text = path
        .to_str()
        .ok_or_else(|| anyhow!("private H3 artifact path is not UTF-8"))?;
    Ok(text.replace(std::path::MAIN_SEPARATOR, "/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_exact_comfy_manifest_names_are_qualifiable() {
        for model in [contract::FL2VA_COMFY, contract::REF2VA_COMFY] {
            let (manifest, _, _) = qualification_manifest(model).unwrap();
            assert!(manifest.hidden);
            assert_eq!(manifest.family, contract::FAMILY);
        }
        for model in [
            "minimax-h3",
            contract::FL2VA_OFFICIAL,
            contract::REF2VA_OFFICIAL,
            "hf:MiniMaxAI/MiniMax-H3",
        ] {
            assert!(qualification_manifest(model).is_err());
        }
    }

    #[test]
    fn claim_marker_and_scope_are_stable_and_private() {
        assert_eq!(
            H3_PRIVATE_UAT_CLAIM_MARKER,
            "mold.minimax-h3.private-uat-artifact-reader.v1"
        );
        assert_eq!(H3_PRIVATE_AUTHORIZATION_SCOPE, "private-h3-uat");
        assert!(!contract::capabilities(Task::Fl2va).runtime_available);
        assert!(!contract::capabilities(Task::Ref2va).runtime_available);
    }

    #[cfg(unix)]
    struct PrivateCampaignFixture {
        _temporary: tempfile::TempDir,
        campaign_root: PathBuf,
        models_root: PathBuf,
        compliance_root: PathBuf,
        source_document: PathBuf,
        authorization_record: PathBuf,
    }

    #[cfg(unix)]
    impl PrivateCampaignFixture {
        fn new() -> Self {
            let temporary = tempfile::tempdir().unwrap();
            let campaign_root = temporary.path().canonicalize().unwrap().join("campaign");
            let mold_home = campaign_root.join("mold-home");
            let models_root = mold_home.join("models");
            let compliance_root = campaign_root.join("compliance");
            for directory in [&campaign_root, &mold_home, &models_root, &compliance_root] {
                fs::create_dir_all(directory).unwrap();
                set_mode(directory, 0o700);
            }
            let source_document = compliance_root.join("authorization-evidence.bin");
            fs::write(&source_document, b"reviewed private authorization evidence").unwrap();
            set_mode(&source_document, 0o600);
            let authorization_record = compliance_root.join("authorization.v1.json");
            let fixture = Self {
                _temporary: temporary,
                campaign_root,
                models_root,
                compliance_root,
                source_document,
                authorization_record,
            };
            fixture.write_record(fixture.valid_record());
            fixture
        }

        fn valid_record(&self) -> serde_json::Value {
            serde_json::json!({
                "schema_version": "mold.minimax-h3.authorization.v1",
                "family": contract::FAMILY,
                "decision": "approved",
                "license_revision": contract::OFFICIAL_REVISION,
                "license_sha256": contract::LICENSE_SHA256,
                "approved_scopes": [
                    "checkpoint-execution",
                    "fixture-capture",
                    "generated-output-retention"
                ],
                "source_document_path": self.source_document,
                "source_document_sha256": sha256_path(&self.source_document),
                "review_reference": "external-test-review"
            })
        }

        fn write_record(&self, value: serde_json::Value) {
            fs::write(
                &self.authorization_record,
                serde_json::to_vec_pretty(&value).unwrap(),
            )
            .unwrap();
            set_mode(&self.authorization_record, 0o600);
        }
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }

    #[cfg(unix)]
    fn sha256_path(path: &Path) -> String {
        format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
    }

    #[cfg(unix)]
    fn validate_fixture_scope(
        fixture: &PrivateCampaignFixture,
        models_root: &Path,
        authorization_record: &Path,
    ) -> Result<ValidatedPrivateScope> {
        validate_private_scope_against_evidence(
            models_root,
            authorization_record,
            &sha256_path(&fixture.source_document),
        )
    }

    #[cfg(unix)]
    #[test]
    fn production_scope_rejects_unreviewed_evidence() {
        let fixture = PrivateCampaignFixture::new();
        let error = validate_private_scope(&fixture.models_root, &fixture.authorization_record)
            .unwrap_err();
        assert!(error.to_string().contains("reviewed evidence"));
    }

    #[cfg(unix)]
    #[test]
    fn authorization_record_and_owner_only_campaign_layout_are_required() {
        let fixture = PrivateCampaignFixture::new();
        validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .unwrap();
        require_effective_process_owner(effective_process_owner()).unwrap();
        assert!(
            require_effective_process_owner(effective_process_owner().wrapping_add(1)).is_err()
        );

        let other = PrivateCampaignFixture::new();
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &other.authorization_record
        )
        .is_err());

        set_mode(&fixture.authorization_record, 0o644);
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());
        set_mode(&fixture.authorization_record, 0o600);
        set_mode(&fixture.source_document, 0o644);
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn authorization_record_fields_and_source_digest_fail_closed() {
        let fixture = PrivateCampaignFixture::new();
        let valid = fixture.valid_record();

        let mut broader_scope = valid.clone();
        broader_scope["approved_scopes"] = serde_json::json!([
            "checkpoint-execution",
            "fixture-capture",
            "generated-output-retention",
            "later-reviewed-private-activity"
        ]);
        fixture.write_record(broader_scope);
        validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .unwrap();

        let mut wrong_license = valid.clone();
        wrong_license["license_sha256"] = serde_json::Value::String("0".repeat(64));
        fixture.write_record(wrong_license);
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());

        let mut missing_scope = valid.clone();
        missing_scope["approved_scopes"] = serde_json::json!(["fixture-capture"]);
        fixture.write_record(missing_scope);
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());

        let mut duplicate_scope = valid.clone();
        duplicate_scope["approved_scopes"] = serde_json::json!([
            "checkpoint-execution",
            "fixture-capture",
            "fixture-capture",
            "generated-output-retention"
        ]);
        fixture.write_record(duplicate_scope);
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());

        let mut wrong_source = valid;
        wrong_source["source_document_sha256"] = serde_json::Value::String("0".repeat(64));
        fixture.write_record(wrong_source);
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn authorization_source_and_models_root_reject_symlink_aliases() {
        use std::os::unix::fs::symlink;

        let fixture = PrivateCampaignFixture::new();
        let campaign_alias = fixture
            .campaign_root
            .parent()
            .unwrap()
            .join("campaign-alias");
        symlink(&fixture.campaign_root, &campaign_alias).unwrap();
        let aliased_models_root = campaign_alias.join("mold-home/models");
        assert!(validate_fixture_scope(
            &fixture,
            &aliased_models_root,
            &fixture.authorization_record,
        )
        .is_err());

        let source_target = fixture.compliance_root.join("source-target.bin");
        fs::rename(&fixture.source_document, &source_target).unwrap();
        symlink(&source_target, &fixture.source_document).unwrap();
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());

        fs::remove_file(&fixture.source_document).unwrap();
        fs::rename(&source_target, &fixture.source_document).unwrap();
        let models_target = fixture.campaign_root.join("models-target");
        fs::rename(&fixture.models_root, &models_target).unwrap();
        symlink(&models_target, &fixture.models_root).unwrap();
        assert!(validate_fixture_scope(
            &fixture,
            &fixture.models_root,
            &fixture.authorization_record,
        )
        .is_err());
    }

    #[test]
    fn qwen_artifact_requires_manifest_directory_and_canonical_basename() {
        let canonical_path = Path::new("/private/text_encoders").join(H3_QWEN_NVFP4_AWQ_FILENAME);
        assert!(qwen_nvfp4_awq_filename_is_canonical(
            &format!("text_encoders/{H3_QWEN_NVFP4_AWQ_FILENAME}"),
            &canonical_path,
        ));
        assert!(!qwen_nvfp4_awq_filename_is_canonical(
            H3_QWEN_NVFP4_AWQ_FILENAME,
            &canonical_path,
        ));
        assert!(!qwen_nvfp4_awq_filename_is_canonical(
            &format!("text_encoders/{H3_QWEN_NVFP4_AWQ_FILENAME}"),
            Path::new("/private/text_encoders/renamed.safetensors"),
        ));
    }

    #[test]
    fn released_transformer_and_scheduler_configs_are_exact() {
        let transformer = br#"{
          "_class_name":"MiniMaxH3Transformer3DModel",
          "_diffusers_version":"0.36.0.dev0",
          "num_attention_heads":56,"attention_head_dim":128,"hidden_size":5376,
          "num_layers":50,"num_refiner_layers":2,"ffn_dim":14336,
          "in_channels":24,"audio_in_channels":32,"patch_size":[1,2,2],
          "text_dim":5120,"freq_dim":256,"time_embed_hidden_dim":5376,
          "time_embed_dim":2688,"rope_freq_dim":16,"rope_theta":10000.0,
          "norm_eps":0.00001,"qk_norm_eps":0.00001,"final_norm_eps":0.00001
        }"#;
        validate_transformer_config(transformer).unwrap();
        let mut wrong = transformer.to_vec();
        let index = wrong
            .windows(b"\"num_layers\":50".len())
            .position(|window| window == b"\"num_layers\":50")
            .unwrap();
        wrong[index + b"\"num_layers\":".len()] = b'4';
        assert!(validate_transformer_config(&wrong).is_err());

        validate_scheduler_config(
            br#"{"_class_name":"MiniMaxH3Scheduler","_diffusers_version":"0.36.0.dev0","shift":12.0}"#,
            12.0,
        )
        .unwrap();
        assert!(validate_scheduler_config(
            br#"{"_class_name":"MiniMaxH3Scheduler","_diffusers_version":"0.36.0.dev0","shift":3.0}"#,
            12.0,
        )
        .is_err());
    }

    #[cfg(unix)]
    #[test]
    fn authenticated_identity_rejects_replacement_and_symlink_aliases() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("artifact.safetensors");
        let replacement = directory.path().join("replacement.safetensors");
        fs::write(&path, b"authenticated bytes").unwrap();
        fs::write(&replacement, b"different replacement bytes").unwrap();

        let manifest = qualification_manifest(contract::FL2VA_COMFY).unwrap().0;
        let artifact = ResolvedArtifact {
            file: &manifest.files[0],
            relative_path: PathBuf::from("artifact.safetensors"),
            canonical_path: path.clone(),
        };
        let expected = FileIdentity::from_metadata(&fs::symlink_metadata(&path).unwrap());
        require_artifact_identity(&artifact, &expected, "test").unwrap();

        fs::rename(&replacement, &path).unwrap();
        assert!(require_artifact_identity(&artifact, &expected, "test replacement").is_err());

        fs::remove_file(&path).unwrap();
        symlink("missing-target", &path).unwrap();
        assert!(require_artifact_identity(&artifact, &expected, "test symlink").is_err());
    }
}
