//! Candidate producer for an independently reviewed private H3 runtime record.
//!
//! This module cannot activate the runtime. It authenticates the complete
//! artifact set again, binds thirteen externally measured bounds to retained
//! evidence, and emits deterministic candidate bytes. A separate source
//! review must add the exact candidate file hash to the deliberately empty
//! allowlist in `private_server` before the record can authorize execution.

use std::collections::BTreeSet;
use std::fs::{File, Metadata};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use mold_core::minimax_h3;
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::private_qualification::{
    qualify_private_artifacts, H3ArtifactHashProgress, H3PrivateArtifactQualificationReport,
};
use super::private_server::{
    runtime_qualification_identity, validate_runtime_qualification_record_shape,
    H3PrivateRuntimeBoundRecord, H3PrivateRuntimeEvidenceArtifact,
    H3PrivateRuntimeQualificationRecord, RUNTIME_QUALIFICATION_DECISION,
    RUNTIME_QUALIFICATION_SCHEMA,
};

pub const H3_PRIVATE_RUNTIME_RECORD_PRODUCER_MARKER: &str =
    "mold.minimax-h3.private-runtime-record-producer.v1";

const CAPTURE_SCHEMA: &str = "mold.minimax-h3.private-runtime-bound-capture.v1";
const MAX_CAPTURE_BYTES: u64 = 128 * 1024;
const MAX_EVIDENCE_ARTIFACTS: usize = 128;
const MAX_EVIDENCE_ARTIFACT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_EVIDENCE_TOTAL_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct H3PrivateRuntimeBoundCapture {
    observed_bytes: u64,
    bound_bytes: u64,
    evidence_artifact: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct H3PrivateRuntimeBoundCaptureSet {
    fixed_runtime_host_bytes: H3PrivateRuntimeBoundCapture,
    fixed_runtime_device_bytes: H3PrivateRuntimeBoundCapture,
    qwen_activation_workspace_bytes: H3PrivateRuntimeBoundCapture,
    vae_construction_device_workspace_bytes: H3PrivateRuntimeBoundCapture,
    condition_vae_workspace_device_bytes: H3PrivateRuntimeBoundCapture,
    attention_workspace_device_bytes: H3PrivateRuntimeBoundCapture,
    ffn_workspace_device_bytes: H3PrivateRuntimeBoundCapture,
    decoder_tile_workspace_device_bytes: H3PrivateRuntimeBoundCapture,
    audio_decode_workspace_device_bytes: H3PrivateRuntimeBoundCapture,
    encoded_video_host_bytes_bound: H3PrivateRuntimeBoundCapture,
    thumbnail_host_bytes_bound: H3PrivateRuntimeBoundCapture,
    mux_output_host_bytes_bound: H3PrivateRuntimeBoundCapture,
    aac_mux_staging_host_bytes: H3PrivateRuntimeBoundCapture,
}

impl H3PrivateRuntimeBoundCaptureSet {
    fn entries(&self) -> [(&'static str, &H3PrivateRuntimeBoundCapture); 13] {
        [
            ("fixed_runtime_host_bytes", &self.fixed_runtime_host_bytes),
            (
                "fixed_runtime_device_bytes",
                &self.fixed_runtime_device_bytes,
            ),
            (
                "qwen_activation_workspace_bytes",
                &self.qwen_activation_workspace_bytes,
            ),
            (
                "vae_construction_device_workspace_bytes",
                &self.vae_construction_device_workspace_bytes,
            ),
            (
                "condition_vae_workspace_device_bytes",
                &self.condition_vae_workspace_device_bytes,
            ),
            (
                "attention_workspace_device_bytes",
                &self.attention_workspace_device_bytes,
            ),
            (
                "ffn_workspace_device_bytes",
                &self.ffn_workspace_device_bytes,
            ),
            (
                "decoder_tile_workspace_device_bytes",
                &self.decoder_tile_workspace_device_bytes,
            ),
            (
                "audio_decode_workspace_device_bytes",
                &self.audio_decode_workspace_device_bytes,
            ),
            (
                "encoded_video_host_bytes_bound",
                &self.encoded_video_host_bytes_bound,
            ),
            (
                "thumbnail_host_bytes_bound",
                &self.thumbnail_host_bytes_bound,
            ),
            (
                "mux_output_host_bytes_bound",
                &self.mux_output_host_bytes_bound,
            ),
            (
                "aac_mux_staging_host_bytes",
                &self.aac_mux_staging_host_bytes,
            ),
        ]
    }

    fn validate(
        &self,
        evidence_artifacts: &BTreeSet<String>,
    ) -> Result<H3PrivateRuntimeBoundRecord> {
        for (label, capture) in self.entries() {
            if capture.observed_bytes == 0
                || capture.bound_bytes == 0
                || capture.observed_bytes > capture.bound_bytes
            {
                bail!(
                    "private H3 runtime bound {label} must retain a nonzero observation no larger than its bound"
                )
            }
            validate_relative_path(&capture.evidence_artifact, "bound evidence")?;
            if !evidence_artifacts.contains(&capture.evidence_artifact) {
                bail!("private H3 runtime bound {label} names unretained evidence")
            }
        }
        Ok(H3PrivateRuntimeBoundRecord {
            fixed_runtime_host_bytes: self.fixed_runtime_host_bytes.bound_bytes,
            fixed_runtime_device_bytes: self.fixed_runtime_device_bytes.bound_bytes,
            qwen_activation_workspace_bytes: self.qwen_activation_workspace_bytes.bound_bytes,
            vae_construction_device_workspace_bytes: self
                .vae_construction_device_workspace_bytes
                .bound_bytes,
            condition_vae_workspace_device_bytes: self
                .condition_vae_workspace_device_bytes
                .bound_bytes,
            attention_workspace_device_bytes: self.attention_workspace_device_bytes.bound_bytes,
            ffn_workspace_device_bytes: self.ffn_workspace_device_bytes.bound_bytes,
            decoder_tile_workspace_device_bytes: self
                .decoder_tile_workspace_device_bytes
                .bound_bytes,
            audio_decode_workspace_device_bytes: self
                .audio_decode_workspace_device_bytes
                .bound_bytes,
            encoded_video_host_bytes_bound: self.encoded_video_host_bytes_bound.bound_bytes,
            thumbnail_host_bytes_bound: self.thumbnail_host_bytes_bound.bound_bytes,
            mux_output_host_bytes_bound: self.mux_output_host_bytes_bound.bound_bytes,
            aac_mux_staging_host_bytes: self.aac_mux_staging_host_bytes.bound_bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct H3PrivateRuntimeBoundCaptureManifest {
    schema: String,
    canonical_model: String,
    task: String,
    source_sha: String,
    authorization_record_sha256: String,
    authorization_source_document_sha256: String,
    artifact_qualification_identity_sha256: String,
    device_id: String,
    device_ordinal: usize,
    compute_capability: [u16; 2],
    attention_runtime_identity_sha256: String,
    attention_kernel_identity: String,
    attention_qualification_sha256: String,
    bounds: H3PrivateRuntimeBoundCaptureSet,
    evidence_artifacts: Vec<String>,
}

impl H3PrivateRuntimeBoundCaptureManifest {
    fn validate(&self, artifact: &H3PrivateArtifactQualificationReport) -> Result<()> {
        if self.schema != CAPTURE_SCHEMA
            || self.canonical_model != minimax_h3::FL2VA_COMFY
            || self.task != "fl2va"
            || self.canonical_model != artifact.canonical_model
            || self.task != artifact.task
            || self.authorization_record_sha256 != artifact.authorization_record_sha256
            || self.authorization_source_document_sha256
                != artifact.authorization_source_document_sha256
            || self.artifact_qualification_identity_sha256 != artifact.qualification_identity_sha256
        {
            bail!("private H3 runtime capture differs from authenticated campaign authority")
        }
        if self.source_sha.len() != 40
            || !self
                .source_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            bail!("private H3 runtime capture source SHA is not canonical")
        }
        if self.device_id != format!("cuda:{}", self.device_ordinal)
            || self.compute_capability[0] == 0
            || self.compute_capability[1] > 99
            || !valid_lower_sha256(&self.attention_runtime_identity_sha256)
            || !valid_lower_sha256(&self.attention_qualification_sha256)
            || self.attention_kernel_identity.trim().is_empty()
            || self.attention_kernel_identity.len() > 256
            || self.attention_kernel_identity.chars().any(char::is_control)
        {
            bail!("private H3 runtime capture has invalid CUDA attention authority")
        }
        if self.evidence_artifacts.is_empty()
            || self.evidence_artifacts.len() > MAX_EVIDENCE_ARTIFACTS
            || self
                .evidence_artifacts
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("private H3 runtime capture evidence paths must be sorted and unique")
        }
        for relative_path in &self.evidence_artifacts {
            validate_relative_path(relative_path, "runtime evidence")?;
        }
        let retained = self.evidence_artifacts.iter().cloned().collect();
        self.bounds.validate(&retained)?;
        Ok(())
    }
}

/// Deterministic candidate bytes and identities for separate source review.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3PrivateRuntimeQualificationCandidate {
    json_bytes: Vec<u8>,
    record_file_sha256: String,
    identity_sha256: String,
    evidence_artifact_count: usize,
}

impl H3PrivateRuntimeQualificationCandidate {
    pub fn json_bytes(&self) -> &[u8] {
        &self.json_bytes
    }

    pub fn record_file_sha256(&self) -> &str {
        &self.record_file_sha256
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub const fn evidence_artifact_count(&self) -> usize {
        self.evidence_artifact_count
    }
}

/// Re-authenticate every private artifact and produce one review candidate.
///
/// The returned file hash is deliberately not accepted by the runtime. A
/// later reviewed change must add that exact hash to the source allowlist.
pub fn produce_h3_private_runtime_qualification_candidate(
    models_root: &Path,
    authorization_record: &Path,
    evidence_root: &Path,
    capture_manifest: &Path,
    progress: impl FnMut(H3ArtifactHashProgress),
) -> Result<H3PrivateRuntimeQualificationCandidate> {
    let evidence_root = canonical_private_evidence_root(evidence_root)?;
    let capture_relative = relative_evidence_path(&evidence_root, capture_manifest)?;
    let capture_artifact =
        hash_evidence_artifact(&evidence_root, &capture_relative, Some(MAX_CAPTURE_BYTES))?;
    let capture_bytes = read_evidence_bytes(&evidence_root, &capture_relative, MAX_CAPTURE_BYTES)?;
    if capture_artifact.bytes != capture_bytes.len() as u64
        || capture_artifact.sha256 != format!("{:x}", Sha256::digest(&capture_bytes))
    {
        bail!("private H3 runtime capture changed between authentication and parsing")
    }
    let capture: H3PrivateRuntimeBoundCaptureManifest = serde_json::from_slice(&capture_bytes)
        .context("private H3 runtime capture manifest is not exact-schema JSON")?;
    let artifact = qualify_private_artifacts(
        models_root,
        minimax_h3::FL2VA_COMFY,
        authorization_record,
        progress,
    )?;
    build_candidate(
        &artifact,
        &evidence_root,
        capture_relative,
        capture_artifact,
        capture,
    )
}

fn build_candidate(
    artifact: &H3PrivateArtifactQualificationReport,
    evidence_root: &Path,
    capture_relative: String,
    capture_artifact: H3PrivateRuntimeEvidenceArtifact,
    capture: H3PrivateRuntimeBoundCaptureManifest,
) -> Result<H3PrivateRuntimeQualificationCandidate> {
    if artifact.decision != "qualified-private-artifacts"
        || artifact.canonical_model != minimax_h3::FL2VA_COMFY
        || artifact.task != "fl2va"
        || artifact.total_bytes == 0
        || artifact.runtime_constructed
        || artifact.generated_media
        || artifact.public_activation != "rejected"
        || !valid_lower_sha256(&artifact.authorization_record_sha256)
        || !valid_lower_sha256(&artifact.authorization_source_document_sha256)
        || !valid_lower_sha256(&artifact.qualification_identity_sha256)
    {
        bail!("private H3 runtime candidate lacks exact artifact qualification")
    }
    capture.validate(artifact)?;
    if capture
        .evidence_artifacts
        .iter()
        .any(|path| path == &capture_relative)
    {
        bail!("private H3 runtime capture manifest must not list itself as bound evidence")
    }
    let bounds = capture
        .bounds
        .validate(&capture.evidence_artifacts.iter().cloned().collect())?;
    let mut evidence_artifacts = Vec::with_capacity(capture.evidence_artifacts.len() + 1);
    evidence_artifacts.push(capture_artifact);
    let mut total_bytes = evidence_artifacts[0].bytes;
    for relative_path in &capture.evidence_artifacts {
        let evidence = hash_evidence_artifact(evidence_root, relative_path, None)?;
        total_bytes = total_bytes
            .checked_add(evidence.bytes)
            .ok_or_else(|| anyhow!("private H3 runtime evidence byte count overflow"))?;
        if total_bytes > MAX_EVIDENCE_TOTAL_BYTES {
            bail!("private H3 runtime evidence exceeds the retained byte limit")
        }
        evidence_artifacts.push(evidence);
    }
    evidence_artifacts.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let mut record = H3PrivateRuntimeQualificationRecord {
        schema: RUNTIME_QUALIFICATION_SCHEMA.into(),
        decision: RUNTIME_QUALIFICATION_DECISION.into(),
        canonical_model: artifact.canonical_model.clone(),
        task: artifact.task.into(),
        authorization_record_sha256: artifact.authorization_record_sha256.clone(),
        authorization_source_document_sha256: artifact.authorization_source_document_sha256.clone(),
        artifact_qualification_identity_sha256: artifact.qualification_identity_sha256.clone(),
        artifact_total_bytes: artifact.total_bytes,
        device_id: capture.device_id,
        device_ordinal: capture.device_ordinal,
        compute_capability: capture.compute_capability,
        attention_runtime_identity_sha256: capture.attention_runtime_identity_sha256,
        attention_kernel_identity: capture.attention_kernel_identity,
        attention_qualification_sha256: capture.attention_qualification_sha256,
        bounds,
        evidence_artifacts,
        identity_sha256: String::new(),
    };
    record.identity_sha256 = runtime_qualification_identity(&record);
    validate_runtime_qualification_record_shape(&record)?;
    let mut json_bytes = serde_json::to_vec_pretty(&record)?;
    json_bytes.push(b'\n');
    let record_file_sha256 = format!("{:x}", Sha256::digest(&json_bytes));
    Ok(H3PrivateRuntimeQualificationCandidate {
        evidence_artifact_count: record.evidence_artifacts.len(),
        identity_sha256: record.identity_sha256,
        json_bytes,
        record_file_sha256,
    })
}

fn canonical_private_evidence_root(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("private H3 runtime evidence root must be absolute")
    }
    let canonical = path
        .canonicalize()
        .context("failed to canonicalize private H3 runtime evidence root")?;
    if canonical != path {
        bail!("private H3 runtime evidence root must not contain aliases")
    }
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("cannot resolve the Mold repository root"))?
        .canonicalize()?;
    if canonical.starts_with(repository_root) {
        bail!("private H3 runtime evidence must live outside the Mold repository")
    }
    require_private_directory(&canonical, None)?;
    Ok(canonical)
}

fn relative_evidence_path(root: &Path, path: &Path) -> Result<String> {
    if !path.is_absolute() {
        bail!("private H3 runtime capture path must be absolute")
    }
    let relative = path
        .strip_prefix(root)
        .context("private H3 runtime capture must be inside the evidence root")?;
    let relative = relative
        .to_str()
        .ok_or_else(|| anyhow!("private H3 runtime evidence path is not UTF-8"))?
        .to_owned();
    validate_relative_path(&relative, "runtime capture")?;
    Ok(relative)
}

fn validate_relative_path(path: &str, label: &str) -> Result<()> {
    let candidate = Path::new(path);
    if path.is_empty()
        || path.contains('\\')
        || path.chars().any(char::is_control)
        || candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("private H3 {label} path is not a canonical relative path")
    }
    Ok(())
}

fn hash_evidence_artifact(
    root: &Path,
    relative_path: &str,
    explicit_limit: Option<u64>,
) -> Result<H3PrivateRuntimeEvidenceArtifact> {
    validate_relative_path(relative_path, "runtime evidence")?;
    let path = root.join(relative_path);
    validate_private_parent_chain(root, &path)?;
    let file = open_regular_file_no_follow(&path)
        .with_context(|| format!("failed to open private H3 runtime evidence {relative_path}"))?;
    let before = require_private_file(&file, relative_path)?;
    let limit = explicit_limit.unwrap_or(MAX_EVIDENCE_ARTIFACT_BYTES);
    if before.len == 0 || before.len > limit {
        bail!("private H3 runtime evidence {relative_path} has an invalid size")
    }
    let sha256 = sha256_open_file(&file)?;
    let after = EvidenceFileIdentity::from_metadata(&file.metadata()?);
    let current = open_regular_file_no_follow(&path)?;
    let current_identity = EvidenceFileIdentity::from_metadata(&current.metadata()?);
    if before != after || before != current_identity || sha256_open_file(&current)? != sha256 {
        bail!("private H3 runtime evidence {relative_path} changed while hashing")
    }
    Ok(H3PrivateRuntimeEvidenceArtifact {
        relative_path: relative_path.into(),
        bytes: before.len,
        sha256,
    })
}

fn read_evidence_bytes(root: &Path, relative_path: &str, limit: u64) -> Result<Vec<u8>> {
    let path = root.join(relative_path);
    let mut file = open_regular_file_no_follow(&path)?;
    let before = require_private_file(&file, relative_path)?;
    if before.len == 0 || before.len > limit {
        bail!("private H3 runtime evidence {relative_path} has an invalid size")
    }
    let mut bytes = Vec::with_capacity(before.len as usize);
    file.read_to_end(&mut bytes)?;
    if EvidenceFileIdentity::from_metadata(&file.metadata()?) != before {
        bail!("private H3 runtime evidence {relative_path} changed while reading")
    }
    Ok(bytes)
}

fn validate_private_parent_chain(root: &Path, path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("private H3 runtime evidence path has no parent"))?;
    let relative = parent
        .strip_prefix(root)
        .context("private H3 runtime evidence escapes its root")?;
    let owner = require_private_directory(root, None)?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            bail!("private H3 runtime evidence parent is not canonical")
        };
        current.push(component);
        require_private_directory(&current, Some(owner))?;
    }
    Ok(())
}

#[cfg(unix)]
fn require_private_directory(path: &Path, expected_owner: Option<u32>) -> Result<u32> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.mode() & 0o7777 != 0o700
        || expected_owner.is_some_and(|owner| owner != metadata.uid())
    {
        bail!("private H3 runtime evidence directories must be owner-only and non-symlink")
    }
    // SAFETY: `geteuid` has no preconditions and only reads process state.
    if metadata.uid() != unsafe { libc::geteuid() } {
        bail!("private H3 runtime evidence must be owned by the invoking process user")
    }
    Ok(metadata.uid())
}

#[cfg(not(unix))]
fn require_private_directory(_path: &Path, _expected_owner: Option<u32>) -> Result<u32> {
    bail!("private H3 runtime evidence currently requires Unix ownership semantics")
}

#[cfg(unix)]
fn require_private_file(file: &File, relative_path: &str) -> Result<EvidenceFileIdentity> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata()?;
    // SAFETY: `geteuid` has no preconditions and only reads process state.
    let effective_uid = unsafe { libc::geteuid() };
    if !metadata.is_file() || metadata.uid() != effective_uid || metadata.mode() & 0o7777 != 0o600 {
        bail!("private H3 runtime evidence {relative_path} must be an owner-only regular file")
    }
    Ok(EvidenceFileIdentity::from_metadata(&metadata))
}

#[cfg(not(unix))]
fn require_private_file(_file: &File, _relative_path: &str) -> Result<EvidenceFileIdentity> {
    bail!("private H3 runtime evidence currently requires Unix permission semantics")
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvidenceFileIdentity {
    device: u64,
    inode: u64,
    len: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    owner: u32,
    mode: u32,
}

#[cfg(unix)]
impl EvidenceFileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;

        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            len: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            owner: metadata.uid(),
            mode: metadata.mode(),
        }
    }
}

#[cfg(not(unix))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvidenceFileIdentity;

#[cfg(not(unix))]
impl EvidenceFileIdentity {
    fn from_metadata(_metadata: &Metadata) -> Self {
        Self
    }
}

fn valid_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn artifact_report() -> H3PrivateArtifactQualificationReport {
        H3PrivateArtifactQualificationReport {
            schema: "mold.minimax-h3.private-artifact-qualification.v2",
            claim_marker: "test-private-marker",
            decision: "qualified-private-artifacts",
            authorization_scope: "checkpoint-execution",
            authorization_schema: "mold.minimax-h3.authorization.v1",
            authorization_record_sha256: sha('a'),
            authorization_source_document_sha256: sha('b'),
            authorization_review_reference: "review-1".into(),
            canonical_model: minimax_h3::FL2VA_COMFY.into(),
            task: "fl2va",
            layout: "test-layout",
            official_model_revision: "official",
            comfy_model_revision: "comfy",
            official_implementation_revision: "official-code",
            comfy_implementation_revision: "comfy-code",
            artifacts: vec![],
            total_bytes: 42,
            qualification_identity_sha256: sha('c'),
            runtime_constructed: false,
            generated_media: false,
            public_activation: "rejected",
            remaining_release_requirements: vec!["private-cuda-runtime-qualification"],
        }
    }

    fn bound(path: &str, value: u64) -> H3PrivateRuntimeBoundCapture {
        H3PrivateRuntimeBoundCapture {
            observed_bytes: value,
            bound_bytes: value + 1,
            evidence_artifact: path.into(),
        }
    }

    fn capture(path: &str) -> H3PrivateRuntimeBoundCaptureManifest {
        H3PrivateRuntimeBoundCaptureManifest {
            schema: CAPTURE_SCHEMA.into(),
            canonical_model: minimax_h3::FL2VA_COMFY.into(),
            task: "fl2va".into(),
            source_sha: std::iter::repeat_n('d', 40).collect(),
            authorization_record_sha256: sha('a'),
            authorization_source_document_sha256: sha('b'),
            artifact_qualification_identity_sha256: sha('c'),
            device_id: "cuda:0".into(),
            device_ordinal: 0,
            compute_capability: [8, 9],
            attention_runtime_identity_sha256: sha('d'),
            attention_kernel_identity: "h3-flash-attention-sm89".into(),
            attention_qualification_sha256: sha('e'),
            bounds: H3PrivateRuntimeBoundCaptureSet {
                fixed_runtime_host_bytes: bound(path, 1),
                fixed_runtime_device_bytes: bound(path, 2),
                qwen_activation_workspace_bytes: bound(path, 3),
                vae_construction_device_workspace_bytes: bound(path, 4),
                condition_vae_workspace_device_bytes: bound(path, 5),
                attention_workspace_device_bytes: bound(path, 6),
                ffn_workspace_device_bytes: bound(path, 7),
                decoder_tile_workspace_device_bytes: bound(path, 8),
                audio_decode_workspace_device_bytes: bound(path, 9),
                encoded_video_host_bytes_bound: bound(path, 10),
                thumbnail_host_bytes_bound: bound(path, 11),
                mux_output_host_bytes_bound: bound(path, 12),
                aac_mux_staging_host_bytes: bound(path, 13),
            },
            evidence_artifacts: vec![path.into()],
        }
    }

    #[cfg(unix)]
    fn private_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let logs = root.path().join("logs");
        fs::create_dir(&logs).unwrap();
        fs::set_permissions(&logs, fs::Permissions::from_mode(0o700)).unwrap();
        let evidence = logs.join("runtime.log");
        fs::write(&evidence, b"measured runtime evidence").unwrap();
        fs::set_permissions(&evidence, fs::Permissions::from_mode(0o600)).unwrap();
        let capture_path = root.path().join("capture.json");
        let capture = capture("logs/runtime.log");
        fs::write(&capture_path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
        fs::set_permissions(&capture_path, fs::Permissions::from_mode(0o600)).unwrap();
        (root, capture_path, evidence)
    }

    #[cfg(unix)]
    #[test]
    fn candidate_is_deterministic_and_accepted_by_the_runtime_shape_validator() {
        let (root, capture_path, _) = private_fixture();
        let capture_relative = relative_evidence_path(root.path(), &capture_path).unwrap();
        let capture_artifact =
            hash_evidence_artifact(root.path(), &capture_relative, Some(MAX_CAPTURE_BYTES))
                .unwrap();
        let manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(&capture_path).unwrap()).unwrap();
        let first = build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative.clone(),
            capture_artifact.clone(),
            manifest.clone(),
        )
        .unwrap();
        let second = build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative,
            capture_artifact,
            manifest,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.evidence_artifact_count(), 2);
        assert!(valid_lower_sha256(first.record_file_sha256()));
        let record: H3PrivateRuntimeQualificationRecord =
            serde_json::from_slice(first.json_bytes()).unwrap();
        validate_runtime_qualification_record_shape(&record).unwrap();
        assert_eq!(record.identity_sha256, first.identity_sha256());
    }

    #[cfg(unix)]
    #[test]
    fn candidate_rejects_an_unmeasured_or_unretained_bound() {
        let (root, capture_path, _) = private_fixture();
        let capture_relative = relative_evidence_path(root.path(), &capture_path).unwrap();
        let capture_artifact =
            hash_evidence_artifact(root.path(), &capture_relative, Some(MAX_CAPTURE_BYTES))
                .unwrap();
        let mut manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(&capture_path).unwrap()).unwrap();
        manifest.bounds.ffn_workspace_device_bytes.observed_bytes = 0;
        assert!(build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative.clone(),
            capture_artifact.clone(),
            manifest,
        )
        .unwrap_err()
        .to_string()
        .contains("ffn_workspace_device_bytes"));

        let mut manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(&capture_path).unwrap()).unwrap();
        manifest.bounds.ffn_workspace_device_bytes.evidence_artifact = "logs/missing.log".into();
        assert!(build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative,
            capture_artifact,
            manifest,
        )
        .unwrap_err()
        .to_string()
        .contains("unretained evidence"));
    }

    #[cfg(unix)]
    #[test]
    fn evidence_permissions_and_order_fail_closed() {
        use std::os::unix::fs::PermissionsExt;

        let (root, capture_path, evidence) = private_fixture();
        fs::set_permissions(&evidence, fs::Permissions::from_mode(0o640)).unwrap();
        assert!(
            hash_evidence_artifact(root.path(), "logs/runtime.log", None)
                .unwrap_err()
                .to_string()
                .contains("owner-only")
        );

        let mut manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(capture_path).unwrap()).unwrap();
        manifest.evidence_artifacts = vec!["logs/z.log".into(), "logs/a.log".into()];
        assert!(manifest
            .validate(&artifact_report())
            .unwrap_err()
            .to_string()
            .contains("sorted and unique"));
    }
}
