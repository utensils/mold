//! Candidate producer for an independently reviewed private H3 runtime record.
//!
//! This module cannot activate the runtime. It authenticates the complete
//! artifact set again, binds thirteen externally measured bounds to retained
//! evidence, and emits deterministic candidate bytes. A separate source
//! review must add the exact candidate file hash to the deliberately empty
//! allowlist in `private_server` before the record can authorize execution.

use std::collections::BTreeSet;
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use mold_core::minimax_h3;
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::private_qualification::{
    qualify_private_artifacts, H3ArtifactHashProgress, H3PrivateArtifactQualificationReport,
    H3_PRIVATE_AUTHORIZATION_SCOPE, H3_PRIVATE_UAT_CLAIM_MARKER, QUALIFICATION_SCHEMA,
};
#[cfg(test)]
use super::private_runtime_observer::H3PrivateRuntimeAuthorityObservation;
use super::private_runtime_observer::{
    H3PrivateRuntimeBoundObservation, H3PrivateRuntimeProcessObservation,
    H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA, H3_PUBLIC_RUNTIME_BOUND_OBSERVATION_SCHEMA,
};
use super::private_server::{
    runtime_qualification_identity, valid_stable_cuda_device_id,
    validate_runtime_qualification_record_shape, H3PrivateRuntimeBoundRecord,
    H3PrivateRuntimeEnvelopeRecord, H3PrivateRuntimeEvidenceArtifact,
    H3PrivateRuntimeQualificationRecord, MAX_RUNTIME_QUALIFICATION_BYTES,
    RUNTIME_QUALIFICATION_DECISION, RUNTIME_QUALIFICATION_SCHEMA,
};

pub const H3_PRIVATE_RUNTIME_RECORD_PRODUCER_MARKER: &str =
    "mold.minimax-h3.private-runtime-record-producer.v1";

const CAPTURE_SCHEMA: &str = "mold.minimax-h3.private-runtime-bound-capture.v5";
const MAX_CAPTURE_BYTES: u64 = 128 * 1024;
const MAX_RUNTIME_OBSERVATION_BYTES: u64 = 128 * 1024;
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
            let zero_observation_allowed = label == "vae_construction_device_workspace_bytes";
            if (!zero_observation_allowed && capture.observed_bytes == 0)
                || capture.bound_bytes == 0
                || capture.observed_bytes > capture.bound_bytes
            {
                bail!(
                    "private H3 runtime bound {label} must retain an allowed observation no larger than its nonzero bound"
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

    fn validate_observation(&self, observation: &H3PrivateRuntimeBoundObservation) -> Result<()> {
        let observed = [
            observation.fixed_runtime_host_bytes,
            observation.fixed_runtime_device_bytes,
            observation.qwen_activation_workspace_bytes,
            observation.vae_construction_device_workspace_bytes,
            observation.condition_vae_workspace_device_bytes,
            observation.attention_workspace_device_bytes,
            observation.ffn_workspace_device_bytes,
            observation.decoder_tile_workspace_device_bytes,
            observation.audio_decode_workspace_device_bytes,
            observation.encoded_video_host_bytes_bound,
            observation.thumbnail_host_bytes_bound,
            observation.mux_output_host_bytes_bound,
            observation.aac_mux_staging_host_bytes,
        ];
        for (((label, capture), observed), index) in
            self.entries().into_iter().zip(observed).zip(0_usize..)
        {
            if capture.observed_bytes != observed {
                bail!(
                    "private H3 runtime bound {label} differs from structured observation field {index}"
                )
            }
        }
        Ok(())
    }
}

fn validate_observed_envelope(
    reviewed: &H3PrivateRuntimeEnvelopeRecord,
    observation: &H3PrivateRuntimeBoundObservation,
) -> Result<()> {
    if observation.schema != H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA
        && observation.schema != H3_PUBLIC_RUNTIME_BOUND_OBSERVATION_SCHEMA
    {
        bail!("private H3 runtime observation has an unknown schema")
    }
    let observed = &observation.envelope;
    if reviewed.width != observed.width
        || reviewed.height != observed.height
        || reviewed.frames != observed.frames
        || reviewed.fps != observed.fps
        || reviewed.batch_size != observed.batch_size
        || reviewed.max_steps != observed.steps
        || reviewed.endpoint_count != observed.endpoint_count
        || reviewed.endpoint_anchor != observed.endpoint_anchor
        || reviewed.max_qwen_output_text_rows != observed.qwen_output_text_rows
        || reviewed.max_qwen_vision_rows != observed.qwen_vision_rows
        || reviewed.max_condition_visual_rows != observed.condition_visual_rows
        || reviewed.max_target_video_rows != observed.target_video_rows
        || reviewed.max_target_audio_rows != observed.target_audio_rows
        || reviewed.max_total_packed_rows != observed.total_packed_rows
    {
        bail!("private H3 runtime envelope differs from its structured observation")
    }
    Ok(())
}

fn validate_observed_authority(
    capture: &H3PrivateRuntimeBoundCaptureManifest,
    observation: &H3PrivateRuntimeBoundObservation,
) -> Result<()> {
    let authority = &observation.authority;
    if authority.bootstrap_record_sha256 != capture.bootstrap_runtime_record_sha256
        || authority.runtime_qualification_identity_sha256
            != capture.bootstrap_runtime_qualification_identity_sha256
        || authority.device_id != capture.device_id
        || authority.device_ordinal != capture.device_ordinal
        || authority.compute_capability != capture.compute_capability
        || authority.attention_runtime_identity_sha256 != capture.attention_runtime_identity_sha256
        || authority.attention_kernel_identity != capture.attention_kernel_identity
        || authority.attention_qualification_sha256 != capture.attention_qualification_sha256
        || authority.process != capture.process
    {
        bail!("private H3 runtime authority differs from its structured observation")
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct H3PrivateRuntimeBoundCaptureManifest {
    schema: String,
    canonical_model: String,
    task: String,
    source_sha: String,
    runtime_code_identity_sha256: String,
    bootstrap_runtime_record_sha256: String,
    bootstrap_runtime_qualification_identity_sha256: String,
    measured_server_executable: String,
    authorization_record_sha256: String,
    authorization_source_document_sha256: String,
    artifact_qualification_identity_sha256: String,
    device_id: String,
    device_ordinal: usize,
    compute_capability: [u16; 2],
    attention_runtime_identity_sha256: String,
    attention_kernel_identity: String,
    attention_qualification_sha256: String,
    process: H3PrivateRuntimeProcessObservation,
    runtime_observation_artifact: String,
    envelope: H3PrivateRuntimeEnvelopeRecord,
    bounds: H3PrivateRuntimeBoundCaptureSet,
    evidence_artifacts: Vec<String>,
}

impl H3PrivateRuntimeBoundCaptureManifest {
    fn validate(
        &self,
        artifact: &H3PrivateArtifactQualificationReport,
        embedded_source_sha: &str,
        embedded_runtime_code_identity_sha256: &str,
    ) -> Result<()> {
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
        if !valid_lower_hex(&self.source_sha, 40)
            || self.source_sha != embedded_source_sha
            || !valid_lower_sha256(&self.runtime_code_identity_sha256)
            || self.runtime_code_identity_sha256 != embedded_runtime_code_identity_sha256
        {
            bail!("private H3 runtime capture differs from the embedded campaign build")
        }
        if !valid_lower_sha256(&self.bootstrap_runtime_record_sha256)
            || !valid_lower_sha256(&self.bootstrap_runtime_qualification_identity_sha256)
        {
            bail!("private H3 runtime capture has no exact bootstrap record")
        }
        if !valid_stable_cuda_device_id(&self.device_id)
            || self.compute_capability[0] == 0
            || self.compute_capability[0] > 99
            || self.compute_capability[1] > 99
            || !valid_lower_sha256(&self.attention_runtime_identity_sha256)
            || !valid_lower_sha256(&self.attention_qualification_sha256)
            || self.attention_kernel_identity.trim().is_empty()
            || self.attention_kernel_identity.len() > 256
            || self.attention_kernel_identity.chars().any(char::is_control)
        {
            bail!("private H3 runtime capture has invalid CUDA attention authority")
        }
        validate_process_observation(&self.process)?;
        self.envelope.validate()?;
        validate_relative_path(
            &self.runtime_observation_artifact,
            "runtime observation evidence",
        )?;
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
        if !self
            .evidence_artifacts
            .contains(&self.runtime_observation_artifact)
        {
            bail!("private H3 runtime capture does not retain its structured observation")
        }
        validate_relative_path(
            &self.measured_server_executable,
            "measured server executable",
        )?;
        if !self
            .evidence_artifacts
            .contains(&self.measured_server_executable)
        {
            bail!("private H3 runtime capture omits its measured server executable")
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
    let embedded_source_sha = mold_core::build_info::GIT_SHA;
    if !valid_lower_hex(embedded_source_sha, 40) {
        bail!("private H3 runtime candidate requires an exact embedded source SHA")
    }
    let embedded_runtime_code_identity_sha256 = super::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256;
    if !valid_lower_sha256(embedded_runtime_code_identity_sha256) {
        bail!("private H3 runtime candidate lacks an embedded runtime-code identity")
    }
    let evidence_root = canonical_private_evidence_root(evidence_root)?;
    let capture_relative = relative_evidence_path(&evidence_root, capture_manifest)?;
    let (capture_artifact, capture_bytes) =
        read_authenticated_evidence_artifact(&evidence_root, &capture_relative, MAX_CAPTURE_BYTES)?;
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
        embedded_source_sha,
        embedded_runtime_code_identity_sha256,
    )
}

fn build_candidate(
    artifact: &H3PrivateArtifactQualificationReport,
    evidence_root: &Path,
    capture_relative: String,
    capture_artifact: H3PrivateRuntimeEvidenceArtifact,
    capture: H3PrivateRuntimeBoundCaptureManifest,
    embedded_source_sha: &str,
    embedded_runtime_code_identity_sha256: &str,
) -> Result<H3PrivateRuntimeQualificationCandidate> {
    validate_candidate_artifact_qualification(artifact, cfg!(feature = "h3"))?;
    capture.validate(
        artifact,
        embedded_source_sha,
        embedded_runtime_code_identity_sha256,
    )?;
    if capture
        .evidence_artifacts
        .iter()
        .any(|path| path == &capture_relative)
    {
        bail!("private H3 runtime capture manifest must not list itself as bound evidence")
    }
    let observation_relative = capture.runtime_observation_artifact.clone();
    let bounds = capture
        .bounds
        .validate(&capture.evidence_artifacts.iter().cloned().collect())?;
    let mut evidence_artifacts = Vec::with_capacity(capture.evidence_artifacts.len() + 1);
    evidence_artifacts.push(capture_artifact);
    let mut total_bytes = evidence_artifacts[0].bytes;
    let mut observation = None;
    for relative_path in &capture.evidence_artifacts {
        let evidence = if relative_path == &capture.measured_server_executable {
            hash_measured_server_executable(
                evidence_root,
                relative_path,
                &capture.source_sha,
                &capture.runtime_code_identity_sha256,
            )?
        } else if relative_path == &observation_relative {
            let (evidence, bytes) = read_authenticated_evidence_artifact(
                evidence_root,
                relative_path,
                MAX_RUNTIME_OBSERVATION_BYTES,
            )?;
            let parsed = serde_json::from_slice(&bytes)
                .context("private H3 runtime observation is not exact-schema JSON")?;
            observation = Some(parsed);
            evidence
        } else {
            hash_evidence_artifact(evidence_root, relative_path, None)?
        };
        total_bytes = total_bytes
            .checked_add(evidence.bytes)
            .ok_or_else(|| anyhow!("private H3 runtime evidence byte count overflow"))?;
        if total_bytes > MAX_EVIDENCE_TOTAL_BYTES {
            bail!("private H3 runtime evidence exceeds the retained byte limit")
        }
        evidence_artifacts.push(evidence);
    }
    let observation = observation
        .ok_or_else(|| anyhow!("private H3 runtime capture lost its structured observation"))?;
    validate_observed_envelope(&capture.envelope, &observation)?;
    validate_observed_authority(&capture, &observation)?;
    capture.bounds.validate_observation(&observation)?;
    evidence_artifacts.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let measured_server_executable = evidence_artifacts
        .iter()
        .find(|artifact| artifact.relative_path == capture.measured_server_executable)
        .ok_or_else(|| anyhow!("private H3 runtime candidate lost its measured executable"))?;
    if capture.process.executable_sha256 != measured_server_executable.sha256
        || capture.process.executable_bytes != measured_server_executable.bytes
    {
        bail!("private H3 process attestation differs from the retained server executable")
    }

    let mut record = H3PrivateRuntimeQualificationRecord {
        schema: RUNTIME_QUALIFICATION_SCHEMA.into(),
        decision: RUNTIME_QUALIFICATION_DECISION.into(),
        canonical_model: artifact.canonical_model.clone(),
        task: artifact.task.into(),
        campaign_source_sha: capture.source_sha.clone(),
        campaign_runtime_code_identity_sha256: capture.runtime_code_identity_sha256.clone(),
        campaign_bootstrap_record_sha256: capture.bootstrap_runtime_record_sha256.clone(),
        campaign_bootstrap_identity_sha256: capture
            .bootstrap_runtime_qualification_identity_sha256
            .clone(),
        measured_server_executable_relative_path: measured_server_executable.relative_path.clone(),
        measured_server_executable_sha256: measured_server_executable.sha256.clone(),
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
        campaign_process: capture.process,
        envelope: capture.envelope,
        bounds,
        evidence_artifacts,
        identity_sha256: String::new(),
    };
    record.identity_sha256 = runtime_qualification_identity(&record);
    finish_candidate(record)
}

fn validate_candidate_artifact_qualification(
    artifact: &H3PrivateArtifactQualificationReport,
    public_runtime: bool,
) -> Result<()> {
    let (expected_claim, expected_decision, expected_scope, expected_activation) = if public_runtime
    {
        (
            "mold.minimax-h3.public-artifact-reader.v1",
            "verified-public-artifacts",
            "public-h3-integration",
            "supported-compact-fl2va-cuda",
        )
    } else {
        (
            H3_PRIVATE_UAT_CLAIM_MARKER,
            "qualified-private-artifacts",
            H3_PRIVATE_AUTHORIZATION_SCOPE,
            "rejected",
        )
    };
    if artifact.schema != QUALIFICATION_SCHEMA
        || artifact.claim_marker != expected_claim
        || artifact.decision != expected_decision
        || artifact.authorization_scope != expected_scope
        || artifact.canonical_model != minimax_h3::FL2VA_COMFY
        || artifact.task != "fl2va"
        || artifact.total_bytes == 0
        || artifact.runtime_constructed
        || artifact.generated_media
        || artifact.public_activation != expected_activation
        || !valid_lower_sha256(&artifact.authorization_record_sha256)
        || !valid_lower_sha256(&artifact.authorization_source_document_sha256)
        || !valid_lower_sha256(&artifact.qualification_identity_sha256)
    {
        bail!("private H3 runtime candidate lacks exact artifact qualification")
    }
    Ok(())
}

fn finish_candidate(
    record: H3PrivateRuntimeQualificationRecord,
) -> Result<H3PrivateRuntimeQualificationCandidate> {
    validate_runtime_qualification_record_shape(&record)?;
    let mut json_bytes = serde_json::to_vec_pretty(&record)?;
    json_bytes.push(b'\n');
    let json_len = u64::try_from(json_bytes.len())
        .map_err(|_| anyhow!("private H3 runtime candidate length exceeded u64"))?;
    if json_len > MAX_RUNTIME_QUALIFICATION_BYTES {
        bail!("private H3 runtime candidate exceeds the activation record limit")
    }
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
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
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

fn hash_measured_server_executable(
    root: &Path,
    relative_path: &str,
    source_sha: &str,
    runtime_code_identity_sha256: &str,
) -> Result<H3PrivateRuntimeEvidenceArtifact> {
    if !valid_lower_hex(source_sha, 40) || !valid_lower_sha256(runtime_code_identity_sha256) {
        bail!("private H3 measured server requires exact embedded identities")
    }
    validate_relative_path(relative_path, "measured server executable")?;
    let path = root.join(relative_path);
    validate_private_parent_chain(root, &path)?;
    let mut file = open_regular_file_no_follow(&path)
        .with_context(|| format!("failed to open private H3 measured server {relative_path}"))?;
    let before = require_private_file(&file, relative_path)?;
    if before.len == 0 || before.len > MAX_EVIDENCE_ARTIFACT_BYTES {
        bail!("private H3 measured server {relative_path} has an invalid size")
    }

    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)
        .context("failed to read private H3 measured server header")?;
    if magic != *b"\x7fELF" {
        bail!("private H3 measured server evidence is not an ELF executable")
    }
    file.seek(SeekFrom::Start(0))?;
    let markers = [
        ("source SHA", source_sha.as_bytes()),
        (
            "runtime-code identity",
            runtime_code_identity_sha256.as_bytes(),
        ),
    ];
    verify_embedded_markers(&mut file, &markers)?;
    let sha256 = sha256_open_file(&file)?;

    let after = EvidenceFileIdentity::from_metadata(&file.metadata()?);
    let current = open_regular_file_no_follow(&path)?;
    let current_identity = EvidenceFileIdentity::from_metadata(&current.metadata()?);
    if before != after || before != current_identity || sha256_open_file(&current)? != sha256 {
        bail!("private H3 measured server {relative_path} changed while authenticating")
    }
    Ok(H3PrivateRuntimeEvidenceArtifact {
        relative_path: relative_path.into(),
        bytes: before.len,
        sha256,
    })
}

fn verify_embedded_markers(file: &mut File, markers: &[(&str, &[u8])]) -> Result<()> {
    let prefixes = markers
        .iter()
        .map(|(_, marker)| marker_prefix(marker))
        .collect::<Vec<_>>();
    let mut states = vec![0_usize; markers.len()];
    let mut found = vec![false; markers.len()];
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            for (index, (_, marker)) in markers.iter().enumerate() {
                if found[index] {
                    continue;
                }
                while states[index] > 0 && marker[states[index]] != *byte {
                    states[index] = prefixes[index][states[index] - 1];
                }
                if marker[states[index]] == *byte {
                    states[index] += 1;
                    if states[index] == marker.len() {
                        found[index] = true;
                        states[index] = prefixes[index][states[index] - 1];
                    }
                }
            }
        }
    }
    for ((label, _), found) in markers.iter().zip(found) {
        if !found {
            bail!("private H3 measured server does not embed its campaign {label}")
        }
    }
    Ok(())
}

fn marker_prefix(marker: &[u8]) -> Vec<usize> {
    let mut prefix = vec![0_usize; marker.len()];
    let mut matched = 0_usize;
    for index in 1..marker.len() {
        while matched > 0 && marker[matched] != marker[index] {
            matched = prefix[matched - 1];
        }
        if marker[matched] == marker[index] {
            matched += 1;
            prefix[index] = matched;
        }
    }
    prefix
}

fn read_authenticated_evidence_artifact(
    root: &Path,
    relative_path: &str,
    limit: u64,
) -> Result<(H3PrivateRuntimeEvidenceArtifact, Vec<u8>)> {
    validate_relative_path(relative_path, "runtime evidence")?;
    let path = root.join(relative_path);
    validate_private_parent_chain(root, &path)?;
    let mut file = open_regular_file_no_follow(&path)?;
    let before = require_private_file(&file, relative_path)?;
    if before.len == 0 || before.len > limit {
        bail!("private H3 runtime evidence {relative_path} has an invalid size")
    }
    let mut bytes = Vec::with_capacity(before.len as usize);
    file.read_to_end(&mut bytes)?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let after = EvidenceFileIdentity::from_metadata(&file.metadata()?);
    let current = open_regular_file_no_follow(&path)?;
    let current_identity = EvidenceFileIdentity::from_metadata(&current.metadata()?);
    if before != after || before != current_identity || sha256_open_file(&current)? != sha256 {
        bail!("private H3 runtime evidence {relative_path} changed while reading")
    }
    Ok((
        H3PrivateRuntimeEvidenceArtifact {
            relative_path: relative_path.into(),
            bytes: before.len,
            sha256,
        },
        bytes,
    ))
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
    valid_lower_hex(value, 64)
}

fn validate_process_observation(process: &H3PrivateRuntimeProcessObservation) -> Result<()> {
    if process.process_id == 0
        || process.process_start_time_ticks == 0
        || process.executable_device == 0
        || process.executable_inode == 0
        || process.executable_bytes == 0
        || process.cuda_driver_version == 0
        || process.cuda_toolkit_version == 0
        || [
            process.linux_boot_id_sha256.as_str(),
            process.executable_sha256.as_str(),
            process.launch_argv_sha256.as_str(),
            process.launch_environment_sha256.as_str(),
        ]
        .into_iter()
        .any(|value| !valid_lower_sha256(value))
    {
        bail!("private H3 process attestation is incomplete")
    }
    Ok(())
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const DEVICE_0: &str = "cuda:00000000000000000000000000000000";

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn source_sha() -> String {
        std::iter::repeat_n('d', 40).collect()
    }

    fn runtime_code_identity() -> String {
        sha('f')
    }

    fn artifact_report() -> H3PrivateArtifactQualificationReport {
        H3PrivateArtifactQualificationReport {
            schema: "mold.minimax-h3.private-artifact-qualification.v2",
            claim_marker: H3_PRIVATE_UAT_CLAIM_MARKER,
            decision: "qualified-private-artifacts",
            authorization_scope: H3_PRIVATE_AUTHORIZATION_SCOPE,
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

    #[test]
    fn candidate_artifact_authority_matches_the_compiled_runtime() {
        let mut private = artifact_report();
        validate_candidate_artifact_qualification(&private, false).unwrap();
        for field in ["decision", "claim", "scope", "activation"] {
            let mut changed = private.clone();
            match field {
                "decision" => changed.decision = "verified-public-artifacts",
                "claim" => changed.claim_marker = "mold.minimax-h3.public-artifact-reader.v1",
                "scope" => changed.authorization_scope = "public-h3-integration",
                "activation" => changed.public_activation = "supported-compact-fl2va-cuda",
                _ => unreachable!(),
            }
            assert!(validate_candidate_artifact_qualification(&changed, false).is_err());
        }

        private.decision = "verified-public-artifacts";
        private.claim_marker = "mold.minimax-h3.public-artifact-reader.v1";
        private.authorization_scope = "public-h3-integration";
        private.public_activation = "supported-compact-fl2va-cuda";
        validate_candidate_artifact_qualification(&private, true).unwrap();
        for field in ["decision", "claim", "scope", "activation"] {
            let mut changed = private.clone();
            match field {
                "decision" => changed.decision = "qualified-private-artifacts",
                "claim" => changed.claim_marker = H3_PRIVATE_UAT_CLAIM_MARKER,
                "scope" => changed.authorization_scope = H3_PRIVATE_AUTHORIZATION_SCOPE,
                "activation" => changed.public_activation = "rejected",
                _ => unreachable!(),
            }
            assert!(validate_candidate_artifact_qualification(&changed, true).is_err());
        }
    }

    fn bound(path: &str, value: u64) -> H3PrivateRuntimeBoundCapture {
        H3PrivateRuntimeBoundCapture {
            observed_bytes: value,
            bound_bytes: value + 1,
            evidence_artifact: path.into(),
        }
    }

    fn capped_bound(
        path: &str,
        observed_bytes: u64,
        bound_bytes: u64,
    ) -> H3PrivateRuntimeBoundCapture {
        H3PrivateRuntimeBoundCapture {
            observed_bytes,
            bound_bytes,
            evidence_artifact: path.into(),
        }
    }

    fn envelope() -> H3PrivateRuntimeEnvelopeRecord {
        H3PrivateRuntimeEnvelopeRecord {
            width: minimax_h3::DEFAULT_WIDTH,
            height: minimax_h3::DEFAULT_HEIGHT,
            frames: minimax_h3::MIN_FRAMES,
            fps: minimax_h3::FIXED_FPS,
            batch_size: 1,
            max_steps: minimax_h3::COMFY_DEFAULT_STEPS,
            endpoint_count: 1,
            endpoint_anchor: "first".into(),
            max_qwen_output_text_rows: 128,
            max_qwen_vision_rows: 1_024,
            max_condition_visual_rows: 1_024,
            max_target_video_rows: 16_384,
            max_target_audio_rows: 1_024,
            max_total_packed_rows: 19_560,
        }
    }

    fn process() -> H3PrivateRuntimeProcessObservation {
        H3PrivateRuntimeProcessObservation {
            process_id: 42,
            process_start_time_ticks: 99,
            linux_boot_id_sha256: sha('1'),
            executable_device: 7,
            executable_inode: 8,
            executable_bytes: 1,
            executable_sha256: sha('4'),
            launch_argv_sha256: sha('2'),
            launch_environment_sha256: sha('3'),
            cuda_driver_version: 12_080,
            cuda_toolkit_version: 12_080,
        }
    }

    fn capture(path: &str) -> H3PrivateRuntimeBoundCaptureManifest {
        H3PrivateRuntimeBoundCaptureManifest {
            schema: CAPTURE_SCHEMA.into(),
            canonical_model: minimax_h3::FL2VA_COMFY.into(),
            task: "fl2va".into(),
            source_sha: source_sha(),
            runtime_code_identity_sha256: runtime_code_identity(),
            bootstrap_runtime_record_sha256: sha('f'),
            bootstrap_runtime_qualification_identity_sha256: sha('9'),
            measured_server_executable: "bin/mold-server".into(),
            authorization_record_sha256: sha('a'),
            authorization_source_document_sha256: sha('b'),
            artifact_qualification_identity_sha256: sha('c'),
            device_id: DEVICE_0.into(),
            device_ordinal: 0,
            compute_capability: [8, 9],
            attention_runtime_identity_sha256: sha('d'),
            attention_kernel_identity: "h3-flash-attention-sm89".into(),
            attention_qualification_sha256: sha('e'),
            process: process(),
            runtime_observation_artifact: path.into(),
            envelope: envelope(),
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
                encoded_video_host_bytes_bound: capped_bound(
                    path,
                    10,
                    super::super::pipeline::SMALL_ENCODED_VIDEO_HOST_BYTES_BOUND,
                ),
                thumbnail_host_bytes_bound: capped_bound(
                    path,
                    11,
                    super::super::pipeline::SMALL_THUMBNAIL_HOST_BYTES_BOUND,
                ),
                mux_output_host_bytes_bound: capped_bound(
                    path,
                    12,
                    super::super::pipeline::SMALL_MUX_OUTPUT_HOST_BYTES_BOUND,
                ),
                aac_mux_staging_host_bytes: capped_bound(
                    path,
                    13,
                    super::super::pipeline::SMALL_AAC_MUX_STAGING_HOST_BYTES,
                ),
            },
            evidence_artifacts: vec!["bin/mold-server".into(), path.into()],
        }
    }

    fn observation(
        capture: &H3PrivateRuntimeBoundCaptureManifest,
    ) -> H3PrivateRuntimeBoundObservation {
        H3PrivateRuntimeBoundObservation {
            schema: H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA.into(),
            authority: H3PrivateRuntimeAuthorityObservation {
                bootstrap_record_sha256: capture.bootstrap_runtime_record_sha256.clone(),
                runtime_qualification_identity_sha256: capture
                    .bootstrap_runtime_qualification_identity_sha256
                    .clone(),
                device_id: capture.device_id.clone(),
                device_ordinal: capture.device_ordinal,
                compute_capability: capture.compute_capability,
                attention_runtime_identity_sha256: capture
                    .attention_runtime_identity_sha256
                    .clone(),
                attention_kernel_identity: capture.attention_kernel_identity.clone(),
                attention_qualification_sha256: capture.attention_qualification_sha256.clone(),
                process: capture.process.clone(),
            },
            envelope: super::super::private_runtime_observer::H3PrivateRuntimeEnvelopeObservation {
                width: capture.envelope.width,
                height: capture.envelope.height,
                frames: capture.envelope.frames,
                fps: capture.envelope.fps,
                batch_size: capture.envelope.batch_size,
                steps: capture.envelope.max_steps,
                endpoint_count: capture.envelope.endpoint_count,
                endpoint_anchor: capture.envelope.endpoint_anchor.clone(),
                qwen_output_text_rows: capture.envelope.max_qwen_output_text_rows,
                qwen_vision_rows: capture.envelope.max_qwen_vision_rows,
                condition_visual_rows: capture.envelope.max_condition_visual_rows,
                target_video_rows: capture.envelope.max_target_video_rows,
                target_audio_rows: capture.envelope.max_target_audio_rows,
                total_packed_rows: capture.envelope.max_total_packed_rows,
            },
            fixed_runtime_host_bytes: capture.bounds.fixed_runtime_host_bytes.observed_bytes,
            fixed_runtime_device_bytes: capture.bounds.fixed_runtime_device_bytes.observed_bytes,
            qwen_activation_workspace_bytes: capture
                .bounds
                .qwen_activation_workspace_bytes
                .observed_bytes,
            vae_construction_device_workspace_bytes: capture
                .bounds
                .vae_construction_device_workspace_bytes
                .observed_bytes,
            condition_vae_workspace_device_bytes: capture
                .bounds
                .condition_vae_workspace_device_bytes
                .observed_bytes,
            attention_workspace_device_bytes: capture
                .bounds
                .attention_workspace_device_bytes
                .observed_bytes,
            ffn_workspace_device_bytes: capture.bounds.ffn_workspace_device_bytes.observed_bytes,
            decoder_tile_workspace_device_bytes: capture
                .bounds
                .decoder_tile_workspace_device_bytes
                .observed_bytes,
            audio_decode_workspace_device_bytes: capture
                .bounds
                .audio_decode_workspace_device_bytes
                .observed_bytes,
            encoded_video_host_bytes_bound: capture
                .bounds
                .encoded_video_host_bytes_bound
                .observed_bytes,
            thumbnail_host_bytes_bound: capture.bounds.thumbnail_host_bytes_bound.observed_bytes,
            mux_output_host_bytes_bound: capture.bounds.mux_output_host_bytes_bound.observed_bytes,
            aac_mux_staging_host_bytes: capture.bounds.aac_mux_staging_host_bytes.observed_bytes,
        }
    }

    #[test]
    fn structured_observation_binds_every_envelope_axis_and_runtime_measurement() {
        let capture = capture("logs/runtime-observation.json");
        let reviewed = &capture.envelope;
        let observation = observation(&capture);
        validate_observed_envelope(reviewed, &observation).unwrap();
        capture.bounds.validate_observation(&observation).unwrap();

        macro_rules! reject_envelope_change {
            ($field:ident) => {{
                let mut changed = observation.clone();
                changed.envelope.$field += 1;
                assert!(validate_observed_envelope(reviewed, &changed).is_err());
            }};
        }
        reject_envelope_change!(width);
        reject_envelope_change!(height);
        reject_envelope_change!(frames);
        reject_envelope_change!(fps);
        reject_envelope_change!(batch_size);
        reject_envelope_change!(steps);
        reject_envelope_change!(endpoint_count);
        reject_envelope_change!(qwen_output_text_rows);
        reject_envelope_change!(qwen_vision_rows);
        reject_envelope_change!(condition_visual_rows);
        reject_envelope_change!(target_video_rows);
        reject_envelope_change!(target_audio_rows);
        reject_envelope_change!(total_packed_rows);
        let mut changed = observation.clone();
        changed.envelope.endpoint_anchor = "last".into();
        assert!(validate_observed_envelope(reviewed, &changed).is_err());

        macro_rules! reject_measurement_change {
            ($field:ident) => {{
                let mut changed = observation.clone();
                changed.$field += 1;
                assert!(capture.bounds.validate_observation(&changed).is_err());
            }};
        }
        reject_measurement_change!(fixed_runtime_host_bytes);
        reject_measurement_change!(fixed_runtime_device_bytes);
        reject_measurement_change!(qwen_activation_workspace_bytes);
        reject_measurement_change!(vae_construction_device_workspace_bytes);
        reject_measurement_change!(condition_vae_workspace_device_bytes);
        reject_measurement_change!(attention_workspace_device_bytes);
        reject_measurement_change!(ffn_workspace_device_bytes);
        reject_measurement_change!(decoder_tile_workspace_device_bytes);
        reject_measurement_change!(audio_decode_workspace_device_bytes);
        reject_measurement_change!(encoded_video_host_bytes_bound);
        reject_measurement_change!(thumbnail_host_bytes_bound);
        reject_measurement_change!(mux_output_host_bytes_bound);
        reject_measurement_change!(aac_mux_staging_host_bytes);
    }

    #[test]
    fn structured_observation_binds_every_runtime_authority_axis() {
        let capture = capture("logs/runtime-observation.json");
        let observation = observation(&capture);
        validate_observed_authority(&capture, &observation).unwrap();

        macro_rules! reject_string_change {
            ($field:ident) => {{
                let mut changed = observation.clone();
                changed.authority.$field = sha('0');
                assert!(validate_observed_authority(&capture, &changed).is_err());
            }};
        }
        reject_string_change!(bootstrap_record_sha256);
        reject_string_change!(runtime_qualification_identity_sha256);
        reject_string_change!(device_id);
        reject_string_change!(attention_runtime_identity_sha256);
        reject_string_change!(attention_kernel_identity);
        reject_string_change!(attention_qualification_sha256);
        let mut changed = observation.clone();
        changed.authority.device_ordinal += 1;
        assert!(validate_observed_authority(&capture, &changed).is_err());
        let mut changed = observation.clone();
        changed.authority.compute_capability[0] += 1;
        assert!(validate_observed_authority(&capture, &changed).is_err());

        macro_rules! reject_process_number_change {
            ($field:ident) => {{
                let mut changed = observation.clone();
                changed.authority.process.$field += 1;
                assert!(validate_observed_authority(&capture, &changed).is_err());
            }};
        }
        reject_process_number_change!(process_id);
        reject_process_number_change!(process_start_time_ticks);
        reject_process_number_change!(executable_device);
        reject_process_number_change!(executable_inode);
        reject_process_number_change!(executable_bytes);
        reject_process_number_change!(cuda_driver_version);
        reject_process_number_change!(cuda_toolkit_version);
        for field in [
            "linux_boot_id_sha256",
            "executable_sha256",
            "launch_argv_sha256",
            "launch_environment_sha256",
        ] {
            let mut changed = observation.clone();
            match field {
                "linux_boot_id_sha256" => changed.authority.process.linux_boot_id_sha256 = sha('0'),
                "executable_sha256" => changed.authority.process.executable_sha256 = sha('0'),
                "launch_argv_sha256" => changed.authority.process.launch_argv_sha256 = sha('0'),
                "launch_environment_sha256" => {
                    changed.authority.process.launch_environment_sha256 = sha('0')
                }
                _ => unreachable!(),
            }
            assert!(validate_observed_authority(&capture, &changed).is_err());
        }
    }

    #[cfg(unix)]
    fn private_fixture() -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let bin = root.path().join("bin");
        fs::create_dir(&bin).unwrap();
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o700)).unwrap();
        let executable = bin.join("mold-server");
        let executable_bytes = format!(
            "\x7fELF\0fixture\0{}\0{}\0",
            source_sha(),
            runtime_code_identity()
        );
        fs::write(&executable, executable_bytes.as_bytes()).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o600)).unwrap();
        let logs = root.path().join("logs");
        fs::create_dir(&logs).unwrap();
        fs::set_permissions(&logs, fs::Permissions::from_mode(0o700)).unwrap();
        let evidence = logs.join("runtime-observation.json");
        let mut capture = capture("logs/runtime-observation.json");
        capture.process.executable_bytes = executable_bytes.len() as u64;
        capture.process.executable_sha256 = format!("{:x}", Sha256::digest(executable_bytes));
        fs::write(
            &evidence,
            serde_json::to_vec_pretty(&observation(&capture)).unwrap(),
        )
        .unwrap();
        fs::set_permissions(&evidence, fs::Permissions::from_mode(0o600)).unwrap();
        let capture_path = root.path().join("capture.json");
        fs::write(&capture_path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
        fs::set_permissions(&capture_path, fs::Permissions::from_mode(0o600)).unwrap();
        (root, capture_path, evidence, executable)
    }

    #[cfg(unix)]
    #[test]
    fn candidate_is_deterministic_and_accepted_by_the_runtime_shape_validator() {
        let (root, capture_path, _, _) = private_fixture();
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
            &source_sha(),
            &runtime_code_identity(),
        )
        .unwrap();
        let second = build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative,
            capture_artifact,
            manifest,
            &source_sha(),
            &runtime_code_identity(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.evidence_artifact_count(), 3);
        assert!(valid_lower_sha256(first.record_file_sha256()));
        let record: H3PrivateRuntimeQualificationRecord =
            serde_json::from_slice(first.json_bytes()).unwrap();
        validate_runtime_qualification_record_shape(&record).unwrap();
        assert_eq!(record.identity_sha256, first.identity_sha256());
    }

    #[cfg(unix)]
    #[test]
    fn candidate_rejects_an_unmeasured_or_unretained_bound() {
        let (root, capture_path, _, _) = private_fixture();
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
            &source_sha(),
            &runtime_code_identity(),
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
            &source_sha(),
            &runtime_code_identity(),
        )
        .unwrap_err()
        .to_string()
        .contains("unretained evidence"));
    }

    #[test]
    fn candidate_accepts_zero_only_for_retained_vae_construction_growth() {
        let mut manifest = capture("logs/runtime.log");
        let evidence = manifest.evidence_artifacts.iter().cloned().collect();
        manifest
            .bounds
            .vae_construction_device_workspace_bytes
            .observed_bytes = 0;
        assert_eq!(
            manifest
                .bounds
                .validate(&evidence)
                .unwrap()
                .vae_construction_device_workspace_bytes,
            5
        );

        manifest
            .bounds
            .audio_decode_workspace_device_bytes
            .observed_bytes = 0;
        assert!(manifest
            .bounds
            .validate(&evidence)
            .unwrap_err()
            .to_string()
            .contains("audio_decode_workspace_device_bytes"));
    }

    #[cfg(unix)]
    #[test]
    fn evidence_permissions_and_order_fail_closed() {
        use std::os::unix::fs::PermissionsExt;

        let (root, capture_path, evidence, _) = private_fixture();
        fs::set_permissions(&evidence, fs::Permissions::from_mode(0o640)).unwrap();
        assert!(
            hash_evidence_artifact(root.path(), "logs/runtime-observation.json", None)
                .unwrap_err()
                .to_string()
                .contains("owner-only")
        );

        let mut manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(capture_path).unwrap()).unwrap();
        manifest.evidence_artifacts = vec!["logs/z.log".into(), "logs/a.log".into()];
        assert!(manifest
            .validate(&artifact_report(), &source_sha(), &runtime_code_identity())
            .unwrap_err()
            .to_string()
            .contains("sorted and unique"));
    }

    #[cfg(unix)]
    #[test]
    fn capture_rejects_unembedded_source_or_invalid_cuda_authority() {
        let mut manifest = capture("logs/runtime-observation.json");
        assert!(manifest
            .validate(
                &artifact_report(),
                &std::iter::repeat_n('e', 40).collect::<String>(),
                &runtime_code_identity()
            )
            .unwrap_err()
            .to_string()
            .contains("embedded campaign build"));

        manifest.compute_capability = [100, 0];
        assert!(manifest
            .validate(&artifact_report(), &source_sha(), &runtime_code_identity())
            .unwrap_err()
            .to_string()
            .contains("CUDA attention authority"));

        manifest.compute_capability = [8, 9];
        manifest.device_id = "cuda:0".into();
        assert!(manifest
            .validate(&artifact_report(), &source_sha(), &runtime_code_identity())
            .unwrap_err()
            .to_string()
            .contains("CUDA attention authority"));

        manifest.device_id = DEVICE_0.into();
        manifest.envelope.frames += 1;
        assert!(manifest
            .validate(&artifact_report(), &source_sha(), &runtime_code_identity())
            .unwrap_err()
            .to_string()
            .contains("compact-quality envelope"));
    }

    #[cfg(unix)]
    #[test]
    fn candidate_rejects_a_measured_server_without_embedded_identity() {
        use std::os::unix::fs::PermissionsExt;

        let (root, capture_path, _, executable) = private_fixture();
        fs::write(&executable, b"\x7fELF\0unbound-server\0").unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o600)).unwrap();
        let capture_relative = relative_evidence_path(root.path(), &capture_path).unwrap();
        let capture_artifact =
            hash_evidence_artifact(root.path(), &capture_relative, Some(MAX_CAPTURE_BYTES))
                .unwrap();
        let manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(&capture_path).unwrap()).unwrap();
        let error = build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative,
            capture_artifact,
            manifest,
            &source_sha(),
            &runtime_code_identity(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not embed"), "{error:#}");
    }

    #[cfg(unix)]
    #[test]
    fn candidate_rejects_coordinated_process_and_manifest_executable_substitution() {
        use std::os::unix::fs::PermissionsExt;

        let (root, capture_path, observation_path, _) = private_fixture();
        let mut manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(&capture_path).unwrap()).unwrap();
        manifest.process.executable_sha256 = sha('0');
        let mut observed = observation(&manifest);
        observed.authority.process.executable_sha256 = sha('0');
        fs::write(
            &observation_path,
            serde_json::to_vec_pretty(&observed).unwrap(),
        )
        .unwrap();
        fs::set_permissions(&observation_path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&capture_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        fs::set_permissions(&capture_path, fs::Permissions::from_mode(0o600)).unwrap();
        let capture_relative = relative_evidence_path(root.path(), &capture_path).unwrap();
        let capture_artifact =
            hash_evidence_artifact(root.path(), &capture_relative, Some(MAX_CAPTURE_BYTES))
                .unwrap();
        let error = build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative,
            capture_artifact,
            manifest,
            &source_sha(),
            &runtime_code_identity(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("retained server executable"),
            "{error:#}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn candidate_emission_rejects_records_over_the_activation_limit() {
        let (root, capture_path, _, _) = private_fixture();
        let capture_relative = relative_evidence_path(root.path(), &capture_path).unwrap();
        let capture_artifact =
            hash_evidence_artifact(root.path(), &capture_relative, Some(MAX_CAPTURE_BYTES))
                .unwrap();
        let manifest: H3PrivateRuntimeBoundCaptureManifest =
            serde_json::from_slice(&fs::read(&capture_path).unwrap()).unwrap();
        let candidate = build_candidate(
            &artifact_report(),
            root.path(),
            capture_relative,
            capture_artifact,
            manifest,
            &source_sha(),
            &runtime_code_identity(),
        )
        .unwrap();
        let mut record: H3PrivateRuntimeQualificationRecord =
            serde_json::from_slice(candidate.json_bytes()).unwrap();
        for index in 0..128 {
            record
                .evidence_artifacts
                .push(H3PrivateRuntimeEvidenceArtifact {
                    relative_path: format!("oversized/{index:03}-{}", "x".repeat(1_024)),
                    bytes: 1,
                    sha256: sha('9'),
                });
        }
        record
            .evidence_artifacts
            .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        record.identity_sha256 = runtime_qualification_identity(&record);
        let error = finish_candidate(record).unwrap_err();
        assert!(error.to_string().contains("activation record limit"));
    }
}
