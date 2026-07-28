//! Durable chain-job manifest and job-directory layout.
//!
//! A chain job is a persistent, resumable chained-video generation. Its
//! portable source of truth is `manifest.toml` (schema `mold.chainjob.v1`)
//! inside a self-contained job directory; `mold.db` holds a queryable index
//! of the same state (see `mold_db::chain_jobs`). When the two disagree,
//! the manifest wins. See `docs/superpowers/specs/2026-07-03-durable-chain-jobs-design.md` §3.
//!
//! The embedded generation request is stored as canonical JSON
//! (`request_json`), not TOML tables: TOML 0.8 integers are i64-limited and
//! full-range u64 seeds (`base ^ seed_offset`) would abort the manifest
//! write. JSON is also the exact encoding the DB row uses, so both stores
//! share one canonical request serialization.

use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chain::ChainRequest;
use crate::error::{MoldError, Result};

/// Manifest schema identifier. `from_toml` rejects anything else.
pub const CHAIN_JOB_SCHEMA: &str = "mold.chainjob.v1";

// ── Job-dir layout: fixed relative names (spec §3.2) ──────────────────

pub const MANIFEST_FILE: &str = "manifest.toml";
pub const STAGES_DIR: &str = "stages";
pub const FINAL_DIR: &str = "final";
pub const SEGMENT_FILE: &str = "segment.mp4";
pub const TAIL_DIR: &str = "tail";
pub const BOUNDARY_IN_DIR: &str = "boundary-in";
pub const BOUNDARY_OUT_DIR: &str = "boundary-out";
pub const AUDIO_FILE: &str = "audio.pcm";
pub const PREVIEW_FILE: &str = "preview.jpg";

// ── State enums (canonical here; mold-db imports them) ────────────────

/// Job lifecycle state, stored as snake_case TEXT in `chain_jobs.state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChainJobState {
    Queued,
    Running,
    Interrupted,
    Failed,
    Completed,
    Cancelled,
}

/// Per-stage state, stored as snake_case TEXT in `chain_job_stages.state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Retake mode (spec §8): cascade re-renders N..end, splice re-renders N only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum RetakeMode {
    Cascade,
    Splice,
}

impl ChainJobState {
    pub fn as_str(self) -> &'static str {
        match self {
            ChainJobState::Queued => "queued",
            ChainJobState::Running => "running",
            ChainJobState::Interrupted => "interrupted",
            ChainJobState::Failed => "failed",
            ChainJobState::Completed => "completed",
            ChainJobState::Cancelled => "cancelled",
        }
    }

    /// terminal = "excluded from startup reconcile and will never transition again;
    /// only Completed. Failed/Interrupted/Cancelled are resumable (cancel is
    /// pause-with-intent, spec §6)."
    pub fn is_terminal(self) -> bool {
        matches!(self, ChainJobState::Completed)
    }
}

/// Settled jobs have reached a durable outcome for this attempt:
/// completed, failed, or cancelled.
///
/// This is deliberately broader than [`ChainJobState::is_terminal`]. Only
/// `completed` is terminal in the resume/reconcile sense; `failed` and
/// `cancelled` are settled for cancel/bus cleanup but may be re-queued by
/// resume or retake.
pub fn settled(state: ChainJobState) -> bool {
    matches!(
        state,
        ChainJobState::Completed | ChainJobState::Failed | ChainJobState::Cancelled
    )
}

impl std::str::FromStr for ChainJobState {
    type Err = MoldError;

    fn from_str(s: &str) -> std::result::Result<Self, MoldError> {
        match s {
            "queued" => Ok(ChainJobState::Queued),
            "running" => Ok(ChainJobState::Running),
            "interrupted" => Ok(ChainJobState::Interrupted),
            "failed" => Ok(ChainJobState::Failed),
            "completed" => Ok(ChainJobState::Completed),
            "cancelled" => Ok(ChainJobState::Cancelled),
            other => Err(MoldError::Validation(format!(
                "unknown chain job state '{other}'"
            ))),
        }
    }
}

impl StageState {
    pub fn as_str(self) -> &'static str {
        match self {
            StageState::Pending => "pending",
            StageState::Running => "running",
            StageState::Completed => "completed",
            StageState::Failed => "failed",
        }
    }
}

impl std::str::FromStr for StageState {
    type Err = MoldError;

    fn from_str(s: &str) -> std::result::Result<Self, MoldError> {
        match s {
            "pending" => Ok(StageState::Pending),
            "running" => Ok(StageState::Running),
            "completed" => Ok(StageState::Completed),
            "failed" => Ok(StageState::Failed),
            other => Err(MoldError::Validation(format!(
                "unknown chain job stage state '{other}'"
            ))),
        }
    }
}

impl RetakeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            RetakeMode::Cascade => "cascade",
            RetakeMode::Splice => "splice",
        }
    }
}

// ── Manifest types (spec §3.2, current) ───────────

/// Portable job description + per-stage status. Everything needed to
/// resume or retake lives here; all paths are relative to the job dir.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainJobManifest {
    /// Must equal [`CHAIN_JOB_SCHEMA`] on read.
    pub schema: String,
    pub job_id: String,
    pub created_at_unix_ms: u64,
    /// Sync-shim jobs (spec §9): artifacts deleted immediately after success.
    #[serde(default)]
    pub ephemeral: bool,
    /// Full normalised [`ChainRequest`], canonically serde_json-encoded.
    pub request_json: String,
    #[serde(default)]
    pub stage_status: Vec<StageStatus>,
    /// Retake amendment history (spec §8.3). The original request stays
    /// intact for provenance; edits are recorded here. An amend folds any
    /// pending retakes into the rewritten `request_json` and clears this
    /// list (their content lives on in the amend's request snapshot).
    #[serde(default)]
    pub retakes: Vec<RetakeAmendment>,
    /// Versioned finalize history (spec §8.3).
    #[serde(default)]
    pub finalizes: Vec<FinalizeRecord>,
    /// Amend history (spec §17): each entry snapshots the pre-amend
    /// EFFECTIVE request so any preserved stage remains attributable.
    #[serde(default)]
    pub amends: Vec<AmendRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageStatus {
    pub idx: u32,
    pub state: StageState,
    /// Effective seed (`base ^ seed_offset`); full-range u64, so encoded
    /// as a decimal string in TOML (i64-limited integers).
    #[serde(with = "u64_as_string")]
    pub seed: u64,
    pub frames_emitted: Option<u32>,
    pub generation_time_ms: Option<u64>,
    /// Relative to the job dir. Never absolute (portability contract).
    pub segment: Option<String>,
    pub tail_frames: Option<u32>,
    /// Relative path to the stage's PCM sidecar, when audio was rendered.
    pub audio: Option<String>,
    pub error: Option<String>,
    /// `true` for stages written under the raw-segment contract (2026-07-28):
    /// the segment holds every frame the engine emitted (no boundary trims or
    /// fade blends) and all boundary math is deferred to finalize. `false`
    /// (the pre-amend default) marks a legacy stage whose segment was trimmed
    /// and blended at write time and passes through finalize untouched.
    #[serde(default)]
    pub raw_segment: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RetakeAmendment {
    pub stage_idx: u32,
    pub mode: RetakeMode,
    #[serde(with = "u64_as_string")]
    pub old_seed: u64,
    #[serde(with = "u64_as_string")]
    pub new_seed: u64,
    /// Set only when the retake changed the prompt.
    pub old_prompt: Option<String>,
    pub new_prompt: Option<String>,
    pub at_unix_ms: u64,
}

/// One amend applied to a chain job: the full edited stage list replaced the
/// request's stages (plus optional chain-level overlays), and rendering
/// requeued from the earliest genuinely-dirty stage. `previous_request_json`
/// is the pre-amend EFFECTIVE request (retakes folded in), serialised with
/// the same canonical JSON encoding as `request_json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AmendRecord {
    pub at_unix_ms: u64,
    pub previous_request_json: String,
    pub preserved_stages: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct FinalizeRecord {
    /// Relative path under `final/`, e.g. `final/output-1.mp4`.
    pub output: String,
    pub at_unix_ms: u64,
    /// Per-stage effective seeds that produced this take (spec §8.3:
    /// every take attributable and reproducible).
    #[serde(with = "u64_vec_as_strings")]
    pub stage_seeds: Vec<u64>,
}

impl ChainJobManifest {
    /// Build a fresh manifest for a new job, serialising `request` into
    /// `request_json` (canonical JSON). `ephemeral` defaults to false;
    /// shim callers set the pub field directly.
    pub fn new(job_id: String, created_at_unix_ms: u64, request: &ChainRequest) -> Result<Self> {
        let request_json = serde_json::to_string(request).map_err(|e| {
            MoldError::Other(anyhow::anyhow!(
                "chain job request JSON serialise failed: {e}"
            ))
        })?;
        let base_seed = request.seed.unwrap_or(0);
        let stage_status = request
            .stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| StageStatus {
                idx: idx as u32,
                state: StageState::Pending,
                seed: effective_stage_seed(base_seed, stage.seed_offset),
                frames_emitted: None,
                generation_time_ms: None,
                segment: None,
                tail_frames: None,
                audio: None,
                error: None,
                raw_segment: false,
            })
            .collect();

        Ok(Self {
            schema: CHAIN_JOB_SCHEMA.into(),
            job_id,
            created_at_unix_ms,
            ephemeral: false,
            request_json,
            stage_status,
            retakes: vec![],
            finalizes: vec![],
            amends: vec![],
        })
    }

    /// Parsed-value access to the embedded request. Reconcile logic
    /// compares THIS, never raw `request_json` strings (field-order
    /// changes alter bytes without altering meaning).
    pub fn request(&self) -> Result<ChainRequest> {
        serde_json::from_str(&self.request_json)
            .map_err(|e| MoldError::Validation(format!("chain job request JSON parse failed: {e}")))
    }

    /// Parse a manifest, rejecting `schema != mold.chainjob.v1` with a
    /// clear error naming the supported schema (chain_toml.rs precedent).
    pub fn from_toml(s: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct SchemaPeek {
            schema: Option<String>,
        }

        let peek: SchemaPeek = toml::from_str(s).map_err(|e| {
            MoldError::Validation(format!("chain job manifest TOML parse failed: {e}"))
        })?;
        let schema = peek.schema.as_deref().unwrap_or("<missing>");
        if schema != CHAIN_JOB_SCHEMA {
            return Err(MoldError::Validation(format!(
                "chain job manifest schema '{schema}' is not supported by this mold version \
                 (supported: '{CHAIN_JOB_SCHEMA}')"
            )));
        }

        let manifest: Self = toml::from_str(s).map_err(|e| {
            MoldError::Validation(format!("chain job manifest TOML parse failed: {e}"))
        })?;
        manifest.validate_artifact_paths()?;
        Ok(manifest)
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string(self).map_err(|e| {
            MoldError::Other(anyhow::anyhow!(
                "chain job manifest TOML serialise failed: {e}"
            ))
        })
    }

    /// Write `<job_dir>/manifest.toml` via write-temp + rename so a crash
    /// mid-write never leaves a truncated manifest (spec §3.3 ordering:
    /// artifacts → manifest → DB).
    pub fn write_atomic(&self, job_dir: &Path) -> Result<()> {
        let manifest_path = job_dir.join(MANIFEST_FILE);
        let tmp_path = job_dir.join(format!("{MANIFEST_FILE}.tmp"));
        fs::write(&tmp_path, self.to_toml()?).map_err(|e| {
            MoldError::Other(anyhow::anyhow!(
                "writing chain job manifest temp file '{}': {e}",
                tmp_path.display()
            ))
        })?;
        fs::rename(&tmp_path, &manifest_path).map_err(|e| {
            MoldError::Other(anyhow::anyhow!(
                "renaming chain job manifest '{}' to '{}': {e}",
                tmp_path.display(),
                manifest_path.display()
            ))
        })?;
        Ok(())
    }

    pub fn read_from_dir(job_dir: &Path) -> Result<Self> {
        let manifest_path = job_dir.join(MANIFEST_FILE);
        let body = fs::read_to_string(&manifest_path).map_err(|e| {
            MoldError::Other(anyhow::anyhow!(
                "reading chain job manifest '{}': {e}",
                manifest_path.display()
            ))
        })?;
        Self::from_toml(&body)
    }

    fn validate_artifact_paths(&self) -> Result<()> {
        for (idx, stage) in self.stage_status.iter().enumerate() {
            if let Some(segment) = &stage.segment {
                validate_manifest_relative_path(&format!("stage_status[{idx}].segment"), segment)?;
            }
            if let Some(audio) = &stage.audio {
                validate_manifest_relative_path(&format!("stage_status[{idx}].audio"), audio)?;
            }
        }
        for (idx, finalize) in self.finalizes.iter().enumerate() {
            validate_manifest_relative_path(&format!("finalizes[{idx}].output"), &finalize.output)?;
        }
        Ok(())
    }
}

fn validate_manifest_relative_path(field: &str, value: &str) -> Result<()> {
    let path = Path::new(value);
    if is_manifest_absolute_path(path, value) || has_parent_path_component(path, value) {
        return Err(MoldError::Validation(format!(
            "chain job manifest {field} path '{value}' must be relative and must not contain '..'"
        )));
    }
    Ok(())
}

fn is_manifest_absolute_path(path: &Path, value: &str) -> bool {
    path.is_absolute()
        || value.starts_with('/')
        || value.starts_with('\\')
        || has_windows_drive_absolute_prefix(value)
}

fn has_windows_drive_absolute_prefix(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\')
}

fn has_parent_path_component(path: &Path, value: &str) -> bool {
    path.components()
        .any(|component| matches!(component, Component::ParentDir))
        || value.split(['/', '\\']).any(|component| component == "..")
}

// ── Job-directory layout helpers (pure path math + mkdir) ─────────────

/// Path helpers for one job directory. Pure path math except the
/// `ensure_*` mkdir helpers; all artifact paths follow spec §3.2.
pub struct JobDirLayout {
    root: PathBuf,
}

impl JobDirLayout {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE)
    }

    /// `stages/NNN/` with zero-padded 3-digit stage index.
    pub fn stage_dir(&self, idx: u32) -> PathBuf {
        self.root.join(STAGES_DIR).join(format!("{idx:03}"))
    }

    pub fn segment_path(&self, idx: u32) -> PathBuf {
        self.stage_dir(idx).join(SEGMENT_FILE)
    }

    pub fn tail_dir(&self, idx: u32) -> PathBuf {
        self.stage_dir(idx).join(TAIL_DIR)
    }

    pub fn boundary_in_dir(&self, idx: u32) -> PathBuf {
        self.stage_dir(idx).join(BOUNDARY_IN_DIR)
    }

    pub fn boundary_out_dir(&self, idx: u32) -> PathBuf {
        self.stage_dir(idx).join(BOUNDARY_OUT_DIR)
    }

    pub fn audio_path(&self, idx: u32) -> PathBuf {
        self.stage_dir(idx).join(AUDIO_FILE)
    }

    pub fn preview_path(&self, idx: u32) -> PathBuf {
        self.stage_dir(idx).join(PREVIEW_FILE)
    }

    /// `final/output-<n>.mp4`. `n` derives from `manifest.finalizes.len()`,
    /// never from a directory scan.
    pub fn final_output_path(&self, n: u32) -> PathBuf {
        self.root.join(FINAL_DIR).join(format!("output-{n}.mp4"))
    }

    /// Relative-to-root form for manifest fields (portability contract).
    pub fn segment_rel(&self, idx: u32) -> String {
        format!("{STAGES_DIR}/{idx:03}/{SEGMENT_FILE}")
    }

    pub fn audio_rel(&self, idx: u32) -> String {
        format!("{STAGES_DIR}/{idx:03}/{AUDIO_FILE}")
    }

    pub fn ensure_root(&self) -> Result<()> {
        fs::create_dir_all(&self.root).map_err(|e| {
            MoldError::Other(anyhow::anyhow!(
                "creating chain job root '{}': {e}",
                self.root.display()
            ))
        })
    }

    /// Create `stages/NNN/` plus its `tail/`, `boundary-in/`,
    /// `boundary-out/` subdirectories.
    pub fn ensure_stage_dirs(&self, idx: u32) -> Result<()> {
        for dir in [
            self.stage_dir(idx),
            self.tail_dir(idx),
            self.boundary_in_dir(idx),
            self.boundary_out_dir(idx),
        ] {
            fs::create_dir_all(&dir).map_err(|e| {
                MoldError::Other(anyhow::anyhow!(
                    "creating chain job stage directory '{}': {e}",
                    dir.display()
                ))
            })?;
        }
        Ok(())
    }
}

// ── serde helpers: full-range u64 as decimal strings in TOML ──────────

mod u64_as_string {
    use serde::de::Error as _;
    use serde::Deserialize;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &u64, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_str(v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<u64, D::Error> {
        let s = String::deserialize(d)?;
        s.parse::<u64>()
            .map_err(|_| D::Error::custom("expected u64 encoded as a decimal string"))
    }
}

mod u64_vec_as_strings {
    use serde::de::Error as _;
    use serde::ser::SerializeSeq;
    use serde::Deserialize;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec<u64>, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for seed in v {
            seq.serialize_element(&seed.to_string())?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<u64>, D::Error> {
        let strings = Vec::<String>::deserialize(d)?;
        strings
            .into_iter()
            .map(|s| {
                s.parse::<u64>()
                    .map_err(|_| D::Error::custom("expected u64 encoded as a decimal string"))
            })
            .collect()
    }
}

// ── Chain-job API wire types ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainJobSummary {
    pub id: String,
    pub state: ChainJobState,
    pub model: String,
    pub stage_count: u32,
    pub current_stage: u32,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
    pub error: Option<String>,
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainJobStageDetail {
    pub idx: u32,
    pub state: StageState,
    #[serde(with = "u64_as_string")]
    pub seed: u64,
    pub frames_emitted: Option<u32>,
    pub generation_time_ms: Option<u64>,
    pub has_preview: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainJobDetail {
    #[serde(flatten)]
    pub summary: ChainJobSummary,
    pub stages: Vec<ChainJobStageDetail>,
    pub finalizes: Vec<FinalizeRecord>,
    pub retakes: Vec<RetakeAmendment>,
    /// Amend history (additive; empty for never-amended jobs).
    #[serde(default)]
    pub amends: Vec<AmendRecord>,
    /// EFFECTIVE script (original request + retake amendments applied);
    /// provenance stays in `retakes`.
    pub script: crate::chain::ChainScript,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainJobListing {
    pub jobs: Vec<ChainJobSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CreateChainJobResponse {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RetakeRequest {
    pub stage_idx: u32,
    pub mode: RetakeMode,
    #[serde(default, with = "u64_opt_as_string")]
    pub seed_offset: Option<u64>,
    pub prompt: Option<String>,
}

/// Body of `POST /api/chain-jobs/:id/amend`: the FULL edited stage list (in
/// canonical order) replaces the job's stages, plus optional chain-level
/// overlays (omitted = keep current). NOT amendable — the client must create
/// a fresh job instead: model, width, height, output_format, placement,
/// strength, and batch provenance.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AmendRequest {
    pub stages: Vec<crate::chain::ChainStage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion_tail_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Full-range u64, encoded as a decimal string on the wire.
    #[serde(default, with = "u64_opt_as_string")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_audio: Option<bool>,
}

/// 202 body of `POST /api/chain-jobs/:id/amend`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AmendResponse {
    #[serde(flatten)]
    pub summary: ChainJobSummary,
    /// Leading stages whose cached artifacts were preserved; rendering
    /// requeues from this index.
    pub preserved_stages: u32,
}

/// Result shape for GC passes; also the JSON body of POST /api/chain-jobs/gc.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GcOutcome {
    pub swept_ephemeral_jobs: usize,
    pub pruned_artifact_dirs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[expect(
    clippy::large_enum_variant,
    reason = "approved wire contract keeps Snapshot inline for utoipa/serde shape"
)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChainJobEvent {
    Snapshot {
        job: ChainJobDetail,
    },
    StageStart {
        stage_idx: u32,
    },
    DenoiseStep {
        stage_idx: u32,
        step: u32,
        total: u32,
    },
    StageDone {
        stage_idx: u32,
        frames_emitted: u32,
        has_preview: bool,
    },
    Yielded {
        pending_small_jobs: usize,
    },
    Finalizing {
        total_frames: u32,
    },
    Finalized {
        output: String,
        take: u32,
    },
    StateChanged {
        state: ChainJobState,
        error: Option<String>,
    },
}

pub fn effective_stage_seed(base_seed: u64, seed_offset: Option<u64>) -> u64 {
    seed_offset.map_or(base_seed, |offset| base_seed ^ offset)
}

mod u64_opt_as_string {
    use serde::de::Error as _;
    use serde::Deserialize;

    pub fn serialize<S: serde::Serializer>(
        v: &Option<u64>,
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match v {
            Some(seed) => s.serialize_some(&seed.to_string()),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> std::result::Result<Option<u64>, D::Error> {
        let raw = Option::<String>::deserialize(d)?;
        raw.map(|s| {
            s.parse::<u64>()
                .map_err(|_| D::Error::custom("expected optional u64 encoded as a decimal string"))
        })
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{ChainStage, TransitionMode};
    use crate::types::OutputFormat;
    use std::str::FromStr;

    fn sample_stage(prompt: &str, seed_offset: Option<u64>) -> ChainStage {
        ChainStage {
            prompt: prompt.into(),
            frames: 97,
            source_image: None,
            negative_prompt: None,
            seed_offset,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        }
    }

    fn sample_request() -> ChainRequest {
        ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![
                sample_stage("stage zero", None),
                sample_stage("stage one", Some(u64::MAX - 3)),
            ],
            motion_tail_frames: 17,
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(42),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: Some(true),
        }
    }

    #[test]
    fn state_enums_round_trip_through_canonical_snake_case() {
        for (state, text) in [
            (ChainJobState::Queued, "queued"),
            (ChainJobState::Running, "running"),
            (ChainJobState::Interrupted, "interrupted"),
            (ChainJobState::Failed, "failed"),
            (ChainJobState::Completed, "completed"),
            (ChainJobState::Cancelled, "cancelled"),
        ] {
            assert_eq!(state.as_str(), text);
            assert_eq!(ChainJobState::from_str(text).unwrap(), state);
            assert_eq!(serde_json::to_value(state).unwrap(), text);
        }

        for (state, text) in [
            (StageState::Pending, "pending"),
            (StageState::Running, "running"),
            (StageState::Completed, "completed"),
            (StageState::Failed, "failed"),
        ] {
            assert_eq!(state.as_str(), text);
            assert_eq!(StageState::from_str(text).unwrap(), state);
            assert_eq!(serde_json::to_value(state).unwrap(), text);
        }

        for (mode, text) in [
            (RetakeMode::Cascade, "cascade"),
            (RetakeMode::Splice, "splice"),
        ] {
            assert_eq!(mode.as_str(), text);
            assert_eq!(serde_json::to_value(mode).unwrap(), text);
        }
    }

    #[test]
    fn only_completed_job_state_is_terminal() {
        assert!(!ChainJobState::Queued.is_terminal());
        assert!(!ChainJobState::Running.is_terminal());
        assert!(!ChainJobState::Interrupted.is_terminal());
        assert!(!ChainJobState::Failed.is_terminal());
        assert!(ChainJobState::Completed.is_terminal());
        assert!(!ChainJobState::Cancelled.is_terminal());
    }

    #[test]
    fn settled_states_are_distinct_from_terminal_states() {
        assert!(!settled(ChainJobState::Queued));
        assert!(!settled(ChainJobState::Running));
        assert!(!settled(ChainJobState::Interrupted));
        assert!(settled(ChainJobState::Failed));
        assert!(settled(ChainJobState::Completed));
        assert!(settled(ChainJobState::Cancelled));

        assert!(!ChainJobState::Failed.is_terminal());
        assert!(ChainJobState::Completed.is_terminal());
        assert!(!ChainJobState::Cancelled.is_terminal());
    }

    #[test]
    fn manifest_toml_round_trips_full_range_seeds() {
        let request = sample_request();
        let mut manifest =
            ChainJobManifest::new("01JBR55TEST".into(), 1_783_200_000_000, &request).unwrap();
        manifest.stage_status[1].state = StageState::Completed;
        manifest.stage_status[1].frames_emitted = Some(97);
        manifest.stage_status[1].generation_time_ms = Some(12_345);
        manifest.stage_status[1].segment = Some("stages/001/segment.mp4".into());
        manifest.stage_status[1].tail_frames = Some(17);
        manifest.stage_status[1].audio = Some("stages/001/audio.pcm".into());
        manifest.finalizes.push(FinalizeRecord {
            output: "final/output-1.mp4".into(),
            at_unix_ms: 1_783_200_000_500,
            stage_seeds: vec![u64::MAX - 3, u64::MAX - 2],
        });

        assert_eq!(
            manifest.request_json,
            serde_json::to_string(&request).unwrap()
        );
        assert_eq!(manifest.stage_status[0].seed, 42);
        assert_eq!(manifest.stage_status[1].seed, 42 ^ (u64::MAX - 3));

        let toml = manifest.to_toml().unwrap();
        assert!(toml.contains("seed = \"18446744073709551574\""));
        assert!(toml.contains("\"18446744073709551612\""));

        let round_tripped = ChainJobManifest::from_toml(&toml).unwrap();
        assert_eq!(round_tripped, manifest);
        assert_eq!(round_tripped.request().unwrap(), request);
    }

    #[test]
    fn effective_stage_seed_matches_approved_vectors() {
        assert_eq!(effective_stage_seed(42, None), 42);
        assert_eq!(effective_stage_seed(42, Some(0)), 42);
        assert_eq!(effective_stage_seed(42, Some(1)), 43);
        assert_eq!(
            effective_stage_seed(0xF0F0_F0F0_F0F0_F0F0, Some(0xFFFF_0000_5555_AAAA)),
            0x0F0F_F0F0_A5A5_5A5A
        );
        assert_eq!(
            effective_stage_seed(42, Some(u64::MAX - 3)),
            42 ^ (u64::MAX - 3)
        );
    }

    #[test]
    fn retake_request_seed_offset_round_trips_as_optional_string() {
        let max = u64::MAX;
        let json = serde_json::json!({
            "stage_idx": 7,
            "mode": "cascade",
            "seed_offset": max.to_string(),
            "prompt": "new prompt"
        });

        let req: RetakeRequest = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(req.seed_offset, Some(max));
        assert_eq!(
            serde_json::to_value(&req).unwrap()["seed_offset"],
            max.to_string()
        );

        let none_json = serde_json::json!({
            "stage_idx": 0,
            "mode": "splice"
        });
        let none_req: RetakeRequest = serde_json::from_value(none_json).unwrap();
        assert_eq!(none_req.seed_offset, None);
        assert!(serde_json::to_value(&none_req).unwrap()["seed_offset"].is_null());
    }

    #[test]
    fn amend_request_round_trips_u64_seed_as_string() {
        let max = u64::MAX;
        let json = serde_json::json!({
            "stages": [
                {
                    "prompt": "edited clip",
                    "frames": 97,
                    "transition": "cut"
                }
            ],
            "seed": max.to_string(),
            "steps": 8
        });

        let req: AmendRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.seed, Some(max));
        assert_eq!(req.steps, Some(8));
        assert_eq!(req.motion_tail_frames, None);
        assert_eq!(req.fps, None);
        assert_eq!(req.guidance, None);
        assert_eq!(req.enable_audio, None);
        assert_eq!(req.stages.len(), 1);
        assert_eq!(req.stages[0].prompt, "edited clip");
        assert_eq!(
            serde_json::to_value(&req).unwrap()["seed"],
            max.to_string(),
            "u64 seeds must round-trip as decimal strings"
        );

        let none: AmendRequest = serde_json::from_value(serde_json::json!({
            "stages": []
        }))
        .unwrap();
        assert_eq!(none.seed, None);
    }

    /// Old manifests written before the amend/raw-segment fields existed
    /// must keep parsing under `mold.chainjob.v1`: `amends` defaults to
    /// empty and every stage is a legacy (`raw_segment == false`) stage.
    #[test]
    fn manifest_amends_and_raw_segment_default_for_v1_toml() {
        let toml = r#"
schema = "mold.chainjob.v1"
job_id = "01JBR55V1"
created_at_unix_ms = 1

request_json = "{}"

[[stage_status]]
idx = 0
state = "completed"
seed = "42"
frames_emitted = 97
segment = "stages/000/segment.mp4"
tail_frames = 17
"#;
        let manifest = ChainJobManifest::from_toml(toml).unwrap();
        assert!(manifest.amends.is_empty());
        assert!(!manifest.stage_status[0].raw_segment);
        assert_eq!(manifest.stage_status[0].state, StageState::Completed);
    }

    #[test]
    fn chain_job_event_serde_uses_tagged_snapshot_shape() {
        let request = sample_request();
        let manifest =
            ChainJobManifest::new("01JBR55EVENT".into(), 1_783_200_000_000, &request).unwrap();
        let detail = ChainJobDetail {
            summary: ChainJobSummary {
                id: manifest.job_id.clone(),
                state: ChainJobState::Queued,
                model: request.model.clone(),
                stage_count: request.stages.len() as u32,
                current_stage: 0,
                created_at_unix_ms: manifest.created_at_unix_ms,
                updated_at_unix_ms: manifest.created_at_unix_ms,
                error: None,
                ephemeral: false,
            },
            stages: manifest
                .stage_status
                .iter()
                .map(|stage| ChainJobStageDetail {
                    idx: stage.idx,
                    state: stage.state,
                    seed: stage.seed,
                    frames_emitted: stage.frames_emitted,
                    generation_time_ms: stage.generation_time_ms,
                    has_preview: false,
                    error: stage.error.clone(),
                })
                .collect(),
            finalizes: vec![],
            retakes: vec![],
            amends: vec![],
            script: crate::chain::ChainScript::from(&request),
        };
        let value = serde_json::to_value(ChainJobEvent::Snapshot { job: detail }).unwrap();
        assert_eq!(value["type"], "snapshot");
        assert_eq!(value["job"]["id"], "01JBR55EVENT");

        let step = serde_json::to_value(ChainJobEvent::DenoiseStep {
            stage_idx: 2,
            step: 3,
            total: 8,
        })
        .unwrap();
        assert_eq!(
            step,
            serde_json::json!({
                "type": "denoise_step",
                "stage_idx": 2,
                "step": 3,
                "total": 8
            })
        );
    }

    #[test]
    fn from_toml_rejects_wrong_schema() {
        let err = ChainJobManifest::from_toml(
            r#"
schema = "mold.chainjob.v2"
job_id = "01JBR55TEST"
created_at_unix_ms = 1
request_json = "{}"
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("mold.chainjob.v2"));
        assert!(err.to_string().contains("supported: 'mold.chainjob.v1'"));
    }

    #[test]
    fn from_toml_rejects_absolute_stage_segment_path() {
        let request = sample_request();
        let mut manifest =
            ChainJobManifest::new("01JBR55TEST".into(), 1_783_200_000_000, &request).unwrap();
        manifest.stage_status[0].segment = Some("/tmp/segment.mp4".into());

        let err = ChainJobManifest::from_toml(&manifest.to_toml().unwrap()).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("stage_status[0].segment"));
        assert!(msg.contains("/tmp/segment.mp4"));
    }

    #[test]
    fn from_toml_rejects_parent_component_in_stage_audio_path() {
        let request = sample_request();
        let mut manifest =
            ChainJobManifest::new("01JBR55TEST".into(), 1_783_200_000_000, &request).unwrap();
        manifest.stage_status[1].audio = Some("stages/001/../audio.pcm".into());

        let err = ChainJobManifest::from_toml(&manifest.to_toml().unwrap()).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("stage_status[1].audio"));
        assert!(msg.contains("stages/001/../audio.pcm"));
    }

    #[test]
    fn from_toml_rejects_parent_component_in_finalize_output_path() {
        let request = sample_request();
        let mut manifest =
            ChainJobManifest::new("01JBR55TEST".into(), 1_783_200_000_000, &request).unwrap();
        manifest.finalizes.push(FinalizeRecord {
            output: "final/../output.mp4".into(),
            at_unix_ms: 1_783_200_000_500,
            stage_seeds: vec![42],
        });

        let err = ChainJobManifest::from_toml(&manifest.to_toml().unwrap()).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("finalizes[0].output"));
        assert!(msg.contains("final/../output.mp4"));
    }

    #[test]
    fn from_toml_accepts_normal_relative_artifact_paths() {
        let request = sample_request();
        let mut manifest =
            ChainJobManifest::new("01JBR55TEST".into(), 1_783_200_000_000, &request).unwrap();
        manifest.stage_status[0].segment = Some("stages/000/segment.mp4".into());
        manifest.stage_status[0].audio = Some("stages/000/audio.pcm".into());
        manifest.finalizes.push(FinalizeRecord {
            output: "final/output-1.mp4".into(),
            at_unix_ms: 1_783_200_000_500,
            stage_seeds: vec![42],
        });

        let parsed = ChainJobManifest::from_toml(&manifest.to_toml().unwrap()).unwrap();

        assert_eq!(parsed, manifest);
    }

    #[test]
    fn write_atomic_and_read_from_dir_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let request = sample_request();
        let manifest =
            ChainJobManifest::new("01JBR55TEST".into(), 1_783_200_000_000, &request).unwrap();

        manifest.write_atomic(dir.path()).unwrap();

        let manifest_path = dir.path().join(MANIFEST_FILE);
        assert!(manifest_path.exists());
        assert!(!dir.path().join("manifest.toml.tmp").exists());
        let read = ChainJobManifest::read_from_dir(dir.path()).unwrap();
        assert_eq!(read, manifest);
    }

    #[test]
    fn job_dir_layout_paths_match_spec() {
        let root = PathBuf::from("/tmp/mold-job");
        let layout = JobDirLayout::new(root.clone());

        assert_eq!(layout.root(), root.as_path());
        assert_eq!(layout.manifest_path(), root.join("manifest.toml"));
        assert_eq!(layout.stage_dir(7), root.join("stages").join("007"));
        assert_eq!(
            layout.segment_path(7),
            root.join("stages").join("007").join("segment.mp4")
        );
        assert_eq!(
            layout.tail_dir(7),
            root.join("stages").join("007").join("tail")
        );
        assert_eq!(
            layout.boundary_in_dir(7),
            root.join("stages").join("007").join("boundary-in")
        );
        assert_eq!(
            layout.boundary_out_dir(7),
            root.join("stages").join("007").join("boundary-out")
        );
        assert_eq!(
            layout.audio_path(7),
            root.join("stages").join("007").join("audio.pcm")
        );
        assert_eq!(
            layout.preview_path(7),
            root.join("stages").join("007").join("preview.jpg")
        );
        assert_eq!(
            layout.final_output_path(3),
            root.join("final").join("output-3.mp4")
        );
        assert_eq!(layout.segment_rel(7), "stages/007/segment.mp4");
        assert_eq!(layout.audio_rel(7), "stages/007/audio.pcm");
    }

    #[test]
    fn job_dir_layout_ensure_helpers_create_required_directories() {
        let dir = tempfile::tempdir().unwrap();
        let layout = JobDirLayout::new(dir.path().join("job"));

        layout.ensure_root().unwrap();
        layout.ensure_stage_dirs(7).unwrap();

        assert!(layout.root().is_dir());
        assert!(layout.stage_dir(7).is_dir());
        assert!(layout.tail_dir(7).is_dir());
        assert!(layout.boundary_in_dir(7).is_dir());
        assert!(layout.boundary_out_dir(7).is_dir());
    }

    fn event_detail_fixture() -> ChainJobDetail {
        let request = crate::chain::ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![sample_stage("stage zero", None)],
            motion_tail_frames: 0,
            width: 64,
            height: 64,
            fps: 12,
            seed: Some(42),
            steps: 4,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        };
        ChainJobDetail {
            summary: ChainJobSummary {
                id: "job-1".into(),
                state: ChainJobState::Running,
                model: request.model.clone(),
                stage_count: 1,
                current_stage: 0,
                created_at_unix_ms: 1,
                updated_at_unix_ms: 2,
                error: None,
                ephemeral: false,
            },
            stages: vec![ChainJobStageDetail {
                idx: 0,
                state: StageState::Pending,
                seed: 42,
                frames_emitted: None,
                generation_time_ms: None,
                has_preview: false,
                error: None,
            }],
            finalizes: vec![],
            retakes: vec![],
            amends: vec![],
            script: crate::chain::ChainScript::from(&request),
        }
    }

    #[test]
    fn chain_job_event_serde_tag_fixtures_match_web_contract() {
        let fixtures = vec![
            (
                ChainJobEvent::Snapshot {
                    job: event_detail_fixture(),
                },
                serde_json::json!("snapshot"),
            ),
            (
                ChainJobEvent::StageStart { stage_idx: 2 },
                serde_json::json!({"type":"stage_start","stage_idx":2}),
            ),
            (
                ChainJobEvent::DenoiseStep {
                    stage_idx: 2,
                    step: 3,
                    total: 8,
                },
                serde_json::json!({"type":"denoise_step","stage_idx":2,"step":3,"total":8}),
            ),
            (
                ChainJobEvent::StageDone {
                    stage_idx: 2,
                    frames_emitted: 97,
                    has_preview: true,
                },
                serde_json::json!({"type":"stage_done","stage_idx":2,"frames_emitted":97,"has_preview":true}),
            ),
            (
                ChainJobEvent::Yielded {
                    pending_small_jobs: 4,
                },
                serde_json::json!({"type":"yielded","pending_small_jobs":4}),
            ),
            (
                ChainJobEvent::Finalizing { total_frames: 194 },
                serde_json::json!({"type":"finalizing","total_frames":194}),
            ),
            (
                ChainJobEvent::Finalized {
                    output: "final/output-1.mp4".into(),
                    take: 1,
                },
                serde_json::json!({"type":"finalized","output":"final/output-1.mp4","take":1}),
            ),
            (
                ChainJobEvent::StateChanged {
                    state: ChainJobState::Completed,
                    error: None,
                },
                serde_json::json!({"type":"state_changed","state":"completed","error":null}),
            ),
        ];

        for (event, expected) in fixtures {
            let value = serde_json::to_value(&event).expect("event serializes");
            if expected == serde_json::json!("snapshot") {
                assert_eq!(value.get("type"), Some(&serde_json::json!("snapshot")));
                assert_eq!(value.pointer("/job/id"), Some(&serde_json::json!("job-1")));
            } else {
                assert_eq!(value, expected);
            }
        }
    }
}
