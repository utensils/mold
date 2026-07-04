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

use std::path::{Path, PathBuf};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Retake mode (spec §8): cascade re-renders N..end, splice re-renders N only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetakeMode {
    Cascade,
    Splice,
}

impl ChainJobState {
    pub fn as_str(self) -> &'static str {
        // TODO(BR55 Phase 6): canonical snake_case mapping, round-trip
        // tested against FromStr and the serde rename.
        todo!()
    }

    /// Whether no runner will pick this job up without an explicit
    /// resume/retake. `Completed` is terminal; `Failed`, `Interrupted`,
    /// and `Cancelled` are resumable (cancel is pause-with-intent, spec
    /// §6) but still "not active" — Phase 6 pins the exact semantics with
    /// tests per the Phase 2 gate note.
    pub fn is_terminal(self) -> bool {
        // TODO(BR55 Phase 6): implement + test the pinned semantics.
        todo!()
    }
}

impl std::str::FromStr for ChainJobState {
    type Err = MoldError;

    fn from_str(_s: &str) -> std::result::Result<Self, MoldError> {
        // TODO(BR55 Phase 6): inverse of as_str; MoldError::Validation on
        // unknown values naming the offending string.
        todo!()
    }
}

impl StageState {
    pub fn as_str(self) -> &'static str {
        // TODO(BR55 Phase 6): canonical snake_case mapping.
        todo!()
    }
}

impl std::str::FromStr for StageState {
    type Err = MoldError;

    fn from_str(_s: &str) -> std::result::Result<Self, MoldError> {
        // TODO(BR55 Phase 6): inverse of as_str.
        todo!()
    }
}

impl RetakeMode {
    pub fn as_str(self) -> &'static str {
        // TODO(BR55 Phase 6): canonical snake_case mapping (serde-only
        // round-trip; RetakeMode never passes through DB TEXT).
        todo!()
    }
}

// ── Manifest types (spec §3.2, amended per approved Phase 1) ───────────

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
    /// intact for provenance; edits are recorded here.
    #[serde(default)]
    pub retakes: Vec<RetakeAmendment>,
    /// Versioned finalize history (spec §8.3).
    #[serde(default)]
    pub finalizes: Vec<FinalizeRecord>,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub fn new(_job_id: String, _created_at_unix_ms: u64, _request: &ChainRequest) -> Result<Self> {
        // TODO(BR55 Phase 6): serde_json::to_string(request); stage_status
        // pre-populated Pending per request stage with effective seeds.
        todo!()
    }

    /// Parsed-value access to the embedded request. Reconcile logic
    /// compares THIS, never raw `request_json` strings (field-order
    /// changes alter bytes without altering meaning).
    pub fn request(&self) -> Result<ChainRequest> {
        // TODO(BR55 Phase 6): serde_json::from_str with context.
        todo!()
    }

    /// Parse a manifest, rejecting `schema != mold.chainjob.v1` with a
    /// clear error naming the supported schema (chain_toml.rs precedent).
    pub fn from_toml(_s: &str) -> Result<Self> {
        // TODO(BR55 Phase 6): schema peek + full parse.
        todo!()
    }

    pub fn to_toml(&self) -> Result<String> {
        // TODO(BR55 Phase 6): toml::to_string round-trip counterpart.
        todo!()
    }

    /// Write `<job_dir>/manifest.toml` via write-temp + rename so a crash
    /// mid-write never leaves a truncated manifest (spec §3.3 ordering:
    /// artifacts → manifest → DB).
    pub fn write_atomic(&self, _job_dir: &Path) -> Result<()> {
        // TODO(BR55 Phase 6): temp file in job_dir + fs::rename.
        todo!()
    }

    pub fn read_from_dir(_job_dir: &Path) -> Result<Self> {
        // TODO(BR55 Phase 6): read manifest.toml + from_toml.
        todo!()
    }
}

// ── Job-directory layout helpers (pure path math + mkdir) ─────────────

/// Path helpers for one job directory. Pure path math except the
/// `ensure_*` mkdir helpers; all artifact paths follow spec §3.2.
pub struct JobDirLayout {
    // Phase 3 placeholder allowance: nothing reads the field until the
    // Phase 6 bodies land. Remove the allow with the TODOs.
    #[allow(dead_code)]
    root: PathBuf,
}

impl JobDirLayout {
    pub fn new(_root: PathBuf) -> Self {
        // TODO(BR55 Phase 6): trivial constructor.
        todo!()
    }

    pub fn root(&self) -> &Path {
        // TODO(BR55 Phase 6): &self.root
        todo!()
    }

    pub fn manifest_path(&self) -> PathBuf {
        // TODO(BR55 Phase 6): root.join(MANIFEST_FILE)
        todo!()
    }

    /// `stages/NNN/` with zero-padded 3-digit stage index.
    pub fn stage_dir(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): root.join(STAGES_DIR).join(format!("{idx:03}"))
        todo!()
    }

    pub fn segment_path(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): stage_dir(idx).join(SEGMENT_FILE)
        todo!()
    }

    pub fn tail_dir(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): stage_dir(idx).join(TAIL_DIR)
        todo!()
    }

    pub fn boundary_in_dir(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): stage_dir(idx).join(BOUNDARY_IN_DIR)
        todo!()
    }

    pub fn boundary_out_dir(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): stage_dir(idx).join(BOUNDARY_OUT_DIR)
        todo!()
    }

    pub fn audio_path(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): stage_dir(idx).join(AUDIO_FILE)
        todo!()
    }

    pub fn preview_path(&self, _idx: u32) -> PathBuf {
        // TODO(BR55 Phase 6): stage_dir(idx).join(PREVIEW_FILE)
        todo!()
    }

    /// `final/output-<n>.mp4`. `n` derives from `manifest.finalizes.len()`,
    /// never from a directory scan (Phase 2 gate note).
    pub fn final_output_path(&self, _n: u32) -> PathBuf {
        // TODO(BR55 Phase 6): root.join(FINAL_DIR).join(format!("output-{n}.mp4"))
        todo!()
    }

    /// Relative-to-root form for manifest fields (portability contract).
    pub fn segment_rel(&self, _idx: u32) -> String {
        // TODO(BR55 Phase 6): format!("{STAGES_DIR}/{idx:03}/{SEGMENT_FILE}")
        todo!()
    }

    pub fn audio_rel(&self, _idx: u32) -> String {
        // TODO(BR55 Phase 6): format!("{STAGES_DIR}/{idx:03}/{AUDIO_FILE}")
        todo!()
    }

    pub fn ensure_root(&self) -> Result<()> {
        // TODO(BR55 Phase 6): create_dir_all(root).
        todo!()
    }

    /// Create `stages/NNN/` plus its `tail/`, `boundary-in/`,
    /// `boundary-out/` subdirectories.
    pub fn ensure_stage_dirs(&self, _idx: u32) -> Result<()> {
        // TODO(BR55 Phase 6): create_dir_all for stage_dir + tail/ +
        // boundary-in/ + boundary-out/
        todo!()
    }
}

// ── serde helpers: full-range u64 as decimal strings in TOML ──────────

mod u64_as_string {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(_v: &u64, _s: S) -> std::result::Result<S::Ok, S::Error> {
        // TODO(BR55 Phase 6): collect_str(v).
        todo!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(_d: D) -> std::result::Result<u64, D::Error> {
        // TODO(BR55 Phase 6): String::deserialize + str::parse with a
        // custom error naming the field's expected format.
        todo!()
    }
}

mod u64_vec_as_strings {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(_v: &Vec<u64>, _s: S) -> std::result::Result<S::Ok, S::Error> {
        // TODO(BR55 Phase 6): seq of collect_str.
        todo!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        _d: D,
    ) -> std::result::Result<Vec<u64>, D::Error> {
        // TODO(BR55 Phase 6): Vec<String> + parse each.
        todo!()
    }
}
