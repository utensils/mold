//! Chain-job index: typed CRUD over the `chain_jobs` + `chain_job_stages`
//! tables (schema v11).
//!
//! The DB is the queryable index; the job directory's `manifest.toml`
//! (`mold_core::chain_job`) is the portable source of truth. When they
//! disagree, the manifest wins and reconcile repairs the row.
//!
//! Free functions over [`MetadataDb`] (closest existing precedent:
//! `model_prefs.rs` associated fns taking `db`). Seeds are full-range u64
//! stored as i64 bit-casts at the SQL boundary; state enums round-trip
//! through their canonical `as_str`/`FromStr` mapping in
//! `mold_core::chain_job`.

use std::path::PathBuf;

use anyhow::Result;
use mold_core::chain_job::{ChainJobState, StageState};

use crate::db::MetadataDb;

/// One row of `chain_jobs`, mirroring the v11 DDL 1:1.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainJobRow {
    pub id: String,
    pub state: ChainJobState,
    pub model: String,
    /// Canonical serde_json of the normalised ChainRequest — the same
    /// encoding the manifest embeds. Compare parsed values, not strings.
    pub request_json: String,
    /// Absolute path to the job directory (the only absolute path in the
    /// whole scheme; manifest-internal paths are all relative).
    pub job_dir: PathBuf,
    pub stage_count: u32,
    pub current_stage: u32,
    pub error: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub finalized_at_ms: Option<i64>,
}

/// One row of `chain_job_stages`, mirroring the v11 DDL 1:1.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainJobStageRow {
    pub job_id: String,
    pub stage_idx: u32,
    pub state: StageState,
    /// Effective u64 seed; i64 bit-cast at the SQL boundary.
    pub seed: u64,
    pub frames_emitted: Option<u32>,
    pub generation_time_ms: Option<u64>,
    pub segment_rel_path: Option<String>,
    pub error: Option<String>,
    pub updated_at_ms: i64,
}

pub fn insert_job(_db: &MetadataDb, _row: &ChainJobRow) -> Result<()> {
    // TODO(BR55 Phase 6): INSERT with i64 bit-cast for seeds n/a here;
    // state via as_str.
    todo!()
}

pub fn get_job(_db: &MetadataDb, _id: &str) -> Result<Option<ChainJobRow>> {
    // TODO(BR55 Phase 6): SELECT by id; state via FromStr.
    todo!()
}

/// All jobs, newest first (`created_at` desc).
pub fn list_jobs(_db: &MetadataDb) -> Result<Vec<ChainJobRow>> {
    // TODO(BR55 Phase 6): SELECT * FROM chain_jobs ORDER BY created_at
    // DESC; states via FromStr.
    todo!()
}

pub fn jobs_in_state(_db: &MetadataDb, _state: ChainJobState) -> Result<Vec<ChainJobRow>> {
    // TODO(BR55 Phase 6): startup reconcile uses this for Running →
    // Interrupted flips (Run 2 consumes it).
    todo!()
}

/// Returns `false` when `id` is unknown (caller decides whether that is
/// an error). Clears `error` when `error` is `None`.
pub fn update_job_state(
    _db: &MetadataDb,
    _id: &str,
    _state: ChainJobState,
    _error: Option<&str>,
    _updated_at_ms: i64,
) -> Result<bool> {
    // TODO(BR55 Phase 6): UPDATE state/error/updated_at WHERE id; return
    // rows_affected > 0.
    todo!()
}

pub fn set_current_stage(
    _db: &MetadataDb,
    _id: &str,
    _current_stage: u32,
    _updated_at_ms: i64,
) -> Result<bool> {
    // TODO(BR55 Phase 6): UPDATE current_stage + updated_at WHERE id;
    // return rows_affected > 0.
    todo!()
}

pub fn set_finalized_at(_db: &MetadataDb, _id: &str, _finalized_at_ms: i64) -> Result<bool> {
    // TODO(BR55 Phase 6): UPDATE finalized_at + updated_at WHERE id;
    // return rows_affected > 0.
    todo!()
}

/// Deletes the job row; stage rows cascade via the v11 foreign key
/// (`ON DELETE CASCADE` is the contract — see the cascade contract test).
pub fn delete_job(_db: &MetadataDb, _id: &str) -> Result<bool> {
    // TODO(BR55 Phase 6): DELETE FROM chain_jobs WHERE id (stages cascade
    // via FK); return rows_affected > 0.
    todo!()
}

pub fn upsert_stage(_db: &MetadataDb, _row: &ChainJobStageRow) -> Result<()> {
    // TODO(BR55 Phase 6): INSERT ... ON CONFLICT(job_id, stage_idx) DO
    // UPDATE; seed as i64 bit-cast.
    todo!()
}

/// Stages for one job, `stage_idx` ascending.
pub fn stages_for_job(_db: &MetadataDb, _job_id: &str) -> Result<Vec<ChainJobStageRow>> {
    // TODO(BR55 Phase 6): SELECT WHERE job_id ORDER BY stage_idx ASC;
    // seed via u64 bit-cast, state via FromStr.
    todo!()
}
