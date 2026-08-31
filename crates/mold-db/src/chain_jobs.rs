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
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use mold_core::chain_job::{ChainJobState, StageState};
use rusqlite::{
    params,
    types::{Type, Value},
    OptionalExtension, Row,
};

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

pub fn insert_job(db: &MetadataDb, row: &ChainJobRow) -> Result<()> {
    db.with_conn(|conn| {
        let job_dir = row.job_dir.to_string_lossy().into_owned();
        conn.execute(
            "INSERT INTO chain_jobs (
                id, state, model, request_json, job_dir, stage_count,
                current_stage, error, created_at, updated_at, finalized_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &row.id,
                row.state.as_str(),
                &row.model,
                &row.request_json,
                job_dir,
                row.stage_count as i64,
                row.current_stage as i64,
                row.error.as_deref(),
                row.created_at_ms,
                row.updated_at_ms,
                row.finalized_at_ms,
            ],
        )?;
        Ok(())
    })
}

pub fn get_job(db: &MetadataDb, id: &str) -> Result<Option<ChainJobRow>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT id, state, model, request_json, job_dir, stage_count,
                    current_stage, error, created_at, updated_at, finalized_at
             FROM chain_jobs WHERE id = ?1",
            params![id],
            row_to_job,
        )
        .optional()
        .map_err(Into::into)
    })
}

/// All jobs, newest first (`created_at` desc).
pub fn list_jobs(db: &MetadataDb) -> Result<Vec<ChainJobRow>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, state, model, request_json, job_dir, stage_count,
                    current_stage, error, created_at, updated_at, finalized_at
             FROM chain_jobs ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], row_to_job)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

pub fn jobs_in_state(db: &MetadataDb, state: ChainJobState) -> Result<Vec<ChainJobRow>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, state, model, request_json, job_dir, stage_count,
                    current_stage, error, created_at, updated_at, finalized_at
             FROM chain_jobs
             WHERE state = ?1
             ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map(params![state.as_str()], row_to_job)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Park every pre-existing runnable sequence during startup. This is one
/// database transition so the runner can never observe only part of the
/// recovered backlog.
pub fn pause_active_for_restart(db: &MetadataDb, updated_at_ms: i64) -> Result<usize> {
    db.with_conn(|conn| {
        let changed = conn.execute(
            "UPDATE chain_jobs
                SET state = 'paused',
                    error = CASE
                        WHEN state = 'running' THEN 'server restarted while chain job was running'
                        ELSE NULL
                    END,
                    updated_at = MAX(?1, updated_at + 1)
              WHERE state IN ('queued', 'running')",
            params![updated_at_ms],
        )?;
        Ok(changed)
    })
}

/// Park work that has not been claimed yet while a graceful shutdown fences
/// the runner. Running attempts settle themselves so an already accepted user
/// cancellation cannot be overwritten by this restart transition.
pub fn pause_queued_for_shutdown(db: &MetadataDb, updated_at_ms: i64) -> Result<usize> {
    db.with_conn(|conn| {
        let changed = conn.execute(
            "UPDATE chain_jobs
                SET state = 'paused',
                    error = NULL,
                    updated_at = MAX(?1, updated_at + 1)
              WHERE state = 'queued'",
            params![updated_at_ms],
        )?;
        Ok(changed)
    })
}

/// Oldest queued job by `created_at` (FIFO runner contract).
pub fn next_queued_job(db: &MetadataDb) -> Result<Option<ChainJobRow>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT id, state, model, request_json, job_dir, stage_count,
                    current_stage, error, created_at, updated_at, finalized_at
             FROM chain_jobs
             WHERE state = ?1
             ORDER BY created_at ASC, id ASC
             LIMIT 1",
            params![ChainJobState::Queued.as_str()],
            row_to_job,
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Atomically claim a queued job for execution.
///
/// Returns `false` when another caller already moved the row out of `queued`
/// (or the row disappeared). The predicate is part of the `UPDATE`, so callers
/// must not pre-check `state` and then perform an unconditional mutation.
pub fn claim_job(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET state = ?1, updated_at = ?2
             WHERE id = ?3 AND state = ?4",
            params![
                ChainJobState::Running.as_str(),
                now_ms(),
                id,
                ChainJobState::Queued.as_str()
            ],
        )?;
        Ok(n > 0)
    })
}

/// Atomically transition a job from any state in `from` to `to`.
///
/// Returns `false` on compare-and-swap loss. `from` is intentionally explicit
/// at every call site so route handlers encode their allowed source states in
/// the SQL predicate that mutates the row.
pub fn try_transition(
    db: &MetadataDb,
    id: &str,
    from: &[ChainJobState],
    to: ChainJobState,
    error: Option<&str>,
    updated_at_ms: i64,
) -> Result<bool> {
    if from.is_empty() {
        return Ok(false);
    }

    db.with_conn(|conn| {
        let placeholders = (0..from.len())
            .map(|idx| format!("?{}", idx + 5))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "UPDATE chain_jobs
             SET state = ?1, error = ?2, updated_at = ?3
             WHERE id = ?4 AND state IN ({placeholders})"
        );
        let mut values = vec![
            Value::Text(to.as_str().to_string()),
            error
                .map(|error| Value::Text(error.to_string()))
                .unwrap_or(Value::Null),
            Value::Integer(updated_at_ms),
            Value::Text(id.to_string()),
        ];
        values.extend(
            from.iter()
                .map(|state| Value::Text(state.as_str().to_string())),
        );
        let n = conn.execute(&sql, rusqlite::params_from_iter(values))?;
        Ok(n > 0)
    })
}

/// Returns `false` when `id` is unknown (caller decides whether that is
/// an error). Clears `error` when `error` is `None`.
pub fn update_job_state(
    db: &MetadataDb,
    id: &str,
    state: ChainJobState,
    error: Option<&str>,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET state = ?1, error = ?2, updated_at = ?3
             WHERE id = ?4",
            params![state.as_str(), error, updated_at_ms, id],
        )?;
        Ok(n > 0)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn repair_job_from_manifest(
    db: &MetadataDb,
    id: &str,
    state: ChainJobState,
    current_stage: u32,
    error: Option<&str>,
    updated_at_ms: i64,
    finalized_at_ms: Option<i64>,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET state = ?1, current_stage = ?2, error = ?3, updated_at = ?4, finalized_at = ?5
             WHERE id = ?6",
            params![
                state.as_str(),
                current_stage as i64,
                error,
                updated_at_ms,
                finalized_at_ms,
                id
            ],
        )?;
        Ok(n > 0)
    })
}

pub fn set_current_stage(
    db: &MetadataDb,
    id: &str,
    current_stage: u32,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET current_stage = ?1, updated_at = ?2
             WHERE id = ?3",
            params![current_stage as i64, updated_at_ms, id],
        )?;
        Ok(n > 0)
    })
}

pub fn set_job_queued(
    db: &MetadataDb,
    id: &str,
    current_stage: u32,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET state = ?1, current_stage = ?2, error = NULL, updated_at = ?3
             WHERE id = ?4",
            params![
                ChainJobState::Queued.as_str(),
                current_stage as i64,
                updated_at_ms,
                id
            ],
        )?;
        Ok(n > 0)
    })
}

/// Sets both `finalized_at` and `updated_at` to `finalized_at_ms`.
/// Finalization is the update event, so this timestamp equality is intentional.
pub fn set_finalized_at(db: &MetadataDb, id: &str, finalized_at_ms: i64) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET finalized_at = ?1, updated_at = ?1
             WHERE id = ?2",
            params![finalized_at_ms, id],
        )?;
        Ok(n > 0)
    })
}

pub fn reset_stages_from(
    db: &MetadataDb,
    job_id: &str,
    start_stage: u32,
    updated_at_ms: i64,
) -> Result<usize> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_job_stages
             SET state = ?1,
                 frames_emitted = NULL,
                 generation_time_ms = NULL,
                 segment_rel_path = NULL,
                 error = NULL,
                 updated_at = ?2
             WHERE job_id = ?3 AND stage_idx >= ?4",
            params![
                StageState::Pending.as_str(),
                updated_at_ms,
                job_id,
                start_stage as i64
            ],
        )?;
        Ok(n)
    })
}

pub fn reset_one_stage(
    db: &MetadataDb,
    job_id: &str,
    stage_idx: u32,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_job_stages
             SET state = ?1,
                 frames_emitted = NULL,
                 generation_time_ms = NULL,
                 segment_rel_path = NULL,
                 error = NULL,
                 updated_at = ?2
             WHERE job_id = ?3 AND stage_idx = ?4",
            params![
                StageState::Pending.as_str(),
                updated_at_ms,
                job_id,
                stage_idx as i64
            ],
        )?;
        Ok(n > 0)
    })
}

/// Delete every stage row with `stage_idx >= from_idx` (amend shrink /
/// suffix invalidation). Returns the number of rows removed.
pub fn delete_stages_from(db: &MetadataDb, job_id: &str, from_idx: u32) -> Result<usize> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "DELETE FROM chain_job_stages
             WHERE job_id = ?1 AND stage_idx >= ?2",
            params![job_id, from_idx as i64],
        )?;
        Ok(n)
    })
}

/// Persist an amended job's new shape: stage count and the stage rendering
/// resumes from. Returns `false` when `job_id` is unknown.
pub fn update_stage_shape(
    db: &MetadataDb,
    job_id: &str,
    stage_count: u32,
    current_stage: u32,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET stage_count = ?1, current_stage = ?2, updated_at = ?3
             WHERE id = ?4",
            params![
                stage_count as i64,
                current_stage as i64,
                updated_at_ms,
                job_id
            ],
        )?;
        Ok(n > 0)
    })
}

/// Rewrite the indexed copy of the canonical request JSON (amend rewrites
/// the manifest's `request_json`; the DB row mirrors it). Returns `false`
/// when `job_id` is unknown.
pub fn set_request_json(
    db: &MetadataDb,
    job_id: &str,
    request_json: &str,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "UPDATE chain_jobs
             SET request_json = ?1, updated_at = ?2
             WHERE id = ?3",
            params![request_json, updated_at_ms, job_id],
        )?;
        Ok(n > 0)
    })
}

/// Deletes the job row; stage rows cascade via the v11 foreign key
/// (`ON DELETE CASCADE` is the contract — see the cascade contract test).
pub fn delete_job(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute("DELETE FROM chain_jobs WHERE id = ?1", params![id])?;
        Ok(n > 0)
    })
}

/// Delete a job row only when it is not currently running.
///
/// The running-state predicate is part of the `DELETE`, so racing claim/delete
/// callers cannot remove a row after another caller has made it active.
pub fn delete_job_not_running(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        let n = conn.execute(
            "DELETE FROM chain_jobs
             WHERE id = ?1 AND state != ?2",
            params![id, ChainJobState::Running.as_str()],
        )?;
        Ok(n > 0)
    })
}

pub fn upsert_stage(db: &MetadataDb, row: &ChainJobStageRow) -> Result<()> {
    let generation_time_ms = row
        .generation_time_ms
        .map(i64::try_from)
        .transpose()
        .context("chain job stage generation_time_ms exceeds SQLite INTEGER range")?;
    db.with_conn(|conn| {
        conn.execute(
            "INSERT INTO chain_job_stages (
                job_id, stage_idx, state, seed, frames_emitted,
                generation_time_ms, segment_rel_path, error, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(job_id, stage_idx) DO UPDATE SET
                state = excluded.state,
                seed = excluded.seed,
                frames_emitted = excluded.frames_emitted,
                generation_time_ms = excluded.generation_time_ms,
                segment_rel_path = excluded.segment_rel_path,
                error = excluded.error,
                updated_at = excluded.updated_at",
            params![
                &row.job_id,
                row.stage_idx as i64,
                row.state.as_str(),
                row.seed as i64,
                row.frames_emitted.map(|v| v as i64),
                generation_time_ms,
                row.segment_rel_path.as_deref(),
                row.error.as_deref(),
                row.updated_at_ms,
            ],
        )?;
        Ok(())
    })
}

/// Stages for one job, `stage_idx` ascending.
pub fn stages_for_job(db: &MetadataDb, job_id: &str) -> Result<Vec<ChainJobStageRow>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT job_id, stage_idx, state, seed, frames_emitted,
                    generation_time_ms, segment_rel_path, error, updated_at
             FROM chain_job_stages
             WHERE job_id = ?1
             ORDER BY stage_idx ASC",
        )?;
        let rows = stmt.query_map(params![job_id], row_to_stage)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<ChainJobRow> {
    let state_raw: String = row.get(1)?;
    let stage_count = int_to_u32(row.get(5)?, 5)?;
    let current_stage = int_to_u32(row.get(6)?, 6)?;
    let job_dir: String = row.get(4)?;
    Ok(ChainJobRow {
        id: row.get(0)?,
        state: ChainJobState::from_str(&state_raw)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(1, Type::Text, Box::new(e)))?,
        model: row.get(2)?,
        request_json: row.get(3)?,
        job_dir: PathBuf::from(job_dir),
        stage_count,
        current_stage,
        error: row.get(7)?,
        created_at_ms: row.get(8)?,
        updated_at_ms: row.get(9)?,
        finalized_at_ms: row.get(10)?,
    })
}

fn row_to_stage(row: &Row<'_>) -> rusqlite::Result<ChainJobStageRow> {
    let state_raw: String = row.get(2)?;
    let seed: i64 = row.get(3)?;
    Ok(ChainJobStageRow {
        job_id: row.get(0)?,
        stage_idx: int_to_u32(row.get(1)?, 1)?,
        state: StageState::from_str(&state_raw)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(2, Type::Text, Box::new(e)))?,
        seed: seed as u64,
        frames_emitted: opt_int_to_u32(row.get(4)?, 4)?,
        generation_time_ms: opt_int_to_u64(row.get(5)?, 5)?,
        segment_rel_path: row.get(6)?,
        error: row.get(7)?,
        updated_at_ms: row.get(8)?,
    })
}

fn int_to_u32(value: i64, col: usize) -> rusqlite::Result<u32> {
    u32::try_from(value)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(col, Type::Integer, Box::new(e)))
}

fn opt_int_to_u32(value: Option<i64>, col: usize) -> rusqlite::Result<Option<u32>> {
    value.map(|v| int_to_u32(v, col)).transpose()
}

fn opt_int_to_u64(value: Option<i64>, col: usize) -> rusqlite::Result<Option<u64>> {
    value
        .map(|v| {
            u64::try_from(v).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(col, Type::Integer, Box::new(e))
            })
        })
        .transpose()
}

fn now_ms() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    fn job(id: &str, state: ChainJobState, created_at_ms: i64) -> ChainJobRow {
        ChainJobRow {
            id: id.into(),
            state,
            model: "ltx-2-19b-distilled:fp8".into(),
            request_json: r#"{"model":"ltx-2-19b-distilled:fp8","stages":[]}"#.into(),
            job_dir: PathBuf::from(format!("/tmp/mold/jobs/{id}")),
            stage_count: 2,
            current_stage: 0,
            error: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
            finalized_at_ms: None,
        }
    }

    fn stage(job_id: &str, idx: u32, state: StageState, seed: u64) -> ChainJobStageRow {
        ChainJobStageRow {
            job_id: job_id.into(),
            stage_idx: idx,
            state,
            seed,
            frames_emitted: Some(97),
            generation_time_ms: Some(12_345),
            segment_rel_path: Some(format!("stages/{idx:03}/segment.mp4")),
            error: None,
            updated_at_ms: 2_000,
        }
    }

    fn table_exists(db: &MetadataDb, table: &str) -> bool {
        db.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table],
                |r| r.get(0),
            )?;
            Ok(count == 1)
        })
        .unwrap()
    }

    #[test]
    fn in_memory_schema_reaches_current_version_and_creates_chain_job_tables() {
        let db = db();
        assert_eq!(db.schema_version().unwrap(), crate::SCHEMA_VERSION);
        assert!(table_exists(&db, "chain_jobs"));
        assert!(table_exists(&db, "chain_job_stages"));
    }

    #[test]
    fn insert_get_list_jobs_in_state_and_stage_round_trip() {
        let db = db();
        let older = job("01JBR55OLDER", ChainJobState::Queued, 1_000);
        let newer = job("01JBR55NEWER", ChainJobState::Running, 2_000);
        insert_job(&db, &older).unwrap();
        insert_job(&db, &newer).unwrap();

        let high_seed = u64::MAX - 3;
        upsert_stage(
            &db,
            &ChainJobStageRow {
                seed: high_seed,
                ..stage(&newer.id, 1, StageState::Completed, high_seed)
            },
        )
        .unwrap();

        let got = get_job(&db, &newer.id).unwrap().expect("job exists");
        assert_eq!(got, newer);

        let listed = list_jobs(&db).unwrap();
        assert_eq!(
            listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["01JBR55NEWER", "01JBR55OLDER"]
        );

        let running = jobs_in_state(&db, ChainJobState::Running).unwrap();
        assert_eq!(running, vec![newer.clone()]);

        let stages = stages_for_job(&db, &newer.id).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].seed, high_seed);
        assert_eq!(stages[0].state, StageState::Completed);
    }

    #[test]
    fn update_job_state_returns_false_for_unknown_id() {
        let db = db();
        assert!(!claim_job(&db, "missing").unwrap());
        assert!(!try_transition(
            &db,
            "missing",
            &[ChainJobState::Queued],
            ChainJobState::Running,
            None,
            3_000,
        )
        .unwrap());
        assert!(!update_job_state(
            &db,
            "missing",
            ChainJobState::Interrupted,
            Some("server restarted"),
            3_000,
        )
        .unwrap());
        assert!(!set_current_stage(&db, "missing", 1, 3_000).unwrap());
        assert!(!set_finalized_at(&db, "missing", 4_000).unwrap());
        assert!(!delete_job_not_running(&db, "missing").unwrap());
    }

    #[test]
    fn claim_job_only_moves_queued_rows_to_running() {
        let db = db();
        let queued = job("01JBR55CLAIM", ChainJobState::Queued, 1_000);
        let failed = job("01JBR55NOCLAIM", ChainJobState::Failed, 1_000);
        insert_job(&db, &queued).unwrap();
        insert_job(&db, &failed).unwrap();

        assert!(claim_job(&db, &queued.id).unwrap());
        assert!(!claim_job(&db, &queued.id).unwrap());
        assert!(!claim_job(&db, &failed.id).unwrap());

        let got = get_job(&db, &queued.id).unwrap().unwrap();
        assert_eq!(got.state, ChainJobState::Running);
        assert!(got.updated_at_ms >= queued.updated_at_ms);
        assert_eq!(
            get_job(&db, &failed.id).unwrap().unwrap().state,
            ChainJobState::Failed
        );
    }

    #[test]
    fn graceful_shutdown_parks_only_unclaimed_jobs() {
        let db = db();
        let queued = job("01JBR55SHUTDOWNQ", ChainJobState::Queued, 1_000);
        let running = job("01JBR55SHUTDOWNR", ChainJobState::Running, 1_000);
        insert_job(&db, &queued).unwrap();
        insert_job(&db, &running).unwrap();

        assert_eq!(pause_queued_for_shutdown(&db, 2_000).unwrap(), 1);
        assert_eq!(
            get_job(&db, &queued.id).unwrap().unwrap().state,
            ChainJobState::Paused
        );
        assert_eq!(
            get_job(&db, &running.id).unwrap().unwrap().state,
            ChainJobState::Running,
            "the worker must settle a running attempt so an accepted user cancel wins"
        );
    }

    #[test]
    fn try_transition_is_state_predicated_and_clears_error() {
        let db = db();
        let row = ChainJobRow {
            error: Some("boom".into()),
            ..job("01JBR55CAS", ChainJobState::Failed, 1_000)
        };
        insert_job(&db, &row).unwrap();

        assert!(!try_transition(
            &db,
            &row.id,
            &[ChainJobState::Interrupted],
            ChainJobState::Queued,
            None,
            2_000,
        )
        .unwrap());
        assert_eq!(
            get_job(&db, &row.id).unwrap().unwrap().state,
            ChainJobState::Failed
        );

        assert!(try_transition(
            &db,
            &row.id,
            &[ChainJobState::Interrupted, ChainJobState::Failed],
            ChainJobState::Queued,
            None,
            3_000,
        )
        .unwrap());
        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.state, ChainJobState::Queued);
        assert_eq!(got.error, None);
        assert_eq!(got.updated_at_ms, 3_000);
    }

    #[test]
    fn delete_job_not_running_refuses_running_rows() {
        let db = db();
        let running = job("01JBR55RUNDEL", ChainJobState::Running, 1_000);
        let failed = job("01JBR55FAILDEL", ChainJobState::Failed, 1_000);
        insert_job(&db, &running).unwrap();
        insert_job(&db, &failed).unwrap();

        assert!(!delete_job_not_running(&db, &running.id).unwrap());
        assert!(get_job(&db, &running.id).unwrap().is_some());
        assert!(delete_job_not_running(&db, &failed.id).unwrap());
        assert!(get_job(&db, &failed.id).unwrap().is_none());
    }

    #[test]
    fn state_stage_and_finalize_updates_mutate_existing_job() {
        let db = db();
        let row = job("01JBR55UPDATE", ChainJobState::Queued, 1_000);
        insert_job(&db, &row).unwrap();

        assert!(update_job_state(
            &db,
            &row.id,
            ChainJobState::Failed,
            Some("stage failed"),
            2_000,
        )
        .unwrap());
        assert!(set_current_stage(&db, &row.id, 1, 3_000).unwrap());
        assert!(set_finalized_at(&db, &row.id, 4_000).unwrap());

        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.state, ChainJobState::Failed);
        assert_eq!(got.error.as_deref(), Some("stage failed"));
        assert_eq!(got.current_stage, 1);
        assert_eq!(got.finalized_at_ms, Some(4_000));
        assert_eq!(got.updated_at_ms, 4_000);

        assert!(update_job_state(&db, &row.id, ChainJobState::Running, None, 5_000).unwrap());
        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.error, None);
        assert_eq!(got.updated_at_ms, 5_000);
    }

    #[test]
    fn upsert_stage_twice_updates_in_place() {
        let db = db();
        let row = job("01JBR55STAGE", ChainJobState::Running, 1_000);
        insert_job(&db, &row).unwrap();

        upsert_stage(&db, &stage(&row.id, 0, StageState::Running, 42)).unwrap();
        upsert_stage(
            &db,
            &ChainJobStageRow {
                state: StageState::Completed,
                frames_emitted: Some(89),
                generation_time_ms: Some(99_999),
                segment_rel_path: Some("stages/000/segment-retake.mp4".into()),
                updated_at_ms: 3_000,
                ..stage(&row.id, 0, StageState::Running, 123)
            },
        )
        .unwrap();

        let stages = stages_for_job(&db, &row.id).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].state, StageState::Completed);
        assert_eq!(stages[0].seed, 123);
        assert_eq!(stages[0].frames_emitted, Some(89));
        assert_eq!(stages[0].generation_time_ms, Some(99_999));
        assert_eq!(
            stages[0].segment_rel_path.as_deref(),
            Some("stages/000/segment-retake.mp4")
        );
        assert_eq!(stages[0].updated_at_ms, 3_000);
    }

    #[test]
    fn delete_job_cascades_stage_rows() {
        let db = db();
        let row = job("01JBR55CASCADE", ChainJobState::Running, 1_000);
        insert_job(&db, &row).unwrap();
        upsert_stage(&db, &stage(&row.id, 0, StageState::Completed, 42)).unwrap();
        upsert_stage(&db, &stage(&row.id, 1, StageState::Running, 43)).unwrap();
        assert_eq!(stages_for_job(&db, &row.id).unwrap().len(), 2);

        assert!(delete_job(&db, &row.id).unwrap());
        assert!(get_job(&db, &row.id).unwrap().is_none());
        assert!(stages_for_job(&db, &row.id).unwrap().is_empty());
        assert!(!delete_job(&db, &row.id).unwrap());
    }

    #[test]
    fn next_queued_job_is_fifo_by_created_at() {
        let db = db();
        let running = job("01JBR55RUNNING", ChainJobState::Running, 500);
        let older = job("01JBR55OLDER", ChainJobState::Queued, 1_000);
        let newer = job("01JBR55NEWER", ChainJobState::Queued, 2_000);
        insert_job(&db, &newer).unwrap();
        insert_job(&db, &running).unwrap();
        insert_job(&db, &older).unwrap();

        let got = next_queued_job(&db).unwrap().expect("queued job");
        assert_eq!(got.id, older.id);
    }

    #[test]
    fn queued_and_reset_helpers_clear_resumable_state() {
        let db = db();
        let row = ChainJobRow {
            state: ChainJobState::Failed,
            current_stage: 2,
            error: Some("boom".into()),
            ..job("01JBR55RESET", ChainJobState::Failed, 1_000)
        };
        insert_job(&db, &row).unwrap();
        upsert_stage(&db, &stage(&row.id, 0, StageState::Completed, 42)).unwrap();
        upsert_stage(&db, &stage(&row.id, 1, StageState::Completed, 43)).unwrap();
        upsert_stage(&db, &stage(&row.id, 2, StageState::Failed, 44)).unwrap();

        assert!(set_job_queued(&db, &row.id, 2, 4_000).unwrap());
        assert_eq!(reset_stages_from(&db, &row.id, 1, 4_001).unwrap(), 2);

        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.state, ChainJobState::Queued);
        assert_eq!(got.current_stage, 2);
        assert_eq!(got.error, None);

        let stages = stages_for_job(&db, &row.id).unwrap();
        assert_eq!(stages[0].state, StageState::Completed);
        assert_eq!(
            stages[0].segment_rel_path.as_deref(),
            Some("stages/000/segment.mp4")
        );
        for stage in &stages[1..] {
            assert_eq!(stage.state, StageState::Pending);
            assert_eq!(stage.frames_emitted, None);
            assert_eq!(stage.generation_time_ms, None);
            assert_eq!(stage.segment_rel_path, None);
            assert_eq!(stage.error, None);
        }
    }

    #[test]
    fn delete_stages_from_removes_trailing_rows() {
        let db = db();
        let row = job("01JBR55SHRINK", ChainJobState::Completed, 1_000);
        insert_job(&db, &row).unwrap();
        for idx in 0..4 {
            upsert_stage(
                &db,
                &stage(&row.id, idx, StageState::Completed, 42 + idx as u64),
            )
            .unwrap();
        }

        assert_eq!(delete_stages_from(&db, &row.id, 2).unwrap(), 2);

        let remaining = stages_for_job(&db, &row.id).unwrap();
        assert_eq!(
            remaining.iter().map(|s| s.stage_idx).collect::<Vec<_>>(),
            vec![0, 1],
            "only rows with stage_idx >= from_idx are removed"
        );
        assert_eq!(delete_stages_from(&db, &row.id, 2).unwrap(), 0);
        assert_eq!(delete_stages_from(&db, "missing", 0).unwrap(), 0);
    }

    #[test]
    fn update_stage_shape_persists_new_count_and_current_stage() {
        let db = db();
        let row = job("01JBR55SHAPE", ChainJobState::Completed, 1_000);
        insert_job(&db, &row).unwrap();

        assert!(update_stage_shape(&db, &row.id, 5, 3, 9_000).unwrap());

        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.stage_count, 5);
        assert_eq!(got.current_stage, 3);
        assert_eq!(got.updated_at_ms, 9_000);
        assert!(!update_stage_shape(&db, "missing", 1, 0, 9_000).unwrap());
    }

    #[test]
    fn set_request_json_rewrites_indexed_request() {
        let db = db();
        let row = job("01JBR55REQJSON", ChainJobState::Completed, 1_000);
        insert_job(&db, &row).unwrap();

        assert!(set_request_json(&db, &row.id, r#"{"model":"amended"}"#, 9_500).unwrap());

        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.request_json, r#"{"model":"amended"}"#);
        assert_eq!(got.updated_at_ms, 9_500);
        assert!(!set_request_json(&db, "missing", "{}", 9_500).unwrap());
    }

    #[test]
    fn repair_job_from_manifest_updates_query_index() {
        let db = db();
        let row = job("01JBR55REPAIR", ChainJobState::Running, 1_000);
        insert_job(&db, &row).unwrap();

        assert!(repair_job_from_manifest(
            &db,
            &row.id,
            ChainJobState::Failed,
            1,
            Some("manifest says failed"),
            5_000,
            Some(4_900),
        )
        .unwrap());

        let got = get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(got.state, ChainJobState::Failed);
        assert_eq!(got.current_stage, 1);
        assert_eq!(got.error.as_deref(), Some("manifest says failed"));
        assert_eq!(got.updated_at_ms, 5_000);
        assert_eq!(got.finalized_at_ms, Some(4_900));
    }
}
