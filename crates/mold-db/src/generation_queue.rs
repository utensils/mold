//! Durable singleton generation admission queue: typed CRUD over the
//! `generation_queue` table (schema v18).
//!
//! Unlike [`crate::chain_jobs`] there is no companion manifest — a queued
//! singleton owns no artifacts, so the row is the whole state and the DB is
//! the source of truth. A row present at startup means this installation died
//! owing that output.
//!
//! Two counters bound replay, and they are deliberately charged at different
//! moments. `dispatch_attempts` is incremented by the same statement that
//! claims the row for execution ([`mark_dispatched`]), so a job that waits
//! behind a long render through ten deploys is charged zero and a job that
//! kills the process during its own load is held. `replay_seen` is incremented
//! once per boot that replays the row, which is the only bound on a row that
//! loops without ever being claimed.
//!
//! Free functions over [`MetadataDb`], matching `chain_jobs.rs`.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use rusqlite::{params, OptionalExtension, Row};

use crate::db::MetadataDb;

/// Settings key holding this installation's journal identity.
const OWNER_UUID_KEY: &str = "queue.owner_uuid";

/// Lifecycle of a journal row. Deliberately narrow: the row records what to do
/// next, not a full job history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueRowState {
    /// Admitted, not yet claimed by a worker. Replayed at boot.
    Queued,
    /// Claimed by a worker. Flipped back to `Queued` by startup reconcile.
    Running,
    /// Exceeded an attempt cap or failed to reconcile. Listed, never auto-run.
    Held,
}

impl QueueRowState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Held => "held",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "held" => Some(Self::Held),
            _ => None,
        }
    }
}

/// One row of `generation_queue`, mirroring the v18 DDL 1:1.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationQueueRow {
    pub id: String,
    pub owner_uuid: String,
    pub state: QueueRowState,
    pub model: String,
    /// Canonical serde_json of the admitted `GenerateRequest`. Compare parsed
    /// values, not strings.
    pub request_json: String,
    pub output_dir: PathBuf,
    pub target_gpu: Option<usize>,
    /// `"full"` or `"metadata_only"` — the SSE completion payload the original
    /// caller asked for. Recorded so a replayed row keeps the same shape.
    pub completion_payload: String,
    pub seed_pinned: bool,
    pub dispatch_attempts: u32,
    pub replay_seen: u32,
    pub held_reason: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub started_at_ms: Option<i64>,
}

/// Resolve (creating on first use) this installation's journal identity.
///
/// Deliberately distinct from `instance_id`, which is scoped to
/// `(data dir, port)`: a server that comes back on a different port must still
/// replay its own queue rather than orphaning every row.
pub fn resolve_owner_uuid(db: &MetadataDb) -> Result<String> {
    let settings = crate::settings::Settings::new(db);
    if let Some(existing) = settings.get_str(OWNER_UUID_KEY)? {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    let fresh = uuid::Uuid::new_v4().to_string();
    settings.set_str(OWNER_UUID_KEY, &fresh)?;
    Ok(fresh)
}

pub fn insert(db: &MetadataDb, row: &GenerationQueueRow) -> Result<()> {
    db.with_conn(|conn| {
        conn.execute(
            "INSERT INTO generation_queue (
                id, owner_uuid, state, model, request_json, output_dir,
                target_gpu, completion_payload, seed_pinned, dispatch_attempts,
                replay_seen, held_reason, created_at, updated_at, started_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                &row.id,
                &row.owner_uuid,
                row.state.as_str(),
                &row.model,
                &row.request_json,
                row.output_dir.to_string_lossy().into_owned(),
                row.target_gpu.map(|gpu| gpu as i64),
                &row.completion_payload,
                row.seed_pinned as i64,
                row.dispatch_attempts as i64,
                row.replay_seen as i64,
                row.held_reason.as_deref(),
                row.created_at_ms,
                row.updated_at_ms,
                row.started_at_ms,
            ],
        )?;
        Ok(())
    })
}

/// Remove one row. Returns whether it existed.
pub fn delete(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        let removed = conn.execute("DELETE FROM generation_queue WHERE id = ?1", params![id])?;
        Ok(removed > 0)
    })
}

/// Remove every row this installation owns that has not yet been claimed.
/// Backs `DELETE /api/queue`, whose registry side cancels the same rows.
pub fn delete_all_queued(db: &MetadataDb, owner_uuid: &str) -> Result<usize> {
    db.with_conn(|conn| {
        let removed = conn.execute(
            "DELETE FROM generation_queue WHERE owner_uuid = ?1 AND state = 'queued'",
            params![owner_uuid],
        )?;
        Ok(removed)
    })
}

pub fn get(db: &MetadataDb, id: &str) -> Result<Option<GenerationQueueRow>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, completion_payload, seed_pinned, dispatch_attempts,
                    replay_seen, held_reason, created_at, updated_at, started_at
             FROM generation_queue WHERE id = ?1",
            params![id],
            row_to_queue_row,
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Every row this installation owns, in the order it was admitted.
///
/// `rowid` breaks ties for same-millisecond inserts, which SQLite orders
/// arbitrarily without it — replay ordering is the whole point of the table.
pub fn list_all(db: &MetadataDb, owner_uuid: &str) -> Result<Vec<GenerationQueueRow>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, completion_payload, seed_pinned, dispatch_attempts,
                    replay_seen, held_reason, created_at, updated_at, started_at
             FROM generation_queue
             WHERE owner_uuid = ?1
             ORDER BY created_at, rowid",
        )?;
        let rows = stmt.query_map(params![owner_uuid], row_to_queue_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Rows eligible for replay: this installation's, not held, oldest first.
pub fn list_replayable(db: &MetadataDb, owner_uuid: &str) -> Result<Vec<GenerationQueueRow>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, completion_payload, seed_pinned, dispatch_attempts,
                    replay_seen, held_reason, created_at, updated_at, started_at
             FROM generation_queue
             WHERE owner_uuid = ?1 AND state IN ('queued', 'running')
             ORDER BY created_at, rowid",
        )?;
        let rows = stmt.query_map(params![owner_uuid], row_to_queue_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Claim a row for execution: one statement that flips it to `running`,
/// charges an attempt, stamps `started_at`, and returns the new attempt count.
///
/// Single-statement on purpose — the caller is the GPU owner thread and the
/// count it acts on must be the one it wrote. Returns `None` when the row is
/// gone (it was cancelled, or the journal was disabled after admission).
pub fn mark_dispatched(db: &MetadataDb, id: &str, now_ms: i64) -> Result<Option<u32>> {
    db.with_conn(|conn| {
        conn.query_row(
            "UPDATE generation_queue
                SET state = 'running',
                    dispatch_attempts = dispatch_attempts + 1,
                    started_at = ?2,
                    updated_at = ?2
              WHERE id = ?1
          RETURNING dispatch_attempts",
            params![id, now_ms],
            |row| row.get::<_, i64>(0).map(|count| count as u32),
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Charge one boot's replay against a row and return the new count.
pub fn bump_replay_seen(db: &MetadataDb, id: &str, now_ms: i64) -> Result<Option<u32>> {
    db.with_conn(|conn| {
        conn.query_row(
            "UPDATE generation_queue
                SET replay_seen = replay_seen + 1,
                    updated_at = ?2
              WHERE id = ?1
          RETURNING replay_seen",
            params![id, now_ms],
            |row| row.get::<_, i64>(0).map(|count| count as u32),
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Park a row: listed by `GET /api/queue`, never auto-run.
pub fn hold(db: &MetadataDb, id: &str, reason: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|conn| {
        let updated = conn.execute(
            "UPDATE generation_queue
                SET state = 'held', held_reason = ?2, updated_at = ?3
              WHERE id = ?1",
            params![id, reason, now_ms],
        )?;
        Ok(updated > 0)
    })
}

/// Flip every `running` row this installation owns back to `queued`.
///
/// A `running` row at boot means the process died mid-dispatch; the state
/// column records what to do next, so it becomes `queued` again. Mirrors
/// `chain_job_runner`'s startup flip.
pub fn requeue_running(db: &MetadataDb, owner_uuid: &str, now_ms: i64) -> Result<usize> {
    db.with_conn(|conn| {
        let updated = conn.execute(
            "UPDATE generation_queue
                SET state = 'queued', started_at = NULL, updated_at = ?2
              WHERE owner_uuid = ?1 AND state = 'running'",
            params![owner_uuid, now_ms],
        )?;
        Ok(updated)
    })
}

/// Rewrite a row's output directory after startup re-resolution.
pub fn set_output_dir(db: &MetadataDb, id: &str, output_dir: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|conn| {
        let updated = conn.execute(
            "UPDATE generation_queue SET output_dir = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, output_dir, now_ms],
        )?;
        Ok(updated > 0)
    })
}

/// Which of `ids` already produced a gallery row.
///
/// The idempotence gate for replay: a print records the queue job that made it
/// in `OutputMetadata.job_id`, which lands in `generations.metadata_json`.
/// Without this, replay duplicates prints — output filenames are wall-clock,
/// so no client-side dedupe can merge them afterwards.
pub fn find_completed_job_ids(db: &MetadataDb, ids: &[String]) -> Result<HashSet<String>> {
    if ids.is_empty() {
        return Ok(HashSet::new());
    }
    db.with_conn(|conn| {
        let mut found = HashSet::new();
        let mut stmt = conn.prepare(
            "SELECT 1 FROM generations
              WHERE metadata_json IS NOT NULL
                AND json_extract(metadata_json, '$.job_id') = ?1
              LIMIT 1",
        )?;
        for id in ids {
            let hit: Option<i64> = stmt.query_row(params![id], |row| row.get(0)).optional()?;
            if hit.is_some() {
                found.insert(id.clone());
            }
        }
        Ok(found)
    })
}

fn row_to_queue_row(row: &Row<'_>) -> rusqlite::Result<GenerationQueueRow> {
    let state_raw: String = row.get(2)?;
    let state = QueueRowState::parse(&state_raw).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Text,
            format!("unknown generation_queue state '{state_raw}'").into(),
        )
    })?;
    let output_dir: String = row.get(5)?;
    Ok(GenerationQueueRow {
        id: row.get(0)?,
        owner_uuid: row.get(1)?,
        state,
        model: row.get(3)?,
        request_json: row.get(4)?,
        output_dir: PathBuf::from(output_dir),
        target_gpu: row.get::<_, Option<i64>>(6)?.map(|gpu| gpu as usize),
        completion_payload: row.get(7)?,
        seed_pinned: row.get::<_, i64>(8)? != 0,
        dispatch_attempts: row.get::<_, i64>(9)? as u32,
        replay_seen: row.get::<_, i64>(10)? as u32,
        held_reason: row.get(11)?,
        created_at_ms: row.get(12)?,
        updated_at_ms: row.get(13)?,
        started_at_ms: row.get(14)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, owner: &str, created_at_ms: i64) -> GenerationQueueRow {
        GenerationQueueRow {
            id: id.to_string(),
            owner_uuid: owner.to_string(),
            state: QueueRowState::Queued,
            model: "flux-dev:q4".to_string(),
            request_json: r#"{"prompt":"a cat"}"#.to_string(),
            output_dir: PathBuf::from("/gallery"),
            target_gpu: None,
            completion_payload: "full".to_string(),
            seed_pinned: false,
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
            started_at_ms: None,
        }
    }

    #[test]
    fn request_json_round_trips_byte_identically() {
        let db = MetadataDb::open_in_memory().unwrap();
        let request = serde_json::json!({
            "prompt": "a cat",
            "source_image": "iVBORw0KGgoAAAANSUhEUg==",
            "width": 1024,
        });
        let mut stored = row("job-1", "owner-a", 1);
        stored.request_json = serde_json::to_string(&request).unwrap();
        insert(&db, &stored).unwrap();

        let loaded = get(&db, "job-1").unwrap().unwrap();
        assert_eq!(loaded, stored);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&loaded.request_json).unwrap(),
            request
        );
    }

    #[test]
    fn list_replayable_orders_same_millisecond_inserts_by_insertion() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "second", "third"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }

        let ids: Vec<String> = list_replayable(&db, "owner-a")
            .unwrap()
            .into_iter()
            .map(|row| row.id)
            .collect();
        assert_eq!(ids, vec!["first", "second", "third"]);
    }

    #[test]
    fn list_replayable_is_scoped_to_the_owner_and_skips_held_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("mine", "owner-a", 1)).unwrap();
        insert(&db, &row("theirs", "owner-b", 2)).unwrap();
        insert(&db, &row("parked", "owner-a", 3)).unwrap();
        hold(&db, "parked", "dispatch attempts exhausted", 9).unwrap();

        let ids: Vec<String> = list_replayable(&db, "owner-a")
            .unwrap()
            .into_iter()
            .map(|row| row.id)
            .collect();
        assert_eq!(ids, vec!["mine"]);
        assert_eq!(list_all(&db, "owner-a").unwrap().len(), 2);
    }

    #[test]
    fn mark_dispatched_sets_running_and_returns_the_incremented_count() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("job-1", "owner-a", 1)).unwrap();

        assert_eq!(mark_dispatched(&db, "job-1", 100).unwrap(), Some(1));
        let after = get(&db, "job-1").unwrap().unwrap();
        assert_eq!(after.state, QueueRowState::Running);
        assert_eq!(after.dispatch_attempts, 1);
        assert_eq!(after.started_at_ms, Some(100));

        assert_eq!(mark_dispatched(&db, "job-1", 200).unwrap(), Some(2));
        assert_eq!(mark_dispatched(&db, "missing", 200).unwrap(), None);
    }

    #[test]
    fn replays_without_a_dispatch_never_charge_a_dispatch_attempt() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("job-1", "owner-a", 1)).unwrap();

        for boot in 1..=3 {
            assert_eq!(
                bump_replay_seen(&db, "job-1", boot).unwrap(),
                Some(boot as u32)
            );
        }

        let after = get(&db, "job-1").unwrap().unwrap();
        assert_eq!(
            after.dispatch_attempts, 0,
            "a job that only ever waited must not be charged for running"
        );
        assert_eq!(after.state, QueueRowState::Queued);
    }

    #[test]
    fn requeue_running_returns_interrupted_rows_to_the_queue() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("job-1", "owner-a", 1)).unwrap();
        insert(&db, &row("job-2", "owner-b", 1)).unwrap();
        mark_dispatched(&db, "job-1", 10).unwrap();
        mark_dispatched(&db, "job-2", 10).unwrap();

        assert_eq!(requeue_running(&db, "owner-a", 20).unwrap(), 1);
        let mine = get(&db, "job-1").unwrap().unwrap();
        assert_eq!(mine.state, QueueRowState::Queued);
        assert_eq!(mine.started_at_ms, None);
        assert_eq!(
            mine.dispatch_attempts, 1,
            "requeueing must not refund the attempt that killed the process"
        );
        assert_eq!(
            get(&db, "job-2").unwrap().unwrap().state,
            QueueRowState::Running,
            "another installation's rows are never touched"
        );
    }

    #[test]
    fn find_completed_job_ids_matches_the_saved_metadata_key() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO generations
                    (filename, output_dir, created_at_ms, format, model, metadata_json)
                 VALUES ('a.png', '/gallery', 1, 'png', 'flux-dev:q4', ?1)",
                params![r#"{"job_id":"done","seed":7}"#],
            )?;
            conn.execute(
                "INSERT INTO generations
                    (filename, output_dir, created_at_ms, format, model, metadata_json)
                 VALUES ('b.png', '/gallery', 2, 'png', 'flux-dev:q4', ?1)",
                params![r#"{"seed":8}"#],
            )?;
            Ok(())
        })
        .unwrap();

        let found =
            find_completed_job_ids(&db, &["done".to_string(), "pending".to_string()]).unwrap();
        assert_eq!(found, HashSet::from(["done".to_string()]));
        assert!(find_completed_job_ids(&db, &[]).unwrap().is_empty());
    }

    #[test]
    fn owner_uuid_is_stable_across_resolutions() {
        let db = MetadataDb::open_in_memory().unwrap();
        let first = resolve_owner_uuid(&db).unwrap();
        assert!(!first.is_empty());
        assert_eq!(resolve_owner_uuid(&db).unwrap(), first);
    }

    #[test]
    fn delete_all_queued_leaves_running_and_foreign_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("queued", "owner-a", 1)).unwrap();
        insert(&db, &row("running", "owner-a", 2)).unwrap();
        insert(&db, &row("theirs", "owner-b", 3)).unwrap();
        mark_dispatched(&db, "running", 5).unwrap();

        assert_eq!(delete_all_queued(&db, "owner-a").unwrap(), 1);
        assert!(get(&db, "queued").unwrap().is_none());
        assert!(get(&db, "running").unwrap().is_some());
        assert!(get(&db, "theirs").unwrap().is_some());
    }
}
