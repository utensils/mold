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
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use rusqlite::{params, OptionalExtension, Row};

use crate::db::MetadataDb;

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
///
/// `owner_uuid` is supplied by the caller. The server claims it at boot with
/// an exclusive lock (`mold_server::queue_journal::claim_queue_owner`) rather
/// than deriving it from settings: two servers can share one `MOLD_HOME`, and
/// a derived identity would let the second adopt the first's running rows.
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
    /// Stable device id the user pinned, when they pinned one. Survives the
    /// renumbering that makes `target_gpu` unreliable across a restart.
    pub target_device_id: Option<String>,
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

/// Payload-free row used by the hot queue-listing path.
///
/// This is intentionally not a partial [`GenerationQueueRow`]. Keeping a
/// separate type makes it impossible for the listing query to accidentally
/// grow `request_json`, `output_dir`, or `completion_payload` back into the
/// async HTTP read path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationQueueProjection {
    pub id: String,
    pub state: QueueRowState,
    pub model: String,
    pub target_gpu: Option<usize>,
    pub seed_pinned: bool,
    pub dispatch_attempts: u32,
    pub replay_seen: u32,
    pub held_reason: Option<String>,
    pub created_at_ms: i64,
}

/// Exclusive keyset cursor for [`list_projection_page`]. `rowid` is SQLite's
/// stable tie-break for rows admitted in the same millisecond. The HTTP layer
/// encodes this value opaquely; database callers never parse wire cursors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueProjectionCursor {
    pub created_at_ms: i64,
    pub rowid: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationQueueProjectionPage {
    pub rows: Vec<GenerationQueueProjection>,
    pub next_cursor: Option<QueueProjectionCursor>,
}

/// The only column projection allowed on the paginated queue-listing path.
/// Keep this as a literal so the regression test can prove that none of the
/// durable payload columns are selected.
const QUEUE_PROJECTION_PAGE_SQL: &str = "
    SELECT q.id, q.state, q.model, q.target_gpu, q.seed_pinned,
           q.dispatch_attempts, q.replay_seen, q.held_reason, q.created_at,
           q.rowid,
           EXISTS (
               SELECT 1
                 FROM generation_queue AS later
                WHERE later.owner_uuid = q.owner_uuid
                  AND (later.created_at > q.created_at
                       OR (later.created_at = q.created_at AND later.rowid > q.rowid))
           ) AS has_later
      FROM generation_queue AS q
     WHERE q.owner_uuid = ?1
       AND (?2 IS NULL
            OR q.created_at > ?2
            OR (q.created_at = ?2 AND q.rowid > ?3))
     ORDER BY q.created_at, q.rowid
     LIMIT ?4";

/// One runtime reservation of an oldest queued row.
///
/// The token is deliberately kept outside [`GenerationQueueRow`] so existing
/// callers and struct literals remain source-compatible. Every mutation after
/// hydration must present it together with the expected durable state.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueClaim {
    pub row: GenerationQueueRow,
    pub claim_token: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeClaimRecovery {
    pub claims_cleared: usize,
    pub running_requeued: usize,
}

pub fn insert(db: &MetadataDb, row: &GenerationQueueRow) -> Result<()> {
    db.with_conn(|conn| insert_on_conn(conn, row))
}

pub(crate) fn insert_on_conn(conn: &rusqlite::Connection, row: &GenerationQueueRow) -> Result<()> {
    insert_on_conn_with_claim(conn, row, None)
}

/// Insert a row already owned by the live runtime that admitted it.
///
/// Direct HTTP generation submits the job to the in-memory scheduler itself,
/// so the durable feeder must not claim the same row concurrently. Startup
/// recovery clears this runtime-only token after a process death, at which
/// point the ordinary oldest-first feeder claim can safely adopt the row.
pub fn insert_claimed(db: &MetadataDb, row: &GenerationQueueRow, claim_token: &str) -> Result<()> {
    if claim_token.is_empty() {
        bail!("queue claim token must not be empty");
    }
    db.with_conn(|conn| insert_on_conn_with_claim(conn, row, Some(claim_token)))
}

fn insert_on_conn_with_claim(
    conn: &rusqlite::Connection,
    row: &GenerationQueueRow,
    claim_token: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO generation_queue (
                id, owner_uuid, state, model, request_json, output_dir,
                target_gpu, target_device_id, completion_payload, seed_pinned,
                dispatch_attempts, replay_seen, held_reason, created_at, updated_at,
                started_at, claim_token
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
        params![
            &row.id,
            &row.owner_uuid,
            row.state.as_str(),
            &row.model,
            &row.request_json,
            row.output_dir.to_string_lossy().into_owned(),
            row.target_gpu.map(|gpu| gpu as i64),
            row.target_device_id.as_deref(),
            &row.completion_payload,
            row.seed_pinned as i64,
            row.dispatch_attempts as i64,
            row.replay_seen as i64,
            row.held_reason.as_deref(),
            row.created_at_ms,
            row.updated_at_ms,
            row.started_at_ms,
            claim_token,
        ],
    )?;
    Ok(())
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

pub fn delete_legacy(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "DELETE FROM generation_queue
              WHERE id = ?1
                AND NOT EXISTS (
                    SELECT 1 FROM generation_batch_children WHERE job_id = ?1
                )",
            params![id],
        )? > 0)
    })
}

pub fn delete_all_queued_legacy(db: &MetadataDb, owner_uuid: &str) -> Result<usize> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "DELETE FROM generation_queue
              WHERE owner_uuid = ?1 AND state = 'queued'
                AND NOT EXISTS (
                    SELECT 1 FROM generation_batch_children AS child
                     WHERE child.job_id = generation_queue.id
                )",
            params![owner_uuid],
        )?)
    })
}

pub fn get(db: &MetadataDb, id: &str) -> Result<Option<GenerationQueueRow>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, target_device_id, completion_payload, seed_pinned,
                    dispatch_attempts, replay_seen, held_reason, created_at, updated_at,
                    started_at
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
                    target_gpu, target_device_id, completion_payload, seed_pinned,
                    dispatch_attempts, replay_seen, held_reason, created_at, updated_at,
                    started_at
             FROM generation_queue
             WHERE owner_uuid = ?1
             ORDER BY created_at, rowid",
        )?;
        let rows = stmt.query_map(params![owner_uuid], row_to_queue_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// One payload-free page of rows this installation owns, oldest first.
///
/// The cursor is an exclusive `(created_at, rowid)` key rather than a row id.
/// Deleting the row that produced a cursor therefore has no special case: a
/// later request deterministically continues after the same ordering key.
/// Inserts after that key are visible; inserts or reorders before it are not.
///
/// `limit` has no implicit default or server-side cap. The caller must supply
/// a positive value, and this query returns at most that many rows. `has_later`
/// is computed in the same SQLite statement so discovering the next page does
/// not require materializing a `limit + 1` row.
pub fn list_projection_page(
    db: &MetadataDb,
    owner_uuid: &str,
    cursor: Option<QueueProjectionCursor>,
    limit: usize,
) -> Result<GenerationQueueProjectionPage> {
    if limit == 0 {
        bail!("queue projection page limit must be positive");
    }
    let sql_limit = i64::try_from(limit)
        .map_err(|_| anyhow::anyhow!("queue projection page limit is outside SQLite's range"))?;
    let cursor_created_at = cursor.map(|cursor| cursor.created_at_ms);
    let cursor_rowid = cursor.map(|cursor| cursor.rowid);

    db.with_conn(|conn| {
        let mut stmt = conn.prepare(QUEUE_PROJECTION_PAGE_SQL)?;
        let mapped = stmt.query_map(
            params![owner_uuid, cursor_created_at, cursor_rowid, sql_limit],
            |row| {
                let state_raw: String = row.get(1)?;
                let state = QueueRowState::parse(&state_raw).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Text,
                        format!("unknown generation_queue state '{state_raw}'").into(),
                    )
                })?;
                Ok((
                    GenerationQueueProjection {
                        id: row.get(0)?,
                        state,
                        model: row.get(2)?,
                        target_gpu: row.get::<_, Option<i64>>(3)?.map(|gpu| gpu as usize),
                        seed_pinned: row.get::<_, i64>(4)? != 0,
                        dispatch_attempts: row.get::<_, i64>(5)? as u32,
                        replay_seen: row.get::<_, i64>(6)? as u32,
                        held_reason: row.get(7)?,
                        created_at_ms: row.get(8)?,
                    },
                    QueueProjectionCursor {
                        created_at_ms: row.get(8)?,
                        rowid: row.get(9)?,
                    },
                    row.get::<_, i64>(10)? != 0,
                ))
            },
        )?;
        let rows_with_keys = mapped.collect::<rusqlite::Result<Vec<_>>>()?;
        let next_cursor = rows_with_keys
            .last()
            .filter(|(_, _, has_later)| *has_later)
            .map(|(_, cursor, _)| *cursor);
        let rows = rows_with_keys.into_iter().map(|(row, _, _)| row).collect();
        Ok(GenerationQueueProjectionPage { rows, next_cursor })
    })
}

/// Which live registry ids also have a durable row owned by this server.
///
/// The registry is bounded by the runtime queue capacity. Probe only those
/// ids so the paginated route can expose the complementary active,
/// non-durable set without scanning or materializing the deep journal.
pub fn find_owned_ids(
    db: &MetadataDb,
    owner_uuid: &str,
    ids: &[String],
) -> Result<HashSet<String>> {
    db.with_conn(|conn| {
        let mut found = HashSet::new();
        let mut stmt = conn
            .prepare("SELECT 1 FROM generation_queue WHERE owner_uuid = ?1 AND id = ?2 LIMIT 1")?;
        for id in ids {
            if stmt
                .query_row(params![owner_uuid, id], |_| Ok(()))
                .optional()?
                .is_some()
            {
                found.insert(id.clone());
            }
        }
        Ok(found)
    })
}

/// Rows eligible for replay: this installation's, not held, oldest first.
pub fn list_replayable(db: &MetadataDb, owner_uuid: &str) -> Result<Vec<GenerationQueueRow>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, target_device_id, completion_payload, seed_pinned,
                    dispatch_attempts, replay_seen, held_reason, created_at, updated_at,
                    started_at
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

/// Reserve the oldest unclaimed queued row without materializing the backlog.
///
/// The single UPDATE is the concurrency boundary across processes. A token
/// may own at most one row; retrying an already-used token returns `None`
/// rather than advancing to another job.
pub fn claim_next(
    db: &MetadataDb,
    owner_uuid: &str,
    claim_token: &str,
    now_ms: i64,
) -> Result<Option<QueueClaim>> {
    if claim_token.is_empty() {
        bail!("queue claim token must not be empty");
    }
    db.with_conn(|conn| {
        conn.query_row(
            "UPDATE generation_queue
                SET claim_token = ?2, updated_at = ?3
              WHERE id = (
                    SELECT id
                      FROM generation_queue
                     WHERE owner_uuid = ?1
                       AND state = 'queued'
                       AND claim_token IS NULL
                       AND NOT EXISTS (
                            SELECT 1 FROM generation_queue WHERE claim_token = ?2
                       )
                     ORDER BY created_at, rowid
                     LIMIT 1
              )
                AND state = 'queued'
                AND claim_token IS NULL
          RETURNING id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, target_device_id, completion_payload, seed_pinned,
                    dispatch_attempts, replay_seen, held_reason, created_at, updated_at,
                    started_at",
            params![owner_uuid, claim_token, now_ms],
            |row| {
                Ok(QueueClaim {
                    row: row_to_queue_row(row)?,
                    claim_token: claim_token.to_string(),
                })
            },
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Clear a reservation that has not started execution.
pub fn release_claim(db: &MetadataDb, id: &str, claim_token: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_queue
                SET claim_token = NULL, updated_at = ?3
              WHERE id = ?1 AND state = 'queued' AND claim_token = ?2",
            params![id, claim_token, now_ms],
        )? > 0)
    })
}

/// Return an interrupted, token-owned execution attempt to the durable queue.
///
/// Unlike [`refund_dispatched_claim`], this preserves `dispatch_attempts`: the
/// owner did begin executing and shutdown/cancellation interrupted it, so the
/// crash-loop budget must still account for that attempt.
pub fn requeue_running_claimed(
    db: &MetadataDb,
    id: &str,
    claim_token: &str,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_queue
                SET state = 'queued',
                    claim_token = NULL,
                    started_at = NULL,
                    updated_at = ?3
              WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
            params![id, claim_token, now_ms],
        )? > 0)
    })
}

/// Token-fenced dispatch transition. Unlike legacy [`mark_dispatched`], a
/// stale runtime owner cannot charge or start the row.
pub fn mark_dispatched_claimed(
    db: &MetadataDb,
    id: &str,
    claim_token: &str,
    now_ms: i64,
) -> Result<Option<u32>> {
    db.with_conn(|conn| {
        conn.query_row(
            "UPDATE generation_queue
                SET state = 'running',
                    dispatch_attempts = dispatch_attempts + 1,
                    started_at = ?3,
                    updated_at = ?3
              WHERE id = ?1 AND state = 'queued' AND claim_token = ?2
          RETURNING dispatch_attempts",
            params![id, claim_token, now_ms],
            |row| row.get::<_, i64>(0).map(|count| count as u32),
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Refund a token-fenced dispatch that was never handed to the execution
/// owner. This is the inverse of [`mark_dispatched_claimed`]: the attempt is
/// decremented in the same CAS that returns the row to the queue.
pub fn refund_dispatched_claim(
    db: &MetadataDb,
    id: &str,
    claim_token: &str,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_queue
                SET state = 'queued',
                    claim_token = NULL,
                    dispatch_attempts = CASE
                        WHEN dispatch_attempts > 0 THEN dispatch_attempts - 1
                        ELSE 0
                    END,
                    started_at = NULL,
                    updated_at = ?3
              WHERE id = ?1 AND state = 'running' AND claim_token = ?2",
            params![id, claim_token, now_ms],
        )? > 0)
    })
}

/// Park a claimed row only while the caller still owns the exact runtime
/// token. The token is retained as forensic ownership and is cleared by the
/// ordinary startup recovery pass.
pub fn hold_claimed(
    db: &MetadataDb,
    id: &str,
    claim_token: &str,
    expected_state: QueueRowState,
    reason: &str,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_queue
                SET state = 'held', held_reason = ?4, updated_at = ?5
              WHERE id = ?1 AND state = ?2 AND claim_token = ?3",
            params![id, expected_state.as_str(), claim_token, reason, now_ms],
        )? > 0)
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
    Ok(recover_runtime_claims(db, owner_uuid, now_ms)?.running_requeued)
}

/// Startup recovery for runtime-only ownership.
///
/// All running rows are replayable after a process death, including legacy
/// untokened rows. Every token is cleared in the same statement, so a stale
/// feeder or worker from the prior runtime is fenced from later CAS writes.
pub fn recover_runtime_claims(
    db: &MetadataDb,
    owner_uuid: &str,
    now_ms: i64,
) -> Result<RuntimeClaimRecovery> {
    db.transact_immediate(|conn| {
        let claims_cleared: i64 = conn.query_row(
            "SELECT COUNT(*) FROM generation_queue
              WHERE owner_uuid = ?1 AND claim_token IS NOT NULL",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        let running_requeued: i64 = conn.query_row(
            "SELECT COUNT(*) FROM generation_queue
              WHERE owner_uuid = ?1 AND state = 'running'",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        conn.execute(
            "UPDATE generation_queue
                SET state = CASE WHEN state = 'running' THEN 'queued' ELSE state END,
                    claim_token = NULL,
                    started_at = CASE WHEN state = 'running' THEN NULL ELSE started_at END,
                    updated_at = ?2
              WHERE owner_uuid = ?1
                AND (state = 'running' OR claim_token IS NOT NULL)",
            params![owner_uuid, now_ms],
        )?;
        Ok(RuntimeClaimRecovery {
            claims_cleared: claims_cleared as usize,
            running_requeued: running_requeued as usize,
        })
    })
}

/// Re-lane a row after `PATCH /api/queue/:id` moved it.
pub fn set_target_gpu(
    db: &MetadataDb,
    id: &str,
    target_gpu: Option<usize>,
    target_device_id: Option<&str>,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let updated = conn.execute(
            "UPDATE generation_queue
                SET target_gpu = ?2, target_device_id = ?3, updated_at = ?4
              WHERE id = ?1",
            params![
                id,
                target_gpu.map(|gpu| gpu as i64),
                target_device_id,
                now_ms
            ],
        )?;
        Ok(updated > 0)
    })
}

/// Re-stamp `created_at` so replay follows `ids` in the order given.
///
/// `created_at` is the replay sort key, so a queue reorder has to move it or
/// the restart quietly restores the admission order. Values are rewritten as a
/// dense sequence anchored at the oldest row's existing timestamp, which keeps
/// them plausible wall-clock times and makes the pass self-healing: any prior
/// drift is corrected. Ids absent from the table are ignored, and rows absent
/// from `ids` (held work) keep their stamps.
pub fn apply_queue_order(db: &MetadataDb, owner_uuid: &str, ids: &[String]) -> Result<usize> {
    if ids.is_empty() {
        return Ok(0);
    }
    db.with_conn(|conn| {
        let anchor: Option<i64> = conn.query_row(
            "SELECT MIN(created_at) FROM generation_queue WHERE owner_uuid = ?1",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        let Some(anchor) = anchor else {
            return Ok(0);
        };
        let mut stmt = conn.prepare(
            "UPDATE generation_queue
                SET created_at = ?2, updated_at = ?3
              WHERE id = ?1 AND owner_uuid = ?4",
        )?;
        let mut moved = 0;
        for (index, id) in ids.iter().enumerate() {
            let stamp = anchor.saturating_add(index as i64);
            moved += stmt.execute(params![id, stamp, stamp, owner_uuid])?;
        }
        Ok(moved)
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

/// Exact gallery identity recovered for a durable child that published its
/// output before its queue authority was deleted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedGenerationOutput {
    /// The ordinary payload (or the post-generation upscaled payload).
    pub filename: String,
    /// The separately saved pre-upscale image, when present.
    pub original_filename: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletedOutputKind {
    Primary,
    Original,
    Upscaled,
}

fn completed_output_kind(filename: &str) -> Result<CompletedOutputKind> {
    let path = Path::new(filename);
    if path.file_name().and_then(|name| name.to_str()) != Some(filename) {
        bail!("completed generation filename is not one gallery basename");
    }
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("completed generation filename has no UTF-8 stem"))?;
    let stem = mold_core::strip_title_slug(stem);
    if stem.ends_with("-original") {
        Ok(CompletedOutputKind::Original)
    } else if stem.ends_with("-upscaled") {
        Ok(CompletedOutputKind::Upscaled)
    } else {
        Ok(CompletedOutputKind::Primary)
    }
}

/// Recover one completed child's exact saved output from its owned gallery.
///
/// The queue row supplies both the owner fence and output-directory scope.
/// Every metadata document in that gallery is parsed rather than asking
/// SQLite's permissive JSON projection to guess: malformed metadata or an
/// ambiguous set of matching rows is an error, because deleting queue
/// authority in either case could make reconnect report the wrong output.
pub fn find_completed_output(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
) -> Result<Option<CompletedGenerationOutput>> {
    let output_dir: Option<String> = db.with_conn(|conn| {
        conn.query_row(
            "SELECT output_dir FROM generation_queue
              WHERE id = ?1 AND owner_uuid = ?2",
            params![job_id, owner_uuid],
            |row| row.get(0),
        )
        .optional()
        .map_err(Into::into)
    })?;
    let Some(output_dir) = output_dir else {
        return Ok(None);
    };
    // Canonicalization may touch the filesystem. Keep it outside the DB mutex
    // so a slow gallery mount cannot stall unrelated metadata readers.
    let output_dir = crate::canonical_dir_string(Path::new(&output_dir));
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT filename, metadata_json FROM generations
              WHERE output_dir = ?1 AND metadata_json IS NOT NULL
              ORDER BY id",
        )?;
        let rows = stmt.query_map(params![output_dir], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut primary = Vec::new();
        let mut originals = Vec::new();
        let mut upscaled = Vec::new();
        for row in rows {
            let (filename, metadata_json) = row?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
                .map_err(|error| anyhow::anyhow!("malformed gallery metadata: {error}"))?;
            let Some(recorded_job_id) = metadata.get("job_id") else {
                continue;
            };
            let recorded_job_id = recorded_job_id
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("gallery metadata job_id is not a string"))?;
            if recorded_job_id != job_id {
                continue;
            }
            match completed_output_kind(&filename)? {
                CompletedOutputKind::Primary => primary.push(filename),
                CompletedOutputKind::Original => originals.push(filename),
                CompletedOutputKind::Upscaled => upscaled.push(filename),
            }
        }

        match (
            primary.as_slice(),
            originals.as_slice(),
            upscaled.as_slice(),
        ) {
            ([], [], []) => Ok(None),
            ([filename], [], []) => Ok(Some(CompletedGenerationOutput {
                filename: filename.clone(),
                original_filename: None,
            })),
            ([], [original], [filename]) => Ok(Some(CompletedGenerationOutput {
                filename: filename.clone(),
                original_filename: Some(original.clone()),
            })),
            _ => bail!("gallery rows for queue job {job_id} are ambiguous"),
        }
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
        target_device_id: row.get(7)?,
        completion_payload: row.get(8)?,
        seed_pinned: row.get::<_, i64>(9)? != 0,
        dispatch_attempts: row.get::<_, i64>(10)? as u32,
        replay_seen: row.get::<_, i64>(11)? as u32,
        held_reason: row.get(12)?,
        created_at_ms: row.get(13)?,
        updated_at_ms: row.get(14)?,
        started_at_ms: row.get(15)?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

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
            target_device_id: None,
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
    fn projection_page_is_payload_free_bounded_and_keeps_all_states() {
        let db = MetadataDb::open_in_memory().unwrap();
        for (id, state) in [
            ("queued", QueueRowState::Queued),
            ("running", QueueRowState::Running),
            ("held", QueueRowState::Held),
        ] {
            let mut stored = row(id, "owner-a", 500);
            stored.state = state;
            stored.request_json = format!(
                r#"{{"source_image":"{}"}}"#,
                "inline-media".repeat(256 * 1024)
            );
            stored.output_dir = PathBuf::from(format!("/payload/{id}"));
            stored.completion_payload = format!("full-{id}");
            stored.target_gpu = Some(2);
            stored.seed_pinned = true;
            stored.dispatch_attempts = 3;
            stored.replay_seen = 1;
            stored.held_reason = (state == QueueRowState::Held).then(|| "review".to_string());
            insert(&db, &stored).unwrap();
        }

        let first = list_projection_page(&db, "owner-a", None, 2).unwrap();
        assert_eq!(first.rows.len(), 2, "the SQL limit is the result bound");
        assert!(first.next_cursor.is_some());
        assert_eq!(
            first.rows.iter().map(|row| row.state).collect::<Vec<_>>(),
            vec![QueueRowState::Queued, QueueRowState::Running]
        );

        // Exhaustive destructuring makes this test fail to compile if payload
        // fields are ever added to the projection type.
        let GenerationQueueProjection {
            id,
            state: _,
            model: _,
            target_gpu,
            seed_pinned,
            dispatch_attempts,
            replay_seen,
            held_reason: _,
            created_at_ms: _,
        } = &first.rows[0];
        assert_eq!(id, "queued");
        assert_eq!(*target_gpu, Some(2));
        assert!(*seed_pinned);
        assert_eq!(*dispatch_attempts, 3);
        assert_eq!(*replay_seen, 1);

        let second = list_projection_page(&db, "owner-a", first.next_cursor, 2).unwrap();
        assert_eq!(
            second
                .rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["held"]
        );
        assert!(second.next_cursor.is_none());
    }

    #[test]
    fn projection_sql_never_selects_durable_payload_columns() {
        let normalized = QUEUE_PROJECTION_PAGE_SQL.to_ascii_lowercase();
        for forbidden in ["request_json", "output_dir", "completion_payload"] {
            assert!(
                !normalized.contains(forbidden),
                "projection SQL selected {forbidden}: {QUEUE_PROJECTION_PAGE_SQL}"
            );
        }
    }

    #[test]
    fn projection_cursor_continues_after_its_row_is_deleted() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "second", "third"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }

        let first = list_projection_page(&db, "owner-a", None, 2).unwrap();
        let cursor = first.next_cursor.expect("third row requires another page");
        assert_eq!(first.rows[1].id, "second");
        delete(&db, "second").unwrap();
        insert(&db, &row("admitted-after-cursor", "owner-a", 500)).unwrap();

        let second = list_projection_page(&db, "owner-a", Some(cursor), 10).unwrap();
        assert_eq!(
            second
                .rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["third", "admitted-after-cursor"]
        );
        assert!(second.next_cursor.is_none());
    }

    #[test]
    fn projection_page_rejects_a_zero_limit_and_scopes_live_id_probes() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("mine", "owner-a", 1)).unwrap();
        insert(&db, &row("theirs", "owner-b", 2)).unwrap();
        assert!(list_projection_page(&db, "owner-a", None, 0).is_err());
        assert_eq!(
            find_owned_ids(
                &db,
                "owner-a",
                &[
                    "mine".to_string(),
                    "theirs".to_string(),
                    "missing".to_string()
                ]
            )
            .unwrap(),
            HashSet::from(["mine".to_string()])
        );
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
    fn apply_queue_order_rewrites_the_replay_order() {
        let db = MetadataDb::open_in_memory().unwrap();
        for (id, created) in [("first", 100), ("second", 200), ("third", 300)] {
            insert(&db, &row(id, "owner-a", created)).unwrap();
        }
        insert(&db, &row("parked", "owner-a", 400)).unwrap();
        hold(&db, "parked", "held for review", 500).unwrap();

        let moved = apply_queue_order(
            &db,
            "owner-a",
            &[
                "third".to_string(),
                "first".to_string(),
                "second".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(moved, 3);

        let ids: Vec<String> = list_replayable(&db, "owner-a")
            .unwrap()
            .into_iter()
            .map(|row| row.id)
            .collect();
        assert_eq!(ids, vec!["third", "first", "second"]);
        assert_eq!(
            get(&db, "parked").unwrap().unwrap().created_at_ms,
            400,
            "held rows are not part of the dispatch order"
        );
    }

    #[test]
    fn concurrent_claims_are_exclusive_and_duplicate_tokens_claim_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        insert(&db, &row("only", "owner-a", 1)).unwrap();
        drop(db);

        let barrier = Arc::new(Barrier::new(2));
        let handles = ["claim-a", "claim-b"].map(|token| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let db = MetadataDb::open(&path).unwrap();
                barrier.wait();
                claim_next(&db, "owner-a", token, 10).unwrap()
            })
        });
        let claimed = handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].row.id, "only");

        let db = MetadataDb::open(&path).unwrap();
        insert(&db, &row("next", "owner-a", 2)).unwrap();
        assert!(claim_next(&db, "owner-a", &claimed[0].claim_token, 20)
            .unwrap()
            .is_none());
        assert_eq!(
            claim_next(&db, "owner-a", "fresh-token", 21)
                .unwrap()
                .unwrap()
                .row
                .id,
            "next"
        );
    }

    #[test]
    fn runtime_owned_insert_is_invisible_until_startup_recovery() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_claimed(&db, &row("live-direct", "owner-a", 1), "live-token").unwrap();

        assert!(
            claim_next(&db, "owner-a", "feeder-before-restart", 10)
                .unwrap()
                .is_none(),
            "a live direct submitter owns the row and the feeder must not race it"
        );

        let recovered = recover_runtime_claims(&db, "owner-a", 20).unwrap();
        assert_eq!(recovered.claims_cleared, 1);
        assert_eq!(recovered.running_requeued, 0);

        let replay = claim_next(&db, "owner-a", "feeder-after-restart", 21)
            .unwrap()
            .expect("the dead runtime's direct row becomes feeder-owned after recovery");
        assert_eq!(replay.row.id, "live-direct");
        assert_eq!(replay.claim_token, "feeder-after-restart");
    }

    #[test]
    fn state_and_claim_cas_fences_stale_tokens() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("job-1", "owner-a", 1)).unwrap();
        let claimed = claim_next(&db, "owner-a", "current", 10).unwrap().unwrap();
        assert_eq!(claimed.row.state, QueueRowState::Queued);

        assert_eq!(
            mark_dispatched_claimed(&db, "job-1", "stale", 11).unwrap(),
            None
        );
        assert_eq!(
            mark_dispatched_claimed(&db, "job-1", "current", 12).unwrap(),
            Some(1)
        );
        assert_eq!(
            get(&db, "job-1").unwrap().unwrap().state,
            QueueRowState::Running
        );
        assert!(!release_claim(&db, "job-1", "current", 13).unwrap());
    }

    #[test]
    fn release_and_dispatch_refund_are_token_fenced() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("release", "owner-a", 1)).unwrap();
        insert(&db, &row("refund", "owner-a", 2)).unwrap();
        insert(&db, &row("retain-running", "owner-a", 3)).unwrap();

        claim_next(&db, "owner-a", "release-token", 10).unwrap();
        assert!(!release_claim(&db, "release", "stale", 11).unwrap());
        assert!(release_claim(&db, "release", "release-token", 12).unwrap());
        delete(&db, "release").unwrap();

        claim_next(&db, "owner-a", "refund-token", 13).unwrap();
        assert_eq!(
            mark_dispatched_claimed(&db, "refund", "refund-token", 14).unwrap(),
            Some(1)
        );
        assert!(!refund_dispatched_claim(&db, "refund", "stale", 15).unwrap());
        assert!(refund_dispatched_claim(&db, "refund", "refund-token", 16).unwrap());
        let refunded = get(&db, "refund").unwrap().unwrap();
        assert_eq!(refunded.state, QueueRowState::Queued);
        assert_eq!(refunded.dispatch_attempts, 0);
        assert_eq!(refunded.started_at_ms, None);
        delete(&db, "refund").unwrap();

        claim_next(&db, "owner-a", "retain-token", 17).unwrap();
        assert_eq!(
            mark_dispatched_claimed(&db, "retain-running", "retain-token", 18).unwrap(),
            Some(1)
        );
        assert!(!requeue_running_claimed(&db, "retain-running", "stale", 19).unwrap());
        assert!(requeue_running_claimed(&db, "retain-running", "retain-token", 20).unwrap());
        let retained = get(&db, "retain-running").unwrap().unwrap();
        assert_eq!(retained.state, QueueRowState::Queued);
        assert_eq!(retained.dispatch_attempts, 1);
        assert_eq!(retained.started_at_ms, None);
    }

    #[test]
    fn recovery_clears_queued_claims_requeues_running_and_fences_old_tokens() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("queued", "owner-a", 1)).unwrap();
        insert(&db, &row("running", "owner-a", 2)).unwrap();
        claim_next(&db, "owner-a", "token-queued", 10).unwrap();
        claim_next(&db, "owner-a", "token-running", 11).unwrap();
        mark_dispatched_claimed(&db, "running", "token-running", 12).unwrap();

        let recovered = recover_runtime_claims(&db, "owner-a", 20).unwrap();
        assert_eq!(recovered.claims_cleared, 2);
        assert_eq!(recovered.running_requeued, 1);
        assert_eq!(
            get(&db, "queued").unwrap().unwrap().state,
            QueueRowState::Queued
        );
        assert_eq!(
            get(&db, "running").unwrap().unwrap().state,
            QueueRowState::Queued
        );
        assert_eq!(
            mark_dispatched_claimed(&db, "running", "token-running", 21).unwrap(),
            None
        );
    }

    #[test]
    fn repeated_claim_next_is_bounded_and_preserves_existing_queue_order() {
        let db = MetadataDb::open_in_memory().unwrap();
        for index in 0..25 {
            insert(&db, &row(&format!("job-{index}"), "owner-a", 1)).unwrap();
        }

        for index in 0..3 {
            let claim = claim_next(&db, "owner-a", &format!("claim-{index}"), 10 + index)
                .unwrap()
                .unwrap();
            assert_eq!(claim.row.id, format!("job-{index}"));
        }
        assert_eq!(
            list_replayable(&db, "owner-a")
                .unwrap()
                .into_iter()
                .filter(|row| row.state == QueueRowState::Queued)
                .count(),
            25,
            "claiming is a one-row SQL primitive; it does not materialize or remove the backlog"
        );
    }

    #[test]
    fn set_target_gpu_relanes_a_row() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("job-1", "owner-a", 1)).unwrap();

        assert!(set_target_gpu(&db, "job-1", Some(3), Some("cuda:abc"), 9).unwrap());
        let pinned = get(&db, "job-1").unwrap().unwrap();
        assert_eq!(pinned.target_gpu, Some(3));
        assert_eq!(
            pinned.target_device_id.as_deref(),
            Some("cuda:abc"),
            "the stable pin is what survives an ordinal renumbering"
        );

        assert!(set_target_gpu(&db, "job-1", None, None, 10).unwrap());
        let auto = get(&db, "job-1").unwrap().unwrap();
        assert_eq!(auto.target_gpu, None);
        assert_eq!(auto.target_device_id, None);
        assert!(!set_target_gpu(&db, "missing", Some(1), None, 11).unwrap());
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
    fn completed_output_recovers_exact_primary_and_upscaled_names_in_its_owned_gallery() {
        let db = MetadataDb::open_in_memory().unwrap();
        let gallery = tempfile::tempdir().unwrap();
        let other_gallery = tempfile::tempdir().unwrap();
        let mut queued = row("done", "owner-a", 1);
        queued.output_dir = gallery.path().to_path_buf();
        insert(&db, &queued).unwrap();

        db.with_conn(|conn| {
            for (filename, output_dir, metadata) in [
                (
                    "mold-flux-1-original~portrait.png",
                    gallery.path(),
                    r#"{"job_id":"done","seed":7}"#,
                ),
                (
                    "mold-flux-2-upscaled~portrait.png",
                    gallery.path(),
                    r#"{"job_id":"done","seed":7}"#,
                ),
                (
                    "wrong-gallery.png",
                    other_gallery.path(),
                    r#"{"job_id":"done","seed":7}"#,
                ),
            ] {
                conn.execute(
                    "INSERT INTO generations
                        (filename, output_dir, created_at_ms, format, model, metadata_json)
                     VALUES (?1, ?2, 1, 'png', 'flux-dev:q4', ?3)",
                    params![filename, crate::canonical_dir_string(output_dir), metadata],
                )?;
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(
            find_completed_output(&db, "owner-a", "done").unwrap(),
            Some(CompletedGenerationOutput {
                filename: "mold-flux-2-upscaled~portrait.png".to_string(),
                original_filename: Some("mold-flux-1-original~portrait.png".to_string()),
            })
        );
        assert_eq!(find_completed_output(&db, "owner-b", "done").unwrap(), None);
    }

    #[test]
    fn completed_output_recovers_one_ordinary_primary_name() {
        let db = MetadataDb::open_in_memory().unwrap();
        let gallery = tempfile::tempdir().unwrap();
        let mut queued = row("done", "owner-a", 1);
        queued.output_dir = gallery.path().to_path_buf();
        insert(&db, &queued).unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO generations
                    (filename, output_dir, created_at_ms, format, model, metadata_json)
                 VALUES ('print.mp4', ?1, 1, 'mp4', 'wan', ?2)",
                params![
                    crate::canonical_dir_string(gallery.path()),
                    r#"{"job_id":"done","seed":7}"#
                ],
            )?;
            Ok(())
        })
        .unwrap();

        assert_eq!(
            find_completed_output(&db, "owner-a", "done").unwrap(),
            Some(CompletedGenerationOutput {
                filename: "print.mp4".to_string(),
                original_filename: None,
            })
        );
    }

    #[test]
    fn completed_output_fails_closed_on_ambiguous_or_malformed_gallery_rows() {
        for rows in [
            vec![
                ("first.png", r#"{"job_id":"done"}"#),
                ("second.png", r#"{"job_id":"done"}"#),
            ],
            vec![("broken.png", r#"{"job_id":"done""#)],
        ] {
            let db = MetadataDb::open_in_memory().unwrap();
            let gallery = tempfile::tempdir().unwrap();
            let mut queued = row("done", "owner-a", 1);
            queued.output_dir = gallery.path().to_path_buf();
            insert(&db, &queued).unwrap();
            db.with_conn(|conn| {
                for (filename, metadata) in &rows {
                    conn.execute(
                        "INSERT INTO generations
                            (filename, output_dir, created_at_ms, format, model, metadata_json)
                         VALUES (?1, ?2, 1, 'png', 'flux-dev:q4', ?3)",
                        params![
                            filename,
                            crate::canonical_dir_string(gallery.path()),
                            metadata
                        ],
                    )?;
                }
                Ok(())
            })
            .unwrap();

            assert!(find_completed_output(&db, "owner-a", "done").is_err());
            assert!(get(&db, "done").unwrap().is_some());
        }
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
