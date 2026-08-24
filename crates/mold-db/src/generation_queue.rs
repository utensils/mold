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

use anyhow::{bail, ensure, Result};
use rusqlite::{params, OptionalExtension, Row};

use crate::db::MetadataDb;
use crate::generation_queue_media::{self, QueueMediaObligation};

const COMPLETED_JOB_ID_LOOKUP_SQL: &str = "SELECT 1 FROM generations
      WHERE queue_job_metadata_state = 1
        AND queue_job_id = ?1
      LIMIT 1";

const INVALID_JOB_METADATA_LOOKUP_SQL: &str = "SELECT 1 FROM generations
      WHERE queue_job_metadata_state = 0
      LIMIT 1";

const UNKNOWN_JOB_METADATA_LOOKUP_SQL: &str = "SELECT metadata_json FROM generations
      WHERE queue_job_metadata_state IS NULL
      ORDER BY id";

const COMPLETED_OUTPUT_LOOKUP_SQL: &str = "SELECT filename FROM generations
      WHERE output_dir = ?1
        AND queue_job_metadata_state = 1
        AND queue_job_id = ?2
      ORDER BY id
      LIMIT 3";

const COMPLETED_OUTPUT_INVALID_METADATA_SQL: &str = "SELECT 1 FROM generations
      WHERE output_dir = ?1
        AND queue_job_metadata_state = 0
        AND created_at_ms >= ?2
      LIMIT 1";

const COMPLETED_OUTPUT_UNKNOWN_METADATA_SQL: &str =
    "SELECT filename, metadata_json FROM generations
      WHERE output_dir = ?1
        AND queue_job_metadata_state IS NULL
        AND created_at_ms >= ?2
      ORDER BY created_at_ms, id";

fn parse_queue_job_id(metadata_json: &str) -> Result<Option<String>> {
    let metadata: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| anyhow::anyhow!("malformed gallery metadata: {error}"))?;
    let Some(recorded_job_id) = metadata.get("job_id") else {
        return Ok(None);
    };
    Ok(Some(
        recorded_job_id
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("gallery metadata job_id is not a string"))?
            .to_string(),
    ))
}

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
    /// Opaque staged-media set owned by this queue row. The database never
    /// stores or interprets the set's members or filesystem layout.
    pub media_set_id: Option<String>,
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

/// Payload-free stable scan used to complete a bounded hydrated reorder with
/// every queued row that still lives only in SQLite.
const QUEUE_REORDER_CANDIDATES_SQL: &str = "
    SELECT id, created_at
      FROM generation_queue
     WHERE owner_uuid = ?1 AND state = 'queued'
     ORDER BY created_at, rowid";

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
    /// Recovery-active rows that spent one boot of replay budget.
    pub replays_charged: usize,
    /// Charged claims parked after exceeding the configured replay budget.
    pub replays_held: usize,
}

pub fn insert(db: &MetadataDb, row: &GenerationQueueRow) -> Result<()> {
    db.with_conn(|conn| insert_on_conn(conn, row))
}

pub(crate) fn insert_on_conn(conn: &rusqlite::Connection, row: &GenerationQueueRow) -> Result<()> {
    ensure_media_free(row)?;
    insert_on_conn_with_claim(conn, row, None)
}

/// Atomically create an active staged-media obligation and its queue row.
pub fn insert_with_media(
    db: &MetadataDb,
    row: &GenerationQueueRow,
    obligation: &QueueMediaObligation,
) -> Result<()> {
    db.transact_immediate(|conn| insert_on_conn_with_media(conn, row, obligation))
}

pub(crate) fn insert_on_conn_with_media(
    conn: &rusqlite::Connection,
    row: &GenerationQueueRow,
    obligation: &QueueMediaObligation,
) -> Result<()> {
    generation_queue_media::validate_row_obligation(
        row.media_set_id.as_deref(),
        &row.owner_uuid,
        obligation,
    )?;
    generation_queue_media::insert_active_on_conn(conn, obligation)?;
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
    ensure_media_free(row)?;
    db.with_conn(|conn| insert_on_conn_with_claim(conn, row, Some(claim_token)))
}

/// Atomically create an active staged-media obligation and a queue row already
/// reserved by the live runtime that admitted it.
pub fn insert_claimed_with_media(
    db: &MetadataDb,
    row: &GenerationQueueRow,
    claim_token: &str,
    obligation: &QueueMediaObligation,
) -> Result<()> {
    if claim_token.is_empty() {
        bail!("queue claim token must not be empty");
    }
    db.transact_immediate(|conn| {
        generation_queue_media::validate_row_obligation(
            row.media_set_id.as_deref(),
            &row.owner_uuid,
            obligation,
        )?;
        generation_queue_media::insert_active_on_conn(conn, obligation)?;
        insert_on_conn_with_claim(conn, row, Some(claim_token))
    })
}

fn ensure_media_free(row: &GenerationQueueRow) -> Result<()> {
    ensure!(
        row.media_set_id.is_none(),
        "queue rows with staged media require an atomic media insertion API"
    );
    Ok(())
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
                started_at, claim_token, media_set_id
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
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
            row.media_set_id.as_deref(),
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
                    started_at, media_set_id
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
                    started_at, media_set_id
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
                    started_at, media_set_id
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
                    started_at, media_set_id",
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

/// Reserve one exact queued row without changing durable queue order.
///
/// The caller has already selected `id` using its operation-aware policy. This
/// statement is only the ownership CAS: owner, id, queued state, absent claim,
/// and globally unused token must all still match. Ineligible or deleted rows
/// return `None` without changing timestamps or any other queue state. The
/// ordinary [`claim_next`] feeder therefore retains FIFO selection for every
/// remaining unclaimed row.
pub fn claim_by_id(
    db: &MetadataDb,
    owner_uuid: &str,
    id: &str,
    claim_token: &str,
    now_ms: i64,
) -> Result<Option<QueueClaim>> {
    if claim_token.is_empty() {
        bail!("queue claim token must not be empty");
    }
    db.with_conn(|conn| {
        conn.query_row(
            "UPDATE generation_queue
                SET claim_token = ?3, updated_at = ?4
              WHERE id = ?2
                AND owner_uuid = ?1
                AND state = 'queued'
                AND claim_token IS NULL
                AND NOT EXISTS (
                    SELECT 1 FROM generation_queue WHERE claim_token = ?3
                )
          RETURNING id, owner_uuid, state, model, request_json, output_dir,
                    target_gpu, target_device_id, completion_payload, seed_pinned,
                    dispatch_attempts, replay_seen, held_reason, created_at, updated_at,
                    started_at, media_set_id",
            params![owner_uuid, id, claim_token, now_ms],
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

/// Atomically quarantine selected staged-media jobs during startup.
///
/// The owner and non-NULL media marker fences prevent a corrupt/missing set
/// report from parking another installation's row or an ordinary media-free
/// job. The returned count is the number of unique requested jobs proven held
/// at commit, including rows a prior startup already held. Held rows retain
/// their marker and active cleanup obligation because the user may inspect or
/// explicitly cancel them later.
pub fn hold_media_jobs(
    db: &MetadataDb,
    owner_uuid: &str,
    job_ids: &[String],
    reason: &str,
    now_ms: i64,
) -> Result<usize> {
    if job_ids.is_empty() {
        return Ok(0);
    }
    db.transact_immediate(|conn| {
        let mut seen = HashSet::new();
        let mut stmt = conn.prepare(
            "UPDATE generation_queue
                SET state = 'held', held_reason = ?3, claim_token = NULL,
                    started_at = NULL, updated_at = ?4
              WHERE id = ?1 AND owner_uuid = ?2
                AND media_set_id IS NOT NULL
                AND state IN ('queued', 'running')",
        )?;
        let mut held = 0;
        let mut verify = conn.prepare(
            "SELECT 1 FROM generation_queue
              WHERE id = ?1 AND owner_uuid = ?2
                AND media_set_id IS NOT NULL AND state = 'held'",
        )?;
        for job_id in job_ids {
            if seen.insert(job_id.as_str()) {
                stmt.execute(params![job_id, owner_uuid, reason, now_ms])?;
                if verify.exists(params![job_id, owner_uuid])? {
                    held += 1;
                }
            }
        }
        Ok(held)
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
    recover_runtime_claims_inner(db, owner_uuid, now_ms, None)
}

/// Recover runtime ownership and atomically charge only recovery-active rows.
///
/// Untouched durable backlog rows have no claim token and spend no budget:
/// otherwise a deep queue could be held merely because the server was
/// deployed several times before those rows reached the bounded runtime
/// window. A claimed row enters recovery on its first boot and remains active
/// (`replay_seen > 0`) until it runs or is held, so a crash immediately after
/// token recovery cannot reset the budget. It spends one unit per recovery
/// boot. Exhaustion parks both queue authority and its batch-child summary in
/// the same transaction.
pub fn recover_runtime_claims_and_charge_replays(
    db: &MetadataDb,
    owner_uuid: &str,
    now_ms: i64,
    max_replay_seen: u32,
) -> Result<RuntimeClaimRecovery> {
    if max_replay_seen == 0 {
        bail!("replay budget must be at least one boot");
    }
    recover_runtime_claims_inner(db, owner_uuid, now_ms, Some(max_replay_seen))
}

fn recover_runtime_claims_inner(
    db: &MetadataDb,
    owner_uuid: &str,
    now_ms: i64,
    max_replay_seen: Option<u32>,
) -> Result<RuntimeClaimRecovery> {
    db.transact_immediate(|conn| {
        let claims_cleared: i64 = conn.query_row(
            "SELECT COUNT(*) FROM generation_queue
              WHERE owner_uuid = ?1 AND claim_token IS NOT NULL",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        let mut replays_charged = 0usize;
        let mut replays_held = 0usize;
        if let Some(cap) = max_replay_seen {
            let claimed = {
                let mut stmt = conn.prepare(
                    "SELECT id, replay_seen
                      FROM generation_queue
                      WHERE owner_uuid = ?1 AND state IN ('queued', 'running')
                        AND (state = 'running' OR claim_token IS NOT NULL OR replay_seen > 0)
                      ORDER BY created_at, rowid",
                )?;
                let rows = stmt
                    .query_map(params![owner_uuid], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                rows
            };
            for (id, previous_seen) in claimed {
                let previous_seen = u32::try_from(previous_seen)
                    .map_err(|_| anyhow::anyhow!("queue row {id} has invalid replay_seen"))?;
                let seen = previous_seen
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("queue row {id} exhausted replay counter"))?;
                replays_charged += 1;
                if seen > cap {
                    let reason =
                        format!("replayed by {seen} boots without ever running (limit {cap})");
                    conn.execute(
                        "UPDATE generation_queue
                            SET replay_seen = ?3, state = 'held', held_reason = ?4,
                                started_at = NULL, updated_at = ?5
                          WHERE id = ?1 AND owner_uuid = ?2
                            AND state IN ('queued', 'running')
                            AND (claim_token IS NOT NULL OR replay_seen > 0)",
                        params![id, owner_uuid, seen as i64, reason, now_ms],
                    )?;
                    conn.execute(
                        "UPDATE generation_batch_children
                            SET state = 'held', error = ?3, updated_at_ms = ?4
                          WHERE job_id = ?1
                            AND EXISTS (
                                SELECT 1 FROM generation_batches AS batch
                                 WHERE batch.id = generation_batch_children.batch_id
                                   AND batch.owner_uuid = ?2
                            )",
                        params![id, owner_uuid, reason, now_ms],
                    )?;
                    replays_held += 1;
                } else {
                    conn.execute(
                        "UPDATE generation_queue
                            SET replay_seen = ?3, updated_at = ?4
                          WHERE id = ?1 AND owner_uuid = ?2
                            AND state IN ('queued', 'running')
                            AND (claim_token IS NOT NULL OR replay_seen > 0)",
                        params![id, owner_uuid, seen as i64, now_ms],
                    )?;
                }
            }
        }
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
            replays_charged,
            replays_held,
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

/// Re-stamp queued rows so replay follows the hydrated `ids` prefix, followed
/// by every still-unhydrated queued row in its existing stable order.
///
/// `created_at` is the replay sort key, so a queue reorder has to move the
/// complete queued order or a same-millisecond tail can overtake the hydrated
/// prefix after restart. The immediate transaction reads only ids and ordering
/// keys, then rewrites a dense sequence anchored at the oldest queued row's
/// existing timestamp. Payloads are never materialized, and held, running,
/// foreign, missing, and duplicate prefix ids are ignored.
pub fn apply_queue_order(db: &MetadataDb, owner_uuid: &str, ids: &[String]) -> Result<usize> {
    if ids.is_empty() {
        return Ok(0);
    }
    db.transact_immediate(|conn| {
        let queued = {
            let mut stmt = conn.prepare(QUEUE_REORDER_CANDIDATES_SQL)?;
            let rows = stmt
                .query_map(params![owner_uuid], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let Some(oldest_created_at) = queued.first().map(|(_, created_at)| *created_at) else {
            return Ok(0);
        };

        let mut remaining = queued
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<HashSet<_>>();
        let mut ordered = Vec::with_capacity(queued.len());
        for id in ids {
            if remaining.remove(id) {
                ordered.push(id.clone());
            }
        }
        let hydrated = ordered.len();
        if hydrated == 0 {
            return Ok(0);
        }
        for (id, _) in queued {
            if remaining.remove(&id) {
                ordered.push(id);
            }
        }

        let span = i64::try_from(ordered.len().saturating_sub(1))
            .map_err(|_| anyhow::anyhow!("queued generation count exceeds SQLite's range"))?;
        let anchor = oldest_created_at.min(i64::MAX - span);
        let mut stmt = conn.prepare(
            "UPDATE generation_queue
                SET created_at = ?2, updated_at = ?3
              WHERE id = ?1 AND owner_uuid = ?4 AND state = 'queued'",
        )?;
        let mut moved = 0;
        for (index, id) in ordered.iter().enumerate() {
            let stamp = anchor + index as i64;
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
        let invalid: Option<i64> = conn
            .query_row(INVALID_JOB_METADATA_LOOKUP_SQL, [], |row| row.get(0))
            .optional()?;
        ensure!(
            invalid.is_none(),
            "gallery contains malformed queue publication metadata"
        );

        let requested = ids.iter().map(String::as_str).collect::<HashSet<_>>();
        let mut unknown = conn.prepare(UNKNOWN_JOB_METADATA_LOOKUP_SQL)?;
        let unknown_rows = unknown.query_map([], |row| row.get::<_, Option<String>>(0))?;
        for metadata_json in unknown_rows {
            let Some(metadata_json) = metadata_json? else {
                continue;
            };
            if let Some(job_id) = parse_queue_job_id(&metadata_json)? {
                if requested.contains(job_id.as_str()) {
                    found.insert(job_id);
                }
            }
        }

        let mut stmt = conn.prepare(COMPLETED_JOB_ID_LOOKUP_SQL)?;
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

/// Failure class for one indexed publication lookup.
///
/// `InvalidAuthority` is deterministic for the retained row/gallery and may
/// be held without blocking later queue work. `Infrastructure` means the
/// lookup itself could not be completed and must remain retryable.
#[derive(Debug)]
pub enum CompletedOutputLookupError {
    InvalidAuthority(String),
    Infrastructure(anyhow::Error),
}

impl CompletedOutputLookupError {
    pub fn is_invalid_authority(&self) -> bool {
        matches!(self, Self::InvalidAuthority(_))
    }
}

impl std::fmt::Display for CompletedOutputLookupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAuthority(message) => formatter.write_str(message),
            Self::Infrastructure(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompletedOutputLookupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidAuthority(_) => None,
            Self::Infrastructure(error) => error.source(),
        }
    }
}

impl From<rusqlite::Error> for CompletedOutputLookupError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Infrastructure(error.into())
    }
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

/// Resolve the exact reconnect payload from filenames that share one durable
/// queue job id. Callers may source those names from SQLite or another
/// validated gallery authority; the ordinary/upscale shape is identical.
pub fn resolve_completed_output_filenames(
    job_id: &str,
    filenames: impl IntoIterator<Item = String>,
) -> Result<Option<CompletedGenerationOutput>> {
    let mut primary = Vec::new();
    let mut originals = Vec::new();
    let mut upscaled = Vec::new();
    for filename in filenames {
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
        // Publication intentionally saves the original first. If the process
        // dies before the upscaled file is archived, that original is already
        // a valid gallery print and is the same degraded result Mold returns
        // when post-upscale fails. Adopt it as the terminal output instead of
        // rerendering a second, potentially different print.
        ([], [filename], []) => Ok(Some(CompletedGenerationOutput {
            filename: filename.clone(),
            original_filename: None,
        })),
        ([], [original], [filename]) => Ok(Some(CompletedGenerationOutput {
            filename: filename.clone(),
            original_filename: Some(original.clone()),
        })),
        _ => bail!("gallery outputs for queue job {job_id} are ambiguous"),
    }
}

/// Recover one completed child's exact saved output from its owned gallery.
///
/// The queue row supplies both the owner fence and output-directory scope.
/// Current rows are reached through the serde-populated
/// `(output_dir, queue_job_id)` index. Only compatibility rows written by an
/// older process after migration are parsed, through their own partial index.
/// This keeps normal replay independent of total gallery size without giving
/// SQLite's permissive JSON projection authority over malformed metadata.
pub fn find_completed_output(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
) -> std::result::Result<Option<CompletedGenerationOutput>, CompletedOutputLookupError> {
    let queue_authority: Option<(String, i64)> = db.with_conn_typed(|conn| {
        conn.query_row(
            "SELECT output_dir, created_at FROM generation_queue
              WHERE id = ?1 AND owner_uuid = ?2",
            params![job_id, owner_uuid],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(CompletedOutputLookupError::from)
    })?;
    let Some((output_dir, queue_created_at_ms)) = queue_authority else {
        return Ok(None);
    };
    // Canonicalization may touch the filesystem. Keep it outside the DB mutex
    // so a slow gallery mount cannot stall unrelated metadata readers.
    let output_dir = crate::canonical_dir_string(Path::new(&output_dir));
    let filenames = db.with_conn_typed(|conn| {
        let invalid: Option<i64> = conn
            .query_row(
                COMPLETED_OUTPUT_INVALID_METADATA_SQL,
                params![output_dir, queue_created_at_ms],
                |row| row.get(0),
            )
            .optional()?;
        if invalid.is_some() {
            return Err(CompletedOutputLookupError::InvalidAuthority(
                "gallery contains malformed queue publication metadata".to_string(),
            ));
        }
        let mut filenames = Vec::new();
        let mut unknown = conn.prepare(COMPLETED_OUTPUT_UNKNOWN_METADATA_SQL)?;
        let unknown_rows = unknown.query_map(params![output_dir, queue_created_at_ms], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?;
        for row in unknown_rows {
            let (filename, metadata_json) = row?;
            let Some(metadata_json) = metadata_json else {
                continue;
            };
            let parsed = parse_queue_job_id(&metadata_json).map_err(|error| {
                CompletedOutputLookupError::InvalidAuthority(format!("{error:#}"))
            })?;
            if parsed.as_deref() == Some(job_id) {
                filenames.push(filename);
            }
        }

        let mut stmt = conn.prepare(COMPLETED_OUTPUT_LOOKUP_SQL)?;
        let rows = stmt.query_map(params![output_dir, job_id], |row| row.get::<_, String>(0))?;
        for row in rows {
            filenames.push(row?);
        }
        Ok(filenames)
    })?;
    resolve_completed_output_filenames(job_id, filenames)
        .map_err(|error| CompletedOutputLookupError::InvalidAuthority(format!("{error:#}")))
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
        media_set_id: row.get(16)?,
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
            media_set_id: None,
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
    fn queue_reorder_scan_never_selects_durable_payload_columns() {
        let normalized = QUEUE_REORDER_CANDIDATES_SQL.to_ascii_lowercase();
        for forbidden in ["request_json", "output_dir", "completion_payload"] {
            assert!(
                !normalized.contains(forbidden),
                "queue reorder selected {forbidden}: {QUEUE_REORDER_CANDIDATES_SQL}"
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
    fn apply_queue_order_appends_the_deep_same_millisecond_backlog_and_claims_it_fifo_after_reopen()
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        let runtime_capacity = 3;
        let admitted = (0..runtime_capacity * 3)
            .map(|index| format!("job-{index}"))
            .collect::<Vec<_>>();
        for id in &admitted {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }

        // Only this prefix is hydrated into the bounded runtime registry. The
        // remaining rows stay payload-bearing in SQLite and must still follow
        // the reordered prefix after a restart.
        let hydrated_order = vec![
            admitted[2].clone(),
            admitted[0].clone(),
            admitted[1].clone(),
        ];
        assert_eq!(
            apply_queue_order(&db, "owner-a", &hydrated_order).unwrap(),
            admitted.len()
        );
        drop(db);

        let db = MetadataDb::open(&path).unwrap();
        let expected = hydrated_order
            .iter()
            .cloned()
            .chain(admitted[runtime_capacity..].iter().cloned())
            .collect::<Vec<_>>();
        let mut claimed = Vec::new();
        for index in 0..expected.len() {
            claimed.push(
                claim_next(
                    &db,
                    "owner-a",
                    &format!("claim-{index}"),
                    1_000 + index as i64,
                )
                .unwrap()
                .expect("every queued row remains claimable")
                .row
                .id,
            );
        }
        assert_eq!(claimed, expected);
    }

    #[test]
    fn apply_queue_order_only_moves_queued_rows_owned_by_the_runtime() {
        let db = MetadataDb::open_in_memory().unwrap();
        for (id, created) in [
            ("running", 100),
            ("first", 200),
            ("second", 300),
            ("tail", 400),
            ("held", 500),
        ] {
            insert(&db, &row(id, "owner-a", created)).unwrap();
        }
        insert(&db, &row("foreign", "owner-b", 50)).unwrap();
        mark_dispatched(&db, "running", 600).unwrap();
        hold(&db, "held", "operator review", 601).unwrap();

        let moved = apply_queue_order(
            &db,
            "owner-a",
            &[
                "held".to_string(),
                "second".to_string(),
                "running".to_string(),
                "first".to_string(),
                "foreign".to_string(),
            ],
        )
        .unwrap();
        assert_eq!(moved, 3, "the prefix plus omitted queued tail were moved");

        let replayable = list_replayable(&db, "owner-a").unwrap();
        assert_eq!(
            replayable
                .iter()
                .filter(|row| row.state == QueueRowState::Queued)
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["second", "first", "tail"]
        );
        let running = get(&db, "running").unwrap().unwrap();
        assert_eq!(running.state, QueueRowState::Running);
        assert_eq!(running.created_at_ms, 100);
        let held = get(&db, "held").unwrap().unwrap();
        assert_eq!(held.state, QueueRowState::Held);
        assert_eq!(held.created_at_ms, 500);
        assert_eq!(get(&db, "foreign").unwrap().unwrap().created_at_ms, 50);
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
    fn claim_by_id_is_exact_and_leaves_fifo_for_the_ordinary_feeder() {
        let db = MetadataDb::open_in_memory().unwrap();
        for (id, created_at_ms) in [("first", 1), ("target", 2), ("last", 3)] {
            insert(&db, &row(id, "owner-a", created_at_ms)).unwrap();
        }

        let exact = claim_by_id(&db, "owner-a", "target", "exact-token", 10)
            .unwrap()
            .unwrap();
        assert_eq!(exact.row.id, "target");
        assert_eq!(exact.claim_token, "exact-token");
        assert_eq!(exact.row.updated_at_ms, 10);

        assert_eq!(
            claim_next(&db, "owner-a", "fifo-first", 11)
                .unwrap()
                .unwrap()
                .row
                .id,
            "first"
        );
        assert_eq!(
            claim_next(&db, "owner-a", "fifo-last", 12)
                .unwrap()
                .unwrap()
                .row
                .id,
            "last"
        );
    }

    #[test]
    fn claim_by_id_refuses_ineligible_rows_without_mutation() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in [
            "held",
            "running",
            "foreign",
            "claimed",
            "cancelled",
            "eligible",
        ] {
            let owner = if id == "foreign" {
                "owner-b"
            } else {
                "owner-a"
            };
            insert(&db, &row(id, owner, 1)).unwrap();
        }
        hold(&db, "held", "review", 2).unwrap();
        mark_dispatched(&db, "running", 3).unwrap();
        claim_by_id(&db, "owner-a", "claimed", "existing-token", 4)
            .unwrap()
            .unwrap();
        delete(&db, "cancelled").unwrap();

        let snapshot = || {
            db.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, state, claim_token, updated_at
                       FROM generation_queue ORDER BY id",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .map_err(Into::into)
            })
            .unwrap()
        };
        let before = snapshot();
        for (owner, id, token) in [
            ("owner-a", "held", "held-token"),
            ("owner-a", "running", "running-token"),
            ("owner-a", "foreign", "foreign-token"),
            ("owner-a", "claimed", "new-token"),
            ("owner-a", "cancelled", "cancelled-token"),
            ("owner-a", "eligible", "existing-token"),
        ] {
            assert!(claim_by_id(&db, owner, id, token, 99).unwrap().is_none());
        }
        assert_eq!(snapshot(), before);
        assert!(claim_by_id(&db, "owner-a", "held", "", 99).is_err());
        assert_eq!(snapshot(), before);
    }

    #[test]
    fn concurrent_claim_by_id_has_one_token_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        insert(&db, &row("target", "owner-a", 1)).unwrap();
        drop(db);

        let barrier = Arc::new(Barrier::new(2));
        let handles = ["claim-a", "claim-b"].map(|token| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let db = MetadataDb::open(&path).unwrap();
                barrier.wait();
                claim_by_id(&db, "owner-a", "target", token, 10).unwrap()
            })
        });
        let claims = handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].row.id, "target");
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
    fn publication_lookup_query_plans_are_index_bounded() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.with_conn(|conn| {
            let plans = [
                (
                    format!("EXPLAIN QUERY PLAN {COMPLETED_JOB_ID_LOOKUP_SQL}"),
                    vec![rusqlite::types::Value::Text("job".to_string())],
                    "generations_queue_job_id",
                ),
                (
                    format!("EXPLAIN QUERY PLAN {COMPLETED_OUTPUT_LOOKUP_SQL}"),
                    vec![
                        rusqlite::types::Value::Text("/gallery".to_string()),
                        rusqlite::types::Value::Text("job".to_string()),
                    ],
                    "generations_output_queue_job_id",
                ),
                (
                    format!("EXPLAIN QUERY PLAN {COMPLETED_OUTPUT_INVALID_METADATA_SQL}"),
                    vec![
                        rusqlite::types::Value::Text("/gallery".to_string()),
                        rusqlite::types::Value::Integer(1),
                    ],
                    "generations_output_invalid_queue_metadata",
                ),
                (
                    format!("EXPLAIN QUERY PLAN {INVALID_JOB_METADATA_LOOKUP_SQL}"),
                    vec![],
                    "generations_invalid_queue_metadata",
                ),
                (
                    format!("EXPLAIN QUERY PLAN {COMPLETED_OUTPUT_UNKNOWN_METADATA_SQL}"),
                    vec![
                        rusqlite::types::Value::Text("/gallery".to_string()),
                        rusqlite::types::Value::Integer(1),
                    ],
                    "generations_output_unknown_queue_metadata",
                ),
            ];
            for (sql, values, expected_index) in plans {
                let mut stmt = conn.prepare(&sql)?;
                let details = stmt
                    .query_map(rusqlite::params_from_iter(values), |row| {
                        row.get::<_, String>(3)
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                assert!(
                    details.iter().any(|detail| detail.contains(expected_index)),
                    "publication lookup must use {expected_index}, got {details:?}"
                );
                assert!(
                    details.iter().all(|detail| {
                        !detail.contains("SCAN generations") || detail.contains(expected_index)
                    }),
                    "publication lookup may scan only its bounded partial index: {details:?}"
                );
            }
            Ok(())
        })
        .unwrap();
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
    fn completed_output_adopts_one_interrupted_upscale_original_as_terminal_output() {
        let db = MetadataDb::open_in_memory().unwrap();
        let gallery = tempfile::tempdir().unwrap();
        let mut queued = row("done", "owner-a", 1);
        queued.output_dir = gallery.path().to_path_buf();
        insert(&db, &queued).unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO generations
                    (filename, output_dir, created_at_ms, format, model, metadata_json)
                 VALUES ('mold-flux-1-original~portrait.png', ?1, 1, 'png', 'flux-dev:q4', ?2)",
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
                filename: "mold-flux-1-original~portrait.png".to_string(),
                original_filename: None,
            })
        );
    }

    #[test]
    fn completed_output_ignores_malformed_rows_that_predate_the_claimed_job() {
        for legacy_projection_state in [Some(0_i64), None] {
            let db = MetadataDb::open_in_memory().unwrap();
            let gallery = tempfile::tempdir().unwrap();
            let gallery_key = crate::canonical_dir_string(gallery.path());
            let mut queued = row("done", "owner-a", 100);
            queued.output_dir = gallery.path().to_path_buf();
            insert(&db, &queued).unwrap();

            db.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO generations
                        (filename, output_dir, created_at_ms, format, model, metadata_json,
                         queue_job_id, queue_job_metadata_state)
                     VALUES ('historical-broken.png', ?1, 99, 'png', 'legacy', ?2, NULL, ?3)",
                    params![
                        gallery_key,
                        r#"{"job_id":"unparseable""#,
                        legacy_projection_state
                    ],
                )?;
                conn.execute(
                    "INSERT INTO generations
                        (filename, output_dir, created_at_ms, format, model, metadata_json,
                         queue_job_id, queue_job_metadata_state)
                     VALUES ('done.png', ?1, 101, 'png', 'model', ?2, 'done', 1)",
                    params![gallery_key, r#"{"job_id":"done"}"#],
                )?;
                Ok(())
            })
            .unwrap();

            assert_eq!(
                find_completed_output(&db, "owner-a", "done").unwrap(),
                Some(CompletedGenerationOutput {
                    filename: "done.png".to_string(),
                    original_filename: None,
                }),
                "an unrelated malformed row from before admission cannot poison this job"
            );

            db.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO generations
                        (filename, output_dir, created_at_ms, format, model, metadata_json,
                         queue_job_id, queue_job_metadata_state)
                     VALUES ('possible-broken.png', ?1, 100, 'png', 'legacy', ?2, NULL, ?3)",
                    params![
                        gallery_key,
                        r#"{"job_id":"unparseable""#,
                        legacy_projection_state
                    ],
                )?;
                Ok(())
            })
            .unwrap();
            assert!(
                find_completed_output(&db, "owner-a", "done")
                    .unwrap_err()
                    .is_invalid_authority(),
                "a malformed row in the job's possible publication window stays fail-closed"
            );
        }
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

            assert!(find_completed_output(&db, "owner-a", "done")
                .unwrap_err()
                .is_invalid_authority());
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
