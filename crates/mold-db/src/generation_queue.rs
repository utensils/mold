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
//! kills the process during its own load is held. `replay_seen` is charged by
//! startup recovery ([`recover_runtime_claims_and_charge_replays`]) once per
//! boot that finds the row claimed, running, or already charged — never for
//! the untouched backlog behind them — which is the only bound on a row that
//! keeps entering the runtime window without ever running.
//!
//! Free functions over [`MetadataDb`], matching `chain_jobs.rs`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Result};
use rusqlite::{params, OptionalExtension, Row};

use crate::db::MetadataDb;
use crate::generation_queue_media::{self, QueueMediaObligation};

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
    /// Opaque authenticated ciphertext for reconstructing a server-owned
    /// admission grant after restart. The media-store key binds it to this
    /// owner + job; the server then revalidates request, instance, and policy.
    pub admission_authority: Option<String>,
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
    /// Whether an explicit retry may safely return this held row to the queue.
    pub retryable: bool,
    pub created_at_ms: i64,
    /// Durable batch this row is a child of, when it has one.
    ///
    /// `POST /api/queue/{id}/retry` requires the whole authority — instance,
    /// batch, client batch, job — and every part but the instance belongs to
    /// the row. Projecting it here is what lets a listing offer the retry
    /// instead of making each client guess the parent. `None` is an honest
    /// answer: a queue row admitted outside a batch has no parent.
    pub batch_id: Option<String>,
    /// The client-minted idempotency id of [`Self::batch_id`].
    pub client_batch_id: Option<String>,
    /// One-based position of this child within its batch, as
    /// `GenerationBatchChild::index` reports it.
    pub batch_index: Option<u32>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnedQueuedLoad {
    pub queued_count: usize,
    pub live_overlap: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueTargetPatch {
    pub target_gpu: Option<usize>,
    pub target_device_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedQueuedPatch {
    /// `None` leaves affinity untouched; `Some` replaces both the ordinal and
    /// stable device identity, including clearing both to Auto.
    pub target: Option<QueueTargetPatch>,
    /// New zero-based position among this owner's queued rows. Values beyond
    /// the tail are clamped to the tail.
    pub position: Option<usize>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnedQueuedPatchOutcome {
    Updated {
        position: usize,
        projection: GenerationQueueProjection,
    },
    NotOwned,
    NotQueued,
}

/// The only column projection allowed on the paginated queue-listing path.
/// Keep this as a literal so the regression test can prove that none of the
/// durable payload columns are selected.
const QUEUE_PROJECTION_FIRST_PAGE_SQL: &str = "
    SELECT q.id, q.state, q.model, q.target_gpu, q.seed_pinned,
           q.dispatch_attempts, q.replay_seen, q.held_reason, q.retryable, q.created_at,
           q.rowid, c.batch_id, c.batch_index, b.client_batch_id
      FROM generation_queue AS q
      LEFT JOIN generation_batch_children AS c ON c.job_id = q.id
      LEFT JOIN generation_batches AS b ON b.id = c.batch_id
     WHERE q.owner_uuid = ?1
     ORDER BY q.created_at, q.rowid
     LIMIT ?2";

const QUEUE_PROJECTION_AFTER_SQL: &str = "
    SELECT q.id, q.state, q.model, q.target_gpu, q.seed_pinned,
           q.dispatch_attempts, q.replay_seen, q.held_reason, q.retryable, q.created_at,
           q.rowid, c.batch_id, c.batch_index, b.client_batch_id
      FROM generation_queue AS q
      LEFT JOIN generation_batch_children AS c ON c.job_id = q.id
      LEFT JOIN generation_batches AS b ON b.id = c.batch_id
     WHERE q.owner_uuid = ?1
       AND (q.created_at, q.rowid) > (?2, ?3)
     ORDER BY q.created_at, q.rowid
     LIMIT ?4";

/// Payload-free stable scan used to complete a bounded hydrated reorder with
/// every queued row that still lives only in SQLite.
const QUEUE_REORDER_CANDIDATES_SQL: &str = "
    SELECT id, created_at
      FROM generation_queue
     WHERE owner_uuid = ?1 AND state = 'queued'
     ORDER BY created_at, rowid";

/// Payload-free prefix used when a claimed row crosses into the bounded live
/// registry. The v18 replay index covers `(owner_uuid, state, created_at)` and
/// SQLite supplies `rowid` as the stable final key, so the read stops at the
/// runtime window instead of walking or materializing the retained backlog.
const CLAIMED_QUEUE_RUNTIME_WINDOW_SQL: &str = "
    SELECT id
      FROM generation_queue
     WHERE owner_uuid = ?1 AND state = 'queued'
     ORDER BY created_at, rowid
     LIMIT ?2";

/// One runtime reservation of an oldest queued row.
///
/// The token is deliberately kept outside [`GenerationQueueRow`] so existing
/// callers and struct literals remain source-compatible. Every mutation after
/// hydration must present it together with the expected durable state.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueClaim {
    pub row: GenerationQueueRow,
    pub claim_token: String,
    /// Stable SQLite admission order, used when bounded preparation workers
    /// finish out of order before scheduler publication.
    pub queue_rank: u64,
}

/// Position of an exact feeder claim in SQLite's bounded live-order window.
///
/// `Some(position)` means the row belongs in that queued slot. `None` means
/// the claim is valid but deeper than the window and must be released for a
/// later FIFO claim; appending it would bypass durable predecessors. A missing
/// outer value means the owner/state/token fence is stale and the row must not
/// be published into the registry. Affinity is read in the same snapshot so a
/// claimed row never hydrates a pre-PATCH target from its old claim payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedQueueRuntimePosition {
    pub position: Option<usize>,
    /// Durable queued predecessors from the same bounded snapshot. A feeder
    /// may publish only after each is already represented in the live
    /// registry (or has left the queued set).
    pub predecessor_ids: Vec<String>,
    pub target_gpu: Option<usize>,
    pub target_device_id: Option<String>,
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
                started_at, claim_token, media_set_id, admission_authority
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
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
            row.admission_authority.as_deref(),
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

/// One expired held row, payload-free.
///
/// Deliberately not a [`GenerationQueueRow`]: the retention sweeper needs an
/// identity and a media reference, and must never pull `request_json` — the
/// prompts and staged-media handles of every abandoned hold — into a
/// periodic background task's memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpiredHeldRow {
    pub id: String,
    pub media_set_id: Option<String>,
    pub held_reason: Option<String>,
    pub updated_at_ms: i64,
}

/// List this owner's held rows whose retention has elapsed.
///
/// `retention_days == 0` keeps held rows forever and returns nothing, the
/// same contract `gallery.trash_retention_days` keeps. Age is measured from
/// `updated_at_ms` — the moment the row was held — not from admission, so
/// work that queued for a week and held yesterday gets its full window.
pub fn expired_held(
    db: &MetadataDb,
    owner_uuid: &str,
    retention_days: u32,
    now_ms: i64,
) -> Result<Vec<ExpiredHeldRow>> {
    if retention_days == 0 {
        return Ok(Vec::new());
    }
    let cutoff = now_ms.saturating_sub(i64::from(retention_days) * 86_400_000);
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, media_set_id, held_reason, updated_at
               FROM generation_queue
              WHERE owner_uuid = ?1 AND state = 'held' AND updated_at <= ?2
              ORDER BY updated_at",
        )?;
        let rows = stmt
            .query_map(params![owner_uuid, cutoff], |row| {
                Ok(ExpiredHeldRow {
                    id: row.get(0)?,
                    media_set_id: row.get(1)?,
                    held_reason: row.get(2)?,
                    updated_at_ms: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    })
}

/// Count this owner's held rows, for the sweep's "remaining" report.
pub fn held_count(db: &MetadataDb, owner_uuid: &str) -> Result<u64> {
    db.with_conn(|conn| {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM generation_queue WHERE owner_uuid = ?1 AND state = 'held'",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        Ok(count.max(0) as u64)
    })
}

/// Purge one expired held row and settle its batch child in the same
/// transaction.
///
/// The row must still be held and still owned: a retry or a cancel between
/// the listing and this call wins, and this returns `false` without touching
/// anything. Deleting the queue row fires the `generation_queue_media_retire`
/// trigger, which is the singular authority that moves the media obligation
/// to `gc_pending` — the sweeper never writes that table itself.
///
/// The child becomes `failed` rather than being deleted: a batch's terminal
/// summary is what a reconnecting client reads after the queue row is gone,
/// and silently dropping the child would report the print as never admitted.
pub fn purge_held(db: &MetadataDb, owner_uuid: &str, id: &str, now_ms: i64) -> Result<bool> {
    db.transact_immediate(|conn| {
        let held_reason: Option<Option<String>> = conn
            .query_row(
                "SELECT held_reason FROM generation_queue
                  WHERE id = ?1 AND owner_uuid = ?2 AND state = 'held'",
                params![id, owner_uuid],
                |row| row.get(0),
            )
            .optional()?;
        let Some(held_reason) = held_reason else {
            return Ok(false);
        };
        let reason = held_reason.unwrap_or_else(|| "held".to_string());
        let error = format!("held work expired before it was retried: {reason}");
        conn.execute(
            "UPDATE generation_batch_children
                SET state = 'failed', error = ?2,
                    terminal_error_json = ?3,
                    completed_at_ms = ?4, updated_at_ms = ?4,
                    revision = revision + 1
              WHERE job_id = ?1 AND state = 'held'",
            params![
                id,
                error,
                serde_json::json!({ "message": error }).to_string(),
                now_ms
            ],
        )?;
        let removed = conn.execute(
            "DELETE FROM generation_queue
              WHERE id = ?1 AND owner_uuid = ?2 AND state = 'held'",
            params![id, owner_uuid],
        )?;
        Ok(removed == 1)
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
                    started_at, media_set_id, admission_authority
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
                    started_at, media_set_id, admission_authority
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
/// a positive value, and this query returns at most that many rows. SQLite
/// reads one additional ordering key to prove whether another page exists;
/// the extra row is discarded before returning.
pub fn list_projection_page(
    db: &MetadataDb,
    owner_uuid: &str,
    cursor: Option<QueueProjectionCursor>,
    limit: usize,
) -> Result<GenerationQueueProjectionPage> {
    if limit == 0 {
        bail!("queue projection page limit must be positive");
    }
    let fetch_limit = limit
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("queue projection page limit is outside SQLite's range"))?;
    let sql_limit = i64::try_from(fetch_limit)
        .map_err(|_| anyhow::anyhow!("queue projection page limit is outside SQLite's range"))?;
    db.with_conn(|conn| {
        let mut rows_with_keys = if let Some(cursor) = cursor {
            let mut stmt = conn.prepare(QUEUE_PROJECTION_AFTER_SQL)?;
            let rows = stmt
                .query_map(
                    params![owner_uuid, cursor.created_at_ms, cursor.rowid, sql_limit],
                    projection_page_row,
                )?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        } else {
            let mut stmt = conn.prepare(QUEUE_PROJECTION_FIRST_PAGE_SQL)?;
            let rows = stmt
                .query_map(params![owner_uuid, sql_limit], projection_page_row)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let has_later = rows_with_keys.len() > limit;
        if has_later {
            rows_with_keys.pop();
        }
        let next_cursor = has_later
            .then(|| rows_with_keys.last().map(|(_, cursor)| *cursor))
            .flatten();
        let rows = rows_with_keys.into_iter().map(|(row, _)| row).collect();
        Ok(GenerationQueueProjectionPage { rows, next_cursor })
    })
}

fn projection_page_row(
    row: &Row<'_>,
) -> rusqlite::Result<(GenerationQueueProjection, QueueProjectionCursor)> {
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
            retryable: row.get::<_, i64>(8)? != 0,
            created_at_ms: row.get(9)?,
            batch_id: row.get(11)?,
            batch_index: row.get::<_, Option<i64>>(12)?.map(|index| index as u32),
            client_batch_id: row.get(13)?,
        },
        QueueProjectionCursor {
            created_at_ms: row.get(9)?,
            rowid: row.get(10)?,
        },
    ))
}

/// Exact owner-fenced durable waiting depth and its bounded overlap with live
/// registry waiting ids, read from one SQLite connection snapshot.
///
/// Claimed queued rows already belong to the live scheduler and are excluded
/// so callers can add registry-only waiting ids without double counting.
/// Running and held rows are likewise not waiting backlog. The overlap probes
/// only caller-supplied bounded ids; this never lists the durable backlog.
pub fn owned_queued_load(
    db: &MetadataDb,
    owner_uuid: &str,
    live_waiting_ids: &[String],
) -> Result<OwnedQueuedLoad> {
    db.transact(|conn| owned_queued_load_on_conn(conn, owner_uuid, live_waiting_ids, || {}))
}

fn owned_queued_load_on_conn(
    conn: &rusqlite::Connection,
    owner_uuid: &str,
    live_waiting_ids: &[String],
    after_count: impl FnOnce(),
) -> Result<OwnedQueuedLoad> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM generation_queue
          WHERE owner_uuid = ?1 AND state = 'queued' AND claim_token IS NULL",
        params![owner_uuid],
        |row| row.get(0),
    )?;
    let queued_count = usize::try_from(count)
        .map_err(|_| anyhow::anyhow!("owned queued generation count is outside usize"))?;
    after_count();
    let mut overlap = 0usize;
    let mut seen = HashSet::with_capacity(live_waiting_ids.len());
    let mut stmt = conn.prepare(
        "SELECT 1 FROM generation_queue
          WHERE id = ?1 AND owner_uuid = ?2
            AND state = 'queued' AND claim_token IS NULL
          LIMIT 1",
    )?;
    for id in live_waiting_ids {
        if seen.insert(id.as_str()) && stmt.exists(params![id, owner_uuid])? {
            overlap = overlap
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("owned queued overlap exceeds usize"))?;
        }
    }
    Ok(OwnedQueuedLoad {
        queued_count,
        live_overlap: overlap,
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
                    started_at, media_set_id, admission_authority, rowid",
            params![owner_uuid, claim_token, now_ms],
            |row| {
                Ok(QueueClaim {
                    row: row_to_queue_row(row)?,
                    claim_token: claim_token.to_string(),
                    queue_rank: row.get::<_, i64>(18)? as u64,
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
                    started_at, media_set_id, admission_authority, rowid",
            params![owner_uuid, id, claim_token, now_ms],
            |row| {
                Ok(QueueClaim {
                    row: row_to_queue_row(row)?,
                    claim_token: claim_token.to_string(),
                    queue_rank: row.get::<_, i64>(18)? as u64,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Resolve one exact feeder claim against the authoritative durable order,
/// reading at most `limit` payload-free ids.
///
/// The claim validation and prefix read share one SQLite snapshot. Callers
/// hold the scheduler mutation fence while invoking this primitive, which
/// serializes it with queue PATCH/cancellation in the owning server; the
/// owner and exact token predicates keep stale, foreign, running, and held
/// rows from being published even if a retained database is inspected by a
/// non-owner process.
pub fn claimed_runtime_position(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    claim_token: &str,
    limit: usize,
) -> Result<Option<ClaimedQueueRuntimePosition>> {
    if claim_token.is_empty() {
        bail!("queue claim token must not be empty");
    }
    if limit == 0 {
        bail!("queue runtime order window must be positive");
    }
    let sql_limit = i64::try_from(limit)
        .map_err(|_| anyhow::anyhow!("queue runtime order window is outside SQLite's range"))?;
    db.transact(|conn| {
        let current = conn
            .query_row(
                "SELECT target_gpu, target_device_id
                   FROM generation_queue
                  WHERE id = ?1 AND owner_uuid = ?2 AND state = 'queued'
                    AND claim_token = ?3",
                params![job_id, owner_uuid, claim_token],
                |row| {
                    Ok((
                        row.get::<_, Option<i64>>(0)?.map(|gpu| gpu as usize),
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?;
        let Some((target_gpu, target_device_id)) = current else {
            return Ok(None);
        };

        let ids = conn
            .prepare(CLAIMED_QUEUE_RUNTIME_WINDOW_SQL)?
            .query_map(params![owner_uuid, sql_limit], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let position = ids.iter().position(|id| id == job_id);
        let predecessor_ids = position
            .map(|position| ids[..position].to_vec())
            .unwrap_or_default();
        Ok(Some(ClaimedQueueRuntimePosition {
            position,
            predecessor_ids,
            target_gpu,
            target_device_id,
        }))
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
                SET state = 'held', held_reason = ?4, retryable = 0, updated_at = ?5
              WHERE id = ?1 AND state = ?2 AND claim_token = ?3",
            params![id, expected_state.as_str(), claim_token, reason, now_ms],
        )? > 0)
    })
}

/// Park a row: listed by `GET /api/queue`, never auto-run.
pub fn hold(db: &MetadataDb, id: &str, reason: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|conn| {
        let updated = conn.execute(
            "UPDATE generation_queue
                SET state = 'held', held_reason = ?2, retryable = 0, updated_at = ?3
              WHERE id = ?1",
            params![id, reason, now_ms],
        )?;
        Ok(updated > 0)
    })
}

/// Atomically quarantine selected staged-media jobs and their batch summaries
/// during startup.
///
/// The owner and non-NULL media marker fences prevent a corrupt/missing set
/// report from parking another installation's row or an ordinary media-free
/// job. The returned count is the number of unique requested jobs proven held
/// at commit, including rows a prior startup already held. Held rows retain
/// their marker and active cleanup obligation because the user may inspect or
/// explicitly cancel them later. A batch child is moved to `held` in the same
/// transaction; cancellation, terminal, and unknown child states fail closed.
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
        let mut select = conn.prepare(
            "SELECT q.state,
                    (SELECT child.state
                       FROM generation_batch_children AS child
                      WHERE child.job_id = q.id)
               FROM generation_queue AS q
              WHERE q.id = ?1 AND q.owner_uuid = ?2
                AND q.media_set_id IS NOT NULL",
        )?;
        let mut stmt = conn.prepare(
            "UPDATE generation_queue
                SET state = 'held', held_reason = ?3, retryable = 0, claim_token = NULL,
                    started_at = NULL, updated_at = ?4
              WHERE id = ?1 AND owner_uuid = ?2
                AND media_set_id IS NOT NULL
                AND state = ?5",
        )?;
        let mut held = 0;
        let mut update_child = conn.prepare(
            "UPDATE generation_batch_children
                SET state = 'held', error = ?2, updated_at_ms = ?3,
                    revision = revision + 1
              WHERE job_id = ?1 AND state = ?4",
        )?;
        for job_id in job_ids {
            if !seen.insert(job_id.as_str()) {
                continue;
            }
            let row = select
                .query_row(params![job_id, owner_uuid], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })
                .optional()?;
            let Some((queue_state, child_state)) = row else {
                continue;
            };
            if !matches!(queue_state.as_str(), "queued" | "running" | "held") {
                bail!("media queue row {job_id} has invalid state {queue_state}");
            }
            if let Some(child_state) = child_state.as_deref() {
                if !matches!(child_state, "accepted" | "running" | "held") {
                    bail!(
                        "media queue row {job_id} cannot be held beside batch child state {child_state}"
                    );
                }
            }
            if stmt.execute(params![job_id, owner_uuid, reason, now_ms, queue_state])? != 1 {
                bail!("media queue row {job_id} changed during startup hold");
            }
            if let Some(child_state) = child_state {
                if update_child.execute(params![job_id, reason, now_ms, child_state])? != 1 {
                    bail!("batch child {job_id} changed during startup media hold");
                }
            }
            held += 1;
        }
        Ok(held)
    })
}

/// Flip every `running` row this installation owns back to `queued`.
///
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
                            SET state = 'held', error = ?3, updated_at_ms = ?4,
                                revision = revision + 1
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

/// Atomically patch one owner-fenced queued row, including rows deeper than
/// the live registry window.
///
/// The state check, affinity replacement, and optional queued-order rewrite
/// share one IMMEDIATE transaction. This deep-row path owns only unclaimed
/// durable work; claimed rows belong to the live registry handoff even while
/// their durable state is still `queued`. Claimed, running, and held targets
/// are refused without a partial write.
pub fn patch_owned_queued(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    patch: &OwnedQueuedPatch,
) -> Result<OwnedQueuedPatchOutcome> {
    patch_owned_queued_with_claim_fence(
        db,
        owner_uuid,
        job_id,
        patch,
        QueuePatchClaimFence::Unclaimed,
    )
}

/// Atomically patch one owner-fenced queued row already claimed by the live
/// registry handoff.
///
/// This is the claimed counterpart to [`patch_owned_queued`]. It deliberately
/// accepts only a non-NULL durable claim so an unclaimed deep-tail row cannot
/// cross into the live mutation path. The owner, state, and claim-presence
/// checks remain part of every target mutation in the transaction.
pub fn patch_owned_claimed_queued(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    patch: &OwnedQueuedPatch,
) -> Result<OwnedQueuedPatchOutcome> {
    patch_owned_queued_with_claim_fence(
        db,
        owner_uuid,
        job_id,
        patch,
        QueuePatchClaimFence::Claimed,
    )
}

/// Atomically patch an owner-fenced queued row while preserving whichever
/// exact claim token the transaction observes.
///
/// The server's durable-transition protocol excludes claim publication and
/// same-id cancellation around this call. Accepting either claim state here
/// removes the registry-existence TOCTOU without weakening the transaction:
/// every target update and final projection is still fenced by the exact
/// token (including NULL) read at the start of the IMMEDIATE transaction.
pub fn patch_owned_any_queued(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    patch: &OwnedQueuedPatch,
) -> Result<OwnedQueuedPatchOutcome> {
    patch_owned_queued_with_claim_fence(
        db,
        owner_uuid,
        job_id,
        patch,
        QueuePatchClaimFence::AnyExact,
    )
}

#[derive(Debug, Clone, Copy)]
enum QueuePatchClaimFence {
    Unclaimed,
    Claimed,
    AnyExact,
}

impl QueuePatchClaimFence {
    fn accepts(self, claim_token: Option<&str>) -> bool {
        match self {
            Self::Unclaimed => claim_token.is_none(),
            Self::Claimed => claim_token.is_some(),
            Self::AnyExact => true,
        }
    }

    fn affinity_sql(self) -> &'static str {
        match self {
            Self::Unclaimed => {
                "UPDATE generation_queue
                    SET target_gpu = ?3, target_device_id = ?4, updated_at = ?5
                  WHERE id = ?1 AND owner_uuid = ?2
                    AND state = 'queued' AND claim_token IS NULL"
            }
            Self::Claimed => {
                "UPDATE generation_queue
                    SET target_gpu = ?3, target_device_id = ?4, updated_at = ?5
                  WHERE id = ?1 AND owner_uuid = ?2
                    AND state = 'queued' AND claim_token IS NOT NULL"
            }
            Self::AnyExact => {
                "UPDATE generation_queue
                    SET target_gpu = ?3, target_device_id = ?4, updated_at = ?5
                  WHERE id = ?1 AND owner_uuid = ?2 AND state = 'queued'
                    AND claim_token IS ?6"
            }
        }
    }

    fn reorder_sql(self) -> &'static str {
        match self {
            Self::Unclaimed => {
                "UPDATE generation_queue
                    SET created_at = ?2, updated_at = ?3
                  WHERE id = ?1 AND owner_uuid = ?4 AND state = 'queued'
                    AND (id != ?5 OR claim_token IS NULL)"
            }
            Self::Claimed => {
                "UPDATE generation_queue
                    SET created_at = ?2, updated_at = ?3
                  WHERE id = ?1 AND owner_uuid = ?4 AND state = 'queued'
                    AND (id != ?5 OR claim_token IS NOT NULL)"
            }
            Self::AnyExact => {
                "UPDATE generation_queue
                    SET created_at = ?2, updated_at = ?3
                  WHERE id = ?1 AND owner_uuid = ?4 AND state = 'queued'
                    AND (id != ?5 OR claim_token IS ?6)"
            }
        }
    }

    fn projection_sql(self) -> &'static str {
        match self {
            Self::Unclaimed => {
                "SELECT q.id, q.state, q.model, q.target_gpu, q.seed_pinned,
                        q.dispatch_attempts, q.replay_seen, q.held_reason, q.retryable,
                        q.created_at, c.batch_id, c.batch_index, b.client_batch_id
                   FROM generation_queue AS q
                   LEFT JOIN generation_batch_children AS c ON c.job_id = q.id
                   LEFT JOIN generation_batches AS b ON b.id = c.batch_id
                  WHERE q.id = ?1 AND q.owner_uuid = ?2
                    AND q.state = 'queued' AND claim_token IS NULL"
            }
            Self::Claimed => {
                "SELECT q.id, q.state, q.model, q.target_gpu, q.seed_pinned,
                        q.dispatch_attempts, q.replay_seen, q.held_reason, q.retryable,
                        q.created_at, c.batch_id, c.batch_index, b.client_batch_id
                   FROM generation_queue AS q
                   LEFT JOIN generation_batch_children AS c ON c.job_id = q.id
                   LEFT JOIN generation_batches AS b ON b.id = c.batch_id
                  WHERE q.id = ?1 AND q.owner_uuid = ?2
                    AND q.state = 'queued' AND claim_token IS NOT NULL"
            }
            Self::AnyExact => {
                "SELECT q.id, q.state, q.model, q.target_gpu, q.seed_pinned,
                        q.dispatch_attempts, q.replay_seen, q.held_reason, q.retryable,
                        q.created_at, c.batch_id, c.batch_index, b.client_batch_id
                   FROM generation_queue AS q
                   LEFT JOIN generation_batch_children AS c ON c.job_id = q.id
                   LEFT JOIN generation_batches AS b ON b.id = c.batch_id
                  WHERE q.id = ?1 AND q.owner_uuid = ?2
                    AND q.state = 'queued' AND claim_token IS ?3"
            }
        }
    }
}

fn patch_owned_queued_with_claim_fence(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    patch: &OwnedQueuedPatch,
    claim_fence: QueuePatchClaimFence,
) -> Result<OwnedQueuedPatchOutcome> {
    db.transact_immediate(|conn| {
        let owned = conn
            .query_row(
                "SELECT state, created_at, rowid, claim_token
                   FROM generation_queue
                  WHERE id = ?1 AND owner_uuid = ?2",
                params![job_id, owner_uuid],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, created_at, rowid, claim_token)) = owned else {
            return Ok(OwnedQueuedPatchOutcome::NotOwned);
        };
        if state != QueueRowState::Queued.as_str() || !claim_fence.accepts(claim_token.as_deref()) {
            return Ok(OwnedQueuedPatchOutcome::NotQueued);
        }

        if let Some(target) = &patch.target {
            let updated = if matches!(claim_fence, QueuePatchClaimFence::AnyExact) {
                conn.execute(
                    claim_fence.affinity_sql(),
                    params![
                        job_id,
                        owner_uuid,
                        target.target_gpu.map(|gpu| gpu as i64),
                        target.target_device_id.as_deref(),
                        patch.updated_at_ms,
                        claim_token.as_deref(),
                    ],
                )?
            } else {
                conn.execute(
                    claim_fence.affinity_sql(),
                    params![
                        job_id,
                        owner_uuid,
                        target.target_gpu.map(|gpu| gpu as i64),
                        target.target_device_id.as_deref(),
                        patch.updated_at_ms,
                    ],
                )?
            };
            if updated != 1 {
                bail!("owned queued row changed during affinity patch");
            }
        }

        let position = if let Some(requested_position) = patch.position {
            let queued = {
                let mut stmt = conn.prepare(QUEUE_REORDER_CANDIDATES_SQL)?;
                let rows = stmt.query_map(params![owner_uuid], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            let current_position = queued
                .iter()
                .position(|(id, _)| id == job_id)
                .ok_or_else(|| anyhow::anyhow!("owned queued row disappeared during reorder"))?;
            let oldest_created_at = queued[0].1;
            let mut order = queued.into_iter().map(|(id, _)| id).collect::<Vec<_>>();
            order.remove(current_position);
            let position = requested_position.min(order.len());
            order.insert(position, job_id.to_string());

            if position != current_position {
                let span = i64::try_from(order.len().saturating_sub(1)).map_err(|_| {
                    anyhow::anyhow!("queued generation count exceeds SQLite's range")
                })?;
                let anchor = oldest_created_at.min(i64::MAX - span);
                let mut update = conn.prepare(claim_fence.reorder_sql())?;
                let mut moved = 0usize;
                for (index, id) in order.iter().enumerate() {
                    moved += if matches!(claim_fence, QueuePatchClaimFence::AnyExact) {
                        update.execute(params![
                            id,
                            anchor + index as i64,
                            patch.updated_at_ms,
                            owner_uuid,
                            job_id,
                            claim_token.as_deref(),
                        ])?
                    } else {
                        update.execute(params![
                            id,
                            anchor + index as i64,
                            patch.updated_at_ms,
                            owner_uuid,
                            job_id,
                        ])?
                    };
                }
                if moved != order.len() {
                    bail!("owned queued set changed during reorder");
                }
            }
            position
        } else {
            let before: i64 = conn.query_row(
                "SELECT COUNT(*) FROM generation_queue
                  WHERE owner_uuid = ?1 AND state = 'queued'
                    AND (created_at < ?2 OR (created_at = ?2 AND rowid < ?3))",
                params![owner_uuid, created_at, rowid],
                |row| row.get(0),
            )?;
            usize::try_from(before)
                .map_err(|_| anyhow::anyhow!("queued generation position is outside usize"))?
        };

        let read_projection = |row: &rusqlite::Row<'_>| {
            Ok(GenerationQueueProjection {
                id: row.get(0)?,
                state: QueueRowState::Queued,
                model: row.get(2)?,
                target_gpu: row.get::<_, Option<i64>>(3)?.map(|gpu| gpu as usize),
                seed_pinned: row.get::<_, i64>(4)? != 0,
                dispatch_attempts: row.get::<_, i64>(5)? as u32,
                replay_seen: row.get::<_, i64>(6)? as u32,
                held_reason: row.get(7)?,
                retryable: row.get::<_, i64>(8)? != 0,
                created_at_ms: row.get(9)?,
                batch_id: row.get(10)?,
                batch_index: row.get::<_, Option<i64>>(11)?.map(|index| index as u32),
                client_batch_id: row.get(12)?,
            })
        };
        let projection = if matches!(claim_fence, QueuePatchClaimFence::AnyExact) {
            conn.query_row(
                claim_fence.projection_sql(),
                params![job_id, owner_uuid, claim_token.as_deref()],
                read_projection,
            )?
        } else {
            conn.query_row(
                claim_fence.projection_sql(),
                params![job_id, owner_uuid],
                read_projection,
            )?
        };
        Ok(OwnedQueuedPatchOutcome::Updated {
            position,
            projection,
        })
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
        admission_authority: row.get(17)?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc, Barrier};

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
            admission_authority: None,
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

    /// This owner's rows the feeder may still claim, in claim order: every
    /// row that is not held, oldest first with `rowid` breaking ties.
    fn claimable(db: &MetadataDb, owner: &str) -> Vec<GenerationQueueRow> {
        list_all(db, owner)
            .unwrap()
            .into_iter()
            .filter(|row| row.state != QueueRowState::Held)
            .collect()
    }

    fn claimable_ids(db: &MetadataDb, owner: &str) -> Vec<String> {
        claimable(db, owner).into_iter().map(|row| row.id).collect()
    }

    #[test]
    fn list_all_orders_same_millisecond_inserts_by_insertion() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "second", "third"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }

        assert_eq!(claimable_ids(&db, "owner-a"), ["first", "second", "third"]);
    }

    #[test]
    fn list_all_is_scoped_to_the_owner_and_keeps_held_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("mine", "owner-a", 1)).unwrap();
        insert(&db, &row("theirs", "owner-b", 2)).unwrap();
        insert(&db, &row("parked", "owner-a", 3)).unwrap();
        hold(&db, "parked", "dispatch attempts exhausted", 9).unwrap();

        assert_eq!(claimable_ids(&db, "owner-a"), ["mine"]);
        assert_eq!(
            list_all(&db, "owner-a")
                .unwrap()
                .into_iter()
                .map(|row| row.id)
                .collect::<Vec<_>>(),
            ["mine", "parked"]
        );
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
            retryable: _,
            created_at_ms: _,
            batch_id: _,
            batch_index: _,
            client_batch_id: _,
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
    fn the_projection_carries_a_row_s_batch_identity_so_a_client_can_retry_it() {
        // `POST /api/queue/{id}/retry` demands the complete authority
        // (instance, batch, client batch, job). Everything but the instance is
        // a property of the row, so a listing that omits it forces every
        // client to guess which batch a held job belongs to.
        let db = MetadataDb::open_in_memory().unwrap();
        let mut child_row = row("child", "owner-a", 10);
        child_row.state = QueueRowState::Held;
        let solo = row("solo", "owner-a", 20);
        crate::generation_batches::insert_or_get(
            &db,
            &crate::generation_batches::GenerationBatchRow {
                id: "batch-1".into(),
                client_batch_id: "client-1".into(),
                owner_uuid: "owner-a".into(),
                request_sha256: "sha".into(),
                created_at_ms: 10,
            },
            &[(
                crate::generation_batches::GenerationBatchChildRow {
                    batch_id: "batch-1".into(),
                    job_id: "child".into(),
                    batch_index: 2,
                    state: "held".into(),
                    error: None,
                    updated_at_ms: 10,
                },
                child_row,
            )],
        )
        .unwrap();
        insert(&db, &solo).unwrap();

        let page = list_projection_page(&db, "owner-a", None, 10).unwrap();
        let batched = page.rows.iter().find(|row| row.id == "child").unwrap();
        assert_eq!(batched.batch_id.as_deref(), Some("batch-1"));
        assert_eq!(batched.client_batch_id.as_deref(), Some("client-1"));
        assert_eq!(
            batched.batch_index,
            Some(2),
            "one-based, as the wire reports it"
        );
        // A row that belongs to no batch says so rather than inventing one.
        let unbatched = page.rows.iter().find(|row| row.id == "solo").unwrap();
        assert_eq!(unbatched.batch_id, None);
        assert_eq!(unbatched.client_batch_id, None);
        assert_eq!(unbatched.batch_index, None);
    }

    #[test]
    fn projection_sql_never_selects_durable_payload_columns() {
        for sql in [QUEUE_PROJECTION_FIRST_PAGE_SQL, QUEUE_PROJECTION_AFTER_SQL] {
            let normalized = sql.to_ascii_lowercase();
            for forbidden in ["request_json", "output_dir", "completion_payload"] {
                assert!(
                    !normalized.contains(forbidden),
                    "projection SQL selected {forbidden}: {sql}"
                );
            }
        }
    }

    #[test]
    fn projection_page_plan_uses_owner_order_index_without_sort_or_correlated_scan() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.with_conn(|conn| {
            let plans = [
                (
                    QUEUE_PROJECTION_FIRST_PAGE_SQL,
                    vec![
                        rusqlite::types::Value::Text("owner-a".into()),
                        rusqlite::types::Value::Integer(11),
                    ],
                    false,
                ),
                (
                    QUEUE_PROJECTION_AFTER_SQL,
                    vec![
                        rusqlite::types::Value::Text("owner-a".into()),
                        rusqlite::types::Value::Integer(500),
                        rusqlite::types::Value::Integer(7),
                        rusqlite::types::Value::Integer(11),
                    ],
                    true,
                ),
            ];
            for (query, values, must_seek) in plans {
                let sql = format!("EXPLAIN QUERY PLAN {query}");
                let mut stmt = conn.prepare(&sql)?;
                let details = stmt
                    .query_map(rusqlite::params_from_iter(values), |row| {
                        row.get::<_, String>(3)
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let plan = details.join("\n").to_ascii_lowercase();
                assert!(
                    plan.contains("generation_queue_owner_order"),
                    "owner pagination did not use its order index: {plan}"
                );
                if must_seek {
                    assert!(
                        plan.contains("created_at>?") || plan.contains("created_at>"),
                        "cursor page scanned the owner's prior backlog instead of seeking: {plan}"
                    );
                }
                assert!(
                    !plan.contains("temp b-tree"),
                    "pagination sorted under the DB mutex: {plan}"
                );
                assert!(
                    !plan.contains("correlated"),
                    "pagination retained a correlated scan: {plan}"
                );
            }
            Ok(())
        })
        .unwrap();
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
    fn claimed_runtime_window_is_payload_free_and_uses_the_replay_index() {
        let normalized = CLAIMED_QUEUE_RUNTIME_WINDOW_SQL.to_ascii_lowercase();
        for forbidden in ["request_json", "output_dir", "completion_payload"] {
            assert!(
                !normalized.contains(forbidden),
                "runtime order window selected {forbidden}: {CLAIMED_QUEUE_RUNTIME_WINDOW_SQL}"
            );
        }

        let db = MetadataDb::open_in_memory().unwrap();
        let plan = db
            .with_conn(|conn| {
                let mut stmt = conn.prepare(&format!(
                    "EXPLAIN QUERY PLAN {CLAIMED_QUEUE_RUNTIME_WINDOW_SQL}"
                ))?;
                let rows = stmt
                    .query_map(params!["owner-a", 3_i64], |row| row.get::<_, String>(3))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows.join("\n"))
            })
            .unwrap();
        assert!(
            plan.contains("generation_queue_replay"),
            "runtime window did not use the owner/state/order index: {plan}"
        );
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
    fn requeue_running_returns_interrupted_rows_to_the_queue() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("job-1", "owner-a", 1)).unwrap();
        insert(&db, &row("job-2", "owner-b", 1)).unwrap();
        mark_dispatched(&db, "job-1", 10).unwrap();
        mark_dispatched(&db, "job-2", 10).unwrap();

        assert_eq!(
            recover_runtime_claims(&db, "owner-a", 20)
                .unwrap()
                .running_requeued,
            1
        );
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

        let ids = claimable_ids(&db, "owner-a");
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

        let replayable = claimable(&db, "owner-a");
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
    fn claimed_runtime_position_is_bounded_ordered_and_exactly_fenced() {
        let db = MetadataDb::open_in_memory().unwrap();
        for (id, created) in [
            ("first", 100),
            ("second", 200),
            ("deep", 300),
            ("outside", 400),
            ("running", 500),
            ("held", 600),
        ] {
            insert(&db, &row(id, "owner-a", created)).unwrap();
        }
        insert(&db, &row("foreign", "owner-b", 50)).unwrap();

        let running = claim_by_id(&db, "owner-a", "running", "running-token", 700)
            .unwrap()
            .unwrap();
        mark_dispatched_claimed(&db, "running", &running.claim_token, 701).unwrap();
        let held = claim_by_id(&db, "owner-a", "held", "held-token", 702)
            .unwrap()
            .unwrap();
        hold_claimed(
            &db,
            "held",
            &held.claim_token,
            QueueRowState::Queued,
            "operator review",
            703,
        )
        .unwrap();

        set_target_gpu(&db, "deep", Some(3), Some("cuda:stable"), 704).unwrap();
        let deep = claim_by_id(&db, "owner-a", "deep", "deep-token", 705)
            .unwrap()
            .unwrap();
        assert_eq!(
            claimed_runtime_position(&db, "owner-a", "deep", &deep.claim_token, 3).unwrap(),
            Some(ClaimedQueueRuntimePosition {
                position: Some(2),
                predecessor_ids: vec!["first".to_string(), "second".to_string()],
                target_gpu: Some(3),
                target_device_id: Some("cuda:stable".to_string()),
            })
        );
        assert_eq!(
            claimed_runtime_position(&db, "owner-a", "deep", &deep.claim_token, 2).unwrap(),
            Some(ClaimedQueueRuntimePosition {
                position: None,
                predecessor_ids: vec![],
                target_gpu: Some(3),
                target_device_id: Some("cuda:stable".to_string()),
            }),
            "a valid claim beyond the bounded prefix must be released for a later pass"
        );

        assert!(
            claimed_runtime_position(&db, "owner-a", "deep", "stale-token", 3)
                .unwrap()
                .is_none()
        );
        assert!(
            claimed_runtime_position(&db, "owner-b", "deep", &deep.claim_token, 3)
                .unwrap()
                .is_none()
        );
        assert!(
            claimed_runtime_position(&db, "owner-a", "running", &running.claim_token, 3)
                .unwrap()
                .is_none()
        );
        assert!(
            claimed_runtime_position(&db, "owner-a", "held", &held.claim_token, 3)
                .unwrap()
                .is_none()
        );
        assert!(claimed_runtime_position(&db, "owner-a", "deep", &deep.claim_token, 0).is_err());
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

    /// Another installation sharing this `mold.db` owns its own rows. The
    /// feeder's claim is the only thing that ever hands a durable row to a
    /// runtime, so owner scoping has to hold at this primitive, not in some
    /// listing above it.
    #[test]
    fn claim_next_is_scoped_to_the_owner() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("theirs", "owner-b", 1)).unwrap();
        insert(&db, &row("mine", "owner-a", 2)).unwrap();

        let claim = claim_next(&db, "owner-a", "token-a", 10)
            .unwrap()
            .expect("our own row is claimable");
        assert_eq!(claim.row.id, "mine");
        assert!(
            claim_next(&db, "owner-a", "token-a-2", 11)
                .unwrap()
                .is_none(),
            "the older foreign row is never handed to this owner's runtime"
        );

        let theirs = get(&db, "theirs").unwrap().unwrap();
        assert_eq!(theirs.state, QueueRowState::Queued);
        let claim_token: Option<String> = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT claim_token FROM generation_queue WHERE id = 'theirs'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(
            claim_token, None,
            "the foreign row keeps its own claim slot"
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
            claimable(&db, "owner-a")
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
    fn owned_queued_count_excludes_claimed_running_held_and_foreign_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        for (id, owner, created) in [
            ("claimed", "owner-a", 1),
            ("running", "owner-a", 2),
            ("waiting", "owner-a", 3),
            ("held", "owner-a", 4),
            ("foreign", "owner-b", 5),
        ] {
            insert(&db, &row(id, owner, created)).unwrap();
        }
        claim_by_id(&db, "owner-a", "claimed", "claimed-token", 10)
            .unwrap()
            .unwrap();
        claim_by_id(&db, "owner-a", "running", "running-token", 11)
            .unwrap()
            .unwrap();
        mark_dispatched_claimed(&db, "running", "running-token", 12).unwrap();
        hold(&db, "held", "review", 13).unwrap();

        assert_eq!(
            owned_queued_load(
                &db,
                "owner-a",
                &[
                    "waiting".into(),
                    "waiting".into(),
                    "claimed".into(),
                    "running".into(),
                    "held".into(),
                    "foreign".into(),
                ],
            )
            .unwrap(),
            OwnedQueuedLoad {
                queued_count: 1,
                live_overlap: 1,
            },
            "bounded overlap deduplicates ids and uses the exact waiting predicate"
        );
        assert_eq!(
            owned_queued_load(&db, "owner-b", &[]).unwrap(),
            OwnedQueuedLoad {
                queued_count: 1,
                live_overlap: 0,
            }
        );
    }

    #[test]
    fn owned_queued_load_count_and_overlap_share_one_sqlite_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        insert(&db, &row("waiting", "owner-a", 1)).unwrap();
        let (start_tx, start_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let writer_path = path.clone();
        let writer = std::thread::spawn(move || {
            start_rx.recv().unwrap();
            let writer_db = MetadataDb::open(&writer_path).unwrap();
            claim_by_id(&writer_db, "owner-a", "waiting", "live-claim", 2)
                .unwrap()
                .unwrap();
            done_tx.send(()).unwrap();
        });

        let snapshot = db
            .transact(|conn| {
                owned_queued_load_on_conn(conn, "owner-a", &["waiting".into()], || {
                    start_tx.send(()).unwrap();
                    done_rx.recv().unwrap();
                })
            })
            .unwrap();
        writer.join().unwrap();
        assert_eq!(
            snapshot,
            OwnedQueuedLoad {
                queued_count: 1,
                live_overlap: 1,
            },
            "the writer committed between the two reads, but one read transaction retained one snapshot"
        );
        assert_eq!(
            owned_queued_load(&db, "owner-a", &["waiting".into()]).unwrap(),
            OwnedQueuedLoad {
                queued_count: 0,
                live_overlap: 0,
            },
            "a later snapshot observes the committed claim"
        );
    }

    #[test]
    fn owned_queued_patch_atomically_relanes_and_reorders_a_deep_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        for id in ["first", "second", "deep", "tail"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }

        let outcome = patch_owned_queued(
            &db,
            "owner-a",
            "deep",
            &OwnedQueuedPatch {
                target: Some(QueueTargetPatch {
                    target_gpu: Some(3),
                    target_device_id: Some("cuda:stable".into()),
                }),
                position: Some(0),
                updated_at_ms: 900,
            },
        )
        .unwrap();
        let OwnedQueuedPatchOutcome::Updated {
            position,
            projection,
        } = outcome
        else {
            panic!("owned queued row was not updated");
        };
        assert_eq!(position, 0);
        assert_eq!(projection.id, "deep");
        assert_eq!(projection.target_gpu, Some(3));
        let replay = claimable_ids(&db, "owner-a");
        assert_eq!(replay, ["deep", "first", "second", "tail"]);
        let deep = get(&db, "deep").unwrap().unwrap();
        assert_eq!(deep.target_gpu, Some(3));
        assert_eq!(deep.target_device_id.as_deref(), Some("cuda:stable"));

        drop(db);
        let db = MetadataDb::open(&path).unwrap();
        assert_eq!(
            claimable_ids(&db, "owner-a"),
            ["deep", "first", "second", "tail"],
            "the acknowledged order is identical after reopening SQLite"
        );

        let outcome = patch_owned_queued(
            &db,
            "owner-a",
            "deep",
            &OwnedQueuedPatch {
                target: Some(QueueTargetPatch {
                    target_gpu: None,
                    target_device_id: None,
                }),
                position: Some(usize::MAX),
                updated_at_ms: 901,
            },
        )
        .unwrap();
        assert!(matches!(
            outcome,
            OwnedQueuedPatchOutcome::Updated {
                position: 3,
                projection: GenerationQueueProjection {
                    target_gpu: None,
                    ..
                },
            }
        ));
        assert_eq!(
            claimable_ids(&db, "owner-a"),
            ["first", "second", "tail", "deep"]
        );
        assert_eq!(get(&db, "deep").unwrap().unwrap().target_gpu, None);
    }

    #[test]
    fn owned_queued_patch_is_owner_state_fenced_and_rolls_back_partial_writes() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "deep", "tail"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }
        insert(&db, &row("foreign", "owner-b", 500)).unwrap();
        hold(&db, "tail", "review", 600).unwrap();

        let patch = OwnedQueuedPatch {
            target: Some(QueueTargetPatch {
                target_gpu: Some(7),
                target_device_id: Some("cuda:7".into()),
            }),
            position: Some(0),
            updated_at_ms: 700,
        };
        assert_eq!(
            patch_owned_queued(&db, "owner-a", "foreign", &patch).unwrap(),
            OwnedQueuedPatchOutcome::NotOwned
        );
        assert_eq!(
            patch_owned_queued(&db, "owner-a", "tail", &patch).unwrap(),
            OwnedQueuedPatchOutcome::NotQueued
        );

        db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER reject_deep_reorder
                 BEFORE UPDATE OF created_at ON generation_queue
                 WHEN OLD.id = 'first'
                 BEGIN SELECT RAISE(ABORT, 'injected reorder failure'); END;",
            )?;
            Ok(())
        })
        .unwrap();
        assert!(patch_owned_queued(&db, "owner-a", "deep", &patch).is_err());
        assert_eq!(get(&db, "deep").unwrap().unwrap().target_gpu, None);
        assert_eq!(claimable_ids(&db, "owner-a"), ["first", "deep"]);
    }

    #[test]
    fn owned_queued_patch_refuses_a_claimed_handoff_without_mutation() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "handoff", "tail"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }
        claim_by_id(&db, "owner-a", "handoff", "feeder-claim", 600)
            .unwrap()
            .unwrap();
        let before = claimable(&db, "owner-a");

        let outcome = patch_owned_queued(
            &db,
            "owner-a",
            "handoff",
            &OwnedQueuedPatch {
                target: Some(QueueTargetPatch {
                    target_gpu: Some(7),
                    target_device_id: Some("cuda:7".into()),
                }),
                position: Some(0),
                updated_at_ms: 700,
            },
        )
        .unwrap();

        assert_eq!(outcome, OwnedQueuedPatchOutcome::NotQueued);
        assert_eq!(claimable(&db, "owner-a"), before);
        let handoff = get(&db, "handoff").unwrap().unwrap();
        assert_eq!(handoff.target_gpu, None);
        assert_eq!(handoff.target_device_id, None);
        let claim_token: Option<String> = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT claim_token FROM generation_queue WHERE id = 'handoff'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(claim_token.as_deref(), Some("feeder-claim"));
    }

    #[test]
    fn owned_claimed_queued_patch_persists_live_affinity_and_position() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "live", "tail"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }
        claim_by_id(&db, "owner-a", "live", "registry-claim", 600)
            .unwrap()
            .unwrap();

        let outcome = patch_owned_claimed_queued(
            &db,
            "owner-a",
            "live",
            &OwnedQueuedPatch {
                target: Some(QueueTargetPatch {
                    target_gpu: Some(4),
                    target_device_id: Some("cuda:stable-live".into()),
                }),
                position: Some(0),
                updated_at_ms: 700,
            },
        )
        .unwrap();
        assert!(matches!(
            outcome,
            OwnedQueuedPatchOutcome::Updated {
                position: 0,
                projection: GenerationQueueProjection {
                    ref id,
                    target_gpu: Some(4),
                    ..
                },
            } if id == "live"
        ));
        let live = get(&db, "live").unwrap().unwrap();
        assert_eq!(live.target_gpu, Some(4));
        assert_eq!(live.target_device_id.as_deref(), Some("cuda:stable-live"));
        assert_eq!(claimable_ids(&db, "owner-a"), ["live", "first", "tail"]);

        recover_runtime_claims(&db, "owner-a", 800).unwrap();
        let live = get(&db, "live").unwrap().unwrap();
        assert_eq!(live.target_gpu, Some(4));
        assert_eq!(live.target_device_id.as_deref(), Some("cuda:stable-live"));
        assert_eq!(claimable_ids(&db, "owner-a"), ["live", "first", "tail"]);

        assert_eq!(
            patch_owned_claimed_queued(
                &db,
                "owner-a",
                "first",
                &OwnedQueuedPatch {
                    target: None,
                    position: Some(2),
                    updated_at_ms: 900,
                },
            )
            .unwrap(),
            OwnedQueuedPatchOutcome::NotQueued
        );
    }

    #[test]
    fn owned_any_queued_patch_accepts_both_claim_states_and_preserves_exact_tokens() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["unclaimed", "claimed", "tail"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }
        claim_by_id(&db, "owner-a", "claimed", "exact-claim", 600)
            .unwrap()
            .unwrap();
        let patch = |position, gpu| OwnedQueuedPatch {
            target: Some(QueueTargetPatch {
                target_gpu: Some(gpu),
                target_device_id: Some(format!("cuda:{gpu}")),
            }),
            position: Some(position),
            updated_at_ms: 700 + gpu as i64,
        };

        assert!(matches!(
            patch_owned_any_queued(&db, "owner-a", "unclaimed", &patch(2, 1)).unwrap(),
            OwnedQueuedPatchOutcome::Updated { .. }
        ));
        assert!(matches!(
            patch_owned_any_queued(&db, "owner-a", "claimed", &patch(0, 2)).unwrap(),
            OwnedQueuedPatchOutcome::Updated { position: 0, .. }
        ));
        let claim_token: Option<String> = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT claim_token FROM generation_queue WHERE id = 'claimed'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(claim_token.as_deref(), Some("exact-claim"));
        assert_eq!(get(&db, "claimed").unwrap().unwrap().target_gpu, Some(2));
        assert_eq!(
            claimable_ids(&db, "owner-a"),
            ["claimed", "tail", "unclaimed"]
        );
    }

    #[test]
    fn owned_claimed_queued_patch_is_owner_and_state_fenced() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert(&db, &row("foreign", "owner-b", 500)).unwrap();
        insert(&db, &row("running", "owner-a", 501)).unwrap();
        insert(&db, &row("held", "owner-a", 502)).unwrap();
        claim_by_id(&db, "owner-b", "foreign", "foreign-claim", 600)
            .unwrap()
            .unwrap();
        claim_by_id(&db, "owner-a", "running", "running-claim", 601)
            .unwrap()
            .unwrap();
        mark_dispatched_claimed(&db, "running", "running-claim", 602)
            .unwrap()
            .unwrap();
        claim_by_id(&db, "owner-a", "held", "held-claim", 603)
            .unwrap()
            .unwrap();
        hold_claimed(
            &db,
            "held",
            "held-claim",
            QueueRowState::Queued,
            "review",
            604,
        )
        .unwrap();
        let patch = OwnedQueuedPatch {
            target: Some(QueueTargetPatch {
                target_gpu: Some(8),
                target_device_id: Some("cuda:8".into()),
            }),
            position: Some(0),
            updated_at_ms: 700,
        };

        assert_eq!(
            patch_owned_claimed_queued(&db, "owner-a", "foreign", &patch).unwrap(),
            OwnedQueuedPatchOutcome::NotOwned
        );
        assert_eq!(
            patch_owned_claimed_queued(&db, "owner-a", "running", &patch).unwrap(),
            OwnedQueuedPatchOutcome::NotQueued
        );
        assert_eq!(
            patch_owned_claimed_queued(&db, "owner-a", "held", &patch).unwrap(),
            OwnedQueuedPatchOutcome::NotQueued
        );
        for id in ["foreign", "running", "held"] {
            let unchanged = get(&db, id).unwrap().unwrap();
            assert_eq!(unchanged.target_gpu, None);
            assert_eq!(unchanged.target_device_id, None);
        }
    }

    #[test]
    fn owned_claimed_queued_patch_rolls_back_partial_writes() {
        let db = MetadataDb::open_in_memory().unwrap();
        for id in ["first", "live", "tail"] {
            insert(&db, &row(id, "owner-a", 500)).unwrap();
        }
        claim_by_id(&db, "owner-a", "live", "registry-claim", 600)
            .unwrap()
            .unwrap();
        let before = claimable(&db, "owner-a");
        db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER reject_live_reorder
                 BEFORE UPDATE OF created_at ON generation_queue
                 WHEN OLD.id = 'first'
                 BEGIN SELECT RAISE(ABORT, 'injected live reorder failure'); END;",
            )?;
            Ok(())
        })
        .unwrap();

        let result = patch_owned_claimed_queued(
            &db,
            "owner-a",
            "live",
            &OwnedQueuedPatch {
                target: Some(QueueTargetPatch {
                    target_gpu: Some(9),
                    target_device_id: Some("cuda:9".into()),
                }),
                position: Some(0),
                updated_at_ms: 700,
            },
        );
        assert!(result.is_err());
        assert_eq!(claimable(&db, "owner-a"), before);
        let live = get(&db, "live").unwrap().unwrap();
        assert_eq!(live.target_gpu, None);
        assert_eq!(live.target_device_id, None);
        let claim_token: Option<String> = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT claim_token FROM generation_queue WHERE id = 'live'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(claim_token.as_deref(), Some("registry-claim"));
    }

    #[test]
    fn publication_lookup_query_plans_are_index_bounded() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.with_conn(|conn| {
            let plans = [
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
