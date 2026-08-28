//! Lightweight durable grouping for heterogeneous generation admission.
//!
//! Child execution authority remains `generation_queue`; these tables only
//! make one client admission idempotent and retain terminal child summaries
//! after the queue rows are removed.

use anyhow::{bail, ensure, Result};
use rusqlite::{params, OptionalExtension};
use std::collections::{HashMap, HashSet};

use crate::generation_queue::{self, GenerationQueueRow};
use crate::generation_queue_media::{self, QueueMediaObligation};
use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchRow {
    pub id: String,
    pub client_batch_id: String,
    pub owner_uuid: String,
    pub request_sha256: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchChildRow {
    pub batch_id: String,
    pub job_id: String,
    pub batch_index: u32,
    pub state: String,
    pub error: Option<String>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchDetail {
    pub batch: GenerationBatchRow,
    pub children: Vec<GenerationBatchChildRow>,
}

/// Outcome of a file-first batch admission carrying staged media.
///
/// An idempotency loser is not an error from the cleanup perspective: its
/// distinct staged sets are durably recorded as gc-pending in the same
/// transaction that observes the winner. `request_sha256` is an opaque,
/// randomized receipt for this API: the DB returns the winner's stored receipt
/// without comparing or classifying it. Current servers use a store-independent
/// verifier; older encrypted receipts remain readable at the server boundary
/// during rolling upgrades.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationBatchMediaInsertOutcome {
    Inserted(GenerationBatchDetail),
    Existing {
        detail: GenerationBatchDetail,
        gc_pending_media_set_ids: Vec<String>,
        colliding_media_set_ids: Vec<String>,
    },
}

/// Additive reconnect view. The legacy detail remains unchanged for callers
/// that only understand `state` and `error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableGenerationBatchChildRow {
    pub batch_id: String,
    pub job_id: String,
    pub batch_index: u32,
    pub state: String,
    pub error: Option<String>,
    /// Typed cause of a held child (`MODEL_NOT_FOUND`, `UNKNOWN_MODEL`, …),
    /// persisted beside the sentence so clients can act on it; `None` for a
    /// hold with no typed cause and for every non-held state.
    pub error_code: Option<String>,
    pub retryable: bool,
    pub updated_at_ms: i64,
    /// Monotonic per-child version. Every authoritative state transition
    /// increments it; nothing else writes it. Clients order snapshots by
    /// this rather than by `updated_at_ms`, which collides within a
    /// millisecond. See migration v29.
    pub revision: i64,
    pub terminal_error_json: Option<String>,
    pub result_json: Option<String>,
    pub completed_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableGenerationBatchDetail {
    pub batch: GenerationBatchRow,
    pub children: Vec<DurableGenerationBatchChildRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableGenerationBatchLookup {
    pub batches: Vec<DurableGenerationBatchDetail>,
    pub missing_client_batch_ids: Vec<String>,
    pub missing_batch_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationBatchTerminalState {
    Complete,
    Failed,
    Cancelled,
}

impl GenerationBatchTerminalState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Opaque structured terminal values are validated by the server/core layer.
/// The DB commits them atomically with the legacy terminal summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationBatchTerminal<'a> {
    pub state: GenerationBatchTerminalState,
    pub error: Option<&'a str>,
    pub terminal_error_json: Option<&'a str>,
    pub result_json: Option<&'a str>,
    pub completed_at_ms: i64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClaimedTerminalCommit {
    pub queue_deleted: bool,
    pub batch_child_updated: bool,
    /// The child was already cancelling, so the transaction committed
    /// cancellation instead of the requested terminal result.
    pub cancelled: bool,
}

/// Durable result of cancelling one queue row through its owning server.
///
/// `Requested` retains the token-fenced queue row so the live worker can
/// settle it. `Settled` removes execution authority in the same transaction
/// that records any batch-child terminal outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnedCancellation {
    NotOwned,
    Requested,
    Settled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnedHold {
    Held,
    Cancelled,
    Fenced,
}

/// Result of an explicit retry request for a parked durable generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnedRetry {
    Retried,
    NotOwned,
    AuthorityMismatch,
    NotHeld,
    NotRetryable,
}

/// Replace one opaque admission receipt only while the exact previously
/// verified value still owns the row. This is the rolling-upgrade fence from
/// legacy encrypted receipts to the store-independent receipt authority.
pub fn replace_request_receipt(
    db: &MetadataDb,
    owner_uuid: &str,
    batch_id: &str,
    expected: &str,
    replacement: &str,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_batches
                SET request_sha256 = ?1
              WHERE id = ?2 AND owner_uuid = ?3 AND request_sha256 = ?4",
            params![replacement, batch_id, owner_uuid, expected],
        )? == 1)
    })
}

/// Whether any queue owner has durable receipts issued by the named protocol.
/// The admission key is database/MOLD_HOME-global, so creation must preserve
/// receipt evidence belonging to inactive and orphaned owners too.
pub fn has_any_request_receipt_prefix(db: &MetadataDb, prefix: &str) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM generation_batches
                  WHERE substr(request_sha256, 1, length(?1)) = ?1
             )",
            params![prefix],
            |row| row.get(0),
        )?)
    })
}

/// Insert the grouping rows and every ordinary durable queue row atomically.
/// A retry with the same client id returns the existing detail; a different
/// payload using that id is rejected.
pub fn insert_or_get(
    db: &MetadataDb,
    batch: &GenerationBatchRow,
    children: &[(GenerationBatchChildRow, GenerationQueueRow)],
) -> Result<(GenerationBatchDetail, bool)> {
    db.transact_immediate(|conn| {
        if let Some(existing) =
            get_by_client_on_conn(conn, &batch.owner_uuid, &batch.client_batch_id)?
        {
            if existing.batch.request_sha256 != batch.request_sha256 {
                bail!("client_batch_id was already used for a different request");
            }
            return Ok((existing, false));
        }
        conn.execute(
            "INSERT INTO generation_batches
                (id, client_batch_id, owner_uuid, request_sha256, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                batch.id,
                batch.client_batch_id,
                batch.owner_uuid,
                batch.request_sha256,
                batch.created_at_ms,
            ],
        )?;
        for (child, queue_row) in children {
            generation_queue::insert_on_conn(conn, queue_row)?;
            conn.execute(
                "INSERT INTO generation_batch_children
                    (batch_id, job_id, batch_index, state, error, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    child.batch_id,
                    child.job_id,
                    child.batch_index,
                    child.state,
                    child.error,
                    child.updated_at_ms,
                ],
            )?;
        }
        Ok((
            GenerationBatchDetail {
                batch: batch.clone(),
                children: children.iter().map(|(child, _)| child.clone()).collect(),
            },
            true,
        ))
    })
}

/// Insert a batch, every queue row, and every active media obligation in one
/// immediate transaction.
///
/// `obligations` contains exactly one entry for each child whose queue row has
/// a non-NULL media marker. When the client id already exists, the incoming
/// obligations belong to the losing file-first contender, so they are instead
/// committed as gc-pending and returned by id for immediate cleanup. The
/// existing row's opaque receipt is returned verbatim; only the server may
/// authenticate it and decide whether the operation matches.
pub fn insert_or_get_with_media(
    db: &MetadataDb,
    batch: &GenerationBatchRow,
    children: &[(GenerationBatchChildRow, GenerationQueueRow)],
    obligations: &[QueueMediaObligation],
) -> Result<GenerationBatchMediaInsertOutcome> {
    let mut obligations_by_id = HashMap::new();
    for obligation in obligations {
        if obligations_by_id
            .insert(obligation.media_set_id.as_str(), obligation)
            .is_some()
        {
            bail!(
                "duplicate queue media obligation {}",
                obligation.media_set_id
            );
        }
    }
    let mut referenced = HashSet::new();
    for (child, queue_row) in children {
        ensure!(
            child.batch_id == batch.id,
            "batch child points at a different batch"
        );
        ensure!(
            child.job_id == queue_row.id,
            "batch child and queue row have different job ids"
        );
        ensure!(
            queue_row.owner_uuid == batch.owner_uuid,
            "batch and queue row have different owners"
        );
        let Some(media_set_id) = queue_row.media_set_id.as_deref() else {
            continue;
        };
        let obligation = obligations_by_id.get(media_set_id).ok_or_else(|| {
            anyhow::anyhow!(
                "queue row {0} is missing its media obligation",
                queue_row.id
            )
        })?;
        generation_queue_media::validate_row_obligation(
            Some(media_set_id),
            &queue_row.owner_uuid,
            obligation,
        )?;
        referenced.insert(media_set_id);
    }
    ensure!(
        referenced.len() == obligations_by_id.len(),
        "batch includes an unreferenced queue media obligation"
    );

    db.transact_immediate(|conn| {
        if let Some(existing) =
            get_by_client_on_conn(conn, &batch.owner_uuid, &batch.client_batch_id)?
        {
            let mut gc_pending_media_set_ids = Vec::new();
            let mut colliding_media_set_ids = Vec::new();
            for obligation in obligations {
                if generation_queue_media::ensure_gc_pending_on_conn(conn, obligation)? {
                    gc_pending_media_set_ids.push(obligation.media_set_id.clone());
                } else {
                    colliding_media_set_ids.push(obligation.media_set_id.clone());
                }
            }
            return Ok(GenerationBatchMediaInsertOutcome::Existing {
                detail: existing,
                gc_pending_media_set_ids,
                colliding_media_set_ids,
            });
        }
        conn.execute(
            "INSERT INTO generation_batches
                (id, client_batch_id, owner_uuid, request_sha256, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                batch.id,
                batch.client_batch_id,
                batch.owner_uuid,
                batch.request_sha256,
                batch.created_at_ms,
            ],
        )?;
        for (child, queue_row) in children {
            if let Some(media_set_id) = queue_row.media_set_id.as_deref() {
                generation_queue::insert_on_conn_with_media(
                    conn,
                    queue_row,
                    obligations_by_id[media_set_id],
                )?;
            } else {
                generation_queue::insert_on_conn(conn, queue_row)?;
            }
            conn.execute(
                "INSERT INTO generation_batch_children
                    (batch_id, job_id, batch_index, state, error, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    child.batch_id,
                    child.job_id,
                    child.batch_index,
                    child.state,
                    child.error,
                    child.updated_at_ms,
                ],
            )?;
        }
        Ok(GenerationBatchMediaInsertOutcome::Inserted(
            GenerationBatchDetail {
                batch: batch.clone(),
                children: children.iter().map(|(child, _)| child.clone()).collect(),
            },
        ))
    })
}

pub fn get(db: &MetadataDb, owner_uuid: &str, id: &str) -> Result<Option<GenerationBatchDetail>> {
    db.with_conn(|conn| {
        let batch = conn
            .query_row(
                "SELECT id, client_batch_id, owner_uuid, request_sha256, created_at_ms
                   FROM generation_batches WHERE id = ?1 AND owner_uuid = ?2",
                params![id, owner_uuid],
                batch_from_row,
            )
            .optional()?;
        batch.map(|batch| detail_on_conn(conn, batch)).transpose()
    })
}

pub fn get_by_client(
    db: &MetadataDb,
    owner_uuid: &str,
    client_batch_id: &str,
) -> Result<Option<GenerationBatchDetail>> {
    db.with_conn(|conn| get_by_client_on_conn(conn, owner_uuid, client_batch_id))
}

pub fn get_durable(
    db: &MetadataDb,
    owner_uuid: &str,
    id: &str,
) -> Result<Option<DurableGenerationBatchDetail>> {
    db.with_conn(|conn| get_durable_on_conn(conn, owner_uuid, id))
}

pub fn get_durable_by_client(
    db: &MetadataDb,
    owner_uuid: &str,
    client_batch_id: &str,
) -> Result<Option<DurableGenerationBatchDetail>> {
    db.with_conn(|conn| get_durable_by_client_on_conn(conn, owner_uuid, client_batch_id))
}

/// Resolve reconnect state without materializing any unrelated batch rows.
/// Inputs are de-duplicated in first-seen order; matches requested through
/// both identities appear once while each unknown identity remains explicit.
pub fn lookup_durable(
    db: &MetadataDb,
    owner_uuid: &str,
    client_batch_ids: &[String],
    batch_ids: &[String],
) -> Result<DurableGenerationBatchLookup> {
    db.with_conn(|conn| {
        let mut batches = Vec::new();
        let mut missing_client_batch_ids = Vec::new();
        let mut missing_batch_ids = Vec::new();
        let mut seen_client_ids = HashSet::new();
        let mut seen_batch_ids = HashSet::new();
        let mut returned_batch_ids = HashSet::new();

        for client_batch_id in client_batch_ids {
            if !seen_client_ids.insert(client_batch_id.as_str()) {
                continue;
            }
            match get_durable_by_client_on_conn(conn, owner_uuid, client_batch_id)? {
                Some(detail) => {
                    returned_batch_ids.insert(detail.batch.id.clone());
                    batches.push(detail);
                }
                None => missing_client_batch_ids.push(client_batch_id.clone()),
            }
        }

        for batch_id in batch_ids {
            if !seen_batch_ids.insert(batch_id.as_str()) {
                continue;
            }
            match get_durable_on_conn(conn, owner_uuid, batch_id)? {
                Some(detail) if returned_batch_ids.insert(detail.batch.id.clone()) => {
                    batches.push(detail);
                }
                Some(_) => {}
                None => missing_batch_ids.push(batch_id.clone()),
            }
        }

        Ok(DurableGenerationBatchLookup {
            batches,
            missing_client_batch_ids,
            missing_batch_ids,
        })
    })
}

pub fn set_child_state(
    db: &MetadataDb,
    job_id: &str,
    state: &str,
    error: Option<&str>,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_batch_children
                SET state = ?2, error = ?3, updated_at_ms = ?4,
                    revision = revision + 1
              WHERE job_id = ?1
                AND state NOT IN ('cancelling', 'complete', 'failed', 'cancelled')",
            params![job_id, state, error, updated_at_ms],
        )? > 0)
    })
}

/// Commit a terminal outcome and remove its active queue authority in one
/// transaction. Both the expected durable state and runtime token must match;
/// stale owners receive a clean no-op and cannot alter the child summary.
/// Singleton rows are supported: `batch_child_updated` is false while the
/// claimed queue row is still deleted atomically.
pub fn finish_claimed(
    db: &MetadataDb,
    job_id: &str,
    claim_token: &str,
    expected_queue_state: crate::generation_queue::QueueRowState,
    terminal: GenerationBatchTerminal<'_>,
) -> Result<ClaimedTerminalCommit> {
    db.transact_immediate(|conn| {
        let owned: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM generation_queue
                  WHERE id = ?1 AND state = ?2 AND claim_token = ?3",
                params![job_id, expected_queue_state.as_str(), claim_token],
                |row| row.get(0),
            )
            .optional()?;
        if owned.is_none() {
            return Ok(ClaimedTerminalCommit::default());
        }

        let cancelled = conn
            .query_row(
                "SELECT state FROM generation_batch_children WHERE job_id = ?1",
                params![job_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .as_deref()
            == Some("cancelling");
        let child_updated = update_terminal_child(conn, job_id, terminal)?;
        let queue_deleted = conn.execute(
            "DELETE FROM generation_queue
              WHERE id = ?1 AND state = ?2 AND claim_token = ?3",
            params![job_id, expected_queue_state.as_str(), claim_token],
        )?;
        if queue_deleted != 1 {
            bail!("claimed queue row changed during terminal commit");
        }
        Ok(ClaimedTerminalCommit {
            queue_deleted: true,
            batch_child_updated: child_updated,
            cancelled,
        })
    })
}

/// A fully settled batch whose retention has elapsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpiredSettledBatch {
    pub id: String,
    /// The newest child settlement — the moment the batch became a receipt.
    pub settled_at_ms: i64,
}

/// The one definition of "settled": every child terminal and no child still
/// owning a `generation_queue` row. Bound as an aggregate `HAVING` clause over
/// `generation_batches AS b JOIN generation_batch_children AS c`, so the
/// listing, the count, and the purge cannot disagree.
///
/// A batch with an `accepted`-only child that never reached the queue is not
/// settled and is deliberately out of scope here.
const SETTLED_BATCH_HAVING_SQL: &str = "SUM(c.state NOT IN ('complete', 'failed', 'cancelled')) = 0
     AND NOT EXISTS (
         SELECT 1 FROM generation_queue AS q
           JOIN generation_batch_children AS cc ON cc.job_id = q.id
          WHERE cc.batch_id = b.id
     )";

fn settled_retention_cutoff_ms(retention_days: u32, now_ms: i64) -> Option<i64> {
    (retention_days > 0).then(|| now_ms.saturating_sub(i64::from(retention_days) * 86_400_000))
}

/// List this owner's settled batches whose retention has elapsed, oldest
/// settlement first.
///
/// Shares `queue.held_retention_days` with the held-row sweep: both answer
/// how long the durable queue remembers work that can no longer run, and
/// `retention_days == 0` keeps settled summaries forever exactly as it keeps
/// held rows. Age is `MAX(children.updated_at_ms)`, the last child
/// settlement, so a batch with a held child waits for the held sweep to
/// settle that child and only then starts its own clock.
pub fn expired_settled(
    db: &MetadataDb,
    owner_uuid: &str,
    retention_days: u32,
    now_ms: i64,
) -> Result<Vec<ExpiredSettledBatch>> {
    let Some(cutoff) = settled_retention_cutoff_ms(retention_days, now_ms) else {
        return Ok(Vec::new());
    };
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(&format!(
            "SELECT b.id, MAX(c.updated_at_ms) AS settled_at_ms
               FROM generation_batches AS b
               JOIN generation_batch_children AS c ON c.batch_id = b.id
              WHERE b.owner_uuid = ?1
              GROUP BY b.id
             HAVING {SETTLED_BATCH_HAVING_SQL}
                AND MAX(c.updated_at_ms) <= ?2
              ORDER BY settled_at_ms, b.rowid"
        ))?;
        let rows = stmt
            .query_map(params![owner_uuid, cutoff], |row| {
                Ok(ExpiredSettledBatch {
                    id: row.get(0)?,
                    settled_at_ms: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    })
}

/// Count this owner's settled batches, expired or not — the sweep's
/// "remaining" report: receipts that retention will reclaim someday.
pub fn settled_count(db: &MetadataDb, owner_uuid: &str) -> Result<u64> {
    db.with_conn(|conn| {
        let count: i64 = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM (
                    SELECT b.id
                      FROM generation_batches AS b
                      JOIN generation_batch_children AS c ON c.batch_id = b.id
                     WHERE b.owner_uuid = ?1
                     GROUP BY b.id
                    HAVING {SETTLED_BATCH_HAVING_SQL}
                 )"
            ),
            params![owner_uuid],
            |row| row.get(0),
        )?;
        Ok(count.max(0) as u64)
    })
}

/// Purge one settled batch; its child summaries cascade with it.
///
/// Eligibility is re-checked inside the transaction — owned, every child
/// still terminal, no child back in the queue, and the newest settlement
/// still past `retention_days` — so a retry that re-queued a child between
/// the listing and this call wins, and this returns `false` having touched
/// nothing. A settled batch holds no media: its media obligations were
/// retired when its queue rows were deleted, so nothing else is released.
pub fn purge_settled(
    db: &MetadataDb,
    owner_uuid: &str,
    id: &str,
    retention_days: u32,
    now_ms: i64,
) -> Result<bool> {
    let Some(cutoff) = settled_retention_cutoff_ms(retention_days, now_ms) else {
        return Ok(false);
    };
    db.transact_immediate(|conn| {
        let eligible: Option<String> = conn
            .query_row(
                &format!(
                    "SELECT b.id
                       FROM generation_batches AS b
                       JOIN generation_batch_children AS c ON c.batch_id = b.id
                      WHERE b.id = ?1 AND b.owner_uuid = ?2
                      GROUP BY b.id
                     HAVING {SETTLED_BATCH_HAVING_SQL}
                        AND MAX(c.updated_at_ms) <= ?3"
                ),
                params![id, owner_uuid, cutoff],
                |row| row.get(0),
            )
            .optional()?;
        if eligible.is_none() {
            return Ok(false);
        }
        let removed = conn.execute(
            "DELETE FROM generation_batches WHERE id = ?1 AND owner_uuid = ?2",
            params![id, owner_uuid],
        )?;
        Ok(removed == 1)
    })
}

/// Cancel exactly one durable row owned by `owner_uuid`.
///
/// This is one immediate transaction because a feeder may claim or hold the
/// row while the API request is in flight. Queued work is terminalized and
/// removed immediately even when the feeder already claimed it: a queued
/// claim is preparation authority, not permission to survive cancellation.
/// Running work records cancellation intent while retaining its token-fenced
/// execution authority until the worker acknowledges the cooperative stop.
/// Legacy/singleton rows have no reconnect summary, so removing their owned
/// queue row is the terminal cancellation record.
pub fn cancel_owned(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    terminal: GenerationBatchTerminal<'_>,
) -> Result<OwnedCancellation> {
    if terminal.state != GenerationBatchTerminalState::Cancelled {
        bail!("owned cancellation requires a cancelled terminal outcome");
    }
    db.transact_immediate(|conn| {
        let row = conn
            .query_row(
                "SELECT q.state,
                        EXISTS (
                            SELECT 1 FROM generation_batch_children AS child
                             WHERE child.job_id = q.id
                        )
                   FROM generation_queue AS q
                  WHERE q.id = ?1 AND q.owner_uuid = ?2",
                params![job_id, owner_uuid],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? != 0)),
            )
            .optional()?;
        let Some((state, has_child)) = row else {
            return Ok(OwnedCancellation::NotOwned);
        };

        let can_settle_now = matches!(state.as_str(), "held" | "queued" | "paused");
        if can_settle_now || !has_child {
            if has_child {
                update_terminal_child(conn, job_id, terminal)?;
            }
            let deleted = conn.execute(
                "DELETE FROM generation_queue WHERE id = ?1 AND owner_uuid = ?2",
                params![job_id, owner_uuid],
            )?;
            if deleted != 1 {
                bail!("owned queue row changed during cancellation");
            }
            return Ok(OwnedCancellation::Settled);
        }

        let requested = conn.execute(
            "UPDATE generation_batch_children
                SET state = 'cancelling', error = 'Cancelled', updated_at_ms = ?2,
                    revision = revision + 1
              WHERE job_id = ?1
                AND state NOT IN ('complete', 'failed', 'cancelled')",
            params![job_id, terminal.completed_at_ms],
        )?;
        if requested != 1 {
            bail!("claimed batch child was already terminal during cancellation");
        }
        Ok(OwnedCancellation::Requested)
    })
}

/// Atomically park one owner- and token-scoped row together with its child
/// summary, unless cancellation already won.
///
/// `claim_token = None` is the legacy unclaimed attachment path and matches
/// only an unclaimed row. A cancellation request is terminalized instead of
/// being overwritten with `held`; the queue row is removed in that same
/// transaction so a 204 cancellation cannot later become replayable work.
#[allow(clippy::too_many_arguments)] // one transaction's fence tuple plus the hold's reason, code, and retryability
pub fn hold_owned(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    claim_token: Option<&str>,
    reason: &str,
    code: Option<&str>,
    retryable: bool,
    now_ms: i64,
) -> Result<OwnedHold> {
    db.transact_immediate(|conn| {
        let row = conn
            .query_row(
                "SELECT q.state,
                        (SELECT child.state
                           FROM generation_batch_children AS child
                          WHERE child.job_id = q.id)
                   FROM generation_queue AS q
                  WHERE q.id = ?1 AND q.owner_uuid = ?2
                    AND q.claim_token IS ?3",
                params![job_id, owner_uuid, claim_token],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        let Some((queue_state, child_state)) = row else {
            let cancelled = conn
                .query_row(
                    "SELECT child.state = 'cancelled'
                       FROM generation_batch_children AS child
                       JOIN generation_batches AS batch ON batch.id = child.batch_id
                      WHERE child.job_id = ?1 AND batch.owner_uuid = ?2",
                    params![job_id, owner_uuid],
                    |row| row.get::<_, bool>(0),
                )
                .optional()?
                .unwrap_or(false);
            return Ok(if cancelled {
                OwnedHold::Cancelled
            } else {
                OwnedHold::Fenced
            });
        };

        if child_state.as_deref() == Some("cancelling") {
            update_terminal_child(
                conn,
                job_id,
                GenerationBatchTerminal {
                    state: GenerationBatchTerminalState::Cancelled,
                    error: Some("Cancelled"),
                    terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                    result_json: None,
                    completed_at_ms: now_ms,
                },
            )?;
            let deleted = conn.execute(
                "DELETE FROM generation_queue
                  WHERE id = ?1 AND owner_uuid = ?2 AND claim_token IS ?3",
                params![job_id, owner_uuid, claim_token],
            )?;
            if deleted != 1 {
                bail!("owned queue row changed while terminalizing cancellation");
            }
            return Ok(OwnedHold::Cancelled);
        }
        if child_state
            .as_deref()
            .is_some_and(|state| matches!(state, "complete" | "failed" | "cancelled"))
        {
            bail!("terminal batch child retained active queue authority during hold");
        }

        let held = conn.execute(
            "UPDATE generation_queue
                SET state = 'held', held_reason = ?4, retryable = ?5, updated_at = ?6
              WHERE id = ?1 AND owner_uuid = ?2 AND claim_token IS ?3 AND state = ?7",
            params![
                job_id,
                owner_uuid,
                claim_token,
                reason,
                retryable,
                now_ms,
                queue_state
            ],
        )?;
        if held != 1 {
            bail!("owned queue row changed during hold");
        }
        let child_updated = conn.execute(
            "UPDATE generation_batch_children
                SET state = 'held', error = ?2, error_code = ?4, updated_at_ms = ?3,
                    revision = revision + 1
              WHERE job_id = ?1
                AND state NOT IN ('cancelling', 'complete', 'failed', 'cancelled')",
            params![job_id, reason, now_ms, code],
        )?;
        if child_state.is_some() && child_updated != 1 {
            bail!("batch child changed during hold");
        }
        Ok(OwnedHold::Held)
    })
}

/// Atomically return one explicitly retryable held row to the durable queue.
///
/// The captured instance/batch/client/job tuple, current owner, held state and
/// retryable bit are checked in the same transaction as the mutation.
/// Clearing the runtime claim and both crash-loop counters gives the
/// operator-approved attempt a fresh budget. A heterogeneous child is restored
/// in that transaction so its status cannot remain held while the queue has
/// resumed it.
/// Return one explicitly retryable held row to the feeder backlog.
///
/// `dispatch_attempts` resets because an operator-approved retry IS a fresh
/// attempt. `replay_seen` deliberately does NOT: it is the only bound on a
/// boot crash loop (a job that kills the process during its own load), it is
/// charged once per boot rather than per attempt, and it is not the operator's
/// to spend.
pub fn retry_held_owned(
    db: &MetadataDb,
    owner_uuid: &str,
    serving_instance_id: &str,
    authority: &mold_core::GenerationRetryRequest,
    now_ms: i64,
) -> Result<OwnedRetry> {
    db.transact_immediate(|conn| {
        if authority.instance_id != serving_instance_id {
            return Ok(OwnedRetry::AuthorityMismatch);
        }
        let row = conn
            .query_row(
                "SELECT state, retryable,
                        (SELECT child.state
                           FROM generation_batch_children AS child
                          WHERE child.job_id = generation_queue.id),
                        (SELECT child.batch_id
                           FROM generation_batch_children AS child
                          WHERE child.job_id = generation_queue.id),
                        (SELECT batch.client_batch_id
                           FROM generation_batch_children AS child
                           JOIN generation_batches AS batch ON batch.id = child.batch_id
                          WHERE child.job_id = generation_queue.id)
                   FROM generation_queue
                  WHERE id = ?1 AND owner_uuid = ?2",
                params![authority.job_id, owner_uuid],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)? != 0,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, retryable, child_state, batch_id, client_batch_id)) = row else {
            return Ok(OwnedRetry::NotOwned);
        };
        if batch_id.as_deref() != Some(authority.batch_id.as_str())
            || client_batch_id.as_deref() != Some(authority.client_batch_id.as_str())
        {
            return Ok(OwnedRetry::AuthorityMismatch);
        }
        if state != "held" {
            return Ok(OwnedRetry::NotHeld);
        }
        if !retryable {
            return Ok(OwnedRetry::NotRetryable);
        }
        if child_state.as_deref().is_some_and(|state| state != "held") {
            bail!("retryable queue row has a non-held batch child");
        }

        // `MAX(?3, updated_at + 1)`, not `?3`: a retry is the first transition in
        // the codebase that moves a job BACKWARD through the browser reducer's
        // rank ordering (`generationLifecycle.ts` FORWARD_PHASE_RANK — held is 2,
        // queued is 1). That reducer accepts a backward move only when the
        // snapshot is strictly newer, `revision` is never supplied on the wire,
        // so the comparison falls through to this timestamp. A retry landing in
        // the same millisecond as its own hold would be neither newer nor
        // forward, and the client would silently keep rendering Held for a job
        // that is queued. Forcing monotonicity here fixes it server-side for
        // every client, rather than redesigning the tie-break.
        let updated = conn.execute(
            "UPDATE generation_queue
                SET state = 'queued', held_reason = NULL, retryable = 0,
                    claim_token = NULL, dispatch_attempts = 0,
                    started_at = NULL, updated_at = MAX(?3, updated_at + 1)
              WHERE id = ?1 AND owner_uuid = ?2
                AND state = 'held' AND retryable = 1",
            params![authority.job_id, owner_uuid, now_ms],
        )?;
        if updated != 1 {
            bail!("retryable queue row changed during retry");
        }
        if child_state.is_some() {
            // Same monotonicity requirement: this is the row the browser
            // reducer actually reads.
            let child_updated = conn.execute(
                "UPDATE generation_batch_children
                    SET state = 'accepted', error = NULL, error_code = NULL,
                        updated_at_ms = MAX(?2, updated_at_ms + 1),
                        revision = revision + 1
                  WHERE job_id = ?1 AND state = 'held'",
                params![authority.job_id, now_ms],
            )?;
            if child_updated != 1 {
                bail!("held batch child changed during retry");
            }
        }
        Ok(OwnedRetry::Retried)
    })
}

/// Cancel one feeder-owned row that has not been hydrated or claimed yet.
/// The child summary and queue deletion commit together so reconnect never
/// observes an accepted child whose execution authority was already removed.
pub fn finish_unclaimed_queued(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    terminal: GenerationBatchTerminal<'_>,
) -> Result<ClaimedTerminalCommit> {
    db.transact_immediate(|conn| {
        let owned: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM generation_queue
                  WHERE id = ?1 AND owner_uuid = ?2
                    AND state = 'queued' AND claim_token IS NULL
                    AND EXISTS (
                        SELECT 1 FROM generation_batch_children WHERE job_id = ?1
                    )",
                params![job_id, owner_uuid],
                |row| row.get(0),
            )
            .optional()?;
        if owned.is_none() {
            return Ok(ClaimedTerminalCommit::default());
        }
        let child_updated = update_terminal_child(conn, job_id, terminal)?;
        let queue_deleted = conn.execute(
            "DELETE FROM generation_queue
              WHERE id = ?1 AND owner_uuid = ?2
                AND state = 'queued' AND claim_token IS NULL",
            params![job_id, owner_uuid],
        )?;
        Ok(ClaimedTerminalCommit {
            queue_deleted: queue_deleted == 1,
            batch_child_updated: child_updated,
            cancelled: false,
        })
    })
}

/// Atomically terminalize every unclaimed feeder child owned by this queue.
pub fn finish_all_unclaimed_queued(
    db: &MetadataDb,
    owner_uuid: &str,
    terminal: GenerationBatchTerminal<'_>,
) -> Result<usize> {
    db.transact_immediate(|conn| {
        let updated = conn.execute(
            "UPDATE generation_batch_children
                SET state = ?2, error = ?3, terminal_error_json = ?4,
                    result_json = ?5, completed_at_ms = ?6, updated_at_ms = ?6,
                    revision = revision + 1
              WHERE job_id IN (
                    SELECT q.id FROM generation_queue AS q
                     WHERE q.owner_uuid = ?1 AND q.state = 'queued'
                       AND q.claim_token IS NULL
              )",
            params![
                owner_uuid,
                terminal.state.as_str(),
                terminal.error,
                terminal.terminal_error_json,
                terminal.result_json,
                terminal.completed_at_ms,
            ],
        )?;
        conn.execute(
            "DELETE FROM generation_queue
              WHERE owner_uuid = ?1 AND state = 'queued' AND claim_token IS NULL
                AND EXISTS (
                    SELECT 1 FROM generation_batch_children AS child
                     WHERE child.job_id = generation_queue.id
                )",
            params![owner_uuid],
        )?;
        Ok(updated)
    })
}

/// Cancel every still-queued durable row owned by this server in one
/// transaction and return the number not already counted by the live
/// registry.
///
/// Every queued batch child becomes terminal with its queue row removed,
/// including feeder-claimed preparation that has not reached runtime. Running
/// work is outside this operation and keeps cooperative cancellation authority.
/// Legacy rows have no reconnect history and are deleted directly.
/// `already_counted_live` is bounded by runtime queue capacity, so probing that
/// intersection never materializes the deep durable backlog.
pub fn cancel_all_queued(
    db: &MetadataDb,
    owner_uuid: &str,
    already_counted_live: &[String],
    terminal: GenerationBatchTerminal<'_>,
) -> Result<usize> {
    if terminal.state != GenerationBatchTerminalState::Cancelled {
        bail!("bulk queue cancellation requires a cancelled terminal outcome");
    }
    db.transact_immediate(|conn| {
        let inconsistent_child: Option<(String, String)> = conn
            .query_row(
                "SELECT q.id, child.state
                   FROM generation_queue AS q
                   JOIN generation_batch_children AS child ON child.job_id = q.id
                  WHERE q.owner_uuid = ?1 AND q.state IN ('queued', 'paused')
                    AND child.state NOT IN ('accepted', 'running', 'paused', 'held', 'cancelling')
                  LIMIT 1",
                params![owner_uuid],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((job_id, state)) = inconsistent_child {
            bail!("queued authority {job_id} retained inconsistent batch child state {state}");
        }
        let orphaned_cancellation: Option<String> = conn
            .query_row(
                "SELECT child.job_id
                   FROM generation_batch_children AS child
                   JOIN generation_batches AS batch ON batch.id = child.batch_id
                  WHERE batch.owner_uuid = ?1 AND child.state = 'cancelling'
                    AND NOT EXISTS (
                        SELECT 1 FROM generation_queue AS q WHERE q.id = child.job_id
                    )
                  LIMIT 1",
                params![owner_uuid],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(job_id) = orphaned_cancellation {
            bail!("cancelling batch child {job_id} has no durable queue authority");
        }

        let eligible_sql = "SELECT COUNT(*) FROM generation_queue AS q
             WHERE q.owner_uuid = ?1 AND q.state IN ('queued', 'paused')";
        let total: i64 = conn.query_row(eligible_sql, params![owner_uuid], |row| row.get(0))?;

        let mut live_overlap = 0usize;
        let mut seen = HashSet::with_capacity(already_counted_live.len());
        let mut overlap_stmt = conn.prepare(
            "SELECT 1 FROM generation_queue AS q
              WHERE q.owner_uuid = ?1 AND q.id = ?2 AND q.state IN ('queued', 'paused')
              LIMIT 1",
        )?;
        for id in already_counted_live {
            if seen.insert(id.as_str())
                && overlap_stmt
                    .query_row(params![owner_uuid, id], |_| Ok(()))
                    .optional()?
                    .is_some()
            {
                live_overlap += 1;
            }
        }
        drop(overlap_stmt);

        let queued_children: i64 = conn.query_row(
            "SELECT COUNT(*)
               FROM generation_queue AS q
               JOIN generation_batch_children AS child ON child.job_id = q.id
              WHERE q.owner_uuid = ?1 AND q.state IN ('queued', 'paused')",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        let terminalized = conn.execute(
            "UPDATE generation_batch_children
                SET state = ?2, error = ?3, terminal_error_json = ?4,
                    result_json = ?5, completed_at_ms = ?6, updated_at_ms = ?6,
                    revision = revision + 1
              WHERE job_id IN (
                    SELECT q.id FROM generation_queue AS q
                     WHERE q.owner_uuid = ?1 AND q.state IN ('queued', 'paused')
                )",
            params![
                owner_uuid,
                terminal.state.as_str(),
                terminal.error,
                terminal.terminal_error_json,
                terminal.result_json,
                terminal.completed_at_ms,
            ],
        )?;
        if i64::try_from(terminalized).ok() != Some(queued_children) {
            bail!(
                "bulk cancellation terminalized {terminalized} of {queued_children} queued batch children"
            );
        }
        let deleted_children = conn.execute(
            "DELETE FROM generation_queue
              WHERE owner_uuid = ?1 AND state IN ('queued', 'paused')
                AND EXISTS (
                    SELECT 1 FROM generation_batch_children AS child
                     WHERE child.job_id = generation_queue.id
                       AND child.state = 'cancelled'
                       AND child.completed_at_ms = ?2
                )",
            params![owner_uuid, terminal.completed_at_ms],
        )?;
        if i64::try_from(deleted_children).ok() != Some(queued_children) {
            bail!(
                "bulk cancellation removed {deleted_children} of {queued_children} terminalized queue authorities"
            );
        }
        conn.execute(
            "DELETE FROM generation_queue
              WHERE owner_uuid = ?1 AND state IN ('queued', 'paused')
                AND NOT EXISTS (
                    SELECT 1 FROM generation_batch_children AS child
                     WHERE child.job_id = generation_queue.id
                )",
            params![owner_uuid],
        )?;

        let remaining: i64 = conn.query_row(eligible_sql, params![owner_uuid], |row| row.get(0))?;
        if remaining != 0 {
            bail!("bulk queue cancellation left {remaining} eligible durable rows");
        }
        let invalid_remaining: i64 = conn.query_row(
            "SELECT COUNT(*)
               FROM generation_queue AS q
               JOIN generation_batch_children AS child ON child.job_id = q.id
              WHERE q.owner_uuid = ?1 AND q.state IN ('queued', 'paused')
                AND (q.claim_token IS NULL OR child.state != 'cancelling')",
            params![owner_uuid],
            |row| row.get(0),
        )?;
        if invalid_remaining != 0 {
            bail!(
                "bulk queue cancellation left {invalid_remaining} batch children without retained claimed cancellation authority"
            );
        }
        let total = usize::try_from(total)
            .map_err(|_| anyhow::anyhow!("durable cancellation count is outside usize"))?;
        total
            .checked_sub(live_overlap)
            .ok_or_else(|| anyhow::anyhow!("live cancellation overlap exceeded durable rows"))
    })
}

pub fn child_cancel_requested(db: &MetadataDb, owner_uuid: &str, job_id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn
            .query_row(
                "SELECT child.state = 'cancelling'
                   FROM generation_batch_children AS child
                   JOIN generation_queue AS q ON q.id = child.job_id
                  WHERE child.job_id = ?1 AND q.owner_uuid = ?2",
                params![job_id, owner_uuid],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(false))
    })
}

/// Restore a retained feeder child to `accepted` only while its exact owning
/// queue row is queued and unclaimed and the child is still accepted/running.
/// The queue claim release and this summary update are separate calls, so all
/// predicates are necessary: cancellation may mark or terminalize the child,
/// while a concurrent hold may park both authorities between the calls.
pub fn restore_child_after_retain(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_batch_children
                SET state = 'accepted', error = NULL, updated_at_ms = ?3,
                    revision = revision + 1
              WHERE job_id = ?1
                AND state IN ('accepted', 'running')
                AND EXISTS (
                    SELECT 1 FROM generation_queue AS q
                     WHERE q.id = ?1 AND q.owner_uuid = ?2
                       AND q.state = 'queued' AND q.claim_token IS NULL
                )",
            params![job_id, owner_uuid, updated_at_ms],
        )? > 0)
    })
}

fn update_terminal_child(
    conn: &rusqlite::Connection,
    job_id: &str,
    terminal: GenerationBatchTerminal<'_>,
) -> Result<bool> {
    Ok(conn.execute(
        "UPDATE generation_batch_children
            SET state = CASE WHEN state = 'cancelling' THEN 'cancelled' ELSE ?2 END,
                error = CASE WHEN state = 'cancelling' THEN 'Cancelled' ELSE ?3 END,
                terminal_error_json = CASE
                    WHEN state = 'cancelling' THEN '{\"message\":\"Cancelled\"}'
                    ELSE ?4
                END,
                result_json = CASE WHEN state = 'cancelling' THEN NULL ELSE ?5 END,
                completed_at_ms = ?6,
                updated_at_ms = ?6,
                revision = revision + 1
          WHERE job_id = ?1",
        params![
            job_id,
            terminal.state.as_str(),
            terminal.error,
            terminal.terminal_error_json,
            terminal.result_json,
            terminal.completed_at_ms,
        ],
    )? > 0)
}

fn get_by_client_on_conn(
    conn: &rusqlite::Connection,
    owner_uuid: &str,
    client_batch_id: &str,
) -> Result<Option<GenerationBatchDetail>> {
    let batch = conn
        .query_row(
            "SELECT id, client_batch_id, owner_uuid, request_sha256, created_at_ms
               FROM generation_batches WHERE owner_uuid = ?1 AND client_batch_id = ?2",
            params![owner_uuid, client_batch_id],
            batch_from_row,
        )
        .optional()?;
    batch.map(|batch| detail_on_conn(conn, batch)).transpose()
}

fn detail_on_conn(
    conn: &rusqlite::Connection,
    batch: GenerationBatchRow,
) -> Result<GenerationBatchDetail> {
    let mut stmt = conn.prepare(
        "SELECT batch_id, job_id, batch_index, state, error, updated_at_ms
           FROM generation_batch_children WHERE batch_id = ?1 ORDER BY batch_index",
    )?;
    let children = stmt
        .query_map(params![batch.id], |row| {
            Ok(GenerationBatchChildRow {
                batch_id: row.get(0)?,
                job_id: row.get(1)?,
                batch_index: row.get::<_, i64>(2)? as u32,
                state: row.get(3)?,
                error: row.get(4)?,
                updated_at_ms: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(GenerationBatchDetail { batch, children })
}

fn durable_detail_on_conn(
    conn: &rusqlite::Connection,
    batch: GenerationBatchRow,
) -> Result<DurableGenerationBatchDetail> {
    let mut stmt = conn.prepare(
        "SELECT child.batch_id, child.job_id, child.batch_index, child.state,
                child.error, child.updated_at_ms, child.terminal_error_json,
                child.result_json, child.completed_at_ms,
                COALESCE(queue.retryable, 0), child.revision, child.error_code
           FROM generation_batch_children child
           LEFT JOIN generation_queue queue ON queue.id = child.job_id
          WHERE child.batch_id = ?1 ORDER BY child.batch_index",
    )?;
    let children = stmt
        .query_map(params![batch.id], |row| {
            Ok(DurableGenerationBatchChildRow {
                batch_id: row.get(0)?,
                job_id: row.get(1)?,
                batch_index: row.get::<_, i64>(2)? as u32,
                state: row.get(3)?,
                error: row.get(4)?,
                updated_at_ms: row.get(5)?,
                terminal_error_json: row.get(6)?,
                result_json: row.get(7)?,
                completed_at_ms: row.get(8)?,
                retryable: row.get::<_, i64>(9)? != 0,
                revision: row.get(10)?,
                error_code: row.get(11)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(DurableGenerationBatchDetail { batch, children })
}

fn get_durable_on_conn(
    conn: &rusqlite::Connection,
    owner_uuid: &str,
    id: &str,
) -> Result<Option<DurableGenerationBatchDetail>> {
    let batch = conn
        .query_row(
            "SELECT id, client_batch_id, owner_uuid, request_sha256, created_at_ms
               FROM generation_batches WHERE id = ?1 AND owner_uuid = ?2",
            params![id, owner_uuid],
            batch_from_row,
        )
        .optional()?;
    batch
        .map(|batch| durable_detail_on_conn(conn, batch))
        .transpose()
}

fn get_durable_by_client_on_conn(
    conn: &rusqlite::Connection,
    owner_uuid: &str,
    client_batch_id: &str,
) -> Result<Option<DurableGenerationBatchDetail>> {
    let batch = conn
        .query_row(
            "SELECT id, client_batch_id, owner_uuid, request_sha256, created_at_ms
               FROM generation_batches WHERE owner_uuid = ?1 AND client_batch_id = ?2",
            params![owner_uuid, client_batch_id],
            batch_from_row,
        )
        .optional()?;
    batch
        .map(|batch| durable_detail_on_conn(conn, batch))
        .transpose()
}

fn batch_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<GenerationBatchRow> {
    Ok(GenerationBatchRow {
        id: row.get(0)?,
        client_batch_id: row.get(1)?,
        owner_uuid: row.get(2)?,
        request_sha256: row.get(3)?,
        created_at_ms: row.get(4)?,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::generation_queue::QueueRowState;
    use crate::generation_queue_media::{list_obligations, QueueMediaObligationState};

    fn rows(count: usize) -> Vec<(GenerationBatchChildRow, GenerationQueueRow)> {
        (0..count)
            .map(|index| {
                let id = format!("job-{index}");
                (
                    GenerationBatchChildRow {
                        batch_id: "batch-1".into(),
                        job_id: id.clone(),
                        batch_index: index as u32 + 1,
                        state: "accepted".into(),
                        error: None,
                        updated_at_ms: 1,
                    },
                    GenerationQueueRow {
                        id,
                        owner_uuid: "owner-1".into(),
                        state: QueueRowState::Queued,
                        model: format!("model-{index}"),
                        request_json: format!(r#"{{"prompt":"prompt-{index}"}}"#),
                        output_dir: PathBuf::from("/gallery"),
                        target_gpu: None,
                        target_device_id: None,
                        completion_payload: "metadata_only".into(),
                        seed_pinned: false,
                        dispatch_attempts: 0,
                        replay_seen: 0,
                        held_reason: None,
                        created_at_ms: 1,
                        updated_at_ms: 1,
                        started_at_ms: None,
                        media_set_id: None,
                        admission_authority: None,
                    },
                )
            })
            .collect()
    }

    fn batch(hash: &str) -> GenerationBatchRow {
        GenerationBatchRow {
            id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            owner_uuid: "owner-1".into(),
            request_sha256: hash.into(),
            created_at_ms: 1,
        }
    }

    #[test]
    fn receipt_evidence_scan_includes_inactive_owners() {
        let db = MetadataDb::open_in_memory().unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO generation_batches
                    (id, client_batch_id, owner_uuid, request_sha256, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    "orphan-batch",
                    "orphan-client",
                    "orphan-owner",
                    "generation-v2.opaque-receipt",
                    1_i64
                ],
            )?;
            Ok(())
        })
        .unwrap();

        assert!(has_any_request_receipt_prefix(&db, "generation-v2.").unwrap());
        assert!(!has_any_request_receipt_prefix(&db, "generation-v3.").unwrap());
    }

    fn media_obligation(id: &str) -> QueueMediaObligation {
        QueueMediaObligation {
            media_set_id: id.to_string(),
            owner_uuid: "owner-1".to_string(),
            state: QueueMediaObligationState::Active,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    fn media_rows(
        prefix: &str,
        count: usize,
    ) -> Vec<(GenerationBatchChildRow, GenerationQueueRow)> {
        let mut rows = rows(count);
        for (index, (child, queue)) in rows.iter_mut().enumerate() {
            let job_id = format!("{prefix}-job-{index}");
            child.job_id = job_id.clone();
            queue.id = job_id;
            queue.media_set_id = Some(format!("{prefix}-set-{index}"));
        }
        rows
    }

    fn media_for_rows(
        rows: &[(GenerationBatchChildRow, GenerationQueueRow)],
    ) -> Vec<QueueMediaObligation> {
        rows.iter()
            .map(|(_, queue)| media_obligation(queue.media_set_id.as_deref().unwrap()))
            .collect()
    }

    #[test]
    fn media_batch_insert_never_classifies_opaque_receipts_and_losers_become_gc_pending() {
        let db = MetadataDb::open_in_memory().unwrap();
        let winner_rows = media_rows("winner", 2);
        let winner_media = media_for_rows(&winner_rows);
        let winner_receipt = "opaque-winner-receipt";
        assert!(matches!(
            insert_or_get_with_media(&db, &batch(winner_receipt), &winner_rows, &winner_media,)
                .unwrap(),
            GenerationBatchMediaInsertOutcome::Inserted(_)
        ));

        let loser_rows = media_rows("loser", 2);
        let loser_media = media_for_rows(&loser_rows);
        let outcome =
            insert_or_get_with_media(&db, &batch(winner_receipt), &loser_rows, &loser_media)
                .unwrap();
        assert!(matches!(
            outcome,
            GenerationBatchMediaInsertOutcome::Existing {
                detail,
                gc_pending_media_set_ids,
                ..
            } if detail.batch.request_sha256 == winner_receipt
                && gc_pending_media_set_ids == vec!["loser-set-0", "loser-set-1"]
        ));

        let active = list_obligations(&db, "owner-1", QueueMediaObligationState::Active)
            .unwrap()
            .into_iter()
            .map(|media| media.media_set_id)
            .collect::<Vec<_>>();
        assert_eq!(active, vec!["winner-set-0", "winner-set-1"]);
        let pending = list_obligations(&db, "owner-1", QueueMediaObligationState::GcPending)
            .unwrap()
            .into_iter()
            .map(|media| media.media_set_id)
            .collect::<Vec<_>>();
        assert_eq!(pending, vec!["loser-set-0", "loser-set-1"]);
        assert!(crate::generation_queue::get(&db, "loser-job-0")
            .unwrap()
            .is_none());

        let conflict_rows = media_rows("conflict", 1);
        let conflict_media = media_for_rows(&conflict_rows);
        assert!(matches!(
            insert_or_get_with_media(
                &db,
                &batch("different-opaque-receipt"),
                &conflict_rows,
                &conflict_media,
            )
            .unwrap(),
            GenerationBatchMediaInsertOutcome::Existing {
                detail,
                gc_pending_media_set_ids,
                colliding_media_set_ids,
            } if detail.batch.request_sha256 == winner_receipt
                && gc_pending_media_set_ids == vec!["conflict-set-0"]
                && colliding_media_set_ids.is_empty()
        ));
        assert!(matches!(
            insert_or_get_with_media(
                &db,
                &batch(winner_receipt),
                &loser_rows,
                &loser_media,
            )
            .unwrap(),
            GenerationBatchMediaInsertOutcome::Existing {
                detail,
                gc_pending_media_set_ids,
                colliding_media_set_ids,
                ..
            } if detail.batch.request_sha256 == winner_receipt
                && gc_pending_media_set_ids == vec!["loser-set-0", "loser-set-1"]
                && colliding_media_set_ids.is_empty()
        ));
        assert_eq!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::GcPending)
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn concurrent_media_contenders_return_the_stored_opaque_receipt_and_gc_the_loser() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        MetadataDb::open(&path).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let handles = [
            ("first", "opaque-receipt-first"),
            ("second", "opaque-receipt-second"),
        ]
        .into_iter()
        .map(|(prefix, receipt)| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let db = MetadataDb::open(&path).unwrap();
                let rows = media_rows(prefix, 1);
                let media = media_for_rows(&rows);
                barrier.wait();
                insert_or_get_with_media(&db, &batch(receipt), &rows, &media).unwrap()
            })
        })
        .collect::<Vec<_>>();
        let outcomes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        let inserted = outcomes
            .iter()
            .find_map(|outcome| match outcome {
                GenerationBatchMediaInsertOutcome::Inserted(detail) => Some(detail),
                _ => None,
            })
            .expect("one contender inserts");
        let (existing, gc_pending, collisions) = outcomes
            .iter()
            .find_map(|outcome| match outcome {
                GenerationBatchMediaInsertOutcome::Existing {
                    detail,
                    gc_pending_media_set_ids,
                    colliding_media_set_ids,
                } => Some((detail, gc_pending_media_set_ids, colliding_media_set_ids)),
                _ => None,
            })
            .expect("one contender observes the winner");
        assert_eq!(existing.batch.request_sha256, inserted.batch.request_sha256);
        assert_eq!(gc_pending.len(), 1);
        assert!(collisions.is_empty());

        let reopened = MetadataDb::open(&path).unwrap();
        assert_eq!(
            list_obligations(&reopened, "owner-1", QueueMediaObligationState::Active,)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            list_obligations(&reopened, "owner-1", QueueMediaObligationState::GcPending,)
                .unwrap()
                .into_iter()
                .map(|obligation| obligation.media_set_id)
                .collect::<Vec<_>>(),
            *gc_pending
        );
        assert_eq!(
            crate::generation_queue::list_all(&reopened, "owner-1")
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn media_batch_failure_rolls_back_queue_rows_and_obligations() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut broken_rows = media_rows("broken", 2);
        broken_rows[1].0.batch_index = broken_rows[0].0.batch_index;
        let media = media_for_rows(&broken_rows);

        assert!(insert_or_get_with_media(&db, &batch("hash"), &broken_rows, &media).is_err());
        assert!(get_by_client(&db, "owner-1", "client-1").unwrap().is_none());
        assert!(crate::generation_queue::list_all(&db, "owner-1")
            .unwrap()
            .is_empty());
        assert!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::Active)
                .unwrap()
                .is_empty()
        );
        assert!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::GcPending)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn loser_set_id_collision_cannot_retire_the_winner() {
        let db = MetadataDb::open_in_memory().unwrap();
        let winner_rows = media_rows("winner", 1);
        let winner_media = media_for_rows(&winner_rows);
        let winner_receipt = "opaque-winner-receipt";
        insert_or_get_with_media(&db, &batch(winner_receipt), &winner_rows, &winner_media).unwrap();

        let mut loser_rows = media_rows("loser", 2);
        loser_rows[0].1.media_set_id = Some("winner-set-0".to_string());
        let contender = vec![
            media_obligation("winner-set-0"),
            media_obligation("loser-set-1"),
        ];
        assert!(matches!(
            insert_or_get_with_media(
                &db,
                &batch("different-opaque-receipt"),
                &loser_rows,
                &contender,
            )
            .unwrap(),
            GenerationBatchMediaInsertOutcome::Existing {
                detail,
                gc_pending_media_set_ids,
                colliding_media_set_ids,
                ..
            } if detail.batch.request_sha256 == winner_receipt
                && gc_pending_media_set_ids == vec!["loser-set-1"]
                && colliding_media_set_ids == vec!["winner-set-0"]
        ));

        assert_eq!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::Active)
                .unwrap()
                .into_iter()
                .map(|media| media.media_set_id)
                .collect::<Vec<_>>(),
            vec!["winner-set-0"]
        );
        assert_eq!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::GcPending)
                .unwrap()
                .into_iter()
                .map(|media| media.media_set_id)
                .collect::<Vec<_>>(),
            vec!["loser-set-1"]
        );
        assert!(crate::generation_queue::get(&db, "winner-job-0")
            .unwrap()
            .is_some());
    }

    #[test]
    fn inserts_all_children_and_replays_same_ids() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = rows(30);
        let (first, inserted) = insert_or_get(&db, &batch("same"), &children).unwrap();
        assert!(inserted);
        assert_eq!(first.children.len(), 30);

        let (retry, inserted) = insert_or_get(&db, &batch("same"), &children).unwrap();
        assert!(!inserted);
        assert_eq!(retry, first);
        assert_eq!(
            crate::generation_queue::list_all(&db, "owner-1")
                .unwrap()
                .len(),
            30
        );
    }

    #[test]
    fn all_admitted_children_survive_database_reopen_for_queue_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let first_ids = {
            let db = MetadataDb::open(&path).unwrap();
            let children = rows(30);
            let (detail, inserted) = insert_or_get(&db, &batch("same"), &children).unwrap();
            assert!(inserted);
            detail
                .children
                .into_iter()
                .map(|child| child.job_id)
                .collect::<Vec<_>>()
        };

        let reopened = MetadataDb::open(&path).unwrap();
        let detail = get_by_client(&reopened, "owner-1", "client-1")
            .unwrap()
            .expect("batch grouping survives restart");
        assert_eq!(
            detail
                .children
                .into_iter()
                .map(|child| child.job_id)
                .collect::<Vec<_>>(),
            first_ids
        );
        assert_eq!(
            crate::generation_queue::list_all(&reopened, "owner-1")
                .unwrap()
                .into_iter()
                .map(|row| row.id)
                .collect::<Vec<_>>(),
            first_ids,
            "the ordinary durable queue reopens with every child exactly once"
        );
    }

    #[test]
    fn rejects_changed_payload_without_partial_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = rows(2);
        insert_or_get(&db, &batch("first"), &children).unwrap();
        assert!(insert_or_get(&db, &batch("changed"), &children).is_err());
        assert_eq!(
            crate::generation_queue::list_all(&db, "owner-1")
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn concurrent_duplicate_client_admission_returns_one_existing_batch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        MetadataDb::open(&path).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let handles = (0..2)
            .map(|_| {
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let db = MetadataDb::open(&path).unwrap();
                    let children = rows(2);
                    barrier.wait();
                    insert_or_get(&db, &batch("same"), &children).unwrap()
                })
            })
            .collect::<Vec<_>>();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|(_, inserted)| *inserted).count(), 1);
        assert_eq!(results[0].0, results[1].0);
        let reopened = MetadataDb::open(&path).unwrap();
        assert_eq!(
            get_by_client(&reopened, "owner-1", "client-1")
                .unwrap()
                .unwrap()
                .children
                .len(),
            2
        );
    }

    #[test]
    fn child_insert_failure_rolls_back_parent_and_every_child() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut children = rows(2);
        children[1].1.id = children[0].1.id.clone();
        assert!(insert_or_get(&db, &batch("same"), &children).is_err());
        assert!(get(&db, "owner-1", "batch-1").unwrap().is_none());
        assert!(crate::generation_queue::list_all(&db, "owner-1")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn separate_batch_admissions_preserve_existing_created_at_rowid_order() {
        let db = MetadataDb::open_in_memory().unwrap();
        let first = rows(2);
        insert_or_get(&db, &batch("first"), &first).unwrap();

        let mut second = rows(2);
        for (index, (child, queue)) in second.iter_mut().enumerate() {
            child.batch_id = "batch-2".into();
            child.job_id = format!("later-{index}");
            queue.id = child.job_id.clone();
        }
        let second_batch = GenerationBatchRow {
            id: "batch-2".into(),
            client_batch_id: "client-2".into(),
            owner_uuid: "owner-1".into(),
            request_sha256: "second".into(),
            created_at_ms: 1,
        };
        insert_or_get(&db, &second_batch, &second).unwrap();

        let mut claimed = Vec::new();
        for index in 0..4 {
            claimed.push(
                crate::generation_queue::claim_next(&db, "owner-1", &format!("claim-{index}"), 10)
                    .unwrap()
                    .unwrap()
                    .row
                    .id,
            );
        }
        assert_eq!(claimed, vec!["job-0", "job-1", "later-0", "later-1"]);
    }

    #[test]
    fn durable_bulk_lookup_is_owner_scoped_ordered_and_projects_terminal_data() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("first"), &rows(1)).unwrap();
        finish_unclaimed_queued(
            &db,
            "owner-1",
            "job-0",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Complete,
                error: None,
                terminal_error_json: None,
                result_json: Some(
                    r#"{"filename":"finished.png","original_filename":"original.png"}"#,
                ),
                completed_at_ms: 9,
            },
        )
        .unwrap();

        let mut second_rows = rows(1);
        second_rows[0].0.batch_id = "batch-2".into();
        second_rows[0].0.job_id = "job-2".into();
        second_rows[0].1.id = "job-2".into();
        let second = GenerationBatchRow {
            id: "batch-2".into(),
            client_batch_id: "client-2".into(),
            owner_uuid: "owner-1".into(),
            request_sha256: "second".into(),
            created_at_ms: 2,
        };
        insert_or_get(&db, &second, &second_rows).unwrap();

        let lookup = lookup_durable(
            &db,
            "owner-1",
            &[
                "client-2".into(),
                "missing-client".into(),
                "client-2".into(),
            ],
            &["batch-1".into(), "missing-batch".into(), "batch-2".into()],
        )
        .unwrap();
        assert_eq!(
            lookup
                .batches
                .iter()
                .map(|detail| detail.batch.id.as_str())
                .collect::<Vec<_>>(),
            ["batch-2", "batch-1"]
        );
        assert_eq!(lookup.missing_client_batch_ids, ["missing-client"]);
        assert_eq!(lookup.missing_batch_ids, ["missing-batch"]);
        let completed = &lookup.batches[1].children[0];
        assert_eq!(completed.completed_at_ms, Some(9));
        assert_eq!(
            completed.result_json.as_deref(),
            Some(r#"{"filename":"finished.png","original_filename":"original.png"}"#)
        );

        assert!(lookup_durable(
            &db,
            "other-owner",
            &["client-1".into()],
            &["batch-1".into()]
        )
        .unwrap()
        .batches
        .is_empty());
    }

    #[test]
    fn terminal_child_update_and_claimed_queue_delete_are_atomic() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = rows(1);
        insert_or_get(&db, &batch("same"), &children).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "claim-1", 2)
            .unwrap()
            .unwrap();
        crate::generation_queue::mark_dispatched_claimed(&db, "job-0", "claim-1", 3).unwrap();

        db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER reject_terminal_delete
                 BEFORE DELETE ON generation_queue
                 BEGIN SELECT RAISE(ABORT, 'injected delete failure'); END;",
            )?;
            Ok(())
        })
        .unwrap();
        let terminal = GenerationBatchTerminal {
            state: GenerationBatchTerminalState::Failed,
            error: Some("render failed"),
            terminal_error_json: Some(r#"{"code":"render_failed"}"#),
            result_json: Some(r#"{"filename":"partial.png"}"#),
            completed_at_ms: 4,
        };
        assert!(finish_claimed(
            &db,
            "job-0",
            "claim-1",
            crate::generation_queue::QueueRowState::Running,
            terminal,
        )
        .is_err());
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "accepted"
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_some());

        db.with_conn(|conn| {
            conn.execute_batch("DROP TRIGGER reject_terminal_delete")?;
            Ok(())
        })
        .unwrap();
        let committed = finish_claimed(
            &db,
            "job-0",
            "claim-1",
            crate::generation_queue::QueueRowState::Running,
            terminal,
        )
        .unwrap();
        assert!(committed.queue_deleted);
        assert!(committed.batch_child_updated);
        let child = &get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0];
        assert_eq!(child.state, "failed");
        assert_eq!(child.error.as_deref(), Some("render failed"));
        let durable = get_durable(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert_eq!(
            durable.children[0].terminal_error_json.as_deref(),
            Some(r#"{"code":"render_failed"}"#)
        );
        assert_eq!(durable.children[0].completed_at_ms, Some(4));
        assert_eq!(
            durable.children[0].result_json.as_deref(),
            Some(r#"{"filename":"partial.png"}"#)
        );
        assert_eq!(
            get_durable_by_client(&db, "owner-1", "client-1")
                .unwrap()
                .unwrap(),
            durable
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
    }

    #[test]
    fn claimed_singleton_can_use_the_same_fenced_terminal_primitive() {
        let db = MetadataDb::open_in_memory().unwrap();
        crate::generation_queue::insert(&db, &rows(1).pop().unwrap().1).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "singleton", 2)
            .unwrap()
            .unwrap();

        let committed = finish_claimed(
            &db,
            "job-0",
            "singleton",
            crate::generation_queue::QueueRowState::Queued,
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("cancelled before dispatch"),
                terminal_error_json: None,
                result_json: None,
                completed_at_ms: 3,
            },
        )
        .unwrap();
        assert!(committed.queue_deleted);
        assert!(!committed.batch_child_updated);
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
    }

    #[test]
    fn stale_terminal_token_cannot_update_child_or_delete_queue_row() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "current", 2)
            .unwrap()
            .unwrap();
        let committed = finish_claimed(
            &db,
            "job-0",
            "stale",
            crate::generation_queue::QueueRowState::Queued,
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("cancelled"),
                terminal_error_json: None,
                result_json: None,
                completed_at_ms: 3,
            },
        )
        .unwrap();
        assert!(!committed.queue_deleted);
        assert!(!committed.batch_child_updated);
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "accepted"
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_some());
    }

    #[test]
    fn cancelling_an_unclaimed_child_is_atomic_and_visible_on_reconnect() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        let commit = finish_unclaimed_queued(
            &db,
            "owner-1",
            "job-0",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 9,
            },
        )
        .unwrap();
        assert!(commit.queue_deleted);
        assert!(commit.batch_child_updated);
        let detail = get_durable(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert_eq!(detail.children[0].state, "cancelled");
        assert_eq!(detail.children[0].completed_at_ms, Some(9));
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
    }

    #[test]
    fn held_child_cancellation_is_terminal_and_removes_execution_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::hold(&db, "job-0", "operator review", 5).unwrap();
        set_child_state(&db, "job-0", "held", Some("operator review"), 5).unwrap();

        let outcome = cancel_owned(
            &db,
            "owner-1",
            "job-0",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 9,
            },
        )
        .unwrap();

        assert_eq!(outcome, OwnedCancellation::Settled);
        let detail = get_durable(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert_eq!(detail.children[0].state, "cancelled");
        assert_eq!(detail.children[0].completed_at_ms, Some(9));
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
    }

    #[test]
    fn restart_pause_and_resume_move_batch_child_with_queue_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        let before_revision: i64 = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT revision FROM generation_batch_children WHERE job_id = 'job-0'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();

        crate::generation_queue::recover_runtime_claims(&db, "owner-1", 5).unwrap();
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "paused"
        );
        let paused_revision: i64 = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT revision FROM generation_batch_children WHERE job_id = 'job-0'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(paused_revision, before_revision + 1);

        crate::generation_queue::resume_all_paused(&db, "owner-1", 6).unwrap();
        let detail = get(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert_eq!(detail.children[0].state, "accepted");
        let resumed_revision: i64 = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT revision FROM generation_batch_children WHERE job_id = 'job-0'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(resumed_revision, paused_revision + 1);
    }

    #[test]
    fn paused_child_cancellation_is_terminal_and_removes_execution_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::recover_runtime_claims(&db, "owner-1", 5).unwrap();

        let outcome = cancel_owned(
            &db,
            "owner-1",
            "job-0",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 9,
            },
        )
        .unwrap();

        assert_eq!(outcome, OwnedCancellation::Settled);
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "cancelled"
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
    }

    #[test]
    fn cancellation_cannot_cross_queue_owner_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();

        let outcome = cancel_owned(
            &db,
            "owner-2",
            "job-0",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 9,
            },
        )
        .unwrap();

        assert_eq!(outcome, OwnedCancellation::NotOwned);
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "accepted"
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_some());
    }

    #[test]
    fn cancellation_revokes_claimed_queued_work_before_completion() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "worker", 2)
            .unwrap()
            .unwrap();

        let cancelled = cancel_owned(
            &db,
            "owner-1",
            "job-0",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 3,
            },
        )
        .unwrap();
        assert_eq!(cancelled, OwnedCancellation::Settled);
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
        assert!(
            !set_child_state(&db, "job-0", "running", None, 3).unwrap(),
            "a late nonterminal mirror must not erase cancellation"
        );

        let committed = finish_claimed(
            &db,
            "job-0",
            "worker",
            crate::generation_queue::QueueRowState::Queued,
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Complete,
                error: None,
                terminal_error_json: None,
                result_json: Some(r#"{"filename":"too-late.png"}"#),
                completed_at_ms: 4,
            },
        );

        let committed = committed.unwrap();
        assert!(!committed.queue_deleted);
        assert!(!committed.batch_child_updated);
        let child = &get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0];
        assert_eq!(child.state, "cancelled");
        assert_eq!(child.error.as_deref(), Some("Cancelled"));
        assert!(child.result_json.is_none());
    }

    #[test]
    fn revoked_claim_cannot_erase_or_resurrect_cancellation() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "worker", 2)
            .unwrap()
            .unwrap();
        assert_eq!(
            cancel_owned(
                &db,
                "owner-1",
                "job-0",
                GenerationBatchTerminal {
                    state: GenerationBatchTerminalState::Cancelled,
                    error: Some("Cancelled"),
                    terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                    result_json: None,
                    completed_at_ms: 3,
                },
            )
            .unwrap(),
            OwnedCancellation::Settled
        );
        assert!(!crate::generation_queue::release_claim(&db, "job-0", "worker", 4).unwrap());

        assert!(!restore_child_after_retain(&db, "owner-1", "job-0", 4).unwrap());
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "cancelled"
        );

        assert_eq!(
            cancel_owned(
                &db,
                "owner-1",
                "job-0",
                GenerationBatchTerminal {
                    state: GenerationBatchTerminalState::Cancelled,
                    error: Some("Cancelled"),
                    terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                    result_json: None,
                    completed_at_ms: 5,
                },
            )
            .unwrap(),
            OwnedCancellation::NotOwned
        );
        assert!(!restore_child_after_retain(&db, "owner-1", "job-0", 6).unwrap());
        assert_eq!(
            get(&db, "owner-1", "batch-1").unwrap().unwrap().children[0].state,
            "cancelled"
        );
    }

    #[test]
    fn cancellation_linearizes_before_a_claimed_hold() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "worker", 2)
            .unwrap()
            .unwrap();
        assert_eq!(
            cancel_owned(
                &db,
                "owner-1",
                "job-0",
                GenerationBatchTerminal {
                    state: GenerationBatchTerminalState::Cancelled,
                    error: Some("Cancelled"),
                    terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                    result_json: None,
                    completed_at_ms: 3,
                },
            )
            .unwrap(),
            OwnedCancellation::Settled
        );

        assert_eq!(
            hold_owned(
                &db,
                "owner-1",
                "job-0",
                Some("worker"),
                "unusable output",
                None,
                false,
                4,
            )
            .unwrap(),
            OwnedHold::Cancelled
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
        let child = &get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0];
        assert_eq!(child.state, "cancelled");
        assert_eq!(child.completed_at_ms, Some(3));
    }

    #[test]
    fn bulk_cancel_settles_only_unclaimed_children_and_keeps_claimed_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(3)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "claimed", 2)
            .unwrap()
            .unwrap();
        let count = finish_all_unclaimed_queued(
            &db,
            "owner-1",
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 9,
            },
        )
        .unwrap();
        assert_eq!(count, 2);
        let detail = get(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert_eq!(detail.children[0].state, "accepted");
        assert_eq!(detail.children[1].state, "cancelled");
        assert_eq!(detail.children[2].state, "cancelled");
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_some());
        assert!(crate::generation_queue::get(&db, "job-1")
            .unwrap()
            .is_none());
        assert!(crate::generation_queue::get(&db, "job-2")
            .unwrap()
            .is_none());
    }

    #[test]
    fn bulk_cancel_reports_only_durable_rows_not_already_counted_live() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(5)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "claimed", 2)
            .unwrap()
            .unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "UPDATE generation_batch_children
                    SET state = 'cancelling', error = 'Cancelled', updated_at_ms = 2
                  WHERE job_id = 'job-0'",
                [],
            )?;
            Ok(())
        })
        .unwrap();

        for index in 0..3 {
            let mut legacy = rows(1).pop().unwrap().1;
            legacy.id = format!("legacy-{index}");
            legacy.created_at_ms = 10 + index;
            crate::generation_queue::insert(&db, &legacy).unwrap();
        }
        let mut running = rows(1).pop().unwrap().1;
        running.id = "legacy-running".to_string();
        running.created_at_ms = 20;
        crate::generation_queue::insert(&db, &running).unwrap();
        crate::generation_queue::mark_dispatched(&db, "legacy-running", 21).unwrap();

        let additional = cancel_all_queued(
            &db,
            "owner-1",
            &["job-0".to_string(), "legacy-0".to_string()],
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 30,
            },
        )
        .unwrap();

        assert_eq!(
            additional, 6,
            "all eight durable queued rows, including legacy cancelling work, minus two live rows"
        );
        let detail = get_durable(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert!(detail
            .children
            .iter()
            .all(|child| child.state == "cancelled"));
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_none());
        assert!(crate::generation_queue::get(&db, "legacy-running")
            .unwrap()
            .is_some());
        assert!(crate::generation_queue::get(&db, "legacy-0")
            .unwrap()
            .is_none());

        assert_eq!(
            cancel_all_queued(
                &db,
                "owner-1",
                &[],
                GenerationBatchTerminal {
                    state: GenerationBatchTerminalState::Cancelled,
                    error: Some("Cancelled"),
                    terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                    result_json: None,
                    completed_at_ms: 31,
                },
            )
            .unwrap(),
            0,
            "an already-requested claimed cancellation is not counted twice"
        );
    }

    #[test]
    fn bulk_cancel_settles_a_claimed_child_immediately_and_retires_media() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = media_rows("cancel", 1);
        let obligations = media_for_rows(&children);
        insert_or_get_with_media(&db, &batch("receipt"), &children, &obligations).unwrap();
        crate::generation_queue::claim_by_id(&db, "owner-1", "cancel-job-0", "live-claim", 2)
            .unwrap()
            .unwrap();
        let cancelled = GenerationBatchTerminal {
            state: GenerationBatchTerminalState::Cancelled,
            error: Some("Cancelled"),
            terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
            result_json: None,
            completed_at_ms: 3,
        };

        assert_eq!(
            cancel_all_queued(&db, "owner-1", &["cancel-job-0".into()], cancelled).unwrap(),
            0
        );
        assert_eq!(
            get_durable(&db, "owner-1", "batch-1")
                .unwrap()
                .unwrap()
                .children[0]
                .state,
            "cancelled"
        );
        assert!(
            !crate::generation_queue::release_claim(&db, "cancel-job-0", "live-claim", 4).unwrap()
        );

        assert_eq!(
            cancel_all_queued(
                &db,
                "owner-1",
                &[],
                GenerationBatchTerminal {
                    completed_at_ms: 5,
                    ..cancelled
                }
            )
            .unwrap(),
            0,
            "the repeated call converges work already counted by the first cancellation"
        );
        let child = &get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0];
        assert_eq!(child.state, "cancelled");
        assert_eq!(child.completed_at_ms, Some(3));
        assert!(crate::generation_queue::get(&db, "cancel-job-0")
            .unwrap()
            .is_none());
        assert!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::Active)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::GcPending)
                .unwrap()
                .into_iter()
                .map(|row| row.media_set_id)
                .collect::<Vec<_>>(),
            ["cancel-set-0"]
        );
    }

    #[test]
    fn bulk_cancel_rejects_terminal_child_with_live_queue_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = media_rows("inconsistent", 1);
        let obligations = media_for_rows(&children);
        insert_or_get_with_media(&db, &batch("receipt"), &children, &obligations).unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "UPDATE generation_batch_children
                    SET state = 'complete', completed_at_ms = 2
                  WHERE job_id = 'inconsistent-job-0'",
                [],
            )?;
            Ok(())
        })
        .unwrap();

        let error = cancel_all_queued(
            &db,
            "owner-1",
            &[],
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: 3,
            },
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("retained inconsistent batch child state complete"));
        assert!(crate::generation_queue::get(&db, "inconsistent-job-0")
            .unwrap()
            .is_some());
        assert_eq!(
            list_obligations(&db, "owner-1", QueueMediaObligationState::Active)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn startup_media_hold_updates_queue_and_child_atomically() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = media_rows("hold", 2);
        let obligations = media_for_rows(&children);
        insert_or_get_with_media(&db, &batch("receipt"), &children, &obligations).unwrap();
        crate::generation_queue::claim_by_id(&db, "owner-1", "hold-job-1", "worker", 2)
            .unwrap()
            .unwrap();
        crate::generation_queue::mark_dispatched_claimed(&db, "hold-job-1", "worker", 3).unwrap();
        set_child_state(&db, "hold-job-1", "running", None, 3).unwrap();

        assert_eq!(
            crate::generation_queue::hold_media_jobs(
                &db,
                "owner-1",
                &["hold-job-0".into(), "hold-job-1".into()],
                "media invalid",
                4,
            )
            .unwrap(),
            2
        );
        assert!(crate::generation_queue::list_all(&db, "owner-1")
            .unwrap()
            .iter()
            .all(|row| row.state == QueueRowState::Held));
        assert!(get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children
            .iter()
            .all(|child| child.state == "held" && child.error.as_deref() == Some("media invalid")));
    }

    #[test]
    fn startup_media_hold_rolls_back_queue_when_child_update_fails() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = media_rows("rollback", 1);
        let obligations = media_for_rows(&children);
        insert_or_get_with_media(&db, &batch("receipt"), &children, &obligations).unwrap();
        db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TRIGGER reject_media_child_hold
                 BEFORE UPDATE ON generation_batch_children
                 BEGIN SELECT RAISE(ABORT, 'injected child hold failure'); END;",
            )?;
            Ok(())
        })
        .unwrap();

        assert!(crate::generation_queue::hold_media_jobs(
            &db,
            "owner-1",
            &["rollback-job-0".into()],
            "media invalid",
            4,
        )
        .is_err());
        assert_eq!(
            crate::generation_queue::get(&db, "rollback-job-0")
                .unwrap()
                .unwrap()
                .state,
            QueueRowState::Queued
        );
        assert_eq!(
            get_durable(&db, "owner-1", "batch-1")
                .unwrap()
                .unwrap()
                .children[0]
                .state,
            "accepted"
        );
    }

    #[test]
    fn restore_after_retain_requires_exact_unclaimed_queue_and_never_overwrites_hold() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        crate::generation_queue::claim_next(&db, "owner-1", "worker", 2)
            .unwrap()
            .unwrap();
        set_child_state(&db, "job-0", "running", None, 3).unwrap();
        assert!(!restore_child_after_retain(&db, "owner-1", "job-0", 4).unwrap());

        assert!(crate::generation_queue::release_claim(&db, "job-0", "worker", 5).unwrap());
        assert!(restore_child_after_retain(&db, "owner-1", "job-0", 5).unwrap());
        set_child_state(&db, "job-0", "running", None, 6).unwrap();
        assert_eq!(
            hold_owned(
                &db,
                "owner-1",
                "job-0",
                None,
                "operator hold",
                None,
                false,
                7
            )
            .unwrap(),
            OwnedHold::Held
        );
        assert!(!restore_child_after_retain(&db, "owner-1", "job-0", 8).unwrap());
        let child = &get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0];
        assert_eq!(child.state, "held");
        assert_eq!(child.error.as_deref(), Some("operator hold"));
    }

    #[test]
    fn explicit_retry_restores_only_retryable_held_work_atomically() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(2)).unwrap();
        let authority = |job_id: &str| mold_core::GenerationRetryRequest {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: job_id.into(),
        };

        assert_eq!(
            hold_owned(
                &db,
                "owner-1",
                "job-0",
                None,
                "dependency failed",
                None,
                true,
                2
            )
            .unwrap(),
            OwnedHold::Held
        );
        assert_eq!(
            hold_owned(
                &db,
                "owner-1",
                "job-1",
                None,
                "corrupt media",
                None,
                false,
                2
            )
            .unwrap(),
            OwnedHold::Held
        );
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority("job-1"), 3).unwrap(),
            OwnedRetry::NotRetryable
        );
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority("job-0"), 3).unwrap(),
            OwnedRetry::Retried
        );

        let queue = crate::generation_queue::get(&db, "job-0").unwrap().unwrap();
        assert_eq!(queue.state, QueueRowState::Queued);
        assert_eq!(queue.held_reason, None);
        let detail = get_durable(&db, "owner-1", "batch-1").unwrap().unwrap();
        assert_eq!(detail.children[0].state, "accepted");
        assert_eq!(detail.children[0].error, None);
        assert_eq!(detail.children[1].state, "held");
    }

    /// A retry is the first transition that moves a job BACKWARD through the
    /// browser reducer's rank ordering (`generationLifecycle.ts`
    /// FORWARD_PHASE_RANK: held is 2, queued is 1). That reducer accepts a
    /// backward move only when the snapshot is strictly newer, and since
    /// `revision` is never supplied on the wire the comparison falls through to
    /// this timestamp. Retrying in the same millisecond as the hold must still
    /// advance it, or the client silently keeps rendering Held for a job the
    /// server has queued.
    #[test]
    fn a_retry_advances_its_timestamps_even_within_one_millisecond() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        let authority = mold_core::GenerationRetryRequest {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        };

        // Hold and retry at the SAME now_ms — the collision case.
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "dependency failed",
            None,
            true,
            7,
        )
        .unwrap();
        let held_queue = crate::generation_queue::get(&db, "job-0").unwrap().unwrap();
        let held_child_ms = get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0]
            .updated_at_ms;

        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 7).unwrap(),
            OwnedRetry::Retried
        );

        let queued = crate::generation_queue::get(&db, "job-0").unwrap().unwrap();
        assert!(
            queued.updated_at_ms > held_queue.updated_at_ms,
            "queue row must advance: held {} -> queued {}",
            held_queue.updated_at_ms,
            queued.updated_at_ms
        );
        let child_ms = get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0]
            .updated_at_ms;
        assert!(
            child_ms > held_child_ms,
            "child row must advance: held {held_child_ms} -> accepted {child_ms}"
        );
    }

    // ── Held-row retention ────────────────────────────────────────────────
    // A hold is durable so a human can return to it. One nobody returns to
    // pins a queue row and its encrypted media forever.

    #[test]
    fn retention_lists_only_held_rows_past_their_window() {
        use crate::generation_queue::{expired_held, held_count};
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(3)).unwrap();
        let day = 86_400_000_i64;
        let now = 40 * day;

        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "old",
            None,
            true,
            now - 31 * day,
        )
        .unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-1",
            None,
            "recent",
            None,
            true,
            now - 3 * day,
        )
        .unwrap();
        // job-2 stays queued.

        let expired = expired_held(&db, "owner-1", 30, now).unwrap();
        assert_eq!(
            expired
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec!["job-0"],
            "only the hold past its window expires; a queued row is never swept"
        );
        assert_eq!(held_count(&db, "owner-1").unwrap(), 2);

        assert!(
            expired_held(&db, "owner-1", 0, now).unwrap().is_empty(),
            "0 keeps held rows forever, exactly as gallery trash retention does"
        );
        assert!(
            expired_held(&db, "other-owner", 30, now)
                .unwrap()
                .is_empty(),
            "another installation's holds are not ours to purge"
        );
    }

    #[test]
    fn purging_a_held_row_settles_its_child_instead_of_dropping_it() {
        use crate::generation_queue::purge_held;
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "model went missing",
            None,
            true,
            5,
        )
        .unwrap();

        assert!(purge_held(&db, "owner-1", "job-0", 9).unwrap());
        assert!(
            crate::generation_queue::get(&db, "job-0")
                .unwrap()
                .is_none(),
            "the queue row is gone, which is what retires its media obligation"
        );

        // The batch child survives as a terminal summary: a reconnecting
        // client reads it after the queue row is gone, and deleting it would
        // report the print as never admitted.
        let child = &get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0];
        assert_eq!(child.state, "failed");
        assert!(
            child
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("model went missing"),
            "the settled child must keep the reason the work was held: {:?}",
            child.error
        );
        assert!(child.completed_at_ms.is_some());
        assert!(
            child.revision > 0,
            "settlement is an authoritative transition"
        );
    }

    const DAY_MS: i64 = 86_400_000;

    /// One batch under `owner-1` with the given child job ids, admitted with
    /// its queue rows exactly as `insert_or_get` does.
    fn admit_batch(db: &MetadataDb, batch_id: &str, job_ids: &[&str]) {
        let batch = GenerationBatchRow {
            id: batch_id.into(),
            client_batch_id: format!("client-{batch_id}"),
            owner_uuid: "owner-1".into(),
            request_sha256: format!("sha-{batch_id}"),
            created_at_ms: 1,
        };
        let rows = job_ids
            .iter()
            .enumerate()
            .map(|(index, job_id)| {
                let (mut child, mut queue) = rows(1).pop().unwrap();
                child.batch_id = batch_id.into();
                child.job_id = (*job_id).into();
                child.batch_index = index as u32 + 1;
                queue.id = (*job_id).into();
                (child, queue)
            })
            .collect::<Vec<_>>();
        insert_or_get(db, &batch, &rows).unwrap();
    }

    /// Settle one unclaimed queued child terminally at `at_ms`, deleting its
    /// queue row in the same transaction exactly as the worker does.
    fn settle(db: &MetadataDb, job_id: &str, state: GenerationBatchTerminalState, at_ms: i64) {
        let commit = finish_unclaimed_queued(
            db,
            "owner-1",
            job_id,
            GenerationBatchTerminal {
                state,
                error: None,
                terminal_error_json: None,
                result_json: None,
                completed_at_ms: at_ms,
            },
        )
        .unwrap();
        assert!(commit.queue_deleted && commit.batch_child_updated);
    }

    fn child_count(db: &MetadataDb, batch_id: &str) -> i64 {
        db.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(*) FROM generation_batch_children WHERE batch_id = ?1",
                params![batch_id],
                |row| row.get(0),
            )
            .map_err(Into::into)
        })
        .unwrap()
    }

    #[test]
    fn expired_settled_lists_only_batches_whose_every_child_is_terminal_and_old() {
        let db = MetadataDb::open_in_memory().unwrap();
        let now = 100 * DAY_MS;
        // Fully settled, newest settlement 35 days old: eligible.
        admit_batch(&db, "old", &["old-0", "old-1"]);
        settle(
            &db,
            "old-0",
            GenerationBatchTerminalState::Complete,
            now - 40 * DAY_MS,
        );
        settle(
            &db,
            "old-1",
            GenerationBatchTerminalState::Failed,
            now - 35 * DAY_MS,
        );
        // One child still queued (its queue row survives): never eligible.
        admit_batch(&db, "open", &["open-0", "open-1"]);
        settle(
            &db,
            "open-0",
            GenerationBatchTerminalState::Complete,
            now - 40 * DAY_MS,
        );
        // Fully settled but the newest settlement is two days old.
        admit_batch(&db, "fresh", &["fresh-0", "fresh-1"]);
        settle(
            &db,
            "fresh-0",
            GenerationBatchTerminalState::Cancelled,
            now - 40 * DAY_MS,
        );
        settle(
            &db,
            "fresh-1",
            GenerationBatchTerminalState::Complete,
            now - 2 * DAY_MS,
        );
        // A held child is not terminal, however old the hold is.
        admit_batch(&db, "parked", &["parked-0", "parked-1"]);
        settle(
            &db,
            "parked-0",
            GenerationBatchTerminalState::Complete,
            now - 40 * DAY_MS,
        );
        hold_owned(
            &db,
            "owner-1",
            "parked-1",
            None,
            "model went missing",
            None,
            true,
            now - 40 * DAY_MS,
        )
        .unwrap();

        let expired = expired_settled(&db, "owner-1", 30, now).unwrap();
        assert_eq!(
            expired,
            vec![ExpiredSettledBatch {
                id: "old".into(),
                settled_at_ms: now - 35 * DAY_MS,
            }],
            "age is the NEWEST child settlement; a queued, fresh, or held child keeps the batch"
        );
        assert_eq!(
            settled_count(&db, "owner-1").unwrap(),
            2,
            "`old` and `fresh` are settled receipts; `open` and `parked` still own work"
        );
    }

    #[test]
    fn expired_settled_is_scoped_to_the_owner_and_off_at_zero() {
        let db = MetadataDb::open_in_memory().unwrap();
        let now = 100 * DAY_MS;
        admit_batch(&db, "old", &["old-0"]);
        settle(
            &db,
            "old-0",
            GenerationBatchTerminalState::Complete,
            now - 40 * DAY_MS,
        );

        assert_eq!(
            expired_settled(&db, "owner-1", 30, now)
                .unwrap()
                .iter()
                .map(|batch| batch.id.as_str())
                .collect::<Vec<_>>(),
            ["old"]
        );
        assert!(
            expired_settled(&db, "owner-1", 0, now).unwrap().is_empty(),
            "0 keeps settled summaries forever, exactly as it keeps held rows"
        );
        assert!(
            expired_settled(&db, "other-owner", 30, now)
                .unwrap()
                .is_empty(),
            "another installation's receipts are not ours to purge"
        );
        assert_eq!(settled_count(&db, "other-owner").unwrap(), 0);
        assert!(
            !purge_settled(&db, "other-owner", "old", 30, now).unwrap(),
            "nor ours to purge by id"
        );
        assert!(get_durable(&db, "owner-1", "old").unwrap().is_some());
    }

    #[test]
    fn purge_settled_cascades_children() {
        let db = MetadataDb::open_in_memory().unwrap();
        let now = 100 * DAY_MS;
        admit_batch(&db, "old", &["old-0", "old-1"]);
        settle(
            &db,
            "old-0",
            GenerationBatchTerminalState::Complete,
            now - 40 * DAY_MS,
        );
        settle(
            &db,
            "old-1",
            GenerationBatchTerminalState::Failed,
            now - 40 * DAY_MS,
        );
        assert_eq!(child_count(&db, "old"), 2);

        assert!(purge_settled(&db, "owner-1", "old", 30, now).unwrap());
        assert!(get_durable(&db, "owner-1", "old").unwrap().is_none());
        assert_eq!(
            child_count(&db, "old"),
            0,
            "the child summaries go with their batch"
        );
        assert_eq!(settled_count(&db, "owner-1").unwrap(), 0);
        assert!(
            !purge_settled(&db, "owner-1", "old", 30, now).unwrap(),
            "a second purge finds nothing"
        );
    }

    /// The listing is a snapshot; the purge re-checks inside its transaction
    /// so an explicit retry that landed in between wins over retention.
    #[test]
    fn purge_settled_refuses_a_batch_whose_child_was_re_queued() {
        let db = MetadataDb::open_in_memory().unwrap();
        let now = 100 * DAY_MS;
        admit_batch(&db, "old", &["old-0", "old-1"]);
        settle(
            &db,
            "old-0",
            GenerationBatchTerminalState::Complete,
            now - 40 * DAY_MS,
        );
        settle(
            &db,
            "old-1",
            GenerationBatchTerminalState::Failed,
            now - 40 * DAY_MS,
        );
        let listed = expired_settled(&db, "owner-1", 30, now).unwrap();
        assert_eq!(listed.len(), 1);

        // Between the listing and the purge, a child is back in the queue.
        let (_, mut queue) = rows(1).pop().unwrap();
        queue.id = "old-1".into();
        generation_queue::insert(&db, &queue).unwrap();
        assert!(
            !purge_settled(&db, "owner-1", "old", 30, now).unwrap(),
            "a surviving queue row keeps the batch"
        );
        assert_eq!(child_count(&db, "old"), 2);
        assert!(expired_settled(&db, "owner-1", 30, now).unwrap().is_empty());

        // The queue row is gone again but the child is no longer terminal.
        db.with_conn(|conn| {
            conn.execute("DELETE FROM generation_queue WHERE id = 'old-1'", [])?;
            conn.execute(
                "UPDATE generation_batch_children SET state = 'queued' WHERE job_id = 'old-1'",
                [],
            )?;
            Ok(())
        })
        .unwrap();
        assert!(
            !purge_settled(&db, "owner-1", "old", 30, now).unwrap(),
            "a non-terminal child keeps the batch"
        );
        // And retention must still hold at purge time, not only at listing.
        db.with_conn(|conn| {
            conn.execute(
                "UPDATE generation_batch_children SET state = 'failed', updated_at_ms = ?1
                  WHERE job_id = 'old-1'",
                params![now - DAY_MS],
            )?;
            Ok(())
        })
        .unwrap();
        assert!(
            !purge_settled(&db, "owner-1", "old", 30, now).unwrap(),
            "a fresh settlement keeps the batch"
        );
        assert!(
            !purge_settled(&db, "owner-1", "old", 0, now).unwrap(),
            "0 keeps forever at the purge too"
        );
        assert_eq!(child_count(&db, "old"), 2);
    }

    /// A hold persists its typed cause beside the sentence, and a retry
    /// clears both: the code is what a client acts on (a `MODEL_NOT_FOUND`
    /// hold gets the pull-and-resume offer), so it must be exactly as
    /// current as the hold it explains.
    #[test]
    fn a_hold_persists_its_code_and_a_retry_clears_it() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "model 'x' is not downloaded",
            Some("MODEL_NOT_FOUND"),
            true,
            5,
        )
        .unwrap();
        let held = get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children
            .remove(0);
        assert_eq!(held.state, "held");
        assert_eq!(held.error_code.as_deref(), Some("MODEL_NOT_FOUND"));
        assert_eq!(held.error.as_deref(), Some("model 'x' is not downloaded"));

        let authority = mold_core::GenerationRetryRequest {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        };
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 6).unwrap(),
            OwnedRetry::Retried
        );
        let retried = get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children
            .remove(0);
        assert_eq!(retried.state, "accepted");
        assert_eq!(retried.error_code, None);
        assert_eq!(retried.error, None);
    }

    #[test]
    fn a_retried_row_wins_the_race_against_its_own_expiry() {
        // The sweeper lists, then purges. A retry landing in between is a
        // human's explicit decision and outranks retention.
        use crate::generation_queue::purge_held;
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "dependency failed",
            None,
            true,
            5,
        )
        .unwrap();
        let authority = mold_core::GenerationRetryRequest {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        };
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 6).unwrap(),
            OwnedRetry::Retried
        );

        assert!(
            !purge_held(&db, "owner-1", "job-0", 9).unwrap(),
            "a row that is no longer held must not be purged"
        );
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_some());
    }

    #[test]
    fn purging_never_reaches_another_installations_row() {
        use crate::generation_queue::purge_held;
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(&db, "owner-1", "job-0", None, "held", None, true, 5).unwrap();
        assert!(!purge_held(&db, "other-owner", "job-0", 9).unwrap());
        assert!(crate::generation_queue::get(&db, "job-0")
            .unwrap()
            .is_some());
    }

    #[test]
    fn every_state_transition_advances_the_child_revision() {
        // The revision is the ordering token clients compare. A transition
        // that does not advance it is invisible to a reducer whose snapshot
        // landed in the same millisecond, which is exactly the retry case.
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        let revision = |db: &MetadataDb| {
            get_durable(db, "owner-1", "batch-1")
                .unwrap()
                .unwrap()
                .children[0]
                .revision
        };
        let admitted = revision(&db);
        assert_eq!(admitted, 0, "an admitted child starts unversioned");

        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "dependency failed",
            None,
            true,
            7,
        )
        .unwrap();
        let held = revision(&db);
        assert!(held > admitted, "hold must advance: {admitted} -> {held}");

        let authority = mold_core::GenerationRetryRequest {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        };
        // Same millisecond as the hold: the timestamp cannot separate these.
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 7).unwrap(),
            OwnedRetry::Retried
        );
        let retried = revision(&db);
        assert!(retried > held, "retry must advance: {held} -> {retried}");
    }

    #[test]
    fn a_refused_transition_leaves_the_revision_alone() {
        // Only an authoritative transition may spend a revision. A refused
        // retry that still bumped it would look, to a reconciling client,
        // exactly like a retry that landed.
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "dependency failed",
            None,
            true,
            2,
        )
        .unwrap();
        let held = get_durable(&db, "owner-1", "batch-1")
            .unwrap()
            .unwrap()
            .children[0]
            .revision;

        let foreign = mold_core::GenerationRetryRequest {
            instance_id: "replacement".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        };
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &foreign, 3).unwrap(),
            OwnedRetry::AuthorityMismatch
        );
        assert_eq!(
            get_durable(&db, "owner-1", "batch-1")
                .unwrap()
                .unwrap()
                .children[0]
                .revision,
            held,
            "a refused retry must not spend a revision"
        );
    }

    #[test]
    fn explicit_retry_rejects_replacement_and_batch_identity_without_mutation() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "dependency failed",
            None,
            true,
            2,
        )
        .unwrap();
        let mut authority = mold_core::GenerationRetryRequest {
            instance_id: "replacement".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        };
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 3).unwrap(),
            OwnedRetry::AuthorityMismatch
        );
        authority.instance_id = "instance-1".into();
        authority.batch_id = "foreign-batch".into();
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 3).unwrap(),
            OwnedRetry::AuthorityMismatch
        );
        authority.batch_id = "batch-1".into();
        authority.client_batch_id = "foreign-client".into();
        assert_eq!(
            retry_held_owned(&db, "owner-1", "instance-1", &authority, 3).unwrap(),
            OwnedRetry::AuthorityMismatch
        );
        let queue = crate::generation_queue::get(&db, "job-0").unwrap().unwrap();
        assert_eq!(queue.state, QueueRowState::Held);
        assert!(
            get_durable(&db, "owner-1", "batch-1")
                .unwrap()
                .unwrap()
                .children[0]
                .retryable
        );
    }

    #[test]
    fn concurrent_explicit_retry_has_one_transactional_winner() {
        let db = Arc::new(MetadataDb::open_in_memory().unwrap());
        insert_or_get(&db, &batch("same"), &rows(1)).unwrap();
        hold_owned(
            &db,
            "owner-1",
            "job-0",
            None,
            "dependency failed",
            None,
            true,
            2,
        )
        .unwrap();
        let authority = Arc::new(mold_core::GenerationRetryRequest {
            instance_id: "instance-1".into(),
            batch_id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            job_id: "job-0".into(),
        });
        let barrier = Arc::new(Barrier::new(3));
        let attempts = (0..2)
            .map(|offset| {
                let db = Arc::clone(&db);
                let authority = Arc::clone(&authority);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    retry_held_owned(&db, "owner-1", "instance-1", &authority, 3 + offset).unwrap()
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let outcomes = attempts
            .into_iter()
            .map(|attempt| attempt.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| **outcome == OwnedRetry::Retried)
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| **outcome == OwnedRetry::NotHeld)
                .count(),
            1
        );
    }
}
