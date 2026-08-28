//! Retention for the durable queue's leftovers: HELD rows and SETTLED batch
//! summaries, on one horizon (`queue.held_retention_days`).
//!
//! A hold is durable on purpose: it is work parked for a human, and the
//! durable queue exists so a restart does not lose it. But nothing ever
//! reclaimed one. Every deterministically-failing request left a row that
//! `GET /api/queue` listed forever, and `hold_claimed` deliberately RETAINS
//! that row's `media_set_id` so a retry still has its source images — so the
//! encrypted media store grew with it, unbounded.
//!
//! A settled batch is the other leftover. Once every child is terminal the
//! `generation_batches` row is only a receipt for a client reconnecting
//! after a dropped stream — its media left with the queue rows — and nothing
//! ever deleted one either. Both answer "how long does the durable queue
//! remember work that can no longer run", so they share the one key and
//! `0 = forever` stays coherent. A batch's age is its last child settlement;
//! a batch with a held child therefore waits for the held pass to settle
//! that child (worst case twice the retention), which is why each tick runs
//! held first, then settled.
//!
//! This is the queue's peer of `gallery_trash`'s retention sweeper and is
//! shaped like it on purpose: one `sweep_*_once` pass reading its retention
//! fresh from the live config, one hourly background task, one manual POST
//! route per pass. What it deliberately does NOT share is the deletion
//! mechanism — a trashed print is a file to unlink, whereas a held row's
//! encrypted media is released by the `generation_queue_media_retire`
//! trigger the moment the queue row is deleted. The sweeper never writes
//! `generation_queue_media` itself; it deletes the row and then asks the
//! lifecycle to collect what the trigger just marked. A settled batch purge
//! releases nothing but the rows.

use crate::routes::ApiError;
use crate::state::AppState;

/// Interval between retention sweeps. Matches the trash sweeper: retention
/// is measured in days, so an hour is already far finer than the smallest
/// window a user can configure.
pub(crate) const QUEUE_SWEEP_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(60 * 60);

pub(crate) use mold_core::{HeldSweepResult, SettledBatchSweepResult};

fn durable_queue_unavailable() -> ApiError {
    ApiError::with_code(
        "the metadata DB is disabled; the durable queue is unavailable",
        "DURABLE_QUEUE_UNAVAILABLE",
        axum::http::StatusCode::NOT_IMPLEMENTED,
    )
}

/// Run one retention sweep now.
#[utoipa::path(
    post,
    path = "/api/queue/held/sweep",
    tag = "queue",
    responses(
        (status = 200, description = "Purged and remaining held-row counts", body = mold_core::HeldSweepResult),
        (status = 501, description = "Metadata DB disabled — the durable queue is unavailable"),
    )
)]
pub(crate) async fn sweep_held_queue(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<axum::Json<HeldSweepResult>, ApiError> {
    if state.metadata_db.as_ref().is_none() {
        return Err(durable_queue_unavailable());
    }
    let result = sweep_held_once(&state)
        .await
        .map_err(|e| ApiError::internal(format!("queue retention sweep failed: {e:#}")))?;
    Ok(axum::Json(result))
}

/// Run one settled-batch retention sweep now.
#[utoipa::path(
    post,
    path = "/api/generation-batches/sweep",
    tag = "generation",
    responses(
        (status = 200, description = "Purged and remaining settled-batch counts", body = mold_core::SettledBatchSweepResult),
        (status = 501, description = "Metadata DB disabled — the durable queue is unavailable"),
    )
)]
pub(crate) async fn sweep_settled_batches(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<axum::Json<SettledBatchSweepResult>, ApiError> {
    if state.metadata_db.as_ref().is_none() {
        return Err(durable_queue_unavailable());
    }
    let result = sweep_settled_batches_once(&state)
        .await
        .map_err(|e| ApiError::internal(format!("settled-batch retention sweep failed: {e:#}")))?;
    Ok(axum::Json(result))
}

/// The retention both passes read, fresh from the live config every pass
/// (`0` keeps everything forever), exactly as the trash sweeper reads its own.
async fn effective_retention_days(state: &AppState) -> u32 {
    let config = state.config.read().await;
    config.queue.effective_held_retention_days()
}

/// The owner whose durable rows a pass may touch. `None` when the durable
/// queue is off (no metadata DB) or no owner was claimed this boot, in which
/// case nothing durable was written and there is nothing to sweep.
fn sweep_owner(state: &AppState) -> Option<String> {
    state.metadata_db.as_ref().as_ref()?;
    state.queue_journal.owner_uuid().map(str::to_string)
}

/// One retention pass over this owner's held rows.
///
/// Reads `queue.held_retention_days` fresh from the live config every pass
/// (`0` keeps held rows forever), exactly as the trash sweeper reads its own.
///
/// The whole loop runs on the blocking pool, like `sweep_trash_once`: every
/// step is synchronous SQLite plus filesystem work, and a backlog of expired
/// rows would otherwise hold a Tokio worker for the length of the sweep — on
/// the startup pass and on every manual `POST /api/queue/held/sweep`.
pub(crate) async fn sweep_held_once(state: &AppState) -> anyhow::Result<HeldSweepResult> {
    let retention = effective_retention_days(state).await;
    let Some(owner_uuid) = sweep_owner(state) else {
        return Ok(HeldSweepResult::default());
    };
    let db = state.metadata_db.clone();
    let lifecycle = state.queue_journal.queue_media_lifecycle();

    tokio::task::spawn_blocking(move || -> anyhow::Result<HeldSweepResult> {
        let Some(db) = db.as_ref().as_ref() else {
            return Ok(HeldSweepResult::default());
        };
        let expired = mold_db::generation_queue::expired_held(
            db,
            &owner_uuid,
            retention,
            mold_core::time::now_epoch_ms(),
        )?;

        let mut purged = 0_u64;
        let mut media_deferred = 0_u64;
        for row in expired {
            // Resolve the GC candidate BEFORE deleting the queue row: the
            // lifecycle resolves it by job id, and the row is what carries
            // the association. Afterwards there is nothing left to ask.
            //
            // A LOOKUP FAILURE IS NOT "no media". Collapsing the error into
            // `None` would purge the row, leave its bytes `gc_pending`, and
            // still report `media_deferred: 0` — the one number that tells
            // an operator startup reconciliation has work left to do.
            let mut candidate = None;
            let mut candidate_unresolved = false;
            if let Some(lifecycle) = lifecycle.as_ref() {
                match lifecycle.candidate_for_job(&row.id) {
                    Ok(found) => candidate = found,
                    Err(error) => {
                        candidate_unresolved = true;
                        tracing::warn!(
                            job = %row.id,
                            %error,
                            "could not resolve expired held media before purge"
                        );
                    }
                }
            }

            let deleted = mold_db::generation_queue::purge_held(
                db,
                &owner_uuid,
                &row.id,
                mold_core::time::now_epoch_ms(),
            )?;
            if !deleted {
                // A retry or cancel won the race between listing and purge.
                // That caller's decision outranks retention.
                continue;
            }
            purged += 1;
            if candidate_unresolved {
                // The row is gone and the retire trigger has marked its
                // obligation, but we never learned which set to collect.
                media_deferred += 1;
                continue;
            }
            // The row is gone, so the retire trigger has already moved the
            // obligation to `gc_pending`. Collect the bytes now; if that
            // fails, startup reconciliation still owns it.
            if let (Some(lifecycle), Some(candidate)) = (lifecycle.as_ref(), candidate) {
                if let Err(error) = lifecycle.cleanup_after_committed_delete(&candidate) {
                    media_deferred += 1;
                    tracing::warn!(
                        job = %row.id,
                        %error,
                        "expired held media remains GC-pending until startup reconciliation"
                    );
                }
            }
        }

        let remaining = mold_db::generation_queue::held_count(db, &owner_uuid)?;
        Ok(HeldSweepResult {
            purged,
            remaining,
            media_deferred,
        })
    })
    .await?
}

/// One retention pass over this owner's settled batches.
///
/// Reads the same `queue.held_retention_days` as the held pass. Runs on the
/// blocking pool for the same reason: it is synchronous SQLite work whose
/// backlog must not hold a Tokio worker.
pub(crate) async fn sweep_settled_batches_once(
    state: &AppState,
) -> anyhow::Result<SettledBatchSweepResult> {
    let retention = effective_retention_days(state).await;
    let Some(owner_uuid) = sweep_owner(state) else {
        return Ok(SettledBatchSweepResult::default());
    };
    let db = state.metadata_db.clone();

    tokio::task::spawn_blocking(move || -> anyhow::Result<SettledBatchSweepResult> {
        let Some(db) = db.as_ref().as_ref() else {
            return Ok(SettledBatchSweepResult::default());
        };
        let expired = mold_db::generation_batches::expired_settled(
            db,
            &owner_uuid,
            retention,
            mold_core::time::now_epoch_ms(),
        )?;
        let mut purged = 0_u64;
        for batch in expired {
            // `purge_settled` re-checks eligibility inside its transaction:
            // a retry that re-queued a child since the listing wins.
            if mold_db::generation_batches::purge_settled(
                db,
                &owner_uuid,
                &batch.id,
                retention,
                mold_core::time::now_epoch_ms(),
            )? {
                purged += 1;
            }
        }
        let remaining = mold_db::generation_batches::settled_count(db, &owner_uuid)?;
        Ok(SettledBatchSweepResult { purged, remaining })
    })
    .await?
}

/// Background retention sweeper: one pass at startup, then hourly, until
/// `shutdown` is cancelled. Each tick runs the held pass and then the
/// settled pass, in that order, so a hold the first pass settles is a
/// receipt the second pass can age from the next tick on.
pub(crate) fn spawn_queue_sweeper(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    use tokio::time::{interval, MissedTickBehavior};
    tokio::spawn(async move {
        let mut tick = interval(QUEUE_SWEEP_INTERVAL);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return,
                _ = tick.tick() => {}
            }
            match sweep_held_once(&state).await {
                Ok(result) if result.purged > 0 => tracing::info!(
                    purged = result.purged,
                    remaining = result.remaining,
                    media_deferred = result.media_deferred,
                    "queue retention sweep purged expired held work"
                ),
                Ok(result) => tracing::debug!(
                    remaining = result.remaining,
                    "queue retention sweep found no held work to purge"
                ),
                Err(error) => tracing::warn!(%error, "queue retention sweep failed"),
            }
            match sweep_settled_batches_once(&state).await {
                Ok(result) if result.purged > 0 => tracing::info!(
                    purged = result.purged,
                    remaining = result.remaining,
                    "queue retention sweep purged settled batch summaries"
                ),
                Ok(result) => tracing::debug!(
                    remaining = result.remaining,
                    "queue retention sweep found no settled batch to purge"
                ),
                Err(error) => {
                    tracing::warn!(%error, "settled-batch retention sweep failed")
                }
            }
        }
    })
}
