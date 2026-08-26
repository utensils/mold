//! Retention for HELD durable queue rows.
//!
//! A hold is durable on purpose: it is work parked for a human, and the
//! durable queue exists so a restart does not lose it. But nothing ever
//! reclaimed one. Every deterministically-failing request left a row that
//! `GET /api/queue` listed forever, and `hold_claimed` deliberately RETAINS
//! that row's `media_set_id` so a retry still has its source images — so the
//! encrypted media store grew with it, unbounded.
//!
//! This is the queue's peer of `gallery_trash`'s retention sweeper and is
//! shaped like it on purpose: one `sweep_*_once` pass reading its retention
//! fresh from the live config, one hourly background task, one manual POST
//! route. What it deliberately does NOT share is the deletion mechanism —
//! a trashed print is a file to unlink, whereas a held row's encrypted media
//! is released by the `generation_queue_media_retire` trigger the moment the
//! queue row is deleted. The sweeper never writes `generation_queue_media`
//! itself; it deletes the row and then asks the lifecycle to collect what
//! the trigger just marked.

use crate::routes::ApiError;
use crate::state::AppState;

/// Interval between held-row retention sweeps. Matches the trash sweeper:
/// retention is measured in days, so an hour is already far finer than the
/// smallest window a user can configure.
pub(crate) const HELD_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60 * 60);

pub(crate) use mold_core::HeldSweepResult;

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
        return Err(ApiError::with_code(
            "the metadata DB is disabled; the durable queue is unavailable",
            "DURABLE_QUEUE_UNAVAILABLE",
            axum::http::StatusCode::NOT_IMPLEMENTED,
        ));
    }
    let result = sweep_held_once(&state)
        .await
        .map_err(|e| ApiError::internal(format!("queue retention sweep failed: {e:#}")))?;
    Ok(axum::Json(result))
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
    let retention = {
        let config = state.config.read().await;
        config.queue.effective_held_retention_days()
    };
    if state.metadata_db.as_ref().is_none() {
        return Ok(HeldSweepResult::default());
    }
    let Some(owner_uuid) = state.queue_journal.owner_uuid().map(str::to_string) else {
        // No claimed queue owner: nothing durable was written this boot.
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

/// Background retention sweeper: one pass at startup, then hourly, until
/// `shutdown` is cancelled.
pub(crate) fn spawn_held_sweeper(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    use tokio::time::{interval, MissedTickBehavior};
    tokio::spawn(async move {
        let mut tick = interval(HELD_SWEEP_INTERVAL);
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
                    "queue retention sweep found nothing to purge"
                ),
                Err(error) => tracing::warn!(%error, "queue retention sweep failed"),
            }
        }
    })
}
