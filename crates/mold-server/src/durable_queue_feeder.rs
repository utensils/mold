//! Bounded bridge from SQLite's durable generation backlog into the runtime
//! scheduler queue.
//!
//! SQLite owns every not-yet-hydrated child. This task is the sole producer
//! for rows admitted through `/api/generation-batches`; legacy singleton rows
//! are deliberately excluded by the batch-child ownership join.

use std::sync::Arc;

use crate::state::{AppState, GenerationJob, SubmitError};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FeederReport {
    pub submitted: usize,
    pub held: usize,
    stop: FeederStop,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum FeederStop {
    /// No durable row was claimable. Only a durable admission can make the
    /// next scan useful; the speculative capacity release is not work.
    #[default]
    Drained,
    /// Runtime hydration is bounded and every slot is currently occupied.
    AtCapacity,
    /// A persistence boundary could not be checked safely. Keep the sole
    /// feeder alive and retry without treating an error as an empty result.
    RecoverableFailure,
    /// The runtime queue receiver is gone, so the server is shutting down.
    TransportClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeederControl {
    Continue,
    Stop,
}

#[derive(Debug, Default)]
struct FeederArbiter {
    /// Process/restart begins with ordinary FIFO. The bit flips only after a
    /// successful DB claim, never after a vanished attached hint.
    prefer_attached: bool,
}

/// Wake publication waiters whenever one claimed row settles or enters the
/// live registry. SQLite remains the ordering authority; this notification is
/// only a latency optimization over the bounded retry timer.
struct PublicationWake(Arc<tokio::sync::Notify>);

impl Drop for PublicationWake {
    fn drop(&mut self) {
        self.0.notify_waiters();
    }
}

/// Clear ownership tokens left by the prior runtime before any new HTTP
/// admission can mint one. This is a serving precondition: callers must await
/// it and propagate failure before constructing the router.
pub(crate) async fn recover_runtime(
    state: &AppState,
) -> anyhow::Result<mold_db::generation_queue::RuntimeClaimRecovery> {
    let journal = state.queue_journal.clone();
    let report = tokio::task::spawn_blocking(move || journal.recover_feeder_runtime())
        .await
        .map_err(|error| anyhow::anyhow!("durable feeder recovery task failed: {error}"))??;
    if report.claims_cleared > 0
        || report.running_requeued > 0
        || report.replays_charged > 0
        || report.replays_held > 0
    {
        tracing::info!(
            claims_cleared = report.claims_cleared,
            running_requeued = report.running_requeued,
            replays_charged = report.replays_charged,
            replays_held = report.replays_held,
            "durable feeder recovered prior runtime claims"
        );
    }
    Ok(report)
}

pub(crate) fn spawn(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Dependency preparation is bounded independently from GPU execution.
        // Multiple workers prevent one catalog/download/probe from blocking
        // unrelated accepted work; SQLite claims and durable queue ranks keep
        // ownership unique and scheduler ordering stable.
        let worker_count = state.queue_capacity.clamp(1, 8);
        let mut workers = tokio::task::JoinSet::new();
        let publication = Arc::new(tokio::sync::Notify::new());
        for _ in 0..worker_count {
            workers.spawn(run_with_retry_delay(
                state.clone(),
                shutdown.clone(),
                mold_db::METADATA_DB_BUSY_TIMEOUT,
                Arc::clone(&publication),
            ));
        }
        while let Some(result) = workers.join_next().await {
            if let Err(error) = result {
                tracing::error!(%error, "durable preparation worker stopped unexpectedly");
                shutdown.cancel();
            }
        }
    })
}

async fn run_with_retry_delay(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
    retry_delay: std::time::Duration,
    publication: Arc<tokio::sync::Notify>,
) {
    tracing::info!(
        capacity = state.queue_capacity,
        "durable generation queue feeder started"
    );
    let current_output_dir = {
        let config = state.config.read().await;
        (!state.is_output_disabled(&config)).then(|| config.effective_output_dir())
    };
    let mut arbiter = FeederArbiter::default();
    loop {
        // Construct both futures before the scan. Notify retains one permit,
        // so a commit or capacity release in the scan-to-wait gap is consumed
        // by the select below instead of stranding a row.
        let durable_wake = state.queue_journal.feeder_notified();
        let capacity_wake = state.queue.capacity_notified();
        tokio::pin!(durable_wake);
        tokio::pin!(capacity_wake);

        let report = feed_available(
            &state,
            current_output_dir.as_deref(),
            &shutdown,
            &mut arbiter,
            &publication,
        )
        .await;
        if report.submitted > 0 || report.held > 0 {
            tracing::debug!(
                submitted = report.submitted,
                held = report.held,
                "durable feeder pass complete"
            );
        }
        if shutdown.is_cancelled() {
            break;
        }
        if wait_for_next_pass(
            report.stop,
            retry_delay,
            &shutdown,
            &mut durable_wake,
            &mut capacity_wake,
        )
        .await
            == FeederControl::Stop
        {
            break;
        }
    }
    tracing::info!("durable generation queue feeder stopped");
}

/// Wait only on events that can make the preceding stop reason actionable.
///
/// In particular, an empty scan has just dropped a speculative queue-slot
/// reservation. That drop correctly wakes other producers, but must not wake
/// this feeder into another empty SQLite scan. Recoverable failures similarly
/// ignore notifications produced by retaining their own claim and retry on
/// the database's existing contention window.
async fn wait_for_next_pass<D, C>(
    stop: FeederStop,
    retry_delay: std::time::Duration,
    shutdown: &tokio_util::sync::CancellationToken,
    durable_wake: D,
    capacity_wake: C,
) -> FeederControl
where
    D: std::future::Future<Output = ()>,
    C: std::future::Future<Output = ()>,
{
    tokio::pin!(durable_wake);
    tokio::pin!(capacity_wake);
    match stop {
        FeederStop::TransportClosed => FeederControl::Stop,
        FeederStop::RecoverableFailure => {
            tokio::select! {
                _ = shutdown.cancelled() => FeederControl::Stop,
                _ = tokio::time::sleep(retry_delay) => FeederControl::Continue,
            }
        }
        FeederStop::Drained => {
            tokio::select! {
                _ = shutdown.cancelled() => FeederControl::Stop,
                _ = &mut durable_wake => FeederControl::Continue,
            }
        }
        FeederStop::AtCapacity => {
            tokio::select! {
                _ = shutdown.cancelled() => FeederControl::Stop,
                _ = &mut durable_wake => FeederControl::Continue,
                _ = &mut capacity_wake => FeederControl::Continue,
            }
        }
    }
}

/// Return a token-owned row to the durable backlog before the feeder resumes.
///
/// A failed release leaves the exact claim token on both the SQLite row and
/// the returned ticket. Retry that ticket in process after the metadata DB's
/// authoritative contention window. If shutdown wins the wait, dropping the
/// returned ticket is inert and startup recovery clears the preserved token.
async fn retain_for_retry(
    mut ticket: crate::queue_journal::QueueTicket,
    shutdown: &tokio_util::sync::CancellationToken,
) {
    loop {
        let job = ticket.id().to_owned();
        let retain_task = tokio::task::spawn_blocking(move || ticket.retain());
        let retained = tokio::select! {
            retained = retain_task => retained,
            _ = shutdown.cancelled() => {
                // The blocking attempt owns the ticket until it finishes. If
                // release fails, its returned Retry ticket is inert when the
                // detached task result is dropped; if it succeeds, the row is
                // already back in the durable backlog.
                tracing::info!(%job, "shutdown left a durable claim release to finish safely");
                return;
            }
        };
        match retained {
            Ok(crate::queue_journal::RetainOutcome::Released) => return,
            Ok(crate::queue_journal::RetainOutcome::Stale) => {
                tracing::warn!(%job, "durable feeder claim became stale while retaining");
                return;
            }
            Ok(crate::queue_journal::RetainOutcome::Retry {
                ticket: retry_ticket,
                error,
            }) => {
                tracing::warn!(%job, %error, "durable feeder claim release will retry");
                ticket = retry_ticket;
                tokio::select! {
                    _ = shutdown.cancelled() => {
                        tracing::info!(%job, "shutdown preserved a token-owned durable row for startup recovery");
                        return;
                    }
                    _ = tokio::time::sleep(mold_db::METADATA_DB_BUSY_TIMEOUT) => {}
                }
            }
            Err(error) => {
                tracing::error!(%job, %error, "durable feeder claim-release task failed");
                return;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HoldClaimOutcome {
    Held,
    Retained,
}

/// Park one claimed row transactionally and resolve any attached HTTP
/// observer. A failed hold returns the exact token to the replay backlog;
/// attached callers receive a terminal error instead of hanging or seeing EOF.
async fn hold_claimed(
    ticket: crate::queue_journal::QueueTicket,
    ingress: Option<&crate::queue_media_ingress::QueueMediaIngress>,
    job_id: &str,
    reason: String,
    retryable: bool,
    shutdown: &tokio_util::sync::CancellationToken,
) -> HoldClaimOutcome {
    let message = reason.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        if retryable {
            ticket.hold_retryable(&reason)
        } else {
            ticket.hold(&reason)
        }
    })
    .await;
    match outcome {
        Ok(crate::queue_journal::RetainOutcome::Released)
        | Ok(crate::queue_journal::RetainOutcome::Stale) => {
            if let Some(ingress) = ingress {
                ingress.fail_claimed(job_id, message);
            }
            HoldClaimOutcome::Held
        }
        Ok(crate::queue_journal::RetainOutcome::Retry { ticket, error }) => {
            tracing::warn!(job = %job_id, %error, "hold transition failed; returning durable row to replay");
            retain_for_retry(ticket, shutdown).await;
            HoldClaimOutcome::Retained
        }
        Err(error) => {
            tracing::error!(job = %job_id, %error, "hold transition worker failed; startup recovery retains the claim");
            if let Some(ingress) = ingress {
                ingress.fail_claimed(
                    job_id,
                    "generation remains queued after a persistence error".into(),
                );
            }
            HoldClaimOutcome::Retained
        }
    }
}

/// Make one best-effort release after the runtime queue transport has closed.
/// There is no live feeder to resume, so a failed release deliberately drops
/// the inert retry ticket and leaves the exact token for startup recovery.
async fn retain_for_shutdown(ticket: crate::queue_journal::QueueTicket) {
    let job = ticket.id().to_owned();
    match tokio::task::spawn_blocking(move || ticket.retain()).await {
        Ok(crate::queue_journal::RetainOutcome::Released) => {}
        Ok(crate::queue_journal::RetainOutcome::Stale) => {
            tracing::warn!(%job, "shutdown found a stale durable feeder claim");
        }
        Ok(crate::queue_journal::RetainOutcome::Retry { error, .. }) => {
            tracing::warn!(%job, %error, "shutdown preserved a token-owned durable row for startup recovery");
        }
        Err(error) => {
            tracing::error!(%job, %error, "shutdown claim-release task failed");
        }
    }
}

struct SelectedClaim {
    claim: mold_db::generation_queue::QueueClaim,
    claimed_as_attached: bool,
}

fn claim_next(
    journal: &std::sync::Arc<crate::queue_journal::QueueJournal>,
    ingress: Option<&crate::queue_media_ingress::QueueMediaIngress>,
    prefer_attached: bool,
) -> anyhow::Result<Option<SelectedClaim>> {
    let claim_attached = || -> anyhow::Result<Option<SelectedClaim>> {
        let Some(ingress) = ingress else {
            return Ok(None);
        };
        while let Some(job_id) = ingress.next_committed_id() {
            match journal.claim_feeder_by_id(&job_id)? {
                Some(claim) => {
                    return Ok(Some(SelectedClaim {
                        claim,
                        claimed_as_attached: true,
                    }));
                }
                None => ingress.discard_hint(&job_id),
            }
        }
        Ok(None)
    };
    let claim_fifo = || {
        journal.claim_next_feeder().map(|claim| {
            claim.map(|claim| SelectedClaim {
                claim,
                claimed_as_attached: false,
            })
        })
    };

    if prefer_attached {
        if let Some(claim) = claim_attached()? {
            return Ok(Some(claim));
        }
        claim_fifo()
    } else {
        if let Some(claim) = claim_fifo()? {
            return Ok(Some(claim));
        }
        claim_attached()
    }
}

fn projection_failure_holds(error: &crate::queue_media_store::QueueMediaError) -> bool {
    matches!(
        error,
        crate::queue_media_store::QueueMediaError::Authentication
            | crate::queue_media_store::QueueMediaError::Corrupt(_)
            | crate::queue_media_store::QueueMediaError::InsecurePath(_)
            | crate::queue_media_store::QueueMediaError::NotFound
            | crate::queue_media_store::QueueMediaError::InvalidIdentity(_)
            | crate::queue_media_store::QueueMediaError::ProjectionUnavailable(_)
    )
}

fn restore_admission_authority(
    lifecycle: Option<&crate::queue_media_lifecycle::QueueMediaLifecycle>,
    instance_id: &str,
    row: &mold_db::generation_queue::GenerationQueueRow,
    request: mold_core::GenerateRequest,
    deferred_media: Option<&crate::queue_media_runtime::DeferredQueueMedia>,
) -> Result<
    crate::durable_admission_authority::RuntimeAuthority,
    crate::durable_admission_authority::Failure,
> {
    use crate::durable_admission_authority::{Failure, FailureDisposition};
    let envelope = match row.admission_authority.as_deref() {
        Some(encoded) => {
            let authority = crate::queue_media_store::QueueMediaAdmissionAuthority::parse(encoded)
                .map_err(|error| Failure {
                    disposition: FailureDisposition::Hold,
                    message: error.to_string(),
                })?;
            let lifecycle = lifecycle.ok_or_else(|| Failure {
                disposition: FailureDisposition::Retain,
                message: "durable admission authority storage is unavailable".into(),
            })?;
            Some(
                lifecycle
                    .open_admission_authority(&row.id, &authority)
                    .map_err(|error| Failure {
                        disposition: if projection_failure_holds(&error) {
                            FailureDisposition::Hold
                        } else {
                            FailureDisposition::Retain
                        },
                        message: error.to_string(),
                    })?,
            )
        }
        None => None,
    };
    let mut request = crate::queue_media_runtime::ZeroizingGenerateRequest::from_owned(request);
    let hydration = match (envelope.as_ref(), deferred_media) {
        (Some(_), Some(media)) => {
            Some(media.hydrate_into(&row.id, &mut request).map_err(|error| {
                match error.disposition() {
                    crate::queue_media_runtime::DeferredHydrationDisposition::Hold => Failure {
                        disposition: FailureDisposition::Hold,
                        message: error.to_string(),
                    },
                    crate::queue_media_runtime::DeferredHydrationDisposition::Retain => Failure {
                        disposition: FailureDisposition::Retain,
                        message: error.to_string(),
                    },
                }
            })?)
        }
        _ => None,
    };
    let restored = crate::durable_admission_authority::restore(
        &request,
        envelope.as_deref().map(|value| value.as_slice()),
        instance_id,
    );
    // Scrub hydrated request media before its private staging lease is
    // released. The restored grant is payload-free and owns no request data.
    drop(request);
    drop(hydration);
    restored
}

/// Resolve the exact claim's bounded durable position, then publish its
/// current affinity and bounded durable position, then publish its runtime
/// cancellation token under the scheduler fence.
///
/// The order read selects at most `queue_capacity` payload-free ids using the
/// durable replay index. The separate durable-transition gate serializes this
/// DB snapshot and registry publication with PATCH and cancellation, without
/// making the scheduler fence wait for SQLite. The database read validates the
/// exact owner/state/claim token in one snapshot. A deep row moved into the
/// runtime window therefore enters the registry at that global position.
/// Valid claims outside the bounded prefix are released for a later FIFO pass;
/// appending them would let an attached deep row bypass unhydrated predecessors.
enum ClaimRegistration {
    Registered {
        cancel: Arc<tokio::sync::Notify>,
        /// Held until the reserved transport accepts the job, preventing a
        /// successor from treating a registry-only predecessor as published.
        publication_guard: tokio::sync::OwnedMutexGuard<()>,
    },
    WaitingForPredecessor,
    Stale,
    OutsideRuntimeWindow,
}

enum ClaimWindow {
    Eligible,
    Stale,
    OutsideRuntimeWindow,
}

/// Reject an exact attached hint before any deferred preparation when the row
/// is not yet in the bounded runtime prefix. The authoritative registration
/// below repeats this check after preparation because PATCH/cancellation may
/// race the work; this first read exists to keep deep attached rows cheap and
/// to preserve their observer for the later FIFO claim.
async fn attached_claim_is_in_runtime_window(
    state: &AppState,
    row_id: &str,
    claim_token: &str,
) -> anyhow::Result<ClaimWindow> {
    let journal = state.queue_journal.clone();
    let order_id = row_id.to_string();
    let order_claim = claim_token.to_string();
    let window = state.queue_capacity;
    let order = tokio::task::spawn_blocking(move || {
        journal.claimed_runtime_position(&order_id, &order_claim, window)
    })
    .await
    .map_err(|error| anyhow::anyhow!("durable queue order task failed: {error}"))??;
    Ok(match order {
        None => ClaimWindow::Stale,
        Some(order) if order.position.is_none() => ClaimWindow::OutsideRuntimeWindow,
        Some(_) => ClaimWindow::Eligible,
    })
}

async fn register_claimed_runtime(
    state: &AppState,
    row: &mold_db::generation_queue::GenerationQueueRow,
    claim_token: &str,
    request: &mut mold_core::GenerateRequest,
) -> anyhow::Result<ClaimRegistration> {
    let (cancel, publication_guard) = {
        // Strict lock order: durable transition -> completed DB read ->
        // scheduler fence. Never move this DB await beneath `_mutation`.
        let _durable_transition = state.queue_journal.lock_durable_transition().await;
        let journal = state.queue_journal.clone();
        let order_id = row.id.clone();
        let order_claim = claim_token.to_string();
        let window = state.queue_capacity;
        let order = tokio::task::spawn_blocking(move || {
            journal.claimed_runtime_position(&order_id, &order_claim, window)
        })
        .await
        .map_err(|error| anyhow::anyhow!("durable queue order task failed: {error}"))??;
        let Some(order) = order else {
            return Ok(ClaimRegistration::Stale);
        };
        let Some(position) = order.position else {
            return Ok(ClaimRegistration::OutsideRuntimeWindow);
        };
        let target_gpu = crate::queue_journal::resolve_replay_affinity(
            request,
            order.target_gpu,
            order.target_device_id.as_deref(),
            |device_id| {
                state
                    .gpu_pool
                    .workers
                    .iter()
                    .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
                    .map(|worker| worker.gpu.ordinal)
            },
        );
        if order.target_gpu.is_some() && target_gpu.is_none() {
            tracing::warn!(
                job = %row.id,
                device = ?order.target_device_id,
                "durable GPU identity is absent or unavailable; resuming on Auto"
            );
        }
        let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
            request,
            request.seed.unwrap_or(0),
            request.scheduler,
            mold_core::build_info::version_string(),
        ));
        let publication_guard = state.scheduler_mutation_fence.clone().lock_owned().await;
        if order
            .predecessor_ids
            .iter()
            .any(|id| state.job_registry.scheduler_lifecycle(id).is_none())
        {
            return Ok(ClaimRegistration::WaitingForPredecessor);
        }
        let cancel = state.job_registry.register_job_at_queued_position(
            &row.id,
            &row.model,
            target_gpu,
            Some(row.seed_pinned),
            Some(metadata),
            position,
        );
        (cancel, publication_guard)
    };
    Ok(ClaimRegistration::Registered {
        cancel,
        publication_guard,
    })
}

async fn feed_available(
    state: &AppState,
    current_output_dir: Option<&std::path::Path>,
    shutdown: &tokio_util::sync::CancellationToken,
    arbiter: &mut FeederArbiter,
    publication: &Arc<tokio::sync::Notify>,
) -> FeederReport {
    let mut report = FeederReport::default();
    'feed: loop {
        let reservation = match state.queue.try_reserve(state.queue_capacity) {
            Ok(reservation) => reservation,
            Err(SubmitError::Full { .. }) => {
                report.stop = FeederStop::AtCapacity;
                return report;
            }
            Err(SubmitError::Cancelled) => {
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
            Err(SubmitError::Shutdown) => {
                report.stop = FeederStop::TransportClosed;
                return report;
            }
        };
        let journal = state.queue_journal.clone();
        let ingress = journal
            .queue_media_admission()
            .map(|admission| admission.ingress().clone());
        let claim_ingress = ingress.clone();
        let prefer_attached = arbiter.prefer_attached;
        let claim = match tokio::task::spawn_blocking(move || {
            claim_next(&journal, claim_ingress.as_deref(), prefer_attached)
        })
        .await
        {
            Ok(Ok(Some(claim))) => claim,
            Ok(Ok(None)) => {
                drop(reservation);
                report.stop = FeederStop::Drained;
                return report;
            }
            Ok(Err(error)) => {
                drop(reservation);
                tracing::warn!(error = %format!("{error:#}"), "durable feeder could not claim the next row");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
            Err(error) => {
                drop(reservation);
                tracing::warn!(%error, "durable feeder claim task failed");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
        };
        let _publication_wake = PublicationWake(Arc::clone(publication));
        arbiter.prefer_attached = !claim.claimed_as_attached;

        let mold_db::generation_queue::QueueClaim {
            mut row,
            claim_token,
            queue_rank,
        } = claim.claim;
        let ticket = state
            .queue_journal
            .attach_claimed(&row.id, claim_token.clone());
        if claim.claimed_as_attached {
            match attached_claim_is_in_runtime_window(state, &row.id, &claim_token).await {
                Ok(ClaimWindow::Eligible) => {}
                Ok(ClaimWindow::Stale) => {
                    if let Some(ingress) = ingress.as_deref() {
                        ingress.discard_hint(&row.id);
                    }
                    let _ = tokio::task::spawn_blocking(move || ticket.discard()).await;
                    drop(reservation);
                    continue;
                }
                Ok(ClaimWindow::OutsideRuntimeWindow) => {
                    if let Some(ingress) = ingress.as_deref() {
                        ingress.defer_claimed_hint(&row.id);
                    }
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    continue;
                }
                Err(error) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::error!(job = %row.id, %error, "durable feeder attached-order lookup failed");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
            }
        }
        let journal = state.queue_journal.clone();
        let completion_id = row.id.clone();
        let db_completion =
            tokio::task::spawn_blocking(move || journal.completed_output(&completion_id)).await;
        let (mut completed_output, db_invalid_authority) = match db_completion {
            Ok(Ok(output)) => (output, None),
            Ok(Err(error)) if error.is_invalid_authority() => (None, Some(error.to_string())),
            Ok(Err(error)) => {
                drop(reservation);
                retain_for_retry(ticket, shutdown).await;
                tracing::error!(job = %row.id, %error, "durable feeder idempotence infrastructure failed; retaining for retry");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
            Err(error) => {
                drop(reservation);
                retain_for_retry(ticket, shutdown).await;
                tracing::error!(job = %row.id, %error, "durable feeder idempotence task failed; retaining for retry");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
        };
        if completed_output.is_none() && row.output_dir.is_dir() {
            // Resolve only the claimed row. Startup remains bounded by runtime
            // queue capacity instead of reconciling the retained backlog or
            // waiting for the independent whole-gallery DB projection pass.
            let gallery_gate = state.gallery_publication_gate.clone();
            let output_dir = row.output_dir.clone();
            let completion_id = row.id.clone();
            let archive_completion = {
                let _gallery_reader = state.gallery_publication_gate.read().await;
                tokio::task::spawn_blocking(move || {
                    crate::batch_transaction::find_completed_output_in_committed_archive(
                        &gallery_gate,
                        &output_dir,
                        &completion_id,
                    )
                })
                .await
            };
            match archive_completion {
                Ok(Ok(output)) => completed_output = output,
                Ok(Err(error)) if error.is_invalid_authority() => {
                    let reason = format!("durable publication authority is invalid: {error}");
                    let held =
                        hold_claimed(ticket, ingress.as_deref(), &row.id, reason, false, shutdown)
                            .await;
                    drop(reservation);
                    if held == HoldClaimOutcome::Retained {
                        report.stop = FeederStop::RecoverableFailure;
                        return report;
                    }
                    tracing::error!(job = %row.id, %error, "held durable generation with invalid publication authority");
                    report.held += 1;
                    continue;
                }
                Ok(Err(error)) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::error!(job = %row.id, %error, "durable archive lookup infrastructure failed; retaining for retry");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
                Err(error) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::error!(job = %row.id, %error, "durable archive lookup task failed; retaining for retry");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
            }
        }
        if completed_output.is_none() {
            if let Some(error) = db_invalid_authority {
                let reason = format!("durable publication metadata is invalid: {error}");
                let held =
                    hold_claimed(ticket, ingress.as_deref(), &row.id, reason, false, shutdown)
                        .await;
                drop(reservation);
                if held == HoldClaimOutcome::Retained {
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
                tracing::error!(job = %row.id, %error, "held durable generation with invalid publication metadata");
                report.held += 1;
                continue;
            }
        } else if let Some(output) = completed_output {
            let result_json = serde_json::json!({
                "filename": output.filename,
                "original_filename": output.original_filename,
            })
            .to_string();
            let _ = tokio::task::spawn_blocking(move || {
                ticket.complete_before_dispatch_with_result(Some(&result_json));
            })
            .await;
            drop(reservation);
            continue;
        }
        if !row.output_dir.is_dir() {
            match current_output_dir {
                Some(replacement) if replacement != row.output_dir => {
                    let journal = state.queue_journal.clone();
                    let id = row.id.clone();
                    let replacement = replacement.to_path_buf();
                    let db_replacement = replacement.clone();
                    let repointed = tokio::task::spawn_blocking(move || {
                        journal.repoint_output(&id, &db_replacement)
                    })
                    .await;
                    if let Err(error) = repointed
                        .map_err(|error| anyhow::anyhow!(error))
                        .and_then(|result| result)
                    {
                        let held = hold_claimed(
                            ticket,
                            ingress.as_deref(),
                            &row.id,
                            "the gallery directory could not be reconciled".into(),
                            false,
                            shutdown,
                        )
                        .await;
                        drop(reservation);
                        if held == HoldClaimOutcome::Retained {
                            report.stop = FeederStop::RecoverableFailure;
                            return report;
                        }
                        tracing::warn!(job = %row.id, %error, "held durable generation with an unusable output target");
                        report.held += 1;
                        continue;
                    }
                    row.output_dir = replacement;
                }
                Some(_) => {
                    let output_dir = row.output_dir.clone();
                    let created =
                        tokio::task::spawn_blocking(move || std::fs::create_dir_all(output_dir))
                            .await;
                    if let Err(error) = created
                        .map_err(|error| anyhow::anyhow!(error))
                        .and_then(|result| result.map_err(anyhow::Error::from))
                    {
                        let held = hold_claimed(
                            ticket,
                            ingress.as_deref(),
                            &row.id,
                            "the gallery directory this job targets cannot be created".into(),
                            false,
                            shutdown,
                        )
                        .await;
                        drop(reservation);
                        if held == HoldClaimOutcome::Retained {
                            report.stop = FeederStop::RecoverableFailure;
                            return report;
                        }
                        tracing::warn!(job = %row.id, %error, "held durable generation with an unusable output target");
                        report.held += 1;
                        continue;
                    }
                }
                None => {
                    let held = hold_claimed(
                        ticket,
                        ingress.as_deref(),
                        &row.id,
                        "server gallery output is disabled".into(),
                        false,
                        shutdown,
                    )
                    .await;
                    drop(reservation);
                    if held == HoldClaimOutcome::Retained {
                        report.stop = FeederStop::RecoverableFailure;
                        return report;
                    }
                    report.held += 1;
                    continue;
                }
            }
        }
        let mut request: mold_core::GenerateRequest = match serde_json::from_str(&row.request_json)
        {
            Ok(request) => request,
            Err(error) => {
                let held = hold_claimed(
                    ticket,
                    ingress.as_deref(),
                    &row.id,
                    "the recorded request could not be deserialized".into(),
                    false,
                    shutdown,
                )
                .await;
                drop(reservation);
                if held == HoldClaimOutcome::Retained {
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
                tracing::warn!(job = %row.id, %error, "held unreadable durable generation");
                report.held += 1;
                continue;
            }
        };
        // Normalize a persisted accelerator identity before deferred
        // preparation validates placement. A device missing after restart
        // falls back to Auto; it must not be held as an invalid request before
        // the final registry fence gets a chance to resolve replay affinity.
        let _ = crate::queue_journal::resolve_replay_affinity(
            &mut request,
            row.target_gpu,
            row.target_device_id.as_deref(),
            |device_id| {
                state
                    .gpu_pool
                    .workers
                    .iter()
                    .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
                    .map(|worker| worker.gpu.ordinal)
            },
        );
        let deferred_media = if let Some(set_id) = row.media_set_id.as_ref() {
            let Some(lifecycle) = state.queue_journal.queue_media_lifecycle() else {
                drop(reservation);
                retain_for_retry(ticket, shutdown).await;
                tracing::error!(job = %row.id, "durable media lifecycle is unavailable; retaining claim");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            };
            let media_set = crate::queue_media_store::MediaSetRef {
                owner_id: row.owner_uuid.clone(),
                job_id: row.id.clone(),
                set_id: set_id.clone(),
            };
            match lifecycle.deferred_media(media_set) {
                Ok(deferred) => Some(deferred),
                Err(error) if projection_failure_holds(&error) => {
                    let reason = format!("durable media projection is invalid: {error}");
                    let held =
                        hold_claimed(ticket, ingress.as_deref(), &row.id, reason, false, shutdown)
                            .await;
                    drop(reservation);
                    if held == HoldClaimOutcome::Retained {
                        report.stop = FeederStop::RecoverableFailure;
                        return report;
                    }
                    tracing::error!(job = %row.id, %error, "held durable generation with invalid media projection");
                    report.held += 1;
                    continue;
                }
                Err(error) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::warn!(job = %row.id, %error, "durable media projection infrastructure failed; retaining for retry");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
            }
        } else {
            None
        };
        let runtime_authority = {
            let lifecycle = state.queue_journal.queue_media_lifecycle();
            let row_for_authority = row.clone();
            let request_for_authority = request.clone();
            let deferred_for_authority = deferred_media.clone();
            let instance_id = state.instance_id.clone();
            match tokio::task::spawn_blocking(move || {
                restore_admission_authority(
                    lifecycle.as_deref(),
                    &instance_id,
                    &row_for_authority,
                    request_for_authority,
                    deferred_for_authority.as_ref(),
                )
            })
            .await
            {
                Ok(Ok(authority)) => authority,
                Ok(Err(crate::durable_admission_authority::Failure {
                    disposition: crate::durable_admission_authority::FailureDisposition::Hold,
                    message: reason,
                })) => {
                    let logged_reason = reason.clone();
                    let held =
                        hold_claimed(ticket, ingress.as_deref(), &row.id, reason, false, shutdown)
                            .await;
                    drop(reservation);
                    if held == HoldClaimOutcome::Retained {
                        report.stop = FeederStop::RecoverableFailure;
                        return report;
                    }
                    tracing::error!(job = %row.id, reason = %logged_reason, "held durable generation with invalid admission authority");
                    report.held += 1;
                    continue;
                }
                Ok(Err(crate::durable_admission_authority::Failure {
                    disposition:
                        crate::durable_admission_authority::FailureDisposition::HoldRetryable,
                    message: reason,
                })) => {
                    let logged_reason = reason.clone();
                    let held =
                        hold_claimed(ticket, ingress.as_deref(), &row.id, reason, true, shutdown)
                            .await;
                    drop(reservation);
                    if held == HoldClaimOutcome::Retained {
                        report.stop = FeederStop::RecoverableFailure;
                        return report;
                    }
                    tracing::warn!(job = %row.id, reason = %logged_reason, "held durable generation until its admission runtime is available");
                    report.held += 1;
                    continue;
                }
                Ok(Err(crate::durable_admission_authority::Failure {
                    disposition: crate::durable_admission_authority::FailureDisposition::Retain,
                    message: reason,
                })) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::warn!(job = %row.id, reason = %reason, "durable admission authority is temporarily unavailable; retaining for replay");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
                Err(error) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::error!(job = %row.id, %error, "durable admission authority worker failed; retaining for replay");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
            }
        };
        // Durable acknowledgement is already complete. All model/catalog,
        // expansion, media probing, server-path resolution, and dependency
        // materialization now happens here, before registry/scheduler
        // publication. A crash simply releases the claim for startup replay.
        let mut preparation_request =
            crate::queue_media_runtime::ZeroizingGenerateRequest::from_owned(request);
        let preparation_lease = if let Some(media) = deferred_media.as_ref() {
            let media = media.clone();
            let job_id = row.id.clone();
            match tokio::task::spawn_blocking(move || {
                let result = media.hydrate_into(&job_id, &mut preparation_request);
                (preparation_request, result)
            })
            .await
            {
                Ok((hydrated, Ok(lease))) => {
                    preparation_request = hydrated;
                    Some(lease)
                }
                Ok((_request, Err(error))) => {
                    let reason = error.to_string();
                    match error.disposition() {
                        crate::queue_media_runtime::DeferredHydrationDisposition::Hold => {
                            let held = hold_claimed(
                                ticket,
                                ingress.as_deref(),
                                &row.id,
                                reason,
                                false,
                                shutdown,
                            )
                            .await;
                            drop(reservation);
                            if held == HoldClaimOutcome::Retained {
                                report.stop = FeederStop::RecoverableFailure;
                                return report;
                            }
                            report.held += 1;
                            continue;
                        }
                        crate::queue_media_runtime::DeferredHydrationDisposition::Retain => {
                            drop(reservation);
                            retain_for_retry(ticket, shutdown).await;
                            report.stop = FeederStop::RecoverableFailure;
                            return report;
                        }
                    }
                }
                Err(error) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::error!(job = %row.id, %error, "deferred admission hydration worker failed");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
            }
        } else {
            None
        };
        let prepared_route = crate::routes::prepare_generation_after_durable_ack(
            state,
            &mut preparation_request,
            runtime_authority,
        )
        .await;
        // Publish only a payload-free copy. Dropping the RAII owner before the
        // staging lease preserves scrub-before-release on success; unwind and
        // cancellation take the same order because locals drop in reverse.
        let mut request = preparation_request.scrubbed_clone();
        drop(preparation_request);
        drop(preparation_lease);
        let prepared_route = match prepared_route {
            Ok(route) => route,
            Err(error) => {
                let reason = format!("deferred generation preparation failed: {}", error.error);
                let retryable = match crate::durable_admission_authority::preparation_disposition(
                    &error,
                ) {
                    crate::durable_admission_authority::PreparationDisposition::Hold => false,
                    crate::durable_admission_authority::PreparationDisposition::HoldRetryable => {
                        true
                    }
                    crate::durable_admission_authority::PreparationDisposition::Retain => {
                        drop(reservation);
                        retain_for_retry(ticket, shutdown).await;
                        report.stop = FeederStop::RecoverableFailure;
                        return report;
                    }
                };
                let held = hold_claimed(
                    ticket,
                    ingress.as_deref(),
                    &row.id,
                    reason,
                    retryable,
                    shutdown,
                )
                .await;
                drop(reservation);
                if held == HoldClaimOutcome::Retained {
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
                report.held += 1;
                continue;
            }
        };
        let preparation_warnings = prepared_route.warnings;
        let resolved_references = prepared_route.resolved_references;
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        let h3_private_ingress_grant = prepared_route.h3_private_ingress_grant;
        // Check SQLite before taking the scheduler publication fence. A later
        // cancellation waits on that fence, observes the registered token,
        // and cancels the submitted job through the ordinary live path.
        let journal = state.queue_journal.clone();
        let cancel_id = row.id.clone();
        let cancel_requested =
            tokio::task::spawn_blocking(move || journal.feeder_cancel_requested(&cancel_id)).await;
        match cancel_requested {
            Ok(Ok(true)) => {
                let _ = tokio::task::spawn_blocking(move || ticket.discard()).await;
                drop(reservation);
                continue;
            }
            Ok(Ok(false)) => {}
            Ok(Err(error)) => {
                drop(reservation);
                retain_for_retry(ticket, shutdown).await;
                tracing::error!(job = %row.id, %error, "durable feeder cancellation check failed");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
            Err(error) => {
                drop(reservation);
                retain_for_retry(ticket, shutdown).await;
                tracing::error!(job = %row.id, %error, "durable feeder cancellation task failed");
                report.stop = FeederStop::RecoverableFailure;
                return report;
            }
        }
        // Registry publication shares the scheduler mutation fence with
        // DELETE /api/queue, but SQLite is synchronous and can be stalled by
        // another connection. Publish the cancellation token under the fence,
        // release it, and only then perform the blocking durable check. The
        // runtime queue does not receive this job until the check completes:
        // cancellation before/during the check is observed in SQLite, while a
        // later cancellation observes and trips the registered live token.
        let (cancel, publication_guard) = loop {
            let publication_wake = publication.notified();
            match register_claimed_runtime(state, &row, &claim_token, &mut request).await {
                Ok(ClaimRegistration::Registered {
                    cancel,
                    publication_guard,
                }) => break (cancel, publication_guard),
                Ok(ClaimRegistration::WaitingForPredecessor) => {
                    tokio::select! {
                        _ = shutdown.cancelled() => {
                            drop(reservation);
                            retain_for_retry(ticket, shutdown).await;
                            report.stop = FeederStop::TransportClosed;
                            return report;
                        }
                        _ = publication_wake => {}
                        _ = tokio::time::sleep(std::time::Duration::from_millis(25)) => {}
                    }
                }
                Ok(ClaimRegistration::Stale) => {
                    let _ = tokio::task::spawn_blocking(move || ticket.discard()).await;
                    drop(reservation);
                    continue 'feed;
                }
                Ok(ClaimRegistration::OutsideRuntimeWindow) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    continue 'feed;
                }
                Err(error) => {
                    drop(reservation);
                    retain_for_retry(ticket, shutdown).await;
                    tracing::error!(job = %row.id, %error, "durable feeder order lookup failed");
                    report.stop = FeederStop::RecoverableFailure;
                    return report;
                }
            }
        };
        // Transfer the observer only after every fallible preparation and the
        // final durable-order/cancellation fence. Retains keep it attached;
        // holds resolve it explicitly through `hold_claimed`.
        let observer = ingress
            .as_deref()
            .and_then(|ingress| ingress.take_claimed(&row.id));
        let crate::job_supervisor::SupervisedJob {
            result_tx,
            outcome_rx,
        } = crate::job_supervisor::supervise_job(row.id.clone(), cancel);
        let progress_tx = match observer {
            Some(observer) => match observer.mode() {
                crate::queue_media_ingress::ObserverMode::Raw => {
                    observer.deliver(crate::queue_media_ingress::AttachedObserver::Raw {
                        outcome: outcome_rx,
                        warnings: preparation_warnings,
                    });
                    None
                }
                crate::queue_media_ingress::ObserverMode::Sse(_) => {
                    let (progress_tx, messages) = tokio::sync::mpsc::unbounded_channel();
                    for warning in preparation_warnings.all() {
                        let _ = progress_tx.send(crate::state::SseMessage::Progress(
                            mold_core::SseProgressEvent::Info {
                                message: warning.to_string(),
                            },
                        ));
                    }
                    let cancellation_tx = progress_tx.clone();
                    tokio::spawn(async move {
                        if let Ok(crate::job_supervisor::SupervisedOutcome::Cancelled) =
                            outcome_rx.await
                        {
                            let _ = cancellation_tx.send(crate::state::SseMessage::Error(
                                mold_core::SseErrorEvent::failed("cancelled".to_string()),
                            ));
                        }
                    });
                    observer
                        .deliver(crate::queue_media_ingress::AttachedObserver::Sse { messages });
                    Some(progress_tx)
                }
            },
            None => {
                drop(outcome_rx);
                None
            }
        };
        let id = row.id.clone();
        let job = GenerationJob {
            id: id.clone(),
            durable_queue_rank: Some(queue_rank),
            request,
            deferred_media,
            resolved_references,
            completion_payload: crate::queue_journal::completion_payload_from_str(
                &row.completion_payload,
            ),
            progress_tx,
            result_tx,
            output_dir: Some(row.output_dir),
            batch_child: None,
            journal: Some(ticket),
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant,
        };
        match reservation.submit(job).await {
            Ok(_) => {
                drop(publication_guard);
                report.submitted += 1;
            }
            Err(returned) => {
                let (error, mut job) = *returned;
                if let Some(ticket) = job.journal.take() {
                    retain_for_shutdown(ticket).await;
                }
                state.job_registry.remove(&id);
                drop(publication_guard);
                tracing::warn!(job = %id, ?error, "durable feeder transport stopped; row retained");
                report.stop = FeederStop::TransportClosed;
                return report;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::queue_journal::{BatchJournalAdmission, JournalAdmission, QueueJournal};
    use crate::state::{AppState, SseCompletionPayload};

    fn request(prompt: &str) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": prompt,
            "model": "mock-model",
            "width": 512,
            "height": 512,
            "steps": 1,
            "batch_size": 1,
            "output_format": "png"
        }))
        .unwrap()
    }

    fn state(capacity: usize) -> (AppState, tokio::sync::mpsc::Receiver<GenerationJob>) {
        let (state, rx, _) = state_with_home(capacity);
        (state, rx)
    }

    fn state_with_home(
        capacity: usize,
    ) -> (
        AppState,
        tokio::sync::mpsc::Receiver<GenerationJob>,
        std::path::PathBuf,
    ) {
        let root = tempfile::tempdir().unwrap().keep();
        let gallery = root.join("gallery");
        std::fs::create_dir_all(&gallery).unwrap();
        let mut state = AppState::for_tests();
        let (tx, rx) = tokio::sync::mpsc::channel(capacity.max(1));
        state.queue = crate::state::QueueHandle::new(tx);
        state.queue_capacity = capacity;
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.queue_journal = Arc::new(QueueJournal::new(
            state.metadata_db.clone(),
            Some(&root),
            "feeder-test",
        ));
        let transformer = root.join("mock-model.safetensors");
        let vae = root.join("mock-vae.safetensors");
        std::fs::write(&transformer, b"test transformer fixture").unwrap();
        std::fs::write(&vae, b"test vae fixture").unwrap();
        let mut config = state.config.try_write().unwrap();
        config.output_dir = Some(gallery.to_string_lossy().into());
        config.models.insert(
            "mock-model".to_string(),
            mold_core::ModelConfig {
                family: Some("sdxl".to_string()),
                transformer: Some(transformer.to_string_lossy().into()),
                vae: Some(vae.to_string_lossy().into()),
                ..Default::default()
            },
        );
        drop(config);
        (state, rx, root)
    }

    fn admit(state: &AppState, count: usize) -> Vec<String> {
        let requests = (0..count)
            .map(|index| request(&format!("job {index}")))
            .collect::<Vec<_>>();
        let ids = (0..count)
            .map(|index| format!("job-{index}"))
            .collect::<Vec<_>>();
        let output = state.config.try_read().unwrap().effective_output_dir();
        let children = requests
            .iter()
            .zip(&ids)
            .map(|(request, id)| JournalAdmission {
                id,
                request,
                output_dir: Some(&output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .collect::<Vec<_>>();
        state
            .queue_journal
            .record_batch(BatchJournalAdmission {
                id: "batch",
                client_batch_id: "client",
                request_sha256: "sha",
                children: &children,
            })
            .unwrap();
        ids
    }

    fn archive_output_without_db(
        state: &AppState,
        output_dir: &std::path::Path,
        filename: &str,
        job_id: &str,
    ) {
        let path = output_dir.join(filename);
        std::fs::write(&path, format!("published bytes for {filename}")).unwrap();
        crate::batch_transaction::sync_ordinary_gallery_directory(output_dir).unwrap();
        let mut metadata =
            mold_core::OutputMetadata::from_generate_request(&request("published"), 7, None, "v");
        metadata.job_id = Some(job_id.to_string());
        let params = mold_db::persist::OutputRecordParams {
            format: mold_core::OutputFormat::Png,
            metadata: &metadata,
            source: mold_db::RecordSource::Server,
            generation_time_ms: Some(1),
            backend: Some("test"),
        };
        let record =
            mold_db::persist::build_saved_output_record(output_dir, filename, &path, &params);
        let authority =
            crate::batch_transaction::acquire_gallery_bookkeeping_lock(output_dir).unwrap();
        crate::batch_transaction::archive_ordinary_gallery_record(
            output_dir,
            &path,
            record,
            &state.gallery_publication_gate,
            &authority,
        )
        .unwrap();
    }

    async fn await_completed_batch(
        state: &AppState,
    ) -> mold_db::generation_batches::DurableGenerationBatchDetail {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let detail = mold_db::generation_batches::get_durable(
                    state.metadata_db.as_ref().as_ref().unwrap(),
                    state.queue_journal.owner_uuid().unwrap(),
                    "batch",
                )
                .unwrap()
                .unwrap();
                if detail.children[0].state == "complete" {
                    break detail;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("published output is reconciled without dispatch")
    }

    #[tokio::test]
    async fn deep_backlog_hydrates_no_more_than_runtime_capacity() {
        let (state, mut rx) = state(3);
        admit(&state, 20);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut jobs = Vec::new();
        for _ in 0..3 {
            jobs.push(
                tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        assert_eq!(state.queue.pending(), 3);
        assert_eq!(state.job_registry.len(), 3);
        assert!(rx.try_recv().is_err());
        assert_eq!(state.queue_journal.list_all().len(), 20);
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(jobs);
    }

    #[tokio::test]
    async fn default_capacity_publishes_more_than_one_worker_width_without_deadlock() {
        let (state, mut rx) = state(200);
        let ids = admit(&state, 9);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());

        let mut jobs = Vec::new();
        for _ in 0..ids.len() {
            jobs.push(
                tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("durable publication must not wait for the full queue capacity")
                    .expect("the runtime queue remains open"),
            );
        }
        assert_eq!(
            jobs.iter().map(|job| job.id.as_str()).collect::<Vec<_>>(),
            ids.iter().map(String::as_str).collect::<Vec<_>>()
        );

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(jobs);
    }

    #[tokio::test]
    async fn deep_reorder_hydrates_at_its_global_position_ahead_of_live_tail() {
        let (state, mut rx) = state(3);
        let ids = admit(&state, 4);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());

        let mut hydrated = Vec::new();
        for _ in 0..3 {
            hydrated.push(
                tokio::time::timeout(Duration::from_secs(2), rx.recv())
                    .await
                    .expect("the initial runtime window hydrates")
                    .expect("the runtime queue remains open"),
            );
        }
        assert_eq!(state.job_registry.queued_ids_in_order(), ids[..3]);

        let outcome = state
            .queue_journal
            .patch_owned_queued(&ids[3], None, None, Some(0))
            .unwrap();
        assert!(matches!(
            outcome,
            mold_db::generation_queue::OwnedQueuedPatchOutcome::Updated { position: 0, .. }
        ));
        assert_eq!(
            state
                .queue_journal
                .projection_page(None, 3)
                .unwrap()
                .rows
                .into_iter()
                .map(|row| row.id)
                .collect::<Vec<_>>(),
            vec![ids[3].clone(), ids[0].clone(), ids[1].clone()],
            "the durable response position and first page share one global order"
        );

        // Retire one hydrated row to release exactly one bounded runtime slot.
        // The reordered deep row is then the feeder's next claim, but its
        // transport necessarily arrives after the two already-buffered jobs.
        // Registry insertion must nevertheless publish it ahead of both.
        let mut retired = hydrated.remove(0);
        retired
            .journal
            .take()
            .expect("hydrated durable row owns its exact claim")
            .discard();
        state.job_registry.remove(&ids[0]);
        state.queue.decrement();

        let reordered = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("released capacity wakes the feeder")
            .expect("the runtime queue remains open");
        assert_eq!(reordered.id, ids[3]);
        assert_eq!(
            state.job_registry.queued_ids_in_order(),
            vec![ids[3].clone(), ids[1].clone(), ids[2].clone()],
            "the deep row enters the actual scheduler order at global position zero"
        );
        assert_eq!(
            state
                .job_registry
                .snapshot()
                .entries
                .into_iter()
                .map(|entry| (entry.id, entry.position))
                .collect::<Vec<_>>(),
            vec![
                (ids[3].clone(), 0),
                (ids[1].clone(), 1),
                (ids[2].clone(), 2)
            ]
        );

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(reordered);
        drop(hydrated);
    }

    #[tokio::test]
    async fn claimed_row_outside_runtime_prefix_is_released_instead_of_appended() {
        let (state, _rx) = state(2);
        let ids = admit(&state, 3);
        let claim = state
            .queue_journal
            .claim_feeder_by_id(&ids[2])
            .unwrap()
            .expect("an attached hint can claim a deep row");
        let mut request = request("deep attached");
        assert!(matches!(
            register_claimed_runtime(&state, &claim.row, &claim.claim_token, &mut request,)
                .await
                .unwrap(),
            ClaimRegistration::OutsideRuntimeWindow
        ));
        assert!(state.job_registry.entry(&ids[2]).is_none());
        assert!(matches!(
            state
                .queue_journal
                .attach_claimed(&ids[2], claim.claim_token)
                .retain(),
            crate::queue_journal::RetainOutcome::Released
        ));
        assert_eq!(
            state
                .queue_journal
                .claim_next_feeder()
                .unwrap()
                .unwrap()
                .row
                .id,
            ids[0],
            "the unhydrated FIFO predecessor remains authoritative"
        );
    }

    #[tokio::test]
    async fn attached_deep_hint_is_deferred_without_consuming_its_observer() {
        let (state, _rx) = state(2);
        let ids = admit(&state, 3);
        let ingress = crate::queue_media_ingress::QueueMediaIngress::new(2);
        let _registration = ingress
            .reserve(&ids[2], crate::queue_media_ingress::ObserverMode::Raw)
            .unwrap();
        ingress.publish_committed(&ids[2]);
        let selected = claim_next(&state.queue_journal, Some(&ingress), true)
            .unwrap()
            .expect("the direct observer supplies an exact claim hint");
        assert!(selected.claimed_as_attached);
        assert!(matches!(
            attached_claim_is_in_runtime_window(
                &state,
                &selected.claim.row.id,
                &selected.claim.claim_token,
            )
            .await
            .unwrap(),
            ClaimWindow::OutsideRuntimeWindow
        ));

        ingress.defer_claimed_hint(&selected.claim.row.id);
        assert!(matches!(
            state
                .queue_journal
                .attach_claimed(&selected.claim.row.id, selected.claim.claim_token)
                .retain(),
            crate::queue_journal::RetainOutcome::Released
        ));

        assert_eq!(ingress.next_committed_id(), None);
        assert_eq!(ingress.attached_len(), 1);
        assert_eq!(
            state
                .queue_journal
                .claim_next_feeder()
                .unwrap()
                .unwrap()
                .row
                .id,
            ids[0],
            "the durable FIFO predecessor remains next"
        );
    }

    #[tokio::test]
    async fn exact_claim_waits_for_an_unclaimed_durable_predecessor() {
        let (state, _rx) = state(2);
        let ids = admit(&state, 2);
        let later = state
            .queue_journal
            .claim_feeder_by_id(&ids[1])
            .unwrap()
            .unwrap();
        let mut later_request = request("later");
        assert!(matches!(
            register_claimed_runtime(&state, &later.row, &later.claim_token, &mut later_request,)
                .await
                .unwrap(),
            ClaimRegistration::WaitingForPredecessor
        ));

        let first = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        let mut first_request = request("first");
        let ClaimRegistration::Registered {
            publication_guard, ..
        } = register_claimed_runtime(&state, &first.row, &first.claim_token, &mut first_request)
            .await
            .unwrap()
        else {
            panic!("oldest claim must publish first");
        };
        drop(publication_guard);

        let ClaimRegistration::Registered {
            publication_guard, ..
        } = register_claimed_runtime(&state, &later.row, &later.claim_token, &mut later_request)
            .await
            .unwrap()
        else {
            panic!("successor publishes after its predecessor");
        };
        drop(publication_guard);
        state.job_registry.remove(&ids[0]);
        state.job_registry.remove(&ids[1]);
        state
            .queue_journal
            .attach_claimed(&first.row.id, first.claim_token)
            .discard();
        state
            .queue_journal
            .attach_claimed(&later.row.id, later.claim_token)
            .discard();
    }

    #[tokio::test]
    async fn claimed_reorder_is_authoritative_before_publication() {
        let (state, _rx) = state(2);
        let ids = admit(&state, 2);
        let first = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        let second = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert!(matches!(
            state
                .queue_journal
                .patch_owned_claimed_queued(&ids[1], None, None, Some(0))
                .unwrap(),
            mold_db::generation_queue::OwnedQueuedPatchOutcome::Updated { position: 0, .. }
        ));

        let mut first_request = request("formerly first");
        assert!(matches!(
            register_claimed_runtime(&state, &first.row, &first.claim_token, &mut first_request,)
                .await
                .unwrap(),
            ClaimRegistration::WaitingForPredecessor
        ));
        let mut second_request = request("reordered first");
        let ClaimRegistration::Registered {
            publication_guard, ..
        } = register_claimed_runtime(
            &state,
            &second.row,
            &second.claim_token,
            &mut second_request,
        )
        .await
        .unwrap()
        else {
            panic!("reordered claim must publish at its current durable position");
        };
        drop(publication_guard);
        assert_eq!(
            state.job_registry.queued_ids_in_order(),
            vec![ids[1].clone()]
        );

        state.job_registry.remove(&ids[1]);
        state
            .queue_journal
            .attach_claimed(&first.row.id, first.claim_token)
            .discard();
        state
            .queue_journal
            .attach_claimed(&second.row.id, second.claim_token)
            .discard();
    }

    #[tokio::test]
    async fn deep_reorder_racing_an_older_handoff_controls_the_next_live_grant_order() {
        let (state, mut rx) = state(2);
        let ids = admit(&state, 4);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());

        let mut first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        let mut second = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!([first.id.as_str(), second.id.as_str()], ["job-0", "job-1"]);

        // Freeze only final scheduler publication. The feeder remains free to
        // claim job-2 and complete its bounded SQLite order read, then waits
        // at the scheduler fence while owning the durable transition.
        let scheduler_guard = state.scheduler_mutation_fence.lock().await;
        first
            .journal
            .take()
            .expect("first hydrated job owns its exact claim")
            .discard();
        state.job_registry.remove(&ids[0]);
        state.queue.decrement();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !state.queue_journal.durable_transition_is_locked() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the older handoff reaches final scheduler publication");

        // A PATCH of still-deep job-3 queues behind that exact handoff. Once
        // it commits position zero, its bounded runtime projection remains
        // authoritative when the next slot opens.
        let patch_state = state.clone();
        let deep_id = ids[3].clone();
        let patch = tokio::spawn(async move {
            let _transition = patch_state.queue_journal.lock_durable_transition().await;
            let journal = patch_state.queue_journal.clone();
            let mutation_id = deep_id.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                journal.patch_owned_queued(&mutation_id, None, None, Some(0))
            })
            .await
            .unwrap()
            .unwrap();
            let _scheduler = patch_state.scheduler_mutation_fence.lock().await;
            if patch_state.job_registry.entry(&deep_id).is_some() {
                patch_state
                    .job_registry
                    .reorder_queued(&deep_id, 0)
                    .unwrap();
            }
            outcome
        });
        assert!(!patch.is_finished(), "PATCH waits behind the exact handoff");
        drop(scheduler_guard);

        let third = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("the older handoff publishes")
            .expect("runtime queue remains open");
        assert_eq!(third.id, ids[2]);
        let outcome = patch.await.unwrap();
        assert!(matches!(
            outcome,
            mold_db::generation_queue::OwnedQueuedPatchOutcome::Updated { position: 0, .. }
        ));

        second
            .journal
            .take()
            .expect("second hydrated job owns its exact claim")
            .discard();
        state.job_registry.remove(&ids[1]);
        state.queue.decrement();
        let reordered = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("deep reordered row hydrates after capacity opens")
            .expect("runtime queue remains open");
        assert_eq!(reordered.id, ids[3]);
        assert_eq!(
            state.job_registry.queued_ids_in_order(),
            vec![ids[3].clone(), ids[2].clone()],
            "the deep durable reorder, not handoff/channel arrival, controls live grant order"
        );

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(first);
        drop(second);
        drop(third);
        drop(reordered);
    }

    #[tokio::test]
    async fn runtime_reservation_precedes_every_db_claim() {
        let (state, _rx) = state(0);
        admit(&state, 1);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        tokio::task::yield_now().await;

        let rows = state.queue_journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].state,
            mold_db::generation_queue::QueueRowState::Queued
        );
        assert_eq!(rows[0].dispatch_attempts, 0);
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[test]
    fn successful_claims_alternate_fifo_and_exact_attached_authority() {
        let (state, _rx) = state(3);
        let ids = admit(&state, 3);
        let ingress = crate::queue_media_ingress::QueueMediaIngress::new(3);
        let attached = ingress
            .reserve(&ids[1], crate::queue_media_ingress::ObserverMode::Raw)
            .unwrap();
        ingress.publish_committed(&ids[1]);

        let first = claim_next(&state.queue_journal, Some(&ingress), false)
            .unwrap()
            .unwrap();
        assert_eq!(first.claim.row.id, ids[0]);
        assert!(!first.claimed_as_attached);

        let second = claim_next(&state.queue_journal, Some(&ingress), true)
            .unwrap()
            .unwrap();
        assert_eq!(second.claim.row.id, ids[1]);
        assert!(second.claimed_as_attached);

        let third = claim_next(&state.queue_journal, Some(&ingress), false)
            .unwrap()
            .unwrap();
        assert_eq!(third.claim.row.id, ids[2]);
        assert!(!third.claimed_as_attached);

        for selected in [first, second, third] {
            state
                .queue_journal
                .attach_claimed(&selected.claim.row.id, selected.claim.claim_token)
                .retain();
        }
        drop(attached);
    }

    #[tokio::test]
    async fn invalid_oldest_publication_is_held_without_blocking_the_next_job() {
        let (state, mut rx) = state(1);
        admit(&state, 2);
        let output_dir =
            mold_db::canonical_dir_string(&state.config.try_read().unwrap().effective_output_dir());
        state
            .metadata_db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                for filename in ["first.png", "second.png"] {
                    conn.execute(
                        "INSERT INTO generations
                            (filename, output_dir, created_at_ms, format, model, metadata_json,
                             queue_job_id, queue_job_metadata_state)
                         VALUES (?1, ?2, 1, 'png', 'mock-model', ?3, 'job-0', 1)",
                        (
                            filename,
                            output_dir.as_str(),
                            r#"{"job_id":"job-0","seed":7}"#,
                        ),
                    )?;
                }
                Ok(())
            })
            .unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut next = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("the next valid job must not be head-of-line blocked")
            .expect("the runtime queue remains open");
        assert_eq!(next.id, "job-1");
        let held = state
            .queue_journal
            .list_all()
            .into_iter()
            .find(|row| row.id == "job-0")
            .unwrap();
        assert_eq!(held.state, mold_db::generation_queue::QueueRowState::Held);
        assert!(held
            .held_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("publication metadata is invalid")));

        next.journal.take().unwrap().complete_before_dispatch();
        state.job_registry.remove(&next.id);
        state.queue.decrement();
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn insecure_oldest_media_projection_is_held_once_and_later_work_proceeds() {
        use std::os::unix::fs::symlink;

        use crate::queue_journal::{MediaBatchJournalAdmission, MediaJournalAdmission};
        use crate::queue_media_lifecycle::QueueMediaLifecycle;
        use crate::queue_media_startup::reconcile_claimed_owner;
        use crate::queue_media_store::{
            QueueMediaOperationFingerprint, QueueMediaProjection, QueueMediaStore, SealMedia,
        };

        let (state, mut rx, home) = state_with_home(1);
        let owner = state.queue_journal.owner_uuid().unwrap().to_string();
        let store = QueueMediaStore::open(&home).unwrap().store;
        let media_set = store
            .seal_v2_with_operation_fingerprint(
                &owner,
                "job-0",
                &QueueMediaOperationFingerprint::sha256_v1(b"feeder-insecure-projection"),
                &QueueMediaProjection::default(),
                vec![SealMedia::bytes("source", "source.png", vec![1, 2, 3])],
            )
            .unwrap();
        let requests = [request("job 0"), request("job 1")];
        let request_json = requests
            .iter()
            .map(|request| serde_json::to_string(request).unwrap())
            .collect::<Vec<_>>();
        let output = state.config.try_read().unwrap().effective_output_dir();
        state
            .queue_journal
            .record_batch_with_media(MediaBatchJournalAdmission {
                id: "batch",
                client_batch_id: "client",
                operation_receipt: "receipt",
                children: &[
                    MediaJournalAdmission {
                        id: "job-0",
                        model: &requests[0].model,
                        request_json: &request_json[0],
                        media_set: Some(&media_set),
                        output_dir: &output,
                        target_gpu: None,
                        target_device_id: None,
                        completion_payload: SseCompletionPayload::MetadataOnly,
                        seed_pinned: false,
                        admission_authority: None,
                    },
                    MediaJournalAdmission {
                        id: "job-1",
                        model: &requests[1].model,
                        request_json: &request_json[1],
                        media_set: None,
                        output_dir: &output,
                        target_gpu: None,
                        target_device_id: None,
                        completion_payload: SseCompletionPayload::MetadataOnly,
                        seed_pinned: false,
                        admission_authority: None,
                    },
                ],
                observer_job_ids: &[],
            })
            .unwrap();

        let lifecycle = Arc::new(QueueMediaLifecycle::new(
            state.metadata_db.clone(),
            home.clone(),
            owner,
        ));
        state
            .queue_journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .unwrap();
        assert!(
            reconcile_claimed_owner(&state.queue_journal, lifecycle.as_ref())
                .unwrap()
                .durable_media_ready
        );

        let active_root = home.join("queue-media").join("v1").join("active");
        let bundle = std::fs::read_dir(active_root)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let bundle = std::fs::read_dir(bundle)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let bundle = std::fs::read_dir(bundle)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let symlink_target = home.join("projection-target.qms");
        std::fs::rename(&bundle, &symlink_target).unwrap();
        symlink(&symlink_target, &bundle).unwrap();
        let target_before = std::fs::read(&symlink_target).unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let current_output = state.config.try_read().unwrap().effective_output_dir();
        let mut arbiter = FeederArbiter::default();
        let publication = Arc::new(tokio::sync::Notify::new());
        let report = feed_available(
            &state,
            Some(&current_output),
            &shutdown,
            &mut arbiter,
            &publication,
        )
        .await;
        assert_eq!(report.held, 1);
        assert_eq!(report.submitted, 1);
        assert_eq!(report.stop, FeederStop::AtCapacity);

        let rows = state.queue_journal.list_all();
        let held = rows.iter().find(|row| row.id == "job-0").unwrap();
        assert_eq!(held.state, mold_db::generation_queue::QueueRowState::Held);
        assert!(held
            .held_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("projection is invalid")));
        let mut later = rx.try_recv().expect("later ordinary work must proceed");
        assert_eq!(later.id, "job-1");
        assert_eq!(std::fs::read(&symlink_target).unwrap(), target_before);
        assert!(std::fs::symlink_metadata(&bundle)
            .unwrap()
            .file_type()
            .is_symlink());

        later.journal.take().unwrap().complete_before_dispatch();
        state.job_registry.remove(&later.id);
        state.queue.decrement();
        let drained = feed_available(
            &state,
            Some(&current_output),
            &shutdown,
            &mut arbiter,
            &publication,
        )
        .await;
        assert_eq!(drained, FeederReport::default());
        let rows = state.queue_journal.list_all();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "job-0");
        assert_eq!(
            rows[0].state,
            mold_db::generation_queue::QueueRowState::Held
        );
    }

    #[tokio::test]
    async fn fifo_continues_after_each_capacity_release() {
        let (state, mut rx) = state(1);
        let ids = admit(&state, 4);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        for expected in ids {
            let mut job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(job.id, expected);
            job.journal.take().unwrap().complete_before_dispatch();
            state.job_registry.remove(&job.id);
            state.queue.decrement();
        }
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn startup_scan_and_post_commit_notification_both_feed_work() {
        let (state, mut rx) = state(2);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        tokio::task::yield_now().await;
        admit(&state, 1);
        let job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.id, "job-0");
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(job);
    }

    #[tokio::test]
    async fn restart_after_gallery_publication_recovers_exact_batch_result_without_rerender() {
        let (state, mut rx) = state(1);
        admit(&state, 1);
        let dead_claim = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(dead_claim.row.id, "job-0");
        let output_dir = mold_db::canonical_dir_string(&dead_claim.row.output_dir);
        let published_at_ms = dead_claim.row.created_at_ms;
        state
            .metadata_db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                for filename in [
                    "mold-mock-model-1-original~portrait.png",
                    "mold-mock-model-2-upscaled~portrait.png",
                ] {
                    conn.execute(
                        "INSERT INTO generations
                            (filename, output_dir, created_at_ms, format, model, metadata_json)
                         VALUES (?1, ?2, ?3, 'png', 'mock-model', ?4)",
                        (
                            filename,
                            output_dir.as_str(),
                            published_at_ms,
                            r#"{"job_id":"job-0","seed":7}"#,
                        ),
                    )?;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(
            state
                .queue_journal
                .recover_feeder_runtime()
                .unwrap()
                .claims_cleared,
            1
        );

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let detail = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let detail = mold_db::generation_batches::get_durable(
                    state.metadata_db.as_ref().as_ref().unwrap(),
                    state.queue_journal.owner_uuid().unwrap(),
                    "batch",
                )
                .unwrap()
                .unwrap();
                if detail.children[0].state == "complete" {
                    break detail;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("published output is reconciled without dispatch");

        assert!(
            rx.try_recv().is_err(),
            "the completed child must not rerender"
        );
        assert!(state.queue_journal.list_all().is_empty());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                detail.children[0].result_json.as_deref().unwrap()
            )
            .unwrap(),
            serde_json::json!({
                "filename": "mold-mock-model-2-upscaled~portrait.png",
                "original_filename": "mold-mock-model-1-original~portrait.png",
            })
        );

        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn restart_after_ordinary_archive_before_db_upsert_does_not_rerender() {
        let (mut state, mut rx) = state(1);
        admit(&state, 1);
        let dead_claim = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        archive_output_without_db(
            &state,
            &dead_claim.row.output_dir,
            "mold-mock-model-1~portrait.png",
            "job-0",
        );
        state.gallery_publication_gate = Default::default();
        assert_eq!(
            state
                .metadata_db
                .as_ref()
                .as_ref()
                .unwrap()
                .count()
                .unwrap(),
            0,
            "the test must stop in the archive-before-generations-upsert crash window"
        );
        state.queue_journal.recover_feeder_runtime().unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let detail = await_completed_batch(&state).await;

        assert!(
            rx.try_recv().is_err(),
            "the completed child must not rerender"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                detail.children[0].result_json.as_deref().unwrap()
            )
            .unwrap(),
            serde_json::json!({
                "filename": "mold-mock-model-1~portrait.png",
                "original_filename": null,
            })
        );
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn restart_after_upscale_archives_before_db_upsert_recovers_exact_pair() {
        let (mut state, mut rx) = state(1);
        admit(&state, 1);
        let dead_claim = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        for filename in [
            "mold-mock-model-1-original~portrait.png",
            "mold-mock-model-2-upscaled~portrait.png",
        ] {
            archive_output_without_db(&state, &dead_claim.row.output_dir, filename, "job-0");
        }
        state.gallery_publication_gate = Default::default();
        assert_eq!(
            state
                .metadata_db
                .as_ref()
                .as_ref()
                .unwrap()
                .count()
                .unwrap(),
            0,
            "the test must stop before either best-effort generations upsert"
        );
        state.queue_journal.recover_feeder_runtime().unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let detail = await_completed_batch(&state).await;

        assert!(
            rx.try_recv().is_err(),
            "the completed child must not rerender"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                detail.children[0].result_json.as_deref().unwrap()
            )
            .unwrap(),
            serde_json::json!({
                "filename": "mold-mock-model-2-upscaled~portrait.png",
                "original_filename": "mold-mock-model-1-original~portrait.png",
            })
        );
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn interrupted_upscale_archive_adopts_original_and_does_not_block_later_work() {
        let (mut state, mut rx) = state(1);
        admit(&state, 2);
        let dead_claim = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(dead_claim.row.id, "job-0");
        archive_output_without_db(
            &state,
            &dead_claim.row.output_dir,
            "mold-mock-model-1-original~portrait.png",
            "job-0",
        );
        state.gallery_publication_gate = Default::default();
        assert_eq!(
            state
                .metadata_db
                .as_ref()
                .as_ref()
                .unwrap()
                .count()
                .unwrap(),
            0,
            "the crash window is after archive publication but before the DB upsert"
        );
        recover_runtime(&state).await.unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let detail = await_completed_batch(&state).await;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                detail.children[0].result_json.as_deref().unwrap()
            )
            .unwrap(),
            serde_json::json!({
                "filename": "mold-mock-model-1-original~portrait.png",
                "original_filename": null,
            })
        );

        let later = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("later work must not be poisoned by the interrupted upscale")
            .unwrap();
        assert_eq!(later.id, "job-1");
        assert!(state
            .queue_journal
            .list_all()
            .iter()
            .all(|row| row.id != "job-0"));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(later);
    }

    #[tokio::test]
    async fn committed_archive_lookup_is_scoped_to_the_claimed_output_directory() {
        let (mut state, mut rx) = state(1);
        admit(&state, 1);
        let owned_output = state.config.try_read().unwrap().effective_output_dir();
        let foreign_output = owned_output.parent().unwrap().join("foreign-gallery");
        std::fs::create_dir_all(&foreign_output).unwrap();
        archive_output_without_db(&state, &foreign_output, "foreign.png", "job-0");
        state.gallery_publication_gate = Default::default();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.id, "job-0");
        job.journal.take().unwrap().complete_before_dispatch();
        state.job_registry.remove(&job.id);
        state.queue.decrement();
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[test]
    fn committed_archive_lookup_fails_closed_on_ambiguous_job_outputs() {
        let (mut state, _rx) = state(1);
        let output_dir = state.config.try_read().unwrap().effective_output_dir();
        for filename in ["first.png", "second.png"] {
            archive_output_without_db(&state, &output_dir, filename, "job-0");
        }
        state.gallery_publication_gate = Default::default();

        let _reader = state.gallery_publication_gate.blocking_read();
        let error = crate::batch_transaction::find_completed_output_in_committed_archive(
            &state.gallery_publication_gate,
            &output_dir,
            "job-0",
        )
        .unwrap_err();
        assert!(error.is_invalid_authority());
        assert!(format!("{error:#}").contains("ambiguous"));
    }

    #[test]
    fn committed_archive_lookup_fails_closed_on_malformed_authority() {
        let (mut state, _rx) = state(1);
        let output_dir = state.config.try_read().unwrap().effective_output_dir();
        archive_output_without_db(&state, &output_dir, "print.png", "job-0");
        state.gallery_publication_gate = Default::default();
        std::fs::write(
            output_dir
                .join(crate::batch_transaction::TRANSACTION_DIR)
                .join(crate::gallery_authority::authority_dir_name())
                .join("generation.json"),
            b"{",
        )
        .unwrap();

        let _reader = state.gallery_publication_gate.blocking_read();
        assert!(
            crate::batch_transaction::find_completed_output_in_committed_archive(
                &state.gallery_publication_gate,
                &output_dir,
                "job-0",
            )
            .unwrap_err()
            .is_invalid_authority()
        );
    }

    #[tokio::test]
    async fn startup_recovery_feeds_an_interrupted_direct_generation() {
        let (state, mut rx) = state(1);
        let request = request("legacy stream interrupted");
        let output = state.config.try_read().unwrap().effective_output_dir();
        let direct_ticket = state
            .queue_journal
            .record(JournalAdmission {
                id: "legacy-direct",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();
        assert!(state.queue_journal.claim_next_feeder().unwrap().is_none());

        state.queue_journal.retain_all();
        drop(direct_ticket);
        assert_eq!(
            state
                .queue_journal
                .recover_feeder_runtime()
                .unwrap()
                .claims_cleared,
            1
        );

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut replayed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replayed.id, "legacy-direct");
        replayed.journal.take().unwrap().complete_before_dispatch();
        assert!(state.queue_journal.list_all().is_empty());
        state.job_registry.remove(&replayed.id);
        state.queue.decrement();

        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn production_recovery_charges_only_dead_runtime_claims_and_holds_exhausted_work() {
        let (state, _rx) = state(1);
        admit(&state, 2);
        let cap = state.queue_journal.max_replay_seen();
        let dead_claim = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(dead_claim.row.id, "job-0");

        for boot in 1..=cap + 1 {
            recover_runtime(&state).await.unwrap();
            let recovered = state
                .queue_journal
                .list_all()
                .into_iter()
                .find(|row| row.id == "job-0")
                .unwrap();
            assert_eq!(recovered.replay_seen, boot);
        }

        let rows = state.queue_journal.list_all();
        let exhausted = rows.iter().find(|row| row.id == "job-0").unwrap();
        assert_eq!(
            exhausted.state,
            mold_db::generation_queue::QueueRowState::Held
        );
        assert!(exhausted
            .held_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("replayed by")));
        let untouched = rows.iter().find(|row| row.id == "job-1").unwrap();
        assert_eq!(untouched.replay_seen, 0);

        let next = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(next.row.id, "job-1");
        let batch = state.queue_journal.generation_batch("batch").unwrap();
        assert_eq!(batch.children[0].state, "held");
        assert_eq!(batch.children[1].state, "accepted");
    }

    #[tokio::test]
    async fn post_recovery_direct_admission_keeps_its_live_runtime_token() {
        let (state, mut rx) = state(2);
        let request = request("interrupted before startup barrier");
        let output = state.config.try_read().unwrap().effective_output_dir();
        let interrupted_ticket = state
            .queue_journal
            .record(JournalAdmission {
                id: "interrupted-direct",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();
        state.queue_journal.retain_all();
        drop(interrupted_ticket);

        let recovered = recover_runtime(&state).await.unwrap();
        assert_eq!(recovered.claims_cleared, 1);

        // This admission represents an HTTP request accepted only after the
        // startup barrier has completed. The feeder must never run recovery
        // again and erase this live submitter's ownership token.
        let live_ticket = state
            .queue_journal
            .record(JournalAdmission {
                id: "post-barrier-direct",
                request: &request,
                output_dir: Some(&output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::MetadataOnly,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let mut replayed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(replayed.id, "interrupted-direct");
        assert_eq!(
            live_ticket.claim_dispatch(),
            crate::queue_journal::DispatchClaim::Granted,
            "post-barrier admission must remain owned by its live submitter"
        );

        replayed.journal.take().unwrap().complete_before_dispatch();
        live_ticket.discard();
        state.job_registry.remove(&replayed.id);
        state.queue.decrement();
        shutdown.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn missing_stable_device_pin_resumes_on_auto_not_recorded_ordinal() {
        let (state, mut rx) = state(1);
        let mut pinned = request("mixed placement");
        pinned.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Cpu,
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::device("cuda:missing-this-boot"),
                vae: mold_core::DeviceRef::gpu(7),
                clip_l: Some(mold_core::DeviceRef::Cpu),
                clip_g: Some(mold_core::DeviceRef::Auto),
                t5: Some(mold_core::DeviceRef::device("cuda:also-stale")),
                qwen: Some(mold_core::DeviceRef::gpu(9)),
            }),
        });
        let output = state.config.try_read().unwrap().effective_output_dir();
        state
            .queue_journal
            .record_batch(BatchJournalAdmission {
                id: "batch",
                client_batch_id: "client",
                request_sha256: "sha",
                children: &[JournalAdmission {
                    id: "job-0",
                    request: &pinned,
                    output_dir: Some(&output),
                    target_gpu: None,
                    target_device_id: None,
                    completion_payload: SseCompletionPayload::MetadataOnly,
                    batch_child: false,
                    carries_reference_authority: false,
                }],
            })
            .unwrap();
        mold_db::generation_queue::set_target_gpu(
            state.metadata_db.as_ref().as_ref().unwrap(),
            "job-0",
            Some(7),
            Some("cuda:missing-this-boot"),
            2,
        )
        .unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.id, "job-0");
        assert_eq!(state.job_registry.target_gpu("job-0"), Some(None));
        assert_eq!(
            state
                .gpu_pool
                .resolve_explicit_placement_gpu(job.request.placement.as_ref()),
            Ok(None),
            "the hydrated job must not retain an ordinal or stable accelerator pin"
        );
        let placement = job.request.placement.as_ref().unwrap();
        assert_eq!(placement.text_encoders, mold_core::DeviceRef::Cpu);
        let advanced = placement.advanced.as_ref().unwrap();
        assert_eq!(advanced.transformer, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.vae, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.clip_l, Some(mold_core::DeviceRef::Cpu));
        assert_eq!(advanced.clip_g, Some(mold_core::DeviceRef::Auto));
        assert_eq!(advanced.t5, Some(mold_core::DeviceRef::Auto));
        assert_eq!(advanced.qwen, Some(mold_core::DeviceRef::Auto));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(job);
    }

    #[tokio::test]
    async fn cancellation_that_wins_before_registry_handoff_skips_the_exact_child() {
        let (state, mut rx) = state(1);
        admit(&state, 2);
        let claimed = state.queue_journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(claimed.row.id, "job-0");
        state.queue_journal.cancel_id("job-0").unwrap();
        recover_runtime(&state).await.unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let next = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(next.id, "job-1");
        let detail = state.queue_journal.generation_batch("batch").unwrap();
        assert_eq!(detail.children[0].state, "cancelled");
        assert_eq!(detail.children[1].state, "accepted");
        assert!(state
            .queue_journal
            .list_all()
            .iter()
            .all(|row| row.id != "job-0"));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(next);
    }

    #[tokio::test]
    async fn blocking_durable_cancel_check_does_not_hold_scheduler_mutation_fence() {
        let (state, _rx) = state(1);
        admit(&state, 1);
        let claim = state
            .queue_journal
            .claim_next_feeder()
            .unwrap()
            .expect("test row is claimed by the feeder handoff");
        let mold_db::generation_queue::QueueClaim {
            row, claim_token, ..
        } = claim;
        let mut request = request("fence");
        let registration = register_claimed_runtime(&state, &row, &claim_token, &mut request)
            .await
            .unwrap();
        let ClaimRegistration::Registered {
            publication_guard, ..
        } = registration
        else {
            panic!("the exact claim belongs to the runtime window");
        };
        drop(publication_guard);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<bool>();
        let mut durable_check = Box::pin(async { release_rx.await.unwrap() });

        assert!(
            futures::poll!(durable_check.as_mut()).is_pending(),
            "the synthetic durable lookup remains blocked"
        );
        let scheduler_guard = state
            .scheduler_mutation_fence
            .try_lock()
            .expect("durable cancellation lookup must not own the scheduler mutation fence");
        drop(scheduler_guard);
        release_tx.send(false).unwrap();
        let requested = durable_check.await;
        assert!(!requested);
        state.job_registry.remove(&row.id);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocked_durable_order_read_never_blocks_scheduler_grant_or_cancellation() {
        let (state, _rx) = state(1);
        admit(&state, 1);
        let claim = state
            .queue_journal
            .claim_next_feeder()
            .unwrap()
            .expect("test row is claimed by the feeder handoff");
        let mold_db::generation_queue::QueueClaim {
            row, claim_token, ..
        } = claim;
        let row_id = row.id.clone();
        state.job_registry.register("grant-live", "model-grant");
        state.job_registry.register("cancel-live", "model-cancel");

        // Hold the journal's real connection mutex. The handoff can enter its
        // durable transition, but the bounded exact-claim order read cannot
        // complete until this blocker is released.
        let locked_db = state.metadata_db.clone();
        let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let blocker = tokio::task::spawn_blocking(move || {
            locked_db.as_ref().as_ref().unwrap().with_conn(|_| {
                locked_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });
        locked_rx.await.unwrap();

        let handoff_state = state.clone();
        let handoff = tokio::spawn(async move {
            let mut request = request("blocked order");
            register_claimed_runtime(&handoff_state, &row, &claim_token, &mut request).await
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            while !state.queue_journal.durable_transition_is_locked() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("handoff enters the durable transition before waiting on SQLite");
        assert!(!handoff.is_finished(), "the order read remains DB-blocked");

        let scheduler = tokio::time::timeout(
            Duration::from_secs(2),
            state.scheduler_mutation_fence.lock(),
        )
        .await
        .expect("blocked order lookup must not own the scheduler fence");
        state.job_registry.reorder_queued("grant-live", 0).unwrap();
        state.job_registry.cancel_queued("cancel-live").unwrap();
        state
            .job_registry
            .dispatch_if_queued("grant-live", 0, (), |_| Ok(()))
            .unwrap();
        drop(scheduler);

        release_tx.send(()).unwrap();
        blocker.await.unwrap().unwrap();
        assert!(matches!(
            handoff.await.unwrap().unwrap(),
            ClaimRegistration::Registered { .. }
        ));
        assert_eq!(
            state.job_registry.entry("grant-live").unwrap().state,
            crate::job_registry::JobLifecycle::Running
        );
        assert!(state.job_registry.entry("cancel-live").is_none());
        state.job_registry.remove(&row_id);
    }

    #[tokio::test]
    async fn cancellation_before_dispatch_settles_the_exact_claim_and_keeps_later_work() {
        let (state, mut rx) = state(1);
        admit(&state, 2);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        state.job_registry.cancel_queued(&first.id).unwrap();
        state.queue_journal.cancel_id(&first.id).unwrap();
        let first_id = first.id.clone();
        drop(first);
        state.queue.decrement();

        let second = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.id, "job-1");
        let detail = state.queue_journal.generation_batch("batch").unwrap();
        assert_eq!(detail.children[0].job_id, first_id);
        assert_eq!(detail.children[0].state, "cancelled");
        assert!(state
            .queue_journal
            .list_all()
            .iter()
            .all(|row| row.id != first_id));

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(second);
    }

    #[tokio::test]
    async fn shutdown_keeps_claimed_and_unclaimed_rows_for_restart_recovery() {
        let (state, mut rx) = state(1);
        admit(&state, 3);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = spawn(state.clone(), shutdown.clone());
        let claimed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(claimed);
        assert_eq!(state.queue_journal.list_all().len(), 3);
        let report = state.queue_journal.recover_feeder_runtime().unwrap();
        assert_eq!(report.claims_cleared, 1);
        assert_eq!(report.running_requeued, 0);
        let ids = state
            .queue_journal
            .list_all()
            .into_iter()
            .map(|row| row.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["job-0", "job-1", "job-2"]);
    }

    #[tokio::test]
    async fn recoverable_persistence_failure_does_not_stop_the_only_feeder() {
        let (state, mut rx) = state(1);
        admit(&state, 1);
        state.queue_journal.fail_completion_lookup_for_tests();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn(run_with_retry_delay(
            state.clone(),
            shutdown.clone(),
            Duration::ZERO,
            Arc::new(tokio::sync::Notify::new()),
        ));
        let job = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("the feeder must retry after a recoverable persistence failure")
            .expect("the runtime queue remains open");
        assert_eq!(job.id, "job-0");
        assert!(
            !handle.is_finished(),
            "a recoverable row failure must not terminate the sole feeder task"
        );

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(job);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_claim_release_retries_in_process_and_hydrates_without_restart() {
        let (state, mut rx) = state(1);
        admit(&state, 1);
        state.queue_journal.fail_completion_lookup_for_tests();
        state.queue_journal.fail_claim_release_for_tests();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn(run_with_retry_delay(
            state.clone(),
            shutdown.clone(),
            Duration::ZERO,
            Arc::new(tokio::sync::Notify::new()),
        ));
        while state
            .queue_journal
            .claim_release_failure_pending_for_tests()
        {
            tokio::task::yield_now().await;
        }
        tokio::task::yield_now().await;
        assert_eq!(
            state.queue.pending(),
            0,
            "claim-release backoff must not retain a runtime queue reservation"
        );
        assert!(rx.try_recv().is_err());

        let job = tokio::time::timeout(
            mold_db::METADATA_DB_BUSY_TIMEOUT + Duration::from_secs(1),
            rx.recv(),
        )
        .await
        .expect("the same feeder runtime must retry the claim release")
        .expect("the same feeder runtime must hydrate after claim release recovers");
        assert_eq!(job.id, "job-0");
        assert!(
            !handle.is_finished(),
            "a failed claim release must not terminate the sole feeder"
        );

        state.queue_journal.retain_all();
        shutdown.cancel();
        handle.await.unwrap();
        drop(job);
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_during_claim_release_retry_leaves_startup_recoverable_authority() {
        let (state, _rx) = state(1);
        admit(&state, 1);
        state.queue_journal.fail_completion_lookup_for_tests();
        state.queue_journal.fail_claim_release_for_tests();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn(run_with_retry_delay(
            state.clone(),
            shutdown.clone(),
            Duration::ZERO,
            Arc::new(tokio::sync::Notify::new()),
        ));
        let claimed_token = loop {
            let token = state
                .metadata_db
                .as_ref()
                .as_ref()
                .unwrap()
                .with_conn(|conn| {
                    conn.query_row(
                        "SELECT claim_token FROM generation_queue WHERE id = 'job-0'",
                        [],
                        |row| row.get::<_, Option<String>>(0),
                    )
                    .map_err(anyhow::Error::from)
                })
                .unwrap();
            if let Some(token) = token {
                break token;
            }
            tokio::task::yield_now().await;
        };
        while state
            .queue_journal
            .claim_release_failure_pending_for_tests()
        {
            tokio::task::yield_now().await;
        }

        shutdown.cancel();
        handle.await.unwrap();
        let retained_token = state
            .metadata_db
            .as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT claim_token FROM generation_queue WHERE id = 'job-0'",
                    [],
                    |row| row.get::<_, Option<String>>(0),
                )
                .map_err(anyhow::Error::from)
            })
            .unwrap();
        assert_eq!(retained_token.as_deref(), Some(claimed_token.as_str()));

        let recovery = state.queue_journal.recover_feeder_runtime().unwrap();
        assert_eq!(recovery.claims_cleared, 1);
        assert_eq!(state.queue_journal.list_all()[0].id, "job-0");
    }

    #[tokio::test]
    async fn drained_pass_ignores_capacity_wake_from_its_speculative_reservation() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let durable = Arc::new(tokio::sync::Notify::new());
        let capacity = Arc::new(tokio::sync::Notify::new());
        capacity.notify_one();

        let waiting = tokio::spawn({
            let shutdown = shutdown.clone();
            let durable = durable.clone();
            let capacity = capacity.clone();
            async move {
                wait_for_next_pass(
                    FeederStop::Drained,
                    Duration::ZERO,
                    &shutdown,
                    durable.notified(),
                    capacity.notified(),
                )
                .await
            }
        });
        tokio::task::yield_now().await;
        assert!(
            !waiting.is_finished(),
            "an idle feeder must not consume the slot release it produced itself"
        );

        durable.notify_one();
        assert_eq!(waiting.await.unwrap(), FeederControl::Continue);
    }

    #[tokio::test(start_paused = true)]
    async fn recoverable_pass_ignores_self_wakes_until_the_database_retry_window() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let durable = Arc::new(tokio::sync::Notify::new());
        let capacity = Arc::new(tokio::sync::Notify::new());
        durable.notify_one();
        capacity.notify_one();

        let waiting = tokio::spawn({
            let shutdown = shutdown.clone();
            let durable = durable.clone();
            let capacity = capacity.clone();
            async move {
                wait_for_next_pass(
                    FeederStop::RecoverableFailure,
                    mold_db::METADATA_DB_BUSY_TIMEOUT,
                    &shutdown,
                    durable.notified(),
                    capacity.notified(),
                )
                .await
            }
        });
        tokio::task::yield_now().await;
        assert!(
            !waiting.is_finished(),
            "a retained claim's notifications must not create a retry spin"
        );

        tokio::time::advance(mold_db::METADATA_DB_BUSY_TIMEOUT).await;
        assert_eq!(waiting.await.unwrap(), FeederControl::Continue);
    }
}
