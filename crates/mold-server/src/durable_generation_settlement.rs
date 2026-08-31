//! One ordering boundary for durable generation terminal/blocked state.
//!
//! An accepted job's observer must never resolve before SQLite reflects the
//! disposition the observer is about to report. Both the Tokio single-worker
//! path and dedicated GPU owner threads pass through this module.

use crate::durable_disposition::DurableDisposition;
use crate::queue_journal::{QueueTicket, RetainOutcome};
use crate::state::SseMessage;
use mold_core::SseErrorEvent;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SettlementOutcome {
    /// No durable ticket was attached; preserve legacy observer behavior.
    Untracked,
    /// SQLite committed the requested terminal/blocked transition.
    Settled,
    /// User cancellation committed instead of the requested terminal result.
    Cancelled,
    /// The claim is intentionally replayable, or its exact transition could
    /// not be committed within the bounded owner-thread retry budget.
    Retained,
}

impl SettlementOutcome {
    pub(crate) fn is_retained(self) -> bool {
        self == Self::Retained
    }

    pub(crate) fn is_cancelled(self) -> bool {
        self == Self::Cancelled
    }
}

/// Build the observer frame from the durable transition that actually won.
/// Callers still resolve their raw result channel with the original error;
/// direct raw routes reconcile that result against the authoritative row.
pub(crate) fn terminal_error_event(
    settlement: SettlementOutcome,
    message: impl Into<String>,
) -> SseErrorEvent {
    let message = message.into();
    match settlement {
        SettlementOutcome::Cancelled => SseErrorEvent::cancelled(message),
        SettlementOutcome::Retained => SseErrorEvent::retained(message),
        SettlementOutcome::Untracked | SettlementOutcome::Settled => SseErrorEvent::failed(message),
    }
}

/// The three things a terminal disposition must reach, in this order: SQLite,
/// then the SSE observer, then the raw result channel.
///
/// Every worker path used to spell that order out for itself — 39 copies of
/// the same twelve lines — and two of them had drifted (the scheduler's
/// refusals reported before they persisted, which is exactly what this
/// module's opening paragraph forbids).
pub(crate) struct SettlementChannels {
    pub journal: Option<QueueTicket>,
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    pub result_tx:
        Option<tokio::sync::oneshot::Sender<Result<crate::state::GenerationJobResult, String>>>,
}

impl SettlementChannels {
    /// Report one terminal outcome to both observers. Called only after the
    /// durable transition has committed.
    fn report(&mut self, event: SseErrorEvent, message: String) {
        if let Some(tx) = self.progress_tx.as_ref() {
            let _ = tx.send(SseMessage::Error(event));
        }
        if let Some(tx) = self.result_tx.take() {
            let _ = tx.send(Err(message));
        }
    }

    /// Deliver a successful render. Separate from [`Self::report`] because a
    /// completion frame is built by the caller from the response it holds —
    /// and only when there is an observer, because building one copies the
    /// print and a non-streaming caller has nobody to read it.
    pub(crate) fn complete(
        &mut self,
        completion: Option<SseMessage>,
        result: crate::state::GenerationJobResult,
    ) {
        if let (Some(tx), Some(completion)) = (self.progress_tx.as_ref(), completion) {
            let _ = tx.send(completion);
        }
        if let Some(tx) = self.result_tx.take() {
            let _ = tx.send(Ok(result));
        }
    }
}

/// One job's terminal channels, however that job is shaped.
pub(crate) trait IntoSettlementChannels {
    fn into_settlement_channels(self) -> SettlementChannels;
}

impl IntoSettlementChannels for SettlementChannels {
    fn into_settlement_channels(self) -> SettlementChannels {
        self
    }
}

/// What SQLite records versus what the observer reads.
///
/// `Same` is the ordinary case. `Split` exists because two refusals owe the
/// operator a different sentence than the user: the row records why it is
/// parked, the observer reads what happened to their print.
pub(crate) enum FailureMessage {
    Same(String),
    Split { durable: String, observer: String },
}

impl FailureMessage {
    fn split(self) -> (String, String) {
        match self {
            Self::Same(message) => (message.clone(), message),
            Self::Split { durable, observer } => (durable, observer),
        }
    }
}

impl From<String> for FailureMessage {
    fn from(message: String) -> Self {
        Self::Same(message)
    }
}

impl From<&str> for FailureMessage {
    fn from(message: &str) -> Self {
        Self::Same(message.to_string())
    }
}

/// Persist, then report. Consumes the job: every caller already owns it, and
/// a borrowed view could not move the oneshot sender.
pub(crate) fn fail_blocking(
    job: impl IntoSettlementChannels,
    disposition: DurableDisposition,
    message: impl Into<FailureMessage>,
) -> SettlementOutcome {
    let mut channels = job.into_settlement_channels();
    let (durable, observer) = message.into().split();
    let settlement = settle_blocking(&mut channels.journal, disposition, &durable);
    channels.report(terminal_error_event(settlement, observer.clone()), observer);
    settlement
}

/// The async twin. The durable transition still runs on the blocking pool —
/// SQLite must never be committed from a runtime worker thread.
pub(crate) async fn fail_async(
    job: impl IntoSettlementChannels,
    disposition: DurableDisposition,
    message: impl Into<FailureMessage>,
) -> SettlementOutcome {
    let mut channels = job.into_settlement_channels();
    let (durable, observer) = message.into().split();
    let settlement = settle_async(&mut channels.journal, disposition, &durable).await;
    channels.report(terminal_error_event(settlement, observer.clone()), observer);
    settlement
}

/// Explicit user cancellation is terminal and discards the row; a shutdown or
/// attempt cancellation retains it for the next boot.
pub(crate) fn finish_cancelled_blocking(
    job: impl IntoSettlementChannels,
    model: &str,
    user_requested: bool,
) {
    let mut channels = job.into_settlement_channels();
    if let Some(ticket) = channels.journal.take() {
        if user_requested {
            ticket.discard();
        } else {
            channels.journal = Some(ticket);
            settle_blocking(
                &mut channels.journal,
                DurableDisposition::Retain,
                CANCEL_RETENTION_REASON,
            );
        }
    }
    let (event, message) = cancellation_report(model, user_requested);
    channels.report(event, message);
}

pub(crate) async fn finish_cancelled_async(
    job: impl IntoSettlementChannels,
    model: &str,
    user_requested: bool,
) {
    let mut channels = job.into_settlement_channels();
    if let Some(ticket) = channels.journal.take() {
        if user_requested {
            let _ = tokio::task::spawn_blocking(move || ticket.discard()).await;
        } else {
            channels.journal = Some(ticket);
            settle_async(
                &mut channels.journal,
                DurableDisposition::Retain,
                CANCEL_RETENTION_REASON,
            )
            .await;
        }
    }
    let (event, message) = cancellation_report(model, user_requested);
    channels.report(event, message);
}

const CANCEL_RETENTION_REASON: &str = "server shutdown interrupted generation";

fn cancellation_report(model: &str, user_requested: bool) -> (SseErrorEvent, String) {
    if user_requested {
        let message = "Cancelled".to_string();
        (SseErrorEvent::cancelled(message.clone()), message)
    } else {
        let message = crate::gpu_worker::shutdown_retention_user_message(model);
        (SseErrorEvent::retained(message.clone()), message)
    }
}

/// One home for the durable-media hydration failure both worker paths report.
pub(crate) fn fail_hydration_blocking(
    job: impl IntoSettlementChannels,
    job_id: &str,
    error: crate::queue_media_runtime::DeferredQueueMediaError,
) {
    let (disposition, message) = hydration_failure(job_id, error);
    fail_blocking(job, disposition, message);
}

pub(crate) async fn fail_hydration_async(
    job: impl IntoSettlementChannels,
    job_id: &str,
    error: crate::queue_media_runtime::DeferredQueueMediaError,
) {
    let (disposition, message) = hydration_failure(job_id, error);
    fail_async(job, disposition, message).await;
}

fn hydration_failure(
    job_id: &str,
    error: crate::queue_media_runtime::DeferredQueueMediaError,
) -> (DurableDisposition, FailureMessage) {
    let disposition = error.disposition();
    tracing::error!(job = %job_id, %error, "durable queue-media hydration failed");
    let durable = match disposition {
        DurableDisposition::Hold { .. } => "durable queue-media validation failed",
        DurableDisposition::Retain => "durable queue-media hydration must be retried after restart",
    };
    (
        disposition,
        FailureMessage::Split {
            durable: durable.to_string(),
            observer: error.public_message().to_string(),
        },
    )
}

/// The dispatch-claim ladder's two refusals, shared by both dispatchers.
pub(crate) fn refuse_exhausted_dispatch_blocking(
    job: impl IntoSettlementChannels,
    model: &str,
    attempts: u32,
    cap: u32,
) {
    fail_blocking(
        job,
        DurableDisposition::Hold { retryable: false },
        exhausted_dispatch_message(model, attempts, cap),
    );
}

pub(crate) async fn refuse_exhausted_dispatch_async(
    job: impl IntoSettlementChannels,
    model: &str,
    attempts: u32,
    cap: u32,
) {
    fail_async(
        job,
        DurableDisposition::Hold { retryable: false },
        exhausted_dispatch_message(model, attempts, cap),
    )
    .await;
}

fn exhausted_dispatch_message(model: &str, attempts: u32, cap: u32) -> FailureMessage {
    tracing::error!(
        %model,
        attempts,
        cap,
        "held an exhausted durable queue row"
    );
    FailureMessage::Split {
        durable: "dispatch attempts exhausted".to_string(),
        observer: format!(
            "'{model}' was started {attempts} times without finishing (limit {cap}); \
             it is held for review instead of being retried"
        ),
    }
}

/// A stale claim belongs to a newer owner: refuse without any durable
/// transition, and let the ticket's ordinary drop answer for the row.
pub(crate) fn refuse_fenced_dispatch(job: impl IntoSettlementChannels, job_id: &str) {
    let mut channels = job.into_settlement_channels();
    tracing::warn!(job = %job_id, "refused a stale durable feeder claim");
    let message = "durable generation claim is stale; refusing dispatch".to_string();
    channels.report(SseErrorEvent::failed(message.clone()), message);
}

/// Settle a finished render on what actually reached the gallery, then report
/// the cancelled and retained cases.
///
/// `Ok(())` means the caller may send its completion frame and `Ok(result)`.
/// `Err(outcome)` means the observers have already been resolved. The save
/// helpers answer `None` when publication fails — an unwritable directory, a
/// full disk, a refused archive — and for a replayed job the gallery file IS
/// the delivery, so clearing the row there would lose the generation outright.
pub(crate) fn settle_publication_blocking(
    channels: &mut SettlementChannels,
    job_id: &str,
    output_dir: Option<&std::path::Path>,
    registry: &crate::job_registry::SharedJobRegistry,
    saved: &crate::queue::SavedOutputNames,
    response: &mold_core::GenerateResponse,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> Result<(), SettlementOutcome> {
    let settlement = match publication_result(job_id, output_dir, saved, response) {
        Some(result_json) => {
            match handoff_gallery_media_blocking(&channels.journal, output_dir, gallery_gate) {
                Ok(()) => settle_completion_blocking(&mut channels.journal, &result_json),
                Err(error) => {
                    tracing::error!(job = %job_id, %error, "retaining completed generation after source-media handoff failure");
                    settle_blocking(
                        &mut channels.journal,
                        DurableDisposition::Hold { retryable: true },
                        SOURCE_MEDIA_HANDOFF_REASON,
                    )
                }
            }
        }
        None => settle_blocking(
            &mut channels.journal,
            DurableDisposition::Hold { retryable: true },
            UNPUBLISHED_OUTPUT_REASON,
        ),
    };
    finish_publication(channels, job_id, registry, settlement)
}

pub(crate) async fn settle_publication_async(
    channels: &mut SettlementChannels,
    job_id: &str,
    output_dir: Option<&std::path::Path>,
    registry: &crate::job_registry::SharedJobRegistry,
    saved: &crate::queue::SavedOutputNames,
    response: &mold_core::GenerateResponse,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> Result<(), SettlementOutcome> {
    let settlement = match publication_result(job_id, output_dir, saved, response) {
        Some(result_json) => {
            // This is the same short, file-first authority transition used by
            // blocking GPU owners. It must finish before the ticket moves to
            // the blocking SQLite settlement task below.
            let handoff =
                handoff_gallery_media_blocking(&channels.journal, output_dir, gallery_gate);
            match handoff {
                Ok(()) => settle_completion_async(&mut channels.journal, &result_json).await,
                Err(error) => {
                    tracing::error!(job = %job_id, %error, "retaining completed generation after source-media handoff failure");
                    settle_async(
                        &mut channels.journal,
                        DurableDisposition::Hold { retryable: true },
                        SOURCE_MEDIA_HANDOFF_REASON,
                    )
                    .await
                }
            }
        }
        None => {
            settle_async(
                &mut channels.journal,
                DurableDisposition::Hold { retryable: true },
                UNPUBLISHED_OUTPUT_REASON,
            )
            .await
        }
    };
    finish_publication(channels, job_id, registry, settlement)
}

const UNPUBLISHED_OUTPUT_REASON: &str = "the generated output could not be saved to the gallery";
const SOURCE_MEDIA_HANDOFF_REASON: &str =
    "the generated output was saved but its retained source media could not be committed";

fn handoff_gallery_media_blocking(
    ticket: &Option<QueueTicket>,
    output_dir: Option<&std::path::Path>,
    gate: &crate::batch_transaction::GalleryPublicationGate,
) -> anyhow::Result<()> {
    match (ticket.as_ref(), output_dir) {
        (Some(ticket), Some(output_dir)) => ticket.handoff_media_to_gallery(output_dir, gate),
        _ => Ok(()),
    }
}

fn publication_result(
    job_id: &str,
    output_dir: Option<&std::path::Path>,
    saved: &crate::queue::SavedOutputNames,
    response: &mold_core::GenerateResponse,
) -> Option<String> {
    if saved.output.is_some() {
        return Some(saved.terminal_json(response));
    }
    tracing::error!(
        job = %job_id,
        dir = ?output_dir,
        "generation finished but its output could not be saved; \
         holding the queue row for review"
    );
    None
}

fn finish_publication(
    channels: &mut SettlementChannels,
    job_id: &str,
    registry: &crate::job_registry::SharedJobRegistry,
    settlement: SettlementOutcome,
) -> Result<(), SettlementOutcome> {
    registry.finish_completion(job_id);
    if settlement.is_cancelled() {
        let message = "Cancelled".to_string();
        channels.report(terminal_error_event(settlement, message.clone()), message);
        return Err(settlement);
    }
    if settlement.is_retained() {
        let message =
            "generation output is retained for durable reconciliation after restart".to_string();
        channels.report(SseErrorEvent::retained(message.clone()), message);
        return Err(settlement);
    }
    Ok(())
}

const MAX_EXACT_ATTEMPTS: usize = 3;

fn settle_one(
    ticket: QueueTicket,
    disposition: DurableDisposition,
    message: &str,
) -> RetainOutcome {
    match disposition {
        DurableDisposition::Retain => ticket.retain(),
        _ if ticket.retention_requested() => ticket.retain(),
        DurableDisposition::Hold { retryable } => ticket.hold_exact(message, retryable),
    }
}

/// Complete the requested durable transition before a worker publishes its
/// observer result. A retry outcome always keeps and reuses the token-owning
/// ticket; shutdown/fatal-CUDA retention fences take precedence over a hold.
pub(crate) fn settle_blocking(
    ticket: &mut Option<QueueTicket>,
    disposition: DurableDisposition,
    message: &str,
) -> SettlementOutcome {
    let Some(mut owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    for attempt in 1..=MAX_EXACT_ATTEMPTS {
        let job_id = owned.id().to_string();
        let effective_retain =
            disposition == DurableDisposition::Retain || owned.retention_requested();
        match settle_one(owned, disposition, message) {
            RetainOutcome::Released | RetainOutcome::Stale => {
                return if effective_retain {
                    SettlementOutcome::Retained
                } else {
                    SettlementOutcome::Settled
                };
            }
            RetainOutcome::Cancelled => return SettlementOutcome::Cancelled,
            RetainOutcome::Retry { ticket, error } => {
                tracing::warn!(
                    job = %job_id,
                    %error,
                    ?disposition,
                    attempt,
                    "durable generation settlement will retry before observer delivery"
                );
                owned = ticket;
            }
        }
    }
    tracing::error!(
        job = %owned.id(),
        ?disposition,
        attempts = MAX_EXACT_ATTEMPTS,
        "durable generation settlement exhausted its bounded retry budget; retaining for restart"
    );
    // `RetainOutcome::Retry` made this ticket's drop inert. Leaving the exact
    // claim in SQLite lets startup recovery reconcile it without pinning a GPU
    // owner or process shutdown forever.
    drop(owned);
    SettlementOutcome::Retained
}

pub(crate) fn settle_completion_blocking(
    ticket: &mut Option<QueueTicket>,
    result_json: &str,
) -> SettlementOutcome {
    let Some(mut owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    for attempt in 1..=MAX_EXACT_ATTEMPTS {
        let job_id = owned.id().to_string();
        match owned.complete_exact_with_result(Some(result_json)) {
            RetainOutcome::Released | RetainOutcome::Stale => return SettlementOutcome::Settled,
            RetainOutcome::Cancelled => return SettlementOutcome::Cancelled,
            RetainOutcome::Retry { ticket, error } => {
                tracing::warn!(
                    job = %job_id,
                    %error,
                    attempt,
                    "durable completion commit will retry before success observer delivery"
                );
                owned = ticket;
            }
        }
    }
    tracing::error!(
        job = %owned.id(),
        attempts = MAX_EXACT_ATTEMPTS,
        "durable completion commit exhausted its bounded retry budget; retaining for restart"
    );
    drop(owned);
    SettlementOutcome::Retained
}

pub(crate) async fn settle_async(
    ticket: &mut Option<QueueTicket>,
    disposition: DurableDisposition,
    message: &str,
) -> SettlementOutcome {
    let Some(owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    let message = message.to_string();
    tokio::task::spawn_blocking(move || {
        let mut ticket = Some(owned);
        settle_blocking(&mut ticket, disposition, &message)
    })
    .await
    .expect("durable generation settlement worker panicked")
}

pub(crate) async fn settle_completion_async(
    ticket: &mut Option<QueueTicket>,
    result_json: &str,
) -> SettlementOutcome {
    let Some(owned) = ticket.take() else {
        return SettlementOutcome::Untracked;
    };
    let result_json = result_json.to_string();
    tokio::task::spawn_blocking(move || {
        let mut ticket = Some(owned);
        settle_completion_blocking(&mut ticket, &result_json)
    })
    .await
    .expect("durable completion settlement worker panicked")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{GenerationJobResult, SseCompletionPayload};
    use std::sync::Arc;

    fn request() -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "settlement",
            "model": "mock-model",
            "width": 512,
            "height": 512,
            "steps": 1,
            "batch_size": 1,
            "output_format": "png"
        }))
        .unwrap()
    }

    struct Harness {
        journal: Arc<crate::queue_journal::QueueJournal>,
        db: Arc<Option<mold_db::MetadataDb>>,
        progress_rx: tokio::sync::mpsc::UnboundedReceiver<SseMessage>,
        result_rx: tokio::sync::oneshot::Receiver<Result<GenerationJobResult, String>>,
        channels: SettlementChannels,
    }

    impl Harness {
        fn new(id: &str) -> Self {
            let root = tempfile::tempdir().unwrap().keep();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            let journal = Arc::new(crate::queue_journal::QueueJournal::new(
                db.clone(),
                Some(&root),
                "settlement-test",
            ));
            let request = request();
            journal
                .record_batch(crate::queue_journal::BatchJournalAdmission {
                    id: "batch",
                    client_batch_id: "client",
                    request_sha256: "sha",
                    children: &[crate::queue_journal::JournalAdmission {
                        id,
                        request: &request,
                        output_dir: Some(root.as_path()),
                        target_gpu: None,
                        target_device_id: None,
                        completion_payload: SseCompletionPayload::MetadataOnly,
                        batch_child: false,
                    }],
                })
                .unwrap();
            let claim = journal.claim_next_feeder().unwrap().unwrap();
            let ticket = journal.attach_claimed(id, claim.claim_token);
            let (progress_tx, progress_rx) = tokio::sync::mpsc::unbounded_channel();
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();
            Self {
                journal,
                db,
                progress_rx,
                result_rx,
                channels: SettlementChannels {
                    journal: Some(ticket),
                    progress_tx: Some(progress_tx),
                    result_tx: Some(result_tx),
                },
            }
        }

        fn row(&self, id: &str) -> Option<mold_db::generation_queue::GenerationQueueRow> {
            mold_db::generation_queue::get(self.db.as_ref().as_ref().unwrap(), id).unwrap()
        }

        fn child(&self, id: &str) -> mold_db::generation_batches::DurableGenerationBatchChildRow {
            mold_db::generation_batches::get_durable(
                self.db.as_ref().as_ref().unwrap(),
                self.journal.owner_uuid().unwrap(),
                "batch",
            )
            .unwrap()
            .unwrap()
            .children
            .into_iter()
            .find(|child| child.job_id == id)
            .unwrap()
        }

        fn take_channels(&mut self) -> SettlementChannels {
            std::mem::replace(
                &mut self.channels,
                SettlementChannels {
                    journal: None,
                    progress_tx: None,
                    result_tx: None,
                },
            )
        }

        fn frame(&mut self) -> mold_core::SseErrorEvent {
            match self.progress_rx.try_recv().expect("one terminal frame") {
                SseMessage::Error(event) => event,
                _ => panic!("expected a terminal error frame"),
            }
        }
    }

    #[test]
    fn fail_blocking_persists_before_it_reports() {
        let mut harness = Harness::new("held");
        let outcome = fail_blocking(
            harness.take_channels(),
            DurableDisposition::Hold { retryable: true },
            "the model is not installed".to_string(),
        );
        assert_eq!(outcome, SettlementOutcome::Settled);
        let row = harness.row("held").expect("a held row survives");
        assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Held);
        assert_eq!(
            row.held_reason.as_deref(),
            Some("the model is not installed")
        );
        assert!(harness.child("held").retryable);
        let frame = harness.frame();
        assert_eq!(frame.message, "the model is not installed");
        assert!(!frame.retained);
        assert_eq!(frame.code, None);
        assert_eq!(
            harness
                .result_rx
                .try_recv()
                .unwrap()
                .err()
                .expect("the raw observer reads the failure"),
            "the model is not installed"
        );
    }

    #[test]
    fn a_split_message_records_the_durable_half_and_reports_the_observer_half() {
        let mut harness = Harness::new("split");
        fail_blocking(
            harness.take_channels(),
            DurableDisposition::Hold { retryable: false },
            FailureMessage::Split {
                durable: "dispatch attempts exhausted".to_string(),
                observer: "'mock-model' was started 2 times without finishing".to_string(),
            },
        );
        let row = harness.row("split").unwrap();
        assert_eq!(
            row.held_reason.as_deref(),
            Some("dispatch attempts exhausted"),
            "the row records the operator's half"
        );
        assert!(!harness.child("split").retryable);
        assert_eq!(
            harness.frame().message,
            "'mock-model' was started 2 times without finishing",
            "the observer reads the user's half"
        );
    }

    #[test]
    fn fail_blocking_reports_cancelled_when_cancellation_won() {
        let mut harness = Harness::new("cancelled");
        assert_eq!(
            harness.channels.journal.as_ref().unwrap().claim_dispatch(),
            crate::queue_journal::DispatchClaim::Granted
        );
        assert!(harness.journal.cancel_id("cancelled").unwrap());
        let outcome = fail_blocking(
            harness.take_channels(),
            DurableDisposition::Hold { retryable: true },
            "generation error".to_string(),
        );
        assert_eq!(outcome, SettlementOutcome::Cancelled);
        assert_eq!(
            harness.frame().code.as_deref(),
            Some(mold_core::SSE_ERROR_CODE_QUEUED_CANCELLED)
        );
    }

    #[test]
    fn a_panic_hold_is_retained_by_the_quarantine_fence() {
        let mut harness = Harness::new("panicked");
        // `quarantine_poisoned_worker` raises exactly this fence before the
        // GPU owner settles an inference panic, which is why both worker
        // paths can answer a panic with the same non-retryable hold.
        harness.journal.retain_all();
        let outcome = fail_blocking(
            harness.take_channels(),
            DurableDisposition::Hold { retryable: false },
            "inference panicked".to_string(),
        );
        assert_eq!(outcome, SettlementOutcome::Retained);
        assert_eq!(
            harness.row("panicked").unwrap().state,
            mold_db::generation_queue::QueueRowState::Queued,
            "the fence outranks the hold and the row replays after restart"
        );
        assert!(harness.frame().retained);
    }

    #[test]
    fn settle_publication_holds_a_render_that_never_reached_the_gallery() {
        let mut harness = Harness::new("unsaved");
        let registry = crate::job_registry::JobRegistry::new();
        registry.register("unsaved", "mock-model");
        let response = mold_core::GenerateResponse {
            images: Vec::new(),
            video: None,
            audio: None,
            generation_time_ms: 1,
            model: "mock-model".to_string(),
            seed_used: 7,
            gpu: None,
            request_warnings: Vec::new(),
        };
        settle_publication_blocking(
            &mut harness.channels,
            "unsaved",
            None,
            &registry,
            &crate::queue::SavedOutputNames::default(),
            &response,
            &crate::batch_transaction::GalleryPublicationGate::default(),
        )
        .expect("the caller still delivers the render it holds in memory");
        let row = harness.row("unsaved").expect("the row is kept for review");
        assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Held);
        assert_eq!(row.held_reason.as_deref(), Some(UNPUBLISHED_OUTPUT_REASON));
        assert!(
            harness.child("unsaved").retryable,
            "a failed publication is retryable: the gallery may be writable again"
        );
    }

    #[test]
    fn settle_publication_reports_retained_before_completion() {
        let mut harness = Harness::new("retained");
        let registry = crate::job_registry::JobRegistry::new();
        registry.register("retained", "mock-model");
        // A shutdown racing a failed publication: the fence outranks the hold,
        // so the row survives for the next boot and the observer is told the
        // print was interrupted rather than that it failed.
        harness.journal.retain_all();
        let response = mold_core::GenerateResponse {
            images: Vec::new(),
            video: None,
            audio: None,
            generation_time_ms: 1,
            model: "mock-model".to_string(),
            seed_used: 7,
            gpu: None,
            request_warnings: Vec::new(),
        };
        let outcome = settle_publication_blocking(
            &mut harness.channels,
            "retained",
            None,
            &registry,
            &crate::queue::SavedOutputNames::default(),
            &response,
            &crate::batch_transaction::GalleryPublicationGate::default(),
        )
        .expect_err("a retained render never reports completion");
        assert_eq!(outcome, SettlementOutcome::Retained);
        assert!(harness.frame().retained);
        assert_eq!(
            harness.row("retained").unwrap().state,
            mold_db::generation_queue::QueueRowState::Queued
        );
    }
}
