//! Server-side ledger of in-flight generation jobs.
//!
//! The web UI tracks each "Generate" click as a card in `useGenerateStream`
//! and relies on the SSE stream as the only signal that the work is still
//! happening. When that stream silently drops (network blip, proxy idle
//! timeout, server restart, browser tab suspended past keepalive), the card
//! gets stuck `running` forever because no terminal event arrives.
//!
//! `JobRegistry` is the server's authoritative list of "things still owed an
//! output" — every job between `submit()` and worker completion. The new
//! `GET /api/queue` endpoint exposes this list so the SPA can poll it and
//! dead-letter cards whose server-assigned `id` is no longer present.
//!
//! The registry deliberately doesn't track *completed* jobs — the gallery DB
//! is the source of truth for those. Anything in here is currently queued or
//! actively running on some worker.

use crate::events::EventBroadcaster;
use mold_core::queue_progress::QueueJobProgress;
use mold_core::types::SseProgressEvent;
use mold_core::ServerEvent;
use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;

/// Wire-facing job state. Mirrors the actual lifecycle:
///
/// - `queued` — accepted by `submit()`, sitting in the channel awaiting a
///   dispatcher decision OR the dispatcher is mid-retry across workers.
/// - `running` — handed off to a GPU worker thread; flipping happens when
///   the worker pulls the job off its channel and starts loading / inferring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobLifecycle {
    Queued,
    Running,
    /// Durable work recovered after restart. It is listed but never hydrated
    /// into this runtime registry until an operator resumes it.
    Paused,
    /// Retained across a restart but parked: it exceeded an attempt cap, or
    /// its recorded request could not be reconciled. Listed so an operator can
    /// see it, never auto-run.
    Held,
}

/// Number a listing's rows in dispatch order, skipping held ones.
///
/// `position` answers "how many jobs are ahead of me", and a held row is
/// ahead of nobody: it is parked until an operator retries it. Counting it
/// put a parked job at 0 (rendered "Next up") and everything queued behind
/// it one place further back than the host would actually run it. A held row
/// keeps the position of the next schedulable row so the field stays a
/// `usize`; every renderer reads `state` first and never shows it.
pub(crate) fn assign_positions(entries: &mut [JobEntry], offset: usize) {
    let mut next = offset;
    for entry in entries.iter_mut() {
        entry.position = next;
        if matches!(entry.state, JobLifecycle::Queued | JobLifecycle::Running) {
            next += 1;
        }
    }
}

/// One row in the `GET /api/queue` response.
///
/// `position` is the 0-based index in the registry's dispatch-priority order
/// at the time of the snapshot — 0 is at the head (about to be dispatched, or
/// already running on a worker), N-1 is last in line. It starts as insertion
/// order but a `PATCH /api/queue/:id {position}` reorder moves a job within it,
/// and it shifts as earlier jobs finish and drop out. The dispatch loops align
/// their lookahead buffer to this order, so it reflects real execution order,
/// not just the reported snapshot.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct JobEntry {
    pub id: String,
    pub model: String,
    pub state: JobLifecycle,
    pub started_at_unix_ms: u64,
    pub position: usize,
    /// GPU ordinal currently running this job (`null` for queued rows).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
    /// Preferred GPU ordinal for queued jobs (`None` means Auto).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_gpu: Option<usize>,
    /// Whether the submitted request pinned a seed. Additive — `metadata`'s
    /// required seed field can't distinguish an explicit 0 from unpinned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_pinned: Option<bool>,
    /// The submitted request's parameters, metadata-shaped so any client can
    /// inspect a queued job and reuse its settings. Additive — absent on
    /// older servers; never carries image payloads.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub metadata: Option<Box<mold_core::OutputMetadata>>,
    /// Whether this job survives a restart. Additive, and deliberately
    /// per-job: `QueueCapabilities.durable_queue` says the server can promise
    /// durability, this says whether *this* job got it. A job can be
    /// non-durable on a durable host — no gallery target, reference-upload
    /// authority, or an oversized request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable: Option<bool>,
    /// Whether this job was resumed from the journal rather than submitted by
    /// a live client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replayed: Option<bool>,
    /// How many times a worker has claimed this job for execution. Diagnoses a
    /// held row: a job that keeps taking the process down shows its count here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch_attempts: Option<u32>,
    /// Why a held job is parked. Present only for `state: held`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_reason: Option<String>,
    /// Durable preparation error for a held job. Additive alias with clearer
    /// lifecycle semantics than the legacy `held_reason` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Whether `POST /api/queue/{id}/retry` may safely resume this held job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    /// Durable batch this row is a child of. Retry demands the whole
    /// authority (instance + batch + client batch + job) and only the
    /// instance belongs to the server, so a listing that withheld this left
    /// every client guessing which batch a held job belonged to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    /// The client-minted idempotency id of [`Self::batch_id`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_batch_id: Option<String>,
    /// One-based position of this child within its batch, as
    /// `GenerationBatchChild::index` reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_index: Option<u32>,
}

/// Whole-queue listing returned by `GET /api/queue`. Wrapped in a struct so
/// the response can grow extra fields (totals, byte counts, etc.) without a
/// breaking change.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct QueueListing {
    pub entries: Vec<JobEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<mold_core::QueuePlan>,
}

/// Payload-free registry row for the scheduler's frequent reconciliation.
///
/// This is deliberately distinct from [`JobEntry`]: the coordinator needs
/// only queue identity, lifecycle, order, and lane preference. Keeping the
/// type exhaustive prevents a 10 ms scheduler tick from accidentally cloning
/// request metadata, denoise previews, cancellation handles, or other
/// registry-owned state as those types grow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerQueueEntry {
    pub(crate) id: String,
    pub(crate) state: JobLifecycle,
    pub(crate) position: usize,
    pub(crate) target_gpu: Option<usize>,
}

#[derive(Debug, Clone)]
struct EntryInternal {
    id: String,
    model: String,
    state: JobLifecycle,
    started_at_unix_ms: u64,
    gpu: Option<usize>,
    target_gpu: Option<usize>,
    seed_pinned: Option<bool>,
    metadata: Option<Box<mold_core::OutputMetadata>>,
    /// Latest folded progress for this live row. It is served separately so
    /// `/api/queue` polling never carries image payloads.
    progress: Option<QueueJobProgress>,
    /// Cancellation signal for `DELETE /api/queue/:id`. The submitting
    /// handler holds the clone returned by `register*()` and selects on
    /// `notified()` alongside the job's result channel; `cancel_queued`
    /// fires `notify_one()` so the permit survives even when the cancel
    /// lands before the waiter starts awaiting.
    cancel: Arc<Notify>,
    /// Attempt token installed by the owner once this row is running. A
    /// cancel may arrive before installation; `cancel_requested` closes that
    /// hand-off race and cancels the token as soon as it appears.
    running_cancel: Option<mold_inference::InferenceCancellationToken>,
    cancel_requested: bool,
    /// Terminal publication has won the lifecycle race. Keep the row visible
    /// until SQLite is settled so DELETE cannot fall through to the durable
    /// row and revoke authority after bytes have started publishing.
    completion_claimed: bool,
    /// Exact route-owned token that temporarily excludes this queued row from
    /// scheduler planning/grant while its durable PATCH is in flight.
    queue_patch_token: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetGpuUpdateError {
    NotFound,
    AlreadyRunning,
}

/// Why a `DELETE /api/queue/:id` cancel attempt was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuedJobCancelError {
    NotFound,
    AlreadyRunning,
    CompletionClaimed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompletionClaim {
    Claimed,
    UserCancelled,
    AttemptCancelled,
}

/// Why a `PATCH /api/queue/:id` `position` reorder was rejected. Same shape as
/// the target-gpu/cancel errors — only queued jobs are movable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueReorderError {
    NotFound,
    AlreadyRunning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DispatchClaimError {
    NotFound,
    NotQueued,
}

#[derive(Debug)]
pub(crate) enum DispatchAttemptError<T> {
    Claim(DispatchClaimError, T),
    Transport(T),
}

/// The registry itself. Construct via `JobRegistry::new` and share through
/// `AppState`. All mutation is fire-and-forget — if the inner lock is
/// poisoned (extremely unlikely in practice) we recover from the inner
/// state rather than propagating the panic into the dispatcher hot path.
pub struct JobRegistry {
    inner: RwLock<Vec<EntryInternal>>,
    mutation_sequence: AtomicU64,
    queue_patch_sequence: AtomicU64,
    mutation_notify: Arc<Notify>,
    /// Optional lifecycle broadcast (`GET /api/events`). Emitting from the
    /// registry — rather than each call site — guarantees every submit /
    /// promote / terminal path produces exactly one event. `None` keeps
    /// event-less construction (tests) cheap.
    events: Option<Arc<EventBroadcaster>>,
}

/// Cheap-cloneable handle. Workers and routes pass this around by value.
pub type SharedJobRegistry = Arc<JobRegistry>;

/// Every fed job's progress channel.
///
/// A durable child has no attached observer of its own — the snapshot
/// `GET /api/queue/{id}/preview` serves is the ONE channel every surface
/// reads — so the channel a worker sends progress on is a relay that folds
/// every `Progress` frame into that snapshot and forwards every message to
/// the attached SSE observer when there is one. Producers keep sending to
/// `progress_tx` exactly as they always did, from the denoise loop to the
/// model pull, the dependency wait, the queue position and the H3 phases;
/// none of them knows the registry, and none can bypass it.
pub fn progress_relay(
    registry: &SharedJobRegistry,
    job_id: &str,
    observer: Option<tokio::sync::mpsc::UnboundedSender<crate::state::SseMessage>>,
) -> tokio::sync::mpsc::UnboundedSender<crate::state::SseMessage> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let registry = Arc::clone(registry);
    let job_id = job_id.to_string();
    tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if let crate::state::SseMessage::Progress(event) = &message {
                registry.record_progress(&job_id, event);
            }
            if let Some(observer) = observer.as_ref() {
                let _ = observer.send(message);
            }
        }
    });
    tx
}

impl JobRegistry {
    pub fn new() -> SharedJobRegistry {
        Arc::new(Self {
            inner: RwLock::new(Vec::new()),
            mutation_sequence: AtomicU64::new(0),
            queue_patch_sequence: AtomicU64::new(0),
            mutation_notify: Arc::new(Notify::new()),
            events: None,
        })
    }

    /// Like [`JobRegistry::new`] but mirrors every lifecycle change onto the
    /// server-wide event broadcast.
    pub fn with_events(events: Arc<EventBroadcaster>) -> SharedJobRegistry {
        Arc::new(Self {
            inner: RwLock::new(Vec::new()),
            mutation_sequence: AtomicU64::new(0),
            queue_patch_sequence: AtomicU64::new(0),
            mutation_notify: Arc::new(Notify::new()),
            events: Some(events),
        })
    }

    /// Publish outside the registry lock — callers must have dropped the
    /// write guard first so a slow broadcast can never extend the critical
    /// section.
    fn emit(&self, event: ServerEvent) {
        if let Some(events) = &self.events {
            events.publish(event);
        }
    }

    /// Insert a freshly-submitted job at the tail in `Queued` state.
    /// Returns the job's cancellation signal — see `register_with_target_gpu`.
    pub fn register(&self, id: impl Into<String>, model: impl Into<String>) -> Arc<Notify> {
        self.register_with_target_gpu(id, model, None)
    }

    /// Insert a freshly-submitted job with an optional queued lane target.
    ///
    /// Returns the job's cancellation signal. The submitting handler must
    /// hold it and select on `notified()` alongside the result channel —
    /// `cancel_queued` resolves it when `DELETE /api/queue/:id` removes the
    /// entry. Callers that never wait (tests poking the registry directly)
    /// can drop the handle.
    pub fn register_with_target_gpu(
        &self,
        id: impl Into<String>,
        model: impl Into<String>,
        target_gpu: Option<usize>,
    ) -> Arc<Notify> {
        self.register_job(id, model, target_gpu, None, None)
    }

    /// Full-form insert: `register_with_target_gpu` plus the request's
    /// metadata so `GET /api/queue` can expose the job's settings.
    pub fn register_job(
        &self,
        id: impl Into<String>,
        model: impl Into<String>,
        target_gpu: Option<usize>,
        seed_pinned: Option<bool>,
        metadata: Option<Box<mold_core::OutputMetadata>>,
    ) -> Arc<Notify> {
        self.register_job_at_queued_position(
            id,
            model,
            target_gpu,
            seed_pinned,
            metadata,
            usize::MAX,
        )
    }

    /// Insert a hydrated durable job at its authoritative queued position.
    ///
    /// `position` is interpreted among queued rows, exactly like
    /// [`reorder_queued`](Self::reorder_queued), and clamps to the live tail.
    /// The feeder supplies a position from SQLite's bounded runtime-order
    /// window while holding the scheduler mutation fence. Keeping insertion
    /// and ordinary PATCH reorders on the same registry lock makes the new
    /// row visible to the scheduler in one fully ordered mutation.
    pub(crate) fn register_job_at_queued_position(
        &self,
        id: impl Into<String>,
        model: impl Into<String>,
        target_gpu: Option<usize>,
        seed_pinned: Option<bool>,
        metadata: Option<Box<mold_core::OutputMetadata>>,
        position: usize,
    ) -> Arc<Notify> {
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let id = id.into();
        let model = model.into();
        let cancel = Arc::new(Notify::new());
        {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let queued_slots = entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.state == JobLifecycle::Queued)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let insert_at = queued_slots.get(position).copied().unwrap_or(entries.len());
            entries.insert(
                insert_at,
                EntryInternal {
                    id: id.clone(),
                    model: model.clone(),
                    state: JobLifecycle::Queued,
                    started_at_unix_ms,
                    gpu: None,
                    target_gpu,
                    seed_pinned,
                    metadata,
                    progress: None,
                    cancel: cancel.clone(),
                    running_cancel: None,
                    cancel_requested: false,
                    completion_claimed: false,
                    queue_patch_token: None,
                },
            );
        }
        self.mark_mutated();
        self.emit(ServerEvent::JobQueued { id, model });
        cancel
    }

    /// Cancel a queued or running job. Queued work is removed immediately;
    /// running work keeps its row until the owner acknowledges the cooperative
    /// token. The running signal and terminal publication claim use this same
    /// lock, so DELETE and completion have one deterministic winner.
    ///
    /// The state check and removal happen under the same write lock that
    /// `mark_running` takes, so a job can never be both cancelled and
    /// promoted. (A worker that already dequeued the job before the cancel
    /// landed will observe the closed result channel and skip it.)
    pub fn cancel_queued(&self, id: &str) -> Result<(), QueuedJobCancelError> {
        {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let Some(pos) = entries.iter().position(|e| e.id == id) else {
                return Err(QueuedJobCancelError::NotFound);
            };
            if entries[pos].completion_claimed {
                return Err(QueuedJobCancelError::CompletionClaimed);
            }
            if entries[pos].state == JobLifecycle::Running || entries[pos].running_cancel.is_some()
            {
                entries[pos].cancel_requested = true;
                if let Some(token) = &entries[pos].running_cancel {
                    token.cancel();
                }
                return Err(QueuedJobCancelError::AlreadyRunning);
            }
            let entry = entries.remove(pos);
            if let Some(token) = &entry.running_cancel {
                token.cancel();
            }
            entry.cancel.notify_one();
        }
        self.mark_mutated();
        self.emit(ServerEvent::JobEnded { id: id.to_string() });
        Ok(())
    }

    /// Cancel every still-queued job in one pass, backing `DELETE /api/queue`.
    /// Running jobs remain untouched. After dropping the entry lock this emits
    /// one `JobEnded` per removed queued row. The public wrapper returns the
    /// established queued-row count; the route uses the exact ids to avoid
    /// double-counting durable rows in the same cancellation.
    pub fn cancel_all_queued(&self) -> usize {
        self.cancel_all_queued_ids().len()
    }

    pub(crate) fn cancel_all_queued_ids(&self) -> Vec<String> {
        let cancelled_ids = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let mut ids = Vec::new();
            entries.retain_mut(|e| {
                if e.state == JobLifecycle::Queued && e.running_cancel.is_none() {
                    e.cancel.notify_one();
                    ids.push(e.id.clone());
                    false
                } else {
                    true
                }
            });
            ids
        };
        for id in &cancelled_ids {
            self.emit(ServerEvent::JobEnded { id: id.clone() });
        }
        if !cancelled_ids.is_empty() {
            self.mark_mutated();
        }
        cancelled_ids
    }

    /// Promote a registry entry from `Queued` to `Running`. No-op if `id`
    /// isn't present (the entry may have been removed concurrently).
    pub fn mark_running(&self, id: &str, gpu: Option<usize>) {
        let model = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            entries.iter_mut().find(|e| e.id == id).map(|e| {
                e.state = JobLifecycle::Running;
                e.gpu = gpu;
                e.target_gpu = None;
                e.model.clone()
            })
        };
        if let Some(model) = model {
            self.mark_mutated();
            self.emit(ServerEvent::JobStarted {
                id: id.to_string(),
                model,
                gpu,
            });
        }
    }

    /// Attach the exact attempt token to a running row. If DELETE won the
    /// dispatch hand-off race, installation immediately observes that fact.
    /// Returns `false` when the row was already removed before hand-off.
    pub(crate) fn install_running_cancellation(
        &self,
        id: &str,
        token: mold_inference::InferenceCancellationToken,
    ) -> bool {
        let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) {
            if entry.cancel_requested {
                token.cancel();
            }
            entry.running_cancel = Some(token);
            true
        } else {
            false
        }
    }

    /// Claim the right to publish a terminal result. DELETE takes the same
    /// write lock, so once this marks the row a later cancel is refused; if
    /// cancellation won first, publication is refused. The row remains until
    /// [`Self::finish_completion`] so the durable fallback cannot cancel it
    /// during gallery publication.
    pub(crate) fn claim_completion(&self, id: &str) -> CompletionClaim {
        let claim = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let Some(pos) = entries.iter().position(|entry| entry.id == id) else {
                // Legacy/unit jobs without a public registry row retain their
                // historical completion behavior.
                return CompletionClaim::Claimed;
            };
            if entries[pos].cancel_requested {
                CompletionClaim::UserCancelled
            } else if entries[pos]
                .running_cancel
                .as_ref()
                .is_some_and(|token| token.is_cancelled())
            {
                CompletionClaim::AttemptCancelled
            } else {
                entries[pos].completion_claimed = true;
                CompletionClaim::Claimed
            }
        };
        claim
    }

    /// Remove a row after its durable terminal disposition is settled and
    /// gallery publication is no longer cancellable.
    pub(crate) fn finish_completion(&self, id: &str) {
        let removed = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            entries
                .iter()
                .position(|entry| entry.id == id && entry.completion_claimed)
                .map(|pos| entries.remove(pos))
                .is_some()
        };
        if removed {
            self.mark_mutated();
            self.emit(ServerEvent::JobEnded { id: id.to_string() });
        }
    }

    #[cfg(test)]
    pub(crate) fn completion_claimed_for_tests(&self, id: &str) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .any(|entry| entry.id == id && entry.completion_claimed)
    }

    pub(crate) fn cancel_requested(&self, id: &str) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|entry| entry.id == id)
            .is_some_and(|entry| entry.cancel_requested)
    }

    /// Claim a queued row for one scheduler grant.
    ///
    /// The scheduler holds `AppState::scheduler_mutation_fence` across this
    /// claim and the nonblocking worker send. Queue mutation routes take the
    /// same fence, so once this returns, cancel/reorder/retarget observes
    /// `Running` and cannot race the transported grant.
    pub(crate) fn dispatch_if_queued<T>(
        &self,
        id: &str,
        gpu: usize,
        payload: T,
        send: impl FnOnce(T) -> Result<(), T>,
    ) -> Result<Option<usize>, DispatchAttemptError<T>> {
        let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
        let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) else {
            return Err(DispatchAttemptError::Claim(
                DispatchClaimError::NotFound,
                payload,
            ));
        };
        if entry.state != JobLifecycle::Queued || entry.queue_patch_token.is_some() {
            return Err(DispatchAttemptError::Claim(
                DispatchClaimError::NotQueued,
                payload,
            ));
        }
        let previous_target = entry.target_gpu;
        if let Err(payload) = send(payload) {
            return Err(DispatchAttemptError::Transport(payload));
        }
        entry.state = JobLifecycle::Running;
        entry.gpu = Some(gpu);
        entry.target_gpu = None;
        let model = entry.model.clone();
        drop(entries);
        self.mark_mutated();
        self.emit(ServerEvent::JobStarted {
            id: id.to_string(),
            model,
            gpu: Some(gpu),
        });
        Ok(previous_target)
    }

    pub(crate) fn requeue_rejected_dispatch(&self, id: &str, target_gpu: Option<usize>) {
        let restored = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            entries
                .iter_mut()
                .find(|entry| entry.id == id)
                .is_some_and(|entry| {
                    if entry.state != JobLifecycle::Running {
                        return false;
                    }
                    entry.state = JobLifecycle::Queued;
                    entry.gpu = None;
                    entry.target_gpu = target_gpu;
                    true
                })
        };
        if restored {
            self.mark_mutated();
        }
    }

    /// Exclude one hydrated queued row from scheduler planning and final
    /// dispatch while its durable PATCH is persisted without the scheduler
    /// mutation fence. The returned token fences cleanup from any later PATCH.
    pub(crate) fn begin_queue_patch(&self, id: &str) -> Result<u64, TargetGpuUpdateError> {
        let token = self
            .queue_patch_sequence
            .fetch_add(1, Ordering::SeqCst)
            .wrapping_add(1);
        {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) else {
                return Err(TargetGpuUpdateError::NotFound);
            };
            if entry.state == JobLifecycle::Running {
                return Err(TargetGpuUpdateError::AlreadyRunning);
            }
            entry.queue_patch_token = Some(token);
        }
        self.mark_mutated();
        Ok(token)
    }

    pub(crate) fn queue_patch_token_matches(&self, id: &str, token: u64) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .find(|entry| entry.id == id)
            .is_some_and(|entry| entry.queue_patch_token == Some(token))
    }

    /// Release only the exact PATCH exclusion installed by
    /// [`begin_queue_patch`](Self::begin_queue_patch).
    pub(crate) fn finish_queue_patch(&self, id: &str, token: u64) {
        let cleared = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            entries
                .iter_mut()
                .find(|entry| entry.id == id)
                .is_some_and(|entry| {
                    if entry.queue_patch_token != Some(token) {
                        return false;
                    }
                    entry.queue_patch_token = None;
                    true
                })
        };
        if cleared {
            self.mark_mutated();
        }
    }

    /// Complete an exact queue mutation by changing the hydrated lifecycle
    /// before releasing its scheduler exclusion. Used by per-job pause/resume
    /// so buffered work stays registered but only that row leaves dispatch.
    pub(crate) fn finish_queue_patch_state(
        &self,
        id: &str,
        token: u64,
        state: JobLifecycle,
    ) -> bool {
        debug_assert!(matches!(state, JobLifecycle::Queued | JobLifecycle::Paused));
        let changed = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            entries
                .iter_mut()
                .find(|entry| entry.id == id)
                .is_some_and(|entry| {
                    if entry.queue_patch_token != Some(token) {
                        return false;
                    }
                    entry.state = state;
                    entry.queue_patch_token = None;
                    true
                })
        };
        if changed {
            self.mark_mutated();
        }
        changed
    }

    pub fn set_target_gpu(
        &self,
        id: &str,
        target_gpu: Option<usize>,
    ) -> Result<(), TargetGpuUpdateError> {
        {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let Some(e) = entries.iter_mut().find(|e| e.id == id) else {
                return Err(TargetGpuUpdateError::NotFound);
            };
            if e.state == JobLifecycle::Running {
                return Err(TargetGpuUpdateError::AlreadyRunning);
            }
            e.target_gpu = target_gpu;
        }
        self.mark_mutated();
        Ok(())
    }

    /// Move a still-queued job to be the `position`-th queued entry (0-based)
    /// among the queued jobs. This order is authoritative for dispatch: the
    /// single- and multi-GPU loops align their lookahead buffer to
    /// [`queued_ids_in_order`](Self::queued_ids_in_order) before picking, so a
    /// reorder changes real execution order, not just the `GET /api/queue`
    /// snapshot.
    ///
    /// `position` is clamped into the queued range, so a caller can pass a
    /// large number to mean "send to the back". Running jobs are ignored for
    /// indexing (they've left the reorderable set) and keep their absolute
    /// slot, so a queued job can never jump ahead of one already executing.
    /// The relative order of the other queued jobs is preserved. Moving a job
    /// that is already at `position` is a successful no-op.
    pub fn reorder_queued(&self, id: &str, position: usize) -> Result<(), QueueReorderError> {
        {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let Some(from) = entries.iter().position(|e| e.id == id) else {
                return Err(QueueReorderError::NotFound);
            };
            if entries[from].state == JobLifecycle::Running {
                return Err(QueueReorderError::AlreadyRunning);
            }
            // Vec indices occupied by queued entries, in order — running entries
            // are skipped so `position` indexes purely among movable jobs.
            let queued_slots: Vec<usize> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.state == JobLifecycle::Queued)
                .map(|(i, _)| i)
                .collect();
            // `from` is queued (checked above), so it is present here.
            let cur = queued_slots
                .iter()
                .position(|&i| i == from)
                .expect("queued job must appear in its own queued-slot list");
            let target = position.min(queued_slots.len() - 1);
            if cur == target {
                return Ok(());
            }
            // Pull the job out, then reinsert at the Vec index that makes it the
            // `target`-th queued entry among the survivors.
            let entry = entries.remove(from);
            let remaining_slots: Vec<usize> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.state == JobLifecycle::Queued)
                .map(|(i, _)| i)
                .collect();
            let dest = remaining_slots
                .get(target)
                .copied()
                .unwrap_or(entries.len());
            entries.insert(dest, entry);
        }
        self.mark_mutated();
        Ok(())
    }

    pub fn target_gpu(&self, id: &str) -> Option<Option<usize>> {
        let entries = self.inner.read().unwrap_or_else(|e| e.into_inner());
        entries.iter().find(|e| e.id == id).map(|e| e.target_gpu)
    }

    pub fn entry(&self, id: &str) -> Option<JobEntry> {
        let entries = self.inner.read().unwrap_or_else(|e| e.into_inner());
        entries.iter().enumerate().find_map(|(i, e)| {
            (e.id == id).then(|| JobEntry {
                id: e.id.clone(),
                model: e.model.clone(),
                state: e.state,
                started_at_unix_ms: e.started_at_unix_ms,
                position: i,
                gpu: e.gpu,
                target_gpu: e.target_gpu,
                seed_pinned: e.seed_pinned,
                metadata: e.metadata.clone(),
                // Durability is projected onto the listing by the route from
                // the journal, which is the authority. Deriving it here would
                // mean mirroring journal state into the registry and having
                // two answers to keep in sync.
                durable: None,
                replayed: None,
                dispatch_attempts: None,
                held_reason: None,
                error: None,
                retryable: None,
                // Batch membership is durable state, projected by the route
                // from the journal for the same reason `durable` is.
                batch_id: None,
                client_batch_id: None,
                batch_index: None,
            })
        })
    }

    /// Fold one progress event into the selected job's live snapshot. A late
    /// callback cannot resurrect terminal work because only a currently-live
    /// registry row is allowed to publish.
    ///
    /// This is fed from the same `progress_tx` fan-out the attached observer
    /// reads, so a durable child and an attached stream describe one render.
    pub fn record_progress(&self, id: &str, event: &SseProgressEvent) {
        let mut entries = self
            .inner
            .write()
            .unwrap_or_else(|error| error.into_inner());
        let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) else {
            return;
        };
        entry
            .progress
            .get_or_insert_with(QueueJobProgress::default)
            .apply(event, mold_core::time::now_epoch_ms_u64());
    }

    /// The outer option is live-row existence; the inner option is whether
    /// that live row has reported any progress yet.
    pub fn progress_snapshot(&self, id: &str) -> Option<Option<QueueJobProgress>> {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| entry.progress.clone())
    }

    /// Drop the entry — call once on every terminal path (success, error,
    /// client-disconnect skip, dispatch failure). Idempotent.
    pub fn remove(&self, id: &str) {
        let _ = self.remove_if_present(id);
    }

    /// Drop the entry and report whether this call owned terminal settlement.
    ///
    /// The scheduler uses this to pair queue-depth decrement with the exact
    /// terminal path that actually removed a job after an owner disappeared.
    pub(crate) fn remove_if_present(&self, id: &str) -> bool {
        if id.is_empty() {
            return false;
        }
        let removed = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let before = entries.len();
            entries.retain(|e| e.id != id);
            entries.len() != before
        };
        // `remove` is called on every terminal path and is deliberately
        // idempotent — only the call that actually dropped the entry emits,
        // so subscribers see exactly one `job_ended` per job.
        if removed {
            self.mark_mutated();
            self.emit(ServerEvent::JobEnded { id: id.to_string() });
        }
        removed
    }

    /// The ids of currently-queued jobs, in dispatch-priority order — the same
    /// order [`reorder_queued`](Self::reorder_queued) maintains and `GET
    /// /api/queue` reports. Running jobs are excluded (they've left the
    /// reorderable set).
    ///
    /// The dispatch loops call this each iteration to align their lookahead
    /// buffer to the registry, which is what makes the registry the single
    /// source of truth for order: a `PATCH /api/queue/:id {position}` reorder
    /// changes real dispatch order, not just the snapshot.
    pub fn queued_ids_in_order(&self) -> Vec<String> {
        let entries = self.inner.read().unwrap_or_else(|e| e.into_inner());
        entries
            .iter()
            .filter(|e| e.state == JobLifecycle::Queued)
            .map(|e| e.id.clone())
            .collect()
    }

    /// Monotonic signal consumed by the scheduler coordinator. Unlike a
    /// snapshot comparison, this cannot miss two rapid mutations that happen
    /// to restore the same visible queue shape.
    pub fn mutation_sequence(&self) -> u64 {
        self.mutation_sequence.load(Ordering::SeqCst)
    }

    pub fn mutation_notifier(&self) -> Arc<Notify> {
        self.mutation_notify.clone()
    }

    fn mark_mutated(&self) {
        self.mutation_sequence.fetch_add(1, Ordering::SeqCst);
        self.mutation_notify.notify_one();
    }

    /// Snapshot the registry as a wire-shaped listing. Positions are assigned
    /// in the registry's dispatch-priority order at snapshot time (insertion
    /// order until a `reorder_queued` moves a job).
    pub fn snapshot(&self) -> QueueListing {
        let entries = self.inner.read().unwrap_or_else(|e| e.into_inner());
        let out = entries
            .iter()
            .enumerate()
            .map(|(i, e)| JobEntry {
                id: e.id.clone(),
                model: e.model.clone(),
                state: e.state,
                started_at_unix_ms: e.started_at_unix_ms,
                position: i,
                gpu: e.gpu,
                target_gpu: e.target_gpu,
                seed_pinned: e.seed_pinned,
                metadata: e.metadata.clone(),
                // Durability is projected onto the listing by the route from
                // the journal, which is the authority. Deriving it here would
                // mean mirroring journal state into the registry and having
                // two answers to keep in sync.
                durable: None,
                replayed: None,
                dispatch_attempts: None,
                held_reason: None,
                error: None,
                retryable: None,
                // Batch membership is durable state, projected by the route
                // from the journal for the same reason `durable` is.
                batch_id: None,
                client_batch_id: None,
                batch_index: None,
            })
            .collect();
        QueueListing {
            entries: out,
            plan: None,
        }
    }

    /// Snapshot only the fields consumed by scheduler reconciliation.
    ///
    /// Unlike [`snapshot`](Self::snapshot), work here is independent of the
    /// size of retained request metadata and previews: one small row is
    /// projected per registry entry and no payload-bearing field is touched.
    pub(crate) fn scheduler_snapshot(&self) -> Vec<SchedulerQueueEntry> {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .enumerate()
            .map(|(position, entry)| SchedulerQueueEntry {
                id: entry.id.clone(),
                state: if entry.queue_patch_token.is_some() {
                    JobLifecycle::Held
                } else {
                    entry.state
                },
                position,
                target_gpu: entry.target_gpu,
            })
            .collect()
    }

    /// Exact queued rows temporarily excluded while a durable PATCH owns
    /// their runtime-projection token.
    ///
    /// The coordinator omits these ids from planning without removing their
    /// pending generation or changing its durable lifecycle. Clearing the
    /// exact token marks the registry mutated, making the retained work
    /// eligible on the next immediate replan.
    pub(crate) fn queue_patch_blocked_ids(&self) -> BTreeSet<String> {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .filter(|entry| entry.queue_patch_token.is_some())
            .map(|entry| entry.id.clone())
            .collect()
    }

    /// Read only the lifecycle needed by scheduler cancellation/grant checks.
    ///
    /// This retains the full snapshot API for wire callers without making a
    /// scheduler existence check clone the selected row's metadata.
    pub(crate) fn scheduler_lifecycle(&self, id: &str) -> Option<JobLifecycle> {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| {
                if entry.queue_patch_token.is_some() {
                    JobLifecycle::Held
                } else {
                    entry.state
                }
            })
    }

    /// Currently-tracked job count. Exposed for tests and metrics.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Returns true when nothing is queued or running. Public so other
    /// callers (metrics, integration tests) can check emptiness without
    /// allocating a full snapshot.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true while any accepted generation is executing.
    ///
    /// The registry is the lifecycle authority across scheduler/worker
    /// hand-offs, so consumers deciding whether the server is idle must use
    /// this signal rather than relying only on transport-local counters.
    pub(crate) fn has_running(&self) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|entry| entry.state == JobLifecycle::Running)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The relay is what makes the snapshot complete: a frame sent on the
    /// job's `progress_tx` lands in the registry whether or not anyone is
    /// attached, and an attached observer still sees every message.
    #[tokio::test]
    async fn a_progress_relay_folds_every_frame_and_forwards_to_the_observer() {
        let registry: SharedJobRegistry = JobRegistry::new();
        registry.register("relayed", "flux-dev:q8");
        let (observer_tx, mut observer_rx) = tokio::sync::mpsc::unbounded_channel();
        let progress_tx = progress_relay(&registry, "relayed", Some(observer_tx));

        progress_tx
            .send(crate::state::SseMessage::Progress(
                SseProgressEvent::DenoiseStep {
                    step: 3,
                    total: 20,
                    elapsed_ms: 0,
                },
            ))
            .unwrap();
        let forwarded = tokio::time::timeout(std::time::Duration::from_secs(2), observer_rx.recv())
            .await
            .expect("the observer sees the frame")
            .expect("channel open");
        assert!(matches!(
            forwarded,
            crate::state::SseMessage::Progress(SseProgressEvent::DenoiseStep {
                step: 3,
                total: 20,
                ..
            })
        ));
        let snapshot = registry
            .progress_snapshot("relayed")
            .flatten()
            .expect("the frame folded into the snapshot");
        assert_eq!((snapshot.step, snapshot.total), (Some(3), Some(20)));

        // No observer: the snapshot is still fed.
        registry.register("quiet", "flux-dev:q8");
        let quiet_tx = progress_relay(&registry, "quiet", None);
        quiet_tx
            .send(crate::state::SseMessage::Progress(
                SseProgressEvent::DenoiseStep {
                    step: 1,
                    total: 4,
                    elapsed_ms: 0,
                },
            ))
            .unwrap();
        for _ in 0..50 {
            if registry.progress_snapshot("quiet").flatten().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(registry.progress_snapshot("quiet").flatten().is_some());
    }

    #[test]
    fn held_rows_take_no_place_in_line() {
        let mut entries = ["held-a", "queued-b", "held-c", "queued-d"]
            .into_iter()
            .map(|id| JobEntry {
                id: id.to_string(),
                model: "m".to_string(),
                state: if id.starts_with("held") {
                    JobLifecycle::Held
                } else {
                    JobLifecycle::Queued
                },
                started_at_unix_ms: 0,
                position: 99,
                gpu: None,
                target_gpu: None,
                seed_pinned: None,
                metadata: None,
                durable: None,
                replayed: None,
                dispatch_attempts: None,
                held_reason: None,
                error: None,
                retryable: None,
                batch_id: None,
                client_batch_id: None,
                batch_index: None,
            })
            .collect::<Vec<_>>();
        assign_positions(&mut entries, 0);
        let positions = entries
            .iter()
            .map(|entry| (entry.id.as_str(), entry.position))
            .collect::<Vec<_>>();
        // The first queued row is next up even though a held row is listed
        // ahead of it; the second is #1, not #3.
        assert_eq!(
            positions,
            vec![
                ("held-a", 0),
                ("queued-b", 0),
                ("held-c", 1),
                ("queued-d", 1)
            ]
        );
        let mut paged = entries.split_off(2);
        assign_positions(&mut paged, 5);
        assert_eq!(paged[1].position, 5);
    }

    #[test]
    fn register_appends_in_fifo_order_with_queued_state() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        let snap = reg.snapshot();
        assert_eq!(snap.entries.len(), 2);
        assert_eq!(snap.entries[0].id, "a");
        assert_eq!(snap.entries[0].position, 0);
        assert_eq!(snap.entries[0].state, JobLifecycle::Queued);
        assert_eq!(snap.entries[1].id, "b");
        assert_eq!(snap.entries[1].position, 1);
    }

    #[test]
    fn exact_queue_patch_block_excludes_only_its_target_from_dispatch() {
        let reg = JobRegistry::new();
        reg.register("patching", "model-a");
        reg.register("unrelated", "model-b");

        let first = reg.begin_queue_patch("patching").unwrap();
        assert_eq!(
            reg.scheduler_lifecycle("patching"),
            Some(JobLifecycle::Held)
        );
        assert!(reg
            .dispatch_if_queued("patching", 0, (), |_| Ok(()))
            .is_err());
        assert!(reg
            .dispatch_if_queued("unrelated", 0, (), |_| Ok(()))
            .is_ok());

        reg.finish_queue_patch("patching", first);
        let second = reg.begin_queue_patch("patching").unwrap();
        reg.finish_queue_patch("patching", first);
        assert_eq!(
            reg.scheduler_lifecycle("patching"),
            Some(JobLifecycle::Held),
            "stale cleanup cannot clear a later PATCH exclusion"
        );
        reg.finish_queue_patch("patching", second);
        assert_eq!(
            reg.scheduler_lifecycle("patching"),
            Some(JobLifecycle::Queued)
        );
    }

    #[test]
    fn one_job_pause_excludes_only_that_job_until_exact_resume() {
        let reg = JobRegistry::new();
        reg.register("paused", "model-a");
        reg.register("sibling", "model-b");

        let pause = reg.begin_queue_patch("paused").unwrap();
        assert!(reg.finish_queue_patch_state("paused", pause, JobLifecycle::Paused));
        assert_eq!(reg.queued_ids_in_order(), ["sibling"]);
        assert_eq!(
            reg.scheduler_lifecycle("paused"),
            Some(JobLifecycle::Paused)
        );

        let resume = reg.begin_queue_patch("paused").unwrap();
        assert!(reg.finish_queue_patch_state("paused", resume, JobLifecycle::Queued));
        assert_eq!(reg.queued_ids_in_order(), ["paused", "sibling"]);
    }

    #[test]
    fn scheduler_mutation_sequence_observes_target_reorder_and_cancel() {
        let reg = JobRegistry::new();
        let initial = reg.mutation_sequence();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        let after_enqueue = reg.mutation_sequence();
        assert!(after_enqueue >= initial + 2);

        reg.set_target_gpu("a", Some(1)).unwrap();
        let after_target = reg.mutation_sequence();
        assert!(after_target > after_enqueue);

        reg.reorder_queued("b", 0).unwrap();
        let after_reorder = reg.mutation_sequence();
        assert!(after_reorder > after_target);

        reg.cancel_queued("a").unwrap();
        assert!(reg.mutation_sequence() > after_reorder);
    }

    #[test]
    fn mark_running_flips_state_and_records_gpu_ordinal() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        assert!(!reg.has_running());
        reg.mark_running("a", Some(1));
        assert!(reg.has_running());
        let snap = reg.snapshot();
        assert_eq!(snap.entries[0].state, JobLifecycle::Running);
        assert_eq!(snap.entries[0].gpu, Some(1));
    }

    #[test]
    fn atomic_dispatch_marks_running_only_after_transport_accepts() {
        let reg = JobRegistry::new();
        reg.register_with_target_gpu("a", "flux-dev:fp16", Some(2));

        let previous_target = reg
            .dispatch_if_queued("a", 1, "payload", |_| Ok(()))
            .expect("transport accepted");

        assert_eq!(previous_target, Some(2));
        let entry = reg.entry("a").unwrap();
        assert_eq!(entry.state, JobLifecycle::Running);
        assert_eq!(entry.gpu, Some(1));
        assert_eq!(entry.target_gpu, None);
    }

    #[test]
    fn failed_atomic_dispatch_preserves_queued_state_and_target() {
        let reg = JobRegistry::new();
        reg.register_with_target_gpu("a", "flux-dev:fp16", Some(2));

        let error = reg
            .dispatch_if_queued("a", 1, "payload", Err)
            .expect_err("transport rejected");

        assert!(matches!(error, DispatchAttemptError::Transport("payload")));
        let entry = reg.entry("a").unwrap();
        assert_eq!(entry.state, JobLifecycle::Queued);
        assert_eq!(entry.gpu, None);
        assert_eq!(entry.target_gpu, Some(2));
    }

    #[test]
    fn queued_entries_can_carry_target_gpu_metadata() {
        let reg = JobRegistry::new();
        reg.register_with_target_gpu("a", "flux-dev:fp16", Some(1));
        let snap = reg.snapshot();
        assert_eq!(snap.entries[0].state, JobLifecycle::Queued);
        assert_eq!(snap.entries[0].target_gpu, Some(1));
        assert_eq!(snap.entries[0].gpu, None);
    }

    #[test]
    fn target_gpu_updates_only_apply_to_queued_entries() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.set_target_gpu("a", Some(1)).unwrap();
        assert_eq!(reg.target_gpu("a"), Some(Some(1)));

        reg.mark_running("a", Some(1));
        let err = reg.set_target_gpu("a", None).unwrap_err();
        assert_eq!(err, TargetGpuUpdateError::AlreadyRunning);
        assert_eq!(reg.target_gpu("a"), Some(None));
    }

    fn queued_order(reg: &JobRegistry) -> Vec<String> {
        reg.snapshot().entries.into_iter().map(|e| e.id).collect()
    }

    #[test]
    fn reorder_moves_queued_job_to_the_front() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        reg.reorder_queued("c", 0).unwrap();
        assert_eq!(queued_order(&reg), ["c", "a", "b"]);
        // Positions recompute from the new order.
        let snap = reg.snapshot();
        assert_eq!(snap.entries[0].position, 0);
        assert_eq!(snap.entries[2].position, 2);
    }

    #[test]
    fn reorder_moves_queued_job_to_the_back() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        reg.reorder_queued("a", 2).unwrap();
        assert_eq!(queued_order(&reg), ["b", "c", "a"]);
    }

    #[test]
    fn reorder_moves_a_middle_job_up_and_preserves_the_others_order() {
        let reg = JobRegistry::new();
        for id in ["a", "b", "c", "d", "e"] {
            reg.register(id, "flux-dev:fp16");
        }
        // Move the last job to slot 1 — everything else keeps relative order.
        reg.reorder_queued("e", 1).unwrap();
        assert_eq!(queued_order(&reg), ["a", "e", "b", "c", "d"]);
    }

    #[test]
    fn reorder_clamps_an_out_of_range_position_to_the_back() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        reg.reorder_queued("a", 999).unwrap();
        assert_eq!(queued_order(&reg), ["b", "c", "a"]);
    }

    #[test]
    fn reorder_to_the_current_position_is_a_noop() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        reg.reorder_queued("b", 1).unwrap();
        assert_eq!(queued_order(&reg), ["a", "b", "c"]);
    }

    #[test]
    fn reorder_indexes_among_queued_only_and_keeps_running_jobs_in_place() {
        // `a` is running: it must stay at the head and be skipped when
        // `position` indexes the queued jobs, so a reorder can never promote a
        // queued job ahead of the job already executing.
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        reg.mark_running("a", Some(0));
        // Among the queued jobs [b, c], send c to slot 0.
        reg.reorder_queued("c", 0).unwrap();
        assert_eq!(queued_order(&reg), ["a", "c", "b"]);
        let snap = reg.snapshot();
        assert_eq!(snap.entries[0].state, JobLifecycle::Running);
        assert_eq!(snap.entries[1].id, "c");
        assert_eq!(snap.entries[1].state, JobLifecycle::Queued);
    }

    #[test]
    fn reorder_rejects_a_running_job() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.mark_running("a", Some(0));
        let err = reg.reorder_queued("a", 0).unwrap_err();
        assert_eq!(err, QueueReorderError::AlreadyRunning);
    }

    #[test]
    fn reorder_unknown_id_is_not_found() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        let err = reg.reorder_queued("ghost", 0).unwrap_err();
        assert_eq!(err, QueueReorderError::NotFound);
    }

    #[test]
    fn mark_running_is_a_noop_for_unknown_ids() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        // No panic, no insertion — bogus id is ignored entirely.
        reg.mark_running("not-here", Some(0));
        let snap = reg.snapshot();
        assert_eq!(snap.entries.len(), 1);
        assert_eq!(snap.entries[0].state, JobLifecycle::Queued);
    }

    #[test]
    fn progress_exists_only_for_the_live_job() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.record_progress(
            "a",
            &SseProgressEvent::Preview {
                image: "UFJFVklFVw==".into(),
                step: 3,
                total: 20,
            },
        );
        let progress = reg
            .progress_snapshot("a")
            .expect("live row")
            .expect("progress recorded");
        assert_eq!(progress.preview_image.as_deref(), Some("UFJFVklFVw=="));
        assert_eq!(progress.step, Some(3));
        assert_eq!(progress.total, Some(20));
        reg.remove("a");
        assert_eq!(reg.progress_snapshot("a"), None);
        reg.record_progress(
            "a",
            &SseProgressEvent::Preview {
                image: "TEFURQ==".into(),
                step: 20,
                total: 20,
            },
        );
        assert_eq!(reg.progress_snapshot("a"), None);
    }

    #[test]
    fn a_step_counter_is_recorded_with_previews_disabled() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.record_progress(
            "a",
            &SseProgressEvent::StageStart {
                name: "Denoising".into(),
            },
        );
        reg.record_progress(
            "a",
            &SseProgressEvent::DenoiseStep {
                step: 4,
                total: 8,
                elapsed_ms: 90,
            },
        );
        let progress = reg
            .progress_snapshot("a")
            .expect("live row")
            .expect("progress recorded");
        assert_eq!(progress.step, Some(4));
        assert_eq!(progress.total, Some(8));
        assert_eq!(progress.stage.as_deref(), Some("Denoising"));
        assert_eq!(progress.preview_image, None);
        assert!(progress.updated_at_ms > 0);
    }

    #[test]
    fn remove_compacts_positions_for_the_survivors() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        reg.remove("b");
        let snap = reg.snapshot();
        assert_eq!(snap.entries.len(), 2);
        assert_eq!(snap.entries[0].id, "a");
        assert_eq!(snap.entries[0].position, 0);
        assert_eq!(snap.entries[1].id, "c");
        assert_eq!(snap.entries[1].position, 1);
    }

    #[test]
    fn remove_is_idempotent_and_ignores_empty_ids() {
        // The worker's QueueSlot drop guard removes unconditionally — if the
        // dispatcher already removed the entry on an error path, the worker's
        // remove must not panic. Same for jobs that bypassed the registry
        // entirely (id == "").
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.remove("a");
        reg.remove("a"); // second remove is a no-op
        reg.remove("");
        reg.remove("never-existed");
        assert!(reg.is_empty());
    }

    #[test]
    fn cancel_queued_removes_the_entry() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.cancel_queued("a").unwrap();
        let snap = reg.snapshot();
        assert_eq!(snap.entries.len(), 1);
        assert_eq!(snap.entries[0].id, "b");
        assert_eq!(snap.entries[0].position, 0);
    }

    #[test]
    fn cancel_queued_latches_owner_token_installed_before_promotion() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        let token = mold_inference::InferenceCancellationToken::default();
        reg.install_running_cancellation("a", token.clone());
        assert_eq!(
            reg.cancel_queued("a"),
            Err(QueuedJobCancelError::AlreadyRunning)
        );
        assert!(token.is_cancelled());
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.claim_completion("a"), CompletionClaim::UserCancelled);
    }

    #[test]
    fn cancel_running_signals_the_exact_attempt_and_refuses_publication() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.mark_running("a", Some(0));
        let token = mold_inference::InferenceCancellationToken::default();
        reg.install_running_cancellation("a", token.clone());
        let err = reg.cancel_queued("a").unwrap_err();
        assert_eq!(err, QueuedJobCancelError::AlreadyRunning);
        assert!(token.is_cancelled());
        assert_eq!(reg.claim_completion("a"), CompletionClaim::UserCancelled);
        assert_eq!(reg.len(), 1, "running entry must survive a cancel attempt");
    }

    #[test]
    fn cancel_before_attempt_installation_is_latched() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.mark_running("a", Some(0));
        assert_eq!(
            reg.cancel_queued("a"),
            Err(QueuedJobCancelError::AlreadyRunning)
        );
        let token = mold_inference::InferenceCancellationToken::default();
        reg.install_running_cancellation("a", token.clone());
        assert!(token.is_cancelled());
    }

    #[test]
    fn completion_claim_wins_before_a_late_cancel() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.mark_running("a", Some(0));
        reg.install_running_cancellation(
            "a",
            mold_inference::InferenceCancellationToken::default(),
        );
        assert_eq!(reg.claim_completion("a"), CompletionClaim::Claimed);
        assert_eq!(
            reg.cancel_queued("a"),
            Err(QueuedJobCancelError::CompletionClaimed)
        );
        assert_eq!(reg.len(), 1);
        reg.finish_completion("a");
        assert!(reg.is_empty());
    }

    #[test]
    fn cancel_queued_unknown_id_is_not_found() {
        let reg = JobRegistry::new();
        let err = reg.cancel_queued("never-existed").unwrap_err();
        assert_eq!(err, QueuedJobCancelError::NotFound);
    }

    #[test]
    fn cancel_all_queued_removes_only_queued_and_returns_count() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.register("b", "sdxl:q8");
        reg.register("c", "ltx-video:q8");
        // `b` is running — it must survive the bulk cancel.
        reg.mark_running("b", Some(0));

        let cancelled = reg.cancel_all_queued();
        assert_eq!(cancelled, 2, "both queued jobs cancelled, running one kept");
        let snap = reg.snapshot();
        assert_eq!(snap.entries.len(), 1);
        assert_eq!(snap.entries[0].id, "b");
        assert_eq!(snap.entries[0].state, JobLifecycle::Running);
    }

    #[test]
    fn cancel_all_queued_on_empty_registry_returns_zero() {
        let reg = JobRegistry::new();
        assert_eq!(reg.cancel_all_queued(), 0);
    }

    #[tokio::test]
    async fn cancel_all_queued_signals_every_registered_waiter() {
        // Each queued job's cancel handle must resolve — cancel_all_queued
        // fires notify_one() per entry, so the permit survives even when the
        // cancel lands before the waiter awaits.
        let reg = JobRegistry::new();
        let cancel_a = reg.register("a", "flux-dev:fp16");
        let cancel_b = reg.register("b", "sdxl:q8");
        assert_eq!(reg.cancel_all_queued(), 2);
        tokio::time::timeout(std::time::Duration::from_secs(1), cancel_a.notified())
            .await
            .expect("cancel signal for a must resolve");
        tokio::time::timeout(std::time::Duration::from_secs(1), cancel_b.notified())
            .await
            .expect("cancel signal for b must resolve");
    }

    #[tokio::test]
    async fn cancel_queued_signals_the_registered_waiter() {
        // The handle returned by register() must resolve `notified()` even
        // when the cancel fires before the waiter starts awaiting — Notify
        // stores the permit from notify_one().
        let reg = JobRegistry::new();
        let cancel = reg.register("a", "flux-dev:fp16");
        reg.cancel_queued("a").unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), cancel.notified())
            .await
            .expect("cancel signal must resolve the waiter");
    }

    #[test]
    fn snapshot_serializes_with_snake_case_state_and_omits_gpu_when_queued() {
        // Wire contract: queued rows must NOT carry a `gpu` field at all
        // (clients shouldn't see `"gpu": null` and infer GPU 0). The state
        // tag is lowercase to match the rest of the SSE/JSON style.
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        let snap = reg.snapshot();
        let json = serde_json::to_string(&snap.entries[0]).unwrap();
        assert!(json.contains(r#""state":"queued""#), "got: {json}");
        assert!(
            !json.contains("gpu"),
            "queued row leaked a gpu field: {json}"
        );

        reg.mark_running("a", Some(0));
        let snap2 = reg.snapshot();
        let json2 = serde_json::to_string(&snap2.entries[0]).unwrap();
        assert!(json2.contains(r#""state":"running""#));
        assert!(json2.contains(r#""gpu":0"#));
    }

    #[test]
    fn snapshot_carries_request_metadata_only_when_registered_with_it() {
        let reg = JobRegistry::new();
        reg.register("plain", "flux-dev:fp16");
        let req: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:fp16","width":512,"height":512,"steps":4,"guidance":3.5}"#,
        )
        .expect("minimal request parses");
        let meta = Box::new(mold_core::OutputMetadata::from_generate_request(
            &req,
            0,
            None,
            "test-version",
        ));
        reg.register_job("rich", "flux-dev:fp16", None, Some(true), Some(meta));

        let snap = reg.snapshot();
        // Wire contract: rows without settings omit the key entirely.
        let plain_json = serde_json::to_string(&snap.entries[0]).unwrap();
        assert!(!plain_json.contains("metadata"), "got: {plain_json}");
        assert_eq!(snap.entries[1].seed_pinned, Some(true));
        let rich = snap.entries[1].metadata.as_ref().expect("metadata rides");
        assert_eq!(rich.prompt, "a cat");
        assert_eq!(rich.width, 512);
    }

    #[test]
    fn scheduler_projection_matches_wire_queue_semantics_without_payload_fields() {
        let reg = JobRegistry::new();
        let mut request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"payload","model":"flux-dev:fp16","width":512,"height":512,"steps":4,"guidance":3.5}"#,
        )
        .expect("minimal request parses");
        request.prompt = "metadata".repeat(8 * 1024);
        let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
            &request,
            0,
            None,
            "test-version",
        ));
        reg.register_job(
            "queued",
            "flux-dev:fp16",
            Some(2),
            Some(true),
            Some(metadata),
        );
        reg.record_progress(
            "queued",
            &SseProgressEvent::Preview {
                image: "preview".repeat(8 * 1024),
                step: 3,
                total: 20,
            },
        );
        reg.register("running", "sdxl:q8");
        reg.mark_running("running", Some(1));

        let full = reg.snapshot();
        let scheduler = reg.scheduler_snapshot();
        let full_semantics = full
            .entries
            .iter()
            .map(|entry| {
                (
                    entry.id.as_str(),
                    entry.state,
                    entry.position,
                    entry.target_gpu,
                )
            })
            .collect::<Vec<_>>();
        let scheduler_semantics = scheduler
            .iter()
            .map(|entry| {
                (
                    entry.id.as_str(),
                    entry.state,
                    entry.position,
                    entry.target_gpu,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(scheduler_semantics, full_semantics);

        // Exhaustive destructuring is a compile-time guard on the hot-path
        // contract: adding metadata, previews, request bodies, or any other
        // field to this type must make this test fail to compile.
        let SchedulerQueueEntry {
            id,
            state,
            position,
            target_gpu,
        } = &scheduler[0];
        assert_eq!(id, "queued");
        assert_eq!(*state, JobLifecycle::Queued);
        assert_eq!(*position, 0);
        assert_eq!(*target_gpu, Some(2));
        assert_eq!(
            reg.scheduler_lifecycle("queued"),
            Some(JobLifecycle::Queued)
        );
        assert_eq!(
            reg.scheduler_lifecycle("running"),
            Some(JobLifecycle::Running)
        );
        assert_eq!(reg.scheduler_lifecycle("missing"), None);
        assert!(full.entries[0].metadata.is_some());
        assert!(reg.progress_snapshot("queued").flatten().is_some());
    }

    #[test]
    fn scheduler_projection_work_is_structurally_independent_of_payload_size_at_depth() {
        fn projection(depth: usize, payload_bytes: usize) -> Vec<SchedulerQueueEntry> {
            let reg = JobRegistry::new();
            for index in 0..depth {
                let mut request: mold_core::GenerateRequest = serde_json::from_str(
                    r#"{"prompt":"","model":"flux-dev:fp16","width":512,"height":512,"steps":4,"guidance":3.5}"#,
                )
                .expect("minimal request parses");
                request.prompt = "m".repeat(payload_bytes);
                let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
                    &request,
                    index as u64,
                    None,
                    "test-version",
                ));
                let id = format!("job-{index:03}");
                reg.register_job(
                    &id,
                    "flux-dev:fp16",
                    Some(index % 3),
                    Some(index % 2 == 0),
                    Some(metadata),
                );
                reg.record_progress(
                    &id,
                    &SseProgressEvent::Preview {
                        image: "p".repeat(payload_bytes),
                        step: 1,
                        total: 2,
                    },
                );
                if index % 5 == 0 {
                    reg.mark_running(&id, Some(index % 3));
                }
            }
            reg.scheduler_snapshot()
        }

        const DEPTH: usize = 64;
        let empty_payloads = projection(DEPTH, 0);
        let large_payloads = projection(DEPTH, 64 * 1024);

        // This is deliberately not a timing assertion. The exhaustive entry
        // shape above and equality here establish that projection work is one
        // fixed-size row per queue entry, regardless of retained payload size.
        assert_eq!(large_payloads, empty_payloads);
        assert_eq!(large_payloads.len(), DEPTH);
    }

    mod event_emission {
        use super::*;
        use crate::events::EventBroadcaster;
        use mold_core::ServerEvent;
        use tokio::sync::broadcast::error::TryRecvError;

        fn wired() -> (
            SharedJobRegistry,
            tokio::sync::broadcast::Receiver<ServerEvent>,
        ) {
            let events = EventBroadcaster::new();
            let rx = events.subscribe();
            (JobRegistry::with_events(events), rx)
        }

        #[test]
        fn register_emits_job_queued() {
            let (reg, mut rx) = wired();
            reg.register("a", "flux-dev:fp16");
            match rx.try_recv().unwrap() {
                ServerEvent::JobQueued { id, model } => {
                    assert_eq!(id, "a");
                    assert_eq!(model, "flux-dev:fp16");
                }
                other => panic!("expected job_queued, got {other:?}"),
            }
        }

        #[test]
        fn mark_running_emits_job_started_with_model_and_gpu() {
            let (reg, mut rx) = wired();
            reg.register("a", "flux-dev:fp16");
            let _ = rx.try_recv(); // drain job_queued
            reg.mark_running("a", Some(1));
            match rx.try_recv().unwrap() {
                ServerEvent::JobStarted { id, model, gpu } => {
                    assert_eq!(id, "a");
                    assert_eq!(model, "flux-dev:fp16");
                    assert_eq!(gpu, Some(1));
                }
                other => panic!("expected job_started, got {other:?}"),
            }
        }

        #[test]
        fn mark_running_unknown_id_emits_nothing() {
            let (reg, mut rx) = wired();
            reg.mark_running("ghost", None);
            assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
        }

        #[test]
        fn remove_emits_job_ended_exactly_once_across_double_call() {
            let (reg, mut rx) = wired();
            reg.register("a", "flux-dev:fp16");
            let _ = rx.try_recv(); // drain job_queued
            reg.remove("a");
            reg.remove("a"); // idempotent second call on another terminal path
            match rx.try_recv().unwrap() {
                ServerEvent::JobEnded { id } => assert_eq!(id, "a"),
                other => panic!("expected job_ended, got {other:?}"),
            }
            assert!(
                matches!(rx.try_recv(), Err(TryRecvError::Empty)),
                "second remove must not emit a duplicate job_ended"
            );
        }

        #[test]
        fn cancel_queued_emits_job_ended() {
            let (reg, mut rx) = wired();
            reg.register("a", "flux-dev:fp16");
            let _ = rx.try_recv(); // drain job_queued
            reg.cancel_queued("a").unwrap();
            match rx.try_recv().unwrap() {
                ServerEvent::JobEnded { id } => assert_eq!(id, "a"),
                other => panic!("expected job_ended, got {other:?}"),
            }
        }

        #[test]
        fn cancel_all_queued_emits_exactly_one_job_ended_per_cancelled_job() {
            let (reg, mut rx) = wired();
            reg.register("a", "flux-dev:fp16");
            reg.register("b", "sdxl:q8");
            reg.register("c", "ltx-video:q8");
            reg.mark_running("c", Some(0));
            // Drain the three job_queued + one job_started emissions.
            while rx.try_recv().is_ok() {}

            assert_eq!(reg.cancel_all_queued(), 2);
            let mut ended = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    ServerEvent::JobEnded { id } => ended.push(id),
                    other => panic!("expected only job_ended, got {other:?}"),
                }
            }
            ended.sort();
            assert_eq!(ended, vec!["a".to_string(), "b".to_string()]);
        }
    }
}
