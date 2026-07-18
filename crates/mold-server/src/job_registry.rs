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
use mold_core::ServerEvent;
use serde::Serialize;
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
}

/// One row in the `GET /api/queue` response.
///
/// `position` is the 0-based FIFO index at the time of the snapshot — 0 is at
/// the head (about to be dispatched, or already running on a worker), N-1 is
/// the most recently submitted. Position is derived from insertion order, so
/// it shifts as earlier jobs finish and drop out.
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
}

/// Whole-queue listing returned by `GET /api/queue`. Wrapped in a struct so
/// the response can grow extra fields (totals, byte counts, etc.) without a
/// breaking change.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct QueueListing {
    pub entries: Vec<JobEntry>,
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
    /// Cancellation signal for `DELETE /api/queue/:id`. The submitting
    /// handler holds the clone returned by `register*()` and selects on
    /// `notified()` alongside the job's result channel; `cancel_queued`
    /// fires `notify_one()` so the permit survives even when the cancel
    /// lands before the waiter starts awaiting.
    cancel: Arc<Notify>,
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
}

/// The registry itself. Construct via `JobRegistry::new` and share through
/// `AppState`. All mutation is fire-and-forget — if the inner lock is
/// poisoned (extremely unlikely in practice) we recover from the inner
/// state rather than propagating the panic into the dispatcher hot path.
pub struct JobRegistry {
    inner: RwLock<Vec<EntryInternal>>,
    /// Optional lifecycle broadcast (`GET /api/events`). Emitting from the
    /// registry — rather than each call site — guarantees every submit /
    /// promote / terminal path produces exactly one event. `None` keeps
    /// event-less construction (tests) cheap.
    events: Option<Arc<EventBroadcaster>>,
}

/// Cheap-cloneable handle. Workers and routes pass this around by value.
pub type SharedJobRegistry = Arc<JobRegistry>;

impl JobRegistry {
    pub fn new() -> SharedJobRegistry {
        Arc::new(Self {
            inner: RwLock::new(Vec::new()),
            events: None,
        })
    }

    /// Like [`JobRegistry::new`] but mirrors every lifecycle change onto the
    /// server-wide event broadcast.
    pub fn with_events(events: Arc<EventBroadcaster>) -> SharedJobRegistry {
        Arc::new(Self {
            inner: RwLock::new(Vec::new()),
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
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let id = id.into();
        let model = model.into();
        let cancel = Arc::new(Notify::new());
        {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            entries.push(EntryInternal {
                id: id.clone(),
                model: model.clone(),
                state: JobLifecycle::Queued,
                started_at_unix_ms,
                gpu: None,
                target_gpu,
                seed_pinned,
                metadata,
                cancel: cancel.clone(),
            });
        }
        self.emit(ServerEvent::JobQueued { id, model });
        cancel
    }

    /// Cancel a still-queued job: remove its entry and fire the cancel
    /// signal returned by `register*()` so the waiting request future
    /// resolves with a cancellation error. Running jobs are not cancelable
    /// — the GPU worker owns them and there is no safe preemption point.
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
            if entries[pos].state == JobLifecycle::Running {
                return Err(QueuedJobCancelError::AlreadyRunning);
            }
            let entry = entries.remove(pos);
            entry.cancel.notify_one();
        }
        self.emit(ServerEvent::JobEnded { id: id.to_string() });
        Ok(())
    }

    /// Cancel every still-queued job in one pass, backing `DELETE /api/queue`.
    /// Under a single write lock this removes each `Queued` entry and fires its
    /// cancel signal; running jobs are left untouched (same rule as
    /// [`cancel_queued`](Self::cancel_queued) — a GPU worker owns them). After
    /// dropping the lock it emits one `JobEnded` per cancelled job (the
    /// emit-outside-lock discipline). Returns the number of jobs cancelled.
    pub fn cancel_all_queued(&self) -> usize {
        let cancelled_ids = {
            let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
            let mut ids = Vec::new();
            entries.retain(|e| {
                if e.state == JobLifecycle::Queued {
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
        cancelled_ids.len()
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
            self.emit(ServerEvent::JobStarted {
                id: id.to_string(),
                model,
                gpu,
            });
        }
    }

    pub fn set_target_gpu(
        &self,
        id: &str,
        target_gpu: Option<usize>,
    ) -> Result<(), TargetGpuUpdateError> {
        let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
        let Some(e) = entries.iter_mut().find(|e| e.id == id) else {
            return Err(TargetGpuUpdateError::NotFound);
        };
        if e.state == JobLifecycle::Running {
            return Err(TargetGpuUpdateError::AlreadyRunning);
        }
        e.target_gpu = target_gpu;
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
            })
        })
    }

    /// Drop the entry — call once on every terminal path (success, error,
    /// client-disconnect skip, dispatch failure). Idempotent.
    pub fn remove(&self, id: &str) {
        if id.is_empty() {
            return;
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
            self.emit(ServerEvent::JobEnded { id: id.to_string() });
        }
    }

    /// Snapshot the registry as a wire-shaped listing. Positions are assigned
    /// in insertion order at snapshot time.
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
            })
            .collect();
        QueueListing { entries: out }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn mark_running_flips_state_and_records_gpu_ordinal() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.mark_running("a", Some(1));
        let snap = reg.snapshot();
        assert_eq!(snap.entries[0].state, JobLifecycle::Running);
        assert_eq!(snap.entries[0].gpu, Some(1));
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
    fn cancel_queued_rejects_running_jobs_and_keeps_the_entry() {
        let reg = JobRegistry::new();
        reg.register("a", "flux-dev:fp16");
        reg.mark_running("a", Some(0));
        let err = reg.cancel_queued("a").unwrap_err();
        assert_eq!(err, QueuedJobCancelError::AlreadyRunning);
        assert_eq!(reg.len(), 1, "running entry must survive a cancel attempt");
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
