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

use serde::Serialize;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

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
}

/// The registry itself. Construct via `JobRegistry::new` and share through
/// `AppState`. All mutation is fire-and-forget — if the inner lock is
/// poisoned (extremely unlikely in practice) we recover from the inner
/// state rather than propagating the panic into the dispatcher hot path.
pub struct JobRegistry {
    inner: RwLock<Vec<EntryInternal>>,
}

/// Cheap-cloneable handle. Workers and routes pass this around by value.
pub type SharedJobRegistry = Arc<JobRegistry>;

impl JobRegistry {
    pub fn new() -> SharedJobRegistry {
        Arc::new(Self {
            inner: RwLock::new(Vec::new()),
        })
    }

    /// Insert a freshly-submitted job at the tail in `Queued` state.
    pub fn register(&self, id: impl Into<String>, model: impl Into<String>) {
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
        entries.push(EntryInternal {
            id: id.into(),
            model: model.into(),
            state: JobLifecycle::Queued,
            started_at_unix_ms,
            gpu: None,
        });
    }

    /// Promote a registry entry from `Queued` to `Running`. No-op if `id`
    /// isn't present (the entry may have been removed concurrently).
    pub fn mark_running(&self, id: &str, gpu: Option<usize>) {
        let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = entries.iter_mut().find(|e| e.id == id) {
            e.state = JobLifecycle::Running;
            e.gpu = gpu;
        }
    }

    /// Drop the entry — call once on every terminal path (success, error,
    /// client-disconnect skip, dispatch failure). Idempotent.
    pub fn remove(&self, id: &str) {
        if id.is_empty() {
            return;
        }
        let mut entries = self.inner.write().unwrap_or_else(|e| e.into_inner());
        entries.retain(|e| e.id != id);
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
}
