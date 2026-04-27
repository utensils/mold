# Downloads UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser-driven model download queue with progress, cancel, auto-retry, and persistent SSE streaming, so users can pull any manifest model from the web UI without shelling into the GPU host.

**Architecture:** A single-writer `DownloadQueue` lives in `AppState`, driven by one long-running tokio task that pulls jobs off a `VecDeque` and wraps the existing `mold_core::download::pull_model_with_callback` with a `CancellationToken`. Progress events fan out over a `tokio::sync::broadcast::Sender<DownloadEvent>`. A new SSE endpoint `GET /api/downloads/stream` multiplexes those events; a new `useDownloads` Vue composable consumes them as a singleton mounted in `App.vue`, feeding `DownloadsDrawer.vue` (new) and `ModelPicker.vue` (updated in place).

**Tech Stack:** Rust, axum, tokio (broadcast/mpsc/CancellationToken), hf-hub, Vue 3, Vitest, bun

**Spec:** `docs/superpowers/specs/2026-04-19-model-ui-overhaul-design.md` — sections §1, §4.1, §4.2, §4.3

**Worktree:** This plan should be executed in `wt/downloads-ui` off branch `feat/model-ui-overhaul`. Create the worktree with `git worktree add ../mold-downloads wt/downloads-ui feat/model-ui-overhaul` before starting Task 1.

**Agent boundary:** You own the download queue. Other agents own resources (§2) and placement (§3). Do NOT touch `/api/resources*` or `DevicePlacement`. Keep changes to shared files (`types.rs`, `routes.rs`, `state.rs`, `lib.rs`, `App.vue`, `TopBar.vue`) confined to clearly-labeled additive sections — see spec §4.2.

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `crates/mold-server/src/downloads.rs` | `DownloadJob`, `JobStatus`, `DownloadQueue`, `DownloadEvent`, driver task, cancellation with partial cleanup, 1x auto-retry. |
| `crates/mold-server/src/downloads_test.rs` | Unit tests: queue transitions, cancel while active, cancel while queued, retry-on-failure via a faked pull. |
| `web/src/composables/useDownloads.ts` | Singleton Vue composable wrapping `/api/downloads/stream` + `/api/downloads`, reactive `active`/`queued`/`history`, client-side ETA math. |
| `web/src/composables/useDownloads.test.ts` | ETA calc, SSE reconnect, history truncation. |
| `web/src/components/DownloadsDrawer.vue` | Drawer with active/queued/recent sections, cancel+retry buttons. |
| `web/src/components/DownloadsDrawer.test.ts` | Renders active/queued/recent from mock state. |

### Modified Files

| File | What Changes |
|------|-------------|
| `crates/mold-core/src/types.rs` | Additive: `DownloadJob`, `JobStatus`, `DownloadEvent` serde types under a clearly labeled "Downloads UI (Agent A)" section. |
| `crates/mold-server/src/lib.rs` | `pub mod downloads;` + spawn the driver task in `run_server`. |
| `crates/mold-server/src/state.rs` | Add `pub downloads: Arc<DownloadQueue>` field to `AppState`; thread through constructors. |
| `crates/mold-server/src/routes.rs` | New routes: `POST /api/downloads`, `DELETE /api/downloads/:id`, `GET /api/downloads`, `GET /api/downloads/stream`. Rewire `POST /api/models/pull` to delegate to the queue. |
| `crates/mold-server/src/routes_test.rs` | New tests for the four new routes + duplicate enqueue idempotency. |
| `web/src/types.ts` | Additive: `DownloadJobWire`, `JobStatusWire`, `DownloadEventWire` under a "Downloads UI (Agent A)" section. |
| `web/src/api.ts` | `postDownload`, `cancelDownload`, `fetchDownloads` helpers. |
| `web/src/components/ModelPicker.vue` | Add `(3.2 GB)` size column, inline Download button for non-downloaded models, progress bar for active, queued chip for queued. |
| `web/src/App.vue` | Mount `useDownloads()` singleton and `<DownloadsDrawer />`. |
| `web/src/components/TopBar.vue` | Add Downloads button + badge (active + queued count). |

---

## Task 1: Create worktree

**Files:** None (shell only)

- [ ] **Step 1: Create the worktree**

Run from the repo root:

```bash
git fetch origin
git worktree add ../mold-downloads wt/downloads-ui feat/model-ui-overhaul
cd ../mold-downloads
```

Expected output:

```
Preparing worktree (new branch 'wt/downloads-ui')
HEAD is now at <sha> <subject>
```

- [ ] **Step 2: Verify branch + clean tree**

```bash
git -C ../mold-downloads rev-parse --abbrev-ref HEAD
git -C ../mold-downloads status --short
```

Expected:

```
wt/downloads-ui
```

(empty `status --short`).

All subsequent tasks run inside `../mold-downloads`.

---

## Task 2: Add shared serde types (`mold-core/src/types.rs`)

**Files:**
- Modify: `crates/mold-core/src/types.rs`

Depends on: Task 1

- [ ] **Step 1: Write failing serde round-trip test**

Append to `crates/mold-core/src/types.rs` under a new `#[cfg(test)]` section (or, if one exists, inside it). Use the existing file-level test module — most types.rs files in this crate have tests at the bottom. Add:

```rust
#[cfg(test)]
mod downloads_types_tests {
    use super::*;

    #[test]
    fn job_status_serde_snake_case() {
        let cases = [
            (JobStatus::Queued, "\"queued\""),
            (JobStatus::Active, "\"active\""),
            (JobStatus::Completed, "\"completed\""),
            (JobStatus::Failed, "\"failed\""),
            (JobStatus::Cancelled, "\"cancelled\""),
        ];
        for (status, wire) in cases {
            let s = serde_json::to_string(&status).unwrap();
            assert_eq!(s, wire);
            let back: JobStatus = serde_json::from_str(&s).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn download_job_round_trip() {
        let job = DownloadJob {
            id: "11111111-1111-1111-1111-111111111111".to_string(),
            model: "flux-dev:q4".to_string(),
            status: JobStatus::Active,
            files_done: 2,
            files_total: 5,
            bytes_done: 1_000_000,
            bytes_total: 3_000_000,
            current_file: Some("transformer.gguf".to_string()),
            started_at: Some(1_700_000_000_000),
            completed_at: None,
            error: None,
        };
        let s = serde_json::to_string(&job).unwrap();
        let back: DownloadJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id, job.id);
        assert_eq!(back.model, job.model);
        assert_eq!(back.status, JobStatus::Active);
        assert_eq!(back.files_done, 2);
        assert_eq!(back.files_total, 5);
        assert_eq!(back.bytes_done, 1_000_000);
        assert_eq!(back.bytes_total, 3_000_000);
        assert_eq!(back.current_file.as_deref(), Some("transformer.gguf"));
    }

    #[test]
    fn download_event_enqueued_tag_shape() {
        let evt = DownloadEvent::Enqueued {
            id: "abc".to_string(),
            model: "flux-dev:q4".to_string(),
            position: 2,
        };
        let s = serde_json::to_string(&evt).unwrap();
        assert!(s.contains("\"type\":\"enqueued\""), "wire: {s}");
        assert!(s.contains("\"id\":\"abc\""), "wire: {s}");
        assert!(s.contains("\"model\":\"flux-dev:q4\""), "wire: {s}");
        assert!(s.contains("\"position\":2"), "wire: {s}");
    }

    #[test]
    fn download_event_progress_tag_shape() {
        let evt = DownloadEvent::Progress {
            id: "abc".to_string(),
            files_done: 1,
            bytes_done: 2_000_000,
            current_file: Some("clip.safetensors".to_string()),
        };
        let s = serde_json::to_string(&evt).unwrap();
        assert!(s.contains("\"type\":\"progress\""), "wire: {s}");
        assert!(s.contains("\"bytes_done\":2000000"), "wire: {s}");
    }
}
```

- [ ] **Step 2: Run the failing test**

```bash
cargo test -p mold-ai-core downloads_types_tests --lib
```

Expected: fails with `error[E0412]: cannot find type 'JobStatus' in this scope` and similar for `DownloadJob`, `DownloadEvent`.

- [ ] **Step 3: Implement the types**

Append to `crates/mold-core/src/types.rs` (clearly fenced to avoid Agent B/C conflicts):

```rust
// ─── Downloads UI (Agent A) ─────────────────────────────────────────────────

/// Lifecycle state of a download job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Active,
    Completed,
    Failed,
    Cancelled,
}

/// Download queue entry. Mirrored 1:1 on the wire; the SPA consumes this as
/// `DownloadJobWire` in `web/src/types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJob {
    pub id: String,
    pub model: String,
    pub status: JobStatus,
    pub files_done: usize,
    pub files_total: usize,
    pub bytes_done: u64,
    pub bytes_total: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Internally tagged enum — on the wire each variant is `{"type": "...", ...}`.
/// Keep `#[serde(tag = "type", rename_all = "snake_case")]` stable; the SPA's
/// `DownloadEventWire` union in `types.ts` depends on this exact shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DownloadEvent {
    Enqueued {
        id: String,
        model: String,
        position: usize,
    },
    Dequeued {
        id: String,
    },
    Started {
        id: String,
        files_total: usize,
        bytes_total: u64,
    },
    Progress {
        id: String,
        files_done: usize,
        bytes_done: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        current_file: Option<String>,
    },
    FileDone {
        id: String,
        filename: String,
    },
    JobDone {
        id: String,
        model: String,
    },
    JobFailed {
        id: String,
        error: String,
    },
    JobCancelled {
        id: String,
    },
}

/// Listing returned from `GET /api/downloads`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadsListing {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<DownloadJob>,
    pub queued: Vec<DownloadJob>,
    pub history: Vec<DownloadJob>,
}
```

- [ ] **Step 4: Re-run the test**

```bash
cargo test -p mold-ai-core downloads_types_tests --lib
```

Expected: `test result: ok. 4 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-core/src/types.rs
git commit -m "$(cat <<'EOF'
feat(downloads): add DownloadJob / JobStatus / DownloadEvent serde types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `DownloadQueue` skeleton (empty module + AppState wiring)

**Files:**
- Create: `crates/mold-server/src/downloads.rs`
- Modify: `crates/mold-server/src/lib.rs`
- Modify: `crates/mold-server/src/state.rs`

Depends on: Task 2

- [ ] **Step 1: Write failing smoke test**

Create `crates/mold-server/src/downloads_test.rs`:

```rust
//! Tests for the download queue. Run with:
//!   cargo test -p mold-ai-server downloads --lib
//!
//! These tests never touch HuggingFace — they inject a fake `PullDriver` so
//! the queue logic can be exercised in isolation.

use crate::downloads::{DownloadQueue, JobStatus};
use std::sync::Arc;

#[tokio::test]
async fn queue_starts_empty() {
    let queue = DownloadQueue::new_for_test();
    let listing = queue.listing().await;
    assert!(listing.active.is_none());
    assert!(listing.queued.is_empty());
    assert!(listing.history.is_empty());
}

#[tokio::test]
async fn enqueue_unknown_model_errors() {
    let queue = DownloadQueue::new_for_test();
    let err = queue.enqueue("no-such-model:xyz".to_string()).await;
    assert!(err.is_err(), "expected unknown-model error");
}
```

- [ ] **Step 2: Run it — expect compile error**

```bash
cargo test -p mold-ai-server downloads --lib
```

Expected: `error[E0432]: unresolved import 'crate::downloads'` (the module does not yet exist).

- [ ] **Step 3: Implement the skeleton module**

Create `crates/mold-server/src/downloads.rs`:

```rust
//! Single-writer download queue wrapping `mold_core::download::pull_model_with_callback`.
//!
//! One long-running driver task pulls jobs off a `VecDeque`, spawns a pull
//! task per job, forwards `DownloadProgressEvent` → `DownloadEvent` on a
//! broadcast channel, and cleans up on completion or cancellation.
//!
//! **Agent A boundary** — this module is owned by the Downloads phase. Do
//! not take a lock on resources (Agent B) or placement (Agent C) from here.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};

use mold_core::types::{DownloadEvent, DownloadJob, DownloadsListing, JobStatus};
use tokio::sync::{broadcast, Mutex as AsyncMutex, Notify};
use tokio_util::sync::CancellationToken;

/// Capacity of the broadcast channel. Slow subscribers see the oldest events
/// lag (we accept this — the SPA will also re-fetch `/api/downloads` on reconnect).
pub const EVENT_BUFFER: usize = 256;

/// Max history entries retained for the drawer's "Recent" section.
pub const HISTORY_CAP: usize = 20;

/// Active download handle.
pub struct ActiveHandle {
    pub job: DownloadJob,
    pub abort: CancellationToken,
    pub task: tokio::task::JoinHandle<()>,
}

pub struct DownloadQueue {
    active: AsyncMutex<Option<ActiveHandle>>,
    queued: StdMutex<VecDeque<DownloadJob>>,
    history: StdMutex<VecDeque<DownloadJob>>,
    events: broadcast::Sender<DownloadEvent>,
    /// Wakes the driver task when new work arrives.
    notify: Notify,
}

#[derive(Debug, thiserror::Error)]
pub enum EnqueueError {
    #[error("unknown model '{0}'. Run 'mold list' to see available models.")]
    UnknownModel(String),
    #[error("download queue lock poisoned")]
    LockPoisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// Already in the queue or currently active — no new job created.
    AlreadyPresent,
    /// Created a new job at the given 0-based position (0 = will start next).
    Created,
}

impl DownloadQueue {
    pub fn new() -> Arc<Self> {
        let (events, _rx) = broadcast::channel(EVENT_BUFFER);
        Arc::new(Self {
            active: AsyncMutex::new(None),
            queued: StdMutex::new(VecDeque::new()),
            history: StdMutex::new(VecDeque::new()),
            events,
            notify: Notify::new(),
        })
    }

    /// Testing helper — same as `new()` but returns a non-`Arc` for
    /// direct `await` use. Callers that need the driver running must wrap in Arc.
    #[cfg(test)]
    pub fn new_for_test() -> Arc<Self> {
        Self::new()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DownloadEvent> {
        self.events.subscribe()
    }

    /// Returns the current active/queued/history snapshot.
    pub async fn listing(&self) -> DownloadsListing {
        let active = self.active.lock().await.as_ref().map(|a| a.job.clone());
        let queued: Vec<DownloadJob> = self
            .queued
            .lock()
            .expect("queued lock poisoned")
            .iter()
            .cloned()
            .collect();
        let history: Vec<DownloadJob> = self
            .history
            .lock()
            .expect("history lock poisoned")
            .iter()
            .cloned()
            .collect();
        DownloadsListing {
            active,
            queued,
            history,
        }
    }

    /// Enqueue a model for download. Returns `(job_id, position, outcome)`.
    /// Position 0 means the job will run immediately after the driver wakes;
    /// N means N jobs are ahead. Duplicate enqueue returns the existing id.
    pub async fn enqueue(
        &self,
        model: String,
    ) -> Result<(String, usize, EnqueueOutcome), EnqueueError> {
        // Manifest validation up front so the caller gets a real 400 instead of a
        // background failure.
        let canonical = mold_core::manifest::resolve_model_name(&model);
        if mold_core::manifest::find_manifest(&canonical).is_none() {
            return Err(EnqueueError::UnknownModel(model));
        }

        // Check for active/queued duplicate.
        {
            let active = self.active.lock().await;
            if let Some(handle) = active.as_ref() {
                if handle.job.model == canonical {
                    return Ok((handle.job.id.clone(), 0, EnqueueOutcome::AlreadyPresent));
                }
            }
        }
        {
            let queued = self.queued.lock().map_err(|_| EnqueueError::LockPoisoned)?;
            if let Some((idx, existing)) =
                queued.iter().enumerate().find(|(_, j)| j.model == canonical)
            {
                return Ok((existing.id.clone(), idx + 1, EnqueueOutcome::AlreadyPresent));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let job = DownloadJob {
            id: id.clone(),
            model: canonical.clone(),
            status: JobStatus::Queued,
            files_done: 0,
            files_total: 0,
            bytes_done: 0,
            bytes_total: 0,
            current_file: None,
            started_at: None,
            completed_at: None,
            error: None,
        };

        let position = {
            let mut queued = self.queued.lock().map_err(|_| EnqueueError::LockPoisoned)?;
            queued.push_back(job);
            queued.len() // position shown to the user is 1-based in the drawer
        };
        let _ = self.events.send(DownloadEvent::Enqueued {
            id: id.clone(),
            model: canonical,
            position,
        });
        self.notify.notify_one();
        Ok((id, position, EnqueueOutcome::Created))
    }

    /// Cancel an in-flight or queued download. Returns `true` if a job was
    /// found and cancelled; `false` if the id is unknown.
    pub async fn cancel(&self, id: &str) -> bool {
        // Check queued first — cheap.
        {
            let mut queued = self.queued.lock().expect("queued lock poisoned");
            if let Some(pos) = queued.iter().position(|j| j.id == id) {
                queued.remove(pos);
                drop(queued);
                let _ = self.events.send(DownloadEvent::Dequeued { id: id.to_string() });
                return true;
            }
        }
        // Active?
        let mut active = self.active.lock().await;
        if let Some(handle) = active.as_ref() {
            if handle.job.id == id {
                handle.abort.cancel();
                // Driver task will observe the cancel, emit JobCancelled, and clear `active`.
                return true;
            }
        }
        // Not found.
        false
    }

    /// Called by the driver task to push a finished job into history and keep
    /// the most recent 20.
    pub(crate) fn push_history(&self, job: DownloadJob) {
        let mut history = self.history.lock().expect("history lock poisoned");
        if history.len() >= HISTORY_CAP {
            history.pop_front();
        }
        history.push_back(job);
    }

    /// Wait until either new work is notified or `shutdown` fires.
    pub(crate) async fn wait_for_work(&self, shutdown: &CancellationToken) {
        tokio::select! {
            _ = self.notify.notified() => {},
            _ = shutdown.cancelled() => {},
        }
    }

    pub(crate) fn take_next_queued(&self) -> Option<DownloadJob> {
        self.queued
            .lock()
            .expect("queued lock poisoned")
            .pop_front()
    }

    pub(crate) async fn set_active(&self, handle: ActiveHandle) {
        *self.active.lock().await = Some(handle);
    }

    pub(crate) async fn clear_active(&self) -> Option<ActiveHandle> {
        self.active.lock().await.take()
    }

    pub(crate) async fn with_active<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut DownloadJob) -> R,
    {
        let mut active = self.active.lock().await;
        active.as_mut().map(|a| f(&mut a.job))
    }

    pub(crate) fn emit(&self, event: DownloadEvent) {
        let _ = self.events.send(event);
    }
}

#[cfg(test)]
#[path = "downloads_test.rs"]
mod tests;
```

- [ ] **Step 4: Register the module and add `uuid` dependency**

Open `crates/mold-server/Cargo.toml` and verify `uuid` is available. If not, add to `[dependencies]`:

```toml
uuid = { version = "1", features = ["v4"] }
tokio-util = { version = "0.7", features = ["rt"] }
```

Then modify `crates/mold-server/src/lib.rs`. Find the `pub mod …;` block near the top and append (under an `// Agent A (downloads)` comment to keep the merge clean):

```rust
// Agent A (downloads)
pub mod downloads;
```

- [ ] **Step 5: Thread `downloads` into `AppState`**

Modify `crates/mold-server/src/state.rs`. Under the existing `use` block, add:

```rust
use crate::downloads::DownloadQueue;
```

Inside `pub struct AppState { ... }`, in a clearly fenced additive section at the bottom of the field list (before the closing brace), add:

```rust
    // ── Downloads UI (Agent A) ──────────────────────────────────────────────
    /// Single-writer download queue.
    pub downloads: Arc<DownloadQueue>,
```

Update every constructor to initialize it. In `AppState::new`, `AppState::empty`, `AppState::with_engine`, and `AppState::with_engine_and_queue`, inside the struct literal add:

```rust
            downloads: DownloadQueue::new(),
```

- [ ] **Step 6: Re-run the two smoke tests**

```bash
cargo test -p mold-ai-server downloads --lib
```

Expected: 2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/mold-server/src/downloads.rs \
        crates/mold-server/src/downloads_test.rs \
        crates/mold-server/src/lib.rs \
        crates/mold-server/src/state.rs \
        crates/mold-server/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(downloads): add DownloadQueue skeleton + AppState wiring

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Driver task with injectable puller (TDD for happy path)

**Files:**
- Modify: `crates/mold-server/src/downloads.rs`
- Modify: `crates/mold-server/src/downloads_test.rs`

Depends on: Task 3

- [ ] **Step 1: Write failing happy-path test**

Append to `crates/mold-server/src/downloads_test.rs`:

```rust
use crate::downloads::{spawn_driver, PullDriver};
use mold_core::types::DownloadEvent;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Fake pull driver that emits a single `Progress` then returns `Ok`.
#[derive(Clone, Default)]
struct FakePuller {
    pub called: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl PullDriver for FakePuller {
    async fn pull(
        &self,
        _model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        self.called.store(true, Ordering::SeqCst);
        // Emit one Status + one FileProgress event so the driver sees work.
        on_progress(mold_core::download::DownloadProgressEvent::Status {
            message: "starting".into(),
        });
        on_progress(mold_core::download::DownloadProgressEvent::FileStart {
            filename: "transformer.gguf".into(),
            file_index: 0,
            total_files: 1,
            size_bytes: 1_000,
            batch_bytes_downloaded: 0,
            batch_bytes_total: 1_000,
            batch_elapsed_ms: 0,
        });
        on_progress(mold_core::download::DownloadProgressEvent::FileProgress {
            filename: "transformer.gguf".into(),
            file_index: 0,
            bytes_downloaded: 500,
            bytes_total: 1_000,
            batch_bytes_downloaded: 500,
            batch_bytes_total: 1_000,
            batch_elapsed_ms: 5,
        });
        on_progress(mold_core::download::DownloadProgressEvent::FileDone {
            filename: "transformer.gguf".into(),
            file_index: 0,
            total_files: 1,
            batch_bytes_downloaded: 1_000,
            batch_bytes_total: 1_000,
            batch_elapsed_ms: 10,
        });
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        Ok(())
    }
}

use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn driver_happy_path_emits_started_progress_jobdone() {
    let queue = DownloadQueue::new();
    let puller = FakePuller::default();
    let shutdown = CancellationToken::new();

    let mut rx = queue.subscribe();
    let driver_handle = spawn_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());

    // Enqueue with a manifest-known model so validation passes. Use a model
    // that's cheap to resolve — the real pull is replaced by the fake.
    let (id, _pos, _outcome) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    // Collect events for up to 2 seconds.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    let mut seen_started = false;
    let mut seen_progress = false;
    let mut seen_done = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Ok(DownloadEvent::Started { id: ev_id, .. })) if ev_id == id => {
                seen_started = true;
            }
            Ok(Ok(DownloadEvent::Progress { id: ev_id, .. })) if ev_id == id => {
                seen_progress = true;
            }
            Ok(Ok(DownloadEvent::JobDone { id: ev_id, .. })) if ev_id == id => {
                seen_done = true;
                break;
            }
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => {}
        }
    }

    shutdown.cancel();
    let _ = driver_handle.await;

    assert!(puller.called.load(Ordering::SeqCst), "puller was not invoked");
    assert!(seen_started, "missing Started event");
    assert!(seen_progress, "missing Progress event");
    assert!(seen_done, "missing JobDone event");

    // History should now hold one Completed entry.
    let listing = queue.listing().await;
    assert!(listing.active.is_none());
    assert_eq!(listing.history.len(), 1);
    assert_eq!(listing.history[0].status, JobStatus::Completed);
}
```

- [ ] **Step 2: Run the test — expect compile error**

```bash
cargo test -p mold-ai-server driver_happy_path --lib
```

Expected: `error[E0432]: unresolved import crate::downloads::{spawn_driver, PullDriver}`.

- [ ] **Step 3: Implement `PullDriver` + `spawn_driver`**

Append to `crates/mold-server/src/downloads.rs`:

```rust
// ── PullDriver trait + real & test implementations ──────────────────────────

/// Trait that hides the HuggingFace pull behind something the tests can fake.
///
/// The real implementation in `HfPullDriver` calls
/// `mold_core::download::pull_model_with_callback`. Tests inject a stub.
#[async_trait::async_trait]
pub trait PullDriver: Send + Sync + 'static {
    async fn pull(
        &self,
        model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String>;
}

/// Production driver — wraps the real HF pull.
pub struct HfPullDriver;

#[async_trait::async_trait]
impl PullDriver for HfPullDriver {
    async fn pull(
        &self,
        model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        let on_progress: mold_core::download::DownloadProgressCallback =
            std::sync::Arc::from(on_progress);
        let opts = mold_core::download::PullOptions::default();
        let model = model.to_string();
        // Race the pull against cancellation. pull_and_configure_with_callback
        // is not cancel-aware internally, but dropping its future on cancel
        // aborts the underlying tokio-based HTTP calls.
        tokio::select! {
            res = mold_core::download::pull_and_configure_with_callback(&model, on_progress, &opts) => {
                res.map(|_| ()).map_err(|e| e.to_string())
            }
            _ = cancel.cancelled() => {
                // Best-effort cleanup: the `.pulling` marker + partial clean-path
                // files may remain. `cleanup_partials_for_model` below wipes them.
                cleanup_partials_for_model(&model);
                Err("cancelled".into())
            }
        }
    }
}

/// Delete partial files under `<models_dir>/<sanitized>/` for a cancelled pull.
/// Leaves the HF cache under `~/.cache/huggingface` intact so retry is cheap.
fn cleanup_partials_for_model(model: &str) {
    use mold_core::manifest::resolve_model_name;
    let canonical = resolve_model_name(model);
    let sanitized = canonical.replace(':', "-");
    // Remove `.pulling` marker first so `has_pulling_marker` returns false.
    mold_core::download::remove_pulling_marker(&canonical);
    if let Ok(models_dir) = std::env::var("MOLD_MODELS_DIR") {
        let target = std::path::PathBuf::from(models_dir).join(&sanitized);
        let _ = std::fs::remove_dir_all(target);
        return;
    }
    if let Some(home) = dirs::home_dir() {
        let target = home.join(".mold/models").join(&sanitized);
        let _ = std::fs::remove_dir_all(target);
    }
}

/// Spawn the single driver task. Returns the JoinHandle; drop or await to
/// stop the driver (combined with `shutdown.cancel()`).
pub fn spawn_driver(
    queue: Arc<DownloadQueue>,
    driver: Arc<dyn PullDriver>,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!("download queue driver started");
        loop {
            queue.wait_for_work(&shutdown).await;
            if shutdown.is_cancelled() {
                break;
            }
            while let Some(mut job) = queue.take_next_queued() {
                if shutdown.is_cancelled() {
                    break;
                }
                run_one_job(&queue, &driver, &mut job).await;
                if shutdown.is_cancelled() {
                    break;
                }
            }
        }
        tracing::info!("download queue driver exiting");
    })
}

async fn run_one_job(
    queue: &Arc<DownloadQueue>,
    driver: &Arc<dyn PullDriver>,
    job: &mut DownloadJob,
) {
    job.status = JobStatus::Active;
    job.started_at = Some(now_ms());
    let cancel = CancellationToken::new();
    let id_for_handle = job.id.clone();
    let handle_job = job.clone();

    // Wrap the outer JoinHandle later; first install the job as active.
    let active_handle = ActiveHandle {
        job: handle_job,
        abort: cancel.clone(),
        task: tokio::spawn(async {}), // placeholder, replaced below
    };
    queue.set_active(active_handle).await;

    let _ = try_pull_with_retry(queue, driver, job, cancel.clone()).await;

    // Move the finished job into history.
    let final_job = queue
        .with_active(|a| a.clone())
        .await
        .unwrap_or_else(|| job.clone());
    queue.clear_active().await;
    queue.push_history(final_job);
    let _ = id_for_handle; // silence unused if we drop references
}

async fn try_pull_with_retry(
    queue: &Arc<DownloadQueue>,
    driver: &Arc<dyn PullDriver>,
    job: &mut DownloadJob,
    cancel: CancellationToken,
) -> Result<(), ()> {
    // Initial attempt + optional single retry.
    for attempt in 0..=1u8 {
        let result = run_single_attempt(queue, driver, job, cancel.clone()).await;
        match result {
            Ok(()) => {
                job.status = JobStatus::Completed;
                job.completed_at = Some(now_ms());
                // Reflect into the active-job snapshot so history includes the final state.
                queue
                    .with_active(|a| {
                        *a = job.clone();
                    })
                    .await;
                queue.emit(DownloadEvent::JobDone {
                    id: job.id.clone(),
                    model: job.model.clone(),
                });
                return Ok(());
            }
            Err(AttemptError::Cancelled) => {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(now_ms());
                queue
                    .with_active(|a| {
                        *a = job.clone();
                    })
                    .await;
                queue.emit(DownloadEvent::JobCancelled { id: job.id.clone() });
                return Err(());
            }
            Err(AttemptError::Failed(msg)) => {
                if attempt == 0 {
                    tracing::warn!(model = %job.model, "pull attempt 1 failed: {msg} — retrying in 5s");
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {},
                        _ = cancel.cancelled() => {
                            job.status = JobStatus::Cancelled;
                            job.completed_at = Some(now_ms());
                            queue.emit(DownloadEvent::JobCancelled { id: job.id.clone() });
                            return Err(());
                        }
                    }
                    continue;
                }
                job.status = JobStatus::Failed;
                job.error = Some(msg.clone());
                job.completed_at = Some(now_ms());
                queue
                    .with_active(|a| {
                        *a = job.clone();
                    })
                    .await;
                queue.emit(DownloadEvent::JobFailed {
                    id: job.id.clone(),
                    error: msg,
                });
                return Err(());
            }
        }
    }
    Err(())
}

enum AttemptError {
    Cancelled,
    Failed(String),
}

async fn run_single_attempt(
    queue: &Arc<DownloadQueue>,
    driver: &Arc<dyn PullDriver>,
    job: &mut DownloadJob,
    cancel: CancellationToken,
) -> Result<(), AttemptError> {
    // Closure that mutates active job and emits Progress/Started/FileDone events.
    let queue_for_cb = queue.clone();
    let job_id = job.id.clone();
    let on_progress = Box::new(move |evt: mold_core::download::DownloadProgressEvent| {
        translate_event(&queue_for_cb, &job_id, evt);
    });

    match driver.pull(&job.model, on_progress, cancel.clone()).await {
        Ok(()) => Ok(()),
        Err(msg) if cancel.is_cancelled() => {
            let _ = msg;
            Err(AttemptError::Cancelled)
        }
        Err(msg) => Err(AttemptError::Failed(msg)),
    }
}

fn translate_event(
    queue: &Arc<DownloadQueue>,
    job_id: &str,
    evt: mold_core::download::DownloadProgressEvent,
) {
    use mold_core::download::DownloadProgressEvent as P;
    let queue = queue.clone();
    let id = job_id.to_string();
    // Lock-free fan-out: do all work synchronously and rely on broadcast's
    // lock-free ring buffer. `with_active` uses an async mutex, so we
    // schedule a short task to update state.
    tokio::spawn(async move {
        match evt {
            P::FileStart {
                total_files,
                batch_bytes_total,
                filename,
                file_index,
                ..
            } => {
                let emitted_started = queue
                    .with_active(|j| {
                        if j.files_total == 0 {
                            j.files_total = total_files;
                            j.bytes_total = batch_bytes_total;
                            true
                        } else {
                            false
                        }
                    })
                    .await
                    .unwrap_or(false);
                if emitted_started {
                    queue.emit(DownloadEvent::Started {
                        id: id.clone(),
                        files_total: total_files,
                        bytes_total: batch_bytes_total,
                    });
                }
                let _ = file_index;
                queue
                    .with_active(|j| {
                        j.current_file = Some(filename.clone());
                    })
                    .await;
            }
            P::FileProgress {
                filename,
                bytes_downloaded,
                batch_bytes_downloaded,
                batch_bytes_total,
                file_index,
                bytes_total,
                ..
            } => {
                let _ = (bytes_downloaded, bytes_total, file_index);
                queue
                    .with_active(|j| {
                        j.bytes_done = batch_bytes_downloaded;
                        if j.bytes_total == 0 {
                            j.bytes_total = batch_bytes_total;
                        }
                        j.current_file = Some(filename.clone());
                    })
                    .await;
                queue.emit(DownloadEvent::Progress {
                    id: id.clone(),
                    files_done: 0, // populated on FileDone
                    bytes_done: batch_bytes_downloaded,
                    current_file: Some(filename),
                });
            }
            P::FileDone {
                filename,
                file_index,
                total_files,
                batch_bytes_downloaded,
                batch_bytes_total,
                ..
            } => {
                let _ = batch_bytes_total;
                queue
                    .with_active(|j| {
                        j.files_done = file_index + 1;
                        j.files_total = total_files;
                        j.bytes_done = batch_bytes_downloaded;
                    })
                    .await;
                queue.emit(DownloadEvent::FileDone {
                    id: id.clone(),
                    filename,
                });
            }
            P::Status { message } => {
                // Only surface as a transient info on the active job's current_file
                // placeholder. The drawer shows `current_file` literally.
                queue
                    .with_active(|j| {
                        j.current_file = Some(message.clone());
                    })
                    .await;
            }
        }
    });
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
```

Also add `async-trait` and `dirs` to `Cargo.toml` under `[dependencies]` if not present:

```toml
async-trait = "0.1"
dirs = "5"
```

- [ ] **Step 4: Run the happy-path test**

```bash
cargo test -p mold-ai-server driver_happy_path --lib
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/downloads.rs \
        crates/mold-server/src/downloads_test.rs \
        crates/mold-server/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(downloads): PullDriver trait + driver task with happy-path event translation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Cancellation (active and queued)

**Files:**
- Modify: `crates/mold-server/src/downloads_test.rs`
- Verify: `crates/mold-server/src/downloads.rs` (logic already drafted in Task 4; add tests)

Depends on: Task 4

- [ ] **Step 1: Write failing tests for both cancellation paths**

Append to `crates/mold-server/src/downloads_test.rs`:

```rust
/// Puller that waits for cancel, then returns Err("cancelled").
#[derive(Clone, Default)]
struct SlowPuller;

#[async_trait::async_trait]
impl PullDriver for SlowPuller {
    async fn pull(
        &self,
        _model: &str,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        cancel.cancelled().await;
        Err("cancelled".into())
    }
}

#[tokio::test]
async fn cancel_active_emits_job_cancelled_and_clears_active() {
    let queue = DownloadQueue::new();
    let shutdown = CancellationToken::new();
    let driver = spawn_driver(queue.clone(), Arc::new(SlowPuller), shutdown.clone());

    let mut rx = queue.subscribe();
    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    // Wait for the job to become active (driver picks it up).
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(queue.listing().await.active.is_some());

    assert!(queue.cancel(&id).await, "cancel should find the active job");

    // Consume events until JobCancelled fires or timeout.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    let mut seen = false;
    while tokio::time::Instant::now() < deadline {
        if let Ok(Ok(DownloadEvent::JobCancelled { id: ev_id })) =
            tokio::time::timeout(Duration::from_millis(200), rx.recv()).await
        {
            if ev_id == id {
                seen = true;
                break;
            }
        }
    }
    assert!(seen, "expected JobCancelled event");

    shutdown.cancel();
    let _ = driver.await;

    let listing = queue.listing().await;
    assert!(listing.active.is_none());
    assert_eq!(listing.history.last().unwrap().status, JobStatus::Cancelled);
}

#[tokio::test]
async fn cancel_queued_removes_from_queue_and_emits_dequeued() {
    let queue = DownloadQueue::new();
    // No driver needed — we never let these jobs run.
    let (id_a, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    let (id_b, _, _) = queue.enqueue("flux-dev:q4".into()).await.unwrap();

    // Before cancel, queued.len() == 2 (no driver, none become active).
    assert_eq!(queue.listing().await.queued.len(), 2);

    let mut rx = queue.subscribe();
    assert!(queue.cancel(&id_b).await);

    let got = tokio::time::timeout(Duration::from_millis(200), rx.recv())
        .await
        .expect("timed out waiting for Dequeued")
        .expect("broadcast channel closed");
    match got {
        DownloadEvent::Dequeued { id } => assert_eq!(id, id_b),
        other => panic!("expected Dequeued, got {other:?}"),
    }

    let listing = queue.listing().await;
    assert_eq!(listing.queued.len(), 1);
    assert_eq!(listing.queued[0].id, id_a);
}

#[tokio::test]
async fn cancel_unknown_id_returns_false() {
    let queue = DownloadQueue::new();
    assert!(!queue.cancel("no-such-id").await);
}
```

- [ ] **Step 2: Run the tests**

```bash
cargo test -p mold-ai-server -- cancel_active_emits cancel_queued_removes cancel_unknown_id --lib
```

Expected: all three pass (the logic was implemented in Task 4).

If `cancel_active_emits_job_cancelled_and_clears_active` fails because the driver never gets scheduled before `sleep(100ms)`, increase the sleep to 250ms.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-server/src/downloads_test.rs
git commit -m "$(cat <<'EOF'
test(downloads): cancellation tests for active + queued paths

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Auto-retry on failure

**Files:**
- Modify: `crates/mold-server/src/downloads_test.rs`

Depends on: Task 4

- [ ] **Step 1: Write failing retry test**

Append to `crates/mold-server/src/downloads_test.rs`:

```rust
/// Fails once, then succeeds.
#[derive(Clone, Default)]
struct FlakyPuller {
    pub calls: Arc<AtomicUsize>,
}

use std::sync::atomic::AtomicUsize;

#[async_trait::async_trait]
impl PullDriver for FlakyPuller {
    async fn pull(
        &self,
        _model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        // Emit a minimal FileStart so Started fires.
        on_progress(mold_core::download::DownloadProgressEvent::FileStart {
            filename: "f".into(),
            file_index: 0,
            total_files: 1,
            size_bytes: 10,
            batch_bytes_downloaded: 0,
            batch_bytes_total: 10,
            batch_elapsed_ms: 0,
        });
        if n == 1 {
            Err("simulated network blip".into())
        } else {
            on_progress(mold_core::download::DownloadProgressEvent::FileDone {
                filename: "f".into(),
                file_index: 0,
                total_files: 1,
                batch_bytes_downloaded: 10,
                batch_bytes_total: 10,
                batch_elapsed_ms: 1,
            });
            Ok(())
        }
    }
}

#[tokio::test]
async fn driver_retries_once_then_succeeds() {
    // Shorten the retry backoff via an env override for tests. If you'd rather
    // not add an env, just wait 6 seconds — CI is fine with that, but this is
    // one place to use `tokio::time::pause` if the 5 s is painful.
    tokio::time::pause();

    let queue = DownloadQueue::new();
    let puller = FlakyPuller::default();
    let shutdown = CancellationToken::new();
    let handle = spawn_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());
    let mut rx = queue.subscribe();

    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    // Advance past the 5 s backoff quickly.
    tokio::time::advance(std::time::Duration::from_secs(6)).await;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    let mut seen_done = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Ok(DownloadEvent::JobDone { id: ev, .. })) if ev == id => {
                seen_done = true;
                break;
            }
            Ok(Ok(DownloadEvent::JobFailed { .. })) => {
                panic!("expected JobDone after retry, got JobFailed");
            }
            _ => {}
        }
    }

    shutdown.cancel();
    let _ = handle.await;

    assert!(seen_done, "expected JobDone after retry");
    assert_eq!(puller.calls.load(Ordering::SeqCst), 2, "expected exactly 2 attempts");
}
```

- [ ] **Step 2: Run it**

```bash
cargo test -p mold-ai-server driver_retries_once_then_succeeds --lib
```

Expected: passes on the first run (retry logic already implemented in Task 4). If the test times out because of `tokio::time::pause`'s interaction with `tokio::spawn`'d translator, remove `pause` and just let the real 5 s backoff run.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-server/src/downloads_test.rs
git commit -m "$(cat <<'EOF'
test(downloads): auto-retry path (fails once, succeeds on retry)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Boot the driver from `run_server`

**Files:**
- Modify: `crates/mold-server/src/lib.rs`

Depends on: Task 4

- [ ] **Step 1: Locate the server-startup function**

Open `crates/mold-server/src/lib.rs`. Find the `run_server` (or equivalent startup) function — the one that constructs `AppState` and binds axum. In the spot just after `AppState` is built and before `axum::serve` runs, add an additive block:

```rust
// ── Downloads UI (Agent A) ──────────────────────────────────────────────────
let downloads_shutdown = tokio_util::sync::CancellationToken::new();
let _downloads_driver = crate::downloads::spawn_driver(
    state.downloads.clone(),
    std::sync::Arc::new(crate::downloads::HfPullDriver),
    downloads_shutdown.clone(),
);
// The driver handle lives for the process lifetime; on graceful shutdown we
// cancel the token and let it drop on its own.
```

If the existing startup already has a shutdown signal, chain `downloads_shutdown.cancel()` into the same exit path. Otherwise leaving it cancelled-on-drop is fine.

- [ ] **Step 2: Verify workspace compiles**

```bash
cargo check --workspace
```

Expected: success.

- [ ] **Step 3: Commit**

```bash
git add crates/mold-server/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(downloads): boot the queue driver on server startup

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `POST /api/downloads` route

**Files:**
- Modify: `crates/mold-server/src/routes.rs`
- Modify: `crates/mold-server/src/routes_test.rs`

Depends on: Task 3

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/routes_test.rs`:

```rust
#[tokio::test]
async fn post_api_downloads_enqueues_job() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state.clone());

    let body = serde_json::json!({ "model": "flux-schnell:q4" });
    let req = Request::builder()
        .method("POST")
        .uri("/api/downloads")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("id").and_then(|x| x.as_str()).is_some());
    assert!(v.get("position").and_then(|x| x.as_u64()).is_some());

    let listing = state.downloads.listing().await;
    // No driver running in this test, so the job sits in `queued`.
    assert_eq!(listing.queued.len(), 1);
    assert_eq!(listing.queued[0].model, "flux-schnell:q4");
}

#[tokio::test]
async fn post_api_downloads_unknown_model_400() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state);
    let body = serde_json::json!({ "model": "not-a-real-model:xyz" });
    let req = Request::builder()
        .method("POST")
        .uri("/api/downloads")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_api_downloads_duplicate_is_idempotent_409() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state.clone());

    let body = serde_json::json!({ "model": "flux-schnell:q4" });
    let make_req = || {
        Request::builder()
            .method("POST")
            .uri("/api/downloads")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };

    let res1 = app.clone().oneshot(make_req()).await.unwrap();
    assert_eq!(res1.status(), StatusCode::OK);
    let bytes1 = axum::body::to_bytes(res1.into_body(), 64 * 1024).await.unwrap();
    let v1: serde_json::Value = serde_json::from_slice(&bytes1).unwrap();
    let id1 = v1["id"].as_str().unwrap().to_string();

    let res2 = app.oneshot(make_req()).await.unwrap();
    assert_eq!(res2.status(), StatusCode::CONFLICT);
    let bytes2 = axum::body::to_bytes(res2.into_body(), 64 * 1024).await.unwrap();
    let v2: serde_json::Value = serde_json::from_slice(&bytes2).unwrap();
    let id2 = v2["id"].as_str().unwrap().to_string();

    assert_eq!(id1, id2, "duplicate enqueue must return the same id");
}
```

If `AppState::empty_gpu_pool_for_test` doesn't exist yet, expose `AppState::empty_gpu_pool()` by removing the `#[cfg(test)]` guard temporarily with a `pub(crate)` visibility OR add a sibling helper `pub(crate) fn empty_gpu_pool_for_test()` that just returns `empty_gpu_pool()`. Also verify `crate::router::build` exists — it may be named `crate::build_router` or `crate::app::build`. Substitute whatever name is used in `lib.rs`.

- [ ] **Step 2: Run the failing tests**

```bash
cargo test -p mold-ai-server post_api_downloads --lib
```

Expected: `error[E0432]` or `404 Not Found` because the route does not exist.

- [ ] **Step 3: Implement the route**

Modify `crates/mold-server/src/routes.rs`. Add under a clearly-labelled section (search for an existing section header and append):

```rust
// ─── Downloads UI (Agent A) ──────────────────────────────────────────────────

#[derive(serde::Deserialize, utoipa::ToSchema)]
pub struct CreateDownloadBody {
    pub model: String,
}

#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct CreateDownloadResponse {
    pub id: String,
    pub position: usize,
}

#[utoipa::path(
    post,
    path = "/api/downloads",
    tag = "downloads",
    request_body = CreateDownloadBody,
    responses(
        (status = 200, description = "Enqueued; position 0 = will start immediately", body = CreateDownloadResponse),
        (status = 400, description = "Unknown model"),
        (status = 409, description = "Already active or queued; body contains existing id", body = CreateDownloadResponse),
    )
)]
pub async fn create_download(
    State(state): State<AppState>,
    Json(body): Json<CreateDownloadBody>,
) -> impl IntoResponse {
    use crate::downloads::{EnqueueError, EnqueueOutcome};
    match state.downloads.enqueue(body.model.clone()).await {
        Ok((id, position, EnqueueOutcome::Created)) => (
            StatusCode::OK,
            Json(CreateDownloadResponse { id, position }),
        )
            .into_response(),
        Ok((id, position, EnqueueOutcome::AlreadyPresent)) => (
            StatusCode::CONFLICT,
            Json(CreateDownloadResponse { id, position }),
        )
            .into_response(),
        Err(EnqueueError::UnknownModel(_)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("unknown model '{}'. Run 'mold list' to see available models.", body.model)
            })),
        )
            .into_response(),
        Err(EnqueueError::LockPoisoned) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "download queue state is corrupt" })),
        )
            .into_response(),
    }
}
```

Wire the route into the router. Locate the axum router builder (grep for `.route("/api/models"`) and append under the downloads section:

```rust
        .route("/api/downloads", post(create_download))
```

If a new `utoipa` path collection is used, add `CreateDownloadBody` and `CreateDownloadResponse` to the `components(schemas(...))` list.

- [ ] **Step 4: Run tests again**

```bash
cargo test -p mold-ai-server post_api_downloads --lib
```

Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "$(cat <<'EOF'
feat(downloads): POST /api/downloads (enqueue + idempotent 409 + 400 unknown)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `DELETE /api/downloads/:id` route

**Files:**
- Modify: `crates/mold-server/src/routes.rs`
- Modify: `crates/mold-server/src/routes_test.rs`

Depends on: Task 8

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/routes_test.rs`:

```rust
#[tokio::test]
async fn delete_api_downloads_204_for_queued() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state.clone());

    let (id, _, _) = state
        .downloads
        .enqueue("flux-schnell:q4".into())
        .await
        .unwrap();

    let req = Request::builder()
        .method("DELETE")
        .uri(format!("/api/downloads/{id}"))
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let listing = state.downloads.listing().await;
    assert!(listing.queued.is_empty());
}

#[tokio::test]
async fn delete_api_downloads_404_when_unknown() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state);
    let req = Request::builder()
        .method("DELETE")
        .uri("/api/downloads/nonexistent-id")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
```

- [ ] **Step 2: Run it — expect 404 for both (route missing)**

```bash
cargo test -p mold-ai-server delete_api_downloads --lib
```

- [ ] **Step 3: Implement**

Append to `routes.rs` under the downloads section:

```rust
#[utoipa::path(
    delete,
    path = "/api/downloads/{id}",
    tag = "downloads",
    params(("id" = String, Path, description = "Job id")),
    responses(
        (status = 204, description = "Cancelled"),
        (status = 404, description = "Unknown id"),
    )
)]
pub async fn delete_download(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if state.downloads.cancel(&id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown download id '{id}'") })),
        )
            .into_response()
    }
}
```

Router wire-up:

```rust
        .route("/api/downloads/:id", delete(delete_download))
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p mold-ai-server delete_api_downloads --lib
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "$(cat <<'EOF'
feat(downloads): DELETE /api/downloads/:id

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `GET /api/downloads` listing route

**Files:**
- Modify: `crates/mold-server/src/routes.rs`
- Modify: `crates/mold-server/src/routes_test.rs`

Depends on: Task 8

- [ ] **Step 1: Write failing test**

Append to `routes_test.rs`:

```rust
#[tokio::test]
async fn get_api_downloads_returns_listing_shape() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state.clone());

    let _ = state.downloads.enqueue("flux-schnell:q4".into()).await.unwrap();

    let req = Request::builder()
        .uri("/api/downloads")
        .body(Body::empty())
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["queued"].is_array());
    assert!(v["history"].is_array());
    assert_eq!(v["queued"].as_array().unwrap().len(), 1);
    assert_eq!(v["queued"][0]["model"], "flux-schnell:q4");
}
```

- [ ] **Step 2: Run it**

```bash
cargo test -p mold-ai-server get_api_downloads_returns_listing_shape --lib
```

Expected: fail (404).

- [ ] **Step 3: Implement**

Add to `routes.rs`:

```rust
#[utoipa::path(
    get,
    path = "/api/downloads",
    tag = "downloads",
    responses((status = 200, description = "Current queue state", body = mold_core::types::DownloadsListing))
)]
pub async fn list_downloads(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.downloads.listing().await)
}
```

Router:

```rust
        .route("/api/downloads", get(list_downloads).post(create_download))
```

(Merge the get+post onto the same route.)

- [ ] **Step 4: Run it**

```bash
cargo test -p mold-ai-server get_api_downloads_returns_listing_shape --lib
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "$(cat <<'EOF'
feat(downloads): GET /api/downloads listing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `GET /api/downloads/stream` SSE route

**Files:**
- Modify: `crates/mold-server/src/routes.rs`

Depends on: Task 10

- [ ] **Step 1: Implement the SSE handler**

Append to `routes.rs`:

```rust
#[utoipa::path(
    get,
    path = "/api/downloads/stream",
    tag = "downloads",
    responses((status = 200, description = "SSE stream of DownloadEvent JSON")),
)]
pub async fn stream_downloads(
    State(state): State<AppState>,
) -> Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    use axum::response::sse::{Event, KeepAlive};
    use tokio_stream::wrappers::BroadcastStream;
    use tokio_stream::StreamExt;

    let rx = state.downloads.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| match res {
        Ok(event) => {
            let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
            Some(Ok(Event::default().event("download").data(data)))
        }
        Err(_lagged) => None, // client will re-fetch /api/downloads on reconnect
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}
```

Router:

```rust
        .route("/api/downloads/stream", get(stream_downloads))
```

Imports (merge with existing):

```rust
use axum::response::sse::Sse;
```

- [ ] **Step 2: Write failing smoke test**

Append to `routes_test.rs`:

```rust
#[tokio::test]
async fn sse_stream_emits_enqueued_event() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use futures::StreamExt;
    use tower::ServiceExt;

    let state = crate::state::AppState::empty(
        mold_core::Config::default(),
        crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
        crate::state::AppState::empty_gpu_pool_for_test(),
        200,
    );
    let app = crate::router::build(state.clone());

    let req = Request::builder()
        .uri("/api/downloads/stream")
        .body(Body::empty())
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Enqueue AFTER subscribing (SSE response already established).
    let state_for_send = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let _ = state_for_send.downloads.enqueue("flux-schnell:q4".into()).await;
    });

    let mut body = res.into_body().into_data_stream();
    let mut saw_enqueued = false;
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(std::time::Duration::from_millis(300), body.next()).await {
            Ok(Some(Ok(bytes))) => {
                let text = String::from_utf8_lossy(&bytes).to_string();
                if text.contains("\"type\":\"enqueued\"") {
                    saw_enqueued = true;
                    break;
                }
            }
            _ => continue,
        }
    }
    assert!(saw_enqueued, "did not observe an 'enqueued' SSE event");
}
```

- [ ] **Step 3: Run it**

```bash
cargo test -p mold-ai-server sse_stream_emits_enqueued_event --lib
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "$(cat <<'EOF'
feat(downloads): GET /api/downloads/stream SSE multiplexer

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Compat shim — rewire `POST /api/models/pull` through the queue

**Files:**
- Modify: `crates/mold-server/src/routes.rs`

Depends on: Task 11

- [ ] **Step 1: Refactor `pull_model_endpoint` to delegate**

Replace the body of `pull_model_endpoint` (keep the function signature and the two `PullResponse::{Sse,Text}` branches). The SSE path now subscribes to the queue's broadcast channel and filters to this job id; the blocking path enqueues, awaits `JobDone`/`JobFailed`, and returns the plain-text result.

Replacement implementation:

```rust
async fn pull_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, ApiError> {
    let wants_sse = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));

    // Enqueue via the queue. Treat idempotent AlreadyPresent as success.
    let (job_id, _position) = match state.downloads.enqueue(body.model.clone()).await {
        Ok((id, pos, _)) => (id, pos),
        Err(crate::downloads::EnqueueError::UnknownModel(_)) => {
            return Err(ApiError::unknown_model(format!(
                "unknown model '{}'. Run 'mold list' to see available models.",
                body.model
            )));
        }
        Err(crate::downloads::EnqueueError::LockPoisoned) => {
            return Err(ApiError::internal("download queue state is corrupt".into()));
        }
    };

    if !wants_sse {
        // Await terminal event for this job, return plain text.
        let mut rx = state.downloads.subscribe();
        loop {
            match rx.recv().await {
                Ok(mold_core::types::DownloadEvent::JobDone { id, model })
                    if id == job_id =>
                {
                    return Ok(PullResponse::Text(format!(
                        "model '{model}' pulled successfully"
                    )));
                }
                Ok(mold_core::types::DownloadEvent::JobFailed { id, error })
                    if id == job_id =>
                {
                    return Err(ApiError::internal(format!(
                        "failed to pull model '{}': {error}",
                        body.model
                    )));
                }
                Ok(mold_core::types::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "pull of '{}' was cancelled",
                        body.model
                    )));
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err(ApiError::internal("download queue channel closed".into()));
                }
            }
        }
    }

    // SSE: re-emit queue events shaped like the legacy SseProgressEvent::DownloadProgress
    // so the TUI's existing consumer continues to work unchanged.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();
    let mut events = state.downloads.subscribe();
    let model_for_cb = body.model.clone();
    tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(mold_core::types::DownloadEvent::Started {
                    id, files_total, bytes_total,
                }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::DownloadProgress {
                        filename: String::new(),
                        file_index: 0,
                        total_files: files_total,
                        bytes_downloaded: 0,
                        bytes_total,
                        batch_bytes_downloaded: 0,
                        batch_bytes_total: bytes_total,
                        batch_elapsed_ms: 0,
                    }));
                }
                Ok(mold_core::types::DownloadEvent::Progress {
                    id, files_done, bytes_done, current_file,
                }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::DownloadProgress {
                        filename: current_file.unwrap_or_default(),
                        file_index: files_done,
                        total_files: 0,
                        bytes_downloaded: bytes_done,
                        bytes_total: 0,
                        batch_bytes_downloaded: bytes_done,
                        batch_bytes_total: 0,
                        batch_elapsed_ms: 0,
                    }));
                }
                Ok(mold_core::types::DownloadEvent::JobDone { id, .. }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                        model: model_for_cb.clone(),
                    }));
                    break;
                }
                Ok(mold_core::types::DownloadEvent::JobFailed { id, error })
                    if id == job_id =>
                {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent { message: error }));
                    break;
                }
                Ok(mold_core::types::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent {
                        message: "pull cancelled".into(),
                    }));
                    break;
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, std::convert::Infallible>(sse_message_to_event(msg)));

    Ok(PullResponse::Sse(
        Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15))
                    .text("ping"),
            )
            .into_response(),
    ))
}
```

- [ ] **Step 2: Verify compilation**

```bash
cargo check -p mold-ai-server
```

Expected: success.

- [ ] **Step 3: Run existing tests**

```bash
cargo test -p mold-ai-server --lib
```

Expected: all pass (the TUI wire shape is preserved).

- [ ] **Step 4: Commit**

```bash
git add crates/mold-server/src/routes.rs
git commit -m "$(cat <<'EOF'
feat(downloads): delegate POST /api/models/pull through the queue (compat shim)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Web — add shared types to `web/src/types.ts`

**Files:**
- Modify: `web/src/types.ts`

Depends on: Task 2

- [ ] **Step 1: Write failing type-sanity test**

Create `web/src/composables/useDownloads.test.ts` now so later tasks can append. Bare-minimum first block:

```ts
import { describe, it, expect } from "vitest";
import type {
  DownloadJobWire,
  JobStatusWire,
  DownloadEventWire,
} from "../types";

describe("downloads types", () => {
  it("job status values match server enum", () => {
    const all: JobStatusWire[] = [
      "queued",
      "active",
      "completed",
      "failed",
      "cancelled",
    ];
    expect(all).toHaveLength(5);
  });

  it("download job shape", () => {
    const job: DownloadJobWire = {
      id: "abc",
      model: "flux-dev:q4",
      status: "active",
      files_done: 0,
      files_total: 5,
      bytes_done: 0,
      bytes_total: 10_000,
      current_file: null,
      started_at: null,
      completed_at: null,
      error: null,
    };
    expect(job.status).toBe("active");
  });

  it("event discriminator shapes", () => {
    const e: DownloadEventWire = {
      type: "enqueued",
      id: "x",
      model: "flux-dev:q4",
      position: 1,
    };
    expect(e.type).toBe("enqueued");
  });
});
```

- [ ] **Step 2: Run it**

```bash
cd web && bun run test useDownloads
```

Expected: fails — types don't exist.

- [ ] **Step 3: Append types to `web/src/types.ts`**

Append (clearly fenced, at the bottom):

```ts
// ─── Downloads UI (Agent A) ───────────────────────────────────────────────────
// Mirror of `mold_core::types::{DownloadJob, JobStatus, DownloadEvent,
// DownloadsListing}`. Keep field names / string literals in sync with the
// server's serde output.

export type JobStatusWire =
  | "queued"
  | "active"
  | "completed"
  | "failed"
  | "cancelled";

export interface DownloadJobWire {
  id: string;
  model: string;
  status: JobStatusWire;
  files_done: number;
  files_total: number;
  bytes_done: number;
  bytes_total: number;
  current_file?: string | null;
  started_at?: number | null;
  completed_at?: number | null;
  error?: string | null;
}

export interface DownloadsListingWire {
  active?: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
}

export type DownloadEventWire =
  | { type: "enqueued"; id: string; model: string; position: number }
  | { type: "dequeued"; id: string }
  | {
      type: "started";
      id: string;
      files_total: number;
      bytes_total: number;
    }
  | {
      type: "progress";
      id: string;
      files_done: number;
      bytes_done: number;
      current_file?: string | null;
    }
  | { type: "file_done"; id: string; filename: string }
  | { type: "job_done"; id: string; model: string }
  | { type: "job_failed"; id: string; error: string }
  | { type: "job_cancelled"; id: string };
```

- [ ] **Step 4: Re-run**

```bash
cd web && bun run test useDownloads
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/composables/useDownloads.test.ts
git commit -m "$(cat <<'EOF'
feat(web): add DownloadJobWire / DownloadEventWire types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Web — API helpers (`web/src/api.ts`)

**Files:**
- Modify: `web/src/api.ts`

Depends on: Task 13

- [ ] **Step 1: Append helpers**

Open `web/src/api.ts`. Append:

```ts
// ─── Downloads UI (Agent A) ───────────────────────────────────────────────────
import type { DownloadJobWire, DownloadsListingWire } from "./types";

export interface CreateDownloadResponse {
  id: string;
  position: number;
}

export async function postDownload(
  model: string,
  signal?: AbortSignal,
): Promise<{ status: "created" | "duplicate"; id: string; position: number }> {
  const res = await fetch(`${base}/api/downloads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
    signal,
  });
  if (res.status === 400) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `unknown model '${model}'`);
  }
  if (!res.ok && res.status !== 409) {
    throw new Error(`POST /api/downloads failed: ${res.status}`);
  }
  const json = (await res.json()) as CreateDownloadResponse;
  return {
    status: res.status === 409 ? "duplicate" : "created",
    id: json.id,
    position: json.position,
  };
}

export async function cancelDownload(id: string): Promise<void> {
  const res = await fetch(
    `${base}/api/downloads/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (res.status === 404) {
    // Idempotent — treat as already gone.
    return;
  }
  if (!res.ok && res.status !== 204) {
    throw new Error(`DELETE /api/downloads failed: ${res.status}`);
  }
}

export async function fetchDownloads(
  signal?: AbortSignal,
): Promise<DownloadsListingWire> {
  const res = await fetch(`${base}/api/downloads`, { signal });
  if (!res.ok) throw new Error(`GET /api/downloads failed: ${res.status}`);
  const raw = (await res.json()) as DownloadsListingWire;
  // Server may omit `active`/`history` as null; normalise.
  return {
    active: raw.active ?? null,
    queued: raw.queued ?? [],
    history: raw.history ?? [],
  };
}

/** Returns the absolute URL for the SSE stream (consumed via EventSource). */
export function downloadsStreamUrl(): string {
  return `${base}/api/downloads/stream`;
}

export type { DownloadJobWire, DownloadsListingWire };
```

- [ ] **Step 2: Verify no regressions**

```bash
cd web && bun run fmt:check && bun run verify && bun run test
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add web/src/api.ts
git commit -m "$(cat <<'EOF'
feat(web): postDownload / cancelDownload / fetchDownloads helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Web — `useDownloads` composable

**Files:**
- Create: `web/src/composables/useDownloads.ts`
- Modify: `web/src/composables/useDownloads.test.ts`

Depends on: Task 14

- [ ] **Step 1: Write failing ETA + reducer tests**

Append to `web/src/composables/useDownloads.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  applyDownloadEvent,
  computeEtaSeconds,
  newDownloadsState,
} from "./useDownloads";
import type { DownloadEventWire } from "../types";

describe("computeEtaSeconds", () => {
  it("returns null when fewer than 2 samples", () => {
    expect(computeEtaSeconds([], 1000)).toBeNull();
    expect(computeEtaSeconds([{ ts: 0, bytes: 0 }], 1000)).toBeNull();
  });

  it("computes eta from sliding window", () => {
    const history = [
      { ts: 0, bytes: 0 },
      { ts: 2000, bytes: 1_000_000 },
    ];
    // rate = 500_000 B/s; remaining = 1_000_000; eta = 2 s
    expect(computeEtaSeconds(history, 2_000_000)).toBe(2);
  });

  it("returns null when rate is non-positive", () => {
    const history = [
      { ts: 0, bytes: 1_000_000 },
      { ts: 1000, bytes: 1_000_000 },
    ];
    expect(computeEtaSeconds(history, 2_000_000)).toBeNull();
  });
});

describe("applyDownloadEvent", () => {
  it("Enqueued appends a queued job", () => {
    const state = newDownloadsState();
    const evt: DownloadEventWire = {
      type: "enqueued",
      id: "a",
      model: "flux-dev:q4",
      position: 1,
    };
    applyDownloadEvent(state, evt);
    expect(state.queued).toHaveLength(1);
    expect(state.queued[0].id).toBe("a");
    expect(state.queued[0].status).toBe("queued");
  });

  it("Started promotes queued→active", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "flux-dev:q4",
      position: 1,
    });
    applyDownloadEvent(state, {
      type: "started",
      id: "a",
      files_total: 3,
      bytes_total: 9_000,
    });
    expect(state.queued).toHaveLength(0);
    expect(state.active?.id).toBe("a");
    expect(state.active?.files_total).toBe(3);
    expect(state.active?.bytes_total).toBe(9_000);
    expect(state.active?.status).toBe("active");
  });

  it("Progress updates bytes_done", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "m",
      position: 1,
    });
    applyDownloadEvent(state, {
      type: "started",
      id: "a",
      files_total: 1,
      bytes_total: 1000,
    });
    applyDownloadEvent(state, {
      type: "progress",
      id: "a",
      files_done: 0,
      bytes_done: 500,
      current_file: "foo.bin",
    });
    expect(state.active?.bytes_done).toBe(500);
    expect(state.active?.current_file).toBe("foo.bin");
  });

  it("JobDone moves active into history, capped at 20", () => {
    const state = newDownloadsState();
    for (let i = 0; i < 25; i++) {
      applyDownloadEvent(state, {
        type: "enqueued",
        id: `id-${i}`,
        model: "m",
        position: 1,
      });
      applyDownloadEvent(state, {
        type: "started",
        id: `id-${i}`,
        files_total: 1,
        bytes_total: 1,
      });
      applyDownloadEvent(state, {
        type: "job_done",
        id: `id-${i}`,
        model: "m",
      });
    }
    expect(state.active).toBeNull();
    expect(state.history.length).toBe(20);
    // Newest first or newest last? We assert newest last (matches server).
    expect(state.history.at(-1)?.id).toBe("id-24");
  });

  it("JobCancelled from queued emits via dequeued event", () => {
    const state = newDownloadsState();
    applyDownloadEvent(state, {
      type: "enqueued",
      id: "a",
      model: "m",
      position: 1,
    });
    applyDownloadEvent(state, { type: "dequeued", id: "a" });
    expect(state.queued).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run — expect failure (module missing)**

```bash
cd web && bun run test useDownloads
```

- [ ] **Step 3: Implement the composable**

Create `web/src/composables/useDownloads.ts`:

```ts
import { ref, type Ref } from "vue";
import {
  cancelDownload,
  downloadsStreamUrl,
  fetchDownloads,
  postDownload,
} from "../api";
import type {
  DownloadEventWire,
  DownloadJobWire,
  DownloadsListingWire,
} from "../types";

export interface DownloadsState {
  active: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
}

export function newDownloadsState(): DownloadsState {
  return { active: null, queued: [], history: [] };
}

const HISTORY_CAP = 20;

/**
 * Pure reducer — applied to a plain `DownloadsState` so it can be unit-tested
 * without a Vue runtime.
 */
export function applyDownloadEvent(
  state: DownloadsState,
  event: DownloadEventWire,
): void {
  switch (event.type) {
    case "enqueued": {
      state.queued.push({
        id: event.id,
        model: event.model,
        status: "queued",
        files_done: 0,
        files_total: 0,
        bytes_done: 0,
        bytes_total: 0,
        current_file: null,
        started_at: null,
        completed_at: null,
        error: null,
      });
      return;
    }
    case "dequeued": {
      const idx = state.queued.findIndex((j) => j.id === event.id);
      if (idx >= 0) state.queued.splice(idx, 1);
      return;
    }
    case "started": {
      const idx = state.queued.findIndex((j) => j.id === event.id);
      const from =
        idx >= 0
          ? state.queued.splice(idx, 1)[0]
          : {
              id: event.id,
              model: "",
              status: "queued" as const,
              files_done: 0,
              files_total: 0,
              bytes_done: 0,
              bytes_total: 0,
              current_file: null,
              started_at: null,
              completed_at: null,
              error: null,
            };
      state.active = {
        ...from,
        status: "active",
        files_total: event.files_total,
        bytes_total: event.bytes_total,
        started_at: Date.now(),
      };
      return;
    }
    case "progress": {
      if (state.active?.id !== event.id) return;
      state.active.files_done = event.files_done;
      state.active.bytes_done = event.bytes_done;
      state.active.current_file = event.current_file ?? null;
      return;
    }
    case "file_done": {
      if (state.active?.id !== event.id) return;
      state.active.files_done += 1;
      return;
    }
    case "job_done": {
      const active = state.active;
      if (!active || active.id !== event.id) return;
      const completed: DownloadJobWire = {
        ...active,
        status: "completed",
        completed_at: Date.now(),
      };
      state.active = null;
      state.history.push(completed);
      while (state.history.length > HISTORY_CAP) state.history.shift();
      return;
    }
    case "job_failed": {
      const active = state.active;
      if (!active || active.id !== event.id) return;
      const failed: DownloadJobWire = {
        ...active,
        status: "failed",
        error: event.error,
        completed_at: Date.now(),
      };
      state.active = null;
      state.history.push(failed);
      while (state.history.length > HISTORY_CAP) state.history.shift();
      return;
    }
    case "job_cancelled": {
      const active = state.active;
      if (!active || active.id !== event.id) return;
      const cancelled: DownloadJobWire = {
        ...active,
        status: "cancelled",
        completed_at: Date.now(),
      };
      state.active = null;
      state.history.push(cancelled);
      while (state.history.length > HISTORY_CAP) state.history.shift();
      return;
    }
  }
}

/**
 * Client-side ETA math — server only emits raw counters.
 * history = sliding window of {ts, bytes} samples (last ~10 s).
 */
export function computeEtaSeconds(
  history: Array<{ ts: number; bytes: number }>,
  bytesTotal: number,
): number | null {
  if (history.length < 2) return null;
  const first = history[0];
  const last = history[history.length - 1];
  const deltaBytes = last.bytes - first.bytes;
  const deltaMs = last.ts - first.ts;
  if (deltaMs <= 0 || deltaBytes <= 0) return null;
  const ratePerSec = (deltaBytes * 1000) / deltaMs;
  const remaining = Math.max(0, bytesTotal - last.bytes);
  const eta = remaining / ratePerSec;
  return Number.isFinite(eta) ? Math.round(eta) : null;
}

// ── Vue runtime singleton ────────────────────────────────────────────────────

export interface UseDownloads {
  active: Ref<DownloadJobWire | null>;
  queued: Ref<DownloadJobWire[]>;
  history: Ref<DownloadJobWire[]>;
  ratesByJob: Ref<Record<string, Array<{ ts: number; bytes: number }>>>;
  enqueue: (model: string) => Promise<void>;
  cancel: (id: string) => Promise<void>;
  connected: Ref<boolean>;
  close: () => void;
}

type Listener = () => void;
const completionListeners = new Set<Listener>();

export function onDownloadComplete(cb: Listener): () => void {
  completionListeners.add(cb);
  return () => completionListeners.delete(cb);
}

let singleton: UseDownloads | null = null;

export function useDownloads(): UseDownloads {
  if (singleton) return singleton;
  singleton = buildSingleton();
  return singleton;
}

function buildSingleton(): UseDownloads {
  const active = ref<DownloadJobWire | null>(null);
  const queued = ref<DownloadJobWire[]>([]);
  const history = ref<DownloadJobWire[]>([]);
  const ratesByJob = ref<Record<string, Array<{ ts: number; bytes: number }>>>(
    {},
  );
  const connected = ref(false);
  let es: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;

  function state(): DownloadsState {
    return {
      active: active.value,
      queued: queued.value,
      history: history.value,
    };
  }

  function writeBack(next: DownloadsState) {
    active.value = next.active;
    queued.value = [...next.queued];
    history.value = [...next.history];
  }

  function applyListing(listing: DownloadsListingWire) {
    active.value = listing.active ?? null;
    queued.value = [...listing.queued];
    history.value = [...listing.history];
  }

  function onEvent(raw: string) {
    let evt: DownloadEventWire;
    try {
      evt = JSON.parse(raw) as DownloadEventWire;
    } catch {
      return;
    }
    const snap = state();
    applyDownloadEvent(snap, evt);
    writeBack(snap);

    // Maintain rate sample window for the active job.
    if (evt.type === "progress" && active.value && active.value.id === evt.id) {
      const id = evt.id;
      const samples = ratesByJob.value[id] ?? [];
      const now = Date.now();
      samples.push({ ts: now, bytes: evt.bytes_done });
      // Drop samples older than 10 s.
      while (samples.length > 0 && now - samples[0].ts > 10_000) samples.shift();
      ratesByJob.value = { ...ratesByJob.value, [id]: samples };
    }

    if (evt.type === "job_done") {
      for (const cb of completionListeners) cb();
    }
  }

  function connect() {
    if (closed) return;
    try {
      es = new EventSource(downloadsStreamUrl());
    } catch (err) {
      scheduleReconnect();
      return;
    }
    es.onopen = () => {
      connected.value = true;
    };
    es.onmessage = (ev) => onEvent(ev.data);
    // The server emits named events ("download"); fall back to default too.
    es.addEventListener("download", (ev) =>
      onEvent((ev as MessageEvent).data as string),
    );
    es.onerror = () => {
      connected.value = false;
      es?.close();
      es = null;
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (closed) return;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => {
      void fetchDownloads()
        .then(applyListing)
        .catch(() => undefined);
      connect();
    }, 2000);
  }

  // Boot: initial snapshot then subscribe.
  void fetchDownloads()
    .then(applyListing)
    .catch(() => undefined);
  connect();

  async function enqueue(model: string): Promise<void> {
    await postDownload(model);
  }

  async function cancel(id: string): Promise<void> {
    await cancelDownload(id);
  }

  function close() {
    closed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    es?.close();
    es = null;
  }

  return { active, queued, history, ratesByJob, enqueue, cancel, connected, close };
}
```

- [ ] **Step 4: Run reducer tests**

```bash
cd web && bun run test useDownloads
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/composables/useDownloads.ts web/src/composables/useDownloads.test.ts
git commit -m "$(cat <<'EOF'
feat(web): useDownloads composable with SSE, reducer, ETA math

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Web — `DownloadsDrawer.vue`

**Files:**
- Create: `web/src/components/DownloadsDrawer.vue`
- Create: `web/src/components/DownloadsDrawer.test.ts`

Depends on: Task 15

- [ ] **Step 1: Write failing render test**

Create `web/src/components/DownloadsDrawer.test.ts`:

```ts
import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import DownloadsDrawer from "./DownloadsDrawer.vue";

const makeJob = (over: Record<string, unknown> = {}) => ({
  id: "a",
  model: "flux-dev:q4",
  status: "active",
  files_done: 1,
  files_total: 5,
  bytes_done: 500_000,
  bytes_total: 1_000_000,
  current_file: "transformer.gguf",
  started_at: Date.now(),
  completed_at: null,
  error: null,
  ...over,
});

describe("DownloadsDrawer", () => {
  it("renders active job with progress text", () => {
    const wrapper = mount(DownloadsDrawer, {
      props: {
        open: true,
        active: makeJob(),
        queued: [],
        history: [],
        etaSeconds: 5,
      },
    });
    expect(wrapper.text()).toContain("flux-dev:q4");
    expect(wrapper.text()).toContain("transformer.gguf");
    expect(wrapper.text()).toMatch(/1\s*\/\s*5/);
  });

  it("renders queued chips with position", () => {
    const wrapper = mount(DownloadsDrawer, {
      props: {
        open: true,
        active: null,
        queued: [
          makeJob({ id: "q1", model: "sd1.5:fp16", status: "queued" }),
          makeJob({ id: "q2", model: "flux-schnell:q4", status: "queued" }),
        ],
        history: [],
        etaSeconds: null,
      },
    });
    expect(wrapper.text()).toContain("sd1.5:fp16");
    expect(wrapper.text()).toContain("flux-schnell:q4");
    expect(wrapper.text()).toMatch(/#1/);
    expect(wrapper.text()).toMatch(/#2/);
  });

  it("shows retry button for failed history entries", () => {
    const onRetry = vi.fn();
    const wrapper = mount(DownloadsDrawer, {
      props: {
        open: true,
        active: null,
        queued: [],
        history: [
          makeJob({
            id: "h1",
            status: "failed",
            error: "network blip",
          }),
        ],
        etaSeconds: null,
      },
      attrs: { onRetry },
    });
    const btn = wrapper.get("[data-test=retry-h1]");
    btn.trigger("click");
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
```

- [ ] **Step 2: Run — expect failure**

```bash
cd web && bun run test DownloadsDrawer
```

- [ ] **Step 3: Implement the component**

Create `web/src/components/DownloadsDrawer.vue`:

```vue
<script setup lang="ts">
import { computed } from "vue";
import type { DownloadJobWire } from "../types";

const props = defineProps<{
  open: boolean;
  active: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  etaSeconds: number | null;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "cancel", id: string): void;
  (e: "retry", model: string): void;
}>();

function formatGb(bytes: number): string {
  if (!bytes) return "—";
  const gb = bytes / 1_073_741_824;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1_048_576).toFixed(0)} MB`;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

const activePct = computed(() => {
  const a = props.active;
  if (!a || a.bytes_total === 0) return 0;
  return Math.min(100, Math.round((a.bytes_done / a.bytes_total) * 100));
});
</script>

<template>
  <aside
    v-if="open"
    class="glass fixed inset-y-4 right-4 z-40 flex w-[min(420px,90vw)] flex-col gap-4 overflow-y-auto rounded-3xl p-5"
    aria-label="Downloads"
  >
    <header class="flex items-center justify-between">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-ink-200">
        Downloads
      </h2>
      <button
        class="rounded-full border border-white/5 bg-white/5 px-3 py-1 text-xs text-ink-200 hover:text-white"
        @click="emit('close')"
      >
        Close
      </button>
    </header>

    <!-- Active -->
    <section v-if="active" class="rounded-2xl border border-white/5 bg-white/5 p-3">
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">Active</div>
      <div class="flex items-center justify-between">
        <div class="text-sm font-medium text-ink-50">{{ active.model }}</div>
        <button
          class="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-200 hover:bg-red-500/40"
          @click="emit('cancel', active.id)"
        >
          Cancel
        </button>
      </div>
      <div class="mt-1 text-xs text-ink-300">
        {{ formatGb(active.bytes_done) }} / {{ formatGb(active.bytes_total) }}
        · {{ active.files_done }}/{{ active.files_total }}
        · ETA {{ formatEta(etaSeconds) }}
      </div>
      <div
        class="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10"
        role="progressbar"
        :aria-valuenow="activePct"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div
          class="h-full bg-brand-400 transition-[width]"
          :style="{ width: activePct + '%' }"
        />
      </div>
      <div v-if="active.current_file" class="mt-2 truncate text-xs text-ink-400">
        {{ active.current_file }}
      </div>
    </section>

    <!-- Queued -->
    <section v-if="queued.length" class="rounded-2xl border border-white/5 bg-white/5 p-3">
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Queued ({{ queued.length }})
      </div>
      <ul class="flex flex-col gap-1">
        <li
          v-for="(job, idx) in queued"
          :key="job.id"
          class="flex items-center justify-between text-sm"
        >
          <span class="truncate text-ink-100">{{ job.model }}</span>
          <span class="text-xs text-ink-400">#{{ idx + 1 }}</span>
          <button
            class="ml-2 rounded-full bg-white/10 px-2 py-0.5 text-xs text-ink-200 hover:text-white"
            :aria-label="'Cancel queued ' + job.model"
            @click="emit('cancel', job.id)"
          >
            ×
          </button>
        </li>
      </ul>
    </section>

    <!-- Recent -->
    <section v-if="history.length" class="rounded-2xl border border-white/5 bg-white/5 p-3">
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">Recent</div>
      <ul class="flex flex-col gap-1">
        <li
          v-for="job in [...history].reverse()"
          :key="job.id"
          class="flex items-center justify-between gap-2 text-sm"
        >
          <span class="flex min-w-0 items-center gap-2">
            <span
              class="rounded-full px-1.5 py-0.5 text-[10px] uppercase"
              :class="{
                'bg-emerald-500/20 text-emerald-200': job.status === 'completed',
                'bg-red-500/20 text-red-200': job.status === 'failed',
                'bg-slate-500/30 text-slate-200': job.status === 'cancelled',
              }"
            >
              {{ job.status }}
            </span>
            <span class="truncate text-ink-100">{{ job.model }}</span>
          </span>
          <button
            v-if="job.status === 'failed'"
            :data-test="'retry-' + job.id"
            class="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-100 hover:bg-brand-500/40"
            @click="emit('retry', job.model)"
          >
            Retry
          </button>
        </li>
      </ul>
    </section>

    <p
      v-if="!active && !queued.length && !history.length"
      class="text-center text-sm text-ink-400"
    >
      No downloads yet.
    </p>
  </aside>
</template>
```

- [ ] **Step 4: Run tests**

```bash
cd web && bun run test DownloadsDrawer
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/DownloadsDrawer.vue web/src/components/DownloadsDrawer.test.ts
git commit -m "$(cat <<'EOF'
feat(web): DownloadsDrawer.vue (active/queued/recent sections)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Web — `ModelPicker.vue` changes

**Files:**
- Modify: `web/src/components/ModelPicker.vue`

Depends on: Task 15

- [ ] **Step 1: Wire in `useDownloads` and size/download UI**

Replace the `<script setup>` block and the per-model `<button>` with this richer form. Full file:

```vue
<script setup lang="ts">
import { computed, ref } from "vue";
import type { ModelInfoExtended } from "../types";
import { VIDEO_FAMILIES } from "../types";
import { useDownloads } from "../composables/useDownloads";

const props = defineProps<{
  models: ModelInfoExtended[];
  modelValue: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: string): void;
  (e: "select", model: ModelInfoExtended): void;
}>();

const SHOW_ALL_KEY = "mold.generate.showAllModels";
const showAll = ref(localStorage.getItem(SHOW_ALL_KEY) === "true");

function setShowAll(v: boolean) {
  showAll.value = v;
  try {
    localStorage.setItem(SHOW_ALL_KEY, String(v));
  } catch {
    /* ignore */
  }
}

const visibleModels = computed(() =>
  props.models.filter((m) => (showAll.value ? true : m.downloaded)),
);

const imageModels = computed(() =>
  visibleModels.value.filter((m) => !VIDEO_FAMILIES.includes(m.family)),
);
const videoModels = computed(() =>
  visibleModels.value.filter((m) => VIDEO_FAMILIES.includes(m.family)),
);

const downloads = useDownloads();

function downloadStateFor(name: string):
  | { kind: "idle" }
  | { kind: "active"; pct: number }
  | { kind: "queued"; position: number; id: string }
  | { kind: "failed"; id: string } {
  if (downloads.active.value && downloads.active.value.model === name) {
    const a = downloads.active.value;
    const pct = a.bytes_total
      ? Math.min(100, Math.round((a.bytes_done / a.bytes_total) * 100))
      : 0;
    return { kind: "active", pct };
  }
  const q = downloads.queued.value.findIndex((j) => j.model === name);
  if (q >= 0)
    return { kind: "queued", position: q + 1, id: downloads.queued.value[q].id };
  const failed = downloads.history.value.find(
    (j) => j.model === name && j.status === "failed",
  );
  if (failed) return { kind: "failed", id: failed.id };
  return { kind: "idle" };
}

function fmtSize(m: ModelInfoExtended): string {
  if (m.size_gb >= 1) return `${m.size_gb.toFixed(1)} GB`;
  return `${(m.size_gb * 1024).toFixed(0)} MB`;
}

function onPick(model: ModelInfoExtended) {
  if (!model.downloaded) return;
  emit("update:modelValue", model.name);
  emit("select", model);
}

async function startDownload(model: ModelInfoExtended) {
  try {
    await downloads.enqueue(model.name);
  } catch (err) {
    console.error("failed to enqueue download", err);
  }
}

async function cancelQueued(id: string) {
  await downloads.cancel(id);
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <label
      class="flex items-center justify-between text-xs uppercase text-slate-400"
    >
      <span>Model</span>
      <span class="flex items-center gap-2 normal-case">
        <input
          id="mold-show-all-models"
          type="checkbox"
          :checked="showAll"
          @change="setShowAll(($event.target as HTMLInputElement).checked)"
        />
        <label for="mold-show-all-models">Show all</label>
      </span>
    </label>

    <div class="flex max-h-80 flex-col gap-3 overflow-y-auto pr-1">
      <div>
        <div class="text-xs font-medium text-slate-500">Images</div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in imageModels" :key="m.name">
            <div
              class="group w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
              ]"
            >
              <button
                type="button"
                class="flex w-full items-center justify-between gap-2"
                :disabled="!m.downloaded"
                :class="!m.downloaded ? 'cursor-default opacity-70' : ''"
                :title="m.description"
                @click="onPick(m)"
              >
                <span class="flex items-center gap-2">
                  <span>{{ m.name }}</span>
                  <span class="text-xs text-slate-400">({{ fmtSize(m) }})</span>
                </span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </button>
              <div class="text-xs text-slate-400">{{ m.description }}</div>

              <!-- Download affordance row -->
              <div v-if="!m.downloaded" class="mt-2">
                <template v-if="downloadStateFor(m.name).kind === 'idle'">
                  <button
                    class="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-100 hover:bg-brand-500/40"
                    @click="startDownload(m)"
                  >
                    Download
                  </button>
                </template>
                <template v-else-if="downloadStateFor(m.name).kind === 'active'">
                  <div
                    class="h-1.5 w-full overflow-hidden rounded-full bg-white/10"
                    role="progressbar"
                  >
                    <div
                      class="h-full bg-brand-400"
                      :style="{ width: (downloadStateFor(m.name) as any).pct + '%' }"
                    />
                  </div>
                  <div class="mt-1 text-xs text-slate-400">
                    Downloading… {{ (downloadStateFor(m.name) as any).pct }}%
                  </div>
                </template>
                <template v-else-if="downloadStateFor(m.name).kind === 'queued'">
                  <span
                    class="inline-flex items-center gap-1 rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-200"
                  >
                    Queued (#{{ (downloadStateFor(m.name) as any).position }})
                    <button
                      class="ml-1 text-slate-300 hover:text-white"
                      aria-label="Cancel queued download"
                      @click="cancelQueued((downloadStateFor(m.name) as any).id)"
                    >×</button>
                  </span>
                </template>
                <template v-else-if="downloadStateFor(m.name).kind === 'failed'">
                  <button
                    class="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-200 hover:bg-red-500/40"
                    @click="startDownload(m)"
                  >
                    Retry
                  </button>
                </template>
              </div>
            </div>
          </li>
        </ul>
      </div>

      <div v-if="videoModels.length">
        <div class="flex items-center gap-2 text-xs font-medium text-slate-500">
          <span>🎬</span><span>Video</span>
        </div>
        <ul class="mt-1 flex flex-col gap-1">
          <li v-for="m in videoModels" :key="m.name">
            <div
              class="w-full rounded-xl px-3 py-2 text-left text-sm"
              :class="[
                modelValue === m.name
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200',
              ]"
            >
              <button
                type="button"
                class="flex w-full items-center justify-between gap-2"
                :disabled="!m.downloaded"
                :class="!m.downloaded ? 'cursor-default opacity-70' : ''"
                :title="m.description"
                @click="onPick(m)"
              >
                <span>
                  {{ m.name }}
                  <span class="italic text-xs text-slate-400">video</span>
                  <span class="ml-1 text-xs text-slate-400">({{ fmtSize(m) }})</span>
                </span>
                <span class="text-xs text-slate-400">{{ m.family }}</span>
              </button>
              <div class="text-xs text-slate-400">{{ m.description }}</div>
              <div v-if="!m.downloaded" class="mt-2">
                <button
                  class="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-100 hover:bg-brand-500/40"
                  @click="startDownload(m)"
                >
                  Download
                </button>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>
```

- [ ] **Step 2: Run unit tests**

```bash
cd web && bun run test
```

Expected: pass. If `ModelPicker` has existing tests, they must still pass — fix any expected-text drift.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/ModelPicker.vue
git commit -m "$(cat <<'EOF'
feat(web): ModelPicker size column + inline download/queued UI

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Web — mount `DownloadsDrawer` + `useDownloads` in `App.vue`; badge in `TopBar.vue`

**Files:**
- Modify: `web/src/App.vue`
- Modify: `web/src/components/TopBar.vue`

Depends on: Task 16

- [ ] **Step 1: Update `App.vue`**

Replace `web/src/App.vue` with:

```vue
<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import DownloadsDrawer from "./components/DownloadsDrawer.vue";
import {
  computeEtaSeconds,
  onDownloadComplete,
  useDownloads,
} from "./composables/useDownloads";
import { fetchModels } from "./api";

// Singleton — mounted once, survives navigation.
const downloads = useDownloads();
const drawerOpen = ref(false);

function openDownloads() {
  drawerOpen.value = true;
}
function closeDownloads() {
  drawerOpen.value = false;
}

// Exposed to child components via provide/inject-free singleton from useDownloads.
// We additionally listen for the page-level event "mold:open-downloads" so
// TopBar can open the drawer without a prop drill when it lives inside a page.
function onOpenEvent() {
  openDownloads();
}
window.addEventListener("mold:open-downloads", onOpenEvent);

const off = onDownloadComplete(() => {
  // Best-effort: if the Generate page listens, it can refresh its own models
  // list too. We refresh anyway for the picker.
  void fetchModels().catch(() => undefined);
});

onBeforeUnmount(() => {
  window.removeEventListener("mold:open-downloads", onOpenEvent);
  off();
});

const etaSeconds = computed(() => {
  const a = downloads.active.value;
  if (!a) return null;
  const samples = downloads.ratesByJob.value[a.id] ?? [];
  return computeEtaSeconds(samples, a.bytes_total);
});

async function handleCancel(id: string) {
  await downloads.cancel(id);
}
async function handleRetry(model: string) {
  await downloads.enqueue(model);
}
</script>

<template>
  <router-view />
  <DownloadsDrawer
    :open="drawerOpen"
    :active="downloads.active.value"
    :queued="downloads.queued.value"
    :history="downloads.history.value"
    :eta-seconds="etaSeconds"
    @close="closeDownloads"
    @cancel="handleCancel"
    @retry="handleRetry"
  />
</template>
```

- [ ] **Step 2: Update `TopBar.vue` to add a Downloads button + badge**

In `web/src/components/TopBar.vue`, add inside the `<script setup>` block (append to existing imports):

```ts
import { computed } from "vue";
import { useDownloads } from "../composables/useDownloads";

const downloads = useDownloads();
const badgeCount = computed(
  () =>
    (downloads.active.value ? 1 : 0) + downloads.queued.value.length,
);

function openDownloadsDrawer() {
  window.dispatchEvent(new CustomEvent("mold:open-downloads"));
}
```

Then, inside the `<template>` — specifically inside the right-hand button cluster (the `<div v-if="$route.name === 'gallery'">` block), insert **outside** the `v-if` so it renders on every page. Place this immediately after the `<nav>` block (around line 109):

```html
    <button
      type="button"
      class="relative inline-flex h-10 items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3 text-sm text-ink-200 hover:text-white"
      aria-label="Open downloads"
      @click="openDownloadsDrawer"
    >
      <svg
        class="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M12 3v12" />
        <path d="m7 10 5 5 5-5" />
        <path d="M5 21h14" />
      </svg>
      <span class="hidden sm:inline">Downloads</span>
      <span
        v-if="badgeCount > 0"
        class="absolute -right-1 -top-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-brand-500 px-1 text-[11px] font-medium text-white"
        aria-label="Pending download count"
      >
        {{ badgeCount }}
      </span>
    </button>
```

- [ ] **Step 3: Smoke test compilation**

```bash
cd web && bun run fmt:check && bun run verify && bun run build
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/App.vue web/src/components/TopBar.vue
git commit -m "$(cat <<'EOF'
feat(web): mount useDownloads singleton + DownloadsDrawer + TopBar badge

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Full green gate + PR into umbrella

**Files:** None (verification)

Depends on: Tasks 1-18

- [ ] **Step 1: Run the full gate from spec §4.3**

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
( cd web && bun run fmt:check && bun run test && bun run build )
```

Expected: every command exits 0. Fix any drift before continuing.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin wt/downloads-ui
```

Expected: push succeeds.

- [ ] **Step 3: Open PR into umbrella `feat/model-ui-overhaul`**

```bash
gh pr create --base feat/model-ui-overhaul --head wt/downloads-ui \
  --title "feat: downloads UI (Phase 1 — Agent A)" \
  --body "$(cat <<'EOF'
## Summary
- Server-side download queue (`DownloadQueue`) with single-writer driver task,
  cancellation (active + queued), 1x auto-retry, and broadcast-backed SSE.
- New routes: `POST /api/downloads`, `DELETE /api/downloads/:id`,
  `GET /api/downloads`, `GET /api/downloads/stream`.
- `POST /api/models/pull` now delegates to the queue (compat shim — TUI wire
  shape preserved).
- Web: `useDownloads` singleton composable, `DownloadsDrawer.vue`, TopBar
  button + badge, `ModelPicker.vue` inline Download / Queued / Retry UI, client
  ETA math.

Targets spec `docs/superpowers/specs/2026-04-19-model-ui-overhaul-design.md`
§1 and §4.{1,2,3}. Does NOT touch `/api/resources*` or `DevicePlacement`.

## Test plan
- [x] `cargo test --workspace`
- [x] `cargo clippy --workspace -- -D warnings`
- [x] `cargo fmt --check`
- [x] `cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4`
- [x] `cd web && bun run fmt:check && bun run test && bun run build`
- [ ] UAT on <gpu-host> (pull sd1.5:fp16, enqueue flux-schnell:q4, cancel second, cancel first mid-stream, reload browser mid-download)
EOF
)"
```

Return the PR URL in the final report.

---

## Dependency graph

```
Task 1  (worktree)
  └─► Task 2 (types.rs downloads types)
        └─► Task 3 (DownloadQueue skeleton + AppState wiring)
              └─► Task 4 (PullDriver + driver task)
                    ├─► Task 5 (cancellation tests)
                    ├─► Task 6 (retry tests)
                    ├─► Task 7 (boot driver from run_server)
                    └─► Task 8 (POST /api/downloads)
                          ├─► Task 9 (DELETE /api/downloads/:id)
                          ├─► Task 10 (GET /api/downloads listing)
                          │     └─► Task 11 (GET /api/downloads/stream SSE)
                          │           └─► Task 12 (compat shim for /api/models/pull)
                          └─► Task 13 (web types.ts)
                                └─► Task 14 (web api.ts helpers)
                                      └─► Task 15 (useDownloads composable)
                                            ├─► Task 16 (DownloadsDrawer.vue)
                                            │     └─► Task 18 (App.vue + TopBar badge)
                                            └─► Task 17 (ModelPicker.vue)
                                                  └─► Task 18 (…)
                                                        └─► Task 19 (full gate + PR)
```

**Parallel opportunities inside a single session:** Tasks 5, 6, 7 can each run right after Task 4 with independent edits. Tasks 9 and 10 can run in either order after Task 8. Tasks 13 and 8 are independent and can be interleaved.
