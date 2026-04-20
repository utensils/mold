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
        let active = self.active.lock().await;
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
