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
            if let Some((idx, existing)) = queued
                .iter()
                .enumerate()
                .find(|(_, j)| j.model == canonical)
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
                let _ = self
                    .events
                    .send(DownloadEvent::Dequeued { id: id.to_string() });
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

// ── PullDriver trait + real & test implementations ──────────────────────────

/// Trait that hides the HuggingFace pull behind something the tests can fake.
///
/// The real implementation in `HfPullDriver` calls
/// `mold_core::download::pull_and_configure_with_callback`. Tests inject a stub.
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
                // Terminal cleanup (markers + partials) happens in
                // `try_pull_with_retry` so it covers both cancel and failure
                // paths uniformly. Just surface the cancel error here.
                Err("cancelled".into())
            }
        }
    }
}

/// Delete partial files under `<models_dir>/<sanitized>/` for a cancelled or
/// failed pull. Preserves any file that has a sibling `<file>.sha256-verified`
/// marker (and the marker itself) — those represent fully-downloaded,
/// integrity-checked content that should survive a retry/cancel cycle. Also
/// leaves the HF cache under `~/.cache/huggingface` intact so retry is cheap.
fn cleanup_partials_for_model(model: &str) {
    use mold_core::manifest::resolve_model_name;
    let canonical = resolve_model_name(model);
    let sanitized = canonical.replace(':', "-");
    // Remove `.pulling` marker first so `has_pulling_marker` returns false.
    mold_core::download::remove_pulling_marker(&canonical);
    if let Ok(models_dir) = std::env::var("MOLD_MODELS_DIR") {
        let target = std::path::PathBuf::from(models_dir).join(&sanitized);
        cleanup_partials_in_dir(&target);
        return;
    }
    if let Some(home) = dirs::home_dir() {
        let target = home.join(".mold/models").join(&sanitized);
        cleanup_partials_in_dir(&target);
    }
}

/// Marker suffix written after a successful SHA-256 verification.
/// A file `foo.safetensors` is "verified" when `foo.safetensors.sha256-verified`
/// exists next to it.
const SHA256_VERIFIED_SUFFIX: &str = ".sha256-verified";

/// Walk `dir` and delete every regular file that does NOT have a sibling
/// `<file>.sha256-verified` marker. Files that are themselves `*.sha256-verified`
/// markers are preserved. Best-effort — I/O errors are swallowed because cleanup
/// runs on terminal paths where we can't recover anyway.
///
/// If `dir` does not exist or is not a directory, this is a no-op. Subdirectories
/// are descended into and pruned if they end up empty.
pub(crate) fn cleanup_partials_in_dir(dir: &std::path::Path) {
    if !dir.is_dir() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            cleanup_partials_in_dir(&path);
            // Prune the directory if it's now empty.
            let _ = std::fs::remove_dir(&path);
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // Keep marker files themselves.
        if name.ends_with(SHA256_VERIFIED_SUFFIX) {
            continue;
        }
        // Keep any file that has a sibling `<file>.sha256-verified` marker.
        let marker = path.with_file_name(format!("{name}{SHA256_VERIFIED_SUFFIX}"));
        if marker.exists() {
            continue;
        }
        let _ = std::fs::remove_file(&path);
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
    let handle_job = job.clone();

    // Install the job as active. Placeholder task handle — we're not tracking
    // it separately because the pull runs inline in this function.
    let active_handle = ActiveHandle {
        job: handle_job,
        abort: cancel.clone(),
        task: tokio::spawn(async {}),
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
                cleanup_partials_for_model(&job.model);
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
                            queue
                                .with_active(|a| {
                                    *a = job.clone();
                                })
                                .await;
                            cleanup_partials_for_model(&job.model);
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
                cleanup_partials_for_model(&job.model);
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
    // Pipe progress events through an mpsc so we can process them serially,
    // maintain ordering, and guarantee they drain before JobDone fires.
    let (tx, mut rx) =
        tokio::sync::mpsc::unbounded_channel::<mold_core::download::DownloadProgressEvent>();
    let on_progress = Box::new(move |evt: mold_core::download::DownloadProgressEvent| {
        let _ = tx.send(evt);
    });

    let queue_for_drain = queue.clone();
    let job_id = job.id.clone();
    let drain_handle = tokio::spawn(async move {
        while let Some(evt) = rx.recv().await {
            translate_event(&queue_for_drain, &job_id, evt).await;
        }
    });

    let result = driver.pull(&job.model, on_progress, cancel.clone()).await;
    // Wait for the drain task to finish (sender dropped when `pull` returned).
    let _ = drain_handle.await;

    match result {
        Ok(()) => Ok(()),
        Err(msg) if cancel.is_cancelled() => {
            let _ = msg;
            Err(AttemptError::Cancelled)
        }
        Err(msg) => Err(AttemptError::Failed(msg)),
    }
}

async fn translate_event(
    queue: &Arc<DownloadQueue>,
    job_id: &str,
    evt: mold_core::download::DownloadProgressEvent,
) {
    use mold_core::download::DownloadProgressEvent as P;
    let queue = queue.clone();
    let id = job_id.to_string();
    {
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
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
