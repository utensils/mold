//! Tests for the download queue. Run with:
//!   cargo test -p mold-ai-server downloads --lib
//!
//! These tests never touch HuggingFace — they inject a fake `PullDriver` so
//! the queue logic can be exercised in isolation.

// The tests use `std::sync::Mutex<()>` to serialize process-global env-var
// mutations; holding the guard across `.await` is intentional under the
// current-thread tokio test runtime.
#![allow(clippy::await_holding_lock)]

use crate::downloads::DownloadQueue;

#[tokio::test]
async fn queue_starts_empty() {
    let queue = DownloadQueue::new_for_test();
    let listing = queue.listing().await;
    assert!(listing.active.is_none());
    assert!(listing.active_jobs.is_empty());
    assert!(listing.queued.is_empty());
    assert!(listing.history.is_empty());
}

#[tokio::test]
async fn enqueue_unknown_model_errors() {
    let queue = DownloadQueue::new_for_test();
    let err = queue.enqueue("no-such-model:xyz".to_string()).await;
    assert!(err.is_err(), "expected unknown-model error");
}

#[tokio::test]
async fn h3_raw_manifest_enqueue_rejects_without_queueing_or_events() {
    let queue = DownloadQueue::new_for_test();
    let mut events = queue.subscribe();

    let error = queue
        .enqueue("hf:MiniMaxAI/MiniMax-H3".to_string())
        .await
        .expect_err("H3 must be rejected before manifest lookup or queue mutation");

    assert!(matches!(
        error,
        crate::downloads::EnqueueError::ModelActivation(_)
    ));
    let listing = queue.listing().await;
    assert!(listing.active_jobs.is_empty());
    assert!(listing.queued.is_empty());
    assert!(listing.history.is_empty());
    assert!(matches!(
        events.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Empty)
    ));
}

use crate::downloads::{spawn_driver, PullDriver, RecipePayload, RecipePullDriver};
use mold_core::types::{DownloadEvent, JobStatus};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Recipe driver that the manifest-flow tests never invoke. If a test
/// expects manifest behavior and the queue accidentally takes the recipe
/// branch, the test sees `Err("noop recipe driver")` rather than silent
/// success.
struct NoopRecipeDriver;

#[async_trait::async_trait]
impl RecipePullDriver for NoopRecipeDriver {
    async fn pull(
        &self,
        _payload: RecipePayload,
        _models_dir: std::path::PathBuf,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        Err("noop recipe driver".into())
    }
}

fn test_models_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mold_dl_test_{}", uuid::Uuid::new_v4().simple()))
}

/// Wrap `spawn_driver` for the manifest-only tests — they don't care
/// about the recipe driver or models dir, so default both.
fn spawn_test_driver(
    queue: Arc<DownloadQueue>,
    driver: Arc<dyn PullDriver>,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    spawn_driver(
        queue,
        driver,
        Arc::new(NoopRecipeDriver),
        test_models_dir(),
        shutdown,
    )
}

/// Fake pull driver that emits a single `Progress` then returns `Ok`.
#[derive(Clone, Default)]
struct FakePuller {
    pub called: Arc<AtomicBool>,
    pub hf_fallback_token: Arc<std::sync::Mutex<Option<String>>>,
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

    async fn pull_with_token_provider(
        &self,
        model: &str,
        hf_fallback_token: crate::downloads::HfFallbackTokenProvider,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        *self.hf_fallback_token.lock().unwrap() = hf_fallback_token();
        self.pull(model, on_progress, cancel).await
    }
}

#[tokio::test]
async fn driver_happy_path_emits_started_progress_jobdone() {
    let queue = DownloadQueue::new();
    let puller = FakePuller::default();
    let shutdown = CancellationToken::new();

    let mut rx = queue.subscribe();
    let driver_handle =
        spawn_test_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());

    // Enqueue with a manifest-known model so validation passes. Use a model
    // that's cheap to resolve — the real pull is replaced by the fake.
    let (id, _pos, _outcome) = queue
        .enqueue_with_hf_fallback("flux-schnell:q4".into(), Some("hf_desktop".into()))
        .await
        .unwrap();

    // Collect events for up to 2 seconds.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    let mut started_totals = Vec::new();
    let mut seen_progress = false;
    let mut seen_preparing_status = false;
    let mut seen_done = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Ok(DownloadEvent::Started {
                id: ev_id,
                files_total,
                bytes_total,
            })) if ev_id == id => {
                started_totals.push((files_total, bytes_total));
            }
            Ok(Ok(DownloadEvent::Progress {
                id: ev_id,
                current_file,
                ..
            })) if ev_id == id => {
                seen_progress = true;
                seen_preparing_status |= current_file.as_deref() == Some("starting");
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

    assert!(
        puller.called.load(Ordering::SeqCst),
        "puller was not invoked"
    );
    assert_eq!(
        *puller.hf_fallback_token.lock().unwrap(),
        Some("hf_desktop".into()),
        "request-scoped HF fallback was not forwarded to the pull driver"
    );
    assert_eq!(
        started_totals.first(),
        Some(&(0, 0)),
        "the queue must publish active/preparing before the pull driver resolves file metadata"
    );
    assert!(
        started_totals.contains(&(1, 1_000)),
        "the later Started update must publish the real transfer totals"
    );
    assert!(seen_progress, "missing Progress event");
    assert!(
        seen_preparing_status,
        "pre-transfer status must be visible to SSE clients"
    );
    assert!(seen_done, "missing JobDone event");

    // History should now hold one Completed entry.
    let listing = queue.listing().await;
    assert!(listing.active.is_none());
    assert_eq!(listing.history.len(), 1);
    assert_eq!(listing.history[0].status, JobStatus::Completed);
}

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
    let driver = spawn_test_driver(queue.clone(), Arc::new(SlowPuller), shutdown.clone());

    let mut rx = queue.subscribe();
    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    // Wait for the job to become active (driver picks it up).
    tokio::time::sleep(Duration::from_millis(250)).await;
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
async fn driver_runs_two_downloads_in_parallel() {
    let queue = DownloadQueue::new();
    let shutdown = CancellationToken::new();
    let driver = spawn_test_driver(queue.clone(), Arc::new(SlowPuller), shutdown.clone());

    let (first, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    let (second, _, _) = queue.enqueue("flux-dev:q4".into()).await.unwrap();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        if queue.listing().await.active_jobs.len() == 2 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "both downloads should become active"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    assert!(queue.cancel(&first).await);
    assert!(queue.cancel(&second).await);
    shutdown.cancel();
    let _ = driver.await;
}

#[tokio::test]
async fn cancel_queued_moves_to_history_and_emits_cancelled() {
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
        DownloadEvent::JobCancelled { id } => assert_eq!(id, id_b),
        other => panic!("expected JobCancelled, got {other:?}"),
    }

    let listing = queue.listing().await;
    assert_eq!(listing.queued.len(), 1);
    assert_eq!(listing.queued[0].id, id_a);
    assert_eq!(listing.history.last().unwrap().status, JobStatus::Cancelled);
}

#[tokio::test]
async fn cancel_queued_forgets_request_scoped_hf_token() {
    let queue = DownloadQueue::new_for_test();
    let (id, _, _) = queue
        .enqueue_with_hf_fallback("flux-schnell:q4".into(), Some("hf_desktop".into()))
        .await
        .unwrap();

    assert!(queue.cancel(&id).await);
    assert_eq!(queue.hf_fallback_token(&id), None);
}

#[tokio::test]
async fn duplicate_queued_request_can_add_request_scoped_hf_token() {
    let queue = DownloadQueue::new_for_test();
    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    let (duplicate_id, _, outcome) = queue
        .enqueue_with_hf_fallback("flux-schnell:q4".into(), Some("hf_desktop".into()))
        .await
        .unwrap();

    assert_eq!(duplicate_id, id);
    assert_eq!(outcome, crate::downloads::EnqueueOutcome::AlreadyPresent);
    assert_eq!(queue.hf_fallback_token(&id).as_deref(), Some("hf_desktop"));
}

#[derive(Clone, Default)]
struct LateTokenPuller {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    seen: Arc<std::sync::Mutex<Option<String>>>,
}

#[async_trait::async_trait]
impl PullDriver for LateTokenPuller {
    async fn pull(
        &self,
        _model: &str,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        Err("late-token test requires provider path".into())
    }

    async fn pull_with_token_provider(
        &self,
        _model: &str,
        hf_fallback_token: crate::downloads::HfFallbackTokenProvider,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        self.started.notify_one();
        self.release.notified().await;
        let token = hf_fallback_token();
        *self.seen.lock().unwrap() = token.clone();
        token
            .map(|_| ())
            .ok_or_else(|| "missing fallback token".into())
    }
}

#[tokio::test]
async fn duplicate_active_request_supplies_late_token_to_running_driver() {
    let queue = DownloadQueue::new_for_test();
    let puller = LateTokenPuller::default();
    let shutdown = CancellationToken::new();
    let driver = spawn_test_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());
    let mut events = queue.subscribe();
    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    puller.started.notified().await;

    let (duplicate_id, position, outcome) = queue
        .enqueue_with_hf_fallback("flux-schnell:q4".into(), Some("hf_desktop".into()))
        .await
        .unwrap();

    assert_eq!(duplicate_id, id);
    assert_eq!(position, 0);
    assert_eq!(outcome, crate::downloads::EnqueueOutcome::AlreadyPresent);
    puller.release.notify_one();

    let completed = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(DownloadEvent::JobDone { id: completed, .. }) = events.recv().await {
                if completed == id {
                    break;
                }
            }
        }
    })
    .await;
    shutdown.cancel();
    let _ = driver.await;

    assert!(completed.is_ok(), "late-token pull did not complete");
    assert_eq!(puller.seen.lock().unwrap().as_deref(), Some("hf_desktop"));
}

#[tokio::test]
async fn cancel_unknown_id_returns_false() {
    let queue = DownloadQueue::new();
    assert!(!queue.cancel("no-such-id").await);
}

#[test]
fn cleanup_preserves_sha256_verified_files_and_deletes_partials() {
    use crate::downloads::cleanup_partials_in_dir;
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    // A "verified" file: both the file and its marker exist — both should survive.
    let verified = dir.join("transformer.gguf");
    let verified_marker = dir.join("transformer.gguf.sha256-verified");
    std::fs::write(&verified, b"verified payload").unwrap();
    std::fs::write(&verified_marker, b"").unwrap();

    // A "partial" file with no marker — should be deleted.
    let partial = dir.join("vae.safetensors");
    std::fs::write(&partial, b"partial payload").unwrap();

    // A stray orphan marker (no matching file) — should survive (marker files
    // are always kept; they're cheap and harmless).
    let orphan_marker = dir.join("orphan.bin.sha256-verified");
    std::fs::write(&orphan_marker, b"").unwrap();

    cleanup_partials_in_dir(dir);

    assert!(
        verified.exists(),
        "verified file should be preserved (has .sha256-verified marker)"
    );
    assert!(
        verified_marker.exists(),
        ".sha256-verified marker should be preserved"
    );
    assert!(
        !partial.exists(),
        "partial file (no marker) should be deleted"
    );
    assert!(
        orphan_marker.exists(),
        "orphan .sha256-verified marker should be preserved"
    );
}

#[test]
fn cleanup_handles_missing_dir_gracefully() {
    use crate::downloads::cleanup_partials_in_dir;
    let tmp = tempfile::tempdir().expect("tempdir");
    let missing = tmp.path().join("does-not-exist");
    // Should not panic.
    cleanup_partials_in_dir(&missing);
    assert!(!missing.exists());
}

#[cfg(unix)]
#[test]
fn cleanup_does_not_follow_symlinks_outside_models_dir() {
    use crate::downloads::cleanup_partials_in_dir;
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().expect("tempdir");
    let models_dir = tmp.path().join("models");
    let model_dir = models_dir.join("flux-schnell-q4");
    std::fs::create_dir_all(&model_dir).unwrap();

    // An "outside" file that must NOT be deleted by cleanup.
    let outside_dir = tmp.path().join("outside");
    std::fs::create_dir_all(&outside_dir).unwrap();
    let outside_file = outside_dir.join("precious.bin");
    std::fs::write(&outside_file, b"do not delete me").unwrap();

    // A real partial inside the model dir — should be deleted.
    let partial = model_dir.join("vae.safetensors");
    std::fs::write(&partial, b"partial payload").unwrap();

    // A file symlink pointing to the outside file.
    let file_link = model_dir.join("escape-file.bin");
    symlink(&outside_file, &file_link).unwrap();

    // A directory symlink pointing to the outside dir.
    let dir_link = model_dir.join("escape-dir");
    symlink(&outside_dir, &dir_link).unwrap();

    cleanup_partials_in_dir(&model_dir);

    // Absolute guarantees: outside-file and outside-dir contents stay intact.
    assert!(
        outside_file.exists(),
        "outside file must not be deleted by cleanup"
    );
    assert!(
        outside_dir.exists(),
        "outside directory must not be deleted by cleanup"
    );
    // The partial inside the model dir must be gone.
    assert!(!partial.exists(), "real partial should be deleted");
    // Symlinks themselves are skipped (we don't chase hand-placed pointers).
    // They may remain as-is; what matters is the target files survive.
    // Accept both "link removed" and "link preserved" outcomes, but never a
    // followed-and-deleted outcome.
    let _ = file_link;
    let _ = dir_link;
}

/// Process-wide gate for tests that call `drain_cleanups()`. The cleanup
/// observation buffer is shared across tests in this binary, so tests that
/// care about it must run serially.
static CLEANUP_OBSERVER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Puller that always fails. Used to verify terminal-failure cleanup runs
/// after both the initial attempt AND the retry have exhausted.
#[derive(Clone, Default)]
struct AlwaysFailsPuller {
    pub calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl PullDriver for AlwaysFailsPuller {
    async fn pull(
        &self,
        _model: &str,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err("simulated terminal failure".into())
    }
}

#[tokio::test]
async fn driver_failed_retry_sequence_cleans_up_partials() {
    let _guard = CLEANUP_OBSERVER_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    // Drain any leftover observations from prior tests.
    let _ = crate::downloads::test_hooks::drain_cleanups();

    let queue = DownloadQueue::new();
    let puller = AlwaysFailsPuller::default();
    let shutdown = CancellationToken::new();
    let handle = spawn_test_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());
    let mut rx = queue.subscribe();

    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    // Wait up to 12 s for the retry sequence (5 s backoff + overhead).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(12);
    let mut seen_failed = false;
    while tokio::time::Instant::now() < deadline {
        if let Ok(Ok(DownloadEvent::JobFailed { id: ev, .. })) =
            tokio::time::timeout(Duration::from_millis(500), rx.recv()).await
        {
            if ev == id {
                seen_failed = true;
                break;
            }
        }
    }
    shutdown.cancel();
    let _ = handle.await;

    assert!(seen_failed, "expected JobFailed after retry");
    assert_eq!(
        puller.calls.load(Ordering::SeqCst),
        2,
        "expected exactly 2 attempts"
    );
    let cleanups = crate::downloads::test_hooks::drain_cleanups();
    assert!(
        cleanups.iter().any(|m| m == "flux-schnell:q4"),
        "cleanup_partials_for_model should have been invoked for the failed model, got {cleanups:?}"
    );
}

/// Puller that waits for cancel, then returns Err("cancelled").
/// Same as `SlowPuller` but defined locally to avoid coupling.
#[derive(Clone, Default)]
struct CancellablePuller;

#[async_trait::async_trait]
impl PullDriver for CancellablePuller {
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
async fn driver_cancel_also_cleans_up_partials() {
    let _guard = CLEANUP_OBSERVER_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let _ = crate::downloads::test_hooks::drain_cleanups();

    let queue = DownloadQueue::new();
    let shutdown = CancellationToken::new();
    let handle = spawn_test_driver(queue.clone(), Arc::new(CancellablePuller), shutdown.clone());
    let mut rx = queue.subscribe();

    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert!(queue.cancel(&id).await);

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
    shutdown.cancel();
    let _ = handle.await;

    assert!(seen, "expected JobCancelled");
    let cleanups = crate::downloads::test_hooks::drain_cleanups();
    assert!(
        cleanups.iter().any(|m| m == "flux-schnell:q4"),
        "cleanup_partials_for_model should have been invoked for the cancelled model, got {cleanups:?}"
    );
}

/// Fails once, then succeeds.
#[derive(Clone, Default)]
struct FlakyPuller {
    pub calls: Arc<AtomicUsize>,
}

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

/// Puller that downloads two files and emits a few `FileProgress` events
/// between the file boundaries. Used to verify that Progress events carry
/// the current `files_done` counter rather than resetting it to 0.
#[derive(Clone, Default)]
struct MultiFilePuller;

#[async_trait::async_trait]
impl PullDriver for MultiFilePuller {
    async fn pull(
        &self,
        _model: &str,
        on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        use mold_core::download::DownloadProgressEvent as P;
        // File 0 — start, mid-progress, done.
        on_progress(P::FileStart {
            filename: "a.safetensors".into(),
            file_index: 0,
            total_files: 2,
            size_bytes: 100,
            batch_bytes_downloaded: 0,
            batch_bytes_total: 200,
            batch_elapsed_ms: 0,
        });
        on_progress(P::FileProgress {
            filename: "a.safetensors".into(),
            file_index: 0,
            bytes_downloaded: 50,
            bytes_total: 100,
            batch_bytes_downloaded: 50,
            batch_bytes_total: 200,
            batch_elapsed_ms: 1,
        });
        on_progress(P::FileDone {
            filename: "a.safetensors".into(),
            file_index: 0,
            total_files: 2,
            batch_bytes_downloaded: 100,
            batch_bytes_total: 200,
            batch_elapsed_ms: 2,
        });
        // File 1 — two mid-progress events (this is where the bug flickered).
        on_progress(P::FileStart {
            filename: "b.safetensors".into(),
            file_index: 1,
            total_files: 2,
            size_bytes: 100,
            batch_bytes_downloaded: 100,
            batch_bytes_total: 200,
            batch_elapsed_ms: 3,
        });
        on_progress(P::FileProgress {
            filename: "b.safetensors".into(),
            file_index: 1,
            bytes_downloaded: 25,
            bytes_total: 100,
            batch_bytes_downloaded: 125,
            batch_bytes_total: 200,
            batch_elapsed_ms: 4,
        });
        on_progress(P::FileProgress {
            filename: "b.safetensors".into(),
            file_index: 1,
            bytes_downloaded: 75,
            bytes_total: 100,
            batch_bytes_downloaded: 175,
            batch_bytes_total: 200,
            batch_elapsed_ms: 5,
        });
        on_progress(P::FileDone {
            filename: "b.safetensors".into(),
            file_index: 1,
            total_files: 2,
            batch_bytes_downloaded: 200,
            batch_bytes_total: 200,
            batch_elapsed_ms: 6,
        });
        Ok(())
    }
}

#[tokio::test]
async fn progress_events_carry_current_files_done_counter() {
    let queue = DownloadQueue::new();
    let shutdown = CancellationToken::new();
    let handle = spawn_test_driver(queue.clone(), Arc::new(MultiFilePuller), shutdown.clone());
    let mut rx = queue.subscribe();

    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    // Collect (files_done, bytes_done) from each Progress event for this job,
    // in order, until JobDone arrives or we time out.
    let mut progress_samples: Vec<(usize, u64)> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Ok(DownloadEvent::Progress {
                id: ev_id,
                files_done,
                bytes_done,
                ..
            })) if ev_id == id => {
                progress_samples.push((files_done, bytes_done));
            }
            Ok(Ok(DownloadEvent::JobDone { id: ev_id, .. })) if ev_id == id => {
                break;
            }
            _ => {}
        }
    }

    shutdown.cancel();
    let _ = handle.await;

    // We expect three Progress events in order:
    //   1. during file 0 (before any file has finished) → files_done = 0
    //   2. during file 1 (after file 0 is done)          → files_done = 1
    //   3. during file 1 again                            → files_done = 1
    assert!(
        progress_samples.len() >= 3,
        "expected at least 3 Progress events, got {progress_samples:?}"
    );
    assert_eq!(
        progress_samples[0].0, 0,
        "first Progress (mid file 0) should report files_done = 0, got {progress_samples:?}",
    );
    // Every Progress sample emitted after the first FileDone must report
    // `files_done >= 1` — i.e. the counter must never reset to 0 mid-stream.
    // The old bug unconditionally emitted 0, causing the drawer to flicker.
    let after_first_done = &progress_samples[1..];
    for (files_done, _) in after_first_done {
        assert!(
            *files_done >= 1,
            "Progress events after the first FileDone must carry files_done >= 1; \
             got sequence {progress_samples:?}",
        );
    }
}

#[tokio::test]
async fn driver_retries_once_then_succeeds() {
    // The retry backoff is 5 s. We let it run in real time —
    // `tokio::time::pause` interacts poorly with the broadcast receivers +
    // mpsc drain task in this driver, and would make the test flaky.
    let queue = DownloadQueue::new();
    let puller = FlakyPuller::default();
    let shutdown = CancellationToken::new();
    let handle = spawn_test_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());
    let mut rx = queue.subscribe();

    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(12);
    let mut seen_done = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
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
    assert_eq!(
        puller.calls.load(Ordering::SeqCst),
        2,
        "expected exactly 2 attempts"
    );
}

// ── Recipe-queue tests (round 4 / Option A') ────────────────────────────────

use crate::downloads::OwnedRecipeFile;

fn synthetic_recipe(catalog_id: &str) -> RecipePayload {
    RecipePayload {
        catalog_id: catalog_id.to_string(),
        files: vec![OwnedRecipeFile {
            url: "http://recipe.example.invalid/primary.safetensors".to_string(),
            dest: "primary.safetensors".to_string(),
            sha256: None,
            size_bytes: Some(123),
        }],
        auth: mold_core::download::RecipeAuth::None,
    }
}

fn synthetic_manifest(name: &str, family: &str, repo: &str) -> mold_core::manifest::ModelManifest {
    use mold_core::manifest::{ManifestDefaults, ModelComponent, ModelFile, ModelManifest};

    ModelManifest {
        name: name.to_string(),
        family: family.to_string(),
        description: "queue policy fixture".to_string(),
        files: vec![ModelFile {
            hf_repo: repo.to_string(),
            hf_filename: "weights.safetensors".to_string(),
            component: ModelComponent::Transformer,
            size_bytes: 1,
            gated: false,
            sha256: None,
        }],
        defaults: ManifestDefaults {
            steps: 1,
            guidance: 1.0,
            width: 32,
            height: 32,
            is_schnell: false,
            scheduler: None,
            negative_prompt: None,
            frames: None,
            fps: None,
            source_image: None,
        },
        hidden: true,
    }
}

#[tokio::test]
async fn h3_resolved_manifest_name_family_and_repo_reject_before_queue_mutation() {
    let cases = [
        ("MiniMax-H3-FL2VA", "custom", "example/weights"),
        ("opaque-model", "minimax-h3", "example/weights"),
        ("opaque-model", "custom", "Comfy-Org/MiniMax-H3"),
    ];

    for (name, family, repo) in cases {
        let queue = DownloadQueue::new_for_test();
        let mut events = queue.subscribe();
        let manifest = synthetic_manifest(name, family, repo);

        let error = queue
            .enqueue_resolved_manifest("opaque-model:q8".to_string(), &manifest, None)
            .await
            .expect_err("resolved H3 metadata must be compliance-gated");

        assert!(matches!(
            error,
            crate::downloads::EnqueueError::ModelActivation(_)
        ));
        let listing = queue.listing().await;
        assert!(listing.active_jobs.is_empty(), "{name}/{family}/{repo}");
        assert!(listing.queued.is_empty(), "{name}/{family}/{repo}");
        assert!(listing.history.is_empty(), "{name}/{family}/{repo}");
        assert!(
            queue.recipe_payloads.lock().unwrap().is_empty(),
            "{name}/{family}/{repo}"
        );
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }
}

#[tokio::test]
async fn h3_recipe_id_url_and_dest_reject_before_queue_mutation() {
    let cases = [
        (
            "hf:MiniMaxAI/MiniMax-H3",
            "https://example.invalid/weights.safetensors",
            "weights.safetensors",
        ),
        (
            "cv:opaque",
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/weights.safetensors",
            "weights.safetensors",
        ),
        (
            "cv:opaque",
            "https://example.invalid/weights.safetensors",
            "MiniMax-H3/weights.safetensors",
        ),
    ];

    for (catalog_id, url, dest) in cases {
        let queue = DownloadQueue::new_for_test();
        let mut events = queue.subscribe();
        let payload = RecipePayload {
            catalog_id: catalog_id.to_string(),
            files: vec![OwnedRecipeFile {
                url: url.to_string(),
                dest: dest.to_string(),
                sha256: None,
                size_bytes: Some(1),
            }],
            auth: mold_core::download::RecipeAuth::None,
        };

        let error = queue
            .enqueue_recipe(payload)
            .await
            .expect_err("H3 recipe metadata must be compliance-gated");

        assert!(matches!(
            error,
            crate::downloads::EnqueueError::ModelActivation(_)
        ));
        let listing = queue.listing().await;
        assert!(listing.active_jobs.is_empty(), "{catalog_id}/{url}/{dest}");
        assert!(listing.queued.is_empty(), "{catalog_id}/{url}/{dest}");
        assert!(listing.history.is_empty(), "{catalog_id}/{url}/{dest}");
        assert!(
            queue.recipe_payloads.lock().unwrap().is_empty(),
            "{catalog_id}/{url}/{dest}"
        );
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }
}

#[tokio::test]
async fn enqueue_recipe_returns_job_id() {
    let queue = DownloadQueue::new_for_test();
    let payload = synthetic_recipe("cv:618692");
    let (job_id, position, outcome) = queue
        .enqueue_recipe(payload.clone())
        .await
        .expect("enqueue_recipe must succeed");
    assert!(!job_id.is_empty());
    assert_eq!(position, 1);
    assert_eq!(outcome, crate::downloads::EnqueueOutcome::Created);

    // Listing should reflect the queued recipe job with the catalog id as `model`.
    let listing = queue.listing().await;
    assert_eq!(listing.queued.len(), 1);
    assert_eq!(listing.queued[0].id, job_id);
    assert_eq!(listing.queued[0].model, "cv:618692");
}

#[tokio::test]
async fn enqueue_recipe_idempotent_on_same_catalog_id() {
    let queue = DownloadQueue::new_for_test();
    let payload = synthetic_recipe("cv:42");
    let (id1, _, outcome1) = queue.enqueue_recipe(payload.clone()).await.unwrap();
    let (id2, _, outcome2) = queue.enqueue_recipe(payload).await.unwrap();
    assert_eq!(
        id1, id2,
        "duplicate enqueue must return the existing job id"
    );
    assert_eq!(outcome1, crate::downloads::EnqueueOutcome::Created);
    assert_eq!(outcome2, crate::downloads::EnqueueOutcome::AlreadyPresent);
    let listing = queue.listing().await;
    assert_eq!(listing.queued.len(), 1, "no duplicate job should be queued");
}

#[tokio::test]
async fn enqueue_recipe_rejects_blank_catalog_id() {
    let queue = DownloadQueue::new_for_test();
    let mut payload = synthetic_recipe("cv:1");
    payload.catalog_id = "   ".into();
    let err = queue.enqueue_recipe(payload).await;
    assert!(err.is_err(), "blank catalog id must error");
}

/// Recipe driver that captures the payload it was called with so tests
/// can assert dispatch.
#[derive(Clone, Default)]
struct CapturingRecipeDriver {
    pub captured: Arc<std::sync::Mutex<Option<RecipePayload>>>,
}

#[async_trait::async_trait]
impl RecipePullDriver for CapturingRecipeDriver {
    async fn pull(
        &self,
        payload: RecipePayload,
        _models_dir: std::path::PathBuf,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        *self.captured.lock().unwrap() = Some(payload);
        Ok(())
    }
}

/// Manifest driver that should NOT be invoked when a recipe payload is
/// present. Calling it surfaces a test failure.
#[derive(Clone, Default)]
struct UnexpectedManifestDriver {
    pub called: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl PullDriver for UnexpectedManifestDriver {
    async fn pull(
        &self,
        _model: &str,
        _on_progress: Box<dyn Fn(mold_core::download::DownloadProgressEvent) + Send + Sync>,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        self.called.store(true, Ordering::SeqCst);
        Err("manifest driver should not have been invoked for a recipe job".into())
    }
}

#[tokio::test]
async fn recipe_job_dispatches_to_recipe_driver_not_manifest_driver() {
    let queue = DownloadQueue::new();
    let recipe_driver = CapturingRecipeDriver::default();
    let manifest_driver = UnexpectedManifestDriver::default();
    let shutdown = CancellationToken::new();
    let models_dir = test_models_dir();
    let handle = spawn_driver(
        queue.clone(),
        Arc::new(manifest_driver.clone()),
        Arc::new(recipe_driver.clone()),
        models_dir,
        shutdown.clone(),
    );

    let mut rx = queue.subscribe();
    let payload = synthetic_recipe("cv:99");
    let (job_id, _, _) = queue.enqueue_recipe(payload.clone()).await.unwrap();

    // Wait for JobDone (recipe driver returns Ok immediately).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    let mut seen_done = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Ok(DownloadEvent::JobDone { id: ev, .. })) if ev == job_id => {
                seen_done = true;
                break;
            }
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => {}
        }
    }

    shutdown.cancel();
    let _ = handle.await;

    assert!(seen_done, "recipe job should complete via recipe driver");
    assert!(
        !manifest_driver.called.load(Ordering::SeqCst),
        "manifest driver must not be invoked for recipe jobs"
    );
    let captured = recipe_driver.captured.lock().unwrap();
    let captured = captured.as_ref().expect("recipe driver should have run");
    assert_eq!(captured.catalog_id, "cv:99");
}

// ── C: catalog group completion event ──────────────────────────────────

/// Drain ready CatalogReady events from the queue's broadcast channel.
/// Returns `(id, ok)` for each. Used by the C tests below.
fn drain_catalog_ready(
    rx: &mut tokio::sync::broadcast::Receiver<DownloadEvent>,
) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(DownloadEvent::CatalogReady { id, ok }) => out.push((id, ok)),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    out
}

#[tokio::test]
async fn catalog_group_emits_ready_when_all_jobs_complete() {
    // Three companions + a primary all complete; CatalogReady fires once
    // with ok=true. Pre-C the SPA fired its model-list refresh on the
    // primary's JobDone, sometimes before companions hit disk — this is
    // the regression test for that bug.
    let queue = DownloadQueue::new_for_test();
    let mut rx = queue.subscribe();

    let (j1, _, _) = queue
        .enqueue_in_group("flux-schnell:q4".into(), "cv:42")
        .await
        .unwrap();
    let (j2, _, _) = queue
        .enqueue_in_group("flux-dev:q4".into(), "cv:42")
        .await
        .unwrap();
    let (j3, _, _) = queue
        .enqueue_in_group("real-esrgan-x4plus:fp16".into(), "cv:42")
        .await
        .unwrap();

    queue.settle_for_group(&j1, JobStatus::Completed);
    queue.settle_for_group(&j2, JobStatus::Completed);
    assert!(
        drain_catalog_ready(&mut rx).is_empty(),
        "no event until last job settles"
    );

    queue.settle_for_group(&j3, JobStatus::Completed);
    let ready = drain_catalog_ready(&mut rx);
    assert_eq!(ready, vec![("cv:42".to_string(), true)]);
}

#[tokio::test]
async fn catalog_group_emits_ready_with_ok_false_on_failure() {
    // Mixed outcome: one Failed → group is settled but ok=false. The SPA
    // can still react (refresh, show repair pill) without losing the signal.
    let queue = DownloadQueue::new_for_test();
    let mut rx = queue.subscribe();

    let (j1, _, _) = queue
        .enqueue_in_group("flux-schnell:q4".into(), "cv:99")
        .await
        .unwrap();
    let (j2, _, _) = queue
        .enqueue_in_group("flux-dev:q4".into(), "cv:99")
        .await
        .unwrap();

    queue.settle_for_group(&j1, JobStatus::Completed);
    queue.settle_for_group(&j2, JobStatus::Failed);

    let ready = drain_catalog_ready(&mut rx);
    assert_eq!(ready, vec![("cv:99".to_string(), false)]);
}

#[tokio::test]
async fn catalog_group_emits_ready_with_ok_false_on_cancellation() {
    // User cancels one of the companions → group settled with ok=false.
    let queue = DownloadQueue::new_for_test();
    let mut rx = queue.subscribe();

    let (j1, _, _) = queue
        .enqueue_in_group("flux-schnell:q4".into(), "cv:7")
        .await
        .unwrap();
    let (j2, _, _) = queue
        .enqueue_in_group("flux-dev:q4".into(), "cv:7")
        .await
        .unwrap();

    queue.settle_for_group(&j1, JobStatus::Completed);
    queue.settle_for_group(&j2, JobStatus::Cancelled);

    let ready = drain_catalog_ready(&mut rx);
    assert_eq!(ready, vec![("cv:7".to_string(), false)]);
}

#[tokio::test]
async fn catalog_group_emits_ready_only_once() {
    // After the group emits, repeated settle calls for the same job ids
    // are no-ops (the group has been removed from the ledger). Guards
    // against a double-refresh storm in the SPA if a stale event leaks.
    let queue = DownloadQueue::new_for_test();
    let mut rx = queue.subscribe();

    let (j1, _, _) = queue
        .enqueue_in_group("flux-schnell:q4".into(), "cv:1")
        .await
        .unwrap();
    queue.settle_for_group(&j1, JobStatus::Completed);
    queue.settle_for_group(&j1, JobStatus::Completed);
    queue.settle_for_group(&j1, JobStatus::Completed);

    let ready = drain_catalog_ready(&mut rx);
    assert_eq!(ready.len(), 1);
}

#[tokio::test]
async fn settle_for_group_is_no_op_for_ungrouped_jobs() {
    // Manifest pulls outside any catalog group must not fire CatalogReady.
    let queue = DownloadQueue::new_for_test();
    let mut rx = queue.subscribe();

    let (id, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    queue.settle_for_group(&id, JobStatus::Completed);

    assert!(
        drain_catalog_ready(&mut rx).is_empty(),
        "ungrouped jobs must not produce CatalogReady"
    );
}

#[tokio::test]
async fn enqueue_in_group_dedupes_membership_for_already_present() {
    // If a manual `mold pull` raced ahead and put `flux-schnell:q4` in the
    // queue, then a catalog enqueue tries to register it under group "cv:5",
    // we still want the existing job to count toward the group — otherwise
    // CatalogReady would never fire because that job's settle never finds
    // the group.
    let queue = DownloadQueue::new_for_test();
    let mut rx = queue.subscribe();

    let (j_pre, _, _) = queue.enqueue("flux-schnell:q4".into()).await.unwrap();
    let (j_in_group, _, outcome) = queue
        .enqueue_in_group("flux-schnell:q4".into(), "cv:5")
        .await
        .unwrap();
    assert_eq!(j_pre, j_in_group, "dedup must return the same job id");
    assert_eq!(
        outcome,
        crate::downloads::EnqueueOutcome::AlreadyPresent,
        "second enqueue should be AlreadyPresent"
    );

    queue.settle_for_group(&j_in_group, JobStatus::Completed);

    let ready = drain_catalog_ready(&mut rx);
    assert_eq!(ready, vec![("cv:5".to_string(), true)]);
}
