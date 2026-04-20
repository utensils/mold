//! Tests for the download queue. Run with:
//!   cargo test -p mold-ai-server downloads --lib
//!
//! These tests never touch HuggingFace — they inject a fake `PullDriver` so
//! the queue logic can be exercised in isolation.

use crate::downloads::DownloadQueue;

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

use crate::downloads::{spawn_driver, PullDriver};
use mold_core::types::{DownloadEvent, JobStatus};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

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

    assert!(
        puller.called.load(Ordering::SeqCst),
        "puller was not invoked"
    );
    assert!(seen_started, "missing Started event");
    assert!(seen_progress, "missing Progress event");
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
    let driver = spawn_driver(queue.clone(), Arc::new(SlowPuller), shutdown.clone());

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

/// Process-wide gate for tests that mutate `MOLD_MODELS_DIR`. Env vars are
/// shared across threads, so these tests can't run in parallel without
/// clobbering each other.
static MODELS_DIR_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    let _env_guard = MODELS_DIR_ENV_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let tmp = tempfile::tempdir().expect("tempdir");
    let prev = std::env::var("MOLD_MODELS_DIR").ok();
    // `MOLD_MODELS_DIR` is read-only elsewhere in this test binary; the
    // `MODELS_DIR_ENV_LOCK` gate serializes tests that mutate it.
    std::env::set_var("MOLD_MODELS_DIR", tmp.path());

    // Seed the model dir with a partial file (no `.sha256-verified` marker)
    // and a verified file (with marker). The canonical sanitized dir name
    // for `flux-schnell:q4` is `flux-schnell-q4`.
    let model_dir = tmp.path().join("flux-schnell-q4");
    std::fs::create_dir_all(&model_dir).unwrap();
    let partial = model_dir.join("partial.safetensors");
    let verified = model_dir.join("good.safetensors");
    let verified_marker = model_dir.join("good.safetensors.sha256-verified");
    std::fs::write(&partial, b"partial").unwrap();
    std::fs::write(&verified, b"good").unwrap();
    std::fs::write(&verified_marker, b"").unwrap();

    let queue = DownloadQueue::new();
    let puller = AlwaysFailsPuller::default();
    let shutdown = CancellationToken::new();
    let handle = spawn_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());
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

    // Restore env.
    match prev {
        Some(v) => std::env::set_var("MOLD_MODELS_DIR", v),
        None => std::env::remove_var("MOLD_MODELS_DIR"),
    }

    assert!(seen_failed, "expected JobFailed after retry");
    assert_eq!(
        puller.calls.load(Ordering::SeqCst),
        2,
        "expected exactly 2 attempts"
    );
    assert!(
        !partial.exists(),
        "partial file should be cleaned up after terminal failure"
    );
    assert!(
        verified.exists(),
        "verified file (with marker) should survive terminal failure"
    );
    assert!(
        verified_marker.exists(),
        ".sha256-verified marker should survive terminal failure"
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
    let _env_guard = MODELS_DIR_ENV_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let tmp = tempfile::tempdir().expect("tempdir");
    let prev = std::env::var("MOLD_MODELS_DIR").ok();
    std::env::set_var("MOLD_MODELS_DIR", tmp.path());

    let model_dir = tmp.path().join("flux-schnell-q4");
    std::fs::create_dir_all(&model_dir).unwrap();
    let partial = model_dir.join("partial.safetensors");
    std::fs::write(&partial, b"partial").unwrap();

    let queue = DownloadQueue::new();
    let shutdown = CancellationToken::new();
    let handle = spawn_driver(queue.clone(), Arc::new(CancellablePuller), shutdown.clone());
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

    match prev {
        Some(v) => std::env::set_var("MOLD_MODELS_DIR", v),
        None => std::env::remove_var("MOLD_MODELS_DIR"),
    }

    assert!(seen, "expected JobCancelled");
    assert!(
        !partial.exists(),
        "partial file should be cleaned up after cancel"
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

#[tokio::test]
async fn driver_retries_once_then_succeeds() {
    // The retry backoff is 5 s. We let it run in real time —
    // `tokio::time::pause` interacts poorly with the broadcast receivers +
    // mpsc drain task in this driver, and would make the test flaky.
    let queue = DownloadQueue::new();
    let puller = FlakyPuller::default();
    let shutdown = CancellationToken::new();
    let handle = spawn_driver(queue.clone(), Arc::new(puller.clone()), shutdown.clone());
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
