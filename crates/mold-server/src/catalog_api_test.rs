use std::sync::Arc;

use mold_catalog::scanner::{ProgressHandle, ScanReport};
use tokio::sync::Mutex;

use super::{CatalogScanQueue, CatalogScanStatus, ScanDriver};

struct FakeDriver {
    report: Arc<Mutex<Option<ScanReport>>>,
}

#[async_trait::async_trait]
impl ScanDriver for FakeDriver {
    async fn run(
        &self,
        _opts: mold_catalog::scanner::ScanOptions,
        _progress: ProgressHandle,
    ) -> ScanReport {
        self.report.lock().await.take().unwrap_or_default()
    }
}

#[tokio::test]
async fn enqueue_then_status_transitions_to_done() {
    let report = Arc::new(Mutex::new(Some(ScanReport {
        per_family: Default::default(),
        total_entries: 7,
    })));
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");
    // Wait for completion.
    for _ in 0..50 {
        if let Some(CatalogScanStatus::Done { total_entries, .. }) = queue.status(&id).await {
            assert_eq!(total_entries, 7);
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("scan never reached Done");
}

#[tokio::test]
async fn second_enqueue_while_running_is_rejected() {
    let report = Arc::new(Mutex::new(None)); // empty → driver hangs forever
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });
    queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("first enqueue");
    let second = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await;
    assert!(second.is_err(), "single-writer guarantee");
}

/// `active()` is what the web UI's `GET /api/catalog/refresh` handler
/// reads to decide whether to disable the refresh button. It must
/// return `None` for an idle queue and `Some((id, status))` for any
/// in-flight scan, regardless of who enqueued it.
#[tokio::test]
async fn active_returns_in_flight_scan_id_and_status() {
    let report = Arc::new(Mutex::new(None));
    let driver = Arc::new(FakeDriver { report });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver.clone()).await });

    assert!(queue.active().await.is_none(), "idle queue → no active");

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");

    let (active_id, active_status) = queue.active().await.expect("active scan present");
    assert_eq!(active_id, id);
    assert!(
        matches!(
            active_status,
            CatalogScanStatus::Pending | CatalogScanStatus::Running { .. }
        ),
        "expected pending or running, got {:?}",
        active_status
    );
}

/// While a scan is running, `status()` must derive Running from the
/// live progress handle. The driver writes `families_done = 4`; the
/// queue must surface that — not a stale `families_done = 0`.
#[tokio::test]
async fn status_reflects_live_progress_during_running_scan() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());

    struct ProgressDriver {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    #[async_trait::async_trait]
    impl ScanDriver for ProgressDriver {
        async fn run(
            &self,
            _opts: mold_catalog::scanner::ScanOptions,
            progress: ProgressHandle,
        ) -> ScanReport {
            {
                let mut p = progress.lock().expect("progress");
                p.families_total = 9;
                p.families_done = 4;
                p.current_family = Some(mold_catalog::families::Family::Sdxl);
                p.current_stage = Some("hf");
            }
            self.started.notify_one();
            self.release.notified().await;
            ScanReport::default()
        }
    }

    let driver = Arc::new(ProgressDriver {
        started: started.clone(),
        release: release.clone(),
    });
    let queue = CatalogScanQueue::new();
    let q2 = queue.clone();
    tokio::spawn(async move { q2.drive(driver).await });

    let id = queue
        .enqueue(mold_catalog::scanner::ScanOptions::default())
        .await
        .expect("enqueue");
    started.notified().await;

    let status = queue.status(&id).await.expect("status while running");
    match status {
        CatalogScanStatus::Running {
            families_total,
            families_done,
            current_family,
            current_stage,
            started_at_ms,
        } => {
            assert_eq!(families_total, 9);
            assert_eq!(families_done, 4);
            assert_eq!(current_family.as_deref(), Some("sdxl"));
            assert_eq!(current_stage.as_deref(), Some("hf"));
            assert!(started_at_ms > 0);
        }
        other => panic!("expected Running with live counters, got {:?}", other),
    }

    release.notify_one();
}
