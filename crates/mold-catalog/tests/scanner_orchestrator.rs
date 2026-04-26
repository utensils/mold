use std::sync::{Arc, Mutex};

use mold_catalog::families::Family;
use mold_catalog::scanner::{
    run_scan, run_scan_with_progress, FamilyScanOutcome, ScanOptions, ScanProgress,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn run_scan_aggregates_per_family_outcomes_and_isolates_failures() {
    let hf_server = MockServer::start().await;
    let cv_server = MockServer::start().await;

    // FLUX seed succeeds.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&hf_server)
        .await;

    // Other families: HF returns 503; civitai returns 401. Per-family
    // failure isolation means FLUX still succeeds.
    // (Mounted last so the more specific FLUX matchers above match first.)
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&cv_server)
        .await;

    let opts = ScanOptions {
        families: vec![Family::Flux, Family::Sdxl],
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };
    let report = run_scan(&hf_server.uri(), &cv_server.uri(), &opts).await;

    let flux = report.per_family.get(&Family::Flux).expect("flux outcome");
    assert!(matches!(flux, FamilyScanOutcome::Ok { .. }));
    let sdxl = report.per_family.get(&Family::Sdxl).expect("sdxl outcome");
    // SDXL hit the catch-all 503 / 401 mocks → NetworkError or AuthRequired.
    assert!(matches!(
        sdxl,
        FamilyScanOutcome::NetworkError { .. } | FamilyScanOutcome::AuthRequired
    ));
    assert!(report.total_entries > 0);
}

/// `run_scan_with_progress` must seed `families_total` immediately and
/// advance `families_done` past zero by the time it finishes — the web
/// UI binds a determinate progress bar to those fields. Anything that
/// silently breaks this contract makes the bar permanently stuck at 0%.
#[tokio::test]
async fn run_scan_with_progress_advances_families_done() {
    let hf_server = MockServer::start().await;
    let cv_server = MockServer::start().await;

    // All requests are 503 — every family ends in NetworkError, but the
    // scanner must still advance `families_done` per family attempted.
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&hf_server)
        .await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&cv_server)
        .await;

    let opts = ScanOptions {
        families: vec![Family::Flux, Family::Sdxl, Family::Sd15],
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };
    let progress: Arc<Mutex<ScanProgress>> = Arc::new(Mutex::new(ScanProgress::default()));
    let _ = run_scan_with_progress(
        &hf_server.uri(),
        &cv_server.uri(),
        &opts,
        Some(progress.clone()),
    )
    .await;

    let final_progress = progress.lock().expect("progress lock");
    assert_eq!(final_progress.families_total, 3);
    assert_eq!(final_progress.families_done, 3);
    assert!(final_progress.current_family.is_none());
    assert!(final_progress.current_stage.is_none());
}
