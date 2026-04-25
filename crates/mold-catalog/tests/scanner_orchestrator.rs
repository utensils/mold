use mold_catalog::families::Family;
use mold_catalog::scanner::{run_scan, FamilyScanOutcome, ScanOptions};
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
