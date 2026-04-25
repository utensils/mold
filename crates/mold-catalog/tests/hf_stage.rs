use mold_catalog::families::Family;
use mold_catalog::scanner::{ScanError, ScanOptions};
use mold_catalog::stages::hf;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn scan_family_yields_seed_plus_finetunes() {
    let server = MockServer::start().await;

    // Detail for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;

    // Tree for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;

    // Empty finetune page (no finetunes for this fixture).
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param(
            "filter",
            "base_model:black-forest-labs/FLUX.1-dev",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };
    let entries = hf::scan_family(
        &server.uri(),
        &opts,
        Family::Flux,
        &["black-forest-labs/FLUX.1-dev"],
    )
    .await
    .expect("scan");

    assert!(!entries.is_empty(), "expected at least the seed entry");
    let seed = entries
        .iter()
        .find(|e| e.source_id == "black-forest-labs/FLUX.1-dev")
        .expect("seed present");
    assert_eq!(seed.family, Family::Flux);
}

/// Verifies the throttle layer retries on HTTP 429: the first response
/// for the detail endpoint returns 429 with `Retry-After: 0`, the second
/// returns the real fixture, and the scanner ends up with the seed entry.
/// Without retry, the seed call would fail and the scanner would return
/// `RateLimited` instead.
#[tokio::test]
async fn http_429_is_retried_when_max_retries_allows_it() {
    let server = MockServer::start().await;

    // Detail endpoint: first call → 429 with Retry-After: 0 (so the test
    // doesn't actually wait); second call → success.
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param(
            "filter",
            "base_model:black-forest-labs/FLUX.1-dev",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 2,
        ..ScanOptions::default()
    };

    let entries = hf::scan_family(
        &server.uri(),
        &opts,
        Family::Flux,
        &["black-forest-labs/FLUX.1-dev"],
    )
    .await
    .expect("scan should succeed via retry");
    assert!(!entries.is_empty(), "seed entry must come back after retry");
}

/// Companion to the retry test: with `max_429_retries=0`, the same
/// scenario must surface `RateLimited` to the caller.
#[tokio::test]
async fn http_429_returns_rate_limited_when_no_retries_allowed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };

    let err = hf::scan_family(
        &server.uri(),
        &opts,
        Family::Flux,
        &["black-forest-labs/FLUX.1-dev"],
    )
    .await
    .expect_err("should be rate-limited");
    assert!(matches!(
        err,
        ScanError::RateLimited {
            host: "huggingface.co"
        }
    ));
}
