use std::sync::{Arc, Mutex};

use mold_catalog::families::Family;
use mold_catalog::scanner::{ProgressHandle, ScanError, ScanOptions, ScanProgress};
use mold_catalog::stages::hf;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Custom wiremock responder that snapshots a [`ScanProgress`] handle
/// each time it is invoked, then serves a fixed body. The snapshot
/// captures the scanner's progress at the precise moment a request is
/// fired, which lets tests assert on **per-request** progress
/// granularity instead of just end-of-scan state — i.e. it catches
/// implementations that update progress only once at the end of the
/// walk.
struct SnapshottingResponder {
    progress: ProgressHandle,
    captured: Arc<Mutex<Vec<ScanProgress>>>,
    body: String,
}

impl Respond for SnapshottingResponder {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let snap = self.progress.lock().unwrap().clone();
        self.captured.lock().unwrap().push(snap);
        ResponseTemplate::new(200).set_body_string(self.body.clone())
    }
}

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
        None,
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
        None,
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
        None,
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

/// Asserts the scanner writes per-page progress *during* the walk. With
/// 3 non-empty pages followed by an empty page, the snapshotting
/// responder fires 4 times and `pages_done` should advance 0→1→2→3
/// across those snapshots. An implementation that only updates progress
/// at end-of-walk would fail this test.
#[tokio::test]
async fn scan_family_writes_per_page_progress_within_seed() {
    let server = MockServer::start().await;

    // Foundation + tree for the seed.
    Mock::given(method("GET"))
        .and(path("/api/models/test/seed-a"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/test/seed-a/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;

    // One reusable stub for every list page's finetune lookup.
    Mock::given(method("GET"))
        .and(path("/api/models/test/finetune-1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/test/finetune-1/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;

    let progress: ProgressHandle = Arc::new(Mutex::new(ScanProgress::default()));
    let captured: Arc<Mutex<Vec<ScanProgress>>> = Arc::new(Mutex::new(Vec::new()));

    // 3 non-empty pages + 1 empty page (terminator).
    for (page, body) in [
        ("1", r#"[{"id":"test/finetune-1"}]"#),
        ("2", r#"[{"id":"test/finetune-1"}]"#),
        ("3", r#"[{"id":"test/finetune-1"}]"#),
        ("4", "[]"),
    ] {
        let responder = SnapshottingResponder {
            progress: Arc::clone(&progress),
            captured: Arc::clone(&captured),
            body: body.to_string(),
        };
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .and(query_param("filter", "base_model:test/seed-a"))
            .and(query_param("page", page))
            .respond_with(responder)
            .mount(&server)
            .await;
    }

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };

    hf::scan_family(
        &server.uri(),
        &opts,
        Family::Flux,
        &["test/seed-a"],
        Some(&progress),
    )
    .await
    .expect("scan");

    let captured = captured.lock().unwrap();
    assert_eq!(
        captured.len(),
        4,
        "expected 4 list-page requests (3 non-empty + 1 empty terminator)",
    );
    for (i, snap) in captured.iter().enumerate() {
        assert_eq!(
            snap.pages_done,
            i,
            "list-page request #{} should snapshot pages_done={}, was {}",
            i + 1,
            i,
            snap.pages_done,
        );
        assert_eq!(
            snap.current_seed.as_deref(),
            Some("test/seed-a"),
            "current_seed should be seed-a throughout walk",
        );
    }

    let final_snap = progress.lock().unwrap().clone();
    assert_eq!(
        final_snap.pages_done, 3,
        "final pages_done counts non-empty pages"
    );
    assert_eq!(final_snap.current_seed.as_deref(), Some("test/seed-a"));
}

/// Asserts `current_seed` rotates and `pages_done` resets when the
/// scanner moves from one seed to the next. Two seeds, one non-empty
/// page each. The third snapshot (first request of seed-b) must show
/// `pages_done = 0`.
#[tokio::test]
async fn scan_family_resets_pages_done_per_seed() {
    let server = MockServer::start().await;

    for seed in ["test/seed-a", "test/seed-b"] {
        Mock::given(method("GET"))
            .and(path(format!("/api/models/{seed}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(include_str!("fixtures/hf_flux_dev.json")),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/api/models/{seed}/tree/main")))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
            )
            .mount(&server)
            .await;
    }

    Mock::given(method("GET"))
        .and(path("/api/models/test/finetune-1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/test/finetune-1/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;

    let progress: ProgressHandle = Arc::new(Mutex::new(ScanProgress::default()));
    let captured: Arc<Mutex<Vec<ScanProgress>>> = Arc::new(Mutex::new(Vec::new()));

    // Each seed: 1 non-empty page + 1 empty terminator = 2 list requests.
    for seed in ["test/seed-a", "test/seed-b"] {
        let filter = format!("base_model:{seed}");
        for (page, body) in [("1", r#"[{"id":"test/finetune-1"}]"#), ("2", "[]")] {
            let responder = SnapshottingResponder {
                progress: Arc::clone(&progress),
                captured: Arc::clone(&captured),
                body: body.to_string(),
            };
            Mock::given(method("GET"))
                .and(path("/api/models"))
                .and(query_param("filter", filter.as_str()))
                .and(query_param("page", page))
                .respond_with(responder)
                .mount(&server)
                .await;
        }
    }

    let opts = ScanOptions {
        hf_request_delay: std::time::Duration::ZERO,
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        ..ScanOptions::default()
    };

    hf::scan_family(
        &server.uri(),
        &opts,
        Family::Flux,
        &["test/seed-a", "test/seed-b"],
        Some(&progress),
    )
    .await
    .expect("scan");

    let captured = captured.lock().unwrap();
    assert_eq!(captured.len(), 4, "2 seeds × 2 list-page requests each");

    // seed-a: pages_done snapshots 0 then 1.
    assert_eq!(captured[0].current_seed.as_deref(), Some("test/seed-a"));
    assert_eq!(captured[0].pages_done, 0);
    assert_eq!(captured[1].current_seed.as_deref(), Some("test/seed-a"));
    assert_eq!(captured[1].pages_done, 1);

    // seed-b: pages_done MUST reset to 0 here. This is the assertion the
    // bug-symptom UX needs — without per-seed reset the user would see a
    // monotonic page counter that's meaningless across seeds.
    assert_eq!(captured[2].current_seed.as_deref(), Some("test/seed-b"));
    assert_eq!(
        captured[2].pages_done, 0,
        "pages_done must reset to 0 when current_seed changes",
    );
    assert_eq!(captured[3].current_seed.as_deref(), Some("test/seed-b"));
    assert_eq!(captured[3].pages_done, 1);

    // entries_so_far is per-family cumulative, so it should be monotonic
    // across the seed boundary even though pages_done resets.
    for window in captured.windows(2) {
        assert!(
            window[1].entries_so_far >= window[0].entries_so_far,
            "entries_so_far must be monotonic across seeds within a family",
        );
    }
}

/// Defensive bound: with `max_family_wallclock` set, even a runaway
/// pagination loop must terminate. This is the belt-and-braces against
/// hypothesis B in the investigation handoff (a true tokio-future hang
/// or infinite pagination); without it, a stuck scan can wedge for
/// hours while the user stares at a stale `families_done:0`.
///
/// We assert termination via `tokio::time::timeout` rather than
/// measuring wall-clock with `Instant::elapsed()`. The latter conflates
/// scan_family's actual runtime with tokio runtime teardown overhead
/// (reqwest's hyper-pool background tasks linger for ~2-3s after the
/// future resolves), making wall-clock assertions flake in a way that
/// has nothing to do with whether the cap fired.
///
/// The contract here is: with cap=150ms set, scan_family completes
/// well under the 10-second outer timeout. With the cap removed, the
/// wiremock setup paginates forever and the outer timeout fires —
/// failing the test with a clear "cap not enforced" signal.
#[tokio::test]
async fn scan_family_honours_max_family_wallclock_cap() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/models/test/runaway"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/test/runaway/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/test/finetune-1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(include_str!("fixtures/hf_flux_dev.json")),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models/test/finetune-1/tree/main"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(include_str!("fixtures/hf_flux_dev_tree.json")),
        )
        .mount(&server)
        .await;
    // Every list request returns a non-empty page — without the cap the
    // scanner would walk forever.
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("filter", "base_model:test/runaway"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"[{"id":"test/finetune-1"}]"#))
        .mount(&server)
        .await;

    let opts = ScanOptions {
        // Slow each request via the throttle so the cap has something
        // to bound; with zero delay the pagination would burn through
        // millions of pages before any timer arithmetic could matter.
        hf_request_delay: std::time::Duration::from_millis(20),
        civitai_request_delay: std::time::Duration::ZERO,
        max_429_retries: 0,
        max_family_wallclock: Some(std::time::Duration::from_millis(150)),
        ..ScanOptions::default()
    };

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        hf::scan_family(&server.uri(), &opts, Family::Flux, &["test/runaway"], None),
    )
    .await
    .expect("scan_family did not return within 10s — max_family_wallclock cap not enforced");

    // Bailing on the cap is a graceful exit (Ok with partial entries),
    // not an error condition.
    result.expect("scan should return Ok with partial entries on cap, not error");
}
