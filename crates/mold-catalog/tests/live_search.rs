//! Live catalog search — wiremock-driven coverage for the in-process
//! proxy that replaces the bulk-scrape model on the read path.
//!
//! These tests pin three load-bearing behaviours:
//!
//! 1. `search()` normalizes Civitai responses through `from_civitai` so
//!    the wire shape is identical to the existing `/api/catalog` rows.
//! 2. The TTL cache short-circuits a repeated query — the mock server
//!    asserts upstream is hit only once across two `search()` calls.
//! 3. `family=` forwards as Civitai `baseModels=` query params.

use std::time::Duration;

use mold_catalog::entry::{Kind, Source};
use mold_catalog::families::Family;
use mold_catalog::live::{
    family_from_hf, fetch_civitai_version, fetch_hf_repo, search, LiveCache, LiveSearchOpts,
};
use wiremock::matchers::{method, path, path_regex, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ONE_FLUX_LORA: &str = r#"{
    "items": [{
        "id": 9001,
        "name": "Test Flux LoRA",
        "type": "LORA",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 12345, "rating": 4.7, "favoriteCount": 88 },
        "tags": [],
        "modelVersions": [{
            "id": 8001,
            "name": "v1",
            "baseModel": "Flux.1 D",
            "baseModelType": "Standard",
            "trainedWords": ["mold trigger"],
            "files": [{
                "id": 1,
                "name": "test.safetensors",
                "sizeKB": 100000,
                "downloadCount": 1,
                "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/x.safetensors",
                "hashes": { "SHA256": "deadbeef" }
            }],
            "images": [{ "url": "https://civitai.example/preview.png", "nsfwLevel": 1 }]
        }]
    }],
    "metadata": { "totalPages": 1 }
}"#;

fn flux_lora_opts(q: &str) -> LiveSearchOpts {
    LiveSearchOpts {
        q: Some(q.into()),
        family: Some(Family::Flux),
        kind: Some(Kind::Lora),
        source: Some(Source::Civitai),
        page: 1,
        page_size: 20,
        include_nsfw: false,
        civitai_token: None,
        hf_token: None,
    }
}

#[tokio::test]
async fn civitai_search_returns_normalized_entries() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("query", "test"))
        .and(query_param("types", "LORA"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let entries = search(
        &server.uri(),
        "https://hf.unused",
        &cache,
        &flux_lora_opts("test"),
    )
    .await
    .expect("live search");

    assert_eq!(entries.len(), 1, "one normalized civitai LoRA expected");
    let row = &entries[0];
    assert_eq!(row.id.0, "cv:8001");
    assert_eq!(row.kind, Kind::Lora);
    assert_eq!(row.family, Family::Flux);
    assert_eq!(row.trained_words, vec!["mold trigger".to_string()]);
}

#[tokio::test]
async fn cache_short_circuits_repeat_query() {
    let server = MockServer::start().await;
    // expect(1) is the load-bearing assertion: a second `search()` call
    // with the same opts MUST be served from the in-process cache so we
    // don't hammer Civitai with duplicate searches every time the SPA
    // re-renders or re-mounts.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let opts = flux_lora_opts("cache-me");

    let first = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .unwrap();
    let second = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .unwrap();

    assert_eq!(first.len(), second.len());
    assert_eq!(first[0].id.0, second[0].id.0);
}

#[tokio::test]
async fn family_filter_forwards_as_civitai_base_models_when_no_query() {
    let server = MockServer::start().await;
    // The full FLUX bucket contains 4 baseModel strings (Flux.1 S/D/Krea/Kontext);
    // we assert that at least the load-bearing two are in the URL — the
    // full set is enforced by `civitai_map::CIVITAI_BASE_MODELS` and the
    // completeness test there.
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "Flux.1 D"))
        .and(query_param("baseModels", "Flux.1 S"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("ignored");
    opts.q = None;
    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .unwrap();
    assert_eq!(entries.len(), 1);
}

/// Civitai bug: `query=` + `baseModels=` returns empty first-cursor
/// windows. We drop the upstream baseModels filter when a query is
/// present and rely on local family filtering instead. This test pins
/// that contract — without it the LoRA picker's search box returns
/// zero results in production.
#[tokio::test]
async fn base_models_dropped_when_query_present() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("query", "anime"))
        .and(wiremock::matchers::query_param_is_missing("baseModels"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let entries = search(
        &server.uri(),
        "https://hf.unused",
        &cache,
        &flux_lora_opts("anime"),
    )
    .await
    .unwrap();
    // Local family filter still applies; the fixture row IS Flux.1 D
    // so it survives.
    assert_eq!(entries.len(), 1);
}

#[tokio::test]
async fn nsfw_filtered_out_when_include_nsfw_false() {
    let server = MockServer::start().await;
    let body = r#"{
        "items": [
            {
                "id": 1, "name": "SFW", "type": "Checkpoint", "nsfw": false,
                "creator": { "username": "a" },
                "stats": { "downloadCount": 1, "favoriteCount": 0 },
                "tags": [],
                "modelVersions": [{
                    "id": 11, "name": "v1",
                    "baseModel": "Flux.1 D", "baseModelType": "Standard",
                    "files": [{
                        "id": 1, "name": "x.safetensors", "sizeKB": 100,
                        "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                        "downloadUrl": "u", "hashes": {}
                    }],
                    "images": []
                }]
            },
            {
                "id": 2, "name": "NSFW", "type": "Checkpoint", "nsfw": true,
                "creator": { "username": "a" },
                "stats": { "downloadCount": 1, "favoriteCount": 0 },
                "tags": [],
                "modelVersions": [{
                    "id": 22, "name": "v1",
                    "baseModel": "Flux.1 D", "baseModelType": "Standard",
                    "files": [{
                        "id": 2, "name": "y.safetensors", "sizeKB": 100,
                        "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                        "downloadUrl": "u", "hashes": {}
                    }],
                    "images": []
                }]
            }
        ],
        "metadata": { "totalPages": 1 }
    }"#;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("any");
    opts.kind = None;
    opts.include_nsfw = false;
    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .unwrap();

    assert_eq!(
        entries.len(),
        1,
        "nsfw row must be filtered when opt-in is off"
    );
    assert_eq!(entries[0].name, "SFW");
}

/// Regression for the 400 "Cannot use page param with query search.
/// Use cursor-based pagination." Civitai rejects `page=` whenever
/// `query=` is also present. Pin the URL builder so a future "always
/// send page" refactor doesn't silently break live LoRA search.
#[tokio::test]
async fn page_param_omitted_when_query_present() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("query", "anime"))
        .and(wiremock::matchers::query_param_is_missing("page"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("anime");
    opts.page = 2;
    let _ = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .unwrap();
}

#[tokio::test]
async fn page_param_present_when_query_absent() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("page", "2"))
        .and(wiremock::matchers::query_param_is_missing("query"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("ignored");
    opts.q = None;
    opts.page = 2;
    let _ = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .unwrap();
}

// ── Single-id Civitai ──────────────────────────────────────────────────────

const CV_VERSION_DETAIL: &str = r#"{
    "id": 8001,
    "name": "v1",
    "baseModel": "Flux.1 D",
    "baseModelType": "Standard",
    "trainedWords": ["mold trigger"],
    "files": [{
        "id": 1,
        "name": "test.safetensors",
        "sizeKB": 100000,
        "downloadCount": 1,
        "metadata": { "format": "SafeTensor" },
        "downloadUrl": "https://civitai.example/x.safetensors",
        "hashes": { "SHA256": "deadbeef" }
    }],
    "images": [{ "url": "https://civitai.example/preview.png" }],
    "model": {
        "name": "Test Flux LoRA",
        "type": "LORA",
        "nsfw": false,
        "tags": [],
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 12345, "favoriteCount": 88 }
    }
}"#;

#[tokio::test]
async fn fetch_civitai_version_normalizes_single_id() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/model-versions/8001"))
        .respond_with(ResponseTemplate::new(200).set_body_string(CV_VERSION_DETAIL))
        .expect(1)
        .mount(&server)
        .await;

    let entry = fetch_civitai_version(&server.uri(), "8001", None)
        .await
        .expect("single-id fetch");
    assert_eq!(entry.id.0, "cv:8001");
    assert_eq!(entry.kind, Kind::Lora);
    assert_eq!(entry.family, Family::Flux);
    assert_eq!(entry.trained_words, vec!["mold trigger".to_string()]);
    // Recipe carries one safetensors file with the right URL.
    assert_eq!(entry.download_recipe.files.len(), 1);
    assert_eq!(
        entry.download_recipe.files[0].url,
        "https://civitai.example/x.safetensors"
    );
}

#[tokio::test]
async fn fetch_civitai_version_surfaces_404() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/model-versions/9999999"))
        .respond_with(ResponseTemplate::new(404).set_body_string("{\"error\":\"not found\"}"))
        .mount(&server)
        .await;

    let err = fetch_civitai_version(&server.uri(), "9999999", None)
        .await
        .expect_err("404 should surface");
    let msg = format!("{err}");
    assert!(msg.contains("404"), "expected 404 in error: {msg}");
}

// ── HF live search ──────────────────────────────────────────────────────────

const HF_SEARCH_HITS: &str = r#"[
    {
        "id": "black-forest-labs/FLUX.1-dev",
        "author": "black-forest-labs",
        "downloads": 42000,
        "likes": 100,
        "tags": ["text-to-image"],
        "pipeline_tag": "text-to-image"
    },
    {
        "id": "some-org/random-llm",
        "author": "some-org",
        "downloads": 1,
        "likes": 0,
        "tags": ["text-generation"],
        "pipeline_tag": "text-generation"
    }
]"#;

#[tokio::test]
async fn hf_search_drops_non_mold_repos() {
    // The HF search endpoint returns a flat list. Repos that don't
    // map to a supported mold family must be dropped, not surfaced
    // with engine_phase=99.
    let civ = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"items\":[]}"))
        .mount(&civ)
        .await;
    let hf = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(HF_SEARCH_HITS))
        .mount(&hf)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let opts = LiveSearchOpts {
        q: Some("flux".into()),
        family: None,
        kind: None,
        source: Some(Source::Hf),
        page: 1,
        page_size: 20,
        include_nsfw: false,
        civitai_token: None,
        hf_token: None,
    };
    let entries = search(&civ.uri(), &hf.uri(), &cache, &opts)
        .await
        .expect("hf search");

    assert_eq!(entries.len(), 1, "non-mold repo must be dropped");
    assert_eq!(entries[0].id.0, "hf:black-forest-labs/FLUX.1-dev");
    assert_eq!(entries[0].family, Family::Flux);
    // Recipe is empty by design — filled in lazily at download time.
    assert!(entries[0].download_recipe.files.is_empty());
}

#[tokio::test]
async fn hf_search_failure_does_not_nuke_civitai_results() {
    // When source=None we want both backends, but a flaky HF must
    // not turn a successful Civitai response into an error.
    let civ = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .mount(&civ)
        .await;
    let hf = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(503).set_body_string("hf is sad"))
        .mount(&hf)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("test");
    opts.source = None;
    let entries = search(&civ.uri(), &hf.uri(), &cache, &opts)
        .await
        .expect("civitai must still surface");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].source, Source::Civitai);
}

// ── Family inference ───────────────────────────────────────────────────────

#[test]
fn family_from_hf_id_substring() {
    let cases: &[(&str, Family)] = &[
        ("black-forest-labs/FLUX.1-dev", Family::Flux),
        ("black-forest-labs/FLUX.2-dev", Family::Flux2),
        ("Lightricks/LTX-Video-2.3", Family::Ltx2),
        ("Lightricks/LTX-Video", Family::LtxVideo),
        ("stabilityai/stable-diffusion-xl-base-1.0", Family::Sdxl),
        ("Tongyi-MAI/Z-Image-Turbo", Family::ZImage),
    ];
    for (id, want) in cases {
        let (got, _role) =
            family_from_hf(id, &[], None).unwrap_or_else(|| panic!("expected family for {id}"));
        assert_eq!(got, *want, "id {id}");
    }
    assert!(
        family_from_hf("some-org/llama-3-instruct", &[], None).is_none(),
        "non-mold repos must return None"
    );
}

#[test]
fn family_from_hf_marks_seed_repos_as_foundation() {
    use mold_catalog::entry::FamilyRole;
    let (_, role) = family_from_hf("black-forest-labs/FLUX.1-dev", &[], None).unwrap();
    assert_eq!(role, FamilyRole::Foundation);
    let (_, role) = family_from_hf("some-fork/FLUX.1-dev-fp8", &[], None).unwrap();
    assert_eq!(role, FamilyRole::Finetune);
}

// ── Single-id HF ───────────────────────────────────────────────────────────

const HF_DETAIL: &str = r#"{
    "id": "black-forest-labs/FLUX.1-dev",
    "author": "black-forest-labs",
    "downloads": 42000,
    "likes": 100,
    "tags": ["text-to-image"],
    "pipeline_tag": "text-to-image",
    "createdAt": "2024-08-01T12:00:00.000Z",
    "lastModified": "2024-09-01T12:00:00.000Z"
}"#;

const HF_TREE: &str = r#"[
    { "type": "file", "path": "model_index.json", "size": 100 },
    { "type": "file", "path": "transformer/diffusion_pytorch_model.safetensors", "size": 1000000 }
]"#;

#[tokio::test]
async fn fetch_hf_repo_returns_full_recipe() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models/black-forest-labs/FLUX.1-dev"))
        .respond_with(ResponseTemplate::new(200).set_body_string(HF_DETAIL))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path_regex(
            r"^/api/models/black-forest-labs/FLUX\.1-dev/tree/main$",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(HF_TREE))
        .mount(&server)
        .await;

    let entry = fetch_hf_repo(&server.uri(), "black-forest-labs/FLUX.1-dev", None)
        .await
        .expect("hf single-id fetch");
    assert_eq!(entry.id.0, "hf:black-forest-labs/FLUX.1-dev");
    assert_eq!(entry.family, Family::Flux);
    assert!(
        !entry.download_recipe.files.is_empty(),
        "tree must populate the recipe"
    );
}

#[tokio::test]
async fn cache_evicts_when_max_entries_exceeded() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .mount(&server)
        .await;

    // Tiny cap so the third unique key must evict the first.
    let cache = LiveCache::new(Duration::from_secs(300), 2);

    for q in ["one", "two", "three"] {
        let mut opts = flux_lora_opts(q);
        opts.kind = None;
        let _ = search(&server.uri(), "https://hf.unused", &cache, &opts)
            .await
            .unwrap();
    }
    assert!(
        cache.len() <= 2,
        "cache must respect max_entries cap; got {}",
        cache.len()
    );
}
