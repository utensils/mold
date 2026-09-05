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
    family_from_hf, fetch_civitai_version, fetch_hf_repo, search, CatalogSort, LiveCache,
    LiveSearchError, LiveSearchOpts,
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

const ONE_CONTROLNET: &str = r#"{
    "items": [{
        "id": 9101,
        "name": "Test SDXL ControlNet",
        "type": "Controlnet",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 99, "rating": 4.5, "favoriteCount": 7 },
        "tags": [],
        "modelVersions": [{
            "id": 8101,
            "name": "v1",
            "baseModel": "SDXL 1.0",
            "baseModelType": "Standard",
            "trainedWords": [],
            "files": [{
                "id": 1,
                "name": "controlnet.safetensors",
                "sizeKB": 100000,
                "downloadCount": 1,
                "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/controlnet.safetensors",
                "hashes": { "SHA256": "feedface" }
            }],
            "images": []
        }]
    }],
    "metadata": { "totalPages": 1 }
}"#;

const ONE_QWEN_CHECKPOINT: &str = r#"{
    "items": [{
        "id": 9201,
        "name": "Test Qwen Checkpoint",
        "type": "Checkpoint",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 101, "rating": 4.8, "favoriteCount": 11 },
        "tags": [],
        "modelVersions": [{
            "id": 8201,
            "name": "fp8",
            "baseModel": "Qwen",
            "baseModelType": "Standard",
            "trainedWords": [],
            "files": [{
                "id": 1,
                "name": "qwenImage_fp8.safetensors",
                "type": "Model",
                "sizeKB": 19951792,
                "downloadCount": 1,
                "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/qwen.safetensors",
                "hashes": { "SHA256": "abc123" }
            }],
            "images": []
        }]
    }],
    "metadata": { "totalPages": 1 }
}"#;

const ONE_QWEN_EDIT_CHECKPOINT: &str = r#"{
    "items": [{
        "id": 9202,
        "name": "Community Qwen Image Edit",
        "type": "Checkpoint",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 102, "rating": 4.8, "favoriteCount": 12 },
        "tags": [],
        "modelVersions": [{
            "id": 8202,
            "name": "QWEN_IMAGE_EDIT fp8",
            "baseModel": "Qwen",
            "baseModelType": "Standard",
            "trainedWords": [],
            "files": [{
                "id": 1,
                "name": "qwenImageEdit_fp8.safetensors",
                "type": "Model",
                "sizeKB": 19951792,
                "downloadCount": 1,
                "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/qwen-edit.safetensors",
                "hashes": { "SHA256": "def456" }
            }],
            "images": []
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
        sort: CatalogSort::Downloads,
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
async fn civitai_controlnet_search_uses_controlnet_type_filter() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("query", "control"))
        .and(query_param("types", "Controlnet"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_CONTROLNET))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("control");
    opts.family = Some(Family::Sdxl);
    opts.kind = Some(Kind::ControlNet);

    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .expect("controlnet live search");

    assert_eq!(entries.len(), 1, "one normalized ControlNet expected");
    assert_eq!(entries[0].kind, Kind::ControlNet);
    assert_eq!(entries[0].family, Family::Sdxl);
    assert!(entries[0].supported);
}

#[tokio::test]
async fn civitai_qwen_checkpoint_is_installable_with_runtime_companion() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("query", "qwen"))
        .and(query_param("types", "Checkpoint"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_QWEN_CHECKPOINT))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("qwen");
    opts.family = Some(Family::QwenImage);
    opts.kind = Some(Kind::Checkpoint);

    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .expect("qwen checkpoint live search");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].kind, Kind::Checkpoint);
    assert_eq!(entries[0].family, Family::QwenImage);
    assert!(entries[0].supported);
    assert_eq!(entries[0].companions, vec!["qwen-image-runtime"]);
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

#[tokio::test]
async fn qwen_edit_family_reuses_generic_civitai_bucket_then_filters_locally() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "Qwen"))
        .and(query_param("baseModels", "Qwen 2"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_QWEN_EDIT_CHECKPOINT))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let opts = LiveSearchOpts {
        q: None,
        family: Some(Family::QwenImageEdit),
        kind: Some(Kind::Checkpoint),
        source: Some(Source::Civitai),
        page: 1,
        page_size: 20,
        include_nsfw: false,
        sort: CatalogSort::Downloads,
        civitai_token: None,
        hf_token: None,
    };
    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .expect("qwen edit search");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].family, Family::QwenImageEdit);
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
    assert_eq!(entries[0].name, "SFW - v1");
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

/// Civitai ignores `page=` on browse (no-query) requests too — the
/// endpoint is cursor-paginated and any page value returns the same
/// first window, which made every "Load more" page identical. Pin that
/// the builder never sends `page=` and fetches one `page_size` window
/// per cursor step (never a widened `page × page_size` window, whose
/// 100-cap made rows past 100 unreachable).
#[tokio::test]
async fn page_param_never_sent_and_limit_is_one_page() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(wiremock::matchers::query_param_is_missing("page"))
        .and(query_param("limit", "20"))
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

// ── Sort mapping ────────────────────────────────────────────────────────────

/// Second Flux LoRA fixture (distinct ids) so sort-keyed cache tests can
/// tell which upstream response produced a result set.
const NEWEST_FLUX_LORA: &str = r#"{
    "items": [{
        "id": 9002,
        "name": "Fresh Flux LoRA",
        "type": "LORA",
        "nsfw": false,
        "creator": { "username": "bob" },
        "stats": { "downloadCount": 3, "rating": 4.0, "favoriteCount": 1 },
        "tags": [],
        "modelVersions": [{
            "id": 8002,
            "name": "v1",
            "baseModel": "Flux.1 D",
            "baseModelType": "Standard",
            "trainedWords": [],
            "files": [{
                "id": 1,
                "name": "fresh.safetensors",
                "sizeKB": 100000,
                "downloadCount": 1,
                "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/fresh.safetensors",
                "hashes": { "SHA256": "cafebabe" }
            }],
            "images": []
        }]
    }],
    "metadata": { "totalPages": 1 }
}"#;

#[tokio::test]
async fn civitai_sort_defaults_to_most_downloaded() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("sort", "Most Downloaded"))
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
    .expect("default sort search");
    assert_eq!(entries.len(), 1);
}

#[tokio::test]
async fn civitai_sort_recent_maps_to_newest() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("sort", "Newest"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("test");
    opts.sort = CatalogSort::Recent;
    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .expect("recent sort search");
    assert_eq!(entries.len(), 1);
}

#[tokio::test]
async fn civitai_sort_rating_maps_to_highest_rated() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("sort", "Highest Rated"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("test");
    opts.sort = CatalogSort::Rating;
    let entries = search(&server.uri(), "https://hf.unused", &cache, &opts)
        .await
        .expect("rating sort search");
    assert_eq!(entries.len(), 1);
}

fn hf_only_opts(sort: CatalogSort) -> LiveSearchOpts {
    LiveSearchOpts {
        q: Some("flux".into()),
        family: None,
        kind: None,
        source: Some(Source::Hf),
        page: 1,
        page_size: 20,
        include_nsfw: false,
        sort,
        civitai_token: None,
        hf_token: None,
    }
}

#[tokio::test]
async fn hf_sort_defaults_to_downloads_descending() {
    let hf = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("sort", "downloads"))
        .and(query_param("direction", "-1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(HF_SEARCH_HITS))
        .expect(1)
        .mount(&hf)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let entries = search(
        "https://civitai.unused",
        &hf.uri(),
        &cache,
        &hf_only_opts(CatalogSort::Downloads),
    )
    .await
    .expect("hf default sort search");
    assert_eq!(entries.len(), 1);
}

#[tokio::test]
async fn hf_sort_recent_maps_to_last_modified_descending() {
    let hf = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("sort", "lastModified"))
        .and(query_param("direction", "-1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(HF_SEARCH_HITS))
        .expect(1)
        .mount(&hf)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let entries = search(
        "https://civitai.unused",
        &hf.uri(),
        &cache,
        &hf_only_opts(CatalogSort::Recent),
    )
    .await
    .expect("hf recent sort search");
    assert_eq!(entries.len(), 1);
}

#[tokio::test]
async fn hf_sort_rating_maps_to_likes_descending() {
    let hf = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("sort", "likes"))
        .and(query_param("direction", "-1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(HF_SEARCH_HITS))
        .expect(1)
        .mount(&hf)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let entries = search(
        "https://civitai.unused",
        &hf.uri(),
        &cache,
        &hf_only_opts(CatalogSort::Rating),
    )
    .await
    .expect("hf rating sort search");
    assert_eq!(entries.len(), 1);
}

/// Regression: the 5-minute cache must key on the sort. Two searches
/// identical except for `sort` must each hit upstream (expect(1) per
/// mock) and must return their own result set — before sort was part of
/// [`LiveSearchOpts`] the second sort would silently replay the first
/// sort's cached rows.
#[tokio::test]
async fn cache_keys_on_sort_so_sorts_do_not_cross_contaminate() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("sort", "Most Downloaded"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("sort", "Newest"))
        .respond_with(ResponseTemplate::new(200).set_body_string(NEWEST_FLUX_LORA))
        .expect(1)
        .mount(&server)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let downloads_opts = flux_lora_opts("sorted");
    let mut recent_opts = flux_lora_opts("sorted");
    recent_opts.sort = CatalogSort::Recent;

    let by_downloads = search(&server.uri(), "https://hf.unused", &cache, &downloads_opts)
        .await
        .expect("downloads search");
    let by_recent = search(&server.uri(), "https://hf.unused", &cache, &recent_opts)
        .await
        .expect("recent search");
    assert_eq!(by_downloads[0].id.0, "cv:8001");
    assert_eq!(
        by_recent[0].id.0, "cv:8002",
        "recent sort must not replay the downloads-sorted cache entry"
    );

    // Repeat both — the expect(1) on each mock proves these are cache
    // hits, and each sort keeps its own rows.
    let by_downloads_again = search(&server.uri(), "https://hf.unused", &cache, &downloads_opts)
        .await
        .expect("cached downloads search");
    let by_recent_again = search(&server.uri(), "https://hf.unused", &cache, &recent_opts)
        .await
        .expect("cached recent search");
    assert_eq!(by_downloads_again[0].id.0, "cv:8001");
    assert_eq!(by_recent_again[0].id.0, "cv:8002");
}

// ── Single-id Civitai ──────────────────────────────────────────────────────

const CV_VERSION_DETAIL: &str = r#"{
    "id": 8001,
    "name": "v1",
    "description": "<p>Portrait adapter &amp; lighting guide.</p>",
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
    assert_eq!(
        entry.description.as_deref(),
        Some("Portrait adapter & lighting guide.")
    );
    // Recipe carries one safetensors file with the right URL.
    assert_eq!(entry.download_recipe.files.len(), 1);
    assert_eq!(
        entry.download_recipe.files[0].url,
        "https://civitai.example/x.safetensors?format=SafeTensor"
    );
    // The model-version detail body above has no `modelId`, so the model
    // page can't be composed — the entry must stay usable with no link.
    assert_eq!(entry.page_url, None);
}

#[tokio::test]
async fn fetch_civitai_version_builds_page_url_from_model_id() {
    // Same shape as CV_VERSION_DETAIL but with the top-level `modelId`
    // Civitai actually sends — that's what lets a single-version fetch
    // link back to the parent model page.
    let body = CV_VERSION_DETAIL.replacen(
        "{\n    \"id\": 8001,",
        "{\n    \"modelId\": 9001,\n    \"id\": 8001,",
        1,
    );
    assert!(body.contains("modelId"), "fixture rewrite must apply");

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/model-versions/8001"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;

    let entry = fetch_civitai_version(&server.uri(), "8001", None)
        .await
        .expect("single-id fetch");
    assert_eq!(entry.id.0, "cv:8001");
    assert_eq!(
        entry.page_url.as_deref(),
        Some("https://civitai.com/models/9001?modelVersionId=8001")
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
    // as fake unsupported catalog rows.
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
        sort: CatalogSort::Downloads,
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
    // Search-summary rows still know their repo page.
    assert_eq!(
        entries[0].page_url.as_deref(),
        Some("https://huggingface.co/black-forest-labs/FLUX.1-dev")
    );
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

#[tokio::test]
async fn hf_auth_failure_is_returned_so_the_server_can_retry_a_fallback_token() {
    let civ = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ONE_FLUX_LORA))
        .mount(&civ)
        .await;
    let hf = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid token"))
        .mount(&hf)
        .await;

    let cache = LiveCache::new(Duration::from_secs(300), 64);
    let mut opts = flux_lora_opts("test");
    opts.source = None;
    let error = search(&civ.uri(), &hf.uri(), &cache, &opts)
        .await
        .expect_err("auth failures must be available to the server fallback path");

    assert!(matches!(
        error,
        LiveSearchError::Upstream { status: 401, .. }
    ));
}

// ── Family inference ───────────────────────────────────────────────────────

#[test]
fn family_from_hf_id_substring() {
    let cases: &[(&str, Family)] = &[
        ("black-forest-labs/FLUX.1-dev", Family::Flux),
        ("black-forest-labs/FLUX.2-dev", Family::Flux2),
        ("Lightricks/LTX-Video-2.3", Family::Ltx2),
        ("Lightricks/LTX-2.3", Family::Ltx2),
        ("Lightricks/LTX-Video", Family::LtxVideo),
        ("MiniMaxAI/MiniMax-H3", Family::MinimaxH3),
        ("someone/MiniMax-H3-FL2VA-finetune", Family::MinimaxH3),
        ("stabilityai/stable-diffusion-xl-base-1.0", Family::Sdxl),
        ("stabilityai/stable-diffusion-3.5-large", Family::Sd3),
        ("city96/stable-diffusion-3.5-medium-gguf", Family::Sd3),
        ("Tongyi-MAI/Z-Image-Turbo", Family::ZImage),
        ("Qwen/Qwen-Image-Edit-2511", Family::QwenImageEdit),
        ("someone/QWEN_IMAGE_EDIT_finetune", Family::QwenImageEdit),
        ("someone/qwen-image-edit-lightning", Family::QwenImageEdit),
        ("someone/QwenImageEdit2511", Family::QwenImageEdit),
        ("Qwen/Qwen-Image", Family::QwenImage),
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
    for id in ["someone/notminimax-h3", "someone/minimax-h30"] {
        assert!(
            family_from_hf(id, &[], None).is_none(),
            "H3 lookalike {id} must not be classified"
        );
    }
}

#[test]
fn family_from_hf_uses_official_qwen_edit_metadata_without_overmatching() {
    use mold_catalog::entry::FamilyRole;

    let tags = vec!["diffusers:QwenImageEditPlusPipeline".to_string()];
    let (family, role) = family_from_hf("Qwen/Qwen-Image-2511", &tags, Some("image-to-image"))
        .expect("official Qwen edit metadata");
    assert_eq!(family, Family::QwenImageEdit);
    assert_eq!(role, FamilyRole::Finetune);

    let (_, role) = family_from_hf("Qwen/Qwen-Image-Edit-2511", &tags, Some("image-to-image"))
        .expect("official seed");
    assert_eq!(role, FamilyRole::Foundation);

    for id in [
        "someone/qwen-image-editorial-style",
        "someone/qwen-image-editable-captioner",
        "someone/qwen-image",
    ] {
        let (family, _) = family_from_hf(id, &[], Some("image-to-image"))
            .unwrap_or_else(|| panic!("expected generic Qwen family for {id}"));
        assert_eq!(family, Family::QwenImage, "{id}");
    }

    assert!(
        family_from_hf(
            "someorg/instruct-image-edit-v2",
            &[],
            Some("image-to-image")
        )
        .is_none(),
        "a generic image-edit repository has no Qwen context"
    );
}

#[test]
fn family_from_hf_recognizes_sd3_metadata_without_misclassifying_sd15() {
    for tags in [
        vec!["diffusers:StableDiffusion3Pipeline".to_string()],
        vec!["base_model:stabilityai/stable-diffusion-3.5-large".to_string()],
    ] {
        let (family, _) = family_from_hf("someone/opaque-image-model", &tags, None).unwrap();
        assert_eq!(family, Family::Sd3);
    }

    let (family, _) = family_from_hf(
        "stabilityai/stable-diffusion-v1-5",
        &["stable-diffusion".to_string()],
        None,
    )
    .unwrap();
    assert_eq!(family, Family::Sd15);
}

/// Every spelling of Wan that the ecosystem actually ships must land on the
/// family, and — the part that needs care — nothing else may.
///
/// `wan` is three letters that occur inside ordinary words. The `flux` arm can
/// afford an owner-prefix match (`/flux`) because "flux" rarely prefixes an
/// unrelated model name; `/wan` would swallow `wanderlust`, `wan-derby`, and
/// anything from an org whose name starts with those letters, then hand the
/// row a frame count and route it to a video pipeline it cannot run.
#[test]
fn family_from_hf_recognizes_every_wan_spelling_without_swallowing_words() {
    for id in [
        "Wan-AI/Wan2.1-T2V-1.3B",
        "Wan-AI/Wan2.2-T2V-A14B",
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "city96/Wan2.1-T2V-14B-gguf",
        "Kijai/WanVideo_comfy",
        "lightx2v/Wan2.2-Distill-Loras",
        "someone/wan-2.2-finetune",
    ] {
        let (got, _role) =
            family_from_hf(id, &[], None).unwrap_or_else(|| panic!("expected a family for {id}"));
        assert_eq!(got, Family::Wan, "id {id}");
    }

    // The collision cases. None of these is a Wan model.
    for id in [
        "someone/wanderlust-xl",
        "someone/wandering-diffusion",
        "taiwan-labs/some-sdxl-merge",
        "wanda/pruning-research",
    ] {
        assert!(
            family_from_hf(id, &[], None)
                != Some((Family::Wan, mold_catalog::entry::FamilyRole::Finetune))
                && family_from_hf(id, &[], None)
                    != Some((Family::Wan, mold_catalog::entry::FamilyRole::Foundation)),
            "{id} must not be classified as Wan"
        );
    }

    // An explicit `wan` tag is a claim, not a coincidence, so it is honoured
    // even when the id says nothing.
    let (got, _) = family_from_hf("some-org/mystery-video-model", &["wan".to_string()], None)
        .expect("tag fallback");
    assert_eq!(got, Family::Wan);
}

/// TI2V-5B is the one Wan checkpoint whose VAE and timing differ from the
/// rest of the family, and HF — where `sub_family` is otherwise always `None`
/// — is where it is published. Without this inference a 5B row is handed the
/// 16-channel 2.1 VAE, which its 48-channel DiT rejects at load, after the
/// download has already run.
#[test]
fn wan_sub_family_is_inferred_for_the_one_variant_that_needs_it() {
    use mold_catalog::live::wan_sub_family_from_id;

    for id in [
        "Wan-AI/Wan2.2-TI2V-5B",
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged-ti2v_5b",
        "someone/wan2.2-TI2V-5B-gguf",
    ] {
        assert_eq!(
            wan_sub_family_from_id(id).as_deref(),
            Some("wan22-ti2v-5b"),
            "id {id}"
        );
    }

    // Everything else keeps `None` and takes the family default, which is
    // already correct for it — guessing further would add risk for no gain.
    for id in [
        "Wan-AI/Wan2.1-T2V-1.3B",
        "Wan-AI/Wan2.2-T2V-A14B",
        "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "black-forest-labs/FLUX.1-dev",
    ] {
        assert_eq!(wan_sub_family_from_id(id), None, "id {id}");
    }
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
    // `len()` sums the per-source maps (Civitai pages, HF pages, cursor
    // chains) — each individually respects the max_entries cap. Three
    // unique Civitai queries populate a page and a chain apiece, so with
    // a cap of 2 the third must evict the first in both maps.
    assert!(
        cache.len() <= 4,
        "each per-source map must respect the max_entries cap; got {}",
        cache.len()
    );
}
