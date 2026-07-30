//! Integration tests for the live-search and installed catalog routes.
//!
//! `live_search_catalog` is exercised against a wiremock-backed Civitai;
//! `list_installed_catalog` is exercised against tempdir sidecar
//! fixtures. Both go through the real axum router so the route wiring
//! and JSON shape are tested end-to-end.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mold_catalog::sidecar::{sidecar_from_entry, write_sidecar, CatalogSidecar, SIDECAR_FILENAME};

#[test]
fn forwarded_catalog_credentials_are_trimmed_and_read_from_headers() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("x-mold-hf-token", "  hf_desktop  ".parse().unwrap());
    headers.insert("x-mold-civitai-token", "cv_desktop".parse().unwrap());

    let credentials = super::ForwardedCatalogCredentials::from_headers(&headers);

    assert_eq!(credentials.hf.as_deref(), Some("hf_desktop"));
    assert_eq!(credentials.civitai.as_deref(), Some("cv_desktop"));
}

#[test]
fn server_catalog_credentials_are_attempted_before_forwarded_fallbacks() {
    assert_eq!(
        super::credential_candidates(Some("server".into()), Some("desktop")),
        vec![Some("server".into()), Some("desktop".into())]
    );
    assert_eq!(
        super::credential_candidates(None, Some("desktop")),
        vec![Some("desktop".into())]
    );
}

#[test]
fn combined_search_replaces_each_rejected_server_credential() {
    let forwarded = super::ForwardedCatalogCredentials {
        hf: Some("hf_desktop".into()),
        civitai: Some("cv_desktop".into()),
    };
    let mut opts = mold_catalog::live::LiveSearchOpts {
        civitai_token: Some("cv_server".into()),
        hf_token: Some("hf_server".into()),
        ..Default::default()
    };
    let civitai_error = mold_catalog::live::LiveSearchError::Upstream {
        host: "civitai.com",
        status: 401,
        body: "stale".into(),
    };
    let hf_error = mold_catalog::live::LiveSearchError::Upstream {
        host: "huggingface.co",
        status: 403,
        body: "stale".into(),
    };

    assert!(super::replace_failed_search_credential(
        &civitai_error,
        &mut opts,
        &forwarded,
    ));
    assert_eq!(opts.civitai_token.as_deref(), Some("cv_desktop"));
    assert!(super::replace_failed_search_credential(
        &hf_error, &mut opts, &forwarded,
    ));
    assert_eq!(opts.hf_token.as_deref(), Some("hf_desktop"));
    assert!(super::replace_failed_search_credential(
        &hf_error, &mut opts, &forwarded,
    ));
    assert_eq!(opts.hf_token, None);
    assert!(!super::replace_failed_search_credential(
        &hf_error, &mut opts, &forwarded,
    ));
}

#[test]
fn combined_search_retries_without_auth_when_the_server_credential_is_rejected() {
    let forwarded = super::ForwardedCatalogCredentials::default();
    let mut opts = mold_catalog::live::LiveSearchOpts {
        hf_token: Some("hf_server".into()),
        ..Default::default()
    };
    let error = mold_catalog::live::LiveSearchError::Upstream {
        host: "huggingface.co",
        status: 401,
        body: "stale".into(),
    };

    assert!(super::replace_failed_search_credential(
        &error, &mut opts, &forwarded,
    ));
    assert_eq!(opts.hf_token, None);
    assert!(!super::replace_failed_search_credential(
        &error, &mut opts, &forwarded,
    ));
}
use tower::ServiceExt;
use wiremock::matchers::{method, path as wm_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::routes::create_router;
use crate::state::AppState;

const FLUX_LORA_FIXTURE: &str = r#"{
    "items": [{
        "id": 9001,
        "name": "Live Flux LoRA",
        "type": "LORA",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 4242, "rating": 4.6, "favoriteCount": 1 },
        "tags": [],
        "modelVersions": [{
            "id": 8001,
            "name": "v1",
            "baseModel": "Flux.1 D",
            "baseModelType": "Standard",
            "trainedWords": ["live trigger"],
            "files": [{
                "id": 1, "name": "x.safetensors", "sizeKB": 100,
                "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/x.safetensors",
                "hashes": { "SHA256": "deadbeef" }
            }],
            "images": []
        }]
    }],
    "metadata": { "totalPages": 37, "totalItems": 37, "currentPage": 1 }
}"#;

async fn build_state() -> (AppState, MockServer, tempfile::TempDir) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(FLUX_LORA_FIXTURE))
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let mut state = AppState::for_tests().with_civitai_base(server.uri());
    {
        let mut cfg = state.config.write().await;
        cfg.models_dir = tmp.path().to_string_lossy().into_owned();
    }
    // Wire the SPA build-state so the router constructs cleanly even
    // without an embedded web bundle. (`build_state` here just sets up
    // the catalog-only fields the test needs.)
    let _ = &mut state;
    (state, server, tmp)
}

async fn get(router: axum::Router, uri: &str) -> (StatusCode, String) {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn post(router: axum::Router, uri: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn hf_checkpoint(source_id: &str) -> mold_catalog::entry::CatalogEntry {
    use mold_catalog::entry::{
        Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind,
        LicenseFlags, Modality, Source,
    };

    CatalogEntry {
        id: CatalogId::from(format!("hf:{source_id}")),
        source: Source::Hf,
        source_id: source_id.into(),
        name: source_id.into(),
        author: source_id.split_once('/').map(|(author, _)| author.into()),
        family: mold_catalog::families::Family::Ltx2,
        family_role: FamilyRole::Foundation,
        sub_family: None,
        modality: Modality::Video,
        kind: Kind::Checkpoint,
        file_format: FileFormat::Safetensors,
        bundling: Bundling::Separated,
        size_bytes: None,
        download_count: 0,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: DownloadRecipe {
            files: vec![],
            needs_token: None,
        },
        supported: true,
        created_at: None,
        updated_at: None,
        added_at: 0,
        trained_words: vec![],
        page_url: None,
    }
}

#[test]
fn aggregate_hugging_face_repositories_are_not_offered_for_download() {
    let entry = hf_checkpoint("Lightricks/LTX-2.3");
    let tmp = tempfile::tempdir().expect("tempdir");

    assert!(!super::entry_download_supported(&entry));
    assert_eq!(
        super::live_entry_to_wire(&entry, tmp.path())["supported"],
        false
    );
}

#[test]
fn known_transformer_repositories_are_offered_as_builtin_downloads() {
    let entry = hf_checkpoint("black-forest-labs/FLUX.1-dev");

    assert!(super::entry_download_supported(&entry));
}

#[test]
fn named_ltx23_bf16_manifests_remain_downloadable() {
    for model in ["ltx-2.3-22b-dev:bf16", "ltx-2.3-22b-distilled:bf16"] {
        let entry = hf_checkpoint(model);
        assert!(super::entry_download_supported(&entry), "{model}");
    }
}

#[tokio::test]
async fn live_search_returns_normalized_civitai_rows() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    let (status, body) = get(
        router,
        "/api/catalog/search?q=test&family=flux&kind=lora&source=civitai",
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "live search returned non-200: {body}"
    );

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    let entries = parsed["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], "cv:8001");
    assert_eq!(entries[0]["family"], "flux");
    assert_eq!(entries[0]["kind"], "lora");
    assert_eq!(
        entries[0]["installed"], false,
        "no sidecar yet → not installed"
    );
    assert_eq!(entries[0]["trained_words"][0], "live trigger");
    assert_eq!(
        entries[0]["page_url"], "https://civitai.com/models/9001?modelVersionId=8001",
        "wire rows carry the model page link"
    );
}

/// Two Flux LoRAs so page 2 with page_size 1 has a distinct row to
/// return. Served for every request regardless of paging params —
/// exactly how the real Civitai API behaves (`page=` is ignored;
/// pagination is cursor-only), which is why the proxy walks a cursor
/// chain and buffers leftover rows for the following pages.
const TWO_FLUX_LORA_FIXTURE: &str = r#"{
    "items": [{
        "id": 9001,
        "name": "Live Flux LoRA",
        "type": "LORA",
        "nsfw": false,
        "creator": { "username": "alice" },
        "stats": { "downloadCount": 4242, "rating": 4.6, "favoriteCount": 1 },
        "tags": [],
        "modelVersions": [{
            "id": 8001,
            "name": "v1",
            "baseModel": "Flux.1 D",
            "baseModelType": "Standard",
            "trainedWords": [],
            "files": [{
                "id": 1, "name": "x.safetensors", "sizeKB": 100,
                "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/x.safetensors",
                "hashes": { "SHA256": "deadbeef" }
            }],
            "images": []
        }]
    }, {
        "id": 9002,
        "name": "Second Flux LoRA",
        "type": "LORA",
        "nsfw": false,
        "creator": { "username": "bob" },
        "stats": { "downloadCount": 41, "rating": 4.0, "favoriteCount": 1 },
        "tags": [],
        "modelVersions": [{
            "id": 8002,
            "name": "v1",
            "baseModel": "Flux.1 D",
            "baseModelType": "Standard",
            "trainedWords": [],
            "files": [{
                "id": 2, "name": "y.safetensors", "sizeKB": 100,
                "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/y.safetensors",
                "hashes": { "SHA256": "cafef00d" }
            }],
            "images": []
        }]
    }],
    "metadata": { "totalPages": 37, "totalItems": 37, "currentPage": 1 }
}"#;

#[tokio::test]
async fn live_search_honors_page_size_and_reports_the_upstream_total() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string(TWO_FLUX_LORA_FIXTURE))
        .mount(&server)
        .await;
    let tmp = tempfile::tempdir().expect("tempdir");
    let state = AppState::for_tests().with_civitai_base(server.uri());
    {
        let mut cfg = state.config.write().await;
        cfg.models_dir = tmp.path().to_string_lossy().into_owned();
    }
    let router = create_router(state);

    let (status, body) = get(
        router,
        "/api/catalog/search?source=civitai&page=2&page_size=1",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "live search failed: {body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(parsed["page"], 2);
    assert_eq!(parsed["page_size"], 1);
    let entries = parsed["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0]["id"], "cv:8002",
        "page 2 must surface the next row, not repeat page 1"
    );
    assert_eq!(
        parsed["total"], 37,
        "total must describe the complete filtered feed, not this page"
    );

    let requests = server.received_requests().await.expect("requests");
    let request = requests.last().expect("catalog request");
    let query = request
        .url
        .query_pairs()
        .collect::<std::collections::HashMap<_, _>>();
    assert!(
        !query.contains_key("page"),
        "page= is ignored by Civitai and must not be sent"
    );
    assert_eq!(
        query.get("limit").map(|value| value.as_ref()),
        Some("1"),
        "limit must be one page — deep pages ride the cursor chain, not a widened window"
    );
}

/// `?sort=` must be validated, not silently ignored — silent ignoring is
/// exactly the bug this endpoint shipped with. Unknown values (including
/// the retired client-side "name" sort) get a 422 naming the vocabulary.
#[tokio::test]
async fn live_search_rejects_unknown_sort_naming_allowed_values() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    for bad in ["name", "most-downloaded", "bogus"] {
        let (status, body) = get(
            router.clone(),
            &format!("/api/catalog/search?q=test&source=civitai&sort={bad}"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "sort={bad} must be rejected: {body}"
        );
        assert!(
            body.contains("downloads") && body.contains("recent") && body.contains("rating"),
            "422 body must name the allowed values: {body}"
        );
    }
}

const NEWEST_FLUX_LORA_FIXTURE: &str = r#"{
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
                "id": 1, "name": "fresh.safetensors", "sizeKB": 100,
                "downloadCount": 1, "metadata": { "format": "SafeTensor" },
                "downloadUrl": "https://civitai.example/fresh.safetensors",
                "hashes": { "SHA256": "cafebabe" }
            }],
            "images": []
        }]
    }],
    "metadata": { "totalPages": 1 }
}"#;

/// Regression: the 5-minute in-process cache must key on the sort. The
/// same query issued with two different sorts must forward each sort
/// upstream and keep separate cache entries — repeated requests per sort
/// return that sort's rows, never the other's.
#[tokio::test]
async fn live_search_forwards_sort_and_keys_cache_per_sort() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/models"))
        .and(wiremock::matchers::query_param("sort", "Most Downloaded"))
        .respond_with(ResponseTemplate::new(200).set_body_string(FLUX_LORA_FIXTURE))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/models"))
        .and(wiremock::matchers::query_param("sort", "Newest"))
        .respond_with(ResponseTemplate::new(200).set_body_string(NEWEST_FLUX_LORA_FIXTURE))
        .expect(1)
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let state = AppState::for_tests().with_civitai_base(server.uri());
    {
        let mut cfg = state.config.write().await;
        cfg.models_dir = tmp.path().to_string_lossy().into_owned();
    }
    let router = create_router(state);

    let downloads_uri =
        "/api/catalog/search?q=test&family=flux&kind=lora&source=civitai&sort=downloads";
    let recent_uri = "/api/catalog/search?q=test&family=flux&kind=lora&source=civitai&sort=recent";

    let (status, body) = get(router.clone(), downloads_uri).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["entries"][0]["id"], "cv:8001");

    let (status, body) = get(router.clone(), recent_uri).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        parsed["entries"][0]["id"], "cv:8002",
        "recent sort must not replay the downloads-sorted cache entry: {body}"
    );

    // Second pass per sort — the expect(1) on each mock proves these are
    // cache hits, and each sort still returns its own rows.
    let (_, body) = get(router.clone(), downloads_uri).await;
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["entries"][0]["id"], "cv:8001");
    let (_, body) = get(router, recent_uri).await;
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["entries"][0]["id"], "cv:8002");
}

/// Omitting `?sort=` keeps the historical most-downloaded ordering, and
/// an explicit `sort=downloads` shares its cache entry (same opts).
#[tokio::test]
async fn live_search_defaults_to_downloads_sort() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/models"))
        .and(wiremock::matchers::query_param("sort", "Most Downloaded"))
        .respond_with(ResponseTemplate::new(200).set_body_string(FLUX_LORA_FIXTURE))
        .expect(1)
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let state = AppState::for_tests().with_civitai_base(server.uri());
    {
        let mut cfg = state.config.write().await;
        cfg.models_dir = tmp.path().to_string_lossy().into_owned();
    }
    let router = create_router(state);

    let (status, body) = get(
        router.clone(),
        "/api/catalog/search?q=test&family=flux&kind=lora&source=civitai",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["entries"][0]["id"], "cv:8001");

    // Explicit sort=downloads is the same cache key as the implicit
    // default — expect(1) above holds across both requests.
    let (status, body) = get(
        router,
        "/api/catalog/search?q=test&family=flux&kind=lora&source=civitai&sort=downloads",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn live_search_returns_manual_clip_component_rows() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    let (status, body) = get(
        router,
        "/api/catalog/search?q=clip&kind=clip&source=hf&family=sdxl",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    let entries = parsed["entries"].as_array().expect("entries array");
    let ids = entries
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"hf:companion/clip-l"), "{ids:?}");
    assert!(ids.contains(&"hf:companion/clip-g"), "{ids:?}");
    assert!(entries.iter().all(|entry| entry["kind"] == "clip"));
}

#[tokio::test]
async fn live_search_free_text_can_find_manual_clip_components() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    let (status, body) = get(router, "/api/catalog/search?q=clip&source=hf&family=sdxl").await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    let entries = parsed["entries"].as_array().expect("entries array");
    assert!(
        entries
            .iter()
            .any(|entry| entry["id"] == "hf:companion/clip-l" && entry["kind"] == "clip"),
        "{entries:?}"
    );
}

#[tokio::test]
async fn live_search_returns_manual_tokenizer_component_rows() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    let (status, body) = get(
        router,
        "/api/catalog/search?q=tokenizer&kind=tokenizer&source=hf&family=flux2",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    let entries = parsed["entries"].as_array().expect("entries array");
    let ids = entries
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"hf:companion/flux2-te/tokenizer"), "{ids:?}");
    assert!(
        ids.contains(&"hf:companion/flux2-te-9b/tokenizer"),
        "{ids:?}"
    );
    assert!(entries.iter().all(|entry| entry["kind"] == "tokenizer"));
}

#[tokio::test]
async fn catalog_download_enqueues_manual_component_companion() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    let (status, body) = post(router, "/api/catalog/hf%3Acompanion%2Fclip-l/download").await;

    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert!(
        parsed["primary_job_id"].as_str().is_some(),
        "component downloads should enqueue the backing companion manifest: {body}"
    );
    assert_eq!(parsed["companion_jobs"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn catalog_download_rejects_missing_token_before_enqueuing_companions() {
    if std::env::var("CIVITAI_TOKEN").is_ok() {
        // This regression exercises the no-credential server path. Environments
        // that deliberately inject a token cannot represent that state.
        return;
    }
    let server = MockServer::start().await;
    let version = r#"{
        "id": 8001,
        "modelId": 9001,
        "name": "v1",
        "baseModel": "SDXL 1.0",
        "baseModelType": "Standard",
        "trainedWords": [],
        "files": [{
            "id": 1,
            "name": "x.safetensors",
            "sizeKB": 100,
            "downloadCount": 1,
            "metadata": { "format": "SafeTensor" },
            "downloadUrl": "https://civitai.example/x.safetensors",
            "hashes": { "SHA256": "deadbeef" }
        }],
        "images": [],
        "model": {
            "name": "Token-gated SDXL",
            "type": "Checkpoint",
            "nsfw": false,
            "tags": []
        }
    }"#;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/model-versions/8001"))
        .respond_with(ResponseTemplate::new(200).set_body_string(version))
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let state = AppState::for_tests().with_civitai_base(server.uri());
    {
        let mut cfg = state.config.write().await;
        cfg.models_dir = tmp.path().to_string_lossy().into_owned();
    }
    let router = create_router(state);

    let (status, body) = post(router.clone(), "/api/catalog/cv%3A8001/download").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert!(body.contains("saved in Settings"), "{body}");

    let (downloads_status, downloads_body) = get(router, "/api/downloads").await;
    assert_eq!(downloads_status, StatusCode::OK, "{downloads_body}");
    let downloads: serde_json::Value = serde_json::from_str(&downloads_body).unwrap();
    assert!(
        downloads["active"].is_null()
            && downloads["queued"].as_array().is_some_and(Vec::is_empty)
            && downloads["history"].as_array().is_some_and(Vec::is_empty),
        "a rejected primary must not leave partial companion jobs: {downloads_body}"
    );
}

#[tokio::test]
async fn live_search_marks_installed_when_sidecar_and_file_present() {
    let (state, _server, tmp) = build_state().await;

    // Fabricate the post-install state: a sanitized subdir under
    // models_dir containing a sidecar AND the primary file.
    let cv_dir = tmp.path().join("cv-8001");
    std::fs::create_dir_all(&cv_dir).unwrap();
    let entry = mold_catalog::entry::CatalogEntry {
        id: mold_catalog::entry::CatalogId::from("cv:8001"),
        source: mold_catalog::entry::Source::Civitai,
        source_id: "8001".into(),
        name: "Live Flux LoRA".into(),
        author: Some("alice".into()),
        family: mold_catalog::families::Family::Flux,
        family_role: mold_catalog::entry::FamilyRole::Finetune,
        sub_family: Some("flux1-d".into()),
        modality: mold_catalog::entry::Modality::Image,
        kind: mold_catalog::entry::Kind::Lora,
        file_format: mold_catalog::entry::FileFormat::Safetensors,
        bundling: mold_catalog::entry::Bundling::SingleFile,
        size_bytes: Some(12),
        download_count: 0,
        rating: None,
        likes: 0,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: None,
        license_flags: mold_catalog::entry::LicenseFlags::default(),
        tags: vec![],
        companions: vec![],
        download_recipe: mold_catalog::entry::DownloadRecipe {
            files: vec![],
            needs_token: Some(mold_catalog::entry::TokenKind::Civitai),
        },
        supported: true,
        created_at: None,
        updated_at: None,
        added_at: 0,
        trained_words: vec!["live trigger".into()],
        page_url: None,
    };
    let sidecar = sidecar_from_entry(&entry, "x.safetensors".into());
    write_sidecar(&cv_dir.join(SIDECAR_FILENAME), &sidecar).unwrap();
    std::fs::write(cv_dir.join("x.safetensors"), b"fake-weights").unwrap();

    let router = create_router(state);
    let (status, body) = get(
        router,
        "/api/catalog/search?q=test&family=flux&kind=lora&source=civitai",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entry0 = &parsed["entries"][0];
    assert_eq!(entry0["installed"], true);
    assert!(entry0["primary_path"]
        .as_str()
        .unwrap()
        .ends_with("cv-8001/x.safetensors"));
}

/// `/api/models` rows for installed catalog checkpoints must carry the
/// sidecar's human-readable title as an additive `display_name`, so UIs
/// can stop rendering opaque `cv:<id>` slugs. Manifest rows stay
/// untouched — no `display_name` key at all.
#[tokio::test]
async fn list_models_carries_display_name_for_catalog_rows() {
    let (state, _server, tmp) = build_state().await;

    let ckpt_dir = tmp.path().join("cv-1759168");
    std::fs::create_dir_all(&ckpt_dir).unwrap();
    let sidecar = CatalogSidecar {
        schema: 1,
        id: "cv:1759168".into(),
        source: "civitai".into(),
        source_id: "1759168".into(),
        name: "Juggernaut XL - Ragnarok".into(),
        author: Some("KandooAI".into()),
        family: "sdxl".into(),
        family_role: "finetune".into(),
        sub_family: None,
        kind: "checkpoint".into(),
        modality: "image".into(),
        nsfw: Some(true),
        description: Some("A cinematic SDXL checkpoint.".into()),
        tags: vec!["cinematic".into()],
        license: Some("creativeml-openrail-m".into()),
        page_url: Some("https://civitai.com/models/1759168".into()),
        thumbnail_url: None,
        // Must match the fixture file's real size or the partial-download
        // guard in `primary_path_if_present` hides the row.
        size_bytes: Some(1),
        supported: true,
        trained_words: vec![],
        primary_filename_rel: "juggernaut.safetensors".into(),
        written_at: 0,
    };
    write_sidecar(&ckpt_dir.join(SIDECAR_FILENAME), &sidecar).unwrap();
    std::fs::write(ckpt_dir.join("juggernaut.safetensors"), b"x").unwrap();

    let router = create_router(state);
    let (status, body) = get(router, "/api/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let models: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();

    let row = models
        .iter()
        .find(|m| m["name"] == "cv:1759168")
        .expect("installed catalog checkpoint listed");
    assert_eq!(row["display_name"], "Juggernaut XL - Ragnarok");
    assert_eq!(row["kind"], "checkpoint");
    assert_eq!(row["modality"], "image");
    assert_eq!(row["nsfw"], true);
    assert_eq!(row["description"], "A cinematic SDXL checkpoint.");

    let manifest_row = models
        .iter()
        .find(|m| m["name"].as_str().is_some_and(|n| !n.starts_with("cv:")))
        .expect("manifest rows present");
    assert!(
        manifest_row.get("display_name").is_none(),
        "manifest rows must not grow a display_name key"
    );
}

#[tokio::test]
async fn installed_endpoint_returns_only_kind_filtered_sidecars() {
    let (state, _server, tmp) = build_state().await;

    // Drop two sidecars: one LoRA, one Checkpoint. The picker query
    // filters to kind=lora — the checkpoint MUST NOT appear.
    let lora_dir = tmp.path().join("cv-1");
    let ckpt_dir = tmp.path().join("cv-2");
    std::fs::create_dir_all(&lora_dir).unwrap();
    std::fs::create_dir_all(&ckpt_dir).unwrap();

    let lora_sc = CatalogSidecar {
        schema: 1,
        id: "cv:1".into(),
        source: "civitai".into(),
        source_id: "1".into(),
        name: "Lora A".into(),
        author: Some("alice".into()),
        family: "flux".into(),
        family_role: "finetune".into(),
        sub_family: None,
        kind: "lora".into(),
        modality: "image".into(),
        nsfw: None,
        description: None,
        tags: vec![],
        license: None,
        page_url: None,
        thumbnail_url: None,
        size_bytes: None,
        supported: true,
        trained_words: vec!["trigger-A".into()],
        primary_filename_rel: "lora.safetensors".into(),
        written_at: 0,
    };
    let ckpt_sc = CatalogSidecar {
        kind: "checkpoint".into(),
        primary_filename_rel: "ckpt.safetensors".into(),
        ..lora_sc.clone()
    };
    write_sidecar(&lora_dir.join(SIDECAR_FILENAME), &lora_sc).unwrap();
    std::fs::write(lora_dir.join("lora.safetensors"), b"x").unwrap();
    write_sidecar(&ckpt_dir.join(SIDECAR_FILENAME), &ckpt_sc).unwrap();
    std::fs::write(ckpt_dir.join("ckpt.safetensors"), b"x").unwrap();

    let router = create_router(state);
    let (status, body) = get(router, "/api/catalog/installed?kind=lora&family=flux").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entries = parsed["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], "cv:1");
    assert_eq!(entries[0]["installed"], true);
    assert_eq!(entries[0]["trained_words"][0], "trigger-A");
    assert!(
        entries[0]["nsfw"].is_null(),
        "a legacy sidecar without a safety classification must stay unknown"
    );
}

#[tokio::test]
async fn installed_endpoint_treats_qwen_edit_loras_as_qwen_image_compatible() {
    let (state, _server, tmp) = build_state().await;
    write_lora_sidecar(tmp.path(), 42, "qwen-image", 42);
    write_lora_sidecar(tmp.path(), 43, "flux", 43);

    let router = create_router(state);
    let (status, body) = get(
        router,
        "/api/catalog/installed?kind=lora&family=qwen-image-edit",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entries = parsed["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], "cv:42");
    assert_eq!(entries[0]["family"], "qwen-image");
}

#[tokio::test]
async fn loras_endpoint_filters_by_model_family_and_returns_all_matches() {
    let (state, _server, tmp) = build_state().await;

    for idx in 0..=10 {
        write_lora_sidecar(tmp.path(), idx, "flux", idx as i64);
    }
    write_lora_sidecar(tmp.path(), 99, "sdxl", 99);

    let router = create_router(state);
    let (status, body) = get(router.clone(), "/api/loras?model=flux-dev:q8").await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entries = parsed.as_array().unwrap();
    assert_eq!(entries.len(), 11);
    assert!(entries.iter().all(|entry| entry["family"] == "flux"));
    assert_eq!(entries[0]["id"], "cv:10");
    assert_eq!(entries[0]["trained_words"][0], "trigger-10");
    assert!(entries.iter().any(|entry| entry["id"] == "cv:0"));
    assert!(entries.iter().all(|entry| entry["id"] != "cv:99"));
    assert!(entries[0]["path"]
        .as_str()
        .unwrap()
        .ends_with("cv-10/lora-10.safetensors"));

    let (status, body) = get(router, "/api/loras").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entries = parsed.as_array().unwrap();
    assert_eq!(entries.len(), 12);
    assert!(entries.iter().any(|entry| entry["id"] == "cv:99"));
}

#[tokio::test]
async fn loras_endpoint_returns_qwen_image_loras_for_qwen_edit_models() {
    let (state, _server, tmp) = build_state().await;
    write_lora_sidecar(tmp.path(), 42, "qwen-image", 42);
    write_lora_sidecar(tmp.path(), 43, "flux", 43);

    let router = create_router(state);
    let (status, body) = get(router, "/api/loras?model=qwen-image-edit-2511:q4").await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entries = parsed.as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], "cv:42");
    assert_eq!(entries[0]["family"], "qwen-image");
}

#[tokio::test]
async fn loras_endpoint_rejects_unknown_model_filter() {
    let (state, _server, _tmp) = build_state().await;
    let router = create_router(state);

    let (status, body) = get(router, "/api/loras?model=not-a-real-model").await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["code"], "UNKNOWN_MODEL");
}

fn write_lora_sidecar(root: &std::path::Path, idx: usize, family: &str, written_at: i64) {
    let dir = root.join(format!("cv-{idx}"));
    std::fs::create_dir_all(&dir).unwrap();
    let filename = format!("lora-{idx}.safetensors");
    let sidecar = CatalogSidecar {
        schema: 1,
        id: format!("cv:{idx}"),
        source: "civitai".into(),
        source_id: idx.to_string(),
        name: format!("LoRA {idx}"),
        author: Some("alice".into()),
        family: family.into(),
        family_role: "finetune".into(),
        sub_family: None,
        kind: "lora".into(),
        modality: "image".into(),
        nsfw: None,
        description: None,
        tags: vec![],
        license: None,
        page_url: None,
        thumbnail_url: None,
        size_bytes: Some(1),
        supported: true,
        trained_words: vec![format!("trigger-{idx}")],
        primary_filename_rel: filename.clone(),
        written_at,
    };
    write_sidecar(&dir.join(SIDECAR_FILENAME), &sidecar).unwrap();
    std::fs::write(dir.join(filename), b"x").unwrap();
}

#[tokio::test]
async fn installed_endpoint_marks_uninstalled_when_primary_missing() {
    let (state, _server, tmp) = build_state().await;

    // Sidecar is present, but the primary file was removed (e.g. user
    // deleted it manually). The endpoint MUST surface installed=false
    // instead of pointing at a non-existent path.
    let dir = tmp.path().join("cv-7");
    std::fs::create_dir_all(&dir).unwrap();
    let sc = CatalogSidecar {
        schema: 1,
        id: "cv:7".into(),
        source: "civitai".into(),
        source_id: "7".into(),
        name: "Stale".into(),
        author: None,
        family: "flux".into(),
        family_role: "finetune".into(),
        sub_family: None,
        kind: "lora".into(),
        modality: "image".into(),
        nsfw: None,
        description: None,
        tags: vec![],
        license: None,
        page_url: None,
        thumbnail_url: None,
        size_bytes: None,
        supported: true,
        trained_words: vec![],
        primary_filename_rel: "missing.safetensors".into(),
        written_at: 0,
    };
    write_sidecar(&dir.join(SIDECAR_FILENAME), &sc).unwrap();
    // Deliberately do NOT write missing.safetensors.

    let router = create_router(state);
    let (status, body) = get(router, "/api/catalog/installed").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let entries = parsed["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["installed"], false);
    assert!(entries[0]["primary_path"].is_null());
}

// ── install_catalog_model error-path tests (Task 4) ─────────────────────────

/// `install_catalog_model` must surface upstream 404s as `NotFound` so
/// the user gets a 404 with a helpful message rather than the legacy
/// "not installed" 404 every Civitai outage looked like.
#[tokio::test]
async fn install_catalog_model_returns_not_found_for_unknown_id() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/model-versions/99999999"))
        .respond_with(ResponseTemplate::new(404).set_body_string(r#"{"error":"not found"}"#))
        .mount(&server)
        .await;

    let state = AppState::for_tests().with_civitai_base(server.uri());
    let res = crate::model_manager::install_catalog_model(&state, "cv:99999999").await;
    let err = res.expect_err("must fail");
    assert!(
        matches!(err, mold_core::InstallError::NotFound(_)),
        "got {err:?}"
    );
}

/// `install_catalog_model` must distinguish a 5xx upstream payload from
/// a clean 404 — the former indicates either a mold parsing bug or a
/// genuinely broken upstream response, and is mapped to RecipeMalformed.
#[tokio::test]
async fn install_catalog_model_returns_recipe_malformed_on_upstream_5xx() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/api/v1/model-versions/123"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal"))
        .mount(&server)
        .await;

    let state = AppState::for_tests().with_civitai_base(server.uri());
    let res = crate::model_manager::install_catalog_model(&state, "cv:123").await;
    let err = res.expect_err("must fail");
    assert!(
        matches!(err, mold_core::InstallError::RecipeMalformed(_)),
        "got {err:?}"
    );
}

/// `install_catalog_model` must surface unreachable-upstream as the
/// Network variant, NOT as "not installed". This is the load-bearing
/// fix for the "Civitai is down" scenario the plan calls out.
#[tokio::test]
async fn install_catalog_model_returns_network_error_when_civitai_unreachable() {
    // Point at a port that nothing's listening on. Connection refused →
    // reqwest::Error → InstallError::Network.
    let state = AppState::for_tests().with_civitai_base("http://127.0.0.1:1");
    let res = crate::model_manager::install_catalog_model(&state, "cv:42").await;
    let err = res.expect_err("must fail");
    assert!(
        matches!(err, mold_core::InstallError::Network(_)),
        "got {err:?}"
    );
}

/// Pins the exact `/api/catalog/installed` entry wire shape. Written
/// against the original ad-hoc `serde_json::json!` payload before the
/// shared `mold_core::catalog_wire::InstalledCatalogEntry` struct took
/// over serialization — the SPA and older CLIs depend on every key
/// (including explicit nulls) being present.
#[test]
fn sidecar_to_wire_shape_is_pinned() {
    let sc = CatalogSidecar {
        schema: 1,
        id: "cv:99".into(),
        source: "civitai".into(),
        source_id: "99".into(),
        name: "Pinned Lora".into(),
        author: Some("alice".into()),
        family: "flux".into(),
        family_role: "finetune".into(),
        sub_family: Some("dev".into()),
        kind: "lora".into(),
        modality: "image".into(),
        nsfw: Some(true),
        description: Some("A portrait style adapter.".into()),
        tags: vec!["portrait".into(), "style".into()],
        license: Some("creativeml-openrail-m".into()),
        page_url: Some("https://civitai.com/models/99".into()),
        thumbnail_url: Some("https://example.com/t.png".into()),
        size_bytes: Some(123_456),
        supported: true,
        trained_words: vec!["trigger".into()],
        primary_filename_rel: "lora.safetensors".into(),
        written_at: 1_700_000_000,
    };
    let got = serde_json::to_value(crate::catalog_api::sidecar_to_wire(
        sc,
        true,
        Some("/models/cv-99/lora.safetensors".into()),
    ))
    .unwrap();
    let expected = serde_json::json!({
        "id": "cv:99",
        "source": "civitai",
        "source_id": "99",
        "name": "Pinned Lora",
        "author": "alice",
        "family": "flux",
        "family_role": "finetune",
        "sub_family": "dev",
        "modality": "image",
        "kind": "lora",
        "file_format": "safetensors",
        "bundling": "single-file",
        "size_bytes": 123456,
        "download_count": 0,
        "rating": null,
        "likes": 0,
        "nsfw": true,
        "thumbnail_url": "https://example.com/t.png",
        "description": "A portrait style adapter.",
        "license": "creativeml-openrail-m",
        "license_flags": null,
        "tags": ["portrait", "style"],
        "companions": [],
        "companion_details": [],
        "download_recipe": { "files": [], "needs_token": null },
        "supported": true,
        "installed": true,
        "primary_path": "/models/cv-99/lora.safetensors",
        "created_at": null,
        "updated_at": null,
        "added_at": 1_700_000_000,
        "trained_words": ["trigger"],
        "page_url": "https://civitai.com/models/99",
    });
    assert_eq!(got, expected);
}
