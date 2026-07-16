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
    assert!(!super::replace_failed_search_credential(
        &hf_error, &mut opts, &forwarded,
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
    "metadata": { "totalPages": 1 }
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
        engine_phase: 3,
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
        thumbnail_url: None,
        size_bytes: None,
        engine_phase: 3,
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
        thumbnail_url: None,
        size_bytes: Some(1),
        engine_phase: 3,
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
        thumbnail_url: None,
        size_bytes: None,
        engine_phase: 3,
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
        thumbnail_url: Some("https://example.com/t.png".into()),
        size_bytes: Some(123_456),
        engine_phase: 3,
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
        "nsfw": false,
        "thumbnail_url": "https://example.com/t.png",
        "description": null,
        "license": null,
        "license_flags": null,
        "tags": [],
        "companions": [],
        "companion_details": [],
        "download_recipe": { "files": [], "needs_token": null },
        "engine_phase": 3,
        "installed": true,
        "primary_path": "/models/cv-99/lora.safetensors",
        "created_at": null,
        "updated_at": null,
        "added_at": 1_700_000_000,
        "trained_words": ["trigger"],
    });
    assert_eq!(got, expected);
}
