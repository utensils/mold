use axum::http::StatusCode;
use mold_server::test_support::TestApp;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn families_endpoint_returns_static_taxonomy() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog/families").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    let families = v["families"].as_array().expect("families array");
    let names: Vec<&str> = families
        .iter()
        .map(|row| row["family"].as_str().unwrap_or(""))
        .collect();
    for expected in [
        "flux",
        "flux2",
        "sd15",
        "sdxl",
        "sd3",
        "z-image",
        "ltx-video",
        "ltx2",
        "minimax-h3",
        "qwen-image",
        "qwen-image-edit",
        "wuerstchen",
    ] {
        assert!(
            names.contains(&expected),
            "family {expected:?} missing from sidebar list, got {names:?}",
        );
    }
    // Per-family counts are gone from the wire — live search hits one
    // family at a time, so the SPA's sidebar shows just the family name.
    let flux = families
        .iter()
        .find(|row| row["family"].as_str() == Some("flux"))
        .expect("flux row present");
    assert!(flux.get("foundation").is_none());
    assert!(flux.get("finetune").is_none());
}

#[tokio::test]
async fn capabilities_includes_catalog_block() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/capabilities").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert_eq!(v["catalog"]["available"], serde_json::Value::Bool(true));
    assert!(v["catalog"]["families"].is_array());
    assert!(v["catalog"]["families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|family| family == "minimax-h3"));
    assert!(v["catalog"]["families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|family| family == "sd3"));
    assert!(v["catalog"]["families"]
        .as_array()
        .unwrap()
        .iter()
        .any(|family| family == "qwen-image-edit"));
    // The reviewed compact H3 rows are ordinary model identities. Execution
    // availability is advertised by the task/backend capability instead of a
    // family-wide licensing restriction.
    assert_eq!(v["model_access"]["restrictions"], serde_json::json!([]));
}

#[tokio::test]
async fn qwen_image_edit_family_filter_is_accepted_and_applied() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("baseModels", "Qwen"))
        .and(query_param("baseModels", "Qwen 2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "items": [{
                "id": 1,
                "name": "QwenImageEdit2511 community",
                "type": "Checkpoint",
                "nsfw": false,
                "modelVersions": [{
                    "id": 2,
                    "name": "fp8",
                    "baseModel": "Qwen",
                    "baseModelType": "Standard",
                    "files": [{
                        "id": 3,
                        "name": "model.safetensors",
                        "sizeKB": 100,
                        "metadata": { "format": "SafeTensor" },
                        "downloadUrl": "https://civitai.example/model",
                        "hashes": {}
                    }],
                    "images": []
                }]
            }],
            "metadata": { "totalPages": 1 }
        })))
        .expect(1)
        .mount(&upstream)
        .await;

    let app = TestApp::with_civitai_base(upstream.uri()).await;
    let response = app
        .get("/api/catalog/search?family=qwen-image-edit&source=civitai&page=1&page_size=20")
        .await;
    assert_eq!(response.status, StatusCode::OK, "{}", response.body);
    let body: serde_json::Value = serde_json::from_str(&response.body).unwrap();
    let entries = body["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["family"], "qwen-image-edit");
}
