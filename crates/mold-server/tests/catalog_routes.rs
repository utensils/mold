use axum::http::StatusCode;
use mold_server::test_support::TestApp;

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
    // The reviewed compact H3 rows are ordinary model identities. Execution
    // availability is advertised by the task/backend capability instead of a
    // family-wide licensing restriction.
    assert_eq!(v["model_access"]["restrictions"], serde_json::json!([]));
}
