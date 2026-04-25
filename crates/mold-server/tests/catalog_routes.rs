use axum::http::StatusCode;
use mold_server::test_support::TestApp;

#[tokio::test]
async fn list_catalog_returns_200_and_paginates() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog?family=sdxl&limit=5&offset=0").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["entries"].is_array());
    assert!(v["page_size"].as_i64().unwrap() <= 5);
}

#[tokio::test]
async fn get_catalog_entry_404_for_unknown() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog/hf:does/not-exist").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn families_endpoint_returns_counts() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog/families").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["families"].is_array());
}

#[tokio::test]
async fn list_with_search_uses_fts() {
    let app = TestApp::with_seeded_catalog().await;
    let resp = app.get("/api/catalog?q=Juggernaut&limit=10").await;
    assert_eq!(resp.status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(v["entries"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| { e["name"].as_str().unwrap_or("").contains("Juggernaut") }));
}

#[tokio::test]
async fn refresh_enqueue_then_status_returns_pending_or_running() {
    let app = TestApp::with_seeded_catalog().await;
    let post = app.post_json("/api/catalog/refresh", "{}").await;
    assert_eq!(post.status, axum::http::StatusCode::ACCEPTED);
    let v: serde_json::Value = serde_json::from_str(&post.body).unwrap();
    let id = v["id"].as_str().expect("id field").to_string();

    let status = app.get(&format!("/api/catalog/refresh/{id}")).await;
    assert_eq!(status.status, axum::http::StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&status.body).unwrap();
    let state = body["state"].as_str().unwrap();
    assert!(matches!(state, "pending" | "running" | "done"));
}

#[tokio::test]
async fn refresh_returns_409_when_already_running() {
    let app = TestApp::with_seeded_catalog().await;
    let first = app.post_json("/api/catalog/refresh", "{}").await;
    assert_eq!(first.status, axum::http::StatusCode::ACCEPTED);
    let second = app.post_json("/api/catalog/refresh", "{}").await;
    assert_eq!(second.status, axum::http::StatusCode::CONFLICT);
}
