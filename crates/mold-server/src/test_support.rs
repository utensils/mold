//! Lightweight in-process test client for catalog route integration tests.
//! Avoids the full hyper boot — uses `tower::ServiceExt::oneshot` directly.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

pub struct TestResponse {
    pub status: StatusCode,
    pub body: String,
}

pub struct TestApp {
    router: axum::Router,
}

impl TestApp {
    /// Build an empty AppState (catalog endpoints proxy live HF/Civitai;
    /// tests that exercise live behaviour point catalog_live_civitai_base
    /// at a wiremock instance via `with_civitai_base`).
    pub async fn with_seeded_catalog() -> Self {
        let state = crate::state::AppState::for_tests();
        let router = crate::routes::create_router(state);
        Self { router }
    }

    pub async fn get(&self, uri: &str) -> TestResponse {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        TestResponse { status, body }
    }

    pub async fn post_json(&self, uri: &str, body: &str) -> TestResponse {
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        TestResponse { status, body }
    }
}
