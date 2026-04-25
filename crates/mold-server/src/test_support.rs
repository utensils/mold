//! Lightweight in-process test client for catalog route integration tests.
//! Avoids the full hyper boot — uses `tower::ServiceExt::oneshot` directly.

use std::sync::Arc;

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
    /// Build an in-memory DB, seed a couple of fixture rows, and wire up the
    /// full router so integration tests can exercise the catalog endpoints.
    pub async fn with_seeded_catalog() -> Self {
        let db =
            mold_db::MetadataDb::open_in_memory().expect("open in-memory catalog DB for tests");

        // Seed a Juggernaut XL row so the FTS and family tests have data.
        db.catalog_upsert(
            "sdxl",
            &[mold_db::catalog::CatalogRow {
                id: "hf:RunDiffusion/Juggernaut-XL".into(),
                source: "hf".into(),
                source_id: "RunDiffusion/Juggernaut-XL".into(),
                name: "Juggernaut XL".into(),
                author: Some("RunDiffusion".into()),
                family: "sdxl".into(),
                family_role: "finetune".into(),
                sub_family: None,
                modality: "image".into(),
                kind: "checkpoint".into(),
                file_format: "safetensors".into(),
                bundling: "separated".into(),
                size_bytes: Some(1),
                download_count: 100,
                rating: Some(4.7),
                likes: 0,
                nsfw: 0,
                thumbnail_url: None,
                description: None,
                license: None,
                license_flags: None,
                tags: Some("[]".into()),
                companions: Some("[]".into()),
                download_recipe: r#"{"files":[],"needs_token":null}"#.into(),
                engine_phase: 1,
                created_at: None,
                updated_at: None,
                added_at: 0,
            }],
        )
        .expect("seed catalog fixture");

        let state = crate::state::AppState::for_tests(Arc::new(db));
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
}
