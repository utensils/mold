#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use mold_core::{GenerateRequest, GenerateResponse, ImageData};
    use mold_inference::InferenceEngine;
    use tower::ServiceExt;

    use crate::{routes::create_router, state::AppState};

    /// Minimal mock engine for testing routes without loading models.
    struct MockEngine {
        loaded: bool,
        fail: bool,
    }

    impl MockEngine {
        fn ready() -> Self {
            Self {
                loaded: true,
                fail: false,
            }
        }
        fn failing() -> Self {
            Self {
                loaded: true,
                fail: true,
            }
        }
    }

    impl InferenceEngine for MockEngine {
        fn generate(&mut self, req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            if self.fail {
                anyhow::bail!("mock engine error");
            }
            // Return a minimal 1x1 transparent PNG
            Ok(GenerateResponse {
                images: vec![ImageData {
                    data: minimal_png(),
                    format: req.output_format,
                    width: req.width,
                    height: req.height,
                    index: 0,
                }],
                generation_time_ms: 1,
                model: req.model.clone(),
                seed_used: req.seed.unwrap_or(42),
            })
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }

        fn is_loaded(&self) -> bool {
            self.loaded
        }

        fn load(&mut self) -> anyhow::Result<()> {
            self.loaded = true;
            Ok(())
        }
    }

    fn app_with(engine: MockEngine) -> axum::Router {
        let state = AppState::with_engine(engine);
        create_router(state)
    }

    fn app_empty() -> axum::Router {
        let state = AppState::empty(mold_core::Config::default());
        create_router(state)
    }

    fn generate_body(prompt: &str, width: u32, height: u32) -> String {
        // Use "mock-model" to match MockEngine::model_name() — avoids hot-swap path.
        format!(
            r#"{{"prompt":"{prompt}","model":"mock-model","width":{width},"height":{height},"steps":4,"batch_size":1,"output_format":"png"}}"#
        )
    }

    /// Returns a valid 1×1 PNG (8-byte signature + IHDR + IDAT + IEND).
    fn minimal_png() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // bit depth, color, CRC
            0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT length + type
            0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00,
            0x01, // compressed
            0xE2, 0x21, 0xBC, 0x33, // IDAT CRC
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82, // IEND
        ]
    }

    // ── /health ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn health_returns_200() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_when_no_model() {
        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── /api/status ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn status_returns_json() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("application/json"));
    }

    #[tokio::test]
    async fn status_when_no_model() {
        let app = app_empty();
        let resp = app
            .oneshot(Request::get("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(status["models_loaded"], serde_json::json!([]));
    }

    // ── /api/models ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn list_models_returns_json_array() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn list_models_uses_manifest_defaults_for_unpulled() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let models: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        // SD1.5 model should have manifest defaults (512x512, guidance 7.5, 25 steps)
        let sd15 = models.iter().find(|m| m["name"] == "sd15:fp16");
        if let Some(sd15) = sd15 {
            assert_eq!(sd15["default_width"], 512, "SD1.5 width should be 512");
            assert_eq!(sd15["default_height"], 512, "SD1.5 height should be 512");
            assert_eq!(sd15["default_steps"], 25, "SD1.5 steps should be 25");
            assert_eq!(
                sd15["default_guidance"], 7.5,
                "SD1.5 guidance should be 7.5"
            );
        }

        // FLUX schnell should have manifest defaults (1024x1024, guidance 0.0, 4 steps)
        let schnell = models.iter().find(|m| m["name"] == "flux-schnell:q8");
        if let Some(schnell) = schnell {
            assert_eq!(schnell["default_width"], 1024);
            assert_eq!(schnell["default_height"], 1024);
            assert_eq!(schnell["default_steps"], 4);
        }
    }

    // ── /api/generate — validation ───────────────────────────────────────────

    #[tokio::test]
    async fn generate_empty_prompt_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn generate_zero_width_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a cat", 0, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn generate_non_multiple_of_16_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a cat", 769, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn generate_oversized_returns_422() {
        let app = app_with(MockEngine::ready());
        // 1280x1280 = 1.64MP > 1.1MP limit
        let body = generate_body("a cat", 1280, 1280);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn generate_zero_steps_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = r#"{"prompt":"a cat","model":"mock-model","width":768,"height":768,"steps":0,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    // ── /api/generate — success path ─────────────────────────────────────────

    #[tokio::test]
    async fn generate_valid_request_returns_image_bytes() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a glowing robot", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(ct, "image/png");
    }

    // ── /api/generate — engine error ─────────────────────────────────────────

    #[tokio::test]
    async fn generate_engine_error_returns_500() {
        let app = app_with(MockEngine::failing());
        let body = generate_body("a cat", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ── /api/generate — unknown model ────────────────────────────────────────

    #[tokio::test]
    async fn generate_unknown_model_returns_400() {
        let app = app_empty();
        let body = r#"{"prompt":"a cat","model":"nonexistent-model-xyz","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ── /api/generate — known but not downloaded model returns 404 ───────────

    #[tokio::test]
    async fn generate_known_model_not_downloaded_returns_404() {
        let app = app_empty();
        // flux-schnell:q8 is a known manifest model but not configured/downloaded
        let body = r#"{"prompt":"a cat","model":"flux-schnell:q8","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── /api/openapi.json ────────────────────────────────────────────────────

    #[tokio::test]
    async fn openapi_json_returns_valid_spec() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(
                Request::get("/api/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let spec: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Must have openapi version field
        assert!(
            spec["openapi"].is_string(),
            "spec should have openapi version"
        );
        // Must have paths
        assert!(spec["paths"].is_object(), "spec should have paths");
        // Must have our generate endpoint
        assert!(
            spec["paths"]["/api/generate"].is_object(),
            "spec should have /api/generate path"
        );
    }

    // ── /api/docs ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn docs_returns_html() {
        let app = app_with(MockEngine::ready());
        let resp = app
            .oneshot(Request::get("/api/docs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/html"),
            "docs should return HTML, got: {ct}"
        );
    }

    // ── /api/generate/stream — SSE streaming ────────────────────────────────

    #[tokio::test]
    async fn stream_valid_request_returns_sse() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("a robot", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/event-stream"),
            "stream should return text/event-stream, got: {ct}"
        );

        // Collect body and verify it contains a complete event with base64 image
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event: complete"),
            "stream should contain a complete event"
        );
        assert!(
            text.contains("\"image\""),
            "complete event should contain base64 image"
        );
    }

    #[tokio::test]
    async fn stream_empty_prompt_returns_422() {
        let app = app_with(MockEngine::ready());
        let body = generate_body("", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn stream_unknown_model_returns_400() {
        let app = app_empty();
        let body = r#"{"prompt":"a cat","model":"nonexistent-model-xyz","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stream_known_model_not_downloaded_returns_404() {
        let app = app_empty();
        let body = r#"{"prompt":"a cat","model":"flux-schnell:q8","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn stream_engine_error_returns_sse_error() {
        let app = app_with(MockEngine::failing());
        let body = generate_body("a cat", 768, 768);
        let resp = app
            .oneshot(
                Request::post("/api/generate/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        // SSE stream starts with 200 — error is in the event stream
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event: error"),
            "stream should contain an error event"
        );
        assert!(
            text.contains("mock engine error"),
            "error event should contain the engine error message"
        );
    }
}
