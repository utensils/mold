#[cfg(all(test, feature = "metrics"))]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    use crate::{
        metrics::{self, MetricsState},
        routes::create_router,
        state::AppState,
    };

    /// Build a test app with the /metrics route and HTTP metrics middleware.
    fn app_with_metrics() -> Router {
        let handle = metrics::install_recorder_for_test();

        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = crate::state::QueueHandle::new(tx);
        let gpu_pool = std::sync::Arc::new(crate::gpu_pool::GpuPool {
            workers: Vec::new(),
        });
        let state = AppState::empty(mold_core::Config::default(), queue, gpu_pool, 200);

        let start_time = state.start_time;
        let metrics_state = MetricsState { handle, start_time };

        create_router(state)
            .layer(middleware::from_fn(metrics::http_metrics_middleware))
            .route(
                "/metrics",
                get(metrics::metrics_endpoint).with_state(metrics_state),
            )
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_format() {
        let app = app_with_metrics();

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            content_type.contains("text/plain"),
            "expected text/plain content type, got: {content_type}"
        );

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        // Should contain uptime gauge (recorded on each scrape).
        assert!(
            text.contains("mold_uptime_seconds"),
            "expected mold_uptime_seconds in metrics output:\n{text}"
        );
    }

    #[tokio::test]
    async fn metrics_records_http_requests() {
        let app = app_with_metrics();

        // Hit /health to generate an HTTP metric.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Now scrape /metrics and check for the recorded request.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            text.contains("mold_http_requests_total"),
            "expected mold_http_requests_total in output:\n{text}"
        );
        assert!(
            text.contains("mold_http_request_duration_seconds"),
            "expected mold_http_request_duration_seconds in output:\n{text}"
        );
    }

    #[tokio::test]
    async fn metrics_records_queue_submit() {
        // Ensure the recorder is installed regardless of test execution order.
        metrics::install_recorder_for_test();
        metrics::record_queue_submit();
        metrics::record_queue_depth(3);
        metrics::record_generation("test-model", 1.5);
        metrics::record_model_load("test-model", 2.0);
        metrics::set_model_loaded("test-model");
        metrics::record_gpu_memory(1_000_000_000);
        metrics::record_generation_error("test-model");

        // If we got here without panicking, the metric calls are valid.
    }
}
