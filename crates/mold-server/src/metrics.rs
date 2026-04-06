//! Prometheus metrics instrumentation (behind the `metrics` feature flag).
//!
//! Installs a global [`metrics_exporter_prometheus`] recorder at startup and
//! provides a tower middleware layer that records per-request HTTP metrics.
//! The `/metrics` endpoint renders the Prometheus text exposition format.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::time::Instant;

// ── Metric name constants ──────────────────────────────────────────────────

pub const HTTP_REQUESTS_TOTAL: &str = "mold_http_requests_total";
pub const HTTP_REQUEST_DURATION: &str = "mold_http_request_duration_seconds";
pub const GENERATION_DURATION: &str = "mold_generation_duration_seconds";
pub const QUEUE_DEPTH: &str = "mold_queue_depth";
pub const QUEUE_TOTAL: &str = "mold_queue_total";
pub const MODEL_LOADED: &str = "mold_model_loaded";
pub const MODEL_LOAD_DURATION: &str = "mold_model_load_duration_seconds";
pub const MODEL_LOADS_TOTAL: &str = "mold_model_loads_total";
pub const GPU_MEMORY_USED: &str = "mold_gpu_memory_used_bytes";
pub const UPTIME: &str = "mold_uptime_seconds";
pub const GENERATION_ERRORS_TOTAL: &str = "mold_generation_errors_total";

// ── Recorder installation ──────────────────────────────────────────────────

/// Install the global Prometheus recorder and return a handle for rendering.
///
/// Must be called exactly once, early in server startup. Panics if a
/// recorder is already installed.
pub fn install_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

/// Test-friendly recorder installation: returns a handle to a shared recorder,
/// installing one if none exists yet. Safe to call from multiple tests.
#[cfg(test)]
pub fn install_recorder_for_test() -> PrometheusHandle {
    use std::sync::OnceLock;
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("failed to install Prometheus recorder for tests")
        })
        .clone()
}

// ── /metrics endpoint ──────────────────────────────────────────────────────

/// State for the /metrics endpoint: the Prometheus handle plus the server
/// start time for computing uptime on each scrape.
#[derive(Clone)]
pub struct MetricsState {
    pub handle: PrometheusHandle,
    pub start_time: std::time::Instant,
}

/// Render all collected metrics in Prometheus text exposition format.
///
/// Records uptime and GPU memory gauges on each scrape so they are always
/// current without needing a background ticker.
pub async fn metrics_endpoint(
    axum::extract::State(state): axum::extract::State<MetricsState>,
) -> impl IntoResponse {
    // Update point-in-time gauges right before rendering.
    record_uptime(state.start_time.elapsed().as_secs_f64());
    record_gpu_memory(mold_inference::device::vram_used_estimate());

    let body = state.handle.render();
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

// ── HTTP metrics middleware ────────────────────────────────────────────────

/// Tower middleware that records `mold_http_requests_total` and
/// `mold_http_request_duration_seconds` for every request.
///
/// The path label is normalized: `/api/gallery/image/<filename>` becomes
/// `/api/gallery/image/:filename`, etc.
pub async fn http_metrics_middleware(request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = normalize_path(request.uri().path());

    let start = Instant::now();
    let response = next.run(request).await;
    let duration = start.elapsed().as_secs_f64();

    let status = response.status().as_u16().to_string();

    counter!(HTTP_REQUESTS_TOTAL, "method" => method.clone(), "path" => path.clone(), "status" => status).increment(1);
    histogram!(HTTP_REQUEST_DURATION, "method" => method, "path" => path).record(duration);

    response
}

/// Collapse dynamic path segments into parameter placeholders so we don't
/// create unbounded label cardinality.
fn normalize_path(path: &str) -> String {
    // /api/gallery/image/<anything> → /api/gallery/image/:filename
    // /api/gallery/thumbnail/<anything> → /api/gallery/thumbnail/:filename
    if let Some(rest) = path.strip_prefix("/api/gallery/image/") {
        if !rest.is_empty() {
            return "/api/gallery/image/:filename".to_string();
        }
    }
    if let Some(rest) = path.strip_prefix("/api/gallery/thumbnail/") {
        if !rest.is_empty() {
            return "/api/gallery/thumbnail/:filename".to_string();
        }
    }
    path.to_string()
}

// ── Recording helpers ──────────────────────────────────────────────────────

/// Record a completed generation's duration.
pub fn record_generation(model: &str, duration_secs: f64) {
    histogram!(GENERATION_DURATION, "model" => model.to_string()).record(duration_secs);
}

/// Increment the generation error counter.
pub fn record_generation_error(model: &str) {
    counter!(GENERATION_ERRORS_TOTAL, "model" => model.to_string()).increment(1);
}

/// Record current queue depth (absolute gauge value).
pub fn record_queue_depth(depth: usize) {
    gauge!(QUEUE_DEPTH).set(depth as f64);
}

/// Increment the total-enqueued counter.
pub fn record_queue_submit() {
    counter!(QUEUE_TOTAL).increment(1);
}

/// Record that a model was loaded, with its duration.
pub fn record_model_load(model: &str, duration_secs: f64) {
    counter!(MODEL_LOADS_TOTAL, "name" => model.to_string()).increment(1);
    histogram!(MODEL_LOAD_DURATION, "name" => model.to_string()).record(duration_secs);
}

/// Set the `mold_model_loaded` gauge for the given model to 1, and clear
/// any previously active model.
pub fn set_model_loaded(model: &str) {
    gauge!(MODEL_LOADED, "name" => model.to_string()).set(1.0);
}

/// Clear the `mold_model_loaded` gauge for a model.
pub fn clear_model_loaded(model: &str) {
    gauge!(MODEL_LOADED, "name" => model.to_string()).set(0.0);
}

/// Record GPU memory usage in bytes.
pub fn record_gpu_memory(bytes: u64) {
    gauge!(GPU_MEMORY_USED).set(bytes as f64);
}

/// Record server uptime in seconds.
pub fn record_uptime(seconds: f64) {
    gauge!(UPTIME).set(seconds);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_gallery_image_path() {
        assert_eq!(
            normalize_path("/api/gallery/image/some-file.png"),
            "/api/gallery/image/:filename"
        );
    }

    #[test]
    fn normalize_gallery_thumbnail_path() {
        assert_eq!(
            normalize_path("/api/gallery/thumbnail/some-file.png"),
            "/api/gallery/thumbnail/:filename"
        );
    }

    #[test]
    fn normalize_preserves_static_paths() {
        assert_eq!(normalize_path("/api/generate"), "/api/generate");
        assert_eq!(normalize_path("/api/status"), "/api/status");
        assert_eq!(normalize_path("/health"), "/health");
        assert_eq!(normalize_path("/metrics"), "/metrics");
    }

    #[test]
    fn normalize_gallery_base_preserved() {
        // Bare /api/gallery/image/ without a filename stays as-is
        assert_eq!(normalize_path("/api/gallery/image/"), "/api/gallery/image/");
    }
}
