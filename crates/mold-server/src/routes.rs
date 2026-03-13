use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mold_core::{GenerateRequest, ModelInfo, OutputFormat, ServerStatus};
use mold_inference::{model_registry, InferenceEngine};

use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/models", get(list_models))
        .route("/api/status", get(server_status))
        .route("/health", get(health))
        .with_state(state)
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut engine = state.engine.lock().await;

    // Lazy-load the model on first request
    if !engine.is_loaded() {
        tracing::info!("first request — loading model...");
        engine.load().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("model load error: {e}"),
            )
        })?;
    }

    let mut response = engine.generate(&req).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation error: {e}"),
        )
    })?;

    // Return raw image bytes with correct Content-Type
    let img = response.images.remove(0);
    let content_type = match img.format {
        OutputFormat::Png => "image/png",
        OutputFormat::Jpeg => "image/jpeg",
    };
    let headers = [(header::CONTENT_TYPE, content_type)];
    Ok((headers, img.data))
}

async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    let engine = state.engine.lock().await;
    let loaded_name = engine.model_name().to_string();
    let is_loaded = engine.is_loaded();

    let models: Vec<ModelInfo> = model_registry::known_models()
        .into_iter()
        .map(|mut m| {
            m.is_loaded = is_loaded && m.name == loaded_name;
            m
        })
        .collect();

    Json(models)
}

async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    let engine = state.engine.lock().await;

    let models_loaded = if engine.is_loaded() {
        vec![engine.model_name().to_string()]
    } else {
        vec![]
    };

    Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        models_loaded,
        gpu_info: None,
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}
