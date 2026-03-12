use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use mold_core::{GenerateRequest, GenerateResponse, LoadModelRequest, ModelInfo, ServerStatus};
use mold_inference::{model_registry, InferenceEngine};

use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/models", get(list_models))
        .route("/api/status", get(server_status))
        .route("/api/models/load", post(load_model))
        .route("/api/models/{name}", delete(unload_model))
        .route("/health", get(health))
        .with_state(state)
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let engine = state.engine.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("lock error: {e}"),
        )
    })?;

    let response = engine.generate(&req).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation error: {e}"),
        )
    })?;

    Ok(Json(response))
}

async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    let engine = state.engine.lock().unwrap();
    let loaded = engine.loaded_models();

    let models: Vec<ModelInfo> = model_registry::known_models()
        .into_iter()
        .map(|mut m| {
            m.is_loaded = loaded.contains(&m.name);
            m
        })
        .collect();

    Json(models)
}

async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    let engine = state.engine.lock().unwrap();

    Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        models_loaded: engine.loaded_models(),
        gpu_info: None,
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut engine = state.engine.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("lock error: {e}"),
        )
    })?;

    engine.load_model(&req.model).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("load error: {e}"),
        )
    })?;

    Ok(StatusCode::OK)
}

async fn unload_model(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut engine = state.engine.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("lock error: {e}"),
        )
    })?;

    engine.unload_model(&name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unload error: {e}"),
        )
    })?;

    Ok(StatusCode::OK)
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}
