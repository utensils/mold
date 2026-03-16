use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use mold_core::{GpuInfo, ModelInfo, ModelPaths, OutputFormat, ServerStatus};
use mold_inference::model_registry;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/models", get(list_models))
        .route("/api/models/load", post(load_model))
        .route("/api/models/unload", delete(unload_model))
        .route("/api/status", get(server_status))
        .route("/health", get(health))
        .with_state(state)
}

// ── /api/generate ─────────────────────────────────────────────────────────────

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<mold_core::GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Validate request before touching the engine
    if let Err(e) = validate_generate_request(&req) {
        return Err((StatusCode::UNPROCESSABLE_ENTITY, e));
    }

    // If the requested model differs from the currently loaded one, hot-swap.
    {
        let mut engine = state.engine.lock().await;
        let current = engine.model_name().to_string();
        if current != req.model {
            tracing::info!(
                from = %current,
                to = %req.model,
                "hot-swapping model"
            );
            let paths = ModelPaths::resolve(&req.model, &state.config).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    format!(
                        "no paths configured for model '{}'. Add [models.{}] to config.",
                        req.model, req.model
                    ),
                )
            })?;
            *engine = mold_inference::create_engine(
                req.model.clone(),
                paths,
                &state.config,
                mold_inference::LoadStrategy::Eager,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to create engine for {}: {e}", req.model),
                )
            })?;
        }

        // Load on first request (or after hot-swap)
        if !engine.is_loaded() {
            tracing::info!(model = %req.model, "loading model...");
            engine.load().map_err(|e| {
                tracing::error!("model load failed: {e:#}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("model load error: {e}"),
                )
            })?;
        }
    }

    // Run inference in a blocking task — panics caught → 500 with body.
    let engine = state.engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = engine.blocking_lock();
            guard.generate(&req)
        }))
    })
    .await
    .map_err(|e| {
        tracing::error!("inference task join error: {e:?}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference task failed".to_string(),
        )
    })?;

    let mut response = match result {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            tracing::error!("generation error: {e:#}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation error: {e}"),
            ));
        }
        Err(panic_payload) => {
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!("inference panicked: {msg}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("inference panicked: {msg}"),
            ));
        }
    };

    let img = response.images.remove(0);
    let content_type = match img.format {
        OutputFormat::Png => "image/png",
        OutputFormat::Jpeg => "image/jpeg",
    };
    Ok(([(header::CONTENT_TYPE, content_type)], img.data))
}

fn validate_generate_request(req: &mold_core::GenerateRequest) -> Result<(), String> {
    mold_core::validate_generate_request(req)
}

// ── /api/models ───────────────────────────────────────────────────────────────

/// Extended model info including generation defaults.
#[derive(Debug, Serialize)]
pub struct ModelInfoExtended {
    #[serde(flatten)]
    pub info: ModelInfo,
    pub default_steps: u32,
    pub default_guidance: f64,
    pub default_width: u32,
    pub default_height: u32,
    pub description: String,
}

async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfoExtended>> {
    let engine = state.engine.lock().await;
    let loaded_name = engine.model_name().to_string();
    let is_loaded = engine.is_loaded();
    drop(engine);

    // Start with known static models, merge with config-defined models.
    let mut known: std::collections::HashMap<String, ModelInfo> = model_registry::known_models()
        .into_iter()
        .map(|m| (m.name.clone(), m))
        .collect();

    // Add any models from config that aren't in the static registry.
    for (name, mcfg) in &state.config.models {
        let size_gb = mcfg
            .transformer
            .as_deref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as f32 / 1_073_741_824.0)
            .unwrap_or(0.0);
        known.entry(name.clone()).or_insert_with(|| ModelInfo {
            name: name.clone(),
            family: mcfg.family.clone().unwrap_or_else(|| "flux".to_string()),
            size_gb,
            is_loaded: false,
            last_used: None,
            hf_repo: String::new(),
        });
    }

    let models: Vec<ModelInfoExtended> = known
        .into_values()
        .map(|mut m| {
            m.is_loaded = is_loaded && m.name == loaded_name;
            let mcfg = state.config.model_config(&m.name);
            ModelInfoExtended {
                default_steps: mcfg.effective_steps(&state.config),
                default_guidance: mcfg.effective_guidance(),
                default_width: mcfg.effective_width(&state.config),
                default_height: mcfg.effective_height(&state.config),
                description: mcfg.description.clone().unwrap_or_else(|| m.name.clone()),
                info: m,
            }
        })
        .collect();

    Json(models)
}

// ── /api/models/load ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct LoadModelBody {
    pub model: String,
}

async fn load_model(
    State(state): State<AppState>,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let paths = ModelPaths::resolve(&body.model, &state.config).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "no paths configured for model '{}'. Add [models.{}] to config.",
                body.model, body.model
            ),
        )
    })?;

    let mut engine = state.engine.lock().await;
    *engine = mold_inference::create_engine(
        body.model.clone(),
        paths,
        &state.config,
        mold_inference::LoadStrategy::Eager,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to create engine for {}: {e}", body.model),
        )
    })?;
    engine.load().map_err(|e| {
        tracing::error!("model load failed: {e:#}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to load {}: {e}", body.model),
        )
    })?;

    tracing::info!(model = %body.model, "model loaded via API");
    Ok(StatusCode::OK)
}

// ── /api/models/unload ────────────────────────────────────────────────────────

async fn unload_model(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut engine = state.engine.lock().await;
    if !engine.is_loaded() {
        return Ok((StatusCode::OK, "no model loaded".to_string()));
    }
    let name = engine.model_name().to_string();
    engine.unload();
    tracing::info!(model = %name, "model unloaded via API");
    Ok((StatusCode::OK, format!("unloaded {name}")))
}

// ── /api/status ───────────────────────────────────────────────────────────────

async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    let engine = state.engine.lock().await;
    let models_loaded = if engine.is_loaded() {
        vec![engine.model_name().to_string()]
    } else {
        vec![]
    };
    drop(engine);

    Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        models_loaded,
        gpu_info: query_gpu_info(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

// ── /health ───────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    StatusCode::OK
}

// ── GPU info ──────────────────────────────────────────────────────────────────

fn query_gpu_info() -> Option<GpuInfo> {
    let nvidia_smi = if std::path::Path::new("/run/current-system/sw/bin/nvidia-smi").exists() {
        "/run/current-system/sw/bin/nvidia-smi"
    } else {
        "nvidia-smi"
    };

    let output = std::process::Command::new(nvidia_smi)
        .args([
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }

    Some(GpuInfo {
        name: parts[0].to_string(),
        vram_total_mb: parts[1].parse().ok()?,
        vram_used_mb: parts[2].parse().ok()?,
    })
}
