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
use utoipa::OpenApi;

use crate::state::AppState;

#[derive(OpenApi)]
#[openapi(
    paths(generate, list_models, load_model, pull_model_endpoint, unload_model, server_status, health),
    components(schemas(
        mold_core::GenerateRequest,
        mold_core::GenerateResponse,
        mold_core::ImageData,
        mold_core::OutputFormat,
        mold_core::ModelInfo,
        mold_core::ServerStatus,
        mold_core::GpuInfo,
        ModelInfoExtended,
        LoadModelBody,
    )),
    tags(
        (name = "generation", description = "Image generation"),
        (name = "models", description = "Model management"),
        (name = "server", description = "Server status and health"),
    ),
    info(
        title = "mold",
        description = "Local AI image generation server — FLUX, SD3.5, SD1.5, SDXL, Z-Image, Flux.2, Qwen-Image",
        version = env!("CARGO_PKG_VERSION"),
    )
)]
pub struct ApiDoc;

pub fn create_router(state: AppState) -> Router {
    // Stateful routes (need AppState) are added first, then .with_state() converts
    // Router<AppState> → Router<()>. Stateless routes (OpenAPI, docs) are merged after.
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/models", get(list_models))
        .route("/api/models/load", post(load_model))
        .route("/api/models/pull", post(pull_model_endpoint))
        .route("/api/models/unload", delete(unload_model))
        .route("/api/status", get(server_status))
        .route("/health", get(health))
        .with_state(state)
        .route("/api/openapi.json", get(openapi_json))
        .route("/api/docs", get(scalar_docs))
}

// ── Model readiness ──────────────────────────────────────────────────────────

/// Ensure the requested model is loaded and ready for inference.
/// Handles: no engine, model swap, and lazy loading.
/// Does NOT auto-pull — returns 404 so the client can pull and retry.
async fn ensure_model_ready(
    state: &AppState,
    model_name: &str,
) -> Result<(), (StatusCode, String)> {
    // Fast path: engine exists, correct model, already loaded
    {
        let engine = state.engine.lock().await;
        if let Some(ref e) = *engine {
            if e.model_name() == model_name && e.is_loaded() {
                return Ok(());
            }
        }
    }

    // Try to resolve paths from current config
    let paths = {
        let config = state.config.read().await;
        ModelPaths::resolve(model_name, &config)
    };

    if let Some(paths) = paths {
        // Paths exist — create engine and load
        return create_and_load_engine(state, model_name, paths).await;
    }

    // Model paths not found — tell the client to pull or report unknown model
    if mold_core::manifest::find_manifest(model_name).is_some() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("model '{model_name}' is not downloaded. Run: mold pull {model_name}"),
        ));
    }
    Err((
        StatusCode::BAD_REQUEST,
        format!("unknown model '{model_name}'. Run 'mold list' to see available models."),
    ))
}

/// Create an inference engine and load it into AppState.
async fn create_and_load_engine(
    state: &AppState,
    model_name: &str,
    paths: ModelPaths,
) -> Result<(), (StatusCode, String)> {
    let config = state.config.read().await;
    let new_engine = mold_inference::create_engine(
        model_name.to_string(),
        paths,
        &config,
        mold_inference::LoadStrategy::Eager,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to create engine for '{model_name}': {e}"),
        )
    })?;
    drop(config);

    let mut engine = state.engine.lock().await;

    // Log hot-swap if replacing an existing engine
    if let Some(ref old) = *engine {
        if old.model_name() != model_name {
            tracing::info!(
                from = %old.model_name(),
                to = %model_name,
                "hot-swapping model"
            );
        }
    }

    *engine = Some(new_engine);

    // Load the model (weights into GPU)
    if let Some(ref mut e) = *engine {
        if !e.is_loaded() {
            tracing::info!(model = %model_name, "loading model...");
            e.load().map_err(|e| {
                tracing::error!("model load failed: {e:#}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("model load error: {e}"),
                )
            })?;
        }
    }

    Ok(())
}

// ── /api/generate ─────────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/generate",
    tag = "generation",
    request_body = mold_core::GenerateRequest,
    responses(
        (status = 200, description = "Generated image bytes", content_type = "image/png"),
        (status = 404, description = "Model not downloaded"),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Inference error"),
    )
)]
async fn generate(
    State(state): State<AppState>,
    Json(req): Json<mold_core::GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!(
        model = %req.model,
        prompt = %req.prompt,
        width = req.width,
        height = req.height,
        steps = req.steps,
        guidance = req.guidance,
        seed = ?req.seed,
        format = %req.output_format,
        "generate request"
    );

    // Validate request before touching the engine
    if let Err(e) = validate_generate_request(&req) {
        return Err((StatusCode::UNPROCESSABLE_ENTITY, e));
    }

    // Ensure model is ready (handles empty state, hot-swap, lazy load)
    ensure_model_ready(&state, &req.model).await?;

    // Run inference in a blocking task — panics caught → 500 with body.
    let engine = state.engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = engine.blocking_lock();
            guard.as_mut().unwrap().generate(&req)
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
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ModelInfoExtended {
    #[serde(flatten)]
    pub info: ModelInfo,
    #[schema(example = 4)]
    pub default_steps: u32,
    #[schema(example = 3.5)]
    pub default_guidance: f64,
    #[schema(example = 1024)]
    pub default_width: u32,
    #[schema(example = 1024)]
    pub default_height: u32,
    #[schema(example = "FLUX Schnell Q8 — fast 4-step generation")]
    pub description: String,
}

#[utoipa::path(
    get,
    path = "/api/models",
    tag = "models",
    responses(
        (status = 200, description = "List of available models", body = Vec<ModelInfoExtended>),
    )
)]
async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfoExtended>> {
    let engine = state.engine.lock().await;
    let (loaded_name, is_loaded) = match engine.as_ref() {
        Some(e) => (e.model_name().to_string(), e.is_loaded()),
        None => (String::new(), false),
    };
    drop(engine);

    let config = state.config.read().await;

    // Start with known static models, merge with config-defined models.
    let mut known: std::collections::HashMap<String, ModelInfo> = model_registry::known_models()
        .into_iter()
        .map(|m| (m.name.clone(), m))
        .collect();

    // Add any models from config that aren't in the static registry.
    for (name, mcfg) in &config.models {
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
            let mut mcfg = config.model_config(&m.name);
            // Fall back to manifest defaults when config has no model-specific values
            if mcfg.default_steps.is_none() {
                if let Some(manifest) = mold_core::manifest::find_manifest(&m.name) {
                    mcfg.default_steps = Some(manifest.defaults.steps);
                    mcfg.default_guidance = Some(manifest.defaults.guidance);
                    mcfg.default_width = Some(manifest.defaults.width);
                    mcfg.default_height = Some(manifest.defaults.height);
                    if mcfg.description.is_none() {
                        mcfg.description = Some(manifest.description);
                    }
                }
            }
            ModelInfoExtended {
                default_steps: mcfg.effective_steps(&config),
                default_guidance: mcfg.effective_guidance(),
                default_width: mcfg.effective_width(&config),
                default_height: mcfg.effective_height(&config),
                description: mcfg.description.clone().unwrap_or_else(|| m.name.clone()),
                info: m,
            }
        })
        .collect();

    Json(models)
}

// ── /api/models/load ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct LoadModelBody {
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
}

#[utoipa::path(
    post,
    path = "/api/models/load",
    tag = "models",
    request_body = LoadModelBody,
    responses(
        (status = 200, description = "Model loaded successfully"),
        (status = 404, description = "Model not downloaded"),
        (status = 400, description = "Unknown model"),
        (status = 500, description = "Failed to load model"),
    )
)]
async fn load_model(
    State(state): State<AppState>,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    ensure_model_ready(&state, &body.model).await?;
    tracing::info!(model = %body.model, "model loaded via API");
    Ok(StatusCode::OK)
}

// ── /api/models/pull ──────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/models/pull",
    tag = "models",
    request_body = LoadModelBody,
    responses(
        (status = 200, description = "Model pulled successfully"),
        (status = 400, description = "Unknown model"),
        (status = 500, description = "Download failed"),
    )
)]
async fn pull_model_endpoint(
    State(state): State<AppState>,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let _guard = state.pull_lock.lock().await;

    // Re-check if already available while we waited for the lock
    {
        let config = state.config.read().await;
        if ModelPaths::resolve(&body.model, &config).is_some() {
            return Ok(format!("model '{}' already available", body.model));
        }
    }

    tracing::info!(model = %body.model, "pulling model via API");

    let (new_config, _paths) = mold_core::download::pull_and_configure(&body.model)
        .await
        .map_err(|e| {
            tracing::error!("pull failed for {}: {e}", body.model);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to pull model '{}': {e}", body.model),
            )
        })?;

    // Update shared config so subsequent load/generate can find the model
    {
        let mut config = state.config.write().await;
        *config = new_config;
    }

    tracing::info!(model = %body.model, "pull complete");
    Ok(format!("model '{}' pulled successfully", body.model))
}

// ── /api/models/unload ────────────────────────────────────────────────────────

#[utoipa::path(
    delete,
    path = "/api/models/unload",
    tag = "models",
    responses(
        (status = 200, description = "Model unloaded or no model was loaded", body = String),
    )
)]
async fn unload_model(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut engine = state.engine.lock().await;
    match engine.as_mut() {
        Some(e) if e.is_loaded() => {
            let name = e.model_name().to_string();
            e.unload();
            tracing::info!(model = %name, "model unloaded via API");
            Ok((StatusCode::OK, format!("unloaded {name}")))
        }
        _ => Ok((StatusCode::OK, "no model loaded".to_string())),
    }
}

// ── /api/status ───────────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/status",
    tag = "server",
    responses(
        (status = 200, description = "Server status", body = ServerStatus),
    )
)]
async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    let engine = state.engine.lock().await;
    let models_loaded = match engine.as_ref() {
        Some(e) if e.is_loaded() => vec![e.model_name().to_string()],
        _ => vec![],
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

#[utoipa::path(
    get,
    path = "/health",
    tag = "server",
    responses(
        (status = 200, description = "Server is healthy"),
    )
)]
async fn health() -> impl IntoResponse {
    StatusCode::OK
}

// ── /api/openapi.json ─────────────────────────────────────────────────────────

async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

// ── /api/docs ─────────────────────────────────────────────────────────────────

async fn scalar_docs() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html")],
        r#"<!DOCTYPE html>
<html>
<head>
  <title>mold API</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <script id="api-reference" data-url="/api/openapi.json"></script>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>"#,
    )
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
