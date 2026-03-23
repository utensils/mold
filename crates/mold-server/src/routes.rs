use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{delete, get, post},
    Json, Router,
};
use base64::Engine as _;
use mold_core::{
    GpuInfo, ModelInfo, ModelPaths, OutputFormat, ServerStatus, SseCompleteEvent, SseErrorEvent,
    SseProgressEvent,
};
use mold_inference::model_registry;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::StreamExt as _;
use utoipa::OpenApi;

use crate::state::AppState;

// ── ApiError — structured JSON error response ────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub code: String,
    #[serde(skip)]
    status: StatusCode,
}

impl ApiError {
    pub fn validation(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "VALIDATION_ERROR".to_string(),
            status: StatusCode::UNPROCESSABLE_ENTITY,
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "MODEL_NOT_FOUND".to_string(),
            status: StatusCode::NOT_FOUND,
        }
    }

    pub fn unknown_model(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "UNKNOWN_MODEL".to_string(),
            status: StatusCode::BAD_REQUEST,
        }
    }

    pub fn inference(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "INFERENCE_ERROR".to_string(),
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "INTERNAL_ERROR".to_string(),
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status;
        (status, Json(self)).into_response()
    }
}

/// Extract a clean, user-facing error message from an anyhow error.
/// Strips backtrace frames and internal source locations that candle
/// embeds into its Display output via `Error::bt()`.
fn clean_error_message(e: &anyhow::Error) -> String {
    let full = format!("{e}");
    // Candle backtraces start with a frame number like "   0: candle_core::..."
    // Take only lines before the first backtrace frame.
    let mut lines: Vec<&str> = Vec::new();
    for line in full.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("0:") || trimmed.starts_with("1:") {
            // Check if this looks like a backtrace frame (digit + colon + space + path)
            if trimmed.len() > 3
                && trimmed
                    .as_bytes()
                    .first()
                    .is_some_and(|b| b.is_ascii_digit())
            {
                break;
            }
        }
        // Also catch higher-numbered frames like "  10: ..."
        if trimmed.len() > 2
            && trimmed.as_bytes()[0].is_ascii_digit()
            && trimmed.contains("::")
            && trimmed.contains("at ")
        {
            break;
        }
        lines.push(line);
    }
    let msg = lines.join("\n").trim().to_string();
    if msg.is_empty() {
        // Fallback: just the root cause
        format!("{}", e.root_cause())
    } else {
        msg
    }
}

#[derive(OpenApi)]
#[openapi(
    paths(generate, generate_stream, list_models, load_model, pull_model_endpoint, unload_model, server_status, health),
    components(schemas(
        mold_core::GenerateRequest,
        mold_core::GenerateResponse,
        mold_core::ImageData,
        mold_core::OutputFormat,
        mold_core::ModelInfo,
        mold_core::ServerStatus,
        mold_core::GpuInfo,
        mold_core::SseProgressEvent,
        mold_core::SseCompleteEvent,
        mold_core::SseErrorEvent,
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
        .route("/api/generate/stream", post(generate_stream))
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

/// Internal SSE message type for the streaming channel.
enum SseMessage {
    Progress(SseProgressEvent),
    Complete(SseCompleteEvent),
    Error(SseErrorEvent),
}

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

/// Check model availability without loading. Returns:
/// - `Ok(None)` — model is already loaded and ready
/// - `Ok(Some(paths))` — model paths resolved, needs loading
/// - `Err(...)` — model not found (404) or unknown model (400)
async fn check_model_available(
    state: &AppState,
    model_name: &str,
) -> Result<Option<ModelPaths>, ApiError> {
    // Fast path: engine exists, correct model, already loaded
    {
        let engine = state.engine.lock().await;
        if let Some(ref e) = *engine {
            if e.model_name() == model_name && e.is_loaded() {
                return Ok(None);
            }
        }
    }

    // Try to resolve paths from current config
    let paths = {
        let config = state.config.read().await;
        ModelPaths::resolve(model_name, &config)
    };
    if let Some(paths) = paths {
        return Ok(Some(paths));
    }

    // Config miss — re-read from disk in case config.toml was updated externally
    {
        let fresh_config = mold_core::Config::load_or_default();
        if let Some(paths) = ModelPaths::resolve(model_name, &fresh_config) {
            let mut config = state.config.write().await;
            *config = fresh_config;
            return Ok(Some(paths));
        }
    }

    // Model paths not found — tell the client to pull or report unknown model
    if mold_core::manifest::find_manifest(model_name).is_some() {
        return Err(ApiError::not_found(format!(
            "model '{model_name}' is not downloaded. Run: mold pull {model_name}"
        )));
    }
    Err(ApiError::unknown_model(format!(
        "unknown model '{model_name}'. Run 'mold list' to see available models."
    )))
}

/// Ensure the requested model is loaded and ready for inference.
async fn ensure_model_ready(state: &AppState, model_name: &str) -> Result<(), ApiError> {
    match check_model_available(state, model_name).await? {
        Some(paths) => create_and_load_engine(state, model_name, paths, None).await,
        None => Ok(()),
    }
}

/// Create an inference engine and load it into AppState.
/// When `progress_tx` is `Some`, a progress callback is installed before
/// `engine.load()` so loading stages are streamed to the SSE client.
async fn create_and_load_engine(
    state: &AppState,
    model_name: &str,
    paths: ModelPaths,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<(), ApiError> {
    let config = state.config.read().await;
    let mut new_engine = mold_inference::create_engine(
        model_name.to_string(),
        paths,
        &config,
        mold_inference::LoadStrategy::Eager,
    )
    .map_err(|e| ApiError::internal(format!("failed to create engine for '{model_name}': {e}")))?;
    drop(config);

    // Install progress callback for SSE streaming (captures load events)
    if let Some(tx) = progress_tx {
        let tx = tx.clone();
        new_engine.set_on_progress(Box::new(move |event| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }));
    }

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
                ApiError::internal(format!("model load error: {e}"))
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
// The server always produces 1 image per request; batch looping (--batch N)
// is handled client-side by the CLI, which sends N requests with incrementing seeds.
async fn generate(
    State(state): State<AppState>,
    Json(req): Json<mold_core::GenerateRequest>,
) -> Result<impl IntoResponse, ApiError> {
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
        return Err(ApiError::validation(e));
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
        ApiError::inference("inference task failed")
    })?;

    let mut response = match result {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            tracing::error!("generation error: {e:#}");
            return Err(ApiError::inference(format!(
                "generation error: {}",
                clean_error_message(&e)
            )));
        }
        Err(panic_payload) => {
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!("inference panicked: {msg}");
            return Err(ApiError::inference(format!("inference panicked: {msg}")));
        }
    };

    let img = response.images.remove(0);
    let content_type = match img.format {
        OutputFormat::Png => "image/png",
        OutputFormat::Jpeg => "image/jpeg",
    };
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    headers.insert(
        "x-mold-seed-used",
        response.seed_used.to_string().parse().unwrap(),
    );
    Ok((headers, img.data))
}

fn validate_generate_request(req: &mold_core::GenerateRequest) -> Result<(), String> {
    mold_core::validate_generate_request(req)
}

// ── /api/generate/stream (SSE) ───────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/generate/stream",
    tag = "generation",
    request_body = mold_core::GenerateRequest,
    responses(
        (status = 200, description = "SSE event stream with progress and result"),
        (status = 404, description = "Model not downloaded"),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Inference error"),
    )
)]
async fn generate_stream(
    State(state): State<AppState>,
    Json(req): Json<mold_core::GenerateRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    tracing::info!(
        model = %req.model,
        prompt = %req.prompt,
        "generate/stream request"
    );

    // Validate before starting the SSE stream (returns HTTP error, not SSE event)
    if let Err(e) = validate_generate_request(&req) {
        return Err(ApiError::validation(e));
    }

    // Check model availability (404/400 returned as HTTP errors before SSE starts)
    let model_paths = check_model_available(&state, &req.model).await?;

    // Create SSE channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();

    // Spawn background task for model loading + inference
    let bg_tx = tx;
    tokio::spawn(async move {
        // Load model if needed (with progress events)
        if let Some(paths) = model_paths {
            if let Err(api_err) =
                create_and_load_engine(&state, &req.model, paths, Some(&bg_tx)).await
            {
                let _ = bg_tx.send(SseMessage::Error(SseErrorEvent {
                    message: api_err.error,
                }));
                return;
            }
        }

        // Run inference in blocking thread with progress callback
        let engine = state.engine.clone();
        let gen_tx = bg_tx.clone();
        let gen_req = req.clone();
        let result = tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut guard = engine.blocking_lock();
                let e = guard.as_mut().unwrap();
                // Install progress callback for the generate phase
                let progress_tx = gen_tx.clone();
                e.set_on_progress(Box::new(move |event| {
                    let _ = progress_tx.send(SseMessage::Progress(progress_to_sse(event)));
                }));
                e.generate(&gen_req)
            }))
        })
        .await;

        match result {
            Ok(Ok(Ok(mut response))) => {
                let img = response.images.remove(0);
                let _ = bg_tx.send(SseMessage::Complete(SseCompleteEvent {
                    image: base64::engine::general_purpose::STANDARD.encode(&img.data),
                    format: img.format,
                    width: img.width,
                    height: img.height,
                    seed_used: response.seed_used,
                    generation_time_ms: response.generation_time_ms,
                }));
            }
            Ok(Ok(Err(e))) => {
                tracing::error!("generation error: {e:#}");
                let _ = bg_tx.send(SseMessage::Error(SseErrorEvent {
                    message: format!("generation error: {}", clean_error_message(&e)),
                }));
            }
            Ok(Err(panic_payload)) => {
                let msg = panic_payload
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                tracing::error!("inference panicked: {msg}");
                let _ = bg_tx.send(SseMessage::Error(SseErrorEvent {
                    message: format!("inference panicked: {msg}"),
                }));
            }
            Err(join_err) => {
                tracing::error!("inference task join error: {join_err:?}");
                let _ = bg_tx.send(SseMessage::Error(SseErrorEvent {
                    message: "inference task failed".to_string(),
                }));
            }
        }
    });

    // Build SSE stream from the channel receiver
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|msg| {
        let event = match msg {
            SseMessage::Progress(p) => SseEvent::default()
                .event("progress")
                .data(serde_json::to_string(&p).unwrap()),
            SseMessage::Complete(c) => SseEvent::default()
                .event("complete")
                .data(serde_json::to_string(&c).unwrap()),
            SseMessage::Error(e) => SseEvent::default()
                .event("error")
                .data(serde_json::to_string(&e).unwrap()),
        };
        Ok::<_, Infallible>(event)
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
}

// ── /api/models ───────────────────────────────────────────────────────────────

/// Extended model info including generation defaults.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ModelInfoExtended {
    #[serde(flatten)]
    pub info: ModelInfo,
    /// Whether the model files are downloaded and available for inference.
    pub downloaded: bool,
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
            .all_file_paths()
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as f32 / 1_073_741_824.0)
            .sum::<f32>();
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
                        mcfg.description = Some(manifest.description.clone());
                    }
                }
            }
            let downloaded = config.models.contains_key(&m.name);
            ModelInfoExtended {
                downloaded,
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
) -> Result<impl IntoResponse, ApiError> {
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
        (status = 200, description = "Model pulled (SSE stream or plain text)"),
        (status = 400, description = "Unknown model"),
        (status = 500, description = "Download failed"),
    )
)]
async fn pull_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, ApiError> {
    let wants_sse = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));

    if !wants_sse {
        // Legacy: blocking pull with plain text response
        return pull_model_blocking(state, body.model)
            .await
            .map(PullResponse::Text);
    }

    // SSE streaming pull
    let model = body.model.clone();

    // Validate model exists in manifest before starting SSE
    if mold_core::manifest::find_manifest(&mold_core::manifest::resolve_model_name(&model))
        .is_none()
    {
        return Err(ApiError::unknown_model(format!(
            "unknown model '{model}'. Run 'mold list' to see available models."
        )));
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();

    tokio::spawn(async move {
        // Acquire pull lock inside the task so the SSE stream starts immediately
        let _guard = state.pull_lock.lock().await;

        // Re-check availability after acquiring lock
        {
            let config = state.config.read().await;
            if ModelPaths::resolve(&model, &config).is_some() {
                let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                    model: model.clone(),
                }));
                return;
            }
        }

        tracing::info!(model = %model, "pulling model via API (SSE)");

        let progress_tx = tx.clone();
        let model_for_cb = model.clone();
        let callback =
            std::sync::Arc::new(move |event: mold_core::download::DownloadProgressEvent| {
                let sse_event = match event {
                    mold_core::download::DownloadProgressEvent::FileStart {
                        filename,
                        file_index,
                        total_files,
                        size_bytes,
                    } => SseProgressEvent::DownloadProgress {
                        filename,
                        file_index,
                        total_files,
                        bytes_downloaded: 0,
                        bytes_total: size_bytes,
                    },
                    mold_core::download::DownloadProgressEvent::FileProgress {
                        filename,
                        file_index,
                        bytes_downloaded,
                        bytes_total,
                    } => SseProgressEvent::DownloadProgress {
                        filename,
                        file_index,
                        total_files: 0, // not available here, client tracks from FileStart
                        bytes_downloaded,
                        bytes_total,
                    },
                    mold_core::download::DownloadProgressEvent::FileDone {
                        filename,
                        file_index,
                        total_files,
                    } => SseProgressEvent::DownloadDone {
                        filename,
                        file_index,
                        total_files,
                    },
                };
                let _ = progress_tx.send(SseMessage::Progress(sse_event));
            });

        match mold_core::download::pull_and_configure_with_callback(&model, callback).await {
            Ok((new_config, _paths)) => {
                {
                    let mut config = state.config.write().await;
                    *config = new_config;
                }
                tracing::info!(model = %model, "pull complete");
                let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                    model: model_for_cb,
                }));
            }
            Err(e) => {
                tracing::error!("pull failed for {}: {e}", model);
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: format!("failed to pull model '{}': {e}", model),
                }));
            }
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|msg| {
        let event = match msg {
            SseMessage::Progress(p) => SseEvent::default()
                .event("progress")
                .data(serde_json::to_string(&p).unwrap()),
            SseMessage::Complete(c) => SseEvent::default()
                .event("complete")
                .data(serde_json::to_string(&c).unwrap()),
            SseMessage::Error(e) => SseEvent::default()
                .event("error")
                .data(serde_json::to_string(&e).unwrap()),
        };
        Ok::<_, Infallible>(event)
    });

    Ok(PullResponse::Sse(
        Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15))
                    .text("ping"),
            )
            .into_response(),
    ))
}

/// Legacy blocking pull — returns plain text.
async fn pull_model_blocking(state: AppState, model: String) -> Result<String, ApiError> {
    let _guard = state.pull_lock.lock().await;

    {
        let config = state.config.read().await;
        if ModelPaths::resolve(&model, &config).is_some() {
            return Ok(format!("model '{}' already available", model));
        }
    }

    tracing::info!(model = %model, "pulling model via API");

    let (new_config, _paths) = mold_core::download::pull_and_configure(&model)
        .await
        .map_err(|e| {
            tracing::error!("pull failed for {}: {e}", model);
            ApiError::internal(format!("failed to pull model '{}': {e}", model))
        })?;

    {
        let mut config = state.config.write().await;
        *config = new_config;
    }

    tracing::info!(model = %model, "pull complete");
    Ok(format!("model '{}' pulled successfully", model))
}

/// Response type that can be either SSE stream or plain text.
enum PullResponse {
    Sse(axum::response::Response),
    Text(String),
}

impl IntoResponse for PullResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            PullResponse::Sse(resp) => resp,
            PullResponse::Text(text) => text.into_response(),
        }
    }
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
async fn unload_model(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_error_message_strips_backtrace() {
        let err = anyhow::anyhow!(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")\n\
             \x20  0: candle_core::error::Error::bt\n\
             \x20           at /home/user/.cargo/git/candle/src/error.rs:264:25\n\
             \x20  1: <core::result::Result<O,E> as candle_core::cuda_backend::error::WrapErr<O>>::w\n\
             \x20           at /home/user/.cargo/git/candle/src/cuda_backend/error.rs:60:65"
        );
        let msg = clean_error_message(&err);
        assert_eq!(
            msg,
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        );
    }

    #[test]
    fn clean_error_message_preserves_simple_error() {
        let err = anyhow::anyhow!("model not found: flux-dev:q4");
        let msg = clean_error_message(&err);
        assert_eq!(msg, "model not found: flux-dev:q4");
    }

    #[test]
    fn clean_error_message_preserves_multiline_without_backtrace() {
        let err = anyhow::anyhow!("validation failed\nprompt is empty\nsteps must be > 0");
        let msg = clean_error_message(&err);
        assert_eq!(msg, "validation failed\nprompt is empty\nsteps must be > 0");
    }

    #[test]
    fn clean_error_message_strips_high_numbered_frames() {
        let err = anyhow::anyhow!(
            "some error\n\
             \x20 10: tokio::runtime::task::core::Core<T,S>::poll at /home/user/.cargo/tokio/src/core.rs:375\n\
             \x20 11: std::panicking::catch_unwind at /nix/store/rust/src/panicking.rs:544"
        );
        let msg = clean_error_message(&err);
        assert_eq!(msg, "some error");
    }

    #[test]
    fn clean_error_message_empty_fallback() {
        // An error whose Display starts immediately with a backtrace-like line
        let err = anyhow::anyhow!("0: candle_core::error::Error::bt at /some/path.rs:10:5");
        let msg = clean_error_message(&err);
        // Should fall back to root_cause since all lines look like backtrace
        assert!(!msg.is_empty());
    }
}
