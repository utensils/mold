use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{delete, get, post},
    Json, Router,
};
use base64::Engine as _;
use mold_core::{
    ActiveGenerationStatus, GpuInfo, ModelInfoExtended, OutputFormat, ServerStatus, SseErrorEvent,
    SseProgressEvent,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::StreamExt as _;
use utoipa::OpenApi;

use crate::model_manager;
use crate::state::{AppState, GenerationJob, SseMessage};

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

    pub fn insufficient_memory(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "INSUFFICIENT_MEMORY".to_string(),
            status: StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status;
        (status, Json(self)).into_response()
    }
}

// Re-export for tests — the canonical implementation lives in queue.rs.
#[cfg(test)]
use crate::queue::clean_error_message;

#[derive(OpenApi)]
#[openapi(
    paths(generate, generate_stream, expand_prompt, list_models, load_model, pull_model_endpoint, unload_model, server_status, health),
    components(schemas(
        mold_core::GenerateRequest,
        mold_core::GenerateResponse,
        mold_core::ExpandRequest,
        mold_core::ExpandResponse,
        mold_core::ImageData,
        mold_core::OutputFormat,
        mold_core::ModelInfo,
        mold_core::ServerStatus,
        mold_core::ActiveGenerationStatus,
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
        .route("/api/expand", post(expand_prompt))
        .route("/api/models", get(list_models))
        .route("/api/models/load", post(load_model))
        .route("/api/models/pull", post(pull_model_endpoint))
        .route("/api/models/unload", delete(unload_model))
        .route("/api/gallery", get(list_gallery))
        .route(
            "/api/gallery/image/:filename",
            get(get_gallery_image).delete(delete_gallery_image),
        )
        .route(
            "/api/gallery/thumbnail/:filename",
            get(get_gallery_thumbnail),
        )
        .route("/api/upscale", post(upscale))
        .route("/api/upscale/stream", post(upscale_stream))
        .route("/api/status", get(server_status))
        .route("/api/shutdown", post(shutdown_server))
        .route("/health", get(health))
        .with_state(state)
        .route("/api/openapi.json", get(openapi_json))
        .route("/api/docs", get(scalar_docs))
}

// ── Model readiness ──────────────────────────────────────────────────────────

fn sse_message_to_event(msg: SseMessage) -> SseEvent {
    fn serialize_event<T: Serialize>(event_name: &str, payload: &T) -> SseEvent {
        match serde_json::to_string(payload) {
            Ok(data) => SseEvent::default().event(event_name).data(data),
            Err(err) => SseEvent::default().event("error").data(
                serde_json::json!({
                    "message": format!("failed to serialize SSE payload: {err}")
                })
                .to_string(),
            ),
        }
    }

    match msg {
        SseMessage::Progress(payload) => serialize_event("progress", &payload),
        SseMessage::Complete(payload) => serialize_event("complete", &payload),
        SseMessage::UpscaleComplete(payload) => serialize_event("complete", &payload),
        SseMessage::Error(payload) => serialize_event("error", &payload),
    }
}

#[cfg(test)]
fn save_image_to_dir(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return;
    }
    // Use milliseconds for server-side filenames to avoid overwrites when
    // concurrent requests finish in the same second.
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let ext = img.format.to_string();
    let filename =
        mold_core::default_output_filename(model, timestamp_ms, &ext, batch_size, img.index);
    let path = dir.join(&filename);
    match std::fs::write(&path, &img.data) {
        Ok(()) => tracing::info!("saved image to {}", path.display()),
        Err(e) => tracing::warn!("failed to save image to {}: {e}", path.display()),
    }
}

// ── Shared pre-queue validation ───────────────────────────────────────────────

/// Validate a generate request and resolve server-side defaults.
///
/// Performs the identical pre-queue checks used by both `generate` and
/// `generate_stream`: applies the default metadata setting, validates the
/// request, checks model availability, and resolves the output directory.
async fn prepare_generation(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Result<(Option<std::path::PathBuf>, Option<String>), ApiError> {
    apply_default_metadata_setting(state, request).await;

    // Expand prompt if requested (before validation, so the expanded prompt gets validated)
    maybe_expand_prompt(state, request).await?;

    if let Err(e) = validate_generate_request(request) {
        return Err(ApiError::validation(e));
    }

    let _ = model_manager::check_model_available(state, &request.model).await?;

    let (output_dir, dim_warning) = {
        let config = state.config.read().await;
        let output_dir = if config.is_output_disabled() {
            None
        } else {
            Some(config.effective_output_dir())
        };
        let family = config.resolved_model_config(&request.model).family;
        let dim_warning = family
            .as_deref()
            .and_then(|f| mold_core::dimension_warning(request.width, request.height, f));
        (output_dir, dim_warning)
    };

    Ok((output_dir, dim_warning))
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
    Json(mut req): Json<mold_core::GenerateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let (output_dir, dim_warning) = prepare_generation(&state, &mut req).await?;

    tracing::info!(
        model = %req.model,
        prompt = %req.prompt,
        width = req.width,
        height = req.height,
        steps = req.steps,
        guidance = req.guidance,
        seed = ?req.seed,
        format = %req.output_format,
        lora = ?req.lora.as_ref().map(|l| &l.path),
        lora_scale = ?req.lora.as_ref().map(|l| l.scale),
        "generate request"
    );

    // Submit to generation queue
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let job = GenerationJob {
        request: req,
        progress_tx: None,
        result_tx,
        output_dir,
    };

    let _position = state.queue.submit(job).await.map_err(ApiError::internal)?;

    // Wait for the queue worker to process the job
    let result = result_rx
        .await
        .map_err(|_| ApiError::internal("generation queue worker dropped the job"))?;

    match result {
        Ok(job_result) => {
            let img = job_result.image;
            let response = job_result.response;
            let content_type = match img.format {
                OutputFormat::Png => HeaderValue::from_static("image/png"),
                OutputFormat::Jpeg => HeaderValue::from_static("image/jpeg"),
            };
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, content_type);
            headers.insert(
                "x-mold-seed-used",
                HeaderValue::from_str(&response.seed_used.to_string()).map_err(|e| {
                    ApiError::internal(format!("failed to serialize seed header: {e}"))
                })?,
            );
            if let Some(warning) = dim_warning {
                match HeaderValue::from_str(&warning.replace('\n', " ")) {
                    Ok(val) => {
                        headers.insert("x-mold-dimension-warning", val);
                    }
                    Err(e) => {
                        tracing::warn!("dimension warning could not be encoded as header: {e}");
                    }
                }
            }
            Ok((headers, img.data))
        }
        Err(err_msg) => Err(ApiError::inference(err_msg)),
    }
}

fn validate_generate_request(req: &mold_core::GenerateRequest) -> Result<(), String> {
    mold_core::validate_generate_request(req)
}

async fn apply_default_metadata_setting(state: &AppState, req: &mut mold_core::GenerateRequest) {
    if req.embed_metadata.is_some() {
        return;
    }

    let config = state.config.read().await;
    req.embed_metadata = Some(config.effective_embed_metadata(None));
}

/// Apply prompt expansion if `expand: true` is set on a generate request.
async fn maybe_expand_prompt(
    state: &AppState,
    req: &mut mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    if req.expand != Some(true) {
        return Ok(());
    }

    let config = state.config.read().await;
    let expand_settings = config.expand.clone().with_env_overrides();

    // Resolve model family for prompt style
    let model_family = config
        .resolved_model_config(&req.model)
        .family
        .or_else(|| mold_core::manifest::find_manifest(&req.model).map(|m| m.family.clone()))
        .unwrap_or_else(|| {
            tracing::warn!(
                model = %req.model,
                "could not resolve model family for prompt expansion, defaulting to \"flux\""
            );
            "flux".to_string()
        });

    let expand_config = expand_settings.to_expand_config(&model_family, 1);
    let original_prompt = req.prompt.clone();

    // Drop config lock before blocking
    drop(config);

    let expander = create_server_expander(&expand_settings)?;
    let result =
        tokio::task::spawn_blocking(move || expander.expand(&original_prompt, &expand_config))
            .await
            .map_err(|e| ApiError::internal(format!("expand task failed: {e}")))?
            .map_err(|e| ApiError::internal(format!("prompt expansion failed: {e}")))?;

    if let Some(expanded) = result.expanded.first() {
        req.original_prompt = Some(req.prompt.clone());
        req.prompt = expanded.clone();
    }

    Ok(())
}

/// Create the appropriate expander for server-side use.
fn create_server_expander(
    settings: &mold_core::ExpandSettings,
) -> Result<Box<dyn mold_core::PromptExpander>, ApiError> {
    if let Some(api_expander) = settings.create_api_expander() {
        return Ok(Box::new(api_expander));
    }

    #[cfg(feature = "expand")]
    {
        let config = mold_core::Config::load_or_default();
        if let Some(local) =
            mold_inference::expand::LocalExpander::from_config(&config, Some(&settings.model))
        {
            return Ok(Box::new(local));
        }
        return Err(ApiError::validation(
            "local expand model not found — run: mold pull qwen3-expand".to_string(),
        ));
    }

    #[cfg(not(feature = "expand"))]
    {
        Err(ApiError::validation(
            "local prompt expansion not available — built without expand feature. \
             Configure an API backend in [expand] settings."
                .to_string(),
        ))
    }
}

// ── /api/expand ──────────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/expand",
    tag = "generation",
    request_body = mold_core::ExpandRequest,
    responses(
        (status = 200, description = "Expanded prompt(s)", body = mold_core::ExpandResponse),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Expansion failed"),
    )
)]
async fn expand_prompt(
    State(state): State<AppState>,
    Json(req): Json<mold_core::ExpandRequest>,
) -> Result<Json<mold_core::ExpandResponse>, ApiError> {
    if req.variations == 0 || req.variations > mold_core::expand::MAX_VARIATIONS {
        return Err(ApiError::validation(format!(
            "variations must be between 1 and {}",
            mold_core::expand::MAX_VARIATIONS,
        )));
    }

    let config = state.config.read().await;
    let expand_settings = config.expand.clone().with_env_overrides();
    let expand_config = expand_settings.to_expand_config(&req.model_family, req.variations);
    let prompt = req.prompt.clone();
    drop(config);

    let expander = create_server_expander(&expand_settings)?;
    let result = tokio::task::spawn_blocking(move || expander.expand(&prompt, &expand_config))
        .await
        .map_err(|e| ApiError::internal(format!("expand task failed: {e}")))?
        .map_err(|e| ApiError::internal(format!("prompt expansion failed: {e}")))?;

    Ok(Json(mold_core::ExpandResponse {
        original: req.prompt,
        expanded: result.expanded,
    }))
}

// ── /api/upscale ────────────────────────────────────────────────────────────

async fn upscale(
    State(state): State<AppState>,
    Json(req): Json<mold_core::UpscaleRequest>,
) -> Result<Json<mold_core::UpscaleResponse>, ApiError> {
    if let Err(msg) = mold_core::validate_upscale_request(&req) {
        return Err(ApiError::validation(msg));
    }

    let config = state.config.read().await;
    let model_name = mold_core::manifest::resolve_model_name(&req.model);

    // Get model weights path from config
    let weights_path = config
        .models
        .get(&model_name)
        .and_then(|c| c.transformer.as_ref())
        .ok_or_else(|| {
            ApiError::not_found(format!(
                "upscaler model '{}' not downloaded. Pull it first with: mold pull {}",
                model_name, model_name
            ))
        })?;
    let weights_path = std::path::PathBuf::from(weights_path);
    let model_name_owned = model_name.clone();
    drop(config);

    let upscaler_cache = state.upscaler_cache.clone();
    let resp =
        tokio::task::spawn_blocking(move || -> anyhow::Result<mold_core::UpscaleResponse> {
            let mut cache = upscaler_cache.lock().unwrap_or_else(|e| e.into_inner());

            // Reuse cached engine if same model
            let needs_new = cache
                .as_ref()
                .is_none_or(|e| e.model_name() != model_name_owned);
            if needs_new {
                let new_engine = mold_inference::create_upscale_engine(
                    model_name_owned,
                    weights_path,
                    mold_inference::LoadStrategy::Eager,
                )?;
                *cache = Some(new_engine);
            }

            cache.as_mut().unwrap().upscale(&req)
        })
        .await
        .map_err(|e| ApiError::internal(format!("upscale task panicked: {e}")))?
        .map_err(|e| ApiError::internal(format!("upscale failed: {e}")))?;

    Ok(Json(resp))
}

// ── /api/upscale/stream (SSE) ──────────────────────────────────────────────

async fn upscale_stream(
    State(state): State<AppState>,
    Json(req): Json<mold_core::UpscaleRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    if let Err(msg) = mold_core::validate_upscale_request(&req) {
        return Err(ApiError::validation(msg));
    }

    let config = state.config.read().await;
    let model_name = mold_core::manifest::resolve_model_name(&req.model);

    let weights_path = config
        .models
        .get(&model_name)
        .and_then(|c| c.transformer.as_ref())
        .ok_or_else(|| {
            ApiError::not_found(format!(
                "upscaler model '{}' not downloaded. Pull it first with: mold pull {}",
                model_name, model_name
            ))
        })?;
    let weights_path = std::path::PathBuf::from(weights_path);
    let model_name_owned = model_name.clone();
    drop(config);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();
    let upscaler_cache = state.upscaler_cache.clone();

    tokio::task::spawn_blocking(move || {
        let mut cache = upscaler_cache.lock().unwrap();

        let needs_new = cache
            .as_ref()
            .is_none_or(|e| e.model_name() != model_name_owned);
        if needs_new {
            let _ = tx.send(SseMessage::Progress(
                mold_core::SseProgressEvent::StageStart {
                    name: "Loading upscaler model".to_string(),
                },
            ));
            match mold_inference::create_upscale_engine(
                model_name_owned,
                weights_path,
                mold_inference::LoadStrategy::Eager,
            ) {
                Ok(new_engine) => {
                    *cache = Some(new_engine);
                }
                Err(e) => {
                    let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent {
                        message: format!("failed to load upscaler: {e}"),
                    }));
                    return;
                }
            }
        }

        let engine = cache.as_mut().unwrap();

        // Install progress callback for tile-by-tile progress
        let tx_progress = tx.clone();
        engine.set_on_progress(Box::new(move |event| {
            let sse_event: mold_core::SseProgressEvent = event.into();
            let _ = tx_progress.send(SseMessage::Progress(sse_event));
        }));

        match engine.upscale(&req) {
            Ok(resp) => {
                let image_b64 = base64::engine::general_purpose::STANDARD.encode(&resp.image.data);
                let _ = tx.send(SseMessage::UpscaleComplete(
                    mold_core::SseUpscaleCompleteEvent {
                        image: image_b64,
                        format: resp.image.format,
                        model: resp.model,
                        scale_factor: resp.scale_factor,
                        original_width: resp.original_width,
                        original_height: resp.original_height,
                        upscale_time_ms: resp.upscale_time_ms,
                    },
                ));
            }
            Err(e) => {
                let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent {
                    message: format!("upscale failed: {e}"),
                }));
            }
        }

        engine.clear_on_progress();
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, Infallible>(sse_message_to_event(msg)));

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
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
    Json(mut req): Json<mold_core::GenerateRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let (output_dir, dim_warning) = prepare_generation(&state, &mut req).await?;

    tracing::info!(
        model = %req.model,
        prompt = %req.prompt,
        "generate/stream request"
    );

    // Create SSE channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();

    // Send dimension warning before queuing so the client sees it early
    if let Some(warning) = dim_warning {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::Info {
            message: warning,
        }));
    }

    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let job = GenerationJob {
        request: req,
        progress_tx: Some(tx.clone()),
        result_tx,
        output_dir,
    };

    let position = state.queue.submit(job).await.map_err(ApiError::internal)?;

    // Send initial queue position to the client
    let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued { position }));

    // Hold `tx` alive in a background task until the job completes, so the SSE
    // stream never closes prematurely even if the queue worker hasn't received
    // the job yet.
    tokio::spawn(async move {
        let _ = result_rx.await;
        drop(tx); // closes the SSE stream
    });

    // Build SSE stream from the channel receiver.
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, Infallible>(sse_message_to_event(msg)));

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
}

// ── /api/models ───────────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/models",
    tag = "models",
    responses(
        (status = 200, description = "List of available models", body = Vec<ModelInfoExtended>),
    )
)]
async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfoExtended>> {
    Json(model_manager::list_models(&state).await)
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
    model_manager::ensure_model_ready(&state, &body.model, None).await?;
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
        let progress_tx = tx.clone();
        let model_for_cb = model.clone();
        let callback =
            std::sync::Arc::new(move |event: mold_core::download::DownloadProgressEvent| {
                let sse_event = match event {
                    mold_core::download::DownloadProgressEvent::Status { message } => {
                        SseProgressEvent::Info { message }
                    }
                    mold_core::download::DownloadProgressEvent::FileStart {
                        filename,
                        file_index,
                        total_files,
                        size_bytes,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    } => SseProgressEvent::DownloadProgress {
                        filename,
                        file_index,
                        total_files,
                        bytes_downloaded: 0,
                        bytes_total: size_bytes,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    },
                    mold_core::download::DownloadProgressEvent::FileProgress {
                        filename,
                        file_index,
                        bytes_downloaded,
                        bytes_total,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    } => SseProgressEvent::DownloadProgress {
                        filename,
                        file_index,
                        total_files: 0,
                        bytes_downloaded,
                        bytes_total,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    },
                    mold_core::download::DownloadProgressEvent::FileDone {
                        filename,
                        file_index,
                        total_files,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    } => SseProgressEvent::DownloadDone {
                        filename,
                        file_index,
                        total_files,
                        batch_bytes_downloaded,
                        batch_bytes_total,
                        batch_elapsed_ms,
                    },
                };
                let _ = progress_tx.send(SseMessage::Progress(sse_event));
            });

        match model_manager::pull_model(&state, &model, Some(callback)).await {
            Ok(model_manager::PullStatus::AlreadyAvailable) => {
                let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                    model: model_for_cb,
                }));
            }
            Ok(model_manager::PullStatus::Pulled) => {
                let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                    model: model_for_cb,
                }));
            }
            Err(e) => {
                let _ = tx.send(SseMessage::Error(SseErrorEvent { message: e.error }));
            }
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, Infallible>(sse_message_to_event(msg)));

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
    match model_manager::pull_model(&state, &model, None).await? {
        model_manager::PullStatus::AlreadyAvailable => {
            Ok(format!("model '{}' already available", model))
        }
        model_manager::PullStatus::Pulled => Ok(format!("model '{}' pulled successfully", model)),
    }
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
    Ok((StatusCode::OK, model_manager::unload_model(&state).await))
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
    let snapshot = state.engine_snapshot.read().await.clone();
    let models_loaded = match (snapshot.model_name, snapshot.is_loaded) {
        (Some(model_name), true) => vec![model_name],
        _ => vec![],
    };
    let current_generation = state
        .active_generation
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|active| ActiveGenerationStatus {
            model: active.model.clone(),
            prompt_sha256: active.prompt_sha256.clone(),
            started_at_unix_ms: active.started_at_unix_ms,
            elapsed_ms: active.started_at.elapsed().as_millis() as u64,
        });
    let busy = current_generation.is_some();

    Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: if mold_core::build_info::GIT_SHA == "unknown" {
            None
        } else {
            Some(mold_core::build_info::GIT_SHA.to_string())
        },
        build_date: if mold_core::build_info::BUILD_DATE == "unknown" {
            None
        } else {
            Some(mold_core::build_info::BUILD_DATE.to_string())
        },
        models_loaded,
        busy,
        current_generation,
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

// ── /api/shutdown ─────────────────────────────────────────────────────────────

/// Trigger graceful server shutdown.
///
/// When API key auth is enabled, the auth middleware protects this endpoint.
/// When auth is disabled, only requests from loopback addresses (127.0.0.1, ::1)
/// are accepted to prevent remote shutdown.
#[utoipa::path(
    post,
    path = "/api/shutdown",
    tag = "server",
    responses(
        (status = 200, description = "Shutdown initiated"),
        (status = 403, description = "Forbidden — remote shutdown requires API key auth"),
    )
)]
async fn shutdown_server(State(state): State<AppState>, request: Request) -> impl IntoResponse {
    // When auth is disabled (no AuthState extension or AuthState is None),
    // restrict shutdown to loopback addresses only.
    let auth_enabled = request
        .extensions()
        .get::<crate::auth::AuthState>()
        .is_some_and(|s| s.is_some());

    if !auth_enabled {
        let is_loopback = request
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip().is_loopback())
            .unwrap_or(false);
        if !is_loopback {
            return (
                StatusCode::FORBIDDEN,
                "shutdown requires API key auth or localhost access\n",
            );
        }
    }

    tracing::info!("shutdown requested via API");
    if let Some(tx) = state.shutdown_tx.lock().await.take() {
        let _ = tx.send(());
    }
    (StatusCode::OK, "shutdown initiated\n")
}

// ── /api/gallery ──────────────────────────────────────────────────────────────

/// List gallery images from the server's output directory.
/// Returns metadata from PNG `mold:parameters` chunks, sorted newest-first.
async fn list_gallery(
    State(state): State<AppState>,
) -> Result<Json<Vec<mold_core::GalleryImage>>, ApiError> {
    let config = state.config.read().await;
    if config.is_output_disabled() {
        return Ok(Json(Vec::new()));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    if !output_dir.is_dir() {
        return Ok(Json(Vec::new()));
    }

    let images = tokio::task::spawn_blocking(move || scan_gallery_dir(&output_dir))
        .await
        .map_err(|e| ApiError::internal(format!("gallery scan failed: {e}")))?;

    Ok(Json(images))
}

/// Serve a gallery image file by filename.
async fn get_gallery_image(
    State(state): State<AppState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let config = state.config.read().await;
    if config.is_output_disabled() {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    // Sanitize: prevent directory traversal
    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    let path = output_dir.join(&clean_name);
    if !path.is_file() {
        return Err(ApiError::not_found(format!(
            "image not found: {clean_name}"
        )));
    }

    let data = tokio::fs::read(&path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to read image: {e}")))?;

    let content_type = if clean_name.ends_with(".png") {
        "image/png"
    } else if clean_name.ends_with(".jpg") || clean_name.ends_with(".jpeg") {
        "image/jpeg"
    } else {
        "application/octet-stream"
    };

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));

    Ok((headers, data))
}

/// Delete a gallery image and its server-side thumbnail.
async fn delete_gallery_image(
    State(state): State<AppState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let config = state.config.read().await;
    if config.is_output_disabled() {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    let path = output_dir.join(&clean_name);
    if path.is_file() {
        std::fs::remove_file(&path)
            .map_err(|e| ApiError::internal(format!("failed to delete image: {e}")))?;
    }

    // Also remove server-side thumbnail
    let thumb_path = server_thumbnail_dir().join(&clean_name);
    let _ = std::fs::remove_file(&thumb_path);

    Ok(StatusCode::NO_CONTENT)
}

/// Serve a thumbnail for a gallery image. Generated on-demand and cached
/// at ~/.mold/cache/thumbnails/ on the server side.
async fn get_gallery_thumbnail(
    State(state): State<AppState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let config = state.config.read().await;
    if config.is_output_disabled() {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    let source_path = output_dir.join(&clean_name);
    if !source_path.is_file() {
        return Err(ApiError::not_found(format!(
            "image not found: {clean_name}"
        )));
    }

    // Check if server-side thumbnail already exists
    let thumb_dir = server_thumbnail_dir();
    let thumb_path = thumb_dir.join(&clean_name);

    if !thumb_path.is_file() {
        // Generate thumbnail on-demand
        let source = source_path.clone();
        let dest = thumb_path.clone();
        tokio::task::spawn_blocking(move || generate_server_thumbnail(&source, &dest))
            .await
            .map_err(|e| ApiError::internal(format!("thumbnail generation failed: {e}")))?
            .map_err(|e| ApiError::internal(format!("thumbnail generation failed: {e}")))?;
    }

    let data = tokio::fs::read(&thumb_path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to read thumbnail: {e}")))?;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));

    Ok((headers, data))
}

/// Server-side thumbnail cache directory.
fn server_thumbnail_dir() -> std::path::PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails")
}

/// Generate a 256x256 max thumbnail from source image.
fn generate_server_thumbnail(
    source: &std::path::Path,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    let img = image::open(source)?;
    let thumb = img.thumbnail(256, 256);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    thumb.save(dest)?;
    Ok(())
}

/// Pre-generate thumbnails for all gallery images on server startup.
pub fn spawn_thumbnail_warmup(config: &mold_core::Config) {
    let output_dir = config.effective_output_dir();
    std::thread::spawn(move || {
        if !output_dir.is_dir() {
            return;
        }
        let thumb_dir = server_thumbnail_dir();
        let walker = walkdir::WalkDir::new(&output_dir).max_depth(1).into_iter();
        for entry in walker.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
                continue;
            }
            let filename = path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();
            let thumb_path = thumb_dir.join(&filename);
            if !thumb_path.is_file() {
                if let Err(e) = generate_server_thumbnail(path, &thumb_path) {
                    tracing::warn!("failed to generate thumbnail for {}: {e}", path.display());
                }
            }
        }
        tracing::info!("thumbnail warmup complete");
    });
}

/// Scan a directory for image files with mold metadata.
fn scan_gallery_dir(dir: &std::path::Path) -> Vec<mold_core::GalleryImage> {
    let mut images = Vec::new();

    let walker = walkdir::WalkDir::new(dir).max_depth(1).into_iter();
    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        if !matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            continue;
        }

        let timestamp = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        let meta = if ext.as_deref() == Some("png") {
            read_png_metadata(path)
        } else {
            read_jpeg_metadata(path)
        };

        if let Some(meta) = meta {
            images.push(mold_core::GalleryImage {
                filename,
                metadata: meta,
                timestamp,
            });
        }
    }

    images.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    images
}

/// Read OutputMetadata from a PNG file's text chunks.
fn read_png_metadata(path: &std::path::Path) -> Option<mold_core::OutputMetadata> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().ok()?;
    let info = reader.info();

    for chunk in &info.uncompressed_latin1_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&chunk.text) {
                return Some(meta);
            }
        }
    }
    for chunk in &info.utf8_text {
        if chunk.keyword == "mold:parameters" {
            if let Ok(text) = chunk.get_text() {
                if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(&text) {
                    return Some(meta);
                }
            }
        }
    }
    None
}

/// Read OutputMetadata from a JPEG file's COM marker.
fn read_jpeg_metadata(path: &std::path::Path) -> Option<mold_core::OutputMetadata> {
    let data = std::fs::read(path).ok()?;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        match marker {
            // Standalone markers (no length field): SOI, EOI, RST0-7, TEM
            0xD8 | 0x01 => {
                i += 2;
            }
            0xD9 => break, // EOI — end of image
            0xD0..=0xD7 => {
                i += 2; // RST markers
            }
            // COM marker — check for mold:parameters
            0xFE => {
                if i + 3 >= data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                if len < 2 || i + 2 + len > data.len() {
                    break;
                }
                let comment = &data[i + 4..i + 2 + len];
                if let Ok(text) = std::str::from_utf8(comment) {
                    if let Some(json) = text.strip_prefix("mold:parameters ") {
                        if let Ok(meta) = serde_json::from_str::<mold_core::OutputMetadata>(json) {
                            return Some(meta);
                        }
                    }
                }
                i += 2 + len;
            }
            // All other markers have a 2-byte length field
            _ => {
                if i + 3 >= data.len() {
                    break;
                }
                let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                if len < 2 || i + 2 + len > data.len() {
                    break;
                }
                i += 2 + len;
            }
        }
    }
    None
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

    #[test]
    fn save_image_to_dir_creates_directory_and_writes_file() {
        let dir = std::env::temp_dir().join(format!(
            "mold-save-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!dir.exists());

        let img = mold_core::ImageData {
            data: vec![0x89, 0x50, 0x4E, 0x47], // PNG magic bytes
            format: mold_core::OutputFormat::Png,
            width: 64,
            height: 64,
            index: 0,
        };

        save_image_to_dir(&dir, &img, "test-model:q8", 1);

        assert!(dir.exists(), "directory should be created");
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(files.len(), 1, "should have exactly one file");
        let file = files[0].as_ref().unwrap();
        let filename = file.file_name().to_str().unwrap().to_string();
        assert!(filename.starts_with("mold-test-model-q8-"), "{filename}");
        assert!(filename.ends_with(".png"), "{filename}");
        let contents = std::fs::read(file.path()).unwrap();
        assert_eq!(contents, vec![0x89, 0x50, 0x4E, 0x47]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_image_to_dir_batch_includes_index() {
        let dir = std::env::temp_dir().join(format!(
            "mold-save-batch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let img = mold_core::ImageData {
            data: vec![0xFF, 0xD8], // JPEG magic
            format: mold_core::OutputFormat::Jpeg,
            width: 64,
            height: 64,
            index: 2,
        };

        save_image_to_dir(&dir, &img, "flux-dev", 4);

        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(files.len(), 1);
        let filename = files[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            filename.contains("-2.jpeg"),
            "batch index in name: {filename}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_image_to_dir_invalid_path_logs_warning_no_panic() {
        // Saving to a path that can't be created should not panic
        let img = mold_core::ImageData {
            data: vec![0x00],
            format: mold_core::OutputFormat::Png,
            width: 1,
            height: 1,
            index: 0,
        };
        // /dev/null/impossible can't be created as a directory
        save_image_to_dir(
            std::path::Path::new("/dev/null/impossible"),
            &img,
            "test",
            1,
        );
        // Test passes if no panic occurred
    }
}
