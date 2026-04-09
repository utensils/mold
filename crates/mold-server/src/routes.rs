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
    ActiveGenerationStatus, GpuInfo, ModelInfoExtended, ServerStatus, SseErrorEvent,
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

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "FORBIDDEN".to_string(),
            status: StatusCode::FORBIDDEN,
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
        .route("/api/capabilities", get(server_capabilities))
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
            let content_type = HeaderValue::from_static(img.format.content_type());
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
            // For video responses, return the actual video data (not the thumbnail)
            // and send video metadata in headers so the client can reconstruct VideoData.
            let output_data = if let Some(ref video) = response.video {
                let ct = HeaderValue::from_static(video.format.content_type());
                headers.insert(header::CONTENT_TYPE, ct);
                if let Ok(v) = HeaderValue::from_str(&video.frames.to_string()) {
                    headers.insert("x-mold-video-frames", v);
                }
                if let Ok(v) = HeaderValue::from_str(&video.fps.to_string()) {
                    headers.insert("x-mold-video-fps", v);
                }
                if let Ok(v) = HeaderValue::from_str(&video.width.to_string()) {
                    headers.insert("x-mold-video-width", v);
                }
                if let Ok(v) = HeaderValue::from_str(&video.height.to_string()) {
                    headers.insert("x-mold-video-height", v);
                }
                if video.has_audio {
                    headers.insert("x-mold-video-has-audio", HeaderValue::from_static("1"));
                }
                if let Some(dur) = video.duration_ms {
                    if let Ok(v) = HeaderValue::from_str(&dur.to_string()) {
                        headers.insert("x-mold-video-duration-ms", v);
                    }
                }
                if let Some(sr) = video.audio_sample_rate {
                    if let Ok(v) = HeaderValue::from_str(&sr.to_string()) {
                        headers.insert("x-mold-video-audio-sample-rate", v);
                    }
                }
                if let Some(ch) = video.audio_channels {
                    if let Ok(v) = HeaderValue::from_str(&ch.to_string()) {
                        headers.insert("x-mold-video-audio-channels", v);
                    }
                }
                video.data.clone()
            } else {
                img.data
            };
            Ok((headers, output_data))
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

    let model_name = mold_core::manifest::resolve_model_name(&req.model);

    // Auto-pull upscaler model if not downloaded
    let needs_pull = {
        let config = state.config.read().await;
        config
            .models
            .get(&model_name)
            .and_then(|c| c.transformer.as_ref())
            .is_none()
    };
    if needs_pull {
        if mold_core::manifest::find_manifest(&model_name).is_none() {
            return Err(ApiError::not_found(format!(
                "unknown upscaler model '{}'. Run 'mold list' to see available models.",
                model_name
            )));
        }
        model_manager::pull_model(&state, &model_name, None).await?;
    }

    let config = state.config.read().await;
    let weights_path = config
        .models
        .get(&model_name)
        .and_then(|c| c.transformer.as_ref())
        .ok_or_else(|| {
            ApiError::not_found(format!(
                "upscaler model '{}' not configured after pull",
                model_name
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

    let model_name = mold_core::manifest::resolve_model_name(&req.model);

    // Check if model needs pulling before spawning the SSE stream
    let needs_pull = {
        let config = state.config.read().await;
        config
            .models
            .get(&model_name)
            .and_then(|c| c.transformer.as_ref())
            .is_none()
    };

    // Validate the model exists in the manifest if we need to pull
    if needs_pull && mold_core::manifest::find_manifest(&model_name).is_none() {
        return Err(ApiError::not_found(format!(
            "unknown upscaler model '{}'. Run 'mold list' to see available models.",
            model_name
        )));
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();
    let model_name_owned = model_name.clone();
    let state_clone = state.clone();
    let upscaler_cache = state.upscaler_cache.clone();

    tokio::spawn(async move {
        // Auto-pull the upscaler model if not downloaded
        if needs_pull {
            let progress_tx = tx.clone();
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

            match model_manager::pull_model(&state_clone, &model_name_owned, Some(callback)).await {
                Ok(_) => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                        model: model_name_owned.clone(),
                    }));
                }
                Err(e) => {
                    let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent {
                        message: format!("failed to pull upscaler model: {}", e.error),
                    }));
                    return;
                }
            }
        }

        // Read weights path after potential pull
        let weights_path = {
            let config = state_clone.config.read().await;
            config
                .models
                .get(&model_name_owned)
                .and_then(|c| c.transformer.as_ref())
                .map(std::path::PathBuf::from)
        };

        let Some(weights_path) = weights_path else {
            let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent {
                message: format!(
                    "upscaler model '{}' not configured after pull",
                    model_name_owned
                ),
            }));
            return;
        };

        let result = tokio::task::spawn_blocking(move || {
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
                    let image_b64 =
                        base64::engine::general_purpose::STANDARD.encode(&resp.image.data);
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
        })
        .await;

        if let Err(e) = result {
            tracing::error!("upscale task panicked: {e}");
        }
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
        hostname: hostname::get().ok().and_then(|h| h.into_string().ok()),
        memory_status: mold_inference::device::memory_status_string(),
        gpus: None,
        queue_depth: None,
        queue_capacity: None,
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

// ── /api/capabilities ────────────────────────────────────────────────────────

/// Report the feature toggles a client needs to render correctly (hide the
/// delete button when delete isn't allowed, etc.). No auth required — this
/// is a read-only introspection endpoint.
async fn server_capabilities() -> Json<mold_core::ServerCapabilities> {
    Json(mold_core::ServerCapabilities {
        gallery: mold_core::GalleryCapabilities {
            can_delete: gallery_delete_allowed(),
        },
    })
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
///
/// Prefers the SQLite metadata DB when available so listings stay fast on
/// large galleries (no per-request directory walk). Falls back to the
/// filesystem scan when the DB is disabled, can't be opened, or — as a
/// safety net — has no rows for this directory yet (e.g. the reconciliation
/// background task has not finished on first startup).
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

    if state.metadata_db.is_some() {
        let db_arc = state.metadata_db.clone();
        let dir = output_dir.clone();
        let listed = tokio::task::spawn_blocking(move || {
            db_arc
                .as_ref()
                .as_ref()
                .map(|db| db.list(Some(&dir)))
                .transpose()
        })
        .await
        .map_err(|e| ApiError::internal(format!("gallery DB query failed: {e}")))?
        .map_err(|e| ApiError::internal(format!("gallery DB query failed: {e:#}")))?;
        if let Some(rows) = listed {
            if !rows.is_empty() {
                let images = rows.iter().map(|r| r.to_gallery_image()).collect();
                return Ok(Json(images));
            }
        }
    }

    let images = tokio::task::spawn_blocking(move || scan_gallery_dir(&output_dir))
        .await
        .map_err(|e| ApiError::internal(format!("gallery scan failed: {e}")))?;

    Ok(Json(images))
}

/// Serve a gallery file by filename.
///
/// Supports HTTP `Range` requests so `<video>` elements can scrub MP4
/// outputs without downloading the whole clip up front. Partial responses
/// stream straight from disk via `tokio_util::io::ReaderStream` — nothing
/// buffers the full file in server RAM, which matters once a gallery
/// contains multi-GB LTX-2 outputs. Non-range requests still return the
/// whole file (streamed) with `Accept-Ranges: bytes` so the client knows
/// it can seek on subsequent requests.
async fn get_gallery_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<axum::response::Response, ApiError> {
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
    let meta = match tokio::fs::metadata(&path).await {
        Ok(m) if m.is_file() => m,
        _ => {
            return Err(ApiError::not_found(format!(
                "image not found: {clean_name}"
            )));
        }
    };
    let total_len = meta.len();
    let content_type = content_type_for_filename(&clean_name);

    let range_header = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let file = tokio::fs::File::open(&path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to open file: {e}")))?;

    if let Some(raw) = range_header {
        if let Some((start, end)) = parse_byte_range(&raw, total_len) {
            return serve_range(file, start, end, total_len, content_type).await;
        } else {
            // A `Range` header we can't satisfy ⇒ 416 per RFC 9110 §15.6.2.
            return Ok(axum::response::Response::builder()
                .status(StatusCode::RANGE_NOT_SATISFIABLE)
                .header(header::CONTENT_RANGE, format!("bytes */{total_len}"))
                .body(axum::body::Body::empty())
                .unwrap());
        }
    }

    // Full response: stream the file rather than buffer it in RAM.
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, total_len)
        .header(header::CACHE_CONTROL, "public, max-age=3600")
        .body(body)
        .unwrap())
}

/// Parse a `Range: bytes=start-end` header into a concrete (start, end)
/// byte range inclusive on both ends. Returns `None` for unsatisfiable or
/// malformed ranges — the caller translates that into a 416 response.
///
/// Only the single-range form is supported (multipart ranges are vanishingly
/// rare in practice and substantially more complex to implement correctly;
/// browsers for `<video>` always send single ranges).
fn parse_byte_range(header: &str, total_len: u64) -> Option<(u64, u64)> {
    let spec = header.strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (start_s, end_s) = spec.split_once('-')?;
    let start_s = start_s.trim();
    let end_s = end_s.trim();

    if total_len == 0 {
        return None;
    }

    if start_s.is_empty() {
        // Suffix range: `bytes=-N` means "the last N bytes".
        let suffix: u64 = end_s.parse().ok()?;
        if suffix == 0 {
            return None;
        }
        let start = total_len.saturating_sub(suffix);
        return Some((start, total_len - 1));
    }

    let start: u64 = start_s.parse().ok()?;
    if start >= total_len {
        return None;
    }
    let end: u64 = if end_s.is_empty() {
        total_len - 1
    } else {
        end_s.parse().ok()?
    };
    let end = end.min(total_len - 1);
    if end < start {
        return None;
    }
    Some((start, end))
}

/// Emit a `206 Partial Content` response streaming `[start, end]` inclusive
/// from the already-open file handle. `take(len)` bounds the reader so the
/// body terminates exactly at `end + 1` instead of reading the tail.
async fn serve_range(
    mut file: tokio::fs::File,
    start: u64,
    end: u64,
    total_len: u64,
    content_type: &'static str,
) -> Result<axum::response::Response, ApiError> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    file.seek(std::io::SeekFrom::Start(start))
        .await
        .map_err(|e| ApiError::internal(format!("seek failed: {e}")))?;
    let len = end - start + 1;
    let stream = tokio_util::io::ReaderStream::new(file.take(len));
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, len)
        .header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{total_len}"),
        )
        // Partial content is less cacheable at intermediaries than a plain
        // 200; keep a short TTL so the client's own cache still helps.
        .header(header::CACHE_CONTROL, "public, max-age=300")
        .body(body)
        .unwrap())
}

/// Pick an HTTP Content-Type for a gallery filename. Covers every format
/// `OutputFormat` can emit plus a safe default.
fn content_type_for_filename(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else if lower.ends_with(".apng") {
        "image/apng"
    } else if lower.ends_with(".mp4") {
        "video/mp4"
    } else {
        "application/octet-stream"
    }
}

/// Delete a gallery image and its server-side thumbnail.
///
/// Opt-in: the endpoint returns `403 Forbidden` unless
/// `MOLD_GALLERY_ALLOW_DELETE=1` is set on the server. Destructive writes
/// should not be reachable by default — operators explicitly allow them,
/// ideally in combination with the existing API-key middleware.
async fn delete_gallery_image(
    State(state): State<AppState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if !gallery_delete_allowed() {
        return Err(ApiError::forbidden(
            "gallery delete is disabled; set MOLD_GALLERY_ALLOW_DELETE=1 to enable",
        ));
    }
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

    // Also remove server-side thumbnail (both legacy no-suffix and current
    // `.png`-suffixed cache layouts).
    let thumb_dir = server_thumbnail_dir();
    let _ = std::fs::remove_file(thumb_dir.join(&clean_name));
    let _ = std::fs::remove_file(thumb_dir.join(format!("{clean_name}.png")));

    // Drop the matching metadata row if the DB is enabled. Errors here are
    // logged — they don't roll back the disk delete since the file is the
    // source of truth and reconciliation will re-sync on the next restart.
    if let Some(db) = state.metadata_db.as_ref().as_ref() {
        match db.delete(&output_dir, &clean_name) {
            Ok(true) => {}
            Ok(false) => tracing::debug!(
                "delete: no metadata row for {}",
                output_dir.join(&clean_name).display()
            ),
            Err(e) => tracing::warn!(
                "metadata DB delete failed for {}: {e:#}",
                output_dir.join(&clean_name).display()
            ),
        }
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Whether the destructive `DELETE /api/gallery/image/:filename` route
/// is enabled. Off by default — operators opt in with
/// `MOLD_GALLERY_ALLOW_DELETE=1` (accepts `1` / `true` / `yes`, any case).
fn gallery_delete_allowed() -> bool {
    std::env::var("MOLD_GALLERY_ALLOW_DELETE")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
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

    // Thumbnail cache path: always `.png` regardless of the source extension,
    // so mp4 / gif / apng / webp / jpg all coexist cleanly in the same cache
    // dir and `image.save()` doesn't pick the wrong format from the path.
    let thumb_dir = server_thumbnail_dir();
    let thumb_path = thumb_dir.join(format!("{clean_name}.png"));
    let lower = clean_name.to_ascii_lowercase();
    let is_video = lower.ends_with(".mp4");

    if !thumb_path.is_file() {
        // Generate thumbnail on-demand. Videos go through openh264 for a real
        // first-frame extract; everything else decodes via the `image` crate.
        // If either path fails, we fall back to serving the source bytes
        // directly — browsers are more lenient about partial / checksum-
        // mismatched images than either decoder, and the SPA would rather
        // show something than a 500.
        let source = source_path.clone();
        let dest = thumb_path.clone();
        let gen_result = tokio::task::spawn_blocking(move || {
            if is_video {
                generate_video_thumbnail(&source, &dest)
            } else {
                generate_server_thumbnail(&source, &dest)
            }
        })
        .await
        .map_err(|e| ApiError::internal(format!("thumbnail generation failed: {e}")))?;

        if let Err(err) = gen_result {
            tracing::warn!(
                file = %clean_name,
                error = %err,
                "thumbnail decode failed; falling back to source bytes"
            );
            // For videos, the browser can't render the raw mp4 as an <img>
            // either, so serving the source doesn't help — fall back to the
            // SVG play-icon placeholder instead.
            if is_video {
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("image/svg+xml"),
                );
                headers.insert(
                    header::CACHE_CONTROL,
                    HeaderValue::from_static("public, max-age=300"),
                );
                return Ok((headers, VIDEO_PLACEHOLDER_SVG.as_bytes().to_vec()));
            }
            let raw = tokio::fs::read(&source_path)
                .await
                .map_err(|e| ApiError::internal(format!("failed to read source: {e}")))?;
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(content_type_for_filename(&clean_name)),
            );
            headers.insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=300"),
            );
            return Ok((headers, raw));
        }
    }

    let data = tokio::fs::read(&thumb_path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to read thumbnail: {e}")))?;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=3600"),
    );

    Ok((headers, data))
}

const VIDEO_PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs><rect width="256" height="256" fill="url(#g)"/><circle cx="128" cy="128" r="52" fill="rgba(255,255,255,0.08)"/><polygon points="112,100 112,156 160,128" fill="rgba(226,232,240,0.85)"/></svg>"##;

/// Server-side thumbnail cache directory.
fn server_thumbnail_dir() -> std::path::PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails")
}

/// Generate a 256x256 max thumbnail from source image. The result is always
/// written as a PNG regardless of the source format, so callers should pass
/// a `.png`-suffixed `dest` to keep the on-disk cache unambiguous.
fn generate_server_thumbnail(
    source: &std::path::Path,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    let img = image::open(source)?;
    let thumb = img.thumbnail(256, 256);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    thumb.save_with_format(dest, image::ImageFormat::Png)?;
    Ok(())
}

/// Extract the first frame of an MP4 as a PNG thumbnail and downscale to
/// 256px max via the `image` crate. Uses the openh264 pipeline that
/// `mold_inference::ltx2::media` already ships for video probes.
///
/// The full-frame PNG is written to a sibling temp path first, then decoded
/// and resized — this keeps `mold_inference`'s existing helper surface stable
/// while still producing a compact thumbnail.
fn generate_video_thumbnail(
    source: &std::path::Path,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Decode the first frame to a temporary full-resolution PNG, then
    // thumbnail-resize via the `image` crate. We stage through a temp file
    // rather than through memory to reuse `extract_thumbnail`'s existing
    // I/O-based API.
    let tmp = dest.with_extension("firstframe.png");
    mold_inference::ltx2::media::extract_thumbnail(source, &tmp)?;
    let decode_result = (|| -> anyhow::Result<()> {
        let img = image::open(&tmp)?;
        let thumb = img.thumbnail(256, 256);
        thumb.save_with_format(dest, image::ImageFormat::Png)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&tmp);
    decode_result
}

/// Pre-generate thumbnails for all gallery images on server startup.
pub fn spawn_thumbnail_warmup(config: &mold_core::Config) {
    if !thumbnail_warmup_enabled() {
        tracing::info!("thumbnail warmup disabled; thumbnails will be generated on demand");
        return;
    }

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
            let is_raster = matches!(
                ext.as_deref(),
                Some("png" | "jpg" | "jpeg" | "gif" | "apng" | "webp")
            );
            let is_video = matches!(ext.as_deref(), Some("mp4"));
            if !is_raster && !is_video {
                continue;
            }
            let filename = path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();
            let thumb_path = thumb_dir.join(format!("{filename}.png"));
            if thumb_path.is_file() {
                continue;
            }
            let result = if is_video {
                generate_video_thumbnail(path, &thumb_path)
            } else {
                generate_server_thumbnail(path, &thumb_path)
            };
            if let Err(e) = result {
                tracing::warn!("failed to generate thumbnail for {}: {e}", path.display());
            }
        }
        tracing::info!("thumbnail warmup complete");
    });
}

fn thumbnail_warmup_enabled() -> bool {
    std::env::var("MOLD_THUMBNAIL_WARMUP")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

/// Scan a directory for gallery outputs (images + videos).
///
/// Picks up every format `OutputFormat` can emit: png / jpg / jpeg / gif /
/// apng / webp / mp4. For files with no embedded `mold:parameters` chunk
/// (notably gif / webp / mp4), we synthesize a stub `OutputMetadata` from
/// the filename so the UI can still display them alongside annotated items.
///
/// Invalid files are filtered out at scan time rather than surfaced as
/// broken tiles in the UI. "Invalid" here means any of:
/// - below a format-specific size floor (tiny stubs left by abandoned
///   writes, aborted generations, or test harnesses)
/// - no decodable image header (raster formats)
/// - no `ftyp` box at the start of the file (mp4)
///
/// This is a header-only validation, not a full pixel decode, so a file
/// that passes the check can still be corrupt mid-stream (e.g. broken
/// IDAT CRC). Those fall through to the thumbnail endpoint which serves
/// the raw bytes as a last resort.
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
        let format = match ext.as_deref() {
            Some("png") => Some(mold_core::OutputFormat::Png),
            Some("jpg") | Some("jpeg") => Some(mold_core::OutputFormat::Jpeg),
            Some("gif") => Some(mold_core::OutputFormat::Gif),
            Some("apng") => Some(mold_core::OutputFormat::Apng),
            Some("webp") => Some(mold_core::OutputFormat::Webp),
            Some("mp4") => Some(mold_core::OutputFormat::Mp4),
            _ => None,
        };
        let Some(format) = format else { continue };

        let fs_meta = entry.metadata().ok();
        let timestamp = fs_meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let size_bytes = fs_meta.as_ref().map(|m| m.len()).unwrap_or(0);

        // Size floor: anything below this is guaranteed not a real output.
        if size_bytes < min_valid_size(format) {
            continue;
        }

        // Header-level validation (fast, O(1) bytes per file).
        let header_ok = match format {
            mold_core::OutputFormat::Mp4 => has_ftyp_box(path),
            _ => image_header_dims(path).is_some(),
        };
        if !header_ok {
            continue;
        }

        // Solid-black detection. Only inspect small files where a solid-color
        // image is plausible (real renderings at any meaningful resolution
        // weigh tens of KB or more). For those, we decode and sample a 16×16
        // thumbnail so a failed / empty generation doesn't pollute the feed.
        if !matches!(format, mold_core::OutputFormat::Mp4)
            && is_probably_solid_black(path, format, size_bytes)
        {
            continue;
        }

        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        // Try embedded metadata first — PNG text chunks (also covers APNG
        // since APNG files are valid PNGs) and JPEG COM markers.
        let embedded = match ext.as_deref() {
            Some("png") | Some("apng") => read_png_metadata(path),
            Some("jpg") | Some("jpeg") => read_jpeg_metadata(path),
            _ => None,
        };

        let (metadata, synthetic) = match embedded {
            Some(m) => (m, false),
            None => {
                // Synthesize. If the file is a raster whose header decodes,
                // use its real dimensions so the UI can render the card at
                // the correct aspect ratio even without mold metadata.
                let mut meta = synthesize_metadata_from_filename(&filename, timestamp);
                if !matches!(format, mold_core::OutputFormat::Mp4) {
                    if let Some((w, h)) = image_header_dims(path) {
                        meta.width = w;
                        meta.height = h;
                    }
                }
                (meta, true)
            }
        };

        images.push(mold_core::GalleryImage {
            filename,
            metadata,
            timestamp,
            format: Some(format),
            size_bytes: Some(size_bytes),
            metadata_synthetic: synthetic,
        });
    }

    images.sort_by_key(|img| std::cmp::Reverse(img.timestamp));
    images
}

/// Minimum on-disk size (in bytes) below which a file is treated as a
/// corrupt / aborted output and hidden from the gallery listing. The
/// thresholds are well below any real mold-generated output but above any
/// parseable-but-empty stub — a 1×1 pixel PNG is ~67 bytes, a real 512×512
/// PNG is at least tens of KB.
fn min_valid_size(format: mold_core::OutputFormat) -> u64 {
    match format {
        // Raster images: any real mold output is multi-KB. The 256-byte
        // floor comfortably filters truncated PNG stubs (signature + IHDR
        // only, ~45 bytes) and similar degenerate cases without touching
        // legitimate tiny gifs (e.g. a few hundred bytes for a 1-frame GIF).
        mold_core::OutputFormat::Png
        | mold_core::OutputFormat::Apng
        | mold_core::OutputFormat::Jpeg
        | mold_core::OutputFormat::Webp => 256,
        mold_core::OutputFormat::Gif => 128,
        // An mp4 with a single frame and no audio is still many KB.
        mold_core::OutputFormat::Mp4 => 4096,
    }
}

/// Fast "does this decode as an image?" check. Returns the image's
/// pixel dimensions (width, height) on success. Only reads the header —
/// typically under 1 KB — so it's safe to call for every file on every
/// `/api/gallery` request.
fn image_header_dims(path: &std::path::Path) -> Option<(u32, u32)> {
    image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Heuristic detector for "solid black" (or near-black) raster images —
/// typically the artefact of an aborted / NaN-poisoned generation that
/// wrote an all-zero image tensor. We only even consider images below a
/// format-specific suspect size (any real content at meaningful resolution
/// compresses to tens of KB at minimum; a solid-color PNG fits in a few
/// hundred bytes), then decode and sample a 16×16 thumbnail to check
/// whether any pixel's max channel exceeds a small threshold.
fn is_probably_solid_black(
    path: &std::path::Path,
    format: mold_core::OutputFormat,
    size_bytes: u64,
) -> bool {
    const SAMPLE_DIM: u32 = 16;
    // Allow any single channel up to this intensity before we conclude the
    // file is "real" content. 16 out of 255 is ~6%: enough to accept dark
    // images that aren't literal black, but tight enough to reject the
    // artefacts we actually want to filter.
    const CHANNEL_CEILING: u8 = 16;

    let suspect_threshold: u64 = match format {
        // PNG / APNG: zlib-compressed raw pixels; 8 KB is comfortably above
        // any solid-color encoding at 1k-ish resolution.
        mold_core::OutputFormat::Png | mold_core::OutputFormat::Apng => 8 * 1024,
        // JPEG compresses solid color to a few hundred bytes; generous ceiling.
        mold_core::OutputFormat::Jpeg => 4 * 1024,
        mold_core::OutputFormat::Gif | mold_core::OutputFormat::Webp => 4 * 1024,
        mold_core::OutputFormat::Mp4 => return false,
    };
    if size_bytes > suspect_threshold {
        return false;
    }

    let Ok(img) = image::open(path) else {
        return false;
    };
    let thumb = img.thumbnail(SAMPLE_DIM, SAMPLE_DIM).to_rgb8();
    let mut max_channel: u8 = 0;
    for pixel in thumb.pixels() {
        let m = pixel.0[0].max(pixel.0[1]).max(pixel.0[2]);
        if m > max_channel {
            max_channel = m;
        }
        if max_channel > CHANNEL_CEILING {
            return false;
        }
    }
    max_channel <= CHANNEL_CEILING
}

/// Check for the ISO BMFF `ftyp` box at offset 4 of the file. A real mp4
/// always starts with a top-level `ftyp` box; files that fail this check
/// are typically truncated writes or wrong-extension text files.
fn has_ftyp_box(path: &std::path::Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 12];
    if f.read_exact(&mut buf).is_err() {
        return false;
    }
    &buf[4..8] == b"ftyp"
}

/// Build a best-effort `OutputMetadata` from a filename like
/// `mold-<model>-<unix>[-<idx>].<ext>`. Fields we can't recover (seed, steps,
/// guidance, resolution, prompt) are left at zero / empty so the UI can
/// render them as "unknown". The client reads `metadata_synthetic=true`
/// from the enclosing `GalleryImage` to treat these as placeholders.
fn synthesize_metadata_from_filename(filename: &str, timestamp: u64) -> mold_core::OutputMetadata {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let model = stem
        .strip_prefix("mold-")
        .and_then(|rest| {
            // Trim trailing `-<unix>` and optional `-<idx>` suffixes by
            // walking back across numeric segments.
            let mut parts: Vec<&str> = rest.split('-').collect();
            while parts
                .last()
                .map(|p| p.chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
                && parts.len() > 1
            {
                parts.pop();
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("-"))
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    mold_core::OutputMetadata {
        prompt: String::new(),
        negative_prompt: None,
        original_prompt: None,
        model,
        seed: 0,
        steps: 0,
        guidance: 0.0,
        width: 0,
        height: 0,
        strength: None,
        scheduler: None,
        lora: None,
        lora_scale: None,
        frames: None,
        fps: None,
        version: format!("synthesized@{timestamp}"),
    }
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

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        &ENV_LOCK
    }

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

    #[test]
    fn thumbnail_warmup_is_disabled_by_default() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::remove_var("MOLD_THUMBNAIL_WARMUP");
        }
        assert!(!thumbnail_warmup_enabled());
    }

    #[test]
    fn thumbnail_warmup_accepts_truthy_env_values() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "1");
        }
        assert!(thumbnail_warmup_enabled());
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "true");
        }
        assert!(thumbnail_warmup_enabled());
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "YES");
        }
        assert!(thumbnail_warmup_enabled());
        unsafe {
            std::env::remove_var("MOLD_THUMBNAIL_WARMUP");
        }
    }

    #[test]
    fn thumbnail_warmup_rejects_falsey_env_values() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "0");
        }
        assert!(!thumbnail_warmup_enabled());
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "false");
        }
        assert!(!thumbnail_warmup_enabled());
        unsafe {
            std::env::remove_var("MOLD_THUMBNAIL_WARMUP");
        }
    }

    #[test]
    fn content_type_covers_every_output_format() {
        assert_eq!(content_type_for_filename("a.png"), "image/png");
        assert_eq!(content_type_for_filename("a.PNG"), "image/png");
        assert_eq!(content_type_for_filename("a.jpg"), "image/jpeg");
        assert_eq!(content_type_for_filename("a.jpeg"), "image/jpeg");
        assert_eq!(content_type_for_filename("a.gif"), "image/gif");
        assert_eq!(content_type_for_filename("a.webp"), "image/webp");
        assert_eq!(content_type_for_filename("a.apng"), "image/apng");
        assert_eq!(content_type_for_filename("a.mp4"), "video/mp4");
        assert_eq!(
            content_type_for_filename("a.unknown"),
            "application/octet-stream"
        );
    }

    #[test]
    fn synthesized_metadata_parses_model_from_filename() {
        let meta = synthesize_metadata_from_filename("mold-flux-dev-q8-1710000000.mp4", 1710000000);
        // Trailing unix timestamp should be stripped; model tag preserved.
        assert_eq!(meta.model, "flux-dev-q8");
        assert_eq!(meta.prompt, "");
        assert_eq!(meta.seed, 0);
        assert!(meta.version.starts_with("synthesized@"));

        // Batch suffix (trailing `-<idx>`) also stripped along with timestamp.
        let meta =
            synthesize_metadata_from_filename("mold-ltx-video-bf16-1710000030-2.gif", 1710000030);
        assert_eq!(meta.model, "ltx-video-bf16");

        // Non-mold filename falls back to "unknown".
        let meta = synthesize_metadata_from_filename("unrelated.png", 0);
        assert_eq!(meta.model, "unknown");
    }

    // ── Gallery validation ───────────────────────────────────────────────

    /// Create a scratch directory unique to this test and delete it on drop.
    /// Using `std::env::temp_dir()` rather than pulling in a `tempfile`
    /// dev-dep for three tests' worth of fixtures.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("mold-gallery-test-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).expect("create tempdir");
            Self(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Encode a noisy PNG in-memory via the `image` crate. The checkerboard
    /// pattern resists zlib compression so the encoded bytes exceed the
    /// gallery size floor — a solid-color PNG of the same dimensions would
    /// compress to ~80 bytes and be filtered out by `min_valid_size`.
    fn make_png_bytes(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            let n = (x.wrapping_mul(37) ^ y.wrapping_mul(131)) as u8;
            image::Rgb([n, n.wrapping_add(85), n.wrapping_sub(17)])
        });
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("encode png");
        buf
    }

    #[test]
    fn min_valid_size_thresholds_are_sensible() {
        // Raster formats: high enough to catch every truncated / stub file
        // we've seen in dev harnesses (largest known bad fixture was ~200 B).
        assert!(min_valid_size(mold_core::OutputFormat::Png) >= 128);
        assert!(min_valid_size(mold_core::OutputFormat::Jpeg) >= 128);
        assert!(min_valid_size(mold_core::OutputFormat::Apng) >= 128);
        assert!(min_valid_size(mold_core::OutputFormat::Webp) >= 128);
        // But not so high we filter out legitimate tiny GIFs.
        assert!(min_valid_size(mold_core::OutputFormat::Gif) <= 512);
        // MP4: no real rendering is < 1 KB; 4 KB is a comfortable floor.
        assert!(min_valid_size(mold_core::OutputFormat::Mp4) >= 1024);
    }

    #[test]
    fn has_ftyp_box_accepts_real_header_and_rejects_garbage() {
        let td = TempDir::new("ftyp");

        // Real-ish mp4 header: `\0\0\0\x20 ftypisom ...`
        let mut real = Vec::new();
        real.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]);
        real.extend_from_slice(b"ftyp");
        real.extend_from_slice(b"isom\x00\x00\x02\x00isomiso2mp41");
        let real_path = td.path().join("real.mp4");
        std::fs::write(&real_path, &real).unwrap();
        assert!(has_ftyp_box(&real_path));

        // Wrong magic — random text with an mp4 extension.
        let fake_path = td.path().join("fake.mp4");
        std::fs::write(&fake_path, b"this is not an mp4 file at all").unwrap();
        assert!(!has_ftyp_box(&fake_path));

        // Too short (fewer than 12 bytes) — can't contain an ftyp box.
        let trunc_path = td.path().join("truncated.mp4");
        std::fs::write(&trunc_path, b"\x00\x00\x00\x20").unwrap();
        assert!(!has_ftyp_box(&trunc_path));

        // Missing entirely.
        assert!(!has_ftyp_box(&td.path().join("nope.mp4")));
    }

    #[test]
    fn image_header_dims_returns_real_dimensions() {
        let td = TempDir::new("header");
        let p = td.path().join("valid.png");
        std::fs::write(&p, make_png_bytes(42, 24)).unwrap();
        assert_eq!(image_header_dims(&p), Some((42, 24)));

        // Truncated: PNG signature only, no IHDR.
        let stub = td.path().join("stub.png");
        std::fs::write(&stub, b"\x89PNG\r\n\x1a\n").unwrap();
        assert!(image_header_dims(&stub).is_none());

        // Non-image bytes entirely.
        let text = td.path().join("text.png");
        std::fs::write(&text, b"hello world, not a png").unwrap();
        assert!(image_header_dims(&text).is_none());
    }

    #[test]
    fn scan_gallery_dir_filters_invalid_and_keeps_valid() {
        let td = TempDir::new("scan");
        let dir = td.path();

        // A valid PNG large enough to exceed the 256-byte raster size floor.
        std::fs::write(dir.join("mold-model-1000.png"), make_png_bytes(32, 32)).unwrap();

        // Truncated raster that passes size floor but has no valid header.
        let mut junk = vec![0u8; 512];
        junk[..4].copy_from_slice(b"JUNK");
        std::fs::write(dir.join("mold-broken-2000.png"), &junk).unwrap();

        // Tiny raster under the size floor (sub-IHDR).
        std::fs::write(
            dir.join("mold-tiny-3000.png"),
            b"\x89PNG\r\n\x1a\n", // 8 bytes: signature only
        )
        .unwrap();

        // Valid-enough mp4 (ftyp at offset 4) — should survive.
        let mut mp4 = Vec::new();
        mp4.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]);
        mp4.extend_from_slice(b"ftyp");
        mp4.extend_from_slice(b"isom\x00\x00\x02\x00");
        // Pad above the 4096-byte mp4 size floor so it isn't filtered on
        // size alone — the scan still checks ftyp either way.
        mp4.resize(8192, 0);
        std::fs::write(dir.join("mold-ltx-4000.mp4"), &mp4).unwrap();

        // Mp4 extension but no ftyp.
        let bad_mp4 = vec![0u8; 8192];
        std::fs::write(dir.join("mold-no-ftyp-5000.mp4"), &bad_mp4).unwrap();

        // Unsupported extension — ignored entirely.
        std::fs::write(dir.join("random.txt"), b"not an output").unwrap();

        let results = scan_gallery_dir(dir);
        let names: Vec<&str> = results.iter().map(|i| i.filename.as_str()).collect();
        assert!(
            names.contains(&"mold-model-1000.png"),
            "valid PNG should survive: {names:?}"
        );
        assert!(
            names.contains(&"mold-ltx-4000.mp4"),
            "valid MP4 with ftyp should survive: {names:?}"
        );
        assert!(
            !names.contains(&"mold-broken-2000.png"),
            "PNG with no valid header should be filtered: {names:?}"
        );
        assert!(
            !names.contains(&"mold-tiny-3000.png"),
            "under-size PNG stub should be filtered: {names:?}"
        );
        assert!(
            !names.contains(&"mold-no-ftyp-5000.mp4"),
            "MP4 without ftyp should be filtered: {names:?}"
        );
        assert_eq!(names.len(), 2, "only the 2 valid fixtures remain");
    }

    #[test]
    fn solid_black_png_is_filtered_at_scan_time() {
        let td = TempDir::new("black");
        let dir = td.path();

        // A 256×256 solid-black PNG — definitely below the suspect-size
        // threshold (compresses to a few hundred bytes) and every pixel is
        // below the channel ceiling.
        let black = image::RgbImage::from_pixel(256, 256, image::Rgb([0, 0, 0]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(black)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(dir.join("mold-noisy-1000.png"), &buf).unwrap();

        // A normal noisy PNG with the same dimensions — should survive.
        std::fs::write(dir.join("mold-valid-2000.png"), make_png_bytes(256, 256)).unwrap();

        let results = scan_gallery_dir(dir);
        let names: Vec<&str> = results.iter().map(|i| i.filename.as_str()).collect();
        assert!(
            !names.contains(&"mold-noisy-1000.png"),
            "solid-black PNG should be filtered: {names:?}"
        );
        assert!(
            names.contains(&"mold-valid-2000.png"),
            "noisy PNG should survive: {names:?}"
        );
    }

    #[test]
    fn probably_solid_black_ignores_large_files() {
        // Files above the per-format suspect size are trusted without a full
        // decode (the check is purely a cheap heuristic to catch NaN /
        // abort-flavored dev outputs). Verify we bail out on size alone.
        let td = TempDir::new("bigblack");
        let big_path = td.path().join("big.png");
        // Write arbitrary bytes — we never decode because of the size guard.
        std::fs::write(&big_path, vec![0u8; 20 * 1024]).unwrap();
        assert!(!is_probably_solid_black(
            &big_path,
            mold_core::OutputFormat::Png,
            20 * 1024,
        ));
    }

    #[test]
    fn parse_byte_range_handles_common_forms() {
        // `bytes=0-499` — first 500 bytes
        assert_eq!(parse_byte_range("bytes=0-499", 2000), Some((0, 499)));
        // open-ended `bytes=100-` — from byte 100 to EOF
        assert_eq!(parse_byte_range("bytes=100-", 2000), Some((100, 1999)));
        // suffix `bytes=-500` — last 500 bytes
        assert_eq!(parse_byte_range("bytes=-500", 2000), Some((1500, 1999)));
        // end past EOF — clamped to last byte
        assert_eq!(parse_byte_range("bytes=0-9999", 2000), Some((0, 1999)));
        // whole file
        assert_eq!(parse_byte_range("bytes=0-1999", 2000), Some((0, 1999)));
    }

    #[test]
    fn parse_byte_range_rejects_malformed_and_unsatisfiable() {
        assert_eq!(parse_byte_range("bytes=", 1000), None);
        assert_eq!(parse_byte_range("bytes=abc-100", 1000), None);
        // start past EOF
        assert_eq!(parse_byte_range("bytes=2000-", 1000), None);
        // end before start
        assert_eq!(parse_byte_range("bytes=500-100", 1000), None);
        // multi-range not supported
        assert_eq!(parse_byte_range("bytes=0-10,20-30", 1000), None);
        // suffix of 0 bytes is meaningless
        assert_eq!(parse_byte_range("bytes=-0", 1000), None);
        // empty file can't satisfy any range
        assert_eq!(parse_byte_range("bytes=0-10", 0), None);
        // wrong unit prefix
        assert_eq!(parse_byte_range("items=0-10", 1000), None);
    }

    #[test]
    fn gallery_delete_toggle_reads_env_var() {
        // Use a plausible-but-unique key so we don't clobber a caller's env.
        let key = "MOLD_GALLERY_ALLOW_DELETE";
        // Safety: env mutation is test-global; we restore the pre-test value
        // below regardless of assertion outcomes.
        let prev = std::env::var(key).ok();
        for val in ["1", "true", "YES"] {
            unsafe {
                std::env::set_var(key, val);
            }
            assert!(
                gallery_delete_allowed(),
                "delete should be allowed for env {val:?}"
            );
        }
        for val in ["0", "false", "no", ""] {
            unsafe {
                std::env::set_var(key, val);
            }
            assert!(
                !gallery_delete_allowed(),
                "delete should be blocked for env {val:?}"
            );
        }
        unsafe {
            std::env::remove_var(key);
        }
        assert!(!gallery_delete_allowed(), "default is off");
        // Restore.
        if let Some(v) = prev {
            unsafe {
                std::env::set_var(key, v);
            }
        }
    }

    #[test]
    fn scan_populates_real_dimensions_for_synthesized_metadata() {
        // Files without an embedded mold:parameters chunk still get their
        // actual width/height filled in from the header decode — useful for
        // the SPA's aspect-ratio-preserving layout.
        let td = TempDir::new("dims");
        let dir = td.path();
        std::fs::write(dir.join("mold-nometa-1000.png"), make_png_bytes(128, 96)).unwrap();

        let results = scan_gallery_dir(dir);
        assert_eq!(results.len(), 1);
        let entry = &results[0];
        assert!(entry.metadata_synthetic);
        assert_eq!(entry.metadata.width, 128);
        assert_eq!(entry.metadata.height, 96);
    }
}
