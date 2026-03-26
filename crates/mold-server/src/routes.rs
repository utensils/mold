use axum::{
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{delete, get, post},
    Json, Router,
};
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
#[cfg(test)]
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
    apply_default_metadata_setting(&state, &mut req).await;
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

    // Check model availability (404/400 returned as HTTP errors before queueing)
    let _ = model_manager::check_model_available(&state, &req.model).await?;

    // Resolve output directory before queueing
    let output_dir = {
        let config = state.config.read().await;
        config.resolved_output_dir().map(|p| p.to_path_buf())
    };

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
    apply_default_metadata_setting(&state, &mut req).await;
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
    let _ = model_manager::check_model_available(&state, &req.model).await?;

    // Create SSE channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();

    // Resolve output directory now (before the spawn) to avoid lock contention later.
    let output_dir = {
        let config = state.config.read().await;
        config.resolved_output_dir().map(|p| p.to_path_buf())
    };

    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let job = GenerationJob {
        request: req,
        progress_tx: Some(tx.clone()),
        result_tx,
        output_dir,
    };

    let position = state.queue.submit(job).await.map_err(ApiError::internal)?;

    // Send initial queue position to the client
    if position > 0 {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued { position }));
    }

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
                        total_files: 0,
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
