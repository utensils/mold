use axum::{
    extract::{Extension, Path, Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{delete, get, patch, post},
    Json, Router,
};
use base64::Engine as _;
use mold_core::{
    types::GpuSelection, ActiveGenerationStatus, DiskUsage, GenerateRequest, GpuBackend, GpuInfo,
    GpuWorkerState, ModelInfoExtended, ResourceSnapshot, ServerStatus, SseErrorEvent,
    SseProgressEvent,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use tokio_stream::StreamExt as _;
use utoipa::OpenApi;

use crate::model_manager;
use crate::state::{AppState, GenerationJob, SseCompletionPayload, SseMessage, SubmitError};

fn submit_error_to_api(e: SubmitError) -> ApiError {
    match e {
        SubmitError::Full { pending, capacity } => {
            ApiError::queue_full(format!("generation queue is full ({pending}/{capacity})"))
        }
        SubmitError::Shutdown => ApiError::internal("generation queue shut down"),
    }
}

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

    pub fn internal_with_status(msg: impl Into<String>, status: StatusCode) -> Self {
        Self {
            error: msg.into(),
            code: "INTERNAL_ERROR".to_string(),
            status,
        }
    }

    pub fn with_code(msg: impl Into<String>, code: impl Into<String>, status: StatusCode) -> Self {
        Self {
            error: msg.into(),
            code: code.into(),
            status,
        }
    }

    pub fn queue_job_not_found(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "QUEUE_JOB_NOT_FOUND".to_string(),
            status: StatusCode::NOT_FOUND,
        }
    }

    pub fn queue_job_running(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "QUEUE_JOB_RUNNING".to_string(),
            status: StatusCode::CONFLICT,
        }
    }

    /// Job cancelled while queued. 499 is the de-facto "client closed
    /// request" status (nginx); we reuse it for "request cancelled before
    /// the server did any work" so clients can distinguish cancellation
    /// from real inference failures.
    pub fn cancelled(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "CANCELLED".to_string(),
            status: StatusCode::from_u16(499).expect("499 is a valid status code"),
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

    pub fn queue_full(msg: impl Into<String>) -> Self {
        Self {
            error: msg.into(),
            code: "QUEUE_FULL".to_string(),
            status: StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status;
        // On queue-full (503), hint clients to retry with a short delay.
        if self.code == "QUEUE_FULL" {
            let mut headers = HeaderMap::new();
            headers.insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
            return (status, headers, Json(self)).into_response();
        }
        (status, Json(self)).into_response()
    }
}

// Re-export for tests — the canonical implementation lives in queue.rs.
#[cfg(test)]
use crate::queue::clean_error_message;

#[derive(OpenApi)]
#[openapi(
    paths(
        generate,
        generate_stream,
        expand_prompt,
        list_models,
        crate::catalog_api::list_loras,
        load_model,
        pull_model_endpoint,
        unload_model,
        delete_model,
        create_gallery_media_token,
        server_status,
        list_queue,
        patch_queue_job,
        cancel_queue_job,
        pause_queue,
        resume_queue,
        cancel_all_queue,
        list_history,
        delete_history,
        crate::routes_config::list_config,
        crate::routes_config::get_config_key,
        crate::routes_config::put_config_key,
        crate::routes_config::delete_config_key,
        crate::routes_config::list_config_profiles,
        crate::routes_config::put_config_profile,
        health,
        discovery_peers,
        capabilities_chain_limits,
        stream_events,
        crate::routes_chain::generate_chain,
        crate::routes_chain::generate_chain_stream,
        crate::routes_chain_jobs::create_chain_job,
        crate::routes_chain_jobs::list_chain_jobs,
        crate::routes_chain_jobs::get_chain_job,
        crate::routes_chain_jobs::chain_job_events,
        crate::routes_chain_jobs::resume_chain_job,
        crate::routes_chain_jobs::retake_chain_job,
        crate::routes_chain_jobs::cancel_chain_job,
        crate::routes_chain_jobs::delete_chain_job,
        crate::routes_chain_jobs::gc_chain_jobs,
        crate::routes_chain_jobs::chain_job_stage_preview,
    ),
    components(schemas(
        mold_core::GenerateRequest,
        mold_core::GenerateResponse,
        mold_core::ExpandRequest,
        mold_core::ExpandResponse,
        mold_core::ImageData,
        mold_core::OutputFormat,
        mold_core::ModelInfo,
        mold_core::LoraInfo,
        mold_core::ServerStatus,
        mold_core::ActiveGenerationStatus,
        mold_core::GpuInfo,
        mold_core::DiskUsage,
        mold_core::DiscoveryPeer,
        mold_core::SseProgressEvent,
        mold_core::SseCompleteEvent,
        mold_core::SseErrorEvent,
        mold_core::ChainRequest,
        mold_core::ChainResponse,
        mold_core::ChainStage,
        mold_core::ChainProgressEvent,
        mold_core::SseChainCompleteEvent,
        mold_core::chain_job::ChainJobSummary,
        mold_core::chain_job::ChainJobStageDetail,
        mold_core::chain_job::ChainJobDetail,
        mold_core::chain_job::ChainJobListing,
        mold_core::chain_job::CreateChainJobResponse,
        mold_core::chain_job::RetakeRequest,
        mold_core::chain_job::ChainJobEvent,
        mold_core::chain_job::FinalizeRecord,
        mold_core::chain_job::RetakeAmendment,
        mold_core::chain_job::ChainJobState,
        mold_core::chain_job::StageState,
        mold_core::chain_job::RetakeMode,
        mold_core::chain_job::GcOutcome,
        ModelInfoExtended,
        LoadModelBody,
        UnloadRequest,
        mold_core::ModelRemovalResponse,
        mold_core::KeptComponent,
        QueuePatchRequest,
        QueuePauseResponse,
        QueueCancelAllResponse,
        mold_core::HistoryEntry,
        mold_core::HistoryListing,
        mold_core::ConfigEntry,
        mold_core::ConfigListing,
        mold_core::ConfigProfiles,
        crate::routes_config::ConfigSetRequest,
        crate::routes_config::ProfileSetRequest,
        crate::job_registry::JobEntry,
        crate::job_registry::QueueListing,
        crate::chain_limits::ChainLimits,
        GalleryMediaTokenRequest,
        GalleryMediaTokenResponse,
    )),
    tags(
        (name = "generation", description = "Image generation"),
        (name = "models", description = "Model management"),
        (name = "server", description = "Server status and health"),
        (name = "chain-jobs", description = "Durable chained video jobs"),
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
        .route("/api/generate/estimate", post(generate_estimate))
        .route("/api/generate/stream", post(generate_stream))
        .route(
            "/api/generate/chain",
            post(crate::routes_chain::generate_chain),
        )
        .route(
            "/api/generate/chain/stream",
            post(crate::routes_chain::generate_chain_stream),
        )
        .route(
            "/api/chain-jobs",
            post(crate::routes_chain_jobs::create_chain_job)
                .get(crate::routes_chain_jobs::list_chain_jobs),
        )
        .route(
            "/api/chain-jobs/:id",
            get(crate::routes_chain_jobs::get_chain_job)
                .delete(crate::routes_chain_jobs::delete_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/events",
            get(crate::routes_chain_jobs::chain_job_events),
        )
        .route(
            "/api/chain-jobs/:id/resume",
            post(crate::routes_chain_jobs::resume_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/retake",
            post(crate::routes_chain_jobs::retake_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/cancel",
            post(crate::routes_chain_jobs::cancel_chain_job),
        )
        .route(
            "/api/chain-jobs/gc",
            post(crate::routes_chain_jobs::gc_chain_jobs),
        )
        .route(
            "/api/chain-jobs/:id/stages/:idx/preview",
            get(crate::routes_chain_jobs::chain_job_stage_preview),
        )
        .route("/api/expand", post(expand_prompt))
        .route("/api/models", get(list_models))
        .route("/api/models/:model", delete(delete_model))
        .route("/api/models/:model/components", get(model_components))
        .route("/api/loras", get(crate::catalog_api::list_loras))
        .route("/api/models/load", post(load_model))
        .route("/api/models/pull", post(pull_model_endpoint))
        .route("/api/models/unload", delete(unload_model))
        .route("/api/gallery", get(list_gallery))
        .route("/api/gallery/media-token", post(create_gallery_media_token))
        .route(
            "/api/gallery/image/:filename",
            get(get_gallery_image).delete(delete_gallery_image),
        )
        .route(
            "/api/gallery/thumbnail/:filename",
            get(get_gallery_thumbnail),
        )
        .route("/api/gallery/preview/:filename", get(get_gallery_preview))
        // ─── Downloads UI (Agent A) ────────────────────────────────────────
        .route("/api/downloads", get(list_downloads).post(create_download))
        .route("/api/downloads/:id", delete(delete_download))
        .route("/api/downloads/stream", get(stream_downloads))
        // ─── Catalog (live HF + Civitai proxy) ──────────────────────────
        .route(
            "/api/catalog/families",
            get(crate::catalog_api::list_families),
        )
        .route(
            "/api/catalog/search",
            get(crate::catalog_api::live_search_catalog),
        )
        .route(
            "/api/catalog/installed",
            get(crate::catalog_api::list_installed_catalog),
        )
        .route(
            "/api/catalog/credentials",
            get(crate::catalog_credentials::get_catalog_credentials),
        )
        .route(
            "/api/catalog/credentials/:provider",
            axum::routing::put(crate::catalog_credentials::put_catalog_credential)
                .delete(crate::catalog_credentials::delete_catalog_credential),
        )
        .route(
            "/api/catalog/*id",
            get(crate::catalog_api::get_catalog_entry)
                .post(crate::catalog_api::post_catalog_dispatch),
        )
        .route("/api/upscale", post(upscale))
        .route("/api/upscale/stream", post(upscale_stream))
        .route("/api/resources", get(get_resources))
        .route("/api/resources/stream", get(get_resources_stream))
        .route("/api/events", get(stream_events))
        .route("/api/status", get(server_status))
        .route("/api/queue", get(list_queue).delete(cancel_all_queue))
        .route("/api/queue/pause", post(pause_queue))
        .route("/api/queue/resume", post(resume_queue))
        .route(
            "/api/queue/:id",
            patch(patch_queue_job).delete(cancel_queue_job),
        )
        .route("/api/history", get(list_history).delete(delete_history))
        .route("/api/capabilities", get(server_capabilities))
        .route("/api/discovery/peers", get(discovery_peers))
        .route(
            "/api/capabilities/chain-limits",
            get(capabilities_chain_limits),
        )
        .route("/api/shutdown", post(shutdown_server))
        // ─── /api/config — HTTP counterpart of the `mold config` verbs ────
        .route("/api/config", get(crate::routes_config::list_config))
        .route(
            "/api/config/profiles",
            get(crate::routes_config::list_config_profiles),
        )
        .route(
            "/api/config/profile",
            axum::routing::put(crate::routes_config::put_config_profile),
        )
        .route(
            "/api/config/:key",
            get(crate::routes_config::get_config_key)
                .put(crate::routes_config::put_config_key)
                .delete(crate::routes_config::delete_config_key),
        )
        // Agent C (model-ui-overhaul §3): placement persistence.
        .route(
            "/api/config/model/:name/placement",
            get(get_model_placement)
                .put(put_model_placement)
                .delete(delete_model_placement),
        )
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
) -> Result<(Option<std::path::PathBuf>, Option<String>, Option<usize>), ApiError> {
    // NOTE: the capacity check is enforced inside `state.queue.submit(...)` so
    // that a burst of concurrent callers can't all slip past an open check
    // (classic TOCTOU).  The submit call in `generate`/`generate_stream` will
    // return `SubmitError::Full`, which is mapped to `ApiError::queue_full()`.
    apply_default_metadata_setting(state, request).await;

    let preferred_gpu = validate_multi_gpu_placement(state, request.placement.as_ref())?;

    // Expand prompt if requested (before validation, so the expanded prompt gets validated)
    maybe_expand_prompt(state, request, preferred_gpu).await?;

    // Catalog (`cv:*` / `hf:*`) IDs aren't in the static manifest, so the
    // pure-mold-core family lookup returns `None` for them. Run the live
    // single-id install first so the intent cache has the entry; then
    // feed its family string through as a hint so audio / keyframes /
    // pipeline gates work for installed Civitai LTX-2 checkpoints.
    //
    // A `Network` error here means Civitai/HF is unreachable — surface it
    // immediately as 502 rather than letting the user fall through to
    // a "not installed" 404 they can't act on.
    if let Err(e) = model_manager::install_catalog_model(state, &request.model).await {
        return Err(model_manager::install_error_to_api_error(&e));
    }
    let family_hint = model_manager::catalog_family_for(state, &request.model).await;

    // Resolve the model family for normalisation. `family_for_model` checks the
    // static manifest first (covers all built-in models), then falls back to the
    // catalog DB entry (covers `cv:*` / `hf:*` models installed above).
    let resolved_family = model_manager::family_for_model(state, &request.model).await;
    // Fill in a family-aware output format default when the caller omitted the
    // field. This must happen before validation so the validator sees a concrete
    // format and can gate on it correctly.
    request.normalise_output_format(resolved_family.as_deref());

    if let Err(e) = validate_generate_request(request, family_hint.as_deref()) {
        return Err(ApiError::validation(e));
    }

    resolve_server_local_media_paths(state, request).await?;

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

    Ok((output_dir, dim_warning, preferred_gpu))
}

/// Record an accepted generation prompt into prompt history (best-effort;
/// no-op when the metadata DB is disabled). Consecutive identical rows are
/// collapsed so batch siblings and retries don't spam duplicates. Records
/// what the user actually typed — callers capture the prompt before
/// `prepare_generation` runs prompt expansion.
fn record_prompt_history(state: &AppState, prompt: &str, negative: Option<&str>, model: &str) {
    let Some(db) = state.metadata_db.as_ref().as_ref() else {
        return;
    };
    let history = mold_db::PromptHistory::new(db);
    if let Ok(rows) = history.recent(1) {
        if rows.first().is_some_and(|latest| {
            latest.prompt == prompt
                && latest.model == model
                && latest.negative.as_deref() == negative
        }) {
            return;
        }
    }
    if let Err(e) = history.push(&mold_db::HistoryEntry {
        prompt: prompt.to_string(),
        negative: negative.map(str::to_string),
        model: model.to_string(),
        created_at_ms: 0, // stamped with now() by push()
    }) {
        tracing::warn!("failed to record prompt history: {e:#}");
    }
}

pub(crate) async fn resolve_server_local_media_paths(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    if request.audio_file_path.is_none() && request.source_video_path.is_none() {
        return Ok(());
    }

    let roots = state.config.read().await.resolved_media_roots();
    if let Some(path) = request.audio_file_path.as_deref() {
        let resolved = mold_core::resolve_server_media_path(path, &roots)
            .map_err(|e| ApiError::validation(format!("audio_file_path: {e}")))?;
        request.audio_file_path = Some(resolved.to_string_lossy().to_string());
    }
    if let Some(path) = request.source_video_path.as_deref() {
        let resolved = mold_core::resolve_server_media_path(path, &roots)
            .map_err(|e| ApiError::validation(format!("source_video_path: {e}")))?;
        request.source_video_path = Some(resolved.to_string_lossy().to_string());
    }

    Ok(())
}

fn active_gpu_selection(state: &AppState) -> GpuSelection {
    let ordinals: Vec<usize> = state
        .gpu_pool
        .workers
        .iter()
        .map(|w| w.gpu.ordinal)
        .collect();
    if ordinals.is_empty() {
        GpuSelection::All
    } else {
        GpuSelection::Specific(ordinals)
    }
}

fn validate_multi_gpu_placement(
    state: &AppState,
    placement: Option<&mold_core::types::DevicePlacement>,
) -> Result<Option<usize>, ApiError> {
    state
        .gpu_pool
        .resolve_explicit_placement_gpu(placement)
        .map_err(ApiError::validation)
}

fn select_aux_worker(
    state: &AppState,
) -> Result<std::sync::Arc<crate::gpu_pool::GpuWorker>, ApiError> {
    let mut workers: Vec<_> = state
        .gpu_pool
        .workers
        .iter()
        .filter(|w| !w.is_degraded())
        .cloned()
        .collect();
    workers.sort_by_key(|w| {
        (
            w.in_flight.load(Ordering::SeqCst),
            Reverse(w.gpu.total_vram_bytes),
        )
    });
    workers
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::internal("no GPU worker available for auxiliary workload"))
}

fn clear_global_upscaler_cache(state: &AppState) {
    if let Ok(mut cache) = state.upscaler_cache.try_lock() {
        if cache.is_some() {
            *cache = None;
            tracing::info!("upscaler cache cleared");
        }
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
        (status = 503, description = "Generation queue full"),
    )
)]
// The server always produces 1 image per request; batch looping (--batch N)
// is handled client-side by the CLI, which sends N requests with incrementing seeds.
async fn generate(
    State(state): State<AppState>,
    Json(mut req): Json<mold_core::GenerateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Capture before prepare_generation: prompt expansion mutates req.prompt,
    // and history should hold what the user typed.
    let typed = (
        req.prompt.clone(),
        req.negative_prompt.clone(),
        req.model.clone(),
    );
    let (output_dir, dim_warning, preferred_gpu) = prepare_generation(&state, &mut req).await?;
    record_prompt_history(&state, &typed.0, typed.1.as_deref(), &typed.2);

    tracing::info!(
        model = %req.model,
        prompt = %req.prompt,
        width = req.width,
        height = req.height,
        steps = req.steps,
        guidance = req.guidance,
        seed = ?req.seed,
        format = %req.resolved_output_format(),
        lora = ?req.lora.as_ref().map(|l| &l.path),
        lora_scale = ?req.lora.as_ref().map(|l| l.scale),
        loras = ?req.loras.as_ref().map(|v| v.iter().map(|l| &l.path).collect::<Vec<_>>()),
        "generate request"
    );

    // Submit to generation queue
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    // Non-streaming path still gets a registry entry so an HTTP client
    // hanging on the response is visible via GET /api/queue alongside
    // SSE clients. Cleanup happens unconditionally on every terminal
    // path (drop guard in `gpu_worker::process_job`).
    let job_id = uuid::Uuid::new_v4().to_string();
    // Metadata-shaped request params ride the queue listing so any client
    // can inspect the job and reuse its settings. `seed_pinned` records
    // whether the request pinned a seed — metadata's required seed field
    // can't distinguish an explicit 0 from "let the server pick".
    let queue_metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
        &req,
        req.seed.unwrap_or(0),
        req.scheduler,
        mold_core::build_info::version_string(),
    ));
    let cancel = state.job_registry.register_job(
        &job_id,
        &req.model,
        preferred_gpu,
        Some(req.seed.is_some()),
        Some(queue_metadata),
    );
    let job = GenerationJob {
        id: job_id.clone(),
        request: req,
        completion_payload: SseCompletionPayload::Full,
        progress_tx: None,
        result_tx,
        output_dir,
    };

    let _position = state
        .queue
        .submit(job, state.queue_capacity)
        .await
        .inspect_err(|_| state.job_registry.remove(&job_id))
        .map_err(submit_error_to_api)?;

    // Wait for the queue worker to process the job — or for
    // `DELETE /api/queue/:id` to cancel it while it's still queued.
    // Returning here drops `result_rx`, so the worker's is_closed()
    // check skips the job when it eventually reaches the head.
    let result = tokio::select! {
        result = result_rx => result
            .map_err(|_| ApiError::internal("generation queue worker dropped the job"))?,
        _ = cancel.notified() => {
            return Err(ApiError::cancelled(format!(
                "generation job {job_id} was cancelled while queued"
            )));
        }
    };

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
            if let Some(ordinal) = response.gpu {
                headers.insert(
                    "x-mold-gpu",
                    HeaderValue::from_str(&ordinal.to_string()).map_err(|e| {
                        ApiError::internal(format!("failed to serialize gpu header: {e}"))
                    })?,
                );
            }
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
        Err(err_msg) => {
            // The multi-GPU dispatcher sends a queue-full error through result_tx
            // when a per-worker channel is saturated; surface that as a proper 503
            // instead of the generic INFERENCE_ERROR 500.
            if err_msg.contains("queue is full") {
                Err(ApiError::queue_full(err_msg))
            } else {
                Err(ApiError::inference(err_msg))
            }
        }
    }
}

fn validate_generate_request(
    req: &mold_core::GenerateRequest,
    family_hint: Option<&str>,
) -> Result<(), String> {
    mold_core::validate_generate_request_with_family(req, family_hint)
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
    preferred_gpu: Option<usize>,
) -> Result<(), ApiError> {
    if req.expand != Some(true) {
        return Ok(());
    }

    let config = state.config.read().await;
    let config_snapshot = config.clone();
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

    let expander = create_server_expander(
        &config_snapshot,
        &expand_settings,
        active_gpu_selection(state),
        preferred_gpu,
    )?;
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
    _config: &mold_core::Config,
    settings: &mold_core::ExpandSettings,
    _gpu_selection: GpuSelection,
    _preferred_gpu: Option<usize>,
) -> Result<Box<dyn mold_core::PromptExpander>, ApiError> {
    if let Some(api_expander) = settings.create_api_expander() {
        return Ok(Box::new(api_expander));
    }

    #[cfg(feature = "expand")]
    {
        if let Some(local) =
            mold_inference::expand::LocalExpander::from_config(_config, Some(&settings.model))
        {
            return Ok(Box::new(
                local
                    .with_gpu_selection(_gpu_selection)
                    .with_preferred_gpu(_preferred_gpu),
            ));
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

/// Build the per-request expand config: settings-derived knobs plus the
/// request's optional visual style (bake-and-clear — the style reaches the
/// expander as a natural-language instruction, never a literal suffix).
fn expand_config_for_request(
    settings: &mold_core::ExpandSettings,
    req: &mold_core::ExpandRequest,
) -> mold_core::ExpandConfig {
    let mut config = settings.to_expand_config(&req.model_family, req.variations);
    config.style = req.style.clone();
    config
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
    let expand_config = expand_config_for_request(&expand_settings, &req);
    let prompt = req.prompt.clone();
    let config_snapshot = config.clone();
    drop(config);

    let expander = create_server_expander(
        &config_snapshot,
        &expand_settings,
        active_gpu_selection(&state),
        None,
    )?;
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

    let resp = if state.gpu_pool.worker_count() > 0 {
        let worker = select_aux_worker(&state)?;
        worker.in_flight.fetch_add(1, Ordering::SeqCst);
        let worker_clone = worker.clone();
        let joined =
            tokio::task::spawn_blocking(move || -> anyhow::Result<mold_core::UpscaleResponse> {
                struct ThreadGpuGuard;
                impl Drop for ThreadGpuGuard {
                    fn drop(&mut self) {
                        mold_inference::device::clear_thread_gpu_ordinal();
                    }
                }

                mold_inference::device::init_thread_gpu_ordinal(worker_clone.gpu.ordinal);
                let _thread_gpu = ThreadGpuGuard;
                let _load_lock = worker_clone.model_load_lock.lock().unwrap();
                crate::gpu_worker::ensure_worker_not_poisoned(&worker_clone, &model_name_owned)?;
                let mut engine = mold_inference::create_upscale_engine(
                    model_name_owned,
                    weights_path,
                    mold_inference::LoadStrategy::Eager,
                    worker_clone.gpu.ordinal,
                )?;
                engine.upscale(&req)
            })
            .await;
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        let result =
            joined.map_err(|e| ApiError::internal(format!("upscale task panicked: {e}")))?;
        if let Err(error) = &result {
            crate::gpu_worker::quarantine_if_fatal_cuda_error(&worker, error);
        }
        result.map_err(|e| ApiError::internal(format!("upscale failed: {e}")))?
    } else {
        let upscaler_cache = state.upscaler_cache.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<mold_core::UpscaleResponse> {
            let mut cache = upscaler_cache.lock().unwrap_or_else(|e| e.into_inner());

            // Reuse cached engine if same model.
            let needs_new = cache
                .as_ref()
                .is_none_or(|e| e.model_name() != model_name_owned);
            if needs_new {
                let new_engine = mold_inference::create_upscale_engine(
                    model_name_owned,
                    weights_path,
                    mold_inference::LoadStrategy::Eager,
                    0,
                )?;
                *cache = Some(new_engine);
            }

            cache.as_mut().unwrap().upscale(&req)
        })
        .await
        .map_err(|e| ApiError::internal(format!("upscale task panicked: {e}")))?
        .map_err(|e| ApiError::internal(format!("upscale failed: {e}")))?
    };

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

        let result = if state_clone.gpu_pool.worker_count() > 0 {
            match select_aux_worker(&state_clone) {
                Ok(worker) => {
                    worker.in_flight.fetch_add(1, Ordering::SeqCst);
                    let worker_clone = worker.clone();
                    let tx_for_worker = tx.clone();
                    let model_name_for_worker = model_name_owned.clone();
                    let weights_path_for_worker = weights_path.clone();
                    let req_for_worker = req.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        struct ThreadGpuGuard;
                        impl Drop for ThreadGpuGuard {
                            fn drop(&mut self) {
                                mold_inference::device::clear_thread_gpu_ordinal();
                            }
                        }

                        mold_inference::device::init_thread_gpu_ordinal(worker_clone.gpu.ordinal);
                        let _thread_gpu = ThreadGpuGuard;
                        let _load_lock = worker_clone.model_load_lock.lock().unwrap();
                        if let Err(error) = crate::gpu_worker::ensure_worker_not_poisoned(
                            &worker_clone,
                            &model_name_for_worker,
                        ) {
                            let _ =
                                tx_for_worker.send(SseMessage::Error(mold_core::SseErrorEvent {
                                    message: error.to_string(),
                                }));
                            return;
                        }
                        let _ = tx_for_worker.send(SseMessage::Progress(
                            mold_core::SseProgressEvent::StageStart {
                                name: format!(
                                    "Loading upscaler model on GPU {}",
                                    worker_clone.gpu.ordinal
                                ),
                            },
                        ));
                        let mut engine = match mold_inference::create_upscale_engine(
                            model_name_for_worker,
                            weights_path_for_worker,
                            mold_inference::LoadStrategy::Eager,
                            worker_clone.gpu.ordinal,
                        ) {
                            Ok(engine) => engine,
                            Err(e) => {
                                crate::gpu_worker::quarantine_if_fatal_cuda_error(
                                    &worker_clone,
                                    &e,
                                );
                                let _ = tx_for_worker.send(SseMessage::Error(
                                    mold_core::SseErrorEvent {
                                        message: format!("failed to load upscaler: {e}"),
                                    },
                                ));
                                return;
                            }
                        };

                        let tx_progress = tx_for_worker.clone();
                        engine.set_on_progress(Box::new(move |event| {
                            let sse_event: mold_core::SseProgressEvent = event.into();
                            let _ = tx_progress.send(SseMessage::Progress(sse_event));
                        }));

                        match engine.upscale(&req_for_worker) {
                            Ok(resp) => {
                                let image_b64 = base64::engine::general_purpose::STANDARD
                                    .encode(&resp.image.data);
                                let _ = tx_for_worker.send(SseMessage::UpscaleComplete(
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
                                crate::gpu_worker::quarantine_if_fatal_cuda_error(
                                    &worker_clone,
                                    &e,
                                );
                                let _ = tx_for_worker.send(SseMessage::Error(
                                    mold_core::SseErrorEvent {
                                        message: format!("upscale failed: {e}"),
                                    },
                                ));
                            }
                        }

                        engine.clear_on_progress();
                    })
                    .await;
                    worker.in_flight.fetch_sub(1, Ordering::SeqCst);
                    result
                }
                Err(e) => {
                    let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent {
                        message: e.error,
                    }));
                    return;
                }
            }
        } else {
            let model_name_for_cache = model_name_owned.clone();
            let weights_path_for_cache = weights_path.clone();
            let req_for_cache = req.clone();
            tokio::task::spawn_blocking(move || {
                let mut cache = upscaler_cache.lock().unwrap();

                let needs_new = cache
                    .as_ref()
                    .is_none_or(|e| e.model_name() != model_name_for_cache);
                if needs_new {
                    let _ = tx.send(SseMessage::Progress(
                        mold_core::SseProgressEvent::StageStart {
                            name: "Loading upscaler model".to_string(),
                        },
                    ));
                    match mold_inference::create_upscale_engine(
                        model_name_for_cache,
                        weights_path_for_cache,
                        mold_inference::LoadStrategy::Eager,
                        0,
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
                let tx_progress = tx.clone();
                engine.set_on_progress(Box::new(move |event| {
                    let sse_event: mold_core::SseProgressEvent = event.into();
                    let _ = tx_progress.send(SseMessage::Progress(sse_event));
                }));

                match engine.upscale(&req_for_cache) {
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
            .await
        };

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

const SSE_PAYLOAD_HEADER: &str = "x-mold-sse-payload";

pub(crate) fn requested_sse_completion_payload(
    headers: &HeaderMap,
) -> Result<SseCompletionPayload, ApiError> {
    let Some(value) = headers.get(SSE_PAYLOAD_HEADER) else {
        return Ok(SseCompletionPayload::Full);
    };
    match value.to_str().ok() {
        Some("metadata-only") => Ok(SseCompletionPayload::MetadataOnly),
        _ => Err(ApiError::validation(
            "X-Mold-SSE-Payload must be 'metadata-only' when provided",
        )),
    }
}

#[utoipa::path(
    post,
    path = "/api/generate/stream",
    tag = "generation",
    request_body = mold_core::GenerateRequest,
    params((
        "X-Mold-SSE-Payload" = Option<String>,
        Header,
        description = "Set to metadata-only to omit encoded media and return the saved gallery filename"
    )),
    responses(
        (status = 200, description = "SSE event stream with progress and result"),
        (status = 404, description = "Model not downloaded"),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Inference error"),
        (status = 503, description = "Generation queue full"),
    )
)]
async fn generate_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<mold_core::GenerateRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let completion_payload = requested_sse_completion_payload(&headers)?;
    // Capture before prepare_generation: prompt expansion mutates req.prompt,
    // and history should hold what the user typed.
    let typed = (
        req.prompt.clone(),
        req.negative_prompt.clone(),
        req.model.clone(),
    );
    let (output_dir, dim_warning, preferred_gpu) = prepare_generation(&state, &mut req).await?;
    if completion_payload == SseCompletionPayload::MetadataOnly && output_dir.is_none() {
        return Err(ApiError::validation(
            "metadata-only SSE completions require server gallery output to be enabled",
        ));
    }
    record_prompt_history(&state, &typed.0, typed.1.as_deref(), &typed.2);

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
    // Assign a server-side ID and register before submit so the entry is
    // visible to /api/queue from the moment we accept the request.
    let job_id = uuid::Uuid::new_v4().to_string();
    // Metadata-shaped request params ride the queue listing so any client
    // can inspect the job and reuse its settings. `seed_pinned` records
    // whether the request pinned a seed — metadata's required seed field
    // can't distinguish an explicit 0 from "let the server pick".
    let queue_metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
        &req,
        req.seed.unwrap_or(0),
        req.scheduler,
        mold_core::build_info::version_string(),
    ));
    let cancel = state.job_registry.register_job(
        &job_id,
        &req.model,
        preferred_gpu,
        Some(req.seed.is_some()),
        Some(queue_metadata),
    );
    let job = GenerationJob {
        id: job_id.clone(),
        request: req,
        completion_payload,
        progress_tx: Some(tx.clone()),
        result_tx,
        output_dir,
    };

    let position = state
        .queue
        .submit(job, state.queue_capacity)
        .await
        .inspect_err(|_| state.job_registry.remove(&job_id))
        .map_err(submit_error_to_api)?;

    // First event the client sees — carries the position AND the server
    // ID so the SPA can later reconcile this job against /api/queue.
    let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued {
        position,
        id: job_id,
    }));

    // Hold `tx` alive in a background task until the job completes, so the SSE
    // stream never closes prematurely even if the queue worker hasn't received
    // the job yet. A `DELETE /api/queue/:id` cancel resolves the select's
    // second arm: emit a terminal error event, then drop both channel ends —
    // dropping `result_rx` closes the job's result channel, which is what
    // makes the queue worker/dispatcher skip the job when it dequeues it.
    tokio::spawn(async move {
        tokio::select! {
            _ = result_rx => {}
            _ = cancel.notified() => {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: "cancelled".to_string(),
                }));
            }
        }
        drop(tx); // closes the SSE stream
    });

    // Build SSE stream from the channel receiver. Error events are terminal
    // on this endpoint (the worker always follows them by resolving the
    // job), so end the stream right after one — a cancelled job's queued
    // `GenerationJob` still holds a sender clone, and without the explicit
    // break the stream would stay open until the worker drained it.
    let stream = async_stream::stream! {
        let mut rx = rx;
        while let Some(msg) = rx.recv().await {
            let is_error = matches!(msg, SseMessage::Error(_));
            yield Ok::<_, Infallible>(sse_message_to_event(msg));
            if is_error {
                break;
            }
        }
    };

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

async fn generate_estimate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<mold_core::GenerationMemoryEstimate>, ApiError> {
    Ok(Json(
        model_manager::estimate_generation_memory(&state, &req).await?,
    ))
}

async fn model_components(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<mold_core::ModelComponentsResponse>, ApiError> {
    Ok(Json(
        model_manager::model_component_status(&state, &model).await?,
    ))
}

// ── /api/models/load ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct LoadModelBody {
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    /// Target GPU ordinal (multi-GPU only). If omitted, the server uses its
    /// default placement strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
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
    if let Err(e) = model_manager::install_catalog_model(&state, &body.model).await {
        return Err(model_manager::install_error_to_api_error(&e));
    }
    let _ = model_manager::check_model_available(&state, &body.model).await?;

    // Multi-GPU path: route through the pool.
    if state.gpu_pool.worker_count() > 0 {
        let worker = match body.gpu {
            Some(ordinal) => state
                .gpu_pool
                .workers
                .iter()
                .find(|w| w.gpu.ordinal == ordinal)
                .cloned()
                .ok_or_else(|| {
                    ApiError::not_found(format!("no GPU worker with ordinal {ordinal}"))
                })?,
            None => {
                let est = crate::queue::estimate_model_vram(&body.model);
                state
                    .gpu_pool
                    .select_worker(&body.model, est)
                    .ok_or_else(|| {
                        ApiError::internal(format!(
                            "no GPU available to load model '{}'",
                            body.model
                        ))
                    })?
            }
        };
        let config_snapshot = state.config.read().await.clone();
        let model_name = body.model.clone();
        let worker_clone = worker.clone();
        tokio::task::spawn_blocking(move || {
            crate::gpu_worker::load_blocking(&worker_clone, &model_name, &config_snapshot)
        })
        .await
        .map_err(|e| ApiError::internal(format!("model load task failed: {e}")))?
        .map_err(|e| ApiError::internal(format!("model load error: {e}")))?;

        tracing::info!(
            model = %body.model,
            gpu = worker.gpu.ordinal,
            "model loaded via API"
        );
        return Ok(StatusCode::OK);
    }

    // Legacy single-GPU path (no workers discovered). No resolution context
    // here — admin load uses size-only peak via the `None` hint.
    model_manager::ensure_model_ready(&state, &body.model, None, None, false).await?;
    tracing::info!(model = %body.model, gpu = ?body.gpu, "model loaded via API (legacy)");
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

    if body.model.starts_with("cv:") || body.model.starts_with("hf:") {
        if let Err(e) = model_manager::install_catalog_model(&state, &body.model).await {
            return Err(model_manager::install_error_to_api_error(&e));
        }
        if model_manager::check_model_available(&state, &body.model)
            .await
            .is_ok()
        {
            return Ok(PullResponse::Text(format!(
                "model '{}' is already present",
                body.model
            )));
        }
        let companion_names = {
            let intents = state.catalog_intents.read().await;
            intents
                .get(&body.model)
                .map(|intent| {
                    intent
                        .companions
                        .iter()
                        .map(|companion| companion.name.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let models_dir = state.config.read().await.resolved_models_dir();
        let companion_jobs = crate::catalog_api::enqueue_missing_companions(
            &companion_names,
            &models_dir,
            &state.downloads,
            Some(&body.model),
            None,
        )
        .await;
        let primary_job = crate::catalog_api::enqueue_catalog_primary_repair(&state, &body.model)
            .await
            .map_err(|(status, msg)| ApiError::internal_with_status(msg, status))?;
        if !companion_jobs.is_empty() || primary_job.is_some() {
            let primary_count = usize::from(primary_job.is_some());
            return Ok(PullResponse::Text(format!(
                "queued repair for model '{}' ({} primary job(s), {} companion job(s))",
                body.model,
                primary_count,
                companion_jobs.len()
            )));
        }
        model_manager::check_model_available(&state, &body.model).await?;
        return Ok(PullResponse::Text(format!(
            "model '{}' is already present",
            body.model
        )));
    }

    // Enqueue via the queue. Treat idempotent AlreadyPresent as success.
    let (job_id, _position) = match state.downloads.enqueue(body.model.clone()).await {
        Ok((id, pos, _)) => (id, pos),
        Err(crate::downloads::EnqueueError::UnknownModel(_)) => {
            return Err(ApiError::unknown_model(format!(
                "unknown model '{}'. Run 'mold list' to see available models.",
                body.model
            )));
        }
        Err(crate::downloads::EnqueueError::LockPoisoned) => {
            return Err(ApiError::internal("download queue state is corrupt"));
        }
    };

    if !wants_sse {
        // Await terminal event for this job, return plain text.
        let mut rx = state.downloads.subscribe();
        loop {
            match rx.recv().await {
                Ok(mold_core::types::DownloadEvent::JobDone { id, model }) if id == job_id => {
                    return Ok(PullResponse::Text(format!(
                        "model '{model}' pulled successfully"
                    )));
                }
                Ok(mold_core::types::DownloadEvent::JobFailed { id, error }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "failed to pull model '{}': {error}",
                        body.model
                    )));
                }
                Ok(mold_core::types::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "pull of '{}' was cancelled",
                        body.model
                    )));
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err(ApiError::internal("download queue channel closed"));
                }
            }
        }
    }

    // SSE: re-emit queue events shaped like the legacy SseProgressEvent::DownloadProgress
    // so the TUI's existing consumer continues to work unchanged.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();
    let mut events = state.downloads.subscribe();
    let model_for_cb = body.model.clone();
    tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(mold_core::types::DownloadEvent::Started {
                    id,
                    files_total,
                    bytes_total,
                }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::DownloadProgress {
                        filename: String::new(),
                        file_index: 0,
                        total_files: files_total,
                        bytes_downloaded: 0,
                        bytes_total,
                        batch_bytes_downloaded: 0,
                        batch_bytes_total: bytes_total,
                        batch_elapsed_ms: 0,
                    }));
                }
                Ok(mold_core::types::DownloadEvent::Progress {
                    id,
                    files_done,
                    bytes_done,
                    current_file,
                }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::DownloadProgress {
                        filename: current_file.unwrap_or_default(),
                        file_index: files_done,
                        total_files: 0,
                        bytes_downloaded: bytes_done,
                        bytes_total: 0,
                        batch_bytes_downloaded: bytes_done,
                        batch_bytes_total: 0,
                        batch_elapsed_ms: 0,
                    }));
                }
                Ok(mold_core::types::DownloadEvent::JobDone { id, .. }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                        model: model_for_cb.clone(),
                    }));
                    break;
                }
                Ok(mold_core::types::DownloadEvent::JobFailed { id, error }) if id == job_id => {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent { message: error }));
                    break;
                }
                Ok(mold_core::types::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent {
                        message: "pull cancelled".into(),
                    }));
                    break;
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
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

/// Optional request body for unload — clients may specify a model or GPU target.
/// An empty body (or no body) unloads the active model on the legacy path.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
pub struct UnloadRequest {
    /// Specific model to unload. If omitted, the active model is unloaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Target GPU ordinal (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
}

#[utoipa::path(
    delete,
    path = "/api/models/unload",
    tag = "models",
    request_body(content = Option<UnloadRequest>, content_type = "application/json"),
    responses(
        (status = 200, description = "Model unloaded or no model was loaded", body = String),
    )
)]
async fn unload_model(
    State(state): State<AppState>,
    body: Option<Json<UnloadRequest>>,
) -> Result<impl IntoResponse, ApiError> {
    let req = body.map(|b| b.0).unwrap_or_default();
    tracing::debug!(model = ?req.model, gpu = ?req.gpu, "unload request");
    clear_global_upscaler_cache(&state);

    // Multi-GPU path: target specific GPU or model across the pool.
    if state.gpu_pool.worker_count() > 0 {
        // Select the workers to unload from.
        let targets: Vec<_> = match (req.gpu, req.model.as_deref()) {
            (Some(ordinal), _) => state
                .gpu_pool
                .workers
                .iter()
                .filter(|w| w.gpu.ordinal == ordinal)
                .cloned()
                .collect(),
            (None, Some(model)) => state
                .gpu_pool
                .workers
                .iter()
                .filter(|w| {
                    let cache = w.model_cache.lock().unwrap();
                    cache
                        .get(model)
                        .map(|e| e.residency == crate::model_cache::ModelResidency::Gpu)
                        .unwrap_or(false)
                })
                .cloned()
                .collect(),
            (None, None) => state.gpu_pool.workers.clone(),
        };

        if targets.is_empty() {
            return Ok((StatusCode::OK, "no model loaded".to_string()));
        }

        let mut unloaded_pairs: Vec<(usize, String)> = Vec::new();
        for worker in targets {
            let worker_clone = worker.clone();
            let result = tokio::task::spawn_blocking(move || {
                crate::gpu_worker::unload_blocking(&worker_clone)
            })
            .await
            .map_err(|e| ApiError::internal(format!("unload task failed: {e}")))?;
            if let Some(name) = result {
                unloaded_pairs.push((worker.gpu.ordinal, name));
            }
        }

        let msg = if unloaded_pairs.is_empty() {
            "no model loaded".to_string()
        } else {
            let joined: Vec<String> = unloaded_pairs
                .iter()
                .map(|(o, m)| format!("gpu{o}:{m}"))
                .collect();
            format!("unloaded {}", joined.join(", "))
        };
        return Ok((StatusCode::OK, msg));
    }

    // Legacy single-GPU path.
    Ok((StatusCode::OK, model_manager::unload_model(&state).await))
}

// ── DELETE /api/models/:model ─────────────────────────────────────────────────

/// True when an engine for `canonical` is currently GPU-resident (or mid-
/// generation) anywhere on this server — the single-GPU cache, any pool
/// worker's cache, or an active-generation snapshot on either path.
async fn model_is_gpu_resident(state: &AppState, canonical: &str) -> bool {
    {
        let cache = state.model_cache.lock().await;
        if cache.active_model() == Some(canonical) {
            return true;
        }
    }
    if state
        .active_generation
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|g| g.model == canonical)
    {
        return true;
    }
    for worker in &state.gpu_pool.workers {
        if let Ok(active) = worker.active_generation.read() {
            if active.as_ref().is_some_and(|g| g.model == canonical) {
                return true;
            }
        }
        if let Ok(cache) = worker.model_cache.lock() {
            if cache.active_model() == Some(canonical) {
                return true;
            }
        }
    }
    false
}

/// Remove a downloaded model's files — the HTTP counterpart of `mold rm`.
///
/// Ref-counts every file path across all installed models and deletes only
/// paths exclusively owned by this model; components still referenced by
/// another downloaded model (shared T5/CLIP/Qwen encoders, VAEs) are kept
/// and reported in `kept` with the surviving referents. hf-hub cache blobs
/// hardlinked to the deleted clean paths are removed too, so `freed_bytes`
/// reflects real disk savings.
#[utoipa::path(
    delete,
    path = "/api/models/{model}",
    tag = "models",
    params(("model" = String, Path, description = "Model name (e.g. flux-schnell:q8)")),
    responses(
        (status = 200, description = "Model removed", body = mold_core::ModelRemovalResponse),
        (status = 404, description = "Model not installed"),
        (status = 409, description = "Model is currently loaded — unload it first"),
    )
)]
async fn delete_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<mold_core::ModelRemovalResponse>, ApiError> {
    let canonical = mold_core::manifest::resolve_model_name(&model);

    // Refuse while the model is GPU-resident — there is no safe way to pull
    // files out from under a loaded engine. Check the raw input too: engines
    // register under their own model_name, which for non-manifest models may
    // not round-trip through resolve_model_name. (Best-effort check: a load
    // that races past it just keeps working off its already-open mmaps.)
    if model_is_gpu_resident(&state, &canonical).await
        || (model != canonical && model_is_gpu_resident(&state, &model).await)
    {
        return Err(ApiError::with_code(
            format!(
                "model '{canonical}' is currently loaded; unload it first (DELETE /api/models/unload)"
            ),
            "MODEL_LOADED",
            StatusCode::CONFLICT,
        ));
    }

    // Hold the config write lock across plan + delete so a concurrent pull
    // or placement write can't interleave with the removal.
    let mut config = state.config.write().await;
    let in_config = config.models.contains_key(&canonical);
    if !in_config && !config.manifest_model_is_downloaded(&canonical) {
        return Err(ApiError::with_code(
            format!("model '{canonical}' is not installed"),
            "UNKNOWN_MODEL",
            StatusCode::NOT_FOUND,
        ));
    }

    tracing::info!(model = %canonical, "model removal requested");
    let plan = mold_core::removal::plan_removal(&config, &canonical);
    let outcome = mold_core::removal::execute_removal(&config, &plan);
    for warning in &outcome.warnings {
        tracing::warn!(model = %canonical, "model removal: {warning}");
    }

    mold_core::download::remove_pulling_marker(&canonical);
    if in_config {
        config.remove_model(&canonical);
        if let Err(e) = config.save() {
            tracing::warn!("failed to persist model removal to config.toml: {e}");
        }
    }
    drop(config);

    // Evict any parked (non-GPU-resident) engine so a later request can't
    // reactivate an engine whose files are gone.
    {
        let mut cache = state.model_cache.lock().await;
        let _ = cache.remove(&canonical);
    }
    for worker in &state.gpu_pool.workers {
        if let Ok(mut cache) = worker.model_cache.lock() {
            let _ = cache.remove(&canonical);
        }
    }

    let kept = plan
        .shared_files
        .iter()
        .map(|(path, used_by)| mold_core::KeptComponent {
            component: path.clone(),
            used_by: used_by.clone(),
        })
        .collect();

    Ok(Json(mold_core::ModelRemovalResponse {
        removed: outcome.removed,
        kept,
        freed_bytes: outcome.freed_bytes,
    }))
}

// ── /api/status ───────────────────────────────────────────────────────────────

/// Disk usage for the filesystem holding `dir`: among mounts that are a
/// prefix of the path, the longest one wins (`/data/models` must resolve to
/// the `/data` mount, not `/`). `None` when no mount matches.
fn disk_usage_for_path(
    disks: &[(std::path::PathBuf, u64, u64)],
    dir: &std::path::Path,
) -> Option<DiskUsage> {
    disks
        .iter()
        .filter(|(mount, _, _)| dir.starts_with(mount))
        .max_by_key(|(mount, _, _)| mount.as_os_str().len())
        .map(|&(_, total_bytes, free_bytes)| DiskUsage {
            total_bytes,
            free_bytes,
        })
}

/// Resolve symlinks in the models dir before mount matching — a symlinked
/// models dir (`ln -s /Volumes/Big/models ~/.mold/models`) must report the
/// target's volume, not the symlink's. Falls back to the literal path when
/// canonicalization fails (missing dir, permissions).
fn canonical_models_dir(dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// Snapshot disk stats for the filesystem backing the models dir. Blocking:
/// canonicalize hits the filesystem and the disk refresh calls statvfs(2) on
/// every mount (which can stall on wedged FUSE mounts) — callers on the async
/// path must wrap this in `spawn_blocking`.
fn models_disk_usage(dir: &std::path::Path) -> Option<DiskUsage> {
    let dir = canonical_models_dir(dir);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let entries: Vec<(std::path::PathBuf, u64, u64)> = disks
        .list()
        .iter()
        .map(|d| {
            (
                d.mount_point().to_path_buf(),
                d.total_space(),
                d.available_space(),
            )
        })
        .collect();
    disk_usage_for_path(&entries, &dir)
}

#[utoipa::path(
    get,
    path = "/api/status",
    tag = "server",
    responses(
        (status = 200, description = "Server status", body = ServerStatus),
    )
)]
async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    // Disk stats are blocking syscalls (canonicalize + statvfs per mount, and
    // statvfs can hang outright on a wedged FUSE mount) — never run or await
    // them on the request path. Serve the cached snapshot; when it has gone
    // stale, the single winning request kicks a background refresh. The first
    // poll after boot reports no disk stats — pollers pick them up next round.
    let (models_disk, needs_disk_refresh) = state.models_disk_cache.read();
    if needs_disk_refresh {
        let cache = state.models_disk_cache.clone();
        let models_dir = state.config.read().await.resolved_models_dir();
        tokio::task::spawn_blocking(move || cache.store(models_disk_usage(&models_dir)));
    }

    // Aggregate worker state from the pool, then overlay physical memory from
    // the always-on NVML/nvidia-smi resource sampler. CUDA context queries can
    // fail after an illegal-access Xid and used to turn a real 35 GB leak into
    // a misleading zero in /api/status.
    let mut gpu_statuses = state.gpu_pool.gpu_status();
    if let Some(resources) = state.resources.latest() {
        let visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        for status in &mut gpu_statuses {
            let Some(physical_ordinal) = crate::resources::physical_ordinal_for_worker(
                status.ordinal,
                visible_devices.as_deref(),
            ) else {
                continue;
            };
            if let Some(snapshot) = resources
                .gpus
                .iter()
                .find(|gpu| gpu.ordinal == physical_ordinal)
            {
                status.vram_total_bytes = snapshot.vram_total;
                status.vram_used_bytes = snapshot.vram_used;
            }
        }
    }
    let has_gpus = !gpu_statuses.is_empty();

    // Collect loaded models from GPU workers.
    let gpu_models_loaded: Vec<String> = gpu_statuses
        .iter()
        .filter_map(|g| g.loaded_model.clone())
        .collect();
    let gpu_busy = gpu_statuses
        .iter()
        .any(|g| g.state == GpuWorkerState::Generating);

    // Pull current_generation from the first busy worker (multi-GPU) or
    // from the legacy snapshot.
    let multi_gpu_current_gen = if has_gpus {
        state.gpu_pool.workers.iter().find_map(|w| {
            let gen = w.active_generation.read().ok()?;
            gen.as_ref().map(|g| ActiveGenerationStatus {
                model: g.model.clone(),
                prompt_sha256: g.prompt_sha256.clone(),
                started_at_unix_ms: g.started_at_unix_ms,
                elapsed_ms: g.started_at.elapsed().as_millis() as u64,
            })
        })
    } else {
        None
    };

    // Fall back to legacy single-GPU snapshot for backwards compat.
    let (models_loaded, busy, current_generation) = if has_gpus {
        (gpu_models_loaded, gpu_busy, multi_gpu_current_gen)
    } else {
        let snapshot = state.model_cache.lock().await.snapshot();
        let models = match (snapshot.model_name, snapshot.is_loaded) {
            (Some(model_name), true) => vec![model_name],
            _ => vec![],
        };
        let gen = state
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
        let is_busy = gen.is_some();
        (models, is_busy, gen)
    };

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
        gpus: if has_gpus { Some(gpu_statuses) } else { None },
        queue_depth: Some(state.queue.pending()),
        queue_capacity: Some(state.queue_capacity),
        queue_paused: Some(state.queue_pause.is_paused()),
        instance_id: Some(state.instance_id.as_ref().clone()),
        models_disk,
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

// ── /api/queue ───────────────────────────────────────────────────────────────

/// Snapshot of every job currently queued or running on the server. Clients
/// (notably the web SPA) poll this to reconcile their local card list — any
/// "running" card whose server id isn't here is a zombie left over from a
/// dropped SSE stream and should be dead-lettered.
#[utoipa::path(
    get,
    path = "/api/queue",
    tag = "queue",
    responses(
        (status = 200, description = "Queue snapshot", body = crate::job_registry::QueueListing),
    )
)]
async fn list_queue(State(state): State<AppState>) -> Json<crate::job_registry::QueueListing> {
    Json(state.job_registry.snapshot())
}

/// Wrap any present JSON value (including `null`) in `Some`, so a field using
/// this as its `deserialize_with` distinguishes *absent* (`None`) from an
/// explicit `null` (`Some(None)`). Lets a `position`-only PATCH omit
/// `target_gpu` without resetting the lane to Auto.
fn deserialize_some<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct QueuePatchRequest {
    /// Preferred GPU lane. Absent leaves the lane unchanged; `null` means Auto;
    /// a number pins that ordinal. The double `Option` distinguishes absent
    /// from an explicit `null` so a `position`-only reorder never clobbers it.
    #[serde(default, deserialize_with = "deserialize_some")]
    #[schema(value_type = Option<usize>)]
    target_gpu: Option<Option<usize>>,
    /// New 0-based index for the job among the queued jobs. Clamped into range
    /// (a large value sends it to the back). Absent means no reorder.
    #[serde(default)]
    position: Option<usize>,
}

#[utoipa::path(
    patch,
    path = "/api/queue/{id}",
    tag = "queue",
    request_body = QueuePatchRequest,
    responses(
        (status = 200, description = "Updated queue entry", body = crate::job_registry::JobEntry),
        (status = 404, description = "Queue job not found"),
        (status = 409, description = "Queue job is already running"),
        (status = 422, description = "Invalid GPU target"),
    )
)]
async fn patch_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<QueuePatchRequest>,
) -> Result<Json<crate::job_registry::JobEntry>, ApiError> {
    if let Some(Some(target)) = req.target_gpu {
        let available = state
            .gpu_pool
            .workers
            .iter()
            .any(|w| w.gpu.ordinal == target);
        if !available {
            return Err(ApiError::validation(format!(
                "gpu:{target} is not available in this server's worker pool"
            )));
        }
    }

    // Both edits are independent and additive — apply whichever the request
    // supplied. `target_gpu` only when the field was present (absent leaves the
    // lane untouched); `position` only when a reorder was requested.
    if let Some(target_gpu) = req.target_gpu {
        state
            .job_registry
            .set_target_gpu(&id, target_gpu)
            .map_err(|e| match e {
                crate::job_registry::TargetGpuUpdateError::NotFound => {
                    ApiError::queue_job_not_found(format!("queue job {id} not found"))
                }
                crate::job_registry::TargetGpuUpdateError::AlreadyRunning => {
                    ApiError::queue_job_running(format!(
                        "queue job {id} is already running; lane changes only apply to queued jobs"
                    ))
                }
            })?;
    }

    if let Some(position) = req.position {
        state
            .job_registry
            .reorder_queued(&id, position)
            .map_err(|e| match e {
                crate::job_registry::QueueReorderError::NotFound => {
                    ApiError::queue_job_not_found(format!("queue job {id} not found"))
                }
                crate::job_registry::QueueReorderError::AlreadyRunning => {
                    ApiError::queue_job_running(format!(
                        "queue job {id} is already running; only queued jobs can be reordered"
                    ))
                }
            })?;
    }

    let entry = state
        .job_registry
        .entry(&id)
        .ok_or_else(|| ApiError::queue_job_not_found(format!("queue job {id} not found")))?;
    Ok(Json(entry))
}

/// Cancel a still-queued generation job. Only queued jobs are cancelable —
/// once a GPU worker owns the job there is no safe preemption point, so
/// running jobs return 409. The waiting client observes the cancellation as
/// a 499 `CANCELLED` error (blocking `POST /api/generate`) or a terminal
/// SSE `error` event (`POST /api/generate/stream`).
#[utoipa::path(
    delete,
    path = "/api/queue/{id}",
    tag = "queue",
    params(("id" = String, Path, description = "Queue job id")),
    responses(
        (status = 204, description = "Queued job cancelled"),
        (status = 404, description = "Queue job not found"),
        (status = 409, description = "Queue job is already running"),
    )
)]
async fn cancel_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.job_registry.cancel_queued(&id).map_err(|e| match e {
        crate::job_registry::QueuedJobCancelError::NotFound => {
            ApiError::queue_job_not_found(format!("queue job {id} not found"))
        }
        crate::job_registry::QueuedJobCancelError::AlreadyRunning => ApiError::queue_job_running(
            format!("queue job {id} is already running; only queued jobs can be cancelled"),
        ),
    })?;
    Ok(StatusCode::NO_CONTENT)
}

/// Response of `POST /api/queue/pause` and `POST /api/queue/resume` — the
/// resulting pause state (`true` after pause, `false` after resume).
#[derive(Debug, Serialize, utoipa::ToSchema)]
struct QueuePauseResponse {
    paused: bool,
}

/// Response of `DELETE /api/queue` — how many queued jobs were cancelled.
#[derive(Debug, Serialize, utoipa::ToSchema)]
struct QueueCancelAllResponse {
    cancelled: usize,
}

/// Pause dispatch of new generation jobs. The job currently running on a
/// worker finishes; only the *next* job is held. Idempotent — repeat pauses
/// return `{"paused": true}` without re-emitting the `queue_paused` event.
#[utoipa::path(
    post,
    path = "/api/queue/pause",
    tag = "queue",
    responses(
        (status = 200, description = "Queue dispatch paused", body = QueuePauseResponse),
    )
)]
async fn pause_queue(State(state): State<AppState>) -> Json<QueuePauseResponse> {
    if state.queue_pause.pause() {
        state.events.publish(mold_core::ServerEvent::QueuePaused);
    }
    Json(QueuePauseResponse { paused: true })
}

/// Resume dispatch of new generation jobs. Idempotent — repeat resumes return
/// `{"paused": false}` without re-emitting the `queue_resumed` event.
#[utoipa::path(
    post,
    path = "/api/queue/resume",
    tag = "queue",
    responses(
        (status = 200, description = "Queue dispatch resumed", body = QueuePauseResponse),
    )
)]
async fn resume_queue(State(state): State<AppState>) -> Json<QueuePauseResponse> {
    if state.queue_pause.resume() {
        state.events.publish(mold_core::ServerEvent::QueueResumed);
    }
    Json(QueuePauseResponse { paused: false })
}

/// Cancel every still-queued generation job. Running jobs are left untouched
/// (same rule as `DELETE /api/queue/:id`). Returns the number of jobs
/// cancelled; each waiting client observes the same cancellation signal as a
/// single-job cancel.
#[utoipa::path(
    delete,
    path = "/api/queue",
    tag = "queue",
    responses(
        (status = 200, description = "Queued jobs cancelled", body = QueueCancelAllResponse),
    )
)]
async fn cancel_all_queue(State(state): State<AppState>) -> Json<QueueCancelAllResponse> {
    let cancelled = state.job_registry.cancel_all_queued();
    Json(QueueCancelAllResponse { cancelled })
}

// ── /api/history ─────────────────────────────────────────────────────────────

/// Default number of history rows returned when `limit` is omitted.
const HISTORY_DEFAULT_LIMIT: usize = 50;
/// Hard cap on `limit` — matches the legacy 500-entry history bound.
const HISTORY_MAX_LIMIT: usize = 500;

/// 503 error code when the metadata DB is disabled (`MOLD_DB_DISABLE=1`).
const HISTORY_UNAVAILABLE: &str = "HISTORY_UNAVAILABLE";

fn history_db(state: &AppState) -> Result<&mold_db::MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "prompt history is unavailable because the metadata DB is disabled",
            HISTORY_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

#[derive(Debug, Deserialize)]
struct HistoryListQuery {
    /// Case-insensitive substring filter over the prompt text.
    query: Option<String>,
    /// Max rows to return (default 50, capped at 500).
    limit: Option<usize>,
}

#[utoipa::path(
    get,
    path = "/api/history",
    tag = "server",
    params(
        ("query" = Option<String>, Query, description = "Substring filter over prompt text (case-insensitive)"),
        ("limit" = Option<usize>, Query, description = "Max rows to return (default 50, max 500)"),
    ),
    responses(
        (status = 200, description = "Prompt history, newest first", body = mold_core::HistoryListing),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
async fn list_history(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HistoryListQuery>,
) -> Result<Json<mold_core::HistoryListing>, ApiError> {
    let db = history_db(&state)?;
    let limit = params
        .limit
        .unwrap_or(HISTORY_DEFAULT_LIMIT)
        .min(HISTORY_MAX_LIMIT);
    let history = mold_db::PromptHistory::new(db);
    let rows = match params
        .query
        .as_deref()
        .map(str::trim)
        .filter(|q| !q.is_empty())
    {
        Some(query) => history.search(query, limit),
        None => history.recent(limit),
    }
    .map_err(|e| ApiError::internal(format!("failed to read prompt history: {e:#}")))?;
    let entries = rows
        .into_iter()
        .map(|e| mold_core::HistoryEntry {
            prompt: e.prompt,
            model: e.model,
            used_at: e.created_at_ms,
        })
        .collect();
    Ok(Json(mold_core::HistoryListing { entries }))
}

#[derive(Debug, Deserialize)]
struct HistoryDeleteQuery {
    /// When present, trim to the most recent N entries instead of clearing.
    keep: Option<usize>,
}

#[utoipa::path(
    delete,
    path = "/api/history",
    tag = "server",
    params(
        ("keep" = Option<usize>, Query, description = "Keep only the most recent N entries instead of clearing everything"),
    ),
    responses(
        (status = 204, description = "Prompt history cleared (or trimmed)"),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
async fn delete_history(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HistoryDeleteQuery>,
) -> Result<StatusCode, ApiError> {
    let db = history_db(&state)?;
    let history = mold_db::PromptHistory::new(db);
    match params.keep {
        Some(keep) => history.trim_to(keep),
        None => history.clear(),
    }
    .map_err(|e| ApiError::internal(format!("failed to clear prompt history: {e:#}")))?;
    Ok(StatusCode::NO_CONTENT)
}

// ── /api/discovery/peers + /api/capabilities ─────────────────────────────────

/// Return the current DNS-SD cache. The serving instance is omitted by stable
/// UUID so the browser is only offered other machines it can connect to.
#[utoipa::path(
    get,
    path = "/api/discovery/peers",
    tag = "server",
    responses(
        (status = 200, description = "Mold servers visible on the serving host's LAN", body = Vec<mold_core::DiscoveryPeer>),
        (status = 503, description = "DNS-SD browsing is unavailable or disabled"),
    )
)]
async fn discovery_peers(
    State(state): State<AppState>,
) -> Result<Json<Vec<mold_core::DiscoveryPeer>>, ApiError> {
    if !state.discovery.can_browse() {
        return Err(ApiError::with_code(
            "LAN discovery is unavailable on this server",
            "DISCOVERY_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    let own_instance_id = state.instance_id.as_str();
    let peers = state
        .discovery
        .peers
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .filter(|peer| {
            !peer.is_this_machine && peer.instance_id.as_deref() != Some(own_instance_id)
        })
        .cloned()
        .collect();
    Ok(Json(peers))
}

/// Report the feature toggles a client needs to render correctly (hide the
/// delete button when delete isn't allowed, etc.). Authentication follows the
/// rest of `/api/*` when an API key is configured.
async fn server_capabilities(State(state): State<AppState>) -> Json<mold_core::ServerCapabilities> {
    let catalog_available = std::env::var("MOLD_CATALOG_DISABLE")
        .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let config = state.config.read().await;
    let expand_settings = config.expand.clone().with_env_overrides();
    let expand_model_present =
        expand_settings.is_local() && config.manifest_model_is_downloaded(&expand_settings.model);
    let expand = expand_capabilities(&expand_settings, expand_model_present);

    Json(mold_core::ServerCapabilities {
        gallery: mold_core::GalleryCapabilities { can_delete: true },
        catalog: mold_core::CatalogCapabilities {
            available: catalog_available,
            families: mold_catalog::families::ALL_FAMILIES
                .iter()
                .map(|f| f.as_str().to_string())
                .collect::<Vec<_>>(),
            sort: mold_catalog::live::CatalogSort::WIRE_VALUES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        },
        discovery: mold_core::DiscoveryCapabilities {
            can_browse: state.discovery.can_browse(),
        },
        events: mold_core::EventsCapabilities { available: true },
        queue: mold_core::QueueCapabilities {
            can_pause: true,
            can_cancel_all: true,
            can_reorder: true,
        },
        expand: Some(expand),
    })
}

/// Derive the wire capability without constructing an expander or probing an
/// external API. `model_present` is supplied by the caller so the feature and
/// backend permutations remain pure and directly testable.
fn expand_capabilities(
    settings: &mold_core::expand::ExpandSettings,
    model_present: bool,
) -> mold_core::ExpandCapabilities {
    if settings.is_local() {
        mold_core::ExpandCapabilities {
            configured: cfg!(feature = "expand"),
            model_present: Some(model_present),
            backend: mold_core::ExpandBackend::Local,
        }
    } else {
        mold_core::ExpandCapabilities {
            configured: !settings.backend.trim().is_empty(),
            model_present: None,
            backend: mold_core::ExpandBackend::Api,
        }
    }
}

// ── /api/capabilities/chain-limits ───────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/capabilities/chain-limits",
    tag = "server",
    params(
        ("model" = String, Query, description = "Model name (e.g. ltx-2-19b-distilled:fp8)")
    ),
    responses(
        (status = 200, description = "Chain limits for the requested model",
         body = crate::chain_limits::ChainLimits),
        (status = 400, description = "Missing required 'model' query parameter"),
        (status = 404, description = "Unknown or unsupported model"),
    )
)]
async fn capabilities_chain_limits(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> axum::response::Response {
    let raw_model = match params.get("model") {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "missing required 'model' query parameter\n",
            )
                .into_response();
        }
    };

    let resolved = mold_core::manifest::resolve_model_name(&raw_model);
    let (family, quant, supports_audio) =
        if let Some(manifest) = mold_core::manifest::find_manifest(&resolved) {
            let quant = resolved
                .split_once(':')
                .map(|(_, tag)| tag.to_string())
                .unwrap_or_default();
            let family = manifest.family.clone();
            let supports_audio = crate::chain_limits::family_supports_audio(&family);
            (family, quant, supports_audio)
        } else {
            // Installed live-catalog models retain opaque `cv:` / `hf:` ids and
            // therefore cannot be found in the built-in manifest. Resolve them
            // through the same installed-sidecar inventory as `/api/models`;
            // config lookup alone deliberately does not synthesize catalog
            // metadata and would incorrectly return 404 for runnable installs.
            let Some(entry) = model_manager::list_models(&state)
                .await
                .into_iter()
                .find(|entry| entry.downloaded && entry.info.name == raw_model)
            else {
                return (StatusCode::NOT_FOUND, "unknown model\n").into_response();
            };
            let family = entry.info.family;
            let supports_audio = entry
                .supports_audio
                .unwrap_or_else(|| crate::chain_limits::family_supports_audio(&family));
            (family, String::new(), supports_audio)
        };

    if crate::chain_limits::family_cap(&family).is_none() {
        return (StatusCode::NOT_FOUND, "model is not chain-capable\n").into_response();
    }

    // TODO(sub-project D): pass live free VRAM from AppState.
    let mut limits = crate::chain_limits::compute_limits(&raw_model, &family, &quant, 0);
    limits.supports_audio = supports_audio;
    Json(limits).into_response()
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

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct GalleryMediaTokenRequest {
    path: String,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct GalleryMediaTokenResponse {
    token: Option<String>,
    expires_at: Option<u64>,
    auth_required: bool,
}

/// Issue a short-lived credential for a browser media element.
///
/// The endpoint itself always uses normal `X-Api-Key` authentication. The
/// auth middleware records only an authenticated marker in request extensions;
/// this handler signs with an independent random per-process secret. The
/// resulting ticket can authenticate GET, HEAD, and Range requests to
/// `/api/gallery/image/:filename` until `expires_at` without exposing the
/// long-lived API key in the URL or making the URL an offline API-key verifier.
#[utoipa::path(
    post,
    path = "/api/gallery/media-token",
    tag = "server",
    request_body = GalleryMediaTokenRequest,
    responses(
        (status = 200, description = "Short-lived gallery media ticket", body = GalleryMediaTokenResponse),
        (status = 401, description = "API key authentication is required"),
        (status = 422, description = "Requested path is not a gallery media path"),
    )
)]
async fn create_gallery_media_token(
    auth_state: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Json(request): Json<GalleryMediaTokenRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if !crate::auth::is_gallery_image_path(&request.path) {
        return Err(ApiError::validation(
            "media token path must match /api/gallery/image/:filename",
        ));
    }

    let key_set = auth_state.and_then(|Extension(state)| state);
    let (token, expires_at, auth_required) = match key_set {
        Some(key_set) => {
            let _authenticated = authenticated.ok_or_else(|| {
                ApiError::with_code(
                    "API key authentication is required to issue a media token",
                    "UNAUTHORIZED",
                    StatusCode::UNAUTHORIZED,
                )
            })?;
            let (token, expires_at) = key_set.issue_gallery_media_token(&request.path);
            (Some(token), Some(expires_at), true)
        }
        // Headerless servers need no ticket. Returning an explicit response
        // lets clients with a stale saved key use the ordinary direct URL.
        None => (None, None, false),
    };

    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok((
        headers,
        Json(GalleryMediaTokenResponse {
            token,
            expires_at,
            auth_required,
        }),
    ))
}

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
        .header(header::CACHE_CONTROL, "private, no-store")
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
        // Gallery media may be authorized by a short-lived bearer ticket, so
        // neither browsers nor intermediaries should retain partial responses.
        .header(header::CACHE_CONTROL, "private, no-store")
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
/// Destructive, but always enabled — pair with the `MOLD_API_KEY` middleware
/// when the server is exposed beyond localhost.
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

    // File removal + DB delete are blocking syscalls / a synchronous SQLite
    // call — run the whole batch on the blocking pool so a slow disk can't
    // stall an async worker thread mid-generation.
    let db = state.metadata_db.clone();
    let name = clean_name.clone();
    let dir = output_dir.clone();
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        let path = dir.join(&name);
        if path.is_file() {
            std::fs::remove_file(&path)
                .map_err(|e| ApiError::internal(format!("failed to delete image: {e}")))?;
        }

        // Also remove server-side thumbnail (both legacy no-suffix and current
        // `.png`-suffixed cache layouts) and the animated preview sidecar so
        // `/api/gallery/preview/:filename` doesn't keep serving the GIF after
        // the source MP4 is gone.
        let thumb_dir = server_thumbnail_dir();
        let _ = std::fs::remove_file(thumb_dir.join(&name));
        let _ = std::fs::remove_file(thumb_dir.join(format!("{name}.png")));
        let _ = std::fs::remove_file(
            server_preview_gif_dir().join(mold_core::media_paths::preview_gif_filename(&name)),
        );

        // Drop the matching metadata row if the DB is enabled. Errors here are
        // logged — they don't roll back the disk delete since the file is the
        // source of truth and reconciliation will re-sync on the next restart.
        if let Some(db) = db.as_ref().as_ref() {
            match db.delete(&dir, &name) {
                Ok(true) => {}
                Ok(false) => {
                    tracing::debug!("delete: no metadata row for {}", dir.join(&name).display())
                }
                Err(e) => tracing::warn!(
                    "metadata DB delete failed for {}: {e:#}",
                    dir.join(&name).display()
                ),
            }
        }
        Ok(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery delete task failed: {e}")))??;

    state
        .events
        .publish(mold_core::ServerEvent::GalleryRemoved {
            filename: clean_name,
        });

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

/// Serve a cached animated GIF preview for a gallery video output.
///
/// Looks up `<preview_dir>/<filename>.preview.gif` (default:
/// `~/.mold/cache/previews/`). When present, streams the file back as
/// `image/gif`; otherwise returns 404. This exists so the TUI's remote
/// gallery detail pane can animate video entries the same way it animates
/// local ones — previously it fell through to fetching the raw MP4 over
/// `/api/gallery/image/:filename`, which `image::open` couldn't decode,
/// leaving the panel on `Loading…` forever.
async fn get_gallery_preview(
    State(state): State<AppState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<axum::response::Response, ApiError> {
    let config = state.config.read().await;
    if config.is_output_disabled() {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    // Sanitize: prevent directory traversal — the filename must be a bare
    // basename with no separators.
    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    // The preview cache lifecycle is tied to the underlying gallery file:
    // if the MP4 has been deleted (via `DELETE /api/gallery/image/:filename`
    // or an out-of-band `rm`), the sidecar may still be on disk but is
    // orphaned and must not be served.
    // Check the source file first and 404 before touching the cache so a
    // stale `.preview.gif` never leaks deleted content.
    let source_path = output_dir.join(&clean_name);
    if !tokio::fs::metadata(&source_path)
        .await
        .map(|m| m.is_file())
        .unwrap_or(false)
    {
        return Err(ApiError::not_found(format!(
            "image not found: {clean_name}"
        )));
    }

    let preview_path =
        server_preview_gif_dir().join(mold_core::media_paths::preview_gif_filename(&clean_name));
    let meta = match tokio::fs::metadata(&preview_path).await {
        Ok(m) if m.is_file() => m,
        _ => {
            return Err(ApiError::not_found(format!(
                "preview not found: {clean_name}"
            )));
        }
    };
    let total_len = meta.len();

    let file = tokio::fs::File::open(&preview_path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to open preview: {e}")))?;
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/gif")
        .header(header::CONTENT_LENGTH, total_len)
        .header(header::CACHE_CONTROL, "public, max-age=3600")
        .body(body)
        .unwrap())
}

/// Server-side GIF preview cache directory. Mirrors the layout the TUI
/// writes to (`crates/mold-tui/src/thumbnails.rs::preview_dir`) so a
/// single preview.gif authored on either side is reachable via this
/// endpoint.
fn server_preview_gif_dir() -> std::path::PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("previews")
}

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
    let mut images: Vec<mold_core::GalleryImage> = mold_db::scan::scan_output_dir(dir)
        .filter_map(|item| match item {
            mold_db::scan::ScanItem::Valid(file) => Some(file),
            _ => None,
        })
        .map(|file| {
            let timestamp = file.timestamp_secs();
            let size_bytes = file.size_u64();
            let (metadata, synthetic) = mold_db::metadata_io::read_or_synthesize(
                &file.path,
                file.format,
                &file.filename,
                timestamp,
            );
            mold_core::GalleryImage {
                filename: file.filename,
                metadata,
                timestamp,
                format: Some(file.format),
                size_bytes: Some(size_bytes),
                metadata_synthetic: synthetic,
            }
        })
        .collect();

    images.sort_by_key(|img| std::cmp::Reverse(img.timestamp));
    images
}

// ── /api/config/model/:name/placement (Agent C, model-ui-overhaul §3) ────────

/// Read the saved per-model placement default so an editor can hydrate its
/// controls before letting the user edit-and-save (without which a save
/// silently clobbers the persisted placement with defaults). Returns the raw
/// persisted value — not the env-overlaid `resolved_placement` — so a `404`
/// faithfully means "nothing saved for this model".
async fn get_model_placement(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<mold_core::types::DevicePlacement>, ApiError> {
    let cfg = state.config.read().await;
    match cfg.models.get(&name).and_then(|mc| mc.placement.clone()) {
        Some(placement) => Ok(Json(placement)),
        None => Err(ApiError::not_found(format!(
            "no placement saved for model '{name}'"
        ))),
    }
}

async fn put_model_placement(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(placement): Json<mold_core::types::DevicePlacement>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_multi_gpu_placement(&state, Some(&placement))?;
    {
        let mut cfg = state.config.write().await;
        cfg.set_model_placement(&name, Some(placement.clone()));
        cfg.save().map_err(|e| {
            tracing::warn!("failed to persist placement to config.toml: {e}");
            ApiError::internal(format!("failed to persist placement to config.toml: {e}"))
        })?;
    }
    Ok(Json(serde_json::json!({
        "ok": true,
        "model": name,
    })))
}

async fn delete_model_placement(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut cfg = state.config.write().await;
    cfg.set_model_placement(&name, None);
    cfg.save().map_err(|e| {
        tracing::warn!("failed to persist placement removal to config.toml: {e}");
        ApiError::internal(format!(
            "failed to persist placement removal to config.toml: {e}"
        ))
    })?;
    Ok(Json(serde_json::json!({ "ok": true })))
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
    query_cuda_gpu_info().or_else(query_metal_gpu_info)
}

fn query_cuda_gpu_info() -> Option<GpuInfo> {
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
        backend: Some(GpuBackend::Cuda),
    })
}

/// Metal fallback (macOS only elsewhere returns an empty snapshot): unified
/// memory means "VRAM" is the addressable system RAM — same convention as
/// `/api/resources`. Gives Macs a non-null `gpu_info` so clients can rank
/// hosts by backend/VRAM without a resources stream.
fn query_metal_gpu_info() -> Option<GpuInfo> {
    let gpu = crate::resources::metal_snapshot().into_iter().next()?;
    Some(GpuInfo {
        name: gpu.name,
        // Resource snapshots are bytes; GpuInfo is MB (1 MB = 1_000_000,
        // matching `parse_nvidia_smi_line`).
        vram_total_mb: gpu.vram_total / 1_000_000,
        vram_used_mb: gpu.vram_used / 1_000_000,
        backend: Some(GpuBackend::Metal),
    })
}

// ─── Downloads UI (Agent A) ──────────────────────────────────────────────────

#[derive(serde::Deserialize, utoipa::ToSchema)]
pub struct CreateDownloadBody {
    pub model: String,
}

#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct CreateDownloadResponse {
    pub id: String,
    pub position: usize,
}

#[utoipa::path(
    post,
    path = "/api/downloads",
    tag = "downloads",
    request_body = CreateDownloadBody,
    responses(
        (status = 200, description = "Enqueued; position 0 = will start immediately", body = CreateDownloadResponse),
        (status = 400, description = "Unknown model"),
        (status = 409, description = "Already active or queued; body contains existing id", body = CreateDownloadResponse),
    )
)]
pub async fn create_download(
    State(state): State<AppState>,
    Json(body): Json<CreateDownloadBody>,
) -> axum::response::Response {
    use crate::downloads::{EnqueueError, EnqueueOutcome};
    match state.downloads.enqueue(body.model.clone()).await {
        Ok((id, position, EnqueueOutcome::Created)) => (
            StatusCode::OK,
            Json(CreateDownloadResponse { id, position }),
        )
            .into_response(),
        Ok((id, position, EnqueueOutcome::AlreadyPresent)) => (
            StatusCode::CONFLICT,
            Json(CreateDownloadResponse { id, position }),
        )
            .into_response(),
        Err(EnqueueError::UnknownModel(_)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("unknown model '{}'. Run 'mold list' to see available models.", body.model)
            })),
        )
            .into_response(),
        Err(EnqueueError::LockPoisoned) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "download queue state is corrupt" })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    delete,
    path = "/api/downloads/{id}",
    tag = "downloads",
    params(("id" = String, Path, description = "Job id")),
    responses(
        (status = 204, description = "Cancelled"),
        (status = 404, description = "Unknown id"),
    )
)]
pub async fn delete_download(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if state.downloads.cancel(&id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown download id '{id}'") })),
        )
            .into_response()
    }
}

#[utoipa::path(
    get,
    path = "/api/downloads",
    tag = "downloads",
    responses((status = 200, description = "Current queue state"))
)]
pub async fn list_downloads(State(state): State<AppState>) -> axum::response::Response {
    Json(state.downloads.listing().await).into_response()
}

#[utoipa::path(
    get,
    path = "/api/downloads/stream",
    tag = "downloads",
    responses((status = 200, description = "SSE stream of DownloadEvent JSON")),
)]
pub async fn stream_downloads(
    State(state): State<AppState>,
) -> Sse<
    impl futures_core::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse::Event;
    use tokio_stream::wrappers::BroadcastStream;
    use tokio_stream::StreamExt as _;

    // Subscribe BEFORE snapshotting so any event arriving during the
    // snapshot read is queued in the broadcast channel instead of being
    // missed. The first frame we yield is `Snapshot { listing }` —
    // mirrors `/api/resources/stream`'s initial-snapshot pattern so a
    // freshly-mounted SPA paints current state without waiting for the
    // next delta.
    let rx = state.downloads.subscribe();
    let initial = state.downloads.listing().await;
    let snapshot_event = mold_core::types::DownloadEvent::Snapshot { listing: initial };

    let stream = async_stream::stream! {
        let data = serde_json::to_string(&snapshot_event).unwrap_or_else(|_| "{}".to_string());
        yield Ok::<_, std::convert::Infallible>(Event::default().event("download").data(data));

        let mut bs = BroadcastStream::new(rx);
        while let Some(item) = bs.next().await {
            match item {
                Ok(event) => {
                    let data = serde_json::to_string(&event)
                        .unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("download").data(data));
                }
                // Slow subscribers see lag silently; the snapshot above
                // already carries the full state so we don't need to
                // resync on every drop.
                Err(_lagged) => continue,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

// ── Resource telemetry (Agent B scope) ───────────────────────────────────────

/// `GET /api/resources` — one-shot JSON snapshot from the aggregator cache.
/// Returns 503 if the aggregator has not yet fired (first 1 s after startup
/// and before `spawn_aggregator` has run).
async fn get_resources(State(state): State<AppState>) -> Result<Json<ResourceSnapshot>, ApiError> {
    match state.resources.latest() {
        Some(snap) => Ok(Json(snap)),
        None => Err(ApiError::internal_with_status(
            "resource telemetry not ready",
            StatusCode::SERVICE_UNAVAILABLE,
        )),
    }
}

/// `GET /api/resources/stream` — SSE stream of `ResourceSnapshot` frames.
/// Event name: `snapshot`. Matches the keepalive cadence of `/api/generate/stream`.
async fn get_resources_stream(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>> {
    use tokio_stream::wrappers::BroadcastStream;

    let rx = state.resources.subscribe();
    // Attach the cached `latest` snapshot as the first frame so clients
    // don't wait up to one full tick for their initial value.
    let initial = state.resources.latest();

    let stream = async_stream::stream! {
        if let Some(snap) = initial {
            yield Ok::<_, Infallible>(snapshot_to_sse(&snap));
        }
        let mut bs = BroadcastStream::new(rx);
        while let Some(item) = bs.next().await {
            match item {
                Ok(snap) => yield Ok(snapshot_to_sse(&snap)),
                // Lag is normal for slow clients — skip dropped frames
                // silently; the next one will catch them up.
                Err(_lagged) => continue,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

fn snapshot_to_sse(snap: &ResourceSnapshot) -> SseEvent {
    match serde_json::to_string(snap) {
        Ok(data) => SseEvent::default().event("snapshot").data(data),
        Err(e) => SseEvent::default()
            .event("error")
            .data(format!("{{\"message\":\"serialize failed: {e}\"}}")),
    }
}

/// `GET /api/events` — SSE stream of server-wide [`mold_core::ServerEvent`]s:
/// job lifecycle (queued/started/ended, mirrored off the job registry) and
/// gallery mutations (added/removed). One connection observes the whole
/// server, so clients don't need a held stream per job to know when the
/// gallery changed. Deltas only — bootstrap current state from
/// `GET /api/queue` + `GET /api/gallery` after subscribing. Event name:
/// `event`. Feature-detect via `capabilities.events.available`.
#[utoipa::path(
    get,
    path = "/api/events",
    tag = "server",
    responses(
        (status = 200, description = "SSE stream of server lifecycle events", content_type = "text/event-stream")
    )
)]
async fn stream_events(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>> {
    use tokio_stream::wrappers::BroadcastStream;

    let rx = state.events.subscribe();
    let stream = async_stream::stream! {
        let mut bs = BroadcastStream::new(rx);
        while let Some(item) = bs.next().await {
            match item {
                Ok(ev) => yield Ok::<_, Infallible>(server_event_to_sse(&ev)),
                // Lagged receivers skip the gap; REST endpoints are the
                // recovery path for anything missed.
                Err(_lagged) => continue,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

fn server_event_to_sse(ev: &mold_core::ServerEvent) -> SseEvent {
    match serde_json::to_string(ev) {
        Ok(data) => SseEvent::default().event("event").data(data),
        // json! escapes the error text — quotes/newlines in `e` must not
        // produce an invalid JSON frame.
        Err(e) => SseEvent::default()
            .event("error")
            .data(serde_json::json!({ "message": format!("serialize failed: {e}") }).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_config_for_request_threads_style() {
        let settings = mold_core::expand::ExpandSettings::default();
        let req = mold_core::ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "flux".to_string(),
            variations: 2,
            style: Some("oil painting".to_string()),
        };
        let config = expand_config_for_request(&settings, &req);
        assert_eq!(config.style.as_deref(), Some("oil painting"));
        assert_eq!(config.model_family, "flux");
        assert_eq!(config.variations, 2);

        let bare = mold_core::ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "flux".to_string(),
            variations: 1,
            style: None,
        };
        assert!(expand_config_for_request(&settings, &bare).style.is_none());
    }

    #[test]
    fn local_expand_capability_reports_feature_and_model_facts_separately() {
        let settings = mold_core::expand::ExpandSettings::default();
        let missing = expand_capabilities(&settings, false);
        let present = expand_capabilities(&settings, true);

        assert_eq!(missing.backend, mold_core::ExpandBackend::Local);
        assert_eq!(missing.configured, cfg!(feature = "expand"));
        assert_eq!(missing.model_present, Some(false));
        assert_eq!(present.configured, cfg!(feature = "expand"));
        assert_eq!(present.model_present, Some(true));
    }

    #[test]
    fn api_expand_capability_does_not_claim_external_reachability() {
        let settings = mold_core::expand::ExpandSettings {
            backend: "http://localhost:11434".into(),
            ..Default::default()
        };
        let capability = expand_capabilities(&settings, false);

        assert_eq!(capability.backend, mold_core::ExpandBackend::Api);
        assert!(capability.configured);
        assert_eq!(capability.model_present, None);

        let unconfigured = mold_core::expand::ExpandSettings {
            backend: "   ".into(),
            ..Default::default()
        };
        let capability = expand_capabilities(&unconfigured, true);
        assert_eq!(capability.backend, mold_core::ExpandBackend::Api);
        assert!(!capability.configured);
        assert_eq!(capability.model_present, None);
    }

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        &ENV_LOCK
    }

    #[test]
    fn disk_usage_for_path_picks_longest_matching_mount() {
        let disks = vec![
            (std::path::PathBuf::from("/"), 100, 10),
            (std::path::PathBuf::from("/data"), 500, 50),
        ];
        let usage = disk_usage_for_path(&disks, std::path::Path::new("/data/models")).unwrap();
        assert_eq!(usage.total_bytes, 500);
        assert_eq!(usage.free_bytes, 50);
        // A path outside /data falls back to the root mount.
        let usage = disk_usage_for_path(&disks, std::path::Path::new("/home/u/models")).unwrap();
        assert_eq!(usage.total_bytes, 100);
    }

    #[test]
    fn disk_usage_for_path_component_boundaries_and_no_match() {
        let disks = vec![(std::path::PathBuf::from("/data"), 500, 50)];
        // `/database` is not under the `/data` mount — prefix matching must be
        // per path component, not per byte.
        assert_eq!(
            disk_usage_for_path(&disks, std::path::Path::new("/database/models")),
            None
        );
        // No mount matches at all (e.g. relative path) → None.
        assert_eq!(
            disk_usage_for_path(&disks, std::path::Path::new("relative/models")),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn canonical_models_dir_resolves_symlinks_for_mount_matching() {
        // A symlinked models dir (`ln -s /Volumes/Big/models ~/.mold/models`)
        // must resolve to the target's filesystem: the longest-prefix mount
        // match has to run against the canonical target path, not the
        // symlink's own location.
        let tmp = tempfile::tempdir().unwrap();
        // Canonicalize the tempdir itself so the fake mount table matches on
        // platforms where the temp root is itself a symlink (/tmp, /var).
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let target = root.join("big-volume").join("models");
        std::fs::create_dir_all(&target).unwrap();
        let link = root.join("link-models");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let resolved = canonical_models_dir(&link);
        assert_eq!(resolved, target);

        // With a mount table containing the symlink target's volume, the
        // canonicalized path must pick that mount over the root fallback.
        let disks = vec![
            (std::path::PathBuf::from("/"), 100, 10),
            (root.join("big-volume"), 500, 50),
        ];
        let usage = disk_usage_for_path(&disks, &resolved).unwrap();
        assert_eq!(usage.total_bytes, 500);
        assert_eq!(usage.free_bytes, 50);
    }

    #[test]
    fn canonical_models_dir_falls_back_to_the_literal_path() {
        // A models dir that doesn't exist yet must not panic or change the
        // matching behavior — the raw path is used as-is.
        let missing = std::path::Path::new("/definitely/not/a/real/mold/models/dir");
        assert_eq!(canonical_models_dir(missing), missing.to_path_buf());
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
    fn clean_error_message_renders_full_anyhow_chain() {
        // Wrapped errors must surface the root cause; previously the outer
        // `with_context` swallowed everything below it (cv:2739091 truncated
        // checkpoint surfaced as "mmap single-file checkpoint at …" with no
        // hint that the safetensors data was short).
        let root = std::io::Error::new(std::io::ErrorKind::InvalidData, "bytes past end");
        let err: anyhow::Error = anyhow::Error::new(root)
            .context("validate single-file checkpoint at /tmp/foo.safetensors");
        let msg = clean_error_message(&err);
        assert!(
            msg.contains("validate single-file checkpoint") && msg.contains("bytes past end"),
            "expected both context layers in the rendered chain, got: {msg}",
        );
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
    // ── Gallery validation ───────────────────────────────────────────────
    // The guard-rail pure functions live in `mold_db::metadata_io` (with
    // their own unit tests); these end-to-end scans pin that the server
    // gallery keeps consuming them correctly.

    /// Create a scratch directory unique to this test and delete it on drop.
    /// Using `std::env::temp_dir()` rather than pulling in a `tempfile`
    /// dev-dep for two tests' worth of fixtures.
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
    /// compress to ~80 bytes and be filtered out by the size guard.
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
