//! Server-side chained video generation endpoints.
//!
//! Exposes `POST /api/generate/chain` (synchronous) and
//! `POST /api/generate/chain/stream` (SSE). Both drive
//! [`mold_inference::ltx2::Ltx2ChainOrchestrator`] through an engine's
//! [`mold_inference::ltx2::ChainStageRenderer`] view.
//!
//! Unlike the single-shot generate path (which queues through
//! [`crate::state::QueueHandle`] to keep small GPU jobs FIFO-fair), chains
//! are multi-minute compound jobs — the handler take/restores the engine
//! out of the model cache and runs the full sequence in a
//! [`tokio::task::spawn_blocking`] so the sync orchestrator never blocks
//! the async runtime. While the chain is running the engine is removed
//! from the cache, so concurrent generate/chain requests for the same
//! model cannot race.

use std::convert::Infallible;

use axum::{
    extract::State,
    response::sse::{Event as SseEvent, KeepAlive, Sse},
    Json,
};
use base64::Engine as _;
use mold_core::chain::{
    ChainProgressEvent, ChainRequest, ChainResponse, ChainScript, SseChainCompleteEvent,
};
use mold_core::{OutputFormat, OutputMetadata, VideoData};
use tokio_stream::StreamExt as _;

use crate::model_cache::CachedEngine;
use crate::model_manager;
use crate::queue::save_video_to_dir;
use crate::routes::ApiError;
use crate::state::AppState;

/// Internal wire event used by the chain SSE stream before per-event
/// serialization. Separate from [`crate::state::SseMessage`] because chain
/// complete events carry a different payload (`SseChainCompleteEvent`) and
/// progress events are chain-shaped (`ChainProgressEvent`) rather than the
/// single-stage `SseProgressEvent`.
pub(crate) enum ChainSseMessage {
    Progress(ChainProgressEvent),
    Complete(Box<SseChainCompleteEvent>),
    Error(String),
}

fn chain_sse_event(msg: ChainSseMessage) -> SseEvent {
    match msg {
        ChainSseMessage::Progress(ev) => match serde_json::to_string(&ev) {
            Ok(data) => SseEvent::default().event("progress").data(data),
            Err(e) => SseEvent::default()
                .event("error")
                .data(format!(r#"{{"message":"serialize progress: {e}"}}"#)),
        },
        ChainSseMessage::Complete(ev) => match serde_json::to_string(&ev) {
            Ok(data) => SseEvent::default().event("complete").data(data),
            Err(e) => SseEvent::default()
                .event("error")
                .data(format!(r#"{{"message":"serialize complete: {e}"}}"#)),
        },
        ChainSseMessage::Error(message) => SseEvent::default()
            .event("error")
            .data(serde_json::json!({ "message": message }).to_string()),
    }
}

/// Encode chain frames into bytes for the requested output format. Returns
/// the encoded payload plus a best-effort animated-GIF preview for the
/// gallery.
///
/// MP4 is gated behind the `mp4` feature flag; when the flag is disabled,
/// the handler falls back to APNG so the endpoint still produces a usable
/// animation on every build.
fn encode_chain_output(
    frames: &[image::RgbImage],
    fps: u32,
    format: OutputFormat,
) -> anyhow::Result<(Vec<u8>, OutputFormat, Vec<u8>)> {
    use mold_inference::ltx_video::video_enc;

    // Always produce a GIF preview for the gallery UI. Non-fatal.
    let gif_preview = match video_enc::encode_gif(frames, fps) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("chain gif preview encode failed: {e:#}");
            Vec::new()
        }
    };

    let (bytes, actual_format) = match format {
        OutputFormat::Mp4 => {
            #[cfg(feature = "mp4")]
            {
                (video_enc::encode_mp4(frames, fps)?, OutputFormat::Mp4)
            }
            #[cfg(not(feature = "mp4"))]
            {
                tracing::warn!(
                    "chain requested MP4 but server was built without the `mp4` feature — \
                     falling back to APNG"
                );
                (
                    video_enc::encode_apng(frames, fps, None)?,
                    OutputFormat::Apng,
                )
            }
        }
        OutputFormat::Apng => (
            video_enc::encode_apng(frames, fps, None)?,
            OutputFormat::Apng,
        ),
        OutputFormat::Gif => (video_enc::encode_gif(frames, fps)?, OutputFormat::Gif),
        // WebP is always available here because mold-inference's webp
        // feature would need to gate at the transitive-dep level; for the
        // chain route v1 we fall back to APNG when WebP is requested so
        // we don't bind the server crate to another optional dep.
        OutputFormat::Webp => {
            tracing::warn!(
                "chain WebP output is not supported on the server yet — falling back to APNG"
            );
            (
                video_enc::encode_apng(frames, fps, None)?,
                OutputFormat::Apng,
            )
        }
        other => anyhow::bail!("{other:?} is not a video output format for chain generation"),
    };

    Ok((bytes, actual_format, gif_preview))
}

/// Build the `OutputMetadata` for a stitched chain output. Pulls chain-
/// level parameters (dimensions, seed, steps) from `req` and the prompt /
/// negative prompt from `stages[0]`.
fn chain_output_metadata(req: &ChainRequest, frame_count: u32) -> OutputMetadata {
    let first_stage = req.stages.first();
    OutputMetadata {
        prompt: first_stage.map(|s| s.prompt.clone()).unwrap_or_default(),
        negative_prompt: first_stage.and_then(|s| s.negative_prompt.clone()),
        original_prompt: None,
        model: req.model.clone(),
        seed: req.seed.unwrap_or(0),
        steps: req.steps,
        guidance: req.guidance,
        width: req.width,
        height: req.height,
        strength: Some(req.strength),
        scheduler: None,
        lora: None,
        lora_scale: None,
        frames: Some(frame_count),
        fps: Some(req.fps),
        version: mold_core::build_info::version_string().to_string(),
    }
}

/// Trim a frame buffer to the caller's requested total frame count, per
/// the signed-off "trim from tail" decision (2026-04-20). The orchestrator
/// always over-produces to hit or exceed `total_frames`; trimming here
/// keeps the output length deterministic without altering per-stage
/// denoise behaviour.
fn trim_to_total_frames(frames: &mut Vec<image::RgbImage>, total_frames: Option<u32>) {
    if let Some(target) = total_frames {
        let target = target as usize;
        if frames.len() > target {
            frames.truncate(target);
        }
    }
}

/// Assemble per-stage frame clips into a single output buffer using
/// [`mold_inference::ltx2::stitch::StitchPlan`], honouring per-boundary
/// transition rules (Smooth / Cut / Fade).
pub(crate) fn stitch_chain_output(
    chain_output: mold_inference::ltx2::chain::ChainRunOutput,
    req: &mold_core::chain::ChainRequest,
) -> Result<Vec<image::RgbImage>, mold_inference::ltx2::stitch::StitchError> {
    use mold_inference::ltx2::stitch::StitchPlan;
    let boundaries: Vec<_> = req.stages.iter().skip(1).map(|s| s.transition).collect();
    let fade_lens: Vec<_> = req
        .stages
        .iter()
        .skip(1)
        .map(|s| s.fade_frames.unwrap_or(8))
        .collect();
    let plan = StitchPlan {
        clips: chain_output.stage_frames,
        boundaries,
        fade_lens,
        motion_tail_frames: req.motion_tail_frames,
    };
    plan.assemble()
}

/// Produce a PNG thumbnail for the chain output — best-effort, returns
/// an empty `Vec` on failure so the save/response paths still succeed.
fn chain_thumbnail(frames: &[image::RgbImage]) -> Vec<u8> {
    match mold_inference::ltx_video::video_enc::first_frame_png(frames) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("chain thumbnail encode failed: {e:#}");
            Vec::new()
        }
    }
}

/// Build a `VideoData` for the `ChainResponse` body.
fn build_video_data(
    bytes: Vec<u8>,
    format: OutputFormat,
    req: &ChainRequest,
    frame_count: u32,
    thumbnail: Vec<u8>,
    gif_preview: Vec<u8>,
) -> VideoData {
    let duration_ms = if req.fps == 0 {
        None
    } else {
        Some((frame_count as u64 * 1000) / req.fps as u64)
    };
    VideoData {
        data: bytes,
        format,
        width: req.width,
        height: req.height,
        frames: frame_count,
        fps: req.fps,
        thumbnail,
        gif_preview,
        has_audio: false,
        duration_ms,
        audio_sample_rate: None,
        audio_channels: None,
    }
}

/// Build the SSE `complete` payload for a finished chain run. Sibling of
/// [`crate::queue::build_sse_complete_event`] — kept in this module so the
/// chain-specific payload can evolve independently from the single-shot
/// one.
fn build_sse_chain_complete_event(
    resp: &ChainResponse,
    generation_time_ms: u64,
) -> SseChainCompleteEvent {
    let b64 = base64::engine::general_purpose::STANDARD;
    let video = &resp.video;
    SseChainCompleteEvent {
        video: b64.encode(&video.data),
        format: video.format,
        width: video.width,
        height: video.height,
        frames: video.frames,
        fps: video.fps,
        thumbnail: if video.thumbnail.is_empty() {
            None
        } else {
            Some(b64.encode(&video.thumbnail))
        },
        gif_preview: if video.gif_preview.is_empty() {
            None
        } else {
            Some(b64.encode(&video.gif_preview))
        },
        has_audio: video.has_audio,
        duration_ms: video.duration_ms,
        audio_sample_rate: video.audio_sample_rate,
        audio_channels: video.audio_channels,
        stage_count: resp.stage_count,
        gpu: resp.gpu,
        generation_time_ms: Some(generation_time_ms),
        script: resp.script.clone(),
        vram_estimate: resp.vram_estimate.clone(),
    }
}

/// Errors surfaced from the chain-run helper. Mapped to appropriate HTTP
/// status codes by the route handlers.
#[derive(Debug)]
enum ChainRunError {
    /// Model family doesn't support chain rendering (422).
    UnsupportedModel(String),
    /// Engine missing from cache after `ensure_model_ready` (500).
    CacheMiss(String),
    /// Orchestrator returned an error mid-chain from an invalid request (502).
    Inference(String),
    /// Orchestrator returned a typed stage failure mid-chain (502 with body).
    StageFailed(mold_core::chain::ChainFailure),
    /// Output encoding failure (500).
    Encode(String),
    /// `StitchPlan::assemble` failed (500).
    StitchFailed(String),
    /// Task panic or join error (500).
    Internal(String),
    /// No GPU worker available to service this chain (503).
    #[allow(dead_code)] // Constructed by run_chain_pooled in Task 5.
    NoWorker(String),
    /// `spawn_blocking` task failed to join (500).
    #[allow(dead_code)] // Constructed by run_chain_pooled in Task 5.
    Join(String),
}

impl From<ChainRunError> for ApiError {
    fn from(err: ChainRunError) -> Self {
        match err {
            ChainRunError::UnsupportedModel(msg) => ApiError::validation(msg),
            ChainRunError::CacheMiss(msg) => ApiError::internal(msg),
            ChainRunError::Inference(msg) => {
                ApiError::internal_with_status(msg, axum::http::StatusCode::BAD_GATEWAY)
            }
            // The SSE error channel is string-only (`ChainSseMessage::Error(String)`),
            // so the structured fields (`failed_stage_idx`, `elapsed_stages`,
            // `elapsed_ms`) are deliberately collapsed to `stage_error` here.
            // Clients that need the typed shape use the non-streaming
            // `/api/generate/chain` handler which returns a `ChainFailure`
            // body at status 502.
            ChainRunError::StageFailed(failure) => ApiError::internal_with_status(
                failure.stage_error,
                axum::http::StatusCode::BAD_GATEWAY,
            ),
            ChainRunError::Encode(msg) => ApiError::internal(msg),
            ChainRunError::StitchFailed(msg) => ApiError::internal(msg),
            ChainRunError::Internal(msg) => ApiError::internal(msg),
            ChainRunError::NoWorker(msg) => {
                ApiError::internal_with_status(msg, axum::http::StatusCode::SERVICE_UNAVAILABLE)
            }
            ChainRunError::Join(msg) => ApiError::internal(msg),
        }
    }
}

/// Dispatch a chain request to the pooled or legacy handler based on
/// whether the server discovered any GPU workers at startup.
///
/// In multi-worker mode (production CUDA / Metal), the pooled path
/// uses `gpu_worker::run_chain_blocking` to acquire the target GPU's
/// per-worker `model_load_lock` — preventing the SEGV race that arose
/// when the legacy path's `reclaim_gpu_memory(0)` collided with a
/// single-clip worker's reset on the same context.
///
/// No-worker mode (CPU-only dev boxes, CI) falls through to the legacy
/// path, which still uses `state.chain_lock` + `state.model_cache`.
async fn run_chain(
    state: &AppState,
    req: ChainRequest,
    progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    if state.gpu_pool.worker_count() > 0 {
        run_chain_pooled(state, req, progress_cb).await
    } else {
        run_chain_legacy(state, req, progress_cb).await
    }
}

/// Multi-worker chain path (stub — filled in by Task 5).
async fn run_chain_pooled(
    _state: &AppState,
    _req: ChainRequest,
    _progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    Err(ChainRunError::Internal(
        "run_chain_pooled not yet implemented (Task 5)".to_string(),
    ))
}

/// Drive the chain to completion. Shared between the non-streaming and SSE
/// paths — the only caller-provided variable is `progress_cb`, which is
/// `None` for the plain JSON endpoint and `Some` for the SSE endpoint.
async fn run_chain_legacy(
    state: &AppState,
    req: ChainRequest,
    progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    // Serialize concurrent chain requests. The chain handler deliberately
    // takes the engine out of `model_cache` for the full multi-minute run
    // (see below) — without this lock a second chain request arriving
    // mid-run calls `ensure_model_ready`, sees an empty cache, tries to
    // load a second copy of the model, and the subsequent `cache.take()`
    // reports "engine vanished from cache after ensure_model_ready".
    // Holding for the whole chain is intentional: single-clip requests
    // keep flowing through the normal generation queue; only chains wait
    // on each other.
    let _chain_guard = state.chain_lock.lock().await;

    // Ensure the model is loaded. Progress forwarding is not plumbed yet —
    // load-time events go through the model manager's own tracing. Chain
    // stage events (StageStart/DenoiseStep/StageDone/Stitching) come from
    // the orchestrator during the blocking task below.
    model_manager::ensure_model_ready(state, &req.model, None)
        .await
        .map_err(|e| ChainRunError::CacheMiss(e.error))?;

    // Take the engine out of the cache so the blocking orchestrator run
    // owns it for the full multi-minute chain without holding the async
    // mutex guard across an await. Restore when we're done (or on error).
    let mut cache = state.model_cache.lock().await;
    let cached: CachedEngine = cache.take(&req.model).ok_or_else(|| {
        ChainRunError::CacheMiss(format!(
            "engine '{}' vanished from cache after ensure_model_ready",
            req.model
        ))
    })?;
    drop(cache);

    let req_for_task = req.clone();
    let join_handle = tokio::task::spawn_blocking(move || {
        let mut cached = cached;
        let mut progress_cb = progress_cb;
        let outcome = {
            let engine = &mut cached.engine;
            match engine.as_chain_renderer() {
                Some(renderer) => {
                    let mut orch = mold_inference::ltx2::Ltx2ChainOrchestrator::new(renderer);
                    // The orchestrator expects `Option<&mut dyn FnMut(...)>`
                    // — synthesise that from the optional boxed callback we
                    // moved into this task.
                    let result = if let Some(cb) = progress_cb.as_deref_mut() {
                        orch.run(&req_for_task, Some(cb))
                    } else {
                        orch.run(&req_for_task, None)
                    };
                    result.map_err(|e| {
                        use mold_inference::ltx2::ChainOrchestratorError;
                        match e {
                            ChainOrchestratorError::StageFailed {
                                stage_idx,
                                elapsed_stages,
                                elapsed_ms,
                                inner,
                            } => ChainRunError::StageFailed(mold_core::chain::ChainFailure {
                                error: "stage render failed".into(),
                                failed_stage_idx: stage_idx,
                                elapsed_stages,
                                elapsed_ms,
                                stage_error: format!("{inner:#}"),
                            }),
                            ChainOrchestratorError::Invalid(inner) => {
                                ChainRunError::Inference(format!("{inner:#}"))
                            }
                        }
                    })
                }
                None => Err(ChainRunError::UnsupportedModel(format!(
                    "model '{}' does not support chained video generation",
                    req_for_task.model
                ))),
            }
        };
        (cached, outcome)
    });

    let (cached, outcome) = match join_handle.await {
        Ok(pair) => pair,
        Err(join_err) => {
            return Err(ChainRunError::Internal(format!(
                "chain orchestrator task failed: {join_err}"
            )));
        }
    };

    // Restore the engine to the cache regardless of success/failure so the
    // next request can reuse it.
    {
        let mut cache = state.model_cache.lock().await;
        cache.restore(cached);
    }

    let chain_output = outcome?;
    let stage_count = chain_output.stage_count;
    let generation_time_ms = chain_output.generation_time_ms;

    let mut frames = stitch_chain_output(chain_output, &req)
        .map_err(|e| ChainRunError::StitchFailed(e.to_string()))?;
    trim_to_total_frames(&mut frames, req.total_frames);

    if frames.is_empty() {
        return Err(ChainRunError::Encode(
            "chain run emitted zero frames after trim".to_string(),
        ));
    }

    let (bytes, output_format, gif_preview) =
        encode_chain_output(&frames, req.fps, req.output_format)
            .map_err(|e| ChainRunError::Encode(format!("encode chain output: {e:#}")))?;
    let thumbnail = chain_thumbnail(&frames);
    let frame_count = frames.len() as u32;

    // Save to the gallery directory (best-effort, non-blocking).
    let output_dir = {
        let config = state.config.read().await;
        if config.is_output_disabled() {
            None
        } else {
            Some(config.effective_output_dir())
        }
    };
    if let Some(dir) = output_dir {
        let metadata = chain_output_metadata(&req, frame_count);
        let bytes_clone = bytes.clone();
        let gif_clone = gif_preview.clone();
        let model = req.model.clone();
        let db = state.metadata_db.clone();
        tokio::task::spawn_blocking(move || {
            save_video_to_dir(
                &dir,
                &bytes_clone,
                &gif_clone,
                output_format,
                &model,
                &metadata,
                Some(generation_time_ms as i64),
                db.as_ref().as_ref(),
            );
        });
    }

    let video = build_video_data(
        bytes,
        output_format,
        &req,
        frame_count,
        thumbnail,
        gif_preview,
    );
    let response = ChainResponse {
        video,
        stage_count,
        gpu: None,
        script: ChainScript::from(&req),
        vram_estimate: None,
    };
    Ok((response, generation_time_ms))
}

/// `POST /api/generate/chain` — synchronous chained video generation.
#[utoipa::path(
    post,
    path = "/api/generate/chain",
    tag = "generation",
    request_body = mold_core::ChainRequest,
    responses(
        (status = 200, description = "Stitched chain video", body = mold_core::ChainResponse),
        (status = 422, description = "Invalid request or unsupported model"),
        (status = 500, description = "Chain render failed"),
        (status = 502, description = "Chain render failed mid-stage", body = mold_core::ChainFailure),
    )
)]
pub async fn generate_chain(
    State(state): State<AppState>,
    Json(req): Json<ChainRequest>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let req = match req.normalise() {
        Ok(r) => r,
        Err(e) => return ApiError::validation(e.to_string()).into_response(),
    };

    tracing::info!(
        model = %req.model,
        stages = req.stages.len(),
        width = req.width,
        height = req.height,
        fps = req.fps,
        "generate/chain request"
    );

    match run_chain(&state, req, None).await {
        Ok((response, _elapsed_ms)) => Json(response).into_response(),
        Err(ChainRunError::StageFailed(failure)) => {
            (StatusCode::BAD_GATEWAY, Json(failure)).into_response()
        }
        Err(other) => ApiError::from(other).into_response(),
    }
}

/// `POST /api/generate/chain/stream` — SSE-streamed chain generation. Emits
/// [`ChainProgressEvent`]s as `event: progress` frames while the chain
/// runs, and a single `event: complete` frame with a [`SseChainCompleteEvent`]
/// payload when the stitched output is ready. Mid-chain failure closes the
/// stream with an `event: error` frame carrying the orchestrator message.
#[utoipa::path(
    post,
    path = "/api/generate/chain/stream",
    tag = "generation",
    request_body = mold_core::ChainRequest,
    responses(
        (status = 200, description = "SSE event stream with chain progress and completion"),
        (status = 422, description = "Invalid request or unsupported model"),
        (status = 500, description = "Chain render failed"),
    )
)]
pub async fn generate_chain_stream(
    State(state): State<AppState>,
    Json(req): Json<ChainRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let req = req
        .normalise()
        .map_err(|e| ApiError::validation(e.to_string()))?;

    tracing::info!(
        model = %req.model,
        stages = req.stages.len(),
        width = req.width,
        height = req.height,
        fps = req.fps,
        "generate/chain/stream request"
    );

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainSseMessage>();
    let state_clone = state.clone();
    let tx_for_task = tx.clone();

    tokio::spawn(async move {
        let tx_for_cb = tx_for_task.clone();
        let cb: Box<dyn FnMut(ChainProgressEvent) + Send> = Box::new(move |event| {
            let _ = tx_for_cb.send(ChainSseMessage::Progress(event));
        });
        match run_chain(&state_clone, req, Some(cb)).await {
            Ok((response, elapsed_ms)) => {
                let complete = build_sse_chain_complete_event(&response, elapsed_ms);
                let _ = tx_for_task.send(ChainSseMessage::Complete(Box::new(complete)));
            }
            Err(err) => {
                let api_err: ApiError = err.into();
                let _ = tx_for_task.send(ChainSseMessage::Error(api_err.error));
            }
        }
        // `tx_for_task` is dropped here, closing the channel and finalizing
        // the SSE stream after the last complete/error frame.
    });
    drop(tx); // ensure only the task holds the sender

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, Infallible>(chain_sse_event(msg)));

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use image::{Rgb, RgbImage};
    use mold_core::chain::{ChainProgressEvent, ChainRequest, ChainStage, TransitionMode};
    use mold_core::{GenerateRequest, GenerateResponse};
    use mold_inference::ltx2::{ChainStageRenderer, ChainTail, StageOutcome, StageProgressEvent};
    use mold_inference::InferenceEngine;
    use std::sync::{Arc, Mutex};

    /// Mock engine that delegates to a simple chain renderer producing
    /// deterministic solid-color frames + a zero-valued latent tail. The
    /// chain renderer is owned by the engine so `as_chain_renderer` can
    /// hand out a `&mut dyn ChainStageRenderer` over it.
    struct ChainMockEngine {
        loaded: bool,
        fail_on_stage: Option<usize>,
        renderer_calls: Arc<Mutex<usize>>,
    }

    impl ChainMockEngine {
        fn ready() -> Self {
            Self {
                loaded: true,
                fail_on_stage: None,
                renderer_calls: Arc::new(Mutex::new(0)),
            }
        }
        fn failing_at(idx: usize) -> Self {
            Self {
                loaded: true,
                fail_on_stage: Some(idx),
                renderer_calls: Arc::new(Mutex::new(0)),
            }
        }
    }

    impl ChainStageRenderer for ChainMockEngine {
        fn render_stage(
            &mut self,
            stage_req: &GenerateRequest,
            _carry: Option<&ChainTail>,
            _motion_tail_pixel_frames: u32,
            _stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>,
        ) -> Result<StageOutcome> {
            let idx = {
                let mut calls = self.renderer_calls.lock().unwrap();
                let idx = *calls;
                *calls += 1;
                idx
            };
            if self.fail_on_stage == Some(idx) {
                anyhow::bail!("simulated chain failure at stage {idx}");
            }
            let frame_count = stage_req.frames.expect("chain stage missing frame count") as usize;
            let width = stage_req.width;
            let height = stage_req.height;
            let mut frames = Vec::with_capacity(frame_count);
            for f in 0..frame_count {
                let shade = (idx as u8).wrapping_mul(17).wrapping_add(f as u8);
                frames.push(RgbImage::from_pixel(width, height, Rgb([shade, 0, 0])));
            }
            let tail_pixel_frames = 4usize;
            let take_from = frames
                .len()
                .saturating_sub(tail_pixel_frames)
                .min(frames.len());
            let tail_rgb_frames = frames[take_from..].to_vec();
            Ok(StageOutcome {
                frames,
                tail: ChainTail {
                    frames: tail_pixel_frames as u32,
                    tail_rgb_frames,
                },
                generation_time_ms: 10,
            })
        }
    }

    impl InferenceEngine for ChainMockEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> Result<GenerateResponse> {
            anyhow::bail!("chain mock engine does not support single-shot generate")
        }
        fn model_name(&self) -> &str {
            "ltx-2-19b-distilled:mock"
        }
        fn is_loaded(&self) -> bool {
            self.loaded
        }
        fn load(&mut self) -> Result<()> {
            self.loaded = true;
            Ok(())
        }
        fn as_chain_renderer(
            &mut self,
        ) -> Option<&mut dyn mold_inference::ltx2::ChainStageRenderer> {
            Some(self)
        }
    }

    /// Build an AppState whose model cache already contains a chain-capable
    /// mock engine under the model name the tests pass in their requests.
    fn state_with_chain_engine(engine: ChainMockEngine) -> AppState {
        AppState::with_engine(engine)
    }

    fn chain_req_for_mock(model: &str, stages: u32) -> ChainRequest {
        ChainRequest {
            model: model.to_string(),
            stages: (0..stages)
                .map(|_| ChainStage {
                    prompt: "a cat walking".into(),
                    frames: 9,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: TransitionMode::Smooth,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                })
                .collect(),
            motion_tail_frames: 0, // simplifies frame accounting for the mock
            width: 64,
            height: 64,
            fps: 12,
            seed: Some(42),
            steps: 4,
            guidance: 3.0,
            strength: 1.0,
            output_format: OutputFormat::Apng, // avoid needing the mp4 feature in tests
            placement: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
        }
    }

    #[tokio::test]
    async fn chain_happy_path_returns_stage_count_and_video() {
        let engine = ChainMockEngine::ready();
        let state = state_with_chain_engine(engine);
        let req = chain_req_for_mock("ltx-2-19b-distilled:mock", 3);

        let (resp, elapsed_ms) = run_chain(&state, req, None)
            .await
            .expect("chain run succeeds");

        assert_eq!(resp.stage_count, 3, "response must report all 3 stages");
        assert_eq!(resp.video.fps, 12);
        assert_eq!(resp.video.frames, 9 * 3, "3 stages × 9 frames with tail=0");
        assert_eq!(resp.video.format, OutputFormat::Apng);
        assert!(!resp.video.data.is_empty(), "apng bytes written");
        // elapsed_ms is the sum of the mock's reported per-stage time (10ms each).
        assert_eq!(elapsed_ms, 30);
    }

    #[tokio::test]
    async fn chain_stream_emits_progress_then_complete_in_order() {
        let engine = ChainMockEngine::ready();
        let state = state_with_chain_engine(engine);
        let req = chain_req_for_mock("ltx-2-19b-distilled:mock", 2);

        let collected: Arc<Mutex<Vec<ChainProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let collected_cb = collected.clone();
        let cb: Box<dyn FnMut(ChainProgressEvent) + Send> = Box::new(move |ev| {
            collected_cb.lock().unwrap().push(ev);
        });
        let (resp, _) = run_chain(&state, req, Some(cb))
            .await
            .expect("chain run succeeds");

        assert_eq!(resp.stage_count, 2);
        let events = collected.lock().unwrap();
        assert!(!events.is_empty(), "progress events must flow");
        assert!(
            matches!(
                events[0],
                ChainProgressEvent::ChainStart { stage_count: 2, .. }
            ),
            "first event must be ChainStart, got {:?}",
            events[0]
        );
        assert!(
            matches!(events.last().unwrap(), ChainProgressEvent::Stitching { .. }),
            "last event must be Stitching, got {:?}",
            events.last()
        );
        // There must be exactly one StageStart + StageDone per stage.
        let stage_starts = events
            .iter()
            .filter(|e| matches!(e, ChainProgressEvent::StageStart { .. }))
            .count();
        let stage_dones = events
            .iter()
            .filter(|e| matches!(e, ChainProgressEvent::StageDone { .. }))
            .count();
        assert_eq!(stage_starts, 2);
        assert_eq!(stage_dones, 2);
    }

    #[tokio::test]
    async fn chain_mid_chain_failure_maps_to_bad_gateway() {
        let engine = ChainMockEngine::failing_at(1);
        let state = state_with_chain_engine(engine);
        let req = chain_req_for_mock("ltx-2-19b-distilled:mock", 3);

        let err = run_chain(&state, req, None)
            .await
            .expect_err("mid-chain failure must bubble up");
        match err {
            ChainRunError::StageFailed(failure) => {
                assert_eq!(
                    failure.failed_stage_idx, 1,
                    "failed_stage_idx must be 1, got {}",
                    failure.failed_stage_idx
                );
                assert!(
                    failure.stage_error.contains("simulated chain failure"),
                    "stage_error must carry renderer message, got: {}",
                    failure.stage_error
                );
            }
            other => panic!("expected StageFailed error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn generate_chain_handler_returns_502_with_chain_failure_body() {
        use axum::body::to_bytes;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;

        let engine = ChainMockEngine::failing_at(1);
        let state = state_with_chain_engine(engine);
        let req = chain_req_for_mock("ltx-2-19b-distilled:mock", 3);

        let resp = generate_chain(State(state), Json(req)).await;
        let (parts, body) = resp.into_response().into_parts();

        assert_eq!(parts.status, StatusCode::BAD_GATEWAY, "must be 502");

        let bytes = to_bytes(body, usize::MAX).await.unwrap();
        let failure: mold_core::chain::ChainFailure =
            serde_json::from_slice(&bytes).expect("body must be ChainFailure JSON");

        assert_eq!(
            failure.failed_stage_idx, 1,
            "failed_stage_idx must be 1, got {}",
            failure.failed_stage_idx
        );
        assert!(
            failure.stage_error.contains("simulated chain failure"),
            "stage_error must carry renderer message, got: {}",
            failure.stage_error
        );
    }

    #[tokio::test]
    async fn chain_unsupported_model_rejects_with_validation() {
        /// Engine that is fully capable of single-shot generate but refuses
        /// chain rendering (mirrors every non-LTX-2 family).
        struct NonChainEngine;
        impl InferenceEngine for NonChainEngine {
            fn generate(&mut self, _req: &GenerateRequest) -> Result<GenerateResponse> {
                anyhow::bail!("no single-shot generate in this test either")
            }
            fn model_name(&self) -> &str {
                "flux-dev:q8"
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn load(&mut self) -> Result<()> {
                Ok(())
            }
            // No override for as_chain_renderer — default returns None.
        }

        let state = AppState::with_engine(NonChainEngine);
        let mut req = chain_req_for_mock("flux-dev:q8", 2);
        req.model = "flux-dev:q8".into();
        let err = run_chain(&state, req, None)
            .await
            .expect_err("non-chain model must fail");
        match err {
            ChainRunError::UnsupportedModel(msg) => {
                assert!(
                    msg.contains("does not support chained video generation"),
                    "unsupported-model error must name the constraint, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedModel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chain_trims_frames_from_tail_when_total_frames_set() {
        let engine = ChainMockEngine::ready();
        let state = state_with_chain_engine(engine);
        let mut req = chain_req_for_mock("ltx-2-19b-distilled:mock", 2);
        // Each stage produces 9 frames with tail=0 → 18 total. Trim to 10.
        req.total_frames = Some(10);

        let (resp, _) = run_chain(&state, req, None).await.expect("chain runs");
        assert_eq!(
            resp.video.frames, 10,
            "total_frames must trim the stitched output length"
        );
    }
}
