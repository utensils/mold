//! Server-side chained video generation endpoints.
//!
//! Exposes `POST /api/generate/chain` (synchronous) and
//! `POST /api/generate/chain/stream` (SSE). Both are compatibility shims over
//! the durable chain-job runner: create an ephemeral job, map runner events
//! back to the legacy wire shape, and delete successful ephemeral artifacts
//! after assembling the response.

use axum::{
    extract::State,
    http::HeaderMap,
    response::sse::{Event as SseEvent, KeepAlive, Sse},
    Json,
};
use base64::Engine as _;
use mold_core::chain::{
    ChainFailure, ChainProgressEvent, ChainRequest, ChainResponse, ChainScript,
    SseChainCompleteEvent,
};
use mold_core::chain_job::{settled, ChainJobEvent, ChainJobState};
use mold_core::{OutputFormat, OutputMetadata, VideoData};
use mold_db::MetadataDb;
use std::convert::Infallible;
use tokio_stream::StreamExt as _;

use crate::chain_job_runner::{ChainJobRunnerHandle, EphemeralClaimGuard};
use crate::model_manager::ExistingModelAuthority;
use crate::queue::save_video_to_dir;
use crate::routes::{requested_sse_completion_payload, ApiError};
use crate::state::{AppState, SseCompletionPayload};

fn chain_freeze_error(model: &str, error: impl std::fmt::Display) -> ApiError {
    ApiError::validation(format!(
        "cannot freeze concrete chain model companions for '{model}': {error}"
    ))
}

pub(crate) async fn resolve_chain_model_authority(
    state: &AppState,
    model: &str,
) -> Result<ExistingModelAuthority, ApiError> {
    let config = state.config.read().await.clone();
    crate::model_manager::resolve_existing_model_authority(model, &config)
        .map_err(|error| chain_freeze_error(model, error.error))?
        .ok_or_else(|| {
            chain_freeze_error(
                model,
                format!("model '{model}' has no concrete local artifact paths"),
            )
        })
}

pub(crate) fn freeze_chain_model(
    authority: ExistingModelAuthority,
    model: &str,
) -> Result<mold_core::chain_job::FrozenChainModel, ApiError> {
    crate::execution_plan::freeze_chain_model_with_paths(&authority.config, model, authority.paths)
        .map_err(|error| chain_freeze_error(model, error))
}

/// Internal wire event used by the chain SSE stream before per-event
/// serialization. Separate from [`crate::state::SseMessage`] because chain
/// complete events carry a different payload (`SseChainCompleteEvent`) and
/// progress events are chain-shaped (`ChainProgressEvent`) rather than the
/// single-stage `SseProgressEvent`.
pub(crate) enum ChainSseMessage {
    Progress(serde_json::Value),
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

async fn shim_start_job(state: &AppState, req: ChainRequest) -> Result<ShimJob, ApiError> {
    let handle = state.chain_jobs.as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable chain jobs are unavailable because the metadata DB is disabled",
            "CHAIN_JOBS_UNAVAILABLE",
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
        )
    })?;
    let db = state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable chain jobs are unavailable because the metadata DB is disabled",
            "CHAIN_JOBS_UNAVAILABLE",
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
        )
    })?;

    let mut req = req;
    validate_chain_build_features(&req)?;
    let authority = resolve_chain_model_authority(state, &req.model).await?;
    validate_and_normalize_chain_family(&authority.config, &mut req)?;
    let mut req = req
        .normalise()
        .map_err(|e| ApiError::validation(e.to_string()))?;
    validate_and_normalize_chain_family(&authority.config, &mut req)?;

    let original_format = req.output_format;
    req.output_format = OutputFormat::Mp4;
    let job_id = uuid::Uuid::new_v4().to_string();
    let guard = handle.claim_ephemeral(&job_id);
    let jobs_root = crate::routes_chain_jobs::jobs_root()?;
    if let Err(err) = crate::chain_job_runner::create_job_with_params(
        db,
        &jobs_root,
        crate::chain_job_runner::CreateJobParams {
            id: job_id.clone(),
            ephemeral: true,
            frozen_model: Some(freeze_chain_model(authority, &req.model)?),
            request: req,
        },
    ) {
        drop(guard);
        return Err(ApiError::internal(format!(
            "failed to create legacy chain shim job: {err:#}"
        )));
    }
    handle.kick();
    Ok(ShimJob {
        job_id,
        original_format,
        guard,
    })
}

struct ShimJob {
    job_id: String,
    original_format: OutputFormat,
    guard: EphemeralClaimGuard,
}

/// Internal result from assembling a legacy shim response. `response` keeps
/// the synchronous endpoint's historical wire shape, while the additive
/// gallery provenance is available to the SSE completion builder.
struct ShimBuildResult {
    response: ChainResponse,
    filename: Option<String>,
    metadata: Option<OutputMetadata>,
}

/// Subscribe FIRST (subscribe requires &MetadataDb — R2 signature), then
/// settlement is observed as: StateChanged(settled) event ∪ (closed/lagged
/// receiver → DB re-read of state+error — subscribe_for_job hands settled
/// jobs a closed receiver by design). Timeout-free because the R2 terminal
/// contract guarantees settlement AND the closed-receiver branch covers
/// publish_then_remove racing the subscribe.
async fn shim_wait_settled(
    handle: &ChainJobRunnerHandle,
    db: &MetadataDb,
    job_id: &str,
) -> Result<(ChainJobState, Option<String>), ApiError> {
    loop {
        let mut live = handle
            .subscribe(db, job_id)
            .map_err(|e| ApiError::internal(format!("failed to subscribe to shim job: {e:#}")))?;
        match live.recv().await {
            Ok(ChainJobEvent::StateChanged { state, error }) if settled(state) => {
                return Ok((state, error));
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_))
            | Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                let row = mold_db::chain_jobs::get_job(db, job_id).map_err(|e| {
                    ApiError::internal(format!("failed to re-read shim job state: {e:#}"))
                })?;
                let Some(row) = row else {
                    return Err(ApiError::internal(format!(
                        "legacy chain shim job {job_id} disappeared before settlement"
                    )));
                };
                if settled(row.state) {
                    return Ok((row.state, row.error));
                }
            }
        }
    }
}

fn shim_chain_failure(
    db: &MetadataDb,
    job_id: &str,
    settled_error: Option<String>,
) -> Result<ChainFailure, ApiError> {
    let row = mold_db::chain_jobs::get_job(db, job_id)
        .map_err(|e| ApiError::internal(format!("failed to read failed shim job row: {e:#}")))?
        .ok_or_else(|| {
            ApiError::internal(format!(
                "legacy chain shim job {job_id} disappeared before failure response"
            ))
        })?;
    let stages = mold_db::chain_jobs::stages_for_job(db, job_id)
        .map_err(|e| ApiError::internal(format!("failed to read failed shim job stages: {e:#}")))?;
    let failed = stages
        .iter()
        .find(|stage| stage.state == mold_core::chain_job::StageState::Failed);
    let elapsed_stages = stages
        .iter()
        .filter(|stage| stage.state == mold_core::chain_job::StageState::Completed)
        .count() as u32;
    let elapsed_ms = stages
        .iter()
        .filter(|stage| stage.state == mold_core::chain_job::StageState::Completed)
        .filter_map(|stage| stage.generation_time_ms)
        .sum::<u64>();
    let failed_stage_idx = failed
        .map(|stage| stage.stage_idx)
        .unwrap_or_else(|| row.current_stage.min(row.stage_count.saturating_sub(1)));
    let stage_error = failed
        .and_then(|stage| stage.error.clone())
        .or(row.error)
        .or(settled_error)
        .unwrap_or_else(|| "chain job settled as failed".to_string());

    Ok(ChainFailure {
        error: "stage render failed".into(),
        failed_stage_idx,
        elapsed_stages,
        elapsed_ms,
        stage_error,
    })
}

/// Build the legacy ChainResponse from the settled job. MP4 PASSTHROUGH
/// PIN: original_format == Mp4 → response bytes are final/ MP4 VERBATIM
/// (preserves the AAC track finalize muxed; no redundant re-encode); frames
/// are decoded ONLY for thumbnail + gif_preview. Non-Mp4 → decode
/// (decode_video_frames_from_path) + transcode via encode_chain_output
/// (format field reports ACTUAL bytes, incl. the Webp→Apng fallback).
/// generation_time_ms = Σ stage rows, script echo. THEN delete job dir +
/// row (read-then-delete; single requester owns the claim).
fn shim_build_response_and_cleanup(
    state: &AppState,
    shim: ShimJob,
) -> Result<ShimBuildResult, ApiError> {
    let handle = state.chain_jobs.as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable chain jobs are unavailable because the metadata DB is disabled",
            "CHAIN_JOBS_UNAVAILABLE",
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
        )
    })?;
    let db = state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable chain jobs are unavailable because the metadata DB is disabled",
            "CHAIN_JOBS_UNAVAILABLE",
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
        )
    })?;
    let _claim = shim.guard;
    let _guard = handle.blocking_lock_job(&shim.job_id);
    let row = mold_db::chain_jobs::get_job(db, &shim.job_id)
        .map_err(|e| ApiError::internal(format!("failed to read shim job row: {e:#}")))?
        .ok_or_else(|| {
            ApiError::internal(format!(
                "legacy chain shim job {} disappeared before response build",
                shim.job_id
            ))
        })?;
    let manifest = mold_core::chain_job::ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|e| ApiError::internal(format!("failed to read shim job manifest: {e:#}")))?;
    let final_record = manifest.finalizes.last().ok_or_else(|| {
        ApiError::internal(format!(
            "legacy chain shim job {} completed without a final output",
            shim.job_id
        ))
    })?;
    let final_path = row.job_dir.join(&final_record.output);
    let (probe, frames) = mold_inference::ltx2::media::decode_video_frames_from_path(&final_path)
        .map_err(|e| {
        ApiError::internal(format!(
            "failed to decode legacy chain shim output '{}': {e:#}",
            final_path.display()
        ))
    })?;
    if frames.is_empty() {
        return Err(ApiError::internal(
            "legacy chain shim output decoded to zero frames",
        ));
    }
    let mut req = crate::chain_job_runner::effective_request(&manifest)
        .map_err(|e| ApiError::internal(format!("failed to read effective shim request: {e:#}")))?;
    req.output_format = shim.original_format;

    let (bytes, actual_format, gif_preview) = if shim.original_format == OutputFormat::Mp4 {
        let bytes = std::fs::read(&final_path).map_err(|e| {
            ApiError::internal(format!(
                "failed to read legacy chain shim final MP4 '{}': {e}",
                final_path.display()
            ))
        })?;
        let gif_preview = match mold_inference::ltx_video::video_enc::encode_gif(&frames, req.fps) {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!("legacy chain shim GIF preview encode failed: {err:#}");
                Vec::new()
            }
        };
        (bytes, OutputFormat::Mp4, gif_preview)
    } else {
        encode_chain_output(&frames, req.fps, shim.original_format, None)
            .map_err(|e| ApiError::internal(format!("encode legacy chain shim output: {e:#}")))?
    };
    let thumbnail = chain_thumbnail(&frames);
    let frame_count = probe.frames.unwrap_or(frames.len() as u32);
    let generation_time_ms = manifest
        .stage_status
        .iter()
        .filter_map(|stage| stage.generation_time_ms)
        .sum::<u64>();

    let output_dir = {
        let config = state.config.blocking_read();
        if state.is_output_disabled(&config) {
            None
        } else {
            Some(config.effective_output_dir())
        }
    };
    let (filename, output_metadata) = if let Some(dir) = output_dir {
        let _gallery_writer = state.gallery_publication_gate.blocking_write();
        // The ephemeral shim job dir is deleted right after this response is
        // assembled, so a job id would dangle — record stage seeds only.
        let stage_seeds: Vec<u64> = manifest
            .stage_status
            .iter()
            .map(|stage| stage.seed)
            .collect();
        let provenance = mold_core::chain::ChainProvenance {
            chain_job_id: None,
            stage_seeds: Some(&stage_seeds),
        };
        let metadata = req.stitched_output_metadata(actual_format, frame_count, Some(&provenance));
        let filename = save_video_to_dir(
            &dir,
            &bytes,
            &gif_preview,
            actual_format,
            &req.model,
            &metadata,
            Some(generation_time_ms as i64),
            Some(db),
            Some(&state.events),
            &state.gallery_publication_gate,
        );
        (filename, Some(metadata))
    } else {
        (None, None)
    };

    let video = VideoData {
        data: bytes,
        format: actual_format,
        width: probe.width,
        height: probe.height,
        frames: frame_count,
        fps: probe.fps,
        thumbnail,
        gif_preview,
        has_audio: actual_format == OutputFormat::Mp4 && probe.has_audio,
        duration_ms: probe
            .duration_ms
            .or_else(|| (probe.fps > 0).then_some((frame_count as u64 * 1000) / probe.fps as u64)),
        audio_sample_rate: (actual_format == OutputFormat::Mp4)
            .then_some(probe.audio_sample_rate)
            .flatten(),
        audio_channels: (actual_format == OutputFormat::Mp4)
            .then_some(probe.audio_channels)
            .flatten(),
    };
    let response = ChainResponse {
        video,
        stage_count: manifest.stage_status.len() as u32,
        gpu: None,
        script: ChainScript::from(&req),
        vram_estimate: None,
    };

    if row.job_dir.exists() {
        std::fs::remove_dir_all(&row.job_dir).map_err(|e| {
            ApiError::internal(format!(
                "failed to remove legacy chain shim job dir '{}': {e}",
                row.job_dir.display()
            ))
        })?;
    }
    mold_db::chain_jobs::delete_job_not_running(db, &shim.job_id).map_err(|e| {
        ApiError::internal(format!("failed to delete legacy chain shim row: {e:#}"))
    })?;
    handle.cleanup_deleted(&shim.job_id);
    handle.remove_job_lock(&shim.job_id);
    Ok(ShimBuildResult {
        response,
        filename,
        metadata: output_metadata,
    })
}

/// Legacy event mapping for the streaming shim: ChainJobEvent → Option of
/// legacy-shaped progress JSON with the ADDITIVE job_id field injected
/// (serialize ChainProgressEvent to Value, insert "job_id" — P1 rev2
/// mechanism). Snapshot → synthesized chain_start; StageStart/DenoiseStep/
/// StageDone map 1:1; Finalizing → stitching; Yielded/Finalized/
/// StateChanged(non-settled) → None.
fn map_job_event_to_legacy(ev: &ChainJobEvent, job_id: &str) -> Option<serde_json::Value> {
    let progress = match ev {
        ChainJobEvent::Snapshot { job } => {
            let estimated_total_frames = job
                .script
                .stages
                .iter()
                .map(|stage| stage.frames)
                .sum::<u32>();
            ChainProgressEvent::ChainStart {
                stage_count: job.summary.stage_count,
                estimated_total_frames,
            }
        }
        ChainJobEvent::StageStart { stage_idx } => ChainProgressEvent::StageStart {
            stage_idx: *stage_idx,
        },
        ChainJobEvent::DenoiseStep {
            stage_idx,
            step,
            total,
        } => ChainProgressEvent::DenoiseStep {
            stage_idx: *stage_idx,
            step: *step,
            total: *total,
        },
        ChainJobEvent::StageDone {
            stage_idx,
            frames_emitted,
            ..
        } => ChainProgressEvent::StageDone {
            stage_idx: *stage_idx,
            frames_emitted: *frames_emitted,
        },
        ChainJobEvent::Finalizing { total_frames } => ChainProgressEvent::Stitching {
            total_frames: *total_frames,
        },
        ChainJobEvent::Yielded { .. }
        | ChainJobEvent::Finalized { .. }
        | ChainJobEvent::StateChanged { .. } => return None,
    };
    let mut value = serde_json::to_value(progress).ok()?;
    if let serde_json::Value::Object(object) = &mut value {
        object.insert(
            "job_id".to_string(),
            serde_json::Value::String(job_id.to_string()),
        );
    }
    Some(value)
}

/// Encode chain frames into bytes for the requested output format. Returns
/// the encoded payload plus a best-effort animated-GIF preview for the
/// gallery.
///
/// MP4 is gated behind the `mp4` feature flag; when the flag is disabled,
/// the handler falls back to APNG so the endpoint still produces a usable
/// animation on every build. When `audio` is `Some` and the output format
/// is MP4, the audio is muxed in as an AAC track via the same path the
/// single-clip pipeline uses; non-MP4 formats silently drop audio because
/// they have no carrier for it.
fn encode_chain_output(
    frames: &[image::RgbImage],
    fps: u32,
    format: OutputFormat,
    audio: Option<&mold_inference::chain::NativeAudioTrack>,
) -> anyhow::Result<(Vec<u8>, OutputFormat, Vec<u8>)> {
    let encoded = mold_inference::chain::encode_chain_frames(frames, fps, format, audio)?;
    for warning in &encoded.warnings {
        tracing::warn!("{}", warning.message());
    }
    Ok((encoded.bytes, encoded.format, encoded.gif_preview))
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

/// Build the SSE `complete` payload for a finished chain run. Sibling of
/// [`crate::queue::build_sse_complete_event`] — kept in this module so the
/// chain-specific payload can evolve independently from the single-shot
/// one.
fn build_sse_chain_complete_event(
    result: &ShimBuildResult,
    generation_time_ms: u64,
    payload: SseCompletionPayload,
) -> SseChainCompleteEvent {
    let b64 = base64::engine::general_purpose::STANDARD;
    let resp = &result.response;
    let video = &resp.video;
    let include_media = payload == SseCompletionPayload::Full;
    SseChainCompleteEvent {
        video: if include_media {
            b64.encode(&video.data)
        } else {
            String::new()
        },
        format: video.format,
        width: video.width,
        height: video.height,
        frames: video.frames,
        fps: video.fps,
        thumbnail: if !include_media || video.thumbnail.is_empty() {
            None
        } else {
            Some(b64.encode(&video.thumbnail))
        },
        gif_preview: if !include_media || video.gif_preview.is_empty() {
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
        filename: result.filename.clone(),
        metadata: result.metadata.clone().map(Box::new),
    }
}

/// Validate the chain request's model family and apply family-specific
/// fixups. Returns `Err` with a 422 response if the family doesn't support
/// chain generation. Mutates `req.motion_tail_frames` for families that lack
/// latent context handoff (currently `ltx-video`) so the stitch layer doesn't
/// trim independent fresh frames at Smooth boundaries.
pub(crate) fn validate_chain_build_features(_req: &ChainRequest) -> Result<(), ApiError> {
    #[cfg(not(feature = "mp4"))]
    if _req.enable_audio == Some(true) {
        return Err(ApiError::validation(
            "chain audio requires a mold build with the mp4 feature; rebuild with `--features mp4` or remove `enable_audio: true`",
        ));
    }
    Ok(())
}

pub(crate) fn validate_and_normalize_chain_family(
    config: &mold_core::Config,
    req: &mut ChainRequest,
) -> Result<(), ApiError> {
    validate_chain_build_features(req)?;

    let resolved_model = config.resolved_model_config(&req.model);
    let canonical_model = mold_core::manifest::resolve_model_name(&req.model);
    let manifest = mold_core::manifest::find_manifest(&canonical_model);
    let family = resolved_model
        .family
        .clone()
        .or_else(|| manifest.map(|model| model.family.clone()))
        .unwrap_or_default();
    // Only reject early when we positively know the family is non-chain-capable.
    // An empty family means the model isn't in the manifest yet (catalog
    // synth, mock test, etc.) — let it through; the engine's
    // `as_chain_renderer()` check will still fire if it really can't render.
    if !family.is_empty() && crate::chain_limits::family_cap(&family).is_none() {
        return Err(ApiError::validation(format!(
            "model '{}' (family '{}') does not support chained video generation",
            req.model, family
        )));
    }
    if !family.is_empty() {
        let sequence = crate::chain_limits::sequence_support(
            &req.model,
            &family,
            resolved_model.spatial_upscaler.is_some()
                || manifest.is_some_and(|model| {
                    model.files.iter().any(|file| {
                        file.component == mold_core::manifest::ModelComponent::SpatialUpscaler
                    })
                }),
        );
        if !sequence.supported {
            return Err(ApiError::validation(sequence.reason.unwrap_or_else(|| {
                format!("model '{}' cannot render sequences", req.model)
            })));
        }
    }
    if family == "ltx-video" && req.motion_tail_frames > 0 {
        // LtxVideoEngine has no img2vid path, so the carry tail can't anchor
        // the next stage's denoise. Zero motion_tail makes Smooth boundaries
        // collapse to clean concatenation at the stitch layer (Smooth is
        // implemented as `next_clip.skip(motion_tail)` — with 0, no skip).
        tracing::debug!(
            model = %req.model,
            original = req.motion_tail_frames,
            "ltx-video has no context handoff; forcing motion_tail_frames=0"
        );
        req.motion_tail_frames = 0;
    }
    // Audio is only emitted by AV-capable families (currently LTX-2 / LTX-2.3).
    // Reject `enable_audio: true` for video-only families (e.g. ltx-video) at
    // the wire boundary so users get a clear upfront error instead of
    // silently muted output. Empty `family` (mock / catalog-synth models) is
    // permissive — the engine's renderer abstraction is the final gate.
    if req.enable_audio == Some(true)
        && !family.is_empty()
        && !crate::chain_limits::family_supports_audio(&family)
    {
        return Err(ApiError::validation(format!(
            "model '{}' (family '{}') does not support chain audio; \
             remove `enable_audio: true` or pick an LTX-2 / LTX-2.3 model",
            req.model, family
        )));
    }
    Ok(())
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

    if let Some(reason) = state.generation_unavailable() {
        return ApiError::generation_unavailable(reason).into_response();
    }
    let shim = match shim_start_job(&state, req).await {
        Ok(shim) => shim,
        Err(api_err) => return api_err.into_response(),
    };
    tracing::info!(job_id = %shim.job_id, "generate/chain legacy shim request");
    let handle = match state.chain_jobs.as_ref() {
        Some(handle) => handle.clone(),
        None => {
            return ApiError::with_code(
                "durable chain jobs are unavailable because the metadata DB is disabled",
                "CHAIN_JOBS_UNAVAILABLE",
                StatusCode::SERVICE_UNAVAILABLE,
            )
            .into_response()
        }
    };
    let db = match state.metadata_db.as_ref().as_ref() {
        Some(db) => db,
        None => {
            return ApiError::with_code(
                "durable chain jobs are unavailable because the metadata DB is disabled",
                "CHAIN_JOBS_UNAVAILABLE",
                StatusCode::SERVICE_UNAVAILABLE,
            )
            .into_response()
        }
    };
    let (settled_state, error) = match shim_wait_settled(&handle, db, &shim.job_id).await {
        Ok(result) => result,
        Err(api_err) => return api_err.into_response(),
    };
    if settled_state != ChainJobState::Completed {
        if settled_state == ChainJobState::Failed {
            return match shim_chain_failure(db, &shim.job_id, error) {
                Ok(failure) => (StatusCode::BAD_GATEWAY, Json(failure)).into_response(),
                Err(api_err) => api_err.into_response(),
            };
        }
        return ApiError::internal_with_status(
            error.unwrap_or_else(|| format!("chain job settled as {}", settled_state.as_str())),
            StatusCode::BAD_GATEWAY,
        )
        .into_response();
    }
    let state_for_build = state.clone();
    match tokio::task::spawn_blocking(move || {
        shim_build_response_and_cleanup(&state_for_build, shim)
    })
    .await
    {
        Ok(Ok(result)) => Json(result.response).into_response(),
        Ok(Err(api_err)) => api_err.into_response(),
        Err(err) => ApiError::internal(format!("legacy chain shim response task failed: {err}"))
            .into_response(),
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
    params((
        "X-Mold-SSE-Payload" = Option<String>,
        Header,
        description = "Set to metadata-only to omit encoded media and return the saved gallery filename"
    )),
    responses(
        (status = 200, description = "SSE event stream with chain progress and completion"),
        (status = 422, description = "Invalid request or unsupported model"),
        (status = 500, description = "Chain render failed"),
    )
)]
pub async fn generate_chain_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChainRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    if let Some(reason) = state.generation_unavailable() {
        return Err(ApiError::generation_unavailable(reason));
    }
    let completion_payload = requested_sse_completion_payload(&headers)?;
    let output_disabled = {
        let config = state.config.read().await;
        state.is_output_disabled(&config)
    };
    if completion_payload == SseCompletionPayload::MetadataOnly && output_disabled {
        return Err(ApiError::validation(
            "metadata-only SSE completions require server gallery output to be enabled",
        ));
    }
    let shim = shim_start_job(&state, req).await?;
    tracing::info!(job_id = %shim.job_id, "generate/chain/stream legacy shim request");
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChainSseMessage>();
    let state_clone = state.clone();
    let db_holder = state.metadata_db.clone();
    let tx_for_task = tx.clone();
    let handle = state
        .chain_jobs
        .as_ref()
        .ok_or_else(|| {
            ApiError::with_code(
                "durable chain jobs are unavailable because the metadata DB is disabled",
                "CHAIN_JOBS_UNAVAILABLE",
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
            )
        })?
        .clone();

    tokio::spawn(async move {
        let job_id = shim.job_id.clone();
        let db = match db_holder.as_ref().as_ref() {
            Some(db) => db,
            None => {
                let _ = tx_for_task.send(ChainSseMessage::Error(
                    "durable chain jobs are unavailable because the metadata DB is disabled"
                        .to_string(),
                ));
                return;
            }
        };
        let root = match crate::routes_chain_jobs::jobs_root() {
            Ok(root) => root,
            Err(err) => {
                let _ = tx_for_task.send(ChainSseMessage::Error(err.error));
                return;
            }
        };
        match crate::routes_chain_jobs::job_detail_for(db, &root, &job_id) {
            Ok(Some(detail)) => {
                if let Some(progress) =
                    map_job_event_to_legacy(&ChainJobEvent::Snapshot { job: detail }, &job_id)
                {
                    let _ = tx_for_task.send(ChainSseMessage::Progress(progress));
                }
            }
            Ok(None) => {
                let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                    "legacy chain shim job {job_id} disappeared before stream snapshot"
                )));
                return;
            }
            Err(err) => {
                let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                    "failed to load legacy chain shim snapshot: {err:#}"
                )));
                return;
            }
        }

        let mut live = match handle.subscribe(db, &job_id) {
            Ok(live) => live,
            Err(err) => {
                let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                    "failed to subscribe to legacy chain shim job: {err:#}"
                )));
                return;
            }
        };
        let mut terminal: Option<(ChainJobState, Option<String>)> = None;
        while terminal.is_none() {
            match live.recv().await {
                Ok(event) => {
                    if let Some(progress) = map_job_event_to_legacy(&event, &job_id) {
                        let _ = tx_for_task.send(ChainSseMessage::Progress(progress));
                    }
                    if let ChainJobEvent::StateChanged { state, error } = event {
                        if settled(state) {
                            terminal = Some((state, error));
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_))
                | Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    match mold_db::chain_jobs::get_job(db, &job_id) {
                        Ok(Some(row)) if settled(row.state) => {
                            terminal = Some((row.state, row.error));
                        }
                        Ok(Some(_)) => {
                            live = match handle.subscribe(db, &job_id) {
                                Ok(live) => live,
                                Err(err) => {
                                    let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                                        "failed to re-subscribe to legacy chain shim job: {err:#}"
                                    )));
                                    return;
                                }
                            };
                        }
                        Ok(None) => {
                            let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                                "legacy chain shim job {job_id} disappeared before settlement"
                            )));
                            return;
                        }
                        Err(err) => {
                            let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                                "failed to re-read legacy chain shim state: {err:#}"
                            )));
                            return;
                        }
                    }
                }
            }
        }

        let (settled_state, error) = terminal.expect("terminal set");
        if settled_state != ChainJobState::Completed {
            let _ =
                tx_for_task.send(ChainSseMessage::Error(error.unwrap_or_else(|| {
                    format!("chain job settled as {}", settled_state.as_str())
                })));
            return;
        }
        let generation_time_ms = mold_db::chain_jobs::get_job(db, &job_id)
            .ok()
            .flatten()
            .and_then(|row| {
                mold_core::chain_job::ChainJobManifest::read_from_dir(&row.job_dir).ok()
            })
            .map(|manifest| {
                manifest
                    .stage_status
                    .iter()
                    .filter_map(|stage| stage.generation_time_ms)
                    .sum::<u64>()
            })
            .unwrap_or(0);
        match tokio::task::spawn_blocking(move || {
            shim_build_response_and_cleanup(&state_clone, shim)
        })
        .await
        {
            Ok(Ok(result)) => {
                if completion_payload == SseCompletionPayload::MetadataOnly
                    && result.filename.is_none()
                {
                    let _ = tx_for_task.send(ChainSseMessage::Error(
                        "chain generation completed but the output could not be saved for streaming"
                            .to_string(),
                    ));
                } else {
                    let complete = build_sse_chain_complete_event(
                        &result,
                        generation_time_ms,
                        completion_payload,
                    );
                    let _ = tx_for_task.send(ChainSseMessage::Complete(Box::new(complete)));
                }
            }
            Ok(Err(api_err)) => {
                let _ = tx_for_task.send(ChainSseMessage::Error(api_err.error));
            }
            Err(err) => {
                let _ = tx_for_task.send(ChainSseMessage::Error(format!(
                    "legacy chain shim response task failed: {err}"
                )));
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
    use axum::{body::to_bytes, response::IntoResponse};
    use image::{AnimationDecoder, Rgb, RgbImage};
    use mold_core::chain::{ChainFailure, ChainRequest, ChainStage, TransitionMode};
    use mold_core::chain_job::{
        ChainJobDetail, ChainJobManifest, ChainJobStageDetail, ChainJobState, ChainJobSummary,
        FinalizeRecord, JobDirLayout, StageState,
    };
    use mold_db::{chain_jobs, MetadataDb};
    use std::ops::ControlFlow;
    use std::sync::Arc;

    fn req(format: OutputFormat) -> ChainRequest {
        ChainRequest {
            model: "ltx-2-19b-distilled:mock".into(),
            stages: vec![
                ChainStage {
                    prompt: "stage zero".into(),
                    frames: 9,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: None,
                    transition: TransitionMode::Smooth,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
                ChainStage {
                    prompt: "stage one".into(),
                    frames: 17,
                    source_image: None,
                    negative_prompt: None,
                    seed_offset: Some(7),
                    transition: TransitionMode::Cut,
                    fade_frames: None,
                    model: None,
                    loras: vec![],
                    references: vec![],
                },
            ],
            motion_tail_frames: 0,
            width: 64,
            height: 64,
            fps: 12,
            seed: Some(42),
            steps: 4,
            guidance: 3.0,
            strength: 1.0,
            output_format: format,
            placement: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    fn detail() -> ChainJobDetail {
        let request = req(OutputFormat::Mp4)
            .normalise()
            .expect("request normalises");
        ChainJobDetail {
            summary: ChainJobSummary {
                id: "job-1".into(),
                state: ChainJobState::Running,
                model: request.model.clone(),
                stage_count: request.stages.len() as u32,
                current_stage: 0,
                created_at_unix_ms: 1,
                updated_at_unix_ms: 2,
                error: None,
                ephemeral: true,
                cancelling: false,
            },
            stages: request
                .stages
                .iter()
                .enumerate()
                .map(|(idx, _)| ChainJobStageDetail {
                    idx: idx as u32,
                    state: StageState::Pending,
                    seed: 42,
                    frames_emitted: None,
                    generation_time_ms: None,
                    has_preview: false,
                    has_media: false,
                    cache_ready: false,
                    error: None,
                })
                .collect(),
            finalizes: vec![],
            retakes: vec![],
            amends: vec![],
            script: ChainScript::from(&request),
        }
    }

    struct MoldHomeGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Option<std::ffi::OsString>,
    }

    impl MoldHomeGuard {
        fn set(path: &std::path::Path) -> Self {
            let lock = crate::test_support::env_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var_os("MOLD_HOME");
            std::env::set_var("MOLD_HOME", path);
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for MoldHomeGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("MOLD_HOME", value),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    struct OutputDirEnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Option<std::ffi::OsString>,
    }

    impl OutputDirEnvGuard {
        fn disable() -> Self {
            let lock = crate::test_support::env_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var_os("MOLD_OUTPUT_DIR");
            std::env::set_var("MOLD_OUTPUT_DIR", "");
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for OutputDirEnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("MOLD_OUTPUT_DIR", value),
                None => std::env::remove_var("MOLD_OUTPUT_DIR"),
            }
        }
    }

    fn test_frame(value: u8) -> RgbImage {
        RgbImage::from_pixel(64, 64, Rgb([value, value, value]))
    }

    fn test_frames(value: u8) -> Vec<RgbImage> {
        (0..9).map(|idx| test_frame(value + idx)).collect()
    }

    fn audio_muxed_mp4_fixture() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(include_str!("testdata/audio_muxed_final_mp4.b64").trim())
            .unwrap()
    }

    fn assert_decodable_apng(bytes: &[u8]) {
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoder = image::codecs::png::PngDecoder::new(std::io::Cursor::new(bytes))
            .expect("APNG response should parse as PNG");
        let apng = decoder.apng().expect("PNG response should be animated");
        let frames = apng
            .into_frames()
            .collect_frames()
            .expect("APNG frames should decode");
        assert!(!frames.is_empty(), "APNG should contain animation frames");
    }

    async fn wait_for_bus_subscription(
        handle: &crate::chain_job_runner::ChainJobRunnerHandle,
        job_id: &str,
    ) {
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if handle.events_for_tests().contains_for_tests(job_id) {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("shim task should subscribe to job events");
    }

    async fn wait_for_single_chain_job(db: &MetadataDb) -> mold_db::chain_jobs::ChainJobRow {
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let mut jobs = chain_jobs::list_jobs(db).unwrap();
                if jobs.len() == 1 {
                    return jobs.remove(0);
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("streaming shim should create one chain job")
    }

    fn parse_sse_json_events(body: &str) -> Vec<(String, serde_json::Value)> {
        body.split("\n\n")
            .filter_map(|raw| {
                let mut event = None;
                let mut data = None;
                for line in raw.lines() {
                    if let Some(value) = line.strip_prefix("event:") {
                        event = Some(value.trim().to_string());
                    } else if let Some(value) = line.strip_prefix("data:") {
                        data = Some(value.trim().to_string());
                    }
                }
                Some((event?, serde_json::from_str(&data?).unwrap()))
            })
            .collect()
    }

    fn completed_ephemeral_job(
        db: &MetadataDb,
        jobs_root: &std::path::Path,
        job_id: &str,
        req: &ChainRequest,
        stage_times: &[u64],
        final_bytes: &[u8],
    ) -> mold_db::chain_jobs::ChainJobRow {
        let row = crate::chain_job_runner::create_job_with_params(
            db,
            jobs_root,
            crate::chain_job_runner::CreateJobParams {
                id: job_id.to_string(),
                ephemeral: true,
                frozen_model: None,
                request: req.clone(),
            },
        )
        .unwrap();
        mark_ephemeral_job_completed(db, &row, req, stage_times, final_bytes);
        chain_jobs::get_job(db, job_id).unwrap().unwrap()
    }

    fn mark_ephemeral_job_completed(
        db: &MetadataDb,
        row: &mold_db::chain_jobs::ChainJobRow,
        req: &ChainRequest,
        stage_times: &[u64],
        final_bytes: &[u8],
    ) {
        let layout = JobDirLayout::new(row.job_dir.clone());
        let final_path = layout.final_output_path(1);
        std::fs::create_dir_all(final_path.parent().unwrap()).unwrap();
        std::fs::write(&final_path, final_bytes).unwrap();

        let mut manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        for (idx, status) in manifest.stage_status.iter_mut().enumerate() {
            status.state = StageState::Completed;
            status.frames_emitted = Some(req.stages[idx].frames);
            status.generation_time_ms = Some(stage_times[idx]);
            status.error = None;
            chain_jobs::upsert_stage(
                db,
                &mold_db::chain_jobs::ChainJobStageRow {
                    job_id: row.id.clone(),
                    stage_idx: idx as u32,
                    state: StageState::Completed,
                    seed: status.seed,
                    frames_emitted: status.frames_emitted,
                    generation_time_ms: status.generation_time_ms,
                    segment_rel_path: None,
                    error: None,
                    updated_at_ms: 5_000,
                },
            )
            .unwrap();
        }
        manifest.finalizes.push(FinalizeRecord {
            output: "final/output-1.mp4".into(),
            at_unix_ms: 6_000,
            stage_seeds: manifest
                .stage_status
                .iter()
                .map(|stage| stage.seed)
                .collect(),
        });
        manifest.write_atomic(&row.job_dir).unwrap();
        chain_jobs::repair_job_from_manifest(
            db,
            &row.id,
            ChainJobState::Completed,
            req.stages.len() as u32,
            None,
            6_000,
            Some(6_000),
        )
        .unwrap();
    }

    fn state_with_db_and_handle(
        db: MetadataDb,
        handle: Arc<crate::chain_job_runner::ChainJobRunnerHandle>,
        output_dir: &std::path::Path,
    ) -> AppState {
        let model_dir = output_dir
            .parent()
            .expect("test output directory has a parent")
            .join("chain-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let transformer = model_dir.join("transformer.safetensors");
        let vae = model_dir.join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let mut config = mold_core::Config {
            output_dir: Some(output_dir.to_string_lossy().into_owned()),
            ..Default::default()
        };
        config.models.insert(
            "ltx-2-19b-distilled:mock".into(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                transformer: Some(transformer.to_string_lossy().into_owned()),
                vae: Some(vae.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        let mut state = AppState::for_tests();
        state.config = Arc::new(tokio::sync::RwLock::new(config));
        state.metadata_db = Arc::new(Some(db));
        state.chain_jobs = Some(handle);
        state
    }

    #[tokio::test]
    async fn chain_preflight_rejects_ltx2_two_stage_before_creating_a_job() {
        let state = AppState::for_tests();
        let mut request = req(OutputFormat::Mp4);
        request.model = "ltx-2.3-22b-dev:fp8".into();

        let config = state.config.read().await;
        let error = validate_and_normalize_chain_family(&config, &mut request)
            .expect_err("two-stage LTX-2 must be rejected at the API boundary");

        assert!(error.error.contains("two-stage"));
        assert!(error.error.contains("distilled"));
    }

    struct HandlerFailingExecutor;

    impl crate::chain_job_runner::StageExecutor for HandlerFailingExecutor {
        fn freeze_model(
            &self,
            model: &str,
        ) -> anyhow::Result<mold_core::chain_job::FrozenChainModel> {
            Ok(mold_core::chain_job::FrozenChainModel {
                runtime_model_id: format!("mold-frozen-chain:{model}:fixture"),
                model_fingerprint: "fixture".into(),
                config: mold_core::ModelConfig {
                    family: Some("ltx2".into()),
                    transformer: Some("/test/transformer.safetensors".into()),
                    vae: Some("/test/vae.safetensors".into()),
                    ..Default::default()
                },
            })
        }

        fn render_stage(
            &self,
            _model: &str,
            stage_req: &mold_core::GenerateRequest,
            _carry: Option<&mold_inference::chain::ChainTail>,
            _motion_tail_frames: u32,
            progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
            _cancelled: &(dyn Fn() -> bool + Send + Sync),
        ) -> anyhow::Result<crate::chain_job_runner::StageRenderOutcome> {
            let _ = progress(1, 2);
            if stage_req.prompt == "stage one" {
                anyhow::bail!("simulated chain failure at stage one");
            }
            Ok(crate::chain_job_runner::StageRenderOutcome::Done(
                mold_inference::chain::StageOutcome {
                    tail: mold_inference::chain::ChainTail {
                        frames: 0,
                        tail_rgb_frames: Vec::new(),
                    },
                    frames: test_frames(80),
                    audio: None,
                    generation_time_ms: 10,
                },
            ))
        }
    }

    struct NoQueuedSmallJobs;

    impl crate::chain_job_runner::QueueProbe for NoQueuedSmallJobs {
        fn small_jobs_waiting(&self) -> usize {
            0
        }
    }

    #[test]
    fn shim_mp4_passthrough_returns_final_file_bytes_audio_metadata_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("mold-home");
        let _home_guard = MoldHomeGuard::set(&home);
        let db = MetadataDb::open_in_memory().unwrap();
        let jobs_root = dir.path().join("jobs");
        let output_dir = dir.path().join("gallery");
        let mut request = req(OutputFormat::Mp4);
        request.original_prompt = Some("source prompt".into());
        request.batch_id = Some("prepared-batch-1".into());
        request.batch_index = Some(2);
        request.batch_count = Some(3);
        let request = request.normalise().unwrap();
        let final_bytes = audio_muxed_mp4_fixture();
        let row = completed_ephemeral_job(
            &db,
            &jobs_root,
            "01JBR55SHIMDONE",
            &request,
            &[11, 22],
            &final_bytes,
        );
        let job_dir = row.job_dir.clone();
        let final_path = JobDirLayout::new(job_dir.clone()).final_output_path(1);
        let expected_final_bytes = std::fs::read(&final_path).unwrap();
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        let guard = handle.claim_ephemeral(&row.id);
        let state = state_with_db_and_handle(db, handle, &output_dir);

        let result = shim_build_response_and_cleanup(
            &state,
            ShimJob {
                job_id: row.id.clone(),
                original_format: OutputFormat::Mp4,
                guard,
            },
        )
        .unwrap();
        let response = &result.response;

        assert_eq!(response.video.format, OutputFormat::Mp4);
        assert_eq!(response.video.data, expected_final_bytes);
        assert!(!response.video.thumbnail.is_empty());
        assert!(!response.video.gif_preview.is_empty());
        assert!(response.video.has_audio);
        assert_eq!(response.video.audio_sample_rate, Some(48_000));
        assert_eq!(response.video.audio_channels, Some(2));
        assert_eq!(response.stage_count, 2);
        assert_eq!(
            serde_json::to_value(&response.script).unwrap(),
            serde_json::to_value(ChainScript::from(&request)).unwrap()
        );
        let complete = build_sse_chain_complete_event(&result, 33, SseCompletionPayload::Full);
        assert_eq!(complete.generation_time_ms, Some(33));
        let complete_metadata = complete.metadata.expect("saved metadata rides completion");
        assert_eq!(
            complete_metadata.original_prompt.as_deref(),
            Some("source prompt")
        );
        assert_eq!(
            complete_metadata.batch_id.as_deref(),
            Some("prepared-batch-1")
        );
        assert_eq!(complete_metadata.batch_index, Some(2));
        assert_eq!(complete_metadata.batch_count, Some(3));
        let db = state.metadata_db.as_ref().as_ref().unwrap();
        let rows = db.list(Some(&output_dir)).unwrap();
        assert_eq!(rows.len(), 1, "shim owns the sole ephemeral gallery row");
        assert_eq!(rows[0].generation_time_ms, Some(33));
        assert_eq!(
            rows[0].metadata.batch_id.as_deref(),
            Some("prepared-batch-1")
        );
        assert!(!job_dir.exists(), "shim deletes the ephemeral job dir");
        assert!(chain_jobs::get_job(db, "01JBR55SHIMDONE")
            .unwrap()
            .is_none());
    }

    #[test]
    fn shim_apng_request_returns_decodable_apng_with_previews() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("mold-home");
        let _home_guard = MoldHomeGuard::set(&home);
        let db = MetadataDb::open_in_memory().unwrap();
        let jobs_root = dir.path().join("jobs");
        let output_dir = dir.path().join("gallery");
        let request = req(OutputFormat::Apng).normalise().unwrap();
        let final_bytes = audio_muxed_mp4_fixture();
        let row = completed_ephemeral_job(
            &db,
            &jobs_root,
            "01JBR55SHIMAPNG",
            &request,
            &[11, 22],
            &final_bytes,
        );
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        let guard = handle.claim_ephemeral(&row.id);
        let state = state_with_db_and_handle(db, handle, &output_dir);

        let result = shim_build_response_and_cleanup(
            &state,
            ShimJob {
                job_id: row.id.clone(),
                original_format: OutputFormat::Apng,
                guard,
            },
        )
        .unwrap();
        let response = &result.response;

        assert_eq!(response.video.format, OutputFormat::Apng);
        assert_decodable_apng(&response.video.data);
        assert!(!response.video.thumbnail.is_empty());
        assert!(!response.video.gif_preview.is_empty());
        assert!(!response.video.has_audio);
        assert_eq!(response.video.audio_sample_rate, None);
        assert_eq!(response.video.audio_channels, None);
    }

    #[tokio::test]
    async fn shim_wait_settled_observes_live_completed_state_change() {
        let db = MetadataDb::open_in_memory().unwrap();
        let jobs_root = tempfile::tempdir().unwrap();
        let request = req(OutputFormat::Mp4).normalise().unwrap();
        let row = crate::chain_job_runner::create_job_with_params(
            &db,
            jobs_root.path(),
            crate::chain_job_runner::CreateJobParams {
                id: "01JBR55WAITLIVE".into(),
                ephemeral: true,
                frozen_model: None,
                request,
            },
        )
        .unwrap();
        let handle = crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests();

        let waiter = shim_wait_settled(&handle, &db, &row.id);
        tokio::pin!(waiter);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                tokio::select! {
                    result = &mut waiter => {
                        panic!("waiter settled before completion publish: {result:?}");
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {
                        if handle.events_for_tests().contains_for_tests(&row.id) {
                            break;
                        }
                    }
                }
            }
        })
        .await
        .expect("waiter should subscribe before completion publish");
        chain_jobs::update_job_state(&db, &row.id, ChainJobState::Completed, None, 7_000).unwrap();
        handle.publish_settled_state(&row.id, ChainJobState::Completed, None);

        let (state, error) = tokio::time::timeout(std::time::Duration::from_secs(2), waiter)
            .await
            .expect("settled event should wake waiter")
            .unwrap();
        assert_eq!(state, ChainJobState::Completed);
        assert_eq!(error, None);
    }

    #[tokio::test]
    async fn shim_wait_settled_rereads_completed_state_after_closed_receiver() {
        let db = MetadataDb::open_in_memory().unwrap();
        let jobs_root = tempfile::tempdir().unwrap();
        let request = req(OutputFormat::Mp4).normalise().unwrap();
        let row = crate::chain_job_runner::create_job_with_params(
            &db,
            jobs_root.path(),
            crate::chain_job_runner::CreateJobParams {
                id: "01JBR55WAITCLOSED".into(),
                ephemeral: true,
                frozen_model: None,
                request,
            },
        )
        .unwrap();
        chain_jobs::update_job_state(&db, &row.id, ChainJobState::Completed, None, 7_000).unwrap();
        let handle = crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests();

        let (state, error) = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            shim_wait_settled(&handle, &db, &row.id),
        )
        .await
        .expect("closed receiver should fall back to DB read")
        .unwrap();
        assert_eq!(state, ChainJobState::Completed);
        assert_eq!(error, None);
    }

    #[tokio::test]
    async fn generate_chain_stream_rejects_invalid_completion_payload_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-mold-sse-payload", "compact".parse().unwrap());

        let result = generate_chain_stream(
            State(AppState::for_tests()),
            headers,
            Json(req(OutputFormat::Mp4)),
        )
        .await;
        let error = match result {
            Ok(_) => panic!("invalid completion payload header must be rejected"),
            Err(error) => error,
        };

        assert_eq!(error.code, "VALIDATION_ERROR");
        assert_eq!(
            error.error,
            "X-Mold-SSE-Payload must be 'metadata-only' when provided"
        );
        assert_eq!(
            error.into_response().status(),
            axum::http::StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[tokio::test]
    async fn metadata_only_chain_stream_rejects_disabled_gallery_before_job_creation() {
        let _output_guard = OutputDirEnvGuard::disable();
        let db = MetadataDb::open_in_memory().unwrap();
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        let mut state = AppState::for_tests();
        state.config = Arc::new(tokio::sync::RwLock::new(mold_core::Config {
            output_dir: Some(String::new()),
            ..Default::default()
        }));
        state.metadata_db = Arc::new(Some(db));
        state.chain_jobs = Some(handle);
        let mut headers = HeaderMap::new();
        headers.insert("x-mold-sse-payload", "metadata-only".parse().unwrap());

        let result =
            generate_chain_stream(State(state.clone()), headers, Json(req(OutputFormat::Mp4)))
                .await;
        let error = match result {
            Ok(_) => panic!("metadata-only completion requires gallery output"),
            Err(error) => error,
        };

        assert_eq!(error.code, "VALIDATION_ERROR");
        assert_eq!(
            error.error,
            "metadata-only SSE completions require server gallery output to be enabled"
        );
        assert!(
            chain_jobs::list_jobs(state.metadata_db.as_ref().as_ref().unwrap())
                .unwrap()
                .is_empty(),
            "rejection must happen before the shim creates or schedules a job"
        );
    }

    #[tokio::test]
    async fn generate_chain_stream_task_emits_snapshot_progress_then_complete() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("mold-home");
        let _home_guard = MoldHomeGuard::set(&home);
        let db = MetadataDb::open_in_memory().unwrap();
        let output_dir = dir.path().join("gallery");
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        let state = state_with_db_and_handle(db, handle.clone(), &output_dir);

        let sse = generate_chain_stream(
            State(state.clone()),
            HeaderMap::new(),
            Json(req(OutputFormat::Mp4)),
        )
        .await
        .unwrap();
        let body_task = tokio::spawn(async move {
            let response = sse.into_response();
            let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            String::from_utf8(bytes.to_vec()).unwrap()
        });

        let db = state.metadata_db.as_ref().as_ref().unwrap();
        let row = wait_for_single_chain_job(db).await;
        wait_for_bus_subscription(&handle, &row.id).await;
        handle
            .events_for_tests()
            .publish(&row.id, ChainJobEvent::StageStart { stage_idx: 0 });
        handle.events_for_tests().publish(
            &row.id,
            ChainJobEvent::DenoiseStep {
                stage_idx: 0,
                step: 1,
                total: 2,
            },
        );
        handle.events_for_tests().publish(
            &row.id,
            ChainJobEvent::StageDone {
                stage_idx: 0,
                frames_emitted: 9,
                has_preview: true,
                has_media: true,
                cache_ready: true,
            },
        );
        handle
            .events_for_tests()
            .publish(&row.id, ChainJobEvent::Finalizing { total_frames: 26 });

        let request = ChainJobManifest::read_from_dir(&row.job_dir)
            .unwrap()
            .request()
            .unwrap();
        let final_bytes = audio_muxed_mp4_fixture();
        mark_ephemeral_job_completed(db, &row, &request, &[11, 22], &final_bytes);
        handle.publish_settled_state(&row.id, ChainJobState::Completed, None);

        let body = tokio::time::timeout(std::time::Duration::from_secs(5), body_task)
            .await
            .expect("stream should close after complete")
            .unwrap();
        let events = parse_sse_json_events(&body);
        assert_eq!(
            events
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["progress", "progress", "progress", "progress", "progress", "complete"],
            "unexpected SSE body: {body}"
        );

        let progress = events
            .iter()
            .filter(|(name, _)| name == "progress")
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        assert_eq!(progress[0]["type"], "chain_start");
        assert_eq!(progress[0]["job_id"], row.id);
        assert_eq!(progress[1]["type"], "stage_start");
        assert_eq!(progress[1]["job_id"], row.id);
        assert_eq!(progress[2]["type"], "denoise_step");
        assert_eq!(progress[2]["job_id"], row.id);
        assert_eq!(progress[3]["type"], "stage_done");
        assert_eq!(progress[3]["job_id"], row.id);
        assert!(progress[3].get("has_preview").is_none());
        assert_eq!(progress[4]["type"], "stitching");
        assert_eq!(progress[4]["job_id"], row.id);

        let complete = &events.last().unwrap().1;
        assert_eq!(complete["format"], "mp4");
        assert_eq!(complete["stage_count"], 2);
        assert_eq!(complete["generation_time_ms"], 33);
        assert!(
            complete.get("job_id").is_none(),
            "complete keeps legacy shape"
        );
        assert!(complete["video"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert!(complete["thumbnail"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert!(complete["gif_preview"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert!(complete["filename"]
            .as_str()
            .is_some_and(|value| value.ends_with(".mp4")));
        assert!(complete["metadata"].is_object());
    }

    #[tokio::test]
    async fn metadata_only_chain_stream_returns_saved_filename_and_exact_metadata_without_media() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("mold-home");
        let _home_guard = MoldHomeGuard::set(&home);
        let db = MetadataDb::open_in_memory().unwrap();
        let output_dir = dir.path().join("gallery");
        let handle = Arc::new(crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests());
        let state = state_with_db_and_handle(db, handle.clone(), &output_dir);
        let mut headers = HeaderMap::new();
        headers.insert("x-mold-sse-payload", "metadata-only".parse().unwrap());

        let sse =
            generate_chain_stream(State(state.clone()), headers, Json(req(OutputFormat::Mp4)))
                .await
                .unwrap();
        let body_task = tokio::spawn(async move {
            let response = sse.into_response();
            let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            String::from_utf8(bytes.to_vec()).unwrap()
        });

        let db = state.metadata_db.as_ref().as_ref().unwrap();
        let row = wait_for_single_chain_job(db).await;
        wait_for_bus_subscription(&handle, &row.id).await;
        let request = ChainJobManifest::read_from_dir(&row.job_dir)
            .unwrap()
            .request()
            .unwrap();
        let final_bytes = audio_muxed_mp4_fixture();
        mark_ephemeral_job_completed(db, &row, &request, &[11, 22], &final_bytes);
        handle.publish_settled_state(&row.id, ChainJobState::Completed, None);

        let body = tokio::time::timeout(std::time::Duration::from_secs(5), body_task)
            .await
            .expect("stream should close after complete")
            .unwrap();
        let events = parse_sse_json_events(&body);
        let complete = &events
            .iter()
            .find(|(name, _)| name == "complete")
            .expect("metadata-only stream should complete")
            .1;

        assert_eq!(complete["video"], "");
        assert!(complete.get("thumbnail").is_none());
        assert!(complete.get("gif_preview").is_none());
        let filename = complete["filename"]
            .as_str()
            .expect("metadata-only completion should name saved gallery video");
        assert!(output_dir.join(filename).is_file());
        let rows = db.list(Some(&output_dir)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].filename, filename);
        assert_eq!(
            complete["metadata"],
            serde_json::to_value(&rows[0].metadata).unwrap(),
            "completion metadata must exactly match the gallery record"
        );
    }

    #[tokio::test]
    async fn generate_chain_handler_returns_502_with_chain_failure_body() {
        let dir = tempfile::tempdir().unwrap();
        let _home_guard = MoldHomeGuard::set(dir.path());
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let deps = crate::chain_job_runner::RunnerDeps {
            db: db.clone(),
            jobs_root: dir.path().join("jobs"),
            executor: Arc::new(HandlerFailingExecutor),
            queue_probe: Arc::new(NoQueuedSmallJobs),
            events: Arc::new(crate::chain_job_runner::JobEventBus::new()),
            cancel: Arc::new(crate::chain_job_runner::CancelRegistry::new()),
            job_locks: Arc::new(crate::chain_job_runner::JobMutationLocks::new()),
            claims: Arc::new(crate::chain_job_runner::EphemeralClaims::default()),
            output_dir: None,
            server_events: None,
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            dispatch_mode: crate::dispatch_mode::DispatchMode::Legacy,
            pause: None,
        };
        let handle = crate::chain_job_runner::spawn_runner(deps);
        let mut state = AppState::for_tests();
        let transformer = dir.path().join("transformer.safetensors");
        let vae = dir.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let mut config = mold_core::Config::default();
        config.models.insert(
            "ltx-2-19b-distilled:mock".into(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                transformer: Some(transformer.to_string_lossy().into_owned()),
                vae: Some(vae.to_string_lossy().into_owned()),
                ..Default::default()
            },
        );
        state.config = Arc::new(tokio::sync::RwLock::new(config));
        state.metadata_db = db;
        state.chain_jobs = Some(Arc::new(handle));

        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            generate_chain(State(state), Json(req(OutputFormat::Mp4))),
        )
        .await
        .expect("generate_chain should settle");
        let (parts, body) = resp.into_response().into_parts();

        assert_eq!(parts.status, axum::http::StatusCode::BAD_GATEWAY);
        let bytes = to_bytes(body, usize::MAX).await.unwrap();
        let failure: ChainFailure =
            serde_json::from_slice(&bytes).expect("502 body must be ChainFailure JSON");
        assert_eq!(failure.failed_stage_idx, 1);
        assert_eq!(failure.elapsed_stages, 1);
        assert_eq!(failure.elapsed_ms, 10);
        assert!(
            failure
                .stage_error
                .contains("simulated chain failure at stage one"),
            "stage_error must carry renderer message, got {}",
            failure.stage_error
        );
    }

    #[test]
    fn legacy_progress_mapping_injects_job_id_on_every_progress_event() {
        let events = vec![
            ChainJobEvent::Snapshot { job: detail() },
            ChainJobEvent::StageStart { stage_idx: 1 },
            ChainJobEvent::DenoiseStep {
                stage_idx: 1,
                step: 2,
                total: 4,
            },
            ChainJobEvent::StageDone {
                stage_idx: 1,
                frames_emitted: 11,
                has_preview: true,
                has_media: true,
                cache_ready: true,
            },
            ChainJobEvent::Finalizing { total_frames: 20 },
        ];

        for event in events {
            let value = map_job_event_to_legacy(&event, "job-1").expect("mapped progress event");
            assert_eq!(value.get("job_id").and_then(|v| v.as_str()), Some("job-1"));
            assert!(
                value.get("type").is_some(),
                "legacy progress type preserved: {value:?}"
            );
        }
    }

    #[test]
    fn legacy_progress_mapping_keeps_event_names_and_shapes() {
        let snapshot = map_job_event_to_legacy(&ChainJobEvent::Snapshot { job: detail() }, "job-2")
            .expect("snapshot maps");
        assert_eq!(
            snapshot.get("type").and_then(|v| v.as_str()),
            Some("chain_start")
        );
        assert_eq!(
            snapshot
                .get("estimated_total_frames")
                .and_then(|v| v.as_u64()),
            Some(26)
        );

        let stage_done = map_job_event_to_legacy(
            &ChainJobEvent::StageDone {
                stage_idx: 3,
                frames_emitted: 15,
                has_preview: true,
                has_media: true,
                cache_ready: true,
            },
            "job-2",
        )
        .expect("stage_done maps");
        assert_eq!(
            stage_done.get("type").and_then(|v| v.as_str()),
            Some("stage_done")
        );
        assert_eq!(
            stage_done.get("stage_idx").and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            stage_done.get("frames_emitted").and_then(|v| v.as_u64()),
            Some(15)
        );
        assert!(
            stage_done.get("has_preview").is_none(),
            "legacy event remains additive only"
        );
    }

    #[test]
    fn legacy_progress_mapping_drops_non_progress_events() {
        for event in [
            ChainJobEvent::Yielded {
                pending_small_jobs: 1,
            },
            ChainJobEvent::Finalized {
                output: "final/output-1.mp4".into(),
                take: 1,
            },
            ChainJobEvent::StateChanged {
                state: ChainJobState::Running,
                error: None,
            },
        ] {
            assert!(map_job_event_to_legacy(&event, "job-3").is_none());
        }
    }
}
