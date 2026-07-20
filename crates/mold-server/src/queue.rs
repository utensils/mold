use std::collections::VecDeque;
use std::sync::Arc;

use base64::Engine as _;
use mold_core::{
    ImageData, OutputFormat, OutputMetadata, SseCompleteEvent, SseErrorEvent, SseProgressEvent,
};
use mold_db::{MetadataDb, RecordSource};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::Notify;

use crate::gpu_pool::GpuJob;
use crate::model_manager;
use crate::state::{
    ActiveGenerationSnapshot, AppState, GenerationJob, GenerationJobResult, SseCompletionPayload,
    SseMessage,
};

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

/// Strips backtrace frames from candle error messages.
///
/// Renders the full anyhow cause chain (`{:#}`) so wrappers like
/// `with_context("mmap single-file checkpoint at …")` carry their root cause
/// through to the wire — otherwise users see the outer wrapper only.
pub(crate) fn clean_error_message(e: &anyhow::Error) -> String {
    let full = format!("{e:#}");
    let mut lines: Vec<&str> = Vec::new();
    for line in full.lines() {
        let trimmed = line.trim_start();
        if (trimmed.starts_with("0:") || trimmed.starts_with("1:"))
            && trimmed.len() > 3
            && trimmed
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit())
        {
            break;
        }
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
        format!("{}", e.root_cause())
    } else {
        msg
    }
}

fn set_active_generation(state: &AppState, model: &str, prompt: &str) {
    let prompt_sha256 = format!("{:x}", Sha256::digest(prompt.as_bytes()));
    let started_at_unix_ms = mold_core::time::now_epoch_ms_u64();

    let mut active = state
        .active_generation
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *active = Some(ActiveGenerationSnapshot {
        model: model.to_string(),
        prompt_sha256,
        started_at_unix_ms,
        started_at: Instant::now(),
    });
}

fn clear_active_generation(state: &AppState) {
    let mut active = state
        .active_generation
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *active = None;
}

/// Test-facing single-image wrapper around the shared output persistence path.
///
/// Errors writing to disk are logged and skipped. DB errors are also logged
/// but do not fail the save — the file is the source of truth.
///
/// Shared between the legacy single-GPU `process_job` (this file) and the
/// per-GPU worker (`gpu_worker.rs`). Keep these on one helper so the DB
/// upsert can never silently regress on one path while the other keeps
/// working.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn save_image_to_dir(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
    metadata: Option<&OutputMetadata>,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
) {
    save_image_to_dir_with_suffix(
        dir,
        img,
        model,
        batch_size,
        None,
        metadata,
        generation_time_ms,
        db,
        events,
    );
}

#[allow(clippy::too_many_arguments)]
fn save_image_to_dir_with_suffix(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
    suffix: Option<&str>,
    metadata: Option<&OutputMetadata>,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
) -> Option<String> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return None;
    }
    let timestamp_ms = mold_core::time::now_epoch_ms_u64();
    let ext = img.format.to_string();
    let mut filename =
        mold_core::default_output_filename(model, timestamp_ms, &ext, batch_size, img.index);
    if let Some(suffix) = suffix {
        filename = format!(
            "{}-{suffix}.{ext}",
            filename.trim_end_matches(&format!(".{ext}"))
        );
    }
    let path = dir.join(&filename);
    match std::fs::write(&path, &img.data) {
        Ok(()) => tracing::info!("saved image to {}", path.display()),
        Err(e) => {
            tracing::warn!("failed to save image to {}: {e}", path.display());
            return None;
        }
    }
    let mut image_row = None;
    if let (Some(db), Some(meta)) = (db, metadata) {
        image_row = mold_db::persist::record_saved_output_returning(
            db,
            dir,
            &filename,
            &path,
            &mold_db::persist::OutputRecordParams {
                format: img.format,
                metadata: meta,
                source: RecordSource::Server,
                generation_time_ms,
                backend: Some(mold_inference::compiled_backend_label()),
            },
        )
        .map(|rec| Box::new(rec.to_gallery_image()));
    }
    // Emit even without a DB row — `image: None` tells clients to refetch
    // `/api/gallery` instead of inserting in place.
    if let Some(events) = events {
        events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.clone(),
            image: image_row,
        });
    }
    Some(filename)
}

/// Gallery filenames a generation's outputs were saved under, threaded into
/// the SSE complete event so mirroring clients keep the same identity.
#[derive(Debug, Default, Clone)]
pub(crate) struct SavedOutputNames {
    /// The payload the complete event carries (upscaled when upscaling ran).
    pub output: Option<String>,
    /// The pre-upscale original, when one was saved separately.
    pub original: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn save_generated_image_outputs(
    dir: &std::path::Path,
    original: Option<&ImageData>,
    output: &ImageData,
    model: &str,
    batch_size: u32,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
) -> SavedOutputNames {
    let mut names = SavedOutputNames::default();
    if let Some(original) = original {
        let mut original_metadata = metadata.clone();
        apply_output_dimensions_to_metadata(&mut original_metadata, original);
        names.original = save_image_to_dir_with_suffix(
            dir,
            original,
            model,
            batch_size,
            Some("original"),
            Some(&original_metadata),
            generation_time_ms,
            db,
            events,
        );
    }
    let mut output_metadata = metadata.clone();
    apply_output_dimensions_to_metadata(&mut output_metadata, output);
    names.output = save_image_to_dir_with_suffix(
        dir,
        output,
        model,
        batch_size,
        original.map(|_| "upscaled"),
        Some(&output_metadata),
        generation_time_ms,
        db,
        events,
    );
    names
}

/// Save a video file to disk and (best-effort) record its metadata row.
/// Mirrors `save_image_to_dir` for the video-output path. See that helper
/// for the multi-path-callers note.
///
/// When `gif_preview` is non-empty, also persists
/// `$MOLD_HOME/cache/previews/<filename>.preview.gif`. The gallery preview
/// endpoint (`GET /api/gallery/preview/:filename`) streams from that path
/// so remote TUI clients can animate the detail pane without re-fetching
/// the full MP4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_video_to_dir(
    dir: &std::path::Path,
    bytes: &[u8],
    gif_preview: &[u8],
    format: OutputFormat,
    model: &str,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
) -> Option<String> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return None;
    }
    let ts = mold_core::time::now_epoch_ms_u64();
    let ext = format.extension();
    let filename = mold_core::default_output_filename(model, ts, ext, 1, 0);
    let path = dir.join(&filename);
    if let Err(e) = std::fs::write(&path, bytes) {
        tracing::error!("failed to save video to {}: {e}", path.display());
        return None;
    }
    if !gif_preview.is_empty() {
        save_video_preview_gif(&filename, gif_preview);
    }
    let mut image_row = None;
    if let Some(db) = db {
        image_row = mold_db::persist::record_saved_output_returning(
            db,
            dir,
            &filename,
            &path,
            &mold_db::persist::OutputRecordParams {
                format,
                metadata,
                source: RecordSource::Server,
                generation_time_ms,
                backend: Some(mold_inference::compiled_backend_label()),
            },
        )
        .map(|rec| Box::new(rec.to_gallery_image()));
    }
    if let Some(events) = events {
        events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.clone(),
            image: image_row,
        });
    }
    Some(filename)
}

fn requested_post_upscale_model(req: &mold_core::GenerateRequest) -> Option<&str> {
    req.upscale_model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
}

fn post_upscale_model_to_pull(
    config: &mold_core::Config,
    req: &mold_core::GenerateRequest,
) -> Result<Option<String>, String> {
    let Some(requested) = requested_post_upscale_model(req) else {
        return Ok(None);
    };
    let model_name = mold_core::manifest::resolve_model_name(requested);
    if model_manager::configured_upscaler_weights_exist(config, &model_name) {
        return Ok(None);
    }
    if mold_core::manifest::find_manifest(&model_name).is_none() {
        return Err(format!("unknown upscaler model '{model_name}'"));
    }
    Ok(Some(model_name))
}

async fn ensure_post_upscale_model_downloaded(
    state: &AppState,
    req: &mold_core::GenerateRequest,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<(), String> {
    let model_to_pull = {
        let config = state.config.read().await;
        post_upscale_model_to_pull(&config, req)?
    };
    let Some(model_name) = model_to_pull else {
        return Ok(());
    };

    if let Some(tx) = progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::StageStart {
            name: format!("Downloading upscaler {model_name}"),
        }));
    }
    let progress = progress_tx.cloned().map(|tx| {
        Arc::new(move |event: mold_core::download::DownloadProgressEvent| {
            let event = match event {
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
            let _ = tx.send(SseMessage::Progress(event));
        }) as model_manager::DownloadProgressCallback
    });
    model_manager::pull_model(state, &model_name, progress)
        .await
        .map_err(|e| format!("failed to pull upscaler model: {}", e.error))?;
    if let Some(tx) = progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
            model: model_name,
        }));
    }
    Ok(())
}

pub(crate) fn apply_output_dimensions_to_metadata(metadata: &mut OutputMetadata, img: &ImageData) {
    metadata.apply_output_dimensions(img.width, img.height);
}

pub(crate) fn apply_upscale_response_to_image_generation(
    req: &mold_core::GenerateRequest,
    response: &mut mold_core::GenerateResponse,
    original: ImageData,
    upscaled: mold_core::UpscaleResponse,
) -> anyhow::Result<ImageData> {
    if response.video.is_some() || requested_post_upscale_model(req).is_none() {
        return Ok(original);
    }
    if upscaled.image.data.is_empty() {
        anyhow::bail!("upscaler returned an empty image");
    }
    response.generation_time_ms = response
        .generation_time_ms
        .saturating_add(upscaled.upscale_time_ms);
    Ok(ImageData {
        index: original.index,
        ..upscaled.image
    })
}

pub(crate) fn settle_post_generation_upscale(
    original: ImageData,
    result: Result<ImageData, String>,
) -> (ImageData, Option<ImageData>, Option<String>) {
    match result {
        Ok(upscaled) => (upscaled, Some(original), None),
        Err(error) => (original, None, Some(error)),
    }
}

async fn upscale_generated_image_on_single_worker(
    state: &AppState,
    req: &mold_core::GenerateRequest,
    seed_used: u64,
    img: ImageData,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<ImageData, String> {
    let Some(upscale_model) = requested_post_upscale_model(req).map(str::to_string) else {
        return Ok(img);
    };
    let model_name = mold_core::manifest::resolve_model_name(&upscale_model);
    if let Some(tx) = progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::StageStart {
            name: format!("Loading upscaler {model_name}"),
        }));
    }

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
            return Err(format!("unknown upscaler model '{model_name}'"));
        }
        model_manager::pull_model(state, &model_name, None)
            .await
            .map_err(|e| format!("failed to pull upscaler model: {}", e.error))?;
    }

    let weights_path = {
        let config = state.config.read().await;
        config
            .models
            .get(&model_name)
            .and_then(|c| c.transformer.as_ref())
            .map(std::path::PathBuf::from)
    }
    .ok_or_else(|| format!("upscaler model '{model_name}' not configured after pull"))?;

    let upscale_req = mold_core::UpscaleRequest {
        model: model_name.clone(),
        image: img.data.clone(),
        output_format: img.format,
        tile_size: None,
        metadata: Some(OutputMetadata::from_generate_request(
            req,
            seed_used,
            None,
            mold_core::build_info::version_string(),
        )),
    };
    let upscaler_cache = state.upscaler_cache.clone();
    let progress_tx_for_blocking = progress_tx.cloned();
    let upscaled =
        tokio::task::spawn_blocking(move || -> anyhow::Result<mold_core::UpscaleResponse> {
            let mut cache = upscaler_cache.lock().unwrap_or_else(|e| e.into_inner());
            let needs_new = cache.as_ref().is_none_or(|e| e.model_name() != model_name);
            if needs_new {
                let new_engine = mold_inference::create_upscale_engine(
                    model_name.clone(),
                    weights_path,
                    mold_inference::LoadStrategy::Eager,
                    0,
                )?;
                *cache = Some(new_engine);
            }
            let engine = cache.as_mut().unwrap();
            if let Some(tx) = progress_tx_for_blocking {
                engine.set_on_progress(Box::new(move |event| {
                    let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
                }));
            }
            let result = engine.upscale(&upscale_req);
            engine.clear_on_progress();
            result
        })
        .await
        .map_err(|e| format!("upscale task failed: {e}"))?
        .map_err(|e| format!("upscale failed: {e}"))?;

    let mut response = mold_core::GenerateResponse {
        images: vec![],
        video: None,
        generation_time_ms: 0,
        model: req.model.clone(),
        seed_used: req.seed.unwrap_or(0),
        gpu: None,
    };
    apply_upscale_response_to_image_generation(req, &mut response, img, upscaled)
        .map_err(|e| format!("upscale failed: {e}"))
}

/// Persist a video's `.preview.gif` sidecar to the server's preview cache
/// (`$MOLD_HOME/cache/previews/<filename>.preview.gif`). Best-effort —
/// warnings log and return so a failure here never fails the save path.
///
/// Shared with the multi-GPU worker path (`gpu_worker::process_job`) so
/// video outputs land a preview regardless of which save flow wrote the
/// MP4; otherwise `/api/gallery/preview/:filename` would 404 whenever the
/// server is running with GPU workers enabled.
pub(crate) fn save_video_preview_gif(filename: &str, gif_bytes: &[u8]) {
    let preview_dir = mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("previews");
    save_video_preview_gif_to(&preview_dir, filename, gif_bytes);
}

/// Testable inner of [`save_video_preview_gif`] that accepts an explicit
/// preview directory (lets unit tests exercise the write path without
/// racing on the `MOLD_HOME` env var).
fn save_video_preview_gif_to(preview_dir: &std::path::Path, filename: &str, gif_bytes: &[u8]) {
    if let Err(e) = std::fs::create_dir_all(preview_dir) {
        tracing::warn!(
            "failed to create preview cache dir {}: {e}",
            preview_dir.display()
        );
        return;
    }
    let preview_path = preview_dir.join(mold_core::media_paths::preview_gif_filename(filename));
    if let Err(e) = std::fs::write(&preview_path, gif_bytes) {
        tracing::warn!(
            "failed to write preview gif {}: {e}",
            preview_path.display()
        );
    }
}

/// Build the SSE `complete` wire event from a finished generation response.
///
/// Video responses encode the actual video bytes (MP4/GIF/APNG/WebP) as the
/// payload and populate every `video_*` metadata field; image responses
/// encode the image bytes with the video fields cleared. `img` is the
/// `ImageData` chosen by the caller — either the first generated image or an
/// `ImageData` synthesized from the video thumbnail (the single-primary-image
/// shape that the internal `GenerationJobResult` still expects).
///
/// Shared between the single-GPU path (`process_job` in this file) and the
/// multi-GPU path (`gpu_worker::process_job`) so the two can never drift on
/// which `video_*` fields are populated. Before this helper existed the
/// multi-GPU worker always encoded the thumbnail PNG as the payload and
/// hard-coded every `video_*` field to `None`, which silently degraded every
/// LTX-Video / LTX-2 generation into an image response on hosts with at
/// least one GPU worker.
pub(crate) fn build_sse_complete_event(
    response: &mold_core::GenerateResponse,
    img: &mold_core::ImageData,
    original: Option<&mold_core::ImageData>,
    metadata: Option<&OutputMetadata>,
    saved: &SavedOutputNames,
    payload: SseCompletionPayload,
) -> SseCompleteEvent {
    let b64 = base64::engine::general_purpose::STANDARD;
    let include_media = payload == SseCompletionPayload::Full;
    // Mirror exactly what the save path records: video metadata is used
    // as-built, image metadata gets the payload's actual dimensions.
    let event_metadata = metadata.map(|meta| {
        let mut meta = meta.clone();
        if response.video.is_none() {
            apply_output_dimensions_to_metadata(&mut meta, img);
        }
        Box::new(meta)
    });
    if let Some(ref video) = response.video {
        SseCompleteEvent {
            image: if include_media {
                b64.encode(&video.data)
            } else {
                String::new()
            },
            format: video.format,
            width: video.width,
            height: video.height,
            original_image: None,
            original_width: None,
            original_height: None,
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: Some(video.frames),
            video_fps: Some(video.fps),
            video_thumbnail: include_media.then(|| b64.encode(&video.thumbnail)),
            video_gif_preview: if !include_media || video.gif_preview.is_empty() {
                None
            } else {
                Some(b64.encode(&video.gif_preview))
            },
            video_has_audio: video.has_audio,
            video_duration_ms: video.duration_ms,
            video_audio_sample_rate: video.audio_sample_rate,
            video_audio_channels: video.audio_channels,
            gpu: response.gpu,
            filename: saved.output.clone(),
            original_filename: None,
            metadata: event_metadata,
        }
    } else {
        SseCompleteEvent {
            image: if include_media {
                b64.encode(&img.data)
            } else {
                String::new()
            },
            format: img.format,
            width: img.width,
            height: img.height,
            original_image: include_media
                .then(|| original.map(|image| b64.encode(&image.data)))
                .flatten(),
            original_width: original.map(|image| image.width),
            original_height: original.map(|image| image.height),
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: None,
            video_fps: None,
            video_thumbnail: None,
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
            gpu: response.gpu,
            filename: saved.output.clone(),
            original_filename: saved.original.clone(),
            metadata: event_metadata,
        }
    }
}

pub(crate) fn build_sse_completion_message(
    response: &mold_core::GenerateResponse,
    img: &mold_core::ImageData,
    original: Option<&mold_core::ImageData>,
    metadata: Option<&OutputMetadata>,
    saved: &SavedOutputNames,
    payload: SseCompletionPayload,
) -> SseMessage {
    if payload == SseCompletionPayload::MetadataOnly && saved.output.is_none() {
        return SseMessage::Error(SseErrorEvent {
            message: "generation completed but the output could not be saved for streaming"
                .to_string(),
        });
    }
    SseMessage::Complete(Box::new(build_sse_complete_event(
        response, img, original, metadata, saved, payload,
    )))
}

/// Dispatch gate shared through `AppState`, toggled by `POST /api/queue/pause`
/// and `POST /api/queue/resume`. When paused the dispatch loops stop pulling
/// *new* jobs off the channel; the job already running on a worker finishes
/// untouched. Cheap to poll (a single relaxed-ish atomic) so it can sit at the
/// top of every loop iteration.
pub struct QueuePause {
    paused: AtomicBool,
    /// Wakes every gated dispatch loop on resume. Resume calls
    /// `notify_waiters()` so *all* loops (single- and multi-GPU) proceed, not
    /// just one.
    notify: Notify,
}

impl QueuePause {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            paused: AtomicBool::new(false),
            notify: Notify::new(),
        })
    }

    /// Pause new-job dispatch. Returns `true` iff this call flipped the state
    /// (was running); idempotent repeat pauses return `false` so the route can
    /// suppress a duplicate `queue_paused` event.
    pub fn pause(&self) -> bool {
        !self.paused.swap(true, Ordering::SeqCst)
    }

    /// Resume dispatch and wake every gated loop. Returns `true` iff this call
    /// flipped the state (was paused).
    pub fn resume(&self) -> bool {
        let was_paused = self.paused.swap(false, Ordering::SeqCst);
        if was_paused {
            self.notify.notify_waiters();
        }
        was_paused
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Park the caller while paused, returning as soon as dispatch is resumed
    /// (immediately when not paused). Registers the wakeup *before* the second
    /// flag check so a concurrent `resume()`'s `notify_waiters()` can't slip
    /// between the check and the await — the classic lost-wakeup race — and
    /// re-loops in case of a spurious wake.
    pub async fn wait_if_paused(&self) {
        while self.paused.load(Ordering::SeqCst) {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.paused.load(Ordering::SeqCst) {
                break;
            }
            notified.await;
        }
    }
}

/// Runs the generation queue worker loop. Processes one job at a time (FIFO),
/// but uses a small bounded lookahead buffer to prefer jobs whose model is
/// already loaded — minimizing model swaps when the queue interleaves models.
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_worker(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    tracing::debug!("generation queue worker started");
    let buffer_size = resolve_lookahead_buffer();
    let max_deferrals = resolve_max_deferrals();
    let mut buffer: VecDeque<BufferedJob> = VecDeque::with_capacity(buffer_size);

    loop {
        // Hold new-job dispatch while paused. A job already running finishes
        // untouched — this only gates the pull of the *next* job.
        state.queue_pause.wait_if_paused().await;
        if buffer.is_empty() {
            match job_rx.recv().await {
                Some(j) => buffer.push_back(BufferedJob::new(j)),
                None => break,
            }
        }
        // Top up the buffer without blocking — drain the channel up to capacity.
        top_up_buffer(&mut buffer, &mut job_rx, buffer_size);
        // Re-check after the recv: a pause that landed while this loop was
        // parked waiting for work must hold the job that woke it, not leak
        // it into dispatch.
        state.queue_pause.wait_if_paused().await;

        let loaded = single_gpu_loaded_models(&state).await;
        let job = pick_next_job(&mut buffer, &loaded, max_deferrals);
        let job_id = job.id.clone();

        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
        process_job(&state, job).await;
        state.queue.decrement();
        // Drop the registry entry on every terminal path — the worker
        // here doesn't own a drop guard, so we do it inline alongside
        // the queue counter decrement.
        state.job_registry.remove(&job_id);
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
    }
    tracing::info!("generation queue worker shutting down");
}

async fn single_gpu_loaded_models(state: &AppState) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    let cache = state.model_cache.lock().await;
    if let Some(name) = cache.active_model() {
        set.insert(name.to_string());
    }
    set
}

/// Build the set of "currently loaded somewhere" model names across every
/// worker in the multi-GPU pool. A worker counts the model as loaded if
/// either it's in the worker's cache as Gpu-resident OR it's the worker's
/// `active_generation` (covering the take-and-restore window where the
/// cache entry briefly disappears).
fn multi_gpu_loaded_models(state: &AppState) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    for worker in &state.gpu_pool.workers {
        if let Ok(active_gen) = worker.active_generation.read() {
            if let Some(g) = active_gen.as_ref() {
                set.insert(g.model.clone());
            }
        }
        if let Ok(cache) = worker.model_cache.lock() {
            if let Some(name) = cache.active_model() {
                set.insert(name.to_string());
            }
        }
    }
    set
}

/// In-flight wrapper that tracks how many times the picker has skipped this
/// job. Once the count exceeds `max_deferrals`, the picker force-dispatches
/// it to bound starvation.
pub(crate) struct BufferedJob {
    pub(crate) job: GenerationJob,
    pub(crate) deferred: usize,
}

impl BufferedJob {
    fn new(job: GenerationJob) -> Self {
        Self { job, deferred: 0 }
    }
}

/// Drain the receive channel into the lookahead buffer, capped at
/// `buffer_size`. Returns when the buffer is full or the channel has no
/// immediately-available jobs (the receiver is unchanged on `Empty`). Pure
/// helper extracted so tests can lock in the cap as a load-bearing invariant
/// without spinning up the full async dispatcher.
pub(crate) fn top_up_buffer(
    buffer: &mut VecDeque<BufferedJob>,
    job_rx: &mut tokio::sync::mpsc::Receiver<GenerationJob>,
    buffer_size: usize,
) {
    while buffer.len() < buffer_size {
        match job_rx.try_recv() {
            Ok(j) => buffer.push_back(BufferedJob::new(j)),
            Err(_) => break,
        }
    }
}

/// Pure picker for the lookahead buffer. Selects the buffered job whose
/// model is already loaded somewhere in `loaded`; ties broken by arrival
/// order (front of the deque wins). The head's `deferred` count bounds
/// starvation: if the head has been skipped `max_deferrals` times, it wins
/// regardless of `loaded` membership.
///
/// The returned job is removed from the buffer; remaining buffered jobs that
/// were skipped have their `deferred` count incremented. Increments
/// `mold_queue_reorders_total` whenever a non-head job is picked.
pub(crate) fn pick_next_job(
    buffer: &mut VecDeque<BufferedJob>,
    loaded: &std::collections::HashSet<String>,
    max_deferrals: usize,
) -> GenerationJob {
    debug_assert!(
        !buffer.is_empty(),
        "pick_next_job requires non-empty buffer"
    );

    // Force-dispatch the head if it's hit the starvation budget.
    if let Some(head) = buffer.pop_front_if(|head| head.deferred >= max_deferrals) {
        return head.job;
    }

    // Find the front-most buffered job whose model is already loaded.
    let pick_idx = buffer
        .iter()
        .position(|b| loaded.contains(&b.job.request.model))
        .unwrap_or(0);

    if pick_idx > 0 {
        for (i, b) in buffer.iter_mut().enumerate() {
            if i < pick_idx {
                b.deferred += 1;
            }
        }
        let model = buffer[pick_idx].job.request.model.clone();
        tracing::debug!(
            picked_model = %model,
            head_model = %buffer.front().map(|b| b.job.request.model.as_str()).unwrap_or(""),
            picked_index = pick_idx,
            "queue reorder picked non-head job"
        );
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_reorder();
    }

    buffer.remove(pick_idx).expect("pick_idx in range").job
}

pub(crate) const DEFAULT_LOOKAHEAD_BUFFER: usize = 8;
pub(crate) const DEFAULT_MAX_DEFERRALS: usize = 3;
pub(crate) const LOOKAHEAD_BUFFER_ENV: &str = "MOLD_QUEUE_LOOKAHEAD_BUFFER";
pub(crate) const MAX_DEFERRALS_ENV: &str = "MOLD_QUEUE_MAX_DEFERRALS";
const LOOKAHEAD_BUFFER_LOWER: usize = 1;
const LOOKAHEAD_BUFFER_UPPER: usize = 64;
const MAX_DEFERRALS_UPPER: usize = 32;

/// Resolve the lookahead buffer size from env, falling back to the default.
/// Out-of-range or unparseable values log a warning and use the default —
/// matching the warn-then-default pattern of `resolve_max_cached_models`.
pub(crate) fn resolve_lookahead_buffer() -> usize {
    match std::env::var(LOOKAHEAD_BUFFER_ENV) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if (LOOKAHEAD_BUFFER_LOWER..=LOOKAHEAD_BUFFER_UPPER).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    env = LOOKAHEAD_BUFFER_ENV,
                    value = n,
                    lower = LOOKAHEAD_BUFFER_LOWER,
                    upper = LOOKAHEAD_BUFFER_UPPER,
                    "ignoring out-of-range queue lookahead buffer; using default"
                );
                DEFAULT_LOOKAHEAD_BUFFER
            }
            Err(e) => {
                tracing::warn!(
                    env = LOOKAHEAD_BUFFER_ENV,
                    raw = %raw,
                    error = %e,
                    "ignoring unparseable queue lookahead buffer; using default"
                );
                DEFAULT_LOOKAHEAD_BUFFER
            }
        },
        Err(_) => DEFAULT_LOOKAHEAD_BUFFER,
    }
}

/// Resolve the max-deferrals starvation budget from env. Out-of-range or
/// unparseable values log a warning and use the default.
pub(crate) fn resolve_max_deferrals() -> usize {
    match std::env::var(MAX_DEFERRALS_ENV) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if n <= MAX_DEFERRALS_UPPER => n,
            Ok(n) => {
                tracing::warn!(
                    env = MAX_DEFERRALS_ENV,
                    value = n,
                    upper = MAX_DEFERRALS_UPPER,
                    "ignoring out-of-range queue max-deferrals; using default"
                );
                DEFAULT_MAX_DEFERRALS
            }
            Err(e) => {
                tracing::warn!(
                    env = MAX_DEFERRALS_ENV,
                    raw = %raw,
                    error = %e,
                    "ignoring unparseable queue max-deferrals; using default"
                );
                DEFAULT_MAX_DEFERRALS
            }
        },
        Err(_) => DEFAULT_MAX_DEFERRALS,
    }
}

async fn process_job(state: &AppState, job: GenerationJob) {
    // Check if client already disconnected before doing any work
    if job.result_tx.is_closed() {
        tracing::debug!("skipping queued job — client disconnected");
        return;
    }

    // Single-GPU path: there's only one slot. `gpu=None` keeps the wire
    // shape consistent with multi-GPU even when we don't know the ordinal.
    state.job_registry.mark_running(&job.id, None);

    // Send "now processing" event (position 0). `id` echoes the
    // server-assigned UUID so reconnecting clients can match progress
    // updates to their persisted card.
    if let Some(ref tx) = job.progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued {
            position: 0,
            id: job.id.clone(),
        }));
    }

    // 1. Ensure model is ready (with progress forwarding)
    let progress_callback = job.progress_tx.as_ref().map(|tx| {
        let tx = tx.clone();
        Arc::new(move |event: mold_inference::ProgressEvent| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }) as model_manager::EngineProgressCallback
    });

    let activation_hint = model_manager::activation_hint_for_request(state, &job.request).await;
    let request_has_lora = model_manager::request_has_effective_lora(&job.request);
    if let Err(api_err) = model_manager::ensure_model_ready(
        state,
        &job.request.model,
        progress_callback,
        activation_hint,
        request_has_lora,
    )
    .await
    {
        let err_msg = api_err.error.clone();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        return;
    }

    // 2. Low-memory warning (MPS/unified memory only — observability aid)
    #[cfg(target_os = "macos")]
    if let Some(available) = mold_inference::device::available_system_memory_bytes() {
        if available < 1_000_000_000 {
            tracing::warn!(
                available_mb = available / 1_000_000,
                "low memory before inference — system may become unstable"
            );
        }
    }

    // 3. Take the engine out of the cache so the cache mutex stays free during
    //    generation. Mirrors the multi-GPU `gpu_worker::process_job` pattern —
    //    holding the cache lock through inference would block /api/models,
    //    /api/cache, and any concurrent gallery/admin reads.
    let taken = {
        let mut cache = state.model_cache.lock().await;
        cache.take(&job.request.model)
    };
    let Some(mut cached_engine) = taken else {
        let err_msg = "no engine available after model readiness check".to_string();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        return;
    };

    let active_gen = state.active_generation.clone();
    let gen_req = job.request.clone();
    let progress_tx = job.progress_tx.clone();

    set_active_generation(state, &job.request.model, &job.request.prompt);

    // Install progress callback before crossing into spawn_blocking — keeps
    // the callback installation off the blocking thread. Mirrors the pre-
    // refactor behavior: when streaming, set the callback; when not, clear
    // it (and only clear after generate when streaming).
    let was_streaming = progress_tx.is_some();
    if let Some(ref ptx) = progress_tx {
        let ptx = ptx.clone();
        cached_engine.engine.set_on_progress(Box::new(move |event| {
            let _ = ptx.send(SseMessage::Progress(progress_to_sse(event)));
        }));
    } else {
        cached_engine.engine.clear_on_progress();
    }

    #[cfg(feature = "metrics")]
    let inference_start = Instant::now();
    // RSS sample taken just before inference; the post-inference sample below
    // logs the per-job delta so RAM growth can be attributed to a specific
    // generation rather than tracked at process granularity.
    let rss_before = crate::resources::ram_snapshot().used_by_mold;
    // Run generation on the blocking pool. Move the engine in, return it back
    // out (alongside the result + any panic payload) so we can restore it to
    // the cache in async context regardless of outcome.
    let join_result = tokio::task::spawn_blocking(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cached_engine.engine.generate(&gen_req)
        }));
        if was_streaming {
            cached_engine.engine.clear_on_progress();
        }
        (cached_engine, result)
    })
    .await;

    let rss_after = crate::resources::ram_snapshot().used_by_mold;
    let rss_delta = rss_after as i64 - rss_before as i64;
    tracing::info!(
        model = %job.request.model,
        rss_before_mb = rss_before / 1_000_000,
        rss_after_mb = rss_after / 1_000_000,
        rss_delta_mb = rss_delta / 1_000_000,
        "generation memory delta"
    );

    #[cfg(feature = "metrics")]
    let inference_duration = inference_start.elapsed().as_secs_f64();

    // Restore the engine to the cache as soon as the blocking task joins —
    // even panics must restore so the cache isn't left with a hole. If the
    // tokio task itself failed (JoinError), the engine is gone — restoration
    // is impossible. Without `clear_in_flight` the model name would leak
    // forever in `in_flight`, so `ensure_model_ready` keeps fast-pathing
    // through `contains()` while every subsequent `take()` returns `None`,
    // permanently jamming this model. Clear the marker so the cache will
    // legitimately re-load the engine on the next request.
    let result = match join_result {
        Ok((cached_engine, panic_or_result)) => {
            {
                let mut cache = state.model_cache.lock().await;
                cache.restore(cached_engine);
            }
            clear_active_generation(state);
            Ok(panic_or_result)
        }
        Err(join_err) => {
            {
                let mut cache = state.model_cache.lock().await;
                cache.clear_in_flight(&job.request.model);
            }
            clear_active_generation(state);
            Err(join_err)
        }
    };

    match result {
        Ok(Ok(Ok(mut response))) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation(&job.request.model, inference_duration);

            if response.images.is_empty() && response.video.is_none() {
                let err_msg = "generation error: engine returned no images or video".to_string();
                if let Some(ref tx) = job.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent {
                        message: err_msg.clone(),
                    }));
                }
                let _ = job.result_tx.send(Err(err_msg));
                return;
            }
            // For video-only responses, synthesize an ImageData from the thumbnail
            // so the existing queue/SSE pipeline can handle it.
            let mut img = if !response.images.is_empty() {
                response.images.remove(0)
            } else if let Some(ref video) = response.video {
                ImageData {
                    data: video.thumbnail.clone(),
                    format: OutputFormat::Png,
                    width: video.width,
                    height: video.height,
                    index: 0,
                }
            } else {
                unreachable!("checked above");
            };
            let mut original_img = None;
            if response.video.is_none() && requested_post_upscale_model(&job.request).is_some() {
                let upscale_result = upscale_generated_image_on_single_worker(
                    state,
                    &job.request,
                    response.seed_used,
                    img.clone(),
                    job.progress_tx.as_ref(),
                )
                .await;
                let (output, preserved_original, upscale_error) =
                    settle_post_generation_upscale(img, upscale_result);
                img = output;
                original_img = preserved_original;
                if let Some(error) = upscale_error {
                    tracing::warn!(%error, "post-generation upscale failed; keeping original image");
                }
            }

            // Save to output directory if configured.
            // Builds OutputMetadata from the request + the engine's actual
            // seed_used so the DB and embedded chunks agree. Awaited (still
            // off the async loop via spawn_blocking) so the complete event
            // below can carry the saved gallery filenames.
            let metadata = OutputMetadata::from_generate_request(
                &job.request,
                response.seed_used,
                None,
                mold_core::build_info::version_string(),
            );
            let mut saved_names = SavedOutputNames::default();
            if let Some(ref dir) = job.output_dir {
                let dir = dir.clone();
                let model = job.request.model.clone();
                let batch_size = job.request.batch_size;
                let generation_time_ms = response.generation_time_ms as i64;
                let db = state.metadata_db.clone();
                let events = state.events.clone();
                let save_task = if let Some(ref video) = response.video {
                    let video_data = video.data.clone();
                    let video_gif_preview = video.gif_preview.clone();
                    let video_format = video.format;
                    let video_metadata = metadata.clone();
                    tokio::task::spawn_blocking(move || SavedOutputNames {
                        output: save_video_to_dir(
                            &dir,
                            &video_data,
                            &video_gif_preview,
                            video_format,
                            &model,
                            &video_metadata,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                            Some(&events),
                        ),
                        original: None,
                    })
                } else {
                    let img_clone = img.clone();
                    let original_clone = original_img.clone();
                    let metadata_clone = metadata.clone();
                    tokio::task::spawn_blocking(move || {
                        save_generated_image_outputs(
                            &dir,
                            original_clone.as_ref(),
                            &img_clone,
                            &model,
                            batch_size,
                            &metadata_clone,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                            Some(&events),
                        )
                    })
                };
                saved_names = save_task.await.unwrap_or_default();
            }

            // Send SSE complete event
            if let Some(ref tx) = job.progress_tx {
                let message = build_sse_completion_message(
                    &response,
                    &img,
                    original_img.as_ref(),
                    Some(&metadata),
                    &saved_names,
                    job.completion_payload,
                );
                let _ = tx.send(message);
            }

            // Send result through oneshot
            let _ = job.result_tx.send(Ok(GenerationJobResult {
                image: img,
                response,
            }));
        }
        Ok(Ok(Err(e))) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation_error(&job.request.model);

            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            tracing::error!("generation error: {e:#}");
            let err_msg = format!("generation error: {}", clean_error_message(&e));
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
        Ok(Err(panic_payload)) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation_error(&job.request.model);

            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!("inference panicked: {msg}");
            let err_msg = format!("inference panicked: {msg}");
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
        Err(join_err) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation_error(&job.request.model);

            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            tracing::error!("inference task join error: {join_err:?}");
            let err_msg = "inference task failed".to_string();
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
    }
}

// ── Multi-GPU queue dispatcher ──────────────────────────────────────────────

/// Runs the multi-GPU dispatch loop. Routes each generation job to the best
/// GPU worker per `select_worker`'s tier order: model loaded + idle > idle
/// empty GPU that fits > model loaded but busy > idle empty GPU that does
/// not fit > most-headroom fallback (evict LRU there).
/// Uses a small lookahead buffer so an interleaved queue (`[A, B, A, B]`)
/// doesn't force a sibling worker to swap models when one already has the
/// right one warm.
///
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_dispatcher(
    job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    tracing::debug!("multi-GPU queue dispatcher started");
    let buffer_size = resolve_lookahead_buffer();
    let max_deferrals = resolve_max_deferrals();
    run_queue_dispatcher_with_tuning(job_rx, state, buffer_size, max_deferrals).await;
}

async fn run_queue_dispatcher_with_tuning(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
    buffer_size: usize,
    max_deferrals: usize,
) {
    let mut buffer: VecDeque<BufferedJob> = VecDeque::with_capacity(buffer_size);

    loop {
        // Hold new-job dispatch while paused; in-flight worker jobs continue.
        state.queue_pause.wait_if_paused().await;
        if buffer.is_empty() {
            match job_rx.recv().await {
                Some(j) => buffer.push_back(BufferedJob::new(j)),
                None => break,
            }
        }
        top_up_buffer(&mut buffer, &mut job_rx, buffer_size);
        // Re-check after the recv: a pause that landed while this loop was
        // parked waiting for work must hold the job that woke it, not leak
        // it into dispatch.
        state.queue_pause.wait_if_paused().await;

        let loaded = multi_gpu_loaded_models(&state);
        let job = pick_next_job(&mut buffer, &loaded, max_deferrals);

        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());

        let job_id = job.id.clone();
        let model_name = job.request.model.clone();
        let estimated_vram = estimate_model_vram(&model_name);

        if let Some(err_msg) = crate::gpu_pool::model_unschedulable_message(&model_name) {
            tracing::warn!(model = %model_name, "{err_msg}");
            if let Some(tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
            state.queue.decrement();
            state.job_registry.remove(&job_id);
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        let placement_gpu = match state
            .gpu_pool
            .resolve_explicit_placement_gpu(job.request.placement.as_ref())
        {
            Ok(ordinal) => ordinal,
            Err(err_msg) => {
                tracing::warn!(model = %model_name, "{err_msg}");
                if let Some(tx) = job.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent {
                        message: err_msg.clone(),
                    }));
                }
                let _ = job.result_tx.send(Err(err_msg));
                state.queue.decrement();
                state.job_registry.remove(&job_id);
                #[cfg(feature = "metrics")]
                crate::metrics::record_queue_depth(state.queue.pending());
                continue;
            }
        };
        let preferred_gpu = state
            .job_registry
            .target_gpu(&job_id)
            .flatten()
            .or(placement_gpu);

        if job.result_tx.is_closed() {
            tracing::debug!(model = %model_name, "skipping queued multi-GPU job — client disconnected");
            state.queue.decrement();
            state.job_registry.remove(&job_id);
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        // Multi-GPU workers are synchronous threads and cannot pull missing
        // assets themselves. Resolve a first-use post-generation upscaler at
        // this async boundary, on the server/host that accepted the job,
        // before handing it to the selected GPU.
        if let Err(err_msg) =
            ensure_post_upscale_model_downloaded(&state, &job.request, job.progress_tx.as_ref())
                .await
        {
            tracing::warn!(
                model = %model_name,
                upscaler = ?job.request.upscale_model,
                "{err_msg}"
            );
            if let Some(tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
            state.queue.decrement();
            state.job_registry.remove(&job_id);
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        // Build the GpuJob once; the retry loop moves it between attempts.
        let mut gpu_job = Some(GpuJob {
            id: job.id.clone(),
            model: model_name.clone(),
            request: job.request,
            completion_payload: job.completion_payload,
            progress_tx: job.progress_tx,
            result_tx: job.result_tx,
            output_dir: job.output_dir,
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            queue: state.queue.clone(),
            registry: state.job_registry.clone(),
            events: state.events.clone(),
        });

        let mut skip: Vec<usize> = if preferred_gpu.is_none() {
            let failed = crate::gpu_pool::failed_ordinals_for_model(&model_name);
            if failed.len() < state.gpu_pool.worker_count() {
                failed
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let mut dispatched = false;

        while !dispatched {
            if gpu_job
                .as_ref()
                .is_some_and(|pending| pending.result_tx.is_closed())
            {
                tracing::debug!(
                    model = %model_name,
                    "dropping queued multi-GPU job before dispatch — client disconnected"
                );
                state.queue.decrement();
                state.job_registry.remove(&job_id);
                break;
            }

            let worker = if let Some(ordinal) = preferred_gpu {
                state.gpu_pool.worker_by_ordinal(ordinal)
            } else {
                state
                    .gpu_pool
                    .select_worker_excluding(&model_name, estimated_vram, &skip)
            };

            let Some(worker) = worker else {
                if preferred_gpu.is_none() && state.gpu_pool.worker_count() > 0 {
                    tracing::warn!(
                        model = %model_name,
                        "all GPU workers are temporarily unavailable; keeping job queued"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }
                let rejected = gpu_job
                    .take()
                    .expect("gpu_job retained after failed dispatch");
                let err_msg = if state.gpu_pool.worker_count() == 0 {
                    format!("no GPU available for model {model_name}")
                } else if let Some(ordinal) = preferred_gpu {
                    format!("gpu:{ordinal} is not available for model {model_name}")
                } else {
                    format!("no GPU worker available for model {model_name}")
                };
                tracing::error!(model = %model_name, "{err_msg}");
                if let Some(tx) = rejected.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent {
                        message: err_msg.clone(),
                    }));
                }
                let _ = rejected.result_tx.send(Err(err_msg));
                state.queue.decrement();
                state.job_registry.remove(&job_id);
                break;
            };

            // Increment in-flight BEFORE sending to reserve the slot.
            worker.in_flight.fetch_add(1, Ordering::SeqCst);
            let pending = gpu_job.take().expect("gpu_job present in retry loop");
            if preferred_gpu.is_none() {
                let _ = state
                    .job_registry
                    .set_target_gpu(&job_id, Some(worker.gpu.ordinal));
            }
            match worker.job_tx.try_send(pending) {
                Ok(()) => {
                    dispatched = true;
                }
                Err(std::sync::mpsc::TrySendError::Full(j)) => {
                    worker.in_flight.fetch_sub(1, Ordering::SeqCst);
                    if preferred_gpu.is_none() {
                        let _ = state.job_registry.set_target_gpu(&job_id, None);
                    }
                    gpu_job = Some(j);
                    if preferred_gpu.is_none() {
                        skip.push(worker.gpu.ordinal);
                        if skip.len() >= state.gpu_pool.worker_count().max(1) {
                            skip.clear();
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                    } else {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(j)) => {
                    worker.in_flight.fetch_sub(1, Ordering::SeqCst);
                    if preferred_gpu.is_none() {
                        let _ = state.job_registry.set_target_gpu(&job_id, None);
                    }
                    tracing::warn!(
                        gpu = worker.gpu.ordinal,
                        "GPU worker disconnected — retrying dispatch"
                    );
                    gpu_job = Some(j);
                    if preferred_gpu.is_none() {
                        skip.push(worker.gpu.ordinal);
                    } else {
                        let rejected = gpu_job.take().expect("gpu_job retained after disconnect");
                        let err_msg = format!(
                            "gpu:{} disconnected while dispatching model {model_name}",
                            worker.gpu.ordinal
                        );
                        if let Some(tx) = rejected.progress_tx {
                            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                                message: err_msg.clone(),
                            }));
                        }
                        let _ = rejected.result_tx.send(Err(err_msg));
                        state.queue.decrement();
                        state.job_registry.remove(&job_id);
                        break;
                    }
                }
            }
        }
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
    }
    tracing::info!("multi-GPU queue dispatcher shutting down");
}

/// Rough VRAM estimate for a model (used for placement decisions).
pub fn estimate_model_vram(model_name: &str) -> u64 {
    // Use a simple heuristic based on model name patterns.
    // Quantized models are smaller; BF16/FP16 are larger.
    let lower = model_name.to_lowercase();
    if lower.contains("flux2")
        && lower.contains("9b")
        && (lower.contains(":bf16") || lower.contains(":fp16"))
    {
        32_000_000_000 // Klein-9B BF16 needs a 32GB-class card in practice.
    } else if lower.contains(":q4") {
        6_000_000_000 // ~6GB
    } else if lower.contains(":q8") || lower.contains(":fp8") {
        12_000_000_000 // ~12GB
    } else if lower.contains(":bf16") || lower.contains(":fp16") {
        24_000_000_000 // ~24GB
    } else if lower.contains("sd15") || lower.contains("sd1.5") {
        4_000_000_000 // ~4GB
    } else {
        // SDXL (~8GB) and other models default to 8GB.
        8_000_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_pool::{GpuPool, GpuWorker};
    use crate::model_cache::ModelCache;
    use crate::state::QueueHandle;
    use mold_core::{GenerateRequest, ImageData, ModelConfig, OutputFormat};
    use mold_db::MetadataDb;
    use mold_inference::device::DiscoveredGpu;
    use mold_inference::shared_pool::SharedPool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex, RwLock};
    use tempfile::TempDir;

    /// A `GenerateRequest` with the bare minimum fields populated — enough to
    /// hand to `OutputMetadata::from_generate_request` in tests.
    fn fake_request(model: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: "a cat".to_string(),
            negative_prompt: None,
            model: model.to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        }
    }

    fn fake_image() -> ImageData {
        ImageData {
            // PNG magic bytes — the helpers don't validate, but this keeps
            // the on-disk file from being trivially mistaken for empty.
            data: vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            format: OutputFormat::Png,
            width: 512,
            height: 512,
            index: 0,
        }
    }

    #[test]
    fn multi_gpu_dispatch_identifies_missing_post_upscaler_for_auto_pull() {
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());

        assert_eq!(
            post_upscale_model_to_pull(&mold_core::Config::default(), &req).unwrap(),
            Some("real-esrgan-x4plus:fp16".to_string())
        );

        let tmp = TempDir::new().unwrap();
        let weights = tmp.path().join("realesrgan.safetensors");
        std::fs::write(&weights, b"test weights").unwrap();
        let mut config = mold_core::Config::default();
        config.models.insert(
            "real-esrgan-x4plus:fp16".to_string(),
            ModelConfig {
                transformer: Some(weights.display().to_string()),
                ..Default::default()
            },
        );
        assert_eq!(post_upscale_model_to_pull(&config, &req).unwrap(), None);

        config
            .models
            .get_mut("real-esrgan-x4plus:fp16")
            .unwrap()
            .transformer = Some(tmp.path().join("missing.safetensors").display().to_string());
        assert_eq!(
            post_upscale_model_to_pull(&config, &req).unwrap(),
            Some("real-esrgan-x4plus:fp16".to_string()),
            "stale config paths should trigger a repair pull"
        );
    }

    fn test_worker(
        ordinal: usize,
        channel_size: usize,
    ) -> (
        Arc<GpuWorker>,
        std::sync::mpsc::Receiver<crate::gpu_pool::GpuJob>,
    ) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(channel_size);
        let worker = Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal,
                name: format!("gpu{ordinal}"),
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(ModelCache::new(3))),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    fn empty_test_state(config: mold_core::Config) -> crate::state::AppState {
        crate::state::AppState::empty(
            config,
            QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            crate::state::AppState::empty_gpu_pool(),
            200,
        )
    }

    #[test]
    fn save_image_to_dir_writes_file_and_creates_missing_dir() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("sub/output");
        assert!(!nested.exists());

        save_image_to_dir(
            &nested,
            &fake_image(),
            "flux-dev:q4",
            1,
            None,
            None,
            None,
            None,
        );

        assert!(nested.exists(), "save should mkdir -p");
        let entries: Vec<_> = std::fs::read_dir(&nested).unwrap().collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0].as_ref().unwrap().file_name();
        let name_str = name.to_string_lossy();
        // Filename uses model-with-colon-replaced-by-dash + ms timestamp + .png.
        assert!(name_str.starts_with("mold-flux-dev-q4-"), "{name_str}");
        assert!(name_str.ends_with(".png"), "{name_str}");
    }

    #[test]
    fn save_image_to_dir_includes_batch_index_when_batch_size_gt_1() {
        let tmp = TempDir::new().unwrap();
        let mut img = fake_image();
        img.index = 3;
        img.format = OutputFormat::Jpeg;
        img.data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic

        save_image_to_dir(tmp.path(), &img, "sdxl", 4, None, None, None, None);

        let entries: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().collect();
        let name = entries[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(
            name.contains("-3.jpeg"),
            "expected batch index suffix: {name}"
        );
    }

    #[test]
    fn save_image_to_dir_upserts_metadata_row_when_db_provided() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("flux-dev:q4");
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            Some(1234),
            Some(&db),
            None,
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1, "exactly one DB row for the saved file");
        let rec = &rows[0];
        assert_eq!(rec.metadata.prompt, "a cat");
        assert_eq!(rec.metadata.seed, 42);
        assert_eq!(rec.metadata.version, "test-version");
        assert_eq!(rec.format, OutputFormat::Png);
        assert_eq!(rec.generation_time_ms, Some(1234));
        // stat_from_disk should have populated the size from the actual file.
        assert!(rec.file_size_bytes.unwrap_or(0) > 0);
    }

    #[test]
    fn save_generated_image_outputs_persists_original_and_upscaled_dimensions() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");
        let original = fake_image();
        let mut upscaled = fake_image();
        upscaled.width = 2048;
        upscaled.height = 2048;
        upscaled.data = vec![4, 5, 6];

        save_generated_image_outputs(
            tmp.path(),
            Some(&original),
            &upscaled,
            "flux-dev:q4",
            1,
            &meta,
            Some(1234),
            Some(&db),
            None,
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 2);
        let original_row = rows
            .iter()
            .find(|row| row.filename.contains("-original."))
            .expect("original row");
        let upscaled_row = rows
            .iter()
            .find(|row| row.filename.contains("-upscaled."))
            .expect("upscaled row");
        assert_eq!(
            (original_row.metadata.width, original_row.metadata.height),
            (512, 512)
        );
        assert_eq!(
            (upscaled_row.metadata.width, upscaled_row.metadata.height),
            (2048, 2048)
        );
        assert_eq!(upscaled_row.metadata.generation_width, Some(512));
        assert_eq!(upscaled_row.metadata.generation_height, Some(512));
    }

    #[test]
    fn save_image_to_dir_skips_db_when_metadata_is_none() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            None, // ← metadata absent
            Some(1234),
            Some(&db),
            None,
        );

        // File still on disk, but no DB row recorded — both gates must hold
        // for the upsert to fire.
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
        assert_eq!(db.list(None).unwrap().len(), 0);
    }

    #[test]
    fn save_image_to_dir_invalid_path_does_not_panic() {
        // /dev/null is a file, not a directory — create_dir_all should fail
        // and the helper must log + return cleanly rather than panic.
        save_image_to_dir(
            std::path::Path::new("/dev/null/cant-mkdir-here"),
            &fake_image(),
            "test",
            1,
            None,
            None,
            None,
            None,
        );
    }

    #[test]
    fn save_image_to_dir_emits_gallery_added_with_row_when_db_records() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("flux-dev:q4");
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            Some(1234),
            Some(&db),
            Some(&events),
        );

        match rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryAdded { filename, image } => {
                assert!(filename.ends_with(".png"), "{filename}");
                let img = image.expect("DB recorded — event must carry the gallery row");
                assert_eq!(img.filename, filename);
                assert_eq!(img.metadata.prompt, "a cat");
            }
            other => panic!("expected gallery_added, got {other:?}"),
        }
    }

    #[test]
    fn save_image_to_dir_emits_gallery_added_without_row_when_db_absent() {
        let tmp = TempDir::new().unwrap();
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            None,
            None,
            None, // no DB
            Some(&events),
        );

        match rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryAdded { image, .. } => {
                assert!(image.is_none(), "no DB → clients must refetch");
            }
            other => panic!("expected gallery_added, got {other:?}"),
        }
    }

    #[test]
    fn save_image_to_dir_emits_nothing_on_write_failure() {
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_image_to_dir(
            std::path::Path::new("/dev/null/cant-mkdir-here"),
            &fake_image(),
            "test",
            1,
            None,
            None,
            None,
            Some(&events),
        );

        assert!(
            rx.try_recv().is_err(),
            "failed save must not announce a gallery entry"
        );
    }

    #[test]
    fn save_video_to_dir_emits_gallery_added() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 1, None, "v");
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_video_to_dir(
            tmp.path(),
            b"fake mp4 bytes",
            b"",
            OutputFormat::Mp4,
            "ltx-video:fp16",
            &meta,
            Some(5000),
            Some(&db),
            Some(&events),
        );

        match rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryAdded { filename, image } => {
                assert!(filename.ends_with(".mp4"), "{filename}");
                assert!(image.is_some());
            }
            other => panic!("expected gallery_added, got {other:?}"),
        }
    }

    #[test]
    fn save_video_to_dir_writes_mp4_and_records_metadata() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let mut req = fake_request("ltx-video:fp16");
        req.frames = Some(25);
        req.fps = Some(24);
        let meta = OutputMetadata::from_generate_request(&req, 99, None, "test-version");

        // Minimal MP4-ish bytes: an `ftyp` box header. The helper writes
        // bytes verbatim — content validation happens at gallery scan time.
        let bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom".to_vec();

        save_video_to_dir(
            tmp.path(),
            &bytes,
            b"",
            OutputFormat::Mp4,
            "ltx-video:fp16",
            &meta,
            Some(5000),
            Some(&db),
            None,
        );

        let entries: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(name.starts_with("mold-ltx-video-fp16-"), "{name}");
        assert!(name.ends_with(".mp4"), "{name}");

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].format, OutputFormat::Mp4);
        assert_eq!(rows[0].metadata.frames, Some(25));
        assert_eq!(rows[0].metadata.fps, Some(24));
        assert_eq!(rows[0].generation_time_ms, Some(5000));
    }

    #[test]
    fn save_video_to_dir_without_db_still_writes_file() {
        let tmp = TempDir::new().unwrap();
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 1, None, "v");

        save_video_to_dir(
            tmp.path(),
            b"fake gif bytes",
            b"",
            OutputFormat::Gif,
            "ltx-video:fp16",
            &meta,
            None,
            None,
            None,
        );

        let entries: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(name.ends_with(".gif"), "{name}");
    }

    #[test]
    fn save_video_to_dir_invalid_path_does_not_panic() {
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 1, None, "v");
        save_video_to_dir(
            std::path::Path::new("/dev/null/nope"),
            b"x",
            b"",
            OutputFormat::Mp4,
            "test",
            &meta,
            None,
            None,
            None,
        );
    }

    /// `save_video_preview_gif_to` must write to
    /// `<preview_dir>/<filename>.preview.gif` — the exact location
    /// `GET /api/gallery/preview/:filename` streams from. Without this
    /// sidecar the preview endpoint would 404 on every real generation
    /// and the TUI detail pane would only ever see the PNG thumbnail
    /// fallback.
    #[test]
    fn save_video_preview_gif_writes_to_preview_cache() {
        let td = tempfile::tempdir().unwrap();
        let preview_dir = td.path().join("cache").join("previews");

        const GIF: &[u8] = b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x3b";
        save_video_preview_gif_to(&preview_dir, "ltx2-42.mp4", GIF);

        let expected = preview_dir.join("ltx2-42.mp4.preview.gif");
        assert!(
            expected.is_file(),
            "preview gif should land at {}",
            expected.display()
        );
        assert_eq!(std::fs::read(&expected).unwrap(), GIF);
    }

    #[test]
    fn build_sse_complete_event_video_carries_mp4_payload_and_metadata() {
        // Regression guard for the multi-GPU bug: if `response.video` is set,
        // the SSE complete event must encode the actual video bytes and
        // populate every `video_*` field so the client can reconstruct a
        // `VideoData`. Before the shared helper, `gpu_worker.rs` encoded the
        // thumbnail PNG and hard-coded every `video_*` field to `None`,
        // silently degrading every LTX-Video / LTX-2 response to an image.
        let video = mold_core::VideoData {
            data: vec![0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p'],
            format: OutputFormat::Mp4,
            width: 768,
            height: 512,
            frames: 25,
            fps: 24,
            thumbnail: vec![0x89, 0x50, 0x4E, 0x47],
            gif_preview: vec![b'G', b'I', b'F', b'8'],
            has_audio: true,
            duration_ms: Some(1040),
            audio_sample_rate: Some(44100),
            audio_channels: Some(2),
        };
        let resp = mold_core::GenerateResponse {
            images: vec![],
            video: Some(video.clone()),
            generation_time_ms: 1234,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            seed_used: 7,
            gpu: Some(0),
        };
        // The `img` the caller synthesizes from the video thumbnail — must be
        // ignored for the video branch.
        let thumb_img = ImageData {
            data: video.thumbnail.clone(),
            format: OutputFormat::Png,
            width: video.width,
            height: video.height,
            index: 0,
        };

        let event = build_sse_complete_event(
            &resp,
            &thumb_img,
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );

        let b64 = base64::engine::general_purpose::STANDARD;
        assert_eq!(event.image, b64.encode(&video.data));
        assert_eq!(event.format, OutputFormat::Mp4);
        assert_eq!(event.video_frames, Some(25));
        assert_eq!(event.video_fps, Some(24));
        assert_eq!(event.video_thumbnail, Some(b64.encode(&video.thumbnail)));
        assert_eq!(
            event.video_gif_preview,
            Some(b64.encode(&video.gif_preview))
        );
        assert!(event.video_has_audio);
        assert_eq!(event.video_duration_ms, Some(1040));
        assert_eq!(event.gpu, Some(0));

        let saved = SavedOutputNames {
            output: Some("generated-video.mp4".to_string()),
            original: None,
        };
        let metadata_only = build_sse_complete_event(
            &resp,
            &thumb_img,
            None,
            None,
            &saved,
            SseCompletionPayload::MetadataOnly,
        );
        assert!(metadata_only.image.is_empty());
        assert!(metadata_only.video_thumbnail.is_none());
        assert!(metadata_only.video_gif_preview.is_none());
        assert_eq!(metadata_only.video_frames, Some(25));
        assert_eq!(
            metadata_only.filename.as_deref(),
            Some("generated-video.mp4")
        );
    }

    #[test]
    fn build_sse_complete_event_video_empty_gif_preview_omits_field() {
        let video = mold_core::VideoData {
            data: vec![0x00, 0x00, 0x00, 0x18],
            format: OutputFormat::Mp4,
            width: 256,
            height: 256,
            frames: 17,
            fps: 12,
            thumbnail: vec![0x89, 0x50],
            gif_preview: Vec::new(),
            has_audio: false,
            duration_ms: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        let resp = mold_core::GenerateResponse {
            images: vec![],
            video: Some(video),
            generation_time_ms: 0,
            model: "m".to_string(),
            seed_used: 0,
            gpu: None,
        };
        let event = build_sse_complete_event(
            &resp,
            &fake_image(),
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert!(event.video_gif_preview.is_none());
        assert!(!event.video_has_audio);
    }

    #[test]
    fn build_sse_complete_event_image_clears_all_video_fields() {
        let resp = mold_core::GenerateResponse {
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 100,
            model: "flux-schnell:q8".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let event = build_sse_complete_event(
            &resp,
            &fake_image(),
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert_eq!(event.format, OutputFormat::Png);
        assert!(event.video_frames.is_none());
        assert!(event.video_fps.is_none());
        assert!(event.video_thumbnail.is_none());
        assert!(event.video_gif_preview.is_none());
        assert!(!event.video_has_audio);
        assert!(event.video_duration_ms.is_none());
    }

    #[test]
    fn build_sse_complete_event_carries_saved_names_and_recorded_metadata() {
        let req = fake_request("flux-dev:q4");
        let resp = mold_core::GenerateResponse {
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let metadata =
            OutputMetadata::from_generate_request(&req, resp.seed_used, None, "test-version");
        let saved = SavedOutputNames {
            output: Some("flux-dev-q4-123.png".to_string()),
            original: Some("flux-dev-q4-123-original.png".to_string()),
        };
        let event = build_sse_complete_event(
            &resp,
            &fake_image(),
            None,
            Some(&metadata),
            &saved,
            SseCompletionPayload::Full,
        );
        assert_eq!(event.filename.as_deref(), Some("flux-dev-q4-123.png"));
        assert_eq!(
            event.original_filename.as_deref(),
            Some("flux-dev-q4-123-original.png")
        );
        // The event metadata mirrors what the save path records: the
        // payload's actual dimensions, not the request's.
        let meta = event.metadata.expect("metadata rides the complete event");
        assert_eq!(meta.seed, 5);
        assert_eq!(meta.width, fake_image().width);
        assert_eq!(meta.height, fake_image().height);

        let metadata_only = build_sse_complete_event(
            &resp,
            &fake_image(),
            Some(&fake_image()),
            Some(&metadata),
            &saved,
            SseCompletionPayload::MetadataOnly,
        );
        assert!(metadata_only.image.is_empty());
        assert!(metadata_only.original_image.is_none());
        assert_eq!(
            metadata_only.filename.as_deref(),
            Some("flux-dev-q4-123.png")
        );
        assert!(metadata_only.metadata.is_some());
    }

    #[test]
    fn metadata_only_completion_fails_when_the_output_was_not_saved() {
        let response = mold_core::GenerateResponse {
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let message = build_sse_completion_message(
            &response,
            &fake_image(),
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::MetadataOnly,
        );
        match message {
            SseMessage::Error(error) => assert!(error.message.contains("could not be saved")),
            _ => panic!("metadata-only completion without a file must be an SSE error"),
        }
    }

    #[test]
    fn post_generation_upscale_replaces_image_response_dimensions() {
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let mut response = mold_core::GenerateResponse {
            images: vec![],
            video: None,
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let img = fake_image();
        let upscaled = mold_core::UpscaleResponse {
            image: ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 2048,
                height: 2048,
                index: 0,
            },
            upscale_time_ms: 42,
            model: "real-esrgan-x4plus:fp16".to_string(),
            scale_factor: 4,
            original_width: 512,
            original_height: 512,
        };

        let next = apply_upscale_response_to_image_generation(&req, &mut response, img, upscaled)
            .expect("image upscale should apply");
        let event = build_sse_complete_event(
            &response,
            &next,
            Some(&fake_image()),
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert!(event.original_image.is_some());
        assert_eq!(event.original_width, Some(512));
        assert_eq!(event.original_height, Some(512));
        let mut metadata =
            OutputMetadata::from_generate_request(&req, response.seed_used, None, "test-version");
        apply_output_dimensions_to_metadata(&mut metadata, &next);

        assert_eq!(next.width, 2048);
        assert_eq!(next.height, 2048);
        assert_eq!(event.width, 2048);
        assert_eq!(event.height, 2048);
        assert_eq!(metadata.width, 2048);
        assert_eq!(metadata.height, 2048);
        assert_eq!(metadata.generation_width, Some(512));
        assert_eq!(metadata.generation_height, Some(512));
        assert_eq!(
            metadata.upscale_model.as_deref(),
            Some("real-esrgan-x4plus:fp16")
        );
    }

    #[test]
    fn failed_post_generation_upscale_keeps_only_the_original_output() {
        let original = fake_image();
        let (output, preserved_original, error) = settle_post_generation_upscale(
            original.clone(),
            Err("upscaler unavailable".to_string()),
        );

        assert_eq!(output.data, original.data);
        assert!(preserved_original.is_none());
        assert_eq!(error.as_deref(), Some("upscaler unavailable"));
    }

    #[test]
    fn post_generation_upscale_skips_video_responses() {
        let mut req = fake_request("ltx-video:fp16");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let video = mold_core::VideoData {
            data: vec![0, 0, 0, 24],
            format: OutputFormat::Mp4,
            width: 512,
            height: 512,
            frames: 25,
            fps: 24,
            thumbnail: vec![9, 9],
            gif_preview: vec![],
            has_audio: false,
            duration_ms: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        let mut response = mold_core::GenerateResponse {
            images: vec![],
            video: Some(video),
            generation_time_ms: 100,
            model: "ltx-video:fp16".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let img = fake_image();
        let upscaled = mold_core::UpscaleResponse {
            image: ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 2048,
                height: 2048,
                index: 0,
            },
            upscale_time_ms: 42,
            model: "real-esrgan-x4plus:fp16".to_string(),
            scale_factor: 4,
            original_width: 512,
            original_height: 512,
        };

        let next = apply_upscale_response_to_image_generation(&req, &mut response, img, upscaled)
            .expect("video upscale should be skipped");

        assert_eq!(next.width, 512);
        assert_eq!(next.height, 512);
        assert!(response.video.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_post_upscale_noops_without_model() {
        let state = empty_test_state(mold_core::Config::default());
        let req = fake_request("flux-dev:q4");

        let next = upscale_generated_image_on_single_worker(&state, &req, 5, fake_image(), None)
            .await
            .expect("missing upscale model should leave the image unchanged");

        assert_eq!(next.width, 512);
        assert_eq!(next.height, 512);
        assert_eq!(next.index, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_post_upscale_rejects_unknown_upscaler_manifest() {
        let state = empty_test_state(mold_core::Config::default());
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("definitely-not-a-real-upscaler:fp16".to_string());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

        let err = upscale_generated_image_on_single_worker(
            &state,
            &req,
            5,
            fake_image(),
            Some(&progress_tx),
        )
        .await
        .expect_err("unknown upscalers should fail before generation completes");

        assert!(err.contains("unknown upscaler model"), "got: {err}");
        let first_progress = progress_rx
            .try_recv()
            .expect("loading stage should be emitted before validation fails");
        assert!(matches!(
            first_progress,
            SseMessage::Progress(SseProgressEvent::StageStart { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_post_upscale_surfaces_missing_weights_path() {
        let tmp = TempDir::new().unwrap();
        let missing_weights = tmp.path().join("missing-upscaler.safetensors");
        let mut config = mold_core::Config::default();
        config.models.insert(
            "real-esrgan-x4plus:fp16".to_string(),
            ModelConfig {
                transformer: Some(missing_weights.display().to_string()),
                ..Default::default()
            },
        );
        let state = empty_test_state(config);
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

        let err = upscale_generated_image_on_single_worker(
            &state,
            &req,
            5,
            fake_image(),
            Some(&progress_tx),
        )
        .await
        .expect_err("missing weight files should be surfaced");

        assert!(err.contains("upscale failed"), "got: {err}");
        assert!(err.contains("upscaler weights not found"), "got: {err}");
        let first_progress = progress_rx
            .try_recv()
            .expect("loading stage should be emitted before loading fails");
        assert!(matches!(
            first_progress,
            SseMessage::Progress(SseProgressEvent::StageStart { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_waits_for_worker_capacity_instead_of_rejecting() {
        let (worker, worker_rx) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()],
            }),
            8,
        );

        let (filler_result_tx, _filler_result_rx) = tokio::sync::oneshot::channel();
        let filler_job = crate::gpu_pool::GpuJob {
            id: String::new(),
            model: "busy-model".to_string(),
            request: fake_request("busy-model"),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx: filler_result_tx,
            output_dir: None,
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            queue: state.queue.clone(),
            registry: state.job_registry.clone(),
            events: state.events.clone(),
        };
        worker.job_tx.send(filler_job).unwrap();

        let dispatcher = tokio::spawn(run_queue_dispatcher_with_tuning(
            job_rx,
            state.clone(),
            8,
            DEFAULT_MAX_DEFERRALS,
        ));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: String::new(),
            request: fake_request("flux-dev:q4"),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(
            result_rx.try_recv().is_err(),
            "dispatcher should keep the job pending while all worker channels are full"
        );

        let _filler = worker_rx
            .recv()
            .expect("filler job should occupy the local channel");
        let dispatched = worker_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("queued job should dispatch once capacity is available");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_waits_for_degraded_worker_recovery_instead_of_rejecting() {
        let (worker, worker_rx) = test_worker(0, 1);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() =
            Some(Instant::now() + std::time::Duration::from_secs(60));

        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()],
            }),
            8,
        );
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: String::new(),
            request: fake_request("flux-dev:q4"),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
        };
        queue.submit(job, 8).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(
            result_rx.try_recv().is_err(),
            "dispatcher should keep the job pending while all workers are degraded"
        );
        assert!(
            worker_rx.try_recv().is_err(),
            "degraded worker must not receive work before recovery"
        );

        worker.consecutive_failures.store(0, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() = None;

        let dispatched = worker_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("queued job should dispatch once a worker recovers");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }

    /// Regression for the take-and-restore refactor in `process_job`: when
    /// the engine vanishes from the cache between `ensure_model_ready` and
    /// `cache.take()`, the take path must produce `None` (handled with a
    /// clean error in `process_job`) rather than panicking. The pure cache
    /// invariant — `take()` on an absent model returns `None` — is what
    /// keeps the take-and-restore safe.
    #[tokio::test]
    async fn cache_take_on_vanished_engine_returns_none_not_panic() {
        use crate::model_cache::ModelCache;
        use mold_core::GenerateResponse;
        use mold_inference::InferenceEngine;

        struct StubEngine(&'static str);
        impl InferenceEngine for StubEngine {
            fn generate(&mut self, _r: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
                unimplemented!()
            }
            fn model_name(&self) -> &str {
                self.0
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn load(&mut self) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let mut cache = ModelCache::new(3);
        // Cache empty (engine never inserted, or evicted/removed by a
        // concurrent admin call between `ensure_model_ready` and `take`).
        assert!(cache.take("vanished-model").is_none());

        // After a take of a present engine, a subsequent take of the same
        // name must also return None — guards against double-take in the
        // restore path.
        cache.insert(Box::new(StubEngine("present-model")), 0);
        let first = cache.take("present-model");
        assert!(first.is_some());
        assert!(
            cache.take("present-model").is_none(),
            "double-take must return None"
        );
    }

    fn buf_job(model: &str) -> BufferedJob {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        BufferedJob::new(crate::state::GenerationJob {
            id: String::new(),
            request: fake_request(model),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx: tx,
            output_dir: None,
        })
    }

    #[test]
    fn pick_next_job_picks_head_when_head_model_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        buffer.push_back(buf_job("a"));
        let loaded: HashSet<String> = ["a".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "a");
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.front().unwrap().job.request.model, "b");
        assert_eq!(
            buffer.front().unwrap().deferred,
            0,
            "head shouldn't be deferred when picker chose the head itself"
        );
    }

    #[test]
    fn pick_next_job_picks_non_head_when_only_non_head_model_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        buffer.push_back(buf_job("a"));
        let loaded: HashSet<String> = ["b".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "b");
        assert_eq!(buffer.len(), 2);
        // The head ("a") was skipped once and now sits at deferral=1.
        assert_eq!(buffer.front().unwrap().job.request.model, "a");
        assert_eq!(buffer.front().unwrap().deferred, 1);
    }

    #[test]
    fn pick_next_job_force_dispatches_head_after_max_deferrals() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        let mut head = buf_job("a");
        head.deferred = 3;
        buffer.push_back(head);
        buffer.push_back(buf_job("b"));
        // Even though only `b` is loaded, head ("a") has hit the budget and wins.
        let loaded: HashSet<String> = ["b".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "a");
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front().unwrap().job.request.model, "b");
    }

    #[test]
    fn pick_next_job_falls_back_to_head_when_nothing_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        let loaded: HashSet<String> = HashSet::new();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "a");
    }

    /// Fix D: with `max_deferrals = 0`, every reorder would exceed the
    /// budget on the very first skip, so the picker degenerates to FIFO —
    /// the head wins regardless of which model is loaded.
    #[test]
    fn pick_next_job_max_deferrals_zero_picks_head_even_when_non_head_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("b")); // head
        buffer.push_back(buf_job("a")); // non-head
        let loaded: HashSet<String> = ["a".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 0);
        assert_eq!(
            picked.request.model, "b",
            "max_deferrals=0 must force FIFO — head must win even when only the non-head model is loaded"
        );
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front().unwrap().job.request.model, "a");
    }

    /// Fix D: with `max_deferrals = 0` and an empty `loaded` set, the head
    /// is the only candidate anyway. Locks in the FIFO behaviour when
    /// nothing is warm.
    #[test]
    fn pick_next_job_max_deferrals_zero_with_empty_loaded_picks_head() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a")); // head
        buffer.push_back(buf_job("b"));
        let loaded: HashSet<String> = HashSet::new();
        let picked = pick_next_job(&mut buffer, &loaded, 0);
        assert_eq!(picked.request.model, "a");
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front().unwrap().job.request.model, "b");
    }

    /// Fix E: when both head and a non-head match `loaded`, the picker must
    /// pick the front-most match — i.e. the first `A` in `[A, B, A, B]`
    /// when both `A` and `B` are loaded. Locks in arrival-order stability
    /// across multiple matching jobs.
    #[test]
    fn pick_next_job_picks_front_most_match_when_multiple_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        let loaded: HashSet<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(
            picked.request.model, "a",
            "front-most match wins (the first `a`), not the loaded model with the most copies later in the buffer"
        );
        // Three jobs remain: [b, a, b]; head was the picked first `a` so the
        // new head is the original-index-1 `b`. Nothing was deferred because
        // the picker chose the head itself.
        assert_eq!(buffer.len(), 3);
        let remaining: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(remaining, vec!["b", "a", "b"]);
        assert_eq!(buffer.front().unwrap().deferred, 0);
    }

    /// Integration: an interleaved `[A, B, A, B]` queue dispatched against a
    /// single worker that has model `A` warm should reorder so both `A` jobs
    /// run first, then both `B` jobs — minimizing model swaps from 4 → 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_reorders_interleaved_jobs_to_minimize_swaps() {
        let (worker, worker_rx) = test_worker(0, 8);
        // Pre-mark the worker as having model "a" loaded so the picker
        // recognises it as warm.
        {
            let mut cache = worker.model_cache.lock().unwrap();
            struct Engine(&'static str);
            impl mold_inference::InferenceEngine for Engine {
                fn generate(
                    &mut self,
                    _r: &GenerateRequest,
                ) -> anyhow::Result<mold_core::GenerateResponse> {
                    unimplemented!()
                }
                fn model_name(&self) -> &str {
                    self.0
                }
                fn is_loaded(&self) -> bool {
                    true
                }
                fn load(&mut self) -> anyhow::Result<()> {
                    Ok(())
                }
            }
            cache.insert(Box::new(Engine("a")), 0);
        }

        let (job_tx, job_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()],
            }),
            8,
        );

        // Submit [a, b, a, b] BEFORE the dispatcher spins up so the buffer
        // top-up sees all four at once.
        let mut result_rxs = Vec::new();
        for model in ["a", "b", "a", "b"] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let job = crate::state::GenerationJob {
                id: String::new(),
                request: fake_request(model),
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
            };
            queue.submit(job, 8).await.unwrap();
            result_rxs.push(rx);
        }

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let mut order = Vec::new();
        for _ in 0..4 {
            let dispatched = worker_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("worker should receive the dispatched job");
            order.push(dispatched.model);
        }
        drop(job_tx);
        dispatcher.abort();

        assert_eq!(
            order,
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
            ],
            "lookahead reorder should batch all `a` jobs together before swapping to `b`"
        );
    }

    /// Fix F: the `top_up_buffer` helper must never grow the buffer past
    /// `buffer_size`, no matter how many jobs are sitting in the channel.
    /// This is the load-bearing invariant that bounds the working set the
    /// picker considers — without it a burst submission could let the
    /// dispatcher reorder across the entire pending queue, defeating the
    /// fairness guarantees the `deferred` counter is built around.
    #[tokio::test]
    async fn top_up_buffer_never_exceeds_capacity() {
        use std::collections::VecDeque;
        let (job_tx, mut job_rx) = tokio::sync::mpsc::channel::<GenerationJob>(32);

        // Submit 10 jobs into the channel synchronously so the buffer's top-up
        // call sees them all immediately available via try_recv.
        for i in 0..10 {
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let job = GenerationJob {
                id: String::new(),
                request: fake_request(&format!("model-{i}")),
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
            };
            job_tx.send(job).await.unwrap();
        }

        // buffer_size = 4 — top_up must stop at 4 even with 10 in the channel.
        let mut buffer: VecDeque<BufferedJob> = VecDeque::with_capacity(4);
        top_up_buffer(&mut buffer, &mut job_rx, 4);
        assert_eq!(
            buffer.len(),
            4,
            "top_up_buffer must cap at buffer_size, leaving the rest in the channel"
        );

        // Drain the four buffered jobs, then top up again; the next call must
        // pull only the next four from the channel (FIFO order preserved).
        while buffer.pop_front().is_some() {}
        top_up_buffer(&mut buffer, &mut job_rx, 4);
        assert_eq!(buffer.len(), 4);
        let names: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["model-4", "model-5", "model-6", "model-7"],
            "second top-up must drain the next FIFO window from the channel"
        );

        // Drop sender so the channel reports closed; remaining 2 jobs still
        // arrive via try_recv before the channel goes dry.
        drop(job_tx);
        while buffer.pop_front().is_some() {}
        top_up_buffer(&mut buffer, &mut job_rx, 4);
        assert_eq!(
            buffer.len(),
            2,
            "top_up_buffer drains the channel tail when fewer jobs than capacity remain"
        );
        let names: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(names, vec!["model-8", "model-9"]);
    }

    /// Same invariant, but reached via the dispatcher loop (integration). A
    /// burst of N > buffer_size jobs must still dispatch in FIFO order with
    /// no jobs lost — the buffer cap can't drop traffic, only delay it. We
    /// drain the worker channel as fast as the dispatcher fills it, so the
    /// test exercises buffer rotation rather than worker-channel back-pressure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_dispatches_all_jobs_when_submission_exceeds_buffer() {
        let (worker, worker_rx) = test_worker(0, 4);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(32);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()],
            }),
            32,
        );

        // Drain the worker channel concurrently and decrement in_flight as
        // a real worker would, so the dispatcher's worker-selection sees the
        // worker as idle for each subsequent send (otherwise `in_flight`
        // grows unbounded and the worker never re-classifies as eligible
        // when the sync-channel fills).
        let drain_worker = worker.clone();
        let drainer = std::thread::spawn(move || {
            let mut order = Vec::new();
            while order.len() < 10 {
                match worker_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                    Ok(j) => {
                        drain_worker.in_flight.fetch_sub(1, Ordering::SeqCst);
                        order.push(j.model);
                    }
                    Err(e) => panic!("drain stalled at {:?}: {e:?}", order),
                }
            }
            order
        });

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        // Submit AFTER the dispatcher and drainer are running so we exercise
        // the live top-up loop rather than a one-shot drain of a pre-filled
        // channel. Hold result_rx values past the dispatch — the dispatcher
        // skips jobs whose result_tx is closed, which would otherwise drop
        // every job before it reaches the worker channel.
        let mut held_rxs = Vec::new();
        for i in 0..10 {
            let (tx, rx) = tokio::sync::oneshot::channel();
            held_rxs.push(rx);
            let job = crate::state::GenerationJob {
                id: String::new(),
                request: fake_request(&format!("model-{i}")),
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
            };
            queue.submit(job, 32).await.unwrap();
        }

        let order = drainer.join().expect("drainer thread panic");
        drop(job_tx);
        dispatcher.abort();

        let expected: Vec<String> = (0..10).map(|i| format!("model-{i}")).collect();
        assert_eq!(
            order, expected,
            "10 distinct jobs must come out in FIFO across buffer rotations"
        );
    }

    /// Serializes every test that mutates queue env vars (process-global).
    static QUEUE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_queue_env<R>(name: &str, value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _g = QUEUE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(name).ok();
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        out
    }

    #[test]
    fn resolve_lookahead_buffer_uses_default_when_env_missing() {
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, None, resolve_lookahead_buffer);
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
    }

    #[test]
    fn resolve_lookahead_buffer_honors_env_within_range() {
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, Some("4"), resolve_lookahead_buffer);
        assert_eq!(n, 4);
    }

    #[test]
    fn resolve_lookahead_buffer_falls_back_when_out_of_range() {
        // 0 is below the 1 lower bound; 999 is above the 64 upper bound.
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, Some("0"), resolve_lookahead_buffer);
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, Some("999"), resolve_lookahead_buffer);
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
    }

    #[test]
    fn resolve_lookahead_buffer_falls_back_when_unparseable() {
        let n = with_queue_env(
            LOOKAHEAD_BUFFER_ENV,
            Some("not-a-number"),
            resolve_lookahead_buffer,
        );
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
    }

    #[test]
    fn resolve_max_deferrals_uses_default_when_env_missing() {
        let n = with_queue_env(MAX_DEFERRALS_ENV, None, resolve_max_deferrals);
        assert_eq!(n, DEFAULT_MAX_DEFERRALS);
    }

    #[test]
    fn resolve_max_deferrals_honors_env_within_range() {
        // 0 is the in-range "FIFO" sentinel, 32 is the upper edge.
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("0"), resolve_max_deferrals);
        assert_eq!(n, 0);
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("32"), resolve_max_deferrals);
        assert_eq!(n, 32);
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("5"), resolve_max_deferrals);
        assert_eq!(n, 5);
    }

    #[test]
    fn resolve_max_deferrals_falls_back_when_out_of_range() {
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("999"), resolve_max_deferrals);
        assert_eq!(n, DEFAULT_MAX_DEFERRALS);
    }

    #[test]
    fn resolve_max_deferrals_falls_back_when_unparseable() {
        let n = with_queue_env(
            MAX_DEFERRALS_ENV,
            Some("not-a-number"),
            resolve_max_deferrals,
        );
        assert_eq!(n, DEFAULT_MAX_DEFERRALS);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_honors_explicit_placement_gpu() {
        let (worker0, rx0) = test_worker(0, 1);
        let (worker1, rx1) = test_worker(1, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0, worker1],
            }),
            8,
        );

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state));

        let mut request = fake_request("flux-dev:q4");
        request.placement = Some(mold_core::types::DevicePlacement {
            text_encoders: mold_core::types::DeviceRef::Auto,
            advanced: Some(mold_core::types::AdvancedPlacement {
                transformer: mold_core::types::DeviceRef::gpu(1),
                ..mold_core::types::AdvancedPlacement::default()
            }),
        });

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: String::new(),
            request,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        let dispatched = rx1
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("explicit placement should route to gpu 1");
        assert_eq!(dispatched.model, "flux-dev:q4");
        assert!(rx0.try_recv().is_err(), "gpu 0 should not receive the job");

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_records_auto_selected_gpu_before_worker_starts() {
        let (worker0, rx0) = test_worker(0, 1);
        let (worker1, rx1) = test_worker(1, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0, worker1],
            }),
            8,
        );
        state.job_registry.register("auto-job", "flux-dev:q4");

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: "auto-job".to_string(),
            request: fake_request("flux-dev:q4"),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        let (dispatched, ordinal) = match rx0.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(job) => (job, 0),
            Err(_) => (
                rx1.recv_timeout(std::time::Duration::from_secs(1))
                    .expect("auto job should dispatch to one GPU"),
                1,
            ),
        };
        assert_eq!(dispatched.model, "flux-dev:q4");
        assert_eq!(
            state.job_registry.target_gpu("auto-job"),
            Some(Some(ordinal))
        );

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paused_dispatcher_holds_new_jobs_until_resumed() {
        let (worker0, rx0) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0],
            }),
            8,
        );

        // Pause before the dispatcher runs — a submitted job must stay queued.
        assert!(state.queue_pause.pause());
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: "paused-job".to_string(),
            request: fake_request("flux-dev:q4"),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        // While paused the worker never receives the job.
        assert!(
            rx0.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "paused dispatcher must not hand a job to a worker"
        );

        // Resume → the queued job dispatches.
        assert!(state.queue_pause.resume());
        let dispatched = rx0
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("resumed dispatcher should dispatch the queued job");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_while_dispatcher_is_parked_on_an_empty_queue_still_holds_the_next_job() {
        // The subtle ordering: the dispatcher passes the top-of-loop gate,
        // then parks in job_rx.recv() on an EMPTY queue. A pause that lands
        // while it is parked must hold the very job whose arrival wakes it —
        // without the post-recv re-check, that job leaks into dispatch.
        let (worker0, rx0) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0],
            }),
            8,
        );

        // Dispatcher starts UNPAUSED and parks waiting for work.
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pause lands while it is parked, then a job arrives.
        assert!(state.queue_pause.pause());
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: "parked-job".to_string(),
            request: fake_request("flux-dev:q4"),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        assert!(
            rx0.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "a job arriving while paused must not wake straight into dispatch"
        );

        assert!(state.queue_pause.resume());
        let dispatched = rx0
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("resume should release the held job");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }
}

#[cfg(test)]
mod queue_pause_tests {
    use super::QueuePause;
    use std::time::Duration;

    #[test]
    fn pause_and_resume_report_state_transitions() {
        let gate = QueuePause::new();
        assert!(!gate.is_paused());
        assert!(gate.pause(), "first pause flips state");
        assert!(gate.is_paused());
        assert!(!gate.pause(), "second pause is a no-op transition");
        assert!(gate.resume(), "first resume flips state");
        assert!(!gate.is_paused());
        assert!(!gate.resume(), "second resume is a no-op transition");
    }

    #[tokio::test]
    async fn wait_if_paused_returns_immediately_when_not_paused() {
        let gate = QueuePause::new();
        // Not paused → the await resolves without needing a resume.
        tokio::time::timeout(Duration::from_secs(1), gate.wait_if_paused())
            .await
            .expect("wait_if_paused must not block when the gate is open");
    }

    #[tokio::test]
    async fn wait_if_paused_blocks_until_resumed() {
        let gate = QueuePause::new();
        assert!(gate.pause());

        let waiter = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.wait_if_paused().await })
        };

        // While paused the waiter must stay parked.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "waiter must block while paused");

        // Resume wakes it via notify_waiters().
        assert!(gate.resume());
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter must unblock within the timeout after resume")
            .expect("waiter task must not panic");
    }

    #[tokio::test]
    async fn resume_wakes_every_gated_waiter() {
        // notify_waiters (not notify_one) so all dispatch loops proceed.
        let gate = QueuePause::new();
        assert!(gate.pause());

        let waiters: Vec<_> = (0..3)
            .map(|_| {
                let gate = gate.clone();
                tokio::spawn(async move { gate.wait_if_paused().await })
            })
            .collect();

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(gate.resume());

        for waiter in waiters {
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .expect("every gated waiter must wake on a single resume")
                .expect("waiter task must not panic");
        }
    }
}
