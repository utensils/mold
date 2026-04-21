use std::sync::Arc;

use base64::Engine as _;
use mold_core::{
    ImageData, OutputFormat, OutputMetadata, SseCompleteEvent, SseErrorEvent, SseProgressEvent,
};
use mold_db::{GenerationRecord, MetadataDb, RecordSource};
use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::gpu_pool::GpuJob;
use crate::model_manager;
use crate::state::{
    ActiveGenerationSnapshot, AppState, GenerationJob, GenerationJobResult, SseMessage,
};

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

/// Strips backtrace frames from candle error messages.
pub(crate) fn clean_error_message(e: &anyhow::Error) -> String {
    let full = format!("{e}");
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
    let started_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

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

/// Save an image to disk and (best-effort) record a row in the metadata DB.
///
/// Errors writing to disk are logged and skipped. DB errors are also logged
/// but do not fail the save — the file is the source of truth.
///
/// Shared between the legacy single-GPU `process_job` (this file) and the
/// per-GPU worker (`gpu_worker.rs`). Keep these on one helper so the DB
/// upsert can never silently regress on one path while the other keeps
/// working.
pub(crate) fn save_image_to_dir(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
    metadata: Option<&OutputMetadata>,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp_ms = now.as_millis() as u64;
    let ext = img.format.to_string();
    let filename =
        mold_core::default_output_filename(model, timestamp_ms, &ext, batch_size, img.index);
    let path = dir.join(&filename);
    match std::fs::write(&path, &img.data) {
        Ok(()) => tracing::info!("saved image to {}", path.display()),
        Err(e) => {
            tracing::warn!("failed to save image to {}: {e}", path.display());
            return;
        }
    }
    if let (Some(db), Some(meta)) = (db, metadata) {
        let mut rec = GenerationRecord::from_save(
            dir,
            filename,
            img.format,
            meta.clone(),
            RecordSource::Server,
            now.as_millis() as i64,
        );
        rec.stat_from_disk(&path);
        rec.generation_time_ms = generation_time_ms;
        rec.hostname = hostname_string();
        rec.backend = current_backend_label();
        if let Err(e) = db.upsert(&rec) {
            tracing::warn!("metadata DB upsert failed for {}: {e:#}", rec.filename);
        }
    }
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
) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let ts = now.as_millis() as u64;
    let ext = format.extension();
    let filename = mold_core::default_output_filename(model, ts, ext, 1, 0);
    let path = dir.join(&filename);
    if let Err(e) = std::fs::write(&path, bytes) {
        tracing::error!("failed to save video to {}: {e}", path.display());
        return;
    }
    if !gif_preview.is_empty() {
        save_video_preview_gif(&filename, gif_preview);
    }
    if let Some(db) = db {
        let mut rec = GenerationRecord::from_save(
            dir,
            filename,
            format,
            metadata.clone(),
            RecordSource::Server,
            now.as_millis() as i64,
        );
        rec.stat_from_disk(&path);
        rec.generation_time_ms = generation_time_ms;
        rec.hostname = hostname_string();
        rec.backend = current_backend_label();
        if let Err(e) = db.upsert(&rec) {
            tracing::warn!("metadata DB upsert failed for {}: {e:#}", rec.filename);
        }
    }
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
    let preview_path = preview_dir.join(format!("{filename}.preview.gif"));
    if let Err(e) = std::fs::write(&preview_path, gif_bytes) {
        tracing::warn!(
            "failed to write preview gif {}: {e}",
            preview_path.display()
        );
    }
}

/// Best-effort hostname for the `hostname` DB column. Falls back to `None`.
fn hostname_string() -> Option<String> {
    hostname::get().ok().and_then(|s| s.into_string().ok())
}

/// Compile-time backend label so DB rows say where the work happened.
fn current_backend_label() -> Option<String> {
    if cfg!(feature = "cuda") {
        Some("cuda".into())
    } else if cfg!(feature = "metal") {
        Some("metal".into())
    } else {
        Some("cpu".into())
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
) -> SseCompleteEvent {
    let b64 = base64::engine::general_purpose::STANDARD;
    if let Some(ref video) = response.video {
        SseCompleteEvent {
            image: b64.encode(&video.data),
            format: video.format,
            width: video.width,
            height: video.height,
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: Some(video.frames),
            video_fps: Some(video.fps),
            video_thumbnail: Some(b64.encode(&video.thumbnail)),
            video_gif_preview: if video.gif_preview.is_empty() {
                None
            } else {
                Some(b64.encode(&video.gif_preview))
            },
            video_has_audio: video.has_audio,
            video_duration_ms: video.duration_ms,
            video_audio_sample_rate: video.audio_sample_rate,
            video_audio_channels: video.audio_channels,
            gpu: response.gpu,
        }
    } else {
        SseCompleteEvent {
            image: b64.encode(&img.data),
            format: img.format,
            width: img.width,
            height: img.height,
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
        }
    }
}

/// Runs the generation queue worker loop. Processes one job at a time (FIFO).
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_worker(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    tracing::debug!("generation queue worker started");
    while let Some(job) = job_rx.recv().await {
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
        process_job(&state, job).await;
        state.queue.decrement();
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
    }
    tracing::info!("generation queue worker shutting down");
}

async fn process_job(state: &AppState, job: GenerationJob) {
    // Check if client already disconnected before doing any work
    if job.result_tx.is_closed() {
        tracing::debug!("skipping queued job — client disconnected");
        return;
    }

    // Send "now processing" event (position 0)
    if let Some(ref tx) = job.progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued {
            position: 0,
        }));
    }

    // 1. Ensure model is ready (with progress forwarding)
    let progress_callback = job.progress_tx.as_ref().map(|tx| {
        let tx = tx.clone();
        Arc::new(move |event: mold_inference::ProgressEvent| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }) as model_manager::EngineProgressCallback
    });

    if let Err(api_err) =
        model_manager::ensure_model_ready(state, &job.request.model, progress_callback).await
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

    // 3. Run inference in spawn_blocking
    let model_cache = state.model_cache.clone();
    let active_gen = state.active_generation.clone();
    let gen_state = state.clone();
    let gen_req = job.request.clone();
    let progress_tx = job.progress_tx.clone();

    #[cfg(feature = "metrics")]
    let inference_start = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = model_cache.blocking_lock();
            let entry = guard.get_mut(&gen_req.model).ok_or_else(|| {
                anyhow::anyhow!("no engine available after model readiness check")
            })?;
            let e = &mut entry.engine;
            set_active_generation(&gen_state, &gen_req.model, &gen_req.prompt);

            // Install progress callback for the generate phase
            if let Some(ref ptx) = progress_tx {
                let ptx = ptx.clone();
                e.set_on_progress(Box::new(move |event| {
                    let _ = ptx.send(SseMessage::Progress(progress_to_sse(event)));
                }));
            } else {
                e.clear_on_progress();
            }

            let generate_result = e.generate(&gen_req);
            if progress_tx.is_some() {
                e.clear_on_progress();
            }
            clear_active_generation(&gen_state);
            generate_result
        }))
    })
    .await;

    #[cfg(feature = "metrics")]
    let inference_duration = inference_start.elapsed().as_secs_f64();

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
            let img = if !response.images.is_empty() {
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

            // Save to output directory if configured.
            // Builds OutputMetadata from the request + the engine's actual
            // seed_used so the DB and embedded chunks agree.
            if let Some(ref dir) = job.output_dir {
                let dir = dir.clone();
                let model = job.request.model.clone();
                let batch_size = job.request.batch_size;
                let generation_time_ms = response.generation_time_ms as i64;
                let metadata = OutputMetadata::from_generate_request(
                    &job.request,
                    response.seed_used,
                    None,
                    mold_core::build_info::version_string(),
                );
                let db = state.metadata_db.clone();
                if let Some(ref video) = response.video {
                    let video_data = video.data.clone();
                    let video_gif_preview = video.gif_preview.clone();
                    let video_format = video.format;
                    let video_metadata = metadata.clone();
                    tokio::task::spawn_blocking(move || {
                        save_video_to_dir(
                            &dir,
                            &video_data,
                            &video_gif_preview,
                            video_format,
                            &model,
                            &video_metadata,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                        );
                    });
                } else {
                    let img_clone = img.clone();
                    let metadata_clone = metadata.clone();
                    tokio::task::spawn_blocking(move || {
                        save_image_to_dir(
                            &dir,
                            &img_clone,
                            &model,
                            batch_size,
                            Some(&metadata_clone),
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                        );
                    });
                }
            }

            // Send SSE complete event
            if let Some(ref tx) = job.progress_tx {
                let event = build_sse_complete_event(&response, &img);
                let _ = tx.send(SseMessage::Complete(event));
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
/// GPU worker based on the placement strategy (model-loaded > idle > evict LRU).
///
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_dispatcher(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    tracing::debug!("multi-GPU queue dispatcher started");
    while let Some(job) = job_rx.recv().await {
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());

        let model_name = job.request.model.clone();
        let estimated_vram = estimate_model_vram(&model_name);
        let preferred_gpu = match state
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
                #[cfg(feature = "metrics")]
                crate::metrics::record_queue_depth(state.queue.pending());
                continue;
            }
        };

        if job.result_tx.is_closed() {
            tracing::debug!(model = %model_name, "skipping queued multi-GPU job — client disconnected");
            state.queue.decrement();
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        // Build the GpuJob once; the retry loop moves it between attempts.
        let mut gpu_job = Some(GpuJob {
            model: model_name.clone(),
            request: job.request,
            progress_tx: job.progress_tx,
            result_tx: job.result_tx,
            output_dir: job.output_dir,
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            queue: state.queue.clone(),
        });

        let mut skip: Vec<usize> = Vec::new();
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
                break;
            };

            // Increment in-flight BEFORE sending to reserve the slot.
            worker.in_flight.fetch_add(1, Ordering::SeqCst);
            let pending = gpu_job.take().expect("gpu_job present in retry loop");
            match worker.job_tx.try_send(pending) {
                Ok(()) => {
                    dispatched = true;
                }
                Err(std::sync::mpsc::TrySendError::Full(j)) => {
                    worker.in_flight.fetch_sub(1, Ordering::SeqCst);
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
    if lower.contains(":q4") {
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
    use mold_core::{GenerateRequest, ImageData, OutputFormat};
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
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
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
            source_video: None,
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
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    #[test]
    fn save_image_to_dir_writes_file_and_creates_missing_dir() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("sub/output");
        assert!(!nested.exists());

        save_image_to_dir(&nested, &fake_image(), "flux-dev:q4", 1, None, None, None);

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

        save_image_to_dir(tmp.path(), &img, "sdxl", 4, None, None, None);

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
        );
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

        let event = build_sse_complete_event(&resp, &thumb_img);

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
        let event = build_sse_complete_event(&resp, &fake_image());
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
        let event = build_sse_complete_event(&resp, &fake_image());
        assert_eq!(event.format, OutputFormat::Png);
        assert!(event.video_frames.is_none());
        assert!(event.video_fps.is_none());
        assert!(event.video_thumbnail.is_none());
        assert!(event.video_gif_preview.is_none());
        assert!(!event.video_has_audio);
        assert!(event.video_duration_ms.is_none());
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
            model: "busy-model".to_string(),
            request: fake_request("busy-model"),
            progress_tx: None,
            result_tx: filler_result_tx,
            output_dir: None,
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            queue: state.queue.clone(),
        };
        worker.job_tx.send(filler_job).unwrap();

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            request: fake_request("flux-dev:q4"),
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
            request,
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
}
