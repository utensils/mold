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
fn save_image_to_dir(
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
/// Mirrors `save_image_to_dir` for the video-output path.
///
/// When `gif_preview` is non-empty, also persists
/// `$MOLD_HOME/cache/previews/<filename>.preview.gif`. The gallery preview
/// endpoint (`GET /api/gallery/preview/:filename`) streams from that path
/// so remote TUI clients can animate the detail pane without re-fetching
/// the full MP4.
#[allow(clippy::too_many_arguments)]
fn save_video_to_dir(
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
fn save_video_preview_gif(filename: &str, gif_bytes: &[u8]) {
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
                let b64 = base64::engine::general_purpose::STANDARD;
                let event = if let Some(ref video) = response.video {
                    // Video response: encode the actual video data + metadata
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
                    // Image response: same as before
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
                };
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

        // Build the GpuJob once; the retry loop moves it between attempts.
        let mut gpu_job = Some(GpuJob {
            model: model_name.clone(),
            request: job.request,
            progress_tx: job.progress_tx,
            result_tx: job.result_tx,
            output_dir: job.output_dir,
            config: state.config.clone(),
            queue: state.queue.clone(),
        });

        let mut skip: Vec<usize> = Vec::new();
        let max_attempts = state.gpu_pool.worker_count().max(1);
        let mut dispatched = false;

        for _ in 0..max_attempts {
            let worker =
                match state
                    .gpu_pool
                    .select_worker_excluding(&model_name, estimated_vram, &skip)
                {
                    Some(w) => w,
                    None => break,
                };

            // Increment in-flight BEFORE sending to reserve the slot.
            worker.in_flight.fetch_add(1, Ordering::SeqCst);
            let pending = gpu_job.take().expect("gpu_job present in retry loop");
            match worker.job_tx.try_send(pending) {
                Ok(()) => {
                    dispatched = true;
                    break;
                }
                Err(std::sync::mpsc::TrySendError::Full(j))
                | Err(std::sync::mpsc::TrySendError::Disconnected(j)) => {
                    worker.in_flight.fetch_sub(1, Ordering::SeqCst);
                    tracing::warn!(
                        gpu = worker.gpu.ordinal,
                        "GPU worker channel full — retrying on another worker"
                    );
                    skip.push(worker.gpu.ordinal);
                    gpu_job = Some(j);
                }
            }
        }

        if !dispatched {
            // Either no workers are eligible or every candidate's channel is full.
            let rejected = gpu_job.expect("gpu_job retained after failed dispatch");
            let err_msg = if state.gpu_pool.worker_count() == 0 {
                format!("no GPU available for model {model_name}")
            } else {
                format!("all GPU workers are busy for model {model_name} — queue is full")
            };
            tracing::error!(model = %model_name, "{err_msg}");
            if let Some(tx) = rejected.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = rejected.result_tx.send(Err(err_msg));
            // Job was rejected before the worker could observe it, so we must
            // release the global queue slot here — the worker-side decrement
            // won't run.
            state.queue.decrement();
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
    use super::save_video_preview_gif_to;

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
}
