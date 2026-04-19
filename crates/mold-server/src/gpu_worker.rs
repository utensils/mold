use crate::gpu_pool::{ActiveGeneration, GpuJob, GpuWorker};
use crate::model_cache::ModelResidency;
use crate::queue::clean_error_message;
use crate::state::{GenerationJobResult, SseMessage};
use base64::Engine as _;
use mold_core::{
    Config, ImageData, ModelPaths, OutputFormat, SseCompleteEvent, SseErrorEvent, SseProgressEvent,
};
use mold_inference::device;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Spawn the dedicated OS thread for a GPU worker.
/// Returns the JoinHandle (caller should keep it alive).
pub fn spawn_gpu_thread(
    worker: Arc<GpuWorker>,
    job_rx: std::sync::mpsc::Receiver<GpuJob>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("gpu-worker-{}", worker.gpu.ordinal))
        .spawn(move || {
            tracing::info!(
                gpu = worker.gpu.ordinal,
                name = %worker.gpu.name,
                "GPU worker thread started"
            );
            for job in job_rx.iter() {
                process_job(&worker, job);
            }
            tracing::info!(gpu = worker.gpu.ordinal, "GPU worker thread exiting");
        })
        .expect("failed to spawn GPU worker thread")
}

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

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
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
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

fn process_job(worker: &GpuWorker, job: GpuJob) {
    let model_name = job.model.clone();
    let ordinal = worker.gpu.ordinal;

    // Release the global queue slot when this job finishes, regardless of
    // which exit path runs. The dispatcher only decrements when it *fails* to
    // dispatch — once we own the GpuJob, we own the slot.
    struct QueueSlot(crate::state::QueueHandle);
    impl Drop for QueueSlot {
        fn drop(&mut self) {
            self.0.decrement();
        }
    }
    let _slot = QueueSlot(job.queue.clone());

    tracing::info!(gpu = ordinal, model = %model_name, "dispatched job");

    // Acquire per-GPU load lock — ensures only one model load at a time per GPU.
    let _load_lock = worker.model_load_lock.lock().unwrap();

    // Ensure model is loaded on this GPU.
    let config_snapshot = job.config.blocking_read().clone();
    if let Err(e) = ensure_model_ready_sync(worker, &model_name, &config_snapshot) {
        tracing::error!(gpu = ordinal, model = %model_name, "Failed to load model: {e}");
        let err_msg = format!("model load error: {}", clean_error_message(&e));
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        record_failure(worker);
        return;
    }

    // Set active generation state.
    {
        let mut gen = worker.active_generation.write().unwrap();
        *gen = Some(ActiveGeneration {
            model: model_name.clone(),
            started_at: Instant::now(),
        });
    }

    // Take-and-restore: remove engine from cache, release lock during inference.
    let taken = {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.take(&model_name)
    };

    let Some(mut cached_engine) = taken else {
        let err_msg = "engine not found in cache after load".to_string();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        clear_active_generation(worker);
        return;
    };

    // Set progress callback if SSE streaming.
    if let Some(ref progress_tx) = job.progress_tx {
        let tx = progress_tx.clone();
        cached_engine.engine.set_on_progress(Box::new(move |event| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }));
    }

    // Run inference — cache mutex is FREE during this.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cached_engine.engine.generate(&job.request)
    }));

    // Clear progress callback.
    cached_engine.engine.clear_on_progress();

    // Restore engine to cache.
    {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.restore(cached_engine);
    }

    // Clear active generation.
    clear_active_generation(worker);

    // Decrement in-flight.
    worker.in_flight.fetch_sub(1, Ordering::SeqCst);

    match result {
        Ok(Ok(mut response)) => {
            // Reset failure counter on success.
            worker.consecutive_failures.store(0, Ordering::SeqCst);

            // Attach GPU ordinal to response.
            response.gpu = Some(ordinal);

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

            // Extract the primary image (or video thumbnail).
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
            if let Some(ref dir) = job.output_dir {
                if let Some(ref video) = response.video {
                    let ext = video.format.extension().to_string();
                    let ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    let filename = mold_core::default_output_filename(&job.model, ts, &ext, 1, 0);
                    let path = dir.join(filename);
                    if let Err(e) = std::fs::write(&path, &video.data) {
                        tracing::error!("failed to save video to {}: {e}", path.display());
                    }
                } else {
                    save_image_to_dir(dir, &img, &job.model, job.request.batch_size);
                }
            }

            // Send SSE complete event.
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Complete(SseCompleteEvent {
                    image: base64::engine::general_purpose::STANDARD.encode(&img.data),
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
                }));
            }

            // Send result through oneshot.
            let _ = job.result_tx.send(Ok(GenerationJobResult {
                image: img,
                response,
            }));
        }
        Ok(Err(e)) => {
            tracing::warn!(gpu = ordinal, model = %model_name, "Generation failed: {e}");
            record_failure(worker);
            let err_msg = format!("generation error: {}", clean_error_message(&e));
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
        Err(panic_payload) => {
            tracing::error!(gpu = ordinal, model = %model_name, "Inference panicked");
            record_failure(worker);
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            let err_msg = format!("inference panicked: {msg}");
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
    }
}

/// Ensure a model is loaded on this worker's GPU.
///
/// Holds `worker.model_load_lock` implicitly via the caller for generation
/// jobs; the admin API path acquires it explicitly via `load_blocking`.
pub fn ensure_model_ready_sync(
    worker: &GpuWorker,
    model_name: &str,
    config: &Config,
) -> anyhow::Result<()> {
    let cache = worker.model_cache.lock().unwrap();

    // Already loaded?
    if let Some(entry) = cache.get(model_name) {
        if entry.residency == ModelResidency::Gpu {
            return Ok(());
        }
    }

    // Check if we have it cached but not on GPU (Parked/Unloaded).
    let has_cached = cache.contains(model_name);
    drop(cache);

    if has_cached {
        // Unload active model first.
        {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.unload_active();
        }
        device::reclaim_gpu_memory(worker.gpu.ordinal);

        // Take the engine out and reload it.
        let mut engine = {
            let mut cache = worker.model_cache.lock().unwrap();
            cache
                .remove(model_name)
                .ok_or_else(|| anyhow::anyhow!("cache race: model '{model_name}' vanished"))?
        };

        tracing::info!(
            gpu = worker.gpu.ordinal,
            model = %model_name,
            "reloading cached engine..."
        );
        engine.load()?;

        let vram = device::vram_used_estimate(worker.gpu.ordinal);
        let mut cache = worker.model_cache.lock().unwrap();
        cache.insert_loaded(model_name.to_string(), engine, vram);
        return Ok(());
    }

    // Not in cache — need to create from scratch.
    // Unload active model first.
    {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.unload_active();
    }
    device::reclaim_gpu_memory(worker.gpu.ordinal);

    // Resolve model paths.
    let paths = ModelPaths::resolve(model_name, config).ok_or_else(|| {
        anyhow::anyhow!("model '{model_name}' is not downloaded. Run: mold pull {model_name}")
    })?;

    let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
    let mut engine = mold_inference::create_engine_with_pool(
        model_name.to_string(),
        paths,
        config,
        mold_inference::LoadStrategy::Eager,
        worker.gpu.ordinal,
        offload,
        Some(worker.shared_pool.clone()),
    )?;

    tracing::info!(
        gpu = worker.gpu.ordinal,
        model = %model_name,
        "loading model..."
    );
    engine.load()?;

    let vram = device::vram_used_estimate(worker.gpu.ordinal);
    let mut cache = worker.model_cache.lock().unwrap();
    cache.insert_loaded(model_name.to_string(), engine, vram);

    Ok(())
}

/// Synchronously load a model on this GPU worker for the admin API.
///
/// Acquires the per-GPU load lock, then delegates to `ensure_model_ready_sync`.
/// Intended to be called inside `tokio::task::spawn_blocking`.
pub fn load_blocking(worker: &GpuWorker, model_name: &str, config: &Config) -> anyhow::Result<()> {
    let _lock = worker.model_load_lock.lock().unwrap();
    ensure_model_ready_sync(worker, model_name, config)
}

/// Synchronously unload the currently active model on this GPU worker.
///
/// Returns the name of the model that was unloaded, or `None` if the GPU was
/// already idle.
pub fn unload_blocking(worker: &GpuWorker) -> Option<String> {
    let _lock = worker.model_load_lock.lock().unwrap();
    let unloaded = {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.unload_active()
    };
    if unloaded.is_some() {
        device::reclaim_gpu_memory(worker.gpu.ordinal);
    }
    unloaded
}

fn record_failure(worker: &GpuWorker) {
    let failures = worker.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
    if failures >= 3 {
        let mut degraded = worker.degraded_until.write().unwrap();
        *degraded = Some(Instant::now() + Duration::from_secs(60));
        tracing::warn!(
            gpu = worker.gpu.ordinal,
            "GPU marked degraded after {failures} consecutive failures (60s cooldown)"
        );
    }
}

fn clear_active_generation(worker: &GpuWorker) {
    let mut gen = worker.active_generation.write().unwrap();
    *gen = None;
}
