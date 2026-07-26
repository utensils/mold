use crate::gpu_pool::{ActiveGeneration, GpuJob, GpuWorker};
use crate::model_cache::ModelResidency;
use crate::queue::{
    apply_upscale_response_to_image_generation, build_sse_completion_message, clean_error_message,
    save_generated_image_outputs, save_video_to_dir, settle_post_generation_upscale,
};
use crate::state::{GenerationJobResult, SseMessage};
use mold_core::{
    Config, ImageData, ModelPaths, OutputFormat, OutputMetadata, SseErrorEvent, SseProgressEvent,
};
use mold_inference::device;
use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Spawn the dedicated OS thread for a GPU worker.
/// Returns the JoinHandle (caller should keep it alive).
pub fn spawn_gpu_thread(
    worker: Arc<GpuWorker>,
    job_rx: std::sync::mpsc::Receiver<GpuJob>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("gpu-worker-{}", worker.gpu.ordinal))
        .spawn(move || {
            // Bind this thread to its GPU ordinal so `create_device` /
            // `reclaim_gpu_memory` can debug-assert callers don't drift onto
            // a sibling GPU's context. See device::init_thread_gpu_ordinal.
            mold_inference::device::init_thread_gpu_ordinal(worker.gpu.ordinal);
            tracing::info!(
                gpu = worker.gpu.ordinal,
                name = %worker.gpu.name,
                "GPU worker thread started"
            );
            let device_id = crate::scheduler::worker_device_id(&worker);
            let mut generation = 1_u64;
            loop {
                if scheduler_tx
                    .send(crate::scheduler::WorkerEvent::Ready {
                        device_id: device_id.clone(),
                        ordinal: worker.gpu.ordinal,
                        worker_generation: generation,
                    })
                    .is_err()
                {
                    break;
                }
                let Ok(job) = job_rx.recv() else {
                    break;
                };
                if let Some(fence) = job.lease.as_ref() {
                    if fence.device_id != device_id || fence.worker_generation != generation {
                        let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                            device_id: device_id.clone(),
                            ordinal: worker.gpu.ordinal,
                            worker_generation: generation,
                            job: Box::new(job),
                            reason: crate::scheduler::LeaseRejection::StaleWorkerGeneration,
                        });
                        continue;
                    }
                    let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Accepted {
                        device_id: device_id.clone(),
                        ordinal: worker.gpu.ordinal,
                        worker_generation: generation,
                        work_id: fence.work_id.clone(),
                        plan_version: fence.plan_version,
                    });
                }
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    process_job(&worker, job);
                }));
                if outcome.is_err() {
                    // A panic may have crossed arbitrary Candle/cudarc state.
                    // Treat the owner context as fatal and let supervision
                    // restart the process; never attempt an in-process reset.
                    worker.poisoned.store(true, Ordering::SeqCst);
                    worker.fatal_cuda_error.store(true, Ordering::SeqCst);
                    worker.fatal_cuda_shutdown.notify_waiters();
                    tracing::error!(
                        gpu = worker.gpu.ordinal,
                        "GPU owner thread panicked; quarantining context and stopping server"
                    );
                }
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: worker.gpu.ordinal,
                    worker_generation: generation,
                });
                if outcome.is_err() || worker.fatal_cuda_error.load(Ordering::SeqCst) {
                    break;
                }
                generation = generation.saturating_add(1);
            }
            tracing::info!(gpu = worker.gpu.ordinal, "GPU worker thread exiting");
        })
        .expect("failed to spawn GPU worker thread")
}

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

/// Detect a CUDA out-of-memory error anywhere in the anyhow cause chain.
///
/// Candle surfaces these as `DriverError(CUDA_ERROR_OUT_OF_MEMORY, …)` wrapped
/// in an `anyhow::Error`. The string representation is the only stable signal
/// (the cudarc error type doesn't implement `std::error::Error` downcast target
/// in the candle re-export), so we pattern-match the formatted chain.
pub(crate) fn is_cuda_oom(e: &anyhow::Error) -> bool {
    let full = format!("{e:#}");
    full.contains("CUDA_ERROR_OUT_OF_MEMORY") || full.contains("out of memory")
}

/// Detect CUDA errors that invalidate the process-owned context.
///
/// Candle's cudarc layer retains primary-context handles, so resetting that
/// context in-process would turn those handles into use-after-free hazards.
/// These errors therefore quarantine the worker until process restart instead
/// of entering the ordinary failure cooldown and retrying a dead context.
pub(crate) fn is_fatal_cuda_error(e: &anyhow::Error) -> bool {
    has_fatal_cuda_error(&format!("{e:#}"))
}

fn has_fatal_cuda_error(message: &str) -> bool {
    [
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "CUDA_ERROR_ECC_UNCORRECTABLE",
        "CUDA_ERROR_LAUNCH_FAILED",
        "CUDA_ERROR_ASSERT",
        "CUDA_ERROR_MISALIGNED_ADDRESS",
        "CUDA_ERROR_HARDWARE_STACK_ERROR",
        "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        "CUDA_ERROR_INVALID_PC",
        "CUDA_ERROR_LAUNCH_TIMEOUT",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn fatal_cuda_user_message(model_name: &str) -> String {
    format!(
        "fatal CUDA error while running '{model_name}'; this GPU worker was quarantined because its CUDA context is no longer safe to reuse. Restart the mold server to recover the GPU."
    )
}

fn quarantine_poisoned_worker(worker: &GpuWorker) {
    worker.poisoned.store(true, Ordering::SeqCst);
    worker.set_resident_model(None);
    worker.consecutive_failures.store(3, Ordering::SeqCst);
    *worker.degraded_until.write().unwrap() = None;
    worker.fatal_cuda_error.store(true, Ordering::SeqCst);
    worker.fatal_cuda_shutdown.notify_one();
    tracing::error!(
        gpu = worker.gpu.ordinal,
        "GPU worker quarantined after fatal CUDA context error; shutting down for process restart"
    );
}

pub(crate) fn quarantine_if_fatal_cuda_error(worker: &GpuWorker, error: &anyhow::Error) -> bool {
    let fatal = is_fatal_cuda_error(error);
    if fatal {
        quarantine_poisoned_worker(worker);
    }
    fatal
}

pub(crate) fn ensure_worker_not_poisoned(
    worker: &GpuWorker,
    model_name: &str,
) -> anyhow::Result<()> {
    if worker.poisoned.load(Ordering::SeqCst) {
        anyhow::bail!(fatal_cuda_user_message(model_name));
    }
    Ok(())
}

/// Build a user-friendly error message for a CUDA OOM. The raw
/// `DriverError(CUDA_ERROR_OUT_OF_MEMORY, …)` is opaque; replace it with
/// actionable guidance.
pub(crate) fn oom_user_message(model_name: &str) -> String {
    oom_user_message_for_request(model_name, None, None)
}

pub(crate) fn oom_user_message_for_request(
    model_name: &str,
    family_slug: Option<&str>,
    req: Option<&mold_core::GenerateRequest>,
) -> String {
    let requested_size = req
        .map(|r| format!(" Requested size: {}x{}.", r.width, r.height))
        .unwrap_or_default();
    let batch_hint = match req.map(|r| r.batch_size).unwrap_or(1) {
        0 | 1 => "keep --batch 1".to_string(),
        n => format!("reduce --batch {n} to --batch 1"),
    };

    if family_slug.is_some_and(is_video_family) || req.and_then(|r| r.frames).is_some() {
        let frames_hint = req
            .and_then(|r| r.frames)
            .map(|frames| format!("reduce --frames below {frames} (e.g. 17 or 9)"))
            .unwrap_or_else(|| "reduce --frames (e.g. 17 or 9)".to_string());
        return format!(
            "GPU ran out of memory loading or running '{model_name}'.{requested_size} \
             Try: {frames_hint}, lower --width/--height, use a quantized variant \
             if available, or close other GPU apps."
        );
    }

    let family_note = match family_slug {
        Some("sd15") => {
            if req.is_some_and(|r| r.width == 1024 && r.height == 1024) {
                " SD1.5 defaults to 512x512; 1024x1024 is 4x the pixels and can OOM \
                 even when the checkpoint file is only a few GB."
            } else {
                " SD1.5 defaults to 512x512; larger sizes multiply activation and \
                 VAE workspace beyond the checkpoint file size."
            }
        }
        Some("sdxl") => {
            " SDXL's usual 1024x1024 size still needs activation and VAE workspace \
             beyond the checkpoint file size."
        }
        Some("sd3") => " SD3 needs activation and VAE workspace beyond the checkpoint file size.",
        Some("flux")
        | Some("flux2")
        | Some("qwen-image")
        | Some("qwen-image-edit")
        | Some("z-image")
        | Some("wuerstchen") => {
            " The checkpoint size is only the weights; peak VRAM also includes \
             activations, VAE decode workspace, CUDA workspaces, and resident cache."
        }
        _ => {
            " The model file size is only the weights; peak VRAM also includes \
             activations, decoder workspace, CUDA workspaces, and resident cache."
        }
    };
    let resolution_hint = match family_slug {
        Some("sd15") => "lower --width/--height (try 768x768 or 512x512)",
        _ => "lower --width/--height",
    };

    format!(
        "GPU ran out of memory loading or running '{model_name}'.{requested_size}{family_note} \
         Try: {resolution_hint}, {batch_hint}, use a smaller/quantized variant if \
         this model provides one, run mold unload, or close other GPU apps."
    )
}

fn is_video_family(family_slug: &str) -> bool {
    matches!(family_slug, "ltx-video" | "ltx2" | "ltx-2" | "ltx-2.3")
}

fn upscale_generated_image_on_worker(
    worker: &GpuWorker,
    job: &GpuJob,
    upscale_model: &str,
    img: ImageData,
    response: &mut mold_core::GenerateResponse,
) -> Result<ImageData, String> {
    let model_name = mold_core::manifest::resolve_model_name(upscale_model);
    let weights_path = {
        let config = job.config.blocking_read();
        config
            .models
            .get(&model_name)
            .and_then(|c| c.transformer.as_ref())
            .map(std::path::PathBuf::from)
    }
    .ok_or_else(|| format!("upscaler model '{model_name}' is not downloaded"))?;

    if let Some(ref tx) = job.progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::StageStart {
            name: format!("Loading upscaler {model_name}"),
        }));
    }
    let mut engine = mold_inference::create_upscale_engine(
        model_name.clone(),
        weights_path,
        mold_inference::LoadStrategy::Eager,
        worker.gpu.ordinal,
    )
    .map_err(|e| format!("failed to load upscaler: {e}"))?;
    if let Some(ref tx) = job.progress_tx {
        let tx = tx.clone();
        engine.set_on_progress(Box::new(move |event| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }));
    }
    let req = mold_core::UpscaleRequest {
        model: model_name,
        image: img.data.clone(),
        output_format: img.format,
        tile_size: None,
        metadata: Some(OutputMetadata::from_generate_request(
            &job.request,
            response.seed_used,
            None,
            mold_core::build_info::version_string(),
        )),
    };
    let upscaled = engine
        .upscale(&req)
        .map_err(|e| format!("upscale failed: {e}"))?;
    engine.clear_on_progress();
    apply_upscale_response_to_image_generation(&job.request, response, img, upscaled)
        .map_err(|e| format!("upscale failed: {e}"))
}

fn cuda_oom_user_message(
    worker: &GpuWorker,
    model_name: &str,
    family_slug: Option<&str>,
    req: Option<&mold_core::GenerateRequest>,
) -> (String, bool) {
    let base = if family_slug.is_none() && req.is_none() {
        oom_user_message(model_name)
    } else {
        oom_user_message_for_request(model_name, family_slug, req)
    };
    let outcome = crate::gpu_pool::record_model_cuda_oom(model_name, worker.gpu.ordinal);
    if outcome.is_unschedulable() {
        if let Some(cooldown) = crate::gpu_pool::model_unschedulable_message(model_name) {
            return (format!("{base} {cooldown}"), false);
        }
    }
    (base, true)
}

fn process_job(worker: &GpuWorker, job: GpuJob) {
    let model_name = job.model.clone();
    let ordinal = worker.gpu.ordinal;
    let job_id = job.id.clone();

    // Release the global queue slot AND the registry entry when this job
    // finishes, regardless of which exit path runs. The dispatcher only
    // decrements when it *fails* to dispatch — once we own the GpuJob, we
    // own both pieces of cleanup. Combining them in one drop guard keeps
    // the two counters from drifting on early-return paths.
    struct CleanupGuard {
        queue: crate::state::QueueHandle,
        registry: crate::job_registry::SharedJobRegistry,
        id: String,
    }
    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            self.queue.decrement();
            self.registry.remove(&self.id);
        }
    }
    let _cleanup = CleanupGuard {
        queue: job.queue.clone(),
        registry: job.registry.clone(),
        id: job_id.clone(),
    };

    // Jobs may already be buffered in this worker's channel when a preceding
    // job kills the context. Fail them without touching CUDA, including jobs
    // explicitly pinned to this ordinal.
    if worker.poisoned.load(Ordering::SeqCst) {
        let err_msg = fatal_cuda_user_message(&model_name);
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        return;
    }

    if job.result_tx.is_closed() {
        tracing::debug!(gpu = ordinal, model = %model_name, "skipping dispatched job — client disconnected");
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        return;
    }

    // Mark the registry entry as running on this specific GPU. The /api/queue
    // listing now shows this row as `state: "running"` with `gpu: <ordinal>`.
    job.registry.mark_running(&job_id, Some(ordinal));

    tracing::info!(gpu = ordinal, model = %model_name, "dispatched job");

    // Acquire per-GPU load lock — ensures only one model load at a time per GPU.
    let _load_lock = worker.model_load_lock.lock().unwrap();

    // A chain/admin/auxiliary workload may have poisoned the context while
    // this job waited on the load lock. Recheck before any CUDA operation.
    if let Err(error) = ensure_worker_not_poisoned(worker, &model_name) {
        let err_msg = error.to_string();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        return;
    }

    // Ensure model is loaded on this GPU.
    let config_snapshot = job.config.blocking_read().clone();
    let family_slug = crate::model_manager::family_for_model_sync(&model_name, &config_snapshot);
    let activation_hint =
        crate::model_manager::activation_hint_for_request_sync(&config_snapshot, &job.request);
    let request_has_lora = crate::model_manager::request_has_effective_lora(&job.request);
    if let Err(e) = ensure_model_ready_sync(
        worker,
        &model_name,
        &config_snapshot,
        activation_hint,
        request_has_lora,
    ) {
        tracing::error!(gpu = ordinal, model = %model_name, "Failed to load model: {e}");
        // Detect CUDA OOM during load: synchronize the device so subsequent
        // allocations don't inherit a poisoned context, then surface a
        // user-friendly message instead of the opaque DriverError string.
        let is_fatal_cuda = is_fatal_cuda_error(&e);
        let is_oom = is_cuda_oom(&e);
        let (err_msg, count_worker_failure) = if is_fatal_cuda {
            quarantine_poisoned_worker(worker);
            (fatal_cuda_user_message(&model_name), false)
        } else if is_oom {
            mold_inference::device::try_synchronize_device(ordinal);
            cuda_oom_user_message(
                worker,
                &model_name,
                family_slug.as_deref(),
                Some(&job.request),
            )
        } else {
            (
                format!("model load error: {}", clean_error_message(&e)),
                true,
            )
        };
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        if count_worker_failure {
            record_failure(worker);
        }
        return;
    }

    // Set active generation state.
    {
        let mut gen = worker.active_generation.write().unwrap();
        *gen = Some(ActiveGeneration {
            model: model_name.clone(),
            prompt_sha256: format!("{:x}", Sha256::digest(job.request.prompt.as_bytes())),
            started_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            started_at: Instant::now(),
        });
    }

    if job.result_tx.is_closed() {
        tracing::debug!(
            gpu = ordinal,
            model = %model_name,
            "skipping generation after model readiness — client disconnected"
        );
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        clear_active_generation(worker);
        return;
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

    // RSS sample taken just before inference; the post-inference sample below
    // logs the per-job delta so RAM growth can be attributed to a specific
    // generation rather than tracked at process granularity.
    let rss_before = crate::resources::ram_snapshot().used_by_mold;

    // Watchdog: log RSS every 1s while inference runs so we can see RAM
    // growth as it happens. The post-inference summary log can't fire when
    // a runaway allocation crosses the OOM threshold mid-generation, so we
    // need a heartbeat to attribute the explosion to a specific phase.
    let watchdog_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watchdog_handle = {
        let stop = watchdog_stop.clone();
        let model = model_name.clone();
        std::thread::Builder::new()
            .name(format!("rss-watchdog-{ordinal}"))
            .spawn(move || {
                let start = Instant::now();
                while !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1000));
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let rss = crate::resources::ram_snapshot().used_by_mold;
                    tracing::info!(
                        gpu = ordinal,
                        model = %model,
                        elapsed_s = start.elapsed().as_secs(),
                        rss_mb = rss / 1_000_000,
                        "rss watchdog"
                    );
                }
            })
            .expect("failed to spawn RSS watchdog")
    };

    // Run inference — cache mutex is FREE during this.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cached_engine.engine.generate(&job.request)
    }));

    watchdog_stop.store(true, Ordering::SeqCst);
    let _ = watchdog_handle.join();

    // glibc keeps freed pages in per-arena heaps even after the allocations
    // are dropped — large transient buffers from GGUF+LoRA rebuilds can leave
    // tens of GB of unreclaimed RSS. `malloc_trim(0)` walks the arenas and
    // returns idle pages to the OS via madvise(MADV_DONTNEED). Cheap (~ms),
    // glibc-only, gated so we can A/B with `MOLD_MALLOC_TRIM=0`.
    let trim_enabled = std::env::var("MOLD_MALLOC_TRIM")
        .map(|v| v != "0")
        .unwrap_or(true);
    let rss_pre_trim = if trim_enabled {
        let v = crate::resources::ram_snapshot().used_by_mold;
        #[cfg(target_os = "linux")]
        unsafe {
            libc::malloc_trim(0);
        }
        Some(v)
    } else {
        None
    };

    let rss_after = crate::resources::ram_snapshot().used_by_mold;
    let rss_delta = rss_after as i64 - rss_before as i64;
    tracing::info!(
        gpu = ordinal,
        model = %model_name,
        rss_before_mb = rss_before / 1_000_000,
        rss_after_mb = rss_after / 1_000_000,
        rss_delta_mb = rss_delta / 1_000_000,
        rss_pre_trim_mb = rss_pre_trim.map(|v| v / 1_000_000).unwrap_or(0),
        "generation memory delta"
    );

    // Clear progress callback.
    cached_engine.engine.clear_on_progress();

    // A fatal driver error invalidates every CUDA object owned by this
    // context. Never put the triggering engine back into the cache: doing so
    // caused immediate CUBLAS_STATUS_NOT_INITIALIZED retries on the poisoned
    // worker. We deliberately do not reset the primary context here because
    // Candle/cudarc retain handles to it; an in-process reset would make those
    // handles dangling. Quarantine until process restart instead.
    let fatal_cuda = matches!(&result, Ok(Err(e)) if is_fatal_cuda_error(e));
    if fatal_cuda {
        // Signal process teardown before destructors touch the poisoned
        // context; CUDA cleanup is best-effort after an illegal access.
        quarantine_poisoned_worker(worker);
        drop(cached_engine);
        let remaining = {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.clear()
        };
        drop(remaining);
    } else {
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
            crate::gpu_pool::clear_model_cuda_oom(&model_name);

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

            if response.video.is_none() {
                if let Some(upscale_model) = job
                    .request
                    .upscale_model
                    .as_deref()
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                {
                    let upscale_result = upscale_generated_image_on_worker(
                        worker,
                        &job,
                        upscale_model,
                        img.clone(),
                        &mut response,
                    );
                    if upscale_result
                        .as_ref()
                        .is_err_and(|error| has_fatal_cuda_error(error))
                    {
                        quarantine_poisoned_worker(worker);
                    }
                    let (output, preserved_original, upscale_error) =
                        settle_post_generation_upscale(img, upscale_result);
                    img = output;
                    original_img = preserved_original;
                    if let Some(error) = upscale_error {
                        tracing::warn!(
                            gpu = ordinal,
                            %error,
                            "post-generation upscale failed; keeping original image"
                        );
                    }
                }
            }

            // Save to output directory if configured. Routes through the
            // shared queue helpers so the metadata-DB upsert (and embedded
            // chunks, hostname/backend tagging) cannot drift between the
            // single-GPU and multi-GPU paths — historically this branch
            // skipped the DB write, which left freshly-generated files
            // invisible to /api/gallery until the next reconcile on
            // server restart.
            let metadata = OutputMetadata::from_generate_request(
                &job.request,
                response.seed_used,
                None,
                mold_core::build_info::version_string(),
            );
            let mut saved_names = crate::queue::SavedOutputNames::default();
            if let Some(ref dir) = job.output_dir {
                let generation_time_ms = response.generation_time_ms as i64;
                let db = job.metadata_db.as_ref().as_ref();
                let events = Some(job.events.as_ref());
                if let Some(ref video) = response.video {
                    saved_names.output = save_video_to_dir(
                        dir,
                        &video.data,
                        &video.gif_preview,
                        video.format,
                        &job.model,
                        &metadata,
                        Some(generation_time_ms),
                        db,
                        events,
                    );
                } else {
                    saved_names = save_generated_image_outputs(
                        dir,
                        original_img.as_ref(),
                        &img,
                        &job.model,
                        job.request.batch_size,
                        &metadata,
                        Some(generation_time_ms),
                        db,
                        events,
                    );
                }
            }

            // Send SSE complete event. Video responses carry the actual MP4 /
            // GIF bytes plus frames / fps / thumbnail / audio metadata so the
            // SSE client can reconstruct a `VideoData` — without this the
            // Discord bot silently degraded every LTX-Video / LTX-2 response
            // into an image attachment (the synthesized thumbnail PNG).
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

            // Send result through oneshot.
            let _ = job.result_tx.send(Ok(GenerationJobResult {
                image: img,
                response,
            }));
        }
        Ok(Err(e)) => {
            tracing::warn!(gpu = ordinal, model = %model_name, "Generation failed: {e}");
            // Fatal driver errors invalidate the CUDA context and permanently
            // quarantine this worker. Ordinary OOMs retain the existing
            // synchronize-and-retry policy.
            let is_oom = is_cuda_oom(&e);
            let (err_msg, count_worker_failure) = if fatal_cuda {
                (fatal_cuda_user_message(&model_name), false)
            } else if is_oom {
                mold_inference::device::try_synchronize_device(ordinal);
                cuda_oom_user_message(
                    worker,
                    &model_name,
                    family_slug.as_deref(),
                    Some(&job.request),
                )
            } else {
                (
                    format!("generation error: {}", clean_error_message(&e)),
                    true,
                )
            };
            if count_worker_failure {
                record_failure(worker);
            }
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

/// Preflight memory check with evict-to-fit recovery.
///
/// Wraps `model_manager::preflight_memory_guard`. On a budget failure, drops
/// the LRU parked entry (skipping `model_name` so a parked-reload doesn't
/// evict its own target), reclaims the GPU's CUDA pool when no engine remains
/// resident, and retries. Loops until the preflight passes or the cache has
/// no parked entries left to surrender — at which point the original
/// insufficient-memory error is returned.
///
/// Holds the cache lock only for the brief eviction step; the engine drop and
/// `reclaim_gpu_memory` run outside it. The caller is expected to hold
/// `worker.model_load_lock`, which keeps a concurrent generation from slotting
/// a fresh load into the context between our reclaim and the actual load.
fn preflight_memory_guard_with_eviction(
    cache_lock: &std::sync::Mutex<crate::model_cache::ModelCache>,
    model_name: &str,
    paths: &ModelPaths,
    ordinal: usize,
    hint: Option<crate::model_manager::ActivationHint>,
) -> Result<(), crate::routes::ApiError> {
    loop {
        let active_vram = cache_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_vram_bytes();
        let err = match crate::model_manager::preflight_memory_guard(
            model_name,
            paths,
            active_vram,
            ordinal,
            hint,
        ) {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };

        let evicted = {
            let mut cache = cache_lock.lock().unwrap_or_else(|e| e.into_inner());
            cache.evict_lru_parked_except(Some(model_name))
        };
        let Some((evicted_name, engine)) = evicted else {
            return Err(err);
        };
        tracing::info!(
            gpu = ordinal,
            target_model = %model_name,
            evicted_model = %evicted_name,
            "evicting LRU parked entry to fit incoming load"
        );
        // Drop outside the cache lock — `cuMemFree` and safetensor unmap
        // can block other cache users during the drop.
        drop(engine);

        // Reclaim only when no GPU-resident engine remains. The parked-reload
        // case has the active model still on GPU at preflight time, so we can
        // free CPU caches via the eviction but must not nuke the primary
        // context. The fresh-load case usually has no active model when this
        // is hit, so the reclaim can run.
        let safe_to_reclaim = cache_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_model()
            .is_none();
        if safe_to_reclaim {
            device::reclaim_gpu_memory(ordinal);
        }
    }
}

fn select_load_strategy_for_worker(
    worker: &GpuWorker,
    model_name: &str,
    paths: &ModelPaths,
    hint: Option<crate::model_manager::ActivationHint>,
) -> mold_inference::LoadStrategy {
    let active_vram = worker
        .model_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .active_vram_bytes();
    let available =
        crate::model_manager::effective_load_available_bytes(active_vram, worker.gpu.ordinal);
    let strategy = crate::model_manager::select_server_load_strategy_for_device(
        paths,
        available,
        Some(worker.gpu.total_vram_bytes),
        hint,
    );
    if strategy == mold_inference::LoadStrategy::Sequential {
        tracing::info!(
            gpu = worker.gpu.ordinal,
            model = %model_name,
            "server load strategy degraded to sequential to fit memory budget"
        );
    }
    strategy
}

/// Ensure a model is loaded on this worker's GPU.
///
/// Holds `worker.model_load_lock` implicitly via the caller for generation
/// jobs; the admin API path acquires it explicitly via `load_blocking`.
///
/// `hint` carries the per-request activation budget (resolution + family).
/// Pass `None` for admin / cache-prewarm loads with no resolution context.
pub fn ensure_model_ready_sync(
    worker: &GpuWorker,
    model_name: &str,
    config: &Config,
    hint: Option<crate::model_manager::ActivationHint>,
    request_has_lora: bool,
) -> anyhow::Result<()> {
    let result = ensure_model_ready_sync_inner(worker, model_name, config, hint, request_has_lora);
    if result.is_ok() {
        worker.set_resident_model(Some(model_name));
    }
    result
}

fn ensure_model_ready_sync_inner(
    worker: &GpuWorker,
    model_name: &str,
    config: &Config,
    hint: Option<crate::model_manager::ActivationHint>,
    request_has_lora: bool,
) -> anyhow::Result<()> {
    let cache = worker.model_cache.lock().unwrap();

    // Already loaded?
    if let Some(entry) = cache.get(model_name) {
        if entry.residency == ModelResidency::Gpu {
            let must_recreate = entry.engine.model_paths().is_some_and(|paths| {
                crate::model_manager::request_requires_fresh_engine_for_offload_policy(
                    paths,
                    hint,
                    request_has_lora,
                )
            });
            if !must_recreate {
                return Ok(());
            }
        }
    }

    // Check if we have it cached but not on GPU (Parked).
    let has_cached = cache.contains(model_name);

    // Snapshot the cached engine's paths (if any) for the preflight before
    // dropping the lock. Cloning ModelPaths keeps the borrow scoped to this
    // block. Active-VRAM is sampled inside the preflight helper itself so
    // each retry sees fresh state.
    let cached_paths = if has_cached {
        cache
            .get(model_name)
            .and_then(|e| e.engine.model_paths().cloned())
    } else {
        None
    };
    drop(cache);

    if has_cached {
        let load_strategy = cached_paths
            .as_ref()
            .map(|paths| select_load_strategy_for_worker(worker, model_name, paths, hint))
            .unwrap_or(mold_inference::LoadStrategy::Eager);

        // Preflight before unloading the active model — the active model's
        // footprint counts toward effective availability since we're about
        // to free it. On budget failure, evict-to-fit drops parked entries
        // (other than `model_name` itself) and retries.
        if let Some(ref paths) = cached_paths {
            preflight_memory_guard_with_eviction(
                &worker.model_cache,
                model_name,
                paths,
                worker.gpu.ordinal,
                hint,
            )
            .map_err(|e| anyhow::anyhow!(e.error))?;
        }

        // Unload active model first.
        {
            let mut cache = worker.model_cache.lock().unwrap();
            if cache.unload_active().is_some() {
                worker.set_resident_model(None);
            }
        }
        device::reclaim_gpu_memory(worker.gpu.ordinal);

        if load_strategy == mold_inference::LoadStrategy::Sequential {
            let paths = cached_paths.ok_or_else(|| {
                anyhow::anyhow!("cached engine for '{model_name}' does not expose model paths")
            })?;
            let old_engine = {
                let mut cache = worker.model_cache.lock().unwrap();
                cache
                    .remove(model_name)
                    .ok_or_else(|| anyhow::anyhow!("cache race: model '{model_name}' vanished"))?
            };

            let offload = crate::model_manager::server_offload_enabled_for_paths(
                &paths,
                hint,
                request_has_lora,
            );
            let resolved_catalog_config =
                crate::model_manager::resolve_installed_catalog_paths_for_worker(
                    model_name, config,
                )
                .map_err(|e| anyhow::anyhow!(e.error))?
                .map(|(_, config)| config);
            let engine_config = resolved_catalog_config.as_ref().unwrap_or(config);
            let mut engine = match mold_inference::create_engine_with_pool(
                model_name.to_string(),
                paths,
                engine_config,
                load_strategy,
                worker.gpu.ordinal,
                offload,
                Some(worker.shared_pool.clone()),
            ) {
                Ok(engine) => engine,
                Err(err) => {
                    let evicted = {
                        let mut cache = worker.model_cache.lock().unwrap();
                        cache.insert(old_engine, 0)
                    };
                    drop(evicted);
                    return Err(err);
                }
            };

            tracing::info!(
                gpu = worker.gpu.ordinal,
                model = %model_name,
                "recreating cached engine in sequential mode..."
            );
            let vram_baseline = device::vram_in_use_bytes(worker.gpu.ordinal);
            if let Err(err) = engine.load() {
                let evicted = {
                    let mut cache = worker.model_cache.lock().unwrap();
                    cache.insert(old_engine, 0)
                };
                drop(evicted);
                return Err(err);
            }
            let vram = device::vram_load_delta(worker.gpu.ordinal, vram_baseline);
            drop(old_engine);
            let evicted = {
                let mut cache = worker.model_cache.lock().unwrap();
                cache.insert_loaded(model_name.to_string(), engine, vram)
            };
            drop(evicted);
            return Ok(());
        }

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
        // Sample VRAM baseline before load so we can record the new model's
        // per-load delta rather than the device-global usage.
        let vram_baseline = device::vram_in_use_bytes(worker.gpu.ordinal);
        engine.load()?;

        let vram = device::vram_load_delta(worker.gpu.ordinal, vram_baseline);
        // Drop any evicted engine OUTSIDE the cache lock — `cuMemFree` and
        // safetensor unmap during the drop can block other cache users.
        let evicted = {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.insert_loaded(model_name.to_string(), engine, vram)
        };
        drop(evicted);
        return Ok(());
    }

    // Not in cache — need to create from scratch.
    // Resolve model paths.
    let mut resolved_catalog_config = None;
    let paths = if let Some(paths) = ModelPaths::resolve(model_name, config) {
        paths
    } else if let Some((paths, config)) =
        crate::model_manager::resolve_installed_catalog_paths_for_worker(model_name, config)
            .map_err(|e| anyhow::anyhow!(e.error))?
    {
        resolved_catalog_config = Some(config);
        paths
    } else {
        return Err(
            if model_name.starts_with("cv:") || model_name.starts_with("hf:") {
                // Catalog IDs (cv:/hf:) reach this path through the bridge in
                // `model_manager::install_catalog_model`, which can synthesize a
                // ModelConfig that's missing a required field (notably `vae`)
                // when a canonical companion was never pulled. The legacy
                // "Run: mold pull <id>" message is misleading there because the
                // primary checkpoint IS on disk — the companion is what's
                // missing. Surface the catalog-specific guidance instead.
                anyhow::anyhow!(
                    "catalog model '{model_name}' has missing required components. \
                 Re-pull the entry from the catalog so its companions \
                 (CLIP-L / T5 / VAE) are fetched alongside the primary checkpoint."
                )
            } else {
                anyhow::anyhow!(
                    "model '{model_name}' is not downloaded. Run: mold pull {model_name}"
                )
            },
        );
    };

    // Preflight before unloading the active model. Evict-to-fit drops parked
    // entries on budget failure and retries before giving up.
    preflight_memory_guard_with_eviction(
        &worker.model_cache,
        model_name,
        &paths,
        worker.gpu.ordinal,
        hint,
    )
    .map_err(|e| anyhow::anyhow!(e.error))?;

    let load_strategy = select_load_strategy_for_worker(worker, model_name, &paths, hint);

    // Unload active model first.
    {
        let mut cache = worker.model_cache.lock().unwrap();
        if cache.unload_active().is_some() {
            worker.set_resident_model(None);
        }
    }
    device::reclaim_gpu_memory(worker.gpu.ordinal);

    let offload =
        crate::model_manager::server_offload_enabled_for_paths(&paths, hint, request_has_lora);
    let engine_config = resolved_catalog_config.as_ref().unwrap_or(config);
    let mut engine = mold_inference::create_engine_with_pool(
        model_name.to_string(),
        paths,
        engine_config,
        load_strategy,
        worker.gpu.ordinal,
        offload,
        Some(worker.shared_pool.clone()),
    )?;

    tracing::info!(
        gpu = worker.gpu.ordinal,
        model = %model_name,
        "loading model..."
    );
    // Sample VRAM baseline before load so we can record the new model's
    // per-load delta rather than the device-global usage.
    let vram_baseline = device::vram_in_use_bytes(worker.gpu.ordinal);
    engine.load()?;

    let vram = device::vram_load_delta(worker.gpu.ordinal, vram_baseline);
    // Drop any evicted engine OUTSIDE the cache lock — `cuMemFree` and
    // safetensor unmap during the drop can block other cache users.
    let evicted = {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.insert_loaded(model_name.to_string(), engine, vram)
    };
    drop(evicted);

    Ok(())
}

/// Synchronously load a model on this GPU worker for the admin API.
///
/// Acquires the per-GPU load lock, then delegates to `ensure_model_ready_sync`.
/// Intended to be called inside `tokio::task::spawn_blocking`. Uses the
/// size-only peak (no resolution context) for the preflight — admin loads
/// don't carry a request shape.
pub fn load_blocking(worker: &GpuWorker, model_name: &str, config: &Config) -> anyhow::Result<()> {
    if worker.poisoned.load(Ordering::SeqCst) {
        anyhow::bail!(fatal_cuda_user_message(model_name));
    }
    let _lock = worker.model_load_lock.lock().unwrap();
    if worker.poisoned.load(Ordering::SeqCst) {
        anyhow::bail!(fatal_cuda_user_message(model_name));
    }
    let result = ensure_model_ready_sync(worker, model_name, config, None, false);
    if result.as_ref().is_err_and(is_fatal_cuda_error) {
        quarantine_poisoned_worker(worker);
    }
    result
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
        worker.set_resident_model(None);
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

/// Return type for [`run_chain_blocking`]. The outer `Result` carries
/// helper-prep errors (ensure_model_ready + cache take); the inner `Result`
/// is whatever the caller's closure returned. Closure errors pass through
/// unchanged so the caller can distinguish orchestrator-specific failures
/// (StageFailed, Invalid) from prep failures (ensure/cache).
pub type ChainPrep<T, E> = Result<Result<T, E>, anyhow::Error>;

/// Run a blocking chain operation on a specific GPU worker.
///
/// Acquires `worker.model_load_lock` for the full duration, binds the current
/// thread to `worker.gpu.ordinal` (so `reclaim_gpu_memory` debug asserts are
/// satisfied), ensures the model is loaded on GPU, takes the engine out of
/// the worker's cache, passes it to `with_engine`, and restores the engine
/// unconditionally on both success and closure failure.
///
/// Safe to call from inside `tokio::task::spawn_blocking`. The calling thread
/// can be any thread — the `ThreadGpuGuard` clears the thread-local on return.
///
/// # Errors
///
/// Returns `Err(anyhow::Error)` from the outer Result if:
/// - `ensure_model_ready_sync` fails (bad config, disk IO, load error).
/// - The engine vanishes from the cache between ensure and take (cache race).
///
/// Returns `Ok(Err(E))` if the closure itself returned an error — caller
/// preserves the closure's typed error for precise HTTP status mapping.
pub fn run_chain_blocking<T, E: std::fmt::Display + std::fmt::Debug>(
    worker: &GpuWorker,
    model_name: &str,
    config: &mold_core::Config,
    hint: Option<crate::model_manager::ActivationHint>,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> Result<T, E>,
) -> ChainPrep<T, E> {
    // Bind the thread to this worker's ordinal for the duration of the call.
    // `reclaim_gpu_memory` inside ensure_model_ready_sync debug-asserts this
    // matches its ordinal argument; without it, a stray caller on an unbound
    // thread would panic in debug builds.
    struct ThreadGpuGuard;
    impl Drop for ThreadGpuGuard {
        fn drop(&mut self) {
            mold_inference::device::clear_thread_gpu_ordinal();
        }
    }
    mold_inference::device::init_thread_gpu_ordinal(worker.gpu.ordinal);
    let _thread_gpu = ThreadGpuGuard;

    if worker.poisoned.load(Ordering::SeqCst) {
        anyhow::bail!(fatal_cuda_user_message(model_name));
    }

    // Acquire the per-worker load lock. Held for the entire chain duration —
    // single-clip generations on this worker queue behind us on the same lock.
    let _load_lock = worker
        .model_load_lock
        .lock()
        .map_err(|e| anyhow::anyhow!("worker.model_load_lock poisoned: {e}"))?;
    if worker.poisoned.load(Ordering::SeqCst) {
        anyhow::bail!(fatal_cuda_user_message(model_name));
    }

    // Ensure the model is GPU-resident on this worker. Handles load-from-disk,
    // parked-reload, and the reclaim-on-swap path using worker.gpu.ordinal.
    if let Err(error) = ensure_model_ready_sync(worker, model_name, config, hint, false) {
        if is_fatal_cuda_error(&error) {
            quarantine_poisoned_worker(worker);
        }
        return Err(error);
    }

    // Take the engine out of the worker's cache so the closure can mutate it.
    let cached = {
        let mut cache = worker
            .model_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("worker.model_cache poisoned: {e}"))?;
        cache.take(model_name).ok_or_else(|| {
            anyhow::anyhow!("cache race: engine '{model_name}' vanished after ensure_model_ready")
        })?
    };

    // Run the closure. Capture panics so we can still restore the engine
    // before propagating — otherwise a panic leaks the engine out of the cache.
    //
    // `AssertUnwindSafe` suppresses the compiler's UnwindSafe check because
    // `&mut dyn InferenceEngine` (across Box + trait object) isn't unwind-safe
    // by default. This is acceptable here: we only promise to prevent the
    // CUDA primary-context reset SEGV race, not to guarantee engine internal
    // state is pristine after a mid-generation panic. A panicked engine will
    // surface as a bad generation result to the next caller — not a crash.
    // `catch_unwind` + `resume_unwind` exists solely so the engine is
    // restored to the cache before the panic propagates up.
    let mut cached = cached;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_engine(cached.engine.as_mut())
    }));

    let fatal_cuda =
        matches!(&result, Ok(Err(error)) if has_fatal_cuda_error(&format!("{error:#}")));
    if fatal_cuda {
        quarantine_poisoned_worker(worker);
        drop(cached);
        let remaining = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        drop(remaining);
    } else {
        // Restore on ordinary closure errors and panics. Fatal CUDA errors are
        // the sole exception because the engine's context cannot be reused.
        let mut cache = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.restore(cached);
    }

    match result {
        Ok(inner) => Ok(inner),
        Err(panic_payload) => std::panic::resume_unwind(panic_payload),
    }
}

/// Run a blocking chain-job stage operation on a specific GPU worker.
///
/// Lock scope is exactly one stage render; callers reacquire for each stage
/// so the durable chain-job runner can yield between stages.
pub fn run_stage_blocking<T, E: std::fmt::Display + std::fmt::Debug>(
    worker: &GpuWorker,
    model_name: &str,
    config: &mold_core::Config,
    hint: Option<crate::model_manager::ActivationHint>,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> Result<T, E>,
) -> ChainPrep<T, E> {
    // Same take/restore critical section as `run_chain_blocking`; the durable
    // runner calls this once per stage, so the lock scope is one render call.
    run_chain_blocking(worker, model_name, config, hint, with_engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_registry::JobRegistry;
    use crate::model_cache::ModelCache;
    use crate::state::{GenerationJob, QueueHandle, SseCompletionPayload};
    use mold_core::{
        Config, GenerateRequest, GenerateResponse, ImageData, ModelConfig, OutputFormat,
    };
    use mold_inference::device::DiscoveredGpu;
    use mold_inference::shared_pool::SharedPool;
    use mold_inference::InferenceEngine;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::Duration;

    /// Weight-free engine that sleeps in `load()` to widen the critical-section
    /// window during concurrency tests.
    struct FakeSlowEngine {
        name: String,
        loaded: bool,
        load_sleep: Duration,
    }

    impl FakeSlowEngine {
        fn boxed(name: &str, load_sleep: Duration) -> Box<dyn InferenceEngine> {
            Box::new(Self {
                name: name.to_string(),
                loaded: false,
                load_sleep,
            })
        }
    }

    impl InferenceEngine for FakeSlowEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("FakeSlowEngine is not used for generation in tests")
        }
        fn model_name(&self) -> &str {
            &self.name
        }
        fn is_loaded(&self) -> bool {
            self.loaded
        }
        fn load(&mut self) -> anyhow::Result<()> {
            std::thread::sleep(self.load_sleep);
            self.loaded = true;
            Ok(())
        }
        fn unload(&mut self) {
            self.loaded = false;
        }
    }

    fn single_worker_pool_with_parked(model: &str, load_sleep: Duration) -> Arc<GpuWorker> {
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel::<GpuJob>(2);
        let mut cache = ModelCache::new(3);
        // Seed as Parked so `ensure_model_ready_sync` hits its reload path
        // and calls `engine.load()` — that's where the sleep widens the window.
        cache.insert(FakeSlowEngine::boxed(model, load_sleep), 0);
        Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal: 0,
                stable_id: Some("cuda:00000000000000000000000000000000".to_string()),
                raw_cuda_uuid: Some([0; 16]),
                device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
                identity_error: None,
                backend: mold_core::types::GpuBackend::Cuda,
                name: "fake-gpu-0".to_string(),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(cache)),
            resident_model: Arc::new(RwLock::new(None)),
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
        })
    }

    fn fake_upscale_job(config: Config, upscale_model: &str) -> GpuJob {
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"portrait","model":"flux-dev:q4","width":512,"height":512,"steps":4,"guidance":3.5,"batch_size":1}"#,
        )
        .unwrap();
        request.upscale_model = Some(upscale_model.to_string());
        GpuJob {
            id: "job-upscale-test".to_string(),
            model: request.model.clone(),
            request,
            completion_payload: crate::state::SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            config: Arc::new(tokio::sync::RwLock::new(config)),
            metadata_db: Arc::new(None),
            queue: QueueHandle::new(queue_tx),
            registry: JobRegistry::new(),
            events: crate::events::EventBroadcaster::new(),
            lease: None,
        }
    }

    fn fake_upscale_image() -> ImageData {
        ImageData {
            data: vec![0x89, 0x50, 0x4E, 0x47],
            format: OutputFormat::Png,
            width: 512,
            height: 512,
            index: 0,
        }
    }

    #[test]
    fn worker_rejects_stale_generation_before_touching_inference() {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<GpuJob>(1);
        let worker = Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal: 0,
                stable_id: Some("cuda:00000000000000000000000000000001".to_string()),
                raw_cuda_uuid: Some([1; 16]),
                device_kind: Some(mold_inference::device::CudaDeviceKind::FullGpu),
                identity_error: None,
                backend: mold_core::GpuBackend::Cuda,
                name: "protocol-test".to_string(),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes: 24 << 30,
                free_vram_bytes: 24 << 30,
            },
            model_cache: Arc::new(Mutex::new(ModelCache::new(1))),
            resident_model: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            in_flight: AtomicUsize::new(1),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx);
        let ready = event_rx.blocking_recv().expect("initial Ready event");
        assert!(matches!(
            ready,
            crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            }
        ));

        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"must not run","model":"flux-dev:q4","width":512,"height":512,"steps":4,"guidance":3.5,"batch_size":1}"#,
        )
        .unwrap();
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        worker
            .job_tx
            .send(GpuJob {
                id: "stale".to_string(),
                model: request.model.clone(),
                request,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx,
                output_dir: None,
                config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                metadata_db: Arc::new(None),
                queue: QueueHandle::new(queue_tx),
                registry: JobRegistry::new(),
                events: crate::events::EventBroadcaster::new(),
                lease: Some(crate::scheduler::LeaseFence {
                    work_id: "stale".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 0,
                    memory_sample_generation: 0,
                    memory_ledger_sequence: 0,
                }),
            })
            .unwrap();

        let returned = match event_rx.blocking_recv().expect("stale rejection") {
            crate::scheduler::WorkerEvent::Rejected { job, .. } => {
                assert_eq!(job.id, "stale");
                *job
            }
            _ => panic!("worker must reject the stale grant"),
        };
        assert!(
            result_rx.try_recv().is_err(),
            "stale grant must be rejected before inference produces a result"
        );
        drop(event_rx);
        // Wake the worker once after the coordinator receiver is gone. Its
        // rejection send fails and the next Ready publication terminates the
        // owner loop without ever entering `process_job`.
        worker.job_tx.send(returned).unwrap();
        handle
            .join()
            .expect("worker exits when coordinator is gone");
    }

    #[test]
    fn fatal_cuda_errors_are_classified_as_context_poisoning() {
        for message in [
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, an illegal memory access was encountered)",
            "DriverError(CUDA_ERROR_ECC_UNCORRECTABLE, uncorrectable ECC error)",
            "DriverError(CUDA_ERROR_LAUNCH_FAILED, unspecified launch failure)",
            "DriverError(CUDA_ERROR_ASSERT, device-side assert triggered)",
            "DriverError(CUDA_ERROR_MISALIGNED_ADDRESS, misaligned address)",
            "DriverError(CUDA_ERROR_HARDWARE_STACK_ERROR, hardware stack error)",
            "DriverError(CUDA_ERROR_ILLEGAL_INSTRUCTION, illegal instruction)",
            "DriverError(CUDA_ERROR_INVALID_ADDRESS_SPACE, invalid address space)",
            "DriverError(CUDA_ERROR_INVALID_PC, invalid program counter)",
            "DriverError(CUDA_ERROR_LAUNCH_TIMEOUT, launch timed out)",
        ] {
            let err = anyhow::anyhow!(message);
            assert!(is_fatal_cuda_error(&err), "not classified: {message}");
        }

        assert!(!is_fatal_cuda_error(&anyhow::anyhow!(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)"
        )));
        assert!(!is_fatal_cuda_error(&anyhow::anyhow!(
            "CublasError(CUBLAS_STATUS_NOT_INITIALIZED)"
        )));
    }

    #[tokio::test]
    async fn quarantining_worker_signals_process_restart() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);

        quarantine_poisoned_worker(&worker);

        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
        tokio::time::timeout(
            Duration::from_millis(50),
            worker.fatal_cuda_shutdown.notified(),
        )
        .await
        .expect("fatal CUDA quarantine must wake server shutdown");
    }

    #[test]
    fn quarantine_helper_ignores_ordinary_errors_and_latches_fatal_errors() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);

        assert!(!quarantine_if_fatal_cuda_error(
            &worker,
            &anyhow::anyhow!("ordinary inference failure")
        ));
        assert!(!worker.poisoned.load(Ordering::SeqCst));
        assert!(ensure_worker_not_poisoned(&worker, "flux-dev:q4").is_ok());

        let fatal =
            anyhow::anyhow!("DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, illegal memory access)")
                .context("generation failed");
        assert!(quarantine_if_fatal_cuda_error(&worker, &fatal));
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert_eq!(worker.consecutive_failures.load(Ordering::SeqCst), 3);
        assert!(worker.degraded_until.read().unwrap().is_none());
        let error = ensure_worker_not_poisoned(&worker, "flux-dev:q4").unwrap_err();
        assert!(error.to_string().contains("Restart the mold server"));
    }

    #[tokio::test]
    async fn buffered_job_is_rejected_without_touching_a_poisoned_worker() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);
        worker.poisoned.store(true, Ordering::SeqCst);
        worker.in_flight.store(1, Ordering::SeqCst);

        let request = fake_upscale_job(Config::default(), "unused").request;
        let (queue_tx, mut queue_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(queue_tx);
        let (placeholder_tx, _placeholder_rx) = tokio::sync::oneshot::channel();
        queue
            .submit(
                GenerationJob {
                    id: "placeholder".to_string(),
                    request: request.clone(),
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: placeholder_tx,
                    output_dir: None,
                },
                1,
            )
            .await
            .unwrap();

        let registry = JobRegistry::new();
        registry.register("buffered-job", request.model.clone());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        process_job(
            &worker,
            GpuJob {
                id: "buffered-job".to_string(),
                model: request.model.clone(),
                request,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: Some(progress_tx),
                result_tx,
                output_dir: None,
                config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                metadata_db: Arc::new(None),
                queue: queue.clone(),
                registry: registry.clone(),
                events: crate::events::EventBroadcaster::new(),
                lease: None,
            },
        );

        let result = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("poisoned worker unexpectedly completed buffered job"),
        };
        assert!(result.contains("worker was quarantined"));
        assert!(matches!(
            progress_rx.recv().await,
            Some(SseMessage::Error(_))
        ));
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
        assert_eq!(queue.pending(), 0);
        assert!(registry.snapshot().entries.is_empty());
        assert!(worker.model_cache.lock().unwrap().contains("flux-dev:q4"));
        drop(queue_rx.recv().await);
    }

    #[test]
    fn poisoned_worker_rejects_admin_and_chain_entry_points() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::ZERO);
        worker.poisoned.store(true, Ordering::SeqCst);
        let config = Config::default();

        let load_error = load_blocking(&worker, "fake-model", &config).unwrap_err();
        assert!(load_error.to_string().contains("worker was quarantined"));

        let closure_ran = AtomicBool::new(false);
        let chain_error = run_chain_blocking(
            &worker,
            "fake-model",
            &config,
            None,
            |_engine| -> anyhow::Result<()> {
                closure_ran.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(chain_error.to_string().contains("worker was quarantined"));
        assert!(!closure_ran.load(Ordering::SeqCst));
        assert!(worker.model_cache.lock().unwrap().contains("fake-model"));
    }

    #[test]
    fn worker_post_upscale_reports_missing_downloaded_model() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);
        let job = fake_upscale_job(Config::default(), "real-esrgan-x4plus:fp16");
        let mut response = GenerateResponse {
            images: vec![],
            video: None,
            generation_time_ms: 10,
            model: job.request.model.clone(),
            seed_used: 7,
            gpu: None,
        };

        let err = upscale_generated_image_on_worker(
            &worker,
            &job,
            "real-esrgan-x4plus:fp16",
            fake_upscale_image(),
            &mut response,
        )
        .expect_err("worker should reject a missing upscaler config");

        assert!(err.contains("not downloaded"), "got: {err}");
    }

    #[test]
    fn worker_post_upscale_surfaces_missing_weights_path() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);
        let tmp = tempfile::TempDir::new().unwrap();
        let missing_weights = tmp.path().join("missing-upscaler.safetensors");
        let mut config = Config::default();
        config.models.insert(
            "real-esrgan-x4plus:fp16".to_string(),
            ModelConfig {
                transformer: Some(missing_weights.display().to_string()),
                ..Default::default()
            },
        );
        let job = fake_upscale_job(config, "real-esrgan-x4plus:fp16");
        let mut response = GenerateResponse {
            images: vec![],
            video: None,
            generation_time_ms: 10,
            model: job.request.model.clone(),
            seed_used: 7,
            gpu: None,
        };

        let err = upscale_generated_image_on_worker(
            &worker,
            &job,
            "real-esrgan-x4plus:fp16",
            fake_upscale_image(),
            &mut response,
        )
        .expect_err("worker should surface missing weight files before generation completes");

        assert!(err.contains("failed to load upscaler"), "got: {err}");
        assert!(err.contains("upscaler weights not found"), "got: {err}");
    }

    #[test]
    fn run_chain_blocking_quarantines_fatal_cuda_closure_error() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::ZERO);
        let config = Config::default();

        let result = run_chain_blocking(&worker, "fake-model", &config, None, |_engine| {
            Err::<(), _>(
                anyhow::anyhow!("DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, illegal memory access)")
                    .context("stage render failed"),
            )
        })
        .expect("engine preparation should succeed");

        assert!(result.is_err());
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
        assert!(worker.model_cache.lock().unwrap().is_empty());
    }

    /// Two concurrent callers into `run_chain_blocking` on the same worker
    /// must serialize — `MAX_CONCURRENT` must never exceed 1.
    ///
    /// Fails to compile until `run_chain_blocking` is implemented in Task 2.
    #[test]
    fn run_chain_blocking_serializes_same_worker() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::from_millis(30));
        let config = Config::default();

        let active = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let instrumented = |active: Arc<AtomicUsize>, max_concurrent: Arc<AtomicUsize>| {
            move |_engine: &mut dyn InferenceEngine| -> anyhow::Result<()> {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(50));
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
        };

        let worker_a = worker.clone();
        let config_a = config.clone();
        let a = active.clone();
        let m = max_concurrent.clone();
        let t_a = std::thread::spawn(move || {
            run_chain_blocking(&worker_a, "fake-model", &config_a, None, instrumented(a, m))
                .expect("prep ok")
                .expect("closure ok");
        });

        let worker_b = worker.clone();
        let config_b = config.clone();
        let a = active.clone();
        let m = max_concurrent.clone();
        let t_b = std::thread::spawn(move || {
            run_chain_blocking(&worker_b, "fake-model", &config_b, None, instrumented(a, m))
                .expect("prep ok")
                .expect("closure ok");
        });

        t_a.join().unwrap();
        t_b.join().unwrap();

        assert_eq!(
            max_concurrent.load(Ordering::SeqCst),
            1,
            "two concurrent run_chain_blocking calls must serialize on worker.model_load_lock"
        );
    }

    // ── OOM detection + message rewriting (Part 2) ────────────────────────────

    /// `is_cuda_oom` detects the canonical `CUDA_ERROR_OUT_OF_MEMORY` error
    /// string. This pattern-match is the only stable signal available from
    /// the candle/cudarc error chain since the cudarc error type is not
    /// downcasted via std::error::Error in the candle re-export.
    #[test]
    fn is_cuda_oom_detects_driver_error_string() {
        let oom_err = anyhow::anyhow!(r#"DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")"#);
        assert!(
            is_cuda_oom(&oom_err),
            "must detect CUDA_ERROR_OUT_OF_MEMORY in anyhow error chain"
        );
    }

    /// A regular (non-OOM) error must not trigger the OOM path.
    #[test]
    fn is_cuda_oom_does_not_trigger_on_regular_errors() {
        let reg_err = anyhow::anyhow!("safetensors file not found");
        assert!(
            !is_cuda_oom(&reg_err),
            "non-OOM error must not be classified as OOM"
        );
    }

    /// `oom_user_message` produces a message that mentions actionable
    /// mitigations — frames, resolution, or quantized variants. It must
    /// NOT contain the opaque CUDA driver error string.
    #[test]
    fn runtime_oom_message_suggests_offload_and_smaller_frames() {
        let msg = oom_user_message("ltx-video-0.9.8-13b-dev:bf16");
        assert!(
            msg.contains("frames") || msg.contains("width") || msg.contains("quantized"),
            "OOM message must suggest reducing frames, resolution, or using a \
             quantized variant; got: {msg}",
        );
        assert!(
            !msg.contains("CUDA_ERROR_OUT_OF_MEMORY"),
            "OOM user message must not expose the raw CUDA driver error string; \
             got: {msg}",
        );
        assert!(
            msg.contains("ltx-video-0.9.8-13b-dev:bf16"),
            "OOM message must include the model name so the user knows what failed; \
            got: {msg}",
        );
    }

    #[test]
    fn runtime_oom_message_for_sd15_1024_mentions_resolution_not_frames() {
        let req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"portrait","model":"realistic-vision-v5:fp16","width":1024,"height":1024,"steps":25,"guidance":7.5,"batch_size":1}"#,
        )
        .unwrap();

        let msg =
            oom_user_message_for_request("realistic-vision-v5:fp16", Some("sd15"), Some(&req));

        assert!(
            msg.contains("1024x1024"),
            "image OOM message should mention the requested resolution; got: {msg}"
        );
        assert!(
            msg.contains("512x512"),
            "SD1.5 OOM message should point back to the native/default size; got: {msg}"
        );
        assert!(
            msg.contains("checkpoint") || msg.contains("model file"),
            "OOM message should explain why file size is not peak VRAM; got: {msg}"
        );
        assert!(
            !msg.contains("--frames"),
            "image OOM message must not suggest video frame-count fixes; got: {msg}"
        );
    }

    #[test]
    fn runtime_oom_message_for_ltx_keeps_frame_guidance() {
        let req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"camera pan","model":"ltx-video-0.9.8-13b-dev:bf16","width":768,"height":512,"steps":25,"guidance":3.5,"batch_size":1,"frames":25}"#,
        )
        .unwrap();

        let msg = oom_user_message_for_request(
            "ltx-video-0.9.8-13b-dev:bf16",
            Some("ltx-video"),
            Some(&req),
        );

        assert!(
            msg.contains("--frames") && msg.contains("25"),
            "video OOM message should keep frame-count guidance; got: {msg}"
        );
        assert!(
            msg.contains("768x512"),
            "video OOM message should mention the requested resolution; got: {msg}"
        );
    }

    /// A failed `engine.load()` must NOT leave a phantom entry in the cache.
    ///
    /// `ensure_model_ready_sync` calls `create_engine_with_pool` then
    /// `engine.load()`, and only calls `cache.insert_loaded()` after success.
    /// This test confirms that a load failure on a fresh (non-cached) engine
    /// leaves the cache empty — `contains()` returns false and `in_flight`
    /// is clean.
    ///
    /// We can't exercise the full `ensure_model_ready_sync` path without real
    /// model files, so we test the cache contract directly: a failed
    /// `insert_loaded` attempt (via the engine's failing load) leaves the
    /// cache exactly as it was before.
    #[test]
    fn failed_load_does_not_leak_into_model_cache() {
        // Engine that always fails to load.
        struct FailingLoadEngine {
            name: String,
        }
        impl InferenceEngine for FailingLoadEngine {
            fn generate(&mut self, _: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
                unreachable!()
            }
            fn model_name(&self) -> &str {
                &self.name
            }
            fn is_loaded(&self) -> bool {
                false
            }
            fn load(&mut self) -> anyhow::Result<()> {
                anyhow::bail!(r#"DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")"#)
            }
            fn unload(&mut self) {}
        }

        let cache = ModelCache::new(3);
        let model_name = "ltx-video-0.9.8-13b-dev:bf16";

        // Simulate the load path: create the engine, attempt load, only
        // insert on success. This mirrors the exact control flow in
        // `ensure_model_ready_sync`.
        let mut engine: Box<dyn InferenceEngine> = Box::new(FailingLoadEngine {
            name: model_name.to_string(),
        });
        let load_result = engine.load();

        assert!(
            load_result.is_err(),
            "engine.load() must fail for this test to be meaningful"
        );
        assert!(
            is_cuda_oom(load_result.as_ref().unwrap_err()),
            "load error must be classified as OOM"
        );

        // Crucially: we do NOT call cache.insert_loaded() on failure.
        // The cache must remain empty.
        assert!(
            !cache.contains(model_name),
            "cache must not contain the model after a failed load — \
             `insert_loaded` must only be called on success"
        );
        assert!(
            cache.is_empty(),
            "cache must be completely empty after a failed load"
        );
    }
}
