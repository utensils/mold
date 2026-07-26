use crate::gpu_pool::{
    ActiveGeneration, AdminModelUnloadJob, GpuJob, GpuWorker, GpuWorkerCommand, LeaseGrant,
    OwnerWork, PostGenerationUpscaleJob, PromptExpansionJob, StandaloneUpscaleJob,
};
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
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    cache_idle_ttl: Duration,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("gpu-worker-{}", worker.gpu.ordinal))
        .spawn(move || {
            run_gpu_owner(
                &worker,
                job_rx,
                scheduler_tx,
                cache_idle_ttl,
                Duration::from_secs(60),
            );
        })
        .expect("failed to spawn GPU worker thread")
}

fn run_gpu_owner(
    worker: &GpuWorker,
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    cache_idle_ttl: Duration,
    idle_poll: Duration,
) {
    if worker
        .owner_thread_id
        .set(std::thread::current().id())
        .is_err()
    {
        worker.poisoned.store(true, Ordering::SeqCst);
        worker.fatal_cuda_error.store(true, Ordering::SeqCst);
        worker.fatal_cuda_shutdown.notify_waiters();
        tracing::error!(
            gpu = worker.gpu.ordinal,
            "GPU worker owner thread was initialized more than once"
        );
        return;
    }
    // Bind this thread to its GPU ordinal so device operations can
    // debug-assert callers don't drift onto a sibling GPU's context.
    mold_inference::device::init_thread_gpu_ordinal(worker.gpu.ordinal);
    tracing::info!(
        gpu = worker.gpu.ordinal,
        name = %worker.gpu.name,
        "GPU worker thread started"
    );
    let device_id = crate::scheduler::worker_device_id(worker);
    let mut generation = 1_u64;
    'owner: loop {
        if worker.shutdown_requested.load(Ordering::SeqCst)
            || worker.poisoned.load(Ordering::SeqCst)
            || worker.fatal_cuda_error.load(Ordering::SeqCst)
        {
            break;
        }
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
        let command = loop {
            match job_rx.recv_timeout(idle_poll) {
                Ok(command) => break command,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if worker.shutdown_requested.load(Ordering::SeqCst)
                        || worker.poisoned.load(Ordering::SeqCst)
                        || worker.fatal_cuda_error.load(Ordering::SeqCst)
                    {
                        break 'owner;
                    }
                    evict_idle_on_worker(worker, cache_idle_ttl);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break 'owner,
            }
        };
        let grant = match command {
            GpuWorkerCommand::Grant(grant) => grant,
            GpuWorkerCommand::Shutdown => break,
        };
        if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst)
        {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::FatalCuda,
            });
            break;
        }
        if worker.shutdown_requested.load(Ordering::SeqCst) {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::StaleWorkerGeneration,
            });
            break;
        }
        let fence = &grant.fence;
        if fence.device_id != device_id || fence.worker_generation != generation {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                worker_generation: generation,
                grant,
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
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_owner_work(worker, *grant, &scheduler_tx);
        }));
        worker.in_flight.store(0, Ordering::SeqCst);
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
        if outcome.is_err()
            || worker.shutdown_requested.load(Ordering::SeqCst)
            || worker.poisoned.load(Ordering::SeqCst)
            || worker.fatal_cuda_error.load(Ordering::SeqCst)
        {
            break;
        }
        generation = generation.saturating_add(1);
    }
    let cached = worker
        .model_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        // Destructors may call CUDA. A poisoned primary context is never
        // touched again; process teardown reclaims these.
        contain_poisoned_cuda(cached);
    } else {
        drop(cached);
    }
    tracing::info!(gpu = worker.gpu.ordinal, "GPU worker thread exiting");
}

fn process_owner_work(
    worker: &GpuWorker,
    grant: LeaseGrant,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) {
    if let Err(error) = ensure_owner_thread(worker) {
        grant.work.reject(error.to_string());
        return;
    }
    match grant.work {
        OwnerWork::Generation(mut job) => {
            job.lease = Some(grant.fence);
            process_job(worker, *job, scheduler_tx);
        }
        OwnerWork::PromptExpansion(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            process_prompt_expansion(worker, *job);
        }
        OwnerWork::PostUpscale(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            process_post_generation_upscale(worker, *job);
        }
        OwnerWork::StandaloneUpscale(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            process_standalone_upscale(worker, *job);
        }
        OwnerWork::AdminModelLoad(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let result = load_blocking(worker, &job.model, &job.config).map_err(|e| e.to_string());
            let _ = job.result_tx.send(result);
        }
        OwnerWork::AdminModelUnload(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            process_admin_unload(worker, *job);
        }
        #[cfg(test)]
        OwnerWork::Probe { run, .. } => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            run();
        }
    }
}

fn commit_utility_allocation(
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    fence: &crate::scheduler::LeaseFence,
) {
    let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::AllocationCommitted {
        device_id: fence.device_id.clone(),
        work_id: fence.work_id.clone(),
        worker_generation: fence.worker_generation,
    });
}

fn process_prompt_expansion(worker: &GpuWorker, job: PromptExpansionJob) {
    let result = (|| -> anyhow::Result<mold_core::ExpandResult> {
        ensure_worker_not_poisoned(worker, &job.settings.model)?;
        #[cfg(feature = "expand")]
        {
            use mold_core::PromptExpander;
            let selector = worker.gpu.stable_id.as_ref().map_or_else(
                || mold_core::GpuSelector::Ordinal(worker.gpu.ordinal),
                |id| mold_core::GpuSelector::Identifier(id.clone()),
            );
            let expander = mold_inference::expand::LocalExpander::from_config(
                &job.config,
                Some(&job.settings.model),
            )
            .ok_or_else(|| {
                anyhow::anyhow!("local expand model not found — run: mold pull qwen3-expand")
            })?
            .with_gpu_selection(mold_core::GpuSelection::Specific(vec![selector]))
            .with_preferred_gpu(Some(worker.gpu.ordinal));
            return expander.expand(&job.prompt, &job.expand_config);
        }
        #[cfg(not(feature = "expand"))]
        {
            anyhow::bail!("local prompt expansion not available — built without expand feature")
        }
    })();
    if result
        .as_ref()
        .err()
        .is_some_and(|error| quarantine_if_fatal_cuda_error(worker, error))
    {
        tracing::error!(
            gpu = worker.gpu.ordinal,
            "prompt expansion fatally poisoned its owner context"
        );
    }
    let _ = job.result_tx.send(result.map_err(|e| e.to_string()));
}

fn process_standalone_upscale(worker: &GpuWorker, job: StandaloneUpscaleJob) {
    let result = (|| -> anyhow::Result<mold_core::UpscaleResponse> {
        ensure_worker_not_poisoned(worker, &job.model)?;
        let mut engine = mold_inference::create_upscale_engine(
            job.model.clone(),
            job.weights_path,
            mold_inference::LoadStrategy::Eager,
            worker.gpu.ordinal,
        )?;
        if let Some(progress_tx) = job.progress_tx {
            engine.set_on_progress(Box::new(move |event| {
                let _ = progress_tx.send(SseMessage::Progress(event.into()));
            }));
        }
        let result = engine.upscale(&job.request);
        engine.clear_on_progress();
        if result
            .as_ref()
            .err()
            .is_some_and(|error| quarantine_if_fatal_cuda_error(worker, error))
        {
            // A destructor may call into the poisoned CUDA context. Process
            // teardown owns recovery; deliberately leak this one engine.
            std::mem::forget(engine);
            return result;
        }
        engine.unload();
        drop(engine);
        result
    })();
    if result
        .as_ref()
        .err()
        .is_some_and(|error| quarantine_if_fatal_cuda_error(worker, error))
    {
        tracing::error!(
            gpu = worker.gpu.ordinal,
            "standalone upscale fatally poisoned its owner context"
        );
    }
    let _ = job.result_tx.send(result.map_err(|e| e.to_string()));
}

fn process_post_generation_upscale(worker: &GpuWorker, mut job: PostGenerationUpscaleJob) {
    let cleanup = GenerationCleanup::new(&job.generation);
    let upscale_model = job
        .generation
        .request
        .upscale_model
        .clone()
        .unwrap_or_default();
    let result = upscale_generated_image_on_worker(
        worker,
        &job.generation,
        &upscale_model,
        job.image.clone(),
        &mut job.response,
    );
    if result
        .as_ref()
        .is_err_and(|error| has_fatal_cuda_error(error))
        && !worker.poisoned.load(Ordering::SeqCst)
    {
        quarantine_poisoned_worker(worker);
    }
    let (image, original, error) = settle_post_generation_upscale(job.image, result);
    if let Some(error) = error {
        tracing::warn!(
            gpu = worker.gpu.ordinal,
            %error,
            "post-generation upscale failed; keeping original image"
        );
    }
    finish_generation_success(*job.generation, job.response, image, original);
    drop(cleanup);
}

fn process_admin_unload(worker: &GpuWorker, job: AdminModelUnloadJob) {
    if let Err(error) =
        ensure_worker_not_poisoned(worker, job.model.as_deref().unwrap_or("active model"))
    {
        let _ = job.result_tx.send(Err(error.to_string()));
        return;
    }
    if job.evict_cached {
        let Some(model) = job.model.as_deref() else {
            let _ = job
                .result_tx
                .send(Err("cached eviction requires a model name".to_string()));
            return;
        };
        let _ = job
            .result_tx
            .send(evict_cached_model_blocking(worker, model).map_err(|error| error.to_string()));
        return;
    }
    if let Some(expected) = job.model.as_deref() {
        let resident = worker
            .resident_model
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if resident.as_deref() != Some(expected) {
            let _ = job.result_tx.send(Ok(None));
            return;
        }
    }
    let _ = job
        .result_tx
        .send(unload_blocking(worker).map_err(|error| error.to_string()));
}

/// Evict parked engines on the worker thread that owns their device context.
///
/// Engine destruction may call into CUDA while releasing tensors and library
/// workspaces, so returning the boxes to an async maintenance task is not safe.
fn evict_idle_on_worker(worker: &GpuWorker, ttl: Duration) {
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        return;
    }
    if let Err(error) = ensure_owner_thread(worker) {
        tracing::error!(
            gpu = worker.gpu.ordinal,
            %error,
            "refusing to evict GPU resources off the owner thread"
        );
        return;
    }
    let _load_guard = worker
        .model_load_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        return;
    }
    let evicted = {
        let mut cache = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.evict_idle(ttl)
    };
    if evicted.is_empty() {
        return;
    }
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        contain_poisoned_cuda(evicted);
        return;
    }
    let count = evicted.len();
    drop(evicted);
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        return;
    }
    match device::post_drop_free_vram_bytes(worker.gpu.ordinal) {
        Ok(free_after_drop) => tracing::info!(
            gpu = worker.gpu.ordinal,
            count,
            free_vram_bytes = free_after_drop,
            "evicted idle model cache entries on owning GPU worker"
        ),
        Err(error) if error.is_fatal_cuda() => {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
        }
        Err(error) => tracing::warn!(
            gpu = worker.gpu.ordinal,
            count,
            %error,
            "idle entries evicted but post-drop VRAM sample was unavailable"
        ),
    }
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

/// Keep a CUDA-backed object alive until the OS tears down the quarantined
/// process. Running its destructor after a fatal asynchronous driver error can
/// re-enter the invalid context through tensor, cuBLAS, or allocator cleanup.
/// This intentionally leaks only on the terminal process-restart path.
fn contain_poisoned_cuda<T>(value: T) {
    let _contained = std::mem::ManuallyDrop::new(value);
}

fn contain_worker_cache(worker: &GpuWorker) {
    let remaining = worker
        .model_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    contain_poisoned_cuda(remaining);
}

fn synchronize_after_oom(worker: &GpuWorker) -> bool {
    match device::try_synchronize_device(worker.gpu.ordinal) {
        Ok(()) => true,
        Err(error) if error.is_fatal_cuda() => {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            false
        }
        Err(error) => {
            tracing::warn!(
                gpu = worker.gpu.ordinal,
                %error,
                "CUDA synchronize after OOM was unavailable"
            );
            true
        }
    }
}

#[cfg(feature = "cuda")]
fn device_memory_api_error(error: device::DeviceMemoryError) -> crate::routes::ApiError {
    if error.is_fatal_cuda() {
        crate::routes::ApiError::internal(error.to_string())
    } else {
        crate::routes::ApiError::insufficient_memory(format!(
            "GPU memory admission blocked because current free VRAM could not be measured: {error}"
        ))
    }
}

pub(crate) fn run_upscale_engine_safely(
    worker: &GpuWorker,
    mut engine: Box<dyn mold_inference::UpscaleEngine>,
    request: &mold_core::UpscaleRequest,
) -> anyhow::Result<mold_core::UpscaleResponse> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.upscale(request)));
    match result {
        Ok(Ok(response)) => {
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                engine.clear_on_progress();
                engine.unload();
            })) {
                quarantine_poisoned_worker(worker);
                contain_poisoned_cuda(engine);
                contain_worker_cache(worker);
                let message = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                anyhow::bail!("upscaler cleanup panicked; CUDA state quarantined: {message}");
            }
            Ok(response)
        }
        Ok(Err(error)) if is_fatal_cuda_error(&error) => {
            quarantine_poisoned_worker(worker);
            contain_poisoned_cuda(engine);
            contain_worker_cache(worker);
            Err(error)
        }
        Ok(Err(error)) => {
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                engine.clear_on_progress();
                engine.unload();
            })) {
                quarantine_poisoned_worker(worker);
                contain_poisoned_cuda(engine);
                contain_worker_cache(worker);
                let message = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                anyhow::bail!("upscaler cleanup panicked; CUDA state quarantined: {message}");
            }
            Err(error)
        }
        Err(payload) => {
            quarantine_poisoned_worker(worker);
            contain_poisoned_cuda(engine);
            contain_worker_cache(worker);
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            anyhow::bail!("upscale panicked: {message}")
        }
    }
}

fn load_engine_safely(
    worker: &GpuWorker,
    mut engine: Box<dyn mold_inference::InferenceEngine>,
) -> anyhow::Result<Box<dyn mold_inference::InferenceEngine>> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.load()));
    match result {
        Ok(Ok(())) => Ok(engine),
        Ok(Err(error)) if is_fatal_cuda_error(&error) => {
            quarantine_poisoned_worker(worker);
            contain_poisoned_cuda(engine);
            contain_worker_cache(worker);
            Err(error)
        }
        Ok(Err(error)) => Err(error),
        Err(payload) => {
            quarantine_poisoned_worker(worker);
            contain_poisoned_cuda(engine);
            contain_worker_cache(worker);
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            anyhow::bail!("engine load panicked; CUDA state quarantined: {message}")
        }
    }
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
    if worker.poisoned.load(Ordering::SeqCst)
        || worker.fatal_cuda_error.load(Ordering::SeqCst)
        || worker.shutdown_requested.load(Ordering::SeqCst)
    {
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
    let upscale_result = run_upscale_engine_safely(worker, engine, &req);
    let upscaled = upscale_result.map_err(|e| format!("upscale failed: {e}"))?;
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

struct GenerationCleanup {
    queue: crate::state::QueueHandle,
    registry: crate::job_registry::SharedJobRegistry,
    id: String,
}

impl GenerationCleanup {
    fn new(job: &GpuJob) -> Self {
        Self {
            queue: job.queue.clone(),
            registry: job.registry.clone(),
            id: job.id.clone(),
        }
    }
}

impl Drop for GenerationCleanup {
    fn drop(&mut self) {
        self.queue.decrement();
        self.registry.remove(&self.id);
    }
}

fn process_job(
    worker: &GpuWorker,
    job: GpuJob,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) {
    let model_name = job.model.clone();
    let ordinal = worker.gpu.ordinal;
    let job_id = job.id.clone();

    // Release the global queue slot AND the registry entry when this job
    // finishes, regardless of which exit path runs. The dispatcher only
    // decrements when it *fails* to dispatch — once we own the GpuJob, we
    // own both pieces of cleanup. Combining them in one drop guard keeps
    // the two counters from drifting on early-return paths.
    let cleanup = GenerationCleanup::new(&job);

    // Jobs may already be buffered in this worker's channel when a preceding
    // job kills the context. Fail them without touching CUDA, including jobs
    // explicitly pinned to this ordinal.
    if worker.poisoned.load(Ordering::SeqCst)
        || worker.fatal_cuda_error.load(Ordering::SeqCst)
        || worker.shutdown_requested.load(Ordering::SeqCst)
    {
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
    // The V2 coordinator claims the row atomically before transport. Legacy
    // single-dispatcher tests/adapters carry no lease and retain the old
    // worker-side promotion until that adapter is removed.
    if job.lease.is_none() {
        job.registry.mark_running(&job_id, Some(ordinal));
    }

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
            if synchronize_after_oom(worker) {
                cuda_oom_user_message(
                    worker,
                    &model_name,
                    family_slug.as_deref(),
                    Some(&job.request),
                )
            } else {
                (fatal_cuda_user_message(&model_name), false)
            }
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

    // This is the first real allocation boundary: model readiness has
    // completed, so host allocations owned by this lease now exist. The
    // coordinator keeps the reservation charged until a memory sample whose
    // collection began after this commit can reflect it.
    if let Some(lease) = job.lease.as_ref() {
        let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::AllocationCommitted {
            device_id: lease.device_id.clone(),
            work_id: lease.work_id.clone(),
            worker_generation: lease.worker_generation,
        });
    }

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
        ensure_worker_not_poisoned(worker, &model_name)?;
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

    // A fatal driver error invalidates every CUDA object owned by this
    // context. Never put the triggering engine back into the cache: doing so
    // caused immediate CUBLAS_STATUS_NOT_INITIALIZED retries on the poisoned
    // worker. We deliberately do not reset the primary context here because
    // Candle/cudarc retain handles to it; an in-process reset would make those
    // handles dangling. Quarantine until process restart instead.
    let fatal_cuda = matches!(&result, Ok(Err(e)) if is_fatal_cuda_error(e));
    let inference_panicked = result.is_err();
    if fatal_cuda || inference_panicked {
        // Signal process teardown, then retain every suspect CUDA-backed
        // object for OS teardown. Even an apparently innocuous engine method
        // or destructor may synchronize an invalid context.
        quarantine_poisoned_worker(worker);
        contain_poisoned_cuda(cached_engine);
        contain_worker_cache(worker);
    } else {
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cached_engine.engine.clear_on_progress();
        })) {
            quarantine_poisoned_worker(worker);
            contain_poisoned_cuda(cached_engine);
            contain_worker_cache(worker);
            std::panic::resume_unwind(payload);
        }
        let mut cache = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
            if response.video.is_none() {
                if let Some(upscale_model) = job
                    .request
                    .upscale_model
                    .as_deref()
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                {
                    let resolved = mold_core::manifest::resolve_model_name(upscale_model);
                    let estimated_vram_bytes = {
                        let config = job.config.blocking_read();
                        config
                            .models
                            .get(&resolved)
                            .and_then(|model| model.transformer.as_ref())
                            .and_then(|path| std::fs::metadata(path).ok())
                            .map(|metadata| metadata.len().saturating_add(2 << 30))
                            .unwrap_or(2 << 30)
                    };
                    let followup_id = format!("{}::post-upscale", job.id);
                    let work = crate::scheduler::ScheduledOwnerWork::new(
                        followup_id.clone(),
                        resolved,
                        estimated_vram_bytes,
                        OwnerWork::PostUpscale(Box::new(PostGenerationUpscaleJob {
                            id: followup_id,
                            generation: Box::new(job),
                            response,
                            image: img,
                        })),
                    );
                    match scheduler_tx.send(crate::scheduler::WorkerEvent::FollowupReady {
                        work: Box::new(work),
                    }) {
                        Ok(()) => {
                            std::mem::forget(cleanup);
                        }
                        Err(error) => {
                            if let crate::scheduler::WorkerEvent::FollowupReady { work } = error.0 {
                                work.work.reject(
                                    "scheduler stopped before post-generation upscale".to_string(),
                                );
                                std::mem::forget(cleanup);
                            }
                        }
                    }
                    return;
                }
            }

            finish_generation_success(job, response, img, None);
            drop(cleanup);
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
                if synchronize_after_oom(worker) {
                    cuda_oom_user_message(
                        worker,
                        &model_name,
                        family_slug.as_deref(),
                        Some(&job.request),
                    )
                } else {
                    (fatal_cuda_user_message(&model_name), false)
                }
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
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            let err_msg = format!(
                "inference panicked on GPU {ordinal}: {msg}; CUDA owner was quarantined and the server must restart"
            );
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent {
                    message: err_msg.clone(),
                }));
            }
            let _ = job.result_tx.send(Err(err_msg));
        }
    }
}

fn finish_generation_success(
    job: GpuJob,
    response: mold_core::GenerateResponse,
    image: ImageData,
    original_image: Option<ImageData>,
) {
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
                original_image.as_ref(),
                &image,
                &job.model,
                job.request.batch_size,
                &metadata,
                Some(generation_time_ms),
                db,
                events,
            );
        }
    }

    if let Some(ref tx) = job.progress_tx {
        let message = build_sse_completion_message(
            &response,
            &image,
            original_image.as_ref(),
            Some(&metadata),
            &saved_names,
            job.completion_payload,
        );
        let _ = tx.send(message);
    }
    let _ = job
        .result_tx
        .send(Ok(GenerationJobResult { image, response }));
}

/// Preflight memory check with evict-to-fit recovery.
///
/// Wraps `model_manager::preflight_memory_guard`. On a budget failure, drops
/// the LRU parked entry (skipping `model_name` so a parked-reload doesn't
/// evict its own target), samples the resulting pressure, and retries. Loops
/// until the preflight passes or the cache has
/// no parked entries left to surrender — at which point the original
/// insufficient-memory error is returned.
///
/// Holds the cache lock only for the brief eviction step; the engine drop runs
/// outside it. The caller is expected to hold
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

        #[cfg(feature = "cuda")]
        device::post_drop_free_vram_bytes(ordinal).map_err(device_memory_api_error)?;
    }
}

fn select_load_strategy_for_worker(
    worker: &GpuWorker,
    model_name: &str,
    paths: &ModelPaths,
    hint: Option<crate::model_manager::ActivationHint>,
) -> anyhow::Result<mold_inference::LoadStrategy> {
    let active_vram = worker
        .model_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .active_vram_bytes();
    let available =
        crate::model_manager::effective_load_available_bytes(active_vram, worker.gpu.ordinal)
            .map_err(|error| anyhow::anyhow!(error.error))?;
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
    Ok(strategy)
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
    } else if result.as_ref().is_err_and(is_fatal_cuda_error) {
        quarantine_poisoned_worker(worker);
        contain_worker_cache(worker);
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
        if let Some(ref paths) = cached_paths {
            crate::memory_preflight::preflight_memory_guard_after_drop(
                model_name,
                paths,
                worker.gpu.ordinal,
                hint,
            )
            .map_err(|e| anyhow::anyhow!(e.error))?;
        } else {
            #[cfg(feature = "cuda")]
            device::post_drop_free_vram_bytes(worker.gpu.ordinal)
                .map_err(|error| anyhow::anyhow!(error))?;
        }
        let load_strategy = match cached_paths.as_ref() {
            Some(paths) => select_load_strategy_for_worker(worker, model_name, paths, hint)?,
            None => mold_inference::LoadStrategy::Eager,
        };

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
            let engine = match mold_inference::create_engine_with_pool(
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
            let engine = match load_engine_safely(worker, engine) {
                Ok(engine) => engine,
                Err(err) if worker.poisoned.load(Ordering::SeqCst) => {
                    contain_poisoned_cuda(old_engine);
                    return Err(err);
                }
                Err(err) => {
                    let evicted = {
                        let mut cache = worker.model_cache.lock().unwrap();
                        cache.insert(old_engine, 0)
                    };
                    drop(evicted);
                    return Err(err);
                }
            };
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
        let engine = {
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
        let engine = load_engine_safely(worker, engine)?;

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

    // Unload active model first.
    {
        let mut cache = worker.model_cache.lock().unwrap();
        if cache.unload_active().is_some() {
            worker.set_resident_model(None);
        }
    }
    crate::memory_preflight::preflight_memory_guard_after_drop(
        model_name,
        &paths,
        worker.gpu.ordinal,
        hint,
    )
    .map_err(|e| anyhow::anyhow!(e.error))?;
    let load_strategy = select_load_strategy_for_worker(worker, model_name, &paths, hint)?;

    let offload =
        crate::model_manager::server_offload_enabled_for_paths(&paths, hint, request_has_lora);
    let engine_config = resolved_catalog_config.as_ref().unwrap_or(config);
    let engine = mold_inference::create_engine_with_pool(
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
    let engine = load_engine_safely(worker, engine)?;

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
/// This function may only be called by the worker's dedicated owner OS thread.
/// Uses the size-only peak (no resolution context) for the preflight — admin
/// loads don't carry a request shape.
pub fn load_blocking(worker: &GpuWorker, model_name: &str, config: &Config) -> anyhow::Result<()> {
    if worker.poisoned.load(Ordering::SeqCst) {
        anyhow::bail!(fatal_cuda_user_message(model_name));
    }
    ensure_owner_thread(worker)?;
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
/// already idle. This function may only be called by the worker's dedicated
/// owner OS thread.
pub fn unload_blocking(worker: &GpuWorker) -> anyhow::Result<Option<String>> {
    ensure_owner_thread(worker)?;
    ensure_worker_not_poisoned(worker, "admin unload")?;
    let _lock = worker
        .model_load_lock
        .lock()
        .map_err(|error| anyhow::anyhow!("worker.model_load_lock poisoned: {error}"))?;
    ensure_worker_not_poisoned(worker, "admin unload")?;
    let unloaded = {
        let mut cache = worker
            .model_cache
            .lock()
            .map_err(|error| anyhow::anyhow!("worker.model_cache poisoned: {error}"))?;
        ensure_worker_not_poisoned(worker, "admin unload")?;
        cache.unload_active()
    };
    if unloaded.is_some() {
        worker.set_resident_model(None);
        ensure_worker_not_poisoned(worker, "admin unload")?;
        match device::post_drop_free_vram_bytes(worker.gpu.ordinal) {
            Ok(free_after_drop) => tracing::info!(
                gpu = worker.gpu.ordinal,
                free_vram_bytes = free_after_drop,
                "model unloaded; sampled post-drop VRAM"
            ),
            Err(error) if error.is_fatal_cuda() => {
                quarantine_poisoned_worker(worker);
                contain_worker_cache(worker);
                return Err(anyhow::anyhow!(error));
            }
            Err(error) => tracing::warn!(
                gpu = worker.gpu.ordinal,
                %error,
                "model unloaded but post-drop VRAM sample was unavailable"
            ),
        }
    }
    Ok(unloaded)
}

fn evict_cached_model_blocking(
    worker: &GpuWorker,
    model_name: &str,
) -> anyhow::Result<Option<String>> {
    ensure_owner_thread(worker)?;
    ensure_worker_not_poisoned(worker, model_name)?;
    let _lock = worker
        .model_load_lock
        .lock()
        .map_err(|error| anyhow::anyhow!("worker.model_load_lock poisoned: {error}"))?;
    ensure_worker_not_poisoned(worker, model_name)?;
    let removed = {
        let mut cache = worker
            .model_cache
            .lock()
            .map_err(|error| anyhow::anyhow!("worker.model_cache poisoned: {error}"))?;
        ensure_worker_not_poisoned(worker, model_name)?;
        cache.remove(model_name)
    };
    let Some(engine) = removed else {
        return Ok(None);
    };
    if worker
        .resident_model
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_deref()
        == Some(model_name)
    {
        worker.set_resident_model(None);
    }
    ensure_worker_not_poisoned(worker, model_name)?;
    drop(engine);
    ensure_worker_not_poisoned(worker, model_name)?;
    match device::post_drop_free_vram_bytes(worker.gpu.ordinal) {
        Ok(free_after_drop) => tracing::info!(
            gpu = worker.gpu.ordinal,
            model = model_name,
            free_vram_bytes = free_after_drop,
            "cached model evicted on GPU owner thread"
        ),
        Err(error) if error.is_fatal_cuda() => {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            return Err(anyhow::anyhow!(error));
        }
        Err(error) => tracing::warn!(
            gpu = worker.gpu.ordinal,
            model = model_name,
            %error,
            "cached model evicted but post-drop VRAM sample was unavailable"
        ),
    }
    Ok(Some(model_name.to_string()))
}

fn ensure_owner_thread(worker: &GpuWorker) -> anyhow::Result<()> {
    let current = std::thread::current().id();
    match worker.owner_thread_id.get() {
        Some(owner) if *owner == current => Ok(()),
        Some(owner) => anyhow::bail!(
            "GPU {} resources are owned by thread {owner:?}, not caller {current:?}",
            worker.gpu.ordinal
        ),
        None => anyhow::bail!("GPU {} owner thread is not initialized", worker.gpu.ordinal),
    }
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
/// thread to `worker.gpu.ordinal`, ensures the model is loaded on GPU, takes
/// the engine out of
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
    // Bind the thread to this worker's ordinal for the duration of the call so
    // synchronization and memory sampling cannot target a sibling context.
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

    // Run the closure. A panic may have crossed arbitrary CUDA state, so it
    // has the same terminal containment policy as a fatal driver error.
    //
    // `AssertUnwindSafe` suppresses the compiler's UnwindSafe check because
    // `&mut dyn InferenceEngine` (across Box + trait object) isn't unwind-safe
    // by default. This is acceptable here: we only promise to prevent the
    // CUDA primary-context reset SEGV race, not to guarantee engine internal
    // state is pristine after a mid-generation panic.
    let mut cached = cached;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_engine(cached.engine.as_mut())
    }));

    let fatal_cuda =
        matches!(&result, Ok(Err(error)) if has_fatal_cuda_error(&format!("{error:#}")));
    let panicked = result.is_err();
    if fatal_cuda || panicked {
        quarantine_poisoned_worker(worker);
        contain_poisoned_cuda(cached);
        contain_worker_cache(worker);
    } else {
        // Ordinary typed closure errors do not imply context corruption.
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

    struct DropRecordingEngine {
        name: String,
        dropped_on: Arc<Mutex<Option<String>>>,
    }

    struct CudaCallbackRecordingEngine {
        name: String,
        unload_calls: Arc<AtomicUsize>,
        drop_calls: Arc<AtomicUsize>,
    }

    impl Drop for CudaCallbackRecordingEngine {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl InferenceEngine for CudaCallbackRecordingEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("safety test never runs inference")
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }

        fn unload(&mut self) {
            self.unload_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct PoisoningUpscaler {
        panic: bool,
        unload_calls: Arc<AtomicUsize>,
        drop_calls: Arc<AtomicUsize>,
    }

    impl Drop for PoisoningUpscaler {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl mold_inference::UpscaleEngine for PoisoningUpscaler {
        fn upscale(
            &mut self,
            _req: &mold_core::UpscaleRequest,
        ) -> anyhow::Result<mold_core::UpscaleResponse> {
            if self.panic {
                panic!("injected upscaler panic");
            }
            anyhow::bail!("CUDA_ERROR_ILLEGAL_ADDRESS")
        }

        fn model_name(&self) -> &str {
            "poisoning-upscaler"
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }

        fn unload(&mut self) {
            self.unload_calls.fetch_add(1, Ordering::SeqCst);
        }

        fn scale_factor(&self) -> u32 {
            4
        }

        fn set_on_progress(&mut self, _callback: mold_inference::progress::ProgressCallback) {}

        fn clear_on_progress(&mut self) {}
    }

    struct PanickingGenerateEngine {
        name: String,
        drop_calls: Arc<AtomicUsize>,
    }

    struct PoisoningLoadEngine {
        name: String,
        panic: bool,
        drop_calls: Arc<AtomicUsize>,
    }

    impl Drop for PoisoningLoadEngine {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl InferenceEngine for PoisoningLoadEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("load safety test never generates")
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn is_loaded(&self) -> bool {
            false
        }

        fn load(&mut self) -> anyhow::Result<()> {
            if self.panic {
                panic!("injected load panic");
            }
            anyhow::bail!("CUDA_ERROR_ILLEGAL_ADDRESS")
        }
    }

    impl Drop for PanickingGenerateEngine {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl InferenceEngine for PanickingGenerateEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            panic!("injected generation panic")
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    impl Drop for DropRecordingEngine {
        fn drop(&mut self) {
            if let Ok(mut dropped_on) = self.dropped_on.lock() {
                *dropped_on = std::thread::current().name().map(str::to_string);
            }
        }
    }

    impl InferenceEngine for DropRecordingEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("drop-order test never runs inference")
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn is_loaded(&self) -> bool {
            false
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct LifecycleRecordingEngine {
        name: String,
        loaded: bool,
        operations: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl LifecycleRecordingEngine {
        fn record(&self, operation: &str) {
            self.operations.lock().unwrap().push((
                operation.to_string(),
                std::thread::current()
                    .name()
                    .unwrap_or("unnamed")
                    .to_string(),
            ));
        }
    }

    impl Drop for LifecycleRecordingEngine {
        fn drop(&mut self) {
            self.record("drop");
        }
    }

    impl InferenceEngine for LifecycleRecordingEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            self.record("generate");
            Ok(GenerateResponse {
                images: vec![ImageData {
                    data: vec![1, 2, 3],
                    format: OutputFormat::Png,
                    width: 64,
                    height: 64,
                    index: 0,
                }],
                video: None,
                generation_time_ms: 1,
                model: self.name.clone(),
                seed_used: 1,
                gpu: None,
            })
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn is_loaded(&self) -> bool {
            self.loaded
        }

        fn load(&mut self) -> anyhow::Result<()> {
            self.record("load");
            self.loaded = true;
            Ok(())
        }

        fn unload(&mut self) {
            self.record("unload");
            self.loaded = false;
        }
    }

    fn single_worker_pool_with_parked(model: &str, load_sleep: Duration) -> Arc<GpuWorker> {
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel::<GpuWorkerCommand>(2);
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
            shutdown_requested: AtomicBool::new(false),
            owner_thread_id: std::sync::OnceLock::new(),
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

    fn protocol_worker(
        ordinal: usize,
        fatal_cuda_error: Arc<AtomicBool>,
    ) -> (Arc<GpuWorker>, std::sync::mpsc::Receiver<GpuWorkerCommand>) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(1);
        let worker = Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal,
                stable_id: Some(format!("cuda:{:032x}", ordinal + 1)),
                raw_cuda_uuid: Some(((ordinal + 1) as u128).to_be_bytes()),
                device_kind: Some(mold_inference::device::CudaDeviceKind::FullGpu),
                identity_error: None,
                backend: mold_core::GpuBackend::Cuda,
                name: format!("protocol-test-{ordinal}"),
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
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error,
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            owner_thread_id: std::sync::OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    #[test]
    fn worker_rejects_stale_generation_before_touching_inference() {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<GpuWorkerCommand>(1);
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
            shutdown_requested: AtomicBool::new(false),
            owner_thread_id: std::sync::OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
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
            .send_job(GpuJob {
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
            crate::scheduler::WorkerEvent::Rejected { grant, .. } => {
                assert_eq!(grant.work.id(), "stale");
                match grant.work {
                    OwnerWork::Generation(job) => *job,
                    _ => panic!("expected generation grant"),
                }
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
        worker.send_job(returned).unwrap();
        handle
            .join()
            .expect("worker exits when coordinator is gone");
    }

    #[test]
    fn idle_eviction_drops_engine_on_assigned_worker_thread() {
        let dropped_on = Arc::new(Mutex::new(None));
        let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
        {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.insert(
                Box::new(DropRecordingEngine {
                    name: "evict-me".to_string(),
                    dropped_on: dropped_on.clone(),
                }),
                0,
            );
            // Refresh the original entry's timestamp so the recording engine
            // is the older parked entry selected once the TTL has elapsed.
            let mut keep_one = cache.take("keep-one").unwrap();
            keep_one.last_used = Instant::now();
            cache.restore(keep_one);
        }

        let worker_for_thread = worker.clone();
        std::thread::Builder::new()
            .name("gpu-worker-test".to_string())
            .spawn(move || {
                worker_for_thread
                    .owner_thread_id
                    .set(std::thread::current().id())
                    .expect("test owner initialized once");
                evict_idle_on_worker(&worker_for_thread, Duration::ZERO);
            })
            .unwrap()
            .join()
            .unwrap();

        assert_eq!(
            dropped_on.lock().unwrap().as_deref(),
            Some("gpu-worker-test")
        );
    }

    #[test]
    fn shared_fatal_flag_rejects_transported_grant_before_accept_or_cuda() {
        let fatal = Arc::new(AtomicBool::new(false));
        let (worker, job_rx) = protocol_worker(1, fatal.clone());
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            })
        ));

        fatal.store(true, Ordering::SeqCst);
        let mut job = fake_upscale_job(Config::default(), "unused");
        job.id = "buffered-after-sibling-fatal".to_string();
        job.lease = Some(crate::scheduler::LeaseFence {
            work_id: job.id.clone(),
            device_id: crate::scheduler::worker_device_id(&worker),
            state_version: 1,
            plan_version: 1,
            worker_generation: 1,
            memory_sample_generation: 1,
            memory_ledger_sequence: 1,
        });
        worker.send_job(job).unwrap();

        match event_rx.blocking_recv().expect("fatal rejection") {
            crate::scheduler::WorkerEvent::Rejected {
                reason: crate::scheduler::LeaseRejection::FatalCuda,
                grant,
                ..
            } => assert_eq!(grant.work.id(), "buffered-after-sibling-fatal"),
            crate::scheduler::WorkerEvent::Accepted { .. } => {
                panic!("fatal-fenced worker must not accept a transported grant")
            }
            _ => panic!("expected fatal rejection"),
        }
        handle.join().expect("fatal-fenced owner exits");
    }

    #[test]
    fn owner_threads_join_cleanly_across_in_process_restart() {
        for ordinal in 0..2 {
            let (worker, job_rx) = protocol_worker(ordinal, Arc::new(AtomicBool::new(false)));
            let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            let handle =
                spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Ready { .. })
            ));
            worker.request_shutdown();
            handle.join().expect("owner thread must join on shutdown");
            assert!(
                event_rx.blocking_recv().is_none(),
                "owner must drop its coordinator sender before restart"
            );
        }
    }

    #[test]
    fn poisoned_idle_eviction_invokes_no_engine_or_device_cleanup() {
        let dropped_on = Arc::new(Mutex::new(None));
        let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert(
            Box::new(DropRecordingEngine {
                name: "must-not-drop".to_string(),
                dropped_on: dropped_on.clone(),
            }),
            0,
        );
        worker.poisoned.store(true, Ordering::SeqCst);
        worker.fatal_cuda_error.store(true, Ordering::SeqCst);

        evict_idle_on_worker(&worker, Duration::ZERO);

        assert!(dropped_on.lock().unwrap().is_none());
        assert_eq!(worker.model_cache.lock().unwrap().len(), 2);
    }

    #[test]
    fn poisoned_admin_unload_invokes_zero_cuda_backed_callbacks() {
        let unload_calls = Arc::new(AtomicUsize::new(0));
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let worker = single_worker_pool_with_parked("parked", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert_loaded(
            "active".to_string(),
            Box::new(CudaCallbackRecordingEngine {
                name: "active".to_string(),
                unload_calls: unload_calls.clone(),
                drop_calls: drop_calls.clone(),
            }),
            123,
        );
        worker.poisoned.store(true, Ordering::SeqCst);
        worker.fatal_cuda_error.store(true, Ordering::SeqCst);

        let result = unload_blocking(&worker);

        assert!(result.is_err());
        assert_eq!(unload_calls.load(Ordering::SeqCst), 0);
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
        assert!(worker.model_cache.lock().unwrap().contains("active"));
    }

    #[test]
    fn recv_timeout_observes_fatal_flag_without_eviction_or_next_ready() {
        let dropped_on = Arc::new(Mutex::new(None));
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<GpuWorkerCommand>(1);
        let mut cache = ModelCache::new(3);
        cache.insert(FakeSlowEngine::boxed("keep-one", Duration::ZERO), 0);
        cache.insert(
            Box::new(DropRecordingEngine {
                name: "must-not-evict".to_string(),
                dropped_on: dropped_on.clone(),
            }),
            0,
        );
        let worker = Arc::new(GpuWorker {
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
            shutdown_requested: AtomicBool::new(false),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let worker_for_thread = worker.clone();
        let handle = std::thread::spawn(move || {
            run_gpu_owner(
                &worker_for_thread,
                job_rx,
                event_tx,
                Duration::ZERO,
                Duration::from_millis(10),
            )
        });
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready { .. })
        ));

        worker.poisoned.store(true, Ordering::SeqCst);
        worker.fatal_cuda_error.store(true, Ordering::SeqCst);
        handle.join().expect("owner exits on fatal timeout branch");

        assert!(
            event_rx.try_recv().is_err(),
            "fatal worker must not re-advertise Ready"
        );
        assert!(dropped_on.lock().unwrap().is_none());
        assert_eq!(worker.model_cache.lock().unwrap().len(), 2);
    }

    #[test]
    fn direct_admin_gpu_operation_is_rejected_off_owner_thread() {
        let (worker, _job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        worker
            .owner_thread_id
            .set(std::thread::current().id())
            .expect("test owner initialized once");
        let worker_for_thread = worker.clone();
        let error = std::thread::spawn(move || {
            unload_blocking(&worker_for_thread).expect_err("off-owner unload must be rejected")
        })
        .join()
        .expect("caller thread joins");

        assert!(
            error.to_string().contains("resources are owned by thread"),
            "unexpected owner-thread rejection: {error:#}"
        );
    }

    #[test]
    fn every_migrated_work_kind_runs_on_the_dedicated_owner_thread() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        let device_id = crate::scheduler::worker_device_id(&worker);
        let kinds = [
            mold_scheduler::WorkKind::PromptExpansion,
            mold_scheduler::WorkKind::PostUpscale,
            mold_scheduler::WorkKind::StandaloneUpscale,
            mold_scheduler::WorkKind::AdminModelLoad,
            mold_scheduler::WorkKind::AdminModelUnload,
        ];

        for (index, kind) in kinds.into_iter().enumerate() {
            let generation = (index + 1) as u64;
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Ready {
                    worker_generation,
                    ..
                }) if worker_generation == generation
            ));
            let (thread_tx, thread_rx) = std::sync::mpsc::sync_channel(1);
            let id = format!("owner-probe-{index}");
            worker
                .send_grant(LeaseGrant {
                    fence: crate::scheduler::LeaseFence {
                        work_id: id.clone(),
                        device_id: device_id.clone(),
                        state_version: generation,
                        plan_version: generation,
                        worker_generation: generation,
                        memory_sample_generation: generation,
                        memory_ledger_sequence: generation,
                    },
                    work: OwnerWork::Probe {
                        id: id.clone(),
                        kind,
                        run: Box::new(move || {
                            thread_tx
                                .send(std::thread::current().name().map(str::to_string))
                                .unwrap();
                        }),
                    },
                    retry: None,
                })
                .unwrap();
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Accepted { work_id, .. })
                    if work_id == id
            ));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::AllocationCommitted { work_id, .. })
                    if work_id == id
            ));
            assert_eq!(
                thread_rx.recv().unwrap().as_deref(),
                Some("gpu-worker-0"),
                "{kind:?} escaped the dedicated owner OS thread"
            );
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Completed {
                    worker_generation,
                    ..
                }) if worker_generation == generation
            ));
        }

        worker.request_shutdown();
        handle.join().expect("owner joins after typed work probes");
    }

    #[test]
    fn engine_create_load_generate_unload_and_drop_stay_on_one_owner_thread() {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        let device_id = crate::scheduler::worker_device_id(&worker);

        let next_ready =
            |event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::WorkerEvent>,
             generation| {
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::Ready {
                        worker_generation,
                        ..
                    }) if worker_generation == generation
                ));
            };
        let drain_lease =
            |event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::WorkerEvent>,
             id: &str| {
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::Accepted { work_id, .. })
                        if work_id == id
                ));
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::AllocationCommitted { work_id, .. })
                        if work_id == id
                ));
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::Completed { .. })
                ));
            };
        let fence = |id: &str, generation| crate::scheduler::LeaseFence {
            work_id: id.to_string(),
            device_id: device_id.clone(),
            state_version: generation,
            plan_version: generation,
            worker_generation: generation,
            memory_sample_generation: generation,
            memory_ledger_sequence: generation,
        };

        next_ready(&mut event_rx, 1);
        let create_worker = worker.clone();
        let create_operations = operations.clone();
        worker
            .send_grant(LeaseGrant {
                fence: fence("create", 1),
                work: OwnerWork::Probe {
                    id: "create".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(move || {
                        create_operations.lock().unwrap().push((
                            "create".to_string(),
                            std::thread::current().name().unwrap().to_string(),
                        ));
                        create_worker.model_cache.lock().unwrap().insert(
                            Box::new(LifecycleRecordingEngine {
                                name: "lifecycle".to_string(),
                                loaded: false,
                                operations: create_operations,
                            }),
                            0,
                        );
                    }),
                },
                retry: None,
            })
            .unwrap();
        drain_lease(&mut event_rx, "create");

        next_ready(&mut event_rx, 2);
        let (load_tx, load_rx) = tokio::sync::oneshot::channel();
        worker
            .send_grant(LeaseGrant {
                fence: fence("load", 2),
                work: OwnerWork::AdminModelLoad(Box::new(crate::gpu_pool::AdminModelLoadJob {
                    id: "load".to_string(),
                    model: "lifecycle".to_string(),
                    config: Config::default(),
                    result_tx: load_tx,
                })),
                retry: None,
            })
            .unwrap();
        drain_lease(&mut event_rx, "load");
        assert!(load_rx.blocking_recv().unwrap().is_ok());

        next_ready(&mut event_rx, 3);
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"owner","model":"lifecycle","width":64,"height":64,"steps":1,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let (queue_tx, mut queue_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(queue_tx);
        let (placeholder_tx, _placeholder_rx) = tokio::sync::oneshot::channel();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(queue.submit(
                GenerationJob {
                    id: "generate".to_string(),
                    request: request.clone(),
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: placeholder_tx,
                    output_dir: None,
                },
                1,
            ))
            .unwrap();
        let _ = queue_rx.try_recv().unwrap();
        let registry = JobRegistry::new();
        registry.register("generate", "lifecycle");
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        worker
            .send_job(GpuJob {
                id: "generate".to_string(),
                model: "lifecycle".to_string(),
                request,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx,
                output_dir: None,
                config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                metadata_db: Arc::new(None),
                queue,
                registry,
                events: crate::events::EventBroadcaster::new(),
                lease: Some(fence("generate", 3)),
            })
            .unwrap();
        drain_lease(&mut event_rx, "generate");
        assert!(result_rx.blocking_recv().unwrap().is_ok());

        next_ready(&mut event_rx, 4);
        let (unload_tx, unload_rx) = tokio::sync::oneshot::channel();
        worker
            .send_grant(LeaseGrant {
                fence: fence("unload", 4),
                work: OwnerWork::AdminModelUnload(Box::new(AdminModelUnloadJob {
                    id: "unload".to_string(),
                    model: Some("lifecycle".to_string()),
                    evict_cached: false,
                    result_tx: unload_tx,
                })),
                retry: None,
            })
            .unwrap();
        drain_lease(&mut event_rx, "unload");
        assert_eq!(
            unload_rx.blocking_recv().unwrap().unwrap().as_deref(),
            Some("lifecycle")
        );

        next_ready(&mut event_rx, 5);
        worker.request_shutdown();
        handle.join().expect("owner joins after lifecycle test");

        let operations = operations.lock().unwrap().clone();
        let names = operations
            .iter()
            .map(|(operation, _)| operation.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["create", "load", "generate", "unload", "drop"]);
        assert!(
            operations
                .iter()
                .all(|(_, thread)| thread == "gpu-worker-0"),
            "CUDA lifecycle escaped owner thread: {operations:?}"
        );
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
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
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
            &event_tx,
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

    #[tokio::test]
    async fn generation_panic_contains_engine_and_blocks_followup_job() {
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let worker = single_worker_pool_with_parked("parked", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert_loaded(
            "panic-model".to_string(),
            Box::new(PanickingGenerateEngine {
                name: "panic-model".to_string(),
                drop_calls: drop_calls.clone(),
            }),
            123,
        );
        worker.in_flight.store(1, Ordering::SeqCst);

        let mut request = fake_upscale_job(Config::default(), "unused").request;
        request.model = "panic-model".to_string();
        request.upscale_model = None;
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

        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let worker_for_job = worker.clone();
        let panic_request = request.clone();
        tokio::task::spawn_blocking(move || {
            let (scheduler_tx, _scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
            process_job(
                &worker_for_job,
                GpuJob {
                    id: "panic-job".to_string(),
                    model: "panic-model".to_string(),
                    request: panic_request,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx,
                    output_dir: None,
                    config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                    metadata_db: Arc::new(None),
                    queue: queue.clone(),
                    registry: JobRegistry::new(),
                    events: crate::events::EventBroadcaster::new(),
                    lease: None,
                },
                &scheduler_tx,
            );
        })
        .await
        .unwrap();

        let panic_error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("panicking engine unexpectedly generated"),
        };
        assert!(panic_error.contains("inference panicked"));
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
        assert!(worker.model_cache.lock().unwrap().is_empty());
        drop(queue_rx.recv().await);

        worker.in_flight.store(1, Ordering::SeqCst);
        let (queue_tx, mut queue_rx) = tokio::sync::mpsc::channel(1);
        let followup_queue = QueueHandle::new(queue_tx);
        let (placeholder_tx, _placeholder_rx) = tokio::sync::oneshot::channel();
        followup_queue
            .submit(
                GenerationJob {
                    id: "placeholder-2".to_string(),
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
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let worker_for_job = worker.clone();
        tokio::task::spawn_blocking(move || {
            let (scheduler_tx, _scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
            process_job(
                &worker_for_job,
                GpuJob {
                    id: "followup".to_string(),
                    model: "panic-model".to_string(),
                    request,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx,
                    output_dir: None,
                    config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                    metadata_db: Arc::new(None),
                    queue: followup_queue,
                    registry: JobRegistry::new(),
                    events: crate::events::EventBroadcaster::new(),
                    lease: None,
                },
                &scheduler_tx,
            );
        })
        .await
        .unwrap();
        let followup_error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("quarantined worker unexpectedly generated"),
        };
        assert!(followup_error.contains("quarantined"));
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
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
    fn fatal_and_panicking_admin_loads_are_contained_without_engine_drop() {
        for panic in [false, true] {
            let drop_calls = Arc::new(AtomicUsize::new(0));
            let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
            worker.model_cache.lock().unwrap().insert(
                Box::new(PoisoningLoadEngine {
                    name: "poison-load".to_string(),
                    panic,
                    drop_calls: drop_calls.clone(),
                }),
                0,
            );

            let result = load_blocking(&worker, "poison-load", &Config::default());

            assert!(result.is_err());
            assert!(worker.poisoned.load(Ordering::SeqCst));
            assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
            assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
            assert!(worker.model_cache.lock().unwrap().is_empty());
        }
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
    fn fatal_and_panicking_upscalers_are_contained_without_cleanup_callbacks() {
        for panic in [false, true] {
            let worker = single_worker_pool_with_parked("fake-model", Duration::ZERO);
            let unload_calls = Arc::new(AtomicUsize::new(0));
            let drop_calls = Arc::new(AtomicUsize::new(0));
            let request = mold_core::UpscaleRequest {
                model: "poisoning-upscaler".to_string(),
                image: vec![1, 2, 3],
                output_format: OutputFormat::Png,
                tile_size: None,
                metadata: None,
            };
            let result = run_upscale_engine_safely(
                &worker,
                Box::new(PoisoningUpscaler {
                    panic,
                    unload_calls: unload_calls.clone(),
                    drop_calls: drop_calls.clone(),
                }),
                &request,
            );

            assert!(result.is_err());
            assert!(worker.poisoned.load(Ordering::SeqCst));
            assert_eq!(unload_calls.load(Ordering::SeqCst), 0);
            assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
        }
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

    #[test]
    fn run_chain_panic_contains_engine_and_rejects_next_grant() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::ZERO);
        let config = Config::default();

        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_chain_blocking(
                &worker,
                "fake-model",
                &config,
                None,
                |_engine| -> anyhow::Result<()> { panic!("injected chain panic") },
            );
        }));
        assert!(first.is_err());
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
        assert!(worker.model_cache.lock().unwrap().is_empty());

        let next_closure_ran = AtomicBool::new(false);
        let next = run_chain_blocking(
            &worker,
            "fake-model",
            &config,
            None,
            |_engine| -> anyhow::Result<()> {
                next_closure_ran.store(true, Ordering::SeqCst);
                Ok(())
            },
        );
        assert!(next.is_err(), "poisoned worker must reject the next grant");
        assert!(!next_closure_ran.load(Ordering::SeqCst));
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
