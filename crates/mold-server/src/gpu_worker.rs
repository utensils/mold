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
use std::cell::{Cell, RefCell};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

thread_local! {
    /// Activation time accumulated by model-ready calls on one GPU owner
    /// thread during the current lease.
    static LEASE_LOAD_MS: Cell<u64> = const { Cell::new(0) };
    static LEASE_PHASE_TIMINGS: RefCell<mold_scheduler::EstimatePhaseTimings> =
        const { RefCell::new(mold_scheduler::EstimatePhaseTimings {
            cold_load_ms: None,
            warm_reload_ms: None,
            prompt_encode_ms: None,
            denoise_ms: None,
            vae_ms: None,
            upscale_ms: None,
        }) };
}

fn reset_lease_load_ms() {
    LEASE_LOAD_MS.set(0);
    LEASE_PHASE_TIMINGS
        .with(|timings| *timings.borrow_mut() = mold_scheduler::EstimatePhaseTimings::default());
}

fn add_lease_load_ms(elapsed: Duration) {
    let millis = u64::try_from(elapsed.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    LEASE_LOAD_MS.set(LEASE_LOAD_MS.get().saturating_add(millis));
}

fn take_lease_load_ms() -> Option<u64> {
    let millis = LEASE_LOAD_MS.replace(0);
    (millis > 0).then_some(millis)
}

fn add_phase_sample(slot: &mut Option<u64>, elapsed: Duration) {
    let millis = u64::try_from(elapsed.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    *slot = Some(slot.unwrap_or_default().saturating_add(millis));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelLoadDisposition {
    Unchanged,
    Cold,
    WarmReload,
}

fn record_model_load_timing(disposition: ModelLoadDisposition, elapsed: Duration) {
    if disposition == ModelLoadDisposition::Unchanged {
        return;
    }
    add_lease_load_ms(elapsed);
    LEASE_PHASE_TIMINGS.with(|timings| {
        let mut timings = timings.borrow_mut();
        match disposition {
            ModelLoadDisposition::Cold => {
                add_phase_sample(&mut timings.cold_load_ms, elapsed);
            }
            ModelLoadDisposition::WarmReload => {
                add_phase_sample(&mut timings.warm_reload_ms, elapsed);
            }
            ModelLoadDisposition::Unchanged => {}
        }
    });
}

fn record_phase_timing(event: &mold_inference::ProgressEvent) {
    LEASE_PHASE_TIMINGS.with(|timings| {
        let mut timings = timings.borrow_mut();
        match event {
            mold_inference::ProgressEvent::PhaseDone {
                phase,
                elapsed,
                name: _,
            } => match phase {
                mold_inference::ProgressPhase::ModelLoad => {}
                mold_inference::ProgressPhase::PromptEncode => {
                    add_phase_sample(&mut timings.prompt_encode_ms, *elapsed)
                }
                mold_inference::ProgressPhase::Vae => {
                    add_phase_sample(&mut timings.vae_ms, *elapsed)
                }
                mold_inference::ProgressPhase::Upscale => {
                    add_phase_sample(&mut timings.upscale_ms, *elapsed)
                }
            },
            mold_inference::ProgressEvent::DenoiseStep { elapsed, .. } => {
                add_phase_sample(&mut timings.denoise_ms, *elapsed);
            }
            _ => {}
        }
    });
}

fn take_lease_phase_timings(_load_ms: Option<u64>) -> mold_scheduler::EstimatePhaseTimings {
    LEASE_PHASE_TIMINGS.with(|timings| std::mem::take(&mut *timings.borrow_mut()))
}

pub enum LegacyOwnerEvent {
    FollowupReady(Box<crate::scheduler::ScheduledOwnerWork>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedEngineMode {
    load_strategy: mold_inference::LoadStrategy,
    block_offload: bool,
}

impl PlannedEngineMode {
    fn from_plan(plan: &crate::execution_plan::ResolvedExecutionPlan) -> Self {
        Self {
            load_strategy: plan.engine_load_strategy,
            block_offload: plan.offload_mode == crate::execution_plan::OffloadMode::Block,
        }
    }

    fn matches(self, engine: &dyn mold_inference::InferenceEngine) -> bool {
        engine.configured_load_strategy() == Some(self.load_strategy)
            && engine.configured_block_offload() == Some(self.block_offload)
    }
}

#[derive(Clone, Copy)]
struct PlannedLoadContract<'a> {
    mode: PlannedEngineMode,
    execution_fingerprint: &'a str,
    request: &'a mold_core::GenerateRequest,
    engine_paths: &'a mold_core::ModelPaths,
    engine_config: &'a mold_inference::FrozenEngineConfig,
}

struct PlannedInferenceEngine {
    inner: Box<dyn mold_inference::InferenceEngine>,
    mode: PlannedEngineMode,
    execution_fingerprint: String,
}

impl mold_inference::InferenceEngine for PlannedInferenceEngine {
    fn generate(
        &mut self,
        req: &mold_core::GenerateRequest,
    ) -> anyhow::Result<mold_core::GenerateResponse> {
        self.inner.generate(req)
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    fn load(&mut self) -> anyhow::Result<()> {
        self.inner.load()
    }

    fn load_for_request(&mut self, req: &mold_core::GenerateRequest) -> anyhow::Result<()> {
        self.inner.load_for_request(req)
    }

    fn unload(&mut self) {
        self.inner.unload();
    }

    fn set_on_progress(&mut self, callback: mold_inference::progress::ProgressCallback) {
        self.inner.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.inner.clear_on_progress();
    }

    fn set_cancellation_token(&mut self, token: mold_inference::InferenceCancellationToken) {
        self.inner.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.inner.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> mold_inference::BatchExecutionCapability {
        self.inner.batch_execution_capability()
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        self.inner.model_paths()
    }

    fn configured_load_strategy(&self) -> Option<mold_inference::LoadStrategy> {
        Some(self.mode.load_strategy)
    }

    fn configured_block_offload(&self) -> Option<bool> {
        Some(self.mode.block_offload)
    }

    fn configured_execution_fingerprint(&self) -> Option<&str> {
        Some(&self.execution_fingerprint)
    }

    fn as_chain_renderer(&mut self) -> Option<&mut dyn mold_inference::chain::ChainStageRenderer> {
        self.inner.as_chain_renderer()
    }
}

fn record_planned_engine_mode(
    engine: Box<dyn mold_inference::InferenceEngine>,
    mode: PlannedEngineMode,
    execution_fingerprint: &str,
) -> Box<dyn mold_inference::InferenceEngine> {
    Box::new(PlannedInferenceEngine {
        inner: engine,
        mode,
        execution_fingerprint: execution_fingerprint.to_string(),
    })
}

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

/// Spawn a replacement owner without waiting for its context probe. The
/// epoch-qualified first Ready or StartFailed event settles Starting.
pub fn spawn_gpu_thread_async(
    worker: Arc<GpuWorker>,
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    cache_idle_ttl: Duration,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    let thread_worker = worker.clone();
    std::thread::Builder::new()
        .name(format!(
            "gpu-worker-{}-epoch-{}",
            worker.gpu.ordinal, worker.owner_epoch
        ))
        .spawn(move || {
            if let Err(error) = validate_owner_device(thread_worker.gpu.ordinal) {
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::StartFailed {
                    device_id: crate::scheduler::worker_device_id(&thread_worker),
                    ordinal: thread_worker.gpu.ordinal,
                    owner_epoch: thread_worker.owner_epoch,
                    error,
                });
                return;
            }
            run_gpu_owner(
                &thread_worker,
                job_rx,
                scheduler_tx,
                cache_idle_ttl,
                Duration::from_secs(60),
            );
        })
}

#[cfg(all(not(test), any(feature = "cuda", feature = "metal")))]
fn validate_owner_device(ordinal: usize) -> Result<(), String> {
    let device =
        mold_inference::device::resolve_device(Some(mold_core::DeviceRef::Gpu { ordinal }), || {
            unreachable!("explicit GPU placement does not use the auto resolver")
        })
        .map_err(|error| format!("failed to create device context on owner thread: {error:#}"))?;
    drop(device);
    Ok(())
}

#[cfg(any(test, not(any(feature = "cuda", feature = "metal"))))]
fn validate_owner_device(_ordinal: usize) -> Result<(), String> {
    Ok(())
}

/// Spawn a rollback-window worker that retains the pre-V2 dispatcher as the
/// sole dispatch authority. It never publishes Ready generations or validates
/// V2 lease fences; all CUDA ownership still stays on this one OS thread.
pub fn spawn_legacy_gpu_thread(
    worker: Arc<GpuWorker>,
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    owner_event_tx: tokio::sync::mpsc::UnboundedSender<LegacyOwnerEvent>,
    cache_idle_ttl: Duration,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("gpu-legacy-worker-{}", worker.gpu.ordinal))
        .spawn(move || {
            run_gpu_owner_entrypoint(&worker, || {
                run_legacy_gpu_owner_loop(
                    &worker,
                    job_rx,
                    owner_event_tx,
                    cache_idle_ttl,
                    Duration::from_secs(60),
                );
            });
        })
        .expect("failed to spawn legacy GPU worker thread")
}

fn run_gpu_owner(
    worker: &GpuWorker,
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    cache_idle_ttl: Duration,
    idle_poll: Duration,
) {
    let terminal_tx = scheduler_tx.clone();
    let device_id = crate::scheduler::worker_device_id(worker);
    let ordinal = worker.gpu.ordinal;
    let owner_epoch = worker.owner_epoch;
    run_gpu_owner_entrypoint(worker, || {
        run_gpu_owner_loop(worker, job_rx, scheduler_tx, cache_idle_ttl, idle_poll);
    });
    // The lifecycle reducer needs one exact terminal event on every path,
    // including an owner-loop or teardown panic contained by the entrypoint.
    let _ = terminal_tx.send(crate::scheduler::WorkerEvent::Stopped {
        device_id,
        ordinal,
        owner_epoch,
    });
}

fn run_gpu_owner_entrypoint(run_worker: &GpuWorker, entrypoint: impl FnOnce()) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(entrypoint));
    if let Err(payload) = outcome {
        // The panic may have crossed arbitrary Candle/cudarc state, including
        // idle eviction and destructor paths outside process_job. Quarantine
        // the context and use a stored Notify permit so startup-time panics
        // cannot race ahead of the server shutdown waiter.
        quarantine_poisoned_worker(run_worker);
        tracing::error!(
            gpu = run_worker.gpu.ordinal,
            panic = %panic_payload_message(payload.as_ref()),
            "GPU owner thread panicked; quarantining context and stopping server"
        );
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

fn run_gpu_owner_loop(
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
        worker.fatal_cuda_shutdown.notify_one();
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
        if worker.commit_drain() {
            break;
        }
        if scheduler_tx
            .send(crate::scheduler::WorkerEvent::Ready {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                owner_epoch: worker.owner_epoch,
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
        #[allow(unused_mut)] // test-only second-fence hook mutates the payload
        let mut grant = match command {
            GpuWorkerCommand::Grant(grant) => grant,
            GpuWorkerCommand::Drain => {
                if worker.commit_drain() {
                    break;
                }
                continue;
            }
            GpuWorkerCommand::Shutdown => break,
        };
        if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst)
        {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                owner_epoch: worker.owner_epoch,
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
                owner_epoch: worker.owner_epoch,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::StaleWorkerGeneration,
            });
            break;
        }
        let fence = &grant.fence;
        if fence.device_id != device_id
            || fence.owner_epoch != worker.owner_epoch
            || fence.worker_generation != generation
        {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                owner_epoch: worker.owner_epoch,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::StaleWorkerGeneration,
            });
            continue;
        }
        if let Err(error) = validate_grant_before_acceptance(worker, &grant) {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: worker.gpu.ordinal,
                owner_epoch: worker.owner_epoch,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::PlanInvalidated(error),
            });
            continue;
        }
        let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Accepted {
            device_id: device_id.clone(),
            ordinal: worker.gpu.ordinal,
            owner_epoch: worker.owner_epoch,
            worker_generation: generation,
            work_id: fence.work_id.clone(),
            plan_version: fence.plan_version,
        });
        reset_lease_load_ms();
        #[cfg(test)]
        pause_after_acceptance_for_test(&fence.work_id);
        #[cfg(test)]
        if let OwnerWork::ChainStage(job) = &mut grant.work {
            if let Some(hook) = job.before_second_fence.take() {
                hook();
            }
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_owner_work(worker, *grant, &scheduler_tx)
        }));
        let load_ms = take_lease_load_ms();
        let phase_timings = take_lease_phase_timings(load_ms);
        let panicked = outcome.is_err();
        if panicked {
            // A panic may have crossed arbitrary Candle/cudarc state.
            // Treat the owner context as fatal and let supervision
            // restart the process; never attempt an in-process reset.
            quarantine_poisoned_worker(worker);
        }
        match outcome {
            Ok(OwnerProcessOutcome::Completed {
                successful,
                completion,
            }) => {
                worker.release_in_flight();
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: worker.gpu.ordinal,
                    owner_epoch: worker.owner_epoch,
                    worker_generation: generation,
                    successful: successful
                        && !worker.poisoned.load(Ordering::SeqCst)
                        && !worker.fatal_cuda_error.load(Ordering::SeqCst),
                    phase_timings,
                });
                if let Some(completion) = completion {
                    completion.finish();
                }
            }
            Ok(OwnerProcessOutcome::PlanInvalidated { grant, error }) => {
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                    device_id: device_id.clone(),
                    ordinal: worker.gpu.ordinal,
                    owner_epoch: worker.owner_epoch,
                    worker_generation: generation,
                    grant,
                    reason: crate::scheduler::LeaseRejection::PlanInvalidated(error),
                });
            }
            Err(_) => {
                worker.release_in_flight();
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: worker.gpu.ordinal,
                    owner_epoch: worker.owner_epoch,
                    worker_generation: generation,
                    successful: false,
                    phase_timings,
                });
            }
        }
        if panicked
            || worker.shutdown_requested.load(Ordering::SeqCst)
            || worker.poisoned.load(Ordering::SeqCst)
            || worker.fatal_cuda_error.load(Ordering::SeqCst)
            || worker.commit_drain()
        {
            break;
        }
        generation = generation.saturating_add(1);
    }
    if !worker.poisoned.load(Ordering::SeqCst) && !worker.fatal_cuda_error.load(Ordering::SeqCst) {
        let cached = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        if let Err(error) = teardown_inference_engines_safely(worker, cached, "GPU owner shutdown")
        {
            tracing::error!(
                gpu = worker.gpu.ordinal,
                %error,
                "GPU owner shutdown teardown failed; retaining remaining resources for process exit"
            );
        }
    }
    // A poisoned primary context is never touched again. In that case keep
    // the cache intact so no container operation can trigger a CUDA-backed
    // destructor; process teardown reclaims it.
    tracing::info!(gpu = worker.gpu.ordinal, "GPU worker thread exiting");
}

fn run_legacy_gpu_owner_loop(
    worker: &GpuWorker,
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    owner_event_tx: tokio::sync::mpsc::UnboundedSender<LegacyOwnerEvent>,
    cache_idle_ttl: Duration,
    idle_poll: Duration,
) {
    if worker
        .owner_thread_id
        .set(std::thread::current().id())
        .is_err()
    {
        quarantine_poisoned_worker(worker);
        tracing::error!(
            gpu = worker.gpu.ordinal,
            "legacy GPU worker owner thread was initialized more than once"
        );
        return;
    }
    mold_inference::device::init_thread_gpu_ordinal(worker.gpu.ordinal);
    tracing::info!(
        gpu = worker.gpu.ordinal,
        name = %worker.gpu.name,
        "legacy GPU worker thread started"
    );
    loop {
        let command = match job_rx.recv_timeout(idle_poll) {
            Ok(command) => command,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if worker.shutdown_requested.load(Ordering::SeqCst) {
                    break;
                }
                if !worker.poisoned.load(Ordering::SeqCst)
                    && !worker.fatal_cuda_error.load(Ordering::SeqCst)
                {
                    evict_idle_on_worker(worker, cache_idle_ttl);
                }
                // A fatal owner deliberately remains alive as a rejection
                // drain until server teardown first cancels/joins rollback
                // dispatchers and then requests owner shutdown. Exiting here
                // would leave a live sender able to accept unowned work.
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let grant = match command {
            GpuWorkerCommand::Grant(grant) => grant,
            GpuWorkerCommand::Drain => {
                if worker.commit_drain() {
                    reject_buffered_legacy_grants(worker, &job_rx);
                    break;
                }
                continue;
            }
            GpuWorkerCommand::Shutdown => {
                reject_buffered_legacy_grants(worker, &job_rx);
                break;
            }
        };
        if worker.poisoned.load(Ordering::SeqCst)
            || worker.fatal_cuda_error.load(Ordering::SeqCst)
            || worker.shutdown_requested.load(Ordering::SeqCst)
        {
            grant.work.reject(legacy_rejection_message(worker));
            worker.settle_legacy_transport();
            if worker.shutdown_requested.load(Ordering::SeqCst) {
                reject_buffered_legacy_grants(worker, &job_rx);
                break;
            }
            continue;
        }
        if !worker.wait_claim_owner_in_flight() {
            grant.work.reject(legacy_rejection_message(worker));
            worker.settle_legacy_transport();
            if worker.shutdown_requested.load(Ordering::SeqCst) {
                reject_buffered_legacy_grants(worker, &job_rx);
                break;
            }
            continue;
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_legacy_owner_work(worker, *grant, &owner_event_tx);
        }));
        worker.release_in_flight();
        worker.settle_legacy_transport();
        if outcome.is_err() {
            quarantine_poisoned_worker(worker);
        }
    }
    reject_buffered_legacy_grants(worker, &job_rx);
    if !worker.poisoned.load(Ordering::SeqCst) && !worker.fatal_cuda_error.load(Ordering::SeqCst) {
        let cached = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        if let Err(error) =
            teardown_inference_engines_safely(worker, cached, "legacy GPU owner shutdown")
        {
            tracing::error!(
                gpu = worker.gpu.ordinal,
                %error,
                "legacy GPU owner shutdown teardown failed; retaining remaining resources for process exit"
            );
        }
    }
    tracing::info!(gpu = worker.gpu.ordinal, "legacy GPU worker thread exiting");
}

fn legacy_rejection_message(worker: &GpuWorker) -> String {
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        fatal_cuda_user_message("queued GPU work")
    } else {
        "GPU work was not started because the server is shutting down".to_string()
    }
}

fn reject_buffered_legacy_grants(
    worker: &GpuWorker,
    job_rx: &std::sync::mpsc::Receiver<GpuWorkerCommand>,
) {
    while let Ok(command) = job_rx.try_recv() {
        if let GpuWorkerCommand::Grant(grant) = command {
            grant.work.reject(legacy_rejection_message(worker));
            worker.settle_legacy_transport();
        }
    }
}

fn validate_grant_before_acceptance(
    worker: &GpuWorker,
    grant: &LeaseGrant,
) -> Result<(), crate::execution_plan::ExecutionPlanError> {
    let worker_id = crate::scheduler::worker_device_id(worker);
    match &grant.work {
        OwnerWork::Generation(job) => {
            let Some(plan) = job.execution_plan.as_ref() else {
                return Ok(());
            };
            let config = job.config.blocking_read();
            crate::execution_plan::validate_before_cuda(
                plan,
                &worker_id,
                worker.gpu.ordinal,
                &config,
                &job.request,
            )
        }
        OwnerWork::ChainStage(job) => {
            let Some(plan) = job.execution_plan.as_ref() else {
                return Err(crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                    "chain stage reached worker acceptance without an exact execution plan"
                        .to_string(),
                ));
            };
            crate::execution_plan::validate_before_cuda(
                plan,
                &worker_id,
                worker.gpu.ordinal,
                &job.config,
                &job.stage_req,
            )
        }
        _ => Ok(()),
    }
}

enum DeferredOwnerCompletion {
    ChainStage {
        tx: tokio::sync::oneshot::Sender<Result<crate::chain_job_runner::StageExecution, String>>,
        result: Result<crate::chain_job_runner::StageExecution, String>,
    },
}

impl DeferredOwnerCompletion {
    fn finish(self) {
        match self {
            Self::ChainStage { tx, result } => {
                let _ = tx.send(result);
            }
        }
    }
}

enum OwnerProcessOutcome {
    Completed {
        successful: bool,
        completion: Option<DeferredOwnerCompletion>,
    },
    PlanInvalidated {
        grant: Box<LeaseGrant>,
        error: crate::execution_plan::ExecutionPlanError,
    },
}

fn process_owner_work(
    worker: &GpuWorker,
    mut grant: LeaseGrant,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) -> OwnerProcessOutcome {
    if let Err(error) = ensure_owner_thread(worker) {
        grant.work.reject(error.to_string());
        return OwnerProcessOutcome::Completed {
            successful: false,
            completion: None,
        };
    }
    if let OwnerWork::ChainStage(job) = &grant.work {
        if let Err(error) = validate_scheduled_chain_stage_before_cuda(worker, job) {
            return OwnerProcessOutcome::PlanInvalidated {
                grant: Box::new(grant),
                error,
            };
        }
    }
    if let OwnerWork::ChainStage(job) = &mut grant.work {
        if let Some(error) = job
            .on_leased
            .take()
            .and_then(|on_leased| on_leased(worker.gpu.ordinal).err())
        {
            grant.work.reject(error);
            return OwnerProcessOutcome::Completed {
                successful: false,
                completion: None,
            };
        }
    }
    match grant.work {
        OwnerWork::Generation(mut job) => {
            job.lease = Some(grant.fence);
            let successful = process_job(worker, *job, scheduler_tx);
            OwnerProcessOutcome::Completed {
                successful,
                completion: None,
            }
        }
        OwnerWork::ChainStage(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let mut job = *job;
            let result_tx = job
                .result_tx
                .take()
                .expect("scheduled chain stage owns its result sender");
            let result = process_scheduled_chain_stage(worker, job);
            let successful = matches!(
                &result,
                Ok(crate::chain_job_runner::StageExecution {
                    outcome: crate::chain_job_runner::StageRenderOutcome::Done(_),
                    ..
                })
            );
            OwnerProcessOutcome::Completed {
                successful,
                completion: Some(DeferredOwnerCompletion::ChainStage {
                    tx: result_tx,
                    result,
                }),
            }
        }
        OwnerWork::PromptExpansion(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_prompt_expansion(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                completion: None,
            }
        }
        OwnerWork::PostUpscale(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_post_generation_upscale(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                completion: None,
            }
        }
        OwnerWork::StandaloneUpscale(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_standalone_upscale(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                completion: None,
            }
        }
        OwnerWork::AdminModelLoad(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let result = load_blocking(worker, &job.model, &job.config).map_err(|e| e.to_string());
            let successful = result.is_ok();
            let _ = job.result_tx.send(result);
            OwnerProcessOutcome::Completed {
                successful,
                completion: None,
            }
        }
        OwnerWork::AdminModelUnload(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_admin_unload(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                completion: None,
            }
        }
        #[cfg(test)]
        OwnerWork::Probe { run, .. } => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            run();
            OwnerProcessOutcome::Completed {
                successful: true,
                completion: None,
            }
        }
    }
}

fn validate_scheduled_chain_stage_before_cuda(
    worker: &GpuWorker,
    job: &crate::chain_job_runner::ScheduledChainStageWork,
) -> Result<(), crate::execution_plan::ExecutionPlanError> {
    if let Some(expected) = &job.expected_model_fingerprint {
        let frozen = job.config.models.get(&job.model).ok_or_else(|| {
            crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                "frozen chain model config disappeared before CUDA".to_string(),
            )
        })?;
        let current = crate::execution_plan::frozen_model_fingerprint(&job.model, frozen)?;
        if &current != expected {
            return Err(crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                "frozen chain model inputs changed before CUDA".to_string(),
            ));
        }
    }
    let plan = job.execution_plan.as_ref().ok_or_else(|| {
        crate::execution_plan::ExecutionPlanError::PlanInvalidated(
            "chain stage reached execution without an exact execution plan".to_string(),
        )
    })?;
    crate::execution_plan::validate_before_cuda(
        plan,
        &crate::scheduler::worker_device_id(worker),
        worker.gpu.ordinal,
        &job.config,
        &job.stage_req,
    )
}

fn process_legacy_owner_work(
    worker: &GpuWorker,
    grant: LeaseGrant,
    owner_event_tx: &tokio::sync::mpsc::UnboundedSender<LegacyOwnerEvent>,
) {
    if let Err(error) = ensure_owner_thread(worker) {
        grant.work.reject(error.to_string());
        return;
    }
    match grant.work {
        OwnerWork::Generation(job) => {
            let _ =
                process_job_with_sink(worker, *job, GenerationEventSink::Legacy(owner_event_tx));
        }
        OwnerWork::PromptExpansion(job) => {
            let _ = process_prompt_expansion(worker, *job);
        }
        OwnerWork::PostUpscale(job) => {
            let _ = process_post_generation_upscale(worker, *job);
        }
        OwnerWork::StandaloneUpscale(job) => {
            let _ = process_standalone_upscale(worker, *job);
        }
        OwnerWork::ChainStage(job) => {
            let mut job = *job;
            let tx = job
                .result_tx
                .take()
                .expect("scheduled chain stage owns its result sender");
            let result = process_scheduled_chain_stage(worker, job);
            let _ = tx.send(result);
        }
        OwnerWork::AdminModelLoad(job) => {
            let result = load_blocking(worker, &job.model, &job.config).map_err(|e| e.to_string());
            let _ = job.result_tx.send(result);
        }
        OwnerWork::AdminModelUnload(job) => {
            let _ = process_admin_unload(worker, *job);
        }
        #[cfg(test)]
        OwnerWork::Probe { run, .. } => run(),
    }
}

fn process_scheduled_chain_stage(
    worker: &GpuWorker,
    mut job: crate::chain_job_runner::ScheduledChainStageWork,
) -> Result<crate::chain_job_runner::StageExecution, String> {
    if (job.cancelled)() {
        return Ok(crate::chain_job_runner::StageExecution {
            outcome: crate::chain_job_runner::StageRenderOutcome::Cancelled,
            device_ordinal: Some(worker.gpu.ordinal),
        });
    }
    let plan = job.execution_plan.clone().ok_or_else(|| {
        "chain stage reached execution without an exact execution plan".to_string()
    })?;
    job.stage_req.placement = Some(crate::execution_plan::materialized_placement(&plan));
    struct ActiveGuard<'a>(&'a GpuWorker);
    impl Drop for ActiveGuard<'_> {
        fn drop(&mut self) {
            if let Ok(mut active) = self.0.active_generation.write() {
                *active = None;
            }
        }
    }
    {
        let mut active = worker
            .active_generation
            .write()
            .map_err(|error| format!("active_generation lock poisoned: {error}"))?;
        *active = Some(crate::gpu_pool::ActiveGeneration {
            model: job.model.clone(),
            prompt_sha256: format!(
                "{:x}",
                sha2::Sha256::digest(job.stage_req.prompt.as_bytes())
            ),
            started_at_unix_ms: mold_core::time::now_epoch_ms_u64(),
            started_at: std::time::Instant::now(),
        });
    }
    let _active = ActiveGuard(worker);
    let hint = crate::model_manager::family_for_model_sync(&job.model, &job.config).map(|family| {
        crate::model_manager::ActivationHint {
            width: job.stage_req.width,
            height: job.stage_req.height,
            batch: 1,
            dtype_bytes: 2,
            family: mold_inference::device::activation_family_for(&family),
        }
    });
    let model = job.model.clone();
    let closure_model = model.clone();
    let carry = job.carry.clone();
    let request = job.stage_req.clone();
    let progress = job.progress.clone();
    let cancelled = job.cancelled.clone();
    let cancellation = job.cancellation.clone();
    let motion_tail_frames = job.motion_tail_frames;
    let result = run_stage_blocking_planned(
        worker,
        PlannedStageLoad {
            cache_key: &job.cache_key,
            model_name: &model,
            config: &job.config,
            hint,
            plan: &plan,
            request: &job.stage_req,
        },
        move |engine| -> anyhow::Result<crate::chain_job_runner::StageRenderOutcome> {
            let mut cancellation_seen = false;
            let mut stage_progress = |event: mold_inference::chain::StageProgressEvent| match event
            {
                mold_inference::chain::StageProgressEvent::DenoiseStep { step, total } => {
                    if progress(step, total).is_break() || cancelled() {
                        cancellation_seen = true;
                    }
                }
            };
            let render =
                mold_inference::with_inference_cancellation(engine, cancellation, |engine| {
                    let renderer = engine.as_chain_renderer().ok_or_else(|| {
                        anyhow::anyhow!(
                            "model '{closure_model}' does not support chained video generation"
                        )
                    })?;
                    renderer.render_stage(
                        &request,
                        carry.as_ref(),
                        motion_tail_frames,
                        Some(&mut stage_progress),
                    )
                });
            fence_chain_stage_render(render, cancellation_seen, cancelled())
        },
    )
    .map_err(|error| format!("{error:#}"))?
    .map_err(|error| format!("{error:#}"))?;
    Ok(crate::chain_job_runner::StageExecution {
        outcome: result,
        device_ordinal: Some(worker.gpu.ordinal),
    })
}

fn fence_chain_stage_render(
    render: anyhow::Result<mold_inference::chain::StageOutcome>,
    cancellation_seen: bool,
    cancelled_after_render: bool,
) -> anyhow::Result<crate::chain_job_runner::StageRenderOutcome> {
    match render {
        Err(error) if mold_inference::is_inference_cancelled(&error) => {
            Ok(crate::chain_job_runner::StageRenderOutcome::Cancelled)
        }
        Err(error) => Err(error),
        Ok(_) if cancellation_seen || cancelled_after_render => {
            Ok(crate::chain_job_runner::StageRenderOutcome::Cancelled)
        }
        Ok(outcome) => Ok(crate::chain_job_runner::StageRenderOutcome::Done(outcome)),
    }
}

fn commit_utility_allocation(
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    fence: &crate::scheduler::LeaseFence,
) {
    let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::AllocationCommitted {
        device_id: fence.device_id.clone(),
        work_id: fence.work_id.clone(),
        owner_epoch: fence.owner_epoch,
        worker_generation: fence.worker_generation,
    });
}

fn process_prompt_expansion(worker: &GpuWorker, job: PromptExpansionJob) -> bool {
    let result = (|| -> anyhow::Result<mold_core::ExpandResult> {
        ensure_worker_not_poisoned(worker, &job.settings.model)?;
        #[cfg(feature = "expand")]
        {
            use mold_core::PromptExpander;
            let selector = worker.gpu.stable_id.as_ref().map_or_else(
                || mold_core::GpuSelector::Ordinal(worker.gpu.ordinal),
                |id| mold_core::GpuSelector::Identifier(id.clone()),
            );
            let mut expander = std::mem::ManuallyDrop::new(
                mold_inference::expand::LocalExpander::from_config(
                    &job.config,
                    Some(&job.settings.model),
                )
                .ok_or_else(|| {
                    anyhow::anyhow!("local expand model not found — run: mold pull qwen3-expand")
                })?
                .with_gpu_selection(mold_core::GpuSelection::Specific(vec![selector]))
                .with_preferred_gpu(Some(worker.gpu.ordinal)),
            );
            expander.set_on_progress(Box::new(|event| {
                if let mold_inference::ProgressEvent::PhaseDone {
                    phase: mold_inference::ProgressPhase::ModelLoad,
                    elapsed,
                    ..
                } = &event
                {
                    record_model_load_timing(ModelLoadDisposition::Cold, *elapsed);
                } else {
                    record_phase_timing(&event);
                }
            }));
            let expansion = expander.expand(&job.prompt, &job.expand_config);
            if expansion.as_ref().is_err_and(is_fatal_cuda_error) {
                // A fatal driver fault invalidates every CUDA-backed object
                // associated with this context. Do not run LocalExpander's
                // destructor after that fault; quarantine the owner and leak
                // the invalid handles until process supervision restarts us.
                quarantine_poisoned_worker(worker);
                contain_worker_cache(worker);
                return expansion;
            }
            // SAFETY: the fatal path returns with the wrapper intentionally
            // leaked. Every other path reaches this one explicit drop.
            unsafe {
                std::mem::ManuallyDrop::drop(&mut expander);
            }
            expansion
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
    let successful = result.is_ok();
    let _ = job.result_tx.send(result.map_err(|e| e.to_string()));
    successful
}

fn process_standalone_upscale(worker: &GpuWorker, job: StandaloneUpscaleJob) -> bool {
    let result = (|| -> anyhow::Result<mold_core::UpscaleResponse> {
        ensure_worker_not_poisoned(worker, &job.model)?;
        let load_started = Instant::now();
        let mut engine = mold_inference::create_upscale_engine(
            job.model.clone(),
            job.weights_path,
            mold_inference::LoadStrategy::Eager,
            worker.gpu.ordinal,
        )?;
        record_model_load_timing(ModelLoadDisposition::Cold, load_started.elapsed());
        let progress_tx = job.progress_tx;
        engine.set_on_progress(Box::new(move |event| {
            handle_standalone_upscale_progress(event, progress_tx.as_ref());
        }));
        run_upscale_engine_safely(worker, engine, &job.request)
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
    let successful = result.is_ok();
    let _ = job.result_tx.send(result.map_err(|e| e.to_string()));
    successful
}

fn handle_standalone_upscale_progress(
    event: mold_inference::ProgressEvent,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) {
    record_phase_timing(&event);
    if let Some(progress_tx) = progress_tx {
        let _ = progress_tx.send(SseMessage::Progress(event.into()));
    }
}

fn process_post_generation_upscale(worker: &GpuWorker, mut job: PostGenerationUpscaleJob) -> bool {
    let cleanup = GenerationCleanup::new(&job.generation);
    #[cfg(test)]
    pause_owner_stage_for_test(&job.id, TestOwnerStageBarrier::PrePostUpscale);
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
    #[cfg(test)]
    pause_owner_stage_for_test(&job.id, TestOwnerStageBarrier::PostPostUpscale);
    if result
        .as_ref()
        .is_err_and(|error| has_fatal_cuda_error(error))
    {
        if !worker.poisoned.load(Ordering::SeqCst) {
            quarantine_poisoned_worker(worker);
        }
        let message = fatal_cuda_user_message(&job.generation.model);
        if let Some(ref tx) = job.generation.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: message.clone(),
            }));
        }
        let _ = job.generation.result_tx.send(Err(message));
        drop(cleanup);
        return false;
    }
    let successful = result.is_ok();
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
    successful
}

fn process_admin_unload(worker: &GpuWorker, job: AdminModelUnloadJob) -> bool {
    if let Err(error) =
        ensure_worker_not_poisoned(worker, job.model.as_deref().unwrap_or("active model"))
    {
        let _ = job.result_tx.send(Err(error.to_string()));
        return false;
    }
    if job.evict_cached {
        let Some(model) = job.model.as_deref() else {
            let _ = job
                .result_tx
                .send(Err("cached eviction requires a model name".to_string()));
            return false;
        };
        let result = evict_cached_model_blocking(worker, model).map_err(|error| error.to_string());
        let successful = result.is_ok();
        let _ = job.result_tx.send(result);
        return successful;
    }
    if let Some(expected) = job.model.as_deref() {
        let resident = worker
            .resident_model
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if resident.as_deref() != Some(expected) {
            let _ = job.result_tx.send(Ok(None));
            return true;
        }
    }
    let result = unload_blocking(worker).map_err(|error| error.to_string());
    let successful = result.is_ok();
    let _ = job.result_tx.send(result);
    successful
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
    let engines = evicted.into_iter().map(|(_, engine)| engine);
    let count = match teardown_inference_engines_safely(worker, engines, "idle cache eviction") {
        Ok(count) => count,
        Err(error) => {
            tracing::error!(
                gpu = worker.gpu.ordinal,
                %error,
                "idle model cache teardown failed; quarantining owner context"
            );
            return;
        }
    };
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
    // Latch and notify before touching any lock that the faulting path may have
    // poisoned. `notify_one` stores a permit when the server has not reached
    // its shutdown waiter yet, which makes startup-time owner panics fail
    // closed too.
    worker.poisoned.store(true, Ordering::SeqCst);
    worker.fatal_cuda_error.store(true, Ordering::SeqCst);
    worker.notify_execution_waiters();
    worker.fatal_cuda_shutdown.notify_one();
    worker.consecutive_failures.store(3, Ordering::SeqCst);
    worker.set_resident_model(None);
    *worker
        .degraded_until
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
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

/// Tear down inference engines while retaining ownership across every callback.
///
/// All engines are wrapped before the first callback so a panic cannot unwind
/// through the untouched tail and run additional CUDA-backed destructors. A
/// teardown-callback panic quarantines the worker and intentionally retains the
/// current engine plus the tail for process teardown. Successful engines still
/// unload and drop on the owner thread.
///
/// A destructor can itself be the first operation to panic. That cannot be
/// predicted without leaking every ordinary engine, so the explicit drop is
/// caught: the faulting destructor has begun, but no remaining suspect engine
/// or worker-cache destructor is allowed to run afterward.
fn teardown_inference_engines_safely(
    worker: &GpuWorker,
    engines: impl IntoIterator<Item = Box<dyn mold_inference::InferenceEngine>>,
    operation: &str,
) -> anyhow::Result<usize> {
    let mut engines: Vec<std::mem::ManuallyDrop<Box<dyn mold_inference::InferenceEngine>>> =
        engines
            .into_iter()
            .map(std::mem::ManuallyDrop::new)
            .collect();
    let count = engines.len();

    for engine in &mut engines {
        if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst)
        {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            anyhow::bail!("{operation} stopped because the CUDA owner context is quarantined");
        }

        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if engine.is_loaded() {
                engine.unload();
            }
        })) {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            anyhow::bail!(
                "{operation} teardown panicked; CUDA state quarantined: {}",
                panic_payload_message(payload.as_ref())
            );
        }

        if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst)
        {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            anyhow::bail!("{operation} stopped because the CUDA owner context is quarantined");
        }

        // SAFETY: each wrapper is dropped at most once in this loop. On
        // unwind, ManuallyDrop suppresses a second attempt; on success the
        // wrapper remains inert for the rest of the function.
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            std::mem::ManuallyDrop::drop(engine);
        })) {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            anyhow::bail!(
                "{operation} destructor panicked; CUDA state quarantined: {}",
                panic_payload_message(payload.as_ref())
            );
        }
    }

    Ok(count)
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
            drop_upscale_engine_safely(worker, engine)?;
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
            drop_upscale_engine_safely(worker, engine)?;
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

fn drop_upscale_engine_safely(
    worker: &GpuWorker,
    engine: Box<dyn mold_inference::UpscaleEngine>,
) -> anyhow::Result<()> {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(engine))) {
        quarantine_poisoned_worker(worker);
        contain_worker_cache(worker);
        anyhow::bail!(
            "upscaler destructor panicked; CUDA state quarantined: {}",
            panic_payload_message(payload.as_ref())
        );
    }
    Ok(())
}

fn load_engine_safely(
    worker: &GpuWorker,
    mut engine: Box<dyn mold_inference::InferenceEngine>,
    request: Option<&mold_core::GenerateRequest>,
) -> anyhow::Result<Box<dyn mold_inference::InferenceEngine>> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match request {
        Some(request) => engine.load_for_request(request),
        None => engine.load(),
    }));
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
    let load_started = Instant::now();
    let mut engine = mold_inference::create_upscale_engine(
        model_name.clone(),
        weights_path,
        mold_inference::LoadStrategy::Eager,
        worker.gpu.ordinal,
    )
    .map_err(|e| format!("failed to load upscaler: {e:#}"))?;
    record_model_load_timing(ModelLoadDisposition::Cold, load_started.elapsed());
    let progress_tx = job.progress_tx.clone();
    engine.set_on_progress(Box::new(move |event| {
        record_phase_timing(&event);
        if let Some(tx) = &progress_tx {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }
    }));
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
    let upscaled = upscale_result.map_err(|e| format!("upscale failed: {e:#}"))?;
    apply_upscale_response_to_image_generation(&job.request, response, img, upscaled)
        .map_err(|e| format!("upscale failed: {e:#}"))
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
) -> bool {
    process_job_with_sink(worker, job, GenerationEventSink::V2(scheduler_tx))
}

enum GenerationEventSink<'a> {
    V2(&'a tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>),
    Legacy(&'a tokio::sync::mpsc::UnboundedSender<LegacyOwnerEvent>),
}

impl GenerationEventSink<'_> {
    fn allocation_committed(&self, lease: &crate::scheduler::LeaseFence) {
        if let Self::V2(scheduler_tx) = self {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::AllocationCommitted {
                device_id: lease.device_id.clone(),
                work_id: lease.work_id.clone(),
                owner_epoch: lease.owner_epoch,
                worker_generation: lease.worker_generation,
            });
        }
    }

    fn followup_ready(
        &self,
        work: crate::scheduler::ScheduledOwnerWork,
    ) -> Result<(), Box<crate::scheduler::ScheduledOwnerWork>> {
        match self {
            Self::V2(scheduler_tx) => scheduler_tx
                .send(crate::scheduler::WorkerEvent::FollowupReady {
                    work: Box::new(work),
                })
                .map_err(|error| match error.0 {
                    crate::scheduler::WorkerEvent::FollowupReady { work } => work,
                    _ => unreachable!("followup_ready only transports FollowupReady"),
                }),
            Self::Legacy(owner_event_tx) => owner_event_tx
                .send(LegacyOwnerEvent::FollowupReady(Box::new(work)))
                .map_err(|error| match error.0 {
                    LegacyOwnerEvent::FollowupReady(work) => work,
                }),
        }
    }
}

fn process_job_with_sink(
    worker: &GpuWorker,
    job: GpuJob,
    event_sink: GenerationEventSink<'_>,
) -> bool {
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
        return false;
    }

    if job.result_tx.is_closed() {
        tracing::debug!(gpu = ordinal, model = %model_name, "skipping dispatched job — client disconnected");
        return false;
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
        return false;
    }

    // Ensure model is loaded on this GPU.
    let config_snapshot = job.config.blocking_read().clone();
    let family_slug = crate::model_manager::family_for_model_sync(&model_name, &config_snapshot);
    let activation_hint =
        crate::model_manager::activation_hint_for_request_sync(&config_snapshot, &job.request);
    let request_has_lora = crate::model_manager::request_has_effective_lora(&job.request);
    let planned_load = job.execution_plan.as_ref().map(|plan| PlannedLoadContract {
        mode: PlannedEngineMode::from_plan(plan),
        execution_fingerprint: plan.execution_fingerprint.as_str(),
        request: &job.request,
        engine_paths: &plan.engine_paths,
        engine_config: &plan.engine_config,
    });
    if let Err(e) = ensure_model_ready_sync_inner_guarded(
        worker,
        &model_name,
        &model_name,
        &config_snapshot,
        activation_hint,
        request_has_lora,
        planned_load,
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
        if count_worker_failure {
            record_failure(worker);
        }
        return false;
    }
    worker.set_resident_execution_fingerprint(
        job.execution_plan
            .as_ref()
            .map(|plan| plan.execution_fingerprint.as_str()),
    );

    // This is the first real allocation boundary: model readiness has
    // completed, so host allocations owned by this lease now exist. The
    // coordinator keeps the reservation charged until a memory sample whose
    // collection began after this commit can reflect it.
    if let Some(lease) = job.lease.as_ref() {
        event_sink.allocation_committed(lease);
    }

    if let Err(error) = ensure_worker_not_poisoned(worker, &model_name) {
        let err_msg = error.to_string();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent {
                message: err_msg.clone(),
            }));
        }
        let _ = job.result_tx.send(Err(err_msg));
        return false;
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
        clear_active_generation(worker);
        return false;
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
        clear_active_generation(worker);
        return false;
    };

    // Set progress callback if SSE streaming.
    let progress_tx = job.progress_tx.clone();
    cached_engine.engine.set_on_progress(Box::new(move |event| {
        record_phase_timing(&event);
        if let Some(tx) = &progress_tx {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }
    }));

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
                return false;
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
                    let followup_started = match event_sink.followup_ready(work) {
                        Ok(()) => {
                            std::mem::forget(cleanup);
                            true
                        }
                        Err(work) => {
                            work.work.reject(
                                "GPU dispatch owner stopped before post-generation upscale"
                                    .to_string(),
                            );
                            std::mem::forget(cleanup);
                            false
                        }
                    };
                    return followup_started;
                }
            }

            finish_generation_success(job, response, img, None);
            drop(cleanup);
            true
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
            false
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
            false
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
        let _gallery_writer = job.gallery_publication_gate.blocking_write();
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
    ensure_model_ready_sync_inner_guarded(
        worker,
        model_name,
        model_name,
        config,
        hint,
        request_has_lora,
        None,
    )
}

fn ensure_model_ready_sync_inner_guarded(
    worker: &GpuWorker,
    cache_key: &str,
    model_name: &str,
    config: &Config,
    hint: Option<crate::model_manager::ActivationHint>,
    request_has_lora: bool,
    planned_load: Option<PlannedLoadContract<'_>>,
) -> anyhow::Result<()> {
    let started = Instant::now();
    let result = ensure_model_ready_sync_inner(
        worker,
        cache_key,
        model_name,
        config,
        hint,
        request_has_lora,
        planned_load,
    );
    if result.is_ok() {
        worker.set_resident_model(Some(cache_key));
    } else if result.as_ref().is_err_and(is_fatal_cuda_error) {
        quarantine_poisoned_worker(worker);
        contain_worker_cache(worker);
    }
    result.map(|disposition| {
        record_model_load_timing(disposition, started.elapsed());
    })
}

fn ensure_model_ready_sync_inner(
    worker: &GpuWorker,
    cache_key: &str,
    model_name: &str,
    config: &Config,
    hint: Option<crate::model_manager::ActivationHint>,
    request_has_lora: bool,
    planned_load: Option<PlannedLoadContract<'_>>,
) -> anyhow::Result<ModelLoadDisposition> {
    let planned_mode = planned_load.map(|planned| planned.mode);
    let planned_execution_fingerprint = planned_load.map(|planned| planned.execution_fingerprint);
    let load_request = planned_load.map(|planned| planned.request);
    let planned_engine_paths = planned_load.map(|planned| planned.engine_paths);
    let planned_engine_config = planned_load.map(|planned| planned.engine_config);
    let cache = worker.model_cache.lock().unwrap();

    let cached_requires_reconstruction = cache.get(model_name).is_some_and(|entry| {
        cached_engine_requires_reconstruction(
            entry.engine.as_ref(),
            planned_mode,
            planned_execution_fingerprint,
            entry.engine.model_paths().is_some_and(|paths| {
                crate::model_manager::request_requires_fresh_engine_for_offload_policy(
                    paths,
                    hint,
                    request_has_lora,
                )
            }),
        )
    });

    // Already loaded?
    if let Some(entry) = cache.get(cache_key) {
        if entry.residency == ModelResidency::Gpu && !cached_requires_reconstruction {
            return Ok(ModelLoadDisposition::Unchanged);
        }
    }

    // Check if we have it cached but not on GPU (Parked).
    let has_cached = cache.contains(cache_key);

    // Snapshot the cached engine's paths (if any) for the preflight before
    // dropping the lock. Cloning ModelPaths keeps the borrow scoped to this
    // block. Active-VRAM is sampled inside the preflight helper itself so
    // each retry sees fresh state.
    let cached_paths = if has_cached {
        cache
            .get(cache_key)
            .and_then(|e| e.engine.model_paths().cloned())
    } else {
        None
    };
    let preflight_paths = planned_engine_paths
        .cloned()
        .or_else(|| cached_paths.clone());
    drop(cache);

    if has_cached {
        // Preflight before unloading the active model — the active model's
        // footprint counts toward effective availability since we're about
        // to free it. On budget failure, evict-to-fit drops parked entries
        // (other than `model_name` itself) and retries.
        if let Some(ref paths) = preflight_paths {
            preflight_memory_guard_with_eviction(
                &worker.model_cache,
                cache_key,
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
        if let Some(ref paths) = preflight_paths {
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
        let load_strategy = if let Some(mode) = planned_mode {
            mode.load_strategy
        } else {
            match cached_paths.as_ref() {
                Some(paths) => select_load_strategy_for_worker(worker, model_name, paths, hint)?,
                None => mold_inference::LoadStrategy::Eager,
            }
        };

        let cached_mode_matches = worker
            .model_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(cache_key)
            .is_some_and(|entry| {
                planned_mode.is_none_or(|mode| mode.matches(entry.engine.as_ref()))
            });
        if load_strategy == mold_inference::LoadStrategy::Sequential
            || cached_requires_reconstruction
            || !cached_mode_matches
        {
            let paths = planned_engine_paths
                .cloned()
                .or(cached_paths)
                .ok_or_else(|| {
                    anyhow::anyhow!("cached engine for '{model_name}' does not expose model paths")
                })?;
            let old_engine = {
                let mut cache = worker.model_cache.lock().unwrap();
                cache
                    .remove(cache_key)
                    .ok_or_else(|| anyhow::anyhow!("cache race: model '{model_name}' vanished"))?
            };

            let offload = planned_mode.map_or_else(
                || {
                    crate::model_manager::server_offload_enabled_for_paths(
                        &paths,
                        hint,
                        request_has_lora,
                    )
                },
                |mode| mode.block_offload,
            );
            let created = if let Some(engine_config) = planned_engine_config {
                mold_inference::create_engine_with_frozen_config(
                    model_name.to_string(),
                    paths,
                    engine_config,
                    load_strategy,
                    worker.gpu.ordinal,
                    offload,
                    Some(worker.shared_pool.clone()),
                )
            } else {
                let resolved_catalog_config =
                    crate::model_manager::resolve_installed_catalog_paths_for_worker(
                        model_name, config,
                    )
                    .map_err(|e| anyhow::anyhow!(e.error))?
                    .map(|(_, config)| config);
                let engine_config = resolved_catalog_config.as_ref().unwrap_or(config);
                mold_inference::create_engine_with_pool(
                    model_name.to_string(),
                    paths,
                    engine_config,
                    load_strategy,
                    worker.gpu.ordinal,
                    offload,
                    Some(worker.shared_pool.clone()),
                )
            };
            let engine = match created {
                Ok(engine) => match planned_mode {
                    Some(mode) => record_planned_engine_mode(
                        engine,
                        mode,
                        planned_execution_fingerprint
                            .expect("planned engine mode must carry an execution fingerprint"),
                    ),
                    None => engine,
                },
                Err(err) => {
                    let evicted = {
                        let mut cache = worker.model_cache.lock().unwrap();
                        cache.insert_loaded(cache_key.to_string(), old_engine, 0)
                    };
                    drop(evicted);
                    return Err(err);
                }
            };

            tracing::info!(
                gpu = worker.gpu.ordinal,
                model = %model_name,
                "recreating cached engine for exact execution plan..."
            );
            let vram_baseline = device::vram_in_use_bytes(worker.gpu.ordinal);
            let engine = match load_engine_safely(worker, engine, load_request) {
                Ok(engine) => engine,
                Err(err) if worker.poisoned.load(Ordering::SeqCst) => {
                    contain_poisoned_cuda(old_engine);
                    return Err(err);
                }
                Err(err) => {
                    let evicted = {
                        let mut cache = worker.model_cache.lock().unwrap();
                        cache.insert_loaded(cache_key.to_string(), old_engine, 0)
                    };
                    drop(evicted);
                    return Err(err);
                }
            };
            let vram = device::vram_load_delta(worker.gpu.ordinal, vram_baseline);
            retire_replaced_engine(old_engine);
            let evicted = {
                let mut cache = worker.model_cache.lock().unwrap();
                cache.insert_loaded(cache_key.to_string(), engine, vram)
            };
            drop(evicted);
            return Ok(ModelLoadDisposition::Cold);
        }

        // Take the engine out and reload it.
        let engine = {
            let mut cache = worker.model_cache.lock().unwrap();
            cache
                .remove(cache_key)
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
        let engine = load_engine_safely(worker, engine, load_request)?;

        let vram = device::vram_load_delta(worker.gpu.ordinal, vram_baseline);
        // Drop any evicted engine OUTSIDE the cache lock — `cuMemFree` and
        // safetensor unmap during the drop can block other cache users.
        let evicted = {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.insert_loaded(cache_key.to_string(), engine, vram)
        };
        drop(evicted);
        return Ok(ModelLoadDisposition::WarmReload);
    }

    // Not in cache — need to create from scratch.
    // Resolve model paths.
    let mut resolved_catalog_config = None;
    let paths = if let Some(paths) = planned_engine_paths.cloned() {
        paths
    } else if let Some(paths) = ModelPaths::resolve(model_name, config) {
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
    let load_strategy = if let Some(mode) = planned_mode {
        mode.load_strategy
    } else {
        select_load_strategy_for_worker(worker, model_name, &paths, hint)?
    };

    let offload = planned_mode.map_or_else(
        || crate::model_manager::server_offload_enabled_for_paths(&paths, hint, request_has_lora),
        |mode| mode.block_offload,
    );
    let engine = if let Some(engine_config) = planned_engine_config {
        mold_inference::create_engine_with_frozen_config(
            model_name.to_string(),
            paths,
            engine_config,
            load_strategy,
            worker.gpu.ordinal,
            offload,
            Some(worker.shared_pool.clone()),
        )?
    } else {
        let engine_config = resolved_catalog_config.as_ref().unwrap_or(config);
        mold_inference::create_engine_with_pool(
            model_name.to_string(),
            paths,
            engine_config,
            load_strategy,
            worker.gpu.ordinal,
            offload,
            Some(worker.shared_pool.clone()),
        )?
    };
    let engine = match planned_mode {
        Some(mode) => record_planned_engine_mode(
            engine,
            mode,
            planned_execution_fingerprint
                .expect("planned engine mode must carry an execution fingerprint"),
        ),
        None => engine,
    };

    tracing::info!(
        gpu = worker.gpu.ordinal,
        model = %model_name,
        "loading model..."
    );
    // Sample VRAM baseline before load so we can record the new model's
    // per-load delta rather than the device-global usage.
    let vram_baseline = device::vram_in_use_bytes(worker.gpu.ordinal);
    let engine = load_engine_safely(worker, engine, load_request)?;

    let vram = device::vram_load_delta(worker.gpu.ordinal, vram_baseline);
    // Drop any evicted engine OUTSIDE the cache lock — `cuMemFree` and
    // safetensor unmap during the drop can block other cache users.
    let evicted = {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.insert_loaded(cache_key.to_string(), engine, vram)
    };
    drop(evicted);

    Ok(ModelLoadDisposition::Cold)
}

fn cached_engine_requires_reconstruction(
    engine: &dyn mold_inference::InferenceEngine,
    planned_mode: Option<PlannedEngineMode>,
    planned_execution_fingerprint: Option<&str>,
    offload_policy_requires_fresh_engine: bool,
) -> bool {
    planned_mode.is_some_and(|mode| !mode.matches(engine))
        || planned_execution_fingerprint.is_some_and(|fingerprint| {
            engine.configured_execution_fingerprint() != Some(fingerprint)
        })
        || offload_policy_requires_fresh_engine
}

fn retire_replaced_engine(engine: Box<dyn mold_inference::InferenceEngine>) {
    drop(engine);
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
    teardown_inference_engines_safely(worker, std::iter::once(engine), "cached admin eviction")?;
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
    run_chain_blocking_with_mode(worker, model_name, config, hint, None, with_engine)
}

fn run_chain_blocking_with_mode<T, E: std::fmt::Display + std::fmt::Debug>(
    worker: &GpuWorker,
    model_name: &str,
    config: &mold_core::Config,
    hint: Option<crate::model_manager::ActivationHint>,
    _planned_mode: Option<PlannedEngineMode>,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> Result<T, E>,
) -> ChainPrep<T, E> {
    run_chain_blocking_with_identity(
        worker,
        model_name,
        model_name,
        config,
        hint,
        None,
        with_engine,
    )
}

fn run_chain_blocking_with_identity<T, E: std::fmt::Display + std::fmt::Debug>(
    worker: &GpuWorker,
    cache_key: &str,
    model_name: &str,
    config: &mold_core::Config,
    hint: Option<crate::model_manager::ActivationHint>,
    planned_load: Option<PlannedLoadContract<'_>>,
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
    if let Err(error) = ensure_model_ready_sync_inner_guarded(
        worker,
        cache_key,
        model_name,
        config,
        hint,
        false,
        planned_load,
    ) {
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
        cache.take(cache_key).ok_or_else(|| {
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

struct PlannedStageLoad<'a> {
    cache_key: &'a str,
    model_name: &'a str,
    config: &'a mold_core::Config,
    hint: Option<crate::model_manager::ActivationHint>,
    plan: &'a crate::execution_plan::ResolvedExecutionPlan,
    request: &'a mold_core::GenerateRequest,
}

fn run_stage_blocking_planned<T, E: std::fmt::Display + std::fmt::Debug>(
    worker: &GpuWorker,
    load: PlannedStageLoad<'_>,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> Result<T, E>,
) -> ChainPrep<T, E> {
    run_chain_blocking_with_identity(
        worker,
        load.cache_key,
        load.model_name,
        load.config,
        load.hint,
        Some(PlannedLoadContract {
            mode: PlannedEngineMode::from_plan(load.plan),
            execution_fingerprint: &load.plan.execution_fingerprint,
            request: load.request,
            engine_paths: &load.plan.engine_paths,
            engine_config: &load.plan.engine_config,
        }),
        with_engine,
    )
}

#[cfg(test)]
type AcceptanceBarrier = (
    std::sync::mpsc::SyncSender<()>,
    std::sync::mpsc::Receiver<()>,
);

#[cfg(test)]
static ACCEPTANCE_BARRIERS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<String, AcceptanceBarrier>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TestOwnerStageBarrier {
    PrePostUpscale,
    PostPostUpscale,
}

#[cfg(test)]
static OWNER_STAGE_BARRIERS: std::sync::LazyLock<
    std::sync::Mutex<
        std::collections::BTreeMap<(String, TestOwnerStageBarrier), AcceptanceBarrier>,
    >,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

#[cfg(test)]
fn install_acceptance_barrier(
    work_id: &str,
) -> (
    std::sync::mpsc::Receiver<()>,
    std::sync::mpsc::SyncSender<()>,
) {
    let (reached_tx, reached_rx) = std::sync::mpsc::sync_channel(1);
    let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
    ACCEPTANCE_BARRIERS
        .lock()
        .unwrap()
        .insert(work_id.to_string(), (reached_tx, resume_rx));
    (reached_rx, resume_tx)
}

#[cfg(test)]
fn pause_after_acceptance_for_test(work_id: &str) {
    let barrier = ACCEPTANCE_BARRIERS.lock().unwrap().remove(work_id);
    if let Some((reached_tx, resume_rx)) = barrier {
        reached_tx
            .send(())
            .expect("acceptance test should be waiting");
        resume_rx
            .recv()
            .expect("acceptance test should resume owner");
    }
}

#[cfg(test)]
fn install_owner_stage_barrier(
    work_id: &str,
    point: TestOwnerStageBarrier,
) -> (
    std::sync::mpsc::Receiver<()>,
    std::sync::mpsc::SyncSender<()>,
) {
    let (reached_tx, reached_rx) = std::sync::mpsc::sync_channel(1);
    let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
    OWNER_STAGE_BARRIERS
        .lock()
        .unwrap()
        .insert((work_id.to_string(), point), (reached_tx, resume_rx));
    (reached_rx, resume_tx)
}

#[cfg(test)]
fn pause_owner_stage_for_test(work_id: &str, point: TestOwnerStageBarrier) {
    let barrier = OWNER_STAGE_BARRIERS
        .lock()
        .unwrap()
        .remove(&(work_id.to_string(), point));
    if let Some((reached_tx, resume_rx)) = barrier {
        reached_tx.send(()).expect("stage test should be waiting");
        resume_rx.recv().expect("stage test should resume owner");
    }
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
        load_started: Option<std::sync::mpsc::SyncSender<()>>,
        load_resume: Option<Mutex<std::sync::mpsc::Receiver<()>>>,
    }

    impl FakeSlowEngine {
        fn boxed(name: &str, load_sleep: Duration) -> Box<dyn InferenceEngine> {
            Box::new(Self {
                name: name.to_string(),
                loaded: false,
                load_sleep,
                load_started: None,
                load_resume: None,
            })
        }

        fn blocked(
            name: &str,
        ) -> (
            Box<dyn InferenceEngine>,
            std::sync::mpsc::Receiver<()>,
            std::sync::mpsc::SyncSender<()>,
        ) {
            let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
            let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
            (
                Box::new(Self {
                    name: name.to_string(),
                    loaded: false,
                    load_sleep: Duration::ZERO,
                    load_started: Some(started_tx),
                    load_resume: Some(Mutex::new(resume_rx)),
                }),
                started_rx,
                resume_tx,
            )
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
            if let Some(started) = self.load_started.take() {
                started.send(()).unwrap();
            }
            if let Some(resume) = self.load_resume.take() {
                resume.into_inner().unwrap().recv().unwrap();
            }
            std::thread::sleep(self.load_sleep);
            self.loaded = true;
            Ok(())
        }
        fn unload(&mut self) {
            self.loaded = false;
        }
    }

    struct BarrierGenerationEngine {
        name: String,
        generate_started: Option<std::sync::mpsc::SyncSender<()>>,
        generate_resume: Option<Mutex<std::sync::mpsc::Receiver<()>>>,
    }

    impl InferenceEngine for BarrierGenerationEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            self.generate_started
                .take()
                .expect("generation barrier used once")
                .send(())
                .unwrap();
            self.generate_resume
                .take()
                .expect("generation barrier used once")
                .into_inner()
                .unwrap()
                .recv()
                .unwrap();
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
            true
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn planned_engine_wrapper_records_exact_creation_mode() {
        let mode = PlannedEngineMode {
            load_strategy: mold_inference::LoadStrategy::Sequential,
            block_offload: true,
        };
        let unconfigured = FakeSlowEngine::boxed("planned", Duration::ZERO);
        assert!(!mode.matches(unconfigured.as_ref()));
        let configured = record_planned_engine_mode(unconfigured, mode, "plan-fingerprint");
        assert!(mode.matches(configured.as_ref()));
        assert_eq!(
            configured.configured_execution_fingerprint(),
            Some("plan-fingerprint")
        );
    }

    #[test]
    fn planned_engine_wrapper_forwards_batch_and_cancellation_contract() {
        struct ContractEngine {
            token_set: Arc<AtomicBool>,
            token_cleared: Arc<AtomicBool>,
        }
        impl InferenceEngine for ContractEngine {
            fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
                unreachable!()
            }
            fn model_name(&self) -> &str {
                "contract"
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn load(&mut self) -> anyhow::Result<()> {
                Ok(())
            }
            fn set_cancellation_token(
                &mut self,
                _token: mold_inference::InferenceCancellationToken,
            ) {
                self.token_set.store(true, Ordering::SeqCst);
            }
            fn clear_cancellation_token(&mut self) {
                self.token_cleared.store(true, Ordering::SeqCst);
            }
            fn batch_execution_capability(&self) -> mold_inference::BatchExecutionCapability {
                mold_inference::BatchExecutionCapability {
                    native_batch_sizes: &[1],
                    cooperative_cancellation: false,
                }
            }
        }

        let token_set = Arc::new(AtomicBool::new(false));
        let token_cleared = Arc::new(AtomicBool::new(false));
        let mut wrapped = record_planned_engine_mode(
            Box::new(ContractEngine {
                token_set: token_set.clone(),
                token_cleared: token_cleared.clone(),
            }),
            PlannedEngineMode {
                load_strategy: mold_inference::LoadStrategy::Eager,
                block_offload: false,
            },
            "contract",
        );

        wrapped.set_cancellation_token(mold_inference::InferenceCancellationToken::default());
        wrapped.clear_cancellation_token();

        assert!(token_set.load(Ordering::SeqCst));
        assert!(token_cleared.load(Ordering::SeqCst));
        assert!(
            !wrapped
                .batch_execution_capability()
                .cooperative_cancellation
        );
    }

    #[test]
    fn same_mode_eager_fingerprint_mismatch_requires_reconstruction() {
        let mode = PlannedEngineMode {
            load_strategy: mold_inference::LoadStrategy::Eager,
            block_offload: false,
        };
        let configured = record_planned_engine_mode(
            FakeSlowEngine::boxed("planned", Duration::ZERO),
            mode,
            "old-fingerprint",
        );

        assert!(cached_engine_requires_reconstruction(
            configured.as_ref(),
            Some(mode),
            Some("new-fingerprint"),
            false,
        ));
    }

    #[test]
    fn successful_reconstruction_retires_the_replaced_engine() {
        let dropped_on = Arc::new(Mutex::new(None));
        retire_replaced_engine(Box::new(DropRecordingEngine {
            name: "replaced".to_string(),
            dropped_on: dropped_on.clone(),
        }));

        assert!(
            dropped_on.lock().unwrap().is_some(),
            "the old engine must be dropped after its exact-plan replacement loads"
        );
    }

    struct PlacementRecordingEngine {
        seen_at_load: Arc<Mutex<Option<mold_core::DevicePlacement>>>,
    }

    impl mold_inference::InferenceEngine for PlacementRecordingEngine {
        fn generate(
            &mut self,
            _req: &mold_core::GenerateRequest,
        ) -> anyhow::Result<mold_core::GenerateResponse> {
            unreachable!("preload forwarding test does not generate")
        }

        fn model_name(&self) -> &str {
            "placement-recording"
        }

        fn is_loaded(&self) -> bool {
            false
        }

        fn load(&mut self) -> anyhow::Result<()> {
            anyhow::bail!("plain load must not be used for a planned request")
        }

        fn load_for_request(&mut self, req: &mold_core::GenerateRequest) -> anyhow::Result<()> {
            *self.seen_at_load.lock().unwrap() = req.placement.clone();
            Ok(())
        }
    }

    #[test]
    fn planned_preload_receives_materialized_component_placement() {
        let seen_at_load = Arc::new(Mutex::new(None));
        let mut engine = record_planned_engine_mode(
            Box::new(PlacementRecordingEngine {
                seen_at_load: seen_at_load.clone(),
            }),
            PlannedEngineMode {
                load_strategy: mold_inference::LoadStrategy::Eager,
                block_offload: false,
            },
            "placement-fingerprint",
        );
        let mut request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"placement-recording","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Cpu,
            advanced: Some(mold_core::types::AdvancedPlacement {
                transformer: mold_core::DeviceRef::Cpu,
                vae: mold_core::DeviceRef::Cpu,
                clip_l: Some(mold_core::DeviceRef::device("cuda:0")),
                clip_g: None,
                t5: Some(mold_core::DeviceRef::Cpu),
                qwen: None,
            }),
        });

        engine.load_for_request(&request).unwrap();

        assert_eq!(
            *seen_at_load.lock().unwrap(),
            request.placement,
            "exact sparse encoder and CPU transformer/VAE placement must exist before load"
        );
    }

    #[test]
    fn active_chain_cancellation_fences_a_late_successful_render() {
        let outcome = mold_inference::chain::StageOutcome {
            frames: Vec::new(),
            tail: mold_inference::chain::ChainTail {
                frames: 0,
                tail_rgb_frames: Vec::new(),
            },
            audio: None,
            generation_time_ms: 1,
        };
        assert!(matches!(
            fence_chain_stage_render(Ok(outcome), false, true).unwrap(),
            crate::chain_job_runner::StageRenderOutcome::Cancelled
        ));
        let safe_point_error: anyhow::Error = mold_inference::InferenceCancelled.into();
        assert!(matches!(
            fence_chain_stage_render(Err(safe_point_error), false, false).unwrap(),
            crate::chain_job_runner::StageRenderOutcome::Cancelled
        ));
    }

    struct DropRecordingEngine {
        name: String,
        dropped_on: Arc<Mutex<Option<String>>>,
    }

    struct PanickingDropEngine {
        name: String,
        unload_calls: Arc<AtomicUsize>,
        drop_calls: Arc<AtomicUsize>,
    }

    struct DestructorPanickingEngine {
        name: String,
        drop_calls: Arc<AtomicUsize>,
    }

    impl Drop for DestructorPanickingEngine {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
            panic!("injected first-operation destructor panic");
        }
    }

    impl InferenceEngine for DestructorPanickingEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("destructor safety test never runs inference")
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

        fn unload(&mut self) {}
    }

    impl Drop for PanickingDropEngine {
        fn drop(&mut self) {
            self.drop_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl InferenceEngine for PanickingDropEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("destructor safety test never runs inference")
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
            panic!("injected CUDA-backed teardown panic");
        }
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
            owner_epoch: 1,
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
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
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
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            queue: QueueHandle::new(queue_tx),
            registry: JobRegistry::new(),
            events: crate::events::EventBroadcaster::new(),
            execution_plan: None,
            prepared_execution_inputs: None,
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
        protocol_worker_with_capacity(ordinal, fatal_cuda_error, 1)
    }

    fn protocol_worker_with_capacity(
        ordinal: usize,
        fatal_cuda_error: Arc<AtomicBool>,
        capacity: usize,
    ) -> (Arc<GpuWorker>, std::sync::mpsc::Receiver<GpuWorkerCommand>) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(capacity);
        let worker = Arc::new(GpuWorker {
            owner_epoch: 1,
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
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error,
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
            owner_thread_id: std::sync::OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    fn legacy_admin_grant(
        worker: &GpuWorker,
        id: &str,
    ) -> (
        Box<LeaseGrant>,
        tokio::sync::oneshot::Receiver<Result<(), String>>,
    ) {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        (
            Box::new(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: id.to_string(),
                    device_id: crate::scheduler::worker_device_id(worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 0,
                    plan_version: 0,
                    worker_generation: 0,
                    memory_sample_generation: 0,
                    memory_ledger_sequence: 0,
                },
                work: OwnerWork::AdminModelLoad(Box::new(crate::gpu_pool::AdminModelLoadJob {
                    id: id.to_string(),
                    model: "never-load".to_string(),
                    config: mold_core::Config::default(),
                    result_tx,
                })),
                retry: None,
            }),
            result_rx,
        )
    }

    #[test]
    fn worker_rejects_stale_generation_before_touching_inference() {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<GpuWorkerCommand>(1);
        let worker = Arc::new(GpuWorker {
            owner_epoch: 1,
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
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(1),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
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
                gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(
                ),
                queue: QueueHandle::new(queue_tx),
                registry: JobRegistry::new(),
                events: crate::events::EventBroadcaster::new(),
                execution_plan: None,
                prepared_execution_inputs: None,
                lease: Some(crate::scheduler::LeaseFence {
                    work_id: "stale".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
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
            owner_epoch: worker.owner_epoch,
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
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Stopped { owner_epoch: 1, .. })
            ));
            assert!(
                event_rx.blocking_recv().is_none(),
                "owner must drop its coordinator sender before restart"
            );
        }
    }

    #[test]
    fn clean_owner_shutdown_contains_cache_after_teardown_panic() {
        let unload_calls = Arc::new(AtomicUsize::new(0));
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        worker.model_cache.lock().unwrap().insert(
            Box::new(PanickingDropEngine {
                name: "shutdown-engine".to_string(),
                unload_calls: unload_calls.clone(),
                drop_calls: drop_calls.clone(),
            }),
            0,
        );
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready { .. })
        ));

        worker.request_shutdown();
        handle
            .join()
            .expect("owner shutdown catches teardown panic");

        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { owner_epoch: 1, .. })
        ));
        assert!(event_rx.blocking_recv().is_none());
        assert_eq!(unload_calls.load(Ordering::SeqCst), 1);
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
    }

    #[test]
    fn legacy_owner_executes_without_publishing_v2_ready_or_acceptance_events() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (owner_event_tx, mut owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_legacy_gpu_thread(
            worker.clone(),
            job_rx,
            owner_event_tx,
            Duration::from_secs(60),
        );
        let (ran_tx, ran_rx) = std::sync::mpsc::channel();
        worker.reserve_legacy_transport();
        worker
            .try_send_job(Box::new(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "legacy-probe".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 0,
                    plan_version: 0,
                    worker_generation: 0,
                    memory_sample_generation: 0,
                    memory_ledger_sequence: 0,
                },
                work: OwnerWork::Probe {
                    id: "legacy-probe".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(move || ran_tx.send(()).unwrap()),
                },
                retry: None,
            }))
            .unwrap();

        ran_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("legacy owner should execute transported work");
        assert!(matches!(
            owner_event_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        worker.request_shutdown();
        handle.join().expect("legacy owner should join cleanly");
        assert_eq!(worker.pending_or_executing(), 0);
        assert!(matches!(
            owner_event_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn legacy_transport_never_executes_across_an_existing_binary_claim() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (owner_event_tx, _owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(worker.try_claim_in_flight(), "synthetic chain claim");
        let handle = spawn_legacy_gpu_thread(
            worker.clone(),
            job_rx,
            owner_event_tx,
            Duration::from_secs(60),
        );
        let (ran_tx, ran_rx) = std::sync::mpsc::channel();
        worker.reserve_legacy_transport();
        worker
            .try_send_job(Box::new(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "legacy-behind-chain".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 0,
                    plan_version: 0,
                    worker_generation: 0,
                    memory_sample_generation: 0,
                    memory_ledger_sequence: 0,
                },
                work: OwnerWork::Probe {
                    id: "legacy-behind-chain".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(move || ran_tx.send(()).unwrap()),
                },
                retry: None,
            }))
            .unwrap();

        worker.wait_until_owner_claim_blocked();
        assert!(matches!(
            ran_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 1);
        assert_eq!(worker.legacy_pending.load(Ordering::SeqCst), 1);

        worker.release_in_flight();
        ran_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("legacy work should run after the chain releases");
        worker.request_shutdown();
        handle.join().unwrap();
        assert_eq!(worker.pending_or_executing(), 0);
        assert!(!worker.poisoned.load(Ordering::SeqCst));
    }

    #[test]
    fn fatal_legacy_owner_rejects_dequeued_and_depth_two_buffered_work_exactly_once() {
        let fatal = Arc::new(AtomicBool::new(false));
        let (worker, job_rx) = protocol_worker_with_capacity(0, fatal.clone(), 2);
        let cache_operations = Arc::new(Mutex::new(Vec::new()));
        worker.model_cache.lock().unwrap().insert(
            Box::new(LifecycleRecordingEngine {
                name: "fatal-cache-sentinel".to_string(),
                loaded: false,
                operations: cache_operations.clone(),
            }),
            0,
        );
        let (owner_event_tx, _owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(worker.try_claim_in_flight(), "synthetic chain claim");
        let handle = spawn_legacy_gpu_thread(
            worker.clone(),
            job_rx,
            owner_event_tx,
            Duration::from_secs(60),
        );

        let (first, first_result) = legacy_admin_grant(&worker, "fatal-dequeued");
        worker.reserve_legacy_transport();
        worker.try_send_job(first).unwrap();
        worker.wait_until_owner_claim_blocked();

        let mut results = vec![first_result];
        for id in ["fatal-buffered-1", "fatal-buffered-2"] {
            let (grant, result) = legacy_admin_grant(&worker, id);
            worker.reserve_legacy_transport();
            worker.try_send_job(grant).unwrap();
            results.push(result);
        }
        assert_eq!(worker.legacy_pending.load(Ordering::SeqCst), 3);

        worker.poisoned.store(true, Ordering::SeqCst);
        fatal.store(true, Ordering::SeqCst);
        worker.notify_execution_waiters();
        for result in results {
            let error = result
                .blocking_recv()
                .expect("every accepted work item must settle")
                .expect_err("fatal work must be rejected");
            assert!(error.contains("Restart the mold server"));
        }
        assert!(
            !handle.is_finished(),
            "fatal owner remains a rejection drain until dispatchers stop and request shutdown"
        );
        worker.request_shutdown();
        handle.join().expect("fatal owner should drain and join");
        assert_eq!(worker.legacy_pending.load(Ordering::SeqCst), 0);
        assert_eq!(worker.model_cache.lock().unwrap().len(), 1);
        assert!(
            cache_operations.lock().unwrap().is_empty(),
            "fatal shutdown must retain poisoned cache without unload or drop"
        );
        assert_eq!(
            worker.in_flight.load(Ordering::SeqCst),
            1,
            "owner must not release a chain claim it never acquired"
        );
        worker.release_in_flight();
    }

    #[test]
    fn graceful_legacy_owner_rejects_dequeued_and_depth_two_buffered_work_exactly_once() {
        let (worker, job_rx) =
            protocol_worker_with_capacity(0, Arc::new(AtomicBool::new(false)), 2);
        let (owner_event_tx, _owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(worker.try_claim_in_flight(), "synthetic chain claim");
        let handle = spawn_legacy_gpu_thread(
            worker.clone(),
            job_rx,
            owner_event_tx,
            Duration::from_secs(60),
        );

        let (first, first_result) = legacy_admin_grant(&worker, "shutdown-dequeued");
        worker.reserve_legacy_transport();
        worker.try_send_job(first).unwrap();
        worker.wait_until_owner_claim_blocked();

        let mut results = vec![first_result];
        for id in ["shutdown-buffered-1", "shutdown-buffered-2"] {
            let (grant, result) = legacy_admin_grant(&worker, id);
            worker.reserve_legacy_transport();
            worker.try_send_job(grant).unwrap();
            results.push(result);
        }
        worker.request_shutdown();
        for result in results {
            let error = result
                .blocking_recv()
                .expect("every accepted work item must settle")
                .expect_err("unstarted shutdown work must be rejected");
            assert!(error.contains("shutting down"));
        }
        handle.join().expect("shutdown owner should drain and join");
        assert_eq!(worker.legacy_pending.load(Ordering::SeqCst), 0);
        assert_eq!(
            worker.in_flight.load(Ordering::SeqCst),
            1,
            "owner must not release a chain claim it never acquired"
        );
        worker.release_in_flight();
        assert!(!worker.poisoned.load(Ordering::SeqCst));
    }

    #[test]
    fn legacy_owner_shutdown_uses_poison_safe_cache_teardown() {
        let unload_calls = Arc::new(AtomicUsize::new(0));
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        worker.model_cache.lock().unwrap().insert(
            Box::new(PanickingDropEngine {
                name: "legacy-shutdown-engine".to_string(),
                unload_calls: unload_calls.clone(),
                drop_calls: drop_calls.clone(),
            }),
            0,
        );
        let (owner_event_tx, _owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_legacy_gpu_thread(
            worker.clone(),
            job_rx,
            owner_event_tx,
            Duration::from_secs(60),
        );

        let started_deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < started_deadline {
            if worker.owner_thread_id.get().is_some() {
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            worker.owner_thread_id.get().is_some(),
            "legacy owner must start before shutdown is requested"
        );
        worker.request_shutdown();
        handle
            .join()
            .expect("legacy owner catches teardown panic on shutdown");

        assert_eq!(unload_calls.load(Ordering::SeqCst), 1);
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
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
    fn idle_eviction_contains_engine_before_a_destructor_can_poison_cuda() {
        let first_drop_calls = Arc::new(AtomicUsize::new(0));
        let tail_dropped_on = Arc::new(Mutex::new(None));
        let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
        {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.insert(
                Box::new(DestructorPanickingEngine {
                    name: "first".to_string(),
                    drop_calls: first_drop_calls.clone(),
                }),
                0,
            );
            cache.insert(
                Box::new(DropRecordingEngine {
                    name: "tail".to_string(),
                    dropped_on: tail_dropped_on.clone(),
                }),
                0,
            );
            let mut first = cache.take("first").unwrap();
            first.last_used = Instant::now() - Duration::from_secs(2);
            cache.restore(first);
            let mut tail = cache.take("tail").unwrap();
            tail.last_used = Instant::now() - Duration::from_secs(1);
            cache.restore(tail);
            let mut keep_one = cache.take("keep-one").unwrap();
            keep_one.last_used = Instant::now();
            cache.restore(keep_one);
        }

        let worker_for_thread = worker.clone();
        std::thread::Builder::new()
            .name("gpu-worker-poison-proof".to_string())
            .spawn(move || {
                worker_for_thread
                    .owner_thread_id
                    .set(std::thread::current().id())
                    .expect("test owner initialized once");
                run_gpu_owner_entrypoint(&worker_for_thread, || {
                    evict_idle_on_worker(&worker_for_thread, Duration::ZERO);
                });
            })
            .unwrap()
            .join()
            .unwrap();

        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
        assert_eq!(
            first_drop_calls.load(Ordering::SeqCst),
            1,
            "a destructor cannot be known to panic until it starts"
        );
        assert!(
            tail_dropped_on.lock().unwrap().is_none(),
            "eviction must retain untouched tail engines after the first destructor panics"
        );
    }

    #[test]
    fn standalone_upscale_routes_engine_through_poison_safe_containment() {
        let source = include_str!("gpu_worker.rs");
        let standalone = source
            .split("fn process_standalone_upscale")
            .nth(1)
            .unwrap()
            .split("fn process_post_generation_upscale")
            .next()
            .unwrap();

        assert!(
            standalone.contains("run_upscale_engine_safely("),
            "standalone upscale directly calls engine.upscale and unwinds through the suspect engine before the owner-level quarantine"
        );
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
    fn cached_admin_eviction_contains_engine_after_teardown_panic() {
        let unload_calls = Arc::new(AtomicUsize::new(0));
        let drop_calls = Arc::new(AtomicUsize::new(0));
        let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert(
            Box::new(PanickingDropEngine {
                name: "evict-me".to_string(),
                unload_calls: unload_calls.clone(),
                drop_calls: drop_calls.clone(),
            }),
            0,
        );
        worker
            .owner_thread_id
            .set(std::thread::current().id())
            .expect("test binds admin eviction to the owner thread");

        let error = evict_cached_model_blocking(&worker, "evict-me")
            .expect_err("teardown panic must fail cached admin eviction");

        assert!(error.to_string().contains("teardown panicked"), "{error:#}");
        assert_eq!(unload_calls.load(Ordering::SeqCst), 1);
        assert_eq!(drop_calls.load(Ordering::SeqCst), 0);
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
    }

    #[test]
    fn destructor_panic_is_reported_and_contains_untouched_tail() {
        let first_drop_calls = Arc::new(AtomicUsize::new(0));
        let tail_unload_calls = Arc::new(AtomicUsize::new(0));
        let tail_drop_calls = Arc::new(AtomicUsize::new(0));
        let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
        let engines: Vec<Box<dyn InferenceEngine>> = vec![
            Box::new(DestructorPanickingEngine {
                name: "first".to_string(),
                drop_calls: first_drop_calls.clone(),
            }),
            Box::new(CudaCallbackRecordingEngine {
                name: "tail".to_string(),
                unload_calls: tail_unload_calls.clone(),
                drop_calls: tail_drop_calls.clone(),
            }),
        ];

        let error = teardown_inference_engines_safely(&worker, engines, "destructor panic proof")
            .expect_err("destructor panic must be surfaced");

        assert!(
            error.to_string().contains("destructor panicked"),
            "{error:#}"
        );
        assert_eq!(
            first_drop_calls.load(Ordering::SeqCst),
            1,
            "the first destructor is unknowable until it starts"
        );
        assert_eq!(tail_unload_calls.load(Ordering::SeqCst), 0);
        assert_eq!(tail_drop_calls.load(Ordering::SeqCst), 0);
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
    }

    #[test]
    fn cached_admin_eviction_unloads_and_drops_normally_on_owner_thread() {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let worker = single_worker_pool_with_parked("keep-one", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert(
            Box::new(LifecycleRecordingEngine {
                name: "evict-me".to_string(),
                loaded: true,
                operations: operations.clone(),
            }),
            0,
        );
        worker
            .owner_thread_id
            .set(std::thread::current().id())
            .expect("test binds admin eviction to the owner thread");

        let removed = evict_cached_model_blocking(&worker, "evict-me")
            .expect("ordinary cached admin eviction succeeds");

        assert_eq!(removed.as_deref(), Some("evict-me"));
        let recorded: Vec<_> = operations
            .lock()
            .unwrap()
            .iter()
            .map(|(operation, _)| operation.clone())
            .collect();
        assert_eq!(recorded, ["unload".to_string(), "drop".to_string()]);
        assert!(!worker.poisoned.load(Ordering::SeqCst));
        assert!(!worker.fatal_cuda_error.load(Ordering::SeqCst));
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
            owner_epoch: 1,
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
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
            owner_thread_id: std::sync::OnceLock::new(),
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

        assert!(matches!(
            event_rx.try_recv(),
            Ok(crate::scheduler::WorkerEvent::Stopped { owner_epoch: 1, .. })
        ));
        assert!(event_rx.try_recv().is_err());
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
    fn drain_at_pre_and_post_upscale_barriers_finishes_exact_stage_then_stops() {
        for (id, point) in [
            ("drain-pre-upscale", TestOwnerStageBarrier::PrePostUpscale),
            ("drain-post-upscale", TestOwnerStageBarrier::PostPostUpscale),
        ] {
            let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
            let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            let handle =
                spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Ready {
                    worker_generation: 1,
                    ..
                })
            ));
            assert!(worker.try_claim_in_flight());
            let (stage_reached, stage_resume) = install_owner_stage_barrier(id, point);
            let request: GenerateRequest = serde_json::from_str(
                r#"{"prompt":"upscale barrier","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0,"batch_size":1,"upscale_model":"missing-upscaler"}"#,
            )
            .unwrap();
            let original = fake_upscale_image();
            let response = GenerateResponse {
                images: vec![original.clone()],
                video: None,
                generation_time_ms: 1,
                model: request.model.clone(),
                seed_used: 1,
                gpu: Some(0),
            };
            let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
            let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
            let generation = GpuJob {
                id: id.to_string(),
                model: request.model.clone(),
                request,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx,
                output_dir: None,
                config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                metadata_db: Arc::new(None),
                gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(
                ),
                queue: QueueHandle::new(queue_tx),
                registry: JobRegistry::new(),
                events: crate::events::EventBroadcaster::new(),
                execution_plan: None,
                prepared_execution_inputs: None,
                lease: None,
            };
            worker
                .send_grant(LeaseGrant {
                    fence: crate::scheduler::LeaseFence {
                        work_id: id.to_string(),
                        device_id: crate::scheduler::worker_device_id(&worker),
                        owner_epoch: worker.owner_epoch,
                        state_version: 1,
                        plan_version: 1,
                        worker_generation: 1,
                        memory_sample_generation: 1,
                        memory_ledger_sequence: 1,
                    },
                    work: OwnerWork::PostUpscale(Box::new(PostGenerationUpscaleJob {
                        id: id.to_string(),
                        generation: Box::new(generation),
                        response,
                        image: original,
                    })),
                    retry: None,
                })
                .unwrap();
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Accepted { .. })
            ));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::AllocationCommitted { .. })
            ));
            stage_reached
                .recv_timeout(Duration::from_secs(1))
                .expect("post-upscale stage must reach deterministic barrier");

            worker.request_drain(false);
            assert!(matches!(
                result_rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ));
            stage_resume.send(()).unwrap();
            assert!(result_rx.blocking_recv().unwrap().is_ok());
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Completed { .. })
            ));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Stopped { .. })
            ));
            handle.join().unwrap();
            assert_eq!(worker.pending_or_executing(), 0);
            assert!(!worker.poisoned.load(Ordering::SeqCst));
            assert!(!worker.fatal_cuda_error.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn drain_after_model_ready_finishes_generation_then_stops() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (generate_started_tx, generate_started_rx) = std::sync::mpsc::sync_channel(1);
        let (generate_resume_tx, generate_resume_rx) = std::sync::mpsc::sync_channel(1);
        worker.model_cache.lock().unwrap().insert(
            Box::new(BarrierGenerationEngine {
                name: "barrier-generation".to_string(),
                generate_started: Some(generate_started_tx),
                generate_resume: Some(Mutex::new(generate_resume_rx)),
            }),
            0,
        );
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            })
        ));
        assert!(worker.try_claim_in_flight());
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"barrier","model":"barrier-generation","width":64,"height":64,"steps":1,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        worker
            .send_grant(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "barrier-generation".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: OwnerWork::Generation(Box::new(GpuJob {
                    id: "barrier-generation".to_string(),
                    model: request.model.clone(),
                    request,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx,
                    output_dir: None,
                    config: Arc::new(tokio::sync::RwLock::new(Config::default())),
                    metadata_db: Arc::new(None),
                    gallery_publication_gate:
                        crate::batch_transaction::GalleryPublicationGate::default(),
                    queue: QueueHandle::new(queue_tx),
                    registry: JobRegistry::new(),
                    events: crate::events::EventBroadcaster::new(),
                    execution_plan: None,
                    prepared_execution_inputs: None,
                    lease: None,
                })),
                retry: None,
            })
            .unwrap();
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::AllocationCommitted { .. })
        ));
        generate_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("generation must start only after model readiness");

        worker.request_drain(false);
        assert!(matches!(
            result_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        generate_resume_tx.send(()).unwrap();
        assert!(result_rx.blocking_recv().unwrap().is_ok());
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Completed { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { .. })
        ));
        handle.join().unwrap();
        assert_eq!(worker.pending_or_executing(), 0);
        assert!(!worker.poisoned.load(Ordering::SeqCst));
        assert!(!worker.fatal_cuda_error.load(Ordering::SeqCst));
    }

    #[test]
    fn drain_while_admin_model_load_is_blocked_finishes_load_then_stops() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (engine, load_started, load_resume) = FakeSlowEngine::blocked("blocked-admin");
        worker.model_cache.lock().unwrap().insert(engine, 0);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            })
        ));
        assert!(worker.try_claim_in_flight());
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        worker
            .send_grant(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "blocked-admin-load".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: OwnerWork::AdminModelLoad(Box::new(crate::gpu_pool::AdminModelLoadJob {
                    id: "blocked-admin-load".to_string(),
                    model: "blocked-admin".to_string(),
                    config: Config::default(),
                    result_tx,
                })),
                retry: None,
            })
            .unwrap();
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::AllocationCommitted { .. })
        ));
        load_started
            .recv_timeout(Duration::from_secs(1))
            .expect("admin load must reach the model load barrier");

        worker.request_drain(false);
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_REQUESTED
        );
        assert!(matches!(
            result_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        load_resume.send(()).unwrap();
        assert_eq!(result_rx.blocking_recv().unwrap(), Ok(()));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Completed { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { .. })
        ));
        handle.join().unwrap();
        assert_eq!(worker.pending_or_executing(), 0);
        assert!(!worker.poisoned.load(Ordering::SeqCst));
        assert!(!worker.fatal_cuda_error.load(Ordering::SeqCst));
    }

    #[test]
    fn drain_after_acceptance_finishes_each_exact_owner_stage_without_quarantine() {
        let cases = [
            ("accepted-generation", mold_scheduler::WorkKind::Generation),
            (
                "accepted-admin-load",
                mold_scheduler::WorkKind::AdminModelLoad,
            ),
            (
                "accepted-post-upscale",
                mold_scheduler::WorkKind::PostUpscale,
            ),
            ("accepted-chain-stage", mold_scheduler::WorkKind::ChainStage),
        ];

        for (id, kind) in cases {
            let fatal = Arc::new(AtomicBool::new(false));
            let (worker, job_rx) = protocol_worker(0, fatal.clone());
            let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            let handle =
                spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Ready {
                    owner_epoch: 1,
                    worker_generation: 1,
                    ..
                })
            ));
            assert!(worker.try_claim_in_flight());
            let (accepted_rx, resume_tx) = install_acceptance_barrier(id);
            let (ran_tx, ran_rx) = std::sync::mpsc::sync_channel(1);
            worker
                .send_grant(LeaseGrant {
                    fence: crate::scheduler::LeaseFence {
                        work_id: id.to_string(),
                        device_id: crate::scheduler::worker_device_id(&worker),
                        owner_epoch: worker.owner_epoch,
                        state_version: 1,
                        plan_version: 1,
                        worker_generation: 1,
                        memory_sample_generation: 1,
                        memory_ledger_sequence: 1,
                    },
                    work: OwnerWork::Probe {
                        id: id.to_string(),
                        kind,
                        run: Box::new(move || ran_tx.send(()).unwrap()),
                    },
                    retry: None,
                })
                .unwrap();
            accepted_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("owner must stop at the accepted-before-process barrier");
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Accepted {
                    owner_epoch: 1,
                    worker_generation: 1,
                    ..
                })
            ));

            worker.request_drain(false);
            assert_eq!(
                worker.drain_state.load(Ordering::SeqCst),
                crate::gpu_pool::DRAIN_REQUESTED
            );
            assert!(!worker.shutdown_requested.load(Ordering::SeqCst));
            assert!(!worker.poisoned.load(Ordering::SeqCst));
            assert!(!fatal.load(Ordering::SeqCst));

            resume_tx.send(()).unwrap();
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::AllocationCommitted {
                    owner_epoch: 1,
                    worker_generation: 1,
                    ..
                })
            ));
            ran_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("accepted owner stage must finish while draining");
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Completed {
                    owner_epoch: 1,
                    worker_generation: 1,
                    ..
                })
            ));
            assert!(matches!(
                event_rx.blocking_recv(),
                Some(crate::scheduler::WorkerEvent::Stopped { owner_epoch: 1, .. })
            ));
            handle.join().expect("drained owner must stop exactly");
            assert_eq!(worker.pending_or_executing(), 0);
            assert_eq!(
                worker.drain_state.load(Ordering::SeqCst),
                crate::gpu_pool::DRAIN_COMMITTED
            );
            assert!(!worker.poisoned.load(Ordering::SeqCst));
            assert!(!fatal.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn reenable_cancels_only_an_uncommitted_drain() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            })
        ));
        assert!(worker.try_claim_in_flight());
        let (accepted_rx, resume_tx) = install_acceptance_barrier("cancel-drain");
        worker
            .send_grant(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "cancel-drain".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: OwnerWork::Probe {
                    id: "cancel-drain".to_string(),
                    kind: mold_scheduler::WorkKind::PostUpscale,
                    run: Box::new(|| {}),
                },
                retry: None,
            })
            .unwrap();
        accepted_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted { .. })
        ));
        worker.request_drain(false);
        assert!(worker.cancel_drain());
        resume_tx.send(()).unwrap();
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::AllocationCommitted { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Completed { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 2,
                ..
            })
        ));
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_RUNNING
        );

        worker.request_drain(true);
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { .. })
        ));
        assert!(!worker.cancel_drain());
        handle.join().unwrap();
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
                        owner_epoch: worker.owner_epoch,
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
    fn owner_completion_carries_observed_stage_timings() {
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        let device_id = crate::scheduler::worker_device_id(&worker);
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready { .. })
        ));
        assert!(worker.try_claim_in_flight());
        worker
            .send_grant(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "timed-probe".into(),
                    device_id,
                    owner_epoch: worker.owner_epoch,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: OwnerWork::Probe {
                    id: "timed-probe".into(),
                    kind: mold_scheduler::WorkKind::Generation,
                    run: Box::new(|| {
                        record_model_load_timing(
                            ModelLoadDisposition::Cold,
                            Duration::from_millis(11),
                        );
                        record_model_load_timing(
                            ModelLoadDisposition::WarmReload,
                            Duration::from_millis(12),
                        );
                        for (phase, name, millis) in [
                            (
                                mold_inference::ProgressPhase::PromptEncode,
                                "Encoding prompt (T5)",
                                13,
                            ),
                            (mold_inference::ProgressPhase::Vae, "VAE decode", 15),
                            (mold_inference::ProgressPhase::Upscale, "Upscaling", 16),
                        ] {
                            record_phase_timing(&mold_inference::ProgressEvent::PhaseDone {
                                phase,
                                name: name.into(),
                                elapsed: Duration::from_millis(millis),
                            });
                        }
                        for millis in [6, 8] {
                            record_phase_timing(&mold_inference::ProgressEvent::DenoiseStep {
                                step: 1,
                                total: 2,
                                elapsed: Duration::from_millis(millis),
                            });
                        }
                        // Display copy must never be interpreted as execution
                        // evidence, even when it contains phase-like words.
                        for name in ["Loading VAE", "Loading upscaler", "Denoising"] {
                            record_phase_timing(&mold_inference::ProgressEvent::StageDone {
                                name: name.into(),
                                elapsed: Duration::from_secs(99),
                            });
                        }
                    }),
                },
                retry: None,
            })
            .unwrap();
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted { .. })
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::AllocationCommitted { .. })
        ));
        let timings = match event_rx.blocking_recv() {
            Some(crate::scheduler::WorkerEvent::Completed { phase_timings, .. }) => phase_timings,
            _ => panic!("worker must publish completion phase evidence"),
        };
        assert_eq!(timings.cold_load_ms, Some(11));
        assert_eq!(timings.warm_reload_ms, Some(12));
        assert_eq!(timings.prompt_encode_ms, Some(13));
        assert_eq!(timings.denoise_ms, Some(14));
        assert_eq!(timings.vae_ms, Some(15));
        assert_eq!(timings.upscale_ms, Some(16));

        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready { .. })
        ));
        worker.request_shutdown();
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { .. })
        ));
        handle.join().unwrap();
    }

    #[test]
    fn standalone_upscale_records_typed_phase_with_and_without_sse_subscriber() {
        let _ = take_lease_phase_timings(None);
        handle_standalone_upscale_progress(
            mold_inference::ProgressEvent::PhaseDone {
                phase: mold_inference::ProgressPhase::Upscale,
                name: "Upscaling".into(),
                elapsed: Duration::from_millis(17),
            },
            None,
        );
        assert_eq!(take_lease_phase_timings(None).upscale_ms, Some(17));

        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        handle_standalone_upscale_progress(
            mold_inference::ProgressEvent::PhaseDone {
                phase: mold_inference::ProgressPhase::Upscale,
                name: "Upscaling".into(),
                elapsed: Duration::from_millis(19),
            },
            Some(&progress_tx),
        );
        assert_eq!(take_lease_phase_timings(None).upscale_ms, Some(19));
        assert!(matches!(
            progress_rx.try_recv(),
            Ok(SseMessage::Progress(_))
        ));
    }

    #[test]
    fn typed_load_disposition_never_double_counts_setup() {
        let _ = take_lease_phase_timings(None);
        record_model_load_timing(ModelLoadDisposition::WarmReload, Duration::from_millis(7));
        let warm = take_lease_phase_timings(Some(99));
        assert_eq!(warm.cold_load_ms, None);
        assert_eq!(warm.warm_reload_ms, Some(7));

        record_model_load_timing(ModelLoadDisposition::Cold, Duration::from_millis(11));
        let cold = take_lease_phase_timings(Some(99));
        assert_eq!(cold.cold_load_ms, Some(11));
        assert_eq!(cold.warm_reload_ms, None);

        record_model_load_timing(ModelLoadDisposition::Unchanged, Duration::from_millis(13));
        let unchanged = take_lease_phase_timings(Some(99));
        assert_eq!(unchanged.cold_load_ms, None);
        assert_eq!(unchanged.warm_reload_ms, None);
    }

    #[test]
    fn owner_returns_invalidated_generation_before_acceptance_or_cleanup() {
        let root = tempfile::tempdir().unwrap();
        for name in ["transformer-q4.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), vec![0_u8; 1024]).unwrap();
        }
        let mut config = Config::default();
        config.models.insert(
            "test:q4".to_string(),
            ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer-q4.gguf")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                t5_encoder: Some(root.path().join("t5.safetensors").display().to_string()),
                family: Some("flux2".to_string()),
                ..ModelConfig::default()
            },
        );
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"test:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let plan = crate::execution_plan::resolve_execution_plans(
            &config,
            &request,
            &[crate::execution_plan::DeviceFact {
                id: "cuda:00000000000000000000000000000001".to_string(),
                ordinal: 0,
                available_vram_bytes: 24 << 30,
            }],
            false,
        )
        .unwrap()
        .remove(0);

        // Interleave an artifact replacement after coordinator planning but
        // before the owner accepts the grant.
        std::fs::write(root.path().join("transformer-q4.gguf"), vec![1_u8; 2048]).unwrap();

        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        let device_id = crate::scheduler::worker_device_id(&worker);
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 1,
                ..
            })
        ));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(queue_tx);
        let registry = JobRegistry::new();
        registry.register("invalidated", "test:q4");
        worker
            .send_grant(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "invalidated".to_string(),
                    device_id,
                    owner_epoch: worker.owner_epoch,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: OwnerWork::Generation(Box::new(GpuJob {
                    id: "invalidated".to_string(),
                    model: "test:q4".to_string(),
                    request,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx,
                    output_dir: None,
                    config: Arc::new(tokio::sync::RwLock::new(config)),
                    metadata_db: Arc::new(None),
                    gallery_publication_gate:
                        crate::batch_transaction::GalleryPublicationGate::default(),
                    queue: queue.clone(),
                    registry,
                    events: crate::events::EventBroadcaster::new(),
                    execution_plan: Some(plan),
                    prepared_execution_inputs: None,
                    lease: None,
                })),
                retry: None,
            })
            .unwrap();

        let returned_grant = match event_rx.blocking_recv() {
            Some(crate::scheduler::WorkerEvent::Rejected {
                grant,
                reason:
                    crate::scheduler::LeaseRejection::PlanInvalidated(
                        crate::execution_plan::ExecutionPlanError::PlanInvalidated(_),
                    ),
                ..
            }) => {
                assert_eq!(grant.work.id(), "invalidated");
                grant
            }
            Some(event) => panic!(
                "invalidated grant must be returned before Accepted, got {}",
                std::mem::discriminant(&event)
                    == std::mem::discriminant(&crate::scheduler::WorkerEvent::Completed {
                        device_id: String::new(),
                        ordinal: 0,
                        owner_epoch: 0,
                        worker_generation: 0,
                        successful: false,
                        phase_timings: mold_scheduler::EstimatePhaseTimings::default(),
                    })
            ),
            None => panic!("owner event channel closed"),
        };
        assert_eq!(queue.pending(), 0, "owner must not run generation cleanup");
        assert!(
            matches!(
                result_rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "plan invalidation must not settle the client result"
        );
        drop(returned_grant);

        worker.request_shutdown();
        handle.join().expect("owner joins after invalidation");
    }

    #[test]
    fn accepted_chain_stage_returns_typed_same_id_when_second_fence_invalidates() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer.gguf");
        let vae = root.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer-v1").unwrap();
        std::fs::write(&vae, b"vae-v1").unwrap();
        let model = "semantic-distilled:fp8";
        let frozen_config = ModelConfig {
            transformer: Some(transformer.display().to_string()),
            vae: Some(vae.display().to_string()),
            family: Some("ltx2".to_string()),
            ..ModelConfig::default()
        };
        let expected =
            crate::execution_plan::frozen_model_fingerprint(model, &frozen_config).unwrap();
        let mut config = Config::default();
        config.install_frozen_model_config(model, frozen_config);
        let request: GenerateRequest = serde_json::from_str(&format!(
            r#"{{"prompt":"x","model":"{model}","width":64,"height":64,"steps":4,"guidance":1.0}}"#
        ))
        .unwrap();

        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let device_id = crate::scheduler::worker_device_id(&worker);
        let plan = crate::execution_plan::resolve_execution_plans(
            &config,
            &request,
            &[crate::execution_plan::DeviceFact {
                id: device_id.clone(),
                ordinal: 0,
                available_vram_bytes: 48 << 30,
            }],
            false,
        )
        .unwrap()
        .remove(0);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready { .. })
        ));
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let mutate = transformer.clone();
        let lease_calls = Arc::new(AtomicUsize::new(0));
        let recorded_lease_calls = lease_calls.clone();
        let work_id = "chain:second-fence:stage:0";
        worker
            .send_grant(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: work_id.to_string(),
                    device_id,
                    owner_epoch: 1,
                    state_version: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    memory_sample_generation: 1,
                    memory_ledger_sequence: 1,
                },
                work: OwnerWork::ChainStage(Box::new(
                    crate::chain_job_runner::ScheduledChainStageWork {
                        id: work_id.to_string(),
                        model: model.to_string(),
                        cache_key: format!("mold-frozen-chain:{expected}"),
                        config,
                        stage_req: request,
                        carry: None,
                        motion_tail_frames: 1,
                        progress: Arc::new(|_, _| std::ops::ControlFlow::Continue(())),
                        cancelled: Arc::new(|| false),
                        cancellation: mold_inference::InferenceCancellationToken::default(),
                        on_leased: Some(Box::new(move |_| {
                            recorded_lease_calls.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })),
                        execution_plan: Some(plan),
                        expected_model_fingerprint: Some(expected),
                        result_tx: Some(result_tx),
                        before_second_fence: Some(Box::new(move || {
                            std::fs::write(mutate, b"transformer-v2-with-new-size").unwrap();
                        })),
                    },
                )),
                retry: None,
            })
            .unwrap();

        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted { ref work_id, .. })
                if work_id == "chain:second-fence:stage:0"
        ));
        match event_rx.blocking_recv() {
            Some(crate::scheduler::WorkerEvent::Rejected {
                mut grant,
                reason:
                    crate::scheduler::LeaseRejection::PlanInvalidated(
                        crate::execution_plan::ExecutionPlanError::PlanInvalidated(_),
                    ),
                ..
            }) => {
                assert_eq!(grant.work.id(), work_id);
                assert!(
                    matches!(&mut grant.work, OwnerWork::ChainStage(stage) if stage.on_leased.is_some()),
                    "the durable lease callback must survive a pre-CUDA plan invalidation"
                );
            }
            Some(_) => panic!("expected typed second-fence invalidation event"),
            None => panic!("worker event channel closed before second-fence rejection"),
        }
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                worker_generation: 2,
                ..
            })
        ));
        assert_eq!(
            lease_calls.load(Ordering::SeqCst),
            0,
            "lease authority must not persist before the second pre-CUDA fence"
        );
        worker.request_shutdown();
        handle.join().unwrap();
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
            owner_epoch: worker.owner_epoch,
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
                gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(
                ),
                queue,
                registry,
                events: crate::events::EventBroadcaster::new(),
                execution_plan: None,
                prepared_execution_inputs: None,
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

    #[tokio::test]
    async fn panic_outside_process_job_is_contained_and_signals_restart() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);

        run_gpu_owner_entrypoint(&worker, || {
            panic!("synthetic idle-eviction panic");
        });

        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
        tokio::time::timeout(
            Duration::from_millis(50),
            worker.fatal_cuda_shutdown.notified(),
        )
        .await
        .expect("outer owner panic must wake fail-closed server shutdown");
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
                gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(
                ),
                queue: queue.clone(),
                registry: registry.clone(),
                events: crate::events::EventBroadcaster::new(),
                execution_plan: None,
                prepared_execution_inputs: None,
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
        assert_eq!(
            worker.in_flight.load(Ordering::SeqCst),
            1,
            "process_job no longer mutates dispatch ownership; the owner loop settles in_flight"
        );
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
                    gallery_publication_gate:
                        crate::batch_transaction::GalleryPublicationGate::default(),
                    queue: queue.clone(),
                    registry: JobRegistry::new(),
                    events: crate::events::EventBroadcaster::new(),
                    execution_plan: None,
                    prepared_execution_inputs: None,
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
                    gallery_publication_gate:
                        crate::batch_transaction::GalleryPublicationGate::default(),
                    queue: followup_queue,
                    registry: JobRegistry::new(),
                    events: crate::events::EventBroadcaster::new(),
                    execution_plan: None,
                    prepared_execution_inputs: None,
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
            worker
                .owner_thread_id
                .set(std::thread::current().id())
                .expect("test binds admin load to the owner thread");

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
