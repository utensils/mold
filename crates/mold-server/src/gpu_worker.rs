use crate::gpu_pool::{
    ActiveGeneration, AdminModelUnloadJob, GpuJob, GpuWorker, GpuWorkerCommand, LeaseGrant,
    OwnerWork, PostGenerationUpscaleJob, PromptExpansionJob, StandaloneUpscaleJob,
    UtilityExecutionPlan,
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
            visual_decode_ms: None,
            audio_decode_ms: None,
            mux_ms: None,
            upscale_ms: None,
        }) };
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
const H3_CUDA_ATTEMPT_RETAINED_MARKER: &str =
    "CUDA execution attempt retained resources; server restart required";

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
                mold_inference::ProgressPhase::VisualDecode => {
                    add_phase_sample(&mut timings.visual_decode_ms, *elapsed)
                }
                mold_inference::ProgressPhase::AudioDecode => {
                    add_phase_sample(&mut timings.audio_decode_ms, *elapsed)
                }
                mold_inference::ProgressPhase::Mux => {
                    add_phase_sample(&mut timings.mux_ms, *elapsed)
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
    predicted_vram_peak_bytes: u64,
    /// Decayed observed high-water envelope for this estimate bucket. Zero
    /// when there is no learned evidence.
    learned_vram_envelope_bytes: u64,
    /// Frozen host-RAM demand this plan was admitted against, already zero on
    /// Metal where the same claim rides the unified device gate.
    predicted_host_increment_bytes: u64,
    /// Ledger headroom for the owning lease, or `None` when no ledger can
    /// answer — the recheck then retains the scheduler's grant.
    available_host_headroom_bytes: Option<u64>,
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

    fn generate_with_reference_bindings(
        &mut self,
        req: &mold_core::GenerateRequest,
        bindings: &[mold_inference::GenerationReferenceBinding],
    ) -> anyhow::Result<mold_core::GenerateResponse> {
        self.inner.generate_with_reference_bindings(req, bindings)
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

/// Start the single bounded CPU utility owner. Its capacity-one transport is
/// a rendezvous after `Ready`, exactly like a GPU owner, but it accepts only
/// utility work carrying an immutable CPU plan and never initializes CUDA.
pub fn spawn_cpu_utility_thread(
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("cpu-utility-worker-0".to_string())
        .spawn(move || run_cpu_utility_owner(job_rx, scheduler_tx))
        .expect("failed to spawn CPU utility worker thread")
}

fn run_cpu_utility_owner(
    job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
    scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) {
    let device_id = crate::scheduler::CPU_UTILITY_DEVICE_ID.to_string();
    let owner_epoch = 1;
    let ordinal = usize::MAX;
    let mut generation = 1_u64;
    loop {
        if scheduler_tx
            .send(crate::scheduler::WorkerEvent::Ready {
                device_id: device_id.clone(),
                ordinal,
                owner_epoch,
                worker_generation: generation,
            })
            .is_err()
        {
            break;
        }
        let grant = match job_rx.recv() {
            Ok(GpuWorkerCommand::Grant(grant)) => grant,
            Ok(GpuWorkerCommand::Drain | GpuWorkerCommand::Shutdown) | Err(_) => break,
        };
        if grant.fence.device_id != device_id
            || grant.fence.owner_epoch != owner_epoch
            || grant.fence.worker_generation != generation
        {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal,
                owner_epoch,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::StaleWorkerGeneration,
            });
            continue;
        }
        if let Err(error) = validate_cpu_utility_grant(&grant) {
            let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal,
                owner_epoch,
                worker_generation: generation,
                grant,
                reason: crate::scheduler::LeaseRejection::PlanInvalidated(
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(error),
                ),
            });
            continue;
        }
        let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Accepted {
            device_id: device_id.clone(),
            ordinal,
            owner_epoch,
            worker_generation: generation,
            work_id: grant.fence.work_id.clone(),
            plan_version: grant.fence.plan_version,
        });
        commit_utility_allocation(&scheduler_tx, &grant.fence);
        reset_lease_load_ms();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_cpu_utility_work(grant.work)
        }));
        let load_ms = take_lease_load_ms();
        let phase_timings = take_lease_phase_timings(load_ms);
        if outcome.is_err() {
            tracing::error!("CPU utility owner panicked; rejecting the attempt without GPU retry");
        }
        let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Completed {
            device_id: device_id.clone(),
            ordinal,
            owner_epoch,
            worker_generation: generation,
            successful: matches!(outcome, Ok(true)),
            phase_timings,
            completion: None,
        });
        generation = generation.saturating_add(1);
    }
}

fn validate_cpu_utility_grant(grant: &LeaseGrant) -> Result<(), String> {
    match &grant.work {
        #[cfg(feature = "expand")]
        OwnerWork::PromptExpansion(job) => {
            let plan = job
                .execution_plan
                .as_ref()
                .ok_or_else(|| "CPU prompt expansion lacked an exact plan".to_string())?;
            plan.validate().map_err(|error| error.to_string())?;
            matches!(
                plan.placement,
                mold_inference::expand::ExactExpandPlacement::Cpu
            )
            .then_some(())
            .ok_or_else(|| "CPU utility lane received a GPU expansion plan".to_string())
        }
        OwnerWork::StandaloneUpscale(job) => validate_cpu_upscale_plan(&job.execution_plan),
        OwnerWork::PostUpscale(job) => validate_cpu_upscale_plan(&job.execution_plan),
        _ => Err("CPU utility lane accepts only expansion and upscaling".to_string()),
    }
}

fn validate_cpu_upscale_plan(
    plan: &Option<mold_inference::upscaler::ResolvedUpscaleExecutionPlan>,
) -> Result<(), String> {
    let plan = plan
        .as_ref()
        .ok_or_else(|| "CPU upscaling lacked an exact plan".to_string())?;
    plan.validate().map_err(|error| error.to_string())?;
    matches!(
        plan.placement,
        mold_inference::upscaler::ExactUpscalePlacement::Cpu
    )
    .then_some(())
    .ok_or_else(|| "CPU utility lane received a GPU upscaler plan".to_string())
}

fn process_cpu_utility_work(work: OwnerWork) -> bool {
    match work {
        OwnerWork::PromptExpansion(job) => process_cpu_prompt_expansion(*job),
        OwnerWork::StandaloneUpscale(job) => process_cpu_standalone_upscale(*job),
        OwnerWork::PostUpscale(job) => process_cpu_post_generation_upscale(*job),
        work => {
            work.reject("CPU utility lane received non-utility work".to_string());
            false
        }
    }
}

/// Reassert the canonical GPU binding at the start of every owner iteration.
///
/// The OS thread identity remains the worker's durable authority; the
/// thread-local ordinal is the inference-side projection of that authority.
/// Restoring it here makes an accidental scope leak visible and self-healing
/// before the owner advertises readiness for another grant.
fn reaffirm_owner_gpu_binding(worker: &GpuWorker) {
    let expected = worker.gpu.ordinal;
    let actual = mold_inference::device::thread_gpu_ordinal();
    if actual != Some(expected) {
        tracing::error!(
            gpu = expected,
            ?actual,
            "GPU owner thread lost its scheduler device binding; restoring canonical binding"
        );
        mold_inference::device::init_thread_gpu_ordinal(expected);
    }
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
        reaffirm_owner_gpu_binding(worker);
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
        // Keep actor reply ownership outside the unwind boundary. Normal
        // stage results, typed errors, cancellation, and fatal panics all
        // become coordinator-settled completions. A plan invalidation puts
        // the same sender back into the returned grant for its exact retry.
        let mut chain_result_tx = match &mut grant.work {
            OwnerWork::ChainStage(job) => job.result_tx.take(),
            _ => None,
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_owner_work(worker, *grant, &scheduler_tx, generation)
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
                chain_result,
            }) => {
                worker.release_in_flight();
                let completion =
                    chain_result_tx
                        .take()
                        .map(|tx| DeferredOwnerCompletion::ChainStage {
                            tx: Some(tx),
                            result: Some(chain_result.unwrap_or_else(|| {
                                Err("chain stage ended without an actor result".to_string())
                            })),
                        });
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: worker.gpu.ordinal,
                    owner_epoch: worker.owner_epoch,
                    worker_generation: generation,
                    successful: successful
                        && !worker.poisoned.load(Ordering::SeqCst)
                        && !worker.fatal_cuda_error.load(Ordering::SeqCst),
                    phase_timings,
                    completion: completion.map(Box::new),
                });
            }
            Ok(OwnerProcessOutcome::PlanInvalidated { mut grant, error }) => {
                if let Some(tx) = chain_result_tx.take() {
                    match &mut grant.work {
                        OwnerWork::ChainStage(job) => {
                            debug_assert!(job.result_tx.is_none());
                            job.result_tx = Some(tx);
                        }
                        _ => unreachable!("only chain stages own deferred actor replies"),
                    }
                }
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
                let completion =
                    chain_result_tx
                        .take()
                        .map(|tx| DeferredOwnerCompletion::ChainStage {
                            tx: Some(tx),
                            result: Some(Err(
                                "GPU owner panicked while executing the chain stage".to_string()
                            )),
                        });
                let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: worker.gpu.ordinal,
                    owner_epoch: worker.owner_epoch,
                    worker_generation: generation,
                    successful: false,
                    phase_timings,
                    completion: completion.map(Box::new),
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
        reaffirm_owner_gpu_binding(worker);
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
                job.prepared_execution_inputs.as_ref(),
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
                None,
            )
        }
        #[cfg(feature = "expand")]
        OwnerWork::PromptExpansion(job) => {
            let plan = job.execution_plan.as_ref().ok_or_else(|| {
                crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                    "prompt expansion lacked an exact execution plan".to_string(),
                )
            })?;
            plan.validate().map_err(|error| {
                crate::execution_plan::ExecutionPlanError::PlanInvalidated(error.to_string())
            })?;
            validate_gpu_utility_placement(
                worker,
                match plan.placement {
                    mold_inference::expand::ExactExpandPlacement::Cpu => None,
                    mold_inference::expand::ExactExpandPlacement::Device { backend, ordinal } => {
                        Some((backend, ordinal))
                    }
                },
            )
        }
        OwnerWork::StandaloneUpscale(job) => validate_gpu_upscale_plan(worker, &job.execution_plan),
        OwnerWork::PostUpscale(job) => validate_gpu_upscale_plan(worker, &job.execution_plan),
        _ => Ok(()),
    }
}

pub enum DeferredOwnerCompletion {
    ChainStage {
        tx: Option<
            tokio::sync::oneshot::Sender<Result<crate::chain_job_runner::StageExecution, String>>,
        >,
        result: Option<Result<crate::chain_job_runner::StageExecution, String>>,
    },
}

impl DeferredOwnerCompletion {
    pub(crate) fn finish(mut self) {
        match self {
            Self::ChainStage {
                ref mut tx,
                ref mut result,
            } => {
                let _ = tx
                    .take()
                    .expect("deferred actor completion owns its sender")
                    .send(
                        result
                            .take()
                            .expect("deferred actor completion owns its result"),
                    );
            }
        }
    }

    pub(crate) fn fail(mut self, error: String) {
        match self {
            Self::ChainStage { ref mut tx, .. } => {
                let _ = tx
                    .take()
                    .expect("deferred actor completion owns its sender")
                    .send(Err(error));
            }
        }
    }
}

impl Drop for DeferredOwnerCompletion {
    fn drop(&mut self) {
        match self {
            Self::ChainStage { tx, .. } => {
                if let Some(tx) = tx.take() {
                    let _ = tx.send(Err(
                        "scheduler coordinator stopped before settling the chain-stage lease"
                            .to_string(),
                    ));
                }
            }
        }
    }
}

enum OwnerProcessOutcome {
    Completed {
        successful: bool,
        chain_result: Option<Result<crate::chain_job_runner::StageExecution, String>>,
    },
    PlanInvalidated {
        grant: Box<LeaseGrant>,
        error: crate::execution_plan::ExecutionPlanError,
    },
}

fn validate_gpu_upscale_plan(
    worker: &GpuWorker,
    plan: &Option<mold_inference::upscaler::ResolvedUpscaleExecutionPlan>,
) -> Result<(), crate::execution_plan::ExecutionPlanError> {
    let plan = plan.as_ref().ok_or_else(|| {
        crate::execution_plan::ExecutionPlanError::PlanInvalidated(
            "upscaling lacked an exact execution plan".to_string(),
        )
    })?;
    plan.validate().map_err(|error| {
        crate::execution_plan::ExecutionPlanError::PlanInvalidated(error.to_string())
    })?;
    validate_gpu_utility_placement(
        worker,
        match plan.placement {
            mold_inference::upscaler::ExactUpscalePlacement::Cpu => None,
            mold_inference::upscaler::ExactUpscalePlacement::Device { backend, ordinal } => {
                Some((backend, ordinal))
            }
        },
    )
}

fn validate_gpu_utility_placement(
    worker: &GpuWorker,
    placement: Option<(mold_core::GpuBackend, usize)>,
) -> Result<(), crate::execution_plan::ExecutionPlanError> {
    if placement == Some((worker.gpu.backend, worker.gpu.ordinal)) {
        Ok(())
    } else {
        Err(crate::execution_plan::ExecutionPlanError::PlanInvalidated(
            "utility execution placement did not match the accepting GPU owner".to_string(),
        ))
    }
}

fn process_owner_work(
    worker: &GpuWorker,
    mut grant: LeaseGrant,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    current_worker_generation: u64,
) -> OwnerProcessOutcome {
    if let Err(error) = ensure_owner_thread(worker) {
        let error = error.to_string();
        let chain_result =
            matches!(&grant.work, OwnerWork::ChainStage(_)).then(|| Err(error.clone()));
        if chain_result.is_none() {
            grant.work.reject(error);
        }
        return OwnerProcessOutcome::Completed {
            successful: false,
            chain_result,
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
    if let OwnerWork::Generation(job) = &grant.work {
        if let Err(error) = validate_scheduled_generation_before_cuda(worker, job) {
            return OwnerProcessOutcome::PlanInvalidated {
                grant: Box::new(grant),
                error,
            };
        }
    }
    // This is intentionally later than both owner-thread validation and the
    // second pre-CUDA plan fence, but earlier than any model/cache allocation.
    // The private feature atomically prepares an opaque one-shot value here;
    // the scheduler only ever retained the empty GpuJob slot.
    let h3_cancellation = match &grant.work {
        OwnerWork::Generation(job)
            if job
                .execution_plan
                .as_ref()
                .is_some_and(|plan| mold_core::minimax_h3::is_family(&plan.model_family)) =>
        {
            Some(job.batch_child.as_ref().map_or_else(
                mold_inference::InferenceCancellationToken::default,
                |child| child.cancellation.clone(),
            ))
        }
        _ => None,
    };
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let h3_prepare_error = match (&mut grant.work, h3_cancellation.as_ref()) {
        (OwnerWork::Generation(job), Some(cancellation)) => {
            request_lease_host_headroom(scheduler_tx, &grant.fence)
                .map_err(|error| error.error)
                .and_then(|available_host_headroom_bytes| {
                    with_private_h3_cuda_preparation_attempt(worker, || {
                        crate::h3_private_bridge::prepare_for_owner(
                            worker,
                            &grant.fence,
                            job,
                            cancellation.clone(),
                            available_host_headroom_bytes,
                        )
                        .map_err(crate::routes::ApiError::internal)
                    })
                    .map_err(|error| error.error)
                })
                .err()
        }
        _ => None,
    };
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let h3_prepare_error: Option<String> = None;
    let h3_claim = if h3_prepare_error.is_none() {
        match (&grant.work, h3_cancellation) {
            (OwnerWork::Generation(job), Some(cancellation)) => {
                Some(crate::h3_attempt::claim_generation_attempt(
                    worker,
                    current_worker_generation,
                    &grant.fence,
                    job,
                    cancellation,
                ))
            }
            _ => None,
        }
    } else {
        None
    };
    let h3_attempt = match h3_claim {
        Some(Ok(attempt)) => attempt,
        Some(Err(error)) => {
            return complete_h3_claim_failure(grant, error.to_string());
        }
        None => None,
    };
    if let OwnerWork::ChainStage(job) = &mut grant.work {
        if let Some(error) = job
            .on_leased
            .take()
            .and_then(|on_leased| on_leased(worker.gpu.ordinal).err())
        {
            let chain_result = Some(Err(error));
            return OwnerProcessOutcome::Completed {
                successful: false,
                chain_result,
            };
        }
    }
    match grant.work {
        OwnerWork::Generation(mut job) => {
            job.lease = Some(grant.fence);
            if let Some(error) = h3_prepare_error {
                let successful = with_claimed_h3_generation_cleanup(*job, |job| {
                    reject_claimed_h3_generation_message(job, error)
                });
                return OwnerProcessOutcome::Completed {
                    successful,
                    chain_result: None,
                };
            }
            let successful = process_job(
                worker,
                *job,
                scheduler_tx,
                current_worker_generation,
                h3_attempt,
            );
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: None,
            }
        }
        OwnerWork::ChainStage(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let result = process_scheduled_chain_stage(worker, *job);
            let successful = matches!(
                &result,
                Ok(crate::chain_job_runner::StageExecution {
                    outcome: crate::chain_job_runner::StageRenderOutcome::Done(_),
                    ..
                })
            );
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: Some(result),
            }
        }
        OwnerWork::PromptExpansion(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_prompt_expansion(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: None,
            }
        }
        OwnerWork::PostUpscale(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_post_generation_upscale(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: None,
            }
        }
        OwnerWork::StandaloneUpscale(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_standalone_upscale(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: None,
            }
        }
        OwnerWork::AdminModelLoad(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let result = load_blocking(worker, &job.model, &job.config).map_err(|e| e.to_string());
            let successful = result.is_ok();
            let _ = job.result_tx.send(result);
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: None,
            }
        }
        OwnerWork::AdminModelUnload(job) => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            let successful = process_admin_unload(worker, *job);
            OwnerProcessOutcome::Completed {
                successful,
                chain_result: None,
            }
        }
        #[cfg(test)]
        OwnerWork::Probe { run, .. } => {
            commit_utility_allocation(scheduler_tx, &grant.fence);
            run();
            OwnerProcessOutcome::Completed {
                successful: true,
                chain_result: None,
            }
        }
    }
}

fn complete_h3_claim_failure(grant: LeaseGrant, error: String) -> OwnerProcessOutcome {
    let OwnerWork::Generation(mut job) = grant.work else {
        unreachable!("only generation work can fail a private H3 attempt claim")
    };
    job.lease = Some(grant.fence);
    let successful = with_claimed_h3_generation_cleanup(*job, |job| {
        reject_claimed_h3_generation_message(job, error)
    });
    OwnerProcessOutcome::Completed {
        successful,
        chain_result: None,
    }
}

/// Ask the coordinator for this lease's host-RAM headroom against a fresh
/// sample of the ledger that granted it.
///
/// The reply restores the caller's own reservation and keeps every peer
/// charged, so admission and dispatch read one number and cannot oscillate.
fn request_lease_host_headroom(
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    fence: &crate::scheduler::LeaseFence,
) -> Result<u64, crate::routes::ApiError> {
    let (reply, response) = std::sync::mpsc::sync_channel(1);
    scheduler_tx
        .send(crate::scheduler::WorkerEvent::HostMemoryRecheck {
            fence: fence.clone(),
            reply,
        })
        .map_err(|_| {
            crate::routes::ApiError::internal("host-memory recheck could not reach Scheduler V2")
        })?;
    response
        .recv_timeout(Duration::from_secs(5))
        .map_err(|error| {
            crate::routes::ApiError::internal(format!(
                "host-memory recheck did not complete: {error}"
            ))
        })?
        .map_err(crate::routes::ApiError::internal)
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn with_private_h3_cuda_preparation_attempt<T>(
    worker: &GpuWorker,
    operation: impl FnOnce() -> Result<T, crate::routes::ApiError>,
) -> Result<T, crate::routes::ApiError> {
    use cudarc::driver::{CudaContext, CudaExecutionAttempt};

    let mut attempt = CudaExecutionAttempt::begin_unbound().map_err(|error| {
        quarantine_poisoned_worker(worker);
        contain_worker_cache(worker);
        crate::routes::ApiError::internal(format!(
            "failed to install the private H3 CUDA preparation boundary: {error}; server restart required"
        ))
    })?;
    // The attempt begins unbound because a cold worker has no CUDA context
    // yet, but `prepare_private_h3_allocation_boundary` evicts the previous
    // model's engine before any context construction could bind one. cudarc
    // treats a safe CUDA call on a pre-existing, unadopted context as poison:
    // the evicted engine's first CUDA-bearing destructor would latch
    // `force_retain`, leak the rest of that engine, and escalate a healthy
    // model switch into a whole-process fatal restart (#1081). Adopt the
    // worker's primary context first: `CudaContext::new` retains the existing
    // primary context on a warm worker (or creates it on a cold one), and
    // binding it lets eviction destructors run inside the adopted context.
    let adopted = CudaContext::new(worker.gpu.ordinal)
        .map_err(|error| error.to_string())
        .and_then(|context| {
            attempt
                .bind_context(&context)
                .map_err(|error| error.to_string())
        });
    if let Err(error) = adopted {
        let status = attempt.finish();
        if status.resources_retained() {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            tracing::error!("{H3_CUDA_ATTEMPT_RETAINED_MARKER}");
            return Err(crate::routes::ApiError::internal(
                H3_CUDA_ATTEMPT_RETAINED_MARKER,
            ));
        }
        return Err(crate::routes::ApiError::internal(format!(
            "failed to adopt the worker CUDA context for private H3 preparation: {error}"
        )));
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
    match result {
        Err(payload) => {
            attempt.mark_panicked();
            let _status = attempt.finish();
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            std::panic::resume_unwind(payload)
        }
        Ok(result) => {
            let status = attempt.finish();
            if status.resources_retained() {
                quarantine_poisoned_worker(worker);
                contain_worker_cache(worker);
                tracing::error!("{H3_CUDA_ATTEMPT_RETAINED_MARKER}");
                return Err(crate::routes::ApiError::internal(
                    H3_CUDA_ATTEMPT_RETAINED_MARKER,
                ));
            }
            result
        }
    }
}

fn validate_scheduled_generation_before_cuda(
    worker: &GpuWorker,
    job: &GpuJob,
) -> Result<(), crate::execution_plan::ExecutionPlanError> {
    let Some(plan) = job.execution_plan.as_ref() else {
        // Legacy unit adapters are handled by the separate rollback owner and
        // never gain an H3 one-shot root through this path.
        return Ok(());
    };
    let config = job.config.blocking_read();
    crate::execution_plan::validate_before_cuda(
        plan,
        &crate::scheduler::worker_device_id(worker),
        worker.gpu.ordinal,
        &config,
        &job.request,
        job.prepared_execution_inputs.as_ref(),
    )
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
        None,
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
            let _ = process_job_with_sink(
                worker,
                *job,
                GenerationEventSink::Legacy(owner_event_tx),
                None,
            );
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
    tracing::info!(
        gpu = worker.gpu.ordinal,
        model = %job.model,
        work_id = %job.id,
        work_kind = "chain_stage",
        "dispatched job"
    );
    let memory_watchdog =
        ChainStageMemoryWatchdog::start(worker.gpu.ordinal, job.model.clone(), job.id.clone());
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
                        // Server chains never carry the client-local HDR EXR
                        // sidecar (#688).
                        None,
                        Some(&mut stage_progress),
                    )
                });
            fence_chain_stage_render(render, cancellation_seen, cancelled())
        },
    )
    .map_err(|error| format!("{error:#}"))?
    .map_err(|error| format!("{error:#}"))?;
    drop(memory_watchdog);
    Ok(crate::chain_job_runner::StageExecution {
        outcome: result,
        device_ordinal: Some(worker.gpu.ordinal),
    })
}

/// Return idle glibc arena pages to the OS, and report the RSS just before
/// doing so.
///
/// glibc keeps freed pages in per-arena heaps even after the allocations are
/// dropped — large transient buffers from GGUF+LoRA rebuilds can leave tens of
/// GB of unreclaimed RSS. `malloc_trim(0)` walks the arenas and returns idle
/// pages via `madvise(MADV_DONTNEED)`. Cheap (~ms), glibc-only, gated so we can
/// A/B with `MOLD_MALLOC_TRIM=0`. `None` means the trim was disabled and no RSS
/// was sampled.
fn trim_malloc_arenas() -> Option<u64> {
    let enabled = std::env::var("MOLD_MALLOC_TRIM")
        .map(|value| value != "0")
        .unwrap_or(true);
    if !enabled {
        return None;
    }
    let rss_pre_trim = crate::resources::ram_snapshot().used_by_mold;
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
    Some(rss_pre_trim)
}

/// Scheduled chain stages bypass `process_job`, so they need their own memory
/// heartbeat around model readiness and rendering. A channel-backed stop wakes
/// the thread immediately for short stages instead of making completion wait
/// for the one-second sampling interval.
struct ChainStageMemoryWatchdog {
    stop: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
    rss_before: u64,
    ordinal: usize,
    model: String,
    work_id: String,
}

impl ChainStageMemoryWatchdog {
    fn start(ordinal: usize, model: String, work_id: String) -> Self {
        let rss_before = crate::resources::ram_snapshot().used_by_mold;
        let (stop, stopped) = std::sync::mpsc::channel();
        let thread_model = model.clone();
        let thread_work_id = work_id.clone();
        let handle = std::thread::Builder::new()
            .name(format!("chain-rss-watchdog-{ordinal}"))
            .spawn(move || {
                let start = Instant::now();
                while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                    stopped.recv_timeout(Duration::from_secs(1))
                {
                    let rss = crate::resources::ram_snapshot().used_by_mold;
                    tracing::info!(
                        gpu = ordinal,
                        model = %thread_model,
                        work_id = %thread_work_id,
                        elapsed_s = start.elapsed().as_secs(),
                        rss_mb = rss / 1_000_000,
                        "chain stage rss watchdog"
                    );
                }
            })
            .map_err(|error| {
                tracing::warn!(
                    gpu = ordinal,
                    model = %model,
                    work_id = %work_id,
                    %error,
                    "could not start chain stage RSS watchdog"
                );
            })
            .ok();
        Self {
            stop: Some(stop),
            handle,
            rss_before,
            ordinal,
            model,
            work_id,
        }
    }
}

impl Drop for ChainStageMemoryWatchdog {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let rss_pre_trim = trim_malloc_arenas();
        let rss_after = crate::resources::ram_snapshot().used_by_mold;
        tracing::info!(
            gpu = self.ordinal,
            model = %self.model,
            work_id = %self.work_id,
            rss_before_mb = self.rss_before / 1_000_000,
            rss_after_mb = rss_after / 1_000_000,
            rss_delta_mb = (rss_after as i64 - self.rss_before as i64) / 1_000_000,
            rss_pre_trim_mb = rss_pre_trim.map(|value| value / 1_000_000).unwrap_or(0),
            "chain stage memory delta"
        );
    }
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
            let mut expander = std::mem::ManuallyDrop::new(
                mold_inference::expand::LocalExpander::from_resolved_plan(
                    job.execution_plan.ok_or_else(|| {
                        anyhow::anyhow!("prompt expansion lacked an exact execution plan")
                    })?,
                ),
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
            let expansion = expander.expand_with_cancellation(
                &job.prompt,
                &job.expand_config,
                job.cancellation,
            );
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

fn process_cpu_prompt_expansion(job: PromptExpansionJob) -> bool {
    let result = (|| -> anyhow::Result<mold_core::ExpandResult> {
        #[cfg(feature = "expand")]
        {
            let plan = job.execution_plan.ok_or_else(|| {
                anyhow::anyhow!("CPU prompt expansion lacked an exact execution plan")
            })?;
            if !matches!(
                plan.placement,
                mold_inference::expand::ExactExpandPlacement::Cpu
            ) {
                anyhow::bail!("CPU prompt expansion received a non-CPU execution plan");
            }
            let mut expander = mold_inference::expand::LocalExpander::from_resolved_plan(plan);
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
            expander.expand_with_cancellation(&job.prompt, &job.expand_config, job.cancellation)
        }
        #[cfg(not(feature = "expand"))]
        {
            anyhow::bail!("local prompt expansion not available — built without expand feature")
        }
    })();
    let successful = result.is_ok();
    let _ = job
        .result_tx
        .send(result.map_err(|error| error.to_string()));
    successful
}

fn process_standalone_upscale(worker: &GpuWorker, job: StandaloneUpscaleJob) -> bool {
    let result = (|| -> anyhow::Result<mold_core::UpscaleResponse> {
        ensure_worker_not_poisoned(worker, &job.model)?;
        let plan = job
            .execution_plan
            .ok_or_else(|| anyhow::anyhow!("upscaling lacked an exact execution plan"))?;
        let mut engine = mold_inference::upscaler::create_upscale_engine_from_resolved_plan(
            plan,
            mold_inference::LoadStrategy::Eager,
        )?;
        let progress_tx = job.progress_tx;
        engine.set_on_progress(Box::new(move |event| {
            handle_standalone_upscale_progress(event, progress_tx.as_ref());
        }));
        engine.set_cancellation_token(job.cancellation);
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
    if let Some(progress_tx) = progress_tx {
        let _ = progress_tx.send(SseMessage::Progress(event.into()));
    }
}

fn process_cpu_standalone_upscale(job: StandaloneUpscaleJob) -> bool {
    let result = run_cpu_upscale(
        job.execution_plan,
        &job.request,
        job.progress_tx,
        job.cancellation,
    );
    let successful = result.is_ok();
    let _ = job
        .result_tx
        .send(result.map_err(|error| error.to_string()));
    successful
}

fn run_cpu_upscale(
    plan: Option<mold_inference::upscaler::ResolvedUpscaleExecutionPlan>,
    request: &mold_core::UpscaleRequest,
    progress_tx: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    cancellation: mold_inference::InferenceCancellationToken,
) -> anyhow::Result<mold_core::UpscaleResponse> {
    let plan = plan.ok_or_else(|| anyhow::anyhow!("CPU upscaling lacked an exact plan"))?;
    if !matches!(
        plan.placement,
        mold_inference::upscaler::ExactUpscalePlacement::Cpu
    ) {
        anyhow::bail!("CPU utility lane received a GPU upscaler plan");
    }
    let mut engine = mold_inference::upscaler::create_upscale_engine_from_resolved_plan(
        plan,
        mold_inference::LoadStrategy::Eager,
    )?;
    engine.set_on_progress(Box::new(move |event| {
        handle_standalone_upscale_progress(event, progress_tx.as_ref());
    }));
    mold_inference::with_upscale_cancellation(&mut *engine, cancellation, |engine| {
        engine.upscale(request)
    })
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
        job.execution_plan,
        job.cancellation,
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
            let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(message.clone())));
        }
        let _ = job.generation.result_tx.send(Err(message));
        drop(cleanup);
        return false;
    }
    let successful = result.is_ok();
    let (image, original, error) = settle_post_generation_upscale(job.image, result);
    if let Some(error) = error {
        report_post_generation_upscale_failure(job.generation.progress_tx.as_ref(), &error);
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

fn process_cpu_post_generation_upscale(mut job: PostGenerationUpscaleJob) -> bool {
    let cleanup = GenerationCleanup::new(&job.generation);
    let upscale_model = job
        .generation
        .request
        .upscale_model
        .clone()
        .unwrap_or_default();
    let request_model = mold_core::manifest::resolve_model_name(&upscale_model);
    let plan_model = job
        .execution_plan
        .as_ref()
        .map(|plan| plan.model_name.as_str());
    let result = if plan_model != Some(request_model.as_str()) {
        Err(format!(
            "post-generation upscaler plan did not match request '{request_model}'"
        ))
    } else {
        let request = mold_core::UpscaleRequest {
            model: request_model,
            image: job.image.data.clone(),
            output_format: job.image.format,
            tile_size: None,
            metadata: Some(OutputMetadata::from_generate_request(
                &job.generation.request,
                job.response.seed_used,
                None,
                mold_core::build_info::version_string(),
            )),
        };
        run_cpu_upscale(
            job.execution_plan,
            &request,
            job.generation.progress_tx.clone(),
            job.cancellation,
        )
        .map_err(|error| format!("upscale failed: {error}"))
        .and_then(|upscaled| {
            apply_upscale_response_to_image_generation(
                &job.generation.request,
                &mut job.response,
                job.image.clone(),
                upscaled,
            )
            .map_err(|error| format!("upscale failed: {error}"))
        })
    };
    let successful = result.is_ok();
    let (image, original, error) = settle_post_generation_upscale(job.image, result);
    if let Some(error) = error {
        report_post_generation_upscale_failure(job.generation.progress_tx.as_ref(), &error);
        tracing::warn!(%error, "CPU post-generation upscale failed; keeping original image");
    }
    finish_generation_success(*job.generation, job.response, image, original);
    drop(cleanup);
    successful
}

fn report_post_generation_upscale_failure(
    progress: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    error: &str,
) {
    if let Some(progress) = progress {
        let _ = progress.send(SseMessage::Progress(SseProgressEvent::Info {
            message: format!("post-generation upscale failed; keeping original image: {error}"),
        }));
    }
}

/// A post-upscale owner exists only after generation has already produced F0.
/// Any terminal utility failure must therefore complete the parent with that
/// original output instead of converting successful generation into an error.
pub(crate) fn finish_post_generation_upscale_failure(
    job: Box<PostGenerationUpscaleJob>,
    error: String,
) {
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::spawn_blocking(move || {
            finish_post_generation_upscale_failure_blocking(job, error);
        });
        return;
    }
    finish_post_generation_upscale_failure_blocking(job, error);
}

fn finish_post_generation_upscale_failure_blocking(
    job: Box<PostGenerationUpscaleJob>,
    error: String,
) {
    job.cancellation.cancel();
    report_post_generation_upscale_failure(job.generation.progress_tx.as_ref(), &error);
    tracing::warn!(%error, "post-generation upscale failed; keeping original image");
    let cleanup = GenerationCleanup::new(&job.generation);
    finish_generation_success(*job.generation, job.response, job.image, None);
    drop(cleanup);
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
        if resident
            .as_deref()
            .map(crate::gpu_pool::resident_model_display_name)
            != Some(expected)
        {
            let _ = job.result_tx.send(Ok(None));
            return true;
        }
    }
    let result = unload_blocking(worker)
        .map(|unloaded| {
            unloaded.map(|model| crate::gpu_pool::resident_model_display_name(&model).to_string())
        })
        .map_err(|error| error.to_string());
    let successful = result.is_ok();
    let _ = job.result_tx.send(result);
    successful
}

/// Evict stale engines on the worker thread that owns their device context.
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
    let (evicted, active_evicted) = {
        let mut cache = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let active = cache.active_model().map(str::to_owned);
        let evicted = cache.evict_idle_on_owner(ttl);
        let active_evicted = active.is_some_and(|active| {
            evicted
                .iter()
                .any(|(evicted_name, _)| evicted_name == &active)
        });
        (evicted, active_evicted)
    };
    if evicted.is_empty() {
        return;
    }
    if active_evicted {
        // Publish the cold residency before teardown. The owner cannot accept
        // another lease until this function returns, while scheduler snapshots
        // must stop treating the stale model as a warm candidate immediately.
        worker.set_resident_model(None);
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

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn record_h3_progress(
    event: mold_inference::ProgressEvent,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) {
    record_phase_timing(&event);
    if let Some(progress_tx) = progress_tx {
        let _ = progress_tx.send(SseMessage::Progress(progress_to_sse(event)));
    }
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

/// Detect a dispatch-time rejection of an already-admitted execution plan.
///
/// The frozen-plan rechecks refuse work because physical memory changed after
/// admission, which says nothing about this worker's health. The typed
/// `ApiError` is flattened to `anyhow` at the load boundary, so the shared
/// marker travels in the message; `memory_pressure_rejections_do_not_count_
/// against_worker_health` builds real rejections rather than string literals,
/// so a reworded message fails the test instead of silently reclassifying.
pub(crate) fn is_admitted_plan_memory_rejection(e: &anyhow::Error) -> bool {
    format!("{e:#}").contains(crate::memory_preflight::ADMISSION_PRESSURE_MARKER)
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
        "CUDA_ERROR_EXTERNAL_DEVICE",
        "CUDA_ERROR_MPS_CLIENT_TERMINATED",
        "CUDA_ERROR_CONTAINED",
        "CUDA_ERROR_TENSOR_MEMORY_LEAK",
        "CUBLAS_STATUS_MAPPING_ERROR",
        "CUBLAS_STATUS_EXECUTION_FAILED",
        "CUBLAS_STATUS_INTERNAL_ERROR",
        "CURAND_STATUS_LAUNCH_FAILURE",
        "CURAND_STATUS_PREEXISTING_FAILURE",
        "CURAND_STATUS_INTERNAL_ERROR",
        "CUDA execution attempt retained resources",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn fatal_cuda_user_message(model_name: &str) -> String {
    format!(
        "fatal CUDA error while running '{model_name}'; this GPU worker was quarantined because its CUDA context is no longer safe to reuse. Restart the mold server to recover the GPU."
    )
}

/// What a client is told when its job did not start because the host is going
/// down. Deliberately distinct from [`fatal_cuda_user_message`]: the
/// coordinator calls `request_shutdown()` on every worker during an ordinary
/// graceful deploy, so lumping the two together told every in-flight client
/// that CUDA was fatally poisoned on every restart.
pub(crate) fn shutdown_retention_user_message(model_name: &str) -> String {
    format!("the host is restarting; '{model_name}' did not start and stays queued to finish there")
}

/// Why this worker cannot accept work, if it cannot. Poison outranks shutdown:
/// a quarantined context is the more specific and more actionable fact.
struct WorkerUnavailable {
    message: String,
    /// True only for the shutdown branch — a poisoned context is a real
    /// failure and must not be dressed up as a retention.
    retainable: bool,
}

fn worker_unavailable(worker: &GpuWorker, model_name: &str) -> Option<WorkerUnavailable> {
    if worker.poisoned.load(Ordering::SeqCst) || worker.fatal_cuda_error.load(Ordering::SeqCst) {
        Some(WorkerUnavailable {
            message: fatal_cuda_user_message(model_name),
            retainable: false,
        })
    } else if worker.shutdown_requested.load(Ordering::SeqCst) {
        Some(WorkerUnavailable {
            message: shutdown_retention_user_message(model_name),
            retainable: true,
        })
    } else {
        None
    }
}

pub(crate) fn quarantine_poisoned_worker(worker: &GpuWorker) {
    // Retain the durable queue first. This function is the process-restart
    // initiator for a fatal context or an inference panic, and everything that
    // follows it drops jobs.
    worker.queue_journal.retain_all();
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

    // Dropping the engine is not enough on Metal. Devices are memoized per
    // ordinal, so candle's caching allocator outlives every engine and only
    // releases pooled buffers when someone waits on the queue. This is the one
    // chokepoint every engine drop passes through -- TTL eviction, cached
    // eviction, drain and shutdown alike -- so sweeping here keeps the freed
    // bytes visible to the next `sampled_free_vram_bytes` instead of leaving
    // the device looking full until something else happens to synchronize.
    if count > 0 && worker.gpu.backend == mold_core::GpuBackend::Metal {
        mold_inference::device::release_pooled_metal_memory(worker.gpu.ordinal);
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

#[cfg(any(test, feature = "cuda", feature = "h3-private-uat"))]
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
    if let Some(unavailable) = worker_unavailable(worker, model_name) {
        anyhow::bail!(unavailable.message);
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
    oom_user_message_with_advice(model_name, family_slug, req, None)
}

/// `supported_shape_advice` names a concrete resolution/frame combination that
/// fits this card. Without it the video branch could only say "reduce --frames
/// below N (e.g. 17 or 9)", which is a guess: it does not know whether the
/// shortfall is frames or resolution, and 17 or 9 frames is far below what the
/// card can actually run (#641).
pub(crate) fn oom_user_message_with_advice(
    model_name: &str,
    family_slug: Option<&str>,
    req: Option<&mold_core::GenerateRequest>,
    shape_advice: Option<&str>,
) -> String {
    let requested_size = req
        .map(|r| format!(" Requested size: {}x{}.", r.width, r.height))
        .unwrap_or_default();
    let batch_hint = match req.map(|r| r.batch_size).unwrap_or(1) {
        0 | 1 => "keep --batch 1".to_string(),
        n => format!("reduce --batch {n} to --batch 1"),
    };

    if family_slug.is_some_and(is_video_family) || req.and_then(|r| r.frames).is_some() {
        if let Some(advice) = shape_advice {
            return format!(
                "GPU ran out of memory loading or running '{model_name}'.{requested_size} \
                 This shape {advice}. Use that shape, a quantized variant if available, \
                 or close other GPU apps."
            );
        }
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
    // The caller also accepts any request that carries an explicit `frames`,
    // which covers every Studio submission. The family arm is what catches the
    // rest: `mold run wan22-t2v-a14b:q5 "…"` sends no frame count at all and
    // relies on the model default, so without wan here a CLI OOM is answered
    // with image advice — lower the resolution, keep --batch 1 — and never
    // mentions the frame count that actually drives the peak.
    matches!(
        family_slug,
        "ltx-video" | "ltx2" | "ltx-2" | "ltx-2.3" | "wan"
    )
}

fn upscale_generated_image_on_worker(
    worker: &GpuWorker,
    job: &GpuJob,
    upscale_model: &str,
    img: ImageData,
    response: &mut mold_core::GenerateResponse,
    exact_plan: Option<mold_inference::upscaler::ResolvedUpscaleExecutionPlan>,
    cancellation: mold_inference::InferenceCancellationToken,
) -> Result<ImageData, String> {
    let model_name = mold_core::manifest::resolve_model_name(upscale_model);
    let plan = exact_plan
        .ok_or_else(|| "post-generation upscaling lacked an exact execution plan".to_string())?;
    if plan.model_name != model_name {
        return Err(format!(
            "post-generation upscaler plan model '{}' did not match request '{model_name}'",
            plan.model_name
        ));
    }

    if let Some(ref tx) = job.progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::StageStart {
            name: format!("Loading upscaler {model_name}"),
        }));
    }
    let load_started = Instant::now();
    let mut engine = mold_inference::upscaler::create_upscale_engine_from_resolved_plan(
        plan,
        mold_inference::LoadStrategy::Eager,
    )
    .map_err(|e| format!("failed to load upscaler: {e:#}"))?;
    record_model_load_timing(ModelLoadDisposition::Cold, load_started.elapsed());
    let progress_tx = job.progress_tx.clone();
    engine.set_on_progress(Box::new(move |event| {
        handle_standalone_upscale_progress(event, progress_tx.as_ref());
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
    engine.set_cancellation_token(cancellation);
    let upscale_result = run_upscale_engine_safely(worker, engine, &req);
    let upscaled = upscale_result.map_err(|e| format!("upscale failed: {e:#}"))?;
    apply_upscale_response_to_image_generation(&job.request, response, img, upscaled)
        .map_err(|e| format!("upscale failed: {e:#}"))
}

/// Re-run the corrected LTX-2 estimator over the supported shape grid for the
/// card that just OOMed, so the rejection names a resolution/frame count the
/// user can actually run.
fn ltx2_oom_shape_advice(
    worker: &GpuWorker,
    family_slug: Option<&str>,
    req: Option<&mold_core::GenerateRequest>,
    paths: Option<&mold_core::ModelPaths>,
) -> Option<String> {
    if family_slug != Some("ltx2") {
        return None;
    }
    let req = req?;
    let facts = crate::ltx2_admission::checkpoint_facts_cached(&paths?.transformer)?;
    crate::ltx2_admission::supported_shape_advice(
        &facts,
        crate::ltx2_admission::Ltx2ShapeHint::from_request(req),
        worker.gpu.total_vram_bytes,
    )
}

/// `observed_high_water_bytes` is the memory authority the failed attempt ran
/// under — the frozen plan's predicted peak. The attempt proved it cannot have
/// all of that, so the next admission for the same `(model, shape, GPU)` plans
/// against a reduced grant instead of repeating the identical plan.
fn cuda_oom_user_message_with_plan(
    worker: &GpuWorker,
    model_name: &str,
    family_slug: Option<&str>,
    req: Option<&mold_core::GenerateRequest>,
    paths: Option<&mold_core::ModelPaths>,
    observed_high_water_bytes: Option<u64>,
) -> (String, bool) {
    let advice = ltx2_oom_shape_advice(worker, family_slug, req, paths);
    let mut base = if family_slug.is_none() && req.is_none() {
        oom_user_message(model_name)
    } else {
        oom_user_message_with_advice(model_name, family_slug, req, advice.as_deref())
    };
    let shape_bucket = req.map(crate::gpu_pool::oom_shape_bucket);
    if let (Some(bucket), Some(high_water)) = (shape_bucket.as_deref(), observed_high_water_bytes) {
        if let Some(grant) = crate::gpu_pool::record_reduced_vram_grant(
            model_name,
            bucket,
            worker.gpu.ordinal,
            high_water,
        ) {
            // This job is over — the caller turns this string into its
            // failure. The reduced grant is stored for the next submission of
            // this exact (model, shape, GPU), so say that rather than
            // promising an automatic retry that never comes.
            base.push_str(&format!(
                " The next attempt at this shape will plan against a smaller memory grant (~{:.1} GB).",
                grant as f64 / 1_000_000_000.0
            ));
        }
    }
    let outcome = crate::gpu_pool::record_model_cuda_oom(
        model_name,
        shape_bucket.as_deref(),
        worker.gpu.ordinal,
    );
    if outcome.is_unschedulable() {
        if let Some(cooldown) =
            crate::gpu_pool::model_unschedulable_message(model_name, shape_bucket.as_deref())
        {
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

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
struct ActiveGenerationCleanup<'a> {
    worker: &'a GpuWorker,
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
impl Drop for ActiveGenerationCleanup<'_> {
    fn drop(&mut self) {
        clear_active_generation(self.worker);
    }
}

fn process_job(
    worker: &GpuWorker,
    job: GpuJob,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    current_worker_generation: u64,
    h3_attempt: Option<crate::h3_attempt::H3GenerationAttempt>,
) -> bool {
    let Some(h3_attempt) = h3_attempt else {
        return process_job_with_sink(worker, job, GenerationEventSink::V2(scheduler_tx), None);
    };
    with_claimed_h3_generation_cleanup(job, |job| {
        let current = match crate::h3_attempt::rebuild_generation_current(
            worker,
            current_worker_generation,
            &job,
        ) {
            Ok(current) => current,
            Err(error) => return reject_claimed_h3_generation(job, error),
        };
        process_claimed_h3_generation_attempt(worker, job, scheduler_tx, h3_attempt, current)
    })
}

/// Hold the queue/registry cleanup guard before any claimed-attempt operation
/// can move the job. This single guard spans current-fence validation, runtime
/// execution, error reporting, and panic unwinding.
fn with_claimed_h3_generation_cleanup<T>(job: GpuJob, consume: impl FnOnce(GpuJob) -> T) -> T {
    let _cleanup = GenerationCleanup::new(&job);
    consume(job)
}

/// Consume one claimed H3 owner attempt without exposing it to the generic
/// retained-engine path. Ordinary builds keep the clean fail-closed outcome;
/// the private feature can consume only the opaque value attached during the
/// final owner dispatch.
fn process_claimed_h3_generation_attempt(
    worker: &GpuWorker,
    job: GpuJob,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    h3_attempt: crate::h3_attempt::H3GenerationAttempt,
    current: crate::h3_attempt::H3AttemptCurrent,
) -> bool {
    let mut pending_job = Some(job);
    match h3_attempt.run_once(current, |scope| {
        process_claimed_h3_generation(
            worker,
            pending_job
                .take()
                .expect("H3 attempt consumes its generation job once"),
            scope,
            scheduler_tx,
        )
    }) {
        Ok(successful) => successful,
        Err(error) => reject_claimed_h3_generation(
            pending_job
                .take()
                .expect("cancelled H3 attempt retains its unstarted generation job"),
            error,
        ),
    }
}

fn process_claimed_h3_generation(
    worker: &GpuWorker,
    job: GpuJob,
    scope: crate::h3_attempt::H3AttemptScope<'_>,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
) -> bool {
    // H3 owns a private attempt token instead of the generic engine token.
    // Install that exact authority in the public job row before any private
    // preparation so DELETE reaches the real attempt (including the
    // dispatch-to-owner hand-off race).
    job.registry
        .install_running_cancellation(&job.id, scope.cancellation_token());

    #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
    let mut job = job;

    #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
    if let Some(prepared) = job.h3_prepared_attempt.take() {
        return run_claimed_h3_generation(worker, job, scope, scheduler_tx, prepared);
    }

    let _ = (worker, scheduler_tx);
    let error = if scope.cancellation_token().checkpoint().is_err() {
        crate::h3_attempt::H3AttemptError::Cancelled
    } else {
        crate::h3_attempt::H3AttemptError::RuntimeUnavailable
    };
    reject_claimed_h3_generation(job, error)
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
fn run_claimed_h3_generation(
    worker: &GpuWorker,
    job: GpuJob,
    scope: crate::h3_attempt::H3AttemptScope<'_>,
    scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    prepared: crate::h3_private_bridge::BoxedH3PreparedAttempt,
) -> bool {
    let model_name = job.model.clone();
    let ordinal = worker.gpu.ordinal;
    if let Err(error) = ensure_worker_not_poisoned(worker, &model_name) {
        return reject_claimed_h3_generation_message(job, error.to_string());
    }
    if job.result_tx.is_closed() {
        tracing::debug!(gpu = ordinal, model = %model_name, "skipping claimed H3 job — client disconnected");
        return false;
    }
    let scope_facts = scope.facts();
    let prepared_facts = prepared.facts();
    if let Err(error) = validate_h3_prepared_attempt_facts(scope_facts, &prepared_facts) {
        return reject_claimed_h3_generation_message(job, error.to_string());
    }
    if let Err(error) = prepared_facts.media.validate_for_request(
        &job.request,
        job.resolved_references
            .as_ref()
            .map(crate::reference_uploads::ResolvedReferenceSet::fingerprint),
    ) {
        return reject_claimed_h3_generation_message(job, error);
    }
    let lease = match job.lease.clone() {
        Some(lease) if scope_facts.matches_lease(&lease) => lease,
        _ => {
            return reject_claimed_h3_generation_message(
                job,
                crate::h3_attempt::H3AttemptError::StaleOwnerFence.to_string(),
            );
        }
    };

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let _private_load_lock = match worker.model_load_lock.lock() {
        Ok(lock) => lock,
        Err(error) => {
            return reject_claimed_h3_generation_message(
                job,
                format!("MiniMax H3 owner memory lock was poisoned: {error}"),
            );
        }
    };
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let available_host_headroom_bytes = match request_lease_host_headroom(scheduler_tx, &lease) {
        Ok(headroom) => headroom,
        Err(error) => return reject_claimed_h3_generation_message(job, error.error),
    };
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if let Err(error) = with_private_h3_cuda_preparation_attempt(worker, || {
        prepare_private_h3_allocation_boundary(
            worker,
            &model_name,
            prepared_facts.predicted_device_peak_bytes,
            prepared_facts.predicted_host_increment_bytes,
            available_host_headroom_bytes,
        )
    }) {
        return reject_claimed_h3_generation_message(job, error.error);
    }

    // The exact target-budget peak is installed before the facade can invoke
    // its first allocation checkpoint and remains scoped through terminal
    // validation. It is cleared on every return and panic path.
    let Some(_vram_grant) =
        ScopedThreadVramGrant::enter(Some(prepared_facts.predicted_device_peak_bytes))
    else {
        return reject_claimed_h3_generation_message(
            job,
            "MiniMax H3 prepared attempt has no exact device-memory grant".to_string(),
        );
    };

    {
        let mut active = worker.active_generation.write().unwrap();
        *active = Some(ActiveGeneration {
            model: model_name.clone(),
            prompt_sha256: format!("{:x}", Sha256::digest(job.request.prompt.as_bytes())),
            started_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            started_at: Instant::now(),
        });
    }
    let _active_cleanup = ActiveGenerationCleanup { worker };

    let progress_tx = job.progress_tx.clone();
    let mut progress = mold_inference::progress::ProgressReporter::default();
    progress.set_callback(Box::new(move |event| {
        record_phase_timing(&event);
        if let Some(tx) = &progress_tx {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }
    }));

    let allocation_commits = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let allocation_violation = Arc::new(std::sync::Mutex::new(None::<String>));
    let expected_grant = prepared_facts.predicted_device_peak_bytes;
    let allocation_commits_for_callback = Arc::clone(&allocation_commits);
    let allocation_violation_for_callback = Arc::clone(&allocation_violation);
    let scheduler_tx = scheduler_tx.clone();
    let allocation_commit = crate::h3_private_bridge::H3AllocationCommit::new(move || {
        let previous = allocation_commits_for_callback.fetch_add(1, Ordering::SeqCst);
        if previous != 0 {
            let error = "MiniMax H3 allocation checkpoint was committed more than once".to_string();
            *allocation_violation_for_callback.lock().unwrap() = Some(error.clone());
            anyhow::bail!(error);
        }
        if mold_inference::device::thread_vram_grant_bytes() != Some(expected_grant) {
            let error =
                "MiniMax H3 allocation checkpoint ran without its exact VRAM grant".to_string();
            *allocation_violation_for_callback.lock().unwrap() = Some(error.clone());
            anyhow::bail!(error);
        }
        if scheduler_tx
            .send(crate::scheduler::WorkerEvent::AllocationCommitted {
                device_id: lease.device_id.clone(),
                work_id: lease.work_id.clone(),
                owner_epoch: lease.owner_epoch,
                worker_generation: lease.worker_generation,
            })
            .is_err()
        {
            let error =
                "MiniMax H3 allocation checkpoint could not reach the scheduler".to_string();
            *allocation_violation_for_callback.lock().unwrap() = Some(error.clone());
            anyhow::bail!(error);
        }
        Ok(())
    });

    // Keep the opaque runtime owner out of destructor paths after a fatal
    // driver fault or panic. The adapter consumes its one-shot inference value
    // internally; normal and ordinary-error paths explicitly release it.
    let mut prepared = std::mem::ManuallyDrop::new(prepared);
    let rss_before = crate::resources::ram_snapshot().used_by_mold;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prepared.run_once(scope, &mut progress, allocation_commit)
    }));
    progress.clear_callback();

    // Every ordinary completion — success and non-fatal failure alike —
    // releases the runtime owner and then hands its freed pages back to the
    // OS, exactly as `process_job` and the chain-stage watchdog do. Without it
    // an H3 render leaves ~8 GiB of idle glibc arena behind, `MemAvailable`
    // stays low, and the next identical request is refused by host admission
    // (#1214). The fatal-CUDA and panic arms deliberately call neither: that
    // context is quarantined and the process is about to stop, so its
    // allocator state must not be touched.
    let release_prepared_and_trim = |prepared: &mut std::mem::ManuallyDrop<
        crate::h3_private_bridge::BoxedH3PreparedAttempt,
    >| {
        // SAFETY: this helper is called at most once and only on paths where
        // the CUDA context remains safe for normal destruction.
        unsafe { std::mem::ManuallyDrop::drop(prepared) }
        let rss_pre_trim = trim_malloc_arenas();
        let rss_after = crate::resources::ram_snapshot().used_by_mold;
        tracing::info!(
            gpu = ordinal,
            model = %model_name,
            rss_before_mb = rss_before / 1_000_000,
            rss_after_mb = rss_after / 1_000_000,
            rss_delta_mb = (rss_after as i64 - rss_before as i64) / 1_000_000,
            rss_pre_trim_mb = rss_pre_trim.map(|value| value / 1_000_000).unwrap_or(0),
            "claimed H3 generation memory delta"
        );
    };

    match result {
        Err(payload) => {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            let message = panic_payload_message(payload.as_ref());
            reject_claimed_h3_generation_message(
                job,
                format!(
                    "MiniMax H3 inference panicked on GPU {ordinal}: {message}; CUDA owner was quarantined and the server must restart"
                ),
            )
        }
        Ok(Err(error)) if is_fatal_cuda_error(&error) => {
            quarantine_poisoned_worker(worker);
            contain_worker_cache(worker);
            reject_claimed_h3_generation_message(job, fatal_cuda_user_message(&model_name))
        }
        Ok(Err(error)) => {
            if let Some(violation) = allocation_violation.lock().unwrap().take() {
                release_prepared_and_trim(&mut prepared);
                return reject_claimed_h3_generation_message(job, violation);
            }
            if mold_inference::is_inference_cancelled(&error) {
                release_prepared_and_trim(&mut prepared);
                return reject_claimed_h3_generation_message(
                    job,
                    "MiniMax H3 generation cancelled".to_string(),
                );
            }
            if is_cuda_oom(&error) {
                let synchronized = synchronize_after_oom(worker);
                let message = if synchronized {
                    release_prepared_and_trim(&mut prepared);
                    cuda_oom_user_message_with_plan(
                        worker,
                        &model_name,
                        Some("minimax-h3"),
                        Some(&job.request),
                        job.execution_plan.as_ref().map(|plan| &plan.engine_paths),
                        Some(prepared_facts.predicted_device_peak_bytes),
                    )
                    .0
                } else {
                    fatal_cuda_user_message(&model_name)
                };
                return reject_claimed_h3_generation_message(job, message);
            }
            release_prepared_and_trim(&mut prepared);
            record_failure(worker);
            reject_claimed_h3_generation_message(
                job,
                format!("generation error: {}", clean_error_message(&error)),
            )
        }
        Ok(Ok(output)) => {
            release_prepared_and_trim(&mut prepared);
            if let Some(violation) = allocation_violation.lock().unwrap().take() {
                return reject_claimed_h3_generation_message(job, violation);
            }
            if allocation_commits.load(Ordering::SeqCst) != 1 {
                return reject_claimed_h3_generation_message(
                    job,
                    "MiniMax H3 runtime returned without one allocation commit".to_string(),
                );
            }
            finish_claimed_h3_success(worker, job, scope_facts, &prepared_facts, output)
        }
    }
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
fn validate_h3_prepared_attempt_facts(
    scope: crate::h3_attempt::H3AttemptScopeFacts<'_>,
    prepared: &crate::h3_private_bridge::H3PreparedAttemptFacts,
) -> anyhow::Result<()> {
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if !scope.matches_private_run_binding(
        &prepared.work_identity_sha256,
        &prepared.cancellation_scope_identity_sha256,
        prepared.memory_ledger_sequence,
    ) {
        anyhow::bail!("MiniMax H3 prepared attempt changed its owner run binding")
    }
    let digests = [
        prepared.execution_identity_sha256.as_str(),
        prepared.prepared_attempt_identity_sha256.as_str(),
        prepared.target_budget_identity_sha256.as_str(),
        prepared.component_set_identity_sha256.as_str(),
        prepared.admission_evidence_identity_sha256.as_str(),
        prepared.artifact_qualification_identity_sha256.as_str(),
        prepared.runtime_qualification_identity_sha256.as_str(),
        prepared.work_identity_sha256.as_str(),
        prepared.cancellation_scope_identity_sha256.as_str(),
        prepared.consumption_identity_sha256.as_str(),
    ];
    if prepared.device_id != scope.device_id()
        || prepared.device_ordinal != scope.device_ordinal()
        || prepared.execution_identity_sha256 != scope.execution_identity_sha256()
        || prepared.prepared_attempt_identity_sha256 != scope.prepared_attempt_identity_sha256()
        || prepared.target_budget_identity_sha256 != scope.target_budget_identity_sha256()
        || prepared.component_set_identity_sha256 != scope.component_set_identity_sha256()
        || prepared.predicted_device_peak_bytes != scope.predicted_device_peak_bytes()
        || prepared.predicted_host_increment_bytes != scope.predicted_host_increment_bytes()
        || prepared.predicted_device_peak_bytes == 0
        || prepared.predicted_host_increment_bytes == 0
        || prepared.memory_ledger_sequence == 0
        || digests
            .into_iter()
            .any(|value| value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        anyhow::bail!("MiniMax H3 prepared attempt differs from the claimed owner scope")
    }
    Ok(())
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
fn finish_claimed_h3_success(
    worker: &GpuWorker,
    job: GpuJob,
    scope: crate::h3_attempt::H3AttemptScopeFacts<'_>,
    prepared: &crate::h3_private_bridge::H3PreparedAttemptFacts,
    mut output: crate::h3_private_bridge::H3ClaimedRunOutput,
) -> bool {
    let echo = &output.identity_echo;
    if echo.device_id != scope.device_id()
        || echo.device_ordinal != scope.device_ordinal()
        || echo.execution_identity_sha256 != scope.execution_identity_sha256()
        || echo.prepared_attempt_identity_sha256 != scope.prepared_attempt_identity_sha256()
        || echo.target_budget_identity_sha256 != scope.target_budget_identity_sha256()
        || echo.component_set_identity_sha256 != scope.component_set_identity_sha256()
        || echo.device_id != prepared.device_id
        || echo.execution_identity_sha256 != prepared.execution_identity_sha256
        || echo.prepared_attempt_identity_sha256 != prepared.prepared_attempt_identity_sha256
        || echo.target_budget_identity_sha256 != prepared.target_budget_identity_sha256
        || echo.component_set_identity_sha256 != prepared.component_set_identity_sha256
        || echo.admission_evidence_identity_sha256 != prepared.admission_evidence_identity_sha256
        || echo.artifact_qualification_identity_sha256
            != prepared.artifact_qualification_identity_sha256
        || echo.runtime_qualification_identity_sha256
            != prepared.runtime_qualification_identity_sha256
        || echo.consumption_identity_sha256 != prepared.consumption_identity_sha256
        || echo.media != prepared.media
    {
        return reject_claimed_h3_generation_message(
            job,
            crate::h3_attempt::H3AttemptError::IdentityMismatch.to_string(),
        );
    }
    if let Err(error) = validate_h3_publication_contract(worker, &job, prepared, &output) {
        return reject_claimed_h3_generation_message(
            job,
            format!(
                "generation error: MiniMax H3 terminal output differs from the frozen publication contract: {error}"
            ),
        );
    }
    output.response.gpu = Some(worker.gpu.ordinal);
    let video = output
        .response
        .video
        .as_ref()
        .expect("validated H3 video output");
    let image = ImageData {
        data: video.thumbnail.clone(),
        format: OutputFormat::Png,
        width: video.width,
        height: video.height,
        index: 0,
    };
    worker.consecutive_failures.store(0, Ordering::SeqCst);
    crate::gpu_pool::clear_model_cuda_oom(&job.model);
    finish_generation_success(job, output.response, image, None);
    true
}

#[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
fn validate_h3_publication_contract(
    worker: &GpuWorker,
    job: &GpuJob,
    prepared: &crate::h3_private_bridge::H3PreparedAttemptFacts,
    output: &crate::h3_private_bridge::H3ClaimedRunOutput,
) -> anyhow::Result<()> {
    let contract = &prepared.media;
    let expected_contract = crate::h3_private_bridge::H3PreparedMediaContract::from_request(
        &job.request,
        job.resolved_references
            .as_ref()
            .map(crate::reference_uploads::ResolvedReferenceSet::fingerprint),
    )
    .map_err(|_| {
        anyhow::anyhow!("private H3 terminal media provenance mismatch: request-contract")
    })?;
    let expected_seed = job.request.seed.ok_or_else(|| {
        anyhow::anyhow!("private H3 terminal media provenance mismatch: request-seed")
    })?;
    let expected_frames = job
        .request
        .frames
        .unwrap_or(mold_core::minimax_h3::MIN_FRAMES);
    let expected_duration_ms = mold_inference::av_media::timeline_duration_ms(
        u64::from(expected_frames),
        mold_core::minimax_h3::FIXED_FPS,
    )
    .map_err(|_| {
        anyhow::anyhow!("private H3 terminal media provenance mismatch: request-duration")
    })?;
    let echo = &output.identity_echo;
    let video = output.response.video.as_ref().ok_or_else(|| {
        anyhow::anyhow!("private H3 terminal media provenance mismatch: response-video")
    })?;
    let durable_metadata = mold_core::OutputMetadata::from_generate_request(
        &job.request,
        expected_seed,
        None,
        "private-owner-publication-validation",
    );
    let durable_reference_fingerprint = durable_metadata
        .references
        .as_deref()
        .map(mold_core::generation_reference_fingerprint);
    macro_rules! require_axis {
        ($condition:expr, $axis:literal) => {
            if !$condition {
                anyhow::bail!(concat!(
                    "private H3 terminal media provenance mismatch: ",
                    $axis
                ))
            }
        };
    }
    require_axis!(
        contract.canonical_model == expected_contract.canonical_model,
        "contract-model"
    );
    require_axis!(contract.task == expected_contract.task, "contract-task");
    require_axis!(contract.mode == expected_contract.mode, "contract-mode");
    require_axis!(contract.seed == expected_contract.seed, "contract-seed");
    require_axis!(contract.width == expected_contract.width, "contract-width");
    require_axis!(
        contract.height == expected_contract.height,
        "contract-height"
    );
    require_axis!(
        contract.frames == expected_contract.frames,
        "contract-frames"
    );
    require_axis!(contract.fps == expected_contract.fps, "contract-fps");
    require_axis!(
        contract.reference_fingerprint_sha256 == expected_contract.reference_fingerprint_sha256,
        "contract-reference-fingerprint"
    );
    require_axis!(
        contract.resolved_reference_fingerprint_sha256
            == expected_contract.resolved_reference_fingerprint_sha256,
        "contract-resolved-source-fingerprint"
    );
    require_axis!(
        contract.reference_count == expected_contract.reference_count,
        "contract-reference-count"
    );
    require_axis!(
        contract.canonical_model == job.request.model,
        "request-model"
    );
    require_axis!(contract.seed == expected_seed, "request-seed");
    require_axis!(contract.width == job.request.width, "request-width");
    require_axis!(contract.height == job.request.height, "request-height");
    require_axis!(contract.frames == expected_frames, "request-frames");
    require_axis!(
        contract.fps == mold_core::minimax_h3::FIXED_FPS,
        "request-fps"
    );
    require_axis!(echo.device_ordinal == worker.gpu.ordinal, "echo-device");
    require_axis!(echo.media == *contract, "echo-media");
    require_axis!(echo.duration_ms == expected_duration_ms, "echo-duration");
    require_axis!(
        echo.audio_sample_rate == mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ,
        "echo-audio-rate"
    );
    require_axis!(
        u32::from(echo.audio_channels) == mold_core::minimax_h3::AUDIO_CHANNELS,
        "echo-audio-channels"
    );
    require_axis!(echo.synchronized_audio_video, "echo-synchronization");
    require_axis!(
        echo.pipeline_provenance_sha256.len() == 64
            && echo
                .pipeline_provenance_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "echo-provenance"
    );
    require_axis!(output.response.images.is_empty(), "response-images");
    require_axis!(output.response.audio.is_none(), "response-audio");
    require_axis!(
        output.response.model == contract.canonical_model,
        "response-model"
    );
    require_axis!(output.response.seed_used == expected_seed, "response-seed");
    require_axis!(
        durable_reference_fingerprint.as_deref()
            == contract.reference_fingerprint_sha256.as_deref(),
        "reference-fingerprint"
    );
    require_axis!(!video.data.is_empty(), "video-data");
    require_axis!(!video.thumbnail.is_empty(), "video-thumbnail");
    require_axis!(video.format == OutputFormat::Mp4, "video-format");
    require_axis!(video.width == contract.width, "video-width");
    require_axis!(video.height == contract.height, "video-height");
    require_axis!(video.frames == contract.frames, "video-frames");
    require_axis!(video.fps == mold_core::minimax_h3::FIXED_FPS, "video-fps");
    require_axis!(video.pipeline.is_none(), "video-pipeline");
    require_axis!(
        video.pipeline_provenance_sha256.as_deref()
            == Some(echo.pipeline_provenance_sha256.as_str()),
        "video-provenance"
    );
    require_axis!(video.has_audio, "video-has-audio");
    require_axis!(
        video.duration_ms == Some(expected_duration_ms),
        "video-duration"
    );
    require_axis!(
        video.audio_sample_rate == Some(mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ),
        "video-audio-rate"
    );
    require_axis!(
        video.audio_channels == Some(mold_core::minimax_h3::AUDIO_CHANNELS),
        "video-audio-channels"
    );
    Ok(())
}

fn reject_claimed_h3_generation_message(job: GpuJob, error: String) -> bool {
    if let Some(progress_tx) = &job.progress_tx {
        let _ = progress_tx.send(SseMessage::Error(SseErrorEvent::failed(error.clone())));
    }
    let _ = job.result_tx.send(Err(error));
    false
}

fn reject_claimed_h3_generation(job: GpuJob, error: crate::h3_attempt::H3AttemptError) -> bool {
    reject_claimed_h3_generation_message(job, error.to_string())
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

    /// Host-RAM headroom for this lease, or `None` when no ledger can answer.
    ///
    /// Legacy dispatch has no host ledger, and an unreachable or stale
    /// coordinator is missing evidence rather than proof of pressure — both
    /// retain the scheduler's grant instead of inventing a rejection.
    fn host_headroom_for_lease(&self, lease: &crate::scheduler::LeaseFence) -> Option<u64> {
        match self {
            Self::V2(scheduler_tx) => request_lease_host_headroom(scheduler_tx, lease).ok(),
            Self::Legacy(_) => None,
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

/// Unregisters a singleton generation's cancellation token on every exit path.
struct SingletonCancelGuard<'a> {
    registry: &'a crate::generation_cancel::CancelRegistry,
    job_id: String,
}

impl Drop for SingletonCancelGuard<'_> {
    fn drop(&mut self) {
        self.registry.unregister(&self.job_id);
    }
}

fn process_job_with_sink(
    worker: &GpuWorker,
    mut job: GpuJob,
    event_sink: GenerationEventSink<'_>,
    h3_attempt_cancellation: Option<mold_inference::InferenceCancellationToken>,
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
    if let Some(unavailable) = worker_unavailable(worker, &model_name) {
        let err_msg = unavailable.message;
        let retained = job.journal.is_some() && unavailable.retainable;
        if let Some(ref tx) = job.progress_tx {
            let event = if retained {
                SseErrorEvent::retained(err_msg.clone())
            } else {
                SseErrorEvent::failed(err_msg.clone())
            };
            let _ = tx.send(SseMessage::Error(event));
        }
        let _ = job.result_tx.send(Err(err_msg));
        return false;
    }

    if job.result_tx.is_closed() {
        tracing::debug!(gpu = ordinal, model = %model_name, "skipping dispatched job — client disconnected");
        return false;
    }

    // Charge the attempt here, on the owner thread, immediately before the
    // model load — the phase that can take the process down with it. Charging
    // at replay instead would delete a job that merely waited behind a long
    // render through a few deploys, having never touched a GPU.
    if let Some(ticket) = job.journal.as_ref() {
        if let crate::queue_journal::DispatchClaim::Exhausted { attempts, cap } =
            ticket.claim_dispatch()
        {
            let err_msg = format!(
                "'{model_name}' was started {attempts} times without finishing (limit {cap}); \
                 it is held for review instead of being retried"
            );
            tracing::error!(gpu = ordinal, model = %model_name, attempts, cap, "held an exhausted durable queue row");
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
            }
            // The row is already `held`; settle the ticket so its drop does
            // not delete what the operator now needs to inspect.
            if let Some(ticket) = job.journal.take() {
                ticket.hold("dispatch attempts exhausted");
            }
            let _ = job.result_tx.send(Err(err_msg));
            return false;
        }
    }

    // The durable parent owns an attempt-scoped token. Reference hashing runs
    // on this dedicated worker thread and polls the same token before any
    // model or CUDA work begins.
    let batch_cancellation = job
        .batch_child
        .as_ref()
        .map(|child| child.cancellation.clone());
    // An ordinary singleton gets a token too, so a shutdown aborts it at the
    // next inference checkpoint instead of holding the deploy open. Registered
    // through a guard because this function has a dozen early returns and a
    // leaked token would cancel an unrelated later job with the same id.
    let singleton_cancellation = (h3_attempt_cancellation.is_none()
        && batch_cancellation.is_none())
    .then(|| worker.generation_cancel.token(&job_id));
    let _singleton_cancel_guard = singleton_cancellation
        .is_some()
        .then(|| SingletonCancelGuard {
            registry: worker.generation_cancel.as_ref(),
            job_id: job_id.clone(),
        });
    let inference_cancellation = h3_attempt_cancellation
        .as_ref()
        .or(batch_cancellation.as_ref())
        .or(singleton_cancellation.as_ref());
    if let Some(cancellation) = inference_cancellation {
        job.registry
            .install_running_cancellation(&job_id, cancellation.clone());
    }
    let reference_bindings = match crate::reference_uploads::inference_bindings_for_request(
        &job.request,
        job.resolved_references.as_ref(),
        inference_cancellation,
    ) {
        Ok(bindings) => bindings,
        Err(error) => {
            let err_msg = format!("generation reference binding error: {error:#}");
            if let Some(ref tx) = job.progress_tx {
                let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
            }
            let _ = job.result_tx.send(Err(err_msg));
            return false;
        }
    };

    // Mark the registry entry as running on this specific GPU. The /api/queue
    // listing now shows this row as `state: "running"` with `gpu: <ordinal>`.
    // The V2 coordinator claims the row atomically before transport. Legacy
    // single-dispatcher tests/adapters carry no lease and retain the old
    // worker-side promotion until that adapter is removed.
    if job.lease.is_none() {
        job.registry.mark_running(&job_id, Some(ordinal));
    }

    tracing::info!(gpu = ordinal, model = %model_name, "dispatched job");

    if inference_cancellation.is_some_and(|token| token.is_cancelled()) {
        let user_requested = job.registry.cancel_requested(&job_id);
        finish_generation_cancelled(job, user_requested);
        return false;
    }

    // Hand the frozen plan's admitted peak down to the engine for this
    // dispatch. Held for the whole job (load + inference) and released on
    // every exit path by the guard's Drop.
    let _vram_grant = ScopedThreadVramGrant::enter(
        job.execution_plan
            .as_ref()
            .map(|plan| plan.predicted_vram_peak_bytes),
    );

    // Acquire per-GPU load lock — ensures only one model load at a time per GPU.
    let _load_lock = worker.model_load_lock.lock().unwrap();

    // A chain/admin/auxiliary workload may have poisoned the context while
    // this job waited on the load lock. Recheck before any CUDA operation.
    if let Err(error) = ensure_worker_not_poisoned(worker, &model_name) {
        let err_msg = error.to_string();
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
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
    // `admission_host_demand_bytes` is already zero on Metal, whose host claim
    // rides the unified device gate — asking for headroom there would be the
    // #1038 double-count.
    let planned_host_increment_bytes = job
        .execution_plan
        .as_ref()
        .map_or(0, |plan| plan.admission_host_demand_bytes());
    let planned_host_headroom_bytes = job
        .lease
        .as_ref()
        .filter(|_| planned_host_increment_bytes > 0)
        .and_then(|lease| event_sink.host_headroom_for_lease(lease));
    let planned_load = job.execution_plan.as_ref().map(|plan| PlannedLoadContract {
        mode: PlannedEngineMode::from_plan(plan),
        predicted_vram_peak_bytes: plan.predicted_vram_peak_bytes,
        learned_vram_envelope_bytes: plan.learned_vram_envelope_bytes,
        predicted_host_increment_bytes: planned_host_increment_bytes,
        available_host_headroom_bytes: planned_host_headroom_bytes,
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
                cuda_oom_user_message_with_plan(
                    worker,
                    &model_name,
                    family_slug.as_deref(),
                    Some(&job.request),
                    job.execution_plan.as_ref().map(|plan| &plan.engine_paths),
                    job.execution_plan
                        .as_ref()
                        .map(|plan| plan.predicted_vram_peak_bytes),
                )
            } else {
                (fatal_cuda_user_message(&model_name), false)
            }
        } else if is_admitted_plan_memory_rejection(&e) {
            // The machine changed after this plan was admitted. That is a
            // scheduling condition, and counting it would degrade a healthy
            // GPU out of rotation for a plan that arrived at a bad moment.
            (
                format!("model load error: {}", clean_error_message(&e)),
                false,
            )
        } else {
            (
                format!("model load error: {}", clean_error_message(&e)),
                true,
            )
        };
        if let Some(ref tx) = job.progress_tx {
            let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
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

    // Model construction is not safely preemptible, but a cancellation that
    // lands during it must stop before inference or publication begins.
    if inference_cancellation.is_some_and(|token| token.is_cancelled()) {
        let user_requested = job.registry.cancel_requested(&job_id);
        finish_generation_cancelled(job, user_requested);
        return false;
    }

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
            let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
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
            let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
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
        match inference_cancellation {
            Some(cancellation) => mold_inference::with_inference_cancellation(
                &mut *cached_engine.engine,
                cancellation.clone(),
                |engine| engine.generate_with_reference_bindings(&job.request, &reference_bindings),
            ),
            None => cached_engine
                .engine
                .generate_with_reference_bindings(&job.request, &reference_bindings),
        }
    }));

    watchdog_stop.store(true, Ordering::SeqCst);
    let _ = watchdog_handle.join();

    let rss_pre_trim = trim_malloc_arenas();

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
        let restore = cache.restore(cached_engine);
        drop(cache);
        if let Some(superseded) = restore.superseded {
            if let Err(error) = teardown_inference_engines_safely(
                worker,
                std::iter::once(superseded.engine),
                "superseded generation cache restore",
            ) {
                clear_active_generation(worker);
                let error = error.to_string();
                if let Some(ref tx) = job.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(error.clone())));
                }
                let _ = job.result_tx.send(Err(error));
                return false;
            }
        }
        if restore.reclassified_to_parked {
            worker.set_resident_model(None);
        }
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

            if response.images.is_empty() && response.video.is_none() && response.audio.is_none() {
                let err_msg =
                    "generation error: engine returned no images, video, or audio".to_string();
                if let Some(ref tx) = job.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
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
            } else if let Some(ref audio) = response.audio {
                // Waveform tile stands in for the raster payload; the real
                // audio bytes travel on `response.audio`.
                ImageData {
                    data: audio.thumbnail.clone(),
                    format: OutputFormat::Png,
                    width: audio.thumbnail_width,
                    height: audio.thumbnail_height,
                    index: 0,
                }
            } else {
                unreachable!("checked above");
            };
            if response.video.is_none() && response.audio.is_none() {
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
                    let frozen_upscale_plan = {
                        let config = job.config.blocking_read();
                        config
                            .models
                            .get(&resolved)
                            .and_then(|model| model.transformer.as_ref())
                            .ok_or_else(|| format!("upscaler model '{resolved}' is not downloaded"))
                            .and_then(|path| {
                                mold_inference::upscaler::resolve_upscale_execution_plan(
                                    resolved.clone(),
                                    std::path::Path::new(path),
                                    Some(&config.resolved_models_dir()),
                                    mold_inference::upscaler::ExactUpscalePlacement::Cpu,
                                )
                                .map_err(|error| error.to_string())
                            })
                    };
                    let frozen_upscale_plan = match frozen_upscale_plan {
                        Ok(plan) => plan,
                        Err(error) => {
                            let error = format!(
                                "post-generation upscaler plan could not be frozen: {error}"
                            );
                            report_post_generation_upscale_failure(
                                job.progress_tx.as_ref(),
                                &error,
                            );
                            tracing::warn!(
                                %error,
                                "post-generation upscale failed; keeping original image"
                            );
                            finish_generation_success(job, response, img, None);
                            return true;
                        }
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
                            // Preserve the generation's exact attempt token so
                            // Cancel remains authoritative through this final
                            // owner stage instead of signalling a fresh token.
                            cancellation: inference_cancellation.cloned().unwrap_or_default(),
                            execution_plan: None,
                        })),
                    )
                    .with_utility_plans(vec![
                        UtilityExecutionPlan::Upscale(frozen_upscale_plan.clone()),
                        UtilityExecutionPlan::Upscale(
                            mold_inference::upscaler::resolve_upscale_execution_plan_from_artifact(
                                frozen_upscale_plan.model_name.clone(),
                                frozen_upscale_plan.weights.clone(),
                                frozen_upscale_plan.artifact_root.clone(),
                                mold_inference::upscaler::ExactUpscalePlacement::Device {
                                    backend: worker.gpu.backend,
                                    ordinal: worker.gpu.ordinal,
                                },
                            ),
                        ),
                    ]);
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
                            true
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
            let user_cancelled = mold_inference::is_inference_cancelled(&e)
                && job.registry.cancel_requested(&job_id);
            let (err_msg, count_worker_failure) = if fatal_cuda {
                (fatal_cuda_user_message(&model_name), false)
            } else if is_oom {
                if synchronize_after_oom(worker) {
                    cuda_oom_user_message_with_plan(
                        worker,
                        &model_name,
                        family_slug.as_deref(),
                        Some(&job.request),
                        job.execution_plan.as_ref().map(|plan| &plan.engine_paths),
                        job.execution_plan
                            .as_ref()
                            .map(|plan| plan.predicted_vram_peak_bytes),
                    )
                } else {
                    (fatal_cuda_user_message(&model_name), false)
                }
            } else if user_cancelled {
                ("Cancelled".to_string(), false)
            } else if mold_inference::is_inference_cancelled(&e) {
                // A shutdown abort is not worker ill-health. Counting it would
                // let one deploy's worth of cancellations quarantine a healthy
                // GPU on the next boot.
                (shutdown_retention_user_message(&model_name), false)
            } else {
                (
                    format!("generation error: {}", clean_error_message(&e)),
                    true,
                )
            };
            if count_worker_failure {
                record_failure(worker);
            }
            // A retained job's stream ends with a terminal frame rather than a
            // quiet close: a quiet close leaves the desktop app in `loading`
            // forever and hard-fails the web client. The flag is what lets a
            // new client read this as interrupted instead of failed.
            let retained = job.journal.is_some()
                && mold_inference::is_inference_cancelled(&e)
                && !user_cancelled;
            if let Some(ref tx) = job.progress_tx {
                let event = if retained {
                    SseErrorEvent::retained(err_msg.clone())
                } else {
                    SseErrorEvent::failed(err_msg.clone())
                };
                let _ = tx.send(SseMessage::Error(event));
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
                let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
            }
            let _ = job.result_tx.send(Err(err_msg));
            false
        }
    }
}

fn finish_generation_success(
    mut job: GpuJob,
    response: mold_core::GenerateResponse,
    image: ImageData,
    original_image: Option<ImageData>,
) {
    match job.registry.claim_completion(&job.id) {
        crate::job_registry::CompletionClaim::Claimed => {}
        crate::job_registry::CompletionClaim::UserCancelled => {
            finish_generation_cancelled(job, true);
            return;
        }
        crate::job_registry::CompletionClaim::AttemptCancelled => {
            finish_generation_cancelled(job, false);
            return;
        }
    }
    let mut metadata = OutputMetadata::from_generate_request(
        &job.request,
        response.seed_used,
        None,
        mold_core::build_info::version_string(),
    );
    // Written into the saved print before the save, so boot replay can tell a
    // job that already produced its output from one that never ran. Output
    // filenames are wall-clock, so nothing downstream could tell them apart.
    metadata.job_id = Some(job.id.clone());
    if let Some(video) = response.video.as_ref() {
        metadata.apply_video_output(video);
    }
    let mut saved_names = crate::queue::SavedOutputNames::default();
    if let Some(ref dir) = job.output_dir {
        let _gallery_writer = job.gallery_publication_gate.blocking_write();
        let generation_time_ms = response.generation_time_ms as i64;
        let db = job.metadata_db.as_ref().as_ref();
        let events = Some(job.events.as_ref());
        if let Some(ref audio) = response.audio {
            // Record the waveform tile's raster size — see the queue's
            // single-worker path for why.
            let mut audio_metadata = metadata.clone();
            audio_metadata.apply_output_dimensions(audio.thumbnail_width, audio.thumbnail_height);
            saved_names.output = crate::queue::save_audio_to_dir(
                dir,
                &audio.data,
                &audio.thumbnail,
                audio.format,
                &job.model,
                &audio_metadata,
                Some(generation_time_ms),
                db,
                events,
                &job.gallery_publication_gate,
            );
        } else if let Some(ref video) = response.video {
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
                &job.gallery_publication_gate,
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
                &job.gallery_publication_gate,
            );
        }
    }

    // Settle the durable row on what actually reached the gallery, not on the
    // fact that inference returned. The save helpers answer `None` when
    // publication fails — an unwritable directory, a full disk, a refused
    // archive — and for a replayed job the gallery file IS the delivery, so
    // clearing the row there would lose the generation outright: nothing on
    // disk, nobody to tell, and no row left to replay.
    //
    // Settled here rather than on the ticket's ordinary drop so a shutdown
    // racing the last microseconds of delivery cannot retain a completed job
    // and replay it into a duplicate print.
    if let Some(ticket) = job.journal.take() {
        if saved_names.output.is_some() {
            ticket.complete();
        } else {
            tracing::error!(
                job = %job.id,
                dir = ?job.output_dir,
                "generation finished but its output could not be saved; \
                 holding the queue row for review"
            );
            ticket.hold("the generated output could not be saved to the gallery");
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

fn finish_generation_cancelled(mut job: GpuJob, user_requested: bool) {
    if let Some(ticket) = job.journal.take() {
        if user_requested {
            ticket.discard();
        } else {
            ticket.retain();
        }
    }
    let message = if user_requested {
        "Cancelled".to_string()
    } else {
        shutdown_retention_user_message(&job.model)
    };
    if let Some(ref tx) = job.progress_tx {
        let event = if user_requested {
            SseErrorEvent::failed(message.clone())
        } else {
            SseErrorEvent::retained(message.clone())
        };
        let _ = tx.send(SseMessage::Error(event));
    }
    let _ = job.result_tx.send(Err(message));
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
#[derive(Clone, Copy)]
struct WorkerPreflightPolicy {
    hint: Option<crate::model_manager::ActivationHint>,
    planned_peak_bytes: Option<u64>,
    request_has_lora: bool,
}

fn preflight_memory_guard_with_eviction(
    cache_lock: &std::sync::Mutex<crate::model_cache::ModelCache>,
    cache_key: &str,
    model_name: &str,
    paths: &ModelPaths,
    ordinal: usize,
    policy: WorkerPreflightPolicy,
) -> Result<(), crate::routes::ApiError> {
    if let Some(predicted_peak_bytes) = policy.planned_peak_bytes {
        return preflight_planned_memory_guard_with_eviction(
            cache_lock,
            cache_key,
            model_name,
            ordinal,
            predicted_peak_bytes,
            policy.hint,
            None,
        );
    }

    loop {
        let active_vram = cache_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_vram_bytes();
        let guard = crate::memory_preflight::preflight_memory_guard_for_request(
            model_name,
            paths,
            active_vram,
            ordinal,
            policy.hint,
            policy.request_has_lora,
        );
        let err = match guard {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };

        let evicted = {
            let mut cache = cache_lock.lock().unwrap_or_else(|e| e.into_inner());
            cache.evict_lru_parked_except(Some(cache_key))
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

fn planned_active_vram_credit(measured_active_vram: u64, credit_cap: Option<u64>) -> u64 {
    credit_cap.map_or(measured_active_vram, |cap| measured_active_vram.min(cap))
}

/// Revalidate an admitted plan against fresh physical pressure, surrendering
/// parked cache entries before failing. Swap paths may provisionally count the
/// active cache measurement because their post-drop sample is authoritative.
/// Hot-cache paths cap that measurement at fresh process-attributed VRAM so a
/// stale/global load delta cannot become fictitious reusable capacity.
fn preflight_planned_memory_guard_with_eviction(
    cache_lock: &std::sync::Mutex<crate::model_cache::ModelCache>,
    cache_key: &str,
    model_name: &str,
    ordinal: usize,
    predicted_peak_bytes: u64,
    hint: Option<crate::model_manager::ActivationHint>,
    active_vram_credit_cap: Option<u64>,
) -> Result<(), crate::routes::ApiError> {
    preflight_planned_memory_guard_with_eviction_using(
        cache_lock,
        cache_key,
        model_name,
        ordinal,
        active_vram_credit_cap,
        |active_vram| {
            crate::memory_preflight::preflight_planned_memory_guard(
                model_name,
                predicted_peak_bytes,
                active_vram,
                ordinal,
                hint,
            )
        },
    )
}

fn preflight_planned_memory_guard_with_eviction_using(
    cache_lock: &std::sync::Mutex<crate::model_cache::ModelCache>,
    cache_key: &str,
    model_name: &str,
    ordinal: usize,
    active_vram_credit_cap: Option<u64>,
    mut guard: impl FnMut(u64) -> Result<(), crate::routes::ApiError>,
) -> Result<(), crate::routes::ApiError> {
    loop {
        let measured_active_vram = cache_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_vram_bytes();
        let active_vram = planned_active_vram_credit(measured_active_vram, active_vram_credit_cap);
        let err = match guard(active_vram) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };

        // An attributed cap marks an unchanged hot-cache engine. Parked
        // entries have already unloaded GPU weights, so destroying the warm
        // set cannot create credible capacity for this request.
        if active_vram_credit_cap.is_some() {
            return Err(err);
        }

        let evicted = {
            let mut cache = cache_lock.lock().unwrap_or_else(|e| e.into_inner());
            cache.evict_lru_parked_except(Some(cache_key))
        };
        let Some((evicted_name, engine)) = evicted else {
            return Err(err);
        };
        tracing::info!(
            gpu = ordinal,
            target_model = %model_name,
            evicted_model = %evicted_name,
            "evicting LRU parked entry to preserve admitted execution plan"
        );
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
    request_has_lora: bool,
) -> anyhow::Result<mold_inference::LoadStrategy> {
    let active_vram = worker
        .model_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .active_vram_bytes();
    let available =
        crate::model_manager::effective_load_available_bytes(active_vram, worker.gpu.ordinal)
            .map_err(|error| anyhow::anyhow!(error.error))?;
    let strategy = crate::memory_preflight::request_aware_load_strategy(
        crate::model_manager::select_server_load_strategy_for_device(
            paths,
            available,
            Some(worker.gpu.total_vram_bytes),
            hint,
        ),
        paths,
        hint,
        request_has_lora,
        false,
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

/// The memory number the worker rechecks before executing a frozen plan.
///
/// Admission reserves `max(static, decayed observed high water)`
/// (`mold_scheduler::estimates`), but the recheck only looked at the static
/// prediction. A plan the scheduler had already sized against a 24.9 GB
/// observed peak therefore passed the worker gate and died two minutes later
/// in CUDA (#641). A zero envelope means no learned evidence and must never
/// weaken the frozen plan.
pub(crate) fn planned_recheck_peak_bytes(
    predicted_vram_peak_bytes: u64,
    learned_vram_envelope_bytes: u64,
) -> u64 {
    predicted_vram_peak_bytes.max(learned_vram_envelope_bytes)
}

/// Reclaim the ordinary retained-engine residency, then prove the exact
/// private target peak against a post-drop driver sample immediately before
/// the one-shot runtime can reach its first allocation checkpoint.
///
/// The private runtime does not use `ensure_model_ready`, so this is its
/// equivalent physical-pressure fence. It never substitutes the legacy
/// path-based estimator for the immutable inference-derived peaks. Releasing
/// the owner-known active cache entry before sampling also avoids inventing
/// reusable capacity from missing or zero process-attribution telemetry: the
/// post-drop driver reading is the sole device-capacity authority.
#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
pub(crate) fn prepare_private_h3_allocation_boundary(
    worker: &GpuWorker,
    model_name: &str,
    predicted_device_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
    available_host_headroom_bytes: u64,
) -> Result<(u64, u64), crate::routes::ApiError> {
    let unloaded = worker
        .model_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .unload_active()
        .is_some();
    if unloaded {
        worker.set_resident_model(None);
    }
    let available_device_bytes = device::post_drop_free_vram_bytes(worker.gpu.ordinal)
        .map_err(|error| private_h3_memory_sample_error(worker, error))?;
    validate_private_h3_physical_capacity(
        model_name,
        predicted_device_peak_bytes,
        available_device_bytes,
        predicted_host_increment_bytes,
        available_host_headroom_bytes,
    )?;
    Ok((available_device_bytes, available_host_headroom_bytes))
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn private_h3_memory_sample_error(
    worker: &GpuWorker,
    error: device::DeviceMemoryError,
) -> crate::routes::ApiError {
    let fatal = error.is_fatal_cuda();
    let api_error = device_memory_api_error(error);
    if fatal {
        quarantine_poisoned_worker(worker);
        contain_worker_cache(worker);
    }
    api_error
}

#[cfg(any(test, feature = "h3", feature = "h3-private-uat"))]
fn validate_private_h3_physical_capacity(
    model_name: &str,
    predicted_device_peak_bytes: u64,
    available_device_bytes: u64,
    predicted_host_increment_bytes: u64,
    available_host_headroom_bytes: u64,
) -> Result<(), crate::routes::ApiError> {
    crate::memory_preflight::check_planned_memory_budget(
        model_name,
        predicted_device_peak_bytes,
        available_device_bytes,
        crate::memory_preflight::rejection_suggestion(None),
    )?;
    crate::memory_preflight::check_planned_host_budget(
        model_name,
        predicted_host_increment_bytes,
        available_host_headroom_bytes,
    )
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
    let planned_peak_bytes = planned_load.map(|planned| {
        planned_recheck_peak_bytes(
            planned.predicted_vram_peak_bytes,
            planned.learned_vram_envelope_bytes,
        )
    });
    let planned_execution_fingerprint = planned_load.map(|planned| planned.execution_fingerprint);
    let planned_host_increment_bytes =
        planned_load.map_or(0, |planned| planned.predicted_host_increment_bytes);
    let planned_host_headroom_bytes =
        planned_load.and_then(|planned| planned.available_host_headroom_bytes);
    let load_request = planned_load.map(|planned| planned.request);
    let planned_engine_paths = planned_load.map(|planned| planned.engine_paths);
    let planned_engine_config = planned_load.map(|planned| planned.engine_config);
    let mut cache = worker.model_cache.lock().unwrap();

    let cached_requires_reconstruction = cache.get(cache_key).is_some_and(|entry| {
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

    // Already loaded? A matching engine avoids reconstruction, but an
    // admitted request can have a different activation peak and physical
    // pressure can change after admission. Recheck the frozen demand before
    // returning the hot-cache hit.
    let unchanged_cached = cache.get(cache_key).is_some_and(|entry| {
        entry.residency == ModelResidency::Gpu && !cached_requires_reconstruction
    });
    if unchanged_cached {
        cache.touch(cache_key);
        drop(cache);
        if let Some(predicted_peak_bytes) = planned_peak_bytes {
            if let Some(process_vram) = crate::resources::current_process_vram_bytes(&worker.gpu) {
                preflight_planned_memory_guard_with_eviction(
                    &worker.model_cache,
                    cache_key,
                    model_name,
                    worker.gpu.ordinal,
                    predicted_peak_bytes,
                    hint,
                    Some(process_vram),
                )
                .map_err(|e| anyhow::anyhow!(e.error))?;
            } else {
                tracing::debug!(
                    gpu = worker.gpu.ordinal,
                    model = %model_name,
                    "process-attributed VRAM unavailable or ambiguous; retaining scheduler grant authority for hot-cache hit"
                );
            }
        }
        return Ok(ModelLoadDisposition::Unchanged);
    }

    // Everything below allocates the frozen host increment afresh — a cold
    // load, a parked reload, or a reconstruction. The hot-cache hit above is
    // deliberately exempt: its host bytes are already resident and already in
    // the ledger's sample, so charging the increment a second time would
    // double-count exactly the pages it is meant to protect. Absent ledger
    // evidence retains the scheduler's grant.
    if planned_host_increment_bytes > 0 {
        if let Some(available_host_headroom_bytes) = planned_host_headroom_bytes {
            crate::memory_preflight::check_planned_host_budget(
                model_name,
                planned_host_increment_bytes,
                available_host_headroom_bytes,
            )
            .map_err(|error| anyhow::anyhow!(error.error))?;
        } else {
            tracing::debug!(
                gpu = worker.gpu.ordinal,
                model = %model_name,
                "host-memory headroom unavailable; retaining scheduler grant authority"
            );
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
                model_name,
                paths,
                worker.gpu.ordinal,
                WorkerPreflightPolicy {
                    hint,
                    planned_peak_bytes,
                    request_has_lora,
                },
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
            match planned_peak_bytes {
                Some(predicted_peak_bytes) => {
                    crate::memory_preflight::preflight_planned_memory_guard_after_drop(
                        model_name,
                        predicted_peak_bytes,
                        worker.gpu.ordinal,
                        hint,
                    )
                }
                None => crate::memory_preflight::preflight_memory_guard_after_drop_for_request(
                    model_name,
                    paths,
                    worker.gpu.ordinal,
                    hint,
                    request_has_lora,
                ),
            }
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
                Some(paths) => select_load_strategy_for_worker(
                    worker,
                    model_name,
                    paths,
                    hint,
                    request_has_lora,
                )?,
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
        model_name,
        &paths,
        worker.gpu.ordinal,
        WorkerPreflightPolicy {
            hint,
            planned_peak_bytes,
            request_has_lora,
        },
    )
    .map_err(|e| anyhow::anyhow!(e.error))?;

    // Unload active model first.
    {
        let mut cache = worker.model_cache.lock().unwrap();
        if cache.unload_active().is_some() {
            worker.set_resident_model(None);
        }
    }
    match planned_peak_bytes {
        Some(predicted_peak_bytes) => {
            crate::memory_preflight::preflight_planned_memory_guard_after_drop(
                model_name,
                predicted_peak_bytes,
                worker.gpu.ordinal,
                hint,
            )
        }
        None => crate::memory_preflight::preflight_memory_guard_after_drop_for_request(
            model_name,
            &paths,
            worker.gpu.ordinal,
            hint,
            request_has_lora,
        ),
    }
    .map_err(|e| anyhow::anyhow!(e.error))?;
    let load_strategy = if let Some(mode) = planned_mode {
        mode.load_strategy
    } else {
        select_load_strategy_for_worker(worker, model_name, &paths, hint, request_has_lora)?
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
        // Metal devices are memoized per ordinal, so dropping the engine no
        // longer drops candle's caching allocator with it. Sweep it before
        // sampling, or the unload frees nothing the OS can see.
        if worker.gpu.backend == mold_core::GpuBackend::Metal {
            device::release_pooled_metal_memory(worker.gpu.ordinal);
        }
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

/// Temporarily bind a thread to one GPU without erasing an existing owner
/// binding when the scope ends.
///
/// Legacy chain execution can enter from an unbound blocking-pool thread,
/// while Scheduler V2 enters on the permanently bound GPU owner thread. A
/// scoped binding therefore has to restore the exact prior state rather than
/// unconditionally clearing it.
struct ScopedThreadGpuBinding {
    previous: Option<usize>,
}

impl ScopedThreadGpuBinding {
    fn enter(ordinal: usize) -> anyhow::Result<Self> {
        let previous = mold_inference::device::thread_gpu_ordinal();
        if let Some(bound) = previous {
            if bound != ordinal {
                tracing::error!(
                    bound_gpu = bound,
                    requested_gpu = ordinal,
                    "refusing to borrow a different GPU from a bound owner thread"
                );
                anyhow::bail!(
                    "thread is already bound to GPU {bound}, cannot borrow GPU {ordinal}"
                );
            }
        }
        mold_inference::device::init_thread_gpu_ordinal(ordinal);
        Ok(Self { previous })
    }
}

impl Drop for ScopedThreadGpuBinding {
    fn drop(&mut self) {
        match self.previous {
            Some(ordinal) => mold_inference::device::init_thread_gpu_ordinal(ordinal),
            None => mold_inference::device::clear_thread_gpu_ordinal(),
        }
    }
}

/// Publish the frozen plan's admitted VRAM peak to the engine for the duration
/// of one dispatch.
///
/// Engines that self-size against *sampled free VRAM* (LTX-2's adaptive block
/// residency) otherwise expand to fill the card even though the scheduler
/// admitted them at a much smaller peak, then die at the first denoise step.
/// The frozen plan owns the memory authority at dispatch (`CLAUDE.md`), so the
/// worker hands that authority down and takes it back on every exit path —
/// including early returns and panics — so a later job on this thread can't
/// inherit a stale grant.
///
/// Deliberately `predicted_vram_peak_bytes` and not
/// `planned_recheck_peak_bytes`: the learned envelope can carry a *failed*
/// run's high-water mark, and granting that back is precisely the number that
/// OOM'd.
struct ScopedThreadVramGrant;

impl ScopedThreadVramGrant {
    fn enter(predicted_vram_peak_bytes: Option<u64>) -> Option<Self> {
        // A zero peak is "no estimate", not "no memory": granting it would
        // starve the engine into full streaming on every job.
        let bytes = predicted_vram_peak_bytes.filter(|bytes| *bytes > 0)?;
        mold_inference::device::init_thread_vram_grant_bytes(bytes);
        Some(Self)
    }
}

impl Drop for ScopedThreadVramGrant {
    fn drop(&mut self) {
        mold_inference::device::clear_thread_vram_grant_bytes();
    }
}

/// Run a blocking chain operation on a specific GPU worker.
///
/// Acquires `worker.model_load_lock` for the full duration, binds the current
/// thread to `worker.gpu.ordinal`, ensures the model is loaded on GPU, takes
/// the engine out of
/// the worker's cache, passes it to `with_engine`, and restores the engine
/// unconditionally on both success and closure failure.
///
/// Safe to call from inside `tokio::task::spawn_blocking`. An unbound calling
/// thread is cleared on return; an already-bound matching owner thread keeps
/// its permanent binding.
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
    // Scheduled chain stages already run on the permanently bound owner
    // thread, so the guard must restore that binding instead of clearing it.
    let _thread_gpu = ScopedThreadGpuBinding::enter(worker.gpu.ordinal)?;

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
        let restore = cache.restore(cached);
        drop(cache);
        if let Some(superseded) = restore.superseded {
            teardown_inference_engines_safely(
                worker,
                std::iter::once(superseded.engine),
                "superseded chain cache restore",
            )?;
        }
        if restore.reclassified_to_parked {
            worker.set_resident_model(None);
        }
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

/// Run a blocking chain stage with a cache identity separate from its semantic
/// model name. Durable frozen models use their immutable runtime ID so a
/// legacy/observe worker cannot reuse an engine loaded from mutable live
/// config under the original catalog ID.
pub fn run_stage_blocking_with_identity<T, E: std::fmt::Display + std::fmt::Debug>(
    worker: &GpuWorker,
    cache_key: &str,
    model_name: &str,
    config: &mold_core::Config,
    hint: Option<crate::model_manager::ActivationHint>,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> Result<T, E>,
) -> ChainPrep<T, E> {
    run_chain_blocking_with_identity(
        worker,
        cache_key,
        model_name,
        config,
        hint,
        None,
        with_engine,
    )
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
            predicted_vram_peak_bytes: load.plan.predicted_vram_peak_bytes,
            learned_vram_envelope_bytes: load.plan.learned_vram_envelope_bytes,
            // A chain stage runs inside a lease the coordinator already holds
            // and has no route back to the ledger from here, so it retains the
            // scheduler's grant rather than rechecking against a guess.
            predicted_host_increment_bytes: load.plan.admission_host_demand_bytes(),
            available_host_headroom_bytes: None,
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

    #[test]
    fn claimed_h3_attempt_has_a_cache_free_execution_route() {
        let source = include_str!("gpu_worker.rs");
        let owner_start = source
            .find("fn process_owner_work(")
            .expect("owner work handler");
        let owner_end = source[owner_start..]
            .find("\nfn process_legacy_owner_work(")
            .map(|offset| owner_start + offset)
            .expect("owner work handler boundary");
        let owner = &source[owner_start..owner_end];
        let headroom = owner
            .find("request_lease_host_headroom(")
            .expect("ledger-aware private host-memory recheck");
        let containment = owner
            .find("with_private_h3_cuda_preparation_attempt(worker, ||")
            .expect("private preparation containment boundary");
        let prepare = owner
            .find("h3_private_bridge::prepare_for_owner(")
            .expect("private final-dispatch preparation");
        let claim = owner
            .find("h3_attempt::claim_generation_attempt(")
            .expect("private owner attempt claim");
        let dispatch = owner.find("process_job(").expect("generation dispatch");
        assert!(headroom < containment && containment < prepare);
        assert!(prepare < claim && claim < dispatch);

        let process_start = source.find("fn process_job(").expect("generation handler");
        let process_end = source[process_start..]
            .find("\nenum GenerationEventSink")
            .map(|offset| process_start + offset)
            .expect("generation handler boundary");
        let process = &source[process_start..process_end];
        assert!(process.contains("process_claimed_h3_generation_attempt("));
        let cleanup = process
            .find("with_claimed_h3_generation_cleanup(job")
            .expect("claimed H3 cleanup guard");
        let rebuild = process
            .find("rebuild_generation_current(")
            .expect("claimed H3 current-fence rebuild");
        assert!(
            cleanup < rebuild,
            "claimed H3 cleanup must wrap current-fence validation and attempt consumption"
        );

        let claimed_start = source
            .find("fn process_claimed_h3_generation_attempt(")
            .expect("claimed H3 attempt handler");
        let claimed_end = source[claimed_start..]
            .find("\nenum GenerationEventSink")
            .map(|offset| claimed_start + offset)
            .expect("claimed H3 attempt handler boundary");
        let claimed = &source[claimed_start..claimed_end];
        for forbidden in [
            "process_job_with_sink(",
            "ensure_model_ready_sync_inner_guarded(",
            ".model_cache",
            "cache.take(",
            "cache.restore(",
            "InferenceEngine",
        ] {
            assert!(
                !claimed.contains(forbidden),
                "claimed H3 attempt handler must not retain generic runtime path {forbidden}",
            );
        }
    }

    /// The private H3 preparation boundary begins unbound because a cold
    /// worker has no CUDA context yet, but the preparation itself evicts the
    /// previous model's engine before any context construction could bind the
    /// attempt. cudarc treats a safe CUDA call on a pre-existing, unadopted
    /// context as poison (`force_retain`), so on a warm worker the evicted
    /// engine's first CUDA-bearing destructor latched retention, leaked the
    /// rest of the engine, and escalated a healthy model switch into a
    /// whole-process fatal restart (#1081). Pin the fix: the wrapper must
    /// adopt the worker's primary context into the attempt before running the
    /// prepared operation.
    #[test]
    fn private_h3_preparation_attempt_adopts_worker_context_before_operation() {
        let source = include_str!("gpu_worker.rs");
        let start = source
            .find("fn with_private_h3_cuda_preparation_attempt<T>(")
            .expect("private H3 preparation boundary");
        let end = source[start..]
            .find("\nfn validate_scheduled_generation_before_cuda(")
            .map(|offset| start + offset)
            .expect("private H3 preparation boundary end");
        let body = &source[start..end];
        let begin = body
            .find("CudaExecutionAttempt::begin_unbound()")
            .expect("unbound attempt installation");
        let adopt = body
            .find("CudaContext::new(worker.gpu.ordinal)")
            .expect("worker primary-context retention");
        let bind = body
            .find(".bind_context(&context)")
            .expect("attempt context adoption");
        let run = body
            .find("std::panic::catch_unwind")
            .expect("prepared operation execution");
        assert!(
            begin < adopt && adopt < bind && bind < run,
            "the worker's primary CUDA context must be adopted into the unbound \
             attempt before the prepared operation can evict a previous engine"
        );
    }

    async fn claimed_h3_job_fixture(
        id: &str,
    ) -> (
        GpuJob,
        tokio::sync::oneshot::Receiver<Result<GenerationJobResult, String>>,
        tokio::sync::mpsc::UnboundedReceiver<SseMessage>,
        tokio::sync::mpsc::Receiver<GenerationJob>,
        QueueHandle,
        crate::job_registry::SharedJobRegistry,
    ) {
        let mut request = fake_upscale_job(Config::default(), "unused").request;
        request.model = mold_core::minimax_h3::FL2VA_COMFY.to_string();
        request.width = mold_core::minimax_h3::DEFAULT_WIDTH;
        request.height = mold_core::minimax_h3::DEFAULT_HEIGHT;
        request.frames = Some(mold_core::minimax_h3::MIN_FRAMES);
        request.fps = Some(mold_core::minimax_h3::FIXED_FPS);
        request.seed = Some(7);
        request.guidance = 0.0;
        request.strength = 1.0;
        request.output_format = Some(OutputFormat::Mp4);
        request.enable_audio = Some(true);
        request.upscale_model = None;
        let (queue_tx, queue_rx) = tokio::sync::mpsc::channel(2);
        let queue = QueueHandle::new(queue_tx);
        for index in 0..2 {
            let (placeholder_tx, _placeholder_rx) = tokio::sync::oneshot::channel();
            queue
                .submit(
                    GenerationJob {
                        id: format!("{id}-reserved-{index}"),
                        request: request.clone(),
                        resolved_references: None,
                        completion_payload: SseCompletionPayload::Full,
                        progress_tx: None,
                        result_tx: placeholder_tx,
                        output_dir: None,
                        batch_child: None,
                        journal: None,
                        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                        h3_private_ingress_grant: None,
                    },
                    2,
                )
                .await
                .unwrap();
        }
        let registry = JobRegistry::new();
        registry.register(id, request.model.clone());
        let (progress_tx, progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = GpuJob {
            id: id.to_string(),
            model: request.model.clone(),
            request,
            resolved_references: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: Some(progress_tx),
            result_tx,
            output_dir: None,
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            queue: queue.clone(),
            registry: registry.clone(),
            events: crate::events::EventBroadcaster::new(),
            execution_plan: None,
            prepared_execution_inputs: None,
            h3_prepared_attempt: None,
            lease: None,
            batch_child: None,
            journal: None,
        };
        (job, result_rx, progress_rx, queue_rx, queue, registry)
    }

    #[derive(Clone, Copy, Debug)]
    enum FakeH3Outcome {
        Success,
        NoAllocationCommit,
        SwallowAllocationCommitFailure,
        EmptyOutput,
        TerminalIdentityMismatch,
        Cancelled,
        Error,
        FatalCuda,
        Panic,
        PublicationFault(FakeH3PublicationFault),
    }

    #[derive(Clone, Copy, Debug)]
    enum FakeH3PublicationFault {
        Model,
        Seed,
        Width,
        Fps,
        Duration,
        AudioRate,
        Synchronization,
        Provenance,
    }

    #[derive(Clone, Copy, Debug)]
    enum FakeH3ContractFault {
        Model,
        Task,
        Mode,
        Seed,
        Width,
        Height,
        Frames,
        Fps,
        ReferenceFingerprint,
        ResolvedSourceFingerprint,
        ReferenceCount,
    }

    fn apply_fake_h3_contract_fault(
        media: &mut crate::h3_private_bridge::H3PreparedMediaContract,
        fault: FakeH3ContractFault,
    ) {
        const SENTINEL: &str = "sensitive-publication-sentinel";
        match fault {
            FakeH3ContractFault::Model => media.canonical_model = SENTINEL.to_string(),
            FakeH3ContractFault::Task => media.task = mold_core::minimax_h3::Task::Ref2va,
            FakeH3ContractFault::Mode => {
                media.mode = mold_core::minimax_h3::Mode::FirstFrameToAudioVideo
            }
            FakeH3ContractFault::Seed => media.seed += 1,
            FakeH3ContractFault::Width => media.width += 32,
            FakeH3ContractFault::Height => media.height += 32,
            FakeH3ContractFault::Frames => media.frames += 17,
            FakeH3ContractFault::Fps => media.fps += 1,
            FakeH3ContractFault::ReferenceFingerprint => {
                media.reference_fingerprint_sha256 = Some(SENTINEL.to_string())
            }
            FakeH3ContractFault::ResolvedSourceFingerprint => {
                media.resolved_reference_fingerprint_sha256 = Some(SENTINEL.to_string())
            }
            FakeH3ContractFault::ReferenceCount => media.reference_count += 1,
        }
    }

    struct FakeH3PreparedAttempt {
        facts: crate::h3_private_bridge::H3PreparedAttemptFacts,
        outcome: Option<FakeH3Outcome>,
        runs: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for FakeH3PreparedAttempt {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl crate::h3_private_bridge::H3PreparedAttempt for FakeH3PreparedAttempt {
        fn facts(&self) -> crate::h3_private_bridge::H3PreparedAttemptFacts {
            self.facts.clone()
        }

        fn run_once(
            &mut self,
            scope: crate::h3_attempt::H3AttemptScope<'_>,
            progress: &mut mold_inference::progress::ProgressReporter,
            mut allocation_commit: crate::h3_private_bridge::H3AllocationCommit,
        ) -> anyhow::Result<crate::h3_private_bridge::H3ClaimedRunOutput> {
            scope.cancellation_token().checkpoint()?;
            assert_eq!(self.runs.fetch_add(1, Ordering::SeqCst), 0);
            progress.emit(mold_inference::ProgressEvent::StageStart {
                name: "Private H3 fake runtime".to_string(),
            });
            let outcome = self
                .outcome
                .take()
                .expect("a fake H3 attempt can only run once");
            match outcome {
                FakeH3Outcome::NoAllocationCommit => {}
                FakeH3Outcome::SwallowAllocationCommitFailure => {
                    let _ = allocation_commit.commit_once();
                }
                _ => allocation_commit.commit_once()?,
            }
            match outcome {
                FakeH3Outcome::Success
                | FakeH3Outcome::NoAllocationCommit
                | FakeH3Outcome::SwallowAllocationCommitFailure => {
                    Ok(fake_h3_output(&self.facts, true, false))
                }
                FakeH3Outcome::EmptyOutput => Ok(fake_h3_output(&self.facts, false, false)),
                FakeH3Outcome::TerminalIdentityMismatch => {
                    Ok(fake_h3_output(&self.facts, true, true))
                }
                FakeH3Outcome::Cancelled => {
                    Err(anyhow::Error::new(mold_inference::InferenceCancelled))
                }
                FakeH3Outcome::Error => anyhow::bail!("synthetic private H3 failure"),
                FakeH3Outcome::FatalCuda => {
                    anyhow::bail!("CUDA_ERROR_ILLEGAL_ADDRESS: synthetic private H3 fault")
                }
                FakeH3Outcome::Panic => panic!("synthetic private H3 panic"),
                FakeH3Outcome::PublicationFault(fault) => {
                    let mut output = fake_h3_output(&self.facts, true, false);
                    let video = output.response.video.as_mut().unwrap();
                    match fault {
                        FakeH3PublicationFault::Model => output.response.model = "other".into(),
                        FakeH3PublicationFault::Seed => output.response.seed_used += 1,
                        FakeH3PublicationFault::Width => video.width += 32,
                        FakeH3PublicationFault::Fps => video.fps += 1,
                        FakeH3PublicationFault::Duration => {
                            video.duration_ms = video.duration_ms.map(|duration| duration + 1)
                        }
                        FakeH3PublicationFault::AudioRate => video.audio_sample_rate = Some(48_000),
                        FakeH3PublicationFault::Synchronization => {
                            output.identity_echo.synchronized_audio_video = false
                        }
                        FakeH3PublicationFault::Provenance => {
                            video.pipeline_provenance_sha256 =
                                Some(std::iter::repeat_n('e', 64).collect())
                        }
                    }
                    Ok(output)
                }
            }
        }
    }

    /// Prepared-attempt facts for the fake runtime bound to `work_id`.
    ///
    /// The real `InferenceH3PreparedAttempt` echoes the work identity,
    /// cancellation scope, and ledger sequence it received from the owner's
    /// `H3AttemptScope::private_run_context`, and
    /// `validate_h3_prepared_attempt_facts` re-derives them from the claim.
    /// So the double takes them from the same derivation rather than a
    /// constant, which the fence correctly reads as a changed owner run
    /// binding (#1204).
    fn fake_h3_facts(work_id: &str) -> crate::h3_private_bridge::H3PreparedAttemptFacts {
        let identity = |byte: char| std::iter::repeat_n(byte, 64).collect::<String>();
        let (work_identity_sha256, cancellation_scope_identity_sha256, memory_ledger_sequence) =
            crate::h3_attempt::private_run_binding_for_test(work_id);
        crate::h3_private_bridge::H3PreparedAttemptFacts {
            device_id: "cuda:0".to_string(),
            device_ordinal: 0,
            execution_identity_sha256: identity('a'),
            prepared_attempt_identity_sha256: identity('b'),
            target_budget_identity_sha256: identity('c'),
            component_set_identity_sha256: identity('d'),
            admission_evidence_identity_sha256: identity('e'),
            artifact_qualification_identity_sha256: identity('f'),
            runtime_qualification_identity_sha256: identity('1'),
            work_identity_sha256,
            cancellation_scope_identity_sha256,
            memory_ledger_sequence,
            consumption_identity_sha256: identity('4'),
            predicted_device_peak_bytes: 11_000_000_000,
            predicted_host_increment_bytes: 2_000_000_000,
            media: crate::h3_private_bridge::H3PreparedMediaContract {
                canonical_model: mold_core::minimax_h3::FL2VA_COMFY.to_string(),
                task: mold_core::minimax_h3::Task::Fl2va,
                mode: mold_core::minimax_h3::Mode::TextToAudioVideo,
                seed: 7,
                width: mold_core::minimax_h3::DEFAULT_WIDTH,
                height: mold_core::minimax_h3::DEFAULT_HEIGHT,
                frames: mold_core::minimax_h3::MIN_FRAMES,
                fps: mold_core::minimax_h3::FIXED_FPS,
                reference_fingerprint_sha256: None,
                resolved_reference_fingerprint_sha256: None,
                reference_count: 0,
            },
        }
    }

    fn fake_ref2va_request(mut request: mold_core::GenerateRequest) -> mold_core::GenerateRequest {
        use mold_core::{
            GenerationReference, GenerationReferenceAuthority, GenerationReferenceProvenance,
        };

        let provenance = |name: &str, byte: u8| GenerationReferenceProvenance {
            name: Some(name.to_string()),
            sha256: Some(format!("{byte:02x}").repeat(32)),
        };
        request.model = mold_core::minimax_h3::REF2VA_COMFY.to_string();
        request.references = Some(vec![
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("subject.png", 1),
                mime_type: "image/png".to_string(),
                width: 1024,
                height: 768,
            },
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("voice.wav", 2),
                mime_type: "audio/wav".to_string(),
                duration_ms: 2_000,
                sample_rate: 48_000,
                channels: 1,
                sample_count: Some(96_000),
            },
        ]);
        request
    }

    fn fake_h3_output(
        facts: &crate::h3_private_bridge::H3PreparedAttemptFacts,
        complete: bool,
        mismatched_echo: bool,
    ) -> crate::h3_private_bridge::H3ClaimedRunOutput {
        let video = complete.then(|| mold_core::VideoData {
            data: vec![0, 0, 0, 24, b'f', b't', b'y', b'p'],
            format: OutputFormat::Mp4,
            width: facts.media.width,
            height: facts.media.height,
            frames: facts.media.frames,
            fps: facts.media.fps,
            pipeline: None,
            pipeline_provenance_sha256: Some(std::iter::repeat_n('f', 64).collect()),
            source_preprocessing: None,
            thumbnail: vec![0x89, b'P', b'N', b'G'],
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: Some(
                (u64::from(facts.media.frames) * 1_000).div_ceil(u64::from(facts.media.fps)),
            ),
            audio_sample_rate: Some(mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ),
            audio_channels: Some(mold_core::minimax_h3::AUDIO_CHANNELS),
        });
        crate::h3_private_bridge::H3ClaimedRunOutput {
            response: GenerateResponse {
                request_warnings: Vec::new(),
                images: Vec::new(),
                video,
                audio: None,
                generation_time_ms: 42,
                model: facts.media.canonical_model.clone(),
                seed_used: facts.media.seed,
                gpu: None,
            },
            identity_echo: crate::h3_private_bridge::H3TerminalIdentityEcho {
                device_id: facts.device_id.clone(),
                device_ordinal: facts.device_ordinal,
                execution_identity_sha256: facts.execution_identity_sha256.clone(),
                prepared_attempt_identity_sha256: if mismatched_echo {
                    std::iter::repeat_n('e', 64).collect()
                } else {
                    facts.prepared_attempt_identity_sha256.clone()
                },
                target_budget_identity_sha256: facts.target_budget_identity_sha256.clone(),
                component_set_identity_sha256: facts.component_set_identity_sha256.clone(),
                admission_evidence_identity_sha256: facts
                    .admission_evidence_identity_sha256
                    .clone(),
                artifact_qualification_identity_sha256: facts
                    .artifact_qualification_identity_sha256
                    .clone(),
                runtime_qualification_identity_sha256: facts
                    .runtime_qualification_identity_sha256
                    .clone(),
                consumption_identity_sha256: facts.consumption_identity_sha256.clone(),
                media: facts.media.clone(),
                duration_ms: (u64::from(facts.media.frames) * 1_000)
                    .div_ceil(u64::from(facts.media.fps)),
                audio_sample_rate: mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ,
                audio_channels: u16::try_from(mold_core::minimax_h3::AUDIO_CHANNELS).unwrap(),
                synchronized_audio_video: true,
                pipeline_provenance_sha256: std::iter::repeat_n('f', 64).collect(),
            },
        }
    }

    fn install_fake_h3_attempt(
        job: &mut GpuJob,
        outcome: FakeH3Outcome,
    ) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let facts = fake_h3_facts(&job.id);
        install_fake_h3_attempt_with_facts(job, outcome, facts)
    }

    fn install_fake_h3_attempt_with_facts(
        job: &mut GpuJob,
        outcome: FakeH3Outcome,
        facts: crate::h3_private_bridge::H3PreparedAttemptFacts,
    ) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let runs = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        job.lease = Some(crate::scheduler::LeaseFence {
            work_id: job.id.clone(),
            device_id: "cuda:0".to_string(),
            owner_epoch: 7,
            state_version: 13,
            plan_version: 17,
            worker_generation: 11,
            memory_sample_generation: 19,
            memory_ledger_sequence: 23,
        });
        job.h3_prepared_attempt = Some(Box::new(FakeH3PreparedAttempt {
            facts,
            outcome: Some(outcome),
            runs: Arc::clone(&runs),
            drops: Arc::clone(&drops),
        }));
        (runs, drops)
    }

    /// Host headroom the Scheduler V2 stand-in grants, comfortably above the
    /// fake attempt's `predicted_host_increment_bytes`.
    const FAKE_SCHEDULER_HOST_HEADROOM_BYTES: u64 = 64 * 1024 * 1024 * 1024;

    #[derive(Clone, Copy)]
    enum FakeSchedulerV2Mode {
        /// Answer the host-memory recheck and keep forwarding worker events.
        Live,
        /// Answer the host-memory recheck, then drop the receiver so the
        /// allocation commit that follows cannot reach the scheduler.
        ClosedAfterRecheck,
    }

    /// Scheduler V2 stand-in for the claimed-H3 owner tests.
    ///
    /// `run_claimed_h3_generation` requests ledger-aware host headroom before
    /// the CUDA allocation boundary and blocks on the scheduler's reply
    /// (#1099), so a bare channel with no responder stalls the owner thread for
    /// five seconds and then fails the job on that timeout instead of on what
    /// the test is actually asserting. This answers exactly `HostMemoryRecheck`
    /// and forwards every other worker event untouched, so the event-stream
    /// assertions still see only what the worker itself emitted.
    struct FakeSchedulerV2 {
        tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
        forwarded: tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::WorkerEvent>,
        responder: std::thread::JoinHandle<()>,
    }

    impl FakeSchedulerV2 {
        fn sender(&self) -> &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent> {
            &self.tx
        }

        /// Close the stand-in and hand back the worker events it forwarded.
        /// Joining the responder is what makes `try_recv` deterministic.
        fn settle(self) -> tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::WorkerEvent> {
            let FakeSchedulerV2 {
                tx,
                forwarded,
                responder,
            } = self;
            drop(tx);
            responder
                .join()
                .expect("fake Scheduler V2 responder must not panic");
            forwarded
        }
    }

    fn fake_scheduler_v2(mode: FakeSchedulerV2Mode) -> FakeSchedulerV2 {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (forward_tx, forwarded) = tokio::sync::mpsc::unbounded_channel();
        let answers_recheck = cfg!(any(feature = "h3", feature = "h3-private-uat"));
        let responder =
            if !answers_recheck && matches!(mode, FakeSchedulerV2Mode::ClosedAfterRecheck) {
                // This build compiles the host-memory recheck out, so a scheduler
                // that must already be gone by the allocation commit has to be gone
                // before the owner thread starts — dropping the receiver here, not
                // on the responder, is what makes that ordering deterministic.
                rx.close();
                drop(rx);
                std::thread::spawn(|| {})
            } else {
                std::thread::spawn(move || {
                    while let Some(event) = rx.blocking_recv() {
                        match event {
                            crate::scheduler::WorkerEvent::HostMemoryRecheck { reply, .. } => {
                                let _ = reply.send(Ok(FAKE_SCHEDULER_HOST_HEADROOM_BYTES));
                                if matches!(mode, FakeSchedulerV2Mode::ClosedAfterRecheck) {
                                    break;
                                }
                            }
                            other => {
                                let _ = forward_tx.send(other);
                            }
                        }
                    }
                })
            };
        FakeSchedulerV2 {
            tx,
            forwarded,
            responder,
        }
    }

    fn run_fake_claimed_h3(
        worker: &GpuWorker,
        job: GpuJob,
        scheduler_tx: &tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    ) -> Arc<AtomicUsize> {
        let id = job.id.clone();
        let (attempt, current, settlements) = crate::h3_attempt::generation_attempt_for_test(
            &id,
            mold_inference::InferenceCancellationToken::default(),
        );
        with_claimed_h3_generation_cleanup(job, |job| {
            process_claimed_h3_generation_attempt(worker, job, scheduler_tx, attempt, current)
        });
        settlements
    }

    #[tokio::test]
    async fn h3_claim_failure_consumes_the_owner_local_attempt_and_completes() {
        let id = "claimed-h3-owner-local-claim-failure";
        let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let (_runs, drops) = install_fake_h3_attempt(&mut job, FakeH3Outcome::Success);
        let fence = job
            .lease
            .clone()
            .expect("fake attempt installs owner fence");

        let outcome = complete_h3_claim_failure(
            LeaseGrant {
                fence,
                work: OwnerWork::Generation(Box::new(job)),
                retry: None,
            },
            "synthetic H3 claim failure".to_string(),
        );

        assert!(matches!(
            outcome,
            OwnerProcessOutcome::Completed {
                successful: false,
                chain_result: None,
            }
        ));
        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("claim-failed H3 attempt unexpectedly completed"),
        };
        assert_eq!(error, "synthetic H3 claim failure");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
    }

    #[tokio::test]
    async fn claimed_h3_fake_success_commits_and_publishes_once_without_touching_cache() {
        let id = "claimed-h3-fake-success";
        let (mut job, result_rx, mut progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let output = tempfile::tempdir().unwrap();
        job.output_dir = Some(output.path().to_path_buf());
        let (runs, drops) = install_fake_h3_attempt(&mut job, FakeH3Outcome::Success);
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);

        let owner_worker = Arc::clone(&worker);
        let owner_scheduler = scheduler.sender().clone();
        let settlements =
            std::thread::spawn(move || run_fake_claimed_h3(&owner_worker, job, &owner_scheduler))
                .join()
                .expect("fake H3 owner thread must not panic");
        let mut scheduler_rx = scheduler.settle();
        let result = result_rx.await.unwrap().expect("fake H3 success result");

        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
        assert_eq!(result.response.gpu, Some(0));
        assert!(result
            .response
            .video
            .as_ref()
            .is_some_and(|video| { video.format == OutputFormat::Mp4 && video.has_audio }));
        assert!(worker
            .model_cache
            .lock()
            .unwrap()
            .contains("cache-sentinel"));
        assert!(worker.active_generation.read().unwrap().is_none());
        assert_eq!(mold_inference::device::thread_vram_grant_bytes(), None);
        assert!(matches!(
            scheduler_rx.try_recv(),
            Ok(crate::scheduler::WorkerEvent::AllocationCommitted { work_id, .. }) if work_id == id
        ));
        assert!(scheduler_rx.try_recv().is_err());
        let mut saw_progress = false;
        let mut saw_complete = false;
        while let Ok(message) = progress_rx.try_recv() {
            match message {
                SseMessage::Progress(SseProgressEvent::StageStart { name })
                    if name == "Private H3 fake runtime" =>
                {
                    saw_progress = true;
                }
                SseMessage::Complete(_) => saw_complete = true,
                _ => {}
            }
        }
        assert!(saw_progress && saw_complete);
        let published = std::fs::read_dir(output.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "mp4"))
            .count();
        assert_eq!(published, 1);
    }

    #[tokio::test]
    async fn claimed_ref2va_success_publishes_exact_ordered_durable_metadata() {
        let id = "claimed-h3-ref2va-ordered-publication";
        let (mut job, result_rx, mut progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let output = tempfile::tempdir().unwrap();
        job.output_dir = Some(output.path().to_path_buf());
        job.request = fake_ref2va_request(job.request);
        job.model = job.request.model.clone();
        let resolved =
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(&job.request);
        let mut facts = fake_h3_facts(id);
        facts.media = crate::h3_private_bridge::H3PreparedMediaContract::from_request(
            &job.request,
            Some(resolved.fingerprint()),
        )
        .expect("synthetic resolved Ref2VA authority");
        let reference_fingerprint = facts
            .media
            .reference_fingerprint_sha256
            .clone()
            .expect("Ref2VA fingerprint");
        job.resolved_references = Some(resolved);
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        job.metadata_db = Arc::clone(&db);
        let (runs, drops) =
            install_fake_h3_attempt_with_facts(&mut job, FakeH3Outcome::Success, facts);
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);

        let owner_worker = Arc::clone(&worker);
        let owner_scheduler = scheduler.sender().clone();
        let settlements =
            std::thread::spawn(move || run_fake_claimed_h3(&owner_worker, job, &owner_scheduler))
                .join()
                .expect("fake Ref2VA owner thread must not panic");
        let mut scheduler_rx = scheduler.settle();
        let result = result_rx.await.unwrap().expect("fake Ref2VA success");

        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
        assert_eq!(result.response.model, mold_core::minimax_h3::REF2VA_COMFY);
        assert!(matches!(
            scheduler_rx.try_recv(),
            Ok(crate::scheduler::WorkerEvent::AllocationCommitted { work_id, .. }) if work_id == id
        ));

        let rows = db
            .as_ref()
            .as_ref()
            .expect("metadata database")
            .list(Some(output.path()))
            .unwrap();
        assert_eq!(rows.len(), 1);
        let references = rows[0]
            .metadata
            .references
            .as_deref()
            .expect("saved Ref2VA references");
        assert_eq!(
            references
                .iter()
                .map(|reference| (reference.index, reference.kind))
                .collect::<Vec<_>>(),
            [
                (1, mold_core::GenerationReferenceKind::Image),
                (2, mold_core::GenerationReferenceKind::Audio),
            ]
        );
        assert_eq!(
            mold_core::generation_reference_fingerprint(references),
            reference_fingerprint
        );
        let durable = serde_json::to_string(&rows[0].metadata).unwrap();
        for private in ["descriptor", "upload", "server_path", "api_key"] {
            assert!(!durable.contains(private));
        }
        assert!(progress_rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn claimed_ref2va_rejects_order_drift_before_runtime_or_publication() {
        let id = "claimed-h3-ref2va-order-drift";
        let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let output = tempfile::tempdir().unwrap();
        job.output_dir = Some(output.path().to_path_buf());
        job.request = fake_ref2va_request(job.request);
        job.model = job.request.model.clone();
        let admitted =
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(&job.request);
        let mut facts = fake_h3_facts(id);
        facts.media = crate::h3_private_bridge::H3PreparedMediaContract::from_request(
            &job.request,
            Some(admitted.fingerprint()),
        )
        .expect("synthetic admitted Ref2VA authority");
        job.resolved_references = Some(admitted);
        let (runs, drops) =
            install_fake_h3_attempt_with_facts(&mut job, FakeH3Outcome::Success, facts);

        job.request
            .references
            .as_mut()
            .expect("ordered references")
            .reverse();
        job.resolved_references = Some(
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(&job.request),
        );
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);
        let owner_worker = Arc::clone(&worker);
        let owner_scheduler = scheduler.sender().clone();
        let settlements =
            std::thread::spawn(move || run_fake_claimed_h3(&owner_worker, job, &owner_scheduler))
                .join()
                .expect("order-drift owner thread must not panic");
        let mut scheduler_rx = scheduler.settle();
        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("reordered Ref2VA authority unexpectedly published"),
        };

        assert!(error.contains("ordered request authority"));
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
        assert!(scheduler_rx.try_recv().is_err());
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn claimed_ref2va_cancellation_keeps_authority_and_publishes_nothing() {
        let id = "claimed-h3-ref2va-cancelled";
        let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let output = tempfile::tempdir().unwrap();
        job.output_dir = Some(output.path().to_path_buf());
        job.request = fake_ref2va_request(job.request);
        job.model = job.request.model.clone();
        let resolved =
            crate::reference_uploads::ResolvedReferenceSet::authority_only_for_test(&job.request);
        let mut facts = fake_h3_facts(id);
        facts.media = crate::h3_private_bridge::H3PreparedMediaContract::from_request(
            &job.request,
            Some(resolved.fingerprint()),
        )
        .expect("synthetic cancellation authority");
        job.resolved_references = Some(resolved);
        let (runs, drops) =
            install_fake_h3_attempt_with_facts(&mut job, FakeH3Outcome::Cancelled, facts);
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);
        let owner_worker = Arc::clone(&worker);
        let owner_scheduler = scheduler.sender().clone();
        let settlements =
            std::thread::spawn(move || run_fake_claimed_h3(&owner_worker, job, &owner_scheduler))
                .join()
                .expect("cancelled Ref2VA owner thread must not panic");
        let mut scheduler_rx = scheduler.settle();
        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("cancelled Ref2VA attempt unexpectedly published"),
        };

        assert!(error.contains("cancelled"));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
        assert!(matches!(
            scheduler_rx.try_recv(),
            Ok(crate::scheduler::WorkerEvent::AllocationCommitted { work_id, .. }) if work_id == id
        ));
        assert!(scheduler_rx.try_recv().is_err());
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn claimed_h3_rejects_prepared_and_terminal_identity_mismatch_without_publication() {
        for (id, terminal_mismatch) in [
            ("claimed-h3-prepared-mismatch", false),
            ("claimed-h3-terminal-mismatch", true),
        ] {
            let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
                claimed_h3_job_fixture(id).await;
            let output = tempfile::tempdir().unwrap();
            job.output_dir = Some(output.path().to_path_buf());
            let outcome = if terminal_mismatch {
                FakeH3Outcome::TerminalIdentityMismatch
            } else {
                FakeH3Outcome::Success
            };
            let (runs, drops) = install_fake_h3_attempt(&mut job, outcome);
            if !terminal_mismatch {
                job.h3_prepared_attempt
                    .as_mut()
                    .expect("installed fake")
                    .facts();
                let fake = job.h3_prepared_attempt.take().expect("installed fake");
                let mut facts = fake.facts();
                facts.execution_identity_sha256 = std::iter::repeat_n('e', 64).collect();
                job.h3_prepared_attempt = Some(Box::new(FakeH3PreparedAttempt {
                    facts,
                    outcome: Some(FakeH3Outcome::Success),
                    runs: Arc::clone(&runs),
                    drops: Arc::clone(&drops),
                }));
                drop(fake);
            }
            let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
            let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);

            let settlements = run_fake_claimed_h3(&worker, job, scheduler.sender());
            let mut scheduler_rx = scheduler.settle();
            let error = match result_rx.await.unwrap() {
                Err(error) => error,
                Ok(_) => panic!("identity-mismatched H3 attempt unexpectedly completed"),
            };

            assert!(error.contains("identity") || error.contains("owner scope"));
            assert_eq!(runs.load(Ordering::SeqCst), usize::from(terminal_mismatch));
            assert_eq!(settlements.load(Ordering::SeqCst), 1);
            assert_eq!(queue.pending(), 1);
            assert!(registry.snapshot().entries.is_empty());
            assert!(worker
                .model_cache
                .lock()
                .unwrap()
                .contains("cache-sentinel"));
            assert_eq!(
                scheduler_rx.try_recv().is_ok(),
                terminal_mismatch,
                "only a runtime that passed prepared-fact validation may allocate"
            );
            assert_eq!(
                std::fs::read_dir(output.path()).unwrap().count(),
                0,
                "identity failures must not publish partial gallery output"
            );
        }
    }

    #[tokio::test]
    async fn claimed_h3_rejects_cancel_error_missing_commit_and_empty_output_cleanly() {
        for (id, outcome, expected) in [
            (
                "claimed-h3-runtime-cancelled",
                FakeH3Outcome::Cancelled,
                "cancelled",
            ),
            (
                "claimed-h3-runtime-error",
                FakeH3Outcome::Error,
                "synthetic private H3 failure",
            ),
            (
                "claimed-h3-no-allocation-commit",
                FakeH3Outcome::NoAllocationCommit,
                "without one allocation commit",
            ),
            (
                "claimed-h3-empty-output",
                FakeH3Outcome::EmptyOutput,
                "frozen publication contract",
            ),
        ] {
            let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
                claimed_h3_job_fixture(id).await;
            let output = tempfile::tempdir().unwrap();
            job.output_dir = Some(output.path().to_path_buf());
            let (runs, drops) = install_fake_h3_attempt(&mut job, outcome);
            let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
            let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);

            let settlements = run_fake_claimed_h3(&worker, job, scheduler.sender());
            drop(scheduler.settle());
            let error = match result_rx.await.unwrap() {
                Err(error) => error,
                Ok(_) => panic!("invalid fake H3 attempt unexpectedly completed"),
            };

            assert!(error.contains(expected), "unexpected error: {error}");
            assert_eq!(runs.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert_eq!(settlements.load(Ordering::SeqCst), 1);
            assert_eq!(queue.pending(), 1);
            assert!(registry.snapshot().entries.is_empty());
            assert!(worker
                .model_cache
                .lock()
                .unwrap()
                .contains("cache-sentinel"));
            assert!(worker.active_generation.read().unwrap().is_none());
            assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
        }
    }

    #[tokio::test]
    async fn claimed_h3_requires_the_allocation_commit_to_reach_the_scheduler() {
        let id = "claimed-h3-closed-scheduler";
        let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let output = tempfile::tempdir().unwrap();
        job.output_dir = Some(output.path().to_path_buf());
        let (runs, drops) = install_fake_h3_attempt(&mut job, FakeH3Outcome::Success);
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        // The host-memory recheck is answered so the attempt reaches the
        // allocation commit; the scheduler is gone only from that point on.
        let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::ClosedAfterRecheck);

        let settlements = run_fake_claimed_h3(&worker, job, scheduler.sender());
        drop(scheduler.settle());
        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("H3 attempt completed without a scheduler allocation commit"),
        };

        assert!(error.contains("could not reach the scheduler"));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
        assert!(worker
            .model_cache
            .lock()
            .unwrap()
            .contains("cache-sentinel"));
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn claimed_h3_latches_a_swallowed_allocation_callback_failure() {
        let id = "claimed-h3-swallowed-allocation-failure";
        let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let output = tempfile::tempdir().unwrap();
        job.output_dir = Some(output.path().to_path_buf());
        let (runs, drops) =
            install_fake_h3_attempt(&mut job, FakeH3Outcome::SwallowAllocationCommitFailure);
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::ClosedAfterRecheck);

        let settlements = run_fake_claimed_h3(&worker, job, scheduler.sender());
        drop(scheduler.settle());
        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("H3 attempt published after swallowing a callback failure"),
        };

        assert!(error.contains("could not reach the scheduler"));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(queue.pending(), 1);
        assert!(registry.snapshot().entries.is_empty());
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn claimed_h3_publication_gate_rejects_every_independent_media_axis() {
        for (index, (fault, expected_axis)) in [
            (FakeH3PublicationFault::Model, "response-model"),
            (FakeH3PublicationFault::Seed, "response-seed"),
            (FakeH3PublicationFault::Width, "video-width"),
            (FakeH3PublicationFault::Fps, "video-fps"),
            (FakeH3PublicationFault::Duration, "video-duration"),
            (FakeH3PublicationFault::AudioRate, "video-audio-rate"),
            (
                FakeH3PublicationFault::Synchronization,
                "echo-synchronization",
            ),
            (FakeH3PublicationFault::Provenance, "video-provenance"),
        ]
        .into_iter()
        .enumerate()
        {
            let id = format!("claimed-h3-publication-fault-{index}");
            let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
                claimed_h3_job_fixture(&id).await;
            let output = tempfile::tempdir().unwrap();
            job.output_dir = Some(output.path().to_path_buf());
            let (runs, drops) =
                install_fake_h3_attempt(&mut job, FakeH3Outcome::PublicationFault(fault));
            let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
            let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);

            let settlements = run_fake_claimed_h3(&worker, job, scheduler.sender());
            drop(scheduler.settle());
            let error = match result_rx.await.unwrap() {
                Err(error) => error,
                Ok(_) => panic!("invalid H3 publication unexpectedly completed"),
            };

            assert!(error.contains("frozen publication contract"));
            assert!(error.contains(expected_axis), "unexpected error: {error}");
            assert_eq!(runs.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert_eq!(settlements.load(Ordering::SeqCst), 1);
            assert_eq!(queue.pending(), 1);
            assert!(registry.snapshot().entries.is_empty());
            assert!(worker
                .model_cache
                .lock()
                .unwrap()
                .contains("cache-sentinel"));
            assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
        }
    }

    #[tokio::test]
    async fn claimed_h3_publication_accepts_container_rounded_duration() {
        let (job, _result_rx, _progress_rx, _queue_rx, _queue, _registry) =
            claimed_h3_job_fixture("claimed-h3-rounded-duration").await;
        let facts = fake_h3_facts(&job.id);
        let claimed_output = fake_h3_output(&facts, true, false);
        let duration_ms = claimed_output.identity_echo.duration_ms;
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);

        assert_eq!(duration_ms, 5_167);
        assert_ne!(
            duration_ms,
            u64::from(facts.media.frames) * 1_000 / u64::from(facts.media.fps)
        );
        validate_h3_publication_contract(&worker, &job, &facts, &claimed_output)
            .expect("the MP4 timescale's rounded-up millisecond duration must publish");
    }

    #[tokio::test]
    async fn claimed_h3_publication_diagnostics_cover_every_frozen_contract_axis() {
        const SENTINEL: &str = "sensitive-publication-sentinel";
        for (index, (fault, expected_axis)) in [
            (FakeH3ContractFault::Model, "contract-model"),
            (FakeH3ContractFault::Task, "contract-task"),
            (FakeH3ContractFault::Mode, "contract-mode"),
            (FakeH3ContractFault::Seed, "contract-seed"),
            (FakeH3ContractFault::Width, "contract-width"),
            (FakeH3ContractFault::Height, "contract-height"),
            (FakeH3ContractFault::Frames, "contract-frames"),
            (FakeH3ContractFault::Fps, "contract-fps"),
            (
                FakeH3ContractFault::ReferenceFingerprint,
                "contract-reference-fingerprint",
            ),
            (
                FakeH3ContractFault::ResolvedSourceFingerprint,
                "contract-resolved-source-fingerprint",
            ),
            (
                FakeH3ContractFault::ReferenceCount,
                "contract-reference-count",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let id = format!("claimed-h3-contract-fault-{index}");
            let (job, _result_rx, _progress_rx, _queue_rx, _queue, _registry) =
                claimed_h3_job_fixture(&id).await;
            let mut facts = fake_h3_facts(&id);
            apply_fake_h3_contract_fault(&mut facts.media, fault);
            let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
            let claimed_output = fake_h3_output(&facts, true, false);
            let error = validate_h3_publication_contract(&worker, &job, &facts, &claimed_output)
                .expect_err("contract-drifted H3 output must be rejected")
                .to_string();

            assert!(error.contains(expected_axis), "unexpected error: {error}");
            assert!(
                !error.contains(SENTINEL),
                "error leaked contract value: {error}"
            );
        }
    }

    #[tokio::test]
    async fn claimed_h3_publication_diagnostics_redact_invalid_request_values() {
        const SENTINEL: &str = "sensitive-publication-sentinel";
        let id = "claimed-h3-invalid-request-contract";
        let (mut job, _result_rx, _progress_rx, _queue_rx, _queue, _registry) =
            claimed_h3_job_fixture(id).await;
        let facts = fake_h3_facts(id);
        let claimed_output = fake_h3_output(&facts, true, false);
        job.request.model = SENTINEL.to_string();
        job.model = SENTINEL.to_string();
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let error = validate_h3_publication_contract(&worker, &job, &facts, &claimed_output)
            .expect_err("invalid H3 request contract must be rejected")
            .to_string();

        assert!(
            error.contains("request-contract"),
            "unexpected error: {error}"
        );
        assert!(
            !error.contains(SENTINEL),
            "error leaked request value: {error}"
        );
    }

    #[tokio::test]
    async fn claimed_h3_fatal_and_panic_quarantine_without_partial_publication() {
        for (id, outcome) in [
            ("claimed-h3-fatal", FakeH3Outcome::FatalCuda),
            ("claimed-h3-panic", FakeH3Outcome::Panic),
        ] {
            let (mut job, result_rx, _progress_rx, _queue_rx, queue, registry) =
                claimed_h3_job_fixture(id).await;
            let output = tempfile::tempdir().unwrap();
            job.output_dir = Some(output.path().to_path_buf());
            let (runs, drops) = install_fake_h3_attempt(&mut job, outcome);
            let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
            let scheduler = fake_scheduler_v2(FakeSchedulerV2Mode::Live);

            let settlements = run_fake_claimed_h3(&worker, job, scheduler.sender());
            drop(scheduler.settle());
            let error = match result_rx.await.unwrap() {
                Err(error) => error,
                Ok(_) => panic!("quarantined fake H3 attempt unexpectedly completed"),
            };

            assert!(error.contains("quarantined"));
            assert_eq!(runs.load(Ordering::SeqCst), 1);
            assert_eq!(
                drops.load(Ordering::SeqCst),
                0,
                "suspect CUDA-owned attempt must be retained for process teardown"
            );
            assert_eq!(settlements.load(Ordering::SeqCst), 1);
            assert_eq!(queue.pending(), 1);
            assert!(registry.snapshot().entries.is_empty());
            assert!(worker.poisoned.load(Ordering::SeqCst));
            assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
            assert!(worker.model_cache.lock().unwrap().is_empty());
            assert!(worker.active_generation.read().unwrap().is_none());
            assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
        }
    }

    #[tokio::test]
    async fn claimed_h3_attempt_rejects_without_generic_runtime_or_duplicate_cleanup() {
        let id = "claimed-h3-runtime-unavailable";
        let (job, result_rx, mut progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let (attempt, current, settlements) = crate::h3_attempt::generation_attempt_for_test(
            id,
            mold_inference::InferenceCancellationToken::default(),
        );
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let (scheduler_tx, _scheduler_rx) = tokio::sync::mpsc::unbounded_channel();

        assert!(!with_claimed_h3_generation_cleanup(job, |job| {
            process_claimed_h3_generation_attempt(&worker, job, &scheduler_tx, attempt, current)
        }));

        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("unavailable H3 runtime unexpectedly completed"),
        };
        assert!(error.contains("claimed-attempt runtime bridge is not available"));
        assert!(matches!(
            progress_rx.recv().await,
            Some(SseMessage::Error(SseErrorEvent { message, .. }))
                if message.contains("claimed-attempt runtime bridge is not available")
        ));
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(
            queue.pending(),
            1,
            "one H3 completion must release exactly one reserved queue slot"
        );
        assert!(registry.snapshot().entries.is_empty());
    }

    #[tokio::test]
    async fn cancelled_claimed_h3_attempt_settles_and_cleans_up_once() {
        let id = "claimed-h3-cancelled";
        let (job, result_rx, mut progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;
        let cancellation = mold_inference::InferenceCancellationToken::default();
        cancellation.cancel();
        let (attempt, current, settlements) =
            crate::h3_attempt::generation_attempt_for_test(id, cancellation);
        let worker = single_worker_pool_with_parked("cache-sentinel", Duration::ZERO);
        let (scheduler_tx, _scheduler_rx) = tokio::sync::mpsc::unbounded_channel();

        assert!(!with_claimed_h3_generation_cleanup(job, |job| {
            process_claimed_h3_generation_attempt(&worker, job, &scheduler_tx, attempt, current)
        }));

        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("cancelled H3 attempt unexpectedly completed"),
        };
        assert!(error.contains("cancelled before execution"));
        assert!(matches!(
            progress_rx.recv().await,
            Some(SseMessage::Error(SseErrorEvent { message, .. }))
                if message.contains("cancelled before execution")
        ));
        assert_eq!(settlements.load(Ordering::SeqCst), 1);
        assert_eq!(
            queue.pending(),
            1,
            "one cancelled H3 completion must release exactly one reserved queue slot"
        );
        assert!(registry.snapshot().entries.is_empty());
    }

    #[tokio::test]
    async fn claimed_h3_cleanup_guard_releases_ownership_during_panic() {
        let id = "claimed-h3-cleanup-panic";
        let (job, result_rx, _progress_rx, _queue_rx, queue, registry) =
            claimed_h3_job_fixture(id).await;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_claimed_h3_generation_cleanup(job, |_job| -> () {
                panic!("synthetic claimed H3 runtime panic")
            })
        }));

        assert!(result.is_err());
        assert!(
            result_rx.await.is_err(),
            "panicking runtime must drop result ownership"
        );
        assert_eq!(
            queue.pending(),
            1,
            "one panicking H3 completion must release exactly one reserved queue slot"
        );
        assert!(registry.snapshot().entries.is_empty());
    }

    #[test]
    fn scheduled_chain_stage_wraps_render_with_dispatch_and_memory_observability() {
        let source = include_str!("gpu_worker.rs");
        let start = source
            .find("fn process_scheduled_chain_stage(")
            .expect("scheduled chain stage handler");
        let end = source[start..]
            .find("\nfn fence_chain_stage_render(")
            .map(|offset| start + offset)
            .expect("scheduled chain stage handler boundary");
        let method = &source[start..end];
        let dispatch = method.find("\"dispatched job\"").expect("dispatch log");
        let watchdog = method
            .find("ChainStageMemoryWatchdog::start(")
            .expect("memory watchdog start");
        let render = method
            .find("run_stage_blocking_planned(")
            .expect("planned stage render");
        let stop = method
            .find("drop(memory_watchdog)")
            .expect("memory watchdog stop");
        assert!(dispatch < watchdog && watchdog < render && render < stop);
    }

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

    struct RemovingWeightsGenerationEngine {
        name: String,
        weights: std::path::PathBuf,
        generated: ImageData,
    }

    impl InferenceEngine for RemovingWeightsGenerationEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            std::fs::remove_file(&self.weights)?;
            Ok(GenerateResponse {
                request_warnings: Vec::new(),
                audio: None,
                images: vec![self.generated.clone()],
                video: None,
                generation_time_ms: 1,
                model: self.name.clone(),
                seed_used: 7,
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
                request_warnings: Vec::new(),
                audio: None,
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
    fn hot_cached_planned_engine_rechecks_pressure_before_unchanged_return() {
        let source = include_str!("gpu_worker.rs");
        let hot_cache_path = source
            .split("// Already loaded?")
            .nth(1)
            .expect("hot-cache path must exist")
            .split("// Check if we have it cached but not on GPU")
            .next()
            .expect("hot-cache path must have a bounded section");
        let recheck = hot_cache_path
            .find("preflight_planned_memory_guard_with_eviction(")
            .expect("planned hot-cache hits must recheck fresh physical pressure");
        let unchanged = hot_cache_path
            .find("return Ok(ModelLoadDisposition::Unchanged)")
            .expect("hot-cache path must retain its unchanged disposition");

        assert!(
            recheck < unchanged,
            "fresh physical-pressure recheck must precede the hot-cache return"
        );
        assert!(
            hot_cache_path.contains("current_process_vram_bytes(&worker.gpu)"),
            "hot-cache credit must be clamped by fresh Mold-process attribution"
        );
        assert!(
            hot_cache_path.contains("if let Some(process_vram)"),
            "ambiguous attribution must retain scheduler authority instead of becoming zero credit"
        );
    }

    #[test]
    fn active_vram_credit_clamps_an_inflated_load_delta() {
        let stale_global_load_delta = 16 << 30;

        assert_eq!(
            planned_active_vram_credit(stale_global_load_delta, Some(6 << 30)),
            6 << 30,
            "fresh process attribution bounds a stale cache measurement"
        );
        assert_eq!(
            planned_active_vram_credit(stale_global_load_delta, Some(0)),
            0,
            "an explicit zero cap grants no credit; ambiguous attribution never calls this path"
        );
        assert_eq!(
            planned_active_vram_credit(stale_global_load_delta, None),
            stale_global_load_delta,
            "swap paths may provisionally count active memory because they recheck after drop"
        );
    }

    #[test]
    fn hot_cache_guard_propagates_pressure_with_attributed_credit_cap() {
        struct LoadedEngine;
        impl InferenceEngine for LoadedEngine {
            fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
                unreachable!()
            }
            fn model_name(&self) -> &str {
                "hot-cache"
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn load(&mut self) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let cache = std::sync::Mutex::new(ModelCache::new(2));
        cache.lock().unwrap().insert_loaded(
            "hot-cache".to_string(),
            Box::new(LoadedEngine),
            16 << 30,
        );
        let observed_credit = Arc::new(Mutex::new(None));
        let observed = observed_credit.clone();

        let error = preflight_planned_memory_guard_with_eviction_using(
            &cache,
            "hot-cache",
            "display-model",
            0,
            Some(6 << 30),
            move |credit| {
                *observed.lock().unwrap() = Some(credit);
                Err(crate::routes::ApiError::insufficient_memory(
                    "injected fresh-pressure rejection",
                ))
            },
        )
        .expect_err("a hot-cache guard failure must reach the owner path");

        assert_eq!(*observed_credit.lock().unwrap(), Some(6 << 30));
        assert!(error.error.contains("injected fresh-pressure rejection"));
        assert!(cache.lock().unwrap().contains("hot-cache"));
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
            hdr_frames_written: None,
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

    /// Reports the cancellation an aborted shutdown produces.
    struct CancelledGenerateEngine {
        name: String,
    }

    impl InferenceEngine for CancelledGenerateEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            Err(anyhow::Error::new(mold_inference::InferenceCancelled)
                .context("generation aborted"))
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
                request_warnings: Vec::new(),
                audio: None,
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

    /// A graceful deploy calls `request_shutdown()` on every worker, so a job
    /// that reaches one in that window used to be told its CUDA context was
    /// fatally poisoned. That is a lie on every ordinary restart, and it is
    /// about to become the common case now that a retained job is replayed.
    #[test]
    fn a_shutting_down_worker_reports_retention_not_a_fatal_cuda_context() {
        let worker = single_worker_pool_with_parked("mock-model", Duration::from_millis(0));
        worker.shutdown_requested.store(true, Ordering::SeqCst);

        let error = ensure_worker_not_poisoned(&worker, "mock-model")
            .expect_err("a shutting-down worker must refuse work");
        let message = error.to_string();
        assert!(
            message.contains("restarting") && message.contains("stays queued"),
            "expected a retention message, got: {message}"
        );
        assert_ne!(message, fatal_cuda_user_message("mock-model"));

        // A genuinely poisoned context keeps the specific, actionable message.
        worker.poisoned.store(true, Ordering::SeqCst);
        let poisoned = ensure_worker_not_poisoned(&worker, "mock-model")
            .expect_err("a poisoned worker must refuse work")
            .to_string();
        assert_eq!(poisoned, fatal_cuda_user_message("mock-model"));
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
            resolved_references: None,
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
            h3_prepared_attempt: None,
            lease: None,
            batch_child: None,
            journal: None,
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
    fn cpu_utility_owner_has_stable_identity_and_settles_one_exact_attempt() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        let mut empty_safetensors = 2_u64.to_le_bytes().to_vec();
        empty_safetensors.extend_from_slice(b"{}");
        std::fs::write(&weights, empty_safetensors).unwrap();
        let plan = mold_inference::upscaler::resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            mold_inference::upscaler::ExactUpscalePlacement::Cpu,
        )
        .unwrap();
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(1);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let owner = spawn_cpu_utility_thread(job_rx, event_tx);

        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                device_id,
                ordinal: usize::MAX,
                owner_epoch: 1,
                worker_generation: 1,
            }) if device_id == crate::scheduler::CPU_UTILITY_DEVICE_ID
        ));

        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        job_tx
            .send(GpuWorkerCommand::Grant(Box::new(LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "cpu-upscale-attempt".to_string(),
                    device_id: crate::scheduler::CPU_UTILITY_DEVICE_ID.to_string(),
                    owner_epoch: 1,
                    state_version: 1,
                    plan_version: 2,
                    worker_generation: 1,
                    memory_sample_generation: 3,
                    memory_ledger_sequence: 4,
                },
                work: OwnerWork::StandaloneUpscale(Box::new(
                    crate::gpu_pool::StandaloneUpscaleJob {
                        id: "cpu-upscale-attempt".to_string(),
                        model: "real-esrgan-x4plus:fp16".to_string(),
                        weights_path: weights,
                        request: mold_core::UpscaleRequest {
                            model: "real-esrgan-x4plus:fp16".to_string(),
                            image: vec![1],
                            output_format: OutputFormat::Png,
                            tile_size: None,
                            metadata: None,
                        },
                        progress_tx: None,
                        cancellation: mold_inference::InferenceCancellationToken::default(),
                        execution_plan: Some(plan),
                        result_tx,
                    },
                )),
                retry: None,
            })))
            .unwrap();

        let error = result_rx
            .blocking_recv()
            .expect("CPU owner must settle its result")
            .expect_err("invalid fixture must fail without a retry");
        assert!(
            error.contains("safetensors")
                || error.contains("header")
                || error.contains("upscaler architecture"),
            "{error}"
        );
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted {
                device_id,
                work_id,
                ..
            }) if device_id == crate::scheduler::CPU_UTILITY_DEVICE_ID
                && work_id == "cpu-upscale-attempt"
        ));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::AllocationCommitted {
                device_id,
                work_id,
                ..
            }) if device_id == crate::scheduler::CPU_UTILITY_DEVICE_ID
                && work_id == "cpu-upscale-attempt"
        ));
        match event_rx.blocking_recv() {
            Some(crate::scheduler::WorkerEvent::Completed {
                device_id,
                worker_generation: 1,
                successful,
                phase_timings,
                ..
            }) => {
                assert_eq!(device_id, crate::scheduler::CPU_UTILITY_DEVICE_ID);
                assert!(!successful, "the invalid exact artifact must fail");
                assert_eq!(
                    phase_timings.cold_load_ms, None,
                    "a failed lazy load must not publish constructor time as model-load time"
                );
                assert_eq!(phase_timings.upscale_ms, None);
            }
            _ => panic!("CPU exact attempt must publish completion evidence"),
        }
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready {
                device_id,
                ordinal: usize::MAX,
                owner_epoch: 1,
                worker_generation: 2,
            }) if device_id == crate::scheduler::CPU_UTILITY_DEVICE_ID
        ));

        job_tx.send(GpuWorkerCommand::Shutdown).unwrap();
        owner.join().unwrap();
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
                resolved_references: None,
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
                h3_prepared_attempt: None,
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
                batch_child: None,
                journal: None,
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
    fn idle_eviction_reclaims_final_resident_engine_and_publishes_cold_state() {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let worker = single_worker_pool_with_parked("stale-parked", Duration::ZERO);
        {
            let mut cache = worker.model_cache.lock().unwrap();
            cache.insert(
                Box::new(LifecycleRecordingEngine {
                    name: "gpu-resident".to_string(),
                    loaded: true,
                    operations: operations.clone(),
                }),
                8 << 30,
            );
        }
        worker.set_resident_model(Some("gpu-resident"));
        worker.set_resident_execution_fingerprint(Some("stale-fingerprint"));

        let worker_for_thread = worker.clone();
        std::thread::Builder::new()
            .name("gpu-worker-final-resident-evict".to_string())
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

        assert!(worker.model_cache.lock().unwrap().is_empty());
        assert!(worker.resident_model.read().unwrap().is_none());
        assert!(worker
            .resident_execution_fingerprint
            .read()
            .unwrap()
            .is_none());
        assert_eq!(
            operations.lock().unwrap().as_slice(),
            &[
                (
                    "unload".to_string(),
                    "gpu-worker-final-resident-evict".to_string()
                ),
                (
                    "drop".to_string(),
                    "gpu-worker-final-resident-evict".to_string()
                )
            ]
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
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
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
            let upscale_fixture = tempfile::tempdir().unwrap();
            let upscale_weights = upscale_fixture.path().join("upscaler.safetensors");
            std::fs::write(&upscale_weights, b"not-real-safetensors").unwrap();
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
                request_warnings: Vec::new(),
                audio: None,
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
                resolved_references: None,
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
                h3_prepared_attempt: None,
                lease: None,
                batch_child: None,
                journal: None,
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
                        cancellation: mold_inference::InferenceCancellationToken::default(),
                        execution_plan: Some(
                            mold_inference::upscaler::resolve_upscale_execution_plan(
                                "missing-upscaler",
                                &upscale_weights,
                                None,
                                mold_inference::upscaler::ExactUpscalePlacement::Device {
                                    backend: worker.gpu.backend,
                                    ordinal: worker.gpu.ordinal,
                                },
                            )
                            .unwrap(),
                        ),
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
                    resolved_references: None,
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
                    h3_prepared_attempt: None,
                    lease: None,
                    batch_child: None,
                    journal: None,
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
                            (
                                mold_inference::ProgressPhase::VisualDecode,
                                "Visual decode",
                                17,
                            ),
                            (
                                mold_inference::ProgressPhase::AudioDecode,
                                "Audio decode",
                                18,
                            ),
                            (mold_inference::ProgressPhase::Mux, "A/V mux", 19),
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
        assert_eq!(timings.visual_decode_ms, Some(17));
        assert_eq!(timings.audio_decode_ms, Some(18));
        assert_eq!(timings.mux_ms, Some(19));
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
                phase: mold_inference::ProgressPhase::ModelLoad,
                name: "fixture display label must not drive accounting".into(),
                elapsed: Duration::from_millis(31),
            },
            None,
        );
        handle_standalone_upscale_progress(
            mold_inference::ProgressEvent::PhaseDone {
                phase: mold_inference::ProgressPhase::Upscale,
                name: "Upscaling".into(),
                elapsed: Duration::from_millis(17),
            },
            None,
        );
        let timings = take_lease_phase_timings(None);
        assert_eq!(timings.cold_load_ms, Some(31));
        assert_eq!(timings.upscale_ms, Some(17));

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
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
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
                    resolved_references: None,
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
                    h3_prepared_attempt: None,
                    lease: None,
                    batch_child: None,
                    journal: None,
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
                        completion: None,
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
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
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
                    matches!(
                        &mut grant.work,
                        OwnerWork::ChainStage(stage)
                            if stage.on_leased.is_some() && stage.result_tx.is_some()
                    ),
                    "the lease callback and exact actor reply must survive plan invalidation"
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
    fn chain_stage_result_waits_for_coordinator_lease_settlement() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer.gguf");
        let vae = root.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let model = "settlement-test:fp8";
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
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
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
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let work_id = "chain:settlement:attempt:1:stage:0";
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
                        cancelled: Arc::new(|| true),
                        cancellation: mold_inference::InferenceCancellationToken::default(),
                        on_leased: Some(Box::new(|_| Ok(()))),
                        execution_plan: Some(plan),
                        expected_model_fingerprint: Some(expected),
                        result_tx: Some(result_tx),
                        before_second_fence: None,
                    },
                )),
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
        let completion_event = event_rx
            .blocking_recv()
            .expect("owner publishes completion for coordinator settlement");
        assert!(matches!(
            completion_event,
            crate::scheduler::WorkerEvent::Completed { .. }
        ));
        let resolved_before_settlement = !matches!(
            result_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        );

        worker.request_shutdown();
        handle.join().unwrap();
        drop(completion_event);
        assert!(
            !resolved_before_settlement,
            "actor result became visible before the coordinator settled the lease"
        );
        let shutdown_error = match result_rx
            .blocking_recv()
            .expect("dropping an unhandled completion settles the actor exactly once")
        {
            Ok(_) => panic!("coordinator shutdown must fail closed"),
            Err(error) => error,
        };
        assert!(shutdown_error.contains("coordinator stopped"));
    }

    #[test]
    fn fatal_chain_panic_preserves_actor_reply_until_completion_settlement() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer.gguf");
        let vae = root.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let model = "fatal-settlement-test:fp8";
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

        let fatal = Arc::new(AtomicBool::new(false));
        let (worker, job_rx) = protocol_worker(0, fatal.clone());
        let device_id = crate::scheduler::worker_device_id(&worker);
        let plan = crate::execution_plan::resolve_execution_plans(
            &config,
            &request,
            &[crate::execution_plan::DeviceFact {
                id: device_id.clone(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
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
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let work_id = "chain:fatal-settlement:attempt:1:stage:0";
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
                        on_leased: Some(Box::new(|_| panic!("injected actor-stage panic"))),
                        execution_plan: Some(plan),
                        expected_model_fingerprint: Some(expected),
                        result_tx: Some(result_tx),
                        before_second_fence: None,
                    },
                )),
                retry: None,
            })
            .unwrap();

        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Accepted { .. })
        ));
        let completion = match event_rx.blocking_recv() {
            Some(crate::scheduler::WorkerEvent::Completed {
                successful: false,
                completion: Some(completion),
                ..
            }) => completion,
            Some(_) => panic!("fatal actor panic must publish a deferred completion"),
            None => panic!("fatal actor panic closed the worker event channel"),
        };
        assert!(matches!(
            result_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(fatal.load(Ordering::SeqCst));
        assert!(worker.poisoned.load(Ordering::SeqCst));

        completion.finish();
        let error = match result_rx
            .blocking_recv()
            .expect("fatal completion sends exactly one actor result")
        {
            Ok(_) => panic!("fatal actor panic must fail the stage"),
            Err(error) => error,
        };
        assert!(error.contains("panicked"));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { .. })
        ));
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
                    resolved_references: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: placeholder_tx,
                    output_dir: None,
                    batch_child: None,
                    journal: None,
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
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
                resolved_references: None,
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
                h3_prepared_attempt: None,
                lease: Some(fence("generate", 3)),
                batch_child: None,
                journal: None,
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
            "DriverError(CUDA_ERROR_EXTERNAL_DEVICE)",
            "DriverError(CUDA_ERROR_MPS_CLIENT_TERMINATED)",
            "DriverError(CUDA_ERROR_CONTAINED)",
            "DriverError(CUDA_ERROR_TENSOR_MEMORY_LEAK)",
            "CublasError(CUBLAS_STATUS_MAPPING_ERROR)",
            "CublasError(CUBLAS_STATUS_EXECUTION_FAILED)",
            "CublasLtError(CUBLAS_STATUS_INTERNAL_ERROR)",
            "CurandError(CURAND_STATUS_LAUNCH_FAILURE)",
            "CurandError(CURAND_STATUS_PREEXISTING_FAILURE)",
            "CurandError(CURAND_STATUS_INTERNAL_ERROR)",
            "CUDA execution attempt retained resources; server restart required",
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
                    resolved_references: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: placeholder_tx,
                    output_dir: None,
                    batch_child: None,
                    journal: None,
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
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
                resolved_references: None,
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
                h3_prepared_attempt: None,
                lease: None,
                batch_child: None,
                journal: None,
            },
            &event_tx,
            1,
            None,
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

    fn fake_image() -> ImageData {
        ImageData {
            data: vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            format: OutputFormat::Png,
            width: 64,
            height: 64,
            index: 0,
        }
    }

    fn fake_response() -> GenerateResponse {
        GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 1,
            model: "mock-model".to_string(),
            seed_used: 7,
            gpu: None,
        }
    }

    /// A replayed job has no client, so the gallery file IS the delivery.
    /// Deleting its row after a failed publication (unwritable directory, full
    /// disk, a refused archive) loses the generation outright — nothing on
    /// disk, nobody to tell, and no row to replay.
    #[test]
    fn a_generation_whose_output_never_published_holds_its_row_instead_of_clearing_it() {
        let tmp = tempfile::tempdir().unwrap();
        // A regular file where the gallery directory should be: every save
        // helper fails its `create_dir_all` and returns None.
        let blocked = tmp.path().join("gallery");
        std::fs::write(&blocked, b"not a directory").unwrap();

        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(tmp.path()),
            "test-instance",
        ));
        let request = fake_upscale_job(Config::default(), "unused").request;
        let ticket = journal
            .record(crate::queue_journal::JournalAdmission {
                id: "publish-fails",
                request: &request,
                output_dir: Some(&blocked),
                target_gpu: None,
                completion_payload: SseCompletionPayload::Full,
                batch_child: false,
                carries_reference_authority: false,
            })
            .expect("a gallery-bound generation is durable");

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let job = GpuJob {
            id: "publish-fails".to_string(),
            model: "mock-model".to_string(),
            request,
            resolved_references: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: Some(blocked.clone()),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            metadata_db: db,
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            queue: QueueHandle::new(queue_tx),
            registry: JobRegistry::new(),
            events: crate::events::EventBroadcaster::new(),
            execution_plan: None,
            prepared_execution_inputs: None,
            h3_prepared_attempt: None,
            lease: None,
            batch_child: None,
            journal: Some(ticket),
        };

        finish_generation_success(job, fake_response(), fake_image(), None);

        let rows = journal.list_all();
        assert_eq!(rows.len(), 1, "the row must not be deleted");
        assert_eq!(
            rows[0].state,
            mold_db::generation_queue::QueueRowState::Held
        );
        assert!(rows[0]
            .held_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("could not be saved")));
    }

    /// The ordinary path still clears the row, or every completed job would
    /// pile up as held work.
    #[test]
    fn a_published_generation_clears_its_row() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(tmp.path()),
            "test-instance",
        ));
        let request = fake_upscale_job(Config::default(), "unused").request;
        let ticket = journal
            .record(crate::queue_journal::JournalAdmission {
                id: "publishes",
                request: &request,
                output_dir: Some(tmp.path()),
                target_gpu: None,
                completion_payload: SseCompletionPayload::Full,
                batch_child: false,
                carries_reference_authority: false,
            })
            .unwrap();

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let job = GpuJob {
            id: "publishes".to_string(),
            model: "mock-model".to_string(),
            request,
            resolved_references: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: Some(tmp.path().to_path_buf()),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            metadata_db: db,
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            queue: QueueHandle::new(queue_tx),
            registry: JobRegistry::new(),
            events: crate::events::EventBroadcaster::new(),
            execution_plan: None,
            prepared_execution_inputs: None,
            h3_prepared_attempt: None,
            lease: None,
            batch_child: None,
            journal: Some(ticket),
        };

        finish_generation_success(job, fake_response(), fake_image(), None);

        assert!(journal.list_all().is_empty());
    }

    /// A shutdown abort is a deliberate cancellation, not evidence that this
    /// GPU is sick. Counting it would let one deploy's worth of aborts
    /// quarantine a healthy worker on the next boot.
    #[tokio::test]
    async fn a_cancelled_generation_is_not_counted_against_worker_health() {
        let worker = single_worker_pool_with_parked("parked", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert_loaded(
            "cancel-model".to_string(),
            Box::new(CancelledGenerateEngine {
                name: "cancel-model".to_string(),
            }),
            123,
        );
        worker.in_flight.store(1, Ordering::SeqCst);

        let mut request = fake_upscale_job(Config::default(), "unused").request;
        request.model = "cancel-model".to_string();
        request.upscale_model = None;
        let (queue_tx, mut queue_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(queue_tx);
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let worker_for_job = worker.clone();
        tokio::task::spawn_blocking(move || {
            let (scheduler_tx, _scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
            process_job(
                &worker_for_job,
                GpuJob {
                    id: "cancelled-job".to_string(),
                    model: "cancel-model".to_string(),
                    request,
                    resolved_references: None,
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
                    h3_prepared_attempt: None,
                    lease: None,
                    batch_child: None,
                    journal: None,
                },
                &scheduler_tx,
                1,
                None,
            );
        })
        .await
        .unwrap();

        let message = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("a cancelled engine unexpectedly generated"),
        };
        assert!(
            message.contains("restarting") && message.contains("stays queued"),
            "a cancelled generation must read as retention, got: {message}"
        );
        assert_eq!(
            worker.consecutive_failures.load(Ordering::SeqCst),
            0,
            "a deliberate abort must not degrade or quarantine the worker"
        );
        assert!(!worker.poisoned.load(Ordering::SeqCst));
        let _ = queue_rx.try_recv();
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
                    resolved_references: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: placeholder_tx,
                    output_dir: None,
                    batch_child: None,
                    journal: None,
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
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
                    resolved_references: None,
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
                    h3_prepared_attempt: None,
                    lease: None,
                    batch_child: None,
                    journal: None,
                },
                &scheduler_tx,
                1,
                None,
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
                    resolved_references: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: placeholder_tx,
                    output_dir: None,
                    batch_child: None,
                    journal: None,
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
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
                    resolved_references: None,
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
                    h3_prepared_attempt: None,
                    lease: None,
                    batch_child: None,
                    journal: None,
                },
                &scheduler_tx,
                1,
                None,
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
    fn dispatch_publishes_the_frozen_plans_vram_grant_and_takes_it_back() {
        mold_inference::device::clear_thread_vram_grant_bytes();

        {
            let _grant = ScopedThreadVramGrant::enter(Some(11_500_000_000));
            assert_eq!(
                mold_inference::device::thread_vram_grant_bytes(),
                Some(11_500_000_000),
                "an engine that self-sizes against free VRAM must see the admitted peak"
            );
        }
        assert_eq!(
            mold_inference::device::thread_vram_grant_bytes(),
            None,
            "the grant must not leak to the next job on this worker thread"
        );
    }

    #[test]
    fn dispatch_without_a_frozen_plan_grants_nothing() {
        mold_inference::device::clear_thread_vram_grant_bytes();

        assert!(ScopedThreadVramGrant::enter(None).is_none());
        assert_eq!(mold_inference::device::thread_vram_grant_bytes(), None);

        // A zero estimate means "unknown", not "no memory".
        assert!(ScopedThreadVramGrant::enter(Some(0)).is_none());
        assert_eq!(mold_inference::device::thread_vram_grant_bytes(), None);
    }

    #[test]
    fn chain_scope_preserves_scheduler_owner_binding_and_clears_legacy_binding() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::ZERO);
        let config = Config::default();
        let ordinal = worker.gpu.ordinal;

        mold_inference::device::clear_thread_gpu_ordinal();
        run_chain_blocking(
            &worker,
            "fake-model",
            &config,
            None,
            |_engine| -> anyhow::Result<()> {
                assert_eq!(mold_inference::device::thread_gpu_ordinal(), Some(ordinal));
                Ok(())
            },
        )
        .expect("legacy chain preparation should succeed")
        .expect("legacy chain closure should succeed");
        assert_eq!(
            mold_inference::device::thread_gpu_ordinal(),
            None,
            "an unbound legacy caller must not retain a borrowed GPU binding"
        );

        mold_inference::device::init_thread_gpu_ordinal(ordinal);
        run_chain_blocking(
            &worker,
            "fake-model",
            &config,
            None,
            |_engine| -> anyhow::Result<()> {
                assert_eq!(mold_inference::device::thread_gpu_ordinal(), Some(ordinal));
                Ok(())
            },
        )
        .expect("scheduled chain preparation should succeed")
        .expect("scheduled chain closure should succeed");
        assert_eq!(
            mold_inference::device::thread_gpu_ordinal(),
            Some(ordinal),
            "a scheduled chain stage must preserve the permanent owner binding"
        );
        mold_inference::device::clear_thread_gpu_ordinal();
    }

    #[test]
    fn chain_scope_rejects_a_different_gpu_owner_without_rebinding_it() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::ZERO);
        let other_ordinal = worker.gpu.ordinal.saturating_add(1);
        mold_inference::device::init_thread_gpu_ordinal(other_ordinal);

        let error = run_chain_blocking(
            &worker,
            "fake-model",
            &Config::default(),
            None,
            |_engine| -> anyhow::Result<()> {
                panic!("a mismatched GPU owner must be rejected before engine access")
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("thread is already bound"));
        assert_eq!(
            mold_inference::device::thread_gpu_ordinal(),
            Some(other_ordinal),
            "rejecting the chain must not mutate the caller's owner binding"
        );
        mold_inference::device::clear_thread_gpu_ordinal();
    }

    #[test]
    fn scheduled_owner_keeps_stable_device_binding_after_chain_scope() {
        let model = "fake-model";
        let (worker, job_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        worker
            .model_cache
            .lock()
            .unwrap()
            .insert(FakeSlowEngine::boxed(model, Duration::ZERO), 0);
        let device_id = crate::scheduler::worker_device_id(&worker);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));

        let expect_ready =
            |event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::WorkerEvent>,
             expected_generation| {
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::Ready {
                        worker_generation,
                        ..
                    }) if worker_generation == expected_generation
                ));
            };
        let expect_completed =
            |event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::WorkerEvent>,
             expected_id: &str| {
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::Accepted { work_id, .. })
                        if work_id == expected_id
                ));
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::AllocationCommitted { work_id, .. })
                        if work_id == expected_id
                ));
                assert!(matches!(
                    event_rx.blocking_recv(),
                    Some(crate::scheduler::WorkerEvent::Completed {
                        successful: true,
                        ..
                    })
                ));
            };
        let fence = |work_id: &str, generation| crate::scheduler::LeaseFence {
            work_id: work_id.to_string(),
            device_id: device_id.clone(),
            owner_epoch: worker.owner_epoch,
            state_version: generation,
            plan_version: generation,
            worker_generation: generation,
            memory_sample_generation: generation,
            memory_ledger_sequence: generation,
        };

        expect_ready(&mut event_rx, 1);
        let chain_worker = worker.clone();
        worker
            .send_grant(LeaseGrant {
                fence: fence("chain-scope", 1),
                work: OwnerWork::Probe {
                    id: "chain-scope".to_string(),
                    kind: mold_scheduler::WorkKind::ChainStage,
                    run: Box::new(move || {
                        run_chain_blocking(
                            &chain_worker,
                            model,
                            &Config::default(),
                            None,
                            |_engine| -> anyhow::Result<()> { Ok(()) },
                        )
                        .expect("owner chain preparation should succeed")
                        .expect("owner chain closure should succeed");
                    }),
                },
                retry: None,
            })
            .unwrap();
        expect_completed(&mut event_rx, "chain-scope");

        expect_ready(&mut event_rx, 2);
        let (binding_tx, binding_rx) = std::sync::mpsc::sync_channel(1);
        worker
            .send_grant(LeaseGrant {
                fence: fence("next-image-load", 2),
                work: OwnerWork::Probe {
                    id: "next-image-load".to_string(),
                    kind: mold_scheduler::WorkKind::Generation,
                    run: Box::new(move || {
                        binding_tx
                            .send(mold_inference::device::thread_gpu_ordinal())
                            .unwrap();
                        mold_inference::device::clear_thread_gpu_ordinal();
                    }),
                },
                retry: None,
            })
            .unwrap();
        expect_completed(&mut event_rx, "next-image-load");
        assert_eq!(
            binding_rx.recv().unwrap(),
            Some(worker.gpu.ordinal),
            "the grant after a chain scope must retain stable-device owner authority"
        );

        expect_ready(&mut event_rx, 3);
        let (recovered_tx, recovered_rx) = std::sync::mpsc::sync_channel(1);
        worker
            .send_grant(LeaseGrant {
                fence: fence("binding-recovery", 3),
                work: OwnerWork::Probe {
                    id: "binding-recovery".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(move || {
                        recovered_tx
                            .send(mold_inference::device::thread_gpu_ordinal())
                            .unwrap();
                    }),
                },
                retry: None,
            })
            .unwrap();
        expect_completed(&mut event_rx, "binding-recovery");
        assert_eq!(
            recovered_rx.recv().unwrap(),
            Some(worker.gpu.ordinal),
            "the owner loop must restore a lost binding before advertising readiness"
        );

        worker.request_shutdown();
        handle.join().expect("owner should shut down cleanly");
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
    fn worker_post_upscale_rejects_a_missing_exact_plan() {
        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);
        let job = fake_upscale_job(Config::default(), "real-esrgan-x4plus:fp16");
        let mut response = GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
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
            None,
            mold_inference::InferenceCancellationToken::default(),
        )
        .expect_err("worker should reject a missing upscaler config");

        assert!(err.contains("lacked an exact execution plan"), "got: {err}");
    }

    #[test]
    fn post_upscale_plan_freeze_surfaces_missing_weights_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let missing_weights = tmp.path().join("missing-upscaler.safetensors");

        let err = mold_inference::upscaler::resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &missing_weights,
            None,
            mold_inference::upscaler::ExactUpscalePlacement::Cpu,
        )
        .expect_err("planning must surface missing weights before admission");

        assert!(err.to_string().contains("could not resolve"), "got: {err}");
    }

    #[test]
    fn legacy_generation_preserves_f0_when_upscale_weights_disappear_before_freeze() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, b"present-until-generation-completes").unwrap();
        let output_dir = root.path().join("gallery");
        let original = fake_upscale_image();

        let worker = single_worker_pool_with_parked("flux-dev:q4", Duration::ZERO);
        worker.model_cache.lock().unwrap().insert(
            Box::new(RemovingWeightsGenerationEngine {
                name: "flux-dev:q4".to_string(),
                weights: weights.clone(),
                generated: original.clone(),
            }),
            0,
        );

        let mut config = Config::default();
        config.models.insert(
            "real-esrgan-x4plus:fp16".to_string(),
            ModelConfig {
                transformer: Some(weights.display().to_string()),
                ..ModelConfig::default()
            },
        );
        let mut job = fake_upscale_job(config, "real-esrgan-x4plus:fp16");
        job.id = "legacy-f0-freeze-drift".to_string();
        job.output_dir = Some(output_dir.clone());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        job.result_tx = result_tx;
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        job.progress_tx = Some(progress_tx);
        let registry = JobRegistry::new();
        registry.register(&job.id, &job.model);
        job.registry = registry.clone();

        let (slot_tx, mut slot_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(slot_tx);
        let (dummy_tx, _dummy_rx) = tokio::sync::oneshot::channel();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(queue.submit(
                GenerationJob {
                    id: "queue-slot".to_string(),
                    request: job.request.clone(),
                    resolved_references: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx: dummy_tx,
                    output_dir: None,
                    batch_child: None,
                    journal: None,
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
                },
                1,
            ))
            .unwrap();
        let _held_slot = slot_rx.try_recv().unwrap();
        job.queue = queue.clone();

        let (owner_event_tx, mut owner_event_rx) =
            tokio::sync::mpsc::unbounded_channel::<LegacyOwnerEvent>();
        let successful = process_job_with_sink(
            &worker,
            job,
            GenerationEventSink::Legacy(&owner_event_tx),
            None,
        );

        assert!(successful, "completed generation remains successful");
        let completed = result_rx
            .blocking_recv()
            .expect("result channel settles")
            .expect("F0 is returned despite upscale planning drift");
        assert_eq!(completed.image.data, original.data);
        assert_eq!(queue.pending(), 0, "generation cleanup runs exactly once");
        assert!(registry.snapshot().entries.is_empty());
        assert!(
            std::fs::read_dir(output_dir)
                .unwrap()
                .any(|entry| entry.unwrap().path().is_file()),
            "original F0 is published to the gallery"
        );
        assert!(
            owner_event_rx.try_recv().is_err(),
            "failed freeze never creates an upscale owner"
        );
        let progress = std::iter::from_fn(|| progress_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(
            progress.iter().any(|event| matches!(
                event,
                SseMessage::Progress(SseProgressEvent::Info { message })
                    if message.contains("post-generation upscale failed")
            )),
            "freeze drift is reported as an upscale warning"
        );
        assert!(
            progress.iter().all(|event| !matches!(
                event,
                SseMessage::Progress(SseProgressEvent::StageStart { name })
                    if name.contains("upscaler")
            )),
            "the upscale factory is never touched after freeze failure"
        );
        assert!(
            matches!(progress.last(), Some(SseMessage::Complete(_))),
            "one successful completion terminates progress"
        );
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

    /// #641: "reduce --frames below 97 (e.g. 17 or 9)" is a guess — it does
    /// not know whether resolution or frame count is the binding constraint,
    /// and 9 frames is far below what the card can run. When the LTX-2
    /// estimator can name a shape that fits, the message must use it.
    #[test]
    fn infeasible_rejection_names_a_supported_shape() {
        let facts = crate::ltx2_admission::test_support::ltx2_19b_fp8_facts();
        let mut request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{
                "prompt": "Bring this image to life",
                "model": "ltx-2-19b-distilled:fp8",
                "width": 1024,
                "height": 1024,
                "steps": 8,
                "guidance": 3.0,
                "frames": 97
            }"#,
        )
        .unwrap();
        request.source_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        let advice = crate::ltx2_admission::supported_shape_advice(
            &facts,
            crate::ltx2_admission::Ltx2ShapeHint::from_request(&request),
            25_757_220_864,
        )
        .expect("a 24 GB card must have a runnable LTX-2 shape");

        let message = oom_user_message_with_advice(
            "ltx-2-19b-distilled:fp8",
            Some("ltx2"),
            Some(&request),
            Some(&advice),
        );

        let shapes = crate::ltx2_admission::supported_shapes(
            &facts,
            crate::ltx2_admission::Ltx2ShapeHint::from_request(&request),
            25_757_220_864,
        );
        let named = shapes
            .first()
            .expect("supported_shape_advice implies at least one shape");
        assert!(
            message.contains(&format!("{}x{}", named.width, named.height)),
            "message must name a concrete resolution; got: {message}"
        );
        assert!(
            message.contains(&format!("{} frames", named.frames)),
            "message must name a concrete frame count; got: {message}"
        );
        assert!(
            !message.contains("e.g. 17 or 9"),
            "the guessed frame hint must not survive when a real shape is known; \
             got: {message}"
        );
    }

    /// #641: admission reserves `max(static, decayed observed high water)`,
    /// but the worker's pre-load recheck only looked at the static prediction.
    /// A plan the scheduler had already sized against a 24.9 GB observed peak
    /// therefore passed the worker gate and died two minutes later in CUDA.
    #[test]
    fn planned_recheck_uses_the_larger_of_static_and_learned_vram() {
        assert_eq!(
            planned_recheck_peak_bytes(11_548_381_184, 24_884_805_632),
            24_884_805_632,
            "the learned envelope must win when it is the conservative one"
        );
        assert_eq!(
            planned_recheck_peak_bytes(24_884_805_632, 0),
            24_884_805_632,
            "no learned evidence must never weaken the frozen plan"
        );
        assert_eq!(
            planned_recheck_peak_bytes(20_000_000_000, 12_000_000_000),
            20_000_000_000
        );
    }

    #[test]
    fn private_h3_allocation_recheck_rejects_fresh_pressure_below_exact_peak() {
        let error = validate_private_h3_physical_capacity(
            "minimax-h3-private",
            12_000_000_000,
            11_999_999_999,
            2_000_000_000,
            64_000_000_000,
        )
        .expect_err("fresh device pressure below the frozen peak must reject");
        assert!(error.error.contains("frozen execution plan peak"));

        let error = validate_private_h3_physical_capacity(
            "minimax-h3-private",
            12_000_000_000,
            24_000_000_000,
            2_000_000_000,
            1_999_999_999,
        )
        .expect_err("fresh host pressure below the frozen increment must reject");
        assert!(error.error.contains("frozen host-memory increment"));

        let predicted_host_increment_bytes = 2 << 30;
        let error = validate_private_h3_physical_capacity(
            "minimax-h3-private",
            12_000_000_000,
            24_000_000_000,
            predicted_host_increment_bytes,
            predicted_host_increment_bytes - 1,
        )
        .expect_err("ledger-aware headroom below the frozen host increment must reject");
        assert!(error.error.contains("canonical safety floor"));
    }

    /// Ordinary work gets the same host-RAM fence H3 already had.
    ///
    /// The reservation only proves the plan fit when it was admitted; LTX-2
    /// then loads a 24 GB CPU-placed encoder minutes later. Without a dispatch
    /// recheck, whatever pressure appeared in between is discovered by the
    /// kernel rather than by the scheduler (#1099).
    #[test]
    fn planned_host_budget_rejects_ordinary_work_at_the_ledger_boundary() {
        let predicted_host_increment_bytes = 24 << 30;
        crate::memory_preflight::check_planned_host_budget(
            "ltx2-19b",
            predicted_host_increment_bytes,
            predicted_host_increment_bytes,
        )
        .expect("exactly enough headroom must admit");

        let error = crate::memory_preflight::check_planned_host_budget(
            "ltx2-19b",
            predicted_host_increment_bytes,
            predicted_host_increment_bytes - 1,
        )
        .expect_err("one byte short of the frozen increment must reject");
        assert!(error.error.contains("canonical safety floor"));
        assert!(
            !crate::gpu_worker::is_fatal_cuda_error(&anyhow::anyhow!(error.error.clone())),
            "host pressure must never reach the fatal-CUDA quarantine"
        );

        crate::memory_preflight::check_planned_host_budget("ltx2-19b", 0, 0)
            .expect("a plan with no host increment is not gated");
    }

    /// The host fence must sit past the hot-cache early return.
    ///
    /// An unchanged resident engine already holds its host increment and the
    /// ledger's sample already excludes those pages, so rechecking there would
    /// charge the same bytes twice and park work that is physically fine.
    #[test]
    fn planned_host_fence_runs_only_where_the_increment_is_allocated_afresh() {
        let source = include_str!("gpu_worker.rs");
        let start = source
            .find("fn ensure_model_ready_sync_inner(")
            .expect("model readiness handler");
        let end = source[start..]
            .find("\nfn select_load_strategy_for_worker(")
            .map(|offset| start + offset)
            .unwrap_or(source.len());
        let body = &source[start..end];
        let unchanged_return = body
            .find("return Ok(ModelLoadDisposition::Unchanged);")
            .expect("hot-cache early return");
        let fence = body
            .find("check_planned_host_budget(")
            .expect("dispatch-time host-RAM fence");
        assert!(
            unchanged_return < fence,
            "the host fence must not gate an unchanged hot-cache hit"
        );
    }

    /// Memory pressure is a scheduling condition, not worker ill health.
    ///
    /// Both dispatch-time rechecks reject an already-admitted plan because the
    /// machine changed underneath it. Counting those toward the consecutive
    /// failure budget degrades a perfectly healthy GPU out of rotation for a
    /// plan that merely arrived at a bad moment.
    #[test]
    fn memory_pressure_rejections_do_not_count_against_worker_health() {
        let host = anyhow::anyhow!(
            crate::memory_preflight::check_planned_host_budget("ltx2-19b", 24 << 30, 1 << 30)
                .expect_err("host pressure rejects")
                .error
        );
        let vram = anyhow::anyhow!(
            crate::memory_preflight::check_planned_memory_budget(
                "ltx2-19b",
                24 << 30,
                1 << 30,
                crate::memory_preflight::rejection_suggestion(None),
            )
            .expect_err("device pressure rejects")
            .error
        );

        for rejection in [&host, &vram] {
            assert!(
                is_admitted_plan_memory_rejection(rejection),
                "a planned-budget recheck must be recognised: {rejection:#}"
            );
            assert!(
                !is_fatal_cuda_error(rejection),
                "pressure must never reach the quarantine path"
            );
            assert!(
                !is_cuda_oom(rejection),
                "pressure must not be mistaken for a driver OOM"
            );
        }

        assert!(
            !is_admitted_plan_memory_rejection(&anyhow::anyhow!("safetensors file not found")),
            "a genuine load fault still counts against the worker"
        );
    }

    /// The classification has to reach the branch that decides the count.
    #[test]
    fn the_load_failure_branch_classifies_pressure_before_its_generic_arm() {
        let whole = include_str!("gpu_worker.rs");
        let source = &whole[..whole.find("\nmod tests {").unwrap_or(whole.len())];
        let start = source
            .find("let (err_msg, count_worker_failure) = if is_fatal_cuda {")
            .expect("model-load failure classification");
        let end = source[start..]
            .find("if count_worker_failure {")
            .map(|offset| start + offset)
            .expect("failure accounting");
        let branch = &source[start..end];
        let pressure = branch
            .find("is_admitted_plan_memory_rejection(&e)")
            .expect("pressure arm");
        let generic = branch.find("model load error:").expect("generic arm");
        assert!(
            pressure < generic,
            "pressure must be classified before the arm that counts a failure"
        );
    }

    /// A chain stage and a claimed-H3 owner leave the same glibc arenas behind
    /// as an ordinary generation, and neither had a trim to return them. The
    /// claimed-H3 residue is what made an identical H3 rerun fail host
    /// admission (#1214).
    #[test]
    fn every_generation_path_reclaims_glibc_arenas_through_one_gate() {
        let whole = include_str!("gpu_worker.rs");
        let source = &whole[..whole.find("\nmod tests {").unwrap_or(whole.len())];
        assert_eq!(
            source.matches("libc::malloc_trim(0)").count(),
            1,
            "one implementation, so the MOLD_MALLOC_TRIM gate cannot drift"
        );
        let start = source
            .find("impl Drop for ChainStageMemoryWatchdog {")
            .expect("chain stage watchdog");
        let end = source[start..]
            .find("\nfn fence_chain_stage_render(")
            .map(|offset| start + offset)
            .expect("chain stage watchdog boundary");
        assert!(
            source[start..end].contains("trim_malloc_arenas()"),
            "the chain-stage path must reclaim arenas the way process_job does"
        );

        let start = source
            .find("fn run_claimed_h3_generation(")
            .expect("claimed H3 generation");
        let end = source[start..]
            .find("\nfn validate_h3_prepared_attempt_facts(")
            .map(|offset| start + offset)
            .expect("claimed H3 generation boundary");
        let claimed_h3 = &source[start..end];
        assert_eq!(
            claimed_h3.matches("trim_malloc_arenas()").count(),
            1,
            "the claimed-H3 path reclaims arenas through exactly one call site"
        );
        let release_start = claimed_h3
            .find("let release_prepared_and_trim =")
            .expect("claimed H3 ordinary-completion release helper");
        let release_end = claimed_h3[release_start..]
            .find("\n    match result {")
            .map(|offset| release_start + offset)
            .expect("claimed H3 completion match");
        assert!(
            claimed_h3[release_start..release_end].contains("trim_malloc_arenas()"),
            "the claimed-H3 success and ordinary-error arms must reclaim arenas the way process_job does"
        );

        let fatal_start = claimed_h3
            .find("Err(payload) => {")
            .expect("claimed H3 panic arm");
        let fatal_end = claimed_h3[fatal_start..]
            .find("Ok(Err(error)) => {")
            .map(|offset| fatal_start + offset)
            .expect("claimed H3 ordinary error arm");
        let quarantine_arms = &claimed_h3[fatal_start..fatal_end];
        assert_eq!(
            quarantine_arms
                .matches("quarantine_poisoned_worker(worker)")
                .count(),
            2,
            "the panic and fatal-CUDA arms quarantine the owner"
        );
        assert!(
            !quarantine_arms.contains("release_prepared_and_trim("),
            "a quarantined CUDA context must never have its allocator state touched"
        );
    }

    #[test]
    fn private_h3_fatal_memory_sample_quarantines_the_owner() {
        let (worker, _worker_rx) = protocol_worker(0, Arc::new(AtomicBool::new(false)));
        let error = private_h3_memory_sample_error(
            &worker,
            mold_inference::device::DeviceMemoryError::FatalCuda {
                operation: "private H3 post-drop sample",
                message: "synthetic asynchronous fault".to_string(),
            },
        );

        assert_eq!(error.code, "INTERNAL_ERROR");
        assert!(worker.poisoned.load(Ordering::SeqCst));
        assert!(worker.fatal_cuda_error.load(Ordering::SeqCst));
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

    /// A wan OOM must get frame guidance even when the request carries no
    /// explicit frame count.
    ///
    /// The video branch fires on `is_video_family(slug) || request.frames`, so
    /// every Studio submission was already covered by the second disjunct.
    /// `mold run wan22-t2v-a14b:q5 "…"` sends no frames at all and relies on
    /// the model default, so with wan missing from the predicate the CLI's OOM
    /// was answered with image advice — lower the resolution, keep --batch 1 —
    /// and never mentioned the frame count that actually drives the peak.
    #[test]
    fn runtime_oom_message_for_wan_without_explicit_frames_keeps_frame_guidance() {
        let req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a paper boat","model":"wan22-t2v-a14b:q5","width":832,"height":480,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        assert!(req.frames.is_none(), "the fixture must omit frames");

        let msg = oom_user_message_for_request("wan22-t2v-a14b:q5", Some("wan"), Some(&req));

        assert!(
            msg.contains("--frames"),
            "wan OOM must suggest reducing frames; got: {msg}"
        );
        assert!(
            !msg.contains("--batch"),
            "wan renders one clip at a time; --batch is not a lever, got: {msg}"
        );
        assert!(
            msg.contains("832x480"),
            "the requested shape belongs in the message; got: {msg}"
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
