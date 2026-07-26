use crate::model_cache::{ModelCache, ModelResidency};
use mold_core::types::{DevicePlacement, DeviceRef, GpuWorkerState, GpuWorkerStatus};
use mold_db::MetadataDb;
use mold_inference::device::DiscoveredGpu;
use mold_inference::shared_pool::SharedPool;
use mold_scheduler::WorkKind;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

const MODEL_CUDA_OOM_COOLDOWN: Duration = Duration::from_secs(60);
pub(crate) const MAX_OWNER_BYPASSES_FOR_CHAIN: u8 = 3;
pub(crate) const DRAIN_RUNNING: u8 = 0;
pub(crate) const DRAIN_REQUESTED: u8 = 1;
pub(crate) const DRAIN_COMMITTED: u8 = 2;

static LEGACY_CHAIN_TICKET: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
pub(crate) struct LegacyChainWaitQueue {
    waiters: Mutex<Vec<Weak<LegacyChainWaiter>>>,
    execution_wake_sequence: Mutex<u64>,
    execution_wake: Condvar,
    #[cfg(test)]
    owner_claim_waiters: AtomicUsize,
    #[cfg(test)]
    owner_claim_wait: Condvar,
}

#[derive(Debug, Default)]
struct LegacyChainWaitState {
    owner_bypasses: u8,
    claimed: bool,
    wake_sequence: u64,
}

#[derive(Debug)]
pub(crate) struct LegacyChainWaiter {
    ticket: usize,
    state: Mutex<LegacyChainWaitState>,
    wake: Condvar,
}

impl LegacyChainWaiter {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            ticket: LEGACY_CHAIN_TICKET.fetch_add(1, Ordering::SeqCst),
            state: Mutex::new(LegacyChainWaitState::default()),
            wake: Condvar::new(),
        })
    }

    pub(crate) fn wake_sequence(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .wake_sequence
    }

    pub(crate) fn wait_for_wake(&self, observed_sequence: u64) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.claimed || state.wake_sequence != observed_sequence {
            return;
        }
        let _ = self
            .wake
            .wait_timeout(state, Duration::from_millis(100))
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }

    fn notify(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.wake_sequence = state.wake_sequence.saturating_add(1);
        self.wake.notify_all();
    }
}

#[derive(Debug, Default)]
struct ModelCudaOomState {
    failed_ordinals: BTreeSet<usize>,
    unschedulable_until: Option<Instant>,
}

static MODEL_CUDA_OOMS: LazyLock<RwLock<HashMap<String, ModelCudaOomState>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone)]
pub(crate) struct ModelCudaOomOutcome {
    unschedulable_until: Option<Instant>,
}

impl ModelCudaOomOutcome {
    pub(crate) fn is_unschedulable(&self) -> bool {
        self.unschedulable_until
            .is_some_and(|until| Instant::now() < until)
    }
}

pub(crate) fn record_model_cuda_oom(model_name: &str, ordinal: usize) -> ModelCudaOomOutcome {
    let now = Instant::now();
    let mut states = MODEL_CUDA_OOMS.write().unwrap();
    let state = states.entry(model_name.to_string()).or_default();

    if let Some(until) = state.unschedulable_until {
        if now < until {
            return ModelCudaOomOutcome {
                unschedulable_until: Some(until),
            };
        }
        state.unschedulable_until = None;
        state.failed_ordinals.clear();
    }

    state.failed_ordinals.insert(ordinal);
    let unschedulable_until = if state.failed_ordinals.len() >= 2 {
        let until = now + MODEL_CUDA_OOM_COOLDOWN;
        state.unschedulable_until = Some(until);
        tracing::warn!(
            model = %model_name,
            failed_gpus = ?state.failed_ordinals,
            cooldown_secs = MODEL_CUDA_OOM_COOLDOWN.as_secs(),
            "model marked temporarily unschedulable after CUDA OOM on multiple GPUs"
        );
        Some(until)
    } else {
        None
    };

    ModelCudaOomOutcome {
        unschedulable_until,
    }
}

pub(crate) fn model_unschedulable_message(model_name: &str) -> Option<String> {
    let now = Instant::now();
    let mut states = MODEL_CUDA_OOMS.write().unwrap();
    let state = states.get_mut(model_name)?;
    let until = state.unschedulable_until?;
    if now >= until {
        states.remove(model_name);
        return None;
    }
    let remaining = until.saturating_duration_since(now).as_secs().max(1);
    Some(format!(
        "model '{model_name}' is temporarily unschedulable after CUDA OOM on multiple GPUs; \
         retry in {remaining}s or use a quantized/smaller variant."
    ))
}

pub(crate) fn failed_ordinals_for_model(model_name: &str) -> Vec<usize> {
    let now = Instant::now();
    let mut states = MODEL_CUDA_OOMS.write().unwrap();
    let Some(state) = states.get_mut(model_name) else {
        return Vec::new();
    };
    if let Some(until) = state.unschedulable_until {
        if now >= until {
            states.remove(model_name);
        }
        return Vec::new();
    }
    state.failed_ordinals.iter().copied().collect()
}

pub(crate) fn clear_model_cuda_oom(model_name: &str) {
    MODEL_CUDA_OOMS.write().unwrap().remove(model_name);
}

#[cfg(test)]
pub(crate) fn clear_model_cuda_ooms_for_tests() {
    MODEL_CUDA_OOMS.write().unwrap().clear();
}

/// Per-GPU worker state. Each GPU gets its own model cache, load lock, and health tracking.
pub struct GpuWorker {
    pub owner_epoch: u64,
    pub gpu: DiscoveredGpu,
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// Driver-free snapshot of the model currently resident on this worker.
    ///
    /// Status endpoints must never lock `model_cache`: an engine drop/load can
    /// block there while owning CUDA resources. Model-loading code updates this
    /// snapshot only after a successful residency transition.
    pub resident_model: Arc<RwLock<Option<String>>>,
    /// Exact resolved execution fingerprint currently resident. `None` is
    /// intentionally conservative for legacy work that did not execute a
    /// frozen plan.
    pub resident_execution_fingerprint: Arc<RwLock<Option<String>>>,
    pub active_generation: Arc<RwLock<Option<ActiveGeneration>>>,
    pub model_load_lock: Arc<Mutex<()>>,
    pub shared_pool: Arc<Mutex<SharedPool>>,
    /// Rollback-mode commands accepted by the depth-two transport. This is
    /// deliberately separate from `in_flight`, which remains the binary
    /// execution exclusion fence in every dispatch mode.
    pub legacy_pending: AtomicUsize,
    pub in_flight: AtomicUsize,
    pub(crate) legacy_chain_waiters: LegacyChainWaitQueue,
    pub consecutive_failures: AtomicUsize,
    /// Fatal CUDA errors poison a process-owned context permanently. A poisoned
    /// worker is quarantined until the server process is restarted.
    pub poisoned: AtomicBool,
    /// Shared process-level signal. A fatal context requires process teardown;
    /// supervised servers restart after `run_server` returns an error.
    pub fatal_cuda_error: Arc<AtomicBool>,
    pub fatal_cuda_shutdown: Arc<tokio::sync::Notify>,
    /// Explicit owner-thread shutdown. The command wakes an idle blocking
    /// receiver; the flag also fences a grant that was already transported.
    pub shutdown_requested: AtomicBool,
    pub drain_state: AtomicU8,
    /// Exact OS thread authorized to create/use/drop device-backed objects.
    pub owner_thread_id: std::sync::OnceLock<std::thread::ThreadId>,
    pub degraded_until: RwLock<Option<Instant>>,
    pub job_tx: std::sync::mpsc::SyncSender<GpuWorkerCommand>,
}

/// Tracks the currently active generation on a GPU worker.
#[derive(Debug)]
pub struct ActiveGeneration {
    pub model: String,
    pub prompt_sha256: String,
    pub started_at_unix_ms: u64,
    pub started_at: Instant,
}

/// A job dispatched to a GPU worker thread for processing.
pub struct GpuJob {
    /// Server-assigned UUIDv4 carried over from `GenerationJob.id`. Used by
    /// the worker to flip the registry entry from `Queued` → `Running` (and
    /// to remove it when the job finishes), and surfaced to clients via
    /// `GET /api/queue` for zombie-card reconciliation.
    pub id: String,
    pub model: String,
    pub request: mold_core::GenerateRequest,
    pub completion_payload: crate::state::SseCompletionPayload,
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::state::SseMessage>>,
    pub result_tx: tokio::sync::oneshot::Sender<Result<crate::state::GenerationJobResult, String>>,
    pub output_dir: Option<std::path::PathBuf>,
    pub config: Arc<tokio::sync::RwLock<mold_core::Config>>,
    /// Metadata DB handle so the worker can record a row alongside the
    /// on-disk save. `Arc<Option<...>>` mirrors `AppState.metadata_db` —
    /// `None` when the DB failed to open or is disabled.
    pub metadata_db: Arc<Option<MetadataDb>>,
    /// Decrement the global queue counter when the worker finishes this job.
    pub queue: crate::state::QueueHandle,
    /// Job registry handle so the worker can flip state to Running on pickup
    /// and remove the entry on completion / error. Cheap clone — registry is
    /// behind an `Arc<RwLock>` internally.
    pub registry: crate::job_registry::SharedJobRegistry,
    /// Server-wide event broadcast so the worker's save path can emit
    /// `gallery_added` alongside the DB upsert (mirrors `AppState.events`).
    pub events: Arc<crate::events::EventBroadcaster>,
    /// Exact request-aware plan selected by the coordinator. Legacy adapters
    /// use `None`; V2 validates this on the owner thread before CUDA work.
    pub execution_plan: Option<crate::execution_plan::ResolvedExecutionPlan>,
    /// All per-device dependency-prepared inputs retained across a fenced
    /// transport rejection or pre-CUDA plan invalidation.
    pub prepared_execution_inputs: Option<crate::execution_plan::PreparedExecutionInputs>,
    /// Version fence proving this job was granted by the authoritative
    /// scheduler after the worker published the matching Ready generation.
    /// `None` exists only for legacy unit tests and the single-GPU adapter.
    pub lease: Option<crate::scheduler::LeaseFence>,
}

pub struct PromptExpansionJob {
    pub id: String,
    pub config: mold_core::Config,
    pub settings: mold_core::ExpandSettings,
    pub prompt: String,
    pub expand_config: mold_core::ExpandConfig,
    pub result_tx: tokio::sync::oneshot::Sender<Result<mold_core::ExpandResult, String>>,
}

pub struct StandaloneUpscaleJob {
    pub id: String,
    pub model: String,
    pub weights_path: std::path::PathBuf,
    pub request: mold_core::UpscaleRequest,
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::state::SseMessage>>,
    pub result_tx: tokio::sync::oneshot::Sender<Result<mold_core::UpscaleResponse, String>>,
}

pub struct PostGenerationUpscaleJob {
    pub id: String,
    pub generation: Box<GpuJob>,
    pub response: mold_core::GenerateResponse,
    pub image: mold_core::ImageData,
}

impl PostGenerationUpscaleJob {
    fn reject(self, error: String) {
        if let Some(progress) = &self.generation.progress_tx {
            let _ = progress.send(crate::state::SseMessage::Error(mold_core::SseErrorEvent {
                message: error.clone(),
            }));
        }
        let generation_id = self.generation.id.clone();
        let _ = self.generation.result_tx.send(Err(error));
        self.generation.queue.decrement();
        self.generation.registry.remove(&generation_id);
    }
}

pub struct AdminModelLoadJob {
    pub id: String,
    pub model: String,
    pub config: mold_core::Config,
    pub result_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
}

pub struct AdminModelUnloadJob {
    pub id: String,
    pub model: Option<String>,
    /// Remove the cached engine entirely instead of merely parking the active
    /// model. Used before deleting model files from disk.
    pub evict_cached: bool,
    pub result_tx: tokio::sync::oneshot::Sender<Result<Option<String>, String>>,
}

pub enum OwnerWork {
    Generation(Box<GpuJob>),
    PromptExpansion(Box<PromptExpansionJob>),
    PostUpscale(Box<PostGenerationUpscaleJob>),
    StandaloneUpscale(Box<StandaloneUpscaleJob>),
    AdminModelLoad(Box<AdminModelLoadJob>),
    AdminModelUnload(Box<AdminModelUnloadJob>),
    #[cfg(test)]
    Probe {
        id: String,
        kind: WorkKind,
        run: Box<dyn FnOnce() + Send>,
    },
}

impl OwnerWork {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Generation(job) => &job.id,
            Self::PromptExpansion(job) => &job.id,
            Self::PostUpscale(job) => &job.id,
            Self::StandaloneUpscale(job) => &job.id,
            Self::AdminModelLoad(job) => &job.id,
            Self::AdminModelUnload(job) => &job.id,
            #[cfg(test)]
            Self::Probe { id, .. } => id,
        }
    }

    pub(crate) fn kind(&self) -> WorkKind {
        match self {
            Self::Generation(_) => WorkKind::Generation,
            Self::PromptExpansion(_) => WorkKind::PromptExpansion,
            Self::PostUpscale(_) => WorkKind::PostUpscale,
            Self::StandaloneUpscale(_) => WorkKind::StandaloneUpscale,
            Self::AdminModelLoad(_) => WorkKind::AdminModelLoad,
            Self::AdminModelUnload(_) => WorkKind::AdminModelUnload,
            #[cfg(test)]
            Self::Probe { kind, .. } => *kind,
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        match self {
            Self::Generation(job) => job.result_tx.is_closed(),
            Self::PromptExpansion(job) => job.result_tx.is_closed(),
            Self::PostUpscale(job) => job.generation.result_tx.is_closed(),
            Self::StandaloneUpscale(job) => job.result_tx.is_closed(),
            Self::AdminModelLoad(job) => job.result_tx.is_closed(),
            Self::AdminModelUnload(job) => job.result_tx.is_closed(),
            #[cfg(test)]
            Self::Probe { .. } => false,
        }
    }

    pub(crate) fn reject(self, error: String) {
        match self {
            Self::Generation(job) => {
                if let Some(progress) = &job.progress_tx {
                    let _ =
                        progress.send(crate::state::SseMessage::Error(mold_core::SseErrorEvent {
                            message: error.clone(),
                        }));
                }
                let job_id = job.id.clone();
                let _ = job.result_tx.send(Err(error));
                job.queue.decrement();
                job.registry.remove(&job_id);
            }
            Self::PromptExpansion(job) => {
                let _ = job.result_tx.send(Err(error));
            }
            Self::PostUpscale(job) => {
                job.reject(error);
            }
            Self::StandaloneUpscale(job) => {
                let _ = job.result_tx.send(Err(error));
            }
            Self::AdminModelLoad(job) => {
                let _ = job.result_tx.send(Err(error));
            }
            Self::AdminModelUnload(job) => {
                let _ = job.result_tx.send(Err(error));
            }
            #[cfg(test)]
            Self::Probe { .. } => {}
        }
    }
}

pub struct LeaseGrant {
    pub fence: crate::scheduler::LeaseFence,
    pub work: OwnerWork,
    pub retry: Option<OwnerWorkRetry>,
}

#[derive(Clone)]
pub struct OwnerWorkRetry {
    pub model_fingerprint: String,
    pub estimated_vram_bytes: u64,
    pub estimated_host_ram_bytes: u64,
    pub hard_ordinal: Option<usize>,
    pub priority: mold_scheduler::PriorityClass,
    pub queue_rank: u64,
    pub ready_at_ms: u64,
    pub bypass_count: u8,
    pub warm_wait_started_ms: Option<u64>,
}

pub enum GpuWorkerCommand {
    Grant(Box<LeaseGrant>),
    Drain,
    Shutdown,
}

impl GpuWorker {
    pub(crate) fn reserve_legacy_transport(&self) {
        self.legacy_pending.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn settle_legacy_transport(&self) {
        self.legacy_pending
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_sub(1)
            })
            .expect("legacy owner completed work without a transport reservation");
    }

    pub(crate) fn pending_or_executing(&self) -> usize {
        self.legacy_pending
            .load(Ordering::SeqCst)
            .saturating_add(self.in_flight.load(Ordering::SeqCst))
    }

    /// Atomically claim this device for one owner-scheduled or legacy-isolated
    /// GPU operation.
    ///
    /// `in_flight` is a binary exclusion fence in V2. Legacy chain stages use
    /// the same claim until they are migrated to first-class scheduler leases,
    /// preventing a stale Ready advertisement from overlapping owner work.
    pub(crate) fn try_claim_in_flight(&self) -> bool {
        self.in_flight
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Claim this worker for scheduler-owned work while honoring the oldest
    /// legacy chain stage registered for this device.
    ///
    /// The chain waiter's bypass count is shared across every eligible GPU,
    /// so the bound is global rather than `MAX * worker_count`.
    pub(crate) fn try_claim_owner_in_flight(&self) -> bool {
        loop {
            let mut waiters = self
                .legacy_chain_waiters
                .waiters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            waiters.retain(|waiter| waiter.strong_count() > 0);
            waiters
                .sort_by_key(|waiter| waiter.upgrade().map_or(usize::MAX, |waiter| waiter.ticket));
            let Some(waiter) = waiters.first().and_then(Weak::upgrade) else {
                return self.try_claim_in_flight();
            };
            let mut state = waiter
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.claimed {
                waiters.remove(0);
                continue;
            }
            if state.owner_bypasses >= MAX_OWNER_BYPASSES_FOR_CHAIN {
                return false;
            }
            if !self.try_claim_in_flight() {
                return false;
            }
            state.owner_bypasses += 1;
            if state.owner_bypasses == MAX_OWNER_BYPASSES_FOR_CHAIN {
                state.wake_sequence = state.wake_sequence.saturating_add(1);
                waiter.wake.notify_all();
            }
            return true;
        }
    }

    /// Wait until rollback owner work can acquire the same binary execution
    /// claim used by V2 grants and legacy chain stages.
    ///
    /// The sequence lock closes the check-then-sleep race: a release cannot
    /// publish its wake until this waiter is parked on the condition variable.
    pub(crate) fn wait_claim_owner_in_flight(&self) -> bool {
        loop {
            let wake_sequence = self
                .legacy_chain_waiters
                .execution_wake_sequence
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if self.shutdown_requested.load(Ordering::SeqCst)
                || self.poisoned.load(Ordering::SeqCst)
                || self.fatal_cuda_error.load(Ordering::SeqCst)
            {
                return false;
            }
            if self.try_claim_owner_in_flight() {
                return true;
            }
            #[cfg(test)]
            {
                self.legacy_chain_waiters
                    .owner_claim_waiters
                    .fetch_add(1, Ordering::SeqCst);
                self.legacy_chain_waiters.owner_claim_wait.notify_all();
            }
            let _wake_sequence = self
                .legacy_chain_waiters
                .execution_wake
                .wait(wake_sequence)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            #[cfg(test)]
            self.legacy_chain_waiters
                .owner_claim_waiters
                .fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[cfg(test)]
    pub(crate) fn wait_until_owner_claim_blocked(&self) {
        let mut wake_sequence = self
            .legacy_chain_waiters
            .execution_wake_sequence
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while self
            .legacy_chain_waiters
            .owner_claim_waiters
            .load(Ordering::SeqCst)
            == 0
        {
            wake_sequence = self
                .legacy_chain_waiters
                .owner_claim_wait
                .wait(wake_sequence)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    pub(crate) fn register_legacy_chain_waiter(&self, waiter: &Arc<LegacyChainWaiter>) {
        let mut waiters = self
            .legacy_chain_waiters
            .waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        waiters.retain(|queued| queued.strong_count() > 0);
        if waiters
            .iter()
            .filter_map(Weak::upgrade)
            .any(|queued| Arc::ptr_eq(&queued, waiter))
        {
            return;
        }
        waiters.push(Arc::downgrade(waiter));
        waiters.sort_by_key(|queued| queued.upgrade().map_or(usize::MAX, |queued| queued.ticket));
    }

    pub(crate) fn unregister_legacy_chain_waiter(&self, waiter: &Arc<LegacyChainWaiter>) {
        {
            self.legacy_chain_waiters
                .waiters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .retain(|queued| {
                    queued
                        .upgrade()
                        .is_some_and(|queued| !Arc::ptr_eq(&queued, waiter))
                });
        }
        // A rollback owner can be parked solely because this waiter exhausted
        // the global bypass allowance. Cancellation/removal must reopen that
        // owner even though no execution claim was released.
        self.notify_execution_waiters();
    }

    #[cfg(test)]
    pub(crate) fn legacy_chain_waiter_count(&self) -> usize {
        self.legacy_chain_waiters
            .waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|waiter| waiter.strong_count() > 0)
            .count()
    }

    pub(crate) fn try_claim_legacy_chain_in_flight(&self, waiter: &Arc<LegacyChainWaiter>) -> bool {
        let mut waiters = self
            .legacy_chain_waiters
            .waiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        waiters.retain(|queued| queued.strong_count() > 0);
        waiters.sort_by_key(|queued| queued.upgrade().map_or(usize::MAX, |queued| queued.ticket));
        if !waiters
            .first()
            .and_then(Weak::upgrade)
            .is_some_and(|queued| Arc::ptr_eq(&queued, waiter))
        {
            return false;
        }
        let mut state = waiter
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.claimed || !self.try_claim_in_flight() {
            return false;
        }
        state.claimed = true;
        true
    }

    pub(crate) fn release_in_flight(&self) {
        self.in_flight.store(0, Ordering::SeqCst);
        self.notify_legacy_chain_waiters();
        self.notify_execution_waiters();
    }

    pub(crate) fn notify_execution_waiters(&self) {
        let mut sequence = self
            .legacy_chain_waiters
            .execution_wake_sequence
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *sequence = sequence.saturating_add(1);
        self.legacy_chain_waiters.execution_wake.notify_all();
    }

    fn notify_legacy_chain_waiters(&self) {
        let waiters = {
            let mut waiters = self
                .legacy_chain_waiters
                .waiters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let live = waiters.iter().filter_map(Weak::upgrade).collect::<Vec<_>>();
            waiters.retain(|waiter| waiter.strong_count() > 0);
            live
        };
        for waiter in waiters {
            waiter.notify();
        }
    }

    pub(crate) fn try_send_job(
        &self,
        grant: Box<LeaseGrant>,
    ) -> Result<(), std::sync::mpsc::TrySendError<Box<LeaseGrant>>> {
        self.job_tx
            .try_send(GpuWorkerCommand::Grant(grant))
            .map_err(|error| match error {
                std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Grant(grant)) => {
                    std::sync::mpsc::TrySendError::Full(grant)
                }
                std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Grant(grant)) => {
                    std::sync::mpsc::TrySendError::Disconnected(grant)
                }
                std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Drain)
                | std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Drain)
                | std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Shutdown)
                | std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Shutdown) => {
                    unreachable!("try_send_job only transports Grant")
                }
            })
    }

    #[cfg(test)]
    pub(crate) fn send_job(
        &self,
        job: GpuJob,
    ) -> Result<(), std::sync::mpsc::SendError<Box<LeaseGrant>>> {
        let fence = job.lease.clone().unwrap_or(crate::scheduler::LeaseFence {
            work_id: job.id.clone(),
            device_id: crate::scheduler::worker_device_id(self),
            owner_epoch: self.owner_epoch,
            state_version: 0,
            plan_version: 0,
            worker_generation: 1,
            memory_sample_generation: 0,
            memory_ledger_sequence: 0,
        });
        let grant = LeaseGrant {
            fence,
            work: OwnerWork::Generation(Box::new(job)),
            retry: None,
        };
        self.send_grant(grant)
    }

    #[cfg(test)]
    pub(crate) fn send_grant(
        &self,
        grant: LeaseGrant,
    ) -> Result<(), std::sync::mpsc::SendError<Box<LeaseGrant>>> {
        self.job_tx
            .send(GpuWorkerCommand::Grant(Box::new(grant)))
            .map_err(|error| match error.0 {
                GpuWorkerCommand::Grant(grant) => std::sync::mpsc::SendError(grant),
                GpuWorkerCommand::Drain | GpuWorkerCommand::Shutdown => {
                    unreachable!("send_job only transports Grant")
                }
            })
    }

    pub(crate) fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
        self.notify_legacy_chain_waiters();
        self.notify_execution_waiters();
        match self.job_tx.try_send(GpuWorkerCommand::Shutdown) {
            Ok(())
            | Err(std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Shutdown))
            | Err(std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Shutdown)) => {}
            Err(std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Grant(_)))
            | Err(std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Grant(_))) => {
                unreachable!("request_shutdown only transports Shutdown")
            }
            Err(std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Drain))
            | Err(std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Drain)) => {
                unreachable!("request_shutdown only transports Shutdown")
            }
        }
    }

    pub(crate) fn request_drain(&self, wake_idle_owner: bool) {
        let _ = self.drain_state.compare_exchange(
            DRAIN_RUNNING,
            DRAIN_REQUESTED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        if !wake_idle_owner {
            return;
        }
        match self.job_tx.try_send(GpuWorkerCommand::Drain) {
            Ok(())
            | Err(std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Drain))
            | Err(std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Drain)) => {}
            Err(std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Grant(_)))
            | Err(std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Grant(_)))
            | Err(std::sync::mpsc::TrySendError::Full(GpuWorkerCommand::Shutdown))
            | Err(std::sync::mpsc::TrySendError::Disconnected(GpuWorkerCommand::Shutdown)) => {
                unreachable!("request_drain only transports Drain")
            }
        }
    }

    pub(crate) fn cancel_drain(&self) -> bool {
        self.drain_state
            .compare_exchange(
                DRAIN_REQUESTED,
                DRAIN_RUNNING,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
            || self.drain_state.load(Ordering::SeqCst) == DRAIN_RUNNING
    }

    pub(crate) fn commit_drain(&self) -> bool {
        self.drain_state
            .compare_exchange(
                DRAIN_REQUESTED,
                DRAIN_COMMITTED,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
    }
}

pub(crate) trait OwnerThreadSpawner: Send + Sync {
    fn spawn(
        &self,
        worker: Arc<GpuWorker>,
        job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
        scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
        cache_idle_ttl: Duration,
    ) -> std::io::Result<std::thread::JoinHandle<()>>;
}

pub(crate) struct RuntimeOwnerThreadSpawner;

impl OwnerThreadSpawner for RuntimeOwnerThreadSpawner {
    fn spawn(
        &self,
        worker: Arc<GpuWorker>,
        job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
        scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
        cache_idle_ttl: Duration,
    ) -> std::io::Result<std::thread::JoinHandle<()>> {
        crate::gpu_worker::spawn_gpu_thread_async(worker, job_rx, scheduler_tx, cache_idle_ttl)
    }
}

pub(crate) struct WorkerFactory {
    pub devices: BTreeMap<String, DiscoveredGpu>,
    pub shared_pool: Arc<Mutex<SharedPool>>,
    pub fatal_cuda_error: Arc<AtomicBool>,
    pub fatal_cuda_shutdown: Arc<tokio::sync::Notify>,
    pub scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
    pub owner_spawner: Arc<dyn OwnerThreadSpawner>,
    pub max_cached: usize,
    pub cache_idle_ttl: Duration,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OwnerKey {
    device_id: String,
    owner_epoch: u64,
}

#[derive(Default)]
struct WorkerLifecycle {
    handles: BTreeMap<OwnerKey, std::thread::JoinHandle<()>>,
    starting: BTreeMap<String, u64>,
    start_announced: BTreeSet<OwnerKey>,
    ready_seen: BTreeSet<OwnerKey>,
    failed_seen: BTreeMap<OwnerKey, String>,
    start_failures: BTreeMap<String, String>,
    next_owner_epoch: u64,
}

pub(crate) enum StartAnnouncement {
    Pending,
    Ready,
    Failed(String),
}

pub struct WorkerSet {
    active: RwLock<Vec<Arc<GpuWorker>>>,
    lifecycle: Mutex<WorkerLifecycle>,
    factory: OnceLock<WorkerFactory>,
}

impl Default for WorkerSet {
    fn default() -> Self {
        Self::from(Vec::new())
    }
}

impl From<Vec<Arc<GpuWorker>>> for WorkerSet {
    fn from(workers: Vec<Arc<GpuWorker>>) -> Self {
        let next_owner_epoch = workers
            .iter()
            .map(|worker| worker.owner_epoch)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Self {
            active: RwLock::new(workers),
            lifecycle: Mutex::new(WorkerLifecycle {
                next_owner_epoch,
                ..WorkerLifecycle::default()
            }),
            factory: OnceLock::new(),
        }
    }
}

impl FromIterator<Arc<GpuWorker>> for WorkerSet {
    fn from_iter<T: IntoIterator<Item = Arc<GpuWorker>>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl IntoIterator for &WorkerSet {
    type Item = Arc<GpuWorker>;
    type IntoIter = std::vec::IntoIter<Arc<GpuWorker>>;

    fn into_iter(self) -> Self::IntoIter {
        self.snapshot().into_iter()
    }
}

pub struct GpuPool {
    pub workers: WorkerSet,
}

impl WorkerSet {
    pub fn snapshot(&self) -> Vec<Arc<GpuWorker>> {
        self.active
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn len(&self) -> usize {
        self.active
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> std::vec::IntoIter<Arc<GpuWorker>> {
        self.snapshot().into_iter()
    }

    pub(crate) fn install_factory(
        &self,
        factory: WorkerFactory,
        handles: Vec<(String, u64, std::thread::JoinHandle<()>)>,
    ) -> Result<(), &'static str> {
        self.factory
            .set(factory)
            .map_err(|_| "GPU worker factory already installed")?;
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (device_id, owner_epoch, handle) in handles {
            lifecycle.handles.insert(
                OwnerKey {
                    device_id,
                    owner_epoch,
                },
                handle,
            );
            lifecycle.next_owner_epoch = lifecycle
                .next_owner_epoch
                .max(owner_epoch.saturating_add(1));
        }
        Ok(())
    }

    pub(crate) fn is_starting(&self, device_id: &str) -> bool {
        self.lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .starting
            .contains_key(device_id)
    }

    pub(crate) fn last_start_error(&self, device_id: &str) -> Option<String> {
        self.lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .start_failures
            .get(device_id)
            .cloned()
    }

    pub(crate) fn request_disable(&self, device_id: &str) -> Result<bool, String> {
        let _lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(worker) = self
            .snapshot()
            .into_iter()
            .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
        else {
            return Ok(false);
        };
        let busy = worker.pending_or_executing() > 0
            || worker
                .active_generation
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_some();
        worker.request_drain(!busy);
        Ok(busy)
    }

    pub(crate) fn cancel_drain(&self, device_id: &str) -> bool {
        let _lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.snapshot()
            .into_iter()
            .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
            .is_some_and(|worker| worker.cancel_drain())
    }

    pub(crate) fn mark_ready(&self, device_id: &str, owner_epoch: u64) -> bool {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = OwnerKey {
            device_id: device_id.to_string(),
            owner_epoch,
        };
        if lifecycle.starting.get(device_id) == Some(&owner_epoch) {
            lifecycle.ready_seen.insert(key.clone());
            if lifecycle.start_announced.contains(&key) {
                lifecycle.starting.remove(device_id);
            }
            lifecycle.start_failures.remove(device_id);
            return true;
        }
        self.active
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .any(|worker| {
                crate::scheduler::worker_device_id(worker) == device_id
                    && worker.owner_epoch == owner_epoch
            })
    }

    pub(crate) fn announce_start(&self, device_id: &str, owner_epoch: u64) -> StartAnnouncement {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if lifecycle.starting.get(device_id) != Some(&owner_epoch) {
            return StartAnnouncement::Pending;
        }
        let key = OwnerKey {
            device_id: device_id.to_string(),
            owner_epoch,
        };
        lifecycle.start_announced.insert(key.clone());
        if let Some(error) = lifecycle.failed_seen.remove(&key) {
            lifecycle.starting.remove(device_id);
            lifecycle.start_announced.remove(&key);
            lifecycle
                .start_failures
                .insert(device_id.to_string(), error.clone());
            return StartAnnouncement::Failed(error);
        }
        if lifecycle.ready_seen.contains(&key) {
            lifecycle.starting.remove(device_id);
            return StartAnnouncement::Ready;
        }
        StartAnnouncement::Pending
    }

    pub(crate) fn wait_and_reap(&self, device_id: &str, owner_epoch: u64) -> bool {
        let key = OwnerKey {
            device_id: device_id.to_string(),
            owner_epoch,
        };
        let handle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .handles
            .remove(&key);
        if let Some(handle) = handle {
            let _ = handle.join();
        }
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut active = self
            .active
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = active.len();
        active.retain(|worker| {
            crate::scheduler::worker_device_id(worker) != device_id
                || worker.owner_epoch != owner_epoch
        });
        if lifecycle.starting.get(device_id) == Some(&owner_epoch) {
            lifecycle.starting.remove(device_id);
        }
        lifecycle.start_announced.remove(&key);
        lifecycle.ready_seen.remove(&key);
        lifecycle.failed_seen.remove(&key);
        before != active.len()
    }

    pub(crate) fn mark_start_failed(
        &self,
        device_id: &str,
        owner_epoch: u64,
        error: String,
    ) -> Option<bool> {
        let key = OwnerKey {
            device_id: device_id.to_string(),
            owner_epoch,
        };
        let (announced, handle) = {
            let mut lifecycle = self
                .lifecycle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                lifecycle.start_announced.contains(&key),
                lifecycle.handles.remove(&key),
            )
        };
        let had_handle = handle.is_some();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut active = self
            .active
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = active.len();
        active.retain(|worker| {
            crate::scheduler::worker_device_id(worker) != device_id
                || worker.owner_epoch != owner_epoch
        });
        if !had_handle && before == active.len() {
            return None;
        }
        drop(active);
        lifecycle.starting.remove(device_id);
        lifecycle.start_announced.remove(&key);
        lifecycle.ready_seen.remove(&key);
        lifecycle.failed_seen.remove(&key);
        if announced {
            lifecycle
                .start_failures
                .insert(device_id.to_string(), error);
        } else {
            lifecycle
                .starting
                .insert(device_id.to_string(), owner_epoch);
            lifecycle.failed_seen.insert(key, error);
        }
        Some(announced)
    }

    pub(crate) fn start(&self, device_id: &str) -> Result<u64, String> {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(owner_epoch) = lifecycle.starting.get(device_id).copied() {
            let key = OwnerKey {
                device_id: device_id.to_string(),
                owner_epoch,
            };
            if !lifecycle.failed_seen.contains_key(&key) {
                return Ok(owner_epoch);
            }
            let error = lifecycle
                .failed_seen
                .remove(&key)
                .expect("checked deferred failure");
            lifecycle.starting.remove(device_id);
            lifecycle.start_announced.remove(&key);
            lifecycle.ready_seen.remove(&key);
            lifecycle
                .start_failures
                .insert(device_id.to_string(), error);
        }
        if let Some(worker) = self
            .snapshot()
            .iter()
            .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
        {
            return Ok(worker.owner_epoch);
        }
        let factory = self
            .factory
            .get()
            .ok_or_else(|| "dynamic GPU lifecycle is unavailable".to_string())?;
        if factory.fatal_cuda_error.load(Ordering::SeqCst) {
            return Err(
                "process CUDA context is fatally poisoned; server restart required".to_string(),
            );
        }
        let gpu = factory
            .devices
            .get(device_id)
            .cloned()
            .ok_or_else(|| format!("device '{device_id}' cannot be started in this process"))?;
        let owner_epoch = lifecycle.next_owner_epoch.max(1);
        lifecycle.next_owner_epoch = owner_epoch.saturating_add(1);
        lifecycle
            .starting
            .insert(device_id.to_string(), owner_epoch);
        lifecycle.start_failures.remove(device_id);

        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(1);
        let worker = Arc::new(GpuWorker {
            owner_epoch,
            gpu,
            model_cache: Arc::new(Mutex::new(ModelCache::new(factory.max_cached))),
            resident_model: Arc::new(RwLock::new(None)),
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: factory.shared_pool.clone(),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: factory.fatal_cuda_error.clone(),
            fatal_cuda_shutdown: factory.fatal_cuda_shutdown.clone(),
            shutdown_requested: AtomicBool::new(false),
            drain_state: AtomicU8::new(DRAIN_RUNNING),
            owner_thread_id: OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        let owner_key = OwnerKey {
            device_id: device_id.to_string(),
            owner_epoch,
        };
        let handle = match factory.owner_spawner.spawn(
            worker.clone(),
            job_rx,
            factory.scheduler_tx.clone(),
            factory.cache_idle_ttl,
        ) {
            Ok(handle) => handle,
            Err(error) => {
                let error = format!("failed to start GPU owner thread: {error}");
                if lifecycle.starting.get(device_id) == Some(&owner_epoch) {
                    lifecycle.starting.remove(device_id);
                }
                lifecycle.start_announced.remove(&owner_key);
                lifecycle.ready_seen.remove(&owner_key);
                lifecycle.failed_seen.remove(&owner_key);
                lifecycle
                    .start_failures
                    .insert(device_id.to_string(), error.clone());
                return Err(error);
            }
        };
        self.active
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(worker);
        lifecycle.handles.insert(owner_key, handle);
        Ok(owner_epoch)
    }

    pub(crate) fn shutdown_and_join_all(&self) {
        let handles = {
            let mut lifecycle = self
                .lifecycle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for worker in self.snapshot() {
                worker.request_shutdown();
            }
            lifecycle.starting.clear();
            lifecycle.start_announced.clear();
            lifecycle.ready_seen.clear();
            lifecycle.failed_seen.clear();
            std::mem::take(&mut lifecycle.handles)
        };
        for (_, handle) in handles {
            let _ = handle.join();
        }
        self.active
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }
}

impl GpuWorker {
    pub(crate) fn set_resident_model(&self, model: Option<&str>) {
        *self
            .resident_model
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = model.map(str::to_string);
        *self
            .resident_execution_fingerprint
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    pub(crate) fn set_resident_execution_fingerprint(&self, fingerprint: Option<&str>) {
        *self
            .resident_execution_fingerprint
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = fingerprint.map(str::to_string);
    }

    /// Check if this worker is in a degraded state (3+ consecutive failures, within cooldown).
    ///
    /// When the cooldown has expired we clear the failure counter and the
    /// `degraded_until` timestamp so the next single failure doesn't
    /// immediately re-degrade the GPU. Without this lazy reset, a worker
    /// that has 3 historical failures followed by a long idle period would
    /// flip back to Degraded on the very first post-cooldown failure
    /// (because `consecutive_failures` is still >= 3 from before).
    pub fn is_degraded(&self) -> bool {
        if self.poisoned.load(Ordering::SeqCst) {
            return true;
        }
        if self.consecutive_failures.load(Ordering::SeqCst) < 3 {
            return false;
        }
        let cooldown_active = match *self.degraded_until.read().unwrap() {
            Some(until) => Instant::now() < until,
            None => false,
        };
        if !cooldown_active {
            // Lazy clear: cooldown elapsed, treat the worker as healthy
            // again. A new failure burst still has to reach 3 consecutive
            // failures before re-degrading.
            self.consecutive_failures.store(0, Ordering::SeqCst);
            *self.degraded_until.write().unwrap() = None;
        }
        cooldown_active
    }

    /// Build a status snapshot for this worker.
    pub fn status(&self) -> GpuWorkerStatus {
        let active_gen = self.active_generation.read().unwrap();
        let in_flight = self.in_flight.load(Ordering::SeqCst);
        let loaded_model = active_gen
            .as_ref()
            .map(|g| g.model.clone())
            .or_else(|| self.resident_model.read().unwrap().clone());

        let state = if self.is_degraded() {
            GpuWorkerState::Degraded
        } else if active_gen.is_some() || in_flight > 0 {
            GpuWorkerState::Generating
        } else {
            GpuWorkerState::Idle
        };

        GpuWorkerStatus {
            ordinal: self.gpu.ordinal,
            name: self.gpu.name.clone(),
            vram_total_bytes: self.gpu.total_vram_bytes,
            // The registry overlays cached sampler telemetry. Never query the
            // CUDA driver while serving a status request.
            vram_used_bytes: 0,
            loaded_model,
            state,
        }
    }
}

impl GpuPool {
    pub fn schedulable_workers(&self) -> Vec<Arc<GpuWorker>> {
        self.worker_snapshot()
            .into_iter()
            .filter(|worker| {
                !worker.shutdown_requested.load(Ordering::SeqCst)
                    && worker.drain_state.load(Ordering::SeqCst) == DRAIN_RUNNING
                    && !worker.poisoned.load(Ordering::SeqCst)
                    && !worker.is_degraded()
            })
            .collect()
    }

    pub fn schedulable_worker_count(&self) -> usize {
        self.schedulable_workers().len()
    }

    pub fn worker_snapshot(&self) -> Vec<Arc<GpuWorker>> {
        self.workers.snapshot()
    }

    /// Return the worker bound to `ordinal`, if present in this pool.
    pub fn worker_by_ordinal(&self, ordinal: usize) -> Option<Arc<GpuWorker>> {
        self.worker_snapshot()
            .into_iter()
            .find(|w| w.gpu.ordinal == ordinal)
    }

    /// Validate a request/config placement against the active worker pool.
    ///
    /// Every legacy ordinal and durable device-ID pin is resolved to one
    /// active worker. Cross-GPU component placement would bypass worker
    /// affinity, so reject it here instead of letting an engine silently open
    /// a sibling context.
    pub fn resolve_explicit_placement_gpu(
        &self,
        placement: Option<&DevicePlacement>,
    ) -> Result<Option<usize>, String> {
        let Some(placement) = placement else {
            return Ok(None);
        };

        let ordinals = self.placement_worker_ordinals(placement)?;
        if ordinals.is_empty() {
            return Ok(None);
        }
        if ordinals.len() > 1 {
            let rendered = ordinals
                .iter()
                .map(|o| format!("gpu:{o}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "multi-GPU worker mode only supports placement on one GPU ordinal per request; got {rendered}"
            ));
        }

        let ordinal = *ordinals.iter().next().expect("checked non-empty");
        if self.worker_by_ordinal(ordinal).is_none() {
            let available = self
                .worker_snapshot()
                .into_iter()
                .map(|w| w.gpu.ordinal.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "gpu:{ordinal} is not available in this server's worker pool [{available}]"
            ));
        }
        Ok(Some(ordinal))
    }

    fn placement_worker_ordinals(
        &self,
        placement: &DevicePlacement,
    ) -> Result<BTreeSet<usize>, String> {
        let mut ordinals = BTreeSet::new();
        self.collect_placement_worker(&placement.text_encoders, &mut ordinals)?;
        if let Some(adv) = placement.advanced.as_ref() {
            self.collect_placement_worker(&adv.transformer, &mut ordinals)?;
            self.collect_placement_worker(&adv.vae, &mut ordinals)?;
            for device in [
                adv.clip_l.as_ref(),
                adv.clip_g.as_ref(),
                adv.t5.as_ref(),
                adv.qwen.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                self.collect_placement_worker(device, &mut ordinals)?;
            }
        }
        Ok(ordinals)
    }

    fn collect_placement_worker(
        &self,
        device: &DeviceRef,
        out: &mut BTreeSet<usize>,
    ) -> Result<(), String> {
        let worker = match device {
            DeviceRef::Auto | DeviceRef::Cpu => return Ok(()),
            DeviceRef::Gpu { ordinal } => self.worker_by_ordinal(*ordinal).ok_or_else(|| {
                let available = self
                    .worker_snapshot()
                    .into_iter()
                    .map(|worker| worker.gpu.ordinal.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("gpu:{ordinal} is not available in this server's worker pool [{available}]")
            })?,
            DeviceRef::Device { id } => self
                .worker_snapshot()
                .into_iter()
                .find(|worker| worker.gpu.stable_id.as_deref() == Some(id.as_str()))
                .ok_or_else(|| {
                    let available = self
                        .worker_snapshot()
                        .into_iter()
                        .filter_map(|worker| worker.gpu.stable_id.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!(
                        "device '{id}' is not available in this server's worker pool [{available}]"
                    )
                })?,
        };
        out.insert(worker.gpu.ordinal);
        Ok(())
    }

    /// Find a non-degraded worker that already has this model loaded on GPU.
    /// If multiple workers have it, prefer the one with fewer in-flight requests.
    pub fn find_loaded(&self, model_name: &str) -> Option<Arc<GpuWorker>> {
        let mut candidates: Vec<_> = self
            .schedulable_workers()
            .into_iter()
            .filter(|w| {
                let active_gen = w.active_generation.read().unwrap();
                if active_gen.as_ref().is_some_and(|g| g.model == model_name) {
                    return true;
                }
                let cache = w.model_cache.lock().unwrap();
                cache
                    .get(model_name)
                    .map(|e| e.residency == ModelResidency::Gpu)
                    .unwrap_or(false)
            })
            .collect();

        candidates.sort_by_key(|w| w.pending_or_executing());
        candidates.into_iter().next()
    }

    /// Select the best worker for a model, using the placement strategy
    /// (checked in order):
    /// 1. Loaded and idle (model on GPU, no in-flight requests).
    /// 2. Loaded but busy — queue behind the warm copy instead of reloading.
    /// 3. Idle GPU with no model (spreads cold loads across free GPUs).
    /// 4. Non-degraded worker with the most headroom (will evict LRU).
    pub fn select_worker(&self, model_name: &str, estimated_vram: u64) -> Option<Arc<GpuWorker>> {
        self.select_worker_excluding(model_name, estimated_vram, &[])
    }

    /// Same as [`select_worker`], but skips workers whose ordinal is in `skip`.
    /// Used by the dispatcher to retry after a `try_send` failure.
    pub fn select_worker_excluding(
        &self,
        model_name: &str,
        estimated_vram: u64,
        skip: &[usize],
    ) -> Option<Arc<GpuWorker>> {
        let eligible: Vec<Arc<GpuWorker>> = self
            .schedulable_workers()
            .into_iter()
            .filter(|w| !skip.contains(&w.gpu.ordinal))
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // Classify each eligible worker.
        let mut loaded_idle: Vec<&Arc<GpuWorker>> = Vec::new();
        let mut loaded_busy: Vec<&Arc<GpuWorker>> = Vec::new();
        let mut idle_empty: Vec<&Arc<GpuWorker>> = Vec::new();
        let mut other: Vec<&Arc<GpuWorker>> = Vec::new();

        for w in &eligible {
            let active_gen = w.active_generation.read().unwrap();
            let active_model = active_gen.as_ref().map(|g| g.model.as_str());
            let (has_model, has_any_loaded) = {
                let cache = w.model_cache.lock().unwrap();
                let has_model = active_model == Some(model_name)
                    || cache
                        .get(model_name)
                        .map(|e| e.residency == ModelResidency::Gpu)
                        .unwrap_or(false);
                (
                    has_model,
                    active_model.is_some() || cache.active_model().is_some(),
                )
            };
            let in_flight = w.pending_or_executing();
            // During an in-flight generation the worker thread calls
            // `cache.take()`, which removes the entry entirely — so
            // `cache.active_model()` and `cache.get(model).residency == Gpu`
            // both return None/false for the duration of that generation.
            // That used to let a busy GPU mid-inference look identical to
            // a truly empty idle GPU, which meant a new job for a *different*
            // model could be dispatched to the busy card while a sibling GPU
            // sat idle. `in_flight > 0` (set by the dispatcher before send)
            // and `active_generation.is_some()` (set by the worker around
            // the take-and-restore window) together cover every moment
            // between "about to pick up a job" and "just finished".
            let is_busy = in_flight > 0 || active_model.is_some();

            if has_model && !is_busy {
                loaded_idle.push(w);
            } else if has_model {
                loaded_busy.push(w);
            } else if !has_any_loaded && !is_busy {
                idle_empty.push(w);
            } else {
                other.push(w);
            }
        }

        // 1. Loaded and idle — least in-flight first (should all be 0).
        if !loaded_idle.is_empty() {
            loaded_idle.sort_by_key(|w| w.pending_or_executing());
            return loaded_idle.first().map(|w| (*w).clone());
        }

        // 2. Idle GPU with no model that FITS the estimate — spread beats
        //    queueing behind a busy warm card: the cold load is paid once,
        //    then that GPU is warm and tier 1 takes over for later jobs.
        //    (Warm-busy used to outrank this, which serialized same-model
        //    jobs on one card of a multi-GPU box while siblings sat idle.)
        idle_empty.sort_by_key(|w| w.gpu.total_vram_bytes);
        if let Some(w) = idle_empty
            .iter()
            .find(|w| w.gpu.total_vram_bytes >= estimated_vram)
        {
            return Some((*w).clone());
        }

        // 3. Loaded but busy — least in-flight wins. Reached only when no
        //    idle GPU can hold the model: waiting for the warm card beats
        //    cold-loading onto a card that would have to offload.
        if !loaded_busy.is_empty() {
            loaded_busy.sort_by_key(|w| w.pending_or_executing());
            return loaded_busy.first().map(|w| (*w).clone());
        }

        // 4. Idle GPU that doesn't fit — largest wins; block offloading may
        //    still make the model viable there.
        if !idle_empty.is_empty() {
            return idle_empty.last().map(|w| (*w).clone());
        }

        // 5. Everything left — busy with other models, or idle with a
        //    different model loaded (evicting a warm sibling has its own
        //    cost). Most headroom first; the LRU cache evicts there.
        let mut busy = other;
        busy.sort_by(|a, b| {
            let a_headroom = a.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            let b_headroom = b.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            b_headroom.cmp(&a_headroom)
        });
        busy.first().map(|w| (*w).clone())
    }

    /// Collect status from all workers.
    pub fn gpu_status(&self) -> Vec<GpuWorkerStatus> {
        self.worker_snapshot()
            .into_iter()
            .map(|w| w.status())
            .collect()
    }

    /// Number of GPU workers in the pool.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static MODEL_CUDA_OOM_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    use crate::model_cache::ModelCache;
    use mold_core::types::AdvancedPlacement;
    use mold_inference::shared_pool::SharedPool;

    /// Build a test GpuWorker with a scratch job channel and everything else
    /// in neutral defaults. Returns the worker plus the receiver so the test
    /// can verify what was dispatched.
    fn test_worker(
        ordinal: usize,
        total_vram_bytes: u64,
    ) -> (Arc<GpuWorker>, std::sync::mpsc::Receiver<GpuWorkerCommand>) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(2);
        let worker = Arc::new(GpuWorker {
            owner_epoch: 1,
            gpu: DiscoveredGpu {
                ordinal,
                stable_id: Some(format!("cuda:{ordinal:032x}")),
                raw_cuda_uuid: Some((ordinal as u128).to_be_bytes()),
                device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
                identity_error: None,
                backend: mold_core::types::GpuBackend::Cuda,
                name: format!("test-gpu-{ordinal}"),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes,
                free_vram_bytes: total_vram_bytes,
            },
            model_cache: Arc::new(Mutex::new(ModelCache::new(3))),
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
        (worker, job_rx)
    }

    #[test]
    fn one_enable_after_unannounced_start_failure_spawns_a_fresh_epoch() {
        let (template, _template_rx) = test_worker(0, 24_000_000_000);
        let device_id = crate::scheduler::worker_device_id(&template);
        let workers = WorkerSet::from(Vec::new());
        let (scheduler_tx, _scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
        workers
            .install_factory(
                WorkerFactory {
                    devices: BTreeMap::from([(device_id.clone(), template.gpu.clone())]),
                    shared_pool: Arc::new(Mutex::new(SharedPool::new())),
                    fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                    fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                    scheduler_tx,
                    owner_spawner: Arc::new(RuntimeOwnerThreadSpawner),
                    max_cached: 1,
                    cache_idle_ttl: Duration::from_secs(60),
                },
                Vec::new(),
            )
            .unwrap();
        {
            let mut lifecycle = workers.lifecycle.lock().unwrap();
            let failed_key = OwnerKey {
                device_id: device_id.clone(),
                owner_epoch: 1,
            };
            lifecycle.starting.insert(device_id.clone(), 1);
            lifecycle
                .failed_seen
                .insert(failed_key, "probe failed".to_string());
            lifecycle.next_owner_epoch = 2;
        }

        let replacement_epoch = workers
            .start(&device_id)
            .expect("one retry click must replace a failed unannounced owner");

        assert_eq!(replacement_epoch, 2);
        assert_eq!(workers.len(), 1);
        assert_eq!(workers.snapshot()[0].owner_epoch, 2);
        workers.shutdown_and_join_all();
    }

    struct FailFirstOwnerSpawner {
        attempts: AtomicUsize,
    }

    impl OwnerThreadSpawner for FailFirstOwnerSpawner {
        fn spawn(
            &self,
            worker: Arc<GpuWorker>,
            job_rx: std::sync::mpsc::Receiver<GpuWorkerCommand>,
            scheduler_tx: tokio::sync::mpsc::UnboundedSender<crate::scheduler::WorkerEvent>,
            _cache_idle_ttl: Duration,
        ) -> std::io::Result<std::thread::JoinHandle<()>> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                return Err(std::io::Error::other("synthetic thread spawn failure"));
            }
            std::thread::Builder::new()
                .name("synthetic-owner-spawn-retry".to_string())
                .spawn(move || {
                    let _ = scheduler_tx.send(crate::scheduler::WorkerEvent::Ready {
                        device_id: crate::scheduler::worker_device_id(&worker),
                        ordinal: worker.gpu.ordinal,
                        owner_epoch: worker.owner_epoch,
                        worker_generation: 1,
                    });
                    drop(job_rx);
                })
        }
    }

    #[tokio::test]
    async fn synchronous_spawn_failure_clears_starting_and_retry_uses_fresh_epoch() {
        let (template, _template_rx) = test_worker(0, 24_000_000_000);
        let device_id = crate::scheduler::worker_device_id(&template);
        let workers = WorkerSet::from(Vec::new());
        let (scheduler_tx, mut scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
        let spawner = Arc::new(FailFirstOwnerSpawner {
            attempts: AtomicUsize::new(0),
        });
        workers
            .install_factory(
                WorkerFactory {
                    devices: BTreeMap::from([(device_id.clone(), template.gpu.clone())]),
                    shared_pool: Arc::new(Mutex::new(SharedPool::new())),
                    fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                    fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                    scheduler_tx,
                    owner_spawner: spawner.clone(),
                    max_cached: 1,
                    cache_idle_ttl: Duration::from_secs(60),
                },
                Vec::new(),
            )
            .unwrap();

        let error = workers.start(&device_id).unwrap_err();
        assert!(error.contains("synthetic thread spawn failure"));
        assert!(!workers.is_starting(&device_id));
        assert!(workers.is_empty());
        assert!(workers
            .last_start_error(&device_id)
            .is_some_and(|error| error.contains("synthetic thread spawn failure")));
        assert!(
            scheduler_rx.try_recv().is_err(),
            "failed spawn cannot emit Ready"
        );

        let retry_epoch = workers.start(&device_id).unwrap();
        assert_eq!(retry_epoch, 2);
        assert_eq!(spawner.attempts.load(Ordering::SeqCst), 2);
        assert_eq!(workers.len(), 1);
        assert_eq!(workers.snapshot()[0].owner_epoch, retry_epoch);
        assert!(matches!(
            scheduler_rx.recv().await.unwrap(),
            crate::scheduler::WorkerEvent::Ready { owner_epoch: 2, .. }
        ));

        assert!(workers.wait_and_reap(&device_id, retry_epoch));
        assert!(workers.is_empty());
    }

    #[test]
    fn in_flight_claim_is_binary_and_reusable_after_release() {
        let (worker, _job_rx) = test_worker(0, 24_000_000_000);

        assert!(worker.try_claim_in_flight());
        assert!(!worker.try_claim_in_flight());
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 1);

        worker.release_in_flight();
        assert!(worker.try_claim_in_flight());
        worker.release_in_flight();
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cancelled_chain_waiter_wakes_owner_after_global_bypass_limit() {
        let (worker, _job_rx) = test_worker(0, 24_000_000_000);
        let waiter = LegacyChainWaiter::new();
        worker.register_legacy_chain_waiter(&waiter);
        for _ in 0..MAX_OWNER_BYPASSES_FOR_CHAIN {
            assert!(worker.try_claim_owner_in_flight());
            worker.release_in_flight();
        }

        let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
        let waiting_worker = worker.clone();
        let owner = std::thread::spawn(move || {
            claimed_tx
                .send(waiting_worker.wait_claim_owner_in_flight())
                .unwrap();
        });
        worker.wait_until_owner_claim_blocked();
        worker.unregister_legacy_chain_waiter(&waiter);

        assert!(claimed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("waiter removal must wake the rollback owner"));
        owner.join().unwrap();
        worker.release_in_flight();
    }

    #[test]
    fn worker_status_does_not_lock_the_model_cache_or_query_the_driver() {
        let (worker, _job_rx) = test_worker(0, 24_000_000_000);
        *worker.resident_model.write().unwrap() = Some("flux-dev:q4".to_string());

        let cache = worker.model_cache.clone();
        let _ = std::thread::spawn(move || {
            let _guard = cache.lock().unwrap();
            panic!("poison cache to prove status never acquires it");
        })
        .join();

        let status = worker.status();
        assert_eq!(status.loaded_model.as_deref(), Some("flux-dev:q4"));
        assert_eq!(status.vram_used_bytes, 0);
    }

    #[test]
    fn poisoned_worker_stays_degraded_after_cooldown_expiry() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        worker.poisoned.store(true, Ordering::SeqCst);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() = Some(Instant::now() - Duration::from_secs(1));

        assert!(worker.is_degraded());
        assert_eq!(worker.consecutive_failures.load(Ordering::SeqCst), 3);
        assert!(worker.degraded_until.read().unwrap().is_some());
    }

    /// When GPU 0 is actively generating a different model, the cache
    /// take-and-restore pattern has already removed its entry — so
    /// `cache.active_model()` returns None and the worker LOOKS idle
    /// to the old classifier. The dispatcher must fall back to
    /// `in_flight > 0` (or `active_generation`) to avoid routing a
    /// brand-new job to the busy GPU while a sibling sits idle.
    #[test]
    fn select_worker_prefers_truly_idle_gpu_over_busy_gpu_with_empty_cache() {
        let (busy, _busy_rx) = test_worker(0, 24_000_000_000);
        let (idle, _idle_rx) = test_worker(1, 24_000_000_000);

        // Simulate the dispatcher having incremented in_flight before send,
        // and the worker thread having called cache.take() → empty cache.
        busy.in_flight.store(1, Ordering::SeqCst);

        let pool = GpuPool {
            workers: vec![busy.clone(), idle.clone()].into(),
        };

        let picked = pool
            .select_worker("some-small-model:q4", 6_000_000_000)
            .expect("a worker should be selected");
        assert_eq!(
            picked.gpu.ordinal, 1,
            "new job for an unloaded model must go to the truly idle GPU, \
             not to the one whose cache momentarily looks empty because \
             generation is in progress"
        );
    }

    /// active_generation is set before take() and cleared after restore(),
    /// so a worker mid-inference should be treated as busy even if the
    /// dispatcher hasn't yet bumped in_flight (belt-and-suspenders).
    #[test]
    fn select_worker_respects_active_generation_flag() {
        let (busy, _busy_rx) = test_worker(0, 24_000_000_000);
        let (idle, _idle_rx) = test_worker(1, 24_000_000_000);

        *busy.active_generation.write().unwrap() = Some(ActiveGeneration {
            model: "big-model".to_string(),
            prompt_sha256: String::new(),
            started_at_unix_ms: 0,
            started_at: Instant::now(),
        });

        let pool = GpuPool {
            workers: vec![busy.clone(), idle.clone()].into(),
        };

        let picked = pool.select_worker("small-model:q4", 6_000_000_000).unwrap();
        assert_eq!(picked.gpu.ordinal, 1);
    }

    /// Regression guard for the happy path — both GPUs are idle and empty.
    /// The strategy says "prefer the smallest GPU that fits" to spread
    /// hot models across free cards.
    #[test]
    fn select_worker_spreads_to_smallest_fitting_idle_gpu() {
        let (big, _big_rx) = test_worker(0, 24_000_000_000);
        let (small, _small_rx) = test_worker(1, 12_000_000_000);

        let pool = GpuPool {
            workers: vec![big.clone(), small.clone()].into(),
        };

        // A 6GB model fits on both — should pick the smaller card.
        let picked = pool.select_worker("flux-dev:q4", 6_000_000_000).unwrap();
        assert_eq!(picked.gpu.ordinal, 1);
    }

    /// If both eligible GPUs are busy with *other* models, fall back to
    /// the "most headroom" tier instead of deadlocking.
    #[test]
    fn select_worker_falls_back_when_all_gpus_busy_with_other_models() {
        let (a, _a_rx) = test_worker(0, 24_000_000_000);
        let (b, _b_rx) = test_worker(1, 12_000_000_000);
        a.in_flight.store(1, Ordering::SeqCst);
        b.in_flight.store(1, Ordering::SeqCst);

        let pool = GpuPool {
            workers: vec![a.clone(), b.clone()].into(),
        };

        let picked = pool.select_worker("new-model", 6_000_000_000).unwrap();
        // Both busy → "most headroom" — the larger GPU wins.
        assert_eq!(picked.gpu.ordinal, 0);
    }

    /// A second job for the model that's RUNNING on GPU 0 must spread to an
    /// idle empty sibling instead of queueing behind the warm-but-busy card.
    /// The cold load is paid once; afterwards that sibling is warm too and
    /// tier 1 takes over. (Previously warm-busy beat idle-empty, so a 4-GPU
    /// box serialized same-model jobs on one card while three sat idle.)
    #[test]
    fn select_worker_spreads_same_model_to_idle_gpu_over_busy_warm_worker() {
        let (warm_busy, _warm_busy_rx) = test_worker(0, 24_000_000_000);
        let (cold_idle, _cold_idle_rx) = test_worker(1, 24_000_000_000);

        warm_busy.in_flight.store(1, Ordering::SeqCst);
        *warm_busy.active_generation.write().unwrap() = Some(ActiveGeneration {
            model: "flux-dev:q4".to_string(),
            prompt_sha256: String::new(),
            started_at_unix_ms: 0,
            started_at: Instant::now(),
        });

        let pool = GpuPool {
            workers: vec![warm_busy.clone(), cold_idle.clone()].into(),
        };

        let picked = pool
            .select_worker("flux-dev:q4", 6_000_000_000)
            .expect("idle sibling should be preferred");
        assert_eq!(
            picked.gpu.ordinal, 1,
            "same-model job must run in parallel on the idle GPU, not queue \
             behind the busy warm one"
        );
    }

    /// The spread only happens when the idle GPU can actually hold the
    /// model: if no idle card fits, queueing behind the warm busy worker
    /// beats cold-loading onto a card that would have to offload.
    #[test]
    fn select_worker_queues_behind_warm_worker_when_no_idle_gpu_fits() {
        let (warm_busy, _warm_busy_rx) = test_worker(0, 48_000_000_000);
        let (small_idle, _small_idle_rx) = test_worker(1, 12_000_000_000);

        warm_busy.in_flight.store(1, Ordering::SeqCst);
        *warm_busy.active_generation.write().unwrap() = Some(ActiveGeneration {
            model: "ltx-2.3-22b-dev:fp8".to_string(),
            prompt_sha256: String::new(),
            started_at_unix_ms: 0,
            started_at: Instant::now(),
        });

        let pool = GpuPool {
            workers: vec![warm_busy.clone(), small_idle.clone()].into(),
        };

        let picked = pool
            .select_worker("ltx-2.3-22b-dev:fp8", 34_000_000_000)
            .expect("a worker should be selected");
        assert_eq!(
            picked.gpu.ordinal, 0,
            "a too-small idle GPU must not win over the warm busy worker"
        );
    }

    #[test]
    fn resolve_explicit_placement_gpu_accepts_single_worker_ordinal() {
        let (worker, _rx) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker].into(),
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(1),
                ..AdvancedPlacement::default()
            }),
        };

        assert_eq!(
            pool.resolve_explicit_placement_gpu(Some(&placement))
                .unwrap(),
            Some(1)
        );
    }

    #[test]
    fn resolve_explicit_placement_gpu_accepts_durable_device_id() {
        let (worker, _rx) = test_worker(3, 24_000_000_000);
        let stable_id = worker.gpu.stable_id.clone().expect("test stable id");
        let pool = GpuPool {
            workers: vec![worker].into(),
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::device(stable_id),
                ..AdvancedPlacement::default()
            }),
        };

        assert_eq!(
            pool.resolve_explicit_placement_gpu(Some(&placement))
                .unwrap(),
            Some(3)
        );
    }

    #[test]
    fn resolve_explicit_placement_gpu_rejects_unavailable_durable_device_id() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker].into(),
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::device("cuda:ffffffffffffffffffffffffffffffff"),
            advanced: None,
        };

        let error = pool
            .resolve_explicit_placement_gpu(Some(&placement))
            .unwrap_err();
        assert!(error.contains("is not available"), "{error}");
    }

    #[test]
    fn stable_and_legacy_pins_to_same_worker_do_not_conflict() {
        let (worker, _rx) = test_worker(2, 24_000_000_000);
        let stable_id = worker.gpu.stable_id.clone().expect("test stable id");
        let pool = GpuPool {
            workers: vec![worker].into(),
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::gpu(2),
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::device(stable_id),
                ..AdvancedPlacement::default()
            }),
        };

        assert_eq!(
            pool.resolve_explicit_placement_gpu(Some(&placement))
                .unwrap(),
            Some(2)
        );
    }

    #[test]
    fn resolve_explicit_placement_gpu_rejects_cross_gpu_requests() {
        let (worker0, _rx0) = test_worker(0, 24_000_000_000);
        let (worker1, _rx1) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker0, worker1].into(),
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::gpu(0),
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(1),
                ..AdvancedPlacement::default()
            }),
        };

        let err = pool
            .resolve_explicit_placement_gpu(Some(&placement))
            .unwrap_err();
        assert!(err.contains("one GPU ordinal per request"), "{err}");
    }

    #[test]
    fn resolve_explicit_placement_gpu_rejects_ordinals_outside_pool() {
        let (worker1, _rx1) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker1].into(),
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(0),
                ..AdvancedPlacement::default()
            }),
        };

        let err = pool
            .resolve_explicit_placement_gpu(Some(&placement))
            .unwrap_err();
        assert!(err.contains("gpu:0"), "{err}");
        assert!(err.contains("[1]"), "{err}");
    }

    /// `is_degraded()` lazily resets the failure counter when the cooldown
    /// has expired. Without this, a worker that took 3 historical failures
    /// would re-degrade on the very first post-cooldown failure (because
    /// `consecutive_failures` was still ≥ 3 from before, even though the
    /// time-based gate had already opened back up).
    #[test]
    fn is_degraded_clears_counter_when_cooldown_has_expired() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        // Simulate "cooldown expired 1 second ago".
        *worker.degraded_until.write().unwrap() =
            Some(Instant::now() - std::time::Duration::from_secs(1));

        assert!(
            !worker.is_degraded(),
            "expired cooldown must mark the worker as healthy again",
        );
        assert_eq!(
            worker.consecutive_failures.load(Ordering::SeqCst),
            0,
            "expired cooldown must lazy-reset the failure counter so a \
             single post-cooldown failure doesn't immediately re-degrade",
        );
        assert!(
            worker.degraded_until.read().unwrap().is_none(),
            "expired cooldown must clear the timestamp",
        );
    }

    #[test]
    fn is_degraded_respects_active_cooldown() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        // Cooldown still active for another 60s.
        *worker.degraded_until.write().unwrap() =
            Some(Instant::now() + std::time::Duration::from_secs(60));

        assert!(
            worker.is_degraded(),
            "active cooldown must keep the worker degraded",
        );
        assert_eq!(
            worker.consecutive_failures.load(Ordering::SeqCst),
            3,
            "active cooldown must NOT reset the counter",
        );
    }

    #[test]
    fn model_oom_on_sibling_gpu_marks_model_unschedulable() {
        let _guard = MODEL_CUDA_OOM_TEST_LOCK.lock().unwrap();
        clear_model_cuda_ooms_for_tests();
        let model = "flux2-klein-9b:bf16";

        let first = record_model_cuda_oom(model, 0);
        assert!(
            !first.is_unschedulable(),
            "first OOM only records the failed ordinal"
        );
        assert!(
            model_unschedulable_message(model).is_none(),
            "a single-GPU OOM should not cool down the model yet"
        );

        let second = record_model_cuda_oom(model, 1);
        assert!(
            second.is_unschedulable(),
            "OOM on a sibling GPU should mark the model unschedulable"
        );
        let msg = model_unschedulable_message(model).expect("cooldown message");
        assert!(msg.contains(model), "{msg}");
        assert!(msg.contains("temporarily unschedulable"), "{msg}");

        clear_model_cuda_ooms_for_tests();
    }

    #[test]
    fn failed_model_ordinals_can_be_skipped_before_cooldown() {
        let _guard = MODEL_CUDA_OOM_TEST_LOCK.lock().unwrap();
        clear_model_cuda_ooms_for_tests();
        let (failed, _failed_rx) = test_worker(0, 24_000_000_000);
        let (untested, _untested_rx) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![failed, untested.clone()].into(),
        };
        let model = "flux2-klein-9b:bf16";

        record_model_cuda_oom(model, 0);
        let skip = failed_ordinals_for_model(model);
        let picked = pool
            .select_worker_excluding(model, 32_000_000_000, &skip)
            .expect("sibling GPU should be tried before cooldown");

        assert_eq!(picked.gpu.ordinal, untested.gpu.ordinal);
        clear_model_cuda_ooms_for_tests();
    }
}
