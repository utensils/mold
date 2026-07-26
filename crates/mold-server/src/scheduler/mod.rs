//! Runtime adapter between the pure scheduler and GPU owner threads.
//!
//! The coordinator is the sole owner of unstarted generation work. Worker
//! channels are rendezvous transports: a worker must publish a new Ready
//! generation before the coordinator can issue exactly one fenced grant.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mold_scheduler::{
    Backend, BlockedReason, CandidatePlacement, DeviceActivity, DeviceAdminState, DeviceHealth,
    DeviceId, DeviceSnapshot, EstimateBucket, EstimateKey, EstimateObservation, EstimateStore,
    ExecutionFingerprint, GrantValidationSnapshot, HostMemorySnapshot, Plan, Planner,
    PlannerSnapshot, PriorityClass, StaticEstimate, WorkId, WorkSnapshot,
};

use crate::gpu_pool::{GpuJob, GpuWorker, LeaseGrant, OwnerWork};
use crate::state::{AppState, GenerationJob, SseMessage};

const REPLAN_DEBOUNCE: Duration = Duration::from_secs(2);
const REPLAN_MAX_DELAY: Duration = Duration::from_secs(5);
const RECONCILE_INTERVAL: Duration = Duration::from_millis(10);
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const MIN_TRANSIENT_HOST_RAM: u64 = 64 * 1024 * 1024;
const MAX_PLAN_INVALIDATIONS: u8 = 3;
const PLAN_INVALIDATION_BACKOFF_MS: u64 = 25;
const PREPARATION_REFRESH_STABILITY_MS: u64 = 1_000;
const PREPARATION_RETRY_BASE_MS: u64 = 250;
const PREPARATION_RETRY_MAX_MS: u64 = 5_000;
const PREPARATION_CAPACITY_DELTA_BYTES: u64 = 2 << 30;
const MAX_DISPATCH_REPLANS_PER_TURN: u8 = 3;
const DISPATCH_RETRY_BASE_MS: u64 = 25;
const DISPATCH_RETRY_MAX_MS: u64 = 1_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeaseFence {
    pub work_id: String,
    pub device_id: String,
    pub owner_epoch: u64,
    pub state_version: u64,
    pub plan_version: u64,
    pub worker_generation: u64,
    pub memory_sample_generation: u64,
    pub memory_ledger_sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LeaseRejection {
    StaleWorkerGeneration,
    FatalCuda,
    PlanInvalidated(crate::execution_plan::ExecutionPlanError),
}

pub enum WorkerEvent {
    Ready {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
        worker_generation: u64,
    },
    StartFailed {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
        error: String,
    },
    Accepted {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
        worker_generation: u64,
        work_id: String,
        plan_version: u64,
    },
    AllocationCommitted {
        device_id: String,
        work_id: String,
        owner_epoch: u64,
        worker_generation: u64,
    },
    FollowupReady {
        work: Box<ScheduledOwnerWork>,
    },
    Rejected {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
        worker_generation: u64,
        grant: Box<LeaseGrant>,
        reason: LeaseRejection,
    },
    Completed {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
        worker_generation: u64,
        successful: bool,
        load_ms: Option<u64>,
    },
    Stopped {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
    },
}

pub struct ScheduledOwnerWork {
    pub id: String,
    pub model_fingerprint: String,
    pub estimated_vram_bytes: u64,
    pub estimated_host_ram_bytes: u64,
    pub hard_ordinal: Option<usize>,
    pub priority: PriorityClass,
    pub work: OwnerWork,
}

impl ScheduledOwnerWork {
    pub fn new(
        id: impl Into<String>,
        model_fingerprint: impl Into<String>,
        estimated_vram_bytes: u64,
        work: OwnerWork,
    ) -> Self {
        Self {
            id: id.into(),
            model_fingerprint: model_fingerprint.into(),
            estimated_vram_bytes,
            estimated_host_ram_bytes: MIN_TRANSIENT_HOST_RAM,
            hard_ordinal: None,
            priority: PriorityClass::User,
            work,
        }
    }

    pub fn with_hard_ordinal(mut self, ordinal: Option<usize>) -> Self {
        self.hard_ordinal = ordinal;
        self
    }

    pub fn with_priority(mut self, priority: PriorityClass) -> Self {
        self.priority = priority;
        self
    }
}

#[derive(Clone)]
pub struct ScheduledWorkHandle {
    tx: Option<tokio::sync::mpsc::Sender<ScheduledOwnerWork>>,
    dispatch_mode: crate::dispatch_mode::DispatchMode,
    v2_authoritative: bool,
    observes_v2_decisions: bool,
    observations: Arc<crate::dispatch_mode::DispatchObservationRecorder>,
    latest_plan: Arc<std::sync::RwLock<Option<mold_core::QueuePlan>>>,
}

impl Default for ScheduledWorkHandle {
    fn default() -> Self {
        Self {
            tx: None,
            dispatch_mode: crate::dispatch_mode::DispatchMode::V2,
            v2_authoritative: false,
            observes_v2_decisions: false,
            observations: Arc::new(crate::dispatch_mode::DispatchObservationRecorder::default()),
            latest_plan: Arc::new(std::sync::RwLock::new(None)),
        }
    }
}

impl ScheduledWorkHandle {
    pub fn new(tx: tokio::sync::mpsc::Sender<ScheduledOwnerWork>) -> Self {
        Self::for_mode(tx, crate::dispatch_mode::DispatchMode::V2)
    }

    pub fn for_mode(
        tx: tokio::sync::mpsc::Sender<ScheduledOwnerWork>,
        dispatch_mode: crate::dispatch_mode::DispatchMode,
    ) -> Self {
        Self::for_runtime(
            tx,
            dispatch_mode,
            dispatch_mode.owns_v2_workers(),
            dispatch_mode.records_v2_observations(),
        )
    }

    pub fn for_runtime(
        tx: tokio::sync::mpsc::Sender<ScheduledOwnerWork>,
        dispatch_mode: crate::dispatch_mode::DispatchMode,
        v2_authoritative: bool,
        observes_v2_decisions: bool,
    ) -> Self {
        Self {
            tx: Some(tx),
            dispatch_mode,
            v2_authoritative,
            observes_v2_decisions,
            observations: Arc::new(crate::dispatch_mode::DispatchObservationRecorder::default()),
            latest_plan: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    pub const fn dispatch_mode(&self) -> crate::dispatch_mode::DispatchMode {
        self.dispatch_mode
    }

    pub const fn v2_authoritative(&self) -> bool {
        self.v2_authoritative
    }

    pub const fn observes_v2_decisions(&self) -> bool {
        self.observes_v2_decisions
    }

    pub fn observations(&self) -> &crate::dispatch_mode::DispatchObservationRecorder {
        &self.observations
    }

    pub fn latest_plan(&self) -> Option<mold_core::QueuePlan> {
        self.latest_plan
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub async fn submit(&self, work: ScheduledOwnerWork) -> Result<(), String> {
        let Some(tx) = &self.tx else {
            return Err("GPU scheduler is unavailable".to_string());
        };
        tx.send(work)
            .await
            .map_err(|_| "GPU scheduler is shutting down".to_string())
    }
}

pub fn worker_device_id(worker: &GpuWorker) -> String {
    worker
        .gpu
        .stable_id
        .clone()
        // Selected production workers always have a driver UUID. The
        // process-local fallback keeps synthetic CPU-only tests usable
        // without pretending an ordinal is durable identity.
        .unwrap_or_else(|| format!("runtime:gpu:{}", worker.gpu.ordinal))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReadyWorker {
    ordinal: usize,
    owner_epoch: u64,
    generation: u64,
}

#[derive(Clone, Debug)]
struct ActiveLease {
    work_id: String,
    owner_epoch: u64,
    plan_version: u64,
    worker_generation: u64,
    accepted: bool,
    previous_target: Option<usize>,
    estimated_finish_ms: u64,
    ready_at_ms: u64,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    started_at: Instant,
    estimate_key: EstimateKey,
}

struct PendingGeneration {
    job: GenerationJob,
    ready_at_ms: u64,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    preparation: PreparationState,
    prepared_inputs: Option<crate::execution_plan::PreparedExecutionInputs>,
    retry_not_before_ms: Option<u64>,
    preparation_retry_attempts: u8,
    preparation_refresh_observation: Option<PreparationRefreshObservation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparationRefreshObservation {
    signature: Vec<(String, i8)>,
    first_observed_ms: u64,
}

fn capacity_refresh_direction(
    prepared_available: u64,
    current_available: u64,
    capacity_sensitive: bool,
) -> Option<i8> {
    if !capacity_sensitive
        || current_available.abs_diff(prepared_available) < PREPARATION_CAPACITY_DELTA_BYTES
    {
        return None;
    }
    if u128::from(current_available) * 5 <= u128::from(prepared_available) * 4 {
        Some(-1)
    } else if u128::from(current_available) * 4 >= u128::from(prepared_available) * 5 {
        Some(1)
    } else {
        None
    }
}

fn observe_preparation_refresh(
    observation: &mut Option<PreparationRefreshObservation>,
    signature: Vec<(String, i8)>,
    now_ms: u64,
    delay_ms: u64,
) -> bool {
    match observation {
        Some(current) if current.signature == signature => {
            now_ms.saturating_sub(current.first_observed_ms) >= delay_ms
        }
        slot => {
            *slot = Some(PreparationRefreshObservation {
                signature,
                first_observed_ms: now_ms,
            });
            false
        }
    }
}

fn dispatch_retry_delay_ms(round: u8) -> u64 {
    DISPATCH_RETRY_BASE_MS
        .saturating_mul(1_u64 << u32::from(round.saturating_sub(1).min(6)))
        .min(DISPATCH_RETRY_MAX_MS)
}

#[derive(Debug)]
enum GenerationPlanFailure {
    Transient(String),
    StalePreparation(String),
    Terminal(crate::execution_plan::ExecutionPlanError),
}

impl std::fmt::Display for GenerationPlanFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient(error) => formatter.write_str(error),
            Self::StalePreparation(error) => formatter.write_str(error),
            Self::Terminal(error) => error.fmt(formatter),
        }
    }
}

struct PendingOwnerWork {
    model_fingerprint: String,
    estimated_vram_bytes: u64,
    estimated_host_ram_bytes: u64,
    hard_ordinal: Option<usize>,
    priority: PriorityClass,
    queue_rank: u64,
    ready_at_ms: u64,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    work: OwnerWork,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparationState {
    Needed,
    Preparing,
    Ready,
}

enum PreparationEvent {
    Ready {
        work_id: String,
        prepared: PreparedGeneration,
    },
    Failed {
        work_id: String,
        error: String,
    },
}

#[derive(Clone, Debug, Default)]
struct PreparedGeneration {
    expanded_prompt: Option<String>,
    execution_inputs: Option<crate::execution_plan::PreparedExecutionInputs>,
}

type PreparationFuture = Pin<Box<dyn Future<Output = Result<PreparedGeneration, String>> + Send>>;

trait DependencyPreparer: Send + Sync {
    fn prepare(
        &self,
        state: AppState,
        work_id: String,
        request: mold_core::GenerateRequest,
        progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    ) -> PreparationFuture;
}

struct PostUpscalePreparer;

impl DependencyPreparer for PostUpscalePreparer {
    fn prepare(
        &self,
        state: AppState,
        work_id: String,
        request: mold_core::GenerateRequest,
        progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    ) -> PreparationFuture {
        Box::pin(async move {
            crate::queue::ensure_post_upscale_model_downloaded(&state, &request, progress.as_ref())
                .await?;
            let execution_inputs = crate::variant_dependencies::prepare_execution_inputs(
                &state,
                &work_id,
                &request,
                progress.as_ref(),
            )
            .await?;
            if request.expand != Some(true) {
                return Ok(PreparedGeneration {
                    expanded_prompt: None,
                    execution_inputs: Some(execution_inputs),
                });
            }
            let config = state.config.read().await.clone();
            let settings = config.expand.clone().with_env_overrides();
            if !settings.is_local() {
                // API-backed expansion remains CPU/network work and is
                // resolved before queue admission by the route.
                return Ok(PreparedGeneration {
                    expanded_prompt: None,
                    execution_inputs: Some(execution_inputs),
                });
            }
            let family = config
                .resolved_model_config(&request.model)
                .family
                .or_else(|| {
                    mold_core::manifest::find_manifest(&request.model)
                        .map(|manifest| manifest.family.clone())
                })
                .unwrap_or_else(|| "flux".to_string());
            let expand_config = settings.to_expand_config(&family, 1);
            let preferred_gpu = state
                .gpu_pool
                .resolve_explicit_placement_gpu(request.placement.as_ref())?;
            let utility_id = format!("{work_id}::prompt-expansion");
            let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
            let utility = ScheduledOwnerWork::new(
                utility_id.clone(),
                settings.model.clone(),
                6_000_000_000,
                OwnerWork::PromptExpansion(Box::new(crate::gpu_pool::PromptExpansionJob {
                    id: utility_id,
                    config,
                    settings,
                    prompt: request.prompt.clone(),
                    expand_config,
                    result_tx,
                })),
            )
            .with_hard_ordinal(preferred_gpu);
            state.scheduled_work.submit(utility).await?;

            let registry_notify = state.job_registry.mutation_notifier();
            let result = loop {
                tokio::select! {
                    result = &mut result_rx => {
                        break result
                            .map_err(|_| "prompt expansion owner worker dropped its result".to_string())?;
                    }
                    _ = registry_notify.notified() => {
                        if state.job_registry.entry(&work_id).is_none() {
                            return Err(format!(
                                "generation job {work_id} was cancelled during prompt expansion"
                            ));
                        }
                    }
                }
            }?;
            Ok(PreparedGeneration {
                expanded_prompt: result.expanded.first().cloned(),
                execution_inputs: Some(execution_inputs),
            })
        })
    }
}

#[derive(Clone, Debug)]
struct ReplanWindow {
    dirty_since: Option<Instant>,
    last_dirty: Option<Instant>,
    dirty_through_version: u64,
}

impl ReplanWindow {
    fn new() -> Self {
        Self {
            dirty_since: None,
            last_dirty: None,
            dirty_through_version: 0,
        }
    }

    fn mark_dirty(&mut self, now: Instant, state_version: u64) {
        self.dirty_since.get_or_insert(now);
        self.last_dirty = Some(now);
        self.dirty_through_version = self.dirty_through_version.max(state_version);
    }

    fn deadline(&self) -> Option<Instant> {
        Some(
            self.last_dirty?
                .checked_add(REPLAN_DEBOUNCE)?
                .min(self.dirty_since?.checked_add(REPLAN_MAX_DELAY)?),
        )
    }

    fn due(&self, now: Instant) -> bool {
        self.deadline().is_some_and(|deadline| now >= deadline)
    }

    fn clear_through(&mut self, planned_state_version: u64) {
        if self.dirty_since.is_some() && self.dirty_through_version <= planned_state_version {
            self.dirty_since = None;
            self.last_dirty = None;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HostMemoryReading {
    total_bytes: u64,
    available_bytes: u64,
}

trait HostMemorySampler: Send + Sync {
    fn sample(&self) -> HostMemoryReading;
}

struct SystemHostMemorySampler;

impl HostMemorySampler for SystemHostMemorySampler {
    fn sample(&self) -> HostMemoryReading {
        let ram = crate::resources::ram_snapshot();
        HostMemoryReading {
            total_bytes: ram.total,
            available_bytes: ram
                .available
                .unwrap_or_else(|| ram.total.saturating_sub(ram.used)),
        }
    }
}

#[derive(Clone)]
struct HostMemoryLedger {
    sampler: Arc<dyn HostMemorySampler>,
    sample: Option<MemorySample>,
    sequence: u64,
    reservations: BTreeMap<String, HostReservation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MemorySample {
    generation: u64,
    collection_started_sequence: u64,
    total_bytes: u64,
    available_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReservationState {
    Reserved,
    CommittedAfterSample { commit_sequence: u64 },
    ReflectedBySample,
}

#[derive(Clone, Debug)]
struct HostReservation {
    bytes: u64,
    state: ReservationState,
}

impl HostMemoryLedger {
    fn new(sampler: Arc<dyn HostMemorySampler>) -> Self {
        Self {
            sampler,
            sample: None,
            sequence: 0,
            reservations: BTreeMap::new(),
        }
    }

    fn collect_now(&mut self) {
        self.sequence = self.sequence.saturating_add(1);
        let collection_started_sequence = self.sequence;
        let reading = self.sampler.sample();
        self.publish_sample(
            collection_started_sequence,
            reading.total_bytes,
            reading.available_bytes,
        );
    }

    #[cfg(test)]
    fn begin_collection(&mut self) -> u64 {
        self.sequence = self.sequence.saturating_add(1);
        self.sequence
    }

    fn publish_sample(
        &mut self,
        collection_started_sequence: u64,
        total_bytes: u64,
        available_bytes: u64,
    ) {
        let generation = self
            .sample
            .map_or(1, |sample| sample.generation.saturating_add(1));
        self.sample = Some(MemorySample {
            generation,
            collection_started_sequence,
            total_bytes,
            available_bytes,
        });
        // Only allocations committed before collection began are guaranteed
        // to be present in `available_bytes`. A concurrent/later commit stays
        // charged until a following sample proves reflection.
        for reservation in self.reservations.values_mut() {
            if matches!(
                reservation.state,
                ReservationState::CommittedAfterSample { commit_sequence }
                    if commit_sequence < collection_started_sequence
            ) {
                reservation.state = ReservationState::ReflectedBySample;
            }
        }
        self.sequence = self.sequence.saturating_add(1);
    }

    fn bytes_accepted_after_sample_started(&self) -> u64 {
        self.reservations
            .values()
            .filter(|reservation| reservation.state != ReservationState::ReflectedBySample)
            .map(|reservation| reservation.bytes)
            .sum()
    }

    fn safety_floor_bytes(&self) -> u64 {
        self.sample
            .map(|sample| (sample.total_bytes.saturating_mul(15) / 100).max(8 << 30))
            .unwrap_or(u64::MAX)
    }

    fn headroom_bytes(&self) -> u64 {
        let Some(sample) = self.sample else {
            return 0;
        };
        sample.available_bytes.saturating_sub(
            self.safety_floor_bytes()
                .saturating_add(self.bytes_accepted_after_sample_started()),
        )
    }

    fn snapshot(&self) -> HostMemorySnapshot {
        HostMemorySnapshot {
            headroom_bytes: self.headroom_bytes(),
            sample_generation: self.sample.map_or(0, |sample| sample.generation),
            ledger_sequence: self.sequence,
        }
    }

    fn try_reserve(
        &mut self,
        plan: &Plan,
        state_version: u64,
        plan_version: u64,
    ) -> Result<(), GrantFenceError> {
        plan.validate_for_grant(
            state_version,
            plan_version,
            self.sample.map_or(0, |sample| sample.generation),
            self.sequence,
        )
        .map_err(|_| GrantFenceError::StalePlan)?;
        if plan.reservation.total_host_ram_bytes > self.headroom_bytes() {
            return Err(GrantFenceError::InsufficientHostRam);
        }
        for item in &plan.reservation.items {
            self.reservations.insert(
                item.work_id.to_string(),
                HostReservation {
                    bytes: item.host_ram_bytes,
                    state: ReservationState::Reserved,
                },
            );
        }
        self.sequence = self.sequence.saturating_add(1);
        Ok(())
    }

    fn commit(&mut self, work_id: &str) {
        if !self
            .reservations
            .get(work_id)
            .is_some_and(|reservation| reservation.state == ReservationState::Reserved)
        {
            return;
        }
        self.sequence = self.sequence.saturating_add(1);
        if let Some(reservation) = self.reservations.get_mut(work_id) {
            reservation.state = ReservationState::CommittedAfterSample {
                commit_sequence: self.sequence,
            };
        }
    }

    fn release(&mut self, work_id: &str) {
        self.release_matching(std::iter::once(work_id));
    }

    fn release_matching<'a>(&mut self, work_ids: impl IntoIterator<Item = &'a str>) {
        let mut changed = false;
        for work_id in work_ids {
            changed |= self.reservations.remove(work_id).is_some();
        }
        if changed {
            self.sequence = self.sequence.saturating_add(1);
        }
    }

    fn settle_partial_matching<'a>(
        &mut self,
        granted: impl IntoIterator<Item = &'a str>,
        ungranted: impl IntoIterator<Item = &'a str>,
    ) {
        let mut changed = false;
        // Granted items remain Reserved until the worker reaches the actual
        // allocation boundary and sends AllocationCommitted.
        let _ = granted.into_iter().count();
        for work_id in ungranted {
            changed |= self.reservations.remove(work_id).is_some();
        }
        if changed {
            self.sequence = self.sequence.saturating_add(1);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GrantFenceError {
    StalePlan,
    InsufficientHostRam,
    DuplicateDeviceLease,
    WorkerNotReady,
    StaleWorkerGeneration,
}

fn validate_worker_grant(
    ready: &BTreeMap<String, ReadyWorker>,
    leases: &BTreeMap<String, ActiveLease>,
    device_id: &str,
    worker_generation: u64,
) -> Result<ReadyWorker, GrantFenceError> {
    if leases.contains_key(device_id) {
        return Err(GrantFenceError::DuplicateDeviceLease);
    }
    let ready = ready
        .get(device_id)
        .cloned()
        .ok_or(GrantFenceError::WorkerNotReady)?;
    if ready.generation != worker_generation {
        return Err(GrantFenceError::StaleWorkerGeneration);
    }
    Ok(ready)
}

type DeviceEventSignature = Vec<(
    String,
    bool,
    mold_core::DeviceAdminState,
    mold_core::DeviceHealth,
    mold_core::DeviceActivity,
    bool,
    Option<String>,
    Vec<String>,
    Option<String>,
)>;

/// Semantic device transitions only. Raw telemetry and plan membership have
/// dedicated streams/events and must not generate 1 Hz `/api/events` storms.
fn device_event_signature(state: &mold_core::DeviceState) -> DeviceEventSignature {
    state
        .devices
        .iter()
        .map(|device| {
            (
                device.id.clone(),
                device.desired_enabled,
                device.admin_state,
                device.health,
                device.activity,
                device.schedulable,
                device.unschedulable_reason.clone(),
                device.loaded_models.clone(),
                device.active_work_id.clone(),
            )
        })
        .collect()
}

fn queue_plan_semantically_equal(
    left: &mold_core::QueuePlan,
    right: &mold_core::QueuePlan,
) -> bool {
    fn normalized(plan: &mold_core::QueuePlan) -> mold_core::QueuePlan {
        let mut plan = plan.clone();
        plan.plan_version = 0;
        plan.state_version = 0;
        plan.dirty_since_unix_ms = None;
        plan.next_replan_at_unix_ms = None;
        for work in &mut plan.work_items {
            let duration = work
                .estimated_start_unix_ms
                .zip(work.estimated_finish_unix_ms)
                .map(|(start, finish)| finish.saturating_sub(start));
            work.estimated_start_unix_ms = work.estimated_start_unix_ms.map(|_| 0);
            work.estimated_finish_unix_ms = duration;
        }
        plan
    }
    normalized(left) == normalized(right)
}

struct Coordinator {
    state: AppState,
    planner: Planner,
    pending: BTreeMap<String, PendingGeneration>,
    pending_owner_work: BTreeMap<String, PendingOwnerWork>,
    ready: BTreeMap<String, ReadyWorker>,
    leases: BTreeMap<String, ActiveLease>,
    unavailable: BTreeSet<String>,
    state_version: u64,
    plan_version: u64,
    synthetic_id: u64,
    memory: HostMemoryLedger,
    dirty: ReplanWindow,
    preparer: Arc<dyn DependencyPreparer>,
    preparation_tx: tokio::sync::mpsc::UnboundedSender<PreparationEvent>,
    preparation_rx: tokio::sync::mpsc::UnboundedReceiver<PreparationEvent>,
    preparation_tasks: tokio::task::JoinSet<()>,
    last_queue_shape: Vec<(String, Option<usize>)>,
    last_registry_sequence: u64,
    last_paused: bool,
    last_worker_claims: BTreeMap<String, usize>,
    last_device_preferences_sequence: u64,
    last_device_event_signature: Option<DeviceEventSignature>,
    device_state_dirty: bool,
    plan_invalidations: BTreeMap<String, u8>,
    dispatch_retry_round: u8,
    dispatch_retry_not_before_ms: Option<u64>,
    estimates: EstimateStore,
    #[cfg(test)]
    before_grant_hook: Option<BeforeGrantHook>,
}

#[cfg(test)]
#[derive(Clone)]
struct BeforeGrantHook {
    plan_built: Arc<tokio::sync::Notify>,
    resume: Arc<tokio::sync::Notify>,
}

impl Coordinator {
    fn new(state: AppState) -> Self {
        Self::with_preparer_and_sampler(
            state,
            Arc::new(PostUpscalePreparer),
            Arc::new(SystemHostMemorySampler),
        )
    }

    fn with_preparer_and_sampler(
        state: AppState,
        preparer: Arc<dyn DependencyPreparer>,
        sampler: Arc<dyn HostMemorySampler>,
    ) -> Self {
        let mut memory = HostMemoryLedger::new(sampler);
        memory.collect_now();
        Self::with_preparer_and_memory(state, preparer, memory)
    }

    fn with_preparer_and_memory(
        state: AppState,
        preparer: Arc<dyn DependencyPreparer>,
        memory: HostMemoryLedger,
    ) -> Self {
        let (preparation_tx, preparation_rx) = tokio::sync::mpsc::unbounded_channel();
        let estimates = load_estimate_store(&state);
        Self {
            state,
            planner: Planner::default(),
            pending: BTreeMap::new(),
            pending_owner_work: BTreeMap::new(),
            ready: BTreeMap::new(),
            leases: BTreeMap::new(),
            unavailable: BTreeSet::new(),
            state_version: 0,
            plan_version: 0,
            synthetic_id: 0,
            memory,
            dirty: ReplanWindow::new(),
            preparer,
            preparation_tx,
            preparation_rx,
            preparation_tasks: tokio::task::JoinSet::new(),
            last_queue_shape: Vec::new(),
            last_registry_sequence: 0,
            last_paused: false,
            last_worker_claims: BTreeMap::new(),
            last_device_preferences_sequence: 0,
            last_device_event_signature: None,
            device_state_dirty: true,
            plan_invalidations: BTreeMap::new(),
            dispatch_retry_round: 0,
            dispatch_retry_not_before_ms: None,
            estimates,
            #[cfg(test)]
            before_grant_hook: None,
        }
    }

    fn mutate(&mut self, immediate: &mut bool) {
        self.state_version = self.state_version.saturating_add(1);
        self.dirty.mark_dirty(Instant::now(), self.state_version);
        *immediate = true;
    }

    fn defer_dispatch_retry(&mut self) {
        self.dispatch_retry_round = self.dispatch_retry_round.saturating_add(1);
        let backoff_ms = dispatch_retry_delay_ms(self.dispatch_retry_round);
        self.dispatch_retry_not_before_ms = Some(monotonic_ms().saturating_add(backoff_ms));
        self.dirty.mark_dirty(Instant::now(), self.state_version);
    }

    fn clear_dispatch_retry(&mut self) {
        self.dispatch_retry_round = 0;
        self.dispatch_retry_not_before_ms = None;
    }

    fn record_dispatch_progress(&mut self) {
        self.clear_dispatch_retry();
    }

    fn enqueue(&mut self, mut job: GenerationJob, immediate: &mut bool) {
        if job.id.is_empty() {
            self.synthetic_id = self.synthetic_id.saturating_add(1);
            job.id = format!("runtime-generation-{}", self.synthetic_id);
        }
        let id = job.id.clone();
        if let Some(error) = crate::gpu_pool::model_unschedulable_message(&job.request.model) {
            reject_generation(&self.state, job, error);
            return;
        }
        if let Err(error) = self
            .state
            .gpu_pool
            .resolve_explicit_placement_gpu(job.request.placement.as_ref())
        {
            reject_generation(&self.state, job, error);
            return;
        }
        if self.pending.contains_key(&id) || self.leases.values().any(|lease| lease.work_id == id) {
            reject_generation(
                &self.state,
                job,
                format!("duplicate generation job id {id}"),
            );
            return;
        }
        self.pending.insert(
            id,
            PendingGeneration {
                job,
                ready_at_ms: monotonic_ms(),
                bypass_count: 0,
                warm_wait_started_ms: None,
                preparation: PreparationState::Needed,
                prepared_inputs: None,
                retry_not_before_ms: None,
                preparation_retry_attempts: 0,
                preparation_refresh_observation: None,
            },
        );
        self.mutate(immediate);
    }

    fn enqueue_owner_work(&mut self, submission: ScheduledOwnerWork, immediate: &mut bool) {
        if submission.id != submission.work.id() {
            let payload_id = submission.work.id().to_string();
            submission.work.reject(format!(
                "scheduled work id '{}' did not match payload id '{}'",
                submission.id, payload_id
            ));
            return;
        }
        if submission.work.is_cancelled() {
            return;
        }
        if self.pending.contains_key(&submission.id)
            || self.pending_owner_work.contains_key(&submission.id)
            || self
                .leases
                .values()
                .any(|lease| lease.work_id == submission.id)
        {
            submission
                .work
                .reject(format!("duplicate scheduled work id {}", submission.id));
            return;
        }
        let queue_rank = self.synthetic_id;
        self.synthetic_id = self.synthetic_id.saturating_add(1);
        self.pending_owner_work.insert(
            submission.id,
            PendingOwnerWork {
                model_fingerprint: submission.model_fingerprint,
                estimated_vram_bytes: submission.estimated_vram_bytes,
                estimated_host_ram_bytes: submission.estimated_host_ram_bytes,
                hard_ordinal: submission.hard_ordinal,
                priority: submission.priority,
                queue_rank,
                ready_at_ms: monotonic_ms(),
                bypass_count: 0,
                warm_wait_started_ms: None,
                work: submission.work,
            },
        );
        self.mutate(immediate);
    }

    fn start_needed_preparations(&mut self) {
        let ids = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.preparation == PreparationState::Needed)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in ids {
            let Some(pending) = self.pending.get_mut(&id) else {
                continue;
            };
            pending.preparation = PreparationState::Preparing;
            let state = self.state.clone();
            let request = pending.job.request.clone();
            let progress = pending.job.progress_tx.clone();
            let preparer = self.preparer.clone();
            let tx = self.preparation_tx.clone();
            self.preparation_tasks.spawn(async move {
                let event = match preparer.prepare(state, id.clone(), request, progress).await {
                    Ok(prepared) => PreparationEvent::Ready {
                        work_id: id,
                        prepared,
                    },
                    Err(error) => PreparationEvent::Failed { work_id: id, error },
                };
                let _ = tx.send(event);
            });
        }
    }

    fn handle_preparation_event(&mut self, event: PreparationEvent, immediate: &mut bool) {
        match event {
            PreparationEvent::Ready { work_id, prepared } => {
                let Some(pending) = self.pending.get_mut(&work_id) else {
                    return;
                };
                if pending.preparation != PreparationState::Preparing {
                    return;
                }
                if let Some(expanded_prompt) = prepared.expanded_prompt {
                    pending.job.request.original_prompt = Some(pending.job.request.prompt.clone());
                    pending.job.request.prompt = expanded_prompt;
                }
                pending.prepared_inputs = prepared.execution_inputs;
                pending.preparation = PreparationState::Ready;
                pending.preparation_refresh_observation = None;
                if pending
                    .prepared_inputs
                    .as_ref()
                    .is_none_or(|inputs| inputs.retryable_device_failures.is_empty())
                {
                    pending.preparation_retry_attempts = 0;
                }
                self.mutate(immediate);
            }
            PreparationEvent::Failed { work_id, error } => {
                if let Some(pending) = self.pending.remove(&work_id) {
                    reject_generation(&self.state, pending.job, error);
                    self.mutate(immediate);
                }
            }
        }
    }

    async fn stop_preparations(&mut self) {
        self.preparation_tasks.abort_all();
        while self.preparation_tasks.join_next().await.is_some() {}
    }

    fn handle_worker_event(&mut self, event: WorkerEvent, immediate: &mut bool) {
        self.device_state_dirty = true;
        match event {
            WorkerEvent::Ready {
                device_id,
                ordinal,
                owner_epoch,
                worker_generation,
            } => {
                let was_starting = self.state.gpu_pool.workers.is_starting(&device_id);
                if !self
                    .state
                    .gpu_pool
                    .workers
                    .mark_ready(&device_id, owner_epoch)
                {
                    tracing::warn!(
                        device_id,
                        owner_epoch,
                        worker_generation,
                        "ignoring Ready from a stale GPU owner"
                    );
                    return;
                }
                if self.leases.contains_key(&device_id) {
                    tracing::warn!(
                        device_id,
                        worker_generation,
                        "ignoring Ready while device still owns a lease"
                    );
                    return;
                }
                // A transport Full can mark a device unavailable while leaving
                // its same-generation Ready record intact. Clear unavailable
                // before dedupe so a stale cancelled Drain wake cannot strand
                // the GPU forever.
                self.unavailable.remove(&device_id);
                if self.ready.get(&device_id).is_some_and(|ready| {
                    ready.owner_epoch == owner_epoch && ready.generation >= worker_generation
                }) {
                    return;
                }
                self.ready.insert(
                    device_id.clone(),
                    ReadyWorker {
                        ordinal,
                        owner_epoch,
                        generation: worker_generation,
                    },
                );
                if was_starting && !self.state.gpu_pool.workers.is_starting(&device_id) {
                    self.state
                        .events
                        .publish(mold_core::ServerEvent::DeviceStateChanged {
                            device_id: device_id.clone(),
                            desired_enabled: self.state.device_registry.desired_enabled(&device_id),
                            admin_state: mold_core::DeviceAdminState::Enabled,
                        });
                }
                self.mutate(immediate);
            }
            WorkerEvent::StartFailed {
                device_id,
                ordinal,
                owner_epoch,
                error,
            } => {
                let Some(start_was_announced) = self.state.gpu_pool.workers.mark_start_failed(
                    &device_id,
                    owner_epoch,
                    error.clone(),
                ) else {
                    tracing::warn!(
                        device_id,
                        ordinal,
                        owner_epoch,
                        %error,
                        "ignoring startup failure from a stale GPU owner"
                    );
                    return;
                };
                if self
                    .ready
                    .get(&device_id)
                    .is_some_and(|ready| ready.owner_epoch == owner_epoch)
                {
                    self.ready.remove(&device_id);
                }
                self.unavailable.insert(device_id.clone());
                if start_was_announced {
                    self.state
                        .events
                        .publish(mold_core::ServerEvent::DeviceStateChanged {
                            device_id: device_id.clone(),
                            desired_enabled: self.state.device_registry.desired_enabled(&device_id),
                            admin_state: mold_core::DeviceAdminState::Enabled,
                        });
                }
                tracing::error!(
                    device_id,
                    ordinal,
                    owner_epoch,
                    %error,
                    "GPU owner startup failed; device remains desired but unavailable"
                );
                self.mutate(immediate);
            }
            WorkerEvent::Accepted {
                device_id,
                ordinal,
                owner_epoch,
                worker_generation,
                work_id,
                plan_version,
            } => {
                let valid = self.leases.get_mut(&device_id).is_some_and(|lease| {
                    if lease.work_id == work_id
                        && lease.plan_version == plan_version
                        && lease.owner_epoch == owner_epoch
                        && lease.worker_generation == worker_generation
                    {
                        lease.accepted = true;
                        true
                    } else {
                        false
                    }
                });
                if !valid {
                    tracing::error!(
                        device_id,
                        ordinal,
                        owner_epoch,
                        worker_generation,
                        work_id,
                        plan_version,
                        "worker acknowledged an unknown or stale lease"
                    );
                }
            }
            WorkerEvent::AllocationCommitted {
                device_id,
                work_id,
                owner_epoch,
                worker_generation,
            } => {
                let valid = self.leases.get(&device_id).is_some_and(|lease| {
                    lease.work_id == work_id
                        && lease.owner_epoch == owner_epoch
                        && lease.worker_generation == worker_generation
                });
                if valid {
                    self.memory.commit(&work_id);
                    self.mutate(immediate);
                } else {
                    tracing::error!(
                        device_id,
                        work_id,
                        owner_epoch,
                        worker_generation,
                        "ignoring allocation commit for unknown lease"
                    );
                }
            }
            WorkerEvent::FollowupReady { work } => {
                self.enqueue_owner_work(*work, immediate);
            }
            WorkerEvent::Rejected {
                device_id,
                ordinal,
                owner_epoch,
                worker_generation,
                grant,
                reason,
            } => {
                tracing::warn!(
                    device_id,
                    ordinal,
                    owner_epoch,
                    worker_generation,
                    ?reason,
                    "worker rejected a fenced grant"
                );
                let rejected_work_id = grant.work.id().to_string();
                let valid = self.leases.get(&device_id).is_some_and(|lease| {
                    lease.work_id == rejected_work_id
                        && lease.owner_epoch == owner_epoch
                        && lease.worker_generation == worker_generation
                });
                if !valid {
                    tracing::error!(
                        device_id,
                        ordinal,
                        owner_epoch,
                        worker_generation,
                        work_id = %rejected_work_id,
                        "rejecting payload returned by an unknown or stale owner"
                    );
                    grant
                        .work
                        .reject("GPU owner returned work from a stale lifecycle epoch".to_string());
                    return;
                }
                // A rejection returns ownership of the transported payload
                // even when its fence metadata is stale or corrupt. Reclaim
                // the active lease by stable device/work identity, not by
                // mutable epoch/version fields; those are the reason the
                // worker rejected it. Never tear down a newer unrelated
                // lease merely because a delayed event names its device.
                let rejected_lease_device = self
                    .leases
                    .get(&device_id)
                    .filter(|lease| {
                        lease.work_id == rejected_work_id && lease.owner_epoch == owner_epoch
                    })
                    .map(|_| device_id.clone())
                    .or_else(|| {
                        self.leases
                            .get(&grant.fence.device_id)
                            .filter(|lease| lease.work_id == rejected_work_id)
                            .map(|_| grant.fence.device_id.clone())
                    });
                let rejected_lease = rejected_lease_device
                    .as_ref()
                    .and_then(|device| self.leases.remove(device));
                let previous_target = rejected_lease
                    .as_ref()
                    .and_then(|lease| lease.previous_target);
                let preserved_bypass_count = rejected_lease
                    .as_ref()
                    .map_or(0, |lease| lease.bypass_count);
                let preserved_warm_wait_started_ms = rejected_lease
                    .as_ref()
                    .and_then(|lease| lease.warm_wait_started_ms);
                let preserved_ready_at_ms = rejected_lease
                    .as_ref()
                    .map_or_else(monotonic_ms, |lease| lease.ready_at_ms);
                let work_id = rejected_work_id;
                if rejected_lease.is_some() {
                    self.memory.release(&work_id);
                    self.memory.collect_now();
                    let rejected_worker = rejected_lease_device.as_ref().and_then(|device| {
                        self.state
                            .gpu_pool
                            .workers
                            .iter()
                            .find(|worker| worker_device_id(worker) == *device)
                    });
                    if let Some(worker) = rejected_worker {
                        worker.release_in_flight();
                    }
                }
                let LeaseGrant { work, retry, .. } = *grant;
                match work {
                    OwnerWork::Generation(job) => {
                        let (generation_job, prepared_inputs) =
                            generation_and_prepared_from_gpu_job(*job);
                        if matches!(reason, LeaseRejection::FatalCuda) {
                            self.plan_invalidations.remove(&generation_job.id);
                            reject_generation(
                                &self.state,
                                generation_job,
                                "CUDA context is fatally poisoned; server restart required"
                                    .to_string(),
                            );
                        } else if let LeaseRejection::PlanInvalidated(error) = &reason {
                            let attempts = self
                                .plan_invalidations
                                .entry(generation_job.id.clone())
                                .or_default();
                            *attempts = attempts.saturating_add(1);
                            if *attempts >= MAX_PLAN_INVALIDATIONS {
                                let attempts = *attempts;
                                self.plan_invalidations.remove(&generation_job.id);
                                reject_generation(
                                    &self.state,
                                    generation_job,
                                    format!(
                                        "execution plan was invalidated {attempts} consecutive times; \
                                         refusing to retry: {error}"
                                    ),
                                );
                            } else {
                                let backoff_ms = PLAN_INVALIDATION_BACKOFF_MS
                                    .saturating_mul(1_u64 << u32::from(*attempts - 1));
                                self.state
                                    .job_registry
                                    .requeue_rejected_dispatch(&generation_job.id, previous_target);
                                self.pending.insert(
                                    generation_job.id.clone(),
                                    PendingGeneration {
                                        job: generation_job,
                                        ready_at_ms: preserved_ready_at_ms,
                                        bypass_count: preserved_bypass_count,
                                        warm_wait_started_ms: preserved_warm_wait_started_ms,
                                        preparation: PreparationState::Ready,
                                        prepared_inputs: prepared_inputs.clone(),
                                        retry_not_before_ms: Some(
                                            monotonic_ms().saturating_add(backoff_ms),
                                        ),
                                        preparation_retry_attempts: 0,
                                        preparation_refresh_observation: None,
                                    },
                                );
                            }
                        } else {
                            self.state
                                .job_registry
                                .requeue_rejected_dispatch(&generation_job.id, previous_target);
                            self.pending.insert(
                                generation_job.id.clone(),
                                PendingGeneration {
                                    job: generation_job,
                                    ready_at_ms: preserved_ready_at_ms,
                                    bypass_count: preserved_bypass_count,
                                    warm_wait_started_ms: preserved_warm_wait_started_ms,
                                    preparation: PreparationState::Ready,
                                    prepared_inputs,
                                    retry_not_before_ms: None,
                                    preparation_retry_attempts: 0,
                                    preparation_refresh_observation: None,
                                },
                            );
                        }
                    }
                    work => {
                        if matches!(reason, LeaseRejection::FatalCuda) {
                            work.reject(
                                "CUDA context is fatally poisoned; server restart required"
                                    .to_string(),
                            );
                        } else {
                            // A stale worker generation means this payload was
                            // never touched. Put it back into the authoritative
                            // ready set for the next plan.
                            let retry = retry.unwrap_or(crate::gpu_pool::OwnerWorkRetry {
                                model_fingerprint: format!("{:?}", work.kind()),
                                estimated_vram_bytes: 0,
                                estimated_host_ram_bytes: MIN_TRANSIENT_HOST_RAM,
                                hard_ordinal: None,
                                priority: PriorityClass::User,
                                queue_rank: self.synthetic_id,
                                ready_at_ms: preserved_ready_at_ms,
                                bypass_count: preserved_bypass_count,
                                warm_wait_started_ms: preserved_warm_wait_started_ms,
                            });
                            self.pending_owner_work.insert(
                                work_id,
                                PendingOwnerWork {
                                    model_fingerprint: retry.model_fingerprint,
                                    estimated_vram_bytes: retry.estimated_vram_bytes,
                                    estimated_host_ram_bytes: retry.estimated_host_ram_bytes,
                                    hard_ordinal: retry.hard_ordinal,
                                    priority: retry.priority,
                                    queue_rank: retry.queue_rank,
                                    ready_at_ms: retry.ready_at_ms,
                                    bypass_count: retry.bypass_count,
                                    warm_wait_started_ms: retry.warm_wait_started_ms,
                                    work,
                                },
                            );
                            self.synthetic_id = self.synthetic_id.saturating_add(1);
                        }
                    }
                }
                self.mutate(immediate);
            }
            WorkerEvent::Completed {
                device_id,
                ordinal,
                owner_epoch,
                worker_generation,
                successful,
                load_ms,
            } => {
                let valid = self.leases.get(&device_id).is_some_and(|lease| {
                    lease.owner_epoch == owner_epoch && lease.worker_generation == worker_generation
                });
                if valid {
                    let lease = self
                        .leases
                        .remove(&device_id)
                        .expect("validated lease must still exist");
                    self.memory.release(&lease.work_id);
                    self.memory.collect_now();
                    self.plan_invalidations.remove(&lease.work_id);
                    let vram_completion_sample_bytes =
                        self.state.resources.latest().and_then(|snapshot| {
                            snapshot
                                .gpus
                                .into_iter()
                                .find(|gpu| gpu.ordinal == ordinal)
                                .and_then(|gpu| gpu.vram_used_by_mold)
                        });
                    if successful {
                        self.observe_estimate(
                            lease.estimate_key,
                            EstimateObservation {
                                total_ms: lease
                                    .started_at
                                    .elapsed()
                                    .as_millis()
                                    .try_into()
                                    .unwrap_or(u64::MAX),
                                load_ms,
                                vram_completion_sample_bytes,
                                host_completion_sample_bytes: None,
                                observed_at_unix_s: unix_seconds(),
                            },
                        );
                    }
                } else {
                    tracing::warn!(
                        device_id,
                        ordinal,
                        owner_epoch,
                        worker_generation,
                        "ignoring completion from a stale GPU owner"
                    );
                }
                self.mutate(immediate);
            }
            WorkerEvent::Stopped {
                device_id,
                ordinal,
                owner_epoch,
            } => {
                if self
                    .ready
                    .get(&device_id)
                    .is_some_and(|ready| ready.owner_epoch == owner_epoch)
                {
                    self.ready.remove(&device_id);
                }
                if self
                    .leases
                    .get(&device_id)
                    .is_some_and(|lease| lease.owner_epoch == owner_epoch)
                {
                    let lease = self
                        .leases
                        .remove(&device_id)
                        .expect("exact owner lease must still exist");
                    self.memory.release(&lease.work_id);
                    self.memory.collect_now();
                    if self.state.job_registry.remove_if_present(&lease.work_id) {
                        self.state.queue.decrement();
                    }
                    if let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ordinal) {
                        worker.release_in_flight();
                    }
                    tracing::error!(
                        device_id,
                        ordinal,
                        owner_epoch,
                        work_id = %lease.work_id,
                        accepted = lease.accepted,
                        "GPU owner stopped before completing its exact lease"
                    );
                }
                let removed = self
                    .state
                    .gpu_pool
                    .workers
                    .wait_and_reap(&device_id, owner_epoch);
                if removed {
                    if self.state.device_registry.desired_enabled(&device_id) {
                        if let Ok(new_epoch) = self.state.gpu_pool.workers.start(&device_id) {
                            self.state
                                .events
                                .publish(mold_core::ServerEvent::DeviceStateChanged {
                                    device_id: device_id.clone(),
                                    desired_enabled: true,
                                    admin_state: mold_core::DeviceAdminState::Starting,
                                });
                            let _ = self
                                .state
                                .gpu_pool
                                .workers
                                .announce_start(&device_id, new_epoch);
                        }
                    } else {
                        self.state
                            .events
                            .publish(mold_core::ServerEvent::DeviceStateChanged {
                                device_id: device_id.clone(),
                                desired_enabled: false,
                                admin_state: mold_core::DeviceAdminState::Disabled,
                            });
                    }
                }
                self.mutate(immediate);
            }
        }
    }

    async fn handle_worker_event_serialized(&mut self, event: WorkerEvent, immediate: &mut bool) {
        if matches!(
            &event,
            WorkerEvent::Ready { .. }
                | WorkerEvent::StartFailed { .. }
                | WorkerEvent::Stopped { .. }
        ) {
            let mutation_fence = self.state.scheduler_mutation_fence.clone();
            let _mutation = mutation_fence.lock().await;
            self.handle_worker_event(event, immediate);
        } else {
            self.handle_worker_event(event, immediate);
        }
    }

    fn observe_estimate(&mut self, key: EstimateKey, observation: EstimateObservation) {
        self.estimates.observe(key.clone(), observation);
        let normalized = key.normalized();
        if normalized != key {
            self.estimates.observe(normalized.clone(), observation);
            self.persist_estimate(&normalized);
        }
        self.persist_estimate(&key);
    }

    fn persist_estimate(&self, key: &EstimateKey) {
        let Some(bucket) = self.estimates.exact(key) else {
            return;
        };
        let Some(db) = self.state.metadata_db.as_ref().as_ref() else {
            return;
        };
        let record = estimate_record(bucket);
        if let Err(error) = mold_db::SchedulerEstimates::new(db).upsert(&record) {
            tracing::warn!(
                estimate_key = %record.estimate_key,
                error = %format!("{error:#}"),
                "failed to persist learned scheduler estimate"
            );
        }
    }

    fn reconcile_external_mutations(&mut self, immediate: &mut bool) {
        let worker_claims = self
            .state
            .gpu_pool
            .workers
            .iter()
            .map(|worker| {
                (
                    worker_device_id(&worker),
                    worker.in_flight.load(Ordering::SeqCst),
                )
            })
            .collect::<BTreeMap<_, _>>();
        if worker_claims != self.last_worker_claims {
            self.last_worker_claims = worker_claims;
            self.mutate(immediate);
        }
        let device_preferences_sequence = self.state.device_registry.mutation_sequence();
        if device_preferences_sequence != self.last_device_preferences_sequence {
            self.last_device_preferences_sequence = device_preferences_sequence;
            self.device_state_dirty = true;
            self.mutate(immediate);
        }

        let listing = self.state.job_registry.snapshot();
        let queue_shape = listing
            .entries
            .iter()
            .filter(|entry| entry.state == crate::job_registry::JobLifecycle::Queued)
            .map(|entry| (entry.id.clone(), entry.target_gpu))
            .collect::<Vec<_>>();
        let registry_sequence = self.state.job_registry.mutation_sequence();
        let paused = self.state.queue_pause.is_paused();
        if registry_sequence != self.last_registry_sequence
            || queue_shape != self.last_queue_shape
            || paused != self.last_paused
        {
            self.last_queue_shape = queue_shape;
            self.last_registry_sequence = registry_sequence;
            self.last_paused = paused;
            self.mutate(immediate);
        }

        let cancelled = self
            .pending
            .iter()
            .filter(|(id, pending)| {
                pending.job.result_tx.is_closed()
                    || (!id.starts_with("runtime-generation-")
                        && self.state.job_registry.entry(id).is_none())
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in cancelled {
            if let Some(pending) = self.pending.remove(&id) {
                self.plan_invalidations.remove(&id);
                self.state.queue.decrement();
                let _ = pending.job.result_tx.send(Err(format!(
                    "generation job {id} was cancelled while queued"
                )));
                self.mutate(immediate);
            }
        }
        let cancelled_owner_work = self
            .pending_owner_work
            .iter()
            .filter(|(_, pending)| pending.work.is_cancelled())
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in cancelled_owner_work {
            if let Some(pending) = self.pending_owner_work.remove(&id) {
                pending.work.reject(format!(
                    "scheduled GPU work {id} was cancelled while queued"
                ));
                self.mutate(immediate);
            }
        }
    }

    fn device_snapshots(&self) -> Vec<DeviceSnapshot> {
        let sampled_free = self
            .state
            .resources
            .latest()
            .map(|snapshot| {
                snapshot
                    .gpus
                    .into_iter()
                    .map(|gpu| {
                        (
                            (gpu.backend, gpu.ordinal),
                            (
                                gpu.vram_total.saturating_sub(gpu.vram_used),
                                gpu.vram_used_by_mold,
                            ),
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        self.state
            .gpu_pool
            .schedulable_workers()
            .into_iter()
            .map(|worker| {
                let id = worker_device_id(&worker);
                let ready = self.ready.get(&id);
                let desired_enabled = self.state.device_registry.desired_enabled(&id);
                let busy =
                    self.leases.contains_key(&id) || worker.in_flight.load(Ordering::SeqCst) > 0;
                let health = if worker.poisoned.load(Ordering::SeqCst) {
                    DeviceHealth::Poisoned
                } else if self.unavailable.contains(&id) {
                    DeviceHealth::Unavailable
                } else if worker.is_degraded() {
                    DeviceHealth::Degraded
                } else {
                    DeviceHealth::Healthy
                };
                let mut warm = BTreeSet::new();
                if let Some(fingerprint) = worker
                    .resident_execution_fingerprint
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
                {
                    warm.insert(ExecutionFingerprint::new(fingerprint));
                }
                let active_lease = self.leases.get(&id);
                let (sampled_available_vram_bytes, sampled_mold_vram_bytes) = sampled_free
                    .get(&(worker.gpu.backend, worker.gpu.ordinal))
                    .copied()
                    .unwrap_or((worker.gpu.free_vram_bytes, None));
                let measured_cache_bytes = worker
                    .model_cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .active_vram_bytes();
                let reclaimable_cache_bytes = sampled_mold_vram_bytes
                    .map(|used_by_mold| measured_cache_bytes.min(used_by_mold))
                    .unwrap_or(0);
                DeviceSnapshot {
                    id: DeviceId::new(id),
                    backend: match worker.gpu.backend {
                        mold_core::GpuBackend::Metal => Backend::Metal,
                        _ => Backend::Cuda,
                    },
                    admin_state: if desired_enabled {
                        DeviceAdminState::Enabled
                    } else if busy {
                        DeviceAdminState::Draining
                    } else {
                        DeviceAdminState::Disabled
                    },
                    health,
                    activity: if ready.is_some()
                        && !self.leases.contains_key(&worker_device_id(&worker))
                        && worker.in_flight.load(Ordering::SeqCst) == 0
                    {
                        DeviceActivity::Idle
                    } else {
                        DeviceActivity::Busy
                    },
                    available_at_ms: active_lease.map(|lease| lease.estimated_finish_ms),
                    worker_generation: ready.map_or(0, |ready| ready.generation),
                    // The periodic resource sample is authoritative. Before
                    // its first tick, discovery's driver sample is still an
                    // actual free-memory reading; total VRAM is never used as
                    // a proxy for free capacity.
                    available_vram_bytes: effective_available_vram_bytes(
                        sampled_available_vram_bytes,
                        reclaimable_cache_bytes,
                        worker.gpu.total_vram_bytes,
                    ),
                    warm_execution_fingerprints: warm,
                }
            })
            .collect()
    }

    fn device_facts(&self) -> Vec<crate::execution_plan::DeviceFact> {
        self.device_facts_from_snapshots(&self.device_snapshots())
    }

    fn device_facts_from_snapshots(
        &self,
        snapshots: &[DeviceSnapshot],
    ) -> Vec<crate::execution_plan::DeviceFact> {
        snapshots
            .iter()
            .filter(|device| device.is_schedulable())
            .filter_map(|device| {
                let worker = self
                    .state
                    .gpu_pool
                    .workers
                    .iter()
                    .find(|worker| worker_device_id(worker) == device.id.as_str())?;
                Some(crate::execution_plan::DeviceFact {
                    id: device.id.to_string(),
                    ordinal: worker.gpu.ordinal,
                    available_vram_bytes: device.available_vram_bytes,
                })
            })
            .collect()
    }

    fn generation_plans(
        &self,
        pending: &PendingGeneration,
    ) -> Result<Vec<crate::execution_plan::ResolvedExecutionPlan>, GenerationPlanFailure> {
        self.generation_plans_with_device_facts(pending, &self.device_facts())
    }

    fn generation_plans_with_device_facts(
        &self,
        pending: &PendingGeneration,
        device_facts: &[crate::execution_plan::DeviceFact],
    ) -> Result<Vec<crate::execution_plan::ResolvedExecutionPlan>, GenerationPlanFailure> {
        let config = self.state.config.try_read().map_err(|_| {
            GenerationPlanFailure::Transient(
                "configuration changed while resolving execution plan".to_string(),
            )
        })?;
        let offload_requested = matches!(
            mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        let resolved = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &pending.job.request,
            device_facts,
            offload_requested,
            pending.prepared_inputs.as_ref(),
        );
        #[cfg(test)]
        if matches!(
            resolved,
            Err(crate::execution_plan::ExecutionPlanError::MissingArtifacts { .. })
        ) {
            // Coordinator unit tests deliberately use synthetic requests with
            // no filesystem model. Keep their transport/fencing focus without
            // weakening production admission.
            let estimate = crate::queue::estimate_model_vram(&pending.job.request.model);
            return Ok(device_facts
                .iter()
                .filter(|device| device.available_vram_bytes >= estimate)
                .cloned()
                .map(|device| crate::execution_plan::ResolvedExecutionPlan {
                    device_id: device.id,
                    device_ordinal: device.ordinal,
                    model_fingerprint: pending.job.request.model.clone(),
                    effective_placement: crate::execution_plan::EffectivePlacement {
                        components: BTreeMap::new(),
                    },
                    components: BTreeMap::new(),
                    engine_paths: mold_core::ModelPaths {
                        transformer: std::path::PathBuf::from(&pending.job.request.model),
                        transformer_shards: vec![],
                        vae: std::path::PathBuf::from(&pending.job.request.model),
                        spatial_upscaler: None,
                        temporal_upscaler: None,
                        distilled_lora: None,
                        t5_encoder: None,
                        clip_encoder: None,
                        t5_tokenizer: None,
                        clip_tokenizer: None,
                        clip_encoder_2: None,
                        clip_tokenizer_2: None,
                        text_encoder_files: vec![],
                        text_tokenizer: None,
                        decoder: None,
                    },
                    engine_config: mold_inference::FrozenEngineConfig::resolve(
                        &pending.job.request.model,
                        &config,
                    ),
                    admission_paths: mold_core::ModelPaths {
                        transformer: std::path::PathBuf::from(&pending.job.request.model),
                        transformer_shards: vec![],
                        vae: std::path::PathBuf::from(&pending.job.request.model),
                        spatial_upscaler: None,
                        temporal_upscaler: None,
                        distilled_lora: None,
                        t5_encoder: None,
                        clip_encoder: None,
                        t5_tokenizer: None,
                        clip_tokenizer: None,
                        clip_encoder_2: None,
                        clip_tokenizer_2: None,
                        text_encoder_files: vec![],
                        text_tokenizer: None,
                        decoder: None,
                    },
                    admission_engine_config: mold_inference::FrozenEngineConfig::resolve(
                        &pending.job.request.model,
                        &config,
                    ),
                    effective_loras: vec![],
                    attention_backend: crate::execution_plan::AttentionBackend::Math,
                    engine_load_strategy: mold_inference::LoadStrategy::Eager,
                    offload_mode: crate::execution_plan::OffloadMode::None,
                    predicted_vram_peak_bytes: estimate,
                    admitted_available_vram_bytes: device.available_vram_bytes,
                    predicted_host_increment_bytes: MIN_TRANSIENT_HOST_RAM,
                    determinism_class:
                        crate::execution_plan::DeterminismClass::CpuSeededCrossBackend,
                    execution_fingerprint: pending.job.request.model.clone(),
                })
                .collect());
        }
        resolved.map_err(|error| match error {
            crate::execution_plan::ExecutionPlanError::InsufficientVram => {
                GenerationPlanFailure::Transient(error.to_string())
            }
            crate::execution_plan::ExecutionPlanError::PreparedInputsStale(_) => {
                GenerationPlanFailure::StalePreparation(error.to_string())
            }
            _ => GenerationPlanFailure::Terminal(error),
        })
    }

    fn preparation_refresh_signature(
        prepared: &crate::execution_plan::PreparedExecutionInputs,
        device_facts: &[crate::execution_plan::DeviceFact],
    ) -> Vec<(String, i8)> {
        let current = device_facts
            .iter()
            .map(|device| (device.id.as_str(), device.available_vram_bytes))
            .collect::<BTreeMap<_, _>>();
        let mut signature = prepared
            .retryable_device_failures
            .keys()
            .filter(|device| current.contains_key(device.as_str()))
            .map(|device| (device.clone(), 0))
            .collect::<Vec<_>>();
        for (device_id, inputs) in &prepared.by_device {
            if !inputs.capacity_sensitive {
                continue;
            }
            let Some(&available) = current.get(device_id.as_str()) else {
                continue;
            };
            let prepared_available = inputs.prepared_available_vram_bytes;
            if let Some(direction) = capacity_refresh_direction(prepared_available, available, true)
            {
                signature.push((device_id.clone(), direction));
            }
        }
        signature.sort();
        signature
    }

    fn reset_stale_preparations(&mut self) -> bool {
        let stale = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.preparation == PreparationState::Ready)
            .filter(|(_, pending)| {
                matches!(
                    self.generation_plans(pending),
                    Err(GenerationPlanFailure::StalePreparation(_))
                )
            })
            .map(|(id, _)| id.clone())
            .collect::<BTreeSet<_>>();
        let device_facts = self.device_facts_from_snapshots(&self.device_snapshots());
        let now_ms = monotonic_ms();
        let mut refresh = BTreeSet::new();
        for (id, pending) in &mut self.pending {
            if stale.contains(id) || pending.preparation != PreparationState::Ready {
                continue;
            }
            let Some(prepared) = pending.prepared_inputs.as_ref() else {
                pending.preparation_refresh_observation = None;
                continue;
            };
            let signature = Self::preparation_refresh_signature(prepared, &device_facts);
            if signature.is_empty() {
                pending.preparation_refresh_observation = None;
                continue;
            }
            let delay = PREPARATION_RETRY_BASE_MS
                .saturating_mul(1_u64 << u32::from(pending.preparation_retry_attempts.min(4)))
                .clamp(PREPARATION_REFRESH_STABILITY_MS, PREPARATION_RETRY_MAX_MS);
            if observe_preparation_refresh(
                &mut pending.preparation_refresh_observation,
                signature,
                now_ms,
                delay,
            ) {
                refresh.insert(id.clone());
            }
        }
        if stale.is_empty() && refresh.is_empty() {
            return false;
        }
        for id in stale.iter().chain(refresh.iter()) {
            if let Some(pending) = self.pending.get_mut(id) {
                pending.preparation = PreparationState::Needed;
                pending.prepared_inputs = None;
                pending.preparation_refresh_observation = None;
                if refresh.contains(id) {
                    pending.preparation_retry_attempts =
                        pending.preparation_retry_attempts.saturating_add(1);
                } else {
                    pending.preparation_retry_attempts = 0;
                }
            }
        }
        self.state_version = self.state_version.saturating_add(1);
        self.dirty.mark_dirty(Instant::now(), self.state_version);
        self.start_needed_preparations();
        true
    }

    fn generation_plan_catalog(
        &self,
        device_facts: &[crate::execution_plan::DeviceFact],
    ) -> BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>> {
        self.pending
            .iter()
            .filter(|(_, pending)| pending.preparation == PreparationState::Ready)
            .map(|(id, pending)| {
                let plans = self
                    .generation_plans_with_device_facts(pending, device_facts)
                    .unwrap_or_else(|error| {
                        tracing::debug!(
                            job_id = id,
                            %error,
                            "generation has no valid execution plan"
                        );
                        Vec::new()
                    });
                (id.clone(), plans)
            })
            .collect()
    }

    fn reject_terminal_generation_plan_errors(&mut self) -> bool {
        let failures = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.preparation == PreparationState::Ready)
            .filter_map(|(id, pending)| match self.generation_plans(pending) {
                Err(GenerationPlanFailure::Terminal(error)) => {
                    Some((id.clone(), error.to_string()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if failures.is_empty() {
            return false;
        }
        for (id, error) in failures {
            if let Some(pending) = self.pending.remove(&id) {
                self.plan_invalidations.remove(&id);
                reject_generation(&self.state, pending.job, error);
            }
        }
        self.state_version = self.state_version.saturating_add(1);
        true
    }

    fn work_snapshots(
        &self,
        generation_plans: &BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>>,
        device_snapshots: &[DeviceSnapshot],
    ) -> Vec<WorkSnapshot> {
        let queue_order = self.state.job_registry.queued_ids_in_order();
        let ranks = queue_order
            .iter()
            .enumerate()
            .map(|(rank, id)| (id.as_str(), rank as u64))
            .collect::<BTreeMap<_, _>>();
        let now_ms = monotonic_ms();
        let mut snapshots: Vec<WorkSnapshot> = self
            .pending
            .iter()
            .filter(|(_, pending)| {
                pending.preparation == PreparationState::Ready
                    && pending
                        .retry_not_before_ms
                        .is_none_or(|deadline| deadline <= now_ms)
            })
            .map(|(id, pending)| {
                let model = pending.job.request.model.as_str();
                let failed = crate::gpu_pool::failed_ordinals_for_model(model);
                let candidates = generation_plans
                    .get(id)
                    .into_iter()
                    .flatten()
                    .filter(|plan| !failed.contains(&plan.device_ordinal))
                    .cloned()
                    .map(|plan| {
                        let key = self
                            .state
                            .gpu_pool
                            .worker_by_ordinal(plan.device_ordinal)
                            .map(|worker| {
                                generation_estimate_key(
                                    &worker,
                                    &pending.job.request,
                                    &plan.execution_fingerprint,
                                )
                            })
                            .unwrap_or_else(|| EstimateKey {
                                device_class: plan.device_id.clone(),
                                model_fingerprint: pending.job.request.model.clone(),
                                work_kind: "generation".into(),
                                shape_bucket: generation_shape_bucket(&pending.job.request),
                                execution_fingerprint: plan.execution_fingerprint.clone(),
                            });
                        let static_total_ms = static_generation_time_ms(&pending.job.request);
                        let estimate = self.estimates.estimate(
                            &key,
                            StaticEstimate {
                                total_ms: static_total_ms,
                                vram_bytes: plan.predicted_vram_peak_bytes,
                                host_bytes: plan.predicted_host_increment_bytes,
                            },
                        );
                        let load_ms = self
                            .estimates
                            .exact(&key)
                            .and_then(|bucket| bucket.ewma_load_ms)
                            .unwrap_or(1_000.0)
                            .round()
                            .max(0.0) as u64;
                        CandidatePlacement::new(
                            DeviceId::new(plan.device_id),
                            ExecutionFingerprint::new(plan.execution_fingerprint),
                            estimate.host_bytes,
                        )
                        .with_vram(estimate.vram_bytes)
                        .with_timing(
                            load_ms,
                            0,
                            estimate.total_ms.saturating_sub(load_ms),
                        )
                        .with_device_available_vram(plan.admitted_available_vram_bytes)
                    })
                    .collect::<Vec<_>>();
                let mut work = WorkSnapshot::new(
                    WorkId::new(id.clone()),
                    ranks.get(id.as_str()).copied().unwrap_or(u64::MAX),
                    candidates,
                )
                .with_bypass_count(pending.bypass_count)
                .with_ready_at(pending.ready_at_ms);
                if let Some(started) = pending.warm_wait_started_ms {
                    work = work.with_warm_wait_started_at(started);
                }
                let request_pin = self
                    .state
                    .gpu_pool
                    .resolve_explicit_placement_gpu(pending.job.request.placement.as_ref())
                    .ok()
                    .flatten();
                let explicit =
                    request_pin.or_else(|| self.state.job_registry.target_gpu(id).flatten());
                if let Some(ordinal) = explicit {
                    if let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ordinal) {
                        work = work.with_hard_device(DeviceId::new(worker_device_id(&worker)));
                    } else {
                        work = work
                            .with_hard_device(DeviceId::new(format!("unavailable:gpu:{ordinal}")));
                    }
                }
                work
            })
            .collect();
        let authoritative_available_vram = device_snapshots
            .iter()
            .cloned()
            .map(|device| (device.id, device.available_vram_bytes))
            .collect::<BTreeMap<_, _>>();
        snapshots.extend(self.pending_owner_work.iter().map(|(id, pending)| {
            let candidates = self
                .state
                .gpu_pool
                .workers
                .iter()
                .map(|worker| {
                    let key =
                        owner_estimate_key(worker, pending.work.kind(), &pending.model_fingerprint);
                    let estimate = self.estimates.estimate(
                        &key,
                        StaticEstimate {
                            total_ms: 1_000,
                            vram_bytes: pending.estimated_vram_bytes,
                            host_bytes: pending.estimated_host_ram_bytes,
                        },
                    );
                    CandidatePlacement::new(
                        DeviceId::new(worker_device_id(&worker)),
                        ExecutionFingerprint::new(pending.model_fingerprint.clone()),
                        estimate.host_bytes,
                    )
                    .with_vram(estimate.vram_bytes.min(worker.gpu.total_vram_bytes))
                    .with_timing(0, 0, estimate.total_ms)
                    .with_device_available_vram(
                        authoritative_available_vram
                            .get(&DeviceId::new(worker_device_id(&worker)))
                            .copied()
                            .unwrap_or(0),
                    )
                })
                .collect::<Vec<_>>();
            let mut work = WorkSnapshot::new(
                WorkId::new(id.clone()),
                (u64::MAX / 2).saturating_add(pending.queue_rank),
                candidates,
            )
            .with_priority(pending.priority)
            .with_bypass_count(pending.bypass_count)
            .with_ready_at(pending.ready_at_ms);
            if let Some(started) = pending.warm_wait_started_ms {
                work = work.with_warm_wait_started_at(started);
            }
            work.kind = pending.work.kind();
            if let Some(ordinal) = pending.hard_ordinal {
                if let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ordinal) {
                    work = work.with_hard_device(DeviceId::new(worker_device_id(&worker)));
                } else {
                    work =
                        work.with_hard_device(DeviceId::new(format!("unavailable:gpu:{ordinal}")));
                }
            }
            work
        }));
        snapshots
    }

    fn planner_snapshot(
        &self,
    ) -> (
        PlannerSnapshot,
        BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>>,
    ) {
        let devices = self.device_snapshots();
        let device_facts = self.device_facts_from_snapshots(&devices);
        let generation_plans = self.generation_plan_catalog(&device_facts);
        let work = self.work_snapshots(&generation_plans, &devices);
        (
            PlannerSnapshot {
                state_version: self.state_version,
                next_plan_version: self.plan_version.saturating_add(1),
                now_ms: monotonic_ms(),
                next_replan_at_ms: self.dirty.deadline().map(monotonic_deadline_ms),
                host_memory: self.memory.snapshot(),
                devices,
                work,
            },
            generation_plans,
        )
    }

    async fn dispatch_ready(&mut self) -> Option<u64> {
        if self
            .dispatch_retry_not_before_ms
            .is_some_and(|deadline| deadline > monotonic_ms())
        {
            return None;
        }
        self.dispatch_retry_not_before_ms = None;
        if self.reset_stale_preparations() {
            return None;
        }
        if !self.pending.is_empty() {
            self.reject_terminal_generation_plan_errors();
        }
        if self.state.queue_pause.is_paused()
            || (self.pending.is_empty() && self.pending_owner_work.is_empty())
            || self
                .state
                .gpu_pool
                .workers
                .iter()
                .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
        {
            return None;
        }
        let mut replans_this_turn = 0_u8;
        loop {
            if self.reject_terminal_generation_plan_errors()
                && self.pending.is_empty()
                && self.pending_owner_work.is_empty()
            {
                return None;
            }
            let (snapshot, generation_plans) = self.planner_snapshot();
            let plan = match self.planner.plan(&snapshot) {
                Ok(plan) => plan,
                Err(error) => {
                    tracing::error!(
                        state_version = snapshot.state_version,
                        error = %error,
                        "scheduler rejected its runtime snapshot; refusing to grant work"
                    );
                    return None;
                }
            };
            self.plan_version = plan.plan_version;
            self.publish_plan(&snapshot, &plan);
            if plan.immediate_leases.is_empty() {
                self.clear_dispatch_retry();
                log_typed_blocks(&plan);
                return Some(plan.state_version);
            }

            #[cfg(test)]
            if let Some(hook) = self.before_grant_hook.take() {
                hook.plan_built.notify_one();
                hook.resume.notified().await;
            }

            let mutation_fence = self.state.scheduler_mutation_fence.clone();
            let _mutation_guard = mutation_fence.lock().await;
            let mut mutation_detected = false;
            self.reconcile_external_mutations(&mut mutation_detected);
            if self.state.queue_pause.is_paused()
                || self
                    .state
                    .gpu_pool
                    .workers
                    .iter()
                    .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
            {
                return None;
            }
            if mutation_detected {
                replans_this_turn = replans_this_turn.saturating_add(1);
                if replans_this_turn >= MAX_DISPATCH_REPLANS_PER_TURN {
                    self.defer_dispatch_retry();
                    return None;
                }
                continue;
            }

            // Validate every proposed lease against the same mutation-fenced
            // reducer turn before the all-or-none reservation and nonblocking
            // worker sends. No route mutation can enter until this scope ends.
            let current_devices = self
                .device_snapshots()
                .into_iter()
                .map(|device| (device.id.clone(), device))
                .collect::<BTreeMap<_, _>>();
            let current_device_facts = self.device_facts_from_snapshots(
                &current_devices.values().cloned().collect::<Vec<_>>(),
            );
            let current_generation_plans = self.generation_plan_catalog(&current_device_facts);
            let grants_valid = plan.immediate_leases.iter().all(|lease| {
                let device_id = lease.device_id.to_string();
                let Ok(ready) = validate_worker_grant(
                    &self.ready,
                    &self.leases,
                    &device_id,
                    lease.worker_generation,
                ) else {
                    return false;
                };
                let Some(device) = current_devices.get(&lease.device_id) else {
                    return false;
                };
                let work_id = lease.work_id.to_string();
                let generation = self.pending.get(&work_id);
                let utility = self.pending_owner_work.get(&work_id);
                let work_ready = generation
                    .is_some_and(|pending| pending.preparation == PreparationState::Ready)
                    || utility.is_some();
                let work_cancelled = if let Some(pending) = generation {
                    self.state.job_registry.entry(&work_id).is_none_or(|entry| {
                        entry.state != crate::job_registry::JobLifecycle::Queued
                    }) || pending.job.result_tx.is_closed()
                } else {
                    utility.is_none_or(|pending| pending.work.is_cancelled())
                };
                let current_execution_fingerprint = if generation.is_some() {
                    let planned_execution = generation_plans
                        .get(&work_id)
                        .into_iter()
                        .flatten()
                        .find(|execution| {
                            execution.device_id == device_id
                                && execution.execution_fingerprint
                                    == lease.placement.execution_fingerprint.as_str()
                                && execution.predicted_vram_peak_bytes
                                    == lease.placement.predicted_vram_bytes
                                && execution.predicted_host_increment_bytes
                                    == lease.placement.incremental_host_ram_bytes
                                && execution.admitted_available_vram_bytes
                                    == lease.placement.device_available_vram_bytes
                        });
                    let current_execution = current_generation_plans
                        .get(&work_id)
                        .into_iter()
                        .flatten()
                        .find(|execution| execution.device_id == device_id);
                    if planned_execution.is_none() || planned_execution != current_execution {
                        return false;
                    }
                    ExecutionFingerprint::new(
                        current_execution
                            .expect("checked equal concrete execution plans")
                            .execution_fingerprint
                            .clone(),
                    )
                } else {
                    let Some(utility) = utility else {
                        return false;
                    };
                    ExecutionFingerprint::new(utility.model_fingerprint.clone())
                };
                plan.validate_lease_for_grant(
                    lease,
                    &GrantValidationSnapshot {
                        work_id: WorkId::new(work_id.clone()),
                        device_id: DeviceId::new(device_id),
                        state_version: self.state_version,
                        plan_version: self.plan_version,
                        sample_generation: self.memory.snapshot().sample_generation,
                        ledger_sequence: self.memory.sequence,
                        work_ready,
                        work_cancelled,
                        worker_generation: ready.generation,
                        worker_ready: true,
                        device_admin_state: device.admin_state,
                        device_health: device.health,
                        execution_fingerprint: current_execution_fingerprint,
                        available_vram_bytes: device.available_vram_bytes,
                    },
                )
                .is_ok()
            });
            if !grants_valid
                || self
                    .memory
                    .try_reserve(&plan, self.state_version, self.plan_version)
                    .is_err()
            {
                self.state_version = self.state_version.saturating_add(1);
                replans_this_turn = replans_this_turn.saturating_add(1);
                if replans_this_turn >= MAX_DISPATCH_REPLANS_PER_TURN {
                    self.defer_dispatch_retry();
                    return None;
                }
                continue;
            }
            let mut granted = Vec::new();
            let mut grant_failed = false;
            let mut fatal_during_grant = false;
            for lease in &plan.immediate_leases {
                if self
                    .state
                    .gpu_pool
                    .workers
                    .iter()
                    .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
                {
                    grant_failed = true;
                    fatal_during_grant = true;
                    break;
                }
                let device_id = lease.device_id.to_string();
                let Ok(ready) = validate_worker_grant(
                    &self.ready,
                    &self.leases,
                    &device_id,
                    lease.worker_generation,
                ) else {
                    grant_failed = true;
                    break;
                };
                let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ready.ordinal) else {
                    grant_failed = true;
                    break;
                };
                let id = lease.work_id.to_string();
                let fence = LeaseFence {
                    work_id: id.clone(),
                    device_id: device_id.clone(),
                    owner_epoch: ready.owner_epoch,
                    state_version: plan.state_version,
                    plan_version: plan.plan_version,
                    worker_generation: ready.generation,
                    memory_sample_generation: plan.reservation.sample_generation,
                    memory_ledger_sequence: plan.reservation.ledger_sequence,
                };
                if !worker.try_claim_owner_in_flight() {
                    grant_failed = true;
                    break;
                }
                if let Some(pending) = self.pending.remove(&id) {
                    // Transport the exact immutable plan used by matching and
                    // the all-or-none reservation. Never re-resolve a
                    // potentially different plan after admission.
                    let execution_plan = generation_plans
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .find(|execution| {
                            execution.device_id == device_id
                                && execution.execution_fingerprint
                                    == lease.placement.execution_fingerprint.as_str()
                                && execution.predicted_vram_peak_bytes
                                    == lease.placement.predicted_vram_bytes
                                && execution.predicted_host_increment_bytes
                                    == lease.placement.incremental_host_ram_bytes
                        })
                        .cloned();
                    let Some(execution_plan) = execution_plan else {
                        self.pending.insert(id.clone(), pending);
                        worker.release_in_flight();
                        self.state_version = self.state_version.saturating_add(1);
                        grant_failed = true;
                        break;
                    };
                    let bypass_count = pending.bypass_count;
                    let warm_wait_started_ms = pending.warm_wait_started_ms;
                    let ready_at_ms = pending.ready_at_ms;
                    let retry_not_before_ms = pending.retry_not_before_ms;
                    let prepared_inputs = pending.prepared_inputs.clone();
                    let estimate_key = generation_estimate_key(
                        &worker,
                        &pending.job.request,
                        &execution_plan.execution_fingerprint,
                    );
                    let gpu_job = gpu_job_from_generation(
                        &self.state,
                        pending.job,
                        fence.clone(),
                        Some(execution_plan),
                        prepared_inputs.clone(),
                    );
                    let grant = Box::new(LeaseGrant {
                        fence,
                        work: OwnerWork::Generation(Box::new(gpu_job)),
                        retry: None,
                    });
                    let dispatch = self.state.job_registry.dispatch_if_queued(
                        &id,
                        ready.ordinal,
                        grant,
                        |grant| {
                            worker.try_send_job(grant).map_err(|error| match error {
                                std::sync::mpsc::TrySendError::Full(grant)
                                | std::sync::mpsc::TrySendError::Disconnected(grant) => grant,
                            })
                        },
                    );
                    match dispatch {
                        Ok(previous_target) => {
                            self.ready.remove(&device_id);
                            self.leases.insert(
                                device_id.clone(),
                                ActiveLease {
                                    work_id: id.clone(),
                                    owner_epoch: ready.owner_epoch,
                                    plan_version: plan.plan_version,
                                    worker_generation: ready.generation,
                                    accepted: false,
                                    previous_target,
                                    estimated_finish_ms: lease.estimated_finish_ms,
                                    ready_at_ms,
                                    bypass_count,
                                    warm_wait_started_ms,
                                    started_at: Instant::now(),
                                    estimate_key,
                                },
                            );
                            granted.push(id);
                        }
                        Err(crate::job_registry::DispatchAttemptError::Claim(error, returned)) => {
                            worker.release_in_flight();
                            let generation = generation_from_owner_grant(*returned);
                            reject_generation(
                                &self.state,
                                generation,
                                format!(
                                    "generation job {id} lost its queued dispatch claim: {error:?}"
                                ),
                            );
                            grant_failed = true;
                            break;
                        }
                        Err(crate::job_registry::DispatchAttemptError::Transport(returned)) => {
                            worker.release_in_flight();
                            let generation = generation_from_owner_grant(*returned);
                            self.pending.insert(
                                generation.id.clone(),
                                PendingGeneration {
                                    job: generation,
                                    ready_at_ms,
                                    bypass_count,
                                    warm_wait_started_ms,
                                    preparation: PreparationState::Ready,
                                    prepared_inputs,
                                    retry_not_before_ms,
                                    preparation_retry_attempts: 0,
                                    preparation_refresh_observation: None,
                                },
                            );
                            self.unavailable.insert(device_id.clone());
                            self.state.device_registry.mark_unavailable(&device_id);
                            grant_failed = true;
                            break;
                        }
                    }
                } else if let Some(pending) = self.pending_owner_work.remove(&id) {
                    let estimate_key = owner_estimate_key(
                        &worker,
                        pending.work.kind(),
                        &pending.model_fingerprint,
                    );
                    let metadata = (
                        pending.model_fingerprint,
                        pending.estimated_vram_bytes,
                        pending.estimated_host_ram_bytes,
                        pending.hard_ordinal,
                        pending.priority,
                        pending.queue_rank,
                        pending.ready_at_ms,
                        pending.bypass_count,
                        pending.warm_wait_started_ms,
                    );
                    let grant = Box::new(LeaseGrant {
                        fence,
                        work: pending.work,
                        retry: Some(crate::gpu_pool::OwnerWorkRetry {
                            model_fingerprint: metadata.0.clone(),
                            estimated_vram_bytes: metadata.1,
                            estimated_host_ram_bytes: metadata.2,
                            hard_ordinal: metadata.3,
                            priority: metadata.4,
                            queue_rank: metadata.5,
                            ready_at_ms: metadata.6,
                            bypass_count: metadata.7,
                            warm_wait_started_ms: metadata.8,
                        }),
                    });
                    match worker.try_send_job(grant) {
                        Ok(()) => {
                            self.ready.remove(&device_id);
                            self.leases.insert(
                                device_id.clone(),
                                ActiveLease {
                                    work_id: id.clone(),
                                    owner_epoch: ready.owner_epoch,
                                    plan_version: plan.plan_version,
                                    worker_generation: ready.generation,
                                    accepted: false,
                                    previous_target: None,
                                    estimated_finish_ms: lease.estimated_finish_ms,
                                    ready_at_ms: metadata.6,
                                    bypass_count: metadata.7,
                                    warm_wait_started_ms: metadata.8,
                                    started_at: Instant::now(),
                                    estimate_key,
                                },
                            );
                            granted.push(id);
                        }
                        Err(error) => {
                            worker.release_in_flight();
                            let returned = match error {
                                std::sync::mpsc::TrySendError::Full(grant)
                                | std::sync::mpsc::TrySendError::Disconnected(grant) => grant,
                            };
                            self.pending_owner_work.insert(
                                id,
                                PendingOwnerWork {
                                    model_fingerprint: metadata.0,
                                    estimated_vram_bytes: metadata.1,
                                    estimated_host_ram_bytes: metadata.2,
                                    hard_ordinal: metadata.3,
                                    priority: metadata.4,
                                    queue_rank: metadata.5,
                                    ready_at_ms: metadata.6,
                                    bypass_count: metadata.7,
                                    warm_wait_started_ms: metadata.8,
                                    work: returned.work,
                                },
                            );
                            self.unavailable.insert(device_id.clone());
                            self.state.device_registry.mark_unavailable(&device_id);
                            grant_failed = true;
                            break;
                        }
                    }
                } else {
                    worker.release_in_flight();
                    grant_failed = true;
                    break;
                }
            }

            if grant_failed {
                let granted_set = granted.iter().map(String::as_str).collect::<BTreeSet<_>>();
                let ungranted = plan
                    .reservation
                    .items
                    .iter()
                    .filter(|item| !granted_set.contains(item.work_id.as_str()))
                    .map(|item| item.work_id.as_str())
                    .collect::<Vec<_>>();
                self.memory
                    .settle_partial_matching(granted.iter().map(String::as_str), ungranted);
                self.state_version = self.state_version.saturating_add(1);
                // The failed transport and every ungranted reservation settle
                // atomically above. Re-enter planning immediately while the
                // remaining Ready capacity is still in this reducer turn.
                if fatal_during_grant {
                    return None;
                }
                if granted.is_empty() {
                    replans_this_turn = replans_this_turn.saturating_add(1);
                    if replans_this_turn >= MAX_DISPATCH_REPLANS_PER_TURN {
                        self.defer_dispatch_retry();
                        return None;
                    }
                } else {
                    // Successful grants make finite queue progress; do not let
                    // a later transport failure consume the no-progress retry
                    // budget for the remaining Ready capacity.
                    self.record_dispatch_progress();
                    replans_this_turn = 0;
                }
                continue;
            }

            self.record_dispatch_progress();
            for update in &plan.bypass_updates {
                if let Some(pending) = self.pending.get_mut(update.work_id.as_str()) {
                    pending.bypass_count = update.new_count;
                }
                if let Some(pending) = self.pending_owner_work.get_mut(update.work_id.as_str()) {
                    pending.bypass_count = update.new_count;
                }
            }
            for wait in &plan.warm_waits {
                if let Some(pending) = self.pending.get_mut(wait.work_id.as_str()) {
                    pending.warm_wait_started_ms = Some(wait.started_at_ms);
                }
                if let Some(pending) = self.pending_owner_work.get_mut(wait.work_id.as_str()) {
                    pending.warm_wait_started_ms = Some(wait.started_at_ms);
                }
            }
            self.state_version = self.state_version.saturating_add(1);
            return Some(plan.state_version);
        }
    }

    fn publish_plan(&self, snapshot: &PlannerSnapshot, plan: &Plan) {
        let confidence = plan
            .lanes
            .iter()
            .flat_map(|lane| {
                lane.assignments.iter().map(move |assignment| {
                    let worker = self
                        .state
                        .gpu_pool
                        .workers
                        .iter()
                        .find(|worker| worker_device_id(worker) == lane.device_id.as_str());
                    let key = self
                        .pending
                        .get(assignment.work_id.as_str())
                        .and_then(|pending| {
                            worker.map(|worker| {
                                generation_estimate_key(
                                    worker,
                                    &pending.job.request,
                                    assignment.placement.execution_fingerprint.as_str(),
                                )
                            })
                        });
                    let confidence = key
                        .as_ref()
                        .and_then(|key| {
                            self.estimates
                                .exact(key)
                                .or_else(|| self.estimates.exact(&key.normalized()))
                        })
                        .map(|bucket| match bucket.confidence() {
                            mold_scheduler::EstimateConfidence::Low => {
                                mold_core::QueueEstimateConfidence::Low
                            }
                            mold_scheduler::EstimateConfidence::Medium => {
                                mold_core::QueueEstimateConfidence::Medium
                            }
                            mold_scheduler::EstimateConfidence::High => {
                                mold_core::QueueEstimateConfidence::High
                            }
                        })
                        .unwrap_or_default();
                    (assignment.work_id.to_string(), confidence)
                })
            })
            .collect::<BTreeMap<_, _>>();
        let wire = queue_plan_projection(
            snapshot,
            plan,
            &self.state.gpu_pool,
            &confidence,
            self.dirty.dirty_since,
        );
        let mut current = self
            .state
            .scheduled_work
            .latest_plan
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = current.as_ref() {
            if existing.plan_version >= wire.plan_version
                || queue_plan_semantically_equal(existing, &wire)
            {
                return;
            }
        }
        *current = Some(wire.clone());
        drop(current);
        self.state
            .events
            .publish(mold_core::ServerEvent::QueuePlanChanged {
                plan: Box::new(wire),
            });
    }

    fn publish_device_state_if_changed(&mut self) {
        let state = crate::routes::current_device_state(&self.state);
        let signature = device_event_signature(&state);
        if self.last_device_event_signature.as_ref() == Some(&signature) {
            return;
        }
        self.last_device_event_signature = Some(signature);
        self.state
            .events
            .publish(mold_core::ServerEvent::DeviceStateChanged {
                state: Box::new(state),
            });
    }

    fn reject_all_unstarted_for_fatal_cuda(&mut self) {
        self.reject_all_unstarted("CUDA context is fatally poisoned; server restart required");
    }

    fn reject_all_unstarted(&mut self, message: &str) {
        let pending = std::mem::take(&mut self.pending);
        self.plan_invalidations.clear();
        for (_, pending) in pending {
            reject_generation(&self.state, pending.job, message.to_string());
        }
        let pending_owner_work = std::mem::take(&mut self.pending_owner_work);
        for (_, pending) in pending_owner_work {
            pending.work.reject(message.to_string());
        }
    }
}

pub async fn run_scheduler_coordinator(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    mut owner_work_rx: tokio::sync::mpsc::Receiver<ScheduledOwnerWork>,
    mut worker_rx: tokio::sync::mpsc::UnboundedReceiver<WorkerEvent>,
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) {
    tracing::info!(
        workers = state.gpu_pool.worker_count(),
        "multi-GPU scheduler coordinator started"
    );
    let mut coordinator = Coordinator::new(state);
    let registry_notify = coordinator.state.job_registry.mutation_notifier();
    let mut ticker = tokio::time::interval(RECONCILE_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut memory_ticker = tokio::time::interval(MEMORY_SAMPLE_INTERVAL);
    memory_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut fatal = false;
    let mut generation_ingress_open = true;
    let mut owner_ingress_open = true;
    loop {
        let mut immediate = false;
        tokio::select! {
            _ = shutdown.cancelled() => {
                job_rx.close();
                owner_work_rx.close();
                while let Ok(job) = job_rx.try_recv() {
                    reject_generation(
                        &coordinator.state,
                        job,
                        "generation scheduler is shutting down".to_string(),
                    );
                }
                while let Ok(work) = owner_work_rx.try_recv() {
                    work.work
                        .reject("generation scheduler is shutting down".to_string());
                }
                coordinator.reject_all_unstarted("generation scheduler is shutting down");
                break;
            }
            job = job_rx.recv(), if generation_ingress_open => {
                match job {
                    Some(job) => coordinator.enqueue(job, &mut immediate),
                    None => generation_ingress_open = false,
                }
            }
            work = owner_work_rx.recv(), if owner_ingress_open => {
                match work {
                    Some(work) => coordinator.enqueue_owner_work(work, &mut immediate),
                    None => owner_ingress_open = false,
                }
            }
            event = worker_rx.recv() => {
                if let Some(event) = event {
                    coordinator.handle_worker_event_serialized(event, &mut immediate).await;
                }
            }
            event = coordinator.preparation_rx.recv() => {
                if let Some(event) = event {
                    coordinator.handle_preparation_event(event, &mut immediate);
                }
            }
            _ = registry_notify.notified() => {
                coordinator.reconcile_external_mutations(&mut immediate);
            }
            _ = ticker.tick() => {
                coordinator.reconcile_external_mutations(&mut immediate);
            }
            _ = memory_ticker.tick() => {
                coordinator.memory.collect_now();
                coordinator.mutate(&mut immediate);
            }
        }
        if !generation_ingress_open
            && !owner_ingress_open
            && coordinator.pending.is_empty()
            && coordinator.pending_owner_work.is_empty()
            && coordinator.leases.is_empty()
        {
            break;
        }
        coordinator.start_needed_preparations();
        while coordinator.preparation_tasks.try_join_next().is_some() {}
        if coordinator
            .state
            .gpu_pool
            .workers
            .iter()
            .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
        {
            job_rx.close();
            owner_work_rx.close();
            while let Ok(job) = job_rx.try_recv() {
                reject_generation(
                    &coordinator.state,
                    job,
                    "CUDA context is fatally poisoned; server restart required".to_string(),
                );
            }
            while let Ok(work) = owner_work_rx.try_recv() {
                work.work.reject(
                    "CUDA context is fatally poisoned; server restart required".to_string(),
                );
            }
            coordinator.reject_all_unstarted_for_fatal_cuda();
            // Publish the poisoned health transition before supervision tears
            // down the process. REST remains authoritative if the frame races
            // a reconnect or shutdown.
            coordinator.publish_device_state_if_changed();
            fatal = true;
            break;
        }
        if immediate {
            let _ = coordinator.dispatch_ready().await;
        }
        if coordinator.dirty.due(Instant::now()) {
            if let Some(planned_state_version) = coordinator.dispatch_ready().await {
                coordinator.dirty.clear_through(planned_state_version);
            }
        }
        if coordinator.device_state_dirty {
            coordinator.device_state_dirty = false;
            coordinator.publish_device_state_if_changed();
        }
    }
    coordinator.stop_preparations().await;
    for worker in &coordinator.state.gpu_pool.workers {
        worker.request_shutdown();
    }
    if fatal {
        coordinator.reject_all_unstarted_for_fatal_cuda();
    }
    tracing::info!("multi-GPU scheduler coordinator stopped");
}

fn gpu_job_from_generation(
    state: &AppState,
    mut job: GenerationJob,
    lease: LeaseFence,
    execution_plan: Option<crate::execution_plan::ResolvedExecutionPlan>,
    prepared_execution_inputs: Option<crate::execution_plan::PreparedExecutionInputs>,
) -> GpuJob {
    if let Some(plan) = execution_plan.as_ref() {
        crate::execution_plan::materialize_request(plan, &mut job.request);
    }
    GpuJob {
        id: job.id,
        model: job.request.model.clone(),
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
        execution_plan,
        prepared_execution_inputs,
        lease: Some(lease),
    }
}

fn generation_and_prepared_from_gpu_job(
    job: GpuJob,
) -> (
    GenerationJob,
    Option<crate::execution_plan::PreparedExecutionInputs>,
) {
    let prepared = job.prepared_execution_inputs;
    (
        GenerationJob {
            id: job.id,
            request: job.request,
            completion_payload: job.completion_payload,
            progress_tx: job.progress_tx,
            result_tx: job.result_tx,
            output_dir: job.output_dir,
        },
        prepared,
    )
}

fn generation_from_gpu_job(job: GpuJob) -> GenerationJob {
    generation_and_prepared_from_gpu_job(job).0
}

fn generation_from_owner_grant(grant: LeaseGrant) -> GenerationJob {
    match grant.work {
        OwnerWork::Generation(job) => generation_from_gpu_job(*job),
        work => panic!(
            "generation dispatch returned non-generation owner work {:?}",
            work.kind()
        ),
    }
}

fn reject_generation(state: &AppState, job: GenerationJob, error: String) {
    if let Some(progress) = job.progress_tx {
        let _ = progress.send(SseMessage::Error(mold_core::SseErrorEvent {
            message: error.clone(),
        }));
    }
    let id = job.id.clone();
    let _ = job.result_tx.send(Err(error));
    state.queue.decrement();
    state.job_registry.remove(&id);
}

fn log_typed_blocks(plan: &Plan) {
    if plan
        .blocked
        .iter()
        .any(|blocked| blocked.reason == BlockedReason::NoSchedulableDevice)
    {
        tracing::warn!("generation queue blocked: no schedulable device");
    }
}

fn queue_plan_projection(
    snapshot: &PlannerSnapshot,
    plan: &Plan,
    pool: &crate::gpu_pool::GpuPool,
    confidence_by_work: &BTreeMap<String, mold_core::QueueEstimateConfidence>,
    dirty_since: Option<Instant>,
) -> mold_core::QueuePlan {
    let monotonic_now = monotonic_ms();
    let unix_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    let to_unix = |deadline: u64| unix_now.saturating_add(deadline.saturating_sub(monotonic_now));
    let ordinals = pool
        .workers
        .iter()
        .map(|worker| (worker_device_id(worker), worker.gpu.ordinal))
        .collect::<BTreeMap<_, _>>();

    let work_items = snapshot
        .work
        .iter()
        .map(|work| {
            let planned = plan.lanes.iter().find_map(|lane| {
                lane.assignments
                    .iter()
                    .enumerate()
                    .find(|(_, assignment)| assignment.work_id == work.id)
                    .map(|(order, assignment)| (lane, order, assignment))
            });
            let blocked = plan
                .blocked
                .iter()
                .find(|blocked| blocked.work_id == work.id);
            let warm_wait = plan.warm_waits.iter().find(|wait| wait.work_id == work.id);
            let reason = blocked
                .map(|blocked| snake_debug(blocked.reason))
                .or_else(|| warm_wait.map(|_| "warm_wait".to_string()))
                .or_else(|| {
                    plan.immediate_leases
                        .iter()
                        .find(|lease| lease.work_id == work.id)
                        .map(|lease| snake_debug(lease.reason))
                });
            let (planned_device_id, lane_order, start, finish) =
                planned.map_or((None, None, None, None), |(lane, order, assignment)| {
                    (
                        Some(lane.device_id.to_string()),
                        Some(order),
                        Some(to_unix(assignment.estimated_start_ms)),
                        Some(to_unix(assignment.estimated_finish_ms)),
                    )
                });
            let hard_id = work.hard_device_id.as_ref().map(ToString::to_string);
            let planned_gpu = planned_device_id
                .as_ref()
                .and_then(|id| ordinals.get(id).copied());
            let target_gpu = work
                .hard_device_id
                .as_ref()
                .and_then(|id| ordinals.get(id.as_str()).copied());
            mold_core::QueueWorkItem {
                work_id: work.id.to_string(),
                parent_id: work.parent_id.to_string(),
                work_kind: snake_debug(work.kind),
                priority_class: snake_debug(work.priority_class),
                queue_rank: work.queue_rank,
                bypass_count: work.bypass_count,
                gpu: planned_gpu,
                hard_pinned_device_id: hard_id,
                target_gpu,
                planned_device_id,
                lane_order,
                estimated_start_unix_ms: start,
                estimated_finish_unix_ms: finish,
                estimate_confidence: confidence_by_work
                    .get(work.id.as_str())
                    .copied()
                    .unwrap_or_default(),
                reason,
            }
        })
        .collect();

    mold_core::QueuePlan {
        plan_version: plan.plan_version,
        state_version: plan.state_version,
        optimizer_state: snake_debug(plan.optimizer_state),
        dirty_since_unix_ms: dirty_since.map(|started| {
            unix_now.saturating_sub(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }),
        next_replan_at_unix_ms: plan.next_replan_at_ms.map(to_unix),
        work_items,
    }
}

fn load_estimate_store(state: &AppState) -> EstimateStore {
    let Some(db) = state.metadata_db.as_ref().as_ref() else {
        return EstimateStore::default();
    };
    let estimates = mold_db::SchedulerEstimates::new(db);
    let cutoff = unix_seconds().saturating_sub(180 * 24 * 60 * 60);
    if let Err(error) = estimates.prune_before(cutoff, 10_000) {
        tracing::warn!(error = %format!("{error:#}"), "failed to prune scheduler estimates");
    }
    match estimates.list() {
        Ok(records) => {
            EstimateStore::from_buckets(records.into_iter().map(|record| EstimateBucket {
                key: EstimateKey {
                    device_class: record.device_class,
                    model_fingerprint: record.model_fingerprint,
                    work_kind: record.work_kind,
                    shape_bucket: record.shape_bucket,
                    execution_fingerprint: record.execution_fingerprint,
                },
                sample_count: record.sample_count,
                ewma_total_ms: record.ewma_total_ms,
                ewma_load_ms: record.ewma_load_ms,
                vram_conservative_bytes: record.vram_high_water_bytes,
                host_conservative_bytes: record.host_high_water_bytes,
                last_observed_at_unix_s: record.last_observed_at,
            }))
        }
        Err(error) => {
            tracing::warn!(error = %format!("{error:#}"), "failed to load scheduler estimates");
            EstimateStore::default()
        }
    }
}

fn estimate_record(bucket: &EstimateBucket) -> mold_db::SchedulerEstimateRecord {
    mold_db::SchedulerEstimateRecord {
        estimate_key: bucket.key.persistence_key(),
        device_class: bucket.key.device_class.clone(),
        model_fingerprint: bucket.key.model_fingerprint.clone(),
        work_kind: bucket.key.work_kind.clone(),
        shape_bucket: bucket.key.shape_bucket.clone(),
        execution_fingerprint: bucket.key.execution_fingerprint.clone(),
        sample_count: bucket.sample_count,
        ewma_total_ms: bucket.ewma_total_ms,
        ewma_load_ms: bucket.ewma_load_ms,
        // Schema v13 used "high_water" before the runtime semantics were
        // corrected. Retain the column names for migration compatibility.
        vram_high_water_bytes: bucket.vram_conservative_bytes,
        host_high_water_bytes: bucket.host_conservative_bytes,
        last_observed_at: bucket.last_observed_at_unix_s,
    }
}

fn generation_estimate_key(
    worker: &GpuWorker,
    request: &mold_core::GenerateRequest,
    execution_fingerprint: &str,
) -> EstimateKey {
    EstimateKey {
        device_class: device_class(worker),
        model_fingerprint: request.model.clone(),
        work_kind: "generation".into(),
        shape_bucket: generation_shape_bucket(request),
        execution_fingerprint: execution_fingerprint.to_string(),
    }
}

fn owner_estimate_key(
    worker: &GpuWorker,
    kind: mold_scheduler::WorkKind,
    fingerprint: &str,
) -> EstimateKey {
    EstimateKey {
        device_class: device_class(worker),
        model_fingerprint: fingerprint.to_string(),
        work_kind: snake_debug(kind),
        shape_bucket: "utility".into(),
        execution_fingerprint: fingerprint.to_string(),
    }
}

fn device_class(worker: &GpuWorker) -> String {
    let backend = match worker.gpu.backend {
        mold_core::GpuBackend::Cuda => "cuda",
        mold_core::GpuBackend::Metal => "metal",
    };
    let capability = worker.gpu.compute_capability.map_or_else(
        || "unknown".to_string(),
        |(major, minor)| format!("sm{major}{minor}"),
    );
    let gib = worker.gpu.total_vram_bytes.div_ceil(1 << 30);
    format!("{backend}:{capability}:{gib}gb")
}

fn generation_shape_bucket(request: &mold_core::GenerateRequest) -> String {
    format!(
        "{}x{}:s{}:f{}:fps{}:a{}:src{}:edit{}:lora{}:b{}",
        request.width,
        request.height,
        request.steps,
        request.frames.unwrap_or(1),
        request.fps.unwrap_or(0),
        u8::from(request.enable_audio == Some(true)),
        u8::from(request.source_image.is_some() || request.source_video.is_some()),
        request.edit_images.as_ref().map_or(0, Vec::len),
        u8::from(request.lora.is_some() || request.loras.as_ref().is_some_and(|v| !v.is_empty())),
        request.batch_size,
    )
}

fn static_generation_time_ms(request: &mold_core::GenerateRequest) -> u64 {
    let megapixels = (u64::from(request.width) * u64::from(request.height)).div_ceil(1_000_000);
    let frames = u64::from(request.frames.unwrap_or(1));
    1_000u64.saturating_add(
        megapixels
            .max(1)
            .saturating_mul(u64::from(request.steps).max(1))
            .saturating_mul(frames)
            .saturating_mul(125),
    )
}

fn unix_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn snake_debug(value: impl std::fmt::Debug) -> String {
    let debug = format!("{value:?}");
    let mut out = String::with_capacity(debug.len());
    for (index, character) in debug.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.push(character.to_ascii_lowercase());
        } else {
            out.push(character);
        }
    }
    out
}

pub(crate) fn monotonic_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Capacity that can safely be available at the next owner-thread allocation
/// boundary: current driver-reported free bytes plus only the measured active
/// cache entry that the same owner can unload or reuse. Other process and
/// non-cache allocations are deliberately never treated as reclaimable.
pub(crate) fn effective_available_vram_bytes(
    sampled_free_bytes: u64,
    reclaimable_cache_bytes: u64,
    total_vram_bytes: u64,
) -> u64 {
    sampled_free_bytes
        .saturating_add(reclaimable_cache_bytes)
        .min(total_vram_bytes)
}

fn monotonic_deadline_ms(deadline: Instant) -> u64 {
    monotonic_ms().saturating_add(
        deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_pool::GpuPool;
    use crate::model_cache::ModelCache;
    use crate::state::{QueueHandle, SseCompletionPayload};
    use mold_inference::device::{CudaDeviceKind, DiscoveredGpu};
    use mold_inference::shared_pool::SharedPool;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Mutex, RwLock};

    #[derive(Clone)]
    struct FixedHostMemorySampler {
        reading: HostMemoryReading,
    }

    impl HostMemorySampler for FixedHostMemorySampler {
        fn sample(&self) -> HostMemoryReading {
            self.reading
        }
    }

    struct ScriptedHostMemorySampler {
        readings: Mutex<std::collections::VecDeque<HostMemoryReading>>,
        calls: AtomicUsize,
    }

    impl ScriptedHostMemorySampler {
        fn new(readings: impl IntoIterator<Item = HostMemoryReading>) -> Arc<Self> {
            Arc::new(Self {
                readings: Mutex::new(readings.into_iter().collect()),
                calls: AtomicUsize::new(0),
            })
        }
    }

    impl HostMemorySampler for ScriptedHostMemorySampler {
        fn sample(&self) -> HostMemoryReading {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.readings
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted host-memory sample exhausted")
        }
    }

    fn memory_reading(total_gib: u64, available_gib: u64) -> HostMemoryReading {
        HostMemoryReading {
            total_bytes: total_gib << 30,
            available_bytes: available_gib << 30,
        }
    }

    fn unsampled_memory(total_bytes: u64, available_bytes: u64) -> HostMemoryLedger {
        HostMemoryLedger::new(Arc::new(FixedHostMemorySampler {
            reading: HostMemoryReading {
                total_bytes,
                available_bytes,
            },
        }))
    }

    fn sampled_memory(total_bytes: u64, available_bytes: u64) -> HostMemoryLedger {
        let mut memory = unsampled_memory(total_bytes, available_bytes);
        memory.collect_now();
        memory
    }

    fn ample_memory() -> HostMemoryLedger {
        sampled_memory(128 << 30, 112 << 30)
    }

    #[test]
    fn effective_vram_counts_only_measured_reclaimable_cache_memory() {
        const GIB: u64 = 1 << 30;
        assert_eq!(
            effective_available_vram_bytes(4 * GIB, 16 * GIB, 24 * GIB),
            20 * GIB,
            "a warm resident or cold evictable 16 GiB model remains feasible"
        );
        assert_eq!(
            effective_available_vram_bytes(4 * GIB, 0, 24 * GIB),
            4 * GIB,
            "active non-cache allocations must not be invented as reclaimable"
        );
        assert_eq!(
            effective_available_vram_bytes(20 * GIB, 16 * GIB, 24 * GIB),
            24 * GIB,
            "effective capacity must remain capped at physical VRAM"
        );
    }

    struct ImmediatePreparer;

    impl DependencyPreparer for ImmediatePreparer {
        fn prepare(
            &self,
            _state: AppState,
            _work_id: String,
            _request: mold_core::GenerateRequest,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
        ) -> PreparationFuture {
            Box::pin(async { Ok(PreparedGeneration::default()) })
        }
    }

    struct ResidentTestEngine;

    impl mold_inference::InferenceEngine for ResidentTestEngine {
        fn generate(
            &mut self,
            _req: &mold_core::GenerateRequest,
        ) -> anyhow::Result<mold_core::GenerateResponse> {
            unreachable!("capacity test never runs inference")
        }

        fn model_name(&self) -> &str {
            "resident-test"
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn test_worker(
        ordinal: usize,
    ) -> (
        Arc<GpuWorker>,
        std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
    ) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(1);
        let worker = Arc::new(GpuWorker {
            owner_epoch: 1,
            gpu: DiscoveredGpu {
                ordinal,
                stable_id: Some(format!("cuda:{:032x}", ordinal + 1)),
                raw_cuda_uuid: Some(((ordinal + 1) as u128).to_be_bytes()),
                device_kind: Some(CudaDeviceKind::FullGpu),
                identity_error: None,
                backend: mold_core::GpuBackend::Cuda,
                name: format!("gpu-{ordinal}"),
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

    fn recv_grant(rx: &std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>) -> GpuJob {
        match rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker grant")
        {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => match grant.work {
                OwnerWork::Generation(job) => *job,
                work => panic!("expected generation grant, got {:?}", work.kind()),
            },
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain command"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown command"),
        }
    }

    #[test]
    fn duplicate_ready_clears_transport_unavailable_before_generation_dedupe() {
        let (worker, _worker_rx) = test_worker(0);
        let device_id = worker_device_id(&worker);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        coordinator.ready.insert(
            device_id.clone(),
            ReadyWorker {
                ordinal: 0,
                owner_epoch: 1,
                generation: 7,
            },
        );
        coordinator.unavailable.insert(device_id.clone());

        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 7,
            },
            &mut immediate,
        );

        assert!(
            !coordinator.unavailable.contains(&device_id),
            "same-generation Ready must repair a prior transport-Full mark"
        );
        assert_eq!(
            coordinator
                .ready
                .get(&device_id)
                .map(|ready| ready.generation),
            Some(7)
        );
    }

    fn fake_generation(
        id: &str,
    ) -> (
        GenerationJob,
        tokio::sync::oneshot::Receiver<Result<crate::state::GenerationJobResult, String>>,
    ) {
        let request = serde_json::from_str(
            r#"{"prompt":"parallel","model":"flux-dev:q4","width":512,"height":512,"steps":4,"guidance":3.5,"batch_size":1}"#,
        )
        .unwrap();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        (
            GenerationJob {
                id: id.to_string(),
                request,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx,
                output_dir: None,
            },
            result_rx,
        )
    }

    #[test]
    fn replan_timer_slides_to_two_seconds_but_never_past_five() {
        let start = Instant::now();
        let mut window = ReplanWindow::new();
        window.mark_dirty(start, 1);
        assert_eq!(window.deadline(), Some(start + REPLAN_DEBOUNCE));
        window.mark_dirty(start + Duration::from_secs(1), 2);
        assert_eq!(window.deadline(), Some(start + Duration::from_secs(3)));
        window.mark_dirty(start + Duration::from_secs(4), 3);
        assert_eq!(window.deadline(), Some(start + REPLAN_MAX_DELAY));
        window.clear_through(2);
        assert!(
            window.deadline().is_some(),
            "stale plan must not clear timer"
        );
        window.clear_through(3);
        assert!(window.deadline().is_none());
    }

    #[test]
    fn completed_prompt_expansion_freezes_original_and_replaces_generation_prompt() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let (mut generation, _result) = fake_generation("expanded");
        generation.request.prompt = "source prompt".to_string();
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(generation, &mut immediate);
        coordinator
            .pending
            .get_mut("expanded")
            .expect("pending generation")
            .preparation = PreparationState::Preparing;

        coordinator.handle_preparation_event(
            PreparationEvent::Ready {
                work_id: "expanded".to_string(),
                prepared: PreparedGeneration {
                    expanded_prompt: Some("expanded prompt".to_string()),
                    execution_inputs: None,
                },
            },
            &mut immediate,
        );

        let request = &coordinator
            .pending
            .get("expanded")
            .expect("generation remains queued")
            .job
            .request;
        assert_eq!(request.original_prompt.as_deref(), Some("source prompt"));
        assert_eq!(request.prompt, "expanded prompt");
    }

    #[tokio::test]
    async fn changed_config_invalidates_prepared_inputs_and_starts_repreparation() {
        let root = tempfile::TempDir::new().unwrap();
        for name in ["transformer.gguf", "vae.safetensors", "t5.safetensors"] {
            std::fs::write(root.path().join(name), b"prepared").unwrap();
        }
        let mut config = mold_core::Config {
            t5_variant: Some("fp16".to_string()),
            ..Default::default()
        };
        config.models.insert(
            "prepared-test".to_string(),
            mold_core::ModelConfig {
                transformer: Some(root.path().join("transformer.gguf").display().to_string()),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                t5_encoder: Some(root.path().join("t5.safetensors").display().to_string()),
                family: Some("flux".to_string()),
                ..Default::default()
            },
        );
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(config.clone(), QueueHandle::new(ingress_tx), pool, 1);
        let (mut generation, _result) = fake_generation("stale-prepared");
        generation.request.model = "prepared-test".to_string();
        let prepared = crate::variant_dependencies::prepare_local_execution_inputs(
            &config,
            &generation.request,
            vec![crate::execution_plan::DeviceFact {
                id: worker_device_id(&worker),
                ordinal: 0,
                available_vram_bytes: 24 << 30,
            }],
        )
        .await
        .unwrap();
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(generation, &mut immediate);
        let pending = coordinator.pending.get_mut("stale-prepared").unwrap();
        pending.preparation = PreparationState::Ready;
        pending.prepared_inputs = Some(prepared);
        state.config.write().await.t5_variant = Some("q3".to_string());

        assert!(coordinator.reset_stale_preparations());
        let pending = coordinator.pending.get("stale-prepared").unwrap();
        assert_eq!(pending.preparation, PreparationState::Preparing);
        assert!(pending.prepared_inputs.is_none());
        coordinator.stop_preparations().await;
    }

    #[test]
    fn stale_plan_or_memory_generation_cannot_reserve() {
        let mut ledger = unsampled_memory(40 << 30, 20 << 30);
        let started = ledger.begin_collection();
        ledger.publish_sample(started, 40 << 30, 20 << 30);
        let planner = Planner::default();
        let mut snapshot = PlannerSnapshot::new(
            3,
            9,
            0,
            ledger.headroom_bytes(),
            vec![DeviceSnapshot::idle("gpu-a", 24 << 30)],
            vec![WorkSnapshot::new(
                "work",
                0,
                vec![CandidatePlacement::new("gpu-a", "model", 8 << 30)],
            )],
        );
        snapshot.host_memory = ledger.snapshot();
        let plan = planner.plan(&snapshot).unwrap();
        let started = ledger.begin_collection();
        ledger.publish_sample(started, 40 << 30, 20 << 30);
        assert_eq!(
            ledger.try_reserve(&plan, 3, 9),
            Err(GrantFenceError::StalePlan)
        );
        assert!(ledger.reservations.is_empty());
    }

    #[test]
    fn concurrent_ledger_change_rejects_the_entire_matching_without_partial_reservations() {
        let mut ledger = unsampled_memory(48 << 30, 40 << 30);
        let initial = ledger.begin_collection();
        ledger.publish_sample(initial, 48 << 30, 40 << 30);
        let planner = Planner::default();
        let mut snapshot = PlannerSnapshot::new(
            3,
            9,
            0,
            ledger.headroom_bytes(),
            vec![
                DeviceSnapshot::idle("gpu-a", 24 << 30),
                DeviceSnapshot::idle("gpu-b", 24 << 30),
            ],
            vec![
                WorkSnapshot::new(
                    "work-a",
                    0,
                    vec![CandidatePlacement::new("gpu-a", "model", 8 << 30)],
                ),
                WorkSnapshot::new(
                    "work-b",
                    1,
                    vec![CandidatePlacement::new("gpu-b", "model", 8 << 30)],
                ),
            ],
        );
        snapshot.host_memory = ledger.snapshot();
        let stale_plan = planner.plan(&snapshot).expect("valid two-lease plan");
        assert_eq!(stale_plan.immediate_leases.len(), 2);
        assert_eq!(stale_plan.reservation.items.len(), 2);

        let raced_collection = ledger.begin_collection();
        assert_eq!(
            ledger.try_reserve(&stale_plan, 3, 9),
            Err(GrantFenceError::StalePlan)
        );
        assert!(
            ledger.reservations.is_empty(),
            "a stale aggregate reservation must insert neither item"
        );

        ledger.publish_sample(raced_collection, 48 << 30, 40 << 30);
        snapshot.host_memory = ledger.snapshot();
        let fresh_plan = planner.plan(&snapshot).expect("fresh two-lease plan");
        ledger
            .try_reserve(&fresh_plan, 3, 9)
            .expect("fresh aggregate reservation");
        assert_eq!(ledger.reservations.len(), 2);
    }

    #[test]
    fn stale_ready_generation_and_duplicate_device_lease_are_rejected() {
        let ready = BTreeMap::from([(
            "gpu-a".to_string(),
            ReadyWorker {
                ordinal: 0,
                owner_epoch: 1,
                generation: 7,
            },
        )]);
        assert_eq!(
            validate_worker_grant(&ready, &BTreeMap::new(), "gpu-a", 6),
            Err(GrantFenceError::StaleWorkerGeneration)
        );
        let leases = BTreeMap::from([(
            "gpu-a".to_string(),
            ActiveLease {
                work_id: "work-a".to_string(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 7,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: 1,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey::default(),
            },
        )]);
        assert_eq!(
            validate_worker_grant(&ready, &leases, "gpu-a", 7),
            Err(GrantFenceError::DuplicateDeviceLease)
        );
    }

    #[test]
    fn aggregate_eight_plus_eight_is_rejected_against_twelve_gib_headroom() {
        let mut ledger = unsampled_memory(40 << 30, 20 << 30);
        let started = ledger.begin_collection();
        ledger.publish_sample(started, 40 << 30, 20 << 30);
        assert_eq!(ledger.headroom_bytes(), 12 << 30);
        let planner = Planner::default();
        let mut snapshot = PlannerSnapshot::new(
            1,
            1,
            0,
            ledger.headroom_bytes(),
            vec![
                DeviceSnapshot::idle("gpu-a", 24 << 30),
                DeviceSnapshot::idle("gpu-b", 24 << 30),
            ],
            vec![
                WorkSnapshot::new("a", 0, vec![CandidatePlacement::new("gpu-a", "m", 8 << 30)]),
                WorkSnapshot::new("b", 1, vec![CandidatePlacement::new("gpu-b", "m", 8 << 30)]),
            ],
        );
        snapshot.host_memory = ledger.snapshot();
        let plan = planner.plan(&snapshot).unwrap();
        assert_eq!(plan.immediate_leases.len(), 1);
        ledger.try_reserve(&plan, 1, 1).unwrap();
        assert_eq!(ledger.reservations.len(), 1);
    }

    #[test]
    fn concurrent_sample_keeps_commit_charged_until_following_collection() {
        let mut ledger = unsampled_memory(40 << 30, 32 << 30);
        let initial = ledger.begin_collection();
        ledger.publish_sample(initial, 40 << 30, 32 << 30);
        ledger.reservations.insert(
            "work".to_string(),
            HostReservation {
                bytes: 8 << 30,
                state: ReservationState::Reserved,
            },
        );
        assert_eq!(ledger.headroom_bytes(), 16 << 30);

        let raced_collection = ledger.begin_collection();
        ledger.commit("work");
        ledger.publish_sample(raced_collection, 40 << 30, 24 << 30);
        assert_eq!(
            ledger.headroom_bytes(),
            8 << 30,
            "allocation committed after collection began must remain charged"
        );
        let following_collection = ledger.begin_collection();
        ledger.publish_sample(following_collection, 40 << 30, 24 << 30);
        assert_eq!(
            ledger.headroom_bytes(),
            16 << 30,
            "a collection begun after commit safely reflects the allocation"
        );
    }

    #[test]
    fn delayed_allocation_and_unavailable_sampler_remain_conservative() {
        let mut unavailable = unsampled_memory(40 << 30, 32 << 30);
        assert_eq!(unavailable.headroom_bytes(), 0);

        let started = unavailable.begin_collection();
        unavailable.publish_sample(started, 40 << 30, 32 << 30);
        unavailable.reservations.insert(
            "delayed".to_string(),
            HostReservation {
                bytes: 8 << 30,
                state: ReservationState::Reserved,
            },
        );
        for _ in 0..3 {
            let started = unavailable.begin_collection();
            unavailable.publish_sample(started, 40 << 30, 32 << 30);
            assert_eq!(
                unavailable.headroom_bytes(),
                16 << 30,
                "an uncommitted reservation must never be absorbed by samples"
            );
        }
    }

    #[tokio::test]
    async fn injected_low_memory_sample_blocks_dispatch_without_host_pressure_dependency() {
        let (worker, worker_rx) = test_worker(0);
        let device_id = worker_device_id(&worker);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let sampler = ScriptedHostMemorySampler::new([memory_reading(40, 8)]);
        let mut coordinator = Coordinator::with_preparer_and_sampler(
            state,
            Arc::new(ImmediatePreparer),
            sampler.clone(),
        );
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "host-pressure-block",
                "synthetic-model",
                1 << 30,
                OwnerWork::Probe {
                    id: "host-pressure-block".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id,
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );

        coordinator.dispatch_ready().await;

        assert_eq!(sampler.calls.load(Ordering::SeqCst), 1);
        assert_eq!(coordinator.memory.headroom_bytes(), 0);
        assert!(coordinator
            .pending_owner_work
            .contains_key("host-pressure-block"));
        assert!(coordinator.leases.is_empty());
        assert!(
            worker_rx.try_recv().is_err(),
            "low sampled host memory must block transport"
        );
    }

    #[test]
    fn device_event_signature_ignores_raw_telemetry_and_plan_membership() {
        let mut state = mold_core::DeviceState {
            devices: vec![mold_core::DeviceInfo {
                id: "cuda:stable".into(),
                backend: mold_core::GpuBackend::Cuda,
                ordinal: Some(0),
                device_kind: mold_core::DeviceKind::FullGpu,
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                name: "GPU".into(),
                pci_bus_id: None,
                compute_capability: Some("8.6".into()),
                memory: mold_core::DeviceMemoryInfo {
                    total_bytes: Some(24 << 30),
                    used_bytes: Some(1 << 30),
                    mold_used_bytes: Some(1 << 30),
                    other_used_bytes: Some(0),
                },
                telemetry: mold_core::DeviceTelemetry {
                    utilization_percent: Some(10),
                    temperature_c: None,
                    power_w: None,
                },
                desired_enabled: true,
                admin_state: mold_core::DeviceAdminState::Enabled,
                health: mold_core::DeviceHealth::Healthy,
                activity: mold_core::DeviceActivity::Idle,
                schedulable: true,
                unschedulable_reason: None,
                loaded_models: Vec::new(),
                active_work_id: None,
                planned_work_ids: vec!["work-a".into()],
            }],
            plan_version: 1,
        };
        let original = device_event_signature(&state);

        state.devices[0].telemetry.utilization_percent = Some(99);
        state.devices[0].memory.used_bytes = Some(20 << 30);
        state.devices[0].planned_work_ids = vec!["work-b".into()];
        state.plan_version = 2;
        assert_eq!(
            device_event_signature(&state),
            original,
            "resource samples and plan publications have dedicated channels"
        );

        state.devices[0].health = mold_core::DeviceHealth::Unavailable;
        assert_ne!(
            device_event_signature(&state),
            original,
            "health transitions must invalidate the device snapshot"
        );
    }

    #[test]
    fn queue_plan_event_dedup_ignores_versions_and_wall_clock_drift() {
        let work = mold_core::QueueWorkItem {
            work_id: "work-a".into(),
            parent_id: "job-a".into(),
            work_kind: "generation".into(),
            priority_class: "user".into(),
            planned_device_id: Some("cuda:stable".into()),
            lane_order: Some(0),
            estimated_start_unix_ms: Some(10_000),
            estimated_finish_unix_ms: Some(15_000),
            ..Default::default()
        };
        let first = mold_core::QueuePlan {
            plan_version: 1,
            state_version: 2,
            optimizer_state: "optimized".into(),
            dirty_since_unix_ms: Some(9_000),
            next_replan_at_unix_ms: Some(11_000),
            work_items: vec![work.clone()],
        };
        let shifted = mold_core::QueuePlan {
            plan_version: 99,
            state_version: 100,
            dirty_since_unix_ms: Some(10_000),
            next_replan_at_unix_ms: Some(12_000),
            work_items: vec![mold_core::QueueWorkItem {
                estimated_start_unix_ms: Some(11_000),
                estimated_finish_unix_ms: Some(16_000),
                ..work
            }],
            ..first.clone()
        };
        assert!(queue_plan_semantically_equal(&first, &shifted));

        let mut slower = shifted;
        slower.work_items[0].estimated_finish_unix_ms = Some(17_000);
        assert!(!queue_plan_semantically_equal(&first, &slower));
    }

    #[test]
    fn device_events_publish_once_per_semantic_health_transition() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()],
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let mut state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        state.device_registry = Arc::new(crate::device_registry::DeviceRegistry::new(
            Arc::new(crate::device_registry::StaticDeviceDiscovery::new(vec![
                crate::device_registry::DiscoveredDevice::from_runtime_gpu(&worker.gpu, true, None),
            ])),
            Arc::new(None),
        ));
        let mut receiver = state.events.subscribe();
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );

        coordinator.publish_device_state_if_changed();
        let first = receiver.try_recv().expect("initial semantic snapshot");
        assert!(matches!(
            first,
            mold_core::ServerEvent::DeviceStateChanged { state }
                if state.devices[0].health == mold_core::DeviceHealth::Healthy
        ));
        coordinator.publish_device_state_if_changed();
        assert!(
            receiver.try_recv().is_err(),
            "unchanged snapshots must not emit duplicate events"
        );

        let device_id = worker.gpu.stable_id.as_deref().unwrap();
        assert!(coordinator
            .state
            .device_registry
            .set_desired_enabled(device_id, false)
            .unwrap());
        coordinator.publish_device_state_if_changed();
        let disabled = receiver.try_recv().expect("preference transition");
        assert!(matches!(
            disabled,
            mold_core::ServerEvent::DeviceStateChanged { state }
                if state.devices[0].admin_state == mold_core::DeviceAdminState::Disabled
        ));
        assert!(!coordinator
            .state
            .device_registry
            .set_desired_enabled(device_id, false)
            .unwrap());
        coordinator.publish_device_state_if_changed();
        assert!(
            receiver.try_recv().is_err(),
            "an idempotent preference update must not emit another event"
        );

        worker.poisoned.store(true, Ordering::SeqCst);
        coordinator.publish_device_state_if_changed();
        let poisoned = receiver.try_recv().expect("poison transition");
        assert!(matches!(
            poisoned,
            mold_core::ServerEvent::DeviceStateChanged { state }
                if state.devices[0].health == mold_core::DeviceHealth::Poisoned
        ));
    }

    #[test]
    fn failed_post_upscale_completion_never_trains_but_success_does() {
        let make_coordinator = || {
            let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
            let state = AppState::empty(
                mold_core::Config::default(),
                QueueHandle::new(ingress_tx),
                Arc::new(GpuPool {
                    workers: Vec::new(),
                }),
                1,
            );
            Coordinator::with_preparer_and_memory(
                state,
                Arc::new(ImmediatePreparer),
                ample_memory(),
            )
        };
        let key = EstimateKey {
            device_class: "cuda-sm86".into(),
            model_fingerprint: "real-esrgan-x4plus:fp16".into(),
            work_kind: "post_upscale".into(),
            shape_bucket: "1024x1024".into(),
            execution_fingerprint: "fp16".into(),
        };
        let complete = |coordinator: &mut Coordinator, successful| {
            coordinator.leases.insert(
                "cuda:stable".into(),
                ActiveLease {
                    work_id: "post-upscale".into(),
                    plan_version: 1,
                    worker_generation: 1,
                    accepted: true,
                    previous_target: None,
                    started_at: Instant::now(),
                    estimate_key: key.clone(),
                },
            );
            let mut immediate = false;
            coordinator.handle_worker_event(
                WorkerEvent::Completed {
                    device_id: "cuda:stable".into(),
                    ordinal: 0,
                    worker_generation: 1,
                    successful,
                    load_ms: Some(250),
                },
                &mut immediate,
            );
        };

        let mut failed = make_coordinator();
        complete(&mut failed, false);
        assert!(
            failed.estimates.exact(&key).is_none(),
            "downgraded-to-original post-upscale failure must not train EWMA"
        );

        let mut succeeded = make_coordinator();
        complete(&mut succeeded, true);
        assert!(
            succeeded.estimates.exact(&key).is_some(),
            "successful post-upscale must train EWMA"
        );
        assert_eq!(
            succeeded.estimates.exact(&key).unwrap().ewma_load_ms,
            Some(250.0),
            "worker-measured activation time must train the load estimate"
        );
    }

    #[test]
    fn db_disabled_estimator_learns_for_the_process_lifetime() {
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            Arc::new(GpuPool {
                workers: Vec::new(),
            }),
            1,
        );
        assert!(
            state.metadata_db.as_ref().as_ref().is_none(),
            "fixture must exercise metadata-DB-disabled mode"
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let key = EstimateKey {
            device_class: "cuda:sm86:24gb".into(),
            model_fingerprint: "flux-dev:q8".into(),
            work_kind: "generation".into(),
            shape_bucket: "1024x1024".into(),
            execution_fingerprint: "q8".into(),
        };

        coordinator.observe_estimate(
            key.clone(),
            EstimateObservation {
                total_ms: 12_000,
                load_ms: Some(2_000),
                vram_completion_sample_bytes: Some(20 << 30),
                host_completion_sample_bytes: Some(8 << 30),
                observed_at_unix_s: unix_seconds(),
            },
        );

        let learned = coordinator
            .estimates
            .exact(&key)
            .expect("the in-memory estimator remains active without persistence");
        assert_eq!(learned.sample_count, 1);
        assert_eq!(learned.ewma_total_ms, 12_000.0);
    }

    #[test]
    fn planner_is_work_conserving_for_one_two_eight_and_sixty_four_devices() {
        for count in [1_usize, 2, 8, 64] {
            let devices = (0..count)
                .map(|index| DeviceSnapshot::idle(format!("gpu-{index}"), 24 << 30))
                .collect::<Vec<_>>();
            let work = (0..count)
                .map(|index| {
                    WorkSnapshot::new(
                        format!("work-{index}"),
                        index as u64,
                        (0..count)
                            .map(|device| {
                                CandidatePlacement::new(format!("gpu-{device}"), "same-model", 0)
                            })
                            .collect(),
                    )
                })
                .collect();
            let plan = Planner::default()
                .plan(&PlannerSnapshot::new(1, 1, 0, u64::MAX, devices, work))
                .unwrap();
            assert_eq!(plan.immediate_leases.len(), count);
            let leased_devices = plan
                .immediate_leases
                .iter()
                .map(|lease| lease.device_id.clone())
                .collect::<BTreeSet<_>>();
            assert_eq!(leased_devices.len(), count, "duplicate device lease");
        }
    }

    #[test]
    fn scheduler_capacity_uses_sampled_free_vram_not_total() {
        let (worker, _rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".into(),
            timestamp: 1,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu-0".into(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24 << 30,
                vram_used: 19 << 30,
                vram_used_by_mold: Some(1 << 30),
                vram_used_by_other: Some(18 << 30),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 128 << 30,
                used: 1 << 30,
                available: None,
                used_by_mold: 0,
                used_by_other: 1 << 30,
            },
            cpu: None,
        });
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        assert_eq!(
            coordinator.device_snapshots()[0].available_vram_bytes,
            5 << 30
        );
    }

    #[test]
    fn warm_and_cold_resident_capacity_is_safe_while_idle_or_busy() {
        const GIB: u64 = 1 << 30;
        let (worker, _rx) = test_worker(0);
        worker
            .model_cache
            .lock()
            .unwrap()
            .insert(Box::new(ResidentTestEngine), 16 * GIB);
        worker.set_resident_execution_fingerprint(Some("warm-plan"));
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".into(),
            timestamp: 1,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu-0".into(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24 * GIB,
                vram_used: 20 * GIB,
                vram_used_by_mold: Some(16 * GIB),
                vram_used_by_other: Some(4 * GIB),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 128 * GIB,
                used: GIB,
                available: Some(127 * GIB),
                used_by_mold: 0,
                used_by_other: GIB,
            },
            cpu: None,
        });
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let device_id = worker_device_id(&worker);
        coordinator.ready.insert(
            device_id.clone(),
            ReadyWorker {
                ordinal: 0,
                owner_epoch: 1,
                generation: 1,
            },
        );

        let idle = coordinator.device_snapshots().remove(0);
        assert_eq!(idle.available_vram_bytes, 20 * GIB);
        assert!(idle
            .warm_execution_fingerprints
            .contains(&ExecutionFingerprint::new("warm-plan")));

        let checked_out = worker
            .model_cache
            .lock()
            .unwrap()
            .take("resident-test")
            .unwrap();
        worker.in_flight.store(1, Ordering::SeqCst);
        coordinator.leases.insert(
            device_id,
            ActiveLease {
                work_id: "busy".into(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: 5_000,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
            },
        );
        let busy = coordinator.device_snapshots().remove(0);
        assert_eq!(busy.available_vram_bytes, 20 * GIB);
        assert_eq!(busy.available_at_ms, Some(5_000));
        assert_eq!(busy.activity, DeviceActivity::Busy);
        worker.model_cache.lock().unwrap().restore(checked_out);
    }

    #[test]
    fn no_schedulable_device_is_a_typed_block() {
        let plan = Planner::default()
            .plan(&PlannerSnapshot::new(
                1,
                1,
                0,
                1024,
                vec![DeviceSnapshot::idle("gpu", 1024).with_health(DeviceHealth::Degraded)],
                vec![WorkSnapshot::new(
                    "work",
                    0,
                    vec![CandidatePlacement::new("gpu", "model", 0)],
                )],
            ))
            .unwrap();
        assert_eq!(
            plan.blocked_reason(&WorkId::new("work")),
            Some(&BlockedReason::NoSchedulableDevice)
        );
    }

    #[tokio::test]
    async fn two_ready_workers_receive_two_concurrent_leases_without_local_fifo() {
        let (worker_a, worker_a_rx) = test_worker(0);
        let (worker_b, worker_b_rx) = test_worker(1);
        let pool = Arc::new(GpuPool {
            workers: vec![worker_a.clone(), worker_b.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 8);
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 1,
            gpus: Vec::new(),
            system_ram: mold_core::RamSnapshot {
                total: 128 << 30,
                used: 1 << 30,
                available: None,
                used_by_mold: 0,
                used_by_other: 1 << 30,
            },
            cpu: None,
        });
        let (job_a, _result_a) = fake_generation("a");
        let (job_b, _result_b) = fake_generation("b");
        state.job_registry.register("a", "flux-dev:q4");
        state.job_registry.register("b", "flux-dev:q4");
        queue.submit(job_a, 8).await.unwrap();
        queue.submit(job_b, 8).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        for pending in coordinator.pending.values_mut() {
            pending.preparation = PreparationState::Ready;
        }
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker_a),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker_b),
                ordinal: 1,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;

        let dispatched_a = recv_grant(&worker_a_rx);
        let dispatched_b = recv_grant(&worker_b_rx);
        assert_ne!(dispatched_a.id, dispatched_b.id);
        assert!(dispatched_a.lease.is_some());
        assert!(dispatched_b.lease.is_some());
        assert!(matches!(
            worker_a_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert!(matches!(
            worker_b_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert_eq!(coordinator.leases.len(), 2);
        assert!(coordinator.pending.is_empty());
    }

    #[tokio::test]
    async fn request_gpu_pin_wins_over_mutable_queue_target() {
        let (worker_a, worker_a_rx) = test_worker(0);
        let (worker_b, worker_b_rx) = test_worker(1);
        let pool = Arc::new(GpuPool {
            workers: vec![worker_a.clone(), worker_b.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let (mut job, _result) = fake_generation("request-pinned");
        job.request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Auto,
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(1),
                ..mold_core::AdvancedPlacement::default()
            }),
        });
        state.job_registry.register("request-pinned", "flux-dev:q4");
        state
            .job_registry
            .set_target_gpu("request-pinned", Some(0))
            .expect("mutable queue target accepted while queued");

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(job, &mut immediate);
        coordinator
            .pending
            .get_mut("request-pinned")
            .expect("pending generation")
            .preparation = PreparationState::Ready;
        for (ordinal, worker) in [(0, worker_a), (1, worker_b)] {
            coordinator.handle_worker_event(
                WorkerEvent::Ready {
                    device_id: worker_device_id(&worker),
                    ordinal,
                    owner_epoch: 1,
                    worker_generation: 1,
                },
                &mut immediate,
            );
        }

        coordinator.dispatch_ready().await;

        assert_eq!(recv_grant(&worker_b_rx).id, "request-pinned");
        assert!(matches!(
            worker_a_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn generation_and_owner_utility_work_share_the_same_multi_worker_lease_set() {
        let (worker_a, worker_a_rx) = test_worker(0);
        let (worker_b, worker_b_rx) = test_worker(1);
        let pool = Arc::new(GpuPool {
            workers: vec![worker_a.clone(), worker_b.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 4);
        let (job, _result) = fake_generation("generation");
        state.job_registry.register("generation", "flux-dev:q4");
        queue.submit(job, 4).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("generation")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "upscale",
                "real-esrgan-x4plus:fp16",
                2 << 30,
                OwnerWork::Probe {
                    id: "upscale".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );
        for (worker, ordinal) in [(&worker_a, 0), (&worker_b, 1)] {
            coordinator.handle_worker_event(
                WorkerEvent::Ready {
                    device_id: worker_device_id(worker),
                    ordinal,
                    owner_epoch: 1,
                    worker_generation: 1,
                },
                &mut immediate,
            );
        }
        coordinator.dispatch_ready().await;

        let grants = [worker_a_rx.recv().unwrap(), worker_b_rx.recv().unwrap()];
        let kinds = grants
            .into_iter()
            .map(|command| match command {
                crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant.work.kind(),
                crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
                crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
            })
            .collect::<BTreeSet<_>>();
        assert!(kinds.contains(&mold_scheduler::WorkKind::Generation));
        assert!(kinds.contains(&mold_scheduler::WorkKind::StandaloneUpscale));
        assert_eq!(coordinator.leases.len(), 2);
    }

    #[tokio::test]
    async fn legacy_chain_in_flight_blocks_owner_utility_grant() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;

        worker.in_flight.store(1, Ordering::SeqCst);
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "utility-during-chain",
                "real-esrgan-x4plus:fp16",
                2 << 30,
                OwnerWork::Probe {
                    id: "utility-during-chain".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );

        coordinator.dispatch_ready().await;

        assert!(
            matches!(
                worker_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "a legacy chain stage owns this CUDA device; owner work must not overlap it"
        );
        assert!(coordinator
            .pending_owner_work
            .contains_key("utility-during-chain"));
    }

    #[tokio::test]
    async fn legacy_chain_claim_after_plan_fences_owner_transport() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "utility-racing-chain",
                "real-esrgan-x4plus:fp16",
                2 << 30,
                OwnerWork::Probe {
                    id: "utility-racing-chain".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );
        let plan_built = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        coordinator.before_grant_hook = Some(BeforeGrantHook {
            plan_built: plan_built.clone(),
            resume: resume.clone(),
        });
        let dispatch = tokio::spawn(async move {
            let _ = coordinator.dispatch_ready().await;
            coordinator
        });

        plan_built.notified().await;
        assert!(
            worker.try_claim_in_flight(),
            "legacy chain wins the atomic device claim"
        );
        resume.notify_one();
        let coordinator = dispatch.await.unwrap();

        assert!(
            worker_rx.try_recv().is_err(),
            "a stale plan must not transport owner work after a chain claim"
        );
        assert!(coordinator
            .pending_owner_work
            .contains_key("utility-racing-chain"));
        worker.release_in_flight();
    }

    #[tokio::test]
    async fn exhausted_legacy_chain_bypass_yields_with_bounded_dispatch_backoff() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "utility-behind-chain",
                "real-esrgan-x4plus:fp16",
                2 << 30,
                OwnerWork::Probe {
                    id: "utility-behind-chain".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );

        let waiter = crate::gpu_pool::LegacyChainWaiter::new();
        worker.register_legacy_chain_waiter(&waiter);
        for _ in 0..crate::gpu_pool::MAX_OWNER_BYPASSES_FOR_CHAIN {
            assert!(worker.try_claim_owner_in_flight());
            worker.release_in_flight();
        }
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);

        coordinator.dispatch_retry_round = 2;
        coordinator.dispatch_retry_not_before_ms = Some(0);
        let plan_version_before = coordinator.plan_version;
        assert_eq!(coordinator.dispatch_ready().await, None);

        assert_eq!(
            coordinator.plan_version.saturating_sub(plan_version_before),
            u64::from(MAX_DISPATCH_REPLANS_PER_TURN)
        );
        assert_eq!(coordinator.dispatch_retry_round, 3);
        assert!(coordinator
            .dispatch_retry_not_before_ms
            .is_some_and(|deadline| deadline > monotonic_ms()));
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
        assert!(worker_rx.try_recv().is_err());
        assert!(coordinator
            .pending_owner_work
            .contains_key("utility-behind-chain"));

        let yielded_plan_version = coordinator.plan_version;
        assert_eq!(coordinator.dispatch_ready().await, None);
        assert_eq!(coordinator.plan_version, yielded_plan_version);
        worker.unregister_legacy_chain_waiter(&waiter);
    }

    #[test]
    fn legacy_chain_claim_and_release_trigger_coordinator_replans() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.reconcile_external_mutations(&mut immediate);
        immediate = false;

        assert!(worker.try_claim_in_flight());
        coordinator.reconcile_external_mutations(&mut immediate);
        assert!(immediate, "a legacy chain claim must invalidate idle plans");

        immediate = false;
        worker.release_in_flight();
        coordinator.reconcile_external_mutations(&mut immediate);
        assert!(
            immediate,
            "a legacy chain release must promptly reopen scheduler capacity"
        );
    }

    #[tokio::test]
    async fn cancelled_owner_work_is_removed_before_any_worker_grant() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        drop(result_rx);
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "cancelled-admin",
                "admin-unload",
                0,
                OwnerWork::AdminModelUnload(Box::new(crate::gpu_pool::AdminModelUnloadJob {
                    id: "cancelled-admin".to_string(),
                    model: None,
                    evict_cached: false,
                    result_tx,
                })),
            )
            .with_priority(PriorityClass::Admin),
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.reconcile_external_mutations(&mut immediate);
        coordinator.dispatch_ready().await;

        assert!(coordinator.pending_owner_work.is_empty());
        assert!(worker_rx.try_recv().is_err());
        assert!(coordinator.leases.is_empty());
    }

    #[tokio::test]
    async fn post_upscale_followup_waits_for_a_distinct_lease_after_generation_completion() {
        let (worker, worker_rx) = test_worker(0);
        let device_id = worker_device_id(&worker);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let (generation, _result) = fake_generation("parent");
        let parent_fence = LeaseFence {
            work_id: "parent".to_string(),
            device_id: device_id.clone(),
            owner_epoch: 1,
            state_version: 1,
            plan_version: 1,
            worker_generation: 1,
            memory_sample_generation: 1,
            memory_ledger_sequence: 1,
        };
        let gpu_job = gpu_job_from_generation(&state, generation, parent_fence.clone(), None, None);
        let child_id = "parent::post-upscale".to_string();
        let followup = ScheduledOwnerWork::new(
            child_id.clone(),
            "real-esrgan-x4plus:fp16",
            2 << 30,
            OwnerWork::PostUpscale(Box::new(crate::gpu_pool::PostGenerationUpscaleJob {
                id: child_id.clone(),
                generation: Box::new(gpu_job),
                response: mold_core::GenerateResponse {
                    images: Vec::new(),
                    video: None,
                    generation_time_ms: 1,
                    model: "flux-dev:q4".to_string(),
                    seed_used: 1,
                    gpu: Some(0),
                },
                image: mold_core::ImageData {
                    data: vec![1],
                    format: mold_core::OutputFormat::Png,
                    width: 64,
                    height: 64,
                    index: 0,
                },
            })),
        );
        let sampler =
            ScriptedHostMemorySampler::new([memory_reading(128, 112), memory_reading(128, 104)]);
        let mut coordinator = Coordinator::with_preparer_and_sampler(
            state,
            Arc::new(ImmediatePreparer),
            sampler.clone(),
        );
        coordinator.leases.insert(
            device_id.clone(),
            ActiveLease {
                work_id: "parent".to_string(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: 1,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey::default(),
            },
        );
        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::FollowupReady {
                work: Box::new(followup),
            },
            &mut immediate,
        );
        assert!(coordinator.pending_owner_work.contains_key(&child_id));
        assert!(worker_rx.try_recv().is_err());

        coordinator.handle_worker_event(
            WorkerEvent::Completed {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                successful: false,
                load_ms: None,
            },
            &mut immediate,
        );
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            coordinator
                .memory
                .sample
                .map(|sample| sample.available_bytes),
            Some(104 << 30),
            "completion must resample through the injected production path"
        );
        assert_eq!(
            coordinator.estimates.len(),
            0,
            "failed owner work must never train the estimator"
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id,
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 2,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;

        match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => {
                assert_eq!(grant.fence.work_id, child_id);
                assert_eq!(grant.work.kind(), mold_scheduler::WorkKind::PostUpscale);
                assert_eq!(grant.fence.worker_generation, 2);
            }
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        }
    }

    #[tokio::test]
    async fn cancellation_after_plan_before_grant_is_acknowledged_and_never_transported() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 4);
        let (job, mut result) = fake_generation("cancel-race");
        state.job_registry.register("cancel-race", "flux-dev:q4");
        queue.submit(job, 4).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("cancel-race")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        let plan_built = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        coordinator.before_grant_hook = Some(BeforeGrantHook {
            plan_built: plan_built.clone(),
            resume: resume.clone(),
        });
        let dispatch = tokio::spawn(async move {
            let _ = coordinator.dispatch_ready().await;
            coordinator
        });
        plan_built.notified().await;
        {
            let _fence = state.scheduler_mutation_fence.lock().await;
            state.job_registry.cancel_queued("cancel-race").unwrap();
        }
        resume.notify_one();
        let coordinator = dispatch.await.unwrap();

        assert!(
            worker_rx.try_recv().is_err(),
            "a plan invalidated by acknowledged cancel must not reach a worker"
        );
        assert!(!coordinator.pending.contains_key("cancel-race"));
        let outcome = tokio::time::timeout(Duration::from_secs(1), &mut result)
            .await
            .expect("cancelled generation must settle")
            .expect("cancel result sender");
        assert!(outcome.is_err());
    }

    #[tokio::test]
    async fn telemetry_change_after_plan_before_grant_replans_without_transport() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 4);
        let (job, _result) = fake_generation("telemetry-race");
        state.job_registry.register("telemetry-race", "flux-dev:q4");
        queue.submit(job, 4).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("telemetry-race")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        let plan_built = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        coordinator.before_grant_hook = Some(BeforeGrantHook {
            plan_built: plan_built.clone(),
            resume: resume.clone(),
        });
        let dispatch = tokio::spawn(async move {
            let _ = coordinator.dispatch_ready().await;
            coordinator
        });

        plan_built.notified().await;
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 2,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu-0".to_string(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24 << 30,
                vram_used: 23 << 30,
                vram_used_by_mold: Some(0),
                vram_used_by_other: Some(23 << 30),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 128 << 30,
                used: 1 << 30,
                available: Some(127 << 30),
                used_by_mold: 0,
                used_by_other: 1 << 30,
            },
            cpu: None,
        });
        resume.notify_one();
        let coordinator = dispatch.await.unwrap();

        assert!(
            worker_rx.try_recv().is_err(),
            "a candidate invalidated by a lower free-VRAM sample must never reach the worker"
        );
        assert!(coordinator.pending.contains_key("telemetry-race"));
        assert!(coordinator.leases.is_empty());
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn config_artifact_change_after_plan_transports_only_the_replanned_fingerprint() {
        let root = tempfile::TempDir::new().unwrap();
        let transformer = root.path().join("transformer-q4.gguf");
        let vae = root.path().join("vae.safetensors");
        let encoder = root.path().join("t5.safetensors");
        std::fs::write(&transformer, b"old-transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        std::fs::write(&encoder, b"encoder").unwrap();
        let mut config = mold_core::Config::default();
        config.models.insert(
            "config-race:q4".to_string(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                t5_encoder: Some(encoder.display().to_string()),
                family: Some("flux2".to_string()),
                ..mold_core::ModelConfig::default()
            },
        );

        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(config, queue.clone(), pool, 4);
        let (mut job, _result) = fake_generation("config-race");
        job.request.model = "config-race:q4".to_string();
        state.job_registry.register("config-race", "config-race:q4");
        queue.submit(job, 4).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("config-race")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        let (_, original_catalog) = coordinator.planner_snapshot();
        let original_fingerprint = original_catalog["config-race"][0]
            .execution_fingerprint
            .clone();
        let plan_built = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        coordinator.before_grant_hook = Some(BeforeGrantHook {
            plan_built: plan_built.clone(),
            resume: resume.clone(),
        });
        let dispatch = tokio::spawn(async move {
            let _ = coordinator.dispatch_ready().await;
            coordinator
        });

        plan_built.notified().await;
        std::fs::write(&transformer, b"new-transformer").unwrap();
        resume.notify_one();
        let coordinator = dispatch.await.unwrap();

        let grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        let transported_fingerprint = match &grant.work {
            OwnerWork::Generation(job) => job
                .execution_plan
                .as_ref()
                .expect("planned generation")
                .execution_fingerprint
                .clone(),
            other => panic!("unexpected work kind: {:?}", other.kind()),
        };
        assert_ne!(
            transported_fingerprint, original_fingerprint,
            "the pre-mutation execution plan must never cross the transport boundary"
        );
        assert_eq!(coordinator.leases.len(), 1);
    }

    #[tokio::test]
    async fn fatal_cuda_after_plan_before_grant_stops_dispatch_without_spinning() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 4);
        let (job, _result) = fake_generation("fatal-race");
        state.job_registry.register("fatal-race", "flux-dev:q4");
        queue.submit(job, 4).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("fatal-race")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        let plan_built = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        coordinator.before_grant_hook = Some(BeforeGrantHook {
            plan_built: plan_built.clone(),
            resume: resume.clone(),
        });
        let dispatch = tokio::spawn(async move {
            let result =
                tokio::time::timeout(Duration::from_secs(1), coordinator.dispatch_ready()).await;
            (result, coordinator)
        });
        plan_built.notified().await;
        worker.fatal_cuda_error.store(true, Ordering::SeqCst);
        resume.notify_one();
        let (result, coordinator) = dispatch.await.unwrap();

        assert_eq!(result.expect("fatal fence must terminate dispatch"), None);
        assert!(
            worker_rx.try_recv().is_err(),
            "fatal state raised before the grant fence must stop transport"
        );
        assert!(coordinator.pending.contains_key("fatal-race"));
        assert!(coordinator.leases.is_empty());
    }

    #[tokio::test]
    async fn deterministic_plan_error_rejects_instead_of_remaining_zero_candidate_work() {
        let root = tempfile::TempDir::new().unwrap();
        let transformer = root.path().join("sdxl.safetensors");
        let vae = root.path().join("vae.safetensors");
        let encoder = root.path().join("clip.safetensors");
        for (path, bytes) in [
            (&transformer, 2_u64 << 30),
            (&vae, 512_u64 << 20),
            (&encoder, 512_u64 << 20),
        ] {
            let file = std::fs::File::create(path).unwrap();
            file.set_len(bytes).unwrap();
        }
        let mut config = mold_core::Config::default();
        config.models.insert(
            "test-terminal:q4".to_string(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                t5_encoder: Some(encoder.display().to_string()),
                family: Some("sdxl".to_string()),
                ..mold_core::ModelConfig::default()
            },
        );
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(config, queue.clone(), pool, 4);
        let (mut job, mut result) = fake_generation("terminal-plan-error");
        job.request.model = "test-terminal:q4".to_string();
        job.request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Cpu,
            advanced: None,
        });
        state
            .job_registry
            .register("terminal-plan-error", "test-terminal:q4");
        queue.submit(job, 4).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("terminal-plan-error")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.dispatch_ready().await;

        assert!(
            worker_rx.try_recv().is_err(),
            "terminal resolution must not wait for an idle worker"
        );
        assert!(!coordinator.pending.contains_key("terminal-plan-error"));
        assert_eq!(queue.pending(), 0);
        let error = match result
            .try_recv()
            .expect("terminal planning failure must settle the result")
        {
            Ok(_) => panic!("unsupported placement must fail"),
            Err(error) => error,
        };
        assert!(error.contains("pinned to CPU"));
    }

    #[tokio::test]
    async fn owner_plan_invalidation_requeues_and_replans_without_double_release() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 4);
        let (job, mut result) = fake_generation("plan-invalidated");
        state
            .job_registry
            .register("plan-invalidated", "flux-dev:q4");
        queue.submit(job, 4).await.unwrap();

        let sampler = ScriptedHostMemorySampler::new([
            memory_reading(128, 112),
            memory_reading(128, 104),
            memory_reading(128, 104),
            memory_reading(128, 104),
        ]);
        let mut coordinator = Coordinator::with_preparer_and_sampler(
            state.clone(),
            Arc::new(ImmediatePreparer),
            sampler.clone(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        {
            let pending = coordinator.pending.get_mut("plan-invalidated").unwrap();
            pending.preparation = PreparationState::Ready;
            pending.bypass_count = 2;
            pending.warm_wait_started_ms = Some(17);
        }
        let device_id = worker_device_id(&worker);
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;

        let mut first_grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        let first_plan_version = first_grant.fence.plan_version;
        // A worker must return the transported payload even if every mutable
        // fence generation disagrees with the coordinator. The reducer owns
        // the exactly-once requeue/terminal decision from this point.
        first_grant.fence.state_version = first_grant.fence.state_version.saturating_add(100);
        first_grant.fence.plan_version = first_grant.fence.plan_version.saturating_add(100);
        first_grant.fence.worker_generation =
            first_grant.fence.worker_generation.saturating_add(100);
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 1);
        assert_eq!(coordinator.memory.reservations.len(), 1);
        assert_eq!(
            state.job_registry.entry("plan-invalidated").unwrap().state,
            crate::job_registry::JobLifecycle::Running
        );

        coordinator.handle_worker_event(
            WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                grant: first_grant,
                reason: LeaseRejection::PlanInvalidated(
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "artifact changed".to_string(),
                    ),
                ),
            },
            &mut immediate,
        );

        assert!(coordinator.leases.is_empty());
        assert!(coordinator.memory.reservations.is_empty());
        assert_eq!(
            worker.in_flight.load(Ordering::SeqCst),
            0,
            "the invalidated claim must be released exactly once"
        );
        assert!(coordinator.pending.contains_key("plan-invalidated"));
        assert_eq!(coordinator.pending["plan-invalidated"].bypass_count, 2);
        assert_eq!(
            coordinator.pending["plan-invalidated"].warm_wait_started_ms,
            Some(17),
            "plan invalidation must preserve starvation and warm-wait state"
        );
        assert_eq!(
            state.job_registry.entry("plan-invalidated").unwrap().state,
            crate::job_registry::JobLifecycle::Queued
        );
        assert_eq!(queue.pending(), 1);
        assert!(matches!(
            result.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(
            coordinator.plan_invalidations.get("plan-invalidated"),
            Some(&1)
        );
        assert!(
            coordinator.pending["plan-invalidated"]
                .retry_not_before_ms
                .is_some_and(|deadline| deadline > monotonic_ms()),
            "the invalidated plan must back off before redispatch"
        );
        coordinator
            .pending
            .get_mut("plan-invalidated")
            .unwrap()
            .retry_not_before_ms = Some(0);

        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;
        let second_grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        assert_eq!(second_grant.fence.work_id, "plan-invalidated");
        assert!(second_grant.fence.plan_version > first_plan_version);
        assert_eq!(second_grant.fence.worker_generation, 1);
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 1);
        assert_eq!(coordinator.memory.reservations.len(), 1);

        coordinator.handle_worker_event(
            WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                grant: second_grant,
                reason: LeaseRejection::PlanInvalidated(
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "artifact changed again".to_string(),
                    ),
                ),
            },
            &mut immediate,
        );
        coordinator
            .pending
            .get_mut("plan-invalidated")
            .unwrap()
            .retry_not_before_ms = Some(0);
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;
        let third_grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        coordinator.handle_worker_event(
            WorkerEvent::Rejected {
                device_id,
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                grant: third_grant,
                reason: LeaseRejection::PlanInvalidated(
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "artifact never stabilized".to_string(),
                    ),
                ),
            },
            &mut immediate,
        );

        assert!(!coordinator.pending.contains_key("plan-invalidated"));
        assert!(!coordinator
            .plan_invalidations
            .contains_key("plan-invalidated"));
        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
        assert_eq!(queue.pending(), 0);
        let error = match result
            .try_recv()
            .expect("bounded invalidation retries must settle the result")
        {
            Ok(_) => panic!("repeated invalidation must fail"),
            Err(error) => error,
        };
        assert!(error.contains("invalidated 3 consecutive times"));
        assert_eq!(
            sampler.calls.load(Ordering::SeqCst),
            4,
            "initial collection plus every rejected retry must use the injected sampler"
        );
    }

    struct SelectiveBlockingPreparer {
        release_blocked: Arc<tokio::sync::Notify>,
    }

    impl DependencyPreparer for SelectiveBlockingPreparer {
        fn prepare(
            &self,
            _state: AppState,
            _work_id: String,
            request: mold_core::GenerateRequest,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
        ) -> PreparationFuture {
            let release = self.release_blocked.clone();
            Box::pin(async move {
                if request.prompt == "blocked-preparation" {
                    release.notified().await;
                }
                Ok(PreparedGeneration::default())
            })
        }
    }

    #[tokio::test]
    async fn blocked_dependency_preparation_does_not_block_other_ready_gpu_work() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 4);
        let (mut blocked, _blocked_result) = fake_generation("blocked");
        blocked.request.prompt = "blocked-preparation".to_string();
        let (ready, _ready_result) = fake_generation("ready");
        for id in ["blocked", "ready"] {
            state.job_registry.register(id, "flux-dev:q4");
        }
        queue.submit(blocked, 4).await.unwrap();
        queue.submit(ready, 4).await.unwrap();

        let release = Arc::new(tokio::sync::Notify::new());
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(SelectiveBlockingPreparer {
                release_blocked: release.clone(),
            }),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator.start_needed_preparations();
        let event = tokio::time::timeout(Duration::from_secs(1), coordinator.preparation_rx.recv())
            .await
            .expect("ready preparation must complete")
            .expect("preparation event");
        assert!(matches!(
            &event,
            PreparationEvent::Ready { work_id, .. } if work_id == "ready"
        ));
        coordinator.handle_preparation_event(event, &mut immediate);
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        let _ = coordinator.dispatch_ready().await;
        assert_eq!(recv_grant(&worker_rx).id, "ready");
        assert_eq!(
            coordinator.pending.get("blocked").unwrap().preparation,
            PreparationState::Preparing
        );
        release.notify_waiters();
        coordinator.stop_preparations().await;
    }

    #[tokio::test]
    async fn partial_transport_failure_replans_remaining_ready_capacity_same_turn() {
        let (worker_a, rx_a) = test_worker(0);
        let (worker_b, rx_b) = test_worker(1);
        let (worker_c, rx_c) = test_worker(2);
        drop(rx_b);
        let pool = Arc::new(GpuPool {
            workers: vec![worker_a.clone(), worker_b.clone(), worker_c.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 8);
        let mut results = Vec::new();
        for id in ["a", "b", "c"] {
            state.job_registry.register(id, "flux-dev:q4");
            let (job, result) = fake_generation(id);
            results.push(result);
            queue.submit(job, 8).await.unwrap();
        }
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        for _ in 0..3 {
            coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        }
        for pending in coordinator.pending.values_mut() {
            pending.preparation = PreparationState::Ready;
        }
        for (worker, ordinal) in [(&worker_a, 0), (&worker_b, 1), (&worker_c, 2)] {
            coordinator.handle_worker_event(
                WorkerEvent::Ready {
                    device_id: worker_device_id(worker),
                    ordinal,
                    owner_epoch: 1,
                    worker_generation: 1,
                },
                &mut immediate,
            );
        }
        let _ = coordinator.dispatch_ready().await;

        assert!(matches!(
            rx_a.recv_timeout(Duration::from_secs(1)),
            Ok(crate::gpu_pool::GpuWorkerCommand::Grant(_))
        ));
        assert!(matches!(
            rx_c.recv_timeout(Duration::from_secs(1)),
            Ok(crate::gpu_pool::GpuWorkerCommand::Grant(_))
        ));
        assert_eq!(coordinator.leases.len(), 2);
        assert_eq!(
            coordinator.memory.reservations.len(),
            2,
            "failed grant reservation must settle before same-turn replan"
        );
        drop(results);
    }

    #[test]
    fn partial_forward_progress_resets_the_cross_turn_dispatch_retry_round() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        coordinator.dispatch_retry_round = 7;
        coordinator.dispatch_retry_not_before_ms = Some(monotonic_ms().saturating_add(10_000));

        coordinator.record_dispatch_progress();

        assert_eq!(coordinator.dispatch_retry_round, 0);
        assert_eq!(coordinator.dispatch_retry_not_before_ms, None);
    }

    #[test]
    fn partial_materialization_failure_remains_a_refresh_candidate() {
        let mut prepared = crate::execution_plan::PreparedExecutionInputs::default();
        prepared.retryable_device_failures.insert(
            "cuda:1".to_string(),
            "temporary dependency download failure".to_string(),
        );
        let signature = Coordinator::preparation_refresh_signature(
            &prepared,
            &[crate::execution_plan::DeviceFact {
                id: "cuda:1".to_string(),
                ordinal: 1,
                available_vram_bytes: 24 << 30,
            }],
        );
        assert_eq!(signature, vec![("cuda:1".to_string(), 0)]);
    }

    #[test]
    fn auto_capacity_refresh_requires_a_material_sustained_change() {
        assert_eq!(capacity_refresh_direction(24 << 30, 23 << 30, true), None);
        assert_eq!(
            capacity_refresh_direction(24 << 30, 16 << 30, false),
            None,
            "an explicit variant must not churn with telemetry"
        );
        assert_eq!(
            capacity_refresh_direction(24 << 30, 16 << 30, true),
            Some(-1)
        );

        let signature = vec![("cuda:0".to_string(), -1)];
        let mut observation = None;
        assert!(!observe_preparation_refresh(
            &mut observation,
            signature.clone(),
            10_000,
            1_000
        ));
        assert!(!observe_preparation_refresh(
            &mut observation,
            signature.clone(),
            10_999,
            1_000
        ));
        assert!(observe_preparation_refresh(
            &mut observation,
            signature,
            11_000,
            1_000
        ));
        assert!(
            !observe_preparation_refresh(
                &mut observation,
                vec![("cuda:0".to_string(), 1)],
                11_001,
                1_000
            ),
            "direction changes restart the stability window"
        );
    }

    #[test]
    fn exact_sample_churn_retry_is_bounded_and_backed_off() {
        assert_eq!(MAX_DISPATCH_REPLANS_PER_TURN, 3);
        assert_eq!(dispatch_retry_delay_ms(1), 25);
        assert_eq!(dispatch_retry_delay_ms(2), 50);
        assert_eq!(dispatch_retry_delay_ms(3), 100);
        assert_eq!(dispatch_retry_delay_ms(u8::MAX), DISPATCH_RETRY_MAX_MS);
    }
}
