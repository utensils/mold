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
    AssignmentReason, Backend, BlockedReason, CandidatePlacement, DeviceActivity, DeviceAdminState,
    DeviceHealth, DeviceId, DeviceSnapshot, EstimateBucket, EstimateKey, EstimateObservation,
    EstimateOutcome, EstimatePhaseTimings, EstimateStore, ExecutionFingerprint,
    GrantValidationSnapshot, HostMemorySnapshot, Plan, PlannedAssignment, Planner, PlannerError,
    PlannerSnapshot, PriorityClass, ResolvedEstimate, StaticEstimate, WorkId, WorkKind,
    WorkSnapshot,
};

use crate::gpu_pool::{
    GpuJob, GpuWorker, LeaseGrant, OwnerWork, UtilityExecutionPlan, UtilityPlacement,
};
use crate::state::{AppState, GenerationJob, SseMessage};

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
/// How many dependency preparations may resolve variants and download weights
/// at once.
///
/// Preparation is unbounded fan-out in front of a GPU that runs one job at a
/// time: a burst of submissions used to start one resolution per job, all of
/// them competing for the same disk and host RAM to stage work the scheduler
/// cannot dispatch for minutes. Two keeps a single stuck download from
/// stalling the next ready job — the property
/// `blocked_dependency_preparation_does_not_block_other_ready_gpu_work` pins —
/// without letting the queue depth set the parallelism.
const MAX_CONCURRENT_PREPARATIONS: usize = 2;
const DISPATCH_RETRY_BASE_MS: u64 = 25;
const DISPATCH_RETRY_MAX_MS: u64 = 1_000;
/// How long a queued generation with an unclassified empty plan set may keep
/// waiting once the scheduler holds nothing that could change that result.
///
/// Zero-candidate work is reconsidered on every planning turn and dispatched
/// the moment a plan resolves, which is the right answer while something is
/// running: the running job's VRAM and host reservation come back. On an
/// otherwise idle scheduler nothing will change on its own, and #1272 is what
/// that looks like from a client — an H3 print reported `no_schedulable_device`
/// three times, then nothing at all for the rest of a forty-minute wait, with
/// no failure, no message, and a queue row that outlived the request. Typed
/// transient failures are excluded: external resource pressure remains queued
/// for however long it exists and wakes on a changed resource sample.
const UNSCHEDULABLE_IDLE_GRACE_MS: u64 = 60_000;
pub(crate) const CPU_UTILITY_DEVICE_ID: &str = "cpu:utility:0";

pub(crate) fn generation_hard_ordinal(
    state: &AppState,
    id: &str,
    request: &mold_core::GenerateRequest,
) -> Option<usize> {
    let request_pin = state
        .gpu_pool
        .resolve_explicit_placement_gpu(request.placement.as_ref())
        .ok()
        .flatten();
    request_pin.or_else(|| state.job_registry.target_gpu(id).flatten())
}

fn constrained_generation_device_facts(
    devices: &[crate::execution_plan::DeviceFact],
    hard_ordinal: Option<usize>,
    required_device_id: Option<&str>,
) -> Vec<crate::execution_plan::DeviceFact> {
    devices
        .iter()
        .filter(|device| hard_ordinal.is_none_or(|ordinal| device.ordinal == ordinal))
        .filter(|device| required_device_id.is_none_or(|id| device.id == id))
        .cloned()
        .collect()
}

fn wire_estimate_confidence(
    confidence: mold_scheduler::EstimateConfidence,
) -> mold_core::QueueEstimateConfidence {
    match confidence {
        mold_scheduler::EstimateConfidence::Low => mold_core::QueueEstimateConfidence::Low,
        mold_scheduler::EstimateConfidence::Medium => mold_core::QueueEstimateConfidence::Medium,
        mold_scheduler::EstimateConfidence::High => mold_core::QueueEstimateConfidence::High,
    }
}

fn stage_placement_candidate(
    stage_index: u32,
    copy_index: Option<u32>,
    assignment: &PlannedAssignment,
    devices: &[DeviceSnapshot],
    now_ms: u64,
    estimate_confidence: mold_core::QueueEstimateConfidence,
) -> mold_core::ChainStagePlacementCandidate {
    let warm = devices.iter().any(|device| {
        device.id == assignment.device_id
            && device
                .warm_execution_fingerprints
                .contains(&assignment.placement.execution_fingerprint)
    });
    mold_core::ChainStagePlacementCandidate {
        stage_index,
        copy_index,
        candidate: mold_core::GenerationPlacementCandidate {
            device_id: assignment.device_id.to_string(),
            execution_fingerprint: assignment.placement.execution_fingerprint.to_string(),
            execution_equivalence_fingerprint: assignment
                .placement
                .execution_equivalence_fingerprint
                .as_ref()
                .map(ToString::to_string),
            predicted_start_after_ms: assignment.estimated_start_ms.saturating_sub(now_ms),
            predicted_completion_after_ms: assignment.estimated_finish_ms.saturating_sub(now_ms),
            setup_ms: if warm {
                assignment.placement.warm_setup_ms
            } else {
                assignment.placement.cold_setup_ms
            },
            setup_kind: if warm { "warm" } else { "cold" }.to_string(),
            estimate_confidence,
        },
    }
}

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
    /// Request one fresh host-memory sample reconciled with every Scheduler V2
    /// reservation. The reply restores only this lease's own reservation so a
    /// worker cannot spend headroom already promised to a peer GPU.
    HostMemoryRecheck {
        fence: LeaseFence,
        reply: std::sync::mpsc::SyncSender<Result<u64, String>>,
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
        phase_timings: EstimatePhaseTimings,
        /// Actor-only result authority transferred from the GPU owner. The
        /// coordinator settles it only after the matching lease, memory
        /// reservation, timing observation, and published plan are updated.
        completion: Option<Box<crate::gpu_worker::DeferredOwnerCompletion>>,
    },
    Stopped {
        device_id: String,
        ordinal: usize,
        owner_epoch: u64,
    },
}

fn reject_owner_work_preserving_completed_generation(work: OwnerWork, error: String) {
    match work {
        OwnerWork::PostUpscale(job) => {
            crate::gpu_worker::finish_post_generation_upscale_failure(job, error);
        }
        work => work.reject(error),
    }
}

pub struct ScheduledOwnerWork {
    pub id: String,
    pub model_fingerprint: String,
    pub estimated_vram_bytes: u64,
    pub estimated_host_ram_bytes: u64,
    pub hard_ordinal: Option<usize>,
    pub priority: PriorityClass,
    /// Soft stage-to-stage locality preference. This never filters candidates.
    pub preferred_ordinal: Option<usize>,
    /// Optional immutable per-device plans for work that loads an inference
    /// engine outside the ordinary generation payload.
    pub candidate_plans: Vec<crate::execution_plan::ResolvedExecutionPlan>,
    /// Frozen alternatives for utility stages. Empty retains the legacy
    /// GPU-only estimate path for administrative work.
    pub utility_plans: Vec<UtilityExecutionPlan>,
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
            preferred_ordinal: None,
            candidate_plans: Vec::new(),
            utility_plans: Vec::new(),
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

    pub fn with_preferred_ordinal(mut self, ordinal: Option<usize>) -> Self {
        self.preferred_ordinal = ordinal;
        self
    }

    pub fn with_candidate_plans(
        mut self,
        plans: Vec<crate::execution_plan::ResolvedExecutionPlan>,
    ) -> Self {
        self.candidate_plans = plans;
        self
    }

    pub fn with_utility_plans(mut self, utility_plans: Vec<UtilityExecutionPlan>) -> Self {
        self.utility_plans = utility_plans;
        self
    }
}

#[derive(Clone)]
pub struct ScheduledWorkHandle {
    tx: Option<tokio::sync::mpsc::Sender<ScheduledOwnerWork>>,
    preview_tx: Option<tokio::sync::mpsc::Sender<PlacementPreviewQuery>>,
    dispatch_mode: crate::dispatch_mode::DispatchMode,
    v2_authoritative: bool,
    observes_v2_decisions: bool,
    observations: Arc<crate::dispatch_mode::DispatchObservationRecorder>,
    latest_plan: Arc<std::sync::RwLock<Option<mold_core::QueuePlan>>>,
    /// Most recent host-RAM reading from the coordinator's admission ledger,
    /// so `/api/status` reports the same numbers admission spends. `None`
    /// until the first sample, and permanently `None` wherever the coordinator
    /// does not run.
    latest_host_memory: Arc<std::sync::RwLock<Option<mold_core::HostMemorySnapshot>>>,
}

impl Default for ScheduledWorkHandle {
    fn default() -> Self {
        Self {
            tx: None,
            preview_tx: None,
            dispatch_mode: crate::dispatch_mode::DispatchMode::V2,
            v2_authoritative: false,
            observes_v2_decisions: false,
            observations: Arc::new(crate::dispatch_mode::DispatchObservationRecorder::default()),
            latest_plan: Arc::new(std::sync::RwLock::new(None)),
            latest_host_memory: Arc::new(std::sync::RwLock::new(None)),
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
            preview_tx: None,
            dispatch_mode,
            v2_authoritative,
            observes_v2_decisions,
            observations: Arc::new(crate::dispatch_mode::DispatchObservationRecorder::default()),
            latest_plan: Arc::new(std::sync::RwLock::new(None)),
            latest_host_memory: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    pub const fn dispatch_mode(&self) -> crate::dispatch_mode::DispatchMode {
        self.dispatch_mode
    }

    /// Latest host-RAM reading, or `None` when nothing has sampled it.
    pub fn host_memory(&self) -> Option<mold_core::HostMemorySnapshot> {
        *self
            .latest_host_memory
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn publish_host_memory(&self, snapshot: Option<mold_core::HostMemorySnapshot>) {
        *self
            .latest_host_memory
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = snapshot;
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

    pub fn with_placement_preview(
        mut self,
        preview_tx: tokio::sync::mpsc::Sender<PlacementPreviewQuery>,
    ) -> Self {
        self.preview_tx = Some(preview_tx);
        self
    }

    pub fn placement_preview_available(&self) -> bool {
        self.preview_tx.is_some()
    }

    pub async fn preview_placement(
        &self,
        request: mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: crate::execution_plan::PreparedExecutionInputs,
    ) -> Result<mold_core::GenerationPlacementPreview, String> {
        if !self.v2_authoritative {
            return Err("authoritative scheduler placement preview is unavailable".to_string());
        }
        let Some(tx) = &self.preview_tx else {
            return Err("scheduler placement preview channel is unavailable".to_string());
        };
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        tx.send(PlacementPreviewQuery::Generation {
            request,
            copies,
            prepared_inputs,
            reply_tx,
        })
        .await
        .map_err(|_| "scheduler placement preview is shutting down".to_string())?;
        reply_rx
            .await
            .map_err(|_| "scheduler placement preview was cancelled".to_string())
    }

    pub async fn batch_device_profiles(
        &self,
        request: mold_core::GenerateRequest,
        parent_size: u32,
        prepared_inputs: crate::execution_plan::PreparedExecutionInputs,
    ) -> Result<Vec<mold_core::GenerationPlacementCandidate>, String> {
        if !self.v2_authoritative {
            return Err("authoritative scheduler batch profiling is unavailable".to_string());
        }
        let Some(tx) = &self.preview_tx else {
            return Err("scheduler batch profiling channel is unavailable".to_string());
        };
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        tx.send(PlacementPreviewQuery::BatchDevices {
            request,
            parent_size,
            prepared_inputs,
            reply_tx,
        })
        .await
        .map_err(|_| "scheduler batch profiling is shutting down".to_string())?;
        reply_rx
            .await
            .map_err(|_| "scheduler batch profiling was cancelled".to_string())?
    }

    pub async fn set_queue_paused(&self, paused: bool) -> Result<bool, String> {
        if !self.v2_authoritative {
            return Err("authoritative scheduler queue control is unavailable".to_string());
        }
        let Some(tx) = &self.preview_tx else {
            return Err("scheduler queue control channel is unavailable".to_string());
        };
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        tx.send(PlacementPreviewQuery::SetQueuePaused { paused, reply_tx })
            .await
            .map_err(|_| "scheduler queue control is shutting down".to_string())?;
        reply_rx
            .await
            .map_err(|_| "scheduler queue control was cancelled".to_string())?
    }

    pub async fn submit(&self, work: ScheduledOwnerWork) -> Result<(), String> {
        let Some(tx) = &self.tx else {
            return Err("GPU scheduler is unavailable".to_string());
        };
        tx.send(work)
            .await
            .map_err(|_| "GPU scheduler is shutting down".to_string())
    }

    #[cfg(test)]
    pub(crate) fn set_queue_work_items_for_tests(&self, items: Vec<mold_core::QueueWorkItem>) {
        *self
            .latest_plan
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(mold_core::QueuePlan {
            plan_version: 7,
            state_version: 11,
            work_items: items,
            ..mold_core::QueuePlan::default()
        });
    }
}

pub enum PlacementPreviewQuery {
    Generation {
        request: mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: crate::execution_plan::PreparedExecutionInputs,
        reply_tx: tokio::sync::oneshot::Sender<mold_core::GenerationPlacementPreview>,
    },
    BatchDevices {
        request: mold_core::GenerateRequest,
        parent_size: u32,
        prepared_inputs: crate::execution_plan::PreparedExecutionInputs,
        reply_tx: tokio::sync::oneshot::Sender<
            Result<Vec<mold_core::GenerationPlacementCandidate>, String>,
        >,
    },
    SetQueuePaused {
        paused: bool,
        reply_tx: tokio::sync::oneshot::Sender<Result<bool, String>>,
    },
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
    vram_high_water_bytes: Option<u64>,
    host_incremental_high_water_bytes: Option<u64>,
    fallback_reason: Option<String>,
    projection: WorkSnapshot,
    assignment_reason: AssignmentReason,
}

fn chain_work_identity(id: &str) -> Option<(&str, u32)> {
    let rest = id.strip_prefix("chain:")?;
    let (identity, stage) = rest.rsplit_once(":stage:")?;
    let parent = identity
        .rsplit_once(":attempt:")
        .map_or(identity, |(parent, _)| parent);
    Some((parent, stage.parse().ok()?))
}

fn reordered_generation_ranks(
    original_ranks: impl IntoIterator<Item = (String, u64)>,
    queue_order: &[String],
) -> BTreeMap<String, u64> {
    let original_ranks = original_ranks.into_iter().collect::<BTreeMap<_, _>>();
    let mut rank_slots = original_ranks.values().copied().collect::<Vec<_>>();
    rank_slots.sort_unstable();
    let mut ordered_ids = queue_order
        .iter()
        .filter(|id| original_ranks.contains_key(*id))
        .cloned()
        .collect::<Vec<_>>();
    let mut unlisted = original_ranks
        .iter()
        .filter(|(id, _)| !ordered_ids.contains(id))
        .map(|(id, rank)| (*rank, id.clone()))
        .collect::<Vec<_>>();
    unlisted.sort();
    ordered_ids.extend(unlisted.into_iter().map(|(_, id)| id));
    ordered_ids.into_iter().zip(rank_slots).collect()
}

struct PendingGeneration {
    job: GenerationJob,
    ready_at_ms: u64,
    queue_rank: u64,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    preparation: PreparationState,
    prepared_inputs: Option<crate::execution_plan::PreparedExecutionInputs>,
    retry_not_before_ms: Option<u64>,
    preparation_retry_attempts: u8,
    preparation_refresh_observation: Option<PreparationRefreshObservation>,
    /// When this job first resolved no execution plan while the scheduler held
    /// nothing that could free capacity. Cleared the moment a plan resolves or
    /// any work is leased, so it only accrues over a genuinely idle wait.
    unschedulable_since_ms: Option<u64>,
    /// An unclassified planning reason, retained so a refusal can name it
    /// instead of a bare "never scheduled".
    unschedulable_reason: Option<String>,
    /// Place in line this job's client was last told about, so a drained queue
    /// re-announces exactly once per actual move. `None` until first observed.
    announced_position: Option<usize>,
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
    preferred_ordinal: Option<usize>,
    candidate_plans: Vec<crate::execution_plan::ResolvedExecutionPlan>,
    queue_rank: u64,
    ready_at_ms: u64,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    retry_not_before_ms: Option<u64>,
    utility_plans: Vec<UtilityExecutionPlan>,
    work: OwnerWork,
}

#[derive(Clone, Copy)]
struct OwnerWorkSchedulingView<'a> {
    id: &'a str,
    model_fingerprint: &'a str,
    estimated_vram_bytes: u64,
    estimated_host_ram_bytes: u64,
    hard_ordinal: Option<usize>,
    priority: PriorityClass,
    queue_rank: u64,
    ready_at_ms: u64,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    kind: mold_scheduler::WorkKind,
    shape_bucket: &'a str,
    preferred_ordinal: Option<usize>,
    resolved_plans: &'a [crate::execution_plan::ResolvedExecutionPlan],
    requires_exact_plan: bool,
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
        prepared: Box<PreparedGeneration>,
    },
    Failed {
        work_id: String,
        error: String,
    },
}

#[derive(Clone, Debug, Default)]
struct PreparedGeneration {
    expanded_prompt: Option<String>,
    resolved_seed: Option<u64>,
    execution_inputs: Option<crate::execution_plan::PreparedExecutionInputs>,
}

/// Private composition seam for scheduler-owned pre-stages.
///
/// Phase E's typed phase timings are recorded by the utility owner callbacks.
/// Keep all pre-stage output reduction here so that execution never exposes a
/// preview whose later stages have not yet frozen and executed the same plans.
fn compose_prepared_generation(pending: &mut PendingGeneration, prepared: PreparedGeneration) {
    if let Some(seed) = prepared.resolved_seed {
        pending.job.request.seed = Some(seed);
    }
    if let Some(expanded_prompt) = prepared.expanded_prompt {
        pending.job.request.original_prompt = Some(pending.job.request.prompt.clone());
        pending.job.request.prompt = expanded_prompt;
    }
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if let Some(grant) = prepared
        .execution_inputs
        .as_ref()
        .and_then(|inputs| inputs.h3_private_ingress_grant.clone())
    {
        pending.job.h3_private_ingress_grant = Some(grant);
    }
    // The identity a parent request froze outlives any re-preparation of one
    // of its children: preparation is handed the frozen value and must return
    // it unchanged, and this is the backstop for a preparer that did not.
    let frozen_identity = pending
        .prepared_inputs
        .as_ref()
        .and_then(|inputs| inputs.identity_embedding.clone());
    // The advisory the extraction produced belongs to the same value and
    // outlives re-preparation for the same reason: a child never extracts, so
    // it would otherwise lose the parent's "which face did you pick" note.
    let identity_warning = pending
        .prepared_inputs
        .as_ref()
        .and_then(|inputs| inputs.identity_warning.clone());
    // The batch-lifetime identity cell outlives re-preparation for the same
    // reason the frozen value does: a fresh preparation mints a fresh cell, and
    // a child that adopted one would stop sharing its parent's.
    let identity_pin = pending
        .prepared_inputs
        .as_ref()
        .map(|inputs| inputs.identity_pin.clone());
    pending.prepared_inputs = prepared.execution_inputs;
    if let (Some(inputs), Some(pin)) = (pending.prepared_inputs.as_mut(), identity_pin) {
        inputs.identity_pin = pin;
    }
    if let (Some(inputs), Some(identity)) = (pending.prepared_inputs.as_mut(), frozen_identity) {
        inputs.identity_embedding = Some(identity);
    }
    if let (Some(inputs), Some(warning)) = (pending.prepared_inputs.as_mut(), identity_warning) {
        inputs.identity_warning.get_or_insert(warning);
    }
}

type PreparationFuture = Pin<Box<dyn Future<Output = Result<PreparedGeneration, String>> + Send>>;

trait DependencyPreparer: Send + Sync {
    fn prepare(
        &self,
        state: AppState,
        work_id: String,
        request: crate::queue_media_runtime::ZeroizingGenerateRequest,
        progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
        context: crate::variant_dependencies::DependencyPreparationContext,
    ) -> PreparationFuture;
}

struct PostUpscalePreparer;

impl DependencyPreparer for PostUpscalePreparer {
    fn prepare(
        &self,
        state: AppState,
        work_id: String,
        request: crate::queue_media_runtime::ZeroizingGenerateRequest,
        progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
        context: crate::variant_dependencies::DependencyPreparationContext,
    ) -> PreparationFuture {
        Box::pin(async move {
            crate::queue::ensure_post_upscale_model_downloaded(&state, &request, progress.as_ref())
                .await?;
            let execution_inputs = crate::variant_dependencies::prepare_execution_inputs(
                &state,
                &work_id,
                &request,
                progress.as_ref(),
                context,
            )
            .await?;
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            let resolved_seed = execution_inputs
                .h3_private_admission_by_device
                .values()
                .next()
                .map(mold_inference::H3PrivateFl2VaAdmissionEvidence::seed);
            #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
            let resolved_seed = None;
            if request.expand != Some(true) {
                return Ok(PreparedGeneration {
                    expanded_prompt: None,
                    resolved_seed,
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
                    resolved_seed,
                    execution_inputs: Some(execution_inputs),
                });
            }
            let family = config
                .resolved_model_config(&request.model)
                .family
                .or(crate::model_manager::family_for_model(&state, &request.model).await)
                .or_else(|| {
                    mold_core::manifest::find_manifest(&request.model)
                        .map(|manifest| manifest.family.clone())
                })
                .unwrap_or_else(|| "flux".to_string());
            let mut expand_config = settings.to_expand_config(&family, 1);
            expand_config.task = mold_core::ExpandTask::for_generation(&family, &request);
            let preferred_gpu = state
                .gpu_pool
                .resolve_explicit_placement_gpu(request.placement.as_ref())?;
            let utility_id = format!("{work_id}::prompt-expansion");
            let cancellation = mold_inference::InferenceCancellationToken::default();
            #[cfg(feature = "expand")]
            let utility_plans =
                prompt_expansion_candidates(&state, &config, Some(&settings.model))?;
            let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
            let utility = ScheduledOwnerWork::new(
                utility_id.clone(),
                settings.model.clone(),
                6_000_000_000,
                OwnerWork::PromptExpansion(Box::new(crate::gpu_pool::PromptExpansionJob {
                    id: utility_id,
                    parent_id: work_id.clone(),
                    config,
                    settings,
                    prompt: request.prompt.clone(),
                    expand_config,
                    cancellation: cancellation.clone(),
                    #[cfg(feature = "expand")]
                    execution_plan: None,
                    result_tx,
                })),
            )
            .with_hard_ordinal(preferred_gpu)
            .with_utility_plans({
                #[cfg(feature = "expand")]
                {
                    utility_plans
                }
                #[cfg(not(feature = "expand"))]
                {
                    Vec::new()
                }
            });
            state.scheduled_work.submit(utility).await?;

            let registry_notify = state.job_registry.mutation_notifier();
            let result = loop {
                tokio::select! {
                    result = &mut result_rx => {
                        break result
                            .map_err(|_| "prompt expansion owner worker dropped its result".to_string())?;
                    }
                    _ = registry_notify.notified() => {
                        if state
                            .job_registry
                            .scheduler_lifecycle(&work_id)
                            .is_none()
                        {
                            cancellation.cancel();
                            return Err(format!(
                                "generation job {work_id} was cancelled during prompt expansion"
                            ));
                        }
                    }
                }
            }?;
            Ok(PreparedGeneration {
                expanded_prompt: result.expanded.first().cloned(),
                resolved_seed,
                execution_inputs: Some(execution_inputs),
            })
        })
    }
}

#[derive(Clone, Debug)]
struct ReplanWindow {
    debounce: Duration,
    max_delay: Duration,
    dirty_since: Option<Instant>,
    last_dirty: Option<Instant>,
    dirty_through_version: u64,
}

impl ReplanWindow {
    fn new(settings: mold_core::config::SchedulerSettings) -> Self {
        Self {
            debounce: Duration::from_millis(u64::from(settings.replan_debounce_ms)),
            max_delay: Duration::from_millis(u64::from(settings.replan_max_delay_ms)),
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
                .checked_add(self.debounce)?
                .min(self.dirty_since?.checked_add(self.max_delay)?),
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
    /// A worker announces its first allocation before gradual model/host
    /// loading. No OS sample can prove the complete frozen increment is
    /// reflected, so this reservation remains charged until its lease settles.
    charge_until_release: bool,
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
            if !reservation.charge_until_release
                && matches!(
                reservation.state,
                ReservationState::CommittedAfterSample { commit_sequence }
                    if commit_sequence < collection_started_sequence
                )
            {
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

    /// A worker announces AllocationCommitted immediately before its first
    /// CUDA construction, while model and host allocations continue after that
    /// point. A fresh OS sample therefore cannot prove that a lease's entire
    /// frozen host increment is reflected. Every live reservation stays
    /// charged until that lease settles.
    fn headroom_bytes(&self) -> u64 {
        let Some(sample) = self.sample else {
            return 0;
        };
        sample.available_bytes.saturating_sub(
            self.safety_floor_bytes()
                .saturating_add(self.bytes_accepted_after_sample_started()),
        )
    }

    fn headroom_for_reserved_work(&self, work_id: &str) -> Option<u64> {
        self.reservations.get(work_id)?;
        let Some(sample) = self.sample else {
            return Some(0);
        };
        let peer_reserved_bytes = self
            .reservations
            .iter()
            .filter(|(candidate, _)| candidate.as_str() != work_id)
            .map(|(_, reservation)| reservation.bytes)
            .sum::<u64>();
        Some(
            sample.available_bytes.saturating_sub(
                self.safety_floor_bytes()
                    .saturating_add(peer_reserved_bytes),
            ),
        )
    }

    fn collect_headroom_for_reserved_work(&mut self, work_id: &str) -> Option<u64> {
        self.collect_now();
        self.headroom_for_reserved_work(work_id)
    }

    /// Client-facing telemetry, or `None` while the sampler has produced no
    /// reading. Absent means unknown — a zeroed snapshot would read as a host
    /// under total memory pressure.
    fn wire_snapshot(&self) -> Option<mold_core::HostMemorySnapshot> {
        let sample = self.sample?;
        Some(mold_core::HostMemorySnapshot {
            total_bytes: sample.total_bytes,
            available_bytes: sample.available_bytes,
            headroom_bytes: self.headroom_bytes(),
            safety_floor_bytes: self.safety_floor_bytes(),
        })
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
                    charge_until_release: item.host_ram_bytes > 0,
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
        // Host memory is a live 1 Hz reading. Letting it decide semantic
        // equality would publish a plan event every second forever.
        plan.host_memory = None;
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

#[derive(Debug)]
enum PlanPublicationError {
    Planner(PlannerError),
    AuthorityConflict {
        current_plan_version: u64,
        current_state_version: u64,
        produced_plan_version: u64,
        produced_state_version: u64,
    },
}

impl std::fmt::Display for PlanPublicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Planner(error) => write!(formatter, "{error}"),
            Self::AuthorityConflict {
                current_plan_version,
                current_state_version,
                produced_plan_version,
                produced_state_version,
            } => write!(
                formatter,
                "current queue-plan authority ({current_plan_version}/{current_state_version}) \
                 conflicts with produced authority ({produced_plan_version}/{produced_state_version})"
            ),
        }
    }
}

impl std::error::Error for PlanPublicationError {}

impl From<PlannerError> for PlanPublicationError {
    fn from(error: PlannerError) -> Self {
        Self::Planner(error)
    }
}

struct Coordinator {
    state: AppState,
    planner: Planner,
    admission_planner: Planner,
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
    preparation_slots: Arc<tokio::sync::Semaphore>,
    last_queue_shape: Vec<(String, Option<usize>)>,
    last_registry_sequence: u64,
    last_paused: bool,
    last_worker_claims: BTreeMap<String, usize>,
    last_device_preferences_sequence: u64,
    last_device_event_signature: Option<DeviceEventSignature>,
    last_resource_capacity_signature: Vec<(String, u64)>,
    device_state_dirty: bool,
    plan_invalidations: BTreeMap<String, u8>,
    dispatch_retry_round: u8,
    dispatch_retry_not_before_ms: Option<u64>,
    /// Grace before an idle, unschedulable generation is settled. A field so a
    /// test can collapse it — the monotonic clock is near zero early in a
    /// process, so "pretend the observation is old" cannot be expressed by
    /// subtracting from now.
    unschedulable_idle_grace_ms: u64,
    estimates: EstimateStore,
    cpu_utility_tx: Option<std::sync::mpsc::SyncSender<crate::gpu_pool::GpuWorkerCommand>>,
    #[cfg(test)]
    next_plan_error: Option<PlannerError>,
    #[cfg(test)]
    before_grant_hook: Option<BeforeGrantHook>,
    #[cfg(test)]
    before_queue_control_plan_hook: Option<Box<dyn FnOnce() + Send>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlanningPass {
    Admission,
    Optimize,
}

#[cfg(test)]
#[derive(Clone)]
struct BeforeGrantHook {
    plan_built: Arc<tokio::sync::Notify>,
    resume: Arc<tokio::sync::Notify>,
}

impl Coordinator {
    async fn new(state: AppState) -> Self {
        let scheduler = state.config.read().await.scheduler;
        Self::with_preparer_and_sampler(
            state,
            Arc::new(PostUpscalePreparer),
            Arc::new(SystemHostMemorySampler),
            scheduler,
        )
    }

    fn with_preparer_and_sampler(
        state: AppState,
        preparer: Arc<dyn DependencyPreparer>,
        sampler: Arc<dyn HostMemorySampler>,
        scheduler: mold_core::config::SchedulerSettings,
    ) -> Self {
        let mut memory = HostMemoryLedger::new(sampler);
        memory.collect_now();
        Self::with_preparer_and_memory_and_settings(state, preparer, memory, scheduler)
    }

    #[cfg(test)]
    fn with_preparer_and_memory(
        state: AppState,
        preparer: Arc<dyn DependencyPreparer>,
        memory: HostMemoryLedger,
    ) -> Self {
        let scheduler = state
            .config
            .try_read()
            .map(|config| config.scheduler)
            .unwrap_or_default();
        Self::with_preparer_and_memory_and_settings(state, preparer, memory, scheduler)
    }

    fn with_preparer_and_memory_and_settings(
        state: AppState,
        preparer: Arc<dyn DependencyPreparer>,
        memory: HostMemoryLedger,
        scheduler: mold_core::config::SchedulerSettings,
    ) -> Self {
        let (preparation_tx, preparation_rx) = tokio::sync::mpsc::unbounded_channel();
        let estimates = load_estimate_store(&state);
        let planner_config = mold_scheduler::PlannerConfig {
            warm_wait_max_ms: u64::from(scheduler.warm_wait_max_ms),
            ..mold_scheduler::PlannerConfig::default()
        };
        let mut coordinator = Self {
            state,
            planner: Planner::new(planner_config.clone()),
            admission_planner: Planner::new(mold_scheduler::PlannerConfig {
                mode: mold_scheduler::PlanningMode::WatchdogFallback,
                ..planner_config
            }),
            pending: BTreeMap::new(),
            pending_owner_work: BTreeMap::new(),
            ready: BTreeMap::new(),
            leases: BTreeMap::new(),
            unavailable: BTreeSet::new(),
            state_version: 0,
            plan_version: 0,
            synthetic_id: 0,
            memory,
            dirty: ReplanWindow::new(scheduler),
            preparer,
            preparation_tx,
            preparation_rx,
            preparation_tasks: tokio::task::JoinSet::new(),
            preparation_slots: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_PREPARATIONS)),
            last_queue_shape: Vec::new(),
            last_registry_sequence: 0,
            last_paused: false,
            last_worker_claims: BTreeMap::new(),
            last_device_preferences_sequence: 0,
            last_device_event_signature: None,
            last_resource_capacity_signature: Vec::new(),
            device_state_dirty: true,
            plan_invalidations: BTreeMap::new(),
            dispatch_retry_round: 0,
            dispatch_retry_not_before_ms: None,
            unschedulable_idle_grace_ms: UNSCHEDULABLE_IDLE_GRACE_MS,
            estimates,
            cpu_utility_tx: None,
            #[cfg(test)]
            next_plan_error: None,
            #[cfg(test)]
            before_grant_hook: None,
            #[cfg(test)]
            before_queue_control_plan_hook: None,
        };
        coordinator.last_resource_capacity_signature = coordinator.resource_capacity_signature();
        coordinator
    }

    /// Sample host memory and republish it, so `/api/status` and the queue
    /// plan report the numbers admission is actually spending.
    fn collect_host_memory(&mut self) {
        self.memory.collect_now();
        self.state
            .scheduled_work
            .publish_host_memory(self.memory.wire_snapshot());
    }

    /// Re-emit `Queued { position }` for every still-queued job whose place in
    /// line moved, using the registry order `GET /api/queue` reports.
    ///
    /// A client used to learn its position exactly once, in the first SSE
    /// event, and the queue then drained in silence. This needs no new event
    /// type: both terminal clients render `Queued` in place — indicatif's
    /// `set_message` and the TUI's `current_stage` — so a repeat reads as a
    /// live update rather than another line.
    ///
    /// Only an actual change emits. Reconcile runs on every registry
    /// notification and on a 10 ms ticker, so announcing unconditionally would
    /// be a firehose. The first observation seeds silently: the submit-time
    /// event has already told that client where it stands.
    fn announce_queue_positions(&mut self, entries: &[crate::job_registry::SchedulerQueueEntry]) {
        for entry in entries {
            if entry.state != crate::job_registry::JobLifecycle::Queued {
                continue;
            }
            let Some(pending) = self.pending.get_mut(&entry.id) else {
                continue;
            };
            if pending.announced_position == Some(entry.position) {
                continue;
            }
            let seeding = pending.announced_position.is_none();
            pending.announced_position = Some(entry.position);
            if seeding {
                continue;
            }
            if let Some(progress) = pending.job.progress_tx.as_ref() {
                let _ = progress.send(SseMessage::Progress(mold_core::SseProgressEvent::Queued {
                    position: entry.position,
                    id: entry.id.clone(),
                }));
            }
        }
    }

    fn install_cpu_utility_lane(
        &mut self,
        tx: std::sync::mpsc::SyncSender<crate::gpu_pool::GpuWorkerCommand>,
    ) {
        self.cpu_utility_tx = Some(tx);
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
            job.id = format!("runtime-generation-{}", self.synthetic_id);
        }
        let queue_rank = job.durable_queue_rank.unwrap_or(self.synthetic_id);
        self.synthetic_id = self
            .synthetic_id
            .saturating_add(1)
            .max(queue_rank.saturating_add(1));
        let id = job.id.clone();
        let shape_bucket = crate::gpu_pool::oom_shape_bucket_with_projection(
            &job.request,
            job.deferred_media.as_ref().map(|media| media.projection()),
        );
        if let Some(error) =
            crate::gpu_pool::model_unschedulable_message(&job.request.model, Some(&shape_bucket))
        {
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
        let prepared_inputs = job
            .batch_child
            .as_ref()
            .map(|child| child.prepared_inputs.clone());
        self.pending.insert(
            id,
            PendingGeneration {
                job,
                ready_at_ms: monotonic_ms(),
                queue_rank,
                bypass_count: 0,
                warm_wait_started_ms: None,
                preparation: PreparationState::Needed,
                prepared_inputs,
                retry_not_before_ms: None,
                preparation_retry_attempts: 0,
                preparation_refresh_observation: None,
                unschedulable_since_ms: None,
                unschedulable_reason: None,
                announced_position: None,
            },
        );
        self.mutate(immediate);
    }

    fn enqueue_owner_work(&mut self, mut submission: ScheduledOwnerWork, immediate: &mut bool) {
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
        if matches!(
            submission.work.kind(),
            WorkKind::PromptExpansion | WorkKind::StandaloneUpscale | WorkKind::PostUpscale
        ) {
            match self.freeze_utility_candidates(&submission.work, &submission.utility_plans) {
                Ok(plans) => submission.utility_plans = plans,
                Err(error) => {
                    submission.work.reject(format!(
                        "utility execution plan could not be frozen: {error}"
                    ));
                    return;
                }
            }
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
                preferred_ordinal: submission.preferred_ordinal,
                candidate_plans: submission.candidate_plans,
                queue_rank,
                ready_at_ms: monotonic_ms(),
                bypass_count: 0,
                warm_wait_started_ms: None,
                retry_not_before_ms: None,
                utility_plans: submission.utility_plans,
                work: submission.work,
            },
        );
        self.mutate(immediate);
    }

    fn freeze_utility_candidates(
        &self,
        work: &OwnerWork,
        supplied: &[UtilityExecutionPlan],
    ) -> Result<Vec<UtilityExecutionPlan>, String> {
        let placements = std::iter::once(UtilityPlacement::Cpu)
            .chain(
                self.state
                    .gpu_pool
                    .schedulable_workers()
                    .into_iter()
                    .map(|worker| UtilityPlacement::Device {
                        backend: worker.gpu.backend,
                        ordinal: worker.gpu.ordinal,
                    }),
            )
            .collect::<Vec<_>>();
        match work {
            #[cfg(feature = "expand")]
            OwnerWork::PromptExpansion(job) => {
                prompt_expansion_candidates(&self.state, &job.config, Some(&job.settings.model))
            }
            #[cfg(not(feature = "expand"))]
            OwnerWork::PromptExpansion(_) => {
                Err("local prompt expansion is unavailable in this build".to_string())
            }
            OwnerWork::StandaloneUpscale(_) => {
                #[cfg(feature = "expand")]
                let base = supplied.iter().find_map(|plan| match plan {
                    UtilityExecutionPlan::Upscale(plan) => Some(plan),
                    UtilityExecutionPlan::PromptExpansion(_) => None,
                });
                #[cfg(not(feature = "expand"))]
                let base = supplied.first().map(|plan| match plan {
                    UtilityExecutionPlan::Upscale(plan) => plan,
                });
                let base = base.ok_or_else(|| {
                    "standalone upscaling lacked a frozen artifact candidate".to_string()
                })?;
                Ok(upscale_utility_candidates(
                    &base.model_name,
                    &base.weights,
                    base.artifact_root.as_deref(),
                    placements,
                ))
            }
            OwnerWork::PostUpscale(_) => {
                #[cfg(feature = "expand")]
                let base = supplied.iter().find_map(|plan| match plan {
                    UtilityExecutionPlan::Upscale(plan) => Some(plan),
                    UtilityExecutionPlan::PromptExpansion(_) => None,
                });
                #[cfg(not(feature = "expand"))]
                let base = supplied.first().map(|plan| match plan {
                    UtilityExecutionPlan::Upscale(plan) => plan,
                });
                let base = base.ok_or_else(|| {
                    "post-generation upscaling lacked a frozen artifact candidate".to_string()
                })?;
                Ok(upscale_utility_candidates(
                    &base.model_name,
                    &base.weights,
                    base.artifact_root.as_deref(),
                    placements,
                ))
            }
            _ => Ok(supplied.to_vec()),
        }
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
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            let deferred_media = pending.job.deferred_media.clone();
            let queue_media_projection = pending
                .job
                .deferred_media
                .as_ref()
                .map(|media| media.projection().clone());
            let progress = pending.job.progress_tx.clone();
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            let context = crate::variant_dependencies::DependencyPreparationContext {
                h3_private_ingress_grant: pending.job.h3_private_ingress_grant.clone(),
                // Ref2VA admission derives its prepared shapes from the staged
                // media, so it needs the files before the frozen plan exists.
                // The view is payload-free and the job keeps owning the set,
                // its quota, and its staging directory across preparation.
                h3_resolved_references: pending
                    .job
                    .resolved_references
                    .as_ref()
                    .map(crate::reference_uploads::ResolvedReferenceSet::admission_view),
                // The parent's frozen identity is grafted on immediately
                // below. Default the remaining context so additive optional
                // preparation inputs cannot leave this H3-only literal
                // unbuildable.
                ..Default::default()
            };
            // A batch child arrives already holding the parent's frozen
            // identity. Handing it to preparation is what keeps ONE extraction
            // per parent request: without it this re-preparation would run the
            // extractor again for every sibling and then overwrite the
            // parent's value with its own.
            let frozen_identity = pending
                .prepared_inputs
                .as_ref()
                .and_then(|inputs| inputs.identity_embedding.clone());
            // Its advisory travels with it, so the pin this preparation mints
            // holds the pair a sibling reading it will report.
            let frozen_identity_warning = pending
                .prepared_inputs
                .as_ref()
                .and_then(|inputs| inputs.identity_warning.clone());
            #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
            let context = crate::variant_dependencies::DependencyPreparationContext {
                frozen_identity,
                frozen_identity_warning,
                queue_media_projection,
            };
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            let context = crate::variant_dependencies::DependencyPreparationContext {
                frozen_identity,
                frozen_identity_warning,
                queue_media_projection,
                ..context
            };
            let preparer = self.preparer.clone();
            let tx = self.preparation_tx.clone();
            let slots = self.preparation_slots.clone();
            self.preparation_tasks.spawn(async move {
                // The permit is taken inside the task so a queued preparation
                // waits here rather than in `Needed`, where the scheduler
                // would keep re-spawning it.
                let _slot = slots.acquire_owned().await;
                let request =
                    crate::queue_media_runtime::ZeroizingGenerateRequest::from_owned(request);
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                let mut request = request;
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                let h3_hydration = if context.h3_private_ingress_grant.is_some() {
                    match deferred_media.as_ref() {
                        Some(media) => match media.hydrate_into(&id, &mut request) {
                            Ok(lease) => Some(lease),
                            Err(error) => {
                                let event = PreparationEvent::Failed {
                                    work_id: id.clone(),
                                    error: error.to_string(),
                                };
                                let _ = tx.send(event);
                                return;
                            }
                        },
                        None => None,
                    }
                } else {
                    None
                };
                let event = match preparer
                    .prepare(state, id.clone(), request, progress, context)
                    .await
                {
                    Ok(prepared) => PreparationEvent::Ready {
                        work_id: id,
                        prepared: Box::new(prepared),
                    },
                    Err(error) => PreparationEvent::Failed { work_id: id, error },
                };
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                drop(h3_hydration);
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
                compose_prepared_generation(pending, *prepared);
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
                    hold_preparation_failure(&self.state, pending.job, error);
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
                let is_cpu_utility = device_id == CPU_UTILITY_DEVICE_ID;
                let was_starting =
                    !is_cpu_utility && self.state.gpu_pool.workers.is_starting(&device_id);
                if !is_cpu_utility
                    && !self
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
                if !is_cpu_utility {
                    self.state.device_registry.mark_available(&device_id);
                }
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
                self.state.device_registry.mark_unavailable(&device_id);
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
                } else {
                    self.replan_and_publish_with(PlanningPass::Admission);
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
            WorkerEvent::HostMemoryRecheck { fence, reply } => {
                let valid = self.leases.get(&fence.device_id).is_some_and(|lease| {
                    lease.work_id == fence.work_id
                        && lease.plan_version == fence.plan_version
                        && lease.owner_epoch == fence.owner_epoch
                        && lease.worker_generation == fence.worker_generation
                });
                let result = if valid {
                    self.memory
                        .collect_headroom_for_reserved_work(&fence.work_id)
                        .ok_or_else(|| {
                            "host-memory recheck lost the exact scheduler reservation".to_string()
                        })
                } else {
                    Err("host-memory recheck rejected a stale owner lease".to_string())
                };
                let _ = reply.send(result);
                self.mutate(immediate);
                self.replan_and_publish_with(PlanningPass::Admission);
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
                self.sample_active_lease_high_waters();
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
                    reject_owner_work_preserving_completed_generation(
                        grant.work,
                        "GPU owner returned work from a stale lifecycle epoch".to_string(),
                    );
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
                    self.collect_host_memory();
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
                if let (Some(lease), LeaseRejection::PlanInvalidated(error)) =
                    (rejected_lease.as_ref(), &reason)
                {
                    self.observe_estimate(
                        lease.estimate_key.clone(),
                        EstimateObservation {
                            total_ms: None,
                            phases: EstimatePhaseTimings::default(),
                            vram_high_water_bytes: lease.vram_high_water_bytes,
                            host_incremental_high_water_bytes: lease
                                .host_incremental_high_water_bytes,
                            outcome: EstimateOutcome::Invalidated,
                            fallback_reason: lease.fallback_reason.clone(),
                            invalidated_plan_reason: Some(error.to_string()),
                            observed_at_unix_s: unix_seconds(),
                        },
                    );
                }
                let LeaseGrant { work, retry, .. } = *grant;
                match work {
                    OwnerWork::Generation(job) => {
                        let (generation_job, prepared_inputs) =
                            generation_and_prepared_from_gpu_job(*job);
                        if matches!(&reason, LeaseRejection::FatalCuda) {
                            self.plan_invalidations.remove(&generation_job.id);
                            retain_generation(
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
                                let queue_rank = self.synthetic_id;
                                self.synthetic_id = self.synthetic_id.saturating_add(1);
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
                                        queue_rank,
                                        preparation: PreparationState::Ready,
                                        prepared_inputs: prepared_inputs.clone(),
                                        retry_not_before_ms: Some(
                                            monotonic_ms().saturating_add(backoff_ms),
                                        ),
                                        preparation_retry_attempts: 0,
                                        preparation_refresh_observation: None,
                                        unschedulable_since_ms: None,
                                        unschedulable_reason: None,
                                        announced_position: None,
                                    },
                                );
                            }
                        } else {
                            let queue_rank = self.synthetic_id;
                            self.synthetic_id = self.synthetic_id.saturating_add(1);
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
                                    queue_rank,
                                    preparation: PreparationState::Ready,
                                    prepared_inputs,
                                    retry_not_before_ms: None,
                                    preparation_retry_attempts: 0,
                                    preparation_refresh_observation: None,
                                    unschedulable_since_ms: None,
                                    unschedulable_reason: None,
                                    announced_position: None,
                                },
                            );
                        }
                    }
                    work => {
                        if matches!(&reason, LeaseRejection::FatalCuda) {
                            self.plan_invalidations.remove(&work_id);
                            reject_owner_work_preserving_completed_generation(
                                work,
                                "CUDA context is fatally poisoned; server restart required"
                                    .to_string(),
                            );
                        } else if matches!(&reason, LeaseRejection::PlanInvalidated(_))
                            && matches!(
                                work.kind(),
                                WorkKind::PromptExpansion
                                    | WorkKind::PostUpscale
                                    | WorkKind::StandaloneUpscale
                            )
                        {
                            let LeaseRejection::PlanInvalidated(error) = &reason else {
                                unreachable!("matched plan-invalidated utility rejection")
                            };
                            self.plan_invalidations.remove(&work_id);
                            reject_owner_work_preserving_completed_generation(
                                work,
                                format!(
                                    "utility execution plan was invalidated; refusing fallback: {error}"
                                ),
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
                                preferred_ordinal: None,
                                candidate_plans: Vec::new(),
                                queue_rank: self.synthetic_id,
                                ready_at_ms: preserved_ready_at_ms,
                                bypass_count: preserved_bypass_count,
                                warm_wait_started_ms: preserved_warm_wait_started_ms,
                                retry_not_before_ms: None,
                                utility_plans: Vec::new(),
                            });
                            let mut retry_not_before_ms = retry.retry_not_before_ms;
                            if let LeaseRejection::PlanInvalidated(error) = &reason {
                                let attempts =
                                    self.plan_invalidations.entry(work_id.clone()).or_default();
                                *attempts = attempts.saturating_add(1);
                                if *attempts >= MAX_PLAN_INVALIDATIONS {
                                    let attempts = *attempts;
                                    self.plan_invalidations.remove(&work_id);
                                    work.reject(format!(
                                        "execution plan was invalidated {attempts} consecutive times; \
                                         refusing to retry: {error}"
                                    ));
                                    self.mutate(immediate);
                                    return;
                                }
                                let backoff_ms = PLAN_INVALIDATION_BACKOFF_MS
                                    .saturating_mul(1_u64 << u32::from(*attempts - 1));
                                retry_not_before_ms =
                                    Some(monotonic_ms().saturating_add(backoff_ms));
                            }
                            self.pending_owner_work.insert(
                                work_id,
                                PendingOwnerWork {
                                    model_fingerprint: retry.model_fingerprint,
                                    estimated_vram_bytes: retry.estimated_vram_bytes,
                                    estimated_host_ram_bytes: retry.estimated_host_ram_bytes,
                                    hard_ordinal: retry.hard_ordinal,
                                    priority: retry.priority,
                                    preferred_ordinal: retry.preferred_ordinal,
                                    candidate_plans: retry.candidate_plans,
                                    queue_rank: retry.queue_rank,
                                    ready_at_ms: retry.ready_at_ms,
                                    bypass_count: retry.bypass_count,
                                    warm_wait_started_ms: retry.warm_wait_started_ms,
                                    retry_not_before_ms,
                                    utility_plans: retry.utility_plans,
                                    work,
                                },
                            );
                            self.synthetic_id = self.synthetic_id.saturating_add(1);
                        }
                    }
                }
                self.mutate(immediate);
                self.replan_and_publish_with(PlanningPass::Admission);
            }
            WorkerEvent::Completed {
                device_id,
                ordinal,
                owner_epoch,
                worker_generation,
                successful,
                phase_timings,
                completion,
            } => {
                self.sample_active_lease_high_waters();
                let valid = self.leases.get(&device_id).is_some_and(|lease| {
                    lease.owner_epoch == owner_epoch && lease.worker_generation == worker_generation
                });
                if valid {
                    let lease = self
                        .leases
                        .remove(&device_id)
                        .expect("validated lease must still exist");
                    self.memory.release(&lease.work_id);
                    self.collect_host_memory();
                    self.plan_invalidations.remove(&lease.work_id);
                    self.observe_estimate(
                        lease.estimate_key,
                        EstimateObservation {
                            total_ms: successful.then(|| {
                                lease
                                    .started_at
                                    .elapsed()
                                    .as_millis()
                                    .try_into()
                                    .unwrap_or(u64::MAX)
                            }),
                            phases: phase_timings,
                            vram_high_water_bytes: lease.vram_high_water_bytes,
                            host_incremental_high_water_bytes: lease
                                .host_incremental_high_water_bytes,
                            outcome: if successful {
                                EstimateOutcome::Success
                            } else {
                                EstimateOutcome::Failure
                            },
                            fallback_reason: lease.fallback_reason,
                            invalidated_plan_reason: None,
                            observed_at_unix_s: unix_seconds(),
                        },
                    );
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
                let publication = self.try_replan_and_publish_with(PlanningPass::Admission);
                if let Some(completion) = completion {
                    if !valid {
                        completion.fail(
                            "GPU owner completion did not match an authoritative scheduler lease"
                                .to_string(),
                        );
                    } else if let Err(error) = publication.as_ref() {
                        completion.fail(format!(
                            "internal scheduler error: could not publish the authoritative \
                             post-completion queue plan: {error}"
                        ));
                    } else {
                        completion.finish();
                    }
                }
                if let Err(error) = publication {
                    tracing::error!(
                        state_version = self.state_version,
                        %error,
                        "scheduler could not publish an observational queue plan"
                    );
                }
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
                    self.collect_host_memory();
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
                self.replan_and_publish_with(PlanningPass::Admission);
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
        self.estimates.observe(key.clone(), observation.clone());
        let normalized = key.normalized();
        if normalized != key {
            self.estimates.observe(normalized.clone(), observation);
            self.persist_estimate(&normalized);
        }
        self.persist_estimate(&key);
    }

    fn sample_active_lease_high_waters(&mut self) {
        let Some(snapshot) = self.state.resources.latest() else {
            return;
        };
        for (device_id, lease) in &mut self.leases {
            let sample =
                vram_sample_for_stable_device(&snapshot, &self.state.device_registry, device_id);
            lease.vram_high_water_bytes = max_optional(lease.vram_high_water_bytes, sample);
            // ResourceSnapshot does not yet expose process-attributable host
            // RAM. Keep this explicitly unavailable instead of relabeling a
            // completion sample as an execution peak.
            lease.host_incremental_high_water_bytes = None;
        }
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

        let registry_entries = self.state.job_registry.scheduler_snapshot();
        self.announce_queue_positions(&registry_entries);
        let queue_shape = registry_entries
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
                pending.job.should_cancel_for_observer_disconnect()
                    || (!id.starts_with("runtime-generation-")
                        && self.state.job_registry.scheduler_lifecycle(id).is_none())
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
                self.plan_invalidations.remove(&id);
                pending.work.cancel_queued();
                self.mutate(immediate);
            }
        }
    }

    fn resource_capacity_signature(&self) -> Vec<(String, u64)> {
        self.device_snapshots()
            .into_iter()
            .map(|device| (device.id.to_string(), device.available_vram_bytes))
            .collect()
    }

    /// Wake planning when sampled capacity changes, without turning the 1 Hz
    /// telemetry stream into an unconditional scheduler retry loop.
    fn reconcile_resource_capacity(&mut self, immediate: &mut bool) {
        let signature = self.resource_capacity_signature();
        if signature == self.last_resource_capacity_signature {
            return;
        }
        self.last_resource_capacity_signature = signature;
        if !self.pending.is_empty() || !self.pending_owner_work.is_empty() {
            self.mutate(immediate);
        }
    }

    fn device_snapshots(&self) -> Vec<DeviceSnapshot> {
        let resources = self.state.resources.latest();
        let canonical = self.state.device_registry.canonical_snapshot(
            &self.state.gpu_pool,
            resources.as_ref(),
            &self.state.job_registry,
        );
        let workers = self.state.gpu_pool.worker_snapshot();
        let mut snapshots = canonical
            .scheduler_devices
            .into_iter()
            .map(|device| {
                let worker = workers
                    .iter()
                    .find(|worker| worker_device_id(worker) == device.id);
                let ready = self.ready.get(&device.id);
                let active_lease = self.leases.get(&device.id);
                // The public registry spans the hand-off between coordinator
                // lease bookkeeping and the owner thread. During that window
                // a generation is already running even if the local lease or
                // in-flight counter is temporarily absent.
                let has_active_work = active_lease.is_some() || device.active_work;
                let measured_cache_bytes = worker
                    .map(|worker| {
                        worker
                            .model_cache
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .active_vram_bytes()
                    })
                    .unwrap_or(0);
                let reclaimable_cache_bytes = reclaimable_model_cache_bytes(
                    measured_cache_bytes,
                    device.sampled_mold_vram_bytes,
                );
                let mut warm = BTreeSet::new();
                if let Some(fingerprint) = worker.and_then(|worker| {
                    worker
                        .resident_execution_fingerprint
                        .read()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .clone()
                }) {
                    warm.insert(ExecutionFingerprint::new(fingerprint));
                }
                DeviceSnapshot {
                    id: DeviceId::new(device.id),
                    backend: match device.backend {
                        mold_core::GpuBackend::Cuda => Backend::Cuda,
                        mold_core::GpuBackend::Metal => Backend::Metal,
                    },
                    admin_state: match device.admin_state {
                        mold_core::DeviceAdminState::StartupExcluded => {
                            DeviceAdminState::StartupExcluded
                        }
                        mold_core::DeviceAdminState::Starting => DeviceAdminState::Disabled,
                        mold_core::DeviceAdminState::Enabled => DeviceAdminState::Enabled,
                        mold_core::DeviceAdminState::Draining => DeviceAdminState::Draining,
                        mold_core::DeviceAdminState::Disabled => DeviceAdminState::Disabled,
                    },
                    health: match device.health {
                        mold_core::DeviceHealth::Healthy => DeviceHealth::Healthy,
                        mold_core::DeviceHealth::Degraded => DeviceHealth::Degraded,
                        mold_core::DeviceHealth::Unavailable => DeviceHealth::Unavailable,
                        mold_core::DeviceHealth::Poisoned => DeviceHealth::Poisoned,
                    },
                    activity: if device.schedulable
                        && ready.is_some()
                        && !has_active_work
                        && worker.is_some_and(|worker| worker.in_flight.load(Ordering::SeqCst) == 0)
                    {
                        DeviceActivity::Idle
                    } else {
                        DeviceActivity::Busy
                    },
                    available_at_ms: active_lease.map(|lease| lease.estimated_finish_ms),
                    worker_generation: ready.map_or(0, |ready| ready.generation),
                    available_vram_bytes: schedulable_available_vram_bytes(
                        device.sampled_free_vram_bytes,
                        reclaimable_cache_bytes,
                        device.sampled_mold_vram_bytes,
                        has_active_work,
                        worker.map_or(0, |worker| worker.gpu.total_vram_bytes),
                    ),
                    warm_execution_fingerprints: warm,
                }
            })
            .collect::<Vec<_>>();
        if self.cpu_utility_tx.is_some() {
            let ready = self.ready.get(CPU_UTILITY_DEVICE_ID);
            let active_lease = self.leases.get(CPU_UTILITY_DEVICE_ID);
            snapshots.push(DeviceSnapshot {
                id: DeviceId::new(CPU_UTILITY_DEVICE_ID),
                backend: Backend::Cpu,
                admin_state: DeviceAdminState::Enabled,
                health: if self.unavailable.contains(CPU_UTILITY_DEVICE_ID) {
                    DeviceHealth::Unavailable
                } else {
                    DeviceHealth::Healthy
                },
                activity: if ready.is_some() && active_lease.is_none() {
                    DeviceActivity::Idle
                } else {
                    DeviceActivity::Busy
                },
                available_at_ms: active_lease.map(|lease| lease.estimated_finish_ms),
                worker_generation: ready.map_or(0, |ready| ready.generation),
                available_vram_bytes: u64::MAX,
                warm_execution_fingerprints: BTreeSet::new(),
            });
        }
        snapshots
    }

    /// Physical VRAM keyed by the same stable worker IDs carried by execution
    /// planning. Eligibility is retained by `InsufficientVram`; keeping the
    /// pool facts separate prevents an excluded sibling from lending capacity
    /// to a pinned request's terminal/transient verdict.
    fn total_vram_bytes_by_device_id(&self) -> BTreeMap<String, u64> {
        self.state
            .gpu_pool
            .workers
            .iter()
            .map(|worker| (worker_device_id(&worker), worker.gpu.total_vram_bytes))
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
                    backend: worker.gpu.backend,
                    compute_capability: worker.gpu.compute_capability,
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
        if pending
            .prepared_inputs
            .as_ref()
            .is_some_and(|prepared| !prepared.host_memory_retry_after_devices.is_empty())
        {
            return Err(GenerationPlanFailure::Transient(
                "waiting for active GPU work to release host memory".to_string(),
            ));
        }
        let config = self.state.config.try_read().map_err(|_| {
            GenerationPlanFailure::Transient(
                "configuration changed while resolving execution plan".to_string(),
            )
        })?;
        let offload_requested = matches!(
            mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        let hard_ordinal =
            generation_hard_ordinal(&self.state, &pending.job.id, &pending.job.request);
        let eligible_device_facts =
            constrained_generation_device_facts(device_facts, hard_ordinal, None);
        let resolved =
            crate::execution_plan::resolve_execution_plans_for_coordinator_with_projection(
                &config,
                &pending.job.request,
                &eligible_device_facts,
                offload_requested,
                pending.prepared_inputs.as_ref(),
                pending
                    .job
                    .deferred_media
                    .as_ref()
                    .map(|media| media.projection()),
            );
        let resolved = resolved.map(|plans| {
            let Some(expected) = pending
                .job
                .batch_child
                .as_ref()
                .map(|child| child.execution_equivalence_fingerprint.as_str())
            else {
                return plans;
            };
            plans
                .into_iter()
                .filter(|plan| plan.execution_equivalence_fingerprint.as_str() == expected)
                .collect()
        });
        #[cfg(test)]
        if matches!(
            resolved,
            Err(crate::execution_plan::ExecutionPlanError::MissingArtifacts { .. })
        ) {
            // Coordinator unit tests deliberately use synthetic requests with
            // no filesystem model. Keep their transport/fencing focus without
            // weakening production admission.
            let estimate = crate::queue::estimate_model_vram(&pending.job.request.model);
            let plans = eligible_device_facts
                .iter()
                .filter(|device| device.available_vram_bytes >= estimate)
                .cloned()
                .map(|device| {
                    let model_family = crate::model_manager::family_for_model_sync(
                        &pending.job.request.model,
                        &config,
                    )
                    .unwrap_or_else(|| pending.job.request.model.clone());
                    let model_fingerprint = pending.job.request.model.clone();
                    let components = BTreeMap::new();
                    let engine_config = mold_inference::FrozenEngineConfig::resolve(
                        &pending.job.request.model,
                        &config,
                    );
                    let determinism_class =
                        crate::execution_plan::DeterminismClass::CpuSeededCrossBackend;
                    let environment = crate::execution_plan::execution_environment_descriptor(
                        &device,
                        &pending.job.request.model,
                        &model_family,
                        &model_fingerprint,
                        &components,
                        &[],
                        &engine_config,
                        crate::execution_plan::AttentionBackend::Math,
                        mold_inference::LoadStrategy::Eager,
                        crate::execution_plan::OffloadMode::None,
                        pending.job.request.resolved_output_format(),
                        determinism_class,
                        false,
                        &BTreeMap::new(),
                    )
                    .expect(
                        "synthetic coordinator-test descriptor classifies every frozen \
                         engine-shaping variable (guarded by \
                         every_engine_shaping_variable_has_a_semantic_class)",
                    );
                    let equivalence = environment.fingerprint();
                    crate::execution_plan::ResolvedExecutionPlan {
                        device_id: device.id,
                        device_ordinal: device.ordinal,
                        device_backend: device.backend,
                        model_family,
                        model_fingerprint,
                        effective_placement: crate::execution_plan::EffectivePlacement {
                            components: BTreeMap::new(),
                        },
                        components,
                        engine_paths: mold_core::ModelPaths {
                            low_noise_transformer: None,
                            low_noise_distilled_lora: None,
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
                        engine_config: engine_config.clone(),
                        admission_paths: mold_core::ModelPaths {
                            low_noise_transformer: None,
                            low_noise_distilled_lora: None,
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
                        admission_engine_config: engine_config,
                        effective_loras: vec![],
                        attention_backend: crate::execution_plan::AttentionBackend::Math,
                        engine_load_strategy: mold_inference::LoadStrategy::Eager,
                        offload_mode: crate::execution_plan::OffloadMode::None,
                        predicted_vram_peak_bytes: estimate,
                        admitted_available_vram_bytes: device.available_vram_bytes,
                        learned_vram_envelope_bytes: 0,
                        predicted_host_increment_bytes: MIN_TRANSIENT_HOST_RAM,
                        determinism_class,
                        execution_environment: environment,
                        execution_equivalence_fingerprint: equivalence,
                        execution_fingerprint: pending.job.request.model.clone(),
                    }
                })
                .collect::<Vec<_>>();
            if plans.is_empty() {
                let error = crate::execution_plan::insufficient_vram_error(
                    &eligible_device_facts
                        .iter()
                        .map(|device| crate::execution_plan::DeviceInfeasibility {
                            device_id: device.id.clone(),
                            predicted_peak_bytes: estimate,
                            available_bytes: device.available_vram_bytes,
                            advice: None,
                        })
                        .collect::<Vec<_>>(),
                );
                return Err(classify_generation_plan_failure(
                    error,
                    &self.total_vram_bytes_by_device_id(),
                ));
            }
            return Ok(plans);
        }
        resolved.map_err(|error| {
            classify_generation_plan_failure(error, &self.total_vram_bytes_by_device_id())
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
        let busy_worker_devices = crate::host_reclaim::busy_worker_device_ids(&self.state);
        let mut refresh = BTreeSet::new();
        for (id, pending) in &mut self.pending {
            if stale.contains(id) || pending.preparation != PreparationState::Ready {
                continue;
            }
            let Some(prepared) = pending.prepared_inputs.as_ref() else {
                pending.preparation_refresh_observation = None;
                continue;
            };
            if !prepared.host_memory_retry_after_devices.is_empty() {
                pending.preparation_refresh_observation = None;
                if prepared
                    .host_memory_retry_after_devices
                    .is_disjoint(&busy_worker_devices)
                {
                    refresh.insert(id.clone());
                }
                continue;
            }
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

    fn owner_plans(
        &self,
        pending: &PendingOwnerWork,
    ) -> Result<
        Vec<crate::execution_plan::ResolvedExecutionPlan>,
        crate::execution_plan::ExecutionPlanError,
    > {
        let Some((config, request)) = pending.work.chain_plan_inputs() else {
            return Ok(pending.candidate_plans.clone());
        };
        let offload_requested = matches!(
            std::env::var("MOLD_OFFLOAD").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        let frozen_chain = matches!(
            &pending.work,
            OwnerWork::ChainStage(job) if job.expected_model_fingerprint.is_some()
        );
        let plans = crate::execution_plan::resolve_execution_plans_for_coordinator(
            config,
            request,
            &self.device_facts(),
            offload_requested,
            None,
        )
        .map_err(|error| {
            if frozen_chain
                && matches!(
                    error,
                    crate::execution_plan::ExecutionPlanError::MissingArtifacts { .. }
                )
            {
                crate::execution_plan::ExecutionPlanError::PlanInvalidated(format!(
                    "frozen chain artifacts are no longer resolvable: {error}"
                ))
            } else {
                error
            }
        })?;
        if let OwnerWork::ChainStage(work) = &pending.work {
            if let Some(expected) = &work.expected_model_fingerprint {
                let frozen = work.config.models.get(&work.model).ok_or_else(|| {
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "frozen chain model config is missing".to_string(),
                    )
                })?;
                let current = crate::execution_plan::frozen_model_fingerprint(&work.model, frozen)?;
                if &current != expected {
                    return Err(crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "frozen chain inputs no longer match their durable fingerprint".to_string(),
                    ));
                }
            }
        }
        Ok(plans)
    }

    fn owner_plan_cache_and_settle_errors(
        &mut self,
    ) -> BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>> {
        let now_ms = monotonic_ms();
        let resolutions = self
            .pending_owner_work
            .iter()
            .filter(|(_, pending)| {
                pending
                    .retry_not_before_ms
                    .is_none_or(|deadline| deadline <= now_ms)
            })
            .map(|(id, pending)| (id.clone(), self.owner_plans(pending)))
            .collect::<Vec<_>>();
        let mut cache = BTreeMap::new();
        let mut changed = false;
        for (id, resolution) in resolutions {
            let error = match resolution {
                Ok(plans) => {
                    cache.insert(id, plans);
                    continue;
                }
                Err(error) => error,
            };
            if matches!(
                error,
                crate::execution_plan::ExecutionPlanError::InsufficientVram { .. }
            ) {
                continue;
            }
            if let crate::execution_plan::ExecutionPlanError::PlanInvalidated(_) = &error {
                let attempts = self.plan_invalidations.entry(id.clone()).or_default();
                *attempts = attempts.saturating_add(1);
                if *attempts < MAX_PLAN_INVALIDATIONS {
                    let backoff_ms = PLAN_INVALIDATION_BACKOFF_MS
                        .saturating_mul(1_u64 << u32::from(*attempts - 1));
                    if let Some(pending) = self.pending_owner_work.get_mut(&id) {
                        pending.retry_not_before_ms =
                            Some(monotonic_ms().saturating_add(backoff_ms));
                    }
                    changed = true;
                    continue;
                }
            }
            if let Some(pending) = self.pending_owner_work.remove(&id) {
                self.plan_invalidations.remove(&id);
                pending.work.reject(format!(
                    "execution planning failed before worker acceptance: {error}"
                ));
                changed = true;
            }
        }
        if changed {
            self.state_version = self.state_version.saturating_add(1);
        }
        cache
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

    /// Settle an unclassified empty plan set that cannot change from here.
    ///
    /// A ready job whose resolver returns an empty plan set contributes zero
    /// candidates, and the planner has nothing left to say about it: with no
    /// candidate to compare against a device, `classify_no_candidate` reports
    /// the untyped `NoSchedulableDevice`. Typed transient failures do not enter
    /// this settlement path; time cannot turn resource pressure into a terminal
    /// verdict.
    ///
    /// Keep retrying while anything is leased or preparing, and settle only the
    /// unclassified empty result once the wait has been idle for
    /// `UNSCHEDULABLE_IDLE_GRACE_MS`. This deliberately does NOT bound a job
    /// queued behind running work — that job is waiting for something real.
    fn settle_unschedulable_generations(&mut self) -> bool {
        // "Idle" has to mean every way mold itself can be holding the resource,
        // not just the lease table: a preparation can be staging weights, and a
        // legacy chain claims a worker without taking a coordinator lease.
        // The job registry spans coordinator/worker hand-offs and is the
        // authoritative final check for an accepted running generation.
        // Over-counting busy only makes the job wait longer, which is the safe
        // direction.
        let idle = self.leases.is_empty()
            && !self.state.job_registry.has_running()
            && !self
                .pending
                .values()
                .any(|pending| pending.preparation != PreparationState::Ready)
            && !self.state.gpu_pool.workers.iter().any(|worker| {
                worker.in_flight.load(Ordering::SeqCst) > 0
                    || worker.legacy_pending.load(Ordering::SeqCst) > 0
            });
        let now_ms = monotonic_ms();
        let device_facts = self.device_facts();
        let observations = self
            .pending
            .iter()
            .map(|(id, pending)| {
                if pending.preparation != PreparationState::Ready {
                    return (id.clone(), None);
                }
                let failure = match self.generation_plans_with_device_facts(pending, &device_facts)
                {
                    // A plan resolved: this job is schedulable and any earlier
                    // idle wait is void.
                    Ok(plans) if !plans.is_empty() => None,
                    Ok(_) => Some(String::new()),
                    // Terminal failures are rejected by their own pass, which
                    // already names the error.
                    Err(GenerationPlanFailure::Terminal(_)) => None,
                    // Resource pressure and preparation refreshes are retry
                    // states, not evidence that durable work became invalid.
                    // Their duration cannot turn them into terminal failures.
                    Err(GenerationPlanFailure::Transient(_))
                    | Err(GenerationPlanFailure::StalePreparation(_)) => None,
                };
                (id.clone(), failure)
            })
            .collect::<Vec<_>>();
        let grace_ms = self.unschedulable_idle_grace_ms;
        let mut refusals = Vec::new();
        for (id, failure) in observations {
            let Some(pending) = self.pending.get_mut(&id) else {
                continue;
            };
            let Some(reason) = failure else {
                pending.unschedulable_since_ms = None;
                pending.unschedulable_reason = None;
                continue;
            };
            if !reason.is_empty() {
                pending.unschedulable_reason = Some(reason);
            }
            if !idle {
                pending.unschedulable_since_ms = None;
                continue;
            }
            let since_ms = match pending.unschedulable_since_ms {
                Some(since_ms) => since_ms,
                None => {
                    pending.unschedulable_since_ms = Some(now_ms);
                    tracing::warn!(
                        job_id = %id,
                        model = %pending.job.request.model,
                        reason = pending.unschedulable_reason.as_deref().unwrap_or_default(),
                        "queued generation resolves no execution plan on an idle scheduler"
                    );
                    now_ms
                }
            };
            if now_ms.saturating_sub(since_ms) >= grace_ms {
                refusals.push(id);
            }
        }
        if refusals.is_empty() {
            return false;
        }
        for id in refusals {
            let Some(pending) = self.pending.remove(&id) else {
                continue;
            };
            self.plan_invalidations.remove(&id);
            let message = unschedulable_rejection_message(
                &pending.job.request.model,
                pending.unschedulable_reason.as_deref(),
            );
            tracing::warn!(
                job_id = %id,
                %message,
                "refusing a queued generation that stayed unschedulable on an idle scheduler"
            );
            reject_generation(&self.state, pending.job, message);
        }
        self.state_version = self.state_version.saturating_add(1);
        true
    }

    fn owner_work_snapshot(
        &self,
        owner: OwnerWorkSchedulingView<'_>,
        utility_plans: &[UtilityExecutionPlan],
        device_snapshots: &[DeviceSnapshot],
    ) -> WorkSnapshot {
        let authoritative_available_vram = device_snapshots
            .iter()
            .cloned()
            .map(|device| (device.id, device.available_vram_bytes))
            .collect::<BTreeMap<_, _>>();
        let estimate_candidate = |device_id: DeviceId,
                                  worker: Option<&GpuWorker>,
                                  execution_fingerprint: &str,
                                  exact_vram_bytes: u64,
                                  exact_host_bytes: u64,
                                  exact_memory: bool| {
            let key = EstimateKey {
                device_class: worker
                    .map(device_class)
                    .unwrap_or_else(|| CPU_UTILITY_DEVICE_ID.to_string()),
                model_family: owner.model_fingerprint.to_string(),
                model_fingerprint: owner.model_fingerprint.to_string(),
                work_kind: snake_debug(owner.kind),
                shape_bucket: owner.shape_bucket.into(),
                execution_fingerprint: execution_fingerprint.to_string(),
            };
            let placement_backend =
                worker.map_or(Backend::Cpu, |worker| match worker.gpu.backend {
                    mold_core::GpuBackend::Cuda => Backend::Cuda,
                    mold_core::GpuBackend::Metal => Backend::Metal,
                });
            let static_timing =
                mold_scheduler::static_timing_for_placement(owner.kind, placement_backend);
            let static_estimate = StaticEstimate {
                total_ms: static_timing
                    .cold_setup_ms
                    .saturating_add(static_timing.predicted_run_ms),
                cold_setup_ms: static_timing.cold_setup_ms,
                warm_setup_ms: static_timing.warm_setup_ms,
                predicted_run_ms: static_timing.predicted_run_ms,
                vram_bytes: exact_vram_bytes,
                host_bytes: exact_host_bytes,
            };
            let estimate = self.estimates.estimate(&key, static_estimate);
            let (cold_setup_ms, warm_setup_ms, predicted_run_ms) =
                timing_with_static_floors(estimate, static_estimate);
            let (planned_vram_bytes, planned_host_bytes) = if exact_memory {
                (exact_vram_bytes, exact_host_bytes)
            } else {
                (estimate.vram_bytes, estimate.host_bytes)
            };
            let planned_vram_bytes =
                planned_vram_bytes.max(failure_only_vram_floor(&self.estimates, &key));
            let (planned_vram_bytes, planned_host_bytes) =
                planned_memory_bytes(owner.kind, planned_vram_bytes, planned_host_bytes);
            CandidatePlacement::new(
                device_id.clone(),
                ExecutionFingerprint::new(execution_fingerprint),
                planned_host_bytes,
            )
            .with_vram(planned_vram_bytes)
            .with_timing(
                cold_setup_ms,
                if matches!(
                    owner.kind,
                    mold_scheduler::WorkKind::PromptExpansion
                        | mold_scheduler::WorkKind::PostUpscale
                        | mold_scheduler::WorkKind::StandaloneUpscale
                ) {
                    cold_setup_ms
                } else {
                    warm_setup_ms
                },
                predicted_run_ms,
            )
            .with_device_available_vram(
                authoritative_available_vram
                    .get(&device_id)
                    .copied()
                    .unwrap_or(0),
            )
            .with_affinity_penalty(worker.map_or(0, |worker| {
                owner
                    .preferred_ordinal
                    .map_or(0, |preferred| u8::from(preferred != worker.gpu.ordinal))
            }))
        };
        let candidates = if !utility_plans.is_empty() {
            utility_plans
                .iter()
                .filter_map(|plan| match plan.placement() {
                    UtilityPlacement::Cpu => {
                        // `planned_memory_bytes` zeroes releasing work, while
                        // `utility_plan_for_lease` matches a lease back to its
                        // plan on those same byte counts. Nothing submits an
                        // unload with utility plans today; if that changes, the
                        // lease would be silently dropped every turn.
                        debug_assert!(
                            !owner.kind.releases_resources(),
                            "releasing work is priced at zero and cannot carry a utility plan"
                        );
                        let device_id = DeviceId::new(CPU_UTILITY_DEVICE_ID);
                        authoritative_available_vram
                            .contains_key(&device_id)
                            .then(|| {
                                estimate_candidate(
                                    device_id,
                                    None,
                                    plan.execution_fingerprint(),
                                    plan.predicted_vram_bytes(),
                                    plan.predicted_host_ram_bytes(),
                                    true,
                                )
                            })
                    }
                    UtilityPlacement::Device { backend, ordinal } => {
                        let worker = self.state.gpu_pool.workers.iter().find(|worker| {
                            worker.gpu.backend == backend && worker.gpu.ordinal == ordinal
                        })?;
                        let device_id = DeviceId::new(worker_device_id(worker.as_ref()));
                        authoritative_available_vram
                            .contains_key(&device_id)
                            .then(|| {
                                estimate_candidate(
                                    device_id,
                                    Some(worker.as_ref()),
                                    plan.execution_fingerprint(),
                                    plan.predicted_vram_bytes(),
                                    plan.predicted_host_ram_bytes(),
                                    true,
                                )
                            })
                    }
                })
                .collect::<Vec<_>>()
        } else if !owner.resolved_plans.is_empty() || owner.requires_exact_plan {
            owner
                .resolved_plans
                .iter()
                .filter_map(|plan| {
                    self.state
                        .gpu_pool
                        .worker_by_ordinal(plan.device_ordinal)
                        .map(|worker| {
                            estimate_candidate(
                                DeviceId::new(plan.device_id.clone()),
                                Some(worker.as_ref()),
                                &plan.execution_fingerprint,
                                plan.predicted_vram_peak_bytes,
                                plan.predicted_host_increment_bytes,
                                true,
                            )
                        })
                })
                .collect::<Vec<_>>()
        } else {
            self.state
                .gpu_pool
                .workers
                .iter()
                .map(|worker| {
                    let device_id = DeviceId::new(worker_device_id(worker.as_ref()));
                    estimate_candidate(
                        device_id,
                        Some(worker.as_ref()),
                        owner.model_fingerprint,
                        owner.estimated_vram_bytes,
                        owner.estimated_host_ram_bytes,
                        false,
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut work = WorkSnapshot::new(WorkId::new(owner.id), owner.queue_rank, candidates)
            .with_priority(owner.priority)
            .with_bypass_count(owner.bypass_count)
            .with_ready_at(owner.ready_at_ms);
        if let Some(started) = owner.warm_wait_started_ms {
            work = work.with_warm_wait_started_at(started);
        }
        work.kind = owner.kind;
        if owner.kind == mold_scheduler::WorkKind::ChainStage {
            if let Some((parent_id, stage_idx)) = chain_work_identity(owner.id) {
                work.parent_id = mold_scheduler::ParentId::new(parent_id);
                work.chain_stage = Some(stage_idx);
            }
        }
        if let Some(ordinal) = owner.hard_ordinal {
            if let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ordinal) {
                work = work.with_hard_device(DeviceId::new(worker_device_id(&worker)));
            } else {
                work = work.with_hard_device(DeviceId::new(format!("unavailable:gpu:{ordinal}")));
            }
        }
        work
    }

    fn work_snapshots(
        &self,
        generation_plans: &BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>>,
        owner_plan_cache: &BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>>,
        device_snapshots: &[DeviceSnapshot],
    ) -> Vec<WorkSnapshot> {
        // Generation reordering may permute generation jobs, but it must not
        // move the entire legacy queue into a separate 0..N rank namespace
        // ahead of scheduler-owned work. Reassign only the global rank slots
        // originally occupied by generations; owner work keeps its position
        // in the shared monotonic sequence.
        let queue_order = self.state.job_registry.queued_ids_in_order();
        let queue_patch_blocked = self.state.job_registry.queue_patch_blocked_ids();
        let ranks = reordered_generation_ranks(
            self.pending
                .iter()
                .map(|(id, pending)| (id.clone(), pending.queue_rank)),
            &queue_order,
        );
        let now_ms = monotonic_ms();
        let mut snapshots: Vec<WorkSnapshot> = self
            .pending
            .iter()
            .filter(|(id, _)| !queue_patch_blocked.contains(*id))
            .map(|(id, pending)| {
                let ready = pending.preparation == PreparationState::Ready
                    && pending
                        .retry_not_before_ms
                        .is_none_or(|deadline| deadline <= now_ms);
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
                                    &self.state,
                                    &worker,
                                    &pending.job.request,
                                    pending
                                        .job
                                        .deferred_media
                                        .as_ref()
                                        .map(|media| media.projection()),
                                    &plan.execution_fingerprint,
                                )
                            })
                            .unwrap_or_else(|| EstimateKey {
                                device_class: plan.device_id.clone(),
                                model_family: plan.model_family.clone(),
                                model_fingerprint: pending.job.request.model.clone(),
                                work_kind: "generation".into(),
                                shape_bucket: generation_shape_bucket_with_projection(
                                    &pending.job.request,
                                    pending
                                        .job
                                        .deferred_media
                                        .as_ref()
                                        .map(|media| media.projection()),
                                ),
                                execution_fingerprint: plan.execution_fingerprint.clone(),
                            });
                        let static_estimate = static_generation_estimate_with_projection(
                            &pending.job.request,
                            plan.predicted_vram_peak_bytes,
                            plan.admission_host_demand_bytes(),
                            pending
                                .job
                                .deferred_media
                                .as_ref()
                                .map(|media| media.projection()),
                        );
                        let estimate = self.estimates.estimate(&key, static_estimate);
                        let (cold_setup_ms, warm_setup_ms, predicted_run_ms) =
                            timing_with_static_floors(estimate, static_estimate);
                        let candidate = CandidatePlacement::new(
                            DeviceId::new(plan.device_id),
                            ExecutionFingerprint::new(plan.execution_fingerprint),
                            estimate.host_bytes,
                        )
                        .with_execution_equivalence(plan.execution_equivalence_fingerprint)
                        .with_vram(estimate.vram_bytes)
                        .with_timing(cold_setup_ms, warm_setup_ms, predicted_run_ms)
                        .with_device_available_vram(plan.admitted_available_vram_bytes);
                        if generation_uses_frozen_device_capacity(
                            plan.device_backend,
                            &plan.model_family,
                        ) {
                            candidate.with_frozen_device_capacity()
                        } else {
                            candidate
                        }
                    })
                    .collect::<Vec<_>>();
                let mut work = WorkSnapshot::new(
                    WorkId::new(id.clone()),
                    ranks.get(id).copied().unwrap_or(pending.queue_rank),
                    candidates,
                )
                .with_bypass_count(pending.bypass_count)
                .with_ready_at(pending.ready_at_ms)
                .with_ready(ready)
                .with_batch_partition(
                    pending
                        .job
                        .request
                        .batch_index
                        .zip(pending.job.request.batch_count)
                        .map(|(index, count)| mold_scheduler::PlannedBatchPartition {
                            index,
                            count,
                            size: pending.job.request.batch_size,
                        }),
                );
                if pending.job.batch_child.is_some() {
                    work.kind = mold_scheduler::WorkKind::BatchChild;
                } else if pending
                    .job
                    .request
                    .batch_count
                    .is_some_and(|count| count > 1)
                {
                    work.kind = mold_scheduler::WorkKind::PreparedSibling;
                }
                if let Some(batch_id) = pending.job.request.batch_id.as_deref() {
                    work.parent_id = mold_scheduler::ParentId::new(batch_id);
                }
                if let Some(started) = pending.warm_wait_started_ms {
                    work = work.with_warm_wait_started_at(started);
                }
                let explicit = generation_hard_ordinal(&self.state, id, &pending.job.request);
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
        snapshots.extend(self.pending_owner_work.iter().filter_map(|(id, pending)| {
            if pending
                .retry_not_before_ms
                .is_some_and(|deadline| deadline > now_ms)
            {
                return None;
            }
            let shape_bucket = pending.work.scheduling_shape_bucket();
            let resolved_plans = owner_plan_cache.get(id).map_or(&[][..], Vec::as_slice);
            Some(self.owner_work_snapshot(
                OwnerWorkSchedulingView {
                    id,
                    model_fingerprint: &pending.model_fingerprint,
                    estimated_vram_bytes: pending.estimated_vram_bytes,
                    estimated_host_ram_bytes: pending.estimated_host_ram_bytes,
                    hard_ordinal: pending.hard_ordinal,
                    priority: pending.priority,
                    queue_rank: pending.queue_rank,
                    ready_at_ms: pending.ready_at_ms,
                    bypass_count: pending.bypass_count,
                    warm_wait_started_ms: pending.warm_wait_started_ms,
                    kind: pending.work.kind(),
                    shape_bucket: &shape_bucket,
                    preferred_ordinal: pending.preferred_ordinal,
                    resolved_plans,
                    requires_exact_plan: pending.work.chain_plan_inputs().is_some(),
                },
                &pending.utility_plans,
                device_snapshots,
            ))
        }));
        snapshots
    }

    fn utility_plan_for_lease(
        &self,
        pending: &PendingOwnerWork,
        device_id: &str,
        placement: &CandidatePlacement,
    ) -> Option<UtilityExecutionPlan> {
        pending.utility_plans.iter().find_map(|plan| {
            let planned_device_id = match plan.placement() {
                UtilityPlacement::Cpu => CPU_UTILITY_DEVICE_ID.to_string(),
                UtilityPlacement::Device { backend, ordinal } => self
                    .state
                    .gpu_pool
                    .workers
                    .iter()
                    .find(|worker| worker.gpu.backend == backend && worker.gpu.ordinal == ordinal)
                    .map(|worker| worker_device_id(&worker))?,
            };
            (planned_device_id == device_id
                && plan.execution_fingerprint() == placement.execution_fingerprint.as_str()
                && plan.predicted_vram_bytes() == placement.predicted_vram_bytes
                && plan.predicted_host_ram_bytes() == placement.incremental_host_ram_bytes)
                .then(|| plan.clone())
        })
    }

    fn planner_snapshot(
        &self,
        owner_plan_cache: &BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>>,
    ) -> (
        PlannerSnapshot,
        BTreeMap<String, Vec<crate::execution_plan::ResolvedExecutionPlan>>,
    ) {
        let devices = self.device_snapshots();
        let device_facts = self.device_facts_from_snapshots(&devices);
        let generation_plans = self.generation_plan_catalog(&device_facts);
        let work = self.work_snapshots(&generation_plans, owner_plan_cache, &devices);
        (
            PlannerSnapshot {
                state_version: self.state_version,
                next_plan_version: self.plan_version.saturating_add(1),
                now_ms: monotonic_ms(),
                next_replan_at_ms: self.dirty.deadline().map(monotonic_deadline_ms),
                queue_paused: self.state.queue_pause.is_paused(),
                host_memory: self.memory.snapshot(),
                devices,
                work,
            },
            generation_plans,
        )
    }

    #[cfg(test)]
    fn placement_preview(
        &self,
        request: &mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
    ) -> mold_core::GenerationPlacementPreview {
        self.placement_preview_cancellable(request, copies, prepared_inputs, &|| false)
    }

    fn placement_preview_cancellable(
        &self,
        request: &mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
        cancelled: &dyn Fn() -> bool,
    ) -> mold_core::GenerationPlacementPreview {
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        if !(1..=64).contains(&copies) {
            return mold_core::GenerationPlacementPreview {
                version: 1,
                authoritative: true,
                state_version: self.state_version,
                plan_version: self.plan_version,
                outcome: "infeasible".to_string(),
                reason: Some("copies must be between 1 and 64".to_string()),
                candidate: None,
                stage_candidates: Vec::new(),
                pending_downloads: Vec::new(),
                missing_components: Vec::new(),
            };
        }
        let has_local_expansion = request.expand == Some(true)
            && match self.state.config.try_read() {
                Ok(config) => config.expand.clone().with_env_overrides().is_local(),
                // Failure to inspect a moving configuration can never upgrade
                // a utility preview to authoritative.
                Err(_) => true,
            };
        let has_post_upscale = request
            .upscale_model
            .as_deref()
            .is_some_and(|model| !model.trim().is_empty());
        if has_local_expansion || has_post_upscale {
            return mold_core::GenerationPlacementPreview {
                version: 1,
                authoritative: false,
                state_version: self.state_version,
                plan_version: self.plan_version,
                outcome: "unsupported".to_string(),
                reason: Some(
                    "exact utility CPU/GPU placement plans are not available on this server"
                        .to_string(),
                ),
                candidate: None,
                stage_candidates: Vec::new(),
                pending_downloads: Vec::new(),
                missing_components: Vec::new(),
            };
        }
        self.placement_preview_dag_cancellable(request, copies, prepared_inputs, cancelled)
    }

    fn cancelled_placement_preview(&self) -> mold_core::GenerationPlacementPreview {
        mold_core::GenerationPlacementPreview {
            version: 1,
            authoritative: false,
            state_version: self.state_version,
            plan_version: self.plan_version,
            outcome: "temporarily_unavailable".to_string(),
            reason: Some("placement preview cancelled".to_string()),
            candidate: None,
            stage_candidates: Vec::new(),
            pending_downloads: Vec::new(),
            missing_components: Vec::new(),
        }
    }

    /// Non-mutating scheduler projection for the complete ordinary-generation
    /// DAG. Utility stages exercise this path in tests, but the public preview
    /// remains non-authoritative until their CPU/GPU execution plans and host
    /// reservations are frozen rather than dynamically selected at runtime.
    #[cfg(test)]
    fn placement_preview_dag(
        &self,
        request: &mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
    ) -> mold_core::GenerationPlacementPreview {
        self.placement_preview_dag_cancellable(request, copies, prepared_inputs, &|| false)
    }

    fn placement_preview_dag_cancellable(
        &self,
        request: &mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
        cancelled: &dyn Fn() -> bool,
    ) -> mold_core::GenerationPlacementPreview {
        self.placement_preview_dag_for_device_cancellable(
            request,
            copies,
            prepared_inputs,
            None,
            cancelled,
        )
    }

    /// Return one exact singleton timing profile for every currently eligible
    /// generation device without expanding the parent into preview copies.
    ///
    /// The public placement preview deliberately caps visualization at 64
    /// copies. Adaptive batch admission instead needs one arithmetic lane per
    /// device for arbitrary parent sizes, so it profiles each device against
    /// one immutable coordinator snapshot and leaves child-count scaling to
    /// `BatchPartitionPlanner`.
    #[cfg(test)]
    fn batch_device_profiles(
        &self,
        request: &mold_core::GenerateRequest,
        parent_size: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
    ) -> anyhow::Result<Vec<mold_core::GenerationPlacementCandidate>> {
        self.batch_device_profiles_cancellable(request, parent_size, prepared_inputs, &|| false)
    }

    fn batch_device_profiles_cancellable(
        &self,
        request: &mold_core::GenerateRequest,
        parent_size: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
        cancelled: &dyn Fn() -> bool,
    ) -> anyhow::Result<Vec<mold_core::GenerationPlacementCandidate>> {
        anyhow::ensure!(parent_size > 0, "batch parent size must be positive");
        anyhow::ensure!(!cancelled(), "placement preview cancelled");
        let device_ids = self
            .device_snapshots()
            .into_iter()
            .map(|device| device.id.to_string())
            .collect::<Vec<_>>();
        let mut profiles = Vec::with_capacity(device_ids.len());
        let mut rejection = None;
        for device_id in device_ids {
            anyhow::ensure!(!cancelled(), "placement preview cancelled");
            let preview = self.placement_preview_dag_for_device_cancellable(
                request,
                1,
                prepared_inputs,
                Some(&device_id),
                cancelled,
            );
            if preview.authoritative && preview.outcome == "planned" {
                let generation_stage = preview
                    .stage_candidates
                    .iter()
                    .filter(|stage| stage.copy_index.is_some())
                    .map(|stage| stage.stage_index)
                    .min()
                    .ok_or_else(|| {
                        anyhow::anyhow!("batch device preview has no generation stage")
                    })?;
                if let Some(candidate) = preview
                    .stage_candidates
                    .into_iter()
                    .find(|stage| {
                        stage.stage_index == generation_stage
                            && stage.copy_index == Some(0)
                            && stage.candidate.device_id == device_id
                    })
                    .map(|stage| stage.candidate)
                {
                    profiles.push(candidate);
                }
            } else {
                rejection.get_or_insert_with(|| {
                    preview.reason.unwrap_or_else(|| preview.outcome.clone())
                });
            }
        }
        anyhow::ensure!(
            !profiles.is_empty(),
            "authoritative scheduler could not profile batch placement: {}",
            rejection.unwrap_or_else(|| "no eligible device".to_string())
        );
        Ok(profiles)
    }

    #[cfg(test)]
    fn placement_preview_dag_for_device(
        &self,
        request: &mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
        required_device_id: Option<&str>,
    ) -> mold_core::GenerationPlacementPreview {
        self.placement_preview_dag_for_device_cancellable(
            request,
            copies,
            prepared_inputs,
            required_device_id,
            &|| false,
        )
    }

    fn placement_preview_dag_for_device_cancellable(
        &self,
        request: &mold_core::GenerateRequest,
        copies: u32,
        prepared_inputs: &crate::execution_plan::PreparedExecutionInputs,
        required_device_id: Option<&str>,
        cancelled: &dyn Fn() -> bool,
    ) -> mold_core::GenerationPlacementPreview {
        let empty = |outcome: &str, reason: String| mold_core::GenerationPlacementPreview {
            version: 1,
            authoritative: true,
            state_version: self.state_version,
            plan_version: self.plan_version,
            outcome: outcome.to_string(),
            reason: Some(reason),
            candidate: None,
            stage_candidates: Vec::new(),
            pending_downloads: Vec::new(),
            missing_components: Vec::new(),
        };
        if !(1..=64).contains(&copies) {
            return empty("infeasible", "copies must be between 1 and 64".to_string());
        }
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        if self.state.queue_pause.is_paused() {
            return empty(
                "temporarily_unavailable",
                "generation queue is paused".to_string(),
            );
        }
        if self
            .state
            .gpu_pool
            .workers
            .iter()
            .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
        {
            return empty(
                "temporarily_unavailable",
                "CUDA context is fatally poisoned; server restart required".to_string(),
            );
        }

        let owner_plan_cache = self
            .pending_owner_work
            .iter()
            .filter_map(|(id, pending)| {
                self.owner_plans(pending)
                    .ok()
                    .map(|plans| (id.clone(), plans))
            })
            .collect::<BTreeMap<_, _>>();
        let (mut snapshot, _) = self.planner_snapshot(&owner_plan_cache);
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        let device_facts = self.device_facts_from_snapshots(&snapshot.devices);
        let device_facts =
            constrained_generation_device_facts(&device_facts, None, required_device_id);
        let config = match self.state.config.try_read() {
            Ok(config) => config,
            Err(_) => {
                return empty(
                    "temporarily_unavailable",
                    "configuration changed while previewing placement".to_string(),
                )
            }
        };
        let offload_requested = matches!(
            mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        let plans = match crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            request,
            &device_facts,
            offload_requested,
            Some(prepared_inputs),
        ) {
            Ok(plans) => plans,
            Err(error) => {
                let failure =
                    classify_generation_plan_failure(error, &self.total_vram_bytes_by_device_id());
                let (authoritative, outcome) =
                    placement_preview_disposition_for_plan_failure(&failure);
                let mut response = empty(outcome, failure.to_string());
                response.authoritative = authoritative;
                return response;
            }
        };
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        let local_expansion_model = if request.expand == Some(true) {
            let settings = config.expand.clone().with_env_overrides();
            settings.is_local().then(|| settings.model.clone())
        } else {
            None
        };
        let post_upscale = request
            .upscale_model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(|model| {
                let resolved = mold_core::manifest::resolve_model_name(model);
                let estimated_vram_bytes = config
                    .models
                    .get(&resolved)
                    .and_then(|model| model.transformer.as_ref())
                    .and_then(|path| std::fs::metadata(path).ok())
                    .map(|metadata| metadata.len().saturating_add(2 << 30))
                    .unwrap_or(2 << 30);
                (resolved, estimated_vram_bytes)
            });
        drop(config);
        let hard_ordinal = match self
            .state
            .gpu_pool
            .resolve_explicit_placement_gpu(request.placement.as_ref())
        {
            Ok(ordinal) => ordinal,
            Err(error) => return empty("infeasible", error),
        };

        let failed = crate::gpu_pool::failed_ordinals_for_model(&request.model);
        let mut confidence_by_edge = BTreeMap::new();
        let candidates = plans
            .into_iter()
            .filter(|plan| !failed.contains(&plan.device_ordinal))
            .filter(|plan| required_device_id.is_none_or(|device_id| plan.device_id == device_id))
            .filter_map(|plan| {
                if cancelled() {
                    return None;
                }
                let worker = self.state.gpu_pool.worker_by_ordinal(plan.device_ordinal)?;
                let key = generation_estimate_key(
                    &self.state,
                    &worker,
                    request,
                    None,
                    &plan.execution_fingerprint,
                );
                let static_estimate = static_generation_estimate(
                    request,
                    plan.predicted_vram_peak_bytes,
                    plan.admission_host_demand_bytes(),
                );
                let estimate = self.estimates.estimate(&key, static_estimate);
                let (cold_setup_ms, warm_setup_ms, predicted_run_ms) =
                    timing_with_static_floors(estimate, static_estimate);
                confidence_by_edge.insert(
                    (plan.device_id.clone(), plan.execution_fingerprint.clone()),
                    wire_estimate_confidence(estimate.confidence),
                );
                let candidate = CandidatePlacement::new(
                    DeviceId::new(plan.device_id),
                    ExecutionFingerprint::new(plan.execution_fingerprint),
                    estimate.host_bytes,
                )
                .with_execution_equivalence(plan.execution_equivalence_fingerprint)
                .with_vram(estimate.vram_bytes)
                .with_timing(cold_setup_ms, warm_setup_ms, predicted_run_ms)
                .with_device_available_vram(plan.admitted_available_vram_bytes);
                Some(
                    if generation_uses_frozen_device_capacity(
                        plan.device_backend,
                        &plan.model_family,
                    ) {
                        candidate.with_frozen_device_capacity()
                    } else {
                        candidate
                    },
                )
            })
            .collect::<Vec<_>>();
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        if candidates.is_empty() {
            return empty(
                "infeasible",
                "no healthy request-eligible device has a concrete execution plan".to_string(),
            );
        }

        let preview_prefix = format!("__placement-preview-{}-", self.state_version);
        let expansion_prefix = format!("{preview_prefix}expand-");
        let generation_prefix = format!("{preview_prefix}generation-");
        let upscale_prefix = format!("{preview_prefix}upscale-");
        let first_rank = snapshot
            .work
            .iter()
            .map(|work| work.queue_rank)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let mut rank_cursor = first_rank;
        let mut ready_at = vec![snapshot.now_ms; copies as usize];
        let mut predicted_start_ms = snapshot.now_ms;
        let mut stage_candidates = Vec::new();
        let mut generation_stage_index = 0_u32;

        if let Some(expansion_model) = &local_expansion_model {
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            snapshot.work.push(self.owner_work_snapshot(
                OwnerWorkSchedulingView {
                    id: &format!("{expansion_prefix}parent"),
                    model_fingerprint: expansion_model,
                    estimated_vram_bytes: 6_000_000_000,
                    estimated_host_ram_bytes: MIN_TRANSIENT_HOST_RAM,
                    hard_ordinal,
                    priority: PriorityClass::User,
                    queue_rank: rank_cursor,
                    ready_at_ms: snapshot.now_ms,
                    bypass_count: 0,
                    warm_wait_started_ms: None,
                    kind: mold_scheduler::WorkKind::PromptExpansion,
                    shape_bucket: "preview:prompt-expansion",
                    preferred_ordinal: None,
                    resolved_plans: &[],
                    requires_exact_plan: false,
                },
                &[],
                &snapshot.devices,
            ));
            rank_cursor = rank_cursor.saturating_add(1);
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            let expansion_plan = match self.planner.plan(&snapshot) {
                Ok(plan) => plan,
                Err(error) => {
                    return empty(
                        "temporarily_unavailable",
                        format!("scheduler could not preview prompt expansion: {error}"),
                    )
                }
            };
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            let expansion_assignments = expansion_plan
                .lanes
                .iter()
                .flat_map(|lane| &lane.assignments)
                .filter(|assignment| assignment.work_id.as_str().starts_with(&expansion_prefix))
                .collect::<Vec<_>>();
            if expansion_assignments.len() != 1 {
                let reason = expansion_plan
                    .blocked
                    .iter()
                    .find(|blocked| blocked.work_id.as_str().starts_with(&expansion_prefix))
                    .map(|blocked| format!("{:?}", blocked.reason).to_ascii_lowercase())
                    .unwrap_or_else(|| {
                        "prompt expansion is beyond the scheduler planning horizon".to_string()
                    });
                return empty("infeasible", reason);
            }
            predicted_start_ms = expansion_assignments
                .iter()
                .map(|assignment| assignment.estimated_start_ms)
                .min()
                .unwrap_or(snapshot.now_ms);
            let assignment = expansion_assignments[0];
            ready_at.fill(assignment.estimated_finish_ms);
            stage_candidates.push(stage_placement_candidate(
                0,
                None,
                assignment,
                &snapshot.devices,
                snapshot.now_ms,
                mold_core::QueueEstimateConfidence::Unknown("owner_work".to_string()),
            ));
            generation_stage_index = 1;
        }

        for index in 0..copies {
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            snapshot.work.push(
                WorkSnapshot::new(
                    WorkId::new(format!("{generation_prefix}{index}")),
                    rank_cursor,
                    candidates.clone(),
                )
                .with_ready_at(ready_at[index as usize]),
            );
            rank_cursor = rank_cursor.saturating_add(1);
        }
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        let generation_plan = match self.planner.plan(&snapshot) {
            Ok(plan) => plan,
            Err(error) => {
                return empty(
                    "temporarily_unavailable",
                    format!("scheduler could not preview this request: {error}"),
                )
            }
        };
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        let mut selected = generation_plan
            .lanes
            .iter()
            .flat_map(|lane| &lane.assignments)
            .filter(|assignment| assignment.work_id.as_str().starts_with(&generation_prefix))
            .collect::<Vec<_>>();
        selected.sort_by(|left, right| left.work_id.cmp(&right.work_id));
        if selected.len() != copies as usize {
            let reason = generation_plan
                .blocked
                .iter()
                .find(|blocked| blocked.work_id.as_str().starts_with(&generation_prefix))
                .map(|blocked| format!("{:?}", blocked.reason).to_ascii_lowercase())
                .unwrap_or_else(|| "request is beyond the scheduler planning horizon".to_string());
            return empty("infeasible", reason);
        }
        let first_device_id = selected[0].device_id.to_string();
        let first_execution_fingerprint = selected[0].placement.execution_fingerprint.to_string();
        let first_execution_equivalence_fingerprint = selected[0]
            .placement
            .execution_equivalence_fingerprint
            .as_ref()
            .map(ToString::to_string);
        let first_cold_setup_ms = selected[0].placement.cold_setup_ms;
        let first_warm_setup_ms = selected[0].placement.warm_setup_ms;
        if local_expansion_model.is_none() {
            predicted_start_ms = selected
                .iter()
                .map(|assignment| assignment.estimated_start_ms)
                .min()
                .unwrap_or(snapshot.now_ms);
        }
        for (index, assignment) in selected.iter().enumerate() {
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            ready_at[index] = assignment.estimated_finish_ms;
            let confidence = confidence_by_edge
                .get(&(
                    assignment.device_id.to_string(),
                    assignment.placement.execution_fingerprint.to_string(),
                ))
                .cloned()
                .unwrap_or_default();
            stage_candidates.push(stage_placement_candidate(
                generation_stage_index,
                Some(index as u32),
                assignment,
                &snapshot.devices,
                snapshot.now_ms,
                confidence,
            ));
        }
        let mut predicted_finish_ms = ready_at.iter().copied().max().unwrap_or(snapshot.now_ms);
        let mut final_plan_version = generation_plan.plan_version;

        if let Some((upscale_model, estimated_vram_bytes)) = &post_upscale {
            for index in 0..copies {
                if cancelled() {
                    return self.cancelled_placement_preview();
                }
                snapshot.work.push(self.owner_work_snapshot(
                    OwnerWorkSchedulingView {
                        id: &format!("{upscale_prefix}{index}"),
                        model_fingerprint: upscale_model,
                        estimated_vram_bytes: *estimated_vram_bytes,
                        estimated_host_ram_bytes: MIN_TRANSIENT_HOST_RAM,
                        hard_ordinal: None,
                        priority: PriorityClass::User,
                        queue_rank: rank_cursor,
                        ready_at_ms: ready_at[index as usize],
                        bypass_count: 0,
                        warm_wait_started_ms: None,
                        kind: mold_scheduler::WorkKind::PostUpscale,
                        shape_bucket: "preview:post-upscale",
                        preferred_ordinal: None,
                        resolved_plans: &[],
                        requires_exact_plan: false,
                    },
                    &[],
                    &snapshot.devices,
                ));
                rank_cursor = rank_cursor.saturating_add(1);
            }
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            let upscale_plan = match self.planner.plan(&snapshot) {
                Ok(plan) => plan,
                Err(error) => {
                    return empty(
                        "temporarily_unavailable",
                        format!("scheduler could not preview post-upscale work: {error}"),
                    )
                }
            };
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            let mut upscale_assignments = upscale_plan
                .lanes
                .iter()
                .flat_map(|lane| &lane.assignments)
                .filter(|assignment| assignment.work_id.as_str().starts_with(&upscale_prefix))
                .collect::<Vec<_>>();
            upscale_assignments.sort_by(|left, right| left.work_id.cmp(&right.work_id));
            if upscale_assignments.len() != copies as usize {
                let reason = upscale_plan
                    .blocked
                    .iter()
                    .find(|blocked| blocked.work_id.as_str().starts_with(&upscale_prefix))
                    .map(|blocked| format!("{:?}", blocked.reason).to_ascii_lowercase())
                    .unwrap_or_else(|| {
                        "post-upscale work is beyond the scheduler planning horizon".to_string()
                    });
                return empty("infeasible", reason);
            }
            predicted_finish_ms = upscale_assignments
                .iter()
                .map(|assignment| assignment.estimated_finish_ms)
                .max()
                .unwrap_or(predicted_finish_ms);
            for (index, assignment) in upscale_assignments.iter().enumerate() {
                if cancelled() {
                    return self.cancelled_placement_preview();
                }
                stage_candidates.push(stage_placement_candidate(
                    generation_stage_index.saturating_add(1),
                    Some(index as u32),
                    assignment,
                    &snapshot.devices,
                    snapshot.now_ms,
                    mold_core::QueueEstimateConfidence::Unknown("owner_work".to_string()),
                ));
            }
            final_plan_version = upscale_plan.plan_version;
        }

        if cancelled() {
            return self.cancelled_placement_preview();
        }

        let warm = snapshot.devices.iter().any(|device| {
            device.id.as_str() == first_device_id
                && device
                    .warm_execution_fingerprints
                    .iter()
                    .any(|fingerprint| fingerprint.as_str() == first_execution_fingerprint)
        });
        let generation_setup_ms = if warm {
            first_warm_setup_ms
        } else {
            first_cold_setup_ms
        };
        let setup_ms = stage_candidates.iter().fold(0_u64, |total, stage| {
            total.saturating_add(stage.candidate.setup_ms)
        });
        debug_assert!(
            !stage_candidates.is_empty(),
            "a planned preview must contain at least one generation stage"
        );
        let setup_kind = if stage_candidates.len() == 1 {
            if warm { "warm" } else { "cold" }.to_string()
        } else if stage_candidates
            .iter()
            .all(|stage| stage.stage_index == generation_stage_index)
        {
            "batch".to_string()
        } else {
            "pipeline".to_string()
        };
        debug_assert!(
            setup_ms >= generation_setup_ms,
            "aggregate setup must include the primary generation"
        );
        let mut confidence = selected
            .iter()
            .filter_map(|assignment| {
                confidence_by_edge.get(&(
                    assignment.device_id.to_string(),
                    assignment.placement.execution_fingerprint.to_string(),
                ))
            })
            .min_by_key(|confidence| match confidence {
                mold_core::QueueEstimateConfidence::Low => 0,
                mold_core::QueueEstimateConfidence::Medium => 1,
                mold_core::QueueEstimateConfidence::High => 2,
                mold_core::QueueEstimateConfidence::Unknown(_) => 0,
            })
            .cloned()
            .unwrap_or_default();
        let mut pending_by_identity = BTreeMap::new();
        for assignment in &selected {
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            for download in
                prepared_inputs.pending_downloads_for_device(assignment.device_id.as_str())
            {
                pending_by_identity.insert(
                    (
                        download.kind.clone(),
                        download.repo.clone(),
                        download.name.clone(),
                        download.bytes,
                    ),
                    download,
                );
            }
        }
        let pending_downloads = pending_by_identity.into_values().collect::<Vec<_>>();
        if !pending_downloads.is_empty() {
            confidence = mold_core::QueueEstimateConfidence::Low;
        }
        for stage in &mut stage_candidates {
            if cancelled() {
                return self.cancelled_placement_preview();
            }
            if !prepared_inputs
                .pending_downloads_for_device(&stage.candidate.device_id)
                .is_empty()
            {
                stage.candidate.estimate_confidence = mold_core::QueueEstimateConfidence::Low;
            }
        }
        if cancelled() {
            return self.cancelled_placement_preview();
        }
        mold_core::GenerationPlacementPreview {
            version: 1,
            authoritative: true,
            state_version: self.state_version,
            plan_version: final_plan_version,
            outcome: "planned".to_string(),
            reason: None,
            candidate: Some(mold_core::GenerationPlacementCandidate {
                device_id: first_device_id,
                execution_fingerprint: first_execution_fingerprint,
                execution_equivalence_fingerprint: first_execution_equivalence_fingerprint,
                predicted_start_after_ms: predicted_start_ms.saturating_sub(snapshot.now_ms),
                predicted_completion_after_ms: predicted_finish_ms.saturating_sub(snapshot.now_ms),
                setup_ms,
                setup_kind,
                estimate_confidence: confidence,
            }),
            stage_candidates,
            pending_downloads,
            missing_components: Vec::new(),
        }
    }

    async fn dispatch_ready(&mut self) -> Option<u64> {
        self.dispatch_ready_with(PlanningPass::Optimize).await
    }

    async fn dispatch_debounced_replan(&mut self) {
        let planned_state_version = self.dispatch_ready().await.or_else(|| {
            (self.pending.is_empty() && self.pending_owner_work.is_empty())
                .then_some(self.state_version)
        });
        if let Some(planned_state_version) = planned_state_version {
            // An empty queue needs no optimizer publication, but its dirty
            // window is still settled. Otherwise a completion that drains
            // the queue would retry this due timer every reconciliation tick.
            self.dirty.clear_through(planned_state_version);
        }
    }

    async fn dispatch_ready_with(&mut self, pass: PlanningPass) -> Option<u64> {
        let planner = match pass {
            PlanningPass::Admission => self.admission_planner.clone(),
            PlanningPass::Optimize => self.planner.clone(),
        };
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
            self.settle_unschedulable_generations();
        }
        if self.pending.is_empty() && self.pending_owner_work.is_empty() {
            let published_work_remains = self
                .state
                .scheduled_work
                .latest_plan()
                .is_some_and(|plan| !plan.work_items.is_empty());
            if published_work_remains {
                // Cancellation or completion can drain the reducer before
                // the next dispatch turn. Replace the prior authority so a
                // paused cancellation cannot leave a ghost work item.
                self.replan_and_publish_with(pass);
                return Some(self.state_version);
            }
            return None;
        }
        if self
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
            let owner_plan_cache = self.owner_plan_cache_and_settle_errors();
            if self.reject_terminal_generation_plan_errors()
                && self.pending.is_empty()
                && self.pending_owner_work.is_empty()
            {
                return None;
            }
            if self.pending.is_empty() && self.pending_owner_work.is_empty() {
                return None;
            }
            let (snapshot, generation_plans) = self.planner_snapshot(&owner_plan_cache);
            let plan = match planner.plan(&snapshot) {
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
            if let Err(error) = self.publish_plan(&snapshot, &plan, self.dirty.dirty_since) {
                tracing::error!(
                    state_version = snapshot.state_version,
                    %error,
                    "scheduler could not publish the runtime queue plan; refusing to grant work"
                );
                return None;
            }
            self.plan_version = plan.plan_version;
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
            if self.state.queue_pause.is_paused() {
                // Pause may race the interval between publication and the
                // mutation-fenced grant. Replace the now-stale runnable plan
                // with typed queue-paused blockers before releasing the fence.
                self.replan_and_publish_with(pass);
                return Some(self.state_version);
            }
            if self
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
                    self.state.job_registry.scheduler_lifecycle(&work_id)
                        != Some(crate::job_registry::JobLifecycle::Queued)
                        || pending.job.should_cancel_for_observer_disconnect()
                } else {
                    utility.is_none_or(|pending| pending.work.is_cancelled())
                };
                let current_execution_fingerprint = if generation.is_some() {
                    let planned_execution = generation_plans
                        .get(&work_id)
                        .and_then(|plans| exact_leased_execution_plan(plans, lease));
                    let current_execution = current_generation_plans
                        .get(&work_id)
                        .and_then(|plans| exact_leased_execution_plan(plans, lease));
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
                    if !utility.utility_plans.is_empty() {
                        let Some(exact) =
                            self.utility_plan_for_lease(utility, &device_id, &lease.placement)
                        else {
                            return false;
                        };
                        ExecutionFingerprint::new(exact.execution_fingerprint())
                    } else if utility.work.chain_plan_inputs().is_some() {
                        let planned_execution =
                            owner_plan_cache.get(&work_id).into_iter().flatten().find(
                                |execution| {
                                    execution.device_id == device_id
                                        && execution.execution_fingerprint
                                            == lease.placement.execution_fingerprint.as_str()
                                        && execution.predicted_vram_peak_bytes
                                            == lease.placement.predicted_vram_bytes
                                        && execution.predicted_host_increment_bytes
                                            == lease.placement.incremental_host_ram_bytes
                                },
                            );
                        let current_execution = self
                            .owner_plans(utility)
                            .ok()
                            .into_iter()
                            .flatten()
                            .find(|execution| execution.device_id == device_id);
                        if planned_execution.is_none()
                            || current_execution.as_ref() != planned_execution
                        {
                            return false;
                        }
                        ExecutionFingerprint::new(
                            current_execution
                                .expect("checked equal owner execution plans")
                                .execution_fingerprint,
                        )
                    } else {
                        ExecutionFingerprint::new(utility.model_fingerprint.clone())
                    }
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
                let id = lease.work_id.to_string();
                let Some(projection) = snapshot
                    .work
                    .iter()
                    .find(|work| work.id == lease.work_id)
                    .cloned()
                else {
                    grant_failed = true;
                    break;
                };
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
                if device_id == CPU_UTILITY_DEVICE_ID {
                    let Some(mut pending) = self.pending_owner_work.remove(&id) else {
                        grant_failed = true;
                        break;
                    };
                    let shape_bucket = pending.work.scheduling_shape_bucket();
                    let work_kind = pending.work.kind();
                    let Some(selected) =
                        self.utility_plan_for_lease(&pending, &device_id, &lease.placement)
                    else {
                        self.pending_owner_work.insert(id.clone(), pending);
                        grant_failed = true;
                        break;
                    };
                    let execution_fingerprint = selected.execution_fingerprint().to_string();
                    if let Err(error) = pending.work.install_utility_plan(selected) {
                        reject_owner_work_preserving_completed_generation(pending.work, error);
                        grant_failed = true;
                        break;
                    }
                    let estimate_key = owner_estimate_key_for_device(
                        CPU_UTILITY_DEVICE_ID.to_string(),
                        work_kind,
                        &pending.model_fingerprint,
                        &shape_bucket,
                        &execution_fingerprint,
                    );
                    let retry = crate::gpu_pool::OwnerWorkRetry {
                        model_fingerprint: pending.model_fingerprint.clone(),
                        estimated_vram_bytes: pending.estimated_vram_bytes,
                        estimated_host_ram_bytes: pending.estimated_host_ram_bytes,
                        hard_ordinal: pending.hard_ordinal,
                        priority: pending.priority,
                        preferred_ordinal: pending.preferred_ordinal,
                        candidate_plans: pending.candidate_plans.clone(),
                        queue_rank: pending.queue_rank,
                        ready_at_ms: pending.ready_at_ms,
                        bypass_count: pending.bypass_count,
                        warm_wait_started_ms: pending.warm_wait_started_ms,
                        retry_not_before_ms: pending.retry_not_before_ms,
                        utility_plans: pending.utility_plans.clone(),
                    };
                    let ready_at_ms = pending.ready_at_ms;
                    let bypass_count = pending.bypass_count;
                    let warm_wait_started_ms = pending.warm_wait_started_ms;
                    let grant = Box::new(LeaseGrant {
                        fence,
                        work: pending.work,
                        retry: Some(retry),
                    });
                    let Some(cpu_tx) = &self.cpu_utility_tx else {
                        reject_owner_work_preserving_completed_generation(
                            grant.work,
                            "CPU utility lane is unavailable".to_string(),
                        );
                        grant_failed = true;
                        break;
                    };
                    match cpu_tx.try_send(crate::gpu_pool::GpuWorkerCommand::Grant(grant)) {
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
                                    ready_at_ms,
                                    bypass_count,
                                    warm_wait_started_ms,
                                    started_at: Instant::now(),
                                    estimate_key,
                                    vram_high_water_bytes: None,
                                    host_incremental_high_water_bytes: None,
                                    fallback_reason: None,
                                    projection,
                                    assignment_reason: lease.reason,
                                },
                            );
                            granted.push(id);
                        }
                        Err(error) => {
                            let returned = match error {
                                std::sync::mpsc::TrySendError::Full(command)
                                | std::sync::mpsc::TrySendError::Disconnected(command) => command,
                            };
                            let crate::gpu_pool::GpuWorkerCommand::Grant(returned) = returned
                            else {
                                unreachable!("CPU utility dispatch sends only grants")
                            };
                            let retry = returned
                                .retry
                                .expect("CPU utility grant carries exact retry metadata");
                            self.pending_owner_work.insert(
                                id,
                                PendingOwnerWork {
                                    model_fingerprint: retry.model_fingerprint,
                                    estimated_vram_bytes: retry.estimated_vram_bytes,
                                    estimated_host_ram_bytes: retry.estimated_host_ram_bytes,
                                    hard_ordinal: retry.hard_ordinal,
                                    priority: retry.priority,
                                    preferred_ordinal: retry.preferred_ordinal,
                                    candidate_plans: retry.candidate_plans,
                                    queue_rank: retry.queue_rank,
                                    ready_at_ms: retry.ready_at_ms,
                                    bypass_count: retry.bypass_count,
                                    warm_wait_started_ms: retry.warm_wait_started_ms,
                                    retry_not_before_ms: retry.retry_not_before_ms,
                                    utility_plans: retry.utility_plans,
                                    work: returned.work,
                                },
                            );
                            self.unavailable.insert(device_id.clone());
                            grant_failed = true;
                        }
                    }
                    if grant_failed {
                        break;
                    }
                    continue;
                }
                let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ready.ordinal) else {
                    grant_failed = true;
                    break;
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
                        .and_then(|plans| exact_leased_execution_plan(plans, lease));
                    let Some(mut execution_plan) = execution_plan else {
                        self.pending.insert(id.clone(), pending);
                        worker.release_in_flight();
                        self.state_version = self.state_version.saturating_add(1);
                        grant_failed = true;
                        break;
                    };
                    let bypass_count = pending.bypass_count;
                    let queue_rank = pending.queue_rank;
                    let warm_wait_started_ms = pending.warm_wait_started_ms;
                    let ready_at_ms = pending.ready_at_ms;
                    let retry_not_before_ms = pending.retry_not_before_ms;
                    let prepared_inputs = pending.prepared_inputs.clone();
                    let estimate_key = generation_estimate_key(
                        &self.state,
                        &worker,
                        &pending.job.request,
                        pending
                            .job
                            .deferred_media
                            .as_ref()
                            .map(|media| media.projection()),
                        &execution_plan.execution_fingerprint,
                    );
                    // Admission reserved `max(static, learned)`; carry the
                    // learned half so the worker rechecks the same number.
                    execution_plan.learned_vram_envelope_bytes = self
                        .estimates
                        .exact(&estimate_key)
                        .and_then(|bucket| bucket.vram_conservative_bytes)
                        .unwrap_or(0);
                    let fallback_reason = execution_fallback_reason(&execution_plan);
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
                                    vram_high_water_bytes: None,
                                    host_incremental_high_water_bytes: None,
                                    fallback_reason,
                                    projection,
                                    assignment_reason: lease.reason,
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
                                    queue_rank,
                                    bypass_count,
                                    warm_wait_started_ms,
                                    preparation: PreparationState::Ready,
                                    prepared_inputs,
                                    retry_not_before_ms,
                                    preparation_retry_attempts: 0,
                                    preparation_refresh_observation: None,
                                    unschedulable_since_ms: None,
                                    unschedulable_reason: None,
                                    announced_position: None,
                                },
                            );
                            self.unavailable.insert(device_id.clone());
                            self.state.device_registry.mark_unavailable(&device_id);
                            grant_failed = true;
                            break;
                        }
                    }
                } else if let Some(mut pending) = self.pending_owner_work.remove(&id) {
                    let shape_bucket = pending.work.scheduling_shape_bucket();
                    let work_kind = pending.work.kind();
                    if !pending.utility_plans.is_empty() {
                        let Some(selected) =
                            self.utility_plan_for_lease(&pending, &device_id, &lease.placement)
                        else {
                            self.pending_owner_work.insert(id.clone(), pending);
                            worker.release_in_flight();
                            self.state_version = self.state_version.saturating_add(1);
                            grant_failed = true;
                            break;
                        };
                        if let Err(error) = pending.work.install_utility_plan(selected) {
                            worker.release_in_flight();
                            reject_owner_work_preserving_completed_generation(pending.work, error);
                            self.memory.release(&id);
                            self.state_version = self.state_version.saturating_add(1);
                            grant_failed = true;
                            break;
                        }
                    }
                    let estimate_key = owner_estimate_key(
                        &worker,
                        work_kind,
                        &pending.model_fingerprint,
                        &shape_bucket,
                        lease.placement.execution_fingerprint.as_str(),
                    );
                    let resolved_plans = owner_plan_cache.get(&id).cloned().unwrap_or_default();
                    let chosen_plan = exact_leased_execution_plan(&resolved_plans, lease);
                    let metadata = (
                        pending.model_fingerprint,
                        pending.estimated_vram_bytes,
                        pending.estimated_host_ram_bytes,
                        pending.hard_ordinal,
                        pending.priority,
                        pending.preferred_ordinal,
                        pending.candidate_plans,
                        pending.queue_rank,
                        pending.ready_at_ms,
                        pending.bypass_count,
                        pending.warm_wait_started_ms,
                        pending.retry_not_before_ms,
                        pending.utility_plans,
                    );
                    let mut work = pending.work;
                    if let Some(plan) = chosen_plan {
                        work.apply_execution_plan(plan);
                    }
                    let grant = Box::new(LeaseGrant {
                        fence,
                        work,
                        retry: Some(crate::gpu_pool::OwnerWorkRetry {
                            model_fingerprint: metadata.0.clone(),
                            estimated_vram_bytes: metadata.1,
                            estimated_host_ram_bytes: metadata.2,
                            hard_ordinal: metadata.3,
                            priority: metadata.4,
                            preferred_ordinal: metadata.5,
                            candidate_plans: metadata.6.clone(),
                            queue_rank: metadata.7,
                            ready_at_ms: metadata.8,
                            bypass_count: metadata.9,
                            warm_wait_started_ms: metadata.10,
                            retry_not_before_ms: metadata.11,
                            utility_plans: metadata.12.clone(),
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
                                    ready_at_ms: metadata.8,
                                    bypass_count: metadata.9,
                                    warm_wait_started_ms: metadata.10,
                                    started_at: Instant::now(),
                                    estimate_key,
                                    vram_high_water_bytes: None,
                                    host_incremental_high_water_bytes: None,
                                    fallback_reason: None,
                                    projection,
                                    assignment_reason: lease.reason,
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
                                    preferred_ordinal: metadata.5,
                                    candidate_plans: metadata.6,
                                    queue_rank: metadata.7,
                                    ready_at_ms: metadata.8,
                                    bypass_count: metadata.9,
                                    warm_wait_started_ms: metadata.10,
                                    retry_not_before_ms: metadata.11,
                                    utility_plans: metadata.12,
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
                } else if let Some(pending) =
                    self.pending_owner_work.get_mut(update.work_id.as_str())
                {
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
            self.replan_and_publish_with(pass);
            return Some(plan.state_version);
        }
    }

    fn try_replan_and_publish_with(
        &mut self,
        pass: PlanningPass,
    ) -> Result<(), PlanPublicationError> {
        let queue_paused = self.state.queue_pause.is_paused();
        self.try_replan_and_publish_with_queue_state(pass, queue_paused, false)
            .map(|_| ())
    }

    fn try_replan_and_publish_with_queue_state(
        &mut self,
        pass: PlanningPass,
        queue_paused: bool,
        advance_state_version: bool,
    ) -> Result<(u64, Option<ReplanWindow>), PlanPublicationError> {
        let owner_plan_cache = self.owner_plan_cache_and_settle_errors();
        let prospective_state_version = self
            .state_version
            .saturating_add(u64::from(advance_state_version));
        let (mut snapshot, _) = self.planner_snapshot(&owner_plan_cache);
        snapshot.state_version = prospective_state_version;
        snapshot.queue_paused = queue_paused;
        let prospective_dirty = advance_state_version.then(|| {
            let mut dirty = self.dirty.clone();
            dirty.mark_dirty(Instant::now(), prospective_state_version);
            snapshot.next_replan_at_ms = dirty.deadline().map(monotonic_deadline_ms);
            dirty
        });
        let planner = match pass {
            PlanningPass::Admission => &self.admission_planner,
            PlanningPass::Optimize => &self.planner,
        };
        #[cfg(test)]
        let plan_result = self
            .next_plan_error
            .take()
            .map_or_else(|| planner.plan(&snapshot), Err);
        #[cfg(not(test))]
        let plan_result = planner.plan(&snapshot);
        let plan = plan_result?;
        let published_dirty_since = prospective_dirty
            .as_ref()
            .and_then(|dirty| dirty.dirty_since)
            .or(self.dirty.dirty_since);
        self.publish_plan(&snapshot, &plan, published_dirty_since)?;
        self.plan_version = plan.plan_version;
        Ok((prospective_state_version, prospective_dirty))
    }

    #[cfg(test)]
    fn replan_and_publish(&mut self) {
        self.replan_and_publish_with(PlanningPass::Optimize);
    }

    fn replan_and_publish_with(&mut self, pass: PlanningPass) {
        if let Err(error) = self.try_replan_and_publish_with(pass) {
            tracing::error!(
                state_version = self.state_version,
                %error,
                "scheduler could not publish an observational queue plan"
            );
        }
    }

    fn set_queue_paused_and_publish(&mut self, paused: bool) -> Result<bool, String> {
        let was_paused = self.state.queue_pause.is_paused();
        let changed = was_paused != paused;
        let mut immediate = false;
        self.reconcile_external_mutations(&mut immediate);
        #[cfg(test)]
        if let Some(hook) = self.before_queue_control_plan_hook.take() {
            hook();
        }
        match self.try_replan_and_publish_with_queue_state(PlanningPass::Admission, paused, changed)
        {
            Ok((published_state_version, prospective_dirty)) => {
                if changed {
                    if paused {
                        self.state.queue_pause.pause();
                    } else {
                        self.state.queue_pause.resume();
                    }
                    self.last_paused = paused;
                    self.state_version = published_state_version;
                    self.dirty = prospective_dirty
                        .expect("changed queue pause state must carry its replan window");
                    self.state.events.publish(if paused {
                        mold_core::ServerEvent::QueuePaused
                    } else {
                        mold_core::ServerEvent::QueueResumed
                    });
                }
                Ok(changed)
            }
            Err(error) => Err(format!(
                "scheduler could not publish queue pause state: {error}"
            )),
        }
    }

    fn publish_plan(
        &self,
        snapshot: &PlannerSnapshot,
        plan: &Plan,
        dirty_since: Option<Instant>,
    ) -> Result<(), PlanPublicationError> {
        let mut confidence = plan
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
                                    &self.state,
                                    worker.as_ref(),
                                    &pending.job.request,
                                    pending
                                        .job
                                        .deferred_media
                                        .as_ref()
                                        .map(|media| media.projection()),
                                    assignment.placement.execution_fingerprint.as_str(),
                                )
                            })
                        });
                    let confidence = key
                        .as_ref()
                        .map(|key| {
                            wire_estimate_confidence(
                                self.estimates
                                    .estimate(key, StaticEstimate::default())
                                    .confidence,
                            )
                        })
                        .unwrap_or_default();
                    (assignment.work_id.to_string(), confidence)
                })
            })
            .collect::<BTreeMap<_, _>>();
        for lease in self.leases.values() {
            let confidence_value = wire_estimate_confidence(
                self.estimates
                    .estimate(&lease.estimate_key, StaticEstimate::default())
                    .confidence,
            );
            confidence.insert(lease.work_id.clone(), confidence_value);
        }
        let mut wire = queue_plan_projection(
            snapshot,
            plan,
            &self.state.gpu_pool,
            &self.leases,
            &confidence,
            dirty_since,
        );
        wire.host_memory = self.memory.wire_snapshot();
        let mut current = self
            .state
            .scheduled_work
            .latest_plan
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = current.as_ref() {
            let exact_authority = existing.plan_version == wire.plan_version
                && existing.state_version == wire.state_version;
            let semantically_equal = queue_plan_semantically_equal(existing, &wire);
            if exact_authority && semantically_equal {
                return Ok(());
            }
            if existing.plan_version >= wire.plan_version
                || existing.state_version > wire.state_version
            {
                return Err(PlanPublicationError::AuthorityConflict {
                    current_plan_version: existing.plan_version,
                    current_state_version: existing.state_version,
                    produced_plan_version: wire.plan_version,
                    produced_state_version: wire.state_version,
                });
            }
            if semantically_equal {
                // Avoid a redundant event, but refresh the shared authority.
                // Actor acknowledgement may treat a semantic no-op as
                // published only when its version and state are current.
                *current = Some(wire);
                return Ok(());
            }
        }
        *current = Some(wire.clone());
        drop(current);
        self.state
            .events
            .publish(mold_core::ServerEvent::QueuePlanChanged {
                plan: Box::new(wire),
            });
        Ok(())
    }

    fn publish_device_state_if_changed(&mut self) {
        let state = crate::routes::current_device_state(&self.state);
        let signature = device_event_signature(&state);
        if self.last_device_event_signature.as_ref() == Some(&signature) {
            return;
        }
        let previous = self.last_device_event_signature.as_ref();
        let mut changes = signature
            .iter()
            .zip(&state.devices)
            .filter(|(current, _)| {
                previous
                    .and_then(|previous| previous.iter().find(|candidate| candidate.0 == current.0))
                    != Some(*current)
            })
            .map(|(_, device)| {
                (
                    device.id.clone(),
                    device.desired_enabled,
                    device.admin_state,
                )
            })
            .collect::<Vec<_>>();
        if let Some(previous) = previous {
            changes.extend(
                previous
                    .iter()
                    .filter(|old| {
                        !signature
                            .iter()
                            .any(|current| current.0.as_str() == old.0.as_str())
                    })
                    .map(|old| (old.0.clone(), false, mold_core::DeviceAdminState::Disabled)),
            );
        }
        self.last_device_event_signature = Some(signature);
        for (device_id, desired_enabled, admin_state) in changes {
            self.state
                .events
                .publish(mold_core::ServerEvent::DeviceStateChanged {
                    device_id,
                    desired_enabled,
                    admin_state,
                });
        }
    }

    fn reject_all_unstarted_for_fatal_cuda(&mut self) {
        self.retain_all_unstarted("CUDA context is fatally poisoned; server restart required");
    }

    fn retain_all_unstarted(&mut self, message: &str) {
        let pending = std::mem::take(&mut self.pending);
        self.plan_invalidations.clear();
        for (_, pending) in pending {
            retain_generation(&self.state, pending.job, message.to_string());
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
    mut preview_rx: tokio::sync::mpsc::Receiver<PlacementPreviewQuery>,
    mut worker_rx: tokio::sync::mpsc::UnboundedReceiver<WorkerEvent>,
    worker_tx: tokio::sync::mpsc::UnboundedSender<WorkerEvent>,
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) {
    tracing::info!(
        workers = state.gpu_pool.worker_count(),
        "multi-GPU scheduler coordinator started"
    );
    let (cpu_utility_tx, cpu_utility_rx) = std::sync::mpsc::sync_channel(1);
    let cpu_utility_handle = crate::gpu_worker::spawn_cpu_utility_thread(cpu_utility_rx, worker_tx);
    let mut coordinator = Coordinator::new(state).await;
    // The constructor's first reading predates the shared handle, so publish
    // it before serving a status request that would otherwise report nothing.
    coordinator
        .state
        .scheduled_work
        .publish_host_memory(coordinator.memory.wire_snapshot());
    coordinator.install_cpu_utility_lane(cpu_utility_tx.clone());
    let registry_notify = coordinator.state.job_registry.mutation_notifier();
    let mut resource_rx = coordinator.state.resources.subscribe();
    let mut ticker = tokio::time::interval(RECONCILE_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut memory_ticker = tokio::time::interval(MEMORY_SAMPLE_INTERVAL);
    memory_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut fatal = false;
    let mut generation_ingress_open = true;
    let mut owner_ingress_open = true;
    let mut resource_stream_open = true;
    loop {
        let mut immediate = false;
        tokio::select! {
            _ = shutdown.cancelled() => {
                job_rx.close();
                owner_work_rx.close();
                while let Ok(job) = job_rx.try_recv() {
                    retain_generation(
                        &coordinator.state,
                        job,
                        "generation scheduler is shutting down".to_string(),
                    );
                }
                while let Ok(work) = owner_work_rx.try_recv() {
                    work.work
                        .reject("generation scheduler is shutting down".to_string());
                }
                coordinator.retain_all_unstarted("generation scheduler is shutting down");
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
            preview = preview_rx.recv() => {
                if let Some(preview) = preview {
                    match preview {
                        PlacementPreviewQuery::Generation {
                            request,
                            copies,
                            prepared_inputs,
                            reply_tx,
                        } => {
                            // Dropping the HTTP request drops the oneshot
                            // receiver. Do not spend scheduler time planning a
                            // print the caller has already cancelled.
                            if !reply_tx.is_closed() {
                                let response = coordinator.placement_preview_cancellable(
                                    &request,
                                    copies,
                                    &prepared_inputs,
                                    &|| reply_tx.is_closed(),
                                );
                                let _ = reply_tx.send(response);
                            }
                        }
                        PlacementPreviewQuery::BatchDevices {
                            request,
                            parent_size,
                            prepared_inputs,
                            reply_tx,
                        } => {
                            if !reply_tx.is_closed() {
                                let response = coordinator
                                    .batch_device_profiles_cancellable(
                                        &request,
                                        parent_size,
                                        &prepared_inputs,
                                        &|| reply_tx.is_closed(),
                                    )
                                    .map_err(|error| format!("{error:#}"));
                                let _ = reply_tx.send(response);
                            }
                        }
                        PlacementPreviewQuery::SetQueuePaused { paused, reply_tx } => {
                            let response = coordinator.set_queue_paused_and_publish(paused);
                            if !paused && response.is_ok() {
                                immediate = true;
                            }
                            let _ = reply_tx.send(response);
                        }
                    }
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
            resource = resource_rx.recv(), if resource_stream_open => {
                match resource {
                    Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        coordinator.reconcile_resource_capacity(&mut immediate);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        resource_stream_open = false;
                    }
                }
            }
            _ = ticker.tick() => {
                coordinator.reconcile_external_mutations(&mut immediate);
            }
            _ = memory_ticker.tick() => {
                coordinator.collect_host_memory();
                coordinator.sample_active_lease_high_waters();
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
                retain_generation(
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
            let _ = coordinator
                .dispatch_ready_with(PlanningPass::Admission)
                .await;
        }
        if coordinator.dirty.due(Instant::now()) {
            coordinator.dispatch_debounced_replan().await;
        }
        if coordinator.device_state_dirty {
            coordinator.device_state_dirty = false;
            coordinator.publish_device_state_if_changed();
        }
    }
    coordinator.stop_preparations().await;
    let _ = cpu_utility_tx.send(crate::gpu_pool::GpuWorkerCommand::Shutdown);
    if cpu_utility_handle.join().is_err() {
        tracing::error!("CPU utility owner panicked during shutdown");
    }
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
        durable_queue_rank: job.durable_queue_rank,
        model: job.request.model.clone(),
        request: job.request,
        deferred_media: job.deferred_media,
        resolved_references: job.resolved_references,
        completion_payload: job.completion_payload,
        progress_tx: job.progress_tx,
        result_tx: job.result_tx,
        output_dir: job.output_dir,
        config: state.config.clone(),
        metadata_db: state.metadata_db.clone(),
        gallery_publication_gate: state.gallery_publication_gate.clone(),
        queue: state.queue.clone(),
        registry: state.job_registry.clone(),
        events: state.events.clone(),
        execution_plan,
        prepared_execution_inputs,
        #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
        h3_prepared_attempt: None,
        lease: Some(lease),
        batch_child: job.batch_child,
        journal: job.journal,
    }
}

fn generation_and_prepared_from_gpu_job(
    job: GpuJob,
) -> (
    GenerationJob,
    Option<crate::execution_plan::PreparedExecutionInputs>,
) {
    let prepared = job.prepared_execution_inputs;
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let h3_private_ingress_grant = prepared
        .as_ref()
        .and_then(|inputs| inputs.h3_private_ingress_grant.clone());
    (
        GenerationJob {
            id: job.id,
            durable_queue_rank: job.durable_queue_rank,
            request: job.request,
            deferred_media: job.deferred_media,
            resolved_references: job.resolved_references,
            completion_payload: job.completion_payload,
            progress_tx: job.progress_tx,
            result_tx: job.result_tx,
            output_dir: job.output_dir,
            batch_child: job.batch_child,
            journal: job.journal,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant,
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

fn exact_leased_execution_plan(
    plans: &[crate::execution_plan::ResolvedExecutionPlan],
    lease: &mold_scheduler::ImmediateLease,
) -> Option<crate::execution_plan::ResolvedExecutionPlan> {
    // Lease memory is a conservative scheduling envelope and may exceed the
    // immutable plan's static prediction after learned high-water samples.
    // Device plus execution fingerprint is the transport identity; callers
    // separately validate current capacity and full plan equality.
    plans
        .iter()
        .find(|plan| {
            plan.device_id == lease.device_id.as_str()
                && plan.execution_fingerprint == lease.placement.execution_fingerprint.as_str()
        })
        .cloned()
}

fn reject_generation(state: &AppState, mut job: GenerationJob, error: String) {
    if let Some(progress) = job.progress_tx {
        let _ = progress.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
            error.clone(),
        )));
    }
    let id = job.id.clone();
    if let Some(ticket) = job.journal.take() {
        // Deterministic planning and validation refusals remain visible but
        // cannot be retried unchanged. Transient preparation failures use the
        // explicit retryable path below; shutdown uses retention for replay.
        ticket.hold(&error);
    }
    let _ = job.result_tx.send(Err(error));
    state.queue.decrement();
    state.job_registry.remove(&id);
}

/// Deferred dependency failures happen after durable acknowledgement. Preserve
/// that accepted request as an explicitly retryable hold instead of turning a
/// transient download, probe, or preparation error into terminal data loss.
/// Non-durable jobs retain their legacy terminal behavior.
fn hold_preparation_failure(state: &AppState, job: GenerationJob, error: String) {
    let mut job = job;
    if let Some(progress) = job.progress_tx {
        let _ = progress.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
            error.clone(),
        )));
    }
    let id = job.id.clone();
    if let Some(ticket) = job.journal.take() {
        ticket.hold_retryable(&error);
    }
    let _ = job.result_tx.send(Err(error));
    state.queue.decrement();
    state.job_registry.remove(&id);
}

/// A process-level interruption is not a job failure. Release the durable
/// claim so the next feeder pass or process boot replays it automatically.
fn retain_generation(state: &AppState, mut job: GenerationJob, error: String) {
    if let Some(ticket) = job.journal.take() {
        let _ = ticket.retain();
    }
    let id = job.id.clone();
    let _ = job.result_tx.send(Err(error));
    state.queue.decrement();
    state.job_registry.remove(&id);
}

/// Compose the refusal for a queued generation that never resolved a plan.
///
/// The retained `reason` is the planner's own text — for a memory shortfall
/// that is `insufficient_vram_error`'s per-device "needs ~X GB but only ~Y GB
/// is currently available", the same named required-vs-sampled shape an
/// admission refusal gives. Only when the resolver produced no plans AND no
/// error (the coordinator's own zero-candidate case) does the bare sentence
/// stand alone — which is still an answer, and #1272 had none.
fn unschedulable_rejection_message(model: &str, reason: Option<&str>) -> String {
    let base =
        format!("no device could produce an execution plan for '{model}' while the queue was idle");
    match reason {
        Some(reason) if !reason.is_empty() => format!("{base}: {reason}"),
        _ => base,
    }
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

fn queue_blocked_reason(reason: BlockedReason) -> mold_core::QueueBlockedReason {
    use mold_core::QueueBlockedReason as Wire;
    match reason {
        BlockedReason::NotReady => Wire::DependencyWait,
        BlockedReason::DeviceDisabled => Wire::DeviceDisabled,
        BlockedReason::DeviceDraining => Wire::DeviceDraining,
        BlockedReason::DeviceStartupExcluded => Wire::DeviceStartupExcluded,
        BlockedReason::DeviceUnavailable => Wire::DeviceUnavailable,
        BlockedReason::DeviceDegraded => Wire::DeviceDegraded,
        BlockedReason::NoSchedulableDevice => Wire::NoSchedulableDevice,
        BlockedReason::NoIdleDevice => Wire::NoIdleDevice,
        BlockedReason::HardPinUnavailable => Wire::HardPinUnavailable,
        BlockedReason::BackendUnsupported => Wire::BackendUnsupported,
        BlockedReason::InsufficientVram => Wire::InsufficientVram,
        BlockedReason::InsufficientHostRam => Wire::InsufficientHostRam,
        BlockedReason::AggregateHostRamReserved => Wire::AggregateHostRamReserved,
        BlockedReason::ModelNotInstalled => Wire::ModelNotInstalled,
        BlockedReason::ExecutionPlanIncompatible => Wire::ExecutionPlanIncompatible,
        BlockedReason::QueuePaused => Wire::QueuePaused,
        BlockedReason::MaintenanceMode => Wire::MaintenanceMode,
        BlockedReason::Cancelling => Wire::Cancelling,
        BlockedReason::WarmWait => Wire::WarmWait,
        BlockedReason::LowerPriorityOpening => Wire::LowerPriorityOpening,
    }
}

fn queue_plan_projection(
    snapshot: &PlannerSnapshot,
    plan: &Plan,
    pool: &crate::gpu_pool::GpuPool,
    leases: &BTreeMap<String, ActiveLease>,
    confidence_by_work: &BTreeMap<String, mold_core::QueueEstimateConfidence>,
    dirty_since: Option<Instant>,
) -> mold_core::QueuePlan {
    let unix_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    queue_plan_projection_at_unix(
        snapshot,
        plan,
        pool,
        leases,
        confidence_by_work,
        dirty_since,
        unix_now,
    )
}

fn queue_plan_projection_at_unix(
    snapshot: &PlannerSnapshot,
    plan: &Plan,
    pool: &crate::gpu_pool::GpuPool,
    leases: &BTreeMap<String, ActiveLease>,
    confidence_by_work: &BTreeMap<String, mold_core::QueueEstimateConfidence>,
    dirty_since: Option<Instant>,
    unix_now: u64,
) -> mold_core::QueuePlan {
    // Every assignment deadline in `plan` is relative to the single monotonic
    // sample captured by `snapshot`. Sampling the clock again here silently
    // shortened projected durations by the time spent planning and made exact
    // ETA assertions load-dependent.
    let monotonic_now = snapshot.now_ms;
    let to_unix = |deadline: u64| unix_now.saturating_add(deadline.saturating_sub(monotonic_now));
    let ordinals = pool
        .workers
        .iter()
        .map(|worker| (worker_device_id(worker.as_ref()), worker.gpu.ordinal))
        .collect::<BTreeMap<_, _>>();

    let active_devices = leases.keys().cloned().collect::<BTreeSet<_>>();
    let mut work_items = leases
        .iter()
        .map(|(device_id, lease)| {
            let work = &lease.projection;
            let hard_id = work.hard_device_id.as_ref().map(ToString::to_string);
            let host_utility = device_id == CPU_UTILITY_DEVICE_ID;
            mold_core::QueueWorkItem {
                work_id: work.id.to_string(),
                parent_id: work.parent_id.to_string(),
                work_kind: snake_debug(work.kind),
                chain_stage: work.chain_stage,
                batch_partition: work.batch_partition.as_ref().map(|partition| {
                    mold_core::QueueBatchPartition {
                        index: partition.index,
                        count: partition.count,
                        size: partition.size,
                    }
                }),
                priority_class: snake_debug(work.priority_class),
                queue_rank: work.queue_rank,
                bypass_count: work.bypass_count,
                gpu: ordinals.get(device_id).copied(),
                hard_pinned_device_id: hard_id,
                // Legacy clients have always omitted target_gpu after dispatch.
                target_gpu: None,
                planned_lane_kind: Some(if host_utility {
                    mold_core::QueuePlannedLaneKind::HostUtility
                } else {
                    mold_core::QueuePlannedLaneKind::Device
                }),
                planned_device_id: (!host_utility).then(|| device_id.clone()),
                lane_order: Some(0),
                estimated_start_unix_ms: Some(
                    unix_now.saturating_sub(
                        lease
                            .started_at
                            .elapsed()
                            .as_millis()
                            .try_into()
                            .unwrap_or(u64::MAX),
                    ),
                ),
                estimated_finish_unix_ms: Some(to_unix(lease.estimated_finish_ms)),
                estimate_confidence: confidence_by_work
                    .get(work.id.as_str())
                    .cloned()
                    .unwrap_or_default(),
                reason: Some(snake_debug(lease.assignment_reason)),
                blocked_reason: None,
                assignment_reason: Some(snake_debug(lease.assignment_reason)),
                warm_wait_deadline_unix_ms: None,
                activity_phase: if lease.accepted && host_utility {
                    mold_core::QueueActivityPhase::Cpu
                } else if lease.accepted {
                    mold_core::QueueActivityPhase::Active
                } else {
                    mold_core::QueueActivityPhase::Dispatching
                },
                execution_fingerprint: Some(lease.estimate_key.execution_fingerprint.clone()),
                execution_equivalence_fingerprint: work
                    .candidate_placements
                    .iter()
                    .find(|placement| {
                        placement.device_id.as_str() == device_id
                            && placement.execution_fingerprint.as_str()
                                == lease.estimate_key.execution_fingerprint
                    })
                    .and_then(|placement| {
                        placement
                            .execution_equivalence_fingerprint
                            .as_ref()
                            .map(ToString::to_string)
                    }),
            }
        })
        .collect::<Vec<_>>();
    work_items.extend(
        snapshot
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
                let legacy_reason = blocked
                    .map(|blocked| snake_debug(blocked.reason))
                    .or_else(|| warm_wait.map(|_| "warm_wait".to_string()))
                    .or_else(|| {
                        plan.immediate_leases
                            .iter()
                            .find(|lease| lease.work_id == work.id)
                            .map(|lease| snake_debug(lease.reason))
                    });
                let (planned_lane_kind, planned_device_id, lane_order, start, finish) = planned
                    .map_or(
                        (None, None, None, None, None),
                        |(lane, order, assignment)| {
                            let host_utility = lane.device_id.as_str() == CPU_UTILITY_DEVICE_ID;
                            (
                                Some(if host_utility {
                                    mold_core::QueuePlannedLaneKind::HostUtility
                                } else {
                                    mold_core::QueuePlannedLaneKind::Device
                                }),
                                (!host_utility).then(|| lane.device_id.to_string()),
                                Some(
                                    order
                                        + usize::from(
                                            active_devices.contains(lane.device_id.as_str()),
                                        ),
                                ),
                                Some(to_unix(assignment.estimated_start_ms)),
                                Some(to_unix(assignment.estimated_finish_ms)),
                            )
                        },
                    );
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
                    chain_stage: work.chain_stage,
                    batch_partition: work.batch_partition.as_ref().map(|partition| {
                        mold_core::QueueBatchPartition {
                            index: partition.index,
                            count: partition.count,
                            size: partition.size,
                        }
                    }),
                    priority_class: snake_debug(work.priority_class),
                    queue_rank: work.queue_rank,
                    bypass_count: work.bypass_count,
                    gpu: planned_gpu,
                    hard_pinned_device_id: hard_id,
                    target_gpu,
                    planned_lane_kind,
                    planned_device_id,
                    lane_order,
                    estimated_start_unix_ms: start,
                    estimated_finish_unix_ms: finish,
                    estimate_confidence: confidence_by_work
                        .get(work.id.as_str())
                        .cloned()
                        .unwrap_or_default(),
                    reason: legacy_reason,
                    blocked_reason: blocked.map(|blocked| queue_blocked_reason(blocked.reason)),
                    assignment_reason: plan
                        .immediate_leases
                        .iter()
                        .find(|lease| lease.work_id == work.id)
                        .map(|lease| snake_debug(lease.reason)),
                    warm_wait_deadline_unix_ms: warm_wait.map(|wait| to_unix(wait.deadline_ms)),
                    activity_phase: if warm_wait.is_some() {
                        mold_core::QueueActivityPhase::WarmWait
                    } else if blocked.is_some() {
                        mold_core::QueueActivityPhase::Blocked
                    } else {
                        mold_core::QueueActivityPhase::Queued
                    },
                    execution_fingerprint: planned.map(|(_, _, assignment)| {
                        assignment.placement.execution_fingerprint.to_string()
                    }),
                    execution_equivalence_fingerprint: planned.and_then(|(_, _, assignment)| {
                        assignment
                            .placement
                            .execution_equivalence_fingerprint
                            .as_ref()
                            .map(ToString::to_string)
                    }),
                }
            })
            .collect::<Vec<_>>(),
    );

    mold_core::QueuePlan {
        plan_version: plan.plan_version,
        state_version: plan.state_version,
        optimizer_state: snake_debug(plan.optimizer_state),
        dirty_since_unix_ms: dirty_since.map(|started| {
            unix_now.saturating_sub(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
        }),
        next_replan_at_unix_ms: plan.next_replan_at_ms.map(to_unix),
        work_items,
        // Attached by the publisher: telemetry is not a projection of the
        // plan, and it must not widen this signature.
        host_memory: None,
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
                    model_family: record.model_family,
                    model_fingerprint: record.model_fingerprint,
                    work_kind: record.work_kind,
                    shape_bucket: record.shape_bucket,
                    execution_fingerprint: record.execution_fingerprint,
                },
                sample_count: record.sample_count,
                ewma_total_ms: record.ewma_total_ms,
                ewma_runtime_ms: record.ewma_runtime_ms,
                ewma_load_ms: record.ewma_load_ms,
                ewma_warm_reload_ms: record.ewma_warm_reload_ms,
                ewma_prompt_encode_ms: record.ewma_prompt_encode_ms,
                ewma_denoise_ms: record.ewma_denoise_ms,
                ewma_vae_ms: record.ewma_vae_ms,
                ewma_visual_decode_ms: record.ewma_visual_decode_ms,
                ewma_audio_decode_ms: record.ewma_audio_decode_ms,
                ewma_mux_ms: record.ewma_mux_ms,
                ewma_upscale_ms: record.ewma_upscale_ms,
                ewma_identity_extract_ms: record.ewma_identity_extract_ms,
                vram_conservative_bytes: record.vram_high_water_bytes,
                host_conservative_bytes: record.host_high_water_bytes,
                failure_count: record.failure_count,
                invalidated_count: record.invalidated_count,
                last_outcome: parse_estimate_outcome(&record.last_outcome),
                last_fallback_reason: record.last_fallback_reason,
                last_invalidated_plan_reason: record.last_invalidated_plan_reason,
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
        model_family: bucket.key.model_family.clone(),
        model_fingerprint: bucket.key.model_fingerprint.clone(),
        work_kind: bucket.key.work_kind.clone(),
        shape_bucket: bucket.key.shape_bucket.clone(),
        execution_fingerprint: bucket.key.execution_fingerprint.clone(),
        sample_count: bucket.sample_count,
        ewma_total_ms: bucket.ewma_total_ms,
        ewma_runtime_ms: bucket.ewma_runtime_ms,
        ewma_load_ms: bucket.ewma_load_ms,
        ewma_warm_reload_ms: bucket.ewma_warm_reload_ms,
        ewma_prompt_encode_ms: bucket.ewma_prompt_encode_ms,
        ewma_denoise_ms: bucket.ewma_denoise_ms,
        ewma_vae_ms: bucket.ewma_vae_ms,
        ewma_visual_decode_ms: bucket.ewma_visual_decode_ms,
        ewma_audio_decode_ms: bucket.ewma_audio_decode_ms,
        ewma_mux_ms: bucket.ewma_mux_ms,
        ewma_upscale_ms: bucket.ewma_upscale_ms,
        ewma_identity_extract_ms: bucket.ewma_identity_extract_ms,
        vram_high_water_bytes: bucket.vram_conservative_bytes,
        host_high_water_bytes: bucket.host_conservative_bytes,
        failure_count: bucket.failure_count,
        invalidated_count: bucket.invalidated_count,
        last_outcome: snake_debug(bucket.last_outcome),
        last_fallback_reason: bucket.last_fallback_reason.clone(),
        last_invalidated_plan_reason: bucket.last_invalidated_plan_reason.clone(),
        last_observed_at: bucket.last_observed_at_unix_s,
    }
}

fn generation_estimate_key(
    state: &AppState,
    worker: &GpuWorker,
    request: &mold_core::GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
    execution_fingerprint: &str,
) -> EstimateKey {
    let model_family = state
        .config
        .try_read()
        .ok()
        .and_then(|config| crate::model_manager::family_for_model_sync(&request.model, &config))
        // Never split opaque cv:/hf: IDs. An unresolved ID is its own
        // collision-free family until authoritative metadata is available.
        .unwrap_or_else(|| request.model.clone());
    EstimateKey {
        device_class: device_class(worker),
        model_family,
        model_fingerprint: request.model.clone(),
        work_kind: "generation".into(),
        shape_bucket: generation_shape_bucket_with_projection(request, projection),
        execution_fingerprint: execution_fingerprint.to_string(),
    }
}

fn owner_estimate_key(
    worker: &GpuWorker,
    kind: mold_scheduler::WorkKind,
    fingerprint: &str,
    shape_bucket: &str,
    execution_fingerprint: &str,
) -> EstimateKey {
    owner_estimate_key_for_device(
        device_class(worker),
        kind,
        fingerprint,
        shape_bucket,
        execution_fingerprint,
    )
}

fn owner_estimate_key_for_device(
    device_class: String,
    kind: mold_scheduler::WorkKind,
    fingerprint: &str,
    shape_bucket: &str,
    execution_fingerprint: &str,
) -> EstimateKey {
    EstimateKey {
        device_class,
        model_family: fingerprint.to_string(),
        model_fingerprint: fingerprint.to_string(),
        work_kind: snake_debug(kind),
        shape_bucket: shape_bucket.into(),
        execution_fingerprint: execution_fingerprint.to_string(),
    }
}

fn parse_estimate_outcome(value: &str) -> mold_scheduler::EstimateOutcome {
    match value {
        "failure" => mold_scheduler::EstimateOutcome::Failure,
        "invalidated" => mold_scheduler::EstimateOutcome::Invalidated,
        _ => mold_scheduler::EstimateOutcome::Success,
    }
}

fn max_optional(current: Option<u64>, sample: Option<u64>) -> Option<u64> {
    match (current, sample) {
        (Some(current), Some(sample)) => Some(current.max(sample)),
        (current, None) => current,
        (None, sample) => sample,
    }
}

fn vram_sample_for_stable_device(
    snapshot: &mold_core::ResourceSnapshot,
    registry: &crate::device_registry::DeviceRegistry,
    device_id: &str,
) -> Option<u64> {
    let device = registry.discovered_device(device_id)?;
    let logical_ordinal = device.telemetry_ordinal?;
    snapshot
        .gpus
        .iter()
        .find(|gpu| gpu.backend == device.backend && gpu.ordinal == logical_ordinal)?
        .vram_used_by_mold
}

fn execution_fallback_reason(
    plan: &crate::execution_plan::ResolvedExecutionPlan,
) -> Option<String> {
    if plan.offload_mode != crate::execution_plan::OffloadMode::None {
        return Some("block_offload".into());
    }
    let cpu_roles = plan
        .components
        .iter()
        .filter(|(_, component)| {
            component.placement == crate::execution_plan::ResolvedComponentPlacement::Cpu
        })
        .map(|(role, _)| snake_debug(role.clone()))
        .collect::<Vec<_>>();
    (!cpu_roles.is_empty()).then(|| format!("cpu:{}", cpu_roles.join(",")))
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

/// Memory a scheduled work item commits, given what its kind actually does.
///
/// The learned estimator prices every kind the same way, from recorded
/// observations. That is wrong for work that *releases* memory: an unload
/// priced as a consumer is rejected by admission on exactly the full device it
/// was queued to empty, so a resident engine that has exhausted host RAM or
/// VRAM can never be evicted and the user's only recovery is killing the
/// server. Releasing work therefore commits nothing.
///
/// `mold_scheduler::WorkKind::releases_resources` enforces the same rule inside
/// the planner, which is the authority. This keeps the demand the coordinator
/// publishes consistent with the demand the planner admits, so the server-side
/// eligibility filters and the host-memory ledger agree with it too.
fn planned_memory_bytes(
    kind: mold_scheduler::WorkKind,
    vram_bytes: u64,
    host_bytes: u64,
) -> (u64, u64) {
    if kind.releases_resources() {
        (0, 0)
    } else {
        (vram_bytes, host_bytes)
    }
}

/// Memory floor implied by a failure-only `scheduler_estimates` row.
///
/// `EstimateStore::estimate` only promotes buckets with completion samples,
/// which is right for timing and wrong for memory: a row with
/// `sample_count = 0` and `last_outcome = Failure` records the high-water mark
/// of an attempt that *died*, so it is a lower bound on what the shape needs —
/// never evidence that the shape fits. The #641 host carried
/// `vram_high_water_bytes = 24,884,805,632` (96.6% of a 24 GB card) against
/// three failures and zero samples, and admission still said yes.
fn failure_only_vram_floor(estimates: &EstimateStore, key: &EstimateKey) -> u64 {
    estimates
        .exact(key)
        .filter(|bucket| {
            bucket.sample_count == 0
                && bucket.failure_count > 0
                && bucket.last_outcome == EstimateOutcome::Failure
        })
        .and_then(|bucket| bucket.vram_conservative_bytes)
        .unwrap_or(0)
}

/// Split `InsufficientVram` into terminal and transient halves.
///
/// Before #641 every insufficient-VRAM rejection was transient, which was
/// harmless while the LTX-2 estimate was far below the true peak. With an
/// honest estimate an impossible shape would otherwise sit in the queue
/// forever, re-resolving on every scheduler tick and never becoming feasible.
///
/// A peak above the largest *eligible* device's physical capacity can never
/// be satisfied by waiting; a peak that only exceeds what is currently free
/// is ordinary pressure and stays transient. An unknown capacity (`0`) stays
/// transient — never reject on missing evidence.
fn insufficient_vram_is_terminal(
    required_peak_bytes: u64,
    largest_eligible_total_vram_bytes: u64,
) -> bool {
    largest_eligible_total_vram_bytes > 0 && required_peak_bytes > largest_eligible_total_vram_bytes
}

fn largest_eligible_total_vram_bytes(
    eligible_device_ids: &[String],
    total_vram_bytes_by_device_id: &BTreeMap<String, u64>,
) -> u64 {
    let mut largest = 0;
    for device_id in eligible_device_ids {
        let Some(capacity) = total_vram_bytes_by_device_id
            .get(device_id)
            .copied()
            .filter(|capacity| *capacity > 0)
        else {
            return 0;
        };
        largest = largest.max(capacity);
    }
    largest
}

fn classify_generation_plan_failure(
    error: crate::execution_plan::ExecutionPlanError,
    total_vram_bytes_by_device_id: &BTreeMap<String, u64>,
) -> GenerationPlanFailure {
    match &error {
        crate::execution_plan::ExecutionPlanError::InsufficientVram {
            required_peak_bytes,
            eligible_device_ids,
            ..
        } => {
            let largest_eligible_total_vram_bytes = largest_eligible_total_vram_bytes(
                eligible_device_ids,
                total_vram_bytes_by_device_id,
            );
            if insufficient_vram_is_terminal(
                *required_peak_bytes,
                largest_eligible_total_vram_bytes,
            ) {
                GenerationPlanFailure::Terminal(error)
            } else {
                GenerationPlanFailure::Transient(error.to_string())
            }
        }
        crate::execution_plan::ExecutionPlanError::PreparedInputsStale(_) => {
            GenerationPlanFailure::StalePreparation(error.to_string())
        }
        _ => GenerationPlanFailure::Terminal(error),
    }
}

fn placement_preview_disposition_for_plan_failure(
    failure: &GenerationPlanFailure,
) -> (bool, &'static str) {
    match failure {
        GenerationPlanFailure::Terminal(_) => (true, "infeasible"),
        GenerationPlanFailure::Transient(_) => (true, "temporarily_unavailable"),
        // A queued generation already resets stale preparation and runs the
        // admission preparer again. Publishing this as authoritative transient
        // pressure prevents clients from reaching that recovery path. Decline
        // preview authority instead: compatible requests may enter normal
        // admission, where cache reclaim and fresh evidence are both owned.
        GenerationPlanFailure::StalePreparation(_) => (false, "unsupported"),
    }
}

/// The forward passes one denoise step of this request performs, as a
/// permille multiplier over the ordinary single-forward cost.
///
/// PuLID true CFG runs a SECOND transformer forward on every step from
/// `cfg_start_step` onwards (`PuLID/flux/sampling.py:136-149`), and the earlier
/// steps run one. So the denoise cost scales by
/// `1 + (steps - cfg_start_step) / steps` — the branched fraction of the run,
/// never a flat 2x, because a request that starts the branch late genuinely
/// pays less.
///
/// Returns exactly `1_000` for every request that does not engage the branch,
/// which keeps an inert scale and a zero identity weight byte-identical in the
/// estimate rather than merely close.
#[cfg(test)]
fn denoise_forward_multiplier_permille(request: &mold_core::GenerateRequest) -> u64 {
    denoise_forward_multiplier_permille_with_projection(request, None)
}

fn denoise_forward_multiplier_permille_with_projection(
    request: &mold_core::GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> u64 {
    let uses_true_cfg = mold_core::identity::request_uses_true_cfg_with_identity_presence(
        request,
        mold_core::identity::request_carries_identity_photo(request)
            || projection.is_some_and(|projection| projection.identity_present),
    );
    if !uses_true_cfg {
        return 1_000;
    }
    let steps = u64::from(request.steps).max(1);
    let start = u64::from(mold_core::identity::effective_cfg_start_step(request)).min(steps);
    let branched = steps.saturating_sub(start);
    1_000 + branched.saturating_mul(1_000) / steps
}

#[cfg(test)]
fn generation_shape_bucket(request: &mold_core::GenerateRequest) -> String {
    generation_shape_bucket_with_projection(request, None)
}

fn generation_shape_bucket_with_projection(
    request: &mold_core::GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> String {
    // The true-CFG arm is part of the bucket KEY, not just the static estimate:
    // a branched run and an ordinary one of the same geometry take roughly
    // twice the denoise time, and letting their samples share a bucket teaches
    // the learned model an average that is wrong for both. The multiplier is
    // the discriminator rather than a bare flag, so a run that starts the
    // branch at step 1 and one that starts it at step 15 stay separate too.
    //
    // The suffix is APPENDED ONLY for an engaged branch, and that is not a
    // tidiness choice. `scheduler_estimates` is persisted and keyed on this
    // string, and it carries more than learned timings — the failure-only VRAM
    // floors live there too. Renaming every ordinary bucket on upgrade would
    // strand all of it, so a shape already known to OOM would be admitted again
    // until it failed a second time. An unbranched request must therefore
    // produce the byte-identical legacy key, which
    // `an_ordinary_request_keeps_the_legacy_bucket_key` pins against a literal.
    let source = request.source_image.is_some()
        || request.source_video.is_some()
        || request.source_video_path.is_some()
        || projection.is_some_and(|projection| {
            projection.source_image
                || projection.source_video_inline
                || projection.source_video_path
        });
    let edit_count = request.edit_images.as_ref().map_or(0, Vec::len)
        + projection.map_or(0, |projection| projection.edit_image_count());
    let base = format!(
        "{}x{}:s{}:f{}:fps{}:a{}:src{}:edit{}:lora{}:b{}",
        request.width,
        request.height,
        request.steps,
        request.frames.unwrap_or(1),
        request.fps.unwrap_or(0),
        u8::from(request.enable_audio == Some(true)),
        u8::from(source),
        edit_count,
        u8::from(request.lora.is_some() || request.loras.as_ref().is_some_and(|v| !v.is_empty())),
        request.batch_size,
    );
    match denoise_forward_multiplier_permille_with_projection(request, projection) {
        1_000 => base,
        multiplier => format!("{base}:cfg{multiplier}"),
    }
}

#[cfg(test)]
fn static_generation_time_ms(request: &mold_core::GenerateRequest) -> u64 {
    static_generation_time_ms_with_projection(request, None)
}

fn static_generation_time_ms_with_projection(
    request: &mold_core::GenerateRequest,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> u64 {
    let megapixels = (u64::from(request.width) * u64::from(request.height)).div_ceil(1_000_000);
    let frames = u64::from(request.frames.unwrap_or(1));
    // The 1_000 ms term is fixed overhead and is deliberately outside the
    // multiplier: true CFG doubles denoise steps, not setup.
    let denoise_ms = megapixels
        .max(1)
        .saturating_mul(u64::from(request.steps).max(1))
        .saturating_mul(frames)
        .saturating_mul(125)
        .saturating_mul(denoise_forward_multiplier_permille_with_projection(
            request, projection,
        ))
        / 1_000;
    1_000u64.saturating_add(denoise_ms)
}

/// `host_bytes` must be the plan's `admission_host_demand_bytes`, never its raw
/// `predicted_host_increment_bytes`: on Metal the host claim already rides
/// `admission_vram_demand_bytes` against the one unified pool, and charging it
/// again to the host ledger is the #1038 double-count.
fn static_generation_estimate(
    request: &mold_core::GenerateRequest,
    vram_bytes: u64,
    host_bytes: u64,
) -> StaticEstimate {
    static_generation_estimate_with_projection(request, vram_bytes, host_bytes, None)
}

fn static_generation_estimate_with_projection(
    request: &mold_core::GenerateRequest,
    vram_bytes: u64,
    host_bytes: u64,
    projection: Option<&crate::queue_media_store::QueueMediaProjection>,
) -> StaticEstimate {
    let timing = mold_scheduler::static_timing_for(mold_scheduler::WorkKind::Generation);
    let predicted_run_ms =
        static_generation_time_ms_with_projection(request, projection).max(timing.predicted_run_ms);
    StaticEstimate {
        total_ms: timing.cold_setup_ms.saturating_add(predicted_run_ms),
        cold_setup_ms: timing.cold_setup_ms,
        warm_setup_ms: timing.warm_setup_ms,
        predicted_run_ms,
        vram_bytes,
        host_bytes,
    }
}

fn generation_uses_frozen_device_capacity(
    backend: mold_core::GpuBackend,
    model_family: &str,
) -> bool {
    backend == mold_core::GpuBackend::Metal && model_family == mold_core::minimax_h3::FAMILY
}

fn timing_with_static_floors(
    estimate: ResolvedEstimate,
    static_estimate: StaticEstimate,
) -> (u64, u64, u64) {
    (
        estimate.cold_setup_ms.max(static_estimate.cold_setup_ms),
        estimate.warm_setup_ms.max(static_estimate.warm_setup_ms),
        estimate
            .predicted_run_ms
            .max(static_estimate.predicted_run_ms),
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

pub(crate) fn upscale_utility_candidates(
    model_name: &str,
    weights: &mold_inference::upscaler::ResolvedUpscaleArtifact,
    artifact_root: Option<&std::path::Path>,
    placements: impl IntoIterator<Item = UtilityPlacement>,
) -> Vec<UtilityExecutionPlan> {
    placements
        .into_iter()
        .map(|placement| {
            UtilityExecutionPlan::Upscale(
                mold_inference::upscaler::resolve_upscale_execution_plan_from_artifact(
                    model_name,
                    weights.clone(),
                    artifact_root.map(std::path::Path::to_path_buf),
                    match placement {
                        UtilityPlacement::Cpu => {
                            mold_inference::upscaler::ExactUpscalePlacement::Cpu
                        }
                        UtilityPlacement::Device { backend, ordinal } => {
                            mold_inference::upscaler::ExactUpscalePlacement::Device {
                                backend,
                                ordinal,
                            }
                        }
                    },
                ),
            )
        })
        .collect()
}

#[cfg(feature = "expand")]
pub(crate) fn prompt_expansion_candidates(
    state: &AppState,
    config: &mold_core::Config,
    expand_model: Option<&str>,
) -> Result<Vec<UtilityExecutionPlan>, String> {
    let artifacts = mold_inference::expand::resolve_local_expand_artifacts(config, expand_model)
        .map_err(|error| error.to_string())?;
    let placements = std::iter::once(UtilityPlacement::Cpu).chain(
        state
            .gpu_pool
            .schedulable_workers()
            .into_iter()
            .map(|worker| UtilityPlacement::Device {
                backend: worker.gpu.backend,
                ordinal: worker.gpu.ordinal,
            }),
    );
    Ok(placements
        .map(|placement| {
            UtilityExecutionPlan::PromptExpansion(
                mold_inference::expand::resolve_expand_execution_plan_from_artifacts(
                    artifacts.clone(),
                    match placement {
                        UtilityPlacement::Cpu => mold_inference::expand::ExactExpandPlacement::Cpu,
                        UtilityPlacement::Device { backend, ordinal } => {
                            mold_inference::expand::ExactExpandPlacement::Device {
                                backend,
                                ordinal,
                            }
                        }
                    },
                ),
            )
        })
        .collect())
}

pub(crate) fn upscale_candidates(
    state: &AppState,
    model_name: &str,
    weights_path: &std::path::Path,
    artifact_root: Option<&std::path::Path>,
) -> Result<Vec<UtilityExecutionPlan>, String> {
    let base = mold_inference::upscaler::resolve_upscale_execution_plan(
        model_name,
        weights_path,
        artifact_root,
        mold_inference::upscaler::ExactUpscalePlacement::Cpu,
    )
    .map_err(|error| error.to_string())?;
    let placements = std::iter::once(UtilityPlacement::Cpu).chain(
        state
            .gpu_pool
            .schedulable_workers()
            .into_iter()
            .map(|worker| UtilityPlacement::Device {
                backend: worker.gpu.backend,
                ordinal: worker.gpu.ordinal,
            }),
    );
    Ok(upscale_utility_candidates(
        &base.model_name,
        &base.weights,
        base.artifact_root.as_deref(),
        placements,
    ))
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

/// Cache bytes recorded by Mold's owner thread are first-party evidence even
/// when the operating system cannot attribute aggregate process VRAM (Metal,
/// and CUDA telemetry fallbacks). When attribution is available it remains an
/// upper bound, preventing a stale cache counter from reclaiming bytes the
/// process sample says Mold does not own. This is planning evidence only:
/// dispatch still requires an idle owner lane, and the GPU worker retains its
/// final allocation-time predicted-peak validation.
pub(crate) fn reclaimable_model_cache_bytes(
    measured_cache_bytes: u64,
    sampled_mold_bytes: Option<u64>,
) -> u64 {
    sampled_mold_bytes.map_or(measured_cache_bytes, |mold_bytes| {
        measured_cache_bytes.min(mold_bytes)
    })
}

/// Effective capacity for serialized work on a device. While work is active,
/// the sampler's Mold-owned bytes belong to work that must finish before the
/// next lease can start, so those bytes are future-reclaimable.
/// Unknown attribution and memory owned by other processes remain excluded.
pub(crate) fn schedulable_available_vram_bytes(
    sampled_free_bytes: u64,
    reclaimable_cache_bytes: u64,
    sampled_mold_bytes: Option<u64>,
    has_active_work: bool,
    total_vram_bytes: u64,
) -> u64 {
    let immediate = effective_available_vram_bytes(
        sampled_free_bytes,
        reclaimable_cache_bytes,
        total_vram_bytes,
    );
    if !has_active_work {
        return immediate;
    }
    sampled_mold_bytes.map_or(immediate, |mold_bytes| {
        immediate
            .max(sampled_free_bytes.saturating_add(mold_bytes))
            .min(total_vram_bytes)
    })
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
mod true_cfg_estimate_tests {
    use super::*;

    #[test]
    fn scheduler_transport_and_retry_adapters_move_the_opaque_media_handle() {
        let source = include_str!("mod.rs");
        let to_gpu = source
            .find("fn gpu_job_from_generation(")
            .expect("generation-to-GPU adapter");
        let from_gpu = source
            .find("fn generation_and_prepared_from_gpu_job(")
            .expect("GPU retry adapter");
        assert!(source[to_gpu..from_gpu].contains("deferred_media: job.deferred_media"));
        let retry_end = source[from_gpu..]
            .find("\nfn generation_from_gpu_job(")
            .map(|offset| from_gpu + offset)
            .expect("retry adapter boundary");
        assert!(source[from_gpu..retry_end].contains("deferred_media: job.deferred_media"));
    }

    fn request() -> mold_core::GenerateRequest {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1,
        }))
        .expect("the minimal generate-request wire shape");
        request.id_image = Some(vec![0x89, 0x50, 0x4e, 0x47]);
        request
    }

    /// A request that does not engage the branch must be EXACTLY the ordinary
    /// single-forward cost — not merely close. Anything else would move every
    /// existing estimate the day this landed.
    #[test]
    fn an_unbranched_request_has_exactly_the_ordinary_multiplier() {
        let plain = request();
        assert_eq!(denoise_forward_multiplier_permille(&plain), 1_000);

        // An inert scale is inert here too.
        let mut inert = request();
        inert.true_cfg = Some(1.0);
        assert_eq!(denoise_forward_multiplier_permille(&inert), 1_000);

        // And a zero identity weight renders the plain print.
        let mut zero = request();
        zero.true_cfg = Some(2.5);
        zero.id_weight = Some(0.0);
        assert_eq!(denoise_forward_multiplier_permille(&zero), 1_000);
    }

    /// The multiplier is the BRANCHED FRACTION of the run, never a flat 2x: a
    /// request that starts the branch late genuinely runs fewer double steps.
    #[test]
    fn the_multiplier_is_the_branched_fraction_of_the_run() {
        let mut from_zero = request();
        from_zero.true_cfg = Some(2.0);
        from_zero.cfg_start_step = Some(0);
        assert_eq!(denoise_forward_multiplier_permille(&from_zero), 2_000);

        // The default start of 1 leaves 19 of 20 steps branched.
        let mut default_start = request();
        default_start.true_cfg = Some(2.0);
        assert_eq!(denoise_forward_multiplier_permille(&default_start), 1_950);

        let mut halfway = request();
        halfway.true_cfg = Some(2.0);
        halfway.cfg_start_step = Some(10);
        assert_eq!(denoise_forward_multiplier_permille(&halfway), 1_500);

        let mut last_step = request();
        last_step.true_cfg = Some(2.0);
        last_step.cfg_start_step = Some(19);
        assert_eq!(denoise_forward_multiplier_permille(&last_step), 1_050);
    }

    /// The learned model keys on the shape bucket, so a branched run and an
    /// ordinary one of the same geometry must NOT share a bucket — their
    /// samples would teach an average that is wrong for both, and the ETA a
    /// client renders (and the Auto host placement-preview picks) comes from
    /// exactly that number.
    #[test]
    fn the_shape_bucket_separates_branched_runs_from_ordinary_ones() {
        let plain = request();
        let mut branched = request();
        branched.true_cfg = Some(2.0);
        assert_ne!(
            generation_shape_bucket(&plain),
            generation_shape_bucket(&branched)
        );

        // And two branched runs that start the branch at different steps are
        // different amounts of work, so they are different buckets too.
        let mut later = request();
        later.true_cfg = Some(2.0);
        later.cfg_start_step = Some(10);
        assert_ne!(
            generation_shape_bucket(&branched),
            generation_shape_bucket(&later)
        );

        // An inert scale and a zero weight must land in the ORDINARY bucket
        // byte-for-byte, or their samples are stranded from the runs they are
        // identical to.
        let mut inert = request();
        inert.true_cfg = Some(1.0);
        assert_eq!(
            generation_shape_bucket(&plain),
            generation_shape_bucket(&inert)
        );
        let mut zero = request();
        zero.true_cfg = Some(2.5);
        zero.id_weight = Some(0.0);
        assert_eq!(
            generation_shape_bucket(&plain),
            generation_shape_bucket(&zero)
        );
    }

    #[test]
    fn authenticated_projection_matches_hydrated_shape_and_true_cfg_timing() {
        let mut hydrated = request();
        hydrated.true_cfg = Some(2.0);
        hydrated.source_image = Some(vec![1, 2, 3]);
        hydrated.edit_images = Some(vec![vec![4], vec![5]]);
        let mut sanitized = hydrated.clone();
        sanitized.id_image = None;
        sanitized.source_image = None;
        sanitized.edit_images = None;
        let projection = crate::queue_media_store::QueueMediaProjection {
            source_image: true,
            identity_present: true,
            identity_photograph_count: 1,
            edit_image_count: 2,
            edit_images: vec![
                crate::queue_media_store::ProjectedImageDimensions::UnreadableHeader,
                crate::queue_media_store::ProjectedImageDimensions::UnreadableHeader,
            ],
            ..Default::default()
        };

        assert_eq!(
            generation_shape_bucket_with_projection(&hydrated, None),
            generation_shape_bucket_with_projection(&sanitized, Some(&projection)),
        );
        assert_eq!(
            static_generation_time_ms_with_projection(&hydrated, None),
            static_generation_time_ms_with_projection(&sanitized, Some(&projection)),
        );
        assert_eq!(
            crate::gpu_pool::oom_shape_bucket_with_projection(&hydrated, None),
            crate::gpu_pool::oom_shape_bucket_with_projection(&sanitized, Some(&projection)),
        );

        let mut hydrated_path = request();
        hydrated_path.id_image = None;
        hydrated_path.source_video_path = Some("/private/video.mp4".into());
        let mut sanitized_path = hydrated_path.clone();
        sanitized_path.source_video_path = None;
        let path_projection = crate::queue_media_store::QueueMediaProjection {
            source_video_path: true,
            ..Default::default()
        };
        assert_eq!(
            generation_shape_bucket_with_projection(&hydrated_path, None),
            generation_shape_bucket_with_projection(&sanitized_path, Some(&path_projection)),
        );
    }

    #[test]
    fn qwen_edit_count_above_flux_dimension_slots_keeps_hydrated_projection_parity() {
        let mut hydrated = request();
        hydrated.model = "qwen-image-edit".into();
        hydrated.edit_images = Some((0_u8..5).map(|byte| vec![byte]).collect());
        let mut sanitized = hydrated.clone();
        sanitized.edit_images = None;
        let projection = crate::queue_media_store::QueueMediaProjection {
            edit_image_count: 5,
            edit_images: vec![
                crate::queue_media_store::ProjectedImageDimensions::UnreadableHeader;
                crate::queue_media_store::PROJECTED_EDIT_DIMENSION_SLOTS
            ],
            ..Default::default()
        };

        assert_eq!(
            generation_shape_bucket_with_projection(&hydrated, None),
            generation_shape_bucket_with_projection(&sanitized, Some(&projection)),
        );
    }

    /// `scheduler_estimates` is PERSISTED and keyed on this string, and it
    /// carries the failure-only VRAM floors as well as the learned timings. A
    /// suffix on every ordinary bucket would rename all of them on upgrade, so
    /// a shape already known to OOM would be admitted again until it failed
    /// once more. The literal is the current format, captured; changing it is
    /// a migration, not an edit.
    #[test]
    fn an_ordinary_request_keeps_the_legacy_bucket_key() {
        let plain = request();
        assert_eq!(
            generation_shape_bucket(&plain),
            "1024x1024:s20:f1:fps0:a0:src0:edit0:lora0:b1"
        );
        assert!(
            !generation_shape_bucket(&plain).contains("cfg"),
            "an unbranched request must carry no discriminator at all"
        );
    }

    /// The engaged form is the legacy key plus one suffix — an extension of the
    /// existing namespace, never a replacement for it.
    #[test]
    fn an_engaged_branch_extends_the_legacy_bucket_key() {
        let mut branched = request();
        branched.true_cfg = Some(2.0);
        branched.cfg_start_step = Some(0);
        assert_eq!(
            generation_shape_bucket(&branched),
            "1024x1024:s20:f1:fps0:a0:src0:edit0:lora0:b1:cfg2000"
        );
        assert!(
            generation_shape_bucket(&branched).starts_with(&generation_shape_bucket(&request()))
        );
    }

    /// The cold estimate is what a host with no learned samples answers with,
    /// which is exactly the case a first true-CFG render hits. It has to scale
    /// the denoise term and leave the fixed setup term alone.
    #[test]
    fn the_static_estimate_scales_the_denoise_term_only() {
        let plain = request();
        let mut branched = request();
        branched.true_cfg = Some(2.0);
        branched.cfg_start_step = Some(0);

        let plain_ms = static_generation_time_ms(&plain);
        let branched_ms = static_generation_time_ms(&branched);
        // 1_000 ms of fixed setup, then a doubled denoise term.
        assert_eq!(branched_ms - 1_000, 2 * (plain_ms - 1_000));

        let mut halfway = request();
        halfway.true_cfg = Some(2.0);
        halfway.cfg_start_step = Some(10);
        assert_eq!(
            static_generation_time_ms(&halfway) - 1_000,
            (plain_ms - 1_000) * 3 / 2
        );

        // Byte-identical for everything that does not engage the branch.
        let mut inert = request();
        inert.true_cfg = Some(1.0);
        assert_eq!(static_generation_time_ms(&inert), plain_ms);
    }

    /// Placement preview must read the SAME predicate — it drives the Auto
    /// host choice by predicted completion, and a branched render mispriced as
    /// ordinary would win a race it cannot actually win. It shares the estimate
    /// path rather than restating it, and this is the check that says so.
    #[test]
    fn the_placement_preview_estimate_path_is_the_branched_one() {
        let mut branched = request();
        branched.true_cfg = Some(2.0);
        branched.cfg_start_step = Some(0);

        // `placement_preview_dag_cancellable` builds its per-plan estimate from
        // exactly these two functions, so pinning them pins the preview.
        assert!(mold_core::identity::request_uses_true_cfg(&branched));
        assert!(generation_shape_bucket(&branched).ends_with(":cfg2000"));
        assert!(static_generation_time_ms(&branched) > static_generation_time_ms(&request()));

        // `static_generation_estimate` floors `predicted_run_ms` at
        // `WorkKind::Generation`'s 30 s static timing, which dominates every
        // small render — so the geometry here is deliberately large enough to
        // clear it, which is where the branch can actually move the answer.
        // Below the floor the shape bucket is what keeps the two apart, and it
        // does so from the first learned sample.
        let mut big = branched.clone();
        big.width = 2048;
        big.height = 2048;
        big.steps = 50;
        let mut big_plain = request();
        big_plain.width = 2048;
        big_plain.height = 2048;
        big_plain.steps = 50;

        let static_estimate = static_generation_estimate(&big, 1_000, 2_000);
        let plain_estimate = static_generation_estimate(&big_plain, 1_000, 2_000);
        assert!(
            static_estimate.predicted_run_ms > plain_estimate.predicted_run_ms,
            "{} must exceed {}",
            static_estimate.predicted_run_ms,
            plain_estimate.predicted_run_ms
        );
        assert_eq!(static_estimate.cold_setup_ms, plain_estimate.cold_setup_ms);
    }
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
            _request: crate::queue_media_runtime::ZeroizingGenerateRequest,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
            _context: crate::variant_dependencies::DependencyPreparationContext,
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

    #[test]
    fn restart_without_top_level_pin_preserves_v2_component_identity_and_cpu() {
        let (worker, _worker_rx) = test_worker(7);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(queue_tx),
            pool,
            1,
        );
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "restart placement",
            "model": "mock-model",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 1,
            "output_format": "png"
        }))
        .unwrap();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Cpu,
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(7),
                // This is a component's independently stable identity. The
                // absent top-level row pin must not erase it.
                vae: mold_core::DeviceRef::device(format!("cuda:{:032x}", 8)),
                clip_l: Some(mold_core::DeviceRef::Cpu),
                ..mold_core::AdvancedPlacement::default()
            }),
        });

        assert_eq!(
            generation_hard_ordinal(&state, "replayed", &request),
            Some(7)
        );
        crate::queue_journal::resolve_replay_affinity(&mut request, Some(7), None, |id| {
            (id == format!("cuda:{:032x}", 8)).then_some(7)
        });
        assert_eq!(
            generation_hard_ordinal(&state, "replayed", &request),
            Some(7)
        );
        let placement = request.placement.unwrap();
        assert_eq!(placement.text_encoders, mold_core::DeviceRef::Cpu);
        assert_eq!(
            placement.advanced.unwrap().clip_l,
            Some(mold_core::DeviceRef::Cpu)
        );
    }

    /// A worker whose device collapses VRAM and host RAM onto one pool.
    fn metal_test_worker(
        ordinal: usize,
    ) -> (
        Arc<GpuWorker>,
        std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
    ) {
        let (worker, rx) = test_worker(ordinal);
        let Ok(mut worker) = Arc::try_unwrap(worker) else {
            unreachable!("test_worker returns the only reference")
        };
        worker.gpu.backend = mold_core::GpuBackend::Metal;
        worker.gpu.stable_id = Some(format!("metal:{ordinal}"));
        worker.gpu.compute_capability = None;
        worker.gpu.raw_cuda_uuid = None;
        worker.gpu.device_kind = None;
        (Arc::new(worker), rx)
    }

    /// Metal must reserve nothing in the host ledger.
    ///
    /// Its host claim already rides `admission_vram_demand_bytes` against the
    /// one unified pool, so charging it a second time against a second sample
    /// of that pool — minus a safety floor the device gate does not pay — is
    /// the #1038 double-count. It was survivable while a reservation was
    /// discharged a second after dispatch; now that one stays charged for the
    /// whole lease, it would park real work.
    #[tokio::test]
    async fn metal_reserves_no_host_ram_while_cuda_reserves_its_increment() {
        for (backend, expected_reservation_bytes) in [
            (mold_core::GpuBackend::Metal, 0),
            (mold_core::GpuBackend::Cuda, MIN_TRANSIENT_HOST_RAM),
        ] {
            let (worker, worker_rx) = if backend == mold_core::GpuBackend::Metal {
                metal_test_worker(0)
            } else {
                test_worker(0)
            };
            let device_id = worker_device_id(&worker);
            let pool = Arc::new(GpuPool {
                workers: vec![worker].into(),
            });
            let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
            let queue = QueueHandle::new(ingress_tx);
            let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
            state.job_registry.register("job", "flux-dev:q4");
            let (job, _result) = fake_generation("job");
            queue.submit(job, 1).await.unwrap();
            let mut coordinator = Coordinator::with_preparer_and_memory(
                state,
                Arc::new(ImmediatePreparer),
                ample_memory(),
            );
            let mut immediate = false;
            coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
            for pending in coordinator.pending.values_mut() {
                pending.preparation = PreparationState::Ready;
            }
            coordinator.handle_worker_event(
                WorkerEvent::Ready {
                    device_id,
                    ordinal: 0,
                    owner_epoch: 1,
                    worker_generation: 1,
                },
                &mut immediate,
            );
            let _ = coordinator.dispatch_ready().await;
            assert_eq!(recv_grant(&worker_rx).id, "job", "{backend:?}");

            let reserved = coordinator
                .memory
                .reservations
                .get("job")
                .map(|reservation| reservation.bytes)
                .expect("a granted lease holds its reservation");
            assert_eq!(
                reserved, expected_reservation_bytes,
                "{backend:?} reserved {reserved} host bytes"
            );
        }
    }

    fn fake_generation_with_progress(
        id: &str,
    ) -> (
        GenerationJob,
        tokio::sync::mpsc::UnboundedReceiver<SseMessage>,
        tokio::sync::oneshot::Receiver<Result<crate::state::GenerationJobResult, String>>,
    ) {
        let (job, result) = fake_generation(id);
        let (progress_tx, progress_rx) = tokio::sync::mpsc::unbounded_channel();
        (
            GenerationJob {
                progress_tx: Some(progress_tx),
                ..job
            },
            progress_rx,
            result,
        )
    }

    fn drain_queued_positions(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<SseMessage>,
    ) -> Vec<usize> {
        let mut positions = Vec::new();
        while let Ok(message) = rx.try_recv() {
            if let SseMessage::Progress(mold_core::SseProgressEvent::Queued { position, .. }) =
                message
            {
                positions.push(position);
            }
        }
        positions
    }

    /// A queued client is told its place in line as the queue drains.
    ///
    /// The only position a client ever received was the submit-time depth in
    /// its first SSE event; after that the queue moved in silence. Positions
    /// are re-announced from the registry — the same order `GET /api/queue`
    /// reports — and only when a job's own place actually changed, so a
    /// reconcile tick over a still queue emits nothing.
    #[tokio::test]
    async fn a_draining_queue_reannounces_positions_only_when_they_change() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 8);
        let mut progress = Vec::new();
        let mut results = Vec::new();
        for id in ["a", "b", "c"] {
            state.job_registry.register(id, "flux-dev:q4");
            let (job, progress_rx, result) = fake_generation_with_progress(id);
            progress.push(progress_rx);
            results.push(result);
            queue.submit(job, 8).await.unwrap();
        }
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        for _ in 0..3 {
            coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        }

        // First observation seeds each job's known place without re-telling a
        // client what its submit-time event already said.
        coordinator.reconcile_external_mutations(&mut immediate);
        for (index, rx) in progress.iter_mut().enumerate() {
            assert!(
                drain_queued_positions(rx).is_empty(),
                "job {index} must not be re-announced before anything moved"
            );
        }

        state.job_registry.remove("a");
        coordinator.reconcile_external_mutations(&mut immediate);
        let cancelled = match results[0].try_recv() {
            Ok(Err(error)) => error,
            Ok(Ok(_)) => panic!("registry removal unexpectedly completed queued work"),
            Err(error) => panic!("removed registry row did not settle queued work: {error}"),
        };
        assert!(cancelled.contains("generation job a was cancelled while queued"));
        assert_eq!(state.queue.pending(), 2);
        assert_eq!(
            drain_queued_positions(&mut progress[1]),
            vec![0],
            "b moved to the front and must be told once"
        );
        assert_eq!(drain_queued_positions(&mut progress[2]), vec![1]);

        // A tick over an unchanged queue is silent.
        coordinator.reconcile_external_mutations(&mut immediate);
        for (index, rx) in progress.iter_mut().enumerate() {
            assert!(
                drain_queued_positions(rx).is_empty(),
                "job {index} must not be re-announced without a change"
            );
        }
        drop(results);
        coordinator.stop_preparations().await;
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

    fn recv_owner_grant_id(
        rx: &std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
    ) -> String {
        match rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker grant")
        {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant.work.id().to_string(),
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain command"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown command"),
        }
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
                durable_queue_rank: None,
                request,
                deferred_media: None,
                resolved_references: None,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx,
                output_dir: None,
                batch_child: None,
                journal: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_ingress_grant: None,
            },
            result_rx,
        )
    }

    #[tokio::test]
    async fn deferred_preparation_failure_parks_durable_work_for_retry() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("gallery");
        std::fs::create_dir_all(&output).unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(tx);
        let mut state = AppState::for_tests();
        state.queue = queue.clone();
        state.queue_capacity = 1;
        state.metadata_db = db.clone();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db,
            Some(root.path()),
            "preparation-hold-test",
        ));

        let (mut job, mut result) = fake_generation("preparation-failed");
        job.output_dir = Some(output.clone());
        job.journal = state
            .queue_journal
            .clone()
            .record(crate::queue_journal::JournalAdmission {
                id: &job.id,
                request: &job.request,
                output_dir: Some(&output),
                target_gpu: None,
                target_device_id: None,
                completion_payload: SseCompletionPayload::Full,
                batch_child: false,
                carries_reference_authority: false,
            });
        state
            .job_registry
            .register(&job.id, job.request.model.clone());
        queue.submit(job, 1).await.unwrap();
        let job = rx.recv().await.unwrap();

        hold_preparation_failure(&state, job, "dependency unavailable".to_string());

        let outcome = result.try_recv().unwrap();
        assert!(matches!(outcome, Err(ref error) if error == "dependency unavailable"));
        assert_eq!(queue.pending(), 0);
        assert!(state.job_registry.entry("preparation-failed").is_none());
        let page = state.queue_journal.projection_page(None, 1).unwrap();
        assert_eq!(page.rows.len(), 1);
        assert_eq!(
            page.rows[0].state,
            mold_db::generation_queue::QueueRowState::Held
        );
        assert_eq!(
            page.rows[0].held_reason.as_deref(),
            Some("dependency unavailable")
        );
        assert!(page.rows[0].retryable);
    }

    #[test]
    fn replan_timer_uses_configured_sliding_delay_and_maximum() {
        let start = Instant::now();
        let settings = mold_core::config::SchedulerSettings {
            replan_debounce_ms: 700,
            replan_max_delay_ms: 1_900,
            warm_wait_max_ms: 0,
        };
        let mut window = ReplanWindow::new(settings);
        window.mark_dirty(start, 1);
        assert_eq!(window.deadline(), Some(start + Duration::from_millis(700)));
        window.mark_dirty(start + Duration::from_millis(600), 2);
        assert_eq!(
            window.deadline(),
            Some(start + Duration::from_millis(1_300))
        );
        window.mark_dirty(start + Duration::from_millis(1_800), 3);
        assert_eq!(
            window.deadline(),
            Some(start + Duration::from_millis(1_900))
        );
        assert!(!window.due(start + Duration::from_millis(1_899)));
        assert!(window.due(start + Duration::from_millis(1_900)));
        window.clear_through(2);
        assert!(
            window.deadline().is_some(),
            "stale plan must not clear timer"
        );
        window.clear_through(3);
        assert!(window.deadline().is_none());
    }

    #[tokio::test]
    async fn debounced_replan_settles_a_queue_drained_by_immediate_admission() {
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
        coordinator.state_version = 1;
        coordinator
            .dirty
            .mark_dirty(Instant::now(), coordinator.state_version);

        coordinator.dispatch_debounced_replan().await;

        assert!(
            coordinator.dirty.deadline().is_none(),
            "an empty queue must not retry the elapsed debounce on every tick"
        );
    }

    #[test]
    fn immediate_admission_skips_global_optimizer_until_debounced_pass() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let mut config = mold_core::Config::default();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let settings = mold_db::Settings::new(&db);
        settings
            .set_int(mold_db::settings::SCHEDULER_REPLAN_DEBOUNCE_MS, 10)
            .unwrap();
        settings
            .set_int(mold_db::settings::SCHEDULER_REPLAN_MAX_DELAY_MS, 20)
            .unwrap();
        settings
            .set_int(mold_db::settings::SCHEDULER_WARM_WAIT_MAX_MS, 0)
            .unwrap();
        mold_db::config_sync::hydrate_config_from_db(&db, &mut config).unwrap();
        let state = AppState::empty(config, QueueHandle::new(ingress_tx), pool, 1);
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let (snapshot, _) = coordinator.planner_snapshot(&BTreeMap::new());

        assert_eq!(
            coordinator
                .admission_planner
                .plan(&snapshot)
                .unwrap()
                .optimizer_state,
            mold_scheduler::OptimizerState::WatchdogFallback
        );
        assert_ne!(
            coordinator.planner.plan(&snapshot).unwrap().optimizer_state,
            mold_scheduler::OptimizerState::WatchdogFallback
        );
        assert_eq!(coordinator.dirty.debounce, Duration::from_millis(10));
        assert_eq!(coordinator.dirty.max_delay, Duration::from_millis(20));

        let no_warm_hold = coordinator
            .admission_planner
            .plan(&PlannerSnapshot::new(
                1,
                1,
                1_000,
                8 << 30,
                vec![
                    DeviceSnapshot::idle("cold", 24 << 30),
                    DeviceSnapshot::busy("warm", 24 << 30, 1_500).with_warm("exec"),
                ],
                vec![WorkSnapshot::new(
                    "job",
                    0,
                    vec![
                        CandidatePlacement::new("cold", "exec", 0).with_timing(1_000, 0, 1_000),
                        CandidatePlacement::new("warm", "exec", 0).with_timing(1_000, 0, 100),
                    ],
                )],
            ))
            .unwrap();
        assert_eq!(
            no_warm_hold.immediate_leases[0].device_id.as_str(),
            "cold",
            "persisted warm_wait_max_ms=0 must reach both admission and optimize planners"
        );
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
                prepared: Box::new(PreparedGeneration {
                    expanded_prompt: Some("expanded prompt".to_string()),
                    resolved_seed: None,
                    execution_inputs: None,
                }),
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
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
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
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new("work-a", 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
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
    fn live_owner_headroom_subtracts_peer_lease_after_fresh_pressure() {
        let mut ledger = unsampled_memory(48 << 30, 33 << 30);
        let initial = ledger.begin_collection();
        ledger.publish_sample(initial, 48 << 30, 33 << 30);
        for work_id in ["h3-a", "h3-b"] {
            ledger.reservations.insert(
                work_id.to_string(),
                HostReservation {
                    bytes: 10 << 30,
                    state: ReservationState::Reserved,
                    charge_until_release: false,
                },
            );
        }
        assert_eq!(ledger.headroom_for_reserved_work("h3-a"), Some(15 << 30));
        assert_eq!(ledger.headroom_for_reserved_work("h3-b"), Some(15 << 30));

        let pressured = ledger.begin_collection();
        ledger.publish_sample(pressured, 48 << 30, 23 << 30);
        assert_eq!(
            ledger.headroom_for_reserved_work("h3-a"),
            Some(5 << 30),
            "the owner may restore its own reservation but must keep the peer charged"
        );
        assert_eq!(ledger.headroom_for_reserved_work("h3-b"), Some(5 << 30));

        ledger.commit("h3-b");
        let after_premature_commit = ledger.begin_collection();
        ledger.publish_sample(after_premature_commit, 48 << 30, 23 << 30);
        assert_eq!(
            ledger.reservations["h3-b"].state,
            ReservationState::ReflectedBySample
        );
        assert_eq!(
            ledger.headroom_for_reserved_work("h3-a"),
            Some(5 << 30),
            "a pre-allocation commit notification cannot make the peer reservation spendable"
        );
    }

    #[test]
    fn h3_reservation_stays_charged_until_release_and_blocks_ordinary_work() {
        let mut ledger = unsampled_memory(48 << 30, 33 << 30);
        let initial = ledger.begin_collection();
        ledger.publish_sample(initial, 48 << 30, 33 << 30);
        ledger.reservations.insert(
            "h3".to_string(),
            HostReservation {
                bytes: 10 << 30,
                state: ReservationState::Reserved,
                charge_until_release: true,
            },
        );

        ledger.commit("h3");
        let after_commit = ledger.begin_collection();
        ledger.publish_sample(after_commit, 48 << 30, 23 << 30);
        assert!(matches!(
            ledger.reservations["h3"].state,
            ReservationState::CommittedAfterSample { .. }
        ));
        assert_eq!(ledger.headroom_bytes(), 5 << 30);

        let mut snapshot = PlannerSnapshot::new(
            1,
            1,
            0,
            ledger.headroom_bytes(),
            vec![DeviceSnapshot::idle("gpu-b", 24 << 30)],
            vec![WorkSnapshot::new(
                "ordinary",
                0,
                vec![CandidatePlacement::new("gpu-b", "model", 6 << 30)],
            )],
        );
        snapshot.host_memory = ledger.snapshot();
        let plan = Planner::default().plan(&snapshot).unwrap();
        assert!(plan.immediate_leases.is_empty());
        assert_eq!(
            plan.blocked_reason(&WorkId::new("ordinary")),
            Some(&BlockedReason::InsufficientHostRam)
        );

        ledger.release("h3");
        assert_eq!(ledger.headroom_bytes(), 15 << 30);
    }

    #[test]
    fn reserved_work_stays_charged_until_release_for_every_family() {
        // Every worker announces AllocationCommitted before its model finishes
        // loading, so no sample can prove the frozen increment landed. H3 was
        // only the first family to expose it; an LTX-2 lease whose Gemma
        // encoder loads inside generate() is the same shape.
        for work_id in ["h3", "ltx2"] {
            let mut ledger = unsampled_memory(48 << 30, 33 << 30);
            let initial = ledger.begin_collection();
            ledger.publish_sample(initial, 48 << 30, 33 << 30);
            let planner = Planner::default();
            let mut snapshot = PlannerSnapshot::new(
                1,
                1,
                0,
                ledger.headroom_bytes(),
                vec![DeviceSnapshot::idle("gpu-a", 24 << 30)],
                vec![WorkSnapshot::new(
                    work_id,
                    0,
                    vec![CandidatePlacement::new("gpu-a", "model", 10 << 30)],
                )],
            );
            snapshot.host_memory = ledger.snapshot();
            let plan = planner.plan(&snapshot).expect("single-lease plan");
            ledger
                .try_reserve(&plan, 1, 1)
                .expect("reservation fits the sampled headroom");

            // The commit lands before a single weight byte is allocated.
            ledger.commit(work_id);
            let after_commit = ledger.begin_collection();
            ledger.publish_sample(after_commit, 48 << 30, 33 << 30);
            assert!(
                matches!(
                    ledger.reservations[work_id].state,
                    ReservationState::CommittedAfterSample { .. }
                ),
                "{work_id}: a sample cannot absorb an allocation that has not happened"
            );
            assert_eq!(
                ledger.headroom_bytes(),
                15 << 30,
                "{work_id}: the reservation must stay charged until the lease settles"
            );

            let mut snapshot = PlannerSnapshot::new(
                1,
                1,
                0,
                ledger.headroom_bytes(),
                vec![DeviceSnapshot::idle("gpu-b", 24 << 30)],
                vec![WorkSnapshot::new(
                    "peer",
                    0,
                    vec![CandidatePlacement::new("gpu-b", "model", 20 << 30)],
                )],
            );
            snapshot.host_memory = ledger.snapshot();
            let plan = Planner::default().plan(&snapshot).expect("peer plan");
            assert!(plan.immediate_leases.is_empty(), "{work_id}");
            assert_eq!(
                plan.blocked_reason(&WorkId::new("peer")),
                Some(&BlockedReason::InsufficientHostRam),
                "{work_id}: a peer must park rather than double-spend the reservation"
            );

            ledger.release(work_id);
            assert_eq!(ledger.headroom_bytes(), 25 << 30, "{work_id}");
        }
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
                charge_until_release: false,
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
                charge_until_release: false,
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
            mold_core::config::SchedulerSettings::default(),
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

    /// Publish one CUDA device whose sampled free VRAM is exactly `free_bytes`.
    ///
    /// Coordinator tests resolve synthetic execution plans against that number,
    /// so it is the single knob that turns a pending generation between
    /// "planned" and "zero candidates" without touching the planner.
    fn publish_free_vram(state: &AppState, free_bytes: u64) {
        publish_free_vram_for_lanes(state, &[(mold_core::GpuBackend::Cuda, free_bytes)]);
    }

    fn publish_free_vram_for_lanes(state: &AppState, lanes: &[(mold_core::GpuBackend, u64)]) {
        const TOTAL: u64 = 24 << 30;
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".into(),
            timestamp: 1,
            gpus: lanes
                .iter()
                .enumerate()
                .map(|(ordinal, (backend, free_bytes))| mold_core::GpuSnapshot {
                    ordinal,
                    name: format!("gpu-{ordinal}"),
                    backend: *backend,
                    vram_total: TOTAL,
                    vram_used: TOTAL.saturating_sub(*free_bytes),
                    vram_used_by_mold: Some(0),
                    vram_used_by_other: Some(TOTAL.saturating_sub(*free_bytes)),
                    gpu_utilization: Some(0),
                })
                .collect(),
            system_ram: mold_core::RamSnapshot {
                total: 128 << 30,
                used: 1 << 30,
                available: None,
                used_by_mold: 0,
                used_by_other: 1 << 30,
            },
            cpu: None,
        });
    }

    /// A ready coordinator holding exactly one queued generation.
    async fn unschedulable_test_coordinator(
        free_vram_bytes: u64,
    ) -> (
        Coordinator,
        std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
        tokio::sync::oneshot::Receiver<Result<crate::state::GenerationJobResult, String>>,
    ) {
        let (worker, worker_rx) = test_worker(0);
        let device_id = worker_device_id(&worker);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
        state.job_registry.register("stranded", "flux-dev:q4");
        let (job, result_rx) = fake_generation("stranded");
        queue.submit(job, 1).await.unwrap();
        publish_free_vram(&state, free_vram_bytes);
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        for pending in coordinator.pending.values_mut() {
            pending.preparation = PreparationState::Ready;
        }
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id,
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        (coordinator, worker_rx, result_rx)
    }

    async fn pressured_test_coordinator(
        backend: mold_core::GpuBackend,
        lane_count: usize,
        free_vram_bytes: u64,
    ) -> (
        Coordinator,
        Vec<std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>>,
        tokio::sync::oneshot::Receiver<Result<crate::state::GenerationJobResult, String>>,
    ) {
        let mut workers = Vec::new();
        let mut worker_rxs = Vec::new();
        for ordinal in 0..lane_count {
            let (worker, worker_rx) = if backend == mold_core::GpuBackend::Metal {
                metal_test_worker(ordinal)
            } else {
                test_worker(ordinal)
            };
            workers.push(worker);
            worker_rxs.push(worker_rx);
        }
        let pool = Arc::new(GpuPool {
            workers: workers.clone().into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
        state.job_registry.register("pressured", "flux-dev:q4");
        let (mut job, result_rx) = fake_generation("pressured");
        job.request.model = "flux-dev:q4".to_string();
        queue.submit(job, 1).await.unwrap();
        publish_free_vram_for_lanes(&state, &vec![(backend, free_vram_bytes); lane_count]);
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("pressured")
            .expect("queued")
            .preparation = PreparationState::Ready;
        for worker in workers {
            coordinator.handle_worker_event(
                WorkerEvent::Ready {
                    device_id: worker_device_id(&worker),
                    ordinal: worker.gpu.ordinal,
                    owner_epoch: 1,
                    worker_generation: 1,
                },
                &mut immediate,
            );
        }
        (coordinator, worker_rxs, result_rx)
    }

    /// The recovery half of #1272: a shortage that clears must not need a
    /// resubmission. The scheduler re-resolves plans on every planning turn, so
    /// the job dispatches on the first turn after capacity returns — nothing
    /// about the settlement below may make waiting terminal on its own.
    #[tokio::test]
    async fn a_transiently_unschedulable_generation_dispatches_once_capacity_returns() {
        // Below `estimate_model_vram("flux-dev:q4")`, so no plan resolves.
        let (mut coordinator, worker_rx, _result_rx) =
            unschedulable_test_coordinator(2 << 30).await;

        let _ = coordinator.dispatch_ready().await;
        assert!(
            worker_rx.try_recv().is_err(),
            "a job with no execution plan must not be transported"
        );
        assert!(
            coordinator.pending.contains_key("stranded"),
            "a transient shortage must keep the job queued, not refuse it"
        );
        assert!(
            coordinator.pending["stranded"]
                .unschedulable_since_ms
                .is_none(),
            "typed transient pressure must never start a terminal settlement timer"
        );

        publish_free_vram(&coordinator.state, 24 << 30);
        let _ = coordinator.dispatch_ready().await;

        assert_eq!(recv_grant(&worker_rx).id, "stranded");
    }

    /// Physical capacity, rather than elapsed queue time, is the terminal
    /// boundary. This covers Metal's one unified lane and the scheduler's
    /// ordinary N-lane CUDA authority with 24 GiB physical VRAM but 22 GiB held
    /// by another process on every lane.
    #[tokio::test]
    async fn external_vram_pressure_remains_queued_after_idle_grace_then_dispatches() {
        for (label, backend, lane_count) in [
            ("Metal single lane", mold_core::GpuBackend::Metal, 1),
            ("CUDA N lane", mold_core::GpuBackend::Cuda, 2),
        ] {
            let (mut coordinator, worker_rxs, mut result_rx) =
                pressured_test_coordinator(backend, lane_count, 2 << 30).await;

            let failure = coordinator
                .generation_plans(&coordinator.pending["pressured"])
                .expect_err("22 GiB of external pressure must prevent a plan");
            assert!(
                matches!(failure, GenerationPlanFailure::Transient(_)),
                "{label}: a peak that fits physical VRAM is transient"
            );

            // Collapse the existing grace instead of sleeping through it.
            // Typed pressure must remain retryable even after that boundary.
            coordinator.unschedulable_idle_grace_ms = 0;
            for _ in 0..2 {
                let _ = coordinator.dispatch_ready().await;
            }
            assert!(
                coordinator.pending.contains_key("pressured"),
                "{label}: elapsed pressure must not terminalize durable work"
            );
            assert!(
                result_rx.try_recv().is_err(),
                "{label}: the client must remain attached to queued work"
            );
            assert!(
                worker_rxs.iter().all(|rx| rx.try_recv().is_err()),
                "{label}: pressure must prevent transport"
            );

            let unchanged_version = coordinator.state_version;
            let mut unchanged_immediate = false;
            coordinator.reconcile_resource_capacity(&mut unchanged_immediate);
            assert!(
                !unchanged_immediate,
                "{label}: an unchanged telemetry tick must not spin planning"
            );
            assert_eq!(coordinator.state_version, unchanged_version, "{label}");

            publish_free_vram_for_lanes(&coordinator.state, &vec![(backend, 24 << 30); lane_count]);
            let mut immediate = false;
            coordinator.reconcile_resource_capacity(&mut immediate);
            assert!(immediate, "{label}: changed capacity must wake planning");
            let _ = coordinator
                .dispatch_ready_with(PlanningPass::Admission)
                .await;

            let grants = worker_rxs
                .iter()
                .filter_map(|rx| match rx.try_recv() {
                    Ok(crate::gpu_pool::GpuWorkerCommand::Grant(grant)) => match grant.work {
                        OwnerWork::Generation(job) => Some(*job),
                        work => panic!("{label}: expected generation grant, got {:?}", work.kind()),
                    },
                    Ok(crate::gpu_pool::GpuWorkerCommand::Drain) => {
                        panic!("{label}: unexpected drain command")
                    }
                    Ok(crate::gpu_pool::GpuWorkerCommand::Shutdown) => {
                        panic!("{label}: unexpected shutdown command")
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        panic!("{label}: worker command channel disconnected")
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(grants.len(), 1, "{label}: one lane receives the job");
            assert_eq!(grants[0].id, "pressured", "{label}");
            assert!(
                !coordinator.pending.contains_key("pressured"),
                "{label}: cleared pressure restores dispatch eligibility"
            );
        }
    }

    /// Waiting behind real work is not the same failure. A job queued while
    /// something holds a lease is waiting for a resource that is coming back,
    /// so the idle settlement must not touch it however long that takes.
    #[tokio::test]
    async fn a_generation_waiting_behind_a_lease_is_never_settled_as_unschedulable() {
        let (mut coordinator, _worker_rx, mut result_rx) =
            unschedulable_test_coordinator(24 << 30).await;

        // Grant the queued job, which installs an active lease.
        let _ = coordinator.dispatch_ready().await;
        assert!(!coordinator.leases.is_empty(), "the device is leased");

        // A second job arrives that no device can plan while the first runs.
        coordinator
            .state
            .job_registry
            .register("behind", "flux-dev:q4");
        let (job, mut behind_rx) = fake_generation("behind");
        let mut immediate = false;
        coordinator.enqueue(job, &mut immediate);
        coordinator
            .pending
            .get_mut("behind")
            .expect("queued")
            .preparation = PreparationState::Ready;
        publish_free_vram(&coordinator.state, 2 << 30);
        // Zero grace: if the lease were not consulted, the very next turn would
        // settle this job. It must not.
        coordinator.unschedulable_idle_grace_ms = 0;

        for _ in 0..2 {
            let _ = coordinator.dispatch_ready().await;
        }

        assert!(
            coordinator.pending.contains_key("behind"),
            "work waiting on a running job must stay queued"
        );
        assert!(
            behind_rx.try_recv().is_err(),
            "work waiting on a running job is never refused"
        );
        assert!(
            result_rx.try_recv().is_err(),
            "the running job is untouched"
        );
    }

    /// The registry is the lifecycle authority across the non-atomic boundary
    /// between coordinator bookkeeping and owner-thread counters. A following
    /// job must remain queued if that authority still says another generation
    /// is running, even when the transport-local signals are momentarily clear.
    #[tokio::test]
    async fn a_generation_waiting_behind_registry_running_work_is_not_treated_as_idle() {
        let (mut coordinator, worker_rx, mut result_rx) =
            unschedulable_test_coordinator(2 << 30).await;
        coordinator
            .state
            .job_registry
            .register("active", "minimax-h3-fl2va:comfy-pruned-int8");
        coordinator
            .state
            .job_registry
            .mark_running("active", Some(0));
        coordinator
            .state
            .resources
            .publish(mold_core::ResourceSnapshot {
                hostname: "test".into(),
                timestamp: 2,
                gpus: vec![mold_core::GpuSnapshot {
                    ordinal: 0,
                    name: "gpu-0".into(),
                    backend: mold_core::GpuBackend::Cuda,
                    vram_total: 24 << 30,
                    vram_used: 22 << 30,
                    vram_used_by_mold: Some(22 << 30),
                    vram_used_by_other: Some(0),
                    gpu_utilization: Some(100),
                }],
                system_ram: mold_core::RamSnapshot {
                    total: 128 << 30,
                    used: 1 << 30,
                    available: None,
                    used_by_mold: 1 << 30,
                    used_by_other: 0,
                },
                cpu: None,
            });
        assert!(coordinator.leases.is_empty());
        assert!(coordinator
            .state
            .gpu_pool
            .workers
            .iter()
            .all(|worker| worker.in_flight.load(Ordering::SeqCst) == 0));
        let device = &coordinator.device_snapshots()[0];
        assert_eq!(device.activity, DeviceActivity::Busy);
        assert_eq!(device.available_vram_bytes, 24 << 30);

        // Even without a usable attribution sample, the authoritative running
        // row still prevents a terminal idle classification.
        publish_free_vram(&coordinator.state, 2 << 30);
        coordinator.unschedulable_idle_grace_ms = 0;
        for _ in 0..2 {
            let _ = coordinator.dispatch_ready().await;
        }

        assert!(
            coordinator.pending.contains_key("stranded"),
            "authoritatively running work must keep its follower queued"
        );
        assert!(
            coordinator.pending["stranded"]
                .unschedulable_since_ms
                .is_none(),
            "a running registry row must prevent the idle refusal timer"
        );
        assert!(worker_rx.try_recv().is_err());
        assert!(result_rx.try_recv().is_err());

        coordinator.state.job_registry.remove("active");
        assert_eq!(
            coordinator.device_snapshots()[0].activity,
            DeviceActivity::Idle,
            "removing the running row must release the fail-safe busy state"
        );
    }

    #[test]
    fn an_unschedulable_refusal_carries_the_plan_reason_it_was_given() {
        let named = crate::execution_plan::insufficient_vram_error(&[
            crate::execution_plan::DeviceInfeasibility {
                device_id: "cuda:0".to_string(),
                predicted_peak_bytes: 9_663_676_416,
                available_bytes: 2_147_483_648,
                advice: None,
            },
        ])
        .to_string();

        let message = unschedulable_rejection_message("minimax-h3-fl2va", Some(&named));
        assert!(message.contains("minimax-h3-fl2va"), "{message}");
        assert!(message.contains("cuda:0"), "{message}");
        assert!(message.contains("9.7 GB"), "{message}");
        assert!(message.contains("2.1 GB"), "{message}");

        // A resolver that produced neither plans nor an error still answers.
        let bare = unschedulable_rejection_message("flux-dev:q4", None);
        assert!(bare.contains("flux-dev:q4"), "{bare}");
        assert!(!bare.ends_with(": "), "{bare}");
    }

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[test]
    fn an_h3_host_shortfall_names_host_bytes_instead_of_being_sniffed_for_a_substring() {
        let reason = crate::execution_plan::h3_host_headroom_shortfall_reason(
            "cuda:0",
            15_300_615_032,
            8_869_770_650,
        );
        assert!(reason.contains("host"), "{reason}");
        assert!(reason.contains("15300615032"), "{reason}");
        assert!(reason.contains("8869770650"), "{reason}");
        // The opaque revalidation sentence this used to be recovered from
        // never contained "host" at all, which is what filed a host-memory
        // block as a VRAM shortfall.
        assert!(
            !"private H3 allocation-free admission evidence changed or no longer fits"
                .contains("host")
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
                restart_required: false,
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

    /// Telemetry rides the plan but must never be why one is published.
    ///
    /// The ledger resamples every second, so if `host_memory` decided semantic
    /// equality every host would emit a queue-plan SSE event once a second for
    /// as long as the server ran.
    #[test]
    fn host_memory_telemetry_never_triggers_a_queue_plan_event() {
        let base = mold_core::QueuePlan {
            plan_version: 1,
            state_version: 2,
            optimizer_state: "optimized".into(),
            host_memory: Some(mold_core::HostMemorySnapshot {
                total_bytes: 64 << 30,
                available_bytes: 40 << 30,
                headroom_bytes: 20 << 30,
                safety_floor_bytes: 10 << 30,
            }),
            ..Default::default()
        };
        let resampled = mold_core::QueuePlan {
            host_memory: Some(mold_core::HostMemorySnapshot {
                total_bytes: 64 << 30,
                available_bytes: 12 << 30,
                headroom_bytes: 0,
                safety_floor_bytes: 10 << 30,
            }),
            ..base.clone()
        };
        assert!(queue_plan_semantically_equal(&base, &resampled));

        let absent = mold_core::QueuePlan {
            host_memory: None,
            ..base.clone()
        };
        assert!(queue_plan_semantically_equal(&base, &absent));
    }

    /// An unsampled ledger reports nothing rather than a host at zero.
    #[test]
    fn an_unsampled_ledger_publishes_no_host_memory() {
        let mut ledger = unsampled_memory(64 << 30, 40 << 30);
        assert_eq!(ledger.wire_snapshot(), None);

        let started = ledger.begin_collection();
        ledger.publish_sample(started, 64 << 30, 40 << 30);
        let wire = ledger.wire_snapshot().expect("a sampled ledger reports");
        assert_eq!(wire.total_bytes, 64 << 30);
        assert_eq!(wire.available_bytes, 40 << 30);
        assert_eq!(wire.safety_floor_bytes, ledger.safety_floor_bytes());
        assert_eq!(wire.headroom_bytes, ledger.headroom_bytes());

        // Headroom is not `available - floor`: a live reservation spends it
        // while the machine still reports the same free memory.
        ledger.reservations.insert(
            "held".to_string(),
            HostReservation {
                bytes: 8 << 30,
                state: ReservationState::Reserved,
                charge_until_release: true,
            },
        );
        let held = ledger.wire_snapshot().expect("still sampled");
        assert_eq!(held.available_bytes, wire.available_bytes);
        assert_eq!(
            held.headroom_bytes,
            wire.headroom_bytes - (8 << 30),
            "a reservation must be visible as spent headroom"
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
            host_memory: None,
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
    fn semantic_noop_replan_refreshes_current_version_and_state_authority() {
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
        let mut events = coordinator.state.events.subscribe();
        coordinator.replan_and_publish();
        let initial = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("initial empty plan");
        assert!(matches!(
            events.try_recv(),
            Ok(mold_core::ServerEvent::QueuePlanChanged { .. })
        ));

        let mut immediate = false;
        coordinator.mutate(&mut immediate);
        coordinator.replan_and_publish();
        let refreshed = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("semantic no-op refreshes shared plan authority");

        assert!(queue_plan_semantically_equal(&initial, &refreshed));
        assert_eq!(refreshed.state_version, coordinator.state_version);
        assert!(refreshed.state_version > initial.state_version);
        assert!(refreshed.plan_version > initial.plan_version);
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn queue_projection_keeps_active_lease_after_pending_work_was_removed() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = GpuPool {
            workers: vec![worker.clone()].into(),
        };
        let device_id = worker_device_id(&worker);
        let plan = Planner::default()
            .plan(&PlannerSnapshot::new(
                12,
                18,
                monotonic_ms(),
                64 << 30,
                vec![DeviceSnapshot::busy(
                    device_id.clone(),
                    16 << 30,
                    monotonic_ms().saturating_add(30_000),
                )],
                Vec::new(),
            ))
            .unwrap();
        let mut work_snapshot = WorkSnapshot::new("active-a", 7, Vec::new());
        work_snapshot.parent_id = mold_scheduler::ParentId::new("batch-a");
        work_snapshot.kind = mold_scheduler::WorkKind::PreparedSibling;
        let leases = BTreeMap::from([(
            device_id.clone(),
            ActiveLease {
                work_id: "active-a".into(),
                owner_epoch: 1,
                plan_version: 17,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: monotonic_ms().saturating_add(25_000),
                ready_at_ms: 0,
                bypass_count: 2,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey {
                    device_class: "cuda-sm86".into(),
                    model_family: "flux".into(),
                    model_fingerprint: "cv:12345".into(),
                    work_kind: "prepared_sibling".into(),
                    shape_bucket: "1024x1024".into(),
                    execution_fingerprint: "exec-a".into(),
                },
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: work_snapshot,
                assignment_reason: AssignmentReason::Priority,
            },
        )]);

        let projected = queue_plan_projection(
            &PlannerSnapshot::new(12, 18, monotonic_ms(), 64 << 30, Vec::new(), Vec::new()),
            &plan,
            &pool,
            &leases,
            &BTreeMap::from([(
                "active-a".to_string(),
                mold_core::QueueEstimateConfidence::Medium,
            )]),
            None,
        );

        assert_eq!(projected.work_items.len(), 1);
        let active = &projected.work_items[0];
        assert_eq!(active.work_id, "active-a");
        assert_eq!(active.parent_id, "batch-a");
        assert_eq!(active.work_kind, "prepared_sibling");
        assert_eq!(
            active.planned_device_id.as_deref(),
            Some(device_id.as_str())
        );
        assert_eq!(
            active.planned_lane_kind,
            Some(mold_core::QueuePlannedLaneKind::Device)
        );
        assert_eq!(active.gpu, Some(0));
        assert_eq!(active.lane_order, Some(0));
        assert_eq!(active.activity_phase, mold_core::QueueActivityPhase::Active);
        assert_eq!(active.assignment_reason.as_deref(), Some("priority"));
        assert_eq!(active.execution_fingerprint.as_deref(), Some("exec-a"));
        assert_eq!(
            active.estimate_confidence,
            mold_core::QueueEstimateConfidence::Medium
        );
    }

    #[test]
    fn queue_projection_exposes_cpu_utility_identity_and_activity() {
        let pool = GpuPool {
            workers: Vec::new().into(),
        };
        let cpu_timing =
            mold_scheduler::static_timing_for_placement(WorkKind::StandaloneUpscale, Backend::Cpu);
        let cpu_total = cpu_timing
            .cold_setup_ms
            .saturating_add(cpu_timing.predicted_run_ms);
        let mut active_work = WorkSnapshot::new("cpu-active", 0, Vec::new());
        active_work.kind = WorkKind::StandaloneUpscale;
        let cpu_key = owner_estimate_key_for_device(
            CPU_UTILITY_DEVICE_ID.to_string(),
            WorkKind::StandaloneUpscale,
            "real-esrgan-x4plus:fp16",
            "upscale:auto",
            "exact-cpu-plan",
        );
        assert_eq!(cpu_key.device_class, CPU_UTILITY_DEVICE_ID);
        assert_eq!(cpu_key.execution_fingerprint, "exact-cpu-plan");
        assert_eq!(cpu_key.shape_bucket, "upscale:auto");

        let leases = BTreeMap::from([(
            CPU_UTILITY_DEVICE_ID.to_string(),
            ActiveLease {
                work_id: "cpu-active".into(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: monotonic_ms().saturating_add(cpu_total),
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: cpu_key,
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: active_work,
                assignment_reason: AssignmentReason::Priority,
            },
        )]);
        let empty_plan = Planner::default()
            .plan(&PlannerSnapshot::new(
                1,
                1,
                monotonic_ms(),
                64 << 30,
                Vec::new(),
                Vec::new(),
            ))
            .unwrap();
        let active = queue_plan_projection(
            &PlannerSnapshot::new(1, 1, monotonic_ms(), 64 << 30, Vec::new(), Vec::new()),
            &empty_plan,
            &pool,
            &leases,
            &BTreeMap::new(),
            None,
        );
        assert_eq!(active.work_items.len(), 1);
        assert_eq!(
            active.work_items[0].planned_lane_kind,
            Some(mold_core::QueuePlannedLaneKind::HostUtility)
        );
        assert_eq!(active.work_items[0].planned_device_id, None);
        assert_eq!(
            active.work_items[0].activity_phase,
            mold_core::QueueActivityPhase::Cpu
        );
        let active_wire = serde_json::to_value(&active.work_items[0]).unwrap();
        assert_eq!(active_wire["planned_lane_kind"], "host_utility");
        assert!(active_wire
            .as_object()
            .unwrap()
            .contains_key("planned_device_id"));
        assert_eq!(active_wire["planned_device_id"], serde_json::Value::Null);
        assert_eq!(active.work_items[0].gpu, None);
        let active_eta = active.work_items[0]
            .estimated_finish_unix_ms
            .unwrap()
            .saturating_sub(active.work_items[0].estimated_start_unix_ms.unwrap());
        assert!(
            active_eta >= cpu_total.saturating_sub(100) && active_eta <= cpu_total + 100,
            "active utility ETA must retain the placement-aware CPU floor"
        );

        let mut queued_work = WorkSnapshot::new(
            "cpu-queued",
            0,
            vec![
                CandidatePlacement::new(CPU_UTILITY_DEVICE_ID, "queued-exact-cpu-plan", 1)
                    .with_timing(
                        cpu_timing.cold_setup_ms,
                        cpu_timing.cold_setup_ms,
                        cpu_timing.predicted_run_ms,
                    ),
            ],
        );
        queued_work.kind = WorkKind::StandaloneUpscale;
        let queued_snapshot = PlannerSnapshot::new(
            2,
            2,
            monotonic_ms(),
            64 << 30,
            vec![DeviceSnapshot::idle(CPU_UTILITY_DEVICE_ID, u64::MAX).with_backend(Backend::Cpu)],
            vec![queued_work],
        );
        let queued_plan = Planner::default().plan(&queued_snapshot).unwrap();
        let projection_unix_now = 1_000_000;
        let queued = queue_plan_projection_at_unix(
            &queued_snapshot,
            &queued_plan,
            &pool,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
            projection_unix_now,
        );
        assert_eq!(queued.work_items.len(), 1);
        assert_eq!(
            queued.work_items[0].planned_lane_kind,
            Some(mold_core::QueuePlannedLaneKind::HostUtility)
        );
        assert_eq!(queued.work_items[0].planned_device_id, None);
        assert_eq!(
            queued.work_items[0].activity_phase,
            mold_core::QueueActivityPhase::Queued
        );
        let queued_wire = serde_json::to_value(&queued.work_items[0]).unwrap();
        assert_eq!(queued_wire["planned_lane_kind"], "host_utility");
        assert!(queued_wire
            .as_object()
            .unwrap()
            .contains_key("planned_device_id"));
        assert_eq!(queued_wire["planned_device_id"], serde_json::Value::Null);
        assert_eq!(queued.work_items[0].gpu, None);
        assert_eq!(
            queued.work_items[0].estimated_start_unix_ms,
            Some(projection_unix_now),
            "the projection clock must use the planner snapshot as its monotonic origin"
        );
        assert_eq!(
            queued.work_items[0].estimated_finish_unix_ms,
            Some(projection_unix_now + cpu_total),
            "the projection clock must preserve the planner's exact duration"
        );
        assert_eq!(
            queued.work_items[0]
                .estimated_finish_unix_ms
                .unwrap()
                .saturating_sub(queued.work_items[0].estimated_start_unix_ms.unwrap()),
            cpu_total,
            "queued utility ETA must retain the placement-aware CPU floor"
        );
    }

    #[test]
    fn device_events_publish_once_per_semantic_health_transition() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
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
        // This fixture exercises lifecycle events emitted by the authoritative
        // coordinator. Production installs this handle before the coordinator
        // starts; the maintenance default deliberately projects live legacy
        // workers as enabled and read-only.
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work =
            ScheduledWorkHandle::for_mode(scheduled_tx, crate::dispatch_mode::DispatchMode::V2);
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
            mold_core::ServerEvent::DeviceStateChanged {
                ref device_id,
                desired_enabled: true,
                admin_state: mold_core::DeviceAdminState::Enabled,
            } if device_id == worker.gpu.stable_id.as_deref().unwrap()
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
            mold_core::ServerEvent::DeviceStateChanged {
                ref device_id,
                desired_enabled: false,
                admin_state: mold_core::DeviceAdminState::Draining,
            } if device_id == worker.gpu.stable_id.as_deref().unwrap()
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
            mold_core::ServerEvent::DeviceStateChanged {
                ref device_id,
                desired_enabled: false,
                admin_state: mold_core::DeviceAdminState::Draining,
            } if device_id == worker.gpu.stable_id.as_deref().unwrap()
        ));
    }

    #[test]
    fn failed_post_upscale_completion_is_counted_but_never_trains_eta() {
        let make_coordinator = || {
            let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
            let state = AppState::empty(
                mold_core::Config::default(),
                QueueHandle::new(ingress_tx),
                Arc::new(GpuPool {
                    workers: Vec::new().into(),
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
            model_family: "real-esrgan".into(),
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
                    owner_epoch: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    accepted: true,
                    previous_target: None,
                    estimated_finish_ms: 0,
                    ready_at_ms: 0,
                    bypass_count: 0,
                    warm_wait_started_ms: None,
                    started_at: Instant::now(),
                    estimate_key: key.clone(),
                    vram_high_water_bytes: None,
                    host_incremental_high_water_bytes: None,
                    fallback_reason: None,
                    projection: WorkSnapshot::new("post-upscale", 0, Vec::new()),
                    assignment_reason: AssignmentReason::Priority,
                },
            );
            let mut immediate = false;
            coordinator.handle_worker_event(
                WorkerEvent::Completed {
                    device_id: "cuda:stable".into(),
                    ordinal: 0,
                    owner_epoch: 1,
                    worker_generation: 1,
                    successful,
                    phase_timings: EstimatePhaseTimings {
                        cold_load_ms: Some(250),
                        ..Default::default()
                    },
                    completion: None,
                },
                &mut immediate,
            );
        };

        let mut failed = make_coordinator();
        complete(&mut failed, false);
        let failed_bucket = failed.estimates.exact(&key).unwrap();
        assert_eq!(failed_bucket.sample_count, 0);
        assert_eq!(failed_bucket.failure_count, 1);

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
    fn vram_high_water_joins_by_stable_device_before_ordinal() {
        let (gpu_zero, _rx_zero) = test_worker(0);
        let (gpu_one, _rx_one) = test_worker(1);
        let workers = [gpu_one.clone(), gpu_zero];
        let discovered = workers
            .iter()
            .map(|worker| worker.gpu.clone())
            .collect::<Vec<_>>();
        let registry = crate::device_registry::DeviceRegistry::from_runtime_inventory(
            discovered.clone(),
            &discovered,
            Arc::new(None),
        );
        let snapshot = mold_core::ResourceSnapshot {
            hostname: "test".into(),
            timestamp: 1,
            // Deliberately reversed: vector position is not device identity.
            gpus: vec![
                mold_core::GpuSnapshot {
                    ordinal: 1,
                    name: "gpu-1".into(),
                    backend: mold_core::GpuBackend::Cuda,
                    vram_total: 24 << 30,
                    vram_used: 9 << 30,
                    vram_used_by_mold: Some(7 << 30),
                    vram_used_by_other: Some(2 << 30),
                    gpu_utilization: Some(90),
                },
                mold_core::GpuSnapshot {
                    ordinal: 0,
                    name: "gpu-0".into(),
                    backend: mold_core::GpuBackend::Cuda,
                    vram_total: 24 << 30,
                    vram_used: 3 << 30,
                    vram_used_by_mold: Some(1 << 30),
                    vram_used_by_other: Some(2 << 30),
                    gpu_utilization: Some(10),
                },
            ],
            system_ram: mold_core::RamSnapshot {
                total: 64 << 30,
                used: 8 << 30,
                available: Some(56 << 30),
                used_by_mold: 0,
                used_by_other: 8 << 30,
            },
            cpu: None,
        };

        assert_eq!(
            vram_sample_for_stable_device(
                &snapshot,
                &registry,
                gpu_one.gpu.stable_id.as_deref().unwrap()
            ),
            Some(7 << 30)
        );
    }

    #[test]
    fn db_disabled_estimator_learns_for_the_process_lifetime() {
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            Arc::new(GpuPool {
                workers: Vec::new().into(),
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
            model_family: "flux".into(),
            model_fingerprint: "flux-dev:q8".into(),
            work_kind: "generation".into(),
            shape_bucket: "1024x1024".into(),
            execution_fingerprint: "q8".into(),
        };

        coordinator.observe_estimate(
            key.clone(),
            EstimateObservation {
                total_ms: Some(12_000),
                phases: EstimatePhaseTimings {
                    cold_load_ms: Some(2_000),
                    ..Default::default()
                },
                vram_high_water_bytes: Some(20 << 30),
                host_incremental_high_water_bytes: Some(8 << 30),
                observed_at_unix_s: unix_seconds(),
                ..Default::default()
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
    fn only_h3_metal_uses_frozen_device_capacity() {
        assert!(generation_uses_frozen_device_capacity(
            mold_core::GpuBackend::Metal,
            mold_core::minimax_h3::FAMILY,
        ));
        assert!(!generation_uses_frozen_device_capacity(
            mold_core::GpuBackend::Cuda,
            mold_core::minimax_h3::FAMILY,
        ));
        assert!(!generation_uses_frozen_device_capacity(
            mold_core::GpuBackend::Metal,
            "flux",
        ));
    }

    #[test]
    fn queued_capacity_reclaims_only_known_mold_vram_from_active_work() {
        const GIB: u64 = 1 << 30;

        assert_eq!(
            schedulable_available_vram_bytes(10 * GIB, 0, Some(14 * GIB), true, 24 * GIB),
            24 * GIB,
            "a sibling session should queue behind active Mold work"
        );
        assert_eq!(
            schedulable_available_vram_bytes(10 * GIB, 0, Some(14 * GIB), false, 24 * GIB),
            10 * GIB,
            "idle Mold attribution is not automatically reclaimable"
        );
        assert_eq!(
            schedulable_available_vram_bytes(10 * GIB, 0, None, true, 24 * GIB),
            10 * GIB,
            "unknown attribution must remain fail closed"
        );
        assert_eq!(
            schedulable_available_vram_bytes(4 * GIB, 0, Some(14 * GIB), true, 24 * GIB),
            18 * GIB,
            "external allocations remain unavailable after active Mold work completes"
        );
    }

    #[test]
    fn busy_unattributed_metal_pressure_is_transient_when_the_peak_fits_physically() {
        const GIB: u64 = 1 << 30;
        let available = schedulable_available_vram_bytes(4 * GIB, 0, None, true, 24 * GIB);
        assert_eq!(
            available,
            4 * GIB,
            "a busy unattributed lane must not invent reclaimable bytes"
        );

        let failure = classify_generation_plan_failure(
            crate::execution_plan::ExecutionPlanError::InsufficientVram {
                reason: "metal:0 is currently busy".to_string(),
                required_peak_bytes: 20 * GIB,
                eligible_device_ids: vec!["metal:0".to_string()],
            },
            &BTreeMap::from([("metal:0".to_string(), 24 * GIB)]),
        );
        assert!(
            matches!(&failure, GenerationPlanFailure::Transient(_)),
            "a physically fitting job waits for the active lane instead of being refused"
        );
        assert_eq!(
            placement_preview_disposition_for_plan_failure(&failure),
            (true, "temporarily_unavailable"),
            "placement preview must not turn current lane pressure into infeasibility"
        );
    }

    #[test]
    fn stale_preparation_declines_preview_authority_so_admission_can_reclaim() {
        let failure = classify_generation_plan_failure(
            crate::execution_plan::ExecutionPlanError::PreparedInputsStale(
                "host capacity changed after private admission".to_string(),
            ),
            &BTreeMap::new(),
        );

        assert_eq!(
            placement_preview_disposition_for_plan_failure(&failure),
            (false, "unsupported"),
            "the mutating admission path must get the chance to refresh evidence and unload cache"
        );
    }

    #[test]
    fn busy_unattributed_cuda_lanes_keep_an_additional_fitting_job_waiting() {
        const GIB: u64 = 1 << 30;
        let available = (0..2)
            .map(|_| schedulable_available_vram_bytes(3 * GIB, 0, None, true, 24 * GIB))
            .collect::<Vec<_>>();
        assert_eq!(available, vec![3 * GIB, 3 * GIB]);

        let failure = classify_generation_plan_failure(
            crate::execution_plan::ExecutionPlanError::InsufficientVram {
                reason: "both CUDA lanes are currently busy".to_string(),
                required_peak_bytes: 18 * GIB,
                eligible_device_ids: vec!["cuda:0".to_string(), "cuda:1".to_string()],
            },
            &BTreeMap::from([
                ("cuda:0".to_string(), 24 * GIB),
                ("cuda:1".to_string(), 24 * GIB),
            ]),
        );
        assert!(matches!(failure, GenerationPlanFailure::Transient(_)));
    }

    #[test]
    fn unattributed_external_allocations_are_not_dispatch_capacity() {
        const GIB: u64 = 1 << 30;
        assert_eq!(
            schedulable_available_vram_bytes(3 * GIB, 0, None, false, 24 * GIB),
            3 * GIB,
            "neither physical total nor unknown process memory is immediate capacity"
        );
    }

    #[test]
    fn first_party_cache_remains_reclaimable_without_process_attribution() {
        const GIB: u64 = 1 << 30;
        let reclaimable = reclaimable_model_cache_bytes(16 * GIB, None);
        assert_eq!(reclaimable, 16 * GIB);
        assert_eq!(
            schedulable_available_vram_bytes(4 * GIB, reclaimable, None, false, 24 * GIB),
            20 * GIB,
            "the owner can evict its measured cache even when the OS cannot attribute the process"
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
                vram_used_by_mold: None,
                vram_used_by_other: None,
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
                started_at: Instant::now(),
                estimate_key: EstimateKey {
                    device_class: "cuda-sm86".into(),
                    model_family: "resident-test".into(),
                    model_fingerprint: "resident-test".into(),
                    work_kind: "generation".into(),
                    shape_bucket: "test".into(),
                    execution_fingerprint: "warm-plan".into(),
                },
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new("busy", 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
            },
        );
        let busy = coordinator.device_snapshots().remove(0);
        assert_eq!(busy.available_vram_bytes, 20 * GIB);
        assert_eq!(busy.available_at_ms, Some(5_000));
        assert_eq!(
            busy.activity,
            DeviceActivity::Busy,
            "first-party cache credit cannot bypass the scheduler-owned lane gate"
        );
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sqlite_blocked_queue_patch_omits_only_its_target_until_exact_release() {
        use tower::ServiceExt as _;

        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let (worker_a, worker_a_rx) = test_worker(0);
        let (worker_b, worker_b_rx) = test_worker(1);
        let pool = Arc::new(GpuPool {
            workers: vec![worker_a.clone(), worker_b.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(ingress_tx);
        let mut state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 8);
        state.metadata_db = db.clone();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(root.path()),
            "blocked-patch-coordinator",
        ));
        let owner = state
            .queue_journal
            .owner_uuid()
            .expect("real metadata DB must enable the durable journal")
            .to_string();
        mold_db::generation_queue::insert(
            db.as_ref().as_ref().unwrap(),
            &mold_db::generation_queue::GenerationQueueRow {
                id: "patch-target".to_string(),
                owner_uuid: owner,
                state: mold_db::generation_queue::QueueRowState::Queued,
                model: "flux-dev:q4".to_string(),
                request_json: r#"{"prompt":"patch target","model":"flux-dev:q4"}"#.to_string(),
                media_set_id: None,
                admission_authority: None,
                output_dir: root.path().join("gallery"),
                target_gpu: Some(0),
                target_device_id: None,
                completion_payload: "full".to_string(),
                seed_pinned: false,
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                started_at_ms: None,
            },
        )
        .unwrap();

        // The target starts behind another lane-zero job. PATCH will move it
        // to the queued frontier while SQLite is deliberately stalled.
        let mut results = Vec::new();
        for (id, target_gpu) in [
            ("lane-zero-contender", 0),
            ("unrelated-linux-lane", 1),
            ("patch-target", 0),
        ] {
            let (job, result) = fake_generation(id);
            results.push(result);
            state.job_registry.register(id, "flux-dev:q4");
            state
                .job_registry
                .set_target_gpu(id, Some(target_gpu))
                .unwrap();
            queue.submit(job, 8).await.unwrap();
        }

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        for _ in 0..3 {
            coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        }
        coordinator
            .pending
            .get_mut("unrelated-linux-lane")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator
            .pending
            .get_mut("patch-target")
            .unwrap()
            .preparation = PreparationState::Ready;
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

        // Hold the journal's real SQLite connection. The route can install
        // its exact runtime token, but cannot apply the durable edit yet.
        let locked_db = db.clone();
        let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let blocker = tokio::task::spawn_blocking(move || {
            locked_db.as_ref().as_ref().unwrap().with_conn(|_| {
                locked_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });
        locked_rx.await.unwrap();
        let patch = tokio::spawn(
            crate::routes::create_router(state.clone()).oneshot(
                axum::http::Request::patch("/api/queue/patch-target")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(r#"{"target_gpu":0,"position":0}"#))
                    .unwrap(),
            ),
        );
        tokio::time::timeout(Duration::from_secs(2), async {
            while !state
                .job_registry
                .queue_patch_blocked_ids()
                .contains("patch-target")
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("hydrated target must carry the exact PATCH exclusion token");
        assert!(!patch.is_finished(), "PATCH must remain blocked on SQLite");

        coordinator.dispatch_ready().await;
        assert_eq!(
            recv_grant(&worker_b_rx).id,
            "unrelated-linux-lane",
            "the unrelated lane must grant while the PATCH target is omitted"
        );
        assert!(matches!(
            worker_a_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert!(coordinator.pending.contains_key("patch-target"));

        release_tx.send(()).unwrap();
        blocker.await.unwrap().unwrap();
        let response = patch.await.unwrap().unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            state.job_registry.queued_ids_in_order(),
            ["patch-target", "lane-zero-contender"],
            "the durable PATCH order must project before its token is cleared"
        );

        coordinator
            .pending
            .get_mut("lane-zero-contender")
            .unwrap()
            .preparation = PreparationState::Ready;
        coordinator.dispatch_ready().await;
        assert_eq!(
            recv_grant(&worker_a_rx).id,
            "patch-target",
            "exact-token release must immediately restore target eligibility and order"
        );
        assert!(coordinator.pending.contains_key("lane-zero-contender"));
        drop(results);
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

    #[test]
    fn generation_device_facts_apply_ordinal_and_stable_worker_constraints() {
        let facts = vec![
            crate::execution_plan::DeviceFact {
                id: "cuda:stable-small".to_string(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 8 << 30,
            },
            crate::execution_plan::DeviceFact {
                id: "cuda:stable-large".to_string(),
                ordinal: 1,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 9)),
                available_vram_bytes: 24 << 30,
            },
        ];

        assert_eq!(
            constrained_generation_device_facts(&facts, None, None),
            facts,
            "Auto retains every schedulable worker"
        );
        assert_eq!(
            constrained_generation_device_facts(&facts, Some(0), None),
            vec![facts[0].clone()],
            "queue/request ordinal pins constrain planning before admission"
        );
        assert_eq!(
            constrained_generation_device_facts(&facts, None, Some("cuda:stable-large"),),
            vec![facts[1].clone()],
            "per-worker planning uses the stable worker identity"
        );
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
    async fn chain_stages_fill_every_ready_device_for_one_two_and_eight_gpu_inventories() {
        for worker_count in [1_usize, 2, 8] {
            let root = tempfile::tempdir().unwrap();
            let transformer = root.path().join("transformer.safetensors");
            let vae = root.path().join("vae.safetensors");
            std::fs::write(&transformer, b"transformer").unwrap();
            std::fs::write(&vae, b"vae").unwrap();
            let model = "real-chain-stage";
            let frozen_config = mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("sd15".to_string()),
                ..mold_core::ModelConfig::default()
            };
            let expected =
                crate::execution_plan::frozen_model_fingerprint(model, &frozen_config).unwrap();
            let mut work_config = mold_core::Config::default();
            work_config.install_frozen_model_config(model, frozen_config);
            let workers_and_receivers = (0..worker_count).map(test_worker).collect::<Vec<_>>();
            let workers = workers_and_receivers
                .iter()
                .map(|(worker, _)| worker.clone())
                .collect::<Vec<_>>();
            let pool = Arc::new(GpuPool {
                workers: workers.into(),
            });
            let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
            let state = AppState::empty(
                mold_core::Config::default(),
                QueueHandle::new(ingress_tx),
                pool,
                worker_count,
            );
            let mut coordinator = Coordinator::with_preparer_and_memory(
                state,
                Arc::new(ImmediatePreparer),
                ample_memory(),
            );
            let mut immediate = false;
            let mut result_receivers = Vec::new();
            for (ordinal, (worker, _)) in workers_and_receivers.iter().enumerate() {
                let id = format!("chain-{ordinal}:stage:0");
                let (generation, _) = fake_generation(&format!("request-{ordinal}"));
                let mut stage_req = generation.request;
                stage_req.model = model.to_string();
                let (result_tx, result_rx) = tokio::sync::oneshot::channel();
                result_receivers.push(result_rx);
                coordinator.enqueue_owner_work(
                    ScheduledOwnerWork::new(
                        id.clone(),
                        expected.clone(),
                        1 << 30,
                        OwnerWork::ChainStage(Box::new(
                            crate::chain_job_runner::ScheduledChainStageWork {
                                id,
                                model: model.to_string(),
                                cache_key: format!("mold-frozen-chain:{expected}"),
                                config: work_config.clone(),
                                stage_req,
                                carry: None,
                                motion_tail_frames: 1,
                                progress: Arc::new(|_, _| std::ops::ControlFlow::Continue(())),
                                cancelled: Arc::new(|| false),
                                cancellation: mold_inference::InferenceCancellationToken::default(),
                                on_leased: None,
                                execution_plan: None,
                                expected_model_fingerprint: Some(expected.clone()),
                                result_tx: Some(result_tx),
                                before_second_fence: None,
                            },
                        )),
                    ),
                    &mut immediate,
                );
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

            for pending in coordinator.pending_owner_work.values() {
                let plans = coordinator
                    .owner_plans(pending)
                    .expect("real chain-stage plans must resolve");
                assert_eq!(plans.len(), worker_count);
            }
            let debug_cache = coordinator
                .pending_owner_work
                .iter()
                .map(|(id, pending)| (id.clone(), coordinator.owner_plans(pending).unwrap()))
                .collect::<BTreeMap<_, _>>();
            let (debug_snapshot, _) = coordinator.planner_snapshot(&debug_cache);
            let debug_plan = mold_scheduler::Planner::default()
                .plan(&debug_snapshot)
                .unwrap();
            assert!(
                !debug_plan.immediate_leases.is_empty(),
                "real chain stages unexpectedly blocked: {:?}",
                debug_plan.blocked
            );
            coordinator.dispatch_ready().await;

            assert_eq!(coordinator.leases.len(), worker_count);
            assert!(coordinator.pending_owner_work.is_empty());
            for (_, receiver) in &workers_and_receivers {
                let command = receiver.recv().expect("one chain-stage grant per GPU");
                match command {
                    crate::gpu_pool::GpuWorkerCommand::Grant(grant) => {
                        assert_eq!(grant.work.kind(), mold_scheduler::WorkKind::ChainStage);
                    }
                    crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected worker drain"),
                    crate::gpu_pool::GpuWorkerCommand::Shutdown => {
                        panic!("unexpected worker shutdown")
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn chain_and_utility_work_share_typed_phase_e_lanes_without_identity_collision() {
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
            2,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        let chain_id = "chain:durable-parent:attempt:7:stage:2";
        for (id, kind) in [
            (chain_id, mold_scheduler::WorkKind::ChainStage),
            (
                "utility-upscale",
                mold_scheduler::WorkKind::StandaloneUpscale,
            ),
        ] {
            coordinator.enqueue_owner_work(
                ScheduledOwnerWork::new(
                    id,
                    format!("fingerprint:{id}"),
                    1 << 30,
                    OwnerWork::Probe {
                        id: id.to_string(),
                        kind,
                        run: Box::new(|| {}),
                    },
                ),
                &mut immediate,
            );
        }
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

        assert!(matches!(
            worker_a_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            crate::gpu_pool::GpuWorkerCommand::Grant(_)
        ));
        assert!(matches!(
            worker_b_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            crate::gpu_pool::GpuWorkerCommand::Grant(_)
        ));
        let plan = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("authoritative typed plan");
        let chain = plan
            .work_items
            .iter()
            .find(|item| item.work_id == chain_id)
            .expect("chain work item");
        assert_eq!(chain.parent_id, "durable-parent");
        assert_eq!(chain.chain_stage, Some(2));
        assert_eq!(chain.work_kind, "chain_stage");
        assert!(chain.planned_device_id.is_some());
        assert!(chain.estimated_start_unix_ms.is_some());
        assert!(chain.estimated_finish_unix_ms.is_some());
        let utility = plan
            .work_items
            .iter()
            .find(|item| item.work_id == "utility-upscale")
            .expect("utility work item");
        assert_eq!(utility.parent_id, "utility-upscale");
        assert_eq!(utility.chain_stage, None);
        assert_eq!(utility.work_kind, "standalone_upscale");
        assert!(utility.planned_device_id.is_some());
        assert_ne!(chain.work_id, utility.work_id);
    }

    #[tokio::test]
    async fn draining_and_fatal_devices_never_dispatch_planned_chain_or_utility_work() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            2,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        for (id, kind) in [
            (
                "chain:draining-parent:attempt:1:stage:0",
                mold_scheduler::WorkKind::ChainStage,
            ),
            (
                "utility-while-draining",
                mold_scheduler::WorkKind::StandaloneUpscale,
            ),
        ] {
            coordinator.enqueue_owner_work(
                ScheduledOwnerWork::new(
                    id,
                    format!("fingerprint:{id}"),
                    1 << 30,
                    OwnerWork::Probe {
                        id: id.to_string(),
                        kind,
                        run: Box::new(|| {}),
                    },
                ),
                &mut immediate,
            );
        }
        worker.request_drain(false);
        coordinator.replan_and_publish();
        let plan = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("draining plan");
        assert_eq!(plan.work_items.len(), 2);
        let blocked = plan
            .work_items
            .iter()
            .map(|item| (item.work_id.clone(), item.blocked_reason.clone()))
            .collect::<Vec<_>>();
        assert!(
            blocked.iter().all(|(_, reason)| {
                *reason == Some(mold_core::QueueBlockedReason::NoSchedulableDevice)
            }),
            "auto-routed work must remain blocked while the only device drains: {blocked:?}"
        );
        assert_eq!(
            worker.drain_state.load(Ordering::SeqCst),
            crate::gpu_pool::DRAIN_REQUESTED
        );

        worker.fatal_cuda_error.store(true, Ordering::SeqCst);
        assert_eq!(coordinator.dispatch_ready().await, None);
        assert!(worker_rx.try_recv().is_err());
        assert_eq!(coordinator.pending_owner_work.len(), 2);
        assert!(coordinator.leases.is_empty());
    }

    #[test]
    fn lease_reconstruction_selects_the_exact_execution_fingerprint() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer.safetensors");
        let vae = root.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let model = "exact-plan";
        let mut config = mold_core::Config::default();
        config.models.insert(
            model.into(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("sd15".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let (generation, _) = fake_generation("exact-plan-request");
        let mut request = generation.request;
        request.model = model.into();
        let mut wanted = crate::execution_plan::resolve_execution_plans(
            &config,
            &request,
            &[crate::execution_plan::DeviceFact {
                id: "cuda:exact".into(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24 << 30,
            }],
            false,
        )
        .unwrap()
        .remove(0);
        wanted.execution_fingerprint = "wanted".into();
        let mut stale = wanted.clone();
        stale.execution_fingerprint = "stale".into();
        let lease = mold_scheduler::ImmediateLease {
            work_id: WorkId::new("chain:exact:stage:0"),
            device_id: DeviceId::new("cuda:exact"),
            worker_generation: 1,
            placement: CandidatePlacement::new("cuda:exact", "wanted", 0),
            reason: mold_scheduler::AssignmentReason::Priority,
            estimated_start_ms: 0,
            estimated_finish_ms: 1,
        };

        assert_eq!(
            exact_leased_execution_plan(&[stale, wanted], &lease)
                .unwrap()
                .execution_fingerprint,
            "wanted"
        );
    }

    #[tokio::test]
    async fn learned_generation_memory_does_not_block_exact_plan_transport() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
        let (job, _result_rx) = fake_generation("learned-memory");
        state
            .job_registry
            .register("learned-memory", &job.request.model);
        queue.submit(job, 1).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("learned-memory")
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

        let execution = coordinator
            .generation_plans(coordinator.pending.get("learned-memory").unwrap())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let learned_vram = execution.predicted_vram_peak_bytes + (1 << 30);
        assert!(learned_vram < worker.gpu.total_vram_bytes);
        coordinator.observe_estimate(
            generation_estimate_key(
                &coordinator.state,
                &worker,
                &coordinator.pending["learned-memory"].job.request,
                None,
                &execution.execution_fingerprint,
            ),
            EstimateObservation {
                total_ms: Some(1_000),
                phases: EstimatePhaseTimings::default(),
                vram_high_water_bytes: Some(learned_vram),
                host_incremental_high_water_bytes: None,
                outcome: EstimateOutcome::Success,
                observed_at_unix_s: unix_seconds(),
                ..Default::default()
            },
        );

        let (snapshot, _) = coordinator.planner_snapshot(&BTreeMap::new());
        let plan = coordinator.admission_planner.plan(&snapshot).unwrap();
        assert_eq!(
            plan.immediate_leases[0].placement.predicted_vram_bytes, learned_vram,
            "the lease reservation must retain conservative learned memory"
        );

        coordinator.dispatch_ready().await;

        let transported = recv_grant(&worker_rx);
        assert_eq!(transported.id, "learned-memory");
        assert_eq!(
            transported
                .execution_plan
                .unwrap()
                .predicted_vram_peak_bytes,
            execution.predicted_vram_peak_bytes,
            "the worker must receive the immutable execution plan selected by identity"
        );
    }

    #[tokio::test]
    async fn paused_queue_publishes_and_clears_exact_work_without_transport() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
        let (job, _result_rx) = fake_generation("paused-visible");
        state
            .job_registry
            .register("paused-visible", &job.request.model);
        queue.submit(job, 1).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("paused-visible")
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
        assert!(
            coordinator.set_queue_paused_and_publish(true).unwrap(),
            "first scheduler-owned pause must change state"
        );

        assert!(
            worker_rx.try_recv().is_err(),
            "pause must prevent worker transport"
        );
        assert!(coordinator.leases.is_empty());
        assert!(coordinator.pending.contains_key("paused-visible"));
        let paused = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("paused pending work must remain observable");
        let item = paused
            .work_items
            .iter()
            .find(|item| item.work_id == "paused-visible")
            .expect("paused job must retain its exact scheduler work ID");
        assert_eq!(item.parent_id, "paused-visible");
        assert_eq!(item.activity_phase, mold_core::QueueActivityPhase::Blocked);
        assert_eq!(
            item.blocked_reason,
            Some(mold_core::QueueBlockedReason::QueuePaused)
        );
        assert_eq!(item.planned_device_id, None);

        let paused_plan_version = paused.plan_version;
        coordinator
            .state
            .job_registry
            .cancel_queued("paused-visible")
            .unwrap();
        coordinator.reconcile_external_mutations(&mut immediate);
        coordinator
            .dispatch_ready_with(PlanningPass::Admission)
            .await;
        let cancelled = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("cancellation must replace the paused plan");
        assert!(cancelled.plan_version > paused_plan_version);
        assert!(
            cancelled.work_items.is_empty(),
            "cancelled paused work must not remain as a ghost plan item"
        );
        assert!(!coordinator.pending.contains_key("paused-visible"));

        assert!(
            coordinator.set_queue_paused_and_publish(false).unwrap(),
            "scheduler-owned resume must change state"
        );
        assert_eq!(
            worker_rx.try_recv().err(),
            Some(std::sync::mpsc::TryRecvError::Empty),
            "resuming after cancellation must not transport cancelled work"
        );
        assert!(coordinator.leases.is_empty());
    }

    #[tokio::test]
    async fn pause_after_plan_before_grant_replaces_runnable_authority() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
        let (job, _result_rx) = fake_generation("pause-race");
        state
            .job_registry
            .register("pause-race", &job.request.model);
        queue.submit(job, 1).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("pause-race")
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

        let queue_pause = coordinator.state.queue_pause.clone();
        let plan_built = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        coordinator.before_grant_hook = Some(BeforeGrantHook {
            plan_built: plan_built.clone(),
            resume: resume.clone(),
        });
        let dispatch = tokio::spawn(async move {
            coordinator
                .dispatch_ready_with(PlanningPass::Admission)
                .await;
            coordinator
        });

        plan_built.notified().await;
        queue_pause.pause();
        resume.notify_one();
        let coordinator = dispatch.await.unwrap();

        assert!(
            worker_rx.try_recv().is_err(),
            "a pause racing the grant fence must prevent transport"
        );
        assert!(coordinator.leases.is_empty());
        assert!(coordinator.pending.contains_key("pause-race"));
        let paused = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("race must replace the runnable plan");
        let item = paused
            .work_items
            .iter()
            .find(|item| item.work_id == "pause-race")
            .expect("paused race plan must retain exact work identity");
        assert_eq!(item.activity_phase, mold_core::QueueActivityPhase::Blocked);
        assert_eq!(
            item.blocked_reason,
            Some(mold_core::QueueBlockedReason::QueuePaused)
        );
        assert_eq!(item.planned_device_id, None);
    }

    #[test]
    fn failed_resume_plan_publication_restores_paused_gate() {
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
        assert!(coordinator.set_queue_paused_and_publish(true).unwrap());
        coordinator.next_plan_error = Some(PlannerError::DuplicateWorkId {
            work_id: WorkId::new("injected-resume-publication-error"),
        });

        let error = coordinator
            .set_queue_paused_and_publish(false)
            .expect_err("resume must fail when its authority cannot publish");

        assert!(error.contains("could not publish queue pause state"));
        assert!(
            coordinator.state.queue_pause.is_paused(),
            "failed resume must restore the safe paused gate"
        );
    }

    #[test]
    fn failed_resume_never_releases_a_blocked_waiter_before_plan_commit() {
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
        let queue_pause = state.queue_pause.clone();
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        assert!(coordinator.set_queue_paused_and_publish(true).unwrap());
        coordinator.next_plan_error = Some(PlannerError::DuplicateWorkId {
            work_id: WorkId::new("injected-resume-publication-error"),
        });

        let crossed = Arc::new(AtomicBool::new(false));
        let waiter_pause = queue_pause.clone();
        let waiter_crossed = crossed.clone();
        let (start_tx, start_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            start_rx.recv().unwrap();
            waiter_pause.wait_if_paused_blocking(&|| false);
            waiter_crossed.store(true, Ordering::SeqCst);
        });
        let observed_crossed = crossed.clone();
        coordinator.before_queue_control_plan_hook = Some(Box::new(move || {
            start_tx.send(()).unwrap();
            let deadline = Instant::now() + Duration::from_millis(250);
            while !observed_crossed.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::yield_now();
            }
        }));

        coordinator
            .set_queue_paused_and_publish(false)
            .expect_err("injected resume publication failure must be returned");

        assert!(
            !crossed.load(Ordering::SeqCst),
            "a rejected resume must never expose a transient unpaused gate"
        );
        assert!(queue_pause.is_paused());
        queue_pause.resume();
        waiter.join().unwrap();
    }

    #[test]
    fn failed_pause_plan_publication_restores_unpaused_gate() {
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
        coordinator.next_plan_error = Some(PlannerError::DuplicateWorkId {
            work_id: WorkId::new("injected-pause-publication-error"),
        });

        let error = coordinator
            .set_queue_paused_and_publish(true)
            .expect_err("pause must fail when its authority cannot publish");

        assert!(error.contains("could not publish queue pause state"));
        assert!(
            !coordinator.state.queue_pause.is_paused(),
            "failed pause must restore the prior unpaused gate"
        );
    }

    #[test]
    fn scheduler_owned_pause_events_follow_authoritative_transition_order() {
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
        let mut events = state.events.subscribe();
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );

        assert!(coordinator.set_queue_paused_and_publish(true).unwrap());
        assert!(coordinator.set_queue_paused_and_publish(false).unwrap());

        let mut queue_transitions = Vec::new();
        loop {
            match events.try_recv() {
                Ok(mold_core::ServerEvent::QueuePaused) => queue_transitions.push(true),
                Ok(mold_core::ServerEvent::QueueResumed) => queue_transitions.push(false),
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                Err(error) => panic!("unexpected event receive failure: {error}"),
            }
        }
        assert_eq!(
            queue_transitions,
            vec![true, false],
            "queue transition events must match serialized plan publication order"
        );
    }

    #[test]
    fn queue_control_plan_projects_the_complete_committed_replan_window() {
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
        let mut immediate = false;
        coordinator.reconcile_external_mutations(&mut immediate);
        coordinator.dirty.clear_through(coordinator.state_version);
        assert!(
            coordinator.dirty.dirty_since.is_none(),
            "test must begin from a clean replan window"
        );

        assert!(coordinator.set_queue_paused_and_publish(true).unwrap());

        let published = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("pause transition must publish queue authority");
        let published_start = published
            .dirty_since_unix_ms
            .expect("transition plan must publish its dirty-window start");
        let published_deadline = published
            .next_replan_at_unix_ms
            .expect("transition plan must publish its replan deadline");
        let committed_start = coordinator
            .dirty
            .dirty_since
            .expect("transition must commit its dirty-window start");
        let committed_deadline = coordinator
            .dirty
            .deadline()
            .expect("transition must commit its replan deadline");
        let published_window_ms = published_deadline.saturating_sub(published_start);
        let committed_window_ms: u64 = committed_deadline
            .saturating_duration_since(committed_start)
            .as_millis()
            .try_into()
            .unwrap();

        assert_eq!(published.state_version, coordinator.state_version);
        assert!(
            published_window_ms.abs_diff(committed_window_ms) <= 1,
            "published and committed replan windows must describe the same interval"
        );
    }

    #[tokio::test]
    async fn zero_gpu_inventory_keeps_chain_stage_pending_without_fake_device_or_lease() {
        let pool = Arc::new(GpuPool {
            workers: Vec::new().into(),
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
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "chain-zero:stage:0",
                "ltx2",
                1 << 30,
                OwnerWork::Probe {
                    id: "chain-zero:stage:0".to_string(),
                    kind: mold_scheduler::WorkKind::ChainStage,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );

        coordinator.dispatch_ready().await;

        assert!(coordinator.leases.is_empty());
        assert!(coordinator
            .pending_owner_work
            .contains_key("chain-zero:stage:0"));
    }

    #[tokio::test]
    async fn chain_sticky_affinity_is_a_preference_and_moves_when_previous_gpu_is_busy() {
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
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "chain-a:stage:1",
                "ltx2:stage-plan",
                1 << 30,
                OwnerWork::Probe {
                    id: "chain-a:stage:1".to_string(),
                    kind: mold_scheduler::WorkKind::ChainStage,
                    run: Box::new(|| {}),
                },
            )
            .with_preferred_ordinal(Some(0)),
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

        assert!(matches!(
            worker_a_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        match worker_b_rx.recv().unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => {
                assert_eq!(grant.work.id(), "chain-a:stage:1");
            }
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        }
    }

    #[tokio::test]
    async fn chain_sticky_affinity_breaks_equal_cost_ties_without_becoming_a_pin() {
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
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "chain-b:stage:1",
                "ltx2:stage-plan",
                1 << 30,
                OwnerWork::Probe {
                    id: "chain-b:stage:1".to_string(),
                    kind: mold_scheduler::WorkKind::ChainStage,
                    run: Box::new(|| {}),
                },
            )
            .with_preferred_ordinal(Some(1)),
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

        assert!(matches!(
            worker_a_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        match worker_b_rx.recv().unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => {
                assert_eq!(grant.work.id(), "chain-b:stage:1");
            }
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        }
    }

    #[tokio::test]
    async fn queued_chain_stage_cancellation_removes_it_before_any_worker_grant() {
        let (worker, worker_rx) = test_worker(0);
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
        let (generation, _) = fake_generation("chain-request-template");
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancellation_probe = cancelled.clone();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "chain-cancel:stage:0",
                "ltx2",
                1 << 30,
                OwnerWork::ChainStage(Box::new(crate::chain_job_runner::ScheduledChainStageWork {
                    id: "chain-cancel:stage:0".to_string(),
                    model: "ltx2".to_string(),
                    cache_key: "ltx2".to_string(),
                    config: mold_core::Config::default(),
                    stage_req: generation.request,
                    carry: None,
                    motion_tail_frames: 1,
                    progress: Arc::new(|_, _| std::ops::ControlFlow::Continue(())),
                    cancelled: Arc::new(move || cancellation_probe.load(Ordering::SeqCst)),
                    cancellation: mold_inference::InferenceCancellationToken::default(),
                    on_leased: None,
                    execution_plan: None,
                    expected_model_fingerprint: None,
                    result_tx: Some(result_tx),
                    before_second_fence: None,
                })),
            ),
            &mut immediate,
        );
        cancelled.store(true, Ordering::SeqCst);

        coordinator.reconcile_external_mutations(&mut immediate);

        assert!(coordinator.pending_owner_work.is_empty());
        let execution = result_rx.await.unwrap().unwrap();
        assert!(matches!(
            execution.outcome,
            crate::chain_job_runner::StageRenderOutcome::Cancelled
        ));
        assert!(matches!(
            worker_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn chain_plan_invalidation_requeues_same_stage_id_with_backoff_and_open_result() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer.safetensors");
        let vae = root.path().join("vae.safetensors");
        std::fs::write(&transformer, b"weights").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        let model = "test-chain";
        let mut config = mold_core::Config::default();
        let frozen_config = mold_core::ModelConfig {
            transformer: Some(transformer.display().to_string()),
            vae: Some(vae.display().to_string()),
            family: Some("flux".to_string()),
            ..mold_core::ModelConfig::default()
        };
        let expected_model_fingerprint =
            crate::execution_plan::frozen_model_fingerprint(model, &frozen_config).unwrap();
        config.install_frozen_model_config(model, frozen_config);
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(config.clone(), QueueHandle::new(ingress_tx), pool, 1);
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let (generation, _) = fake_generation("chain-request");
        let mut stage_req = generation.request;
        stage_req.model = model.to_string();
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let stage_id = "chain:durable-parent:stage:0";
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                stage_id,
                expected_model_fingerprint.clone(),
                1 << 30,
                OwnerWork::ChainStage(Box::new(crate::chain_job_runner::ScheduledChainStageWork {
                    id: stage_id.to_string(),
                    model: model.to_string(),
                    cache_key: model.to_string(),
                    config,
                    stage_req,
                    carry: None,
                    motion_tail_frames: 1,
                    progress: Arc::new(|_, _| std::ops::ControlFlow::Continue(())),
                    cancelled: Arc::new(|| false),
                    cancellation: mold_inference::InferenceCancellationToken::default(),
                    on_leased: None,
                    execution_plan: None,
                    expected_model_fingerprint: Some(expected_model_fingerprint),
                    result_tx: Some(result_tx),
                    before_second_fence: None,
                })),
            ),
            &mut immediate,
        );
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
        let first_grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        assert_eq!(first_grant.fence.work_id, stage_id);
        assert!(matches!(
            &first_grant.work,
            OwnerWork::ChainStage(stage) if stage.execution_plan.is_some()
        ));

        coordinator.handle_worker_event(
            WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                grant: first_grant,
                reason: LeaseRejection::PlanInvalidated(
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "sample changed before acceptance".to_string(),
                    ),
                ),
            },
            &mut immediate,
        );
        let retry = coordinator
            .pending_owner_work
            .get_mut(stage_id)
            .expect("same durable stage is requeued");
        assert!(retry
            .retry_not_before_ms
            .is_some_and(|at| at > monotonic_ms()));
        retry.retry_not_before_ms = Some(0);
        assert!(matches!(
            result_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));

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
        let second_grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        assert_eq!(second_grant.fence.work_id, stage_id);
        assert!(second_grant.fence.plan_version > 0);
        match second_grant.work {
            OwnerWork::ChainStage(mut stage) => {
                assert!(stage
                    .result_tx
                    .take()
                    .unwrap()
                    .send(Ok(crate::chain_job_runner::StageExecution {
                        outcome: crate::chain_job_runner::StageRenderOutcome::Cancelled,
                        device_ordinal: Some(0),
                    }))
                    .is_ok());
            }
            _ => panic!("expected chain stage"),
        }
        let settled = result_rx.await.unwrap().unwrap();
        assert!(matches!(
            settled.outcome,
            crate::chain_job_runner::StageRenderOutcome::Cancelled
        ));
    }

    #[tokio::test]
    async fn continuously_arriving_generations_cannot_starve_a_chain_stage() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            16,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let stage_id = "chain:waiting-parent:stage:0";
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                stage_id,
                "chain-model",
                1 << 30,
                OwnerWork::Probe {
                    id: stage_id.to_string(),
                    kind: mold_scheduler::WorkKind::ChainStage,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );
        let device_id = worker_device_id(&worker);
        let mut stage_granted_on = None;
        for round in 0..5 {
            let id = format!("ordinary-{round}");
            let (job, _result) = fake_generation(&id);
            state.job_registry.register(&id, &job.request.model);
            coordinator.enqueue(job, &mut immediate);
            coordinator.pending.get_mut(&id).unwrap().preparation = PreparationState::Ready;
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
            let grant = match worker_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
                crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
                crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
                crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
            };
            if grant.fence.work_id == stage_id {
                stage_granted_on = Some(round);
                break;
            }
            assert_eq!(grant.fence.work_id, id);
            worker.release_in_flight();
            coordinator.handle_worker_event(
                WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: 0,
                    owner_epoch: 1,
                    worker_generation: 1,
                    successful: true,
                    phase_timings: EstimatePhaseTimings::default(),
                    completion: None,
                },
                &mut immediate,
            );
        }
        assert!(
            stage_granted_on.is_some_and(|round| round <= 3),
            "owner bypass must force eventual progress under continuous ordinary arrivals"
        );
    }

    #[test]
    fn generation_reorder_preserves_shared_owner_work_rank_slots() {
        let ranks = reordered_generation_ranks(
            [
                ("generation-a".to_string(), 0),
                ("generation-b".to_string(), 2),
            ],
            &["generation-b".to_string(), "generation-a".to_string()],
        );

        assert_eq!(ranks["generation-b"], 0);
        assert_eq!(ranks["generation-a"], 2);
        let owner_rank = 1;
        assert!(
            ranks["generation-b"] < owner_rank && owner_rank < ranks["generation-a"],
            "reordering generations must not collapse their ranks ahead of owner work"
        );
    }

    #[tokio::test]
    async fn durable_position_insertion_dispatches_before_earlier_transport_rows() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(3);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 3);

        state.job_registry.register("hydrated-a", "flux-dev:q4");
        state.job_registry.register("hydrated-b", "flux-dev:q4");
        state.job_registry.register_job_at_queued_position(
            "deep-zero",
            "flux-dev:q4",
            None,
            None,
            None,
            0,
        );

        let mut results = Vec::new();
        for id in ["hydrated-a", "hydrated-b", "deep-zero"] {
            let (job, result) = fake_generation(id);
            results.push(result);
            queue.submit(job, 3).await.unwrap();
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
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;

        assert_eq!(
            recv_grant(&worker_rx).id,
            "deep-zero",
            "scheduler dispatch follows the registry's durable position, not channel arrival"
        );
        drop(results);
    }

    #[test]
    fn internal_chain_stage_does_not_shift_legacy_queue_positions_or_patch_order() {
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
        state.job_registry.register("generation-a", "flux");
        state.job_registry.register("generation-b", "flux");
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "chain-parent:stage:0",
                "ltx2",
                1 << 30,
                OwnerWork::Probe {
                    id: "chain-parent:stage:0".to_string(),
                    kind: mold_scheduler::WorkKind::ChainStage,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );

        let before = coordinator.state.job_registry.snapshot();
        assert_eq!(
            before
                .entries
                .iter()
                .map(|entry| (entry.id.as_str(), entry.position))
                .collect::<Vec<_>>(),
            vec![("generation-a", 0), ("generation-b", 1)]
        );
        coordinator
            .state
            .job_registry
            .reorder_queued("generation-b", 0)
            .unwrap();
        let after = coordinator.state.job_registry.snapshot();
        assert_eq!(
            after
                .entries
                .iter()
                .map(|entry| (entry.id.as_str(), entry.position))
                .collect::<Vec<_>>(),
            vec![("generation-b", 0), ("generation-a", 1)]
        );
        assert!(coordinator
            .pending_owner_work
            .contains_key("chain-parent:stage:0"));
    }

    #[tokio::test]
    async fn stage_boundary_releases_lease_and_interleaves_older_ordinary_work() {
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
        let probe = |id: &str, kind| {
            ScheduledOwnerWork::new(
                id,
                "same-plan",
                1 << 30,
                OwnerWork::Probe {
                    id: id.to_string(),
                    kind,
                    run: Box::new(|| {}),
                },
            )
        };
        coordinator.enqueue_owner_work(
            probe("chain:stage:0", mold_scheduler::WorkKind::ChainStage),
            &mut immediate,
        );
        coordinator.enqueue_owner_work(
            probe(
                "ordinary-upscale",
                mold_scheduler::WorkKind::StandaloneUpscale,
            ),
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
        coordinator.dispatch_ready().await;
        assert_eq!(recv_owner_grant_id(&worker_rx), "chain:stage:0");

        worker.release_in_flight();
        coordinator.handle_worker_event(
            WorkerEvent::Completed {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                successful: true,
                phase_timings: EstimatePhaseTimings::default(),
                completion: None,
            },
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 2,
            },
            &mut immediate,
        );
        coordinator.enqueue_owner_work(
            probe("chain:stage:1", mold_scheduler::WorkKind::ChainStage),
            &mut immediate,
        );
        coordinator.dispatch_ready().await;

        assert_eq!(recv_owner_grant_id(&worker_rx), "ordinary-upscale");
        assert!(coordinator.pending_owner_work.contains_key("chain:stage:1"));
    }

    #[tokio::test]
    async fn queued_chain_completion_cannot_release_actor_while_owner_ingress_wins_first() {
        let (stage_worker, _stage_worker_rx) = test_worker(0);
        let (utility_worker, utility_worker_rx) = test_worker(1);
        let stage_device_id = worker_device_id(&stage_worker);
        let utility_device_id = worker_device_id(&utility_worker);
        let pool = Arc::new(GpuPool {
            workers: vec![stage_worker, utility_worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            2,
        );
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let stage_id = "chain:parent:attempt:1:stage:0";
        coordinator.leases.insert(
            stage_device_id.clone(),
            ActiveLease {
                work_id: stage_id.to_string(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: 100,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey::default(),
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new(stage_id, 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
            },
        );
        coordinator.memory.reservations.insert(
            stage_id.to_string(),
            HostReservation {
                bytes: 1 << 30,
                state: ReservationState::CommittedAfterSample {
                    commit_sequence: coordinator.memory.sequence,
                },
                charge_until_release: false,
            },
        );

        let (actor_result_tx, actor_result_rx) = tokio::sync::oneshot::channel();
        let completion = crate::gpu_worker::DeferredOwnerCompletion::ChainStage {
            tx: Some(actor_result_tx),
            result: Some(Ok(crate::chain_job_runner::StageExecution {
                outcome: crate::chain_job_runner::StageRenderOutcome::Cancelled,
                device_ordinal: Some(0),
            })),
        };
        let queued_completion = WorkerEvent::Completed {
            device_id: stage_device_id.clone(),
            ordinal: 0,
            owner_epoch: 1,
            worker_generation: 1,
            successful: false,
            phase_timings: EstimatePhaseTimings::default(),
            completion: Some(Box::new(completion)),
        };
        let (actor_advanced_tx, mut actor_advanced_rx) = tokio::sync::mpsc::unbounded_channel();
        let actor = tokio::spawn(async move {
            let execution = actor_result_rx
                .await
                .expect("coordinator owns the actor reply")
                .expect("cancelled stage is a typed result");
            assert!(matches!(
                execution.outcome,
                crate::chain_job_runner::StageRenderOutcome::Cancelled
            ));
            actor_advanced_tx.send("stage:1").unwrap();
        });

        // Deterministically emulate tokio::select! choosing owner_work_rx
        // while the Completed event is already queued on worker_rx.
        let mut immediate = false;
        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "utility",
                "utility",
                1 << 30,
                OwnerWork::Probe {
                    id: "utility".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: utility_device_id,
                ordinal: 1,
                owner_epoch: 1,
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.dispatch_ready().await;
        assert_eq!(recv_owner_grant_id(&utility_worker_rx), "utility");
        assert!(matches!(
            actor_advanced_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        assert!(coordinator.leases.contains_key(&stage_device_id));
        assert!(coordinator.memory.reservations.contains_key(stage_id));

        coordinator.handle_worker_event(queued_completion, &mut immediate);

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), actor_advanced_rx.recv())
                .await
                .unwrap(),
            Some("stage:1")
        );
        actor.await.unwrap();
        assert!(!coordinator.leases.contains_key(&stage_device_id));
        assert!(!coordinator.memory.reservations.contains_key(stage_id));
        let published = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("actor acknowledgement requires an authoritative published plan");
        assert_eq!(published.state_version, coordinator.state_version);
        assert!(
            published
                .work_items
                .iter()
                .all(|item| item.work_id != stage_id),
            "settled stage must be absent from the published plan before actor acknowledgement"
        );
    }

    #[tokio::test]
    async fn chain_success_and_error_results_settle_only_after_authoritative_completion() {
        for successful in [true, false] {
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
            let stage_id = format!("chain:result:{successful}:attempt:1:stage:0");
            coordinator.leases.insert(
                device_id.clone(),
                ActiveLease {
                    work_id: stage_id.clone(),
                    owner_epoch: 1,
                    plan_version: 1,
                    worker_generation: 1,
                    accepted: true,
                    previous_target: None,
                    estimated_finish_ms: 100,
                    ready_at_ms: 0,
                    bypass_count: 0,
                    warm_wait_started_ms: None,
                    started_at: Instant::now(),
                    estimate_key: EstimateKey::default(),
                    vram_high_water_bytes: None,
                    host_incremental_high_water_bytes: None,
                    fallback_reason: None,
                    projection: WorkSnapshot::new(stage_id.clone(), 0, Vec::new()),
                    assignment_reason: AssignmentReason::Priority,
                },
            );
            coordinator.memory.reservations.insert(
                stage_id,
                HostReservation {
                    bytes: 1 << 30,
                    state: ReservationState::CommittedAfterSample {
                        commit_sequence: coordinator.memory.sequence,
                    },
                    charge_until_release: false,
                },
            );
            let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
            let result = if successful {
                Ok(crate::chain_job_runner::StageExecution {
                    outcome: crate::chain_job_runner::StageRenderOutcome::Done(
                        mold_inference::chain::StageOutcome {
                            frames: Vec::new(),
                            tail: mold_inference::chain::ChainTail {
                                frames: 0,
                                tail_rgb_frames: Vec::new(),
                            },
                            audio: None,
                            hdr_frames_written: None,
                            generation_time_ms: 1,
                        },
                    ),
                    device_ordinal: Some(0),
                })
            } else {
                Err("typed stage render failure".to_string())
            };
            let completion = crate::gpu_worker::DeferredOwnerCompletion::ChainStage {
                tx: Some(result_tx),
                result: Some(result),
            };
            assert!(matches!(
                result_rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ));

            let mut immediate = false;
            coordinator.handle_worker_event(
                WorkerEvent::Completed {
                    device_id: device_id.clone(),
                    ordinal: 0,
                    owner_epoch: 1,
                    worker_generation: 1,
                    successful,
                    phase_timings: EstimatePhaseTimings::default(),
                    completion: Some(Box::new(completion)),
                },
                &mut immediate,
            );

            let settled = result_rx
                .await
                .expect("completion sends exactly one result");
            if successful {
                assert!(matches!(
                    settled.unwrap().outcome,
                    crate::chain_job_runner::StageRenderOutcome::Done(_)
                ));
            } else {
                match settled {
                    Ok(_) => panic!("failed stage must retain its typed error"),
                    Err(error) => assert_eq!(error, "typed stage render failure"),
                }
            }
            assert!(!coordinator.leases.contains_key(&device_id));
            assert!(coordinator.memory.reservations.is_empty());
        }
    }

    #[tokio::test]
    async fn chain_completion_fails_closed_when_current_state_plan_cannot_be_published() {
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
        let stage_id = "chain:publication-failure:attempt:1:stage:0";
        coordinator.leases.insert(
            device_id.clone(),
            ActiveLease {
                work_id: stage_id.to_string(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: 100,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey::default(),
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new(stage_id, 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
            },
        );
        coordinator.memory.reservations.insert(
            stage_id.to_string(),
            HostReservation {
                bytes: 1 << 30,
                state: ReservationState::CommittedAfterSample {
                    commit_sequence: coordinator.memory.sequence,
                },
                charge_until_release: false,
            },
        );
        coordinator.replan_and_publish();
        let plan_before_completion = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("pre-completion lease plan");
        assert!(plan_before_completion
            .work_items
            .iter()
            .any(|item| item.work_id == stage_id));

        coordinator.next_plan_error = Some(PlannerError::DuplicateWorkId {
            work_id: WorkId::new("injected-plan-publication-failure"),
        });
        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let completion = crate::gpu_worker::DeferredOwnerCompletion::ChainStage {
            tx: Some(result_tx),
            result: Some(Ok(crate::chain_job_runner::StageExecution {
                outcome: crate::chain_job_runner::StageRenderOutcome::Cancelled,
                device_ordinal: Some(0),
            })),
        };
        assert!(matches!(
            result_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        let (successor_tx, mut successor_rx) = tokio::sync::mpsc::unbounded_channel();
        let actor = tokio::spawn(async move {
            let result = result_rx
                .await
                .expect("coordinator must settle the actor without hanging");
            if result.is_ok() {
                successor_tx.send("stage:1").unwrap();
                successor_tx.send("finalize").unwrap();
            }
            result
        });

        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::Completed {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                successful: true,
                phase_timings: EstimatePhaseTimings::default(),
                completion: Some(Box::new(completion)),
            },
            &mut immediate,
        );

        assert!(!coordinator.leases.contains_key(&device_id));
        assert!(!coordinator.memory.reservations.contains_key(stage_id));
        let plan_after_failed_publication = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("last authoritative plan remains available");
        assert_eq!(plan_after_failed_publication, plan_before_completion);
        assert!(
            plan_after_failed_publication.state_version < coordinator.state_version,
            "failed planning must not masquerade as a current-state publication"
        );
        let actor_result = tokio::time::timeout(Duration::from_secs(1), actor)
            .await
            .expect("planner failure must not hang the actor")
            .expect("actor task must not panic");
        let error = match actor_result {
            Ok(_) => panic!("actor must not advance after publication failure"),
            Err(error) => error,
        };
        assert!(error.contains("scheduler"));
        assert!(error.contains("publish"));
        assert!(
            successor_rx.try_recv().is_err(),
            "failed publication must not produce a successor or finalize message"
        );

        coordinator.enqueue_owner_work(
            ScheduledOwnerWork::new(
                "unrelated-recovery-work",
                "unrelated-recovery-work",
                1 << 30,
                OwnerWork::Probe {
                    id: "unrelated-recovery-work".to_string(),
                    kind: mold_scheduler::WorkKind::StandaloneUpscale,
                    run: Box::new(|| {}),
                },
            ),
            &mut immediate,
        );
        coordinator.replan_and_publish();
        let recovered = coordinator
            .state
            .scheduled_work
            .latest_plan()
            .expect("one-shot injected failure must not poison later replanning");
        assert_eq!(recovered.state_version, coordinator.state_version);
        assert!(recovered.plan_version > plan_before_completion.plan_version);
        assert!(recovered
            .work_items
            .iter()
            .any(|item| item.work_id == "unrelated-recovery-work"));
    }

    #[tokio::test]
    async fn chain_completion_fails_closed_on_queue_plan_authority_conflict() {
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
        let stage_id = "chain:authority-conflict:attempt:1:stage:0";
        coordinator.leases.insert(
            device_id.clone(),
            ActiveLease {
                work_id: stage_id.to_string(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
                estimated_finish_ms: 100,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey::default(),
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new(stage_id, 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
            },
        );
        coordinator.memory.reservations.insert(
            stage_id.to_string(),
            HostReservation {
                bytes: 1 << 30,
                state: ReservationState::CommittedAfterSample {
                    commit_sequence: coordinator.memory.sequence,
                },
                charge_until_release: false,
            },
        );
        let conflicting_authority = mold_core::QueuePlan {
            plan_version: 50,
            state_version: 50,
            ..Default::default()
        };
        *coordinator
            .state
            .scheduled_work
            .latest_plan
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(conflicting_authority.clone());

        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let completion = crate::gpu_worker::DeferredOwnerCompletion::ChainStage {
            tx: Some(result_tx),
            result: Some(Ok(crate::chain_job_runner::StageExecution {
                outcome: crate::chain_job_runner::StageRenderOutcome::Cancelled,
                device_ordinal: Some(0),
            })),
        };
        let (successor_tx, mut successor_rx) = tokio::sync::mpsc::unbounded_channel();
        let actor = tokio::spawn(async move {
            let result = result_rx
                .await
                .expect("authority conflict must settle the actor without hanging");
            if result.is_ok() {
                successor_tx.send("stage:1").unwrap();
                successor_tx.send("finalize").unwrap();
            }
            result
        });

        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::Completed {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                successful: true,
                phase_timings: EstimatePhaseTimings::default(),
                completion: Some(Box::new(completion)),
            },
            &mut immediate,
        );

        assert!(!coordinator.leases.contains_key(&device_id));
        assert!(!coordinator.memory.reservations.contains_key(stage_id));
        assert_eq!(
            coordinator.state.scheduled_work.latest_plan(),
            Some(conflicting_authority),
            "a conflicting current authority must never be overwritten"
        );
        let actor_result = tokio::time::timeout(Duration::from_secs(1), actor)
            .await
            .expect("authority conflict must not hang the actor")
            .expect("actor task must not panic");
        let error = match actor_result {
            Ok(_) => panic!("actor must not advance through an authority conflict"),
            Err(error) => error,
        };
        assert!(error.contains("scheduler"));
        assert!(error.contains("conflicts"));
        assert!(
            successor_rx.try_recv().is_err(),
            "authority conflict must not produce a successor or finalize message"
        );
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
        let upscale_fixture = tempfile::tempdir().unwrap();
        let upscale_weights = upscale_fixture.path().join("upscaler.safetensors");
        std::fs::write(&upscale_weights, b"fixture").unwrap();
        let frozen_upscale_plan = mold_inference::upscaler::resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &upscale_weights,
            None,
            mold_inference::upscaler::ExactUpscalePlacement::Cpu,
        )
        .unwrap();
        let followup = ScheduledOwnerWork::new(
            child_id.clone(),
            "real-esrgan-x4plus:fp16",
            2 << 30,
            OwnerWork::PostUpscale(Box::new(crate::gpu_pool::PostGenerationUpscaleJob {
                id: child_id.clone(),
                generation: Box::new(gpu_job),
                response: mold_core::GenerateResponse {
                    request_warnings: Vec::new(),
                    audio: None,
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
                output_metadata: None,
                cancellation: mold_inference::InferenceCancellationToken::default(),
                execution_plan: None,
            })),
        )
        .with_utility_plans(vec![UtilityExecutionPlan::Upscale(frozen_upscale_plan)]);
        let sampler =
            ScriptedHostMemorySampler::new([memory_reading(128, 112), memory_reading(128, 104)]);
        let mut coordinator = Coordinator::with_preparer_and_sampler(
            state,
            Arc::new(ImmediatePreparer),
            sampler.clone(),
            mold_core::config::SchedulerSettings::default(),
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
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new("parent", 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
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
                phase_timings: EstimatePhaseTimings::default(),
                completion: None,
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
        assert!(
            coordinator.estimates.buckets().all(|bucket| {
                bucket.sample_count == 0 && bucket.failure_count == 1 && bucket.ewma_total_ms == 0.0
            }),
            "failed owner work must be counted without training ETA"
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

        assert!(
            !coordinator.pending_owner_work.contains_key(&child_id),
            "ready follow-up must leave the pending set after dispatch"
        );
        assert_eq!(
            coordinator.leases.len(),
            1,
            "ready follow-up must own one distinct lease after dispatch"
        );
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
    async fn rejected_v2_post_upscale_preserves_f0_and_ignores_late_completion() {
        let root = tempfile::tempdir().unwrap();
        let output_dir = root.path().join("gallery");
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, b"frozen-before-owner-validation").unwrap();
        let plan = mold_inference::upscaler::resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            mold_inference::upscaler::ExactUpscalePlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 0,
            },
        )
        .unwrap();

        let (worker, _worker_rx) = test_worker(0);
        let device_id = worker_device_id(&worker);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(ingress_tx);
        let (slot_tx, _slot_rx) = tokio::sync::oneshot::channel();
        let (slot_job, _) = fake_generation("held-slot");
        queue
            .submit(
                GenerationJob {
                    result_tx: slot_tx,
                    ..slot_job
                },
                1,
            )
            .await
            .unwrap();
        let _held_slot = ingress_rx.try_recv().unwrap();
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 1);
        state.job_registry.register("v2-f0-parent", "flux-dev:q4");

        let (mut generation, mut result) = fake_generation("v2-f0-parent");
        generation.output_dir = Some(output_dir.clone());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        generation.progress_tx = Some(progress_tx);
        let fence = LeaseFence {
            work_id: "v2-f0-parent::post-upscale".to_string(),
            device_id: device_id.clone(),
            owner_epoch: 1,
            state_version: 1,
            plan_version: 1,
            worker_generation: 1,
            memory_sample_generation: 1,
            memory_ledger_sequence: 1,
        };
        let gpu_job = gpu_job_from_generation(&state, generation, fence.clone(), None, None);
        let original = mold_core::ImageData {
            data: vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
            format: mold_core::OutputFormat::Png,
            width: 64,
            height: 64,
            index: 0,
        };
        let post = OwnerWork::PostUpscale(Box::new(crate::gpu_pool::PostGenerationUpscaleJob {
            id: fence.work_id.clone(),
            generation: Box::new(gpu_job),
            response: mold_core::GenerateResponse {
                request_warnings: Vec::new(),
                audio: None,
                images: Vec::new(),
                video: None,
                generation_time_ms: 1,
                model: "flux-dev:q4".to_string(),
                seed_used: 7,
                gpu: Some(0),
            },
            image: original.clone(),
            output_metadata: None,
            cancellation: mold_inference::InferenceCancellationToken::default(),
            execution_plan: Some(plan),
        }));
        std::fs::remove_file(&weights).unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state.clone(),
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        coordinator.leases.insert(
            device_id.clone(),
            ActiveLease {
                work_id: fence.work_id.clone(),
                owner_epoch: 1,
                plan_version: 1,
                worker_generation: 1,
                accepted: false,
                previous_target: None,
                estimated_finish_ms: 1,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                started_at: Instant::now(),
                estimate_key: EstimateKey::default(),
                vram_high_water_bytes: None,
                host_incremental_high_water_bytes: None,
                fallback_reason: None,
                projection: WorkSnapshot::new(fence.work_id.clone(), 0, Vec::new()),
                assignment_reason: AssignmentReason::Priority,
            },
        );

        let mut immediate = false;
        coordinator.handle_worker_event(
            WorkerEvent::Rejected {
                device_id: device_id.clone(),
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                grant: Box::new(LeaseGrant {
                    fence,
                    work: post,
                    retry: None,
                }),
                reason: LeaseRejection::PlanInvalidated(
                    crate::execution_plan::ExecutionPlanError::PlanInvalidated(
                        "upscale weights disappeared".to_string(),
                    ),
                ),
            },
            &mut immediate,
        );

        let completed = tokio::time::timeout(Duration::from_secs(2), &mut result)
            .await
            .expect("V2 rejection settles the parent promptly")
            .expect("result sender remains alive")
            .expect("completed F0 is not converted into a generation error");
        assert_eq!(completed.image.data, original.data);
        tokio::time::timeout(Duration::from_secs(2), async {
            while queue.pending() != 0 || !state.job_registry.snapshot().entries.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("post-upscale fallback cleanup settles");
        assert_eq!(queue.pending(), 0);
        assert!(state.job_registry.snapshot().entries.is_empty());
        assert!(coordinator.leases.is_empty());
        let files_before = std::fs::read_dir(&output_dir)
            .unwrap()
            .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
            .count();
        assert_eq!(files_before, 1, "F0 is published exactly once");
        let events = std::iter::from_fn(|| progress_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            SseMessage::Progress(mold_core::SseProgressEvent::Info { message })
                if message.contains("post-generation upscale failed")
        )));
        assert!(events
            .iter()
            .all(|event| !matches!(event, SseMessage::Error(_))));
        assert!(matches!(events.last(), Some(SseMessage::Complete(_))));

        coordinator.handle_worker_event(
            WorkerEvent::Completed {
                device_id,
                ordinal: 0,
                owner_epoch: 1,
                worker_generation: 1,
                successful: false,
                phase_timings: EstimatePhaseTimings::default(),
                completion: None,
            },
            &mut immediate,
        );
        assert_eq!(
            std::fs::read_dir(&output_dir)
                .unwrap()
                .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
                .count(),
            files_before,
            "a late completion cannot publish or settle the parent twice"
        );
        assert_eq!(queue.pending(), 0);
    }

    #[tokio::test]
    async fn missing_cpu_owner_rejection_preserves_post_upscale_f0() {
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            Arc::new(GpuPool {
                workers: Vec::new().into(),
            }),
            1,
        );
        state
            .job_registry
            .register("cpu-owner-parent", "flux-dev:q4");
        let (mut generation, mut result) = fake_generation("cpu-owner-parent");
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        generation.progress_tx = Some(progress_tx);
        let child_id = "cpu-owner-parent::post-upscale".to_string();
        let fence = LeaseFence {
            work_id: child_id.clone(),
            device_id: CPU_UTILITY_DEVICE_ID.to_string(),
            owner_epoch: 1,
            state_version: 1,
            plan_version: 1,
            worker_generation: 1,
            memory_sample_generation: 1,
            memory_ledger_sequence: 1,
        };
        let gpu_job = gpu_job_from_generation(&state, generation, fence, None, None);
        let original = mold_core::ImageData {
            data: vec![1, 2, 3],
            format: mold_core::OutputFormat::Png,
            width: 64,
            height: 64,
            index: 0,
        };
        let work = OwnerWork::PostUpscale(Box::new(crate::gpu_pool::PostGenerationUpscaleJob {
            id: child_id.clone(),
            generation: Box::new(gpu_job),
            response: mold_core::GenerateResponse {
                request_warnings: Vec::new(),
                audio: None,
                images: Vec::new(),
                video: None,
                generation_time_ms: 1,
                model: "flux-dev:q4".to_string(),
                seed_used: 9,
                gpu: Some(0),
            },
            image: original.clone(),
            output_metadata: None,
            cancellation: mold_inference::InferenceCancellationToken::default(),
            execution_plan: None,
        }));

        reject_owner_work_preserving_completed_generation(
            work,
            "CPU utility lane is unavailable".to_string(),
        );

        let completed = tokio::time::timeout(Duration::from_secs(2), &mut result)
            .await
            .expect("missing CPU owner fallback settles promptly")
            .expect("result sender remains alive")
            .expect("missing CPU utility owner cannot discard F0");
        assert_eq!(completed.image.data, original.data);
        tokio::time::timeout(Duration::from_secs(2), async {
            while !state.job_registry.snapshot().entries.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("missing CPU owner cleanup settles");
        let events = std::iter::from_fn(|| progress_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            SseMessage::Progress(mold_core::SseProgressEvent::Info { message })
                if message.contains("CPU utility lane is unavailable")
        )));
        assert!(events
            .iter()
            .all(|event| !matches!(event, SseMessage::Error(_))));
        assert!(matches!(events.last(), Some(SseMessage::Complete(_))));
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
        let owner_plan_cache = coordinator.owner_plan_cache_and_settle_errors();
        let (_, original_catalog) = coordinator.planner_snapshot(&owner_plan_cache);
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
        // `b"new-transformer"` is the same length as `b"old-transformer"`, so an
        // in-place rewrite is only visible through `ctime`, which advances on a
        // coarse (~98 ms) clock. The awaits above usually separate the two
        // writes far enough, but nothing guarantees it. Replace the file so the
        // inode differs and the replan is observed deterministically.
        crate::execution_plan::replace_artifact_bytes(&transformer, b"new-transformer");
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
            mold_core::config::SchedulerSettings::default(),
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
        assert!(
            !coordinator.pending.contains_key("plan-invalidated"),
            "ready invalidated work must leave the pending set after redispatch"
        );
        assert_eq!(
            coordinator.leases.len(),
            1,
            "redispatched invalidated work must own one lease"
        );
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

    /// An H3 sibling submitted while another render owns transient host RAM
    /// must stay in the scheduler queue. Retrying only after the owner settles
    /// avoids both the old terminal refusal and repeated full artifact hashes
    /// while the capacity fact cannot change.
    #[tokio::test]
    async fn h3_host_shortfall_waits_in_queue_until_the_busy_owner_settles() {
        let (worker, _worker_rx) = test_worker(0);
        let device_id = worker_device_id(&worker);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(2);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 2);
        state
            .job_registry
            .register("waiting-h3", "minimax-h3-fl2va");
        let (job, _result) = fake_generation("waiting-h3");
        queue.submit(job, 2).await.unwrap();

        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let mut immediate = false;
        coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        coordinator
            .pending
            .get_mut("waiting-h3")
            .unwrap()
            .preparation = PreparationState::Preparing;
        worker.in_flight.store(1, Ordering::SeqCst);
        let mut deferred = crate::execution_plan::PreparedExecutionInputs::default();
        deferred
            .host_memory_retry_after_devices
            .insert(device_id.clone());
        coordinator.handle_preparation_event(
            PreparationEvent::Ready {
                work_id: "waiting-h3".to_string(),
                prepared: Box::new(PreparedGeneration {
                    execution_inputs: Some(deferred),
                    ..Default::default()
                }),
            },
            &mut immediate,
        );

        assert!(!coordinator.reset_stale_preparations());
        assert_eq!(
            coordinator.pending["waiting-h3"].preparation,
            PreparationState::Ready
        );
        assert!(matches!(
            coordinator.generation_plans(&coordinator.pending["waiting-h3"]),
            Err(GenerationPlanFailure::Transient(_))
        ));

        worker.in_flight.store(0, Ordering::SeqCst);
        assert!(coordinator.reset_stale_preparations());
        assert_eq!(
            coordinator.pending["waiting-h3"].preparation,
            PreparationState::Preparing
        );
        let event = tokio::time::timeout(Duration::from_secs(1), coordinator.preparation_rx.recv())
            .await
            .expect("settled owner must trigger preparation")
            .expect("preparation event");
        coordinator.handle_preparation_event(event, &mut immediate);
        assert!(coordinator.pending.contains_key("waiting-h3"));
        assert_eq!(
            coordinator.pending["waiting-h3"].preparation,
            PreparationState::Ready
        );
        coordinator.stop_preparations().await;
    }

    impl DependencyPreparer for SelectiveBlockingPreparer {
        fn prepare(
            &self,
            _state: AppState,
            _work_id: String,
            request: crate::queue_media_runtime::ZeroizingGenerateRequest,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
            _context: crate::variant_dependencies::DependencyPreparationContext,
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

    struct ConcurrencyProbePreparer {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        released: Arc<AtomicBool>,
    }

    impl DependencyPreparer for ConcurrencyProbePreparer {
        fn prepare(
            &self,
            _state: AppState,
            _work_id: String,
            _request: crate::queue_media_runtime::ZeroizingGenerateRequest,
            _progress: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
            _context: crate::variant_dependencies::DependencyPreparationContext,
        ) -> PreparationFuture {
            let in_flight = self.in_flight.clone();
            let peak = self.peak.clone();
            let released = self.released.clone();
            Box::pin(async move {
                let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                while !released.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(PreparedGeneration::default())
            })
        }
    }

    /// A burst of submissions must not run one dependency resolution per job.
    ///
    /// Every preparation resolves variants and can start a multi-GB download,
    /// all of it behind a GPU that runs one job at a time. Eight submissions
    /// used to mean eight concurrent resolutions competing for the same disk
    /// and host RAM (#1099).
    #[tokio::test]
    async fn a_submission_burst_bounds_concurrent_dependency_preparations() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, mut ingress_rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(ingress_tx);
        let state = AppState::empty(mold_core::Config::default(), queue.clone(), pool, 16);
        let burst = MAX_CONCURRENT_PREPARATIONS * 3;
        let mut results = Vec::new();
        for index in 0..burst {
            let id = format!("burst-{index}");
            state.job_registry.register(&id, "flux-dev:q4");
            let (job, result) = fake_generation(&id);
            results.push(result);
            queue.submit(job, 16).await.unwrap();
        }
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let released = Arc::new(AtomicBool::new(false));
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ConcurrencyProbePreparer {
                in_flight: in_flight.clone(),
                peak: peak.clone(),
                released: released.clone(),
            }),
            ample_memory(),
        );
        let mut immediate = false;
        for _ in 0..burst {
            coordinator.enqueue(ingress_rx.recv().await.unwrap(), &mut immediate);
        }
        coordinator.start_needed_preparations();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            peak.load(Ordering::SeqCst),
            MAX_CONCURRENT_PREPARATIONS,
            "the burst must saturate the bound and never exceed it"
        );

        released.store(true, Ordering::SeqCst);
        for _ in 0..burst {
            let event =
                tokio::time::timeout(Duration::from_secs(5), coordinator.preparation_rx.recv())
                    .await
                    .expect("every held preparation must still complete")
                    .expect("preparation event");
            coordinator.handle_preparation_event(event, &mut immediate);
        }
        assert!(
            peak.load(Ordering::SeqCst) <= MAX_CONCURRENT_PREPARATIONS,
            "draining must not raise the bound"
        );
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
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24 << 30,
            }],
        );
        assert_eq!(signature, vec![("cuda:1".to_string(), 0)]);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn cold_catalog_overlay_survives_full_coordinator_preview_and_config_refresh() {
        let _env = crate::test_support::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = tempfile::tempdir().unwrap();
        let install_dir = root.path().join("cv-2937936");
        let primary = install_dir.join("flux2/catalog/model.safetensors");
        let text_encoder = root.path().join("companions/qwen3.safetensors");
        let tokenizer = root.path().join("companions/tokenizer.json");
        let vae = root.path().join("companions/vae.safetensors");
        for path in [&primary, &text_encoder, &tokenizer, &vae] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
                let header = br#"{}"#;
                let mut contents = (header.len() as u64).to_le_bytes().to_vec();
                contents.extend_from_slice(header);
                std::fs::write(path, contents).unwrap();
            } else {
                std::fs::write(path, b"{}").unwrap();
            }
        }
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
                id: "cv:2937936".to_string(),
                source: "civitai".to_string(),
                source_id: "2937936".to_string(),
                name: "Catalog preview fixture".to_string(),
                author: None,
                family: "flux2".to_string(),
                family_role: "finetune".to_string(),
                sub_family: Some("klein-9b".to_string()),
                kind: "checkpoint".to_string(),
                modality: "image".to_string(),
                nsfw: None,
                description: None,
                tags: Vec::new(),
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: Some(std::fs::metadata(&primary).unwrap().len()),
                supported: true,
                trained_words: Vec::new(),
                primary_filename_rel: "flux2/catalog/model.safetensors".to_string(),
                primary_size_bytes: None,
                low_noise_filename_rel: None,
                low_noise_size_bytes: None,
                written_at: 0,
            },
        )
        .unwrap();

        let mut cold_config = mold_core::Config {
            models_dir: root.path().display().to_string(),
            qwen3_variant: Some("bf16".to_string()),
            ..Default::default()
        };
        cold_config.models.insert(
            "flux2-te-9b".to_string(),
            mold_core::ModelConfig {
                transformer: Some(text_encoder.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![text_encoder.display().to_string()]),
                text_tokenizer: Some(tokenizer.display().to_string()),
                family: Some("flux2".to_string()),
                ..Default::default()
            },
        );
        cold_config.models.insert(
            "flux2-vae".to_string(),
            mold_core::ModelConfig {
                transformer: Some(vae.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("flux2".to_string()),
                ..Default::default()
            },
        );
        let request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"cv:2937936","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let (worker, _worker_rx) = test_worker(0);
        let stable_id = worker_device_id(&worker);
        let device = crate::execution_plan::DeviceFact {
            id: stable_id.clone(),
            ordinal: 0,
            backend: mold_core::GpuBackend::Cuda,
            compute_capability: Some((8, 6)),
            available_vram_bytes: 24 << 30,
        };
        let prepared = crate::variant_dependencies::prepare_local_execution_inputs(
            &cold_config,
            &request,
            vec![device],
        )
        .await
        .unwrap();
        assert!(prepared.model_config_overlay.is_some());
        assert!(!cold_config.models.contains_key(&request.model));

        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(cold_config.clone(), QueueHandle::new(ingress_tx), pool, 1);
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let before_config = format!("{:?}", *coordinator.state.config.read().await);
        assert!(coordinator.state.catalog_intents.read().await.is_empty());

        let first = coordinator.placement_preview(&request, 1, &prepared);
        assert_eq!(first.outcome, "planned", "{:?}", first.reason);
        let cancellation_checks = std::cell::Cell::new(0_u32);
        let cancelled_after_planning =
            coordinator.placement_preview_cancellable(&request, 1, &prepared, &|| {
                let next = cancellation_checks.get().saturating_add(1);
                cancellation_checks.set(next);
                next >= 9
            });
        assert_eq!(
            cancelled_after_planning.reason.as_deref(),
            Some("placement preview cancelled")
        );
        assert!(
            cancellation_checks.get() >= 9,
            "cancellation must be observed after generation planning starts"
        );
        assert_eq!(
            first
                .candidate
                .as_ref()
                .map(|candidate| candidate.device_id.as_str()),
            Some(stable_id.as_str())
        );
        assert_eq!(
            format!("{:?}", *coordinator.state.config.read().await),
            before_config
        );
        assert!(coordinator.state.catalog_intents.read().await.is_empty());

        // Mirror GET /api/models refreshing the runtime config from disk.
        *coordinator.state.config.write().await = cold_config;
        let after_refresh = coordinator.placement_preview(&request, 1, &prepared);
        assert_eq!(
            after_refresh.outcome, "planned",
            "{:?}",
            after_refresh.reason
        );
        assert_eq!(
            after_refresh
                .candidate
                .as_ref()
                .map(|candidate| candidate.execution_fingerprint.as_str()),
            first
                .candidate
                .as_ref()
                .map(|candidate| candidate.execution_fingerprint.as_str())
        );
        assert!(!coordinator
            .state
            .config
            .read()
            .await
            .models
            .contains_key(&request.model));
        assert!(coordinator.state.catalog_intents.read().await.is_empty());
    }

    #[tokio::test]
    async fn placement_preview_reports_only_the_selected_devices_pending_downloads() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer.safetensors");
        let vae = root.path().join("vae.safetensors");
        let encoder = root.path().join("qwen3.safetensors");
        let tokenizer = root.path().join("tokenizer.json");
        for path in [&transformer, &vae, &encoder, &tokenizer] {
            std::fs::write(path, b"preview").unwrap();
        }
        let mut config = mold_core::Config {
            qwen3_variant: Some("bf16".to_string()),
            ..Default::default()
        };
        config.models.insert(
            "pending-z".to_string(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                text_encoder_files: Some(vec![encoder.display().to_string()]),
                text_tokenizer: Some(tokenizer.display().to_string()),
                family: Some("z-image".to_string()),
                ..Default::default()
            },
        );
        let request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"pending-z","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let (worker0, _worker_rx0) = test_worker(0);
        let (worker1, _worker_rx1) = test_worker(1);
        let stable_id0 = worker_device_id(&worker0);
        let stable_id1 = worker_device_id(&worker1);
        let devices = vec![
            crate::execution_plan::DeviceFact {
                id: stable_id0.clone(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24 << 30,
            },
            crate::execution_plan::DeviceFact {
                id: stable_id1.clone(),
                ordinal: 1,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24 << 30,
            },
        ];
        let mut prepared =
            crate::variant_dependencies::prepare_local_execution_inputs(&config, &request, devices)
                .await
                .unwrap();
        let pending_path = root.path().join("pending-qwen3.gguf");
        let sibling = prepared.by_device.get_mut(&stable_id1).unwrap();
        sibling.engine_paths.text_encoder_files = vec![pending_path.clone()];
        sibling.engine_config.qwen3_variant = Some("q8".to_string());
        sibling.engine_config.selected_qwen3_paths = vec![pending_path.clone()];
        sibling.pending_artifacts.insert(
            pending_path,
            crate::execution_plan::PendingArtifactIdentity {
                kind: "text_encoder".to_string(),
                repo: "owner/qwen3".to_string(),
                filename: "qwen3-q8.gguf".to_string(),
                bytes: 1_000_000_000,
                install_model: None,
                licenses: Vec::new(),
                container: crate::execution_plan::PendingArtifactContainer::Gguf,
                quantization: Some(crate::execution_plan::QuantizationVariant::Q8),
            },
        );

        let pool = Arc::new(GpuPool {
            workers: vec![worker0, worker1].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(config, QueueHandle::new(ingress_tx), pool, 1);
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );

        let clean = coordinator.placement_preview(&request, 1, &prepared);
        assert_eq!(
            clean
                .candidate
                .as_ref()
                .map(|candidate| candidate.device_id.as_str()),
            Some(stable_id0.as_str())
        );
        assert!(
            clean.pending_downloads.is_empty(),
            "a clean selected device must not inherit a sibling's pending download"
        );

        let mixed = coordinator.placement_preview(&request, 2, &prepared);
        assert_eq!(mixed.outcome, "planned");
        assert_eq!(mixed.pending_downloads.len(), 1);
        assert_eq!(mixed.pending_downloads[0].name, "qwen3-q8.gguf");
        assert_eq!(
            mixed.candidate.as_ref().unwrap().estimate_confidence,
            mold_core::QueueEstimateConfidence::Low
        );
        assert_eq!(mixed.stage_candidates.len(), 2);
        assert!(mixed.stage_candidates.iter().any(|stage| {
            stage.candidate.device_id == stable_id1
                && stage.candidate.estimate_confidence == mold_core::QueueEstimateConfidence::Low
        }));

        let pending =
            coordinator.placement_preview_dag_for_device(&request, 1, &prepared, Some(&stable_id1));
        assert_eq!(
            pending
                .candidate
                .as_ref()
                .map(|candidate| candidate.device_id.as_str()),
            Some(stable_id1.as_str())
        );
        assert_eq!(pending.pending_downloads.len(), 1);
        assert_eq!(pending.pending_downloads[0].name, "qwen3-q8.gguf");
        assert_eq!(
            pending.candidate.unwrap().estimate_confidence,
            mold_core::QueueEstimateConfidence::Low
        );
    }

    #[tokio::test]
    async fn exact_placement_preview_is_stable_device_specific_and_read_only() {
        let root = tempfile::tempdir().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "qwen3.safetensors",
            "tokenizer.json",
        ] {
            std::fs::write(root.path().join(name), b"preview").unwrap();
        }
        let mut config = mold_core::Config::default();
        config.models.insert(
            "preview-z".into(),
            mold_core::ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                text_encoder_files: Some(vec![root
                    .path()
                    .join("qwen3.safetensors")
                    .display()
                    .to_string()]),
                text_tokenizer: Some(root.path().join("tokenizer.json").display().to_string()),
                family: Some("z-image".into()),
                ..Default::default()
            },
        );
        let request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"preview-z","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let (worker0, _worker_rx0) = test_worker(0);
        let (worker1, _worker_rx1) = test_worker(1);
        let stable_id = worker_device_id(&worker0);
        let stable_id1 = worker_device_id(&worker1);
        let pool = Arc::new(GpuPool {
            workers: vec![worker0, worker1].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(config.clone(), QueueHandle::new(ingress_tx), pool, 1);
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let prepared = crate::variant_dependencies::prepare_local_execution_inputs(
            &config,
            &request,
            vec![
                crate::execution_plan::DeviceFact {
                    id: stable_id.clone(),
                    ordinal: 0,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 24 << 30,
                },
                crate::execution_plan::DeviceFact {
                    id: stable_id1.clone(),
                    ordinal: 1,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 24 << 30,
                },
            ],
        )
        .await
        .unwrap();
        let device_facts = vec![
            crate::execution_plan::DeviceFact {
                id: stable_id.clone(),
                ordinal: 0,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24 << 30,
            },
            crate::execution_plan::DeviceFact {
                id: stable_id1.clone(),
                ordinal: 1,
                backend: mold_core::GpuBackend::Cuda,
                compute_capability: Some((8, 6)),
                available_vram_bytes: 24 << 30,
            },
        ];
        let execution = crate::execution_plan::resolve_execution_plans_with_prepared(
            &config,
            &request,
            &device_facts,
            false,
            Some(&prepared),
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let worker = coordinator.state.gpu_pool.worker_by_ordinal(0).unwrap();
        coordinator.observe_estimate(
            generation_estimate_key(
                &coordinator.state,
                &worker,
                &request,
                None,
                &execution.execution_fingerprint,
            ),
            EstimateObservation {
                total_ms: Some(100),
                phases: EstimatePhaseTimings {
                    cold_load_ms: Some(10),
                    warm_reload_ms: Some(5),
                    ..Default::default()
                },
                outcome: EstimateOutcome::Success,
                observed_at_unix_s: unix_seconds(),
                ..Default::default()
            },
        );
        let before = (
            coordinator.state_version,
            coordinator.plan_version,
            coordinator.pending.len(),
            coordinator.pending_owner_work.len(),
            coordinator.leases.len(),
        );

        let preview = coordinator.placement_preview(&request, 1, &prepared);
        let batch_profiles = coordinator
            .batch_device_profiles(&request, 65, &prepared)
            .unwrap();

        assert_eq!(preview.outcome, "planned", "{:?}", preview.reason);
        assert_eq!(
            batch_profiles.len(),
            2,
            "batch profiling must be independent of the public 64-copy preview ceiling"
        );
        assert_eq!(
            batch_profiles
                .iter()
                .map(|candidate| candidate.device_id.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([stable_id.as_str(), stable_id1.as_str()])
        );
        assert_eq!(
            preview
                .candidate
                .as_ref()
                .map(|candidate| candidate.device_id.as_str()),
            Some(stable_id.as_str())
        );
        let candidate = preview.candidate.as_ref().unwrap();
        assert_eq!(candidate.setup_ms, 8_000);
        assert!(
            candidate
                .predicted_completion_after_ms
                .saturating_sub(candidate.predicted_start_after_ms)
                >= 38_000,
            "an unseen generation must retain the 30s run fallback in addition to cold setup"
        );
        assert_eq!(
            (
                coordinator.state_version,
                coordinator.plan_version,
                coordinator.pending.len(),
                coordinator.pending_owner_work.len(),
                coordinator.leases.len(),
            ),
            before,
            "preview must not reserve, enqueue, lease, or advance scheduler authority"
        );
    }

    #[tokio::test]
    async fn placement_preview_models_one_parent_expansion_and_every_batch_child() {
        let root = tempfile::tempdir().unwrap();
        for name in [
            "transformer.safetensors",
            "vae.safetensors",
            "qwen3.safetensors",
            "tokenizer.json",
            "upscaler.safetensors",
        ] {
            std::fs::write(root.path().join(name), b"preview").unwrap();
        }
        let mut config = mold_core::Config::default();
        config.expand.backend = "local".into();
        config.expand.model = "preview-expander:q8".into();
        config.models.insert(
            "preview-z".into(),
            mold_core::ModelConfig {
                transformer: Some(
                    root.path()
                        .join("transformer.safetensors")
                        .display()
                        .to_string(),
                ),
                vae: Some(root.path().join("vae.safetensors").display().to_string()),
                text_encoder_files: Some(vec![root
                    .path()
                    .join("qwen3.safetensors")
                    .display()
                    .to_string()]),
                text_tokenizer: Some(root.path().join("tokenizer.json").display().to_string()),
                family: Some("z-image".into()),
                ..Default::default()
            },
        );
        config.models.insert(
            "preview-upscaler".into(),
            mold_core::ModelConfig {
                transformer: Some(
                    root.path()
                        .join("upscaler.safetensors")
                        .display()
                        .to_string(),
                ),
                ..Default::default()
            },
        );
        let mut request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"preview-z","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1,"expand":true,"upscale_model":"preview-upscaler"}"#,
        )
        .unwrap();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::gpu(0),
            advanced: None,
        });
        let (worker0, _worker_rx0) = test_worker(0);
        let (worker1, _worker_rx1) = test_worker(1);
        let stable_id0 = worker_device_id(&worker0);
        let stable_id1 = worker_device_id(&worker1);
        let pool = Arc::new(GpuPool {
            workers: vec![worker0, worker1].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(config.clone(), QueueHandle::new(ingress_tx), pool, 1);
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let prepared = crate::variant_dependencies::prepare_local_execution_inputs(
            &config,
            &request,
            vec![
                crate::execution_plan::DeviceFact {
                    id: stable_id0.clone(),
                    ordinal: 0,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 24 << 30,
                },
                crate::execution_plan::DeviceFact {
                    id: stable_id1.clone(),
                    ordinal: 1,
                    backend: mold_core::GpuBackend::Cuda,
                    compute_capability: Some((8, 6)),
                    available_vram_bytes: 24 << 30,
                },
            ],
        )
        .await
        .unwrap();
        let before = (
            coordinator.state_version,
            coordinator.plan_version,
            coordinator.pending.len(),
            coordinator.pending_owner_work.len(),
            coordinator.leases.len(),
        );

        let public_preview = coordinator.placement_preview(&request, 3, &prepared);
        assert!(!public_preview.authoritative);
        assert_eq!(public_preview.outcome, "unsupported");
        assert!(public_preview.candidate.is_none());
        assert!(public_preview.stage_candidates.is_empty());
        let config_guard = coordinator.state.config.try_write().unwrap();
        let contended_preview = coordinator.placement_preview(&request, 3, &prepared);
        assert!(!contended_preview.authoritative);
        assert_eq!(contended_preview.outcome, "unsupported");
        drop(config_guard);

        let preview = coordinator.placement_preview_dag(&request, 3, &prepared);
        let batch_profiles = coordinator
            .batch_device_profiles(&request, 3, &prepared)
            .unwrap();

        assert_eq!(preview.outcome, "planned", "{:?}", preview.reason);
        assert_eq!(
            batch_profiles
                .iter()
                .map(|candidate| candidate.device_id.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([stable_id0.as_str()]),
            "explicit generation placement must constrain batch device profiling"
        );
        assert_eq!(preview.stage_candidates.len(), 7);
        let expansion = preview
            .stage_candidates
            .iter()
            .filter(|stage| stage.stage_index == 0)
            .collect::<Vec<_>>();
        assert_eq!(expansion.len(), 1, "expansion runs once per parent request");
        assert_eq!(expansion[0].copy_index, None);
        let generations = preview
            .stage_candidates
            .iter()
            .filter(|stage| stage.stage_index == 1)
            .collect::<Vec<_>>();
        let upscales = preview
            .stage_candidates
            .iter()
            .filter(|stage| stage.stage_index == 2)
            .collect::<Vec<_>>();
        assert_eq!(generations.len(), 3);
        assert_eq!(upscales.len(), 3);
        for generation in &generations {
            assert!(
                generation.candidate.predicted_start_after_ms
                    >= expansion[0].candidate.predicted_completion_after_ms,
                "generation {:?} starts at {} before expansion completes at {}",
                generation.copy_index,
                generation.candidate.predicted_start_after_ms,
                expansion[0].candidate.predicted_completion_after_ms,
            );
            let upscale = upscales
                .iter()
                .find(|upscale| upscale.copy_index == generation.copy_index)
                .expect("each generated sibling has one upscale child");
            assert!(
                upscale.candidate.predicted_start_after_ms
                    >= generation.candidate.predicted_completion_after_ms
            );
        }
        assert!(
            upscales
                .iter()
                .any(|stage| stage.candidate.device_id != stable_id0),
            "an unpinned post-upscale child may move to the other eligible device"
        );
        let candidate = preview.candidate.as_ref().unwrap();
        assert_eq!(
            candidate.predicted_start_after_ms,
            expansion[0].candidate.predicted_start_after_ms
        );
        assert_eq!(
            candidate.predicted_completion_after_ms,
            upscales
                .iter()
                .map(|stage| stage.candidate.predicted_completion_after_ms)
                .max()
                .unwrap()
        );
        assert_eq!(
            candidate.setup_ms,
            preview
                .stage_candidates
                .iter()
                .map(|stage| stage.candidate.setup_ms)
                .sum::<u64>()
        );
        assert_eq!(candidate.setup_kind, "pipeline");
        assert_eq!(
            (
                coordinator.state_version,
                coordinator.plan_version,
                coordinator.pending.len(),
                coordinator.pending_owner_work.len(),
                coordinator.leases.len(),
            ),
            before,
            "preview must not mutate coordinator state"
        );
    }

    #[test]
    fn placement_preview_rejects_invalid_copies_and_no_eligible_device() {
        let pool = Arc::new(GpuPool {
            workers: Vec::new().into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"missing","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let prepared = crate::execution_plan::PreparedExecutionInputs::default();

        let invalid = coordinator.placement_preview(&request, 0, &prepared);
        assert_eq!(invalid.outcome, "infeasible");
        assert!(invalid.reason.unwrap().contains("between 1 and 64"));

        let unavailable = coordinator.placement_preview(&request, 1, &prepared);
        assert_eq!(unavailable.outcome, "infeasible");
        assert!(unavailable.candidate.is_none());
    }

    #[test]
    fn placement_preview_stops_when_the_reply_receiver_is_dropped() {
        let pool = Arc::new(GpuPool {
            workers: Vec::new().into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        );
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let request: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"","model":"missing","width":512,"height":512,"steps":4,"guidance":1.0,"batch_size":1}"#,
        )
        .unwrap();
        let prepared = crate::execution_plan::PreparedExecutionInputs::default();
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<()>();
        drop(reply_rx);

        let preview = coordinator
            .placement_preview_cancellable(&request, 1, &prepared, &|| reply_tx.is_closed());
        assert_eq!(preview.outcome, "temporarily_unavailable");
        assert_eq!(
            preview.reason.as_deref(),
            Some("placement preview cancelled")
        );
        assert!(coordinator
            .batch_device_profiles_cancellable(&request, 1, &prepared, &|| reply_tx.is_closed(),)
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
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

    fn utility_state_with_workers(count: usize) -> AppState {
        let workers = (0..count)
            .map(|ordinal| test_worker(ordinal).0)
            .collect::<Vec<_>>();
        let pool = Arc::new(GpuPool {
            workers: workers.into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(count.max(1));
        AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            count.max(1),
        )
    }

    fn utility_state_with_backend(backend: mold_core::GpuBackend) -> AppState {
        let (mut worker, _rx) = test_worker(0);
        {
            let worker = Arc::get_mut(&mut worker).expect("fresh test worker");
            worker.gpu.backend = backend;
            worker.gpu.stable_id = Some(match backend {
                mold_core::GpuBackend::Cuda => "cuda:utility-stable".to_string(),
                mold_core::GpuBackend::Metal => "metal:utility-stable".to_string(),
            });
        }
        let pool = Arc::new(GpuPool {
            workers: vec![worker].into(),
        });
        let (ingress_tx, _ingress_rx) = tokio::sync::mpsc::channel(1);
        AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(ingress_tx),
            pool,
            1,
        )
    }

    #[test]
    fn exact_upscale_candidates_are_deterministic_for_1_2_8_and_64_gpus() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();

        for count in [1, 2, 8, 64] {
            let state = utility_state_with_workers(count);
            let first =
                upscale_candidates(&state, "real-esrgan-x4plus:fp16", &weights, None).unwrap();
            let second =
                upscale_candidates(&state, "real-esrgan-x4plus:fp16", &weights, None).unwrap();

            assert_eq!(first, second, "candidate order drifted at {count} GPUs");
            assert_eq!(first.len(), count + 1);
            assert_eq!(first[0].placement(), UtilityPlacement::Cpu);
            for (ordinal, plan) in first.iter().skip(1).enumerate() {
                assert_eq!(
                    plan.placement(),
                    UtilityPlacement::Device {
                        backend: mold_core::GpuBackend::Cuda,
                        ordinal,
                    }
                );
            }
            assert_eq!(
                first
                    .iter()
                    .map(UtilityExecutionPlan::execution_fingerprint)
                    .collect::<BTreeSet<_>>()
                    .len(),
                count + 1,
                "every exact backend/ordinal plan needs a distinct fingerprint"
            );
        }
    }

    #[test]
    fn utility_preview_candidates_recover_the_identical_execution_plan() {
        let state = utility_state_with_workers(2);
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();
        let plans = upscale_candidates(&state, "real-esrgan-x4plus:fp16", &weights, None).unwrap();
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let work = OwnerWork::StandaloneUpscale(Box::new(crate::gpu_pool::StandaloneUpscaleJob {
            id: "preview-equality".to_string(),
            model: "real-esrgan-x4plus:fp16".to_string(),
            weights_path: weights,
            request: mold_core::UpscaleRequest {
                model: "real-esrgan-x4plus:fp16".to_string(),
                image: vec![1],
                output_format: mold_core::OutputFormat::Png,
                tile_size: None,
                metadata: None,
            },
            progress_tx: None,
            cancellation: mold_inference::InferenceCancellationToken::default(),
            execution_plan: None,
            result_tx,
        }));
        let pending = PendingOwnerWork {
            model_fingerprint: "unused-estimate".to_string(),
            estimated_vram_bytes: 1,
            estimated_host_ram_bytes: 1,
            hard_ordinal: None,
            priority: PriorityClass::User,
            preferred_ordinal: None,
            candidate_plans: Vec::new(),
            queue_rank: 0,
            ready_at_ms: 0,
            bypass_count: 0,
            warm_wait_started_ms: None,
            retry_not_before_ms: None,
            utility_plans: plans.clone(),
            work,
        };
        let coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );

        for plan in plans {
            let device_id = match plan.placement() {
                UtilityPlacement::Cpu => CPU_UTILITY_DEVICE_ID.to_string(),
                UtilityPlacement::Device { ordinal, .. } => worker_device_id(
                    &coordinator
                        .state
                        .gpu_pool
                        .worker_by_ordinal(ordinal)
                        .unwrap(),
                ),
            };
            let placement = CandidatePlacement::new(
                DeviceId::new(device_id.clone()),
                ExecutionFingerprint::new(plan.execution_fingerprint()),
                plan.predicted_host_ram_bytes(),
            )
            .with_vram(plan.predicted_vram_bytes());
            assert_eq!(
                coordinator.utility_plan_for_lease(&pending, &device_id, &placement),
                Some(plan),
                "the execution transport must consume the exact preview candidate"
            );
        }
    }

    #[test]
    fn static_timing_floor_uses_explicit_runtime_instead_of_total_subtraction() {
        let estimate = ResolvedEstimate {
            total_ms: 1,
            cold_setup_ms: 1,
            warm_setup_ms: 1,
            predicted_run_ms: 1,
            vram_bytes: 1,
            host_bytes: 1,
            confidence: mold_scheduler::EstimateConfidence::Low,
            learned: false,
        };
        let static_estimate = StaticEstimate {
            total_ms: 9_999,
            cold_setup_ms: 900,
            warm_setup_ms: 90,
            predicted_run_ms: 1_234,
            vram_bytes: 1,
            host_bytes: 1,
        };

        assert_eq!(
            timing_with_static_floors(estimate, static_estimate),
            (900, 90, 1_234)
        );
    }

    #[test]
    fn learned_utility_observation_refines_timing_without_weakening_exact_memory() {
        let state = utility_state_with_workers(1);
        let worker = state.gpu_pool.worker_by_ordinal(0).unwrap();
        let device_id = worker_device_id(&worker);
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();
        let plans = upscale_candidates(&state, "real-esrgan-x4plus:fp16", &weights, None).unwrap();
        let plan = plans
            .iter()
            .find(|plan| {
                matches!(
                    plan.placement(),
                    UtilityPlacement::Device { ordinal: 0, .. }
                )
            })
            .cloned()
            .unwrap();
        let cpu_plan = plans
            .iter()
            .find(|plan| matches!(plan.placement(), UtilityPlacement::Cpu))
            .cloned()
            .unwrap();
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let work = OwnerWork::StandaloneUpscale(Box::new(crate::gpu_pool::StandaloneUpscaleJob {
            id: "exact-memory".to_string(),
            model: "real-esrgan-x4plus:fp16".to_string(),
            weights_path: weights,
            request: mold_core::UpscaleRequest {
                model: "real-esrgan-x4plus:fp16".to_string(),
                image: vec![1],
                output_format: mold_core::OutputFormat::Png,
                tile_size: None,
                metadata: None,
            },
            progress_tx: None,
            cancellation: mold_inference::InferenceCancellationToken::default(),
            execution_plan: None,
            result_tx,
        }));
        let shape_bucket = work.scheduling_shape_bucket();
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        let estimate_key = owner_estimate_key_for_device(
            device_class(&worker),
            WorkKind::StandaloneUpscale,
            "real-esrgan-x4plus:fp16",
            &shape_bucket,
            plan.execution_fingerprint(),
        );
        coordinator.observe_estimate(
            estimate_key,
            EstimateObservation {
                total_ms: Some(100),
                phases: EstimatePhaseTimings {
                    cold_load_ms: Some(10),
                    ..Default::default()
                },
                vram_high_water_bytes: Some(1),
                host_incremental_high_water_bytes: Some(1),
                outcome: EstimateOutcome::Success,
                observed_at_unix_s: unix_seconds(),
                ..Default::default()
            },
        );
        let cpu_low_shape = "cpu-low";
        coordinator.observe_estimate(
            owner_estimate_key_for_device(
                CPU_UTILITY_DEVICE_ID.to_string(),
                WorkKind::StandaloneUpscale,
                "real-esrgan-x4plus:fp16",
                cpu_low_shape,
                cpu_plan.execution_fingerprint(),
            ),
            EstimateObservation {
                total_ms: Some(100),
                phases: EstimatePhaseTimings {
                    cold_load_ms: Some(10),
                    warm_reload_ms: Some(10),
                    ..Default::default()
                },
                outcome: EstimateOutcome::Success,
                observed_at_unix_s: unix_seconds(),
                ..Default::default()
            },
        );
        let cpu_high_shape = "cpu-high";
        coordinator.observe_estimate(
            owner_estimate_key_for_device(
                CPU_UTILITY_DEVICE_ID.to_string(),
                WorkKind::StandaloneUpscale,
                "real-esrgan-x4plus:fp16",
                cpu_high_shape,
                cpu_plan.execution_fingerprint(),
            ),
            EstimateObservation {
                total_ms: Some(60_000),
                phases: EstimatePhaseTimings {
                    cold_load_ms: Some(12_000),
                    ..Default::default()
                },
                outcome: EstimateOutcome::Success,
                observed_at_unix_s: unix_seconds(),
                ..Default::default()
            },
        );

        let snapshot = coordinator.owner_work_snapshot(
            OwnerWorkSchedulingView {
                id: "exact-memory",
                model_fingerprint: "real-esrgan-x4plus:fp16",
                estimated_vram_bytes: 1,
                estimated_host_ram_bytes: 1,
                hard_ordinal: None,
                priority: PriorityClass::User,
                queue_rank: 0,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                kind: WorkKind::StandaloneUpscale,
                shape_bucket: &shape_bucket,
                preferred_ordinal: None,
                resolved_plans: &[],
                requires_exact_plan: false,
            },
            &plans,
            &[DeviceSnapshot::idle(device_id.clone(), 24 << 30)],
        );
        let placement = snapshot
            .candidate_placements
            .iter()
            .find(|candidate| candidate.device_id.as_str() == device_id)
            .unwrap();
        assert_eq!(placement.predicted_vram_bytes, plan.predicted_vram_bytes());
        assert_eq!(
            placement.incremental_host_ram_bytes,
            plan.predicted_host_ram_bytes()
        );
        let static_timing = mold_scheduler::static_timing_for(WorkKind::StandaloneUpscale);
        assert_eq!(placement.cold_setup_ms, static_timing.cold_setup_ms);
        assert_eq!(placement.warm_setup_ms, static_timing.cold_setup_ms);
        assert_eq!(placement.predicted_run_ms, static_timing.predicted_run_ms);

        let cpu_devices = [
            DeviceSnapshot::idle(CPU_UTILITY_DEVICE_ID, u64::MAX).with_backend(Backend::Cpu),
            DeviceSnapshot::idle(device_id, 24 << 30),
        ];
        let cpu_low = coordinator.owner_work_snapshot(
            OwnerWorkSchedulingView {
                id: "cpu-low",
                model_fingerprint: "real-esrgan-x4plus:fp16",
                estimated_vram_bytes: 1,
                estimated_host_ram_bytes: 1,
                hard_ordinal: None,
                priority: PriorityClass::User,
                queue_rank: 0,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                kind: WorkKind::StandaloneUpscale,
                shape_bucket: cpu_low_shape,
                preferred_ordinal: None,
                resolved_plans: &[],
                requires_exact_plan: false,
            },
            &plans,
            &cpu_devices,
        );
        let cpu_low = cpu_low
            .candidate_placements
            .iter()
            .find(|candidate| candidate.device_id.as_str() == CPU_UTILITY_DEVICE_ID)
            .unwrap();
        let cpu_floor =
            mold_scheduler::static_timing_for_placement(WorkKind::StandaloneUpscale, Backend::Cpu);
        assert_eq!(cpu_low.cold_setup_ms, cpu_floor.cold_setup_ms);
        assert_eq!(cpu_low.warm_setup_ms, cpu_floor.cold_setup_ms);
        assert_eq!(cpu_low.predicted_run_ms, cpu_floor.predicted_run_ms);

        let cpu_high = coordinator.owner_work_snapshot(
            OwnerWorkSchedulingView {
                id: "cpu-high",
                model_fingerprint: "real-esrgan-x4plus:fp16",
                estimated_vram_bytes: 1,
                estimated_host_ram_bytes: 1,
                hard_ordinal: None,
                priority: PriorityClass::User,
                queue_rank: 0,
                ready_at_ms: 0,
                bypass_count: 0,
                warm_wait_started_ms: None,
                kind: WorkKind::StandaloneUpscale,
                shape_bucket: cpu_high_shape,
                preferred_ordinal: None,
                resolved_plans: &[],
                requires_exact_plan: false,
            },
            &plans,
            &cpu_devices,
        );
        let cpu_high = cpu_high
            .candidate_placements
            .iter()
            .find(|candidate| candidate.device_id.as_str() == CPU_UTILITY_DEVICE_ID)
            .unwrap();
        assert_eq!(cpu_high.cold_setup_ms, 12_000);
        assert_eq!(cpu_high.warm_setup_ms, 12_000);
        assert_eq!(cpu_high.predicted_run_ms, 48_000);
        assert_eq!(
            cpu_high.predicted_vram_bytes,
            cpu_plan.predicted_vram_bytes()
        );
        assert_eq!(
            cpu_high.incremental_host_ram_bytes,
            cpu_plan.predicted_host_ram_bytes()
        );
    }

    #[test]
    fn exact_utility_planning_prefers_idle_accelerators_and_keeps_cpu_work_conserving() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();

        for (gpu_backend, scheduler_backend) in [
            (mold_core::GpuBackend::Cuda, Backend::Cuda),
            (mold_core::GpuBackend::Metal, Backend::Metal),
        ] {
            let state = utility_state_with_backend(gpu_backend);
            let worker = state.gpu_pool.worker_by_ordinal(0).unwrap();
            let device_id = worker_device_id(&worker);
            let plans =
                upscale_candidates(&state, "real-esrgan-x4plus:fp16", &weights, None).unwrap();
            let coordinator = Coordinator::with_preparer_and_memory(
                state,
                Arc::new(ImmediatePreparer),
                ample_memory(),
            );
            let devices = vec![
                DeviceSnapshot::idle(CPU_UTILITY_DEVICE_ID, u64::MAX).with_backend(Backend::Cpu),
                DeviceSnapshot::idle(device_id.clone(), 24 << 30).with_backend(scheduler_backend),
            ];
            let work = coordinator.owner_work_snapshot(
                OwnerWorkSchedulingView {
                    id: "utility-placement",
                    model_fingerprint: "real-esrgan-x4plus:fp16",
                    estimated_vram_bytes: 1,
                    estimated_host_ram_bytes: 1,
                    hard_ordinal: None,
                    priority: PriorityClass::User,
                    queue_rank: 0,
                    ready_at_ms: 0,
                    bypass_count: 0,
                    warm_wait_started_ms: None,
                    kind: WorkKind::StandaloneUpscale,
                    shape_bucket: "fixture",
                    preferred_ordinal: None,
                    resolved_plans: &[],
                    requires_exact_plan: false,
                },
                &plans,
                &devices,
            );
            let idle = Planner::default()
                .plan(&PlannerSnapshot {
                    state_version: 1,
                    next_plan_version: 1,
                    now_ms: 1_000,
                    next_replan_at_ms: None,
                    queue_paused: false,
                    host_memory: HostMemorySnapshot {
                        headroom_bytes: 64 << 30,
                        sample_generation: 1,
                        ledger_sequence: 1,
                    },
                    devices: devices.clone(),
                    work: vec![work.clone()],
                })
                .unwrap();
            assert_eq!(idle.immediate_leases[0].device_id.as_str(), device_id);

            let busy_devices = vec![
                devices[0].clone(),
                DeviceSnapshot::busy(device_id.clone(), 24 << 30, 120_000)
                    .with_backend(scheduler_backend),
            ];
            let busy_work = coordinator.owner_work_snapshot(
                OwnerWorkSchedulingView {
                    id: "utility-placement",
                    model_fingerprint: "real-esrgan-x4plus:fp16",
                    estimated_vram_bytes: 1,
                    estimated_host_ram_bytes: 1,
                    hard_ordinal: None,
                    priority: PriorityClass::User,
                    queue_rank: 0,
                    ready_at_ms: 0,
                    bypass_count: 0,
                    warm_wait_started_ms: None,
                    kind: WorkKind::StandaloneUpscale,
                    shape_bucket: "fixture",
                    preferred_ordinal: None,
                    resolved_plans: &[],
                    requires_exact_plan: false,
                },
                &plans,
                &busy_devices,
            );
            let busy = Planner::default()
                .plan(&PlannerSnapshot {
                    state_version: 2,
                    next_plan_version: 2,
                    now_ms: 1_000,
                    next_replan_at_ms: None,
                    queue_paused: false,
                    host_memory: HostMemorySnapshot {
                        headroom_bytes: 64 << 30,
                        sample_generation: 2,
                        ledger_sequence: 2,
                    },
                    devices: busy_devices,
                    work: vec![busy_work],
                })
                .unwrap();
            assert_eq!(
                busy.immediate_leases[0].device_id.as_str(),
                CPU_UTILITY_DEVICE_ID
            );
            assert!(busy.immediate_leases[0].estimated_finish_ms < 120_000);
        }
    }

    #[test]
    fn utility_aggregate_ram_admission_and_missing_sample_fail_closed() {
        let devices = vec![
            DeviceSnapshot::idle(CPU_UTILITY_DEVICE_ID, u64::MAX).with_backend(Backend::Cpu),
            DeviceSnapshot::idle("cuda:1", 24 << 30),
        ];
        let work = (0..2)
            .map(|index| {
                WorkSnapshot::new(
                    WorkId::new(format!("utility-{index}")),
                    index,
                    vec![
                        CandidatePlacement::new(
                            DeviceId::new(CPU_UTILITY_DEVICE_ID),
                            ExecutionFingerprint::new(format!("cpu-{index}")),
                            8 << 30,
                        ),
                        CandidatePlacement::new(
                            DeviceId::new("cuda:1"),
                            ExecutionFingerprint::new(format!("gpu-{index}")),
                            8 << 30,
                        )
                        .with_vram(1),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let planner = Planner::default();
        let admitted = planner
            .plan(&PlannerSnapshot {
                state_version: 1,
                next_plan_version: 1,
                now_ms: 0,
                next_replan_at_ms: None,
                queue_paused: false,
                host_memory: HostMemorySnapshot {
                    headroom_bytes: 12 << 30,
                    sample_generation: 1,
                    ledger_sequence: 1,
                },
                devices: devices.clone(),
                work: work.clone(),
            })
            .unwrap();
        assert_eq!(admitted.immediate_leases.len(), 1);
        assert_eq!(admitted.reservation.total_host_ram_bytes, 8 << 30);

        let missing_sample = planner
            .plan(&PlannerSnapshot {
                state_version: 1,
                next_plan_version: 2,
                now_ms: 0,
                next_replan_at_ms: None,
                queue_paused: false,
                host_memory: HostMemorySnapshot {
                    headroom_bytes: 0,
                    sample_generation: 0,
                    ledger_sequence: 0,
                },
                devices,
                work,
            })
            .unwrap();
        assert!(missing_sample.immediate_leases.is_empty());
        assert!(missing_sample
            .blocked
            .iter()
            .all(|blocked| blocked.reason == BlockedReason::InsufficientHostRam));
    }

    #[cfg(feature = "expand")]
    #[test]
    fn parent_owned_expansion_rejection_cancels_only_its_frozen_token() {
        let token = mold_inference::InferenceCancellationToken::default();
        let sibling = mold_inference::InferenceCancellationToken::default();
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let work = OwnerWork::PromptExpansion(Box::new(crate::gpu_pool::PromptExpansionJob {
            id: "parent::prompt-expansion".to_string(),
            parent_id: "parent".to_string(),
            config: mold_core::Config::default(),
            settings: mold_core::ExpandSettings::default(),
            prompt: "frozen prompt".to_string(),
            expand_config: mold_core::ExpandConfig::default(),
            cancellation: token.clone(),
            execution_plan: None,
            result_tx,
        }));
        match &work {
            OwnerWork::PromptExpansion(job) => {
                assert_eq!(job.parent_id, "parent");
                assert_eq!(job.prompt, "frozen prompt");
            }
            _ => unreachable!(),
        }

        work.reject("parent cancelled".to_string());

        assert!(token.is_cancelled());
        assert!(
            !sibling.is_cancelled(),
            "attempt cancellation must not leak to a sibling or retry"
        );
    }

    /// #641: an honest LTX-2 estimate makes an impossible shape's predicted
    /// peak exceed the card. Left as `Transient`, that job would re-resolve on
    /// every scheduler tick and queue forever.
    #[test]
    fn insufficient_vram_is_terminal_when_peak_exceeds_total() {
        const RTX_4090_TOTAL: u64 = 25_757_220_864;

        assert!(
            insufficient_vram_is_terminal(33_474_340_818, RTX_4090_TOTAL),
            "a peak no device could ever hold must be terminal"
        );
        let impossible = classify_generation_plan_failure(
            crate::execution_plan::ExecutionPlanError::InsufficientVram {
                reason: "larger than every device".to_string(),
                required_peak_bytes: 33_474_340_818,
                eligible_device_ids: vec!["cuda:0".to_string()],
            },
            &BTreeMap::from([("cuda:0".to_string(), RTX_4090_TOTAL)]),
        );
        assert_eq!(
            placement_preview_disposition_for_plan_failure(&impossible),
            (true, "infeasible")
        );
        assert!(
            !insufficient_vram_is_terminal(20_000_000_000, RTX_4090_TOTAL),
            "a peak that only exceeds what is currently free stays transient"
        );
        assert!(
            !insufficient_vram_is_terminal(33_474_340_818, 0),
            "unknown capacity must never reject on missing evidence"
        );
    }

    #[test]
    fn insufficient_vram_uses_only_request_eligible_physical_capacity() {
        const GIB: u64 = 1 << 30;
        let capacities = BTreeMap::from([
            ("cuda:small".to_string(), 8 * GIB),
            ("cuda:large".to_string(), 24 * GIB),
        ]);
        let failure = |eligible_device_ids: &[&str]| {
            classify_generation_plan_failure(
                crate::execution_plan::ExecutionPlanError::InsufficientVram {
                    reason: "currently short of VRAM".to_string(),
                    required_peak_bytes: 12 * GIB,
                    eligible_device_ids: eligible_device_ids
                        .iter()
                        .map(|id| (*id).to_string())
                        .collect(),
                },
                &capacities,
            )
        };

        assert!(matches!(
            failure(&["cuda:small"]),
            GenerationPlanFailure::Terminal(_)
        ));
        assert!(matches!(
            failure(&["cuda:small", "cuda:large"]),
            GenerationPlanFailure::Transient(_)
        ));
        assert!(matches!(
            failure(&["cuda:large"]),
            GenerationPlanFailure::Transient(_)
        ));
        assert!(matches!(
            failure(&["cuda:unknown"]),
            GenerationPlanFailure::Transient(_)
        ));
    }

    /// #641: `mold.db` held a `scheduler_estimates` row for the failing shape
    /// with `vram_high_water_bytes = 24,884,805,632` (96.6% of the card),
    /// `sample_count = 0`, `failure_count = 3`, `last_outcome = 'failure'` —
    /// and admission still said yes, because `EstimateStore::estimate` drops
    /// buckets with no completion samples.
    #[test]
    fn failure_only_estimate_is_not_usable() {
        let key = EstimateKey {
            device_class: "cuda:sm89:24gb".to_string(),
            model_family: "ltx-2-19b-distilled:fp8".to_string(),
            model_fingerprint: "ltx-2-19b-distilled:fp8".to_string(),
            work_kind: "generation".to_string(),
            shape_bucket: "1024x1024:s8:f97:fps24:a0:src1:edit0:lora0:b1".to_string(),
            execution_fingerprint: "fingerprint".to_string(),
        };
        let failure_only = mold_scheduler::EstimateBucket {
            key: key.clone(),
            sample_count: 0,
            ewma_total_ms: 0.0,
            ewma_runtime_ms: None,
            ewma_load_ms: None,
            ewma_warm_reload_ms: None,
            ewma_prompt_encode_ms: None,
            ewma_denoise_ms: None,
            ewma_vae_ms: None,
            ewma_visual_decode_ms: None,
            ewma_audio_decode_ms: None,
            ewma_mux_ms: None,
            ewma_upscale_ms: None,
            ewma_identity_extract_ms: None,
            vram_conservative_bytes: Some(24_884_805_632),
            host_conservative_bytes: None,
            failure_count: 3,
            invalidated_count: 0,
            last_outcome: EstimateOutcome::Failure,
            last_fallback_reason: None,
            last_invalidated_plan_reason: None,
            last_observed_at_unix_s: 0,
        };
        let store = EstimateStore::from_buckets([failure_only.clone()]);

        assert_eq!(
            store
                .estimate(&key, static_generation_estimate_for_test())
                .vram_bytes,
            1_000_000_000,
            "the store still discards a failure-only bucket for its own estimate"
        );
        assert_eq!(
            failure_only_vram_floor(&store, &key),
            24_884_805_632,
            "a failure-only high water is a lower bound on demand, never evidence of fit"
        );

        let succeeded = mold_scheduler::EstimateBucket {
            sample_count: 4,
            last_outcome: EstimateOutcome::Success,
            ..failure_only
        };
        assert_eq!(
            failure_only_vram_floor(&EstimateStore::from_buckets([succeeded]), &key),
            0,
            "a bucket with completion samples is ordinary learned evidence"
        );
        assert_eq!(
            failure_only_vram_floor(&EstimateStore::from_buckets([]), &key),
            0
        );
    }

    fn static_generation_estimate_for_test() -> StaticEstimate {
        StaticEstimate {
            total_ms: 1,
            cold_setup_ms: 1,
            warm_setup_ms: 1,
            predicted_run_ms: 1,
            vram_bytes: 1_000_000_000,
            host_bytes: 1_000_000_000,
        }
    }

    #[test]
    fn a_model_unload_is_priced_at_zero_so_admission_can_never_block_it() {
        // The learned estimator cannot know that an unload releases memory —
        // it reports whatever the recorded observations cost. Pricing an
        // unload as a consumer is what deadlocked recovery: a resident engine
        // exhausts host RAM and VRAM, and the only work that frees them is
        // then rejected for lack of the resources it was about to return.
        assert_eq!(
            planned_memory_bytes(
                mold_scheduler::WorkKind::AdminModelUnload,
                20 << 30,
                4 << 30
            ),
            (0, 0)
        );
    }

    #[test]
    fn consuming_work_keeps_its_full_predicted_memory() {
        for kind in [
            mold_scheduler::WorkKind::Generation,
            mold_scheduler::WorkKind::PreparedSibling,
            mold_scheduler::WorkKind::ChainStage,
            mold_scheduler::WorkKind::PostUpscale,
            mold_scheduler::WorkKind::StandaloneUpscale,
            mold_scheduler::WorkKind::PromptExpansion,
            mold_scheduler::WorkKind::AdminModelLoad,
            mold_scheduler::WorkKind::BatchChild,
        ] {
            assert_eq!(
                planned_memory_bytes(kind, 20 << 30, 4 << 30),
                (20 << 30, 4 << 30),
                "{kind:?}"
            );
        }
    }
}
