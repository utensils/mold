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
    DeviceId, DeviceSnapshot, ExecutionFingerprint, GrantValidationSnapshot, HostMemorySnapshot,
    Plan, Planner, PlannerSnapshot, PriorityClass, WorkId, WorkSnapshot,
};

use crate::gpu_pool::{GpuJob, GpuWorker, LeaseGrant, OwnerWork};
use crate::state::{AppState, GenerationJob, SseMessage};

const REPLAN_DEBOUNCE: Duration = Duration::from_secs(2);
const REPLAN_MAX_DELAY: Duration = Duration::from_secs(5);
const RECONCILE_INTERVAL: Duration = Duration::from_millis(10);
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const MIN_TRANSIENT_HOST_RAM: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeaseFence {
    pub work_id: String,
    pub device_id: String,
    pub state_version: u64,
    pub plan_version: u64,
    pub worker_generation: u64,
    pub memory_sample_generation: u64,
    pub memory_ledger_sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseRejection {
    StaleWorkerGeneration,
    FatalCuda,
}

pub enum WorkerEvent {
    Ready {
        device_id: String,
        ordinal: usize,
        worker_generation: u64,
    },
    Accepted {
        device_id: String,
        ordinal: usize,
        worker_generation: u64,
        work_id: String,
        plan_version: u64,
    },
    AllocationCommitted {
        device_id: String,
        work_id: String,
        worker_generation: u64,
    },
    FollowupReady {
        work: Box<ScheduledOwnerWork>,
    },
    Rejected {
        device_id: String,
        ordinal: usize,
        worker_generation: u64,
        grant: Box<LeaseGrant>,
        reason: LeaseRejection,
    },
    Completed {
        device_id: String,
        ordinal: usize,
        worker_generation: u64,
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

#[derive(Clone, Default)]
pub struct ScheduledWorkHandle {
    tx: Option<tokio::sync::mpsc::Sender<ScheduledOwnerWork>>,
}

impl ScheduledWorkHandle {
    pub fn new(tx: tokio::sync::mpsc::Sender<ScheduledOwnerWork>) -> Self {
        Self { tx: Some(tx) }
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
    generation: u64,
}

#[derive(Clone, Debug)]
struct ActiveLease {
    work_id: String,
    plan_version: u64,
    worker_generation: u64,
    accepted: bool,
    previous_target: Option<usize>,
}

struct PendingGeneration {
    job: GenerationJob,
    bypass_count: u8,
    warm_wait_started_ms: Option<u64>,
    preparation: PreparationState,
}

struct PendingOwnerWork {
    model_fingerprint: String,
    estimated_vram_bytes: u64,
    estimated_host_ram_bytes: u64,
    hard_ordinal: Option<usize>,
    priority: PriorityClass,
    queue_rank: u64,
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
        expanded_prompt: Option<String>,
    },
    Failed {
        work_id: String,
        error: String,
    },
}

type PreparationFuture = Pin<Box<dyn Future<Output = Result<Option<String>, String>> + Send>>;

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
            if request.expand != Some(true) {
                return Ok(None);
            }
            let config = state.config.read().await.clone();
            let settings = config.expand.clone().with_env_overrides();
            if !settings.is_local() {
                // API-backed expansion remains CPU/network work and is
                // resolved before queue admission by the route.
                return Ok(None);
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
            Ok(result.expanded.first().cloned())
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

#[derive(Clone, Debug)]
struct HostMemoryLedger {
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
    fn new() -> Self {
        Self {
            sample: None,
            sequence: 0,
            reservations: BTreeMap::new(),
        }
    }

    fn collect_now(&mut self) {
        self.sequence = self.sequence.saturating_add(1);
        let collection_started_sequence = self.sequence;
        let ram = crate::resources::ram_snapshot();
        self.publish_sample(
            collection_started_sequence,
            ram.total,
            ram.total.saturating_sub(ram.used),
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
        let mut memory = HostMemoryLedger::new();
        memory.collect_now();
        Self::with_preparer_and_memory(state, Arc::new(PostUpscalePreparer), memory)
    }

    fn with_preparer_and_memory(
        state: AppState,
        preparer: Arc<dyn DependencyPreparer>,
        memory: HostMemoryLedger,
    ) -> Self {
        let (preparation_tx, preparation_rx) = tokio::sync::mpsc::unbounded_channel();
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
            #[cfg(test)]
            before_grant_hook: None,
        }
    }

    fn mutate(&mut self, immediate: &mut bool) {
        self.state_version = self.state_version.saturating_add(1);
        self.dirty.mark_dirty(Instant::now(), self.state_version);
        *immediate = true;
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
                bypass_count: 0,
                warm_wait_started_ms: None,
                preparation: PreparationState::Needed,
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
                    Ok(expanded_prompt) => PreparationEvent::Ready {
                        work_id: id,
                        expanded_prompt,
                    },
                    Err(error) => PreparationEvent::Failed { work_id: id, error },
                };
                let _ = tx.send(event);
            });
        }
    }

    fn handle_preparation_event(&mut self, event: PreparationEvent, immediate: &mut bool) {
        match event {
            PreparationEvent::Ready {
                work_id,
                expanded_prompt,
            } => {
                let Some(pending) = self.pending.get_mut(&work_id) else {
                    return;
                };
                if pending.preparation != PreparationState::Preparing {
                    return;
                }
                if let Some(expanded_prompt) = expanded_prompt {
                    pending.job.request.original_prompt = Some(pending.job.request.prompt.clone());
                    pending.job.request.prompt = expanded_prompt;
                }
                pending.preparation = PreparationState::Ready;
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
        match event {
            WorkerEvent::Ready {
                device_id,
                ordinal,
                worker_generation,
            } => {
                if self.leases.contains_key(&device_id) {
                    tracing::warn!(
                        device_id,
                        worker_generation,
                        "ignoring Ready while device still owns a lease"
                    );
                    return;
                }
                if self
                    .ready
                    .get(&device_id)
                    .is_some_and(|ready| ready.generation >= worker_generation)
                {
                    return;
                }
                self.unavailable.remove(&device_id);
                self.ready.insert(
                    device_id,
                    ReadyWorker {
                        ordinal,
                        generation: worker_generation,
                    },
                );
                self.mutate(immediate);
            }
            WorkerEvent::Accepted {
                device_id,
                ordinal,
                worker_generation,
                work_id,
                plan_version,
            } => {
                let valid = self.leases.get_mut(&device_id).is_some_and(|lease| {
                    if lease.work_id == work_id
                        && lease.plan_version == plan_version
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
                worker_generation,
            } => {
                let valid = self.leases.get(&device_id).is_some_and(|lease| {
                    lease.work_id == work_id && lease.worker_generation == worker_generation
                });
                if valid {
                    self.memory.commit(&work_id);
                    self.mutate(immediate);
                } else {
                    tracing::error!(
                        device_id,
                        work_id,
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
                worker_generation,
                grant,
                reason,
            } => {
                tracing::warn!(
                    device_id,
                    ordinal,
                    worker_generation,
                    ?reason,
                    "worker rejected a fenced grant"
                );
                let previous_target = self
                    .leases
                    .remove(&device_id)
                    .and_then(|lease| lease.previous_target);
                let work_id = grant.work.id().to_string();
                self.memory.release(&work_id);
                self.memory.collect_now();
                if let Some(worker) = self.state.gpu_pool.worker_by_ordinal(ordinal) {
                    worker.release_in_flight();
                }
                let LeaseGrant { work, retry, .. } = *grant;
                match work {
                    OwnerWork::Generation(job) => {
                        let generation_job = generation_from_gpu_job(*job);
                        if reason == LeaseRejection::FatalCuda {
                            reject_generation(
                                &self.state,
                                generation_job,
                                "CUDA context is fatally poisoned; server restart required"
                                    .to_string(),
                            );
                        } else {
                            self.state
                                .job_registry
                                .requeue_rejected_dispatch(&generation_job.id, previous_target);
                            self.pending.insert(
                                generation_job.id.clone(),
                                PendingGeneration {
                                    job: generation_job,
                                    bypass_count: 0,
                                    warm_wait_started_ms: None,
                                    preparation: PreparationState::Ready,
                                },
                            );
                        }
                    }
                    work => {
                        if reason == LeaseRejection::FatalCuda {
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
                worker_generation,
            } => {
                if let Some(lease) = self.leases.remove(&device_id) {
                    if lease.worker_generation != worker_generation {
                        tracing::error!(
                            device_id,
                            ordinal,
                            worker_generation,
                            leased_generation = lease.worker_generation,
                            "worker completion generation did not match active lease"
                        );
                    }
                    self.memory.release(&lease.work_id);
                    self.memory.collect_now();
                }
                self.mutate(immediate);
            }
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
                    worker_device_id(worker),
                    worker.in_flight.load(Ordering::SeqCst),
                )
            })
            .collect::<BTreeMap<_, _>>();
        if worker_claims != self.last_worker_claims {
            self.last_worker_claims = worker_claims;
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
                            gpu.vram_total.saturating_sub(gpu.vram_used),
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        self.state
            .gpu_pool
            .workers
            .iter()
            .map(|worker| {
                let id = worker_device_id(worker);
                let ready = self.ready.get(&id);
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
                if let Some(model) = worker
                    .resident_model
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
                {
                    warm.insert(ExecutionFingerprint::new(model));
                }
                DeviceSnapshot {
                    id: DeviceId::new(id),
                    backend: match worker.gpu.backend {
                        mold_core::GpuBackend::Metal => Backend::Metal,
                        _ => Backend::Cuda,
                    },
                    admin_state: DeviceAdminState::Enabled,
                    health,
                    activity: if ready.is_some()
                        && !self.leases.contains_key(&worker_device_id(worker))
                        && worker.in_flight.load(Ordering::SeqCst) == 0
                    {
                        DeviceActivity::Idle
                    } else {
                        DeviceActivity::Busy
                    },
                    available_at_ms: None,
                    worker_generation: ready.map_or(0, |ready| ready.generation),
                    // The periodic resource sample is authoritative. Before
                    // its first tick, discovery's driver sample is still an
                    // actual free-memory reading; total VRAM is never used as
                    // a proxy for free capacity.
                    available_vram_bytes: sampled_free
                        .get(&(worker.gpu.backend, worker.gpu.ordinal))
                        .copied()
                        .unwrap_or(worker.gpu.free_vram_bytes),
                    warm_execution_fingerprints: warm,
                }
            })
            .collect()
    }

    fn device_facts(&self) -> Vec<crate::execution_plan::DeviceFact> {
        self.device_snapshots()
            .into_iter()
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
    ) -> Result<Vec<crate::execution_plan::ResolvedExecutionPlan>, String> {
        let config = self
            .state
            .config
            .try_read()
            .map_err(|_| "configuration changed while resolving execution plan".to_string())?;
        let offload_requested = matches!(
            std::env::var("MOLD_OFFLOAD").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        let resolved = crate::execution_plan::resolve_execution_plans(
            &config,
            &pending.job.request,
            &self.device_facts(),
            offload_requested,
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
            return Ok(self
                .device_facts()
                .into_iter()
                .filter(|device| device.available_vram_bytes >= estimate)
                .map(|device| crate::execution_plan::ResolvedExecutionPlan {
                    device_id: device.id,
                    device_ordinal: device.ordinal,
                    model_fingerprint: pending.job.request.model.clone(),
                    effective_placement: crate::execution_plan::EffectivePlacement {
                        components: BTreeMap::new(),
                    },
                    components: BTreeMap::new(),
                    attention_backend: crate::execution_plan::AttentionBackend::Auto,
                    offload_mode: crate::execution_plan::OffloadMode::None,
                    predicted_vram_peak_bytes: estimate,
                    predicted_host_increment_bytes: MIN_TRANSIENT_HOST_RAM,
                    determinism_class:
                        crate::execution_plan::DeterminismClass::CpuSeededCrossBackend,
                    execution_fingerprint: pending.job.request.model.clone(),
                })
                .collect());
        }
        resolved.map_err(|error| error.to_string())
    }

    fn work_snapshots(&self) -> Vec<WorkSnapshot> {
        let queue_order = self.state.job_registry.queued_ids_in_order();
        let ranks = queue_order
            .iter()
            .enumerate()
            .map(|(rank, id)| (id.as_str(), rank as u64))
            .collect::<BTreeMap<_, _>>();
        let mut snapshots: Vec<WorkSnapshot> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.preparation == PreparationState::Ready)
            .map(|(id, pending)| {
                let model = pending.job.request.model.as_str();
                let failed = crate::gpu_pool::failed_ordinals_for_model(model);
                let candidates = self
                    .generation_plans(pending)
                    .unwrap_or_else(|error| {
                        tracing::debug!(job_id = id, %error, "generation has no valid execution plan");
                        Vec::new()
                    })
                    .into_iter()
                    .filter(|plan| !failed.contains(&plan.device_ordinal))
                    .map(|plan| {
                        CandidatePlacement::new(
                            DeviceId::new(plan.device_id),
                            ExecutionFingerprint::new(plan.execution_fingerprint),
                            plan.predicted_host_increment_bytes,
                        )
                        .with_vram(plan.predicted_vram_peak_bytes)
                    })
                    .collect::<Vec<_>>();
                let mut work = WorkSnapshot::new(
                    WorkId::new(id.clone()),
                    ranks.get(id.as_str()).copied().unwrap_or(u64::MAX),
                    candidates,
                )
                .with_bypass_count(pending.bypass_count);
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
        snapshots.extend(self.pending_owner_work.iter().map(|(id, pending)| {
            let candidates = self
                .state
                .gpu_pool
                .workers
                .iter()
                .map(|worker| {
                    CandidatePlacement::new(
                        DeviceId::new(worker_device_id(worker)),
                        ExecutionFingerprint::new(pending.model_fingerprint.clone()),
                        pending.estimated_host_ram_bytes,
                    )
                    .with_vram(
                        pending
                            .estimated_vram_bytes
                            .min(worker.gpu.total_vram_bytes),
                    )
                })
                .collect::<Vec<_>>();
            let mut work = WorkSnapshot::new(
                WorkId::new(id.clone()),
                (u64::MAX / 2).saturating_add(pending.queue_rank),
                candidates,
            )
            .with_priority(pending.priority);
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

    fn planner_snapshot(&self) -> PlannerSnapshot {
        PlannerSnapshot {
            state_version: self.state_version,
            next_plan_version: self.plan_version.saturating_add(1),
            now_ms: monotonic_ms(),
            next_replan_at_ms: self.dirty.deadline().map(monotonic_deadline_ms),
            host_memory: self.memory.snapshot(),
            devices: self.device_snapshots(),
            work: self.work_snapshots(),
        }
    }

    async fn dispatch_ready(&mut self) -> Option<u64> {
        if self.state.queue_pause.is_paused()
            || (self.pending.is_empty() && self.pending_owner_work.is_empty())
            || self.ready.is_empty()
            || self
                .state
                .gpu_pool
                .workers
                .iter()
                .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
        {
            return None;
        }
        loop {
            let snapshot = self.planner_snapshot();
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
            if plan.immediate_leases.is_empty() {
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
                    state_version: plan.state_version,
                    plan_version: plan.plan_version,
                    worker_generation: ready.generation,
                    memory_sample_generation: plan.reservation.sample_generation,
                    memory_ledger_sequence: plan.reservation.ledger_sequence,
                };
                if !worker.try_claim_in_flight() {
                    grant_failed = true;
                    break;
                }
                if let Some(pending) = self.pending.remove(&id) {
                    let execution_plan = self.generation_plans(&pending).ok().and_then(|plans| {
                        plans.into_iter().find(|plan| plan.device_id == device_id)
                    });
                    let Some(execution_plan) = execution_plan else {
                        self.pending.insert(id.clone(), pending);
                        worker.release_in_flight();
                        self.state_version = self.state_version.saturating_add(1);
                        grant_failed = true;
                        break;
                    };
                    let bypass_count = pending.bypass_count;
                    let warm_wait_started_ms = pending.warm_wait_started_ms;
                    let gpu_job = gpu_job_from_generation(
                        &self.state,
                        pending.job,
                        fence.clone(),
                        Some(execution_plan),
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
                                    plan_version: plan.plan_version,
                                    worker_generation: ready.generation,
                                    accepted: false,
                                    previous_target,
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
                                    bypass_count,
                                    warm_wait_started_ms,
                                    preparation: PreparationState::Ready,
                                },
                            );
                            self.unavailable.insert(device_id.clone());
                            grant_failed = true;
                            break;
                        }
                    }
                } else if let Some(pending) = self.pending_owner_work.remove(&id) {
                    let metadata = (
                        pending.model_fingerprint,
                        pending.estimated_vram_bytes,
                        pending.estimated_host_ram_bytes,
                        pending.hard_ordinal,
                        pending.priority,
                        pending.queue_rank,
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
                        }),
                    });
                    match worker.try_send_job(grant) {
                        Ok(()) => {
                            self.ready.remove(&device_id);
                            self.leases.insert(
                                device_id.clone(),
                                ActiveLease {
                                    work_id: id.clone(),
                                    plan_version: plan.plan_version,
                                    worker_generation: ready.generation,
                                    accepted: false,
                                    previous_target: None,
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
                                    work: returned.work,
                                },
                            );
                            self.unavailable.insert(device_id.clone());
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
                continue;
            }

            for update in &plan.bypass_updates {
                if let Some(pending) = self.pending.get_mut(update.work_id.as_str()) {
                    pending.bypass_count = update.new_count;
                }
            }
            for wait in &plan.warm_waits {
                if let Some(pending) = self.pending.get_mut(wait.work_id.as_str()) {
                    pending.warm_wait_started_ms = Some(wait.started_at_ms);
                }
            }
            self.state_version = self.state_version.saturating_add(1);
            return Some(plan.state_version);
        }
    }

    fn reject_all_unstarted_for_fatal_cuda(&mut self) {
        self.reject_all_unstarted("CUDA context is fatally poisoned; server restart required");
    }

    fn reject_all_unstarted(&mut self, message: &str) {
        let pending = std::mem::take(&mut self.pending);
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
                    coordinator.handle_worker_event(event, &mut immediate);
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
) -> GpuJob {
    if let Some(plan) = execution_plan.as_ref() {
        job.request.placement = Some(crate::execution_plan::materialized_placement(plan));
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
        lease: Some(lease),
    }
}

fn generation_from_gpu_job(job: GpuJob) -> GenerationJob {
    GenerationJob {
        id: job.id,
        request: job.request,
        completion_payload: job.completion_payload,
        progress_tx: job.progress_tx,
        result_tx: job.result_tx,
        output_dir: job.output_dir,
    }
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

fn monotonic_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
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

    fn ample_memory() -> HostMemoryLedger {
        let mut memory = HostMemoryLedger::new();
        let started = memory.begin_collection();
        memory.publish_sample(started, 128 << 30, 112 << 30);
        memory
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
            Box::pin(async { Ok(None) })
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
            workers: vec![worker],
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
                expanded_prompt: Some("expanded prompt".to_string()),
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

    #[test]
    fn stale_plan_or_memory_generation_cannot_reserve() {
        let mut ledger = HostMemoryLedger::new();
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
    fn stale_ready_generation_and_duplicate_device_lease_are_rejected() {
        let ready = BTreeMap::from([(
            "gpu-a".to_string(),
            ReadyWorker {
                ordinal: 0,
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
                plan_version: 1,
                worker_generation: 7,
                accepted: true,
                previous_target: None,
            },
        )]);
        assert_eq!(
            validate_worker_grant(&ready, &leases, "gpu-a", 7),
            Err(GrantFenceError::DuplicateDeviceLease)
        );
    }

    #[test]
    fn aggregate_eight_plus_eight_is_rejected_against_twelve_gib_headroom() {
        let mut ledger = HostMemoryLedger::new();
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
        let mut ledger = HostMemoryLedger::new();
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
        let mut unavailable = HostMemoryLedger::new();
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
            workers: vec![worker],
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
            workers: vec![worker_a.clone(), worker_b.clone()],
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
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id: worker_device_id(&worker_b),
                ordinal: 1,
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
            workers: vec![worker_a.clone(), worker_b.clone()],
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
            workers: vec![worker_a.clone(), worker_b.clone()],
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
            workers: vec![worker.clone()],
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
            workers: vec![worker.clone()],
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

    #[test]
    fn legacy_chain_claim_and_release_trigger_coordinator_replans() {
        let (worker, _worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()],
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
            workers: vec![worker.clone()],
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
            workers: vec![worker.clone()],
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
            state_version: 1,
            plan_version: 1,
            worker_generation: 1,
            memory_sample_generation: 1,
            memory_ledger_sequence: 1,
        };
        let gpu_job = gpu_job_from_generation(&state, generation, parent_fence.clone(), None);
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
        let mut coordinator = Coordinator::with_preparer_and_memory(
            state,
            Arc::new(ImmediatePreparer),
            ample_memory(),
        );
        coordinator.leases.insert(
            device_id.clone(),
            ActiveLease {
                work_id: "parent".to_string(),
                plan_version: 1,
                worker_generation: 1,
                accepted: true,
                previous_target: None,
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
                worker_generation: 1,
            },
            &mut immediate,
        );
        coordinator.handle_worker_event(
            WorkerEvent::Ready {
                device_id,
                ordinal: 0,
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
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        }
    }

    #[tokio::test]
    async fn cancellation_after_plan_before_grant_is_acknowledged_and_never_transported() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()],
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
    async fn fatal_cuda_after_plan_before_grant_stops_dispatch_without_spinning() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()],
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
                Ok(None)
            })
        }
    }

    #[tokio::test]
    async fn blocked_dependency_preparation_does_not_block_other_ready_gpu_work() {
        let (worker, worker_rx) = test_worker(0);
        let pool = Arc::new(GpuPool {
            workers: vec![worker.clone()],
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
            workers: vec![worker_a.clone(), worker_b.clone(), worker_c.clone()],
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
}
