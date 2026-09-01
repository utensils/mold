use std::collections::BTreeSet;
use std::fmt;

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id!(DeviceId);
string_id!(WorkId);
string_id!(ParentId);
string_id!(ExecutionFingerprint);
string_id!(
    /// Parent-level execution identity shared by device-specific lease plans
    /// that are expected to preserve the same deterministic output contract.
    ///
    /// This is deliberately distinct from [`ExecutionFingerprint`]: the
    /// latter remains bound to an exact worker/device placement and is the
    /// authority used for residency and pre-grant validation.
    ExecutionEquivalenceFingerprint
);

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Backend {
    #[default]
    Cuda,
    Metal,
    Cpu,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceAdminState {
    #[default]
    Enabled,
    Draining,
    Disabled,
    StartupExcluded,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceHealth {
    #[default]
    Healthy,
    Degraded,
    Unavailable,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceActivity {
    #[default]
    Idle,
    Busy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceSnapshot {
    pub id: DeviceId,
    pub backend: Backend,
    pub admin_state: DeviceAdminState,
    pub health: DeviceHealth,
    pub activity: DeviceActivity,
    /// Credible monotonic availability estimate for a busy device.
    pub available_at_ms: Option<u64>,
    pub worker_generation: u64,
    /// Effective capacity at the next owner-thread allocation boundary:
    /// sampled free bytes plus only memory that the owner can safely reclaim
    /// from its measured resident cache entry. Never raw total VRAM.
    pub available_vram_bytes: u64,
    pub warm_execution_fingerprints: BTreeSet<ExecutionFingerprint>,
}

impl DeviceSnapshot {
    pub fn idle(id: impl Into<DeviceId>, available_vram_bytes: u64) -> Self {
        Self {
            id: id.into(),
            backend: Backend::Cuda,
            admin_state: DeviceAdminState::Enabled,
            health: DeviceHealth::Healthy,
            activity: DeviceActivity::Idle,
            available_at_ms: None,
            worker_generation: 0,
            available_vram_bytes,
            warm_execution_fingerprints: BTreeSet::new(),
        }
    }

    pub fn busy(id: impl Into<DeviceId>, available_vram_bytes: u64, available_at_ms: u64) -> Self {
        Self {
            activity: DeviceActivity::Busy,
            available_at_ms: Some(available_at_ms),
            ..Self::idle(id, available_vram_bytes)
        }
    }

    pub fn with_warm(mut self, fingerprint: impl Into<ExecutionFingerprint>) -> Self {
        self.warm_execution_fingerprints.insert(fingerprint.into());
        self
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_admin_state(mut self, state: DeviceAdminState) -> Self {
        self.admin_state = state;
        self
    }

    pub fn with_health(mut self, health: DeviceHealth) -> Self {
        self.health = health;
        self
    }

    pub fn is_schedulable(&self) -> bool {
        self.admin_state == DeviceAdminState::Enabled && self.health == DeviceHealth::Healthy
    }

    /// Whether this device may still be asked to *release* what it is holding.
    ///
    /// Being taken out of scheduling and being forbidden to free memory are
    /// different statements, and conflating them recreates the deadlock that
    /// [`WorkKind::releases_resources`] exists to break. The reachable case is
    /// **Degraded**: three consecutive failures degrade a worker, and a wedged
    /// model is exactly what causes those failures while still holding the
    /// memory — so refusing the unload there leaves no recovery short of
    /// restarting the process.
    ///
    /// This keys on health alone and deliberately ignores `admin_state`, but
    /// note that `activity` still gates through [`Self::is_idle_for`], so a
    /// Draining device is *not* admitted in practice: the server reports it as
    /// `Stopping`, never `Idle`. That is the right outcome rather than an
    /// oversight — a drain tears the worker down and drops the engine on its
    /// way out, and the teardown path returns the memory itself. Disabled and
    /// StartupExcluded are likewise moot: neither has a worker, so neither
    /// produces a candidate to schedule against.
    ///
    /// Quarantine is the one line this does not cross. A Poisoned or
    /// Unavailable device has a fatal context that must never be touched again
    /// in-process (see CLAUDE.md, "Fatal CUDA contexts are never reused or
    /// reset in-process") — its memory comes back with the process, not with an
    /// unload.
    pub fn accepts_releasing_work(&self) -> bool {
        matches!(self.health, DeviceHealth::Healthy | DeviceHealth::Degraded)
    }

    pub fn is_idle(&self) -> bool {
        self.is_schedulable() && self.activity == DeviceActivity::Idle
    }

    /// Idle in the sense that matters for work of `kind`.
    ///
    /// Releasing work uses [`Self::accepts_releasing_work`] instead of
    /// [`Self::is_schedulable`]; everything else keeps the stricter rule.
    pub fn is_idle_for(&self, kind: WorkKind) -> bool {
        if kind.releases_resources() {
            self.accepts_releasing_work() && self.activity == DeviceActivity::Idle
        } else {
            self.is_idle()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidatePlacement {
    pub device_id: DeviceId,
    pub execution_fingerprint: ExecutionFingerprint,
    /// Device-independent deterministic-output identity. `None` is retained
    /// for legacy/synthetic callers that do not yet model parent batching.
    pub execution_equivalence_fingerprint: Option<ExecutionEquivalenceFingerprint>,
    pub predicted_vram_bytes: u64,
    /// Effective device capacity used to admit this concrete edge. Runtime
    /// adapters derive it from authoritative free and safely reclaimable
    /// resident bytes, then replan if it changes.
    pub device_available_vram_bytes: u64,
    /// The concrete admission owns a stable capacity authority and the runtime
    /// owner performs its own final physical-pressure fence. This is false for
    /// ordinary work: live device snapshots remain authoritative by default.
    pub frozen_device_capacity: bool,
    pub incremental_host_ram_bytes: u64,
    pub cold_setup_ms: u64,
    pub warm_setup_ms: u64,
    pub predicted_run_ms: u64,
    /// Deterministic locality tie-break. Zero is preferred; non-zero never
    /// changes eligibility, priority, cardinality, or predicted time.
    pub affinity_penalty: u8,
}

/// Conservative static scheduler timing used until Phase E has a credible
/// learned estimate for the exact execution fingerprint and request shape.
///
/// These values are planning fallbacks, not performance claims. Both
/// authoritative and observe-mode callers must use this one table so a
/// rollout comparison never invents a separate cost model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticTimingEstimate {
    pub cold_setup_ms: u64,
    pub warm_setup_ms: u64,
    pub predicted_run_ms: u64,
}

pub const fn static_timing_for(kind: WorkKind) -> StaticTimingEstimate {
    match kind {
        WorkKind::Generation | WorkKind::PreparedSibling => StaticTimingEstimate {
            cold_setup_ms: 8_000,
            warm_setup_ms: 250,
            predicted_run_ms: 30_000,
        },
        WorkKind::ChainStage => StaticTimingEstimate {
            cold_setup_ms: 10_000,
            warm_setup_ms: 250,
            predicted_run_ms: 60_000,
        },
        WorkKind::PostUpscale | WorkKind::StandaloneUpscale => StaticTimingEstimate {
            cold_setup_ms: 3_000,
            warm_setup_ms: 100,
            predicted_run_ms: 5_000,
        },
        WorkKind::PromptExpansion => StaticTimingEstimate {
            cold_setup_ms: 2_000,
            warm_setup_ms: 50,
            predicted_run_ms: 2_000,
        },
        WorkKind::AdminModelLoad | WorkKind::AdminModelUnload => StaticTimingEstimate {
            cold_setup_ms: 5_000,
            warm_setup_ms: 100,
            predicted_run_ms: 1_000,
        },
    }
}

/// Conservative static timing for a concrete placement.
///
/// CPU is an independent, capacity-one utility lane, not an accelerator with
/// unlimited VRAM. Prompt expansion and image upscaling are materially slower
/// there, so equal fallback costs would let the VRAM tie-break choose CPU over
/// an idle accelerator and would publish implausibly short CPU ETAs. Learned
/// estimates may raise each component but callers must retain these floors.
pub const fn static_timing_for_placement(kind: WorkKind, backend: Backend) -> StaticTimingEstimate {
    match (kind, backend) {
        (WorkKind::PromptExpansion, Backend::Cpu) => StaticTimingEstimate {
            cold_setup_ms: 5_000,
            warm_setup_ms: 5_000,
            predicted_run_ms: 10_000,
        },
        (WorkKind::PostUpscale | WorkKind::StandaloneUpscale, Backend::Cpu) => {
            StaticTimingEstimate {
                cold_setup_ms: 6_000,
                warm_setup_ms: 6_000,
                predicted_run_ms: 20_000,
            }
        }
        _ => static_timing_for(kind),
    }
}

impl CandidatePlacement {
    pub fn new(
        device_id: impl Into<DeviceId>,
        execution_fingerprint: impl Into<ExecutionFingerprint>,
        incremental_host_ram_bytes: u64,
    ) -> Self {
        Self {
            device_id: device_id.into(),
            execution_fingerprint: execution_fingerprint.into(),
            execution_equivalence_fingerprint: None,
            predicted_vram_bytes: 0,
            device_available_vram_bytes: u64::MAX,
            frozen_device_capacity: false,
            incremental_host_ram_bytes,
            cold_setup_ms: 0,
            warm_setup_ms: 0,
            predicted_run_ms: 0,
            affinity_penalty: 0,
        }
    }

    pub fn with_execution_equivalence(
        mut self,
        fingerprint: impl Into<ExecutionEquivalenceFingerprint>,
    ) -> Self {
        self.execution_equivalence_fingerprint = Some(fingerprint.into());
        self
    }

    pub fn with_vram(mut self, predicted_vram_bytes: u64) -> Self {
        self.predicted_vram_bytes = predicted_vram_bytes;
        self
    }

    pub fn with_device_available_vram(mut self, available_vram_bytes: u64) -> Self {
        self.device_available_vram_bytes = available_vram_bytes;
        self
    }

    /// Use the concrete admission's capacity instead of the generic live
    /// device sample for scheduling and grant validation.
    ///
    /// Callers may opt in only when the runtime owner independently rechecks
    /// physical capacity immediately before its first allocation.
    pub fn with_frozen_device_capacity(mut self) -> Self {
        self.frozen_device_capacity = true;
        self
    }

    pub fn with_timing(
        mut self,
        cold_setup_ms: u64,
        warm_setup_ms: u64,
        predicted_run_ms: u64,
    ) -> Self {
        self.cold_setup_ms = cold_setup_ms;
        self.warm_setup_ms = warm_setup_ms;
        self.predicted_run_ms = predicted_run_ms;
        self
    }

    pub fn with_static_timing(mut self, kind: WorkKind) -> Self {
        let timing = static_timing_for(kind);
        self.cold_setup_ms = timing.cold_setup_ms;
        self.warm_setup_ms = timing.warm_setup_ms;
        self.predicted_run_ms = timing.predicted_run_ms;
        self
    }

    pub fn with_affinity_penalty(mut self, penalty: u8) -> Self {
        self.affinity_penalty = penalty;
        self
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum WorkKind {
    #[default]
    Generation,
    PreparedSibling,
    ChainStage,
    PostUpscale,
    StandaloneUpscale,
    PromptExpansion,
    AdminModelLoad,
    AdminModelUnload,
}

impl WorkKind {
    /// Whether this work returns resources instead of consuming them.
    ///
    /// Admission gates exist to stop work from over-committing memory, so they
    /// price every candidate against free host RAM and free VRAM. An unload is
    /// the operation that *frees* both. Pricing it as a consumer creates an
    /// unbreakable deadlock: once a resident engine has exhausted memory, the
    /// only work that can release it is exactly the work the gate rejects, and
    /// the user's sole recovery is killing the server.
    ///
    /// Releasing work is therefore admitted on device availability alone — it
    /// still needs an idle, schedulable owner thread (engine destruction must
    /// happen on the thread owning the CUDA/Metal context), but never has to
    /// prove headroom it is about to create.
    pub const fn releases_resources(self) -> bool {
        matches!(self, Self::AdminModelUnload)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PriorityClass {
    Critical,
    #[default]
    User,
    Admin,
}

impl PriorityClass {
    pub(crate) fn sort_key(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::User => 1,
            Self::Admin => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedBatchPartition {
    pub index: u32,
    pub count: u32,
    pub size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkSnapshot {
    pub id: WorkId,
    pub parent_id: ParentId,
    pub kind: WorkKind,
    pub ready: bool,
    pub ready_at_ms: u64,
    pub queue_rank: u64,
    pub priority_class: PriorityClass,
    pub bypass_count: u8,
    pub chain_stage: Option<u32>,
    pub batch_partition: Option<PlannedBatchPartition>,
    pub hard_device_id: Option<DeviceId>,
    pub backend_requirement: Option<Backend>,
    pub warm_wait_started_at_ms: Option<u64>,
    pub candidate_placements: Vec<CandidatePlacement>,
}

impl WorkSnapshot {
    pub fn new(
        id: impl Into<WorkId>,
        queue_rank: u64,
        candidate_placements: Vec<CandidatePlacement>,
    ) -> Self {
        let id = id.into();
        Self {
            parent_id: ParentId::new(id.as_str()),
            id,
            kind: WorkKind::Generation,
            ready: true,
            ready_at_ms: 0,
            queue_rank,
            priority_class: PriorityClass::User,
            bypass_count: 0,
            chain_stage: None,
            batch_partition: None,
            hard_device_id: None,
            backend_requirement: None,
            warm_wait_started_at_ms: None,
            candidate_placements,
        }
    }

    pub fn with_priority(mut self, priority_class: PriorityClass) -> Self {
        self.priority_class = priority_class;
        self
    }

    pub fn with_bypass_count(mut self, bypass_count: u8) -> Self {
        self.bypass_count = bypass_count;
        self
    }

    pub fn with_chain_stage(mut self, chain_stage: Option<u32>) -> Self {
        self.chain_stage = chain_stage;
        self
    }

    pub fn with_batch_partition(mut self, batch_partition: Option<PlannedBatchPartition>) -> Self {
        self.batch_partition = batch_partition;
        self
    }

    pub fn with_hard_device(mut self, device_id: impl Into<DeviceId>) -> Self {
        self.hard_device_id = Some(device_id.into());
        self
    }

    pub fn with_backend_requirement(mut self, backend: Backend) -> Self {
        self.backend_requirement = Some(backend);
        self
    }

    pub fn with_warm_wait_started_at(mut self, started_at_ms: u64) -> Self {
        self.warm_wait_started_at_ms = Some(started_at_ms);
        self
    }

    pub fn with_ready_at(mut self, ready_at_ms: u64) -> Self {
        self.ready_at_ms = ready_at_ms;
        self
    }

    pub fn with_ready(mut self, ready: bool) -> Self {
        self.ready = ready;
        self
    }

    pub(crate) fn starvation_forced(&self) -> bool {
        self.bypass_count >= 3
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanningMode {
    Optimize,
    WatchdogFallback,
    InternalErrorFallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannerConfig {
    pub warm_wait_max_ms: u64,
    pub mode: PlanningMode,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            warm_wait_max_ms: 2_000,
            mode: PlanningMode::Optimize,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostMemorySnapshot {
    pub headroom_bytes: u64,
    pub sample_generation: u64,
    pub ledger_sequence: u64,
    /// Evictable ZFS ARC counted into `headroom_bytes` by the SAME sample
    /// (#1439), so a block recorded against this headroom can name the
    /// credit that headroom already contains. `None` off ZFS.
    pub reclaimable_zfs_arc_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannerSnapshot {
    pub state_version: u64,
    pub next_plan_version: u64,
    pub now_ms: u64,
    pub next_replan_at_ms: Option<u64>,
    pub queue_paused: bool,
    pub host_memory: HostMemorySnapshot,
    pub devices: Vec<DeviceSnapshot>,
    pub work: Vec<WorkSnapshot>,
}

impl PlannerSnapshot {
    pub fn new(
        state_version: u64,
        next_plan_version: u64,
        now_ms: u64,
        host_headroom_bytes: u64,
        devices: Vec<DeviceSnapshot>,
        work: Vec<WorkSnapshot>,
    ) -> Self {
        Self {
            state_version,
            next_plan_version,
            now_ms,
            next_replan_at_ms: None,
            queue_paused: false,
            host_memory: HostMemorySnapshot {
                headroom_bytes: host_headroom_bytes,
                sample_generation: 0,
                ledger_sequence: 0,
                reclaimable_zfs_arc_bytes: None,
            },
            devices,
            work,
        }
    }

    pub fn with_queue_paused(mut self, queue_paused: bool) -> Self {
        self.queue_paused = queue_paused;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlannerError {
    DuplicateDeviceId {
        device_id: DeviceId,
    },
    DuplicateWorkId {
        work_id: WorkId,
    },
    StaleEligibilityIndex {
        indexed_state_version: u64,
        snapshot_state_version: u64,
    },
    EligibilityIndexMissingWork {
        work_id: WorkId,
    },
    EligibilityIndexUnexpectedWork {
        work_id: WorkId,
    },
    EligibilityIndexCandidatesChanged {
        work_id: WorkId,
    },
}

impl fmt::Display for PlannerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateDeviceId { device_id } => {
                write!(formatter, "duplicate scheduler device ID: {device_id}")
            }
            Self::DuplicateWorkId { work_id } => {
                write!(formatter, "duplicate scheduler work ID: {work_id}")
            }
            Self::StaleEligibilityIndex {
                indexed_state_version,
                snapshot_state_version,
            } => write!(
                formatter,
                "eligibility index state version {indexed_state_version} does not match snapshot \
                 state version {snapshot_state_version}"
            ),
            Self::EligibilityIndexMissingWork { work_id } => {
                write!(formatter, "eligibility index is missing work {work_id}")
            }
            Self::EligibilityIndexUnexpectedWork { work_id } => {
                write!(
                    formatter,
                    "eligibility index contains unknown work {work_id}"
                )
            }
            Self::EligibilityIndexCandidatesChanged { work_id } => write!(
                formatter,
                "eligibility index candidates no longer match work {work_id}"
            ),
        }
    }
}

impl std::error::Error for PlannerError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssignmentReason {
    Priority,
    StarvationForced,
    WarmResident,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImmediateLease {
    pub work_id: WorkId,
    pub device_id: DeviceId,
    pub worker_generation: u64,
    pub placement: CandidatePlacement,
    pub reason: AssignmentReason,
    pub estimated_start_ms: u64,
    pub estimated_finish_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedAssignment {
    pub work_id: WorkId,
    pub device_id: DeviceId,
    pub placement: CandidatePlacement,
    pub estimated_start_ms: u64,
    pub estimated_finish_ms: u64,
    pub immediate: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceLane {
    pub device_id: DeviceId,
    pub assignments: Vec<PlannedAssignment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockedReason {
    NotReady,
    DeviceDisabled,
    DeviceDraining,
    DeviceStartupExcluded,
    DeviceUnavailable,
    DeviceDegraded,
    NoSchedulableDevice,
    NoIdleDevice,
    HardPinUnavailable,
    BackendUnsupported,
    InsufficientVram,
    InsufficientHostRam,
    AggregateHostRamReserved,
    ModelNotInstalled,
    ExecutionPlanIncompatible,
    QueuePaused,
    MaintenanceMode,
    Cancelling,
    WarmWait,
    LowerPriorityOpening,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockedWork {
    pub work_id: WorkId,
    pub reason: BlockedReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WarmWait {
    pub work_id: WorkId,
    pub warm_device_id: DeviceId,
    pub started_at_ms: u64,
    pub deadline_ms: u64,
    pub predicted_warm_finish_ms: u64,
    pub best_cold_finish_ms: u64,
    pub declined_device_ids: Vec<DeviceId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BypassUpdate {
    pub work_id: WorkId,
    pub previous_count: u8,
    pub new_count: u8,
    pub younger_work_id: WorkId,
    pub opening_device_id: DeviceId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReservationItem {
    pub work_id: WorkId,
    pub device_id: DeviceId,
    pub execution_fingerprint: ExecutionFingerprint,
    pub predicted_vram_bytes: u64,
    pub device_available_vram_bytes: u64,
    pub host_ram_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchingReservation {
    pub sample_generation: u64,
    pub ledger_sequence: u64,
    pub total_host_ram_bytes: u64,
    pub items: Vec<ReservationItem>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizerState {
    Optimized,
    SeedOnly,
    WatchdogFallback,
    InternalErrorFallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub plan_version: u64,
    pub state_version: u64,
    pub created_at_ms: u64,
    pub next_replan_at_ms: Option<u64>,
    pub immediate_leases: Vec<ImmediateLease>,
    pub lanes: Vec<DeviceLane>,
    pub blocked: Vec<BlockedWork>,
    pub warm_waits: Vec<WarmWait>,
    pub bypass_updates: Vec<BypassUpdate>,
    pub reservation: MatchingReservation,
    pub optimization_horizon_length: usize,
    pub operation_budget: usize,
    pub operations_evaluated: usize,
    pub optimizer_state: OptimizerState,
}

impl Plan {
    pub fn blocked_reason(&self, work_id: &WorkId) -> Option<&BlockedReason> {
        self.blocked
            .iter()
            .find(|blocked| &blocked.work_id == work_id)
            .map(|blocked| &blocked.reason)
    }

    pub fn validate_for_grant(
        &self,
        state_version: u64,
        plan_version: u64,
        sample_generation: u64,
        ledger_sequence: u64,
    ) -> Result<(), PlanValidationError> {
        if state_version != self.state_version {
            return Err(PlanValidationError::StaleState {
                planned: self.state_version,
                current: state_version,
            });
        }
        if plan_version != self.plan_version {
            return Err(PlanValidationError::StalePlan {
                planned: self.plan_version,
                current: plan_version,
            });
        }
        if sample_generation != self.reservation.sample_generation {
            return Err(PlanValidationError::StaleMemorySample {
                planned: self.reservation.sample_generation,
                current: sample_generation,
            });
        }
        if ledger_sequence != self.reservation.ledger_sequence {
            return Err(PlanValidationError::StaleMemoryLedger {
                planned: self.reservation.ledger_sequence,
                current: ledger_sequence,
            });
        }
        Ok(())
    }

    /// Validates the plan-wide and runtime-owned fences for one proposed lease.
    ///
    /// The coordinator must build `current` from the same reducer turn and
    /// device/work identities as `lease`. Passing this method is required
    /// immediately before the worker grant; plan construction alone never
    /// authorizes CUDA work.
    pub fn validate_lease_for_grant(
        &self,
        lease: &ImmediateLease,
        current: &GrantValidationSnapshot,
    ) -> Result<(), PlanValidationError> {
        self.validate_for_grant(
            current.state_version,
            current.plan_version,
            current.sample_generation,
            current.ledger_sequence,
        )?;
        if current.work_id != lease.work_id {
            return Err(PlanValidationError::GrantWorkMismatch {
                planned: lease.work_id.clone(),
                current: current.work_id.clone(),
            });
        }
        if current.device_id != lease.device_id {
            return Err(PlanValidationError::GrantDeviceMismatch {
                planned: lease.device_id.clone(),
                current: current.device_id.clone(),
            });
        }
        if current.execution_fingerprint != lease.placement.execution_fingerprint {
            return Err(PlanValidationError::ExecutionFingerprintChanged {
                planned: lease.placement.execution_fingerprint.clone(),
                current: current.execution_fingerprint.clone(),
            });
        }
        if !lease.placement.frozen_device_capacity
            && current.available_vram_bytes < lease.placement.predicted_vram_bytes
        {
            return Err(PlanValidationError::InsufficientCurrentVram {
                device_id: lease.device_id.clone(),
                planned_bytes: lease.placement.predicted_vram_bytes,
                available_bytes: current.available_vram_bytes,
            });
        }
        // Free-VRAM telemetry is an observation, not execution identity. A
        // harmless allocator/context delta may change it between planning and
        // this fence even though the freshly resolved fingerprint is identical.
        // Requiring byte equality turns that ordinary drift into an endless
        // plan/replan loop. The immutable fingerprint above protects the chosen
        // placement and load strategy; the lower-bound check protects capacity.
        if !self.immediate_leases.contains(lease) {
            return Err(PlanValidationError::LeaseNotProposed {
                work_id: lease.work_id.clone(),
                device_id: lease.device_id.clone(),
            });
        }
        if current.work_cancelled {
            return Err(PlanValidationError::WorkCancelled {
                work_id: lease.work_id.clone(),
            });
        }
        if !current.work_ready {
            return Err(PlanValidationError::WorkNotReady {
                work_id: lease.work_id.clone(),
            });
        }
        if current.worker_generation != lease.worker_generation {
            return Err(PlanValidationError::StaleWorkerGeneration {
                device_id: lease.device_id.clone(),
                planned: lease.worker_generation,
                current: current.worker_generation,
            });
        }
        if !current.worker_ready {
            return Err(PlanValidationError::WorkerNotReady {
                device_id: lease.device_id.clone(),
            });
        }
        if current.device_admin_state != DeviceAdminState::Enabled
            || current.device_health != DeviceHealth::Healthy
        {
            return Err(PlanValidationError::DeviceNotSchedulable {
                device_id: lease.device_id.clone(),
                admin_state: current.device_admin_state,
                health: current.device_health,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrantValidationSnapshot {
    pub work_id: WorkId,
    pub device_id: DeviceId,
    pub state_version: u64,
    pub plan_version: u64,
    pub sample_generation: u64,
    pub ledger_sequence: u64,
    pub work_ready: bool,
    pub work_cancelled: bool,
    pub worker_generation: u64,
    pub worker_ready: bool,
    pub device_admin_state: DeviceAdminState,
    pub device_health: DeviceHealth,
    /// Exact plan identity produced by fresh pre-grant resolution.
    pub execution_fingerprint: ExecutionFingerprint,
    /// Fresh effective VRAM capacity for the selected device.
    pub available_vram_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanValidationError {
    StaleState {
        planned: u64,
        current: u64,
    },
    StalePlan {
        planned: u64,
        current: u64,
    },
    StaleMemorySample {
        planned: u64,
        current: u64,
    },
    StaleMemoryLedger {
        planned: u64,
        current: u64,
    },
    LeaseNotProposed {
        work_id: WorkId,
        device_id: DeviceId,
    },
    GrantWorkMismatch {
        planned: WorkId,
        current: WorkId,
    },
    GrantDeviceMismatch {
        planned: DeviceId,
        current: DeviceId,
    },
    ExecutionFingerprintChanged {
        planned: ExecutionFingerprint,
        current: ExecutionFingerprint,
    },
    InsufficientCurrentVram {
        device_id: DeviceId,
        planned_bytes: u64,
        available_bytes: u64,
    },
    /// Retained for downstream source compatibility. Grant validation no
    /// longer emits this error because a changed sample is safe when the
    /// execution fingerprint is unchanged and the predicted peak still fits.
    DeviceVramSampleChanged {
        device_id: DeviceId,
        planned_bytes: u64,
        current_bytes: u64,
    },
    WorkNotReady {
        work_id: WorkId,
    },
    WorkCancelled {
        work_id: WorkId,
    },
    StaleWorkerGeneration {
        device_id: DeviceId,
        planned: u64,
        current: u64,
    },
    WorkerNotReady {
        device_id: DeviceId,
    },
    DeviceNotSchedulable {
        device_id: DeviceId,
        admin_state: DeviceAdminState,
        health: DeviceHealth,
    },
}
