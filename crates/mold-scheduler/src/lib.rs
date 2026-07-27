//! Pure, deterministic scheduling primitives for Mold.
//!
//! This crate deliberately owns no clocks, threads, runtimes, persistence, or
//! device APIs. Callers supply immutable snapshots and apply the returned,
//! versioned plan only after `Plan::validate_lease_for_grant` validates both
//! plan-wide generations and the coordinator-owned work, worker, and device
//! state immediately before dispatch.

mod batch;
mod eligibility;
mod estimates;
mod matching;
mod planner;
mod types;

pub use batch::{
    AdaptiveBatchPartition, AdaptiveBatchPlan, BatchDeviceProfile, BatchInfeasibilityReason,
    BatchOptimization, BatchPartitionError, BatchPartitionPlanner, BatchPartitionRequest,
    BatchSetupDisposition, BatchSizeEstimate, NativeBatchSizes, BOUNDED_HEURISTIC_STRATEGY_LIMIT,
    EXACT_SMALL_DEVICE_LIMIT,
};
pub use eligibility::EligibilityIndex;
pub use estimates::{
    EstimateBucket, EstimateConfidence, EstimateKey, EstimateObservation, EstimateOutcome,
    EstimatePhaseTimings, EstimateStore, ResolvedEstimate, StaticEstimate,
};
pub use planner::{operation_budget, optimization_horizon, Planner};
pub use types::{
    static_timing_for, static_timing_for_placement, AssignmentReason, Backend, BlockedReason,
    BlockedWork, BypassUpdate, CandidatePlacement, DeviceActivity, DeviceAdminState, DeviceHealth,
    DeviceId, DeviceLane, DeviceSnapshot, ExecutionFingerprint, GrantValidationSnapshot,
    HostMemorySnapshot, ImmediateLease, MatchingReservation, OptimizerState, ParentId, Plan,
    PlanValidationError, PlannedAssignment, PlannedBatchPartition, PlannerConfig, PlannerError,
    PlannerSnapshot, PlanningMode, PriorityClass, ReservationItem, StaticTimingEstimate, WarmWait,
    WorkId, WorkKind, WorkSnapshot,
};
