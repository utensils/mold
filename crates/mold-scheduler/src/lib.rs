//! Pure, deterministic scheduling primitives for Mold.
//!
//! This crate deliberately owns no clocks, threads, runtimes, persistence, or
//! device APIs. Callers supply immutable snapshots and apply the returned,
//! versioned plan only after `Plan::validate_lease_for_grant` validates both
//! plan-wide generations and the coordinator-owned work, worker, and device
//! state immediately before dispatch.

mod eligibility;
mod matching;
mod planner;
mod types;

pub use eligibility::EligibilityIndex;
pub use planner::{operation_budget, optimization_horizon, Planner};
pub use types::{
    AssignmentReason, Backend, BlockedReason, BlockedWork, BypassUpdate, CandidatePlacement,
    DeviceActivity, DeviceAdminState, DeviceHealth, DeviceId, DeviceLane, DeviceSnapshot,
    ExecutionFingerprint, GrantValidationSnapshot, HostMemorySnapshot, ImmediateLease,
    MatchingReservation, OptimizerState, ParentId, Plan, PlanValidationError, PlannedAssignment,
    PlannerConfig, PlannerError, PlannerSnapshot, PlanningMode, PriorityClass, ReservationItem,
    WarmWait, WorkId, WorkKind, WorkSnapshot,
};
