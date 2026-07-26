//! Restart-time dispatch rollout contract.
//!
//! Startup owns the actual worker topology. This module intentionally keeps
//! parsing and ownership semantics pure so legacy/observe startup cannot
//! accidentally create V2 owner workers.

use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DispatchMode {
    Legacy,
    Observe,
    #[default]
    V2,
}

impl DispatchMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            "observe" => Ok(Self::Observe),
            "v2" => Ok(Self::V2),
            other => Err(format!(
                "invalid MOLD_DISPATCH_MODE '{other}'; expected legacy, observe, or v2"
            )),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MOLD_DISPATCH_MODE") {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("MOLD_DISPATCH_MODE must be valid UTF-8".to_string())
            }
        }
    }

    pub fn from_optional_value(value: Option<&str>) -> Result<Self, String> {
        value.map_or(Ok(Self::default()), Self::parse)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Observe => "observe",
            Self::V2 => "v2",
        }
    }

    /// Only authoritative V2 dispatch may own rendezvous owner workers.
    /// Observe computes comparison plans at legacy dispatch points.
    pub const fn owns_v2_workers(self) -> bool {
        matches!(self, Self::V2)
    }

    pub const fn records_v2_observations(self) -> bool {
        matches!(self, Self::Observe)
    }
}

impl std::fmt::Display for DispatchMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DispatchObservation {
    pub work_id: String,
    pub work_kind: mold_scheduler::WorkKind,
    pub legacy_device_id: String,
    pub v2_device_id: Option<String>,
    pub blocked_reason: Option<mold_scheduler::BlockedReason>,
    pub eligible_idle_device_ids: Vec<String>,
    pub legacy_setup_warm: bool,
    pub v2_setup_warm: Option<bool>,
}

impl DispatchObservation {
    pub fn outcome(&self) -> &'static str {
        match self.v2_device_id.as_deref() {
            Some(device_id) if device_id == self.legacy_device_id => "matched",
            Some(_) => "diverged",
            None => "blocked",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DispatchObservationSnapshot {
    pub total: u64,
    pub matched: u64,
    pub diverged: u64,
    pub blocked: u64,
    pub last: Option<DispatchObservation>,
}

#[derive(Default)]
pub struct DispatchObservationRecorder {
    state: Mutex<DispatchObservationSnapshot>,
}

impl DispatchObservationRecorder {
    pub fn record(&self, observation: DispatchObservation) {
        let outcome = observation.outcome();
        tracing::debug!(
            work_id = %observation.work_id,
            work_kind = ?observation.work_kind,
            legacy_device_id = %observation.legacy_device_id,
            v2_device_id = ?observation.v2_device_id,
            blocked_reason = ?observation.blocked_reason,
            eligible_idle_device_ids = ?observation.eligible_idle_device_ids,
            legacy_setup = if observation.legacy_setup_warm { "warm" } else { "cold" },
            v2_setup = ?observation.v2_setup_warm.map(|warm| if warm { "warm" } else { "cold" }),
            outcome,
            "observed V2 decision at legacy dispatch point"
        );
        #[cfg(feature = "metrics")]
        crate::metrics::record_dispatch_observation(
            outcome,
            observation.legacy_setup_warm,
            observation.v2_setup_warm,
        );

        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.total = state.total.saturating_add(1);
        match outcome {
            "matched" => state.matched = state.matched.saturating_add(1),
            "diverged" => state.diverged = state.diverged.saturating_add(1),
            "blocked" => state.blocked = state.blocked.saturating_add(1),
            _ => unreachable!("DispatchObservation::outcome is exhaustive"),
        }
        state.last = Some(observation);
    }

    pub fn snapshot(&self) -> DispatchObservationSnapshot {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_modes_have_unambiguous_worker_ownership() {
        assert_eq!(DispatchMode::parse("legacy"), Ok(DispatchMode::Legacy));
        assert_eq!(DispatchMode::parse(" OBSERVE "), Ok(DispatchMode::Observe));
        assert_eq!(DispatchMode::parse("v2"), Ok(DispatchMode::V2));
        assert_eq!(DispatchMode::default(), DispatchMode::V2);
        assert!(!DispatchMode::Legacy.owns_v2_workers());
        assert!(!DispatchMode::Observe.owns_v2_workers());
        assert!(DispatchMode::V2.owns_v2_workers());
        assert!(!DispatchMode::Legacy.records_v2_observations());
        assert!(DispatchMode::Observe.records_v2_observations());
        assert!(!DispatchMode::V2.records_v2_observations());
        assert!(DispatchMode::parse("scheduler").is_err());
        assert!(DispatchMode::parse("").is_err());
    }

    #[test]
    fn missing_environment_uses_v2_but_present_invalid_values_fail() {
        assert_eq!(
            DispatchMode::from_optional_value(None),
            Ok(DispatchMode::V2)
        );
        assert_eq!(
            DispatchMode::from_optional_value(Some("legacy")),
            Ok(DispatchMode::Legacy)
        );
        let error = DispatchMode::from_optional_value(Some("round-robin")).unwrap_err();
        assert!(error.contains("MOLD_DISPATCH_MODE"));
        assert!(error.contains("legacy, observe, or v2"));
    }
}
