//! Restart-time dispatch rollout contract.
//!
//! Startup owns the actual worker topology. This module intentionally keeps
//! parsing and ownership semantics pure so legacy/observe startup cannot
//! accidentally create V2 owner workers.

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
        std::env::var("MOLD_DISPATCH_MODE")
            .ok()
            .map_or(Ok(Self::V2), |value| Self::parse(&value))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_modes_have_unambiguous_worker_ownership() {
        assert_eq!(DispatchMode::parse("legacy"), Ok(DispatchMode::Legacy));
        assert_eq!(DispatchMode::parse(" OBSERVE "), Ok(DispatchMode::Observe));
        assert_eq!(DispatchMode::parse("v2"), Ok(DispatchMode::V2));
        assert!(!DispatchMode::Legacy.owns_v2_workers());
        assert!(!DispatchMode::Observe.owns_v2_workers());
        assert!(DispatchMode::V2.owns_v2_workers());
        assert!(DispatchMode::Observe.records_v2_observations());
        assert!(DispatchMode::parse("scheduler").is_err());
    }
}
