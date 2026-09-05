//! Shared Metal working-set policy. All sizes are bytes except the kernel's MiB.

use serde::{Deserialize, Serialize};

pub const MIB: u64 = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum MetalWiredLimit {
    Automatic,
    Explicit { mib: u32 },
    Unsupported,
    Unavailable,
}

/// Native observations, separate from the legacy CUDA memory attribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MetalMemorySnapshot {
    pub wired_limit: MetalWiredLimit,
    pub physical_bytes: Option<u64>,
    pub available_host_bytes: Option<u64>,
    pub recommended_bytes: Option<u64>,
    pub allocated_bytes: Option<u64>,
    pub effective_capacity_bytes: Option<u64>,
    pub allocation_headroom_bytes: Option<u64>,
    pub error: Option<String>,
}

pub fn host_safety_floor(total: u64) -> u64 {
    (total / 100 * 15 + total % 100 * 15 / 100).max(8 << 30)
}

impl MetalMemorySnapshot {
    pub fn resolve(mut self) -> Self {
        self.effective_capacity_bytes = self
            .physical_bytes
            .zip(self.recommended_bytes)
            .filter(|(physical, recommended)| *physical > 0 && *recommended > 0)
            .and_then(|(physical, recommended)| {
                let capacity =
                    recommended.min(physical.saturating_sub(host_safety_floor(physical)));
                match self.wired_limit {
                    MetalWiredLimit::Unavailable => None,
                    MetalWiredLimit::Explicit { mib } => Some(capacity.min(u64::from(mib) * MIB)),
                    MetalWiredLimit::Automatic | MetalWiredLimit::Unsupported => Some(capacity),
                }
            });
        self.allocation_headroom_bytes = self
            .effective_capacity_bytes
            .zip(self.allocated_bytes)
            .zip(self.available_host_bytes.zip(self.physical_bytes))
            .map(|((capacity, allocated), (available, physical))| {
                capacity.saturating_sub(allocated).min(
                    available
                        .min(physical)
                        .saturating_sub(host_safety_floor(physical)),
                )
            });
        self
    }

    pub fn with_reclaimable(&self, reclaimable: u64) -> u64 {
        self.allocation_headroom_bytes
            .zip(self.effective_capacity_bytes)
            .zip(self.allocated_bytes)
            .zip(self.available_host_bytes.zip(self.physical_bytes))
            .map_or(
                0,
                |(((headroom, capacity), allocated), (available, physical))| {
                    let credit = reclaimable.min(allocated);
                    headroom
                        .saturating_add(credit)
                        .min(capacity.saturating_sub(allocated.saturating_sub(credit)))
                        .min(
                            available
                                .min(physical)
                                .saturating_add(credit)
                                .min(physical)
                                .saturating_sub(host_safety_floor(physical)),
                        )
                },
            )
    }

    /// Shared compact host-status wording for CLI, TUI and MCP projections.
    pub fn budget_label(&self) -> String {
        let gib = |value: Option<u64>| {
            value.map_or_else(
                || "unavailable".into(),
                |bytes| format!("{:.1} GiB", bytes as f64 / (1_u64 << 30) as f64),
            )
        };
        format!(
            "Metal: {} capacity, {} headroom",
            gib(self.effective_capacity_bytes),
            gib(self.allocation_headroom_bytes)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    fn sample(limit: MetalWiredLimit) -> MetalMemorySnapshot {
        MetalMemorySnapshot {
            wired_limit: limit,
            physical_bytes: Some(48 * GIB),
            available_host_bytes: Some(32 * GIB),
            recommended_bytes: Some(37 * GIB),
            allocated_bytes: Some(4 * GIB),
            effective_capacity_bytes: None,
            allocation_headroom_bytes: None,
            error: None,
        }
    }

    #[test]
    fn metal_memory_automatic_is_not_zero_and_preserves_live_host_floor() {
        let s = sample(MetalWiredLimit::Automatic).resolve();
        assert_eq!(s.effective_capacity_bytes, Some(37 * GIB));
        assert_eq!(s.allocation_headroom_bytes, Some(24 * GIB));
    }

    #[test]
    fn metal_memory_explicit_decrease_clamps_stale_recommendation() {
        let s = sample(MetalWiredLimit::Explicit { mib: 16384 }).resolve();
        assert_eq!(s.effective_capacity_bytes, Some(16 * GIB));
        assert_eq!(s.allocation_headroom_bytes, Some(12 * GIB));
        assert_eq!(s.with_reclaimable(100 * GIB), 16 * GIB);
    }

    #[test]
    fn metal_memory_increase_and_reset_retain_the_observed_recommendation() {
        for limit in [
            MetalWiredLimit::Explicit { mib: 32768 },
            MetalWiredLimit::Automatic,
        ] {
            let mut s = sample(limit);
            s.recommended_bytes = Some(16 * GIB);
            let s = s.resolve();
            assert_eq!(s.effective_capacity_bytes, Some(16 * GIB));
            assert_eq!(s.allocation_headroom_bytes, Some(12 * GIB));
            assert_eq!(s.with_reclaimable(4 * GIB), 16 * GIB);
        }
    }

    #[test]
    fn metal_memory_small_host_and_allocations_above_limit_saturate() {
        let mut s = sample(MetalWiredLimit::Automatic);
        s.physical_bytes = Some(16 * GIB);
        s.available_host_bytes = Some(15 * GIB);
        assert_eq!(s.clone().resolve().effective_capacity_bytes, Some(8 * GIB));
        s.allocated_bytes = Some(9 * GIB);
        let s = s.resolve();
        assert_eq!(s.allocation_headroom_bytes, Some(0));
        assert_eq!(s.with_reclaimable(GIB), 0);
        assert_eq!(s.with_reclaimable(2 * GIB), GIB);
    }

    #[test]
    fn metal_memory_reclaim_must_first_recover_missing_host_floor() {
        let mut s = sample(MetalWiredLimit::Automatic);
        s.available_host_bytes = Some(2 * GIB);
        assert_eq!(s.resolve().with_reclaimable(4 * GIB), 0);
    }

    #[test]
    fn metal_memory_failed_probe_cannot_become_ram_capacity() {
        let mut s = sample(MetalWiredLimit::Automatic);
        s.recommended_bytes = None;
        let s = s.resolve();
        assert_eq!(s.effective_capacity_bytes, None);
        assert_eq!(s.allocation_headroom_bytes, None);
        assert_eq!(s.with_reclaimable(12 * GIB), 0);
        assert_eq!(
            sample(MetalWiredLimit::Unavailable)
                .resolve()
                .effective_capacity_bytes,
            None
        );
        assert!(sample(MetalWiredLimit::Unsupported)
            .resolve()
            .effective_capacity_bytes
            .is_some());
    }

    #[test]
    fn metal_memory_unknown_allocation_cannot_claim_incremental_headroom() {
        let mut s = sample(MetalWiredLimit::Automatic);
        s.allocated_bytes = None;
        let s = s.resolve();
        assert_eq!(s.effective_capacity_bytes, Some(37 * GIB));
        assert_eq!(s.allocation_headroom_bytes, None);
        assert_eq!(s.with_reclaimable(4 * GIB), 0);
    }

    #[test]
    fn metal_memory_wire_round_trip_includes_automatic_mode() {
        let s = sample(MetalWiredLimit::Automatic).resolve();
        let wire = serde_json::to_value(&s).unwrap();
        assert_eq!(wire["wired_limit"]["mode"], "automatic");
        assert_eq!(
            serde_json::from_value::<MetalMemorySnapshot>(wire).unwrap(),
            s
        );
    }
}
