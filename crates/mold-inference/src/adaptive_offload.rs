//! Shared adaptive block-residency planner for mold-owned offload paths.

pub(crate) const ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM: u64 = 2_000_000_000;
const RESIDENCY_DP_MAX_UNITS: u64 = 50_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdaptiveResidencyPlan {
    pub(crate) resident: Vec<bool>,
    pub(crate) resident_bytes: u64,
    pub(crate) streamed_bytes: u64,
    pub(crate) largest_streamed_block: u64,
    pub(crate) activation_budget: u64,
    pub(crate) runtime_headroom: u64,
}

impl AdaptiveResidencyPlan {
    pub(crate) fn full_streaming(
        block_sizes: &[usize],
        activation_budget: u64,
        runtime_headroom: u64,
    ) -> Self {
        let total = block_sizes.iter().map(|&s| s as u64).sum();
        let largest_streamed_block = block_sizes.iter().copied().max().unwrap_or(0) as u64;
        Self {
            resident: vec![false; block_sizes.len()],
            resident_bytes: 0,
            streamed_bytes: total,
            largest_streamed_block,
            activation_budget,
            runtime_headroom,
        }
    }

    pub(crate) fn resident_count(&self) -> usize {
        self.resident.iter().filter(|&&r| r).count()
    }

    pub(crate) fn streamed_count(&self) -> usize {
        self.resident.len() - self.resident_count()
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.activation_budget
            .saturating_add(self.runtime_headroom)
            .saturating_add(self.largest_streamed_block)
    }

    pub(crate) fn peak_bytes(&self) -> u64 {
        self.resident_bytes.saturating_add(self.reserved_bytes())
    }

    pub(crate) fn demote_largest_resident(&mut self, block_sizes: &[usize]) -> bool {
        let Some((idx, _)) = self
            .resident
            .iter()
            .enumerate()
            .filter(|(_, resident)| **resident)
            .map(|(idx, _)| (idx, block_sizes[idx]))
            .max_by_key(|(_, size)| *size)
        else {
            return false;
        };
        self.resident[idx] = false;
        self.recompute(block_sizes);
        true
    }

    fn recompute(&mut self, block_sizes: &[usize]) {
        self.resident_bytes = 0;
        self.streamed_bytes = 0;
        self.largest_streamed_block = 0;
        for (&is_resident, &size) in self.resident.iter().zip(block_sizes) {
            let size = size as u64;
            if is_resident {
                self.resident_bytes = self.resident_bytes.saturating_add(size);
            } else {
                self.streamed_bytes = self.streamed_bytes.saturating_add(size);
                self.largest_streamed_block = self.largest_streamed_block.max(size);
            }
        }
    }
}

fn ceil_div(n: u64, d: u64) -> u64 {
    if n == 0 {
        0
    } else {
        1 + (n - 1) / d
    }
}

fn choose_optional_residents(items: &[(usize, u64)], capacity: u64, quantum: u64) -> (u64, u128) {
    if items.is_empty() || capacity == 0 {
        return (0, 0);
    }
    if items.len() > 128 {
        return choose_optional_residents_greedy(items, capacity);
    }

    let units_cap = (capacity / quantum).min(RESIDENCY_DP_MAX_UNITS) as usize;
    if units_cap == 0 {
        return (0, 0);
    }

    let mut dp: Vec<Option<(u64, u128)>> = vec![None; units_cap + 1];
    dp[0] = Some((0, 0));
    for (item_pos, &(_, size)) in items.iter().enumerate() {
        let item_units = ceil_div(size, quantum) as usize;
        if item_units > units_cap {
            continue;
        }
        let bit = 1u128 << item_pos;
        for used in (0..=units_cap - item_units).rev() {
            let Some((value, mask)) = dp[used] else {
                continue;
            };
            let next = used + item_units;
            let candidate = (value.saturating_add(size), mask | bit);
            if dp[next]
                .map(|current| candidate.0 > current.0)
                .unwrap_or(true)
            {
                dp[next] = Some(candidate);
            }
        }
    }

    dp.into_iter()
        .flatten()
        .max_by_key(|(value, _)| *value)
        .unwrap_or((0, 0))
}

fn choose_optional_residents_greedy(items: &[(usize, u64)], capacity: u64) -> (u64, u128) {
    let mut order: Vec<(usize, u64, usize)> = items
        .iter()
        .enumerate()
        .map(|(pos, &(idx, size))| (idx, size, pos))
        .collect();
    order.sort_by_key(|&(idx, size, _)| (std::cmp::Reverse(size), idx));

    let mut used = 0u64;
    let mut mask = 0u128;
    for (_, size, pos) in order {
        if used.saturating_add(size) <= capacity {
            used += size;
            if pos < 128 {
                mask |= 1u128 << pos;
            }
        }
    }
    (used, mask)
}

pub(crate) fn plan_adaptive_residency(
    block_sizes: &[usize],
    free_vram: u64,
    activation_budget: u64,
    runtime_headroom: u64,
) -> AdaptiveResidencyPlan {
    if block_sizes.is_empty() || free_vram == 0 {
        return AdaptiveResidencyPlan::full_streaming(
            block_sizes,
            activation_budget,
            runtime_headroom,
        );
    }

    let base_reserve = activation_budget.saturating_add(runtime_headroom);
    if free_vram <= base_reserve {
        return AdaptiveResidencyPlan::full_streaming(
            block_sizes,
            activation_budget,
            runtime_headroom,
        );
    }

    let mut reserve_candidates: Vec<u64> = block_sizes.iter().map(|&s| s as u64).collect();
    reserve_candidates.push(0);
    reserve_candidates.sort_unstable();
    reserve_candidates.dedup();

    let total_bytes: u64 = block_sizes.iter().map(|&s| s as u64).sum();
    let mut best: Option<AdaptiveResidencyPlan> = None;

    for streamed_reserve in reserve_candidates {
        let Some(capacity) = free_vram
            .checked_sub(base_reserve)
            .and_then(|v| v.checked_sub(streamed_reserve))
        else {
            continue;
        };

        let mut resident = vec![false; block_sizes.len()];
        let mut required_bytes = 0u64;
        let mut optional = Vec::new();
        for (idx, &size) in block_sizes.iter().enumerate() {
            let size = size as u64;
            if size > streamed_reserve {
                resident[idx] = true;
                required_bytes = required_bytes.saturating_add(size);
            } else {
                optional.push((idx, size));
            }
        }
        if required_bytes > capacity {
            continue;
        }

        let optional_capacity = capacity - required_bytes;
        let quantum = (optional_capacity / RESIDENCY_DP_MAX_UNITS).max(1);
        let (optional_bytes, optional_mask) =
            choose_optional_residents(&optional, optional_capacity, quantum);
        for (pos, &(idx, _)) in optional.iter().enumerate() {
            if pos < 128 && (optional_mask & (1u128 << pos)) != 0 {
                resident[idx] = true;
            }
        }

        let mut plan = AdaptiveResidencyPlan {
            resident,
            resident_bytes: required_bytes.saturating_add(optional_bytes),
            streamed_bytes: total_bytes.saturating_sub(required_bytes + optional_bytes),
            largest_streamed_block: 0,
            activation_budget,
            runtime_headroom,
        };
        plan.recompute(block_sizes);

        if plan.peak_bytes() > free_vram {
            continue;
        }

        let replace = best
            .as_ref()
            .map(|current| {
                plan.resident_bytes > current.resident_bytes
                    || (plan.resident_bytes == current.resident_bytes
                        && plan.streamed_count() < current.streamed_count())
            })
            .unwrap_or(true);
        if replace {
            best = Some(plan);
        }
    }

    best.unwrap_or_else(|| {
        AdaptiveResidencyPlan::full_streaming(block_sizes, activation_budget, runtime_headroom)
    })
}

#[cfg(test)]
mod tests {
    use super::{plan_adaptive_residency, AdaptiveResidencyPlan};

    #[test]
    fn adaptive_residency_keeps_all_blocks_when_they_fit() {
        let blocks = [100usize, 200, 300];
        let plan = plan_adaptive_residency(&blocks, 700, 50, 50);

        assert_eq!(plan.resident, vec![true, true, true]);
        assert_eq!(plan.resident_bytes, 600);
        assert_eq!(plan.streamed_bytes, 0);
        assert_eq!(plan.largest_streamed_block, 0);
        assert_eq!(plan.peak_bytes(), 700);
    }

    #[test]
    fn adaptive_residency_partially_streams_when_full_bf16_does_not_fit() {
        let blocks = [100usize, 200, 300];
        let plan = plan_adaptive_residency(&blocks, 650, 50, 50);

        assert_eq!(plan.resident_bytes, 300);
        assert_eq!(plan.streamed_bytes, 300);
        assert_eq!(plan.largest_streamed_block, 200);
        assert!(plan.peak_bytes() <= 650);
        assert_eq!(plan.resident_count(), 1);
    }

    #[test]
    fn adaptive_residency_maximizes_resident_bytes_without_exceeding_budget() {
        let blocks = [6usize, 4, 4];
        let plan = plan_adaptive_residency(&blocks, 10, 0, 0);

        assert_eq!(plan.resident_bytes, 6);
        assert_eq!(plan.streamed_bytes, 8);
        assert_eq!(plan.largest_streamed_block, 4);
        assert!(plan.peak_bytes() <= 10);
    }

    #[test]
    fn adaptive_residency_recomputes_stream_reserve_after_largest_block_is_resident() {
        let blocks = [10usize, 9, 8];
        let plan = plan_adaptive_residency(&blocks, 19, 0, 0);

        assert_eq!(
            plan.resident,
            vec![true, false, false],
            "keeping the 10-byte block resident lowers streamed reserve to 9 bytes"
        );
        assert_eq!(plan.resident_bytes, 10);
        assert_eq!(plan.largest_streamed_block, 9);
        assert!(plan.peak_bytes() <= 19);
    }

    #[test]
    fn adaptive_residency_low_budget_falls_back_to_full_streaming() {
        let blocks = [10usize, 20];
        let plan = plan_adaptive_residency(&blocks, 50, 40, 20);

        assert_eq!(plan, AdaptiveResidencyPlan::full_streaming(&blocks, 40, 20));
        assert_eq!(plan.resident_count(), 0);
        assert_eq!(plan.streamed_count(), 2);
    }
}
