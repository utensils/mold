//! Shared adaptive block-residency planner for mold-owned offload paths.

pub(crate) const ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM: u64 = 2_000_000_000;
const RESIDENCY_DP_MAX_UNITS: u64 = 50_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdaptiveResidencyPlan {
    pub(crate) resident: Vec<bool>,
    pub(crate) resident_bytes: u64,
    pub(crate) streamed_bytes: u64,
    pub(crate) largest_streamed_block: u64,
    /// Weights that land on the GPU unconditionally, outside the streamed
    /// block set — for LTX-2 the ~2.1 GB of non-block transformer tensors
    /// (`patchify_proj`, `adaln_single.linear`, `caption_projection`, the
    /// connectors, `proj_out`) that `new_with_block_source` allocates *after*
    /// every resident block. Priced at zero before this existed, which is
    /// exactly how a plan that "fits" overshot the card.
    pub(crate) fixed_resident_bytes: u64,
    pub(crate) activation_budget: u64,
    pub(crate) runtime_headroom: u64,
}

impl AdaptiveResidencyPlan {
    pub(crate) fn full_streaming(
        block_sizes: &[usize],
        fixed_resident_bytes: u64,
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
            fixed_resident_bytes,
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
            .saturating_add(self.fixed_resident_bytes)
    }

    pub(crate) fn peak_bytes(&self) -> u64 {
        self.resident_bytes.saturating_add(self.reserved_bytes())
    }

    /// Move one resident block to the streamed set, preferring the demotion
    /// that frees the most bytes without growing the streaming staging buffer.
    ///
    /// Demoting a block larger than `largest_streamed_block` raises that
    /// reserve by the difference, so the retry nets only
    /// `largest_streamed_block` — and the bigger staging buffer is a real
    /// allocation the next attempt has to find. With a mixed BF16/FP8 block
    /// set (772 MB / 386 MB) the old "always demote the largest" rule turned
    /// every rung of the OOM ladder into a 386 MB net gain *and* a doubled
    /// staging buffer. Prefer the largest block that already fits inside the
    /// current reserve; fall back to the plain maximum when every resident
    /// block is larger (the first demotion out of a fully resident plan).
    pub(crate) fn demote_largest_resident(&mut self, block_sizes: &[usize]) -> bool {
        let resident_sizes = || {
            self.resident
                .iter()
                .enumerate()
                .filter(|(_, resident)| **resident)
                .map(|(idx, _)| (idx, block_sizes[idx] as u64))
        };
        let reserve = self.largest_streamed_block;
        let choice = resident_sizes()
            .filter(|(_, size)| *size <= reserve)
            .max_by_key(|(_, size)| *size)
            .or_else(|| resident_sizes().max_by_key(|(_, size)| *size));
        let Some((idx, _)) = choice else {
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

/// Choose which transformer blocks stay GPU-resident.
///
/// `fixed_resident_bytes` is weight memory the model allocates outside the
/// block set (see [`AdaptiveResidencyPlan::fixed_resident_bytes`]); it is part
/// of the base reserve, so it shrinks the capacity available to blocks instead
/// of being discovered after the plan has already committed. Callers with no
/// such weights pass `0`.
pub(crate) fn plan_adaptive_residency(
    block_sizes: &[usize],
    free_vram: u64,
    fixed_resident_bytes: u64,
    activation_budget: u64,
    runtime_headroom: u64,
) -> AdaptiveResidencyPlan {
    if block_sizes.is_empty() || free_vram == 0 {
        return AdaptiveResidencyPlan::full_streaming(
            block_sizes,
            fixed_resident_bytes,
            activation_budget,
            runtime_headroom,
        );
    }

    let base_reserve = activation_budget
        .saturating_add(runtime_headroom)
        .saturating_add(fixed_resident_bytes);
    if free_vram <= base_reserve {
        return AdaptiveResidencyPlan::full_streaming(
            block_sizes,
            fixed_resident_bytes,
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
            fixed_resident_bytes,
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
        AdaptiveResidencyPlan::full_streaming(
            block_sizes,
            fixed_resident_bytes,
            activation_budget,
            runtime_headroom,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{plan_adaptive_residency, AdaptiveResidencyPlan};

    /// The 19B FP8 LTX-2 block set: 6 BF16 blocks (block 0 included) and 42
    /// FP8 blocks, as measured from the checkpoint header.
    fn ltx2_19b_fp8_blocks() -> Vec<usize> {
        let mut blocks = vec![772_284_416usize; 6];
        blocks.extend(std::iter::repeat_n(386_408_672usize, 42));
        blocks
    }

    #[test]
    fn adaptive_residency_keeps_all_blocks_when_they_fit() {
        let blocks = [100usize, 200, 300];
        let plan = plan_adaptive_residency(&blocks, 700, 0, 50, 50);

        assert_eq!(plan.resident, vec![true, true, true]);
        assert_eq!(plan.resident_bytes, 600);
        assert_eq!(plan.streamed_bytes, 0);
        assert_eq!(plan.largest_streamed_block, 0);
        assert_eq!(plan.peak_bytes(), 700);
    }

    #[test]
    fn adaptive_residency_partially_streams_when_full_bf16_does_not_fit() {
        let blocks = [100usize, 200, 300];
        let plan = plan_adaptive_residency(&blocks, 650, 0, 50, 50);

        assert_eq!(plan.resident_bytes, 300);
        assert_eq!(plan.streamed_bytes, 300);
        assert_eq!(plan.largest_streamed_block, 200);
        assert!(plan.peak_bytes() <= 650);
        assert_eq!(plan.resident_count(), 1);
    }

    #[test]
    fn adaptive_residency_maximizes_resident_bytes_without_exceeding_budget() {
        let blocks = [6usize, 4, 4];
        let plan = plan_adaptive_residency(&blocks, 10, 0, 0, 0);

        assert_eq!(plan.resident_bytes, 6);
        assert_eq!(plan.streamed_bytes, 8);
        assert_eq!(plan.largest_streamed_block, 4);
        assert!(plan.peak_bytes() <= 10);
    }

    #[test]
    fn adaptive_residency_recomputes_stream_reserve_after_largest_block_is_resident() {
        let blocks = [10usize, 9, 8];
        let plan = plan_adaptive_residency(&blocks, 19, 0, 0, 0);

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
        let plan = plan_adaptive_residency(&blocks, 50, 0, 40, 20);

        assert_eq!(
            plan,
            AdaptiveResidencyPlan::full_streaming(&blocks, 0, 40, 20)
        );
        assert_eq!(plan.resident_count(), 0);
        assert_eq!(plan.streamed_count(), 2);
    }

    /// Weights allocated outside the block set are part of the peak. The
    /// 19B FP8 checkpoint carries 2.107 GB of non-block transformer tensors;
    /// before they were reserved the planner filled a 24 GB card to 19.32 GB
    /// of resident blocks and then allocated those on top.
    #[test]
    fn adaptive_plan_reserves_fixed_resident_bytes() {
        let blocks = ltx2_19b_fp8_blocks();
        const FIXED: u64 = 2_107_091_456;
        const FREE_VRAM: u64 = 25_339_395_072;
        const ACTIVATION: u64 = 3_544_186_880;

        let plan = plan_adaptive_residency(&blocks, FREE_VRAM, FIXED, ACTIVATION, 0);

        assert_eq!(plan.fixed_resident_bytes, FIXED);
        assert!(
            plan.reserved_bytes() >= FIXED,
            "non-block weights must be inside the reserve, got {}",
            plan.reserved_bytes()
        );
        assert!(
            plan.peak_bytes() <= FREE_VRAM,
            "peak {} must fit {FREE_VRAM} with the fixed bytes counted",
            plan.peak_bytes()
        );
        // The true demand is resident blocks + non-block weights + activation
        // + one streamed staging block.
        let true_demand = plan.resident_bytes + FIXED + ACTIVATION + plan.largest_streamed_block;
        assert!(
            true_demand <= FREE_VRAM,
            "true demand {true_demand} must fit {FREE_VRAM}"
        );
    }

    /// Fixed bytes big enough to swallow the card fall back to full streaming
    /// instead of pretending some blocks fit.
    #[test]
    fn adaptive_plan_falls_back_when_fixed_bytes_exceed_free_vram() {
        let blocks = [100usize, 200];
        let plan = plan_adaptive_residency(&blocks, 500, 480, 10, 10);
        assert_eq!(
            plan,
            AdaptiveResidencyPlan::full_streaming(&blocks, 480, 10, 10)
        );
    }

    /// Every rung of the OOM ladder must make progress: the peak never grows,
    /// and once anything is streamed each demotion strictly lowers it.
    #[test]
    fn demote_largest_resident_never_increases_peak() {
        let blocks = ltx2_19b_fp8_blocks();
        let mut plan =
            plan_adaptive_residency(&blocks, 25_339_395_072, 2_107_091_456, 3_544_186_880, 0);
        assert!(plan.resident_count() > 0);
        assert!(
            plan.streamed_count() > 0,
            "mix must start partially streamed"
        );

        let mut previous = plan.peak_bytes();
        while plan.demote_largest_resident(&blocks) {
            let peak = plan.peak_bytes();
            assert!(
                peak < previous,
                "demotion must strictly lower the peak: {previous} -> {peak}"
            );
            previous = peak;
        }
    }

    /// Demoting a 772 MB BF16 block when 386 MB FP8 blocks are already
    /// streamed doubles the staging reserve for no extra net gain. Prefer the
    /// block that already fits inside the current reserve.
    #[test]
    fn demote_largest_resident_never_grows_streamed_reserve() {
        let blocks = ltx2_19b_fp8_blocks();
        let mut plan =
            plan_adaptive_residency(&blocks, 25_339_395_072, 2_107_091_456, 3_544_186_880, 0);
        let reserve = plan.largest_streamed_block;
        assert_eq!(
            reserve, 386_408_672,
            "the planner should stream FP8 blocks first"
        );

        assert!(plan.demote_largest_resident(&blocks));
        assert_eq!(
            plan.largest_streamed_block, reserve,
            "demotion must not grow the streaming staging buffer while a \
             block that fits inside it is still resident"
        );
    }

    /// When every resident block is larger than the current reserve there is
    /// no cheaper choice, so the plain maximum still wins.
    #[test]
    fn demote_largest_resident_falls_back_to_plain_maximum() {
        let blocks = [1000usize, 400];
        let mut plan = plan_adaptive_residency(&blocks, 5_000, 0, 0, 0);
        assert_eq!(plan.resident, vec![true, true]);

        assert!(plan.demote_largest_resident(&blocks));
        assert_eq!(plan.resident, vec![false, true]);
        assert_eq!(plan.largest_streamed_block, 1000);
    }
}
