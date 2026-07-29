use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{DeviceId, PlannedBatchPartition};

/// Validated native output counts declared by one inference family.
///
/// Singleton support is mandatory so every positive parent size has an exact
/// representation. Values are sorted and deduplicated; no geometric or
/// power-of-two sequence is assumed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBatchSizes(Vec<u32>);

impl NativeBatchSizes {
    pub fn canonicalize(sizes: impl IntoIterator<Item = u32>) -> Result<Self, BatchPartitionError> {
        let mut sizes = sizes.into_iter().collect::<Vec<_>>();
        if sizes.is_empty() {
            return Err(BatchPartitionError::EmptyNativeBatchSizes);
        }
        if sizes.contains(&0) {
            return Err(BatchPartitionError::ZeroNativeBatchSize);
        }
        sizes.sort_unstable();
        sizes.dedup();
        if sizes.first() != Some(&1) {
            return Err(BatchPartitionError::MissingSingletonCapability);
        }
        Ok(Self(sizes))
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchSizeEstimate {
    pub size: u32,
    pub predicted_run_ms: u64,
    pub predicted_vram_bytes: u64,
    pub predicted_host_ram_bytes: u64,
}

/// One already-eligible device projection for a future batch parent.
///
/// Callers resolve host, execution-fingerprint, determinism, enablement, and
/// health constraints before constructing this profile. The pure planner then
/// enforces native-size, device-capacity, VRAM, and aggregate host-RAM limits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchDeviceProfile {
    pub device_id: DeviceId,
    pub available_at_ms: u64,
    pub initially_warm: bool,
    pub partition_capacity: u32,
    pub available_vram_bytes: u64,
    pub cold_setup_ms: u64,
    pub warm_setup_ms: u64,
    pub setup_host_ram_bytes: u64,
    pub size_estimates: Vec<BatchSizeEstimate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchPartitionRequest {
    pub child_count: u32,
    pub now_ms: u64,
    pub native_batch_sizes: Vec<u32>,
    pub host_headroom_bytes: u64,
    pub devices: Vec<BatchDeviceProfile>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchSetupDisposition {
    Cold,
    Warm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveBatchPartition {
    /// One-based index matching the public queue batch projection.
    pub partition_index: u32,
    pub partition_count: u32,
    /// Zero-based first child index owned by this partition.
    pub child_start: u32,
    pub size: u32,
    pub device_id: DeviceId,
    pub estimated_start_ms: u64,
    pub estimated_finish_ms: u64,
    pub setup_disposition: BatchSetupDisposition,
    pub predicted_vram_bytes: u64,
    pub predicted_host_ram_bytes: u64,
}

impl AdaptiveBatchPartition {
    pub fn scheduler_projection(&self) -> PlannedBatchPartition {
        PlannedBatchPartition {
            index: self.partition_index,
            count: self.partition_count,
            size: self.size,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchPlanRepresentation {
    Materialized,
    CompactSingleton,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactSingletonLane {
    device_id: DeviceId,
    count: u32,
    first_start_ms: u64,
    first_finish_ms: u64,
    period_ms: u64,
    first_setup_disposition: BatchSetupDisposition,
    predicted_vram_bytes: u64,
    predicted_host_ram_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactSingletonSchedule {
    child_count: u32,
    lanes: Vec<CompactSingletonLane>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveBatchPartitions {
    materialized: Vec<AdaptiveBatchPartition>,
    compact_singleton: Option<CompactSingletonSchedule>,
}

impl AdaptiveBatchPartitions {
    fn materialized(partitions: Vec<AdaptiveBatchPartition>) -> Self {
        Self {
            materialized: partitions,
            compact_singleton: None,
        }
    }

    fn compact_singleton(child_count: u32, lanes: Vec<CompactSingletonLane>) -> Self {
        Self {
            materialized: Vec::new(),
            compact_singleton: Some(CompactSingletonSchedule { child_count, lanes }),
        }
    }

    pub fn representation(&self) -> BatchPlanRepresentation {
        if self.compact_singleton.is_some() {
            BatchPlanRepresentation::CompactSingleton
        } else {
            BatchPlanRepresentation::Materialized
        }
    }

    pub fn len(&self) -> usize {
        self.compact_singleton
            .as_ref()
            .map_or(self.materialized.len(), |schedule| {
                schedule.child_count as usize
            })
    }

    pub fn len_u32(&self) -> u32 {
        self.compact_singleton.as_ref().map_or_else(
            || u32::try_from(self.materialized.len()).unwrap_or(u32::MAX),
            |schedule| schedule.child_count,
        )
    }

    pub fn is_empty(&self) -> bool {
        self.len_u32() == 0
    }

    pub fn retained_records(&self) -> usize {
        self.compact_singleton
            .as_ref()
            .map_or(self.materialized.len(), |schedule| schedule.lanes.len())
    }

    pub fn get(&self, index: u32) -> Option<AdaptiveBatchPartition> {
        if let Some(schedule) = &self.compact_singleton {
            return schedule.partition_at(index);
        }
        self.materialized.get(index as usize).cloned()
    }

    pub fn iter(&self) -> AdaptiveBatchPartitionIter<'_> {
        AdaptiveBatchPartitionIter {
            partitions: self,
            next: 0,
        }
    }
}

pub struct AdaptiveBatchPartitionIter<'a> {
    partitions: &'a AdaptiveBatchPartitions,
    next: u32,
}

impl Iterator for AdaptiveBatchPartitionIter<'_> {
    type Item = AdaptiveBatchPartition;

    fn next(&mut self) -> Option<Self::Item> {
        let partition = self.partitions.get(self.next)?;
        self.next = self.next.checked_add(1)?;
        Some(partition)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.partitions.len_u32().saturating_sub(self.next) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AdaptiveBatchPartitionIter<'_> {}

impl CompactSingletonSchedule {
    fn partition_at(&self, index: u32) -> Option<AdaptiveBatchPartition> {
        if index >= self.child_count {
            return None;
        }
        let lane_count = self.lanes.len() as u32;
        let (lane, ordinal) = if index < lane_count {
            (&self.lanes[index as usize], 0)
        } else {
            self.additional_partition(index - lane_count)?
        };
        let estimated_finish_ms = lane
            .first_finish_ms
            .checked_add(u64::from(ordinal).checked_mul(lane.period_ms)?)?;
        let estimated_start_ms = if ordinal == 0 {
            lane.first_start_ms
        } else {
            lane.first_finish_ms
                .checked_add(u64::from(ordinal - 1).checked_mul(lane.period_ms)?)?
        };
        Some(AdaptiveBatchPartition {
            partition_index: index.checked_add(1)?,
            partition_count: self.child_count,
            child_start: index,
            size: 1,
            device_id: lane.device_id.clone(),
            estimated_start_ms,
            estimated_finish_ms,
            setup_disposition: if ordinal == 0 {
                lane.first_setup_disposition
            } else {
                BatchSetupDisposition::Warm
            },
            predicted_vram_bytes: lane.predicted_vram_bytes,
            predicted_host_ram_bytes: lane.predicted_host_ram_bytes,
        })
    }

    fn additional_partition(&self, rank: u32) -> Option<(&CompactSingletonLane, u32)> {
        let additional_count = self.child_count.checked_sub(self.lanes.len() as u32)?;
        if rank >= additional_count {
            return None;
        }
        let mut low = 0u64;
        let mut high = self
            .lanes
            .iter()
            .filter_map(|lane| {
                lane.count.checked_sub(1).and_then(|tail| {
                    lane.first_finish_ms
                        .checked_add(u64::from(tail).checked_mul(lane.period_ms)?)
                })
            })
            .max()?;
        while low < high {
            let middle = low + (high - low) / 2;
            let covered = self
                .lanes
                .iter()
                .map(|lane| u64::from(lane.additional_count_at(middle)))
                .sum::<u64>();
            if covered > u64::from(rank) {
                high = middle;
            } else {
                low = middle.checked_add(1)?;
            }
        }
        let threshold = low;
        let before_limit = threshold.checked_sub(1);
        let mut before_total = 0u64;
        for lane in &self.lanes {
            before_total = before_total.checked_add(u64::from(
                before_limit.map_or(0, |limit| lane.additional_count_at(limit)),
            ))?;
        }
        let mut tie_offset = u64::from(rank).checked_sub(before_total)?;
        for lane in &self.lanes {
            let before = before_limit.map_or(0, |limit| lane.additional_count_at(limit));
            let through = lane.additional_count_at(threshold);
            let ties = through.checked_sub(before)?;
            if tie_offset < u64::from(ties) {
                return Some((
                    lane,
                    before
                        .checked_add(1)?
                        .checked_add(u32::try_from(tie_offset).ok()?)?,
                ));
            }
            tie_offset = tie_offset.checked_sub(u64::from(ties))?;
        }
        None
    }
}

impl CompactSingletonLane {
    fn additional_count_at(&self, limit: u64) -> u32 {
        let available = self.count.saturating_sub(1);
        if available == 0 {
            return 0;
        }
        if self.period_ms == 0 {
            return if limit >= self.first_finish_ms {
                available
            } else {
                0
            };
        }
        let Some(second) = self.first_finish_ms.checked_add(self.period_ms) else {
            return 0;
        };
        if limit < second {
            0
        } else {
            u32::try_from(1 + ((limit - second) / self.period_ms).min(u64::from(available - 1)))
                .unwrap_or(available)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveBatchPlan {
    pub child_count: u32,
    pub partitions: AdaptiveBatchPartitions,
    pub predicted_parent_makespan_ms: u64,
    pub predicted_sum_completion_ms: u128,
    pub predicted_setup_ms: u128,
    pub predicted_peak_host_ram_bytes: u64,
    pub devices_used: usize,
    /// Whether the returned objective is proven exact within the named
    /// production-safe class or came from the deterministic bounded fallback.
    pub optimization: BatchOptimization,
    /// Compact activation/allocation objectives scored by the selected solver.
    pub optimizer_states_evaluated: usize,
    /// Full `AdaptiveBatchPartition` materialization passes. Compact singleton
    /// plans retain arithmetic lanes and therefore report zero.
    pub materialization_passes: u8,
}

impl AdaptiveBatchPlan {
    pub fn representation(&self) -> BatchPlanRepresentation {
        self.partitions.representation()
    }

    pub fn partition_count(&self) -> u32 {
        self.partitions.len_u32()
    }

    pub fn retained_partition_records(&self) -> usize {
        self.partitions.retained_records()
    }

    pub fn partition_at(&self, index: u32) -> Result<AdaptiveBatchPartition, BatchPartitionError> {
        if index >= self.partition_count() {
            return Err(BatchPartitionError::PartitionIndexOutOfRange {
                index,
                partition_count: self.partition_count(),
            });
        }
        self.partitions
            .get(index)
            .ok_or(BatchPartitionError::TimingOverflow)
    }

    pub fn partition_window(
        &self,
        start: u32,
        count: u32,
    ) -> Result<Vec<AdaptiveBatchPartition>, BatchPartitionError> {
        if count > MAX_PARTITION_WINDOW {
            return Err(BatchPartitionError::PartitionWindowTooLarge {
                requested: count,
                maximum: MAX_PARTITION_WINDOW,
            });
        }
        if start > self.partition_count() || (count > 0 && start == self.partition_count()) {
            return Err(BatchPartitionError::PartitionIndexOutOfRange {
                index: start,
                partition_count: self.partition_count(),
            });
        }
        let available = self.partition_count().saturating_sub(start);
        (start..start.saturating_add(count.min(available)))
            .map(|index| self.partition_at(index))
            .collect()
    }
}

/// Exact exhaustive activation/allocation is deliberately bounded. Larger
/// heterogeneous/native-batch instances use the observable bounded heuristic.
pub const EXACT_SMALL_DEVICE_LIMIT: usize = 8;
pub const BOUNDED_HEURISTIC_STRATEGY_LIMIT: usize = 64;
pub const NATIVE_MATERIALIZATION_CHILD_LIMIT: u32 = 100_003;
pub const MAX_PARTITION_WINDOW: u32 = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchOptimization {
    /// Exact for identical singleton completion streams at arbitrary parent
    /// size and device count.
    ExactHomogeneousSingleton,
    /// Exact exhaustive activation plus completion-stream allocation within
    /// [`EXACT_SMALL_DEVICE_LIMIT`] with arbitrary positive child count.
    ExactSmall,
    /// Deterministic bounded strategy comparison for the general NP-hard
    /// heterogeneous/native-batching problem.
    BoundedHeuristic,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BatchInfeasibilityReason {
    DeviceCapacity,
    InsufficientVram,
    InsufficientHostRam,
    MissingSizeEstimate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchPartitionError {
    EmptyParent,
    EmptyNativeBatchSizes,
    ZeroNativeBatchSize,
    MissingSingletonCapability,
    NoCandidateDevices,
    DuplicateDeviceId {
        device_id: DeviceId,
    },
    ZeroDeviceCapacity {
        device_id: DeviceId,
    },
    DuplicateSizeEstimate {
        device_id: DeviceId,
        size: u32,
    },
    UnsupportedSizeEstimate {
        device_id: DeviceId,
        size: u32,
    },
    TimingOverflow,
    HostMemoryOverflow,
    PartitionCountOverflow,
    PartitionWindowTooLarge {
        requested: u32,
        maximum: u32,
    },
    PartitionIndexOutOfRange {
        index: u32,
        partition_count: u32,
    },
    Infeasible {
        remaining_children: u32,
        reasons: BTreeSet<BatchInfeasibilityReason>,
    },
}

impl fmt::Display for BatchPartitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyParent => formatter.write_str("batch parent has no children"),
            Self::EmptyNativeBatchSizes => {
                formatter.write_str("native batch sizes must not be empty")
            }
            Self::ZeroNativeBatchSize => formatter.write_str("native batch sizes must be non-zero"),
            Self::MissingSingletonCapability => {
                formatter.write_str("native batch sizes must include singleton support")
            }
            Self::NoCandidateDevices => {
                formatter.write_str("batch parent has no candidate devices")
            }
            Self::DuplicateDeviceId { device_id } => {
                write!(
                    formatter,
                    "duplicate batch candidate device ID: {device_id}"
                )
            }
            Self::ZeroDeviceCapacity { device_id } => {
                write!(
                    formatter,
                    "batch candidate device {device_id} has zero capacity"
                )
            }
            Self::DuplicateSizeEstimate { device_id, size } => write!(
                formatter,
                "batch candidate device {device_id} has duplicate size-{size} estimates"
            ),
            Self::UnsupportedSizeEstimate { device_id, size } => write!(
                formatter,
                "batch candidate device {device_id} estimates undeclared native size {size}"
            ),
            Self::TimingOverflow => formatter.write_str("predicted batch timing overflowed"),
            Self::HostMemoryOverflow => formatter.write_str("predicted host memory overflowed"),
            Self::PartitionCountOverflow => {
                formatter.write_str("batch partition count exceeds the u32 projection")
            }
            Self::PartitionWindowTooLarge { requested, maximum } => write!(
                formatter,
                "batch partition window of {requested} exceeds the bounded maximum {maximum}"
            ),
            Self::PartitionIndexOutOfRange {
                index,
                partition_count,
            } => write!(
                formatter,
                "batch partition index {index} is outside the {partition_count}-partition plan"
            ),
            Self::Infeasible {
                remaining_children,
                reasons,
            } => write!(
                formatter,
                "cannot cover the remaining {remaining_children} batch children: {reasons:?}"
            ),
        }
    }
}

impl std::error::Error for BatchPartitionError {}

/// Pure, deterministic parent partition planner.
///
/// The planner compares one complete schedule for every capability-derived
/// maximum partition size. It is exact for homogeneous singleton fleets and
/// for singleton devices whose next completion streams can be merged
/// independently. Singleton plans retain `O(D)` arithmetic lanes and expose
/// lazy random access plus bounded windows rather than allocating `O(N)`
/// partition records. Native heterogeneous batching uses a deterministic
/// bounded strategy comparison for ordinary parent sizes; parents above
/// [`NATIVE_MATERIALIZATION_CHILD_LIMIT`] visibly fall back to the mandatory
/// compact singleton capability. The bounded native path deliberately does
/// not claim global optimality for the NP-hard unrelated-machine batching
/// problem.
pub struct BatchPartitionPlanner;

impl BatchPartitionPlanner {
    pub fn plan(request: &BatchPartitionRequest) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
        let prepared = PreparedRequest::new(request)?;
        if prepared.native_sizes.as_slice() == [1] {
            return plan_singleton(&prepared);
        }
        if request.child_count > NATIVE_MATERIALIZATION_CHILD_LIMIT {
            let mut fallback = plan_singleton(&prepared)?;
            fallback.optimization = BatchOptimization::BoundedHeuristic;
            return Ok(fallback);
        }
        let mut best: Option<AdaptiveBatchPlan> = None;
        let mut infeasible_reasons = BTreeSet::new();
        let mut minimum_remaining = request.child_count;
        let all_strategy_caps = prepared
            .native_sizes
            .as_slice()
            .iter()
            .map(|size| (*size).min(request.child_count))
            .collect::<BTreeSet<_>>();
        let strategy_caps = if all_strategy_caps.len() <= BOUNDED_HEURISTIC_STRATEGY_LIMIT {
            all_strategy_caps
        } else {
            let mut bounded = BTreeSet::from([1]);
            bounded.extend(
                all_strategy_caps
                    .iter()
                    .rev()
                    .take(BOUNDED_HEURISTIC_STRATEGY_LIMIT - 1)
                    .copied(),
            );
            bounded
        };
        let strategies_evaluated = strategy_caps.len();
        let mut successful_materializations = 0u8;

        for cap in strategy_caps {
            match plan_with_size_cap(&prepared, cap) {
                Ok(candidate) => {
                    successful_materializations += 1;
                    if best
                        .as_ref()
                        .is_none_or(|current| plan_cmp(&candidate, current).is_lt())
                    {
                        best = Some(candidate);
                    }
                }
                Err(BatchPartitionError::Infeasible {
                    remaining_children,
                    reasons,
                }) => {
                    minimum_remaining = minimum_remaining.min(remaining_children);
                    infeasible_reasons.extend(reasons);
                }
                Err(error) => return Err(error),
            }
        }

        best.map(|mut plan| {
            plan.optimizer_states_evaluated = strategies_evaluated;
            plan.materialization_passes = successful_materializations;
            plan
        })
        .ok_or(BatchPartitionError::Infeasible {
            remaining_children: minimum_remaining,
            reasons: infeasible_reasons,
        })
    }
}

struct PreparedRequest<'a> {
    request: &'a BatchPartitionRequest,
    native_sizes: NativeBatchSizes,
    devices: Vec<PreparedDevice<'a>>,
}

struct PreparedDevice<'a> {
    profile: &'a BatchDeviceProfile,
    estimates: BTreeMap<u32, BatchSizeEstimate>,
}

impl<'a> PreparedRequest<'a> {
    fn new(request: &'a BatchPartitionRequest) -> Result<Self, BatchPartitionError> {
        if request.child_count == 0 {
            return Err(BatchPartitionError::EmptyParent);
        }
        let native_sizes =
            NativeBatchSizes::canonicalize(request.native_batch_sizes.iter().copied())?;
        if request.devices.is_empty() {
            return Err(BatchPartitionError::NoCandidateDevices);
        }

        let allowed = native_sizes
            .as_slice()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut device_ids = BTreeSet::new();
        let mut devices = Vec::with_capacity(request.devices.len());
        for profile in &request.devices {
            if !device_ids.insert(profile.device_id.clone()) {
                return Err(BatchPartitionError::DuplicateDeviceId {
                    device_id: profile.device_id.clone(),
                });
            }
            if profile.partition_capacity == 0 {
                return Err(BatchPartitionError::ZeroDeviceCapacity {
                    device_id: profile.device_id.clone(),
                });
            }
            let mut estimates = BTreeMap::new();
            for estimate in &profile.size_estimates {
                if !allowed.contains(&estimate.size) {
                    return Err(BatchPartitionError::UnsupportedSizeEstimate {
                        device_id: profile.device_id.clone(),
                        size: estimate.size,
                    });
                }
                if estimates.insert(estimate.size, *estimate).is_some() {
                    return Err(BatchPartitionError::DuplicateSizeEstimate {
                        device_id: profile.device_id.clone(),
                        size: estimate.size,
                    });
                }
            }
            devices.push(PreparedDevice { profile, estimates });
        }
        devices.sort_by(|left, right| left.profile.device_id.cmp(&right.profile.device_id));

        Ok(Self {
            request,
            native_sizes,
            devices,
        })
    }
}

#[derive(Clone, Debug)]
struct DeviceSchedule {
    next_available_ms: u64,
    used: bool,
    peak_partition_host_ram_bytes: u64,
}

struct Candidate<'a> {
    device_index: usize,
    estimate: &'a BatchSizeEstimate,
    start_ms: u64,
    finish_ms: u64,
    setup_ms: u64,
    setup_disposition: BatchSetupDisposition,
    host_delta_bytes: u64,
    activates_device: bool,
}

#[derive(Clone, Copy)]
struct SingletonDevice {
    device_index: usize,
    estimate: BatchSizeEstimate,
    activation_host_bytes: u64,
    first_finish_ms: u128,
}

#[derive(Clone, Debug)]
struct SingletonAllocation {
    counts: Vec<u32>,
    makespan_ms: u128,
    sum_completion_ms: u128,
    setup_ms: u128,
}

fn plan_singleton(
    prepared: &PreparedRequest<'_>,
) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
    let mut reasons = BTreeSet::new();
    let mut viable = Vec::new();
    for (device_index, device) in prepared.devices.iter().enumerate() {
        if device.profile.partition_capacity < 1 {
            reasons.insert(BatchInfeasibilityReason::DeviceCapacity);
            continue;
        }
        let Some(estimate) = device.estimates.get(&1).copied() else {
            reasons.insert(BatchInfeasibilityReason::MissingSizeEstimate);
            continue;
        };
        if estimate.predicted_vram_bytes > device.profile.available_vram_bytes {
            reasons.insert(BatchInfeasibilityReason::InsufficientVram);
            continue;
        }
        let activation_host_bytes = device
            .profile
            .setup_host_ram_bytes
            .checked_add(estimate.predicted_host_ram_bytes)
            .ok_or(BatchPartitionError::HostMemoryOverflow)?;
        let first_setup = if device.profile.initially_warm {
            device.profile.warm_setup_ms
        } else {
            device.profile.cold_setup_ms
        };
        let first_finish_ms =
            u128::from(device.profile.available_at_ms.max(prepared.request.now_ms))
                .checked_add(u128::from(first_setup))
                .and_then(|value| value.checked_add(u128::from(estimate.predicted_run_ms)))
                .ok_or(BatchPartitionError::TimingOverflow)?;
        viable.push(SingletonDevice {
            device_index,
            estimate,
            activation_host_bytes,
            first_finish_ms,
        });
    }
    if viable.is_empty() {
        return Err(BatchPartitionError::Infeasible {
            remaining_children: prepared.request.child_count,
            reasons,
        });
    }

    let homogeneous = viable.windows(2).all(|pair| {
        singleton_profile_key(prepared, pair[0]) == singleton_profile_key(prepared, pair[1])
    });
    if homogeneous {
        let per_device_host = viable[0].activation_host_bytes;
        let host_limit = prepared
            .request
            .host_headroom_bytes
            .checked_div(per_device_host)
            .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX))
            .unwrap_or(viable.len());
        let cardinality = viable
            .len()
            .min(prepared.request.child_count as usize)
            .min(host_limit);
        if cardinality == 0 {
            return Err(BatchPartitionError::Infeasible {
                remaining_children: prepared.request.child_count,
                reasons: BTreeSet::from([BatchInfeasibilityReason::InsufficientHostRam]),
            });
        }
        return singleton_plan_for_devices(
            prepared,
            &viable[..cardinality],
            BatchOptimization::ExactHomogeneousSingleton,
        );
    }

    if viable.len() <= EXACT_SMALL_DEVICE_LIMIT {
        return exact_small_singleton_plan(prepared, &viable);
    }

    // Cardinality is still exact for singleton activation: all devices have
    // equal value (one initial-wave lane), so taking activation costs in
    // ascending order proves the largest feasible count. Which equal-count
    // subset best optimizes later timing is the bounded, non-optimal part.
    viable.sort_by(|left, right| {
        left.activation_host_bytes
            .cmp(&right.activation_host_bytes)
            .then_with(|| left.first_finish_ms.cmp(&right.first_finish_ms))
            .then_with(|| {
                prepared.devices[left.device_index]
                    .profile
                    .device_id
                    .cmp(&prepared.devices[right.device_index].profile.device_id)
            })
    });
    let mut selected = Vec::new();
    let mut host = 0u64;
    for candidate in viable {
        if selected.len() >= prepared.request.child_count as usize {
            break;
        }
        let next = host
            .checked_add(candidate.activation_host_bytes)
            .ok_or(BatchPartitionError::HostMemoryOverflow)?;
        if next <= prepared.request.host_headroom_bytes {
            selected.push(candidate);
            host = next;
        }
    }
    if selected.is_empty() {
        return Err(BatchPartitionError::Infeasible {
            remaining_children: prepared.request.child_count,
            reasons: BTreeSet::from([BatchInfeasibilityReason::InsufficientHostRam]),
        });
    }
    selected.sort_by_key(|candidate| candidate.device_index);
    singleton_plan_for_devices(prepared, &selected, BatchOptimization::BoundedHeuristic)
}

fn singleton_profile_key(
    prepared: &PreparedRequest<'_>,
    candidate: SingletonDevice,
) -> (u64, bool, u64, u64, u64, BatchSizeEstimate) {
    let profile = prepared.devices[candidate.device_index].profile;
    (
        profile.available_at_ms,
        profile.initially_warm,
        profile.cold_setup_ms,
        profile.warm_setup_ms,
        candidate.activation_host_bytes,
        candidate.estimate,
    )
}

fn exact_small_singleton_plan(
    prepared: &PreparedRequest<'_>,
    viable: &[SingletonDevice],
) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
    let mut best: Option<(Vec<SingletonDevice>, SingletonAllocation)> = None;
    let mut states_evaluated = 0usize;
    let subset_count = 1usize << viable.len();
    for mask in 1..subset_count {
        let cardinality = mask.count_ones() as usize;
        if cardinality > prepared.request.child_count as usize {
            continue;
        }
        let mut host = 0u64;
        let mut selected = Vec::with_capacity(cardinality);
        for (index, candidate) in viable.iter().copied().enumerate() {
            if mask & (1usize << index) == 0 {
                continue;
            }
            host = host
                .checked_add(candidate.activation_host_bytes)
                .ok_or(BatchPartitionError::HostMemoryOverflow)?;
            selected.push(candidate);
        }
        if host > prepared.request.host_headroom_bytes {
            continue;
        }
        states_evaluated += 1;
        let allocation = singleton_allocation(prepared, &selected)?;
        if best.as_ref().is_none_or(|(current_devices, current)| {
            singleton_allocation_cmp(prepared, &selected, &allocation, current_devices, current)
                .is_lt()
        }) {
            best = Some((selected, allocation));
        }
    }
    let (selected, allocation) = best.ok_or(BatchPartitionError::Infeasible {
        remaining_children: prepared.request.child_count,
        reasons: BTreeSet::from([BatchInfeasibilityReason::InsufficientHostRam]),
    })?;
    singleton_plan_for_allocation(
        prepared,
        &selected,
        &allocation,
        BatchOptimization::ExactSmall,
        states_evaluated,
    )
}

fn singleton_plan_for_devices(
    prepared: &PreparedRequest<'_>,
    selected: &[SingletonDevice],
    optimization: BatchOptimization,
) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
    let allocation = singleton_allocation(prepared, selected)?;
    singleton_plan_for_allocation(prepared, selected, &allocation, optimization, 1)
}

fn singleton_allocation(
    prepared: &PreparedRequest<'_>,
    selected: &[SingletonDevice],
) -> Result<SingletonAllocation, BatchPartitionError> {
    let child_count = prepared.request.child_count;
    let mandatory =
        u32::try_from(selected.len()).map_err(|_| BatchPartitionError::PartitionCountOverflow)?;
    let additional = child_count
        .checked_sub(mandatory)
        .ok_or(BatchPartitionError::PartitionCountOverflow)?;
    let streams = selected
        .iter()
        .map(|candidate| {
            let profile = prepared.devices[candidate.device_index].profile;
            let first_setup = if profile.initially_warm {
                profile.warm_setup_ms
            } else {
                profile.cold_setup_ms
            };
            let first = u128::from(profile.available_at_ms.max(prepared.request.now_ms))
                .checked_add(u128::from(first_setup))
                .and_then(|value| {
                    value.checked_add(u128::from(candidate.estimate.predicted_run_ms))
                })
                .ok_or(BatchPartitionError::TimingOverflow)?;
            let period = u128::from(profile.warm_setup_ms)
                .checked_add(u128::from(candidate.estimate.predicted_run_ms))
                .ok_or(BatchPartitionError::TimingOverflow)?;
            Ok((first, period, first_setup, profile.warm_setup_ms))
        })
        .collect::<Result<Vec<_>, BatchPartitionError>>()?;
    let mut counts = vec![1u32; selected.len()];

    if additional > 0 {
        let mut low = 0u128;
        let mut high = streams
            .iter()
            .map(|&(first, period, _, _)| {
                u128::from(additional)
                    .checked_mul(period)
                    .and_then(|tail| first.checked_add(tail))
                    .ok_or(BatchPartitionError::TimingOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .min()
            .unwrap_or(0);
        while low < high {
            let middle = low
                .checked_add((high - low) / 2)
                .ok_or(BatchPartitionError::TimingOverflow)?;
            let mut covered = 0u64;
            for stream in streams.iter().copied() {
                covered = covered
                    .checked_add(u64::from(singleton_additional_count_at(
                        stream, middle, additional,
                    )?))
                    .ok_or(BatchPartitionError::PartitionCountOverflow)?;
            }
            if covered >= u64::from(additional) {
                high = middle;
            } else {
                low = middle
                    .checked_add(1)
                    .ok_or(BatchPartitionError::TimingOverflow)?;
            }
        }
        let threshold = low;
        let before_limit = threshold.checked_sub(1);
        let mut assigned = 0u32;
        for (index, stream) in streams.iter().copied().enumerate() {
            let before = match before_limit {
                Some(limit) => singleton_additional_count_at(stream, limit, additional)?,
                None => 0,
            }
            .min(additional - assigned);
            counts[index] = counts[index]
                .checked_add(before)
                .ok_or(BatchPartitionError::PartitionCountOverflow)?;
            assigned = assigned
                .checked_add(before)
                .ok_or(BatchPartitionError::PartitionCountOverflow)?;
            if assigned == additional {
                break;
            }
        }
        let mut boundary_ties = streams
            .iter()
            .copied()
            .enumerate()
            .map(|(index, stream)| {
                let through = singleton_additional_count_at(stream, threshold, additional)?;
                let before = match before_limit {
                    Some(limit) => singleton_additional_count_at(stream, limit, additional)?,
                    None => 0,
                };
                Ok((
                    index,
                    through
                        .checked_sub(before)
                        .ok_or(BatchPartitionError::PartitionCountOverflow)?,
                    stream.3,
                    &prepared.devices[selected[index].device_index]
                        .profile
                        .device_id,
                ))
            })
            .collect::<Result<Vec<_>, BatchPartitionError>>()?;
        boundary_ties.sort_by(|left, right| left.2.cmp(&right.2).then_with(|| right.3.cmp(left.3)));
        for (index, ties, _, _) in boundary_ties {
            if assigned == additional {
                break;
            }
            let take = ties.min(additional - assigned);
            counts[index] = counts[index]
                .checked_add(take)
                .ok_or(BatchPartitionError::PartitionCountOverflow)?;
            assigned = assigned
                .checked_add(take)
                .ok_or(BatchPartitionError::PartitionCountOverflow)?;
        }
        debug_assert_eq!(assigned, additional);
    }

    let mut makespan_ms = 0u128;
    let mut sum_completion_ms = 0u128;
    let mut setup_ms = 0u128;
    for ((first, period, first_setup, warm_setup), &count) in streams.iter().copied().zip(&counts) {
        let tail = u128::from(count - 1);
        let finish = tail
            .checked_mul(period)
            .and_then(|tail_time| first.checked_add(tail_time))
            .ok_or(BatchPartitionError::TimingOverflow)?;
        makespan_ms = makespan_ms.max(finish);
        let first_completion_total = u128::from(count)
            .checked_mul(first)
            .ok_or(BatchPartitionError::TimingOverflow)?;
        let tail_completion_total = period
            .checked_mul(tail)
            .and_then(|value| value.checked_mul(u128::from(count)))
            .map(|value| value / 2)
            .ok_or(BatchPartitionError::TimingOverflow)?;
        sum_completion_ms = sum_completion_ms
            .checked_add(
                first_completion_total
                    .checked_add(tail_completion_total)
                    .ok_or(BatchPartitionError::TimingOverflow)?,
            )
            .ok_or(BatchPartitionError::TimingOverflow)?;
        let repeated_setup = tail
            .checked_mul(u128::from(warm_setup))
            .ok_or(BatchPartitionError::TimingOverflow)?;
        setup_ms = setup_ms
            .checked_add(
                u128::from(first_setup)
                    .checked_add(repeated_setup)
                    .ok_or(BatchPartitionError::TimingOverflow)?,
            )
            .ok_or(BatchPartitionError::TimingOverflow)?;
    }
    let now_total = u128::from(prepared.request.now_ms)
        .checked_mul(u128::from(child_count))
        .ok_or(BatchPartitionError::TimingOverflow)?;
    Ok(SingletonAllocation {
        counts,
        makespan_ms: makespan_ms
            .checked_sub(u128::from(prepared.request.now_ms))
            .ok_or(BatchPartitionError::TimingOverflow)?,
        sum_completion_ms: sum_completion_ms
            .checked_sub(now_total)
            .ok_or(BatchPartitionError::TimingOverflow)?,
        setup_ms,
    })
}

fn singleton_additional_count_at(
    stream: (u128, u128, u64, u64),
    limit: u128,
    additional: u32,
) -> Result<u32, BatchPartitionError> {
    let (first, period, _, _) = stream;
    if period == 0 {
        return Ok(if limit >= first { additional } else { 0 });
    }
    let second = first
        .checked_add(period)
        .ok_or(BatchPartitionError::TimingOverflow)?;
    if limit < second {
        Ok(0)
    } else {
        let through = (limit - second) / period;
        let bounded = through.min(u128::from(additional - 1));
        u32::try_from(
            1u128
                .checked_add(bounded)
                .ok_or(BatchPartitionError::TimingOverflow)?,
        )
        .map_err(|_| BatchPartitionError::PartitionCountOverflow)
    }
}

fn singleton_allocation_cmp(
    prepared: &PreparedRequest<'_>,
    left_devices: &[SingletonDevice],
    left: &SingletonAllocation,
    right_devices: &[SingletonDevice],
    right: &SingletonAllocation,
) -> Ordering {
    right_devices
        .len()
        .cmp(&left_devices.len())
        .then_with(|| left.makespan_ms.cmp(&right.makespan_ms))
        .then_with(|| left.sum_completion_ms.cmp(&right.sum_completion_ms))
        .then_with(|| left.setup_ms.cmp(&right.setup_ms))
        .then_with(|| {
            left_devices
                .iter()
                .zip(&left.counts)
                .map(|(candidate, count)| {
                    (
                        &prepared.devices[candidate.device_index].profile.device_id,
                        count,
                    )
                })
                .cmp(
                    right_devices
                        .iter()
                        .zip(&right.counts)
                        .map(|(candidate, count)| {
                            (
                                &prepared.devices[candidate.device_index].profile.device_id,
                                count,
                            )
                        }),
                )
        })
}

fn singleton_plan_for_allocation(
    prepared: &PreparedRequest<'_>,
    selected: &[SingletonDevice],
    allocation: &SingletonAllocation,
    optimization: BatchOptimization,
    optimizer_states_evaluated: usize,
) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
    let request = prepared.request;
    let lanes = selected
        .iter()
        .zip(&allocation.counts)
        .map(|(candidate, &count)| {
            let profile = prepared.devices[candidate.device_index].profile;
            let (first_setup_disposition, first_setup_ms) = if profile.initially_warm {
                (BatchSetupDisposition::Warm, profile.warm_setup_ms)
            } else {
                (BatchSetupDisposition::Cold, profile.cold_setup_ms)
            };
            let first_start_ms = profile.available_at_ms.max(request.now_ms);
            let first_finish_ms = first_start_ms
                .checked_add(first_setup_ms)
                .and_then(|time| time.checked_add(candidate.estimate.predicted_run_ms))
                .ok_or(BatchPartitionError::TimingOverflow)?;
            let period_ms = profile
                .warm_setup_ms
                .checked_add(candidate.estimate.predicted_run_ms)
                .ok_or(BatchPartitionError::TimingOverflow)?;
            let tail = u64::from(count - 1);
            first_finish_ms
                .checked_add(
                    tail.checked_mul(period_ms)
                        .ok_or(BatchPartitionError::TimingOverflow)?,
                )
                .ok_or(BatchPartitionError::TimingOverflow)?;
            Ok(CompactSingletonLane {
                device_id: profile.device_id.clone(),
                count,
                first_start_ms,
                first_finish_ms,
                period_ms,
                first_setup_disposition,
                predicted_vram_bytes: candidate.estimate.predicted_vram_bytes,
                predicted_host_ram_bytes: candidate.estimate.predicted_host_ram_bytes,
            })
        })
        .collect::<Result<Vec<_>, BatchPartitionError>>()?;
    let predicted_peak_host_ram_bytes = selected.iter().try_fold(0u64, |total, candidate| {
        total
            .checked_add(candidate.activation_host_bytes)
            .ok_or(BatchPartitionError::HostMemoryOverflow)
    })?;
    let predicted_parent_makespan_ms =
        u64::try_from(allocation.makespan_ms).map_err(|_| BatchPartitionError::TimingOverflow)?;
    Ok(AdaptiveBatchPlan {
        child_count: request.child_count,
        partitions: AdaptiveBatchPartitions::compact_singleton(request.child_count, lanes),
        predicted_parent_makespan_ms,
        predicted_sum_completion_ms: allocation.sum_completion_ms,
        predicted_setup_ms: allocation.setup_ms,
        predicted_peak_host_ram_bytes,
        devices_used: selected.len(),
        optimization,
        optimizer_states_evaluated,
        materialization_passes: 0,
    })
}

fn plan_with_size_cap(
    prepared: &PreparedRequest<'_>,
    size_cap: u32,
) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
    let request = prepared.request;
    let mut schedules = prepared
        .devices
        .iter()
        .map(|device| DeviceSchedule {
            next_available_ms: device.profile.available_at_ms.max(request.now_ms),
            used: false,
            peak_partition_host_ram_bytes: 0,
        })
        .collect::<Vec<_>>();
    let mut partitions = Vec::new();
    let mut covered = 0u32;
    let mut host_reserved = 0u64;
    let mut predicted_setup_ms = 0u128;

    while covered < request.child_count {
        let remaining = request.child_count - covered;
        let mut best: Option<Candidate<'_>> = None;
        let mut reasons = BTreeSet::new();

        for (device_index, device) in prepared.devices.iter().enumerate() {
            let schedule = &schedules[device_index];
            let mut device_candidate = None;
            for &size in prepared.native_sizes.as_slice().iter().rev() {
                if size > size_cap || size > remaining {
                    continue;
                }
                if size > device.profile.partition_capacity {
                    reasons.insert(BatchInfeasibilityReason::DeviceCapacity);
                    continue;
                }
                let Some(estimate) = device.estimates.get(&size) else {
                    reasons.insert(BatchInfeasibilityReason::MissingSizeEstimate);
                    continue;
                };
                if estimate.predicted_vram_bytes > device.profile.available_vram_bytes {
                    reasons.insert(BatchInfeasibilityReason::InsufficientVram);
                    continue;
                }
                let host_delta_bytes = if schedule.used {
                    estimate
                        .predicted_host_ram_bytes
                        .saturating_sub(schedule.peak_partition_host_ram_bytes)
                } else {
                    device
                        .profile
                        .setup_host_ram_bytes
                        .checked_add(estimate.predicted_host_ram_bytes)
                        .ok_or(BatchPartitionError::HostMemoryOverflow)?
                };
                if host_reserved
                    .checked_add(host_delta_bytes)
                    .is_none_or(|total| total > request.host_headroom_bytes)
                {
                    reasons.insert(BatchInfeasibilityReason::InsufficientHostRam);
                    continue;
                }
                let (setup_disposition, setup_ms) =
                    if !schedule.used && !device.profile.initially_warm {
                        (BatchSetupDisposition::Cold, device.profile.cold_setup_ms)
                    } else {
                        (BatchSetupDisposition::Warm, device.profile.warm_setup_ms)
                    };
                let finish_ms = schedule
                    .next_available_ms
                    .checked_add(setup_ms)
                    .and_then(|value| value.checked_add(estimate.predicted_run_ms))
                    .ok_or(BatchPartitionError::TimingOverflow)?;
                device_candidate = Some(Candidate {
                    device_index,
                    estimate,
                    start_ms: schedule.next_available_ms,
                    finish_ms,
                    setup_ms,
                    setup_disposition,
                    host_delta_bytes,
                    activates_device: !schedule.used,
                });
                break;
            }

            if let Some(candidate) = device_candidate {
                if best
                    .as_ref()
                    .is_none_or(|current| candidate_cmp(&candidate, current, prepared).is_lt())
                {
                    best = Some(candidate);
                }
            }
        }

        let Some(chosen) = best else {
            return Err(BatchPartitionError::Infeasible {
                remaining_children: remaining,
                reasons,
            });
        };
        let device = &prepared.devices[chosen.device_index];
        let schedule = &mut schedules[chosen.device_index];
        host_reserved = host_reserved
            .checked_add(chosen.host_delta_bytes)
            .ok_or(BatchPartitionError::HostMemoryOverflow)?;
        schedule.peak_partition_host_ram_bytes = schedule
            .peak_partition_host_ram_bytes
            .max(chosen.estimate.predicted_host_ram_bytes);
        schedule.next_available_ms = chosen.finish_ms;
        schedule.used = true;
        predicted_setup_ms += u128::from(chosen.setup_ms);
        partitions.push(AdaptiveBatchPartition {
            partition_index: 0,
            partition_count: 0,
            child_start: covered,
            size: chosen.estimate.size,
            device_id: device.profile.device_id.clone(),
            estimated_start_ms: chosen.start_ms,
            estimated_finish_ms: chosen.finish_ms,
            setup_disposition: chosen.setup_disposition,
            predicted_vram_bytes: chosen.estimate.predicted_vram_bytes,
            predicted_host_ram_bytes: chosen.estimate.predicted_host_ram_bytes,
        });
        covered = covered
            .checked_add(chosen.estimate.size)
            .ok_or(BatchPartitionError::PartitionCountOverflow)?;
    }

    finalize_plan(
        request,
        partitions,
        PlanSummaryInput {
            predicted_setup_ms,
            predicted_peak_host_ram_bytes: host_reserved,
            devices_used: schedules.iter().filter(|schedule| schedule.used).count(),
            optimization: BatchOptimization::BoundedHeuristic,
            optimizer_states_evaluated: 1,
            materialization_passes: 1,
        },
    )
}

fn candidate_cmp(
    left: &Candidate<'_>,
    right: &Candidate<'_>,
    prepared: &PreparedRequest<'_>,
) -> Ordering {
    right
        .activates_device
        .cmp(&left.activates_device)
        .then_with(|| {
            left.finish_ms
                .cmp(&right.finish_ms)
                .then_with(|| left.start_ms.cmp(&right.start_ms))
                .then_with(|| left.setup_ms.cmp(&right.setup_ms))
                .then_with(|| right.estimate.size.cmp(&left.estimate.size))
                .then_with(|| {
                    prepared.devices[left.device_index]
                        .profile
                        .device_id
                        .cmp(&prepared.devices[right.device_index].profile.device_id)
                })
        })
}

fn plan_cmp(left: &AdaptiveBatchPlan, right: &AdaptiveBatchPlan) -> Ordering {
    right.devices_used.cmp(&left.devices_used).then_with(|| {
        left.predicted_parent_makespan_ms
            .cmp(&right.predicted_parent_makespan_ms)
            .then_with(|| {
                left.predicted_sum_completion_ms
                    .cmp(&right.predicted_sum_completion_ms)
            })
            .then_with(|| left.predicted_setup_ms.cmp(&right.predicted_setup_ms))
            .then_with(|| left.partitions.len().cmp(&right.partitions.len()))
            .then_with(|| {
                left.partitions
                    .iter()
                    .map(|partition| (partition.device_id, partition.size))
                    .cmp(
                        right
                            .partitions
                            .iter()
                            .map(|partition| (partition.device_id, partition.size)),
                    )
            })
    })
}

struct PlanSummaryInput {
    predicted_setup_ms: u128,
    predicted_peak_host_ram_bytes: u64,
    devices_used: usize,
    optimization: BatchOptimization,
    optimizer_states_evaluated: usize,
    materialization_passes: u8,
}

fn finalize_plan(
    request: &BatchPartitionRequest,
    mut partitions: Vec<AdaptiveBatchPartition>,
    summary: PlanSummaryInput,
) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
    let partition_count =
        u32::try_from(partitions.len()).map_err(|_| BatchPartitionError::PartitionCountOverflow)?;
    for (index, partition) in partitions.iter_mut().enumerate() {
        partition.partition_index = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or(BatchPartitionError::PartitionCountOverflow)?;
        partition.partition_count = partition_count;
    }
    let latest_finish_ms = partitions
        .iter()
        .map(|partition| partition.estimated_finish_ms)
        .max()
        .unwrap_or(request.now_ms);
    let predicted_sum_completion_ms = partitions.iter().try_fold(0u128, |total, partition| {
        let relative_finish = partition
            .estimated_finish_ms
            .checked_sub(request.now_ms)
            .ok_or(BatchPartitionError::TimingOverflow)?;
        total
            .checked_add(u128::from(relative_finish) * u128::from(partition.size))
            .ok_or(BatchPartitionError::TimingOverflow)
    })?;
    Ok(AdaptiveBatchPlan {
        child_count: request.child_count,
        partitions: AdaptiveBatchPartitions::materialized(partitions),
        predicted_parent_makespan_ms: latest_finish_ms
            .checked_sub(request.now_ms)
            .ok_or(BatchPartitionError::TimingOverflow)?,
        predicted_sum_completion_ms,
        predicted_setup_ms: summary.predicted_setup_ms,
        predicted_peak_host_ram_bytes: summary.predicted_peak_host_ram_bytes,
        devices_used: summary.devices_used,
        optimization: summary.optimization,
        optimizer_states_evaluated: summary.optimizer_states_evaluated,
        materialization_passes: summary.materialization_passes,
    })
}
