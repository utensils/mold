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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveBatchPlan {
    pub child_count: u32,
    pub partitions: Vec<AdaptiveBatchPartition>,
    pub predicted_parent_makespan_ms: u64,
    pub predicted_sum_completion_ms: u128,
    pub predicted_setup_ms: u128,
    pub predicted_peak_host_ram_bytes: u64,
    pub devices_used: usize,
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
    PartitionCountOverflow,
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
            Self::PartitionCountOverflow => {
                formatter.write_str("batch partition count exceeds the u32 projection")
            }
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
/// independently. Native heterogeneous batching uses a deterministic bounded
/// strategy comparison; it deliberately does not claim general global
/// optimality for the NP-hard unrelated-machine batching problem. For `N`
/// children, `D` devices, and `S` declared sizes, work is bounded by
/// `O(N * D * S^2)` and retained memory by `O(N + D * S)`; singleton
/// production capabilities reduce the time bound to `O(N * D)`.
pub struct BatchPartitionPlanner;

impl BatchPartitionPlanner {
    pub fn plan(request: &BatchPartitionRequest) -> Result<AdaptiveBatchPlan, BatchPartitionError> {
        let prepared = PreparedRequest::new(request)?;
        let mut best: Option<AdaptiveBatchPlan> = None;
        let mut infeasible_reasons = BTreeSet::new();
        let mut minimum_remaining = request.child_count;
        let strategy_caps = prepared
            .native_sizes
            .as_slice()
            .iter()
            .map(|size| (*size).min(request.child_count))
            .collect::<BTreeSet<_>>();

        for cap in strategy_caps {
            match plan_with_size_cap(&prepared, cap) {
                Ok(candidate) => {
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

        best.ok_or(BatchPartitionError::Infeasible {
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
                        .ok_or(BatchPartitionError::TimingOverflow)?
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
            .ok_or(BatchPartitionError::TimingOverflow)?;
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
        partitions,
        predicted_parent_makespan_ms: latest_finish_ms
            .checked_sub(request.now_ms)
            .ok_or(BatchPartitionError::TimingOverflow)?,
        predicted_sum_completion_ms,
        predicted_setup_ms,
        predicted_peak_host_ram_bytes: host_reserved,
        devices_used: schedules.iter().filter(|schedule| schedule.used).count(),
    })
}

fn candidate_cmp(
    left: &Candidate<'_>,
    right: &Candidate<'_>,
    prepared: &PreparedRequest<'_>,
) -> Ordering {
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
}

fn plan_cmp(left: &AdaptiveBatchPlan, right: &AdaptiveBatchPlan) -> Ordering {
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
                .map(|partition| (&partition.device_id, partition.size))
                .cmp(
                    right
                        .partitions
                        .iter()
                        .map(|partition| (&partition.device_id, partition.size)),
                )
        })
}
