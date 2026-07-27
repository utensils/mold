use std::collections::BTreeSet;

use mold_scheduler::{
    BatchDeviceProfile, BatchInfeasibilityReason, BatchPartitionError, BatchPartitionPlanner,
    BatchPartitionRequest, BatchSetupDisposition, BatchSizeEstimate, NativeBatchSizes,
};

fn estimate(size: u32, run_ms: u64, vram_bytes: u64, host_ram_bytes: u64) -> BatchSizeEstimate {
    BatchSizeEstimate {
        size,
        predicted_run_ms: run_ms,
        predicted_vram_bytes: vram_bytes,
        predicted_host_ram_bytes: host_ram_bytes,
    }
}

fn device(id: impl Into<String>, estimates: Vec<BatchSizeEstimate>) -> BatchDeviceProfile {
    BatchDeviceProfile {
        device_id: id.into().into(),
        available_at_ms: 0,
        initially_warm: false,
        partition_capacity: u32::MAX,
        available_vram_bytes: u64::MAX,
        cold_setup_ms: 10,
        warm_setup_ms: 1,
        setup_host_ram_bytes: 0,
        size_estimates: estimates,
    }
}

fn request(
    child_count: u32,
    native_batch_sizes: Vec<u32>,
    devices: Vec<BatchDeviceProfile>,
) -> BatchPartitionRequest {
    BatchPartitionRequest {
        child_count,
        now_ms: 0,
        native_batch_sizes,
        host_headroom_bytes: u64::MAX,
        devices,
    }
}

fn assert_exact_coverage(plan: &mold_scheduler::AdaptiveBatchPlan) {
    let mut expected_start = 0;
    for (index, partition) in plan.partitions.iter().enumerate() {
        assert_eq!(partition.partition_index, index as u32 + 1);
        assert_eq!(partition.partition_count, plan.partitions.len() as u32);
        assert_eq!(partition.child_start, expected_start);
        expected_start += partition.size;
        assert_eq!(
            partition.scheduler_projection(),
            mold_scheduler::PlannedBatchPartition {
                index: index as u32 + 1,
                count: plan.partitions.len() as u32,
                size: partition.size,
            }
        );
    }
    assert_eq!(expected_start, plan.child_count);
}

#[test]
fn native_sizes_are_canonicalized_without_power_of_two_assumptions() {
    assert_eq!(
        NativeBatchSizes::canonicalize([5, 3, 1, 2, 3])
            .unwrap()
            .as_slice(),
        &[1, 2, 3, 5]
    );
    assert!(matches!(
        NativeBatchSizes::canonicalize([]),
        Err(BatchPartitionError::EmptyNativeBatchSizes)
    ));
    assert!(matches!(
        NativeBatchSizes::canonicalize([0, 1]),
        Err(BatchPartitionError::ZeroNativeBatchSize)
    ));
    assert!(matches!(
        NativeBatchSizes::canonicalize([2, 3, 5]),
        Err(BatchPartitionError::MissingSingletonCapability)
    ));
}

#[test]
fn singleton_capability_distributes_children_across_1_2_8_and_64_devices() {
    for device_count in [1usize, 2, 8, 64] {
        let devices = (0..device_count)
            .map(|index| device(format!("cuda:{index:032x}"), vec![estimate(1, 100, 8, 4)]))
            .collect::<Vec<_>>();
        let plan =
            BatchPartitionPlanner::plan(&request((device_count * 2) as u32, vec![1], devices))
                .unwrap();

        assert_exact_coverage(&plan);
        assert!(plan.partitions.iter().all(|partition| partition.size == 1));
        assert_eq!(
            plan.partitions
                .iter()
                .map(|partition| partition.device_id.clone())
                .collect::<BTreeSet<_>>()
                .len(),
            device_count
        );
    }
}

#[test]
fn homogeneous_singleton_fleets_reach_the_exact_balanced_wave_optimum() {
    for device_count in [1usize, 2, 8, 64] {
        let child_count = device_count * 2 + 3;
        let devices = (0..device_count)
            .map(|index| device(format!("cuda:{index:032x}"), vec![estimate(1, 100, 8, 4)]))
            .collect::<Vec<_>>();
        let plan =
            BatchPartitionPlanner::plan(&request(child_count as u32, vec![1], devices)).unwrap();
        let waves = child_count.div_ceil(device_count);
        let exact_makespan_ms = 110 + (waves as u64 - 1) * 101;
        let mut per_device = plan
            .partitions
            .iter()
            .fold(
                std::collections::BTreeMap::new(),
                |mut counts, partition| {
                    *counts.entry(partition.device_id.clone()).or_insert(0usize) += 1;
                    counts
                },
            )
            .into_values()
            .collect::<Vec<_>>();
        per_device.sort_unstable();
        let first_wave_devices = plan
            .partitions
            .iter()
            .filter(|partition| partition.estimated_start_ms == 0)
            .map(|partition| partition.device_id.clone())
            .collect::<BTreeSet<_>>();

        assert_exact_coverage(&plan);
        assert_eq!(plan.predicted_parent_makespan_ms, exact_makespan_ms);
        assert_eq!(per_device.len(), device_count);
        assert!(per_device.last().unwrap() - per_device.first().unwrap() <= 1);
        assert_eq!(first_wave_devices.len(), device_count.min(child_count));
        if child_count >= device_count {
            assert!(per_device.iter().all(|count| *count > 0));
        }
    }
}

#[test]
fn asymmetric_singleton_schedule_matches_a_two_device_brute_force_oracle() {
    let child_count = 9u32;
    let mut fast_after_load = device("cuda:a", vec![estimate(1, 37, 1, 1)]);
    fast_after_load.cold_setup_ms = 80;
    fast_after_load.warm_setup_ms = 3;
    let mut slow_but_warm = device("cuda:b", vec![estimate(1, 61, 1, 1)]);
    slow_but_warm.initially_warm = true;
    slow_but_warm.warm_setup_ms = 2;

    let plan = BatchPartitionPlanner::plan(&request(
        child_count,
        vec![1],
        vec![fast_after_load, slow_but_warm],
    ))
    .unwrap();

    let oracle = (0..=child_count)
        .map(|on_a| {
            let on_b = child_count - on_a;
            let completions_a = (0..on_a)
                .map(|index| 80 + 37 + u64::from(index) * (3 + 37))
                .collect::<Vec<_>>();
            let completions_b = (0..on_b)
                .map(|index| 2 + 61 + u64::from(index) * (2 + 61))
                .collect::<Vec<_>>();
            let makespan = completions_a
                .iter()
                .chain(&completions_b)
                .copied()
                .max()
                .unwrap_or(0);
            let sum = completions_a
                .iter()
                .chain(&completions_b)
                .map(|completion| u128::from(*completion))
                .sum::<u128>();
            (makespan, sum, on_a)
        })
        .min()
        .unwrap();

    let planned_on_a = plan
        .partitions
        .iter()
        .filter(|partition| partition.device_id.as_str() == "cuda:a")
        .count() as u32;
    assert_eq!(
        (
            plan.predicted_parent_makespan_ms,
            plan.predicted_sum_completion_ms,
            planned_on_a
        ),
        oracle
    );
}

#[test]
fn synthetic_non_power_of_two_capability_selects_measured_native_partitions() {
    let estimates = vec![
        estimate(1, 100, 10, 10),
        estimate(2, 120, 12, 12),
        estimate(3, 130, 14, 14),
        estimate(5, 150, 18, 18),
    ];
    let plan = BatchPartitionPlanner::plan(&request(
        8,
        vec![5, 1, 3, 2],
        vec![device("cuda:a", estimates)],
    ))
    .unwrap();

    assert_exact_coverage(&plan);
    assert_eq!(
        plan.partitions
            .iter()
            .map(|partition| partition.size)
            .collect::<Vec<_>>(),
        vec![5, 3]
    );
    assert_eq!(plan.predicted_parent_makespan_ms, 291);
    assert_eq!(plan.predicted_sum_completion_ms, 1_673);
}

#[test]
fn capacity_and_vram_filter_native_sizes_without_losing_exact_coverage() {
    let mut constrained = device(
        "cuda:a",
        vec![
            estimate(1, 100, 10, 1),
            estimate(2, 120, 20, 1),
            estimate(3, 130, 30, 1),
            estimate(5, 140, 50, 1),
        ],
    );
    constrained.partition_capacity = 3;
    constrained.available_vram_bytes = 25;

    let plan =
        BatchPartitionPlanner::plan(&request(7, vec![1, 2, 3, 5], vec![constrained])).unwrap();

    assert_exact_coverage(&plan);
    assert!(plan.partitions.iter().all(|partition| partition.size <= 2));
    assert!(plan
        .partitions
        .iter()
        .all(|partition| partition.predicted_vram_bytes <= 25));
}

#[test]
fn aggregate_host_headroom_accounts_for_setup_duplication() {
    let mut first = device("cuda:a", vec![estimate(1, 100, 1, 20)]);
    first.setup_host_ram_bytes = 80;
    let mut second = device("cuda:b", vec![estimate(1, 100, 1, 20)]);
    second.setup_host_ram_bytes = 80;
    let mut constrained = request(4, vec![1], vec![first, second]);
    constrained.host_headroom_bytes = 100;

    let plan = BatchPartitionPlanner::plan(&constrained).unwrap();

    assert_exact_coverage(&plan);
    assert_eq!(plan.devices_used, 1);
    assert_eq!(plan.predicted_peak_host_ram_bytes, 100);
}

#[test]
fn warm_and_cold_setup_are_compared_before_selecting_a_device() {
    let mut warm = device("cuda:warm", vec![estimate(1, 20, 1, 1)]);
    warm.initially_warm = true;
    warm.warm_setup_ms = 1;
    warm.cold_setup_ms = 1_000;

    let mut cold = device("cuda:cold", vec![estimate(1, 1, 1, 1)]);
    cold.cold_setup_ms = 100;

    let plan = BatchPartitionPlanner::plan(&request(1, vec![1], vec![cold, warm])).unwrap();

    assert_eq!(plan.partitions[0].device_id.as_str(), "cuda:warm");
    assert_eq!(
        plan.partitions[0].setup_disposition,
        BatchSetupDisposition::Warm
    );
}

#[test]
fn deterministic_ties_use_device_id_and_ignore_input_order() {
    let left = device("cuda:a", vec![estimate(1, 10, 1, 1)]);
    let right = device("cuda:b", vec![estimate(1, 10, 1, 1)]);

    let forward =
        BatchPartitionPlanner::plan(&request(5, vec![1], vec![left.clone(), right.clone()]))
            .unwrap();
    let reverse = BatchPartitionPlanner::plan(&request(5, vec![1, 1], vec![right, left])).unwrap();

    assert_eq!(forward, reverse);
    assert_eq!(forward.partitions[0].device_id.as_str(), "cuda:a");
}

#[test]
fn large_parent_planning_is_bounded_by_derived_strategies_not_combinatorial() {
    let estimates = vec![
        estimate(1, 100, 1, 1),
        estimate(3, 180, 2, 2),
        estimate(5, 220, 3, 3),
    ];
    let devices = (0..64)
        .map(|index| device(format!("cuda:{index:032x}"), estimates.clone()))
        .collect();

    let plan = BatchPartitionPlanner::plan(&request(100_003, vec![1, 3, 5], devices)).unwrap();

    assert_exact_coverage(&plan);
    assert!(plan.partitions.len() <= 100_003);
    assert_eq!(plan.devices_used, 64);
}

#[test]
fn infeasibility_reports_typed_vram_and_host_reasons() {
    let mut no_vram = device("cuda:a", vec![estimate(1, 1, 10, 1)]);
    no_vram.available_vram_bytes = 9;
    let error = BatchPartitionPlanner::plan(&request(1, vec![1], vec![no_vram])).unwrap_err();
    assert!(matches!(
        error,
        BatchPartitionError::Infeasible {
            reasons,
            ..
        } if reasons.contains(&BatchInfeasibilityReason::InsufficientVram)
    ));

    let mut no_host = request(
        1,
        vec![1],
        vec![device("cuda:a", vec![estimate(1, 1, 1, 10)])],
    );
    no_host.host_headroom_bytes = 9;
    let error = BatchPartitionPlanner::plan(&no_host).unwrap_err();
    assert!(matches!(
        error,
        BatchPartitionError::Infeasible {
            reasons,
            ..
        } if reasons.contains(&BatchInfeasibilityReason::InsufficientHostRam)
    ));
}
