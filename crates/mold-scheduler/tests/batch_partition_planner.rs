use std::collections::BTreeSet;

use mold_scheduler::{
    BatchDeviceProfile, BatchInfeasibilityReason, BatchOptimization, BatchPartitionError,
    BatchPartitionPlanner, BatchPartitionRequest, BatchPlanRepresentation, BatchSetupDisposition,
    BatchSizeEstimate, NativeBatchSizes, BOUNDED_HEURISTIC_STRATEGY_LIMIT,
    EXACT_SMALL_DEVICE_LIMIT, MAX_PARTITION_WINDOW, NATIVE_MATERIALIZATION_CHILD_LIMIT,
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
    assert_eq!(plan.representation(), BatchPlanRepresentation::Materialized);
    assert_eq!(plan.retained_partition_records(), 2);
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
fn aggregate_host_ram_is_part_of_device_activation_before_stream_merging() {
    let mut tempting = device("cuda:a", vec![estimate(1, 5, 1, 10)]);
    tempting.cold_setup_ms = 0;
    tempting.warm_setup_ms = 95;
    let mut optimal = device("cuda:b", vec![estimate(1, 6, 1, 5)]);
    optimal.cold_setup_ms = 0;
    optimal.warm_setup_ms = 0;
    let mut constrained = request(2, vec![1], vec![tempting, optimal]);
    constrained.host_headroom_bytes = 10;

    let plan = BatchPartitionPlanner::plan(&constrained).unwrap();

    assert_eq!(plan.optimization, BatchOptimization::ExactSmall);
    assert_eq!(plan.devices_used, 1);
    assert_eq!(plan.predicted_parent_makespan_ms, 12);
    assert!(plan
        .partitions
        .iter()
        .all(|partition| partition.device_id.as_str() == "cuda:b"));
}

#[test]
fn exact_small_two_device_host_coupled_plans_agree_with_independent_brute_force() {
    let cases = [
        // (children, headroom, A run/cold/warm/host, B run/cold/warm/host)
        (2, 10, (5, 0, 95, 10), (6, 0, 0, 5)),
        (5, 13, (7, 20, 2, 7), (11, 0, 1, 6)),
        (6, 9, (13, 3, 0, 4), (5, 30, 4, 5)),
        (7, 20, (9, 40, 3, 12), (15, 1, 2, 8)),
    ];
    for (children, headroom, a, b) in cases {
        let mut device_a = device("cuda:a", vec![estimate(1, a.0, 1, a.3)]);
        device_a.cold_setup_ms = a.1;
        device_a.warm_setup_ms = a.2;
        let mut device_b = device("cuda:b", vec![estimate(1, b.0, 1, b.3)]);
        device_b.cold_setup_ms = b.1;
        device_b.warm_setup_ms = b.2;
        let mut input = request(children, vec![1], vec![device_a, device_b]);
        input.host_headroom_bytes = headroom;

        let plan = BatchPartitionPlanner::plan(&input).unwrap();
        let oracle = (0..=children)
            .filter_map(|on_a| {
                let on_b = children - on_a;
                let host = u64::from(on_a > 0) * a.3 + u64::from(on_b > 0) * b.3;
                if host > headroom {
                    return None;
                }
                let completions = |count: u32, spec: (u64, u64, u64, u64)| {
                    (0..count)
                        .map(|index| spec.1 + spec.0 + u64::from(index) * (spec.2 + spec.0))
                        .collect::<Vec<_>>()
                };
                let a_completion = completions(on_a, a);
                let b_completion = completions(on_b, b);
                let cardinality = usize::from(on_a > 0) + usize::from(on_b > 0);
                let makespan = a_completion
                    .iter()
                    .chain(&b_completion)
                    .copied()
                    .max()
                    .unwrap_or(0);
                let sum = a_completion
                    .iter()
                    .chain(&b_completion)
                    .map(|completion| u128::from(*completion))
                    .sum::<u128>();
                let setup = u128::from(if on_a == 0 {
                    0
                } else {
                    a.1 + u64::from(on_a - 1) * a.2
                }) + u128::from(if on_b == 0 {
                    0
                } else {
                    b.1 + u64::from(on_b - 1) * b.2
                });
                Some((std::cmp::Reverse(cardinality), makespan, sum, setup, on_a))
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
                std::cmp::Reverse(plan.devices_used),
                plan.predicted_parent_makespan_ms,
                plan.predicted_sum_completion_ms,
                plan.predicted_setup_ms,
                planned_on_a,
            ),
            oracle,
            "case {children}/{headroom}/{a:?}/{b:?}"
        );
    }
}

#[test]
fn exact_small_boundary_tie_minimizes_setup_before_stable_assignment() {
    let mut high_setup = device("cuda:a", vec![estimate(1, 1, 1, 0)]);
    high_setup.cold_setup_ms = 9;
    high_setup.warm_setup_ms = 9;
    let mut low_setup = device("cuda:b", vec![estimate(1, 9, 1, 0)]);
    low_setup.cold_setup_ms = 1;
    low_setup.warm_setup_ms = 1;

    let plan =
        BatchPartitionPlanner::plan(&request(3, vec![1], vec![high_setup, low_setup])).unwrap();
    let devices = plan
        .partitions
        .iter()
        .map(|partition| partition.device_id.to_string())
        .collect::<Vec<_>>();

    assert_eq!(plan.optimization, BatchOptimization::ExactSmall);
    assert_eq!(plan.predicted_parent_makespan_ms, 20);
    assert_eq!(plan.predicted_sum_completion_ms, 40);
    assert_eq!(plan.predicted_setup_ms, 11);
    assert_eq!(devices, ["cuda:a", "cuda:b", "cuda:b"]);
}

#[test]
fn exact_small_zero_period_boundary_ties_apply_setup_then_stable_counts() {
    let profiles = [("cuda:a", 3u64), ("cuda:b", 1u64), ("cuda:c", 1u64)]
        .into_iter()
        .map(|(id, cold_setup)| {
            let mut profile = device(id, vec![estimate(1, 0, 1, 0)]);
            profile.cold_setup_ms = cold_setup;
            profile.warm_setup_ms = 0;
            profile
        })
        .collect();

    let plan = BatchPartitionPlanner::plan(&request(7, vec![1], profiles)).unwrap();
    let counts = ["cuda:a", "cuda:b", "cuda:c"].map(|id| {
        plan.partitions
            .iter()
            .filter(|partition| partition.device_id.as_str() == id)
            .count()
    });

    // All additional completions tie at the same threshold with equal zero
    // marginal setup. The stable assignment comparator minimizes the earlier
    // device counts first, so the highest stable ID receives the tie slots.
    assert_eq!(counts, [1, 1, 5]);
}

#[test]
fn exact_singleton_objective_matches_brute_force_across_edge_case_streams() {
    // available, initially_warm, cold, warm_setup, run, setup_host, partition_host
    let fleets = [
        vec![(0, false, 0, 0, 0, 0, 0), (0, true, 7, 0, 0, 0, 0)],
        vec![(5, false, 9, 2, 3, 4, 0), (0, true, 20, 5, 1, 0, 3)],
        vec![
            (11, true, 40, 0, 7, 2, 2),
            (3, false, 5, 4, 2, 0, 5),
            (0, false, 13, 1, 5, 3, 1),
        ],
    ];
    for specs in fleets {
        for children in 1..=7u32 {
            for headroom in [0, 4, 7, 12, u64::MAX] {
                let devices = specs
                    .iter()
                    .enumerate()
                    .map(
                        |(index, &(available, warm, cold, warm_setup, run, setup_host, host))| {
                            let mut profile =
                                device(format!("cuda:{index:02}"), vec![estimate(1, run, 1, host)]);
                            profile.available_at_ms = available;
                            profile.initially_warm = warm;
                            profile.cold_setup_ms = cold;
                            profile.warm_setup_ms = warm_setup;
                            profile.setup_host_ram_bytes = setup_host;
                            profile
                        },
                    )
                    .collect::<Vec<_>>();
                let mut input = request(children, vec![1], devices);
                input.host_headroom_bytes = headroom;

                let mut allocations = Vec::new();
                fn enumerate(
                    remaining: u32,
                    index: usize,
                    counts: &mut Vec<u32>,
                    out: &mut Vec<Vec<u32>>,
                    devices: usize,
                ) {
                    if index + 1 == devices {
                        counts.push(remaining);
                        out.push(counts.clone());
                        counts.pop();
                        return;
                    }
                    for count in 0..=remaining {
                        counts.push(count);
                        enumerate(remaining - count, index + 1, counts, out, devices);
                        counts.pop();
                    }
                }
                enumerate(children, 0, &mut Vec::new(), &mut allocations, specs.len());
                let oracle = allocations
                    .into_iter()
                    .filter_map(|counts| {
                        let host = counts
                            .iter()
                            .zip(&specs)
                            .filter(|(count, _)| **count > 0)
                            .map(|(_, spec)| spec.5 + spec.6)
                            .sum::<u64>();
                        if host > headroom {
                            return None;
                        }
                        let mut cardinality = 0usize;
                        let mut makespan = 0u64;
                        let mut sum = 0u128;
                        let mut setup = 0u128;
                        for (&count, spec) in counts.iter().zip(&specs) {
                            if count == 0 {
                                continue;
                            }
                            cardinality += 1;
                            let first_setup = if spec.1 { spec.3 } else { spec.2 };
                            let first = spec.0 + first_setup + spec.4;
                            let period = spec.3 + spec.4;
                            let tail = u64::from(count - 1);
                            makespan = makespan.max(first + tail * period);
                            sum += u128::from(count) * u128::from(first)
                                + u128::from(period) * u128::from(count) * u128::from(count - 1)
                                    / 2;
                            setup += u128::from(first_setup)
                                + u128::from(spec.3) * u128::from(count - 1);
                        }
                        let stable_assignment = counts
                            .iter()
                            .enumerate()
                            .filter(|(_, count)| **count > 0)
                            .map(|(index, count)| (index, *count))
                            .collect::<Vec<_>>();
                        Some((
                            std::cmp::Reverse(cardinality),
                            makespan,
                            sum,
                            setup,
                            stable_assignment,
                        ))
                    })
                    .min();

                match (BatchPartitionPlanner::plan(&input), oracle) {
                    (Ok(plan), Some(oracle)) => {
                        assert_eq!(plan.optimization, BatchOptimization::ExactSmall);
                        assert_eq!(
                            (
                                std::cmp::Reverse(plan.devices_used),
                                plan.predicted_parent_makespan_ms,
                                plan.predicted_sum_completion_ms,
                                plan.predicted_setup_ms,
                                specs
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(index, _)| {
                                        let count = plan
                                            .partitions
                                            .iter()
                                            .filter(|partition| {
                                                partition.device_id.as_str()
                                                    == format!("cuda:{index:02}")
                                            })
                                            .count()
                                            as u32;
                                        (count > 0).then_some((index, count))
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                            oracle,
                            "{specs:?}, children={children}, headroom={headroom}"
                        );
                        assert_eq!(
                            plan.predicted_sum_completion_ms,
                            plan.partitions
                                .iter()
                                .map(|partition| {
                                    u128::from(partition.estimated_finish_ms)
                                        * u128::from(partition.size)
                                })
                                .sum::<u128>()
                        );
                    }
                    (Err(BatchPartitionError::Infeasible { .. }), None) => {}
                    (actual, expected) => panic!(
                        "planner/oracle feasibility mismatch for {specs:?}, children={children}, \
                         headroom={headroom}: {actual:?} vs {expected:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn singleton_u32_max_plan_is_compact_and_supports_lazy_random_access() {
    let mut first = device("cuda:a", vec![estimate(1, 3, 1, 2)]);
    first.cold_setup_ms = 7;
    first.warm_setup_ms = 2;
    let mut second = device("cuda:b", vec![estimate(1, 5, 1, 3)]);
    second.cold_setup_ms = 1;
    second.warm_setup_ms = 1;

    let plan =
        BatchPartitionPlanner::plan(&request(u32::MAX, vec![1], vec![first, second])).unwrap();

    assert_eq!(
        plan.representation(),
        BatchPlanRepresentation::CompactSingleton
    );
    assert_eq!(plan.partition_count(), u32::MAX);
    assert_eq!(plan.retained_partition_records(), 2);

    let first_window = plan.partition_window(0, 3).unwrap();
    let middle_window = plan.partition_window(2_000_000_000, 3).unwrap();
    let last_window = plan.partition_window(u32::MAX - 3, 3).unwrap();
    assert_eq!(
        middle_window,
        plan.partition_window(2_000_000_000, 3).unwrap()
    );
    assert_eq!(
        first_window
            .iter()
            .map(|partition| partition.child_start)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        middle_window
            .iter()
            .map(|partition| partition.child_start)
            .collect::<Vec<_>>(),
        [2_000_000_000, 2_000_000_001, 2_000_000_002]
    );
    assert_eq!(
        last_window
            .iter()
            .map(|partition| partition.child_start)
            .collect::<Vec<_>>(),
        [u32::MAX - 3, u32::MAX - 2, u32::MAX - 1]
    );
    assert_eq!(
        plan.partition_at(u32::MAX - 1).ok(),
        last_window.last().cloned()
    );
    assert!(matches!(
        plan.partition_at(u32::MAX),
        Err(BatchPartitionError::PartitionIndexOutOfRange {
            index: u32::MAX,
            partition_count: u32::MAX,
        })
    ));
    assert!(matches!(
        plan.partition_window(0, MAX_PARTITION_WINDOW + 1),
        Err(BatchPartitionError::PartitionWindowTooLarge { .. })
    ));
    assert!(matches!(
        plan.partition_window(u32::MAX, 1),
        Err(BatchPartitionError::PartitionIndexOutOfRange { .. })
    ));
}

#[test]
fn huge_native_parent_uses_compact_singleton_fallback() {
    let estimates = vec![estimate(1, 5, 1, 1), estimate(3, 8, 2, 2)];
    let plan = BatchPartitionPlanner::plan(&request(
        u32::MAX,
        vec![1, 3],
        vec![device("cuda:a", estimates)],
    ))
    .unwrap();

    assert_eq!(
        plan.representation(),
        BatchPlanRepresentation::CompactSingleton
    );
    assert_eq!(plan.optimization, BatchOptimization::BoundedHeuristic);
    assert_eq!(plan.partition_count(), u32::MAX);
    assert_eq!(plan.retained_partition_records(), 1);
    assert_eq!(
        plan.partition_window(u32::MAX - 2, 2)
            .unwrap()
            .iter()
            .map(|partition| partition.size)
            .collect::<Vec<_>>(),
        [1, 1]
    );
}

#[test]
fn singleton_timing_overflow_is_typed_in_every_solver_class() {
    let overflowing = |id: String, offset: u64| {
        let mut profile = device(id, vec![estimate(1, u64::MAX, 0, 0)]);
        profile.cold_setup_ms = u64::MAX - offset;
        profile.warm_setup_ms = u64::MAX - offset;
        profile
    };
    let cases = [
        vec![overflowing("cuda:00".into(), 0)],
        vec![
            overflowing("cuda:00".into(), 0),
            overflowing("cuda:01".into(), 1),
        ],
        (0..=EXACT_SMALL_DEVICE_LIMIT)
            .map(|index| overflowing(format!("cuda:{index:02}"), index as u64))
            .collect(),
    ];

    for devices in cases {
        let error = BatchPartitionPlanner::plan(&request(u32::MAX, vec![1], devices)).unwrap_err();
        assert_eq!(error, BatchPartitionError::TimingOverflow);
    }
}

#[test]
fn initial_wave_cardinality_precedes_makespan() {
    let mut fast = device("cuda:a", vec![estimate(1, 1, 1, 0)]);
    fast.cold_setup_ms = 0;
    fast.warm_setup_ms = 1;
    let mut slow = device("cuda:b", vec![estimate(1, 100, 1, 0)]);
    slow.cold_setup_ms = 0;
    slow.warm_setup_ms = 100;

    let plan = BatchPartitionPlanner::plan(&request(2, vec![1], vec![fast, slow])).unwrap();
    let initial_wave = plan
        .partitions
        .iter()
        .filter(|partition| partition.estimated_start_ms == 0)
        .map(|partition| partition.device_id.clone())
        .collect::<BTreeSet<_>>();

    assert_eq!(plan.devices_used, 2);
    assert_eq!(initial_wave.len(), 2);
}

#[test]
fn exact_small_solver_bounds_are_public_and_observable() {
    assert_eq!(EXACT_SMALL_DEVICE_LIMIT, 8);

    let devices = (0..=EXACT_SMALL_DEVICE_LIMIT)
        .map(|index| {
            let mut profile = device(
                format!("cuda:{index:02}"),
                vec![estimate(1, 10 + index as u64, 1, 0)],
            );
            profile.cold_setup_ms = 0;
            profile.warm_setup_ms = 0;
            profile
        })
        .collect();
    let plan = BatchPartitionPlanner::plan(&request(2, vec![1], devices)).unwrap();
    assert_eq!(plan.optimization, BatchOptimization::BoundedHeuristic);
}

#[test]
fn exact_small_singleton_solver_has_no_child_count_bound() {
    let specs = [
        (0, false, 23, 2, 17),
        (11, true, 99, 1, 19),
        (7, false, 3, 4, 29),
        (41, true, 80, 3, 13),
        (5, false, 31, 0, 23),
        (17, true, 77, 2, 31),
        (2, false, 11, 5, 11),
        (29, true, 60, 1, 37),
    ];
    let devices = specs
        .iter()
        .enumerate()
        .map(|(index, &(available, warm, cold, warm_setup, run))| {
            let mut profile = device(format!("cuda:{index:02}"), vec![estimate(1, run, 1, 0)]);
            profile.available_at_ms = available;
            profile.initially_warm = warm;
            profile.cold_setup_ms = cold;
            profile.warm_setup_ms = warm_setup;
            profile
        })
        .collect::<Vec<_>>();
    let plan = BatchPartitionPlanner::plan(&request(100_003, vec![1], devices)).unwrap();

    let completion_count = |limit: u128, spec: (u64, bool, u64, u64, u64)| {
        let first = u128::from(spec.0)
            + u128::from(if spec.1 { spec.3 } else { spec.2 })
            + u128::from(spec.4);
        if limit < first {
            1u32
        } else {
            let period = u128::from(spec.3 + spec.4);
            (limit - first)
                .checked_div(period)
                .map(|completed_after_first| (1 + completed_after_first.min(100_002)) as u32)
                .unwrap_or(100_003)
        }
    };
    let mut low = specs
        .iter()
        .map(|&spec| {
            u128::from(spec.0)
                + u128::from(if spec.1 { spec.3 } else { spec.2 })
                + u128::from(spec.4)
        })
        .max()
        .unwrap();
    let mut high = specs
        .iter()
        .map(|&spec| {
            let first = u128::from(spec.0)
                + u128::from(if spec.1 { spec.3 } else { spec.2 })
                + u128::from(spec.4);
            first + 100_002 * u128::from(spec.3 + spec.4)
        })
        .min()
        .unwrap()
        .max(low);
    while low < high {
        let middle = (low + high) / 2;
        let covered = specs
            .iter()
            .map(|&spec| completion_count(middle, spec))
            .sum::<u32>();
        if covered >= 100_003 {
            high = middle;
        } else {
            low = middle + 1;
        }
    }

    assert_eq!(plan.optimization, BatchOptimization::ExactSmall);
    assert!(plan.optimizer_states_evaluated < (1 << EXACT_SMALL_DEVICE_LIMIT));
    assert_eq!(plan.materialization_passes, 0);
    assert_eq!(
        plan.representation(),
        BatchPlanRepresentation::CompactSingleton
    );
    assert_eq!(plan.retained_partition_records(), 8);
    assert_eq!(plan.devices_used, 8);
    assert_eq!(u128::from(plan.predicted_parent_makespan_ms), low);
    assert_exact_coverage(&plan);
}

#[test]
fn exact_two_device_host_coupling_scales_to_large_parents() {
    let mut expensive = device("cuda:a", vec![estimate(1, 5, 1, 10)]);
    expensive.cold_setup_ms = 0;
    expensive.warm_setup_ms = 95;
    let mut efficient = device("cuda:b", vec![estimate(1, 6, 1, 5)]);
    efficient.cold_setup_ms = 0;
    efficient.warm_setup_ms = 0;
    let mut input = request(100_003, vec![1], vec![expensive, efficient]);
    input.host_headroom_bytes = 10;

    let plan = BatchPartitionPlanner::plan(&input).unwrap();

    assert_eq!(plan.optimization, BatchOptimization::ExactSmall);
    assert_eq!(plan.devices_used, 1);
    assert_eq!(plan.predicted_parent_makespan_ms, 600_018);
    assert!(plan
        .partitions
        .iter()
        .all(|partition| partition.device_id.as_str() == "cuda:b"));
}

#[test]
fn host_byte_overflow_is_not_reported_as_timing_overflow() {
    let mut overflowing = device("cuda:a", vec![estimate(1, 1, 1, 1)]);
    overflowing.setup_host_ram_bytes = u64::MAX;
    let error = BatchPartitionPlanner::plan(&request(1, vec![1], vec![overflowing])).unwrap_err();
    assert!(matches!(error, BatchPartitionError::HostMemoryOverflow));
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

    assert_eq!(
        plan.partition_at(0).unwrap().device_id.as_str(),
        "cuda:warm"
    );
    assert_eq!(
        plan.partition_at(0).unwrap().setup_disposition,
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
    assert_eq!(
        forward.partition_at(0).unwrap().device_id.as_str(),
        "cuda:a"
    );
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

    let plan = BatchPartitionPlanner::plan(&request(
        NATIVE_MATERIALIZATION_CHILD_LIMIT,
        vec![1, 3, 5],
        devices,
    ))
    .unwrap();

    assert_exact_coverage(&plan);
    assert!(plan.partitions.len() <= NATIVE_MATERIALIZATION_CHILD_LIMIT as usize);
    assert_eq!(plan.devices_used, 64);
    assert_eq!(plan.optimization, BatchOptimization::BoundedHeuristic);
    assert!(plan.optimizer_states_evaluated <= BOUNDED_HEURISTIC_STRATEGY_LIMIT);
    assert!(usize::from(plan.materialization_passes) <= plan.optimizer_states_evaluated);
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
