//! Consolidated, weight-free multi-GPU acceptance suite.
//!
//! CI entry point:
//! `cargo test -p mold-scheduler --test multi_gpu_acceptance`

use std::collections::{BTreeMap, BTreeSet};

use mold_scheduler::{
    operation_budget, AdaptiveBatchPlan, Backend, BatchDeviceProfile, BatchPartitionError,
    BatchPartitionPlanner, BatchPartitionRequest, BatchSizeEstimate, BlockedReason,
    CandidatePlacement, DeviceAdminState, DeviceHealth, DeviceSnapshot, Planner, PlannerSnapshot,
    WorkId, WorkSnapshot,
};

const GIB: u64 = 1024 * 1024 * 1024;

fn device(id: impl Into<String>, vram_gib: u64) -> DeviceSnapshot {
    DeviceSnapshot::idle(id.into(), vram_gib * GIB)
}

fn placement(
    device_id: impl Into<String>,
    vram_gib: u64,
    host_gib: u64,
    run_ms: u64,
) -> CandidatePlacement {
    let device_id = device_id.into();
    CandidatePlacement::new(
        device_id.clone(),
        format!("execution:{device_id}"),
        host_gib * GIB,
    )
    .with_execution_equivalence("stable-cpu-seeded-equivalence")
    .with_vram(vram_gib * GIB)
    .with_device_available_vram(256 * GIB)
    .with_timing(1_000, 100, run_ms)
}

fn work(id: impl Into<String>, rank: u64, placements: Vec<CandidatePlacement>) -> WorkSnapshot {
    WorkSnapshot::new(id.into(), rank, placements)
}

fn snapshot(devices: Vec<DeviceSnapshot>, work: Vec<WorkSnapshot>) -> PlannerSnapshot {
    PlannerSnapshot::new(17, 29, 10_000, 4_096 * GIB, devices, work)
}

fn assigned(plan: &mold_scheduler::Plan) -> BTreeMap<String, String> {
    plan.immediate_leases
        .iter()
        .map(|lease| {
            (
                lease.work_id.as_str().to_owned(),
                lease.device_id.as_str().to_owned(),
            )
        })
        .collect()
}

fn all_placements(ids: &[String], vram_gib: u64, run_ms: u64) -> Vec<CandidatePlacement> {
    ids.iter()
        .map(|id| placement(id, vram_gib, 0, run_ms))
        .collect()
}

fn batch_device(id: impl Into<String>, vram_gib: u64, run_ms: u64) -> BatchDeviceProfile {
    BatchDeviceProfile {
        device_id: id.into().into(),
        available_at_ms: 0,
        initially_warm: false,
        partition_capacity: u32::MAX,
        available_vram_bytes: vram_gib * GIB,
        cold_setup_ms: 1_000,
        warm_setup_ms: 100,
        setup_host_ram_bytes: 0,
        size_estimates: vec![BatchSizeEstimate {
            size: 1,
            predicted_run_ms: run_ms,
            predicted_vram_bytes: 8 * GIB,
            predicted_host_ram_bytes: GIB,
        }],
    }
}

fn assert_ordered_batch_coverage(plan: &AdaptiveBatchPlan) {
    let partitions = plan.partitions.iter().collect::<Vec<_>>();
    assert_eq!(partitions.len(), plan.child_count as usize);
    let mut next_child = 0;
    for (index, partition) in partitions.iter().enumerate() {
        assert_eq!(partition.partition_index, index as u32 + 1);
        assert_eq!(partition.partition_count, plan.child_count);
        assert_eq!(partition.child_start, next_child);
        assert_eq!(partition.size, 1);
        next_child += partition.size;
    }
    assert_eq!(next_child, plan.child_count);
}

#[test]
fn arbitrary_n_inventories_have_no_zero_one_two_or_power_of_two_assumption() {
    for count in [0_usize, 1, 2, 8, 16, 64] {
        let ids = (0..count)
            .map(|index| format!("cuda:GPU-{index:032x}"))
            .collect::<Vec<_>>();
        let devices = ids
            .iter()
            .map(|id| device(id.clone(), 24))
            .collect::<Vec<_>>();
        let jobs = (0..count)
            .map(|index| {
                work(
                    format!("job-{index:03}"),
                    index as u64,
                    all_placements(&ids, 8, 30_000),
                )
            })
            .collect::<Vec<_>>();

        let plan = Planner::default()
            .plan(&snapshot(devices, jobs))
            .expect("arbitrary-N plan");
        assert_eq!(plan.immediate_leases.len(), count);
        assert_eq!(
            plan.immediate_leases
                .iter()
                .map(|lease| lease.device_id.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            count
        );
        assert!(plan.operations_evaluated <= plan.operation_budget);
    }
}

#[test]
fn singleton_parent_batches_fill_every_device_for_arbitrary_n() {
    let empty = BatchPartitionPlanner::plan(&BatchPartitionRequest {
        child_count: 1,
        now_ms: 0,
        native_batch_sizes: vec![1],
        host_headroom_bytes: GIB,
        devices: vec![],
    });
    assert_eq!(empty, Err(BatchPartitionError::NoCandidateDevices));

    for count in [1_usize, 2, 8, 16, 64] {
        let profiles = (0..count)
            .map(|index| batch_device(format!("cuda:GPU-{index:032x}"), 24, 30_000))
            .collect::<Vec<_>>();
        let plan = BatchPartitionPlanner::plan(&BatchPartitionRequest {
            child_count: (count * 3 + 1) as u32,
            now_ms: 0,
            native_batch_sizes: vec![1],
            host_headroom_bytes: 4_096 * GIB,
            devices: profiles,
        })
        .expect("singleton parent plan");

        assert_ordered_batch_coverage(&plan);
        assert_eq!(
            plan.partitions
                .iter()
                .map(|partition| partition.device_id)
                .collect::<BTreeSet<_>>()
                .len(),
            count
        );
    }
}

#[test]
fn homogeneous_two_3090_and_eight_b200_fixtures_saturate_without_pair_logic() {
    for (label, count, vram_gib, run_ms) in [
        ("rtx3090", 2_usize, 24_u64, 30_000_u64),
        ("b200", 8, 180, 4_000),
    ] {
        let ids = (0..count)
            .map(|index| format!("cuda:GPU-{label}-{index:02}"))
            .collect::<Vec<_>>();
        let devices = ids
            .iter()
            .map(|id| device(id.clone(), vram_gib))
            .collect::<Vec<_>>();
        let jobs = (0..count * 2)
            .map(|index| {
                work(
                    format!("{label}-job-{index:02}"),
                    index as u64,
                    all_placements(&ids, 8, run_ms),
                )
            })
            .collect::<Vec<_>>();
        let plan = Planner::default()
            .plan(&snapshot(devices, jobs))
            .expect("homogeneous fixture");

        let first_wave = plan
            .immediate_leases
            .iter()
            .map(|lease| lease.device_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(first_wave.len(), count, "{label}");
        assert_eq!(plan.lanes.len(), count, "{label}");
        assert!(plan.operations_evaluated <= plan.operation_budget);
    }
}

#[test]
fn heterogeneous_speed_estimates_put_more_future_work_on_the_fast_device() {
    let ids = ["cuda:3090".to_owned(), "cuda:b200".to_owned()];
    let jobs = (0..12)
        .map(|index| {
            work(
                format!("mixed-speed-{index:02}"),
                index,
                vec![
                    placement("cuda:3090", 8, 0, 30_000).with_device_available_vram(24 * GIB),
                    placement("cuda:b200", 8, 0, 4_000).with_device_available_vram(180 * GIB),
                ],
            )
        })
        .collect();
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device(ids[0].clone(), 24), device(ids[1].clone(), 180)],
            jobs,
        ))
        .unwrap();
    let lane_counts = plan
        .lanes
        .iter()
        .map(|lane| (lane.device_id.as_str(), lane.assignments.len()))
        .collect::<BTreeMap<_, _>>();

    assert_eq!(plan.immediate_leases.len(), 2);
    assert!(
        lane_counts["cuda:b200"] > lane_counts["cuda:3090"],
        "faster B200 lane should receive more of the predicted future work: {lane_counts:?}"
    );
    assert!(plan.operations_evaluated <= plan.operation_budget);
}

#[test]
fn twelve_gib_pressure_never_overcommits_vram_or_aggregate_host_ram() {
    let physical_pressure =
        Planner::default()
            .plan(&snapshot(
                vec![device("cuda:GPU-12GiB", 12)],
                vec![work(
                    "needs-13GiB",
                    0,
                    vec![placement("cuda:GPU-12GiB", 13, 0, 1_000)
                        .with_device_available_vram(12 * GIB)],
                )],
            ))
            .unwrap();
    assert!(physical_pressure.immediate_leases.is_empty());
    assert_eq!(
        physical_pressure.blocked_reason(&WorkId::from("needs-13GiB")),
        Some(&BlockedReason::InsufficientVram)
    );

    let ids = ["cuda:a".to_owned(), "cuda:b".to_owned()];
    let mut aggregate = snapshot(
        ids.iter().map(|id| device(id.clone(), 24)).collect(),
        vec![
            work("older-8", 0, all_placements(&ids, 8, 10_000)),
            work("younger-8", 1, all_placements(&ids, 8, 10_000)),
        ],
    );
    aggregate.host_memory.headroom_bytes = 12 * GIB;
    for job in &mut aggregate.work {
        for candidate in &mut job.candidate_placements {
            candidate.incremental_host_ram_bytes = 8 * GIB;
        }
    }
    let plan = Planner::default().plan(&aggregate).unwrap();
    assert_eq!(plan.immediate_leases.len(), 1);
    assert_eq!(plan.reservation.total_host_ram_bytes, 8 * GIB);
    assert_eq!(
        plan.blocked_reason(&WorkId::from("younger-8")),
        Some(&BlockedReason::AggregateHostRamReserved)
    );

    let mut priority = snapshot(
        ids.iter().map(|id| device(id.clone(), 24)).collect(),
        vec![
            work("older-8", 0, all_placements(&ids, 8, 10_000)),
            work("younger-4-a", 1, all_placements(&ids, 8, 10_000)),
            work("younger-4-b", 2, all_placements(&ids, 8, 10_000)),
        ],
    );
    priority.host_memory.headroom_bytes = 12 * GIB;
    for (index, job) in priority.work.iter_mut().enumerate() {
        let host = if index == 0 { 8 } else { 4 };
        for candidate in &mut job.candidate_placements {
            candidate.incremental_host_ram_bytes = host * GIB;
        }
    }
    let assignments = assigned(&Planner::default().plan(&priority).unwrap());
    assert!(assignments.contains_key("older-8"));
    assert_eq!(
        assignments
            .keys()
            .filter(|id| id.starts_with("younger-4"))
            .count(),
        1,
        "two younger 4 GiB jobs must not replace the older 8 GiB job"
    );
}

#[test]
fn heterogeneous_capability_speed_vram_and_lifecycle_edges_are_typed() {
    let devices = vec![
        device("cuda:3090-a", 24),
        device("cuda:3090-b", 24),
        device("cuda:b200", 180),
        device("cuda:12g", 12),
        device("metal:default", 64).with_backend(Backend::Metal),
        device("cuda:disabled", 24).with_admin_state(DeviceAdminState::Disabled),
        device("cuda:draining", 24).with_admin_state(DeviceAdminState::Draining),
        device("cuda:degraded", 24).with_health(DeviceHealth::Degraded),
        device("cuda:unavailable", 24).with_health(DeviceHealth::Unavailable),
    ];
    let cuda_general = ["cuda:3090-a", "cuda:3090-b", "cuda:b200"]
        .into_iter()
        .map(|id| placement(id, 8, 0, if id == "cuda:b200" { 4_000 } else { 30_000 }))
        .collect();
    let work = vec![
        work(
            "b200-only",
            0,
            vec![placement("cuda:b200", 80, 0, 4_000).with_device_available_vram(180 * GIB)],
        )
        .with_hard_device("cuda:b200"),
        work("portable-a", 1, cuda_general),
        work(
            "portable-b",
            2,
            vec![
                placement("cuda:3090-a", 8, 0, 30_000),
                placement("cuda:3090-b", 8, 0, 30_000),
            ],
        ),
        work(
            "metal-only",
            3,
            vec![placement("metal:default", 8, 0, 20_000)],
        )
        .with_backend_requirement(Backend::Metal),
        work(
            "too-large-12g",
            4,
            vec![placement("cuda:12g", 13, 0, 30_000).with_device_available_vram(12 * GIB)],
        ),
        work(
            "disabled",
            5,
            vec![placement("cuda:disabled", 8, 0, 30_000)],
        )
        .with_hard_device("cuda:disabled"),
        work(
            "draining",
            6,
            vec![placement("cuda:draining", 8, 0, 30_000)],
        )
        .with_hard_device("cuda:draining"),
        work(
            "degraded",
            7,
            vec![placement("cuda:degraded", 8, 0, 30_000)],
        )
        .with_hard_device("cuda:degraded"),
        work(
            "unavailable",
            8,
            vec![placement("cuda:unavailable", 8, 0, 30_000)],
        )
        .with_hard_device("cuda:unavailable"),
    ];
    let plan = Planner::default().plan(&snapshot(devices, work)).unwrap();
    let assignments = assigned(&plan);

    assert_eq!(assignments["b200-only"], "cuda:b200");
    assert_eq!(assignments["metal-only"], "metal:default");
    assert_eq!(
        assignments.values().collect::<BTreeSet<_>>().len(),
        assignments.len()
    );
    for lease in &plan.immediate_leases {
        assert!(
            lease.placement.predicted_vram_bytes <= lease.placement.device_available_vram_bytes,
            "{} received an infeasible edge",
            lease.work_id
        );
    }
    for (id, reason) in [
        ("too-large-12g", BlockedReason::InsufficientVram),
        ("disabled", BlockedReason::DeviceDisabled),
        ("draining", BlockedReason::DeviceDraining),
        ("degraded", BlockedReason::DeviceDegraded),
        ("unavailable", BlockedReason::DeviceUnavailable),
    ] {
        assert_eq!(
            plan.blocked_reason(&WorkId::from(id)),
            Some(&reason),
            "{id}"
        );
    }
}

#[test]
fn uuid_order_changes_and_mig_children_preserve_exact_stable_assignments() {
    let ids = vec![
        "cuda:MIG-GPU-aaaaaaaa/1/0".to_owned(),
        "cuda:MIG-GPU-aaaaaaaa/2/0".to_owned(),
        "cuda:GPU-bbbbbbbb".to_owned(),
    ];
    let make = |reverse: bool| {
        let mut devices = vec![
            device(ids[0].clone(), 20),
            device(ids[1].clone(), 20),
            device(ids[2].clone(), 24),
        ];
        let mut jobs = (0..3)
            .map(|index| {
                let mut candidates = all_placements(&ids, 8, 10_000);
                if reverse {
                    candidates.reverse();
                }
                work(format!("stable-{index}"), index, candidates)
            })
            .collect::<Vec<_>>();
        if reverse {
            devices.reverse();
            jobs.reverse();
        }
        Planner::default().plan(&snapshot(devices, jobs)).unwrap()
    };

    let forward = make(false);
    let reversed = make(true);
    assert_eq!(forward, reversed);
    assert_eq!(
        forward
            .immediate_leases
            .iter()
            .map(|lease| lease.device_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
    assert!(forward.immediate_leases.iter().all(|lease| {
        lease
            .placement
            .execution_equivalence_fingerprint
            .as_ref()
            .is_some_and(|fingerprint| fingerprint.as_str() == "stable-cpu-seeded-equivalence")
    }));
}

#[test]
fn priority_and_cardinality_reach_compatible_work_beyond_rank_200() {
    let ids = (0..64)
        .map(|index| format!("cuda:GPU-{index:02}"))
        .collect::<Vec<_>>();
    let devices = ids
        .iter()
        .map(|id| device(id.clone(), 24))
        .collect::<Vec<_>>();
    let mut jobs = (0..250)
        .map(|index| {
            work(
                format!("infeasible-{index:03}"),
                index,
                vec![placement("missing", 8, 0, 10_000)],
            )
        })
        .collect::<Vec<_>>();
    jobs.extend((0..63).map(|index| {
        work(
            format!("flex-{index:02}"),
            250 + index,
            all_placements(&ids, 8, 10_000),
        )
    }));
    jobs.push(
        work("rank-313-hard-pin", 313, all_placements(&ids, 8, 10_000))
            .with_hard_device("cuda:GPU-00"),
    );

    let plan = Planner::default()
        .plan(&snapshot(devices, jobs))
        .expect("priority/cardinality plan");
    let assignments = assigned(&plan);
    assert_eq!(assignments.len(), 64);
    assert_eq!(assignments["rank-313-hard-pin"], "cuda:GPU-00");
    assert_eq!(assignments.values().collect::<BTreeSet<_>>().len(), 64);
}

#[test]
fn eight_and_sixty_four_device_envelopes_obey_deterministic_operation_budgets() {
    let eight_ids = (0..8)
        .map(|index| format!("cuda:eight-{index:02}"))
        .collect::<Vec<_>>();
    let eight_plan = Planner::default()
        .plan(&snapshot(
            eight_ids.iter().map(|id| device(id.clone(), 180)).collect(),
            (0..200)
                .map(|index| {
                    let mut candidates = all_placements(&eight_ids, 8, 4_000 + index);
                    candidates[index as usize % 8].affinity_penalty = 1;
                    work(format!("eight-{index:03}"), index, candidates)
                })
                .collect(),
        ))
        .unwrap();
    assert_eq!(eight_plan.operation_budget, operation_budget(8, 64));
    assert!(eight_plan.operations_evaluated <= eight_plan.operation_budget);

    let sixty_four_ids = (0..64)
        .map(|index| format!("cuda:sixty-four-{index:02}"))
        .collect::<Vec<_>>();
    let sixty_four_plan = Planner::default()
        .plan(&snapshot(
            sixty_four_ids
                .iter()
                .map(|id| device(id.clone(), 180))
                .collect(),
            (0..10_000)
                .map(|index| {
                    let target = &sixty_four_ids[index as usize % 64];
                    work(
                        format!("sixty-four-{index:05}"),
                        index,
                        vec![placement(target, 8, 0, 4_000)],
                    )
                })
                .collect(),
        ))
        .unwrap();
    assert_eq!(sixty_four_plan.operation_budget, operation_budget(64, 200));
    assert!(sixty_four_plan.operations_evaluated <= sixty_four_plan.operation_budget);
    assert_eq!(sixty_four_plan.immediate_leases.len(), 64);
}
