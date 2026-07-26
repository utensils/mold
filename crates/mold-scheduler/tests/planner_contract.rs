use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use mold_scheduler::{
    operation_budget, optimization_horizon, BlockedReason, CandidatePlacement, DeviceAdminState,
    DeviceHealth, DeviceSnapshot, EligibilityIndex, GrantValidationSnapshot, OptimizerState,
    PlanValidationError, Planner, PlannerConfig, PlannerError, PlannerSnapshot, PlanningMode,
    PriorityClass, WorkId, WorkSnapshot,
};

const GIB: u64 = 1024 * 1024 * 1024;

fn device(id: &str) -> DeviceSnapshot {
    DeviceSnapshot::idle(id, 24 * GIB)
}

fn candidate(device_id: &str, host_gib: u64) -> CandidatePlacement {
    CandidatePlacement::new(device_id, "exec", host_gib * GIB)
        .with_vram(8 * GIB)
        .with_timing(1_000, 50, 10_000)
}

fn work(id: &str, rank: u64, candidates: Vec<CandidatePlacement>) -> WorkSnapshot {
    WorkSnapshot::new(id, rank, candidates)
}

fn snapshot(
    devices: Vec<DeviceSnapshot>,
    work: Vec<WorkSnapshot>,
    host_gib: u64,
) -> PlannerSnapshot {
    PlannerSnapshot::new(7, 11, 1_000, host_gib * GIB, devices, work)
}

fn assignments(plan: &mold_scheduler::Plan) -> BTreeMap<String, String> {
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

#[test]
fn supports_zero_one_two_eight_sixteen_and_sixty_four_devices() {
    for count in [0, 1, 2, 8, 16, 64] {
        let devices = (0..count)
            .map(|index| device(&format!("cuda:{index:032x}")))
            .collect::<Vec<_>>();
        let jobs = (0..count)
            .map(|index| {
                let device_id = format!("cuda:{index:032x}");
                work(
                    &format!("work-{index:03}"),
                    index as u64,
                    vec![candidate(&device_id, 1)],
                )
            })
            .collect::<Vec<_>>();

        let plan = Planner::default()
            .plan(&snapshot(devices, jobs, 128))
            .expect("valid plan");
        assert_eq!(plan.immediate_leases.len(), count);
        assert_eq!(
            plan.immediate_leases
                .iter()
                .map(|lease| &lease.device_id)
                .collect::<BTreeSet<_>>()
                .len(),
            count
        );
    }
}

#[test]
fn globally_rematches_a_flexible_job_around_a_specialist() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work(
                    "older-flexible",
                    0,
                    vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
                ),
                work(
                    "younger-hard-pinned",
                    1,
                    vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
                )
                .with_hard_device("gpu-0"),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(
        assignments(&plan),
        BTreeMap::from([
            ("older-flexible".into(), "gpu-1".into()),
            ("younger-hard-pinned".into(), "gpu-0".into()),
        ])
    );
}

#[test]
fn sixty_four_way_rematching_preserves_a_late_hard_pin_and_full_cardinality() {
    let devices = (0..64)
        .map(|index| device(&format!("gpu-{index:02}")))
        .collect::<Vec<_>>();
    let all_candidates = || {
        (0..64)
            .map(|index| candidate(&format!("gpu-{index:02}"), 0))
            .collect::<Vec<_>>()
    };
    let mut jobs = (0..63)
        .map(|index| work(&format!("flex-{index:02}"), index, all_candidates()))
        .collect::<Vec<_>>();
    jobs.push(work("late-hard-pin", 63, all_candidates()).with_hard_device("gpu-00"));

    let plan = Planner::default()
        .plan(&snapshot(devices, jobs, 128))
        .expect("valid plan");
    let assigned = assignments(&plan);

    assert_eq!(assigned.len(), 64);
    assert_eq!(assigned["late-hard-pin"], "gpu-00");
    assert_eq!(assigned.values().collect::<BTreeSet<_>>().len(), 64);
}

#[test]
fn rematching_selects_the_minimum_aggregate_host_ram_full_matching() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work(
                    "older-flexible",
                    0,
                    vec![candidate("gpu-0", 8), candidate("gpu-1", 1)],
                ),
                work("younger-specialist", 1, vec![candidate("gpu-0", 1)]),
            ],
            3,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases.len(), 2);
    assert_eq!(plan.reservation.total_host_ram_bytes, 2 * GIB);
    assert_eq!(
        assignments(&plan),
        BTreeMap::from([
            ("older-flexible".into(), "gpu-1".into()),
            ("younger-specialist".into(), "gpu-0".into()),
        ])
    );
}

#[test]
fn aggregate_host_ram_rechecks_the_actual_rematched_total() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work(
                    "older-flexible",
                    0,
                    vec![candidate("gpu-0", 2), candidate("gpu-1", 6)],
                ),
                work("younger-specialist", 1, vec![candidate("gpu-0", 3)]),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(
        assignments(&plan),
        BTreeMap::from([("older-flexible".into(), "gpu-0".into())])
    );
    assert_eq!(plan.reservation.total_host_ram_bytes, 2 * GIB);
    assert_eq!(
        plan.blocked_reason(&WorkId::from("younger-specialist")),
        Some(&BlockedReason::AggregateHostRamReserved)
    );
}

#[test]
fn full_ready_set_matching_reaches_compatible_work_beyond_rank_200() {
    let mut jobs = (0..250)
        .map(|index| {
            work(
                &format!("blocked-{index:03}"),
                index,
                vec![candidate("missing-device", 1)],
            )
        })
        .collect::<Vec<_>>();
    jobs.push(work(
        "rank-250-compatible",
        250,
        vec![candidate("gpu-0", 1)],
    ));

    let plan = Planner::default()
        .plan(&snapshot(vec![device("gpu-0")], jobs, 8))
        .expect("valid plan");
    assert_eq!(
        plan.immediate_leases[0].work_id.as_str(),
        "rank-250-compatible"
    );
    assert!(plan.optimization_horizon_length <= 200);
}

#[test]
fn aggregate_host_ram_rejects_two_individually_admissible_jobs() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work("older-8", 0, vec![candidate("gpu-0", 8)]),
                work("younger-8", 1, vec![candidate("gpu-1", 8)]),
            ],
            12,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases.len(), 1);
    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "older-8");
    assert_eq!(plan.reservation.total_host_ram_bytes, 8 * GIB);
    assert_eq!(
        plan.blocked_reason(&WorkId::from("younger-8")),
        Some(&BlockedReason::AggregateHostRamReserved)
    );
}

#[test]
fn priority_does_not_replace_an_older_eight_gib_job_with_two_younger_four_gib_jobs() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work(
                    "older-8",
                    0,
                    vec![candidate("gpu-0", 8), candidate("gpu-1", 8)],
                ),
                work(
                    "younger-4-a",
                    1,
                    vec![candidate("gpu-0", 4), candidate("gpu-1", 4)],
                ),
                work(
                    "younger-4-b",
                    2,
                    vec![candidate("gpu-0", 4), candidate("gpu-1", 4)],
                ),
            ],
            12,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases.len(), 2);
    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "older-8");
    assert_eq!(plan.immediate_leases[1].work_id.as_str(), "younger-4-a");
    assert_eq!(plan.reservation.total_host_ram_bytes, 12 * GIB);
    assert_eq!(
        plan.blocked_reason(&WorkId::from("younger-4-b")),
        Some(&BlockedReason::LowerPriorityOpening)
    );
}

#[test]
fn warm_wait_is_beneficial_bounded_and_expires_without_sleeping() {
    let cold = device("cold");
    let warm = DeviceSnapshot::busy("warm", 24 * GIB, 1_500).with_warm("exec");
    let waiting = work("job", 0, vec![candidate("cold", 1), candidate("warm", 1)])
        .with_warm_wait_started_at(1_000);
    let config = PlannerConfig {
        warm_wait_max_ms: 2_000,
        ..PlannerConfig::default()
    };

    let held = Planner::new(config.clone())
        .plan(&snapshot(
            vec![cold.clone(), warm.clone()],
            vec![waiting.clone()],
            8,
        ))
        .expect("valid plan");
    assert!(held.immediate_leases.is_empty());
    assert_eq!(held.warm_waits[0].deadline_ms, 1_500);
    assert_eq!(
        held.blocked_reason(&WorkId::from("job")),
        Some(&BlockedReason::WarmWait)
    );

    let expired_snapshot =
        PlannerSnapshot::new(8, 12, 1_500, 8 * GIB, vec![cold, warm], vec![waiting]);
    let expired = Planner::new(config)
        .plan(&expired_snapshot)
        .expect("valid plan");
    assert_eq!(expired.immediate_leases[0].device_id.as_str(), "cold");
}

#[test]
fn warm_wait_never_holds_when_cold_now_finishes_first() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![
                device("cold"),
                DeviceSnapshot::busy("warm", 24 * GIB, 10_000).with_warm("exec"),
            ],
            vec![
                work("job", 0, vec![candidate("cold", 1), candidate("warm", 1)])
                    .with_warm_wait_started_at(1_000),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases[0].device_id.as_str(), "cold");
    assert!(plan.warm_waits.is_empty());
}

#[test]
fn third_bypass_forces_the_next_compatible_opening() {
    let warm = DeviceSnapshot::busy("warm", 24 * GIB, 1_500).with_warm("exec");
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("cold"), warm],
            vec![
                work(
                    "forced",
                    100,
                    vec![candidate("cold", 1), candidate("warm", 1)],
                )
                .with_bypass_count(3)
                .with_warm_wait_started_at(1_000),
                work("ordinary", 0, vec![candidate("cold", 1)]),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "forced");
    assert!(plan.warm_waits.is_empty());
}

#[test]
fn a_younger_start_on_a_declined_warm_wait_edge_increments_bypass_once() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![
                device("cold"),
                DeviceSnapshot::busy("warm", 24 * GIB, 1_500).with_warm("exec"),
            ],
            vec![
                work(
                    "waiting",
                    0,
                    vec![candidate("cold", 1), candidate("warm", 1)],
                )
                .with_bypass_count(2)
                .with_warm_wait_started_at(1_000),
                work("younger", 1, vec![candidate("cold", 1)]),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "younger");
    assert_eq!(plan.bypass_updates.len(), 1);
    assert_eq!(plan.bypass_updates[0].work_id.as_str(), "waiting");
    assert_eq!(plan.bypass_updates[0].new_count, 3);
}

#[test]
fn plan_versions_and_memory_generations_fence_stale_grants() {
    let mut input = snapshot(
        vec![device("gpu-0")],
        vec![work("job", 0, vec![candidate("gpu-0", 1)])],
        8,
    );
    input.host_memory.sample_generation = 19;
    input.host_memory.ledger_sequence = 23;
    let plan = Planner::default().plan(&input).expect("valid plan");

    assert_eq!(plan.plan_version, 11);
    assert_eq!(plan.state_version, 7);
    assert_eq!(plan.reservation.sample_generation, 19);
    assert_eq!(plan.reservation.ledger_sequence, 23);
    assert_eq!(plan.validate_for_grant(7, 11, 19, 23), Ok(()));
    assert_eq!(
        plan.validate_for_grant(8, 11, 19, 23),
        Err(PlanValidationError::StaleState {
            planned: 7,
            current: 8
        })
    );
    assert_eq!(
        plan.validate_for_grant(7, 12, 19, 23),
        Err(PlanValidationError::StalePlan {
            planned: 11,
            current: 12
        })
    );
    assert_eq!(
        plan.validate_for_grant(7, 11, 20, 23),
        Err(PlanValidationError::StaleMemorySample {
            planned: 19,
            current: 20
        })
    );
    assert_eq!(
        plan.validate_for_grant(7, 11, 19, 24),
        Err(PlanValidationError::StaleMemoryLedger {
            planned: 23,
            current: 24
        })
    );
}

#[test]
fn input_and_index_mutation_order_do_not_change_the_plan() {
    let devices = vec![device("gpu-b"), device("gpu-a")];
    let jobs = vec![
        work(
            "work-b",
            1,
            vec![candidate("gpu-b", 1), candidate("gpu-a", 1)],
        ),
        work(
            "work-a",
            0,
            vec![candidate("gpu-a", 1), candidate("gpu-b", 1)],
        ),
    ];
    let forward = snapshot(devices.clone(), jobs.clone(), 8);
    let reverse = snapshot(
        devices.into_iter().rev().collect(),
        jobs.iter().cloned().rev().collect(),
        8,
    );

    let mut index_a = EligibilityIndex::new(forward.state_version);
    for job in &jobs {
        index_a.upsert_work(forward.state_version, job);
    }
    let mut index_b = EligibilityIndex::new(forward.state_version);
    for job in jobs.iter().rev() {
        index_b.upsert_work(forward.state_version, job);
    }

    let planner = Planner::default();
    assert_eq!(planner.plan(&forward), planner.plan(&reverse));
    assert_eq!(
        planner.plan_with_index(&forward, &index_a),
        planner.plan_with_index(&forward, &index_b)
    );

    index_b.remove_work(forward.state_version, &WorkId::from("work-a"));
    index_b.upsert_work(forward.state_version, &jobs[1]);
    assert_eq!(
        planner.plan_with_index(&forward, &index_a),
        planner.plan_with_index(&forward, &index_b)
    );
}

#[test]
fn incrementally_mutated_index_equals_full_recomputation_across_mixed_facts() {
    let mut devices = (0..8)
        .map(|index| device(&format!("gpu-{index}")))
        .collect::<Vec<_>>();
    let mut jobs = (0..48)
        .map(|work_index| {
            work(
                &format!("work-{work_index:02}"),
                work_index,
                (0..8)
                    .filter(|device_index| (work_index + device_index) % 3 != 0)
                    .map(|device_index| {
                        candidate(&format!("gpu-{device_index}"), (work_index % 4) + 1)
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let mut state_version = 1;
    let mut index = EligibilityIndex::from_work(state_version, &jobs);
    let planner = Planner::default();

    for turn in 0..192 {
        let work_index = (turn * 17 + 5) % jobs.len();
        let device_index = (turn * 29 + 3) % devices.len();
        let job_count = jobs.len();
        state_version += 1;

        let item = &mut jobs[work_index];
        item.queue_rank = ((turn * 11) % job_count) as u64;
        item.priority_class = match turn % 5 {
            0 => PriorityClass::Critical,
            1 => PriorityClass::Admin,
            _ => PriorityClass::User,
        };
        item.hard_device_id =
            (turn % 7 == 0).then(|| mold_scheduler::DeviceId::new(format!("gpu-{device_index}")));
        item.candidate_placements = (0..8)
            .filter(|candidate_index| (candidate_index + work_index + turn) % ((turn % 4) + 2) != 0)
            .map(|candidate_index| {
                candidate(
                    &format!("gpu-{candidate_index}"),
                    ((turn + candidate_index) % 6) as u64,
                )
                .with_vram((((turn + candidate_index) % 4) as u64 + 1) * 6 * GIB)
            })
            .rev()
            .collect();
        index.upsert_work(state_version, item);

        let changed_device = &mut devices[device_index];
        changed_device.available_vram_bytes = (((turn + device_index) % 4) as u64 + 1) * 6 * GIB;
        changed_device.health = if turn % 31 == 0 {
            DeviceHealth::Degraded
        } else {
            DeviceHealth::Healthy
        };
        let input = PlannerSnapshot::new(
            state_version,
            turn as u64 + 1,
            turn as u64 * 10,
            (((turn % 10) + 1) as u64) * 8 * GIB,
            devices.clone(),
            jobs.clone(),
        );

        assert_eq!(
            planner.plan_with_index(&input, &index),
            planner.plan(&input),
            "incremental index diverged after reducer turn {turn}"
        );
    }
}

#[test]
fn stale_eligibility_index_is_rejected_before_it_can_under_reserve() {
    let stale_input = PlannerSnapshot::new(
        6,
        10,
        1_000,
        8 * GIB,
        vec![device("gpu-0"), device("gpu-1")],
        vec![work("job", 0, vec![candidate("gpu-0", 1)])],
    );
    let stale_index = EligibilityIndex::from_snapshot(&stale_input);
    let current_input = snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![work("job", 0, vec![candidate("gpu-1", 8)])],
        8,
    );

    assert_eq!(
        Planner::default()
            .plan_with_index(&current_input, &stale_index)
            .expect_err("stale index must fail closed"),
        PlannerError::StaleEligibilityIndex {
            indexed_state_version: 6,
            snapshot_state_version: 7,
        }
    );
}

#[test]
fn same_version_changed_candidates_are_rejected_before_they_can_under_reserve() {
    let stale_input = snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![work("job", 0, vec![candidate("gpu-0", 1)])],
        8,
    );
    let stale_index = EligibilityIndex::from_snapshot(&stale_input);
    let current_input = snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![work("job", 0, vec![candidate("gpu-1", 8)])],
        8,
    );

    assert_eq!(
        Planner::default()
            .plan_with_index(&current_input, &stale_index)
            .expect_err("same-version candidate changes must fail closed"),
        PlannerError::EligibilityIndexCandidatesChanged {
            work_id: "job".into()
        }
    );
}

#[test]
fn eligibility_index_work_set_mismatches_are_typed() {
    let input = snapshot(
        vec![device("gpu-0")],
        vec![work("job", 0, vec![candidate("gpu-0", 1)])],
        8,
    );
    let missing = EligibilityIndex::new(input.state_version);
    assert_eq!(
        Planner::default()
            .plan_with_index(&input, &missing)
            .expect_err("missing work must fail closed"),
        PlannerError::EligibilityIndexMissingWork {
            work_id: "job".into()
        }
    );

    let mut unexpected = EligibilityIndex::from_snapshot(&input);
    unexpected.upsert_work(
        input.state_version,
        &work("other", 1, vec![candidate("gpu-0", 1)]),
    );
    assert_eq!(
        Planner::default()
            .plan_with_index(&input, &unexpected)
            .expect_err("unexpected work must fail closed"),
        PlannerError::EligibilityIndexUnexpectedWork {
            work_id: "other".into()
        }
    );
}

#[test]
fn duplicate_work_ids_are_rejected_before_leases_exist() {
    let duplicate = snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![
            work(
                "same",
                0,
                vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
            ),
            work(
                "same",
                1,
                vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
            ),
        ],
        8,
    );

    assert_eq!(
        Planner::default()
            .plan(&duplicate)
            .expect_err("duplicate work IDs must fail closed"),
        PlannerError::DuplicateWorkId {
            work_id: WorkId::from("same")
        }
    );
}

#[test]
fn duplicate_device_ids_are_rejected_before_leases_exist() {
    let duplicate = snapshot(
        vec![device("gpu-0"), device("gpu-0")],
        vec![work("job", 0, vec![candidate("gpu-0", 1)])],
        8,
    );

    assert_eq!(
        Planner::default()
            .plan(&duplicate)
            .expect_err("duplicate device IDs must fail closed"),
        PlannerError::DuplicateDeviceId {
            device_id: "gpu-0".into()
        }
    );
}

#[test]
fn optimizer_scores_future_makespan_for_every_candidate() {
    let fast =
        |device_id: &str| CandidatePlacement::new(device_id, "exec", 0).with_timing(0, 0, 100);
    let slow =
        |device_id: &str| CandidatePlacement::new(device_id, "exec", 0).with_timing(0, 0, 1_000);
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work("immediate-0", 0, vec![fast("gpu-0")]),
                work("immediate-1", 1, vec![fast("gpu-1")]),
                work("future", 2, vec![fast("gpu-0"), slow("gpu-1")]),
            ],
            8,
        ))
        .expect("valid plan");
    let future = plan
        .lanes
        .iter()
        .flat_map(|lane| &lane.assignments)
        .find(|assignment| assignment.work_id == WorkId::from("future"))
        .expect("future assignment");

    assert_eq!(future.device_id.as_str(), "gpu-0");
    assert_eq!(future.estimated_finish_ms, 1_200);
}

#[test]
fn heterogeneous_single_future_assignment_matches_exhaustive_finish_oracle() {
    for gpu_0_run_ms in [100, 500, 1_500] {
        for gpu_1_run_ms in [100, 500, 1_500] {
            let placement = |device_id: &str, predicted_run_ms| {
                CandidatePlacement::new(device_id, "exec", 0).with_timing(0, 0, predicted_run_ms)
            };
            let plan = Planner::default()
                .plan(&snapshot(
                    vec![device("gpu-0"), device("gpu-1")],
                    vec![
                        work("immediate-0", 0, vec![placement("gpu-0", 100)]),
                        work("immediate-1", 1, vec![placement("gpu-1", 300)]),
                        work(
                            "future",
                            2,
                            vec![
                                placement("gpu-0", gpu_0_run_ms),
                                placement("gpu-1", gpu_1_run_ms),
                            ],
                        ),
                    ],
                    8,
                ))
                .expect("valid plan");
            let future = plan
                .lanes
                .iter()
                .flat_map(|lane| &lane.assignments)
                .find(|assignment| assignment.work_id == WorkId::from("future"))
                .expect("future assignment");
            let expected = [
                (1_100 + gpu_0_run_ms, "gpu-0"),
                (1_300 + gpu_1_run_ms, "gpu-1"),
            ]
            .into_iter()
            .min()
            .expect("two candidates");

            assert_eq!(
                (future.estimated_finish_ms, future.device_id.as_str()),
                expected,
                "gpu-0 run {gpu_0_run_ms}, gpu-1 run {gpu_1_run_ms}"
            );
        }
    }
}

#[test]
fn every_younger_start_counts_toward_the_bypass_limit() {
    let waiting_candidate =
        |device_id: &str| CandidatePlacement::new(device_id, "exec", 0).with_timing(1_000, 0, 100);
    let plan = Planner::default()
        .plan(&snapshot(
            vec![
                device("cold-0"),
                device("cold-1"),
                device("cold-2"),
                device("cold-3"),
                DeviceSnapshot::busy("warm", 24 * GIB, 1_500).with_warm("exec"),
            ],
            vec![
                work(
                    "waiting",
                    0,
                    vec![
                        waiting_candidate("cold-0"),
                        waiting_candidate("cold-1"),
                        waiting_candidate("cold-2"),
                        waiting_candidate("cold-3"),
                        waiting_candidate("warm"),
                    ],
                )
                .with_warm_wait_started_at(1_000),
                work("younger-0", 1, vec![waiting_candidate("cold-0")]),
                work("younger-1", 2, vec![waiting_candidate("cold-1")]),
                work("younger-2", 3, vec![waiting_candidate("cold-2")]),
                work("younger-3", 4, vec![waiting_candidate("cold-3")]),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(plan.immediate_leases.len(), 4);
    assert_eq!(
        plan.bypass_updates
            .iter()
            .map(|update| (update.previous_count, update.new_count))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 2), (2, 3), (3, 3)]
    );
}

#[test]
fn ineligible_work_keeps_its_precise_reason_after_openings_fill() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0")],
            vec![
                work("first", 0, vec![candidate("gpu-0", 1)]),
                work("bad-pin", 1, vec![candidate("missing", 1)]).with_hard_device("missing"),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(
        plan.blocked_reason(&WorkId::from("bad-pin")),
        Some(&BlockedReason::HardPinUnavailable)
    );
}

#[test]
fn busy_only_work_reports_no_idle_device_after_other_openings_fill() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![
                device("gpu-0"),
                DeviceSnapshot::busy("gpu-1", 24 * GIB, 5_000),
            ],
            vec![
                work("first", 0, vec![candidate("gpu-0", 1)]),
                work("busy-only", 1, vec![candidate("gpu-1", 1)]),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(
        plan.blocked_reason(&WorkId::from("busy-only")),
        Some(&BlockedReason::NoIdleDevice)
    );
}

#[test]
fn idle_candidate_short_circuit_matches_the_previous_minimum_reference() {
    for mask in 0_u16..256 {
        let mut devices = vec![DeviceSnapshot::busy("gpu-0", 24 * GIB, 5_000)];
        for device_index in 1..4 {
            let mut candidate_device = device(&format!("gpu-{device_index}"));
            if mask & (1 << (device_index - 1)) != 0 {
                candidate_device.activity = mold_scheduler::DeviceActivity::Busy;
                candidate_device.available_at_ms = Some(5_000);
            }
            if mask & (1 << (device_index + 2)) != 0 {
                candidate_device.health = DeviceHealth::Degraded;
            }
            candidate_device.available_vram_bytes = if mask & (1 << (device_index + 5)) != 0 {
                4 * GIB
            } else {
                24 * GIB
            };
            devices.push(candidate_device);
        }

        let mut target_candidates = vec![candidate("gpu-0", 0)];
        for device_index in 1..4 {
            target_candidates.push(candidate(&format!("gpu-{device_index}"), 0).with_vram(8 * GIB));
        }
        target_candidates.rotate_left((mask as usize) % 4);

        let reference_has_idle = target_candidates
            .iter()
            .filter_map(|candidate| {
                let device = devices
                    .iter()
                    .find(|device| device.id == candidate.device_id)?;
                (device.is_idle() && candidate.predicted_vram_bytes <= device.available_vram_bytes)
                    .then_some(candidate.incremental_host_ram_bytes)
            })
            .min()
            .is_some();

        let mut jobs = devices
            .iter()
            .filter(|device| device.is_idle())
            .enumerate()
            .map(|(rank, device)| {
                work(
                    &format!("critical-filler-{rank}"),
                    rank as u64,
                    vec![candidate(device.id.as_str(), 0)],
                )
                .with_priority(PriorityClass::Critical)
                .with_hard_device(device.id.clone())
            })
            .collect::<Vec<_>>();
        jobs.push(work("target", 10_000, target_candidates));

        let plan = Planner::default()
            .plan(&snapshot(devices, jobs, 128))
            .expect("valid plan");
        assert_eq!(
            plan.blocked_reason(&WorkId::from("target")),
            Some(if reference_has_idle {
                &BlockedReason::LowerPriorityOpening
            } else {
                &BlockedReason::NoIdleDevice
            }),
            "mask {mask:08b}"
        );
    }
}

#[test]
fn runtime_grant_fences_are_typed_and_include_worker_readiness() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0")],
            vec![work("job", 0, vec![candidate("gpu-0", 1)])],
            8,
        ))
        .expect("valid plan");
    let lease = &plan.immediate_leases[0];
    let current = GrantValidationSnapshot {
        work_id: "job".into(),
        device_id: "gpu-0".into(),
        state_version: 7,
        plan_version: 11,
        sample_generation: 0,
        ledger_sequence: 0,
        work_ready: true,
        work_cancelled: false,
        worker_generation: lease.worker_generation,
        worker_ready: true,
        device_admin_state: DeviceAdminState::Enabled,
        device_health: DeviceHealth::Healthy,
    };

    assert_eq!(plan.validate_lease_for_grant(lease, &current), Ok(()));
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                worker_generation: lease.worker_generation + 1,
                ..current.clone()
            }
        ),
        Err(PlanValidationError::StaleWorkerGeneration {
            device_id: "gpu-0".into(),
            planned: lease.worker_generation,
            current: lease.worker_generation + 1,
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                worker_ready: false,
                ..current.clone()
            }
        ),
        Err(PlanValidationError::WorkerNotReady {
            device_id: "gpu-0".into()
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                work_cancelled: true,
                ..current.clone()
            }
        ),
        Err(PlanValidationError::WorkCancelled {
            work_id: "job".into()
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                work_ready: false,
                ..current.clone()
            }
        ),
        Err(PlanValidationError::WorkNotReady {
            work_id: "job".into()
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                device_admin_state: DeviceAdminState::Draining,
                ..current.clone()
            }
        ),
        Err(PlanValidationError::DeviceNotSchedulable {
            device_id: "gpu-0".into(),
            admin_state: DeviceAdminState::Draining,
            health: DeviceHealth::Healthy,
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                device_health: DeviceHealth::Unavailable,
                ..current.clone()
            }
        ),
        Err(PlanValidationError::DeviceNotSchedulable {
            device_id: "gpu-0".into(),
            admin_state: DeviceAdminState::Enabled,
            health: DeviceHealth::Unavailable,
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                work_id: "other-job".into(),
                ..current.clone()
            }
        ),
        Err(PlanValidationError::GrantWorkMismatch {
            planned: "job".into(),
            current: "other-job".into(),
        })
    );
    assert_eq!(
        plan.validate_lease_for_grant(
            lease,
            &GrantValidationSnapshot {
                device_id: "gpu-1".into(),
                ..current.clone()
            }
        ),
        Err(PlanValidationError::GrantDeviceMismatch {
            planned: "gpu-0".into(),
            current: "gpu-1".into(),
        })
    );
    let mut unproposed = lease.clone();
    unproposed.work_id = "other-job".into();
    assert_eq!(
        plan.validate_lease_for_grant(
            &unproposed,
            &GrantValidationSnapshot {
                work_id: "other-job".into(),
                ..current
            }
        ),
        Err(PlanValidationError::LeaseNotProposed {
            work_id: "other-job".into(),
            device_id: "gpu-0".into(),
        })
    );
}

#[test]
fn watchdog_returns_the_same_priority_cardinality_preserving_seed_every_time() {
    let input = snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![
            work(
                "flex",
                0,
                vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
            ),
            work("pin", 1, vec![candidate("gpu-0", 1), candidate("gpu-1", 1)])
                .with_hard_device("gpu-0"),
            work(
                "later",
                2,
                vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
            ),
        ],
        8,
    );
    let watchdog = Planner::new(PlannerConfig {
        mode: PlanningMode::WatchdogFallback,
        ..PlannerConfig::default()
    })
    .plan(&input)
    .expect("valid watchdog plan");
    let watchdog_again = Planner::new(PlannerConfig {
        mode: PlanningMode::WatchdogFallback,
        ..PlannerConfig::default()
    })
    .plan(&input)
    .expect("valid watchdog plan");

    assert_eq!(watchdog, watchdog_again);
    assert_eq!(watchdog.optimizer_state, OptimizerState::WatchdogFallback);
    assert_eq!(watchdog.immediate_leases.len(), 2);
    assert_eq!(watchdog.operations_evaluated, 0);
    assert_eq!(
        assignments(&watchdog),
        BTreeMap::from([
            ("flex".into(), "gpu-1".into()),
            ("pin".into(), "gpu-0".into()),
        ])
    );
    let optimized = Planner::default()
        .plan(&input)
        .expect("valid optimized plan");
    assert_eq!(watchdog.immediate_leases, optimized.immediate_leases);
    assert_eq!(watchdog.reservation, optimized.reservation);

    let mut permuted = input.clone();
    permuted.devices.reverse();
    permuted.work.reverse();
    let permuted_watchdog = Planner::new(PlannerConfig {
        mode: PlanningMode::WatchdogFallback,
        ..PlannerConfig::default()
    })
    .plan(&permuted)
    .expect("valid permuted watchdog plan");
    assert_eq!(watchdog, permuted_watchdog);
}

#[test]
fn uniform_seed_proof_does_not_report_unperformed_operations() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0"), device("gpu-1")],
            vec![
                work(
                    "work-0",
                    0,
                    vec![candidate("gpu-0", 0), candidate("gpu-1", 0)],
                ),
                work(
                    "work-1",
                    1,
                    vec![candidate("gpu-0", 0), candidate("gpu-1", 0)],
                ),
                work(
                    "work-2",
                    2,
                    vec![candidate("gpu-0", 0), candidate("gpu-1", 0)],
                ),
            ],
            8,
        ))
        .expect("valid plan");

    assert_eq!(plan.optimizer_state, OptimizerState::SeedOnly);
    assert_eq!(plan.operations_evaluated, 0);
}

#[test]
fn horizon_and_operation_budget_follow_the_locked_formula() {
    assert_eq!(optimization_horizon(10_000, 0), 64);
    assert_eq!(optimization_horizon(10_000, 8), 64);
    assert_eq!(optimization_horizon(10_000, 16), 128);
    assert_eq!(optimization_horizon(10_000, 64), 200);
    assert_eq!(operation_budget(0, 0), 512);
    assert_eq!(operation_budget(8, 64), 768);
    assert_eq!(operation_budget(64, 200), 4_896);
    assert_eq!(operation_budget(1_000, 200), 8_192);
}

#[test]
fn priority_class_precedes_manual_rank_and_stable_id_breaks_ties() {
    let plan = Planner::default()
        .plan(&snapshot(
            vec![device("gpu-0")],
            vec![
                work("user-z", 0, vec![candidate("gpu-0", 1)]),
                work("critical-b", 99, vec![candidate("gpu-0", 1)])
                    .with_priority(PriorityClass::Critical),
                work("critical-a", 99, vec![candidate("gpu-0", 1)])
                    .with_priority(PriorityClass::Critical),
            ],
            8,
        ))
        .expect("valid plan");
    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "critical-a");
}

#[test]
fn no_fixed_size_device_assumption_in_eligibility_index() {
    let mut index = EligibilityIndex::new(7);
    let candidates = (0..257)
        .map(|index| candidate(&format!("gpu-{index:03}"), 0))
        .collect::<Vec<_>>();
    index.upsert_work(7, &work("wide", 0, candidates));
    assert_eq!(
        index
            .candidates_for(&WorkId::from("wide"))
            .expect("indexed work")
            .len(),
        257
    );
}

#[test]
fn exhaustive_small_oracle_matches_priority_admission_cardinality() {
    for mask in 0_u16..512 {
        let devices = vec![device("gpu-0"), device("gpu-1"), device("gpu-2")];
        let jobs = (0..3)
            .map(|work_index| {
                let candidates = (0..3)
                    .filter(|device_index| {
                        let bit = work_index * 3 + device_index;
                        mask & (1_u16 << bit) != 0
                    })
                    .map(|device_index| candidate(&format!("gpu-{device_index}"), 0))
                    .collect();
                work(&format!("work-{work_index}"), work_index, candidates)
            })
            .collect::<Vec<_>>();
        let expected = exhaustive_priority_cardinality(&jobs, 3);
        let plan = Planner::default()
            .plan(&snapshot(devices, jobs, 64))
            .expect("valid plan");
        assert_eq!(plan.immediate_leases.len(), expected, "mask {mask:09b}");
    }
}

fn exhaustive_priority_cardinality(jobs: &[WorkSnapshot], device_count: usize) -> usize {
    fn can_match(
        jobs: &[WorkSnapshot],
        selected: &[usize],
        cursor: usize,
        used: &mut BTreeSet<String>,
    ) -> bool {
        if cursor == selected.len() {
            return true;
        }
        jobs[selected[cursor]]
            .candidate_placements
            .iter()
            .any(|candidate| {
                let id = candidate.device_id.as_str().to_owned();
                if used.insert(id.clone()) {
                    let matched = can_match(jobs, selected, cursor + 1, used);
                    used.remove(&id);
                    matched
                } else {
                    false
                }
            })
    }

    let mut selected = Vec::new();
    for index in 0..jobs.len() {
        let mut trial = selected.clone();
        trial.push(index);
        if trial.len() <= device_count && can_match(jobs, &trial, 0, &mut BTreeSet::new()) {
            selected = trial;
        }
    }
    selected.len()
}

fn benchmark_input(device_count: usize, ready_count: usize) -> PlannerSnapshot {
    let devices = (0..device_count)
        .map(|index| device(&format!("gpu-{index:03}")))
        .collect::<Vec<_>>();
    let jobs = (0..ready_count)
        .map(|index| {
            let candidates = (0..device_count)
                .map(|device_index| candidate(&format!("gpu-{device_index:03}"), 0))
                .collect();
            work(&format!("work-{index:05}"), index as u64, candidates)
        })
        .collect::<Vec<_>>();
    snapshot(devices, jobs, 128)
}

fn percentile_95(samples: &mut [std::time::Duration]) -> std::time::Duration {
    samples.sort_unstable();
    samples[(samples.len() * 95).div_ceil(100).saturating_sub(1)]
}

#[test]
#[ignore = "local immediate-dispatch benchmark; run on pinned hardware, never a CI timing gate"]
fn local_immediate_dispatch_benchmark_harness() {
    for (device_count, ready_count) in [(8, 200), (64, 200), (8, 10_000), (64, 10_000)] {
        let input = benchmark_input(device_count, ready_count);
        let index = EligibilityIndex::from_snapshot(&input);
        let planner = Planner::new(PlannerConfig {
            mode: PlanningMode::WatchdogFallback,
            ..PlannerConfig::default()
        });
        let expected = planner
            .plan_with_index(&input, &index)
            .expect("valid warmup plan");
        let mut samples = Vec::with_capacity(20);
        for _ in 0..20 {
            let started = Instant::now();
            let measured = planner
                .plan_with_index(&input, &index)
                .expect("valid measured plan");
            samples.push(started.elapsed());
            assert_eq!(measured, expected);
        }
        eprintln!(
            "{ready_count} ready / {device_count} devices immediate p95: {:?}",
            percentile_95(&mut samples)
        );
    }

    let devices = (0..64)
        .map(|index| device(&format!("gpu-{index:03}")))
        .collect::<Vec<_>>();
    let jobs = (0..10_000)
        .map(|index| {
            let candidates = (0..64)
                .map(|device_index| candidate(&format!("gpu-{device_index:03}"), 1))
                .collect();
            work(&format!("ram-blocked-{index:05}"), index as u64, candidates)
        })
        .collect::<Vec<_>>();
    let input = snapshot(devices, jobs, 0);
    let index = EligibilityIndex::from_snapshot(&input);
    let plan = Planner::new(PlannerConfig {
        mode: PlanningMode::WatchdogFallback,
        ..PlannerConfig::default()
    })
    .plan_with_index(&input, &index)
    .expect("valid host-RAM-blocked plan");
    assert!(plan.immediate_leases.is_empty());
    assert_eq!(plan.blocked.len(), 10_000);
    assert!(plan
        .blocked
        .iter()
        .all(|blocked| blocked.reason == BlockedReason::AggregateHostRamReserved));
}

#[test]
#[ignore = "local pure-planner benchmark; run on pinned hardware, never a CI timing gate"]
fn local_pure_optimizer_benchmark_harness() {
    for (device_count, ready_count) in [(8, 200), (64, 200)] {
        let input = benchmark_input(device_count, ready_count);
        let index = EligibilityIndex::from_snapshot(&input);
        let planner = Planner::default();
        let expected = planner
            .plan_with_index(&input, &index)
            .expect("valid warmup plan");
        let mut samples = Vec::with_capacity(20);
        for _ in 0..20 {
            let started = Instant::now();
            let measured = planner
                .plan_with_index(&input, &index)
                .expect("valid measured plan");
            samples.push(started.elapsed());
            assert_eq!(measured, expected);
        }
        eprintln!(
            "{ready_count} ready / {device_count} devices optimizer p95: {:?}",
            percentile_95(&mut samples)
        );
    }
}
