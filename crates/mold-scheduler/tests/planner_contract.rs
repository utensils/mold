use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use mold_scheduler::{
    operation_budget, optimization_horizon, BlockedReason, CandidatePlacement, DeviceSnapshot,
    EligibilityIndex, OptimizerState, PlanValidationError, Planner, PlannerConfig, PlannerSnapshot,
    PlanningMode, PriorityClass, WorkId, WorkSnapshot,
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

        let plan = Planner::default().plan(&snapshot(devices, jobs, 128));
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
    let plan = Planner::default().plan(&snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![
            work(
                "older-flexible",
                0,
                vec![candidate("gpu-0", 1), candidate("gpu-1", 1)],
            ),
            work("younger-specialist", 1, vec![candidate("gpu-0", 1)]),
        ],
        8,
    ));

    assert_eq!(
        assignments(&plan),
        BTreeMap::from([
            ("older-flexible".into(), "gpu-1".into()),
            ("younger-specialist".into(), "gpu-0".into()),
        ])
    );
}

#[test]
fn rematching_selects_the_minimum_aggregate_host_ram_full_matching() {
    let plan = Planner::default().plan(&snapshot(
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
    ));

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

    let plan = Planner::default().plan(&snapshot(vec![device("gpu-0")], jobs, 8));
    assert_eq!(
        plan.immediate_leases[0].work_id.as_str(),
        "rank-250-compatible"
    );
    assert!(plan.optimization_horizon_length <= 200);
}

#[test]
fn aggregate_host_ram_rejects_two_individually_admissible_jobs() {
    let plan = Planner::default().plan(&snapshot(
        vec![device("gpu-0"), device("gpu-1")],
        vec![
            work("older-8", 0, vec![candidate("gpu-0", 8)]),
            work("younger-8", 1, vec![candidate("gpu-1", 8)]),
        ],
        12,
    ));

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
    let plan = Planner::default().plan(&snapshot(
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
        8,
    ));

    assert_eq!(plan.immediate_leases.len(), 1);
    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "older-8");
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

    let held = Planner::new(config.clone()).plan(&snapshot(
        vec![cold.clone(), warm.clone()],
        vec![waiting.clone()],
        8,
    ));
    assert!(held.immediate_leases.is_empty());
    assert_eq!(held.warm_waits[0].deadline_ms, 1_500);
    assert_eq!(
        held.blocked_reason(&WorkId::from("job")),
        Some(&BlockedReason::WarmWait)
    );

    let expired_snapshot =
        PlannerSnapshot::new(8, 12, 1_500, 8 * GIB, vec![cold, warm], vec![waiting]);
    let expired = Planner::new(config).plan(&expired_snapshot);
    assert_eq!(expired.immediate_leases[0].device_id.as_str(), "cold");
}

#[test]
fn warm_wait_never_holds_when_cold_now_finishes_first() {
    let plan = Planner::default().plan(&snapshot(
        vec![
            device("cold"),
            DeviceSnapshot::busy("warm", 24 * GIB, 10_000).with_warm("exec"),
        ],
        vec![
            work("job", 0, vec![candidate("cold", 1), candidate("warm", 1)])
                .with_warm_wait_started_at(1_000),
        ],
        8,
    ));

    assert_eq!(plan.immediate_leases[0].device_id.as_str(), "cold");
    assert!(plan.warm_waits.is_empty());
}

#[test]
fn third_bypass_forces_the_next_compatible_opening() {
    let warm = DeviceSnapshot::busy("warm", 24 * GIB, 1_500).with_warm("exec");
    let plan = Planner::default().plan(&snapshot(
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
    ));

    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "forced");
    assert!(plan.warm_waits.is_empty());
}

#[test]
fn a_younger_start_on_a_declined_warm_wait_edge_increments_bypass_once() {
    let plan = Planner::default().plan(&snapshot(
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
    ));

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
    let plan = Planner::default().plan(&input);

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

    let mut index_a = EligibilityIndex::new();
    for job in &jobs {
        index_a.upsert_work(job);
    }
    let mut index_b = EligibilityIndex::new();
    for job in jobs.iter().rev() {
        index_b.upsert_work(job);
    }

    let planner = Planner::default();
    assert_eq!(planner.plan(&forward), planner.plan(&reverse));
    assert_eq!(
        planner.plan_with_index(&forward, &index_a),
        planner.plan_with_index(&forward, &index_b)
    );

    index_b.remove_work(&WorkId::from("work-a"));
    index_b.upsert_work(&jobs[1]);
    assert_eq!(
        planner.plan_with_index(&forward, &index_a),
        planner.plan_with_index(&forward, &index_b)
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
            work("pin", 1, vec![candidate("gpu-0", 1)]),
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
    .plan(&input);
    let watchdog_again = Planner::new(PlannerConfig {
        mode: PlanningMode::WatchdogFallback,
        ..PlannerConfig::default()
    })
    .plan(&input);

    assert_eq!(watchdog, watchdog_again);
    assert_eq!(watchdog.optimizer_state, OptimizerState::WatchdogFallback);
    assert_eq!(watchdog.immediate_leases.len(), 2);
    assert_eq!(watchdog.operations_evaluated, 0);
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
    let plan = Planner::default().plan(&snapshot(
        vec![device("gpu-0")],
        vec![
            work("user-z", 0, vec![candidate("gpu-0", 1)]),
            work("critical-b", 99, vec![candidate("gpu-0", 1)])
                .with_priority(PriorityClass::Critical),
            work("critical-a", 99, vec![candidate("gpu-0", 1)])
                .with_priority(PriorityClass::Critical),
        ],
        8,
    ));
    assert_eq!(plan.immediate_leases[0].work_id.as_str(), "critical-a");
}

#[test]
fn no_fixed_size_device_assumption_in_eligibility_index() {
    let mut index = EligibilityIndex::new();
    let candidates = (0..257)
        .map(|index| candidate(&format!("gpu-{index:03}"), 0))
        .collect::<Vec<_>>();
    index.upsert_work(&work("wide", 0, candidates));
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
        let plan = Planner::default().plan(&snapshot(devices, jobs, 64));
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

#[test]
#[ignore = "deterministic large-queue harness; latency is measured on pinned hardware"]
fn deterministic_two_hundred_and_ten_thousand_ready_harness() {
    for (device_count, ready_count) in [(8, 200), (64, 200), (8, 10_000), (64, 10_000)] {
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
        let input = snapshot(devices, jobs, 128);
        let index = EligibilityIndex::from_work(&input.work);
        let planner = Planner::new(PlannerConfig {
            mode: PlanningMode::WatchdogFallback,
            ..PlannerConfig::default()
        });
        let started = Instant::now();
        let first = planner.plan_with_index(&input, &index);
        let elapsed = started.elapsed();
        let second = planner.plan_with_index(&input, &index);
        assert_eq!(first, second);
        eprintln!("{ready_count} ready / {device_count} devices immediate seed: {elapsed:?}");
        if ready_count == 200 {
            let optimized_started = Instant::now();
            let optimized = Planner::default().plan_with_index(&input, &index);
            let optimized_elapsed = optimized_started.elapsed();
            assert_eq!(
                optimized,
                Planner::default().plan_with_index(&input, &index)
            );
            eprintln!(
                "{ready_count} ready / {device_count} devices optimized: {optimized_elapsed:?}"
            );
        }
    }
}
