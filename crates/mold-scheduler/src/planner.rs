use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use crate::eligibility::EligibilityIndex;
use crate::matching::{IncrementalMatcher, MatchCandidate, MatchWork, MatchingResult};
use crate::{
    AssignmentReason, BlockedReason, BlockedWork, BypassUpdate, CandidatePlacement, DeviceActivity,
    DeviceId, DeviceLane, DeviceSnapshot, ImmediateLease, MatchingReservation, OptimizerState,
    Plan, PlannedAssignment, PlannerConfig, PlannerError, PlannerSnapshot, PlanningMode,
    ReservationItem, WarmWait, WorkId, WorkKind, WorkSnapshot,
};

#[derive(Clone, Debug)]
pub struct Planner {
    config: PlannerConfig,
}

impl Default for Planner {
    fn default() -> Self {
        Self::new(PlannerConfig::default())
    }
}

impl Planner {
    pub fn new(config: PlannerConfig) -> Self {
        Self { config }
    }

    pub fn plan(&self, snapshot: &PlannerSnapshot) -> Result<Plan, PlannerError> {
        validate_snapshot_ids(snapshot)?;
        let index = EligibilityIndex::from_snapshot(snapshot);
        self.plan_with_index(snapshot, &index)
    }

    pub fn plan_with_index(
        &self,
        snapshot: &PlannerSnapshot,
        index: &EligibilityIndex,
    ) -> Result<Plan, PlannerError> {
        validate_snapshot_ids(snapshot)?;
        validate_eligibility_index(snapshot, index)?;
        if snapshot.queue_paused {
            let mut work = snapshot.work.iter().collect::<Vec<_>>();
            work.sort_by(|left, right| work_order(left).cmp(&work_order(right)));
            return Ok(Plan {
                plan_version: snapshot.next_plan_version,
                state_version: snapshot.state_version,
                created_at_ms: snapshot.now_ms,
                next_replan_at_ms: snapshot.next_replan_at_ms,
                immediate_leases: Vec::new(),
                lanes: Vec::new(),
                blocked: work
                    .into_iter()
                    .map(|work| BlockedWork {
                        work_id: work.id.clone(),
                        reason: BlockedReason::QueuePaused,
                    })
                    .collect(),
                warm_waits: Vec::new(),
                bypass_updates: Vec::new(),
                reservation: MatchingReservation {
                    sample_generation: snapshot.host_memory.sample_generation,
                    ledger_sequence: snapshot.host_memory.ledger_sequence,
                    total_host_ram_bytes: 0,
                    items: Vec::new(),
                },
                optimization_horizon_length: 0,
                operation_budget: 0,
                operations_evaluated: 0,
                optimizer_state: optimizer_state_for_mode(self.config.mode),
            });
        }
        let devices = canonical_devices(&snapshot.devices);
        let device_map = devices
            .iter()
            .map(|device| (device.id.clone(), device))
            .collect::<BTreeMap<_, _>>();
        // A device that only `accepts_releasing_work` still needs a slot in the
        // matcher, or an unload queued against a degraded GPU has nowhere to be
        // matched and comes back `LowerPriorityOpening`. Ordinary work cannot
        // reach those devices: `candidate_is_eligible` keeps gating it on
        // `is_schedulable`, and `try_with` builds edges only from a work item's
        // own candidates, so an extra slot is an isolated node that cannot
        // change the matching.
        //
        // The filter is derived from `is_idle_for` rather than restating its
        // rule, because `immediate_candidates` is selected with the same
        // predicate and one candidate outside this slot set blocks its whole
        // work item. The two must not drift.
        let kinds_present = snapshot
            .work
            .iter()
            .map(|item| item.kind)
            .collect::<BTreeSet<_>>();
        let idle_device_ids = devices
            .iter()
            .filter(|device| kinds_present.iter().any(|kind| device.is_idle_for(*kind)))
            .map(|device| device.id.clone())
            .collect::<Vec<_>>();

        let mut work = snapshot.work.iter().collect::<Vec<_>>();
        work.sort_by(|left, right| work_order(left).cmp(&work_order(right)));

        let mut prepared = BTreeMap::<WorkId, PreparedWork>::new();
        let mut blocked = work
            .iter()
            .filter(|item| !item.ready || item.ready_at_ms > snapshot.now_ms)
            .map(|item| (item.id.clone(), BlockedReason::NotReady))
            .collect::<BTreeMap<_, _>>();
        let mut matcher = IncrementalMatcher::new(&idle_device_ids);
        let mut accepted_ids = BTreeSet::<WorkId>::new();
        let mut final_matching = MatchingResult {
            assignments: BTreeMap::new(),
            total_host_ram_bytes: 0,
        };

        for item in &work {
            if !item.ready
                || item.ready_at_ms > snapshot.now_ms
                || accepted_ids.len() == idle_device_ids.len()
            {
                continue;
            }
            let candidates = index.candidates_for(&item.id).unwrap_or_default();
            // `final_matching` is the minimum-host-RAM matching for the
            // accepted set. Any feasible augmenting path can only keep or
            // increase that set's cost, so adding the new work's cheapest idle
            // edge is a sound lower bound. Keep host RAM first in `Cost`.
            if let Some(minimum_host_ram) =
                minimum_immediate_host_ram(item, candidates, &device_map)
            {
                if minimum_host_ram > snapshot.host_memory.headroom_bytes {
                    blocked.insert(item.id.clone(), BlockedReason::InsufficientHostRam);
                    continue;
                }
                if final_matching
                    .total_host_ram_bytes
                    .checked_add(minimum_host_ram)
                    .is_none_or(|minimum_total| minimum_total > snapshot.host_memory.headroom_bytes)
                {
                    blocked.insert(item.id.clone(), BlockedReason::AggregateHostRamReserved);
                    continue;
                }
            }
            let prepared_work =
                prepare_work(item, candidates, &device_map, snapshot.now_ms, &self.config);
            if prepared_work.all_candidates.is_empty() {
                blocked.insert(
                    item.id.clone(),
                    classify_no_candidate(item, candidates, &device_map),
                );
            } else if prepared_work.immediate_candidates.is_empty() {
                blocked.insert(
                    item.id.clone(),
                    if prepared_work.warm_wait.is_some() {
                        BlockedReason::WarmWait
                    } else {
                        BlockedReason::NoIdleDevice
                    },
                );
            }
            prepared.insert(item.id.clone(), prepared_work);
            let Some(prepared_work) = prepared.get(&item.id) else {
                continue;
            };
            if prepared_work.immediate_candidates.is_empty() {
                continue;
            }

            let trial_work = MatchWork {
                work_id: item.id.clone(),
                candidates: prepared_work.immediate_candidates.clone(),
            };
            let Some((trial_matcher, matching)) = matcher.try_with(trial_work) else {
                blocked.insert(item.id.clone(), BlockedReason::LowerPriorityOpening);
                continue;
            };
            if matching.total_host_ram_bytes > snapshot.host_memory.headroom_bytes {
                blocked.insert(item.id.clone(), BlockedReason::AggregateHostRamReserved);
                continue;
            }

            matcher = trial_matcher;
            final_matching = matching;
            accepted_ids.insert(item.id.clone());
            blocked.remove(&item.id);
        }

        let ready_count = work.iter().filter(|item| item.ready).count();
        let schedulable_count = devices
            .iter()
            .filter(|device| device.is_schedulable())
            .count();
        let base_horizon = optimization_horizon(ready_count, schedulable_count);
        let horizon_work = select_horizon(&work, base_horizon);
        for item in &horizon_work {
            if !prepared.contains_key(&item.id) {
                let candidates = index.candidates_for(&item.id).unwrap_or_default();
                prepared.insert(
                    item.id.clone(),
                    prepare_work(item, candidates, &device_map, snapshot.now_ms, &self.config),
                );
            }
        }

        for item in &work {
            if !item.ready || accepted_ids.contains(&item.id) || blocked.contains_key(&item.id) {
                continue;
            }
            let candidates = index.candidates_for(&item.id).unwrap_or_default();
            let reason = if !has_schedulable_candidate(item, candidates, &device_map) {
                classify_no_candidate(item, candidates, &device_map)
            } else if !has_idle_candidate(item, candidates, &device_map) {
                BlockedReason::NoIdleDevice
            } else {
                BlockedReason::LowerPriorityOpening
            };
            blocked.insert(item.id.clone(), reason);
        }

        let immediate_leases =
            build_immediate_leases(&work, &final_matching, &device_map, snapshot.now_ms);
        let reservation = build_reservation(snapshot, &immediate_leases);
        let warm_wait_ids = blocked
            .iter()
            .filter(|(_, reason)| **reason == BlockedReason::WarmWait)
            .map(|(work_id, _)| work_id.clone())
            .collect::<BTreeSet<_>>();
        let warm_waits = work
            .iter()
            .filter(|item| !accepted_ids.contains(&item.id))
            .filter(|item| warm_wait_ids.contains(&item.id))
            .filter_map(|item| prepared.get(&item.id)?.warm_wait.clone())
            .collect::<Vec<_>>();
        let bypass_updates = build_bypass_updates(
            &work,
            &immediate_leases,
            &prepared,
            snapshot.host_memory.headroom_bytes,
            reservation.total_host_ram_bytes,
        );

        let budget = operation_budget(schedulable_count, horizon_work.len());
        let (lanes, operations_evaluated, optimizer_state) = build_lanes(
            snapshot,
            &devices,
            &horizon_work,
            &prepared,
            &immediate_leases,
            budget,
            self.config.mode,
        );

        let work_positions = work
            .iter()
            .enumerate()
            .map(|(position, item)| (item.id.clone(), position))
            .collect::<BTreeMap<_, _>>();
        let mut blocked = blocked
            .into_iter()
            .map(|(work_id, reason)| BlockedWork { work_id, reason })
            .collect::<Vec<_>>();
        blocked.sort_by(|left, right| {
            work_positions
                .get(&left.work_id)
                .cmp(&work_positions.get(&right.work_id))
                .then_with(|| left.work_id.cmp(&right.work_id))
        });

        Ok(Plan {
            plan_version: snapshot.next_plan_version,
            state_version: snapshot.state_version,
            created_at_ms: snapshot.now_ms,
            next_replan_at_ms: snapshot.next_replan_at_ms,
            immediate_leases,
            lanes,
            blocked,
            warm_waits,
            bypass_updates,
            reservation,
            optimization_horizon_length: horizon_work.len(),
            operation_budget: budget,
            operations_evaluated,
            optimizer_state,
        })
    }
}

fn validate_snapshot_ids(snapshot: &PlannerSnapshot) -> Result<(), PlannerError> {
    let mut device_ids = BTreeSet::new();
    for device in &snapshot.devices {
        if !device_ids.insert(device.id.clone()) {
            return Err(PlannerError::DuplicateDeviceId {
                device_id: device.id.clone(),
            });
        }
    }

    let mut work_ids = BTreeSet::new();
    for work in &snapshot.work {
        if !work_ids.insert(work.id.clone()) {
            return Err(PlannerError::DuplicateWorkId {
                work_id: work.id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_eligibility_index(
    snapshot: &PlannerSnapshot,
    index: &EligibilityIndex,
) -> Result<(), PlannerError> {
    if index.state_version() != snapshot.state_version {
        return Err(PlannerError::StaleEligibilityIndex {
            indexed_state_version: index.state_version(),
            snapshot_state_version: snapshot.state_version,
        });
    }

    let snapshot_ids = snapshot
        .work
        .iter()
        .map(|work| work.id.clone())
        .collect::<BTreeSet<_>>();
    for work in &snapshot.work {
        if index.candidates_for(&work.id).is_none() {
            return Err(PlannerError::EligibilityIndexMissingWork {
                work_id: work.id.clone(),
            });
        }
        if !index.candidates_match(work) {
            return Err(PlannerError::EligibilityIndexCandidatesChanged {
                work_id: work.id.clone(),
            });
        }
    }
    for work_id in index.work_ids() {
        if !snapshot_ids.contains(work_id) {
            return Err(PlannerError::EligibilityIndexUnexpectedWork {
                work_id: work_id.clone(),
            });
        }
    }
    Ok(())
}

pub fn optimization_horizon(ready_work_count: usize, schedulable_device_count: usize) -> usize {
    ready_work_count.min(schedulable_device_count.saturating_mul(8).clamp(64, 200))
}

pub fn operation_budget(
    schedulable_device_count: usize,
    optimization_horizon_length: usize,
) -> usize {
    (64 * schedulable_device_count + 4 * optimization_horizon_length).clamp(512, 8_192)
}

#[derive(Clone, Debug)]
struct PreparedWork {
    all_candidates: Vec<MatchCandidate>,
    immediate_candidates: Vec<MatchCandidate>,
    warm_wait: Option<WarmWait>,
}

fn prepare_work(
    work: &WorkSnapshot,
    candidates: &[CandidatePlacement],
    devices: &BTreeMap<DeviceId, &DeviceSnapshot>,
    now_ms: u64,
    config: &PlannerConfig,
) -> PreparedWork {
    let mut all_candidates = candidates
        .iter()
        .filter_map(|candidate| {
            let device = devices.get(&candidate.device_id)?;
            if !candidate_is_eligible(work, candidate, device, false) {
                return None;
            }
            Some(MatchCandidate {
                setup_ms: setup_ms(candidate, device),
                placement: normalized_placement(work.kind, candidate),
            })
        })
        .collect::<Vec<_>>();
    sort_match_candidates(&mut all_candidates);

    let mut immediate_candidates = all_candidates
        .iter()
        .filter(|candidate| {
            work.ready_at_ms <= now_ms
                && devices
                    .get(&candidate.placement.device_id)
                    .is_some_and(|device| device.is_idle_for(work.kind))
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut declined_candidates = Vec::new();
    let mut warm_wait = None;

    if work.bypass_count < 3 && config.warm_wait_max_ms > 0 {
        let warm_options = all_candidates
            .iter()
            .filter_map(|candidate| {
                let device = devices.get(&candidate.placement.device_id)?;
                if !device
                    .warm_execution_fingerprints
                    .contains(&candidate.placement.execution_fingerprint)
                {
                    return None;
                }
                let ready_at = match device.activity {
                    DeviceActivity::Idle => now_ms,
                    DeviceActivity::Busy => device.available_at_ms?,
                }
                .max(work.ready_at_ms);
                let finish = ready_at
                    .saturating_add(candidate.placement.warm_setup_ms)
                    .saturating_add(candidate.placement.predicted_run_ms);
                Some((finish, ready_at, candidate))
            })
            .collect::<Vec<_>>();

        if let Some((warm_finish, warm_ready_at, warm_candidate)) =
            warm_options.into_iter().min_by(|left, right| {
                (left.0, &left.2.placement.device_id).cmp(&(right.0, &right.2.placement.device_id))
            })
        {
            let started_at = work.warm_wait_started_at_ms.unwrap_or(now_ms);
            let deadline = started_at
                .saturating_add(config.warm_wait_max_ms)
                .min(warm_ready_at);
            let best_cold_finish = immediate_candidates
                .iter()
                .filter(|candidate| {
                    !devices
                        .get(&candidate.placement.device_id)
                        .is_some_and(|device| {
                            device
                                .warm_execution_fingerprints
                                .contains(&candidate.placement.execution_fingerprint)
                        })
                })
                .map(|candidate| {
                    now_ms
                        .saturating_add(candidate.placement.cold_setup_ms)
                        .saturating_add(candidate.placement.predicted_run_ms)
                })
                .min();

            if let Some(best_cold_finish) = best_cold_finish {
                if now_ms < deadline && warm_finish < best_cold_finish {
                    let mut retained = Vec::new();
                    for candidate in immediate_candidates {
                        let is_warm =
                            devices
                                .get(&candidate.placement.device_id)
                                .is_some_and(|device| {
                                    device
                                        .warm_execution_fingerprints
                                        .contains(&candidate.placement.execution_fingerprint)
                                });
                        if !is_warm {
                            declined_candidates.push(candidate);
                        } else {
                            retained.push(candidate);
                        }
                    }
                    immediate_candidates = retained;
                    if !declined_candidates.is_empty() {
                        warm_wait = Some(WarmWait {
                            work_id: work.id.clone(),
                            warm_device_id: warm_candidate.placement.device_id.clone(),
                            started_at_ms: started_at,
                            deadline_ms: deadline,
                            predicted_warm_finish_ms: warm_finish,
                            best_cold_finish_ms: best_cold_finish,
                            declined_device_ids: declined_candidates
                                .iter()
                                .map(|candidate| candidate.placement.device_id.clone())
                                .collect(),
                        });
                    }
                }
            }
        }
    }

    sort_match_candidates(&mut immediate_candidates);
    PreparedWork {
        all_candidates,
        immediate_candidates,
        warm_wait,
    }
}

fn classify_no_candidate(
    work: &WorkSnapshot,
    candidates: &[CandidatePlacement],
    devices: &BTreeMap<DeviceId, &DeviceSnapshot>,
) -> BlockedReason {
    if let Some(hard_device_id) = &work.hard_device_id {
        let Some(device) = devices.get(hard_device_id) else {
            return BlockedReason::HardPinUnavailable;
        };
        // Releasing work is admitted on a degraded, disabled or draining
        // device, so those states must not be reported as its blocker either —
        // only quarantine is, and that is checked below.
        if !work.kind.releases_resources() {
            match device.admin_state {
                crate::DeviceAdminState::Disabled => return BlockedReason::DeviceDisabled,
                crate::DeviceAdminState::Draining => return BlockedReason::DeviceDraining,
                crate::DeviceAdminState::StartupExcluded => {
                    return BlockedReason::DeviceStartupExcluded;
                }
                crate::DeviceAdminState::Enabled => {}
            }
            if device.health == crate::DeviceHealth::Degraded {
                return BlockedReason::DeviceDegraded;
            }
        }
        match device.health {
            crate::DeviceHealth::Degraded => {}
            crate::DeviceHealth::Unavailable | crate::DeviceHealth::Poisoned => {
                return BlockedReason::DeviceUnavailable;
            }
            crate::DeviceHealth::Healthy => {}
        }
    }
    if let Some(backend) = work.backend_requirement {
        if !devices
            .values()
            .any(|device| device.is_schedulable() && device.backend == backend)
        {
            return BlockedReason::BackendUnsupported;
        }
    }
    let vram_blocked = !work.kind.releases_resources()
        && candidates.iter().any(|candidate| {
            devices.get(&candidate.device_id).is_some_and(|device| {
                device.is_schedulable()
                    && candidate.predicted_vram_bytes
                        > candidate_available_vram_bytes(candidate, device)
            })
        });
    if vram_blocked {
        BlockedReason::InsufficientVram
    } else {
        BlockedReason::NoSchedulableDevice
    }
}

fn has_schedulable_candidate(
    work: &WorkSnapshot,
    candidates: &[CandidatePlacement],
    devices: &BTreeMap<DeviceId, &DeviceSnapshot>,
) -> bool {
    candidates.iter().any(|candidate| {
        devices
            .get(&candidate.device_id)
            .is_some_and(|device| candidate_is_eligible(work, candidate, device, false))
    })
}

fn has_idle_candidate(
    work: &WorkSnapshot,
    candidates: &[CandidatePlacement],
    devices: &BTreeMap<DeviceId, &DeviceSnapshot>,
) -> bool {
    candidates.iter().any(|candidate| {
        devices
            .get(&candidate.device_id)
            .is_some_and(|device| candidate_is_eligible(work, candidate, device, true))
    })
}

/// Returns a lower bound for adding `work` to the current minimum-host-RAM
/// matching. This is sound only while the matcher keeps host RAM as its first
/// lexicographic cost component.
fn minimum_immediate_host_ram(
    work: &WorkSnapshot,
    candidates: &[CandidatePlacement],
    devices: &BTreeMap<DeviceId, &DeviceSnapshot>,
) -> Option<u64> {
    // Work that releases resources is never charged for them. `None` skips both
    // the per-item and the aggregate headroom gate below.
    if work.kind.releases_resources() {
        return None;
    }
    candidates
        .iter()
        .filter(|candidate| {
            devices
                .get(&candidate.device_id)
                .is_some_and(|device| candidate_is_eligible(work, candidate, device, true))
        })
        .map(|candidate| candidate.incremental_host_ram_bytes)
        .min()
}

/// Price a candidate as the matching will see it.
///
/// [`prepare_work`] is the one place a `CandidatePlacement` is cloned into the
/// matching, so normalizing here keeps the matcher's cost, the aggregate
/// headroom check, the immediate leases, the reservation the coordinator
/// commits, and the pre-grant VRAM revalidation all telling the same story.
///
/// Releasing work enters at zero. Charging it would shrink the very headroom it
/// is about to enlarge, let one queued unload block a peer through the
/// aggregate gate, and fail `validate_lease_for_grant` on exactly the full
/// device that needs unloading.
fn normalized_placement(kind: WorkKind, candidate: &CandidatePlacement) -> CandidatePlacement {
    if !kind.releases_resources() {
        return candidate.clone();
    }
    CandidatePlacement {
        predicted_vram_bytes: 0,
        incremental_host_ram_bytes: 0,
        ..candidate.clone()
    }
}

fn candidate_is_eligible(
    work: &WorkSnapshot,
    candidate: &CandidatePlacement,
    device: &DeviceSnapshot,
    require_idle: bool,
) -> bool {
    let releasing = work.kind.releases_resources();
    // Releasing work answers to `accepts_releasing_work`, which still excludes
    // a quarantined device but not one that is merely degraded, draining or
    // disabled — those are where an unload is most needed.
    (if releasing {
        device.accepts_releasing_work()
    } else {
        device.is_schedulable()
    }) && (!require_idle || device.is_idle_for(work.kind))
        && work
            .hard_device_id
            .as_ref()
            .is_none_or(|hard| hard == &candidate.device_id)
        && work
            .backend_requirement
            .is_none_or(|backend| backend == device.backend)
        // A full device is precisely when an unload must run, so releasing work
        // is not asked to fit in the VRAM it is about to return.
        && (releasing
            || candidate.predicted_vram_bytes
                <= candidate_available_vram_bytes(candidate, device))
}

fn candidate_available_vram_bytes(candidate: &CandidatePlacement, device: &DeviceSnapshot) -> u64 {
    if candidate.frozen_device_capacity {
        candidate.device_available_vram_bytes
    } else {
        device.available_vram_bytes
    }
}

fn build_immediate_leases(
    ordered_work: &[&WorkSnapshot],
    matching: &MatchingResult,
    devices: &BTreeMap<DeviceId, &DeviceSnapshot>,
    now_ms: u64,
) -> Vec<ImmediateLease> {
    ordered_work
        .iter()
        .filter_map(|work| {
            let candidate = matching.assignments.get(&work.id)?;
            let device = devices.get(&candidate.placement.device_id)?;
            let warm = device
                .warm_execution_fingerprints
                .contains(&candidate.placement.execution_fingerprint);
            let finish = now_ms
                .saturating_add(candidate.setup_ms)
                .saturating_add(candidate.placement.predicted_run_ms);
            Some(ImmediateLease {
                work_id: work.id.clone(),
                device_id: candidate.placement.device_id.clone(),
                worker_generation: device.worker_generation,
                placement: candidate.placement.clone(),
                reason: if work.starvation_forced() {
                    AssignmentReason::StarvationForced
                } else if warm {
                    AssignmentReason::WarmResident
                } else {
                    AssignmentReason::Priority
                },
                estimated_start_ms: now_ms,
                estimated_finish_ms: finish,
            })
        })
        .collect()
}

fn build_reservation(
    snapshot: &PlannerSnapshot,
    immediate_leases: &[ImmediateLease],
) -> MatchingReservation {
    // `lease.placement` already carries `normalized_placement`'s pricing.
    let items = immediate_leases
        .iter()
        .map(|lease| ReservationItem {
            work_id: lease.work_id.clone(),
            device_id: lease.device_id.clone(),
            execution_fingerprint: lease.placement.execution_fingerprint.clone(),
            predicted_vram_bytes: lease.placement.predicted_vram_bytes,
            device_available_vram_bytes: lease.placement.device_available_vram_bytes,
            host_ram_bytes: lease.placement.incremental_host_ram_bytes,
        })
        .collect::<Vec<_>>();
    MatchingReservation {
        sample_generation: snapshot.host_memory.sample_generation,
        ledger_sequence: snapshot.host_memory.ledger_sequence,
        total_host_ram_bytes: items
            .iter()
            .map(|item| item.host_ram_bytes)
            .fold(0_u64, u64::saturating_add),
        items,
    }
}

fn build_bypass_updates(
    ordered_work: &[&WorkSnapshot],
    leases: &[ImmediateLease],
    prepared: &BTreeMap<WorkId, PreparedWork>,
    host_headroom: u64,
    reserved_host_ram: u64,
) -> Vec<BypassUpdate> {
    let work_map = ordered_work
        .iter()
        .map(|work| (work.id.clone(), *work))
        .collect::<BTreeMap<_, _>>();
    let mut updates = Vec::new();
    let leased_ids = leases
        .iter()
        .map(|lease| lease.work_id.clone())
        .collect::<BTreeSet<_>>();
    for waiting_work in ordered_work.iter().copied().filter(|work| {
        work.ready
            && work.priority_class != crate::PriorityClass::Admin
            && work.bypass_count < 3
            && !leased_ids.contains(&work.id)
    }) {
        let Some(prepared_work) = prepared.get(&waiting_work.id) else {
            continue;
        };
        let younger_starts = leases.iter().filter_map(|lease| {
            let younger_work = work_map.get(&lease.work_id)?;
            let arrived_later = younger_work.ready_at_ms > waiting_work.ready_at_ms
                || (younger_work.ready_at_ms == waiting_work.ready_at_ms
                    && work_order(younger_work) > work_order(waiting_work));
            if !arrived_later {
                return None;
            }
            let compatible = prepared_work
                .all_candidates
                .iter()
                .find(|candidate| candidate.placement.device_id == lease.device_id)?;
            let replaced_total = reserved_host_ram
                .saturating_sub(lease.placement.incremental_host_ram_bytes)
                .saturating_add(compatible.placement.incremental_host_ram_bytes);
            (replaced_total <= host_headroom).then_some(lease)
        });
        let mut bypass_count = waiting_work.bypass_count;
        for younger in younger_starts {
            let new_count = bypass_count.saturating_add(1).min(3);
            updates.push(BypassUpdate {
                work_id: waiting_work.id.clone(),
                previous_count: bypass_count,
                new_count,
                younger_work_id: younger.work_id.clone(),
                opening_device_id: younger.device_id.clone(),
            });
            bypass_count = new_count;
        }
    }
    updates
}

fn select_horizon<'a>(
    ordered_work: &[&'a WorkSnapshot],
    base_horizon: usize,
) -> Vec<&'a WorkSnapshot> {
    let mut selected = ordered_work
        .iter()
        .filter(|work| work.ready)
        .take(base_horizon)
        .copied()
        .collect::<Vec<_>>();
    let selected_ids = selected
        .iter()
        .map(|work| work.id.clone())
        .collect::<BTreeSet<_>>();
    selected.extend(
        ordered_work
            .iter()
            .filter(|work| {
                work.ready && work.starvation_forced() && !selected_ids.contains(&work.id)
            })
            .copied(),
    );
    selected.sort_by(|left, right| work_order(left).cmp(&work_order(right)));
    selected
}

fn build_lanes(
    snapshot: &PlannerSnapshot,
    devices: &[DeviceSnapshot],
    horizon_work: &[&WorkSnapshot],
    prepared: &BTreeMap<WorkId, PreparedWork>,
    immediate_leases: &[ImmediateLease],
    budget: usize,
    mode: PlanningMode,
) -> (Vec<DeviceLane>, usize, OptimizerState) {
    let immediate_ids = immediate_leases
        .iter()
        .map(|lease| lease.work_id.clone())
        .collect::<BTreeSet<_>>();
    let future_work = horizon_work
        .iter()
        .filter(|work| !immediate_ids.contains(&work.id))
        .copied()
        .collect::<Vec<_>>();
    let schedule_context = ScheduleContext {
        snapshot,
        devices,
        future_work: &future_work,
        prepared,
        immediate_leases,
    };

    let mut seed = BTreeMap::<WorkId, CandidatePlacement>::new();
    let seed_schedule = evaluate_schedule(&schedule_context, &seed, true, true);
    seed = seed_schedule
        .future_assignments
        .iter()
        .map(|assignment| (assignment.work_id.clone(), assignment.placement.clone()))
        .collect();

    let optimizer_state = optimizer_state_for_mode(mode);
    if mode != PlanningMode::Optimize {
        return (seed_schedule.lanes, 0, optimizer_state);
    }
    if uniform_seed_is_optimal(snapshot, devices, &future_work, prepared, immediate_leases) {
        return (seed_schedule.lanes, 0, OptimizerState::SeedOnly);
    }

    let mut best_mapping = seed;
    let mut best_schedule = seed_schedule;
    let mut operations = 0_usize;

    'moves: for work in &future_work {
        let Some(prepared_work) = prepared.get(&work.id) else {
            continue;
        };
        for candidate in &prepared_work.all_candidates {
            if operations == budget {
                break 'moves;
            }
            let previous = best_mapping.get(&work.id).cloned();
            if previous.as_ref() == Some(&candidate.placement) {
                continue;
            }
            operations += 1;
            best_mapping.insert(work.id.clone(), candidate.placement.clone());
            let schedule = evaluate_schedule(&schedule_context, &best_mapping, false, false);
            if schedule_is_better(&schedule.score, &best_schedule.score) {
                best_schedule = schedule;
            } else if let Some(previous) = previous {
                best_mapping.insert(work.id.clone(), previous);
            } else {
                best_mapping.remove(&work.id);
            }
        }
    }

    if operations < budget {
        'swaps: for left_index in 0..future_work.len() {
            for right_index in left_index + 1..future_work.len() {
                if operations == budget {
                    break 'swaps;
                }
                let left = future_work[left_index];
                let right = future_work[right_index];
                let (Some(left_current), Some(right_current)) = (
                    best_mapping.get(&left.id).cloned(),
                    best_mapping.get(&right.id).cloned(),
                ) else {
                    continue;
                };
                let Some(left_replacement) =
                    candidate_for_device(prepared.get(&left.id), &right_current.device_id)
                else {
                    continue;
                };
                let Some(right_replacement) =
                    candidate_for_device(prepared.get(&right.id), &left_current.device_id)
                else {
                    continue;
                };
                operations += 1;
                best_mapping.insert(left.id.clone(), left_replacement);
                best_mapping.insert(right.id.clone(), right_replacement);
                let schedule = evaluate_schedule(&schedule_context, &best_mapping, false, false);
                if schedule_is_better(&schedule.score, &best_schedule.score) {
                    best_schedule = schedule;
                } else {
                    best_mapping.insert(left.id.clone(), left_current);
                    best_mapping.insert(right.id.clone(), right_current);
                }
            }
        }
    }

    let final_schedule = evaluate_schedule(&schedule_context, &best_mapping, false, true);
    (final_schedule.lanes, operations, optimizer_state)
}

const fn optimizer_state_for_mode(mode: PlanningMode) -> OptimizerState {
    match mode {
        PlanningMode::Optimize => OptimizerState::Optimized,
        PlanningMode::WatchdogFallback => OptimizerState::WatchdogFallback,
        PlanningMode::InternalErrorFallback => OptimizerState::InternalErrorFallback,
    }
}

fn uniform_seed_is_optimal(
    snapshot: &PlannerSnapshot,
    devices: &[DeviceSnapshot],
    future_work: &[&WorkSnapshot],
    prepared: &BTreeMap<WorkId, PreparedWork>,
    immediate_leases: &[ImmediateLease],
) -> bool {
    if future_work.is_empty() {
        return true;
    }
    let schedulable_ids = devices
        .iter()
        .filter(|device| device.is_schedulable())
        .map(|device| device.id.clone())
        .collect::<BTreeSet<_>>();
    if schedulable_ids.is_empty() {
        return true;
    }

    let mut base = devices
        .iter()
        .filter(|device| device.is_schedulable())
        .map(|device| {
            (
                device.id.clone(),
                (
                    match device.activity {
                        DeviceActivity::Idle => snapshot.now_ms,
                        DeviceActivity::Busy => device.available_at_ms.unwrap_or(u64::MAX / 4),
                    },
                    device.warm_execution_fingerprints.clone(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for lease in immediate_leases {
        let entry = base.entry(lease.device_id.clone()).or_default();
        entry.0 = lease.estimated_finish_ms;
        entry
            .1
            .insert(lease.placement.execution_fingerprint.clone());
    }
    let mut base_values = base.values();
    let Some(first_base) = base_values.next() else {
        return true;
    };
    if base_values.any(|value| value != first_base) {
        return false;
    }

    let mut common_signature = None::<(crate::ExecutionFingerprint, u64, u64, u64, u64, u64)>;
    let common_ready_at = future_work[0].ready_at_ms;
    for work in future_work {
        if work.ready_at_ms != common_ready_at {
            return false;
        }
        let Some(item) = prepared.get(&work.id) else {
            return false;
        };
        let candidate_devices = item
            .all_candidates
            .iter()
            .map(|candidate| candidate.placement.device_id.clone())
            .collect::<BTreeSet<_>>();
        if candidate_devices != schedulable_ids {
            return false;
        }
        let mut signatures = item.all_candidates.iter().map(|candidate| {
            (
                candidate.placement.execution_fingerprint.clone(),
                candidate.placement.predicted_vram_bytes,
                candidate.placement.incremental_host_ram_bytes,
                candidate.placement.cold_setup_ms,
                candidate.placement.warm_setup_ms,
                candidate.placement.predicted_run_ms,
            )
        });
        let Some(first) = signatures.next() else {
            return false;
        };
        if signatures.any(|signature| signature != first) {
            return false;
        }
        if common_signature
            .as_ref()
            .is_some_and(|common| common != &first)
        {
            return false;
        }
        common_signature = Some(first);
    }
    true
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ScheduleScore {
    total_lateness_ms: u128,
    makespan_ms: u64,
    sum_completion_ms: u128,
    cold_loads: usize,
}

#[derive(Clone, Debug)]
struct EvaluatedSchedule {
    lanes: Vec<DeviceLane>,
    future_assignments: Vec<PlannedAssignment>,
    score: ScheduleScore,
}

struct ScheduleContext<'a> {
    snapshot: &'a PlannerSnapshot,
    devices: &'a [DeviceSnapshot],
    future_work: &'a [&'a WorkSnapshot],
    prepared: &'a BTreeMap<WorkId, PreparedWork>,
    immediate_leases: &'a [ImmediateLease],
}

fn schedule_is_better(candidate: &ScheduleScore, current: &ScheduleScore) -> bool {
    (
        candidate.total_lateness_ms,
        candidate.makespan_ms,
        candidate.sum_completion_ms,
        candidate.cold_loads,
    ) < (
        current.total_lateness_ms,
        current.makespan_ms,
        current.sum_completion_ms,
        current.cold_loads,
    )
}

fn evaluate_schedule(
    context: &ScheduleContext<'_>,
    mapping: &BTreeMap<WorkId, CandidatePlacement>,
    choose_missing: bool,
    materialize: bool,
) -> EvaluatedSchedule {
    let snapshot = context.snapshot;
    let mut availability = context
        .devices
        .iter()
        .filter(|device| device.is_schedulable())
        .map(|device| {
            (
                device.id.clone(),
                match device.activity {
                    DeviceActivity::Idle => snapshot.now_ms,
                    DeviceActivity::Busy => device.available_at_ms.unwrap_or(u64::MAX / 4),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut warm = context
        .devices
        .iter()
        .filter(|device| device.is_schedulable())
        .map(|device| {
            (
                device.id.clone(),
                device.warm_execution_fingerprints.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut lane_assignments = context
        .devices
        .iter()
        .filter(|device| device.is_schedulable())
        .map(|device| (device.id.clone(), Vec::<PlannedAssignment>::new()))
        .collect::<BTreeMap<_, _>>();

    for lease in context.immediate_leases {
        availability.insert(lease.device_id.clone(), lease.estimated_finish_ms);
        warm.entry(lease.device_id.clone())
            .or_default()
            .insert(lease.placement.execution_fingerprint.clone());
        lane_assignments
            .entry(lease.device_id.clone())
            .or_default()
            .push(PlannedAssignment {
                work_id: lease.work_id.clone(),
                device_id: lease.device_id.clone(),
                placement: lease.placement.clone(),
                estimated_start_ms: lease.estimated_start_ms,
                estimated_finish_ms: lease.estimated_finish_ms,
                immediate: true,
            });
    }

    let mut future_assignments = Vec::new();
    let mut total_lateness_ms = 0_u128;
    let mut sum_completion_ms = 0_u128;
    let mut cold_loads = 0_usize;
    let mut makespan_ms = context
        .immediate_leases
        .iter()
        .map(|lease| lease.estimated_finish_ms)
        .max()
        .unwrap_or(snapshot.now_ms);
    for work in context.future_work {
        let Some(prepared_work) = context.prepared.get(&work.id) else {
            continue;
        };
        let selected = if let Some(candidate) = mapping.get(&work.id) {
            prepared_work
                .all_candidates
                .iter()
                .find(|item| &item.placement == candidate)
                .map(|item| &item.placement)
        } else if choose_missing {
            prepared_work
                .all_candidates
                .iter()
                .min_by(|left, right| {
                    projected_finish(left, work.ready_at_ms, &availability, &warm)
                        .cmp(&projected_finish(
                            right,
                            work.ready_at_ms,
                            &availability,
                            &warm,
                        ))
                        .then_with(|| {
                            left.placement
                                .incremental_host_ram_bytes
                                .cmp(&right.placement.incremental_host_ram_bytes)
                        })
                        .then_with(|| {
                            Reverse(left.placement.device_available_vram_bytes)
                                .cmp(&Reverse(right.placement.device_available_vram_bytes))
                        })
                        .then_with(|| left.placement.device_id.cmp(&right.placement.device_id))
                })
                .map(|item| &item.placement)
        } else {
            None
        };
        let Some(placement) = selected else {
            continue;
        };
        let start = availability
            .get(&placement.device_id)
            .copied()
            .unwrap_or(u64::MAX / 4)
            .max(work.ready_at_ms);
        let is_warm = warm
            .get(&placement.device_id)
            .is_some_and(|fingerprints| fingerprints.contains(&placement.execution_fingerprint));
        let setup = if is_warm {
            placement.warm_setup_ms
        } else {
            cold_loads += 1;
            placement.cold_setup_ms
        };
        let finish = start
            .saturating_add(setup)
            .saturating_add(placement.predicted_run_ms);
        availability.insert(placement.device_id.clone(), finish);
        warm.entry(placement.device_id.clone())
            .or_default()
            .insert(placement.execution_fingerprint.clone());
        let soft_budget = (placement.predicted_run_ms / 4).clamp(5_000, 60_000);
        let deadline = work.ready_at_ms.saturating_add(soft_budget);
        total_lateness_ms += u128::from(start.saturating_sub(deadline));
        sum_completion_ms += u128::from(finish);
        makespan_ms = makespan_ms.max(finish);
        if materialize {
            let assignment = PlannedAssignment {
                work_id: work.id.clone(),
                device_id: placement.device_id.clone(),
                placement: placement.clone(),
                estimated_start_ms: start,
                estimated_finish_ms: finish,
                immediate: false,
            };
            lane_assignments
                .entry(assignment.device_id.clone())
                .or_default()
                .push(assignment.clone());
            future_assignments.push(assignment);
        }
    }

    let lanes = lane_assignments
        .into_iter()
        .map(|(device_id, assignments)| DeviceLane {
            device_id,
            assignments,
        })
        .collect();
    EvaluatedSchedule {
        lanes,
        future_assignments,
        score: ScheduleScore {
            total_lateness_ms,
            makespan_ms,
            sum_completion_ms,
            cold_loads,
        },
    }
}

fn projected_finish(
    candidate: &MatchCandidate,
    ready_at_ms: u64,
    availability: &BTreeMap<DeviceId, u64>,
    warm: &BTreeMap<DeviceId, BTreeSet<crate::ExecutionFingerprint>>,
) -> (u64, u64) {
    let start = availability
        .get(&candidate.placement.device_id)
        .copied()
        .unwrap_or(u64::MAX / 4)
        .max(ready_at_ms);
    let setup = if warm
        .get(&candidate.placement.device_id)
        .is_some_and(|fingerprints| {
            fingerprints.contains(&candidate.placement.execution_fingerprint)
        }) {
        candidate.placement.warm_setup_ms
    } else {
        candidate.placement.cold_setup_ms
    };
    (
        start
            .saturating_add(setup)
            .saturating_add(candidate.placement.predicted_run_ms),
        setup,
    )
}

fn candidate_for_device(
    prepared: Option<&PreparedWork>,
    device_id: &DeviceId,
) -> Option<CandidatePlacement> {
    prepared?
        .all_candidates
        .iter()
        .filter(|candidate| &candidate.placement.device_id == device_id)
        .min_by(|left, right| {
            left.placement
                .incremental_host_ram_bytes
                .cmp(&right.placement.incremental_host_ram_bytes)
                .then_with(|| left.setup_ms.cmp(&right.setup_ms))
                .then_with(|| {
                    left.placement
                        .affinity_penalty
                        .cmp(&right.placement.affinity_penalty)
                })
        })
        .map(|candidate| candidate.placement.clone())
}

fn canonical_devices(devices: &[DeviceSnapshot]) -> Vec<DeviceSnapshot> {
    let mut devices = devices.to_vec();
    devices.sort_by(|left, right| left.id.cmp(&right.id));
    devices
}

fn setup_ms(candidate: &CandidatePlacement, device: &DeviceSnapshot) -> u64 {
    if device
        .warm_execution_fingerprints
        .contains(&candidate.execution_fingerprint)
    {
        candidate.warm_setup_ms
    } else {
        candidate.cold_setup_ms
    }
}

fn sort_match_candidates(candidates: &mut [MatchCandidate]) {
    candidates.sort_by(|left, right| {
        left.placement
            .incremental_host_ram_bytes
            .cmp(&right.placement.incremental_host_ram_bytes)
            .then_with(|| left.setup_ms.cmp(&right.setup_ms))
            .then_with(|| {
                Reverse(left.placement.device_available_vram_bytes)
                    .cmp(&Reverse(right.placement.device_available_vram_bytes))
            })
            .then_with(|| left.placement.device_id.cmp(&right.placement.device_id))
            .then_with(|| {
                left.placement
                    .affinity_penalty
                    .cmp(&right.placement.affinity_penalty)
            })
            .then_with(|| {
                left.placement
                    .execution_fingerprint
                    .cmp(&right.placement.execution_fingerprint)
            })
    });
}

fn work_order(work: &WorkSnapshot) -> (bool, u8, u64, &WorkId) {
    (
        !work.starvation_forced(),
        work.priority_class.sort_key(),
        work.queue_rank,
        &work.id,
    )
}
