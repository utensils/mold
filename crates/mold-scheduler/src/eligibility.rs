use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::{CandidatePlacement, DeviceId, PlannerSnapshot, WorkId, WorkSnapshot};

/// Mutation-maintained candidate index used by immediate matching.
///
/// The coordinator can update only the changed work unit. Candidate vectors
/// are normalized here so insertion and source-map order cannot influence a
/// plan. Device health, activity, VRAM, pins, and warm state remain snapshot
/// facts and are filtered by the planner on every run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EligibilityIndex {
    state_version: u64,
    by_work: BTreeMap<WorkId, Vec<CandidatePlacement>>,
}

impl EligibilityIndex {
    pub fn new(state_version: u64) -> Self {
        Self {
            state_version,
            by_work: BTreeMap::new(),
        }
    }

    pub fn from_work(state_version: u64, work: &[WorkSnapshot]) -> Self {
        let mut index = Self::new(state_version);
        for item in work {
            index.upsert_work(state_version, item);
        }
        index
    }

    pub fn from_snapshot(snapshot: &PlannerSnapshot) -> Self {
        Self::from_work(snapshot.state_version, &snapshot.work)
    }

    pub fn state_version(&self) -> u64 {
        self.state_version
    }

    /// Applies a work mutation and binds the resulting index to the reducer's
    /// new state version. The caller must use the same state version in the
    /// `PlannerSnapshot` produced by that reducer turn.
    pub fn upsert_work(&mut self, state_version: u64, work: &WorkSnapshot) {
        let mut candidates = work.candidate_placements.clone();
        normalize_candidates(&mut candidates);
        self.by_work.insert(work.id.clone(), candidates);
        self.state_version = state_version;
    }

    /// Applies a removal and binds the resulting index to the reducer's new
    /// state version.
    pub fn remove_work(
        &mut self,
        state_version: u64,
        work_id: &WorkId,
    ) -> Option<Vec<CandidatePlacement>> {
        let removed = self.by_work.remove(work_id);
        self.state_version = state_version;
        removed
    }

    pub fn candidates_for(&self, work_id: &WorkId) -> Option<&[CandidatePlacement]> {
        self.by_work.get(work_id).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.by_work.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_work.is_empty()
    }

    pub(crate) fn work_ids(&self) -> impl Iterator<Item = &WorkId> {
        self.by_work.keys()
    }

    pub(crate) fn candidates_match(&self, work: &WorkSnapshot) -> bool {
        let Some(indexed) = self.by_work.get(&work.id) else {
            return false;
        };
        if indexed.as_slice() == work.candidate_placements.as_slice() {
            return true;
        }
        let mut snapshot_candidates = work.candidate_placements.iter().collect::<Vec<_>>();
        snapshot_candidates.sort_by(|left, right| compare_candidates(left, right));
        snapshot_candidates.dedup();
        if indexed.len() != snapshot_candidates.len() {
            return false;
        }
        snapshot_candidates
            .into_iter()
            .zip(indexed)
            .all(|(snapshot, indexed)| snapshot == indexed)
    }
}

fn normalize_candidates(candidates: &mut Vec<CandidatePlacement>) {
    candidates.sort_by(compare_candidates);
    candidates.dedup();
}

fn compare_candidates(left: &CandidatePlacement, right: &CandidatePlacement) -> Ordering {
    candidate_key(left)
        .cmp(&candidate_key(right))
        .then_with(|| left.execution_fingerprint.cmp(&right.execution_fingerprint))
}

fn candidate_key(candidate: &CandidatePlacement) -> (&DeviceId, u64, u64, u64, u64, u64) {
    (
        &candidate.device_id,
        candidate.incremental_host_ram_bytes,
        candidate.predicted_vram_bytes,
        candidate.cold_setup_ms,
        candidate.warm_setup_ms,
        candidate.predicted_run_ms,
    )
}
