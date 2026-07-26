use std::collections::BTreeMap;

use crate::{CandidatePlacement, DeviceId, WorkId, WorkSnapshot};

/// Mutation-maintained candidate index used by immediate matching.
///
/// The coordinator can update only the changed work unit. Candidate vectors
/// are normalized here so insertion and source-map order cannot influence a
/// plan. Device health, activity, VRAM, pins, and warm state remain snapshot
/// facts and are filtered by the planner on every run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EligibilityIndex {
    by_work: BTreeMap<WorkId, Vec<CandidatePlacement>>,
}

impl EligibilityIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_work(work: &[WorkSnapshot]) -> Self {
        let mut index = Self::new();
        for item in work {
            index.upsert_work(item);
        }
        index
    }

    pub fn upsert_work(&mut self, work: &WorkSnapshot) {
        let mut candidates = work.candidate_placements.clone();
        normalize_candidates(&mut candidates);
        self.by_work.insert(work.id.clone(), candidates);
    }

    pub fn remove_work(&mut self, work_id: &WorkId) -> Option<Vec<CandidatePlacement>> {
        self.by_work.remove(work_id)
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
}

fn normalize_candidates(candidates: &mut Vec<CandidatePlacement>) {
    candidates.sort_by(|left, right| {
        candidate_key(left)
            .cmp(&candidate_key(right))
            .then_with(|| left.execution_fingerprint.cmp(&right.execution_fingerprint))
    });
    candidates.dedup();
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
