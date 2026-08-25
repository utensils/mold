//! Owner-scoped startup reconciliation for encrypted durable queue media.
//!
//! This module owns policy and ordering, not storage or schema. The concrete
//! adapter is deliberately supplied by the queue-media store/DB integration:
//! that keeps the safety policy independently testable while those lower
//! layers retain authority over authenticated file operations and SQLite
//! transactions.

use std::collections::{BTreeMap, BTreeSet};

use crate::queue_journal::QueueJournal;

/// How many held job ids one retained degradation reason names before it
/// switches to a count. Enough to start a `GET /api/queue` lookup by hand.
const HELD_JOB_REASON_SAMPLE: usize = 5;

/// How many reconciliation issues are retained verbatim before the rest become
/// a count. `issues` carries one entry per inconsistent obligation, and the
/// durable backlog is deliberately not capped by `queue_capacity`, so an
/// unbounded copy would be logged, retained for the life of the process, and
/// re-serialized on every `/api/status` poll.
const ISSUE_REASON_SAMPLE: usize = 5;

const MEDIA_HOLD_REASON: &str =
    "durable request media is unavailable or failed startup reconciliation";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObligationState {
    Active,
    GcPending,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaObligation {
    /// Present for an active obligation joined to its queue row. Trigger-made
    /// `gc_pending` obligations outlive the deleted queue row and therefore do
    /// not carry a job id.
    pub job_id: Option<String>,
    pub set_id: String,
    pub state: ObligationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum StoreEntryState {
    Staging,
    // Constructed by the concrete store adapter when that independently
    // reviewed slice is integrated; the dependency-free coordinator still
    // has to define and handle the state now.
    #[allow(dead_code)]
    Active,
    Retired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreEntry {
    /// Authenticated job identity captured by the initial owner inspection.
    /// Concrete mutations reuse the full owner/job/set tuple under the store's
    /// owner/job fencing without rescanning every bundle owned by the process.
    pub job_id: String,
    pub set_id: String,
    pub state: StoreEntryState,
}

/// An entry the store refused to adopt as queue-media authority. `set_id_hint`
/// is only an already-validated opaque path component, never an arbitrary
/// path. It prevents a malformed or symlink entry colliding with an obligation
/// from being mistaken for an absent, already-deleted bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UntouchedEntry {
    pub set_id_hint: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StoreInspection {
    pub entries: Vec<StoreEntry>,
    pub untouched: Vec<UntouchedEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnclaimedOwnerRoot {
    pub owner_id_hint: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreInitializationPolicy {
    Deny,
    /// Initialization is permitted only after the implementation proves the
    /// entire queue-media store is empty across every owner root, including
    /// staging and unrecognized entries. Owner-local emptiness is not enough.
    IfGloballyEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdapterFailureKind {
    Database,
    KeyMissing,
    KeyCorrupt,
    Permission,
    UnsafeLayout,
    Io,
    Invariant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdapterError {
    pub kind: AdapterFailureKind,
    pub detail: String,
}

impl AdapterError {
    pub(crate) fn new(kind: AdapterFailureKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

impl std::error::Error for AdapterError {}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.detail)
    }
}

/// Concrete DB/store boundary. Every method is implicitly scoped to the
/// `owner_uuid` argument. Implementations must reject attempts to cross that
/// owner and must never follow a symlink/reparse entry.
pub(crate) trait QueueMediaStartupAdapter {
    /// Read both active and trigger-created `gc_pending` obligations. The
    /// implementation must cross-check active obligation rows against the
    /// queue's own media-set references and return `Invariant` on projection
    /// disagreement; an unreported reference could otherwise be mistaken for
    /// a file-first orphan and deleted.
    fn obligations(&self, owner_uuid: &str) -> Result<Vec<MediaObligation>, AdapterError>;

    /// Enumerate, but never inspect or mutate, queue-media roots not claimed
    /// by this process. This is the reporting counterpart to owner fencing:
    /// live, foreign, and unadopted roots remain untouched but visible.
    fn unclaimed_owner_roots(
        &self,
        owner_uuid: &str,
    ) -> Result<Vec<UnclaimedOwnerRoot>, AdapterError>;

    /// Validate/open the global store under this queue owner's startup
    /// authority. [`StoreInitializationPolicy::IfGloballyEmpty`] is passed only
    /// when SQLite has no obligations for the claimed owner, but the
    /// implementation must independently prove the whole store is empty. A
    /// missing or corrupt key is never regenerated over payloads.
    fn open_store(
        &self,
        owner_uuid: &str,
        initialization: StoreInitializationPolicy,
    ) -> Result<(), AdapterError>;

    /// Authenticated, read-only inspection of only the claimed/adopted owner.
    /// Opening and initialization are deliberately separate from inspection.
    fn inspect_owner(&self, owner_uuid: &str) -> Result<StoreInspection, AdapterError>;

    /// Restore a DB-active bundle found in the retired namespace.
    fn restore(&self, owner_uuid: &str, entry: &StoreEntry) -> Result<(), AdapterError>;

    /// Remove one authenticated owner-local entry. Active entries must be
    /// retired durably before unlink; the implementation owns that sequence.
    fn delete(&self, owner_uuid: &str, entry: &StoreEntry) -> Result<(), AdapterError>;

    /// Clear a trigger-created delete obligation only after its bundle is
    /// absent or `delete` completed successfully.
    fn clear_gc_pending(&self, owner_uuid: &str, set_id: &str) -> Result<(), AdapterError>;

    /// Atomically hold only the named active media jobs belonging to this
    /// owner, retaining their bundles and obligations. A failure is fatal to
    /// startup because allowing the feeder to see an unquarantined sanitized
    /// row could execute a media request without its media.
    fn hold_jobs(
        &self,
        owner_uuid: &str,
        job_ids: &[String],
        reason: &str,
    ) -> Result<(), AdapterError>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StartupReport {
    pub owner_uuid: Option<String>,
    pub durable_media_ready: bool,
    pub restored: Vec<String>,
    pub deleted: Vec<String>,
    pub cleared_gc_pending: Vec<String>,
    pub held_jobs: Vec<String>,
    pub untouched: Vec<UntouchedEntry>,
    pub unclaimed_owner_roots: Vec<UnclaimedOwnerRoot>,
    pub issues: Vec<String>,
}

impl StartupReport {
    /// Every reason this reconciliation withheld media durability, phrased for
    /// an operator reading `/api/status` rather than for a log grep.
    ///
    /// Held jobs are included because they are the half a bare issue list
    /// misses: a run can end with no issue at all and still quarantine work
    /// whose media never came back.
    ///
    /// Both halves are bounded. Every entry here is logged once, retained for
    /// the life of the process, and re-serialized on every `/api/status` poll,
    /// while `issues` carries one string per inconsistent obligation and the
    /// durable backlog has no `queue_capacity` ceiling — so a fleet-wide
    /// inconsistency would otherwise be an unbounded response and an unbounded
    /// log. The full detail stays in `GET /api/queue`, which pages.
    pub(crate) fn degradation_reasons(&self) -> Vec<String> {
        if self.durable_media_ready {
            return Vec::new();
        }
        let mut reasons: Vec<String> = self
            .issues
            .iter()
            .take(ISSUE_REASON_SAMPLE)
            .cloned()
            .collect();
        let unlisted_issues = self.issues.len().saturating_sub(reasons.len());
        if unlisted_issues > 0 {
            reasons.push(format!(
                "and {unlisted_issues} further queue-media startup issue(s) not listed"
            ));
        }
        if !self.held_jobs.is_empty() {
            // The durable backlog is deliberately not bounded by
            // `queue_capacity`, so naming every held job would let one
            // unusable store put an unbounded string on `/api/status` for the
            // life of the process. The count is the actionable number; the
            // sample is only there to start a `GET /api/queue` lookup.
            let sample: Vec<&str> = self
                .held_jobs
                .iter()
                .take(HELD_JOB_REASON_SAMPLE)
                .map(String::as_str)
                .collect();
            let remainder = self.held_jobs.len().saturating_sub(sample.len());
            let mut reason = format!(
                "{} queued job(s) are held because their request media is unavailable: {}",
                self.held_jobs.len(),
                sample.join(", ")
            );
            if remainder > 0 {
                reason.push_str(&format!(" and {remainder} more"));
            }
            reasons.push(reason);
        }
        reasons
    }
}

/// Reconcile encrypted request media under the already-claimed queue owner.
///
/// Call this after [`QueueJournal::new`] and before runtime-claim recovery,
/// feeder creation, or HTTP router construction. The capability is lowered at
/// entry and raised only after the whole owner reaches a clean fixed point.
/// Every returned `Err` is fatal to queue startup: callers must not start the
/// feeder or any queue producer after an infrastructure/invariant failure.
pub(crate) fn reconcile_claimed_owner(
    journal: &QueueJournal,
    adapter: &impl QueueMediaStartupAdapter,
) -> Result<StartupReport, AdapterError> {
    journal.set_durable_media_ready(false);
    // Readiness and its explanation are published from exactly one place. The
    // body has several early returns for a refused or unreadable store, and
    // those are precisely the cases whose reason an operator needs — publishing
    // per return is how they came to be dropped.
    let report = reconcile_claimed_owner_inner(journal, adapter)?;
    journal.set_durable_media_status(report.durable_media_ready, report.degradation_reasons());
    Ok(report)
}

fn reconcile_claimed_owner_inner(
    journal: &QueueJournal,
    adapter: &impl QueueMediaStartupAdapter,
) -> Result<StartupReport, AdapterError> {
    let Some(owner_uuid) = journal.owner_uuid() else {
        return Ok(StartupReport {
            issues: vec!["durable queue owner is unavailable".to_string()],
            ..StartupReport::default()
        });
    };

    let mut report = StartupReport {
        owner_uuid: Some(owner_uuid.to_string()),
        ..StartupReport::default()
    };
    let obligations = adapter.obligations(owner_uuid)?;
    if obligations.iter().any(|obligation| {
        obligation.state == ObligationState::Active && obligation.job_id.is_none()
    }) {
        return Err(AdapterError::new(
            AdapterFailureKind::Invariant,
            "active media obligation is not joined to a queue job",
        ));
    }
    let active_jobs: Vec<String> = obligations
        .iter()
        .filter(|obligation| obligation.state == ObligationState::Active)
        .filter_map(|obligation| obligation.job_id.clone())
        .collect();

    let initialization = if obligations.is_empty() {
        StoreInitializationPolicy::IfGloballyEmpty
    } else {
        StoreInitializationPolicy::Deny
    };
    let store_result = adapter.open_store(owner_uuid, initialization);
    if let Err(error) = store_result {
        if !active_jobs.is_empty() {
            adapter.hold_jobs(owner_uuid, &active_jobs, MEDIA_HOLD_REASON)?;
            report.held_jobs = active_jobs;
        }
        report
            .issues
            .push(format!("owner media store unavailable: {error}"));
        return Ok(report);
    }

    match adapter.unclaimed_owner_roots(owner_uuid) {
        Ok(roots) => {
            report.unclaimed_owner_roots = roots;
        }
        Err(error) => report.issues.push(format!(
            "could not enumerate non-claimed owner roots: {error}"
        )),
    }

    let inspection = match adapter.inspect_owner(owner_uuid) {
        Ok(inspection) => inspection,
        Err(error) => {
            if !active_jobs.is_empty() {
                adapter.hold_jobs(owner_uuid, &active_jobs, MEDIA_HOLD_REASON)?;
                report.held_jobs = active_jobs;
            }
            report
                .issues
                .push(format!("owner media store unavailable: {error}"));
            return Ok(report);
        }
    };

    // Schema constraints are defense in depth, not startup authority. Refuse
    // contradictory obligation projections before any filesystem mutation:
    // otherwise a stale GC row could delete a set still required by active
    // work, or one job could ambiguously claim two independent sets.
    let mut obligations_by_set: BTreeMap<&str, Vec<&MediaObligation>> = BTreeMap::new();
    let mut obligations_by_job: BTreeMap<&str, Vec<&MediaObligation>> = BTreeMap::new();
    for obligation in &obligations {
        obligations_by_set
            .entry(&obligation.set_id)
            .or_default()
            .push(obligation);
        if let Some(job_id) = &obligation.job_id {
            obligations_by_job
                .entry(job_id)
                .or_default()
                .push(obligation);
        }
    }
    let mut conflicting_sets = BTreeSet::new();
    let mut conflicting_jobs = BTreeSet::new();
    for (set_id, matching) in &obligations_by_set {
        if matching.len() > 1 {
            conflicting_sets.insert(*set_id);
            conflicting_jobs.extend(
                matching
                    .iter()
                    .filter_map(|obligation| obligation.job_id.as_deref()),
            );
            report
                .issues
                .push(format!("media set {set_id} has conflicting DB obligations"));
        }
    }
    for (job_id, matching) in &obligations_by_job {
        if matching.len() > 1 {
            conflicting_jobs.insert(*job_id);
            conflicting_sets.extend(matching.iter().map(|obligation| obligation.set_id.as_str()));
            report
                .issues
                .push(format!("media job {job_id} has conflicting DB obligations"));
        }
    }

    report.untouched = inspection.untouched;
    for entry in &report.untouched {
        report.issues.push(format!(
            "left unrecognized owner entry untouched: {}",
            entry.description
        ));
    }

    let untouched_hints: BTreeSet<&str> = report
        .untouched
        .iter()
        .filter_map(|entry| entry.set_id_hint.as_deref())
        .collect();
    let has_hintless_untouched = report
        .untouched
        .iter()
        .any(|entry| entry.set_id_hint.is_none());
    let mut entries: BTreeMap<&str, Vec<&StoreEntry>> = BTreeMap::new();
    for entry in &inspection.entries {
        entries.entry(&entry.set_id).or_default().push(entry);
    }
    for (set_id, matching) in &entries {
        if matching.len() > 1 {
            report.issues.push(format!(
                "media set {set_id} appears in multiple store states"
            ));
        }
    }
    let obligation_ids: BTreeSet<&str> = obligations
        .iter()
        .map(|obligation| obligation.set_id.as_str())
        .collect();
    let mut jobs_to_hold = BTreeSet::new();

    for obligation in &obligations {
        if conflicting_sets.contains(obligation.set_id.as_str())
            || obligation
                .job_id
                .as_deref()
                .is_some_and(|job_id| conflicting_jobs.contains(job_id))
        {
            if obligation.state == ObligationState::Active {
                jobs_to_hold.insert(
                    obligation
                        .job_id
                        .clone()
                        .expect("active obligations were validated above"),
                );
            }
            continue;
        }
        let matching = entries.get(obligation.set_id.as_str());
        let unsafe_collision = untouched_hints.contains(obligation.set_id.as_str());
        let unique = matching.and_then(|matches| (matches.len() == 1).then_some(matches[0]));

        match obligation.state {
            ObligationState::Active => {
                if unsafe_collision || matching.is_some_and(|matches| matches.len() > 1) {
                    jobs_to_hold.insert(
                        obligation
                            .job_id
                            .clone()
                            .expect("active obligations were validated above"),
                    );
                    continue;
                }
                let expected_job_id = obligation
                    .job_id
                    .as_deref()
                    .expect("active obligations were validated above");
                if unique.is_some_and(|entry| entry.job_id != expected_job_id) {
                    let entry = unique.expect("mismatched job came from a unique entry");
                    report.issues.push(format!(
                        "active media obligation {} belongs to job {expected_job_id}, but its authenticated bundle belongs to job {}",
                        obligation.set_id, entry.job_id
                    ));
                    jobs_to_hold.insert(expected_job_id.to_string());
                    continue;
                }
                match unique.map(|entry| entry.state) {
                    Some(StoreEntryState::Active) => {}
                    Some(StoreEntryState::Retired) => {
                        let entry = unique.expect("retired state came from a unique entry");
                        match adapter.restore(owner_uuid, entry) {
                            Ok(()) => report.restored.push(obligation.set_id.clone()),
                            Err(error) => {
                                report.issues.push(format!(
                                    "could not restore media set {}: {error}",
                                    obligation.set_id
                                ));
                                jobs_to_hold.insert(
                                    obligation
                                        .job_id
                                        .clone()
                                        .expect("active obligations were validated above"),
                                );
                            }
                        }
                    }
                    Some(StoreEntryState::Staging) | None => {
                        report.issues.push(format!(
                            "active media obligation {} has no sealed active bundle",
                            obligation.set_id
                        ));
                        jobs_to_hold.insert(
                            obligation
                                .job_id
                                .clone()
                                .expect("active obligations were validated above"),
                        );
                    }
                }
            }
            ObligationState::GcPending => {
                if unsafe_collision || matching.is_some_and(|matches| matches.len() > 1) {
                    report.issues.push(format!(
                        "kept GC obligation {} because its store entry is unsafe",
                        obligation.set_id
                    ));
                    continue;
                }
                if unique.is_none() && has_hintless_untouched {
                    report.issues.push(format!(
                        "kept GC obligation {} because a hintless unsafe entry makes absence unprovable",
                        obligation.set_id
                    ));
                    continue;
                }
                let deleted = match unique {
                    Some(entry) => match adapter.delete(owner_uuid, entry) {
                        Ok(()) => {
                            report.deleted.push(obligation.set_id.clone());
                            true
                        }
                        Err(error) => {
                            report.issues.push(format!(
                                "could not delete GC-pending media set {}: {error}",
                                obligation.set_id
                            ));
                            false
                        }
                    },
                    None => true,
                };
                if deleted {
                    match adapter.clear_gc_pending(owner_uuid, &obligation.set_id) {
                        Ok(()) => report.cleared_gc_pending.push(obligation.set_id.clone()),
                        Err(error) => report.issues.push(format!(
                            "could not clear GC obligation {}: {error}",
                            obligation.set_id
                        )),
                    }
                }
            }
        }
    }

    // A recognized set with no DB obligation can only be an interrupted
    // file-first admission or post-terminal cleanup. Trigger-created
    // `gc_pending` rows were handled above; no age/timeout heuristic exists.
    if report.issues.is_empty() && jobs_to_hold.is_empty() {
        for (set_id, matching) in &entries {
            if obligation_ids.contains(*set_id)
                || untouched_hints.contains(*set_id)
                || matching.len() != 1
            {
                continue;
            }
            match adapter.delete(owner_uuid, matching[0]) {
                Ok(()) => report.deleted.push((*set_id).to_string()),
                Err(error) => report.issues.push(format!(
                    "could not remove unreferenced media set {set_id}: {error}"
                )),
            }
        }
    }

    report.held_jobs = jobs_to_hold.into_iter().collect();
    if !report.held_jobs.is_empty() {
        adapter.hold_jobs(owner_uuid, &report.held_jobs, MEDIA_HOLD_REASON)?;
    }

    report.durable_media_ready = report.issues.is_empty() && report.held_jobs.is_empty();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_journal::QueueJournal;
    use mold_db::generation_queue::{self, GenerationQueueRow, QueueRowState};
    use mold_db::MetadataDb;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct FakeAdapter {
        obligations: Mutex<HashMap<String, Vec<MediaObligation>>>,
        unclaimed_owner_roots: Mutex<HashMap<String, Vec<UnclaimedOwnerRoot>>>,
        open_errors: Mutex<HashMap<String, AdapterError>>,
        inspections: Mutex<HashMap<String, Result<StoreInspection, AdapterError>>>,
        opens: Mutex<Vec<(String, StoreInitializationPolicy)>>,
        actions: Mutex<Vec<(String, String, String)>>,
    }

    impl FakeAdapter {
        fn set_obligations(&self, owner: &str, obligations: Vec<MediaObligation>) {
            self.obligations
                .lock()
                .unwrap()
                .insert(owner.to_string(), obligations);
        }

        fn set_inspection(&self, owner: &str, inspection: Result<StoreInspection, AdapterError>) {
            self.inspections
                .lock()
                .unwrap()
                .insert(owner.to_string(), inspection);
        }

        fn set_open_error(&self, owner: &str, error: &str) {
            self.open_errors.lock().unwrap().insert(
                owner.to_string(),
                AdapterError::new(AdapterFailureKind::KeyMissing, error),
            );
        }

        fn set_unclaimed_owner_roots(&self, owner: &str, roots: Vec<UnclaimedOwnerRoot>) {
            self.unclaimed_owner_roots
                .lock()
                .unwrap()
                .insert(owner.to_string(), roots);
        }

        fn actions(&self) -> Vec<(String, String, String)> {
            self.actions.lock().unwrap().clone()
        }

        fn action(&self, owner: &str, verb: &str, value: &str) {
            self.actions.lock().unwrap().push((
                owner.to_string(),
                verb.to_string(),
                value.to_string(),
            ));
        }
    }

    impl QueueMediaStartupAdapter for FakeAdapter {
        fn obligations(&self, owner_uuid: &str) -> Result<Vec<MediaObligation>, AdapterError> {
            Ok(self
                .obligations
                .lock()
                .unwrap()
                .get(owner_uuid)
                .cloned()
                .unwrap_or_default())
        }

        fn unclaimed_owner_roots(
            &self,
            owner_uuid: &str,
        ) -> Result<Vec<UnclaimedOwnerRoot>, AdapterError> {
            Ok(self
                .unclaimed_owner_roots
                .lock()
                .unwrap()
                .get(owner_uuid)
                .cloned()
                .unwrap_or_default())
        }

        fn open_store(
            &self,
            owner_uuid: &str,
            initialization: StoreInitializationPolicy,
        ) -> Result<(), AdapterError> {
            self.opens
                .lock()
                .unwrap()
                .push((owner_uuid.to_string(), initialization));
            match self.open_errors.lock().unwrap().get(owner_uuid).cloned() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }

        fn inspect_owner(&self, owner_uuid: &str) -> Result<StoreInspection, AdapterError> {
            self.inspections
                .lock()
                .unwrap()
                .get(owner_uuid)
                .cloned()
                .unwrap_or_else(|| Ok(StoreInspection::default()))
        }

        fn restore(&self, owner_uuid: &str, entry: &StoreEntry) -> Result<(), AdapterError> {
            self.action(owner_uuid, "restore", &entry.set_id);
            Ok(())
        }

        fn delete(&self, owner_uuid: &str, entry: &StoreEntry) -> Result<(), AdapterError> {
            self.action(owner_uuid, "delete", &entry.set_id);
            Ok(())
        }

        fn clear_gc_pending(&self, owner_uuid: &str, set_id: &str) -> Result<(), AdapterError> {
            self.action(owner_uuid, "clear-gc", set_id);
            Ok(())
        }

        fn hold_jobs(
            &self,
            owner_uuid: &str,
            job_ids: &[String],
            _reason: &str,
        ) -> Result<(), AdapterError> {
            for job in job_ids {
                self.action(owner_uuid, "hold", job);
            }
            Ok(())
        }
    }

    fn journal(
        home: &std::path::Path,
        db: Arc<Option<MetadataDb>>,
        instance: &str,
    ) -> QueueJournal {
        QueueJournal::new(db, Some(home), instance)
    }

    fn queue_row(owner: &str, id: &str) -> GenerationQueueRow {
        GenerationQueueRow {
            id: id.to_string(),
            owner_uuid: owner.to_string(),
            state: QueueRowState::Queued,
            model: "flux-dev:q4".to_string(),
            request_json: "{}".to_string(),
            output_dir: "/gallery".into(),
            target_gpu: None,
            target_device_id: None,
            completion_payload: "full".to_string(),
            seed_pinned: false,
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms: 1,
            updated_at_ms: 1,
            started_at_ms: None,
            media_set_id: None,
            admission_authority: None,
        }
    }

    #[test]
    fn clean_reconciliation_stays_dark_until_admission_is_installed() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        assert_eq!(journal.durable_media_capabilities(), None);

        let adapter = FakeAdapter::default();
        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();
        assert!(report.durable_media_ready);
        assert_eq!(
            adapter.opens.lock().unwrap().as_slice(),
            &[(
                journal.owner_uuid().unwrap().to_string(),
                StoreInitializationPolicy::IfGloballyEmpty,
            )]
        );
        assert_eq!(journal.durable_media_capabilities(), None);
    }

    /// The reason a store refused to open is the operator's whole diagnosis.
    /// It must survive the startup log line that reports it, so it is retained
    /// on the journal for `/api/status` to serve for the life of the process.
    #[test]
    fn a_refused_store_retains_its_reason_for_the_life_of_the_process() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let adapter = FakeAdapter::default();
        adapter.set_open_error(
            &owner,
            "/srv/mold/queue-media must be a current-user-owned 0700 directory: found mode 0770 \
             (expected 0700); repair with: chmod 0700 /srv/mold/queue-media",
        );

        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();
        assert!(!report.durable_media_ready);

        let status = journal.durable_media_status(true).expect("a claimed owner");
        assert!(!status.available);
        assert!(
            status
                .reasons
                .iter()
                .any(|reason| reason.contains("chmod 0700 /srv/mold/queue-media")),
            "{:?}",
            status.reasons
        );
    }

    /// Held jobs are the half a bare issue list misses: reconciliation can end
    /// with no issue at all and still quarantine work whose media is gone.
    #[test]
    fn held_media_jobs_are_reported_as_a_reason_even_with_no_issue() {
        let mut report = StartupReport {
            durable_media_ready: false,
            held_jobs: vec!["job-a".to_string()],
            ..StartupReport::default()
        };
        let reasons = report.degradation_reasons();
        assert_eq!(reasons.len(), 1);
        assert!(reasons[0].contains("job-a"), "{reasons:?}");

        report.durable_media_ready = true;
        assert!(report.degradation_reasons().is_empty());
    }

    /// The durable backlog is deliberately not bounded by `queue_capacity`, so
    /// a reason retained for the life of the process must not name every held
    /// job: one unusable store would otherwise put an unbounded string on
    /// every `/api/status` response.
    #[test]
    fn a_large_held_backlog_is_summarized_rather_than_enumerated() {
        let report = StartupReport {
            durable_media_ready: false,
            held_jobs: (0..500).map(|index| format!("job-{index}")).collect(),
            ..StartupReport::default()
        };
        let reasons = report.degradation_reasons();
        assert_eq!(reasons.len(), 1);
        assert!(
            reasons[0].starts_with("500 queued job(s) are held"),
            "{reasons:?}"
        );
        assert!(
            reasons[0].contains("job-0, job-1, job-2, job-3, job-4 and 495 more"),
            "{reasons:?}"
        );
        assert!(!reasons[0].contains("job-499"), "{reasons:?}");
        assert!(reasons[0].len() < 256, "{reasons:?}");
    }

    /// `issues` carries one entry per inconsistent obligation, so a fleet-wide
    /// problem must not become one retained log line and one serialized string
    /// per queued row.
    #[test]
    fn a_per_obligation_issue_storm_is_summarized_rather_than_retained_whole() {
        let report = StartupReport {
            durable_media_ready: false,
            issues: (0..400)
                .map(|index| format!("media set set-{index} has conflicting DB obligations"))
                .collect(),
            ..StartupReport::default()
        };
        let reasons = report.degradation_reasons();
        assert_eq!(reasons.len(), ISSUE_REASON_SAMPLE + 1, "{reasons:?}");
        assert!(reasons[0].contains("set-0"), "{reasons:?}");
        assert_eq!(
            reasons[ISSUE_REASON_SAMPLE],
            "and 395 further queue-media startup issue(s) not listed"
        );
        assert!(
            !reasons.iter().any(|reason| reason.contains("set-399")),
            "{reasons:?}"
        );
    }

    #[test]
    fn missing_key_holds_media_jobs_but_keeps_media_free_queue_durability() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let adapter = FakeAdapter::default();
        adapter.set_obligations(
            &owner,
            vec![MediaObligation {
                job_id: Some("media-job".to_string()),
                set_id: "set-a".to_string(),
                state: ObligationState::Active,
            }],
        );
        adapter.set_open_error(&owner, "owner key is missing");

        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();
        assert_eq!(report.held_jobs, vec!["media-job"]);
        assert_eq!(journal.durable_media_capabilities(), None);
        assert!(
            journal.is_enabled(),
            "media-free durability remains enabled"
        );
        assert!(adapter
            .actions()
            .contains(&(owner, "hold".to_string(), "media-job".to_string())));
        assert_eq!(
            adapter.opens.lock().unwrap()[0].1,
            StoreInitializationPolicy::Deny
        );
    }

    #[test]
    fn gc_is_trigger_driven_and_unknown_entries_remain_untouched() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let adapter = FakeAdapter::default();
        adapter.set_obligations(
            &owner,
            vec![
                MediaObligation {
                    job_id: Some("active".to_string()),
                    set_id: "active-set".to_string(),
                    state: ObligationState::Active,
                },
                MediaObligation {
                    job_id: Some("restored".to_string()),
                    set_id: "retired-active-set".to_string(),
                    state: ObligationState::Active,
                },
                MediaObligation {
                    job_id: None,
                    set_id: "gc-set".to_string(),
                    state: ObligationState::GcPending,
                },
                MediaObligation {
                    job_id: None,
                    set_id: "unsafe-set".to_string(),
                    state: ObligationState::GcPending,
                },
                MediaObligation {
                    job_id: None,
                    set_id: "hintless-set".to_string(),
                    state: ObligationState::GcPending,
                },
            ],
        );
        adapter.set_inspection(
            &owner,
            Ok(StoreInspection {
                entries: vec![
                    StoreEntry {
                        job_id: "active".to_string(),
                        set_id: "active-set".to_string(),
                        state: StoreEntryState::Active,
                    },
                    StoreEntry {
                        job_id: "restored".to_string(),
                        set_id: "retired-active-set".to_string(),
                        state: StoreEntryState::Retired,
                    },
                    StoreEntry {
                        job_id: "gc".to_string(),
                        set_id: "gc-set".to_string(),
                        state: StoreEntryState::Retired,
                    },
                    StoreEntry {
                        job_id: "orphan".to_string(),
                        set_id: "orphan-set".to_string(),
                        state: StoreEntryState::Staging,
                    },
                ],
                untouched: vec![
                    UntouchedEntry {
                        set_id_hint: Some("unsafe-set".to_string()),
                        description: "symlink unsafe-set".to_string(),
                    },
                    UntouchedEntry {
                        set_id_hint: None,
                        description: "malformed entry without a safe hint".to_string(),
                    },
                ],
            }),
        );

        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();
        assert!(!report.durable_media_ready);
        let actions = adapter.actions();
        assert!(actions.contains(&(
            owner.clone(),
            "restore".to_string(),
            "retired-active-set".to_string()
        )));
        assert!(!actions.iter().any(|(_, verb, value)| {
            value == "active-set" && (verb == "delete" || verb == "hold")
        }));
        assert!(actions.contains(&(owner.clone(), "delete".to_string(), "gc-set".to_string())));
        assert!(actions.contains(&(owner.clone(), "clear-gc".to_string(), "gc-set".to_string())));
        assert!(!actions.iter().any(|(_, _, value)| value == "orphan-set"));
        assert!(!actions.iter().any(|(_, _, value)| value == "unsafe-set"));
        assert!(!actions.iter().any(|(_, _, value)| value == "hintless-set"));
    }

    #[test]
    fn sole_owner_port_adoption_reconciles_the_adopted_owner() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let first = journal(home.path(), db.clone(), "instance-port-7680");
        let owner = first.owner_uuid().unwrap().to_string();
        generation_queue::insert(db.as_ref().as_ref().unwrap(), &queue_row(&owner, "job-a"))
            .unwrap();
        drop(first);

        let adopted = journal(home.path(), db, "instance-port-7681");
        assert_eq!(adopted.owner_uuid(), Some(owner.as_str()));
        let adapter = FakeAdapter::default();
        adapter.set_inspection(&owner, Ok(StoreInspection::default()));
        let report = reconcile_claimed_owner(&adopted, &adapter).unwrap();
        assert_eq!(report.owner_uuid, Some(owner));
    }

    #[test]
    fn one_live_owner_never_inspects_or_mutates_a_peer_root() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let first = journal(home.path(), db.clone(), "instance-a");
        let second = journal(home.path(), db, "instance-b");
        let first_owner = first.owner_uuid().unwrap().to_string();
        let second_owner = second.owner_uuid().unwrap().to_string();
        assert_ne!(first_owner, second_owner);

        let adapter = FakeAdapter::default();
        adapter.set_obligations(
            &first_owner,
            vec![MediaObligation {
                job_id: None,
                set_id: "foreign-set".to_string(),
                state: ObligationState::GcPending,
            }],
        );
        adapter.set_obligations(&second_owner, Vec::new());
        adapter.set_unclaimed_owner_roots(
            &second_owner,
            vec![UnclaimedOwnerRoot {
                owner_id_hint: Some(first_owner.clone()),
                description: "live or unadopted peer root".to_string(),
            }],
        );
        let report = reconcile_claimed_owner(&second, &adapter).unwrap();
        assert_eq!(report.unclaimed_owner_roots.len(), 1);
        assert!(report.durable_media_ready);
        assert!(adapter
            .actions()
            .iter()
            .all(|(owner, _, _)| owner == &second_owner));
    }

    #[test]
    fn conflicting_db_obligations_hold_active_jobs_and_never_delete() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let adapter = FakeAdapter::default();
        adapter.set_obligations(
            &owner,
            vec![
                MediaObligation {
                    job_id: Some("active-job".to_string()),
                    set_id: "shared-set".to_string(),
                    state: ObligationState::Active,
                },
                MediaObligation {
                    job_id: None,
                    set_id: "shared-set".to_string(),
                    state: ObligationState::GcPending,
                },
                MediaObligation {
                    job_id: Some("ambiguous-job".to_string()),
                    set_id: "first-set".to_string(),
                    state: ObligationState::Active,
                },
                MediaObligation {
                    job_id: Some("ambiguous-job".to_string()),
                    set_id: "second-set".to_string(),
                    state: ObligationState::Active,
                },
            ],
        );
        adapter.set_inspection(
            &owner,
            Ok(StoreInspection {
                entries: ["shared-set", "first-set", "second-set"]
                    .into_iter()
                    .map(|set_id| StoreEntry {
                        job_id: format!("job-for-{set_id}"),
                        set_id: set_id.to_string(),
                        state: StoreEntryState::Active,
                    })
                    .collect(),
                untouched: Vec::new(),
            }),
        );

        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();
        assert!(!report.durable_media_ready);
        assert_eq!(
            report.held_jobs,
            vec!["active-job".to_string(), "ambiguous-job".to_string()]
        );
        let actions = adapter.actions();
        assert!(actions.contains(&(owner.clone(), "hold".to_string(), "active-job".to_string())));
        assert!(actions.contains(&(owner, "hold".to_string(), "ambiguous-job".to_string())));
        assert!(actions
            .iter()
            .all(|(_, verb, _)| verb != "delete" && verb != "clear-gc"));
        assert_eq!(journal.durable_media_capabilities(), None);
    }

    #[test]
    fn projection_disagreement_holds_before_orphan_cleanup() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let adapter = FakeAdapter::default();
        adapter.set_obligations(
            &owner,
            vec![MediaObligation {
                job_id: Some("media-job".to_string()),
                set_id: "expected-set".to_string(),
                state: ObligationState::Active,
            }],
        );
        adapter.set_inspection(
            &owner,
            Ok(StoreInspection {
                entries: vec![StoreEntry {
                    job_id: "apparently-unreferenced-job".to_string(),
                    set_id: "apparently-unreferenced-set".to_string(),
                    state: StoreEntryState::Active,
                }],
                untouched: Vec::new(),
            }),
        );

        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();
        assert_eq!(report.held_jobs, vec!["media-job"]);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.contains("expected-set")));
        assert!(adapter
            .actions()
            .iter()
            .all(|(_, verb, _)| verb != "delete"));
        assert_eq!(journal.durable_media_capabilities(), None);
    }

    #[test]
    fn authenticated_store_job_mismatch_holds_active_and_retired_obligations() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = journal(home.path(), db, "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let adapter = FakeAdapter::default();
        adapter.set_obligations(
            &owner,
            vec![
                MediaObligation {
                    job_id: Some("active-job".to_string()),
                    set_id: "active-set".to_string(),
                    state: ObligationState::Active,
                },
                MediaObligation {
                    job_id: Some("restore-job".to_string()),
                    set_id: "retired-set".to_string(),
                    state: ObligationState::Active,
                },
            ],
        );
        adapter.set_inspection(
            &owner,
            Ok(StoreInspection {
                entries: vec![
                    StoreEntry {
                        job_id: "different-active-job".to_string(),
                        set_id: "active-set".to_string(),
                        state: StoreEntryState::Active,
                    },
                    StoreEntry {
                        job_id: "different-retired-job".to_string(),
                        set_id: "retired-set".to_string(),
                        state: StoreEntryState::Retired,
                    },
                ],
                untouched: Vec::new(),
            }),
        );

        let report = reconcile_claimed_owner(&journal, &adapter).unwrap();

        assert_eq!(
            report.held_jobs,
            vec!["active-job".to_string(), "restore-job".to_string()]
        );
        assert_eq!(report.issues.len(), 2);
        assert!(report
            .issues
            .iter()
            .all(|issue| issue.contains("authenticated bundle belongs to job")));
        assert!(adapter
            .actions()
            .iter()
            .all(|(_, verb, _)| verb != "restore" && verb != "delete"));
        assert_eq!(journal.durable_media_capabilities(), None);
    }
}
