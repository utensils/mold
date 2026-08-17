//! Durable admission journal for public singleton generations.
//!
//! One row per admitted job, written **before** `submit()` and removed by an
//! RAII [`QueueTicket`] on every ordinary terminal path. The row survives only
//! when the process is dying: [`QueueJournal::retain_all`] raises a global
//! fence that turns every subsequent ticket drop into a retention, so whatever
//! the scheduler discards on the way out is replayed on the next boot.
//!
//! **The fence must be raised by every abnormal-exit initiator, not just
//! SIGTERM.** Fatal CUDA is the restart mold deliberately performs on itself;
//! a fence that only covered the signal handler would delete the entire queue
//! on exactly that path. The three call sites are `begin_runtime_shutdown`,
//! the fatal-CUDA notifier, and `quarantine_poisoned_worker`.
//!
//! Rows never carry a secret. Reference-upload handles and resolved reference
//! paths are excluded at admission rather than redacted here, which is why
//! [`QueueJournal::record`] refuses any request carrying them.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use mold_db::generation_queue::{self, GenerationQueueRow, QueueRowState};
use mold_db::MetadataDb;

use crate::state::SseCompletionPayload;

/// Turns the journal off entirely. The server still runs every job; nothing
/// survives a restart.
pub const JOURNAL_DISABLE_ENV: &str = "MOLD_QUEUE_JOURNAL_DISABLE";
/// Serialized-request ceiling. A larger request runs normally and is
/// advertised `durable: false` — never silently half-durable.
pub const JOURNAL_MAX_BYTES_ENV: &str = "MOLD_QUEUE_JOURNAL_MAX_BYTES";
/// How many times a row may be claimed for execution before it is held.
pub const MAX_DISPATCH_ATTEMPTS_ENV: &str = "MOLD_QUEUE_MAX_DISPATCH_ATTEMPTS";
/// How many boots may replay a row that is never claimed before it is held.
pub const MAX_REPLAY_SEEN_ENV: &str = "MOLD_QUEUE_MAX_REPLAY_SEEN";

/// 32 MiB of serialized request. Comfortably holds an inline source image;
/// an inline 4K video does not belong in a SQLite row.
pub const DEFAULT_JOURNAL_MAX_BYTES: usize = 32 * 1024 * 1024;
/// A job that kills the process during its own load is held on the third boot.
pub const DEFAULT_MAX_DISPATCH_ATTEMPTS: u32 = 2;
/// Sized for a 5 s `RestartSec` crash loop, which is the only way a queued row
/// loops without ever being claimed. Ordinary deploys never approach it.
pub const DEFAULT_MAX_REPLAY_SEEN: u32 = 10;

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(env = name, raw = %raw, %error, "ignoring unparseable value");
                default
            }
        },
        Err(_) => default,
    }
}

fn env_u32(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().parse::<u32>() {
            Ok(value) if value >= 1 => value,
            Ok(_) => {
                tracing::warn!(env = name, "cap must be at least 1; using the default");
                default
            }
            Err(error) => {
                tracing::warn!(env = name, raw = %raw, %error, "ignoring unparseable value");
                default
            }
        },
        Err(_) => default,
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|raw| {
        let raw = raw.trim();
        !raw.is_empty() && raw != "0" && !raw.eq_ignore_ascii_case("false")
    })
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Everything the journal needs to persist one admitted job.
pub struct JournalAdmission<'a> {
    pub id: &'a str,
    pub request: &'a mold_core::GenerateRequest,
    /// `None` means no gallery target, and therefore nothing worth replaying:
    /// the only delivery is the HTTP response, which by definition does not
    /// survive the restart.
    pub output_dir: Option<&'a Path>,
    pub target_gpu: Option<usize>,
    pub completion_payload: SseCompletionPayload,
    /// True when the job is a server-owned adaptive-batch child; those are
    /// owned by the batch transaction's own durable recovery.
    pub batch_child: bool,
    /// True when the request carries reference-upload authority. Those bytes
    /// are bearer secrets staged outside the DB and are never journaled.
    pub carries_reference_authority: bool,
}

/// Directory of per-identity claim locks, one file per queue owner.
const QUEUE_OWNERS_DIR: &str = "queue-owners";

/// A queue identity held for this process's lifetime.
///
/// The identity is port-independent so a server that comes back on a different
/// port still adopts its own rows — but two servers can legitimately share one
/// `MOLD_HOME`, and a shared identity would let the second to start adopt the
/// first's queued *and running* rows, requeue work that is actively rendering,
/// and run the same job twice at once.
///
/// So identities are claimed rather than derived: each is a lock file, and a
/// booting server adopts the first one nobody holds. A restart finds its own
/// lock free and reclaims it; a live peer's lock is held, so it is skipped and
/// a fresh identity is minted. Two simultaneous first starts cannot collide,
/// because neither can see a directory entry the other has not created yet and
/// each mints its own — there is no shared setting to race on.
pub struct QueueOwnerClaim {
    owner_uuid: String,
    _lock: std::fs::File,
}

impl QueueOwnerClaim {
    pub fn owner_uuid(&self) -> &str {
        &self.owner_uuid
    }
}

/// Take a queue identity for this process, or `None` when one cannot be held.
///
/// Failing to lock means the filesystem cannot give us exclusive ownership, and
/// an identity we cannot fence is one a peer might share — so the journal is
/// disabled rather than risk two servers replaying each other's work.
pub fn claim_queue_owner(mold_dir: &Path) -> Option<QueueOwnerClaim> {
    let owners = mold_dir.join(QUEUE_OWNERS_DIR);
    if let Err(error) = std::fs::create_dir_all(&owners) {
        tracing::warn!(
            dir = %owners.display(),
            %error,
            "durable generation queue unavailable: could not create its identity directory"
        );
        return None;
    }

    // Deterministic order so a restarting pair reclaims predictably.
    let mut existing: Vec<String> = std::fs::read_dir(&owners)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .strip_suffix(".lock")
                .map(str::to_string)
        })
        .collect();
    existing.sort();

    for owner_uuid in existing {
        let path = owners.join(format!("{owner_uuid}.lock"));
        let Ok(file) = std::fs::OpenOptions::new().write(true).open(&path) else {
            continue;
        };
        if fs2::FileExt::try_lock_exclusive(&file).is_ok() {
            tracing::debug!(owner = %owner_uuid, "adopted an existing durable queue identity");
            return Some(QueueOwnerClaim {
                owner_uuid,
                _lock: file,
            });
        }
        // Held by a live peer sharing this MOLD_HOME.
    }

    let owner_uuid = uuid::Uuid::new_v4().to_string();
    let path = owners.join(format!("{owner_uuid}.lock"));
    let file = match std::fs::File::create(&path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "durable generation queue unavailable: could not create its identity lock"
            );
            return None;
        }
    };
    if fs2::FileExt::try_lock_exclusive(&file).is_err() {
        tracing::warn!(
            path = %path.display(),
            "durable generation queue unavailable: this filesystem cannot fence its identity"
        );
        return None;
    }
    tracing::debug!(owner = %owner_uuid, "minted a durable queue identity");
    Some(QueueOwnerClaim {
        owner_uuid,
        _lock: file,
    })
}

pub struct QueueJournal {
    db: Arc<Option<MetadataDb>>,
    owner_uuid: Option<String>,
    #[cfg(test)]
    fail_completion_lookup: AtomicBool,
    /// Held for the process's lifetime so a peer sharing this `MOLD_HOME`
    /// cannot adopt the same identity.
    _owner_claim: Option<QueueOwnerClaim>,
    retain: AtomicBool,
    max_bytes: usize,
    max_dispatch_attempts: u32,
    max_replay_seen: u32,
}

impl QueueJournal {
    /// Build the journal for a running server. Returns a disabled journal when
    /// the DB is unavailable or `MOLD_QUEUE_JOURNAL_DISABLE` is set — the
    /// server still runs every job, it just cannot promise replay.
    pub fn new(db: Arc<Option<MetadataDb>>, mold_dir: Option<&Path>) -> Self {
        let claim = if env_flag(JOURNAL_DISABLE_ENV) {
            tracing::info!("durable generation queue disabled by environment");
            None
        } else if db.as_ref().is_none() {
            None
        } else {
            match mold_dir {
                Some(dir) => claim_queue_owner(dir),
                None => {
                    tracing::warn!(
                        "durable generation queue unavailable: MOLD_HOME could not be resolved"
                    );
                    None
                }
            }
        };
        Self {
            db,
            owner_uuid: claim.as_ref().map(|claim| claim.owner_uuid.clone()),
            _owner_claim: claim,
            #[cfg(test)]
            fail_completion_lookup: AtomicBool::new(false),
            retain: AtomicBool::new(false),
            max_bytes: env_usize(JOURNAL_MAX_BYTES_ENV, DEFAULT_JOURNAL_MAX_BYTES),
            max_dispatch_attempts: env_u32(
                MAX_DISPATCH_ATTEMPTS_ENV,
                DEFAULT_MAX_DISPATCH_ATTEMPTS,
            ),
            max_replay_seen: env_u32(MAX_REPLAY_SEEN_ENV, DEFAULT_MAX_REPLAY_SEEN),
        }
    }

    /// A journal that persists nothing. Used by tests and by every runtime
    /// where the metadata DB is absent.
    pub fn disabled() -> Self {
        Self {
            db: Arc::new(None),
            owner_uuid: None,
            _owner_claim: None,
            #[cfg(test)]
            fail_completion_lookup: AtomicBool::new(false),
            retain: AtomicBool::new(false),
            max_bytes: DEFAULT_JOURNAL_MAX_BYTES,
            max_dispatch_attempts: DEFAULT_MAX_DISPATCH_ATTEMPTS,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
        }
    }

    /// Whether this server can promise that a queued job survives a restart.
    /// Backs `QueueCapabilities.durable_queue`.
    pub fn is_enabled(&self) -> bool {
        self.owner_uuid.is_some()
    }

    pub fn owner_uuid(&self) -> Option<&str> {
        self.owner_uuid.as_deref()
    }

    pub fn max_dispatch_attempts(&self) -> u32 {
        self.max_dispatch_attempts
    }

    pub fn max_replay_seen(&self) -> u32 {
        self.max_replay_seen
    }

    fn db(&self) -> Option<&MetadataDb> {
        self.db.as_ref().as_ref()
    }

    /// Raise the retention fence. Every ticket dropped from here on keeps its
    /// row, so the scheduler's ordinary discard-on-shutdown becomes a resume
    /// list. Idempotent, and safe to call from any thread.
    pub fn retain_all(&self) {
        if !self.retain.swap(true, Ordering::SeqCst) {
            tracing::info!("retaining the durable generation queue for replay");
        }
    }

    pub fn is_retaining(&self) -> bool {
        self.retain.load(Ordering::SeqCst)
    }

    /// Persist an admitted job, returning the ticket that owns its row.
    ///
    /// `None` means the job is not durable. Every reason is deliberate:
    /// no gallery target, reference-upload authority (bearer secrets), a batch
    /// child (owned by the batch transaction's own recovery), an oversized
    /// payload, or no journal at all.
    pub fn record(self: &Arc<Self>, admission: JournalAdmission<'_>) -> Option<QueueTicket> {
        let owner_uuid = self.owner_uuid.as_deref()?;
        let db = self.db()?;
        let output_dir = admission.output_dir?;
        if admission.batch_child || admission.carries_reference_authority {
            return None;
        }
        let request_json = match serde_json::to_string(admission.request) {
            Ok(json) => json,
            Err(error) => {
                tracing::warn!(
                    job = %admission.id,
                    error = %error,
                    "generation is not durable: its request could not be serialized"
                );
                return None;
            }
        };
        if request_json.len() > self.max_bytes {
            tracing::info!(
                job = %admission.id,
                bytes = request_json.len(),
                max_bytes = self.max_bytes,
                "generation is not durable: its request exceeds the journal payload ceiling"
            );
            return None;
        }

        let now = now_ms();
        let row = GenerationQueueRow {
            id: admission.id.to_string(),
            owner_uuid: owner_uuid.to_string(),
            state: QueueRowState::Queued,
            model: admission.request.model.clone(),
            request_json,
            output_dir: output_dir.to_path_buf(),
            target_gpu: admission.target_gpu,
            completion_payload: completion_payload_as_str(admission.completion_payload).to_string(),
            seed_pinned: admission.request.seed.is_some(),
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms: now,
            updated_at_ms: now,
            started_at_ms: None,
        };
        if let Err(error) = generation_queue::insert(db, &row) {
            tracing::warn!(
                job = %admission.id,
                error = %format!("{error:#}"),
                "generation is not durable: the journal row could not be written"
            );
            return None;
        }
        Some(QueueTicket {
            journal: Arc::clone(self),
            id: admission.id.to_string(),
            settled: false,
        })
    }

    /// Re-attach a ticket to a row that already exists. Used by replay, which
    /// resubmits an existing row rather than writing a new one.
    pub fn attach(self: &Arc<Self>, id: &str) -> QueueTicket {
        QueueTicket {
            journal: Arc::clone(self),
            id: id.to_string(),
            settled: false,
        }
    }

    /// Drop one row regardless of the fence. The cancellation path: a job the
    /// user explicitly removed must not come back after a restart, even when
    /// the cancel lands during the drain.
    pub fn discard_id(&self, id: &str) {
        let Some(db) = self.db() else {
            return;
        };
        if let Err(error) = generation_queue::delete(db, id) {
            tracing::warn!(
                job = %id,
                error = %format!("{error:#}"),
                "could not remove a cancelled job from the durable queue"
            );
        }
    }

    /// Drop every still-queued row this server owns. Backs `DELETE /api/queue`.
    pub fn discard_all_queued(&self) {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return;
        };
        if let Err(error) = generation_queue::delete_all_queued(db, owner) {
            tracing::warn!(
                error = %format!("{error:#}"),
                "could not clear the durable queue after a bulk cancel"
            );
        }
    }

    /// Reconcile the journal against reality before anything is replayed.
    ///
    /// A `running` row means the process died mid-dispatch, so the state
    /// column is rewritten to say what to do next. A row whose output
    /// directory no longer exists is re-pointed at the currently configured
    /// gallery, and held if even that is unusable — a replay whose output has
    /// nowhere to land is a wasted render.
    pub fn startup_reconcile(&self, current_output_dir: Option<&Path>) -> ReconcileReport {
        let mut report = ReconcileReport::default();
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return report;
        };
        let now = now_ms();
        match generation_queue::requeue_running(db, owner, now) {
            Ok(count) => report.requeued = count,
            Err(error) => tracing::warn!(
                error = %format!("{error:#}"),
                "could not requeue interrupted durable generations"
            ),
        }
        for row in self.list_all() {
            if row.state == QueueRowState::Held {
                report.held += 1;
                continue;
            }
            if row.output_dir.is_dir() {
                continue;
            }
            match current_output_dir {
                Some(replacement) if replacement != row.output_dir => {
                    tracing::warn!(
                        job = %row.id,
                        was = %row.output_dir.display(),
                        now = %replacement.display(),
                        "a retained generation's gallery directory moved; re-pointing it"
                    );
                    let replacement = replacement.to_string_lossy().into_owned();
                    if let Err(error) =
                        generation_queue::set_output_dir(db, &row.id, &replacement, now)
                    {
                        tracing::warn!(
                            job = %row.id,
                            error = %format!("{error:#}"),
                            "could not re-point a retained generation"
                        );
                    } else {
                        report.repointed += 1;
                    }
                }
                _ => {
                    let _ = generation_queue::hold(
                        db,
                        &row.id,
                        "the gallery directory this job was admitted for no longer exists",
                        now,
                    );
                    report.held += 1;
                }
            }
        }
        report
    }

    /// Test seam: force the idempotence gate to fail the way a malformed
    /// `metadata_json` or a transient SQLite error would.
    #[cfg(test)]
    pub(crate) fn fail_completion_lookup_for_tests(&self) {
        self.fail_completion_lookup
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Drop every row whose output already exists.
    ///
    /// A print records the queue job that produced it, so a job that finished
    /// between its last save and the crash is recognised and never re-run.
    /// Without this, replay duplicates prints: output filenames are wall-clock,
    /// so no downstream dedupe can merge the two afterwards.
    pub fn drop_already_completed(&self) -> Result<usize, String> {
        let Some(db) = self.db() else {
            return Ok(0);
        };
        #[cfg(test)]
        if self
            .fail_completion_lookup
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err("injected completion-lookup failure".to_string());
        }
        let candidates: Vec<String> = self
            .list_all()
            .into_iter()
            .filter(|row| row.state != QueueRowState::Held)
            .map(|row| row.id)
            .collect();
        // A failure here is NOT "nothing was completed". Reporting it as an
        // empty result would let replay re-render every job whose output was
        // already published, so one malformed `metadata_json` or a transient
        // SQLite error would defeat the idempotence guarantee outright.
        let completed = generation_queue::find_completed_job_ids(db, &candidates)
            .map_err(|error| format!("{error:#}"))?;
        for id in &completed {
            tracing::info!(job = %id, "a retained generation already produced its print");
            self.discard_id(id);
        }
        Ok(completed.len())
    }

    /// Rows to replay, oldest first, after reconcile and the idempotence gate.
    pub fn replayable(&self) -> Vec<GenerationQueueRow> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Vec::new();
        };
        generation_queue::list_replayable(db, owner).unwrap_or_else(|error| {
            tracing::warn!(
                error = %format!("{error:#}"),
                "could not read the durable generation queue"
            );
            Vec::new()
        })
    }

    /// Charge this boot against a row. `Err` carries the reason it was held.
    pub fn charge_replay(&self, id: &str) -> Result<(), String> {
        let Some(db) = self.db() else {
            return Ok(());
        };
        let now = now_ms();
        match generation_queue::bump_replay_seen(db, id, now) {
            Ok(Some(seen)) if seen > self.max_replay_seen => {
                let cap = self.max_replay_seen;
                let reason = format!("replayed by {seen} boots without ever running (limit {cap})");
                let _ = generation_queue::hold(db, id, &reason, now);
                Err(reason)
            }
            Ok(_) => Ok(()),
            Err(error) => {
                tracing::warn!(
                    job = %id,
                    error = %format!("{error:#}"),
                    "could not charge a replay against a durable queue row"
                );
                Ok(())
            }
        }
    }

    /// Mirror an authoritative queue mutation into the durable row.
    ///
    /// `PATCH /api/queue/:id` owns the lane and the dispatch order, so without
    /// this a restart silently restores the admission-time lane — possibly the
    /// GPU the user explicitly moved the job away from — and the original FIFO
    /// position. `order` is the registry's post-mutation order; passing it
    /// wholesale also repairs any drift.
    pub fn apply_queue_mutation(
        &self,
        id: &str,
        target_gpu: Option<Option<usize>>,
        order: Option<&[String]>,
    ) {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return;
        };
        let now = now_ms();
        if let Some(target_gpu) = target_gpu {
            if let Err(error) = generation_queue::set_target_gpu(db, id, target_gpu, now) {
                tracing::warn!(
                    job = %id,
                    error = %format!("{error:#}"),
                    "could not re-lane a durable queue row"
                );
            }
        }
        if let Some(order) = order {
            if let Err(error) = generation_queue::apply_queue_order(db, owner, order) {
                tracing::warn!(
                    error = %format!("{error:#}"),
                    "could not persist the new queue order"
                );
            }
        }
    }

    /// Whether this id names a parked row. `DELETE /api/queue/:id` is the
    /// documented way to clear one, and a held job has no registry entry.
    pub fn is_held(&self, id: &str) -> bool {
        let Some(db) = self.db() else {
            return false;
        };
        generation_queue::get(db, id)
            .ok()
            .flatten()
            .is_some_and(|row| row.state == QueueRowState::Held)
    }

    /// Park a row by id, for a caller that has no ticket.
    pub fn hold_id(&self, id: &str, reason: &str) {
        let Some(db) = self.db() else {
            return;
        };
        if let Err(error) = generation_queue::hold(db, id, reason, now_ms()) {
            tracing::warn!(
                job = %id,
                error = %format!("{error:#}"),
                "could not hold a durable queue row"
            );
        }
    }

    /// Rows this server owns, oldest first. Backs the `held` listing.
    pub fn list_all(&self) -> Vec<GenerationQueueRow> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Vec::new();
        };
        generation_queue::list_all(db, owner).unwrap_or_else(|error| {
            tracing::warn!(
                error = %format!("{error:#}"),
                "could not read the durable generation queue"
            );
            Vec::new()
        })
    }
}

/// What `startup_reconcile` did, for one startup log line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileReport {
    pub requeued: usize,
    pub repointed: usize,
    pub held: usize,
}

/// What one boot's replay admitted, for one startup log line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReplayReport {
    pub resumed: usize,
    pub already_completed: usize,
    pub held: usize,
    /// Rows left untouched because the idempotence gate could not be checked.
    /// They keep their full replay budget for the next boot.
    pub skipped_unverified: usize,
}

/// Resubmit every retained generation through the ordinary admission path.
///
/// **Sequential on purpose.** `submit_when_available` serializes on one global
/// `capacity_waiter` mutex, so parallel replay tasks would arrive in arbitrary
/// order and destroy the ordering the journal exists to preserve. A journal
/// deeper than `queue_capacity` therefore blocks here in order rather than
/// racing, which is why this runs before the router starts serving.
///
/// Replay reuses the original job id, which also emits `ServerEvent::JobQueued`
/// for free, so `/api/events` subscribers see resumed jobs with no new event
/// type. Everything downstream — validation, admission, frozen plans,
/// placement, auto-pull — applies unchanged: a model uninstalled between boots
/// is blocked as `model_not_installed`, not silently rerouted.
pub async fn replay(state: &crate::state::AppState, dispatch_available: bool) -> ReplayReport {
    let journal = state.queue_journal.clone();
    let mut report = ReplayReport::default();
    if !journal.is_enabled() {
        return report;
    }
    // A maintenance boot has no dispatch owner, so there is nothing to replay
    // INTO — `run_server` has already dropped the queue receiver. Attempting
    // it anyway would fail every send, and a failed send used to drop the
    // ticket with this boot's fence still down, deleting the whole queue on a
    // routine `MOLD_GPUS=none` restart. Skipped before anything is charged, so
    // the rows keep their full replay budget for a boot that can run them.
    if !dispatch_available {
        let retained = journal.replayable().len();
        if retained > 0 {
            tracing::info!(
                retained,
                "generation is unavailable on this boot; retained jobs stay queued for the next one"
            );
        }
        return report;
    }

    let current_output_dir = {
        let config = state.config.read().await;
        if state.is_output_disabled(&config) {
            None
        } else {
            Some(config.effective_output_dir())
        }
    };
    let reconcile = journal.startup_reconcile(current_output_dir.as_deref());
    report.held += reconcile.held;
    match journal.drop_already_completed() {
        Ok(dropped) => report.already_completed = dropped,
        Err(error) => {
            // Skip the whole boot's replay rather than replay blind. The rows
            // are untouched and unspent, so the next boot tries again; running
            // them now could duplicate prints that already exist, and a
            // duplicate is unmergeable because output filenames are wall-clock.
            tracing::error!(
                %error,
                "could not check the durable queue against saved prints; \
                 skipping replay this boot so nothing is rendered twice"
            );
            report.skipped_unverified = journal.replayable().len();
            return report;
        }
    }

    let cancellation = tokio_util::sync::CancellationToken::new();
    for row in journal.replayable() {
        if let Err(reason) = journal.charge_replay(&row.id) {
            tracing::warn!(job = %row.id, %reason, "holding a durable queue row");
            report.held += 1;
            continue;
        }
        let request: mold_core::GenerateRequest = match serde_json::from_str(&row.request_json) {
            Ok(request) => request,
            Err(error) => {
                // Fail closed: a request this build cannot read must not be
                // guessed at, and must not be silently discarded either.
                tracing::warn!(
                    job = %row.id,
                    %error,
                    "holding a durable queue row whose request this build cannot read"
                );
                journal.hold_id(&row.id, "the recorded request could not be deserialized");
                report.held += 1;
                continue;
            }
        };

        let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
            &request,
            request.seed.unwrap_or(0),
            request.scheduler,
            mold_core::build_info::version_string(),
        ));
        let cancel = state.job_registry.register_job(
            &row.id,
            &row.model,
            row.target_gpu,
            Some(row.seed_pinned),
            Some(metadata),
        );
        // The supervisor is what keeps `result_tx` open for a job with no
        // client at all — without it every `is_closed()` gate would skip a
        // replayed job on sight.
        let crate::job_supervisor::SupervisedJob {
            result_tx,
            outcome_rx,
        } = crate::job_supervisor::supervise_job(row.id.clone(), cancel);
        let replayed_id = row.id.clone();
        tokio::spawn(async move {
            match outcome_rx.await {
                Ok(crate::job_supervisor::SupervisedOutcome::Finished(outcome)) => match *outcome {
                    Ok(_) => tracing::info!(job = %replayed_id, "resumed generation finished"),
                    Err(error) => {
                        tracing::warn!(job = %replayed_id, %error, "resumed generation failed")
                    }
                },
                Ok(crate::job_supervisor::SupervisedOutcome::Cancelled) => {
                    tracing::info!(job = %replayed_id, "resumed generation cancelled")
                }
                Err(_) => {}
            }
        });

        let job = crate::state::GenerationJob {
            id: row.id.clone(),
            request,
            resolved_references: None,
            completion_payload: completion_payload_from_str(&row.completion_payload),
            // No client to stream to. The output still lands in the gallery,
            // which is the whole reason the row was durable.
            progress_tx: None,
            result_tx,
            output_dir: Some(row.output_dir.clone()),
            batch_child: None,
            journal: Some(journal.attach(&row.id)),
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };

        let mut pending = Some(job);
        match state
            .queue
            .submit_when_available(&mut pending, state.queue_capacity, &cancellation)
            .await
        {
            Ok(_) => report.resumed += 1,
            Err(error) => {
                tracing::warn!(
                    job = %row.id,
                    ?error,
                    "could not resubmit a retained generation; it stays in the journal"
                );
                // Settle the ticket explicitly. Dropping it would delete a row
                // for work that never reached a worker — the opposite of what
                // the log line above promises.
                if let Some(ticket) = pending.take().and_then(|job| job.journal) {
                    ticket.retain();
                }
                state.job_registry.remove(&row.id);
            }
        }
    }
    report
}

fn completion_payload_as_str(payload: SseCompletionPayload) -> &'static str {
    match payload {
        SseCompletionPayload::Full => "full",
        SseCompletionPayload::MetadataOnly => "metadata_only",
    }
}

pub fn completion_payload_from_str(raw: &str) -> SseCompletionPayload {
    match raw {
        "metadata_only" => SseCompletionPayload::MetadataOnly,
        _ => SseCompletionPayload::Full,
    }
}

/// Outcome of charging a dispatch attempt against a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchClaim {
    /// The row was claimed and the job may run.
    Granted,
    /// The row exceeded its dispatch cap and is now held.
    Exhausted { attempts: u32, cap: u32 },
    /// There is no row (journal disabled, or the job was cancelled). The job
    /// still runs — durability is additive, never a gate on execution.
    Untracked,
}

/// RAII owner of one journal row.
///
/// Dropping the ticket deletes the row **unless** the retention fence is up.
/// That inversion is the whole design: every one of the ~20 discard sites in
/// the scheduler and worker already drops the job, so none of them needs to
/// know about durability — they delete a row during normal operation and
/// retain it during shutdown, because the fence, not the call site, decides.
pub struct QueueTicket {
    journal: Arc<QueueJournal>,
    id: String,
    settled: bool,
}

impl QueueTicket {
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The job produced its output. Delete the row unconditionally — a
    /// completed job must never be replayed, fence or no fence.
    pub fn complete(mut self) {
        self.settled = true;
        self.journal.discard_id(&self.id);
    }

    /// The job was explicitly cancelled. Same unconditional delete as
    /// `complete`, and for the same reason: the user's decision outranks the
    /// fence.
    pub fn discard(mut self) {
        self.settled = true;
        self.journal.discard_id(&self.id);
    }

    /// The job never reached a worker, so leave its row exactly as it is.
    ///
    /// Distinct from `hold`: nothing is wrong with the job and the next boot
    /// should replay it normally. The `replay_seen` budget is what stops a row
    /// retrying forever if the condition persists.
    pub fn retain(mut self) {
        self.settled = true;
    }

    /// Charge a dispatch attempt on the GPU owner thread, immediately before
    /// the model load. Over the cap, the row is held rather than deleted.
    pub fn claim_dispatch(&self) -> DispatchClaim {
        let Some(db) = self.journal.db() else {
            return DispatchClaim::Untracked;
        };
        let now = now_ms();
        match generation_queue::mark_dispatched(db, &self.id, now) {
            Ok(Some(attempts)) if attempts > self.journal.max_dispatch_attempts => {
                let cap = self.journal.max_dispatch_attempts;
                let reason =
                    format!("dispatch attempts exhausted ({attempts} > {cap}); held for review");
                if let Err(error) = generation_queue::hold(db, &self.id, &reason, now) {
                    tracing::warn!(
                        job = %self.id,
                        error = %format!("{error:#}"),
                        "could not hold an exhausted durable queue row"
                    );
                }
                DispatchClaim::Exhausted { attempts, cap }
            }
            Ok(Some(_)) => DispatchClaim::Granted,
            Ok(None) => DispatchClaim::Untracked,
            Err(error) => {
                tracing::warn!(
                    job = %self.id,
                    error = %format!("{error:#}"),
                    "could not claim a durable queue row; running the job anyway"
                );
                DispatchClaim::Untracked
            }
        }
    }

    /// Park the row: listed, never auto-run, and no longer owned by a ticket.
    pub fn hold(mut self, reason: &str) {
        self.settled = true;
        let Some(db) = self.journal.db() else {
            return;
        };
        if let Err(error) = generation_queue::hold(db, &self.id, reason, now_ms()) {
            tracing::warn!(
                job = %self.id,
                error = %format!("{error:#}"),
                "could not hold a durable queue row"
            );
        }
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        if self.journal.is_retaining() {
            tracing::info!(
                job = %self.id,
                "retaining an interrupted generation; it will be replayed after restart"
            );
            return;
        }
        self.journal.discard_id(&self.id);
    }
}

impl std::fmt::Debug for QueueTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueTicket").field("id", &self.id).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn journal_with_db() -> Arc<QueueJournal> {
        let db = MetadataDb::open_in_memory().unwrap();
        let owner = "test-owner".to_string();
        Arc::new(QueueJournal {
            db: Arc::new(Some(db)),
            owner_uuid: Some(owner),
            _owner_claim: None,
            fail_completion_lookup: AtomicBool::new(false),
            retain: AtomicBool::new(false),
            max_bytes: DEFAULT_JOURNAL_MAX_BYTES,
            max_dispatch_attempts: DEFAULT_MAX_DISPATCH_ATTEMPTS,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
        })
    }

    fn request() -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q4",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 3.5,
        }))
        .expect("minimal generate request")
    }

    fn admission<'a>(
        id: &'a str,
        request: &'a mold_core::GenerateRequest,
        output_dir: &'a Path,
    ) -> JournalAdmission<'a> {
        JournalAdmission {
            id,
            request,
            output_dir: Some(output_dir),
            target_gpu: None,
            completion_payload: SseCompletionPayload::Full,
            batch_child: false,
            carries_reference_authority: false,
        }
    }

    fn rows(journal: &QueueJournal) -> Vec<String> {
        journal.list_all().into_iter().map(|row| row.id).collect()
    }

    /// Two servers can share one `MOLD_HOME` on different ports. They must
    /// never resolve the same queue identity: the second to start would adopt
    /// the first's queued AND running rows, requeue work that is actively
    /// rendering, and run the same job twice at once.
    #[test]
    fn a_second_live_server_never_adopts_the_first_servers_queue() {
        let home = tempfile::tempdir().unwrap();

        let first = claim_queue_owner(home.path()).expect("first server claims an identity");
        let second = claim_queue_owner(home.path()).expect("second server claims its own");

        assert_ne!(
            first.owner_uuid(),
            second.owner_uuid(),
            "a live peer's identity must never be adopted"
        );
    }

    /// The intentional case the port-independent identity exists for: the same
    /// server stopping and starting again adopts its own rows, including after
    /// a port change.
    #[test]
    fn a_restarted_server_adopts_its_own_queue() {
        let home = tempfile::tempdir().unwrap();

        let first = claim_queue_owner(home.path()).unwrap();
        let owner = first.owner_uuid().to_string();
        drop(first);

        let restarted = claim_queue_owner(home.path()).expect("restart reclaims the identity");
        assert_eq!(restarted.owner_uuid(), owner);
    }

    /// Both peers stopped, then both start: each takes back one identity and
    /// neither is left orphaned.
    #[test]
    fn two_stopped_servers_each_reclaim_one_identity() {
        let home = tempfile::tempdir().unwrap();
        let first = claim_queue_owner(home.path()).unwrap();
        let second = claim_queue_owner(home.path()).unwrap();
        let owners = std::collections::BTreeSet::from([
            first.owner_uuid().to_string(),
            second.owner_uuid().to_string(),
        ]);
        drop(first);
        drop(second);

        let a = claim_queue_owner(home.path()).unwrap();
        let b = claim_queue_owner(home.path()).unwrap();
        assert_eq!(
            std::collections::BTreeSet::from([
                a.owner_uuid().to_string(),
                b.owner_uuid().to_string()
            ]),
            owners,
            "restarting the pair must not mint fresh identities and strand the old rows"
        );
    }

    #[test]
    fn dropping_a_ticket_removes_the_row_during_normal_operation() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .expect("an ordinary gallery-bound generation is durable");
        assert_eq!(rows(&journal), vec!["job-1"]);

        drop(ticket);
        assert!(rows(&journal).is_empty());
    }

    /// The core invariant. Everything the scheduler discards on the way out is
    /// retained, without a single discard site knowing about durability.
    #[test]
    fn dropping_a_ticket_behind_the_fence_retains_the_row() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .unwrap();

        journal.retain_all();
        drop(ticket);

        assert_eq!(rows(&journal), vec!["job-1"]);
    }

    #[test]
    fn an_explicit_cancel_removes_the_row_even_behind_the_fence() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .unwrap();

        journal.retain_all();
        ticket.discard();

        assert!(
            rows(&journal).is_empty(),
            "a cancel during the drain must not resurrect the job after restart"
        );
    }

    #[test]
    fn completion_removes_the_row_even_behind_the_fence() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .unwrap();

        journal.retain_all();
        ticket.complete();

        assert!(rows(&journal).is_empty());
    }

    #[test]
    fn record_refuses_everything_that_must_not_be_journaled() {
        let journal = journal_with_db();
        let request = request();

        let mut no_output = admission("no-output", &request, Path::new("/gallery"));
        no_output.output_dir = None;
        assert!(journal.record(no_output).is_none());

        let mut child = admission("batch-child", &request, Path::new("/gallery"));
        child.batch_child = true;
        assert!(journal.record(child).is_none());

        let mut referenced = admission("with-references", &request, Path::new("/gallery"));
        referenced.carries_reference_authority = true;
        assert!(journal.record(referenced).is_none());

        assert!(rows(&journal).is_empty());
        assert!(QueueJournal::disabled().is_enabled().eq(&false));
    }

    #[test]
    fn an_oversized_request_runs_without_being_journaled() {
        let db = MetadataDb::open_in_memory().unwrap();
        let owner = "test-owner".to_string();
        let journal = Arc::new(QueueJournal {
            db: Arc::new(Some(db)),
            owner_uuid: Some(owner),
            _owner_claim: None,
            fail_completion_lookup: AtomicBool::new(false),
            retain: AtomicBool::new(false),
            max_bytes: 64,
            max_dispatch_attempts: DEFAULT_MAX_DISPATCH_ATTEMPTS,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
        });
        let mut request = request();
        request.prompt = "x".repeat(4096);

        assert!(journal
            .record(admission("huge", &request, Path::new("/gallery")))
            .is_none());
        assert!(rows(&journal).is_empty());
    }

    #[test]
    fn claiming_past_the_dispatch_cap_holds_the_row_instead_of_deleting_it() {
        let db = MetadataDb::open_in_memory().unwrap();
        let owner = "test-owner".to_string();
        let journal = Arc::new(QueueJournal {
            db: Arc::new(Some(db)),
            owner_uuid: Some(owner),
            _owner_claim: None,
            fail_completion_lookup: AtomicBool::new(false),
            retain: AtomicBool::new(false),
            max_bytes: DEFAULT_JOURNAL_MAX_BYTES,
            max_dispatch_attempts: 2,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
        });
        let request = request();
        let ticket = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .unwrap();

        assert_eq!(ticket.claim_dispatch(), DispatchClaim::Granted);
        assert_eq!(ticket.claim_dispatch(), DispatchClaim::Granted);
        assert_eq!(
            ticket.claim_dispatch(),
            DispatchClaim::Exhausted {
                attempts: 3,
                cap: 2
            }
        );

        // Held rows stay listed and are no longer replayable.
        assert_eq!(rows(&journal), vec!["job-1"]);
        let held = journal.list_all().pop().unwrap();
        assert_eq!(held.state, QueueRowState::Held);
        assert!(held.held_reason.is_some());
    }

    #[test]
    fn a_queue_mutation_moves_the_durable_row_with_it() {
        let journal = journal_with_db();
        let request = request();
        let mut tickets = Vec::new();
        for id in ["a", "b", "c"] {
            tickets.push(
                journal
                    .record(admission(id, &request, Path::new("/gallery")))
                    .unwrap(),
            );
        }

        journal.apply_queue_mutation("b", Some(Some(2)), None);
        let relaned = journal
            .list_all()
            .into_iter()
            .find(|row| row.id == "b")
            .unwrap();
        assert_eq!(relaned.target_gpu, Some(2));

        journal.apply_queue_mutation(
            "c",
            None,
            Some(&["c".to_string(), "a".to_string(), "b".to_string()]),
        );
        assert_eq!(rows(&journal), vec!["c", "a", "b"]);

        // Clearing the pin returns the row to Auto rather than leaving it.
        journal.apply_queue_mutation("b", Some(None), None);
        assert_eq!(
            journal
                .list_all()
                .into_iter()
                .find(|row| row.id == "b")
                .unwrap()
                .target_gpu,
            None
        );

        for ticket in tickets {
            ticket.discard();
        }
    }

    #[test]
    fn a_disabled_journal_records_nothing_and_never_claims() {
        let journal = Arc::new(QueueJournal::disabled());
        let request = request();
        assert!(journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .is_none());
        assert_eq!(
            journal.attach("job-1").claim_dispatch(),
            DispatchClaim::Untracked
        );
    }

    /// Reference-upload handles are bearer secrets. The rule is that they are
    /// excluded at admission, not redacted, so nothing resembling one can ever
    /// reach `mold.db`.
    #[test]
    fn a_reference_bearing_request_leaves_no_trace_in_the_database() {
        let journal = journal_with_db();
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "minimax-h3-ref2va",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 3.5,
            "references": [{
                "kind": "image",
                "media": { "authority": "upload", "handle": "super-secret-handle" },
                "mime_type": "image/png",
                "width": 512,
                "height": 512,
            }],
        }))
        .expect("reference-bearing generate request");

        let mut carrying = admission("job-1", &request, Path::new("/gallery"));
        carrying.carries_reference_authority = true;
        assert!(journal.record(carrying).is_none());
        assert!(rows(&journal).is_empty());
    }
}
