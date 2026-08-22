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

/// Whether this request carries a face photograph.
///
/// Derived from the request rather than passed in beside it, deliberately: a
/// caller cannot forget it at a fifth admission site the way it could forget a
/// flag, and there is nothing to resolve — the bytes are either on the request
/// or they are not.
fn carries_identity_photograph(request: &mold_core::GenerateRequest) -> bool {
    // Either wire shape. Asking only about `id_image` would journal every
    // multi-photograph request's faces into `mold.db`, which is the exact
    // outcome this predicate exists to prevent.
    mold_core::identity::request_carries_identity_photo(request)
}

/// Directory of per-identity claim records, one file per queue owner.
const QUEUE_OWNERS_DIR: &str = "queue-owners";

/// Adopt a specific orphaned owner by id. The documented escape hatch for the
/// one case the server cannot decide: several orphans with rows and no hint
/// match.
pub const QUEUE_ADOPT_OWNER_ENV: &str = "MOLD_QUEUE_ADOPT_OWNER";

/// A retained queue nobody is running and this server did not recognise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrphanedOwner {
    pub owner_uuid: String,
    /// The server instance that last held it, when the record names one.
    pub instance_hint: Option<String>,
    pub queued: usize,
    /// Counted separately: a held row that is also orphaned is invisible twice
    /// over, so folding it into the queued total hides it completely.
    pub held: usize,
}

/// A queue identity held for this process's lifetime.
///
/// Identity and liveness are deliberately separate concerns, because conflating
/// them is what four earlier attempts got wrong. The owner id is a minted,
/// port-independent UUID and never changes for the life of that server's queue.
/// The lock file proves only whether that owner is running *right now*; it
/// never confers ownership, so "unlocked" means "not running", not "yours".
///
/// What identifies a returning server is a hint recorded in the record: the
/// instance id it last claimed under. That resolves a requirement which cannot
/// be met from intrinsic state alone — two servers sharing one `MOLD_HOME`
/// differ only by port, so a port-independent id cannot also be per-server
/// distinct — by remembering rather than deriving.
pub struct QueueOwnerClaim {
    owner_uuid: String,
    _lock: std::fs::File,
    /// True when no record matched this instance and exactly one unclaimed
    /// queue existed, so it was taken across an apparent port change.
    pub adopted_across_port_change: bool,
    /// Retained queues this server did not take. Reported at startup so work
    /// is never silently stranded.
    pub orphans: Vec<OrphanedOwner>,
}

impl QueueOwnerClaim {
    pub fn owner_uuid(&self) -> &str {
        &self.owner_uuid
    }
}

struct OwnerCandidate {
    owner_uuid: String,
    instance_hint: Option<String>,
    lock: std::fs::File,
    queued: usize,
    held: usize,
}

impl OwnerCandidate {
    fn has_rows(&self) -> bool {
        self.queued > 0 || self.held > 0
    }

    fn into_orphan(self) -> OrphanedOwner {
        OrphanedOwner {
            owner_uuid: self.owner_uuid,
            instance_hint: self.instance_hint,
            queued: self.queued,
            held: self.held,
        }
    }
}

fn owner_record_path(owners: &Path, owner_uuid: &str) -> std::path::PathBuf {
    owners.join(format!("{owner_uuid}.lock"))
}

/// Record which instance holds this owner now, so the next restart recognises
/// it without guessing.
fn write_instance_hint(file: &std::fs::File, instance_id: &str) {
    use std::io::{Seek, Write};
    let mut file = file;
    let wrote = file
        .set_len(0)
        .and_then(|()| file.seek(std::io::SeekFrom::Start(0)))
        .and_then(|_| file.write_all(instance_id.as_bytes()))
        .and_then(|()| file.flush());
    if let Err(error) = wrote {
        tracing::warn!(%error, "could not record the queue owner's instance hint");
    }
}

/// Take a queue identity for this process, or `None` when one cannot be held.
pub fn claim_queue_owner(
    mold_dir: &Path,
    instance_id: &str,
    db: Option<&MetadataDb>,
) -> Option<QueueOwnerClaim> {
    let requested = std::env::var(QUEUE_ADOPT_OWNER_ENV)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty());
    claim_queue_owner_adopting(mold_dir, instance_id, db, requested.as_deref())
}

/// [`claim_queue_owner`] with the explicit adoption request passed in.
pub fn claim_queue_owner_adopting(
    mold_dir: &Path,
    instance_id: &str,
    db: Option<&MetadataDb>,
    adopt: Option<&str>,
) -> Option<QueueOwnerClaim> {
    let owners = mold_dir.join(QUEUE_OWNERS_DIR);
    if let Err(error) = std::fs::create_dir_all(&owners) {
        tracing::warn!(
            dir = %owners.display(),
            %error,
            "durable generation queue unavailable: could not create its identity directory"
        );
        return None;
    }

    let mut names: Vec<String> = std::fs::read_dir(&owners)
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
    names.sort();

    // Lock what we can. A record we cannot lock belongs to a live peer and is
    // left entirely alone — not read as a candidate, not reported as orphaned.
    let mut candidates: Vec<OwnerCandidate> = Vec::new();
    for owner_uuid in names {
        let path = owner_record_path(&owners, &owner_uuid);
        let Ok(file) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        else {
            continue;
        };
        let instance_hint = std::fs::read_to_string(&path)
            .ok()
            .map(|hint| hint.trim().to_string())
            .filter(|hint| !hint.is_empty());
        if fs2::FileExt::try_lock_exclusive(&file).is_err() {
            continue;
        }
        let (queued, held) = owner_row_counts(db, &owner_uuid);
        candidates.push(OwnerCandidate {
            owner_uuid,
            instance_hint,
            lock: file,
            queued,
            held,
        });
    }

    // 1. An explicit adoption request outranks everything: a human looked.
    let mut chosen = adopt.and_then(|wanted| {
        let found = candidates
            .iter()
            .position(|candidate| candidate.owner_uuid == wanted);
        match found {
            Some(index) => {
                tracing::info!(owner = %wanted, "adopting a queue identity by explicit request");
                Some(candidates.remove(index))
            }
            None => {
                tracing::warn!(
                    owner = %wanted,
                    env = QUEUE_ADOPT_OWNER_ENV,
                    "the requested queue identity is not present or is held by a live server"
                );
                None
            }
        }
    });

    // 2. Our own record, recognised by the instance that last held it. This is
    //    every ordinary restart, including peers restarting in any order.
    let mut adopted_across_port_change = false;
    if chosen.is_none() {
        if let Some(index) = candidates
            .iter()
            .position(|candidate| candidate.instance_hint.as_deref() == Some(instance_id))
        {
            chosen = Some(candidates.remove(index));
        }
    }

    // 3. No record knows us. Exactly one unclaimed queue is unambiguous — the
    //    single-server port change — so take it and say so. More than one and
    //    the machine genuinely cannot tell which is ours: take none.
    if chosen.is_none() {
        let with_rows: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.has_rows())
            .map(|(index, _)| index)
            .collect();
        if with_rows.len() == 1 {
            let candidate = candidates.remove(with_rows[0]);
            tracing::warn!(
                owner = %candidate.owner_uuid,
                was = ?candidate.instance_hint,
                now = %instance_id,
                queued = candidate.queued,
                held = candidate.held,
                "adopting the only retained queue on this host after an apparent port change"
            );
            adopted_across_port_change = true;
            chosen = Some(candidate);
        } else if with_rows.is_empty() && !candidates.is_empty() {
            // Every unclaimed record is empty, so reusing one carries nothing
            // and keeps the directory from growing a record per port change.
            chosen = Some(candidates.remove(0));
        }
    }

    let orphans: Vec<OrphanedOwner> = candidates
        .into_iter()
        .filter(OwnerCandidate::has_rows)
        .map(OwnerCandidate::into_orphan)
        .collect();

    if let Some(candidate) = chosen {
        write_instance_hint(&candidate.lock, instance_id);
        return Some(QueueOwnerClaim {
            owner_uuid: candidate.owner_uuid,
            _lock: candidate.lock,
            adopted_across_port_change,
            orphans,
        });
    }

    // 4. Mint. Two simultaneous first starts each land here with their own id,
    //    because there is no shared record to race on.
    let owner_uuid = uuid::Uuid::new_v4().to_string();
    let path = owner_record_path(&owners, &owner_uuid);
    let file = match std::fs::File::create(&path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "durable generation queue unavailable: could not create its identity record"
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
    write_instance_hint(&file, instance_id);
    Some(QueueOwnerClaim {
        owner_uuid,
        _lock: file,
        adopted_across_port_change: false,
        orphans,
    })
}

fn owner_row_counts(db: Option<&MetadataDb>, owner_uuid: &str) -> (usize, usize) {
    let Some(db) = db else {
        return (0, 0);
    };
    let Ok(rows) = generation_queue::list_all(db, owner_uuid) else {
        return (0, 0);
    };
    let held = rows
        .iter()
        .filter(|row| row.state == QueueRowState::Held)
        .count();
    (rows.len() - held, held)
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
    pub fn new(db: Arc<Option<MetadataDb>>, mold_dir: Option<&Path>, instance_id: &str) -> Self {
        let claim = if env_flag(JOURNAL_DISABLE_ENV) {
            tracing::info!("durable generation queue disabled by environment");
            None
        } else {
            match (db.as_ref().as_ref(), mold_dir) {
                (Some(db), Some(dir)) => claim_queue_owner(dir, instance_id, Some(db)),
                (Some(_), None) => {
                    tracing::warn!(
                        "durable generation queue unavailable: MOLD_HOME could not be resolved"
                    );
                    None
                }
                _ => None,
            }
        };
        for orphan in claim.iter().flat_map(|claim| claim.orphans.iter()) {
            tracing::warn!(
                owner = %orphan.owner_uuid,
                last_instance = ?orphan.instance_hint,
                queued = orphan.queued,
                held = orphan.held,
                env = QUEUE_ADOPT_OWNER_ENV,
                "a retained generation queue on this host belongs to no running server; \
                 set the adoption environment variable to this owner id to replay it"
            );
        }
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
    /// no gallery target, reference-upload authority (bearer secrets), a face
    /// photograph (biometric data), a batch child (owned by the batch
    /// transaction's own recovery), an oversized payload, or no journal at all.
    pub fn record(self: &Arc<Self>, admission: JournalAdmission<'_>) -> Option<QueueTicket> {
        let owner_uuid = self.owner_uuid.as_deref()?;
        let db = self.db()?;
        let output_dir = admission.output_dir?;
        if admission.batch_child || admission.carries_reference_authority {
            return None;
        }
        // `mold.db` never holds a secret, and a face photograph is the most
        // sensitive payload a request can carry: an identity image is
        // biometric data about a real person, supplied for one render.
        // Journaling it would leave it in a SQLite row on disk, surviving the
        // process, for as long as the row is retained.
        //
        // Excluded at admission rather than redacted, exactly as
        // reference-upload authority is (#1223). A redacted row is worse than
        // no row: replay would resubmit the request with no `id_image`, and
        // `resolve_identity_embedding` would either error or — if the weight
        // fields went with it — render the print with a stranger's face and
        // say nothing. The job runs normally and is advertised
        // `durable: false`, which is the honest answer.
        if carries_identity_photograph(admission.request) {
            tracing::info!(
                job = %admission.id,
                "generation is not durable: it conditions on a reference photograph, \
                 which is never written to the database"
            );
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
            // Admission records the ordinal a client asked for; a stable pin
            // only ever arrives later, through PATCH /api/queue/:id.
            target_device_id: None,
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
            // Absent is not the same as moved. If the configured gallery is
            // still this row's own directory, the save helpers would simply
            // create it — so a routine cleanup or a remount must not park work
            // that would have run perfectly well.
            if current_output_dir == Some(row.output_dir.as_path()) {
                match std::fs::create_dir_all(&row.output_dir) {
                    Ok(()) => {
                        tracing::info!(
                            job = %row.id,
                            dir = %row.output_dir.display(),
                            "recreated a retained generation's gallery directory"
                        );
                        report.recreated += 1;
                        continue;
                    }
                    Err(error) => {
                        tracing::warn!(
                            job = %row.id,
                            dir = %row.output_dir.display(),
                            %error,
                            "a retained generation's gallery directory cannot be recreated"
                        );
                    }
                }
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
        target_device_id: Option<Option<&str>>,
        order: Option<&[String]>,
    ) {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return;
        };
        let now = now_ms();
        if let Some(target_gpu) = target_gpu {
            // The stable id is recorded alongside the ordinal so replay can
            // re-resolve it: ordinals are an enumeration artifact that MIG or
            // a changed MOLD_GPUS renumbers across a restart.
            let stable = target_device_id.flatten();
            if let Err(error) = generation_queue::set_target_gpu(db, id, target_gpu, stable, now) {
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

    /// Whether this id names a retained row that is cancellable but has no
    /// registry entry.
    ///
    /// Two kinds qualify: a held row, which exists only in the journal; and a
    /// queued row on a boot with no dispatch owner, which replay deliberately
    /// never registered. `DELETE /api/queue/:id` is the documented way to
    /// clear either, and listing work an operator cannot then act on would be
    /// half an answer. A `running` row is excluded — the endpoint refuses
    /// running work, and the registry is the authority on that.
    pub fn owns_cancellable_row(&self, id: &str) -> bool {
        let Some(db) = self.db() else {
            return false;
        };
        generation_queue::get(db, id)
            .ok()
            .flatten()
            .is_some_and(|row| matches!(row.state, QueueRowState::Held | QueueRowState::Queued))
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
    /// Galleries that were simply absent and could be made again.
    pub recreated: usize,
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
        // Re-resolve a stable pin against THIS boot's device inventory. The
        // recorded ordinal is only a fallback, and only when no stable pin was
        // taken — replaying a renumbered ordinal runs the job on a device the
        // user did not choose.
        let target_gpu = match row.target_device_id.as_deref() {
            Some(device_id) => match resolve_pinned_ordinal(state, device_id) {
                Some(ordinal) => Some(ordinal),
                None => {
                    tracing::warn!(
                        job = %row.id,
                        device = %device_id,
                        "the device this job was pinned to is not present; resuming on Auto"
                    );
                    None
                }
            },
            None => row.target_gpu,
        };
        let cancel = state.job_registry.register_job(
            &row.id,
            &row.model,
            target_gpu,
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

/// Map a stable device id to this boot's ordinal, or `None` when the device is
/// no longer present.
fn resolve_pinned_ordinal(state: &crate::state::AppState, device_id: &str) -> Option<usize> {
    state
        .gpu_pool
        .workers
        .iter()
        .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
        .map(|worker| worker.gpu.ordinal)
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

    fn owner_db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    fn seed_row(db: &MetadataDb, owner: &str, id: &str) {
        generation_queue::insert(
            db,
            &GenerationQueueRow {
                id: id.to_string(),
                owner_uuid: owner.to_string(),
                state: QueueRowState::Queued,
                model: "flux-dev:q4".to_string(),
                request_json: "{}".to_string(),
                output_dir: std::path::PathBuf::from("/gallery"),
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
            },
        )
        .unwrap();
    }

    /// The case four earlier attempts got wrong. A server restarting must
    /// reclaim ITS OWN identity, not merely one nobody is holding — otherwise
    /// it replays a peer's retained jobs under its own GPUs and configuration
    /// while its own rows sit unreplayed under another id.
    #[test]
    fn a_restart_reclaims_its_own_identity_not_whichever_is_unlocked() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();

        // Both servers run at once at least briefly, which is how a genuine
        // peer gets an identity of its own: while the other holds its record,
        // there is nothing unclaimed to adopt.
        let peer = claim_queue_owner(home.path(), "instance-b", Some(&db)).unwrap();
        let mine = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();
        let peer_owner = peer.owner_uuid().to_string();
        let my_owner = mine.owner_uuid().to_string();
        assert_ne!(peer_owner, my_owner);
        seed_row(&db, &peer_owner, "peer-job");
        seed_row(&db, &my_owner, "my-job");
        drop(peer);
        drop(mine);

        // Both records are unlocked; the peer's sorts first roughly half the
        // time, which is what made "first unlocked wins" appear to work.
        let restarted = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();
        assert_eq!(restarted.owner_uuid(), my_owner);
        assert!(!restarted.adopted_across_port_change);
        assert_eq!(
            restarted.orphans.len(),
            1,
            "the peer's rows are reported, not silently taken"
        );
        assert_eq!(restarted.orphans[0].owner_uuid, peer_owner);
        assert_eq!(restarted.orphans[0].queued, 1);
    }

    #[test]
    fn two_stopped_servers_each_reclaim_their_own_in_either_order() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();
        let a = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();
        let b = claim_queue_owner(home.path(), "instance-b", Some(&db)).unwrap();
        let (owner_a, owner_b) = (a.owner_uuid().to_string(), b.owner_uuid().to_string());
        drop(a);
        drop(b);

        let b_again = claim_queue_owner(home.path(), "instance-b", Some(&db)).unwrap();
        assert_eq!(b_again.owner_uuid(), owner_b);
        let a_again = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();
        assert_eq!(a_again.owner_uuid(), owner_a);
    }

    #[test]
    fn a_live_peers_identity_is_never_taken() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();
        let live = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();

        let other = claim_queue_owner(home.path(), "instance-b", Some(&db)).unwrap();
        assert_ne!(other.owner_uuid(), live.owner_uuid());
    }

    /// The single-server port change the port-independent id exists to serve:
    /// exactly one unlocked owner has rows, so it is unambiguous.
    #[test]
    fn a_changed_port_adopts_the_one_unambiguous_orphan() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();
        let before = claim_queue_owner(home.path(), "instance-port-7680", Some(&db)).unwrap();
        let owner = before.owner_uuid().to_string();
        seed_row(&db, &owner, "job-1");
        drop(before);

        let after = claim_queue_owner(home.path(), "instance-port-7681", Some(&db)).unwrap();
        assert_eq!(after.owner_uuid(), owner);
        assert!(
            after.adopted_across_port_change,
            "an adoption across a changed port must be announced, not silent"
        );
    }

    /// Ambiguity is the only case a human has to resolve: two orphans with
    /// rows and no hint match, so adopting either could replay the wrong
    /// server's queue.
    #[test]
    fn several_orphans_with_rows_are_reported_rather_than_guessed_between() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();
        // Two live peers, so each holds an identity of its own, then both stop.
        let claims: Vec<QueueOwnerClaim> = ["instance-a", "instance-b"]
            .iter()
            .map(|instance| claim_queue_owner(home.path(), instance, Some(&db)).unwrap())
            .collect();
        let owners: Vec<String> = claims
            .iter()
            .map(|claim| claim.owner_uuid().to_string())
            .collect();
        for (index, owner) in owners.iter().enumerate() {
            seed_row(&db, owner, &format!("job-{index}"));
        }
        drop(claims);

        let fresh = claim_queue_owner(home.path(), "instance-c", Some(&db)).unwrap();

        assert!(!owners.contains(&fresh.owner_uuid().to_string()));
        assert!(!fresh.adopted_across_port_change);
        assert_eq!(fresh.orphans.len(), 2);
        for orphan in &fresh.orphans {
            assert_eq!(orphan.queued, 1);
            assert!(orphan.instance_hint.is_some());
        }
    }

    /// The documented escape hatch for the ambiguous case.
    #[test]
    fn an_explicitly_named_owner_is_adopted() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();
        let first = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();
        let second = claim_queue_owner(home.path(), "instance-b", Some(&db)).unwrap();
        let wanted = first.owner_uuid().to_string();
        seed_row(&db, &wanted, "job-1");
        seed_row(&db, second.owner_uuid(), "job-2");
        drop(first);
        drop(second);

        let adopted =
            claim_queue_owner_adopting(home.path(), "instance-c", Some(&db), Some(&wanted))
                .unwrap();
        assert_eq!(adopted.owner_uuid(), wanted);
    }

    /// A held row that is also orphaned is doubly invisible, so the report
    /// counts it separately rather than folding it into the queued total.
    #[test]
    fn the_orphan_report_separates_held_rows_from_queued_ones() {
        let home = tempfile::tempdir().unwrap();
        let db = owner_db();
        let claim = claim_queue_owner(home.path(), "instance-a", Some(&db)).unwrap();
        let owner = claim.owner_uuid().to_string();
        seed_row(&db, &owner, "waiting");
        seed_row(&db, &owner, "parked");
        generation_queue::hold(&db, "parked", "attempts exhausted", 9).unwrap();
        drop(claim);

        let fresh = claim_queue_owner(home.path(), "instance-b", Some(&db)).unwrap();
        // One orphan with rows, so it is adopted — but the counts are reported
        // either way.
        let reported = fresh
            .orphans
            .iter()
            .find(|orphan| orphan.owner_uuid == owner);
        let counts = match reported {
            Some(orphan) => (orphan.queued, orphan.held),
            None => {
                assert_eq!(fresh.owner_uuid(), owner, "adopted the sole orphan");
                let rows = generation_queue::list_all(&db, &owner).unwrap();
                (
                    rows.iter()
                        .filter(|row| row.state == QueueRowState::Queued)
                        .count(),
                    rows.iter()
                        .filter(|row| row.state == QueueRowState::Held)
                        .count(),
                )
            }
        };
        assert_eq!(counts, (1, 1));
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

        journal.apply_queue_mutation("b", Some(Some(2)), Some(Some("cuda:sibling")), None);
        let relaned = journal
            .list_all()
            .into_iter()
            .find(|row| row.id == "b")
            .unwrap();
        assert_eq!(relaned.target_gpu, Some(2));
        assert_eq!(
            relaned.target_device_id.as_deref(),
            Some("cuda:sibling"),
            "the stable pin is what replay re-resolves; the ordinal renumbers"
        );

        journal.apply_queue_mutation(
            "c",
            None,
            None,
            Some(&["c".to_string(), "a".to_string(), "b".to_string()]),
        );
        assert_eq!(rows(&journal), vec!["c", "a", "b"]);

        // Clearing the pin returns the row to Auto rather than leaving it.
        journal.apply_queue_mutation("b", Some(None), Some(None), None);
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

    /// The point of persisting the stable pin: if the device is gone at replay
    /// the job resumes on Auto rather than on whatever now holds that ordinal.
    #[test]
    fn a_missing_pinned_device_resolves_to_auto_rather_than_a_renumbered_ordinal() {
        let state = crate::state::AppState::for_tests();
        assert_eq!(
            super::resolve_pinned_ordinal(&state, "cuda:not-present-on-this-boot"),
            None
        );
    }

    /// A queued print is durable, and so is the filing it was submitted
    /// with: the journal stores the whole request as JSON, so a job replayed
    /// after a restart lands under exactly the tags and collection the user
    /// chose. Tags and a collection are ordinary request fields — unlike a
    /// reference handle there is no secret here to exclude.
    #[test]
    fn a_journaled_request_round_trips_its_tags_and_collection() {
        let journal = journal_with_db();
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q4",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 3.5,
            "title": "Smurf Village",
            "tags": ["smurfs", "village"],
            "collection": { "name": "Sequences" },
        }))
        .expect("filed generate request");

        let ticket = journal
            .record(admission("job-filed", &request, Path::new("/gallery")))
            .expect("the row is durable");

        let rows = journal.list_all();
        let row = rows
            .iter()
            .find(|row| row.id == "job-filed")
            .expect("journaled row");
        let replayed: mold_core::GenerateRequest =
            serde_json::from_str(&row.request_json).expect("stored request parses");
        assert_eq!(replayed.title.as_deref(), Some("Smurf Village"));
        assert_eq!(
            replayed.tags.as_deref(),
            Some(["smurfs".to_string(), "village".to_string()].as_slice())
        );
        assert_eq!(
            replayed.collection,
            Some(mold_core::CollectionRef::by_name("Sequences"))
        );

        ticket.discard();
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

    /// A reference photograph is biometric data about a real person, handed
    /// over for one render. It follows the same rule as reference-upload
    /// authority — excluded at admission, never redacted — so no part of it
    /// can reach `mold.db`, and the job is honestly `durable: false` rather
    /// than replayable with the face missing.
    #[test]
    fn an_identity_request_never_writes_the_photograph_to_the_database() {
        let journal = journal_with_db();
        let mut request = request();
        request.model = "flux-dev:q4".to_string();
        request.id_image = Some(b"\x89PNG\r\n\x1a\n-pretend-this-is-a-face".to_vec());
        request.id_image_name = Some("face.png".to_string());
        request.id_weight = Some(1.0);

        assert!(journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .is_none());
        assert!(rows(&journal).is_empty());

        // The exclusion is the photograph itself, not the mere mention of
        // identity: a request whose bytes were already dropped is ordinary.
        let mut without_photo = request.clone();
        without_photo.id_image = None;
        let ticket = journal
            .record(admission("job-2", &without_photo, Path::new("/gallery")))
            .expect("a request carrying no photograph is ordinary durable work");
        let persisted = journal.list_all();
        assert_eq!(persisted.len(), 1);
        assert!(
            !persisted[0].request_json.contains("pretend-this-is-a-face"),
            "no journaled row may contain reference-photograph bytes"
        );
        ticket.discard();
    }
}
