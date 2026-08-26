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
//! Rows never carry a secret. The legacy [`QueueJournal::record`] path refuses
//! reference-upload handles, resolved paths, and MiniMax H3 requests because
//! it cannot reconstruct their authenticated ingress authority. Canonical
//! batch admission instead seals supported request media and journals only an
//! opaque, owner/job/request-bound authority envelope for durable replay.

use std::collections::HashSet;
use std::path::Path;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use mold_db::generation_batches::{
    self, GenerationBatchChildRow, GenerationBatchDetail, GenerationBatchMediaInsertOutcome,
    GenerationBatchRow,
};
use mold_db::generation_queue::{
    self, GenerationQueueProjectionPage, GenerationQueueRow, QueueProjectionCursor, QueueRowState,
};
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

const PRIVATE_H3_BATCH_DURABILITY_ERROR: &str =
    "heterogeneous batches cannot persist private MiniMax H3 requests";

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
    /// Stable identity corresponding to `target_gpu` in the admitting
    /// worker inventory. `None` means the ordinal is not durable authority.
    pub target_device_id: Option<&'a str>,
    pub completion_payload: SseCompletionPayload,
    /// True when the job is a server-owned adaptive-batch child; those are
    /// owned by the batch transaction's own durable recovery.
    pub batch_child: bool,
    /// True when the request carries reference-upload authority. Those bytes
    /// are bearer secrets staged outside the DB and are never journaled.
    pub carries_reference_authority: bool,
}

pub struct BatchJournalAdmission<'a> {
    pub id: &'a str,
    pub client_batch_id: &'a str,
    pub request_sha256: &'a str,
    pub children: &'a [JournalAdmission<'a>],
}

/// Payload-free child row for the encrypted-media batch transaction.
pub(crate) struct MediaJournalAdmission<'a> {
    pub id: &'a str,
    pub model: &'a str,
    pub request_json: &'a str,
    pub media_set: Option<&'a crate::queue_media_store::MediaSetRef>,
    pub output_dir: &'a Path,
    pub target_gpu: Option<usize>,
    pub target_device_id: Option<&'a str>,
    pub completion_payload: SseCompletionPayload,
    pub seed_pinned: bool,
    pub admission_authority: Option<&'a str>,
}

pub(crate) struct MediaBatchJournalAdmission<'a> {
    pub id: &'a str,
    pub client_batch_id: &'a str,
    /// Randomized authenticated receipt, never a plaintext media fingerprint.
    pub operation_receipt: &'a str,
    pub children: &'a [MediaJournalAdmission<'a>],
    /// Direct attached observers to publish after the immediate transaction
    /// commits and before the feeder wake.
    pub observer_job_ids: &'a [String],
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

/// Whether replay would require authenticated MiniMax H3 ingress authority.
///
/// The model capability contract is the existing authority for this
/// partition. Do not infer it from family-shaped strings or reference
/// presence: FL2VA has no references, and replay restores no private grant.
fn requires_h3_replay_authority(request: &mold_core::GenerateRequest) -> bool {
    mold_core::minimax_h3::capability_contract_for_model(&request.model).is_some()
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
    /// Becomes true only after the claimed owner's key, store, and DB/file
    /// obligations have passed the startup coordinator. It is deliberately
    /// independent from `owner_uuid`: a broken media store must not disable
    /// ordinary media-free queue durability.
    durable_media_ready: AtomicBool,
    /// Why `durable_media_ready` is false, retained for the life of the
    /// process. A single startup log line is not a diagnosability surface: it
    /// is easy to miss on a busy server and gone once the log ages out, so the
    /// reasons stay readable on `/api/status` for as long as the degradation
    /// lasts. Empty whenever startup reconciliation reached a clean fixed
    /// point.
    durable_media_issues: std::sync::RwLock<Vec<String>>,
    /// One concrete DB/store authority shared by startup, terminal cleanup,
    /// and the later admission/feeder integration. Default-empty keeps the
    /// existing media-free journal behavior unchanged.
    queue_media_lifecycle: OnceLock<Arc<crate::queue_media_lifecycle::QueueMediaLifecycle>>,
    queue_media_admission: OnceLock<Arc<crate::queue_media_admission::DurableMediaAdmission>>,
    /// Post-commit lifecycle hints for durable clients. Installed after the
    /// app state creates its one server-wide broadcaster.
    events: OnceLock<Arc<crate::events::EventBroadcaster>>,
    #[cfg(test)]
    fail_completion_lookup: AtomicBool,
    #[cfg(test)]
    fail_batch_lookup: AtomicUsize,
    #[cfg(test)]
    fail_claim_release: AtomicBool,
    #[cfg(test)]
    fail_hold_transition: AtomicBool,
    #[cfg(test)]
    fail_completion_transition: AtomicUsize,
    /// Held for the process's lifetime so a peer sharing this `MOLD_HOME`
    /// cannot adopt the same identity.
    _owner_claim: Option<QueueOwnerClaim>,
    retain: AtomicBool,
    max_bytes: usize,
    max_dispatch_attempts: u32,
    max_replay_seen: u32,
    feeder_notify: tokio::sync::Notify,
    /// Serializes durable queue transitions whose SQLite result is later
    /// projected into the bounded runtime registry.
    ///
    /// This gate is always acquired before either authority. A caller may use
    /// the scheduler fence for a bounded pre-DB lifecycle mark, but must drop
    /// it before blocking SQLite work; the final projection may take the
    /// scheduler fence only after SQLite completes. Keeping this gate separate
    /// means a slow database cannot freeze unrelated grants or in-memory
    /// cancellation, while PATCH, feeder publication, and durable cancellation
    /// still agree on one claim/order transition.
    durable_transition_gate: Arc<tokio::sync::Mutex<()>>,
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
            durable_media_ready: AtomicBool::new(false),
            durable_media_issues: std::sync::RwLock::new(Vec::new()),
            queue_media_lifecycle: OnceLock::new(),
            queue_media_admission: OnceLock::new(),
            events: OnceLock::new(),
            _owner_claim: claim,
            #[cfg(test)]
            fail_completion_lookup: AtomicBool::new(false),
            #[cfg(test)]
            fail_batch_lookup: AtomicUsize::new(0),
            #[cfg(test)]
            fail_claim_release: AtomicBool::new(false),
            #[cfg(test)]
            fail_hold_transition: AtomicBool::new(false),
            #[cfg(test)]
            fail_completion_transition: AtomicUsize::new(0),
            retain: AtomicBool::new(false),
            max_bytes: env_usize(JOURNAL_MAX_BYTES_ENV, DEFAULT_JOURNAL_MAX_BYTES),
            max_dispatch_attempts: env_u32(
                MAX_DISPATCH_ATTEMPTS_ENV,
                DEFAULT_MAX_DISPATCH_ATTEMPTS,
            ),
            max_replay_seen: env_u32(MAX_REPLAY_SEEN_ENV, DEFAULT_MAX_REPLAY_SEEN),
            feeder_notify: tokio::sync::Notify::new(),
            durable_transition_gate: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// A journal that persists nothing. Used by tests and by every runtime
    /// where the metadata DB is absent.
    pub fn disabled() -> Self {
        Self {
            db: Arc::new(None),
            owner_uuid: None,
            durable_media_ready: AtomicBool::new(false),
            durable_media_issues: std::sync::RwLock::new(Vec::new()),
            queue_media_lifecycle: OnceLock::new(),
            queue_media_admission: OnceLock::new(),
            events: OnceLock::new(),
            _owner_claim: None,
            #[cfg(test)]
            fail_completion_lookup: AtomicBool::new(false),
            #[cfg(test)]
            fail_batch_lookup: AtomicUsize::new(0),
            #[cfg(test)]
            fail_claim_release: AtomicBool::new(false),
            #[cfg(test)]
            fail_hold_transition: AtomicBool::new(false),
            #[cfg(test)]
            fail_completion_transition: AtomicUsize::new(0),
            retain: AtomicBool::new(false),
            max_bytes: DEFAULT_JOURNAL_MAX_BYTES,
            max_dispatch_attempts: DEFAULT_MAX_DISPATCH_ATTEMPTS,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
            feeder_notify: tokio::sync::Notify::new(),
            durable_transition_gate: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Enter the durable-transition protocol without touching the scheduler
    /// mutation fence. Callers may await SQLite while holding this guard, but
    /// must finish that work before acquiring the scheduler fence.
    pub(crate) async fn lock_durable_transition(
        self: &Arc<Self>,
    ) -> tokio::sync::OwnedMutexGuard<()> {
        self.durable_transition_gate.clone().lock_owned().await
    }

    #[cfg(test)]
    pub(crate) fn durable_transition_is_locked(&self) -> bool {
        self.durable_transition_gate.try_lock().is_err()
    }

    /// Whether this server can promise that a queued job survives a restart.
    /// Backs `QueueCapabilities.durable_queue`.
    pub fn is_enabled(&self) -> bool {
        self.owner_uuid.is_some()
    }

    pub fn owner_uuid(&self) -> Option<&str> {
        self.owner_uuid.as_deref()
    }

    /// Advertise encrypted queue-media durability only after the claimed
    /// owner's startup reconciliation reached a clean fixed point. Absence is
    /// intentionally independent from [`Self::is_enabled`], so key/store
    /// failures do not turn off media-free durable generations.
    pub fn durable_media_capabilities(&self) -> Option<mold_core::DurableMediaCapabilities> {
        (self.durable_media_ready.load(Ordering::Acquire)
            && self.queue_media_lifecycle.get().is_some()
            && self.queue_media_admission.get().is_some())
        .then_some(mold_core::DurableMediaCapabilities::v2(cfg!(any(
            feature = "h3",
            feature = "h3-private-uat"
        ))))
    }

    /// The operator-facing counterpart of [`Self::durable_media_capabilities`].
    ///
    /// `applicable` is the runtime half of the same conjunction that gates
    /// `capabilities.durable_media` — gallery output and an authoritative
    /// scheduler — supplied by the caller because it reads config and the
    /// scheduler rather than the journal. `None` therefore means this server
    /// never offers restart-safe media at all, which is a configuration rather
    /// than a degradation: reporting it as degraded would make every
    /// `MOLD_DB_DISABLE`, output-disabled, or observe-mode host look broken.
    /// Where it IS applicable, `available` is true exactly when the capability
    /// is advertised, and the conjuncts behind a false are reported
    /// separately, because a widened store directory and a never-installed
    /// admission service need different repairs.
    pub fn durable_media_status(&self, applicable: bool) -> Option<mold_core::DurableMediaStatus> {
        if !applicable || !self.is_enabled() {
            return None;
        }
        if self.durable_media_capabilities().is_some() {
            return Some(mold_core::DurableMediaStatus {
                available: true,
                reasons: Vec::new(),
            });
        }
        let mut reasons = match self.durable_media_issues.read() {
            Ok(issues) => issues.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        if self.queue_media_lifecycle.get().is_none() {
            reasons.push(
                "no queue-media lifecycle was installed for the claimed queue owner".to_string(),
            );
        } else if self.queue_media_admission.get().is_none() {
            reasons.push("the queue-media admission service was not installed".to_string());
        }
        if reasons.is_empty() {
            reasons.push("queue-media startup reconciliation did not complete".to_string());
        }
        Some(mold_core::DurableMediaStatus {
            available: false,
            reasons,
        })
    }

    /// Whether restart-safe media is applicable here and switched off, without
    /// composing the reason prose. `/health` answers this on every liveness
    /// poll and needs only the yes/no; the reasons live behind authentication
    /// on `/api/status`, and cloning them per probe would charge every poll for
    /// a list whose length is the held-job backlog.
    pub fn durable_media_is_degraded(&self, applicable: bool) -> bool {
        applicable && self.is_enabled() && self.durable_media_capabilities().is_none()
    }

    pub(crate) fn install_queue_media_lifecycle(
        &self,
        lifecycle: Arc<crate::queue_media_lifecycle::QueueMediaLifecycle>,
    ) -> Result<(), &'static str> {
        self.queue_media_lifecycle
            .set(lifecycle)
            .map_err(|_| "queue-media lifecycle was already installed")
    }

    // Consumed by the admission/runtime slices only after their independent
    // activation review; this lifecycle slice deliberately leaves them dark.
    #[allow(dead_code)]
    pub(crate) fn queue_media_lifecycle(
        &self,
    ) -> Option<Arc<crate::queue_media_lifecycle::QueueMediaLifecycle>> {
        self.queue_media_lifecycle.get().cloned()
    }

    pub(crate) fn install_queue_media_admission(
        &self,
        admission: Arc<crate::queue_media_admission::DurableMediaAdmission>,
    ) -> Result<(), &'static str> {
        self.queue_media_admission
            .set(admission)
            .map_err(|_| "durable-media admission was already installed")
    }

    pub(crate) fn install_event_broadcaster(
        &self,
        events: Arc<crate::events::EventBroadcaster>,
    ) -> Result<(), &'static str> {
        self.events
            .set(events)
            .map_err(|_| "queue journal event broadcaster was already installed")
    }

    fn publish_state_committed(&self, id: &str) {
        if let Some(events) = self.events.get() {
            events.publish(mold_core::ServerEvent::JobStateCommitted { id: id.to_string() });
        }
    }

    fn publish_states_committed(&self) {
        if let Some(events) = self.events.get() {
            events.publish(mold_core::ServerEvent::GenerationStatesCommitted);
        }
    }

    pub(crate) fn queue_media_admission(
        &self,
    ) -> Option<Arc<crate::queue_media_admission::DurableMediaAdmission>> {
        self.queue_media_admission.get().cloned()
    }

    // Called by the default-dark startup coordinator once its independently
    // reviewed concrete DB/store adapter is integrated.
    #[allow(dead_code)]
    pub(crate) fn set_durable_media_ready(&self, ready: bool) {
        self.set_durable_media_status(ready, Vec::new());
    }

    /// Record readiness together with the reasons it was withheld. Readiness
    /// and its explanation are set by one call so a later reader can never see
    /// "unavailable" beside a stale empty reason list.
    pub(crate) fn set_durable_media_status(&self, ready: bool, issues: Vec<String>) {
        let issues = if ready { Vec::new() } else { issues };
        match self.durable_media_issues.write() {
            Ok(mut retained) => *retained = issues,
            Err(poisoned) => *poisoned.into_inner() = issues,
        }
        self.durable_media_ready.store(ready, Ordering::Release);
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
    /// photograph (biometric data), MiniMax H3 replay authority, a batch child
    /// (owned by the batch transaction's own recovery), an oversized payload,
    /// or no journal at all.
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
        if requires_h3_replay_authority(admission.request) {
            tracing::info!(
                job = %admission.id,
                model = %admission.request.model,
                "generation is not durable: MiniMax H3 replay cannot reconstruct its \
                 authenticated ingress authority"
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
            media_set_id: None,
            admission_authority: None,
            output_dir: output_dir.to_path_buf(),
            target_gpu: admission.target_gpu,
            target_device_id: admission.target_device_id.map(ToOwned::to_owned),
            completion_payload: completion_payload_as_str(admission.completion_payload).to_string(),
            seed_pinned: admission.request.seed.is_some(),
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms: now,
            updated_at_ms: now,
            started_at_ms: None,
        };
        let claim_token = uuid::Uuid::new_v4().to_string();
        if let Err(error) = generation_queue::insert_claimed(db, &row, &claim_token) {
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
            claim_token: Some(claim_token),
            settled: false,
        })
    }

    /// Persist one heterogeneous parent index and all ordinary queue children
    /// in a single SQLite transaction. The boolean reports whether this call
    /// inserted the batch rather than returning an idempotent retry.
    pub fn record_batch(
        self: &Arc<Self>,
        admission: BatchJournalAdmission<'_>,
    ) -> Result<(GenerationBatchDetail, bool), String> {
        let owner_uuid = self
            .owner_uuid
            .as_deref()
            .ok_or_else(|| "durable generation queue is unavailable".to_string())?;
        let db = self
            .db()
            .ok_or_else(|| "metadata DB is unavailable".to_string())?;
        if admission.children.is_empty() {
            return Err("batch must contain at least one child".to_string());
        }
        let now = now_ms();
        let mut rows = Vec::with_capacity(admission.children.len());
        for (offset, child) in admission.children.iter().enumerate() {
            let output_dir = child
                .output_dir
                .ok_or_else(|| "heterogeneous batches require gallery output".to_string())?;
            if child.batch_child || child.carries_reference_authority {
                return Err(
                    "heterogeneous batches cannot carry temporary reference authority".to_string(),
                );
            }
            if carries_identity_photograph(child.request) {
                return Err("heterogeneous batches cannot persist identity photographs".to_string());
            }
            if requires_h3_replay_authority(child.request) {
                return Err(PRIVATE_H3_BATCH_DURABILITY_ERROR.to_string());
            }
            let request_json = serde_json::to_string(child.request)
                .map_err(|error| format!("could not serialize batch child: {error}"))?;
            if request_json.len() > self.max_bytes {
                return Err(format!(
                    "batch child {} exceeds the durable queue payload ceiling",
                    offset + 1
                ));
            }
            let queue_row = GenerationQueueRow {
                id: child.id.to_string(),
                owner_uuid: owner_uuid.to_string(),
                state: QueueRowState::Queued,
                model: child.request.model.clone(),
                request_json,
                media_set_id: None,
                admission_authority: None,
                output_dir: output_dir.to_path_buf(),
                target_gpu: child.target_gpu,
                target_device_id: child.target_device_id.map(ToOwned::to_owned),
                completion_payload: completion_payload_as_str(child.completion_payload).to_string(),
                seed_pinned: child.request.seed.is_some(),
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: now,
                updated_at_ms: now,
                started_at_ms: None,
            };
            let batch_child = GenerationBatchChildRow {
                batch_id: admission.id.to_string(),
                job_id: child.id.to_string(),
                batch_index: (offset + 1) as u32,
                state: "accepted".to_string(),
                error: None,
                updated_at_ms: now,
            };
            rows.push((batch_child, queue_row));
        }
        let batch = GenerationBatchRow {
            id: admission.id.to_string(),
            client_batch_id: admission.client_batch_id.to_string(),
            owner_uuid: owner_uuid.to_string(),
            request_sha256: admission.request_sha256.to_string(),
            created_at_ms: now,
        };
        let (detail, inserted) = generation_batches::insert_or_get(db, &batch, &rows)
            .map_err(|error| format!("could not persist generation batch: {error:#}"))?;
        if inserted {
            self.feeder_notify.notify_one();
        }
        Ok((detail, inserted))
    }

    /// Persist one media-bearing operation in the same immediate transaction
    /// as all child rows and active media obligations. Existing media-free
    /// `record_batch` remains byte-for-byte on its legacy equality path.
    pub(crate) fn record_batch_with_media(
        self: &Arc<Self>,
        admission: MediaBatchJournalAdmission<'_>,
    ) -> Result<GenerationBatchMediaInsertOutcome, String> {
        let owner_uuid = self
            .owner_uuid
            .as_deref()
            .ok_or_else(|| "durable generation queue is unavailable".to_string())?;
        let db = self
            .db()
            .ok_or_else(|| "metadata DB is unavailable".to_string())?;
        if admission.children.is_empty() {
            return Err("batch must contain at least one child".to_string());
        }
        let now = now_ms();
        let mut rows = Vec::with_capacity(admission.children.len());
        let mut obligations = Vec::new();
        for (offset, child) in admission.children.iter().enumerate() {
            if child.request_json.len() > self.max_bytes {
                return Err(format!(
                    "batch child {} exceeds the durable queue payload ceiling",
                    offset + 1
                ));
            }
            let media_set_id = match child.media_set {
                Some(media_set)
                    if media_set.owner_id == owner_uuid && media_set.job_id == child.id =>
                {
                    obligations.push(mold_db::generation_queue_media::QueueMediaObligation {
                        media_set_id: media_set.set_id.clone(),
                        owner_uuid: owner_uuid.to_string(),
                        state: mold_db::generation_queue_media::QueueMediaObligationState::Active,
                        created_at_ms: now,
                        updated_at_ms: now,
                    });
                    Some(media_set.set_id.clone())
                }
                Some(_) => {
                    return Err(format!(
                        "batch child {} media authority does not match its owner and job",
                        offset + 1
                    ));
                }
                None => None,
            };
            rows.push((
                GenerationBatchChildRow {
                    batch_id: admission.id.to_string(),
                    job_id: child.id.to_string(),
                    batch_index: (offset + 1) as u32,
                    state: "accepted".to_string(),
                    error: None,
                    updated_at_ms: now,
                },
                GenerationQueueRow {
                    id: child.id.to_string(),
                    owner_uuid: owner_uuid.to_string(),
                    state: QueueRowState::Queued,
                    model: child.model.to_string(),
                    request_json: child.request_json.to_string(),
                    media_set_id,
                    admission_authority: child.admission_authority.map(ToOwned::to_owned),
                    output_dir: child.output_dir.to_path_buf(),
                    target_gpu: child.target_gpu,
                    target_device_id: child.target_device_id.map(ToOwned::to_owned),
                    completion_payload: completion_payload_as_str(child.completion_payload)
                        .to_string(),
                    seed_pinned: child.seed_pinned,
                    dispatch_attempts: 0,
                    replay_seen: 0,
                    held_reason: None,
                    created_at_ms: now,
                    updated_at_ms: now,
                    started_at_ms: None,
                },
            ));
        }
        let batch = GenerationBatchRow {
            id: admission.id.to_string(),
            client_batch_id: admission.client_batch_id.to_string(),
            owner_uuid: owner_uuid.to_string(),
            request_sha256: admission.operation_receipt.to_string(),
            created_at_ms: now,
        };
        let outcome = generation_batches::insert_or_get_with_media(db, &batch, &rows, &obligations)
            .map_err(|error| format!("could not persist media generation batch: {error:#}"))?;
        if matches!(outcome, GenerationBatchMediaInsertOutcome::Inserted(_)) {
            if let Some(service) = self.queue_media_admission.get() {
                for job_id in admission.observer_job_ids {
                    service.publish_observer(job_id);
                }
            }
            self.feeder_notify.notify_one();
        }
        Ok(outcome)
    }

    pub fn generation_batch(&self, id: &str) -> Option<GenerationBatchDetail> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return None;
        };
        generation_batches::get(db, owner, id).ok().flatten()
    }

    pub fn generation_batch_by_client(
        &self,
        client_batch_id: &str,
    ) -> Option<GenerationBatchDetail> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return None;
        };
        generation_batches::get_by_client(db, owner, client_batch_id)
            .ok()
            .flatten()
    }

    pub fn durable_generation_batch(
        &self,
        id: &str,
    ) -> Result<Option<mold_db::generation_batches::DurableGenerationBatchDetail>, String> {
        #[cfg(test)]
        if self
            .fail_batch_lookup
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |remaining| remaining.checked_sub(1),
            )
            .is_ok_and(|previous| previous == 1)
        {
            return Err("injected durable batch lookup failure".to_string());
        }
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(None);
        };
        generation_batches::get_durable(db, owner, id).map_err(|error| format!("{error:#}"))
    }

    pub fn durable_generation_batch_by_client(
        &self,
        client_batch_id: &str,
    ) -> Result<Option<mold_db::generation_batches::DurableGenerationBatchDetail>, String> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(None);
        };
        generation_batches::get_durable_by_client(db, owner, client_batch_id)
            .map_err(|error| format!("{error:#}"))
    }

    pub fn durable_generation_batches(
        &self,
        client_batch_ids: &[String],
        batch_ids: &[String],
    ) -> Result<mold_db::generation_batches::DurableGenerationBatchLookup, String> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            let unique = |values: &[String]| {
                let mut seen = HashSet::new();
                values
                    .iter()
                    .filter(|value| seen.insert(value.as_str()))
                    .cloned()
                    .collect()
            };
            return Ok(mold_db::generation_batches::DurableGenerationBatchLookup {
                batches: Vec::new(),
                missing_client_batch_ids: unique(client_batch_ids),
                missing_batch_ids: unique(batch_ids),
            });
        };
        generation_batches::lookup_durable(db, owner, client_batch_ids, batch_ids)
            .map_err(|error| format!("{error:#}"))
    }

    fn set_batch_child_state(&self, id: &str, state: &str, error: Option<&str>) {
        let Some(db) = self.db() else { return };
        match generation_batches::set_child_state(db, id, state, error, now_ms()) {
            // `job_started` already invalidates clients after the running
            // transition. Reserve the host-wide post-commit hint for states
            // that otherwise race the earlier `job_ended`/gallery events.
            Ok(true) if state != "running" => self.publish_state_committed(id),
            Ok(true) => {}
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(job = %id, %error, "could not update generation batch child state");
            }
        }
    }

    /// Re-attach a ticket to a row that already exists. Used by replay, which
    /// resubmits an existing row rather than writing a new one.
    pub fn attach(self: &Arc<Self>, id: &str) -> QueueTicket {
        QueueTicket {
            journal: Arc::clone(self),
            id: id.to_string(),
            claim_token: None,
            settled: false,
        }
    }

    /// Claim exactly one oldest row not owned by a live direct submitter.
    /// Payload hydration starts only after this returns, so the deep backlog
    /// never enters memory. Startup recovery clears tokens from the prior
    /// runtime, making interrupted legacy rows eligible here too.
    pub(crate) fn claim_next_feeder(
        self: &Arc<Self>,
    ) -> anyhow::Result<Option<mold_db::generation_queue::QueueClaim>> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(None);
        };
        let token = uuid::Uuid::new_v4().to_string();
        generation_queue::claim_next(db, owner, &token, now_ms())
    }

    /// Claim one exact queued row through the same owner/token fence as the
    /// FIFO feeder. Admission can wake and hand off a newly committed id
    /// without acquiring direct runtime authority or bypassing the journal.
    #[allow(dead_code)] // activated by the separately reviewed admission feeder slice
    pub(crate) fn claim_feeder_by_id(
        self: &Arc<Self>,
        id: &str,
    ) -> anyhow::Result<Option<mold_db::generation_queue::QueueClaim>> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(None);
        };
        let token = uuid::Uuid::new_v4().to_string();
        generation_queue::claim_by_id(db, owner, id, &token, now_ms())
    }

    /// Locate an exact live feeder claim in SQLite's payload-free runtime
    /// order window. See [`generation_queue::claimed_runtime_position`].
    pub(crate) fn claimed_runtime_position(
        &self,
        id: &str,
        claim_token: &str,
        limit: usize,
    ) -> anyhow::Result<Option<generation_queue::ClaimedQueueRuntimePosition>> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(None);
        };
        generation_queue::claimed_runtime_position(db, owner, id, claim_token, limit)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn attach_claimed(self: &Arc<Self>, id: &str, claim_token: String) -> QueueTicket {
        QueueTicket {
            journal: Arc::clone(self),
            id: id.to_string(),
            claim_token: Some(claim_token),
            settled: false,
        }
    }

    pub(crate) fn feeder_notified(&self) -> impl std::future::Future<Output = ()> + '_ {
        self.feeder_notify.notified()
    }

    pub(crate) fn wake_feeder(&self) {
        self.feeder_notify.notify_one();
    }

    pub(crate) fn recover_feeder_runtime(
        &self,
    ) -> anyhow::Result<generation_queue::RuntimeClaimRecovery> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(generation_queue::RuntimeClaimRecovery::default());
        };
        generation_queue::recover_runtime_claims_and_charge_replays(
            db,
            owner,
            now_ms(),
            self.max_replay_seen,
        )
    }

    pub(crate) fn completed_output(
        &self,
        id: &str,
    ) -> Result<
        Option<generation_queue::CompletedGenerationOutput>,
        generation_queue::CompletedOutputLookupError,
    > {
        #[cfg(test)]
        if self
            .fail_completion_lookup
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(
                generation_queue::CompletedOutputLookupError::Infrastructure(anyhow::anyhow!(
                    "injected completion-lookup failure"
                )),
            );
        }
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(None);
        };
        generation_queue::find_completed_output(db, owner, id)
    }

    pub(crate) fn repoint_output(&self, id: &str, output_dir: &Path) -> anyhow::Result<()> {
        let Some(db) = self.db() else { return Ok(()) };
        generation_queue::set_output_dir(db, id, &output_dir.to_string_lossy(), now_ms())?;
        Ok(())
    }

    pub(crate) fn feeder_cancel_requested(&self, id: &str) -> anyhow::Result<bool> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(false);
        };
        generation_batches::child_cancel_requested(db, owner, id)
    }

    fn media_candidate(
        &self,
        id: &str,
    ) -> Option<crate::queue_media_lifecycle::QueueMediaGcCandidate> {
        let lifecycle = self.queue_media_lifecycle.get()?;
        match lifecycle.candidate_for_job(id) {
            Ok(candidate) => candidate,
            Err(error) => {
                tracing::warn!(
                    job = %id,
                    %error,
                    "could not snapshot queue-media cleanup authority; the DB trigger will retain GC work"
                );
                None
            }
        }
    }

    fn active_media_candidates(&self) -> Vec<crate::queue_media_lifecycle::QueueMediaGcCandidate> {
        let Some(lifecycle) = self.queue_media_lifecycle.get() else {
            return Vec::new();
        };
        match lifecycle.active_candidates() {
            Ok(candidates) => candidates,
            Err(error) => {
                tracing::warn!(
                    %error,
                    "could not snapshot bulk queue-media cleanup authority; the DB trigger will retain GC work"
                );
                Vec::new()
            }
        }
    }

    fn cleanup_media_candidate(
        &self,
        candidate: Option<crate::queue_media_lifecycle::QueueMediaGcCandidate>,
    ) {
        let (Some(lifecycle), Some(candidate)) = (self.queue_media_lifecycle.get(), candidate)
        else {
            return;
        };
        if let Err(error) = lifecycle.cleanup_after_committed_delete(&candidate) {
            tracing::warn!(
                %error,
                "queue-media cleanup remains GC-pending after terminal queue deletion"
            );
        }
    }

    fn cleanup_media_candidates(
        &self,
        candidates: Vec<crate::queue_media_lifecycle::QueueMediaGcCandidate>,
    ) {
        for candidate in candidates {
            self.cleanup_media_candidate(Some(candidate));
        }
    }

    /// Drop one row regardless of the fence. The cancellation path: a job the
    /// user explicitly removed must not come back after a restart, even when
    /// the cancel lands during the drain.
    pub fn discard_id(&self, id: &str) {
        let Some(db) = self.db() else {
            return;
        };
        let candidate = self.media_candidate(id);
        match generation_queue::delete(db, id) {
            Ok(true) => self.cleanup_media_candidate(candidate),
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(
                    job = %id,
                    error = %format!("{error:#}"),
                    "could not remove a cancelled job from the durable queue"
                );
            }
        }
    }

    /// Cancel an API-visible queue id without stealing a feeder claim.
    /// Unhydrated batch children settle atomically; claimed children are left
    /// for their token-bearing ticket, and legacy rows keep direct deletion.
    pub fn cancel_id(&self, id: &str) -> anyhow::Result<bool> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(false);
        };
        let terminal_error_json = serde_json::json!({ "message": "Cancelled" }).to_string();
        let terminal = generation_batches::GenerationBatchTerminal {
            state: generation_batches::GenerationBatchTerminalState::Cancelled,
            error: Some("Cancelled"),
            terminal_error_json: Some(&terminal_error_json),
            result_json: None,
            completed_at_ms: now_ms(),
        };
        let candidate = self.media_candidate(id);
        let outcome = generation_batches::cancel_owned(db, owner, id, terminal)?;
        if outcome == generation_batches::OwnedCancellation::Settled {
            self.cleanup_media_candidate(candidate);
            if let Some(service) = self.queue_media_admission.get() {
                service.ingress().cancel(id);
            }
            self.publish_state_committed(id);
        }
        let cancelled = outcome != generation_batches::OwnedCancellation::NotOwned;
        self.wake_feeder();
        Ok(cancelled)
    }

    /// Drop every still-queued row this server owns. Backs `DELETE /api/queue`.
    pub fn discard_all_queued(&self) {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return;
        };
        let candidates = self.active_media_candidates();
        match generation_queue::delete_all_queued(db, owner) {
            Ok(_) => self.cleanup_media_candidates(candidates),
            Err(error) => {
                tracing::warn!(
                    error = %format!("{error:#}"),
                    "could not clear the durable queue after a bulk cancel"
                );
            }
        }
    }

    pub fn cancel_all_queued(&self, already_counted_live: &[String]) -> anyhow::Result<usize> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(0);
        };
        let terminal_error_json = serde_json::json!({ "message": "Cancelled" }).to_string();
        let terminal = generation_batches::GenerationBatchTerminal {
            state: generation_batches::GenerationBatchTerminalState::Cancelled,
            error: Some("Cancelled"),
            terminal_error_json: Some(&terminal_error_json),
            result_json: None,
            completed_at_ms: now_ms(),
        };
        let candidates = self.active_media_candidates();
        let additional =
            generation_batches::cancel_all_queued(db, owner, already_counted_live, terminal)?;
        self.cleanup_media_candidates(candidates);
        if additional > 0 || !already_counted_live.is_empty() {
            self.publish_states_committed();
        }
        self.wake_feeder();
        Ok(additional)
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

    #[cfg(test)]
    pub(crate) fn fail_batch_lookup_after_for_tests(&self, successful_reads: usize) {
        self.fail_batch_lookup
            .store(successful_reads + 1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Test seam: make the next feeder claim release fail after the database
    /// has accepted the claim, preserving the exact token for an in-process
    /// retry.
    #[cfg(test)]
    pub(crate) fn fail_claim_release_for_tests(&self) {
        self.fail_claim_release
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Test seam: fail the next claimed hold transaction before it changes
    /// SQLite, proving the ticket falls back to claim release.
    #[cfg(test)]
    pub(crate) fn fail_hold_transition_for_tests(&self) {
        self.fail_hold_transition
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Test seam: fail the next `attempts` exact completion transactions
    /// before SQLite changes, preserving the claimed ticket for retry.
    #[cfg(test)]
    pub(crate) fn fail_completion_transition_for_tests(&self, attempts: usize) {
        self.fail_completion_transition
            .store(attempts, std::sync::atomic::Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn claim_release_failure_pending_for_tests(&self) -> bool {
        self.fail_claim_release
            .load(std::sync::atomic::Ordering::SeqCst)
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

    /// Whether this id names any nonterminal durable row owned by this server.
    ///
    /// The cancellation route performs this blocking lookup before taking the
    /// scheduler mutation fence. Including a feeder-claimed `running` row is
    /// load-bearing: it may not have reached the registry yet, but cancellation
    /// must still record intent for the token-bearing feeder handoff.
    pub fn owns_cancellable_row(&self, id: &str) -> anyhow::Result<bool> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(false);
        };
        Ok(generation_queue::get(db, id)?.is_some_and(|row| row.owner_uuid == owner))
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

    /// Return one explicitly retryable held row to the feeder backlog.
    pub fn retry_held(&self, id: &str) -> anyhow::Result<generation_batches::OwnedRetry> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(generation_batches::OwnedRetry::NotOwned);
        };
        let outcome = generation_batches::retry_held_owned(db, owner, id, now_ms())?;
        if outcome == generation_batches::OwnedRetry::Retried {
            self.publish_state_committed(id);
            self.wake_feeder();
        }
        Ok(outcome)
    }

    /// Read one payload-free durable page. This method is synchronous because
    /// SQLite is synchronous; async callers must run it on a blocking worker.
    pub fn projection_page(
        &self,
        cursor: Option<QueueProjectionCursor>,
        limit: usize,
    ) -> anyhow::Result<GenerationQueueProjectionPage> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(GenerationQueueProjectionPage {
                rows: Vec::new(),
                next_cursor: None,
            });
        };
        generation_queue::list_projection_page(db, owner, cursor, limit)
    }

    /// Identify active registry rows that are durable without reading their
    /// payload columns or scanning the deep journal. Synchronous for the same
    /// reason as [`Self::projection_page`].
    pub fn owned_row_ids(&self, ids: &[String]) -> anyhow::Result<HashSet<String>> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(HashSet::new());
        };
        generation_queue::find_owned_ids(db, owner, ids)
    }

    /// Exact total waiting load without materializing the durable backlog.
    ///
    /// SQLite owns unclaimed durable rows; the bounded registry list supplies
    /// live waiting jobs that have no durable row. The overlap probe is scoped
    /// to that bounded list so hydrated durable jobs contribute exactly once.
    pub fn total_waiting(&self, live_waiting_ids: &[String]) -> anyhow::Result<usize> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(live_waiting_ids.len());
        };
        let load = generation_queue::owned_queued_load(db, owner, live_waiting_ids)?;
        let live_only = live_waiting_ids
            .len()
            .checked_sub(load.live_overlap)
            .ok_or_else(|| anyhow::anyhow!("durable waiting overlap exceeds live waiting rows"))?;
        load.queued_count
            .checked_add(live_only)
            .ok_or_else(|| anyhow::anyhow!("total waiting generation load exceeds usize"))
    }

    /// Atomically patch an owner-fenced durable queued row. This includes rows
    /// deeper than the hydrated registry window and never reads request or
    /// completion payload columns.
    pub fn patch_owned_queued(
        &self,
        id: &str,
        target_gpu: Option<Option<usize>>,
        target_device_id: Option<Option<String>>,
        position: Option<usize>,
    ) -> anyhow::Result<generation_queue::OwnedQueuedPatchOutcome> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(generation_queue::OwnedQueuedPatchOutcome::NotOwned);
        };
        let target = target_gpu.map(|target_gpu| generation_queue::QueueTargetPatch {
            target_gpu,
            target_device_id: target_device_id.flatten(),
        });
        generation_queue::patch_owned_queued(
            db,
            owner,
            id,
            &generation_queue::OwnedQueuedPatch {
                target,
                position,
                updated_at_ms: now_ms(),
            },
        )
    }

    /// Atomically patch the durable counterpart of a queued live-registry
    /// handoff. The DB primitive requires an owner-matching non-NULL claim, so
    /// this path cannot accidentally acquire an unhydrated deep-tail row.
    pub fn patch_owned_claimed_queued(
        &self,
        id: &str,
        target_gpu: Option<Option<usize>>,
        target_device_id: Option<Option<String>>,
        position: Option<usize>,
    ) -> anyhow::Result<generation_queue::OwnedQueuedPatchOutcome> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(generation_queue::OwnedQueuedPatchOutcome::NotOwned);
        };
        let target = target_gpu.map(|target_gpu| generation_queue::QueueTargetPatch {
            target_gpu,
            target_device_id: target_device_id.flatten(),
        });
        generation_queue::patch_owned_claimed_queued(
            db,
            owner,
            id,
            &generation_queue::OwnedQueuedPatch {
                target,
                position,
                updated_at_ms: now_ms(),
            },
        )
    }

    /// Patch an owned queued row while fencing every write to the exact claim
    /// token (including NULL) observed by the DB transaction. The caller must
    /// own [`Self::lock_durable_transition`] across this call and the bounded
    /// runtime projection.
    pub(crate) fn patch_owned_any_queued(
        &self,
        id: &str,
        target_gpu: Option<Option<usize>>,
        target_device_id: Option<Option<String>>,
        position: Option<usize>,
    ) -> anyhow::Result<generation_queue::OwnedQueuedPatchOutcome> {
        let (Some(db), Some(owner)) = (self.db(), self.owner_uuid.as_deref()) else {
            return Ok(generation_queue::OwnedQueuedPatchOutcome::NotOwned);
        };
        let target = target_gpu.map(|target_gpu| generation_queue::QueueTargetPatch {
            target_gpu,
            target_device_id: target_device_id.flatten(),
        });
        generation_queue::patch_owned_any_queued(
            db,
            owner,
            id,
            &generation_queue::OwnedQueuedPatch {
                target,
                position,
                updated_at_ms: now_ms(),
            },
        )
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
        let mut request: mold_core::GenerateRequest = match serde_json::from_str(&row.request_json)
        {
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

        let target_gpu = resolve_replay_affinity(
            &mut request,
            row.target_gpu,
            row.target_device_id.as_deref(),
            |device_id| resolve_pinned_ordinal(state, device_id),
        );
        if row.target_gpu.is_some() && target_gpu.is_none() {
            tracing::warn!(
                job = %row.id,
                device = ?row.target_device_id,
                "durable GPU identity is absent or unavailable; resuming on Auto"
            );
        }
        let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
            &request,
            request.seed.unwrap_or(0),
            request.scheduler,
            mold_core::build_info::version_string(),
        ));
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
            durable_queue_rank: None,
            request,
            deferred_media: None,
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

/// Capture durable affinity only when the active worker exposes a true stable
/// identity. The scheduler's `runtime:gpu:N` fallback is deliberately not
/// persisted: it is process-local and would merely disguise an ordinal as a
/// durable pin.
pub(crate) fn stable_device_id_for_ordinal(
    state: &crate::state::AppState,
    target_gpu: Option<usize>,
) -> Option<String> {
    target_gpu
        .and_then(|ordinal| state.gpu_pool.worker_by_ordinal(ordinal))
        .and_then(|worker| worker.gpu.stable_id.clone())
}

/// Rebuild one durable row's affinity against the current worker inventory.
///
/// A recorded ordinal is never restart authority by itself. The journal's
/// `(recorded_ordinal, target_device_id)` pair is authoritative only for that
/// exact legacy ordinal, so matching `Gpu` component pins are upgraded to the
/// stable identity and every other legacy ordinal becomes `Auto`.
///
/// Existing `Device` component pins are durable identities in their own right.
/// Preserve each independently when that exact device exists on this boot;
/// otherwise return only that missing component to `Auto`. This keeps a valid
/// heterogeneous placement intact without letting a removed device strand the
/// replayed job.
pub(crate) fn resolve_replay_affinity(
    request: &mut mold_core::GenerateRequest,
    recorded_ordinal: Option<usize>,
    target_device_id: Option<&str>,
    resolve: impl Fn(&str) -> Option<usize>,
) -> Option<usize> {
    rebind_replay_accelerator_pins(request, recorded_ordinal.zip(target_device_id), &resolve);
    target_device_id.and_then(resolve)
}

fn rebind_replay_accelerator_pins(
    request: &mut mold_core::GenerateRequest,
    authoritative_mapping: Option<(usize, &str)>,
    resolve: &impl Fn(&str) -> Option<usize>,
) {
    fn rebind(
        device: &mut mold_core::DeviceRef,
        mapping: Option<(usize, &str)>,
        resolve: &impl Fn(&str) -> Option<usize>,
    ) {
        match device {
            mold_core::DeviceRef::Gpu { ordinal } => {
                *device = match mapping {
                    Some((recorded_ordinal, stable_id))
                        if *ordinal == recorded_ordinal && resolve(stable_id).is_some() =>
                    {
                        mold_core::DeviceRef::device(stable_id.to_string())
                    }
                    _ => mold_core::DeviceRef::Auto,
                };
            }
            mold_core::DeviceRef::Device { id } if resolve(id).is_none() => {
                *device = mold_core::DeviceRef::Auto;
            }
            _ => {}
        }
    }

    let Some(placement) = request.placement.as_mut() else {
        return;
    };
    rebind(&mut placement.text_encoders, authoritative_mapping, resolve);
    let Some(advanced) = placement.advanced.as_mut() else {
        return;
    };
    rebind(&mut advanced.transformer, authoritative_mapping, resolve);
    rebind(&mut advanced.vae, authoritative_mapping, resolve);
    for device in [
        advanced.clip_l.as_mut(),
        advanced.clip_g.as_mut(),
        advanced.t5.as_mut(),
        advanced.qwen.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        rebind(device, authoritative_mapping, resolve);
    }
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
    /// A feeder token no longer owns the row. The stale runtime must not run.
    Fenced,
}

/// Result of returning one token-owned row to the feeder backlog.
///
/// `Retry` retains the exact token-bearing ticket. Its drop is inert, so a
/// caller that cannot retry in this runtime leaves the row safely owned for
/// startup recovery rather than terminalizing or deleting it.
#[derive(Debug)]
pub enum RetainOutcome {
    Released,
    /// Durable cancellation won the exact transition race. Observer delivery
    /// must report cancellation rather than the requested success or failure.
    Cancelled,
    Stale,
    Retry {
        ticket: QueueTicket,
        error: anyhow::Error,
    },
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
    claim_token: Option<String>,
    settled: bool,
}

impl QueueTicket {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub(crate) fn retention_requested(&self) -> bool {
        self.journal.is_retaining()
    }

    /// The job produced its output. Delete the row unconditionally — a
    /// completed job must never be replayed, fence or no fence.
    pub fn complete(self) {
        self.complete_with_result(None);
    }

    pub fn complete_with_result(mut self, result_json: Option<&str>) {
        self.settled = true;
        if let Some(token) = self.claim_token.as_deref() {
            self.finish_claimed(
                token,
                QueueRowState::Running,
                mold_db::generation_batches::GenerationBatchTerminal {
                    state: mold_db::generation_batches::GenerationBatchTerminalState::Complete,
                    error: None,
                    terminal_error_json: None,
                    result_json,
                    completed_at_ms: now_ms(),
                },
            );
            return;
        }
        self.journal
            .set_batch_child_state(&self.id, "complete", None);
        self.journal.discard_id(&self.id);
    }

    /// Persist completion while preserving the exact token-owned ticket when
    /// SQLite cannot commit. Observer-bearing execution paths retry this
    /// operation before reporting success; a returned retry ticket has inert
    /// drop semantics and is therefore safe to retain for restart.
    pub(crate) fn complete_exact_with_result(mut self, result_json: Option<&str>) -> RetainOutcome {
        let Some(token) = self.claim_token.as_deref() else {
            self.complete_with_result(result_json);
            return RetainOutcome::Released;
        };
        #[cfg(test)]
        if self
            .journal
            .fail_completion_transition
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            self.settled = true;
            return RetainOutcome::Retry {
                ticket: self,
                error: anyhow::anyhow!("injected completion transition failure"),
            };
        }
        let terminal = mold_db::generation_batches::GenerationBatchTerminal {
            state: mold_db::generation_batches::GenerationBatchTerminalState::Complete,
            error: None,
            terminal_error_json: None,
            result_json,
            completed_at_ms: now_ms(),
        };
        let Some(db) = self.journal.db() else {
            self.settled = true;
            return RetainOutcome::Retry {
                ticket: self,
                error: anyhow::anyhow!("durable generation database is unavailable"),
            };
        };
        let candidate = self.journal.media_candidate(&self.id);
        match generation_batches::finish_claimed(
            db,
            &self.id,
            token,
            QueueRowState::Running,
            terminal,
        ) {
            Ok(commit) if commit.queue_deleted => {
                self.journal.cleanup_media_candidate(candidate);
                if commit.batch_child_updated {
                    self.journal.publish_state_committed(&self.id);
                }
                self.settled = true;
                if commit.cancelled {
                    RetainOutcome::Cancelled
                } else {
                    RetainOutcome::Released
                }
            }
            Ok(commit) => {
                if commit.batch_child_updated {
                    self.journal.publish_state_committed(&self.id);
                }
                self.settled = true;
                RetainOutcome::Stale
            }
            Err(error) => {
                self.settled = true;
                RetainOutcome::Retry {
                    ticket: self,
                    error,
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn complete_before_dispatch(self) {
        self.complete_before_dispatch_with_result(None);
    }

    pub(crate) fn complete_before_dispatch_with_result(mut self, result_json: Option<&str>) {
        self.settled = true;
        let Some(token) = self.claim_token.as_deref() else {
            self.journal.discard_id(&self.id);
            return;
        };
        self.finish_claimed(
            token,
            QueueRowState::Queued,
            mold_db::generation_batches::GenerationBatchTerminal {
                state: mold_db::generation_batches::GenerationBatchTerminalState::Complete,
                error: None,
                terminal_error_json: None,
                result_json,
                completed_at_ms: now_ms(),
            },
        );
    }

    /// The job was explicitly cancelled. Same unconditional delete as
    /// `complete`, and for the same reason: the user's decision outranks the
    /// fence.
    pub fn discard(mut self) {
        self.settled = true;
        if let Some(token) = self.claim_token.as_deref() {
            let terminal = mold_db::generation_batches::GenerationBatchTerminal {
                state: mold_db::generation_batches::GenerationBatchTerminalState::Cancelled,
                error: Some("Cancelled"),
                terminal_error_json: Some(r#"{"message":"Cancelled"}"#),
                result_json: None,
                completed_at_ms: now_ms(),
            };
            if !self.finish_claimed(token, QueueRowState::Running, terminal) {
                self.finish_claimed(token, QueueRowState::Queued, terminal);
            }
            return;
        }
        self.journal
            .set_batch_child_state(&self.id, "cancelled", Some("Cancelled"));
        self.journal.discard_id(&self.id);
    }

    pub fn fail(mut self, message: &str) {
        if self.journal.is_retaining() {
            self.settled = true;
            return;
        }
        self.settled = true;
        if self.claim_token.is_some() {
            self.fail_claimed(message);
            return;
        }
        self.journal
            .set_batch_child_state(&self.id, "failed", Some(message));
        self.journal.discard_id(&self.id);
    }

    /// The job never reached a worker, so leave its row exactly as it is.
    ///
    /// Distinct from `hold`: nothing is wrong with the job and the next boot
    /// should replay it normally. The `replay_seen` budget is what stops a row
    /// retrying forever if the condition persists.
    pub fn retain(mut self) -> RetainOutcome {
        if let Some(token) = self.claim_token.as_deref() {
            let Some(db) = self.journal.db() else {
                self.settled = true;
                return RetainOutcome::Released;
            };
            let now = now_ms();
            #[cfg(test)]
            let released = if self
                .journal
                .fail_claim_release
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                Err(anyhow::anyhow!("injected claim-release failure"))
            } else {
                generation_queue::release_claim(db, &self.id, token, now)
            };
            #[cfg(not(test))]
            let released = generation_queue::release_claim(db, &self.id, token, now);
            let retained = match released {
                Ok(true) => Ok(true),
                Ok(false) => generation_queue::requeue_running_claimed(db, &self.id, token, now),
                Err(error) => Err(error),
            };
            match retained {
                Ok(true) => {
                    if let (Some(db), Some(owner)) =
                        (self.journal.db(), self.journal.owner_uuid.as_deref())
                    {
                        if let Err(error) =
                            generation_batches::restore_child_after_retain(db, owner, &self.id, now)
                        {
                            tracing::warn!(
                                job = %self.id,
                                %error,
                                "could not restore a retained batch child"
                            );
                        }
                    }
                    self.journal.wake_feeder();
                    self.settled = true;
                    return RetainOutcome::Released;
                }
                Ok(false) => {
                    tracing::warn!(
                        job = %self.id,
                        "could not retain a feeder claim because its token was stale"
                    );
                    self.settled = true;
                    return RetainOutcome::Stale;
                }
                Err(error) => {
                    tracing::warn!(job = %self.id, %error, "could not release a retained feeder claim");
                    // The database still carries this exact claim. Make Drop
                    // inert while returning the ticket so an in-process
                    // retry can present the same token again.
                    self.settled = true;
                    return RetainOutcome::Retry {
                        ticket: self,
                        error,
                    };
                }
            }
        }
        self.settled = true;
        RetainOutcome::Released
    }

    /// Charge a dispatch attempt on the GPU owner thread, immediately before
    /// the model load. Over the cap, the row is held rather than deleted.
    pub fn claim_dispatch(&self) -> DispatchClaim {
        let Some(db) = self.journal.db() else {
            return DispatchClaim::Untracked;
        };
        let now = now_ms();
        let claimed = match self.claim_token.as_deref() {
            Some(token) => generation_queue::mark_dispatched_claimed(db, &self.id, token, now),
            None => generation_queue::mark_dispatched(db, &self.id, now),
        };
        match claimed {
            Ok(Some(attempts)) if attempts > self.journal.max_dispatch_attempts => {
                let cap = self.journal.max_dispatch_attempts;
                let reason =
                    format!("dispatch attempts exhausted ({attempts} > {cap}); held for review");
                match self.hold_owned(&reason, false, now) {
                    Ok(generation_batches::OwnedHold::Held) => {}
                    Ok(generation_batches::OwnedHold::Cancelled) => {
                        return DispatchClaim::Fenced;
                    }
                    Ok(generation_batches::OwnedHold::Fenced) => {
                        return if self.claim_token.is_some() {
                            DispatchClaim::Fenced
                        } else {
                            DispatchClaim::Exhausted { attempts, cap }
                        };
                    }
                    Err(error) => tracing::warn!(
                        job = %self.id,
                        error = %format!("{error:#}"),
                        "could not hold an exhausted durable queue row"
                    ),
                }
                DispatchClaim::Exhausted { attempts, cap }
            }
            Ok(Some(_)) => {
                self.journal
                    .set_batch_child_state(&self.id, "running", None);
                DispatchClaim::Granted
            }
            Ok(None) if self.claim_token.is_some() => DispatchClaim::Fenced,
            Ok(None) => DispatchClaim::Untracked,
            Err(error) => {
                tracing::warn!(
                    job = %self.id,
                    error = %format!("{error:#}"),
                    "could not claim a durable queue row; running the job anyway"
                );
                if self.claim_token.is_some() {
                    DispatchClaim::Fenced
                } else {
                    DispatchClaim::Untracked
                }
            }
        }
    }

    /// Park the row: listed, never auto-run, and no longer owned by a ticket.
    pub fn hold(mut self, reason: &str) -> RetainOutcome {
        match self.hold_owned(reason, false, now_ms()) {
            Ok(_) => {
                self.settled = true;
                RetainOutcome::Released
            }
            Err(error) => {
                tracing::warn!(
                    job = %self.id,
                    error = %format!("{error:#}"),
                    "could not hold a durable queue row; returning it to the replay backlog"
                );
                self.retain()
            }
        }
    }

    /// Park a deferred-preparation failure without discarding its durable
    /// request or media. Only these explicitly recoverable holds may be
    /// returned to the queue through the retry API.
    pub fn hold_retryable(mut self, reason: &str) -> RetainOutcome {
        match self.hold_owned(reason, true, now_ms()) {
            Ok(_) => {
                self.settled = true;
                RetainOutcome::Released
            }
            Err(error) => {
                tracing::warn!(
                    job = %self.id,
                    error = %format!("{error:#}"),
                    "could not hold a retryable durable queue row; returning it to the replay backlog"
                );
                self.retain()
            }
        }
    }

    /// Attempt the exact requested hold without silently converting a failed
    /// persistence transition into a queued replay. Execution observers use
    /// this seam so they cannot be told a job failed while its row still says
    /// `running`. A retry ticket keeps the exact claim token and has inert
    /// drop semantics until the caller retries this same transition.
    pub(crate) fn hold_exact(mut self, reason: &str, retryable: bool) -> RetainOutcome {
        match self.hold_owned(reason, retryable, now_ms()) {
            Ok(generation_batches::OwnedHold::Held) => {
                self.settled = true;
                RetainOutcome::Released
            }
            Ok(generation_batches::OwnedHold::Cancelled) => {
                self.settled = true;
                RetainOutcome::Cancelled
            }
            Ok(generation_batches::OwnedHold::Fenced) => {
                self.settled = true;
                RetainOutcome::Stale
            }
            Err(error) => {
                tracing::warn!(
                    job = %self.id,
                    error = %format!("{error:#}"),
                    retryable,
                    "could not persist the requested durable hold"
                );
                self.settled = true;
                RetainOutcome::Retry {
                    ticket: self,
                    error,
                }
            }
        }
    }

    fn hold_owned(
        &self,
        reason: &str,
        retryable: bool,
        now: i64,
    ) -> anyhow::Result<generation_batches::OwnedHold> {
        #[cfg(test)]
        if self
            .journal
            .fail_hold_transition
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(anyhow::anyhow!("injected hold transition failure"));
        }
        let (Some(db), Some(owner)) = (self.journal.db(), self.journal.owner_uuid.as_deref())
        else {
            return Ok(generation_batches::OwnedHold::Fenced);
        };
        let candidate = self.journal.media_candidate(&self.id);
        let outcome = generation_batches::hold_owned(
            db,
            owner,
            &self.id,
            self.claim_token.as_deref(),
            reason,
            retryable,
            now,
        )?;
        if outcome == generation_batches::OwnedHold::Cancelled {
            self.journal.cleanup_media_candidate(candidate);
        }
        if matches!(
            outcome,
            generation_batches::OwnedHold::Held | generation_batches::OwnedHold::Cancelled
        ) {
            self.journal.publish_state_committed(&self.id);
        }
        Ok(outcome)
    }

    fn finish_claimed(
        &self,
        token: &str,
        expected: QueueRowState,
        terminal: mold_db::generation_batches::GenerationBatchTerminal<'_>,
    ) -> bool {
        let Some(db) = self.journal.db() else {
            return false;
        };
        let candidate = self.journal.media_candidate(&self.id);
        match generation_batches::finish_claimed(db, &self.id, token, expected, terminal) {
            Ok(commit) => {
                if commit.queue_deleted {
                    self.journal.cleanup_media_candidate(candidate);
                }
                if commit.batch_child_updated {
                    self.journal.publish_state_committed(&self.id);
                }
                commit.queue_deleted
            }
            Err(error) => {
                tracing::warn!(job = %self.id, %error, "could not atomically settle a claimed generation");
                false
            }
        }
    }

    fn fail_claimed(&self, message: &str) {
        let Some(token) = self.claim_token.as_deref() else {
            return;
        };
        let cancelled = self
            .journal
            .db()
            .zip(self.journal.owner_uuid.as_deref())
            .and_then(|(db, owner)| {
                generation_batches::child_cancel_requested(db, owner, &self.id).ok()
            })
            .unwrap_or(false);
        let terminal_message = if cancelled { "Cancelled" } else { message };
        let terminal_error_json = serde_json::json!({ "message": terminal_message }).to_string();
        let terminal = mold_db::generation_batches::GenerationBatchTerminal {
            state: if cancelled {
                mold_db::generation_batches::GenerationBatchTerminalState::Cancelled
            } else {
                mold_db::generation_batches::GenerationBatchTerminalState::Failed
            },
            error: Some(terminal_message),
            terminal_error_json: Some(&terminal_error_json),
            result_json: None,
            completed_at_ms: now_ms(),
        };
        if !self.finish_claimed(token, QueueRowState::Running, terminal) {
            self.finish_claimed(token, QueueRowState::Queued, terminal);
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
        if self.claim_token.is_some() {
            self.fail_claimed("generation ended before publishing an output");
            return;
        }
        self.journal.set_batch_child_state(
            &self.id,
            "failed",
            Some("generation ended before publishing an output"),
        );
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
            durable_media_ready: AtomicBool::new(false),
            durable_media_issues: std::sync::RwLock::new(Vec::new()),
            queue_media_lifecycle: OnceLock::new(),
            queue_media_admission: OnceLock::new(),
            events: OnceLock::new(),
            _owner_claim: None,
            fail_completion_lookup: AtomicBool::new(false),
            fail_batch_lookup: AtomicUsize::new(0),
            fail_claim_release: AtomicBool::new(false),
            fail_hold_transition: AtomicBool::new(false),
            fail_completion_transition: AtomicUsize::new(0),
            retain: AtomicBool::new(false),
            max_bytes: DEFAULT_JOURNAL_MAX_BYTES,
            max_dispatch_attempts: DEFAULT_MAX_DISPATCH_ATTEMPTS,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
            feeder_notify: tokio::sync::Notify::new(),
            durable_transition_gate: Arc::new(tokio::sync::Mutex::new(())),
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

    fn request_for_model(model: &str) -> mold_core::GenerateRequest {
        let mut request = request();
        request.model = model.to_string();
        request
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
            target_device_id: None,
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
                media_set_id: None,
                admission_authority: None,
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

    #[test]
    fn direct_record_is_feeder_invisible_until_runtime_recovery() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("live-direct", &request, Path::new("/gallery")))
            .expect("an ordinary gallery-bound generation is durable");

        assert!(
            journal.claim_next_feeder().unwrap().is_none(),
            "the direct submitter owns this row while its runtime is alive"
        );

        let recovered = journal.recover_feeder_runtime().unwrap();
        assert_eq!(recovered.claims_cleared, 1);
        let replay = journal
            .claim_next_feeder()
            .unwrap()
            .expect("startup recovery makes the retained direct row replayable");
        assert_eq!(replay.row.id, "live-direct");

        drop(ticket);
        journal
            .attach_claimed(&replay.row.id, replay.claim_token)
            .discard();
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
    fn shared_mold_home_cannot_cancel_another_live_queue_owner() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let first = QueueJournal::new(db.clone(), Some(home.path()), "instance-a");
        let second = QueueJournal::new(db.clone(), Some(home.path()), "instance-b");
        assert_ne!(first.owner_uuid(), second.owner_uuid());

        let second_owner = second.owner_uuid().unwrap();
        seed_row(db.as_ref().as_ref().unwrap(), second_owner, "foreign-job");

        assert!(!first.owns_cancellable_row("foreign-job").unwrap());
        assert!(!first.cancel_id("foreign-job").unwrap());
        assert_eq!(second.list_all().len(), 1);
        assert!(second.cancel_id("foreign-job").unwrap());
        assert!(second.list_all().is_empty());
    }

    #[test]
    fn cancelling_a_held_batch_child_is_not_a_false_acknowledgement() {
        let journal = journal_with_db();
        let request = request();
        journal
            .record_batch(BatchJournalAdmission {
                id: "held-batch",
                client_batch_id: "held-client",
                request_sha256: "held-sha",
                children: &[admission("held-child", &request, Path::new("/gallery"))],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        journal
            .attach_claimed(&claim.row.id, claim.claim_token)
            .hold("operator review");

        assert_eq!(journal.list_all()[0].state, QueueRowState::Held);
        assert!(journal.cancel_id("held-child").unwrap());
        assert!(journal.list_all().is_empty());
        let child = &journal.generation_batch("held-batch").unwrap().children[0];
        assert_eq!(child.state, "cancelled");
    }

    #[test]
    fn cancellation_cannot_be_erased_by_a_late_feeder_hold() {
        let journal = journal_with_db();
        let request = request();
        journal
            .record_batch(BatchJournalAdmission {
                id: "racing-hold-batch",
                client_batch_id: "racing-hold-client",
                request_sha256: "racing-hold-sha",
                children: &[admission(
                    "racing-hold-child",
                    &request,
                    Path::new("/gallery"),
                )],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        let ticket = journal.attach_claimed(&claim.row.id, claim.claim_token);

        assert!(journal.cancel_id("racing-hold-child").unwrap());
        ticket.hold("server gallery output is disabled");

        assert!(journal.list_all().is_empty());
        let child = &journal
            .generation_batch("racing-hold-batch")
            .unwrap()
            .children[0];
        assert_eq!(child.state, "cancelled");
    }

    #[test]
    fn exact_failure_settlement_reports_when_cancellation_won() {
        let journal = journal_with_db();
        let request = request();
        journal
            .record_batch(BatchJournalAdmission {
                id: "exact-cancel-batch",
                client_batch_id: "exact-cancel-client",
                request_sha256: "exact-cancel-sha",
                children: &[admission(
                    "exact-cancel-child",
                    &request,
                    Path::new("/gallery"),
                )],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        let ticket = journal.attach_claimed(&claim.row.id, claim.claim_token);
        assert_eq!(ticket.claim_dispatch(), DispatchClaim::Granted);

        assert!(journal.cancel_id("exact-cancel-child").unwrap());
        let mut ticket = Some(ticket);
        let outcome = crate::durable_generation_settlement::settle_blocking(
            &mut ticket,
            crate::durable_generation_settlement::DurableDisposition::RetryableHold,
            "inference failed",
        );

        assert_eq!(
            outcome,
            crate::durable_generation_settlement::SettlementOutcome::Cancelled
        );
        assert!(journal.list_all().is_empty());
        let child = &journal
            .generation_batch("exact-cancel-batch")
            .unwrap()
            .children[0];
        assert_eq!(child.state, "cancelled");
        assert_eq!(child.error.as_deref(), Some("Cancelled"));
    }

    #[test]
    fn failed_hold_transition_returns_the_exact_claim_to_replay() {
        let journal = journal_with_db();
        let request = request();
        journal
            .record_batch(BatchJournalAdmission {
                id: "failed-hold-batch",
                client_batch_id: "failed-hold-client",
                request_sha256: "failed-hold-sha",
                children: &[admission(
                    "failed-hold-child",
                    &request,
                    Path::new("/gallery"),
                )],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        let ticket = journal.attach_claimed(&claim.row.id, claim.claim_token);
        journal.fail_hold_transition_for_tests();

        assert!(matches!(
            ticket.hold_retryable("temporary preparation failure"),
            RetainOutcome::Released
        ));

        let replay = journal
            .claim_next_feeder()
            .unwrap()
            .expect("a failed hold must not strand or delete accepted work");
        assert_eq!(replay.row.id, "failed-hold-child");
        assert_eq!(replay.row.state, QueueRowState::Queued);
        journal
            .attach_claimed(&replay.row.id, replay.claim_token)
            .discard();
    }

    #[test]
    fn completion_removes_the_row_even_behind_the_fence() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .unwrap();

        journal.retain_all();
        assert_eq!(ticket.claim_dispatch(), DispatchClaim::Granted);
        ticket.complete();

        assert!(rows(&journal).is_empty());
    }

    #[test]
    fn durable_state_event_is_published_after_terminal_commit() {
        let journal = journal_with_db();
        let events = crate::events::EventBroadcaster::new();
        journal.install_event_broadcaster(events.clone()).unwrap();
        let request = request();
        journal
            .record_batch(BatchJournalAdmission {
                id: "event-batch",
                client_batch_id: "event-client",
                request_sha256: "event-sha",
                children: &[admission("event-child", &request, Path::new("/gallery"))],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        let ticket = journal.attach_claimed(&claim.row.id, claim.claim_token);
        assert_eq!(ticket.claim_dispatch(), DispatchClaim::Granted);
        let mut receiver = events.subscribe();

        ticket.complete_with_result(Some(r#"{"filename":"event.png"}"#));

        assert!(matches!(
            receiver.try_recv().unwrap(),
            mold_core::ServerEvent::JobStateCommitted { id } if id == "event-child"
        ));
        let detail = journal
            .durable_generation_batch("event-batch")
            .unwrap()
            .unwrap();
        let child = &detail.children[0];
        assert_eq!(child.state, "complete");
        assert_eq!(
            child.result_json.as_deref(),
            Some(r#"{"filename":"event.png"}"#)
        );
    }

    #[test]
    fn bulk_cancel_publishes_one_post_commit_host_invalidation() {
        let journal = journal_with_db();
        let events = crate::events::EventBroadcaster::new();
        journal.install_event_broadcaster(events.clone()).unwrap();
        let request = request();
        journal
            .record_batch(BatchJournalAdmission {
                id: "bulk-event-batch",
                client_batch_id: "bulk-event-client",
                request_sha256: "bulk-event-sha",
                children: &[
                    admission("bulk-event-claimed", &request, Path::new("/gallery")),
                    admission("bulk-event-deep", &request, Path::new("/gallery")),
                ],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(claim.row.id, "bulk-event-claimed");
        let ticket = journal.attach_claimed(&claim.row.id, claim.claim_token);
        let mut receiver = events.subscribe();

        assert_eq!(
            journal
                .cancel_all_queued(&["bulk-event-claimed".to_string()])
                .unwrap(),
            1
        );

        assert!(matches!(
            receiver.try_recv().unwrap(),
            mold_core::ServerEvent::GenerationStatesCommitted
        ));
        assert!(receiver.try_recv().is_err(), "bulk cancel emits one hint");
        let detail = journal
            .durable_generation_batch("bulk-event-batch")
            .unwrap()
            .unwrap();
        assert_eq!(detail.children[0].state, "cancelling");
        assert_eq!(detail.children[1].state, "cancelled");

        ticket.discard();
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
    fn private_h3_fl2va_without_references_is_non_durable() {
        let journal = journal_with_db();
        let request = request_for_model(mold_core::minimax_h3::FL2VA_COMFY);

        assert!(journal
            .record(admission("private-fl2va", &request, Path::new("/gallery")))
            .is_none());
        assert!(rows(&journal).is_empty());
    }

    #[test]
    fn record_batch_rejects_every_h3_capability_contract_by_name() {
        let journal = journal_with_db();
        let models = [
            mold_core::minimax_h3::FL2VA_OFFICIAL,
            mold_core::minimax_h3::REF2VA_OFFICIAL,
            mold_core::minimax_h3::FL2VA_COMFY,
            mold_core::minimax_h3::REF2VA_COMFY,
            mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
            mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P,
            mold_core::minimax_h3::FL2VA_COMFY_NVFP4,
            mold_core::minimax_h3::REF2VA_COMFY_NVFP4,
        ];

        for (index, model) in models.into_iter().enumerate() {
            assert!(
                mold_core::minimax_h3::capability_contract_for_model(model).is_some(),
                "test model must remain inside the authoritative H3 capability contract: {model}"
            );
            let request = request_for_model(model);
            let child_id = format!("private-h3-{index}");
            let child = admission(&child_id, &request, Path::new("/gallery"));
            let error = journal
                .record_batch(BatchJournalAdmission {
                    id: "private-h3-batch",
                    client_batch_id: "private-h3-client-batch",
                    request_sha256: "private-h3-request",
                    children: &[child],
                })
                .expect_err("private H3 must never enter the durable batch journal");

            assert_eq!(
                error,
                "heterogeneous batches cannot persist private MiniMax H3 requests"
            );
            assert!(rows(&journal).is_empty());
        }
    }

    #[test]
    fn public_non_h3_generation_remains_durable() {
        let journal = journal_with_db();
        let request = request();
        assert!(
            mold_core::minimax_h3::capability_contract_for_model(&request.model).is_none(),
            "the ordinary public control must remain outside H3 authority"
        );

        let ticket = journal
            .record(admission("public-non-h3", &request, Path::new("/gallery")))
            .expect("ordinary public generation remains durable");
        assert_eq!(rows(&journal), vec!["public-non-h3"]);
        ticket.discard();
    }

    #[test]
    fn an_oversized_request_runs_without_being_journaled() {
        let db = MetadataDb::open_in_memory().unwrap();
        let owner = "test-owner".to_string();
        let journal = Arc::new(QueueJournal {
            db: Arc::new(Some(db)),
            owner_uuid: Some(owner),
            durable_media_ready: AtomicBool::new(false),
            durable_media_issues: std::sync::RwLock::new(Vec::new()),
            queue_media_lifecycle: OnceLock::new(),
            queue_media_admission: OnceLock::new(),
            events: OnceLock::new(),
            _owner_claim: None,
            fail_completion_lookup: AtomicBool::new(false),
            fail_batch_lookup: AtomicUsize::new(0),
            fail_claim_release: AtomicBool::new(false),
            fail_hold_transition: AtomicBool::new(false),
            fail_completion_transition: AtomicUsize::new(0),
            retain: AtomicBool::new(false),
            max_bytes: 64,
            max_dispatch_attempts: DEFAULT_MAX_DISPATCH_ATTEMPTS,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
            feeder_notify: tokio::sync::Notify::new(),
            durable_transition_gate: Arc::new(tokio::sync::Mutex::new(())),
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
            durable_media_ready: AtomicBool::new(false),
            durable_media_issues: std::sync::RwLock::new(Vec::new()),
            queue_media_lifecycle: OnceLock::new(),
            queue_media_admission: OnceLock::new(),
            events: OnceLock::new(),
            _owner_claim: None,
            fail_completion_lookup: AtomicBool::new(false),
            fail_batch_lookup: AtomicUsize::new(0),
            fail_claim_release: AtomicBool::new(false),
            fail_hold_transition: AtomicBool::new(false),
            fail_completion_transition: AtomicUsize::new(0),
            retain: AtomicBool::new(false),
            max_bytes: DEFAULT_JOURNAL_MAX_BYTES,
            max_dispatch_attempts: 2,
            max_replay_seen: DEFAULT_MAX_REPLAY_SEEN,
            feeder_notify: tokio::sync::Notify::new(),
            durable_transition_gate: Arc::new(tokio::sync::Mutex::new(())),
        });
        let request = request();
        let first = journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .unwrap();

        assert_eq!(first.claim_dispatch(), DispatchClaim::Granted);
        journal.recover_feeder_runtime().unwrap();
        let second_claim = journal.claim_next_feeder().unwrap().unwrap();
        let second = journal.attach_claimed("job-1", second_claim.claim_token);
        assert_eq!(second.claim_dispatch(), DispatchClaim::Granted);
        journal.recover_feeder_runtime().unwrap();
        let third_claim = journal.claim_next_feeder().unwrap().unwrap();
        let third = journal.attach_claimed("job-1", third_claim.claim_token);
        assert_eq!(
            third.claim_dispatch(),
            DispatchClaim::Exhausted {
                attempts: 3,
                cap: 2
            }
        );
        drop(first);
        drop(second);
        drop(third);

        // Held rows stay listed and are no longer replayable.
        assert_eq!(rows(&journal), vec!["job-1"]);
        let held = journal.list_all().pop().unwrap();
        assert_eq!(held.state, QueueRowState::Held);
        assert!(held.held_reason.is_some());
    }

    #[test]
    fn transient_retain_failure_returns_the_same_token_for_an_in_process_retry() {
        let journal = journal_with_db();
        let request = request();
        let ticket = journal
            .record(admission("retry-release", &request, Path::new("/gallery")))
            .unwrap();
        journal
            .fail_claim_release
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let retry = match ticket.retain() {
            RetainOutcome::Retry { ticket, error } => {
                assert!(error.to_string().contains("injected claim-release failure"));
                ticket
            }
            other => panic!("transient release must preserve its ticket, got {other:?}"),
        };
        assert!(
            journal.claim_next_feeder().unwrap().is_none(),
            "the failed release must leave the exact token as database authority"
        );

        assert!(matches!(retry.retain(), RetainOutcome::Released));
        assert_eq!(
            journal.claim_next_feeder().unwrap().unwrap().row.id,
            "retry-release"
        );
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

    #[test]
    fn direct_and_batch_admission_persist_the_authoritative_stable_device_id() {
        let journal = journal_with_db();
        let request = request();
        let direct = JournalAdmission {
            id: "direct-stable",
            request: &request,
            output_dir: Some(Path::new("/gallery")),
            target_gpu: Some(2),
            target_device_id: Some("cuda:stable-direct"),
            completion_payload: SseCompletionPayload::Full,
            batch_child: false,
            carries_reference_authority: false,
        };
        let direct_ticket = journal.record(direct).expect("direct durable row");

        let batch_child = JournalAdmission {
            id: "batch-stable",
            request: &request,
            output_dir: Some(Path::new("/gallery")),
            target_gpu: Some(5),
            target_device_id: Some("cuda:stable-batch"),
            completion_payload: SseCompletionPayload::MetadataOnly,
            batch_child: false,
            carries_reference_authority: false,
        };
        let (_, inserted) = journal
            .record_batch(BatchJournalAdmission {
                id: "stable-batch",
                client_batch_id: "stable-client",
                request_sha256: "stable-sha",
                children: &[batch_child],
            })
            .expect("batch durable row");
        assert!(inserted);

        let rows = journal.list_all();
        let direct = rows.iter().find(|row| row.id == "direct-stable").unwrap();
        assert_eq!(direct.target_gpu, Some(2));
        assert_eq!(
            direct.target_device_id.as_deref(),
            Some("cuda:stable-direct")
        );
        let batch = rows.iter().find(|row| row.id == "batch-stable").unwrap();
        assert_eq!(batch.target_gpu, Some(5));
        assert_eq!(batch.target_device_id.as_deref(), Some("cuda:stable-batch"));

        direct_ticket.discard();
    }

    #[test]
    fn replay_affinity_renumbers_legacy_pins_without_flattening_component_identities() {
        let mut request = request();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::device("cuda:text"),
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(2),
                vae: mold_core::DeviceRef::device("cuda:vae"),
                clip_l: Some(mold_core::DeviceRef::device("cuda:clip")),
                clip_g: Some(mold_core::DeviceRef::Cpu),
                ..mold_core::AdvancedPlacement::default()
            }),
        });

        // The journal payload crosses a serialization boundary before a
        // later process hydrates it during restart replay.
        let json = serde_json::to_string(&request).unwrap();
        let mut request: mold_core::GenerateRequest = serde_json::from_str(&json).unwrap();

        let target = super::resolve_replay_affinity(
            &mut request,
            Some(2),
            Some("cuda:stable"),
            |id| match id {
                "cuda:stable" => Some(7),
                "cuda:text" => Some(3),
                "cuda:vae" => Some(4),
                "cuda:clip" => Some(5),
                _ => None,
            },
        );

        assert_eq!(target, Some(7));
        let placement = request.placement.unwrap();
        assert_eq!(
            placement.text_encoders,
            mold_core::DeviceRef::device("cuda:text")
        );
        let advanced = placement.advanced.unwrap();
        assert_eq!(
            advanced.transformer,
            mold_core::DeviceRef::device("cuda:stable")
        );
        assert_eq!(advanced.vae, mold_core::DeviceRef::device("cuda:vae"));
        assert_eq!(
            advanced.clip_l,
            Some(mold_core::DeviceRef::device("cuda:clip"))
        );
        assert_eq!(advanced.clip_g, Some(mold_core::DeviceRef::Cpu));
    }

    #[test]
    fn replay_without_top_level_affinity_preserves_stable_component_identities() {
        let mut request = request();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::device("cuda:text"),
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(2),
                vae: mold_core::DeviceRef::device("cuda:vae"),
                ..mold_core::AdvancedPlacement::default()
            }),
        });

        let target = super::resolve_replay_affinity(&mut request, Some(2), None, |id| {
            matches!(id, "cuda:text" | "cuda:vae").then_some(1)
        });

        assert_eq!(target, None);
        let placement = request.placement.unwrap();
        assert_eq!(
            placement.text_encoders,
            mold_core::DeviceRef::device("cuda:text")
        );
        let advanced = placement.advanced.unwrap();
        assert_eq!(advanced.transformer, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.vae, mold_core::DeviceRef::device("cuda:vae"));
    }

    #[test]
    fn replay_with_unresolved_top_level_affinity_preserves_component_identity() {
        let mut request = request();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Auto,
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(4),
                vae: mold_core::DeviceRef::device("cuda:vae"),
                qwen: Some(mold_core::DeviceRef::device("cuda:qwen")),
                ..mold_core::AdvancedPlacement::default()
            }),
        });

        let target =
            super::resolve_replay_affinity(&mut request, Some(4), Some("cuda:dispatch"), |id| {
                matches!(id, "cuda:vae" | "cuda:qwen").then_some(1)
            });

        assert_eq!(target, None);
        let advanced = request.placement.unwrap().advanced.unwrap();
        assert_eq!(advanced.transformer, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.vae, mold_core::DeviceRef::device("cuda:vae"));
        assert_eq!(
            advanced.qwen,
            Some(mold_core::DeviceRef::device("cuda:qwen"))
        );
    }

    #[test]
    fn replay_never_reuses_a_legacy_component_ordinal_without_an_exact_mapping() {
        let mut request = request();
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::gpu(3),
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(2),
                vae: mold_core::DeviceRef::gpu(3),
                ..mold_core::AdvancedPlacement::default()
            }),
        });

        let target =
            super::resolve_replay_affinity(&mut request, Some(2), Some("cuda:recorded"), |_| {
                Some(9)
            });

        assert_eq!(target, Some(9));
        let placement = request.placement.unwrap();
        assert_eq!(placement.text_encoders, mold_core::DeviceRef::Auto);
        let advanced = placement.advanced.unwrap();
        assert_eq!(
            advanced.transformer,
            mold_core::DeviceRef::device("cuda:recorded")
        );
        assert_eq!(advanced.vae, mold_core::DeviceRef::Auto);
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

    /// The plural wire shape is the SAME photograph, so it is excluded the same
    /// way. A predicate that only knew about `id_image` would journal every
    /// multi-reference request's faces into `mold.db` — the exact outcome the
    /// exclusion exists to prevent (#1226).
    #[test]
    fn a_multi_photograph_request_is_excluded_from_the_database_too() {
        let journal = journal_with_db();
        let mut request = request();
        request.model = "flux-dev:q4".to_string();
        request.id_images = Some(vec![
            b"\x89PNG\r\n\x1a\n-first-face".to_vec(),
            b"\x89PNG\r\n\x1a\n-second-face".to_vec(),
        ]);
        request.id_image_names = Some(vec!["one.png".to_string(), "two.png".to_string()]);

        assert!(journal
            .record(admission("job-1", &request, Path::new("/gallery")))
            .is_none());
        assert!(rows(&journal).is_empty());
    }
}
