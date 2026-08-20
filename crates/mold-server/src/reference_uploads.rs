//! Authenticated, bounded ingress for MiniMax H3 reference media.
//!
//! Bearer handles are carried only in headers on stable URLs. The store keeps
//! only salted handle digests, stages under the shared `MOLD_HOME` cache, and
//! converts every public authority into a private [`ResolvedReferenceSet`]
//! before a request may enter queue or recovery code.

use axum::{
    body::Body,
    extract::{Extension, State},
    http::{header, HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use mold_core::{
    minimax_h3, GenerationReference, GenerationReferenceAuthority, GenerationReferenceMetadata,
    GenerationReferenceProvenance, ReferenceUploadCompleteResponse, ReferenceUploadSessionRequest,
    ReferenceUploadSessionResponse, ReferenceUploadSlot,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::io::AsyncWriteExt as _;
use tokio_stream::StreamExt as _;
use tokio_util::sync::CancellationToken;

use crate::{auth::ApiKeyAuthenticated, routes::ApiError, state::AppState};

pub const UPLOAD_HANDLE_HEADER: &str = "x-mold-reference-upload";
pub const SESSION_HANDLE_HEADER: &str = "x-mold-reference-upload-session";
pub const MAX_REFERENCE_UPLOAD_FILE_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_SESSION_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_HOST_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_SESSION_REQUEST_BYTES: usize = 256 * 1024;
pub const MAX_REFERENCE_UPLOAD_SESSIONS_PER_IDENTITY: usize = 4;
pub const MAX_CONCURRENT_REFERENCE_UPLOADS: usize = 4;
pub const REFERENCE_UPLOAD_SESSION_TTL: Duration = Duration::from_secs(30 * 60);

#[derive(Clone)]
pub struct ReferenceUploadStore {
    inner: Arc<tokio::sync::Mutex<StoreInner>>,
    writers: Arc<tokio::sync::Semaphore>,
    root: Arc<PathBuf>,
    _lifetime: Arc<StoreLifetime>,
    resolved_bytes: Arc<AtomicU64>,
    handle_salt: Arc<[u8; 32]>,
}

struct StoreLifetime {
    root: PathBuf,
    /// Advisory lock proving this root is live, so a peer server booting
    /// against the same `MOLD_HOME` cannot mistake it for an orphan. Released
    /// by the OS however this process ends.
    ///
    /// Taken lazily, when the root is first created for authorized use — the
    /// store must not materialize a staging directory before then, and a lock
    /// is a property of a root that exists. The gap between creating the root
    /// and locking it is safe because an unlocked root is never swept.
    claim: std::sync::OnceLock<Option<std::fs::File>>,
}

impl StoreLifetime {
    fn ensure_claimed(&self) {
        self.claim.get_or_init(|| claim_staging_root(&self.root));
    }
}

impl Drop for StoreLifetime {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("failed to remove reference-upload runtime staging: {error}");
            }
        }
    }
}

impl std::fmt::Debug for ReferenceUploadStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUploadStore")
            .field("root", &"<private MOLD_HOME cache>")
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct StoreInner {
    sessions: HashMap<String, UploadSession>,
    upload_sessions: HashMap<String, String>,
    reserved_bytes: u64,
}

struct UploadSession {
    identity: String,
    scope_sha256: String,
    request: mold_core::GenerateRequest,
    expires_at_ms: u64,
    dir: PathBuf,
    cancel: CancellationToken,
    slots: HashMap<String, UploadSlot>,
    reserved_bytes: u64,
    consuming: bool,
}

struct UploadSlot {
    reference: u32,
    descriptor: GenerationReference,
    path: PathBuf,
    state: UploadState,
}

enum UploadState {
    Empty,
    Uploading {
        reserved_bytes: u64,
    },
    Complete {
        size_bytes: u64,
        metadata: Box<GenerationReferenceMetadata>,
    },
}

struct UploadSource {
    path: PathBuf,
    metadata: GenerationReferenceMetadata,
}

/// Private resolved media authority. Paths never enter JSON, queue metadata,
/// logs, or durable recovery state; dropping the last owner removes staging.
pub struct ResolvedReferenceSet {
    entries: Vec<ResolvedReference>,
    fingerprint: String,
    hold: Arc<ResolvedReferenceHold>,
}

/// The quota reservation and staging directory for one resolved set.
///
/// These were drop-scoped to the set itself, which raced admission: cancelling
/// a queued job drops the set, releasing the quota and unlinking the staging
/// while a spawned preparation task is still decoding from descriptors that
/// survive the unlink. Holding both behind an `Arc` means the last holder
/// releases them — the quota exactly once, and the directory only when no
/// admission view can still be reading it.
struct ResolvedReferenceHold {
    root: PathBuf,
    _quota: ResolvedQuotaLease,
    _store_lifetime: Arc<StoreLifetime>,
}

impl Drop for ResolvedReferenceHold {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("failed to remove private reference staging: {error}");
            }
        }
    }
}

struct ResolvedQuotaLease {
    reserved_bytes: u64,
    counter: Arc<AtomicU64>,
}

impl ResolvedQuotaLease {
    fn new(counter: Arc<AtomicU64>) -> Self {
        Self {
            reserved_bytes: 0,
            counter,
        }
    }

    fn adopt_reserved(&mut self, bytes: u64) {
        self.reserved_bytes = self.reserved_bytes.saturating_add(bytes);
    }
}

impl Drop for ResolvedQuotaLease {
    fn drop(&mut self) {
        self.counter
            .fetch_sub(self.reserved_bytes, Ordering::AcqRel);
    }
}

/// Plain staged-file coordinates: probed metadata plus the private path. It
/// owns no descriptor, quota, or lifetime, so copying one grants no authority
/// on its own — every use still opens and re-verifies the file.
#[derive(Clone)]
pub struct ResolvedReference {
    pub metadata: GenerationReferenceMetadata,
    pub path: PathBuf,
}

impl std::fmt::Debug for ResolvedReferenceSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedReferenceSet")
            .field("entries", &self.entries.len())
            .field("fingerprint", &self.fingerprint)
            .field("root", &"<redacted>")
            .finish()
    }
}

impl ResolvedReferenceSet {
    pub fn entries(&self) -> &[ResolvedReference] {
        &self.entries
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Payload-free authority fixture for owner/publication contract tests.
    /// It deliberately contains no opened media and therefore must never be
    /// passed to `inference_bindings`; production sets are built only by the
    /// authenticated resolver above.
    #[cfg(test)]
    pub(crate) fn authority_only_for_test(request: &mold_core::GenerateRequest) -> Self {
        let references = request.references.as_deref().unwrap_or_default();
        let metadata = references
            .iter()
            .enumerate()
            .map(|(index, reference)| reference.redacted_metadata_lossless(index))
            .collect::<Vec<_>>();
        let root = std::env::temp_dir().join(format!(
            "mold-h3-reference-authority-{}",
            uuid::Uuid::new_v4()
        ));
        Self {
            entries: Vec::new(),
            fingerprint: mold_core::generation_reference_fingerprint(&metadata),
            hold: Arc::new(ResolvedReferenceHold {
                root: root.clone(),
                _quota: ResolvedQuotaLease::new(Arc::new(AtomicU64::new(0))),
                // A test/synthetic set owns no live staging root to claim.
                _store_lifetime: Arc::new(StoreLifetime {
                    claim: std::sync::OnceLock::new(),
                    root,
                }),
            }),
        }
    }

    /// Bind the payload-free queued request back to the private staged files
    /// immediately before inference. The order, complete probed metadata, and
    /// aggregate fingerprint must all still match; paths never enter the
    /// request, scheduler plan, logs, SSE, or gallery metadata.
    pub fn inference_bindings(
        &self,
        request: &mold_core::GenerateRequest,
        cancellation: Option<&mold_inference::InferenceCancellationToken>,
    ) -> anyhow::Result<Vec<mold_inference::GenerationReferenceBinding>> {
        mint_inference_bindings(&self.entries, &self.fingerprint, request, cancellation)
    }

    /// A payload-free, quota-free projection of this set for H3 admission.
    ///
    /// Admission runs on a spawned task, so it needs an owned `'static` value,
    /// but `ResolvedReferenceSet` deliberately is not `Clone`: its quota lease
    /// is a drop guard and cloning it would release the reservation twice, and
    /// its `Drop` removes the staging directory. The view therefore carries
    /// neither — the set on the job keeps owning the quota and the directory.
    ///
    /// It holds paths and probed metadata, never bytes, and mints its bindings
    /// through the same verifier the runtime uses.
    ///
    /// The view DOES keep the staged files and their quota reservation alive:
    /// it shares the set's hold, so cancelling and dropping the owning job
    /// mid-preparation cannot unlink the staging or release the reservation
    /// out from under an in-flight decode. The quota is still released exactly
    /// once, by whichever holder drops last.
    pub fn admission_view(&self) -> ResolvedReferenceAdmissionView {
        ResolvedReferenceAdmissionView {
            entries: self.entries.clone(),
            fingerprint: self.fingerprint.clone(),
            _hold: Arc::clone(&self.hold),
        }
    }
}

/// See [`ResolvedReferenceSet::admission_view`].
#[derive(Clone)]
pub struct ResolvedReferenceAdmissionView {
    entries: Vec<ResolvedReference>,
    fingerprint: String,
    /// Keeps the staged files and their quota reservation alive for as long as
    /// admission may still read them, even if the owning job is cancelled and
    /// dropped mid-preparation.
    _hold: Arc<ResolvedReferenceHold>,
}

impl std::fmt::Debug for ResolvedReferenceAdmissionView {
    /// Never render staged paths; the set's own `Debug` has the same rule.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedReferenceAdmissionView")
            .field("entries", &self.entries.len())
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl ResolvedReferenceAdmissionView {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Mint verified bindings for admission. Identical verification to the
    /// runtime's path — same order, metadata, and fingerprint checks, and
    /// fresh descriptors opened without following symlinks.
    pub fn inference_bindings(
        &self,
        request: &mold_core::GenerateRequest,
        cancellation: Option<&mold_inference::InferenceCancellationToken>,
    ) -> anyhow::Result<Vec<mold_inference::GenerationReferenceBinding>> {
        mint_inference_bindings(&self.entries, &self.fingerprint, request, cancellation)
    }
}

/// The one binding verifier. Both the owning set and its admission view route
/// here so a staged file can never be bound under two different rule sets.
fn mint_inference_bindings(
    entries: &[ResolvedReference],
    fingerprint: &str,
    request: &mold_core::GenerateRequest,
    cancellation: Option<&mold_inference::InferenceCancellationToken>,
) -> anyhow::Result<Vec<mold_inference::GenerationReferenceBinding>> {
    {
        let references = request
            .references
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("resolved references lost their queued descriptors"))?;
        anyhow::ensure!(
            references.len() == entries.len(),
            "resolved reference count changed before inference"
        );
        let queued_metadata = references
            .iter()
            .enumerate()
            .map(|(index, reference)| reference.redacted_metadata_lossless(index))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            mold_core::generation_reference_fingerprint(&queued_metadata) == fingerprint,
            "resolved reference fingerprint changed before inference"
        );
        entries
            .iter()
            .zip(queued_metadata)
            .map(|(entry, queued)| {
                anyhow::ensure!(
                    entry.metadata == queued,
                    "resolved reference {} changed before inference",
                    entry.metadata.index
                );
                let file = crate::batch_transaction::open_regular_file_no_follow(&entry.path)
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "failed to safely open resolved reference {}",
                            entry.metadata.index
                        )
                    })?;
                mold_inference::GenerationReferenceBinding::from_opened_file(
                    entry.metadata.clone(),
                    file,
                    MAX_REFERENCE_UPLOAD_FILE_BYTES,
                    cancellation,
                )
                .map_err(|error| {
                    if mold_inference::is_inference_cancelled(&error) {
                        error
                    } else {
                        anyhow::anyhow!(
                            "failed to verify resolved reference {}",
                            entry.metadata.index
                        )
                    }
                })
            })
            .collect()
    }
}

pub(crate) fn inference_bindings_for_request(
    request: &mold_core::GenerateRequest,
    resolved: Option<&ResolvedReferenceSet>,
    cancellation: Option<&mold_inference::InferenceCancellationToken>,
) -> anyhow::Result<Vec<mold_inference::GenerationReferenceBinding>> {
    match (request.references.is_some(), resolved) {
        (false, None) => Ok(Vec::new()),
        (true, Some(resolved)) => resolved.inference_bindings(request, cancellation),
        (true, None) => anyhow::bail!(
            "reference-bearing generation reached inference without private resolved media"
        ),
        (false, Some(_)) => anyhow::bail!(
            "private resolved media reached inference without queued reference descriptors"
        ),
    }
}

/// Name of the advisory lock a live staging root holds for its whole lifetime.
///
/// The OS releases it when the process dies, however it dies, which is what
/// makes "dead" decidable without a heuristic.
pub(crate) const STAGING_LOCK_FILE: &str = ".lock";

/// Serializes claiming a new staging root against sweeping for dead ones.
///
/// Creating a root is not atomic — the directory exists before its lock is
/// held — so without this a sweeper running in that window would see a lock
/// file nobody holds and delete a root that is about to become live. Both
/// sides take this lock, so the window cannot be observed.
const STAGING_SWEEP_LOCK_FILE: &str = ".sweep.lock";

/// Take the claim/sweep mutex, creating it if needed.
fn lock_staging_parent(parent: &std::path::Path) -> Option<std::fs::File> {
    if std::fs::create_dir_all(parent).is_err() {
        return None;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(parent.join(STAGING_SWEEP_LOCK_FILE))
        .ok()?;
    fs2::FileExt::lock_exclusive(&file).ok()?;
    Some(file)
}

/// What one boot sweep found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StagingSweep {
    /// Roots proven dead and deleted.
    pub removed: usize,
    /// Roots another running process still holds.
    pub live: usize,
    /// Roots with no lock file, whose liveness cannot be established.
    pub untracked: usize,
}

/// Claim this process's staging root by taking its advisory lock.
///
/// Held open for the store's lifetime. Returns `None` when the lock cannot be
/// taken — a filesystem without working advisory locks, say — which costs only
/// reclaimability: an unlocked root is never deleted by a peer.
fn claim_staging_root(root: &std::path::Path) -> Option<std::fs::File> {
    if let Err(error) = std::fs::create_dir_all(root) {
        tracing::warn!(root = %root.display(), %error, "could not create staging root");
        return None;
    }
    // Held across create-and-lock so a concurrent sweep cannot see the lock
    // file before we hold it and reclaim a root that is coming alive.
    let _sweep_guard = root.parent().and_then(lock_staging_parent);
    let lock_path = root.join(STAGING_LOCK_FILE);
    let file = match std::fs::File::create(&lock_path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(root = %root.display(), %error, "could not create staging lock");
            return None;
        }
    };
    match fs2::FileExt::try_lock_exclusive(&file) {
        Ok(()) => Some(file),
        Err(error) => {
            // Leave no unlocked lock file behind: a sweeper would read it as a
            // tracked-but-dead root and delete media this server is using.
            // Without the file the root is untracked, which is never removed.
            drop(file);
            let _ = std::fs::remove_file(&lock_path);
            tracing::warn!(
                root = %root.display(),
                %error,
                "could not lock this server's staging root; peers will leave it alone"
            );
            None
        }
    }
}

/// Delete `runtime-*` staging roots whose owning process is provably gone.
///
/// [`StoreLifetime`] removes this process's root on drop, but a SIGKILL runs
/// no destructor — and a durable queue exists precisely because SIGKILLs
/// happen. Without this sweep every hard stop leaks a directory of reference
/// media forever.
///
/// **Being a sibling is not evidence of being dead.** Two servers can share
/// one `MOLD_HOME` on different ports, so excluding only our own root would
/// let the second one to boot delete the first one's media out from under an
/// in-flight upload. A root is removed only when its advisory lock can be
/// acquired, which proves no process holds it; the OS releases that lock
/// however the owner died. A root with no lock file predates lock tracking and
/// is deliberately left alone and reported — guessing there is how a
/// long-running older server loses its media.
pub fn sweep_orphaned_staging_roots(current_root: &std::path::Path) -> StagingSweep {
    let mut sweep = StagingSweep::default();
    let Some(parent) = current_root.parent() else {
        return sweep;
    };
    // Mutually exclusive with `claim_staging_root`, so a root cannot be
    // created-but-not-yet-locked while this pass is looking at it.
    let Some(_sweep_guard) = lock_staging_parent(parent) else {
        tracing::warn!(
            dir = %parent.display(),
            "skipping the staging sweep: its claim mutex could not be taken"
        );
        return sweep;
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return sweep;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == current_root || !path.is_dir() {
            continue;
        }
        if !entry.file_name().to_string_lossy().starts_with("runtime-") {
            continue;
        }
        let lock_path = path.join(STAGING_LOCK_FILE);
        let Ok(lock) = std::fs::OpenOptions::new().write(true).open(&lock_path) else {
            sweep.untracked += 1;
            continue;
        };
        if fs2::FileExt::try_lock_exclusive(&lock).is_err() {
            sweep.live += 1;
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                tracing::info!(
                    root = %path.display(),
                    "removed reference-upload staging left by a stopped process"
                );
                sweep.removed += 1;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                root = %path.display(),
                %error,
                "could not remove orphaned reference-upload staging"
            ),
        }
        let _ = fs2::FileExt::unlock(&lock);
    }
    sweep
}

impl ReferenceUploadStore {
    /// This process's staging root. Exposed so startup can sweep its siblings.
    pub fn root(&self) -> &std::path::Path {
        self.root.as_path()
    }

    pub fn from_mold_home() -> Self {
        let mold_home = mold_core::Config::mold_dir().unwrap_or_else(|| PathBuf::from(".mold"));
        let mold_home = if mold_home.is_absolute() {
            mold_home
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(mold_home)
        };
        let root = mold_home
            .join("cache")
            .join("reference-uploads")
            .join(format!("runtime-{}", uuid::Uuid::new_v4()));
        let mut handle_salt = [0_u8; 32];
        getrandom::fill(&mut handle_salt).expect("OS randomness is required for upload handles");
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(StoreInner::default())),
            writers: Arc::new(tokio::sync::Semaphore::new(
                MAX_CONCURRENT_REFERENCE_UPLOADS,
            )),
            root: Arc::new(root.clone()),
            _lifetime: Arc::new(StoreLifetime {
                claim: std::sync::OnceLock::new(),
                root,
            }),
            resolved_bytes: Arc::new(AtomicU64::new(0)),
            handle_salt: Arc::new(handle_salt),
        }
    }

    #[cfg(test)]
    fn at(root: PathBuf) -> Self {
        let mut store = Self::from_mold_home();
        store.root = Arc::new(root.clone());
        store._lifetime = Arc::new(StoreLifetime {
            claim: std::sync::OnceLock::new(),
            root,
        });
        store
    }

    #[cfg(test)]
    pub(crate) fn staging_exists(&self) -> bool {
        self.root.exists()
    }

    fn digest_handle(&self, domain: &[u8], handle: &str) -> String {
        let mut digest = Sha256::new();
        digest.update(domain);
        digest.update(self.handle_salt.as_ref());
        digest.update(handle.as_bytes());
        format!("{:x}", digest.finalize())
    }

    fn session_digest(&self, handle: &str) -> String {
        self.digest_handle(b"mold.reference-upload.session.v1\0", handle)
    }

    fn upload_digest(&self, handle: &str) -> String {
        self.digest_handle(b"mold.reference-upload.slot.v1\0", handle)
    }

    pub(crate) fn scope_sha256(
        &self,
        request: &mold_core::GenerateRequest,
    ) -> Result<String, ApiError> {
        request_scope_sha256(request)
    }

    async fn ensure_roots(&self) -> Result<(), ApiError> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || create_private_directories(&root))
            .await
            .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
            .map_err(|error| {
                ApiError::internal(format!(
                    "failed to prepare private reference staging: {error:#}"
                ))
            })?;
        // The root now exists, so claim it before anything is staged inside.
        // A peer sweeping in this window leaves it alone anyway — an unlocked
        // root is never removed — so the ordering is safe as well as correct.
        self._lifetime.ensure_claimed();
        Ok(())
    }

    async fn purge_expired(&self) {
        let now = unix_time_ms();
        let dirs = {
            let mut inner = self.inner.lock().await;
            let expired = inner
                .sessions
                .iter()
                .filter_map(|(digest, session)| {
                    (session.expires_at_ms <= now && !session.consuming).then_some(digest.clone())
                })
                .collect::<Vec<_>>();
            expired
                .into_iter()
                .filter_map(|digest| remove_session_locked(&mut inner, &digest))
                .collect::<Vec<_>>()
        };
        for dir in dirs {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
    }

    pub async fn create_session(
        &self,
        identity: &str,
        instance_id: &str,
        mut payload: ReferenceUploadSessionRequest,
    ) -> Result<ReferenceUploadSessionResponse, ApiError> {
        // Session storage and every scope digest must bind the same explicit
        // partition identity that generation preparation will queue later.
        // Keep this defensive normalization inside the store as well as the
        // HTTP ingress so internal callers cannot mint alias-scoped sessions.
        minimax_h3::canonicalize_request_model(&mut payload.request);
        self.purge_expired().await;
        let references = payload.request.references.as_deref().ok_or_else(|| {
            ApiError::structured(
                "reference upload session requires ordered reference descriptors",
                "MINIMAX_H3_REFERENCE_REQUIRED",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("references".to_string()),
            )
        })?;
        minimax_h3::validate_reference_descriptors(references).map_err(ApiError::reference)?;
        let requested_count = payload.upload_references.len();
        let requested = payload
            .upload_references
            .into_iter()
            .collect::<HashSet<_>>();
        if requested.is_empty()
            || requested.len() != requested_count
            || requested.len() > references.len()
        {
            return Err(ApiError::structured(
                "upload_references must name at least one unique one-based reference",
                "MINIMAX_H3_REFERENCE_UPLOAD_SELECTION",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("upload_references".to_string()),
            ));
        }
        if requested
            .iter()
            .any(|reference| *reference == 0 || *reference as usize > references.len())
        {
            return Err(ApiError::structured(
                "upload_references contains an out-of-range one-based reference",
                "MINIMAX_H3_REFERENCE_UPLOAD_SELECTION",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("upload_references".to_string()),
            ));
        }

        self.ensure_roots().await?;
        let session_handle = random_handle("mrs_")?;
        let session_digest = self.session_digest(&session_handle);
        let session_dir = self.root.join(format!("session-{}", uuid::Uuid::new_v4()));
        let dir = session_dir.clone();
        tokio::task::spawn_blocking(move || create_private_directory(&dir))
            .await
            .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
            .map_err(|error| {
                ApiError::internal(format!(
                    "failed to create private reference session: {error:#}"
                ))
            })?;

        let expires_at_ms =
            unix_time_ms().saturating_add(REFERENCE_UPLOAD_SESSION_TTL.as_millis() as u64);
        let scope_sha256 = request_scope_sha256(&payload.request)?;
        let mut response_slots = Vec::with_capacity(requested.len());
        let mut slots = HashMap::new();
        for reference in requested.iter().copied() {
            let handle = random_handle("mru_")?;
            let digest = self.upload_digest(&handle);
            response_slots.push(ReferenceUploadSlot { reference, handle });
            slots.insert(
                digest,
                UploadSlot {
                    reference,
                    descriptor: references[reference as usize - 1].clone(),
                    path: session_dir.join(format!("reference-{reference}.media")),
                    state: UploadState::Empty,
                },
            );
        }
        response_slots.sort_by_key(|slot| slot.reference);

        let mut inner = self.inner.lock().await;
        let owned_sessions = inner
            .sessions
            .values()
            .filter(|session| session.identity == identity)
            .count();
        if owned_sessions >= MAX_REFERENCE_UPLOAD_SESSIONS_PER_IDENTITY {
            drop(inner);
            let _ = tokio::fs::remove_dir_all(session_dir).await;
            return Err(ApiError::with_code(
                "too many active reference upload sessions",
                "REFERENCE_UPLOAD_SESSION_LIMIT",
                StatusCode::TOO_MANY_REQUESTS,
            ));
        }
        for upload_digest in slots.keys() {
            inner
                .upload_sessions
                .insert(upload_digest.clone(), session_digest.clone());
        }
        inner.sessions.insert(
            session_digest,
            UploadSession {
                identity: identity.to_string(),
                scope_sha256: scope_sha256.clone(),
                request: payload.request,
                expires_at_ms,
                dir: session_dir,
                cancel: CancellationToken::new(),
                slots,
                reserved_bytes: 0,
                consuming: false,
            },
        );
        Ok(ReferenceUploadSessionResponse {
            instance_id: instance_id.to_string(),
            expires_at_ms,
            request_scope_sha256: scope_sha256,
            session_handle,
            uploads: response_slots,
        })
    }

    pub async fn upload(
        &self,
        identity: &str,
        instance_id: &str,
        handle: &str,
        content_type: &str,
        content_length: u64,
        body: Body,
    ) -> Result<ReferenceUploadCompleteResponse, ApiError> {
        self.purge_expired().await;
        if content_length == 0 || content_length > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(ApiError::with_code(
                format!(
                    "reference upload Content-Length must be in 1..={MAX_REFERENCE_UPLOAD_FILE_BYTES}"
                ),
                "REFERENCE_UPLOAD_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        let upload_digest = self.upload_digest(handle);
        let (session_digest, path, descriptor, reference, cancel) = {
            let mut inner = self.inner.lock().await;
            let session_digest = inner
                .upload_sessions
                .get(&upload_digest)
                .cloned()
                .ok_or_else(unknown_upload)?;
            let host_reserved = inner
                .reserved_bytes
                .saturating_add(self.resolved_bytes.load(Ordering::Acquire));
            let values = {
                let session = inner
                    .sessions
                    .get_mut(&session_digest)
                    .ok_or_else(unknown_upload)?;
                if session.identity != identity || session.consuming {
                    return Err(unknown_upload());
                }
                if session.reserved_bytes.saturating_add(content_length)
                    > MAX_REFERENCE_UPLOAD_SESSION_BYTES
                    || host_reserved.saturating_add(content_length)
                        > MAX_REFERENCE_UPLOAD_HOST_BYTES
                {
                    return Err(ApiError::with_code(
                        "reference upload quota exceeded",
                        "REFERENCE_UPLOAD_QUOTA",
                        StatusCode::PAYLOAD_TOO_LARGE,
                    ));
                }
                let slot = session
                    .slots
                    .get_mut(&upload_digest)
                    .ok_or_else(unknown_upload)?;
                if !matches!(slot.state, UploadState::Empty) {
                    return Err(ApiError::with_code(
                        "reference upload handle has already been used",
                        "REFERENCE_UPLOAD_ALREADY_USED",
                        StatusCode::CONFLICT,
                    ));
                }
                let declared_type = reference_mime_type(&slot.descriptor);
                if !mime_matches(declared_type, content_type) {
                    return Err(reference_error(
                        slot.reference,
                        "mime_type",
                        "MINIMAX_H3_REFERENCE_MEDIA_TYPE",
                        format!(
                            "reference {} Content-Type does not match its descriptor",
                            slot.reference
                        ),
                    ));
                }
                slot.state = UploadState::Uploading {
                    reserved_bytes: content_length,
                };
                session.reserved_bytes += content_length;
                (
                    slot.path.clone(),
                    slot.descriptor.clone(),
                    slot.reference,
                    session.cancel.clone(),
                )
            };
            inner.reserved_bytes += content_length;
            (session_digest, values.0, values.1, values.2, values.3)
        };

        let _permit = self.writers.clone().acquire_owned().await.map_err(|_| {
            ApiError::generation_unavailable("reference upload service is shutting down")
        })?;
        let result = stream_to_private_file(&path, content_length, body, &cancel).await;
        let digest = match result {
            Ok(digest) => digest,
            Err(error) => {
                self.rollback_upload(&session_digest, &upload_digest).await;
                let _ = tokio::fs::remove_file(&path).await;
                return Err(error);
            }
        };
        let probe_path = path.clone();
        let probe_descriptor = descriptor.clone();
        let probe = tokio::task::spawn_blocking(move || {
            probe_reference(
                &probe_path,
                &probe_descriptor,
                reference,
                &digest,
                ReferenceProbePolicy::CanonicalUpload,
            )
        })
        .await
        .map_err(|error| ApiError::internal(format!("reference probe task failed: {error}")));
        let metadata = match probe {
            Ok(Ok(metadata)) => metadata,
            Ok(Err(error)) | Err(error) => {
                self.rollback_upload(&session_digest, &upload_digest).await;
                let _ = tokio::fs::remove_file(&path).await;
                return Err(error);
            }
        };

        let mut inner = self.inner.lock().await;
        let finalized = (|| -> Result<ReferenceUploadCompleteResponse, ApiError> {
            let session = inner
                .sessions
                .get_mut(&session_digest)
                .ok_or_else(unknown_upload)?;
            let (slot_reference, reserved_bytes) = {
                let slot = session
                    .slots
                    .get(&upload_digest)
                    .ok_or_else(unknown_upload)?;
                let reserved_bytes = match slot.state {
                    UploadState::Uploading { reserved_bytes } => reserved_bytes,
                    _ => return Err(unknown_upload()),
                };
                (slot.reference, reserved_bytes)
            };

            let canonical_descriptor = reference_from_metadata(&metadata);
            let mut canonical_request = session.request.clone();
            let references = canonical_request.references.as_mut().ok_or_else(|| {
                ApiError::internal("reference upload session lost its bound descriptors")
            })?;
            let target = references
                .get_mut(slot_reference.saturating_sub(1) as usize)
                .ok_or_else(unknown_upload)?;
            *target = canonical_descriptor.clone();
            let session_complete = session.slots.iter().all(|(digest, slot)| {
                digest == &upload_digest || matches!(slot.state, UploadState::Complete { .. })
            });
            if session_complete {
                minimax_h3::validate_reference_descriptors(references)
                    .map_err(ApiError::reference)?;
            }
            let request_scope_sha256 = request_scope_sha256(&canonical_request)?;

            // The canonical descriptor, scope, and completed slot become
            // visible under one store lock. A generation can therefore never
            // consume canonical bytes under the earlier provisional scope.
            session.request = canonical_request;
            session.scope_sha256 = request_scope_sha256.clone();
            let slot = session
                .slots
                .get_mut(&upload_digest)
                .ok_or_else(unknown_upload)?;
            slot.descriptor = canonical_descriptor;
            slot.state = UploadState::Complete {
                size_bytes: reserved_bytes,
                metadata: Box::new(metadata.clone()),
            };
            Ok(ReferenceUploadCompleteResponse {
                instance_id: instance_id.to_string(),
                reference: slot_reference,
                metadata,
                request_scope_sha256,
                session_complete,
            })
        })();
        drop(inner);
        match finalized {
            Ok(response) => Ok(response),
            Err(error) => {
                self.rollback_upload(&session_digest, &upload_digest).await;
                let _ = tokio::fs::remove_file(&path).await;
                Err(error)
            }
        }
    }

    async fn rollback_upload(&self, session_digest: &str, upload_digest: &str) {
        let mut inner = self.inner.lock().await;
        let mut release = 0;
        if let Some(session) = inner.sessions.get_mut(session_digest) {
            if let Some(slot) = session.slots.get_mut(upload_digest) {
                if let UploadState::Uploading { reserved_bytes } = slot.state {
                    release = reserved_bytes;
                    slot.state = UploadState::Empty;
                    session.reserved_bytes = session.reserved_bytes.saturating_sub(release);
                }
            }
        }
        inner.reserved_bytes = inner.reserved_bytes.saturating_sub(release);
    }

    pub async fn cancel_session(&self, identity: &str, handle: &str) -> Result<(), ApiError> {
        self.purge_expired().await;
        let digest = self.session_digest(handle);
        let dir = {
            let mut inner = self.inner.lock().await;
            let session = inner.sessions.get(&digest).ok_or_else(unknown_session)?;
            if session.identity != identity || session.consuming {
                return Err(unknown_session());
            }
            remove_session_locked(&mut inner, &digest).ok_or_else(unknown_session)?
        };
        let _ = tokio::fs::remove_dir_all(dir).await;
        Ok(())
    }

    async fn upload_sources_for_request(
        &self,
        identity: &str,
        request: &mold_core::GenerateRequest,
        frozen_scope_sha256: Option<&str>,
    ) -> Result<(Option<String>, HashMap<String, UploadSource>), ApiError> {
        let Some(references) = request.references.as_deref() else {
            return Ok((None, HashMap::new()));
        };
        let handles = references
            .iter()
            .enumerate()
            .filter_map(|(index, reference)| match reference.media() {
                GenerationReferenceAuthority::Upload { handle } => Some((
                    u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1),
                    handle,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        if handles.is_empty() {
            return Ok((None, HashMap::new()));
        }
        let scope = match frozen_scope_sha256 {
            Some(scope) => scope.to_string(),
            None => request_scope_sha256(request)?,
        };
        let mut inner = self.inner.lock().await;
        let mut session_digest = None::<String>;
        let mut sources = HashMap::new();
        let mut seen = HashSet::new();
        for (reference, handle) in handles {
            let upload_digest = self.upload_digest(handle);
            if !seen.insert(upload_digest.clone()) {
                return Err(reference_error(
                    reference,
                    "media.handle",
                    "REFERENCE_UPLOAD_HANDLE_REUSED",
                    "a reference upload handle cannot be reused within one request",
                ));
            }
            let candidate = inner
                .upload_sessions
                .get(&upload_digest)
                .cloned()
                .ok_or_else(unknown_upload)?;
            if session_digest
                .as_ref()
                .is_some_and(|current| current != &candidate)
            {
                return Err(reference_error(
                    reference,
                    "media.handle",
                    "REFERENCE_UPLOAD_SESSION_MISMATCH",
                    "all upload references in one request must come from one bound session",
                ));
            }
            session_digest.get_or_insert(candidate.clone());
            let session = inner.sessions.get(&candidate).ok_or_else(unknown_upload)?;
            if session.identity != identity || session.scope_sha256 != scope || session.consuming {
                return Err(unknown_upload());
            }
            let slot = session
                .slots
                .get(&upload_digest)
                .ok_or_else(unknown_upload)?;
            let UploadState::Complete {
                size_bytes,
                metadata,
            } = &slot.state
            else {
                return Err(ApiError::structured(
                    format!("reference {} upload is not complete", slot.reference),
                    "REFERENCE_UPLOAD_INCOMPLETE",
                    StatusCode::CONFLICT,
                    Some(slot.reference),
                    Some("media.handle".to_string()),
                ));
            };
            debug_assert!(*size_bytes > 0);
            sources.insert(
                upload_digest,
                UploadSource {
                    path: slot.path.clone(),
                    metadata: metadata.as_ref().clone(),
                },
            );
        }
        let digest = session_digest.expect("upload handles yielded a session");
        let session = inner.sessions.get_mut(&digest).ok_or_else(unknown_upload)?;
        if sources.len() != session.slots.len() {
            return Err(ApiError::structured(
                "every upload slot in the bound session must be consumed together",
                "REFERENCE_UPLOAD_SESSION_INCOMPLETE",
                StatusCode::UNPROCESSABLE_ENTITY,
                None,
                Some("references".to_string()),
            ));
        }
        session.consuming = true;
        Ok((Some(digest), sources))
    }

    async fn reserve_resolved_bytes(
        &self,
        lease: &mut ResolvedQuotaLease,
        bytes: u64,
    ) -> Result<(), ApiError> {
        let inner = self.inner.lock().await;
        let active = inner
            .reserved_bytes
            .saturating_add(self.resolved_bytes.load(Ordering::Acquire));
        if active.saturating_add(bytes) > MAX_REFERENCE_UPLOAD_HOST_BYTES {
            return Err(ApiError::with_code(
                "reference staging quota exceeded",
                "REFERENCE_UPLOAD_QUOTA",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        if bytes > 0 {
            self.resolved_bytes.fetch_add(bytes, Ordering::AcqRel);
            lease.adopt_reserved(bytes);
        }
        Ok(())
    }

    async fn finish_consumed_session(
        &self,
        digest: Option<&str>,
        success: bool,
        quota: &mut ResolvedQuotaLease,
    ) {
        let Some(digest) = digest else { return };
        let dir = {
            let mut inner = self.inner.lock().await;
            if success {
                let transferred = inner
                    .sessions
                    .get_mut(digest)
                    .map(|session| {
                        let transferred = session.reserved_bytes;
                        session.reserved_bytes = 0;
                        transferred
                    })
                    .unwrap_or_default();
                inner.reserved_bytes = inner.reserved_bytes.saturating_sub(transferred);
                self.resolved_bytes.fetch_add(transferred, Ordering::AcqRel);
                quota.adopt_reserved(transferred);
                remove_session_locked(&mut inner, digest)
            } else {
                if let Some(session) = inner.sessions.get_mut(digest) {
                    session.consuming = false;
                }
                None
            }
        };
        if let Some(dir) = dir {
            let _ = tokio::fs::remove_dir_all(dir).await;
        }
    }

    pub async fn resolve_request(
        &self,
        identity: &str,
        request: &mut mold_core::GenerateRequest,
        media_roots: &[PathBuf],
        frozen_scope_sha256: Option<&str>,
    ) -> Result<Option<ResolvedReferenceSet>, ApiError> {
        self.purge_expired().await;
        let Some(references) = request.references.clone() else {
            return Ok(None);
        };
        minimax_h3::validate_references(&references).map_err(ApiError::reference)?;
        let (session_digest, upload_sources) = self
            .upload_sources_for_request(identity, request, frozen_scope_sha256)
            .await?;
        let mut quota = ResolvedQuotaLease::new(self.resolved_bytes.clone());
        self.ensure_roots().await?;
        let resolved_root = self.root.join(format!("resolved-{}", uuid::Uuid::new_v4()));
        let root = resolved_root.clone();
        if let Err(error) = tokio::task::spawn_blocking(move || create_private_directory(&root))
            .await
            .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
            .map_err(|error| {
                ApiError::internal(format!("failed to create resolved staging: {error:#}"))
            })
        {
            self.finish_consumed_session(session_digest.as_deref(), false, &mut quota)
                .await;
            return Err(error);
        }

        let result = resolve_all_references(
            &resolved_root,
            &references,
            media_roots,
            &upload_sources,
            self,
            &mut quota,
        )
        .await;
        let (entries, descriptors) = match result {
            Ok(value) => value,
            Err(error) => {
                let _ = tokio::fs::remove_dir_all(&resolved_root).await;
                self.finish_consumed_session(session_digest.as_deref(), false, &mut quota)
                    .await;
                return Err(error);
            }
        };
        self.finish_consumed_session(session_digest.as_deref(), true, &mut quota)
            .await;
        let metadata = entries
            .iter()
            .map(|entry| entry.metadata.clone())
            .collect::<Vec<_>>();
        let fingerprint = mold_core::generation_reference_fingerprint(&metadata);
        request.references = Some(descriptors);
        Ok(Some(ResolvedReferenceSet {
            entries,
            fingerprint,
            hold: Arc::new(ResolvedReferenceHold {
                root: resolved_root,
                _quota: quota,
                _store_lifetime: self._lifetime.clone(),
            }),
        }))
    }
}

async fn resolve_all_references(
    root: &Path,
    references: &[GenerationReference],
    media_roots: &[PathBuf],
    upload_sources: &HashMap<String, UploadSource>,
    store: &ReferenceUploadStore,
    quota: &mut ResolvedQuotaLease,
) -> Result<(Vec<ResolvedReference>, Vec<GenerationReference>), ApiError> {
    let mut entries = Vec::with_capacity(references.len());
    let mut descriptors = Vec::with_capacity(references.len());
    for (index, reference) in references.iter().enumerate() {
        let one_based = u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1);
        let target = root.join(format!("reference-{one_based}.media"));
        let metadata = match reference.media() {
            GenerationReferenceAuthority::Upload { handle } => {
                let digest = store.upload_digest(handle);
                let source = upload_sources.get(&digest).ok_or_else(unknown_upload)?;
                tokio::fs::hard_link(&source.path, &target)
                    .await
                    .map_err(|error| {
                        ApiError::internal(format!("failed to bind uploaded reference: {error}"))
                    })?;
                source.metadata.clone()
            }
            GenerationReferenceAuthority::Inline { data } => {
                let bytes = data.clone();
                store
                    .reserve_resolved_bytes(quota, bytes.len() as u64)
                    .await?;
                let target_clone = target.clone();
                let reference_clone = reference.clone();
                tokio::task::spawn_blocking(move || {
                    write_private_bytes(&target_clone, &bytes)?;
                    let digest = format!("{:x}", Sha256::digest(&bytes));
                    probe_reference(
                        &target_clone,
                        &reference_clone,
                        one_based,
                        &digest,
                        ReferenceProbePolicy::Strict,
                    )
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("reference staging task failed: {error}"))
                })??
            }
            GenerationReferenceAuthority::ServerPath { path } => {
                let source =
                    mold_core::resolve_server_media_path(path, media_roots).map_err(|error| {
                        reference_error(one_based, "media.path", "MINIMAX_H3_REFERENCE_PATH", error)
                    })?;
                let (source, source_bytes) = tokio::task::spawn_blocking(move || {
                    let source = crate::batch_transaction::open_regular_file_no_follow(&source)
                        .map_err(|error| {
                            ApiError::validation(format!(
                                "failed to safely open reference: {error:#}"
                            ))
                        })?;
                    let source_bytes = source
                        .metadata()
                        .map_err(|error| {
                            ApiError::validation(format!("failed to inspect reference: {error}"))
                        })?
                        .len();
                    if source_bytes > MAX_REFERENCE_UPLOAD_FILE_BYTES {
                        return Err(ApiError::with_code(
                            "server-local reference exceeds the per-file quota",
                            "REFERENCE_UPLOAD_TOO_LARGE",
                            StatusCode::PAYLOAD_TOO_LARGE,
                        ));
                    }
                    Ok((source, source_bytes))
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("reference staging task failed: {error}"))
                })??;
                store.reserve_resolved_bytes(quota, source_bytes).await?;
                let target_clone = target.clone();
                let reference_clone = reference.clone();
                tokio::task::spawn_blocking(move || {
                    let digest = copy_private_bounded(source, &target_clone)?;
                    probe_reference(
                        &target_clone,
                        &reference_clone,
                        one_based,
                        &digest,
                        ReferenceProbePolicy::Strict,
                    )
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("reference staging task failed: {error}"))
                })??
            }
            GenerationReferenceAuthority::Descriptor => {
                return Err(reference_error(
                    one_based,
                    "media.authority",
                    "MINIMAX_H3_REFERENCE_DESCRIPTOR_ONLY",
                    "descriptor authority cannot enter generation",
                ));
            }
        };
        descriptors.push(reference_from_metadata(&metadata));
        entries.push(ResolvedReference {
            metadata,
            path: target,
        });
    }
    Ok((entries, descriptors))
}

fn remove_session_locked(inner: &mut StoreInner, digest: &str) -> Option<PathBuf> {
    let session = inner.sessions.remove(digest)?;
    session.cancel.cancel();
    for upload_digest in session.slots.keys() {
        inner.upload_sessions.remove(upload_digest);
    }
    inner.reserved_bytes = inner.reserved_bytes.saturating_sub(session.reserved_bytes);
    Some(session.dir)
}

fn request_scope_sha256(request: &mold_core::GenerateRequest) -> Result<String, ApiError> {
    let mut scoped = request.clone();
    scoped.references = scoped.references.as_ref().map(|references| {
        references
            .iter()
            .map(reference_as_descriptor)
            .collect::<Vec<_>>()
    });
    let wire = serde_json::to_vec(&scoped)
        .map_err(|error| ApiError::internal(format!("failed to bind upload request: {error}")))?;
    let mut digest = Sha256::new();
    digest.update(b"mold.reference-upload.request-scope.v1\0");
    digest.update(wire);
    Ok(format!("{:x}", digest.finalize()))
}

fn reference_as_descriptor(reference: &GenerationReference) -> GenerationReference {
    match reference {
        GenerationReference::Image {
            provenance,
            mime_type,
            width,
            height,
            ..
        } => GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
        },
        GenerationReference::Video {
            provenance,
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => GenerationReference::Video {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            width: *width,
            height: *height,
            frame_count: *frame_count,
            duration_ms: *duration_ms,
            fps: *fps,
            has_audio: *has_audio,
            audio_duration_ms: *audio_duration_ms,
            audio_sample_count: *audio_sample_count,
            audio_sample_rate: *audio_sample_rate,
            audio_channels: *audio_channels,
        },
        GenerationReference::Audio {
            provenance,
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: provenance.clone(),
            mime_type: mime_type.clone(),
            duration_ms: *duration_ms,
            sample_rate: *sample_rate,
            channels: *channels,
            sample_count: *sample_count,
        },
    }
}

fn reference_from_metadata(metadata: &GenerationReferenceMetadata) -> GenerationReference {
    let provenance = GenerationReferenceProvenance {
        name: metadata.name.clone(),
        sha256: Some(metadata.sha256.clone()),
    };
    match metadata.kind {
        mold_core::GenerationReferenceKind::Image => GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: metadata.mime_type.clone(),
            width: metadata.width.unwrap_or_default(),
            height: metadata.height.unwrap_or_default(),
        },
        mold_core::GenerationReferenceKind::Video => GenerationReference::Video {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: metadata.mime_type.clone(),
            width: metadata.width.unwrap_or_default(),
            height: metadata.height.unwrap_or_default(),
            frame_count: metadata.frame_count,
            duration_ms: metadata.duration_ms.unwrap_or_default(),
            fps: metadata.fps.unwrap_or_default(),
            has_audio: metadata.has_audio,
            audio_duration_ms: metadata.audio_duration_ms,
            audio_sample_count: metadata.audio_sample_count,
            audio_sample_rate: metadata.audio_sample_rate,
            audio_channels: metadata.audio_channels,
        },
        mold_core::GenerationReferenceKind::Audio => GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance,
            mime_type: metadata.mime_type.clone(),
            duration_ms: metadata.duration_ms.unwrap_or_default(),
            sample_rate: metadata.sample_rate.unwrap_or_default(),
            channels: metadata.channels.unwrap_or_default(),
            sample_count: metadata.sample_count,
        },
    }
}

async fn stream_to_private_file(
    path: &Path,
    expected: u64,
    body: Body,
    cancel: &CancellationToken,
) -> Result<String, ApiError> {
    let target = path.to_path_buf();
    let file = tokio::task::spawn_blocking(move || {
        crate::batch_transaction::create_private_file_no_follow(&target)
    })
    .await
    .map_err(|error| ApiError::internal(format!("reference staging task failed: {error}")))?
    .map_err(|error| {
        ApiError::internal(format!("failed to create reference staging: {error:#}"))
    })?;
    let mut file = tokio::fs::File::from_std(file);
    let mut stream = body.into_data_stream();
    let mut written = 0_u64;
    let mut digest = Sha256::new();
    loop {
        let next = tokio::select! {
            _ = cancel.cancelled() => {
                return Err(ApiError::cancelled("reference upload session was cancelled"));
            }
            next = stream.next() => next,
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|error| {
            ApiError::validation(format!("reference upload body failed: {error}"))
        })?;
        written = written.saturating_add(chunk.len() as u64);
        if written > expected || written > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(ApiError::with_code(
                "reference upload exceeded its declared or maximum size",
                "REFERENCE_UPLOAD_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        digest.update(&chunk);
        file.write_all(&chunk)
            .await
            .map_err(|error| ApiError::internal(format!("failed to stage reference: {error}")))?;
    }
    if written != expected {
        return Err(ApiError::validation(format!(
            "reference upload ended at {written} bytes; expected {expected}"
        )));
    }
    file.flush()
        .await
        .map_err(|error| ApiError::internal(format!("failed to flush reference: {error}")))?;
    file.sync_all()
        .await
        .map_err(|error| ApiError::internal(format!("failed to fsync reference: {error}")))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn write_private_bytes(path: &Path, bytes: &[u8]) -> Result<(), ApiError> {
    let mut file =
        crate::batch_transaction::create_private_file_no_follow(path).map_err(|error| {
            ApiError::internal(format!("failed to create reference staging: {error:#}"))
        })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| ApiError::internal(format!("failed to stage reference: {error}")))
}

fn copy_private_bounded(mut source: std::fs::File, target: &Path) -> Result<String, ApiError> {
    let mut target =
        crate::batch_transaction::create_private_file_no_follow(target).map_err(|error| {
            ApiError::internal(format!("failed to create reference staging: {error:#}"))
        })?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    let mut digest = Sha256::new();
    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|error| ApiError::validation(format!("failed to read reference: {error}")))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(ApiError::with_code(
                "server-local reference exceeds the per-file quota",
                "REFERENCE_UPLOAD_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        digest.update(&buffer[..read]);
        target
            .write_all(&buffer[..read])
            .map_err(|error| ApiError::internal(format!("failed to stage reference: {error}")))?;
    }
    target
        .sync_all()
        .map_err(|error| ApiError::internal(format!("failed to fsync reference: {error}")))?;
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceProbePolicy {
    /// Direct inline and server-path authorities must already carry exact
    /// descriptors; probing only verifies that frozen authority.
    Strict,
    /// Authenticated upload descriptors are provisional UI facts. The staged
    /// bytes and content decoder replace them before the request scope can be
    /// consumed.
    CanonicalUpload,
}

fn probe_reference(
    path: &Path,
    expected: &GenerationReference,
    reference: u32,
    digest: &str,
    policy: ReferenceProbePolicy,
) -> Result<GenerationReferenceMetadata, ApiError> {
    let declared_digest = expected.provenance().sha256.as_deref();
    if declared_digest.is_some_and(|declared| !declared.eq_ignore_ascii_case(digest)) {
        return Err(reference_error(
            reference,
            "provenance.sha256",
            "MINIMAX_H3_REFERENCE_DIGEST_MISMATCH",
            format!("reference {reference}: declared sha256 does not match staged bytes"),
        ));
    }
    let provenance = GenerationReferenceProvenance {
        name: expected.provenance().name.clone(),
        sha256: Some(digest.to_string()),
    };
    let canonical = match expected {
        GenerationReference::Image {
            mime_type,
            width,
            height,
            ..
        } => {
            let file =
                crate::batch_transaction::open_regular_file_no_follow(path).map_err(|error| {
                    ApiError::validation(format!("failed to safely open reference: {error:#}"))
                })?;
            let reader = image::ImageReader::new(BufReader::new(file))
                .with_guessed_format()
                .map_err(|error| unsupported_media(reference, error))?;
            let format = reader
                .format()
                .ok_or_else(|| unsupported_media(reference, "unknown image format"))?;
            let observed_mime = image_mime(format)
                .ok_or_else(|| unsupported_media(reference, "unsupported image format"))?;
            let (observed_width, observed_height) = reader
                .into_dimensions()
                .map_err(|error| unsupported_media(reference, error))?;
            require_equal(reference, "mime_type", mime_type.as_str(), observed_mime)?;
            if policy == ReferenceProbePolicy::Strict {
                require_equal(reference, "width", *width, observed_width)?;
                require_equal(reference, "height", *height, observed_height)?;
            }
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: observed_mime.to_string(),
                width: observed_width,
                height: observed_height,
            }
        }
        GenerationReference::Video {
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => {
            require_equal(reference, "mime_type", mime_type.as_str(), "video/mp4")?;
            let file =
                crate::batch_transaction::open_regular_file_no_follow(path).map_err(|error| {
                    ApiError::validation(format!("failed to safely open reference: {error:#}"))
                })?;
            let probe = mold_inference::ltx2::media::probe_video_file(file)
                .map_err(|error| unsupported_media(reference, error))?;
            if policy == ReferenceProbePolicy::Strict {
                require_equal(reference, "width", *width, probe.width)?;
                require_equal(reference, "height", *height, probe.height)?;
            }
            let observed_frame_count = probe
                .frames
                .filter(|frames| *frames > 0)
                .ok_or_else(|| unsupported_media(reference, "video frame count is unavailable"))?;
            if policy == ReferenceProbePolicy::Strict {
                if let Some(declared) = *frame_count {
                    require_equal(reference, "frame_count", declared, observed_frame_count)?;
                }
            }
            if policy == ReferenceProbePolicy::Strict && (*fps - f64::from(probe.fps)).abs() > 0.51
            {
                return Err(drift_error(reference, "fps", *fps, probe.fps));
            }
            let observed_duration = probe
                .duration_ms
                .ok_or_else(|| unsupported_media(reference, "video duration is unavailable"))?;
            let tolerance = (1_000_u64 / u64::from(probe.fps.max(1))).max(50);
            if policy == ReferenceProbePolicy::Strict
                && duration_ms.abs_diff(observed_duration) > tolerance
            {
                return Err(drift_error(
                    reference,
                    "duration_ms",
                    *duration_ms,
                    observed_duration,
                ));
            }
            if policy == ReferenceProbePolicy::Strict {
                require_equal(reference, "has_audio", *has_audio, probe.has_audio)?;
            }
            let observed_audio = if probe.has_audio {
                Some(
                    mold_inference::ltx2::media::probe_decoded_mp4_audio_file(path)
                        .map_err(|error| unsupported_media(reference, error))?
                        .ok_or_else(|| {
                            unsupported_media(
                                reference,
                                "MP4 declares audio but decoded no soundtrack samples",
                            )
                        })?,
                )
            } else {
                None
            };
            let observed_audio_duration = observed_audio.map(|audio| {
                audio
                    .samples_per_channel
                    .saturating_mul(1_000)
                    .div_ceil(u64::from(audio.sample_rate))
            });
            if policy == ReferenceProbePolicy::Strict {
                if let (Some(declared), Some(observed)) =
                    (*audio_duration_ms, observed_audio_duration)
                {
                    if declared.abs_diff(observed) > tolerance {
                        return Err(drift_error(
                            reference,
                            "audio_duration_ms",
                            declared,
                            observed,
                        ));
                    }
                }
            }
            if policy == ReferenceProbePolicy::Strict {
                if let Some(audio) = observed_audio {
                    if let Some(declared) = *audio_sample_count {
                        require_equal(
                            reference,
                            "audio_sample_count",
                            declared,
                            audio.samples_per_channel,
                        )?;
                    }
                    if let Some(declared) = *audio_sample_rate {
                        require_equal(reference, "audio_sample_rate", declared, audio.sample_rate)?;
                    }
                    if let Some(declared) = *audio_channels {
                        require_equal(reference, "audio_channels", declared, audio.channels)?;
                    }
                }
            }
            GenerationReference::Video {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: "video/mp4".to_string(),
                width: probe.width,
                height: probe.height,
                frame_count: Some(observed_frame_count),
                duration_ms: observed_duration,
                fps: f64::from(probe.fps),
                has_audio: probe.has_audio,
                audio_duration_ms: observed_audio_duration,
                audio_sample_count: observed_audio.map(|audio| audio.samples_per_channel),
                audio_sample_rate: observed_audio.map(|audio| audio.sample_rate),
                audio_channels: observed_audio.map(|audio| audio.channels),
            }
        }
        GenerationReference::Audio {
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => {
            if !matches!(
                mime_type.trim().to_ascii_lowercase().as_str(),
                "audio/wav" | "audio/x-wav" | "audio/wave"
            ) {
                return Err(unsupported_media(
                    reference,
                    "only RIFF/WAVE audio is accepted by reference ingress",
                ));
            }
            let file =
                crate::batch_transaction::open_regular_file_no_follow(path).map_err(|error| {
                    ApiError::validation(format!("failed to safely open reference: {error:#}"))
                })?;
            let wav = probe_wav(file).map_err(|error| unsupported_media(reference, error))?;
            if policy == ReferenceProbePolicy::Strict {
                require_equal(reference, "sample_rate", *sample_rate, wav.sample_rate)?;
                require_equal(reference, "channels", *channels, wav.channels)?;
                if let Some(declared) = *sample_count {
                    require_equal(reference, "sample_count", declared, wav.sample_count)?;
                }
                if duration_ms.abs_diff(wav.duration_ms) > 2 {
                    return Err(drift_error(
                        reference,
                        "duration_ms",
                        *duration_ms,
                        wav.duration_ms,
                    ));
                }
            }
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: "audio/wav".to_string(),
                duration_ms: wav.duration_ms,
                sample_rate: wav.sample_rate,
                channels: wav.channels,
                sample_count: Some(wav.sample_count),
            }
        }
    };
    minimax_h3::reference_prepared_shape(&canonical).map_err(ApiError::reference)?;
    canonical
        .redacted_metadata(reference.saturating_sub(1) as usize)
        .ok_or_else(|| ApiError::internal("validated reference metadata lost its digest"))
}

#[derive(Debug)]
struct WavProbe {
    duration_ms: u64,
    sample_rate: u32,
    channels: u16,
    sample_count: u64,
}

fn probe_wav(file: std::fs::File) -> anyhow::Result<WavProbe> {
    let file_len = file.metadata()?.len();
    anyhow::ensure!(file_len <= MAX_REFERENCE_UPLOAD_FILE_BYTES);
    let mut reader = BufReader::new(file);
    let mut header = [0_u8; 12];
    reader.read_exact(&mut header)?;
    anyhow::ensure!(
        &header[..4] == b"RIFF" && &header[8..] == b"WAVE",
        "not a RIFF/WAVE file"
    );
    let riff_end = 8_u64
        .checked_add(u64::from(u32::from_le_bytes(header[4..8].try_into()?)))
        .ok_or_else(|| anyhow::anyhow!("WAVE container length overflowed"))?;
    anyhow::ensure!(
        riff_end >= 12 && riff_end <= file_len,
        "WAVE container is truncated"
    );
    let mut sample_rate = None;
    let mut channels = None;
    let mut byte_rate = None;
    let mut block_align = None;
    let mut data_bytes = None;
    while reader.stream_position()?.saturating_add(8) <= riff_end {
        let mut chunk = [0_u8; 8];
        reader.read_exact(&mut chunk)?;
        let length = u32::from_le_bytes(chunk[4..].try_into()?) as u64;
        let data_start = reader.stream_position()?;
        let padded_length = length
            .checked_add(length % 2)
            .ok_or_else(|| anyhow::anyhow!("WAVE chunk length overflowed"))?;
        let next_chunk = data_start
            .checked_add(padded_length)
            .ok_or_else(|| anyhow::anyhow!("WAVE chunk offset overflowed"))?;
        anyhow::ensure!(
            next_chunk <= riff_end && next_chunk <= file_len,
            "WAVE chunk is truncated"
        );
        match &chunk[..4] {
            b"fmt " => {
                anyhow::ensure!(length >= 16, "WAVE fmt chunk is truncated");
                let mut format = [0_u8; 16];
                reader.read_exact(&mut format)?;
                let encoding = u16::from_le_bytes(format[..2].try_into()?);
                anyhow::ensure!(
                    matches!(encoding, 1 | 3),
                    "unsupported WAVE sample encoding"
                );
                let observed_channels = u16::from_le_bytes(format[2..4].try_into()?);
                let observed_sample_rate = u32::from_le_bytes(format[4..8].try_into()?);
                let observed_byte_rate = u32::from_le_bytes(format[8..12].try_into()?);
                let observed_block_align = u16::from_le_bytes(format[12..14].try_into()?);
                let bits_per_sample = u16::from_le_bytes(format[14..16].try_into()?);
                anyhow::ensure!(
                    observed_channels > 0
                        && observed_sample_rate > 0
                        && observed_block_align > 0
                        && bits_per_sample > 0
                        && bits_per_sample % 8 == 0,
                    "WAVE format fields are invalid"
                );
                let expected_align = observed_channels
                    .checked_mul(bits_per_sample / 8)
                    .ok_or_else(|| anyhow::anyhow!("WAVE block alignment overflowed"))?;
                anyhow::ensure!(
                    observed_block_align == expected_align,
                    "WAVE block alignment is inconsistent"
                );
                anyhow::ensure!(
                    observed_byte_rate
                        == observed_sample_rate.saturating_mul(u32::from(observed_block_align)),
                    "WAVE byte rate is inconsistent"
                );
                channels = Some(observed_channels);
                sample_rate = Some(observed_sample_rate);
                byte_rate = Some(observed_byte_rate);
                block_align = Some(observed_block_align);
            }
            b"data" => {
                data_bytes = Some(length);
            }
            _ => {}
        }
        reader.seek(SeekFrom::Start(next_chunk))?;
        if sample_rate.is_some() && data_bytes.is_some() {
            break;
        }
    }
    let sample_rate = sample_rate
        .filter(|rate| *rate > 0)
        .ok_or_else(|| anyhow::anyhow!("WAVE sample rate is missing"))?;
    let channels = channels
        .filter(|channels| *channels > 0)
        .ok_or_else(|| anyhow::anyhow!("WAVE channel count is missing"))?;
    let byte_rate = u64::from(
        byte_rate
            .filter(|rate| *rate > 0)
            .ok_or_else(|| anyhow::anyhow!("WAVE byte rate is missing"))?,
    );
    let block_align =
        u64::from(block_align.ok_or_else(|| anyhow::anyhow!("WAVE block alignment is missing"))?);
    let data_bytes = data_bytes
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow::anyhow!("WAVE data chunk is missing or empty"))?;
    anyhow::ensure!(
        data_bytes % block_align == 0,
        "WAVE data is not frame-aligned"
    );
    let duration_ms = data_bytes.saturating_mul(1_000).div_ceil(byte_rate);
    Ok(WavProbe {
        duration_ms,
        sample_rate,
        channels,
        sample_count: data_bytes / block_align,
    })
}

fn image_mime(format: image::ImageFormat) -> Option<&'static str> {
    match format {
        image::ImageFormat::Png => Some("image/png"),
        image::ImageFormat::Jpeg => Some("image/jpeg"),
        image::ImageFormat::WebP => Some("image/webp"),
        image::ImageFormat::Gif => Some("image/gif"),
        _ => None,
    }
}

fn require_equal<T: PartialEq + std::fmt::Display>(
    reference: u32,
    field: &str,
    declared: T,
    observed: T,
) -> Result<(), ApiError> {
    if declared == observed {
        Ok(())
    } else {
        Err(drift_error(reference, field, declared, observed))
    }
}

fn drift_error(
    reference: u32,
    field: &str,
    declared: impl std::fmt::Display,
    observed: impl std::fmt::Display,
) -> ApiError {
    reference_error(
        reference,
        field,
        "MINIMAX_H3_REFERENCE_DESCRIPTOR_DRIFT",
        format!(
            "reference {reference}: declared {field}={declared} does not match probed {observed}"
        ),
    )
}

fn unsupported_media(reference: u32, error: impl std::fmt::Display) -> ApiError {
    reference_error(
        reference,
        "mime_type",
        "MINIMAX_H3_REFERENCE_MEDIA_UNSUPPORTED",
        format!("reference {reference}: media probe failed: {error}"),
    )
}

fn reference_error(
    reference: u32,
    field: &str,
    code: &str,
    message: impl Into<String>,
) -> ApiError {
    ApiError::structured(
        message,
        code,
        StatusCode::UNPROCESSABLE_ENTITY,
        Some(reference),
        Some(field.to_string()),
    )
}

fn unknown_upload() -> ApiError {
    ApiError::with_code(
        "unknown, expired, or unauthorized reference upload handle",
        "REFERENCE_UPLOAD_NOT_FOUND",
        StatusCode::NOT_FOUND,
    )
}

fn unknown_session() -> ApiError {
    ApiError::with_code(
        "unknown, expired, or unauthorized reference upload session",
        "REFERENCE_UPLOAD_SESSION_NOT_FOUND",
        StatusCode::NOT_FOUND,
    )
}

fn random_handle(prefix: &str) -> Result<String, ApiError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|error| ApiError::internal(format!("failed to mint upload handle: {error}")))?;
    Ok(format!("{prefix}{}", URL_SAFE_NO_PAD.encode(bytes)))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn reference_mime_type(reference: &GenerationReference) -> &str {
    match reference {
        GenerationReference::Image { mime_type, .. }
        | GenerationReference::Video { mime_type, .. }
        | GenerationReference::Audio { mime_type, .. } => mime_type,
    }
}

fn mime_matches(declared: &str, actual: &str) -> bool {
    let declared = declared.trim().to_ascii_lowercase();
    let actual = actual
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    declared == actual
        || matches!(
            (declared.as_str(), actual.as_str()),
            ("audio/wav", "audio/x-wav")
                | ("audio/x-wav", "audio/wav")
                | ("audio/wave", "audio/wav")
        )
}

fn create_private_directories(root: &Path) -> anyhow::Result<()> {
    crate::batch_transaction::create_private_directories_no_follow(root)
}

fn create_private_directory(path: &Path) -> anyhow::Result<()> {
    crate::batch_transaction::create_private_directory_no_follow(path)
}

fn authenticated_identity(
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
) -> Result<String, ApiError> {
    authenticated
        .map(|Extension(authenticated)| authenticated.identity.clone())
        .ok_or_else(|| {
            ApiError::with_code(
                "API key authentication is required for reference uploads",
                "UNAUTHORIZED",
                StatusCode::UNAUTHORIZED,
            )
        })
}

#[utoipa::path(
    post,
    path = "/api/generate/reference-upload-sessions",
    tag = "generation",
    request_body = ReferenceUploadSessionRequest,
    responses(
        (status = 200, description = "Request-bound one-use upload handles", body = ReferenceUploadSessionResponse),
        (status = 401, description = "Explicit API-key authentication is required"),
        (status = 422, description = "Invalid ordered reference descriptor"),
        (status = 451, description = "MiniMax H3 model use requires explicit authorization"),
    )
)]
pub(crate) async fn create_reference_upload_session(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    Json(mut payload): Json<ReferenceUploadSessionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    minimax_h3::canonicalize_request_model(&mut payload.request);
    if payload.request.batch_size > 1 {
        return Err(ApiError::with_code(
            "MiniMax H3 references do not support durable server batches yet",
            "MINIMAX_H3_REFERENCE_BATCH_UNSUPPORTED",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    // Activation policy must run before directory/session allocation or download.
    crate::routes::require_server_generation_request_activation(&state, &payload.request, None)
        .await?;
    if minimax_h3::resolve_model_name(&payload.request.model).is_none() {
        return Err(ApiError::validation(
            "reference upload sessions are only valid for MiniMax H3",
        ));
    }
    let response = state
        .reference_uploads
        .create_session(&identity, &state.instance_id, payload)
        .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)))
}

#[utoipa::path(
    put,
    path = "/api/generate/reference-upload",
    tag = "generation",
    params(
        ("X-Mold-Reference-Upload" = String, Header, description = "One-use upload handle advertised by server capabilities"),
        ("Content-Length" = u64, Header, description = "Exact bounded media byte length"),
        ("Content-Type" = String, Header, description = "Declared media type, verified by content probe"),
    ),
    responses(
        (status = 200, description = "Content-sniffed canonical metadata and rebound request scope", body = ReferenceUploadCompleteResponse),
        (status = 401, description = "Explicit API-key authentication is required"),
        (status = 404, description = "Unknown, expired, or unauthorized upload handle"),
        (status = 413, description = "Upload exceeds a file, session, or host quota"),
        (status = 422, description = "Media differs from its bound descriptor"),
    )
)]
pub(crate) async fn upload_reference(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    headers: HeaderMap,
    body: Body,
) -> Result<impl IntoResponse, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    let handle = required_secret_header(&headers, UPLOAD_HANDLE_HEADER)?;
    let content_length = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| {
            ApiError::with_code(
                "reference uploads require an exact Content-Length",
                "REFERENCE_UPLOAD_LENGTH_REQUIRED",
                StatusCode::LENGTH_REQUIRED,
            )
        })?;
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            ApiError::with_code(
                "reference uploads require Content-Type",
                "REFERENCE_UPLOAD_CONTENT_TYPE_REQUIRED",
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
            )
        })?;
    let response = state
        .reference_uploads
        .upload(
            &identity,
            &state.instance_id,
            handle,
            content_type,
            content_length,
            body,
        )
        .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)))
}

#[utoipa::path(
    delete,
    path = "/api/generate/reference-upload-sessions",
    tag = "generation",
    params(
        ("X-Mold-Reference-Upload-Session" = String, Header, description = "Session handle advertised by server capabilities"),
    ),
    responses(
        (status = 204, description = "Session cancelled and staged bytes removed"),
        (status = 401, description = "Explicit API-key authentication is required"),
        (status = 404, description = "Unknown, expired, or unauthorized session"),
    )
)]
pub(crate) async fn cancel_reference_upload_session(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    headers: HeaderMap,
) -> Result<StatusCode, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    let handle = required_secret_header(&headers, SESSION_HANDLE_HEADER)?;
    state
        .reference_uploads
        .cancel_session(&identity, handle)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

fn required_secret_header<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str, ApiError> {
    let value = headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty() && value.len() <= minimax_h3::MAX_REFERENCE_UPLOAD_HANDLE_BYTES
        })
        .ok_or_else(unknown_upload)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::GenerationReferenceKind;

    /// Create a staging root that looks like a dead process left it: a lock
    /// file nobody holds.
    fn dead_root(cache: &std::path::Path, name: &str) -> PathBuf {
        let root = cache.join(name);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("reference.png"), b"leaked media").unwrap();
        std::fs::File::create(root.join(STAGING_LOCK_FILE)).unwrap();
        root
    }

    /// A SIGKILL leaves a `runtime-*` root behind because no destructor runs.
    /// Eleven of them is a real directory of user media that nothing else
    /// would ever remove.
    #[test]
    fn sweeping_removes_dead_runtime_roots_but_never_the_live_one() {
        let cache = tempfile::tempdir().unwrap();
        let live = cache.path().join("runtime-live");
        std::fs::create_dir_all(live.join("session")).unwrap();
        for index in 0..3 {
            dead_root(cache.path(), &format!("runtime-orphan-{index}"));
        }
        // Anything that is not a runtime root is somebody else's.
        let unrelated = cache.path().join("catalog-credentials");
        std::fs::create_dir_all(&unrelated).unwrap();

        assert_eq!(sweep_orphaned_staging_roots(&live).removed, 3);

        assert!(live.join("session").is_dir());
        assert!(unrelated.is_dir());
        assert!(!cache.path().join("runtime-orphan-0").exists());
        assert_eq!(
            sweep_orphaned_staging_roots(&live).removed,
            0,
            "sweeping is idempotent"
        );
    }

    /// Claiming must not be what creates the staging directory: an
    /// unauthenticated request has to leave no trace on disk, and taking the
    /// lock eagerly at construction broke exactly that. The claim is a
    /// property of a root that already exists, so it happens when authorized
    /// use first creates one.
    #[tokio::test]
    async fn a_store_claims_its_root_only_once_authorized_use_creates_it() {
        let cache = tempfile::tempdir().unwrap();
        let root = cache.path().join("runtime-lazy");
        let store = ReferenceUploadStore::at(root.clone());

        assert!(
            !store.staging_exists(),
            "constructing a store must not materialize staging"
        );
        assert!(!root.join(STAGING_LOCK_FILE).exists());

        store.ensure_roots().await.unwrap();

        assert!(store.staging_exists());
        assert!(
            root.join(STAGING_LOCK_FILE).is_file(),
            "the live root must be claimed once it exists"
        );
        // A peer must now see it as live rather than as an orphan.
        let peer = cache.path().join("runtime-peer");
        std::fs::create_dir_all(&peer).unwrap();
        let sweep = sweep_orphaned_staging_roots(&peer);
        assert_eq!(sweep.removed, 0);
        assert_eq!(sweep.live, 1);
        assert!(root.is_dir());
    }

    /// Two servers can share one MOLD_HOME on different ports. The second one
    /// starting must not delete the first one's media out from under an
    /// in-flight upload — its own root is not the only one that is live.
    #[test]
    fn sweeping_never_touches_a_root_another_process_still_holds() {
        let cache = tempfile::tempdir().unwrap();
        let mine = cache.path().join("runtime-mine");
        std::fs::create_dir_all(&mine).unwrap();

        let sibling = cache.path().join("runtime-sibling");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("reference.png"), b"live media").unwrap();
        // Exactly what a running peer holds for its whole lifetime.
        let held = std::fs::File::create(sibling.join(STAGING_LOCK_FILE)).unwrap();
        fs2::FileExt::try_lock_exclusive(&held).unwrap();

        let report = sweep_orphaned_staging_roots(&mine);

        assert_eq!(report.removed, 0);
        assert_eq!(report.live, 1);
        assert!(
            sibling.join("reference.png").is_file(),
            "a running server's staging must survive a peer's boot"
        );
        fs2::FileExt::unlock(&held).unwrap();
    }

    /// A claim that cannot take the lock must leave no lock file behind. A
    /// leftover unlocked one reads as tracked-and-dead to a peer sweep, which
    /// would then delete the media of a server that is very much alive; with
    /// no file the root is untracked, and untracked is never removed.
    #[test]
    fn a_failed_claim_leaves_no_lock_file_for_a_peer_to_acquire() {
        let cache = tempfile::tempdir().unwrap();
        let root = cache.path().join("runtime-contended");
        std::fs::create_dir_all(&root).unwrap();

        // Another handle in this process already owns the lock, so the claim's
        // `try_lock_exclusive` fails the way an unsupported filesystem would.
        let lock_path = root.join(STAGING_LOCK_FILE);
        let held = std::fs::File::create(&lock_path).unwrap();
        fs2::FileExt::try_lock_exclusive(&held).unwrap();
        let contended = std::fs::OpenOptions::new()
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert!(fs2::FileExt::try_lock_exclusive(&contended).is_err());
        drop(contended);
        fs2::FileExt::unlock(&held).unwrap();
        drop(held);
        std::fs::remove_file(&lock_path).unwrap();

        // Now the real shape: claiming a root whose lock cannot be taken.
        let blocker = std::fs::create_dir(&lock_path);
        assert!(
            blocker.is_ok(),
            "a directory at the lock path fails the claim"
        );
        assert!(claim_staging_root(&root).is_none());
        std::fs::remove_dir(&lock_path).unwrap();

        let peer = cache.path().join("runtime-peer");
        std::fs::create_dir_all(&peer).unwrap();
        let sweep = sweep_orphaned_staging_roots(&peer);
        assert_eq!(sweep.removed, 0, "a live root must not be reclaimed");
        assert_eq!(sweep.untracked, 1);
        assert!(root.is_dir());
    }

    /// A root with no lock file predates lock tracking, so its liveness cannot
    /// be established. Deleting it on a guess is how a long-running older
    /// server loses its media; it is reported instead so an operator can
    /// decide.
    #[test]
    fn sweeping_leaves_untracked_roots_alone_and_reports_them() {
        let cache = tempfile::tempdir().unwrap();
        let mine = cache.path().join("runtime-mine");
        std::fs::create_dir_all(&mine).unwrap();
        let legacy = cache.path().join("runtime-legacy");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("reference.png"), b"unknown provenance").unwrap();

        let report = sweep_orphaned_staging_roots(&mine);

        assert_eq!(report.removed, 0);
        assert_eq!(report.untracked, 1);
        assert!(legacy.join("reference.png").is_file());
    }

    fn png_reference(
        media: GenerationReferenceAuthority,
        sha256: Option<String>,
    ) -> GenerationReference {
        GenerationReference::Image {
            media,
            provenance: GenerationReferenceProvenance {
                name: Some("anchor.png".to_string()),
                sha256,
            },
            mime_type: "image/png".to_string(),
            width: 2,
            height: 2,
        }
    }

    fn png_bytes() -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    fn wav_bytes(sample_rate: u32, channels: u16, sample_count: u32) -> Vec<u8> {
        let block_align = channels * 2;
        let data_bytes = sample_count * u32::from(block_align);
        let mut bytes = Vec::with_capacity(44 + data_bytes as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.resize(44 + data_bytes as usize, 0);
        bytes
    }

    fn wav_reference(
        media: GenerationReferenceAuthority,
        sha256: String,
        duration_ms: u64,
        sample_count: u64,
    ) -> GenerationReference {
        GenerationReference::Audio {
            media,
            provenance: GenerationReferenceProvenance {
                name: Some("timing.wav".to_string()),
                sha256: Some(sha256),
            },
            mime_type: "audio/wav".to_string(),
            duration_ms,
            sample_rate: 24_000,
            channels: 1,
            sample_count: Some(sample_count),
        }
    }

    fn request_with_references(references: Vec<GenerationReference>) -> mold_core::GenerateRequest {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "reference print",
            "model": minimax_h3::REF2VA_COMFY,
            "width": minimax_h3::DEFAULT_WIDTH,
            "height": minimax_h3::DEFAULT_HEIGHT,
            "steps": minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "batch_size": 1,
            "strength": 1.0,
            "frames": minimax_h3::MIN_FRAMES,
            "fps": minimax_h3::FIXED_FPS,
            "output_format": "mp4"
        }))
        .unwrap();
        request.references = Some(references);
        request
    }

    fn request(reference: GenerationReference) -> mold_core::GenerateRequest {
        request_with_references(vec![reference])
    }

    fn with_media(
        mut reference: GenerationReference,
        authority: GenerationReferenceAuthority,
    ) -> GenerationReference {
        match &mut reference {
            GenerationReference::Image { media, .. }
            | GenerationReference::Video { media, .. }
            | GenerationReference::Audio { media, .. } => *media = authority,
        }
        reference
    }

    #[cfg(feature = "mp4")]
    const AAC_LC_SAMPLES_PER_PACKET: u64 = 1_024;

    #[cfg(feature = "mp4")]
    fn aac_video_bytes() -> Vec<u8> {
        let frames = (0..50)
            .map(|index| image::RgbImage::from_pixel(32, 32, image::Rgb([index as u8, 64, 192])))
            .collect::<Vec<_>>();
        let video = mold_inference::ltx_video::video_enc::encode_mp4(&frames, 24).unwrap();
        let samples_per_channel = 92_610_usize;
        let stereo = (0..samples_per_channel)
            .flat_map(|index| {
                let sample =
                    ((index as f32 / 44_100.0) * std::f32::consts::TAU * 440.0).sin() * 0.1;
                [sample, sample]
            })
            .collect::<Vec<_>>();
        mold_inference::av_media::attach_aac_track_to_mp4_bytes(&video, &stereo, 44_100, 2).unwrap()
    }

    #[cfg(feature = "mp4")]
    fn browser_aac_packet_sample_hint(bytes: &[u8]) -> u64 {
        use mp4_rs::{Mp4Reader, TrackType};
        use std::io::Cursor;

        let reader = Mp4Reader::read_header(Cursor::new(bytes), bytes.len() as u64).unwrap();
        let audio_tracks = reader
            .tracks()
            .values()
            .filter(|track| matches!(track.track_type(), Ok(TrackType::Audio)))
            .collect::<Vec<_>>();
        let [audio] = audio_tracks.as_slice() else {
            panic!("fixture must contain exactly one AAC track");
        };
        // `Track::sample_count` reads the MP4 stsz sample count used by the
        // browser probe. AAC-LC with frameLengthFlag=0 carries 1,024 decoded
        // samples per packet, so this is the exact provisional UI arithmetic.
        u64::from(audio.sample_count()) * AAC_LC_SAMPLES_PER_PACKET
    }

    #[cfg(feature = "mp4")]
    fn corrupt_one_aac_packet(mut bytes: Vec<u8>) -> Vec<u8> {
        use mp4_rs::{Mp4Reader, TrackType};
        use std::io::Cursor;

        let mut reader =
            Mp4Reader::read_header(Cursor::new(bytes.as_slice()), bytes.len() as u64).unwrap();
        let (track_id, packet_count) = reader
            .tracks()
            .iter()
            .find_map(|(track_id, track)| {
                matches!(track.track_type(), Ok(TrackType::Audio))
                    .then_some((*track_id, track.sample_count()))
            })
            .expect("fixture must contain an AAC track");
        let packet_id = (packet_count / 2).max(1);
        let packet = reader
            .read_sample(track_id, packet_id)
            .unwrap()
            .expect("fixture AAC packet must exist")
            .bytes
            .to_vec();
        drop(reader);

        let offsets = bytes
            .windows(packet.len())
            .enumerate()
            .filter_map(|(offset, candidate)| (candidate == packet).then_some(offset))
            .collect::<Vec<_>>();
        let [offset] = offsets.as_slice() else {
            panic!("fixture AAC packet payload must occur exactly once");
        };
        // Payload-only corruption keeps stsz and every other container fact
        // intact. The browser therefore retains the same packet hint while
        // Symphonia skips this undecodable AAC frame.
        bytes[*offset..*offset + packet.len()].fill(0);
        bytes
    }

    #[cfg(feature = "mp4")]
    fn provisional_video_reference(
        media: GenerationReferenceAuthority,
        sha256: String,
        audio_sample_count: u64,
    ) -> GenerationReference {
        GenerationReference::Video {
            media,
            provenance: GenerationReferenceProvenance {
                name: Some("motion.mp4".to_string()),
                sha256: Some(sha256),
            },
            mime_type: "video/mp4".to_string(),
            width: 32,
            height: 32,
            frame_count: Some(50),
            duration_ms: 2_083,
            fps: 24.0,
            has_audio: true,
            audio_duration_ms: Some(audio_sample_count.saturating_mul(1_000).div_ceil(44_100)),
            // Provisional stsz packet arithmetic. Only decoded PCM returned by
            // canonical upload probing may enter the final request scope.
            audio_sample_count: Some(audio_sample_count),
            audio_sample_rate: Some(44_100),
            audio_channels: Some(2),
        }
    }

    #[tokio::test]
    async fn upload_session_scope_and_storage_use_canonical_h3_partition() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let descriptor = png_reference(
            GenerationReferenceAuthority::Descriptor,
            Some("11".repeat(32)),
        );
        let mut alias_request = request(descriptor);
        alias_request.model = "MiniMax_H3_Ref2VA".into();
        let mut canonical_request = alias_request.clone();
        assert!(minimax_h3::canonicalize_request_model(
            &mut canonical_request
        ));

        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: alias_request,
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();

        assert_eq!(
            session.request_scope_sha256,
            request_scope_sha256(&canonical_request).unwrap()
        );
        let session_digest = store.session_digest(&session.session_handle);
        let inner = store.inner.lock().await;
        assert_eq!(
            inner.sessions[&session_digest].request.model,
            minimax_h3::REF2VA_COMFY
        );
    }

    /// The admission view is what lets Ref2VA admission reach the staged media
    /// before a frozen plan exists, so it must bind exactly like the runtime
    /// path while carrying none of the owning set's authority.
    #[tokio::test]
    async fn admission_view_binds_like_the_runtime_and_holds_no_owning_authority() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let bytes = png_bytes();
        let byte_count = bytes.len() as u64;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let descriptor = png_reference(GenerationReferenceAuthority::Descriptor, Some(digest));
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor.clone()),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let handle = session.uploads[0].handle.clone();
        let complete = store
            .upload(
                "auth-a",
                "instance-a",
                &handle,
                "image/png",
                byte_count,
                Body::from(bytes),
            )
            .await
            .unwrap();
        let mut final_request = request(png_reference(
            GenerationReferenceAuthority::Upload {
                handle: handle.clone(),
            },
            Some(complete.metadata.sha256.clone()),
        ));
        let resolved = store
            .resolve_request("auth-a", &mut final_request, &[], None)
            .await
            .unwrap()
            .unwrap();

        let view = resolved.admission_view();
        assert_eq!(view.len(), resolved.entries().len());
        assert!(!view.is_empty());
        assert_eq!(view.fingerprint(), resolved.fingerprint());

        // Same verifier, same result: the view must not become a second
        // contract that could admit media the runtime would reject.
        let runtime_bindings = resolved.inference_bindings(&final_request, None).unwrap();
        let admission_bindings = view.inference_bindings(&final_request, None).unwrap();
        assert_eq!(admission_bindings.len(), runtime_bindings.len());
        assert_eq!(
            admission_bindings[0].metadata(),
            runtime_bindings[0].metadata()
        );
        // Independent descriptors, so releasing admission's cannot disturb the
        // runtime's later binding.
        drop(admission_bindings);
        drop(runtime_bindings);
        assert_eq!(
            resolved
                .inference_bindings(&final_request, None)
                .unwrap()
                .len(),
            1
        );

        // It renders neither staged paths nor the upload handle.
        let rendered = format!("{view:?}");
        assert!(!rendered.contains(dir.path().to_string_lossy().as_ref()));
        assert!(!rendered.contains(&handle));

        // It carries no quota lease, so cloning and dropping it must not
        // release the owning set's reservation.
        let cloned = view.clone();
        drop(cloned);
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), byte_count);

        // A drifted request is refused here exactly as on the runtime path.
        let mut drifted = final_request.clone();
        drifted.references = None;
        assert!(view.inference_bindings(&drifted, None).is_err());

        // Cancellation is honoured identically too.
        let cancellation = mold_inference::InferenceCancellationToken::default();
        cancellation.cancel();
        let cancelled = view
            .inference_bindings(&final_request, Some(&cancellation))
            .unwrap_err();
        assert!(mold_inference::is_inference_cancelled(&cancelled));

        drop(view);
        drop(resolved);
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
    }

    /// Cancelling a queued job drops the resolved set while an admission
    /// decode may still be running against it. The staging and its quota must
    /// outlive that decode, and the quota must still be released exactly once.
    #[tokio::test]
    async fn dropping_the_owning_set_mid_admission_keeps_staging_and_releases_quota_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let bytes = png_bytes();
        let byte_count = bytes.len() as u64;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let descriptor = png_reference(GenerationReferenceAuthority::Descriptor, Some(digest));
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor.clone()),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let handle = session.uploads[0].handle.clone();
        let complete = store
            .upload(
                "auth-a",
                "instance-a",
                &handle,
                "image/png",
                byte_count,
                Body::from(bytes),
            )
            .await
            .unwrap();
        let mut final_request = request(png_reference(
            GenerationReferenceAuthority::Upload {
                handle: handle.clone(),
            },
            Some(complete.metadata.sha256.clone()),
        ));
        let resolved = store
            .resolve_request("auth-a", &mut final_request, &[], None)
            .await
            .unwrap()
            .unwrap();

        let staged = resolved.entries()[0].path.clone();
        assert!(staged.is_file());
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), byte_count);

        // Admission is in flight, holding only the view.
        let view = resolved.admission_view();

        // The job is cancelled and its set dropped out from under admission.
        drop(resolved);

        // Staging survives and the reservation is still charged, so the decode
        // is not reading unlinked files on released quota.
        assert!(
            staged.is_file(),
            "staging must outlive the owning set while an admission view holds it"
        );
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), byte_count);
        // And it can still bind, which is the whole point of holding it.
        assert_eq!(
            view.inference_bindings(&final_request, None).unwrap().len(),
            1
        );

        // When admission finishes, the last holder releases both — once.
        drop(view);
        assert!(!staged.exists());
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
    }

    /// A cancelled preparation must stop decoding rather than run to
    /// completion on media whose job is already gone.
    #[tokio::test]
    async fn a_cancelled_admission_stops_binding_cooperatively() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let bytes = png_bytes();
        let byte_count = bytes.len() as u64;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let descriptor = png_reference(GenerationReferenceAuthority::Descriptor, Some(digest));
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor.clone()),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let handle = session.uploads[0].handle.clone();
        let complete = store
            .upload(
                "auth-a",
                "instance-a",
                &handle,
                "image/png",
                byte_count,
                Body::from(bytes),
            )
            .await
            .unwrap();
        let mut final_request = request(png_reference(
            GenerationReferenceAuthority::Upload {
                handle: handle.clone(),
            },
            Some(complete.metadata.sha256.clone()),
        ));
        let resolved = store
            .resolve_request("auth-a", &mut final_request, &[], None)
            .await
            .unwrap()
            .unwrap();

        let view = resolved.admission_view();
        let cancellation = mold_inference::InferenceCancellationToken::default();
        cancellation.cancel();
        let error = view
            .inference_bindings(&final_request, Some(&cancellation))
            .unwrap_err();
        assert!(mold_inference::is_inference_cancelled(&error));
    }

    #[tokio::test]
    async fn upload_session_streams_and_resolves_to_private_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let bytes = png_bytes();
        let expected_bytes = bytes.clone();
        let byte_count = bytes.len() as u64;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let descriptor = png_reference(GenerationReferenceAuthority::Descriptor, Some(digest));
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor.clone()),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let handle = session.uploads[0].handle.clone();
        assert!(!format!("{session:?}").contains(&handle));
        let complete = store
            .upload(
                "auth-a",
                "instance-a",
                &handle,
                "image/png",
                byte_count,
                Body::from(bytes),
            )
            .await
            .unwrap();
        assert_eq!(complete.metadata.kind, GenerationReferenceKind::Image);
        assert!(complete.metadata.prepared_shape.is_some());

        let mut final_request = request(png_reference(
            GenerationReferenceAuthority::Upload {
                handle: handle.clone(),
            },
            Some(complete.metadata.sha256.clone()),
        ));
        let resolved = store
            .resolve_request("auth-a", &mut final_request, &[], None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolved.entries().len(), 1);
        assert!(resolved.entries()[0].path.is_file());
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), byte_count);
        assert!(matches!(
            final_request.references.as_ref().unwrap()[0].media(),
            GenerationReferenceAuthority::Descriptor
        ));
        let (task, prepared) =
            crate::h3_admission::H3PreparedRequestShape::from_resolved_prepared_request(
                &final_request,
                128,
                1_024,
            )
            .unwrap();
        assert_eq!(task, crate::h3_admission::H3FrozenTask::Ref2va);
        assert_eq!(prepared.reference_fingerprint, resolved.fingerprint());
        assert!(!format!("{resolved:?}").contains(dir.path().to_string_lossy().as_ref()));
        let bindings = resolved.inference_bindings(&final_request, None).unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].metadata(), &complete.metadata);
        let debug = format!("{bindings:?}");
        assert!(!debug.contains(dir.path().to_string_lossy().as_ref()));
        assert!(!debug.contains(&handle));
        let cancellation = mold_inference::InferenceCancellationToken::default();
        cancellation.cancel();
        let cancelled = resolved
            .inference_bindings(&final_request, Some(&cancellation))
            .unwrap_err();
        assert!(mold_inference::is_inference_cancelled(&cancelled));

        // Binding hashes and retains the exact no-follow handle. Replacing the
        // staged pathname after dispatch cannot alter the bytes decoded under
        // the frozen digest, and a later bind fails closed on the replacement.
        let staged = resolved.entries()[0].path.clone();
        let displaced = staged.with_extension("displaced");
        std::fs::rename(&staged, &displaced).unwrap();
        std::fs::write(&staged, b"different replacement bytes").unwrap();
        let mut bound = bindings[0].file();
        bound.seek(SeekFrom::Start(0)).unwrap();
        let mut observed = Vec::new();
        bound.read_to_end(&mut observed).unwrap();
        assert_eq!(observed, expected_bytes);
        let replacement_error = resolved
            .inference_bindings(&final_request, None)
            .unwrap_err();
        let replacement_message = format!("{replacement_error:#}");
        assert!(!replacement_message.contains(dir.path().to_string_lossy().as_ref()));
        assert!(!replacement_message.contains(&handle));

        // Even the lower-level safe-open error includes the private path.
        // A non-regular replacement must remain client-safe at this boundary.
        std::fs::remove_file(&staged).unwrap();
        std::fs::create_dir(&staged).unwrap();
        let open_error = resolved
            .inference_bindings(&final_request, None)
            .unwrap_err();
        let open_message = format!("{open_error:#}");
        assert!(!open_message.contains(dir.path().to_string_lossy().as_ref()));
        assert!(!open_message.contains(&handle));

        let mut changed = final_request.clone();
        if let GenerationReference::Image { provenance, .. } =
            &mut changed.references.as_mut().unwrap()[0]
        {
            provenance.name = Some("changed.png".to_string());
        }
        assert!(resolved.inference_bindings(&changed, None).is_err());
        drop(bindings);
        drop(resolved);
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn canonical_scope_waits_for_every_slot() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let image = png_bytes();
        let image_digest = format!("{:x}", Sha256::digest(&image));
        let valid_audio = wav_bytes(24_000, 1, 48_000);
        let audio_digest = format!("{:x}", Sha256::digest(&valid_audio));
        let provisional_audio = wav_reference(
            GenerationReferenceAuthority::Descriptor,
            audio_digest.clone(),
            2_001,
            48_001,
        );
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request_with_references(vec![
                        png_reference(GenerationReferenceAuthority::Descriptor, Some(image_digest)),
                        provisional_audio,
                    ]),
                    upload_references: vec![1, 2],
                },
            )
            .await
            .unwrap();
        let initial_scope = session.request_scope_sha256.clone();
        let first = store
            .upload(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "image/png",
                image.len() as u64,
                Body::from(image),
            )
            .await
            .unwrap();
        assert!(!first.session_complete);

        let final_complete = store
            .upload(
                "auth-a",
                "instance-a",
                &session.uploads[1].handle,
                "audio/wav",
                valid_audio.len() as u64,
                Body::from(valid_audio),
            )
            .await
            .unwrap();
        assert!(final_complete.session_complete);
        assert_eq!(final_complete.metadata.sample_count, Some(48_000));
        assert_eq!(final_complete.metadata.duration_ms, Some(2_000));
        assert_ne!(final_complete.request_scope_sha256, initial_scope);

        let mut canonical_image = reference_from_metadata(&first.metadata);
        canonical_image = with_media(
            canonical_image,
            GenerationReferenceAuthority::Upload {
                handle: session.uploads[0].handle.clone(),
            },
        );
        let mut canonical_audio = reference_from_metadata(&final_complete.metadata);
        canonical_audio = with_media(
            canonical_audio,
            GenerationReferenceAuthority::Upload {
                handle: session.uploads[1].handle.clone(),
            },
        );
        let final_request = request_with_references(vec![canonical_image, canonical_audio]);
        assert_eq!(
            request_scope_sha256(&final_request).unwrap(),
            final_complete.request_scope_sha256
        );
    }

    #[tokio::test]
    async fn final_canonical_validation_failure_rolls_back_upload_state_and_quota() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let image = png_bytes();
        let image_digest = format!("{:x}", Sha256::digest(&image));
        let short_audio = wav_bytes(24_000, 1, 24_000);
        let short_audio_digest = format!("{:x}", Sha256::digest(&short_audio));
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request_with_references(vec![
                        png_reference(GenerationReferenceAuthority::Descriptor, Some(image_digest)),
                        // The provisional browser descriptor is structurally
                        // valid, but canonical probing will reveal a one-second
                        // WAV that violates H3's 2–15 second audio contract.
                        wav_reference(
                            GenerationReferenceAuthority::Descriptor,
                            short_audio_digest,
                            2_001,
                            48_001,
                        ),
                    ]),
                    upload_references: vec![1, 2],
                },
            )
            .await
            .unwrap();
        let session_dir = {
            let inner = store.inner.lock().await;
            inner
                .sessions
                .get(&store.session_digest(&session.session_handle))
                .unwrap()
                .dir
                .clone()
        };
        store
            .upload(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "image/png",
                image.len() as u64,
                Body::from(image.clone()),
            )
            .await
            .unwrap();

        // Final validation failure releases only the failed upload's quota and
        // restores its slot to Empty. Retrying the identical, digest-bound
        // bytes reaches the same validation error rather than ALREADY_USED.
        for _ in 0..2 {
            let error = store
                .upload(
                    "auth-a",
                    "instance-a",
                    &session.uploads[1].handle,
                    "audio/wav",
                    short_audio.len() as u64,
                    Body::from(short_audio.clone()),
                )
                .await
                .unwrap_err();
            assert_eq!(error.reference, Some(2));
            assert_eq!(error.field.as_deref(), Some("duration_ms"));
            let inner = store.inner.lock().await;
            let stored = inner
                .sessions
                .get(&store.session_digest(&session.session_handle))
                .unwrap();
            assert_eq!(inner.reserved_bytes, image.len() as u64);
            assert_eq!(stored.reserved_bytes, image.len() as u64);
            assert!(matches!(
                stored
                    .slots
                    .get(&store.upload_digest(&session.uploads[1].handle))
                    .unwrap()
                    .state,
                UploadState::Empty
            ));
        }

        store
            .cancel_session("auth-a", &session.session_handle)
            .await
            .unwrap();
        assert_eq!(store.inner.lock().await.reserved_bytes, 0);
        assert!(!session_dir.exists());
    }

    #[cfg(feature = "mp4")]
    #[tokio::test]
    async fn decoded_aac_metadata_rebinds_upload_scope_while_direct_authorities_stay_strict() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let bytes = corrupt_one_aac_packet(aac_video_bytes());
        let browser_sample_hint = browser_aac_packet_sample_hint(&bytes);
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let provisional = provisional_video_reference(
            GenerationReferenceAuthority::Descriptor,
            digest.clone(),
            browser_sample_hint,
        );
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(provisional.clone()),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let initial_scope = session.request_scope_sha256.clone();
        let complete = store
            .upload(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "video/mp4",
                bytes.len() as u64,
                Body::from(bytes.clone()),
            )
            .await
            .unwrap();
        assert!(complete.session_complete);
        let decoded_sample_count = complete
            .metadata
            .audio_sample_count
            .expect("canonical host metadata must include decoded AAC samples");
        assert!(
            decoded_sample_count < browser_sample_hint,
            "the corrupted AAC packet must make decoded PCM ({decoded_sample_count}) shorter than the container hint ({browser_sample_hint})"
        );
        assert_eq!(
            decoded_sample_count,
            browser_sample_hint - AAC_LC_SAMPLES_PER_PACKET,
            "exactly the zeroed AAC packet must be absent from canonical decoded PCM"
        );
        assert_ne!(complete.request_scope_sha256, initial_scope);

        let mut final_reference = reference_from_metadata(&complete.metadata);
        final_reference = with_media(
            final_reference,
            GenerationReferenceAuthority::Upload {
                handle: session.uploads[0].handle.clone(),
            },
        );
        let mut final_request = request(final_reference);
        assert_eq!(
            request_scope_sha256(&final_request).unwrap(),
            complete.request_scope_sha256
        );

        // The earlier provisional scope cannot consume the now-canonical
        // session, even with its correct one-use handle.
        let mut stale = request(with_media(
            provisional.clone(),
            GenerationReferenceAuthority::Upload {
                handle: session.uploads[0].handle.clone(),
            },
        ));
        assert!(store
            .resolve_request("auth-a", &mut stale, &[], Some(&initial_scope))
            .await
            .is_err());

        // Direct inline and allowlisted server-path authorities do not get the
        // upload protocol's canonicalization privilege.
        let mut inline = request(with_media(
            provisional.clone(),
            GenerationReferenceAuthority::Inline {
                data: bytes.clone(),
            },
        ));
        assert!(store
            .resolve_request("auth-a", &mut inline, &[], None)
            .await
            .is_err());
        let media_root = dir.path().join("media");
        std::fs::create_dir(&media_root).unwrap();
        std::fs::write(media_root.join("motion.mp4"), &bytes).unwrap();
        let mut server_path = request(with_media(
            provisional,
            GenerationReferenceAuthority::ServerPath {
                path: "motion.mp4".to_string(),
            },
        ));
        assert!(store
            .resolve_request("auth-a", &mut server_path, &[media_root], None)
            .await
            .is_err());

        let resolved = store
            .resolve_request(
                "auth-a",
                &mut final_request,
                &[],
                Some(&complete.request_scope_sha256),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolved.entries()[0].metadata, complete.metadata);
    }

    #[test]
    fn inference_binding_presence_must_match_payload_free_request() {
        let plain: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "plain",
            "model": minimax_h3::FL2VA_COMFY,
            "width": 32,
            "height": 32,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1
        }))
        .unwrap();
        assert!(inference_bindings_for_request(&plain, None, None)
            .unwrap()
            .is_empty());

        let reference_request = request(png_reference(
            GenerationReferenceAuthority::Descriptor,
            Some("11".repeat(32)),
        ));
        assert!(inference_bindings_for_request(&reference_request, None, None).is_err());
    }

    #[tokio::test]
    async fn digest_drift_is_structured_and_partial_file_is_not_reusable() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let descriptor = png_reference(
            GenerationReferenceAuthority::Descriptor,
            Some("00".repeat(32)),
        );
        let session = store
            .create_session(
                "auth-a",
                "instance-a",
                ReferenceUploadSessionRequest {
                    request: request(descriptor),
                    upload_references: vec![1],
                },
            )
            .await
            .unwrap();
        let bytes = png_bytes();
        let error = store
            .upload(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "image/png",
                bytes.len() as u64,
                Body::from(bytes),
            )
            .await
            .unwrap_err();
        assert_eq!(error.reference, Some(1));
        assert_eq!(error.field.as_deref(), Some("provenance.sha256"));
    }

    #[test]
    fn request_scope_redacts_upload_handles() {
        let digest = "11".repeat(32);
        let descriptor = request(png_reference(
            GenerationReferenceAuthority::Descriptor,
            Some(digest.clone()),
        ));
        let upload = request(png_reference(
            GenerationReferenceAuthority::Upload {
                handle: "secret-handle".to_string(),
            },
            Some(digest),
        ));
        assert_eq!(
            request_scope_sha256(&descriptor).unwrap(),
            request_scope_sha256(&upload).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn private_staging_rejects_symlinked_parent_components() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let actual = dir.path().join("actual");
        std::fs::create_dir(&actual).unwrap();
        let link = dir.path().join("link");
        symlink(&actual, &link).unwrap();

        assert!(create_private_directories(&link.join("runtime")).is_err());
        assert!(!actual.join("runtime").exists());
    }

    #[test]
    fn wav_probe_rejects_declared_data_past_the_file_boundary() {
        let mut file = tempfile::tempfile().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&36_u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&32_000_u32.to_le_bytes());
        bytes.extend_from_slice(&128_000_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&4_096_u32.to_le_bytes());
        file.write_all(&bytes).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        assert!(probe_wav(file)
            .unwrap_err()
            .to_string()
            .contains("truncated"));
    }
}
