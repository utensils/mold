//! Crash-recoverable, logically atomic publication for server-owned batches.
//!
//! Child artifacts stay below the gallery's reserved transaction directory
//! until every child is prepared. Publication then holds the exclusive side
//! of [`GalleryPublicationGate`] while it links every file with no-replace
//! semantics, commits all metadata rows in one SQLite transaction, and
//! durably advances the manifest to `committed`.

use crate::batch_parent::{BatchChildLease, BatchChildReceipt};
use anyhow::{bail, ensure, Context};
use mold_db::GenerationRecord;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const TRANSACTION_DIR: &str = ".mold-batch-transactions";
pub(crate) const PARENT_AUTHORITY_DIR: &str = "parent-authority";
const MANIFEST_FILE: &str = "manifest.json";
const JOURNAL_FILE: &str = "journal.jsonl";
const COMMITTED_DIR: &str = "committed";
const DELETED_ARCHIVE_CHILDREN_DIR: &str = "deleted-archive-children";
const DELETED_ARCHIVE_CHILD_VERSION: u32 = 1;
const LEGACY_ATTEMPT_LOCKS_DIR: &str = ".attempt-locks";
const ATTEMPT_LOCK_PREFIX: &str = ".mold-batch-attempt-";
const PARENT_LOCK_PREFIX: &str = ".mold-batch-parent-";
const ATTEMPT_LOCK_SUFFIX: &str = ".lock";
const MANIFEST_VERSION_V1: u32 = 1;
const MANIFEST_VERSION: u32 = 2;
const DISK_SAFETY_FLOOR_BYTES: u64 = 64 * 1024 * 1024;

/// Readers cover gallery observation. Writers cover publication, deletion,
/// reconciliation, and corruption recovery.
#[derive(Clone, Default)]
pub struct GalleryPublicationGate {
    inner: Arc<tokio::sync::RwLock<()>>,
    committed_archive_index: Arc<std::sync::RwLock<Option<(u64, CommittedArchiveIndex)>>>,
}

impl GalleryPublicationGate {
    pub async fn read(&self) -> tokio::sync::OwnedRwLockReadGuard<()> {
        self.inner.clone().read_owned().await
    }

    pub async fn write(&self) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        self.inner.clone().write_owned().await
    }

    pub fn blocking_write(&self) -> tokio::sync::RwLockWriteGuard<'_, ()> {
        self.inner.blocking_write()
    }

    pub fn blocking_read(&self) -> tokio::sync::RwLockReadGuard<'_, ()> {
        self.inner.blocking_read()
    }

    pub(crate) fn install_committed_archive_index(
        &self,
        generation: u64,
        index: CommittedArchiveIndex,
    ) {
        *self
            .committed_archive_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((generation, index));
    }

    pub(crate) fn committed_archive_index(
        &self,
        output_dir: &Path,
    ) -> anyhow::Result<CommittedArchiveIndex> {
        let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
        self.committed_archive_index_while_locked(output_dir, &bookkeeping)
    }

    pub(crate) fn committed_archive_index_while_locked(
        &self,
        output_dir: &Path,
        bookkeeping: &GalleryBookkeepingGuard,
    ) -> anyhow::Result<CommittedArchiveIndex> {
        let generation = crate::gallery_authority::read_generation(output_dir, bookkeeping)?;
        if let (Some(generation), Some((cached_generation, index))) = (
            generation,
            self.committed_archive_index
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref(),
        ) {
            if generation == *cached_generation {
                return Ok(index.clone());
            }
        }
        let loaded = crate::gallery_authority::load_or_initialize(output_dir, bookkeeping, || {
            load_committed_archive_index_legacy(output_dir)
        })?;
        tracing::debug!(
            files_statted = loaded.stats.files_statted,
            files_hashed = loaded.stats.files_hashed,
            quarantined = loaded.stats.quarantined,
            reactivated = loaded.stats.reactivated,
            "refreshed gallery metadata authority"
        );
        self.install_committed_archive_index(loaded.generation, loaded.index.clone());
        Ok(loaded.index)
    }

    fn record_committed_manifest(
        &self,
        output_dir: &Path,
        manifest: &BatchAttemptManifest,
        bookkeeping: &GalleryBookkeepingGuard,
    ) -> anyhow::Result<()> {
        let mut index = self.committed_archive_index_while_locked(output_dir, bookkeeping)?;
        for child in &manifest.children {
            let identity = archived_child_identity(manifest, child)?;
            match index.entries.get(&identity.final_name) {
                Some(existing) if existing.identity == identity => continue,
                Some(_) => anyhow::bail!(
                    "multiple live committed archives claim gallery filename {}",
                    identity.final_name
                ),
                None => {}
            }
            index.retired_names.remove(&identity.final_name);
            index.retired_entries.remove(&identity.final_name);
            index.retirement_epochs.remove(&identity.final_name);
            index
                .retirement_projection_epochs
                .remove(&identity.final_name);
            index.quarantined_names.remove(&identity.final_name);
            index.entries.insert(
                identity.final_name.clone(),
                CommittedArchiveEntry {
                    identity,
                    record: child.record.clone(),
                    facts: Some(ArchiveFileFacts::from_path(
                        &output_dir.join(&child.final_name),
                    )?),
                },
            );
        }
        let prior_generation = crate::gallery_authority::read_generation(output_dir, bookkeeping)?
            .context("gallery authority generation is missing")?;
        let generation = crate::gallery_authority::commit_snapshot(
            output_dir,
            bookkeeping,
            prior_generation,
            &mut index,
            "publish_batch",
            manifest
                .children
                .iter()
                .map(|child| child.final_name.clone())
                .collect(),
        )?;
        self.install_committed_archive_index(generation, index);
        Ok(())
    }

    pub(crate) fn retire_committed_filename(&self, filename: &str) {
        let mut current = self
            .committed_archive_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, index)) = current.as_mut() {
            index.entries.remove(filename);
            index.quarantined_names.remove(filename);
            index.retired_names.insert(filename.to_owned());
        }
    }

    pub(crate) fn acknowledge_retirement_projections(
        &self,
        output_dir: &Path,
        filenames: impl IntoIterator<Item = String>,
    ) -> anyhow::Result<()> {
        let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
        let canonical_output_dir = bookkeeping.canonical_root();
        let mut index =
            self.committed_archive_index_while_locked(canonical_output_dir, &bookkeeping)?;
        let generation =
            crate::gallery_authority::read_generation(canonical_output_dir, &bookkeeping)?
                .context("gallery authority generation is missing")?;
        let projection_generation = generation
            .checked_add(1)
            .context("gallery authority generation overflow")?;
        let mut exact_names = Vec::new();
        for filename in filenames {
            if !index.retired_names.contains(&filename) {
                continue;
            }
            index
                .retirement_epochs
                .entry(filename.clone())
                .or_insert(generation);
            index
                .retirement_projection_epochs
                .insert(filename.clone(), projection_generation);
            exact_names.push(filename);
        }
        if exact_names.is_empty() {
            return Ok(());
        }
        let generation = crate::gallery_authority::commit_snapshot(
            canonical_output_dir,
            &bookkeeping,
            generation,
            &mut index,
            "retirement_projection_complete",
            exact_names,
        )?;
        self.install_committed_archive_index(generation, index);
        Ok(())
    }
}

/// Refuse a batch before inference when the gallery filesystem cannot hold
/// the expected staged children, a portable no-hard-link publication copy,
/// and a bounded safety margin. Filesystems that support hard links use less,
/// but admission stays correct on mounts that do not.
pub fn preflight_disk_space(output_dir: &Path, expected_staging_bytes: u64) -> anyhow::Result<()> {
    use sysinfo::Disks;

    fs::create_dir_all(output_dir)?;
    let canonical = fs::canonicalize(output_dir).unwrap_or_else(|_| output_dir.to_path_buf());
    let disks = Disks::new_with_refreshed_list();
    let available = disks
        .list()
        .iter()
        .filter(|disk| canonical.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| disk.available_space())
        .map(Ok)
        .unwrap_or_else(|| {
            fs2::available_space(&canonical).with_context(|| {
                format!(
                    "querying available space for gallery filesystem {}",
                    canonical.display()
                )
            })
        })?;
    validate_available_space(available, expected_staging_bytes)
}

fn validate_available_space(available: u64, expected_staging_bytes: u64) -> anyhow::Result<()> {
    let margin = DISK_SAFETY_FLOOR_BYTES.max(expected_staging_bytes / 20);
    let required = expected_staging_bytes
        .checked_mul(2)
        .context("batch disk preflight publication size overflow")?
        .checked_add(margin)
        .context("batch disk preflight size overflow")?;
    ensure!(
        available >= required,
        "insufficient gallery disk space for atomic batch: need {required} bytes, have {available}"
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchManifestState {
    Staging,
    Prepared,
    Committing,
    Committed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchManifestChild {
    pub child_index: usize,
    pub staging_name: String,
    pub final_name: String,
    pub checksum_sha256: Option<String>,
    pub size_bytes: Option<u64>,
    pub record: GenerationRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAttemptManifest {
    pub version: u32,
    pub parent_id: String,
    pub attempt_generation: u64,
    pub normalized_request: serde_json::Value,
    pub state: BatchManifestState,
    pub children: Vec<BatchManifestChild>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct ArchivedChildIdentity {
    pub(crate) parent_id: String,
    pub(crate) attempt_generation: u64,
    pub(crate) child_index: usize,
    pub(crate) final_name: String,
    pub(crate) checksum_sha256: String,
    pub(crate) size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeletedArchiveChild {
    version: u32,
    identity: ArchivedChildIdentity,
}

/// Stable file identity and timestamps captured after a published file and
/// its containing directory are fsynced. Matching facts let startup trust the
/// archived checksum without reading unchanged media bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ArchiveFileFacts {
    size_bytes: u64,
    modified_ns: Option<u128>,
    changed_ns: Option<i128>,
    device: Option<u64>,
    inode: Option<u64>,
}

impl ArchiveFileFacts {
    pub(crate) fn from_path(path: &Path) -> anyhow::Result<Self> {
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "gallery authority target is not a regular file: {}",
            path.display()
        );
        Self::from_metadata(&metadata)
    }

    pub(crate) fn from_metadata(metadata: &fs::Metadata) -> anyhow::Result<Self> {
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos());
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self {
                size_bytes: metadata.len(),
                modified_ns,
                changed_ns: Some(
                    (metadata.ctime() as i128) * 1_000_000_000 + metadata.ctime_nsec() as i128,
                ),
                device: Some(metadata.dev()),
                inode: Some(metadata.ino()),
            })
        }
        #[cfg(not(unix))]
        Ok(Self {
            size_bytes: metadata.len(),
            modified_ns,
            changed_ns: None,
            device: None,
            inode: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CommittedArchiveEntry {
    pub(crate) identity: ArchivedChildIdentity,
    pub(crate) record: GenerationRecord,
    #[serde(default)]
    pub(crate) facts: Option<ArchiveFileFacts>,
}

impl CommittedArchiveEntry {
    pub(crate) fn record(&self) -> &GenerationRecord {
        &self.record
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct CommittedArchiveIndex {
    pub(crate) entries: BTreeMap<String, CommittedArchiveEntry>,
    pub(crate) retired_names: BTreeSet<String>,
    #[serde(default)]
    pub(crate) quarantined_names: BTreeSet<String>,
    #[serde(default)]
    pub(crate) retired_entries: BTreeMap<String, CommittedArchiveEntry>,
    #[serde(default)]
    pub(crate) retirement_epochs: BTreeMap<String, u64>,
    #[serde(default)]
    pub(crate) retirement_projection_epochs: BTreeMap<String, u64>,
}

impl CommittedArchiveIndex {
    pub(crate) fn get(&self, filename: &str) -> Option<&CommittedArchiveEntry> {
        (!self.quarantined_names.contains(filename))
            .then(|| self.entries.get(filename))
            .flatten()
    }

    pub(crate) fn is_retired(&self, filename: &str) -> bool {
        self.retired_names.contains(filename)
            && !self.entries.contains_key(filename)
            && !self.quarantined_names.contains(filename)
    }

    pub(crate) fn is_quarantined(&self, filename: &str) -> bool {
        self.quarantined_names.contains(filename)
    }

    fn records(&self) -> impl Iterator<Item = &GenerationRecord> {
        self.entries.values().map(|entry| &entry.record)
    }

    pub(crate) fn overlay_db_gallery(
        &self,
        rows: &[GenerationRecord],
    ) -> Vec<mold_core::GalleryImage> {
        let mut seen = BTreeSet::new();
        let mut images = rows
            .iter()
            .filter_map(|row| {
                if self.is_retired(&row.filename) || self.is_quarantined(&row.filename) {
                    return None;
                }
                seen.insert(row.filename.clone());
                Some(
                    self.get(&row.filename)
                        .map(|entry| entry.record.to_gallery_image())
                        .unwrap_or_else(|| row.to_gallery_image()),
                )
            })
            .collect::<Vec<_>>();
        images.extend(
            self.entries
                .iter()
                .filter(|(filename, _)| {
                    !seen.contains(*filename) && !self.quarantined_names.contains(*filename)
                })
                .map(|(_, entry)| entry.record.to_gallery_image()),
        );
        images.sort_by_key(|image| std::cmp::Reverse(image.timestamp));
        images
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchJournalRecord {
    sequence: u64,
    attempt_generation: u64,
    event: BatchJournalEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BatchJournalEvent {
    ManifestSnapshot {
        manifest: BatchAttemptManifest,
    },
    ChildStaged {
        receipt: BatchChildReceipt,
    },
    ChildUnstaged {
        child_index: usize,
        lease_generation: u64,
        record_identity_sha256: String,
    },
    FinalPublished {
        child_index: usize,
    },
    MetadataCommitted,
    CleanupStarted,
}

#[derive(Clone, Debug)]
struct BatchJournalReplay {
    manifest: BatchAttemptManifest,
    manifest_history: Vec<BatchAttemptManifest>,
    staged_receipts: BTreeMap<usize, BatchChildReceipt>,
    published_children: BTreeSet<usize>,
    post_publish_snapshot: bool,
    metadata_committed: bool,
    cleanup_started: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReservationOwner {
    parent_id: String,
    attempt_generation: u64,
}

/// Proof that the caller owns the one cross-process gallery namespace lock.
///
/// Every production attempt-authority open, claim, sweep, and unlink requires
/// a reference to this guard. Keeping that requirement in the type signature
/// prevents a future path from reintroducing open-versus-cleanup races.
#[derive(Debug)]
pub(crate) struct GalleryBookkeepingGuard {
    lock: File,
    canonical_output_dir: PathBuf,
}

impl GalleryBookkeepingGuard {
    pub(crate) fn ensure_root(&self, output_dir: &Path) -> anyhow::Result<()> {
        let canonical = fs::canonicalize(output_dir)
            .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
        ensure!(
            canonical == self.canonical_output_dir,
            "gallery authority root changed or was addressed through the wrong guard: expected {}, got {}",
            self.canonical_output_dir.display(),
            canonical.display()
        );
        Ok(())
    }

    pub(crate) fn canonical_root(&self) -> &Path {
        &self.canonical_output_dir
    }
}

impl Drop for GalleryBookkeepingGuard {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.lock);
    }
}

pub(crate) struct GalleryNameReservation {
    output_dir: PathBuf,
    final_name: String,
    owner: ReservationOwner,
    // The stable filesystem lock is the liveness authority for an ordinary
    // reservation. Recovery may classify `ordinary:*` records as stale only
    // after it acquires this same lock, so retain it through final-file fsync,
    // directory fsync, reservation release, and cleanup.
    _bookkeeping_lock: GalleryBookkeepingGuard,
}

impl GalleryNameReservation {
    pub(crate) fn final_name(&self) -> &str {
        &self.final_name
    }

    pub(crate) fn authority(&self) -> &GalleryBookkeepingGuard {
        &self._bookkeeping_lock
    }
}

impl Drop for GalleryNameReservation {
    fn drop(&mut self) {
        if let Err(error) = release_reservation(&self.output_dir, &self.final_name, &self.owner) {
            tracing::warn!(
                final_name = %self.final_name,
                %error,
                "failed to release ordinary gallery filename reservation"
            );
            return;
        }
        let reservations = reservations_dir(&self.output_dir);
        let transaction_root = self.output_dir.join(TRANSACTION_DIR);
        let _ = sync_dir(&reservations);
        // Ordinary saves must not leave transaction bookkeeping in an
        // otherwise clean gallery. Both removals are non-recursive and
        // therefore harmless when another save or durable batch still owns
        // anything below these directories.
        let _ = fs::remove_dir(&reservations);
        let _ = fs::remove_dir(&transaction_root);
        let _ = sync_dir(&self.output_dir);
    }
}

#[derive(Debug)]
pub struct BatchTransaction {
    output_dir: PathBuf,
    attempt_dir: PathBuf,
    // A live attempt owns this OS lock from before its first durable manifest
    // through publication, archive/rollback cleanup, and Drop. Recovery may
    // inspect or mutate an attempt only after claiming the same authority.
    _attempt_authority: AttemptAuthority,
    // Joint parent-owned attempts retain the stable parent authority after
    // the generation-scoped authority. Field order is load-bearing: Rust
    // drops `_attempt_authority` before `_parent_authority`.
    _parent_authority: Option<ParentAuthority>,
    manifest: BatchAttemptManifest,
    next_journal_sequence: u64,
    journaled_final_children: BTreeSet<usize>,
    post_publish_snapshot: bool,
    metadata_committed: bool,
    cleanup_started: bool,
    reconstructed_from_journal: bool,
    journal_needs_heal: bool,
    staged_receipts: BTreeMap<usize, BatchChildReceipt>,
    poisoned: bool,
    #[cfg(test)]
    commit_failpoint: Option<CommitFailpoint>,
    #[cfg(test)]
    delta_append_failpoint: Option<DeltaAppendFailpoint>,
}

#[derive(Debug)]
pub(crate) struct LockedAuthorityFile {
    file: Option<File>,
    path: PathBuf,
}

#[derive(Debug)]
pub(crate) enum AttemptAuthority {
    Held {
        legacy: LockedAuthorityFile,
        current: LockedAuthorityFile,
        output_dir: PathBuf,
        attempt_dir: PathBuf,
        reclaim_on_drop: bool,
    },
    // Direct journal/manifest unit tests inspect deliberately live fixtures.
    // Production construction has no unclaimed variant.
    #[cfg(test)]
    Unclaimed,
}

#[derive(Debug)]
pub(crate) struct ParentAuthority {
    file: Option<File>,
    path: PathBuf,
    canonical_output_dir: PathBuf,
    parent_id: String,
}

impl ParentAuthority {
    pub(crate) fn validate_identity(
        &self,
        output_dir: &Path,
        parent_id: &str,
    ) -> anyhow::Result<()> {
        ensure!(
            self.parent_id == parent_id
                && self.canonical_output_dir == fs::canonicalize(output_dir)?
                && self.path == parent_authority_lock_path(output_dir, parent_id)?,
            "batch parent authority identity drift"
        );
        Ok(())
    }
}

impl Drop for ParentAuthority {
    fn drop(&mut self) {
        if let Some(file) = self.file.as_ref() {
            let _ = fs2::FileExt::unlock(file);
        }
        drop(self.file.take());
        // The stable parent pathname is deliberately never unlinked. Every
        // claimant opens this same inode under gallery bookkeeping, so a
        // replaceable lockfile can never split parent authority.
    }
}

impl Drop for AttemptAuthority {
    fn drop(&mut self) {
        match self {
            Self::Held {
                legacy,
                current,
                output_dir,
                attempt_dir,
                reclaim_on_drop,
            } => {
                if !*reclaim_on_drop {
                    return;
                }
                if legacy.file.is_none() || current.file.is_none() {
                    return;
                }
                // Every production authority open is serialized by this same
                // bookkeeping lock. Claim paths never wait for an attempt
                // while holding bookkeeping, so retaining the attempt lock
                // while this Drop waits cannot form a lock cycle. Once both
                // locks are held, unlink-before-unlock safely gives the v2
                // sidecar bounded cleanup. The v1 pathname stays reusable.
                match acquire_gallery_bookkeeping_lock(output_dir) {
                    Ok(bookkeeping) => {
                        reclaim_locked_authorities(current, &bookkeeping, "terminal batch attempt");
                        unlock_and_close_authority(current);
                        unlock_and_close_authority(legacy);
                        cleanup_empty_transaction_hierarchy(output_dir, Some(attempt_dir));
                        drop(bookkeeping);
                    }
                    Err(error) => {
                        tracing::warn!(
                            gallery = %output_dir.display(),
                            attempt = %attempt_dir.display(),
                            %error,
                            "failed to acquire gallery bookkeeping while dropping batch attempt; \
                             retaining authority paths for startup recovery"
                        );
                        unlock_and_close_authority(current);
                        unlock_and_close_authority(legacy);
                    }
                }
            }
            #[cfg(test)]
            Self::Unclaimed => {}
        }
    }
}

impl AttemptAuthority {
    /// Reclaim while the caller already owns gallery bookkeeping. This is
    /// used only for an orphan directory discovered during the locked startup
    /// scan; reacquiring the same lock from Drop would self-deadlock.
    fn reclaim_under_bookkeeping(&mut self, _bookkeeping: &GalleryBookkeepingGuard) {
        let (legacy, current, output_dir, attempt_dir, reclaim_on_drop) = match self {
            Self::Held {
                legacy,
                current,
                output_dir,
                attempt_dir,
                reclaim_on_drop,
            } => (legacy, current, output_dir, attempt_dir, reclaim_on_drop),
            #[cfg(test)]
            Self::Unclaimed => return,
        };
        if legacy.file.is_none() || current.file.is_none() {
            *reclaim_on_drop = false;
            return;
        }
        reclaim_locked_authorities(current, _bookkeeping, "orphaned batch attempt");
        unlock_and_close_authority(current);
        unlock_and_close_authority(legacy);
        cleanup_empty_transaction_hierarchy(output_dir, Some(attempt_dir));
        *reclaim_on_drop = false;
    }
}

fn reclaim_locked_authorities(
    current: &LockedAuthorityFile,
    bookkeeping: &GalleryBookkeepingGuard,
    owner: &str,
) {
    // Only v2 authority names are unlinked. A rolling v1 process can open its
    // legacy pathname before acquiring the OS lock and does not participate
    // in v2 gallery bookkeeping. Retaining every legacy pathname as a stable,
    // reusable inode closes that otherwise unavoidable open-before-lock race.
    let Some(file) = current.file.as_ref() else {
        return;
    };
    match verify_authority_file_at_path(&current.path, file, bookkeeping) {
        Ok(()) => {
            if let Err(error) = remove_locked_authority_path(&current.path, file, bookkeeping) {
                if !anyhow_error_is_not_found(&error) {
                    tracing::warn!(
                        authority = %current.path.display(),
                        %error,
                        "failed to reclaim {owner} authority"
                    );
                }
            } else if let Some(parent) = current.path.parent() {
                let _ = sync_attempt_authority_parent(parent);
            }
        }
        Err(error) => {
            tracing::warn!(
                authority = %current.path.display(),
                %error,
                "retaining replaced {owner} authority path"
            );
        }
    }
}

fn unlock_and_close_authority(authority: &mut LockedAuthorityFile) {
    if let Some(file) = authority.file.as_ref() {
        let _ = fs2::FileExt::unlock(file);
    }
    drop(authority.file.take());
}

fn anyhow_error_is_not_found(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
}

#[derive(Debug)]
struct ClaimedAttempt {
    manifest_path: PathBuf,
    authority: AttemptAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitFailpoint {
    #[cfg(test)]
    PanicAfterCommitting,
    CommittingState,
    FinalPublish(usize),
    FinalFileFsync(usize),
    FinalJournalFsync(usize),
    OutputDirectoryFsync,
    MetadataManifestFsync,
    DatabaseTransaction,
    DatabaseJournalFsync,
    CommittedState,
    ReservationsReleased,
    PrivateStagingCleaned,
    CleanupStartedJournal,
    ArchiveWrite,
    ArchivedManifest,
    AttemptDirectoryRemoved,
    AttemptsDirectoryFsync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeltaAppendFailpoint {
    Open,
    Write,
    Flush,
    FileSync,
    DirectorySync,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub rolled_back: usize,
    pub rolled_forward: usize,
    pub healed_committed_rows: usize,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "batch attempt {parent_id}/{attempt_generation} is still owned by another live transaction"
)]
pub struct BatchAttemptAuthorityContended {
    pub parent_id: String,
    pub attempt_generation: u64,
}

#[derive(Debug, thiserror::Error)]
#[error("batch parent {parent_id} is still owned by another live process")]
pub struct BatchParentAuthorityContended {
    pub parent_id: String,
}

impl BatchTransaction {
    /// Create a generation-scoped transaction and reserve collision-safe final
    /// names. The returned manifest is already durable.
    pub fn begin(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        records: Vec<GenerationRecord>,
    ) -> anyhow::Result<Self> {
        Self::begin_with_reservation_directory_hook(
            output_dir,
            parent_id,
            attempt_generation,
            normalized_request,
            records,
            || {},
        )
    }

    fn begin_with_reservation_directory_hook(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        records: Vec<GenerationRecord>,
        after_reservation_directory_created: impl FnOnce(),
    ) -> anyhow::Result<Self> {
        Self::begin_with_hooks(
            output_dir,
            parent_id,
            attempt_generation,
            normalized_request,
            records,
            || {},
            after_reservation_directory_created,
            false,
        )
    }

    #[cfg(test)]
    fn begin_with_attempt_authority_hook(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        records: Vec<GenerationRecord>,
        after_attempt_directory_created: impl FnOnce(),
    ) -> anyhow::Result<Self> {
        Self::begin_with_hooks(
            output_dir,
            parent_id,
            attempt_generation,
            normalized_request,
            records,
            after_attempt_directory_created,
            || {},
            false,
        )
    }

    pub(crate) fn begin_parent_owned(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        records: Vec<GenerationRecord>,
    ) -> anyhow::Result<Self> {
        Self::begin_with_hooks(
            output_dir,
            parent_id,
            attempt_generation,
            normalized_request,
            records,
            || {},
            || {},
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn begin_under_parent_authority(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        records: Vec<GenerationRecord>,
        parent_authority: &ParentAuthority,
    ) -> anyhow::Result<Self> {
        parent_authority.validate_identity(output_dir, parent_id)?;
        // The stable parent authority is already retained by the caller.
        // `begin` acquires bookkeeping and then the generation authority, so
        // retry preserves parent -> attempt lifetime ordering without trying
        // to recursively lock the parent.
        Self::begin(
            output_dir,
            parent_id,
            attempt_generation,
            normalized_request,
            records,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "test hooks and the parent-claim mode are explicit durability boundaries"
    )]
    fn begin_with_hooks(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        mut records: Vec<GenerationRecord>,
        after_attempt_directory_created: impl FnOnce(),
        after_reservation_directory_created: impl FnOnce(),
        claim_parent_authority: bool,
    ) -> anyhow::Result<Self> {
        validate_component(parent_id, "parent id")?;
        ensure!(!records.is_empty(), "batch transaction must have children");
        for record in &records {
            validate_component(&record.filename, "requested final filename")?;
        }
        fs::create_dir_all(output_dir)
            .with_context(|| format!("creating gallery {}", output_dir.display()))?;
        // Lock the stable gallery directory inode, not a removable file below
        // the transaction tree. This serializes directory lifecycle across
        // threads and processes without leaving cleanup bookkeeping behind.
        // In particular, an ordinary reservation drop cannot rmdir the empty
        // reservations directory between this begin and its first durable
        // create-new reservation.
        // This slot is declared before bookkeeping so panic unwinding releases
        // the later bookkeeping guard before dropping an owned attempt.
        let mut parent_authority_for_unwind: Option<ParentAuthority> = None;
        let mut attempt_authority_for_unwind: Option<AttemptAuthority>;
        let _bookkeeping_lock = acquire_gallery_bookkeeping_lock(output_dir)?;
        let canonical_output_dir = _bookkeeping_lock.canonical_root().to_path_buf();
        sweep_reclaimable_attempt_authorities(&_bookkeeping_lock)?;
        if claim_parent_authority {
            parent_authority_for_unwind = Some(
                try_claim_parent_authority(parent_id, &_bookkeeping_lock)?.ok_or_else(|| {
                    BatchParentAuthorityContended {
                        parent_id: parent_id.to_owned(),
                    }
                })?,
            );
        }
        let attempt_dir = attempt_dir(output_dir, parent_id, attempt_generation);
        let attempts_root = attempt_dir
            .parent()
            .context("batch attempt path has no attempts root")?;
        fs::create_dir_all(attempts_root)?;
        match fs::create_dir(&attempt_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return match try_claim_attempt_authority(&attempt_dir, &_bookkeeping_lock) {
                    Ok(None) => Err(BatchAttemptAuthorityContended {
                        parent_id: parent_id.to_owned(),
                        attempt_generation,
                    }
                    .into()),
                    Ok(Some(authority)) => {
                        // The existing directory is durable orphan evidence,
                        // not a live owner. Release bookkeeping before the
                        // authority destructor reclaims its path.
                        drop(_bookkeeping_lock);
                        drop(authority);
                        Err(error).with_context(|| {
                            format!(
                                "batch attempt {parent_id}/{attempt_generation} already exists \
                                 without a live owner; run startup recovery before retrying"
                            )
                        })
                    }
                    Err(claim_error) => Err(claim_error).with_context(|| {
                        format!(
                            "inspecting existing batch attempt authority \
                             {parent_id}/{attempt_generation}"
                        )
                    }),
                };
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "claiming batch attempt authority {parent_id}/{attempt_generation}; \
                         an existing attempt is never shared or overwritten"
                    )
                });
            }
        }
        after_attempt_directory_created();
        // Never wait for attempt authority while holding gallery bookkeeping.
        // A terminal owner may retain that authority after removing its
        // attempt directory and then need bookkeeping for unrelated work.
        // At this point the replacement directory is still empty, so exact
        // contention can be cleaned up without touching durable evidence.
        attempt_authority_for_unwind = Some(
            match try_claim_attempt_authority(&attempt_dir, &_bookkeeping_lock) {
                Ok(Some(authority)) => authority,
                Ok(None) => {
                    fs::remove_dir(&attempt_dir).with_context(|| {
                        format!(
                            "removing empty contended batch attempt {}",
                            attempt_dir.display()
                        )
                    })?;
                    sync_dir(attempts_root)?;
                    return Err(BatchAttemptAuthorityContended {
                        parent_id: parent_id.to_owned(),
                        attempt_generation,
                    }
                    .into());
                }
                Err(error) => {
                    let _ = fs::remove_dir(&attempt_dir);
                    return Err(error);
                }
            },
        );
        if let Err(error) = (|| {
            fs::create_dir(attempt_dir.join("staging"))?;
            fs::create_dir_all(reservations_dir(output_dir))?;
            sync_transaction_hierarchy(output_dir, &attempt_dir)?;
            after_reservation_directory_created();
            Ok(())
        })() {
            let _ = fs::remove_dir_all(&attempt_dir);
            drop(_bookkeeping_lock);
            drop(attempt_authority_for_unwind.take());
            return Err(error);
        }

        let reservation_owner = ReservationOwner {
            parent_id: parent_id.to_owned(),
            attempt_generation,
        };
        let mut children = Vec::with_capacity(records.len());
        let mut reserved_names: Vec<String> = Vec::with_capacity(records.len());
        for (index, record) in records.iter_mut().enumerate() {
            // Transaction-owned staging establishes fresh filesystem identity.
            // Caller-provided DB/stat fields cannot become part of the initial
            // snapshot because the protocol derives them from staged/final
            // files at its durable boundaries.
            record.id = None;
            record.file_mtime_ms = None;
            record.file_size_bytes = None;
            let final_name =
                match reserve_final_name(output_dir, &record.filename, &reservation_owner) {
                    Ok(name) => name,
                    Err(error) => {
                        for name in &reserved_names {
                            let _ = release_reservation(output_dir, name, &reservation_owner);
                        }
                        let _ = fs::remove_dir_all(&attempt_dir);
                        drop(_bookkeeping_lock);
                        drop(attempt_authority_for_unwind.take());
                        return Err(error);
                    }
                };
            reserved_names.push(final_name.clone());
            record.filename.clone_from(&final_name);
            record.output_dir = canonical_output_dir.to_string_lossy().into_owned();
            children.push(BatchManifestChild {
                child_index: index,
                staging_name: format!("{index:08}.stage"),
                final_name,
                checksum_sha256: None,
                size_bytes: None,
                record: record.clone(),
            });
        }

        // All namespace and reservation lifecycle mutations are now durable.
        // Release bookkeeping before constructing an owning value so any
        // subsequent error can drop its attempt authority in canonical
        // attempt -> bookkeeping order.
        drop(_bookkeeping_lock);
        let attempt_authority = attempt_authority_for_unwind
            .take()
            .context("batch attempt authority disappeared before construction")?;
        let manifest = BatchAttemptManifest {
            version: MANIFEST_VERSION,
            parent_id: parent_id.to_owned(),
            attempt_generation,
            normalized_request,
            state: BatchManifestState::Staging,
            children,
        };
        let mut transaction = Self {
            output_dir: canonical_output_dir,
            attempt_dir,
            _attempt_authority: attempt_authority,
            _parent_authority: parent_authority_for_unwind.take(),
            manifest,
            next_journal_sequence: 0,
            journaled_final_children: BTreeSet::new(),
            post_publish_snapshot: false,
            metadata_committed: false,
            cleanup_started: false,
            reconstructed_from_journal: false,
            journal_needs_heal: false,
            staged_receipts: BTreeMap::new(),
            poisoned: false,
            #[cfg(test)]
            commit_failpoint: None,
            #[cfg(test)]
            delta_append_failpoint: None,
        };
        if let Err(error) = transaction.persist_manifest() {
            transaction.release_reservations();
            let _ = fs::remove_dir_all(&transaction.attempt_dir);
            return Err(error);
        }
        Ok(transaction)
    }

    pub fn manifest(&self) -> &BatchAttemptManifest {
        &self.manifest
    }

    pub(crate) fn take_parent_authority(&mut self) -> anyhow::Result<ParentAuthority> {
        self._parent_authority
            .take()
            .context("joint batch transaction has no parent authority")
    }

    pub(crate) fn protocol_version(&self) -> u32 {
        self.manifest.version
    }

    pub(crate) fn recover_exact_claimed(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        authority: AttemptAuthority,
    ) -> anyhow::Result<Self> {
        let attempt_dir = attempt_dir(output_dir, parent_id, attempt_generation);
        Self::load_claimed(output_dir, &attempt_dir.join(MANIFEST_FILE), authority)
    }

    /// Load a retained commit proof only after re-validating both its immutable
    /// manifest identity and every published child on disk.
    pub(crate) fn load_validated_committed_archive(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
    ) -> anyhow::Result<BatchAttemptManifest> {
        validate_component(parent_id, "batch parent id")?;
        let path = committed_manifests_dir(output_dir, parent_id)
            .join(format!("{attempt_generation}.json"));
        let manifest: BatchAttemptManifest = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("reading archived batch commit {}", path.display()))?;
        validate_loaded_manifest(
            output_dir,
            &attempt_dir(output_dir, parent_id, attempt_generation),
            &manifest,
        )?;
        ensure!(
            manifest.parent_id == parent_id
                && manifest.attempt_generation == attempt_generation
                && manifest.state == BatchManifestState::Committed,
            "archived batch commit identity/state is invalid"
        );
        for child in &manifest.children {
            let expected_checksum = child
                .checksum_sha256
                .as_deref()
                .context("committed archive child has no checksum")?;
            let expected_size = child
                .size_bytes
                .context("committed archive child has no size")?;
            let final_path = output_dir.join(&child.final_name);
            let metadata = fs::metadata(&final_path).with_context(|| {
                format!(
                    "committed archive child is missing: {}",
                    final_path.display()
                )
            })?;
            ensure!(
                metadata.is_file() && metadata.len() == expected_size,
                "committed archive child size changed: {}",
                final_path.display()
            );
            ensure!(
                checksum_file(&final_path)? == expected_checksum,
                "committed archive child checksum changed: {}",
                final_path.display()
            );
        }
        Ok(manifest)
    }

    pub(crate) fn archived_child_identity(child: &BatchManifestChild) -> anyhow::Result<String> {
        immutable_record_identity(child)
    }

    pub fn staging_path(&self, child_index: usize) -> anyhow::Result<PathBuf> {
        let child = self
            .manifest
            .children
            .get(child_index)
            .context("batch child index out of range")?;
        Ok(self.attempt_dir.join("staging").join(&child.staging_name))
    }

    /// Write one private child artifact, fsync it, and journal its checksum.
    pub fn stage_bytes(&mut self, child_index: usize, bytes: &[u8]) -> anyhow::Result<()> {
        let lease = BatchChildLease {
            parent_id: self.manifest.parent_id.clone(),
            child_index,
            attempt_generation: self.manifest.attempt_generation,
            lease_generation: 0,
        };
        self.stage_bytes_for_lease(&lease, bytes).map(|_| ())
    }

    /// Stage a child for one exact reducer lease. The returned receipt is
    /// durable only after the file, staging directory, and ChildStaged journal
    /// delta have all been fsynced, in that order.
    pub fn stage_bytes_for_lease(
        &mut self,
        lease: &BatchChildLease,
        bytes: &[u8],
    ) -> anyhow::Result<BatchChildReceipt> {
        self.ensure_usable()?;
        ensure!(
            self.manifest.state == BatchManifestState::Staging,
            "children can only be staged while the attempt is staging"
        );
        ensure!(
            lease.parent_id == self.manifest.parent_id
                && lease.attempt_generation == self.manifest.attempt_generation,
            "child lease does not belong to this transaction attempt"
        );
        let path = self.staging_path(lease.child_index)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("creating staged child {}", path.display()))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        sync_dir(path.parent().expect("staging path has parent"))?;

        let receipt = BatchChildReceipt {
            version: BatchChildReceipt::VERSION,
            parent_id: self.manifest.parent_id.clone(),
            attempt_generation: self.manifest.attempt_generation,
            child_index: lease.child_index,
            lease_generation: lease.lease_generation,
            checksum_sha256: checksum_bytes(bytes),
            size_bytes: bytes.len() as u64,
            record_identity_sha256: immutable_record_identity(
                &self.manifest.children[lease.child_index],
            )?,
        };
        self.append_journal(BatchJournalEvent::ChildStaged {
            receipt: receipt.clone(),
        })?;
        let child = &mut self.manifest.children[lease.child_index];
        child.checksum_sha256 = Some(receipt.checksum_sha256.clone());
        child.size_bytes = Some(receipt.size_bytes);
        child.record.file_size_bytes = Some(receipt.size_bytes as i64);
        self.staged_receipts
            .insert(lease.child_index, receipt.clone());
        Ok(receipt)
    }

    pub(crate) fn staged_receipts(&self) -> &BTreeMap<usize, BatchChildReceipt> {
        &self.staged_receipts
    }

    /// Remove an unaccepted receipt from only this generation's private
    /// staging directory, fsync the removal, then append its tombstone.
    pub(crate) fn unstage_receipt(&mut self, child_index: usize) -> anyhow::Result<()> {
        self.ensure_usable()?;
        let receipt = self
            .staged_receipts
            .get(&child_index)
            .cloned()
            .context("batch child has no staged receipt")?;
        let path = self.staging_path(child_index)?;
        match fs::remove_file(&path) {
            Ok(()) => sync_dir(path.parent().expect("staging path has parent"))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        self.append_journal(BatchJournalEvent::ChildUnstaged {
            child_index,
            lease_generation: receipt.lease_generation,
            record_identity_sha256: receipt.record_identity_sha256,
        })?;
        let child = &mut self.manifest.children[child_index];
        child.checksum_sha256 = None;
        child.size_bytes = None;
        child.record.file_size_bytes = None;
        child.record.file_mtime_ms = None;
        self.staged_receipts.remove(&child_index);
        Ok(())
    }

    pub(crate) fn remove_unreceipted_staging(&mut self) -> anyhow::Result<usize> {
        self.ensure_usable()?;
        let mut removed = 0;
        for child_index in 0..self.manifest.children.len() {
            if self.staged_receipts.contains_key(&child_index) {
                continue;
            }
            let path = self.staging_path(child_index)?;
            match fs::remove_file(&path) {
                Ok(()) => removed += 1,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        if removed > 0 {
            sync_dir(&self.attempt_dir.join("staging"))?;
        }
        Ok(removed)
    }

    pub(crate) fn rollback_private_attempt(&mut self) -> anyhow::Result<()> {
        self.ensure_usable()?;
        ensure!(
            matches!(
                self.manifest.state,
                BatchManifestState::Staging
                    | BatchManifestState::Prepared
                    | BatchManifestState::Failed
            ),
            "cannot roll back transaction in {:?}",
            self.manifest.state
        );
        ensure!(
            !self.any_final_exists(),
            "private rollback found a final-path artifact"
        );
        if self.manifest.state != BatchManifestState::Failed {
            self.manifest.state = BatchManifestState::Failed;
            self.persist_manifest()?;
        }
        self.release_reservations();
        self.cleanup_private_staging();
        remove_failed_attempt(self);
        Ok(())
    }

    pub(crate) fn retire_committed_attempt(&mut self) -> anyhow::Result<()> {
        self.ensure_usable()?;
        ensure!(
            self.manifest.state == BatchManifestState::Committed,
            "only a committed transaction can be retired"
        );
        self.verify_all_committed()?;
        self.archive_committed_attempt(true)
    }

    /// Seal a child that was streamed directly into its transaction-owned
    /// staging path. Metadata is immutable from `begin`; this transition adds
    /// only the staged file checksum and size before the normal journaled
    /// commit.
    pub fn seal_staged_file(&mut self, child_index: usize) -> anyhow::Result<()> {
        self.ensure_usable()?;
        ensure!(
            self.manifest.state == BatchManifestState::Staging,
            "children can only be staged while the attempt is staging"
        );
        let path = self.staging_path(child_index)?;
        File::open(&path)?.sync_all()?;
        sync_dir(path.parent().expect("staging path has parent"))?;

        let size = fs::metadata(&path)?.len();
        let child = &mut self.manifest.children[child_index];
        child.checksum_sha256 = Some(checksum_file(&path)?);
        child.size_bytes = Some(size);
        child.record.file_size_bytes =
            Some(i64::try_from(size).context("staged gallery import exceeds SQLite size range")?);
        self.persist_manifest()
    }

    /// Retire an unpublished staging/prepared attempt immediately. Startup
    /// recovery performs the same transition after a crash; this eager form is
    /// used when an import discovers that its bytes already exist under the
    /// requested cross-host identity.
    pub fn rollback_unpublished(mut self) -> anyhow::Result<()> {
        self.ensure_usable()?;
        ensure!(
            matches!(
                self.manifest.state,
                BatchManifestState::Staging | BatchManifestState::Prepared
            ),
            "only an unpublished batch attempt can be rolled back"
        );
        ensure!(
            !self.any_final_exists(),
            "unpublished batch attempt unexpectedly has a public final"
        );
        self.manifest.state = BatchManifestState::Failed;
        self.persist_manifest()?;
        self.release_reservations();
        self.cleanup_private_staging();
        remove_failed_attempt(&self);
        Ok(())
    }

    /// Verify every child and durably close the attempt to further staging.
    pub fn mark_prepared(&mut self) -> anyhow::Result<()> {
        self.ensure_usable()?;
        ensure!(
            self.manifest.state == BatchManifestState::Staging,
            "only a staging attempt can become prepared"
        );
        self.verify_all_staged()?;
        self.manifest.state = BatchManifestState::Prepared;
        self.persist_manifest()
    }

    /// Publish every child while excluding all gallery observers and
    /// mutators. Errors after `committing` are returned with the writer guard
    /// retained; dropping that error terminates the process so a live server
    /// can never expose an unresolved transaction.
    pub async fn commit(
        &mut self,
        gate: &GalleryPublicationGate,
        db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Result<(), UnresolvedBatchCommit> {
        if let Err(error) = self.validate_commit_entry() {
            return Err(UnresolvedBatchCommit::pre_commit(error));
        }
        let guard = gate.write().await;
        let mut bookkeeping = match acquire_gallery_bookkeeping_lock(&self.output_dir) {
            Ok(bookkeeping) => Some(bookkeeping),
            Err(error) => return Err(UnresolvedBatchCommit::pre_commit(error)),
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.commit_while_locked(gate, &db, &mut bookkeeping)
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(UnresolvedBatchCommit::committing(error, guard, bookkeeping)),
            Err(payload) => Err(UnresolvedBatchCommit::committing(
                anyhow::anyhow!(
                    "atomic batch commit panicked: {}",
                    panic_payload_message(payload.as_ref())
                ),
                guard,
                bookkeeping,
            )),
        }
    }

    /// Resume a committing attempt whose staging files were lost only after
    /// every reserved final file has been checksum-verified. This is startup
    /// recovery only: ordinary live commits must still enter through verified
    /// staging.
    async fn commit_verified_finals(
        &mut self,
        gate: &GalleryPublicationGate,
        db: Arc<Option<mold_db::MetadataDb>>,
    ) -> Result<(), UnresolvedBatchCommit> {
        if let Err(error) = self.ensure_usable().and_then(|()| {
            ensure!(
                self.manifest.state == BatchManifestState::Committing,
                "only a committing batch can recover from verified finals"
            );
            Ok(())
        }) {
            return Err(UnresolvedBatchCommit::pre_commit(error));
        }
        let guard = gate.write().await;
        let mut bookkeeping = match acquire_gallery_bookkeeping_lock(&self.output_dir) {
            Ok(bookkeeping) => Some(bookkeeping),
            Err(error) => return Err(UnresolvedBatchCommit::pre_commit(error)),
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.verify_owned_reservations()?;
            self.verify_all_committed()?;
            self.commit_while_locked(gate, &db, &mut bookkeeping)
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(UnresolvedBatchCommit::committing(error, guard, bookkeeping)),
            Err(payload) => Err(UnresolvedBatchCommit::committing(
                anyhow::anyhow!(
                    "atomic batch recovery commit panicked: {}",
                    panic_payload_message(payload.as_ref())
                ),
                guard,
                bookkeeping,
            )),
        }
    }

    fn validate_commit_entry(&self) -> anyhow::Result<()> {
        self.ensure_usable()?;
        ensure!(
            matches!(
                self.manifest.state,
                BatchManifestState::Prepared | BatchManifestState::Committing
            ),
            "batch attempt is not prepared for commit"
        );
        self.verify_owned_reservations()?;
        self.verify_all_staged()
    }

    fn verify_owned_reservations(&self) -> anyhow::Result<()> {
        let expected = self.reservation_owner();
        for child in &self.manifest.children {
            let path = reservation_path(&self.output_dir, &child.final_name);
            let actual: ReservationOwner = serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("reading reservation {}", path.display()))?;
            ensure!(
                actual == expected,
                "batch final reservation {} is no longer owned by attempt {}/generation {}",
                child.final_name,
                expected.parent_id,
                expected.attempt_generation
            );
        }
        Ok(())
    }

    fn commit_while_locked(
        &mut self,
        gate: &GalleryPublicationGate,
        db: &Arc<Option<mold_db::MetadataDb>>,
        bookkeeping: &mut Option<GalleryBookkeepingGuard>,
    ) -> anyhow::Result<()> {
        if self.manifest.state == BatchManifestState::Prepared {
            self.manifest.state = BatchManifestState::Committing;
            self.persist_manifest()?;
            self.inject_commit_error(CommitFailpoint::CommittingState)?;
        }
        #[cfg(test)]
        if self.commit_failpoint == Some(CommitFailpoint::PanicAfterCommitting) {
            self.commit_failpoint = None;
            panic!("injected panic after committing");
        }

        for child_index in 0..self.manifest.children.len() {
            let child = &self.manifest.children[child_index];
            let staged = self.attempt_dir.join("staging").join(&child.staging_name);
            let final_path = self.output_dir.join(&child.final_name);
            if final_path.exists() {
                if checksum_file(&final_path)? != child.checksum_sha256.as_deref().unwrap() {
                    ensure!(
                        !self.journaled_final_children.contains(&child_index),
                        "durably published batch child changed after publication: {}",
                        final_path.display()
                    );
                    // A no-replace copy can be interrupted by a power loss
                    // before its FinalPublished journal record. The attempt's
                    // owner-scoped reservation makes that unjournaled path
                    // ours to remove and reconstruct from verified staging.
                    fs::remove_file(&final_path)?;
                    sync_dir(&self.output_dir)?;
                    publish_no_replace(&staged, &final_path)?;
                }
            } else {
                publish_no_replace(&staged, &final_path)?;
            }
            self.inject_commit_error(CommitFailpoint::FinalPublish(child_index))?;
            File::open(&final_path)?.sync_all()?;
            self.inject_commit_error(CommitFailpoint::FinalFileFsync(child_index))?;
            self.manifest.children[child_index]
                .record
                .stat_from_disk(&final_path);
            if !self.journaled_final_children.contains(&child_index) {
                self.append_journal(BatchJournalEvent::FinalPublished { child_index })?;
                self.journaled_final_children.insert(child_index);
                self.inject_commit_error(CommitFailpoint::FinalJournalFsync(child_index))?;
            }
        }
        sync_dir(&self.output_dir)?;
        self.inject_commit_error(CommitFailpoint::OutputDirectoryFsync)?;
        // Make the stat data used for the all-row transaction durable before
        // SQLite can observe it. Recovery replays this exact snapshot.
        if !self.post_publish_snapshot {
            self.persist_manifest()?;
            self.post_publish_snapshot = true;
            self.inject_commit_error(CommitFailpoint::MetadataManifestFsync)?;
        }

        let mut authority_manifest = self.manifest.clone();
        authority_manifest.state = BatchManifestState::Committed;
        self.persist_committed_archive_manifest_value(&authority_manifest)?;
        // Hard-link publication shares inode ctime with private staging.
        // Remove the private link before capturing stable final-file facts,
        // while retaining the attempt manifest as recovery evidence until
        // central authority is durable.
        self.cleanup_private_staging();
        self.inject_commit_crash(CommitFailpoint::PrivateStagingCleaned);
        gate.record_committed_manifest(
            &self.output_dir,
            &authority_manifest,
            bookkeeping
                .as_ref()
                .context("gallery bookkeeping authority released before publication")?,
        )?;
        // The checksummed filesystem authority is now durable. Cross-process
        // readers may observe the complete batch while SQLite is projected
        // below, so do not hold the machine-wide namespace lock over DB I/O.
        bookkeeping.take();

        // SQLite is a projection. It is updated only after the filesystem
        // authority is durable; a DB failure leaves a recoverable committed
        // attempt and can never make a partial batch authoritative.
        if !self.metadata_committed {
            if let Some(db) = db.as_ref() {
                let records: Vec<_> = self
                    .manifest
                    .children
                    .iter()
                    .map(|child| child.record.clone())
                    .collect();
                db.upsert_batch(&records)?;
            }
            self.inject_commit_error(CommitFailpoint::DatabaseTransaction)?;
            self.append_journal(BatchJournalEvent::MetadataCommitted)?;
            self.metadata_committed = true;
            self.inject_commit_error(CommitFailpoint::DatabaseJournalFsync)?;
        }
        self.manifest.state = BatchManifestState::Committed;
        self.persist_manifest()?;
        self.inject_commit_error(CommitFailpoint::CommittedState)?;
        self.release_reservations();
        self.inject_commit_crash(CommitFailpoint::ReservationsReleased);
        self.archive_committed_attempt(false)?;
        Ok(())
    }

    #[cfg(test)]
    fn inject_commit_error(&mut self, point: CommitFailpoint) -> anyhow::Result<()> {
        if self.commit_failpoint == Some(point) {
            self.commit_failpoint = None;
            anyhow::bail!("injected commit fault at {point:?}");
        }
        Ok(())
    }

    #[cfg(not(test))]
    fn inject_commit_error(&mut self, _point: CommitFailpoint) -> anyhow::Result<()> {
        Ok(())
    }

    #[cfg(test)]
    fn inject_commit_crash(&mut self, point: CommitFailpoint) {
        if self.commit_failpoint == Some(point) {
            self.commit_failpoint = None;
            panic!("injected commit crash at {point:?}");
        }
    }

    #[cfg(not(test))]
    fn inject_commit_crash(&mut self, _point: CommitFailpoint) {}

    fn verify_all_staged(&self) -> anyhow::Result<()> {
        for child in &self.manifest.children {
            let expected = child
                .checksum_sha256
                .as_deref()
                .context("batch child has no staged checksum")?;
            let path = self.attempt_dir.join("staging").join(&child.staging_name);
            ensure!(
                checksum_file(&path)? == expected,
                "staged child checksum changed: {}",
                path.display()
            );
        }
        Ok(())
    }

    fn any_final_exists(&self) -> bool {
        self.manifest
            .children
            .iter()
            .any(|child| self.output_dir.join(&child.final_name).exists())
    }

    fn verify_all_committed(&self) -> anyhow::Result<()> {
        for child in &self.manifest.children {
            let expected = child
                .checksum_sha256
                .as_deref()
                .context("committed batch child has no checksum")?;
            let path = self.output_dir.join(&child.final_name);
            ensure!(
                path.is_file(),
                "committed batch child is missing: {}",
                path.display()
            );
            ensure!(
                checksum_file(&path)? == expected,
                "committed batch child checksum changed: {}",
                path.display()
            );
        }
        Ok(())
    }

    fn persist_manifest(&mut self) -> anyhow::Result<()> {
        self.ensure_usable()?;
        let result = (|| {
            atomic_write_json(&self.attempt_dir.join(MANIFEST_FILE), &self.manifest)?;
            self.append_journal(BatchJournalEvent::ManifestSnapshot {
                manifest: self.manifest.clone(),
            })
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn ensure_usable(&self) -> anyhow::Result<()> {
        ensure!(
            !self.poisoned,
            "batch transaction is poisoned after a manifest persistence failure; \
             retire it and recover from durable state"
        );
        Ok(())
    }

    fn append_journal(&mut self, event: BatchJournalEvent) -> anyhow::Result<()> {
        self.ensure_usable()?;
        let result = (|| {
            let record = BatchJournalRecord {
                sequence: self.next_journal_sequence,
                attempt_generation: self.manifest.attempt_generation,
                event,
            };
            let path = self.attempt_dir.join(JOURNAL_FILE);
            let mut bytes = serde_json::to_vec(&record)?;
            bytes.push(b'\n');
            self.inject_delta_append_error(DeltaAppendFailpoint::Open)?;
            let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
            file.write_all(&bytes)?;
            self.inject_delta_append_error(DeltaAppendFailpoint::Write)?;
            file.flush()?;
            self.inject_delta_append_error(DeltaAppendFailpoint::Flush)?;
            file.sync_all()?;
            self.inject_delta_append_error(DeltaAppendFailpoint::FileSync)?;
            sync_dir(&self.attempt_dir)?;
            self.inject_delta_append_error(DeltaAppendFailpoint::DirectorySync)?;
            self.next_journal_sequence = self
                .next_journal_sequence
                .checked_add(1)
                .context("batch journal sequence overflow")?;
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    #[cfg(test)]
    fn inject_delta_append_error(&mut self, point: DeltaAppendFailpoint) -> anyhow::Result<()> {
        if self.delta_append_failpoint == Some(point) {
            self.delta_append_failpoint = None;
            anyhow::bail!("injected batch delta append fault at {point:?}");
        }
        Ok(())
    }

    #[cfg(not(test))]
    fn inject_delta_append_error(&mut self, _point: DeltaAppendFailpoint) -> anyhow::Result<()> {
        Ok(())
    }

    fn reservation_owner(&self) -> ReservationOwner {
        ReservationOwner {
            parent_id: self.manifest.parent_id.clone(),
            attempt_generation: self.manifest.attempt_generation,
        }
    }

    fn release_reservations(&self) {
        let owner = self.reservation_owner();
        for child in &self.manifest.children {
            if let Err(error) = release_reservation(&self.output_dir, &child.final_name, &owner) {
                tracing::warn!(
                    final_name = %child.final_name,
                    %error,
                    "failed to release owned batch filename reservation"
                );
            }
        }
        let _ = sync_dir(&reservations_dir(&self.output_dir));
    }

    fn cleanup_private_staging(&self) {
        if let Err(error) = fs::remove_dir_all(self.attempt_dir.join("staging")) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    attempt = %self.attempt_dir.display(),
                    %error,
                    "failed to remove committed batch staging directory"
                );
            }
        }
        let _ = sync_dir(&self.attempt_dir);
    }

    fn archive_committed_attempt(&mut self, retain_manifest: bool) -> anyhow::Result<()> {
        ensure!(
            self.manifest.state == BatchManifestState::Committed,
            "only a committed attempt can be archived"
        );
        if !self.cleanup_started {
            self.append_journal(BatchJournalEvent::CleanupStarted)?;
            self.cleanup_started = true;
            self.inject_commit_crash(CommitFailpoint::CleanupStartedJournal);
        }
        if retain_manifest {
            self.persist_committed_archive_manifest()?;
        }
        self.remove_committed_attempt()
    }

    fn persist_committed_archive_manifest(&mut self) -> anyhow::Result<()> {
        let manifest = self.manifest.clone();
        self.persist_committed_archive_manifest_value(&manifest)
    }

    fn persist_committed_archive_manifest_value(
        &mut self,
        manifest: &BatchAttemptManifest,
    ) -> anyhow::Result<()> {
        ensure!(
            manifest.state == BatchManifestState::Committed,
            "only committed manifests can enter gallery archive authority"
        );
        let archive_dir = committed_manifests_dir(&self.output_dir, &self.manifest.parent_id);
        fs::create_dir_all(&archive_dir)?;
        sync_dir(
            archive_dir
                .parent()
                .context("committed manifest directory has no parent")?,
        )?;
        sync_dir(&archive_dir)?;
        self.inject_commit_error(CommitFailpoint::ArchiveWrite)?;
        atomic_write_json(
            &archive_dir.join(format!("{}.json", self.manifest.attempt_generation)),
            manifest,
        )?;
        self.inject_commit_crash(CommitFailpoint::ArchivedManifest);
        Ok(())
    }

    fn remove_committed_attempt(&mut self) -> anyhow::Result<()> {
        fs::remove_dir_all(&self.attempt_dir)?;
        self.inject_commit_crash(CommitFailpoint::AttemptDirectoryRemoved);
        if let Some(attempts_dir) = self.attempt_dir.parent() {
            sync_dir(attempts_dir)?;
            self.inject_commit_crash(CommitFailpoint::AttemptsDirectoryFsync);
        }
        Ok(())
    }

    fn load_claimed(
        output_dir: &Path,
        manifest_path: &Path,
        authority: AttemptAuthority,
    ) -> anyhow::Result<Self> {
        Self::load_with_authority(output_dir, manifest_path, authority)
    }

    #[cfg(test)]
    fn load(output_dir: &Path, manifest_path: &Path) -> anyhow::Result<Self> {
        Self::load_with_authority(output_dir, manifest_path, AttemptAuthority::Unclaimed)
    }

    fn load_with_authority(
        output_dir: &Path,
        manifest_path: &Path,
        attempt_authority: AttemptAuthority,
    ) -> anyhow::Result<Self> {
        let attempt_dir = manifest_path
            .parent()
            .context("manifest has no attempt directory")?
            .to_path_buf();
        let journal = load_journal(&attempt_dir.join(JOURNAL_FILE))?;
        let journal_replay = replay_batch_journal(output_dir, &attempt_dir, &journal)
            .with_context(|| {
                format!(
                    "validating atomic batch journal {}",
                    attempt_dir.join(JOURNAL_FILE).display()
                )
            })?;
        let disk_manifest = match fs::read(manifest_path) {
            Ok(bytes) => match serde_json::from_slice::<BatchAttemptManifest>(&bytes) {
                Ok(manifest) => match validate_loaded_manifest(output_dir, &attempt_dir, &manifest)
                {
                    Ok(()) => Some(manifest),
                    Err(error) => {
                        tracing::warn!(
                            manifest = %manifest_path.display(),
                            %error,
                            "ignoring invalid atomic batch manifest in favor of its journal"
                        );
                        None
                    }
                },
                Err(error) => {
                    tracing::warn!(
                        manifest = %manifest_path.display(),
                        %error,
                        "ignoring unreadable atomic batch manifest in favor of its journal"
                    );
                    None
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let (manifest, reconstructed_from_journal, journal_needs_heal, replay) =
            reconcile_batch_manifest(output_dir, &attempt_dir, journal_replay, disk_manifest)
                .with_context(|| {
                    format!(
                        "reconciling atomic batch manifest and journal in {}",
                        attempt_dir.display()
                    )
                })?;
        ensure!(
            matches!(manifest.version, MANIFEST_VERSION_V1 | MANIFEST_VERSION),
            "unsupported batch manifest version {}",
            manifest.version
        );
        let next_journal_sequence = match journal.last() {
            Some(record) => record
                .sequence
                .checked_add(1)
                .context("batch transaction journal sequence overflow")?,
            None => 0,
        };
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            attempt_dir,
            _attempt_authority: attempt_authority,
            _parent_authority: None,
            manifest,
            next_journal_sequence,
            journaled_final_children: replay.published_children,
            post_publish_snapshot: replay.post_publish_snapshot,
            metadata_committed: replay.metadata_committed,
            cleanup_started: replay.cleanup_started,
            reconstructed_from_journal,
            journal_needs_heal,
            staged_receipts: replay.staged_receipts,
            poisoned: false,
            #[cfg(test)]
            commit_failpoint: None,
            #[cfg(test)]
            delta_append_failpoint: None,
        })
    }

    #[cfg(test)]
    fn relinquish_attempt_authority_for_recovery(&mut self) {
        // Unit recovery fixtures retain the Rust value for path assertions;
        // replacing this field models the OS releasing the lock on crash.
        self._attempt_authority = AttemptAuthority::Unclaimed;
    }
}

fn manifest_eq(left: &BatchAttemptManifest, right: &BatchAttemptManifest) -> anyhow::Result<bool> {
    Ok(serde_json::to_value(left)? == serde_json::to_value(right)?)
}

fn record_eq_ignoring_file_stat(
    left: &GenerationRecord,
    right: &GenerationRecord,
) -> anyhow::Result<bool> {
    let mut left = left.clone();
    let mut right = right.clone();
    left.file_mtime_ms = None;
    left.file_size_bytes = None;
    right.file_mtime_ms = None;
    right.file_size_bytes = None;
    Ok(serde_json::to_value(left)? == serde_json::to_value(right)?)
}

fn same_attempt_identity(
    left: &BatchAttemptManifest,
    right: &BatchAttemptManifest,
) -> anyhow::Result<bool> {
    if left.version != right.version
        || left.parent_id != right.parent_id
        || left.attempt_generation != right.attempt_generation
        || left.normalized_request != right.normalized_request
        || left.children.len() != right.children.len()
    {
        return Ok(false);
    }
    for (left, right) in left.children.iter().zip(&right.children) {
        if left.child_index != right.child_index
            || left.staging_name != right.staging_name
            || left.final_name != right.final_name
            || !record_eq_ignoring_file_stat(&left.record, &right.record)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn manifests_equal_except_state(
    previous: &BatchAttemptManifest,
    next: &BatchAttemptManifest,
) -> anyhow::Result<bool> {
    let mut next = next.clone();
    next.state = previous.state;
    manifest_eq(previous, &next)
}

fn is_initial_batch_manifest(manifest: &BatchAttemptManifest) -> bool {
    manifest.state == BatchManifestState::Staging
        && manifest.children.iter().all(|child| {
            child.checksum_sha256.is_none()
                && child.size_bytes.is_none()
                && child.record.file_mtime_ms.is_none()
                && child.record.file_size_bytes.is_none()
        })
}

fn is_staging_successor(
    previous: &BatchAttemptManifest,
    next: &BatchAttemptManifest,
) -> anyhow::Result<bool> {
    if previous.state != BatchManifestState::Staging
        || next.state != BatchManifestState::Staging
        || !same_attempt_identity(previous, next)?
    {
        return Ok(false);
    }
    let mut staged_child = None;
    for (index, (previous, next)) in previous.children.iter().zip(&next.children).enumerate() {
        if manifest_child_eq(previous, next)? {
            continue;
        }
        if staged_child.is_some()
            || previous.checksum_sha256.is_some()
            || previous.size_bytes.is_some()
            || previous.record.file_size_bytes.is_some()
            || previous.record.file_mtime_ms != next.record.file_mtime_ms
            || next.checksum_sha256.is_none()
            || next.size_bytes.is_none()
            || next.record.file_size_bytes != next.size_bytes.map(|size| size as i64)
            || !record_eq_ignoring_file_stat(&previous.record, &next.record)?
        {
            return Ok(false);
        }
        staged_child = Some(index);
    }
    Ok(staged_child.is_some())
}

fn manifest_child_eq(
    left: &BatchManifestChild,
    right: &BatchManifestChild,
) -> anyhow::Result<bool> {
    Ok(serde_json::to_value(left)? == serde_json::to_value(right)?)
}

fn is_post_publish_snapshot(
    previous: &BatchAttemptManifest,
    next: &BatchAttemptManifest,
) -> anyhow::Result<bool> {
    if previous.state != BatchManifestState::Committing
        || next.state != BatchManifestState::Committing
        || !same_attempt_identity(previous, next)?
    {
        return Ok(false);
    }
    for (previous, next) in previous.children.iter().zip(&next.children) {
        if previous.checksum_sha256 != next.checksum_sha256
            || previous.size_bytes != next.size_bytes
            || previous.record.file_size_bytes != next.record.file_size_bytes
            || !record_eq_ignoring_file_stat(&previous.record, &next.record)?
            || match (previous.record.file_mtime_ms, next.record.file_mtime_ms) {
                (Some(previous), Some(next)) => previous != next,
                (Some(_), None) => true,
                (None, _) => false,
            }
        {
            return Ok(false);
        }
    }
    Ok(true)
}

impl BatchJournalReplay {
    fn initial(manifest: BatchAttemptManifest) -> Self {
        Self {
            manifest_history: vec![manifest.clone()],
            manifest,
            staged_receipts: BTreeMap::new(),
            published_children: BTreeSet::new(),
            post_publish_snapshot: false,
            metadata_committed: false,
            cleanup_started: false,
        }
    }

    fn terminal_without_journal(manifest: BatchAttemptManifest) -> Self {
        debug_assert!(matches!(
            manifest.state,
            BatchManifestState::Committed | BatchManifestState::Failed
        ));
        let committed = manifest.state == BatchManifestState::Committed;
        Self {
            manifest_history: vec![manifest.clone()],
            staged_receipts: BTreeMap::new(),
            published_children: if committed {
                (0..manifest.children.len()).collect()
            } else {
                BTreeSet::new()
            },
            manifest,
            post_publish_snapshot: committed,
            metadata_committed: committed,
            // A committed manifest can survive only because recursive
            // teardown removed its journal first. Treat cleanup as already
            // started so recovery verifies finals, archives, and removes the
            // attempt without creating a journal that cannot reconstruct its
            // lost prefix.
            cleanup_started: committed,
        }
    }

    fn apply_manifest_snapshot(
        &mut self,
        output_dir: &Path,
        attempt_dir: &Path,
        next: &BatchAttemptManifest,
    ) -> anyhow::Result<()> {
        validate_loaded_manifest(output_dir, attempt_dir, next)?;
        ensure!(
            same_attempt_identity(&self.manifest, next)?,
            "batch journal changes immutable attempt identity"
        );
        ensure!(
            !self.cleanup_started,
            "batch journal contains an event after cleanup started"
        );
        let legal = match (self.manifest.state, next.state) {
            (BatchManifestState::Staging, BatchManifestState::Staging) => {
                self.manifest.version == MANIFEST_VERSION_V1
                    && is_staging_successor(&self.manifest, next)?
            }
            (BatchManifestState::Staging, BatchManifestState::Prepared) => {
                manifests_equal_except_state(&self.manifest, next)?
                    && next
                        .children
                        .iter()
                        .all(|child| child.checksum_sha256.is_some())
                    && (next.version == MANIFEST_VERSION_V1
                        || self.staged_receipts.len() == next.children.len())
            }
            (BatchManifestState::Staging, BatchManifestState::Failed)
            | (BatchManifestState::Prepared, BatchManifestState::Failed) => {
                self.published_children.is_empty()
                    && !self.metadata_committed
                    && manifests_equal_except_state(&self.manifest, next)?
            }
            (BatchManifestState::Prepared, BatchManifestState::Committing) => {
                manifests_equal_except_state(&self.manifest, next)?
            }
            (BatchManifestState::Committing, BatchManifestState::Committing) => {
                self.published_children.len() == self.manifest.children.len()
                    && !self.post_publish_snapshot
                    && !self.metadata_committed
                    && is_post_publish_snapshot(&self.manifest, next)?
            }
            (BatchManifestState::Committing, BatchManifestState::Committed) => {
                self.post_publish_snapshot
                    && self.metadata_committed
                    && manifests_equal_except_state(&self.manifest, next)?
            }
            (BatchManifestState::Committing, BatchManifestState::Failed) => {
                self.published_children.is_empty()
                    && !self.metadata_committed
                    && manifests_equal_except_state(&self.manifest, next)?
            }
            _ => false,
        };
        ensure!(
            legal,
            "illegal batch manifest transition {:?} -> {:?}",
            self.manifest.state,
            next.state
        );
        if self.manifest.state == BatchManifestState::Committing
            && next.state == BatchManifestState::Committing
        {
            self.post_publish_snapshot = true;
        }
        self.manifest = next.clone();
        self.manifest_history.push(next.clone());
        Ok(())
    }

    fn apply_event(
        &mut self,
        output_dir: &Path,
        attempt_dir: &Path,
        event: &BatchJournalEvent,
    ) -> anyhow::Result<()> {
        ensure!(
            !self.cleanup_started,
            "batch journal contains an event after cleanup started"
        );
        match event {
            BatchJournalEvent::ManifestSnapshot { manifest } => {
                self.apply_manifest_snapshot(output_dir, attempt_dir, manifest)
            }
            BatchJournalEvent::ChildStaged { receipt } => {
                ensure!(
                    self.manifest.version == MANIFEST_VERSION
                        && self.manifest.state == BatchManifestState::Staging
                        && !self.cleanup_started,
                    "ChildStaged is outside a v2 staging attempt"
                );
                ensure!(
                    receipt.parent_id == self.manifest.parent_id
                        && receipt.attempt_generation == self.manifest.attempt_generation
                        && receipt.child_index < self.manifest.children.len(),
                    "ChildStaged receipt does not belong to this attempt"
                );
                let child = &mut self.manifest.children[receipt.child_index];
                ensure!(
                    child.checksum_sha256.is_none()
                        && child.size_bytes.is_none()
                        && !self.staged_receipts.contains_key(&receipt.child_index),
                    "ChildStaged duplicates an existing receipt"
                );
                ensure!(
                    receipt.record_identity_sha256 == immutable_record_identity(child)?,
                    "ChildStaged immutable record identity changed"
                );
                ensure!(
                    receipt.checksum_sha256.len() == 64
                        && receipt
                            .checksum_sha256
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                    "ChildStaged checksum is invalid"
                );
                ensure!(
                    receipt.size_bytes <= i64::MAX as u64,
                    "ChildStaged size is too large"
                );
                child.checksum_sha256 = Some(receipt.checksum_sha256.clone());
                child.size_bytes = Some(receipt.size_bytes);
                child.record.file_size_bytes = Some(receipt.size_bytes as i64);
                self.staged_receipts
                    .insert(receipt.child_index, receipt.clone());
                Ok(())
            }
            BatchJournalEvent::ChildUnstaged {
                child_index,
                lease_generation,
                record_identity_sha256,
            } => {
                ensure!(
                    self.manifest.version == MANIFEST_VERSION
                        && self.manifest.state == BatchManifestState::Staging,
                    "ChildUnstaged is outside a v2 staging attempt"
                );
                let receipt = self
                    .staged_receipts
                    .remove(child_index)
                    .context("ChildUnstaged has no matching receipt")?;
                ensure!(
                    receipt.lease_generation == *lease_generation
                        && receipt.record_identity_sha256 == *record_identity_sha256,
                    "ChildUnstaged does not match its receipt"
                );
                let child = &mut self.manifest.children[*child_index];
                child.checksum_sha256 = None;
                child.size_bytes = None;
                child.record.file_size_bytes = None;
                child.record.file_mtime_ms = None;
                Ok(())
            }
            BatchJournalEvent::FinalPublished { child_index } => {
                ensure!(
                    self.manifest.state == BatchManifestState::Committing
                        && !self.post_publish_snapshot
                        && !self.metadata_committed,
                    "batch final publication is outside the committing publication phase"
                );
                ensure!(
                    *child_index < self.manifest.children.len(),
                    "batch journal publishes out-of-range child {child_index}"
                );
                ensure!(
                    *child_index == self.published_children.len(),
                    "batch journal publishes child {child_index} out of order or more than once"
                );
                ensure!(
                    self.published_children.insert(*child_index),
                    "batch journal publishes child {child_index} more than once"
                );
                Ok(())
            }
            BatchJournalEvent::MetadataCommitted => {
                ensure!(
                    self.manifest.state == BatchManifestState::Committing
                        && self.published_children.len() == self.manifest.children.len()
                        && self.post_publish_snapshot
                        && !self.metadata_committed,
                    "batch metadata commit is out of order or duplicated"
                );
                self.metadata_committed = true;
                Ok(())
            }
            BatchJournalEvent::CleanupStarted => {
                ensure!(
                    self.manifest.state == BatchManifestState::Committed && self.metadata_committed,
                    "batch cleanup starts before the commit is durable"
                );
                self.cleanup_started = true;
                Ok(())
            }
        }
    }
}

fn replay_batch_journal(
    output_dir: &Path,
    attempt_dir: &Path,
    records: &[BatchJournalRecord],
) -> anyhow::Result<Option<BatchJournalReplay>> {
    let mut replay: Option<BatchJournalReplay> = None;
    for record in records {
        if let Some(replay) = replay.as_mut() {
            ensure!(
                record.attempt_generation == replay.manifest.attempt_generation,
                "batch journal mixes attempt generations"
            );
            replay
                .apply_event(output_dir, attempt_dir, &record.event)
                .with_context(|| format!("replaying batch journal record {}", record.sequence))?;
            continue;
        }
        let BatchJournalEvent::ManifestSnapshot { manifest } = &record.event else {
            anyhow::bail!("batch journal does not begin with a manifest snapshot");
        };
        validate_loaded_manifest(output_dir, attempt_dir, manifest)?;
        ensure!(
            record.sequence == 0
                && record.attempt_generation == manifest.attempt_generation
                && is_initial_batch_manifest(manifest),
            "batch journal has an invalid initial manifest"
        );
        replay = Some(BatchJournalReplay::initial(manifest.clone()));
    }
    Ok(replay)
}

fn reconcile_batch_manifest(
    output_dir: &Path,
    attempt_dir: &Path,
    journal_replay: Option<BatchJournalReplay>,
    disk_manifest: Option<BatchAttemptManifest>,
) -> anyhow::Result<(BatchAttemptManifest, bool, bool, BatchJournalReplay)> {
    match (journal_replay, disk_manifest) {
        (None, None) => anyhow::bail!(
            "batch attempt has neither a readable manifest nor a recoverable journal: {}; \
             move this attempt directory out of {} after inspecting it",
            attempt_dir.display(),
            output_dir.join(TRANSACTION_DIR).display()
        ),
        (None, Some(disk)) => {
            if is_initial_batch_manifest(&disk) {
                let replay = BatchJournalReplay::initial(disk.clone());
                Ok((disk, false, true, replay))
            } else if matches!(
                disk.state,
                BatchManifestState::Committed | BatchManifestState::Failed
            ) {
                let replay = BatchJournalReplay::terminal_without_journal(disk.clone());
                Ok((disk, false, false, replay))
            } else {
                anyhow::bail!(
                    "journal-free disk manifest is neither an initial nor a terminal batch snapshot"
                )
            }
        }
        (Some(replay), None) => Ok((replay.manifest.clone(), true, false, replay)),
        (Some(replay), Some(disk)) if manifest_eq(&replay.manifest, &disk)? => {
            Ok((replay.manifest.clone(), false, false, replay))
        }
        (Some(replay), Some(disk)) if manifest_history_contains(&replay, &disk)? => {
            Ok((replay.manifest.clone(), true, false, replay))
        }
        (Some(mut replay), Some(disk)) => {
            replay
                .apply_manifest_snapshot(output_dir, attempt_dir, &disk)
                .context("disk manifest is neither journal state nor one legal snapshot ahead")?;
            Ok((disk, false, true, replay))
        }
    }
}

fn manifest_history_contains(
    replay: &BatchJournalReplay,
    disk: &BatchAttemptManifest,
) -> anyhow::Result<bool> {
    for manifest in &replay.manifest_history {
        if manifest_eq(manifest, disk)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// An error before the manifest enters `committing` is ordinary. An error
/// after that point owns the writer guard and aborts if dropped.
#[must_use = "an unresolved committing transaction must terminate or be recovered"]
pub struct UnresolvedBatchCommit {
    error: anyhow::Error,
    writer: Option<tokio::sync::OwnedRwLockWriteGuard<()>>,
    bookkeeping: Option<GalleryBookkeepingGuard>,
}

impl UnresolvedBatchCommit {
    fn pre_commit(error: anyhow::Error) -> Self {
        Self {
            error,
            writer: None,
            bookkeeping: None,
        }
    }

    fn committing(
        error: anyhow::Error,
        writer: tokio::sync::OwnedRwLockWriteGuard<()>,
        bookkeeping: Option<GalleryBookkeepingGuard>,
    ) -> Self {
        Self {
            error,
            writer: Some(writer),
            bookkeeping,
        }
    }

    pub fn entered_committing(&self) -> bool {
        self.writer.is_some()
    }

    /// Startup recovery runs before the listener binds, so it may release the
    /// barrier and return a normal boot error for an operator-recoverable
    /// filesystem/SQLite failure. Live serving code must never call this.
    pub(crate) fn into_startup_error(mut self) -> anyhow::Error {
        self.bookkeeping.take();
        self.writer.take();
        std::mem::replace(
            &mut self.error,
            anyhow::anyhow!("batch recovery error already consumed"),
        )
    }
}

impl std::fmt::Debug for UnresolvedBatchCommit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnresolvedBatchCommit")
            .field("error", &self.error)
            .field("entered_committing", &self.entered_committing())
            .finish()
    }
}

impl std::fmt::Display for UnresolvedBatchCommit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl Drop for UnresolvedBatchCommit {
    fn drop(&mut self) {
        if self.writer.is_some() {
            tracing::error!(
                error = %self.error,
                "unresolved atomic batch commit; terminating for startup recovery"
            );
            std::process::abort();
        }
    }
}

/// Recover every durable attempt before gallery routes are constructed.
pub async fn recover_transactions(
    output_dir: &Path,
    gate: &GalleryPublicationGate,
    db: Arc<Option<mold_db::MetadataDb>>,
) -> anyhow::Result<RecoveryReport> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("creating gallery {}", output_dir.display()))?;
    // Lock order is gallery bookkeeping -> nonblocking attempt claim. Recovery
    // never waits for an attempt while holding bookkeeping: a live batch can
    // already hold its attempt lock while waiting for the gallery gate, and
    // an ordinary publisher can hold that gate while waiting for bookkeeping.
    // Claimed attempts retain their exact OS authority after bookkeeping is
    // released and across every async gallery-gate/database recovery action.
    // Declare the authority collection before bookkeeping so panic unwinding
    // drops the later bookkeeping guard first. Otherwise an AttemptAuthority
    // destructor would wait forever on the lock still owned by this thread.
    let mut attempts = Vec::new();
    let bookkeeping_lock = acquire_gallery_bookkeeping_lock(output_dir)?;
    sweep_reclaimable_attempt_authorities(&bookkeeping_lock)?;
    let root = output_dir.join(TRANSACTION_DIR);
    if !root.is_dir() {
        let loaded =
            crate::gallery_authority::load_or_initialize(output_dir, &bookkeeping_lock, || {
                Ok(CommittedArchiveIndex::default())
            })?;
        gate.install_committed_archive_index(loaded.generation, loaded.index);
        return Ok(RecoveryReport::default());
    }
    sweep_stale_reservations(&root)?;
    if let Err(error) = collect_claimed_attempts(&root, &mut attempts, &bookkeeping_lock) {
        // Claimed attempts drop in attempt -> bookkeeping order. Explicitly
        // release bookkeeping before unwinding a partially collected scan.
        drop(bookkeeping_lock);
        drop(attempts);
        return Err(error);
    }
    attempts.sort_by(|left, right| left.manifest_path.cmp(&right.manifest_path));
    drop(bookkeeping_lock);

    let mut report = RecoveryReport::default();
    for claimed in attempts {
        let mut transaction =
            BatchTransaction::load_claimed(output_dir, &claimed.manifest_path, claimed.authority)?;
        if transaction.reconstructed_from_journal {
            atomic_write_json(
                &transaction.attempt_dir.join(MANIFEST_FILE),
                &transaction.manifest,
            )
            .with_context(|| {
                format!(
                    "rewriting batch manifest reconstructed from {}",
                    transaction.attempt_dir.join(JOURNAL_FILE).display()
                )
            })?;
            transaction.reconstructed_from_journal = false;
        }
        if transaction.journal_needs_heal {
            transaction
                .append_journal(BatchJournalEvent::ManifestSnapshot {
                    manifest: transaction.manifest.clone(),
                })
                .with_context(|| {
                    format!(
                        "healing batch journal from atomic manifest {}",
                        transaction.attempt_dir.join(MANIFEST_FILE).display()
                    )
                })?;
            transaction.journal_needs_heal = false;
        }
        match transaction.manifest.state {
            BatchManifestState::Staging | BatchManifestState::Prepared => {
                let _guard = gate.write().await;
                ensure!(
                    !transaction.any_final_exists(),
                    "unpublished batch attempt has a final-path artifact; refusing to serve: {}",
                    transaction.attempt_dir.display()
                );
                transaction.manifest.state = BatchManifestState::Failed;
                transaction.persist_manifest()?;
                transaction.release_reservations();
                let _ = fs::remove_dir_all(transaction.attempt_dir.join("staging"));
                remove_failed_attempt(&transaction);
                report.rolled_back += 1;
            }
            BatchManifestState::Committing => {
                if let Err(staged_error) = transaction.verify_all_staged() {
                    let any_final_exists = transaction.any_final_exists();
                    let has_durable_publication_evidence =
                        !transaction.journaled_final_children.is_empty();
                    if !any_final_exists && !has_durable_publication_evidence {
                        let _guard = gate.write().await;
                        transaction.manifest.state = BatchManifestState::Failed;
                        transaction.persist_manifest()?;
                        transaction.release_reservations();
                        remove_failed_attempt(&transaction);
                        report.rolled_back += 1;
                        tracing::warn!(
                            attempt = %transaction.attempt_dir.display(),
                            error = %staged_error,
                            "rolled back unverifiable committing attempt before any final publication"
                        );
                        continue;
                    }

                    match transaction.verify_all_committed() {
                        Ok(()) => {
                            match transaction.commit_verified_finals(gate, db.clone()).await {
                                Ok(()) => {
                                    report.rolled_forward += 1;
                                    continue;
                                }
                                Err(error) => {
                                    return Err(error.into_startup_error()).with_context(|| {
                                        format!(
                                            "rolling forward batch attempt {} from checksum-verified \
                                             finals; startup remains fail-closed",
                                            transaction.attempt_dir.display()
                                        )
                                    });
                                }
                            }
                        }
                        Err(final_error) => {
                            return Err(final_error).with_context(|| {
                                if has_durable_publication_evidence {
                                    format!(
                                        "committing attempt {} has durable publication evidence but \
                                         neither verifiable staging nor a checksum-valid complete set \
                                         of finals; retaining the attempt for operator inspection \
                                         (staging error: {staged_error:#})",
                                        transaction.attempt_dir.display()
                                    )
                                } else {
                                    format!(
                                        "committing attempt {} has published files but neither \
                                         verifiable staging nor a checksum-valid complete set of \
                                         finals; retaining the attempt for operator inspection \
                                         (staging error: {staged_error:#})",
                                        transaction.attempt_dir.display()
                                    )
                                }
                            });
                        }
                    }
                }
                match transaction.commit(gate, db.clone()).await {
                    Ok(()) => report.rolled_forward += 1,
                    Err(error) => {
                        return Err(error.into_startup_error()).with_context(|| {
                            format!(
                                "rolling forward batch attempt {}; startup remains fail-closed",
                                transaction.attempt_dir.display()
                            )
                        });
                    }
                }
            }
            BatchManifestState::Committed => {
                transaction.verify_all_committed()?;
                let _guard = gate.write().await;
                let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
                transaction.persist_committed_archive_manifest()?;
                gate.record_committed_manifest(output_dir, &transaction.manifest, &bookkeeping)?;
                drop(bookkeeping);
                if let Some(db) = db.as_ref() {
                    let records: Vec<_> = transaction
                        .manifest
                        .children
                        .iter()
                        .map(|child| child.record.clone())
                        .collect();
                    db.upsert_batch(&records)?;
                    report.healed_committed_rows += records.len();
                }
                transaction.release_reservations();
                transaction.cleanup_private_staging();
                transaction
                    .archive_committed_attempt(false)
                    .context("removing recovered committed batch attempt")?;
            }
            BatchManifestState::Failed => {
                ensure!(
                    !transaction.any_final_exists(),
                    "failed batch attempt has a final-path artifact; refusing to remove durable evidence: {}",
                    transaction.attempt_dir.display()
                );
                transaction.release_reservations();
                transaction.cleanup_private_staging();
                remove_failed_attempt(&transaction);
            }
        }
    }
    let archive_index = reconcile_committed_archive_index(output_dir)
        .context("validating committed gallery archive index during startup recovery")?;
    let retired_names = archive_index
        .retired_names
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    if let Some(db) = db.as_ref() {
        let records = archive_index.records().cloned().collect::<Vec<_>>();
        if !records.is_empty() {
            db.upsert_batch(&records)
                .context("healing metadata DB from committed gallery archive")?;
        }
        let canonical_output_dir = fs::canonicalize(output_dir)
            .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
        for filename in &retired_names {
            db.delete(&canonical_output_dir, filename)
                .with_context(|| format!("projecting retired gallery row {filename}"))?;
        }
    }
    let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
    let loaded = crate::gallery_authority::load_or_initialize(output_dir, &bookkeeping, || {
        Ok(archive_index)
    })?;
    gate.install_committed_archive_index(loaded.generation, loaded.index);
    drop(bookkeeping);
    gate.acknowledge_retirement_projections(output_dir, retired_names)?;
    Ok(report)
}

fn collect_claimed_attempts(
    root: &Path,
    out: &mut Vec<ClaimedAttempt>,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    // Only generation-scoped active attempts participate in startup
    // recovery. Archived committed manifests and reservation metadata are
    // durable authority, not work to replay on every boot.
    for parent in fs::read_dir(root)? {
        let parent = parent?;
        if !parent.path().is_dir() || parent.file_name() == "reservations" {
            continue;
        }
        // Version-2 parent attempts are recovered only by the joint
        // parent/transaction authority. Generic v1 recovery must never roll
        // back a receipted child before that reducer is reconciled.
        if parent.path().join(PARENT_AUTHORITY_DIR).is_dir() {
            continue;
        }
        let attempts = parent.path().join("attempts");
        if !attempts.is_dir() {
            continue;
        }
        for attempt in fs::read_dir(&attempts)? {
            let attempt = attempt?;
            if !attempt.path().is_dir() {
                continue;
            }
            let generation_name = attempt.file_name();
            let generation_name = generation_name.to_str().with_context(|| {
                format!(
                    "unrecognized non-UTF-8 entry in batch attempts directory {}; \
                     move it outside {TRANSACTION_DIR} before restarting",
                    attempt.path().display()
                )
            })?;
            let generation = generation_name.parse::<u64>().with_context(|| {
                format!(
                    "unrecognized batch attempt directory {}; expected a canonical numeric \
                     generation and retained it for operator inspection; move it outside \
                     {TRANSACTION_DIR} before restarting",
                    attempt.path().display()
                )
            })?;
            ensure!(
                generation.to_string() == generation_name,
                "non-canonical batch attempt directory {} was retained for operator inspection; \
                 move it outside {TRANSACTION_DIR} before restarting",
                attempt.path().display()
            );
            let mut authority = match try_claim_attempt_authority(&attempt.path(), bookkeeping) {
                Ok(Some(authority)) => authority,
                Ok(None) => {
                    tracing::info!(
                        attempt = %attempt.path().display(),
                        "skipping batch recovery for an attempt owned by a live process"
                    );
                    continue;
                }
                Err(error)
                    if error
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
                {
                    // A live owner may archive its terminal attempt between
                    // read_dir and our nonblocking claim.
                    continue;
                }
                Err(error) => return Err(error),
            };
            if !attempt.path().is_dir() {
                // The owner released and unlinked the inode after we opened
                // it but before the claim completed.
                authority.reclaim_under_bookkeeping(bookkeeping);
                continue;
            }
            let manifest = attempt.path().join(MANIFEST_FILE);
            let journal = attempt.path().join(JOURNAL_FILE);
            if manifest.is_file() || journal.is_file() {
                out.push(ClaimedAttempt {
                    manifest_path: manifest,
                    authority,
                });
            } else {
                let owner = ReservationOwner {
                    parent_id: parent.file_name().to_string_lossy().into_owned(),
                    attempt_generation: generation,
                };
                tracing::warn!(
                    attempt = %attempt.path().display(),
                    "removing orphaned batch attempt directory with no manifest or journal"
                );
                if let Err(error) = (|| {
                    release_all_reservations_for_owner(root, &owner)?;
                    fs::remove_dir_all(attempt.path())?;
                    Ok::<_, anyhow::Error>(())
                })() {
                    authority.reclaim_under_bookkeeping(bookkeeping);
                    return Err(error);
                }
                authority.reclaim_under_bookkeeping(bookkeeping);
            }
        }
    }
    Ok(())
}

fn sweep_stale_reservations(transaction_root: &Path) -> anyhow::Result<()> {
    let reservations = transaction_root.join("reservations");
    if !reservations.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&reservations)? {
        let entry = entry?;
        if !entry.path().is_file() {
            continue;
        }
        let owner = fs::read(entry.path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ReservationOwner>(&bytes).ok());
        if owner.is_none() {
            tracing::warn!(
                reservation = %entry.path().display(),
                "retaining unreadable batch reservation for fail-closed collision safety"
            );
            continue;
        }
        let stale = owner.as_ref().is_some_and(|owner| {
            owner.parent_id.starts_with("ordinary:")
                || !transaction_root
                    .join(&owner.parent_id)
                    .join("attempts")
                    .join(owner.attempt_generation.to_string())
                    .is_dir()
        });
        if stale {
            fs::remove_file(entry.path())?;
        }
    }
    sync_dir(&reservations)
}

fn release_all_reservations_for_owner(
    transaction_root: &Path,
    owner: &ReservationOwner,
) -> anyhow::Result<()> {
    let reservations = transaction_root.join("reservations");
    if !reservations.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&reservations)? {
        let entry = entry?;
        if !entry.path().is_file() {
            continue;
        }
        let bytes = match fs::read(entry.path()) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(
                    reservation = %entry.path().display(),
                    %error,
                    "could not inspect orphaned batch reservation"
                );
                continue;
            }
        };
        if serde_json::from_slice::<ReservationOwner>(&bytes)
            .ok()
            .as_ref()
            == Some(owner)
        {
            fs::remove_file(entry.path())?;
        }
    }
    sync_dir(&reservations)
}

fn remove_failed_attempt(transaction: &BatchTransaction) {
    if let Err(error) = fs::remove_dir_all(&transaction.attempt_dir) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(
                attempt = %transaction.attempt_dir.display(),
                %error,
                "failed to remove terminal batch attempt"
            );
        }
    }
    if let Some(attempts_dir) = transaction.attempt_dir.parent() {
        let _ = sync_dir(attempts_dir);
    }
}

fn attempt_dir(output_dir: &Path, parent_id: &str, generation: u64) -> PathBuf {
    output_dir
        .join(TRANSACTION_DIR)
        .join(parent_id)
        .join("attempts")
        .join(generation.to_string())
}

fn reservations_dir(output_dir: &Path) -> PathBuf {
    output_dir.join(TRANSACTION_DIR).join("reservations")
}

/// Reclaim unlocked v2 authority files while gallery bookkeeping excludes
/// every v2 opener. Every path is derived from the guard's canonical gallery,
/// so aliases and Windows verbatim paths cannot split sweep provenance.
///
/// Legacy v1 authority files are deliberately never unlinked. A v1 process
/// can open one before taking its OS lock and does not own v2 bookkeeping; a
/// v2 unlink in that window would let v1 lock a detached inode while a later
/// v2 claimant creates a different one. Stable empty legacy files are harmless
/// and reusable. Suspicious legacy namespaces and exact-shaped entries are
/// warn-retained without aborting unrelated recovery; an exact claim still
/// validates and fails closed.
fn sweep_reclaimable_attempt_authorities(
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    let output_dir = &bookkeeping.canonical_output_dir;
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !is_attempt_authority_file_name(name) {
            continue;
        }
        reclaim_unlocked_authority_path(&entry.path(), bookkeeping)?;
    }

    let legacy = output_dir
        .join(TRANSACTION_DIR)
        .join(LEGACY_ATTEMPT_LOCKS_DIR);
    let metadata = match fs::symlink_metadata(&legacy) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            tracing::warn!(
                authority_namespace = %legacy.display(),
                %error,
                "retaining unverifiable legacy batch authority namespace during sweep"
            );
            return Ok(());
        }
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        tracing::warn!(
            authority_namespace = %legacy.display(),
            "retaining suspicious legacy batch authority namespace during sweep"
        );
        return Ok(());
    }
    let entries = match fs::read_dir(&legacy) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                authority_namespace = %legacy.display(),
                %error,
                "retaining unreadable legacy batch authority namespace during sweep"
            );
            return Ok(());
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(
                    authority_namespace = %legacy.display(),
                    %error,
                    "retaining unreadable legacy batch authority entry during sweep"
                );
                continue;
            }
        };
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !is_legacy_attempt_authority_file_name(&name) {
            continue;
        }
        match fs::symlink_metadata(entry.path()) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                tracing::warn!(
                    authority = %entry.path().display(),
                    "retaining suspicious legacy batch authority entry during sweep"
                );
            }
            Err(error) => {
                tracing::warn!(
                    authority = %entry.path().display(),
                    %error,
                    "retaining unverifiable legacy batch authority entry during sweep"
                );
            }
        }
    }
    Ok(())
}

fn is_attempt_authority_file_name(name: &str) -> bool {
    let Some(hash) = name
        .strip_prefix(ATTEMPT_LOCK_PREFIX)
        .and_then(|name| name.strip_suffix(ATTEMPT_LOCK_SUFFIX))
    else {
        return false;
    };
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_legacy_attempt_authority_file_name(name: &str) -> bool {
    let Some(hash) = name.strip_suffix(ATTEMPT_LOCK_SUFFIX) else {
        return false;
    };
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn reclaim_unlocked_authority_path(
    lock_path: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    let lock = match open_attempt_authority_file(lock_path).with_context(|| {
        format!(
            "opening reclaimable batch attempt authority {}",
            lock_path.display()
        )
    }) {
        Ok(lock) => lock,
        Err(error) if is_attempt_authority_open_contention(&error) => return Ok(()),
        Err(error) => {
            tracing::warn!(
                authority = %lock_path.display(),
                %error,
                "retaining suspicious stale batch attempt authority during sweep"
            );
            return Ok(());
        }
    };
    match fs2::FileExt::try_lock_exclusive(&lock) {
        Ok(()) => {
            if let Err(error) = verify_authority_file_at_path(lock_path, &lock, bookkeeping) {
                tracing::warn!(
                    authority = %lock_path.display(),
                    %error,
                    "retaining unverifiable stale batch attempt authority during sweep"
                );
                let _ = fs2::FileExt::unlock(&lock);
                return Ok(());
            }
            probe_attempt_authority_lock_semantics(lock_path, bookkeeping)?;
            remove_locked_authority_path(lock_path, &lock, bookkeeping)?;
            if let Some(parent) = lock_path.parent() {
                sync_attempt_authority_parent(parent)?;
            }
            let _ = fs2::FileExt::unlock(&lock);
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "probing reclaimable batch attempt authority {}",
                    lock_path.display()
                )
            });
        }
    }
    Ok(())
}

fn cleanup_empty_transaction_hierarchy(output_dir: &Path, attempt_dir: Option<&Path>) {
    if let Some(attempt_dir) = attempt_dir {
        if let Some(attempts) = attempt_dir.parent() {
            let _ = fs::remove_dir(attempts);
            if let Some(parent) = attempts.parent() {
                let _ = fs::remove_dir(parent);
            }
        }
    }
    let root = output_dir.join(TRANSACTION_DIR);
    let reservations = root.join("reservations");
    let legacy = root.join(LEGACY_ATTEMPT_LOCKS_DIR);
    let _ = fs::remove_dir(&reservations);
    let _ = fs::remove_dir(&legacy);
    let _ = fs::remove_dir(&root);
    let _ = sync_dir(output_dir);
}

fn try_claim_attempt_authority(
    attempt_dir: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<Option<AttemptAuthority>> {
    try_claim_attempt_authority_with_post_lock_hook(attempt_dir, bookkeeping, |_| {})
}

fn try_claim_parent_authority(
    parent_id: &str,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<Option<ParentAuthority>> {
    validate_component(parent_id, "batch parent id")?;
    let lock_path = parent_authority_lock_path(&bookkeeping.canonical_output_dir, parent_id)?;
    let Some(authority) = try_claim_authority_file(
        &lock_path,
        &bookkeeping
            .canonical_output_dir
            .join(TRANSACTION_DIR)
            .join(parent_id),
        bookkeeping,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(ParentAuthority {
        file: authority.file,
        path: authority.path,
        canonical_output_dir: bookkeeping.canonical_output_dir.clone(),
        parent_id: parent_id.to_owned(),
    }))
}

pub(crate) fn claim_parent_authority(
    output_dir: &Path,
    parent_id: &str,
) -> anyhow::Result<ParentAuthority> {
    fs::create_dir_all(output_dir)?;
    let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
    try_claim_parent_authority(parent_id, &bookkeeping)?.ok_or_else(|| {
        BatchParentAuthorityContended {
            parent_id: parent_id.to_owned(),
        }
        .into()
    })
}

pub(crate) fn claim_parent_and_attempt_authorities(
    output_dir: &Path,
    parent_id: &str,
    generations: &[u64],
) -> anyhow::Result<(ParentAuthority, BTreeMap<u64, AttemptAuthority>)> {
    fs::create_dir_all(output_dir)?;
    let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
    let parent = try_claim_parent_authority(parent_id, &bookkeeping)?.ok_or_else(|| {
        BatchParentAuthorityContended {
            parent_id: parent_id.to_owned(),
        }
    })?;
    let mut attempts = BTreeMap::new();
    for generation in generations.iter().copied() {
        let directory = attempt_dir(output_dir, parent_id, generation);
        let authority =
            try_claim_attempt_authority(&directory, &bookkeeping)?.ok_or_else(|| {
                BatchAttemptAuthorityContended {
                    parent_id: parent_id.to_owned(),
                    attempt_generation: generation,
                }
            })?;
        attempts.insert(generation, authority);
    }
    drop(bookkeeping);
    parent.validate_identity(output_dir, parent_id)?;
    Ok((parent, attempts))
}

fn parent_authority_lock_path(output_dir: &Path, parent_id: &str) -> anyhow::Result<PathBuf> {
    validate_component(parent_id, "batch parent id")?;
    let canonical_output = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let mut identity = Sha256::new();
    identity.update(b"mold.batch-parent-authority.v1\0");
    identity.update((parent_id.len() as u64).to_le_bytes());
    identity.update(parent_id.as_bytes());
    Ok(canonical_output.join(format!(
        "{PARENT_LOCK_PREFIX}{:x}{ATTEMPT_LOCK_SUFFIX}",
        identity.finalize()
    )))
}

fn try_claim_attempt_authority_with_post_lock_hook(
    attempt_dir: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
    after_lock: impl FnOnce(&Path),
) -> anyhow::Result<Option<AttemptAuthority>> {
    // Rolling overlap must fence both deployed protocols. Acquire the v1
    // authority first because a predecessor knows only that path; every v2
    // process uses this same legacy -> current order, and both acquisitions
    // remain nonblocking while gallery bookkeeping is held.
    let legacy_path = legacy_attempt_authority_lock_path(attempt_dir, bookkeeping)?;
    let current_path = attempt_authority_lock_path(attempt_dir)?;
    let output_dir = current_path
        .parent()
        .context("batch attempt authority has no gallery parent")?
        .to_path_buf();

    let Some(mut legacy) = try_claim_authority_file(&legacy_path, attempt_dir, bookkeeping)? else {
        return Ok(None);
    };
    let current = match try_claim_authority_file(&current_path, attempt_dir, bookkeeping) {
        Ok(Some(current)) => current,
        Ok(None) => {
            reclaim_partial_authority(&mut legacy, bookkeeping);
            return Ok(None);
        }
        Err(error) => {
            reclaim_partial_authority(&mut legacy, bookkeeping);
            return Err(error);
        }
    };

    after_lock(&current_path);
    if let Err(error) =
        verify_authority_file_at_path(&legacy.path, legacy.file.as_ref().unwrap(), bookkeeping)
            .and_then(|_| {
                verify_authority_file_at_path(
                    &current.path,
                    current.file.as_ref().unwrap(),
                    bookkeeping,
                )
            })
    {
        let mut current = current;
        unlock_and_close_authority(&mut current);
        unlock_and_close_authority(&mut legacy);
        return Err(error);
    }

    Ok(Some(AttemptAuthority::Held {
        legacy,
        current,
        output_dir,
        attempt_dir: attempt_dir.to_path_buf(),
        reclaim_on_drop: true,
    }))
}

fn try_claim_authority_file(
    lock_path: &Path,
    attempt_dir: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<Option<LockedAuthorityFile>> {
    let authority = match open_authority_file_at_path(lock_path, attempt_dir, bookkeeping) {
        Ok(authority) => authority,
        Err(error) if is_attempt_authority_open_contention(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    match fs2::FileExt::try_lock_exclusive(&authority) {
        Ok(()) => {
            if let Err(error) = verify_authority_file_at_path(lock_path, &authority, bookkeeping) {
                let _ = fs2::FileExt::unlock(&authority);
                return Err(error);
            }
            if let Err(error) = probe_attempt_authority_lock_semantics(lock_path, bookkeeping) {
                let _ = fs2::FileExt::unlock(&authority);
                return Err(error);
            }
            Ok(Some(LockedAuthorityFile {
                file: Some(authority),
                path: lock_path.to_path_buf(),
            }))
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(error) => Err(error).with_context(|| {
            format!(
                "claiming batch attempt recovery authority {}",
                attempt_dir.display()
            )
        }),
    }
}

fn probe_attempt_authority_lock_semantics(
    lock_path: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    let probe = open_attempt_authority_probe_file(lock_path)?;
    verify_authority_file_at_path(lock_path, &probe, bookkeeping)?;
    match fs2::FileExt::try_lock_exclusive(&probe) {
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Ok(()) => {
            let _ = fs2::FileExt::unlock(&probe);
            bail!(
                "gallery filesystem does not provide exclusive attempt locks across two \
                 descriptors; durable batch publication is unsupported for {}",
                lock_path.display()
            )
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "probing batch attempt lock semantics for {}",
                lock_path.display()
            )
        }),
    }
}

fn reclaim_partial_authority(
    authority: &mut LockedAuthorityFile,
    _bookkeeping: &GalleryBookkeepingGuard,
) {
    // This is always the partially acquired legacy authority. Preserve its
    // pathname as the one reusable inode visible to rolling v1 processes.
    unlock_and_close_authority(authority);
}

#[cfg(windows)]
fn is_attempt_authority_open_contention(error: &anyhow::Error) -> bool {
    // ERROR_SHARING_VIOLATION: the live owner deliberately omits delete
    // sharing, so Windows can reject a contender before fs2 reaches its lock.
    // This is the same typed live-authority contention, not an I/O failure.
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .and_then(std::io::Error::raw_os_error)
            == Some(32)
    })
}

#[cfg(not(windows))]
fn is_attempt_authority_open_contention(_error: &anyhow::Error) -> bool {
    false
}

#[cfg(test)]
fn open_attempt_authority_target_with_hook(
    attempt_dir: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
    after_open: impl FnOnce(&Path),
) -> anyhow::Result<File> {
    let lock_path = attempt_authority_lock_path(attempt_dir)?;
    let lock = open_authority_file_at_path(&lock_path, attempt_dir, bookkeeping)?;
    after_open(&lock_path);
    verify_authority_file_at_path(&lock_path, &lock, bookkeeping)?;
    Ok(lock)
}

fn open_authority_file_at_path(
    lock_path: &Path,
    attempt_dir: &Path,
    _bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<File> {
    let authority_parent = lock_path
        .parent()
        .context("batch attempt authority has no gallery parent")?;
    let lock = open_attempt_authority_file(lock_path).with_context(|| {
        format!(
            "opening batch attempt authority {} for {}",
            lock_path.display(),
            attempt_dir.display()
        )
    })?;
    // The sidecar is retained only while its OS lock can be live. Its Drop
    // path unlinks it under gallery bookkeeping while the lock is still held,
    // so pathname reuse can never fork authority.
    lock.sync_all()?;
    sync_attempt_authority_parent(authority_parent)?;
    Ok(lock)
}

#[cfg(unix)]
fn open_attempt_authority_probe_file(lock_path: &Path) -> anyhow::Result<File> {
    open_attempt_authority_file(lock_path)
}

#[cfg(windows)]
fn open_attempt_authority_probe_file(lock_path: &Path) -> anyhow::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        // The owner requests DELETE access, so this verifier must share it.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .open(lock_path)
        .with_context(|| {
            format!(
                "opening batch attempt authority lock-semantics probe {}",
                lock_path.display()
            )
        })
}

#[cfg(not(any(unix, windows)))]
fn open_attempt_authority_probe_file(lock_path: &Path) -> anyhow::Result<File> {
    Ok(OpenOptions::new().read(true).write(true).open(lock_path)?)
}

#[cfg(unix)]
fn open_attempt_authority_file(lock_path: &Path) -> anyhow::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let locks = lock_path
        .parent()
        .context("batch attempt authority has no lock directory")?;
    let lock_name = lock_path
        .file_name()
        .context("batch attempt authority has no filename")?;
    let lock_name = CString::new(lock_name.as_bytes())
        .context("batch attempt authority filename contains NUL")?;
    let locks = open_unix_directory_without_symlinks(locks)?;
    // SAFETY: `locks` owns a valid directory descriptor, `lock_name` is
    // NUL-terminated, and a nonnegative result is transferred exactly once
    // into `File`.
    let fd = unsafe {
        libc::openat(
            locks.as_raw_fd(),
            lock_name.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "opening no-follow batch attempt authority {}",
                lock_path.display()
            )
        });
    }
    // SAFETY: `fd` is a fresh owned descriptor returned by `openat`.
    let file = unsafe { File::from_raw_fd(fd) };
    verify_unix_authority_entry(&locks, &lock_name, &file, lock_path)?;
    Ok(file)
}

#[cfg(unix)]
fn open_unix_directory_without_symlinks(path: &Path) -> anyhow::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;

    ensure!(
        path.is_absolute(),
        "batch attempt authority directory is not absolute: {}",
        path.display()
    );
    let mut current = File::open("/")?;
    for component in path.components() {
        let Component::Normal(name) = component else {
            ensure!(
                matches!(component, Component::RootDir),
                "invalid component in batch attempt authority directory: {}",
                path.display()
            );
            continue;
        };
        let name =
            CString::new(name.as_bytes()).context("batch authority directory contains NUL")?;
        // SAFETY: `current` owns a valid directory descriptor and `name` is
        // NUL-terminated. Every successful descriptor is transferred once.
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| {
                format!(
                    "opening no-follow batch authority directory {}",
                    path.display()
                )
            });
        }
        // SAFETY: `fd` is a fresh owned descriptor returned by `openat`.
        current = unsafe { File::from_raw_fd(fd) };
    }
    Ok(current)
}

#[cfg(unix)]
fn verify_unix_authority_entry(
    locks: &File,
    lock_name: &std::ffi::CStr,
    file: &File,
    lock_path: &Path,
) -> anyhow::Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    let mut path_stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `locks` and `lock_name` remain valid for the call, and
    // `path_stat` points to correctly sized writable storage.
    let result = unsafe {
        libc::fstatat(
            locks.as_raw_fd(),
            lock_name.as_ptr(),
            path_stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "verifying batch attempt authority entry {}",
                lock_path.display()
            )
        });
    }
    // SAFETY: successful `fstatat` initialized the complete `stat`.
    let path_stat = unsafe { path_stat.assume_init() };
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file()
            && metadata.nlink() == 1
            && (path_stat.st_mode & libc::S_IFMT) == libc::S_IFREG
            && u128::from(metadata.dev()) == path_stat.st_dev as u128
            && u128::from(metadata.ino()) == path_stat.st_ino as u128,
        "batch attempt authority was replaced or is not a private regular file: {}",
        lock_path.display()
    );
    Ok(())
}

#[cfg(unix)]
fn verify_authority_file_at_path(
    lock_path: &Path,
    file: &File,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    ensure!(
        lock_path.starts_with(&bookkeeping.canonical_output_dir),
        "batch attempt authority is outside the locked gallery: {}",
        lock_path.display()
    );
    let parent = lock_path
        .parent()
        .context("batch attempt authority has no gallery parent")?;
    let name = lock_path
        .file_name()
        .context("batch attempt authority has no filename")?;
    let name =
        CString::new(name.as_bytes()).context("batch attempt authority filename contains NUL")?;
    let parent = open_unix_directory_without_symlinks(parent)?;
    verify_unix_authority_entry(&parent, &name, file, lock_path)
}

#[cfg(unix)]
fn remove_locked_authority_path(
    lock_path: &Path,
    file: &File,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    use std::ffi::CString;
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;

    let parent_path = lock_path
        .parent()
        .context("batch attempt authority has no parent for unlink")?;
    let name = lock_path
        .file_name()
        .context("batch attempt authority has no filename for unlink")?;
    let name =
        CString::new(name.as_bytes()).context("batch attempt authority filename contains NUL")?;
    let parent = open_unix_directory_without_symlinks(parent_path)?;
    ensure!(
        lock_path.starts_with(&bookkeeping.canonical_output_dir),
        "batch attempt authority is outside the locked gallery: {}",
        lock_path.display()
    );
    verify_unix_authority_entry(&parent, &name, file, lock_path)?;
    // SAFETY: `parent` is a no-follow descriptor for the verified authority
    // directory and `name` is NUL-terminated. The target identity was checked
    // against the still-locked descriptor immediately before this call.
    let result = unsafe { libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), 0) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "unlinking verified batch attempt authority {}",
                lock_path.display()
            )
        })
    }
}

#[cfg(unix)]
fn sync_attempt_authority_parent(locks: &Path) -> anyhow::Result<()> {
    open_unix_directory_without_symlinks(locks)?
        .sync_all()
        .with_context(|| {
            format!(
                "syncing no-follow batch authority directory {}",
                locks.display()
            )
        })
}

#[cfg(windows)]
fn open_attempt_authority_file(lock_path: &Path) -> anyhow::Result<File> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::{
        DELETE, FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_OPEN_REPARSE_POINT, FILE_GENERIC_READ,
        FILE_GENERIC_WRITE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .access_mode(FILE_GENERIC_READ | FILE_GENERIC_WRITE | DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        // Excluding FILE_SHARE_DELETE makes the authority inode
        // non-replaceable for the lifetime of this handle.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(lock_path)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0,
        "batch attempt authority is not a regular file: {}",
        lock_path.display()
    );
    verify_windows_authority_entry(lock_path, &file)?;
    Ok(file)
}

#[cfg(windows)]
fn canonical_windows_authority_path(lock_path: &Path) -> anyhow::Result<PathBuf> {
    let canonical = fs::canonicalize(lock_path)?;
    let parent = lock_path
        .parent()
        .context("Windows batch authority has no parent")?;
    let canonical_parent = fs::canonicalize(parent)?;
    ensure!(
        canonical.parent() == Some(canonical_parent.as_path()),
        "batch attempt authority escaped its canonical lock directory: {}",
        lock_path.display()
    );
    Ok(canonical)
}

#[cfg(windows)]
fn verify_windows_authority_entry(lock_path: &Path, file: &File) -> anyhow::Result<PathBuf> {
    use std::os::windows::fs::MetadataExt;
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE,
        FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    // Compare canonical parent provenance rather than raw path spelling.
    // Windows canonicalization commonly returns a verbatim `\\?\` path even
    // when the caller used an ordinary path; raw equality would reject the
    // same file and make stale sidecars immortal.
    let canonical = canonical_windows_authority_path(lock_path)?;
    let path_metadata = fs::symlink_metadata(lock_path)?;
    let file_metadata = file.metadata()?;
    let path_file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .open(lock_path)?;
    let path_identity = windows_file_identity(&path_file)?;
    let file_identity = windows_file_identity(file)?;
    ensure!(
        path_metadata.is_file()
            && !path_metadata.file_type().is_symlink()
            && path_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0
            && file_metadata.is_file()
            && file_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0
            && path_identity == file_identity
            && path_identity.2 == 1,
        "batch authority was replaced, linked, or is not a private regular file: {}",
        lock_path.display()
    );
    Ok(canonical)
}

#[cfg(windows)]
fn windows_file_identity(file: &File) -> anyhow::Result<(u32, u64, u32)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
    // SAFETY: the file owns a valid Windows handle and `info` points to
    // correctly sized writable storage for the duration of the call.
    let result =
        unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, info.as_mut_ptr()) };
    if result == 0 {
        return Err(std::io::Error::last_os_error())
            .context("reading stable Windows batch authority file identity");
    }
    // SAFETY: a successful call initialized every field.
    let info = unsafe { info.assume_init() };
    Ok((
        info.dwVolumeSerialNumber,
        (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        info.nNumberOfLinks,
    ))
}

#[cfg(windows)]
fn verify_authority_file_at_path(
    lock_path: &Path,
    file: &File,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    let canonical_lock_path = verify_windows_authority_entry(lock_path, file)?;
    let canonical_gallery = fs::canonicalize(&bookkeeping.canonical_output_dir)?;
    ensure!(
        canonical_lock_path.starts_with(&canonical_gallery)
            && canonical_lock_path != canonical_gallery,
        "batch attempt authority is outside the locked gallery: {}",
        lock_path.display()
    );
    Ok(())
}

#[cfg(windows)]
fn remove_locked_authority_path(
    _lock_path: &Path,
    file: &File,
    _bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileDispositionInfo, SetFileInformationByHandle, FILE_DISPOSITION_INFO,
    };

    let disposition = FILE_DISPOSITION_INFO { DeleteFile: 1 };
    // SAFETY: the file was opened with DELETE access, the handle remains
    // valid, and `disposition` has the exact structure and size required by
    // FileDispositionInfo.
    let result = unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle() as _,
            FileDispositionInfo,
            (&raw const disposition).cast(),
            std::mem::size_of::<FILE_DISPOSITION_INFO>() as u32,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error()).context("marking batch attempt authority for deletion")
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn sync_attempt_authority_parent(_authority_parent: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn open_attempt_authority_file(lock_path: &Path) -> anyhow::Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "batch attempt authority is not a regular file: {}",
        lock_path.display()
    );
    Ok(file)
}

#[cfg(not(any(unix, windows)))]
fn verify_authority_file_at_path(
    lock_path: &Path,
    file: &File,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    ensure!(
        lock_path.starts_with(&bookkeeping.canonical_output_dir),
        "batch attempt authority is outside the locked gallery: {}",
        lock_path.display()
    );
    let canonical = fs::canonicalize(lock_path)?;
    let path_metadata = fs::symlink_metadata(lock_path)?;
    let file_metadata = file.metadata()?;
    ensure!(
        canonical == lock_path
            && path_metadata.is_file()
            && !path_metadata.file_type().is_symlink()
            && file_metadata.is_file(),
        "batch attempt authority was replaced or is not a regular file: {}",
        lock_path.display()
    );
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn remove_locked_authority_path(
    lock_path: &Path,
    _file: &File,
    _bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<()> {
    fs::remove_file(lock_path)?;
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn sync_attempt_authority_parent(locks: &Path) -> anyhow::Result<()> {
    sync_dir(locks)
}

fn attempt_authority_lock_path(attempt_dir: &Path) -> anyhow::Result<PathBuf> {
    let generation_name = attempt_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("batch attempt generation is not UTF-8")?;
    let generation: u64 = generation_name
        .parse()
        .with_context(|| format!("invalid batch attempt generation {generation_name}"))?;
    ensure!(
        generation.to_string() == generation_name,
        "batch attempt generation is not canonical: {generation_name}"
    );
    let attempts_root = attempt_dir
        .parent()
        .context("batch attempt directory has no attempts root")?;
    ensure!(
        attempts_root.file_name().and_then(|name| name.to_str()) == Some("attempts"),
        "batch attempt path is not below an attempts directory: {}",
        attempt_dir.display()
    );
    let parent_root = attempts_root
        .parent()
        .context("batch attempt directory has no parent root")?;
    let parent_id = parent_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("batch attempt parent id is not UTF-8")?;
    validate_component(parent_id, "batch attempt parent id")?;
    let transaction_root = parent_root
        .parent()
        .context("batch attempt parent has no transaction root")?;
    ensure!(
        transaction_root.file_name().and_then(|name| name.to_str()) == Some(TRANSACTION_DIR),
        "batch attempt path is not below {TRANSACTION_DIR}: {}",
        attempt_dir.display()
    );
    let output_dir = transaction_root
        .parent()
        .context("batch transaction root has no gallery parent")?;
    let canonical_output = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let canonical_root = fs::canonicalize(transaction_root).with_context(|| {
        format!(
            "canonicalizing batch transaction root {}",
            transaction_root.display()
        )
    })?;
    ensure!(
        canonical_root.parent() == Some(canonical_output.as_path())
            && canonical_root.file_name().and_then(|name| name.to_str()) == Some(TRANSACTION_DIR),
        "batch transaction root escapes its gallery: {}",
        transaction_root.display()
    );
    let canonical_parent = fs::canonicalize(parent_root).with_context(|| {
        format!(
            "canonicalizing batch attempt parent {}",
            parent_root.display()
        )
    })?;
    ensure!(
        canonical_parent.parent() == Some(canonical_root.as_path())
            && canonical_parent.file_name().and_then(|name| name.to_str()) == Some(parent_id),
        "batch attempt parent escapes transaction root: {}",
        parent_root.display()
    );
    let canonical_attempts = fs::canonicalize(attempts_root).with_context(|| {
        format!(
            "canonicalizing batch attempts directory {}",
            attempts_root.display()
        )
    })?;
    ensure!(
        canonical_attempts.parent() == Some(canonical_parent.as_path())
            && canonical_attempts
                .file_name()
                .and_then(|name| name.to_str())
                == Some("attempts"),
        "batch attempts directory escapes its parent: {}",
        attempts_root.display()
    );
    let canonical_attempt = fs::canonicalize(attempt_dir)
        .with_context(|| format!("canonicalizing batch attempt {}", attempt_dir.display()))?;
    ensure!(
        canonical_attempt.parent() == Some(canonical_attempts.as_path())
            && canonical_attempt.file_name().and_then(|name| name.to_str())
                == Some(generation_name),
        "batch attempt directory escapes its attempts root: {}",
        attempt_dir.display()
    );

    let mut identity = Sha256::new();
    identity.update(b"mold.batch-attempt-authority.v2\0");
    identity.update((parent_id.len() as u64).to_le_bytes());
    identity.update(parent_id.as_bytes());
    identity.update(generation.to_le_bytes());
    Ok(canonical_output.join(format!(
        "{ATTEMPT_LOCK_PREFIX}{:x}{ATTEMPT_LOCK_SUFFIX}",
        identity.finalize()
    )))
}

fn legacy_attempt_authority_lock_path(
    attempt_dir: &Path,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<PathBuf> {
    let current = attempt_authority_lock_path(attempt_dir)?;
    ensure!(
        current.parent() == Some(bookkeeping.canonical_output_dir.as_path()),
        "batch attempt authority does not belong to the locked gallery: {}",
        attempt_dir.display()
    );

    let generation_name = attempt_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("batch attempt generation is not UTF-8")?;
    let generation = generation_name
        .parse::<u64>()
        .context("batch attempt generation is invalid")?;
    let parent_root = attempt_dir
        .parent()
        .and_then(Path::parent)
        .context("batch attempt directory has no parent root")?;
    let parent_id = parent_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("batch attempt parent id is not UTF-8")?;
    let transaction_root = fs::canonicalize(
        parent_root
            .parent()
            .context("batch attempt parent has no transaction root")?,
    )?;
    let locks = transaction_root.join(LEGACY_ATTEMPT_LOCKS_DIR);
    if let Ok(metadata) = fs::symlink_metadata(&locks) {
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "legacy batch attempt authority namespace is not a private directory: {}",
            locks.display()
        );
    }
    fs::create_dir_all(&locks)?;
    let canonical_locks = fs::canonicalize(&locks)?;
    ensure!(
        canonical_locks.parent() == Some(transaction_root.as_path())
            && canonical_locks.file_name().and_then(|name| name.to_str())
                == Some(LEGACY_ATTEMPT_LOCKS_DIR),
        "legacy batch attempt authority namespace escaped its transaction root: {}",
        locks.display()
    );

    let mut identity = Sha256::new();
    identity.update(b"mold.batch-attempt-authority.v1\0");
    identity.update((parent_id.len() as u64).to_le_bytes());
    identity.update(parent_id.as_bytes());
    identity.update(generation.to_le_bytes());
    Ok(canonical_locks.join(format!("{:x}{ATTEMPT_LOCK_SUFFIX}", identity.finalize())))
}

pub(crate) fn acquire_gallery_bookkeeping_lock(
    output_dir: &Path,
) -> anyhow::Result<GalleryBookkeepingGuard> {
    let canonical_output_dir = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let lock = open_gallery_bookkeeping_lock_target(&canonical_output_dir)?;
    verify_gallery_bookkeeping_lock_target(&canonical_output_dir, &lock)?;
    fs2::FileExt::lock_exclusive(&lock)
        .with_context(|| format!("locking gallery bookkeeping {}", output_dir.display()))?;
    let validation = (|| {
        verify_gallery_bookkeeping_lock_target(&canonical_output_dir, &lock)?;
        probe_gallery_bookkeeping_lock_semantics(&canonical_output_dir)?;
        Ok::<_, anyhow::Error>(())
    })();
    if let Err(error) = validation {
        let _ = fs2::FileExt::unlock(&lock);
        return Err(error);
    }
    Ok(GalleryBookkeepingGuard {
        lock,
        canonical_output_dir,
    })
}

#[cfg(unix)]
fn open_gallery_bookkeeping_lock_target(output_dir: &Path) -> anyhow::Result<File> {
    // The resolved gallery directory inode is the API's filesystem trust
    // root. Renaming or replacing the gallery directory itself behind a live
    // server is an out-of-band filesystem mutation outside the logical API
    // atomicity boundary; internal transaction cleanup never does so.
    File::open(output_dir)
        .with_context(|| format!("opening gallery directory {}", output_dir.display()))
}

#[cfg(windows)]
fn open_gallery_bookkeeping_lock_target(output_dir: &Path) -> anyhow::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_SHARE_READ,
        FILE_SHARE_WRITE,
    };

    // std cannot portably open a directory handle on every non-Unix target.
    // Canonicalize first so aliases and Windows junction paths converge on
    // one stable sidecar outside the removable transaction subtree.
    let lock_path = gallery_bookkeeping_sidecar_lock_path(output_dir)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .access_mode(FILE_GENERIC_READ | FILE_GENERIC_WRITE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        // A live bookkeeping inode must not be renamed or replaced.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(&lock_path)
        .with_context(|| {
            format!(
                "opening gallery bookkeeping lock {} for {}",
                lock_path.display(),
                output_dir.display(),
            )
        })?;
    verify_windows_authority_entry(&lock_path, &lock)?;
    Ok(lock)
}

#[cfg(not(any(unix, windows)))]
fn open_gallery_bookkeeping_lock_target(output_dir: &Path) -> anyhow::Result<File> {
    let lock_path = gallery_bookkeeping_sidecar_lock_path(output_dir)?;
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| {
            format!(
                "opening gallery bookkeeping lock {} for {}",
                lock_path.display(),
                output_dir.display(),
            )
        })
}

#[cfg(unix)]
fn verify_gallery_bookkeeping_lock_target(output_dir: &Path, file: &File) -> anyhow::Result<()> {
    use std::os::unix::fs::MetadataExt;

    let path_metadata = fs::metadata(output_dir)?;
    let file_metadata = file.metadata()?;
    ensure!(
        path_metadata.is_dir()
            && file_metadata.is_dir()
            && path_metadata.dev() == file_metadata.dev()
            && path_metadata.ino() == file_metadata.ino(),
        "gallery bookkeeping directory was replaced while locking: {}",
        output_dir.display()
    );
    Ok(())
}

#[cfg(windows)]
fn verify_gallery_bookkeeping_lock_target(output_dir: &Path, file: &File) -> anyhow::Result<()> {
    verify_windows_authority_entry(&gallery_bookkeeping_sidecar_lock_path(output_dir)?, file)?;
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn verify_gallery_bookkeeping_lock_target(output_dir: &Path, file: &File) -> anyhow::Result<()> {
    let path = gallery_bookkeeping_sidecar_lock_path(output_dir)?;
    ensure!(
        fs::canonicalize(&path)? == path && file.metadata()?.is_file(),
        "gallery bookkeeping sidecar was replaced while locking: {}",
        path.display()
    );
    Ok(())
}

fn probe_gallery_bookkeeping_lock_semantics(output_dir: &Path) -> anyhow::Result<()> {
    let probe = open_gallery_bookkeeping_lock_target(output_dir)?;
    match fs2::FileExt::try_lock_exclusive(&probe) {
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Ok(()) => {
            let _ = fs2::FileExt::unlock(&probe);
            bail!(
                "gallery filesystem does not provide exclusive locks across two descriptors; \
                 durable batch publication is unsupported for {}",
                output_dir.display()
            )
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "probing gallery filesystem lock semantics for {}",
                output_dir.display()
            )
        }),
    }
}

#[cfg(any(test, not(unix)))]
fn gallery_bookkeeping_sidecar_lock_path(output_dir: &Path) -> anyhow::Result<PathBuf> {
    let canonical = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let parent = canonical.parent().with_context(|| {
        format!(
            "using a filesystem root directly as the gallery is unsupported on this \
                 platform because stable bookkeeping requires a named sibling; choose a child \
                 directory: {}",
            canonical.display()
        )
    })?;
    let name = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| {
            format!(
                "canonical gallery directory requires a UTF-8 name for stable bookkeeping: {}",
                canonical.display()
            )
        })?;
    Ok(parent.join(format!(".{name}.mold-gallery.lock")))
}

fn committed_manifests_dir(output_dir: &Path, parent_id: &str) -> PathBuf {
    output_dir
        .join(TRANSACTION_DIR)
        .join(parent_id)
        .join(COMMITTED_DIR)
}

fn deleted_archive_children_dir(output_dir: &Path) -> PathBuf {
    output_dir
        .join(TRANSACTION_DIR)
        .join(DELETED_ARCHIVE_CHILDREN_DIR)
}

pub(crate) fn legacy_gallery_evidence_paths(output_dir: &Path) -> anyhow::Result<BTreeSet<String>> {
    let canonical_output_dir = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let root = canonical_output_dir.join(TRANSACTION_DIR);
    let metadata = match fs::symlink_metadata(&root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BTreeSet::new());
        }
        Err(error) => return Err(error.into()),
    };
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "batch transaction root is not a private directory: {}",
        root.display()
    );

    let mut evidence = BTreeSet::new();
    let deleted = root.join(DELETED_ARCHIVE_CHILDREN_DIR);
    if let Ok(metadata) = fs::symlink_metadata(&deleted) {
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "deleted archive-child namespace is not a private directory: {}",
            deleted.display()
        );
        for entry in sorted_directory_entries(&deleted)? {
            let metadata = fs::symlink_metadata(entry.path())?;
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "legacy gallery evidence is not a regular file: {}",
                entry.path().display()
            );
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("legacy gallery evidence filename is not UTF-8"))?;
            evidence.insert(format!("{DELETED_ARCHIVE_CHILDREN_DIR}/{name}"));
        }
    }

    for parent in sorted_directory_entries(&root)? {
        let parent_name = match parent.file_name().into_string() {
            Ok(name) => name,
            Err(_) => continue,
        };
        if matches!(
            parent_name.as_str(),
            "reservations" | LEGACY_ATTEMPT_LOCKS_DIR | DELETED_ARCHIVE_CHILDREN_DIR
        ) || parent_name == crate::gallery_authority::authority_dir_name()
        {
            continue;
        }
        let parent_metadata = fs::symlink_metadata(parent.path())?;
        ensure!(
            parent_metadata.is_dir() && !parent_metadata.file_type().is_symlink(),
            "unrecognized non-directory gallery transaction entry: {}",
            parent.path().display()
        );
        let committed = parent.path().join(COMMITTED_DIR);
        let metadata = match fs::symlink_metadata(&committed) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "committed archive namespace is not a private directory: {}",
            committed.display()
        );
        for entry in sorted_directory_entries(&committed)? {
            let metadata = fs::symlink_metadata(entry.path())?;
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "legacy gallery evidence is not a regular file: {}",
                entry.path().display()
            );
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("legacy gallery evidence filename is not UTF-8"))?;
            evidence.insert(format!("{parent_name}/{COMMITTED_DIR}/{name}"));
        }
    }
    Ok(evidence)
}

pub(crate) fn remove_legacy_gallery_evidence(
    output_dir: &Path,
    evidence: &BTreeSet<String>,
) -> anyhow::Result<()> {
    let canonical_output_dir = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let root = canonical_output_dir.join(TRANSACTION_DIR);
    let mut touched_directories = BTreeSet::new();
    for relative in evidence {
        let components = Path::new(relative)
            .components()
            .map(|component| match component {
                std::path::Component::Normal(value) => Ok(value),
                _ => anyhow::bail!("legacy gallery evidence path is not relative: {relative}"),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let valid_shape = (components.len() == 2
            && components[0] == std::ffi::OsStr::new(DELETED_ARCHIVE_CHILDREN_DIR))
            || (components.len() == 3 && components[1] == std::ffi::OsStr::new(COMMITTED_DIR));
        ensure!(
            valid_shape,
            "legacy gallery evidence path has an invalid shape: {relative}"
        );
        let path = root.join(relative);
        match fs::symlink_metadata(&path) {
            Ok(metadata) => {
                ensure!(
                    metadata.is_file() && !metadata.file_type().is_symlink(),
                    "legacy gallery evidence changed away from a regular file: {}",
                    path.display()
                );
                fs::remove_file(&path)?;
                touched_directories.insert(
                    path.parent()
                        .context("legacy gallery evidence has no parent")?
                        .to_path_buf(),
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    for directory in touched_directories {
        sync_dir(&directory)?;
        if fs::read_dir(&directory)?.next().is_none() {
            fs::remove_dir(&directory)?;
            if let Some(parent) = directory.parent() {
                sync_dir(parent)?;
                if parent != root && fs::read_dir(parent)?.next().is_none() {
                    fs::remove_dir(parent)?;
                    sync_dir(&root)?;
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct CommittedArchiveCatalog {
    index: CommittedArchiveIndex,
    retired: BTreeMap<String, Vec<ArchivedChildIdentity>>,
    missing: Vec<ArchivedChildIdentity>,
}

#[derive(Debug, Clone, Copy)]
enum ArchiveFinalValidation {
    ReconcileMissing,
}

impl ArchiveFinalValidation {
    fn allows_missing(self, filename: &str) -> bool {
        let _ = filename;
        matches!(self, Self::ReconcileMissing)
    }

    fn skips_content_validation(self, filename: &str) -> bool {
        let _ = filename;
        false
    }
}

fn archived_child_identity(
    manifest: &BatchAttemptManifest,
    child: &BatchManifestChild,
) -> anyhow::Result<ArchivedChildIdentity> {
    ensure!(
        manifest.state == BatchManifestState::Committed,
        "only committed batch manifests can provide gallery archive authority"
    );
    let checksum_sha256 = child
        .checksum_sha256
        .clone()
        .context("committed archive child has no checksum")?;
    let size_bytes = child
        .size_bytes
        .context("committed archive child has no size")?;
    Ok(ArchivedChildIdentity {
        parent_id: manifest.parent_id.clone(),
        attempt_generation: manifest.attempt_generation,
        child_index: child.child_index,
        final_name: child.final_name.clone(),
        checksum_sha256,
        size_bytes,
    })
}

fn deleted_archive_child_digest(identity: &ArchivedChildIdentity) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.deleted-archive-child.v1\0");
    digest.update((identity.parent_id.len() as u64).to_le_bytes());
    digest.update(identity.parent_id.as_bytes());
    digest.update(identity.attempt_generation.to_le_bytes());
    digest.update((identity.child_index as u64).to_le_bytes());
    digest.update((identity.final_name.len() as u64).to_le_bytes());
    digest.update(identity.final_name.as_bytes());
    digest.update((identity.checksum_sha256.len() as u64).to_le_bytes());
    digest.update(identity.checksum_sha256.as_bytes());
    digest.update(identity.size_bytes.to_le_bytes());
    format!("{:x}", digest.finalize())
}

fn is_atomic_json_temp(name: &str) -> bool {
    name.starts_with(&format!(".{MANIFEST_FILE}.tmp-"))
}

#[cfg(unix)]
pub(crate) fn open_regular_file_no_follow(path: &Path) -> anyhow::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let parent = path.parent().context("archive JSON path has no parent")?;
    let name = path
        .file_name()
        .context("archive JSON path has no filename")?;
    let name = CString::new(name.as_bytes()).context("archive JSON filename contains NUL")?;
    let parent = open_unix_directory_without_symlinks(parent)?;
    // SAFETY: `parent` owns a valid directory descriptor, `name` is
    // NUL-terminated, and a successful descriptor is transferred once.
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("opening no-follow archive JSON {}", path.display()));
    }
    // SAFETY: `fd` is a fresh owned descriptor returned by `openat`.
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file(),
        "archive JSON is not a regular file: {}",
        path.display()
    );
    Ok(file)
}

#[cfg(not(unix))]
pub(crate) fn open_regular_file_no_follow(path: &Path) -> anyhow::Result<File> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "archive JSON is not a regular no-follow file: {}",
        path.display()
    );
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "archive JSON changed away from a regular file while opening: {}",
        path.display()
    );
    Ok(file)
}

fn read_archive_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    serde_json::from_reader(open_regular_file_no_follow(path)?)
        .with_context(|| format!("reading archive JSON {}", path.display()))
}

fn sorted_directory_entries(path: &Path) -> anyhow::Result<Vec<fs::DirEntry>> {
    let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn read_deleted_archive_children(
    output_dir: &Path,
) -> anyhow::Result<BTreeSet<ArchivedChildIdentity>> {
    let deleted_dir = deleted_archive_children_dir(output_dir);
    let metadata = match fs::symlink_metadata(&deleted_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BTreeSet::new());
        }
        Err(error) => return Err(error.into()),
    };
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "deleted archive-child namespace is not a private directory: {}",
        deleted_dir.display()
    );

    let mut deleted = BTreeSet::new();
    for entry in sorted_directory_entries(&deleted_dir)? {
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("deleted archive-child filename is not UTF-8"))?;
        if is_atomic_json_temp(&name) {
            continue;
        }
        let digest = name
            .strip_suffix(".json")
            .filter(|digest| {
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .with_context(|| {
                format!(
                    "unrecognized deleted archive-child entry {}; move it outside {} before restarting",
                    entry.path().display(),
                    TRANSACTION_DIR
                )
            })?;
        let tombstone: DeletedArchiveChild = read_archive_json(&entry.path())?;
        ensure!(
            tombstone.version == DELETED_ARCHIVE_CHILD_VERSION,
            "unsupported deleted archive-child version {} in {}",
            tombstone.version,
            entry.path().display()
        );
        validate_component(&tombstone.identity.parent_id, "archive parent id")?;
        validate_component(&tombstone.identity.final_name, "archive final filename")?;
        ensure!(
            tombstone.identity.checksum_sha256.len() == 64
                && tombstone
                    .identity
                    .checksum_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "deleted archive child has an invalid checksum in {}",
            entry.path().display()
        );
        ensure!(
            digest == deleted_archive_child_digest(&tombstone.identity),
            "deleted archive-child filename does not match its payload: {}",
            entry.path().display()
        );
        ensure!(
            deleted.insert(tombstone.identity),
            "duplicate deleted archive-child authority: {}",
            entry.path().display()
        );
    }
    Ok(deleted)
}

fn read_committed_archive_catalog(
    output_dir: &Path,
    final_validation: ArchiveFinalValidation,
) -> anyhow::Result<CommittedArchiveCatalog> {
    let canonical_output_dir = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let root = canonical_output_dir.join(TRANSACTION_DIR);
    let root_metadata = match fs::symlink_metadata(&root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CommittedArchiveCatalog::default());
        }
        Err(error) => return Err(error.into()),
    };
    ensure!(
        root_metadata.is_dir() && !root_metadata.file_type().is_symlink(),
        "batch transaction root is not a private directory: {}",
        root.display()
    );

    let mut unconsumed_deletions = read_deleted_archive_children(&canonical_output_dir)?;
    let mut catalog = CommittedArchiveCatalog::default();
    for parent in sorted_directory_entries(&root)? {
        let parent_name = match parent.file_name().into_string() {
            Ok(name) => name,
            Err(_) => continue,
        };
        if matches!(
            parent_name.as_str(),
            "reservations" | LEGACY_ATTEMPT_LOCKS_DIR | DELETED_ARCHIVE_CHILDREN_DIR
        ) || parent_name == crate::gallery_authority::authority_dir_name()
        {
            continue;
        }
        let parent_metadata = fs::symlink_metadata(parent.path())?;
        ensure!(
            parent_metadata.is_dir() && !parent_metadata.file_type().is_symlink(),
            "unrecognized non-directory gallery transaction entry: {}",
            parent.path().display()
        );
        validate_component(&parent_name, "archive parent id")?;
        let committed_dir = parent.path().join(COMMITTED_DIR);
        let committed_metadata = match fs::symlink_metadata(&committed_dir) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            committed_metadata.is_dir() && !committed_metadata.file_type().is_symlink(),
            "committed archive namespace is not a private directory: {}",
            committed_dir.display()
        );

        for archive in sorted_directory_entries(&committed_dir)? {
            let name = archive
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("committed archive filename is not UTF-8"))?;
            if is_atomic_json_temp(&name) {
                continue;
            }
            let generation_name = name.strip_suffix(".json").with_context(|| {
                format!(
                    "unrecognized committed archive entry {}; move it outside {} before restarting",
                    archive.path().display(),
                    TRANSACTION_DIR
                )
            })?;
            let generation = generation_name.parse::<u64>().with_context(|| {
                format!(
                    "committed archive generation is not numeric: {}",
                    archive.path().display()
                )
            })?;
            ensure!(
                generation.to_string() == generation_name,
                "committed archive generation is not canonical: {}",
                archive.path().display()
            );
            let manifest: BatchAttemptManifest = match read_archive_json(&archive.path()) {
                Ok(manifest) => manifest,
                Err(error) => {
                    tracing::error!(
                        archive = %archive.path().display(),
                        %error,
                        "quarantining malformed legacy gallery archive without blocking siblings"
                    );
                    continue;
                }
            };
            ensure!(
                manifest.parent_id == parent_name && manifest.attempt_generation == generation,
                "committed archive path does not match its manifest identity: {}",
                archive.path().display()
            );
            let expected_attempt =
                attempt_dir(output_dir, &manifest.parent_id, manifest.attempt_generation);
            validate_loaded_manifest(output_dir, &expected_attempt, &manifest)?;
            ensure!(
                manifest.state == BatchManifestState::Committed,
                "archived batch manifest is not committed: {}",
                archive.path().display()
            );

            for child in &manifest.children {
                let identity = archived_child_identity(&manifest, child)?;
                if unconsumed_deletions.remove(&identity) {
                    catalog
                        .retired
                        .entry(identity.final_name.clone())
                        .or_default()
                        .push(identity);
                    continue;
                }

                let final_path = canonical_output_dir.join(&identity.final_name);
                let metadata = match fs::symlink_metadata(&final_path) {
                    Ok(metadata) => metadata,
                    Err(error)
                        if final_validation.allows_missing(&identity.final_name)
                            && error.kind() == std::io::ErrorKind::NotFound =>
                    {
                        let name = identity.final_name.clone();
                        catalog.missing.push(identity.clone());
                        catalog.index.quarantined_names.insert(name.clone());
                        catalog
                            .index
                            .entries
                            .entry(name)
                            .or_insert(CommittedArchiveEntry {
                                identity,
                                record: child.record.clone(),
                                facts: None,
                            });
                        continue;
                    }
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!(
                                "committed archive child is missing: {}",
                                final_path.display()
                            )
                        });
                    }
                };
                let content_valid = final_validation.skips_content_validation(&identity.final_name)
                    || (metadata.is_file()
                        && !metadata.file_type().is_symlink()
                        && metadata.len() == identity.size_bytes
                        && checksum_file(&final_path)? == identity.checksum_sha256);
                let name = identity.final_name.clone();
                if !content_valid {
                    catalog.index.quarantined_names.insert(name.clone());
                }
                let prior = catalog.index.entries.insert(
                    name.clone(),
                    CommittedArchiveEntry {
                        identity,
                        record: child.record.clone(),
                        facts: (metadata.is_file() && !metadata.file_type().is_symlink())
                            .then(|| ArchiveFileFacts::from_metadata(&metadata))
                            .transpose()?,
                    },
                );
                if prior.is_some() {
                    catalog.index.entries.remove(&name);
                    catalog.index.quarantined_names.insert(name);
                }
            }
        }
    }
    ensure!(
        unconsumed_deletions.is_empty(),
        "deleted archive-child authority does not match any committed archive"
    );

    for (filename, retired) in &catalog.retired {
        if catalog.index.entries.contains_key(filename) {
            continue;
        }
        let final_path = canonical_output_dir.join(filename);
        match fs::symlink_metadata(&final_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                catalog.index.retired_names.insert(filename.clone());
            }
            Ok(metadata)
                if metadata.is_file()
                    && !metadata.file_type().is_symlink()
                    && retired.iter().any(|identity| {
                        identity.size_bytes == metadata.len()
                            && checksum_file(&final_path)
                                .is_ok_and(|checksum| checksum == identity.checksum_sha256)
                    }) =>
            {
                catalog.index.retired_names.insert(filename.clone());
            }
            Ok(_) => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(catalog)
}

fn load_committed_archive_index_legacy(output_dir: &Path) -> anyhow::Result<CommittedArchiveIndex> {
    Ok(read_committed_archive_catalog(output_dir, ArchiveFinalValidation::ReconcileMissing)?.index)
}

pub(crate) fn load_committed_archive_index(
    output_dir: &Path,
) -> anyhow::Result<CommittedArchiveIndex> {
    let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
    Ok(
        crate::gallery_authority::load_or_initialize(output_dir, &bookkeeping, || {
            load_committed_archive_index_legacy(output_dir)
        })?
        .index,
    )
}

/// Persist one ordinary server publication into the same committed authority
/// domain used by atomic batches. The public file must already be fsynced and
/// remain hidden behind the gallery writer until this function succeeds.
pub(crate) fn archive_ordinary_gallery_record(
    output_dir: &Path,
    final_path: &Path,
    mut record: GenerationRecord,
    gate: &GalleryPublicationGate,
    bookkeeping: &GalleryBookkeepingGuard,
) -> anyhow::Result<GenerationRecord> {
    bookkeeping.ensure_root(output_dir)?;
    let canonical_output_dir = bookkeeping.canonical_root();
    validate_component(&record.filename, "ordinary gallery filename")?;
    ensure!(
        fs::canonicalize(final_path)? == canonical_output_dir.join(&record.filename),
        "ordinary gallery record path does not match its filename"
    );
    record.id = None;
    record.output_dir = canonical_output_dir.to_string_lossy().into_owned();
    record.stat_from_disk(final_path);
    let size_bytes = fs::metadata(final_path)?.len();
    ensure!(
        record.file_size_bytes == Some(size_bytes as i64),
        "ordinary gallery record stat does not match final size"
    );
    let parent_id = format!("gallery-output-{}", uuid::Uuid::new_v4());
    let manifest = BatchAttemptManifest {
        version: MANIFEST_VERSION,
        parent_id: parent_id.clone(),
        attempt_generation: 0,
        normalized_request: serde_json::json!({"kind": "ordinary_gallery_publication"}),
        state: BatchManifestState::Committed,
        children: vec![BatchManifestChild {
            child_index: 0,
            staging_name: record.filename.clone(),
            final_name: record.filename.clone(),
            checksum_sha256: Some(checksum_file(final_path)?),
            size_bytes: Some(size_bytes),
            record: record.clone(),
        }],
    };
    sync_dir(canonical_output_dir)?;
    gate.record_committed_manifest(canonical_output_dir, &manifest, bookkeeping)?;
    Ok(record)
}

/// Durably retire the exact committed archive child currently claiming
/// `filename`. The tombstone precedes final-file deletion, so startup can
/// complete a crash-interrupted delete without discarding sibling metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArchiveDeleteDisposition {
    NoArchive,
    SafeToUnlink,
    PreservedReplacement,
}

pub(crate) fn tombstone_committed_archive_filename(
    output_dir: &Path,
    filename: &str,
    gate: &GalleryPublicationGate,
) -> anyhow::Result<ArchiveDeleteDisposition> {
    validate_component(filename, "gallery delete filename")?;
    let bookkeeping = acquire_gallery_bookkeeping_lock(output_dir)?;
    let canonical_output_dir = bookkeeping.canonical_root().to_path_buf();
    let output_dir = canonical_output_dir.as_path();
    let mut index = gate.committed_archive_index_while_locked(output_dir, &bookkeeping)?;
    let Some(entry) = index.entries.get(filename).cloned() else {
        return Ok(ArchiveDeleteDisposition::NoArchive);
    };
    let current_matches = crate::gallery_authority::current_file_matches(output_dir, &entry)?;
    let generation = crate::gallery_authority::read_generation(output_dir, &bookkeeping)?
        .context("gallery authority generation is missing")?;
    let disposition = if current_matches {
        index.entries.remove(filename);
        index.quarantined_names.remove(filename);
        index.retired_names.insert(filename.to_owned());
        index.retired_entries.insert(filename.to_owned(), entry);
        index
            .retirement_epochs
            .insert(filename.to_owned(), generation.saturating_add(1));
        index.retirement_projection_epochs.remove(filename);
        ArchiveDeleteDisposition::SafeToUnlink
    } else {
        index.quarantined_names.insert(filename.to_owned());
        ArchiveDeleteDisposition::PreservedReplacement
    };
    let mut next_generation = crate::gallery_authority::commit_snapshot(
        output_dir,
        &bookkeeping,
        generation,
        &mut index,
        match disposition {
            ArchiveDeleteDisposition::SafeToUnlink => "user_delete",
            ArchiveDeleteDisposition::PreservedReplacement => "delete_replacement_quarantine",
            ArchiveDeleteDisposition::NoArchive => unreachable!(),
        },
        vec![filename.to_owned()],
    )?;
    if disposition == ArchiveDeleteDisposition::SafeToUnlink {
        let path = output_dir.join(filename);
        if crate::gallery_authority::current_file_matches(
            output_dir,
            index
                .retired_entries
                .get(filename)
                .context("retired gallery identity disappeared before unlink")?,
        )? {
            fs::remove_file(&path)
                .with_context(|| format!("deleting exact gallery identity {}", path.display()))?;
            sync_ordinary_gallery_directory(output_dir)?;
        } else if fs::symlink_metadata(&path).is_ok() {
            index.quarantined_names.insert(filename.to_owned());
            next_generation = crate::gallery_authority::commit_snapshot(
                output_dir,
                &bookkeeping,
                next_generation,
                &mut index,
                "delete_replacement_quarantine",
                vec![filename.to_owned()],
            )?;
            gate.install_committed_archive_index(next_generation, index);
            return Ok(ArchiveDeleteDisposition::PreservedReplacement);
        }
    }
    gate.install_committed_archive_index(next_generation, index);
    Ok(disposition)
}

/// Finish any delete whose durable archive-child tombstone reached disk
/// before the process exited, then return the one validated current index.
pub(crate) fn reconcile_committed_archive_index(
    output_dir: &Path,
) -> anyhow::Result<CommittedArchiveIndex> {
    load_committed_archive_index(output_dir)
}

fn reservation_path(output_dir: &Path, final_name: &str) -> PathBuf {
    reservations_dir(output_dir).join(format!("{final_name}.reserve"))
}

fn sync_transaction_hierarchy(output_dir: &Path, attempt_dir: &Path) -> anyhow::Result<()> {
    let transaction_root = output_dir.join(TRANSACTION_DIR);
    let parent_root = attempt_dir
        .parent()
        .and_then(Path::parent)
        .context("batch attempt path has no parent root")?;
    let attempts_root = attempt_dir
        .parent()
        .context("batch attempt path has no attempts root")?;
    let staging = attempt_dir.join("staging");
    let reservations = reservations_dir(output_dir);
    for directory in [
        output_dir,
        transaction_root.as_path(),
        parent_root,
        attempts_root,
        attempt_dir,
        staging.as_path(),
        reservations.as_path(),
    ] {
        sync_dir(directory)?;
    }
    Ok(())
}

fn reserve_final_name(
    output_dir: &Path,
    desired: &str,
    owner: &ReservationOwner,
) -> anyhow::Result<String> {
    reserve_final_name_with_directory_sync(output_dir, desired, owner, &sync_dir)
}

fn reserve_final_name_with_directory_sync(
    output_dir: &Path,
    desired: &str,
    owner: &ReservationOwner,
    sync_directory: &dyn Fn(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<String> {
    let path = Path::new(desired);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .context("final filename has no UTF-8 stem")?;
    let extension = path.extension().and_then(|value| value.to_str());
    for collision in 0_u32.. {
        let candidate = if collision == 0 {
            desired.to_owned()
        } else if let Some(extension) = extension {
            format!("{stem}-{collision}.{extension}")
        } else {
            format!("{stem}-{collision}")
        };
        if output_dir.join(&candidate).exists() {
            continue;
        }
        let reservation = reservation_path(output_dir, &candidate);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&reservation)
        {
            Ok(mut file) => {
                let result = (|| {
                    file.write_all(&serde_json::to_vec(owner)?)?;
                    file.sync_all()?;
                    sync_directory(&reservations_dir(output_dir))
                })();
                if let Err(error) = result {
                    drop(file);
                    let _ = fs::remove_file(&reservation);
                    return Err(error);
                }
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("u32 collision namespace exhausted")
}

fn release_reservation(
    output_dir: &Path,
    final_name: &str,
    owner: &ReservationOwner,
) -> anyhow::Result<()> {
    let path = reservation_path(output_dir, final_name);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let actual: ReservationOwner = serde_json::from_slice(&bytes)
        .with_context(|| format!("reading reservation owner {}", path.display()))?;
    ensure!(
        actual == *owner,
        "reservation {} belongs to {}/generation {}, not {}/generation {}",
        final_name,
        actual.parent_id,
        actual.attempt_generation,
        owner.parent_id,
        owner.attempt_generation
    );
    fs::remove_file(path)?;
    Ok(())
}

pub(crate) fn reserve_gallery_final_name_with_directory_sync(
    output_dir: &Path,
    desired: &str,
    sync_directory: &dyn Fn(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<GalleryNameReservation> {
    fs::create_dir_all(output_dir)?;
    let bookkeeping_lock = acquire_gallery_bookkeeping_lock(output_dir)?;
    fs::create_dir_all(reservations_dir(output_dir))?;
    let owner = ReservationOwner {
        parent_id: format!("ordinary:{}", uuid::Uuid::new_v4()),
        attempt_generation: 0,
    };
    let final_name =
        reserve_final_name_with_directory_sync(output_dir, desired, &owner, sync_directory)?;
    Ok(GalleryNameReservation {
        output_dir: output_dir.to_path_buf(),
        final_name,
        owner,
        _bookkeeping_lock: bookkeeping_lock,
    })
}

fn load_journal(path: &Path) -> anyhow::Result<Vec<BatchJournalRecord>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut records = Vec::new();
    let has_incomplete_tail = !bytes.ends_with(b"\n");
    let lines: Vec<&[u8]> = bytes.split(|byte| *byte == b'\n').collect();
    let last_nonempty = lines.iter().rposition(|line| !line.is_empty());
    let mut offset = 0_u64;
    let mut discarded_incomplete_tail = false;
    for (index, line) in lines.into_iter().enumerate() {
        if line.is_empty() {
            if offset < bytes.len() as u64 {
                offset += 1;
            }
            continue;
        }
        match serde_json::from_slice::<BatchJournalRecord>(line) {
            Ok(record) => {
                ensure!(
                    record.sequence == records.len() as u64,
                    "batch journal sequence gap at {}",
                    path.display()
                );
                if let BatchJournalEvent::ManifestSnapshot { manifest } = &record.event {
                    ensure!(
                        record.attempt_generation == manifest.attempt_generation,
                        "batch journal generation does not match its manifest at {}",
                        path.display()
                    );
                }
                records.push(record);
                offset = (offset + line.len() as u64 + 1).min(bytes.len() as u64);
            }
            Err(error) if Some(index) == last_nonempty && has_incomplete_tail => {
                tracing::warn!(
                    journal = %path.display(),
                    %error,
                    "ignoring incomplete trailing batch journal record"
                );
                let file = OpenOptions::new().write(true).open(path)?;
                file.set_len(offset)?;
                file.sync_all()?;
                if let Some(parent) = path.parent() {
                    sync_dir(parent)?;
                }
                discarded_incomplete_tail = true;
                break;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("parsing batch journal {}", path.display()));
            }
        }
    }
    if has_incomplete_tail && !discarded_incomplete_tail {
        let mut file = OpenOptions::new().append(true).open(path)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        if let Some(parent) = path.parent() {
            sync_dir(parent)?;
        }
    }
    Ok(records)
}

fn validate_component(value: &str, description: &str) -> anyhow::Result<()> {
    let path = Path::new(value);
    ensure!(!value.is_empty(), "{description} cannot be empty");
    ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(value),
        "{description} must be one path component"
    );
    ensure!(value != "." && value != "..", "{description} is invalid");
    Ok(())
}

fn validate_loaded_manifest(
    output_dir: &Path,
    attempt_dir: &Path,
    manifest: &BatchAttemptManifest,
) -> anyhow::Result<()> {
    ensure!(
        matches!(manifest.version, MANIFEST_VERSION_V1 | MANIFEST_VERSION),
        "unsupported batch manifest version {}",
        manifest.version
    );
    validate_component(&manifest.parent_id, "batch parent id")?;
    let generation_name = manifest.attempt_generation.to_string();
    ensure!(
        attempt_dir.file_name().and_then(|name| name.to_str()) == Some(generation_name.as_str()),
        "batch manifest generation does not match attempt directory {}",
        attempt_dir.display()
    );
    ensure!(
        attempt_dir
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            == Some(manifest.parent_id.as_str()),
        "batch manifest parent does not match attempt directory {}",
        attempt_dir.display()
    );
    ensure!(
        !manifest.children.is_empty(),
        "batch manifest has no children"
    );

    let expected_output_dir = fs::canonicalize(output_dir)
        .with_context(|| format!("canonicalizing gallery {}", output_dir.display()))?;
    let mut staging_names = BTreeSet::new();
    let mut final_names = BTreeSet::new();
    for (expected_index, child) in manifest.children.iter().enumerate() {
        ensure!(
            child.child_index == expected_index,
            "batch manifest child indices are not ordered at {expected_index}"
        );
        validate_component(&child.staging_name, "batch staging filename")?;
        validate_component(&child.final_name, "batch final filename")?;
        ensure!(
            staging_names.insert(child.staging_name.as_str()),
            "batch manifest repeats staging filename {}",
            child.staging_name
        );
        ensure!(
            final_names.insert(child.final_name.as_str()),
            "batch manifest repeats final filename {}",
            child.final_name
        );
        ensure!(
            child.record.filename == child.final_name,
            "batch metadata filename does not match reserved final filename {}",
            child.final_name
        );
        let record_output_dir = fs::canonicalize(Path::new(&child.record.output_dir))
            .with_context(|| {
                format!(
                    "canonicalizing archived batch metadata output directory {}",
                    child.record.output_dir
                )
            })?;
        ensure!(
            record_output_dir == expected_output_dir,
            "batch metadata output directory does not match {}",
            output_dir.display()
        );
        match (&child.checksum_sha256, child.size_bytes) {
            (None, None) => ensure!(
                matches!(
                    manifest.state,
                    BatchManifestState::Staging | BatchManifestState::Failed
                ),
                "non-staging/non-failed batch child {} has no checksum",
                child.child_index
            ),
            (Some(checksum), Some(size)) => {
                ensure!(
                    checksum.len() == 64 && checksum.bytes().all(|byte| byte.is_ascii_hexdigit()),
                    "batch child {} has an invalid SHA-256 checksum",
                    child.child_index
                );
                ensure!(
                    size <= i64::MAX as u64 && child.record.file_size_bytes == Some(size as i64),
                    "batch child {} metadata size does not match staged size",
                    child.child_index
                );
            }
            _ => anyhow::bail!(
                "batch child {} has an incomplete checksum/size pair",
                child.child_index
            ),
        }
    }
    Ok(())
}

fn checksum_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn immutable_record_identity(child: &BatchManifestChild) -> anyhow::Result<String> {
    let mut immutable = child.clone();
    immutable.checksum_sha256 = None;
    immutable.size_bytes = None;
    immutable.record.id = None;
    immutable.record.file_mtime_ms = None;
    immutable.record.file_size_bytes = None;
    Ok(checksum_bytes(&serde_json::to_vec(&immutable)?))
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic payload")
}

fn checksum_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn checksum_file_for_authority(path: &Path) -> anyhow::Result<String> {
    #[cfg(test)]
    AUTHORITY_HASH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    checksum_file(path)
}

#[cfg(test)]
static AUTHORITY_HASH_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
fn reset_authority_hash_count() {
    AUTHORITY_HASH_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
fn authority_hash_count() -> usize {
    AUTHORITY_HASH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

fn publish_no_replace(staged: &Path, final_path: &Path) -> anyhow::Result<()> {
    match fs::hard_link(staged, final_path) {
        Ok(()) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            anyhow::bail!(
                "reserved final path appeared during publication: {}",
                final_path.display()
            );
        }
        Err(hard_link_error) => {
            // Some gallery filesystems do not support hard links. The
            // publication gate keeps API observers out while this
            // create-new copy is written, and create_new preserves the
            // no-overwrite contract.
            let mut source = File::open(staged)?;
            let mut destination = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(final_path)
                .with_context(|| {
                    format!(
                        "publishing {} without replacement after hard-link failure: {hard_link_error}",
                        final_path.display()
                    )
                })?;
            if let Err(error) =
                std::io::copy(&mut source, &mut destination).and_then(|_| destination.sync_all())
            {
                drop(destination);
                let cleanup = fs::remove_file(final_path);
                if let Err(cleanup_error) = cleanup {
                    return Err(error).with_context(|| {
                        format!(
                            "publishing {} failed and partial final cleanup also failed: {cleanup_error}",
                            final_path.display()
                        )
                    });
                }
                return Err(error).with_context(|| {
                    format!("publishing {} by no-replace copy", final_path.display())
                });
            }
        }
    }
    Ok(())
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let parent = path.parent().context("manifest path has no parent")?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".{MANIFEST_FILE}.tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| {
        let bytes = serde_json::to_vec_pretty(value)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        sync_dir(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

pub(crate) fn sync_ordinary_gallery_directory(path: &Path) -> anyhow::Result<()> {
    tolerate_unsupported_ordinary_directory_sync(path, sync_dir(path))
}

pub(crate) fn tolerate_unsupported_ordinary_directory_sync(
    path: &Path,
    result: anyhow::Result<()>,
) -> anyhow::Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(error) if directory_sync_is_unsupported(&error) => {
            tracing::warn!(
                directory = %path.display(),
                %error,
                "gallery filesystem does not support directory fsync; ordinary output remains saved with best-effort directory durability"
            );
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn directory_sync_is_unsupported(error: &anyhow::Error) -> bool {
    let Some(io_error) = error.downcast_ref::<std::io::Error>() else {
        return false;
    };
    if io_error.kind() == std::io::ErrorKind::Unsupported {
        return true;
    }
    #[cfg(unix)]
    {
        io_error.raw_os_error().is_some_and(|code| {
            code == libc::EINVAL || code == libc::EOPNOTSUPP || code == libc::ENOSYS
        })
    }
    #[cfg(not(unix))]
    {
        false
    }
}

#[cfg(unix)]
fn sync_dir(path: &Path) -> anyhow::Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
    use mold_db::RecordSource;
    use std::io::BufRead as _;
    use std::sync::{Arc as StdArc, Barrier};

    fn record(name: &str, seed: u64) -> GenerationRecord {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": format!("prompt {seed}"),
            "model": "test",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0,
            "batch_id": "parent",
            "batch_index": seed + 1,
            "batch_count": 2
        }))
        .unwrap();
        GenerationRecord::from_save(
            Path::new("will-be-replaced"),
            name,
            OutputFormat::Png,
            OutputMetadata::from_generate_request(&request, seed, None, "test"),
            RecordSource::Server,
            1,
        )
    }

    fn output_metadata(seed: u64) -> OutputMetadata {
        record("unused.png", seed).metadata
    }

    #[test]
    #[ignore = "subprocess helper for cross-process gallery authority tests"]
    fn gallery_authority_process_helper() {
        let output = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT").expect("MOLD_TEST_GALLERY_OUTPUT"),
        );
        match std::env::var("MOLD_TEST_GALLERY_ACTION").as_deref() {
            Ok("batch") => {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let mut batch = BatchTransaction::begin(
                        &output,
                        "subprocess-batch",
                        0,
                        serde_json::json!({}),
                        vec![record("subprocess.png", 7)],
                    )
                    .unwrap();
                    batch.stage_bytes(0, b"subprocess").unwrap();
                    batch.mark_prepared().unwrap();
                    batch
                        .commit(&GalleryPublicationGate::default(), Arc::new(None))
                        .await
                        .unwrap();
                });
            }
            Ok("named") => {
                crate::queue::save_video_to_dir_named(
                    &output,
                    "stable-chain.mp4",
                    b"same named bytes",
                    OutputFormat::Mp4,
                    &output_metadata(8),
                    None,
                    None,
                    None,
                    &GalleryPublicationGate::default(),
                )
                .unwrap();
            }
            action => panic!("unknown helper action: {action:?}"),
        }
    }

    fn append_raw_batch_record(attempt_dir: &Path, record: &BatchJournalRecord) {
        let mut bytes = serde_json::to_vec(record).unwrap();
        bytes.push(b'\n');
        let mut journal = OpenOptions::new()
            .append(true)
            .open(attempt_dir.join(JOURNAL_FILE))
            .unwrap();
        journal.write_all(&bytes).unwrap();
        journal.sync_all().unwrap();
    }

    fn next_batch_sequence(transaction: &BatchTransaction) -> u64 {
        load_journal(&transaction.attempt_dir.join(JOURNAL_FILE))
            .unwrap()
            .len() as u64
    }

    #[tokio::test]
    async fn concurrent_begin_admits_one_attempt_authority_and_recovers_it() {
        const CONTENDERS: usize = 64;

        let dir = tempfile::tempdir().unwrap();
        let output_dir = StdArc::new(dir.path().to_path_buf());
        let barrier = StdArc::new(Barrier::new(CONTENDERS));
        let handles: Vec<_> = (0..CONTENDERS)
            .map(|_| {
                let output_dir = StdArc::clone(&output_dir);
                let barrier = StdArc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    BatchTransaction::begin(
                        &output_dir,
                        "parent",
                        0,
                        serde_json::json!({"batch_size": 1}),
                        vec![record("one.png", 0)],
                    )
                    .is_ok()
                })
            })
            .collect();
        let admitted = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|admitted| *admitted)
            .count();

        assert_eq!(
            admitted, 1,
            "one filesystem authority must own a parent attempt generation"
        );
        let attempt = attempt_dir(dir.path(), "parent", 0);
        let records = load_journal(&attempt.join(JOURNAL_FILE)).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].sequence, 0);

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert!(!attempt.exists());
    }

    #[test]
    fn ordinary_reservation_drop_cannot_remove_directory_during_batch_begin() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = StdArc::new(dir.path().to_path_buf());
        let ordinary = reserve_gallery_final_name_with_directory_sync(
            &output_dir,
            "ordinary.png",
            &|_| Ok(()),
        )
        .unwrap();
        let (directory_ready_tx, directory_ready_rx) = std::sync::mpsc::channel();
        let (continue_begin_tx, continue_begin_rx) = std::sync::mpsc::channel();
        let begin_output = StdArc::clone(&output_dir);
        let begin = std::thread::spawn(move || {
            BatchTransaction::begin_with_reservation_directory_hook(
                &begin_output,
                "parent",
                0,
                serde_json::json!({"batch_size": 1}),
                vec![record("batch.png", 0)],
                move || {
                    directory_ready_tx.send(()).unwrap();
                    continue_begin_rx.recv().unwrap();
                },
            )
        });
        assert!(
            matches!(
                directory_ready_rx.recv_timeout(std::time::Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ),
            "batch begin bypassed a live ordinary reservation's filesystem authority"
        );

        drop(ordinary);
        directory_ready_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("batch begin did not continue after ordinary reservation drop");
        continue_begin_tx.send(()).unwrap();

        let transaction = begin.join().unwrap();
        let transaction = transaction.expect("batch begin must retain the reservations directory");
        assert!(
            reservation_path(&output_dir, "batch.png").is_file(),
            "batch reservation authority was lost"
        );
        assert!(
            !reservation_path(&output_dir, "ordinary.png").exists(),
            "ordinary reservation bookkeeping leaked"
        );
        transaction.release_reservations();
        remove_failed_attempt(&transaction);
    }

    #[test]
    fn ordinary_reservation_drop_cleans_an_otherwise_empty_gallery() {
        let dir = tempfile::tempdir().unwrap();
        let reservation =
            reserve_gallery_final_name_with_directory_sync(dir.path(), "ordinary.png", &|_| Ok(()))
                .unwrap();
        drop(reservation);

        assert!(
            !dir.path().join(TRANSACTION_DIR).exists(),
            "ordinary saves must not leave batch bookkeeping in a clean gallery"
        );
    }

    fn write_process_test_marker(marker: &str) {
        println!("MOLD_RESERVATION_TEST:{marker}");
        std::io::stdout().flush().unwrap();
    }

    fn read_process_test_marker(
        reader: &mut impl std::io::BufRead,
        expected: &str,
    ) -> anyhow::Result<()> {
        let expected = format!("MOLD_RESERVATION_TEST:{expected}");
        loop {
            let mut line = String::new();
            ensure!(
                reader.read_line(&mut line)? != 0,
                "reservation helper exited before marker {expected}"
            );
            if line.trim() == expected {
                return Ok(());
            }
        }
    }

    fn predecessor_attempt_authority_path(attempt_dir: &Path) -> PathBuf {
        let generation = attempt_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap()
            .parse::<u64>()
            .unwrap();
        let parent_root = attempt_dir.parent().unwrap().parent().unwrap();
        let parent_id = parent_root.file_name().unwrap().to_str().unwrap();
        let transaction_root = parent_root.parent().unwrap();
        let locks = transaction_root.join(LEGACY_ATTEMPT_LOCKS_DIR);
        fs::create_dir_all(&locks).unwrap();

        let mut identity = Sha256::new();
        identity.update(b"mold.batch-attempt-authority.v1\0");
        identity.update((parent_id.len() as u64).to_le_bytes());
        identity.update(parent_id.as_bytes());
        identity.update(generation.to_le_bytes());
        locks.join(format!("{:x}.lock", identity.finalize()))
    }

    #[test]
    #[ignore = "subprocess helper emulating the predecessor v1 authority protocol"]
    fn predecessor_attempt_authority_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let parent_id =
            std::env::var("MOLD_TEST_BATCH_PARENT").unwrap_or_else(|_| "mixed-parent".to_owned());
        let generation = std::env::var("MOLD_TEST_BATCH_GENERATION")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<u64>()
            .unwrap();
        let path =
            predecessor_attempt_authority_path(&attempt_dir(&output_dir, &parent_id, generation));
        let lock = open_attempt_authority_file(&path).unwrap();
        match std::env::var("MOLD_TEST_PREDECESSOR_MODE")
            .expect("MOLD_TEST_PREDECESSOR_MODE must be set")
            .as_str()
        {
            "owner" => {
                fs2::FileExt::lock_exclusive(&lock).unwrap();
                write_process_test_marker("PREDECESSOR_LOCKED");
                let mut command = String::new();
                std::io::BufReader::new(std::io::stdin())
                    .read_line(&mut command)
                    .unwrap();
                match command.trim() {
                    "RELEASE" => {
                        fs2::FileExt::unlock(&lock).unwrap();
                        write_process_test_marker("PREDECESSOR_RELEASED");
                    }
                    "CRASH" => {
                        write_process_test_marker("PREDECESSOR_CRASHING");
                        std::process::exit(0);
                    }
                    command => panic!("unexpected predecessor owner command {command}"),
                }
            }
            "claim" => match fs2::FileExt::try_lock_exclusive(&lock) {
                Ok(()) => {
                    write_process_test_marker("PREDECESSOR_ACQUIRED");
                    fs2::FileExt::unlock(&lock).unwrap();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    write_process_test_marker("PREDECESSOR_CONTENDED");
                }
                Err(error) => panic!("predecessor authority probe failed: {error}"),
            },
            mode => panic!("unexpected predecessor authority helper mode {mode}"),
        }
    }

    fn spawn_predecessor_authority_helper(
        output_dir: &Path,
        mode: &str,
        parent_id: &str,
    ) -> (
        std::process::Child,
        std::io::BufWriter<std::process::ChildStdin>,
        std::io::BufReader<std::process::ChildStdout>,
    ) {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::predecessor_attempt_authority_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", output_dir)
            .env("MOLD_TEST_PREDECESSOR_MODE", mode)
            .env("MOLD_TEST_BATCH_PARENT", parent_id)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let input = std::io::BufWriter::new(child.stdin.take().unwrap());
        let output = std::io::BufReader::new(child.stdout.take().unwrap());
        (child, input, output)
    }

    #[tokio::test]
    async fn predecessor_v1_owner_blocks_v2_recovery_until_release() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"mixed child").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let (mut predecessor, mut predecessor_input, mut predecessor_output) =
            spawn_predecessor_authority_helper(dir.path(), "owner", "mixed-parent");
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_LOCKED").unwrap();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();
        assert_eq!(
            report,
            RecoveryReport::default(),
            "v2 recovery mutated an attempt still owned by a live v1 process"
        );
        assert!(
            transaction.attempt_dir.exists(),
            "v2 recovery removed a live v1 attempt"
        );

        writeln!(predecessor_input, "RELEASE").unwrap();
        predecessor_input.flush().unwrap();
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_RELEASED").unwrap();
        let _ = std::io::copy(&mut predecessor_output, &mut std::io::sink());
        assert!(predecessor.wait().unwrap().success());

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert!(!transaction.attempt_dir.exists());
    }

    #[test]
    fn predecessor_v1_owner_makes_v2_begin_return_typed_contention() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let (mut predecessor, mut predecessor_input, mut predecessor_output) =
            spawn_predecessor_authority_helper(dir.path(), "owner", "mixed-parent");
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_LOCKED").unwrap();

        let output_dir = dir.path().to_path_buf();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            result_tx
                .send(BatchTransaction::begin(
                    &output_dir,
                    "mixed-parent",
                    0,
                    serde_json::json!({"batch_size": 1}),
                    vec![record("mixed.png", 0)],
                ))
                .unwrap();
        });
        let error = result_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("mixed-version begin formed a legacy/current/bookkeeping lock cycle")
            .unwrap_err();
        error
            .downcast_ref::<BatchAttemptAuthorityContended>()
            .expect("live predecessor ownership must surface typed v2 contention");

        writeln!(predecessor_input, "RELEASE").unwrap();
        predecessor_input.flush().unwrap();
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_RELEASED").unwrap();
        let _ = std::io::copy(&mut predecessor_output, &mut std::io::sink());
        assert!(predecessor.wait().unwrap().success());
    }

    #[test]
    fn v2_owner_blocks_predecessor_style_v1_claimant() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();

        let (mut predecessor, _predecessor_input, mut predecessor_output) =
            spawn_predecessor_authority_helper(dir.path(), "claim", "mixed-parent");
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_CONTENDED").unwrap();
        let _ = std::io::copy(&mut predecessor_output, &mut std::io::sink());
        assert!(predecessor.wait().unwrap().success());

        drop(transaction);
    }

    #[tokio::test]
    async fn predecessor_v1_process_exit_releases_authority_for_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"mixed child").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let (mut predecessor, mut predecessor_input, mut predecessor_output) =
            spawn_predecessor_authority_helper(dir.path(), "owner", "mixed-parent");
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_LOCKED").unwrap();
        writeln!(predecessor_input, "CRASH").unwrap();
        predecessor_input.flush().unwrap();
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_CRASHING").unwrap();
        let _ = std::io::copy(&mut predecessor_output, &mut std::io::sink());
        assert!(predecessor.wait().unwrap().success());

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert!(!transaction.attempt_dir.exists());
    }

    #[test]
    fn simultaneous_v2_recovery_skips_a_predecessor_v1_owner() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"mixed child").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let (mut predecessor, mut predecessor_input, mut predecessor_output) =
            spawn_predecessor_authority_helper(dir.path(), "owner", "mixed-parent");
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_LOCKED").unwrap();

        let output_dir = StdArc::new(dir.path().to_path_buf());
        let barrier = StdArc::new(Barrier::new(2));
        let recoveries: Vec<_> = (0..2)
            .map(|_| {
                let output_dir = StdArc::clone(&output_dir);
                let barrier = StdArc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap()
                        .block_on(recover_transactions(
                            &output_dir,
                            &GalleryPublicationGate::default(),
                            Arc::new(None),
                        ))
                        .unwrap()
                })
            })
            .collect();
        for recovery in recoveries {
            assert_eq!(
                recovery.join().unwrap(),
                RecoveryReport::default(),
                "simultaneous v2 recovery crossed a live predecessor authority"
            );
        }
        assert!(transaction.attempt_dir.exists());

        writeln!(predecessor_input, "RELEASE").unwrap();
        predecessor_input.flush().unwrap();
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_RELEASED").unwrap();
        let _ = std::io::copy(&mut predecessor_output, &mut std::io::sink());
        assert!(predecessor.wait().unwrap().success());
    }

    #[test]
    fn stale_unlocked_predecessor_v1_authority_is_retained_as_a_reusable_inode() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();
        let legacy = predecessor_attempt_authority_path(&transaction.attempt_dir);
        fs::write(&legacy, b"stale predecessor").unwrap();

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        sweep_reclaimable_attempt_authorities(&bookkeeping).unwrap();
        assert!(
            legacy.exists(),
            "rolling recovery must retain the predecessor's reusable stable inode"
        );
        assert!(transaction.attempt_dir.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn public_recovery_sweeps_current_authority_through_a_gallery_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let gallery = dir.path().join("gallery");
        fs::create_dir(&gallery).unwrap();
        let alias = dir.path().join("gallery-alias");
        std::os::unix::fs::symlink(&gallery, &alias).unwrap();

        let current = gallery.join(format!(
            "{ATTEMPT_LOCK_PREFIX}{}{ATTEMPT_LOCK_SUFFIX}",
            "a".repeat(64)
        ));
        let legacy = gallery
            .join(TRANSACTION_DIR)
            .join(LEGACY_ATTEMPT_LOCKS_DIR)
            .join(format!("{}{}", "b".repeat(64), ATTEMPT_LOCK_SUFFIX));
        fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        fs::write(&current, b"stale current authority").unwrap();
        fs::write(&legacy, b"stable predecessor authority").unwrap();

        recover_transactions(&alias, &GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();

        assert!(
            !current.exists(),
            "public recovery through an alias retained a stale current authority"
        );
        assert!(
            legacy.exists(),
            "rolling compatibility must retain the predecessor's reusable stable inode"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn public_recovery_warn_retains_exact_legacy_symlink_and_directory_entries() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir
            .path()
            .join(TRANSACTION_DIR)
            .join(LEGACY_ATTEMPT_LOCKS_DIR);
        fs::create_dir_all(&legacy).unwrap();
        let outside = dir.path().join("outside");
        fs::write(&outside, b"operator data").unwrap();
        let symlink = legacy.join(format!("{}{}", "c".repeat(64), ATTEMPT_LOCK_SUFFIX));
        std::os::unix::fs::symlink(&outside, &symlink).unwrap();
        let directory = legacy.join(format!("{}{}", "d".repeat(64), ATTEMPT_LOCK_SUFFIX));
        fs::create_dir(&directory).unwrap();

        recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .expect("broad recovery must warn-retain suspicious legacy entries");

        assert!(fs::symlink_metadata(&symlink)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(directory.is_dir());
        assert_eq!(fs::read(&outside).unwrap(), b"operator data");
    }

    #[cfg(unix)]
    #[test]
    fn predecessor_open_before_lock_keeps_one_stable_legacy_authority_inode() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "mixed-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("mixed.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();
        let legacy = predecessor_attempt_authority_path(&transaction.attempt_dir);
        let predecessor = open_attempt_authority_file(&legacy).unwrap();

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        sweep_reclaimable_attempt_authorities(&bookkeeping).unwrap();
        fs2::FileExt::lock_exclusive(&predecessor).unwrap();

        let claim = try_claim_attempt_authority(&transaction.attempt_dir, &bookkeeping).unwrap();
        let v2_crossed_predecessor = claim.is_some();
        if let Some(mut authority) = claim {
            authority.reclaim_under_bookkeeping(&bookkeeping);
        }
        fs2::FileExt::unlock(&predecessor).unwrap();
        drop(bookkeeping);

        assert!(
            legacy.exists(),
            "v2 sweep detached an authority file already opened by a rolling v1 process"
        );
        assert!(
            !v2_crossed_predecessor,
            "v2 created and locked a new legacy inode beside the predecessor's detached lock"
        );
    }

    #[cfg(unix)]
    #[test]
    fn exact_claim_fails_closed_on_a_legacy_authority_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();
        let legacy = predecessor_attempt_authority_path(&transaction.attempt_dir);
        let outside = dir.path().join("outside-authority");
        fs::write(&outside, b"outside").unwrap();
        fs::remove_file(&legacy).unwrap();
        std::os::unix::fs::symlink(&outside, &legacy).unwrap();

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let error = try_claim_attempt_authority(&transaction.attempt_dir, &bookkeeping)
            .expect_err("an exact claim must reject a legacy authority symlink");
        assert!(
            format!("{error:#}").contains("no-follow"),
            "unexpected exact legacy-claim error: {error:#}"
        );
        assert_eq!(fs::read(outside).unwrap(), b"outside");
    }

    #[cfg(unix)]
    #[test]
    fn hardlinked_stale_authority_is_warn_skipped_but_exact_claim_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();
        let authority = attempt_authority_lock_path(&transaction.attempt_dir).unwrap();
        let other_link = dir.path().join("operator-inspection-link");
        fs::write(&authority, b"suspicious").unwrap();
        fs::hard_link(&authority, &other_link).unwrap();

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        sweep_reclaimable_attempt_authorities(&bookkeeping)
            .expect("sweep must retain and warn-skip an unverifiable stale sidecar");
        assert!(authority.exists());
        assert!(other_link.exists());

        let error = try_claim_attempt_authority(&transaction.attempt_dir, &bookkeeping)
            .expect_err("an exact claim must fail closed on a hardlinked authority");
        assert!(
            format!("{error:#}").contains("private regular file"),
            "unexpected exact-claim error: {error:#}"
        );
    }

    #[test]
    fn bookkeeping_lock_excludes_a_second_same_process_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let probe =
            open_gallery_bookkeeping_lock_target(&bookkeeping.canonical_output_dir).unwrap();
        assert!(
            matches!(
                fs2::FileExt::try_lock_exclusive(&probe),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
            ),
            "filesystem lock semantics admit two same-process bookkeeping owners"
        );
    }

    #[cfg(unix)]
    #[test]
    fn root_gallery_sidecar_boundary_has_actionable_guidance() {
        let error = gallery_bookkeeping_sidecar_lock_path(Path::new("/")).unwrap_err();
        assert!(error.to_string().contains("choose a child directory"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_bookkeeping_sidecar_cannot_be_replaced_while_guard_is_live() {
        let dir = tempfile::tempdir().unwrap();
        let gallery = dir.path().join("gallery");
        fs::create_dir(&gallery).unwrap();
        let bookkeeping = acquire_gallery_bookkeeping_lock(&gallery).unwrap();
        let sidecar = gallery_bookkeeping_sidecar_lock_path(&gallery).unwrap();
        let displaced = dir.path().join("displaced-gallery-lock");
        assert!(
            fs::rename(&sidecar, &displaced).is_err(),
            "Windows bookkeeping handle unexpectedly shared delete/rename"
        );
        drop(bookkeeping);
        fs::rename(&sidecar, &displaced).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_bookkeeping_rejects_a_hardlinked_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let gallery = dir.path().join("gallery");
        fs::create_dir(&gallery).unwrap();
        let sidecar = gallery_bookkeeping_sidecar_lock_path(&gallery).unwrap();
        let other_link = dir.path().join("bookkeeping-hardlink");
        fs::write(&sidecar, b"suspicious").unwrap();
        fs::hard_link(&sidecar, &other_link).unwrap();

        let error = acquire_gallery_bookkeeping_lock(&gallery).unwrap_err();
        assert!(
            format!("{error:#}").contains("linked"),
            "unexpected Windows hardlink rejection: {error:#}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_root_gallery_boundary_has_actionable_guidance() {
        let error = gallery_bookkeeping_sidecar_lock_path(Path::new(r"C:\")).unwrap_err();
        assert!(error.to_string().contains("choose a child directory"));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_public_recovery_sweeps_current_authority_from_an_ordinary_path() {
        let dir = tempfile::tempdir().unwrap();
        let gallery = dir.path().join("gallery");
        fs::create_dir(&gallery).unwrap();
        let current = gallery.join(format!(
            "{ATTEMPT_LOCK_PREFIX}{}{ATTEMPT_LOCK_SUFFIX}",
            "e".repeat(64)
        ));
        fs::write(&current, b"stale current authority").unwrap();

        recover_transactions(&gallery, &GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();

        assert!(
            !current.exists(),
            "ordinary Windows gallery spelling did not converge on canonical sweep provenance"
        );
    }

    #[test]
    #[ignore = "subprocess helper for the cross-process reservation race test"]
    fn ordinary_reservation_drop_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let reservation = reserve_gallery_final_name_with_directory_sync(
            &output_dir,
            "ordinary-process.png",
            &|_| Ok(()),
        )
        .unwrap();
        write_process_test_marker("READY");
        let mut input = std::io::BufReader::new(std::io::stdin());
        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "CHECK");
        let lock = open_gallery_bookkeeping_lock_target(&output_dir).unwrap();
        match fs2::FileExt::try_lock_exclusive(&lock) {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                write_process_test_marker("CONTENDED");
            }
            Ok(()) => {
                fs2::FileExt::unlock(&lock).unwrap();
                write_process_test_marker("UNEXPECTEDLY_ACQUIRED");
            }
            Err(error) => panic!("unexpected gallery lock probe error: {error}"),
        }
        command.clear();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "DROP");
        write_process_test_marker("DROP_STARTED");
        drop(reservation);
        write_process_test_marker("COMPLETED");
    }

    #[test]
    #[ignore = "subprocess helper for live ordinary publication tests"]
    fn ordinary_reservation_publish_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let desired = std::env::var("MOLD_TEST_GALLERY_NAME")
            .expect("MOLD_TEST_GALLERY_NAME must be set by parent test");
        let reservation =
            reserve_gallery_final_name_with_directory_sync(&output_dir, &desired, &|_| Ok(()))
                .unwrap();
        assert_eq!(reservation.final_name(), desired);
        write_process_test_marker("READY");
        let mut input = std::io::BufReader::new(std::io::stdin());
        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        match command.trim() {
            "PUBLISH" => {
                let path = output_dir.join(reservation.final_name());
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .unwrap();
                file.write_all(b"ordinary publication").unwrap();
                file.sync_all().unwrap();
                sync_dir(&output_dir).unwrap();
                drop(reservation);
                write_process_test_marker("PUBLISHED");
            }
            "CRASH" => {
                write_process_test_marker("CRASHING");
                std::process::exit(0);
            }
            other => panic!("unexpected writer-helper command {other}"),
        }
    }

    #[test]
    #[ignore = "subprocess helper for cross-process batch-attempt liveness tests"]
    fn batch_attempt_liveness_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let state = std::env::var("MOLD_TEST_BATCH_ATTEMPT_STATE")
            .expect("MOLD_TEST_BATCH_ATTEMPT_STATE must be set by parent test");
        let mut transaction = BatchTransaction::begin(
            &output_dir,
            "live-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("live-child.png", 0)],
        )
        .unwrap();
        match state.as_str() {
            "staging" => {}
            "prepared" => {
                transaction.stage_bytes(0, b"live child").unwrap();
                transaction.mark_prepared().unwrap();
            }
            "committing" => {
                transaction.stage_bytes(0, b"live child").unwrap();
                transaction.mark_prepared().unwrap();
                transaction.manifest.state = BatchManifestState::Committing;
                transaction.persist_manifest().unwrap();
            }
            other => panic!("unexpected live batch-attempt state {other}"),
        }

        write_process_test_marker("READY");
        let mut input = std::io::BufReader::new(std::io::stdin());
        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "CRASH");
        write_process_test_marker("CRASHING");
        // Deliberately bypass Rust destructors: only the OS may release the
        // live-attempt authority, matching an abrupt server exit.
        std::process::exit(0);
    }

    #[test]
    #[ignore = "subprocess helper retaining authority after committed archive"]
    fn archived_batch_attempt_authority_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let mut transaction = BatchTransaction::begin(
            &output_dir,
            "archived-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("archived-child.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"archived child").unwrap();
        transaction.mark_prepared().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime
            .block_on(transaction.commit(&GalleryPublicationGate::default(), Arc::new(None)))
            .unwrap();
        assert!(!transaction.attempt_dir.exists());
        write_process_test_marker("ARCHIVED");

        let mut transaction = Some(transaction);
        let mut input = std::io::BufReader::new(std::io::stdin());
        loop {
            let mut command = String::new();
            input.read_line(&mut command).unwrap();
            match command.trim() {
                "BEGIN_SECOND" => {
                    write_process_test_marker("SECOND_STARTED");
                    let second = BatchTransaction::begin(
                        &output_dir,
                        "second-parent",
                        0,
                        serde_json::json!({"batch_size": 1}),
                        vec![record("second-child.png", 0)],
                    )
                    .unwrap();
                    write_process_test_marker("SECOND_ACQUIRED");
                    drop(second);
                }
                "RELEASE" => {
                    drop(transaction.take());
                    write_process_test_marker("RELEASED");
                    break;
                }
                other => panic!("unexpected archived-owner command {other}"),
            }
        }
    }

    #[test]
    #[ignore = "subprocess helper contending for an archived logical attempt"]
    fn archived_batch_attempt_contender_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let mut input = std::io::BufReader::new(std::io::stdin());
        let result = BatchTransaction::begin_with_attempt_authority_hook(
            &output_dir,
            "archived-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("archived-child.png", 0)],
            || {
                write_process_test_marker("REPLACEMENT_CREATED");
                let mut command = String::new();
                input.read_line(&mut command).unwrap();
                assert_eq!(command.trim(), "CLAIM");
                write_process_test_marker("CLAIMING");
            },
        );
        match result {
            Ok(_) => panic!("live archived owner admitted a replacement attempt"),
            Err(error) => {
                error
                    .downcast_ref::<BatchAttemptAuthorityContended>()
                    .expect("replacement begin must return typed attempt contention");
                write_process_test_marker("CONTENDED");
            }
        }

        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        if command.trim() == "EXIT" {
            return;
        }
        assert_eq!(command.trim(), "RETRY");
        let transaction = BatchTransaction::begin(
            &output_dir,
            "archived-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("archived-child.png", 0)],
        )
        .unwrap();
        write_process_test_marker("RETRY_ACQUIRED");
        command.clear();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "DROP");
        drop(transaction);
        write_process_test_marker("DROPPED");
    }

    #[test]
    fn archived_attempt_authority_survives_directory_removal_until_owner_drop() {
        let dir = tempfile::tempdir().unwrap();
        let mut owner = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::archived_batch_attempt_authority_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut owner_input = owner.stdin.take().unwrap();
        let mut owner_output = std::io::BufReader::new(owner.stdout.take().unwrap());
        read_process_test_marker(&mut owner_output, "ARCHIVED").unwrap();
        assert_eq!(
            fs::read(dir.path().join("archived-child.png")).unwrap(),
            b"archived child"
        );
        assert!(!attempt_dir(dir.path(), "archived-parent", 0).exists());

        let (mut predecessor, _predecessor_input, mut predecessor_output) =
            spawn_predecessor_authority_helper(dir.path(), "claim", "archived-parent");
        read_process_test_marker(&mut predecessor_output, "PREDECESSOR_CONTENDED").unwrap();
        let _ = std::io::copy(&mut predecessor_output, &mut std::io::sink());
        assert!(
            predecessor.wait().unwrap().success(),
            "an archived v2 owner did not retain its predecessor v1 bridge lock"
        );

        // Replacing the former per-attempt lock namespace with another real
        // directory must not fork authority. This is the exact failure mode
        // that path-local sidecars permit: the owner keeps the displaced
        // inode locked while a contender creates and locks a new inode at the
        // original pathname.
        let old_lock_namespace = dir
            .path()
            .join(TRANSACTION_DIR)
            .join(LEGACY_ATTEMPT_LOCKS_DIR);
        if !old_lock_namespace.exists() {
            fs::create_dir_all(&old_lock_namespace).unwrap();
        }
        let displaced_lock_namespace = dir.path().join("displaced-attempt-locks");
        fs::rename(&old_lock_namespace, &displaced_lock_namespace).unwrap();
        fs::create_dir(&old_lock_namespace).unwrap();

        let mut contender = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::archived_batch_attempt_contender_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut contender_input = contender.stdin.take().unwrap();
        let mut contender_output = std::io::BufReader::new(contender.stdout.take().unwrap());
        read_process_test_marker(&mut contender_output, "REPLACEMENT_CREATED").unwrap();
        writeln!(contender_input, "CLAIM").unwrap();
        contender_input.flush().unwrap();
        read_process_test_marker(&mut contender_output, "CLAIMING").unwrap();
        let (contended_tx, contended_rx) = std::sync::mpsc::channel();
        let contender_reader = std::thread::spawn(move || {
            let result = read_process_test_marker(&mut contender_output, "CONTENDED");
            contended_tx.send(result).unwrap();
            contender_output
        });
        match contended_rx.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(result) => result.unwrap(),
            Err(error) => {
                let _ = contender.kill();
                let _ = owner.kill();
                let _ = contender_reader.join();
                panic!(
                    "replacement begin blocked on live authority after its deterministic pre-claim barrier: {error}"
                );
            }
        }
        let mut contender_output = contender_reader.join().unwrap();

        writeln!(owner_input, "RELEASE").unwrap();
        owner_input.flush().unwrap();
        read_process_test_marker(&mut owner_output, "RELEASED").unwrap();
        let _ = std::io::copy(&mut owner_output, &mut std::io::sink());
        assert!(owner.wait().unwrap().success());

        writeln!(contender_input, "RETRY").unwrap();
        contender_input.flush().unwrap();
        read_process_test_marker(&mut contender_output, "RETRY_ACQUIRED").unwrap();
        writeln!(contender_input, "DROP").unwrap();
        contender_input.flush().unwrap();
        read_process_test_marker(&mut contender_output, "DROPPED").unwrap();
        let _ = std::io::copy(&mut contender_output, &mut std::io::sink());
        assert!(contender.wait().unwrap().success());

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let report = runtime
            .block_on(recover_transactions(
                dir.path(),
                &GalleryPublicationGate::default(),
                Arc::new(None),
            ))
            .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert_eq!(
            fs::read(dir.path().join("archived-child.png")).unwrap(),
            b"archived child"
        );
    }

    #[test]
    fn contended_replacement_never_deadlocks_gallery_bookkeeping() {
        let dir = tempfile::tempdir().unwrap();
        let mut owner = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::archived_batch_attempt_authority_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut owner_input = owner.stdin.take().unwrap();
        let mut owner_output = std::io::BufReader::new(owner.stdout.take().unwrap());
        read_process_test_marker(&mut owner_output, "ARCHIVED").unwrap();

        let mut contender = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::archived_batch_attempt_contender_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut contender_input = contender.stdin.take().unwrap();
        let mut contender_output = std::io::BufReader::new(contender.stdout.take().unwrap());
        read_process_test_marker(&mut contender_output, "REPLACEMENT_CREATED").unwrap();

        writeln!(owner_input, "BEGIN_SECOND").unwrap();
        owner_input.flush().unwrap();
        read_process_test_marker(&mut owner_output, "SECOND_STARTED").unwrap();
        writeln!(contender_input, "CLAIM").unwrap();
        contender_input.flush().unwrap();
        read_process_test_marker(&mut contender_output, "CLAIMING").unwrap();

        let (contended_tx, contended_rx) = std::sync::mpsc::channel();
        let contender_reader = std::thread::spawn(move || {
            let result = read_process_test_marker(&mut contender_output, "CONTENDED");
            contended_tx.send(result).unwrap();
            contender_output
        });
        match contended_rx.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(result) => result.unwrap(),
            Err(error) => {
                let _ = contender.kill();
                let _ = owner.kill();
                let _ = contender_reader.join();
                panic!(
                    "replacement authority and gallery bookkeeping formed a cross-process lock cycle: {error}"
                );
            }
        }
        let mut contender_output = contender_reader.join().unwrap();
        read_process_test_marker(&mut owner_output, "SECOND_ACQUIRED").unwrap();

        writeln!(contender_input, "EXIT").unwrap();
        contender_input.flush().unwrap();
        let _ = std::io::copy(&mut contender_output, &mut std::io::sink());
        assert!(contender.wait().unwrap().success());
        writeln!(owner_input, "RELEASE").unwrap();
        owner_input.flush().unwrap();
        read_process_test_marker(&mut owner_output, "RELEASED").unwrap();
        let _ = std::io::copy(&mut owner_output, &mut std::io::sink());
        assert!(owner.wait().unwrap().success());
    }

    #[test]
    fn recovery_skips_live_batch_attempts_and_claims_them_after_process_crash() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        for (state, expected_state) in [
            ("staging", BatchManifestState::Staging),
            ("prepared", BatchManifestState::Prepared),
            ("committing", BatchManifestState::Committing),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut writer = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "batch_transaction::tests::batch_attempt_liveness_process_helper",
                    "--nocapture",
                ])
                .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
                .env("MOLD_TEST_BATCH_ATTEMPT_STATE", state)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let mut writer_input = writer.stdin.take().unwrap();
            let mut writer_output = std::io::BufReader::new(writer.stdout.take().unwrap());
            read_process_test_marker(&mut writer_output, "READY").unwrap();

            let live_report = runtime
                .block_on(recover_transactions(
                    dir.path(),
                    &GalleryPublicationGate::default(),
                    Arc::new(None),
                ))
                .unwrap();
            assert_eq!(
                live_report,
                RecoveryReport::default(),
                "recovery mutated a live {state} attempt"
            );
            let live_attempt = attempt_dir(dir.path(), "live-parent", 0);
            let live_manifest: BatchAttemptManifest =
                serde_json::from_slice(&fs::read(live_attempt.join(MANIFEST_FILE)).unwrap())
                    .unwrap();
            assert_eq!(live_manifest.state, expected_state);
            assert!(
                !dir.path().join("live-child.png").exists(),
                "recovery exposed a final file for a live {state} attempt"
            );
            assert!(
                reservation_path(dir.path(), "live-child.png").is_file(),
                "recovery released the live {state} attempt's reservation"
            );

            writeln!(writer_input, "CRASH").unwrap();
            writer_input.flush().unwrap();
            read_process_test_marker(&mut writer_output, "CRASHING").unwrap();
            let _ = std::io::copy(&mut writer_output, &mut std::io::sink());
            assert!(writer.wait().unwrap().success());

            let orphan_report = runtime
                .block_on(recover_transactions(
                    dir.path(),
                    &GalleryPublicationGate::default(),
                    Arc::new(None),
                ))
                .unwrap();
            if state == "committing" {
                assert_eq!(orphan_report.rolled_forward, 1);
                assert_eq!(
                    fs::read(dir.path().join("live-child.png")).unwrap(),
                    b"live child"
                );
            } else {
                assert_eq!(orphan_report.rolled_back, 1);
                assert!(
                    !dir.path().join("live-child.png").exists(),
                    "orphaned {state} attempt unexpectedly published"
                );
            }
            assert!(!live_attempt.exists());
            assert!(!reservation_path(dir.path(), "live-child.png").exists());
        }
    }

    #[test]
    #[ignore = "subprocess helper pausing after authoritative recovery claim"]
    fn paused_batch_recovery_claim_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let bookkeeping = acquire_gallery_bookkeeping_lock(&output_dir).unwrap();
        let root = output_dir.join(TRANSACTION_DIR);
        let mut claimed = Vec::new();
        collect_claimed_attempts(&root, &mut claimed, &bookkeeping).unwrap();
        assert_eq!(claimed.len(), 1);
        drop(bookkeeping);
        write_process_test_marker("CLAIMED");

        let mut input = std::io::BufReader::new(std::io::stdin());
        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "CONTINUE");
        drop(claimed);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let report = runtime
            .block_on(recover_transactions(
                &output_dir,
                &GalleryPublicationGate::default(),
                Arc::new(None),
            ))
            .unwrap();
        assert_eq!(report.rolled_back, 1);
        write_process_test_marker("COMPLETED");
    }

    #[test]
    fn simultaneous_recovery_skips_an_attempt_claimed_by_another_process() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "recovery-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![record("recovery-child.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"orphaned child").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let mut first = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::paused_batch_recovery_claim_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut first_input = first.stdin.take().unwrap();
        let mut first_output = std::io::BufReader::new(first.stdout.take().unwrap());
        read_process_test_marker(&mut first_output, "CLAIMED").unwrap();

        let mut second = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::reservation_recovery_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut second_output = std::io::BufReader::new(second.stdout.take().unwrap());
        read_process_test_marker(&mut second_output, "RECOVERY_STARTED").unwrap();
        read_process_test_marker(&mut second_output, "RECOVERY_COMPLETED").unwrap();
        let _ = std::io::copy(&mut second_output, &mut std::io::sink());
        assert!(second.wait().unwrap().success());
        assert!(transaction.attempt_dir.exists());
        assert!(reservation_path(dir.path(), "recovery-child.png").is_file());
        assert!(!dir.path().join("recovery-child.png").exists());

        writeln!(first_input, "CONTINUE").unwrap();
        first_input.flush().unwrap();
        read_process_test_marker(&mut first_output, "COMPLETED").unwrap();
        let _ = std::io::copy(&mut first_output, &mut std::io::sink());
        assert!(first.wait().unwrap().success());

        assert!(!transaction.attempt_dir.exists());
        assert!(!reservation_path(dir.path(), "recovery-child.png").exists());
        assert!(!dir.path().join("recovery-child.png").exists());
    }

    #[test]
    #[ignore = "subprocess helper for cross-process recovery tests"]
    fn reservation_recovery_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        write_process_test_marker("RECOVERY_STARTED");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let report = runtime
            .block_on(recover_transactions(
                &output_dir,
                &GalleryPublicationGate::default(),
                Arc::new(None),
            ))
            .unwrap();
        assert_eq!(report, RecoveryReport::default());
        write_process_test_marker("RECOVERY_COMPLETED");
    }

    #[test]
    #[ignore = "subprocess helper for blocked ordinary reservation tests"]
    fn ordinary_reservation_contender_process_helper() {
        let output_dir = PathBuf::from(
            std::env::var_os("MOLD_TEST_GALLERY_OUTPUT")
                .expect("MOLD_TEST_GALLERY_OUTPUT must be set by parent test"),
        );
        let desired = std::env::var("MOLD_TEST_GALLERY_NAME")
            .expect("MOLD_TEST_GALLERY_NAME must be set by parent test");
        let expected = std::env::var("MOLD_TEST_GALLERY_EXPECTED_NAME")
            .expect("MOLD_TEST_GALLERY_EXPECTED_NAME must be set by parent test");
        write_process_test_marker("RESERVATION_STARTED");
        let reservation =
            reserve_gallery_final_name_with_directory_sync(&output_dir, &desired, &|_| Ok(()))
                .unwrap();
        assert_eq!(reservation.final_name(), expected);
        write_process_test_marker("RESERVATION_ACQUIRED");

        let mut input = std::io::BufReader::new(std::io::stdin());
        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "RELEASE");
        drop(reservation);
        write_process_test_marker("RESERVATION_RELEASED");
    }

    #[test]
    fn recovery_waits_for_live_ordinary_publication_authority() {
        let dir = tempfile::tempdir().unwrap();
        let desired = "live-publication.png";
        let mut writer = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::ordinary_reservation_publish_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .env("MOLD_TEST_GALLERY_NAME", desired)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut writer_input = writer.stdin.take().unwrap();
        let mut writer_output = std::io::BufReader::new(writer.stdout.take().unwrap());
        read_process_test_marker(&mut writer_output, "READY").unwrap();
        assert!(reservation_path(dir.path(), desired).is_file());

        let mut recovery = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::reservation_recovery_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut recovery_output = std::io::BufReader::new(recovery.stdout.take().unwrap());
        read_process_test_marker(&mut recovery_output, "RECOVERY_STARTED").unwrap();
        let (recovery_completed_tx, recovery_completed_rx) = std::sync::mpsc::channel();
        let recovery_reader = std::thread::spawn(move || {
            let result = read_process_test_marker(&mut recovery_output, "RECOVERY_COMPLETED");
            recovery_completed_tx.send(result).unwrap();
            let _ = std::io::copy(&mut recovery_output, &mut std::io::sink());
        });
        let recovery_completed_while_writer_was_live = recovery_completed_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_ok();

        let mut contender = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::ordinary_reservation_contender_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .env("MOLD_TEST_GALLERY_NAME", desired)
            .env("MOLD_TEST_GALLERY_EXPECTED_NAME", "live-publication-1.png")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut contender_input = contender.stdin.take().unwrap();
        let mut contender_output = std::io::BufReader::new(contender.stdout.take().unwrap());
        read_process_test_marker(&mut contender_output, "RESERVATION_STARTED").unwrap();
        let (contender_acquired_tx, contender_acquired_rx) = std::sync::mpsc::channel();
        let contender_reader = std::thread::spawn(move || {
            let result = read_process_test_marker(&mut contender_output, "RESERVATION_ACQUIRED");
            contender_acquired_tx.send(result).unwrap();
            contender_output
        });
        let contender_acquired_while_writer_was_live = contender_acquired_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_ok();

        writeln!(writer_input, "PUBLISH").unwrap();
        writer_input.flush().unwrap();
        read_process_test_marker(&mut writer_output, "PUBLISHED").unwrap();
        let _ = std::io::copy(&mut writer_output, &mut std::io::sink());
        assert!(writer.wait().unwrap().success());

        if !contender_acquired_while_writer_was_live {
            contender_acquired_rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("third process did not reserve after ordinary publication")
                .unwrap();
        }
        let mut contender_output = contender_reader.join().unwrap();
        writeln!(contender_input, "RELEASE").unwrap();
        contender_input.flush().unwrap();
        read_process_test_marker(&mut contender_output, "RESERVATION_RELEASED").unwrap();
        let _ = std::io::copy(&mut contender_output, &mut std::io::sink());
        assert!(contender.wait().unwrap().success());

        if !recovery_completed_while_writer_was_live {
            recovery_completed_rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("recovery did not continue after ordinary publication")
                .unwrap();
        }
        recovery_reader.join().unwrap();
        assert!(recovery.wait().unwrap().success());

        assert!(
            !recovery_completed_while_writer_was_live,
            "recovery swept a live ordinary reservation"
        );
        assert!(
            !contender_acquired_while_writer_was_live,
            "a third process reserved a gallery name while the writer was live"
        );
        assert_eq!(
            fs::read(dir.path().join(desired)).unwrap(),
            b"ordinary publication"
        );
        assert!(!reservation_path(dir.path(), desired).exists());
        let next = reserve_gallery_final_name_with_directory_sync(dir.path(), desired, &|_| Ok(()))
            .unwrap();
        assert_eq!(next.final_name(), "live-publication-1.png");
        drop(next);
    }

    #[test]
    fn recovery_reclaims_reservation_after_writer_process_crash() {
        let dir = tempfile::tempdir().unwrap();
        let desired = "crashed-publication.png";
        let mut writer = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::ordinary_reservation_publish_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .env("MOLD_TEST_GALLERY_NAME", desired)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut writer_input = writer.stdin.take().unwrap();
        let mut writer_output = std::io::BufReader::new(writer.stdout.take().unwrap());
        read_process_test_marker(&mut writer_output, "READY").unwrap();
        assert!(reservation_path(dir.path(), desired).is_file());

        let mut recovery = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::reservation_recovery_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut recovery_output = std::io::BufReader::new(recovery.stdout.take().unwrap());
        read_process_test_marker(&mut recovery_output, "RECOVERY_STARTED").unwrap();
        let (recovery_completed_tx, recovery_completed_rx) = std::sync::mpsc::channel();
        let recovery_reader = std::thread::spawn(move || {
            let result = read_process_test_marker(&mut recovery_output, "RECOVERY_COMPLETED");
            recovery_completed_tx.send(result).unwrap();
            let _ = std::io::copy(&mut recovery_output, &mut std::io::sink());
        });
        assert!(
            matches!(
                recovery_completed_rx.recv_timeout(std::time::Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ),
            "recovery bypassed a live crashed-writer candidate"
        );

        writeln!(writer_input, "CRASH").unwrap();
        writer_input.flush().unwrap();
        read_process_test_marker(&mut writer_output, "CRASHING").unwrap();
        let _ = std::io::copy(&mut writer_output, &mut std::io::sink());
        assert!(writer.wait().unwrap().success());

        recovery_completed_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("recovery did not continue after the writer process exited")
            .unwrap();
        recovery_reader.join().unwrap();
        assert!(recovery.wait().unwrap().success());

        assert!(
            !reservation_path(dir.path(), desired).exists(),
            "recovery leaked the crashed writer's stale reservation"
        );
        let replacement =
            reserve_gallery_final_name_with_directory_sync(dir.path(), desired, &|_| Ok(()))
                .unwrap();
        assert_eq!(replacement.final_name(), desired);
        drop(replacement);
        let transaction_entries = fs::read_dir(dir.path().join(TRANSACTION_DIR))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .collect::<Vec<_>>();
        assert_eq!(
            transaction_entries,
            vec![std::ffi::OsString::from(
                crate::gallery_authority::authority_dir_name()
            )],
            "recovery left mutable attempt/reservation bookkeeping behind"
        );
    }

    #[test]
    fn ordinary_reservation_drop_is_serialized_across_processes() {
        let dir = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::ordinary_reservation_drop_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut child_input = child.stdin.take().unwrap();
        let mut child_output = std::io::BufReader::new(child.stdout.take().unwrap());
        read_process_test_marker(&mut child_output, "READY").unwrap();

        let output_dir = dir.path().to_path_buf();
        let (begin_paused_tx, begin_paused_rx) = std::sync::mpsc::channel();
        let (continue_begin_tx, continue_begin_rx) = std::sync::mpsc::channel();
        let begin = std::thread::spawn(move || {
            BatchTransaction::begin_with_reservation_directory_hook(
                &output_dir,
                "parent",
                0,
                serde_json::json!({"batch_size": 1}),
                vec![record("batch-process.png", 0)],
                move || {
                    begin_paused_tx.send(()).unwrap();
                    continue_begin_rx.recv().unwrap();
                },
            )
        });
        assert!(
            matches!(
                begin_paused_rx.recv_timeout(std::time::Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ),
            "batch begin bypassed the child process's live reservation authority"
        );
        writeln!(child_input, "CHECK").unwrap();
        child_input.flush().unwrap();
        read_process_test_marker(&mut child_output, "CONTENDED").unwrap();
        writeln!(child_input, "DROP").unwrap();
        child_input.flush().unwrap();
        read_process_test_marker(&mut child_output, "DROP_STARTED").unwrap();
        read_process_test_marker(&mut child_output, "COMPLETED").unwrap();

        begin_paused_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("batch begin did not continue after child reservation drop");
        continue_begin_tx.send(()).unwrap();

        let transaction = begin.join().unwrap();
        // Keep the pipe open and drain libtest's trailing result output;
        // closing stdout immediately after the marker makes the helper process
        // report a BrokenPipe even though the test passed.
        let _ = std::io::copy(&mut child_output, &mut std::io::sink());
        let status = child.wait().unwrap();
        assert!(status.success(), "reservation-drop helper failed: {status}");
        let transaction =
            transaction.expect("cross-process cleanup must not break batch reservation");
        assert!(reservation_path(dir.path(), "batch-process.png").is_file());
        assert!(
            !reservation_path(dir.path(), "ordinary-process.png").exists(),
            "cross-process ordinary reservation bookkeeping leaked"
        );
        transaction.release_reservations();
        remove_failed_attempt(&transaction);
    }

    #[cfg(unix)]
    #[test]
    fn canonical_sidecar_path_collapses_filesystem_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let canonical_gallery = dir.path().join("gallery");
        fs::create_dir(&canonical_gallery).unwrap();
        let gallery_alias = dir.path().join("gallery-alias");
        std::os::unix::fs::symlink(&canonical_gallery, &gallery_alias).unwrap();

        assert_eq!(
            gallery_bookkeeping_sidecar_lock_path(&canonical_gallery).unwrap(),
            gallery_bookkeeping_sidecar_lock_path(&gallery_alias).unwrap(),
            "sidecar locking must retain one authority across filesystem aliases"
        );
    }

    #[cfg(unix)]
    #[test]
    fn attempt_authority_path_collapses_gallery_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let gallery = dir.path().join("gallery");
        fs::create_dir(&gallery).unwrap();
        let transaction = BatchTransaction::begin(
            &gallery,
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let gallery_alias = dir.path().join("gallery-alias");
        std::os::unix::fs::symlink(&gallery, &gallery_alias).unwrap();

        assert_eq!(
            attempt_authority_lock_path(&transaction.attempt_dir).unwrap(),
            attempt_authority_lock_path(&attempt_dir(&gallery_alias, "parent", 0)).unwrap(),
            "one logical attempt must retain one sidecar across gallery aliases"
        );
        let authority = attempt_authority_lock_path(&transaction.attempt_dir).unwrap();
        assert_eq!(
            authority.parent(),
            Some(fs::canonicalize(&gallery).unwrap().as_path()),
            "authority must be rooted directly in the stable canonical gallery namespace"
        );
        assert!(
            is_attempt_authority_file_name(authority.file_name().unwrap().to_str().unwrap()),
            "authority filename must use the exact reserved hash shape"
        );
        let bookkeeping = acquire_gallery_bookkeeping_lock(&gallery).unwrap();
        assert_eq!(
            legacy_attempt_authority_lock_path(&transaction.attempt_dir, &bookkeeping).unwrap(),
            predecessor_attempt_authority_path(&transaction.attempt_dir),
            "the rolling-overlap bridge must reproduce the predecessor v1 hash and pathname"
        );
    }

    #[test]
    fn authority_sweep_never_deletes_unrelated_hidden_gallery_files() {
        let dir = tempfile::tempdir().unwrap();
        let near_match = dir.path().join(".mold-batch-attempt-not-ours.lock");
        let non_hex = dir.path().join(format!(
            "{ATTEMPT_LOCK_PREFIX}{}{ATTEMPT_LOCK_SUFFIX}",
            "g".repeat(64)
        ));
        let legacy_unrelated = dir
            .path()
            .join(TRANSACTION_DIR)
            .join(LEGACY_ATTEMPT_LOCKS_DIR)
            .join("operator-note");
        fs::write(&near_match, b"user data").unwrap();
        fs::write(&non_hex, b"other user data").unwrap();

        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        fs::write(&legacy_unrelated, b"operator data").unwrap();
        drop(transaction);

        assert_eq!(fs::read(&near_match).unwrap(), b"user data");
        assert_eq!(fs::read(&non_hex).unwrap(), b"other user data");
        assert_eq!(fs::read(&legacy_unrelated).unwrap(), b"operator data");
    }

    #[cfg(unix)]
    #[test]
    fn attempt_authority_rejects_an_attempt_directory_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let attempts = dir
            .path()
            .join(TRANSACTION_DIR)
            .join("parent")
            .join("attempts");
        fs::create_dir_all(&attempts).unwrap();
        let outside = dir.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let escaped = attempts.join("0");
        std::os::unix::fs::symlink(&outside, &escaped).unwrap();

        let error = attempt_authority_lock_path(&escaped).unwrap_err();
        assert!(error.to_string().contains("escapes its attempts root"));
    }

    #[cfg(unix)]
    #[test]
    fn attempt_authority_open_rejects_final_file_symlink_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();
        let outside = dir.path().join("outside-file");
        fs::write(&outside, b"outside").unwrap();
        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();

        let result = open_attempt_authority_target_with_hook(
            &transaction.attempt_dir,
            &bookkeeping,
            |lock_path| {
                fs::remove_file(lock_path).unwrap();
                std::os::unix::fs::symlink(&outside, lock_path).unwrap();
            },
        );

        assert!(
            result.is_err(),
            "authority open followed a final-component symlink installed after validation"
        );
        assert_eq!(fs::read(outside).unwrap(), b"outside");
    }

    #[cfg(unix)]
    #[test]
    fn attempt_authority_open_rejects_final_file_regular_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();
        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();

        let result = open_attempt_authority_target_with_hook(
            &transaction.attempt_dir,
            &bookkeeping,
            |lock_path| {
                fs::remove_file(lock_path).unwrap();
                fs::write(lock_path, b"replacement").unwrap();
            },
        );

        assert!(
            result.is_err(),
            "authority open accepted a different regular-file inode installed after open"
        );
    }

    #[cfg(unix)]
    #[test]
    fn attempt_authority_claim_rejects_regular_replacement_after_lock() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let result = try_claim_attempt_authority_with_post_lock_hook(
            &transaction.attempt_dir,
            &bookkeeping,
            |lock_path| {
                fs::remove_file(lock_path).unwrap();
                fs::write(lock_path, b"post-lock replacement").unwrap();
            },
        );

        assert!(
            result.is_err(),
            "authority claim accepted a different inode installed after locking"
        );
    }

    #[cfg(unix)]
    #[test]
    fn legacy_lock_directory_symlink_replacement_cannot_redirect_authority() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let outside = dir.path().join("outside-directory");
        fs::create_dir(&outside).unwrap();
        let displaced = dir.path().join("displaced-locks");
        let legacy = dir
            .path()
            .join(TRANSACTION_DIR)
            .join(LEGACY_ATTEMPT_LOCKS_DIR);
        fs::rename(&legacy, &displaced).unwrap();
        std::os::unix::fs::symlink(&outside, &legacy).unwrap();

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        assert!(
            try_claim_attempt_authority(&transaction.attempt_dir, &bookkeeping).is_err(),
            "a replaced legacy namespace was not rejected before opening authority"
        );
        assert_eq!(
            fs::read_dir(&outside).unwrap().count(),
            0,
            "authority claim created a lock file outside the gallery root"
        );
    }

    #[test]
    fn terminal_attempt_drop_reclaims_current_and_retains_stable_legacy_authority() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let lock_path = attempt_authority_lock_path(&transaction.attempt_dir).unwrap();
        let legacy_path = predecessor_attempt_authority_path(&transaction.attempt_dir);
        transaction.release_reservations();
        remove_failed_attempt(&transaction);
        drop(transaction);

        assert!(
            !lock_path.exists(),
            "terminal batch attempts must not retain one authority file forever"
        );
        assert!(
            legacy_path.is_file(),
            "rolling compatibility must retain one reusable predecessor authority inode"
        );
    }

    #[test]
    fn existing_live_attempt_returns_typed_contention() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();

        let error = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("two.png", 0)],
        )
        .unwrap_err();
        error
            .downcast_ref::<BatchAttemptAuthorityContended>()
            .expect("a live existing attempt must return typed authority contention");

        drop(transaction);
    }

    #[test]
    fn dropped_unfinished_attempt_retains_recoverable_evidence_without_authority_leak() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"unfinished").unwrap();
        let lock_path = attempt_authority_lock_path(&transaction.attempt_dir).unwrap();
        drop(transaction);
        assert!(
            !lock_path.exists(),
            "Drop leaked an unlocked authority path"
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let report = runtime
            .block_on(recover_transactions(
                dir.path(),
                &GalleryPublicationGate::default(),
                Arc::new(None),
            ))
            .unwrap();
        assert_eq!(report.rolled_back, 1);
        assert!(!attempt_dir(dir.path(), "parent", 0).exists());
        assert!(
            dir.path()
                .join(TRANSACTION_DIR)
                .join(LEGACY_ATTEMPT_LOCKS_DIR)
                .is_dir(),
            "recovery must retain the rolling predecessor authority namespace"
        );
    }

    #[test]
    fn recovery_retains_noncanonical_attempt_entries_with_operator_guidance() {
        let dir = tempfile::tempdir().unwrap();
        let stray = dir
            .path()
            .join(TRANSACTION_DIR)
            .join("parent")
            .join("attempts")
            .join("not-a-generation");
        fs::create_dir_all(&stray).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let error = runtime
            .block_on(recover_transactions(
                dir.path(),
                &GalleryPublicationGate::default(),
                Arc::new(None),
            ))
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("canonical numeric generation"));
        assert!(message.contains("move it outside"));
        assert!(
            stray.is_dir(),
            "fail-closed recovery removed unknown evidence"
        );
    }

    fn journal_post_publish_snapshot(transaction: &mut BatchTransaction, bytes: &[u8]) {
        transaction.stage_bytes(0, bytes).unwrap();
        transaction.mark_prepared().unwrap();
        transaction.manifest.state = BatchManifestState::Committing;
        transaction.persist_manifest().unwrap();
        let final_path = transaction
            .output_dir
            .join(&transaction.manifest.children[0].final_name);
        fs::hard_link(transaction.staging_path(0).unwrap(), &final_path).unwrap();
        File::open(&final_path).unwrap().sync_all().unwrap();
        transaction.manifest.children[0]
            .record
            .stat_from_disk(&final_path);
        transaction
            .append_journal(BatchJournalEvent::FinalPublished { child_index: 0 })
            .unwrap();
        transaction.persist_manifest().unwrap();
    }

    fn commit_journal_without_cleanup(transaction: &mut BatchTransaction, bytes: &[u8]) {
        journal_post_publish_snapshot(transaction, bytes);
        transaction
            .append_journal(BatchJournalEvent::MetadataCommitted)
            .unwrap();
        transaction.manifest.state = BatchManifestState::Committed;
        transaction.persist_manifest().unwrap();
    }

    #[tokio::test]
    async fn db_disabled_commit_publishes_all_children_and_durable_central_authority() {
        let dir = tempfile::tempdir().unwrap();
        let gate = GalleryPublicationGate::default();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({"batch_size": 2}),
            vec![record("same.png", 0), record("other.png", 1)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"first").unwrap();
        transaction.stage_bytes(1, b"second").unwrap();
        transaction.mark_prepared().unwrap();

        transaction.commit(&gate, Arc::new(None)).await.unwrap();

        assert_eq!(fs::read(dir.path().join("same.png")).unwrap(), b"first");
        assert_eq!(fs::read(dir.path().join("other.png")).unwrap(), b"second");
        assert_eq!(transaction.manifest().state, BatchManifestState::Committed);
        assert!(!transaction.attempt_dir.exists());
        let index = gate.committed_archive_index(dir.path()).unwrap();
        assert!(index.get("same.png").is_some());
        assert!(index.get("other.png").is_some());
        assert!(
            !committed_manifests_dir(dir.path(), "parent")
                .join("0.json")
                .exists(),
            "legacy recovery evidence must be collected after two central checkpoints"
        );
    }

    #[tokio::test]
    async fn db_enabled_restart_heals_exact_rows_from_a_db_disabled_archive() {
        let dir = tempfile::tempdir().unwrap();
        let original = record("archived.webp", 7);
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "db-disabled-parent",
            0,
            serde_json::json!({"batch_size": 1}),
            vec![original.clone()],
        )
        .unwrap();
        transaction.stage_bytes(0, b"archived webp bytes").unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        drop(transaction);

        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let restarted_gate = GalleryPublicationGate::default();
        recover_transactions(dir.path(), &restarted_gate, db.clone())
            .await
            .unwrap();

        let healed = db
            .as_ref()
            .as_ref()
            .unwrap()
            .get(dir.path(), "archived.webp")
            .unwrap()
            .expect("archive row is restored when SQLite is enabled");
        assert_eq!(healed.metadata, original.metadata);
        assert!(!healed.metadata_synthetic);
        assert_eq!(
            restarted_gate
                .committed_archive_index(dir.path())
                .unwrap()
                .get("archived.webp")
                .unwrap()
                .record()
                .metadata,
            original.metadata
        );
        assert!(
            !committed_manifests_dir(dir.path(), "db-disabled-parent")
                .join("0.json")
                .exists(),
            "two durable central checkpoints must collect migrated per-batch authority"
        );
    }

    #[tokio::test]
    async fn legacy_import_rejects_future_state_but_isolates_a_malformed_archive() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        let original = transaction.manifest().clone();
        drop(transaction);
        let archive = committed_manifests_dir(dir.path(), "parent").join("0.json");
        fs::remove_dir_all(
            dir.path()
                .join(TRANSACTION_DIR)
                .join(crate::gallery_authority::authority_dir_name()),
        )
        .unwrap();
        atomic_write_json(&archive, &original).unwrap();

        let mut future = original.clone();
        future.version = MANIFEST_VERSION + 1;
        atomic_write_json(&archive, &future).unwrap();
        let error = load_committed_archive_index(dir.path()).unwrap_err();
        assert!(
            format!("{error:#}").contains("unsupported batch manifest version"),
            "future archive rejection was not actionable: {error:#}"
        );

        let mut noncommitted = original.clone();
        noncommitted.state = BatchManifestState::Prepared;
        atomic_write_json(&archive, &noncommitted).unwrap();
        let error = load_committed_archive_index(dir.path()).unwrap_err();
        assert!(
            format!("{error:#}").contains("archived batch manifest is not committed")
                || format!("{error:#}").contains("non-staging/non-failed"),
            "noncommitted archive rejection was not actionable: {error:#}"
        );

        fs::write(&archive, b"{malformed").unwrap();
        let gate = GalleryPublicationGate::default();
        recover_transactions(dir.path(), &gate, Arc::new(None))
            .await
            .unwrap();
        assert!(
            gate.committed_archive_index(dir.path())
                .unwrap()
                .get("one.png")
                .is_none(),
            "a malformed legacy archive must be isolated without inventing metadata"
        );
    }

    #[tokio::test]
    async fn committed_archive_index_ignores_active_attempts_and_quarantines_duplicate_live_names()
    {
        let dir = tempfile::tempdir().unwrap();
        let mut committed = BatchTransaction::begin(
            dir.path(),
            "committed-parent",
            0,
            serde_json::json!({}),
            vec![record("same.png", 0)],
        )
        .unwrap();
        committed.stage_bytes(0, b"committed").unwrap();
        committed.mark_prepared().unwrap();
        committed
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        let original = committed.manifest().clone();
        drop(committed);

        let active = BatchTransaction::begin(
            dir.path(),
            "active-parent",
            0,
            serde_json::json!({}),
            vec![record("active.png", 0)],
        )
        .unwrap();
        let index = load_committed_archive_index(dir.path()).unwrap();
        assert!(index.get("same.png").is_some());
        assert!(
            index.get("active.png").is_none(),
            "an unpublished active attempt entered committed archive authority"
        );
        drop(active);

        let original_archive =
            committed_manifests_dir(dir.path(), "committed-parent").join("0.json");
        atomic_write_json(&original_archive, &original).unwrap();
        let mut duplicate = original;
        duplicate.parent_id = "duplicate-parent".into();
        let duplicate_archive =
            committed_manifests_dir(dir.path(), "duplicate-parent").join("0.json");
        atomic_write_json(&duplicate_archive, &duplicate).unwrap();
        fs::remove_dir_all(
            dir.path()
                .join(TRANSACTION_DIR)
                .join(crate::gallery_authority::authority_dir_name()),
        )
        .unwrap();
        let index = load_committed_archive_index(dir.path()).unwrap();
        assert!(index.get("same.png").is_none());
        assert!(index.is_quarantined("same.png"));
    }

    #[tokio::test]
    async fn archive_child_tombstone_finishes_delete_preserves_sibling_and_allows_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "first-parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0), record("sibling.png", 1)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"first bytes").unwrap();
        transaction.stage_bytes(1, b"sibling bytes").unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        drop(transaction);

        let delete_gate = GalleryPublicationGate::default();
        assert_eq!(
            tombstone_committed_archive_filename(dir.path(), "one.png", &delete_gate).unwrap(),
            ArchiveDeleteDisposition::SafeToUnlink
        );
        assert!(
            !dir.path().join("one.png").exists(),
            "delete must unlink the revalidated exact identity while still holding cross-process authority"
        );
        let index = reconcile_committed_archive_index(dir.path()).unwrap();
        assert!(!dir.path().join("one.png").exists());
        assert!(index.get("one.png").is_none());
        assert!(index.is_retired("one.png"));
        assert_eq!(
            index.get("sibling.png").unwrap().record().metadata.prompt,
            "prompt 1"
        );

        let mut replacement = BatchTransaction::begin(
            dir.path(),
            "replacement-parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 9)],
        )
        .unwrap();
        assert_eq!(replacement.manifest.children[0].final_name, "one.png");
        replacement.stage_bytes(0, b"replacement bytes").unwrap();
        replacement.mark_prepared().unwrap();
        replacement
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        drop(replacement);

        let index = load_committed_archive_index(dir.path()).unwrap();
        assert_eq!(
            index.get("one.png").unwrap().record().metadata.prompt,
            "prompt 9"
        );
        assert_eq!(
            index.get("sibling.png").unwrap().record().metadata.prompt,
            "prompt 1"
        );
        assert!(!index.is_retired("one.png"));
    }

    #[tokio::test]
    async fn startup_quarantines_an_externally_missing_file_without_tombstoning_or_deleting_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("removed.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"removed bytes").unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        drop(transaction);
        fs::remove_file(dir.path().join("removed.png")).unwrap();
        sync_dir(dir.path()).unwrap();

        let gate = GalleryPublicationGate::default();
        recover_transactions(dir.path(), &gate, Arc::new(None))
            .await
            .unwrap();
        let index = gate.committed_archive_index(dir.path()).unwrap();
        assert!(index.get("removed.png").is_none());
        assert!(
            !index.is_retired("removed.png"),
            "external absence is not user-delete authority"
        );
        fs::write(dir.path().join("removed.png"), b"removed bytes").unwrap();
        sync_dir(dir.path()).unwrap();
        let restarted_gate = GalleryPublicationGate::default();
        recover_transactions(dir.path(), &restarted_gate, Arc::new(None))
            .await
            .unwrap();
        assert!(
            dir.path().join("removed.png").is_file(),
            "restoring externally removed bytes must never trigger delayed deletion"
        );
        assert!(
            restarted_gate
                .committed_archive_index(dir.path())
                .unwrap()
                .get("removed.png")
                .is_some(),
            "matching restored bytes should reactivate their metadata authority"
        );
    }

    #[tokio::test]
    async fn an_existing_gate_observes_an_archive_committed_by_an_independent_gate() {
        let dir = tempfile::tempdir().unwrap();
        let first_gate = GalleryPublicationGate::default();
        let second_gate = GalleryPublicationGate::default();

        let mut first = BatchTransaction::begin(
            dir.path(),
            "first",
            0,
            serde_json::json!({}),
            vec![record("first.png", 0)],
        )
        .unwrap();
        first.stage_bytes(0, b"first").unwrap();
        first.mark_prepared().unwrap();
        first.commit(&first_gate, Arc::new(None)).await.unwrap();
        assert!(first_gate
            .committed_archive_index(dir.path())
            .unwrap()
            .get("first.png")
            .is_some());

        let mut second = BatchTransaction::begin(
            dir.path(),
            "second",
            0,
            serde_json::json!({}),
            vec![record("second.png", 1)],
        )
        .unwrap();
        second.stage_bytes(0, b"second").unwrap();
        second.mark_prepared().unwrap();
        second.commit(&second_gate, Arc::new(None)).await.unwrap();

        assert!(
            first_gate
                .committed_archive_index(dir.path())
                .unwrap()
                .get("second.png")
                .is_some(),
            "a process-local cache must be invalidated by another process's durable mutation"
        );
    }

    #[tokio::test]
    async fn a_warm_gate_observes_a_batch_committed_by_another_process() {
        let dir = tempfile::tempdir().unwrap();
        let gate = GalleryPublicationGate::default();
        assert!(gate
            .committed_archive_index(dir.path())
            .unwrap()
            .entries
            .is_empty());

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "batch_transaction::tests::gallery_authority_process_helper",
                "--nocapture",
            ])
            .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
            .env("MOLD_TEST_GALLERY_ACTION", "batch")
            .status()
            .unwrap();
        assert!(status.success());
        assert!(
            gate.committed_archive_index(dir.path())
                .unwrap()
                .get("subprocess.png")
                .is_some(),
            "generation fencing must invalidate another process's warm cache"
        );
    }

    #[test]
    fn concurrent_processes_replay_one_named_publication_without_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let spawn = || {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "batch_transaction::tests::gallery_authority_process_helper",
                    "--nocapture",
                ])
                .env("MOLD_TEST_GALLERY_OUTPUT", dir.path())
                .env("MOLD_TEST_GALLERY_ACTION", "named")
                .spawn()
                .unwrap()
        };
        let mut first = spawn();
        let mut second = spawn();
        assert!(first.wait().unwrap().success());
        assert!(second.wait().unwrap().success());

        assert_eq!(
            fs::read(dir.path().join("stable-chain.mp4")).unwrap(),
            b"same named bytes"
        );
        let index = load_committed_archive_index(dir.path()).unwrap();
        assert_eq!(index.entries.len(), 1);
        assert!(index.get("stable-chain.mp4").is_some());
    }

    #[tokio::test]
    async fn authority_restart_stats_every_live_name_but_hashes_only_changed_media() {
        let dir = tempfile::tempdir().unwrap();
        let gate = GalleryPublicationGate::default();
        let mut batch = BatchTransaction::begin(
            dir.path(),
            "stats",
            0,
            serde_json::json!({}),
            vec![record("first.png", 0), record("second.png", 1)],
        )
        .unwrap();
        batch.stage_bytes(0, b"first").unwrap();
        batch.stage_bytes(1, b"second").unwrap();
        batch.mark_prepared().unwrap();
        batch.commit(&gate, Arc::new(None)).await.unwrap();
        drop(batch);

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let unchanged =
            crate::gallery_authority::load_or_initialize(dir.path(), &bookkeeping, || {
                unreachable!("a durable checkpoint already exists")
            })
            .unwrap();
        assert_eq!(unchanged.stats.files_statted, 2);
        assert_eq!(
            unchanged.stats.files_hashed, 0,
            "unchanged startup must trust stable identity facts"
        );
        assert!(
            crate::gallery_authority::storage_file_count(dir.path()) <= 4,
            "current, backup, previous, and generation marker are the bounded steady state"
        );
        drop(bookkeeping);

        fs::write(dir.path().join("first.png"), b"f1rst").unwrap();
        sync_dir(dir.path()).unwrap();
        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let changed =
            crate::gallery_authority::load_or_initialize(dir.path(), &bookkeeping, || {
                unreachable!("a durable checkpoint already exists")
            })
            .unwrap();
        assert_eq!(changed.stats.files_statted, 2);
        assert_eq!(
            changed.stats.files_hashed, 1,
            "only the same-size entry whose stable identity facts changed is hashed"
        );
        assert!(changed.index.is_quarantined("first.png"));
        assert!(changed.index.get("second.png").is_some());
    }

    #[test]
    fn deleting_one_of_ten_thousand_authority_entries_hashes_no_unrelated_media() {
        let dir = tempfile::tempdir().unwrap();
        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let loaded = crate::gallery_authority::load_or_initialize(dir.path(), &bookkeeping, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = loaded.index;
        let checksum = format!("{:x}", Sha256::digest(b"x"));
        for child_index in 0..10_000usize {
            let name = format!("entry-{child_index:05}.png");
            let path = dir.path().join(&name);
            fs::write(&path, b"x").unwrap();
            let mut saved = record(&name, child_index as u64);
            saved.output_dir = dir.path().to_string_lossy().into_owned();
            index.entries.insert(
                name.clone(),
                CommittedArchiveEntry {
                    identity: ArchivedChildIdentity {
                        parent_id: "scale-fixture".into(),
                        attempt_generation: 0,
                        child_index,
                        final_name: name,
                        checksum_sha256: checksum.clone(),
                        size_bytes: 1,
                    },
                    record: saved,
                    facts: Some(ArchiveFileFacts::from_path(&path).unwrap()),
                },
            );
        }
        sync_dir(dir.path()).unwrap();
        let generation = crate::gallery_authority::commit_snapshot(
            dir.path(),
            &bookkeeping,
            loaded.generation,
            &mut index,
            "scale_fixture",
            Vec::new(),
        )
        .unwrap();
        let gate = GalleryPublicationGate::default();
        gate.install_committed_archive_index(generation, index);
        drop(bookkeeping);

        reset_authority_hash_count();
        assert_eq!(
            tombstone_committed_archive_filename(dir.path(), "entry-05000.png", &gate).unwrap(),
            ArchiveDeleteDisposition::SafeToUnlink
        );
        gate.acknowledge_retirement_projections(dir.path(), ["entry-05000.png".to_owned()])
            .unwrap();
        assert_eq!(
            authority_hash_count(),
            0,
            "stable file identity must make delete independent of unrelated gallery bytes"
        );
        assert!(
            crate::gallery_authority::storage_file_count(dir.path()) <= 4,
            "authority persistence must remain bounded independently of publication count"
        );

        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let mut current = gate
            .committed_archive_index_while_locked(dir.path(), &bookkeeping)
            .unwrap();
        let generation = crate::gallery_authority::read_generation(dir.path(), &bookkeeping)
            .unwrap()
            .unwrap();
        let next_generation = crate::gallery_authority::commit_snapshot(
            dir.path(),
            &bookkeeping,
            generation,
            &mut current,
            "retirement_gc_test",
            Vec::new(),
        )
        .unwrap();
        gate.install_committed_archive_index(next_generation, current);
        drop(bookkeeping);
        assert!(
            !gate
                .committed_archive_index(dir.path())
                .unwrap()
                .retired_names
                .contains("entry-05000.png"),
            "a projected retirement must leave current authority after two durable checkpoints"
        );
    }

    #[tokio::test]
    async fn delete_preserves_and_quarantines_a_file_whose_archived_identity_changed() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("damaged.png", 0), record("sibling.png", 1)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"original bytes").unwrap();
        transaction.stage_bytes(1, b"sibling bytes").unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        drop(transaction);

        fs::write(dir.path().join("damaged.png"), b"tampered").unwrap();
        sync_dir(dir.path()).unwrap();
        let delete_gate = GalleryPublicationGate::default();
        assert_eq!(
            tombstone_committed_archive_filename(dir.path(), "damaged.png", &delete_gate).unwrap(),
            ArchiveDeleteDisposition::PreservedReplacement,
            "delete must not unlink bytes that no longer match archived identity"
        );
        assert_eq!(
            fs::read(dir.path().join("damaged.png")).unwrap(),
            b"tampered"
        );

        let gate = GalleryPublicationGate::default();
        recover_transactions(dir.path(), &gate, Arc::new(None))
            .await
            .unwrap();
        let index = gate.committed_archive_index(dir.path()).unwrap();
        assert!(index.get("damaged.png").is_none());
        assert!(index.is_quarantined("damaged.png"));
        assert!(dir.path().join("damaged.png").is_file());
        assert_eq!(
            index.get("sibling.png").unwrap().record().metadata.prompt,
            "prompt 1"
        );
    }

    #[tokio::test]
    async fn collision_is_frozen_without_overwriting_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("same.png"), b"existing").unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("same.png", 0)],
        )
        .unwrap();
        assert_eq!(transaction.manifest.children[0].final_name, "same-1.png");
        transaction.stage_bytes(0, b"new").unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        assert_eq!(fs::read(dir.path().join("same.png")).unwrap(), b"existing");
        assert_eq!(fs::read(dir.path().join("same-1.png")).unwrap(), b"new");
    }

    #[test]
    fn stale_attempt_cleanup_cannot_release_a_retry_owned_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let stale = BatchTransaction::begin(
            dir.path(),
            "parent",
            1,
            serde_json::json!({}),
            vec![record("same.png", 0)],
        )
        .unwrap();
        fs::remove_file(reservation_path(dir.path(), "same.png")).unwrap();
        let retry = BatchTransaction::begin(
            dir.path(),
            "parent",
            2,
            serde_json::json!({}),
            vec![record("same.png", 0)],
        )
        .unwrap();

        stale.release_reservations();

        let owner: ReservationOwner =
            serde_json::from_slice(&fs::read(reservation_path(dir.path(), "same.png")).unwrap())
                .unwrap();
        assert_eq!(owner, retry.reservation_owner());
    }

    #[tokio::test]
    async fn committing_recovery_rolls_forward_idempotently_with_db_rows() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            7,
            serde_json::json!({}),
            vec![record("one.png", 0), record("two.png", 1)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.stage_bytes(1, b"two").unwrap();
        transaction.mark_prepared().unwrap();
        transaction.manifest.state = BatchManifestState::Committing;
        transaction.persist_manifest().unwrap();
        fs::hard_link(
            transaction.staging_path(0).unwrap(),
            dir.path().join("one.png"),
        )
        .unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let gate = GalleryPublicationGate::default();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(dir.path(), &gate, db.clone())
            .await
            .unwrap();
        let second = recover_transactions(dir.path(), &gate, db.clone())
            .await
            .unwrap();

        assert_eq!(report.rolled_forward, 1);
        assert_eq!(second, RecoveryReport::default());
        assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 2);
        assert_eq!(fs::read(dir.path().join("two.png")).unwrap(), b"two");
        assert!(
            db.as_ref()
                .as_ref()
                .unwrap()
                .get(dir.path(), "one.png")
                .unwrap()
                .unwrap()
                .file_mtime_ms
                .is_some(),
            "recovery must not erase gallery ordering metadata"
        );
    }

    #[tokio::test]
    async fn recovery_replaces_an_unjournaled_interrupted_no_replace_copy() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"complete").unwrap();
        transaction.mark_prepared().unwrap();
        transaction.manifest.state = BatchManifestState::Committing;
        transaction.persist_manifest().unwrap();
        fs::write(dir.path().join("one.png"), b"partial").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.rolled_forward, 1);
        assert_eq!(fs::read(dir.path().join("one.png")).unwrap(), b"complete");
    }

    #[tokio::test]
    async fn recovery_fails_closed_if_a_journaled_final_was_modified() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"complete").unwrap();
        transaction.mark_prepared().unwrap();
        transaction.manifest.state = BatchManifestState::Committing;
        transaction.persist_manifest().unwrap();
        fs::write(dir.path().join("one.png"), b"complete").unwrap();
        transaction
            .append_journal(BatchJournalEvent::FinalPublished { child_index: 0 })
            .unwrap();
        fs::write(dir.path().join("one.png"), b"corrupt").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let error = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap_err();

        assert!(format!("{error:#}").contains("changed after publication"));
        assert_eq!(fs::read(dir.path().join("one.png")).unwrap(), b"corrupt");
    }

    #[tokio::test]
    async fn recovery_retains_publication_evidence_when_staged_and_final_bytes_are_lost() {
        for corrupt_staging in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut transaction = BatchTransaction::begin(
                dir.path(),
                "parent",
                0,
                serde_json::json!({}),
                vec![record("one.png", 0)],
            )
            .unwrap();
            transaction.stage_bytes(0, b"complete").unwrap();
            transaction.mark_prepared().unwrap();
            transaction.manifest.state = BatchManifestState::Committing;
            transaction.persist_manifest().unwrap();
            let staged = transaction.staging_path(0).unwrap();
            let final_path = dir.path().join("one.png");
            fs::hard_link(&staged, &final_path).unwrap();
            File::open(&final_path).unwrap().sync_all().unwrap();
            transaction
                .append_journal(BatchJournalEvent::FinalPublished { child_index: 0 })
                .unwrap();
            let journal_path = transaction.attempt_dir.join(JOURNAL_FILE);
            let journal_before = fs::read(&journal_path).unwrap();
            let reservation = reservation_path(dir.path(), "one.png");
            fs::remove_file(&final_path).unwrap();
            if corrupt_staging {
                fs::write(&staged, b"corrupt").unwrap();
            } else {
                fs::remove_file(&staged).unwrap();
            }
            transaction.relinquish_attempt_authority_for_recovery();

            let error = recover_transactions(
                dir.path(),
                &GalleryPublicationGate::default(),
                Arc::new(None),
            )
            .await
            .unwrap_err();

            assert!(format!("{error:#}").contains("durable publication evidence"));
            assert!(transaction.attempt_dir.exists());
            assert!(reservation.exists());
            assert_eq!(
                fs::read(&journal_path).unwrap(),
                journal_before,
                "recovery must not append a Failed transition"
            );
            let loaded =
                BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                    .unwrap();
            assert_eq!(loaded.manifest.state, BatchManifestState::Committing);
            assert_eq!(loaded.journaled_final_children, BTreeSet::from([0]));
        }
    }

    #[tokio::test]
    async fn recovery_rolls_forward_checksum_valid_finals_when_staging_is_lost() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({"batch_size": 2}),
            vec![record("one.png", 0), record("two.png", 1)],
        )
        .unwrap();
        for (child_index, bytes) in [b"one".as_slice(), b"two".as_slice()]
            .into_iter()
            .enumerate()
        {
            transaction.stage_bytes(child_index, bytes).unwrap();
        }
        transaction.mark_prepared().unwrap();
        transaction.manifest.state = BatchManifestState::Committing;
        transaction.persist_manifest().unwrap();
        for child_index in 0..transaction.manifest.children.len() {
            let child = &transaction.manifest.children[child_index];
            let final_path = dir.path().join(&child.final_name);
            fs::hard_link(transaction.staging_path(child_index).unwrap(), &final_path).unwrap();
            File::open(&final_path).unwrap().sync_all().unwrap();
            transaction
                .append_journal(BatchJournalEvent::FinalPublished { child_index })
                .unwrap();
        }
        fs::remove_dir_all(transaction.attempt_dir.join("staging")).unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.rolled_forward, 1);
        assert_eq!(fs::read(dir.path().join("one.png")).unwrap(), b"one");
        assert_eq!(fs::read(dir.path().join("two.png")).unwrap(), b"two");
        assert!(!transaction.attempt_dir.exists());
        assert!(!committed_manifests_dir(dir.path(), "parent")
            .join("0.json")
            .exists());
        assert!(load_committed_archive_index(dir.path())
            .unwrap()
            .get("one.png")
            .is_some());
    }

    #[tokio::test]
    async fn startup_rolls_back_unpublished_attempts_without_touching_retry() {
        let dir = tempfile::tempdir().unwrap();
        let mut stale = BatchTransaction::begin(
            dir.path(),
            "parent",
            1,
            serde_json::json!({}),
            vec![record("stale.png", 0)],
        )
        .unwrap();
        stale.stage_bytes(0, b"stale").unwrap();
        let mut retry = BatchTransaction::begin(
            dir.path(),
            "parent",
            2,
            serde_json::json!({}),
            vec![record("retry.png", 0)],
        )
        .unwrap();
        retry.stage_bytes(0, b"retry").unwrap();
        stale.relinquish_attempt_authority_for_recovery();
        retry.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.rolled_back, 2);
        assert_eq!(stale.manifest.attempt_generation, 1);
        assert_eq!(retry.manifest.attempt_generation, 2);
        assert!(!dir.path().join("stale.png").exists());
        assert!(!dir.path().join("retry.png").exists());
    }

    #[tokio::test]
    async fn startup_reconstructs_a_torn_manifest_from_the_fsynced_journal() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            3,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        fs::write(transaction.attempt_dir.join(MANIFEST_FILE), b"{torn").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.rolled_back, 1);
        assert!(!transaction.attempt_dir.exists());
        assert!(!dir.path().join("one.png").exists());
    }

    #[test]
    fn incomplete_trailing_journal_record_is_truncated_before_append() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let journal_path = transaction.attempt_dir.join(JOURNAL_FILE);
        let mut journal = OpenOptions::new().append(true).open(&journal_path).unwrap();
        journal.write_all(b"{partial").unwrap();
        journal.sync_all().unwrap();
        drop(journal);

        let mut loaded =
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap();
        loaded.stage_bytes(0, b"one").unwrap();

        let records = load_journal(&journal_path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sequence, 0);
        assert_eq!(records[1].sequence, 1);
    }

    #[test]
    fn parseable_journal_tail_without_newline_is_healed_before_append() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let journal_path = transaction.attempt_dir.join(JOURNAL_FILE);
        let mut bytes = fs::read(&journal_path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        fs::write(&journal_path, bytes).unwrap();

        let mut loaded =
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap();
        loaded.stage_bytes(0, b"one").unwrap();

        let bytes = fs::read(&journal_path).unwrap();
        assert!(bytes.ends_with(b"\n"));
        let records = load_journal(&journal_path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sequence, 0);
        assert_eq!(records[1].sequence, 1);
    }

    #[test]
    fn complete_malformed_journal_record_fails_closed_without_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let journal_path = transaction.attempt_dir.join(JOURNAL_FILE);
        let mut journal = OpenOptions::new().append(true).open(&journal_path).unwrap();
        journal.write_all(b"{malformed}\n").unwrap();
        journal.sync_all().unwrap();
        drop(journal);
        let before = fs::read(&journal_path).unwrap();

        assert!(load_journal(&journal_path).is_err());
        assert_eq!(fs::read(&journal_path).unwrap(), before);
    }

    #[test]
    fn semantically_invalid_atomic_manifest_recovers_from_valid_journal() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let manifest_path = transaction.attempt_dir.join(MANIFEST_FILE);
        let mut corrupt: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        corrupt["parent_id"] = serde_json::json!("../escape");
        fs::write(&manifest_path, serde_json::to_vec(&corrupt).unwrap()).unwrap();

        let recovered = BatchTransaction::load(dir.path(), &manifest_path).unwrap();

        assert_eq!(recovered.manifest.parent_id, "parent");
        assert!(recovered.reconstructed_from_journal);
    }

    #[test]
    fn orphan_collection_removes_only_the_orphan_attempt_reservations() {
        let dir = tempfile::tempdir().unwrap();
        let orphan = attempt_dir(dir.path(), "parent", 4);
        fs::create_dir_all(&orphan).unwrap();
        fs::create_dir_all(reservations_dir(dir.path())).unwrap();
        let owner = ReservationOwner {
            parent_id: "parent".into(),
            attempt_generation: 4,
        };
        fs::write(
            reservation_path(dir.path(), "orphan.png"),
            serde_json::to_vec(&owner).unwrap(),
        )
        .unwrap();
        let _retry = BatchTransaction::begin(
            dir.path(),
            "parent",
            5,
            serde_json::json!({}),
            vec![record("retry.png", 0)],
        )
        .unwrap();

        let root = dir.path().join(TRANSACTION_DIR);
        let mut claimed = Vec::new();
        let bookkeeping = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        collect_claimed_attempts(&root, &mut claimed, &bookkeeping).unwrap();

        assert!(!orphan.exists());
        assert!(!reservation_path(dir.path(), "orphan.png").exists());
        assert!(reservation_path(dir.path(), "retry.png").exists());
        assert!(
            claimed.is_empty(),
            "collection must not claim a live retry owned by this process"
        );
    }

    #[tokio::test]
    async fn recovery_refuses_to_serve_an_unjournaled_partial_publish() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            1,
            serde_json::json!({}),
            vec![record("one.png", 0), record("two.png", 1)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.stage_bytes(1, b"two").unwrap();
        transaction.mark_prepared().unwrap();
        fs::hard_link(
            transaction.staging_path(0).unwrap(),
            dir.path().join("one.png"),
        )
        .unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let error = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("refusing to serve"));
        assert_eq!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap()
                .manifest
                .state,
            BatchManifestState::Prepared
        );
    }

    #[tokio::test]
    async fn committed_manifest_with_missing_final_fails_startup_without_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        commit_journal_without_cleanup(&mut transaction, b"one");
        fs::remove_file(dir.path().join("one.png")).unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let error = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("committed batch child is missing"));
        assert_eq!(transaction.manifest.state, BatchManifestState::Committed);
    }

    #[tokio::test]
    async fn publication_writer_excludes_gallery_readers() {
        let gate = GalleryPublicationGate::default();
        let writer = gate.write().await;
        let blocked = tokio::time::timeout(std::time::Duration::from_millis(20), gate.read()).await;
        assert!(blocked.is_err());
        drop(writer);
        tokio::time::timeout(std::time::Duration::from_secs(1), gate.read())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn panic_after_committing_is_captured_with_writer_gate_retained() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        transaction.commit_failpoint = Some(CommitFailpoint::PanicAfterCommitting);

        let error = transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap_err();

        assert!(error.entered_committing());
        assert!(error.to_string().contains("injected panic"));
        // Production drops this value and aborts. Startup recovery is the one
        // supported path that releases both retained authorities in-process.
        let _ = error.into_startup_error();
    }

    #[tokio::test]
    async fn every_durable_commit_boundary_rolls_forward_atomically_after_restart() {
        let mut fault_points = vec![
            CommitFailpoint::CommittingState,
            CommitFailpoint::OutputDirectoryFsync,
            CommitFailpoint::MetadataManifestFsync,
            CommitFailpoint::DatabaseTransaction,
            CommitFailpoint::DatabaseJournalFsync,
            CommitFailpoint::CommittedState,
            CommitFailpoint::ReservationsReleased,
            CommitFailpoint::PrivateStagingCleaned,
            CommitFailpoint::CleanupStartedJournal,
            CommitFailpoint::ArchiveWrite,
            CommitFailpoint::AttemptDirectoryRemoved,
            CommitFailpoint::AttemptsDirectoryFsync,
        ];
        for child_index in 0..2 {
            fault_points.extend([
                CommitFailpoint::FinalPublish(child_index),
                CommitFailpoint::FinalFileFsync(child_index),
                CommitFailpoint::FinalJournalFsync(child_index),
            ]);
        }

        for fault_point in fault_points {
            let dir = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            let gate = GalleryPublicationGate::default();
            let mut transaction = BatchTransaction::begin(
                dir.path(),
                "parent",
                0,
                serde_json::json!({"batch_size": 2}),
                vec![record("one.png", 0), record("two.png", 1)],
            )
            .unwrap();
            transaction.stage_bytes(0, b"one").unwrap();
            transaction.stage_bytes(1, b"two").unwrap();
            transaction.mark_prepared().unwrap();
            transaction.commit_failpoint = Some(fault_point);

            let error = match transaction.commit(&gate, db.clone()).await {
                Err(error) => error,
                Ok(()) => panic!("fault point was not injected: {fault_point:?}"),
            };
            assert!(
                error.entered_committing(),
                "{fault_point:?} must be a post-committing fault"
            );
            let startup_error = error.into_startup_error();
            assert!(startup_error.to_string().contains("injected commit"));
            transaction.relinquish_attempt_authority_for_recovery();

            let report = recover_transactions(dir.path(), &gate, db.clone())
                .await
                .unwrap();
            assert!(
                report.rolled_forward == 1
                    || report.healed_committed_rows == 2
                    || report == RecoveryReport::default(),
                "unexpected recovery report for {fault_point:?}: {report:?}"
            );
            assert_eq!(fs::read(dir.path().join("one.png")).unwrap(), b"one");
            assert_eq!(fs::read(dir.path().join("two.png")).unwrap(), b"two");
            assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 2);
            assert!(
                !attempt_dir(dir.path(), "parent", 0).is_dir(),
                "terminal attempt was not archived for {fault_point:?}"
            );
        }
    }

    #[tokio::test]
    async fn crash_after_archived_manifest_is_terminal_without_a_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(None);
        let gate = GalleryPublicationGate::default();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        transaction.commit_failpoint = Some(CommitFailpoint::ArchivedManifest);

        let error = transaction.commit(&gate, db.clone()).await.unwrap_err();
        assert!(error.entered_committing());
        assert!(error.to_string().contains("ArchivedManifest"));
        let startup_error = error.into_startup_error();
        assert!(startup_error.to_string().contains("ArchivedManifest"));
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(dir.path(), &gate, db).await.unwrap();
        assert!(
            report.rolled_forward == 1 || report == RecoveryReport::default(),
            "unexpected recovery report: {report:?}"
        );
        assert_eq!(fs::read(dir.path().join("one.png")).unwrap(), b"one");
        assert!(!attempt_dir(dir.path(), "parent", 0).exists());
        assert!(
            gate.committed_archive_index(dir.path())
                .unwrap()
                .get("one.png")
                .is_some(),
            "database-free central authority must survive attempt cleanup"
        );
    }

    #[tokio::test]
    async fn transaction_owned_partial_stream_is_removed_on_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let gate = GalleryPublicationGate::default();
        let db = Arc::new(None);
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "gallery-import-test",
            0,
            serde_json::json!({"declared_size": 4096}),
            vec![record("partial.mp4", 0)],
        )
        .unwrap();
        fs::write(transaction.staging_path(0).unwrap(), b"partial body").unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(dir.path(), &gate, db).await.unwrap();
        assert_eq!(report.rolled_back, 1);
        assert!(!dir.path().join("partial.mp4").exists());
        assert!(!attempt_dir(dir.path(), "gallery-import-test", 0).exists());
    }

    #[test]
    fn disk_preflight_keeps_encoder_and_manifest_headroom() {
        let expected = 100 * 1024 * 1024;
        assert!(validate_available_space(expected, expected).is_err());
        assert!(validate_available_space(expected * 2 + DISK_SAFETY_FLOOR_BYTES, expected).is_ok());
        assert!(validate_available_space(u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn ordinary_directory_sync_tolerates_only_unsupported_filesystems() {
        let dir = tempfile::tempdir().unwrap();
        let unsupported = std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "directory fsync is unsupported",
        );
        tolerate_unsupported_ordinary_directory_sync(dir.path(), Err(unsupported.into())).unwrap();

        let denied = std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "directory fsync denied",
        );
        assert!(
            tolerate_unsupported_ordinary_directory_sync(dir.path(), Err(denied.into())).is_err()
        );
    }

    #[test]
    fn durable_reservation_still_fails_closed_when_directory_sync_fails() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(reservations_dir(dir.path())).unwrap();
        let owner = ReservationOwner {
            parent_id: "parent".into(),
            attempt_generation: 0,
        };
        let unsupported_sync = |_path: &Path| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "injected unsupported directory fsync",
            )
            .into())
        };

        let error = reserve_final_name_with_directory_sync(
            dir.path(),
            "strict.png",
            &owner,
            &unsupported_sync,
        )
        .unwrap_err();

        assert!(error.to_string().contains("unsupported directory fsync"));
        assert!(!reservation_path(dir.path(), "strict.png").exists());
    }

    #[test]
    fn batch_journal_rejects_manifest_state_regression() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        let mut regressed = transaction.manifest.clone();
        regressed.state = BatchManifestState::Staging;
        append_raw_batch_record(
            &transaction.attempt_dir,
            &BatchJournalRecord {
                sequence: next_batch_sequence(&transaction),
                attempt_generation: 0,
                event: BatchJournalEvent::ManifestSnapshot {
                    manifest: regressed,
                },
            },
        );

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn batch_journal_rejects_non_monotonic_staging_facts() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        let mut changed = transaction.manifest.clone();
        changed.children[0].checksum_sha256 = Some("a".repeat(64));
        append_raw_batch_record(
            &transaction.attempt_dir,
            &BatchJournalRecord {
                sequence: next_batch_sequence(&transaction),
                attempt_generation: 0,
                event: BatchJournalEvent::ManifestSnapshot { manifest: changed },
            },
        );

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn batch_journal_rejects_immutable_attempt_identity_drift() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({"prompt": "original"}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        for drift in 0..3 {
            let mut drifted = transaction.manifest.clone();
            match drift {
                0 => drifted.normalized_request = serde_json::json!({"prompt": "changed"}),
                1 => drifted.children[0].staging_name = "other.stage".into(),
                2 => {
                    drifted.children[0].final_name = "other.png".into();
                    drifted.children[0].record.filename = "other.png".into();
                }
                _ => unreachable!(),
            }
            append_raw_batch_record(
                &transaction.attempt_dir,
                &BatchJournalRecord {
                    sequence: next_batch_sequence(&transaction),
                    attempt_generation: 0,
                    event: BatchJournalEvent::ManifestSnapshot { manifest: drifted },
                },
            );
            assert!(BatchTransaction::load(
                dir.path(),
                &transaction.attempt_dir.join(MANIFEST_FILE)
            )
            .is_err());
            fs::write(
                transaction.attempt_dir.join(JOURNAL_FILE),
                serde_json::to_vec(&BatchJournalRecord {
                    sequence: 0,
                    attempt_generation: 0,
                    event: BatchJournalEvent::ManifestSnapshot {
                        manifest: transaction.manifest.clone(),
                    },
                })
                .unwrap()
                .into_iter()
                .chain(std::iter::once(b'\n'))
                .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }

    #[test]
    fn batch_journal_rejects_publication_before_committing() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        append_raw_batch_record(
            &transaction.attempt_dir,
            &BatchJournalRecord {
                sequence: next_batch_sequence(&transaction),
                attempt_generation: 0,
                event: BatchJournalEvent::FinalPublished { child_index: 0 },
            },
        );

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn batch_journal_rejects_out_of_order_and_duplicate_publications() {
        for published in [vec![1], vec![0, 0]] {
            let dir = tempfile::tempdir().unwrap();
            let mut transaction = BatchTransaction::begin(
                dir.path(),
                "parent",
                0,
                serde_json::json!({}),
                vec![record("one.png", 0), record("two.png", 1)],
            )
            .unwrap();
            transaction.stage_bytes(0, b"one").unwrap();
            transaction.stage_bytes(1, b"two").unwrap();
            transaction.mark_prepared().unwrap();
            transaction.manifest.state = BatchManifestState::Committing;
            transaction.persist_manifest().unwrap();
            for child_index in published {
                append_raw_batch_record(
                    &transaction.attempt_dir,
                    &BatchJournalRecord {
                        sequence: next_batch_sequence(&transaction),
                        attempt_generation: 0,
                        event: BatchJournalEvent::FinalPublished { child_index },
                    },
                );
            }

            assert!(BatchTransaction::load(
                dir.path(),
                &transaction.attempt_dir.join(MANIFEST_FILE)
            )
            .is_err());
        }
    }

    #[test]
    fn batch_journal_rejects_premature_or_duplicate_metadata_commit() {
        for duplicate in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut transaction = BatchTransaction::begin(
                dir.path(),
                "parent",
                0,
                serde_json::json!({}),
                vec![record("one.png", 0)],
            )
            .unwrap();
            if duplicate {
                transaction.stage_bytes(0, b"one").unwrap();
                transaction.mark_prepared().unwrap();
                transaction.manifest.state = BatchManifestState::Committing;
                transaction.persist_manifest().unwrap();
                transaction
                    .append_journal(BatchJournalEvent::FinalPublished { child_index: 0 })
                    .unwrap();
                transaction.persist_manifest().unwrap();
            }
            transaction
                .append_journal(BatchJournalEvent::MetadataCommitted)
                .unwrap();
            if duplicate {
                transaction
                    .append_journal(BatchJournalEvent::MetadataCommitted)
                    .unwrap();
            }

            assert!(BatchTransaction::load(
                dir.path(),
                &transaction.attempt_dir.join(MANIFEST_FILE)
            )
            .is_err());
        }
    }

    #[test]
    fn batch_journal_rejects_metadata_before_post_publish_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        transaction.manifest.state = BatchManifestState::Committing;
        transaction.persist_manifest().unwrap();
        transaction
            .append_journal(BatchJournalEvent::FinalPublished { child_index: 0 })
            .unwrap();
        transaction
            .append_journal(BatchJournalEvent::MetadataCommitted)
            .unwrap();

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn batch_journal_rejects_committed_snapshot_before_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        journal_post_publish_snapshot(&mut transaction, b"one");
        let mut committed = transaction.manifest.clone();
        committed.state = BatchManifestState::Committed;
        append_raw_batch_record(
            &transaction.attempt_dir,
            &BatchJournalRecord {
                sequence: next_batch_sequence(&transaction),
                attempt_generation: 0,
                event: BatchJournalEvent::ManifestSnapshot {
                    manifest: committed,
                },
            },
        );

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn batch_journal_rejects_cleanup_before_commit() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        append_raw_batch_record(
            &transaction.attempt_dir,
            &BatchJournalRecord {
                sequence: next_batch_sequence(&transaction),
                attempt_generation: 0,
                event: BatchJournalEvent::CleanupStarted,
            },
        );

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn batch_journal_rejects_duplicate_cleanup_and_all_trailing_events() {
        for trailing_event in [
            BatchJournalEvent::CleanupStarted,
            BatchJournalEvent::MetadataCommitted,
            BatchJournalEvent::FinalPublished { child_index: 0 },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut transaction = BatchTransaction::begin(
                dir.path(),
                "parent",
                0,
                serde_json::json!({}),
                vec![record("one.png", 0)],
            )
            .unwrap();
            commit_journal_without_cleanup(&mut transaction, b"one");
            append_raw_batch_record(
                &transaction.attempt_dir,
                &BatchJournalRecord {
                    sequence: next_batch_sequence(&transaction),
                    attempt_generation: 0,
                    event: BatchJournalEvent::CleanupStarted,
                },
            );
            append_raw_batch_record(
                &transaction.attempt_dir,
                &BatchJournalRecord {
                    sequence: next_batch_sequence(&transaction),
                    attempt_generation: 0,
                    event: trailing_event,
                },
            );

            assert!(BatchTransaction::load(
                dir.path(),
                &transaction.attempt_dir.join(MANIFEST_FILE)
            )
            .is_err());
        }
    }

    #[test]
    fn batch_load_heals_exactly_one_atomic_manifest_record_ahead() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        let journal_path = transaction.attempt_dir.join(JOURNAL_FILE);
        let bytes = fs::read(&journal_path).unwrap();
        let mut records: Vec<_> = bytes.split_inclusive(|byte| *byte == b'\n').collect();
        records.pop();
        fs::write(&journal_path, records.concat()).unwrap();

        let mut loaded =
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap();

        assert!(loaded.journal_needs_heal);
        assert_eq!(loaded.manifest.state, BatchManifestState::Prepared);
        assert_eq!(loaded.manifest.children[0].size_bytes, Some(3));
        loaded
            .append_journal(BatchJournalEvent::ManifestSnapshot {
                manifest: loaded.manifest.clone(),
            })
            .unwrap();
        let records = load_journal(&journal_path).unwrap();
        replay_batch_journal(dir.path(), &transaction.attempt_dir, &records)
            .unwrap()
            .unwrap();
        assert_eq!(records.len(), 3);
    }

    #[tokio::test]
    async fn failed_partial_staging_snapshot_is_recoverable() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0), record("two.png", 1)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.manifest.state = BatchManifestState::Failed;
        transaction.persist_manifest().unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report, RecoveryReport::default());
        assert!(!transaction.attempt_dir.exists());
    }

    #[tokio::test]
    async fn terminal_committed_manifest_without_journal_finishes_durable_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        commit_journal_without_cleanup(&mut transaction, b"one");
        transaction
            .append_journal(BatchJournalEvent::CleanupStarted)
            .unwrap();
        transaction.cleanup_started = true;
        let final_path = dir.path().join("one.png");
        let reservation = reservation_path(dir.path(), "one.png");
        assert!(final_path.is_file());
        assert!(reservation.is_file());
        fs::remove_file(transaction.attempt_dir.join(JOURNAL_FILE)).unwrap();
        sync_dir(&transaction.attempt_dir).unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report, RecoveryReport::default());
        assert_eq!(fs::read(final_path).unwrap(), b"one");
        assert!(!reservation.exists());
        assert!(!transaction.attempt_dir.exists());
        assert!(load_committed_archive_index(dir.path())
            .unwrap()
            .get("one.png")
            .is_some());
    }

    #[tokio::test]
    async fn terminal_failed_manifest_without_journal_finishes_durable_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.manifest.state = BatchManifestState::Failed;
        transaction.persist_manifest().unwrap();
        let reservation = reservation_path(dir.path(), "one.png");
        assert!(reservation.is_file());
        fs::remove_file(transaction.attempt_dir.join(JOURNAL_FILE)).unwrap();
        sync_dir(&transaction.attempt_dir).unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let report = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report, RecoveryReport::default());
        assert!(!dir.path().join("one.png").exists());
        assert!(!reservation.exists());
        assert!(!transaction.attempt_dir.exists());
    }

    #[tokio::test]
    async fn terminal_failed_manifest_without_journal_retains_unexpected_final_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.manifest.state = BatchManifestState::Failed;
        transaction.persist_manifest().unwrap();
        let staged = transaction.staging_path(0).unwrap();
        let final_path = dir.path().join("one.png");
        fs::hard_link(staged, &final_path).unwrap();
        File::open(&final_path).unwrap().sync_all().unwrap();
        let reservation = reservation_path(dir.path(), "one.png");
        fs::remove_file(transaction.attempt_dir.join(JOURNAL_FILE)).unwrap();
        sync_dir(&transaction.attempt_dir).unwrap();
        transaction.relinquish_attempt_authority_for_recovery();

        let error = recover_transactions(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap_err();

        assert!(format!("{error:#}").contains("final-path artifact"));
        assert_eq!(fs::read(final_path).unwrap(), b"one");
        assert!(reservation.exists());
        assert!(transaction.attempt_dir.exists());
    }

    #[test]
    fn batch_load_never_prefers_a_stale_valid_disk_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let initial = transaction.manifest.clone();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        atomic_write_json(&transaction.attempt_dir.join(MANIFEST_FILE), &initial).unwrap();

        let loaded =
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap();

        assert_eq!(loaded.manifest.state, BatchManifestState::Prepared);
        assert!(!loaded.journal_needs_heal);
    }

    #[test]
    fn batch_load_rejects_a_divergent_valid_disk_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let mut transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        let mut divergent = transaction.manifest.clone();
        divergent.state = BatchManifestState::Staging;
        divergent.children[0].checksum_sha256 = Some("a".repeat(64));
        atomic_write_json(&transaction.attempt_dir.join(MANIFEST_FILE), &divergent).unwrap();

        assert!(
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .is_err()
        );
    }

    #[test]
    fn child_staging_journal_growth_is_linear_in_child_count() {
        let dir = tempfile::tempdir().unwrap();
        let records = (0..64)
            .map(|index| record(&format!("{index}.png"), index))
            .collect();
        let mut transaction =
            BatchTransaction::begin(dir.path(), "parent", 0, serde_json::json!({}), records)
                .unwrap();

        for child_index in 0..64 {
            transaction.stage_bytes(child_index, b"x").unwrap();
        }

        let journal_bytes = std::fs::metadata(transaction.attempt_dir.join(JOURNAL_FILE))
            .unwrap()
            .len();
        assert!(
            journal_bytes < 1_000_000,
            "per-child staging rewrote the full manifest: {journal_bytes} bytes"
        );
    }

    #[test]
    fn hundred_thousand_child_staged_deltas_have_constant_record_size() {
        let mut bytes = Vec::new();
        for child_index in 0..100_000 {
            let record = BatchJournalRecord {
                sequence: child_index as u64 + 1,
                attempt_generation: 7,
                event: BatchJournalEvent::ChildStaged {
                    receipt: BatchChildReceipt {
                        version: 1,
                        parent_id: "parent".into(),
                        attempt_generation: 7,
                        child_index,
                        lease_generation: 3,
                        checksum_sha256: "a".repeat(64),
                        size_bytes: 123,
                        record_identity_sha256: "b".repeat(64),
                    },
                },
            };
            bytes.extend(serde_json::to_vec(&record).unwrap());
            bytes.push(b'\n');
        }
        assert!(
            bytes.len() < 40_000_000,
            "ChildStaged deltas are not constant-size: {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn uncertain_delta_append_boundaries_poison_until_recovery() {
        for failpoint in [
            DeltaAppendFailpoint::Open,
            DeltaAppendFailpoint::Write,
            DeltaAppendFailpoint::Flush,
            DeltaAppendFailpoint::FileSync,
            DeltaAppendFailpoint::DirectorySync,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut transaction = BatchTransaction::begin(
                dir.path(),
                "parent",
                0,
                serde_json::json!({}),
                vec![record("one.png", 0)],
            )
            .unwrap();
            transaction.delta_append_failpoint = Some(failpoint);
            assert!(transaction.stage_bytes(0, b"one").is_err());
            assert!(transaction.poisoned);
            let sequence = transaction.next_journal_sequence;
            let error = transaction
                .append_journal(BatchJournalEvent::MetadataCommitted)
                .unwrap_err();
            assert!(format!("{error:#}").contains("poisoned"));
            assert_eq!(transaction.next_journal_sequence, sequence);

            transaction.relinquish_attempt_authority_for_recovery();
            let recovered =
                BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE));
            assert!(
                recovered.is_ok(),
                "recovery could not decide the durable prefix after {failpoint:?}: {recovered:?}"
            );
        }
    }

    #[test]
    fn v1_manifest_snapshot_journal_fixture_remains_readable() {
        let dir = tempfile::tempdir().unwrap();
        let transaction = BatchTransaction::begin(
            dir.path(),
            "parent",
            0,
            serde_json::json!({"legacy": true}),
            vec![record("one.png", 0)],
        )
        .unwrap();
        let mut initial = transaction.manifest.clone();
        initial.version = 1;
        atomic_write_json(&transaction.attempt_dir.join(MANIFEST_FILE), &initial).unwrap();
        let initial_record = BatchJournalRecord {
            sequence: 0,
            attempt_generation: 0,
            event: BatchJournalEvent::ManifestSnapshot {
                manifest: initial.clone(),
            },
        };
        fs::write(
            transaction.attempt_dir.join(JOURNAL_FILE),
            format!("{}\n", serde_json::to_string(&initial_record).unwrap()),
        )
        .unwrap();
        let staged_path = transaction.staging_path(0).unwrap();
        fs::write(&staged_path, b"legacy").unwrap();
        let mut staged = initial;
        staged.children[0].checksum_sha256 = Some(checksum_bytes(b"legacy"));
        staged.children[0].size_bytes = Some(6);
        staged.children[0].record.file_size_bytes = Some(6);
        let staged_record = BatchJournalRecord {
            sequence: 1,
            attempt_generation: 0,
            event: BatchJournalEvent::ManifestSnapshot {
                manifest: staged.clone(),
            },
        };
        append_raw_batch_record(&transaction.attempt_dir, &staged_record);
        atomic_write_json(&transaction.attempt_dir.join(MANIFEST_FILE), &staged).unwrap();

        let loaded =
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap();

        assert_eq!(loaded.protocol_version(), 1);
        assert_eq!(loaded.manifest.children[0].size_bytes, Some(6));
    }
}
