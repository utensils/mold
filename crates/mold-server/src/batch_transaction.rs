//! Crash-recoverable, logically atomic publication for server-owned batches.
//!
//! Child artifacts stay below the gallery's reserved transaction directory
//! until every child is prepared. Publication then holds the exclusive side
//! of [`GalleryPublicationGate`] while it links every file with no-replace
//! semantics, commits all metadata rows in one SQLite transaction, and
//! durably advances the manifest to `committed`.

use anyhow::{ensure, Context};
use mold_db::GenerationRecord;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const TRANSACTION_DIR: &str = ".mold-batch-transactions";
const MANIFEST_FILE: &str = "manifest.json";
const JOURNAL_FILE: &str = "journal.jsonl";
const COMMITTED_DIR: &str = "committed";
const MANIFEST_VERSION: u32 = 1;
const DISK_SAFETY_FLOOR_BYTES: u64 = 64 * 1024 * 1024;

/// Readers cover gallery observation. Writers cover publication, deletion,
/// reconciliation, and corruption recovery.
#[derive(Clone, Default)]
pub struct GalleryPublicationGate {
    inner: Arc<tokio::sync::RwLock<()>>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchJournalRecord {
    sequence: u64,
    attempt_generation: u64,
    event: BatchJournalEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BatchJournalEvent {
    ManifestSnapshot { manifest: BatchAttemptManifest },
    FinalPublished { child_index: usize },
    MetadataCommitted,
    CleanupStarted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReservationOwner {
    parent_id: String,
    attempt_generation: u64,
}

pub(crate) struct GalleryNameReservation {
    output_dir: PathBuf,
    final_name: String,
    owner: ReservationOwner,
}

impl GalleryNameReservation {
    pub(crate) fn final_name(&self) -> &str {
        &self.final_name
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
    manifest: BatchAttemptManifest,
    next_journal_sequence: u64,
    journaled_final_children: BTreeSet<usize>,
    reconstructed_from_journal: bool,
    journal_needs_heal: bool,
    poisoned: bool,
    #[cfg(test)]
    commit_failpoint: Option<CommitFailpoint>,
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
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub rolled_back: usize,
    pub rolled_forward: usize,
    pub healed_committed_rows: usize,
}

impl BatchTransaction {
    /// Create a generation-scoped transaction and reserve collision-safe final
    /// names. The returned manifest is already durable.
    pub fn begin(
        output_dir: &Path,
        parent_id: &str,
        attempt_generation: u64,
        normalized_request: serde_json::Value,
        mut records: Vec<GenerationRecord>,
    ) -> anyhow::Result<Self> {
        validate_component(parent_id, "parent id")?;
        ensure!(!records.is_empty(), "batch transaction must have children");
        for record in &records {
            validate_component(&record.filename, "requested final filename")?;
        }
        fs::create_dir_all(output_dir)
            .with_context(|| format!("creating gallery {}", output_dir.display()))?;
        let attempt_dir = attempt_dir(output_dir, parent_id, attempt_generation);
        ensure!(
            !attempt_dir.exists(),
            "batch attempt already exists: {parent_id}/{attempt_generation}"
        );
        fs::create_dir_all(attempt_dir.join("staging"))?;
        fs::create_dir_all(reservations_dir(output_dir))?;
        sync_transaction_hierarchy(output_dir, &attempt_dir)?;

        let reservation_owner = ReservationOwner {
            parent_id: parent_id.to_owned(),
            attempt_generation,
        };
        let mut children = Vec::with_capacity(records.len());
        let mut reserved_names: Vec<String> = Vec::with_capacity(records.len());
        for (index, record) in records.iter_mut().enumerate() {
            let final_name =
                match reserve_final_name(output_dir, &record.filename, &reservation_owner) {
                    Ok(name) => name,
                    Err(error) => {
                        for name in &reserved_names {
                            let _ = release_reservation(output_dir, name, &reservation_owner);
                        }
                        let _ = fs::remove_dir_all(&attempt_dir);
                        return Err(error);
                    }
                };
            reserved_names.push(final_name.clone());
            record.filename.clone_from(&final_name);
            record.output_dir = output_dir.to_string_lossy().into_owned();
            children.push(BatchManifestChild {
                child_index: index,
                staging_name: format!("{index:08}.stage"),
                final_name,
                checksum_sha256: None,
                size_bytes: None,
                record: record.clone(),
            });
        }

        let manifest = BatchAttemptManifest {
            version: MANIFEST_VERSION,
            parent_id: parent_id.to_owned(),
            attempt_generation,
            normalized_request,
            state: BatchManifestState::Staging,
            children,
        };
        let mut transaction = Self {
            output_dir: output_dir.to_path_buf(),
            attempt_dir,
            manifest,
            next_journal_sequence: 0,
            journaled_final_children: BTreeSet::new(),
            reconstructed_from_journal: false,
            journal_needs_heal: false,
            poisoned: false,
            #[cfg(test)]
            commit_failpoint: None,
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
        self.ensure_usable()?;
        ensure!(
            self.manifest.state == BatchManifestState::Staging,
            "children can only be staged while the attempt is staging"
        );
        let path = self.staging_path(child_index)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("creating staged child {}", path.display()))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        sync_dir(path.parent().expect("staging path has parent"))?;

        let child = &mut self.manifest.children[child_index];
        child.checksum_sha256 = Some(checksum_bytes(bytes));
        child.size_bytes = Some(bytes.len() as u64);
        child.record.file_size_bytes = Some(bytes.len() as i64);
        self.persist_manifest()
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
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.commit_while_locked(&db)
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(UnresolvedBatchCommit::committing(error, guard)),
            Err(payload) => Err(UnresolvedBatchCommit::committing(
                anyhow::anyhow!(
                    "atomic batch commit panicked: {}",
                    panic_payload_message(payload.as_ref())
                ),
                guard,
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

    fn commit_while_locked(&mut self, db: &Arc<Option<mold_db::MetadataDb>>) -> anyhow::Result<()> {
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
            self.append_journal(BatchJournalEvent::FinalPublished { child_index })?;
            self.journaled_final_children.insert(child_index);
            self.inject_commit_error(CommitFailpoint::FinalJournalFsync(child_index))?;
        }
        sync_dir(&self.output_dir)?;
        self.inject_commit_error(CommitFailpoint::OutputDirectoryFsync)?;
        // Make the stat data used for the all-row transaction durable before
        // SQLite can observe it. Recovery replays this exact snapshot.
        self.persist_manifest()?;
        self.inject_commit_error(CommitFailpoint::MetadataManifestFsync)?;

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
        self.inject_commit_error(CommitFailpoint::DatabaseJournalFsync)?;

        self.manifest.state = BatchManifestState::Committed;
        self.persist_manifest()?;
        self.inject_commit_error(CommitFailpoint::CommittedState)?;
        self.release_reservations();
        self.cleanup_private_staging();
        if let Err(error) = self.archive_committed_attempt(db.is_none()) {
            tracing::warn!(
                attempt = %self.attempt_dir.display(),
                %error,
                "committed batch is durable but its recovery manifest could not be archived"
            );
        }
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
        let record = BatchJournalRecord {
            sequence: self.next_journal_sequence,
            attempt_generation: self.manifest.attempt_generation,
            event,
        };
        let path = self.attempt_dir.join(JOURNAL_FILE);
        let mut bytes = serde_json::to_vec(&record)?;
        bytes.push(b'\n');
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        sync_dir(&self.attempt_dir)?;
        self.next_journal_sequence = self
            .next_journal_sequence
            .checked_add(1)
            .context("batch journal sequence overflow")?;
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
        self.append_journal(BatchJournalEvent::CleanupStarted)?;
        if retain_manifest {
            let archive_dir = committed_manifests_dir(&self.output_dir, &self.manifest.parent_id);
            fs::create_dir_all(&archive_dir)?;
            sync_dir(
                archive_dir
                    .parent()
                    .context("committed manifest directory has no parent")?,
            )?;
            sync_dir(&archive_dir)?;
            atomic_write_json(
                &archive_dir.join(format!("{}.json", self.manifest.attempt_generation)),
                &self.manifest,
            )?;
        }
        fs::remove_dir_all(&self.attempt_dir)?;
        if let Some(attempts_dir) = self.attempt_dir.parent() {
            sync_dir(attempts_dir)?;
        }
        Ok(())
    }

    fn load(output_dir: &Path, manifest_path: &Path) -> anyhow::Result<Self> {
        let attempt_dir = manifest_path
            .parent()
            .context("manifest has no attempt directory")?
            .to_path_buf();
        let journal = load_journal(&attempt_dir.join(JOURNAL_FILE))?;
        let disk_manifest = fs::read(manifest_path).ok().and_then(|bytes| {
            let manifest = serde_json::from_slice::<BatchAttemptManifest>(&bytes).ok()?;
            match validate_loaded_manifest(output_dir, &attempt_dir, &manifest) {
                Ok(()) => Some(manifest),
                Err(error) => {
                    tracing::warn!(
                        manifest = %manifest_path.display(),
                        %error,
                        "ignoring invalid atomic batch manifest in favor of its journal"
                    );
                    None
                }
            }
        });
        let journal_manifest = journal.iter().rev().find_map(|record| match &record.event {
            BatchJournalEvent::ManifestSnapshot { manifest } => Some(manifest.clone()),
            _ => None,
        });
        if let Some(manifest) = journal_manifest.as_ref() {
            validate_loaded_manifest(output_dir, &attempt_dir, manifest).with_context(|| {
                format!(
                    "validating atomic batch journal snapshot {}",
                    attempt_dir.join(JOURNAL_FILE).display()
                )
            })?;
        }
        let (manifest, reconstructed_from_journal) = match (disk_manifest, journal_manifest.clone())
        {
            (Some(manifest), _) => (manifest, false),
            (None, Some(manifest)) => (manifest, true),
            (None, None) => {
                anyhow::bail!(
                    "batch attempt has neither a readable manifest nor a recoverable journal: {}; \
                     move this attempt directory out of {} after inspecting it",
                    attempt_dir.display(),
                    output_dir.join(TRANSACTION_DIR).display()
                )
            }
        };
        validate_loaded_manifest(output_dir, &attempt_dir, &manifest)?;
        for record in &journal {
            ensure!(
                record.attempt_generation == manifest.attempt_generation,
                "batch journal mixes attempt generations in {}",
                attempt_dir.display()
            );
            match &record.event {
                BatchJournalEvent::ManifestSnapshot { manifest: snapshot } => {
                    validate_loaded_manifest(output_dir, &attempt_dir, snapshot).with_context(
                        || {
                            format!(
                                "validating batch journal record {} in {}",
                                record.sequence,
                                attempt_dir.display()
                            )
                        },
                    )?
                }
                BatchJournalEvent::FinalPublished { child_index } => {
                    ensure!(
                        *child_index < manifest.children.len(),
                        "batch journal publishes out-of-range child {child_index} in {}",
                        attempt_dir.display()
                    );
                }
                BatchJournalEvent::MetadataCommitted | BatchJournalEvent::CleanupStarted => {}
            }
        }
        let journal_needs_heal = match journal_manifest.as_ref() {
            Some(journal_manifest) => {
                serde_json::to_value(journal_manifest)? != serde_json::to_value(&manifest)?
            }
            None => true,
        };
        ensure!(
            manifest.version == MANIFEST_VERSION,
            "unsupported batch manifest version {}",
            manifest.version
        );
        let next_journal_sequence = journal
            .last()
            .map_or(0, |record| record.sequence.saturating_add(1));
        let journaled_final_children = journal
            .iter()
            .filter_map(|record| match &record.event {
                BatchJournalEvent::FinalPublished { child_index } => Some(*child_index),
                _ => None,
            })
            .collect();
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            attempt_dir,
            manifest,
            next_journal_sequence,
            journaled_final_children,
            reconstructed_from_journal,
            journal_needs_heal,
            poisoned: false,
            #[cfg(test)]
            commit_failpoint: None,
        })
    }
}

/// An error before the manifest enters `committing` is ordinary. An error
/// after that point owns the writer guard and aborts if dropped.
#[must_use = "an unresolved committing transaction must terminate or be recovered"]
pub struct UnresolvedBatchCommit {
    error: anyhow::Error,
    writer: Option<tokio::sync::OwnedRwLockWriteGuard<()>>,
}

impl UnresolvedBatchCommit {
    fn pre_commit(error: anyhow::Error) -> Self {
        Self {
            error,
            writer: None,
        }
    }

    fn committing(error: anyhow::Error, writer: tokio::sync::OwnedRwLockWriteGuard<()>) -> Self {
        Self {
            error,
            writer: Some(writer),
        }
    }

    pub fn entered_committing(&self) -> bool {
        self.writer.is_some()
    }

    /// Startup recovery runs before the listener binds, so it may release the
    /// barrier and return a normal boot error for an operator-recoverable
    /// filesystem/SQLite failure. Live serving code must never call this.
    fn into_startup_error(mut self) -> anyhow::Error {
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
    let root = output_dir.join(TRANSACTION_DIR);
    if !root.is_dir() {
        return Ok(RecoveryReport::default());
    }
    sweep_stale_reservations(&root)?;
    let mut manifests = Vec::new();
    collect_manifests(&root, &mut manifests)?;
    manifests.sort();

    let mut report = RecoveryReport::default();
    for path in manifests {
        let mut transaction = BatchTransaction::load(output_dir, &path)?;
        if transaction.reconstructed_from_journal || transaction.journal_needs_heal {
            transaction.persist_manifest().with_context(|| {
                format!(
                    "rewriting batch manifest reconstructed from {}",
                    transaction.attempt_dir.join(JOURNAL_FILE).display()
                )
            })?;
            transaction.reconstructed_from_journal = false;
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
                if let Err(error) = transaction.verify_all_staged() {
                    if !transaction.any_final_exists() {
                        let _guard = gate.write().await;
                        transaction.manifest.state = BatchManifestState::Failed;
                        transaction.persist_manifest()?;
                        transaction.release_reservations();
                        remove_failed_attempt(&transaction);
                        report.rolled_back += 1;
                        tracing::warn!(
                            attempt = %transaction.attempt_dir.display(),
                            %error,
                            "rolled back unverifiable committing attempt before any final publication"
                        );
                        continue;
                    }
                    return Err(error).with_context(|| {
                        format!(
                            "committing attempt {} has published files but unverifiable staging; \
                             inspect the attempt before restarting",
                            transaction.attempt_dir.display()
                        )
                    });
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
                if let Err(error) = transaction.archive_committed_attempt(db.is_none()) {
                    tracing::warn!(
                        attempt = %transaction.attempt_dir.display(),
                        %error,
                        "could not archive recovered committed batch manifest"
                    );
                }
            }
            BatchManifestState::Failed => {
                transaction.release_reservations();
                transaction.cleanup_private_staging();
                remove_failed_attempt(&transaction);
            }
        }
    }
    Ok(report)
}

fn collect_manifests(root: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    // Only generation-scoped active attempts participate in startup
    // recovery. Archived committed manifests and reservation metadata are
    // durable authority, not work to replay on every boot.
    for parent in fs::read_dir(root)? {
        let parent = parent?;
        if !parent.path().is_dir() || parent.file_name() == "reservations" {
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
            let manifest = attempt.path().join(MANIFEST_FILE);
            let journal = attempt.path().join(JOURNAL_FILE);
            if manifest.is_file() || journal.is_file() {
                out.push(manifest);
            } else {
                let owner = ReservationOwner {
                    parent_id: parent.file_name().to_string_lossy().into_owned(),
                    attempt_generation: attempt
                        .file_name()
                        .to_string_lossy()
                        .parse()
                        .with_context(|| {
                            format!(
                                "invalid orphaned batch attempt generation {}",
                                attempt.path().display()
                            )
                        })?,
                };
                tracing::warn!(
                    attempt = %attempt.path().display(),
                    "removing orphaned batch attempt directory with no manifest or journal"
                );
                release_all_reservations_for_owner(root, &owner)?;
                fs::remove_dir_all(attempt.path())?;
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

fn committed_manifests_dir(output_dir: &Path, parent_id: &str) -> PathBuf {
    output_dir
        .join(TRANSACTION_DIR)
        .join(parent_id)
        .join(COMMITTED_DIR)
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
                    sync_dir(&reservations_dir(output_dir))
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

pub(crate) fn reserve_gallery_final_name(
    output_dir: &Path,
    desired: &str,
) -> anyhow::Result<GalleryNameReservation> {
    fs::create_dir_all(reservations_dir(output_dir))?;
    let owner = ReservationOwner {
        parent_id: format!("ordinary:{}", uuid::Uuid::new_v4()),
        attempt_generation: 0,
    };
    let final_name = reserve_final_name(output_dir, desired, &owner)?;
    Ok(GalleryNameReservation {
        output_dir: output_dir.to_path_buf(),
        final_name,
        owner,
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
                break;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("parsing batch journal {}", path.display()));
            }
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
        manifest.version == MANIFEST_VERSION,
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

    let expected_output_dir = output_dir.to_string_lossy();
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
        ensure!(
            child.record.output_dir == expected_output_dir.as_ref(),
            "batch metadata output directory does not match {}",
            output_dir.display()
        );
        match (&child.checksum_sha256, child.size_bytes) {
            (None, None) => ensure!(
                manifest.state == BatchManifestState::Staging,
                "non-staging batch child {} has no checksum",
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

pub(crate) fn sync_gallery_directory(path: &Path) -> anyhow::Result<()> {
    sync_dir(path)
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

    #[tokio::test]
    async fn db_disabled_commit_publishes_all_children_and_durable_manifest() {
        let dir = tempfile::tempdir().unwrap();
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

        transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();

        assert_eq!(fs::read(dir.path().join("same.png")).unwrap(), b"first");
        assert_eq!(fs::read(dir.path().join("other.png")).unwrap(), b"second");
        assert_eq!(transaction.manifest().state, BatchManifestState::Committed);
        assert!(!transaction.attempt_dir.exists());
        let archive = committed_manifests_dir(dir.path(), "parent").join("0.json");
        let archived: BatchAttemptManifest =
            serde_json::from_slice(&fs::read(archive).unwrap()).unwrap();
        assert_eq!(archived.state, BatchManifestState::Committed);
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
        loaded.persist_manifest().unwrap();

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
        let retry = BatchTransaction::begin(
            dir.path(),
            "parent",
            5,
            serde_json::json!({}),
            vec![record("retry.png", 0)],
        )
        .unwrap();

        let root = dir.path().join(TRANSACTION_DIR);
        let mut manifests = Vec::new();
        collect_manifests(&root, &mut manifests).unwrap();

        assert!(!orphan.exists());
        assert!(!reservation_path(dir.path(), "orphan.png").exists());
        assert!(reservation_path(dir.path(), "retry.png").exists());
        assert_eq!(
            manifests,
            vec![retry.attempt_dir.join(MANIFEST_FILE)],
            "the live retry remains available to normal recovery"
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
        transaction.stage_bytes(0, b"one").unwrap();
        transaction.mark_prepared().unwrap();
        let staged = transaction.staging_path(0).unwrap();
        fs::hard_link(&staged, dir.path().join("one.png")).unwrap();
        transaction.manifest.state = BatchManifestState::Committed;
        transaction.persist_manifest().unwrap();
        fs::remove_file(dir.path().join("one.png")).unwrap();

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
        // Production drops this value and aborts. The test intentionally
        // leaks it so it can assert the fatal classification in-process.
        std::mem::forget(error);
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
            assert!(startup_error.to_string().contains("injected commit fault"));

            let report = recover_transactions(dir.path(), &gate, db.clone())
                .await
                .unwrap();
            assert!(
                report.rolled_forward == 1 || report.healed_committed_rows == 2,
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

    #[test]
    fn disk_preflight_keeps_encoder_and_manifest_headroom() {
        let expected = 100 * 1024 * 1024;
        assert!(validate_available_space(expected, expected).is_err());
        assert!(validate_available_space(expected * 2 + DISK_SAFETY_FLOOR_BYTES, expected).is_ok());
        assert!(validate_available_space(u64::MAX, u64::MAX).is_err());
    }
}
