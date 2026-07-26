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
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const TRANSACTION_DIR: &str = ".mold-batch-transactions";
const MANIFEST_FILE: &str = "manifest.json";
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
}

/// Refuse a batch before inference when the gallery filesystem cannot hold
/// the expected staged children plus a bounded safety margin. Hard links make
/// publication itself space-neutral, but encoders and manifest replacement
/// still need headroom.
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
        .context("gallery filesystem was not present in the disk inventory")?;
    validate_available_space(available, expected_staging_bytes)
}

fn validate_available_space(available: u64, expected_staging_bytes: u64) -> anyhow::Result<()> {
    let margin = DISK_SAFETY_FLOOR_BYTES.max(expected_staging_bytes / 20);
    let required = expected_staging_bytes
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

#[derive(Debug)]
pub struct BatchTransaction {
    output_dir: PathBuf,
    attempt_dir: PathBuf,
    manifest: BatchAttemptManifest,
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

        let mut children = Vec::with_capacity(records.len());
        let mut reserved_names: Vec<String> = Vec::with_capacity(records.len());
        for (index, record) in records.iter_mut().enumerate() {
            let final_name = match reserve_final_name(output_dir, &record.filename) {
                Ok(name) => name,
                Err(error) => {
                    for name in &reserved_names {
                        let _ = fs::remove_file(reservation_path(output_dir, name));
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
        let transaction = Self {
            output_dir: output_dir.to_path_buf(),
            attempt_dir,
            manifest,
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
        self.commit_while_locked(&db)
            .map_err(|error| UnresolvedBatchCommit::committing(error, guard))
    }

    fn validate_commit_entry(&self) -> anyhow::Result<()> {
        ensure!(
            matches!(
                self.manifest.state,
                BatchManifestState::Prepared | BatchManifestState::Committing
            ),
            "batch attempt is not prepared for commit"
        );
        self.verify_all_staged()
    }

    fn commit_while_locked(&mut self, db: &Arc<Option<mold_db::MetadataDb>>) -> anyhow::Result<()> {
        if self.manifest.state == BatchManifestState::Prepared {
            self.manifest.state = BatchManifestState::Committing;
            self.persist_manifest()?;
        }

        for child in &self.manifest.children {
            let staged = self.attempt_dir.join("staging").join(&child.staging_name);
            let final_path = self.output_dir.join(&child.final_name);
            if final_path.exists() {
                ensure!(
                    checksum_file(&final_path)? == child.checksum_sha256.as_deref().unwrap(),
                    "reserved final path contains different bytes: {}",
                    final_path.display()
                );
            } else {
                fs::hard_link(&staged, &final_path).with_context(|| {
                    format!("publishing {} without replacement", final_path.display())
                })?;
            }
            File::open(&final_path)?.sync_all()?;
        }
        sync_dir(&self.output_dir)?;

        if let Some(db) = db.as_ref() {
            let records: Vec<_> = self
                .manifest
                .children
                .iter()
                .map(|child| child.record.clone())
                .collect();
            db.upsert_batch(&records)?;
        }

        self.manifest.state = BatchManifestState::Committed;
        self.persist_manifest()?;
        self.release_reservations();
        self.cleanup_private_staging();
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

    fn persist_manifest(&self) -> anyhow::Result<()> {
        atomic_write_json(&self.attempt_dir.join(MANIFEST_FILE), &self.manifest)
    }

    fn release_reservations(&self) {
        for child in &self.manifest.children {
            let _ = fs::remove_file(reservation_path(&self.output_dir, &child.final_name));
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

    fn load(output_dir: &Path, manifest_path: &Path) -> anyhow::Result<Self> {
        let bytes = fs::read(manifest_path)?;
        let manifest: BatchAttemptManifest = serde_json::from_slice(&bytes)?;
        ensure!(
            manifest.version == MANIFEST_VERSION,
            "unsupported batch manifest version {}",
            manifest.version
        );
        let attempt_dir = manifest_path
            .parent()
            .context("manifest has no attempt directory")?
            .to_path_buf();
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            attempt_dir,
            manifest,
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
        if self.writer.is_some() && !std::thread::panicking() {
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
    let mut manifests = Vec::new();
    collect_manifests(&root, &mut manifests)?;
    manifests.sort();

    let mut report = RecoveryReport::default();
    for path in manifests {
        let mut transaction = BatchTransaction::load(output_dir, &path)?;
        match transaction.manifest.state {
            BatchManifestState::Staging | BatchManifestState::Prepared => {
                let _guard = gate.write().await;
                transaction.manifest.state = BatchManifestState::Failed;
                transaction.persist_manifest()?;
                transaction.release_reservations();
                let _ = fs::remove_dir_all(transaction.attempt_dir.join("staging"));
                report.rolled_back += 1;
            }
            BatchManifestState::Committing => {
                transaction
                    .commit(gate, db.clone())
                    .await
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
                report.rolled_forward += 1;
            }
            BatchManifestState::Committed => {
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
            }
            BatchManifestState::Failed => {
                transaction.release_reservations();
                transaction.cleanup_private_staging();
            }
        }
    }
    Ok(report)
}

fn collect_manifests(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_manifests(&path, out)?;
        } else if entry.file_name() == MANIFEST_FILE {
            out.push(path);
        }
    }
    Ok(())
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

fn reservation_path(output_dir: &Path, final_name: &str) -> PathBuf {
    reservations_dir(output_dir).join(format!("{final_name}.reserve"))
}

fn reserve_final_name(output_dir: &Path, desired: &str) -> anyhow::Result<String> {
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
            Ok(file) => {
                file.sync_all()?;
                sync_dir(&reservations_dir(output_dir))?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("u32 collision namespace exhausted")
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

fn checksum_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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

fn atomic_write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let parent = path.parent().context("manifest path has no parent")?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".{MANIFEST_FILE}.tmp-{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(value)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temp, path)?;
    sync_dir(parent)
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
        assert!(!transaction.attempt_dir.join("staging").exists());
        let reloaded =
            BatchTransaction::load(dir.path(), &transaction.attempt_dir.join(MANIFEST_FILE))
                .unwrap();
        assert_eq!(reloaded.manifest.state, BatchManifestState::Committed);
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
        assert_eq!(second.healed_committed_rows, 2);
        assert_eq!(db.as_ref().as_ref().unwrap().count().unwrap(), 2);
        assert_eq!(fs::read(dir.path().join("two.png")).unwrap(), b"two");
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

    #[test]
    fn disk_preflight_keeps_encoder_and_manifest_headroom() {
        let expected = 100 * 1024 * 1024;
        assert!(validate_available_space(expected, expected).is_err());
        assert!(validate_available_space(expected + DISK_SAFETY_FLOOR_BYTES, expected).is_ok());
        assert!(validate_available_space(u64::MAX, u64::MAX).is_err());
    }
}
