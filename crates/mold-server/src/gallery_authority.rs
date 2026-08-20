//! Cross-process authority for committed gallery metadata.
//!
//! The SQLite gallery is a projection, not publication authority. This file
//! owns the compact, checksummed snapshot and the one-entry write-ahead log
//! used to make exact-name metadata mutations visible across server
//! processes. Every caller must already own the gallery bookkeeping guard.

use crate::batch_transaction::{
    CommittedArchiveEntry, CommittedArchiveIndex, GalleryBookkeepingGuard,
};
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const AUTHORITY_DIR: &str = "gallery-authority-v2";
const CHECKPOINT_FILE: &str = "checkpoint.json";
const PREVIOUS_CHECKPOINT_FILE: &str = "checkpoint.previous.json";
const BACKUP_CHECKPOINT_FILE: &str = "checkpoint.backup.json";
const MARKER_FILE: &str = "generation.json";
const WAL_FILE: &str = "mutation.wal";
const STORAGE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuthoritySnapshot {
    version: u32,
    generation: u64,
    index: CommittedArchiveIndex,
    #[serde(default)]
    legacy_evidence_epochs: std::collections::BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize)]
struct ChecksummedSnapshot {
    version: u32,
    payload_sha256: String,
    snapshot: AuthoritySnapshot,
}

#[derive(Debug, Deserialize)]
struct RawChecksummedSnapshot {
    version: u32,
    payload_sha256: String,
    snapshot: Box<serde_json::value::RawValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MutationMarker {
    version: u32,
    committed_generation: u64,
    pending: Option<PendingMutation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingMutation {
    generation: u64,
    kind: String,
    exact_names: Vec<String>,
    snapshot_sha256: String,
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedAuthority {
    pub(crate) generation: u64,
    pub(crate) index: CommittedArchiveIndex,
    pub(crate) stats: ValidationStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ValidationStats {
    pub(crate) files_statted: usize,
    pub(crate) files_hashed: usize,
    pub(crate) quarantined: usize,
    pub(crate) reactivated: usize,
}

pub(crate) fn authority_dir_name() -> &'static str {
    AUTHORITY_DIR
}

fn authority_dir(root: &Path) -> PathBuf {
    root.join(crate::batch_transaction::TRANSACTION_DIR)
        .join(AUTHORITY_DIR)
}

fn checkpoint_path(root: &Path) -> PathBuf {
    authority_dir(root).join(CHECKPOINT_FILE)
}

fn previous_checkpoint_path(root: &Path) -> PathBuf {
    authority_dir(root).join(PREVIOUS_CHECKPOINT_FILE)
}

fn backup_checkpoint_path(root: &Path) -> PathBuf {
    authority_dir(root).join(BACKUP_CHECKPOINT_FILE)
}

fn marker_path(root: &Path) -> PathBuf {
    authority_dir(root).join(MARKER_FILE)
}

fn wal_path(root: &Path) -> PathBuf {
    authority_dir(root).join(WAL_FILE)
}

fn digest_json<T: Serialize>(value: &T) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn wrap_snapshot(snapshot: AuthoritySnapshot) -> anyhow::Result<ChecksummedSnapshot> {
    Ok(ChecksummedSnapshot {
        version: STORAGE_VERSION,
        payload_sha256: digest_json(&snapshot)?,
        snapshot,
    })
}

fn validate_envelope(
    envelope: RawChecksummedSnapshot,
    path: &Path,
) -> anyhow::Result<AuthoritySnapshot> {
    ensure!(
        envelope.version == STORAGE_VERSION,
        "unsupported gallery authority checkpoint version in {}",
        path.display()
    );
    ensure!(
        format!("{:x}", Sha256::digest(envelope.snapshot.get().as_bytes()))
            == envelope.payload_sha256,
        "gallery authority checkpoint checksum mismatch in {}",
        path.display()
    );
    let snapshot: AuthoritySnapshot = serde_json::from_str(envelope.snapshot.get())
        .with_context(|| format!("reading gallery authority snapshot in {}", path.display()))?;
    ensure!(
        snapshot.version == STORAGE_VERSION,
        "unsupported gallery authority checkpoint version in {}",
        path.display()
    );
    Ok(snapshot)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    serde_json::from_reader(crate::batch_transaction::open_regular_file_no_follow(path)?)
        .with_context(|| format!("reading gallery authority {}", path.display()))
}

fn read_checkpoint_at(path: &Path) -> anyhow::Result<AuthoritySnapshot> {
    validate_envelope(read_json(path)?, path)
}

fn read_checkpoint(root: &Path) -> anyhow::Result<Option<AuthoritySnapshot>> {
    let candidates = [
        ("current", checkpoint_path(root)),
        ("backup", backup_checkpoint_path(root)),
        ("previous", previous_checkpoint_path(root)),
    ];
    let mut errors = Vec::new();
    let mut any_present = false;
    for (label, path) in candidates {
        match read_checkpoint_at(&path) {
            Ok(snapshot) => {
                if label != "current" {
                    tracing::warn!(
                        checkpoint = label,
                        "using fallback checksummed gallery authority checkpoint"
                    );
                }
                return Ok(Some(snapshot));
            }
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
            {
                errors.push(format!("{label}: absent"));
            }
            Err(error) => {
                any_present = true;
                errors.push(format!("{label}: {error:#}"));
            }
        }
    }
    if any_present {
        Err(anyhow::anyhow!(
            "all gallery authority checkpoints are invalid: {}",
            errors.join("; ")
        ))
    } else {
        Ok(None)
    }
}

fn read_marker(root: &Path) -> anyhow::Result<Option<MutationMarker>> {
    let path = marker_path(root);
    match read_json::<MutationMarker>(&path) {
        Ok(marker) => {
            ensure!(
                marker.version == STORAGE_VERSION,
                "unsupported gallery authority generation marker in {}",
                path.display()
            );
            Ok(Some(marker))
        }
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn sync_dir(path: &Path) -> anyhow::Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let parent = path.parent().context("authority path has no parent")?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .context("authority path has no filename")?
            .to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    let result = (|| -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        serde_json::to_writer(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        sync_dir(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

fn write_checkpoint(root: &Path, snapshot: &AuthoritySnapshot) -> anyhow::Result<()> {
    let dir = authority_dir(root);
    fs::create_dir_all(&dir)?;
    sync_dir(
        dir.parent()
            .context("gallery authority directory has no parent")?,
    )?;
    sync_dir(&dir)?;
    let current = checkpoint_path(root);
    let previous = previous_checkpoint_path(root);
    if current.is_file() {
        let existing = read_checkpoint_at(&current)
            .or_else(|_| read_checkpoint_at(&backup_checkpoint_path(root)))?;
        ensure!(
            existing.generation <= snapshot.generation,
            "gallery authority checkpoint generation regressed from {} to {}",
            existing.generation,
            snapshot.generation
        );
        if existing.generation < snapshot.generation {
            atomic_write_json(&previous, &wrap_snapshot(existing)?)?;
        }
    }
    atomic_write_json(&current, &wrap_snapshot(snapshot.clone())?)
}

fn backup_checkpoint(root: &Path, snapshot: &AuthoritySnapshot) -> anyhow::Result<()> {
    atomic_write_json(
        &backup_checkpoint_path(root),
        &wrap_snapshot(snapshot.clone())?,
    )
}

fn write_marker(root: &Path, marker: &MutationMarker) -> anyhow::Result<()> {
    atomic_write_json(&marker_path(root), marker)
}

fn remove_wal(root: &Path) -> anyhow::Result<()> {
    match fs::remove_file(wal_path(root)) {
        Ok(()) => sync_dir(&authority_dir(root)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn recover_storage(root: &Path) -> anyhow::Result<Option<AuthoritySnapshot>> {
    let checkpoint = read_checkpoint(root)?;
    let marker = read_marker(root)?;
    let wal = match read_checkpoint_at(&wal_path(root)) {
        Ok(snapshot) => Some(snapshot),
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            None
        }
        Err(error) => return Err(error.context("recovering gallery authority mutation WAL")),
    };

    let mut current = checkpoint;
    let committed_generation = marker
        .as_ref()
        .map(|marker| marker.committed_generation)
        .unwrap_or_else(|| current.as_ref().map_or(0, |snapshot| snapshot.generation));
    if marker
        .as_ref()
        .is_some_and(|marker| marker.pending.is_none())
        && current
            .as_ref()
            .is_some_and(|snapshot| snapshot.generation != committed_generation)
    {
        anyhow::bail!("gallery authority checkpoint generation does not match its stable marker");
    }
    if let Some(pending) = marker.as_ref().and_then(|marker| marker.pending.as_ref()) {
        if let Some(wal_snapshot) = wal.as_ref().filter(|wal| {
            wal.generation == pending.generation
                && digest_json(*wal).is_ok_and(|digest| digest == pending.snapshot_sha256)
        }) {
            write_checkpoint(root, wal_snapshot)?;
            write_marker(
                root,
                &MutationMarker {
                    version: STORAGE_VERSION,
                    committed_generation: wal_snapshot.generation,
                    pending: None,
                },
            )?;
            remove_wal(root)?;
            backup_checkpoint(root, wal_snapshot)?;
            current = Some(wal_snapshot.clone());
        } else {
            write_marker(
                root,
                &MutationMarker {
                    version: STORAGE_VERSION,
                    committed_generation,
                    pending: None,
                },
            )?;
            remove_wal(root)?;
        }
    } else if let Some(wal_snapshot) = wal {
        if wal_snapshot.generation == committed_generation.saturating_add(1) {
            write_checkpoint(root, &wal_snapshot)?;
            write_marker(
                root,
                &MutationMarker {
                    version: STORAGE_VERSION,
                    committed_generation: wal_snapshot.generation,
                    pending: None,
                },
            )?;
            backup_checkpoint(root, &wal_snapshot)?;
            current = Some(wal_snapshot);
        }
        remove_wal(root)?;
    }
    if marker.is_none() {
        if let Some(snapshot) = current.as_ref() {
            write_marker(
                root,
                &MutationMarker {
                    version: STORAGE_VERSION,
                    committed_generation: snapshot.generation,
                    pending: None,
                },
            )?;
        }
    }
    Ok(current)
}

pub(crate) fn read_generation(
    root: &Path,
    guard: &GalleryBookkeepingGuard,
) -> anyhow::Result<Option<u64>> {
    guard.ensure_root(root)?;
    let root = guard.canonical_root();
    Ok(read_marker(root)?.map(|marker| {
        marker
            .pending
            .map_or(marker.committed_generation, |pending| pending.generation)
    }))
}

pub(crate) fn load_or_initialize(
    root: &Path,
    guard: &GalleryBookkeepingGuard,
    legacy: impl FnOnce() -> anyhow::Result<CommittedArchiveIndex>,
) -> anyhow::Result<LoadedAuthority> {
    guard.ensure_root(root)?;
    let root = guard.canonical_root();
    let mut snapshot = match recover_storage(root)? {
        Some(snapshot) => snapshot,
        None => {
            let mut index = legacy()?;
            populate_missing_facts(root, &mut index)?;
            let legacy_evidence_epochs =
                crate::batch_transaction::legacy_gallery_evidence_paths(root)?
                    .into_iter()
                    .map(|path| (path, 0))
                    .collect();
            let snapshot = AuthoritySnapshot {
                version: STORAGE_VERSION,
                generation: 0,
                index,
                legacy_evidence_epochs,
            };
            write_checkpoint(root, &snapshot)?;
            atomic_write_json(
                &previous_checkpoint_path(root),
                &wrap_snapshot(snapshot.clone())?,
            )?;
            backup_checkpoint(root, &snapshot)?;
            write_marker(
                root,
                &MutationMarker {
                    version: STORAGE_VERSION,
                    committed_generation: 0,
                    pending: None,
                },
            )?;
            snapshot
        }
    };
    let (stats, changed) = validate_snapshot_files(root, &mut snapshot.index)?;
    if changed {
        let exact_names = snapshot
            .index
            .quarantined_names
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        snapshot.generation = commit_snapshot(
            root,
            guard,
            snapshot.generation,
            &mut snapshot.index,
            "startup_validation",
            exact_names,
        )?;
    }
    for _ in 0..2 {
        if crate::batch_transaction::legacy_gallery_evidence_paths(root)?.is_empty() {
            break;
        }
        snapshot.generation = commit_snapshot(
            root,
            guard,
            snapshot.generation,
            &mut snapshot.index,
            "legacy_evidence_gc_epoch",
            Vec::new(),
        )?;
    }
    Ok(LoadedAuthority {
        generation: snapshot.generation,
        index: snapshot.index,
        stats,
    })
}

pub(crate) fn commit_snapshot(
    root: &Path,
    guard: &GalleryBookkeepingGuard,
    expected_generation: u64,
    index: &mut CommittedArchiveIndex,
    kind: &str,
    exact_names: Vec<String>,
) -> anyhow::Result<u64> {
    guard.ensure_root(root)?;
    let root = guard.canonical_root();
    let current = recover_storage(root)?.context("gallery authority checkpoint is missing")?;
    ensure!(
        current.generation == expected_generation,
        "gallery authority generation changed from {expected_generation} to {}",
        current.generation
    );
    let generation = expected_generation
        .checked_add(1)
        .context("gallery authority generation overflow")?;
    let mut legacy_evidence_epochs = current.legacy_evidence_epochs;
    let observed_legacy = crate::batch_transaction::legacy_gallery_evidence_paths(root)?;
    legacy_evidence_epochs.retain(|path, _| observed_legacy.contains(path));
    for path in &observed_legacy {
        legacy_evidence_epochs
            .entry(path.clone())
            .or_insert(generation);
    }
    let collect_legacy = legacy_evidence_epochs
        .iter()
        .filter(|(_, first_generation)| **first_generation < generation)
        .map(|(path, _)| path.clone())
        .collect::<std::collections::BTreeSet<_>>();
    for path in &collect_legacy {
        legacy_evidence_epochs.remove(path);
    }

    let collect_retirements = index
        .retirement_projection_epochs
        .iter()
        .filter(|(name, projected_generation)| {
            **projected_generation < generation
                && index
                    .retirement_epochs
                    .get(*name)
                    .is_some_and(|retired_generation| retired_generation <= *projected_generation)
        })
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    for name in collect_retirements {
        index.retired_names.remove(&name);
        index.retired_entries.remove(&name);
        index.retirement_epochs.remove(&name);
        index.retirement_projection_epochs.remove(&name);
    }

    let snapshot = AuthoritySnapshot {
        version: STORAGE_VERSION,
        generation,
        index: index.clone(),
        legacy_evidence_epochs,
    };
    let snapshot_sha256 = digest_json(&snapshot)?;
    guard.ensure_root(root)?;
    write_marker(
        root,
        &MutationMarker {
            version: STORAGE_VERSION,
            committed_generation: expected_generation,
            pending: Some(PendingMutation {
                generation,
                kind: kind.to_owned(),
                exact_names,
                snapshot_sha256: snapshot_sha256.clone(),
            }),
        },
    )?;
    guard.ensure_root(root)?;
    atomic_write_json(&wal_path(root), &wrap_snapshot(snapshot.clone())?)?;
    guard.ensure_root(root)?;
    write_checkpoint(root, &snapshot)?;
    guard.ensure_root(root)?;
    write_marker(
        root,
        &MutationMarker {
            version: STORAGE_VERSION,
            committed_generation: generation,
            pending: None,
        },
    )?;
    guard.ensure_root(root)?;
    remove_wal(root)?;
    guard.ensure_root(root)?;
    backup_checkpoint(root, &snapshot)?;
    crate::batch_transaction::remove_legacy_gallery_evidence(root, &collect_legacy)?;
    Ok(generation)
}

fn populate_missing_facts(root: &Path, index: &mut CommittedArchiveIndex) -> anyhow::Result<()> {
    for (name, entry) in &mut index.entries {
        if entry.facts.is_none() {
            let path = root.join(name);
            if let Ok(metadata) = fs::symlink_metadata(&path) {
                if metadata.is_file() && !metadata.file_type().is_symlink() {
                    entry.facts = Some(crate::batch_transaction::ArchiveFileFacts::from_metadata(
                        &metadata,
                    )?);
                }
            }
        }
    }
    Ok(())
}

fn validate_snapshot_files(
    root: &Path,
    index: &mut CommittedArchiveIndex,
) -> anyhow::Result<(ValidationStats, bool)> {
    let mut stats = ValidationStats::default();
    let mut changed = false;
    let names = index.entries.keys().cloned().collect::<Vec<_>>();
    for name in names {
        let Some(entry) = index.entries.get_mut(&name) else {
            continue;
        };
        let path = root.join(&name);
        stats.files_statted += 1;
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
            Ok(_) => {
                changed |= index.quarantined_names.insert(name.clone());
                stats.quarantined += 1;
                continue;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                changed |= index.quarantined_names.insert(name.clone());
                stats.quarantined += 1;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let facts = crate::batch_transaction::ArchiveFileFacts::from_metadata(&metadata)?;
        let stable = entry.facts.as_ref().is_some_and(|prior| prior == &facts)
            && metadata.len() == entry.identity.size_bytes;
        let valid = if stable {
            true
        } else if metadata.len() != entry.identity.size_bytes {
            false
        } else {
            stats.files_hashed += 1;
            crate::batch_transaction::checksum_file_for_authority(&path)?
                == entry.identity.checksum_sha256
        };
        if valid {
            entry.facts = Some(facts);
            if index.quarantined_names.remove(&name) {
                stats.reactivated += 1;
                changed = true;
            }
        } else {
            changed |= index.quarantined_names.insert(name.clone());
            stats.quarantined += 1;
        }
    }
    let retired = index
        .retired_entries
        .iter()
        .map(|(name, entry)| (name.clone(), entry.clone()))
        .collect::<Vec<_>>();
    let mut removed = false;
    for (name, entry) in retired {
        let path = root.join(&name);
        stats.files_statted += 1;
        match fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) if current_file_matches(root, &entry)? => {
                fs::remove_file(&path).with_context(|| {
                    format!(
                        "finishing crash-interrupted gallery delete {}",
                        path.display()
                    )
                })?;
                removed = true;
            }
            Ok(_) => {
                changed |= index.quarantined_names.insert(name);
                stats.quarantined += 1;
            }
            Err(error) => return Err(error.into()),
        }
    }
    if removed {
        sync_dir(root)?;
    }
    Ok((stats, changed))
}

pub(crate) fn current_file_matches(
    root: &Path,
    entry: &CommittedArchiveEntry,
) -> anyhow::Result<bool> {
    file_matches_entry_at(&root.join(&entry.identity.final_name), entry)
}

/// Whether the regular file at `path` carries exactly the committed identity
/// of `entry` (size, then retained facts or a fresh checksum). Used at the
/// live path by [`current_file_matches`] and at `<dir>/.trash/<name>` when a
/// trashed print is restored.
pub(crate) fn file_matches_entry_at(
    path: &Path,
    entry: &CommittedArchiveEntry,
) -> anyhow::Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Ok(_) => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if metadata.len() != entry.identity.size_bytes {
        return Ok(false);
    }
    let facts = crate::batch_transaction::ArchiveFileFacts::from_metadata(&metadata)?;
    if entry.facts.as_ref().is_some_and(|prior| prior == &facts) {
        return Ok(true);
    }
    Ok(crate::batch_transaction::checksum_file_for_authority(path)?
        == entry.identity.checksum_sha256)
}

#[cfg(test)]
pub(crate) fn storage_file_count(root: &Path) -> usize {
    fs::read_dir(authority_dir(root))
        .map(|entries| entries.filter_map(Result::ok).count())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_transaction::{
        acquire_gallery_bookkeeping_lock, ArchiveFileFacts, ArchivedChildIdentity,
    };
    use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
    use mold_db::{GenerationRecord, RecordSource};

    fn empty_snapshot(generation: u64) -> AuthoritySnapshot {
        AuthoritySnapshot {
            version: STORAGE_VERSION,
            generation,
            index: CommittedArchiveIndex::default(),
            legacy_evidence_epochs: Default::default(),
        }
    }

    #[test]
    fn accepts_valid_checkpoint_from_before_additive_record_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CHECKPOINT_FILE);
        let filename = "legacy.png";
        let media_path = dir.path().join(filename);
        fs::write(&media_path, b"legacy").unwrap();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "legacy prompt",
            "model": "test",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0
        }))
        .unwrap();
        let mut record = GenerationRecord::from_save(
            dir.path(),
            filename,
            OutputFormat::Png,
            OutputMetadata::from_generate_request(&request, 1, None, "test"),
            RecordSource::Server,
            1,
        );
        record.stat_from_disk(&media_path);
        let mut index = CommittedArchiveIndex::default();
        index.entries.insert(
            filename.to_owned(),
            CommittedArchiveEntry {
                identity: ArchivedChildIdentity {
                    parent_id: "legacy".into(),
                    attempt_generation: 0,
                    child_index: 0,
                    final_name: filename.into(),
                    checksum_sha256: format!("{:x}", Sha256::digest(b"legacy")),
                    size_bytes: 6,
                },
                record,
                facts: Some(ArchiveFileFacts::from_path(&media_path).unwrap()),
            },
        );
        let snapshot = AuthoritySnapshot {
            version: STORAGE_VERSION,
            generation: 7,
            index,
            legacy_evidence_epochs: Default::default(),
        };
        let mut legacy = serde_json::to_value(snapshot).unwrap();
        let record = legacy["index"]["entries"][filename]["record"]
            .as_object_mut()
            .unwrap();
        record.remove("title");
        record.remove("favorite");
        record.remove("trashed_at_ms");
        let payload_sha256 = digest_json(&legacy).unwrap();
        fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "version": STORAGE_VERSION,
                "payload_sha256": payload_sha256,
                "snapshot": legacy,
            }))
            .unwrap(),
        )
        .unwrap();

        let loaded = read_checkpoint_at(&path).unwrap();
        assert_eq!(loaded.generation, 7);
        let loaded_record = &loaded.index.entries[filename].record;
        assert_eq!(loaded_record.title, None);
        assert!(!loaded_record.favorite);
        assert_eq!(loaded_record.trashed_at_ms, None);

        let mut tampered: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        tampered["snapshot"]["generation"] = serde_json::json!(8);
        fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert!(
            read_checkpoint_at(&path)
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch"),
            "validating stored bytes must still reject a modified snapshot"
        );
    }

    #[test]
    fn pending_marker_with_durable_wal_rolls_forward() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default())).unwrap();
        let mut next = empty_snapshot(1);
        next.index.retired_names.insert("completed.png".into());
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: 0,
                pending: Some(PendingMutation {
                    generation: 1,
                    kind: "test".into(),
                    exact_names: vec!["completed.png".into()],
                    snapshot_sha256: digest_json(&next).unwrap(),
                }),
            },
        )
        .unwrap();
        atomic_write_json(&wal_path(dir.path()), &wrap_snapshot(next).unwrap()).unwrap();

        let recovered = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(recovered.generation, 1);
        assert!(recovered.index.is_retired("completed.png"));
        assert!(!wal_path(dir.path()).exists());
    }

    #[test]
    fn pending_marker_without_wal_rolls_back_without_inventing_authority() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default())).unwrap();
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: 0,
                pending: Some(PendingMutation {
                    generation: 1,
                    kind: "test".into(),
                    exact_names: vec!["never-committed.png".into()],
                    snapshot_sha256: "0".repeat(64),
                }),
            },
        )
        .unwrap();

        let recovered = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(recovered.generation, 0);
        assert!(!recovered.index.is_retired("never-committed.png"));
        assert!(read_marker(dir.path()).unwrap().unwrap().pending.is_none());
    }

    #[test]
    fn corrupt_current_checkpoint_falls_back_to_current_generation_backup() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        index.retired_names.insert("old.png".into());
        commit_snapshot(
            dir.path(),
            &guard,
            initial.generation,
            &mut index,
            "test",
            vec!["old.png".into()],
        )
        .unwrap();
        fs::write(checkpoint_path(dir.path()), b"{corrupt").unwrap();

        let recovered = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(recovered.generation, 1);
        assert!(recovered.index.is_retired("old.png"));
        let mut recovered_index = recovered.index;
        assert_eq!(
            commit_snapshot(
                dir.path(),
                &guard,
                recovered.generation,
                &mut recovered_index,
                "heal_corrupt_current",
                Vec::new(),
            )
            .unwrap(),
            2,
            "the current-generation backup must remain writable recovery authority"
        );
    }

    #[test]
    fn completed_mutation_keeps_current_backup_and_immediately_prior_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        index.quarantined_names.insert("changed.png".into());
        commit_snapshot(
            dir.path(),
            &guard,
            initial.generation,
            &mut index,
            "test",
            vec!["changed.png".into()],
        )
        .unwrap();

        assert_eq!(
            read_checkpoint_at(&checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            1
        );
        assert_eq!(
            read_checkpoint_at(&backup_checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            1
        );
        assert_eq!(
            read_checkpoint_at(&previous_checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            0
        );
    }
}
