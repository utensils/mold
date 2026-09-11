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
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const AUTHORITY_DIR: &str = "gallery-authority-v2";
const CHECKPOINT_FILE: &str = "checkpoint.json";
const PREVIOUS_CHECKPOINT_FILE: &str = "checkpoint.previous.json";
const BACKUP_CHECKPOINT_FILE: &str = "checkpoint.backup.json";
const MARKER_FILE: &str = "generation.json";
const WAL_FILE: &str = "mutation.wal";
const STORAGE_VERSION: u32 = 2;

/// The mutation kinds a single print's publication walks. These are the
/// per-request hot path, and the only ones that may skip rolling the
/// immediately-prior checkpoint; every other kind still rolls it.
const PUBLICATION_KINDS: &[&str] = &[
    "publish_batch",
    "retirement_projection_complete",
    "bind_retained_source_media",
    "release_retained_source_media",
];

/// Roll `previous` at least this often even on the hot path, so a long run of
/// publications still leaves a forensic trail.
const PREVIOUS_CHECKPOINT_GENERATION_INTERVAL: u64 = 32;
const PREVIOUS_CHECKPOINT_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

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

/// The checksummed envelope's bytes, and the payload digest inside them.
///
/// One serialization of the whole archive index answers four questions that
/// used to cost one apiece: the marker's `snapshot_sha256`, and the WAL, the
/// checkpoint and the backup file contents. `wrap_snapshot` + `atomic_write_json`
/// serialized the index twice per file (once inside `digest_json`, once to
/// the writer), so a commit paid nine serializations of a structure that grows
/// with the gallery.
///
/// The envelope is assembled by hand rather than through `ChecksummedSnapshot`
/// precisely so the digest describes the bytes that land — `validate_envelope`
/// hashes `RawValue::get()`, i.e. the stored payload text — instead of a
/// second serialization that merely ought to match.
/// `the_prebuilt_envelope_matches_the_serde_one` pins the two together.
fn serialize_envelope(snapshot: &AuthoritySnapshot) -> anyhow::Result<(Vec<u8>, String)> {
    let payload = serde_json::to_vec(snapshot)?;
    let digest = format!("{:x}", Sha256::digest(&payload));
    let prefix =
        format!(r#"{{"version":{STORAGE_VERSION},"payload_sha256":"{digest}","snapshot":"#);
    let mut bytes = Vec::with_capacity(prefix.len() + payload.len() + 2);
    bytes.extend_from_slice(prefix.as_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(b"}\n");
    Ok((bytes, digest))
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
    #[cfg(test)]
    CHECKPOINT_PARSE_COUNT.with(|count| count.set(count.get() + 1));
    validate_envelope(read_json(path)?, path)
}

// Full checkpoint parses (a serde pass plus a SHA-256 over the whole archive
// index) this thread has performed since the last reset. Thread-local for the
// reason `AUTHORITY_HASH_COUNT` is: the suite runs many tests in one process.
#[cfg(test)]
thread_local! {
    static CHECKPOINT_PARSE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_checkpoint_parse_count() {
    CHECKPOINT_PARSE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
fn checkpoint_parse_count() -> usize {
    CHECKPOINT_PARSE_COUNT.with(|count| count.get())
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
    // A bare `File::open` on a directory is ERROR_ACCESS_DENIED on Windows, so
    // this was not a silent no-op like the other four copies were — it failed
    // the whole startup recovery with a bare `Access is denied. (os error 5)`
    // and no context. `dir_sync` opens the handle the way Windows requires.
    crate::dir_sync::sync_directory(path)
        .with_context(|| format!("fsync gallery authority directory '{}'", path.display()))
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    atomic_write_bytes(path, &bytes)
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
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
        file.write_all(bytes)?;
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

/// Write the current checkpoint, optionally rolling the previous one.
///
/// `existing_generation` is the generation the caller already knows is on
/// disk. Reading it back is a full parse plus a SHA-256 over the whole archive
/// index, and a commit already knows the answer from the marker it verified;
/// `None` means "the caller does not know", which is the cold recovery and
/// initialization path where the read costs nothing that matters.
fn write_checkpoint_envelope(
    root: &Path,
    generation: u64,
    envelope: &[u8],
    existing_generation: Option<u64>,
    roll_previous: bool,
) -> anyhow::Result<()> {
    let dir = authority_dir(root);
    // The parent and the authority directory only need their own entries
    // fsynced when this call is what created them. Once the directory exists,
    // every write below already fsyncs it after its own rename.
    if !dir.is_dir() {
        fs::create_dir_all(&dir)?;
        sync_dir(
            dir.parent()
                .context("gallery authority directory has no parent")?,
        )?;
        sync_dir(&dir)?;
    }
    let current = checkpoint_path(root);
    if current.is_file() {
        let existing = match existing_generation {
            Some(generation) => generation,
            None => {
                read_checkpoint_at(&current)
                    .or_else(|_| read_checkpoint_at(&backup_checkpoint_path(root)))?
                    .generation
            }
        };
        ensure!(
            existing <= generation,
            "gallery authority checkpoint generation regressed from {existing} to {generation}"
        );
        if roll_previous && existing < generation {
            // The bytes currently at `current` ARE the previous checkpoint;
            // copying the file avoids re-serializing an index we no longer
            // hold in that shape.
            let bytes = fs::read(&current)?;
            atomic_write_bytes(&previous_checkpoint_path(root), &bytes)?;
        }
    }
    atomic_write_bytes(&current, envelope)
}

fn write_checkpoint(root: &Path, snapshot: &AuthoritySnapshot) -> anyhow::Result<()> {
    let (envelope, _) = serialize_envelope(snapshot)?;
    write_checkpoint_envelope(root, snapshot.generation, &envelope, None, true)
}

fn backup_checkpoint(root: &Path, snapshot: &AuthoritySnapshot) -> anyhow::Result<()> {
    let (envelope, _) = serialize_envelope(snapshot)?;
    atomic_write_bytes(&backup_checkpoint_path(root), &envelope)
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

/// Read an already-established authority snapshot without initializing,
/// repairing, validating, or otherwise mutating the gallery.
///
/// Durable queue hydration is a read path. In particular, probing an ordinary
/// empty gallery must not create `gallery-authority-v2`, and a crash-interrupted
/// authority mutation must be left for the explicit startup recovery pass.
pub(crate) fn load_existing_read_only(
    root: &Path,
    guard: &GalleryBookkeepingGuard,
) -> anyhow::Result<Option<LoadedAuthority>> {
    guard.ensure_root(root)?;
    let root = guard.canonical_root();
    let marker = read_marker(root)?;
    let checkpoint = read_checkpoint(root)?;
    match (marker, checkpoint) {
        (None, None) => Ok(None),
        (None, Some(_)) => anyhow::bail!(
            "gallery authority checkpoint has no stable generation marker; startup recovery is required"
        ),
        (Some(_), None) => anyhow::bail!(
            "gallery authority generation marker has no checkpoint; startup recovery is required"
        ),
        (Some(marker), Some(snapshot)) => {
            ensure!(
                marker.pending.is_none(),
                "gallery authority mutation is pending; startup recovery is required"
            );
            ensure!(
                !wal_path(root).try_exists()?,
                "gallery authority WAL is unresolved; startup recovery is required"
            );
            ensure!(
                marker.committed_generation == snapshot.generation,
                "gallery authority checkpoint generation does not match its stable marker"
            );
            remember_authority_tail(
                root,
                AuthorityTail {
                    generation: snapshot.generation,
                    legacy_evidence_epochs: snapshot.legacy_evidence_epochs,
                    previous_rolled_generation: snapshot.generation,
                    previous_rolled_at: std::time::Instant::now(),
                },
            );
            Ok(Some(LoadedAuthority {
                generation: snapshot.generation,
                index: snapshot.index,
                stats: ValidationStats::default(),
            }))
        }
    }
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
    remember_authority_tail(
        root,
        AuthorityTail {
            generation: snapshot.generation,
            legacy_evidence_epochs: snapshot.legacy_evidence_epochs.clone(),
            previous_rolled_generation: snapshot.generation,
            previous_rolled_at: std::time::Instant::now(),
        },
    );
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

/// What a commit needs from the checkpoint it is superseding: the generation,
/// and the legacy-evidence epochs it carries forward. Nothing else in a
/// multi-megabyte snapshot is read.
#[derive(Debug, Clone)]
struct AuthorityTail {
    generation: u64,
    legacy_evidence_epochs: std::collections::BTreeMap<String, u64>,
    previous_rolled_generation: u64,
    previous_rolled_at: std::time::Instant,
}

/// The last tail this PROCESS wrote, per canonical gallery root.
///
/// Trusting it is safe only under the bookkeeping flock, and only when the
/// on-disk marker still names the generation the cache does — see
/// [`cached_commit_tail`]. The marker is a few dozen bytes; the checkpoint it
/// stands in for grows with the gallery.
fn authority_tail_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, AuthorityTail>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, AuthorityTail>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn remember_authority_tail(root: &Path, tail: AuthorityTail) {
    authority_tail_cache()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(root.to_path_buf(), tail);
}

fn forget_authority_tail(root: &Path) {
    authority_tail_cache()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .remove(root);
}

/// The fast path into [`commit_snapshot`], and the whole of its validity
/// argument.
///
/// The caller holds the gallery bookkeeping flock, so no other process can be
/// mid-mutation. Three on-disk facts then say the cached tail still describes
/// the checkpoint: the marker exists with no pending mutation, its committed
/// generation is the one the caller expects AND the one this process last
/// wrote, and there is no unresolved WAL. Any of those failing hands the
/// commit back to the full `recover_storage` read, which is also what a cold
/// process, a foreign writer, or a crash-interrupted mutation gets.
fn cached_commit_tail(
    root: &Path,
    expected_generation: u64,
) -> anyhow::Result<Option<AuthorityTail>> {
    let cached = {
        let cache = authority_tail_cache()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        cache.get(root).cloned()
    };
    let Some(cached) = cached.filter(|tail| tail.generation == expected_generation) else {
        return Ok(None);
    };
    let Some(marker) = read_marker(root)? else {
        return Ok(None);
    };
    if marker.pending.is_some() || marker.committed_generation != expected_generation {
        return Ok(None);
    }
    if wal_path(root).try_exists()? {
        return Ok(None);
    }
    Ok(Some(cached))
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
    let current = match cached_commit_tail(root, expected_generation)? {
        Some(tail) => tail,
        None => {
            let snapshot =
                recover_storage(root)?.context("gallery authority checkpoint is missing")?;
            AuthorityTail {
                generation: snapshot.generation,
                legacy_evidence_epochs: snapshot.legacy_evidence_epochs,
                previous_rolled_generation: snapshot.generation,
                previous_rolled_at: std::time::Instant::now(),
            }
        }
    };
    ensure!(
        current.generation == expected_generation,
        "gallery authority generation changed from {expected_generation} to {}",
        current.generation
    );
    let generation = expected_generation
        .checked_add(1)
        .context("gallery authority generation overflow")?;
    let mut legacy_evidence_epochs = current.legacy_evidence_epochs.clone();
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
        legacy_evidence_epochs: legacy_evidence_epochs.clone(),
    };
    // One serialization, one digest, three files written from the same bytes.
    let (envelope, snapshot_sha256) = serialize_envelope(&snapshot)?;
    // The immediately-prior checkpoint is forensic, not recovery authority:
    // `recover_storage` refuses any checkpoint whose generation disagrees with
    // the marker, so `previous` can never be what a recovery lands on. Rolling
    // it costs a second full-index write on a per-print path, so the hot kinds
    // roll it only periodically. Every other kind still rolls it, and the
    // same-generation BACKUP below is written on every commit — that one IS
    // recovery authority.
    let roll_previous = !PUBLICATION_KINDS.contains(&kind)
        || generation.saturating_sub(current.previous_rolled_generation)
            >= PREVIOUS_CHECKPOINT_GENERATION_INTERVAL
        || current.previous_rolled_at.elapsed() >= PREVIOUS_CHECKPOINT_MIN_INTERVAL;
    // A failed commit must not leave this process believing a tail it did not
    // write; the next attempt then takes the full recovery read.
    forget_authority_tail(root);
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
    atomic_write_bytes(&wal_path(root), &envelope)?;
    guard.ensure_root(root)?;
    write_checkpoint_envelope(
        root,
        generation,
        &envelope,
        Some(expected_generation),
        roll_previous,
    )?;
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
    atomic_write_bytes(&backup_checkpoint_path(root), &envelope)?;
    crate::batch_transaction::remove_legacy_gallery_evidence(root, &collect_legacy)?;
    remember_authority_tail(
        root,
        AuthorityTail {
            generation,
            legacy_evidence_epochs,
            previous_rolled_generation: if roll_previous {
                generation
            } else {
                current.previous_rolled_generation
            },
            previous_rolled_at: if roll_previous {
                std::time::Instant::now()
            } else {
                current.previous_rolled_at
            },
        },
    );
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

    /// Stand in for another process's checkpoint: writes the bytes with no
    /// generation-regression check, so a test can move the authority in
    /// either direction.
    fn write_checkpoint_for_test(root: &Path, snapshot: &AuthoritySnapshot) {
        let (envelope, _) = serialize_envelope(snapshot).unwrap();
        fs::create_dir_all(authority_dir(root)).unwrap();
        atomic_write_bytes(&checkpoint_path(root), &envelope).unwrap();
        forget_authority_tail(&fs::canonicalize(root).unwrap());
    }

    #[test]
    fn the_prebuilt_envelope_matches_the_serde_one() {
        // `serialize_envelope` assembles the checksummed envelope by hand so
        // the digest covers the payload bytes that actually land. If it ever
        // drifts from `ChecksummedSnapshot`'s serde shape, every checkpoint
        // this process writes becomes unreadable — so pin the two together.
        let mut snapshot = empty_snapshot(11);
        snapshot
            .legacy_evidence_epochs
            .insert("evidence/one.json".into(), 3);
        snapshot.index.quarantined_names.insert("q.png".into());
        let (bytes, digest) = serialize_envelope(&snapshot).unwrap();
        let mut expected = serde_json::to_vec(&wrap_snapshot(snapshot.clone()).unwrap()).unwrap();
        expected.push(b'\n');
        assert_eq!(bytes, expected);
        assert_eq!(digest, digest_json(&snapshot).unwrap());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("envelope.json");
        atomic_write_bytes(&path, &bytes).unwrap();
        assert_eq!(read_checkpoint_at(&path).unwrap().generation, 11);
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
                retained_media: Vec::new(),
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
    fn commit_does_not_reread_the_checkpoint() {
        // Every commit used to re-read and re-verify the whole checkpoint —
        // a parse plus a SHA-256 over the entire archive index — and then
        // re-read it AGAIN inside `write_checkpoint` for the generation
        // regression check. Under the bookkeeping flock, at the marker's own
        // generation, the in-memory index IS the checkpoint.
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..3_u64 {
            index
                .quarantined_names
                .insert(format!("publication-{step}.png"));
            reset_checkpoint_parse_count();
            generation = commit_snapshot(
                dir.path(),
                &guard,
                generation,
                &mut index,
                "publish_batch",
                vec![format!("publication-{step}.png")],
            )
            .unwrap();
            assert_eq!(
                checkpoint_parse_count(),
                0,
                "a flock-guarded commit at the marker's generation parses no checkpoint"
            );
        }
        assert_eq!(generation, initial.generation + 3);
        // The bytes on disk are still the authority a cold process reads.
        let reloaded = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(reloaded.generation, generation);
        for step in 0..3_u64 {
            assert!(reloaded
                .index
                .quarantined_names
                .contains(&format!("publication-{step}.png")));
        }
    }

    #[test]
    fn stale_marker_still_forces_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        let generation = commit_snapshot(
            dir.path(),
            &guard,
            initial.generation,
            &mut index,
            "publish_batch",
            Vec::new(),
        )
        .unwrap();

        // Somebody else advanced the authority. The cached tail is now stale,
        // and the commit must notice through the marker rather than writing
        // over a generation it never read.
        let mut ahead = empty_snapshot(generation + 5);
        ahead.index = index.clone();
        write_checkpoint_for_test(dir.path(), &ahead);
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: generation + 5,
                pending: None,
            },
        )
        .unwrap();
        let error = commit_snapshot(
            dir.path(),
            &guard,
            generation,
            &mut index,
            "publish_batch",
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("generation changed"),
            "unexpected error: {error:#}"
        );

        // An unresolved WAL at the cached generation must also drop the fast
        // path: rolling it forward is recovery's job, not a commit's.
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: generation,
                pending: None,
            },
        )
        .unwrap();
        write_checkpoint_for_test(dir.path(), &{
            let mut snapshot = empty_snapshot(generation);
            snapshot.index = index.clone();
            snapshot
        });
        let mut rolled = empty_snapshot(generation + 1);
        rolled.index = index.clone();
        rolled.index.quarantined_names.insert("from-wal.png".into());
        atomic_write_json(&wal_path(dir.path()), &wrap_snapshot(rolled).unwrap()).unwrap();
        reset_checkpoint_parse_count();
        let mut committing = index.clone();
        let error = commit_snapshot(
            dir.path(),
            &guard,
            generation,
            &mut committing,
            "publish_batch",
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("generation changed"),
            "an unresolved WAL must be recovered, not ignored: {error:#}"
        );
        assert!(
            checkpoint_parse_count() > 0,
            "an unresolved WAL forces the full recovery read"
        );
    }

    #[test]
    fn the_previous_checkpoint_is_periodic_and_the_backup_is_not() {
        // The immediately-prior checkpoint is forensic: `recover_storage`
        // refuses any checkpoint whose generation disagrees with the marker,
        // so `previous` can never be the snapshot recovery lands on. Writing
        // the whole index a third time on every publication for it is not
        // worth a per-print fsync pair.
        //
        // The BACKUP is a different thing and stays per-commit: it is the
        // same-generation copy `read_checkpoint` falls back to when `current`
        // is unreadable, and `corrupt_current_checkpoint_falls_back_to_current_generation_backup`
        // is an existing, deliberate durability guarantee.
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..4_u64 {
            index.quarantined_names.insert(format!("hot-{step}.png"));
            generation = commit_snapshot(
                dir.path(),
                &guard,
                generation,
                &mut index,
                "publish_batch",
                Vec::new(),
            )
            .unwrap();
            assert_eq!(
                read_checkpoint_at(&backup_checkpoint_path(dir.path()))
                    .unwrap()
                    .generation,
                generation,
                "the backup tracks every publication"
            );
        }
        assert_eq!(
            read_checkpoint_at(&previous_checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            0,
            "the previous checkpoint did not follow the hot path"
        );

        // A kind that is not on the per-request hot path still rolls it.
        generation = commit_snapshot(
            dir.path(),
            &guard,
            generation,
            &mut index,
            "startup_validation",
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            read_checkpoint_at(&previous_checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            generation - 1,
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
