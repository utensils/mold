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

/// Where a version-2 store lives. Unchanged, and an older mold reads only
/// this name.
const AUTHORITY_DIR: &str = "gallery-authority-v2";
/// Where a version-3 store lives.
///
/// A separate directory, not a rewrite in place. Version 3 bytes written into
/// the `-v2` directory is exactly what took production down: every older mold
/// sharing the `$MOLD_HOME` refused publication, and there was no v2 copy left
/// to roll back to because the backup had been rewritten at v3 too. With two
/// directories the v2 store is frozen intact at the moment of the upgrade, so
/// a rollback needs no repair at all.
///
/// The cost is stated rather than hidden: while BOTH a new mold (writing v3)
/// and an old one (writing v2) publish to one home, the two indexes diverge.
/// That is why writing v3 is opt-in and the documentation says to enable it
/// only when every binary sharing the home is new enough.
const AUTHORITY_DIR_V3: &str = "gallery-authority-v3";
const CHECKPOINT_FILE: &str = "checkpoint.json";
const PREVIOUS_CHECKPOINT_FILE: &str = "checkpoint.previous.json";
const BACKUP_CHECKPOINT_FILE: &str = "checkpoint.backup.json";
const MARKER_FILE: &str = "generation.json";
const WAL_FILE: &str = "mutation.wal";
const MUTATION_LOG_FILE: &str = "mutation.log";
/// The version this process WRITES once a root's startup recovery has
/// succeeded. v3 replaces the per-commit full-snapshot WAL with an append-only
/// delta log; the checkpoint is rewritten only at compaction.
const STORAGE_VERSION: u32 = 3;
/// The version this process still READS, and still writes until recovery has
/// run for a root. A v2 store is upgraded in place on its first v3 recovery.
const LEGACY_STORAGE_VERSION: u32 = 2;
/// Compaction thresholds. Either bound alone is enough: a long run of small
/// mutations and a short run of large ones both want the checkpoint refreshed
/// before a recovery has to replay too much.
const MUTATION_LOG_MAX_RECORDS: usize = 256;
const MUTATION_LOG_MAX_BYTES: u64 = 8 * 1024 * 1024;

fn supported_storage_version(version: u32) -> bool {
    version == STORAGE_VERSION || version == LEGACY_STORAGE_VERSION
}

/// What to tell an operator whose store this build cannot read.
///
/// A version ABOVE what we understand is the shared-home case: some newer
/// mold upgraded the format under a binary that publishes to the same
/// `$MOLD_HOME`. That is recoverable, and the sentence says how. Anything
/// else is a corrupt or foreign file.
fn unsupported_version_message(kind: &str, version: u32, path: &Path) -> String {
    if version > STORAGE_VERSION {
        format!(
            "gallery authority {kind} version {version} is newer than this build supports \
             ({STORAGE_VERSION}) in {}; this gallery was written by a newer mold. Run \
             `mold system gallery-authority downgrade` with that newer build, or upgrade this \
             one.",
            path.display()
        )
    } else {
        format!(
            "unsupported gallery authority {kind} version {version} in {}",
            path.display()
        )
    }
}

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

/// The serde shape of the checksummed envelope.
///
/// Production assembles those bytes by hand in `serialize_envelope`, so this
/// survives as the independent definition that
/// `the_prebuilt_envelope_matches_the_serde_one` pins the hand-built one
/// against — a drift here fails that test instead of silently writing a
/// checkpoint no reader accepts.
#[cfg(test)]
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

/// The directory this root's authority ACTUALLY occupies.
///
/// A v3 store wins once it has a marker; until then everything reads and
/// writes the v2 directory. Resolved rather than configured, so a store that
/// an earlier build upgraded IN PLACE — v3 bytes under the `-v2` name, the
/// production incident's shape — is still found, read, and downgradable.
fn authority_dir(root: &Path) -> PathBuf {
    let v3 = authority_dir_v3(root);
    if v3.join(MARKER_FILE).is_file() {
        return v3;
    }
    legacy_authority_dir(root)
}

fn legacy_authority_dir(root: &Path) -> PathBuf {
    root.join(crate::batch_transaction::TRANSACTION_DIR)
        .join(AUTHORITY_DIR)
}

fn authority_dir_v3(root: &Path) -> PathBuf {
    root.join(crate::batch_transaction::TRANSACTION_DIR)
        .join(AUTHORITY_DIR_V3)
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

#[cfg(test)]
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
    serialize_envelope_at(snapshot, snapshot.version)
}

fn serialize_envelope_at(
    snapshot: &AuthoritySnapshot,
    storage_version: u32,
) -> anyhow::Result<(Vec<u8>, String)> {
    let payload = serde_json::to_vec(snapshot)?;
    let digest = format!("{:x}", Sha256::digest(&payload));
    let prefix =
        format!(r#"{{"version":{storage_version},"payload_sha256":"{digest}","snapshot":"#);
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
        supported_storage_version(envelope.version),
        "{}",
        unsupported_version_message("checkpoint", envelope.version, path)
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
        supported_storage_version(snapshot.version),
        "{}",
        unsupported_version_message("checkpoint", snapshot.version, path)
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
                supported_storage_version(marker.version),
                "{}",
                unsupported_version_message("generation marker", marker.version, &path)
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

fn mutation_log_path(root: &Path) -> PathBuf {
    authority_dir(root).join(MUTATION_LOG_FILE)
}

/// Everything one commit changed, relative to the snapshot before it.
///
/// v2 wrote the WHOLE archive index three times per commit (WAL, checkpoint,
/// backup). On a gallery with tens of thousands of prints that is tens of
/// megabytes of serialization and I/O to record one new filename — the cost
/// of publishing a print grew with the size of the library. A delta is a few
/// hundred bytes whatever the library holds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct IndexDelta {
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    entries_set: std::collections::BTreeMap<String, CommittedArchiveEntry>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    entries_removed: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    retired_entries_set: std::collections::BTreeMap<String, CommittedArchiveEntry>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    retired_entries_removed: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    retired_names_added: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    retired_names_removed: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    quarantined_added: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    quarantined_removed: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    retirement_epochs_set: std::collections::BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    retirement_epochs_removed: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    retirement_projection_epochs_set: std::collections::BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    retirement_projection_epochs_removed: std::collections::BTreeSet<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    legacy_evidence_epochs_set: std::collections::BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    legacy_evidence_epochs_removed: std::collections::BTreeSet<String>,
}

fn map_delta<V: Clone + PartialEq>(
    before: &std::collections::BTreeMap<String, V>,
    after: &std::collections::BTreeMap<String, V>,
) -> (
    std::collections::BTreeMap<String, V>,
    std::collections::BTreeSet<String>,
) {
    let set = after
        .iter()
        .filter(|(key, value)| before.get(*key) != Some(*value))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let removed = before
        .keys()
        .filter(|key| !after.contains_key(*key))
        .cloned()
        .collect();
    (set, removed)
}

fn set_delta(
    before: &std::collections::BTreeSet<String>,
    after: &std::collections::BTreeSet<String>,
) -> (
    std::collections::BTreeSet<String>,
    std::collections::BTreeSet<String>,
) {
    (
        after.difference(before).cloned().collect(),
        before.difference(after).cloned().collect(),
    )
}

/// Entries added, removed, or NAMED by the mutation.
///
/// `CommittedArchiveEntry` holds a whole `GenerationRecord`, so comparing
/// values means serializing every entry — exactly the O(gallery) cost per
/// print the delta log exists to remove. The contract instead is the one
/// `exact_names` already carries: **a mutation that edits an entry in place
/// must name it.** Every v3 caller does — `bind_retained_source_media` passes
/// its changed names, `release_retained_source_media` and the user
/// trash/restore/delete kinds pass the filename they touched — and the two
/// kinds that mutate entries without naming them (`startup_validation`
/// refreshing file facts, `legacy_evidence_gc_epoch`) run inside
/// `load_or_initialize`, before this root is allowed to write v3 at all.
/// Compaction rewrites the full checkpoint from memory on the same schedule,
/// so any drift a future caller introduced heals there;
/// `an_in_place_edit_must_name_the_entry_it_changed` states the rule.
fn entry_map_delta(
    before: &std::collections::BTreeSet<String>,
    after: &std::collections::BTreeMap<String, CommittedArchiveEntry>,
    touched: &std::collections::BTreeSet<&str>,
) -> (
    std::collections::BTreeMap<String, CommittedArchiveEntry>,
    std::collections::BTreeSet<String>,
) {
    let set = after
        .iter()
        .filter(|(name, _)| !before.contains(name.as_str()) || touched.contains(name.as_str()))
        .map(|(name, entry)| (name.clone(), entry.clone()))
        .collect();
    let removed = before
        .iter()
        .filter(|name| !after.contains_key(name.as_str()))
        .cloned()
        .collect();
    (set, removed)
}

/// Everything the next delta needs from the committed index, and nothing
/// else.
///
/// Holding the whole previous `CommittedArchiveIndex` in the tail meant a deep
/// clone of every `GenerationRecord` on every commit — measured 18 ms at
/// 10,000 prints, which is the O(gallery) cost coming back in through the
/// side door. The diff only ever asks the previous state for KEY membership
/// and for the two small epoch maps, so that is all it keeps.
#[derive(Debug, Clone, Default)]
struct AuthorityIndexKeys {
    entries: std::collections::BTreeSet<String>,
    retired_entries: std::collections::BTreeSet<String>,
    retired_names: std::collections::BTreeSet<String>,
    quarantined_names: std::collections::BTreeSet<String>,
    retirement_epochs: std::collections::BTreeMap<String, u64>,
    retirement_projection_epochs: std::collections::BTreeMap<String, u64>,
}

impl AuthorityIndexKeys {
    fn of(index: &CommittedArchiveIndex) -> Self {
        Self {
            entries: index.entries.keys().cloned().collect(),
            retired_entries: index.retired_entries.keys().cloned().collect(),
            retired_names: index.retired_names.clone(),
            quarantined_names: index.quarantined_names.clone(),
            retirement_epochs: index.retirement_epochs.clone(),
            retirement_projection_epochs: index.retirement_projection_epochs.clone(),
        }
    }
}

impl IndexDelta {
    fn between(
        before: (
            &AuthorityIndexKeys,
            &std::collections::BTreeMap<String, u64>,
        ),
        after: (
            &CommittedArchiveIndex,
            &std::collections::BTreeMap<String, u64>,
        ),
        touched: &std::collections::BTreeSet<&str>,
    ) -> Self {
        let (before, before_legacy) = before;
        let (after, after_legacy) = after;
        let (entries_set, entries_removed) =
            entry_map_delta(&before.entries, &after.entries, touched);
        let (retired_entries_set, retired_entries_removed) =
            entry_map_delta(&before.retired_entries, &after.retired_entries, touched);
        let (retired_names_added, retired_names_removed) =
            set_delta(&before.retired_names, &after.retired_names);
        let (quarantined_added, quarantined_removed) =
            set_delta(&before.quarantined_names, &after.quarantined_names);
        let (retirement_epochs_set, retirement_epochs_removed) =
            map_delta(&before.retirement_epochs, &after.retirement_epochs);
        let (retirement_projection_epochs_set, retirement_projection_epochs_removed) = map_delta(
            &before.retirement_projection_epochs,
            &after.retirement_projection_epochs,
        );
        let (legacy_evidence_epochs_set, legacy_evidence_epochs_removed) =
            map_delta(before_legacy, after_legacy);
        Self {
            entries_set,
            entries_removed,
            retired_entries_set,
            retired_entries_removed,
            retired_names_added,
            retired_names_removed,
            quarantined_added,
            quarantined_removed,
            retirement_epochs_set,
            retirement_epochs_removed,
            retirement_projection_epochs_set,
            retirement_projection_epochs_removed,
            legacy_evidence_epochs_set,
            legacy_evidence_epochs_removed,
        }
    }

    fn apply(self, snapshot: &mut AuthoritySnapshot) {
        snapshot.index.entries.extend(self.entries_set);
        for name in &self.entries_removed {
            snapshot.index.entries.remove(name);
        }
        snapshot
            .index
            .retired_entries
            .extend(self.retired_entries_set);
        for name in &self.retired_entries_removed {
            snapshot.index.retired_entries.remove(name);
        }
        snapshot
            .index
            .retired_names
            .extend(self.retired_names_added);
        for name in &self.retired_names_removed {
            snapshot.index.retired_names.remove(name);
        }
        snapshot
            .index
            .quarantined_names
            .extend(self.quarantined_added);
        for name in &self.quarantined_removed {
            snapshot.index.quarantined_names.remove(name);
        }
        snapshot
            .index
            .retirement_epochs
            .extend(self.retirement_epochs_set);
        for name in &self.retirement_epochs_removed {
            snapshot.index.retirement_epochs.remove(name);
        }
        snapshot
            .index
            .retirement_projection_epochs
            .extend(self.retirement_projection_epochs_set);
        for name in &self.retirement_projection_epochs_removed {
            snapshot.index.retirement_projection_epochs.remove(name);
        }
        snapshot
            .legacy_evidence_epochs
            .extend(self.legacy_evidence_epochs_set);
        for path in &self.legacy_evidence_epochs_removed {
            snapshot.legacy_evidence_epochs.remove(path);
        }
    }
}

/// One framed, checksummed mutation as it is read back.
#[derive(Debug, Deserialize)]
struct RawMutationLogRecord {
    version: u32,
    generation: u64,
    delta_sha256: String,
    delta: Box<serde_json::value::RawValue>,
}

/// Serialize one log record. The delta bytes are digested and then written
/// verbatim, exactly as the checkpoint envelope is, so the checksum describes
/// the bytes that land rather than a second serialization that ought to match.
fn serialize_mutation_log_record(
    generation: u64,
    kind: &str,
    exact_names: &[String],
    delta: &IndexDelta,
) -> anyhow::Result<Vec<u8>> {
    let payload = serde_json::to_vec(delta)?;
    let digest = format!("{:x}", Sha256::digest(&payload));
    let prefix = format!(
        r#"{{"version":{STORAGE_VERSION},"generation":{generation},"kind":{},"exact_names":{},"delta_sha256":"{digest}","delta":"#,
        serde_json::to_string(kind)?,
        serde_json::to_string(exact_names)?,
    );
    let mut bytes = Vec::with_capacity(prefix.len() + payload.len() + 2);
    bytes.extend_from_slice(prefix.as_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(b"}\n");
    Ok(bytes)
}

/// Append one record and make it durable. Three fsyncs per commit in total:
/// this one, plus the marker's file and directory.
fn append_mutation_log_record(root: &Path, record: &[u8]) -> anyhow::Result<u64> {
    let path = mutation_log_path(root);
    let existed = path.is_file();
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    file.write_all(record)?;
    file.sync_all()?;
    let len = file.metadata()?.len();
    drop(file);
    if !existed {
        // The log's own directory entry must survive a power cut the first
        // time; afterwards the entry is already there.
        sync_dir(&authority_dir(root))?;
    }
    Ok(len)
}

/// Every intact, contiguous record in the log, and the byte offset the first
/// bad one starts at.
///
/// "Bad" covers a torn append (the process died mid-write), a record whose
/// delta does not match its digest, an unsupported version, and a generation
/// that does not follow the one before it. Everything from there on is
/// discarded: a log is a sequence, so a gap makes the tail meaningless rather
/// than merely unverified.
/// The intact prefix of a mutation log: its records, the byte offset the
/// first bad one starts at, and how many records were discarded.
struct MutationLogScan {
    records: Vec<(u64, IndexDelta)>,
    good_bytes: u64,
    discarded: usize,
}

fn read_mutation_log(root: &Path, from_generation: u64) -> anyhow::Result<MutationLogScan> {
    let path = mutation_log_path(root);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(MutationLogScan {
                records: Vec::new(),
                good_bytes: 0,
                discarded: 0,
            })
        }
        Err(error) => return Err(error.into()),
    };
    let mut records = Vec::new();
    let mut good_bytes = 0_u64;
    let mut expected = from_generation;
    let mut total = 0_usize;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        total += 1;
        if !line.ends_with(b"\n") {
            break;
        }
        let Ok(raw) = serde_json::from_slice::<RawMutationLogRecord>(line) else {
            break;
        };
        if !supported_storage_version(raw.version) {
            break;
        }
        if format!("{:x}", Sha256::digest(raw.delta.get().as_bytes())) != raw.delta_sha256 {
            break;
        }
        if raw.generation != expected.saturating_add(1) {
            break;
        }
        let Ok(delta) = serde_json::from_str::<IndexDelta>(raw.delta.get()) else {
            break;
        };
        expected = raw.generation;
        good_bytes += line.len() as u64;
        records.push((raw.generation, delta));
    }
    let intact = records.len();
    Ok(MutationLogScan {
        discarded: total.saturating_sub(intact),
        records,
        good_bytes,
    })
}

fn mutation_log_bytes(root: &Path) -> u64 {
    fs::metadata(mutation_log_path(root))
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

/// Drop a torn or non-contiguous tail so the next append starts from a
/// well-formed record.
fn truncate_mutation_log(root: &Path, good_bytes: u64) -> anyhow::Result<()> {
    let path = mutation_log_path(root);
    if good_bytes == 0 {
        match fs::remove_file(&path) {
            Ok(()) => sync_dir(&authority_dir(root))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        return Ok(());
    }
    let file = OpenOptions::new().write(true).open(&path)?;
    file.set_len(good_bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Roots this process may write version 3 to.
///
/// TWO conditions, and both are load-bearing.
///
/// The operator must have asked for it (`gallery.authority_log` /
/// `MOLD_GALLERY_AUTHORITY_LOG`), because the format is shared state: one new
/// binary starting against a `$MOLD_HOME` an older one also publishes to
/// upgraded the store and locked that older binary out of publication
/// entirely with "unsupported gallery authority generation marker". Reading
/// v3 is unconditional; only writing it is a decision.
///
/// And this process must have recovered the root itself — a commit that has
/// not resolved the log's tail could append after a torn record and bury the
/// tear under valid-looking bytes.
fn v3_enabled_roots() -> &'static std::sync::Mutex<std::collections::HashSet<PathBuf>> {
    static ROOTS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<PathBuf>>> =
        std::sync::OnceLock::new();
    ROOTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

fn enable_v3_writing(root: &Path) {
    v3_enabled_roots()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(root.to_path_buf());
}

/// Whether the operator has opted this PROCESS in to writing version 3.
///
/// Resolved ONCE per process, and — this is the part that was wrong —
/// resolved wherever the first authority open happens, not only in the server
/// entry point. `run_server` used to be the sole writer of this flag, so a
/// forced-local `mold run`, the TUI, and every other CLI publication path
/// arrived with it false and committed VERSION 2 into a home the operator had
/// explicitly switched to version 3. On a host that also uses the CLI that is
/// not an edge case, it is every day: the setting was silently reverted by the
/// next local render, and the two stores then diverged in the direction the
/// operator least expected.
///
/// A `OnceLock` rather than an atomic because "resolved once" is the actual
/// contract: a request must never see a different answer from the startup
/// recovery that prepared the store. `set_authority_log_requested` still wins
/// when the server installs it first, which it does, before any gallery opens.
static AUTHORITY_LOG_REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Install the resolved `gallery.authority_log` decision. Called from server
/// startup, before any gallery is opened. A no-op if the answer was already
/// resolved, which keeps "one answer per process" true either way.
pub(crate) fn set_authority_log_requested(requested: bool) {
    let _ = AUTHORITY_LOG_REQUESTED.set(requested);
}

/// The effective `gallery.authority_log` for a loaded config.
///
/// One function so the server entry point and the lazy path below cannot
/// disagree about what the switch means.
pub(crate) fn authority_log_from_config(config: &mold_core::Config) -> bool {
    config.gallery.effective_authority_log()
}

fn authority_log_requested() -> bool {
    *AUTHORITY_LOG_REQUESTED.get_or_init(|| {
        // Tests must not read the developer's own `~/.mold/config.toml`: a
        // machine that had opted in would silently change what every
        // authority test exercises. An explicit `set_` still works.
        if cfg!(test) {
            return false;
        }
        authority_log_from_config(&mold_core::Config::load_or_default())
    })
}

fn v3_writing_enabled(root: &Path) -> bool {
    v3_enabled_roots()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .contains(root)
}

fn disable_v3_writing(root: &Path) {
    v3_enabled_roots()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .remove(root);
}

#[cfg(test)]
fn disable_v3_writing_for_test(root: &Path) {
    disable_v3_writing(root);
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
            // The bytes currently at `current` ARE the previous checkpoint, so
            // copying the file avoids re-serializing an index we no longer
            // hold in that shape — but only after they are read back as a
            // VALID checkpoint. `existing` can be a caller's fallback zero
            // when `current` failed to parse, and a raw copy then wrote
            // corruption into the last forensic copy, leaving two of three
            // checkpoints holding the same damage.
            match read_checkpoint_at(&current) {
                Ok(_) => {
                    let bytes = fs::read(&current)?;
                    atomic_write_bytes(&previous_checkpoint_path(root), &bytes)?;
                }
                Err(error) => tracing::warn!(
                    %error,
                    "not rolling an unreadable gallery authority checkpoint into `previous`"
                ),
            }
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

fn recover_storage(root: &Path, authority_log: bool) -> anyhow::Result<Option<AuthoritySnapshot>> {
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
    // Under v3 the checkpoint deliberately trails the marker — the delta log
    // holds the difference — so this agreement is checked AFTER the replay
    // below, not here.
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
    // v3: fold the delta log into the recovered snapshot.
    //
    // The log is the write-ahead record: a commit appends its delta, fsyncs
    // it, and only then advances the marker. So a crash between those two
    // leaves a durable record the marker does not mention, and the log — not
    // the marker — is what says where the authority got to. Records are taken
    // only while they are contiguous and their digest matches; the first that
    // is not ends the replay and everything from there is truncated, because
    // a log is a sequence and a gap makes its tail meaningless.
    let mut replayed = 0_usize;
    let mut discarded = 0_usize;
    if let Some(snapshot) = current.as_mut() {
        let MutationLogScan {
            records,
            good_bytes,
            discarded: dropped,
        } = read_mutation_log(root, snapshot.generation)?;
        replayed = records.len();
        discarded = dropped;
        for (generation, delta) in records {
            delta.apply(snapshot);
            snapshot.generation = generation;
        }
        // The marker is advisory under v3, but it must not name a generation
        // the store cannot reach: that means bytes are missing rather than
        // merely unacknowledged. This is asked BEFORE anything is truncated —
        // a torn tail is the end of an interrupted append and is safe to drop,
        // but corruption in the MIDDLE of a log leaves intact records after it
        // that are still on disk and still repairable by hand. Deleting them
        // first turned a recoverable store into a permanently failing one.
        if marker
            .as_ref()
            .is_some_and(|marker| marker.pending.is_none())
            && snapshot.generation < committed_generation
        {
            anyhow::bail!(
                "gallery authority mutation log stops at generation {} but its marker names {} \
                 — {} record(s) after the break are intact and still on disk. This is mid-log \
                 corruption, not a torn tail; the log has NOT been truncated.",
                snapshot.generation,
                committed_generation,
                dropped
            );
        }
        if dropped > 0 {
            // With no marker there is nothing to check the replay against, so
            // "the tail is torn" is an assumption rather than a finding.
            // Truncating on it silently resurrects deleted prints and loses
            // published ones.
            ensure!(
                marker.is_some(),
                "gallery authority mutation log has {dropped} unresolvable record(s) and no \
                 generation marker to check the replay against; refusing to truncate."
            );
            tracing::warn!(
                records = dropped,
                "discarding an unresolved gallery authority mutation log tail"
            );
            truncate_mutation_log(root, good_bytes)?;
        }
    }

    // Compaction at startup: the recovered snapshot becomes the checkpoint and
    // the log starts empty, so the next process replays nothing. It is also
    // the v2 -> v3 upgrade — the checkpoint is rewritten at the current
    // storage version, read once and never again.
    if let Some(snapshot) = current.as_ref() {
        // Compaction rewrites the checkpoint AT THIS BUILD'S write version,
        // so it is also the v2 -> v3 upgrade. That makes it a decision, not
        // housekeeping: a default build must not upgrade a store just by
        // opening it. It compacts only when the log actually needs folding
        // in, or when the operator has asked for v3.
        let storage_version = if authority_log {
            STORAGE_VERSION
        } else {
            LEGACY_STORAGE_VERSION
        };
        let existing_version = read_checkpoint_at(&checkpoint_path(root))
            .map(|existing| existing.version)
            .ok();
        let needs_compaction = replayed > 0
            || discarded > 0
            || existing_version.is_none()
            || (authority_log && existing_version != Some(STORAGE_VERSION));
        if needs_compaction {
            let checkpoint_generation = read_checkpoint_at(&checkpoint_path(root))
                .map(|existing| existing.generation)
                .unwrap_or(0);
            if authority_log && existing_version == Some(LEGACY_STORAGE_VERSION) {
                // The UPGRADE, and it is the whole reason the switch exists.
                // It writes a NEW store in the v3 directory and does not touch
                // the v2 one, so a rollback finds its store exactly as it left
                // it. A default build never reaches this line.
                upgrade_store_to_v3(root, snapshot)?;
            } else {
                compact_mutation_log_at(root, snapshot, checkpoint_generation, storage_version)?;
            }
        }
        write_marker(
            root,
            &MutationMarker {
                version: storage_version,
                committed_generation: snapshot.generation,
                pending: None,
            },
        )?;
    }
    Ok(current)
}

/// What a downgrade did, for the operator and for a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DowngradeOutcome {
    /// The generation the store now reports, after replaying the log.
    pub generation: u64,
    /// How many delta records were folded into the checkpoint.
    pub replayed_records: usize,
    /// Whether the store was already version 2 (the command is idempotent).
    pub already_legacy: bool,
}

/// Rewrite a version-3 gallery archive authority as version 2.
///
/// This exists because the storage format is SHARED state. A `$MOLD_HOME` can
/// be published to by more than one binary — a production service and a
/// scratch server, or a release and the rollback you are about to perform —
/// and a mold older than 0.29 reads version 2 only. It refuses publication
/// outright against a v3 store, so upgrading one is a decision that has to be
/// reversible.
///
/// The sequence is the one recovery already trusts: take the bookkeeping
/// flock so no writer can be mid-commit, replay the delta log onto the
/// checkpoint, then write the checkpoint, its backup and the marker at
/// version 2 before removing the log. The verification at the end is not
/// ceremony — it reads the result back through the ordinary loader and
/// refuses to report success on a store it could not load.
///
/// It refuses rather than guesses in the two cases where the store is not
/// quiescent: a pending v2 mutation, and a torn or non-contiguous log tail.
/// Both are resolved by letting a mold that understands v3 recover the store
/// once, which is what `mold serve` does at startup.
pub fn downgrade_to_legacy_storage(root: &Path) -> anyhow::Result<DowngradeOutcome> {
    let guard = crate::batch_transaction::acquire_gallery_bookkeeping_lock(root)?;
    let root = guard.canonical_root();

    let marker = read_marker(root)?;
    let Some(snapshot) = read_checkpoint(root)? else {
        anyhow::bail!(
            "no gallery archive authority in {} — there is nothing to downgrade",
            authority_dir(root).display()
        );
    };
    if let Some(marker) = marker.as_ref() {
        ensure!(
            marker.pending.is_none(),
            "a gallery authority mutation is still pending in {}. Start `mold serve` once with a \
             build that understands this store so recovery can resolve it, stop it, then \
             downgrade.",
            authority_dir(root).display()
        );
    }
    ensure!(
        !wal_path(root).try_exists()?,
        "an unresolved gallery authority WAL is present in {}. Start `mold serve` once with a \
         build that understands this store so recovery can resolve it, stop it, then downgrade.",
        authority_dir(root).display()
    );

    let mut snapshot = snapshot;
    let MutationLogScan {
        records, discarded, ..
    } = read_mutation_log(root, snapshot.generation)?;
    ensure!(
        discarded == 0,
        "the gallery authority mutation log in {} has a torn or non-contiguous tail ({discarded} \
         record(s)). Start `mold serve` once with a build that understands this store so recovery \
         can truncate it, stop it, then downgrade.",
        authority_dir(root).display()
    );
    let replayed_records = records.len();
    for (generation, delta) in records {
        delta.apply(&mut snapshot);
        snapshot.generation = generation;
    }
    let already_legacy = replayed_records == 0
        && snapshot.version == LEGACY_STORAGE_VERSION
        && marker
            .as_ref()
            .is_some_and(|marker| marker.version == LEGACY_STORAGE_VERSION)
        && !mutation_log_path(root).exists();
    let generation = snapshot.generation;

    // The replayed store is written into the VERSION-2 directory, whichever
    // one it currently occupies. Two shapes reach here and both must land
    // somewhere an older mold looks: a store an earlier build upgraded in
    // place (v3 bytes under the `-v2` name, the production incident), and a
    // proper v3 store in its own directory.
    let target = legacy_authority_dir(root);

    // The v2 store is SHARED STATE with whatever older binary has been using
    // this home, and the whole point of leaving it intact was that it might
    // still be in use. On a home where both formats were written it can be
    // AHEAD of the v3 one, and folding the v3 replay over it would discard
    // every print the older binary published since the upgrade and walk the
    // generation backwards. Every other checkpoint write in this file passes
    // the `existing <= generation` guard in `write_checkpoint_envelope`; this
    // one wrote the three copies with `atomic_write_bytes` directly and so
    // bypassed it entirely.
    //
    // Refuse rather than merge: the two indexes describe different sets of
    // prints and nothing here can know which membership the operator wants.
    if target != authority_dir(root) {
        if let Ok(existing) = read_checkpoint_at(&target.join(CHECKPOINT_FILE)) {
            ensure!(
                existing.generation <= generation,
                "the version-2 gallery archive authority in {} is at generation {}, AHEAD of the \
                 version-3 store's {}. An older binary has published to this home since the \
                 upgrade, and downgrading would discard those prints. Nothing has been changed. \
                 Decide which store is authoritative: to keep the version-2 one, stop every \
                 writer and remove {}; to keep the version-3 one, move the version-2 store aside \
                 first.",
                target.display(),
                existing.generation,
                generation,
                authority_dir_v3(root).display(),
            );
        }
    }

    fs::create_dir_all(&target)?;
    let mut legacy = snapshot.clone();
    legacy.version = LEGACY_STORAGE_VERSION;
    let (envelope, _) = serialize_envelope_at(&legacy, LEGACY_STORAGE_VERSION)?;
    // Checkpoint and backup first, then the marker, then the log: the same
    // order recovery reads them in, so an interruption anywhere leaves a
    // store a v3-capable build still recovers.
    atomic_write_bytes(&target.join(CHECKPOINT_FILE), &envelope)?;
    atomic_write_bytes(&target.join(BACKUP_CHECKPOINT_FILE), &envelope)?;
    atomic_write_bytes(&target.join(PREVIOUS_CHECKPOINT_FILE), &envelope)?;
    atomic_write_json(
        &target.join(MARKER_FILE),
        &MutationMarker {
            version: LEGACY_STORAGE_VERSION,
            committed_generation: generation,
            pending: None,
        },
    )?;
    // Neutralize every delta log. One left beside a v2 checkpoint is either
    // replayed by a later v3 process against the wrong generation, or
    // discarded with a warning that reads like corruption after a normal
    // boot. The v3 DIRECTORY goes too — `authority_dir` resolves a store by
    // its marker, so leaving that marker behind would make the next start
    // pick the store this command was asked to retire.
    for stale in [mutation_log_path(root), target.join(MUTATION_LOG_FILE)] {
        match fs::remove_file(&stale) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    let retired_v3 = authority_dir_v3(root);
    if retired_v3.is_dir() && retired_v3 != target {
        let parked = retired_v3.with_file_name(format!(
            "{AUTHORITY_DIR_V3}.downgraded-{}",
            mold_core::time::now_epoch_ms_u64()
        ));
        // Renamed rather than deleted: it is the only copy of the authority
        // between the last compaction and now, and this command has just
        // written its contents into the v2 store from a replay it performed
        // in memory. Keeping it costs a directory and buys a way back.
        fs::rename(&retired_v3, &parked)?;
        tracing::info!(
            parked = %parked.display(),
            "parked the version-3 gallery authority beside the downgraded store"
        );
    }
    sync_dir(&target)?;
    // This process must not keep believing it may append to a log that is
    // gone, or that its cached tail describes the checkpoint it just rewrote.
    disable_v3_writing(root);
    forget_authority_tail(root);

    // Read it back the way the old binary will.
    let verified = load_existing_read_only(root, &guard)?
        .context("the downgraded gallery authority did not load back")?;
    ensure!(
        verified.generation == generation,
        "the downgraded gallery authority reports generation {} rather than {generation}",
        verified.generation
    );
    let stored_version = read_checkpoint_at(&target.join(CHECKPOINT_FILE))?.version;
    ensure!(
        stored_version == LEGACY_STORAGE_VERSION,
        "the downgraded gallery authority checkpoint is still version {stored_version}"
    );
    forget_authority_tail(root);

    Ok(DowngradeOutcome {
        generation,
        replayed_records,
        already_legacy,
    })
}

/// A read-only description of the store's on-disk format, for an operator
/// deciding whether a downgrade is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AuthorityStorageStatus {
    pub present: bool,
    pub checkpoint_version: Option<u32>,
    pub marker_version: Option<u32>,
    pub generation: Option<u64>,
    pub log_records: usize,
    pub log_bytes: u64,
    pub pending_mutation: bool,
    pub torn_log_tail: bool,
    /// The version-2 store, when one exists and is not the active one.
    ///
    /// The divergence this whole switch is about is invisible if the status
    /// command only describes whichever store happens to be live: a home where
    /// both formats were written has TWO indexes, and which one is ahead is
    /// precisely the question an operator deciding whether to downgrade needs
    /// answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_store: Option<AuthorityStoreFacts>,
    /// The version-3 store, when one exists and is not the active one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_store: Option<AuthorityStoreFacts>,
}

/// One store's own facts, independent of which one is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AuthorityStoreFacts {
    pub checkpoint_version: Option<u32>,
    pub generation: Option<u64>,
}

/// Read one store directory's facts without resolving which is active.
fn store_facts(dir: &Path) -> Option<AuthorityStoreFacts> {
    if !dir.is_dir() {
        return None;
    }
    let checkpoint = read_checkpoint_at(&dir.join(CHECKPOINT_FILE)).ok();
    Some(AuthorityStoreFacts {
        checkpoint_version: checkpoint.as_ref().map(|snapshot| snapshot.version),
        generation: checkpoint.as_ref().map(|snapshot| snapshot.generation),
    })
}

/// Describe the store without touching it.
pub fn storage_status(root: &Path) -> anyhow::Result<AuthorityStorageStatus> {
    let guard = crate::batch_transaction::acquire_gallery_bookkeeping_lock(root)?;
    let root = guard.canonical_root();
    let active = authority_dir(root);
    let legacy_store = (legacy_authority_dir(root) != active)
        .then(|| store_facts(&legacy_authority_dir(root)))
        .flatten();
    let log_store = (authority_dir_v3(root) != active)
        .then(|| store_facts(&authority_dir_v3(root)))
        .flatten();
    if !active.is_dir() {
        return Ok(AuthorityStorageStatus {
            present: false,
            checkpoint_version: None,
            marker_version: None,
            generation: None,
            log_records: 0,
            log_bytes: 0,
            pending_mutation: false,
            torn_log_tail: false,
            legacy_store,
            log_store,
        });
    }
    let marker = read_marker(root).ok().flatten();
    let checkpoint = read_checkpoint_at(&checkpoint_path(root)).ok();
    let scan = checkpoint
        .as_ref()
        .map(|snapshot| read_mutation_log(root, snapshot.generation))
        .transpose()?;
    Ok(AuthorityStorageStatus {
        present: true,
        checkpoint_version: checkpoint.as_ref().map(|snapshot| snapshot.version),
        marker_version: marker.as_ref().map(|marker| marker.version),
        generation: marker
            .as_ref()
            .map(|marker| marker.committed_generation)
            .or_else(|| checkpoint.as_ref().map(|snapshot| snapshot.generation)),
        log_records: scan.as_ref().map(|scan| scan.records.len()).unwrap_or(0),
        log_bytes: mutation_log_bytes(root),
        pending_mutation: marker.is_some_and(|marker| marker.pending.is_some()),
        torn_log_tail: scan.map(|scan| scan.discarded > 0).unwrap_or(false),
        legacy_store,
        log_store,
    })
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
        (Some(marker), Some(mut snapshot)) => {
            ensure!(
                marker.pending.is_none(),
                "gallery authority mutation is pending; startup recovery is required"
            );
            ensure!(
                !wal_path(root).try_exists()?,
                "gallery authority WAL is unresolved; startup recovery is required"
            );
            // v3 keeps the checkpoint behind the marker and the delta log in
            // between, so this read path replays the log too — WITHOUT
            // truncating anything. A torn tail here is a job for the explicit
            // startup recovery, exactly as an unresolved WAL is.
            let checkpoint_generation = snapshot.generation;
            let MutationLogScan {
                records, discarded, ..
            } = read_mutation_log(root, snapshot.generation)?;
            let log_records = records.len();
            for (generation, delta) in records {
                delta.apply(&mut snapshot);
                snapshot.generation = generation;
            }
            ensure!(
                discarded == 0,
                "gallery authority mutation log has an unresolved tail; startup recovery is required"
            );
            ensure!(
                marker.committed_generation == snapshot.generation,
                "gallery authority checkpoint generation does not match its stable marker"
            );
            remember_authority_tail(
                root,
                AuthorityTail::from_recovered(
                    &snapshot,
                    checkpoint_generation,
                    (log_records, mutation_log_bytes(root)),
                ),
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
    load_or_initialize_with_authority_log(root, guard, authority_log_requested(), legacy)
}

pub(crate) fn load_or_initialize_with_authority_log(
    root: &Path,
    guard: &GalleryBookkeepingGuard,
    authority_log: bool,
    legacy: impl FnOnce() -> anyhow::Result<CommittedArchiveIndex>,
) -> anyhow::Result<LoadedAuthority> {
    guard.ensure_root(root)?;
    let root = guard.canonical_root();
    let mut snapshot = match recover_storage(root, authority_log)? {
        Some(snapshot) => snapshot,
        None => {
            let mut index = legacy()?;
            populate_missing_facts(root, &mut index)?;
            let legacy_evidence_epochs =
                crate::batch_transaction::legacy_gallery_evidence_paths(root)?
                    .into_iter()
                    .map(|path| (path, 0))
                    .collect();
            // A brand-new store is created at the version this process
            // WRITES, not at the newest this build understands: a default
            // build must leave a home an older mold can still publish to.
            let storage_version = if authority_log {
                STORAGE_VERSION
            } else {
                LEGACY_STORAGE_VERSION
            };
            let snapshot = AuthoritySnapshot {
                version: storage_version,
                generation: 0,
                index,
                legacy_evidence_epochs,
            };
            let (envelope, _) = serialize_envelope_at(&snapshot, storage_version)?;
            if storage_version == STORAGE_VERSION {
                // A fresh v3 store goes in the v3 DIRECTORY, for the same
                // reason the upgrade does. The path helpers resolve by
                // marker, and a home with no store yet has none, so writing
                // through them would put version-3 bytes under the `-v2`
                // name — the production incident's shape, reached this time
                // on a home that never had a v2 store at all. An older mold
                // sharing it would then find a version it cannot read at the
                // only path it looks, and refuse to publish, instead of
                // initializing the v2 store beside it that the two-directory
                // design promises.
                write_fresh_store_v3(root, &snapshot, &envelope)?;
            } else {
                write_checkpoint_envelope(root, 0, &envelope, None, true)?;
                atomic_write_bytes(&previous_checkpoint_path(root), &envelope)?;
                atomic_write_bytes(&backup_checkpoint_path(root), &envelope)?;
                write_marker(
                    root,
                    &MutationMarker {
                        version: storage_version,
                        committed_generation: 0,
                        pending: None,
                    },
                )?;
            }
            snapshot
        }
    };
    remember_authority_tail(
        root,
        AuthorityTail::from_recovered(&snapshot, snapshot.generation, (0, 0)),
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
    // Recovery has run for this root, so this process KNOWS the log's tail is
    // intact — but it writes v3 only if the operator asked for it.
    if authority_log {
        enable_v3_writing(root);
    } else {
        disable_v3_writing(root);
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
    /// The key view of the committed index this generation describes; v3
    /// diffs the next commit against it.
    keys: AuthorityIndexKeys,
    /// The generation the on-disk CHECKPOINT is at. Under v3 the checkpoint
    /// is only rewritten at compaction, so this trails `generation`.
    checkpoint_generation: u64,
    /// Records and bytes in `mutation.log`, for the compaction thresholds.
    log_records: usize,
    log_bytes: u64,
}

impl AuthorityTail {
    fn from_recovered(
        snapshot: &AuthoritySnapshot,
        checkpoint_generation: u64,
        log: (usize, u64),
    ) -> Self {
        Self {
            generation: snapshot.generation,
            legacy_evidence_epochs: snapshot.legacy_evidence_epochs.clone(),
            previous_rolled_generation: snapshot.generation,
            previous_rolled_at: std::time::Instant::now(),
            keys: AuthorityIndexKeys::of(&snapshot.index),
            checkpoint_generation,
            log_records: log.0,
            log_bytes: log.1,
        }
    }
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
            let snapshot = recover_storage(root, v3_writing_enabled(root))
                .context("gallery authority recovery")?
                .context("gallery authority checkpoint is missing")?;
            // Recovery just compacted, so the checkpoint IS the tail and the
            // log is empty.
            AuthorityTail::from_recovered(&snapshot, snapshot.generation, (0, 0))
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

    // The full snapshot and its envelope are built ONLY where they are
    // written. Serializing the whole archive index is the O(gallery) cost per
    // print v3 exists to remove, so the delta path must never touch it —
    // building it "once, for whoever needs it" is what made the first v3
    // measurement only 16 % faster than v2.
    // The version this commit WRITES. It stamps the snapshot's own field as
    // well as the envelope's: an older mold checks BOTH, so a payload saying
    // 3 inside a v2 envelope still locks it out.
    let write_version = if v3_writing_enabled(root) {
        STORAGE_VERSION
    } else {
        LEGACY_STORAGE_VERSION
    };
    let build_snapshot =
        |index: &CommittedArchiveIndex, legacy: &std::collections::BTreeMap<String, u64>| {
            AuthoritySnapshot {
                version: write_version,
                generation,
                index: index.clone(),
                legacy_evidence_epochs: legacy.clone(),
            }
        };
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

    let mut tail = AuthorityTail {
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
        keys: AuthorityIndexKeys::default(),
        checkpoint_generation: current.checkpoint_generation,
        log_records: current.log_records,
        log_bytes: current.log_bytes,
    };

    if v3_writing_enabled(root) {
        // v3: one appended delta plus the marker — three fsyncs, and the
        // bytes written are proportional to what CHANGED rather than to the
        // size of the gallery.
        let touched = exact_names
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let delta = IndexDelta::between(
            (&current.keys, &current.legacy_evidence_epochs),
            (index, &tail.legacy_evidence_epochs),
            &touched,
        );
        let record = serialize_mutation_log_record(generation, kind, &exact_names, &delta)?;
        guard.ensure_root(root)?;
        let log_bytes = append_mutation_log_record(root, &record)?;
        guard.ensure_root(root)?;
        write_marker(
            root,
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: generation,
                pending: None,
            },
        )?;
        tail.log_records = current.log_records.saturating_add(1);
        tail.log_bytes = log_bytes;
        if tail.log_records >= MUTATION_LOG_MAX_RECORDS || tail.log_bytes >= MUTATION_LOG_MAX_BYTES
        {
            guard.ensure_root(root)?;
            compact_mutation_log(
                root,
                &build_snapshot(index, &tail.legacy_evidence_epochs),
                tail.checkpoint_generation,
            )?;
            tail.checkpoint_generation = generation;
            tail.previous_rolled_generation = generation;
            tail.previous_rolled_at = std::time::Instant::now();
            tail.log_records = 0;
            tail.log_bytes = 0;
        }
    } else {
        // v2: the pending-marker + whole-snapshot WAL protocol, still what a
        // process writes until its startup recovery has run for this root.
        let snapshot = build_snapshot(index, &tail.legacy_evidence_epochs);
        let (envelope, snapshot_sha256) = serialize_envelope_at(&snapshot, write_version)?;
        guard.ensure_root(root)?;
        write_marker(
            root,
            &MutationMarker {
                version: LEGACY_STORAGE_VERSION,
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
            Some(current.checkpoint_generation),
            roll_previous,
        )?;
        guard.ensure_root(root)?;
        write_marker(
            root,
            &MutationMarker {
                version: LEGACY_STORAGE_VERSION,
                committed_generation: generation,
                pending: None,
            },
        )?;
        guard.ensure_root(root)?;
        remove_wal(root)?;
        guard.ensure_root(root)?;
        atomic_write_bytes(&backup_checkpoint_path(root), &envelope)?;
        // This checkpoint holds everything the log did, so leaving the log
        // behind would strand records the next replay reads as
        // non-contiguous — a hard "startup recovery is required", or a WARN
        // about discarding a tail, after a perfectly normal boot. A v2 writer
        // entering a v3 store (a rollback, or the switch turned off) leaves it
        // consistently v2. Reachable without recovery: the publication gate's
        // process cache can be installed by `load_existing_read_only`, which
        // never enables v3.
        if mutation_log_path(root).exists() {
            truncate_mutation_log(root, 0)?;
        }
        tail.checkpoint_generation = generation;
        tail.log_records = 0;
        tail.log_bytes = 0;
    }
    crate::batch_transaction::remove_legacy_gallery_evidence(root, &collect_legacy)?;
    // The next commit diffs against this one — names and epochs only, never
    // the records.
    tail.keys = AuthorityIndexKeys::of(index);
    remember_authority_tail(root, tail);
    Ok(generation)
}

/// Fold the log into a fresh checkpoint and start a new one.
///
/// Ordering is the same argument the commit protocol makes: the checkpoint
/// lands durably FIRST, then the log is dropped. A crash between them replays
/// records the checkpoint already contains, which the contiguity check
/// discards as non-contiguous — it never loses one.
/// Write a fresh version-3 store into its OWN directory, leaving the version-2
/// store exactly as it stands.
///
/// This is the one-way half of the format change and it happens only when
/// `gallery.authority_log` is on. Writing the v3 bytes over the v2 store in
/// place — what an earlier build did on first start, with no switch at all —
/// took production down: every older mold sharing the `$MOLD_HOME` refused to
/// publish, and the backup had been rewritten at v3 too, so there was nothing
/// left to roll back to.
///
/// The marker is written LAST. `authority_dir` resolves a store by the
/// presence of that file, so until it lands the v2 store is still the live
/// one, and an interruption anywhere before it leaves a half-written v3
/// directory nothing reads.
fn upgrade_store_to_v3(root: &Path, snapshot: &AuthoritySnapshot) -> anyhow::Result<()> {
    let dir = authority_dir_v3(root);
    fs::create_dir_all(&dir)?;
    sync_dir(
        dir.parent()
            .context("gallery authority directory has no parent")?,
    )?;
    sync_dir(&dir)?;
    let mut snapshot = snapshot.clone();
    snapshot.version = STORAGE_VERSION;
    let (envelope, _) = serialize_envelope_at(&snapshot, STORAGE_VERSION)?;
    atomic_write_bytes(&dir.join(CHECKPOINT_FILE), &envelope)?;
    atomic_write_bytes(&dir.join(BACKUP_CHECKPOINT_FILE), &envelope)?;
    atomic_write_bytes(&dir.join(PREVIOUS_CHECKPOINT_FILE), &envelope)?;
    atomic_write_json(
        &dir.join(MARKER_FILE),
        &MutationMarker {
            version: STORAGE_VERSION,
            committed_generation: snapshot.generation,
            pending: None,
        },
    )?;
    tracing::warn!(
        directory = %dir.display(),
        generation = snapshot.generation,
        "upgraded the gallery archive authority to storage version 3; the version-2 store is \
         retained beside it. A mold older than 0.29 will keep publishing to the version-2 store \
         and the two will diverge — run `mold system gallery-authority downgrade` before rolling \
         one back."
    );
    Ok(())
}

/// Lay down a brand-new version-3 store in the version-3 directory.
///
/// Marker LAST, exactly as `upgrade_store_to_v3` does: `authority_dir`
/// resolves a store by that file, so until it lands nothing reads the
/// half-written directory.
fn write_fresh_store_v3(
    root: &Path,
    snapshot: &AuthoritySnapshot,
    envelope: &[u8],
) -> anyhow::Result<()> {
    let dir = authority_dir_v3(root);
    fs::create_dir_all(&dir)?;
    sync_dir(
        dir.parent()
            .context("gallery authority directory has no parent")?,
    )?;
    atomic_write_bytes(&dir.join(CHECKPOINT_FILE), envelope)?;
    atomic_write_bytes(&dir.join(BACKUP_CHECKPOINT_FILE), envelope)?;
    atomic_write_bytes(&dir.join(PREVIOUS_CHECKPOINT_FILE), envelope)?;
    atomic_write_json(
        &dir.join(MARKER_FILE),
        &MutationMarker {
            version: STORAGE_VERSION,
            committed_generation: snapshot.generation,
            pending: None,
        },
    )?;
    sync_dir(&dir)?;

    // An older binary sharing this home looks ONLY at the version-2 name. On
    // a home upgraded from v2 it finds the frozen store the upgrade left
    // behind, but a home initialized FRESH with the switch on has never had
    // one — and `recover_storage` returning `Ok(None)` drops that binary into
    // `load_committed_archive_index_legacy`, which re-derives a whole index
    // by scanning the output directory. That is the path that silently
    // resurrects deleted prints, and the two-directory design would otherwise
    // make it reachable again for exactly this case.
    //
    // So lay down an authoritative, empty version-2 store beside the v3 one.
    // An older binary then reads a real index that says "nothing published
    // yet" and publishes forward from it, instead of inventing one from the
    // filesystem.
    let mut legacy = snapshot.clone();
    legacy.version = LEGACY_STORAGE_VERSION;
    let (legacy_envelope, _) = serialize_envelope_at(&legacy, LEGACY_STORAGE_VERSION)?;
    write_fresh_store_v2(root, &legacy, &legacy_envelope)?;
    Ok(())
}

/// Lay down a version-2 store in the version-2 directory.
fn write_fresh_store_v2(
    root: &Path,
    snapshot: &AuthoritySnapshot,
    envelope: &[u8],
) -> anyhow::Result<()> {
    let dir = legacy_authority_dir(root);
    fs::create_dir_all(&dir)?;
    atomic_write_bytes(&dir.join(CHECKPOINT_FILE), envelope)?;
    atomic_write_bytes(&dir.join(BACKUP_CHECKPOINT_FILE), envelope)?;
    atomic_write_bytes(&dir.join(PREVIOUS_CHECKPOINT_FILE), envelope)?;
    atomic_write_json(
        &dir.join(MARKER_FILE),
        &MutationMarker {
            version: LEGACY_STORAGE_VERSION,
            committed_generation: snapshot.generation,
            pending: None,
        },
    )?;
    sync_dir(&dir)?;
    Ok(())
}

fn compact_mutation_log(
    root: &Path,
    snapshot: &AuthoritySnapshot,
    existing_checkpoint_generation: u64,
) -> anyhow::Result<()> {
    compact_mutation_log_at(
        root,
        snapshot,
        existing_checkpoint_generation,
        STORAGE_VERSION,
    )
}

fn compact_mutation_log_at(
    root: &Path,
    snapshot: &AuthoritySnapshot,
    existing_checkpoint_generation: u64,
    storage_version: u32,
) -> anyhow::Result<()> {
    // A snapshot recovered from one version still carries that version in its
    // own field; the checkpoint compaction writes is the one the caller asked
    // for, which is what makes an upgrade — or a downgrade — happen exactly
    // once.
    let mut snapshot = snapshot.clone();
    snapshot.version = storage_version;
    let snapshot = &snapshot;
    let (envelope, _) = serialize_envelope_at(snapshot, storage_version)?;
    write_checkpoint_envelope(
        root,
        snapshot.generation,
        &envelope,
        Some(existing_checkpoint_generation),
        true,
    )?;
    atomic_write_bytes(&backup_checkpoint_path(root), &envelope)?;
    truncate_mutation_log(root, 0)?;
    Ok(())
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
    /// Stand in for another process's checkpoint: writes the bytes with no
    /// generation-regression check, so a test can move the authority in
    /// either direction.
    ///
    /// It deliberately does NOT drop this process's cached tail — dropping it
    /// made every caller run cold, so `cached_commit_tail` returned `None` at
    /// its first lookup and the fast path's on-disk guards were never
    /// exercised by the test named for them.
    fn write_checkpoint_for_test(root: &Path, snapshot: &AuthoritySnapshot) {
        let (envelope, _) = serialize_envelope(snapshot).unwrap();
        fs::create_dir_all(authority_dir(root)).unwrap();
        atomic_write_bytes(&checkpoint_path(root), &envelope).unwrap();
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

    fn publish(
        dir: &Path,
        guard: &GalleryBookkeepingGuard,
        index: &mut CommittedArchiveIndex,
        generation: u64,
        name: &str,
    ) -> u64 {
        index.quarantined_names.insert(name.to_string());
        commit_snapshot(
            dir,
            guard,
            generation,
            index,
            "publish_batch",
            vec![name.to_string()],
        )
        .unwrap()
    }

    /// The measurement the campaign asked for: is publishing a print still
    /// proportional to the size of the library?
    ///
    /// Run with `--ignored --nocapture`; it prints bytes written and wall
    /// clock per commit at two index sizes on both protocols.
    #[test]
    #[ignore]
    fn measure_commit_cost_against_index_size() {
        fn entry(name: &str) -> CommittedArchiveEntry {
            let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "prompt": "a very long prompt ".repeat(8),
                "model": "flux-dev:q8",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "guidance": 3.5
            }))
            .unwrap();
            CommittedArchiveEntry {
                identity: ArchivedChildIdentity {
                    parent_id: format!("parent-{name}"),
                    attempt_generation: 0,
                    child_index: 0,
                    final_name: name.to_string(),
                    checksum_sha256: format!("{:x}", Sha256::digest(name.as_bytes())),
                    size_bytes: 1_234_567,
                },
                record: GenerationRecord::from_save(
                    Path::new("/tmp"),
                    name,
                    OutputFormat::Png,
                    OutputMetadata::from_generate_request(&request, 1, None, "bench"),
                    RecordSource::Server,
                    1,
                ),
                facts: None,
                retained_media: Vec::new(),
            }
        }

        for size in [1_000_usize, 10_000] {
            for v3 in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
                let mut seeded = CommittedArchiveIndex::default();
                for n in 0..size {
                    let name = format!("seed-{n:06}.png");
                    seeded.entries.insert(name.clone(), entry(&name));
                }
                let loaded = load_or_initialize(dir.path(), &guard, || Ok(seeded.clone())).unwrap();
                if !v3 {
                    disable_v3_writing_for_test(guard.canonical_root());
                }
                let mut index = loaded.index;
                let mut generation = loaded.generation;
                let before = authority_bytes(dir.path());
                let started = std::time::Instant::now();
                const COMMITS: usize = 16;
                for n in 0..COMMITS {
                    let name = format!("new-{n:04}.png");
                    index.entries.insert(name.clone(), entry(&name));
                    generation = commit_snapshot(
                        dir.path(),
                        &guard,
                        generation,
                        &mut index,
                        "publish_batch",
                        vec![name],
                    )
                    .unwrap();
                }
                let elapsed = started.elapsed();
                let written = authority_bytes(dir.path()).saturating_sub(before);
                eprintln!(
                    "index={size:>6} protocol={:<2} {:>8.2} ms/commit  log+checkpoint delta {:>10} B",
                    if v3 { "v3" } else { "v2" },
                    elapsed.as_secs_f64() * 1000.0 / COMMITS as f64,
                    written,
                );
            }
        }
    }

    fn authority_bytes(root: &Path) -> u64 {
        fs::read_dir(authority_dir(root))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter_map(|entry| entry.metadata().ok())
                    .map(|metadata| metadata.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// The regression that made this switch exist.
    ///
    /// On a shared `$MOLD_HOME` a new build started, upgraded the store to
    /// version 3, and the production 0.28.0 service beside it — a v2 reader —
    /// failed every publication with "unsupported gallery authority
    /// generation marker" until the store was downgraded. So writing v3 is
    /// opt-in, and a default build leaves a v2 store alone.
    #[test]
    fn v3_writing_is_off_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        let generation = publish(dir.path(), &guard, &mut index, initial.generation, "p.png");

        assert!(
            !mutation_log_path(dir.path()).exists(),
            "a default build writes no delta log"
        );
        let marker = read_marker(dir.path()).unwrap().unwrap();
        assert_eq!(marker.version, LEGACY_STORAGE_VERSION);
        assert_eq!(marker.committed_generation, generation);
        let checkpoint_bytes = fs::read(checkpoint_path(dir.path())).unwrap();
        let envelope: serde_json::Value = serde_json::from_slice(&checkpoint_bytes).unwrap();
        assert_eq!(envelope["version"], LEGACY_STORAGE_VERSION);
        assert_eq!(envelope["snapshot"]["version"], LEGACY_STORAGE_VERSION);
        assert_eq!(
            read_checkpoint_at(&checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            generation,
            "v2 still writes a whole checkpoint per commit"
        );
    }

    /// Turning the switch on is what enables the log — and a default build
    /// still READS what it wrote.
    #[test]
    fn the_authority_log_switch_turns_v3_writing_on() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let generation = publish(dir.path(), &guard, &mut index, initial.generation, "p.png");
        assert!(mutation_log_path(dir.path()).exists());
        assert_eq!(
            read_marker(dir.path()).unwrap().unwrap().version,
            STORAGE_VERSION
        );

        // A DEFAULT build opening the same store reads it, and then keeps
        // writing v2 into it rather than extending the log.
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let reopened = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(reopened.generation, generation);
        assert!(reopened.index.quarantined_names.contains("p.png"));
        let mut index = reopened.index;
        let next = publish(dir.path(), &guard, &mut index, reopened.generation, "q.png");
        assert_eq!(
            read_marker(dir.path()).unwrap().unwrap().version,
            LEGACY_STORAGE_VERSION,
            "a default build writes v2 back"
        );
        assert!(
            !mutation_log_path(dir.path()).exists(),
            "and drops the log its checkpoint now supersedes"
        );
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let final_read = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(final_read.generation, next);
        assert!(final_read.index.quarantined_names.contains("p.png"));
        assert!(final_read.index.quarantined_names.contains("q.png"));
    }

    /// A store written by a NEWER mold must say what to do about it, not just
    /// that the number is wrong. (The already-released 0.28.0 message cannot
    /// be changed; this is the one every future build prints.)
    #[test]
    fn an_unreadable_store_version_names_the_downgrade_command() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(authority_dir(dir.path())).unwrap();
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: 1,
                pending: None,
            },
        )
        .unwrap();
        // Hand-edit the version past what this build understands.
        let path = marker_path(dir.path());
        let mut marker: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        marker["version"] = serde_json::json!(STORAGE_VERSION + 1);
        fs::write(&path, serde_json::to_vec(&marker).unwrap()).unwrap();

        let error = read_marker(dir.path()).unwrap_err().to_string();
        assert!(
            error.contains("mold system gallery-authority downgrade"),
            "unexpected error: {error}"
        );
        assert!(error.contains("newer"), "unexpected error: {error}");
    }

    /// The shared-home property the production incident was missing: the
    /// upgrade writes a NEW store beside the old one, and the version-2 store
    /// is left byte-for-byte as it stood.
    #[test]
    fn the_upgrade_leaves_the_v2_store_intact_beside_a_new_v3_one() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        // A v2 store with real content, written by a default build.
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        let v2_generation = publish(
            dir.path(),
            &guard,
            &mut index,
            initial.generation,
            "old.png",
        );
        let v2_dir = legacy_authority_dir(dir.path());
        let v2_before: Vec<(String, Vec<u8>)> = fs::read_dir(&v2_dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| {
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    fs::read(entry.path()).unwrap(),
                )
            })
            .collect();
        assert!(!v2_before.is_empty());

        // Now an operator opts in and restarts.
        forget_authority_tail(guard.canonical_root());
        let upgraded =
            load_or_initialize_with_authority_log(dir.path(), &guard, true, || unreachable!())
                .unwrap();
        assert_eq!(upgraded.generation, v2_generation);
        assert!(upgraded.index.quarantined_names.contains("old.png"));

        // The v3 store exists, in its own directory...
        let v3_dir = authority_dir_v3(dir.path());
        assert!(v3_dir.join(MARKER_FILE).is_file());
        assert_eq!(
            read_checkpoint_at(&v3_dir.join(CHECKPOINT_FILE))
                .unwrap()
                .version,
            STORAGE_VERSION
        );
        assert_eq!(
            authority_dir(dir.path()),
            v3_dir,
            "and it is now the live one"
        );

        // ...and every byte of the v2 store is untouched, so an older mold
        // sharing this home keeps publishing and a rollback needs no repair.
        for (name, before) in &v2_before {
            assert_eq!(
                &fs::read(v2_dir.join(name)).unwrap(),
                before,
                "v2 {name} was modified by the upgrade"
            );
        }
    }

    /// A DEFAULT build must not rewrite a store just by starting. The upgrade
    /// used to be a side effect of `recover_storage`, reached from
    /// `mold serve` startup before any request.
    #[test]
    fn a_default_build_never_rewrites_a_v2_store_on_startup() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        let mut index = initial.index;
        publish(dir.path(), &guard, &mut index, initial.generation, "p.png");
        let v2_dir = legacy_authority_dir(dir.path());
        let before: Vec<(String, Vec<u8>)> = fs::read_dir(&v2_dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| {
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    fs::read(entry.path()).unwrap(),
                )
            })
            .collect();

        for _ in 0..3 {
            forget_authority_tail(guard.canonical_root());
            load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        }
        assert!(
            !authority_dir_v3(dir.path()).exists(),
            "a default build creates no v3 store"
        );
        for (name, bytes) in &before {
            assert_eq!(&fs::read(v2_dir.join(name)).unwrap(), bytes, "{name} moved");
        }
    }

    #[test]
    fn downgrade_folds_the_log_into_a_v2_store() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..3_u64 {
            generation = publish(
                dir.path(),
                &guard,
                &mut index,
                generation,
                &format!("p{step}.png"),
            );
        }
        assert!(mutation_log_path(dir.path()).exists());
        drop(guard);

        let outcome = downgrade_to_legacy_storage(dir.path()).unwrap();
        assert_eq!(outcome.generation, generation);
        assert_eq!(outcome.replayed_records, 3);

        // Exactly the shape a v2 reader expects.
        assert!(!mutation_log_path(dir.path()).exists());
        let marker = read_marker(dir.path()).unwrap().unwrap();
        assert_eq!(marker.version, LEGACY_STORAGE_VERSION);
        assert_eq!(marker.committed_generation, generation);
        for path in [
            checkpoint_path(dir.path()),
            backup_checkpoint_path(dir.path()),
        ] {
            let envelope: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(envelope["version"], LEGACY_STORAGE_VERSION, "{path:?}");
            assert_eq!(
                envelope["snapshot"]["version"], LEGACY_STORAGE_VERSION,
                "{path:?}"
            );
            assert_eq!(envelope["snapshot"]["generation"], generation);
        }

        // And the v2 read path — the one the old binary takes — loads it.
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let loaded = load_existing_read_only(dir.path(), &guard)
            .unwrap()
            .expect("a downgraded store is readable");
        assert_eq!(loaded.generation, generation);
        for step in 0..3_u64 {
            assert!(loaded
                .index
                .quarantined_names
                .contains(&format!("p{step}.png")));
        }
    }

    /// A home that opts in before it has any store at all still leaves an
    /// authoritative version-2 store for an older mold.
    ///
    /// Two separate hazards meet here. Writing version-3 bytes under the
    /// `-v2` name reproduced the incident on a brand-new home; but leaving
    /// that name EMPTY is its own bug, because an older binary that finds no
    /// store falls into `load_committed_archive_index_legacy` and re-derives
    /// an index by scanning the output directory — the path that silently
    /// resurrects deleted prints. So the fresh v3 store is accompanied by a
    /// real, empty version-2 one.
    #[test]
    fn a_fresh_v3_home_leaves_the_v2_name_free() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let generation = publish(dir.path(), &guard, &mut index, initial.generation, "a.png");

        // The store is a real v3 one, in its own directory.
        let v3 = authority_dir_v3(dir.path());
        assert!(v3.is_dir(), "a fresh v3 store needs its own directory");
        assert_eq!(authority_dir(dir.path()), v3);
        assert_eq!(
            read_checkpoint_at(&v3.join(CHECKPOINT_FILE))
                .unwrap()
                .version,
            STORAGE_VERSION
        );
        assert_eq!(
            read_marker(dir.path()).unwrap().unwrap().version,
            STORAGE_VERSION
        );
        assert_eq!(read_mutation_log(dir.path(), 0).unwrap().records.len(), 1);
        assert_eq!(generation, 1);

        // And the name an older mold reads holds a real version-2 store it
        // can both read and publish forward from — never a version it cannot
        // read, and never nothing at all.
        let v2 = legacy_authority_dir(dir.path());
        let legacy = read_checkpoint_at(&v2.join(CHECKPOINT_FILE))
            .expect("an older mold must find an authoritative store, not an empty directory");
        assert_eq!(
            legacy.version, LEGACY_STORAGE_VERSION,
            "an older mold must not find a version it cannot read at the v2 path"
        );
        assert!(
            legacy.index.quarantined_names.is_empty(),
            "and it starts empty rather than inventing membership"
        );
        let marker = read_checkpoint_at(&v2.join(BACKUP_CHECKPOINT_FILE)).unwrap();
        assert_eq!(marker.version, LEGACY_STORAGE_VERSION);
        assert!(
            v2.join(MARKER_FILE).exists(),
            "with its own generation marker, so recovery does not treat it as absent"
        );
    }

    /// The shape the production incident actually left behind, and the one
    /// the repair has to survive: version-3 bytes UNDER THE `-v2` NAME.
    ///
    /// An earlier build upgraded the store in place, with no separate
    /// directory and no switch, so a host carries a v3 checkpoint, a v3
    /// marker and a delta log at the path every older mold reads as version 2
    /// — and there is no `-v3` directory to park. `authority_dir` resolves a
    /// store by its marker rather than by its name precisely so this shape is
    /// still found; without this test the repair path that runs against a
    /// real host was the only thing exercising it.
    #[test]
    fn downgrade_repairs_a_store_upgraded_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..3_u64 {
            generation = publish(
                dir.path(),
                &guard,
                &mut index,
                generation,
                &format!("p{step}.png"),
            );
        }
        drop(guard);

        // Collapse the two-directory store into the one-directory shape the
        // older build produced: v3 contents at the v2 path, no v3 directory.
        let v3 = authority_dir_v3(dir.path());
        let v2 = legacy_authority_dir(dir.path());
        assert!(v3.is_dir(), "the fixture needs a real v3 store to collapse");
        // Rebuild the one-directory shape the older build produced: the v3
        // contents move to the v2 path and nothing is left beside them.
        if v2.exists() {
            fs::remove_dir_all(&v2).unwrap();
        }
        fs::rename(&v3, &v2).unwrap();
        assert!(!v3.exists());
        // This is now what an older mold sees at the only path it knows.
        assert_eq!(authority_dir(dir.path()), v2);
        assert_eq!(read_marker(dir.path()).unwrap().unwrap().version, 3);
        assert!(v2.join(MUTATION_LOG_FILE).exists());
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());

        let outcome = downgrade_to_legacy_storage(dir.path()).unwrap();
        assert_eq!(outcome.generation, generation);
        assert_eq!(outcome.replayed_records, 3);
        assert!(!outcome.already_legacy);

        // The one directory an old binary reads is now version 2 throughout,
        // with no log left beside it to be replayed against the wrong
        // generation.
        assert!(!v2.join(MUTATION_LOG_FILE).exists());
        assert!(!mutation_log_path(dir.path()).exists());
        let marker = read_marker(dir.path()).unwrap().unwrap();
        assert_eq!(marker.version, LEGACY_STORAGE_VERSION);
        assert_eq!(marker.committed_generation, generation);
        for file in [CHECKPOINT_FILE, BACKUP_CHECKPOINT_FILE] {
            let envelope: serde_json::Value =
                serde_json::from_slice(&fs::read(v2.join(file)).unwrap()).unwrap();
            assert_eq!(envelope["version"], LEGACY_STORAGE_VERSION, "{file}");
            assert_eq!(
                envelope["snapshot"]["version"], LEGACY_STORAGE_VERSION,
                "{file}"
            );
        }

        // And every published name survived the fold.
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let loaded = load_existing_read_only(dir.path(), &guard)
            .unwrap()
            .expect("a repaired store is readable");
        assert_eq!(loaded.generation, generation);
        for step in 0..3_u64 {
            assert!(loaded
                .index
                .quarantined_names
                .contains(&format!("p{step}.png")));
        }
        drop(guard);

        // Idempotent: running the repair twice is not an error.
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let again = downgrade_to_legacy_storage(dir.path()).unwrap();
        assert_eq!(again.generation, generation);
        assert_eq!(again.replayed_records, 0);
        assert!(again.already_legacy);
    }

    /// The documented remedy must not destroy the half it was called to
    /// rescue.
    ///
    /// On a home where both formats were written — a 0.29 binary publishing
    /// v3 while an older one kept publishing v2 — the two stores diverge, and
    /// the v2 one can be AHEAD. `downgrade` replayed the v3 log straight over
    /// all three v2 checkpoint copies with `atomic_write_bytes`, which bypasses
    /// the `existing <= generation` guard every other checkpoint write goes
    /// through, so it silently discarded the newer v2 prints and walked the
    /// generation backwards.
    /// The tooling built to manage the divergence has to be able to SEE it.
    ///
    /// `storage_status` described only whichever store was active, so on a
    /// home where both formats were written — the one case the switch exists
    /// to manage — an operator could not tell that a second index existed at
    /// all, let alone which one was ahead.
    /// The switch must mean the same thing to every publication path, not
    /// just to `run_server`.
    ///
    /// It used to be written only by the server entry point, so a forced-local
    /// `mold run` or the TUI opened a v3-enabled home with the flag false and
    /// committed version 2 into it — silently reverting the operator's setting
    /// on the next local render. Both sides now read one function.
    #[test]
    fn the_switch_is_one_decision_read_from_the_config() {
        let mut config = mold_core::Config::default();
        assert!(
            !authority_log_from_config(&config),
            "default is off; writing v3 is always a decision"
        );
        config.gallery.authority_log = true;
        assert!(authority_log_from_config(&config));
    }

    #[test]
    fn storage_status_reports_both_stores_and_their_generations() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let v3_generation = publish(dir.path(), &guard, &mut index, initial.generation, "v3.png");
        drop(guard);

        // An older binary publishing into the version-2 store beside it.
        let v2_dir = legacy_authority_dir(dir.path());
        let ahead = v3_generation + 7;
        let mut v2_snapshot = empty_snapshot(ahead);
        v2_snapshot.version = LEGACY_STORAGE_VERSION;
        let (envelope, _) = serialize_envelope_at(&v2_snapshot, LEGACY_STORAGE_VERSION).unwrap();
        atomic_write_bytes(&v2_dir.join(CHECKPOINT_FILE), &envelope).unwrap();
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());

        let status = storage_status(dir.path()).unwrap();
        // The active store is the v3 one.
        assert_eq!(status.checkpoint_version, Some(STORAGE_VERSION));
        assert_eq!(status.generation, Some(v3_generation));
        // And the other one is reported rather than hidden.
        let legacy = status
            .legacy_store
            .expect("the version-2 store beside the active one must be reported");
        assert_eq!(legacy.checkpoint_version, Some(LEGACY_STORAGE_VERSION));
        assert_eq!(legacy.generation, Some(ahead));
        assert!(
            status.log_store.is_none(),
            "the ACTIVE store is not repeated as an 'other' one"
        );
    }

    #[test]
    fn downgrade_refuses_when_the_v2_store_is_ahead_of_the_v3_one() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let v3_generation = publish(dir.path(), &guard, &mut index, initial.generation, "v3.png");
        drop(guard);

        // An older binary kept publishing into the version-2 store, which the
        // upgrade deliberately left intact — and got further than the v3 one.
        let v2_dir = legacy_authority_dir(dir.path());
        fs::create_dir_all(&v2_dir).unwrap();
        let ahead = v3_generation + 50;
        let mut v2_snapshot = empty_snapshot(ahead);
        v2_snapshot.version = LEGACY_STORAGE_VERSION;
        v2_snapshot
            .index
            .quarantined_names
            .insert("only-in-v2.png".into());
        let (envelope, _) = serialize_envelope_at(&v2_snapshot, LEGACY_STORAGE_VERSION).unwrap();
        atomic_write_bytes(&v2_dir.join(CHECKPOINT_FILE), &envelope).unwrap();
        atomic_write_bytes(&v2_dir.join(BACKUP_CHECKPOINT_FILE), &envelope).unwrap();
        atomic_write_json(
            &v2_dir.join(MARKER_FILE),
            &MutationMarker {
                version: LEGACY_STORAGE_VERSION,
                committed_generation: ahead,
                pending: None,
            },
        )
        .unwrap();
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());

        let error = downgrade_to_legacy_storage(dir.path())
            .expect_err("a downgrade that would lose prints must refuse");
        let message = format!("{error:#}");
        assert!(
            message.contains(&ahead.to_string()) && message.contains(&v3_generation.to_string()),
            "the refusal must name BOTH generations so an operator can tell              which store is ahead: {message}"
        );

        // And it must have changed nothing: the v2 store still holds its own
        // print at its own generation.
        let stored = read_checkpoint_at(&v2_dir.join(CHECKPOINT_FILE)).unwrap();
        assert_eq!(stored.generation, ahead);
        assert!(stored.index.quarantined_names.contains("only-in-v2.png"));
        assert!(
            authority_dir_v3(dir.path()).is_dir(),
            "the v3 store must survive a refused downgrade too"
        );
    }

    /// A v2 store BEHIND the v3 one is the ordinary case the command exists
    /// for, and it still folds forward.
    #[test]
    fn downgrade_still_folds_forward_over_an_older_v2_store() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..3_u64 {
            generation = publish(
                dir.path(),
                &guard,
                &mut index,
                generation,
                &format!("p{step}.png"),
            );
        }
        drop(guard);

        // A stale v2 store, strictly behind.
        let v2_dir = legacy_authority_dir(dir.path());
        fs::create_dir_all(&v2_dir).unwrap();
        let mut stale = empty_snapshot(0);
        stale.version = LEGACY_STORAGE_VERSION;
        let (envelope, _) = serialize_envelope_at(&stale, LEGACY_STORAGE_VERSION).unwrap();
        atomic_write_bytes(&v2_dir.join(CHECKPOINT_FILE), &envelope).unwrap();
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());

        let outcome = downgrade_to_legacy_storage(dir.path()).unwrap();
        assert_eq!(outcome.generation, generation);
        let stored = read_checkpoint_at(&v2_dir.join(CHECKPOINT_FILE)).unwrap();
        assert_eq!(stored.version, LEGACY_STORAGE_VERSION);
        assert_eq!(stored.generation, generation);
    }

    #[test]
    fn downgrade_refuses_a_pending_mutation_or_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let generation = publish(dir.path(), &guard, &mut index, initial.generation, "p.png");
        drop(guard);

        // A torn append: the log's last record is half written.
        let log = fs::read(mutation_log_path(dir.path())).unwrap();
        let file = OpenOptions::new()
            .write(true)
            .open(mutation_log_path(dir.path()))
            .unwrap();
        file.set_len(log.len() as u64 - 8).unwrap();
        drop(file);
        let error = downgrade_to_legacy_storage(dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("torn"), "unexpected error: {error}");
        assert!(
            error.contains("mold serve"),
            "the operator must be told what resolves it: {error}"
        );

        // A pending v2 mutation is the other refusal.
        fs::write(mutation_log_path(dir.path()), &log).unwrap();
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: generation,
                pending: Some(PendingMutation {
                    generation: generation + 1,
                    kind: "publish_batch".into(),
                    exact_names: vec!["x.png".into()],
                    snapshot_sha256: "0".repeat(64),
                }),
            },
        )
        .unwrap();
        let error = downgrade_to_legacy_storage(dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("pending"), "unexpected error: {error}");
    }

    #[test]
    fn a_v3_commit_appends_one_delta_and_leaves_the_checkpoint_alone() {
        // v2 wrote the whole archive index three times per commit — WAL,
        // checkpoint, backup — so publishing one print cost bytes
        // proportional to the size of the library. v3 appends a delta and
        // moves the marker: three fsyncs, and the checkpoint stays where the
        // last compaction left it.
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..3_u64 {
            generation = publish(
                dir.path(),
                &guard,
                &mut index,
                generation,
                &format!("p{step}.png"),
            );
        }
        assert_eq!(generation, initial.generation + 3);
        assert_eq!(
            read_checkpoint_at(&checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            initial.generation,
            "the checkpoint stays at the last compaction"
        );
        let MutationLogScan {
            records, discarded, ..
        } = read_mutation_log(dir.path(), initial.generation).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(discarded, 0);
        assert!(
            !wal_path(dir.path()).exists(),
            "v3 has no whole-snapshot WAL"
        );

        // And a cold process reads the same authority back.
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let reloaded =
            load_or_initialize_with_authority_log(dir.path(), &guard, true, || unreachable!())
                .unwrap();
        assert_eq!(reloaded.generation, generation);
        for step in 0..3_u64 {
            assert!(reloaded
                .index
                .quarantined_names
                .contains(&format!("p{step}.png")));
        }
    }

    #[test]
    fn recovery_replays_log_and_discards_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..3_u64 {
            generation = publish(
                dir.path(),
                &guard,
                &mut index,
                generation,
                &format!("p{step}.png"),
            );
        }

        // A crash mid-append: the last record is half written. It must be
        // dropped whole, and the two before it must survive.
        let log = fs::read(mutation_log_path(dir.path())).unwrap();
        let second_record_end = log
            .iter()
            .enumerate()
            .filter(|(_, byte)| **byte == b'\n')
            .map(|(index, _)| index + 1)
            .nth(1)
            .expect("three records");
        let torn_len = second_record_end + (log.len() - second_record_end) / 2;
        let file = OpenOptions::new()
            .write(true)
            .open(mutation_log_path(dir.path()))
            .unwrap();
        file.set_len(torn_len as u64).unwrap();
        drop(file);
        // The marker still names the generation the torn record would have
        // committed; recovery must answer with what it can actually reach.
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let error =
            load_or_initialize_with_authority_log(dir.path(), &guard, true, || unreachable!())
                .unwrap_err();
        assert!(
            error.to_string().contains("mid-log corruption"),
            "unexpected error: {error:#}"
        );
        assert!(
            mutation_log_path(dir.path()).exists(),
            "the log must still be on disk for a manual repair"
        );

        // With the marker corrected to what the log actually holds — which is
        // what a crash BEFORE the marker write leaves behind — recovery
        // replays the intact prefix and truncates the tear.
        write_marker(
            dir.path(),
            &MutationMarker {
                version: STORAGE_VERSION,
                committed_generation: initial.generation + 2,
                pending: None,
            },
        )
        .unwrap();
        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let recovered =
            load_or_initialize_with_authority_log(dir.path(), &guard, true, || unreachable!())
                .unwrap();
        assert_eq!(recovered.generation, initial.generation + 2);
        assert!(recovered.index.quarantined_names.contains("p0.png"));
        assert!(recovered.index.quarantined_names.contains("p1.png"));
        assert!(
            !recovered.index.quarantined_names.contains("p2.png"),
            "a torn record commits nothing"
        );
        assert!(
            !mutation_log_path(dir.path()).exists(),
            "recovery compacts, so the log starts empty again"
        );
    }

    #[test]
    fn read_only_load_sees_log_tail_generation() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let generation = publish(
            dir.path(),
            &guard,
            &mut index,
            initial.generation,
            "only.png",
        );

        forget_authority_tail(&fs::canonicalize(dir.path()).unwrap());
        let loaded = load_existing_read_only(dir.path(), &guard)
            .unwrap()
            .expect("an established authority");
        assert_eq!(loaded.generation, generation);
        assert!(loaded.index.quarantined_names.contains("only.png"));
        assert_eq!(
            read_generation(dir.path(), &guard).unwrap(),
            Some(generation)
        );
        // The read path never repairs: it left the log exactly as it found it.
        assert!(mutation_log_path(dir.path()).exists());
    }

    #[test]
    fn compaction_truncates_log_and_refreshes_backups() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial = load_or_initialize_with_authority_log(dir.path(), &guard, true, || {
            Ok(CommittedArchiveIndex::default())
        })
        .unwrap();
        let mut index = initial.index;
        let mut generation = initial.generation;
        for step in 0..MUTATION_LOG_MAX_RECORDS as u64 {
            generation = publish(
                dir.path(),
                &guard,
                &mut index,
                generation,
                &format!("c{step}.png"),
            );
        }
        assert_eq!(
            read_checkpoint_at(&checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            generation,
            "the record threshold folds the log into a fresh checkpoint"
        );
        assert_eq!(
            read_checkpoint_at(&backup_checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            generation,
            "and the same-generation backup follows it"
        );
        assert!(
            !mutation_log_path(dir.path()).exists(),
            "the log starts again from empty"
        );
        assert_eq!(
            read_checkpoint_at(&previous_checkpoint_path(dir.path()))
                .unwrap()
                .generation,
            initial.generation,
            "and the prior checkpoint is kept"
        );
    }

    #[test]
    fn v2_store_upgrades_once_then_uses_v3() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        // Build a v2 store by hand: a v2-versioned checkpoint and marker,
        // which is exactly what an older binary left behind.
        let mut snapshot = empty_snapshot(4);
        snapshot.version = LEGACY_STORAGE_VERSION;
        snapshot.index.quarantined_names.insert("legacy.png".into());
        let payload = serde_json::to_vec(&snapshot).unwrap();
        let digest = format!("{:x}", Sha256::digest(&payload));
        let mut bytes = format!(
            r#"{{"version":{LEGACY_STORAGE_VERSION},"payload_sha256":"{digest}","snapshot":"#
        )
        .into_bytes();
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(b"}\n");
        fs::create_dir_all(authority_dir(dir.path())).unwrap();
        atomic_write_bytes(&checkpoint_path(dir.path()), &bytes).unwrap();
        write_marker(
            dir.path(),
            &MutationMarker {
                version: LEGACY_STORAGE_VERSION,
                committed_generation: 4,
                pending: None,
            },
        )
        .unwrap();

        let loaded =
            load_or_initialize_with_authority_log(dir.path(), &guard, true, || unreachable!())
                .unwrap();
        assert_eq!(loaded.generation, 4);
        assert!(loaded.index.quarantined_names.contains("legacy.png"));
        // Read once, upgraded once.
        assert_eq!(
            read_checkpoint_at(&checkpoint_path(dir.path()))
                .unwrap()
                .version,
            STORAGE_VERSION
        );

        // And from here on it writes deltas.
        let mut index = loaded.index;
        let generation = publish(
            dir.path(),
            &guard,
            &mut index,
            loaded.generation,
            "after.png",
        );
        assert_eq!(generation, 5);
        let records = read_mutation_log(dir.path(), 4).unwrap().records;
        assert_eq!(records.len(), 1);
    }

    fn archive_entry(name: &str, checksum: &str) -> CommittedArchiveEntry {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q8",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0
        }))
        .unwrap();
        CommittedArchiveEntry {
            identity: ArchivedChildIdentity {
                parent_id: format!("parent-{name}"),
                attempt_generation: 0,
                child_index: 0,
                final_name: name.to_string(),
                checksum_sha256: checksum.to_string(),
                size_bytes: 6,
            },
            record: GenerationRecord::from_save(
                Path::new("/tmp"),
                name,
                OutputFormat::Png,
                OutputMetadata::from_generate_request(&request, 1, None, "test"),
                RecordSource::Server,
                1,
            ),
            facts: None,
            retained_media: Vec::new(),
        }
    }

    /// The one contract a delta rests on: a mutation that edits an ENTRY in
    /// place must name it in `exact_names`.
    ///
    /// Adds and removes come from the key sets, so they need no name. An
    /// in-place edit does — comparing values would mean serializing every
    /// entry, which is the cost the log exists to remove — and this asserts
    /// BOTH directions, because a test that only exercises the name-free
    /// sets passes with the `touched` argument deleted entirely.
    #[test]
    fn an_in_place_edit_must_name_the_entry_it_changed() {
        let mut before = empty_snapshot(1);
        before
            .index
            .entries
            .insert("kept.png".into(), archive_entry("kept.png", "aa"));
        before
            .index
            .entries
            .insert("edited.png".into(), archive_entry("edited.png", "bb"));

        let mut after = before.clone();
        after.generation = 2;
        // An add (no name needed), a remove (no name needed), and an in-place
        // edit of an entry that IS named.
        after
            .index
            .entries
            .insert("added.png".into(), archive_entry("added.png", "cc"));
        after.index.entries.remove("kept.png");
        after
            .index
            .entries
            .insert("edited.png".into(), archive_entry("edited.png", "dd"));

        let named = std::collections::BTreeSet::from(["edited.png"]);
        let delta = IndexDelta::between(
            (
                &AuthorityIndexKeys::of(&before.index),
                &before.legacy_evidence_epochs,
            ),
            (&after.index, &after.legacy_evidence_epochs),
            &named,
        );
        assert!(delta.entries_set.contains_key("added.png"), "an add rides");
        assert!(delta.entries_removed.contains("kept.png"), "a remove rides");
        assert_eq!(
            delta.entries_set["edited.png"].identity.checksum_sha256, "dd",
            "a NAMED in-place edit rides"
        );
        let mut replayed = before.clone();
        delta.apply(&mut replayed);
        assert_eq!(
            replayed.index.entries["edited.png"]
                .identity
                .checksum_sha256,
            "dd"
        );
        assert!(!replayed.index.entries.contains_key("kept.png"));
        assert!(replayed.index.entries.contains_key("added.png"));

        // And the other direction: the same edit, NOT named, is dropped. This
        // is what the `touched` argument buys, and what a caller that forgets
        // `exact_names` loses until the next compaction.
        let unnamed = std::collections::BTreeSet::new();
        let delta = IndexDelta::between(
            (
                &AuthorityIndexKeys::of(&before.index),
                &before.legacy_evidence_epochs,
            ),
            (&after.index, &after.legacy_evidence_epochs),
            &unnamed,
        );
        assert!(
            !delta.entries_set.contains_key("edited.png"),
            "an unnamed in-place edit is invisible to the delta — the contract"
        );
        let mut replayed = before.clone();
        delta.apply(&mut replayed);
        assert_eq!(
            replayed.index.entries["edited.png"]
                .identity
                .checksum_sha256,
            "bb",
            "and replays as the value it had before"
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

    /// The fast path's on-disk guards, exercised with a WARM tail.
    ///
    /// `cached_commit_tail` may only be trusted while the marker still names
    /// the generation this process last wrote and no log or WAL is
    /// outstanding. Each half below leaves the cached tail in place and moves
    /// exactly one of those facts, so the refusal can only come from the
    /// guard under test — and the checkpoint-parse counter says which path
    /// actually ran.
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

        // The tail is warm here: a commit at the same generation takes the
        // fast path and parses nothing.
        reset_checkpoint_parse_count();
        let mut probe = index.clone();
        let next = commit_snapshot(
            dir.path(),
            &guard,
            generation,
            &mut probe,
            "publish_batch",
            Vec::new(),
        )
        .unwrap();
        assert_eq!(next, generation + 1);
        assert_eq!(
            checkpoint_parse_count(),
            0,
            "a warm tail at the marker's generation reads no checkpoint"
        );

        // Somebody else advanced the authority. The cached tail still says
        // `next`, so ONLY the marker guard can catch this.
        let mut ahead = empty_snapshot(next + 5);
        ahead.index = probe.clone();
        write_checkpoint_for_test(dir.path(), &ahead);
        write_marker(
            dir.path(),
            &MutationMarker {
                version: LEGACY_STORAGE_VERSION,
                committed_generation: next + 5,
                pending: None,
            },
        )
        .unwrap();
        reset_checkpoint_parse_count();
        let error = commit_snapshot(
            dir.path(),
            &guard,
            next,
            &mut probe.clone(),
            "publish_batch",
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("generation changed"),
            "unexpected error: {error:#}"
        );
        assert!(
            checkpoint_parse_count() > 0,
            "a marker that disagrees with the cached tail must force the full recovery read"
        );

        // And an unresolved WAL at the cached generation drops the fast path
        // too: rolling one forward is recovery's job, not a commit's.
        let mut restored = empty_snapshot(next);
        restored.index = probe.clone();
        write_checkpoint_for_test(dir.path(), &restored);
        write_marker(
            dir.path(),
            &MutationMarker {
                version: LEGACY_STORAGE_VERSION,
                committed_generation: next,
                pending: None,
            },
        )
        .unwrap();
        let mut rolled = empty_snapshot(next + 1);
        rolled.index = probe.clone();
        rolled.index.quarantined_names.insert("from-wal.png".into());
        atomic_write_json(&wal_path(dir.path()), &wrap_snapshot(rolled).unwrap()).unwrap();
        // Warm the tail back to `next` so the WAL is the only thing the fast
        // path could trip on.
        forget_authority_tail(guard.canonical_root());
        let reloaded = load_or_initialize(dir.path(), &guard, || unreachable!()).unwrap();
        assert_eq!(
            reloaded.generation,
            next + 1,
            "recovery rolled the WAL forward"
        );
        assert!(reloaded.index.quarantined_names.contains("from-wal.png"));
    }

    /// The v2 protocol's checkpoint rotation. v3 writes neither file per
    /// commit — the checkpoint and its backup are refreshed at compaction —
    /// so this pins the path a process takes before its startup recovery has
    /// run for a root.
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
        disable_v3_writing_for_test(guard.canonical_root());
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

    /// The v2 protocol's completed-mutation shape.
    #[test]
    fn completed_mutation_keeps_current_backup_and_immediately_prior_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let guard = acquire_gallery_bookkeeping_lock(dir.path()).unwrap();
        let initial =
            load_or_initialize(dir.path(), &guard, || Ok(CommittedArchiveIndex::default()))
                .unwrap();
        disable_v3_writing_for_test(guard.canonical_root());
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
