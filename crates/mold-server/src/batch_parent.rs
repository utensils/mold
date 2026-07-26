//! Serialized state reducer for one server-owned batch parent.
//!
//! The reducer is deliberately free of filesystem, SQLite, scheduler, and
//! async concerns. Callers serialize access with the parent actor lock and
//! execute returned cleanup/publication actions outside that lock.

use anyhow::Context as _;
use mold_inference::InferenceCancellationToken;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchParentState {
    Queued,
    Running,
    Prepared,
    Committing,
    Committed,
    Cancelling,
    Failing,
    Fenced,
    Retrying,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChildState {
    Pending,
    Active,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchChildLease {
    pub parent_id: String,
    pub child_index: usize,
    pub attempt_generation: u64,
    pub lease_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionDisposition {
    Accepted,
    RetryChild,
    AttemptPrepared,
    AttemptFenced,
    AttemptFencedDeletePrivateArtifact,
    ClosedAttemptDeletePrivateArtifact,
    StaleDeletePrivateArtifact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildCompletion {
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct BatchParentReducer {
    parent_id: String,
    total_children: usize,
    attempt_generation: u64,
    state: BatchParentState,
    children: Vec<ChildState>,
    child_lease_generations: Vec<u64>,
    active: BTreeSet<usize>,
    cancellation_tokens: BTreeMap<usize, InferenceCancellationToken>,
    retry_counts: Vec<u8>,
    terminal_after_fence: Option<BatchParentState>,
}

const PARENT_SNAPSHOT_FILE: &str = "parent-state.json";
const PARENT_JOURNAL_FILE: &str = "parent-journal.jsonl";
const PARENT_JOURNAL_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct BatchParentSnapshot {
    version: u32,
    parent_id: String,
    total_children: usize,
    attempt_generation: u64,
    state: BatchParentState,
    children: Vec<ChildState>,
    child_lease_generations: Vec<u64>,
    active: BTreeSet<usize>,
    retry_counts: Vec<u8>,
    terminal_after_fence: Option<BatchParentState>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchParentJournalRecord {
    sequence: u64,
    attempt_generation: u64,
    from: BatchParentState,
    to: BatchParentState,
    snapshot: BatchParentSnapshot,
}

/// Durable, serialized authority for one batch parent's completion reducer.
///
/// Every mutating operation is reduced under `&mut self`, then the resulting
/// full snapshot is atomically replaced and appended to the fsynced journal
/// before the caller receives the disposition. F1 owns the actor/lock that
/// serializes access to this type.
#[derive(Debug)]
pub struct DurableBatchParent {
    reducer: BatchParentReducer,
    directory: PathBuf,
    next_sequence: u64,
    poisoned: bool,
}

impl DurableBatchParent {
    pub fn create(
        directory: &Path,
        parent_id: impl Into<String>,
        total_children: usize,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(directory)?;
        sync_parent_dir(directory)?;
        let reducer = BatchParentReducer::new(parent_id, total_children)?;
        let mut durable = Self {
            reducer,
            directory: directory.to_path_buf(),
            next_sequence: 0,
            poisoned: false,
        };
        durable.persist_transition(BatchParentState::Queued)?;
        Ok(durable)
    }

    pub fn recover(directory: &Path) -> anyhow::Result<Self> {
        let records = load_parent_journal(&directory.join(PARENT_JOURNAL_FILE))?;
        let journal_snapshot = records.last().map(|record| record.snapshot.clone());
        let disk_snapshot = std::fs::read(directory.join(PARENT_SNAPSHOT_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<BatchParentSnapshot>(&bytes).ok());
        let reconstructed_snapshot = disk_snapshot.is_none();
        let snapshot = disk_snapshot
            .clone()
            .or_else(|| journal_snapshot.clone())
            .with_context(|| {
                format!(
                    "batch parent has no recoverable state in {}",
                    directory.display()
                )
            })?;
        anyhow::ensure!(
            snapshot.version == PARENT_JOURNAL_VERSION,
            "unsupported batch parent journal version {}",
            snapshot.version
        );
        validate_parent_snapshot(&snapshot)?;
        let journal_needs_heal = journal_snapshot.as_ref() != Some(&snapshot);
        let journal_state = records
            .last()
            .map_or(snapshot.state, |record| record.snapshot.state);
        let reducer = BatchParentReducer {
            parent_id: snapshot.parent_id,
            total_children: snapshot.total_children,
            attempt_generation: snapshot.attempt_generation,
            state: snapshot.state,
            children: snapshot.children,
            child_lease_generations: snapshot.child_lease_generations,
            active: snapshot.active,
            cancellation_tokens: BTreeMap::new(),
            retry_counts: snapshot.retry_counts,
            terminal_after_fence: snapshot.terminal_after_fence,
        };
        let mut durable = Self {
            reducer,
            directory: directory.to_path_buf(),
            next_sequence: records
                .last()
                .map_or(0, |record| record.sequence.saturating_add(1)),
            poisoned: false,
        };
        if reconstructed_snapshot {
            atomic_write_parent_json(
                &durable.directory.join(PARENT_SNAPSHOT_FILE),
                &durable.reducer.snapshot(),
            )?;
        } else if journal_needs_heal {
            durable.persist_transition(journal_state)?;
        }
        if !durable.reducer.active.is_empty() {
            let from = durable.reducer.state;
            for child_index in std::mem::take(&mut durable.reducer.active) {
                durable.reducer.children[child_index] = ChildState::Cancelled;
            }
            durable.reducer.cancellation_tokens.clear();
            durable.reducer.terminal_after_fence =
                Some(if durable.reducer.state == BatchParentState::Cancelling {
                    BatchParentState::Cancelled
                } else {
                    BatchParentState::Failed
                });
            durable.reducer.state = BatchParentState::Fenced;
            durable.persist_transition(from)?;
        }
        Ok(durable)
    }

    pub fn state(&self) -> BatchParentState {
        self.reducer.state()
    }

    pub fn attempt_generation(&self) -> u64 {
        self.reducer.attempt_generation()
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        self.mutate(|reducer| reducer.start())
    }

    pub fn grant(
        &mut self,
        child_index: usize,
    ) -> anyhow::Result<(BatchChildLease, InferenceCancellationToken)> {
        let lease = self.mutate(|reducer| reducer.grant(child_index))?;
        let token = self
            .reducer
            .cancellation_token(&lease)
            .context("durable child grant lost its cancellation token")?;
        Ok((lease, token))
    }

    pub fn complete(
        &mut self,
        lease: &BatchChildLease,
        completion: ChildCompletion,
    ) -> anyhow::Result<CompletionDisposition> {
        self.mutate(|reducer| reducer.complete(lease, completion))
    }

    pub fn request_cancel(&mut self) -> anyhow::Result<CompletionDisposition> {
        self.mutate(BatchParentReducer::request_cancel)
    }

    pub fn begin_retry(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::begin_retry)
    }

    pub fn finalize_fence(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::finalize_fence)
    }

    pub fn begin_commit(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::begin_commit)
    }

    pub fn mark_committed(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::mark_committed)
    }

    fn mutate<T>(
        &mut self,
        operation: impl FnOnce(&mut BatchParentReducer) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        anyhow::ensure!(
            !self.poisoned,
            "batch parent durable authority is poisoned after a persistence failure"
        );
        let from = self.reducer.state;
        let result = operation(&mut self.reducer)?;
        if let Err(error) = self.persist_transition(from) {
            self.poisoned = true;
            return Err(error).context(
                "persisting batch parent transition; this authority must be retired and recovered",
            );
        }
        Ok(result)
    }

    fn persist_transition(&mut self, from: BatchParentState) -> anyhow::Result<()> {
        let snapshot = self.reducer.snapshot();
        atomic_write_parent_json(&self.directory.join(PARENT_SNAPSHOT_FILE), &snapshot)?;
        let record = BatchParentJournalRecord {
            sequence: self.next_sequence,
            attempt_generation: snapshot.attempt_generation,
            from,
            to: snapshot.state,
            snapshot,
        };
        let mut bytes = serde_json::to_vec(&record)?;
        bytes.push(b'\n');
        let mut journal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.directory.join(PARENT_JOURNAL_FILE))?;
        journal.write_all(&bytes)?;
        journal.sync_all()?;
        sync_parent_dir(&self.directory)?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .context("batch parent journal sequence overflow")?;
        Ok(())
    }
}

impl BatchParentReducer {
    pub fn new(parent_id: impl Into<String>, total_children: usize) -> anyhow::Result<Self> {
        let parent_id = parent_id.into();
        let parent_path = Path::new(&parent_id);
        anyhow::ensure!(
            !parent_id.is_empty()
                && parent_path.file_name().and_then(|name| name.to_str())
                    == Some(parent_id.as_str())
                && parent_id != "."
                && parent_id != "..",
            "batch parent id must be one path component"
        );
        anyhow::ensure!(
            total_children > 0,
            "batch parent must have at least one child"
        );
        Ok(Self {
            parent_id,
            total_children,
            attempt_generation: 0,
            state: BatchParentState::Queued,
            children: vec![ChildState::Pending; total_children],
            child_lease_generations: vec![0; total_children],
            active: BTreeSet::new(),
            cancellation_tokens: BTreeMap::new(),
            retry_counts: vec![0; total_children],
            terminal_after_fence: None,
        })
    }

    pub fn state(&self) -> BatchParentState {
        self.state
    }

    pub fn attempt_generation(&self) -> u64 {
        self.attempt_generation
    }

    fn snapshot(&self) -> BatchParentSnapshot {
        BatchParentSnapshot {
            version: PARENT_JOURNAL_VERSION,
            parent_id: self.parent_id.clone(),
            total_children: self.total_children,
            attempt_generation: self.attempt_generation,
            state: self.state,
            children: self.children.clone(),
            child_lease_generations: self.child_lease_generations.clone(),
            active: self.active.clone(),
            retry_counts: self.retry_counts.clone(),
            terminal_after_fence: self.terminal_after_fence,
        }
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(
                self.state,
                BatchParentState::Queued | BatchParentState::Retrying
            ),
            "batch parent cannot start from {:?}",
            self.state
        );
        self.state = BatchParentState::Running;
        Ok(())
    }

    pub fn grant(&mut self, child_index: usize) -> anyhow::Result<BatchChildLease> {
        anyhow::ensure!(
            self.state == BatchParentState::Running,
            "batch parent is closed to new grants in {:?}",
            self.state
        );
        anyhow::ensure!(
            child_index < self.total_children,
            "batch child index {child_index} is out of range"
        );
        anyhow::ensure!(
            self.children[child_index] == ChildState::Pending,
            "batch child {child_index} is not pending"
        );
        anyhow::ensure!(
            self.active.insert(child_index),
            "batch child {child_index} already has an active lease"
        );
        self.children[child_index] = ChildState::Active;
        self.child_lease_generations[child_index] = self.child_lease_generations[child_index]
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("batch child lease generation overflow"))?;
        self.cancellation_tokens
            .insert(child_index, InferenceCancellationToken::default());
        Ok(BatchChildLease {
            parent_id: self.parent_id.clone(),
            child_index,
            attempt_generation: self.attempt_generation,
            lease_generation: self.child_lease_generations[child_index],
        })
    }

    pub fn cancellation_token(
        &self,
        lease: &BatchChildLease,
    ) -> Option<InferenceCancellationToken> {
        (lease.parent_id == self.parent_id
            && lease.attempt_generation == self.attempt_generation
            && self.child_lease_generations.get(lease.child_index) == Some(&lease.lease_generation))
        .then(|| self.cancellation_tokens.get(&lease.child_index).cloned())
        .flatten()
    }

    pub fn complete(
        &mut self,
        lease: &BatchChildLease,
        completion: ChildCompletion,
    ) -> anyhow::Result<CompletionDisposition> {
        anyhow::ensure!(
            lease.parent_id == self.parent_id,
            "completion parent does not match reducer"
        );
        anyhow::ensure!(
            lease.child_index < self.total_children,
            "batch child index {} is out of range",
            lease.child_index
        );
        if lease.attempt_generation < self.attempt_generation {
            return Ok(CompletionDisposition::StaleDeletePrivateArtifact);
        }
        anyhow::ensure!(
            lease.attempt_generation == self.attempt_generation,
            "completion generation {} is ahead of current generation {}",
            lease.attempt_generation,
            self.attempt_generation
        );
        if lease.lease_generation
            < *self
                .child_lease_generations
                .get(lease.child_index)
                .context("batch child lease generation is missing")?
        {
            return Ok(CompletionDisposition::StaleDeletePrivateArtifact);
        }
        anyhow::ensure!(
            lease.lease_generation == self.child_lease_generations[lease.child_index],
            "completion lease generation {} is ahead of current lease generation {}",
            lease.lease_generation,
            self.child_lease_generations[lease.child_index]
        );
        anyhow::ensure!(
            self.children[lease.child_index] == ChildState::Active
                && self.active.remove(&lease.child_index),
            "batch child {} has no active lease in generation {}",
            lease.child_index,
            lease.attempt_generation
        );
        self.cancellation_tokens.remove(&lease.child_index);

        match self.state {
            BatchParentState::Running => self.complete_running(lease.child_index, completion),
            BatchParentState::Cancelling | BatchParentState::Failing => {
                self.complete_closed_attempt(lease.child_index, completion)
            }
            _ => anyhow::bail!(
                "batch child completion is invalid while parent is {:?}",
                self.state
            ),
        }
    }

    pub fn request_cancel(&mut self) -> anyhow::Result<CompletionDisposition> {
        anyhow::ensure!(
            matches!(
                self.state,
                BatchParentState::Queued | BatchParentState::Running | BatchParentState::Prepared
            ),
            "batch parent cannot cancel from {:?}",
            self.state
        );
        self.state = BatchParentState::Cancelling;
        self.terminal_after_fence = Some(BatchParentState::Cancelled);
        self.cancel_active_children();
        if self.active.is_empty() {
            self.state = BatchParentState::Fenced;
            Ok(CompletionDisposition::AttemptFenced)
        } else {
            Ok(CompletionDisposition::Accepted)
        }
    }

    pub fn begin_retry(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Fenced
                && self.terminal_after_fence == Some(BatchParentState::Failed),
            "only a fenced failed attempt can retry"
        );
        self.attempt_generation = self
            .attempt_generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("batch attempt generation overflow"))?;
        self.children.fill(ChildState::Pending);
        self.child_lease_generations.fill(0);
        self.active.clear();
        self.cancellation_tokens.clear();
        self.retry_counts.fill(0);
        self.terminal_after_fence = None;
        self.state = BatchParentState::Retrying;
        Ok(())
    }

    pub fn finalize_fence(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Fenced,
            "batch parent is not fenced"
        );
        let terminal = self
            .terminal_after_fence
            .take()
            .ok_or_else(|| anyhow::anyhow!("fenced parent has no terminal disposition"))?;
        anyhow::ensure!(
            matches!(
                terminal,
                BatchParentState::Cancelled | BatchParentState::Failed
            ),
            "invalid terminal state after fence: {terminal:?}"
        );
        self.state = terminal;
        Ok(())
    }

    pub fn begin_commit(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Prepared,
            "batch parent cannot commit from {:?}",
            self.state
        );
        self.state = BatchParentState::Committing;
        Ok(())
    }

    pub fn mark_committed(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Committing,
            "batch parent cannot finish commit from {:?}",
            self.state
        );
        self.state = BatchParentState::Committed;
        Ok(())
    }

    fn complete_running(
        &mut self,
        child_index: usize,
        completion: ChildCompletion,
    ) -> anyhow::Result<CompletionDisposition> {
        match completion {
            ChildCompletion::Succeeded => {
                self.children[child_index] = ChildState::Succeeded;
                if self
                    .children
                    .iter()
                    .all(|state| *state == ChildState::Succeeded)
                {
                    anyhow::ensure!(
                        self.active.is_empty(),
                        "all children succeeded while active leases remain"
                    );
                    self.state = BatchParentState::Prepared;
                    Ok(CompletionDisposition::AttemptPrepared)
                } else {
                    Ok(CompletionDisposition::Accepted)
                }
            }
            ChildCompletion::Failed | ChildCompletion::Cancelled => {
                self.children[child_index] = match completion {
                    ChildCompletion::Failed => ChildState::Failed,
                    ChildCompletion::Cancelled => ChildState::Cancelled,
                    ChildCompletion::Succeeded => unreachable!(),
                };
                if completion == ChildCompletion::Failed && self.retry_counts[child_index] == 0 {
                    self.retry_counts[child_index] = 1;
                    self.children[child_index] = ChildState::Pending;
                    return Ok(CompletionDisposition::RetryChild);
                }
                self.state = BatchParentState::Failing;
                self.terminal_after_fence = Some(BatchParentState::Failed);
                self.cancel_active_children();
                if self.active.is_empty() {
                    self.state = BatchParentState::Fenced;
                    Ok(CompletionDisposition::AttemptFenced)
                } else {
                    Ok(CompletionDisposition::Accepted)
                }
            }
        }
    }

    fn complete_closed_attempt(
        &mut self,
        child_index: usize,
        completion: ChildCompletion,
    ) -> anyhow::Result<CompletionDisposition> {
        let has_private_artifact = completion == ChildCompletion::Succeeded;
        self.children[child_index] = ChildState::Cancelled;
        if self.active.is_empty() {
            self.state = BatchParentState::Fenced;
            Ok(if has_private_artifact {
                CompletionDisposition::AttemptFencedDeletePrivateArtifact
            } else {
                CompletionDisposition::AttemptFenced
            })
        } else {
            Ok(if has_private_artifact {
                CompletionDisposition::ClosedAttemptDeletePrivateArtifact
            } else {
                CompletionDisposition::Accepted
            })
        }
    }

    fn cancel_active_children(&self) {
        for token in self.cancellation_tokens.values() {
            token.cancel();
        }
    }
}

fn validate_parent_snapshot(snapshot: &BatchParentSnapshot) -> anyhow::Result<()> {
    let parent_path = Path::new(&snapshot.parent_id);
    anyhow::ensure!(
        !snapshot.parent_id.is_empty()
            && parent_path.file_name().and_then(|name| name.to_str())
                == Some(snapshot.parent_id.as_str())
            && snapshot.parent_id != "."
            && snapshot.parent_id != "..",
        "batch parent snapshot has an invalid parent id"
    );
    anyhow::ensure!(snapshot.total_children > 0, "batch parent has no children");
    anyhow::ensure!(
        snapshot.children.len() == snapshot.total_children
            && snapshot.child_lease_generations.len() == snapshot.total_children
            && snapshot.retry_counts.len() == snapshot.total_children,
        "batch parent snapshot child vectors do not match total_children"
    );
    anyhow::ensure!(
        snapshot
            .active
            .iter()
            .all(|index| *index < snapshot.total_children),
        "batch parent snapshot contains an out-of-range active child"
    );
    anyhow::ensure!(
        snapshot.children.iter().enumerate().all(
            |(index, state)| (*state == ChildState::Active) == snapshot.active.contains(&index)
        ),
        "batch parent snapshot active set does not match child states"
    );
    let expected_terminal = match snapshot.state {
        BatchParentState::Cancelling => Some(BatchParentState::Cancelled),
        BatchParentState::Failing => Some(BatchParentState::Failed),
        BatchParentState::Fenced => snapshot.terminal_after_fence,
        _ => None,
    };
    anyhow::ensure!(
        snapshot.terminal_after_fence == expected_terminal
            && (!matches!(snapshot.state, BatchParentState::Fenced)
                || matches!(
                    snapshot.terminal_after_fence,
                    Some(BatchParentState::Cancelled | BatchParentState::Failed)
                )),
        "batch parent snapshot has an invalid terminal-after-fence state"
    );
    if matches!(
        snapshot.state,
        BatchParentState::Prepared | BatchParentState::Committing | BatchParentState::Committed
    ) {
        anyhow::ensure!(
            snapshot.active.is_empty()
                && snapshot
                    .children
                    .iter()
                    .all(|state| *state == ChildState::Succeeded),
            "prepared/committing/committed batch parent is not fully succeeded"
        );
    }
    Ok(())
}

fn load_parent_journal(path: &Path) -> anyhow::Result<Vec<BatchParentJournalRecord>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let has_incomplete_tail = !bytes.ends_with(b"\n");
    let lines: Vec<&[u8]> = bytes.split(|byte| *byte == b'\n').collect();
    let last_nonempty = lines.iter().rposition(|line| !line.is_empty());
    let mut records: Vec<BatchParentJournalRecord> = Vec::new();
    let mut offset = 0_u64;
    for (index, line) in lines.into_iter().enumerate() {
        if line.is_empty() {
            if offset < bytes.len() as u64 {
                offset += 1;
            }
            continue;
        }
        match serde_json::from_slice::<BatchParentJournalRecord>(line) {
            Ok(record) => {
                anyhow::ensure!(
                    record.sequence == records.len() as u64,
                    "batch parent journal sequence gap at {}",
                    path.display()
                );
                validate_parent_snapshot(&record.snapshot)?;
                anyhow::ensure!(
                    record.attempt_generation == record.snapshot.attempt_generation,
                    "batch parent journal generation does not match its snapshot at {}",
                    path.display()
                );
                anyhow::ensure!(
                    record.to == record.snapshot.state,
                    "batch parent journal target state does not match its snapshot at {}",
                    path.display()
                );
                if let Some(previous) = records.last() {
                    anyhow::ensure!(
                        record.from == previous.snapshot.state,
                        "batch parent journal transition chain is broken at {}",
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
                    "ignoring incomplete trailing batch parent journal record"
                );
                let file = OpenOptions::new().write(true).open(path)?;
                file.set_len(offset)?;
                file.sync_all()?;
                if let Some(parent) = path.parent() {
                    sync_parent_dir(parent)?;
                }
                break;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("parsing batch parent journal {}", path.display()));
            }
        }
    }
    Ok(records)
}

fn atomic_write_parent_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .context("batch parent state path has no parent")?;
    let temporary = parent.join(format!(
        ".{PARENT_SNAPSHOT_FILE}.tmp-{}",
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec_pretty(value)?)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)?;
        sync_parent_dir(parent)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> anyhow::Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running_parent(children: usize) -> BatchParentReducer {
        let mut parent = BatchParentReducer::new("parent", children).unwrap();
        parent.start().unwrap();
        parent
    }

    #[test]
    fn all_current_generation_children_prepare_exactly_once() {
        let mut parent = running_parent(2);
        let first = parent.grant(0).unwrap();
        let second = parent.grant(1).unwrap();

        assert_eq!(
            parent
                .complete(&second, ChildCompletion::Succeeded)
                .unwrap(),
            CompletionDisposition::Accepted
        );
        assert_eq!(
            parent.complete(&first, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::AttemptPrepared
        );
        assert_eq!(parent.state(), BatchParentState::Prepared);
        assert!(parent.complete(&first, ChildCompletion::Succeeded).is_err());
    }

    #[test]
    fn failure_fences_active_siblings_before_retry_and_late_completion_is_stale() {
        let mut parent = running_parent(2);
        let failed = parent.grant(0).unwrap();
        let sibling = parent.grant(1).unwrap();
        let sibling_token = parent.cancellation_token(&sibling).unwrap();

        assert_eq!(
            parent.complete(&failed, ChildCompletion::Failed).unwrap(),
            CompletionDisposition::RetryChild
        );
        assert_eq!(parent.state(), BatchParentState::Running);
        let child_retry = parent.grant(0).unwrap();
        assert_eq!(
            parent
                .complete(&child_retry, ChildCompletion::Failed)
                .unwrap(),
            CompletionDisposition::Accepted
        );
        assert_eq!(parent.state(), BatchParentState::Failing);
        assert!(sibling_token.is_cancelled());
        assert_eq!(
            parent
                .complete(&sibling, ChildCompletion::Cancelled)
                .unwrap(),
            CompletionDisposition::AttemptFenced
        );
        assert_eq!(parent.state(), BatchParentState::Fenced);

        parent.begin_retry().unwrap();
        assert_eq!(parent.state(), BatchParentState::Retrying);
        assert_eq!(parent.attempt_generation(), 1);
        parent.start().unwrap();
        let retry = parent.grant(1).unwrap();

        assert_eq!(
            parent
                .complete(&sibling, ChildCompletion::Succeeded)
                .unwrap(),
            CompletionDisposition::StaleDeletePrivateArtifact
        );
        assert_eq!(parent.state(), BatchParentState::Running);
        assert_eq!(
            parent.complete(&retry, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::Accepted
        );
    }

    #[test]
    fn cancel_racing_last_success_never_prepares_after_attempt_closes() {
        let mut parent = running_parent(1);
        let child = parent.grant(0).unwrap();
        let token = parent.cancellation_token(&child).unwrap();

        assert_eq!(
            parent.request_cancel().unwrap(),
            CompletionDisposition::Accepted
        );
        assert!(token.is_cancelled());
        assert_eq!(parent.state(), BatchParentState::Cancelling);
        assert_eq!(
            parent.complete(&child, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::AttemptFencedDeletePrivateArtifact
        );
        assert_eq!(parent.state(), BatchParentState::Fenced);
        parent.finalize_fence().unwrap();
        assert_eq!(parent.state(), BatchParentState::Cancelled);
    }

    #[test]
    fn prepared_attempt_commits_in_order_and_cannot_be_cancelled_after_commit_begins() {
        let mut parent = running_parent(1);
        let child = parent.grant(0).unwrap();
        assert_eq!(
            parent.complete(&child, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::AttemptPrepared
        );

        parent.begin_commit().unwrap();
        assert_eq!(parent.state(), BatchParentState::Committing);
        assert!(parent.request_cancel().is_err());
        parent.mark_committed().unwrap();
        assert_eq!(parent.state(), BatchParentState::Committed);
    }

    #[test]
    fn invalid_parent_index_or_generation_cannot_mutate_reducer() {
        let mut parent = running_parent(1);
        let lease = parent.grant(0).unwrap();
        for invalid in [
            BatchChildLease {
                parent_id: "other".into(),
                ..lease.clone()
            },
            BatchChildLease {
                child_index: 1,
                ..lease.clone()
            },
            BatchChildLease {
                attempt_generation: 1,
                ..lease.clone()
            },
        ] {
            assert!(parent
                .complete(&invalid, ChildCompletion::Succeeded)
                .is_err());
            assert_eq!(parent.state(), BatchParentState::Running);
        }
    }

    #[test]
    fn queued_parent_can_cancel_without_granting_children() {
        let mut parent = BatchParentReducer::new("parent", 2).unwrap();
        assert_eq!(
            parent.request_cancel().unwrap(),
            CompletionDisposition::AttemptFenced
        );
        parent.finalize_fence().unwrap();
        assert_eq!(parent.state(), BatchParentState::Cancelled);
    }

    #[test]
    fn retry_lease_generation_fences_late_completion_from_first_device() {
        let mut parent = running_parent(1);
        let first = parent.grant(0).unwrap();
        assert_eq!(
            parent.complete(&first, ChildCompletion::Failed).unwrap(),
            CompletionDisposition::RetryChild
        );
        let retry = parent.grant(0).unwrap();
        assert!(retry.lease_generation > first.lease_generation);
        assert_eq!(
            parent.complete(&first, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::StaleDeletePrivateArtifact
        );
        assert_eq!(
            parent.complete(&retry, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::AttemptPrepared
        );
    }

    #[test]
    fn durable_parent_journals_every_mutation_and_recovers_latest_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent = DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        parent.start().unwrap();
        let (lease, token) = parent.grant(0).unwrap();
        assert!(!token.is_cancelled());
        assert_eq!(
            parent.complete(&lease, ChildCompletion::Succeeded).unwrap(),
            CompletionDisposition::AttemptPrepared
        );
        parent.begin_commit().unwrap();

        let recovered = DurableBatchParent::recover(dir.path()).unwrap();
        assert_eq!(recovered.state(), BatchParentState::Committing);
        assert_eq!(recovered.attempt_generation(), 0);
        let records = load_parent_journal(&dir.path().join(PARENT_JOURNAL_FILE)).unwrap();
        assert_eq!(records.len(), 5);
        assert_eq!(records.last().unwrap().to, BatchParentState::Committing);
    }

    #[test]
    fn durable_parent_recovers_from_corrupt_atomic_snapshot_using_journal() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent = DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        parent.start().unwrap();
        std::fs::write(dir.path().join(PARENT_SNAPSHOT_FILE), b"{truncated").unwrap();

        let recovered = DurableBatchParent::recover(dir.path()).unwrap();
        assert_eq!(recovered.state(), BatchParentState::Running);
    }

    #[test]
    fn recovery_heals_a_snapshot_persisted_before_its_journal_record() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent = DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        parent.start().unwrap();
        let journal_path = dir.path().join(PARENT_JOURNAL_FILE);
        let bytes = std::fs::read(&journal_path).unwrap();
        let mut lines: Vec<_> = bytes.split_inclusive(|byte| *byte == b'\n').collect();
        lines.pop();
        std::fs::write(&journal_path, lines.concat()).unwrap();

        let recovered = DurableBatchParent::recover(dir.path()).unwrap();

        assert_eq!(recovered.state(), BatchParentState::Running);
        let records = load_parent_journal(&journal_path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records.last().unwrap().to, BatchParentState::Running);
    }

    #[test]
    fn restart_fences_lost_active_leases_before_retrying_a_new_generation() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent = DurableBatchParent::create(dir.path(), "parent", 2).unwrap();
        parent.start().unwrap();
        let (_lease, _token) = parent.grant(0).unwrap();

        let mut recovered = DurableBatchParent::recover(dir.path()).unwrap();
        assert_eq!(recovered.state(), BatchParentState::Fenced);
        recovered.begin_retry().unwrap();
        assert_eq!(recovered.attempt_generation(), 1);
        assert_eq!(recovered.state(), BatchParentState::Retrying);
    }

    #[test]
    fn complete_malformed_parent_journal_record_fails_closed_without_truncation() {
        let dir = tempfile::tempdir().unwrap();
        DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        let journal_path = dir.path().join(PARENT_JOURNAL_FILE);
        let mut journal = OpenOptions::new().append(true).open(&journal_path).unwrap();
        journal.write_all(b"{malformed}\n").unwrap();
        journal.sync_all().unwrap();
        drop(journal);
        let before = std::fs::read(&journal_path).unwrap();

        assert!(load_parent_journal(&journal_path).is_err());
        assert_eq!(std::fs::read(&journal_path).unwrap(), before);
    }
}
