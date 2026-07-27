//! Sparse, crash-recoverable state for one server-owned batch parent.
//!
//! Version 2 journals one constant-size delta per reducer mutation. A parent
//! materializes only a bounded child window and compacts ordered successes
//! into `settled_prefix`, so a large lazy batch does not allocate `N` child
//! states. The reader remains compatible with version-1 snapshot journals.

use anyhow::Context as _;
use mold_inference::InferenceCancellationToken;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const PARENT_SNAPSHOT_FILE: &str = "parent-state.json";
const PARENT_JOURNAL_FILE: &str = "parent-journal.jsonl";
const PARENT_PROTOCOL_V1: u32 = 1;
const PARENT_PROTOCOL_V2: u32 = 2;
pub const MAX_MATERIALIZED_CHILDREN: usize = 1024;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchChildLease {
    pub parent_id: String,
    pub child_index: usize,
    pub attempt_generation: u64,
    pub lease_generation: u64,
}

/// Durable proof that a particular lease produced one immutable staged file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchChildReceipt {
    pub version: u32,
    pub parent_id: String,
    pub attempt_generation: u64,
    pub child_index: usize,
    pub lease_generation: u64,
    pub checksum_sha256: String,
    pub size_bytes: u64,
    pub record_identity_sha256: String,
}

impl BatchChildReceipt {
    pub const VERSION: u32 = 1;

    fn validate_for(&self, lease: &BatchChildLease) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.version == Self::VERSION,
            "unsupported child receipt version"
        );
        anyhow::ensure!(
            self.parent_id == lease.parent_id
                && self.attempt_generation == lease.attempt_generation
                && self.child_index == lease.child_index
                && self.lease_generation == lease.lease_generation,
            "staged child receipt does not match its lease"
        );
        for (name, digest) in [
            ("child checksum", self.checksum_sha256.as_str()),
            ("record identity", self.record_identity_sha256.as_str()),
        ] {
            anyhow::ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "{name} is not lowercase SHA-256"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionDisposition {
    Accepted,
    RetryChild,
    AttemptPrepared,
    AttemptFenced,
    AttemptFencedDeletePrivateArtifact,
    ClosedAttemptDeletePrivateArtifact,
    StaleDeletePrivateArtifact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildCompletion {
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingIndexWindow {
    pub indices: Vec<usize>,
    pub next_cursor: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SparseChild {
    state: ChildState,
    lease_generation: u64,
    retry_count: u8,
}

#[derive(Clone, Debug)]
pub struct BatchParentReducer {
    parent_id: String,
    total_children: usize,
    attempt_generation: u64,
    state: BatchParentState,
    settled_prefix: usize,
    children: BTreeMap<usize, SparseChild>,
    active: BTreeSet<usize>,
    cancellation_tokens: BTreeMap<usize, InferenceCancellationToken>,
    terminal_after_fence: Option<BatchParentState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct BatchParentCheckpointV2 {
    version: u32,
    through_sequence: u64,
    parent_id: String,
    total_children: usize,
    attempt_generation: u64,
    state: BatchParentState,
    settled_prefix: usize,
    children: BTreeMap<usize, SparseChild>,
    active: BTreeSet<usize>,
    terminal_after_fence: Option<BatchParentState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct BatchParentSnapshotV1 {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BatchParentJournalRecordV1 {
    sequence: u64,
    attempt_generation: u64,
    from: BatchParentState,
    to: BatchParentState,
    snapshot: BatchParentSnapshotV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BatchParentJournalRecordV2 {
    version: u32,
    sequence: u64,
    attempt_generation: u64,
    event: BatchParentJournalEventV2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BatchParentJournalEventV2 {
    Initialized {
        parent_id: String,
        total_children: usize,
    },
    Started,
    ChildGranted {
        child_index: usize,
        lease_generation: u64,
    },
    ChildCompleted {
        child_index: usize,
        lease_generation: u64,
        completion: ChildCompletion,
        disposition: CompletionDisposition,
        receipt: Option<BatchChildReceipt>,
    },
    CancelRequested {
        disposition: CompletionDisposition,
    },
    AttemptRetried {
        next_attempt_generation: u64,
    },
    FenceFinalized,
    CommitStarted,
    Committed,
    LostLeaseRequeued {
        child_index: usize,
        lease_generation: u64,
    },
    LostLeasesFenced {
        child_indices: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
enum StoredParentRecord {
    V1(BatchParentJournalRecordV1),
    V2(BatchParentJournalRecordV2),
}

impl StoredParentRecord {
    fn sequence(&self) -> u64 {
        match self {
            Self::V1(record) => record.sequence,
            Self::V2(record) => record.sequence,
        }
    }
}

/// Durable, serialized authority for one batch parent.
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
        let reducer = BatchParentReducer::new(parent_id, total_children)?;
        std::fs::create_dir_all(directory)?;
        sync_parent_dir(directory)?;
        anyhow::ensure!(
            !directory.join(PARENT_SNAPSHOT_FILE).exists()
                && !directory.join(PARENT_JOURNAL_FILE).exists(),
            "batch parent authority already exists in {}",
            directory.display()
        );
        let record = BatchParentJournalRecordV2 {
            version: PARENT_PROTOCOL_V2,
            sequence: 0,
            attempt_generation: 0,
            event: BatchParentJournalEventV2::Initialized {
                parent_id: reducer.parent_id.clone(),
                total_children,
            },
        };
        publish_initial_parent_journal(directory, &record)?;
        let durable = Self {
            reducer,
            directory: directory.to_path_buf(),
            next_sequence: 1,
            poisoned: false,
        };
        durable.write_checkpoint(0)?;
        Ok(durable)
    }

    pub fn recover(directory: &Path) -> anyhow::Result<Self> {
        let mut durable = Self::recover_inner(directory)?;
        if !durable.reducer.active.is_empty() {
            durable.fence_lost_active()?;
        }
        Ok(durable)
    }

    /// Joint transaction recovery uses this before deciding whether each lost
    /// lease has a matching durable staging receipt.
    pub(crate) fn recover_unfenced(directory: &Path) -> anyhow::Result<Self> {
        Self::recover_inner(directory)
    }

    fn recover_inner(directory: &Path) -> anyhow::Result<Self> {
        let mut records = load_parent_journal(&directory.join(PARENT_JOURNAL_FILE))?;
        anyhow::ensure!(
            !records.is_empty(),
            "batch parent has no recoverable journal"
        );
        reconcile_v1_disk_ahead(directory, &mut records)?;
        let reducer = replay_parent_journal(&records)?;
        let durable = Self {
            reducer,
            directory: directory.to_path_buf(),
            next_sequence: records
                .last()
                .map_or(0, |record| record.sequence().saturating_add(1)),
            poisoned: false,
        };
        durable.reconcile_checkpoint()?;
        Ok(durable)
    }

    pub fn state(&self) -> BatchParentState {
        self.reducer.state()
    }

    pub fn parent_id(&self) -> &str {
        &self.reducer.parent_id
    }

    pub fn total_children(&self) -> usize {
        self.reducer.total_children
    }

    pub fn attempt_generation(&self) -> u64 {
        self.reducer.attempt_generation()
    }

    pub fn settled_prefix(&self) -> usize {
        self.reducer.settled_prefix
    }

    pub fn tracked_child_count(&self) -> usize {
        self.reducer.children.len()
    }

    pub fn pending_window(
        &self,
        cursor: usize,
        limit: usize,
    ) -> anyhow::Result<PendingIndexWindow> {
        self.reducer.pending_window(cursor, limit)
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        self.mutate(
            |reducer| reducer.start(),
            |_| BatchParentJournalEventV2::Started,
        )
    }

    pub fn grant(
        &mut self,
        child_index: usize,
    ) -> anyhow::Result<(BatchChildLease, InferenceCancellationToken)> {
        let lease = self.mutate(
            |reducer| reducer.grant(child_index),
            |lease| BatchParentJournalEventV2::ChildGranted {
                child_index: lease.child_index,
                lease_generation: lease.lease_generation,
            },
        )?;
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
        self.complete_inner(lease, completion, None)
    }

    pub fn complete_staged(
        &mut self,
        lease: &BatchChildLease,
        receipt: BatchChildReceipt,
    ) -> anyhow::Result<CompletionDisposition> {
        receipt.validate_for(lease)?;
        self.complete_inner(lease, ChildCompletion::Succeeded, Some(receipt))
    }

    fn complete_inner(
        &mut self,
        lease: &BatchChildLease,
        completion: ChildCompletion,
        receipt: Option<BatchChildReceipt>,
    ) -> anyhow::Result<CompletionDisposition> {
        let lease_for_event = lease.clone();
        let receipt_for_event = receipt.clone();
        self.mutate_optional(
            |reducer| reducer.complete(lease, completion),
            move |disposition| BatchParentJournalEventV2::ChildCompleted {
                child_index: lease_for_event.child_index,
                lease_generation: lease_for_event.lease_generation,
                completion,
                disposition: *disposition,
                receipt: receipt_for_event,
            },
        )
    }

    pub fn request_cancel(&mut self) -> anyhow::Result<CompletionDisposition> {
        self.mutate(BatchParentReducer::request_cancel, |disposition| {
            BatchParentJournalEventV2::CancelRequested {
                disposition: *disposition,
            }
        })
    }

    pub fn begin_retry(&mut self) -> anyhow::Result<()> {
        let next_attempt_generation = self
            .reducer
            .attempt_generation
            .checked_add(1)
            .context("batch attempt generation overflow")?;
        self.mutate(BatchParentReducer::begin_retry, |()| {
            BatchParentJournalEventV2::AttemptRetried {
                next_attempt_generation,
            }
        })
    }

    pub fn finalize_fence(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::finalize_fence, |_| {
            BatchParentJournalEventV2::FenceFinalized
        })
    }

    pub fn begin_commit(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::begin_commit, |_| {
            BatchParentJournalEventV2::CommitStarted
        })
    }

    pub fn mark_committed(&mut self) -> anyhow::Result<()> {
        self.mutate(BatchParentReducer::mark_committed, |_| {
            BatchParentJournalEventV2::Committed
        })
    }

    pub(crate) fn requeue_lost_active(&mut self, child_index: usize) -> anyhow::Result<()> {
        let lease_generation = self
            .reducer
            .children
            .get(&child_index)
            .context("lost child is not materialized")?
            .lease_generation;
        self.mutate(
            |reducer| reducer.requeue_lost_active(child_index),
            |_| BatchParentJournalEventV2::LostLeaseRequeued {
                child_index,
                lease_generation,
            },
        )
    }

    pub(crate) fn active_children(&self) -> Vec<usize> {
        self.reducer.active.iter().copied().collect()
    }

    pub(crate) fn fence_lost_active(&mut self) -> anyhow::Result<()> {
        let child_indices = self.active_children();
        anyhow::ensure!(
            !child_indices.is_empty(),
            "parent has no lost active leases"
        );
        self.mutate(BatchParentReducer::fence_lost_active_leases, |_| {
            BatchParentJournalEventV2::LostLeasesFenced { child_indices }
        })
    }

    pub(crate) fn child_is_succeeded(&self, child_index: usize) -> bool {
        self.reducer.child_is_succeeded(child_index)
    }

    pub(crate) fn accepted_receipts(
        &self,
    ) -> anyhow::Result<BTreeMap<usize, Option<BatchChildReceipt>>> {
        accepted_receipts_from_journal(&self.directory.join(PARENT_JOURNAL_FILE))
    }

    fn mutate<T>(
        &mut self,
        operation: impl FnOnce(&mut BatchParentReducer) -> anyhow::Result<T>,
        event: impl FnOnce(&T) -> BatchParentJournalEventV2,
    ) -> anyhow::Result<T> {
        self.mutate_optional(operation, event)
    }

    fn mutate_optional<T>(
        &mut self,
        operation: impl FnOnce(&mut BatchParentReducer) -> anyhow::Result<T>,
        event: impl FnOnce(&T) -> BatchParentJournalEventV2,
    ) -> anyhow::Result<T> {
        anyhow::ensure!(
            !self.poisoned,
            "batch parent durable authority is poisoned after a persistence failure"
        );
        let before = self.reducer.clone();
        let result = match operation(&mut self.reducer) {
            Ok(result) => result,
            Err(error) => {
                self.reducer = before;
                return Err(error);
            }
        };
        if reducer_authority_eq(&self.reducer, &before) {
            return Ok(result);
        }
        if let Err(error) = self.persist_event(event(&result)) {
            self.reducer = before;
            self.poisoned = true;
            return Err(error)
                .context("persisting batch parent delta; retire and recover this authority");
        }
        Ok(result)
    }

    fn persist_event(&mut self, event: BatchParentJournalEventV2) -> anyhow::Result<()> {
        let sequence = self.next_sequence;
        let record = BatchParentJournalRecordV2 {
            version: PARENT_PROTOCOL_V2,
            sequence,
            attempt_generation: self.reducer.attempt_generation,
            event,
        };
        append_parent_record(
            &self.directory.join(PARENT_JOURNAL_FILE),
            &StoredParentRecord::V2(record),
        )?;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .context("batch parent journal sequence overflow")?;
        if checkpoint_barrier(self.reducer.state) {
            self.write_checkpoint(sequence)?;
        }
        Ok(())
    }

    fn write_checkpoint(&self, through_sequence: u64) -> anyhow::Result<()> {
        let checkpoint = self.reducer.checkpoint(through_sequence);
        atomic_write_parent_json(&self.directory.join(PARENT_SNAPSHOT_FILE), &checkpoint)
    }

    fn reconcile_checkpoint(&self) -> anyhow::Result<()> {
        let path = self.directory.join(PARENT_SNAPSHOT_FILE);
        let rewrite = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(value) if value.get("version").and_then(|v| v.as_u64()) == Some(2) => {
                    let checkpoint: BatchParentCheckpointV2 = serde_json::from_value(value)?;
                    validate_v2_checkpoint(&checkpoint)?;
                    anyhow::ensure!(
                        checkpoint.through_sequence < self.next_sequence,
                        "batch parent checkpoint is ahead of its journal"
                    );
                    !checkpoint_matches_reducer(&checkpoint, &self.reducer)
                }
                Ok(value) if value.get("version").and_then(|v| v.as_u64()) == Some(1) => true,
                _ => true,
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
            Err(error) => return Err(error.into()),
        };
        if rewrite {
            self.write_checkpoint(self.next_sequence.saturating_sub(1))?;
        }
        Ok(())
    }
}

fn checkpoint_barrier(state: BatchParentState) -> bool {
    matches!(
        state,
        BatchParentState::Prepared
            | BatchParentState::Committing
            | BatchParentState::Committed
            | BatchParentState::Fenced
            | BatchParentState::Cancelled
            | BatchParentState::Failed
    )
}

impl BatchParentReducer {
    pub fn new(parent_id: impl Into<String>, total_children: usize) -> anyhow::Result<Self> {
        let parent_id = parent_id.into();
        validate_parent_id(&parent_id)?;
        anyhow::ensure!(
            total_children > 0,
            "batch parent must have at least one child"
        );
        Ok(Self {
            parent_id,
            total_children,
            attempt_generation: 0,
            state: BatchParentState::Queued,
            settled_prefix: 0,
            children: BTreeMap::new(),
            active: BTreeSet::new(),
            cancellation_tokens: BTreeMap::new(),
            terminal_after_fence: None,
        })
    }

    pub fn state(&self) -> BatchParentState {
        self.state
    }

    pub fn attempt_generation(&self) -> u64 {
        self.attempt_generation
    }

    pub fn tracked_child_count(&self) -> usize {
        self.children.len()
    }

    pub fn pending_window(
        &self,
        cursor: usize,
        limit: usize,
    ) -> anyhow::Result<PendingIndexWindow> {
        anyhow::ensure!(
            (1..=MAX_MATERIALIZED_CHILDREN).contains(&limit),
            "pending window limit must be between 1 and {MAX_MATERIALIZED_CHILDREN}"
        );
        let start = cursor.max(self.settled_prefix);
        let materialized_end = self
            .settled_prefix
            .saturating_add(MAX_MATERIALIZED_CHILDREN)
            .min(self.total_children);
        let mut indices = Vec::with_capacity(limit);
        let mut next_cursor = None;
        for index in start..materialized_end {
            if self.child_is_pending(index) {
                if indices.len() == limit {
                    next_cursor = Some(index);
                    break;
                }
                indices.push(index);
            }
        }
        Ok(PendingIndexWindow {
            indices,
            next_cursor,
        })
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
            "batch child index out of range"
        );
        anyhow::ensure!(
            child_index
                < self
                    .settled_prefix
                    .saturating_add(MAX_MATERIALIZED_CHILDREN),
            "batch child {child_index} is outside the current materialization window"
        );
        anyhow::ensure!(
            self.child_is_pending(child_index),
            "batch child is not pending"
        );
        let child = self.children.entry(child_index).or_insert(SparseChild {
            state: ChildState::Pending,
            lease_generation: 0,
            retry_count: 0,
        });
        child.lease_generation = child
            .lease_generation
            .checked_add(1)
            .context("batch child lease generation overflow")?;
        child.state = ChildState::Active;
        anyhow::ensure!(
            self.active.insert(child_index),
            "batch child already active"
        );
        self.cancellation_tokens
            .insert(child_index, InferenceCancellationToken::default());
        Ok(BatchChildLease {
            parent_id: self.parent_id.clone(),
            child_index,
            attempt_generation: self.attempt_generation,
            lease_generation: child.lease_generation,
        })
    }

    pub fn cancellation_token(
        &self,
        lease: &BatchChildLease,
    ) -> Option<InferenceCancellationToken> {
        (lease.parent_id == self.parent_id
            && lease.attempt_generation == self.attempt_generation
            && self
                .children
                .get(&lease.child_index)
                .is_some_and(|child| child.lease_generation == lease.lease_generation))
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
            "completion parent mismatch"
        );
        anyhow::ensure!(
            lease.child_index < self.total_children,
            "batch child index out of range"
        );
        if lease.attempt_generation < self.attempt_generation
            || lease.child_index < self.settled_prefix
        {
            return Ok(CompletionDisposition::StaleDeletePrivateArtifact);
        }
        anyhow::ensure!(
            lease.attempt_generation == self.attempt_generation,
            "completion generation is ahead of current generation"
        );
        let Some(child) = self.children.get(&lease.child_index) else {
            anyhow::bail!("batch child has no materialized lease");
        };
        if lease.lease_generation < child.lease_generation || child.state == ChildState::Succeeded {
            return Ok(CompletionDisposition::StaleDeletePrivateArtifact);
        }
        anyhow::ensure!(
            lease.lease_generation == child.lease_generation,
            "completion lease generation is ahead of current lease"
        );
        anyhow::ensure!(
            child.state == ChildState::Active && self.active.remove(&lease.child_index),
            "batch child has no active lease"
        );
        self.cancellation_tokens.remove(&lease.child_index);
        match self.state {
            BatchParentState::Running => self.complete_running(lease.child_index, completion),
            BatchParentState::Cancelling | BatchParentState::Failing => {
                self.complete_closed_attempt(lease.child_index, completion)
            }
            _ => anyhow::bail!("completion invalid while parent is {:?}", self.state),
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
        self.retain_only_active();
        self.cancel_active_children();
        if self.active.is_empty() {
            self.state = BatchParentState::Fenced;
            self.children.clear();
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
            .context("batch attempt generation overflow")?;
        self.settled_prefix = 0;
        self.children.clear();
        self.active.clear();
        self.cancellation_tokens.clear();
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
            .context("fenced parent has no terminal disposition")?;
        anyhow::ensure!(
            matches!(
                terminal,
                BatchParentState::Cancelled | BatchParentState::Failed
            ),
            "invalid terminal state after fence"
        );
        self.state = terminal;
        Ok(())
    }

    pub fn begin_commit(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Prepared,
            "parent is not prepared"
        );
        self.state = BatchParentState::Committing;
        Ok(())
    }

    pub fn mark_committed(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Committing,
            "parent is not committing"
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
                self.children.get_mut(&child_index).unwrap().state = ChildState::Succeeded;
                self.compact_successes();
                if self.settled_prefix == self.total_children {
                    anyhow::ensure!(
                        self.active.is_empty(),
                        "all children succeeded with active leases"
                    );
                    self.children.clear();
                    self.state = BatchParentState::Prepared;
                    Ok(CompletionDisposition::AttemptPrepared)
                } else {
                    Ok(CompletionDisposition::Accepted)
                }
            }
            ChildCompletion::Failed | ChildCompletion::Cancelled => {
                let child = self.children.get_mut(&child_index).unwrap();
                child.state = match completion {
                    ChildCompletion::Failed => ChildState::Failed,
                    ChildCompletion::Cancelled => ChildState::Cancelled,
                    ChildCompletion::Succeeded => unreachable!(),
                };
                if completion == ChildCompletion::Failed && child.retry_count == 0 {
                    child.retry_count = 1;
                    child.state = ChildState::Pending;
                    return Ok(CompletionDisposition::RetryChild);
                }
                self.state = BatchParentState::Failing;
                self.terminal_after_fence = Some(BatchParentState::Failed);
                self.retain_only_active();
                self.cancel_active_children();
                if self.active.is_empty() {
                    self.children.clear();
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
        let has_artifact = completion == ChildCompletion::Succeeded;
        if let Some(child) = self.children.get_mut(&child_index) {
            child.state = ChildState::Cancelled;
        }
        if self.active.is_empty() {
            self.children.clear();
            self.state = BatchParentState::Fenced;
            Ok(if has_artifact {
                CompletionDisposition::AttemptFencedDeletePrivateArtifact
            } else {
                CompletionDisposition::AttemptFenced
            })
        } else {
            Ok(if has_artifact {
                CompletionDisposition::ClosedAttemptDeletePrivateArtifact
            } else {
                CompletionDisposition::Accepted
            })
        }
    }

    fn compact_successes(&mut self) {
        while self
            .children
            .get(&self.settled_prefix)
            .is_some_and(|child| child.state == ChildState::Succeeded)
        {
            self.children.remove(&self.settled_prefix);
            self.settled_prefix += 1;
        }
    }

    fn child_is_pending(&self, child_index: usize) -> bool {
        child_index >= self.settled_prefix
            && self
                .children
                .get(&child_index)
                .is_none_or(|child| child.state == ChildState::Pending)
    }

    fn child_is_succeeded(&self, child_index: usize) -> bool {
        child_index < self.settled_prefix
            || self
                .children
                .get(&child_index)
                .is_some_and(|child| child.state == ChildState::Succeeded)
    }

    fn retain_only_active(&mut self) {
        self.children.retain(|index, _| self.active.contains(index));
    }

    fn cancel_active_children(&self) {
        for token in self.cancellation_tokens.values() {
            token.cancel();
        }
    }

    fn fence_lost_active_leases(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.active.is_empty()
                && matches!(
                    self.state,
                    BatchParentState::Running
                        | BatchParentState::Cancelling
                        | BatchParentState::Failing
                ),
            "batch parent cannot fence lost leases from {:?}",
            self.state
        );
        self.active.clear();
        self.children.clear();
        self.cancellation_tokens.clear();
        self.terminal_after_fence = Some(if self.state == BatchParentState::Cancelling {
            BatchParentState::Cancelled
        } else {
            BatchParentState::Failed
        });
        self.state = BatchParentState::Fenced;
        Ok(())
    }

    fn requeue_lost_active(&mut self, child_index: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state == BatchParentState::Running,
            "parent is not running"
        );
        anyhow::ensure!(self.active.remove(&child_index), "child is not active");
        let child = self
            .children
            .get_mut(&child_index)
            .context("child is missing")?;
        child.state = ChildState::Pending;
        self.cancellation_tokens.remove(&child_index);
        Ok(())
    }

    fn checkpoint(&self, through_sequence: u64) -> BatchParentCheckpointV2 {
        BatchParentCheckpointV2 {
            version: PARENT_PROTOCOL_V2,
            through_sequence,
            parent_id: self.parent_id.clone(),
            total_children: self.total_children,
            attempt_generation: self.attempt_generation,
            state: self.state,
            settled_prefix: self.settled_prefix,
            children: self.children.clone(),
            active: self.active.clone(),
            terminal_after_fence: self.terminal_after_fence,
        }
    }
}

fn reducer_authority_eq(left: &BatchParentReducer, right: &BatchParentReducer) -> bool {
    left.parent_id == right.parent_id
        && left.total_children == right.total_children
        && left.attempt_generation == right.attempt_generation
        && left.state == right.state
        && left.settled_prefix == right.settled_prefix
        && left.children == right.children
        && left.active == right.active
        && left.terminal_after_fence == right.terminal_after_fence
}

fn validate_parent_id(parent_id: &str) -> anyhow::Result<()> {
    let path = Path::new(parent_id);
    anyhow::ensure!(
        !parent_id.is_empty()
            && path.file_name().and_then(|name| name.to_str()) == Some(parent_id)
            && parent_id != "."
            && parent_id != "..",
        "batch parent id must be one path component"
    );
    Ok(())
}

fn validate_v2_checkpoint(checkpoint: &BatchParentCheckpointV2) -> anyhow::Result<()> {
    anyhow::ensure!(
        checkpoint.version == PARENT_PROTOCOL_V2,
        "unsupported parent checkpoint"
    );
    validate_parent_id(&checkpoint.parent_id)?;
    anyhow::ensure!(
        checkpoint.total_children > 0,
        "batch parent has no children"
    );
    anyhow::ensure!(
        checkpoint.settled_prefix <= checkpoint.total_children
            && checkpoint.children.len() <= MAX_MATERIALIZED_CHILDREN
            && checkpoint
                .children
                .keys()
                .all(|index| *index >= checkpoint.settled_prefix
                    && *index < checkpoint.total_children
                    && *index
                        < checkpoint
                            .settled_prefix
                            .saturating_add(MAX_MATERIALIZED_CHILDREN)),
        "batch parent checkpoint violates the sparse materialization bound"
    );
    anyhow::ensure!(
        checkpoint.active.iter().all(|index| {
            checkpoint
                .children
                .get(index)
                .is_some_and(|child| child.state == ChildState::Active)
        }),
        "batch parent checkpoint active set is inconsistent"
    );
    Ok(())
}

fn checkpoint_matches_reducer(
    checkpoint: &BatchParentCheckpointV2,
    reducer: &BatchParentReducer,
) -> bool {
    let expected = reducer.checkpoint(checkpoint.through_sequence);
    expected == *checkpoint
}

fn replay_parent_journal(records: &[StoredParentRecord]) -> anyhow::Result<BatchParentReducer> {
    let mut reducer: Option<BatchParentReducer> = None;
    let mut previous_v1: Option<BatchParentSnapshotV1> = None;
    for (position, record) in records.iter().enumerate() {
        anyhow::ensure!(
            record.sequence() == position as u64,
            "batch parent journal sequence gap"
        );
        match record {
            StoredParentRecord::V1(record) => {
                validate_v1_snapshot(&record.snapshot)?;
                anyhow::ensure!(
                    record.attempt_generation == record.snapshot.attempt_generation
                        && record.to == record.snapshot.state,
                    "invalid v1 parent journal envelope"
                );
                if let Some(previous) = previous_v1.as_ref() {
                    anyhow::ensure!(
                        record.from == previous.state
                            && is_legal_v1_successor(previous, &record.snapshot),
                        "illegal v1 parent journal transition"
                    );
                } else {
                    anyhow::ensure!(
                        position == 0
                            && record.from == BatchParentState::Queued
                            && record.to == BatchParentState::Queued
                            && is_initial_v1_snapshot(&record.snapshot),
                        "invalid initial v1 parent journal"
                    );
                }
                reducer = Some(reducer_from_v1(&record.snapshot));
                previous_v1 = Some(record.snapshot.clone());
            }
            StoredParentRecord::V2(record) => {
                anyhow::ensure!(
                    record.version == PARENT_PROTOCOL_V2,
                    "unsupported parent record"
                );
                if let BatchParentJournalEventV2::Initialized {
                    parent_id,
                    total_children,
                } = &record.event
                {
                    anyhow::ensure!(
                        position == 0 && reducer.is_none(),
                        "duplicate parent initialization"
                    );
                    reducer = Some(BatchParentReducer::new(parent_id.clone(), *total_children)?);
                    continue;
                }
                let reducer = reducer
                    .as_mut()
                    .context("parent delta precedes initialization")?;
                anyhow::ensure!(
                    record.attempt_generation == reducer.attempt_generation
                        || matches!(
                            record.event,
                            BatchParentJournalEventV2::AttemptRetried { .. }
                        ),
                    "parent delta attempt generation mismatch"
                );
                apply_v2_event(reducer, &record.event)?;
            }
        }
    }
    reducer.context("batch parent journal is empty")
}

fn apply_v2_event(
    reducer: &mut BatchParentReducer,
    event: &BatchParentJournalEventV2,
) -> anyhow::Result<()> {
    match event {
        BatchParentJournalEventV2::Initialized { .. } => anyhow::bail!("duplicate initialization"),
        BatchParentJournalEventV2::Started => reducer.start(),
        BatchParentJournalEventV2::ChildGranted {
            child_index,
            lease_generation,
        } => {
            let lease = reducer.grant(*child_index)?;
            anyhow::ensure!(
                lease.lease_generation == *lease_generation,
                "grant generation drift"
            );
            Ok(())
        }
        BatchParentJournalEventV2::ChildCompleted {
            child_index,
            lease_generation,
            completion,
            disposition,
            receipt,
        } => {
            let lease = BatchChildLease {
                parent_id: reducer.parent_id.clone(),
                child_index: *child_index,
                attempt_generation: reducer.attempt_generation,
                lease_generation: *lease_generation,
            };
            if let Some(receipt) = receipt {
                receipt.validate_for(&lease)?;
                anyhow::ensure!(
                    *completion == ChildCompletion::Succeeded,
                    "receipt on non-success"
                );
            }
            let actual = reducer.complete(&lease, *completion)?;
            anyhow::ensure!(actual == *disposition, "completion disposition drift");
            Ok(())
        }
        BatchParentJournalEventV2::CancelRequested { disposition } => {
            anyhow::ensure!(
                reducer.request_cancel()? == *disposition,
                "cancel disposition drift"
            );
            Ok(())
        }
        BatchParentJournalEventV2::AttemptRetried {
            next_attempt_generation,
        } => {
            reducer.begin_retry()?;
            anyhow::ensure!(
                reducer.attempt_generation == *next_attempt_generation,
                "retry generation drift"
            );
            Ok(())
        }
        BatchParentJournalEventV2::FenceFinalized => reducer.finalize_fence(),
        BatchParentJournalEventV2::CommitStarted => reducer.begin_commit(),
        BatchParentJournalEventV2::Committed => reducer.mark_committed(),
        BatchParentJournalEventV2::LostLeaseRequeued {
            child_index,
            lease_generation,
        } => {
            anyhow::ensure!(
                reducer
                    .children
                    .get(child_index)
                    .is_some_and(|child| child.lease_generation == *lease_generation),
                "lost lease generation drift"
            );
            reducer.requeue_lost_active(*child_index)
        }
        BatchParentJournalEventV2::LostLeasesFenced { child_indices } => {
            anyhow::ensure!(
                reducer
                    .active
                    .iter()
                    .copied()
                    .eq(child_indices.iter().copied()),
                "lost lease fence set drift"
            );
            reducer.fence_lost_active_leases()
        }
    }
}

fn accepted_receipts_from_journal(
    path: &Path,
) -> anyhow::Result<BTreeMap<usize, Option<BatchChildReceipt>>> {
    let records = load_parent_journal(path)?;
    let mut accepted = BTreeMap::new();
    let mut previous_v1: Option<BatchParentSnapshotV1> = None;
    for record in records {
        match record {
            StoredParentRecord::V1(record) => {
                if let Some(previous) = previous_v1.as_ref() {
                    for (index, (before, after)) in previous
                        .children
                        .iter()
                        .zip(&record.snapshot.children)
                        .enumerate()
                    {
                        if *before == ChildState::Active && *after == ChildState::Succeeded {
                            accepted.insert(index, None);
                        }
                    }
                }
                previous_v1 = Some(record.snapshot);
            }
            StoredParentRecord::V2(record) => {
                if let BatchParentJournalEventV2::ChildCompleted {
                    child_index,
                    completion: ChildCompletion::Succeeded,
                    disposition:
                        CompletionDisposition::Accepted | CompletionDisposition::AttemptPrepared,
                    receipt,
                    ..
                } = &record.event
                {
                    accepted.insert(*child_index, receipt.clone());
                }
                if matches!(
                    record.event,
                    BatchParentJournalEventV2::AttemptRetried { .. }
                ) {
                    accepted.clear();
                }
            }
        }
    }
    Ok(accepted)
}

fn reconcile_v1_disk_ahead(
    directory: &Path,
    records: &mut Vec<StoredParentRecord>,
) -> anyhow::Result<()> {
    let snapshot_path = directory.join(PARENT_SNAPSHOT_FILE);
    let value = match std::fs::read(&snapshot_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
    {
        Some(value) if value.get("version").and_then(|version| version.as_u64()) == Some(1) => {
            value
        }
        _ => return Ok(()),
    };
    let disk: BatchParentSnapshotV1 = match serde_json::from_value(value) {
        Ok(snapshot) if validate_v1_snapshot(&snapshot).is_ok() => snapshot,
        _ => return Ok(()),
    };
    if records
        .iter()
        .any(|record| matches!(record, StoredParentRecord::V1(record) if record.snapshot == disk))
    {
        return Ok(());
    }
    let previous = match records.last() {
        Some(StoredParentRecord::V1(record)) => &record.snapshot,
        _ => anyhow::bail!("v1 batch parent disk snapshot diverges from its durable journal"),
    };
    anyhow::ensure!(
        is_legal_v1_successor(previous, &disk),
        "v1 batch parent disk snapshot diverges from its durable journal"
    );
    let record = StoredParentRecord::V1(BatchParentJournalRecordV1 {
        sequence: records.len() as u64,
        attempt_generation: disk.attempt_generation,
        from: previous.state,
        to: disk.state,
        snapshot: disk,
    });
    append_parent_record(&directory.join(PARENT_JOURNAL_FILE), &record)?;
    records.push(record);
    Ok(())
}

fn validate_v1_snapshot(snapshot: &BatchParentSnapshotV1) -> anyhow::Result<()> {
    anyhow::ensure!(
        snapshot.version == PARENT_PROTOCOL_V1,
        "unsupported v1 parent snapshot"
    );
    validate_parent_id(&snapshot.parent_id)?;
    anyhow::ensure!(
        snapshot.total_children > 0
            && snapshot.children.len() == snapshot.total_children
            && snapshot.child_lease_generations.len() == snapshot.total_children
            && snapshot.retry_counts.len() == snapshot.total_children,
        "v1 parent child vectors are inconsistent"
    );
    anyhow::ensure!(
        snapshot.children.iter().enumerate().all(
            |(index, state)| (*state == ChildState::Active) == snapshot.active.contains(&index)
        ),
        "v1 parent active set is inconsistent"
    );
    Ok(())
}

fn reducer_from_v1(snapshot: &BatchParentSnapshotV1) -> BatchParentReducer {
    let settled_prefix = snapshot
        .children
        .iter()
        .take_while(|state| **state == ChildState::Succeeded)
        .count();
    let children = snapshot
        .children
        .iter()
        .enumerate()
        .filter(|(index, state)| *index >= settled_prefix && **state != ChildState::Pending)
        .map(|(index, state)| {
            (
                index,
                SparseChild {
                    state: *state,
                    lease_generation: snapshot.child_lease_generations[index],
                    retry_count: snapshot.retry_counts[index],
                },
            )
        })
        .collect();
    BatchParentReducer {
        parent_id: snapshot.parent_id.clone(),
        total_children: snapshot.total_children,
        attempt_generation: snapshot.attempt_generation,
        state: snapshot.state,
        settled_prefix,
        children,
        active: snapshot.active.clone(),
        cancellation_tokens: snapshot
            .active
            .iter()
            .map(|index| (*index, InferenceCancellationToken::default()))
            .collect(),
        terminal_after_fence: snapshot.terminal_after_fence,
    }
}

fn is_initial_v1_snapshot(snapshot: &BatchParentSnapshotV1) -> bool {
    snapshot.attempt_generation == 0
        && snapshot.state == BatchParentState::Queued
        && snapshot
            .children
            .iter()
            .all(|state| *state == ChildState::Pending)
        && snapshot
            .child_lease_generations
            .iter()
            .all(|value| *value == 0)
        && snapshot.active.is_empty()
        && snapshot.retry_counts.iter().all(|value| *value == 0)
        && snapshot.terminal_after_fence.is_none()
}

fn is_legal_v1_successor(previous: &BatchParentSnapshotV1, next: &BatchParentSnapshotV1) -> bool {
    let base = reducer_from_v1(previous);
    let matches = |candidate: &BatchParentReducer| v1_matches_reducer(next, candidate, previous);
    let mut candidate = base.clone();
    if candidate.start().is_ok() && matches(&candidate) {
        return true;
    }
    for child_index in 0..previous.total_children {
        let mut candidate = base.clone();
        if candidate.grant(child_index).is_ok() && matches(&candidate) {
            return true;
        }
    }
    for child_index in previous.active.iter().copied() {
        let lease = BatchChildLease {
            parent_id: previous.parent_id.clone(),
            child_index,
            attempt_generation: previous.attempt_generation,
            lease_generation: previous.child_lease_generations[child_index],
        };
        for completion in [
            ChildCompletion::Succeeded,
            ChildCompletion::Failed,
            ChildCompletion::Cancelled,
        ] {
            let mut candidate = base.clone();
            if candidate.complete(&lease, completion).is_ok() && matches(&candidate) {
                return true;
            }
        }
    }
    let mut candidate = base.clone();
    if candidate.request_cancel().is_ok() && matches(&candidate) {
        return true;
    }
    let mut candidate = base.clone();
    if candidate.begin_retry().is_ok() && matches(&candidate) {
        return true;
    }
    let mut candidate = base.clone();
    if candidate.finalize_fence().is_ok() && matches(&candidate) {
        return true;
    }
    let mut candidate = base.clone();
    if candidate.begin_commit().is_ok() && matches(&candidate) {
        return true;
    }
    let mut candidate = base.clone();
    if candidate.mark_committed().is_ok() && matches(&candidate) {
        return true;
    }
    let mut candidate = base;
    candidate.fence_lost_active_leases().is_ok() && matches(&candidate)
}

fn v1_matches_reducer(
    snapshot: &BatchParentSnapshotV1,
    reducer: &BatchParentReducer,
    previous: &BatchParentSnapshotV1,
) -> bool {
    if snapshot.parent_id != reducer.parent_id
        || snapshot.total_children != reducer.total_children
        || snapshot.attempt_generation != reducer.attempt_generation
        || snapshot.state != reducer.state
        || snapshot.active != reducer.active
        || snapshot.terminal_after_fence != reducer.terminal_after_fence
    {
        return false;
    }
    for index in 0..snapshot.total_children {
        let (state, lease, retry) = if index < reducer.settled_prefix {
            (
                ChildState::Succeeded,
                previous.child_lease_generations[index],
                previous.retry_counts[index],
            )
        } else if let Some(child) = reducer.children.get(&index) {
            (child.state, child.lease_generation, child.retry_count)
        } else {
            (ChildState::Pending, 0, 0)
        };
        if snapshot.children[index] != state
            || snapshot.child_lease_generations[index] != lease
            || snapshot.retry_counts[index] != retry
        {
            return false;
        }
    }
    true
}

fn load_parent_journal(path: &Path) -> anyhow::Result<Vec<StoredParentRecord>> {
    let bytes = std::fs::read(path)?;
    let has_incomplete_tail = !bytes.ends_with(b"\n");
    let lines: Vec<&[u8]> = bytes.split(|byte| *byte == b'\n').collect();
    let last_nonempty = lines.iter().rposition(|line| !line.is_empty());
    let mut records = Vec::new();
    let mut offset = 0_u64;
    for (index, line) in lines.into_iter().enumerate() {
        if line.is_empty() {
            if offset < bytes.len() as u64 {
                offset += 1;
            }
            continue;
        }
        let parsed = serde_json::from_slice::<serde_json::Value>(line).and_then(|value| {
            if value.get("version").and_then(|version| version.as_u64()) == Some(2) {
                serde_json::from_value::<BatchParentJournalRecordV2>(value)
                    .map(StoredParentRecord::V2)
            } else {
                serde_json::from_value::<BatchParentJournalRecordV1>(value)
                    .map(StoredParentRecord::V1)
            }
        });
        match parsed {
            Ok(record) => {
                anyhow::ensure!(
                    record.sequence() == records.len() as u64,
                    "batch parent journal sequence gap at {}",
                    path.display()
                );
                records.push(record);
                offset = (offset + line.len() as u64 + 1).min(bytes.len() as u64);
            }
            Err(error) if Some(index) == last_nonempty && has_incomplete_tail => {
                let file = OpenOptions::new().write(true).open(path)?;
                file.set_len(offset)?;
                file.sync_all()?;
                if let Some(parent) = path.parent() {
                    sync_parent_dir(parent)?;
                }
                tracing::warn!(journal = %path.display(), %error, "discarded incomplete parent journal tail");
                return Ok(records);
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("parsing batch parent journal {}", path.display()));
            }
        }
    }
    if has_incomplete_tail {
        let mut file = OpenOptions::new().append(true).open(path)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    Ok(records)
}

fn publish_initial_parent_journal(
    directory: &Path,
    record: &BatchParentJournalRecordV2,
) -> anyhow::Result<()> {
    let journal_path = directory.join(PARENT_JOURNAL_FILE);
    let temporary = directory.join(format!(
        ".{PARENT_JOURNAL_FILE}.create-{}",
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let mut bytes = serde_json::to_vec(record)?;
        bytes.push(b'\n');
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::hard_link(&temporary, &journal_path)?;
        sync_parent_dir(directory)
    })();
    let _ = std::fs::remove_file(&temporary);
    result
}

fn append_parent_record(path: &Path, record: &StoredParentRecord) -> anyhow::Result<()> {
    let mut bytes = match record {
        StoredParentRecord::V1(record) => serde_json::to_vec(record)?,
        StoredParentRecord::V2(record) => serde_json::to_vec(record)?,
    };
    bytes.push(b'\n');
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    if let Some(parent) = path.parent() {
        sync_parent_dir(parent)?;
    }
    Ok(())
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
    fn n_1_2_8_64_completion_parity_and_compaction() {
        for total in [1, 2, 8, 64] {
            let mut parent = running_parent(total);
            let leases: Vec<_> = (0..total)
                .map(|index| parent.grant(index).unwrap())
                .collect();
            for lease in leases.into_iter().rev() {
                parent.complete(&lease, ChildCompletion::Succeeded).unwrap();
            }
            assert_eq!(parent.state(), BatchParentState::Prepared);
            assert_eq!(parent.settled_prefix, total);
            assert_eq!(parent.tracked_child_count(), 0);
        }
    }

    #[test]
    fn retry_generation_and_late_completion_are_fenced() {
        let mut parent = running_parent(2);
        let failed = parent.grant(0).unwrap();
        let sibling = parent.grant(1).unwrap();
        assert_eq!(
            parent.complete(&failed, ChildCompletion::Failed).unwrap(),
            CompletionDisposition::RetryChild
        );
        let retry = parent.grant(0).unwrap();
        assert_eq!(
            parent.complete(&retry, ChildCompletion::Failed).unwrap(),
            CompletionDisposition::Accepted
        );
        assert_eq!(
            parent
                .complete(&sibling, ChildCompletion::Cancelled)
                .unwrap(),
            CompletionDisposition::AttemptFenced
        );
        parent.begin_retry().unwrap();
        parent.start().unwrap();
        assert_eq!(
            parent
                .complete(&failed, ChildCompletion::Succeeded)
                .unwrap(),
            CompletionDisposition::StaleDeletePrivateArtifact
        );
    }

    #[test]
    fn pending_windows_are_bounded_and_lazy() {
        let parent = running_parent(100_000);
        let window = parent.pending_window(0, 64).unwrap();
        assert_eq!(window.indices, (0..64).collect::<Vec<_>>());
        assert_eq!(window.next_cursor, Some(64));
        assert_eq!(parent.tracked_child_count(), 0);
        assert!(parent
            .pending_window(0, MAX_MATERIALIZED_CHILDREN + 1)
            .is_err());
    }

    #[test]
    fn hundred_thousand_child_parent_journal_growth_is_linear_and_memory_bounded() {
        let mut parent = BatchParentReducer::new("parent", 100_000).unwrap();
        parent.start().unwrap();
        let mut journal = serde_json::to_vec(&BatchParentJournalRecordV2 {
            version: 2,
            sequence: 0,
            attempt_generation: 0,
            event: BatchParentJournalEventV2::Initialized {
                parent_id: "parent".into(),
                total_children: 100_000,
            },
        })
        .unwrap();
        journal.push(b'\n');
        let mut sequence = 1_u64;
        for child_index in 0..100_000 {
            let lease = parent.grant(child_index).unwrap();
            for event in [
                BatchParentJournalEventV2::ChildGranted {
                    child_index,
                    lease_generation: lease.lease_generation,
                },
                BatchParentJournalEventV2::ChildCompleted {
                    child_index,
                    lease_generation: lease.lease_generation,
                    completion: ChildCompletion::Succeeded,
                    disposition: if child_index == 99_999 {
                        CompletionDisposition::AttemptPrepared
                    } else {
                        CompletionDisposition::Accepted
                    },
                    receipt: None,
                },
            ] {
                journal.extend(
                    serde_json::to_vec(&BatchParentJournalRecordV2 {
                        version: 2,
                        sequence,
                        attempt_generation: 0,
                        event,
                    })
                    .unwrap(),
                );
                journal.push(b'\n');
                sequence += 1;
            }
            parent.complete(&lease, ChildCompletion::Succeeded).unwrap();
            assert!(
                parent.tracked_child_count() <= 1,
                "sequential success did not compact"
            );
        }
        let checkpoint = serde_json::to_vec(&parent.checkpoint(sequence - 1)).unwrap();
        assert!(
            journal.len() < 80_000_000,
            "journal growth is not linear: {}",
            journal.len()
        );
        assert!(
            checkpoint.len() < 4096,
            "terminal checkpoint is not compact"
        );
        assert_eq!(parent.state(), BatchParentState::Prepared);
    }

    #[test]
    fn maximal_out_of_order_window_never_exceeds_materialization_bound() {
        let mut parent = running_parent(100_000);
        let leases: Vec<_> = (0..MAX_MATERIALIZED_CHILDREN)
            .map(|index| parent.grant(index).unwrap())
            .collect();
        assert_eq!(parent.tracked_child_count(), MAX_MATERIALIZED_CHILDREN);
        for lease in leases.into_iter().rev() {
            parent.complete(&lease, ChildCompletion::Succeeded).unwrap();
            assert!(parent.tracked_child_count() <= MAX_MATERIALIZED_CHILDREN);
        }
        assert_eq!(parent.settled_prefix, MAX_MATERIALIZED_CHILDREN);
        assert_eq!(parent.tracked_child_count(), 0);
    }

    #[test]
    fn child_beyond_live_window_is_rejected_then_admitted_after_prefix_advance() {
        let mut parent = running_parent(100_000);
        assert!(parent.grant(MAX_MATERIALIZED_CHILDREN).is_err());
        let first = parent.grant(0).unwrap();
        parent.complete(&first, ChildCompletion::Succeeded).unwrap();
        assert!(parent.grant(MAX_MATERIALIZED_CHILDREN).is_ok());
    }

    #[test]
    fn durable_parent_recovers_current_v2_and_fences_standalone_lost_lease() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent = DurableBatchParent::create(dir.path(), "parent", 2).unwrap();
        parent.start().unwrap();
        parent.grant(0).unwrap();
        let recovered = DurableBatchParent::recover(dir.path()).unwrap();
        assert_eq!(recovered.state(), BatchParentState::Fenced);
    }

    #[test]
    fn v1_snapshot_journal_fixture_remains_readable() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = BatchParentSnapshotV1 {
            version: 1,
            parent_id: "parent".into(),
            total_children: 2,
            attempt_generation: 0,
            state: BatchParentState::Queued,
            children: vec![ChildState::Pending; 2],
            child_lease_generations: vec![0; 2],
            active: BTreeSet::new(),
            retry_counts: vec![0; 2],
            terminal_after_fence: None,
        };
        let record = BatchParentJournalRecordV1 {
            sequence: 0,
            attempt_generation: 0,
            from: BatchParentState::Queued,
            to: BatchParentState::Queued,
            snapshot: snapshot.clone(),
        };
        std::fs::write(
            dir.path().join(PARENT_JOURNAL_FILE),
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .unwrap();
        atomic_write_parent_json(&dir.path().join(PARENT_SNAPSHOT_FILE), &snapshot).unwrap();
        let recovered = DurableBatchParent::recover(dir.path()).unwrap();
        assert_eq!(recovered.state(), BatchParentState::Queued);
        assert_eq!(recovered.total_children(), 2);
    }

    #[test]
    fn v1_snapshot_one_transition_ahead_is_appended_before_v2_checkpoint_upgrade() {
        let dir = tempfile::tempdir().unwrap();
        let queued = BatchParentSnapshotV1 {
            version: 1,
            parent_id: "parent".into(),
            total_children: 1,
            attempt_generation: 0,
            state: BatchParentState::Queued,
            children: vec![ChildState::Pending],
            child_lease_generations: vec![0],
            active: BTreeSet::new(),
            retry_counts: vec![0],
            terminal_after_fence: None,
        };
        append_parent_record(
            &dir.path().join(PARENT_JOURNAL_FILE),
            &StoredParentRecord::V1(BatchParentJournalRecordV1 {
                sequence: 0,
                attempt_generation: 0,
                from: BatchParentState::Queued,
                to: BatchParentState::Queued,
                snapshot: queued.clone(),
            }),
        )
        .unwrap();
        let mut running = queued;
        running.state = BatchParentState::Running;
        atomic_write_parent_json(&dir.path().join(PARENT_SNAPSHOT_FILE), &running).unwrap();

        let recovered = DurableBatchParent::recover(dir.path()).unwrap();

        assert_eq!(recovered.state(), BatchParentState::Running);
        assert_eq!(
            load_parent_journal(&dir.path().join(PARENT_JOURNAL_FILE))
                .unwrap()
                .len(),
            2
        );
        let checkpoint: BatchParentCheckpointV2 =
            serde_json::from_slice(&std::fs::read(dir.path().join(PARENT_SNAPSHOT_FILE)).unwrap())
                .unwrap();
        assert_eq!(checkpoint.through_sequence, 1);
    }

    #[test]
    fn v1_snapshot_ahead_with_lease_drift_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let queued = BatchParentSnapshotV1 {
            version: 1,
            parent_id: "parent".into(),
            total_children: 1,
            attempt_generation: 0,
            state: BatchParentState::Queued,
            children: vec![ChildState::Pending],
            child_lease_generations: vec![0],
            active: BTreeSet::new(),
            retry_counts: vec![0],
            terminal_after_fence: None,
        };
        append_parent_record(
            &dir.path().join(PARENT_JOURNAL_FILE),
            &StoredParentRecord::V1(BatchParentJournalRecordV1 {
                sequence: 0,
                attempt_generation: 0,
                from: BatchParentState::Queued,
                to: BatchParentState::Queued,
                snapshot: queued.clone(),
            }),
        )
        .unwrap();
        let mut divergent = queued;
        divergent.state = BatchParentState::Running;
        divergent.child_lease_generations[0] = 7;
        atomic_write_parent_json(&dir.path().join(PARENT_SNAPSHOT_FILE), &divergent).unwrap();

        assert!(DurableBatchParent::recover(dir.path()).is_err());
    }

    #[test]
    fn v2_checkpoint_ahead_of_journal_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let parent = DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        let checkpoint = parent.reducer.checkpoint(99);
        atomic_write_parent_json(&dir.path().join(PARENT_SNAPSHOT_FILE), &checkpoint).unwrap();

        let error = DurableBatchParent::recover(dir.path()).unwrap_err();

        assert!(format!("{error:#}").contains("checkpoint is ahead"));
    }

    #[test]
    fn complete_malformed_record_fails_closed_but_incomplete_tail_is_discarded() {
        let dir = tempfile::tempdir().unwrap();
        DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        let path = dir.path().join(PARENT_JOURNAL_FILE);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{bad}\n")
            .unwrap();
        assert!(DurableBatchParent::recover(dir.path()).is_err());

        let dir = tempfile::tempdir().unwrap();
        DurableBatchParent::create(dir.path(), "parent", 1).unwrap();
        let path = dir.path().join(PARENT_JOURNAL_FILE);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{truncated")
            .unwrap();
        assert_eq!(
            DurableBatchParent::recover(dir.path()).unwrap().state(),
            BatchParentState::Queued
        );
    }
}
