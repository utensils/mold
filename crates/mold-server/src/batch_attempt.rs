//! Joint durable authority for a batch reducer and its publication attempt.
//!
//! A successful child is staged and receipted by the transaction before the
//! parent reducer may accept it. Recovery always reconciles those two journals
//! together; neither side is allowed to invent success for the other.

use crate::batch_parent::{
    BatchChildLease, BatchParentState, CompletionDisposition, DurableBatchParent,
};
use crate::batch_transaction::{
    BatchManifestState, BatchTransaction, GalleryPublicationGate, RecoveryReport,
    PARENT_AUTHORITY_DIR, TRANSACTION_DIR,
};
use anyhow::Context as _;
use mold_db::GenerationRecord;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const COMMITTED_DIR: &str = "committed";

#[derive(Debug, Default, PartialEq, Eq)]
pub struct BatchAttemptReconciliation {
    pub receipts_removed: usize,
    pub leases_requeued: usize,
    pub transaction_prepared: bool,
    pub parent_rolled_forward: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct JointBatchRecoveryReport {
    pub parents_discovered: usize,
    pub running_attempts_retained: usize,
    pub parents_rolled_forward: usize,
    pub receipts_removed: usize,
    pub leases_requeued: usize,
    pub stale_attempts_rolled_back: usize,
    pub legacy_transactions: RecoveryReport,
    pub outcomes: Vec<RecoveredParentOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransactionEvidence {
    Active(BatchManifestState),
    CommittedArchive,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryAction {
    Resume,
    ConvergeCommit,
    RollForwardArchive,
    FenceAndRollback,
    Rollback,
    AlreadyFenced,
    AlreadyTerminal,
    AlreadyCommitted,
}

fn recovery_action(
    parent: BatchParentState,
    transaction: TransactionEvidence,
) -> anyhow::Result<RecoveryAction> {
    use BatchManifestState as Manifest;
    use BatchParentState as Parent;
    use RecoveryAction as Action;
    use TransactionEvidence as Evidence;

    let action = match (parent, transaction) {
        (
            Parent::Queued | Parent::Running | Parent::Retrying,
            Evidence::Active(Manifest::Staging),
        ) => Action::Resume,
        (
            Parent::Prepared,
            Evidence::Active(Manifest::Staging | Manifest::Prepared),
        ) => Action::ConvergeCommit,
        (
            Parent::Committing,
            Evidence::Active(
                Manifest::Prepared | Manifest::Committing | Manifest::Committed,
            ),
        ) => Action::ConvergeCommit,
        (Parent::Committing, Evidence::CommittedArchive) => Action::RollForwardArchive,
        (
            Parent::Cancelling | Parent::Failing,
            Evidence::Active(Manifest::Staging | Manifest::Prepared | Manifest::Failed),
        ) => Action::FenceAndRollback,
        (
            Parent::Fenced,
            Evidence::Active(Manifest::Staging | Manifest::Prepared | Manifest::Failed),
        ) => Action::Rollback,
        (Parent::Fenced, Evidence::Missing) => Action::AlreadyFenced,
        (
            Parent::Cancelled | Parent::Failed,
            Evidence::Active(Manifest::Staging | Manifest::Prepared | Manifest::Failed),
        ) => Action::Rollback,
        (Parent::Cancelled | Parent::Failed, Evidence::Missing) => Action::AlreadyTerminal,
        (Parent::Committed, Evidence::Active(Manifest::Committed)) => Action::AlreadyCommitted,
        (Parent::Committed, Evidence::CommittedArchive) => Action::AlreadyCommitted,
        _ => anyhow::bail!(
            "no durable batch handoff for parent state {parent:?} with transaction evidence {transaction:?}"
        ),
    };
    Ok(action)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveredParentOutcome {
    Resumable {
        parent_id: String,
        attempt_generation: u64,
        state: BatchParentState,
    },
    Fenced {
        parent_id: String,
        attempt_generation: u64,
    },
    Terminal {
        parent_id: String,
        attempt_generation: u64,
        state: BatchParentState,
    },
    Committed {
        parent_id: String,
        attempt_generation: u64,
    },
}

#[derive(Debug)]
pub struct DurableBatchAttempt {
    parent: DurableBatchParent,
    transaction: BatchTransaction,
}

impl DurableBatchAttempt {
    pub fn begin(
        output_dir: &Path,
        parent_id: &str,
        normalized_request: serde_json::Value,
        records: Vec<GenerationRecord>,
    ) -> anyhow::Result<Self> {
        let total_children = records.len();
        let transaction =
            BatchTransaction::begin(output_dir, parent_id, 0, normalized_request, records)?;
        let parent = DurableBatchParent::create(
            &parent_authority_dir(output_dir, parent_id),
            parent_id,
            total_children,
        )
        .with_context(|| {
            format!(
                "creating reducer authority for batch transaction {parent_id}/0; \
                         startup joint recovery must retire the transaction if this fails"
            )
        })?;
        Ok(Self {
            parent,
            transaction,
        })
    }

    pub fn recover(
        output_dir: &Path,
        parent_id: &str,
    ) -> anyhow::Result<(Self, BatchAttemptReconciliation)> {
        let parent_dir = parent_authority_dir(output_dir, parent_id);
        let mut parent = DurableBatchParent::recover_unfenced(&parent_dir)?;
        let generation = parent.attempt_generation();
        let mut transaction = BatchTransaction::recover_exact(output_dir, parent_id, generation)?;
        let report = reconcile_receipts(&mut parent, &mut transaction)?;
        Ok((
            Self {
                parent,
                transaction,
            },
            report,
        ))
    }

    pub fn parent(&self) -> &DurableBatchParent {
        &self.parent
    }

    pub fn transaction(&self) -> &BatchTransaction {
        &self.transaction
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        self.parent.start()
    }

    pub fn grant(
        &mut self,
        child_index: usize,
    ) -> anyhow::Result<(BatchChildLease, mold_inference::InferenceCancellationToken)> {
        self.parent.grant(child_index)
    }

    /// Production ordering is load-bearing: durable transaction receipt first,
    /// reducer acceptance second, prepared checkpoint last.
    pub fn stage_and_accept(
        &mut self,
        lease: &BatchChildLease,
        bytes: &[u8],
    ) -> anyhow::Result<CompletionDisposition> {
        let receipt = self.transaction.stage_bytes_for_lease(lease, bytes)?;
        let disposition = self.parent.complete_staged(lease, receipt)?;
        if disposition == CompletionDisposition::AttemptPrepared {
            self.transaction.mark_prepared()?;
        }
        Ok(disposition)
    }

    pub async fn converge_commit(
        &mut self,
        gate: &GalleryPublicationGate,
        db: Arc<Option<mold_db::MetadataDb>>,
    ) -> anyhow::Result<BatchAttemptReconciliation> {
        let mut report = BatchAttemptReconciliation::default();
        if let (BatchParentState::Prepared, BatchManifestState::Staging) =
            (self.parent.state(), self.transaction.manifest().state)
        {
            self.transaction.mark_prepared()?;
            report.transaction_prepared = true;
        }

        if self.transaction.manifest().state == BatchManifestState::Committed {
            self.transaction.retire_committed_attempt()?;
            roll_parent_to_committed(&mut self.parent)?;
            report.parent_rolled_forward = true;
            return Ok(report);
        }

        if self.parent.state() == BatchParentState::Prepared {
            anyhow::ensure!(
                self.transaction.manifest().state == BatchManifestState::Prepared,
                "prepared parent has no recoverable prepared transaction"
            );
            self.parent.begin_commit()?;
        }
        if self.parent.state() == BatchParentState::Committing {
            anyhow::ensure!(
                matches!(
                    self.transaction.manifest().state,
                    BatchManifestState::Prepared | BatchManifestState::Committing
                ),
                "committing parent has no recoverable transaction"
            );
            self.transaction
                .commit(gate, db)
                .await
                .map_err(|error| error.into_startup_error())?;
            self.parent.mark_committed()?;
            report.parent_rolled_forward = true;
        }
        Ok(report)
    }
}

fn reconcile_receipts(
    parent: &mut DurableBatchParent,
    transaction: &mut BatchTransaction,
) -> anyhow::Result<BatchAttemptReconciliation> {
    anyhow::ensure!(
        parent.parent_id() == transaction.manifest().parent_id
            && parent.attempt_generation() == transaction.manifest().attempt_generation
            && parent.total_children() == transaction.manifest().children.len(),
        "parent and transaction attempt identities do not match"
    );
    let accepted = parent.accepted_receipts()?;
    let unreceipted_files = transaction.remove_unreceipted_staging()?;
    let staged = transaction.staged_receipts().clone();
    for (child_index, expected) in &accepted {
        let actual = staged.get(child_index).with_context(|| {
            format!(
                "reducer accepted child {child_index} without a matching durable staging receipt"
            )
        })?;
        match expected {
            Some(expected) => anyhow::ensure!(
                expected == actual,
                "reducer receipt for child {child_index} does not match transaction receipt"
            ),
            None => anyhow::ensure!(
                transaction.protocol_version() == 1,
                "v2 reducer success for child {child_index} has no bound durable receipt"
            ),
        }
        anyhow::ensure!(
            parent.child_is_succeeded(*child_index),
            "accepted receipt does not correspond to reducer success"
        );
    }

    let mut report = BatchAttemptReconciliation {
        receipts_removed: unreceipted_files,
        ..BatchAttemptReconciliation::default()
    };
    for child_index in staged.keys().copied().collect::<Vec<_>>() {
        if accepted.contains_key(&child_index) {
            continue;
        }
        transaction.unstage_receipt(child_index)?;
        report.receipts_removed += 1;
        if parent.state() == BatchParentState::Running
            && parent.active_children().contains(&child_index)
        {
            parent.requeue_lost_active(child_index)?;
            report.leases_requeued += 1;
        }
    }
    for child_index in parent.active_children() {
        anyhow::ensure!(
            !accepted.contains_key(&child_index),
            "accepted child remains active during recovery"
        );
        if parent.state() == BatchParentState::Running {
            parent.requeue_lost_active(child_index)?;
            report.leases_requeued += 1;
        }
    }

    Ok(report)
}

fn roll_parent_to_committed(parent: &mut DurableBatchParent) -> anyhow::Result<()> {
    match parent.state() {
        BatchParentState::Prepared => {
            parent.begin_commit()?;
            parent.mark_committed()
        }
        BatchParentState::Committing => parent.mark_committed(),
        BatchParentState::Committed => Ok(()),
        state => anyhow::bail!("committed transaction cannot roll parent forward from {state:?}"),
    }
}

/// Roll forward the reducer after a transaction committed and archived before
/// the parent journal recorded `committed`.
pub fn reconcile_archived_commit(
    output_dir: &Path,
    parent_id: &str,
    attempt_generation: u64,
) -> anyhow::Result<BatchAttemptReconciliation> {
    let manifest_path = output_dir
        .join(TRANSACTION_DIR)
        .join(parent_id)
        .join(COMMITTED_DIR)
        .join(format!("{attempt_generation}.json"));
    let manifest = BatchTransaction::load_validated_committed_archive(
        output_dir,
        parent_id,
        attempt_generation,
    )
    .with_context(|| {
        format!(
            "validating archived batch commit {}",
            manifest_path.display()
        )
    })?;
    let mut parent =
        DurableBatchParent::recover_unfenced(&parent_authority_dir(output_dir, parent_id))?;
    anyhow::ensure!(
        parent.parent_id() == manifest.parent_id
            && parent.attempt_generation() == attempt_generation
            && parent.total_children() == manifest.children.len(),
        "archived transaction generation does not match parent"
    );
    for (child_index, receipt) in parent.accepted_receipts()? {
        let Some(receipt) = receipt else {
            anyhow::ensure!(
                manifest.version == 1,
                "v2 parent success has no receipt to validate against committed archive"
            );
            continue;
        };
        let child = &manifest.children[child_index];
        anyhow::ensure!(
            child.checksum_sha256.as_deref() == Some(receipt.checksum_sha256.as_str())
                && child.size_bytes == Some(receipt.size_bytes)
                && BatchTransaction::archived_child_identity(child)?
                    == receipt.record_identity_sha256,
            "archived child {child_index} does not match the reducer receipt"
        );
    }
    let was_committed = parent.state() == BatchParentState::Committed;
    roll_parent_to_committed(&mut parent)?;
    Ok(BatchAttemptReconciliation {
        parent_rolled_forward: !was_committed,
        ..BatchAttemptReconciliation::default()
    })
}

/// Startup entrypoint. Joint parents are discovered and reconciled before the
/// legacy transaction sweep, which explicitly excludes their directories.
/// Existing installations with no parent-authority directory follow the
/// unchanged transaction-only path.
pub async fn recover_batches(
    output_dir: &Path,
    gate: &GalleryPublicationGate,
    db: Arc<Option<mold_db::MetadataDb>>,
) -> anyhow::Result<JointBatchRecoveryReport> {
    std::fs::create_dir_all(output_dir)?;
    let root = output_dir.join(TRANSACTION_DIR);
    let mut parent_ids = Vec::new();
    if root.is_dir() {
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            if !entry.path().is_dir() || !entry.path().join(PARENT_AUTHORITY_DIR).is_dir() {
                continue;
            }
            let parent_id = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("batch parent directory name is not UTF-8"))?;
            parent_ids.push(parent_id);
        }
    }
    parent_ids.sort();

    let mut report = JointBatchRecoveryReport {
        parents_discovered: parent_ids.len(),
        ..JointBatchRecoveryReport::default()
    };
    for parent_id in parent_ids {
        let probe =
            DurableBatchParent::recover_unfenced(&parent_authority_dir(output_dir, &parent_id))?;
        let generation = probe.attempt_generation();
        let state = probe.state();
        drop(probe);
        for stale_generation in attempt_generations(output_dir, &parent_id)? {
            anyhow::ensure!(
                stale_generation <= generation,
                "batch transaction generation {stale_generation} is ahead of parent authority {parent_id}/{generation}"
            );
            if stale_generation == generation {
                continue;
            }
            let mut stale =
                BatchTransaction::recover_exact(output_dir, &parent_id, stale_generation)
                    .with_context(|| {
                        format!("claiming stale batch transaction {parent_id}/{stale_generation}")
                    })?;
            stale.rollback_private_attempt().with_context(|| {
                format!("retiring stale batch transaction {parent_id}/{stale_generation}")
            })?;
            report.stale_attempts_rolled_back += 1;
        }
        for archived_generation in archived_generations(output_dir, &parent_id)? {
            anyhow::ensure!(
                archived_generation == generation,
                "committed archive {parent_id}/{archived_generation} conflicts with parent generation {generation}"
            );
        }
        let archive = output_dir
            .join(TRANSACTION_DIR)
            .join(&parent_id)
            .join(COMMITTED_DIR)
            .join(format!("{generation}.json"));
        let attempt = output_dir
            .join(TRANSACTION_DIR)
            .join(&parent_id)
            .join("attempts")
            .join(generation.to_string());
        if attempt.is_dir() {
            let (mut bridge, reconciled) = DurableBatchAttempt::recover(output_dir, &parent_id)?;
            report.receipts_removed += reconciled.receipts_removed;
            report.leases_requeued += reconciled.leases_requeued;
            let action = recovery_action(
                bridge.parent.state(),
                TransactionEvidence::Active(bridge.transaction.manifest().state),
            )?;
            match action {
                RecoveryAction::Resume => {
                    report.running_attempts_retained += 1;
                    report.outcomes.push(RecoveredParentOutcome::Resumable {
                        parent_id,
                        attempt_generation: generation,
                        state: bridge.parent.state(),
                    });
                }
                RecoveryAction::ConvergeCommit => {
                    let converged = bridge.converge_commit(gate, db.clone()).await?;
                    report.parents_rolled_forward += usize::from(converged.parent_rolled_forward);
                    report.outcomes.push(RecoveredParentOutcome::Committed {
                        parent_id,
                        attempt_generation: generation,
                    });
                }
                RecoveryAction::FenceAndRollback => {
                    bridge.parent.fence_lost_active()?;
                    anyhow::ensure!(
                        bridge.parent.state() == BatchParentState::Fenced,
                        "closed parent did not fence lost leases"
                    );
                    bridge.transaction.rollback_private_attempt()?;
                    report.outcomes.push(RecoveredParentOutcome::Fenced {
                        parent_id,
                        attempt_generation: generation,
                    });
                }
                RecoveryAction::Rollback => {
                    let terminal = bridge.parent.state();
                    bridge.transaction.rollback_private_attempt()?;
                    report
                        .outcomes
                        .push(if terminal == BatchParentState::Fenced {
                            RecoveredParentOutcome::Fenced {
                                parent_id,
                                attempt_generation: generation,
                            }
                        } else {
                            RecoveredParentOutcome::Terminal {
                                parent_id,
                                attempt_generation: generation,
                                state: terminal,
                            }
                        });
                }
                RecoveryAction::AlreadyCommitted => {
                    bridge.transaction.retire_committed_attempt()?;
                    report.outcomes.push(RecoveredParentOutcome::Committed {
                        parent_id,
                        attempt_generation: generation,
                    });
                }
                RecoveryAction::RollForwardArchive
                | RecoveryAction::AlreadyFenced
                | RecoveryAction::AlreadyTerminal => {
                    unreachable!("active transaction cannot select a non-active recovery action")
                }
            }
            continue;
        }
        let evidence = if archive.is_file() {
            TransactionEvidence::CommittedArchive
        } else {
            TransactionEvidence::Missing
        };
        match recovery_action(state, evidence)? {
            RecoveryAction::RollForwardArchive => {
                let reconciled = reconcile_archived_commit(output_dir, &parent_id, generation)?;
                report.parents_rolled_forward += usize::from(reconciled.parent_rolled_forward);
                report.outcomes.push(RecoveredParentOutcome::Committed {
                    parent_id,
                    attempt_generation: generation,
                });
            }
            RecoveryAction::AlreadyCommitted => {
                if evidence == TransactionEvidence::CommittedArchive {
                    reconcile_archived_commit(output_dir, &parent_id, generation)?;
                }
                report.outcomes.push(RecoveredParentOutcome::Committed {
                    parent_id,
                    attempt_generation: generation,
                });
            }
            RecoveryAction::AlreadyFenced => {
                report.outcomes.push(RecoveredParentOutcome::Fenced {
                    parent_id,
                    attempt_generation: generation,
                });
            }
            RecoveryAction::AlreadyTerminal => {
                report.outcomes.push(RecoveredParentOutcome::Terminal {
                    parent_id,
                    attempt_generation: generation,
                    state,
                });
            }
            RecoveryAction::Resume
            | RecoveryAction::ConvergeCommit
            | RecoveryAction::FenceAndRollback
            | RecoveryAction::Rollback => {
                unreachable!("missing transaction cannot select an active recovery action")
            }
        }
    }
    report.legacy_transactions =
        crate::batch_transaction::recover_transactions(output_dir, gate, db).await?;
    Ok(report)
}

fn numeric_children(directory: &Path) -> anyhow::Result<Vec<u64>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut generations = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.path().is_dir() && entry.path().extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("batch generation name is not UTF-8"))?;
        let number = name.strip_suffix(".json").unwrap_or(&name);
        generations.push(number.parse::<u64>().with_context(|| {
            format!(
                "batch generation path is not numeric: {}",
                entry.path().display()
            )
        })?);
    }
    generations.sort_unstable();
    generations.dedup();
    Ok(generations)
}

fn attempt_generations(output_dir: &Path, parent_id: &str) -> anyhow::Result<Vec<u64>> {
    numeric_children(
        &output_dir
            .join(TRANSACTION_DIR)
            .join(parent_id)
            .join("attempts"),
    )
}

fn archived_generations(output_dir: &Path, parent_id: &str) -> anyhow::Result<Vec<u64>> {
    numeric_children(
        &output_dir
            .join(TRANSACTION_DIR)
            .join(parent_id)
            .join(COMMITTED_DIR),
    )
}

fn parent_authority_dir(output_dir: &Path, parent_id: &str) -> PathBuf {
    output_dir
        .join(TRANSACTION_DIR)
        .join(parent_id)
        .join(PARENT_AUTHORITY_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
    use mold_db::RecordSource;

    fn record(name: &str, seed: u64, count: u32) -> GenerationRecord {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": format!("prompt {seed}"),
            "model": "test",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0,
            "batch_id": "parent",
            "batch_index": seed + 1,
            "batch_count": count
        }))
        .unwrap();
        GenerationRecord::from_save(
            Path::new("replaced"),
            name,
            OutputFormat::Png,
            OutputMetadata::from_generate_request(&request, seed, None, "test"),
            RecordSource::Server,
            1,
        )
    }

    #[test]
    fn parent_state_by_transaction_evidence_matrix_is_exhaustive() {
        use BatchManifestState as Manifest;
        use BatchParentState as Parent;
        use RecoveryAction as Action;
        use TransactionEvidence as Evidence;

        let parents = [
            Parent::Queued,
            Parent::Running,
            Parent::Prepared,
            Parent::Committing,
            Parent::Committed,
            Parent::Cancelling,
            Parent::Failing,
            Parent::Fenced,
            Parent::Retrying,
            Parent::Cancelled,
            Parent::Failed,
        ];
        let evidence = [
            Evidence::Active(Manifest::Staging),
            Evidence::Active(Manifest::Prepared),
            Evidence::Active(Manifest::Committing),
            Evidence::Active(Manifest::Committed),
            Evidence::Active(Manifest::Failed),
            Evidence::CommittedArchive,
            Evidence::Missing,
        ];
        let legal = [
            (
                Parent::Queued,
                Evidence::Active(Manifest::Staging),
                Action::Resume,
            ),
            (
                Parent::Running,
                Evidence::Active(Manifest::Staging),
                Action::Resume,
            ),
            (
                Parent::Retrying,
                Evidence::Active(Manifest::Staging),
                Action::Resume,
            ),
            (
                Parent::Prepared,
                Evidence::Active(Manifest::Staging),
                Action::ConvergeCommit,
            ),
            (
                Parent::Prepared,
                Evidence::Active(Manifest::Prepared),
                Action::ConvergeCommit,
            ),
            (
                Parent::Committing,
                Evidence::Active(Manifest::Prepared),
                Action::ConvergeCommit,
            ),
            (
                Parent::Committing,
                Evidence::Active(Manifest::Committing),
                Action::ConvergeCommit,
            ),
            (
                Parent::Committing,
                Evidence::Active(Manifest::Committed),
                Action::ConvergeCommit,
            ),
            (
                Parent::Committing,
                Evidence::CommittedArchive,
                Action::RollForwardArchive,
            ),
            (
                Parent::Committed,
                Evidence::Active(Manifest::Committed),
                Action::AlreadyCommitted,
            ),
            (
                Parent::Committed,
                Evidence::CommittedArchive,
                Action::AlreadyCommitted,
            ),
            (
                Parent::Cancelling,
                Evidence::Active(Manifest::Staging),
                Action::FenceAndRollback,
            ),
            (
                Parent::Cancelling,
                Evidence::Active(Manifest::Prepared),
                Action::FenceAndRollback,
            ),
            (
                Parent::Cancelling,
                Evidence::Active(Manifest::Failed),
                Action::FenceAndRollback,
            ),
            (
                Parent::Failing,
                Evidence::Active(Manifest::Staging),
                Action::FenceAndRollback,
            ),
            (
                Parent::Failing,
                Evidence::Active(Manifest::Prepared),
                Action::FenceAndRollback,
            ),
            (
                Parent::Failing,
                Evidence::Active(Manifest::Failed),
                Action::FenceAndRollback,
            ),
            (
                Parent::Fenced,
                Evidence::Active(Manifest::Staging),
                Action::Rollback,
            ),
            (
                Parent::Fenced,
                Evidence::Active(Manifest::Prepared),
                Action::Rollback,
            ),
            (
                Parent::Fenced,
                Evidence::Active(Manifest::Failed),
                Action::Rollback,
            ),
            (Parent::Fenced, Evidence::Missing, Action::AlreadyFenced),
            (
                Parent::Cancelled,
                Evidence::Active(Manifest::Staging),
                Action::Rollback,
            ),
            (
                Parent::Cancelled,
                Evidence::Active(Manifest::Prepared),
                Action::Rollback,
            ),
            (
                Parent::Cancelled,
                Evidence::Active(Manifest::Failed),
                Action::Rollback,
            ),
            (
                Parent::Cancelled,
                Evidence::Missing,
                Action::AlreadyTerminal,
            ),
            (
                Parent::Failed,
                Evidence::Active(Manifest::Staging),
                Action::Rollback,
            ),
            (
                Parent::Failed,
                Evidence::Active(Manifest::Prepared),
                Action::Rollback,
            ),
            (
                Parent::Failed,
                Evidence::Active(Manifest::Failed),
                Action::Rollback,
            ),
            (Parent::Failed, Evidence::Missing, Action::AlreadyTerminal),
        ];

        for parent in parents {
            for transaction in evidence {
                let expected = legal.iter().find_map(|(legal_parent, legal_tx, action)| {
                    (*legal_parent == parent && *legal_tx == transaction).then_some(*action)
                });
                let actual = recovery_action(parent, transaction);
                match expected {
                    Some(expected) => assert_eq!(actual.unwrap(), expected),
                    None => assert!(
                        actual.is_err(),
                        "unexpected legal handoff for {parent:?} × {transaction:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn unaccepted_receipt_is_removed_and_lease_requeued() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        bridge
            .transaction
            .stage_bytes_for_lease(&lease, b"staged")
            .unwrap();
        let staged = bridge.transaction.staging_path(0).unwrap();
        drop(bridge);

        let (recovered, report) = DurableBatchAttempt::recover(dir.path(), "parent").unwrap();

        assert_eq!(report.receipts_removed, 1);
        assert_eq!(report.leases_requeued, 1);
        assert!(!staged.exists());
        assert_eq!(recovered.parent.state(), BatchParentState::Running);
        recovered.transaction.staging_path(0).unwrap();
    }

    #[test]
    fn reducer_success_without_durable_receipt_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        bridge
            .parent
            .complete(&lease, crate::batch_parent::ChildCompletion::Succeeded)
            .unwrap();
        drop(bridge);

        let error = DurableBatchAttempt::recover(dir.path(), "parent").unwrap_err();

        assert!(format!("{error:#}").contains("without a matching durable staging receipt"));
    }

    #[tokio::test]
    async fn prepared_parent_and_transaction_commit_converge() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        assert_eq!(
            bridge.stage_and_accept(&lease, b"one").unwrap(),
            CompletionDisposition::AttemptPrepared
        );
        bridge
            .converge_commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        assert_eq!(bridge.parent.state(), BatchParentState::Committed);
        assert_eq!(std::fs::read(dir.path().join("one.png")).unwrap(), b"one");
    }

    #[test]
    fn stage_file_without_receipt_is_removed_and_requeued() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        bridge.grant(0).unwrap();
        let staged = bridge.transaction.staging_path(0).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged)
            .unwrap();
        use std::io::Write as _;
        file.write_all(b"crash-gap").unwrap();
        file.sync_all().unwrap();
        drop(bridge);

        let (recovered, report) = DurableBatchAttempt::recover(dir.path(), "parent").unwrap();

        assert_eq!(report.receipts_removed, 1);
        assert_eq!(report.leases_requeued, 1);
        assert!(!staged.exists());
        assert_eq!(recovered.parent.state(), BatchParentState::Running);
    }

    #[test]
    fn incomplete_child_staged_delta_is_discarded_before_private_file_cleanup() {
        use std::io::Write as _;

        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        bridge.grant(0).unwrap();
        let staged = bridge.transaction.staging_path(0).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged)
            .unwrap();
        file.write_all(b"crash-gap").unwrap();
        file.sync_all().unwrap();
        let journal = dir
            .path()
            .join(TRANSACTION_DIR)
            .join("parent/attempts/0/journal.jsonl");
        std::fs::OpenOptions::new()
            .append(true)
            .open(journal)
            .unwrap()
            .write_all(b"{\"sequence\":1,\"attempt_generation\":0,\"event\":{\"kind\":\"child")
            .unwrap();
        drop(bridge);

        let (recovered, report) = DurableBatchAttempt::recover(dir.path(), "parent").unwrap();

        assert_eq!(report.receipts_removed, 1);
        assert_eq!(report.leases_requeued, 1);
        assert!(!staged.exists());
        assert_eq!(recovered.parent.state(), BatchParentState::Running);
    }

    #[test]
    fn incomplete_parent_accept_delta_discards_durable_receipt_and_requeues() {
        use std::io::Write as _;

        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        bridge
            .transaction
            .stage_bytes_for_lease(&lease, b"staged")
            .unwrap();
        let parent_journal =
            parent_authority_dir(dir.path(), "parent").join("parent-journal.jsonl");
        std::fs::OpenOptions::new()
            .append(true)
            .open(parent_journal)
            .unwrap()
            .write_all(b"{\"version\":2,\"sequence\":3,\"attempt_generation\":0,\"event\":")
            .unwrap();
        let staged = bridge.transaction.staging_path(0).unwrap();
        drop(bridge);

        let (recovered, report) = DurableBatchAttempt::recover(dir.path(), "parent").unwrap();

        assert_eq!(report.receipts_removed, 1);
        assert_eq!(report.leases_requeued, 1);
        assert!(!staged.exists());
        assert_eq!(recovered.parent.state(), BatchParentState::Running);
    }

    #[tokio::test]
    async fn startup_discovers_joint_receipt_before_legacy_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        bridge
            .transaction
            .stage_bytes_for_lease(&lease, b"staged")
            .unwrap();
        drop(bridge);

        let report = recover_batches(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.parents_discovered, 1);
        assert_eq!(report.receipts_removed, 1);
        assert_eq!(report.leases_requeued, 1);
        assert_eq!(report.legacy_transactions, RecoveryReport::default());
        assert_eq!(report.running_attempts_retained, 1);
    }

    #[tokio::test]
    async fn archived_commit_is_discovered_and_rolls_parent_forward() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        bridge.stage_and_accept(&lease, b"one").unwrap();
        bridge.parent.begin_commit().unwrap();
        bridge
            .transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        let parent_dir = parent_authority_dir(dir.path(), "parent");
        assert!(
            parent_dir.is_dir(),
            "transaction cleanup removed parent authority"
        );
        assert_eq!(bridge.parent.state(), BatchParentState::Committing);
        drop(bridge);

        let report = recover_batches(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.parents_discovered, 1);
        assert_eq!(report.parents_rolled_forward, 1);
        assert_eq!(
            DurableBatchParent::recover(&parent_dir).unwrap().state(),
            BatchParentState::Committed
        );
    }

    #[tokio::test]
    async fn archived_commit_with_changed_final_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("one.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (lease, _) = bridge.grant(0).unwrap();
        bridge.stage_and_accept(&lease, b"one").unwrap();
        bridge.parent.begin_commit().unwrap();
        bridge
            .transaction
            .commit(&GalleryPublicationGate::default(), Arc::new(None))
            .await
            .unwrap();
        std::fs::write(dir.path().join("one.png"), b"tampered").unwrap();
        drop(bridge);

        let error = recover_batches(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap_err();

        assert!(format!("{error:#}").contains("changed"));
    }

    #[tokio::test]
    async fn retry_recovery_retires_only_stale_generation_attempt() {
        let dir = tempfile::tempdir().unwrap();
        let mut bridge = DurableBatchAttempt::begin(
            dir.path(),
            "parent",
            serde_json::json!({}),
            vec![record("old.png", 0, 1)],
        )
        .unwrap();
        bridge.start().unwrap();
        let (first, _) = bridge.grant(0).unwrap();
        bridge
            .parent
            .complete(&first, crate::batch_parent::ChildCompletion::Failed)
            .unwrap();
        let (retry, _) = bridge.grant(0).unwrap();
        bridge
            .parent
            .complete(&retry, crate::batch_parent::ChildCompletion::Failed)
            .unwrap();
        bridge.parent.begin_retry().unwrap();
        let current = BatchTransaction::begin(
            dir.path(),
            "parent",
            1,
            serde_json::json!({}),
            vec![record("new.png", 0, 1)],
        )
        .unwrap();
        drop(bridge);
        drop(current);

        let report = recover_batches(
            dir.path(),
            &GalleryPublicationGate::default(),
            Arc::new(None),
        )
        .await
        .unwrap();

        assert_eq!(report.stale_attempts_rolled_back, 1);
        assert!(!dir
            .path()
            .join(TRANSACTION_DIR)
            .join("parent/attempts/0")
            .exists());
        assert!(dir
            .path()
            .join(TRANSACTION_DIR)
            .join("parent/attempts/1")
            .is_dir());
        assert!(matches!(
            report.outcomes.as_slice(),
            [RecoveredParentOutcome::Resumable {
                attempt_generation: 1,
                state: BatchParentState::Retrying,
                ..
            }]
        ));
    }

    #[tokio::test]
    async fn startup_state_matrix_has_an_explicit_outcome_for_every_non_commit_state() {
        #[derive(Clone, Copy)]
        enum Case {
            Queued,
            Running,
            Retrying,
            Cancelling,
            Failing,
            Fenced,
            Cancelled,
            Failed,
        }

        for case in [
            Case::Queued,
            Case::Running,
            Case::Retrying,
            Case::Cancelling,
            Case::Failing,
            Case::Fenced,
            Case::Cancelled,
            Case::Failed,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let child_count = if matches!(case, Case::Failing) { 2 } else { 1 };
            let records = (0..child_count)
                .map(|index| record(&format!("{index}.png"), index as u64, child_count))
                .collect();
            let mut bridge =
                DurableBatchAttempt::begin(dir.path(), "parent", serde_json::json!({}), records)
                    .unwrap();
            match case {
                Case::Queued => {}
                Case::Running => bridge.start().unwrap(),
                Case::Retrying => {
                    bridge.start().unwrap();
                    let (first, _) = bridge.grant(0).unwrap();
                    bridge
                        .parent
                        .complete(&first, crate::batch_parent::ChildCompletion::Failed)
                        .unwrap();
                    let (retry, _) = bridge.grant(0).unwrap();
                    bridge
                        .parent
                        .complete(&retry, crate::batch_parent::ChildCompletion::Failed)
                        .unwrap();
                    bridge.transaction.rollback_private_attempt().unwrap();
                    bridge.parent.begin_retry().unwrap();
                    bridge.transaction = BatchTransaction::begin(
                        dir.path(),
                        "parent",
                        1,
                        serde_json::json!({}),
                        vec![record("retry.png", 0, 1)],
                    )
                    .unwrap();
                }
                Case::Cancelling => {
                    bridge.start().unwrap();
                    bridge.grant(0).unwrap();
                    bridge.parent.request_cancel().unwrap();
                }
                Case::Failing => {
                    bridge.start().unwrap();
                    let (failed, _) = bridge.grant(0).unwrap();
                    bridge.grant(1).unwrap();
                    bridge
                        .parent
                        .complete(&failed, crate::batch_parent::ChildCompletion::Failed)
                        .unwrap();
                    let (retry, _) = bridge.grant(0).unwrap();
                    bridge
                        .parent
                        .complete(&retry, crate::batch_parent::ChildCompletion::Failed)
                        .unwrap();
                }
                Case::Fenced => {
                    bridge.parent.request_cancel().unwrap();
                }
                Case::Cancelled => {
                    bridge.parent.request_cancel().unwrap();
                    bridge.parent.finalize_fence().unwrap();
                }
                Case::Failed => {
                    bridge.start().unwrap();
                    let (failed, _) = bridge.grant(0).unwrap();
                    bridge
                        .parent
                        .complete(&failed, crate::batch_parent::ChildCompletion::Failed)
                        .unwrap();
                    let (retry, _) = bridge.grant(0).unwrap();
                    bridge
                        .parent
                        .complete(&retry, crate::batch_parent::ChildCompletion::Failed)
                        .unwrap();
                    bridge.parent.finalize_fence().unwrap();
                }
            }
            drop(bridge);

            let report = recover_batches(
                dir.path(),
                &GalleryPublicationGate::default(),
                Arc::new(None),
            )
            .await
            .unwrap();

            assert_eq!(report.parents_discovered, 1);
            assert_eq!(report.outcomes.len(), 1);
            match case {
                Case::Queued | Case::Running | Case::Retrying => assert!(matches!(
                    report.outcomes[0],
                    RecoveredParentOutcome::Resumable { .. }
                )),
                Case::Cancelling | Case::Failing | Case::Fenced => assert!(matches!(
                    report.outcomes[0],
                    RecoveredParentOutcome::Fenced { .. }
                )),
                Case::Cancelled | Case::Failed => assert!(matches!(
                    report.outcomes[0],
                    RecoveredParentOutcome::Terminal { .. }
                )),
            }
        }
    }
}
