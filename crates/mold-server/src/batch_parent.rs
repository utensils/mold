//! Serialized state reducer for one server-owned batch parent.
//!
//! The reducer is deliberately free of filesystem, SQLite, scheduler, and
//! async concerns. Callers serialize access with the parent actor lock and
//! execute returned cleanup/publication actions outside that lock.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionDisposition {
    Accepted,
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

#[derive(Debug)]
pub struct BatchParentReducer {
    parent_id: String,
    total_children: usize,
    attempt_generation: u64,
    state: BatchParentState,
    children: Vec<ChildState>,
    active: BTreeSet<usize>,
    terminal_after_fence: Option<BatchParentState>,
}

impl BatchParentReducer {
    pub fn new(parent_id: impl Into<String>, total_children: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            total_children > 0,
            "batch parent must have at least one child"
        );
        Ok(Self {
            parent_id: parent_id.into(),
            total_children,
            attempt_generation: 0,
            state: BatchParentState::Queued,
            children: vec![ChildState::Pending; total_children],
            active: BTreeSet::new(),
            terminal_after_fence: None,
        })
    }

    pub fn state(&self) -> BatchParentState {
        self.state
    }

    pub fn attempt_generation(&self) -> u64 {
        self.attempt_generation
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
        Ok(BatchChildLease {
            parent_id: self.parent_id.clone(),
            child_index,
            attempt_generation: self.attempt_generation,
        })
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
        anyhow::ensure!(
            self.children[lease.child_index] == ChildState::Active
                && self.active.remove(&lease.child_index),
            "batch child {} has no active lease in generation {}",
            lease.child_index,
            lease.attempt_generation
        );

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
                BatchParentState::Running | BatchParentState::Prepared
            ),
            "batch parent cannot cancel from {:?}",
            self.state
        );
        self.state = BatchParentState::Cancelling;
        self.terminal_after_fence = Some(BatchParentState::Cancelled);
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
        self.active.clear();
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
                self.state = BatchParentState::Failing;
                self.terminal_after_fence = Some(BatchParentState::Failed);
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

        assert_eq!(
            parent.complete(&failed, ChildCompletion::Failed).unwrap(),
            CompletionDisposition::Accepted
        );
        assert_eq!(parent.state(), BatchParentState::Failing);
        assert!(
            parent.grant(0).is_err(),
            "closed attempt must reject grants"
        );
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

        assert_eq!(
            parent.request_cancel().unwrap(),
            CompletionDisposition::Accepted
        );
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
}
