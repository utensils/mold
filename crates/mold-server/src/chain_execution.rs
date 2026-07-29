use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub(crate) const EXECUTION_AUTHORITY_SCHEMA: &str = "mold.chain-execution.v1";
pub(crate) const EXECUTION_AUTHORITY_FILE: &str = "execution-authority.json";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChainExecutionState {
    Dormant,
    Ready,
    Submitted,
    Leased,
    Checkpointing,
    Finalizing,
    Settled,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChainWorkIdentity {
    pub parent_id: String,
    pub attempt_generation: u64,
    pub stage_index: u32,
    pub work_id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChainExecutionAuthority {
    pub schema: String,
    pub revision: u64,
    pub state: ChainExecutionState,
    pub identity: ChainWorkIdentity,
    pub preferred_ordinal: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecoveryFacts {
    pub stage_count: u32,
    pub first_incomplete_stage: Option<u32>,
    pub finalized: bool,
    pub terminal: bool,
}

impl ChainExecutionAuthority {
    pub(crate) fn dormant(parent_id: impl Into<String>) -> Self {
        let parent_id = parent_id.into();
        Self {
            schema: EXECUTION_AUTHORITY_SCHEMA.to_string(),
            revision: 0,
            state: ChainExecutionState::Dormant,
            identity: identity(parent_id, 0, 0),
            preferred_ordinal: None,
        }
    }

    pub(crate) fn transition(&mut self, next: ChainExecutionState) -> Result<()> {
        use ChainExecutionState::{
            Checkpointing, Dormant, Finalizing, Leased, Ready, Settled, Submitted,
        };
        let legal = matches!(
            (self.state, next),
            (Dormant, Ready | Finalizing | Settled)
                | (Ready, Submitted | Settled)
                | (Submitted, Leased | Ready | Settled)
                | (Leased, Checkpointing | Ready | Settled)
                | (Checkpointing, Ready | Finalizing | Settled)
                | (Finalizing, Settled)
        );
        if !legal {
            bail!(
                "illegal chain execution transition {:?} -> {:?}",
                self.state,
                next
            );
        }
        if self.state == Checkpointing && matches!(next, Ready | Finalizing) {
            self.identity.stage_index = self.identity.stage_index.saturating_add(1);
            self.refresh_work_id();
        }
        self.state = next;
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub(crate) fn recover(&mut self, facts: RecoveryFacts) -> Result<()> {
        self.validate()?;
        let incomplete_stage = facts
            .first_incomplete_stage
            .filter(|stage| *stage < facts.stage_count);
        if facts.terminal {
            self.state = ChainExecutionState::Settled;
        } else if let Some(incomplete_stage) = incomplete_stage {
            if matches!(
                self.state,
                ChainExecutionState::Submitted
                    | ChainExecutionState::Leased
                    | ChainExecutionState::Checkpointing
                    | ChainExecutionState::Finalizing
                    | ChainExecutionState::Settled
            ) {
                self.identity.attempt_generation =
                    self.identity.attempt_generation.saturating_add(1);
            }
            self.identity.stage_index = incomplete_stage;
            self.state = ChainExecutionState::Ready;
        } else if facts.finalized {
            self.state = ChainExecutionState::Settled;
        } else {
            self.state = ChainExecutionState::Finalizing;
            self.identity.stage_index = facts.stage_count;
        }
        self.refresh_work_id();
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub(crate) fn persist_atomic(&self, job_dir: &Path) -> Result<()> {
        self.validate()?;
        std::fs::create_dir_all(job_dir)
            .with_context(|| format!("creating chain job directory '{}'", job_dir.display()))?;
        let final_path = authority_path(job_dir);
        let temp_path = job_dir.join(format!(
            ".{EXECUTION_AUTHORITY_FILE}.{}.{}.tmp",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let bytes = serde_json::to_vec_pretty(self)?;
        let write_result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp_path)
                .with_context(|| {
                    format!(
                        "creating chain execution authority temporary file '{}'",
                        temp_path.display()
                    )
                })?;
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            std::fs::rename(&temp_path, &final_path).with_context(|| {
                format!(
                    "publishing chain execution authority '{}'",
                    final_path.display()
                )
            })?;
            sync_directory(job_dir)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        write_result
    }

    pub(crate) fn read(job_dir: &Path) -> Result<Self> {
        let path = authority_path(job_dir);
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading chain execution authority '{}'", path.display()))?;
        let authority: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing chain execution authority '{}'", path.display()))?;
        authority.validate()?;
        Ok(authority)
    }

    pub(crate) fn read_for_parent(job_dir: &Path, parent_id: &str) -> Result<Self> {
        let authority = Self::read(job_dir)?;
        if authority.identity.parent_id != parent_id {
            bail!(
                "chain execution authority parent '{}' does not match job '{}'",
                authority.identity.parent_id,
                parent_id
            );
        }
        Ok(authority)
    }

    pub(crate) fn set_stage(&mut self, stage_index: u32) {
        self.identity.stage_index = stage_index;
        self.refresh_work_id();
        self.revision = self.revision.saturating_add(1);
    }

    pub(crate) fn set_preferred_ordinal(&mut self, ordinal: Option<usize>) {
        self.preferred_ordinal = ordinal;
        self.revision = self.revision.saturating_add(1);
    }

    fn refresh_work_id(&mut self) {
        self.identity.work_id = work_id(
            &self.identity.parent_id,
            self.identity.attempt_generation,
            self.identity.stage_index,
        );
    }

    fn validate(&self) -> Result<()> {
        if self.schema != EXECUTION_AUTHORITY_SCHEMA {
            bail!(
                "unsupported chain execution authority schema '{}'",
                self.schema
            );
        }
        let expected = work_id(
            &self.identity.parent_id,
            self.identity.attempt_generation,
            self.identity.stage_index,
        );
        if self.identity.work_id != expected {
            bail!(
                "chain execution authority work id '{}' does not match identity '{}'",
                self.identity.work_id,
                expected
            );
        }
        Ok(())
    }
}

pub(crate) fn authority_path(job_dir: &Path) -> PathBuf {
    job_dir.join(EXECUTION_AUTHORITY_FILE)
}

fn identity(parent_id: String, attempt_generation: u64, stage_index: u32) -> ChainWorkIdentity {
    let work_id = work_id(&parent_id, attempt_generation, stage_index);
    ChainWorkIdentity {
        parent_id,
        attempt_generation,
        stage_index,
        work_id,
    }
}

fn work_id(parent_id: &str, attempt_generation: u64, stage_index: u32) -> String {
    format!("chain:{parent_id}:attempt:{attempt_generation}:stage:{stage_index}")
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)
            .with_context(|| format!("opening directory '{}' for fsync", path.display()))?
            .sync_all()
            .with_context(|| format!("fsync directory '{}'", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn reducer_accepts_only_the_declared_actor_lifecycle() {
        let mut authority = ChainExecutionAuthority::dormant("parent");
        for next in [
            ChainExecutionState::Ready,
            ChainExecutionState::Submitted,
            ChainExecutionState::Leased,
            ChainExecutionState::Checkpointing,
            ChainExecutionState::Ready,
            ChainExecutionState::Submitted,
            ChainExecutionState::Leased,
            ChainExecutionState::Checkpointing,
            ChainExecutionState::Finalizing,
            ChainExecutionState::Settled,
        ] {
            authority.transition(next).unwrap();
        }
        assert_eq!(authority.state, ChainExecutionState::Settled);
        assert!(authority
            .transition(ChainExecutionState::Ready)
            .unwrap_err()
            .to_string()
            .contains("illegal"));
    }

    #[test]
    fn restart_fences_active_attempts_and_rebuilds_the_next_ready_identity() {
        for state in [
            ChainExecutionState::Submitted,
            ChainExecutionState::Leased,
            ChainExecutionState::Checkpointing,
        ] {
            let mut authority = ChainExecutionAuthority::dormant("parent");
            authority.transition(ChainExecutionState::Ready).unwrap();
            authority
                .transition(ChainExecutionState::Submitted)
                .unwrap();
            if state != ChainExecutionState::Submitted {
                authority.transition(ChainExecutionState::Leased).unwrap();
            }
            if state == ChainExecutionState::Checkpointing {
                authority
                    .transition(ChainExecutionState::Checkpointing)
                    .unwrap();
            }
            authority
                .recover(RecoveryFacts {
                    stage_count: 4,
                    first_incomplete_stage: Some(2),
                    finalized: false,
                    terminal: false,
                })
                .unwrap();
            assert_eq!(authority.state, ChainExecutionState::Ready);
            assert_eq!(authority.identity.attempt_generation, 1);
            assert_eq!(authority.identity.stage_index, 2);
            assert!(authority.identity.work_id.contains(":attempt:1:stage:2"));
        }
    }

    #[test]
    fn persistence_round_trip_keeps_schema_and_identity() {
        let root = tempfile::tempdir().unwrap();
        let mut authority = ChainExecutionAuthority::dormant("parent");
        authority.transition(ChainExecutionState::Ready).unwrap();
        authority.persist_atomic(root.path()).unwrap();
        let restored = ChainExecutionAuthority::read(root.path()).unwrap();
        assert_eq!(restored, authority);
        assert_eq!(restored.schema, EXECUTION_AUTHORITY_SCHEMA);
        assert!(root.path().join(EXECUTION_AUTHORITY_FILE).is_file());
    }

    #[test]
    fn persisted_authority_is_bound_to_its_parent_directory_owner() {
        let root = tempfile::tempdir().unwrap();
        ChainExecutionAuthority::dormant("parent-a")
            .persist_atomic(root.path())
            .unwrap();

        let error = ChainExecutionAuthority::read_for_parent(root.path(), "parent-b").unwrap_err();
        assert!(error.to_string().contains("does not match job 'parent-b'"));
    }

    #[test]
    fn finalizing_identity_advances_beyond_the_last_checkpointed_stage() {
        let mut authority = ChainExecutionAuthority::dormant("parent");
        for next in [
            ChainExecutionState::Ready,
            ChainExecutionState::Submitted,
            ChainExecutionState::Leased,
            ChainExecutionState::Checkpointing,
            ChainExecutionState::Finalizing,
        ] {
            authority.transition(next).unwrap();
        }

        assert_eq!(authority.identity.stage_index, 1);
        assert_eq!(authority.identity.work_id, "chain:parent:attempt:0:stage:1");
    }

    #[test]
    fn reopening_a_settled_parent_uses_a_new_attempt_generation() {
        let mut authority = ChainExecutionAuthority::dormant("parent");
        authority.transition(ChainExecutionState::Settled).unwrap();
        authority
            .recover(RecoveryFacts {
                stage_count: 3,
                first_incomplete_stage: Some(1),
                finalized: false,
                terminal: false,
            })
            .unwrap();
        assert_eq!(authority.state, ChainExecutionState::Ready);
        assert_eq!(authority.identity.attempt_generation, 1);
        assert_eq!(authority.identity.stage_index, 1);
    }

    #[test]
    fn historical_finalize_never_overrides_live_retake_work() {
        let mut authority = ChainExecutionAuthority::dormant("parent");
        authority.transition(ChainExecutionState::Settled).unwrap();
        authority
            .recover(RecoveryFacts {
                stage_count: 3,
                first_incomplete_stage: Some(1),
                finalized: true,
                terminal: false,
            })
            .unwrap();

        assert_eq!(authority.state, ChainExecutionState::Ready);
        assert_eq!(authority.identity.attempt_generation, 1);
        assert_eq!(authority.identity.stage_index, 1);
        assert_eq!(authority.identity.work_id, "chain:parent:attempt:1:stage:1");
    }

    #[test]
    fn actor_identity_has_no_fixed_inventory_limit() {
        for count in [1_u32, 2, 8, 64] {
            let identities = (0..count)
                .map(|index| {
                    let mut authority = ChainExecutionAuthority::dormant(format!("parent-{index}"));
                    authority.transition(ChainExecutionState::Ready).unwrap();
                    authority.identity.work_id
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(identities.len(), count as usize);
        }
    }
}
