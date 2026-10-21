//! Trusted internal stage ownership. Never reconstructed from prompts or tags.

use anyhow::{ensure, Result};
use rusqlite::{params, Connection, OptionalExtension};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainStageOwner {
    pub chain_id: String,
    pub stage_index: u32,
}

/// Correlated with `generation_queue.id`; evaluated in the same statement as
/// each claim/window/dispatch transition. A deleted owner always refuses.
pub(crate) const EXECUTABLE_SQL: &str = "NOT EXISTS (
 SELECT 1 FROM chain_stage_generations owned
 LEFT JOIN chain_jobs chain ON chain.id=owned.chain_id
 LEFT JOIN video_jobs parent ON parent.id=owned.video_parent_id
 WHERE owned.generation_id=generation_queue.id AND (
  chain.id IS NULL OR chain.state!='running' OR
  (owned.video_parent_id IS NOT NULL AND (parent.id IS NULL OR parent.state!='running' OR parent.cancel_requested!=0))
 ))";

pub(crate) fn record_on_conn(
    conn: &Connection,
    generation_id: &str,
    owner: &ChainStageOwner,
) -> Result<()> {
    let running: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM chain_jobs WHERE id=?1 AND state='running' AND stage_count>?2)",params![owner.chain_id,owner.stage_index],|row|row.get(0))?;
    ensure!(running, "chain stage owner is not running");
    let parent = crate::video_job_child_owners::parent_id_on_conn(conn, &owner.chain_id)?;
    if let Some(parent) = &parent {
        let runnable: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM video_jobs WHERE id=?1 AND state='running' AND cancel_requested=0)",[parent],|row|row.get(0))?;
        ensure!(runnable, "video parent is not running");
    }
    conn.execute("INSERT INTO chain_stage_generations(generation_id,chain_id,stage_index,video_parent_id) VALUES(?1,?2,?3,?4)",params![generation_id,owner.chain_id,owner.stage_index,parent])?;
    Ok(())
}

pub(crate) fn matches_on_conn(
    conn: &Connection,
    generation_id: &str,
    owner: Option<&ChainStageOwner>,
) -> Result<bool> {
    let stored: Option<(String, u32)> = conn
        .query_row(
            "SELECT chain_id,stage_index FROM chain_stage_generations WHERE generation_id=?1",
            [generation_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    Ok(match (stored, owner) {
        (None, None) => true,
        (Some((id, index)), Some(owner)) => id == owner.chain_id && index == owner.stage_index,
        _ => false,
    })
}

pub fn require_batch_owner(
    db: &crate::MetadataDb,
    generation_ids: &[String],
    owner: Option<&ChainStageOwner>,
) -> Result<()> {
    db.with_conn(|conn| {
        for id in generation_ids {
            ensure!(
                matches_on_conn(conn, id, owner)?,
                "existing generation batch has different chain stage ownership"
            );
        }
        Ok(())
    })
}

/// Payload-free authority for Queue/Activity projection and public controls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredChainStageOwner {
    pub chain_id: String,
    pub stage_index: u32,
    pub video_parent_id: Option<String>,
}

pub fn owner_for_generation(
    db: &crate::MetadataDb,
    generation_id: &str,
) -> Result<Option<StoredChainStageOwner>> {
    db.with_conn(|conn| conn.query_row(
        "SELECT chain_id,stage_index,video_parent_id FROM chain_stage_generations WHERE generation_id=?1",
        [generation_id], |row| Ok(StoredChainStageOwner {chain_id:row.get(0)?,stage_index:row.get(1)?,video_parent_id:row.get(2)?})
    ).optional().map_err(Into::into))
}
