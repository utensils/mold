//! Immutable scalar ownership survives parent/part cleanup. UUIDs never turn
//! into ordinary chain authority merely because their parent was deleted.
use crate::MetadataDb;
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};

pub(crate) const EXECUTABLE_SQL: &str = "NOT EXISTS(SELECT 1 FROM video_job_child_owners owner LEFT JOIN video_jobs parent ON parent.id=owner.parent_id WHERE owner.child_id=chain_jobs.id AND (parent.id IS NULL OR parent.state!='running' OR parent.cancel_requested!=0))";

pub fn parent_id(db: &MetadataDb, child_id: &str) -> Result<Option<String>> {
    db.with_conn(|conn| parent_id_on_conn(conn, child_id))
}

pub(crate) fn parent_id_on_conn(conn: &Connection, child_id: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT parent_id FROM video_job_child_owners WHERE child_id=?1",
        [child_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

pub fn execution_allowed(db: &MetadataDb, child_id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        conn.query_row("SELECT NOT EXISTS(SELECT 1 FROM video_job_child_owners owner LEFT JOIN video_jobs parent ON parent.id=owner.parent_id WHERE owner.child_id=?1 AND (parent.id IS NULL OR parent.state!='running' OR parent.cancel_requested!=0))",[child_id],|row|row.get(0)).map_err(Into::into)
    })
}

/// Final transactional gate for global ordinary-queue actions. The route's
/// earlier actionable refusal is not sufficient when a child arrives in flight.
pub(crate) fn require_ordinary_bulk_on_conn(
    conn: &Connection,
    owner_uuid: &str,
    resume_chains: bool,
) -> Result<()> {
    let owned:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM chain_stage_generations owned JOIN generation_queue q ON q.id=owned.generation_id WHERE q.owner_uuid=?1 AND q.state IN ('queued','paused')) OR (?2 AND EXISTS(SELECT 1 FROM video_job_child_owners owned JOIN chain_jobs c ON c.id=owned.child_id WHERE c.state='paused'))",rusqlite::params![owner_uuid,resume_chains],|row|row.get(0))?;
    if owned {
        return Err(OwnedChildMutation.into());
    }
    Ok(())
}

#[derive(Debug)]
pub struct OwnedChildMutation;
impl std::fmt::Display for OwnedChildMutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Use the parent video's Queue controls before changing internal clips.")
    }
}
impl std::error::Error for OwnedChildMutation {}
