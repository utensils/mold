//! Lightweight durable grouping for heterogeneous generation admission.
//!
//! Child execution authority remains `generation_queue`; these tables only
//! make one client admission idempotent and retain terminal child summaries
//! after the queue rows are removed.

use anyhow::{bail, Result};
use rusqlite::{params, OptionalExtension};

use crate::generation_queue::{self, GenerationQueueRow};
use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchRow {
    pub id: String,
    pub client_batch_id: String,
    pub owner_uuid: String,
    pub request_sha256: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchChildRow {
    pub batch_id: String,
    pub job_id: String,
    pub batch_index: u32,
    pub state: String,
    pub error: Option<String>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchDetail {
    pub batch: GenerationBatchRow,
    pub children: Vec<GenerationBatchChildRow>,
}

/// Insert the grouping rows and every ordinary durable queue row atomically.
/// A retry with the same client id returns the existing detail; a different
/// payload using that id is rejected.
pub fn insert_or_get(
    db: &MetadataDb,
    batch: &GenerationBatchRow,
    children: &[(GenerationBatchChildRow, GenerationQueueRow)],
) -> Result<(GenerationBatchDetail, bool)> {
    db.transact_immediate(|conn| {
        if let Some(existing) =
            get_by_client_on_conn(conn, &batch.owner_uuid, &batch.client_batch_id)?
        {
            if existing.batch.request_sha256 != batch.request_sha256 {
                bail!("client_batch_id was already used for a different request");
            }
            return Ok((existing, false));
        }
        conn.execute(
            "INSERT INTO generation_batches
                (id, client_batch_id, owner_uuid, request_sha256, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                batch.id,
                batch.client_batch_id,
                batch.owner_uuid,
                batch.request_sha256,
                batch.created_at_ms,
            ],
        )?;
        for (child, queue_row) in children {
            generation_queue::insert_on_conn(conn, queue_row)?;
            conn.execute(
                "INSERT INTO generation_batch_children
                    (batch_id, job_id, batch_index, state, error, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    child.batch_id,
                    child.job_id,
                    child.batch_index,
                    child.state,
                    child.error,
                    child.updated_at_ms,
                ],
            )?;
        }
        Ok((
            GenerationBatchDetail {
                batch: batch.clone(),
                children: children.iter().map(|(child, _)| child.clone()).collect(),
            },
            true,
        ))
    })
}

pub fn get(db: &MetadataDb, owner_uuid: &str, id: &str) -> Result<Option<GenerationBatchDetail>> {
    db.with_conn(|conn| {
        let batch = conn
            .query_row(
                "SELECT id, client_batch_id, owner_uuid, request_sha256, created_at_ms
                   FROM generation_batches WHERE id = ?1 AND owner_uuid = ?2",
                params![id, owner_uuid],
                batch_from_row,
            )
            .optional()?;
        batch.map(|batch| detail_on_conn(conn, batch)).transpose()
    })
}

pub fn get_by_client(
    db: &MetadataDb,
    owner_uuid: &str,
    client_batch_id: &str,
) -> Result<Option<GenerationBatchDetail>> {
    db.with_conn(|conn| get_by_client_on_conn(conn, owner_uuid, client_batch_id))
}

pub fn set_child_state(
    db: &MetadataDb,
    job_id: &str,
    state: &str,
    error: Option<&str>,
    updated_at_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE generation_batch_children
                SET state = ?2, error = ?3, updated_at_ms = ?4
              WHERE job_id = ?1",
            params![job_id, state, error, updated_at_ms],
        )? > 0)
    })
}

fn get_by_client_on_conn(
    conn: &rusqlite::Connection,
    owner_uuid: &str,
    client_batch_id: &str,
) -> Result<Option<GenerationBatchDetail>> {
    let batch = conn
        .query_row(
            "SELECT id, client_batch_id, owner_uuid, request_sha256, created_at_ms
               FROM generation_batches WHERE owner_uuid = ?1 AND client_batch_id = ?2",
            params![owner_uuid, client_batch_id],
            batch_from_row,
        )
        .optional()?;
    batch.map(|batch| detail_on_conn(conn, batch)).transpose()
}

fn detail_on_conn(
    conn: &rusqlite::Connection,
    batch: GenerationBatchRow,
) -> Result<GenerationBatchDetail> {
    let mut stmt = conn.prepare(
        "SELECT batch_id, job_id, batch_index, state, error, updated_at_ms
           FROM generation_batch_children WHERE batch_id = ?1 ORDER BY batch_index",
    )?;
    let children = stmt
        .query_map(params![batch.id], |row| {
            Ok(GenerationBatchChildRow {
                batch_id: row.get(0)?,
                job_id: row.get(1)?,
                batch_index: row.get::<_, i64>(2)? as u32,
                state: row.get(3)?,
                error: row.get(4)?,
                updated_at_ms: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(GenerationBatchDetail { batch, children })
}

fn batch_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<GenerationBatchRow> {
    Ok(GenerationBatchRow {
        id: row.get(0)?,
        client_batch_id: row.get(1)?,
        owner_uuid: row.get(2)?,
        request_sha256: row.get(3)?,
        created_at_ms: row.get(4)?,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::generation_queue::QueueRowState;

    fn rows(count: usize) -> Vec<(GenerationBatchChildRow, GenerationQueueRow)> {
        (0..count)
            .map(|index| {
                let id = format!("job-{index}");
                (
                    GenerationBatchChildRow {
                        batch_id: "batch-1".into(),
                        job_id: id.clone(),
                        batch_index: index as u32 + 1,
                        state: "accepted".into(),
                        error: None,
                        updated_at_ms: 1,
                    },
                    GenerationQueueRow {
                        id,
                        owner_uuid: "owner-1".into(),
                        state: QueueRowState::Queued,
                        model: format!("model-{index}"),
                        request_json: format!(r#"{{"prompt":"prompt-{index}"}}"#),
                        output_dir: PathBuf::from("/gallery"),
                        target_gpu: None,
                        target_device_id: None,
                        completion_payload: "metadata_only".into(),
                        seed_pinned: false,
                        dispatch_attempts: 0,
                        replay_seen: 0,
                        held_reason: None,
                        created_at_ms: 1,
                        updated_at_ms: 1,
                        started_at_ms: None,
                    },
                )
            })
            .collect()
    }

    fn batch(hash: &str) -> GenerationBatchRow {
        GenerationBatchRow {
            id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            owner_uuid: "owner-1".into(),
            request_sha256: hash.into(),
            created_at_ms: 1,
        }
    }

    #[test]
    fn inserts_all_children_and_replays_same_ids() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = rows(30);
        let (first, inserted) = insert_or_get(&db, &batch("same"), &children).unwrap();
        assert!(inserted);
        assert_eq!(first.children.len(), 30);

        let (retry, inserted) = insert_or_get(&db, &batch("same"), &children).unwrap();
        assert!(!inserted);
        assert_eq!(retry, first);
        assert_eq!(
            crate::generation_queue::list_all(&db, "owner-1")
                .unwrap()
                .len(),
            30
        );
    }

    #[test]
    fn all_admitted_children_survive_database_reopen_for_queue_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold.db");
        let first_ids = {
            let db = MetadataDb::open(&path).unwrap();
            let children = rows(30);
            let (detail, inserted) = insert_or_get(&db, &batch("same"), &children).unwrap();
            assert!(inserted);
            detail
                .children
                .into_iter()
                .map(|child| child.job_id)
                .collect::<Vec<_>>()
        };

        let reopened = MetadataDb::open(&path).unwrap();
        let detail = get_by_client(&reopened, "owner-1", "client-1")
            .unwrap()
            .expect("batch grouping survives restart");
        assert_eq!(
            detail
                .children
                .into_iter()
                .map(|child| child.job_id)
                .collect::<Vec<_>>(),
            first_ids
        );
        assert_eq!(
            crate::generation_queue::list_all(&reopened, "owner-1")
                .unwrap()
                .into_iter()
                .map(|row| row.id)
                .collect::<Vec<_>>(),
            first_ids,
            "the ordinary durable queue reopens with every child exactly once"
        );
    }

    #[test]
    fn rejects_changed_payload_without_partial_rows() {
        let db = MetadataDb::open_in_memory().unwrap();
        let children = rows(2);
        insert_or_get(&db, &batch("first"), &children).unwrap();
        assert!(insert_or_get(&db, &batch("changed"), &children).is_err());
        assert_eq!(
            crate::generation_queue::list_all(&db, "owner-1")
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn child_insert_failure_rolls_back_parent_and_every_child() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut children = rows(2);
        children[1].1.id = children[0].1.id.clone();
        assert!(insert_or_get(&db, &batch("same"), &children).is_err());
        assert!(get(&db, "owner-1", "batch-1").unwrap().is_none());
        assert!(crate::generation_queue::list_all(&db, "owner-1")
            .unwrap()
            .is_empty());
    }
}
