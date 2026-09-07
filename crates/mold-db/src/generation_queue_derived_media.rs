//! Purpose-keyed encrypted media produced while a durable generation runs.
//!
//! The queue row's `media_set_id` remains the authored-input authority. These
//! mappings add independently encrypted stage outputs which share the row's
//! lifetime and are handed to the resulting gallery item before settlement.

use anyhow::{ensure, Result};
use rusqlite::{params, OptionalExtension};

use crate::generation_queue_media::{self, QueueMediaObligation, QueueMediaObligationState};
use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedQueueMediaBinding {
    pub job_id: String,
    pub owner_uuid: String,
    pub kind: String,
    /// Authenticated job identity inside the encrypted bundle. It differs
    /// from `job_id` because the store permits one bundle per owner/job pair.
    pub storage_job_id: String,
    pub obligation: QueueMediaObligation,
}

/// Register the file-first set and its live-job mapping in one transaction.
pub fn register(db: &MetadataDb, binding: &DerivedQueueMediaBinding) -> Result<()> {
    validate(binding)?;
    db.transact_immediate(|conn| {
        generation_queue_media::insert_active_on_conn(conn, &binding.obligation)?;
        conn.execute(
            "INSERT INTO generation_queue_derived_media
                (job_id, owner_uuid, kind, storage_job_id, media_set_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                binding.job_id,
                binding.owner_uuid,
                binding.kind,
                binding.storage_job_id,
                binding.obligation.media_set_id,
            ],
        )?;
        Ok(())
    })
}

/// Resolve one purpose for idempotent attempt replay.
pub fn for_job_kind(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
    kind: &str,
) -> Result<Option<DerivedQueueMediaBinding>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT derived.job_id, derived.owner_uuid, derived.kind,
                    derived.storage_job_id,
                    media.media_set_id, media.state,
                    media.created_at_ms, media.updated_at_ms
               FROM generation_queue_derived_media AS derived
               JOIN generation_queue_media AS media
                 ON media.media_set_id = derived.media_set_id
              WHERE derived.job_id = ?1 AND derived.owner_uuid = ?2
                AND derived.kind = ?3",
            params![job_id, owner_uuid, kind],
            row_to_binding,
        )
        .optional()
        .map_err(Into::into)
    })
}

/// All active primary-adjacent stage outputs for one live queue job.
pub fn active_for_job(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
) -> Result<Vec<DerivedQueueMediaBinding>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT derived.job_id, derived.owner_uuid, derived.kind,
                    derived.storage_job_id,
                    media.media_set_id, media.state,
                    media.created_at_ms, media.updated_at_ms
               FROM generation_queue_derived_media AS derived
               JOIN generation_queue_media AS media
                 ON media.media_set_id = derived.media_set_id
              WHERE derived.job_id = ?1 AND derived.owner_uuid = ?2
                AND media.owner_uuid = ?2 AND media.state = 'active'
              ORDER BY derived.kind",
        )?;
        let rows = stmt.query_map(params![job_id, owner_uuid], row_to_binding)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Every active derived obligation for owner-scoped startup reconciliation.
pub fn list_active(db: &MetadataDb, owner_uuid: &str) -> Result<Vec<DerivedQueueMediaBinding>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT derived.job_id, derived.owner_uuid, derived.kind,
                    derived.storage_job_id,
                    media.media_set_id, media.state,
                    media.created_at_ms, media.updated_at_ms
               FROM generation_queue_derived_media AS derived
               JOIN generation_queue AS queue ON queue.id = derived.job_id
               JOIN generation_queue_media AS media
                 ON media.media_set_id = derived.media_set_id
              WHERE derived.owner_uuid = ?1 AND queue.owner_uuid = ?1
                AND media.owner_uuid = ?1 AND media.state = 'active'
              ORDER BY queue.created_at, queue.rowid, derived.kind",
        )?;
        let rows = stmt.query_map(params![owner_uuid], row_to_binding)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

fn validate(binding: &DerivedQueueMediaBinding) -> Result<()> {
    ensure!(
        !binding.job_id.is_empty(),
        "derived media job id must not be empty"
    );
    ensure!(
        !binding.owner_uuid.is_empty(),
        "derived media owner must not be empty"
    );
    ensure!(
        (1..=64).contains(&binding.kind.len()),
        "derived media kind must contain 1 through 64 bytes"
    );
    ensure!(
        !binding.storage_job_id.is_empty(),
        "derived media storage job id must not be empty"
    );
    ensure!(
        binding.obligation.owner_uuid == binding.owner_uuid,
        "derived media obligation owner mismatch"
    );
    ensure!(
        binding.obligation.state == QueueMediaObligationState::Active,
        "derived media obligation must be active"
    );
    Ok(())
}

fn row_to_binding(row: &rusqlite::Row<'_>) -> rusqlite::Result<DerivedQueueMediaBinding> {
    let raw_state: String = row.get(5)?;
    let state = match raw_state.as_str() {
        "active" => QueueMediaObligationState::Active,
        "gc_pending" => QueueMediaObligationState::GcPending,
        _ => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                5,
                rusqlite::types::Type::Text,
                format!("unknown generation_queue_media state '{raw_state}'").into(),
            ))
        }
    };
    Ok(DerivedQueueMediaBinding {
        job_id: row.get(0)?,
        owner_uuid: row.get(1)?,
        kind: row.get(2)?,
        storage_job_id: row.get(3)?,
        obligation: QueueMediaObligation {
            media_set_id: row.get(4)?,
            owner_uuid: row.get(1)?,
            state,
            created_at_ms: row.get(6)?,
            updated_at_ms: row.get(7)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation_queue::{self, GenerationQueueRow, QueueRowState};
    use std::path::PathBuf;

    fn row(id: &str, owner: &str) -> GenerationQueueRow {
        GenerationQueueRow {
            id: id.into(),
            owner_uuid: owner.into(),
            state: QueueRowState::Queued,
            model: "model".into(),
            request_json: "{}".into(),
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
            media_set_id: None,
            admission_authority: None,
        }
    }

    fn binding(job: &str, owner: &str, kind: &str, set: &str) -> DerivedQueueMediaBinding {
        DerivedQueueMediaBinding {
            job_id: job.into(),
            owner_uuid: owner.into(),
            kind: kind.into(),
            storage_job_id: format!("{job}-derived-{kind}"),
            obligation: QueueMediaObligation {
                media_set_id: set.into(),
                owner_uuid: owner.into(),
                state: QueueMediaObligationState::Active,
                created_at_ms: 2,
                updated_at_ms: 2,
            },
        }
    }

    #[test]
    fn registration_is_owner_scoped_purpose_unique_and_retires_with_the_job() {
        let db = MetadataDb::open_in_memory().unwrap();
        generation_queue::insert(&db, &row("job", "owner")).unwrap();
        let matte = binding("job", "owner", "matting_processed", "set-a");
        register(&db, &matte).unwrap();
        assert_eq!(active_for_job(&db, "owner", "job").unwrap(), vec![matte]);

        assert!(register(&db, &binding("job", "owner", "matting_processed", "set-b")).is_err());
        assert!(register(
            &db,
            &binding("job", "other", "workflow_generated_source", "set-c")
        )
        .is_err());

        generation_queue::delete(&db, "job").unwrap();
        let state: String = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT state FROM generation_queue_media WHERE media_set_id = 'set-a'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(state, "gc_pending");
        assert!(for_job_kind(&db, "owner", "job", "matting_processed")
            .unwrap()
            .is_none());
    }
}
