//! Opaque staged-media cleanup obligations for durable generation queue rows.
//!
//! This module deliberately knows nothing about the media-set filesystem
//! format. SQLite stores only a random set identifier, owner, and lifecycle;
//! the filesystem layer remains the authority for members and bytes.

use std::collections::HashSet;

use anyhow::{bail, ensure, Result};
use rusqlite::{params, OptionalExtension, Row};

use crate::MetadataDb;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMediaObligationState {
    Active,
    GcPending,
}

impl QueueMediaObligationState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::GcPending => "gc_pending",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "active" => Some(Self::Active),
            "gc_pending" => Some(Self::GcPending),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueMediaObligation {
    pub media_set_id: String,
    pub owner_uuid: String,
    pub state: QueueMediaObligationState,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// One live queue row joined to its active opaque media obligation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveQueueMediaObligation {
    pub job_id: String,
    pub obligation: QueueMediaObligation,
}

pub(crate) fn insert_active_on_conn(
    conn: &rusqlite::Connection,
    obligation: &QueueMediaObligation,
) -> Result<()> {
    ensure!(
        !obligation.media_set_id.is_empty(),
        "queue media set id must not be empty"
    );
    ensure!(
        !obligation.owner_uuid.is_empty(),
        "queue media owner must not be empty"
    );
    ensure!(
        obligation.state == QueueMediaObligationState::Active,
        "new queue media obligation must be active"
    );
    conn.execute(
        "INSERT INTO generation_queue_media
            (media_set_id, owner_uuid, state, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, 'active', ?3, ?4)",
        params![
            obligation.media_set_id,
            obligation.owner_uuid,
            obligation.created_at_ms,
            obligation.updated_at_ms,
        ],
    )?;
    Ok(())
}

/// Register a losing file-first set for cleanup without ever reclassifying an
/// existing active/winner obligation. Returns whether this id is now a
/// gc-pending obligation owned by the same runtime. Repeating the same losing
/// admission is idempotent.
pub(crate) fn ensure_gc_pending_on_conn(
    conn: &rusqlite::Connection,
    obligation: &QueueMediaObligation,
) -> Result<bool> {
    ensure!(
        !obligation.media_set_id.is_empty(),
        "queue media set id must not be empty"
    );
    ensure!(
        !obligation.owner_uuid.is_empty(),
        "queue media owner must not be empty"
    );
    ensure!(
        obligation.state == QueueMediaObligationState::Active,
        "rejected queue media obligation must describe a newly staged active set"
    );
    conn.execute(
        "INSERT INTO generation_queue_media
            (media_set_id, owner_uuid, state, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, 'gc_pending', ?3, ?4)
         ON CONFLICT(media_set_id) DO NOTHING",
        params![
            obligation.media_set_id,
            obligation.owner_uuid,
            obligation.created_at_ms,
            obligation.updated_at_ms,
        ],
    )?;
    let existing: (String, String) = conn.query_row(
        "SELECT owner_uuid, state FROM generation_queue_media WHERE media_set_id = ?1",
        params![obligation.media_set_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(existing.0 == obligation.owner_uuid && existing.1 == "gc_pending")
}

pub(crate) fn validate_row_obligation(
    media_set_id: Option<&str>,
    owner_uuid: &str,
    obligation: &QueueMediaObligation,
) -> Result<()> {
    ensure!(
        media_set_id == Some(obligation.media_set_id.as_str()),
        "queue row media marker does not match its obligation"
    );
    ensure!(
        owner_uuid == obligation.owner_uuid,
        "queue row and media obligation have different owners"
    );
    ensure!(
        obligation.state == QueueMediaObligationState::Active,
        "queue row requires an active media obligation"
    );
    Ok(())
}

/// List one owner's obligations in stable creation order.
pub fn list_obligations(
    db: &MetadataDb,
    owner_uuid: &str,
    state: QueueMediaObligationState,
) -> Result<Vec<QueueMediaObligation>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT media_set_id, owner_uuid, state, created_at_ms, updated_at_ms
               FROM generation_queue_media
              WHERE owner_uuid = ?1 AND state = ?2
              ORDER BY created_at_ms, media_set_id",
        )?;
        let rows = stmt.query_map(params![owner_uuid, state.as_str()], row_to_obligation)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Active staged-media obligations joined to their owner-fenced queue jobs.
///
/// Startup uses this view to validate on-disk sets before the feeder begins.
/// A gc-pending obligation is intentionally absent because its queue row has
/// already been deleted and cleanup, not replay, owns it.
pub fn list_active_queue_obligations(
    db: &MetadataDb,
    owner_uuid: &str,
) -> Result<Vec<ActiveQueueMediaObligation>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT queue.id,
                    media.media_set_id, media.owner_uuid, media.state,
                    media.created_at_ms, media.updated_at_ms
               FROM generation_queue AS queue
               JOIN generation_queue_media AS media
                 ON media.media_set_id = queue.media_set_id
              WHERE queue.owner_uuid = ?1
                AND media.owner_uuid = ?1
                AND media.state = 'active'
              ORDER BY queue.created_at, queue.rowid",
        )?;
        let rows = stmt.query_map(params![owner_uuid], |row| {
            let raw_state: String = row.get(3)?;
            let state = QueueMediaObligationState::parse(&raw_state).ok_or_else(|| {
                rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    format!("unknown generation_queue_media state '{raw_state}'").into(),
                )
            })?;
            Ok(ActiveQueueMediaObligation {
                job_id: row.get(0)?,
                obligation: QueueMediaObligation {
                    media_set_id: row.get(1)?,
                    owner_uuid: row.get(2)?,
                    state,
                    created_at_ms: row.get(4)?,
                    updated_at_ms: row.get(5)?,
                },
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    })
}

/// Resolve the active opaque media obligation for one owner-fenced queue row.
///
/// Terminal paths use this lightweight projection before their delete
/// transaction. The request JSON and every other potentially large queue
/// column stay out of memory.
pub fn active_queue_obligation_for_job(
    db: &MetadataDb,
    owner_uuid: &str,
    job_id: &str,
) -> Result<Option<ActiveQueueMediaObligation>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT queue.id,
                    media.media_set_id, media.owner_uuid, media.state,
                    media.created_at_ms, media.updated_at_ms
               FROM generation_queue AS queue
               JOIN generation_queue_media AS media
                 ON media.media_set_id = queue.media_set_id
              WHERE queue.id = ?1 AND queue.owner_uuid = ?2
                AND media.owner_uuid = ?2 AND media.state = 'active'",
            params![job_id, owner_uuid],
            |row| {
                let raw_state: String = row.get(3)?;
                let state = QueueMediaObligationState::parse(&raw_state).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        format!("unknown generation_queue_media state '{raw_state}'").into(),
                    )
                })?;
                Ok(ActiveQueueMediaObligation {
                    job_id: row.get(0)?,
                    obligation: QueueMediaObligation {
                        media_set_id: row.get(1)?,
                        owner_uuid: row.get(2)?,
                        state,
                        created_at_ms: row.get(4)?,
                        updated_at_ms: row.get(5)?,
                    },
                })
            },
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Read one owner-scoped obligation without joining or hydrating its queue
/// row. A terminal cleanup may proceed only after this returns `gc_pending`,
/// which is the proof that the schema's DELETE trigger committed.
pub fn obligation_by_id(
    db: &MetadataDb,
    owner_uuid: &str,
    media_set_id: &str,
) -> Result<Option<QueueMediaObligation>> {
    db.with_conn(|conn| {
        conn.query_row(
            "SELECT media_set_id, owner_uuid, state, created_at_ms, updated_at_ms
               FROM generation_queue_media
              WHERE owner_uuid = ?1 AND media_set_id = ?2",
            params![owner_uuid, media_set_id],
            row_to_obligation,
        )
        .optional()
        .map_err(Into::into)
    })
}

/// Every opaque media set still referenced by a live queue row for `owner_uuid`.
pub fn list_referenced_media_set_ids(db: &MetadataDb, owner_uuid: &str) -> Result<HashSet<String>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT media_set_id
               FROM generation_queue
              WHERE owner_uuid = ?1 AND media_set_id IS NOT NULL",
        )?;
        let ids = stmt.query_map(params![owner_uuid], |row| row.get::<_, String>(0))?;
        ids.collect::<rusqlite::Result<HashSet<_>>>()
            .map_err(Into::into)
    })
}

/// Forget one obligation only after its filesystem cleanup has succeeded.
///
/// Active obligations are never removed by this API. The queue marker's
/// foreign key also rejects deletion if a corrupt/manual state transition
/// left the obligation referenced.
pub fn remove_gc_pending(db: &MetadataDb, owner_uuid: &str, media_set_id: &str) -> Result<bool> {
    if media_set_id.is_empty() {
        bail!("queue media set id must not be empty");
    }
    db.with_conn(|conn| {
        Ok(conn.execute(
            "DELETE FROM generation_queue_media
              WHERE media_set_id = ?1 AND owner_uuid = ?2 AND state = 'gc_pending'",
            params![media_set_id, owner_uuid],
        )? > 0)
    })
}

fn row_to_obligation(row: &Row<'_>) -> rusqlite::Result<QueueMediaObligation> {
    let raw_state: String = row.get(2)?;
    let state = QueueMediaObligationState::parse(&raw_state).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Text,
            format!("unknown generation_queue_media state '{raw_state}'").into(),
        )
    })?;
    Ok(QueueMediaObligation {
        media_set_id: row.get(0)?,
        owner_uuid: row.get(1)?,
        state,
        created_at_ms: row.get(3)?,
        updated_at_ms: row.get(4)?,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::generation_batches::{self, GenerationBatchTerminal, GenerationBatchTerminalState};
    use crate::generation_queue::{self, GenerationQueueRow, QueueRowState};

    fn queue_row(id: &str, owner_uuid: &str, media_set_id: Option<&str>) -> GenerationQueueRow {
        GenerationQueueRow {
            id: id.to_string(),
            owner_uuid: owner_uuid.to_string(),
            state: QueueRowState::Queued,
            model: "model".to_string(),
            request_json: r#"{"prompt":"unchanged"}"#.to_string(),
            output_dir: PathBuf::from("/gallery"),
            target_gpu: None,
            target_device_id: None,
            completion_payload: "metadata_only".to_string(),
            seed_pinned: false,
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms: 10,
            updated_at_ms: 10,
            started_at_ms: None,
            media_set_id: media_set_id.map(str::to_string),
            admission_authority: None,
        }
    }

    fn obligation(media_set_id: &str, owner_uuid: &str) -> QueueMediaObligation {
        QueueMediaObligation {
            media_set_id: media_set_id.to_string(),
            owner_uuid: owner_uuid.to_string(),
            state: QueueMediaObligationState::Active,
            created_at_ms: 10,
            updated_at_ms: 10,
        }
    }

    #[test]
    fn insert_and_mapping_are_atomic_owner_scoped_and_opaque() {
        let db = MetadataDb::open_in_memory().unwrap();
        let row = queue_row("job-a", "owner-a", Some("set-a"));
        let media = obligation("set-a", "owner-a");
        generation_queue::insert_with_media(&db, &row, &media).unwrap();

        assert_eq!(generation_queue::get(&db, "job-a").unwrap(), Some(row));
        assert_eq!(
            list_obligations(&db, "owner-a", QueueMediaObligationState::Active).unwrap(),
            vec![media.clone()]
        );
        assert!(
            list_obligations(&db, "owner-b", QueueMediaObligationState::Active)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            list_referenced_media_set_ids(&db, "owner-a").unwrap(),
            HashSet::from(["set-a".to_string()])
        );
        assert_eq!(
            list_active_queue_obligations(&db, "owner-a").unwrap(),
            vec![ActiveQueueMediaObligation {
                job_id: "job-a".to_string(),
                obligation: media,
            }]
        );
    }

    #[test]
    fn failed_media_insert_rolls_back_the_obligation_and_null_insert_stays_media_free() {
        let db = MetadataDb::open_in_memory().unwrap();
        let row = queue_row("job-a", "owner-a", Some("set-a"));
        assert!(generation_queue::insert_with_media(
            &db,
            &row,
            &obligation("different", "owner-a")
        )
        .is_err());
        assert!(generation_queue::get(&db, "job-a").unwrap().is_none());
        assert!(
            list_obligations(&db, "owner-a", QueueMediaObligationState::Active)
                .unwrap()
                .is_empty()
        );

        let plain = queue_row("plain", "owner-a", None);
        generation_queue::insert(&db, &plain).unwrap();
        assert_eq!(generation_queue::get(&db, "plain").unwrap(), Some(plain));
        let obligations: i64 = db
            .with_conn(|conn| {
                conn.query_row("SELECT COUNT(*) FROM generation_queue_media", [], |row| {
                    row.get(0)
                })
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(obligations, 0);
    }

    #[test]
    fn delete_trigger_retires_single_bulk_legacy_and_claimed_rows() {
        let db = MetadataDb::open_in_memory().unwrap();

        generation_queue::insert_with_media(
            &db,
            &queue_row("single", "owner-single", Some("set-single")),
            &obligation("set-single", "owner-single"),
        )
        .unwrap();
        assert_eq!(
            db.with_conn(|conn| {
                conn.execute("DELETE FROM generation_queue WHERE id = 'single'", [])
                    .map_err(Into::into)
            })
            .unwrap(),
            1,
            "the schema trigger, not a Rust deletion wrapper, retires the obligation"
        );

        for suffix in ["a", "b"] {
            generation_queue::insert_with_media(
                &db,
                &queue_row(
                    &format!("bulk-{suffix}"),
                    "owner-bulk",
                    Some(&format!("set-bulk-{suffix}")),
                ),
                &obligation(&format!("set-bulk-{suffix}"), "owner-bulk"),
            )
            .unwrap();
        }
        assert_eq!(
            generation_queue::delete_all_queued(&db, "owner-bulk").unwrap(),
            2
        );

        generation_queue::insert_with_media(
            &db,
            &queue_row("legacy", "owner-legacy", Some("set-legacy")),
            &obligation("set-legacy", "owner-legacy"),
        )
        .unwrap();
        assert!(generation_queue::delete_legacy(&db, "legacy").unwrap());

        generation_queue::insert_claimed_with_media(
            &db,
            &queue_row("claimed", "owner-claimed", Some("set-claimed")),
            "claim-token",
            &obligation("set-claimed", "owner-claimed"),
        )
        .unwrap();
        generation_queue::mark_dispatched_claimed(&db, "claimed", "claim-token", 20).unwrap();
        let commit = generation_batches::finish_claimed(
            &db,
            "claimed",
            "claim-token",
            QueueRowState::Running,
            GenerationBatchTerminal {
                state: GenerationBatchTerminalState::Failed,
                error: Some("failed"),
                terminal_error_json: None,
                result_json: None,
                completed_at_ms: 30,
            },
        )
        .unwrap();
        assert!(commit.queue_deleted);

        for (owner, expected) in [
            ("owner-single", vec!["set-single"]),
            ("owner-bulk", vec!["set-bulk-a", "set-bulk-b"]),
            ("owner-legacy", vec!["set-legacy"]),
            ("owner-claimed", vec!["set-claimed"]),
        ] {
            let pending = list_obligations(&db, owner, QueueMediaObligationState::GcPending)
                .unwrap()
                .into_iter()
                .map(|row| row.media_set_id)
                .collect::<Vec<_>>();
            assert_eq!(pending, expected);
        }
    }

    #[test]
    fn startup_hold_is_owner_and_media_fenced_without_retiring_the_set() {
        let db = MetadataDb::open_in_memory().unwrap();
        generation_queue::insert_claimed_with_media(
            &db,
            &queue_row("mine", "owner-a", Some("set-mine")),
            "stale-claim",
            &obligation("set-mine", "owner-a"),
        )
        .unwrap();
        generation_queue::mark_dispatched_claimed(&db, "mine", "stale-claim", 20).unwrap();
        generation_queue::insert_with_media(
            &db,
            &queue_row("foreign", "owner-b", Some("set-foreign")),
            &obligation("set-foreign", "owner-b"),
        )
        .unwrap();
        generation_queue::insert(&db, &queue_row("plain", "owner-a", None)).unwrap();

        let ids = vec![
            "mine".to_string(),
            "mine".to_string(),
            "foreign".to_string(),
            "plain".to_string(),
        ];
        assert_eq!(
            generation_queue::hold_media_jobs(&db, "owner-a", &ids, "media invalid", 40).unwrap(),
            1
        );
        assert_eq!(
            generation_queue::hold_media_jobs(&db, "owner-a", &ids, "media invalid", 41).unwrap(),
            1,
            "an already-held media job still satisfies the startup quarantine"
        );
        assert_eq!(
            generation_queue::get(&db, "mine").unwrap().unwrap().state,
            QueueRowState::Held
        );
        let runtime_ownership: (Option<String>, Option<i64>) = db
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT claim_token, started_at FROM generation_queue WHERE id = 'mine'",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(runtime_ownership, (None, None));
        assert_eq!(
            generation_queue::get(&db, "foreign")
                .unwrap()
                .unwrap()
                .state,
            QueueRowState::Queued
        );
        assert_eq!(
            generation_queue::get(&db, "plain").unwrap().unwrap().state,
            QueueRowState::Queued
        );
        assert_eq!(
            list_obligations(&db, "owner-a", QueueMediaObligationState::Active)
                .unwrap()
                .len(),
            1
        );
        assert!(
            list_obligations(&db, "owner-a", QueueMediaObligationState::GcPending)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn gc_removal_requires_pending_state_and_the_matching_owner() {
        let db = MetadataDb::open_in_memory().unwrap();
        generation_queue::insert_with_media(
            &db,
            &queue_row("job", "owner-a", Some("set")),
            &obligation("set", "owner-a"),
        )
        .unwrap();
        assert!(!remove_gc_pending(&db, "owner-a", "set").unwrap());
        assert!(db
            .with_conn(|conn| {
                conn.execute(
                    "UPDATE generation_queue SET media_set_id = NULL WHERE id = 'job'",
                    [],
                )
                .map(|_| ())
                .map_err(Into::into)
            })
            .is_err());
        generation_queue::delete(&db, "job").unwrap();
        assert!(!remove_gc_pending(&db, "owner-b", "set").unwrap());
        assert!(remove_gc_pending(&db, "owner-a", "set").unwrap());
        assert!(!remove_gc_pending(&db, "owner-a", "set").unwrap());
    }

    #[test]
    fn targeted_lookup_is_owner_scoped_and_tracks_trigger_retirement() {
        let db = MetadataDb::open_in_memory().unwrap();
        generation_queue::insert_with_media(
            &db,
            &queue_row("job", "owner-a", Some("set")),
            &obligation("set", "owner-a"),
        )
        .unwrap();

        assert_eq!(
            active_queue_obligation_for_job(&db, "owner-a", "job").unwrap(),
            Some(ActiveQueueMediaObligation {
                job_id: "job".to_string(),
                obligation: obligation("set", "owner-a"),
            })
        );
        assert!(active_queue_obligation_for_job(&db, "owner-b", "job")
            .unwrap()
            .is_none());
        assert_eq!(
            obligation_by_id(&db, "owner-a", "set")
                .unwrap()
                .unwrap()
                .state,
            QueueMediaObligationState::Active
        );

        assert!(generation_queue::delete(&db, "job").unwrap());
        assert!(active_queue_obligation_for_job(&db, "owner-a", "job")
            .unwrap()
            .is_none());
        assert_eq!(
            obligation_by_id(&db, "owner-a", "set")
                .unwrap()
                .unwrap()
                .state,
            QueueMediaObligationState::GcPending
        );
        assert!(obligation_by_id(&db, "owner-b", "set").unwrap().is_none());
    }
}
