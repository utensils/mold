//! Concrete owner-fenced lifecycle adapter for encrypted durable queue media.

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use mold_db::generation_queue;
use mold_db::generation_queue_media::{
    self, ActiveQueueMediaObligation, QueueMediaObligationState,
};
use mold_db::MetadataDb;

use crate::queue_media_startup::{
    AdapterError, AdapterFailureKind, MediaObligation, ObligationState, QueueMediaStartupAdapter,
    StoreEntry, StoreEntryState, StoreInitializationPolicy, StoreInspection, UnclaimedOwnerRoot,
    UntouchedEntry,
};
use crate::queue_media_store::{MediaSetRef, QueueMediaError, QueueMediaStore};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QueueMediaGcCandidate {
    media_set: MediaSetRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CleanupOutcome {
    /// The queue row or its trigger-created GC obligation still exists.
    Retained,
    /// Another cleanup already removed the obligation.
    AlreadyClean,
    /// The authenticated set was unlinked and its GC obligation cleared.
    Deleted,
}

/// Singular concrete bridge between owner-fenced SQLite obligations and the
/// encrypted filesystem store. It contains no route or request admission.
pub(crate) struct QueueMediaLifecycle {
    db: Arc<Option<MetadataDb>>,
    mold_home: PathBuf,
    owner_uuid: String,
    store: Mutex<Option<QueueMediaStore>>,
}

impl QueueMediaLifecycle {
    pub(crate) fn new(db: Arc<Option<MetadataDb>>, mold_home: PathBuf, owner_uuid: String) -> Self {
        Self {
            db,
            mold_home,
            owner_uuid,
            store: Mutex::new(None),
        }
    }

    fn db(&self) -> Result<&MetadataDb, AdapterError> {
        self.db.as_ref().as_ref().ok_or_else(|| {
            AdapterError::new(
                AdapterFailureKind::Database,
                "metadata database is unavailable",
            )
        })
    }

    fn ensure_owner(&self, owner_uuid: &str) -> Result<(), AdapterError> {
        if owner_uuid == self.owner_uuid {
            Ok(())
        } else {
            Err(AdapterError::new(
                AdapterFailureKind::Invariant,
                "queue-media adapter was asked to cross its claimed owner",
            ))
        }
    }

    fn opened_store(&self) -> Result<QueueMediaStore, AdapterError> {
        self.store
            .lock()
            .map_err(|_| {
                AdapterError::new(
                    AdapterFailureKind::Invariant,
                    "queue-media store lock was poisoned",
                )
            })?
            .clone()
            .ok_or_else(|| {
                AdapterError::new(
                    AdapterFailureKind::Invariant,
                    "queue-media store was not opened by startup reconciliation",
                )
            })
    }

    pub(crate) fn candidate_for_job(
        &self,
        job_id: &str,
    ) -> Result<Option<QueueMediaGcCandidate>, AdapterError> {
        let joined = generation_queue_media::active_queue_obligation_for_job(
            self.db()?,
            &self.owner_uuid,
            job_id,
        )
        .map_err(db_error)?;
        Ok(joined.map(|joined| candidate_from_joined(&joined)))
    }

    pub(crate) fn active_candidates(&self) -> Result<Vec<QueueMediaGcCandidate>, AdapterError> {
        generation_queue_media::list_active_queue_obligations(self.db()?, &self.owner_uuid)
            .map(|joined| joined.iter().map(candidate_from_joined).collect())
            .map_err(db_error)
    }

    /// Clean only after the DB trigger proves the queue row was deleted.
    /// Failures intentionally retain `gc_pending` for the next startup pass.
    pub(crate) fn cleanup_after_committed_delete(
        &self,
        candidate: &QueueMediaGcCandidate,
    ) -> Result<CleanupOutcome, AdapterError> {
        self.ensure_owner(&candidate.media_set.owner_id)?;
        let obligation = generation_queue_media::obligation_by_id(
            self.db()?,
            &self.owner_uuid,
            &candidate.media_set.set_id,
        )
        .map_err(db_error)?;
        match obligation {
            Some(obligation) if obligation.state == QueueMediaObligationState::Active => {
                return Ok(CleanupOutcome::Retained);
            }
            None => return Ok(CleanupOutcome::AlreadyClean),
            Some(_) => {}
        }

        self.opened_store()?
            .delete(&candidate.media_set)
            .map_err(store_error)?;
        let removed = generation_queue_media::remove_gc_pending(
            self.db()?,
            &self.owner_uuid,
            &candidate.media_set.set_id,
        )
        .map_err(db_error)?;
        if !removed {
            return Err(AdapterError::new(
                AdapterFailureKind::Invariant,
                format!(
                    "media set {} was unlinked without its GC obligation",
                    candidate.media_set.set_id
                ),
            ));
        }
        Ok(CleanupOutcome::Deleted)
    }

    fn store_ref_for_entry(
        &self,
        owner_uuid: &str,
        entry: &StoreEntry,
    ) -> Result<MediaSetRef, AdapterError> {
        self.ensure_owner(owner_uuid)?;
        Ok(MediaSetRef {
            owner_id: owner_uuid.to_string(),
            job_id: entry.job_id.clone(),
            set_id: entry.set_id.clone(),
        })
    }

    fn inspect_store_owner(
        &self,
        owner_uuid: &str,
    ) -> Result<crate::queue_media_store::StoreInspection, AdapterError> {
        self.ensure_owner(owner_uuid)?;
        Ok(self.opened_store()?.inspect_owner(owner_uuid))
    }
}

impl QueueMediaStartupAdapter for QueueMediaLifecycle {
    fn obligations(&self, owner_uuid: &str) -> Result<Vec<MediaObligation>, AdapterError> {
        self.ensure_owner(owner_uuid)?;
        let db = self.db()?;
        let active = generation_queue_media::list_obligations(
            db,
            owner_uuid,
            QueueMediaObligationState::Active,
        )
        .map_err(db_error)?;
        let joined = generation_queue_media::list_active_queue_obligations(db, owner_uuid)
            .map_err(db_error)?;
        let referenced = generation_queue_media::list_referenced_media_set_ids(db, owner_uuid)
            .map_err(db_error)?;
        let active_ids: HashSet<&str> =
            active.iter().map(|row| row.media_set_id.as_str()).collect();
        let joined_ids: HashSet<&str> = joined
            .iter()
            .map(|row| row.obligation.media_set_id.as_str())
            .collect();
        let joined_jobs: HashSet<&str> = joined.iter().map(|row| row.job_id.as_str()).collect();
        if active_ids.len() != active.len()
            || joined_ids.len() != joined.len()
            || joined_jobs.len() != joined.len()
            || active_ids != referenced.iter().map(String::as_str).collect()
            || joined_ids != active_ids
        {
            return Err(AdapterError::new(
                AdapterFailureKind::Invariant,
                "active queue-media obligations and generation_queue.media_set_id references disagree",
            ));
        }

        let mut obligations = joined
            .into_iter()
            .map(|joined| MediaObligation {
                job_id: Some(joined.job_id),
                set_id: joined.obligation.media_set_id,
                state: ObligationState::Active,
            })
            .collect::<Vec<_>>();
        obligations.extend(
            generation_queue_media::list_obligations(
                db,
                owner_uuid,
                QueueMediaObligationState::GcPending,
            )
            .map_err(db_error)?
            .into_iter()
            .map(|row| MediaObligation {
                job_id: None,
                set_id: row.media_set_id,
                state: ObligationState::GcPending,
            }),
        );
        Ok(obligations)
    }

    fn unclaimed_owner_roots(
        &self,
        owner_uuid: &str,
    ) -> Result<Vec<UnclaimedOwnerRoot>, AdapterError> {
        self.ensure_owner(owner_uuid)?;
        Ok(self
            .opened_store()?
            .unclaimed_owner_roots(owner_uuid)
            .into_iter()
            .map(|root| UnclaimedOwnerRoot {
                owner_id_hint: root.owner_id_hint,
                description: root.description,
            })
            .collect())
    }

    fn open_store(
        &self,
        owner_uuid: &str,
        initialization: StoreInitializationPolicy,
    ) -> Result<(), AdapterError> {
        self.ensure_owner(owner_uuid)?;
        let opened = match initialization {
            StoreInitializationPolicy::Deny => {
                QueueMediaStore::open_existing(&self.mold_home).map_err(store_error)?
            }
            StoreInitializationPolicy::IfGloballyEmpty => {
                QueueMediaStore::open(&self.mold_home)
                    .map_err(store_error)?
                    .store
            }
        };
        let mut store = self.store.lock().map_err(|_| {
            AdapterError::new(
                AdapterFailureKind::Invariant,
                "queue-media store lock was poisoned",
            )
        })?;
        *store = Some(opened);
        Ok(())
    }

    fn inspect_owner(&self, owner_uuid: &str) -> Result<StoreInspection, AdapterError> {
        let inspection = self.inspect_store_owner(owner_uuid)?;
        let mut entries = Vec::new();
        entries.extend(inspection.active.into_iter().map(|entry| StoreEntry {
            job_id: entry.job_id,
            set_id: entry.set_id,
            state: StoreEntryState::Active,
        }));
        entries.extend(inspection.retired.into_iter().map(|entry| StoreEntry {
            job_id: entry.job_id,
            set_id: entry.set_id,
            state: StoreEntryState::Retired,
        }));
        entries.extend(inspection.staging.into_iter().map(|entry| StoreEntry {
            job_id: entry.job_id,
            set_id: entry.set_id,
            state: StoreEntryState::Staging,
        }));
        Ok(StoreInspection {
            entries,
            untouched: inspection
                .unrecognized
                .into_iter()
                .map(|entry| UntouchedEntry {
                    set_id_hint: entry.set_id_hint,
                    description: format!("{}: {}", entry.path.display(), entry.reason),
                })
                .collect(),
        })
    }

    fn restore(&self, owner_uuid: &str, entry: &StoreEntry) -> Result<(), AdapterError> {
        let media_set = self.store_ref_for_entry(owner_uuid, entry)?;
        self.opened_store()?
            .restore(&media_set)
            .map_err(store_error)
    }

    fn delete(&self, owner_uuid: &str, entry: &StoreEntry) -> Result<(), AdapterError> {
        let media_set = self.store_ref_for_entry(owner_uuid, entry)?;
        let store = self.opened_store()?;
        match entry.state {
            StoreEntryState::Staging => store.delete_staging(&media_set).map_err(store_error),
            StoreEntryState::Active | StoreEntryState::Retired => {
                store.delete(&media_set).map_err(store_error)
            }
        }
    }

    fn clear_gc_pending(&self, owner_uuid: &str, set_id: &str) -> Result<(), AdapterError> {
        self.ensure_owner(owner_uuid)?;
        if generation_queue_media::remove_gc_pending(self.db()?, owner_uuid, set_id)
            .map_err(db_error)?
        {
            Ok(())
        } else {
            Err(AdapterError::new(
                AdapterFailureKind::Invariant,
                format!("GC obligation {set_id} disappeared before it could be cleared"),
            ))
        }
    }

    fn hold_jobs(
        &self,
        owner_uuid: &str,
        job_ids: &[String],
        reason: &str,
    ) -> Result<(), AdapterError> {
        self.ensure_owner(owner_uuid)?;
        let expected = job_ids.iter().collect::<BTreeSet<_>>().len();
        let held =
            generation_queue::hold_media_jobs(self.db()?, owner_uuid, job_ids, reason, now_ms())
                .map_err(db_error)?;
        if held == expected {
            Ok(())
        } else {
            Err(AdapterError::new(
                AdapterFailureKind::Invariant,
                format!("held {held} of {expected} media jobs during startup"),
            ))
        }
    }
}

fn candidate_from_joined(joined: &ActiveQueueMediaObligation) -> QueueMediaGcCandidate {
    QueueMediaGcCandidate {
        media_set: MediaSetRef {
            owner_id: joined.obligation.owner_uuid.clone(),
            job_id: joined.job_id.clone(),
            set_id: joined.obligation.media_set_id.clone(),
        },
    }
}

fn db_error(error: anyhow::Error) -> AdapterError {
    AdapterError::new(AdapterFailureKind::Database, format!("{error:#}"))
}

fn store_error(error: QueueMediaError) -> AdapterError {
    let kind = match &error {
        QueueMediaError::MissingKeyWithExistingStore | QueueMediaError::MissingKey => {
            AdapterFailureKind::KeyMissing
        }
        QueueMediaError::Authentication | QueueMediaError::Corrupt(_) => {
            AdapterFailureKind::KeyCorrupt
        }
        QueueMediaError::Io(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            AdapterFailureKind::Permission
        }
        QueueMediaError::InsecurePath(_)
        | QueueMediaError::SecurityUnavailable(_)
        | QueueMediaError::InvalidIdentity(_) => AdapterFailureKind::UnsafeLayout,
        QueueMediaError::JobAlreadySealed { .. }
        | QueueMediaError::ProjectionUnavailable(_)
        | QueueMediaError::MixedSinkHydrationRequired => AdapterFailureKind::Invariant,
        QueueMediaError::Io(_) | QueueMediaError::Json(_) | QueueMediaError::NotFound => {
            AdapterFailureKind::Io
        }
    };
    AdapterError::new(kind, error.to_string())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_journal::QueueJournal;
    use crate::queue_media_startup::{
        reconcile_claimed_owner, AdapterFailureKind, QueueMediaStartupAdapter,
    };
    use crate::queue_media_store::{QueueMediaStore, SealMedia};
    use mold_db::generation_batches::{self, GenerationBatchChildRow, GenerationBatchRow};
    use mold_db::generation_queue::{self, GenerationQueueRow, QueueRowState};
    use mold_db::generation_queue_media::{self, QueueMediaObligation, QueueMediaObligationState};
    use mold_db::MetadataDb;
    use std::sync::Arc;

    fn queue_row(owner: &str, id: &str, set_id: &str) -> GenerationQueueRow {
        GenerationQueueRow {
            id: id.to_string(),
            owner_uuid: owner.to_string(),
            state: QueueRowState::Queued,
            model: "flux-dev:q4".to_string(),
            request_json: "{}".to_string(),
            output_dir: "/gallery".into(),
            target_gpu: None,
            target_device_id: None,
            completion_payload: "metadata_only".to_string(),
            seed_pinned: false,
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms: 1,
            updated_at_ms: 1,
            started_at_ms: None,
            media_set_id: Some(set_id.to_string()),
        }
    }

    fn obligation(owner: &str, set_id: &str) -> QueueMediaObligation {
        QueueMediaObligation {
            media_set_id: set_id.to_string(),
            owner_uuid: owner.to_string(),
            state: QueueMediaObligationState::Active,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    fn install_and_reconcile(
        home: &std::path::Path,
        db: Arc<Option<MetadataDb>>,
        journal: &Arc<QueueJournal>,
    ) -> Arc<QueueMediaLifecycle> {
        let lifecycle = Arc::new(QueueMediaLifecycle::new(
            db,
            home.to_path_buf(),
            journal.owner_uuid().unwrap().to_string(),
        ));
        journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .unwrap();
        assert!(
            reconcile_claimed_owner(journal, lifecycle.as_ref())
                .unwrap()
                .durable_media_ready
        );
        lifecycle
    }

    #[test]
    fn cleanup_requires_trigger_created_gc_pending_before_unlink() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "instance-a",
        ));
        let owner = journal.owner_uuid().unwrap().to_string();
        let set = QueueMediaStore::open(home.path())
            .unwrap()
            .store
            .seal(
                &owner,
                "media-job",
                vec![SealMedia::bytes("source", "source.png", vec![1, 2, 3])],
            )
            .unwrap();
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "media-job", &set.set_id),
            &obligation(&owner, &set.set_id),
        )
        .unwrap();

        let lifecycle = Arc::new(QueueMediaLifecycle::new(
            db.clone(),
            home.path().to_path_buf(),
            owner.clone(),
        ));
        journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .unwrap();
        let report = reconcile_claimed_owner(&journal, lifecycle.as_ref()).unwrap();
        assert!(report.durable_media_ready);

        let candidate = lifecycle.candidate_for_job("media-job").unwrap().unwrap();
        assert_eq!(
            lifecycle
                .cleanup_after_committed_delete(&candidate)
                .unwrap(),
            CleanupOutcome::Retained
        );
        assert_eq!(
            generation_queue_media::obligation_by_id(
                db.as_ref().as_ref().unwrap(),
                &owner,
                &set.set_id,
            )
            .unwrap()
            .unwrap()
            .state,
            QueueMediaObligationState::Active
        );

        assert!(generation_queue::delete(db.as_ref().as_ref().unwrap(), "media-job").unwrap());
        assert_eq!(
            lifecycle
                .cleanup_after_committed_delete(&candidate)
                .unwrap(),
            CleanupOutcome::Deleted
        );
        assert!(generation_queue_media::obligation_by_id(
            db.as_ref().as_ref().unwrap(),
            &owner,
            &set.set_id,
        )
        .unwrap()
        .is_none());
        assert!(QueueMediaStore::open_existing(home.path())
            .unwrap()
            .inspect_owner(&owner)
            .active
            .is_empty());
    }

    #[test]
    fn adapter_rejects_an_under_reported_queue_media_marker_as_invariant() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = QueueJournal::new(db.clone(), Some(home.path()), "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let set = QueueMediaStore::open(home.path())
            .unwrap()
            .store
            .seal(
                &owner,
                "media-job",
                vec![SealMedia::bytes("source", "source.png", vec![1])],
            )
            .unwrap();
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "media-job", &set.set_id),
            &obligation(&owner, &set.set_id),
        )
        .unwrap();
        db.as_ref()
            .as_ref()
            .unwrap()
            .with_conn(|conn| {
                conn.execute_batch(
                    "PRAGMA foreign_keys = OFF;
                     DELETE FROM generation_queue_media;
                     PRAGMA foreign_keys = ON;",
                )
                .map_err(Into::into)
            })
            .unwrap();

        let lifecycle = QueueMediaLifecycle::new(db, home.path().to_path_buf(), owner);
        let error = lifecycle
            .obligations(journal.owner_uuid().unwrap())
            .unwrap_err();
        assert_eq!(error.kind, AdapterFailureKind::Invariant);
    }

    #[test]
    fn startup_reconciliation_scans_owner_once_before_all_mutations() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = QueueJournal::new(db.clone(), Some(home.path()), "instance-a");
        let owner = journal.owner_uuid().unwrap().to_string();
        let store = QueueMediaStore::open(home.path()).unwrap().store;

        let restore_set = store
            .seal(
                &owner,
                "restore-job",
                vec![SealMedia::bytes("source", "restore.png", vec![1])],
            )
            .unwrap();
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "restore-job", &restore_set.set_id),
            &obligation(&owner, &restore_set.set_id),
        )
        .unwrap();
        store.retire(&restore_set).unwrap();

        let gc_set = store
            .seal(
                &owner,
                "gc-job",
                vec![SealMedia::bytes("source", "gc.png", vec![2])],
            )
            .unwrap();
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "gc-job", &gc_set.set_id),
            &obligation(&owner, &gc_set.set_id),
        )
        .unwrap();
        assert!(generation_queue::delete(db.as_ref().as_ref().unwrap(), "gc-job").unwrap());

        let orphan_set = store
            .seal(
                &owner,
                "orphan-job",
                vec![SealMedia::bytes("source", "orphan.png", vec![3])],
            )
            .unwrap();

        let lifecycle =
            QueueMediaLifecycle::new(db.clone(), home.path().to_path_buf(), owner.clone());
        let report = reconcile_claimed_owner(&journal, &lifecycle).unwrap();

        assert!(report.durable_media_ready, "{:#?}", report.issues);
        assert_eq!(report.restored, vec![restore_set.set_id.clone()]);
        assert_eq!(
            report.deleted,
            vec![gc_set.set_id.clone(), orphan_set.set_id.clone()]
        );
        assert_eq!(
            lifecycle.opened_store().unwrap().inspection_calls(),
            1,
            "restore and delete must reuse identities authenticated by the initial inspection"
        );

        let inspection = store.inspect_owner(&owner);
        assert_eq!(inspection.active, vec![restore_set]);
        assert!(inspection.retired.is_empty());
        assert!(inspection.staging.is_empty());
    }

    #[test]
    fn journal_terminal_and_bulk_paths_converge_on_trigger_proven_cleanup() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "instance-a",
        ));
        let owner = journal.owner_uuid().unwrap().to_string();
        let store = QueueMediaStore::open(home.path()).unwrap().store;
        let complete_set = store
            .seal(
                &owner,
                "complete-job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        generation_queue::insert_claimed_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "complete-job", &complete_set.set_id),
            "complete-token",
            &obligation(&owner, &complete_set.set_id),
        )
        .unwrap();
        let bulk_set = store
            .seal(
                &owner,
                "bulk-job",
                vec![SealMedia::bytes("source", "two", vec![2])],
            )
            .unwrap();
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "bulk-job", &bulk_set.set_id),
            &obligation(&owner, &bulk_set.set_id),
        )
        .unwrap();
        install_and_reconcile(home.path(), db.clone(), &journal);

        journal
            .attach_claimed("complete-job", "complete-token".to_string())
            .complete_before_dispatch();
        assert!(generation_queue_media::obligation_by_id(
            db.as_ref().as_ref().unwrap(),
            &owner,
            &complete_set.set_id,
        )
        .unwrap()
        .is_none());

        assert_eq!(journal.cancel_all_queued(&[]).unwrap(), 1);
        assert!(generation_queue_media::list_obligations(
            db.as_ref().as_ref().unwrap(),
            &owner,
            QueueMediaObligationState::GcPending,
        )
        .unwrap()
        .is_empty());
        let inspection = QueueMediaStore::open_existing(home.path())
            .unwrap()
            .inspect_owner(&owner);
        assert!(inspection.active.is_empty());
        assert!(inspection.retired.is_empty());
    }

    #[test]
    fn claimed_batch_cancellation_retains_media_until_token_settlement() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "instance-a",
        ));
        let owner = journal.owner_uuid().unwrap().to_string();
        let set = QueueMediaStore::open(home.path())
            .unwrap()
            .store
            .seal(
                &owner,
                "batch-child",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        let row = queue_row(&owner, "batch-child", &set.set_id);
        generation_batches::insert_or_get_with_media(
            db.as_ref().as_ref().unwrap(),
            &GenerationBatchRow {
                id: "batch".to_string(),
                client_batch_id: "client-batch".to_string(),
                owner_uuid: owner.clone(),
                request_sha256: "request".to_string(),
                created_at_ms: 1,
            },
            &[(
                GenerationBatchChildRow {
                    batch_id: "batch".to_string(),
                    job_id: "batch-child".to_string(),
                    batch_index: 1,
                    state: "accepted".to_string(),
                    error: None,
                    updated_at_ms: 1,
                },
                row,
            )],
            &[obligation(&owner, &set.set_id)],
        )
        .unwrap();
        install_and_reconcile(home.path(), db.clone(), &journal);
        let claim = journal.claim_feeder_by_id("batch-child").unwrap().unwrap();

        assert!(journal.cancel_id("batch-child").unwrap());
        assert!(
            generation_queue::get(db.as_ref().as_ref().unwrap(), "batch-child")
                .unwrap()
                .is_some()
        );
        assert_eq!(
            generation_queue_media::obligation_by_id(
                db.as_ref().as_ref().unwrap(),
                &owner,
                &set.set_id,
            )
            .unwrap()
            .unwrap()
            .state,
            QueueMediaObligationState::Active
        );

        journal
            .attach_claimed("batch-child", claim.claim_token)
            .discard();
        assert!(
            generation_queue::get(db.as_ref().as_ref().unwrap(), "batch-child")
                .unwrap()
                .is_none()
        );
        assert!(generation_queue_media::obligation_by_id(
            db.as_ref().as_ref().unwrap(),
            &owner,
            &set.set_id,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn held_and_shutdown_paths_retain_media_and_active_obligations() {
        for mode in ["held", "transient", "shutdown"] {
            let home = tempfile::tempdir().unwrap();
            let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
            let journal = Arc::new(QueueJournal::new(
                db.clone(),
                Some(home.path()),
                &format!("instance-{mode}"),
            ));
            let owner = journal.owner_uuid().unwrap().to_string();
            let set = QueueMediaStore::open(home.path())
                .unwrap()
                .store
                .seal(
                    &owner,
                    "media-job",
                    vec![SealMedia::bytes("source", "one", vec![1])],
                )
                .unwrap();
            generation_queue::insert_claimed_with_media(
                db.as_ref().as_ref().unwrap(),
                &queue_row(&owner, "media-job", &set.set_id),
                "claim-token",
                &obligation(&owner, &set.set_id),
            )
            .unwrap();
            install_and_reconcile(home.path(), db.clone(), &journal);
            let ticket = journal.attach_claimed("media-job", "claim-token".to_string());
            if mode == "held" {
                ticket.hold("wait for operator");
            } else if mode == "transient" {
                assert!(matches!(
                    ticket.retain(),
                    crate::queue_journal::RetainOutcome::Released
                ));
            } else {
                journal.retain_all();
                drop(ticket);
            }

            assert!(
                generation_queue::get(db.as_ref().as_ref().unwrap(), "media-job")
                    .unwrap()
                    .is_some()
            );
            assert_eq!(
                generation_queue_media::obligation_by_id(
                    db.as_ref().as_ref().unwrap(),
                    &owner,
                    &set.set_id,
                )
                .unwrap()
                .unwrap()
                .state,
                QueueMediaObligationState::Active
            );
            assert_eq!(
                QueueMediaStore::open_existing(home.path())
                    .unwrap()
                    .inspect_owner(&owner)
                    .active,
                vec![set]
            );
        }
    }

    #[test]
    fn failed_unlink_leaves_the_trigger_obligation_for_startup_gc() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "instance-a",
        ));
        let owner = journal.owner_uuid().unwrap().to_string();
        let store = QueueMediaStore::open(home.path()).unwrap().store;
        let set = store
            .seal(
                &owner,
                "media-job",
                vec![SealMedia::bytes("source", "one", vec![1])],
            )
            .unwrap();
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &queue_row(&owner, "media-job", &set.set_id),
            &obligation(&owner, &set.set_id),
        )
        .unwrap();
        install_and_reconcile(home.path(), db.clone(), &journal);

        // Simulate an out-of-band filesystem loss after startup. The terminal
        // delete still commits, but NotFound is not a successful unlink and
        // therefore cannot erase the DB's cleanup obligation.
        store.delete(&set).unwrap();
        assert!(journal.cancel_id("media-job").unwrap());
        assert_eq!(
            generation_queue_media::obligation_by_id(
                db.as_ref().as_ref().unwrap(),
                &owner,
                &set.set_id,
            )
            .unwrap()
            .unwrap()
            .state,
            QueueMediaObligationState::GcPending
        );
    }
}
