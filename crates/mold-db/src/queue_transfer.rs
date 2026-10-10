//! Durable source reservations for a queue transfer. No automatic expiry:
//! lost destination responses must never restart the original concurrently.

use crate::MetadataDb;
use anyhow::Result;
use rusqlite::{params, OptionalExtension};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveOutcome {
    Reserved,
    Existing,
    Conflict,
    NotEligible,
    NotOwned,
}

/// Called with the server durable-transition and scheduler dispatch fences.
/// Pausing the durable row prevents feeder/restart dispatch; the caller also
/// pauses any hydrated registry row before releasing its exclusion token.
pub fn reserve(
    db: &MetadataDb,
    owner: &str,
    job: &str,
    transfer: &str,
    destination: &str,
    now: i64,
) -> Result<ReserveOutcome> {
    db.transact_immediate(|conn| {
        if let Some((id,dest)) = conn.query_row("SELECT transfer_id,destination_identity FROM generation_queue_transfers WHERE job_id=?1 AND owner_uuid=?2",params![job,owner],|row|Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?))).optional()? {
            return Ok(if id==transfer && dest==destination {ReserveOutcome::Existing} else {ReserveOutcome::Conflict});
        }
        let Some((state, explicit)) = conn.query_row("SELECT state,explicitly_paused FROM generation_queue WHERE id=?1 AND owner_uuid=?2",params![job,owner],|row|Ok((row.get::<_,String>(0)?,row.get::<_,i64>(1)?))).optional()? else {return Ok(ReserveOutcome::NotOwned)};
        if !matches!(state.as_str(),"queued"|"paused"|"held") {return Ok(ReserveOutcome::NotEligible)}
        conn.execute("INSERT INTO generation_queue_transfers (job_id,owner_uuid,transfer_id,destination_identity,original_state,original_explicit_pause,created_at) VALUES (?1,?2,?3,?4,?5,?6,?7)",params![job,owner,transfer,destination,state,explicit,now])?;
        if state=="queued" || state=="paused" {
            conn.execute("UPDATE generation_queue SET state='paused',explicitly_paused=1,updated_at=MAX(updated_at+1,?3) WHERE id=?1 AND owner_uuid=?2",params![job,owner,now])?;
            conn.execute("UPDATE generation_batch_children SET state='paused',revision=revision+1,updated_at_ms=MAX(updated_at_ms+1,?2) WHERE job_id=?1 AND state='accepted'",params![job,now])?;
        }
        Ok(ReserveOutcome::Reserved)
    })
}

pub fn contains_on_conn(conn: &rusqlite::Connection, job: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM generation_queue_transfers WHERE job_id=?1)",
        params![job],
        |row| row.get(0),
    )?)
}

pub fn get(db: &MetadataDb, owner: &str, job: &str) -> Result<Option<(String, String)>> {
    db.with_conn(|conn| Ok(conn.query_row("SELECT transfer_id,destination_identity FROM generation_queue_transfers WHERE job_id=?1 AND owner_uuid=?2",params![job,owner],|row|Ok((row.get(0)?,row.get(1)?))).optional()?))
}

pub fn any(db: &MetadataDb, owner: &str) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM generation_queue_transfers WHERE owner_uuid=?1)",
            params![owner],
            |row| row.get(0),
        )?)
    })
}

pub fn contains(db: &MetadataDb, owner: &str, job: &str) -> Result<bool> {
    db.with_conn(|conn| Ok(conn.query_row("SELECT EXISTS (SELECT 1 FROM generation_queue_transfers WHERE job_id=?1 AND owner_uuid=?2)",params![job,owner],|row|row.get(0))?))
}

/// Destination admission and abort share an immediate transaction with batch inserts.
pub fn abort_destination(
    db: &MetadataDb,
    owner: &str,
    transfer: &str,
    receipt: &str,
) -> Result<Option<String>> {
    db.transact_immediate(|conn| {
        let accepted: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM generation_batches WHERE owner_uuid=?1 AND client_batch_id=?2)",params![owner,transfer],|r|r.get(0))?;
        if accepted {
            let terminal_single: bool=conn.query_row("SELECT COUNT(*)=1 AND SUM(c.state IN ('failed','cancelled'))=1 FROM generation_batch_children c JOIN generation_batches b ON b.id=c.batch_id WHERE b.owner_uuid=?1 AND b.client_batch_id=?2",params![owner,transfer],|r|r.get(0))?;
            if !terminal_single { return Ok(None) }
        }
        conn.execute("INSERT OR IGNORE INTO generation_transfer_aborts(owner_uuid,transfer_id,receipt) VALUES (?1,?2,?3)",params![owner,transfer,receipt])?;
        Ok(Some(conn.query_row("SELECT receipt FROM generation_transfer_aborts WHERE owner_uuid=?1 AND transfer_id=?2",params![owner,transfer],|r|r.get(0))?))
    })
}
pub fn admission_aborted(conn: &rusqlite::Connection, owner: &str, transfer: &str) -> Result<bool> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM generation_transfer_aborts WHERE owner_uuid=?1 AND transfer_id=?2)",params![owner,transfer],|r|r.get(0))?)
}
/// Seal before any destination admission. Once sealed, no client may restore the source.
pub fn seal(
    db: &MetadataDb,
    owner: &str,
    job: &str,
    transfer: &str,
    destination: &str,
) -> Result<bool> {
    db.transact_immediate(|conn| Ok(conn.execute("UPDATE generation_queue_transfers SET sealed=1 WHERE job_id=?1 AND owner_uuid=?2 AND transfer_id=?3 AND destination_identity=?4",params![job,owner,transfer,destination])? == 1))
}

/// Explicit release is only safe after the client knows no destination admit
/// was attempted by ANY client. The durable seal fences concurrent admission.
pub fn release(db: &MetadataDb, owner: &str, job: &str, transfer: &str, now: i64) -> Result<bool> {
    release_inner(db, owner, job, transfer, "", now)
}
pub fn release_aborted(
    db: &MetadataDb,
    owner: &str,
    job: &str,
    transfer: &str,
    destination: &str,
    now: i64,
) -> Result<bool> {
    release_inner(db, owner, job, transfer, destination, now)
}
fn release_inner(
    db: &MetadataDb,
    owner: &str,
    job: &str,
    transfer: &str,
    destination: &str,
    now: i64,
) -> Result<bool> {
    db.transact_immediate(|conn| {
        let Some((state,explicit)) = conn.query_row("SELECT original_state,original_explicit_pause FROM generation_queue_transfers WHERE job_id=?1 AND owner_uuid=?2 AND transfer_id=?3 AND ((?4='' AND sealed=0) OR (?4<>'' AND destination_identity=?4))",params![job,owner,transfer,destination],|row|Ok((row.get::<_,String>(0)?,row.get::<_,i64>(1)?))).optional()? else {return Ok(false)};
        conn.execute("UPDATE generation_queue SET state=?3,explicitly_paused=?4,updated_at=MAX(updated_at+1,?5) WHERE id=?1 AND owner_uuid=?2 AND state IN ('paused','held')",params![job,owner,state,explicit,now])?;
        if state=="queued" {
            conn.execute("UPDATE generation_batch_children SET state='accepted',revision=revision+1,updated_at_ms=MAX(updated_at_ms+1,?2) WHERE job_id=?1 AND state='paused'",params![job,now])?;
        }
        conn.execute("DELETE FROM generation_queue_transfers WHERE job_id=?1 AND owner_uuid=?2 AND transfer_id=?3",params![job,owner,transfer])?;
        Ok(true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generation_queue, MetadataDb};

    #[test]
    fn seal_and_release_have_one_winner_across_clients() {
        for seal_first in [false, true] {
            let db = MetadataDb::open_in_memory().unwrap();
            row(&db, "job", "queued", true);
            assert_eq!(
                reserve(&db, "owner", "job", "transfer", "destination", 2).unwrap(),
                ReserveOutcome::Reserved
            );
            if seal_first {
                assert!(seal(&db, "owner", "job", "transfer", "destination").unwrap());
                assert!(!release(&db, "owner", "job", "transfer", 3).unwrap());
                assert!(!seal(&db, "owner", "job", "transfer", "other").unwrap());
                assert!(contains(&db, "owner", "job").unwrap());
            } else {
                assert!(release(&db, "owner", "job", "transfer", 3).unwrap());
                assert!(!seal(&db, "owner", "job", "transfer", "destination").unwrap());
            }
        }
    }

    #[test]
    fn destination_abort_and_admission_have_one_winner_and_survive_restart() {
        for abort_first in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("destination.db");
            {
                let db = MetadataDb::open(&path).unwrap();
                let batch = crate::generation_batches::GenerationBatchRow {
                    id: "batch".into(),
                    client_batch_id: "transfer".into(),
                    owner_uuid: "owner".into(),
                    request_sha256: "hash".into(),
                    created_at_ms: 1,
                };
                if abort_first {
                    assert_eq!(
                        abort_destination(&db, "owner", "transfer", "receipt").unwrap(),
                        Some("receipt".into())
                    );
                    assert!(crate::generation_batches::insert_or_get(&db, &batch, &[]).is_err());
                } else {
                    assert!(crate::generation_batches::insert_or_get(&db, &batch, &[]).is_ok());
                    assert_eq!(
                        abort_destination(&db, "owner", "transfer", "receipt").unwrap(),
                        None
                    );
                }
            }
            let db = MetadataDb::open(&path).unwrap();
            assert_eq!(
                abort_destination(&db, "owner", "transfer", "new-receipt").unwrap(),
                if abort_first {
                    Some("receipt".into())
                } else {
                    None
                }
            );
        }
    }

    #[test]
    fn concurrent_destination_admit_and_abort_cannot_both_win() {
        for _ in 0..8 {
            let db = std::sync::Arc::new(MetadataDb::open_in_memory().unwrap());
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
            let admit_db = db.clone();
            let admit_barrier = barrier.clone();
            let admit = std::thread::spawn(move || {
                admit_barrier.wait();
                let batch = crate::generation_batches::GenerationBatchRow {
                    id: "batch".into(),
                    client_batch_id: "transfer".into(),
                    owner_uuid: "owner".into(),
                    request_sha256: "hash".into(),
                    created_at_ms: 1,
                };
                crate::generation_batches::insert_or_get(&admit_db, &batch, &[]).is_ok()
            });
            barrier.wait();
            let aborted = abort_destination(&db, "owner", "transfer", "receipt")
                .unwrap()
                .is_some();
            assert_ne!(admit.join().unwrap(), aborted);
        }
    }
    #[test]
    fn matching_abort_restores_sealed_source_but_other_destination_cannot() {
        let db = MetadataDb::open_in_memory().unwrap();
        row(&db, "job", "queued", true);
        reserve(&db, "owner", "job", "transfer", "destination", 2).unwrap();
        seal(&db, "owner", "job", "transfer", "destination").unwrap();
        assert!(!release_aborted(&db, "owner", "job", "transfer", "other", 3).unwrap());
        assert!(release_aborted(&db, "owner", "job", "transfer", "destination", 3).unwrap());
        assert_eq!(
            generation_queue::get(&db, "job").unwrap().unwrap().state,
            generation_queue::QueueRowState::Queued
        );
        assert!(!seal(&db, "owner", "job", "transfer", "destination").unwrap());
    }

    #[test]
    fn only_unsuccessful_terminal_singletons_can_abort_after_acceptance() {
        for state in [
            "accepted",
            "held",
            "paused",
            "running",
            "cancelling",
            "complete",
            "failed",
            "cancelled",
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("terminal.db");
            {
                let db = MetadataDb::open(&path).unwrap();
                let batch = crate::generation_batches::GenerationBatchRow {
                    id: "batch".into(),
                    client_batch_id: "transfer".into(),
                    owner_uuid: "owner".into(),
                    request_sha256: "hash".into(),
                    created_at_ms: 1,
                };
                crate::generation_batches::insert_or_get(&db, &batch, &[]).unwrap();
                db.with_conn(|conn| { conn.execute("INSERT INTO generation_batch_children(batch_id,job_id,batch_index,state,updated_at_ms) VALUES ('batch','child',0,?1,1)",params![state])?;Ok(()) }).unwrap();
                let aborted = abort_destination(&db, "owner", "transfer", "receipt").unwrap();
                assert_eq!(
                    aborted.is_some(),
                    matches!(state, "failed" | "cancelled"),
                    "{state}"
                );
                if aborted.is_some() {
                    assert!(crate::generation_batches::insert_or_get(&db, &batch, &[]).is_err());
                }
            }
            let db = MetadataDb::open(&path).unwrap();
            assert_eq!(
                abort_destination(&db, "owner", "transfer", "later").unwrap(),
                if matches!(state, "failed" | "cancelled") {
                    Some("receipt".into())
                } else {
                    None
                }
            );
        }
    }

    #[test]
    fn held_retention_cannot_delete_an_unresolved_transfer_or_its_media_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        row(&db, "job", "held", false);
        reserve(&db, "owner", "job", "transfer", "destination", 2).unwrap();
        assert!(!generation_queue::purge_held(&db, "owner", "job", 99).unwrap());
        assert!(get(&db, "owner", "job").unwrap().is_some());
        assert!(generation_queue::get(&db, "job").unwrap().is_some());
        seal(&db, "owner", "job", "transfer", "destination").unwrap();
        assert!(!generation_queue::purge_held(&db, "owner", "job", 100).unwrap());
    }

    fn row(db: &MetadataDb, id: &str, state: &str, claimed: bool) {
        db.with_conn(|conn| {
            conn.execute("INSERT INTO generation_queue (id,owner_uuid,request_json,model,state,created_at,updated_at,dispatch_attempts,replay_seen,explicitly_paused,claim_token,output_dir,completion_payload) VALUES (?1,'owner','{}','model',?2,1,1,0,0,0,?3,'/gallery','metadata_only')", rusqlite::params![id,state,claimed.then_some("claim")])?;
            Ok(())
        }).unwrap();
    }

    #[test]
    fn reservation_fences_resume_and_cross_destination_retry() {
        let db = MetadataDb::open_in_memory().unwrap();
        row(&db, "job", "queued", true);
        assert_eq!(
            reserve(&db, "owner", "job", "transfer-a", "dest-a", 2).unwrap(),
            ReserveOutcome::Reserved
        );
        assert_eq!(
            reserve(&db, "owner", "job", "transfer-a", "dest-a", 3).unwrap(),
            ReserveOutcome::Existing
        );
        assert_eq!(
            reserve(&db, "owner", "job", "transfer-b", "dest-b", 3).unwrap(),
            ReserveOutcome::Conflict
        );
        assert_eq!(
            generation_queue::set_owned_job_paused(&db, "owner", "job", false, 4).unwrap(),
            generation_queue::OwnedJobPauseOutcome::NotEligible
        );
        assert_eq!(
            generation_queue::resume_all_paused(&db, "owner", 5)
                .unwrap()
                .generation_jobs,
            0
        );
        assert_eq!(
            generation_queue::get(&db, "job").unwrap().unwrap().state,
            generation_queue::QueueRowState::Paused
        );
        assert!(!release(&db, "owner", "job", "wrong", 6).unwrap());
        assert!(release(&db, "owner", "job", "transfer-a", 6).unwrap());
        assert_eq!(
            generation_queue::get(&db, "job").unwrap().unwrap().state,
            generation_queue::QueueRowState::Queued
        );
    }

    #[test]
    fn reservation_survives_disk_reopen_and_runtime_claim_recovery() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("queue.db");
        {
            let db = MetadataDb::open(&path).unwrap();
            row(&db, "job", "queued", true);
            assert_eq!(
                reserve(&db, "owner", "job", "original-transfer", "destination", 2).unwrap(),
                ReserveOutcome::Reserved
            );
        }
        let db = MetadataDb::open(&path).unwrap();
        let recovered =
            generation_queue::recover_runtime_claims_and_charge_replays(&db, "owner", 3, 3)
                .unwrap();
        assert_eq!(recovered.claims_cleared, 1);
        assert_eq!(
            get(&db, "owner", "job").unwrap(),
            Some(("original-transfer".into(), "destination".into()))
        );
        assert_eq!(
            generation_queue::resume_all_paused(&db, "owner", 4)
                .unwrap()
                .generation_jobs,
            0
        );
        assert!(generation_queue::claim_next(&db, "owner", "new-runtime", 5)
            .unwrap()
            .is_none());
        assert_eq!(
            reserve(&db, "owner", "job", "original-transfer", "destination", 6).unwrap(),
            ReserveOutcome::Existing
        );
        assert!(release(&db, "owner", "job", "original-transfer", 7).unwrap());
        assert!(generation_queue::claim_next(&db, "owner", "new-runtime", 8)
            .unwrap()
            .is_some());
    }

    #[test]
    fn sealed_admission_survives_disk_reopen() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("sealed.db");
        {
            let db = MetadataDb::open(&path).unwrap();
            row(&db, "job", "queued", true);
            reserve(&db, "owner", "job", "transfer", "destination", 2).unwrap();
            assert!(seal(&db, "owner", "job", "transfer", "destination").unwrap());
        }
        let db = MetadataDb::open(&path).unwrap();
        assert!(!release(&db, "owner", "job", "transfer", 3).unwrap());
        assert_eq!(
            get(&db, "owner", "job").unwrap(),
            Some(("transfer".into(), "destination".into()))
        );
        assert_eq!(
            generation_queue::get(&db, "job").unwrap().unwrap().state,
            generation_queue::QueueRowState::Paused
        );
    }

    #[test]
    fn reservation_never_takes_running_work_and_restores_paused_or_held() {
        let db = MetadataDb::open_in_memory().unwrap();
        for state in ["running", "paused", "held"] {
            row(&db, state, state, false);
            let expected = if state == "running" {
                ReserveOutcome::NotEligible
            } else {
                ReserveOutcome::Reserved
            };
            assert_eq!(
                reserve(&db, "owner", state, state, "dest", 2).unwrap(),
                expected
            );
            if state != "running" {
                assert!(release(&db, "owner", state, state, 3).unwrap());
                assert_eq!(
                    generation_queue::get(&db, state)
                        .unwrap()
                        .unwrap()
                        .state
                        .as_str(),
                    state
                );
            }
        }
    }
}
