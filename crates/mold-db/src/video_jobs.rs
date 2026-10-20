//! Durable parent ledger; inference remains owned by bounded chain children.

use crate::MetadataDb;
use anyhow::{bail, Context, Result};
use mold_core::video_job::{VideoJob, VideoJobState};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct StoredVideoJob {
    pub job: VideoJob,
    /// Canonical logical request identity, independent of operation UUID.
    pub request_sha256: String,
    /// Server-authored, scrubbed manifest. Never contains inline media or secrets.
    pub manifest_json: String,
    pub manifest_sha256: String,
    pub revision: u64,
    pub cancel_requested: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredVideoJobPart {
    pub parent_id: String,
    pub index: u64,
    pub child_id: String,
    pub first_clip: u64,
    pub clip_count: u32,
    pub keep_frames: u64,
    pub output_sha256: Option<String>,
}

fn state(value: VideoJobState) -> &'static str {
    match value {
        VideoJobState::Queued => "queued",
        VideoJobState::Running => "running",
        VideoJobState::Paused => "paused",
        VideoJobState::Completed => "completed",
        VideoJobState::Failed => "failed",
        VideoJobState::Interrupted => "interrupted",
        VideoJobState::Cancelled => "cancelled",
    }
}
fn decode(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredVideoJob> {
    let json: String = row.get(0)?;
    let job = serde_json::from_str(&json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(StoredVideoJob {
        job,
        request_sha256: row.get(1)?,
        manifest_json: row.get(2)?,
        manifest_sha256: row.get(5)?,
        revision: row.get(3)?,
        cancel_requested: row.get(4)?,
    })
}
pub fn get(db: &MetadataDb, id: &str) -> Result<Option<StoredVideoJob>> {
    db.with_conn(|conn|conn.query_row("SELECT job_json,request_sha256,manifest_json,revision,cancel_requested,manifest_sha256 FROM video_jobs WHERE id=?1",[id],decode).optional().map_err(Into::into))
}

/// Returns false only for an already accepted exact logical operation.
pub fn insert(db: &MetadataDb, value: &StoredVideoJob) -> Result<bool> {
    db.transact_immediate(|conn| {
        if conn.query_row("SELECT 1 FROM video_job_aborts WHERE id=?1",[&value.job.id],|row|row.get::<_,i64>(0)).optional()?.is_some() { bail!("video operation was cancelled before admission"); }
        let old:Option<String>=conn.query_row("SELECT request_sha256 FROM video_jobs WHERE id=?1",[&value.job.id],|row|row.get(0)).optional()?;
        if let Some(old)=old {
            if old!=value.request_sha256 { bail!("video operation identity already belongs to another request"); }
            return Ok(false);
        }
        conn.execute("INSERT INTO video_jobs(id,state,request_sha256,manifest_json,job_json,revision,cancel_requested,created_at_ms,updated_at_ms,manifest_sha256) VALUES(?1,?2,?3,?4,?5,?6,?9,?7,?8,?10)",params![value.job.id,state(value.job.state),value.request_sha256,value.manifest_json,serde_json::to_string(&value.job)?,value.revision,value.job.created_at_ms,value.job.updated_at_ms,value.cancel_requested,value.manifest_sha256])?;
        Ok(true)
    })
}

/// Compare-and-swap the parent and optional child claim in ONE transaction.
/// Callers reload after false: cancellation/recovery may have won the fence.
pub fn checkpoint(
    db: &MetadataDb,
    value: &StoredVideoJob,
    part: Option<&StoredVideoJobPart>,
) -> Result<bool> {
    db.transact_immediate(|conn| {
        let next=value.revision.checked_add(1).context("video job revision exhausted")?;
        let updated=conn.execute("UPDATE video_jobs SET state=?2,job_json=?3,revision=?4,updated_at_ms=?5 WHERE id=?1 AND revision=?6 AND request_sha256=?7 AND manifest_sha256=?8",params![value.job.id,state(value.job.state),serde_json::to_string(&value.job)?,next,value.job.updated_at_ms,value.revision,value.request_sha256,value.manifest_sha256])?;
        if updated==0 { return Ok(false); }
        if let Some(part)=part {
            if part.parent_id!=value.job.id || part.clip_count==0 || part.clip_count>16 || part.keep_frames==0 { bail!("invalid bounded video child claim"); }
            let existing:Option<String>=conn.query_row("SELECT part_json FROM video_job_parts WHERE parent_id=?1 AND part_index=?2",params![part.parent_id,part.index.to_string()],|row|row.get(0)).optional()?;
            if let Some(existing)=existing {
                let old:StoredVideoJobPart=serde_json::from_str(&existing)?;
                if old.child_id!=part.child_id || old.first_clip!=part.first_clip || old.clip_count!=part.clip_count || old.keep_frames!=part.keep_frames || old.output_sha256.as_ref().is_some_and(|sha|Some(sha)!=part.output_sha256.as_ref()) { bail!("video child claim authority changed"); }
            }
            conn.execute("INSERT INTO video_job_parts(parent_id,part_index,child_id,part_json) VALUES(?1,?2,?3,?4) ON CONFLICT(parent_id,part_index) DO UPDATE SET part_json=excluded.part_json",params![part.parent_id,part.index.to_string(),part.child_id,serde_json::to_string(part)?])?;
        }
        Ok(true)
    })
}

pub fn get_part(db: &MetadataDb, id: &str, index: u64) -> Result<Option<StoredVideoJobPart>> {
    db.with_conn(|conn| {
        let json: Option<String> = conn
            .query_row(
                "SELECT part_json FROM video_job_parts WHERE parent_id=?1 AND part_index=?2",
                params![id, index.to_string()],
                |row| row.get(0),
            )
            .optional()?;
        json.map(|json| serde_json::from_str(&json).context("decoding video child ledger"))
            .transpose()
    })
}

/// Paginate scalar projections; never collect lifetime artifacts into RAM.
pub fn list(db: &MetadataDb, limit: u32, offset: u64) -> Result<Vec<VideoJob>> {
    db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT job_json FROM video_jobs ORDER BY created_at_ms DESC,id LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt.query_map(params![limit.min(256), offset], |row| {
            row.get::<_, String>(0)
        })?;
        rows.map(|row| Ok(serde_json::from_str(&row?)?)).collect()
    })
}

/// Cancellation intent is durable even when it beats the create POST. Running
/// state remains owned until the child worker acknowledges its safe point.
pub fn request_cancel(db: &MetadataDb, id: &str, now_ms: i64) -> Result<bool> {
    db.transact_immediate(|conn| {
        let current:Option<String>=conn.query_row("SELECT job_json FROM video_jobs WHERE id=?1",[id],|row|row.get(0)).optional()?;
        let Some(json)=current else {
            conn.execute("INSERT INTO video_job_aborts(id,created_at_ms) VALUES(?1,?2) ON CONFLICT(id) DO NOTHING",params![id,now_ms])?;
            return Ok(true);
        };
        let mut job:VideoJob=serde_json::from_str(&json)?;
        if job.state.is_terminal() { return Ok(false); }
        job.can_cancel=false;job.can_resume=false;job.updated_at_ms=now_ms;
        conn.execute("UPDATE video_jobs SET cancel_requested=1,job_json=?2,revision=revision+1,updated_at_ms=?3 WHERE id=?1",params![id,serde_json::to_string(&job)?,now_ms])?;
        Ok(true)
    })
}

pub fn is_aborted(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn
            .query_row("SELECT 1 FROM video_job_aborts WHERE id=?1", [id], |row| {
                row.get::<_, i64>(0)
            })
            .optional()?
            .is_some())
    })
}

pub fn recover_at_startup(db: &MetadataDb, now_ms: i64) -> Result<usize> {
    db.transact_immediate(|conn| {
        let mut stmt=conn.prepare("SELECT job_json FROM video_jobs WHERE state IN ('queued','running')")?;
        let mut rows=stmt.query([])?;
        let mut count=0;
        while let Some(row)=rows.next()? {
            let mut job:VideoJob=serde_json::from_str(&row.get::<_,String>(0)?)?;
            job.state=VideoJobState::Interrupted;job.can_cancel=false;job.can_resume=true;job.can_delete=true;job.updated_at_ms=now_ms;
            conn.execute("UPDATE video_jobs SET state='interrupted',job_json=?2,revision=revision+1,updated_at_ms=?3 WHERE id=?1 AND state IN ('queued','running')",params![job.id,serde_json::to_string(&job)?,now_ms])?;
            count+=1;
        }
        Ok(count)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::video_job::*;

    fn stored() -> StoredVideoJob {
        let plan = VideoJobPlan {
            validation_revision: "proof".into(),
            model: "model".into(),
            width: 512,
            height: 512,
            fps: 24,
            single_clip_max_frames: 97,
            overlap_frames: 17,
            join_mode: VideoJoinMode::Continuous,
            clip_count: 100,
            part_count: 7,
            added_frames: 8040,
            source_frames: 0,
            total_frames: 8040,
            added_duration_ms: 335000,
            total_duration_ms: 335000,
            warnings: vec![],
        };
        StoredVideoJob {
            job: VideoJob {
                id: "parent".into(),
                state: VideoJobState::Queued,
                plan,
                completed_clips: 0,
                active_child_id: None,
                output_filename: None,
                error: None,
                error_detail: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                can_cancel: true,
                can_resume: false,
                can_delete: true,
            },
            request_sha256: "request".into(),
            manifest_json: "{}".into(),
            manifest_sha256: "sha".into(),
            revision: 0,
            cancel_requested: false,
        }
    }

    #[test]
    fn preadmission_cancel_and_admission_are_mutually_exclusive() {
        let db = MetadataDb::open_in_memory().unwrap();
        assert!(request_cancel(&db, "parent", 5).unwrap());
        assert!(is_aborted(&db, "parent").unwrap());
        assert!(insert(&db, &stored()).is_err());
        assert!(request_cancel(&db, "parent", 6).unwrap());
    }

    #[test]
    fn cancel_intent_does_not_release_running_child() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = stored();
        a.job.state = VideoJobState::Running;
        a.job.active_child_id = Some("child".into());
        insert(&db, &a).unwrap();
        assert!(request_cancel(&db, "parent", 5).unwrap());
        let current = get(&db, "parent").unwrap().unwrap();
        assert!(current.cancel_requested);
        assert_eq!(current.job.state, VideoJobState::Running);
        assert_eq!(current.job.active_child_id.as_deref(), Some("child"));
        assert!(!current.job.can_cancel);
        assert!(!checkpoint(&db, &a, None).unwrap());
    }

    #[test]
    fn immutable_manifest_digest_survives_checkpoint_and_reopen() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = stored();
        a.manifest_sha256 = "immutable-manifest".into();
        insert(&db, &a).unwrap();
        a.job.state = VideoJobState::Running;
        assert!(checkpoint(&db, &a, None).unwrap());
        assert_eq!(
            get(&db, "parent").unwrap().unwrap().manifest_sha256,
            "immutable-manifest"
        );
    }

    #[test]
    fn operation_uuid_is_idempotent_only_for_exact_request() {
        let db = MetadataDb::open_in_memory().unwrap();
        let a = stored();
        assert!(insert(&db, &a).unwrap());
        assert!(!insert(&db, &a).unwrap());
        let mut changed = a.clone();
        changed.request_sha256 = "different".into();
        assert!(insert(&db, &changed).is_err());
        assert_eq!(get(&db, "parent").unwrap().unwrap().revision, 0);
    }

    #[test]
    fn child_claim_and_parent_checkpoint_are_one_fenced_transaction() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = stored();
        insert(&db, &a).unwrap();
        let part = StoredVideoJobPart {
            parent_id: "parent".into(),
            index: 0,
            child_id: "child".into(),
            first_clip: 0,
            clip_count: 16,
            keep_frames: 1297,
            output_sha256: None,
        };
        a.job.state = VideoJobState::Running;
        a.job.active_child_id = Some("child".into());
        assert!(checkpoint(&db, &a, Some(&part)).unwrap());
        assert!(!checkpoint(&db, &a, Some(&part)).unwrap());
        let current = get(&db, "parent").unwrap().unwrap();
        assert_eq!(current.revision, 1);
        assert_eq!(
            get_part(&db, "parent", 0).unwrap().unwrap().child_id,
            "child"
        );
        let mut cancel = current.clone();
        cancel.job.state = VideoJobState::Cancelled;
        assert!(checkpoint(&db, &cancel, None).unwrap());
        assert!(!checkpoint(&db, &current, None).unwrap());
        assert_eq!(
            get(&db, "parent").unwrap().unwrap().job.state,
            VideoJobState::Cancelled
        );
    }

    #[test]
    fn invalid_child_claim_rolls_back_parent_and_revision() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = stored();
        insert(&db, &a).unwrap();
        a.job.state = VideoJobState::Running;
        let part = StoredVideoJobPart {
            parent_id: "different".into(),
            index: 0,
            child_id: "child".into(),
            first_clip: 0,
            clip_count: 16,
            keep_frames: 1297,
            output_sha256: None,
        };
        assert!(checkpoint(&db, &a, Some(&part)).is_err());
        let unchanged = get(&db, "parent").unwrap().unwrap();
        assert_eq!(unchanged.job.state, VideoJobState::Queued);
        assert_eq!(unchanged.revision, 0);
        assert!(get_part(&db, "parent", 0).unwrap().is_none());
    }

    #[test]
    fn file_backed_reopen_preserves_failed_inputs_and_exact_child_claim() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("mold.db");
        let db = MetadataDb::open(&path).unwrap();
        let mut a = stored();
        a.manifest_json = "sealed-media-authority".into();
        insert(&db, &a).unwrap();
        let part = StoredVideoJobPart {
            parent_id: "parent".into(),
            index: u64::MAX,
            child_id: "child".into(),
            first_clip: u64::MAX - 1,
            clip_count: 1,
            keep_frames: 97,
            output_sha256: Some("exact-part".into()),
        };
        a.job.state = VideoJobState::Failed;
        a.job.error_detail = Some("original failure".into());
        assert!(checkpoint(&db, &a, Some(&part)).unwrap());
        drop(db);
        let db = MetadataDb::open(&path).unwrap();
        assert_eq!(recover_at_startup(&db, 9).unwrap(), 0);
        let recovered = get(&db, "parent").unwrap().unwrap();
        assert_eq!(recovered.manifest_json, "sealed-media-authority");
        assert_eq!(
            recovered.job.error_detail.as_deref(),
            Some("original failure")
        );
        assert_eq!(get_part(&db, "parent", u64::MAX).unwrap().unwrap(), part);
    }

    #[test]
    fn startup_parks_work_and_preserves_child_authority_and_failure() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut a = stored();
        a.job.state = VideoJobState::Running;
        a.job.active_child_id = Some("child".into());
        insert(&db, &a).unwrap();
        assert_eq!(recover_at_startup(&db, 9).unwrap(), 1);
        let current = get(&db, "parent").unwrap().unwrap();
        assert_eq!(current.job.state, VideoJobState::Interrupted);
        assert_eq!(current.job.active_child_id.as_deref(), Some("child"));
        assert!(current.job.can_resume);
        assert!(!current.job.can_cancel);
        assert_eq!(recover_at_startup(&db, 10).unwrap(), 0);
    }
}
