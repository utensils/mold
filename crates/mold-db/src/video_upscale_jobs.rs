//! SQLite persistence for durable framewise video-upscale jobs.

use crate::MetadataDb;
use anyhow::{bail, Context, Result};
use mold_core::{
    VideoUpscaleJob, VideoUpscaleJobState, VideoUpscaleMediaFacts, VideoUpscaleSource,
    VIDEO_UPSCALE_CONTRACT_VERSION, VIDEO_UPSCALE_DISCLOSURE,
};
use rusqlite::{params, OptionalExtension, Row};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct StoredVideoUpscaleJob {
    pub job: VideoUpscaleJob,
    pub source_path: PathBuf,
    pub work_dir: PathBuf,
}

fn state_string(state: VideoUpscaleJobState) -> &'static str {
    match state {
        VideoUpscaleJobState::Queued => "queued",
        VideoUpscaleJobState::Running => "running",
        VideoUpscaleJobState::Finalizing => "finalizing",
        VideoUpscaleJobState::Paused => "paused",
        VideoUpscaleJobState::Completed => "completed",
        VideoUpscaleJobState::Failed => "failed",
        VideoUpscaleJobState::Cancelled => "cancelled",
    }
}

fn parse_state(value: &str) -> Result<VideoUpscaleJobState> {
    Ok(match value {
        "queued" => VideoUpscaleJobState::Queued,
        "running" => VideoUpscaleJobState::Running,
        "finalizing" => VideoUpscaleJobState::Finalizing,
        "paused" => VideoUpscaleJobState::Paused,
        "completed" => VideoUpscaleJobState::Completed,
        "failed" => VideoUpscaleJobState::Failed,
        "cancelled" => VideoUpscaleJobState::Cancelled,
        other => bail!("invalid video upscale state {other:?}"),
    })
}

fn decode_optional<T: serde::de::DeserializeOwned>(value: Option<String>) -> Result<Option<T>> {
    value
        .map(|json| serde_json::from_str(&json).context("decoding video upscale job JSON"))
        .transpose()
}

fn row(row: &Row<'_>) -> Result<StoredVideoUpscaleJob, rusqlite::Error> {
    let conversion = |error: anyhow::Error| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::other(error.to_string())),
        )
    };
    let state = parse_state(&row.get::<_, String>(1)?).map_err(conversion)?;
    let source = serde_json::from_str::<VideoUpscaleSource>(&row.get::<_, String>(2)?)
        .context("decoding video upscale source")
        .map_err(conversion)?;
    let source_facts =
        decode_optional::<VideoUpscaleMediaFacts>(row.get(9)?).map_err(conversion)?;
    let output_facts =
        decode_optional::<VideoUpscaleMediaFacts>(row.get(10)?).map_err(conversion)?;
    Ok(StoredVideoUpscaleJob {
        job: VideoUpscaleJob {
            contract_version: VIDEO_UPSCALE_CONTRACT_VERSION,
            id: row.get(0)?,
            state,
            source,
            model: row.get(4)?,
            scale_factor: row.get(5)?,
            tile_size: row.get(6)?,
            completed_frames: row.get(7)?,
            total_frames: row.get(8)?,
            source_facts,
            output_facts,
            output_filename: row.get(11)?,
            error: row.get(12)?,
            created_at_ms: row.get(14)?,
            updated_at_ms: row.get(15)?,
            disclosure: VIDEO_UPSCALE_DISCLOSURE.into(),
        },
        source_path: PathBuf::from(row.get::<_, String>(3)?),
        work_dir: PathBuf::from(row.get::<_, String>(13)?),
    })
}

const COLUMNS: &str = "id,state,source_json,source_path,model,scale_factor,tile_size,completed_frames,total_frames,source_facts_json,output_facts_json,output_filename,error,work_dir,created_at_ms,updated_at_ms";

pub fn insert(db: &MetadataDb, stored: &StoredVideoUpscaleJob) -> Result<()> {
    db.with_conn(|conn| {
        conn.execute("INSERT INTO video_upscale_jobs (id,state,source_json,source_path,model,scale_factor,tile_size,completed_frames,total_frames,source_facts_json,output_facts_json,output_filename,error,work_dir,created_at_ms,updated_at_ms) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)", params![
            stored.job.id, state_string(stored.job.state), serde_json::to_string(&stored.job.source)?,
            stored.source_path.to_string_lossy(), stored.job.model, stored.job.scale_factor,
            stored.job.tile_size, stored.job.completed_frames, stored.job.total_frames,
            stored.job.source_facts.as_ref().map(serde_json::to_string).transpose()?,
            stored.job.output_facts.as_ref().map(serde_json::to_string).transpose()?,
            stored.job.output_filename, stored.job.error, stored.work_dir.to_string_lossy(),
            stored.job.created_at_ms, stored.job.updated_at_ms])?;
        Ok(())
    })
}

pub fn get(db: &MetadataDb, id: &str) -> Result<Option<StoredVideoUpscaleJob>> {
    db.with_conn(|conn| {
        conn.query_row(
            &format!("SELECT {COLUMNS} FROM video_upscale_jobs WHERE id=?1"),
            [id],
            row,
        )
        .optional()
        .map_err(Into::into)
    })
}

pub fn list(db: &MetadataDb) -> Result<Vec<VideoUpscaleJob>> {
    db.with_conn(|conn| {
        let mut statement = conn.prepare(&format!(
            "SELECT {COLUMNS} FROM video_upscale_jobs ORDER BY created_at_ms DESC"
        ))?;
        let stored = statement
            .query_map([], row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(stored.into_iter().map(|stored| stored.job).collect())
    })
}

pub fn list_queued_ids(db: &MetadataDb) -> Result<Vec<String>> {
    db.with_conn(|conn| {
        let mut statement = conn.prepare(
            "SELECT id FROM video_upscale_jobs WHERE state='queued' ORDER BY created_at_ms ASC",
        )?;
        let rows = statement.query_map([], |row| row.get(0))?;
        let ids = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    })
}

pub fn finish_preparation(
    db: &MetadataDb,
    id: &str,
    source_path: &std::path::Path,
    facts: &VideoUpscaleMediaFacts,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE video_upscale_jobs SET source_path=?2,source_facts_json=?3,total_frames=?4,updated_at_ms=?5 WHERE id=?1 AND state='running'",
            params![id, source_path.to_string_lossy(), serde_json::to_string(facts)?, facts.frame_count, now_ms],
        )? == 1)
    })
}

pub fn transition(
    db: &MetadataDb,
    id: &str,
    expected: &[VideoUpscaleJobState],
    next: VideoUpscaleJobState,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| {
        let current: Option<String> = conn.query_row("SELECT state FROM video_upscale_jobs WHERE id=?1", [id], |r| r.get(0)).optional()?;
        let Some(current) = current else { return Ok(false) };
        if parse_state(&current)?.is_terminal() { return Ok(false); }
        if !expected.iter().any(|state| state_string(*state) == current) { return Ok(false); }
        Ok(conn.execute("UPDATE video_upscale_jobs SET state=?2,error=NULL,updated_at_ms=?3 WHERE id=?1 AND state=?4", params![id, state_string(next), now_ms, current])? == 1)
    })
}

pub fn update_probe(
    db: &MetadataDb,
    id: &str,
    facts: &VideoUpscaleMediaFacts,
    now_ms: i64,
) -> Result<()> {
    db.with_conn(|conn| { conn.execute("UPDATE video_upscale_jobs SET source_facts_json=?2,total_frames=?3,updated_at_ms=?4 WHERE id=?1", params![id, serde_json::to_string(facts)?, facts.frame_count, now_ms])?; Ok(()) })
}

pub fn update_progress(db: &MetadataDb, id: &str, frames: u64, now_ms: i64) -> Result<()> {
    db.with_conn(|conn| { conn.execute("UPDATE video_upscale_jobs SET completed_frames=?2,updated_at_ms=?3 WHERE id=?1 AND state='running'", params![id, frames, now_ms])?; Ok(()) })
}

pub fn complete(
    db: &MetadataDb,
    id: &str,
    filename: &str,
    facts: &VideoUpscaleMediaFacts,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|conn| Ok(conn.execute("UPDATE video_upscale_jobs SET state='completed',completed_frames=total_frames,output_filename=?2,output_facts_json=?3,error=NULL,updated_at_ms=?4 WHERE id=?1 AND state='finalizing'", params![id, filename, serde_json::to_string(facts)?, now_ms])? == 1))
}

pub fn fail(db: &MetadataDb, id: &str, error: &str, now_ms: i64) -> Result<()> {
    db.with_conn(|conn| {
        conn.execute(
            "UPDATE video_upscale_jobs SET state='failed',error=?2,updated_at_ms=?3 WHERE id=?1 AND state IN ('queued','running','finalizing')",
            params![id, error, now_ms],
        )?;
        Ok(())
    })
}

/// Publication may already be externally visible when a later bookkeeping
/// write fails. Keep that deterministic finalization retryable instead of
/// reporting a terminal failure beside a committed Gallery output.
pub fn pause_after_error(db: &MetadataDb, id: &str, error: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|conn| {
        Ok(conn.execute(
            "UPDATE video_upscale_jobs SET state='paused',error=?2,updated_at_ms=?3 WHERE id=?1 AND state='finalizing'",
            params![id, error, now_ms],
        )? == 1)
    })
}

/// Recovery is deliberately non-dispatching; unfinished work requires Resume.
pub fn pause_unfinished_for_recovery(db: &MetadataDb, now_ms: i64) -> Result<usize> {
    db.with_conn(|conn| Ok(conn.execute("UPDATE video_upscale_jobs SET state='paused',updated_at_ms=?1 WHERE state IN ('queued','running','finalizing')", [now_ms])?))
}

/// Cooperative server shutdown parks work that can still stop at a frame
/// boundary. A job that already claimed finalization must drain while the
/// server awaits it, keeping Gallery publication and the completed row atomic
/// from the next runtime's perspective. Crash recovery still pauses that state
/// through [`pause_unfinished_for_recovery`].
pub fn pause_interruptible_for_shutdown(db: &MetadataDb, now_ms: i64) -> Result<usize> {
    db.with_conn(|conn| Ok(conn.execute("UPDATE video_upscale_jobs SET state='paused',updated_at_ms=?1 WHERE state IN ('queued','running')", [now_ms])?))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn stored() -> StoredVideoUpscaleJob {
        StoredVideoUpscaleJob {
            job: VideoUpscaleJob {
                contract_version: 1,
                id: "vup-1".into(),
                state: VideoUpscaleJobState::Running,
                source: VideoUpscaleSource::Library {
                    filename: "a.mp4".into(),
                },
                model: "real-esrgan-x4plus:fp16".into(),
                scale_factor: 4,
                tile_size: None,
                completed_frames: 3,
                total_frames: 10,
                source_facts: None,
                output_facts: None,
                output_filename: None,
                error: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                disclosure: VIDEO_UPSCALE_DISCLOSURE.into(),
            },
            source_path: "/gallery/a.mp4".into(),
            work_dir: "/work/vup-1".into(),
        }
    }

    #[test]
    fn recovery_pauses_without_losing_checkpoint() {
        let db = MetadataDb::open_in_memory().unwrap();
        let stored = stored();
        insert(&db, &stored).unwrap();
        assert_eq!(pause_unfinished_for_recovery(&db, 2).unwrap(), 1);
        let recovered = get(&db, "vup-1").unwrap().unwrap();
        assert_eq!(recovered.job.state, VideoUpscaleJobState::Paused);
        assert_eq!(recovered.job.completed_frames, 3);
    }

    #[test]
    fn cooperative_shutdown_leaves_claimed_finalization_to_drain() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut row = stored();
        row.job.state = VideoUpscaleJobState::Finalizing;
        insert(&db, &row).unwrap();
        assert_eq!(pause_interruptible_for_shutdown(&db, 2).unwrap(), 0);
        assert_eq!(
            get(&db, "vup-1").unwrap().unwrap().job.state,
            VideoUpscaleJobState::Finalizing
        );
    }

    #[test]
    fn cancellation_is_terminal_and_keeps_source_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut row = stored();
        row.job.state = VideoUpscaleJobState::Queued;
        insert(&db, &row).unwrap();
        assert!(transition(
            &db,
            "vup-1",
            &[VideoUpscaleJobState::Queued],
            VideoUpscaleJobState::Cancelled,
            2,
        )
        .unwrap());
        let cancelled = get(&db, "vup-1").unwrap().unwrap();
        assert_eq!(cancelled.job.state, VideoUpscaleJobState::Cancelled);
        assert_eq!(cancelled.source_path, PathBuf::from("/gallery/a.mp4"));
        assert!(!transition(
            &db,
            "vup-1",
            &[VideoUpscaleJobState::Cancelled],
            VideoUpscaleJobState::Queued,
            3,
        )
        .unwrap());
    }

    #[test]
    fn finalization_excludes_cancellation_until_completion() {
        let db = MetadataDb::open_in_memory().unwrap();
        let row = stored();
        insert(&db, &row).unwrap();
        assert!(transition(
            &db,
            "vup-1",
            &[VideoUpscaleJobState::Running],
            VideoUpscaleJobState::Finalizing,
            2,
        )
        .unwrap());
        assert!(!transition(
            &db,
            "vup-1",
            &[VideoUpscaleJobState::Running],
            VideoUpscaleJobState::Cancelled,
            3,
        )
        .unwrap());
        let facts = VideoUpscaleMediaFacts {
            container: "mov,mp4,m4a,3gp,3g2,mj2".into(),
            video_codec: "h264".into(),
            width: 32,
            height: 32,
            frame_count: 10,
            fps: "5/1".into(),
            duration_ms: 2_000,
            primary_audio_codec: None,
            primary_audio_sample_rate: None,
            primary_audio_channels: None,
        };
        assert!(complete(&db, "vup-1", "result.mp4", &facts, 4).unwrap());
        assert_eq!(
            get(&db, "vup-1").unwrap().unwrap().job.state,
            VideoUpscaleJobState::Completed
        );
    }

    #[test]
    fn finalization_errors_pause_for_reconciliation() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut row = stored();
        row.job.state = VideoUpscaleJobState::Finalizing;
        insert(&db, &row).unwrap();
        assert!(pause_after_error(&db, "vup-1", "late bookkeeping failure", 2).unwrap());
        let paused = get(&db, "vup-1").unwrap().unwrap();
        assert_eq!(paused.job.state, VideoUpscaleJobState::Paused);
        assert_eq!(
            paused.job.error.as_deref(),
            Some("late bookkeeping failure")
        );
    }

    #[test]
    fn queued_query_and_preparation_update_only_active_authority() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut row = stored();
        row.job.state = VideoUpscaleJobState::Queued;
        row.job.source_facts = None;
        row.job.total_frames = 0;
        insert(&db, &row).unwrap();
        assert_eq!(list_queued_ids(&db).unwrap(), vec!["vup-1"]);
        assert!(transition(
            &db,
            "vup-1",
            &[VideoUpscaleJobState::Queued],
            VideoUpscaleJobState::Running,
            2,
        )
        .unwrap());
        assert!(list_queued_ids(&db).unwrap().is_empty());
        let facts = VideoUpscaleMediaFacts {
            container: "mov,mp4,m4a,3gp,3g2,mj2".into(),
            video_codec: "h264".into(),
            width: 32,
            height: 24,
            frame_count: 4,
            fps: "4/1".into(),
            duration_ms: 1_000,
            primary_audio_codec: None,
            primary_audio_sample_rate: None,
            primary_audio_channels: None,
        };
        assert!(finish_preparation(
            &db,
            "vup-1",
            std::path::Path::new("/work/source.mp4"),
            &facts,
            3
        )
        .unwrap());
        let prepared = get(&db, "vup-1").unwrap().unwrap();
        assert_eq!(prepared.source_path, PathBuf::from("/work/source.mp4"));
        assert_eq!(prepared.job.total_frames, 4);
        assert_eq!(prepared.job.source_facts, Some(facts));
    }
}
