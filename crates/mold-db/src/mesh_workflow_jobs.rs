//! SQLite projection for durable mesh workflow manifests.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use mold_core::mesh_workflow::{
    MeshWorkflowArtifact, MeshWorkflowJobState, MeshWorkflowStageKind, MeshWorkflowStageState,
};
use rusqlite::{params, OptionalExtension, Row};

use crate::MetadataDb;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshWorkflowJobRow {
    pub id: String,
    pub state: MeshWorkflowJobState,
    pub request_json: String,
    pub work_dir: PathBuf,
    pub stage_count: u32,
    pub current_stage: u32,
    pub output_filename: Option<String>,
    pub error: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshWorkflowStageRow {
    pub job_id: String,
    pub stage_index: u32,
    pub kind: MeshWorkflowStageKind,
    pub state: MeshWorkflowStageState,
    pub artifacts: Vec<MeshWorkflowArtifact>,
    pub error: Option<String>,
    pub updated_at_ms: i64,
}

const JOB_COLUMNS: &str = "id,state,request_json,work_dir,stage_count,current_stage,output_filename,error,created_at_ms,updated_at_ms";

pub fn insert_job(db: &MetadataDb, row: &MeshWorkflowJobRow) -> Result<()> {
    db.with_conn(|connection| {
        connection.execute(
            "INSERT INTO mesh_workflow_jobs
             (id,state,request_json,work_dir,stage_count,current_stage,output_filename,error,created_at_ms,updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                row.id,
                row.state.as_str(),
                row.request_json,
                row.work_dir.to_string_lossy(),
                row.stage_count,
                row.current_stage,
                row.output_filename,
                row.error,
                row.created_at_ms,
                row.updated_at_ms,
            ],
        )?;
        Ok(())
    })
}

pub fn get_job(db: &MetadataDb, id: &str) -> Result<Option<MeshWorkflowJobRow>> {
    db.with_conn(|connection| {
        connection
            .query_row(
                &format!("SELECT {JOB_COLUMNS} FROM mesh_workflow_jobs WHERE id=?1"),
                [id],
                job_from_row,
            )
            .optional()
            .map_err(Into::into)
    })
}

pub fn list_jobs(db: &MetadataDb) -> Result<Vec<MeshWorkflowJobRow>> {
    db.with_conn(|connection| {
        let mut statement = connection.prepare(&format!(
            "SELECT {JOB_COLUMNS} FROM mesh_workflow_jobs ORDER BY created_at_ms DESC,id DESC"
        ))?;
        let rows = statement
            .query_map([], job_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into);
        rows
    })
}

pub fn next_queued_job(db: &MetadataDb) -> Result<Option<MeshWorkflowJobRow>> {
    db.with_conn(|connection| {
        connection
            .query_row(
                &format!(
                    "SELECT {JOB_COLUMNS} FROM mesh_workflow_jobs
                     WHERE state='queued' ORDER BY created_at_ms ASC,id ASC LIMIT 1"
                ),
                [],
                job_from_row,
            )
            .optional()
            .map_err(Into::into)
    })
}

pub fn claim_job(db: &MetadataDb, id: &str, now_ms: i64) -> Result<bool> {
    transition(
        db,
        id,
        &[MeshWorkflowJobState::Queued],
        MeshWorkflowJobState::Running,
        None,
        now_ms,
    )
}

pub fn transition(
    db: &MetadataDb,
    id: &str,
    expected: &[MeshWorkflowJobState],
    next: MeshWorkflowJobState,
    error: Option<&str>,
    now_ms: i64,
) -> Result<bool> {
    if expected.is_empty() {
        return Ok(false);
    }
    db.with_conn(|connection| {
        let current = connection
            .query_row(
                "SELECT state FROM mesh_workflow_jobs WHERE id=?1",
                [id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(current) = current else {
            return Ok(false);
        };
        let current_state = MeshWorkflowJobState::from_str(&current)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if current_state == MeshWorkflowJobState::Completed
            || !expected.contains(&current_state)
        {
            return Ok(false);
        }
        Ok(connection.execute(
            "UPDATE mesh_workflow_jobs SET state=?2,error=?3,updated_at_ms=?4 WHERE id=?1 AND state=?5",
            params![id, next.as_str(), error, now_ms, current],
        )? == 1)
    })
}

pub fn set_current_stage(db: &MetadataDb, id: &str, stage_index: u32, now_ms: i64) -> Result<bool> {
    db.with_conn(|connection| {
        Ok(connection.execute(
            "UPDATE mesh_workflow_jobs SET current_stage=?2,updated_at_ms=?3
             WHERE id=?1 AND state='running' AND ?2 < stage_count",
            params![id, stage_index, now_ms],
        )? == 1)
    })
}

pub fn complete_job(db: &MetadataDb, id: &str, output_filename: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|connection| {
        Ok(connection.execute(
            "UPDATE mesh_workflow_jobs
             SET state='completed',current_stage=stage_count-1,output_filename=?2,error=NULL,updated_at_ms=?3
             WHERE id=?1 AND state='running'",
            params![id, output_filename, now_ms],
        )? == 1)
    })
}

pub fn pause_unfinished_for_recovery(db: &MetadataDb, now_ms: i64) -> Result<usize> {
    db.with_conn(|connection| {
        connection
            .execute(
                "UPDATE mesh_workflow_jobs SET state='paused',updated_at_ms=?1
                 WHERE state IN ('queued','running')",
                [now_ms],
            )
            .map_err(Into::into)
    })
}

pub fn upsert_stage(db: &MetadataDb, row: &MeshWorkflowStageRow) -> Result<()> {
    let artifacts = serde_json::to_string(&row.artifacts)
        .context("serializing mesh workflow stage artifacts")?;
    db.with_conn(|connection| {
        connection.execute(
            "INSERT INTO mesh_workflow_stages
             (job_id,stage_index,kind,state,artifacts_json,error,updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(job_id,stage_index) DO UPDATE SET
               kind=excluded.kind,state=excluded.state,artifacts_json=excluded.artifacts_json,
               error=excluded.error,updated_at_ms=excluded.updated_at_ms",
            params![
                row.job_id,
                row.stage_index,
                row.kind.as_str(),
                row.state.as_str(),
                artifacts,
                row.error,
                row.updated_at_ms,
            ],
        )?;
        Ok(())
    })
}

pub fn stages_for_job(db: &MetadataDb, id: &str) -> Result<Vec<MeshWorkflowStageRow>> {
    db.with_conn(|connection| {
        let mut statement = connection.prepare(
            "SELECT job_id,stage_index,kind,state,artifacts_json,error,updated_at_ms
             FROM mesh_workflow_stages WHERE job_id=?1 ORDER BY stage_index ASC",
        )?;
        let rows = statement
            .query_map([id], stage_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into);
        rows
    })
}

fn job_from_row(row: &Row<'_>) -> rusqlite::Result<MeshWorkflowJobRow> {
    Ok(MeshWorkflowJobRow {
        id: row.get(0)?,
        state: parse_text(row, 1, "job state")?,
        request_json: row.get(2)?,
        work_dir: PathBuf::from(row.get::<_, String>(3)?),
        stage_count: row.get(4)?,
        current_stage: row.get(5)?,
        output_filename: row.get(6)?,
        error: row.get(7)?,
        created_at_ms: row.get(8)?,
        updated_at_ms: row.get(9)?,
    })
}

fn stage_from_row(row: &Row<'_>) -> rusqlite::Result<MeshWorkflowStageRow> {
    let artifacts_json = row.get::<_, String>(4)?;
    let artifacts = serde_json::from_str(&artifacts_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(error))
    })?;
    Ok(MeshWorkflowStageRow {
        job_id: row.get(0)?,
        stage_index: row.get(1)?,
        kind: parse_text(row, 2, "stage kind")?,
        state: parse_text(row, 3, "stage state")?,
        artifacts,
        error: row.get(5)?,
        updated_at_ms: row.get(6)?,
    })
}

fn parse_text<T>(row: &Row<'_>, index: usize, label: &str) -> rusqlite::Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display + Send + Sync + 'static,
{
    let value = row.get::<_, String>(index)?;
    value.parse().map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::other(format!(
                "invalid mesh workflow {label}: {error}"
            ))),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, state: MeshWorkflowJobState, created_at_ms: i64) -> MeshWorkflowJobRow {
        MeshWorkflowJobRow {
            id: id.into(),
            state,
            request_json: r#"{"mode":"text_to_mesh"}"#.into(),
            work_dir: PathBuf::from(format!("/tmp/mold/mesh-workflows/{id}")),
            stage_count: 3,
            current_stage: 0,
            output_filename: None,
            error: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
        }
    }

    #[test]
    fn fifo_claim_stage_checkpoint_and_completion_round_trip() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_job(&db, &job("later", MeshWorkflowJobState::Queued, 2)).unwrap();
        insert_job(&db, &job("first", MeshWorkflowJobState::Queued, 1)).unwrap();
        assert_eq!(next_queued_job(&db).unwrap().unwrap().id, "first");
        assert!(claim_job(&db, "first", 3).unwrap());
        assert!(!claim_job(&db, "first", 4).unwrap());

        let stage = MeshWorkflowStageRow {
            job_id: "first".into(),
            stage_index: 0,
            kind: MeshWorkflowStageKind::Image,
            state: MeshWorkflowStageState::Completed,
            artifacts: vec![MeshWorkflowArtifact {
                role: "generated_image".into(),
                relative_path: "stages/000/generated.png".into(),
                media_type: "image/png".into(),
                sha256: "a".repeat(64),
                byte_length: 10,
            }],
            error: None,
            updated_at_ms: 5,
        };
        upsert_stage(&db, &stage).unwrap();
        assert_eq!(stages_for_job(&db, "first").unwrap(), vec![stage]);
        assert!(set_current_stage(&db, "first", 1, 6).unwrap());
        assert!(complete_job(&db, "first", "fox.glb", 7).unwrap());
        let completed = get_job(&db, "first").unwrap().unwrap();
        assert_eq!(completed.state, MeshWorkflowJobState::Completed);
        assert_eq!(completed.output_filename.as_deref(), Some("fox.glb"));
        assert!(!transition(
            &db,
            "first",
            &[MeshWorkflowJobState::Completed],
            MeshWorkflowJobState::Queued,
            None,
            8,
        )
        .unwrap());
    }

    #[test]
    fn recovery_parks_unfinished_jobs_without_losing_stage_artifacts() {
        let db = MetadataDb::open_in_memory().unwrap();
        insert_job(&db, &job("queued", MeshWorkflowJobState::Queued, 1)).unwrap();
        insert_job(&db, &job("running", MeshWorkflowJobState::Running, 2)).unwrap();
        insert_job(&db, &job("done", MeshWorkflowJobState::Completed, 3)).unwrap();
        assert_eq!(pause_unfinished_for_recovery(&db, 4).unwrap(), 2);
        assert_eq!(
            get_job(&db, "running").unwrap().unwrap().state,
            MeshWorkflowJobState::Paused
        );
        assert_eq!(
            get_job(&db, "done").unwrap().unwrap().state,
            MeshWorkflowJobState::Completed
        );
    }
}
