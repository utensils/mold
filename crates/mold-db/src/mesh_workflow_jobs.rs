//! SQLite projection for durable mesh workflow manifests.

use std::path::PathBuf;
use std::str::FromStr;

pub const DELETION_CLAIM_ERROR: &str = "__mold_mesh_workflow_deleting_v1__";

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
    pub execution_batch_id: Option<String>,
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

/// Insert the parent and its complete stage graph in one transaction. A
/// durable workflow is never visible without every stage it will execute.
pub fn insert_job_with_stages(
    db: &MetadataDb,
    job: &MeshWorkflowJobRow,
    stages: &[MeshWorkflowStageRow],
) -> Result<()> {
    if stages.len() != job.stage_count as usize
        || stages
            .iter()
            .enumerate()
            .any(|(index, stage)| stage.job_id != job.id || stage.stage_index != index as u32)
    {
        anyhow::bail!("mesh workflow stage graph does not match its parent");
    }
    db.with_conn(|connection| {
        let transaction = connection.unchecked_transaction()?;
        transaction.execute(
            "INSERT INTO mesh_workflow_jobs
             (id,state,request_json,work_dir,stage_count,current_stage,output_filename,error,created_at_ms,updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                job.id,
                job.state.as_str(),
                job.request_json,
                job.work_dir.to_string_lossy(),
                job.stage_count,
                job.current_stage,
                job.output_filename,
                job.error,
                job.created_at_ms,
                job.updated_at_ms,
            ],
        )?;
        for stage in stages {
            let artifacts = serde_json::to_string(&stage.artifacts)
                .context("serializing mesh workflow stage artifacts")?;
            transaction.execute(
                "INSERT INTO mesh_workflow_stages
                 (job_id,stage_index,kind,state,execution_batch_id,artifacts_json,error,updated_at_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![
                    stage.job_id,
                    stage.stage_index,
                    stage.kind.as_str(),
                    stage.state.as_str(),
                    stage.execution_batch_id,
                    artifacts,
                    stage.error,
                    stage.updated_at_ms,
                ],
            )?;
        }
        transaction.commit()?;
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

pub fn attach_stage_execution(
    db: &MetadataDb,
    id: &str,
    stage_index: u32,
    batch_id: &str,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|connection| {
        Ok(connection.execute(
            "UPDATE mesh_workflow_stages
             SET state='running',execution_batch_id=?3,error=NULL,updated_at_ms=?4
             WHERE job_id=?1 AND stage_index=?2 AND state='pending' AND execution_batch_id IS NULL",
            params![id, stage_index, batch_id, now_ms],
        )? == 1)
    })
}

/// Bind every logical stage driven by one durable child in one transaction.
/// A restart must observe either the whole execution group or none of it.
pub fn attach_stage_executions(
    db: &MetadataDb,
    id: &str,
    stage_indices: &[u32],
    batch_id: &str,
    now_ms: i64,
) -> Result<bool> {
    if stage_indices.is_empty() {
        return Ok(false);
    }
    db.with_conn(|connection| {
        let transaction = connection.unchecked_transaction()?;
        for stage_index in stage_indices {
            if transaction.execute(
                "UPDATE mesh_workflow_stages
                 SET state='running',execution_batch_id=?3,error=NULL,updated_at_ms=?4
                 WHERE job_id=?1 AND stage_index=?2 AND state='pending' AND execution_batch_id IS NULL
                   AND EXISTS (
                     SELECT 1 FROM mesh_workflow_jobs
                      WHERE id=?1 AND state='running'
                   )",
                params![id, stage_index, batch_id, now_ms],
            )? != 1
            {
                return Ok(false);
            }
        }
        transaction.commit()?;
        Ok(true)
    })
}

pub fn complete_stage(
    db: &MetadataDb,
    id: &str,
    stage_index: u32,
    artifacts: &[MeshWorkflowArtifact],
    now_ms: i64,
) -> Result<bool> {
    let artifacts =
        serde_json::to_string(artifacts).context("serializing mesh workflow stage artifacts")?;
    db.with_conn(|connection| {
        Ok(connection.execute(
            "UPDATE mesh_workflow_stages
             SET state='completed',artifacts_json=?3,error=NULL,updated_at_ms=?4
             WHERE job_id=?1 AND stage_index=?2 AND state='running'",
            params![id, stage_index, artifacts, now_ms],
        )? == 1)
    })
}

/// Complete the complete logical stage group owned by one child batch.
pub fn complete_stage_execution(
    db: &MetadataDb,
    id: &str,
    batch_id: &str,
    artifacts: &[MeshWorkflowArtifact],
    now_ms: i64,
) -> Result<bool> {
    let artifacts =
        serde_json::to_string(artifacts).context("serializing mesh workflow stage artifacts")?;
    db.with_conn(|connection| {
        let expected = connection.query_row(
            "SELECT COUNT(*) FROM mesh_workflow_stages
             WHERE job_id=?1 AND execution_batch_id=?2",
            params![id, batch_id],
            |row| row.get::<_, usize>(0),
        )?;
        if expected == 0 {
            return Ok(false);
        }
        // A stage the runner already reported complete from the child's
        // live progress (`complete_stage_live`) is settled here too: the
        // batch's artifact is the whole group's, and the count has to cover
        // every row the batch owns or the claim reads as lost.
        Ok(connection.execute(
            "UPDATE mesh_workflow_stages
             SET state='completed',artifacts_json=?3,error=NULL,updated_at_ms=?4
             WHERE job_id=?1 AND execution_batch_id=?2 AND state IN ('running','completed')",
            params![id, batch_id, artifacts, now_ms],
        )? == expected)
    })
}

/// Report one stage of a shared child batch complete from the child's LIVE
/// progress, before the batch itself settles.
///
/// Shape and paint run inside one queue job, so the workflow only ever learns
/// that the job started and finished — and both stages read RUNNING for the
/// whole render. The runner watches the child's progress stage name and calls
/// this the moment paint begins, so "Build geometry" completes when the
/// geometry is built. No artifact: the batch's GLB lands on the whole group in
/// `complete_stage_execution`, and `resume_job` resets the row with its batch
/// if that batch later fails.
pub fn complete_stage_live(
    db: &MetadataDb,
    id: &str,
    stage_index: u32,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|connection| {
        Ok(connection.execute(
            "UPDATE mesh_workflow_stages SET state='completed',updated_at_ms=?3
             WHERE job_id=?1 AND stage_index=?2 AND state='running'
               AND execution_batch_id IS NOT NULL",
            params![id, stage_index, now_ms],
        )? == 1)
    })
}

pub fn fail_stage_and_job(
    db: &MetadataDb,
    id: &str,
    stage_index: u32,
    error: &str,
    now_ms: i64,
) -> Result<bool> {
    db.with_conn(|connection| {
        let transaction = connection.unchecked_transaction()?;
        // Every running stage sharing the failed stage's child batch goes
        // with it — and the failed stage ITSELF, batch or no batch. A child
        // refused at durable admission was never attached, so the stage has
        // no batch id and is still `pending`; matching only on the batch
        // (`= NULL`) flipped nothing, and Resume then re-queued the same
        // refusal with every stage reading PENDING (#1672).
        let stage_changed = transaction.execute(
            "UPDATE mesh_workflow_stages SET state='failed',error=?3,updated_at_ms=?4
             WHERE job_id=?1 AND (
               (stage_index=?2 AND state IN ('pending','running'))
               OR (state='running' AND execution_batch_id IS NOT NULL
                   AND execution_batch_id = (
                       SELECT execution_batch_id FROM mesh_workflow_stages
                        WHERE job_id=?1 AND stage_index=?2
                   ))
             )",
            params![id, stage_index, error, now_ms],
        )? > 0;
        if !stage_changed {
            return Ok(false);
        }
        let job_changed = transaction.execute(
            "UPDATE mesh_workflow_jobs SET state='failed',error=?2,updated_at_ms=?3
             WHERE id=?1 AND state='running'",
            params![id, error, now_ms],
        )? == 1;
        if !job_changed {
            return Ok(false);
        }
        transaction.commit()?;
        Ok(true)
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

pub fn resume_job(db: &MetadataDb, id: &str, now_ms: i64) -> Result<bool> {
    db.with_conn(|connection| {
        let transaction = connection.unchecked_transaction()?;
        let current_stage = transaction
            .query_row(
                "SELECT current_stage FROM mesh_workflow_jobs
                 WHERE id=?1 AND state IN ('paused','failed')",
                [id],
                |row| row.get::<_, u32>(0),
            )
            .optional()?;
        let Some(current_stage) = current_stage else {
            return Ok(false);
        };
        // The child batch is the unit of retry: a stage that shares a failed
        // stage's batch — a shape stage `complete_stage_live` reported done
        // while paint then failed inside the same job — goes back with it,
        // or the retry would wait on the dead batch forever.
        transaction.execute(
            "UPDATE mesh_workflow_stages
             SET state='pending',execution_batch_id=NULL,error=NULL,
                 updated_at_ms=MAX(updated_at_ms + 1, ?3)
             WHERE job_id=?1 AND (
               (stage_index>=?2 AND state='failed')
               OR (execution_batch_id IS NOT NULL AND execution_batch_id IN (
                     SELECT execution_batch_id FROM mesh_workflow_stages
                      WHERE job_id=?1 AND stage_index>=?2 AND state='failed'
                        AND execution_batch_id IS NOT NULL))
             )",
            params![id, current_stage, now_ms],
        )?;
        let changed = transaction.execute(
            "UPDATE mesh_workflow_jobs SET state='queued',error=NULL,updated_at_ms=?2
             WHERE id=?1 AND state IN ('paused','failed')",
            params![id, now_ms],
        )? == 1;
        transaction.commit()?;
        Ok(changed)
    })
}

pub fn cancel_job(db: &MetadataDb, id: &str, now_ms: i64) -> Result<bool> {
    transition(
        db,
        id,
        &[
            MeshWorkflowJobState::Queued,
            MeshWorkflowJobState::Running,
            MeshWorkflowJobState::Paused,
            MeshWorkflowJobState::Failed,
        ],
        MeshWorkflowJobState::Cancelled,
        None,
        now_ms,
    )
}

/// Atomically fences resume/cancel by turning any settled workflow into a
/// cancelled deletion claim. Repeating the claim is idempotent so cleanup can
/// resume after a process or filesystem failure.
pub fn claim_settled_deletion(
    db: &MetadataDb,
    id: &str,
    now_ms: i64,
) -> Result<Option<MeshWorkflowJobRow>> {
    db.with_conn(|connection| {
        let transaction = connection.unchecked_transaction()?;
        let state = transaction
            .query_row(
                "SELECT state FROM mesh_workflow_jobs WHERE id=?1",
                [id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(state) = state else {
            return Ok(None);
        };
        let state =
            MeshWorkflowJobState::from_str(&state).map_err(|error| anyhow::anyhow!(error))?;
        if !state.is_settled() {
            return Ok(None);
        }
        transaction.execute(
            "UPDATE mesh_workflow_jobs
             SET state='cancelled',error=?2,updated_at_ms=MAX(updated_at_ms,?3)
             WHERE id=?1 AND state IN ('completed','failed','cancelled')",
            params![id, DELETION_CLAIM_ERROR, now_ms],
        )?;
        let row = transaction.query_row(
            &format!("SELECT {JOB_COLUMNS} FROM mesh_workflow_jobs WHERE id=?1"),
            [id],
            job_from_row,
        )?;
        transaction.commit()?;
        Ok(Some(row))
    })
}

pub fn delete_claimed_job(db: &MetadataDb, id: &str) -> Result<bool> {
    db.with_conn(|connection| {
        Ok(connection.execute(
            "DELETE FROM mesh_workflow_jobs
             WHERE id=?1 AND state='cancelled' AND error=?2",
            params![id, DELETION_CLAIM_ERROR],
        )? == 1)
    })
}

pub fn upsert_stage(db: &MetadataDb, row: &MeshWorkflowStageRow) -> Result<()> {
    let artifacts = serde_json::to_string(&row.artifacts)
        .context("serializing mesh workflow stage artifacts")?;
    db.with_conn(|connection| {
        connection.execute(
            "INSERT INTO mesh_workflow_stages
             (job_id,stage_index,kind,state,execution_batch_id,artifacts_json,error,updated_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(job_id,stage_index) DO UPDATE SET
               kind=excluded.kind,state=excluded.state,execution_batch_id=excluded.execution_batch_id,artifacts_json=excluded.artifacts_json,
               error=excluded.error,updated_at_ms=excluded.updated_at_ms",
            params![
                row.job_id,
                row.stage_index,
                row.kind.as_str(),
                row.state.as_str(),
                row.execution_batch_id,
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
            "SELECT job_id,stage_index,kind,state,execution_batch_id,artifacts_json,error,updated_at_ms
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
    let artifacts_json = row.get::<_, String>(5)?;
    let artifacts = serde_json::from_str(&artifacts_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(error))
    })?;
    Ok(MeshWorkflowStageRow {
        job_id: row.get(0)?,
        stage_index: row.get(1)?,
        kind: parse_text(row, 2, "stage kind")?,
        state: parse_text(row, 3, "stage state")?,
        execution_batch_id: row.get(4)?,
        artifacts,
        error: row.get(6)?,
        updated_at_ms: row.get(7)?,
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
            execution_batch_id: Some("batch-image".into()),
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

    #[test]
    fn one_failed_execution_marks_every_stage_sharing_the_child_batch() {
        let db = MetadataDb::open_in_memory().unwrap();
        let job = job("workflow", MeshWorkflowJobState::Queued, 1);
        let stage = |stage_index, kind| MeshWorkflowStageRow {
            job_id: job.id.clone(),
            stage_index,
            kind,
            state: MeshWorkflowStageState::Pending,
            execution_batch_id: None,
            artifacts: Vec::new(),
            error: None,
            updated_at_ms: 1,
        };
        let stages = vec![
            stage(0, MeshWorkflowStageKind::Matting),
            stage(1, MeshWorkflowStageKind::Shape),
            stage(2, MeshWorkflowStageKind::Paint),
        ];
        insert_job_with_stages(&db, &job, &stages).unwrap();
        assert!(claim_job(&db, &job.id, 2).unwrap());
        for index in 0..3 {
            assert!(attach_stage_execution(&db, &job.id, index, "mesh-batch", 3).unwrap());
        }
        assert!(fail_stage_and_job(&db, &job.id, 0, "paint failed", 4).unwrap());
        let stages = stages_for_job(&db, &job.id).unwrap();
        assert!(stages[..3]
            .iter()
            .all(|stage| stage.state == MeshWorkflowStageState::Failed));
        assert_eq!(
            get_job(&db, &job.id).unwrap().unwrap().state,
            MeshWorkflowJobState::Failed
        );
        assert!(resume_job(&db, &job.id, 4).unwrap());
        let retried = stages_for_job(&db, &job.id).unwrap();
        assert!(retried.iter().all(|stage| {
            stage.state == MeshWorkflowStageState::Pending
                && stage.execution_batch_id.is_none()
                && stage.updated_at_ms == 5
        }));
    }

    /// A child refused at durable admission was never attached, so the
    /// current stage has no batch id and is still `pending`. Failing "every
    /// stage sharing the batch" matched nothing (`= NULL`), only the JOB went
    /// failed, and Resume — which resets `failed` stages — re-queued the
    /// identical refusal with every stage still reading PENDING (#1672). The
    /// current stage itself is what failed, batch or no batch.
    #[test]
    fn a_stage_refused_before_attachment_is_marked_failed_and_resumable() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut job = job("workflow", MeshWorkflowJobState::Queued, 1);
        job.stage_count = 4;
        let stage = |stage_index, kind, state| MeshWorkflowStageRow {
            job_id: job.id.clone(),
            stage_index,
            kind,
            state,
            execution_batch_id: None,
            artifacts: Vec::new(),
            error: None,
            updated_at_ms: 1,
        };
        let mut matting = stage(
            0,
            MeshWorkflowStageKind::Matting,
            MeshWorkflowStageState::Completed,
        );
        matting.execution_batch_id = Some("matting-batch".into());
        let stages = vec![
            matting,
            stage(
                1,
                MeshWorkflowStageKind::Shape,
                MeshWorkflowStageState::Pending,
            ),
            stage(
                2,
                MeshWorkflowStageKind::Paint,
                MeshWorkflowStageState::Pending,
            ),
            stage(
                3,
                MeshWorkflowStageKind::Finalize,
                MeshWorkflowStageState::Pending,
            ),
        ];
        insert_job_with_stages(&db, &job, &stages).unwrap();
        assert!(claim_job(&db, &job.id, 2).unwrap());
        assert!(set_current_stage(&db, &job.id, 1, 3).unwrap());

        assert!(fail_stage_and_job(&db, &job.id, 1, "refused at the door", 4).unwrap());
        let failed = stages_for_job(&db, &job.id).unwrap();
        assert_eq!(failed[0].state, MeshWorkflowStageState::Completed);
        assert_eq!(failed[1].state, MeshWorkflowStageState::Failed);
        assert_eq!(failed[1].error.as_deref(), Some("refused at the door"));
        assert_eq!(failed[2].state, MeshWorkflowStageState::Pending);
        assert_eq!(failed[3].state, MeshWorkflowStageState::Pending);
        assert_eq!(
            get_job(&db, &job.id).unwrap().unwrap().state,
            MeshWorkflowJobState::Failed
        );

        assert!(resume_job(&db, &job.id, 5).unwrap());
        let retried = stages_for_job(&db, &job.id).unwrap();
        assert_eq!(retried[0].state, MeshWorkflowStageState::Completed);
        assert_eq!(retried[1].state, MeshWorkflowStageState::Pending);
        assert!(retried[1].error.is_none());
    }

    /// Shape and paint share one child. The runner reports shape complete
    /// from the child's live progress; the batch's artifact then settles both
    /// rows, and a paint failure takes the live-completed shape back to
    /// pending on resume so the retry does not wait on a dead batch.
    #[test]
    fn a_live_completed_shape_settles_with_its_batch_and_retries_with_it() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut job = job("workflow", MeshWorkflowJobState::Queued, 1);
        job.stage_count = 3;
        let stage = |stage_index, kind| MeshWorkflowStageRow {
            job_id: job.id.clone(),
            stage_index,
            kind,
            state: MeshWorkflowStageState::Pending,
            execution_batch_id: None,
            artifacts: Vec::new(),
            error: None,
            updated_at_ms: 1,
        };
        let stages = vec![
            stage(0, MeshWorkflowStageKind::Shape),
            stage(1, MeshWorkflowStageKind::Paint),
            stage(2, MeshWorkflowStageKind::Finalize),
        ];
        insert_job_with_stages(&db, &job, &stages).unwrap();
        assert!(claim_job(&db, &job.id, 2).unwrap());
        assert!(attach_stage_executions(&db, &job.id, &[0, 1], "mesh-batch", 3).unwrap());

        // Not before it is attached, and not twice.
        assert!(complete_stage_live(&db, &job.id, 0, 4).unwrap());
        assert!(!complete_stage_live(&db, &job.id, 0, 5).unwrap());
        assert!(!complete_stage_live(&db, &job.id, 2, 5).unwrap());
        let live = stages_for_job(&db, &job.id).unwrap();
        assert_eq!(live[0].state, MeshWorkflowStageState::Completed);
        assert_eq!(live[0].execution_batch_id.as_deref(), Some("mesh-batch"));
        assert!(live[0].artifacts.is_empty());
        assert_eq!(live[1].state, MeshWorkflowStageState::Running);

        // The batch settles both rows with the one artifact.
        let artifact = MeshWorkflowArtifact {
            role: "final_glb".into(),
            relative_path: "stages/001/final.glb".into(),
            media_type: "model/gltf-binary".into(),
            sha256: "a".repeat(64),
            byte_length: 1,
        };
        assert!(complete_stage_execution(
            &db,
            &job.id,
            "mesh-batch",
            std::slice::from_ref(&artifact),
            6
        )
        .unwrap());
        let settled = stages_for_job(&db, &job.id).unwrap();
        assert!(settled[..2].iter().all(|stage| {
            stage.state == MeshWorkflowStageState::Completed
                && stage.artifacts == vec![artifact.clone()]
        }));

        // The failure branch: a fresh job whose paint fails after shape was
        // reported done.
        let mut retry = MeshWorkflowJobRow {
            id: "retry".into(),
            ..job.clone()
        };
        retry.stage_count = 3;
        let stages = stages
            .iter()
            .map(|stage| MeshWorkflowStageRow {
                job_id: retry.id.clone(),
                ..stage.clone()
            })
            .collect::<Vec<_>>();
        insert_job_with_stages(&db, &retry, &stages).unwrap();
        assert!(claim_job(&db, &retry.id, 2).unwrap());
        assert!(attach_stage_executions(&db, &retry.id, &[0, 1], "retry-batch", 3).unwrap());
        assert!(complete_stage_live(&db, &retry.id, 0, 4).unwrap());
        assert!(set_current_stage(&db, &retry.id, 1, 4).unwrap());
        assert!(fail_stage_and_job(&db, &retry.id, 1, "paint failed", 5).unwrap());
        let failed = stages_for_job(&db, &retry.id).unwrap();
        assert_eq!(failed[0].state, MeshWorkflowStageState::Completed);
        assert_eq!(failed[1].state, MeshWorkflowStageState::Failed);
        assert!(resume_job(&db, &retry.id, 6).unwrap());
        let reset = stages_for_job(&db, &retry.id).unwrap();
        assert!(reset[..2].iter().all(|stage| {
            stage.state == MeshWorkflowStageState::Pending && stage.execution_batch_id.is_none()
        }));
        assert_eq!(reset[2].state, MeshWorkflowStageState::Pending);
    }

    #[test]
    fn cancelled_parent_fences_late_child_attachment() {
        let db = MetadataDb::open_in_memory().unwrap();
        let job = job("workflow", MeshWorkflowJobState::Running, 1);
        let stages = [MeshWorkflowStageKind::Shape, MeshWorkflowStageKind::Paint]
            .into_iter()
            .enumerate()
            .map(|(stage_index, kind)| MeshWorkflowStageRow {
                job_id: job.id.clone(),
                stage_index: stage_index as u32,
                kind,
                state: MeshWorkflowStageState::Pending,
                execution_batch_id: None,
                artifacts: Vec::new(),
                error: None,
                updated_at_ms: 1,
            })
            .collect::<Vec<_>>();
        let mut job = job;
        job.stage_count = 2;
        insert_job_with_stages(&db, &job, &stages).unwrap();
        assert!(cancel_job(&db, &job.id, 2).unwrap());
        assert!(!attach_stage_executions(&db, &job.id, &[0, 1], "late-child", 3).unwrap());
        assert!(stages_for_job(&db, &job.id)
            .unwrap()
            .iter()
            .all(|stage| stage.state == MeshWorkflowStageState::Pending
                && stage.execution_batch_id.is_none()));
    }

    #[test]
    fn deletion_is_limited_to_settled_workflows_and_cascades_stages() {
        let db = MetadataDb::open_in_memory().unwrap();
        let mut row = job("workflow", MeshWorkflowJobState::Running, 1);
        row.stage_count = 1;
        insert_job_with_stages(
            &db,
            &row,
            &[MeshWorkflowStageRow {
                job_id: row.id.clone(),
                stage_index: 0,
                kind: MeshWorkflowStageKind::Shape,
                state: MeshWorkflowStageState::Pending,
                execution_batch_id: None,
                artifacts: Vec::new(),
                error: None,
                updated_at_ms: 1,
            }],
        )
        .unwrap();
        assert!(claim_settled_deletion(&db, &row.id, 2).unwrap().is_none());
        assert!(transition(
            &db,
            &row.id,
            &[MeshWorkflowJobState::Running],
            MeshWorkflowJobState::Failed,
            Some("retryable"),
            2,
        )
        .unwrap());
        let claimed = claim_settled_deletion(&db, &row.id, 3).unwrap().unwrap();
        assert_eq!(claimed.error.as_deref(), Some(DELETION_CLAIM_ERROR));
        assert!(!resume_job(&db, &row.id, 4).unwrap());
        assert!(claim_settled_deletion(&db, &row.id, 5).unwrap().is_some());
        assert!(delete_claimed_job(&db, &row.id).unwrap());
        assert!(get_job(&db, &row.id).unwrap().is_none());
        assert!(stages_for_job(&db, &row.id).unwrap().is_empty());
    }

    #[test]
    fn shared_child_attachment_and_completion_are_all_or_nothing() {
        let db = MetadataDb::open_in_memory().unwrap();
        let job = job("workflow", MeshWorkflowJobState::Queued, 1);
        insert_job(&db, &job).unwrap();
        for (stage_index, kind) in [
            MeshWorkflowStageKind::Matting,
            MeshWorkflowStageKind::Shape,
            MeshWorkflowStageKind::Paint,
        ]
        .into_iter()
        .enumerate()
        {
            upsert_stage(
                &db,
                &MeshWorkflowStageRow {
                    job_id: job.id.clone(),
                    stage_index: stage_index as u32,
                    kind,
                    state: MeshWorkflowStageState::Pending,
                    execution_batch_id: None,
                    artifacts: Vec::new(),
                    error: None,
                    updated_at_ms: 1,
                },
            )
            .unwrap();
        }
        assert!(claim_job(&db, &job.id, 2).unwrap());
        assert!(attach_stage_executions(&db, &job.id, &[0, 1, 2], "batch", 2).unwrap());
        assert!(!attach_stage_executions(&db, &job.id, &[0, 1, 2], "other", 3).unwrap());
        let artifact = MeshWorkflowArtifact {
            role: "final_glb".into(),
            relative_path: "stages/001/final.glb".into(),
            media_type: "model/gltf-binary".into(),
            sha256: "a".repeat(64),
            byte_length: 1,
        };
        assert!(complete_stage_execution(
            &db,
            &job.id,
            "batch",
            std::slice::from_ref(&artifact),
            4
        )
        .unwrap());
        let rows = stages_for_job(&db, &job.id).unwrap();
        assert!(rows.iter().all(|stage| {
            stage.state == MeshWorkflowStageState::Completed
                && stage.artifacts == vec![artifact.clone()]
        }));
    }
}
