//! Long-lived orchestrator for durable mesh workflows.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context};
use mold_core::mesh_workflow::{
    CreateMeshWorkflowRequest, MeshWorkflowArtifact, MeshWorkflowJobState, MeshWorkflowManifest,
    MeshWorkflowStageKind, MeshWorkflowStageState,
};
use mold_core::{GenerationBatchAdmissionRequest, GenerationBatchChildState, OutputFormat};
use mold_db::mesh_workflow_jobs::{self, MeshWorkflowJobRow, MeshWorkflowStageRow};
use sha2::{Digest, Sha256};

use crate::state::{AppState, SseCompletionPayload};

pub struct MeshWorkflowRunnerHandle {
    kick: tokio::sync::mpsc::UnboundedSender<RunnerCommand>,
}

enum RunnerCommand {
    Kick,
    Shutdown,
}

impl MeshWorkflowRunnerHandle {
    pub fn kick(&self) {
        let _ = self.kick.send(RunnerCommand::Kick);
    }

    pub fn shutdown(&self) {
        let _ = self.kick.send(RunnerCommand::Shutdown);
    }
}

pub fn spawn_runner(state: AppState, workflows_root: PathBuf) -> MeshWorkflowRunnerHandle {
    let (kick, receiver) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(run_loop(state, workflows_root, receiver));
    MeshWorkflowRunnerHandle { kick }
}

pub fn startup_reconcile(
    db: &mold_db::MetadataDb,
    workflows_root: &Path,
) -> anyhow::Result<(usize, usize)> {
    let paused = mesh_workflow_jobs::pause_unfinished_for_recovery(db, now_ms())?;
    let mut repaired = 0;
    for row in mesh_workflow_jobs::list_jobs(db)? {
        let manifest = match MeshWorkflowManifest::read_from_dir(&row.work_dir) {
            Ok(manifest) => manifest,
            Err(error) => {
                tracing::warn!(workflow_id = %row.id, %error, "mesh workflow manifest could not be reconciled");
                continue;
            }
        };
        for stage in &manifest.stages {
            mesh_workflow_jobs::upsert_stage(
                db,
                &MeshWorkflowStageRow {
                    job_id: row.id.clone(),
                    stage_index: stage.index,
                    kind: stage.kind,
                    state: if stage.state == MeshWorkflowStageState::Running
                        && stage.execution_batch_id.is_none()
                    {
                        MeshWorkflowStageState::Pending
                    } else {
                        stage.state
                    },
                    execution_batch_id: stage.execution_batch_id.clone(),
                    artifacts: stage.artifacts.clone(),
                    error: stage.error.clone(),
                    updated_at_ms: now_ms(),
                },
            )?;
        }
        if row.request_json != manifest.request_json {
            tracing::warn!(workflow_id = %row.id, "mesh workflow SQLite request differs from manifest; manifest remains recovery authority");
        }
        repaired += 1;
    }
    std::fs::create_dir_all(workflows_root)?;
    Ok((paused, repaired))
}

async fn run_loop(
    state: AppState,
    workflows_root: PathBuf,
    mut commands: tokio::sync::mpsc::UnboundedReceiver<RunnerCommand>,
) {
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            command = commands.recv() => match command {
                Some(RunnerCommand::Shutdown) | None => break,
                Some(RunnerCommand::Kick) => {}
            },
            _ = tick.tick() => {}
        }
        state.queue_pause.wait_if_paused().await;
        let Some(db) = state.metadata_db.as_ref().as_ref() else {
            break;
        };
        let job = match mesh_workflow_jobs::next_queued_job(db) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(%error, "mesh workflow queue lookup failed");
                continue;
            }
        };
        let Some(job) = job else { continue };
        if let Err(error) = drive_job(&state, &workflows_root, job.clone()).await {
            tracing::warn!(workflow_id = %job.id, error = %format!("{error:#}"), "mesh workflow failed");
            let current = mesh_workflow_jobs::get_job(db, &job.id).ok().flatten();
            if let Some(current) = current.filter(|row| row.state == MeshWorkflowJobState::Running)
            {
                let message = format!("{error:#}");
                if !mesh_workflow_jobs::fail_stage_and_job(
                    db,
                    &job.id,
                    current.current_stage,
                    &message,
                    now_ms(),
                )
                .unwrap_or(false)
                {
                    let _ = mesh_workflow_jobs::transition(
                        db,
                        &job.id,
                        &[MeshWorkflowJobState::Running],
                        MeshWorkflowJobState::Failed,
                        Some(&message),
                        now_ms(),
                    );
                }
                let _ = update_manifest_from_db(db, &current);
            }
        }
    }
}

async fn drive_job(
    state: &AppState,
    workflows_root: &Path,
    job: MeshWorkflowJobRow,
) -> anyhow::Result<()> {
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .context("mesh workflow database is unavailable")?;
    if !mesh_workflow_jobs::claim_job(db, &job.id, now_ms())? {
        return Ok(());
    }
    loop {
        let current = mesh_workflow_jobs::get_job(db, &job.id)?
            .context("mesh workflow disappeared during execution")?;
        if current.state != MeshWorkflowJobState::Running {
            return Ok(());
        }
        let stages = mesh_workflow_jobs::stages_for_job(db, &job.id)?;
        let request: CreateMeshWorkflowRequest = serde_json::from_str(&current.request_json)?;
        match next_action(&request, &stages)? {
            NextAction::SubmitImage { attempt_epoch } => {
                let request = crate::mesh_workflow_media::hydrate_for_admission(
                    workflows_root,
                    &current.work_dir,
                    "image",
                )?;
                let batch_id =
                    admit_child(state, &current.id, "image", attempt_epoch, request).await?;
                attach_batch(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Image],
                    &batch_id,
                )?;
                update_manifest_from_db(db, &current)?;
            }
            NextAction::WaitImage { batch_id } => {
                let Some(filename) = completed_child_filename(state, &batch_id).await? else {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                };
                let output_dir = state.config.read().await.effective_output_dir();
                let artifact = retain_gallery_artifact(
                    &output_dir,
                    &current,
                    0,
                    "generated_image",
                    &filename,
                )?;
                complete_kinds(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Image],
                    &[artifact],
                )?;
                update_manifest_from_db(db, &current)?;
            }
            NextAction::SubmitMesh { attempt_epoch } => {
                let label = if matches!(request, CreateMeshWorkflowRequest::TextToMesh { .. }) {
                    "mesh"
                } else {
                    "texture"
                };
                let mut child = crate::mesh_workflow_media::hydrate_for_admission(
                    workflows_root,
                    &current.work_dir,
                    label,
                )?;
                if matches!(child.output_format, None | Some(OutputFormat::Png)) {
                    child.output_format = Some(OutputFormat::Glb);
                }
                if matches!(request, CreateMeshWorkflowRequest::TextToMesh { .. }) {
                    let image = artifact_for_kind(&stages, MeshWorkflowStageKind::Image)?;
                    child.source_image =
                        Some(std::fs::read(current.work_dir.join(&image.relative_path))?);
                }
                let batch_id =
                    admit_child(state, &current.id, "mesh", attempt_epoch, child).await?;
                let kinds = mesh_execution_kinds(&stages);
                attach_batch(db, &current.id, &stages, &kinds, &batch_id)?;
                update_manifest_from_db(db, &current)?;
            }
            NextAction::WaitMesh { batch_id } => {
                let Some(filename) = completed_child_filename(state, &batch_id).await? else {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                };
                let stage_index = stages
                    .iter()
                    .find(|stage| stage.kind == MeshWorkflowStageKind::Shape)
                    .or_else(|| {
                        stages
                            .iter()
                            .find(|stage| stage.kind == MeshWorkflowStageKind::Paint)
                    })
                    .map(|stage| stage.stage_index)
                    .context("mesh workflow has no mesh-producing stage")?;
                let output_dir = state.config.read().await.effective_output_dir();
                let artifact = retain_gallery_artifact(
                    &output_dir,
                    &current,
                    stage_index,
                    "final_glb",
                    &filename,
                )?;
                let kinds = mesh_execution_kinds(&stages);
                complete_kinds(db, &current.id, &stages, &kinds, &[artifact.clone()])?;
                update_manifest_from_db(db, &current)?;
            }
            NextAction::Finalize { batch_id } => {
                let filename = completed_child_filename(state, &batch_id)
                    .await?
                    .context("completed mesh workflow child is no longer complete")?;
                let artifact = mesh_artifact(&stages)?.clone();
                let finalize = stages
                    .iter()
                    .find(|stage| stage.kind == MeshWorkflowStageKind::Finalize)
                    .context("mesh workflow has no finalize stage")?;
                mesh_workflow_jobs::upsert_stage(
                    db,
                    &MeshWorkflowStageRow {
                        state: MeshWorkflowStageState::Completed,
                        artifacts: vec![artifact],
                        updated_at_ms: now_ms(),
                        ..finalize.clone()
                    },
                )?;
                if !mesh_workflow_jobs::complete_job(db, &current.id, &filename, now_ms())? {
                    bail!("mesh workflow completion lost its running claim");
                }
                update_manifest_from_db(db, &current)?;
                return Ok(());
            }
            NextAction::Done => return Ok(()),
        }
    }
}

enum NextAction {
    SubmitImage { attempt_epoch: i64 },
    WaitImage { batch_id: String },
    SubmitMesh { attempt_epoch: i64 },
    WaitMesh { batch_id: String },
    Finalize { batch_id: String },
    Done,
}

fn next_action(
    request: &CreateMeshWorkflowRequest,
    stages: &[MeshWorkflowStageRow],
) -> anyhow::Result<NextAction> {
    if matches!(request, CreateMeshWorkflowRequest::TextToMesh { .. }) {
        let image = stages
            .iter()
            .find(|stage| stage.kind == MeshWorkflowStageKind::Image)
            .context("text-to-mesh workflow has no image stage")?;
        match image.state {
            MeshWorkflowStageState::Pending => {
                return Ok(NextAction::SubmitImage {
                    attempt_epoch: image.updated_at_ms,
                })
            }
            MeshWorkflowStageState::Running => {
                return Ok(NextAction::WaitImage {
                    batch_id: image
                        .execution_batch_id
                        .clone()
                        .context("running image stage has no durable batch")?,
                })
            }
            MeshWorkflowStageState::Failed => bail!("image stage is failed"),
            MeshWorkflowStageState::Completed => {}
        }
    }
    let mesh_stages = stages
        .iter()
        .filter(|stage| {
            matches!(
                stage.kind,
                MeshWorkflowStageKind::Matting
                    | MeshWorkflowStageKind::Shape
                    | MeshWorkflowStageKind::Paint
            )
        })
        .collect::<Vec<_>>();
    if mesh_stages
        .iter()
        .all(|stage| stage.state == MeshWorkflowStageState::Completed)
    {
        let finalize = stages
            .iter()
            .find(|stage| stage.kind == MeshWorkflowStageKind::Finalize)
            .context("mesh workflow has no finalize stage")?;
        return match finalize.state {
            MeshWorkflowStageState::Pending | MeshWorkflowStageState::Running => {
                Ok(NextAction::Finalize {
                    batch_id: mesh_stages
                        .iter()
                        .find_map(|stage| stage.execution_batch_id.clone())
                        .context("completed mesh stage has no durable batch")?,
                })
            }
            MeshWorkflowStageState::Completed => Ok(NextAction::Done),
            MeshWorkflowStageState::Failed => bail!("finalize stage is failed"),
        };
    }
    if let Some(batch_id) = mesh_stages
        .iter()
        .find_map(|stage| stage.execution_batch_id.clone())
    {
        return Ok(NextAction::WaitMesh { batch_id });
    }
    let attempt_epoch = mesh_stages
        .iter()
        .find(|stage| stage.state == MeshWorkflowStageState::Pending)
        .map(|stage| stage.updated_at_ms)
        .context("pending mesh stage group has no attempt epoch")?;
    Ok(NextAction::SubmitMesh { attempt_epoch })
}

fn mesh_artifact(stages: &[MeshWorkflowStageRow]) -> anyhow::Result<&MeshWorkflowArtifact> {
    [MeshWorkflowStageKind::Paint, MeshWorkflowStageKind::Shape]
        .into_iter()
        .find_map(|kind| {
            stages
                .iter()
                .find(|stage| stage.kind == kind)
                .and_then(|stage| stage.artifacts.first())
        })
        .context("completed mesh workflow has no retained GLB artifact")
}

async fn admit_child(
    state: &AppState,
    workflow_id: &str,
    stage: &str,
    attempt_epoch: i64,
    request: mold_core::GenerateRequest,
) -> anyhow::Result<String> {
    let admission = state
        .queue_journal
        .queue_media_admission()
        .context("durable generation admission is unavailable")?;
    let outcome = admission
        .admit_batch(
            state,
            None,
            Some(crate::reference_uploads::ReferenceIdentity::AuthDisabled {
                instance_id: state.instance_id.to_string(),
            }),
            GenerationBatchAdmissionRequest {
                client_batch_id: deterministic_batch_id(workflow_id, stage, attempt_epoch),
                requests: vec![request],
            },
            None,
            SseCompletionPayload::MetadataOnly,
        )
        .await
        .map_err(|error| anyhow!(error.error))?;
    Ok(outcome.status.id)
}

async fn completed_child_filename(
    state: &AppState,
    batch_id: &str,
) -> anyhow::Result<Option<String>> {
    let detail = state
        .queue_journal
        .durable_generation_batch(batch_id)
        .map_err(anyhow::Error::msg)?
        .context("mesh workflow child generation batch is missing")?;
    let status = crate::routes::generation_batch_status(&state.instance_id, detail);
    let child = status
        .children
        .first()
        .context("mesh workflow child batch is empty")?;
    match child.state {
        GenerationBatchChildState::Accepted
        | GenerationBatchChildState::Running
        | GenerationBatchChildState::Cancelling
        | GenerationBatchChildState::Paused => Ok(None),
        GenerationBatchChildState::Complete => Ok(Some(
            child
                .result
                .as_ref()
                .and_then(|result| result.filename.clone())
                .context("completed mesh workflow child has no gallery filename")?,
        )),
        GenerationBatchChildState::Held
        | GenerationBatchChildState::Failed
        | GenerationBatchChildState::Cancelled => bail!(
            "mesh workflow child {}: {}",
            child.job_id,
            child
                .error
                .as_deref()
                .unwrap_or("generation did not complete")
        ),
    }
}

fn attach_batch(
    db: &mold_db::MetadataDb,
    job_id: &str,
    stages: &[MeshWorkflowStageRow],
    kinds: &[MeshWorkflowStageKind],
    batch_id: &str,
) -> anyhow::Result<()> {
    let stage_indices = stages
        .iter()
        .filter(|stage| kinds.contains(&stage.kind))
        .map(|stage| stage.stage_index)
        .collect::<Vec<_>>();
    if !mesh_workflow_jobs::attach_stage_executions(db, job_id, &stage_indices, batch_id, now_ms())?
    {
        bail!("mesh workflow stage group lost its pending claim");
    }
    Ok(())
}

fn complete_kinds(
    db: &mold_db::MetadataDb,
    job_id: &str,
    stages: &[MeshWorkflowStageRow],
    kinds: &[MeshWorkflowStageKind],
    artifacts: &[MeshWorkflowArtifact],
) -> anyhow::Result<()> {
    let batch_id = stages
        .iter()
        .filter(|stage| kinds.contains(&stage.kind))
        .find_map(|stage| stage.execution_batch_id.as_deref())
        .context("mesh workflow stage group has no durable batch")?;
    if !mesh_workflow_jobs::complete_stage_execution(db, job_id, batch_id, artifacts, now_ms())? {
        bail!("mesh workflow stage group lost its running claim");
    }
    if let Some(next) = stages
        .iter()
        .filter(|stage| !kinds.contains(&stage.kind))
        .find(|stage| stage.state == MeshWorkflowStageState::Pending)
    {
        let _ = mesh_workflow_jobs::set_current_stage(db, job_id, next.stage_index, now_ms())?;
    }
    Ok(())
}

fn mesh_execution_kinds(stages: &[MeshWorkflowStageRow]) -> Vec<MeshWorkflowStageKind> {
    stages
        .iter()
        .filter_map(|stage| {
            matches!(
                stage.kind,
                MeshWorkflowStageKind::Matting
                    | MeshWorkflowStageKind::Shape
                    | MeshWorkflowStageKind::Paint
            )
            .then_some(stage.kind)
        })
        .collect()
}

fn artifact_for_kind(
    stages: &[MeshWorkflowStageRow],
    kind: MeshWorkflowStageKind,
) -> anyhow::Result<&MeshWorkflowArtifact> {
    stages
        .iter()
        .find(|stage| stage.kind == kind)
        .and_then(|stage| stage.artifacts.first())
        .with_context(|| format!("mesh workflow {kind:?} artifact is missing"))
}

fn retain_gallery_artifact(
    output_dir: &Path,
    job: &MeshWorkflowJobRow,
    stage_index: u32,
    role: &str,
    filename: &str,
) -> anyhow::Result<MeshWorkflowArtifact> {
    if filename.contains('/') || filename.contains('\\') {
        bail!("generation returned an invalid gallery filename");
    }
    let source = output_dir.join(filename);
    let extension = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("bin");
    let relative_path = format!("stages/{stage_index:03}/{role}.{extension}");
    let destination = job.work_dir.join(&relative_path);
    std::fs::create_dir_all(destination.parent().expect("artifact path has parent"))?;
    std::fs::copy(&source, &destination)
        .with_context(|| format!("retaining workflow artifact from '{}'", source.display()))?;
    let bytes = std::fs::read(&destination)?;
    Ok(MeshWorkflowArtifact {
        role: role.into(),
        relative_path,
        media_type: match extension {
            "png" => "image/png",
            "webp" => "image/webp",
            "jpg" | "jpeg" => "image/jpeg",
            "glb" => "model/gltf-binary",
            _ => "application/octet-stream",
        }
        .into(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        byte_length: bytes.len() as u64,
    })
}

pub(crate) fn update_manifest_from_db(
    db: &mold_db::MetadataDb,
    job: &MeshWorkflowJobRow,
) -> anyhow::Result<()> {
    let current = mesh_workflow_jobs::get_job(db, &job.id)?.context("mesh workflow disappeared")?;
    let stages = mesh_workflow_jobs::stages_for_job(db, &job.id)?;
    let mut manifest = MeshWorkflowManifest::read_from_dir(&job.work_dir)?;
    manifest.stages = stages
        .into_iter()
        .map(|stage| mold_core::mesh_workflow::MeshWorkflowStageRecord {
            index: stage.stage_index,
            kind: stage.kind,
            state: stage.state,
            execution_batch_id: stage.execution_batch_id,
            artifacts: stage.artifacts,
            error: stage.error,
        })
        .collect();
    manifest.output_filename = current.output_filename;
    manifest.write_atomic(&job.work_dir)?;
    Ok(())
}

fn deterministic_batch_id(workflow_id: &str, stage: &str, attempt_epoch: i64) -> String {
    let digest = Sha256::digest(format!(
        "mold.mesh-workflow.batch.v1\0{workflow_id}\0{stage}\0{attempt_epoch}"
    ));
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    uuid::Uuid::from_bytes(bytes).to_string()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(
        index: u32,
        kind: MeshWorkflowStageKind,
        state: MeshWorkflowStageState,
        batch: Option<&str>,
    ) -> MeshWorkflowStageRow {
        MeshWorkflowStageRow {
            job_id: "workflow".into(),
            stage_index: index,
            kind,
            state,
            execution_batch_id: batch.map(str::to_owned),
            artifacts: (kind == MeshWorkflowStageKind::Paint
                && state == MeshWorkflowStageState::Completed)
                .then(|| MeshWorkflowArtifact {
                    role: "final_glb".into(),
                    relative_path: "stages/002/final_glb.glb".into(),
                    media_type: "model/gltf-binary".into(),
                    sha256: "00".repeat(32),
                    byte_length: 1,
                })
                .into_iter()
                .collect(),
            error: None,
            updated_at_ms: 1,
        }
    }

    #[test]
    fn child_batch_identity_is_stable_and_stage_scoped() {
        let first = deterministic_batch_id("workflow", "image", 10);
        assert_eq!(first, deterministic_batch_id("workflow", "image", 10));
        assert_ne!(first, deterministic_batch_id("workflow", "mesh", 10));
        assert_ne!(first, deterministic_batch_id("workflow", "image", 11));
        assert!(uuid::Uuid::parse_str(&first).is_ok());
    }

    #[test]
    fn completed_mesh_stages_recover_through_finalize() {
        let child = serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
            "prompt": "",
            "model": "test",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0,
            "seed": 1
        }))
        .unwrap();
        let request = CreateMeshWorkflowRequest::MeshTexture {
            texture_request: Box::new(child),
        };
        let stages = vec![
            stage(
                0,
                MeshWorkflowStageKind::Matting,
                MeshWorkflowStageState::Completed,
                Some("batch"),
            ),
            stage(
                1,
                MeshWorkflowStageKind::Paint,
                MeshWorkflowStageState::Completed,
                Some("batch"),
            ),
            stage(
                2,
                MeshWorkflowStageKind::Finalize,
                MeshWorkflowStageState::Pending,
                None,
            ),
        ];
        assert!(matches!(
            next_action(&request, &stages).unwrap(),
            NextAction::Finalize { batch_id } if batch_id == "batch"
        ));
        assert_eq!(mesh_artifact(&stages).unwrap().role, "final_glb");
    }
}
