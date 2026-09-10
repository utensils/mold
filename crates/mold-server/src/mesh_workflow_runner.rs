//! Long-lived orchestrator for durable mesh workflows.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context};
use mold_core::mesh_workflow::{
    CreateMeshWorkflowRequest, MeshWorkflowArtifact, MeshWorkflowJobState, MeshWorkflowManifest,
    MeshWorkflowProvenance, MeshWorkflowStageKind, MeshWorkflowStageState,
};
use mold_core::{
    GenerationBatchAdmissionRequest, GenerationBatchChildState, MeshMattingMode, OutputFormat,
};
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
        if row.error.as_deref() == Some(mesh_workflow_jobs::DELETION_CLAIM_ERROR) {
            match crate::mesh_workflow_media::purge_claimed(workflows_root, &row.work_dir)
                .and_then(|()| mesh_workflow_jobs::delete_claimed_job(db, &row.id).map(|_| ()))
            {
                Ok(()) => repaired += 1,
                Err(error) => {
                    tracing::warn!(workflow_id = %row.id, %error, "claimed mesh workflow deletion remains pending")
                }
            }
            continue;
        }
        let manifest = match MeshWorkflowManifest::read_from_dir(&row.work_dir) {
            Ok(manifest) => manifest,
            Err(error) => {
                tracing::warn!(workflow_id = %row.id, %error, "mesh workflow manifest could not be reconciled");
                continue;
            }
        };
        if row.request_json != manifest.request_json {
            tracing::warn!(workflow_id = %row.id, "mesh workflow SQLite request differs from its portable manifest; refreshing the manifest from execution authority");
        }
        // SQLite commits the execution claim before the portable manifest is
        // refreshed. A crash in that narrow window therefore leaves SQLite
        // ahead; projecting the stale manifest back into the database would
        // discard the attached child id and admit duplicate GPU work.
        update_manifest_from_db(db, &row)?;
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
        let request_mode = request.mode_str();
        match next_action(&request, &stages)? {
            NextAction::SubmitImage { attempt_epoch } => {
                let image = crate::mesh_workflow_media::hydrate_for_admission(
                    workflows_root,
                    &current.work_dir,
                    "image",
                )?;
                let batch_id = admit_child(
                    state,
                    "image",
                    MeshWorkflowProvenance {
                        job_id: current.id.clone(),
                        mode: request_mode.to_string(),
                        role: "generated_image".into(),
                        stage_index: stage_index_for(&stages, MeshWorkflowStageKind::Image)?,
                    },
                    attempt_epoch,
                    image,
                )
                .await?;
                if !attach_batch(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Image],
                    &batch_id,
                )? {
                    crate::routes::cancel_generation_batch_children(state, &batch_id)
                        .await
                        .map_err(|error| anyhow!(error.error))?;
                    continue;
                }
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
                    stage_index_for(&stages, MeshWorkflowStageKind::Image)?,
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
            NextAction::SubmitMatting { attempt_epoch } => {
                let source = if matches!(request, CreateMeshWorkflowRequest::TextToMesh { .. }) {
                    let image = artifact_for_kind(&stages, MeshWorkflowStageKind::Image)?;
                    std::fs::read(current.work_dir.join(&image.relative_path))?
                } else {
                    crate::mesh_workflow_media::hydrate_for_admission(
                        workflows_root,
                        &current.work_dir,
                        "texture",
                    )?
                    .source_image
                    .context("mesh-texture matting stage has no appearance image")?
                };
                let matting = workflow_mesh_request(&request)
                    .mesh
                    .as_ref()
                    .and_then(|mesh| mesh.matting)
                    .unwrap_or_default();
                let model = if matting == MeshMattingMode::On {
                    mold_core::manifest::HUNYUAN3D_MATTING_FORCE_MANIFEST
                } else {
                    mold_core::manifest::HUNYUAN3D_MATTING_MANIFEST
                };
                let mut child: mold_core::GenerateRequest =
                    serde_json::from_value(serde_json::json!({
                        "prompt": "",
                        "model": model,
                        "width": 512,
                        "height": 512,
                        "steps": 1,
                        "guidance": 1.0,
                        "seed": 0,
                        "output_format": "png"
                    }))?;
                child.source_image = Some(source);
                let batch_id = admit_child(
                    state,
                    "matting",
                    MeshWorkflowProvenance {
                        job_id: current.id.clone(),
                        mode: request_mode.to_string(),
                        role: "matted_image".into(),
                        stage_index: stage_index_for(&stages, MeshWorkflowStageKind::Matting)?,
                    },
                    attempt_epoch,
                    child,
                )
                .await?;
                if !attach_batch(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Matting],
                    &batch_id,
                )? {
                    crate::routes::cancel_generation_batch_children(state, &batch_id)
                        .await
                        .map_err(|error| anyhow!(error.error))?;
                    continue;
                }
                update_manifest_from_db(db, &current)?;
            }
            NextAction::WaitMatting { batch_id } => {
                let Some(filename) = completed_child_filename(state, &batch_id).await? else {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                };
                let stage_index = stages
                    .iter()
                    .find(|stage| stage.kind == MeshWorkflowStageKind::Matting)
                    .context("mesh workflow has no matting stage")?
                    .stage_index;
                let output_dir = state.config.read().await.effective_output_dir();
                let artifact = retain_gallery_artifact(
                    &output_dir,
                    &current,
                    stage_index,
                    "matted_image",
                    &filename,
                )?;
                complete_kinds(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Matting],
                    &[artifact],
                )?;
                update_manifest_from_db(db, &current)?;
            }
            NextAction::SubmitDelight { attempt_epoch } => {
                let source = if stages
                    .iter()
                    .any(|stage| stage.kind == MeshWorkflowStageKind::Matting)
                {
                    let image = artifact_for_kind(&stages, MeshWorkflowStageKind::Matting)?;
                    std::fs::read(current.work_dir.join(&image.relative_path))?
                } else if matches!(request, CreateMeshWorkflowRequest::TextToMesh { .. }) {
                    let image = artifact_for_kind(&stages, MeshWorkflowStageKind::Image)?;
                    std::fs::read(current.work_dir.join(&image.relative_path))?
                } else {
                    crate::mesh_workflow_media::hydrate_for_admission(
                        workflows_root,
                        &current.work_dir,
                        "texture",
                    )?
                    .source_image
                    .context("mesh-texture delight stage has no appearance image")?
                };
                let mut child: mold_core::GenerateRequest =
                    serde_json::from_value(serde_json::json!({
                        "prompt": "",
                        "model": mold_core::manifest::HUNYUAN3D_DELIGHT_MANIFEST,
                        "width": 512,
                        "height": 512,
                        "steps": 50,
                        "guidance": 1.0,
                        "seed": 42,
                        "scheduler": "euler-ancestral",
                        "output_format": "png"
                    }))?;
                child.source_image = Some(source);
                let batch_id = admit_child(
                    state,
                    "delight",
                    MeshWorkflowProvenance {
                        job_id: current.id.clone(),
                        mode: request_mode.to_string(),
                        role: "delighted_image".into(),
                        stage_index: stage_index_for(&stages, MeshWorkflowStageKind::Delight)?,
                    },
                    attempt_epoch,
                    child,
                )
                .await?;
                if !attach_batch(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Delight],
                    &batch_id,
                )? {
                    crate::routes::cancel_generation_batch_children(state, &batch_id)
                        .await
                        .map_err(|error| anyhow!(error.error))?;
                    continue;
                }
                update_manifest_from_db(db, &current)?;
            }
            NextAction::WaitDelight { batch_id } => {
                let Some(filename) = completed_child_filename(state, &batch_id).await? else {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                };
                let stage_index = stages
                    .iter()
                    .find(|stage| stage.kind == MeshWorkflowStageKind::Delight)
                    .context("mesh workflow has no delight stage")?
                    .stage_index;
                let output_dir = state.config.read().await.effective_output_dir();
                let artifact = retain_gallery_artifact(
                    &output_dir,
                    &current,
                    stage_index,
                    "delighted_image",
                    &filename,
                )?;
                complete_kinds(
                    db,
                    &current.id,
                    &stages,
                    &[MeshWorkflowStageKind::Delight],
                    &[artifact],
                )?;
                update_manifest_from_db(db, &current)?;
            }
            NextAction::SubmitMesh { attempt_epoch } => {
                let label = match request {
                    CreateMeshWorkflowRequest::TextToMesh { .. } => "mesh",
                    CreateMeshWorkflowRequest::MeshTexture { .. } => "texture",
                    CreateMeshWorkflowRequest::MeshRoundtrip { .. } => "roundtrip",
                };
                let mut child = crate::mesh_workflow_media::hydrate_for_admission(
                    workflows_root,
                    &current.work_dir,
                    label,
                )?;
                if matches!(child.output_format, None | Some(OutputFormat::Png)) {
                    child.output_format = Some(OutputFormat::Glb);
                }
                if let Some(mesh) = child.mesh.as_mut() {
                    mesh.matting = Some(MeshMattingMode::Off);
                    mesh.delight = None;
                }
                if matches!(request, CreateMeshWorkflowRequest::TextToMesh { .. }) {
                    let image = artifact_for_kind(
                        &stages,
                        if stages
                            .iter()
                            .any(|stage| stage.kind == MeshWorkflowStageKind::Delight)
                        {
                            MeshWorkflowStageKind::Delight
                        } else if stages
                            .iter()
                            .any(|stage| stage.kind == MeshWorkflowStageKind::Matting)
                        {
                            MeshWorkflowStageKind::Matting
                        } else {
                            MeshWorkflowStageKind::Image
                        },
                    )?;
                    child.source_image =
                        Some(std::fs::read(current.work_dir.join(&image.relative_path))?);
                } else if stages
                    .iter()
                    .any(|stage| stage.kind == MeshWorkflowStageKind::Delight)
                {
                    let image = artifact_for_kind(&stages, MeshWorkflowStageKind::Delight)?;
                    child.source_image =
                        Some(std::fs::read(current.work_dir.join(&image.relative_path))?);
                } else if stages
                    .iter()
                    .any(|stage| stage.kind == MeshWorkflowStageKind::Matting)
                {
                    let image = artifact_for_kind(&stages, MeshWorkflowStageKind::Matting)?;
                    child.source_image =
                        Some(std::fs::read(current.work_dir.join(&image.relative_path))?);
                }
                // The same stage the wait arm retains `final_glb` under.
                let mesh_stage_index = mesh_stage_index(&stages)?;
                let batch_id = admit_child(
                    state,
                    "mesh",
                    MeshWorkflowProvenance {
                        job_id: current.id.clone(),
                        mode: request_mode.to_string(),
                        role: "final_glb".into(),
                        stage_index: mesh_stage_index,
                    },
                    attempt_epoch,
                    child,
                )
                .await?;
                let kinds = mesh_execution_kinds(&stages);
                if !attach_batch(db, &current.id, &stages, &kinds, &batch_id)? {
                    crate::routes::cancel_generation_batch_children(state, &batch_id)
                        .await
                        .map_err(|error| anyhow!(error.error))?;
                    continue;
                }
                update_manifest_from_db(db, &current)?;
            }
            NextAction::WaitMesh { batch_id } => {
                let Some(filename) = completed_child_filename(state, &batch_id).await? else {
                    if let Some(paint_index) =
                        shape_stage_to_complete(&stages, live_child_stage(state, &batch_id).await?)
                            .and_then(|shape_index| {
                                mesh_workflow_jobs::complete_stage_live(
                                    db,
                                    &current.id,
                                    shape_index,
                                    now_ms(),
                                )
                                .ok()
                                .filter(|changed| *changed)
                                .and(stage_index_for(&stages, MeshWorkflowStageKind::Paint).ok())
                            })
                    {
                        let _ = mesh_workflow_jobs::set_current_stage(
                            db,
                            &current.id,
                            paint_index,
                            now_ms(),
                        )?;
                        update_manifest_from_db(db, &current)?;
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                };
                // The same authority the submit arm stamped `final_glb` with:
                // the retained artifact's index and the provenance's must be
                // one number, or the field stops meaning anything.
                let stage_index = mesh_stage_index(&stages)?;
                let output_dir = state.config.read().await.effective_output_dir();
                let artifact = retain_gallery_artifact(
                    &output_dir,
                    &current,
                    stage_index,
                    "final_glb",
                    &filename,
                )?;
                let kinds = mesh_execution_kinds(&stages);
                complete_kinds(
                    db,
                    &current.id,
                    &stages,
                    &kinds,
                    std::slice::from_ref(&artifact),
                )?;
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
    SubmitMatting { attempt_epoch: i64 },
    WaitMatting { batch_id: String },
    SubmitDelight { attempt_epoch: i64 },
    WaitDelight { batch_id: String },
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
    if let Some(matting) = stages
        .iter()
        .find(|stage| stage.kind == MeshWorkflowStageKind::Matting)
    {
        match matting.state {
            MeshWorkflowStageState::Pending => {
                return Ok(NextAction::SubmitMatting {
                    attempt_epoch: matting.updated_at_ms,
                });
            }
            MeshWorkflowStageState::Running => {
                return Ok(NextAction::WaitMatting {
                    batch_id: matting
                        .execution_batch_id
                        .clone()
                        .context("running matting stage has no durable batch")?,
                });
            }
            MeshWorkflowStageState::Failed => bail!("matting stage is failed"),
            MeshWorkflowStageState::Completed => {}
        }
    }
    if let Some(delight) = stages
        .iter()
        .find(|stage| stage.kind == MeshWorkflowStageKind::Delight)
    {
        match delight.state {
            MeshWorkflowStageState::Pending => {
                return Ok(NextAction::SubmitDelight {
                    attempt_epoch: delight.updated_at_ms,
                });
            }
            MeshWorkflowStageState::Running => {
                return Ok(NextAction::WaitDelight {
                    batch_id: delight
                        .execution_batch_id
                        .clone()
                        .context("running delight stage has no durable batch")?,
                });
            }
            MeshWorkflowStageState::Failed => bail!("delight stage is failed"),
            MeshWorkflowStageState::Completed => {}
        }
    }
    let mesh_stages = stages
        .iter()
        .filter(|stage| {
            matches!(
                stage.kind,
                MeshWorkflowStageKind::Shape | MeshWorkflowStageKind::Paint
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

/// The advertised index of `kind` in this job's stage graph.
///
/// The ONE spelling of that lookup. It is what a stamped `stage_index` means,
/// and the value has to match the index its artifact is retained under — so a
/// second hand-inlined copy is a way for the two to drift apart silently.
fn stage_index_for(
    stages: &[MeshWorkflowStageRow],
    kind: MeshWorkflowStageKind,
) -> anyhow::Result<u32> {
    stages
        .iter()
        .find(|stage| stage.kind == kind)
        .map(|stage| stage.stage_index)
        .with_context(|| format!("mesh workflow has no {kind:?} stage"))
}

/// The stage that carries the finished mesh: Shape where the run builds
/// geometry, Paint where it only textures a supplied mesh.
fn mesh_stage_index(stages: &[MeshWorkflowStageRow]) -> anyhow::Result<u32> {
    stage_index_for(stages, MeshWorkflowStageKind::Shape)
        .or_else(|_| stage_index_for(stages, MeshWorkflowStageKind::Paint))
}

/// Admit one stage of a workflow as an ordinary durable generation.
///
/// The stage is stamped with its workflow's provenance HERE, at the one place
/// that knows both. Every stage publishes a real gallery print, so without the
/// stamp the print and the queue row are indistinguishable from a hand-authored
/// render: a queue row could only route back to New image, and a text-to-3-D run
/// scattered its source picture, its matted and delighted copies and its mesh
/// across My images as four unrelated tiles. It rides the request, so the live
/// queue entry (whose `metadata` IS the request) and the published print carry
/// the same answer, and the durable sanitizer retains it across a restart.
async fn admit_child(
    state: &AppState,
    stage: &str,
    provenance: MeshWorkflowProvenance,
    attempt_epoch: i64,
    mut request: mold_core::GenerateRequest,
) -> anyhow::Result<String> {
    let workflow_id = provenance.job_id.clone();
    request.mesh_workflow = Some(provenance);
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
                client_batch_id: deterministic_batch_id(&workflow_id, stage, attempt_epoch),
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

/// The named progress stage the batch's child is reporting right now, if
/// any. `None` while it is queued, loading weights with no named stage,
/// between two stages, or already settled.
async fn live_child_stage(state: &AppState, batch_id: &str) -> anyhow::Result<Option<String>> {
    let detail = state
        .queue_journal
        .durable_generation_batch(batch_id)
        .map_err(anyhow::Error::msg)?
        .context("mesh workflow child generation batch is missing")?;
    let child = detail
        .children
        .first()
        .context("mesh workflow child batch is empty")?;
    Ok(state
        .job_registry
        .progress_snapshot(&child.job_id)
        .flatten()
        .and_then(|progress| progress.stage))
}

/// The Shape stage to report complete from the child's live progress, if the
/// child has moved on to painting.
///
/// Shape and paint run inside ONE queue job — the engine builds geometry,
/// drops the shape checkpoint, then textures — so the workflow only ever
/// learns that the job started and finished, and both stages read RUNNING
/// for the whole render while the queue card already says "Generating PBR
/// views · 3/15". The one live signal separating the halves is the progress
/// stage name; `paint_stages` is the contract both sides read. Progress
/// clears its stage between two named stages, so this fires on the first
/// paint stage observed and `complete_stage_live` makes it idempotent.
fn shape_stage_to_complete(
    stages: &[MeshWorkflowStageRow],
    live_stage: Option<String>,
) -> Option<u32> {
    let live_stage = live_stage?;
    if !mold_inference::hunyuan3d::paint_stages::is_paint_stage(&live_stage) {
        return None;
    }
    let shape = stages
        .iter()
        .find(|stage| stage.kind == MeshWorkflowStageKind::Shape)?;
    let paint = stages
        .iter()
        .find(|stage| stage.kind == MeshWorkflowStageKind::Paint)?;
    (shape.state == MeshWorkflowStageState::Running
        && paint.state == MeshWorkflowStageState::Running
        && shape.execution_batch_id.is_some()
        && shape.execution_batch_id == paint.execution_batch_id)
        .then_some(shape.stage_index)
}

fn attach_batch(
    db: &mold_db::MetadataDb,
    job_id: &str,
    stages: &[MeshWorkflowStageRow],
    kinds: &[MeshWorkflowStageKind],
    batch_id: &str,
) -> anyhow::Result<bool> {
    let stage_indices = stages
        .iter()
        .filter(|stage| kinds.contains(&stage.kind))
        .map(|stage| stage.stage_index)
        .collect::<Vec<_>>();
    mesh_workflow_jobs::attach_stage_executions(db, job_id, &stage_indices, batch_id, now_ms())
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
                MeshWorkflowStageKind::Shape | MeshWorkflowStageKind::Paint
            )
            .then_some(stage.kind)
        })
        .collect()
}

fn workflow_mesh_request(request: &CreateMeshWorkflowRequest) -> &mold_core::GenerateRequest {
    match request {
        CreateMeshWorkflowRequest::TextToMesh { mesh_request, .. } => mesh_request,
        CreateMeshWorkflowRequest::MeshTexture { texture_request } => texture_request,
        CreateMeshWorkflowRequest::MeshRoundtrip { roundtrip_request } => roundtrip_request,
    }
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

    /// Both halves of a textured render read RUNNING for the whole job while
    /// the queue card already said "Generating PBR views" (#1672's sibling).
    /// The first paint stage observed completes Shape; nothing else does.
    #[test]
    fn a_live_paint_stage_completes_shape_and_a_shape_stage_does_not() {
        let running = |index, kind| {
            stage(
                index,
                kind,
                MeshWorkflowStageState::Running,
                Some("mesh-batch"),
            )
        };
        let stages = vec![
            stage(
                0,
                MeshWorkflowStageKind::Matting,
                MeshWorkflowStageState::Completed,
                Some("matting-batch"),
            ),
            running(1, MeshWorkflowStageKind::Shape),
            running(2, MeshWorkflowStageKind::Paint),
            stage(
                3,
                MeshWorkflowStageKind::Finalize,
                MeshWorkflowStageState::Pending,
                None,
            ),
        ];
        for live in [
            "Sampling",
            "Decoding volume",
            "Extracting surface",
            "Simplifying mesh",
        ] {
            assert_eq!(
                shape_stage_to_complete(&stages, Some(live.to_string())),
                None,
                "{live} is geometry"
            );
        }
        assert_eq!(shape_stage_to_complete(&stages, None), None);
        for live in [
            "Unwrapping mesh",
            "Generating PBR views",
            "Baking PBR textures",
        ] {
            assert_eq!(
                shape_stage_to_complete(&stages, Some(live.to_string())),
                Some(1),
                "{live} means geometry is done"
            );
        }

        // Already reported: nothing to do twice.
        let mut reported = stages.clone();
        reported[1].state = MeshWorkflowStageState::Completed;
        assert_eq!(
            shape_stage_to_complete(&reported, Some("Generating PBR views".to_string())),
            None
        );

        // A texture-only run has no Shape stage to complete.
        let texture_only = vec![
            running(0, MeshWorkflowStageKind::Paint),
            stage(
                1,
                MeshWorkflowStageKind::Finalize,
                MeshWorkflowStageState::Pending,
                None,
            ),
        ];
        assert_eq!(
            shape_stage_to_complete(&texture_only, Some("Generating PBR views".to_string())),
            None
        );
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

    #[test]
    fn durable_order_runs_matting_before_delight_before_mesh() {
        let child = serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
            "prompt": "",
            "model": "hunyuan3d:fp16",
            "width": 0,
            "height": 0,
            "steps": 30,
            "guidance": 5.0,
            "seed": 1,
            "output_format": "glb"
        }))
        .unwrap();
        let request = CreateMeshWorkflowRequest::MeshTexture {
            texture_request: Box::new(child),
        };
        let pending = vec![
            stage(
                0,
                MeshWorkflowStageKind::Matting,
                MeshWorkflowStageState::Pending,
                None,
            ),
            stage(
                1,
                MeshWorkflowStageKind::Delight,
                MeshWorkflowStageState::Pending,
                None,
            ),
            stage(
                2,
                MeshWorkflowStageKind::Paint,
                MeshWorkflowStageState::Pending,
                None,
            ),
            stage(
                3,
                MeshWorkflowStageKind::Finalize,
                MeshWorkflowStageState::Pending,
                None,
            ),
        ];
        assert!(matches!(
            next_action(&request, &pending).unwrap(),
            NextAction::SubmitMatting { .. }
        ));

        let mut matted = pending;
        matted[0].state = MeshWorkflowStageState::Completed;
        matted[0].execution_batch_id = Some("matting-batch".into());
        assert!(matches!(
            next_action(&request, &matted).unwrap(),
            NextAction::SubmitDelight { .. }
        ));

        matted[1].state = MeshWorkflowStageState::Completed;
        matted[1].execution_batch_id = Some("delight-batch".into());
        assert!(matches!(
            next_action(&request, &matted).unwrap(),
            NextAction::SubmitMesh { .. }
        ));
    }

    #[test]
    fn startup_finishes_a_claimed_deletion_with_no_manifest() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("mesh-workflows");
        let work_dir = root.join("workflow");
        std::fs::create_dir_all(&work_dir).unwrap();
        std::fs::write(work_dir.join("partial"), b"cleanup-me").unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let request = CreateMeshWorkflowRequest::MeshTexture {
            texture_request: Box::new(
                serde_json::from_value(serde_json::json!({
                    "prompt": "", "model": "test", "width": 0, "height": 0,
                    "steps": 1, "guidance": 1.0, "seed": 1, "output_format": "glb"
                }))
                .unwrap(),
            ),
        };
        let row = MeshWorkflowJobRow {
            id: "workflow".into(),
            state: MeshWorkflowJobState::Cancelled,
            request_json: serde_json::to_string(&request).unwrap(),
            work_dir: work_dir.clone(),
            stage_count: 1,
            current_stage: 0,
            output_filename: None,
            error: Some(mesh_workflow_jobs::DELETION_CLAIM_ERROR.into()),
            created_at_ms: 1,
            updated_at_ms: 2,
        };
        mesh_workflow_jobs::insert_job_with_stages(
            &db,
            &row,
            &[stage(
                0,
                MeshWorkflowStageKind::Paint,
                MeshWorkflowStageState::Failed,
                None,
            )],
        )
        .unwrap();

        assert_eq!(startup_reconcile(&db, &root).unwrap(), (0, 1));
        assert!(mesh_workflow_jobs::get_job(&db, "workflow")
            .unwrap()
            .is_none());
        assert!(!work_dir.exists());
    }
}
