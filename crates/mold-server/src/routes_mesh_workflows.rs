use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::Sse;
use axum::Json;
use mold_core::generation_profile::MeshWorkflowMode;
use mold_core::mesh_workflow::{
    validate_create_mesh_workflow, CreateMeshWorkflowRequest, CreateMeshWorkflowResponse,
    MeshWorkflowEvent, MeshWorkflowJobDetail, MeshWorkflowJobListing, MeshWorkflowJobState,
    MeshWorkflowJobSummary, MeshWorkflowManifest, MeshWorkflowStageRecord, MeshWorkflowStageState,
    MESH_WORKFLOW_CONTRACT_VERSION,
};
use mold_db::mesh_workflow_jobs::{self, MeshWorkflowJobRow, MeshWorkflowStageRow};

use crate::routes::ApiError;
use crate::state::AppState;

const UNAVAILABLE: &str = "MESH_WORKFLOWS_UNAVAILABLE";
const NOT_FOUND: &str = "MESH_WORKFLOW_NOT_FOUND";

fn workflows_root() -> Result<PathBuf, ApiError> {
    mold_core::Config::mold_dir()
        .map(|home| home.join("mesh-workflows"))
        .ok_or_else(|| {
            ApiError::with_code(
                "MOLD_HOME is unavailable",
                UNAVAILABLE,
                StatusCode::SERVICE_UNAVAILABLE,
            )
        })
}

fn db(state: &AppState) -> Result<&mold_db::MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable mesh workflows require the metadata database",
            UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

fn mode(request: &CreateMeshWorkflowRequest) -> MeshWorkflowMode {
    match request {
        CreateMeshWorkflowRequest::TextToMesh { .. } => MeshWorkflowMode::TextToMesh,
        CreateMeshWorkflowRequest::MeshRoundtrip { .. } => MeshWorkflowMode::MeshRoundtrip,
        CreateMeshWorkflowRequest::MeshTexture { .. } => MeshWorkflowMode::MeshTexture,
    }
}

fn mode_name(mode: MeshWorkflowMode) -> &'static str {
    match mode {
        MeshWorkflowMode::ImageToMesh => "image_to_mesh",
        MeshWorkflowMode::MultiviewToMesh => "multiview_to_mesh",
        MeshWorkflowMode::MeshRoundtrip => "mesh_roundtrip",
        MeshWorkflowMode::MeshTexture => "mesh_texture",
        MeshWorkflowMode::TextToMesh => "text_to_mesh",
    }
}

fn stage_records(rows: Vec<MeshWorkflowStageRow>) -> Vec<MeshWorkflowStageRecord> {
    rows.into_iter()
        .map(|row| MeshWorkflowStageRecord {
            index: row.stage_index,
            kind: row.kind,
            state: row.state,
            execution_batch_id: row.execution_batch_id,
            artifacts: row.artifacts,
            error: row.error,
        })
        .collect()
}

fn detail_from_rows(
    row: MeshWorkflowJobRow,
    stages: Vec<MeshWorkflowStageRow>,
) -> Result<MeshWorkflowJobDetail, ApiError> {
    let request: CreateMeshWorkflowRequest =
        serde_json::from_str(&row.request_json).map_err(|error| {
            ApiError::internal(format!("mesh workflow request is corrupt: {error}"))
        })?;
    let current_stage_kind = stages
        .get(row.current_stage as usize)
        .map(|stage| stage.kind);
    Ok(MeshWorkflowJobDetail {
        summary: MeshWorkflowJobSummary {
            contract_version: MESH_WORKFLOW_CONTRACT_VERSION,
            id: row.id,
            state: row.state,
            mode: mode(&request),
            stage_count: row.stage_count,
            current_stage: row.current_stage,
            current_stage_kind,
            output_filename: row.output_filename,
            error: row.error,
            created_at_ms: row.created_at_ms,
            updated_at_ms: row.updated_at_ms,
        },
        request,
        stages: stage_records(stages),
    })
}

fn load_detail(state: &AppState, id: &str) -> Result<MeshWorkflowJobDetail, ApiError> {
    let db = db(state)?;
    let row = mesh_workflow_jobs::get_job(db, id)
        .map_err(|error| ApiError::internal(format!("mesh workflow lookup failed: {error:#}")))?
        .ok_or_else(|| {
            ApiError::with_code("mesh workflow not found", NOT_FOUND, StatusCode::NOT_FOUND)
        })?;
    let stages = mesh_workflow_jobs::stages_for_job(db, id).map_err(|error| {
        ApiError::internal(format!("mesh workflow stage lookup failed: {error:#}"))
    })?;
    detail_from_rows(row, stages)
}

async fn validate_stage_model(
    state: &AppState,
    request: &mold_core::GenerateRequest,
    workflow_mode: Option<MeshWorkflowMode>,
) -> Result<(), ApiError> {
    let family = crate::routes::require_server_model_activation(state, &request.model).await?;
    let mut validation_request = request.clone();
    if workflow_mode == Some(MeshWorkflowMode::TextToMesh)
        && validation_request.source_image.is_none()
    {
        // The preceding durable image stage supplies this field. Validation
        // still needs to exercise the Hunyuan3D image-conditioned recipe.
        validation_request.source_image = Some(vec![0]);
    }
    crate::routes::validate_generate_request(
        &validation_request,
        family.as_deref(),
        mold_core::ReferenceForm::Admitted,
    )
    .map_err(ApiError::validation)?;
    let canonical = mold_core::manifest::resolve_model_name(&request.model);
    let profile = crate::routes::resolved_generation_profile(state, &request.model, &canonical)
        .await
        .ok_or_else(|| {
            ApiError::validation(format!(
                "model '{}' has no generation recipe deliverable by this server build",
                request.model
            ))
        })?;
    mold_core::validate_request_against_generation_profile(&profile, &validation_request)
        .map_err(ApiError::validation)?;
    if let Some(workflow_mode) = workflow_mode {
        let recipe = if let Some(pipeline) = validation_request.pipeline {
            profile
                .recipes
                .iter()
                .find(|recipe| recipe.request_selector.pipeline == Some(pipeline))
        } else {
            profile.default_recipe()
        }
        .ok_or_else(|| ApiError::validation("request has no matching generation recipe"))?;
        let advertised = recipe
            .capabilities
            .mesh
            .as_ref()
            .is_some_and(|mesh| mesh.workflow_modes.contains(&workflow_mode));
        if !advertised {
            return Err(ApiError::validation(format!(
                "model '{}' does not support the '{}' mesh workflow",
                request.model,
                mode_name(workflow_mode)
            )));
        }
    }
    let _ = crate::model_manager::check_model_available(state, &request.model).await?;
    Ok(())
}

async fn validate_workflow_models(
    state: &AppState,
    request: &CreateMeshWorkflowRequest,
) -> Result<(), ApiError> {
    match request {
        CreateMeshWorkflowRequest::TextToMesh {
            image_request,
            mesh_request,
        } => {
            validate_stage_model(state, image_request, None).await?;
            validate_stage_model(state, mesh_request, Some(MeshWorkflowMode::TextToMesh)).await
        }
        CreateMeshWorkflowRequest::MeshTexture { texture_request } => {
            validate_stage_model(state, texture_request, Some(MeshWorkflowMode::MeshTexture)).await
        }
        CreateMeshWorkflowRequest::MeshRoundtrip { roundtrip_request } => {
            validate_stage_model(
                state,
                roundtrip_request,
                Some(MeshWorkflowMode::MeshRoundtrip),
            )
            .await
        }
    }
}

#[utoipa::path(
    post,
    path = "/api/mesh-workflows",
    tag = "mesh-workflows",
    request_body = mold_core::mesh_workflow::CreateMeshWorkflowRequest,
    responses((status = 202, description = "Mesh workflow accepted", body = mold_core::mesh_workflow::CreateMeshWorkflowResponse))
)]
pub(crate) async fn create_mesh_workflow(
    State(state): State<AppState>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    Json(request): Json<CreateMeshWorkflowRequest>,
) -> Result<(StatusCode, Json<CreateMeshWorkflowResponse>), ApiError> {
    validate_create_mesh_workflow(&request).map_err(ApiError::validation)?;
    validate_workflow_models(&state, &request).await?;
    let database = db(&state)?;
    let root = workflows_root()?;
    std::fs::create_dir_all(&root).map_err(|error| {
        ApiError::internal(format!("creating mesh workflow root failed: {error}"))
    })?;
    let id = uuid::Uuid::new_v4().to_string();
    let workflow_dir = root.join(&id);
    std::fs::create_dir_all(&workflow_dir).map_err(|error| {
        ApiError::internal(format!("creating mesh workflow directory failed: {error}"))
    })?;
    let rollback = crate::mesh_workflow_media::CreateRollback::new(&root, &workflow_dir);
    let identity = crate::reference_uploads::ReferenceIdentity::resolve(
        authenticated.as_ref().map(|Extension(value)| value),
        auth_state.as_ref().map(|Extension(value)| value),
        state.instance_id.as_str(),
    );
    let media_roots = state.config.read().await.resolved_media_roots();

    let persisted = match request {
        CreateMeshWorkflowRequest::TextToMesh {
            mut image_request,
            mut mesh_request,
        } => {
            let image_staged = state
                .reference_uploads
                .resolve_request(
                    identity.as_ref(),
                    image_request.as_mut(),
                    &media_roots,
                    None,
                )
                .await?;
            let image = crate::mesh_workflow_media::persist_request(
                &root,
                &workflow_dir,
                &id,
                "image",
                *image_request,
                image_staged.as_ref(),
            )
            .map_err(|error| {
                ApiError::internal(format!("persisting image stage failed: {error:#}"))
            })?;
            let mesh_staged = state
                .reference_uploads
                .resolve_request(identity.as_ref(), mesh_request.as_mut(), &media_roots, None)
                .await?;
            let mesh = crate::mesh_workflow_media::persist_request(
                &root,
                &workflow_dir,
                &id,
                "mesh",
                *mesh_request,
                mesh_staged.as_ref(),
            )
            .map_err(|error| {
                ApiError::internal(format!("persisting mesh stage failed: {error:#}"))
            })?;
            CreateMeshWorkflowRequest::TextToMesh {
                image_request: Box::new(image),
                mesh_request: Box::new(mesh),
            }
        }
        CreateMeshWorkflowRequest::MeshTexture {
            mut texture_request,
        } => {
            let staged = state
                .reference_uploads
                .resolve_request(
                    identity.as_ref(),
                    texture_request.as_mut(),
                    &media_roots,
                    None,
                )
                .await?;
            let texture = crate::mesh_workflow_media::persist_request(
                &root,
                &workflow_dir,
                &id,
                "texture",
                *texture_request,
                staged.as_ref(),
            )
            .map_err(|error| {
                ApiError::internal(format!("persisting texture stage failed: {error:#}"))
            })?;
            CreateMeshWorkflowRequest::MeshTexture {
                texture_request: Box::new(texture),
            }
        }
        CreateMeshWorkflowRequest::MeshRoundtrip {
            mut roundtrip_request,
        } => {
            let staged = state
                .reference_uploads
                .resolve_request(
                    identity.as_ref(),
                    roundtrip_request.as_mut(),
                    &media_roots,
                    None,
                )
                .await?;
            let roundtrip = crate::mesh_workflow_media::persist_request(
                &root,
                &workflow_dir,
                &id,
                "roundtrip",
                *roundtrip_request,
                staged.as_ref(),
            )
            .map_err(|error| {
                ApiError::internal(format!("persisting round-trip stage failed: {error:#}"))
            })?;
            CreateMeshWorkflowRequest::MeshRoundtrip {
                roundtrip_request: Box::new(roundtrip),
            }
        }
    };

    let created_at_ms = now_ms();
    let stages = persisted
        .planned_stage_kinds()
        .into_iter()
        .enumerate()
        .map(|(index, kind)| MeshWorkflowStageRecord {
            index: index as u32,
            kind,
            state: MeshWorkflowStageState::Pending,
            execution_batch_id: None,
            artifacts: Vec::new(),
            error: None,
        })
        .collect::<Vec<_>>();
    let manifest = MeshWorkflowManifest::new(id.clone(), created_at_ms, &persisted, stages.clone())
        .map_err(|error| {
            ApiError::internal(format!("creating mesh workflow manifest failed: {error}"))
        })?;
    manifest.write_atomic(&workflow_dir).map_err(|error| {
        ApiError::internal(format!("writing mesh workflow manifest failed: {error}"))
    })?;
    let request_json = serde_json::to_string(&persisted).map_err(|error| {
        ApiError::internal(format!("serializing mesh workflow failed: {error}"))
    })?;
    let row = MeshWorkflowJobRow {
        id: id.clone(),
        state: MeshWorkflowJobState::Queued,
        request_json,
        work_dir: workflow_dir.clone(),
        stage_count: stages.len() as u32,
        current_stage: 0,
        output_filename: None,
        error: None,
        created_at_ms,
        updated_at_ms: created_at_ms,
    };
    let db_stages = stages
        .into_iter()
        .map(|stage| MeshWorkflowStageRow {
            job_id: id.clone(),
            stage_index: stage.index,
            kind: stage.kind,
            state: stage.state,
            execution_batch_id: None,
            artifacts: Vec::new(),
            error: None,
            updated_at_ms: created_at_ms,
        })
        .collect::<Vec<_>>();
    mesh_workflow_jobs::insert_job_with_stages(database, &row, &db_stages).map_err(|error| {
        ApiError::internal(format!("persisting mesh workflow failed: {error:#}"))
    })?;
    rollback.commit();
    if let Some(runner) = state.mesh_workflows.as_ref() {
        runner.kick();
    }

    Ok((
        StatusCode::ACCEPTED,
        Json(CreateMeshWorkflowResponse {
            job_id: id,
            request_warnings: Vec::new(),
        }),
    ))
}

#[utoipa::path(
    get,
    path = "/api/mesh-workflows",
    tag = "mesh-workflows",
    responses((status = 200, description = "Mesh workflows", body = mold_core::mesh_workflow::MeshWorkflowJobListing))
)]
pub(crate) async fn list_mesh_workflows(
    State(state): State<AppState>,
) -> Result<Json<MeshWorkflowJobListing>, ApiError> {
    let database = db(&state)?;
    let rows = mesh_workflow_jobs::list_jobs(database)
        .map_err(|error| ApiError::internal(format!("listing mesh workflows failed: {error:#}")))?;
    let mut jobs = Vec::with_capacity(rows.len());
    for row in rows {
        let stages = mesh_workflow_jobs::stages_for_job(database, &row.id).map_err(|error| {
            ApiError::internal(format!("listing mesh workflow stages failed: {error:#}"))
        })?;
        jobs.push(detail_from_rows(row, stages)?.summary);
    }
    Ok(Json(MeshWorkflowJobListing { jobs }))
}

#[utoipa::path(
    get,
    path = "/api/mesh-workflows/{id}",
    tag = "mesh-workflows",
    params(("id" = String, Path, description = "Mesh workflow id")),
    responses((status = 200, description = "Mesh workflow detail", body = mold_core::mesh_workflow::MeshWorkflowJobDetail))
)]
pub(crate) async fn get_mesh_workflow(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<MeshWorkflowJobDetail>, ApiError> {
    Ok(Json(load_detail(&state, &id)?))
}

#[utoipa::path(
    get,
    path = "/api/mesh-workflows/{id}/events",
    tag = "mesh-workflows",
    params(("id" = String, Path, description = "Mesh workflow id")),
    responses((status = 200, description = "Mesh workflow event stream"))
)]
pub(crate) async fn mesh_workflow_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let _ = load_detail(&state, &id)?;
    let stream = async_stream::stream! {
        let mut revision = None;
        loop {
            match load_detail(&state, &id) {
                Ok(detail) => {
                    let current = (detail.summary.updated_at_ms, detail.summary.state);
                    if revision != Some(current) {
                        revision = Some(current);
                        let settled = detail.summary.state.is_settled();
                        let event = Event::default()
                            .event("mesh_workflow")
                            .json_data(MeshWorkflowEvent::Snapshot { job: detail })
                            .unwrap_or_else(|error| Event::default().event("error").data(error.to_string()));
                        yield Ok(event);
                        if settled {
                            break;
                        }
                    }
                }
                Err(error) => {
                    yield Ok(Event::default().event("error").data(error.error));
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

#[utoipa::path(
    post,
    path = "/api/mesh-workflows/{id}/resume",
    tag = "mesh-workflows",
    params(("id" = String, Path, description = "Mesh workflow id")),
    responses((status = 202, description = "Mesh workflow queued"))
)]
pub(crate) async fn resume_mesh_workflow(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let existing = load_detail(&state, &id)?;
    if existing.summary.error.as_deref() == Some(mesh_workflow_jobs::DELETION_CLAIM_ERROR) {
        return Err(ApiError::validation(
            "mesh workflow deletion is already in progress",
        ));
    }
    if !matches!(
        existing.summary.state,
        MeshWorkflowJobState::Paused | MeshWorkflowJobState::Failed
    ) {
        return Err(ApiError::validation(format!(
            "mesh workflow cannot resume from {:?}",
            existing.summary.state
        )));
    }
    let batches = existing
        .stages
        .iter()
        .filter_map(|stage| stage.execution_batch_id.as_deref())
        .collect::<std::collections::BTreeSet<_>>();
    for batch_id in batches {
        let Some(batch) = state
            .queue_journal
            .durable_generation_batch(batch_id)
            .map_err(|error| {
                ApiError::internal(format!("loading mesh workflow child batch failed: {error}"))
            })?
        else {
            continue;
        };
        for child in batch.children {
            if child.state == "paused" {
                crate::routes::set_one_queue_job_paused(&state, &child.job_id, false).await?;
            }
        }
    }
    if mesh_workflow_jobs::resume_job(db(&state)?, &id, now_ms())
        .map_err(|error| ApiError::internal(format!("resuming mesh workflow failed: {error:#}")))?
    {
        let row = mesh_workflow_jobs::get_job(db(&state)?, &id)
            .map_err(|error| {
                ApiError::internal(format!("loading resumed mesh workflow failed: {error:#}"))
            })?
            .ok_or_else(|| {
                ApiError::with_code("mesh workflow not found", NOT_FOUND, StatusCode::NOT_FOUND)
            })?;
        crate::mesh_workflow_runner::update_manifest_from_db(db(&state)?, &row).map_err(
            |error| {
                ApiError::internal(format!(
                    "checkpointing resumed mesh workflow failed: {error:#}"
                ))
            },
        )?;
        if let Some(runner) = state.mesh_workflows.as_ref() {
            runner.kick();
        }
        Ok(StatusCode::ACCEPTED)
    } else {
        Err(ApiError::validation(format!(
            "mesh workflow cannot resume from {:?}",
            existing.summary.state
        )))
    }
}

#[utoipa::path(
    post,
    path = "/api/mesh-workflows/{id}/cancel",
    tag = "mesh-workflows",
    params(("id" = String, Path, description = "Mesh workflow id")),
    responses((status = 202, description = "Mesh workflow cancelled"))
)]
pub(crate) async fn cancel_mesh_workflow(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let existing = load_detail(&state, &id)?;
    if existing.summary.error.as_deref() == Some(mesh_workflow_jobs::DELETION_CLAIM_ERROR) {
        return Err(ApiError::validation(
            "mesh workflow deletion is already in progress",
        ));
    }
    let changed = mesh_workflow_jobs::cancel_job(db(&state)?, &id, now_ms()).map_err(|error| {
        ApiError::internal(format!("cancelling mesh workflow failed: {error:#}"))
    })?;
    if !changed && existing.summary.state != MeshWorkflowJobState::Cancelled {
        return Err(ApiError::validation("mesh workflow is already settled"));
    }
    // Cancel the parent first. This fences a runner that admitted a child but
    // has not attached it yet; its conditional attach then loses the claim
    // and cancels that exact child itself. Loading attached ids after the
    // fence covers the inverse ordering.
    let detail = load_detail(&state, &id)?;
    let batches = detail
        .stages
        .iter()
        .filter_map(|stage| stage.execution_batch_id.as_deref())
        .collect::<std::collections::BTreeSet<_>>();
    for batch_id in batches {
        crate::routes::cancel_generation_batch_children(&state, batch_id).await?;
    }
    if changed || existing.summary.state == MeshWorkflowJobState::Cancelled {
        let row = mesh_workflow_jobs::get_job(db(&state)?, &id)
            .map_err(|error| {
                ApiError::internal(format!("loading cancelled mesh workflow failed: {error:#}"))
            })?
            .ok_or_else(|| {
                ApiError::with_code("mesh workflow not found", NOT_FOUND, StatusCode::NOT_FOUND)
            })?;
        crate::mesh_workflow_runner::update_manifest_from_db(db(&state)?, &row).map_err(
            |error| {
                ApiError::internal(format!(
                    "checkpointing cancelled mesh workflow failed: {error:#}"
                ))
            },
        )?;
        Ok(StatusCode::ACCEPTED)
    } else {
        unreachable!("settled states returned before child cancellation")
    }
}

#[utoipa::path(
    delete,
    path = "/api/mesh-workflows/{id}",
    tag = "mesh-workflows",
    params(("id" = String, Path, description = "Mesh workflow id")),
    responses((status = 204, description = "Settled workflow and retained artifacts deleted"))
)]
pub(crate) async fn delete_mesh_workflow(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let row = match mesh_workflow_jobs::claim_settled_deletion(db(&state)?, &id, now_ms()).map_err(
        |error| ApiError::internal(format!("claiming workflow deletion failed: {error:#}")),
    )? {
        Some(row) => row,
        None => {
            return match mesh_workflow_jobs::get_job(db(&state)?, &id).map_err(|error| {
                ApiError::internal(format!("loading workflow failed: {error:#}"))
            })? {
                Some(_) => Err(ApiError::validation(
                    "cancel or wait for the mesh workflow before deleting it",
                )),
                None => Err(ApiError::with_code(
                    "mesh workflow not found",
                    NOT_FOUND,
                    StatusCode::NOT_FOUND,
                )),
            };
        }
    };
    let root = workflows_root()?;
    let release_root = root.clone();
    let release_dir = row.work_dir;
    tokio::task::spawn_blocking(move || {
        crate::mesh_workflow_media::purge_claimed(&release_root, &release_dir)
    })
    .await
    .map_err(|error| ApiError::internal(format!("workflow deletion task failed: {error}")))?
    .map_err(|error: anyhow::Error| {
        ApiError::internal(format!(
            "releasing mesh workflow artifacts failed: {error:#}"
        ))
    })?;
    let _ = mesh_workflow_jobs::delete_claimed_job(db(&state)?, &id)
        .map_err(|error| ApiError::internal(format!("deleting mesh workflow failed: {error:#}")))?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_projection_keeps_execution_identity_and_mode() {
        let request = || {
            serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
                "prompt": "",
                "model": "test",
                "width": 64,
                "height": 64,
                "steps": 1,
                "guidance": 1.0,
                "seed": 1
            }))
            .unwrap()
        };
        let row = MeshWorkflowJobRow {
            id: "workflow".into(),
            state: MeshWorkflowJobState::Running,
            request_json: serde_json::to_string(&CreateMeshWorkflowRequest::TextToMesh {
                image_request: Box::new(request()),
                mesh_request: Box::new(request()),
            })
            .unwrap(),
            work_dir: PathBuf::from("workflow"),
            stage_count: 1,
            current_stage: 0,
            output_filename: None,
            error: None,
            created_at_ms: 1,
            updated_at_ms: 2,
        };
        let detail = detail_from_rows(
            row,
            vec![MeshWorkflowStageRow {
                job_id: "workflow".into(),
                stage_index: 0,
                kind: mold_core::mesh_workflow::MeshWorkflowStageKind::Image,
                state: MeshWorkflowStageState::Running,
                execution_batch_id: Some("batch".into()),
                artifacts: Vec::new(),
                error: None,
                updated_at_ms: 2,
            }],
        )
        .unwrap();
        assert_eq!(detail.summary.mode, MeshWorkflowMode::TextToMesh);
        assert_eq!(
            detail.stages[0].execution_batch_id.as_deref(),
            Some("batch")
        );
    }
}
