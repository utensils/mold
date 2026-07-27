use std::convert::Infallible;
use std::path::Path as FsPath;
use std::pin::Pin;

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{sse, IntoResponse, Response, Sse},
    Json,
};
use futures_core::Stream;
use mold_core::chain::{ChainRequest, ChainScript};
use mold_core::chain_job::{
    settled, ChainJobDetail, ChainJobEvent, ChainJobListing, ChainJobManifest, ChainJobStageDetail,
    ChainJobState, ChainJobSummary, CreateChainJobResponse, GcOutcome, JobDirLayout, RetakeRequest,
    StageState,
};
use mold_db::chain_jobs::{self, ChainJobRow};
use mold_db::MetadataDb;

use crate::routes::ApiError;
use crate::state::AppState;

type ChainJobSseStream =
    Pin<Box<dyn Stream<Item = Result<sse::Event, Infallible>> + Send + 'static>>;

const CHAIN_JOBS_UNAVAILABLE: &str = "CHAIN_JOBS_UNAVAILABLE";
const CHAIN_JOB_NOT_FOUND: &str = "CHAIN_JOB_NOT_FOUND";
const CHAIN_JOB_RUNNING: &str = "CHAIN_JOB_RUNNING";
const CHAIN_JOB_NOT_RESUMABLE: &str = "CHAIN_JOB_NOT_RESUMABLE";
const RETAKE_SPLICE_REQUIRES_CUT_OR_FADE: &str = "RETAKE_SPLICE_REQUIRES_CUT_OR_FADE";
pub const CHAIN_JOB_EPHEMERAL: &str = "CHAIN_JOB_EPHEMERAL";

#[derive(Debug, Clone, serde::Deserialize, utoipa::ToSchema)]
pub struct ChainPlacementPreviewRequest {
    pub request: ChainRequest,
    #[serde(default = "default_preview_copies")]
    pub copies: u32,
}

fn default_preview_copies() -> u32 {
    1
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum ChainPlacementPreviewBody {
    Wrapped(ChainPlacementPreviewRequest),
    Legacy(ChainRequest),
}

impl ChainPlacementPreviewBody {
    fn into_parts(self) -> (ChainRequest, u32) {
        match self {
            Self::Wrapped(body) => (body.request, body.copies),
            Self::Legacy(request) => (request, 1),
        }
    }
}

/// 202. Validates ChainRequest::normalise + chain_limits caps + family
/// chain-capability + audio gate. 503 CHAIN_JOBS_UNAVAILABLE when DB
/// disabled (state.chain_jobs None). Creates job dir + manifest + rows,
/// kicks runner.
#[utoipa::path(
    post,
    path = "/api/chain-jobs",
    tag = "chain-jobs",
    request_body = mold_core::ChainRequest,
    responses((status = 202, description = "Chain job accepted", body = mold_core::chain_job::CreateChainJobResponse))
)]
pub async fn create_chain_job(
    State(state): State<AppState>,
    Json(mut req): Json<ChainRequest>,
) -> Result<(StatusCode, Json<CreateChainJobResponse>), ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    crate::routes_chain::validate_and_normalize_chain_family(&state, &mut req).await?;
    let req = req
        .normalise()
        .map_err(|e| ApiError::validation(e.to_string()))?;
    if req.output_format != mold_core::OutputFormat::Mp4 {
        return Err(ApiError::validation(
            "durable chain jobs currently require output_format = mp4; legacy /api/generate/chain may request gif/webp/apng via the shim",
        ));
    }
    let job_id = uuid::Uuid::new_v4().to_string();
    let jobs_root = jobs_root()?;
    crate::chain_job_runner::create_job_with_params(
        db,
        &jobs_root,
        crate::chain_job_runner::CreateJobParams {
            id: job_id.clone(),
            ephemeral: false,
            request: req,
        },
    )
    .map_err(|e| ApiError::internal(format!("failed to create chain job: {e:#}")))?;

    handle.kick();
    Ok((
        StatusCode::ACCEPTED,
        Json(CreateChainJobResponse { job_id }),
    ))
}

/// Forward-compatible placement capability probe for a normalized durable
/// sequence.
///
/// Until production chain stages retain frozen per-device execution plans,
/// this deliberately returns `authoritative=false` and `outcome=unsupported`.
/// It accepts both the preferred `{ request, copies }` body and the legacy raw
/// `ChainRequest` without creating jobs, downloading assets, or touching CUDA.
#[utoipa::path(
    post,
    path = "/api/chain-jobs/placement-preview",
    tag = "chain-jobs",
    request_body = ChainPlacementPreviewRequest,
    responses(
        (status = 200, description = "Chain placement capability response", body = mold_core::GenerationPlacementPreview)
    )
)]
pub async fn preview_chain_job_placement(
    State(state): State<AppState>,
    Json(body): Json<ChainPlacementPreviewBody>,
) -> Json<mold_core::GenerationPlacementPreview> {
    let (mut req, copies) = body.into_parts();
    let unavailable = |reason: String| mold_core::GenerationPlacementPreview {
        version: 1,
        authoritative: true,
        state_version: 0,
        plan_version: 0,
        outcome: "infeasible".to_string(),
        reason: Some(reason),
        candidate: None,
        stage_candidates: Vec::new(),
    };
    if !(1..=64).contains(&copies) {
        return Json(unavailable("copies must be between 1 and 64".to_string()));
    }
    if let Err(error) =
        crate::routes_chain::validate_and_normalize_chain_family(&state, &mut req).await
    {
        return Json(unavailable(error.error));
    }
    let req = match req.normalise() {
        Ok(req) => req,
        Err(error) => return Json(unavailable(error.to_string())),
    };
    // Production durable-chain stages do not yet retain the frozen per-device
    // execution plans used by ordinary generation. A coarse model-size
    // projection can choose a device that cannot execute the actual component
    // placement on heterogeneous hosts, so it must never be advertised as
    // authoritative. Keep the endpoint and copies=N wire contract available
    // for forward compatibility, while older clients may still send a raw
    // ChainRequest.
    let _ = (state, req);
    let mut response = unavailable(
        "exact durable-chain placement requires frozen per-device stage plans".to_string(),
    );
    response.authoritative = false;
    response.outcome = "unsupported".to_string();
    Json(response)
}

#[utoipa::path(
    get,
    path = "/api/chain-jobs",
    tag = "chain-jobs",
    responses((status = 200, description = "Chain jobs", body = mold_core::chain_job::ChainJobListing))
)]
pub async fn list_chain_jobs(
    State(state): State<AppState>,
) -> Result<Json<ChainJobListing>, ApiError> {
    chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let rows = chain_jobs::list_jobs(db)
        .map_err(|e| ApiError::internal(format!("failed to list chain jobs: {e:#}")))?;
    let jobs = rows
        .into_iter()
        .map(|row| summary_for_row(&row, read_manifest_optional(&row, &root).as_ref()))
        .collect();
    Ok(Json(ChainJobListing { jobs }))
}

/// 404 unknown.
#[utoipa::path(
    get,
    path = "/api/chain-jobs/{id}",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    responses((status = 200, description = "Chain job detail", body = mold_core::chain_job::ChainJobDetail))
)]
pub async fn get_chain_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ChainJobDetail>, ApiError> {
    chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    job_detail_for(db, &root, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .map(Json)
        .ok_or_else(|| not_found(&id))
}

/// subscribe() FIRST, then synthesize Snapshot from DB+manifest, then live.
#[utoipa::path(
    get,
    path = "/api/chain-jobs/{id}/events",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    responses((status = 200, description = "Chain job event stream"))
)]
pub async fn chain_job_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<ChainJobSseStream>, ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let mut live = handle
        .subscribe(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to subscribe to chain job: {e:#}")))?;
    let root = jobs_root()?;
    let detail = job_detail_for(db, &root, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job snapshot: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let snapshot = ChainJobEvent::Snapshot { job: detail };
    let stream = async_stream::stream! {
        yield Ok(sse_event(snapshot));
        loop {
            match live.recv().await {
                Ok(event) => yield Ok(sse_event(event)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => break,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Ok(Sse::new(Box::pin(stream) as ChainJobSseStream))
}

/// 202; 409 CHAIN_JOB_RUNNING when state==Running; 404; Completed -> 409.
#[utoipa::path(
    post,
    path = "/api/chain-jobs/{id}/resume",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    responses((status = 202, description = "Chain job queued", body = mold_core::chain_job::ChainJobSummary))
)]
pub async fn resume_chain_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<ChainJobSummary>), ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let _guard = handle.lock_job(&id).await;
    let row = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    if read_manifest_optional(&row, &root).is_some_and(|manifest| manifest.ephemeral) {
        return Err(conflict(
            CHAIN_JOB_EPHEMERAL,
            "ephemeral chain jobs are internal to legacy generate/chain shims and cannot be resumed",
        ));
    }
    if ![
        ChainJobState::Interrupted,
        ChainJobState::Failed,
        ChainJobState::Cancelled,
    ]
    .contains(&row.state)
    {
        return Err(conflict_for_current_state(
            db,
            &id,
            "resume",
            CHAIN_JOB_NOT_RESUMABLE,
        )?);
    }
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|e| ApiError::internal(format!("failed to read chain manifest: {e:#}")))?;
    let current_stage = manifest
        .stage_status
        .iter()
        .find(|stage| stage.state != StageState::Completed)
        .map(|stage| stage.idx)
        .unwrap_or(manifest.stage_status.len() as u32);
    let now = now_ms();
    let queued = chain_jobs::try_transition(
        db,
        &id,
        &[
            ChainJobState::Interrupted,
            ChainJobState::Failed,
            ChainJobState::Cancelled,
        ],
        ChainJobState::Queued,
        None,
        now,
    )
    .map_err(|e| ApiError::internal(format!("failed to queue chain job: {e:#}")))?;
    if !queued {
        return Err(conflict_for_current_state(
            db,
            &id,
            "resume",
            CHAIN_JOB_NOT_RESUMABLE,
        )?);
    }
    chain_jobs::set_current_stage(db, &id, current_stage, now)
        .map_err(|e| ApiError::internal(format!("failed to update chain stage: {e:#}")))?;
    handle.kick();
    let updated = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to reload chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    Ok((
        StatusCode::ACCEPTED,
        Json(summary_for_row(
            &updated,
            read_manifest_optional(&updated, &root).as_ref(),
        )),
    ))
}

#[utoipa::path(
    post,
    path = "/api/chain-jobs/{id}/retake",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    request_body = mold_core::chain_job::RetakeRequest,
    responses((status = 202, description = "Retake queued", body = mold_core::chain_job::ChainJobSummary))
)]
pub async fn retake_chain_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<RetakeRequest>,
) -> Result<(StatusCode, Json<ChainJobSummary>), ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let _guard = handle.lock_job(&id).await;
    let updated = crate::chain_job_runner::apply_retake(db, &root, &id, &req).map_err(|e| {
        let msg = e.to_string();
        if msg.contains(RETAKE_SPLICE_REQUIRES_CUT_OR_FADE) {
            conflict(RETAKE_SPLICE_REQUIRES_CUT_OR_FADE, msg)
        } else if msg.contains("out of bounds") {
            ApiError::with_code(
                msg,
                "CHAIN_JOB_STAGE_OUT_OF_BOUNDS",
                StatusCode::BAD_REQUEST,
            )
        } else if msg.contains("not found") {
            not_found(&id)
        } else if msg.contains(CHAIN_JOB_RUNNING) {
            conflict(CHAIN_JOB_RUNNING, "chain job is running")
        } else if msg.contains("not retakeable") {
            conflict("CHAIN_JOB_NOT_RETAKEABLE", msg)
        } else {
            ApiError::internal(format!("failed to apply retake: {msg}"))
        }
    })?;
    handle.kick();
    Ok((
        StatusCode::ACCEPTED,
        Json(summary_for_row(
            &updated,
            read_manifest_optional(&updated, &root).as_ref(),
        )),
    ))
}

/// 202, idempotent.
#[utoipa::path(
    post,
    path = "/api/chain-jobs/{id}/cancel",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    responses((status = 202, description = "Cancel requested", body = mold_core::chain_job::ChainJobSummary))
)]
pub async fn cancel_chain_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<ChainJobSummary>), ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let _guard = handle.lock_job(&id).await;
    let row = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    if row.state == ChainJobState::Running {
        let _ = handle.request_cancel(&id);
    } else if row.state == ChainJobState::Queued {
        let changed = chain_jobs::try_transition(
            db,
            &id,
            &[ChainJobState::Queued],
            ChainJobState::Cancelled,
            None,
            now_ms(),
        )
        .map_err(|e| ApiError::internal(format!("failed to cancel queued chain job: {e:#}")))?;
        if changed {
            handle.publish_settled_state(&id, ChainJobState::Cancelled, None);
        } else {
            cancel_after_cas_loss(handle, db, &id)?;
        }
    } else if settled(row.state) {
        // Idempotent: settled jobs have already performed cancel/bus cleanup
        // on the winning settled CAS.
    }
    let updated = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to reload chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    Ok((
        StatusCode::ACCEPTED,
        Json(summary_for_row(
            &updated,
            read_manifest_optional(&updated, &root).as_ref(),
        )),
    ))
}

/// 204; 409 CHAIN_JOB_RUNNING while running; removes job dir + row.
#[utoipa::path(
    delete,
    path = "/api/chain-jobs/{id}",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    responses((status = 204, description = "Chain job deleted"))
)]
pub async fn delete_chain_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    {
        let _guard = handle.lock_job(&id).await;
        let row = chain_jobs::get_job(db, &id)
            .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
            .ok_or_else(|| not_found(&id))?;
        if row.state == ChainJobState::Running {
            return Err(conflict(CHAIN_JOB_RUNNING, "chain job is running"));
        }
        if row.job_dir.exists() {
            std::fs::remove_dir_all(&row.job_dir)
                .map_err(|e| ApiError::internal(format!("failed to remove chain job dir: {e}")))?;
        }
        let deleted = chain_jobs::delete_job_not_running(db, &id)
            .map_err(|e| ApiError::internal(format!("failed to delete chain job: {e:#}")))?;
        if !deleted {
            return Err(conflict_for_current_state(
                db,
                &id,
                "delete",
                CHAIN_JOB_RUNNING,
            )?);
        }
    }
    handle.cleanup_deleted(&id);
    handle.remove_job_lock(&id);
    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(
    post,
    path = "/api/chain-jobs/gc",
    tag = "chain-jobs",
    responses((status = 200, description = "Chain job GC outcome", body = mold_core::chain_job::GcOutcome))
)]
pub async fn gc_chain_jobs(State(state): State<AppState>) -> Result<Json<GcOutcome>, ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let outcome = handle
        .request_gc()
        .await
        .map_err(|e| ApiError::internal(format!("chain job GC failed: {e:#}")))?;
    Ok(Json(outcome))
}

/// image/jpeg; 404 job/stage/preview missing.
#[utoipa::path(
    get,
    path = "/api/chain-jobs/{id}/stages/{idx}/preview",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id"), ("idx" = u32, Path, description = "Stage index")),
    responses((status = 200, description = "Stage preview JPEG", content_type = "image/jpeg"))
)]
pub async fn chain_job_stage_preview(
    State(state): State<AppState>,
    Path((id, idx)): Path<(String, u32)>,
) -> Result<Response, ApiError> {
    chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let detail = job_detail_for(db, &root, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let stage = detail
        .stages
        .iter()
        .find(|stage| stage.idx == idx)
        .ok_or_else(|| not_found(&id))?;
    if !stage.has_preview {
        return Err(not_found(&id));
    }
    let row = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let path = JobDirLayout::new(row.job_dir).preview_path(idx);
    let bytes = std::fs::read(&path).map_err(|_| {
        ApiError::with_code(
            "stage preview not found",
            CHAIN_JOB_NOT_FOUND,
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(([(header::CONTENT_TYPE, "image/jpeg")], bytes).into_response())
}

/// DB rows + manifest join; has_preview from disk; script = EFFECTIVE.
pub(crate) fn job_detail_for(
    db: &MetadataDb,
    _jobs_root: &FsPath,
    id: &str,
) -> anyhow::Result<Option<ChainJobDetail>> {
    let Some(row) = chain_jobs::get_job(db, id)? else {
        return Ok(None);
    };
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir)?;
    let effective = crate::chain_job_runner::effective_request(&manifest)?;
    let layout = JobDirLayout::new(row.job_dir.clone());
    let stages = manifest
        .stage_status
        .iter()
        .map(|stage| ChainJobStageDetail {
            idx: stage.idx,
            state: stage.state,
            seed: stage.seed,
            frames_emitted: stage.frames_emitted,
            generation_time_ms: stage.generation_time_ms,
            has_preview: layout.preview_path(stage.idx).is_file(),
            error: stage.error.clone(),
        })
        .collect();
    Ok(Some(ChainJobDetail {
        summary: summary_for_row(&row, Some(&manifest)),
        stages,
        finalizes: manifest.finalizes.clone(),
        retakes: manifest.retakes.clone(),
        script: ChainScript::from(&effective),
    }))
}

fn chain_jobs_handle(
    state: &AppState,
) -> Result<&std::sync::Arc<crate::chain_job_runner::ChainJobRunnerHandle>, ApiError> {
    state.chain_jobs.as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable chain jobs are unavailable because the metadata DB is disabled",
            CHAIN_JOBS_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

fn metadata_db(state: &AppState) -> Result<&MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "durable chain jobs are unavailable because the metadata DB is disabled",
            CHAIN_JOBS_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

pub(crate) fn jobs_root() -> Result<std::path::PathBuf, ApiError> {
    mold_core::Config::mold_dir()
        .map(|dir| dir.join("jobs"))
        .ok_or_else(|| {
            ApiError::with_code(
                "durable chain jobs are unavailable because MOLD_HOME could not be resolved",
                CHAIN_JOBS_UNAVAILABLE,
                StatusCode::SERVICE_UNAVAILABLE,
            )
        })
}

fn summary_for_row(row: &ChainJobRow, manifest: Option<&ChainJobManifest>) -> ChainJobSummary {
    ChainJobSummary {
        id: row.id.clone(),
        state: row.state,
        model: row.model.clone(),
        stage_count: row.stage_count,
        current_stage: row.current_stage,
        created_at_unix_ms: row.created_at_ms.max(0) as u64,
        updated_at_unix_ms: row.updated_at_ms.max(0) as u64,
        error: row.error.clone(),
        ephemeral: manifest.is_some_and(|manifest| manifest.ephemeral),
    }
}

fn read_manifest_optional(row: &ChainJobRow, _root: &FsPath) -> Option<ChainJobManifest> {
    ChainJobManifest::read_from_dir(&row.job_dir).ok()
}

fn sse_event(event: ChainJobEvent) -> sse::Event {
    match serde_json::to_string(&event) {
        Ok(data) => sse::Event::default().event("chain_job").data(data),
        Err(err) => sse::Event::default().event("error").data(
            serde_json::json!({ "message": format!("failed to serialize chain job event: {err}") })
                .to_string(),
        ),
    }
}

fn not_found(id: &str) -> ApiError {
    ApiError::with_code(
        format!("chain job {id} not found"),
        CHAIN_JOB_NOT_FOUND,
        StatusCode::NOT_FOUND,
    )
}

fn conflict(code: &str, msg: impl Into<String>) -> ApiError {
    ApiError::with_code(msg, code, StatusCode::CONFLICT)
}

fn conflict_for_current_state(
    db: &MetadataDb,
    id: &str,
    action: &str,
    default_code: &str,
) -> Result<ApiError, ApiError> {
    let row = chain_jobs::get_job(db, id)
        .map_err(|e| ApiError::internal(format!("failed to reload chain job: {e:#}")))?
        .ok_or_else(|| not_found(id))?;
    if row.state == ChainJobState::Running {
        return Ok(conflict(CHAIN_JOB_RUNNING, "chain job is running"));
    }
    if action == "resume" && row.state == ChainJobState::Completed {
        return Ok(conflict(
            CHAIN_JOB_NOT_RESUMABLE,
            "completed chain job has nothing to resume",
        ));
    }
    Ok(conflict(
        default_code,
        format!(
            "cannot {action} chain job from current state {}",
            row.state.as_str()
        ),
    ))
}

fn cancel_after_cas_loss(
    handle: &crate::chain_job_runner::ChainJobRunnerHandle,
    db: &MetadataDb,
    id: &str,
) -> Result<(), ApiError> {
    let row = chain_jobs::get_job(db, id)
        .map_err(|e| ApiError::internal(format!("failed to reload chain job: {e:#}")))?
        .ok_or_else(|| not_found(id))?;
    if row.state == ChainJobState::Running {
        let _ = handle.request_cancel(id);
    } else if settled(row.state) {
        // Idempotent no-op: the winning settled CAS already cleaned the
        // cancel registry and event bus.
    }
    Ok(())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use mold_core::chain::{ChainStage, TransitionMode};
    use mold_core::GenerateRequest;
    use mold_core::OutputFormat;
    use mold_inference::chain::ChainTail;
    use std::ops::ControlFlow;
    use std::sync::Arc;

    fn req(format: OutputFormat) -> ChainRequest {
        ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: vec![ChainStage {
                prompt: "stage zero".into(),
                frames: 9,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            }],
            motion_tail_frames: 1,
            width: 64,
            height: 48,
            fps: 8,
            seed: Some(42),
            steps: 2,
            guidance: 1.0,
            strength: 1.0,
            output_format: format,
            placement: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    fn state_with(
        db: Arc<Option<MetadataDb>>,
        handle: crate::chain_job_runner::ChainJobRunnerHandle,
    ) -> AppState {
        let mut state = AppState::for_tests();
        state.metadata_db = db;
        state.chain_jobs = Some(Arc::new(handle));
        state
    }

    fn with_mold_home<R>(home: &std::path::Path, f: impl FnOnce() -> R) -> R {
        let _guard = crate::test_support::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prev = std::env::var_os("MOLD_HOME");
        std::env::set_var("MOLD_HOME", home);
        let out = f();
        match prev {
            Some(value) => std::env::set_var("MOLD_HOME", value),
            None => std::env::remove_var("MOLD_HOME"),
        }
        out
    }

    struct NoopExecutor;

    impl crate::chain_job_runner::StageExecutor for NoopExecutor {
        fn render_stage(
            &self,
            _model: &str,
            _stage_req: &GenerateRequest,
            _carry: Option<&ChainTail>,
            _motion_tail_frames: u32,
            _progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
            _cancelled: &(dyn Fn() -> bool + Send + Sync),
        ) -> anyhow::Result<crate::chain_job_runner::StageRenderOutcome> {
            anyhow::bail!("NoopExecutor should not render during route GC tests")
        }
    }

    #[test]
    fn chain_placement_preview_body_accepts_wrapped_copies_and_legacy_raw_request() {
        let request = req(OutputFormat::Mp4);
        let wrapped: ChainPlacementPreviewBody = serde_json::from_value(serde_json::json!({
            "request": request,
            "copies": 8
        }))
        .unwrap();
        let (wrapped_request, copies) = wrapped.into_parts();
        assert_eq!(wrapped_request.model, "ltx-2-19b-distilled:fp8");
        assert_eq!(copies, 8);

        let legacy: ChainPlacementPreviewBody =
            serde_json::from_value(serde_json::to_value(req(OutputFormat::Mp4)).unwrap()).unwrap();
        let (legacy_request, copies) = legacy.into_parts();
        assert_eq!(legacy_request.model, "ltx-2-19b-distilled:fp8");
        assert_eq!(copies, 1);
    }

    #[tokio::test]
    async fn chain_placement_preview_is_explicitly_non_authoritative_until_exact_stage_plans() {
        let state = state_with(
            Arc::new(Some(MetadataDb::open_in_memory().unwrap())),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let Json(response) = preview_chain_job_placement(
            State(state),
            Json(ChainPlacementPreviewBody::Wrapped(
                ChainPlacementPreviewRequest {
                    request: req(OutputFormat::Mp4),
                    copies: 2,
                },
            )),
        )
        .await;
        assert!(!response.authoritative);
        assert_eq!(response.outcome, "unsupported");
        assert!(response.candidate.is_none());
        assert!(response.stage_candidates.is_empty());
    }

    struct NoopProbe;

    impl crate::chain_job_runner::QueueProbe for NoopProbe {
        fn small_jobs_waiting(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn create_chain_job_persists_public_jobs_as_non_ephemeral() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        let (_status, Json(body)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(
                State(state),
                Json(req(OutputFormat::Mp4)),
            ))
        })
        .unwrap();

        let db_ref = db.as_ref().as_ref().unwrap();
        let row = chain_jobs::get_job(db_ref, &body.job_id).unwrap().unwrap();
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        assert!(!manifest.ephemeral);
    }

    #[tokio::test]
    async fn create_chain_job_rejects_non_mp4_public_jobs() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db,
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        let err = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(
                State(state),
                Json(req(OutputFormat::Apng)),
            ))
        })
        .unwrap_err();

        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn resume_ephemeral_job_returns_409() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let job_id = "01JBR55EPHRESUME";
        with_mold_home(home.path(), || {
            let jobs_root = home.path().join("jobs");
            let db_ref = db.as_ref().as_ref().unwrap();
            let row = crate::chain_job_runner::create_job_with_params(
                db_ref,
                &jobs_root,
                crate::chain_job_runner::CreateJobParams {
                    id: job_id.into(),
                    ephemeral: true,
                    request: req(OutputFormat::Mp4).normalise().unwrap(),
                },
            )
            .unwrap();
            chain_jobs::update_job_state(db_ref, &row.id, ChainJobState::Failed, None, now_ms())
                .unwrap();
        });
        let state = state_with(
            db,
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        let err = with_mold_home(home.path(), || {
            futures::executor::block_on(resume_chain_job(State(state), Path(job_id.to_string())))
        })
        .unwrap_err();

        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn gc_chain_jobs_delegates_to_runner_handle() {
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let home = tempfile::tempdir().unwrap();
        let jobs_root = home.path().join("jobs");
        let deps = crate::chain_job_runner::RunnerDeps {
            db: db.clone(),
            jobs_root,
            executor: Arc::new(NoopExecutor),
            queue_probe: Arc::new(NoopProbe),
            events: Arc::new(crate::chain_job_runner::JobEventBus::new()),
            cancel: Arc::new(crate::chain_job_runner::CancelRegistry::new()),
            job_locks: Arc::new(crate::chain_job_runner::JobMutationLocks::new()),
            claims: Arc::new(crate::chain_job_runner::EphemeralClaims::default()),
            output_dir: None,
            server_events: None,
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
        };
        let state = state_with(db, crate::chain_job_runner::spawn_runner(deps));

        let Json(outcome) = gc_chain_jobs(State(state)).await.unwrap();

        assert_eq!(outcome.swept_ephemeral_jobs, 0);
        assert_eq!(outcome.pruned_artifact_dirs, 0);
    }
}
