use std::convert::Infallible;
use std::path::Path as FsPath;
use std::pin::Pin;

use axum::{
    extract::{Extension, Path, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{sse, IntoResponse, Response, Sse},
    Json,
};
use futures_core::Stream;
use mold_core::chain::{ChainRequest, ChainScript};
use mold_core::chain_job::{
    settled, AmendRequest, AmendResponse, ChainJobDetail, ChainJobEvent, ChainJobListing,
    ChainJobManifest, ChainJobStageDetail, ChainJobState, ChainJobSummary, CreateChainJobResponse,
    GcOutcome, JobDirLayout, RetakeRequest, StageState,
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

#[derive(Debug, Clone, Copy, Default, serde::Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ListChainJobsQuery {
    /// Include internal one-shot long-video shim records. This exists only so
    /// an iPhone can recover an id lost during suspension; sequence surfaces
    /// use the default durable-only listing.
    #[serde(default)]
    include_ephemeral: bool,
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
    crate::routes::ensure_generation_available(&state)?;
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    {
        let config = state.config.read().await;
        crate::routes_chain::require_chain_artifact_activation(&config, &req, None, None)?;
    }
    crate::routes_chain::validate_chain_build_features(&req)?;
    let authority = crate::routes_chain::resolve_chain_model_authority(&state, &req.model).await?;
    crate::routes_chain::validate_and_normalize_chain_family(&authority.config, &mut req)?;
    let req = req
        .normalise()
        .map_err(|e| ApiError::validation(e.to_string()))?;
    if req.output_format != mold_core::OutputFormat::Mp4 {
        return Err(ApiError::validation(
            "durable chain jobs currently require output_format = mp4; legacy /api/generate/chain may request gif/webp/apng via the shim",
        ));
    }
    crate::routes::materialize_chain_camera_controls(&state, &authority.config, &req).await?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let jobs_root = jobs_root()?;
    let model = req.model.clone();
    let stage_count = req.stages.len() as u32;
    crate::chain_job_runner::create_job_with_params(
        db,
        &jobs_root,
        crate::chain_job_runner::CreateJobParams {
            id: job_id.clone(),
            ephemeral: false,
            frozen_model: Some(crate::routes_chain::freeze_chain_model(
                authority, &req.model,
            )?),
            request: req,
        },
    )
    .map_err(|e| ApiError::internal(format!("failed to create chain job: {e:#}")))?;

    handle.kick();
    state
        .events
        .publish(mold_core::ServerEvent::ChainJobQueued {
            id: job_id.clone(),
            model,
            stage_count,
        });
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
        pending_downloads: Vec::new(),
        missing_components: Vec::new(),
    };
    if !(1..=64).contains(&copies) {
        return Json(unavailable("copies must be between 1 and 64".to_string()));
    }
    if let Err(error) = crate::routes_chain::validate_chain_build_features(&req) {
        return Json(unavailable(error.error));
    }
    let base_config = state.config.read().await.clone();
    let validation_config =
        match crate::model_manager::resolve_existing_model_authority(&req.model, &base_config) {
            Ok(Some(authority)) => authority.config,
            Ok(None) => base_config,
            Err(error) => return Json(unavailable(error.error)),
        };
    if let Err(error) =
        crate::routes_chain::validate_and_normalize_chain_family(&validation_config, &mut req)
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
    params(ListChainJobsQuery),
    responses((status = 200, description = "Chain jobs", body = mold_core::chain_job::ChainJobListing))
)]
pub async fn list_chain_jobs(
    State(state): State<AppState>,
    Query(query): Query<ListChainJobsQuery>,
) -> Result<Json<ChainJobListing>, ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let rows = chain_jobs::list_jobs(db)
        .map_err(|e| ApiError::internal(format!("failed to list chain jobs: {e:#}")))?;
    let jobs = rows
        .into_iter()
        .filter_map(|row| {
            let manifest = read_manifest_optional(&row, &root);
            if manifest.as_ref().is_some_and(|manifest| manifest.ephemeral)
                && !query.include_ephemeral
            {
                return None;
            }
            let mut summary = summary_for_row(&row, manifest.as_ref());
            summary.cancelling =
                row.state == ChainJobState::Running && handle.is_cancelling(&row.id);
            Some(summary)
        })
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
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let mut detail = job_detail_for(db, &root, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    detail.summary.cancelling =
        detail.summary.state == ChainJobState::Running && handle.is_cancelling(&id);
    Ok(Json(detail))
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
    let mut detail = job_detail_for(db, &root, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job snapshot: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    detail.summary.cancelling =
        detail.summary.state == ChainJobState::Running && handle.is_cancelling(&id);
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
    crate::routes::ensure_generation_available(&state)?;
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let _guard = handle.lock_job(&id).await;
    let row = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let mut manifest = ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|e| ApiError::internal(format!("failed to read chain manifest: {e:#}")))?;
    if manifest.ephemeral {
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
    require_persisted_chain_model_activation(&state, &row, &manifest).await?;
    crate::chain_job_runner::reset_running_stages_for_retry(
        db,
        &id,
        &row.job_dir,
        &mut manifest,
        now_ms(),
    )
    .map_err(|e| ApiError::internal(format!("failed to reset interrupted chain stage: {e:#}")))?;
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
    state
        .events
        .publish(mold_core::ServerEvent::ChainJobQueued {
            id: id.clone(),
            model: updated.model.clone(),
            stage_count: updated.stage_count,
        });
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
    crate::routes::ensure_generation_available(&state)?;
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let _guard = handle.lock_job(&id).await;
    let row = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|e| ApiError::internal(format!("failed to read chain manifest: {e:#}")))?;
    require_persisted_chain_model_activation(&state, &row, &manifest).await?;
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
    state
        .events
        .publish(mold_core::ServerEvent::ChainJobQueued {
            id: id.clone(),
            model: updated.model.clone(),
            stage_count: updated.stage_count,
        });
    Ok((
        StatusCode::ACCEPTED,
        Json(summary_for_row(
            &updated,
            read_manifest_optional(&updated, &root).as_ref(),
        )),
    ))
}

/// 202 with the preserved-stage count. Accepts the FULL edited stage list
/// (canonical order) plus optional chain-level overlays and requeues from
/// the earliest genuinely-dirty stage; cached raw segments are reused.
/// 409 CHAIN_JOB_RUNNING / CHAIN_JOB_EPHEMERAL; 422 on validation failure;
/// 404 unknown.
#[utoipa::path(
    post,
    path = "/api/chain-jobs/{id}/amend",
    tag = "chain-jobs",
    params(("id" = String, Path, description = "Chain job id")),
    request_body = mold_core::chain_job::AmendRequest,
    responses(
        (status = 202, description = "Amend queued", body = mold_core::chain_job::AmendResponse),
        (status = 409, description = "Job running or ephemeral"),
        (status = 422, description = "Amended request failed validation")
    )
)]
pub async fn amend_chain_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<AmendRequest>,
) -> Result<(StatusCode, Json<AmendResponse>), ApiError> {
    let handle = chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let root = jobs_root()?;
    let _guard = handle.lock_job(&id).await;

    // Run the async create-time family/audio gate on the same candidate
    // `apply_amend` will build (the sync gates — normalise + Mp4-only — run
    // inside `apply_amend`). The gate may normalise chain-level fields (e.g.
    // ltx-video forces motion_tail_frames to 0); carry that back into the
    // amend overlays so the stored request matches what was validated.
    let mut req = req;
    {
        let row = chain_jobs::get_job(db, &id)
            .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
            .ok_or_else(|| not_found(&id))?;
        if row.state == ChainJobState::Running {
            return Err(conflict(CHAIN_JOB_RUNNING, "chain job is running"));
        }
        if read_manifest_optional(&row, &root).is_some_and(|manifest| manifest.ephemeral) {
            return Err(conflict(
                CHAIN_JOB_EPHEMERAL,
                "ephemeral chain jobs are internal to legacy generate/chain shims and cannot be amended",
            ));
        }
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir)
            .map_err(|e| ApiError::internal(format!("failed to read chain manifest: {e:#}")))?;
        require_persisted_chain_model_activation(&state, &row, &manifest).await?;
        let effective = crate::chain_job_runner::effective_request(&manifest)
            .map_err(|e| ApiError::internal(format!("failed to load chain request: {e:#}")))?;
        let mut candidate = crate::chain_job_runner::amend_candidate_request(&effective, &req);
        let motion_tail_before = candidate.motion_tail_frames;
        crate::routes_chain::validate_chain_build_features(&candidate)?;
        let validation_config = if let Some(frozen) = manifest.frozen_model.as_ref() {
            let mut config = state.config.read().await.clone();
            config.install_frozen_model_config(&candidate.model, frozen.config.clone());
            config
        } else {
            crate::routes_chain::resolve_chain_model_authority(&state, &candidate.model)
                .await?
                .config
        };
        crate::routes_chain::validate_and_normalize_chain_family(
            &validation_config,
            &mut candidate,
        )?;
        let candidate = candidate
            .normalise()
            .map_err(|error| ApiError::validation(error.to_string()))?;
        if candidate.output_format != mold_core::OutputFormat::Mp4 {
            return Err(ApiError::validation(
                "durable chain jobs currently require output_format = mp4",
            ));
        }
        crate::routes::materialize_chain_camera_controls(&state, &validation_config, &candidate)
            .await?;
        if candidate.motion_tail_frames != motion_tail_before {
            req.motion_tail_frames = Some(candidate.motion_tail_frames);
        }
    }

    let (updated, preserved_stages) = crate::chain_job_runner::apply_amend(db, &root, &id, &req)
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains(crate::chain_job_runner::CHAIN_JOB_AMEND_INVALID) {
                ApiError::validation(msg.replace(
                    &format!("{}: ", crate::chain_job_runner::CHAIN_JOB_AMEND_INVALID),
                    "",
                ))
            } else if msg.contains("not found") {
                not_found(&id)
            } else if msg.contains(CHAIN_JOB_RUNNING) {
                conflict(CHAIN_JOB_RUNNING, "chain job is running")
            } else if msg.contains(CHAIN_JOB_EPHEMERAL) {
                conflict(
                    CHAIN_JOB_EPHEMERAL,
                    "ephemeral chain jobs cannot be amended",
                )
            } else if msg.contains("not amendable") {
                conflict("CHAIN_JOB_NOT_AMENDABLE", msg)
            } else {
                ApiError::internal(format!("failed to apply amend: {msg}"))
            }
        })?;
    handle.kick();
    state
        .events
        .publish(mold_core::ServerEvent::ChainJobQueued {
            id: id.clone(),
            model: updated.model.clone(),
            stage_count: updated.stage_count,
        });
    let summary = summary_for_row(&updated, read_manifest_optional(&updated, &root).as_ref());
    Ok((
        StatusCode::ACCEPTED,
        Json(AmendResponse {
            summary,
            preserved_stages,
        }),
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
    let mut row = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to load chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let cancelling = if row.state == ChainJobState::Running {
        let requested = handle.request_cancel(&id);
        row = chain_jobs::get_job(db, &id)
            .map_err(|e| ApiError::internal(format!("failed to reload chain job: {e:#}")))?
            .ok_or_else(|| not_found(&id))?;
        requested && row.state == ChainJobState::Running
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
        false
    } else if settled(row.state) {
        // Idempotent: settled jobs have already performed cancel/bus cleanup
        // on the winning settled CAS.
        false
    } else {
        false
    };
    let updated = chain_jobs::get_job(db, &id)
        .map_err(|e| ApiError::internal(format!("failed to reload chain job: {e:#}")))?
        .ok_or_else(|| not_found(&id))?;
    Ok((
        StatusCode::ACCEPTED,
        Json({
            let mut summary =
                summary_for_row(&updated, read_manifest_optional(&updated, &root).as_ref());
            summary.cancelling = cancelling && updated.state == ChainJobState::Running;
            summary
        }),
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

/// Stream a completed stage's standalone MP4 with byte-range support.
#[utoipa::path(
    get,
    path = "/api/chain-jobs/{id}/stages/{idx}/media",
    tag = "chain-jobs",
    params(("id" = String, Path), ("idx" = u32, Path)),
    responses(
        (status = 200, description = "Stage MP4", content_type = "video/mp4"),
        (status = 206, description = "Stage MP4 byte range", content_type = "video/mp4"),
        (status = 404, description = "Job, stage, or stage media missing"),
        (status = 416, description = "Unsatisfiable byte range")
    )
)]
pub async fn chain_job_stage_media(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, idx)): Path<(String, u32)>,
) -> Result<Response, ApiError> {
    chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let row = chain_jobs::get_job(db, &id)
        .map_err(|error| ApiError::internal(format!("failed to load chain job: {error:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|error| ApiError::internal(format!("failed to read chain manifest: {error:#}")))?;
    let status = manifest
        .stage_status
        .get(idx as usize)
        .filter(|status| status.idx == idx)
        .ok_or_else(|| not_found(&id))?;
    let path = stage_segment_path(&row.job_dir, status).ok_or_else(|| {
        ApiError::with_code(
            "stage media not found",
            CHAIN_JOB_NOT_FOUND,
            StatusCode::NOT_FOUND,
        )
    })?;
    crate::routes::serve_media_file(&path, &headers, "video/mp4", "stage media not found").await
}

/// Issue the same short-lived exact-path ticket shape used by gallery media.
#[utoipa::path(
    post,
    path = "/api/chain-jobs/{id}/stages/{idx}/media-token",
    tag = "chain-jobs",
    params(("id" = String, Path), ("idx" = u32, Path)),
    responses(
        (status = 200, description = "Short-lived stage media ticket", body = crate::routes::GalleryMediaTokenResponse),
        (status = 401, description = "API key authentication is required"),
        (status = 404, description = "Job, stage, or stage media missing")
    )
)]
pub(crate) async fn create_chain_job_stage_media_token(
    State(state): State<AppState>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path((id, idx)): Path<(String, u32)>,
) -> Result<impl IntoResponse, ApiError> {
    chain_jobs_handle(&state)?;
    let db = metadata_db(&state)?;
    let row = chain_jobs::get_job(db, &id)
        .map_err(|error| ApiError::internal(format!("failed to load chain job: {error:#}")))?
        .ok_or_else(|| not_found(&id))?;
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|error| ApiError::internal(format!("failed to read chain manifest: {error:#}")))?;
    let status = manifest
        .stage_status
        .get(idx as usize)
        .filter(|status| status.idx == idx)
        .ok_or_else(|| not_found(&id))?;
    if stage_segment_path(&row.job_dir, status).is_none() {
        return Err(ApiError::with_code(
            "stage media not found",
            CHAIN_JOB_NOT_FOUND,
            StatusCode::NOT_FOUND,
        ));
    }

    let path = format!("/api/chain-jobs/{id}/stages/{idx}/media");
    let key_set = auth_state.and_then(|Extension(state)| state);
    let (token, expires_at, auth_required) = match key_set {
        Some(key_set) => {
            let _authenticated = authenticated.ok_or_else(|| {
                ApiError::with_code(
                    "API key authentication is required to issue a media token",
                    "UNAUTHORIZED",
                    StatusCode::UNAUTHORIZED,
                )
            })?;
            let (token, expires_at) = key_set.issue_gallery_media_token(&path);
            (Some(token), Some(expires_at), true)
        }
        None => (None, None, false),
    };
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok((
        headers,
        Json(crate::routes::GalleryMediaTokenResponse {
            token,
            expires_at,
            auth_required,
        }),
    ))
}

/// DB rows + manifest join; media/cache flags come from disk; script = EFFECTIVE.
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
        .map(|stage| {
            let has_media = stage_segment_path(layout.root(), stage).is_some();
            ChainJobStageDetail {
                idx: stage.idx,
                state: stage.state,
                seed: stage.seed,
                frames_emitted: stage.frames_emitted,
                generation_time_ms: stage.generation_time_ms,
                has_preview: layout.preview_path(stage.idx).is_file(),
                has_media,
                cache_ready: stage_cache_ready(&layout, stage, &effective),
                error: stage.error.clone(),
            }
        })
        .collect();
    Ok(Some(ChainJobDetail {
        summary: summary_for_row(&row, Some(&manifest)),
        stages,
        finalizes: manifest.finalizes.clone(),
        retakes: manifest.retakes.clone(),
        amends: manifest.amends.clone(),
        script: ChainScript::from(&effective),
    }))
}

fn stage_segment_path(
    job_dir: &FsPath,
    stage: &mold_core::chain_job::StageStatus,
) -> Option<std::path::PathBuf> {
    if stage.state != StageState::Completed {
        return None;
    }
    let rel = stage.segment.as_ref()?;
    let path = job_dir.join(rel);
    path.is_file().then_some(path)
}

pub(crate) fn stage_cache_ready(
    layout: &JobDirLayout,
    stage: &mold_core::chain_job::StageStatus,
    effective: &ChainRequest,
) -> bool {
    if stage_segment_path(layout.root(), stage).is_none() {
        return false;
    }
    if effective.enable_audio == Some(true)
        && stage
            .audio
            .as_ref()
            .is_none_or(|rel| !layout.root().join(rel).is_file())
    {
        return false;
    }
    if stage.raw_segment && effective.motion_tail_frames > 0 {
        if stage.tail_frames.unwrap_or(0) < effective.motion_tail_frames {
            return false;
        }
        let tail_dir = layout.tail_dir(stage.idx);
        let expected = effective.motion_tail_frames as usize;
        let present = std::fs::read_dir(tail_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
            .count();
        if present < expected {
            return false;
        }
    }
    true
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
    let execution_phase = match row.state {
        ChainJobState::Queued => Some(mold_core::chain_job::ChainExecutionPhase::Queued),
        ChainJobState::Running => manifest.map(|manifest| {
            if manifest
                .stage_status
                .iter()
                .any(|stage| stage.state == mold_core::chain_job::StageState::Running)
            {
                mold_core::chain_job::ChainExecutionPhase::Running
            } else if manifest
                .stage_status
                .iter()
                .all(|stage| stage.state == mold_core::chain_job::StageState::Completed)
            {
                mold_core::chain_job::ChainExecutionPhase::Finalizing
            } else {
                mold_core::chain_job::ChainExecutionPhase::Queued
            }
        }),
        _ => None,
    };
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
        execution_phase,
        cancelling: false,
    }
}

fn read_manifest_optional(row: &ChainJobRow, _root: &FsPath) -> Option<ChainJobManifest> {
    ChainJobManifest::read_from_dir(&row.job_dir).ok()
}

/// Re-check every persisted source of model identity before an old durable
/// job becomes schedulable again. Jobs can outlive both the server process and
/// the policy that admitted them, so the row, request, frozen snapshot,
/// current config, and built-in manifest are all inputs to this fail-closed
/// boundary.
async fn require_persisted_chain_model_activation(
    state: &AppState,
    row: &ChainJobRow,
    manifest: &ChainJobManifest,
) -> Result<(), ApiError> {
    mold_core::require_model_activation(&row.model, None).map_err(ApiError::model_activation)?;
    let effective = crate::chain_job_runner::effective_request(manifest)
        .map_err(|error| ApiError::internal(format!("failed to load chain request: {error:#}")))?;
    let config = state.config.read().await;
    crate::routes_chain::require_chain_artifact_activation(
        &config,
        &effective,
        None,
        manifest.frozen_model.as_ref(),
    )
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
    use mold_core::chain::{ChainStage, LoraSpec, TransitionMode};
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
            height: 64,
            fps: 8,
            seed: Some(42),
            steps: 2,
            guidance: 1.0,
            strength: 1.0,
            output_format: format,
            placement: None,
            original_prompt: None,
            prompt_transform: None,
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

    fn materialize_manifest_companion(models_dir: &std::path::Path, companion: &str) {
        let manifest = mold_core::manifest::find_manifest(companion).unwrap();
        for file in &manifest.files {
            let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"test fixture").unwrap();
            mold_core::download::write_sha256_marker(&path, "test").unwrap();
        }
    }

    fn write_safetensors_with_keys(path: &std::path::Path, keys: &[&str]) {
        use std::io::Write;

        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_json).unwrap();
        file.write_all(&[0u8; 4]).unwrap();
    }

    #[test]
    fn active_summary_phase_tracks_stage_lease_not_parent_actor_state() {
        let request = req(OutputFormat::Mp4).normalise().unwrap();
        let mut manifest = ChainJobManifest::new("phase-job".into(), 1, &request).unwrap();
        let row = ChainJobRow {
            id: "phase-job".into(),
            state: ChainJobState::Running,
            model: request.model.clone(),
            request_json: serde_json::to_string(&request).unwrap(),
            job_dir: std::path::PathBuf::from("/tmp/phase-job"),
            stage_count: 1,
            current_stage: 0,
            error: None,
            created_at_ms: 1,
            updated_at_ms: 1,
            finalized_at_ms: None,
        };

        assert_eq!(
            summary_for_row(&row, Some(&manifest)).execution_phase,
            Some(mold_core::chain_job::ChainExecutionPhase::Queued)
        );
        manifest.stage_status[0].state = StageState::Running;
        assert_eq!(
            summary_for_row(&row, Some(&manifest)).execution_phase,
            Some(mold_core::chain_job::ChainExecutionPhase::Running)
        );
        manifest.stage_status[0].state = StageState::Completed;
        assert_eq!(
            summary_for_row(&row, Some(&manifest)).execution_phase,
            Some(mold_core::chain_job::ChainExecutionPhase::Finalizing)
        );
    }

    fn materialize_ltx23_catalog_model(models_dir: &std::path::Path, model: &str) {
        for companion in ["ltx2-te", "ltx2.3-vae", "ltx2.3-text-projection"] {
            materialize_manifest_companion(models_dir, companion);
        }
        let install_dir = models_dir.join("cv-3143864");
        std::fs::create_dir_all(&install_dir).unwrap();
        let primary = install_dir.join("model.safetensors");
        write_safetensors_with_keys(
            &primary,
            &["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"],
        );
        mold_catalog::sidecar::write_sidecar(
            &install_dir.join(mold_catalog::sidecar::SIDECAR_FILENAME),
            &mold_catalog::sidecar::CatalogSidecar {
                schema: mold_catalog::sidecar::SIDECAR_SCHEMA,
                id: model.to_string(),
                source: "civitai".into(),
                source_id: "3143864".into(),
                name: "LTX 2.3 INT4 ConvRot".into(),
                author: None,
                family: "ltx2".into(),
                family_role: "finetune".into(),
                sub_family: Some("v2.3".into()),
                kind: "checkpoint".into(),
                modality: "video".into(),
                nsfw: None,
                description: None,
                tags: vec![],
                license: None,
                page_url: None,
                thumbnail_url: None,
                size_bytes: Some(primary.metadata().unwrap().len()),
                supported: true,
                trained_words: vec![],
                primary_filename_rel: "model.safetensors".into(),
                written_at: 0,
            },
        )
        .unwrap();
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
        let transformer = home.path().join("transformer.safetensors");
        let vae = home.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        state.config.write().await.models.insert(
            "route-chain-test".into(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("ltx2".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut request = req(OutputFormat::Mp4);
        request.model = "route-chain-test".into();

        let mut events_rx = state.events.subscribe();
        let (_status, Json(body)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(State(state), Json(request)))
        })
        .unwrap();

        let db_ref = db.as_ref().as_ref().unwrap();
        let row = chain_jobs::get_job(db_ref, &body.job_id).unwrap().unwrap();
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        assert!(!manifest.ephemeral);

        // Creation announces the job on the server-wide event stream so the
        // unified activity surface sees it without polling /api/chain-jobs.
        let event = events_rx.try_recv().expect("chain_job_queued published");
        let wire = serde_json::to_value(&event).unwrap();
        assert_eq!(wire["type"], "chain_job_queued");
        assert_eq!(wire["id"], body.job_id.as_str());
        assert_eq!(wire["stage_count"].as_u64(), Some(row.stage_count as u64));
    }

    #[tokio::test]
    async fn create_chain_job_rejects_configured_h3_before_persisting() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        state.config.write().await.models.insert(
            "private-checkpoint".into(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut request = req(OutputFormat::Mp4);
        request.model = "private-checkpoint".into();

        let error = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(State(state), Json(request)))
        })
        .unwrap_err();

        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert!(chain_jobs::list_jobs(db.as_ref().as_ref().unwrap())
            .unwrap()
            .is_empty());
    }

    #[cfg(not(feature = "mp4"))]
    #[tokio::test]
    async fn create_chain_job_keeps_h3_451_ahead_of_mp4_build_validation() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        state.config.write().await.models.insert(
            "private-checkpoint".into(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut request = req(OutputFormat::Mp4);
        request.model = "private-checkpoint".into();
        request.enable_audio = Some(true);

        let error = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(State(state), Json(request)))
        })
        .unwrap_err();

        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert_eq!(
            error.into_response().status(),
            StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS
        );
        assert!(chain_jobs::list_jobs(db.as_ref().as_ref().unwrap())
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn create_chain_job_rejects_every_nested_h3_artifact_before_persisting() {
        for case in [
            "stage-lora",
            "camera-control",
            "spatial-upscaler",
            "model-default-lora",
            "canonicalized-stage-lora",
        ] {
            let home = tempfile::tempdir().unwrap();
            let models_dir = home.path().join("models");
            std::fs::create_dir_all(&models_dir).unwrap();
            let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
            let state = state_with(
                db.clone(),
                crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
            );
            let mut request = freezable_amend_request(&state, home.path()).await;
            state.config.write().await.models_dir = models_dir.display().to_string();
            let gated_path = models_dir.join("MiniMax-H3/adapter.safetensors");

            match case {
                "stage-lora" => request.stages[0].loras.push(LoraSpec {
                    path: gated_path.display().to_string(),
                    scale: 1.0,
                    name: None,
                }),
                "camera-control" => request.stages[0].loras.push(LoraSpec {
                    path: "camera-control:MiniMax-H3".into(),
                    scale: 1.0,
                    name: None,
                }),
                "spatial-upscaler" => {
                    state
                        .config
                        .write()
                        .await
                        .models
                        .get_mut(&request.model)
                        .unwrap()
                        .spatial_upscaler = Some(gated_path.display().to_string());
                }
                "model-default-lora" => {
                    state
                        .config
                        .write()
                        .await
                        .models
                        .get_mut(&request.model)
                        .unwrap()
                        .lora = Some(gated_path.display().to_string());
                }
                "canonicalized-stage-lora" => {
                    std::fs::create_dir_all(gated_path.parent().unwrap()).unwrap();
                    std::fs::write(&gated_path, b"restricted adapter fixture").unwrap();
                    let alias_dir = models_dir.join("ordinary-adapters");
                    std::fs::create_dir_all(&alias_dir).unwrap();
                    let alias = alias_dir.join("style.safetensors");
                    std::os::unix::fs::symlink(&gated_path, &alias).unwrap();
                    request.stages[0].loras.push(LoraSpec {
                        path: alias.display().to_string(),
                        scale: 1.0,
                        name: None,
                    });
                }
                _ => unreachable!(),
            }

            let error = with_mold_home(home.path(), || {
                futures::executor::block_on(create_chain_job(State(state.clone()), Json(request)))
            })
            .unwrap_err();

            assert_eq!(
                error.code,
                mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                "{case}"
            );
            assert_eq!(
                error.into_response().status(),
                StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
                "{case}"
            );
            assert!(
                chain_jobs::list_jobs(db.as_ref().as_ref().unwrap())
                    .unwrap()
                    .is_empty(),
                "{case} must reject before any durable row is created"
            );
            let downloads = state.downloads.listing().await;
            assert!(downloads.active.is_none(), "{case}");
            assert!(downloads.queued.is_empty(), "{case}");
        }
    }

    #[tokio::test]
    async fn web_sequence_freezes_installed_ltx23_catalog_model_without_mutating_live_config() {
        const MODEL: &str = "cv:3143864";

        let home = tempfile::tempdir().unwrap();
        let models_dir = home.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        materialize_ltx23_catalog_model(&models_dir, MODEL);
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        state.config.write().await.models_dir = models_dir.display().to_string();
        let mut request = req(OutputFormat::Mp4);
        request.model = MODEL.into();
        request.stages.push(ChainStage {
            prompt: "stage one".into(),
            frames: 9,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        });

        let (status, Json(created)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(State(state.clone()), Json(request)))
        })
        .unwrap();

        assert_eq!(status, StatusCode::ACCEPTED);
        assert!(
            !state.config.read().await.models.contains_key(MODEL),
            "durable admission must not install a runtime catalog overlay into AppState"
        );
        let row = chain_jobs::get_job(db.as_ref().as_ref().unwrap(), &created.job_id)
            .unwrap()
            .unwrap();
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        let effective = crate::chain_job_runner::effective_request(&manifest).unwrap();
        assert_eq!(effective.model, MODEL);
        assert_eq!(effective.stages.len(), 2);
        let frozen = manifest.frozen_model.expect("catalog model frozen");
        assert!(frozen.runtime_model_id.starts_with("mold-frozen-chain:"));
        assert_eq!(frozen.config.family.as_deref(), Some("ltx2"));
        let paths = mold_core::ModelPaths::resolve_from_model_config_exact(&frozen.config)
            .expect("frozen paths remain exactly resolvable");
        assert_eq!(
            paths.transformer,
            models_dir
                .join("cv-3143864/model.safetensors")
                .canonicalize()
                .unwrap()
        );
        assert_ne!(paths.vae, paths.transformer);
        assert!(
            paths.text_encoder_files.len() > 1,
            "Gemma weights and LTX-2.3 text projection must both be frozen"
        );
        for path in frozen.config.all_file_paths() {
            let path = std::path::Path::new(&path);
            assert!(path.is_absolute());
            assert!(path.is_file(), "missing frozen path {}", path.display());
        }
    }

    #[tokio::test]
    async fn create_chain_job_fails_closed_when_model_freeze_is_unresolvable() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        state.config.write().await.models.insert(
            "unfreezable-chain".into(),
            mold_core::ModelConfig {
                transformer: Some(
                    home.path()
                        .join("missing-transformer")
                        .display()
                        .to_string(),
                ),
                vae: Some(home.path().join("missing-vae").display().to_string()),
                family: Some("ltx2".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut request = req(OutputFormat::Mp4);
        request.model = "unfreezable-chain".into();

        let error = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(State(state), Json(request)))
        })
        .unwrap_err();
        assert!(error
            .error
            .starts_with("cannot freeze concrete chain model companions for 'unfreezable-chain':"));
        assert_eq!(
            error.into_response().status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert!(chain_jobs::list_jobs(db.as_ref().as_ref().unwrap())
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn chain_model_freeze_reloads_paths_added_after_server_start() {
        let home = tempfile::tempdir().unwrap();
        let transformer = home.path().join("transformer.safetensors");
        let vae = home.path().join("vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();

        let state = state_with(
            Arc::new(Some(MetadataDb::open_in_memory().unwrap())),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        assert!(!state
            .config
            .read()
            .await
            .models
            .contains_key("late-installed-chain"));

        let authority = with_mold_home(home.path(), || {
            let mut disk_config = mold_core::Config::default();
            disk_config.models.insert(
                "late-installed-chain".into(),
                mold_core::ModelConfig {
                    transformer: Some(transformer.display().to_string()),
                    vae: Some(vae.display().to_string()),
                    family: Some("ltx2".into()),
                    ..mold_core::ModelConfig::default()
                },
            );
            disk_config.save().unwrap();
            futures::executor::block_on(crate::routes_chain::resolve_chain_model_authority(
                &state,
                "late-installed-chain",
            ))
        })
        .unwrap();

        assert_eq!(authority.paths.transformer, transformer);
        assert_eq!(authority.paths.vae, vae);
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
    async fn stage_media_streams_ranges_and_detail_reflects_disk_cache() {
        use axum::body::to_bytes;

        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let job_id = "stage-media-route-test";
        with_mold_home(home.path(), || {
            crate::chain_job_runner::create_job_with_params(
                db.as_ref().as_ref().unwrap(),
                &home.path().join("jobs"),
                crate::chain_job_runner::CreateJobParams {
                    id: job_id.into(),
                    ephemeral: false,
                    frozen_model: None,
                    request: req(OutputFormat::Mp4).normalise().unwrap(),
                },
            )
            .unwrap();
        });
        let db_ref = db.as_ref().as_ref().unwrap();
        let row = chain_jobs::get_job(db_ref, job_id).unwrap().unwrap();
        let layout = JobDirLayout::new(row.job_dir.clone());
        std::fs::create_dir_all(layout.tail_dir(0)).unwrap();
        std::fs::write(layout.segment_path(0), b"0123456789").unwrap();
        std::fs::write(layout.tail_dir(0).join("000000.png"), b"tail").unwrap();
        let mut manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        manifest.stage_status[0].state = StageState::Completed;
        manifest.stage_status[0].segment = Some(layout.segment_rel(0));
        manifest.stage_status[0].tail_frames = Some(1);
        manifest.stage_status[0].raw_segment = true;
        manifest.write_atomic(&row.job_dir).unwrap();

        let detail = job_detail_for(db_ref, home.path(), job_id)
            .unwrap()
            .unwrap();
        assert!(detail.stages[0].has_media);
        assert!(detail.stages[0].cache_ready);

        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=2-5"));
        let response = chain_job_stage_media(State(state), headers, Path((job_id.into(), 0)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            response.headers()[header::CONTENT_RANGE],
            HeaderValue::from_static("bytes 2-5/10")
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            "2345"
        );

        std::fs::remove_file(layout.tail_dir(0).join("000000.png")).unwrap();
        let detail = job_detail_for(db_ref, home.path(), &row.id)
            .unwrap()
            .unwrap();
        assert!(detail.stages[0].has_media);
        assert!(!detail.stages[0].cache_ready);
    }

    fn amend_body(stages: Vec<ChainStage>) -> AmendRequest {
        AmendRequest {
            stages,
            motion_tail_frames: None,
            fps: None,
            seed: None,
            steps: None,
            guidance: None,
            strength: None,
            enable_audio: None,
        }
    }

    async fn freezable_amend_request(state: &AppState, home: &std::path::Path) -> ChainRequest {
        let transformer = home.join("amend-transformer.safetensors");
        let vae = home.join("amend-vae.safetensors");
        std::fs::write(&transformer, b"transformer").unwrap();
        std::fs::write(&vae, b"vae").unwrap();
        state.config.write().await.models.insert(
            "route-chain-amend-test".into(),
            mold_core::ModelConfig {
                transformer: Some(transformer.display().to_string()),
                vae: Some(vae.display().to_string()),
                family: Some("ltx2".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut request = req(OutputFormat::Mp4);
        request.model = "route-chain-amend-test".into();
        request
    }

    #[tokio::test]
    async fn amend_chain_job_returns_202_with_preserved_stages_and_requeues() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let request = freezable_amend_request(&state, home.path()).await;

        let (_status, Json(created)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(
                State(state.clone()),
                Json(request.clone()),
            ))
        })
        .unwrap();

        let mut events_rx = state.events.subscribe();
        let mut stages = request.stages;
        stages.push(ChainStage {
            prompt: "appended clip".into(),
            frames: 9,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        });
        let (status, Json(body)) = with_mold_home(home.path(), || {
            futures::executor::block_on(amend_chain_job(
                State(state.clone()),
                Path(created.job_id.clone()),
                Json(amend_body(stages)),
            ))
        })
        .unwrap();

        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(
            body.preserved_stages, 0,
            "no leading completed stages on a queued job"
        );
        assert_eq!(body.summary.state, ChainJobState::Queued);
        assert_eq!(body.summary.stage_count, 2);

        let event = events_rx.try_recv().expect("chain_job_queued published");
        let wire = serde_json::to_value(&event).unwrap();
        assert_eq!(wire["type"], "chain_job_queued");
        assert_eq!(wire["id"], created.job_id.as_str());
        assert_eq!(wire["stage_count"].as_u64(), Some(2));
    }

    #[tokio::test]
    async fn amend_rejects_new_h3_stage_lora_without_mutating_or_requeueing() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let request = freezable_amend_request(&state, home.path()).await;
        let (_status, Json(created)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(
                State(state.clone()),
                Json(request.clone()),
            ))
        })
        .unwrap();
        let db_ref = db.as_ref().as_ref().unwrap();
        let row = chain_jobs::get_job(db_ref, &created.job_id)
            .unwrap()
            .unwrap();
        let before_manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        let before_stages = chain_jobs::stages_for_job(db_ref, &created.job_id).unwrap();

        let mut stages = request.stages;
        stages[0].loras.push(LoraSpec {
            path: "/models/MiniMax-H3/amend-bypass.safetensors".into(),
            scale: 1.0,
            name: None,
        });
        let error = with_mold_home(home.path(), || {
            futures::executor::block_on(amend_chain_job(
                State(state.clone()),
                Path(created.job_id.clone()),
                Json(amend_body(stages)),
            ))
        })
        .unwrap_err();

        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert_eq!(
            error.into_response().status(),
            StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS
        );
        assert_eq!(
            chain_jobs::get_job(db_ref, &created.job_id)
                .unwrap()
                .unwrap(),
            row
        );
        assert_eq!(
            ChainJobManifest::read_from_dir(&row.job_dir).unwrap(),
            before_manifest
        );
        assert_eq!(
            chain_jobs::stages_for_job(db_ref, &created.job_id).unwrap(),
            before_stages
        );
    }

    #[tokio::test]
    async fn amend_running_job_returns_409() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let request = freezable_amend_request(&state, home.path()).await;
        let (_status, Json(created)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(
                State(state.clone()),
                Json(request.clone()),
            ))
        })
        .unwrap();
        let db_ref = db.as_ref().as_ref().unwrap();
        assert!(chain_jobs::claim_job(db_ref, &created.job_id).unwrap());

        let err = with_mold_home(home.path(), || {
            futures::executor::block_on(amend_chain_job(
                State(state.clone()),
                Path(created.job_id.clone()),
                Json(amend_body(request.stages)),
            ))
        })
        .unwrap_err();

        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn amend_ephemeral_job_returns_409() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let job_id = "01JBR55EPHAMEND";
        with_mold_home(home.path(), || {
            let jobs_root = home.path().join("jobs");
            let db_ref = db.as_ref().as_ref().unwrap();
            crate::chain_job_runner::create_job_with_params(
                db_ref,
                &jobs_root,
                crate::chain_job_runner::CreateJobParams {
                    id: job_id.into(),
                    ephemeral: true,
                    frozen_model: None,
                    request: req(OutputFormat::Mp4).normalise().unwrap(),
                },
            )
            .unwrap();
        });
        let state = state_with(
            db,
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        let err = with_mold_home(home.path(), || {
            futures::executor::block_on(amend_chain_job(
                State(state.clone()),
                Path(job_id.to_string()),
                Json(amend_body(req(OutputFormat::Mp4).stages)),
            ))
        })
        .unwrap_err();

        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn amend_with_invalid_stages_returns_422() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db,
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let request = freezable_amend_request(&state, home.path()).await;
        let (_status, Json(created)) = with_mold_home(home.path(), || {
            futures::executor::block_on(create_chain_job(
                State(state.clone()),
                Json(request.clone()),
            ))
        })
        .unwrap();

        let mut stages = request.stages;
        stages[0].frames = 10; // not 8k+1
        let err = with_mold_home(home.path(), || {
            futures::executor::block_on(amend_chain_job(
                State(state.clone()),
                Path(created.job_id.clone()),
                Json(amend_body(stages)),
            ))
        })
        .unwrap_err();

        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn resume_rewinds_stale_cancelled_and_interrupted_stage_leases() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        for (job_id, resumable_state) in [
            ("01JBR55CANCELRESUME", ChainJobState::Cancelled),
            ("01JBR55INTRRESUME", ChainJobState::Interrupted),
        ] {
            let row = with_mold_home(home.path(), || {
                let db_ref = db.as_ref().as_ref().unwrap();
                let row = crate::chain_job_runner::create_job_with_params(
                    db_ref,
                    &home.path().join("jobs"),
                    crate::chain_job_runner::CreateJobParams {
                        id: job_id.into(),
                        ephemeral: false,
                        frozen_model: None,
                        request: req(OutputFormat::Mp4).normalise().unwrap(),
                    },
                )
                .unwrap();
                let mut manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
                manifest.stage_status[0].state = StageState::Running;
                manifest.write_atomic(&row.job_dir).unwrap();
                chain_jobs::upsert_stage(
                    db_ref,
                    &mold_db::chain_jobs::ChainJobStageRow {
                        job_id: job_id.into(),
                        stage_idx: 0,
                        state: StageState::Running,
                        seed: manifest.stage_status[0].seed,
                        frames_emitted: None,
                        generation_time_ms: None,
                        segment_rel_path: None,
                        error: None,
                        updated_at_ms: now_ms(),
                    },
                )
                .unwrap();
                chain_jobs::update_job_state(db_ref, job_id, resumable_state, None, now_ms())
                    .unwrap();
                row
            });

            let (status, Json(summary)) = with_mold_home(home.path(), || {
                futures::executor::block_on(resume_chain_job(
                    State(state.clone()),
                    Path(job_id.to_string()),
                ))
            })
            .unwrap();

            assert_eq!(status, StatusCode::ACCEPTED);
            assert_eq!(summary.state, ChainJobState::Queued);
            assert_eq!(
                summary.execution_phase,
                Some(mold_core::chain_job::ChainExecutionPhase::Queued)
            );
            let manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
            assert_eq!(manifest.stage_status[0].state, StageState::Pending);
            let stages = chain_jobs::stages_for_job(db.as_ref().as_ref().unwrap(), job_id).unwrap();
            assert_eq!(stages[0].state, StageState::Pending);
        }
    }

    #[tokio::test]
    async fn resume_rejects_persisted_configured_h3_without_mutating_the_job() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        state.config.write().await.models.insert(
            "legacy-private-chain".into(),
            mold_core::ModelConfig {
                family: Some("minimax-h3".into()),
                ..mold_core::ModelConfig::default()
            },
        );
        let job_id = "01JBR55H3CFGRESUME";
        let row = with_mold_home(home.path(), || {
            let mut request = req(OutputFormat::Mp4).normalise().unwrap();
            request.model = "legacy-private-chain".into();
            let row = crate::chain_job_runner::create_job_with_params(
                db.as_ref().as_ref().unwrap(),
                &home.path().join("jobs"),
                crate::chain_job_runner::CreateJobParams {
                    id: job_id.into(),
                    ephemeral: false,
                    frozen_model: None,
                    request,
                },
            )
            .unwrap();
            chain_jobs::update_job_state(
                db.as_ref().as_ref().unwrap(),
                job_id,
                ChainJobState::Failed,
                Some("old failure"),
                now_ms(),
            )
            .unwrap();
            row
        });
        let db_ref = db.as_ref().as_ref().unwrap();
        let before_row = chain_jobs::get_job(db_ref, job_id).unwrap().unwrap();
        let before_manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        let before_stages = chain_jobs::stages_for_job(db_ref, job_id).unwrap();

        let error = with_mold_home(home.path(), || {
            futures::executor::block_on(resume_chain_job(State(state), Path(job_id.to_string())))
        })
        .unwrap_err();

        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert_eq!(
            error.into_response().status(),
            StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS
        );
        assert_eq!(
            chain_jobs::get_job(db_ref, job_id).unwrap().unwrap(),
            before_row
        );
        assert_eq!(
            ChainJobManifest::read_from_dir(&row.job_dir).unwrap(),
            before_manifest
        );
        assert_eq!(
            chain_jobs::stages_for_job(db_ref, job_id).unwrap(),
            before_stages
        );
    }

    #[tokio::test]
    async fn resume_rejects_nested_persisted_h3_artifacts_without_mutation() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        for case in [
            "request-stage-lora",
            "frozen-default-lora",
            "frozen-upscaler",
            "live-config-default-lora",
        ] {
            let model = format!("legacy-neutral-{case}");
            let job_id = format!("01JBR55H3{}", case.replace('-', "").to_ascii_uppercase());
            let mut request = req(OutputFormat::Mp4).normalise().unwrap();
            request.model = model.clone();
            let mut frozen_config = mold_core::ModelConfig {
                family: Some("ltx2".into()),
                ..mold_core::ModelConfig::default()
            };
            match case {
                "request-stage-lora" => request.stages[0].loras.push(LoraSpec {
                    path: "/models/MiniMax-H3/persisted-stage.safetensors".into(),
                    scale: 1.0,
                    name: None,
                }),
                "frozen-default-lora" => {
                    frozen_config.lora =
                        Some("/models/MiniMax-H3/frozen-default.safetensors".into());
                }
                "frozen-upscaler" => {
                    frozen_config.spatial_upscaler =
                        Some("/models/MiniMax-H3/frozen-upscaler.safetensors".into());
                }
                "live-config-default-lora" => {
                    state.config.write().await.models.insert(
                        model.clone(),
                        mold_core::ModelConfig {
                            family: Some("ltx2".into()),
                            lora: Some("/models/MiniMax-H3/live-default.safetensors".into()),
                            ..mold_core::ModelConfig::default()
                        },
                    );
                }
                _ => unreachable!(),
            }
            let frozen_model = if matches!(case, "request-stage-lora" | "live-config-default-lora")
            {
                None
            } else {
                Some(mold_core::chain_job::FrozenChainModel {
                    runtime_model_id: format!("mold-frozen-chain:{model}"),
                    config: frozen_config,
                    model_fingerprint: format!("fixture-{case}"),
                })
            };
            let row = with_mold_home(home.path(), || {
                let row = crate::chain_job_runner::create_job_with_params(
                    db.as_ref().as_ref().unwrap(),
                    &home.path().join("jobs"),
                    crate::chain_job_runner::CreateJobParams {
                        id: job_id.clone(),
                        ephemeral: false,
                        frozen_model,
                        request,
                    },
                )
                .unwrap();
                chain_jobs::update_job_state(
                    db.as_ref().as_ref().unwrap(),
                    &job_id,
                    ChainJobState::Failed,
                    Some("legacy failure"),
                    now_ms(),
                )
                .unwrap();
                row
            });
            let db_ref = db.as_ref().as_ref().unwrap();
            let before_row = chain_jobs::get_job(db_ref, &job_id).unwrap().unwrap();
            let before_manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
            let before_stages = chain_jobs::stages_for_job(db_ref, &job_id).unwrap();

            let error = with_mold_home(home.path(), || {
                futures::executor::block_on(resume_chain_job(
                    State(state.clone()),
                    Path(job_id.clone()),
                ))
            })
            .unwrap_err();

            assert_eq!(
                error.code,
                mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                "{case}"
            );
            assert_eq!(
                error.into_response().status(),
                StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
                "{case}"
            );
            assert_eq!(
                chain_jobs::get_job(db_ref, &job_id).unwrap().unwrap(),
                before_row,
                "{case}"
            );
            assert_eq!(
                ChainJobManifest::read_from_dir(&row.job_dir).unwrap(),
                before_manifest,
                "{case}"
            );
            assert_eq!(
                chain_jobs::stages_for_job(db_ref, &job_id).unwrap(),
                before_stages,
                "{case}"
            );
        }
    }

    #[tokio::test]
    async fn retake_rejects_persisted_h3_identity_and_frozen_family_without_mutation() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );

        for (job_id, model, frozen_family) in [
            ("01JBR55H3RAWRETAKE", "MiniMaxAI/MiniMax-H3", None),
            (
                "01JBR55H3FROZENRETAKE",
                "legacy-private-chain",
                Some("minimax-h3"),
            ),
        ] {
            let row = with_mold_home(home.path(), || {
                let mut request = req(OutputFormat::Mp4).normalise().unwrap();
                request.model = model.into();
                let frozen_model =
                    frozen_family.map(|family| mold_core::chain_job::FrozenChainModel {
                        runtime_model_id: "mold-frozen-chain:legacy-private".into(),
                        config: mold_core::ModelConfig {
                            family: Some(family.into()),
                            ..mold_core::ModelConfig::default()
                        },
                        model_fingerprint: "legacy-fixture".into(),
                    });
                let row = crate::chain_job_runner::create_job_with_params(
                    db.as_ref().as_ref().unwrap(),
                    &home.path().join("jobs"),
                    crate::chain_job_runner::CreateJobParams {
                        id: job_id.into(),
                        ephemeral: false,
                        frozen_model,
                        request,
                    },
                )
                .unwrap();
                chain_jobs::update_job_state(
                    db.as_ref().as_ref().unwrap(),
                    job_id,
                    ChainJobState::Completed,
                    None,
                    now_ms(),
                )
                .unwrap();
                row
            });
            let db_ref = db.as_ref().as_ref().unwrap();
            let before_row = chain_jobs::get_job(db_ref, job_id).unwrap().unwrap();
            let before_manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
            let before_stages = chain_jobs::stages_for_job(db_ref, job_id).unwrap();

            let error = with_mold_home(home.path(), || {
                futures::executor::block_on(retake_chain_job(
                    State(state.clone()),
                    Path(job_id.to_string()),
                    Json(RetakeRequest {
                        stage_idx: 0,
                        mode: mold_core::chain_job::RetakeMode::Cascade,
                        seed_offset: Some(7),
                        prompt: Some("new prompt".into()),
                    }),
                ))
            })
            .unwrap_err();

            assert_eq!(
                error.code,
                mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                "{job_id}"
            );
            assert_eq!(
                error.into_response().status(),
                StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
                "{job_id}"
            );
            assert_eq!(
                chain_jobs::get_job(db_ref, job_id).unwrap().unwrap(),
                before_row,
                "{job_id}"
            );
            assert_eq!(
                ChainJobManifest::read_from_dir(&row.job_dir).unwrap(),
                before_manifest,
                "{job_id}"
            );
            assert_eq!(
                chain_jobs::stages_for_job(db_ref, job_id).unwrap(),
                before_stages,
                "{job_id}"
            );
        }
    }

    #[tokio::test]
    async fn retake_still_requeues_an_ordinary_persisted_job() {
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let state = state_with(
            db.clone(),
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        );
        let job_id = "01JBR55ORDRETAKE";
        let row = with_mold_home(home.path(), || {
            let row = crate::chain_job_runner::create_job_with_params(
                db.as_ref().as_ref().unwrap(),
                &home.path().join("jobs"),
                crate::chain_job_runner::CreateJobParams {
                    id: job_id.into(),
                    ephemeral: false,
                    frozen_model: None,
                    request: req(OutputFormat::Mp4).normalise().unwrap(),
                },
            )
            .unwrap();
            chain_jobs::update_job_state(
                db.as_ref().as_ref().unwrap(),
                job_id,
                ChainJobState::Completed,
                None,
                now_ms(),
            )
            .unwrap();
            row
        });

        let (status, Json(summary)) = with_mold_home(home.path(), || {
            futures::executor::block_on(retake_chain_job(
                State(state),
                Path(job_id.to_string()),
                Json(RetakeRequest {
                    stage_idx: 0,
                    mode: mold_core::chain_job::RetakeMode::Cascade,
                    seed_offset: Some(9),
                    prompt: Some("ordinary retake".into()),
                }),
            ))
        })
        .unwrap();

        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(summary.state, ChainJobState::Queued);
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        assert_eq!(manifest.retakes.len(), 1);
        assert_eq!(
            manifest.retakes[0].new_prompt.as_deref(),
            Some("ordinary retake")
        );
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
                    frozen_model: None,
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
            dispatch_mode: crate::dispatch_mode::DispatchMode::Legacy,
            pause: None,
        };
        let state = state_with(db, crate::chain_job_runner::spawn_runner(deps));

        let Json(outcome) = gc_chain_jobs(State(state)).await.unwrap();

        assert_eq!(outcome.swept_ephemeral_jobs, 0);
        assert_eq!(outcome.pruned_artifact_dirs, 0);
    }
}
