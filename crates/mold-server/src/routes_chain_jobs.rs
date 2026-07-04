#![allow(dead_code)] // Phase 3 placeholder — removed in Phase 6

use std::convert::Infallible;
use std::path::Path as FsPath;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{sse, Response, Sse},
    Json,
};
use mold_core::chain::ChainRequest;
use mold_core::chain_job::{
    ChainJobDetail, ChainJobListing, ChainJobSummary, CreateChainJobResponse, RetakeRequest,
};
use mold_db::MetadataDb;

use crate::routes::ApiError;
use crate::state::AppState;

// Phase 3 placeholder — TODO(BR55 Phase 6): redefine as the real
// snapshot-then-live event stream type (P2 contract: Sse<impl Stream>).
type ChainJobSseStream = futures::stream::Empty<Result<sse::Event, Infallible>>;

/// 202. Validates ChainRequest::normalise + chain_limits caps + family
/// chain-capability + audio gate. 503 CHAIN_JOBS_UNAVAILABLE when DB
/// disabled (state.chain_jobs None). Creates job dir + manifest + rows,
/// kicks runner.
#[allow(clippy::doc_lazy_continuation)] // Phase 3 placeholder — removed in Phase 6
pub async fn create_chain_job(
    State(_state): State<AppState>,
    Json(_req): Json<ChainRequest>,
) -> Result<(StatusCode, Json<CreateChainJobResponse>), ApiError> {
    // TODO(BR55 Phase 6): validate and persist a new chain job before kicking the runner.
    todo!()
}

pub async fn list_chain_jobs(
    State(_state): State<AppState>,
) -> Result<Json<ChainJobListing>, ApiError> {
    // TODO(BR55 Phase 6): list chain job summaries from the metadata database.
    todo!()
}

/// 404 unknown
pub async fn get_chain_job(
    State(_state): State<AppState>,
    Path(_id): Path<String>,
) -> Result<Json<ChainJobDetail>, ApiError> {
    // TODO(BR55 Phase 6): load a chain job detail from DB rows and manifest state.
    todo!()
}

/// subscribe() FIRST, then synthesize Snapshot from DB+manifest, then live.
pub async fn chain_job_events(
    State(_state): State<AppState>,
    Path(_id): Path<String>,
) -> Result<Sse<ChainJobSseStream>, ApiError> {
    // TODO(BR55 Phase 6): attach to the job event bus before streaming snapshot and live events.
    todo!()
}

/// 202; 409 CHAIN_JOB_RUNNING when state==Running; 404; no-op guard for Completed → 409 too (nothing to resume) — final code list in P6 docs.
pub async fn resume_chain_job(
    State(_state): State<AppState>,
    Path(_id): Path<String>,
) -> Result<(StatusCode, Json<ChainJobSummary>), ApiError> {
    // TODO(BR55 Phase 6): transition an interrupted or failed chain job back to queued and kick the runner.
    todo!()
}

/// 202; 409 CHAIN_JOB_RUNNING; 409 RETAKE_SPLICE_REQUIRES_CUT_OR_FADE; 404; 400 stage_idx OOB.
/// Performs the runner kick after apply_retake returns (see apply_retake doc).
/// Code casing: SCREAMING_SNAKE governs per the P1 rev1 approved decision
/// (codebase wire convention over the spec's lowercase prose).
pub async fn retake_chain_job(
    State(_state): State<AppState>,
    Path(_id): Path<String>,
    Json(_req): Json<RetakeRequest>,
) -> Result<(StatusCode, Json<ChainJobSummary>), ApiError> {
    // TODO(BR55 Phase 6): apply a retake amendment, return the updated summary, and kick the runner.
    todo!()
}

/// 202, idempotent
pub async fn cancel_chain_job(
    State(_state): State<AppState>,
    Path(_id): Path<String>,
) -> Result<(StatusCode, Json<ChainJobSummary>), ApiError> {
    // TODO(BR55 Phase 6): mark cancellation requested and return the current job summary.
    todo!()
}

/// 204; 409 CHAIN_JOB_RUNNING while running; removes job dir + row (cascade)
pub async fn delete_chain_job(
    State(_state): State<AppState>,
    Path(_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    // TODO(BR55 Phase 6): remove a non-running chain job row and its artifact directory.
    todo!()
}

/// image/jpeg; 404 job/stage/preview missing
pub async fn chain_job_stage_preview(
    State(_state): State<AppState>,
    Path((_id, _idx)): Path<(String, u32)>,
) -> Result<Response, ApiError> {
    // TODO(BR55 Phase 6): serve the persisted stage preview JPEG after DB and file existence checks.
    todo!()
}

/// DB rows + manifest join; has_preview from disk; script = EFFECTIVE
/// (original request with retake amendments applied).
fn job_detail_for(
    _db: &MetadataDb,
    _jobs_root: &FsPath,
    _id: &str,
) -> anyhow::Result<Option<ChainJobDetail>> {
    // TODO(BR55 Phase 6): join DB rows with manifest data into the effective job detail response.
    todo!()
}
