//! Owner-fenced HTTP lifecycle for durable duration parents.

use crate::{routes::ApiError, state::AppState};
use anyhow::Context;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    Extension, Json,
};
use mold_core::video_job::{VideoJob, VideoJobPlan, VideoJobRequest};
use mold_db::video_jobs as db;
use std::sync::{Arc, OnceLock};

/// MP4 builds expose the literal protocol; capabilities also require runtime readiness.
pub(crate) fn router() -> axum::Router<AppState> {
    #[cfg(feature = "mp4")]
    {
        use axum::routing::{get as get_route, post};
        axum::Router::new()
            .route(
                "/api/video-jobs/validate",
                post(validate).layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024)),
            )
            .route(
                "/api/video-jobs",
                post(crate::video_job_create::create)
                    .get(list)
                    .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024)),
            )
            .route("/api/video-jobs/:id", get_route(get).delete(remove))
            .route("/api/video-jobs/:id/events", get_route(events))
            .route("/api/video-jobs/:id/cancel", post(cancel))
            .route("/api/video-jobs/:id/resume", post(resume))
            .route(
                "/api/video-jobs/:id/restart-options",
                get_route(crate::video_job_restart_routes::options),
            )
            .route(
                "/api/video-jobs/:id/restart-export",
                post(crate::video_job_restart_routes::export),
            )
            .route(
                "/api/video-jobs/:id/restart-source",
                get_route(crate::video_job_restart_routes::source),
            )
    }
    #[cfg(not(feature = "mp4"))]
    {
        axum::Router::new()
    }
}

pub(crate) fn require_instance(instance: &str, headers: &HeaderMap) -> Result<(), ApiError> {
    if headers
        .get("x-mold-destination-instance")
        .and_then(|value| value.to_str().ok())
        != Some(instance)
    {
        return Err(ApiError::with_code(
            "Destination installation changed. Refresh before continuing.",
            "VIDEO_DESTINATION_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    Ok(())
}

pub(crate) fn apply_headers(
    headers: &HeaderMap,
    request: &mut VideoJobRequest,
) -> Result<(), ApiError> {
    if let Some(value) = headers.get("x-mold-video-validation-revision") {
        let revision = value
            .to_str()
            .ok()
            .filter(|value| !value.is_empty() && value.len() <= 128)
            .ok_or_else(|| ApiError::validation("Invalid video validation revision"))?;
        if request
            .validation_revision
            .as_ref()
            .is_some_and(|body| body != revision)
        {
            return Err(ApiError::validation(
                "Body and header validation revisions differ",
            ));
        }
        request.validation_revision = Some(revision.to_owned());
    }
    if let Some(value) = headers.get("x-mold-video-source-upload") {
        if value.len() > 1024 {
            return Err(ApiError::validation(
                "Video source upload header exceeds its budget",
            ));
        }
        let uploaded: mold_core::video_job::VideoSourceUploadRef =
            serde_json::from_slice(value.as_bytes())
                .map_err(|_| ApiError::validation("Invalid video source upload header"))?;
        let source = request
            .source
            .as_mut()
            .ok_or_else(|| ApiError::validation("A source upload requires source provenance"))?;
        if source
            .retained_upload
            .as_ref()
            .is_some_and(|body| body != &uploaded)
        {
            return Err(ApiError::validation(
                "Body and header source uploads differ",
            ));
        }
        source.retained_upload = Some(uploaded);
    }
    request.validate_shape().map_err(ApiError::validation)
}

pub(crate) fn database(state: &AppState) -> Result<&mold_db::MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "Durable video storage is unavailable",
            "VIDEO_JOBS_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

pub(crate) fn canonical_id(value: &str) -> Result<String, ApiError> {
    let id = uuid::Uuid::parse_str(value)
        .map_err(|_| ApiError::validation("Video operation ID must be a UUID"))?;
    if id.to_string() != value {
        return Err(ApiError::validation(
            "Video operation ID must be a canonical UUID",
        ));
    }
    Ok(value.to_owned())
}

fn mutation_mutex(id: &str) -> Result<Arc<tokio::sync::Mutex<()>>, ApiError> {
    let canonical = canonical_id(id)?;
    let uuid = uuid::Uuid::parse_str(&canonical).expect("canonical UUID");
    static LOCKS: OnceLock<[Arc<tokio::sync::Mutex<()>>; 32]> = OnceLock::new();
    let locks =
        LOCKS.get_or_init(|| std::array::from_fn(|_| Arc::new(tokio::sync::Mutex::new(()))));
    Ok(locks[usize::from(uuid.as_bytes()[0]) % locks.len()].clone())
}
pub(crate) async fn lock(id: &str) -> Result<tokio::sync::OwnedMutexGuard<()>, ApiError> {
    Ok(mutation_mutex(id)?.lock_owned().await)
}
pub(crate) fn try_lock(id: &str) -> Result<Option<tokio::sync::OwnedMutexGuard<()>>, ApiError> {
    Ok(mutation_mutex(id)?.try_lock_owned().ok())
}

pub(crate) fn private_headers(state: &AppState) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "cache-control",
        HeaderValue::from_static("private, no-store"),
    );
    if let Ok(value) = HeaderValue::from_str(state.instance_id.as_str()) {
        headers.insert("x-mold-instance-id", value);
    }
    headers
}

#[derive(Default, serde::Deserialize)]
pub(crate) struct ValidateQuery {
    #[serde(default)]
    pub planning_only: bool,
}

#[utoipa::path(post,path="/api/video-jobs/validate",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("planning_only"=Option<bool>,Query,description="Read-only scalar length preview")),request_body=VideoJobRequest,responses((status=200,body=VideoJobPlan)))]
pub(crate) async fn validate(
    State(state): State<AppState>,
    Query(query): Query<ValidateQuery>,
    headers: HeaderMap,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    auth: Option<Extension<crate::auth::AuthState>>,
    Json(mut request): Json<VideoJobRequest>,
) -> Result<(HeaderMap, Json<VideoJobPlan>), ApiError> {
    require_instance(state.instance_id.as_str(), &headers)?;
    database(&state)?;
    apply_headers(&headers, &mut request)?;
    let _permit = if query.planning_only {
        None
    } else {
        Some(admission_permit().await?)
    };
    let prepared = crate::video_job_admission::prepare(
        &state,
        request,
        &headers,
        authenticated.as_ref(),
        auth.as_ref(),
        query.planning_only,
    )
    .await?;
    Ok((private_headers(&state), Json(prepared.plan)))
}

pub(crate) async fn admission_permit() -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    static ADMISSION: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    ADMISSION
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("Video admission is unavailable"))
}

pub(crate) fn load(state: &AppState, id: &str) -> Result<db::StoredVideoJob, ApiError> {
    canonical_id(id)?;
    let database = database(state)?;
    if let Some(row) = db::get(database, id).map_err(internal)? {
        return Ok(row);
    }
    if db::is_aborted(database, id).map_err(internal)? {
        return Err(ApiError::with_code(
            "This video operation was stopped before admission",
            "VIDEO_OPERATION_CANCELLED",
            StatusCode::GONE,
        ));
    }
    Err(ApiError::with_code(
        "Video is not in this machine's queue",
        "VIDEO_JOB_NOT_FOUND",
        StatusCode::NOT_FOUND,
    ))
}
fn internal(error: impl std::fmt::Display) -> ApiError {
    ApiError::internal(error.to_string())
}

#[utoipa::path(get,path="/api/video-jobs/{id}",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),responses((status=200,body=VideoJob),(status=404,description="Unknown operation"),(status=410,description="Stopped operation")))]
pub(crate) async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
) -> Result<(HeaderMap, Json<VideoJob>), ApiError> {
    require_instance(state.instance_id.as_str(), &headers)?;
    canonical_id(&id)?;
    let guard = lock(&id).await?;
    if db::get(database(&state)?, &id).map_err(internal)?.is_none()
        && !db::is_aborted(database(&state)?, &id).map_err(internal)?
    {
        // Preserve CREATE's parent-lock -> I/O-slot order. Only reconciliation
        // hashes retained media; ordinary status reads need no I/O slot.
        let permit = admission_permit().await?;
        let owner = crate::gallery_source_media::reuse_credential_identity(authenticated.as_ref());
        let recover_state = state.clone();
        let recover_id = id.clone();
        tokio::task::spawn_blocking(move || {
            // Reconciliation can outlive its HTTP request during hashing.
            let _guard = guard;
            let _permit = permit;
            let directory = crate::video_job_manifest::job_dir(&recover_state, &recover_id)
                .map_err(internal)?;
            let Some(pending) = crate::video_job_pending::read_pending(
                &directory,
                &recover_state.instance_id,
                &owner,
            )
            .map_err(internal)?
            else {
                return Ok::<_, ApiError>(());
            };
            if pending.job.id != recover_id {
                return Err(ApiError::validation("Pending operation identity changed"));
            }
            if pending.ready() {
                crate::video_job_create::verify_owned_pending(
                    &recover_state,
                    &directory,
                    &pending,
                )?;
                db::insert(database(&recover_state)?, &pending.row()).map_err(internal)?;
            }
            Ok(())
        })
        .await
        .map_err(internal)??;
    }
    kick(&state);
    Ok((private_headers(&state), Json(load(&state, &id)?.job)))
}

pub(crate) fn kick(state: &AppState) {
    if let Some(runner) = &state.video_jobs {
        runner.kick();
    }
}

/// Cleanup runs behind the admission lock, after the abort transaction wins.
/// A public GET can already acknowledge Stop while a large copy unwinds.
fn cleanup_aborted(state: AppState, id: String) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Ok(_guard) = lock(&id).await else {
            return;
        };
        let copy = state.clone();
        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let database = database(&copy).map_err(|e| anyhow::anyhow!(e.error))?;
            if db::get(database, &id)?.is_some() || !db::is_aborted(database, &id)? {
                return Ok(());
            }
            let directory = crate::video_job_manifest::job_dir(&copy, &id)?;
            if !directory.try_exists()? {
                return Ok(());
            }
            crate::chain_source_media::release_all(
                &crate::video_job_manifest::jobs_root(&copy)?,
                &directory,
            )?;
            std::fs::remove_dir_all(&directory)?;
            Ok(())
        })
        .await;
        if !matches!(result, Ok(Ok(()))) {
            tracing::warn!("Could not clean stopped private video admission");
        }
    })
}

#[cfg(feature = "mp4")]
#[utoipa::path(post,path="/api/video-jobs/{id}/resume",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),responses((status=200,body=VideoJob),(status=409,description="Frozen inputs or recipe changed")))]
pub(crate) async fn resume(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<(StatusCode, HeaderMap, Json<VideoJob>), ApiError> {
    require_instance(state.instance_id.as_str(), &headers)?;
    let guard = lock(&id).await?;
    let row = load(&state, &id)?;
    if !row.job.state.can_resume() {
        return Err(ApiError::validation(
            "Only a failed, paused or interrupted video can resume",
        ));
    }
    let config = state.config.read().await.clone();
    let copy = state.clone();
    let checked = row.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let _guard = guard;
        let manifest = crate::video_job_manifest::read_manifest(&checked)?;
        crate::video_job_actor::verify_recipe(&config, &manifest)?;
        crate::chain_source_media::hydrate(
            &crate::video_job_manifest::jobs_root(&copy)?,
            &crate::video_job_manifest::job_dir(&copy, &checked.job.id)?,
            manifest.template.clone(),
        )?;
        anyhow::ensure!(
            db::resume(
                database(&copy).map_err(|e| anyhow::anyhow!(e.error))?,
                &checked.job.id,
                checked.revision,
                now_ms()
            )?,
            "Video changed while validating Resume; refresh and try again"
        );
        Ok(())
    })
    .await
    .map_err(internal)?
    .map_err(|error| {
        ApiError::with_code(
            error.to_string(),
            "VIDEO_RESUME_UNAVAILABLE",
            StatusCode::CONFLICT,
        )
    })?;
    kick(&state);
    Ok((
        StatusCode::ACCEPTED,
        private_headers(&state),
        Json(load(&state, &id)?.job),
    ))
}

#[derive(serde::Deserialize)]
pub(crate) struct ListQuery {
    #[serde(default = "default_limit")]
    limit: u32,
    #[serde(default = "yes")]
    recoverable_only: bool,
    cursor: Option<String>,
}
fn default_limit() -> u32 {
    50
}
fn yes() -> bool {
    true
}
#[derive(serde::Serialize, utoipa::ToSchema)]
pub(crate) struct Listing {
    jobs: Vec<VideoJob>,
    has_more: bool,
    next_cursor: Option<String>,
}

#[utoipa::path(get,path="/api/video-jobs",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("limit"=Option<u32>,Query,description="Bounded page size"),("cursor"=Option<String>,Query,description="Installation-bound opaque cursor"),("recoverable_only"=Option<bool>,Query,description="Defaults to true")),responses((status=200,body=Listing)))]
pub(crate) async fn list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
    headers: HeaderMap,
) -> Result<(HeaderMap, Json<Listing>), ApiError> {
    use base64::Engine;
    require_instance(state.instance_id.as_str(), &headers)?;
    let cursor = if let Some(cursor) = query.cursor {
        if cursor.len() > 2048 {
            return Err(ApiError::validation(
                "Video queue cursor exceeds its budget",
            ));
        }
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(cursor)
            .map_err(|_| ApiError::validation("Invalid video queue cursor"))?;
        let (instance, recoverable, cursor): (String, bool, db::VideoJobCursor) =
            serde_json::from_slice(&bytes)
                .map_err(|_| ApiError::validation("Invalid video queue cursor"))?;
        if instance != state.instance_id.as_str() || recoverable != query.recoverable_only {
            return Err(ApiError::validation(
                "Video queue cursor belongs to another installation or filter",
            ));
        }
        Some(cursor)
    } else {
        None
    };
    let page = db::list_page(
        database(&state)?,
        query.limit,
        query.recoverable_only,
        cursor.as_ref(),
    )
    .map_err(|error| ApiError::validation(error.to_string()))?;
    let next_cursor = page
        .next_cursor
        .map(|cursor| {
            serde_json::to_vec(&(state.instance_id.as_str(), query.recoverable_only, cursor))
        })
        .transpose()
        .map_err(internal)?
        .map(|bytes| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes));
    Ok((
        private_headers(&state),
        Json(Listing {
            jobs: page.jobs,
            has_more: page.has_more,
            next_cursor,
        }),
    ))
}

/// Stop intent is a durable transaction, including before a POST is accepted.
/// It never waits for the per-parent filesystem mutation lock or a GPU lease.
#[utoipa::path(post,path="/api/video-jobs/{id}/cancel",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),responses((status=202,description="Durable Stop intent accepted; response accepted=true")))]
pub(crate) async fn cancel(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<(StatusCode, HeaderMap, Json<serde_json::Value>), ApiError> {
    require_instance(state.instance_id.as_str(), &headers)?;
    canonical_id(&id)?;
    db::request_cancel(database(&state)?, &id, now_ms()).map_err(internal)?;
    db::settle_cancel_if_idle(database(&state)?, &id, now_ms()).map_err(internal)?;
    if let Some(row) = db::get(database(&state)?, &id).map_err(internal)? {
        if let (Some(child), Some(runner)) = (&row.job.active_child_id, &state.chain_jobs) {
            runner.request_cancel(child);
            runner.kick();
        }
    }
    kick(&state);
    let _cleanup = cleanup_aborted(state.clone(), id);
    Ok((
        StatusCode::ACCEPTED,
        private_headers(&state),
        Json(serde_json::json!({"accepted":true})),
    ))
}

/// Removing a settled queue entry keeps its published Library authority.
#[utoipa::path(delete,path="/api/video-jobs/{id}",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),responses((status=204,description="Settled queue entry discarded; Library movie retained"),(status=409,description="Stop active work first")))]
pub(crate) async fn remove(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<(StatusCode, HeaderMap), ApiError> {
    require_instance(state.instance_id.as_str(), &headers)?;
    let _guard = lock(&id).await?;
    let row = load(&state, &id)?;
    if !row.job.can_delete
        || matches!(
            row.job.state,
            mold_core::video_job::VideoJobState::Queued
                | mold_core::video_job::VideoJobState::Running
        )
    {
        return Err(ApiError::validation(
            "Stop this video and wait for it to settle before removing it",
        ));
    }
    crate::video_job_child_authority::cleanup_owned_stage_generations(&state, &id).await?;
    let copy = state.clone();
    let removed = row.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let database=database(&copy).map_err(|e|anyhow::anyhow!(e.error))?;
        let mut cursor=String::new();
        loop {
            let children=database.with_conn(|conn| {
                let mut statement=conn.prepare("SELECT child_id FROM video_job_child_owners WHERE parent_id=?1 AND child_id>?2 ORDER BY child_id LIMIT 32")?;
                let rows=statement.query_map((&removed.job.id,&cursor),|row|row.get::<_,String>(0))?;
                Ok(rows.collect::<Result<Vec<_>,_>>()?)
            })?;
            if children.is_empty() {break;}
            for child in children {
                cursor=child.clone();
                let Some(runner)=&copy.chain_jobs else {anyhow::bail!("Chain recovery is unavailable");};
                let _child_guard=runner.blocking_lock_job(&child);
                if let Some(child_row)=mold_db::chain_jobs::get_job(database,&child)? {
                    anyhow::ensure!(child_row.state!=mold_core::chain_job::ChainJobState::Running,"A child is still stopping; refresh and try again");
                    let root=database.path().parent().context("Missing captured data home")?.join("jobs");
                    anyhow::ensure!(child_row.job_dir==root.join(&child),"Child storage does not match its captured owner");
                    crate::chain_source_media::release_all(&root,&child_row.job_dir)?;
                    if child_row.job_dir.try_exists()? {std::fs::remove_dir_all(&child_row.job_dir)?;}
                    anyhow::ensure!(mold_db::chain_jobs::delete_job_not_running(database,&child)?,"Child changed during removal");
                    runner.cleanup_deleted(&child);
                }
            }
        }
        anyhow::ensure!(db::delete_settled(database,&removed.job.id,removed.revision,now_ms())?,"Video changed during removal; refresh and try again");
        Ok(())
    }).await.map_err(internal)?.map_err(internal)?;
    let _cleanup = cleanup_aborted(state.clone(), id);
    Ok((StatusCode::NO_CONTENT, private_headers(&state)))
}

/// Status frames contain only the public parent projection, never retained media.
#[utoipa::path(get,path="/api/video-jobs/{id}/events",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),responses((status=200,description="SSE snapshot events containing the full public VideoJob",content_type="text/event-stream")))]
pub(crate) async fn events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<
    (
        HeaderMap,
        axum::response::Sse<
            impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
        >,
    ),
    ApiError,
> {
    require_instance(state.instance_id.as_str(), &headers)?;
    load(&state, &id)?;
    let response_headers = private_headers(&state);
    let shutdown = state.events.shutdown_token();
    let stream = async_stream::stream! {
        let mut previous=None;
        let mut tick=tokio::time::interval(std::time::Duration::from_millis(300));
        loop {
            tokio::select! { _=shutdown.cancelled()=>break,_=tick.tick()=>{} }
            let row=match load(&state,&id) { Ok(row)=>row,Err(error)=> {
                yield Ok(axum::response::sse::Event::default().event("error").data(serde_json::to_string(&error).unwrap_or_default()));
                break;
            }};
            if previous!=Some(row.revision) {
                previous=Some(row.revision);
                let data=serde_json::to_string(&row.job).expect("public video projection serializes");
                yield Ok(axum::response::sse::Event::default().event("snapshot").data(data));
            }
            if row.job.state.is_terminal() || row.job.state==mold_core::video_job::VideoJobState::Paused { break; }
        }
    };
    Ok((
        response_headers,
        axum::response::Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()),
    ))
}

pub(crate) fn spawn_pending_sweeper(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Ok(root) = crate::video_job_manifest::jobs_root(&state) else {
            return;
        };
        let mut timer = tokio::time::interval(std::time::Duration::from_secs(3600));
        loop {
            tokio::select! { _=shutdown.cancelled()=>return,_=timer.tick()=>{} }
            let Ok(mut entries) = tokio::fs::read_dir(&root).await else {
                continue;
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                if shutdown.is_cancelled() {
                    return;
                }
                let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
                    continue;
                };
                if canonical_id(&id).is_err() {
                    continue;
                }
                if !entry
                    .file_type()
                    .await
                    .is_ok_and(|kind| kind.is_dir() && !kind.is_symlink())
                {
                    continue;
                }
                let Ok(database) = database(&state) else {
                    return;
                };
                if db::is_aborted(database, &id).unwrap_or(false)
                    && db::get(database, &id).is_ok_and(|row| row.is_none())
                {
                    let _ = cleanup_aborted(state.clone(), id).await;
                }
                tokio::task::yield_now().await;
            }
        }
    })
}

pub(crate) fn now_ms() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    #[cfg(feature = "mp4")]
    #[tokio::test]
    async fn mounted_video_status_rejects_replaced_installation_before_lookup() {
        use tower::ServiceExt;
        let state = AppState::for_tests();
        let response = crate::routes::create_router(state)
            .oneshot(
                axum::http::Request::builder()
                    .uri(format!("/api/video-jobs/{}", uuid::Uuid::new_v4()))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn status_events_use_snapshot_and_lossless_public_parent_projection() {
        use axum::response::IntoResponse;
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let mut row = crate::video_job_manifest::tests::fixture();
        row.job.id = uuid::Uuid::new_v4().to_string();
        row.job.state = mold_core::video_job::VideoJobState::Failed;
        row.job.plan.clip_count = u64::MAX;
        db::insert(database(&state).unwrap(), &row).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-mold-destination-instance",
            state.instance_id.parse().unwrap(),
        );
        let response = events(State(state), Path(row.job.id), headers)
            .await
            .unwrap()
            .into_response();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("event: snapshot"));
        assert!(text.contains("\"clip_count\":\"18446744073709551615\""));
        assert!(!text.contains("manifest_json"));
    }

    #[test]
    fn video_parent_headers_require_owner_and_preserve_opaque_u64_payloads() {
        let mut headers = HeaderMap::new();
        assert!(require_instance("owner", &headers).is_err());
        headers.insert("x-mold-destination-instance", "other".parse().unwrap());
        assert!(require_instance("owner", &headers).is_err());
        headers.insert("x-mold-destination-instance", "owner".parse().unwrap());
        assert!(require_instance("owner", &headers).is_ok());
        let mut request:mold_core::video_job::VideoJobRequest=serde_json::from_value(serde_json::json!({"generation":{"model":"ltx-2-19b-distilled:fp8","prompt":"literal\\n","width":512,"height":512,"steps":4,"seed":18446744073709551615_u64},"duration_ms":"30000"})).unwrap();
        headers.insert("x-mold-video-validation-revision", "proof".parse().unwrap());
        apply_headers(&headers, &mut request).unwrap();
        assert_eq!(request.validation_revision.as_deref(), Some("proof"));
        assert_eq!(request.generation.seed, Some(u64::MAX));
        assert_eq!(request.generation.prompt, "literal\\n");
        request.validation_revision = Some("other".into());
        assert!(apply_headers(&headers, &mut request).is_err());
    }
}
