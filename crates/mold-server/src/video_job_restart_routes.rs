//! Private revision-fenced HTTP adapters for new video attempts.
use crate::{routes::ApiError, state::AppState, video_jobs};
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    Extension, Json,
};
use mold_core::{
    job_restart::RestartExportRequest,
    video_restart::{VideoRestartExport, VideoRestartOptions},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn unavailable(error: impl std::fmt::Display) -> ApiError {
    tracing::debug!(%error, "video restart export unavailable");
    ApiError::with_code("The original video changed or its retained inputs are unavailable. Refresh the Queue before trying again.", "VIDEO_RESTART_REVISION_CHANGED", StatusCode::CONFLICT)
}

fn authorize(
    auth: Option<&Extension<crate::auth::AuthState>>,
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
) -> Result<(), ApiError> {
    if !crate::gallery_source_media::private_media_authorized(auth, authenticated) {
        return Err(ApiError::with_code(
            "API-key authentication required",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    Ok(())
}

#[utoipa::path(get,path="/api/video-jobs/{id}/restart-options",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),responses((status=200,body=VideoRestartOptions)))]
pub(crate) async fn options(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<(HeaderMap, Json<VideoRestartOptions>), ApiError> {
    video_jobs::require_instance(&state.instance_id, &headers)?;
    let _guard = video_jobs::lock(&id).await?;
    let row = video_jobs::load(&state, &id)?;
    let options =
        crate::video_job_restart::options_for(&state.instance_id, &row).map_err(unavailable)?;
    Ok((video_jobs::private_headers(&state), Json(options)))
}

/// A dropped HTTP request interrupts hashing/copying even though spawn_blocking
/// itself cannot be aborted. No private bytes are persisted in a client journal.
struct RequestLifetime(Arc<AtomicBool>);
impl Drop for RequestLifetime {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[utoipa::path(post,path="/api/video-jobs/{id}/restart-export",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("id"=String, Path, description="Video operation UUID")),request_body=RestartExportRequest,responses((status=200,body=VideoRestartExport),(status=409,description="Source revision changed")))]
pub(crate) async fn export(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Json(authority): Json<RestartExportRequest>,
) -> Result<(HeaderMap, Json<VideoRestartExport>), ApiError> {
    video_jobs::require_instance(&state.instance_id, &headers)?;
    authorize(auth.as_ref(), authenticated.as_ref())?;
    let _guard = video_jobs::lock(&id).await?;
    let _permit = video_jobs::admission_permit().await?;
    let row = video_jobs::load(&state, &id)?;
    let original =
        crate::video_job_restart::options_for(&state.instance_id, &row).map_err(unavailable)?;
    if authority.source_instance_id != original.source_instance_id
        || authority.source_revision != original.source_revision
    {
        return Err(unavailable("source revision changed"));
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let _lifetime = RequestLifetime(cancelled.clone());
    let worker_state = state.clone();
    let worker_id = id.clone();
    let (_guards, payload) = crate::video_job_create::blocking_admission(
        crate::video_job_create::AdmissionGuards::new(_guard, _permit),
        move || {
            (|| -> anyhow::Result<_> {
                let manifest = crate::video_job_manifest::read_manifest(&row)?;
                let database = video_jobs::database(&worker_state)
                    .map_err(|_| anyhow::anyhow!("video database unavailable"))?;
                let mut calls = 0usize;
                let mut checkpoint = || -> anyhow::Result<()> {
                    anyhow::ensure!(!cancelled.load(Ordering::Acquire), "request disconnected");
                    if calls.is_multiple_of(16) {
                        let current = mold_db::video_jobs::get(database, &worker_id)?
                            .ok_or_else(|| anyhow::anyhow!("video removed"))?;
                        anyhow::ensure!(
                            current.revision == row.revision
                                && current.request_sha256 == row.request_sha256
                                && current.manifest_sha256 == row.manifest_sha256,
                            "source revision changed"
                        );
                    }
                    calls = calls.wrapping_add(1);
                    Ok(())
                };
                crate::video_job_restart::export_for_with_checkpoint(
                    &worker_state,
                    &row,
                    &manifest,
                    &authority,
                    &mut checkpoint,
                )
            })()
            .map_err(unavailable)
        },
    )
    .await?;
    let payload = payload?;
    let current = video_jobs::load(&state, &id)?;
    let current =
        crate::video_job_restart::options_for(&state.instance_id, &current).map_err(unavailable)?;
    if current.source_revision != original.source_revision {
        return Err(unavailable("source revision changed during export"));
    }
    Ok((video_jobs::private_headers(&state), Json(payload)))
}

#[derive(serde::Deserialize)]
pub(crate) struct SourceQuery {
    offset_bytes: u64,
}

#[utoipa::path(get,path="/api/video-jobs/{id}/restart-source",tag="video-jobs",params(("id"=String, Path, description="Video operation UUID"),("x-mold-source-instance"=String,Header,description="Captured source installation UUID"),("x-mold-video-source-ref"=String,Header,description="Revision-bound opaque prefix locator"),("offset_bytes"=u64,Query,description="Source byte offset")),responses((status=200,description="At most 4 MiB of the retained original prefix",content_type="application/octet-stream")))]
pub(crate) async fn source(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<SourceQuery>,
    headers: HeaderMap,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
) -> Result<(HeaderMap, Vec<u8>), ApiError> {
    if headers
        .get("x-mold-source-instance")
        .and_then(|value| value.to_str().ok())
        != Some(state.instance_id.as_str())
    {
        return Err(ApiError::with_code(
            "The source installation changed. Refresh before continuing.",
            "VIDEO_SOURCE_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    authorize(auth.as_ref(), authenticated.as_ref())?;
    let locator = headers
        .get("x-mold-video-source-ref")
        .and_then(|value| value.to_str().ok())
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| ApiError::validation("A revision-bound video source reference is required"))?
        .to_owned();
    let _guard = video_jobs::lock(&id).await?;
    let _permit = video_jobs::admission_permit().await?;
    let row = video_jobs::load(&state, &id)?;
    let worker_state = state.clone();
    let (_guards, bytes) = crate::video_job_create::blocking_admission(
        crate::video_job_create::AdmissionGuards::new(_guard, _permit),
        move || {
            (|| -> anyhow::Result<_> {
                let manifest = crate::video_job_manifest::read_manifest(&row)?;
                crate::video_job_restart::source_chunk(
                    &worker_state,
                    &row,
                    &manifest,
                    &locator,
                    query.offset_bytes,
                )
            })()
            .map_err(unavailable)
        },
    )
    .await?;
    let bytes = bytes?;
    let mut headers = video_jobs::private_headers(&state);
    headers.insert(
        "content-type",
        "application/octet-stream"
            .parse()
            .expect("static media type"),
    );
    headers.insert(
        "content-length",
        bytes.len().to_string().parse().expect("bounded length"),
    );
    headers.insert(
        "x-mold-source-instance",
        state.instance_id.parse().map_err(unavailable)?,
    );
    Ok((headers, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (AppState, mold_db::video_jobs::StoredVideoJob, HeaderMap) {
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let mut row = crate::video_job_manifest::tests::fixture();
        row.job.id = uuid::Uuid::new_v4().to_string();
        row.job.state = mold_core::video_job::VideoJobState::Failed;
        mold_db::video_jobs::insert(video_jobs::database(&state).unwrap(), &row).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-mold-destination-instance",
            state.instance_id.parse().unwrap(),
        );
        (state, row, headers)
    }
    #[tokio::test]
    async fn restart_options_are_private_and_bound_to_current_revision() {
        let (state, row, headers) = fixture();
        let (response_headers, Json(response)) =
            options(State(state.clone()), Path(row.job.id.clone()), headers)
                .await
                .unwrap();
        assert_eq!(response_headers["cache-control"], "private, no-store");
        assert_eq!(
            response_headers["x-mold-instance-id"],
            state.instance_id.as_str()
        );
        assert_eq!(response.source_instance_id, *state.instance_id);
        assert!(response.restart.supported);
        assert_eq!(
            response.source_revision,
            crate::video_job_restart::options_for(&state.instance_id, &row)
                .unwrap()
                .source_revision
        );
    }
    #[tokio::test]
    async fn stale_export_is_refused_before_hydrating_private_inputs() {
        let (state, row, headers) = fixture();
        let authority = RestartExportRequest {
            destination_instance_id: uuid::Uuid::new_v4().to_string(),
            source_instance_id: state.instance_id.to_string(),
            source_revision: "old-revision".into(),
        };
        let result = export(
            State(state),
            Path(row.job.id),
            headers,
            None,
            None,
            Json(authority),
        )
        .await
        .unwrap_err();
        assert_eq!(result.code, "VIDEO_RESTART_REVISION_CHANGED");
    }
    #[tokio::test]
    async fn source_owner_and_locator_are_required_before_lookup() {
        let state = AppState::for_tests();
        let id = uuid::Uuid::new_v4().to_string();
        let result = source(
            State(state.clone()),
            Path(id.clone()),
            Query(SourceQuery { offset_bytes: 0 }),
            HeaderMap::new(),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(result.code, "VIDEO_SOURCE_CHANGED");
        let mut headers = HeaderMap::new();
        headers.insert("x-mold-source-instance", state.instance_id.parse().unwrap());
        let result = source(
            State(state),
            Path(id),
            Query(SourceQuery { offset_bytes: 0 }),
            headers,
            None,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(result.code, "VALIDATION_ERROR");
    }
    #[tokio::test]
    async fn waiting_exports_do_not_reserve_io_slots_needed_by_create() {
        let (state, row, headers) = fixture();
        let held_parent = video_jobs::lock(&row.job.id).await.unwrap();
        let first_slot = video_jobs::admission_permit().await.unwrap();
        let worker = tokio::spawn(export(
            State(state.clone()),
            Path(row.job.id),
            headers,
            None,
            None,
            Json(RestartExportRequest {
                destination_instance_id: state.instance_id.to_string(),
                source_instance_id: state.instance_id.to_string(),
                source_revision: "revision".into(),
            }),
        ));
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        let available = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            video_jobs::admission_permit(),
        )
        .await;
        worker.abort();
        let _ = worker.await;
        drop(held_parent);
        drop(first_slot);
        assert!(available.is_ok(), "a request waiting on a parent must leave an I/O slot for the holder of that parent lock");
    }
    #[tokio::test]
    async fn restart_options_refuse_a_replaced_installation_before_lookup() {
        let state = AppState::for_tests();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-mold-destination-instance",
            "replacement".parse().unwrap(),
        );
        let result = options(
            State(state),
            Path(uuid::Uuid::new_v4().to_string()),
            headers,
        )
        .await
        .unwrap_err();
        assert_eq!(result.code, "VIDEO_DESTINATION_CHANGED");
    }
    #[test]
    fn disconnect_cancels_blocking_export_work() {
        let cancelled = Arc::new(AtomicBool::new(false));
        {
            let _request = RequestLifetime(cancelled.clone());
            assert!(!cancelled.load(Ordering::Acquire));
        }
        assert!(cancelled.load(Ordering::Acquire));
    }
    #[tokio::test]
    async fn export_preserves_literal_prompt_and_full_seed_without_client_numeric_parsing() {
        use sha2::{Digest, Sha256};
        let home = tempfile::tempdir().unwrap();
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(
            mold_db::MetadataDb::open(&home.path().join("mold.db")).unwrap(),
        ));
        let mut row = crate::video_job_manifest::tests::fixture();
        row.job.id = uuid::Uuid::new_v4().to_string();
        row.job.state = mold_core::video_job::VideoJobState::Failed;
        let mut manifest: serde_json::Value = serde_json::from_str(&row.manifest_json).unwrap();
        let prompt = "literal\\n camera\\t path \\\" keep";
        manifest["request"]["generation"]["prompt"] = prompt.into();
        manifest["request"]["generation"]["seed"] = u64::MAX.into();
        manifest["template"]["stages"][0]["prompt"] = prompt.into();
        manifest["template"]["seed"] = u64::MAX.into();
        row.manifest_json = serde_json::to_string(&manifest).unwrap();
        row.manifest_sha256 = format!("{:x}", Sha256::digest(row.manifest_json.as_bytes()));
        mold_db::video_jobs::insert(video_jobs::database(&state).unwrap(), &row).unwrap();
        let current = crate::video_job_restart::options_for(&state.instance_id, &row).unwrap();
        let authority = RestartExportRequest {
            destination_instance_id: state.instance_id.to_string(),
            source_instance_id: current.source_instance_id,
            source_revision: current.source_revision,
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-mold-destination-instance",
            state.instance_id.parse().unwrap(),
        );
        let (headers, Json(result)) = export(
            State(state),
            Path(row.job.id),
            headers,
            None,
            None,
            Json(authority),
        )
        .await
        .unwrap();
        assert_eq!(headers["cache-control"], "private, no-store");
        let request: mold_core::video_job::VideoJobRequest =
            serde_json::from_str(&result.request_json).unwrap();
        assert_eq!(request.generation.prompt, prompt);
        assert_eq!(request.generation.seed, Some(u64::MAX));
        assert!(request.validation_revision.is_none());
        assert!(request.generation.placement.is_none());
        assert!(result.prefix.is_none());
    }
}
