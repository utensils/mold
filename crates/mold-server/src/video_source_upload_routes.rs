//! Explicit authenticated destination transport; source API keys never transit hosts.
use crate::{routes::ApiError, state::AppState};
use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    Json,
};
use mold_core::video_job::{VideoSourceUploadRef, VideoSourceUploadSpec, VideoSourceUploadStatus};
use serde::Deserialize;
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

type PrivateJson<T> = ([(header::HeaderName, &'static str); 1], Json<T>);
fn private<T>(value: T) -> PrivateJson<T> {
    ([(header::CACHE_CONTROL, "private, no-store")], Json(value))
}

pub(crate) fn root(state: &AppState) -> Result<PathBuf, ApiError> {
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .ok_or_else(|| ApiError::internal("Durable video storage is unavailable"))?;
    let parent = db
        .path()
        .parent()
        .filter(|parent| parent.is_absolute())
        .ok_or_else(|| ApiError::internal("Durable video storage has no captured home"))?;
    Ok(parent.join("video-source-uploads"))
}

fn destination(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if headers
        .get("x-mold-destination-instance")
        .and_then(|value| value.to_str().ok())
        != Some(state.instance_id.as_str())
    {
        return Err(ApiError::with_code(
            "The destination installation changed. Reconnect before transferring this video.",
            "VIDEO_DESTINATION_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    Ok(())
}
pub(crate) fn upload_error(error: anyhow::Error) -> ApiError {
    if error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    {
        return ApiError::not_found("The retained video upload is unavailable");
    }
    tracing::debug!(%error,"retained video upload refused");
    ApiError::with_code(
        "The retained video upload changed or is incomplete. Check its progress and try again.",
        "VIDEO_SOURCE_UPLOAD_CONFLICT",
        StatusCode::CONFLICT,
    )
}
async fn locked<R: Send + 'static>(
    state: AppState,
    headers: HeaderMap,
    id: String,
    work: impl FnOnce(PathBuf, String, String) -> anyhow::Result<R> + Send + 'static,
) -> Result<R, ApiError> {
    destination(&state, &headers)?;
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|_| ApiError::validation("Upload handle must be a UUID"))?;
    if uuid.to_string() != id {
        return Err(ApiError::validation(
            "Upload handle must be a canonical UUID",
        ));
    }
    // Fixed stripes bound the lock registry independently of upload lifetime.
    static LOCKS: OnceLock<[Arc<tokio::sync::Mutex<()>>; 32]> = OnceLock::new();
    let locks =
        LOCKS.get_or_init(|| std::array::from_fn(|_| Arc::new(tokio::sync::Mutex::new(()))));
    let guard = locks[usize::from(uuid.as_bytes()[0]) % locks.len()]
        .clone()
        .lock_owned()
        .await;
    let root = root(&state)?;
    let instance = state.instance_id.to_string();
    tokio::task::spawn_blocking(move || {
        let _guard = guard;
        work(root, instance, id)
    })
    .await
    .map_err(|_| ApiError::internal("Video transfer did not finish"))?
    .map_err(upload_error)
}

#[utoipa::path(post,path="/api/video-source-uploads/{id}",tag="video-jobs",params(("id"=String,Path,description="Client-generated upload UUID")),request_body=VideoSourceUploadSpec,responses((status=200,description="Upload progress",body=VideoSourceUploadStatus)))]
pub(crate) async fn begin(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(source): Json<VideoSourceUploadSpec>,
) -> Result<PrivateJson<VideoSourceUploadStatus>, ApiError> {
    let source = crate::video_source::RetainedVideoSource {
        authority_kind: source.authority_kind,
        filename: source.filename,
        archive_identity_sha256: source.archive_identity_sha256,
        checksum_sha256: source.sha256,
        size_bytes: source.size_bytes,
        metadata: source.metadata,
    };
    locked(state, headers, id, move |root, instance, id| {
        crate::video_source_upload::begin(&root, &instance, &id, source)
    })
    .await
    .map(private)
}
#[utoipa::path(get,path="/api/video-source-uploads/{id}",tag="video-jobs",params(("id"=String,Path,description="Upload UUID")),responses((status=200,description="Upload progress",body=VideoSourceUploadStatus)))]
pub(crate) async fn status(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<PrivateJson<VideoSourceUploadStatus>, ApiError> {
    locked(state, headers, id, |root, instance, id| {
        crate::video_source_upload::current(&root, &instance, &id)
    })
    .await
    .map(private)
}
#[derive(Deserialize)]
pub(crate) struct Offset {
    offset_bytes: u64,
}
#[utoipa::path(put,path="/api/video-source-uploads/{id}/chunks",tag="video-jobs",params(("id"=String,Path,description="Upload UUID"),("offset_bytes"=u64,Query,description="Exact committed byte offset")),request_body(content=Vec<u8>,content_type="application/octet-stream"),responses((status=200,description="Chunk committed",body=VideoSourceUploadStatus)))]
pub(crate) async fn append(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(offset): Query<Offset>,
    headers: HeaderMap,
    bytes: Bytes,
) -> Result<PrivateJson<VideoSourceUploadStatus>, ApiError> {
    locked(state, headers, id, move |root, instance, id| {
        crate::video_source_upload::append(&root, &instance, &id, offset.offset_bytes, &bytes)
    })
    .await
    .map(private)
}
#[utoipa::path(post,path="/api/video-source-uploads/{id}/complete",tag="video-jobs",params(("id"=String,Path,description="Upload UUID")),responses((status=200,description="Checksummed immutable source",body=VideoSourceUploadRef)))]
pub(crate) async fn complete(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<PrivateJson<VideoSourceUploadRef>, ApiError> {
    locked(state, headers, id, |root, instance, id| {
        crate::video_source_upload::complete(&root, &instance, &id, &mut || Ok(()))
    })
    .await
    .map(private)
}
#[utoipa::path(delete,path="/api/video-source-uploads/{id}",tag="video-jobs",params(("id"=String,Path,description="Upload UUID")),responses((status=204,description="Upload removed; admitted parent retains its own copy")))]
pub(crate) async fn remove(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<StatusCode, ApiError> {
    locked(state, headers, id, |root, instance, id| {
        crate::video_source_upload::remove(&root, &instance, &id)
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_namespace_uses_the_captured_database_home() {
        let home = tempfile::tempdir().unwrap();
        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(
            mold_db::MetadataDb::open(&home.path().join("mold.db")).unwrap(),
        ));
        assert_eq!(
            root(&state).unwrap(),
            home.path().join("video-source-uploads")
        );
    }
    #[tokio::test]
    async fn replaced_destination_is_refused_before_any_filesystem_or_operation() {
        let state = AppState::for_tests();
        let id = uuid::Uuid::new_v4().to_string();
        let result = locked(
            state,
            HeaderMap::new(),
            id,
            |_, _, _| -> anyhow::Result<()> { panic!("replaced installation reached mutation") },
        )
        .await;
        assert_eq!(result.unwrap_err().code, "VIDEO_DESTINATION_CHANGED");
    }
}
