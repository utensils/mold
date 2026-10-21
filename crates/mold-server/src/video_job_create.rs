//! Durable admission, separate from readonly preparation and bounded execution.

use crate::{
    routes::ApiError,
    state::AppState,
    video_job_manifest::{self, VideoJobManifest},
    video_job_pending::{self, PendingAdmission},
};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Extension, Json,
};
use mold_core::video_job::{VideoJob, VideoJobRequest, VideoJobState};
use mold_db::video_jobs as db;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

fn cancelled() -> ApiError {
    ApiError::with_code(
        "This video operation was stopped before admission",
        "VIDEO_OPERATION_CANCELLED",
        StatusCode::GONE,
    )
}
fn storage(error: impl std::fmt::Display) -> ApiError {
    ApiError::internal(error.to_string())
}
fn request_identity(request: &VideoJobRequest) -> Result<String, ApiError> {
    let mut request = request.clone();
    request.validation_revision = None;
    serde_json::to_vec(&request)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(storage)
}
fn check_abort(state: &AppState, id: &str) -> Result<(), ApiError> {
    if db::is_aborted(crate::video_jobs::database(state)?, id).map_err(storage)? {
        return Err(cancelled());
    }
    Ok(())
}
fn private_directory(path: &Path) -> anyhow::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(path) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error.into()),
    }
    let metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "invalid private video storage"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "video admission storage is not private"
        );
    }
    Ok(())
}

pub(crate) struct AdmissionGuards {
    _lock: tokio::sync::OwnedMutexGuard<()>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl AdmissionGuards {
    pub(crate) fn new(
        lock: tokio::sync::OwnedMutexGuard<()>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Self {
        Self {
            _lock: lock,
            _permit: permit,
        }
    }
}

pub(crate) async fn blocking_admission<T: Send + 'static>(
    guards: AdmissionGuards,
    work: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<(AdmissionGuards, Result<T, ApiError>), ApiError> {
    // Blocking tasks outlive dropped HTTP futures. Their mutation lock and I/O
    // slot must outlive sealing/rollback too, then return for one-use claims.
    tokio::task::spawn_blocking(move || (guards, work()))
        .await
        .map_err(storage)
}

/// Only an accepted fully-owned operation appears in the public parent table.
#[utoipa::path(post,path="/api/video-jobs",tag="video-jobs",params(("x-mold-destination-instance"=String, Header, description="Captured installation UUID"),("x-mold-operation-id"=String,Header,description="Client-generated canonical UUID, retained across uncertain replies"),("x-mold-video-validation-revision"=Option<String>,Header,description="Full validation proof")),request_body=VideoJobRequest,responses((status=202,body=VideoJob),(status=409,description="Admission or authority changed"),(status=410,description="Stopped operation")))]
pub(crate) async fn create(
    State(state): State<AppState>,
    headers: HeaderMap,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    auth: Option<Extension<crate::auth::AuthState>>,
    Json(mut request): Json<VideoJobRequest>,
) -> Result<(StatusCode, HeaderMap, Json<VideoJob>), ApiError> {
    crate::video_jobs::require_instance(&state.instance_id, &headers)?;
    let id = crate::video_jobs::canonical_id(
        headers
            .get("x-mold-operation-id")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                ApiError::validation("Capture a video operation UUID before submitting")
            })?,
    )?;
    let lock = crate::video_jobs::lock(&id).await?;
    check_abort(&state, &id)?;
    crate::video_jobs::apply_headers(&headers, &mut request)?;
    let identity = request_identity(&request)?;
    let database = crate::video_jobs::database(&state)?;
    let owner = crate::gallery_source_media::reuse_credential_identity(authenticated.as_ref());
    let directory = video_job_manifest::job_dir(&state, &id).map_err(storage)?;
    if let Some(row) = db::get(database, &id).map_err(storage)? {
        if row.request_sha256 != identity {
            return Err(ApiError::with_code(
                "This operation UUID already belongs to another request",
                "VIDEO_OPERATION_CONFLICT",
                StatusCode::CONFLICT,
            ));
        }
        return Ok((
            StatusCode::ACCEPTED,
            crate::video_jobs::private_headers(&state),
            Json(row.job),
        ));
    }
    if let Some(pending) =
        video_job_pending::read_pending(&directory, &state.instance_id, &owner).map_err(storage)?
    {
        if pending.request_sha256 != identity {
            return Err(ApiError::with_code(
                "This operation UUID already belongs to another request",
                "VIDEO_OPERATION_CONFLICT",
                StatusCode::CONFLICT,
            ));
        }
        if !pending.ready() {
            return Err(ApiError::with_code("Input ownership was interrupted before admission. Stop this operation, then prepare a new request.", "VIDEO_ADMISSION_INPUTS_PENDING", StatusCode::CONFLICT));
        }
        let recovery_state = state.clone();
        let recovery_directory = directory.clone();
        let permit = crate::video_jobs::admission_permit().await?;
        let (_guards, row) = blocking_admission(AdmissionGuards::new(lock, permit), move || {
            verify_owned_pending(&recovery_state, &recovery_directory, &pending)?;
            check_abort(&recovery_state, &pending.job.id)?;
            let row = pending.row();
            db::insert(crate::video_jobs::database(&recovery_state)?, &row).map_err(storage)?;
            Ok::<_, ApiError>(row)
        })
        .await?;
        let row = row?;
        return Ok((
            StatusCode::ACCEPTED,
            crate::video_jobs::private_headers(&state),
            Json(row.job),
        ));
    }
    let expected = request
        .validation_revision
        .clone()
        .filter(|revision| !revision.is_empty())
        .ok_or_else(|| {
            ApiError::with_code(
                "Validate the prepared video request before generating",
                "VIDEO_VALIDATION_REQUIRED",
                StatusCode::CONFLICT,
            )
        })?;
    crate::routes::ensure_generation_available(&state)?;
    if state.chain_jobs.is_none() {
        return Err(ApiError::with_code(
            "The video runner is unavailable",
            "VIDEO_JOBS_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    let permit = crate::video_jobs::admission_permit().await?;
    let original = request.clone();
    let prepared = crate::video_job_admission::prepare(
        &state,
        request,
        &headers,
        authenticated.as_ref(),
        auth.as_ref(),
        false,
    )
    .await?;
    if prepared.plan.validation_revision != expected {
        return Err(ApiError::with_code(
            "The recipe, inputs or duration changed. Validate this request again.",
            "VIDEO_VALIDATION_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    check_abort(&state, &id)?;
    let seal_state = state.clone();
    let seal_id = id.clone();
    let seal_owner = owner.clone();
    // Retain returned guards through one-use input claims and parent insertion.
    let (_guards, pending) = blocking_admission(AdmissionGuards::new(lock, permit), move || {
        seal(&seal_state, &seal_id, &identity, &seal_owner, prepared)
    })
    .await?;
    let pending = pending?;
    // The original logical request still carries one-use handles. Every byte
    // has already been independently sealed before acquiring those handles.
    let mut inputs = original.generation;
    let mut receipt = |claim: &crate::reference_uploads::InputConsumptionClaim| {
        check_abort(&state, &id)
            .map_err(crate::reference_uploads::ConsumptionReceiptError::NotPublished)?;
        video_job_pending::record_claim(&directory, &pending, claim)
    };
    crate::gallery_source_media::hydrate_reuse_session_with_receipt(
        &state,
        authenticated.as_ref(),
        auth.as_ref(),
        &headers,
        &mut inputs,
        32 * 1024 * 1024,
        &mut receipt,
    )
    .await?;
    let input_identity = crate::reference_uploads::ReferenceIdentity::resolve(
        authenticated.as_ref().map(|value| &value.0),
        auth.as_ref().map(|value| &value.0),
        &state.instance_id,
    );
    let roots = state.config.read().await.resolved_media_roots();
    let staged = state
        .reference_uploads
        .resolve_request_with_receipt(
            input_identity.as_ref(),
            &mut inputs,
            &roots,
            None,
            &mut receipt,
        )
        .await?;
    drop(staged);
    let pending = video_job_pending::read_pending(&directory, &state.instance_id, &owner)
        .map_err(storage)?
        .ok_or_else(|| ApiError::internal("Private admission disappeared"))?;
    if !pending.ready() {
        return Err(ApiError::internal("Input ownership is incomplete"));
    }
    check_abort(&state, &id)?;
    let row = pending.row();
    db::insert(database, &row).map_err(|error| {
        if db::is_aborted(database, &id).unwrap_or(false) {
            cancelled()
        } else {
            storage(error)
        }
    })?;
    if let Some(upload) = original.source.and_then(|source| source.retained_upload) {
        if let Ok(root) = crate::video_source_upload_routes::root(&state) {
            let instance = state.instance_id.to_string();
            // Acceptance already owns its private prefix. Ingress cleanup
            // cannot turn a successful durable admission into an error.
            tokio::task::spawn_blocking(move || {
                if crate::video_source_upload::remove(&root, &instance, &upload.handle).is_err() {
                    tracing::warn!("could not release accepted video source ingress");
                }
            });
        }
    }
    Ok((
        StatusCode::ACCEPTED,
        crate::video_jobs::private_headers(&state),
        Json(row.job),
    ))
}

fn seal(
    state: &AppState,
    id: &str,
    identity: &str,
    owner: &str,
    mut prepared: crate::video_job_admission::PreparedVideo,
) -> Result<PendingAdmission, ApiError> {
    check_abort(state, id)?;
    let root = video_job_manifest::jobs_root(state).map_err(storage)?;
    private_directory(&root).map_err(storage)?;
    let directory = video_job_manifest::job_dir(state, id).map_err(storage)?;
    private_directory(&directory).map_err(storage)?;
    let mut rollback = Some(crate::chain_source_media::CreateRollback::new(
        &root, &directory,
    ));
    let retained_source = if let Some((file, binding)) = prepared.source.take() {
        crate::video_source::retain(file, &binding, &directory, &mut || {
            check_abort(state, id).map_err(|error| anyhow::anyhow!(error.error))
        })
        .map_err(storage)?;
        Some(binding)
    } else {
        None
    };
    let prefix_audio = crate::video_prefix_audio::seal(&directory, &prepared.plan, &mut || {
        check_abort(state, id).map_err(|error| anyhow::anyhow!(error.error))
    })
    .map_err(storage)?;
    prepared.template.materialize_seed();
    prepared.request.validation_revision = None;
    prepared.request.generation.seed = prepared.template.seed;
    let frames = prepared.template.stages[0].frames;
    if let Some(keyframes) = prepared.template.stages[0].keyframes.as_mut() {
        for frame in keyframes {
            if frame.frame != 0 {
                frame.frame = frames - 1;
            }
        }
    }
    prepared.request.generation.frames = Some(frames);
    prepared.request.generation.source_image = None;
    prepared.request.generation.keyframes = None;
    prepared.request.generation.references = None;
    if let Some(source) = prepared.request.source.as_mut() {
        source.retained_upload = None;
        if let Some(binding) = &retained_source {
            source.expected_archive_identity_sha256 = Some(binding.archive_identity_sha256.clone());
        }
    }
    let template =
        crate::chain_source_media::persist_video_parent(&root, &directory, id, prepared.template)
            .map_err(storage)?;
    let manifest = VideoJobManifest {
        version: 1,
        request: prepared.request,
        template,
        frozen: prepared
            .frozen
            .ok_or_else(|| ApiError::internal("Admission has no frozen recipe"))?,
        frozen_loras: prepared.frozen_loras,
        envelope: prepared.envelope,
        retained_source,
        prefix_audio,
        plan: prepared.plan.clone(),
    };
    let json = serde_json::to_string(&manifest).map_err(storage)?;
    let now = crate::video_jobs::now_ms();
    let row = db::StoredVideoJob {
        job: VideoJob {
            id: id.into(),
            state: VideoJobState::Queued,
            plan: prepared.plan,
            completed_clips: 0,
            active_child_id: None,
            output_filename: None,
            error: None,
            error_detail: None,
            created_at_ms: now,
            updated_at_ms: now,
            can_cancel: true,
            can_resume: false,
            can_delete: false,
        },
        request_sha256: identity.into(),
        manifest_sha256: format!("{:x}", Sha256::digest(json.as_bytes())),
        manifest_json: json,
        revision: 0,
        cancel_requested: false,
    };
    video_job_manifest::read_manifest(&row).map_err(storage)?;
    let pending = PendingAdmission::new(&state.instance_id, owner, row, prepared.required_claims);
    let publication = video_job_pending::write_pending(&directory, &pending);
    // Even an ambiguous fsync error may have published a complete private
    // record. Keep its encrypted ownership for exact reconciliation.
    if directory
        .join("admission.json")
        .try_exists()
        .map_err(storage)?
    {
        rollback.take().expect("rollback exists").commit();
    }
    publication.map_err(storage)?;
    check_abort(state, id)?;
    Ok(pending)
}

pub(crate) fn verify_owned_pending(
    state: &AppState,
    directory: &Path,
    pending: &PendingAdmission,
) -> Result<(), ApiError> {
    if !pending.ready() {
        return Err(ApiError::validation("Input ownership is incomplete"));
    }
    let row = pending.row();
    let manifest = video_job_manifest::read_manifest(&row).map_err(storage)?;
    if let Some(audio) = &manifest.prefix_audio {
        crate::video_prefix_audio::open(directory, audio, &mut || {
            check_abort(state, &row.job.id).map_err(|error| anyhow::anyhow!(error.error))
        })
        .map_err(storage)?;
    }
    crate::chain_source_media::hydrate(
        &video_job_manifest::jobs_root(state).map_err(storage)?,
        directory,
        manifest.template.clone(),
    )
    .map_err(storage)?;
    if let Some(source) = &manifest.retained_source {
        let mut file =
            mold_core::secure_file::open_regular_file_no_follow(&directory.join("prefix.mp4"))
                .map_err(storage)?;
        if file.metadata().map_err(storage)?.len() != source.size_bytes {
            return Err(ApiError::validation("Retained source size changed"));
        }
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        let mut input = (&mut file).take(
            source
                .size_bytes
                .checked_add(1)
                .ok_or_else(|| ApiError::validation("Source size overflow"))?,
        );
        let mut length = 0u64;
        loop {
            check_abort(state, &row.job.id)?;
            let count = input.read(&mut buffer).map_err(storage)?;
            if count == 0 {
                break;
            }
            length = length
                .checked_add(count as u64)
                .ok_or_else(|| ApiError::validation("Source size overflow"))?;
            hash.update(&buffer[..count]);
        }
        if length != source.size_bytes
            || file.metadata().map_err(storage)?.len() != length
            || format!("{:x}", hash.finalize()) != source.checksum_sha256
        {
            return Err(ApiError::validation("Retained source identity changed"));
        }
    }
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(storage)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::State,
        http::{HeaderMap, HeaderValue},
        Json,
    };
    use std::sync::Arc;

    #[tokio::test]
    async fn dropped_create_retains_uuid_lock_and_io_slot_until_sealing_finishes() {
        let id = uuid::Uuid::new_v4().to_string();
        let lock = crate::video_jobs::lock(&id).await.unwrap();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let (entered, entered_rx) = tokio::sync::oneshot::channel();
        let (release, release_rx) = std::sync::mpsc::channel();
        let (finished, finished_rx) = tokio::sync::oneshot::channel();
        let handler = tokio::spawn(blocking_admission(
            AdmissionGuards {
                _lock: lock,
                _permit: permit,
            },
            move || {
                entered.send(()).unwrap();
                release_rx.recv().unwrap();
                let _ = finished.send(());
                Ok(())
            },
        ));
        entered_rx.await.unwrap();
        handler.abort();
        assert!(matches!(handler.await, Err(error) if error.is_cancelled()));
        // A retry or Stop cleanup must not enter this operation while detached
        // sealing still owns its rollback and retained media.
        let retry_blocked = crate::video_jobs::try_lock(&id).unwrap().is_none();
        let io_slot_held = semaphore.available_permits() == 0;
        release.send(()).unwrap();
        finished_rx.await.unwrap();
        assert!(
            retry_blocked,
            "dropped CREATE released its operation lock during sealing"
        );
        assert!(
            io_slot_held,
            "dropped CREATE released its bounded I/O slot during sealing"
        );
        let _guard = crate::video_jobs::lock(&id).await.unwrap();
        let released_permit = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            semaphore.clone().acquire_owned(),
        )
        .await
        .unwrap()
        .unwrap();
        drop(released_permit);
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn sealing_returns_both_guards_through_claims_on_success_and_error() {
        for succeeds in [true, false] {
            // This helper contract does not depend on the globally striped UUID
            // registry, which other concurrently running tests legitimately use.
            let mutex = Arc::new(tokio::sync::Mutex::new(()));
            let lock = mutex.clone().lock_owned().await;
            let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let (guards, result) = blocking_admission(
                AdmissionGuards {
                    _lock: lock,
                    _permit: permit,
                },
                move || {
                    if succeeds {
                        Ok(())
                    } else {
                        Err(ApiError::internal("injected sealing failure"))
                    }
                },
            )
            .await
            .unwrap();
            assert_eq!(result.is_ok(), succeeds);
            assert!(mutex.try_lock().is_err());
            assert_eq!(semaphore.available_permits(), 0);
            drop(guards);
            assert!(mutex.try_lock().is_ok());
            assert_eq!(semaphore.available_permits(), 1);
        }
    }

    #[tokio::test]
    async fn cancelled_operation_cannot_admit_or_require_its_expired_input_handles() {
        let home = tempfile::tempdir().unwrap();
        let mut state = crate::state::AppState::for_tests();
        state.metadata_db = Arc::new(Some(
            mold_db::MetadataDb::open(&home.path().join("mold.db")).unwrap(),
        ));
        let id = uuid::Uuid::new_v4().to_string();
        mold_db::video_jobs::request_cancel(crate::video_jobs::database(&state).unwrap(), &id, 1)
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-mold-destination-instance",
            HeaderValue::from_str(&state.instance_id).unwrap(),
        );
        headers.insert("x-mold-operation-id", HeaderValue::from_str(&id).unwrap());
        let request = serde_json::from_value(serde_json::json!({"generation":{"model":"missing-model","prompt":"exact \\n bytes","width":512,"height":512,"steps":4},"duration_ms":"30000","validation_revision":"old-proof"})).unwrap();
        let error = create(State(state.clone()), headers, None, None, Json(request))
            .await
            .unwrap_err();
        assert_eq!(error.code, "VIDEO_OPERATION_CANCELLED");
        assert!(
            mold_db::video_jobs::get(crate::video_jobs::database(&state).unwrap(), &id)
                .unwrap()
                .is_none()
        );
    }
}
