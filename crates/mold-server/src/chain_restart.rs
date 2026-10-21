//! Private, revision-fenced export for a NEW attempt. Never changes the source.
use crate::{routes::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    Json,
};
use mold_core::chain::ChainRequest;
use mold_core::chain_job::{ChainJobManifest, ChainJobState};
use mold_core::job_restart::{
    ChainRestartExport, ChainRestartOptions, RestartAvailability, RestartExportRequest,
};
use sha2::{Digest, Sha256};

fn refusal(message: &str) -> ApiError {
    ApiError::with_code(message, "CHAIN_RESTART_UNAVAILABLE", StatusCode::CONFLICT)
}

fn options_for(
    instance: &str,
    row: &mold_db::chain_jobs::ChainJobRow,
    manifest: Option<&ChainJobManifest>,
) -> Result<ChainRestartOptions, ApiError> {
    let request =
        manifest.and_then(|manifest| crate::chain_job_runner::effective_request(manifest).ok());
    let reason = if !matches!(
        row.state,
        ChainJobState::Failed | ChainJobState::Paused | ChainJobState::Interrupted
    ) {
        Some("Only a failed or paused video can start a new attempt.")
    } else if request.is_none() {
        Some("The original request is unavailable. Only its recovery data can be deleted.")
    } else if request
        .as_ref()
        .is_some_and(|request| request.continue_from.is_some())
    {
        Some("This video retains a predecessor's motion. Resume it on its original machine; portable restart is unavailable.")
    } else {
        None
    };
    let revision = serde_json::to_vec(&(instance, &row.id, row.updated_at_ms, row.state, manifest))
        .map_err(|_| ApiError::internal("Could not inspect original video"))?;
    let availability = RestartAvailability {
        supported: reason.is_none(),
        reason: reason.map(str::to_owned),
    };
    let discardable = matches!(
        row.state,
        ChainJobState::Failed | ChainJobState::Interrupted
    );
    Ok(ChainRestartOptions {
        source_instance_id: instance.into(),
        source_revision: format!("{:x}", Sha256::digest(revision)),
        restart: availability.clone(),
        discard: Some(RestartAvailability { supported: discardable, reason: (!discardable).then(|| "Only failed or interrupted recovery data can be deleted. Cancel paused work first.".into()) }),
        send: if availability.supported && request.as_ref().is_some_and(|request| request.stages.iter().any(|stage| !stage.loras.is_empty())) {
            RestartAvailability { supported: false, reason: Some("This video uses machine-local adapters. Restart on this machine or select adapters on the destination.".into()) }
        } else { availability },
    })
}

#[utoipa::path(post, path = "/api/chain-jobs/{id}/discard", tag = "chain-jobs",
    params(("id" = String, Path, description = "Failed or interrupted recovery job")),
    request_body = mold_core::job_restart::DiscardChainJobRequest,
    responses((status = 204, description = "Recovery data deleted; Library outputs retained"), (status = 409, description = "Source changed or no longer discardable")))]
pub(crate) async fn discard(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(authority): Json<mold_core::job_restart::DiscardChainJobRequest>,
) -> Result<StatusCode, ApiError> {
    if authority.source_instance_id != *state.instance_id {
        return Err(ApiError::with_code(
            "The source installation changed. Reopen the video.",
            "CHAIN_RESTART_REVISION_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    let handle = state
        .chain_jobs
        .as_ref()
        .ok_or_else(|| refusal("Durable video jobs are unavailable."))?;
    {
        let guard = handle.lock_job(&id).await;
        let db = state
            .metadata_db
            .as_ref()
            .as_ref()
            .ok_or_else(|| refusal("Durable video jobs are unavailable."))?;
        crate::video_job_child_authority::require_public_chain(db, &id)?;
        let Some(row) = mold_db::chain_jobs::get_job(db, &id)
            .map_err(|_| ApiError::internal("Could not read recovery job"))?
        else {
            return Ok(StatusCode::NO_CONTENT);
        };
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).ok();
        let options = options_for(&state.instance_id, &row, manifest.as_ref())?;
        if options.source_revision != authority.source_revision {
            return Err(ApiError::with_code(
                "The video changed. Reopen it before deleting recovery data.",
                "CHAIN_RESTART_REVISION_CHANGED",
                StatusCode::CONFLICT,
            ));
        }
        if !options
            .discard
            .is_some_and(|availability| availability.supported)
        {
            return Err(ApiError::with_code(
                "This job is no longer failed or interrupted. Nothing was deleted.",
                "CHAIN_DISCARD_UNAVAILABLE",
                StatusCode::CONFLICT,
            ));
        }
        let root = crate::routes_chain_jobs::jobs_root()?;
        let metadata = state.metadata_db.clone();
        let deleting = id.clone();
        tokio::task::spawn_blocking(move || {
            // Deletion and retained-input release outlive a dropped HTTP future.
            let _guard = guard;
            crate::chain_source_media::release_all(&root, &row.job_dir).map_err(|error| {
                ApiError::internal(format!("Could not release retained inputs: {error:#}"))
            })?;
            if row.job_dir.exists() {
                std::fs::remove_dir_all(&row.job_dir).map_err(|error| {
                    ApiError::internal(format!("Could not delete recovery files: {error}"))
                })?;
            }
            let db = metadata
                .as_ref()
                .as_ref()
                .ok_or_else(|| refusal("Durable video jobs are unavailable."))?;
            if !mold_db::chain_jobs::delete_job_not_running(db, &deleting)
                .map_err(|_| ApiError::internal("Could not delete recovery job"))?
            {
                return Err(ApiError::with_code(
                    "The video changed. Nothing more was deleted.",
                    "CHAIN_DISCARD_UNAVAILABLE",
                    StatusCode::CONFLICT,
                ));
            }
            Ok::<(), ApiError>(())
        })
        .await
        .map_err(|_| ApiError::internal("Recovery deletion did not finish"))??;
    }
    handle.cleanup_deleted(&id);
    handle.remove_job_lock(&id);
    Ok(StatusCode::NO_CONTENT)
}

fn load_row(state: &AppState, id: &str) -> Result<mold_db::chain_jobs::ChainJobRow, ApiError> {
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .ok_or_else(|| refusal("Durable video jobs are unavailable."))?;
    crate::video_job_child_authority::require_public_chain(db, id)?;
    let row = mold_db::chain_jobs::get_job(db, id)
        .map_err(|_| ApiError::internal("Could not read original video"))?
        .ok_or_else(|| {
            ApiError::with_code(
                "Original video job not found",
                "CHAIN_JOB_NOT_FOUND",
                StatusCode::NOT_FOUND,
            )
        })?;
    Ok(row)
}
fn load(
    state: &AppState,
    id: &str,
) -> Result<(mold_db::chain_jobs::ChainJobRow, ChainJobManifest), ApiError> {
    let row = load_row(state, id)?;
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir)
        .map_err(|_| refusal("The original video manifest is unavailable."))?;
    Ok((row, manifest))
}

#[utoipa::path(get, path = "/api/chain-jobs/{id}/restart-options", tag = "chain-jobs",
    params(("id" = String, Path, description = "Original durable video job")),
    responses((status = 200, body = mold_core::job_restart::ChainRestartOptions)))]
pub(crate) async fn options(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let handle = state
        .chain_jobs
        .as_ref()
        .ok_or_else(|| refusal("Durable video jobs are unavailable."))?;
    let _guard = handle.lock_job(&id).await;
    let row = load_row(&state, &id)?;
    let manifest = ChainJobManifest::read_from_dir(&row.job_dir).ok();
    Ok((
        [(header::CACHE_CONTROL, "private, no-store")],
        Json(options_for(&state.instance_id, &row, manifest.as_ref())?),
    ))
}

fn portable_request(
    mut request: ChainRequest,
    db: &mold_db::MetadataDb,
    portable: bool,
) -> Result<ChainRequest, ApiError> {
    if request.continue_from.is_some()
        || (portable && request.stages.iter().any(|stage| !stage.loras.is_empty()))
    {
        return Err(refusal(
            "Original conditioning is not portable. Resume on the original machine.",
        ));
    }
    request
        .validate_inline_media_budget()
        .map_err(|_| refusal("Original media exceeds the portable input budget."))?;
    if request
        .stages
        .iter()
        .flat_map(|stage| stage.generation_references.iter().flatten())
        .any(|reference| {
            !matches!(
                reference.media(),
                mold_core::GenerationReferenceAuthority::Inline { .. }
            )
        })
    {
        return Err(refusal("Original reference bytes are unavailable."));
    }
    crate::routes::resolve_collection_reference(db, &mut request.collection);
    if let Some(collection) = &mut request.collection {
        if collection.name.is_none() && collection.id.is_some() {
            return Err(refusal(
                "The original collection is unavailable. Reuse settings to choose another.",
            ));
        }
        collection.id = None;
    }
    request.placement = None;
    request.batch_id = None;
    request.batch_index = None;
    request.batch_count = None;
    for stage in &mut request.stages {
        stage.authored_media_sha256 = None;
    }
    Ok(request)
}

#[utoipa::path(post, path = "/api/chain-jobs/{id}/restart-export", tag = "chain-jobs",
    params(("id" = String, Path, description = "Original durable video job")),
    request_body = mold_core::job_restart::RestartExportRequest,
    responses((status = 200, body = mold_core::job_restart::ChainRestartExport), (status = 409, description = "Original changed or is not portable")))]
pub(crate) async fn export(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(authority): Json<RestartExportRequest>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let handle = state
        .chain_jobs
        .as_ref()
        .ok_or_else(|| refusal("Durable video jobs are unavailable."))?;
    let _guard = handle.lock_job(&id).await;
    let (row, manifest) = load(&state, &id)?;
    let options = options_for(&state.instance_id, &row, Some(&manifest))?;
    if authority.source_instance_id != options.source_instance_id
        || authority.source_revision != options.source_revision
    {
        return Err(ApiError::with_code(
            "The original video changed. Refresh before starting another attempt.",
            "CHAIN_RESTART_REVISION_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    if authority.destination_instance_id.trim().is_empty() {
        return Err(refusal(
            "Capture the destination installation before exporting.",
        ));
    }
    let portable = authority.destination_instance_id != options.source_instance_id;
    let availability = if portable {
        &options.send
    } else {
        &options.restart
    };
    if !availability.supported {
        return Err(refusal(
            availability
                .reason
                .as_deref()
                .unwrap_or("Restart unavailable"),
        ));
    }
    let manifest_io = manifest.clone();
    let dir = row.job_dir.clone();
    let request = tokio::task::spawn_blocking(move || {
        crate::chain_job_runner::effective_request_with_media(&manifest_io, &dir)
    })
    .await
    .map_err(|_| ApiError::internal("Could not read original inputs"))?
    .map_err(|_| refusal("Exact original inputs are unavailable. Nothing was restarted."))?;
    // Actor lock fences amendment, deletion and resume while media is authenticated.
    let (latest, current_manifest) = load(&state, &id)?;
    if options_for(&state.instance_id, &latest, Some(&current_manifest))?.source_revision
        != authority.source_revision
    {
        return Err(refusal(
            "The original video changed while reading its inputs. Refresh and retry.",
        ));
    }
    let db = state
        .metadata_db
        .as_ref()
        .as_ref()
        .ok_or_else(|| refusal("Durable jobs are unavailable."))?;
    let request = portable_request(request, db, portable)?;
    let request_json = serde_json::to_string(
        &mold_core::prompt_text::protect_chain_request_for_wire(&request),
    )
    .map_err(|_| ApiError::internal("Could not encode original video inputs"))?;
    Ok((
        [(header::CACHE_CONTROL, "private, no-store")],
        Json(ChainRestartExport {
            destination_instance_id: authority.destination_instance_id,
            source_instance_id: options.source_instance_id,
            source_revision: options.source_revision,
            request_json,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (mold_db::chain_jobs::ChainJobRow, ChainJobManifest) {
        let request: ChainRequest = serde_json::from_value(serde_json::json!({"model":"ltx-2-19b-distilled:fp8","width":512,"height":512,"steps":4,"guidance":1.0,"stages":[{"prompt":"original","frames":97}]})).unwrap();
        let manifest = ChainJobManifest::new("original".into(), 1, &request).unwrap();
        let row = mold_db::chain_jobs::ChainJobRow {
            id: "original".into(),
            state: ChainJobState::Failed,
            model: request.model.clone(),
            request_json: serde_json::to_string(&request).unwrap(),
            job_dir: "/unused".into(),
            stage_count: 1,
            current_stage: 0,
            error: Some("original failure".into()),
            created_at_ms: 1,
            updated_at_ms: 2,
            finalized_at_ms: None,
        };
        (row, manifest)
    }

    #[tokio::test]
    async fn discard_refuses_reserved_duration_children_before_lookup_or_deletion() {
        let mut state = AppState::for_tests();
        state.metadata_db =
            std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.chain_jobs = Some(std::sync::Arc::new(
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        ));
        let child = uuid::Uuid::new_v4().to_string();
        state.metadata_db.as_ref().as_ref().unwrap().with_conn(|conn| {
            conn.execute("INSERT INTO video_job_child_owners(child_id,parent_id) VALUES(?1,'removed-parent')", [&child])?;
            Ok(())
        }).unwrap();
        let error = discard(
            State(state.clone()),
            Path(child),
            Json(mold_core::job_restart::DiscardChainJobRequest {
                source_instance_id: state.instance_id.to_string(),
                source_revision: "captured".into(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, "VIDEO_JOB_CHILD_OWNED");
    }

    #[tokio::test]
    async fn discard_refuses_replaced_owner_revision_and_resumed_work_without_deleting_inputs() {
        use mold_core::job_restart::DiscardChainJobRequest;
        for (lifecycle, wrong_owner, wrong_revision) in [
            (ChainJobState::Failed, true, false),
            (ChainJobState::Failed, false, true),
            (ChainJobState::Queued, false, false),
            (ChainJobState::Running, false, false),
            (ChainJobState::Paused, false, false),
            (ChainJobState::Completed, false, false),
            (ChainJobState::Cancelled, false, false),
        ] {
            let home = tempfile::tempdir().unwrap();
            let (mut row, manifest) = fixture();
            row.state = lifecycle;
            row.job_dir = home.path().join("job");
            std::fs::create_dir_all(&row.job_dir).unwrap();
            manifest.write_atomic(&row.job_dir).unwrap();
            let original = std::fs::read(row.job_dir.join("manifest.toml")).unwrap();
            let mut state = AppState::for_tests();
            state.metadata_db =
                std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            state.chain_jobs = Some(std::sync::Arc::new(
                crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
            ));
            let db = state.metadata_db.as_ref().as_ref().unwrap();
            mold_db::chain_jobs::insert_job(db, &row).unwrap();
            let captured = options_for(&state.instance_id, &row, Some(&manifest)).unwrap();
            let result = discard(
                State(state.clone()),
                Path(row.id.clone()),
                Json(DiscardChainJobRequest {
                    source_instance_id: if wrong_owner {
                        "replacement".into()
                    } else {
                        captured.source_instance_id
                    },
                    source_revision: if wrong_revision {
                        "stale".into()
                    } else {
                        captured.source_revision
                    },
                }),
            )
            .await;
            assert!(result.is_err(), "discard must refuse {lifecycle:?} with changed owner={wrong_owner}, revision={wrong_revision}");
            assert!(mold_db::chain_jobs::get_job(db, &row.id).unwrap().is_some());
            assert_eq!(
                std::fs::read(row.job_dir.join("manifest.toml")).unwrap(),
                original
            );
        }
    }
    #[tokio::test]
    async fn explicit_discard_removes_only_settled_recovery_data_even_with_missing_manifest() {
        use mold_core::job_restart::DiscardChainJobRequest;
        for (lifecycle, has_manifest) in [
            (ChainJobState::Failed, true),
            (ChainJobState::Interrupted, false),
        ] {
            let home = tempfile::tempdir().unwrap();
            let (mut row, manifest) = fixture();
            row.state = lifecycle;
            row.job_dir = home.path().join("job");
            std::fs::create_dir_all(&row.job_dir).unwrap();
            if has_manifest {
                manifest.write_atomic(&row.job_dir).unwrap();
            } else {
                std::fs::write(row.job_dir.join("manifest.toml"), b"broken manifest").unwrap();
            }
            let gallery = home.path().join("saved-library.mp4");
            std::fs::write(&gallery, b"saved output").unwrap();
            let mut state = AppState::for_tests();
            state.metadata_db =
                std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
            state.chain_jobs = Some(std::sync::Arc::new(
                crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
            ));
            let db = state.metadata_db.as_ref().as_ref().unwrap();
            mold_db::chain_jobs::insert_job(db, &row).unwrap();
            let loaded = ChainJobManifest::read_from_dir(&row.job_dir).ok();
            let captured = options_for(&state.instance_id, &row, loaded.as_ref()).unwrap();
            assert!(captured.discard.as_ref().unwrap().supported);
            if !has_manifest {
                assert!(!captured.restart.supported);
            }
            for _ in 0..2 {
                assert_eq!(
                    discard(
                        State(state.clone()),
                        Path(row.id.clone()),
                        Json(DiscardChainJobRequest {
                            source_instance_id: captured.source_instance_id.clone(),
                            source_revision: captured.source_revision.clone(),
                        })
                    )
                    .await
                    .unwrap(),
                    StatusCode::NO_CONTENT
                );
            }
            assert!(mold_db::chain_jobs::get_job(db, &row.id).unwrap().is_none());
            assert!(!row.job_dir.exists());
            assert_eq!(std::fs::read(gallery).unwrap(), b"saved output");
        }
    }

    #[tokio::test]
    async fn revision_fenced_export_is_private_and_leaves_source_untouched() {
        use axum::response::IntoResponse;
        let home = tempfile::tempdir().unwrap();
        let (mut row, manifest) = fixture();
        row.job_dir = home.path().join("original");
        std::fs::create_dir_all(&row.job_dir).unwrap();
        manifest.write_atomic(&row.job_dir).unwrap();
        let before = std::fs::read(row.job_dir.join("manifest.toml")).unwrap();
        let mut state = AppState::for_tests();
        state.metadata_db =
            std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.chain_jobs = Some(std::sync::Arc::new(
            crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests(),
        ));
        let db = state.metadata_db.as_ref().as_ref().unwrap();
        mold_db::chain_jobs::insert_job(db, &row).unwrap();
        let authority = options_for(&state.instance_id, &row, Some(&manifest)).unwrap();
        let stale = RestartExportRequest {
            destination_instance_id: state.instance_id.as_ref().clone(),
            source_instance_id: "different-installation".into(),
            source_revision: authority.source_revision.clone(),
        };
        assert!(
            export(State(state.clone()), Path(row.id.clone()), Json(stale))
                .await
                .is_err()
        );
        let response = export(
            State(state.clone()),
            Path(row.id.clone()),
            Json(RestartExportRequest {
                destination_instance_id: state.instance_id.as_ref().clone(),
                source_instance_id: authority.source_instance_id,
                source_revision: authority.source_revision,
            }),
        )
        .await
        .unwrap()
        .into_response();
        assert_eq!(
            response.headers()[header::CACHE_CONTROL],
            "private, no-store"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let exported: ChainRestartExport = serde_json::from_slice(&bytes).unwrap();
        let mut decoded: ChainRequest = serde_json::from_str(&exported.request_json).unwrap();
        decoded.normalize_prompt_newlines();
        assert_eq!(decoded.stages[0].prompt, "original");
        assert_eq!(
            mold_db::chain_jobs::get_job(db, &row.id).unwrap().unwrap(),
            row
        );
        assert_eq!(
            std::fs::read(row.job_dir.join("manifest.toml")).unwrap(),
            before
        );
    }

    #[tokio::test]
    async fn new_attempt_refuses_replaced_destination_before_admission() {
        let state = AppState::for_tests();
        let (_, manifest) = fixture();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-mold-destination-instance",
            "another-installation".parse().unwrap(),
        );
        let result = crate::routes_chain_jobs::create_chain_job(
            State(state),
            headers,
            Json(manifest.request().unwrap()),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "CHAIN_DESTINATION_CHANGED");
    }

    #[test]
    fn raw_restart_wire_preserves_full_range_seeds_and_literal_prompt_text() {
        let (_, manifest) = fixture();
        let mut request = manifest.request().unwrap();
        request.seed = Some(u64::MAX);
        request.stages[0].seed_offset = Some(u64::MAX - 1);
        request.stages[0].prompt = r"literal \n and \t".into();
        let wire = serde_json::to_string(&mold_core::prompt_text::protect_chain_request_for_wire(
            &request,
        ))
        .unwrap();
        let mut decoded: ChainRequest = serde_json::from_str(&wire).unwrap();
        decoded.normalize_prompt_newlines();
        assert_eq!(decoded.seed, Some(u64::MAX));
        assert_eq!(decoded.stages[0].seed_offset, Some(u64::MAX - 1));
        assert_eq!(decoded.stages[0].prompt, request.stages[0].prompt);
    }

    #[test]
    fn restart_export_is_revision_and_installation_bound() {
        let (mut row, manifest) = fixture();
        let first = options_for("installation-a", &row, Some(&manifest)).unwrap();
        assert!(first.restart.supported && first.send.supported);
        row.updated_at_ms += 1;
        assert_ne!(
            first.source_revision,
            options_for("installation-a", &row, Some(&manifest))
                .unwrap()
                .source_revision
        );
        assert_ne!(
            first.source_revision,
            options_for("installation-b", &row, Some(&manifest))
                .unwrap()
                .source_revision
        );
    }
    #[test]
    fn restart_refuses_live_and_linked_sources_without_lossy_fallback() {
        let (mut row, mut manifest) = fixture();
        row.state = ChainJobState::Running;
        assert!(
            !options_for("a", &row, Some(&manifest))
                .unwrap()
                .restart
                .supported
        );
        row.state = ChainJobState::Failed;
        let mut request = manifest.request().unwrap();
        request.continue_from = Some(mold_core::chain::ChainContinuation {
            job_id: "predecessor".into(),
            finalize_take: 1,
        });
        manifest.request_json = serde_json::to_string(&request).unwrap();
        let options = options_for("a", &row, Some(&manifest)).unwrap();
        assert!(!options.send.supported);
        assert!(options.send.reason.unwrap().contains("predecessor"));
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        assert!(portable_request(request, &db, true).is_err());
    }
    #[test]
    fn local_adapters_restart_on_same_installation_but_never_portably() {
        let (row, mut manifest) = fixture();
        let mut request = manifest.request().unwrap();
        request.stages[0].loras.push(mold_core::chain::LoraSpec {
            path: "/models/original.safetensors".into(),
            scale: 0.5,
            name: Some("original".into()),
        });
        manifest.request_json = serde_json::to_string(&request).unwrap();
        let options = options_for("original-installation", &row, Some(&manifest)).unwrap();
        assert!(options.restart.supported);
        assert!(!options.send.supported);
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        assert!(portable_request(request.clone(), &db, true).is_err());
        assert_eq!(
            portable_request(request.clone(), &db, false)
                .unwrap()
                .stages[0]
                .loras,
            request.stages[0].loras
        );
    }

    #[test]
    fn restart_preserves_authored_pixels_and_clears_only_host_authorities() {
        let (_, manifest) = fixture();
        let mut request = manifest.request().unwrap();
        request.stages[0].source_image = Some(vec![1, 2, 3]);
        request.stages[0].authored_media_sha256 = Some("old-stamp".into());
        request.batch_id = Some("old-batch".into());
        request.batch_index = Some(1);
        request.batch_count = Some(1);
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let exported = portable_request(request.clone(), &db, true).unwrap();
        assert_eq!(
            exported.stages[0].source_image,
            request.stages[0].source_image
        );
        assert_eq!(exported.stages[0].prompt, "original");
        assert!(
            exported.batch_id.is_none()
                && exported.batch_index.is_none()
                && exported.batch_count.is_none()
        );
        assert!(exported.stages[0].authored_media_sha256.is_none());
    }
}
