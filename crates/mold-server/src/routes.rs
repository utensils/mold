use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Extension, Path, Query, Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, patch, post, put},
    Json, Router,
};
use base64::Engine as _;
use mold_core::{
    types::GpuSelection, ActiveGenerationStatus, DeviceAdminState, DeviceMutationRequest,
    DeviceState, DiskUsage, GenerateRequest, GpuWorkerState, ModelInfoExtended, ResourceSnapshot,
    ServerStatus, SseErrorEvent, SseProgressEvent,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt as _;
use utoipa::OpenApi;

use crate::model_manager;
use crate::state::{AppState, SseCompletionPayload, SseMessage};

// ── ApiError — structured JSON error response ────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Machine-readable detail for a `LICENSE_NOT_ACCEPTED` refusal.
    ///
    /// Additive and absent on every other error, so a UI can render its own
    /// acceptance prompt from structured data instead of parsing `error`.
    ///
    /// Boxed because `ApiError` is the `Err` half of most handler signatures:
    /// inlining five `String`s here grows EVERY `Result` in the crate and
    /// trips clippy's `result_large_err`. The box costs one pointer on the
    /// overwhelmingly common `None` path, and `Box<T>` serializes
    /// transparently so the wire shape is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<Box<mold_core::LicenseRefusal>>,
    #[serde(skip)]
    status: StatusCode,
}

impl ApiError {
    pub fn model_activation(error: mold_core::ModelActivationError) -> Self {
        // A licensing refusal and a missing engine arm are different answers
        // and must not share a status code. 451 asserts a legal obstacle;
        // returning it because mold has not written the loader tells the user
        // to go read a license about a problem that is ours.
        let (code, status) = match error.refusal() {
            mold_core::ActivationRefusal::ComplianceGated => (
                mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED,
                StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS,
            ),
            mold_core::ActivationRefusal::RuntimeUnavailable(_) => (
                mold_core::MINIMAX_H3_RUNTIME_UNAVAILABLE,
                StatusCode::NOT_IMPLEMENTED,
            ),
        };
        Self::with_code(error.to_string(), code, status)
    }

    pub fn validation(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "VALIDATION_ERROR", StatusCode::UNPROCESSABLE_ENTITY)
    }

    pub fn reference(error: mold_core::minimax_h3::ReferenceContractError) -> Self {
        Self {
            error: error.message,
            code: error.code.to_string(),
            reference: error.reference,
            field: error.field.map(str::to_string),
            license: None,
            status: StatusCode::UNPROCESSABLE_ENTITY,
        }
    }

    pub(crate) fn structured(
        msg: impl Into<String>,
        code: impl Into<String>,
        status: StatusCode,
        reference: Option<u32>,
        field: Option<String>,
    ) -> Self {
        Self {
            error: msg.into(),
            code: code.into(),
            reference,
            field,
            license: None,
            status,
        }
    }

    /// A download refused because a third-party license has not been accepted
    /// on THIS server.
    ///
    /// `403` rather than `422`: the request is well-formed and the model
    /// exists — the server is declining to act until consent is on record.
    /// The structured `license` payload lets a client offer acceptance and
    /// retry with `accept_licenses`, which a prose-only error could not.
    /// The caller accepted a different revision of a license this server
    /// knows.
    ///
    /// `409` rather than `400`: the request is well-formed and the license is
    /// real — the two sides disagree about the current terms, which a client
    /// resolves by re-reading `GET /api/licenses` and accepting again. The
    /// structured payload carries THIS server's `url`/`sha256`/`canonical` so
    /// it can display them without a second round trip.
    pub fn license_terms_mismatch(
        license: &mold_core::license_acceptance::ThirdPartyLicense,
    ) -> Self {
        Self {
            error: format!(
                "the accepted terms for '{}' are not the ones this server pins.\n\n  {}\n  Terms (pinned): {}\n  sha256: {}\n  Project terms: {}\n\nReview those terms and accept again.",
                license.id, license.summary, license.url, license.sha256, license.canonical
            ),
            code: mold_core::LICENSE_TERMS_MISMATCH.to_string(),
            reference: None,
            field: None,
            license: Some(Box::new(mold_core::license_acceptance::refusal(license))),
            status: StatusCode::CONFLICT,
        }
    }

    pub fn license_not_accepted(
        model: &str,
        license: &mold_core::license_acceptance::ThirdPartyLicense,
    ) -> Self {
        Self {
            error: mold_core::license_acceptance::acceptance_required_message(model, license),
            code: mold_core::LICENSE_NOT_ACCEPTED.to_string(),
            reference: None,
            field: None,
            license: Some(Box::new(mold_core::license_acceptance::refusal(license))),
            status: StatusCode::FORBIDDEN,
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "MODEL_NOT_FOUND", StatusCode::NOT_FOUND)
    }

    pub fn unknown_model(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "UNKNOWN_MODEL", StatusCode::BAD_REQUEST)
    }

    pub fn inference(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "INFERENCE_ERROR", StatusCode::INTERNAL_SERVER_ERROR)
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "INTERNAL_ERROR", StatusCode::INTERNAL_SERVER_ERROR)
    }

    pub fn internal_with_status(msg: impl Into<String>, status: StatusCode) -> Self {
        Self::with_code(msg, "INTERNAL_ERROR", status)
    }

    pub fn with_code(msg: impl Into<String>, code: impl Into<String>, status: StatusCode) -> Self {
        Self {
            error: msg.into(),
            code: code.into(),
            reference: None,
            field: None,
            license: None,
            status,
        }
    }

    /// The HTTP status this error renders with.
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn queue_job_not_found(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "QUEUE_JOB_NOT_FOUND", StatusCode::NOT_FOUND)
    }

    pub fn queue_job_running(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "QUEUE_JOB_RUNNING", StatusCode::CONFLICT)
    }

    /// Job cancelled while queued. 499 is the de-facto "client closed
    /// request" status (nginx); we reuse it for "request cancelled before
    /// the server did any work" so clients can distinguish cancellation
    /// from real inference failures.
    pub fn cancelled(msg: impl Into<String>) -> Self {
        Self::with_code(
            msg,
            "CANCELLED",
            StatusCode::from_u16(499).expect("499 is a valid status code"),
        )
    }

    pub fn insufficient_memory(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "INSUFFICIENT_MEMORY", StatusCode::SERVICE_UNAVAILABLE)
    }

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "FORBIDDEN", StatusCode::FORBIDDEN)
    }

    pub fn queue_full(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "QUEUE_FULL", StatusCode::SERVICE_UNAVAILABLE)
    }

    /// The host is going down. Distinct from `QUEUE_FULL` so a client can tell
    /// "come back in a second" from "this instance is restarting"; both carry
    /// `Retry-After`.
    pub fn server_restarting(msg: impl Into<String>) -> Self {
        Self::with_code(msg, "SERVER_RESTARTING", StatusCode::SERVICE_UNAVAILABLE)
    }

    pub fn generation_unavailable(msg: impl Into<String>) -> Self {
        Self::with_code(
            msg,
            "GENERATION_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        )
    }

    pub fn no_schedulable_device(msg: impl Into<String>) -> Self {
        Self::with_code(
            msg,
            "NO_SCHEDULABLE_DEVICE",
            StatusCode::SERVICE_UNAVAILABLE,
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status;
        // On queue-full (503), hint clients to retry with a short delay.
        if self.code == "QUEUE_FULL" || self.code == "SERVER_RESTARTING" {
            let mut headers = HeaderMap::new();
            headers.insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
            return (status, headers, Json(self)).into_response();
        }
        (status, Json(self)).into_response()
    }
}

// Re-export for tests — the canonical implementation lives in queue.rs.
#[cfg(test)]
use crate::queue::clean_error_message;

#[derive(OpenApi)]
#[openapi(
    paths(
        generate,
        generate_stream,
        admit_generation_batch,
        get_generation_batch,
        cancel_generation_batch,
        generation_batch_events,
        get_generation_batch_by_client,
        reconcile_generation_batches,
        generate_placement_preview,
        crate::reference_uploads::create_reference_upload_session,
        crate::reference_uploads::upload_reference,
        crate::reference_uploads::cancel_reference_upload_session,
        crate::gallery_source_media::create_reuse_session,
        expand_prompt,
        remix_prompt,
        list_models,
        crate::catalog_api::list_loras,
        load_model,
        pull_model_endpoint,
        unload_model,
        delete_model,
        create_gallery_media_token,
        gallery_export_options,
        export_gallery_video,
        create_pairing_session,
        claim_pairing_session,
        list_paired_clients,
        revoke_paired_client,
        import_gallery_file,
        crate::gallery_organization::patch_gallery_image,
        crate::gallery_organization::organize_gallery,
        crate::gallery_organization::mutate_gallery_bulk,
        crate::gallery_organization::list_collections,
        crate::gallery_organization::create_collection,
        crate::gallery_organization::get_collection,
        crate::gallery_organization::update_collection,
        crate::gallery_organization::delete_collection,
        crate::gallery_organization::put_collection_items,
        crate::gallery_organization::list_tags,
        crate::gallery_organization::rename_tag,
        crate::gallery_organization::delete_tag,
        crate::gallery_trash::delete_gallery_image,
        crate::gallery_trash::trash_gallery_files,
        crate::gallery_trash::restore_gallery_files,
        crate::gallery_trash::delete_gallery_files_forever,
        crate::gallery_trash::empty_gallery_trash,
        crate::gallery_trash::sweep_gallery_trash,
        crate::gallery_source_media::inventory,
        crate::gallery_source_media::download,
        crate::queue_retention::sweep_held_queue,
        crate::queue_retention::sweep_settled_batches,
        server_status,
        list_devices,
        patch_device,
        list_queue,
        get_queue_job,
        get_queue_job_preview,
        patch_queue_job,
        cancel_queue_job,
        retry_queue_job,
        pause_queue_job,
        resume_queue_job,
        pause_queue,
        resume_queue,
        cancel_all_queue,
        list_history,
        delete_history,
        crate::routes_config::list_config,
        crate::routes_config::get_config_key,
        crate::routes_config::put_config_key,
        crate::routes_config::delete_config_key,
        crate::routes_config::list_config_profiles,
        crate::routes_config::put_config_profile,
        health,
        discovery_peers,
        capabilities_chain_limits,
        capabilities_ltx2_control_adapters,
        capabilities_ltx2_camera_controls,
        list_licenses_endpoint,
        stream_events,
        crate::routes_chain::validate_chain,
        crate::routes_chain_jobs::create_chain_job,
        crate::routes_chain_jobs::preview_chain_job_placement,
        crate::routes_chain_jobs::list_chain_jobs,
        crate::routes_chain_jobs::get_chain_job,
        crate::routes_chain_jobs::chain_job_events,
        crate::routes_chain_jobs::resume_chain_job,
        crate::routes_chain_jobs::retake_chain_job,
        crate::routes_chain_jobs::amend_chain_job,
        crate::routes_chain_jobs::cancel_chain_job,
        crate::routes_chain_jobs::cancel_chain_job_mutation,
        crate::routes_chain_jobs::delete_chain_job,
        crate::routes_chain_jobs::gc_chain_jobs,
        crate::routes_chain_jobs::chain_job_stage_preview,
        crate::routes_chain_jobs::chain_job_stage_media,
        crate::routes_chain_jobs::create_chain_job_stage_media_token,
        crate::routes_activity::list_active_work,
    ),
    components(schemas(
        mold_core::GenerateRequest,
        mold_core::CollectionRef,
        mold_core::Ltx2ControlAdapterInfo,
        mold_core::Ltx2CameraControlInfo,
        mold_core::Ltx2GuidanceOverrides,
        mold_core::GenerateResponse,
        mold_core::GenerationBatchStatus,
        mold_core::GenerationBatchChild,
        mold_core::GenerationBatchResult,
        mold_core::GenerationBatchStatusRequest,
        mold_core::GenerationBatchStatusResponse,
        mold_core::GenerationBatchMissing,
        mold_core::GenerationPlacementPreviewRequest,
        mold_core::GenerationPlacementPreview,
        mold_core::GenerationPlacementCandidate,
        mold_core::ChainStagePlacementCandidate,
        mold_core::ReferenceUploadSessionRequest,
        mold_core::ReferenceUploadSessionResponse,
        mold_core::ReferenceUploadSlot,
        mold_core::ReferenceUploadCompleteResponse,
        mold_core::minimax_h3::GenerationReferencePreparedShape,
        crate::routes_chain_jobs::ChainPlacementPreviewRequest,
        mold_core::ExpandRequest,
        mold_core::ExpandResponse,
        mold_core::RemixRequest,
        mold_core::RemixResponse,
        mold_core::RemixVariant,
        mold_core::RemixDimension,
        mold_core::ImageData,
        mold_core::MeshData,
        mold_core::MeshCapabilities,
        mold_core::OutputFormat,
        mold_core::ModelInfo,
        mold_core::GenerationProfileSet,
        mold_core::GenerationRecipeProfile,
        mold_core::GenerationDefaultsProfile,
        mold_core::GenerationCapabilitiesProfile,
        mold_core::ResolutionProfile,
        mold_core::ResolutionDomain,
        mold_core::ResolutionPreset,
        mold_core::AspectGroup,
        mold_core::IntegerControl,
        mold_core::FloatControl,
        mold_core::ControlMode,
        mold_core::TemporalProfile,
        mold_core::FpsControl,
        mold_core::RecipeSelector,
        mold_core::ProfileProvenance,
        mold_core::ProvenanceKind,
        mold_core::LoraInfo,
        mold_core::ServerStatus,
        mold_core::HealthStatus,
        mold_core::HealthState,
        mold_core::DurableMediaStatus,
        PairingSessionResponse,
        PairingClaimRequest,
        PairingClaimResponse,
        PairedClientsResponse,
        PairedClientResponse,
        mold_core::ActiveGenerationStatus,
        mold_core::GpuInfo,
        mold_core::DeviceState,
        mold_core::DeviceInfo,
        mold_core::DeviceKind,
        mold_core::DeviceAdminState,
        mold_core::DeviceHealth,
        mold_core::DeviceActivity,
        mold_core::DeviceMemoryInfo,
        mold_core::DeviceTelemetry,
        mold_core::DiskUsage,
        mold_core::ActiveWorkItem,
        mold_core::ActiveWorkSnapshot,
        mold_core::DiscoveryPeer,
        mold_core::SseProgressEvent,
        mold_core::SseCompleteEvent,
        mold_core::SseErrorEvent,
        mold_core::ChainRequest,
        mold_core::ChainResponse,
        mold_core::ChainStage,
        mold_core::ChainProgressEvent,
        mold_core::SseChainCompleteEvent,
        mold_core::ChainValidationResponse,
        mold_core::ChainValidationStage,
        mold_core::chain_job::ChainJobSummary,
        mold_core::chain_job::ChainJobStageDetail,
        mold_core::chain_job::ChainJobDetail,
        mold_core::chain_job::ChainJobListing,
        mold_core::chain_job::CreateChainJobResponse,
        mold_core::chain_job::RetakeRequest,
        mold_core::chain_job::AmendRequest,
        mold_core::chain_job::AmendResponse,
        mold_core::chain_job::AmendRecord,
        mold_core::chain_job::ChainJobEvent,
        mold_core::chain_job::FinalizeRecord,
        mold_core::chain_job::RetakeAmendment,
        mold_core::chain_job::ChainJobState,
        mold_core::chain_job::StageState,
        mold_core::chain_job::RetakeMode,
        mold_core::chain_job::GcOutcome,
        mold_core::chain::ChainStageMetadata,
        mold_core::chain::ChainOutputMetadata,
        ModelInfoExtended,
        LoadModelBody,
        UnloadRequest,
        mold_core::ModelRemovalResponse,
        mold_core::KeptComponent,
        QueueJobEntry,
        QueuePatchRequest,
        QueuePauseResponse,
        QueueCancelAllResponse,
        mold_core::HistoryEntry,
        mold_core::HistoryListing,
        mold_core::ConfigEntry,
        mold_core::ConfigListing,
        mold_core::ConfigProfiles,
        crate::routes_config::ConfigSetRequest,
        crate::routes_config::ProfileSetRequest,
        crate::job_registry::JobEntry,
        crate::job_registry::QueueListing,
        crate::chain_limits::ChainLimits,
        GalleryMediaTokenRequest,
        GalleryMediaTokenResponse,
        GalleryExportRequest,
        GalleryExportOptionsResponse,
        GalleryExportFormat,
        GalleryGifPlayback,
        GalleryGifRepeat,
        mold_core::GalleryPatchRequest,
        mold_core::GalleryOrganizeRequest,
        mold_core::GalleryBulkMutationRequest,
        mold_core::GalleryBulkMutationResult,
        mold_core::Collection,
        mold_core::CollectionDetail,
        mold_core::CollectionCreateRequest,
        mold_core::CollectionUpdateRequest,
        mold_core::CollectionItemsRequest,
        mold_core::TagCount,
        mold_core::TagRenameRequest,
        mold_core::TrashFilenamesRequest,
        mold_core::TrashSweepResult,
        mold_core::EmptyTrashResult,
        mold_core::RetainedSourceMediaAvailability,
        mold_core::RetainedSourceMediaMember,
        mold_core::RetainedSourceMediaInventory,
    )),
    tags(
        (name = "generation", description = "Image generation"),
        (name = "models", description = "Model management"),
        (name = "server", description = "Server status and health"),
        (name = "chain-jobs", description = "Durable chained video jobs"),
    ),
    info(
        title = "mold",
        description = "Local AI image generation server — FLUX, SD3.5, SD1.5, SDXL, Z-Image, Flux.2, Qwen-Image",
        version = env!("CARGO_PKG_VERSION"),
    )
)]
pub struct ApiDoc;

pub fn create_router(state: AppState) -> Router {
    // Stateful routes (need AppState) are added first, then .with_state() converts
    // Router<AppState> → Router<()>. Stateless routes (OpenAPI, docs) are merged after.
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/generate/estimate", post(generate_estimate))
        .route(
            "/api/generate/placement-preview",
            post(generate_placement_preview),
        )
        .route("/api/generate/stream", post(generate_stream))
        .route("/api/generation-batches", post(admit_generation_batch))
        .route(
            "/api/generation-batches/by-client/:client_batch_id",
            get(get_generation_batch_by_client),
        )
        .route(
            "/api/generation-batches/status",
            post(reconcile_generation_batches)
                .layer(DefaultBodyLimit::max(GENERATION_BATCH_STATUS_BODY_BYTES)),
        )
        .route(
            "/api/generation-batches/:id",
            get(get_generation_batch).delete(cancel_generation_batch),
        )
        .route(
            "/api/generation-batches/:id/events",
            get(generation_batch_events),
        )
        .route(
            "/api/generate/reference-upload-sessions",
            post(crate::reference_uploads::create_reference_upload_session)
                .delete(crate::reference_uploads::cancel_reference_upload_session)
                .layer(DefaultBodyLimit::max(
                    crate::reference_uploads::MAX_REFERENCE_UPLOAD_SESSION_REQUEST_BYTES,
                )),
        )
        .route(
            "/api/generate/reference-upload",
            put(crate::reference_uploads::upload_reference).layer(DefaultBodyLimit::max(
                usize::try_from(
                    crate::reference_uploads::MAX_REFERENCE_UPLOAD_FILE_BYTES.saturating_add(1),
                )
                .unwrap_or(usize::MAX),
            )),
        )
        .route(
            "/api/generate/chain/validate",
            post(crate::routes_chain::validate_chain),
        )
        .route(
            "/api/chain-jobs",
            post(crate::routes_chain_jobs::create_chain_job)
                .get(crate::routes_chain_jobs::list_chain_jobs),
        )
        .route(
            "/api/chain-jobs/placement-preview",
            post(crate::routes_chain_jobs::preview_chain_job_placement),
        )
        .route(
            "/api/chain-jobs/:id",
            get(crate::routes_chain_jobs::get_chain_job)
                .delete(crate::routes_chain_jobs::delete_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/events",
            get(crate::routes_chain_jobs::chain_job_events),
        )
        .route(
            "/api/chain-jobs/:id/resume",
            post(crate::routes_chain_jobs::resume_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/retake",
            post(crate::routes_chain_jobs::retake_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/amend",
            post(crate::routes_chain_jobs::amend_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/cancel",
            post(crate::routes_chain_jobs::cancel_chain_job),
        )
        .route(
            "/api/chain-jobs/:id/operations/:operation_id/cancel",
            post(crate::routes_chain_jobs::cancel_chain_job_mutation),
        )
        .route(
            "/api/chain-jobs/gc",
            post(crate::routes_chain_jobs::gc_chain_jobs),
        )
        .route(
            "/api/chain-jobs/:id/stages/:idx/preview",
            get(crate::routes_chain_jobs::chain_job_stage_preview),
        )
        .route(
            "/api/chain-jobs/:id/stages/:idx/media",
            get(crate::routes_chain_jobs::chain_job_stage_media),
        )
        .route(
            "/api/chain-jobs/:id/stages/:idx/media-token",
            post(crate::routes_chain_jobs::create_chain_job_stage_media_token),
        )
        .route("/api/expand", post(expand_prompt))
        .route("/api/remix", post(remix_prompt))
        .route("/api/models", get(list_models))
        .route("/api/models/:model", delete(delete_model))
        .route("/api/models/:model/components", get(model_components))
        .route("/api/loras", get(crate::catalog_api::list_loras))
        .route("/api/models/load", post(load_model))
        .route("/api/models/pull", post(pull_model_endpoint))
        .route("/api/models/unload", delete(unload_model))
        .route("/api/gallery", get(list_gallery))
        .route(
            "/api/gallery/source-media/:filename",
            get(crate::gallery_source_media::inventory),
        )
        .route(
            "/api/gallery/source-media/:filename/:member_id",
            get(crate::gallery_source_media::download),
        )
        .route(
            "/api/gallery/source-media/:filename/reuse-sessions",
            post(crate::gallery_source_media::create_reuse_session),
        )
        .route("/api/gallery/media-token", post(create_gallery_media_token))
        .route("/api/gallery/export-options", get(gallery_export_options))
        .route("/api/gallery/export/:filename", post(export_gallery_video))
        // ─── Library organization + trash ───────────────────────────────
        .route(
            "/api/gallery/organize",
            post(crate::gallery_organization::organize_gallery),
        )
        .route(
            "/api/gallery/mutations",
            post(crate::gallery_organization::mutate_gallery_bulk),
        )
        .route(
            "/api/gallery/collections",
            get(crate::gallery_organization::list_collections)
                .post(crate::gallery_organization::create_collection),
        )
        .route(
            "/api/gallery/collections/:id",
            get(crate::gallery_organization::get_collection)
                .patch(crate::gallery_organization::update_collection)
                .delete(crate::gallery_organization::delete_collection),
        )
        .route(
            "/api/gallery/collections/:id/items",
            put(crate::gallery_organization::put_collection_items),
        )
        .route(
            "/api/gallery/tags",
            get(crate::gallery_organization::list_tags),
        )
        .route(
            "/api/gallery/tags/:name",
            patch(crate::gallery_organization::rename_tag)
                .delete(crate::gallery_organization::delete_tag),
        )
        .route(
            "/api/gallery/trash",
            post(crate::gallery_trash::trash_gallery_files)
                .delete(crate::gallery_trash::empty_gallery_trash),
        )
        .route(
            "/api/gallery/trash/restore",
            post(crate::gallery_trash::restore_gallery_files),
        )
        .route(
            "/api/gallery/trash/delete-forever",
            post(crate::gallery_trash::delete_gallery_files_forever),
        )
        .route(
            "/api/gallery/trash/sweep",
            post(crate::gallery_trash::sweep_gallery_trash),
        )
        .route("/api/pairing/sessions", post(create_pairing_session))
        .route("/api/pairing/claim", post(claim_pairing_session))
        .route("/api/pairing/clients", get(list_paired_clients))
        .route("/api/pairing/clients/:id", delete(revoke_paired_client))
        .route(
            "/api/gallery/import/:filename",
            put(import_gallery_file).layer(DefaultBodyLimit::max(
                usize::try_from(
                    MAX_GALLERY_IMPORT_FILE_BYTES
                        + MAX_GALLERY_IMPORT_METADATA_BYTES as u64
                        + GALLERY_IMPORT_HEADER_BYTES as u64,
                )
                .unwrap_or(usize::MAX),
            )),
        )
        .route(
            "/api/gallery/image/:filename",
            get(get_gallery_image)
                .delete(crate::gallery_trash::delete_gallery_image)
                .patch(crate::gallery_organization::patch_gallery_image),
        )
        .route(
            "/api/gallery/thumbnail/:filename",
            get(get_gallery_thumbnail),
        )
        .route("/api/gallery/preview/:filename", get(get_gallery_preview))
        // ─── Downloads UI (Agent A) ────────────────────────────────────────
        .route("/api/downloads", get(list_downloads).post(create_download))
        .route("/api/downloads/:id", delete(delete_download))
        .route("/api/downloads/stream", get(stream_downloads))
        // ─── Catalog (live HF + Civitai proxy) ──────────────────────────
        .route(
            "/api/catalog/families",
            get(crate::catalog_api::list_families),
        )
        .route(
            "/api/catalog/search",
            get(crate::catalog_api::live_search_catalog),
        )
        .route(
            "/api/catalog/installed",
            get(crate::catalog_api::list_installed_catalog),
        )
        .route(
            "/api/catalog/credentials",
            get(crate::catalog_credentials::get_catalog_credentials),
        )
        .route(
            "/api/catalog/credentials/:provider",
            axum::routing::put(crate::catalog_credentials::put_catalog_credential)
                .delete(crate::catalog_credentials::delete_catalog_credential),
        )
        .route(
            "/api/catalog/*id",
            get(crate::catalog_api::get_catalog_entry)
                .post(crate::catalog_api::post_catalog_dispatch),
        )
        .route("/api/upscale", post(upscale))
        .route("/api/upscale/stream", post(upscale_stream))
        .route(
            "/api/gallery/upscale",
            post(crate::video_upscale::upscale_gallery_image),
        )
        .route(
            "/api/video-upscale-jobs",
            get(crate::video_upscale::list_jobs).post(crate::video_upscale::create_job),
        )
        .route(
            "/api/video-upscale-jobs/:id",
            get(crate::video_upscale::get_job).delete(crate::video_upscale::cancel_job),
        )
        .route(
            "/api/video-upscale-jobs/:id/events",
            get(crate::video_upscale::job_events),
        )
        .route(
            "/api/video-upscale-jobs/:id/pause",
            post(crate::video_upscale::pause_job),
        )
        .route(
            "/api/video-upscale-jobs/:id/resume",
            post(crate::video_upscale::resume_job),
        )
        .route("/api/resources", get(get_resources))
        .route("/api/resources/stream", get(get_resources_stream))
        .route("/api/events", get(stream_events))
        .route("/api/status", get(server_status))
        .route("/api/devices", get(list_devices))
        .route("/api/devices/:id", patch(patch_device))
        .route("/api/queue", get(list_queue).delete(cancel_all_queue))
        .route(
            "/api/activity",
            get(crate::routes_activity::list_active_work),
        )
        .route("/api/queue/pause", post(pause_queue))
        .route("/api/queue/resume", post(resume_queue))
        .route(
            "/api/queue/:id",
            get(get_queue_job)
                .patch(patch_queue_job)
                .delete(cancel_queue_job),
        )
        .route("/api/queue/:id/retry", post(retry_queue_job))
        .route("/api/queue/:id/pause", post(pause_queue_job))
        .route("/api/queue/:id/resume", post(resume_queue_job))
        .route(
            "/api/queue/held/sweep",
            post(crate::queue_retention::sweep_held_queue),
        )
        .route(
            "/api/generation-batches/sweep",
            post(crate::queue_retention::sweep_settled_batches),
        )
        .route("/api/queue/:id/preview", get(get_queue_job_preview))
        .route("/api/history", get(list_history).delete(delete_history))
        .route("/api/capabilities", get(server_capabilities))
        .route("/api/licenses", get(list_licenses_endpoint))
        .route("/api/discovery/peers", get(discovery_peers))
        .route(
            "/api/capabilities/chain-limits",
            get(capabilities_chain_limits),
        )
        .route(
            "/api/capabilities/ltx2-control-adapters",
            get(capabilities_ltx2_control_adapters),
        )
        .route(
            "/api/capabilities/ltx2-camera-controls",
            get(capabilities_ltx2_camera_controls),
        )
        .route("/api/shutdown", post(shutdown_server))
        // ─── /api/config — HTTP counterpart of the `mold config` verbs ────
        .route("/api/config", get(crate::routes_config::list_config))
        .route(
            "/api/config/profiles",
            get(crate::routes_config::list_config_profiles),
        )
        .route(
            "/api/config/profile",
            axum::routing::put(crate::routes_config::put_config_profile),
        )
        .route(
            "/api/config/:key",
            get(crate::routes_config::get_config_key)
                .put(crate::routes_config::put_config_key)
                .delete(crate::routes_config::delete_config_key),
        )
        // Agent C (model-ui-overhaul §3): placement persistence.
        .route(
            "/api/config/model/:name/placement",
            get(get_model_placement)
                .put(put_model_placement)
                .delete(delete_model_placement),
        )
        .route("/health", get(health))
        .with_state(state)
        .route("/api/openapi.json", get(openapi_json))
        .route("/api/docs", get(scalar_docs))
}

// ── Model readiness ──────────────────────────────────────────────────────────

fn sse_message_to_event(msg: SseMessage) -> SseEvent {
    fn serialize_event<T: Serialize>(event_name: &str, payload: &T) -> SseEvent {
        match serde_json::to_string(payload) {
            Ok(data) => SseEvent::default().event(event_name).data(data),
            Err(err) => SseEvent::default().event("error").data(
                serde_json::json!({
                    "message": format!("failed to serialize SSE payload: {err}")
                })
                .to_string(),
            ),
        }
    }

    match msg {
        SseMessage::Progress(payload) => serialize_event("progress", &payload),
        SseMessage::Complete(payload) => serialize_event("complete", &payload),
        SseMessage::UpscaleComplete(payload) => serialize_event("complete", &payload),
        SseMessage::Error(payload) => serialize_event("error", &payload),
    }
}

#[cfg(test)]
fn save_image_to_dir(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return;
    }
    // Use milliseconds for server-side filenames to avoid overwrites when
    // concurrent requests finish in the same second.
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let ext = img.format.to_string();
    let filename =
        mold_core::default_output_filename(model, timestamp_ms, &ext, batch_size, img.index);
    let path = dir.join(&filename);
    match std::fs::write(&path, &img.data) {
        Ok(()) => tracing::info!("saved image to {}", path.display()),
        Err(e) => tracing::warn!("failed to save image to {}: {e}", path.display()),
    }
}

// ── Shared pre-queue validation ───────────────────────────────────────────────

pub(crate) fn ensure_generation_available(state: &AppState) -> Result<(), ApiError> {
    if let Some(reason) = state.generation_unavailable() {
        return Err(ApiError::generation_unavailable(reason));
    }
    Ok(())
}

fn ensure_schedulable_device(state: &AppState) -> Result<(), ApiError> {
    ensure_generation_available(state)?;
    if state.device_registry.has_devices() && state.gpu_pool.schedulable_worker_count() == 0 {
        return Err(ApiError::no_schedulable_device(
            "no enabled, healthy GPU device is available",
        ));
    }
    Ok(())
}

/// Validate a generate request and resolve server-side defaults.
///
/// Performs the identical pre-queue checks used by `generate`,
/// `generate_stream`, and durable batch admission: applies server defaults,
/// validates policy and the resolved generation profile, checks model
/// availability, freezes request transformations, and resolves publication.
/// Pre-queue advisories about a request that was still accepted.
///
/// Dimension adjustments keep their own long-standing header and documented
/// meaning; everything else rides a general one. Collapsing the two would make
/// `x-mold-dimension-warning` mean "some warning", which a client that
/// special-cases dimension handling could reasonably drop on the floor — and a
/// lip-dub timing substitution is exactly the kind of thing that must not be
/// silently discarded.
#[derive(Debug, Default)]
pub(crate) struct RequestWarnings {
    dimension: Option<String>,
    other: Vec<String>,
}

impl RequestWarnings {
    fn is_empty(&self) -> bool {
        self.dimension.is_none() && self.other.is_empty()
    }

    /// Every advisory, request-specific ones first.
    pub(crate) fn all(&self) -> impl Iterator<Item = &str> {
        self.other
            .iter()
            .map(String::as_str)
            .chain(self.dimension.as_deref())
    }
}

/// Fold advisories the RENDER produced into the ones admission produced.
///
/// Admission's advisories are known before the job runs; a render's are not —
/// the identity extraction's "several faces, largest used" is decided while
/// preparing the job's dependencies, long after this handler built its
/// `RequestWarnings`. Both belong on the one header, because a caller reading
/// `x-mold-request-warning` is asking "is there anything I should know about
/// this print", not "at which stage was it noticed".
///
/// Deduplicating rather than appending blindly: a batch child and its parent
/// carry the same identity advisory, and a caller should see it once.
fn merge_render_warnings(mut warnings: RequestWarnings, from_render: &[String]) -> RequestWarnings {
    for warning in from_render {
        if !warnings.other.iter().any(|held| held == warning) {
            warnings.other.push(warning.clone());
        }
    }
    warnings
}

fn merge_request_warnings(
    mut admitted: RequestWarnings,
    deferred: RequestWarnings,
) -> RequestWarnings {
    if admitted.dimension.is_none() {
        admitted.dimension = deferred.dimension;
    } else if let Some(dimension) = deferred.dimension {
        if !admitted.other.iter().any(|warning| warning == &dimension) {
            admitted.other.push(dimension);
        }
    }
    for warning in deferred.other {
        if !admitted.other.iter().any(|held| held == &warning) {
            admitted.other.push(warning);
        }
    }
    admitted
}

pub(crate) async fn require_server_model_activation(
    state: &AppState,
    model_name: &str,
) -> Result<Option<String>, ApiError> {
    let family = model_manager::family_for_model(state, model_name).await;
    mold_core::require_model_activation(model_name, family.as_deref())
        .map_err(ApiError::model_activation)?;

    let resolved = mold_core::manifest::resolve_model_name(model_name);
    let config = state.config.read().await;
    let artifact_root = config.resolved_models_dir();
    if let Some(model) = config
        .models
        .get(&resolved)
        .or_else(|| config.models.get(model_name))
    {
        mold_core::require_model_activation(&resolved, model.family.as_deref())
            .map_err(ApiError::model_activation)?;
        for path in model.all_file_paths() {
            mold_core::require_model_artifact_activation(
                std::path::Path::new(&path),
                Some(&artifact_root),
                model.family.as_deref(),
            )
            .map_err(ApiError::model_activation)?;
        }
    }
    if let Some(paths) = mold_core::config::ModelPaths::resolve(model_name, &config) {
        for path in paths.all_file_paths() {
            mold_core::require_model_artifact_activation(
                path,
                Some(&artifact_root),
                family.as_deref(),
            )
            .map_err(ApiError::model_activation)?;
        }
    }
    drop(config);

    if let Some(manifest) = mold_core::manifest::find_manifest(&resolved) {
        mold_core::require_registered_manifest_activation(manifest)
            .map_err(ApiError::model_activation)?;
    }
    Ok(family)
}

/// Validate discovery/storage authority without implying that the same model
/// can execute on this server. Only download and repair ingress may use this;
/// generation, loading, cloud, expansion, and utility routes retain
/// [`require_server_model_activation`].
async fn require_server_model_acquisition(
    state: &AppState,
    model_name: &str,
) -> Result<Option<String>, ApiError> {
    let family = model_manager::family_for_model(state, model_name).await;
    mold_core::require_model_acquisition(model_name, family.as_deref())
        .map_err(ApiError::model_activation)?;

    let resolved = mold_core::manifest::resolve_model_name(model_name);
    if let Some(manifest) = mold_core::manifest::find_manifest(&resolved) {
        mold_core::require_model_acquisition(&manifest.name, Some(&manifest.family))
            .map_err(ApiError::model_activation)?;
    }
    Ok(family)
}

/// Apply a download request's `accept_licenses`, then refuse if the model
/// still needs one.
///
/// Ordering is the whole point: acceptance is recorded in THIS server's Mold
/// data root before the pull is enqueued, because a client that recorded it
/// locally told the wrong machine — that was the remote-`MOLD_HOST` bug.
///
/// Unknown ids are a `400` and nothing is written; see
/// [`mold_core::license_acceptance::record_acceptances`] for why they are
/// rejected rather than ignored.
async fn apply_download_license_acceptances(
    state: &AppState,
    model: &str,
    accept_licenses: &[mold_core::LicenseAcceptance],
) -> Result<(), ApiError> {
    use mold_core::license_acceptance;

    // This server's own root — the same one `catalog_credentials` writes to.
    let mold_home = mold_core::Config::mold_dir()
        .ok_or_else(|| ApiError::internal("could not resolve the Mold data directory"))?;

    if !accept_licenses.is_empty() {
        license_acceptance::record_acceptances(&mold_home, accept_licenses).map_err(|error| {
            match error {
                license_acceptance::RecordAcceptancesError::Unknown(unknown) => {
                    ApiError::with_code(
                        unknown.to_string(),
                        "UNKNOWN_LICENSE",
                        StatusCode::BAD_REQUEST,
                    )
                }
                license_acceptance::RecordAcceptancesError::TermsMismatch(ours) => {
                    ApiError::license_terms_mismatch(ours)
                }
                license_acceptance::RecordAcceptancesError::Io(io) => {
                    ApiError::internal(format!("failed to record license acceptance: {io}"))
                }
            }
        })?;
    }

    // Re-derive every term from the manifest rather than trusting the
    // accepted list: a request may name one license while the model needs
    // another. Only files this pull would fetch are gated; an already-present
    // restricted artifact does not require retroactive consent merely because
    // an unrelated file in the same bundle needs repair.
    let resolved = mold_core::manifest::resolve_model_name(model);
    let Some(manifest) = mold_core::manifest::find_manifest(&resolved) else {
        // Unknown models are the enqueue path's 400 to report, not ours.
        return Ok(());
    };
    let config = state.config.read().await;
    for file in &manifest.files {
        if config.complete_manifest_file_path(manifest, file).is_some() {
            continue;
        }
        for license in
            license_acceptance::licenses_for_manifest_file(&manifest.name, &file.hf_filename)
        {
            if !license_acceptance::is_accepted(&mold_home, license) {
                return Err(ApiError::license_not_accepted(&manifest.name, license));
            }
        }
    }
    Ok(())
}

pub(crate) async fn require_server_generation_request_activation(
    state: &AppState,
    request: &mold_core::GenerateRequest,
    family: Option<&str>,
) -> Result<(), ApiError> {
    let models_root = state.config.read().await.resolved_models_dir();
    mold_core::require_generate_request_model_activation(request, Some(&models_root), family)
        .map_err(ApiError::model_activation)?;

    if let Some(control_model) = request.control_model.as_deref() {
        require_server_model_activation(state, control_model).await?;
    }
    if let Some(upscale_model) = request.upscale_model.as_deref() {
        require_server_model_activation(state, upscale_model).await?;
    }
    Ok(())
}

pub(crate) struct PreparedGenerationRoute {
    pub(crate) warnings: RequestWarnings,
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    pub(crate) h3_private_ingress_grant: Option<crate::h3_private_bridge::H3PrivateIngressGrant>,
}

/// Name the first request-owned media authority that cannot be replayed from
/// the durable queue. The first-party clients apply the same fence before
/// selecting this endpoint, but the server must enforce its own persistence
/// contract for direct and mixed-version callers too.
fn durable_generation_unsupported(message: impl Into<String>) -> ApiError {
    ApiError::with_code(
        message,
        "GENERATION_BATCH_NOT_DURABLE",
        StatusCode::UNPROCESSABLE_ENTITY,
    )
}

pub(crate) async fn prepare_generation_after_durable_ack(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
    authority: crate::durable_admission_authority::RuntimeAuthority,
) -> Result<PreparedGenerationRoute, ApiError> {
    prepare_generation_inner(state, request, None, Some(authority)).await
}

async fn prepare_generation_inner(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
    restored_authority: Option<crate::durable_admission_authority::RuntimeAuthority>,
) -> Result<PreparedGenerationRoute, ApiError> {
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let _ = (&restored_authority, authenticated);
    // This seam deliberately does not load an inference engine, prepare model
    // weights, or reserve execution memory. Scheduler V2 owns those bounded
    // operations after durable acknowledgement. Local prompt expansion and
    // post-upscale downloads likewise remain scheduler dependencies. The
    // request-boundary work retained here is what must be frozen in the
    // journal for replay (including API-backed expansion); first-party adapter
    // downloads with no resumable dependency record are handled explicitly at
    // the materialization site below.
    // Stop admitting once the retention fence is up. Anything accepted after
    // that point is queued into a process that is already tearing down, so the
    // honest answer is "not now" rather than a job that immediately retains.
    if restored_authority.is_none() && state.queue_journal.is_retaining() {
        return Err(ApiError::server_restarting(
            "the host is restarting; retry shortly",
        ));
    }
    // Collapse every released H3 alias to one exact task/layout identity
    // before activation, upload scope, admission, placement, queue, metadata,
    // and retry state can observe the request. This is deliberately a no-op
    // for every non-H3 configured alias and catalog ID.
    mold_core::minimax_h3::canonicalize_request_model(request);

    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if mold_core::minimax_h3::task_for_model(&request.model).is_some() {
        request.normalise_output_format(Some(mold_core::minimax_h3::FAMILY));
    }
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let h3_private_ingress_grant = match restored_authority.as_ref() {
        Some(authority) => authority.h3_grant().cloned(),
        None => crate::h3_private_bridge::classify_h3_private_ingress(
            request,
            authenticated,
            state.instance_id.as_str(),
        )?,
    };
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let authority_bound_request = restored_authority
        .as_ref()
        .and_then(|_| h3_private_ingress_grant.as_ref().map(|_| request.clone()));
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let private_h3_ingress = h3_private_ingress_grant.is_some();
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let private_h3_ingress = false;

    let family = if private_h3_ingress {
        Some(mold_core::minimax_h3::FAMILY.to_string())
    } else {
        require_server_model_activation(state, &request.model).await?
    };
    if !private_h3_ingress {
        require_server_generation_request_activation(state, request, family.as_deref()).await?;
    }
    if let Some(references) = request.references.as_deref() {
        // Admission resolved every public authority before the row was
        // acknowledged; what reaches preparation is the descriptor list whose
        // media the consumer hydrates under its own lease. Anything else here
        // is a row that was written wrong, held rather than guessed at.
        mold_core::minimax_h3::validate_reference_descriptors(references)
            .map_err(ApiError::reference)?;
    }
    if request.expand == Some(true) && !request.prompt.trim().is_empty() {
        let settings = state
            .config
            .read()
            .await
            .expand
            .clone()
            .with_env_overrides();
        require_expand_model_activation(&settings)?;
    }
    ensure_schedulable_device(state)?;
    // NOTE: the capacity check is enforced inside `state.queue.submit(...)` so
    // that a burst of concurrent callers can't all slip past an open check
    // (classic TOCTOU).  The submit call in `generate`/`generate_stream` will
    // return `SubmitError::Full`, which is mapped to `ApiError::queue_full()`.
    apply_default_metadata_setting(state, request).await;
    normalize_generation_placement(state, request).await;

    let preferred_gpu = validate_multi_gpu_placement(state, request.placement.as_ref())?;

    // Catalog (`cv:*` / `hf:*`) IDs aren't in the static manifest, so the
    // pure-mold-core family lookup returns `None` for them. Run the live
    // single-id install first so the intent cache has the entry; then
    // feed its family string through as a hint so audio / keyframes /
    // pipeline gates work for installed Civitai LTX-2 checkpoints.
    //
    // A `Network` error here means Civitai/HF is unreachable — surface it
    // immediately as 502 rather than letting the user fall through to
    // a "not installed" 404 they can't act on.
    if !private_h3_ingress {
        if let Err(e) = model_manager::install_catalog_model(state, &request.model).await {
            return Err(model_manager::install_error_to_api_error(&e));
        }
    }
    // Resolve the model family for normalisation. `family_for_model` checks the
    // static manifest first (covers all built-in models), then configured
    // models, and finally catalog metadata (`cv:*` / `hf:*` installed above).
    let resolved_family = if private_h3_ingress {
        Some(mold_core::minimax_h3::FAMILY.to_string())
    } else {
        require_server_model_activation(state, &request.model).await?
    };
    // Expand only after live catalog resolution, so opaque cv:/hf: IDs use
    // their authoritative family and conditioning-aware task template.
    maybe_expand_prompt(state, request, preferred_gpu, resolved_family.as_deref()).await?;
    let canonical_model = mold_core::manifest::resolve_model_name(&request.model);
    let resolved_profile = if private_h3_ingress {
        None
    } else {
        model_manager::list_models(state)
            .await
            .into_iter()
            .find(|entry| entry.info.name == request.model || entry.info.name == canonical_model)
            .and_then(|entry| entry.generation_profile)
    };
    // The effective, delivery-qualified recipe owns the output default. A
    // family heuristic here can select MP4 even when this binary did not link
    // the encoder, causing an omitted field to fail its own advertised profile.
    if request.output_format.is_none() {
        if let Some(profile) = resolved_profile.as_ref() {
            mold_core::materialize_generation_profile_output_default(profile, request)
                .map_err(ApiError::validation)?;
        } else if private_h3_ingress {
            // Private H3 ingress is outside the public catalog contract.
            request.normalise_output_format(resolved_family.as_deref());
        }
    }
    // Materialize the family's tuned default negative (wan) into the request
    // when the caller omitted the field, so the queue/worker metadata — and
    // therefore saved gallery provenance and "Reuse settings" — record the
    // uncond that actually conditions the render. Same engine semantics
    // either way; an explicit value (the empty-string opt-out included) is
    // authoritative and passes through untouched.
    materialize_default_negative_prompt(request, resolved_family.as_deref());
    // Same seam, same reason: a continuation that named no overlap renders
    // with its family's carryover, and the metadata every queue/worker path
    // builds resolves the family through the manifest — which an installed
    // `cv:` / `hf:` wan checkpoint has none of. Filling the field in here,
    // with the family admission already resolved, is what keeps saved
    // provenance equal to what actually rendered (#783).
    mold_core::validation::materialize_extend_overlap_frames(request, resolved_family.as_deref());

    // Same discipline for the creation-time filing: `OutputMetadata` is built
    // from this request while the gallery row is seeded through a path that
    // re-normalizes, so raw spellings left here would put a different filing
    // in the print's provenance than the one it actually receives. First-party
    // clients normalize before sending; a direct HTTP caller does not.
    // Refusal is reported by the validation below, which runs the same check.
    let _ = mold_core::validation::materialize_request_organization(request);

    // A model with no deliverable recipe cannot reach media-dependent
    // preparation. Preserve basic request/unknown-model diagnostics first,
    // then fail before control planning, LipDub probing, reference resolution,
    // or server-local media access.
    if !private_h3_ingress && resolved_profile.is_none() {
        validate_generate_request(
            request,
            resolved_family.as_deref(),
            mold_core::ReferenceForm::Resolved,
        )
        .map_err(ApiError::validation)?;
        let _ = model_manager::check_model_available(state, &request.model).await?;
        return Err(ApiError::validation(format!(
            "model '{}' has no generation recipe deliverable by this server build",
            request.model
        )));
    }

    let planned_control = plan_builtin_ltx2_control(state, request).await?;
    let planned_camera_controls = plan_builtin_ltx2_camera_controls(state, request).await?;

    // Lip dub's length and rate belong to the reference clip. Resolve them
    // here, before validation and before the scheduler prices the job: a plan
    // admitted for 97 frames that then renders 481 would blow its VRAM grant.
    let mut warnings = RequestWarnings {
        other: apply_lip_dub_reference_timing(state, request).await?,
        ..RequestWarnings::default()
    };

    let mut singleton_validation;
    let validation_request = if request.batch_size > 1 && state.scheduled_work.v2_authoritative() {
        singleton_validation = request.clone();
        singleton_validation.batch_size = 1;
        &singleton_validation
    } else {
        &*request
    };
    // This runs AFTER durable admission resolved every reference to its
    // descriptor (bytes sealed in the media set), so the request is
    // validated in its resolved form; the admission rule would hold every
    // durable Ref2VA print here.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let validation = if private_h3_ingress {
        mold_core::validation::validate_h3_private_uat_request_with(
            validation_request,
            mold_core::ReferenceForm::Resolved,
        )
    } else {
        validate_generate_request(
            validation_request,
            resolved_family.as_deref(),
            mold_core::ReferenceForm::Resolved,
        )
    };
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let validation = validate_generate_request(
        validation_request,
        resolved_family.as_deref(),
        mold_core::ReferenceForm::Resolved,
    );
    if let Err(e) = validation {
        return Err(ApiError::validation(e));
    }
    if !private_h3_ingress {
        if let Some(profile) = resolved_profile.as_ref() {
            mold_core::validate_request_against_generation_profile(profile, validation_request)
                .map_err(ApiError::validation)?;
        }
        let _ = model_manager::check_model_available(state, &request.model).await?;
    }
    enforce_source_image_capability(state, request, resolved_family.as_deref()).await?;

    // Resolve the creation-time filing against this host now that the
    // request is otherwise admissible. Publication only ever files by name,
    // so an `{id}` reference is turned into its name here — that is also the
    // name the print's embedded provenance will record, and resolving it
    // later would risk filing under one collection and recording another.
    warnings
        .other
        .extend(resolve_request_filing(state, request).await);

    resolve_server_local_media_paths(state, request).await?;
    if let Some((adapter, path)) = planned_control {
        // The ordinary attached route may wait for this first-party adapter
        // download. A durable acknowledgement may not: there is no persisted
        // dependency-download state for the feeder to resume after a crash.
        // Installed adapters are materialized cheaply and identically; absent
        // adapters are refused honestly instead of acknowledging a row whose
        // request cannot reproduce the legacy execution payload.
        if !control_artifact_is_complete(adapter, &path) {
            return Err(durable_generation_unsupported(format!(
                "IC-LoRA control '{}' must be downloaded before durable batch admission",
                adapter.id
            )));
        }
        materialize_builtin_ltx2_control(state, request, adapter, path).await?;
    }
    if let Some((preset, _)) = planned_camera_controls
        .iter()
        .find(|(preset, path)| !camera_control_artifact_is_complete(preset, path))
    {
        return Err(durable_generation_unsupported(format!(
            "camera control '{}' must be downloaded before durable batch admission",
            preset.id
        )));
    }
    materialize_builtin_ltx2_camera_controls(state, &planned_camera_controls).await?;
    // Durable admission accepts a request naming a server-local adapter and
    // preparation may run minutes — or a restart — later, so the path is
    // re-asked HERE rather than trusted from admission. A LoRA that has since
    // been moved or deleted holds its row by name, which is the same shape a
    // missing model takes: the print is parked with an actionable reason, not
    // rendered without the adapter it asked for and not silently dropped.
    {
        let config = state.config.read().await;
        if let Some(missing) = missing_lora_path(&config, request) {
            return Err(ApiError::not_found(format!(
                "LoRA adapter is no longer readable at {missing}; \
                 restore the file and retry this job"
            )));
        }
    }

    let dim_warning = {
        let config = state.config.read().await;
        let family = config.resolved_model_config(&request.model).family;
        family.as_deref().and_then(|f| {
            // Per model: a composing LTX-2 checkpoint advertises the
            // composed rungs, so judging it against the single-pass list
            // would flag the very shapes it exists to render.
            let composition = if f == "ltx2" {
                mold_core::validation::ltx2_spatial_composition(&request.model, request.pipeline)
            } else {
                mold_core::validation::Ltx2SpatialComposition::SinglePass
            };
            mold_core::dimension_warning_composed(request.width, request.height, f, composition)
        })
    };

    warnings.dimension = dim_warning;
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let h3_private_ingress_grant = if let (Some(grant), Some(submitted)) = (
        h3_private_ingress_grant.as_ref(),
        authority_bound_request.as_ref(),
    ) {
        Some(grant.rebind_server_prepared_request(
            submitted,
            request,
            state.instance_id.as_str(),
        )?)
    } else if h3_private_ingress_grant.is_some() {
        crate::h3_private_bridge::classify_h3_private_ingress(
            request,
            authenticated,
            state.instance_id.as_str(),
        )?
    } else {
        None
    };
    Ok(PreparedGenerationRoute {
        warnings,
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        h3_private_ingress_grant,
    })
}

pub(crate) async fn plan_builtin_ltx2_camera_controls(
    state: &AppState,
    request: &mold_core::GenerateRequest,
) -> Result<
    Vec<(
        &'static mold_core::ltx2_camera::Ltx2CameraControlPreset,
        std::path::PathBuf,
    )>,
    ApiError,
> {
    let config = state.config.read().await;
    plan_builtin_ltx2_camera_controls_in_config(&config, request)
}

fn plan_builtin_ltx2_camera_controls_in_config(
    config: &mold_core::Config,
    request: &mold_core::GenerateRequest,
) -> Result<
    Vec<(
        &'static mold_core::ltx2_camera::Ltx2CameraControlPreset,
        std::path::PathBuf,
    )>,
    ApiError,
> {
    let aliases = request
        .loras
        .as_deref()
        .unwrap_or_default()
        .iter()
        .chain(request.lora.iter())
        .filter_map(|lora| lora.path.strip_prefix("camera-control:"))
        .collect::<Vec<_>>();
    if aliases.is_empty() {
        return Ok(Vec::new());
    }

    let effective_config =
        crate::model_manager::resolve_existing_model_authority(&request.model, config)?
            .map_or_else(|| config.clone(), |authority| authority.config);
    let effective = effective_config.resolved_model_config(&request.model);
    mold_core::ltx2_camera::camera_profile_for_model(&request.model, &effective)
        .map_err(ApiError::validation)?;
    let models_dir = config.resolved_models_dir();
    let mut planned: Vec<(
        &'static mold_core::ltx2_camera::Ltx2CameraControlPreset,
        std::path::PathBuf,
    )> = Vec::new();
    for alias in aliases {
        let preset = mold_core::ltx2_camera::resolve_camera_control_preset(alias)
            .map_err(ApiError::validation)?;
        if planned.iter().any(|(existing, _)| existing.id == preset.id) {
            continue;
        }
        let manifest = mold_core::manifest::find_manifest(preset.download_model)
            .expect("camera-control registry and hidden manifests must stay in sync");
        let file = manifest
            .files
            .first()
            .expect("camera-control manifests contain one adapter file");
        planned.push((
            preset,
            models_dir.join(mold_core::manifest::storage_path(manifest, file)),
        ));
    }
    Ok(planned)
}

pub(crate) async fn materialize_chain_camera_controls(
    state: &AppState,
    config: &mold_core::Config,
    request: &mold_core::ChainRequest,
) -> Result<(), ApiError> {
    let mut generate = request.synthetic_generate_request(
        mold_core::OutputFormat::Mp4,
        request.estimated_total_frames(),
        request.fps,
    );
    generate.loras = Some(
        request
            .stages
            .iter()
            .flat_map(|stage| stage.loras.iter())
            .map(|lora| mold_core::LoraWeight {
                path: lora.path.clone(),
                scale: lora.scale,

                expert: None,
            })
            .collect(),
    );
    let planned = plan_builtin_ltx2_camera_controls_in_config(config, &generate)?;
    materialize_builtin_ltx2_camera_controls(state, &planned).await
}

async fn materialize_builtin_ltx2_camera_controls(
    state: &AppState,
    planned: &[(
        &'static mold_core::ltx2_camera::Ltx2CameraControlPreset,
        std::path::PathBuf,
    )],
) -> Result<(), ApiError> {
    for (preset, path) in planned {
        if camera_control_artifact_is_complete(preset, path) {
            continue;
        }
        let mut events = state.downloads.subscribe();
        let (job_id, _, _) = state
            .downloads
            .enqueue(preset.download_model.to_string())
            .await
            .map_err(|error| {
                ApiError::internal(format!("cannot queue camera-control preset: {error}"))
            })?;
        loop {
            match events.recv().await {
                Ok(mold_core::DownloadEvent::JobDone { id, .. }) if id == job_id => break,
                Ok(mold_core::DownloadEvent::JobFailed { id, error }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "failed to download camera-control preset '{}': {error}",
                        preset.id
                    )));
                }
                Ok(mold_core::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "camera-control preset '{}' download was cancelled",
                        preset.id
                    )));
                }
                Ok(_) => {}
                Err(error) => {
                    return Err(ApiError::internal(format!(
                        "lost camera-control download status: {error}"
                    )));
                }
            }
        }
        if !camera_control_artifact_is_complete(preset, path) {
            return Err(ApiError::internal(format!(
                "camera-control preset '{}' download completed without a verified {}",
                preset.id,
                path.display()
            )));
        }
    }
    Ok(())
}

fn camera_control_artifact_is_complete(
    preset: &mold_core::ltx2_camera::Ltx2CameraControlPreset,
    path: &std::path::Path,
) -> bool {
    mold_core::download::has_sha256_marker(path)
        || path
            .metadata()
            .is_ok_and(|metadata| metadata.len() == preset.size_bytes)
}

async fn plan_builtin_ltx2_control(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Result<
    Option<(
        &'static mold_core::ltx2_control::Ltx2ControlAdapter,
        std::path::PathBuf,
    )>,
    ApiError,
> {
    let Some(control) = request.ic_lora_control.as_deref() else {
        return Ok(None);
    };
    let control = mold_core::ltx2_control::normalize_control_id(control);
    request.ic_lora_control = Some(control.clone());
    // Lip dub is a pipeline, not just an adapter: routing it through `ic-lora`
    // would load the right weights and then run a graph that drops them before
    // stage 2 and never conditions on the reference voice at all.
    let required = mold_core::ltx2_control::pipeline_for_control_id(&control);
    match request.pipeline {
        None => request.pipeline = Some(required),
        Some(pipeline) if pipeline == required => {}
        Some(other) => {
            return Err(ApiError::validation(format!(
                "ic_lora_control '{control}' conflicts with pipeline={other}; use \
                 pipeline={required}"
            )));
        }
    }

    let config = state.config.read().await;
    let effective_config =
        crate::model_manager::resolve_existing_model_authority(&request.model, &config)?
            .map_or_else(|| config.clone(), |authority| authority.config);
    let effective = effective_config.resolved_model_config(&request.model);
    let profile = mold_core::ltx2_control::control_profile_for_model(&request.model, &effective)
        .map_err(ApiError::validation)?;
    let adapter = mold_core::ltx2_control::resolve_control_adapter(profile, &control)
        .map_err(ApiError::validation)?;
    let manifest = mold_core::manifest::find_manifest(adapter.download_model)
        .expect("control registry and hidden manifests must stay in sync");
    // An adapter may ship companion files (HDR carries pre-computed prompt
    // embeddings), so resolve the weights by name instead of taking whichever
    // file happens to be first.
    let file = manifest
        .files
        .iter()
        .find(|file| file.hf_filename == adapter.hf_filename)
        .expect("control registry and hidden manifests must stay in sync");
    let path = config
        .resolved_models_dir()
        .join(mold_core::manifest::storage_path(manifest, file));
    Ok(Some((adapter, path)))
}

/// Resolve the engine's absence-fallback negative prompt into the request.
///
/// The wan engine substitutes its tuned default when `negative_prompt` is
/// absent (`wan/pipeline.rs::resolve_negative_prompt`); recording the request
/// as-received left saved metadata claiming no negative while ~60 Chinese
/// tokens conditioned the render (#787). Resolving here — after family
/// resolution, before validation and metadata capture — makes provenance
/// truthful without changing what renders. `Some("")` is the explicit
/// opt-out and must never be replaced.
pub(crate) fn materialize_default_negative_prompt(
    request: &mut mold_core::GenerateRequest,
    family: Option<&str>,
) {
    if request.negative_prompt.is_none() {
        if let Some(default) =
            family.and_then(mold_core::manifest::default_negative_prompt_for_family)
        {
            request.negative_prompt = Some(default.to_string());
        }
    }
}

/// Replace a lip-dub request's `frames` / `fps` with the reference clip's own.
///
/// Lip dub re-voices an existing video, so its output has to sit on that
/// video's timeline — upstream reads both numbers straight off the reference
/// stream (`lipdub.py:190-192`). Doing it here, before validation and before
/// the scheduler prices the job, is what keeps the VRAM grant honest: a plan
/// admitted for 97 frames that then rendered a 20-second reference would be
/// five times the size it was measured at.
///
/// Resolve the creation-time filing a request carries against this host,
/// rewriting `request.collection` into the `{name}` form publication files
/// by. Returns advisories for anything that was dropped.
///
/// Three things can go wrong, and none of them may refuse the render — a
/// print is the expensive artifact and its filing is not:
///
/// * **No metadata DB** (`MOLD_DB_DISABLE=1`): there is nowhere to file. The
///   filing is dropped and said so on `x-mold-request-warning`, never
///   silently and never as a refusal.
/// * **An `{id}` that no longer exists**: the collection was deleted between
///   the client reading the list and pressing Generate. Dropped and reported.
/// * **A DB read failure**: reported the same way rather than escalated.
///
/// A `{name}` reference needs no resolution at all: publication creates it
/// when absent, which is the cross-host create-by-name rule.
async fn resolve_request_filing(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Vec<String> {
    let filing = describe_request_filing(request);
    if filing.is_none() {
        return Vec::new();
    }

    let Some(db) = state.metadata_db.as_ref().as_ref() else {
        let filing = filing.expect("checked above");
        request.tags = None;
        request.collection = None;
        return vec![format!(
            "this host has no metadata database, so {filing} was not applied; \
             the print was generated and saved normally"
        )];
    };

    resolve_collection_reference(db, &mut request.collection)
}

/// Rewrite an `{id}` collection reference into the `{id, name}` form
/// publication files by, dropping it with an advisory when this host cannot
/// resolve it. A `{name}` reference needs no lookup — publication creates it
/// when absent — and is left exactly as it arrived.
///
/// Shared by the one-shot and chain admission paths so a sequence and a
/// single print resolve their filing identically.
pub(crate) fn resolve_collection_reference(
    db: &mold_db::MetadataDb,
    collection: &mut Option<mold_core::CollectionRef>,
) -> Vec<String> {
    let Some(id) = collection
        .as_ref()
        .filter(|reference| reference.name.is_none())
        .and_then(|reference| reference.id.as_deref())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
    else {
        return Vec::new();
    };

    match db.get_collection(&id) {
        Ok(Some(row)) => {
            *collection = Some(mold_core::CollectionRef {
                id: Some(row.id),
                name: Some(row.name),
            });
            Vec::new()
        }
        Ok(None) => {
            *collection = None;
            vec![format!(
                "collection '{id}' no longer exists on this host, so the print was not filed \
                 into it; its tags and everything else were applied normally"
            )]
        }
        Err(error) => {
            tracing::warn!("collection lookup failed for '{id}': {error:#}");
            *collection = None;
            vec![format!(
                "collection '{id}' could not be read on this host, so the print was not filed \
                 into it; its tags and everything else were applied normally"
            )]
        }
    }
}

/// Build the `x-mold-request-warning` header for a set of advisories, or an
/// empty map when there are none.
///
/// The generate path assembles this header from [`RequestWarnings`] on its
/// own response; the chain endpoints have no such struct, so they share this
/// so a dropped filing is never a silent drop there either.
pub(crate) fn request_warning_headers(warnings: &[String]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if warnings.is_empty() {
        return headers;
    }
    let joined = warnings.join("; ").replace('\n', " ");
    match HeaderValue::from_str(&joined) {
        Ok(value) => {
            headers.insert("x-mold-request-warning", value);
        }
        Err(e) => tracing::warn!("request warning could not be encoded as a header: {e}"),
    }
    headers
}

/// Human-readable summary of what a request asked to be filed under, for the
/// advisory text. `None` when it asked for nothing.
fn describe_request_filing(request: &mold_core::GenerateRequest) -> Option<String> {
    let tags = request.tags.as_deref().filter(|tags| !tags.is_empty());
    let collection = request
        .collection
        .as_ref()
        .filter(|reference| !reference.is_unset());
    match (tags, collection) {
        (None, None) => None,
        (Some(tags), None) => Some(format!("the requested {}", tag_phrase(tags))),
        (None, Some(_)) => Some("the requested collection".to_string()),
        (Some(tags), Some(_)) => Some(format!("the requested collection and {}", tag_phrase(tags))),
    }
}

fn tag_phrase(tags: &[String]) -> String {
    match tags.len() {
        1 => "tag".to_string(),
        n => format!("{n} tags"),
    }
}

/// Returns anything the caller asked for that the reference overrode, so the
/// client is told rather than quietly retimed.
async fn apply_lip_dub_reference_timing(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Result<Vec<String>, ApiError> {
    if request.pipeline != Some(mold_core::Ltx2PipelineMode::LipDub) {
        return Ok(Vec::new());
    }
    let probe = if let Some(bytes) = request.source_video.as_deref() {
        mold_inference::ltx2::media::probe_video_bytes(bytes)
    } else if let Some(path) = request.source_video_path.as_deref() {
        let roots = state.config.read().await.resolved_media_roots();
        let resolved = mold_core::resolve_server_media_path(path, &roots)
            .map_err(|e| ApiError::validation(format!("source_video_path: {e}")))?;
        mold_inference::ltx2::media::probe_video(&resolved)
    } else {
        // No reference at all: `validate_generate_request` says so far better
        // than a probe failure would.
        return Ok(Vec::new());
    };
    let probe = probe.map_err(|e| {
        ApiError::validation(format!("could not read the lip-dub reference video: {e:#}"))
    })?;
    let frames = probe.frames.ok_or_else(|| {
        ApiError::validation("the lip-dub reference video reports no frame count")
    })?;
    let timing = mold_core::validation::resolve_lip_dub_timing(
        mold_core::validation::LipDubReference {
            frames,
            fps: probe.fps,
            has_audio: probe.has_audio,
        },
        request.frames,
        request.fps,
    )
    .map_err(ApiError::validation)?;
    request.frames = Some(timing.frames);
    request.fps = Some(timing.fps);
    for warning in &timing.warnings {
        tracing::info!("{warning}");
    }
    Ok(timing.warnings)
}

async fn materialize_builtin_ltx2_control(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
    adapter: &'static mold_core::ltx2_control::Ltx2ControlAdapter,
    path: std::path::PathBuf,
) -> Result<(), ApiError> {
    if !control_artifact_is_complete(adapter, &path) {
        let mut events = state.downloads.subscribe();
        let (job_id, _, _) = state
            .downloads
            .enqueue(adapter.download_model.to_string())
            .await
            .map_err(|error| {
                ApiError::internal(format!("cannot queue control adapter: {error}"))
            })?;
        loop {
            match events.recv().await {
                Ok(mold_core::DownloadEvent::JobDone { id, .. }) if id == job_id => break,
                Ok(mold_core::DownloadEvent::JobFailed { id, error }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "failed to download IC-LoRA control '{}': {error}",
                        adapter.id
                    )));
                }
                Ok(mold_core::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "IC-LoRA control '{}' download was cancelled",
                        adapter.id
                    )));
                }
                Ok(_) => {}
                Err(error) => {
                    return Err(ApiError::internal(format!(
                        "lost IC-LoRA control download status: {error}"
                    )));
                }
            }
        }
    }
    if !control_artifact_is_complete(adapter, &path) {
        return Err(ApiError::internal(format!(
            "IC-LoRA control '{}' download completed without a verified {}",
            adapter.id,
            path.display()
        )));
    }

    let mut ordered = vec![mold_core::LoraWeight {
        path: path.to_string_lossy().into_owned(),
        scale: 1.0,

        expert: None,
    }];
    if let Some(lora) = request.lora.take() {
        ordered.push(lora);
    }
    if let Some(loras) = request.loras.take() {
        ordered.extend(loras);
    }
    request.loras = Some(ordered);
    Ok(())
}

/// Which of this adapter's files have not landed and verified.
///
/// `path` is the **weights** file, and it is authoritative as given — the
/// caller resolved it through the manifest, so its own name is checked rather
/// than one reconstructed from the registry. Companion files have no such
/// resolved path; they land beside the weights, so each is probed in that
/// directory at its own recorded size. Checking only the weights would let a
/// half-finished multi-file pull look complete and then fail at load.
///
/// A path with no parent cannot host companions, so an adapter that ships
/// them counts them outstanding.
fn control_missing_files(
    adapter: &mold_core::ltx2_control::Ltx2ControlAdapter,
    path: &std::path::Path,
) -> Vec<mold_core::ltx2_control::Ltx2ControlAdapterFile> {
    let landed = |candidate: &std::path::Path, size: u64| {
        mold_core::download::has_sha256_marker(candidate)
            || candidate
                .metadata()
                .is_ok_and(|metadata| metadata.len() == size)
    };
    adapter
        .files()
        .filter(|file| {
            if file.hf_filename == adapter.hf_filename {
                return !landed(path, file.size_bytes);
            }
            match path.parent() {
                Some(dir) => !landed(&dir.join(file.hf_filename), file.size_bytes),
                None => true,
            }
        })
        .collect()
}

/// Whether every file this adapter needs has landed and verified.
fn control_artifact_is_complete(
    adapter: &mold_core::ltx2_control::Ltx2ControlAdapter,
    path: &std::path::Path,
) -> bool {
    control_missing_files(adapter, path).is_empty()
}

/// Bytes an admission would actually fetch — the outstanding files only.
///
/// The adapter total would over-report a pull that already has its weights,
/// which is the common case when a companion file is added to an adapter a
/// user already installed.
fn control_pending_download_bytes(
    adapter: &mold_core::ltx2_control::Ltx2ControlAdapter,
    path: &std::path::Path,
) -> u64 {
    control_missing_files(adapter, path)
        .iter()
        .map(|file| file.size_bytes)
        .sum()
}

pub(crate) async fn normalize_generation_placement(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) {
    let effective = state
        .config
        .read()
        .await
        .effective_placement(&request.model, request.placement.as_ref());
    request.placement = Some(effective);
}

/// Record an accepted generation prompt into prompt history (best-effort;
/// no-op when the metadata DB is disabled or the prompt is empty). Consecutive
/// identical rows are collapsed so batch siblings and retries don't spam
/// duplicates. Records what the user actually typed — callers capture the
/// prompt before `prepare_generation` runs prompt expansion.
pub(crate) fn record_prompt_history(
    state: &AppState,
    prompt: &str,
    negative: Option<&str>,
    model: &str,
) {
    let Some(db) = state.metadata_db.as_ref().as_ref() else {
        return;
    };
    record_prompt_history_in_db(db, prompt, negative, model);
}

fn record_prompt_history_in_db(
    db: &mold_db::MetadataDb,
    prompt: &str,
    negative: Option<&str>,
    model: &str,
) {
    // Video requests may legitimately carry no prompt at all; there is nothing
    // to recall later, so keep those rows out of history entirely (same rule as
    // the TUI's `History::push_entry`).
    if prompt.trim().is_empty() {
        return;
    }
    let history = mold_db::PromptHistory::new(db);
    if let Ok(rows) = history.recent(1) {
        if rows.first().is_some_and(|latest| {
            latest.prompt == prompt
                && latest.model == model
                && latest.negative.as_deref() == negative
        }) {
            return;
        }
    }
    if let Err(e) = history.push(&mold_db::HistoryEntry {
        prompt: prompt.to_string(),
        negative: negative.map(str::to_string),
        model: model.to_string(),
        created_at_ms: 0, // stamped with now() by push()
    }) {
        tracing::warn!("failed to record prompt history: {e:#}");
    }
}

pub(crate) async fn resolve_server_local_media_paths(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    if request.audio_file_path.is_none()
        && request.source_video_path.is_none()
        && request.extend_video_path.is_none()
    {
        return Ok(());
    }

    let roots = state.config.read().await.resolved_media_roots();
    if let Some(path) = request.audio_file_path.as_deref() {
        let resolved = mold_core::resolve_server_media_path(path, &roots)
            .map_err(|e| ApiError::validation(format!("audio_file_path: {e}")))?;
        request.audio_file_path = Some(resolved.to_string_lossy().to_string());
    }
    if let Some(path) = request.source_video_path.as_deref() {
        let resolved = mold_core::resolve_server_media_path(path, &roots)
            .map_err(|e| ApiError::validation(format!("source_video_path: {e}")))?;
        request.source_video_path = Some(resolved.to_string_lossy().to_string());
    }
    if let Some(path) = request.extend_video_path.as_deref() {
        let resolved = mold_core::resolve_server_media_path(path, &roots)
            .map_err(|e| ApiError::validation(format!("extend_video_path: {e}")))?;
        request.extend_video_path = Some(resolved.to_string_lossy().to_string());
    }

    Ok(())
}

fn active_gpu_selection(state: &AppState) -> GpuSelection {
    let selectors: Vec<mold_core::types::GpuSelector> = state
        .gpu_pool
        .workers
        .iter()
        .map(|worker| {
            worker.gpu.stable_id.as_ref().map_or_else(
                || mold_core::types::GpuSelector::Ordinal(worker.gpu.ordinal),
                |id| mold_core::types::GpuSelector::Identifier(id.clone()),
            )
        })
        .collect();
    if selectors.is_empty() {
        GpuSelection::None
    } else {
        GpuSelection::Specific(selectors)
    }
}

pub(crate) fn validate_multi_gpu_placement(
    state: &AppState,
    placement: Option<&mold_core::types::DevicePlacement>,
) -> Result<Option<usize>, ApiError> {
    state
        .gpu_pool
        .resolve_explicit_placement_gpu(placement)
        .map_err(ApiError::validation)
}

async fn clear_global_upscaler_cache(state: &AppState) {
    if state.gpu_pool.worker_count() > 0 {
        // GPU-worker mode never populates the legacy global cache. All
        // upscaler engines are created and dropped inside an owner command.
        return;
    }
    let cache = state.upscaler_cache.clone();
    if let Err(error) = tokio::task::spawn_blocking(move || {
        if let Ok(mut cache) = cache.try_lock() {
            if let Some(mut engine) = cache.take() {
                engine.unload();
                tracing::info!("upscaler cache cleared");
            }
        }
    })
    .await
    {
        tracing::warn!(%error, "upscaler cache teardown task panicked");
    }
}

/// Hand back the host RAM an unload just made unreachable.
///
/// `ModelCache::unload_active` parks the engine, which drops its loaded state,
/// but two things survive that on their own: component weight maps the shared
/// pool is keeping alive for reuse, and freed pages sitting in idle glibc
/// arenas. Neither is visible to the caller, and before #1273 the observable
/// result was an unload that changed the process's RSS by nothing at all while
/// host admission kept refusing work. Release what nothing references and trim,
/// so `DELETE /api/models/unload` is a real recovery action.
///
/// Entries an engine still streams from (SD3's offloaded MMDiT) have a live
/// strong count and are left alone.
pub(crate) fn release_host_memory_after_unload(state: &AppState) {
    let released = state
        .shared_pool
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .release_unreferenced_cpu_tensors();
    let rss_pre_trim = crate::gpu_worker::trim_malloc_arenas();
    let rss_after = crate::resources::ram_snapshot_from_system().used_by_mold;
    tracing::info!(
        shared_pool_released_mb = released / 1_000_000,
        rss_pre_trim_mb = rss_pre_trim.map(|value| value / 1_000_000).unwrap_or(0),
        rss_after_mb = rss_after / 1_000_000,
        "released host memory after model unload"
    );
}

pub(crate) async fn schedule_standalone_upscale(
    state: &AppState,
    model: String,
    weights_path: std::path::PathBuf,
    request: mold_core::UpscaleRequest,
    progress_tx: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<mold_core::UpscaleResponse, ApiError> {
    let id = format!("standalone-upscale-{}", uuid::Uuid::new_v4());
    let estimated_vram_bytes = std::fs::metadata(&weights_path)
        .map(|metadata| metadata.len().saturating_add(2 << 30))
        .unwrap_or(2 << 30);
    let artifact_root = state.config.read().await.resolved_models_dir();
    let utility_plans =
        crate::scheduler::upscale_candidates(state, &model, &weights_path, Some(&artifact_root))
            .map_err(|error| {
                ApiError::generation_unavailable(format!(
                    "upscaler execution plan could not be frozen: {error}"
                ))
            })?;
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let job = crate::gpu_pool::StandaloneUpscaleJob {
        id: id.clone(),
        model: model.clone(),
        weights_path,
        request,
        progress_tx,
        cancellation: mold_inference::InferenceCancellationToken::default(),
        execution_plan: None,
        result_tx,
    };
    let work = crate::gpu_pool::OwnerWork::StandaloneUpscale(Box::new(job));
    state
        .scheduled_work
        .submit(
            crate::scheduler::ScheduledOwnerWork::new(id, model, estimated_vram_bytes, work)
                .with_utility_plans(utility_plans),
        )
        .await
        .map_err(ApiError::generation_unavailable)?;
    result_rx
        .await
        .map_err(|_| ApiError::internal("upscale owner worker dropped its result"))?
        .map_err(|error| ApiError::internal(format!("upscale failed: {error}")))
}

async fn schedule_local_expansion(
    state: &AppState,
    config: mold_core::Config,
    settings: mold_core::ExpandSettings,
    prompt: String,
    expand_config: mold_core::ExpandConfig,
    preferred_gpu: Option<usize>,
) -> Result<mold_core::ExpandResult, ApiError> {
    require_expand_model_activation(&settings)?;
    let id = format!("prompt-expansion-{}", uuid::Uuid::new_v4());
    let estimated_vram_bytes = 6_000_000_000;
    let model = settings.model.clone();
    #[cfg(feature = "expand")]
    let utility_plans = crate::scheduler::prompt_expansion_candidates(state, &config, Some(&model))
        .map_err(|error| {
            ApiError::generation_unavailable(format!(
                "prompt expansion execution plan could not be frozen: {error}"
            ))
        })?;
    let cancellation = mold_inference::InferenceCancellationToken::default();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let work = crate::gpu_pool::OwnerWork::PromptExpansion(Box::new(
        crate::gpu_pool::PromptExpansionJob {
            id: id.clone(),
            parent_id: id.clone(),
            config,
            settings,
            prompt,
            expand_config,
            cancellation,
            #[cfg(feature = "expand")]
            execution_plan: None,
            result_tx,
        },
    ));
    state
        .scheduled_work
        .submit(
            crate::scheduler::ScheduledOwnerWork::new(id, model, estimated_vram_bytes, work)
                .with_hard_ordinal(preferred_gpu)
                .with_utility_plans({
                    #[cfg(feature = "expand")]
                    {
                        utility_plans
                    }
                    #[cfg(not(feature = "expand"))]
                    {
                        Vec::new()
                    }
                }),
        )
        .await
        .map_err(ApiError::generation_unavailable)?;
    result_rx
        .await
        .map_err(|_| ApiError::internal("prompt expansion owner worker dropped its result"))?
        .map_err(|error| ApiError::internal(format!("prompt expansion failed: {error}")))
}

fn require_expand_model_activation(settings: &mold_core::ExpandSettings) -> Result<(), ApiError> {
    mold_core::require_model_activation(settings.active_model(), None)
        .map_err(ApiError::model_activation)
}

// ── /api/generate ─────────────────────────────────────────────────────────────

pub(crate) fn canonical_client_batch_id(value: &str) -> Result<String, ApiError> {
    uuid::Uuid::parse_str(value.trim())
        .map(|id| id.to_string())
        .map_err(|_| ApiError::validation("client_batch_id must be a UUID"))
}

/// `hdr_exr_dir` names an output directory on the machine doing inference.
///
/// An HTTP client must never choose that server-local path: unlike media
/// inputs there is no useful remote artifact to return and no safe root to
/// resolve it beneath. Forced-local CLI generation bypasses this boundary and
/// continues to own EXR export and metadata recording.
///
/// Asked at ADMISSION, before the row is durable, because it is a property of
/// the request that no amount of retrying changes — deferring it to
/// preparation would turn an actionable `422` into an accepted job that holds.
/// The first LoRA this request would load whose file is not readable.
///
/// Camera-control aliases resolve through the config and are checked by their
/// own materialization, so only plain paths are asked here. A zero-scale entry
/// is skipped for the same reason `effective_loras` drops it: it is never
/// merged, so its file is never opened.
fn missing_lora_path(
    config: &mold_core::Config,
    request: &mold_core::GenerateRequest,
) -> Option<String> {
    const ZERO_SCALE_EPS: f64 = 1e-8;
    let loras = request
        .loras
        .as_ref()
        .filter(|stack| !stack.is_empty())
        .cloned()
        .or_else(|| request.lora.clone().map(|lora| vec![lora]))?;
    let _ = config;
    loras
        .into_iter()
        .filter(|lora| lora.scale.abs() > ZERO_SCALE_EPS)
        .find(|lora| {
            !lora.path.starts_with("camera-control:") && !std::path::Path::new(&lora.path).is_file()
        })
        .map(|lora| lora.path)
}

pub(crate) fn reject_client_supplied_hdr_output(
    request: &mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    if request.hdr_exr_dir.is_some() {
        return Err(ApiError::validation(
            "hdr_exr_dir is local-only and cannot be set through the server API; \
             re-run the CLI with --local so the EXR sidecar is written on your machine",
        ));
    }
    Ok(())
}

fn validate_direct_generation_request(
    request: &mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    if request.batch_size != 1 {
        return Err(ApiError::with_code(
            "direct generation accepts one output; submit durable singleton siblings through /api/generation-batches",
            "DIRECT_BATCH_UNSUPPORTED",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    Ok(())
}

/// Admit one singleton through the single durable path, or refuse.
///
/// There is no second pipeline. A host that cannot admit durably does not
/// generate, so every gate here is a refusal rather than a fallback: silently
/// running a non-durable render on a host whose queue cannot replay it is
/// exactly the ambiguity this route exists to remove.
/// The identity reference media is resolved under: the API-key identity when
/// one authenticated the request, the explicit auth-disabled host otherwise,
/// and nothing at all when neither holds (a router without the auth layers).
fn reference_identity(
    state: &AppState,
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
    auth_state: Option<&Extension<crate::auth::AuthState>>,
) -> Option<crate::reference_uploads::ReferenceIdentity> {
    crate::reference_uploads::ReferenceIdentity::resolve(
        authenticated.map(|Extension(auth)| auth),
        auth_state.map(|Extension(auth_state)| auth_state),
        state.instance_id.as_str(),
    )
}

pub(crate) async fn direct_durable_admission(
    state: &AppState,
    request: &mut mold_core::GenerateRequest,
) -> Result<Arc<crate::queue_media_admission::DurableMediaAdmission>, ApiError> {
    // Resolve the small, config-only control contracts before availability or
    // encrypted-media gates. This preserves precise 422 diagnostics without
    // doing model, placement, download, or device work before durable ack.
    let _ = plan_builtin_ltx2_control(state, request).await?;
    let _ = plan_builtin_ltx2_camera_controls(state, request).await?;
    crate::queue_media_admission::durable_media_preflight(request)?;
    // Asked before readiness: a host on its way down is a more specific — and
    // more actionable — answer than "this host cannot admit durably", and it
    // carries the `Retry-After` that tells the caller to come back. The batch
    // route asks the same question at the top of `admit_batch`.
    if state.queue_journal.is_retaining() {
        return Err(ApiError::server_restarting(
            "server is restarting; this generation was not accepted",
        ));
    }
    ensure_generation_available(state)?;
    let config = state.config.read().await;
    // One resolved value, read once, by every consumer.
    let readiness = DurableAdmissionReadiness::resolve(state, &config);
    drop(config);
    if let Some(unready) = readiness.unready() {
        return Err(unready.api_error());
    }
    // The encrypted store is a separate axis from the four conjuncts: the
    // admission service can be installed while the store is still degraded.
    // It gates MEDIA, not generation, so a media-free request is unaffected.
    if !readiness.media_ready()
        && crate::queue_media_admission::request_requires_encrypted_durable_media(request)
    {
        return Err(ApiError::with_code(
            "encrypted durable request media is unavailable",
            "DURABLE_MEDIA_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    readiness
        .admission()
        .ok_or_else(|| DurableAdmissionUnready::AdmissionServiceMissing.api_error())
}

fn durable_reconciliation_response(
    status: mold_core::GenerationBatchStatus,
    reason: &'static str,
) -> Response {
    tracing::warn!(
        batch_id = %status.id,
        client_batch_id = %status.client_batch_id,
        reason,
        "durable direct observer detached; returning reconciliation state"
    );
    (StatusCode::ACCEPTED, Json(status)).into_response()
}

/// Header through which a direct-facade caller names its own idempotency
/// key. It IS the batch's `client_batch_id`: a retry of a lost response under
/// the same value is answered with the batch the first attempt admitted.
pub(crate) const CLIENT_BATCH_ID_HEADER: &str = "x-mold-client-batch-id";

/// The `client_batch_id` a direct facade admits under: the caller's own when
/// it sent one, otherwise a fresh key the caller cannot replay.
fn direct_client_batch_id(headers: &HeaderMap) -> Result<String, ApiError> {
    let Some(value) = headers.get(CLIENT_BATCH_ID_HEADER) else {
        return Ok(uuid::Uuid::new_v4().to_string());
    };
    let text = value
        .to_str()
        .map_err(|_| ApiError::validation("X-Mold-Client-Batch-Id must be a UUID"))?;
    canonical_client_batch_id(text)
        .map_err(|_| ApiError::validation("X-Mold-Client-Batch-Id must be a UUID"))
}

/// Answer the idempotent replay of a direct facade: the batch an earlier POST
/// under this `client_batch_id` admitted, as JSON on both facades. A raw
/// caller reads the gallery filename off the completed child; there is no
/// second render to stream.
fn direct_replay_response(status: mold_core::GenerationBatchStatus) -> Response {
    (StatusCode::OK, Json(status)).into_response()
}

/// The error a failed direct generation answers with while its caller is
/// still attached: the singleton contract's own shape — a 404 carrying the
/// held child's typed code for a model the host cannot resolve, 503 for a
/// saturated queue, otherwise 500 with the engine's sentence — naming the
/// durable job the caller can resume, because the row is parked, not lost.
fn direct_generation_failure(status: &mold_core::GenerationBatchStatus, error: String) -> ApiError {
    let child = status.children.first();
    let message = child
        .and_then(|child| child.error.clone())
        .filter(|sentence| !sentence.is_empty())
        .unwrap_or(error);
    let message = match child {
        Some(child) if child.retryable == Some(true) => format!(
            "{message}. Durable job {job} is held and can be resumed with POST /api/queue/{job}/retry, or reconciled as batch {batch}",
            job = child.job_id,
            batch = status.id
        ),
        Some(child)
            if matches!(
                child.state,
                mold_core::GenerationBatchChildState::Failed
                    | mold_core::GenerationBatchChildState::Cancelled
                    | mold_core::GenerationBatchChildState::Held
            ) =>
        {
            format!(
                "{message}. Durable job {job} settled as {state}; reconcile it as batch {batch}",
                job = child.job_id,
                state = format!("{:?}", child.state).to_lowercase(),
                batch = status.id
            )
        }
        // The refreshed row could not be read: name the identity without
        // claiming a state this response has not seen.
        Some(child) => format!(
            "{message}. Durable job {job} belongs to batch {batch}; reconcile it there",
            job = child.job_id,
            batch = status.id
        ),
        None => message,
    };
    match child.and_then(|child| child.error_code.as_deref()) {
        Some(
            code @ (mold_core::SSE_ERROR_CODE_MODEL_NOT_FOUND
            | mold_core::SSE_ERROR_CODE_UNKNOWN_MODEL),
        ) => ApiError::with_code(message, code, StatusCode::NOT_FOUND),
        Some("QUEUE_FULL") => ApiError::queue_full(message),
        _ if message.contains("queue is full") => ApiError::queue_full(message),
        _ => ApiError::inference(message),
    }
}

const MAX_HETEROGENEOUS_BATCH_OUTPUTS: usize = 64;
const MAX_GENERATION_BATCH_STATUS_IDENTITIES: usize = 256;
const GENERATION_BATCH_STATUS_BODY_BYTES: usize = 64 * 1024;

pub(crate) fn generation_batch_status(
    instance_id: &str,
    detail: mold_db::generation_batches::DurableGenerationBatchDetail,
) -> mold_core::GenerationBatchStatus {
    use mold_core::GenerationBatchChildState as State;
    let created_at_ms = detail.batch.created_at_ms;
    mold_core::GenerationBatchStatus {
        id: detail.batch.id,
        client_batch_id: detail.batch.client_batch_id,
        instance_id: instance_id.to_string(),
        durable: true,
        children: detail
            .children
            .into_iter()
            .map(|child| {
                let (state, corrupt_state_error) = match child.state.as_str() {
                    "accepted" | "queued" => (State::Accepted, None),
                    "cancelling" => (State::Cancelling, None),
                    "running" => (State::Running, None),
                    "complete" => (State::Complete, None),
                    "failed" => (State::Failed, None),
                    "cancelled" => (State::Cancelled, None),
                    "held" => (State::Held, None),
                    unknown => (
                        State::Failed,
                        Some(format!("invalid durable child state '{unknown}'")),
                    ),
                };
                let terminal_error = child
                    .terminal_error_json
                    .as_deref()
                    .and_then(|value| serde_json::from_str(value).ok())
                    .filter(serde_json::Value::is_object)
                    .or_else(|| {
                        child
                            .error
                            .as_deref()
                            .map(|message| serde_json::json!({ "message": message }))
                    })
                    .or_else(|| {
                        corrupt_state_error
                            .as_deref()
                            .map(|message| serde_json::json!({ "message": message }))
                    });
                let result = child
                    .result_json
                    .as_deref()
                    .and_then(|value| serde_json::from_str(value).ok());
                mold_core::GenerationBatchChild {
                    index: child.batch_index,
                    job_id: child.job_id,
                    state,
                    error: child.error.or(corrupt_state_error),
                    error_code: (child.state == "held")
                        .then_some(child.error_code)
                        .flatten(),
                    retryable: (child.state == "held").then_some(child.retryable),
                    created_at_ms,
                    updated_at_ms: child.updated_at_ms,
                    revision: child.revision.max(0) as u64,
                    completed_at_ms: child.completed_at_ms,
                    terminal_error,
                    result,
                }
            })
            .collect(),
    }
}

/// Admit distinct prepared prompts as one durable, idempotent parent. Every
/// child validates before the single DB commit; after admission children run
/// independently through the ordinary generation queue.
#[utoipa::path(
    post,
    path = "/api/generation-batches",
    tag = "generation",
    request_body = mold_core::GenerationBatchAdmissionRequest,
    params((
        "X-Mold-Retained-Media-Session" = Option<String>,
        Header,
        description = "One-time same-host retained-media authority; valid only for a singleton batch whose child exactly matches the session-bound request"
    )),
    responses(
        (status = 202, description = "Every child durably admitted", body = mold_core::GenerationBatchStatus),
        (status = 200, description = "Idempotent replay of an admitted batch", body = mold_core::GenerationBatchStatus),
        (status = 409, description = "Client batch id reused with changed requests"),
        (status = 422, description = "A child is invalid; nothing admitted"),
        (status = 503, description = "Durable heterogeneous admission unavailable"),
    )
)]
async fn admit_generation_batch(
    State(state): State<AppState>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    headers: HeaderMap,
    Json(mut body): Json<mold_core::GenerationBatchAdmissionRequest>,
) -> Result<(StatusCode, Json<mold_core::GenerationBatchStatus>), ApiError> {
    // Every conjunct at one evaluation point. This used to be two statements
    // separated by the config read, which made the one correct site correct by
    // sequencing rather than by construction.
    let config = state.config.read().await;
    let readiness = DurableAdmissionReadiness::resolve(&state, &config);
    drop(config);
    if let Some(unready) = readiness.unready() {
        return Err(unready.api_error());
    }
    body.client_batch_id = canonical_client_batch_id(&body.client_batch_id)?;
    if body.requests.is_empty() || body.requests.len() > MAX_HETEROGENEOUS_BATCH_OUTPUTS {
        return Err(ApiError::validation(format!(
            "requests must contain 1..={MAX_HETEROGENEOUS_BATCH_OUTPUTS} children"
        )));
    }
    // A host can never refuse one route for a reason it accepts on another:
    // maintenance mode (every device disabled) refuses `/api/generate`, so it
    // refuses NEW batch work too rather than parking rows nothing will run. An
    // idempotent replay of an operation this host already holds still answers
    // with that operation — nothing new is queued, and a client reconciling a
    // lost response must not be told the host is unavailable for its own work.
    if let Err(unavailable) = ensure_generation_available(&state) {
        let client_batch_id = canonical_client_batch_id(&body.client_batch_id)?;
        let journal = state.queue_journal.clone();
        let existing = spawn_queue_read(move || {
            journal
                .durable_generation_batch_by_client(&client_batch_id)
                .map_err(anyhow::Error::msg)
        })
        .await?;
        return match existing {
            Some(detail) => Ok((
                StatusCode::OK,
                Json(generation_batch_status(&state.instance_id, detail)),
            )),
            None => Err(unavailable),
        };
    }
    let retained_reuse = crate::gallery_source_media::validate_reuse_batch_cardinality(
        &headers,
        body.requests.len(),
    )?;
    if retained_reuse && state.queue_journal.durable_media_capabilities().is_none() {
        return Err(ApiError::with_code(
            "encrypted durable request media is unavailable",
            "DURABLE_MEDIA_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    if retained_reuse {
        let request = body.requests.first_mut().expect("length checked above");
        mold_core::minimax_h3::canonicalize_request_model(request);
        crate::gallery_source_media::hydrate_reuse_session(
            &state,
            authenticated.as_ref(),
            auth_state.as_ref(),
            &headers,
            request,
        )
        .await?;
    }
    if state.queue_journal.durable_media_capabilities().is_none()
        && body
            .requests
            .iter()
            .any(crate::queue_media_admission::request_requires_encrypted_durable_media)
    {
        return Err(ApiError::with_code(
            "encrypted durable request media is unavailable",
            "DURABLE_MEDIA_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    let admission = state
        .queue_journal
        .queue_media_admission()
        .ok_or_else(|| ApiError::internal("durable admission service is unavailable"))?;
    let outcome = admission
        .admit_batch(
            &state,
            authenticated.as_ref().map(|Extension(auth)| auth),
            reference_identity(&state, authenticated.as_ref(), auth_state.as_ref()),
            body,
            None,
            SseCompletionPayload::MetadataOnly,
        )
        .await?;
    Ok((outcome.status_code, Json(outcome.status)))
}

#[utoipa::path(
    get,
    path = "/api/generation-batches/{id}",
    tag = "generation",
    params(("id" = String, Path, description = "Generation batch id")),
    responses(
        (status = 200, description = "Authoritative child states", body = mold_core::GenerationBatchStatus),
        (status = 404, description = "Batch not found"),
    )
)]
async fn get_generation_batch(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<mold_core::GenerationBatchStatus>, ApiError> {
    let journal = state.queue_journal.clone();
    let detail = spawn_queue_read(move || {
        journal
            .durable_generation_batch(&id)
            .map_err(anyhow::Error::msg)
    })
    .await?
    .ok_or_else(|| {
        ApiError::with_code(
            "generation batch not found",
            "GENERATION_BATCH_NOT_FOUND",
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(Json(generation_batch_status(&state.instance_id, detail)))
}

/// Cancel every non-terminal child of one durable batch.
///
/// A batch's children are independent durable rows, so this is exactly the
/// per-child cancel applied to each of them under ONE durable transition —
/// there is no second cancellation path. Running inference stops at the next
/// model safe point, so the returned status is the authority as of the
/// revocation rather than a promise that every child has already stopped; a
/// child that settled first is left alone.
#[utoipa::path(
    delete,
    path = "/api/generation-batches/{id}",
    tag = "generation",
    params(("id" = String, Path, description = "Generation batch id")),
    responses(
        (status = 200, description = "Cancellation accepted; authoritative child states", body = mold_core::GenerationBatchStatus),
        (status = 404, description = "Batch not found"),
    )
)]
async fn cancel_generation_batch(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<mold_core::GenerationBatchStatus>, ApiError> {
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let journal = state.queue_journal.clone();
    let probe_id = id.clone();
    let detail = spawn_queue_read(move || {
        journal
            .durable_generation_batch(&probe_id)
            .map_err(anyhow::Error::msg)
    })
    .await?
    .ok_or_else(|| {
        ApiError::with_code(
            "generation batch not found",
            "GENERATION_BATCH_NOT_FOUND",
            StatusCode::NOT_FOUND,
        )
    })?;
    let pending: Vec<String> = detail
        .children
        .iter()
        .filter(|child| !matches!(child.state.as_str(), "complete" | "failed" | "cancelled"))
        .map(|child| child.job_id.clone())
        .collect();
    for job_id in pending {
        cancel_one_queue_job(&state, &job_id).await?;
    }
    let journal = state.queue_journal.clone();
    let detail = spawn_queue_read(move || {
        journal
            .durable_generation_batch(&id)
            .map_err(anyhow::Error::msg)
    })
    .await?
    .ok_or_else(|| {
        ApiError::with_code(
            "generation batch not found",
            "GENERATION_BATCH_NOT_FOUND",
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(Json(generation_batch_status(&state.instance_id, detail)))
}

/// Live authoritative state for one durable batch.
///
/// A batch's children are independent durable rows, so the thing a client
/// needs streamed is the AUTHORITATIVE state each one commits — which is
/// exactly what `ServerEvent::JobStateCommitted` announces, after the SQLite
/// transaction. Each announcement re-reads the batch and emits the whole
/// status, so a client that connects late, reconnects, or misses a frame is
/// correct from the first event it sees rather than having to replay a delta
/// log.
///
/// Deliberately NOT per-step progress or denoise previews: those ride the
/// single-consumer observer a job's own admission registered, and one job has
/// exactly one. `/api/generate/stream` is that consumer for a singleton, and
/// `GET /api/queue/{id}/preview` is the snapshot every other surface reads.
#[utoipa::path(
    get,
    path = "/api/generation-batches/{id}/events",
    tag = "generation",
    params(("id" = String, Path, description = "Generation batch id")),
    responses(
        (status = 200, description = "SSE stream of authoritative batch status"),
        (status = 404, description = "Batch not found"),
    )
)]
async fn generation_batch_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let status = read_generation_batch_status(&state, &id).await?;
    let mut events = state.events.subscribe();
    let children = status
        .children
        .iter()
        .map(|child| child.job_id.clone())
        .collect::<std::collections::HashSet<_>>();
    let stream = async_stream::stream! {
        let mut settled = generation_batch_is_settled(&status);
        yield Ok::<_, Infallible>(generation_batch_event(&status));
        while !settled {
            let event = match events.recv().await {
                Ok(event) => Some(event),
                // A lagged subscriber missed announcements, but every frame is
                // a whole status, so one fresh read restores it.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => None,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
            };
            if let Some(event) = event {
                let concerns_batch = match &event {
                    mold_core::ServerEvent::JobStateCommitted { id } => children.contains(id),
                    // One transaction settled several rows at once; ask rather
                    // than guess whether any of them was ours.
                    mold_core::ServerEvent::GenerationStatesCommitted => true,
                    _ => false,
                };
                if !concerns_batch {
                    continue;
                }
            }
            let Ok(status) = read_generation_batch_status(&state, &id).await else {
                return;
            };
            settled = generation_batch_is_settled(&status);
            yield Ok(generation_batch_event(&status));
        }
    };
    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text("ping"),
        )
        .into_response())
}

fn generation_batch_event(status: &mold_core::GenerationBatchStatus) -> axum::response::sse::Event {
    match serde_json::to_string(status) {
        Ok(data) => axum::response::sse::Event::default()
            .event("generation_batch")
            .data(data),
        Err(error) => axum::response::sse::Event::default().event("error").data(
            serde_json::json!({ "message": format!("failed to serialize batch status: {error}") })
                .to_string(),
        ),
    }
}

/// Every child has reached a state this server will not leave on its own. A
/// `held` child counts: it is waiting for an explicit
/// `POST /api/queue/{id}/retry`, not for this host.
fn generation_batch_is_settled(status: &mold_core::GenerationBatchStatus) -> bool {
    use mold_core::GenerationBatchChildState as ChildState;
    status.children.iter().all(|child| {
        matches!(
            child.state,
            ChildState::Complete | ChildState::Failed | ChildState::Cancelled | ChildState::Held
        )
    })
}

async fn read_generation_batch_status(
    state: &AppState,
    id: &str,
) -> Result<mold_core::GenerationBatchStatus, ApiError> {
    let journal = state.queue_journal.clone();
    let batch_id = id.to_string();
    let detail = spawn_queue_read(move || {
        journal
            .durable_generation_batch(&batch_id)
            .map_err(anyhow::Error::msg)
    })
    .await?
    .ok_or_else(|| {
        ApiError::with_code(
            "generation batch not found",
            "GENERATION_BATCH_NOT_FOUND",
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(generation_batch_status(&state.instance_id, detail))
}

#[utoipa::path(
    get,
    path = "/api/generation-batches/by-client/{client_batch_id}",
    tag = "generation",
    params(("client_batch_id" = String, Path, description = "Client-generated idempotency UUID")),
    responses(
        (status = 200, description = "Authoritative batch recovered by client id", body = mold_core::GenerationBatchStatus),
        (status = 404, description = "Batch not found"),
    )
)]
async fn get_generation_batch_by_client(
    State(state): State<AppState>,
    Path(client_batch_id): Path<String>,
) -> Result<Json<mold_core::GenerationBatchStatus>, ApiError> {
    let client_batch_id = canonical_client_batch_id(&client_batch_id)?;
    let journal = state.queue_journal.clone();
    let detail = spawn_queue_read(move || {
        journal
            .durable_generation_batch_by_client(&client_batch_id)
            .map_err(anyhow::Error::msg)
    })
    .await?
    .ok_or_else(|| {
        ApiError::with_code(
            "generation batch not found",
            "GENERATION_BATCH_NOT_FOUND",
            StatusCode::NOT_FOUND,
        )
    })?;
    Ok(Json(generation_batch_status(&state.instance_id, detail)))
}

#[utoipa::path(
    post,
    path = "/api/generation-batches/status",
    tag = "generation",
    request_body = mold_core::GenerationBatchStatusRequest,
    responses(
        (status = 200, description = "Authoritative statuses and explicit missing identities", body = mold_core::GenerationBatchStatusResponse),
    )
)]
async fn reconcile_generation_batches(
    State(state): State<AppState>,
    Json(body): Json<mold_core::GenerationBatchStatusRequest>,
) -> Result<Json<mold_core::GenerationBatchStatusResponse>, ApiError> {
    let client_batch_ids =
        canonical_generation_batch_status_ids(body.client_batch_ids, "client_batch_ids")?;
    let batch_ids = canonical_generation_batch_status_ids(body.batch_ids, "batch_ids")?;
    if client_batch_ids.len() + batch_ids.len() > MAX_GENERATION_BATCH_STATUS_IDENTITIES {
        return Err(ApiError::with_code(
            format!(
                "generation batch status accepts at most {MAX_GENERATION_BATCH_STATUS_IDENTITIES} unique identities"
            ),
            "GENERATION_BATCH_STATUS_LIMIT_EXCEEDED",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    let journal = state.queue_journal.clone();
    let lookup = spawn_queue_read(move || {
        journal
            .durable_generation_batches(&client_batch_ids, &batch_ids)
            .map_err(anyhow::Error::msg)
    })
    .await?;
    let instance_id = state.instance_id.as_ref().clone();
    Ok(Json(mold_core::GenerationBatchStatusResponse {
        batches: lookup
            .batches
            .into_iter()
            .map(|detail| generation_batch_status(&instance_id, detail))
            .collect(),
        instance_id,
        missing: mold_core::GenerationBatchMissing {
            client_batch_ids: lookup.missing_client_batch_ids,
            batch_ids: lookup.missing_batch_ids,
        },
    }))
}

fn canonical_generation_batch_status_ids(
    values: Vec<String>,
    field: &str,
) -> Result<Vec<String>, ApiError> {
    let mut seen = std::collections::HashSet::with_capacity(values.len());
    let mut canonical = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let id = canonical_client_batch_id(&value).map_err(|mut error| {
            error.error = format!("{field}[{index}]: must be a UUID");
            error
        })?;
        if seen.insert(id.clone()) {
            canonical.push(id);
        }
    }
    Ok(canonical)
}

#[utoipa::path(
    post,
    path = "/api/generate",
    tag = "generation",
    request_body = mold_core::GenerateRequest,
    params((
        "X-Mold-Client-Batch-Id" = Option<String>,
        Header,
        description = "Optional caller-chosen UUID used as the durable client_batch_id; a replay answers with the admitted batch status"
    ), (
        "X-Mold-Retained-Media-Session" = Option<String>,
        Header,
        description = "One-time same-host authority for hydrating selected retained source media into this exact request"
    )),
    responses(
        (status = 200, description = "Generated media bytes with the matching image/video/audio Content-Type. A replayed X-Mold-Client-Batch-Id answers 200 application/json with the admitted GenerationBatchStatus instead; read the Content-Type."),
        (status = 202, description = "Durable singleton accepted but the attached observer detached; reconcile the returned batch status", body = mold_core::GenerationBatchStatus),
        (status = 404, description = "Model not downloaded or unknown (MODEL_NOT_FOUND / UNKNOWN_MODEL); the held durable job is named in the error"),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Inference error"),
        (status = 503, description = "Generation queue full"),
    )
)]
// Direct generation is singleton-only. Multi-output clients submit durable
// singleton siblings through `/api/generation-batches`.
async fn generate(
    State(state): State<AppState>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    headers: HeaderMap,
    Json(mut req): Json<mold_core::GenerateRequest>,
) -> Result<Response, ApiError> {
    mold_core::minimax_h3::canonicalize_request_model(&mut req);
    crate::gallery_source_media::hydrate_reuse_session(
        &state,
        authenticated.as_ref(),
        auth_state.as_ref(),
        &headers,
        &mut req,
    )
    .await?;
    let client_batch_id = direct_client_batch_id(&headers)?;
    validate_direct_generation_request(&req)?;
    let admission = direct_durable_admission(&state, &mut req).await?;
    let outcome = admission
        .admit_batch(
            &state,
            authenticated.as_ref().map(|Extension(auth)| auth),
            reference_identity(&state, authenticated.as_ref(), auth_state.as_ref()),
            mold_core::GenerationBatchAdmissionRequest {
                client_batch_id,
                requests: vec![req],
            },
            Some(crate::queue_media_ingress::ObserverMode::Raw),
            SseCompletionPayload::Full,
        )
        .await?;
    if outcome.status_code == StatusCode::OK {
        return Ok(direct_replay_response(outcome.status));
    }
    let status = outcome.status;
    let warnings = outcome.warnings.unwrap_or_default();
    if outcome.observers.len() != status.children.len()
        || outcome.observers.iter().any(Option::is_none)
    {
        return Err(ApiError::internal(
            "durable admission returned without its direct observer",
        ));
    }
    let observer = outcome
        .observers
        .into_iter()
        .next()
        .flatten()
        .expect("checked above");
    let attached = match observer.attached().await {
        Ok(attached) => attached,
        Err(_) => {
            return Ok(durable_reconciliation_response(
                status,
                "observer detached before feeder handoff",
            ));
        }
    };
    let crate::queue_media_ingress::AttachedObserver::Raw {
        outcome,
        warnings: deferred_warnings,
    } = attached
    else {
        return Err(ApiError::internal(
            "durable raw observer received an SSE delivery",
        ));
    };
    let result = match outcome.await {
        Ok(crate::job_supervisor::SupervisedOutcome::Finished(result)) => *result,
        Ok(crate::job_supervisor::SupervisedOutcome::Cancelled) => {
            return Err(ApiError::cancelled(format!(
                "generation job {} was cancelled while queued",
                status.children[0].job_id
            )));
        }
        Err(_) => {
            return Ok(durable_reconciliation_response(
                status,
                "worker observer dropped before terminal result",
            ));
        }
    };
    if let Err(error) = result {
        // The row settled before the worker answered, so the refreshed batch
        // carries the held child's sentence and typed code; the caller is
        // still attached, so the failure is ITS error, in the singleton
        // contract's shape, rather than a reconciliation body.
        let batch_id = status.id.clone();
        let journal = state.queue_journal.clone();
        let refreshed = match spawn_queue_read(move || {
            journal
                .durable_generation_batch(&batch_id)
                .map_err(anyhow::Error::msg)
        })
        .await
        {
            Ok(Some(detail)) => generation_batch_status(&state.instance_id, detail),
            Ok(None) => status,
            Err(refresh_error) => {
                tracing::warn!(
                    batch = %status.id,
                    error = ?refresh_error,
                    "durable generation status refresh failed after a render error; answering from the admission identity"
                );
                status
            }
        };
        return Err(direct_generation_failure(&refreshed, error));
    }
    generation_result_response(result, merge_request_warnings(warnings, deferred_warnings))
}

fn generation_result_response(
    result: Result<crate::state::GenerationJobResult, String>,
    warnings: RequestWarnings,
) -> Result<Response, ApiError> {
    match result {
        Ok(job_result) => {
            let img = job_result.image;
            let response = job_result.response;
            let warnings = merge_render_warnings(warnings, &response.request_warnings);
            let content_type = HeaderValue::from_static(img.format.content_type());
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, content_type);
            headers.insert(
                "x-mold-seed-used",
                HeaderValue::from_str(&response.seed_used.to_string()).map_err(|e| {
                    ApiError::internal(format!("failed to serialize seed header: {e}"))
                })?,
            );
            if let Some(ordinal) = response.gpu {
                headers.insert(
                    "x-mold-gpu",
                    HeaderValue::from_str(&ordinal.to_string()).map_err(|e| {
                        ApiError::internal(format!("failed to serialize gpu header: {e}"))
                    })?,
                );
            }
            if !warnings.is_empty() {
                let mut set = |name: &'static str, text: String| match HeaderValue::from_str(
                    &text.replace('\n', " "),
                ) {
                    Ok(val) => {
                        headers.insert(name, val);
                    }
                    Err(e) => {
                        tracing::warn!("{name} could not be encoded as a header: {e}");
                    }
                };
                // Every advisory, for a client that wants them all …
                set(
                    "x-mold-request-warning",
                    warnings.all().collect::<Vec<_>>().join("; "),
                );
                // … and the dimension header keeps carrying only dimension
                // adjustments, which is what its documentation promises.
                if let Some(dimension) = warnings.dimension.clone() {
                    set("x-mold-dimension-warning", dimension);
                }
            }
            let output_data = apply_media_headers(&response, img, &mut headers);
            Ok((headers, output_data).into_response())
        }
        Err(err_msg) => {
            // The multi-GPU dispatcher sends a queue-full error through result_tx
            // when a per-worker channel is saturated; surface that as a proper 503
            // instead of the generic INFERENCE_ERROR 500.
            if err_msg.contains("queue is full") {
                Err(ApiError::queue_full(err_msg))
            } else {
                Err(ApiError::inference(err_msg))
            }
        }
    }
}

/// Pick the bytes the non-streaming `/api/generate` returns and stamp the
/// media headers a client needs to rebuild the typed response.
///
/// The queue hands every completed job back with a raster in
/// `GenerationJobResult.image`: a real still, a clip's thumbnail, or an audio
/// print's waveform tile. That tile exists so the queue and SSE pipeline have
/// something to lay out — it is never the artifact the caller asked for. A
/// branch that special-cases only video therefore answers an audio render
/// with `image/png` and drops the WAV entirely.
pub(crate) fn apply_media_headers(
    response: &mold_core::GenerateResponse,
    img: mold_core::ImageData,
    headers: &mut HeaderMap,
) -> Vec<u8> {
    // Narrowest probe first: mesh, then audio, then video. Each is missing
    // whatever the next probe keys on — a mesh has no sample rate and no
    // frames, an audio print has no frames — so a wider probe running first
    // falls through and answers with the sidecar tile instead of the
    // artifact the caller asked for.
    if let Some(mesh) = response.mesh.as_ref() {
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(mesh.format.content_type()),
        );
        let mut set = |name: &'static str, value: String| {
            if let Ok(v) = HeaderValue::from_str(&value) {
                headers.insert(name, v);
            }
        };
        // Stated rather than inferred, exactly as for audio: a caller may
        // omit `output_format` and let the server normalise a mesh family to
        // GLB, so the request is no evidence of what came back.
        set("x-mold-mesh-format", mesh.format.extension().to_string());
        set("x-mold-mesh-vertices", mesh.vertex_count.to_string());
        set("x-mold-mesh-faces", mesh.face_count.to_string());
        set("x-mold-mesh-textured", mesh.textured.to_string());
        // A mesh has no raster of its own. These are the poster tile's size,
        // which is what a gallery row records so the grid has a real aspect
        // ratio — the tile bytes cannot ride along in a body that is the GLB.
        let fmt_bounds = |bounds: [f32; 3]| format!("{},{},{}", bounds[0], bounds[1], bounds[2]);
        set("x-mold-mesh-bounds-min", fmt_bounds(mesh.bounds_min));
        set("x-mold-mesh-bounds-max", fmt_bounds(mesh.bounds_max));
        set("x-mold-mesh-poster-width", mesh.poster_width.to_string());
        set("x-mold-mesh-poster-height", mesh.poster_height.to_string());
        return mesh.data.clone();
    }

    if let Some(audio) = response.audio.as_ref() {
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(audio.format.content_type()),
        );
        let mut set = |name: &'static str, value: String| {
            if let Ok(v) = HeaderValue::from_str(&value) {
                headers.insert(name, v);
            }
        };
        // Stated rather than inferred: a caller may omit `output_format` and
        // let the server normalise an audio-only pipeline to wav, in which
        // case the request is not evidence of what came back.
        set("x-mold-audio-format", audio.format.extension().to_string());
        set("x-mold-audio-sample-rate", audio.sample_rate.to_string());
        set("x-mold-audio-channels", audio.channels.to_string());
        set("x-mold-audio-duration-ms", audio.duration_ms.to_string());
        // Audio has no raster of its own. These are the waveform tile's size,
        // which is what a gallery row records so the grid has a real aspect
        // ratio — the tile bytes themselves cannot ride along in the body.
        set(
            "x-mold-audio-thumbnail-width",
            audio.thumbnail_width.to_string(),
        );
        set(
            "x-mold-audio-thumbnail-height",
            audio.thumbnail_height.to_string(),
        );
        return audio.data.clone();
    }

    // For video responses, return the actual video data (not the thumbnail)
    // and send video metadata in headers so the client can reconstruct VideoData.
    if let Some(video) = response.video.as_ref() {
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(video.format.content_type()),
        );
        if let Ok(v) = HeaderValue::from_str(&video.frames.to_string()) {
            headers.insert("x-mold-video-frames", v);
        }
        if let Ok(v) = HeaderValue::from_str(&video.fps.to_string()) {
            headers.insert("x-mold-video-fps", v);
        }
        if let Ok(v) = HeaderValue::from_str(&video.width.to_string()) {
            headers.insert("x-mold-video-width", v);
        }
        if let Ok(v) = HeaderValue::from_str(&video.height.to_string()) {
            headers.insert("x-mold-video-height", v);
        }
        if let Some(pipeline) = video.pipeline {
            if let Ok(v) = HeaderValue::from_str(pipeline.as_str()) {
                headers.insert("x-mold-video-pipeline", v);
            }
        }
        if let Some(provenance) = video.pipeline_provenance_sha256.as_deref() {
            if let Ok(v) = HeaderValue::from_str(provenance) {
                headers.insert("x-mold-video-pipeline-provenance-sha256", v);
            }
        }
        if let Some(preprocessing) = video.source_preprocessing.as_ref() {
            if let Ok(json) = serde_json::to_string(preprocessing) {
                if let Ok(v) = HeaderValue::from_str(&json) {
                    headers.insert("x-mold-video-source-preprocessing", v);
                }
            }
        }
        // Runtime provenance is output authority (CLAUDE.md): the client-side
        // save must record what actually ran, so these ride the response like
        // `pipeline` does.
        if let Some(path) = video.attention_path.as_deref() {
            if let Ok(v) = HeaderValue::from_str(path) {
                headers.insert("x-mold-video-attention-path", v);
            }
        }
        if let Some(arm) = video.int8_arm.as_deref() {
            if let Ok(v) = HeaderValue::from_str(arm) {
                headers.insert("x-mold-video-int8-arm", v);
            }
        }
        if video.video_only == Some(true) {
            headers.insert("x-mold-video-video-only", HeaderValue::from_static("1"));
        }
        if video.has_audio {
            headers.insert("x-mold-video-has-audio", HeaderValue::from_static("1"));
        }
        if let Some(dur) = video.duration_ms {
            if let Ok(v) = HeaderValue::from_str(&dur.to_string()) {
                headers.insert("x-mold-video-duration-ms", v);
            }
        }
        if let Some(sr) = video.audio_sample_rate {
            if let Ok(v) = HeaderValue::from_str(&sr.to_string()) {
                headers.insert("x-mold-video-audio-sample-rate", v);
            }
        }
        if let Some(ch) = video.audio_channels {
            if let Ok(v) = HeaderValue::from_str(&ch.to_string()) {
                headers.insert("x-mold-video-audio-channels", v);
            }
        }
        return video.data.clone();
    }

    img.data
}

pub(crate) fn validate_generate_request(
    req: &mold_core::GenerateRequest,
    family_hint: Option<&str>,
    reference_form: mold_core::ReferenceForm,
) -> Result<(), String> {
    // The print title is embedded into provenance and folded into the output
    // filename, so refuse control characters / over-long titles before any
    // model work is paid for. An empty title means "untitled", not an error.
    if let Some(title) = req.title.as_deref() {
        mold_core::validate_print_title(title)?;
    }
    match reference_form {
        mold_core::ReferenceForm::Admitted => {
            mold_core::validate_generate_request_with_family(req, family_hint)
        }
        mold_core::ReferenceForm::Resolved => {
            mold_core::validate_resolved_generate_request_with_family(req, family_hint)
        }
    }
}

/// Enforce the per-model source-image contract at admission (#772), with the
/// engine's own wording — both failure modes used to surface only after the
/// user had paid for the UMT5 encode and the expert load.
///
/// The contract resolves manifest-first (cold tiers classify from their own
/// task structure) and falls back to the downloaded checkpoint's headers.
/// An unknown contract enforces nothing: the engine remains the authority
/// and its late error is no worse than today's behavior.
async fn enforce_source_image_capability(
    state: &AppState,
    request: &mold_core::GenerateRequest,
    resolved_family: Option<&str>,
) -> Result<(), ApiError> {
    // Wan probes the resolved checkpoint's own headers FIRST: `ModelPaths`
    // honors config/env path overrides, so the artifacts actually loaded can
    // differ from the manifest's task structure — the shape-driven read is
    // the engine's exact truth. The manifest stays the cold fallback (not
    // yet downloaded, unreadable headers). Non-wan families have no header
    // probe; their manifest contract binds directly — plain LTX-Video
    // declares Unsupported and its engine really does ignore an attached
    // image.
    let manifest_contract = mold_core::manifest::find_manifest(&request.model)
        .and_then(|manifest| manifest.defaults.source_image);
    let capability = if resolved_family == Some("wan") {
        let probed = {
            let config = state.config.read().await;
            mold_core::ModelPaths::resolve(&request.model, &config).and_then(|paths| {
                mold_inference::wan_source_image_capability(&paths.transformer, &paths.vae)
            })
        };
        probed.or(manifest_contract)
    } else {
        manifest_contract
    };
    // Keyframes (#779) and an extend (#783) carry the source frames too, so
    // the shared predicate owns the whole list — an extend's first frames come
    // from the tail of the clip it continues, and counting only an image left
    // admission refusing every Wan I2V continuation with the very contract
    // that makes the checkpoint extend-capable.
    let has_source = mold_core::validation::request_carries_source_frames(request);
    match mold_core::validation::source_image_contract_violation(
        resolved_family,
        &request.model,
        capability,
        has_source,
    ) {
        Some(message) => Err(ApiError::validation(message)),
        None => Ok(()),
    }
}

pub(crate) async fn apply_default_metadata_setting(
    state: &AppState,
    req: &mut mold_core::GenerateRequest,
) {
    if req.embed_metadata.is_some() {
        return;
    }

    let config = state.config.read().await;
    req.embed_metadata = Some(config.effective_embed_metadata(None));
}

/// Apply prompt expansion if `expand: true` is set on a generate request.
async fn maybe_expand_prompt(
    state: &AppState,
    req: &mut mold_core::GenerateRequest,
    preferred_gpu: Option<usize>,
    resolved_family: Option<&str>,
) -> Result<(), ApiError> {
    if req.expand != Some(true) {
        return Ok(());
    }
    // An empty prompt is a deliberate signal that the visual conditioning
    // carries the shot (see `mold_core::prompt_required_for`). Feeding "" to
    // the expander would let it invent a prompt that then becomes the frozen,
    // recorded one — expand nothing instead. Clear the flag rather than just
    // returning: scheduler-owned local expansion re-reads `request.expand`
    // when it plans the PromptExpansion dependency stage, so leaving it set
    // would hand "" to the expander one layer down.
    if req.prompt.trim().is_empty() {
        req.expand = Some(false);
        return Ok(());
    }

    let config = state.config.read().await;
    let config_snapshot = config.clone();
    let expand_settings = config.expand.clone().with_env_overrides();
    require_expand_model_activation(&expand_settings)?;
    if (state.scheduled_work.v2_authoritative() || state.gpu_pool.worker_count() > 0)
        && expand_settings.is_local()
    {
        // The scheduler owns local expansion as a PromptExpansion dependency
        // stage. Leaving the request untouched lets the parent enter the
        // queue immediately; the coordinator freezes the expanded prompt
        // before making Generation ready.
        return Ok(());
    }

    // Resolve model family for prompt style
    let model_family = resolved_family
        .map(str::to_owned)
        .or_else(|| config.resolved_model_config(&req.model).family)
        .or_else(|| mold_core::manifest::find_manifest(&req.model).map(|m| m.family.clone()))
        .unwrap_or_else(|| {
            tracing::warn!(
                model = %req.model,
                "could not resolve model family for prompt expansion, defaulting to \"flux\""
            );
            "flux".to_string()
        });

    let mut expand_config = expand_settings.to_expand_config(&model_family, 1);
    expand_config.task = mold_core::ExpandTask::for_generation(&model_family, req);
    let original_prompt = req.prompt.clone();

    // Drop config lock before blocking
    drop(config);

    let expander = create_server_expander(
        &config_snapshot,
        &expand_settings,
        active_gpu_selection(state),
        preferred_gpu,
    )?;
    let result =
        tokio::task::spawn_blocking(move || expander.expand(&original_prompt, &expand_config))
            .await
            .map_err(|e| ApiError::internal(format!("expand task failed: {e}")))?
            .map_err(|e| ApiError::internal(format!("prompt expansion failed: {e}")))?;

    if let Some(expanded) = result.expanded.first() {
        req.original_prompt = Some(req.prompt.clone());
        req.prompt = expanded.clone();
    }

    Ok(())
}

/// Create the appropriate expander for server-side use.
fn create_server_expander(
    _config: &mold_core::Config,
    settings: &mold_core::ExpandSettings,
    _gpu_selection: GpuSelection,
    _preferred_gpu: Option<usize>,
) -> Result<Box<dyn mold_core::PromptExpander>, ApiError> {
    require_expand_model_activation(settings)?;
    if let Some(api_expander) = settings
        .create_api_expander()
        .map_err(ApiError::model_activation)?
    {
        return Ok(Box::new(api_expander));
    }

    #[cfg(feature = "expand")]
    {
        match mold_inference::expand::LocalExpander::from_config(_config, Some(&settings.model)) {
            Some(local) => Ok(Box::new(
                local
                    .with_gpu_selection(_gpu_selection)
                    .with_preferred_gpu(_preferred_gpu),
            )),
            None => Err(ApiError::validation(
                "local expand model not found — run: mold pull qwen3-expand".to_string(),
            )),
        }
    }

    #[cfg(not(feature = "expand"))]
    {
        Err(ApiError::validation(
            "local prompt expansion not available — built without expand feature. \
             Configure an API backend in [expand] settings."
                .to_string(),
        ))
    }
}

/// Build the per-request expand config: settings-derived knobs plus the
/// request's optional visual style (bake-and-clear — the style reaches the
/// expander as a natural-language instruction, never a literal suffix).
fn expand_config_for_request(
    settings: &mold_core::ExpandSettings,
    req: &mold_core::ExpandRequest,
) -> mold_core::ExpandConfig {
    let mut config = settings.to_expand_config(&req.model_family, req.variations);
    config.style = req.style.clone();
    config.task = req
        .task
        .unwrap_or_else(|| mold_core::ExpandTask::for_family(&req.model_family));
    config
}

// ── /api/expand ──────────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/expand",
    tag = "generation",
    request_body = mold_core::ExpandRequest,
    responses(
        (status = 200, description = "Expanded prompt(s)", body = mold_core::ExpandResponse),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Expansion failed"),
    )
)]
async fn expand_prompt(
    State(state): State<AppState>,
    Json(req): Json<mold_core::ExpandRequest>,
) -> Result<Json<mold_core::ExpandResponse>, ApiError> {
    validate_expand_variations(req.variations)?;

    let config = state.config.read().await;
    let expand_settings = config.expand.clone().with_env_overrides();
    let expand_config = expand_config_for_request(&expand_settings, &req);
    let prompt = req.prompt.clone();
    let config_snapshot = config.clone();
    drop(config);

    if state.gpu_pool.worker_count() > 0 && expand_settings.is_local() {
        let result = schedule_local_expansion(
            &state,
            config_snapshot,
            expand_settings,
            prompt,
            expand_config,
            None,
        )
        .await?;
        return Ok(Json(mold_core::ExpandResponse {
            original: req.prompt,
            expanded: result.expanded,
        }));
    }

    let expander = create_server_expander(
        &config_snapshot,
        &expand_settings,
        active_gpu_selection(&state),
        None,
    )?;
    let result = tokio::task::spawn_blocking(move || expander.expand(&prompt, &expand_config))
        .await
        .map_err(|e| ApiError::internal(format!("expand task failed: {e}")))?
        .map_err(|e| ApiError::internal(format!("prompt expansion failed: {e}")))?;

    Ok(Json(mold_core::ExpandResponse {
        original: req.prompt,
        expanded: result.expanded,
    }))
}

#[utoipa::path(
    post,
    path = "/api/remix",
    tag = "generation",
    request_body = mold_core::RemixRequest,
    responses(
        (status = 200, description = "Subject-preserving prompt alternatives", body = mold_core::RemixResponse),
        (status = 422, description = "Invalid or conditioning-incompatible request"),
        (status = 500, description = "Remix failed"),
    )
)]
async fn remix_prompt(
    State(state): State<AppState>,
    Json(req): Json<mold_core::RemixRequest>,
) -> Result<Json<mold_core::RemixResponse>, ApiError> {
    validate_expand_variations(req.variations)?;
    if req.source_prompt.trim().is_empty() {
        return Err(ApiError::validation(
            "source_prompt must not be empty".to_string(),
        ));
    }
    let task = req
        .task
        .unwrap_or_else(|| mold_core::ExpandTask::for_family(&req.model_family));
    let dimensions = mold_core::expand::resolve_remix_dimensions(
        &req.dimensions,
        task,
        req.style
            .as_ref()
            .is_some_and(|style| !style.trim().is_empty()),
    )
    .map_err(|error| ApiError::validation(error.to_string()))?;

    let config = state.config.read().await;
    let expand_settings = config.expand.clone().with_env_overrides();
    let mut remix_config = expand_settings.to_expand_config(&req.model_family, req.variations);
    remix_config.task = task;
    remix_config.operation = mold_core::PromptTransformOperation::Remix;
    remix_config.remix_dimensions = dimensions.clone();
    remix_config.style = req.style.clone();
    // Expansion template overrides are intentionally not Remix overrides.
    remix_config.system_prompt = None;
    remix_config.batch_prompt = None;
    let prompt = req.source_prompt.clone();
    let config_snapshot = config.clone();
    drop(config);

    let result = if state.gpu_pool.worker_count() > 0 && expand_settings.is_local() {
        schedule_local_expansion(
            &state,
            config_snapshot,
            expand_settings,
            prompt,
            remix_config,
            None,
        )
        .await?
    } else {
        let expander = create_server_expander(
            &config_snapshot,
            &expand_settings,
            active_gpu_selection(&state),
            None,
        )?;
        tokio::task::spawn_blocking(move || expander.expand(&prompt, &remix_config))
            .await
            .map_err(|error| ApiError::internal(format!("remix task failed: {error}")))?
            .map_err(|error| ApiError::internal(format!("prompt remix failed: {error}")))?
    };

    let variants = result
        .expanded
        .into_iter()
        .enumerate()
        .map(|(index, prompt)| mold_core::RemixVariant {
            prompt,
            dimensions: mold_core::expand::remix_dimensions_for_position(&dimensions, index + 1),
        })
        .collect();
    Ok(Json(mold_core::RemixResponse {
        source_prompt: req.source_prompt,
        root_prompt: req.root_prompt,
        source_kind: req.source_kind,
        task,
        variants,
    }))
}

fn validate_expand_variations(variations: usize) -> Result<(), ApiError> {
    mold_core::expand::validate_expansion_variation_count(variations)
        .map_err(|error| ApiError::validation(error.to_string()))
}

// ── /api/upscale ────────────────────────────────────────────────────────────

async fn require_upscale_model_activation(
    state: &AppState,
    requested_model: &str,
    resolved_model: &str,
) -> Result<(), ApiError> {
    let advertised_family = require_server_model_activation(state, requested_model).await?;
    let (configured_family, configured_weights, artifact_root) = {
        let config = state.config.read().await;
        let configured = config.models.get(resolved_model);
        (
            configured.and_then(|model| model.family.clone()),
            configured.and_then(|model| model.transformer.clone()),
            config.resolved_models_dir(),
        )
    };
    let manifest = mold_core::manifest::find_manifest(resolved_model);
    let family = configured_family
        .as_deref()
        .or(advertised_family.as_deref())
        .or_else(|| manifest.map(|model| model.family.as_str()));

    mold_core::require_model_activation(requested_model, family)
        .map_err(ApiError::model_activation)?;
    mold_core::require_model_activation(resolved_model, family)
        .map_err(ApiError::model_activation)?;
    if let Some(weights) = configured_weights.as_deref() {
        mold_core::require_model_artifact_activation(
            std::path::Path::new(weights),
            Some(&artifact_root),
            family,
        )
        .map_err(ApiError::model_activation)?;
    }
    if let Some(manifest) = manifest {
        mold_core::require_registered_manifest_activation(manifest)
            .map_err(ApiError::model_activation)?;
    }
    Ok(())
}

async fn upscale(
    State(state): State<AppState>,
    Json(req): Json<mold_core::UpscaleRequest>,
) -> Result<Json<mold_core::UpscaleResponse>, ApiError> {
    ensure_generation_available(&state)?;
    let model_name = mold_core::manifest::resolve_model_name(&req.model);
    require_upscale_model_activation(&state, &req.model, &model_name).await?;
    if !state.scheduled_work.v2_authoritative() {
        ensure_schedulable_device(&state)?;
    }
    if let Err(msg) = mold_core::validate_upscale_request(&req) {
        return Err(ApiError::validation(msg));
    }

    // Auto-pull upscaler model if not downloaded
    let needs_pull = {
        let config = state.config.read().await;
        model_manager::upscaler_model_needs_pull(&config, &model_name)
    };
    if needs_pull {
        if mold_core::manifest::find_manifest(&model_name).is_none() {
            return Err(ApiError::not_found(format!(
                "unknown upscaler model '{}'. Run 'mold list' to see available models.",
                model_name
            )));
        }
        model_manager::pull_model(&state, &model_name, None).await?;
    }

    let config = state.config.read().await;
    let weights_path = config
        .models
        .get(&model_name)
        .and_then(|c| c.transformer.as_ref())
        .ok_or_else(|| {
            ApiError::not_found(format!(
                "upscaler model '{}' not configured after pull",
                model_name
            ))
        })?;
    let weights_path = std::path::PathBuf::from(weights_path);
    let artifact_root = config.resolved_models_dir();
    let model_name_owned = model_name.clone();
    drop(config);

    let resp = if state.scheduled_work.v2_authoritative() || state.gpu_pool.worker_count() > 0 {
        schedule_standalone_upscale(&state, model_name_owned, weights_path, req, None).await?
    } else {
        let upscaler_cache = state.upscaler_cache.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<mold_core::UpscaleResponse> {
            let mut cache = upscaler_cache.lock().unwrap_or_else(|e| e.into_inner());

            // Reuse cached engine if same model.
            let needs_new = cache
                .as_ref()
                .is_none_or(|e| e.model_name() != model_name_owned);
            if needs_new {
                let new_engine = mold_inference::create_upscale_engine(
                    model_name_owned,
                    weights_path,
                    Some(&artifact_root),
                    mold_inference::LoadStrategy::Eager,
                    0,
                )?;
                if let Some(mut old_engine) = cache.take() {
                    old_engine.unload();
                }
                *cache = Some(new_engine);
            }

            cache.as_mut().unwrap().upscale(&req)
        })
        .await
        .map_err(|e| ApiError::internal(format!("upscale task panicked: {e}")))?
        .map_err(|e| ApiError::internal(format!("upscale failed: {e}")))?
    };

    Ok(Json(resp))
}

// ── /api/upscale/stream (SSE) ──────────────────────────────────────────────

async fn upscale_stream(
    State(state): State<AppState>,
    Json(req): Json<mold_core::UpscaleRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    ensure_generation_available(&state)?;
    let model_name = mold_core::manifest::resolve_model_name(&req.model);
    require_upscale_model_activation(&state, &req.model, &model_name).await?;
    if !state.scheduled_work.v2_authoritative() {
        ensure_schedulable_device(&state)?;
    }
    if let Err(msg) = mold_core::validate_upscale_request(&req) {
        return Err(ApiError::validation(msg));
    }

    // Check if model needs pulling before spawning the SSE stream
    let needs_pull = {
        let config = state.config.read().await;
        model_manager::upscaler_model_needs_pull(&config, &model_name)
    };

    // Validate the model exists in the manifest if we need to pull
    if needs_pull && mold_core::manifest::find_manifest(&model_name).is_none() {
        return Err(ApiError::not_found(format!(
            "unknown upscaler model '{}'. Run 'mold list' to see available models.",
            model_name
        )));
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();
    let model_name_owned = model_name.clone();
    let state_clone = state.clone();
    let upscaler_cache = state.upscaler_cache.clone();

    tokio::spawn(async move {
        // Auto-pull the upscaler model if not downloaded
        if needs_pull {
            let progress_tx = tx.clone();
            let callback =
                std::sync::Arc::new(move |event: mold_core::download::DownloadProgressEvent| {
                    let sse_event = match event {
                        mold_core::download::DownloadProgressEvent::Status { message } => {
                            SseProgressEvent::Info { message }
                        }
                        mold_core::download::DownloadProgressEvent::FileStart {
                            filename,
                            file_index,
                            total_files,
                            size_bytes,
                            batch_bytes_downloaded,
                            batch_bytes_total,
                            batch_elapsed_ms,
                        } => SseProgressEvent::DownloadProgress {
                            filename,
                            file_index,
                            total_files,
                            bytes_downloaded: 0,
                            bytes_total: size_bytes,
                            batch_bytes_downloaded,
                            batch_bytes_total,
                            batch_elapsed_ms,
                        },
                        mold_core::download::DownloadProgressEvent::FileProgress {
                            filename,
                            file_index,
                            bytes_downloaded,
                            bytes_total,
                            batch_bytes_downloaded,
                            batch_bytes_total,
                            batch_elapsed_ms,
                        } => SseProgressEvent::DownloadProgress {
                            filename,
                            file_index,
                            total_files: 0,
                            bytes_downloaded,
                            bytes_total,
                            batch_bytes_downloaded,
                            batch_bytes_total,
                            batch_elapsed_ms,
                        },
                        mold_core::download::DownloadProgressEvent::FileDone {
                            filename,
                            file_index,
                            total_files,
                            batch_bytes_downloaded,
                            batch_bytes_total,
                            batch_elapsed_ms,
                        } => SseProgressEvent::DownloadDone {
                            filename,
                            file_index,
                            total_files,
                            batch_bytes_downloaded,
                            batch_bytes_total,
                            batch_elapsed_ms,
                        },
                    };
                    let _ = progress_tx.send(SseMessage::Progress(sse_event));
                });

            match model_manager::pull_model(&state_clone, &model_name_owned, Some(callback)).await {
                Ok(_) => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                        model: model_name_owned.clone(),
                    }));
                }
                Err(e) => {
                    let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
                        format!("failed to pull upscaler model: {}", e.error),
                    )));
                    return;
                }
            }
        }

        // Read weights path after potential pull
        let (weights_path, artifact_root) = {
            let config = state_clone.config.read().await;
            (
                config
                    .models
                    .get(&model_name_owned)
                    .and_then(|c| c.transformer.as_ref())
                    .map(std::path::PathBuf::from),
                config.resolved_models_dir(),
            )
        };

        let Some(weights_path) = weights_path else {
            let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
                format!(
                    "upscaler model '{}' not configured after pull",
                    model_name_owned
                ),
            )));
            return;
        };

        if state_clone.scheduled_work.v2_authoritative() || state_clone.gpu_pool.worker_count() > 0
        {
            match schedule_standalone_upscale(
                &state_clone,
                model_name_owned,
                weights_path,
                req,
                Some(tx.clone()),
            )
            .await
            {
                Ok(resp) => {
                    let image_b64 =
                        base64::engine::general_purpose::STANDARD.encode(&resp.image.data);
                    let _ = tx.send(SseMessage::UpscaleComplete(
                        mold_core::SseUpscaleCompleteEvent {
                            image: image_b64,
                            format: resp.image.format,
                            model: resp.model,
                            scale_factor: resp.scale_factor,
                            original_width: resp.original_width,
                            original_height: resp.original_height,
                            upscale_time_ms: resp.upscale_time_ms,
                        },
                    ));
                }
                Err(error) => {
                    let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
                        error.error,
                    )));
                }
            }
            return;
        }

        let result = {
            let model_name_for_cache = model_name_owned.clone();
            let weights_path_for_cache = weights_path.clone();
            let req_for_cache = req.clone();
            tokio::task::spawn_blocking(move || {
                let mut cache = upscaler_cache.lock().unwrap();

                let needs_new = cache
                    .as_ref()
                    .is_none_or(|e| e.model_name() != model_name_for_cache);
                if needs_new {
                    let _ = tx.send(SseMessage::Progress(
                        mold_core::SseProgressEvent::StageStart {
                            name: "Loading upscaler model".to_string(),
                        },
                    ));
                    match mold_inference::create_upscale_engine(
                        model_name_for_cache,
                        weights_path_for_cache,
                        Some(&artifact_root),
                        mold_inference::LoadStrategy::Eager,
                        0,
                    ) {
                        Ok(new_engine) => {
                            if let Some(mut old_engine) = cache.take() {
                                old_engine.unload();
                            }
                            *cache = Some(new_engine);
                        }
                        Err(e) => {
                            let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
                                format!("failed to load upscaler: {e}"),
                            )));
                            return;
                        }
                    }
                }

                let engine = cache.as_mut().unwrap();
                let tx_progress = tx.clone();
                engine.set_on_progress(Box::new(move |event| {
                    let sse_event: mold_core::SseProgressEvent = event.into();
                    let _ = tx_progress.send(SseMessage::Progress(sse_event));
                }));

                match engine.upscale(&req_for_cache) {
                    Ok(resp) => {
                        let image_b64 =
                            base64::engine::general_purpose::STANDARD.encode(&resp.image.data);
                        let _ = tx.send(SseMessage::UpscaleComplete(
                            mold_core::SseUpscaleCompleteEvent {
                                image: image_b64,
                                format: resp.image.format,
                                model: resp.model,
                                scale_factor: resp.scale_factor,
                                original_width: resp.original_width,
                                original_height: resp.original_height,
                                upscale_time_ms: resp.upscale_time_ms,
                            },
                        ));
                    }
                    Err(e) => {
                        let _ = tx.send(SseMessage::Error(mold_core::SseErrorEvent::failed(
                            format!("upscale failed: {e}"),
                        )));
                    }
                }

                engine.clear_on_progress();
            })
            .await
        };

        if let Err(e) = result {
            tracing::error!("upscale task panicked: {e}");
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, Infallible>(sse_message_to_event(msg)));

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
}

// ── /api/generate/stream (SSE) ───────────────────────────────────────────────

const SSE_PAYLOAD_HEADER: &str = "x-mold-sse-payload";

pub(crate) fn requested_sse_completion_payload(
    headers: &HeaderMap,
) -> Result<SseCompletionPayload, ApiError> {
    let Some(value) = headers.get(SSE_PAYLOAD_HEADER) else {
        return Ok(SseCompletionPayload::Full);
    };
    match value.to_str().ok() {
        Some("metadata-only") => Ok(SseCompletionPayload::MetadataOnly),
        _ => Err(ApiError::validation(
            "X-Mold-SSE-Payload must be 'metadata-only' when provided",
        )),
    }
}

#[utoipa::path(
    post,
    path = "/api/generate/stream",
    tag = "generation",
    request_body = mold_core::GenerateRequest,
    params((
        "X-Mold-SSE-Payload" = Option<String>,
        Header,
        description = "Set to metadata-only to omit encoded media and return the saved gallery filename"
    ), (
        "X-Mold-Client-Batch-Id" = Option<String>,
        Header,
        description = "Optional caller-chosen UUID used as the durable client_batch_id; a replay answers with the admitted batch status as JSON"
    ), (
        "X-Mold-Retained-Media-Session" = Option<String>,
        Header,
        description = "One-time same-host authority for hydrating selected retained source media into this exact request"
    )),
    responses(
        (status = 200, description = "SSE event stream with progress and result. A replayed X-Mold-Client-Batch-Id answers 200 application/json with the admitted GenerationBatchStatus instead; read the Content-Type."),
        (status = 404, description = "Model not downloaded"),
        (status = 422, description = "Invalid request parameters"),
        (status = 500, description = "Inference error"),
        (status = 503, description = "Generation queue full"),
    )
)]
async fn generate_stream(
    State(state): State<AppState>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    headers: HeaderMap,
    Json(mut req): Json<mold_core::GenerateRequest>,
) -> Result<Response, ApiError> {
    mold_core::minimax_h3::canonicalize_request_model(&mut req);
    crate::gallery_source_media::hydrate_reuse_session(
        &state,
        authenticated.as_ref(),
        auth_state.as_ref(),
        &headers,
        &mut req,
    )
    .await?;
    let completion_payload = requested_sse_completion_payload(&headers)?;
    let client_batch_id = direct_client_batch_id(&headers)?;
    validate_direct_generation_request(&req)?;
    let admission = direct_durable_admission(&state, &mut req).await?;
    let outcome = admission
        .admit_batch(
            &state,
            authenticated.as_ref().map(|Extension(auth)| auth),
            reference_identity(&state, authenticated.as_ref(), auth_state.as_ref()),
            mold_core::GenerationBatchAdmissionRequest {
                client_batch_id,
                requests: vec![req],
            },
            Some(crate::queue_media_ingress::ObserverMode::Sse(
                completion_payload,
            )),
            completion_payload,
        )
        .await?;
    if outcome.status_code == StatusCode::OK {
        return Ok(direct_replay_response(outcome.status));
    }
    let status = outcome.status;
    if outcome.observers.len() != status.children.len()
        || outcome.observers.iter().any(Option::is_none)
    {
        return Err(ApiError::internal(
            "durable admission returned without its direct observer",
        ));
    }
    let warnings = outcome.warnings.unwrap_or_default();
    let observer = outcome
        .observers
        .into_iter()
        .next()
        .flatten()
        .expect("checked above");
    let job_id = status.children[0].job_id.clone();
    let batch_id = status.id.clone();
    let client_batch_id = status.client_batch_id.clone();
    // The first frame carries the position AND the server id, so a client can
    // say "#N in line" before anything renders and reconcile against
    // `/api/queue` afterwards. Read on the REGISTRY scale — the count of live
    // jobs this admission now sits behind — because the durable row has not
    // been claimed by the feeder yet, so there is no dispatch position to
    // read. Seeding 0 unconditionally told every caller it was next up.
    let queued_position = state.job_registry.len();
    let stream = async_stream::stream! {
        for warning in warnings.all() {
            yield Ok::<_, Infallible>(sse_message_to_event(SseMessage::Progress(
                SseProgressEvent::Info { message: warning.to_string() }
            )));
        }
        yield Ok::<_, Infallible>(sse_message_to_event(SseMessage::Progress(
            SseProgressEvent::Queued { position: queued_position, id: job_id }
        )));
        let Ok(attached) = observer.attached().await else {
            yield Ok::<_, Infallible>(sse_message_to_event(SseMessage::Error(
                mold_core::SseErrorEvent::retained_with_code(
                    format!(
                        "durable generation remains queued; reconcile batch {batch_id} or client operation {client_batch_id}"
                    ),
                    mold_core::SSE_ERROR_CODE_DURABLE_OBSERVER_DETACHED,
                ),
            )));
            return;
        };
        let crate::queue_media_ingress::AttachedObserver::Sse { mut messages } = attached else {
            return;
        };
        let mut terminal = false;
        while let Some(message) = messages.recv().await {
            terminal = matches!(message, SseMessage::Complete(_) | SseMessage::Error(_));
            yield Ok::<_, Infallible>(sse_message_to_event(message));
            if terminal {
                break;
            }
        }
        if !terminal {
            yield Ok::<_, Infallible>(sse_message_to_event(SseMessage::Error(
                mold_core::SseErrorEvent::retained_with_code(
                    format!(
                        "durable generation remains queued; reconcile batch {batch_id} or client operation {client_batch_id}"
                    ),
                    mold_core::SSE_ERROR_CODE_DURABLE_OBSERVER_DETACHED,
                ),
            )));
        }
    };
    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text("ping"),
        )
        .into_response())
}

// ── /api/models ───────────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/models",
    tag = "models",
    responses(
        (status = 200, description = "List of available models", body = Vec<ModelInfoExtended>),
    )
)]
async fn list_models(
    State(state): State<AppState>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
) -> Json<Vec<ModelInfoExtended>> {
    let mut models = model_manager::list_models(&state).await;
    let api_key_auth_enabled = auth_state
        .as_ref()
        .is_some_and(|Extension(state)| state.is_some());
    let private_capability = {
        let config = state.config.read().await;
        advertised_private_h3_capability(
            api_key_auth_enabled,
            &config.resolved_models_dir(),
            &current_device_state(&state),
        )
    };
    for row in private_capability
        .as_ref()
        .map(authenticated_private_h3_model_rows)
        .unwrap_or_default()
    {
        models.retain(|entry| entry.info.name != row.info.name);
        models.push(row);
    }
    Json(models)
}

async fn generate_estimate(
    State(state): State<AppState>,
    Json(mut req): Json<GenerateRequest>,
) -> Result<Json<mold_core::GenerationMemoryEstimate>, ApiError> {
    mold_core::minimax_h3::canonicalize_request_model(&mut req);
    Ok(Json(
        model_manager::estimate_generation_memory(&state, &req).await?,
    ))
}

#[utoipa::path(
    post,
    path = "/api/generate/placement-preview",
    tag = "generation",
    request_body = mold_core::GenerationPlacementPreviewRequest,
    responses(
        (status = 200, description = "Read-only scheduler placement projection", body = mold_core::GenerationPlacementPreview)
    )
)]
async fn generate_placement_preview(
    State(state): State<AppState>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Json(preview): Json<mold_core::GenerationPlacementPreviewRequest>,
) -> Result<Json<mold_core::GenerationPlacementPreview>, ApiError> {
    Ok(Json(
        placement_preview_for_request_authenticated(
            &state,
            preview.request,
            preview.copies,
            authenticated.as_ref().map(|Extension(auth)| auth),
        )
        .await?,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) async fn placement_preview_for_request(
    state: &AppState,
    request: GenerateRequest,
    copies: u32,
) -> mold_core::GenerationPlacementPreview {
    match placement_preview_for_request_authenticated(state, request, copies, None).await {
        Ok(preview) => preview,
        Err(error) => {
            let plan = state.scheduled_work.latest_plan();
            mold_core::GenerationPlacementPreview {
                version: 1,
                authoritative: true,
                state_version: plan.as_ref().map_or(0, |plan| plan.state_version),
                plan_version: plan.as_ref().map_or(0, |plan| plan.plan_version),
                outcome: "infeasible".to_string(),
                reason: Some(error.error),
                candidate: None,
                stage_candidates: Vec::new(),
                pending_downloads: Vec::new(),
                missing_components: Vec::new(),
            }
        }
    }
}

async fn placement_preview_for_request_authenticated(
    state: &AppState,
    mut request: GenerateRequest,
    copies: u32,
    authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
) -> Result<mold_core::GenerationPlacementPreview, ApiError> {
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let _ = authenticated;
    mold_core::minimax_h3::canonicalize_request_model(&mut request);
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if mold_core::minimax_h3::task_for_model(&request.model).is_some() {
        request.normalise_output_format(Some(mold_core::minimax_h3::FAMILY));
        crate::h3_private_bridge::substitute_redacted_preview_endpoints(&mut request);
        crate::h3_private_bridge::pin_private_preview_seed(&mut request)?;
    }
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let h3_private_ingress_grant = crate::h3_private_bridge::classify_h3_private_ingress(
        &request,
        authenticated,
        state.instance_id.as_str(),
    )?;
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let private_h3_ingress = h3_private_ingress_grant.is_some();
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let private_h3_ingress = false;
    let plan = state.scheduled_work.latest_plan();
    let unavailable = |outcome: &str, reason: String| mold_core::GenerationPlacementPreview {
        version: 1,
        authoritative: false,
        state_version: plan.as_ref().map_or(0, |plan| plan.state_version),
        plan_version: plan.as_ref().map_or(0, |plan| plan.plan_version),
        outcome: outcome.to_string(),
        reason: Some(reason),
        candidate: None,
        stage_candidates: Vec::new(),
        pending_downloads: Vec::new(),
        missing_components: Vec::new(),
    };
    if !(1..=64).contains(&copies) {
        let mut response = unavailable("infeasible", "copies must be between 1 and 64".to_string());
        response.authoritative = true;
        return Ok(response);
    }
    let resolved_family = if private_h3_ingress {
        Some(mold_core::minimax_h3::FAMILY.to_string())
    } else {
        match require_server_model_activation(state, &request.model).await {
            Ok(family) => family,
            Err(error) => {
                let mut response = unavailable("infeasible", error.error);
                response.authoritative = true;
                return Ok(response);
            }
        }
    };
    // The source-image contract (#772) is part of admission: a preview that
    // says `planned` for a request generation would 422 breaks the
    // authoritative-preview contract, so the same gate answers here as an
    // authoritative infeasible.
    if let Err(error) =
        enforce_source_image_capability(state, &request, resolved_family.as_deref()).await
    {
        let mut response = unavailable("infeasible", error.error);
        response.authoritative = true;
        return Ok(response);
    }
    // Face identity is part of admission for exactly the same reason, and the
    // preview reaches dependency preparation BEFORE the shared request
    // validator runs. Without this, an unqualified checkpoint carrying an
    // `id_image` — or a build without the `pulid` feature, or a LoRA beside
    // the photograph — would have its preview plan the PuLID bundle and
    // answer `planned` for a request generation then refuses.
    if let Err(error) = mold_core::identity::validate_identity_conditioning_with_family(
        &request,
        resolved_family.as_deref(),
    ) {
        let mut response = unavailable("infeasible", error);
        response.authoritative = true;
        return Ok(response);
    }
    // Admission materializes the family's tuned default negative (wan) before
    // the scheduler prices the job, and `request_sensitive_activation_memory`
    // doubles its CFG activation factor for `guidance > 1` only when a
    // negative is present. An authoritative preview priced without the same
    // materialization would promise a 1x plan admission re-prices at 2x —
    // exactly the divergence the placement-preview authority contract forbids.
    // Same seam, same semantics: absence fills in, the explicit `""` opt-out
    // and typed values pass through untouched.
    materialize_default_negative_prompt(&mut request, resolved_family.as_deref());
    if let Some(task) = mold_core::minimax_h3::task_for_model(&request.model) {
        match (task, request.references.as_deref()) {
            (mold_core::minimax_h3::Task::Ref2va, Some(references)) => {
                if let Err(error) =
                    mold_core::minimax_h3::validate_reference_descriptors(references)
                {
                    let mut response = unavailable("infeasible", error.to_string());
                    response.authoritative = true;
                    return Ok(response);
                }
            }
            (mold_core::minimax_h3::Task::Ref2va, None) => {
                let mut response = unavailable(
                    "infeasible",
                    "MiniMax H3 Ref2VA placement preview requires ordered reference descriptors"
                        .to_string(),
                );
                response.authoritative = true;
                return Ok(response);
            }
            (mold_core::minimax_h3::Task::Fl2va, Some(_)) => {
                let mut response = unavailable(
                    "infeasible",
                    "MiniMax H3 FL2VA does not accept Ref2VA ordered references".to_string(),
                );
                response.authoritative = true;
                return Ok(response);
            }
            (mold_core::minimax_h3::Task::Fl2va, None) => {}
        }
    }
    let planned_control = match plan_builtin_ltx2_control(state, &mut request).await {
        Ok(control) => control,
        Err(error) => {
            let mut response = unavailable("infeasible", error.error);
            response.authoritative = true;
            return Ok(response);
        }
    };
    let planned_camera_controls = match plan_builtin_ltx2_camera_controls(state, &request).await {
        Ok(controls) => controls,
        Err(error) => {
            let mut response = unavailable("infeasible", error.error);
            response.authoritative = true;
            return Ok(response);
        }
    };
    if planned_control.is_some() {
        if let Err(error) = mold_core::validate_generate_request_with_family(&request, Some("ltx2"))
        {
            let mut response = unavailable("infeasible", error);
            response.authoritative = true;
            return Ok(response);
        }
    }
    let has_post_upscale = request
        .upscale_model
        .as_deref()
        .is_some_and(|model| !model.trim().is_empty());
    let has_local_expansion = request.expand == Some(true)
        && state
            .config
            .read()
            .await
            .expand
            .clone()
            .with_env_overrides()
            .is_local();
    if has_local_expansion || has_post_upscale {
        return Ok(unavailable(
            "unsupported",
            "exact utility CPU/GPU placement plans are not available on this server".to_string(),
        ));
    }
    if !state.scheduled_work.v2_authoritative()
        || !state.scheduled_work.placement_preview_available()
    {
        return Ok(unavailable(
            "unsupported",
            "authoritative scheduler placement preview is unavailable".to_string(),
        ));
    }
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    let dependency_context = crate::variant_dependencies::DependencyPreparationContext {
        h3_private_ingress_grant: if h3_private_ingress_grant.is_some() {
            crate::h3_private_bridge::classify_h3_private_ingress(
                &request,
                authenticated,
                state.instance_id.as_str(),
            )?
        } else {
            None
        },
        // Placement preview is a read-only probe that must stay media-free, so
        // it never carries staged references. A Ref2VA preview therefore has
        // no prepared shapes to derive and stays non-authoritative, which is
        // the documented behaviour for a plan this endpoint cannot model.
        h3_resolved_references: None,
        // Same boundary: only the scheduler carries a parent's frozen
        // identity, from the owning job. This probe has no job and must not
        // invent one. Default the remaining context so additive optional
        // preparation inputs cannot leave this H3-only literal unbuildable.
        ..Default::default()
    };
    // Ref2VA plans depend on the staged reference media, which this probe
    // deliberately never carries. Preparation would therefore fail for the
    // absence of media rather than for anything about the device, and
    // reporting that as an authoritative `infeasible` would tell the client a
    // capacity answer this endpoint cannot give. Decline before that runs.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    if mold_core::minimax_h3::capability_contract_for_model(&request.model)
        .is_some_and(|contract| contract.task == mold_core::minimax_h3::Task::Ref2va)
    {
        return Ok(unavailable(
            "unsupported",
            "MiniMax H3 Ref2VA placement preview is not modeled without staged reference media"
                .to_string(),
        ));
    }
    #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
    let dependency_context = crate::variant_dependencies::DependencyPreparationContext::default();
    let prepared = match crate::variant_dependencies::prepare_execution_inputs_existing_only(
        state,
        &request,
        dependency_context,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => {
            let mut response = unavailable("infeasible", error);
            response.authoritative = true;
            response.missing_components = missing_model_components(state, &request.model).await;
            return Ok(response);
        }
    };
    if prepared.capacity_park.is_some() {
        return Ok(unavailable(
            "unsupported",
            "MiniMax H3 placement is waiting for active GPU work to release memory; the request may be queued through the compatible fallback"
                .to_string(),
        ));
    }
    match state
        .scheduled_work
        .preview_placement(request, copies, prepared)
        .await
    {
        Ok(mut response) => {
            if let Some((adapter, path)) = planned_control {
                if !control_artifact_is_complete(adapter, &path) {
                    response
                        .pending_downloads
                        .push(mold_core::PendingModelDownload {
                            kind: "ic-lora-control".to_string(),
                            // The adapter, not one of its files: a multi-file
                            // adapter with only its companion outstanding
                            // would otherwise name the weights already on
                            // disk and quote their size.
                            name: adapter.download_model.to_string(),
                            repo: adapter.hf_repo.to_string(),
                            bytes: control_pending_download_bytes(adapter, &path),
                            install_model: Some(adapter.download_model.to_string()),
                            licenses: Vec::new(),
                        });
                }
                let artifact_fingerprint = format!(":ic-lora:{}", adapter.sha256);
                if let Some(candidate) = response.candidate.as_mut() {
                    candidate
                        .execution_fingerprint
                        .push_str(&artifact_fingerprint);
                    if let Some(equivalence) = candidate.execution_equivalence_fingerprint.as_mut()
                    {
                        equivalence.push_str(&artifact_fingerprint);
                    }
                }
                for stage in &mut response.stage_candidates {
                    stage
                        .candidate
                        .execution_fingerprint
                        .push_str(&artifact_fingerprint);
                    if let Some(equivalence) =
                        stage.candidate.execution_equivalence_fingerprint.as_mut()
                    {
                        equivalence.push_str(&artifact_fingerprint);
                    }
                }
            }
            for (preset, path) in planned_camera_controls {
                if !camera_control_artifact_is_complete(preset, &path) {
                    response
                        .pending_downloads
                        .push(mold_core::PendingModelDownload {
                            kind: "camera-control".to_string(),
                            name: preset.hf_filename.to_string(),
                            repo: preset.hf_repo.to_string(),
                            bytes: preset.size_bytes,
                            install_model: None,
                            licenses: Vec::new(),
                        });
                }
                let artifact_fingerprint = format!(":camera-control:{}", preset.sha256);
                if let Some(candidate) = response.candidate.as_mut() {
                    candidate
                        .execution_fingerprint
                        .push_str(&artifact_fingerprint);
                    if let Some(equivalence) = candidate.execution_equivalence_fingerprint.as_mut()
                    {
                        equivalence.push_str(&artifact_fingerprint);
                    }
                }
                for stage in &mut response.stage_candidates {
                    stage
                        .candidate
                        .execution_fingerprint
                        .push_str(&artifact_fingerprint);
                    if let Some(equivalence) =
                        stage.candidate.execution_equivalence_fingerprint.as_mut()
                    {
                        equivalence.push_str(&artifact_fingerprint);
                    }
                }
            }
            Ok(response)
        }
        Err(error) => Ok(unavailable("temporarily_unavailable", error)),
    }
}

async fn missing_model_components(
    state: &AppState,
    model: &str,
) -> Vec<mold_core::ModelComponentStatus> {
    match model_manager::model_component_status_existing_only(state, model).await {
        Ok(status) => status
            .components
            .into_iter()
            .filter(|component| !component.present)
            .collect(),
        Err(error) => {
            tracing::warn!(
                model,
                error = %error.error,
                "strictly local component inspection failed during placement preview"
            );
            if mold_catalog::resolve::looks_like_catalog_id(model) {
                vec![mold_core::ModelComponentStatus {
                    kind: "transformer".to_string(),
                    name: "primary checkpoint".to_string(),
                    present: false,
                    path: None,
                    repair_model: Some(model.to_string()),
                    options: Vec::new(),
                }]
            } else {
                Vec::new()
            }
        }
    }
}

async fn model_components(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<mold_core::ModelComponentsResponse>, ApiError> {
    Ok(Json(
        model_manager::model_component_status(&state, &model).await?,
    ))
}

// ── /api/models/load ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct LoadModelBody {
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    /// Target GPU ordinal (multi-GPU only). If omitted, the server uses its
    /// default placement strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
    /// Third-party licenses the user has accepted, each carrying the exact
    /// terms they were shown. Honoured by `POST /api/models/pull`; ignored by
    /// `/api/models/load`, which moves no bytes over the network. Additive and
    /// empty by default.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accept_licenses: Vec<mold_core::LicenseAcceptance>,
}

#[utoipa::path(
    post,
    path = "/api/models/load",
    tag = "models",
    request_body = LoadModelBody,
    responses(
        (status = 200, description = "Model loaded successfully"),
        (status = 404, description = "Model not downloaded"),
        (status = 400, description = "Unknown model"),
        (status = 500, description = "Failed to load model"),
    )
)]
async fn load_model(
    State(state): State<AppState>,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_server_model_activation(&state, &body.model).await?;
    ensure_schedulable_device(&state)?;
    if let Err(e) = model_manager::install_catalog_model(&state, &body.model).await {
        return Err(model_manager::install_error_to_api_error(&e));
    }
    let _ = model_manager::check_model_available(&state, &body.model).await?;

    // Multi-GPU path: route through the pool.
    if state.gpu_pool.worker_count() > 0 {
        if let Some(ordinal) = body.gpu {
            if !state
                .gpu_pool
                .schedulable_workers()
                .iter()
                .any(|worker| worker.gpu.ordinal == ordinal)
            {
                return Err(ApiError::not_found(format!(
                    "no GPU worker with ordinal {ordinal}"
                )));
            }
        }
        let config_snapshot = state.config.read().await.clone();
        let model_name = body.model.clone();
        let id = format!("admin-model-load-{}", uuid::Uuid::new_v4());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let work = crate::gpu_pool::OwnerWork::AdminModelLoad(Box::new(
            crate::gpu_pool::AdminModelLoadJob {
                id: id.clone(),
                model: model_name.clone(),
                config: config_snapshot,
                result_tx,
            },
        ));
        state
            .scheduled_work
            .submit(
                crate::scheduler::ScheduledOwnerWork::new(
                    id,
                    model_name,
                    crate::queue::estimate_model_vram(&body.model),
                    work,
                )
                .with_hard_ordinal(body.gpu)
                .with_priority(mold_scheduler::PriorityClass::Admin),
            )
            .await
            .map_err(ApiError::generation_unavailable)?;
        result_rx
            .await
            .map_err(|_| ApiError::internal("model-load owner worker dropped its result"))?
            .map_err(|e| ApiError::internal(format!("model load error: {e}")))?;

        tracing::info!(
            model = %body.model,
            gpu = ?body.gpu,
            "model loaded via API"
        );
        return Ok(StatusCode::OK);
    }

    // Legacy single-GPU path (no workers discovered). No resolution context
    // here — admin load uses size-only peak via the `None` hint.
    model_manager::ensure_model_ready(&state, &body.model, None, None, false).await?;
    tracing::info!(model = %body.model, gpu = ?body.gpu, "model loaded via API (legacy)");
    Ok(StatusCode::OK)
}

// ── /api/models/pull ──────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/models/pull",
    tag = "models",
    request_body = LoadModelBody,
    responses(
        (status = 200, description = "Model pulled (SSE stream or plain text)"),
        (status = 400, description = "Unknown model"),
        (status = 500, description = "Download failed"),
    )
)]
async fn pull_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoadModelBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_server_model_acquisition(&state, &body.model).await?;
    // Before ANY branch below — the catalog repair path enqueues downloads
    // too, and a refusal must happen before bytes move on either route.
    apply_download_license_acceptances(&state, &body.model, &body.accept_licenses).await?;
    let wants_sse = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));

    if body.model.starts_with("cv:") || body.model.starts_with("hf:") {
        if let Err(e) = model_manager::install_catalog_model(&state, &body.model).await {
            return Err(model_manager::install_error_to_api_error(&e));
        }
        match model_manager::check_model_available(&state, &body.model).await {
            Ok(_) => {
                return Ok(PullResponse::Text(format!(
                    "model '{}' is already present",
                    body.model
                )));
            }
            Err(error) if error.code == mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED => {
                return Err(error);
            }
            Err(_) => {}
        }
        let companion_names = {
            let intents = state.catalog_intents.read().await;
            intents
                .get(&body.model)
                .map(|intent| {
                    intent
                        .companions
                        .iter()
                        .map(|companion| companion.name.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let models_dir = state.config.read().await.resolved_models_dir();
        let companion_jobs = crate::catalog_api::enqueue_missing_companions(
            &companion_names,
            &models_dir,
            &state.downloads,
            Some(&body.model),
            None,
        )
        .await;
        let primary_job =
            crate::catalog_api::enqueue_catalog_primary_repair(&state, &body.model).await?;
        if !companion_jobs.is_empty() || primary_job.is_some() {
            let primary_count = usize::from(primary_job.is_some());
            return Ok(PullResponse::Text(format!(
                "queued repair for model '{}' ({} primary job(s), {} companion job(s))",
                body.model,
                primary_count,
                companion_jobs.len()
            )));
        }
        model_manager::check_model_available(&state, &body.model).await?;
        return Ok(PullResponse::Text(format!(
            "model '{}' is already present",
            body.model
        )));
    }

    // Enqueue via the queue. Treat idempotent AlreadyPresent as success.
    let (job_id, _position) = match state.downloads.enqueue(body.model.clone()).await {
        Ok((id, pos, _)) => (id, pos),
        Err(crate::downloads::EnqueueError::ModelActivation(error)) => {
            return Err(ApiError::model_activation(error));
        }
        Err(crate::downloads::EnqueueError::UnknownModel(_)) => {
            return Err(ApiError::unknown_model(format!(
                "unknown model '{}'. Run 'mold list' to see available models.",
                body.model
            )));
        }
        Err(crate::downloads::EnqueueError::LockPoisoned) => {
            return Err(ApiError::internal("download queue state is corrupt"));
        }
    };

    if !wants_sse {
        // Await terminal event for this job, return plain text.
        let mut rx = state.downloads.subscribe();
        loop {
            match rx.recv().await {
                Ok(mold_core::types::DownloadEvent::JobDone { id, model }) if id == job_id => {
                    return Ok(PullResponse::Text(format!(
                        "model '{model}' pulled successfully"
                    )));
                }
                Ok(mold_core::types::DownloadEvent::JobFailed { id, error }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "failed to pull model '{}': {error}",
                        body.model
                    )));
                }
                Ok(mold_core::types::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    return Err(ApiError::internal(format!(
                        "pull of '{}' was cancelled",
                        body.model
                    )));
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err(ApiError::internal("download queue channel closed"));
                }
            }
        }
    }

    // SSE: re-emit queue events shaped like the legacy SseProgressEvent::DownloadProgress
    // so the TUI's existing consumer continues to work unchanged.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseMessage>();
    let mut events = state.downloads.subscribe();
    let model_for_cb = body.model.clone();
    tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(mold_core::types::DownloadEvent::Started {
                    id,
                    files_total,
                    bytes_total,
                }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::DownloadProgress {
                        filename: String::new(),
                        file_index: 0,
                        total_files: files_total,
                        bytes_downloaded: 0,
                        bytes_total,
                        batch_bytes_downloaded: 0,
                        batch_bytes_total: bytes_total,
                        batch_elapsed_ms: 0,
                    }));
                }
                Ok(mold_core::types::DownloadEvent::Progress {
                    id,
                    files_done,
                    bytes_done,
                    current_file,
                }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::DownloadProgress {
                        filename: current_file.unwrap_or_default(),
                        file_index: files_done,
                        total_files: 0,
                        bytes_downloaded: bytes_done,
                        bytes_total: 0,
                        batch_bytes_downloaded: bytes_done,
                        batch_bytes_total: 0,
                        batch_elapsed_ms: 0,
                    }));
                }
                Ok(mold_core::types::DownloadEvent::JobDone { id, .. }) if id == job_id => {
                    let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
                        model: model_for_cb.clone(),
                    }));
                    break;
                }
                Ok(mold_core::types::DownloadEvent::JobFailed { id, error }) if id == job_id => {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(error)));
                    break;
                }
                Ok(mold_core::types::DownloadEvent::JobCancelled { id }) if id == job_id => {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent::failed("pull cancelled")));
                    break;
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|msg| Ok::<_, Infallible>(sse_message_to_event(msg)));

    Ok(PullResponse::Sse(
        Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15))
                    .text("ping"),
            )
            .into_response(),
    ))
}

/// Response type that can be either SSE stream or plain text.
enum PullResponse {
    Sse(axum::response::Response),
    Text(String),
}

impl IntoResponse for PullResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            PullResponse::Sse(resp) => resp,
            PullResponse::Text(text) => text.into_response(),
        }
    }
}

// ── /api/models/unload ────────────────────────────────────────────────────────

/// Optional request body for unload — clients may specify a model or GPU target.
/// An empty body (or no body) unloads the active model on the legacy path.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
pub struct UnloadRequest {
    /// Specific model to unload. If omitted, the active model is unloaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Target GPU ordinal (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
}

#[utoipa::path(
    delete,
    path = "/api/models/unload",
    tag = "models",
    request_body(content = Option<UnloadRequest>, content_type = "application/json"),
    responses(
        (status = 200, description = "Model unloaded or no model was loaded", body = String),
    )
)]
async fn unload_model(
    State(state): State<AppState>,
    body: Option<Json<UnloadRequest>>,
) -> Result<impl IntoResponse, ApiError> {
    let req = body.map(|b| b.0).unwrap_or_default();
    tracing::debug!(model = ?req.model, gpu = ?req.gpu, "unload request");
    // Multi-GPU path: target specific GPU or model across the pool.
    if state.gpu_pool.worker_count() > 0 {
        // Select the workers to unload from.
        let targets: Vec<_> = match (req.gpu, req.model.as_deref()) {
            (Some(ordinal), _) => state
                .gpu_pool
                .workers
                .iter()
                .filter(|w| w.gpu.ordinal == ordinal)
                .collect(),
            (None, Some(model)) => state
                .gpu_pool
                .workers
                .iter()
                .filter(|w| {
                    w.resident_model
                        .read()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .as_deref()
                        .map(crate::gpu_pool::resident_model_display_name)
                        == Some(model)
                })
                .collect(),
            (None, None) => state.gpu_pool.worker_snapshot(),
        };

        if targets.is_empty() {
            return Ok((StatusCode::OK, "no model loaded".to_string()));
        }

        let mut unloaded_pairs: Vec<(usize, String)> = Vec::new();
        for worker in targets {
            let id = format!("admin-model-unload-{}", uuid::Uuid::new_v4());
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();
            let work = crate::gpu_pool::OwnerWork::AdminModelUnload(Box::new(
                crate::gpu_pool::AdminModelUnloadJob {
                    id: id.clone(),
                    model: req.model.clone(),
                    evict_cached: false,
                    result_tx,
                },
            ));
            state
                .scheduled_work
                .submit(
                    crate::scheduler::ScheduledOwnerWork::new(
                        id,
                        req.model
                            .clone()
                            .unwrap_or_else(|| "admin-unload".to_string()),
                        0,
                        work,
                    )
                    .with_hard_ordinal(Some(worker.gpu.ordinal))
                    .with_priority(mold_scheduler::PriorityClass::Admin),
                )
                .await
                .map_err(ApiError::generation_unavailable)?;
            let result = result_rx
                .await
                .map_err(|_| ApiError::internal("model-unload owner worker dropped its result"))?
                .map_err(|error| ApiError::internal(format!("unload failed: {error}")))?;
            if let Some(name) = result {
                unloaded_pairs.push((worker.gpu.ordinal, name));
            }
        }

        let msg = if unloaded_pairs.is_empty() {
            "no model loaded".to_string()
        } else {
            let joined: Vec<String> = unloaded_pairs
                .iter()
                .map(|(o, m)| format!("gpu{o}:{m}"))
                .collect();
            format!("unloaded {}", joined.join(", "))
        };
        release_host_memory_after_unload(&state);
        return Ok((StatusCode::OK, msg));
    }

    // Legacy single-GPU path.
    clear_global_upscaler_cache(&state).await;
    let message = model_manager::unload_model(&state).await;
    release_host_memory_after_unload(&state);
    Ok((StatusCode::OK, message))
}

// ── DELETE /api/models/:model ─────────────────────────────────────────────────

/// True when an engine for `canonical` is currently GPU-resident (or mid-
/// generation) anywhere on this server — the single-GPU cache, any pool
/// worker's cache, or an active-generation snapshot on either path.
async fn model_is_gpu_resident(state: &AppState, canonical: &str) -> bool {
    {
        let cache = state.model_cache.lock().await;
        if cache.active_model() == Some(canonical) {
            return true;
        }
    }
    if state
        .active_generation
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|g| g.model == canonical)
    {
        return true;
    }
    for worker in &state.gpu_pool.workers {
        if let Ok(active) = worker.active_generation.read() {
            if active.as_ref().is_some_and(|g| g.model == canonical) {
                return true;
            }
        }
        if let Ok(resident) = worker.resident_model.read() {
            if resident
                .as_deref()
                .map(crate::gpu_pool::resident_model_display_name)
                == Some(canonical)
            {
                return true;
            }
        }
    }
    false
}

/// Remove a downloaded model's files — the HTTP counterpart of `mold rm`.
///
/// Ref-counts every file path across all installed models and deletes only
/// paths exclusively owned by this model; components still referenced by
/// another downloaded model (shared T5/CLIP/Qwen encoders, VAEs) are kept
/// and reported in `kept` with the surviving referents. hf-hub cache blobs
/// hardlinked to the deleted clean paths are removed too, so `freed_bytes`
/// reflects real disk savings.
#[utoipa::path(
    delete,
    path = "/api/models/{model}",
    tag = "models",
    params(("model" = String, Path, description = "Model name (e.g. flux-schnell:q8)")),
    responses(
        (status = 200, description = "Model removed", body = mold_core::ModelRemovalResponse),
        (status = 404, description = "Model not installed"),
        (status = 409, description = "Model is currently loaded — unload it first"),
    )
)]
async fn delete_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<mold_core::ModelRemovalResponse>, ApiError> {
    let canonical = mold_core::manifest::resolve_model_name(&model);

    // Refuse while the model is GPU-resident — there is no safe way to pull
    // files out from under a loaded engine. Check the raw input too: engines
    // register under their own model_name, which for non-manifest models may
    // not round-trip through resolve_model_name. (Best-effort check: a load
    // that races past it just keeps working off its already-open mmaps.)
    if model_is_gpu_resident(&state, &canonical).await
        || (model != canonical && model_is_gpu_resident(&state, &model).await)
    {
        return Err(ApiError::with_code(
            format!(
                "model '{canonical}' is currently loaded; unload it first (DELETE /api/models/unload)"
            ),
            "MODEL_LOADED",
            StatusCode::CONFLICT,
        ));
    }

    // Hold the config write lock across plan + delete so a concurrent pull
    // or placement write can't interleave with the removal.
    let mut config = state.config.write().await;
    let in_config = config.models.contains_key(&canonical);
    if !in_config && !config.manifest_model_is_downloaded(&canonical) {
        return Err(ApiError::with_code(
            format!("model '{canonical}' is not installed"),
            "UNKNOWN_MODEL",
            StatusCode::NOT_FOUND,
        ));
    }

    tracing::info!(model = %canonical, "model removal requested");
    let plan = mold_core::removal::plan_removal(&config, &canonical);
    let outcome = mold_core::removal::execute_removal(&config, &plan);
    for warning in &outcome.warnings {
        tracing::warn!(model = %canonical, "model removal: {warning}");
    }

    mold_core::download::remove_pulling_marker(&canonical);
    if in_config {
        config.remove_model(&canonical);
        if let Err(e) = config.save() {
            tracing::warn!("failed to persist model removal to config.toml: {e}");
        }
    }
    drop(config);

    // Evict any parked (non-GPU-resident) engine so a later request can't
    // reactivate an engine whose files are gone.
    {
        let mut cache = state.model_cache.lock().await;
        let _ = cache.remove(&canonical);
    }
    for worker in &state.gpu_pool.workers {
        let id = format!("admin-model-evict-{}", uuid::Uuid::new_v4());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let work = crate::gpu_pool::OwnerWork::AdminModelUnload(Box::new(
            crate::gpu_pool::AdminModelUnloadJob {
                id: id.clone(),
                model: Some(canonical.clone()),
                evict_cached: true,
                result_tx,
            },
        ));
        state
            .scheduled_work
            .submit(
                crate::scheduler::ScheduledOwnerWork::new(id, canonical.clone(), 0, work)
                    .with_hard_ordinal(Some(worker.gpu.ordinal))
                    .with_priority(mold_scheduler::PriorityClass::Admin),
            )
            .await
            .map_err(ApiError::generation_unavailable)?;
        result_rx
            .await
            .map_err(|_| ApiError::internal("model-eviction owner worker dropped its result"))?
            .map_err(|error| ApiError::internal(format!("model eviction failed: {error}")))?;
    }

    let kept = plan
        .shared_files
        .iter()
        .map(|(path, used_by)| mold_core::KeptComponent {
            component: path.clone(),
            used_by: used_by.clone(),
        })
        .collect();

    Ok(Json(mold_core::ModelRemovalResponse {
        removed: outcome.removed,
        kept,
        freed_bytes: outcome.freed_bytes,
    }))
}

// ── /api/status ───────────────────────────────────────────────────────────────

/// Disk usage for the filesystem holding `dir`: among mounts that are a
/// prefix of the path, the longest one wins (`/data/models` must resolve to
/// the `/data` mount, not `/`). `None` when no mount matches.
fn disk_usage_for_path(
    disks: &[(std::path::PathBuf, u64, u64)],
    dir: &std::path::Path,
) -> Option<DiskUsage> {
    disks
        .iter()
        .filter(|(mount, _, _)| dir.starts_with(mount))
        .max_by_key(|(mount, _, _)| mount.as_os_str().len())
        .map(|&(_, total_bytes, free_bytes)| DiskUsage {
            total_bytes,
            free_bytes,
        })
}

/// Resolve symlinks in the models dir before mount matching — a symlinked
/// models dir (`ln -s /Volumes/Big/models ~/.mold/models`) must report the
/// target's volume, not the symlink's. Falls back to the literal path when
/// canonicalization fails (missing dir, permissions).
fn canonical_models_dir(dir: &std::path::Path) -> std::path::PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// Snapshot disk stats for the filesystem backing the models dir. Blocking:
/// canonicalize hits the filesystem and the disk refresh calls statvfs(2) on
/// every mount (which can stall on wedged FUSE mounts) — callers on the async
/// path must wrap this in `spawn_blocking`.
fn models_disk_usage(dir: &std::path::Path) -> Option<DiskUsage> {
    let dir = canonical_models_dir(dir);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let entries: Vec<(std::path::PathBuf, u64, u64)> = disks
        .list()
        .iter()
        .map(|d| {
            (
                d.mount_point().to_path_buf(),
                d.total_space(),
                d.available_space(),
            )
        })
        .collect();
    disk_usage_for_path(&entries, &dir)
}

#[utoipa::path(
    get,
    path = "/api/status",
    tag = "server",
    responses(
        (status = 200, description = "Server status", body = ServerStatus),
    )
)]
async fn server_status(State(state): State<AppState>) -> Result<Json<ServerStatus>, ApiError> {
    // Disk stats are blocking syscalls (canonicalize + statvfs per mount, and
    // statvfs can hang outright on a wedged FUSE mount) — never run or await
    // them on the request path. Serve the cached snapshot; when it has gone
    // stale, the single winning request kicks a background refresh. The first
    // poll after boot reports no disk stats — pollers pick them up next round.
    let (models_disk, needs_disk_refresh) = state.models_disk_cache.read();
    let (durable_media_applicable, models_dir) = {
        let config = state.config.read().await;
        (
            DurableAdmissionReadiness::resolve(&state, &config).applicable(),
            config.resolved_models_dir(),
        )
    };
    if needs_disk_refresh {
        let cache = state.models_disk_cache.clone();
        tokio::task::spawn_blocking(move || cache.store(models_disk_usage(&models_dir)));
    }

    // `queue_capacity` is only the hydrated runtime window. Route selection
    // needs the total waiting load, including SQLite-owned work that has not
    // reached that window yet. Probe only the bounded live queued IDs for
    // overlap so hydrated durable work contributes exactly once.
    let live_waiting_ids = state
        .job_registry
        .snapshot()
        .entries
        .into_iter()
        .filter(|entry| entry.state == crate::job_registry::JobLifecycle::Queued)
        .map(|entry| entry.id)
        .collect::<Vec<_>>();
    let journal = state.queue_journal.clone();
    let queue_depth = spawn_queue_read(move || journal.total_waiting(&live_waiting_ids)).await?;

    // One registry snapshot backs both the additive device API and legacy
    // status projections. It reads only the 1 Hz telemetry cache and worker
    // atomics/locks; status never shells out or queries CUDA.
    let devices = current_device_state(&state);
    let gpu_statuses =
        crate::device_registry::DeviceRegistry::legacy_gpu_status_from_snapshot(&devices);
    let has_gpus = !gpu_statuses.is_empty();
    let has_device_inventory = !devices.devices.is_empty();

    // Collect loaded models from GPU workers.
    let gpu_models_loaded: Vec<String> = gpu_statuses
        .iter()
        .filter_map(|g| g.loaded_model.clone())
        .collect();
    let gpu_busy = gpu_statuses
        .iter()
        .any(|g| g.state == GpuWorkerState::Generating);

    // Pull current_generation from the first busy worker (multi-GPU) or
    // from the legacy snapshot.
    let multi_gpu_current_gen = if has_gpus {
        state.gpu_pool.workers.iter().find_map(|w| {
            let gen = w.active_generation.read().ok()?;
            gen.as_ref().map(|g| ActiveGenerationStatus {
                model: g.model.clone(),
                prompt_sha256: g.prompt_sha256.clone(),
                started_at_unix_ms: g.started_at_unix_ms,
                elapsed_ms: g.started_at.elapsed().as_millis() as u64,
            })
        })
    } else {
        None
    };

    // Fall back to legacy single-GPU snapshot for backwards compat.
    let (models_loaded, busy, current_generation) = if has_gpus {
        (gpu_models_loaded, gpu_busy, multi_gpu_current_gen)
    } else {
        let snapshot = state.model_cache.lock().await.snapshot();
        let models = match (snapshot.model_name, snapshot.is_loaded) {
            (Some(model_name), true) => vec![model_name],
            _ => vec![],
        };
        let gen = state
            .active_generation
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|active| ActiveGenerationStatus {
                model: active.model.clone(),
                prompt_sha256: active.prompt_sha256.clone(),
                started_at_unix_ms: active.started_at_unix_ms,
                elapsed_ms: active.started_at.elapsed().as_millis() as u64,
            });
        let is_busy = gen.is_some();
        (models, is_busy, gen)
    };

    Ok(Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: if mold_core::build_info::GIT_SHA == "unknown" {
            None
        } else {
            Some(mold_core::build_info::GIT_SHA.to_string())
        },
        build_date: if mold_core::build_info::BUILD_DATE == "unknown" {
            None
        } else {
            Some(mold_core::build_info::BUILD_DATE.to_string())
        },
        models_loaded,
        busy,
        current_generation,
        gpu_info: crate::device_registry::DeviceRegistry::legacy_gpu_info(&devices),
        uptime_secs: state.start_time.elapsed().as_secs(),
        hostname: hostname::get().ok().and_then(|h| h.into_string().ok()),
        memory_status: crate::device_registry::DeviceRegistry::legacy_memory_status(&devices),
        gpus: if has_device_inventory {
            Some(gpu_statuses)
        } else {
            None
        },
        queue_depth: Some(queue_depth),
        queue_capacity: Some(state.queue_capacity),
        queue_paused: Some(state.queue_pause.is_paused()),
        instance_id: Some(state.instance_id.as_ref().clone()),
        models_disk,
        host_memory: state.scheduled_work.host_memory(),
        durable_media: state
            .queue_journal
            .durable_media_status(durable_media_applicable),
    }))
}

/// Stable read-only inventory of every runtime-visible device. Unsupported
/// telemetry stays null and the handler never performs device discovery.
#[utoipa::path(
    get,
    path = "/api/devices",
    tag = "server",
    responses(
        (status = 200, description = "Runtime-visible device inventory", body = DeviceState),
    )
)]
async fn list_devices(State(state): State<AppState>) -> Json<DeviceState> {
    Json(current_device_state(&state))
}

pub(crate) fn current_device_state(state: &AppState) -> DeviceState {
    let resources = state.resources.latest();
    let mut snapshot =
        state
            .device_registry
            .snapshot(&state.gpu_pool, resources.as_ref(), &state.job_registry);
    let authoritative_v2 = state.scheduled_work.v2_authoritative();
    if !authoritative_v2 {
        // Legacy and observe modes keep their restart-time workers as the
        // runtime authority. A persisted V2 preference must never make an
        // actively dispatching legacy worker appear disabled.
        for device in &mut snapshot.devices {
            let live_worker = device
                .ordinal
                .and_then(|ordinal| state.gpu_pool.worker_by_ordinal(ordinal));
            if live_worker.is_some() {
                // Only administrative ownership rolls back outside
                // authoritative V2. Health, cooldowns, routing eligibility,
                // and their reason remain the registry's authority.
                device.admin_state = mold_core::DeviceAdminState::Enabled;
            }
        }
    }
    annotate_restart_required(state, &mut snapshot);
    if authoritative_v2 {
        if let Some(plan) = state.scheduled_work.latest_plan() {
            snapshot.plan_version = plan.plan_version;
            for device in &mut snapshot.devices {
                device.planned_work_ids = plan
                    .work_items
                    .iter()
                    .filter(|work| work.planned_device_id.as_deref() == Some(device.id.as_str()))
                    .map(|work| work.work_id.clone())
                    .collect();
            }
        }
    }
    snapshot
}

fn annotate_restart_required(state: &AppState, snapshot: &mut DeviceState) {
    if state.scheduled_work.v2_authoritative() {
        return;
    }
    let live_owners: std::collections::BTreeSet<String> = state
        .gpu_pool
        .worker_snapshot()
        .iter()
        .map(|worker| crate::scheduler::worker_device_id(worker))
        .collect();
    for device in &mut snapshot.devices {
        device.restart_required = device.desired_enabled
            && device.admin_state != DeviceAdminState::StartupExcluded
            && !live_owners.contains(&device.id)
            && !state.gpu_pool.workers.is_starting(&device.id);
    }
}

struct DeviceMutationOutcome {
    previous_desired: bool,
    device: mold_core::DeviceInfo,
    asynchronous: bool,
}

/// Complete one ordered device mutation independently of the request future.
/// Persistence can block, so it runs before the scheduler fence and on the
/// blocking pool. Once persistence succeeds, publication and the owner state
/// transition are synchronous under the fence; any owner join and projection
/// work happen after releasing it. Spawning this operation from the route also
/// prevents request cancellation from leaving SQLite ahead of live authority.
async fn apply_device_mutation(
    state: AppState,
    device_id: String,
    enabled: bool,
    mutate_runtime: bool,
) -> Result<DeviceMutationOutcome, ApiError> {
    let mut preference = state
        .device_registry
        .prepare_desired_enabled_mutation(&device_id, enabled)
        .await
        .map_err(|error| {
            ApiError::internal(format!("failed to persist device preference: {error:#}"))
        })?;
    let previous_desired = preference.previous();
    let preference_changed = preference.changed();
    let mut asynchronous = false;
    let mut reap_owner_epoch = None;

    if mutate_runtime {
        let started_epoch;
        {
            let _mutation_guard = state.scheduler_mutation_fence.lock().await;
            preference.publish();
            started_epoch = if enabled {
                if state.gpu_pool.workers.cancel_drain(&device_id) {
                    None
                } else if state
                    .gpu_pool
                    .worker_snapshot()
                    .iter()
                    .any(|worker| crate::scheduler::worker_device_id(worker) == device_id)
                {
                    // The old owner already committed to exit. Its exact
                    // Stopped reduction observes desired=true and creates the
                    // replacement.
                    asynchronous = true;
                    None
                } else {
                    let owner_epoch =
                        state.gpu_pool.workers.start(&device_id).map_err(|error| {
                            ApiError::with_code(
                                format!("device '{device_id}' remains unavailable: {error}"),
                                "NO_SCHEDULABLE_DEVICE",
                                StatusCode::SERVICE_UNAVAILABLE,
                            )
                        })?;
                    asynchronous = true;
                    Some(owner_epoch)
                }
            } else if let Some(worker) = state
                .gpu_pool
                .worker_snapshot()
                .into_iter()
                .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
            {
                let owner_epoch = worker.owner_epoch;
                let busy = state
                    .gpu_pool
                    .workers
                    .request_disable(&device_id)
                    .map_err(ApiError::internal)?;
                if busy {
                    asynchronous = true;
                } else {
                    reap_owner_epoch = Some(owner_epoch);
                }
                None
            } else {
                None
            };

            if let Some(owner_epoch) = started_epoch {
                match state
                    .gpu_pool
                    .workers
                    .announce_start(&device_id, owner_epoch)
                {
                    crate::gpu_pool::StartAnnouncement::Ready => {
                        state
                            .events
                            .publish(mold_core::ServerEvent::DeviceStateChanged {
                                device_id: device_id.clone(),
                                desired_enabled: true,
                                admin_state: DeviceAdminState::Enabled,
                            });
                    }
                    crate::gpu_pool::StartAnnouncement::Failed(error) => {
                        tracing::error!(
                            device_id,
                            owner_epoch,
                            %error,
                            "GPU owner failed during asynchronous lifecycle start"
                        );
                    }
                    crate::gpu_pool::StartAnnouncement::Pending => {}
                }
            }
        }

        if let Some(owner_epoch) = reap_owner_epoch {
            let pool = state.gpu_pool.clone();
            let id = device_id.clone();
            tokio::task::spawn_blocking(move || pool.workers.wait_and_reap(&id, owner_epoch))
                .await
                .map_err(|error| {
                    ApiError::internal(format!("failed to join GPU owner thread: {error}"))
                })?;
        }
    } else {
        let _mutation_guard = state.scheduler_mutation_fence.lock().await;
        preference.publish();
    }

    let resources = state.resources.latest();
    let mut snapshot =
        state
            .device_registry
            .snapshot(&state.gpu_pool, resources.as_ref(), &state.job_registry);
    if !mutate_runtime {
        annotate_restart_required(&state, &mut snapshot);
    }
    let device = snapshot
        .devices
        .into_iter()
        .find(|device| device.id == device_id)
        .ok_or_else(|| ApiError::internal("device disappeared during lifecycle mutation"))?;
    if !mutate_runtime {
        asynchronous = device.restart_required && preference_changed;
    }
    Ok(DeviceMutationOutcome {
        previous_desired,
        device,
        asynchronous,
    })
}

#[utoipa::path(
    patch,
    path = "/api/devices/{id}",
    tag = "server",
    params(("id" = String, Path, description = "Opaque stable device ID")),
    request_body = DeviceMutationRequest,
    responses(
        (status = 200, description = "Requested lifecycle state reached"),
        (status = 202, description = "Device is draining or starting"),
        (status = 404, description = "Unknown stable device ID"),
        (status = 409, description = "Runtime lifecycle unavailable in this dispatch mode"),
        (status = 503, description = "Fresh owner thread could not be started"),
    )
)]
async fn patch_device(
    State(state): State<AppState>,
    Path(device_id): Path<String>,
    connect: Option<axum::extract::ConnectInfo<std::net::SocketAddr>>,
    request_id: Option<Extension<crate::request_id::RequestId>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Json(request): Json<DeviceMutationRequest>,
) -> Result<axum::response::Response, ApiError> {
    let discovered = state
        .device_registry
        .discovered_device(&device_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown device '{device_id}'")))?;
    if request.enabled && !discovered.startup_allowed {
        return Err(ApiError::with_code(
            format!(
                "device '{device_id}' was excluded by startup selection and requires a restart"
            ),
            "DEVICE_STARTUP_EXCLUDED",
            StatusCode::CONFLICT,
        ));
    }

    // A non-authoritative runtime cannot touch live owners, but it must let an
    // operator recover a persistently-disabled GPU for the next restart.
    if !state.scheduled_work.v2_authoritative() && request.enabled {
        let operation_state = state.clone();
        let operation_device_id = device_id.clone();
        let outcome = tokio::spawn(async move {
            apply_device_mutation(operation_state, operation_device_id, true, false).await
        })
        .await
        .map_err(|error| ApiError::internal(format!("device mutation task failed: {error}")))??;
        let status = if outcome.asynchronous {
            StatusCode::ACCEPTED
        } else {
            StatusCode::OK
        };
        return Ok((status, Json(outcome.device)).into_response());
    }
    if !state.scheduled_work.v2_authoritative() {
        return Err(ApiError::with_code(
            "disabling a live GPU requires an authoritative scheduler V2 runtime",
            "DEVICE_LIFECYCLE_MODE_CONFLICT",
            StatusCode::CONFLICT,
        ));
    }

    let operation_state = state.clone();
    let operation_device_id = device_id.clone();
    let enabled = request.enabled;
    let outcome = tokio::spawn(async move {
        apply_device_mutation(operation_state, operation_device_id, enabled, true).await
    })
    .await
    .map_err(|error| ApiError::internal(format!("device mutation task failed: {error}")))??;
    tracing::info!(
        device_id,
        old_desired_enabled = outcome.previous_desired,
        desired_enabled = request.enabled,
        result = ?outcome.device.admin_state,
        request_id = request_id.as_ref().map(|id| id.0.0.as_str()),
        authenticated_key = authenticated
            .as_ref()
            .map(|identity| identity.0.identity.as_str()),
        remote_addr = ?connect.map(|address| address.0),
        "device lifecycle mutation"
    );

    let status = if outcome.asynchronous
        || matches!(
            outcome.device.admin_state,
            DeviceAdminState::Draining | DeviceAdminState::Starting
        ) {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(outcome.device)).into_response())
}

// ── Durable-admission readiness ───────────────────────────────────────────────

/// Which conjunct of the durable-admission question is unmet.
///
/// The resolved value must retain WHICH conjunct failed rather than collapsing
/// to a bool, because two consumers report a per-conjunct cause and would
/// otherwise be handed a lossy answer: the two HTTP refusal codes below, and
/// `QueueJournal::durable_media_status`'s `reasons`, which exist because "a
/// widened store directory and a never-installed admission service need
/// different repairs" (`queue_journal.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurableAdmissionUnready {
    /// Conjunct 1 — the queue journal claimed no owner.
    QueueJournalDisabled,
    /// Conjunct 2 — this server writes no gallery output.
    GalleryOutputDisabled,
    /// Conjunct 3 — Scheduler V2 is not authoritative (legacy/observe).
    SchedulerNotAuthoritative,
    /// Conjunct 4 — the admission service failed to install (corrupt or
    /// missing `generation-admission.key`); the process still serves.
    AdmissionServiceMissing,
}

impl DurableAdmissionUnready {
    /// One refusal code for every route and every conjunct.
    ///
    /// A client cannot act differently on "the journal is off" than on "the
    /// scheduler is not authoritative": both mean this host does not generate
    /// until an operator changes its configuration. The per-conjunct MESSAGE
    /// is what names the repair.
    pub(crate) const CODE: &'static str = "DURABLE_ADMISSION_UNAVAILABLE";

    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::GalleryOutputDisabled => "durable admission requires server gallery output",
            Self::QueueJournalDisabled => "durable admission requires the durable queue journal",
            Self::SchedulerNotAuthoritative => "durable admission requires Scheduler V2",
            Self::AdmissionServiceMissing => "durable admission is unavailable on this host",
        }
    }

    pub(crate) fn api_error(self) -> ApiError {
        ApiError::with_code(self.message(), Self::CODE, StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// The complete durable-admission answer for one request-time snapshot.
///
/// Before this existed, four sites took four DIFFERENT subsets of the same
/// four conjuncts — the advertised batch limit took 1∧2∧3,
/// `durable_media_is_applicable` took 2∧3, `direct_durable_admission` took
/// 2∧4, and only `admit_generation_batch` took all four (and only by
/// sequencing across a `config.read().await`). The gaps between those subsets
/// were four separate defects. One resolved value read by every consumer is
/// what stops a fifth subset appearing.
pub(crate) struct DurableAdmissionReadiness {
    journal_enabled: bool,
    output_enabled: bool,
    authoritative: bool,
    admission: Option<Arc<crate::queue_media_admission::DurableMediaAdmission>>,
    /// Already the conjunction of reconciliation, lifecycle and the admission
    /// service (`QueueJournal::durable_media_capabilities`); carried, never
    /// recomputed.
    media: Option<mold_core::DurableMediaCapabilities>,
}

impl DurableAdmissionReadiness {
    /// The only constructor, and it is TOTAL over all four conjuncts at ONE
    /// evaluation point.
    ///
    /// `config` is a required parameter and is never read inside. That is what
    /// keeps this from becoming a fifth wrong subset: conjunct 2 is
    /// `!is_output_disabled(config)`, so a caller without a config cannot call
    /// this at all. There is deliberately no config-free constructor.
    pub(crate) fn resolve(state: &AppState, config: &mold_core::Config) -> Self {
        Self {
            journal_enabled: state.queue_journal.is_enabled(),
            output_enabled: !state.is_output_disabled(config),
            authoritative: state.scheduled_work.v2_authoritative(),
            admission: state.queue_journal.queue_media_admission(),
            media: state.queue_journal.durable_media_capabilities(),
        }
    }

    /// The first unmet conjunct, or `None` when this host generates.
    ///
    /// ONE order, read by `/api/generate`, `/api/generate/stream`,
    /// `/api/generation-batches`, and the capability that advertises them, so
    /// a host can never refuse one route for a reason it accepts on another.
    pub(crate) fn unready(&self) -> Option<DurableAdmissionUnready> {
        if !self.authoritative {
            return Some(DurableAdmissionUnready::SchedulerNotAuthoritative);
        }
        if !self.journal_enabled {
            return Some(DurableAdmissionUnready::QueueJournalDisabled);
        }
        if self.admission.is_none() {
            return Some(DurableAdmissionUnready::AdmissionServiceMissing);
        }
        if !self.output_enabled {
            return Some(DurableAdmissionUnready::GalleryOutputDisabled);
        }
        None
    }

    /// Would this server ever offer restart-safe request media — gallery
    /// output on and an authoritative scheduler. False is a configuration,
    /// never a degradation, which is why `/health` must not report it.
    pub(crate) fn applicable(&self) -> bool {
        self.output_enabled && self.authoritative
    }

    /// True exactly when generation would be admitted, so a capability can
    /// never advertise a protocol the same server refuses.
    pub(crate) fn generation_admitted(&self) -> bool {
        self.unready().is_none()
    }

    /// `capabilities.durable_media`. The ONLY route to the advertised media
    /// value, so applicability cannot be forgotten at the call site — which is
    /// exactly how it was forgotten before.
    pub(crate) fn advertised_media(&self) -> Option<mold_core::DurableMediaCapabilities> {
        self.generation_admitted()
            .then(|| self.media.clone())
            .flatten()
    }

    /// The encrypted request-media store is reconciled and usable.
    pub(crate) fn media_ready(&self) -> bool {
        self.media.is_some()
    }

    pub(crate) fn admission(
        &self,
    ) -> Option<Arc<crate::queue_media_admission::DurableMediaAdmission>> {
        self.admission.clone()
    }
}

// ── /health ───────────────────────────────────────────────────────────────────

/// Which subsystems this process has switched off, by name.
///
/// `/health` is auth-exempt, so it must not carry the reasons: they name host
/// filesystem paths. `GET /api/status` carries those behind authentication as
/// `durable_media.reasons`.
fn degraded_subsystems(state: &AppState, applicable: bool) -> Vec<String> {
    let mut degraded = Vec::new();
    if state.queue_journal.durable_media_is_degraded(applicable) {
        degraded.push(mold_core::HEALTH_SUBSYSTEM_DURABLE_MEDIA.to_string());
    }
    degraded.sort();
    degraded
}

#[utoipa::path(
    get,
    path = "/health",
    tag = "server",
    responses(
        (status = 200, description = "Server is serving", body = mold_core::HealthStatus),
    )
)]
async fn health(State(state): State<AppState>) -> impl IntoResponse {
    // A liveness probe must never wait on anything, so the config lock is
    // sampled rather than awaited: a probe landing during a `/api/config`
    // write reports healthy instead of blocking behind it. `/api/status` is
    // the authority for the degraded state and does await the lock.
    let degraded = match state.config.try_read() {
        Ok(config) => degraded_subsystems(
            &state,
            DurableAdmissionReadiness::resolve(&state, &config).applicable(),
        ),
        Err(_) => Vec::new(),
    };
    // Deliberately still 200 while degraded. A subsystem being off does not
    // stop this process serving, and failing the check would pull a working
    // server out of a load balancer over a degradation that generation
    // survives. Callers reading only the status code are unaffected; the body
    // is what makes the degradation visible long after the startup log.
    (
        StatusCode::OK,
        Json(mold_core::HealthStatus::from_degraded(degraded)),
    )
}

// ── /api/queue ───────────────────────────────────────────────────────────────

fn enrich_queue_plan_runtime(
    plan: &mut Option<mold_core::QueuePlan>,
    registry: &crate::job_registry::JobRegistry,
) {
    let Some(plan) = plan else { return };
    for work in &mut plan.work_items {
        if work.activity_phase != mold_core::QueueActivityPhase::Active {
            continue;
        }
        let Some(progress) = registry.progress_snapshot(&work.work_id) else {
            continue;
        };
        let progress = progress.unwrap_or_default();
        let running = progress.step.is_some();
        work.runtime_phase = Some(if running { "running" } else { "loading" }.to_string());
        work.runtime_stage = progress.stage.clone().or_else(|| {
            progress
                .weight_load
                .as_ref()
                .map(|load| format!("Loading {}", load.component))
        });
        let (current, total) = if running {
            (
                progress.step.map(|value| value as u64),
                progress.total.map(|value| value as u64),
            )
        } else {
            progress.weight_load.as_ref().map_or((None, None), |load| {
                (Some(load.bytes_loaded), Some(load.bytes_total))
            })
        };
        work.runtime_current = current;
        work.runtime_total = total;
    }
}

/// Bounded snapshot of jobs currently queued or running on the server. Clients
/// (notably the web SPA) poll this to reconcile their local card list — any
/// "running" card whose server id isn't here is a zombie left over from a
/// dropped SSE stream and should be dead-lettered. Durable continuation is
/// explicit: the default and maximum page size are the hydrated runtime
/// window, so an omitted query can never materialize an unbounded backlog.
#[utoipa::path(
    get,
    path = "/api/queue",
    tag = "queue",
    params(
        ("limit" = Option<usize>, Query, description = "Positive durable-row page size, bounded by queue_capacity; omit to use queue_capacity"),
        ("cursor" = Option<String>, Query, description = "Opaque exclusive cursor returned by the preceding page; limit may be omitted to use queue_capacity"),
    ),
    responses(
        (status = 200, description = "Queue snapshot", body = QueueListingResponse),
        (status = 400, description = "Invalid pagination request"),
    )
)]
async fn list_queue(
    State(state): State<AppState>,
    Query(query): Query<QueueListQuery>,
) -> Result<Json<QueueListingResponse>, ApiError> {
    let mut listing = state.job_registry.snapshot();
    listing.plan = state.scheduled_work.latest_plan();
    enrich_queue_plan_runtime(&mut listing.plan, &state.job_registry);
    let requested_page = QueuePageRequest::parse(query, state.queue_capacity)?;
    let explicit_page = requested_page.explicit;

    if !state.queue_journal.is_enabled() {
        // With no durable authority there is nothing to continue. Preserve
        // the legacy `entries` projection for old clients; the registry is
        // already bounded by the same runtime queue window.
        for entry in &mut listing.entries {
            entry.durable = Some(false);
        }
        return Ok(Json(QueueListingResponse::legacy(listing)));
    }

    let live_ids = listing
        .entries
        .iter()
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    let journal = state.queue_journal.clone();
    let durable_cursor = requested_page.cursor.map(|cursor| cursor.durable);
    let limit = requested_page.limit;
    let (durable_page, durable_live_ids) = spawn_queue_read(move || {
        let page = journal.projection_page(durable_cursor, limit)?;
        let durable_live_ids = journal.owned_row_ids(&live_ids)?;
        Ok((page, durable_live_ids))
    })
    .await?;

    let offset = requested_page.cursor.map_or(0, |cursor| cursor.offset);
    let next_offset = offset.checked_add(durable_page.rows.len()).ok_or_else(|| {
        invalid_queue_page("queue cursor offset cannot represent the returned page")
    })?;
    // Positions count only rows that can run, so the cursor carries that
    // count separately from the traversal offset: a held row on page 1 must
    // not push the first runnable row of page 2 to "#1 in line".
    let schedulable_offset = requested_page
        .cursor
        .map_or(0, |cursor| cursor.schedulable_offset);
    let schedulable_in_page = durable_page
        .rows
        .iter()
        .filter(|row| {
            matches!(
                row.state,
                mold_db::generation_queue::QueueRowState::Queued
                    | mold_db::generation_queue::QueueRowState::Running
            )
        })
        .count();
    let next_schedulable_offset = schedulable_offset
        .checked_add(schedulable_in_page)
        .ok_or_else(|| {
            invalid_queue_page("queue cursor offset cannot represent the returned page")
        })?;
    let next_cursor = durable_page.next_cursor.map(|durable| {
        encode_queue_cursor(QueuePageCursor {
            durable,
            offset: next_offset,
            schedulable_offset: next_schedulable_offset,
        })
    });
    let returned = durable_page.rows.len();
    let (entries, live_only_entries) = project_durable_queue_page(
        durable_page.rows,
        listing.entries,
        &durable_live_ids,
        offset,
        schedulable_offset,
    )?;

    let page = mold_core::QueuePage {
        limit,
        offset,
        returned,
        next_cursor,
    };
    if explicit_page {
        return Ok(Json(QueueListingResponse {
            entries,
            live_only_entries: Some(live_only_entries),
            plan: listing.plan,
            page: Some(page),
        }));
    }

    // Keep old clients useful on the bounded default path: they only know the
    // `entries` array, so fold the bounded registry-only set into the first
    // durable page. Reindex the merged projection to avoid counting a live
    // durable overlay once in the registry and again in SQLite.
    let mut entries = entries;
    entries.extend(live_only_entries);
    crate::job_registry::assign_positions(&mut entries, 0);
    let page = page.next_cursor.is_some().then_some(page);
    Ok(Json(QueueListingResponse {
        entries,
        live_only_entries: None,
        plan: listing.plan,
        page,
    }))
}

/// One queued job in full: its entry and, when the planner has placed it, the
/// plan's own work item for it.
#[derive(Debug, Serialize, utoipa::ToSchema)]
struct QueueJobEntry {
    job: crate::job_registry::JobEntry,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_item: Option<mold_core::QueueWorkItem>,
}

/// Read ONE queued job, settings included.
///
/// `GET /api/queue` cannot answer this. Its durable projection is
/// payload-free on purpose — `QUEUE_PROJECTION_FIRST_PAGE_SQL` never selects
/// `request_json`, because a listing must not read a request body per row — so
/// `job_entry_from_durable_projection` hardcodes `metadata: None` and every
/// durably admitted job shows no settings at all for the whole pre-dispatch
/// window. Asking about one job is the opposite case: reading one body is
/// exactly the point.
///
/// The registry answers when the job is live, because it already holds the
/// metadata the submitting request derived. Otherwise the journal row is read
/// and its metadata derived from `request_json` the same way the durable
/// feeder derives it at replay — media excluded by `OutputMetadata`'s own
/// shape, and never a secret.
#[utoipa::path(
    get,
    path = "/api/queue/{id}",
    tag = "queue",
    params(("id" = String, Path, description = "Queue job id")),
    responses(
        (status = 200, description = "Queue job detail", body = QueueJobEntry),
        (status = 404, description = "Queue job not found"),
    )
)]
async fn get_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<QueueJobEntry>, ApiError> {
    // The drawer's own rule (`studio/lib/queuePosition.ts`): a job's work item
    // is the one whose `work_id` IS the job, or — for a batch parent, whose
    // plan entries are its children — the first child that names it as parent.
    // Matching only `work_id` would answer `null` for exactly the batch parent
    // whose phase a client is asking about.
    let mut plan = state.scheduled_work.latest_plan();
    enrich_queue_plan_runtime(&mut plan, &state.job_registry);
    let work_item = plan.and_then(|plan| {
        let items = plan.work_items;
        items
            .iter()
            .position(|item| item.work_id == id)
            .or_else(|| items.iter().position(|item| item.parent_id == id))
            .map(|index| {
                let mut item = items[index].clone();
                item.normalize_planned_lane_for_presentation();
                item
            })
    });
    // Batch membership and the hold's `retryable` bit are durable state that
    // no live registry row carries, and neither can be derived from the
    // payload row. The projection is read once, before either branch, so a
    // job describes itself the same way whether or not it is hydrated.
    let (projection, window) = if state.queue_journal.is_enabled() {
        let journal = state.queue_journal.clone();
        let lookup_id = id.clone();
        let limit = state.queue_capacity;
        spawn_queue_read(move || {
            let projection = journal.row_projection(&lookup_id)?;
            // The position comes from the SAME bounded durable window
            // `GET /api/queue` pages by default, so the two agree; a row
            // beyond that window has no position either listing can name and
            // reports the window's own length.
            let window = journal.projection_page(None, limit)?;
            Ok((projection, Some(window)))
        })
        .await?
    } else {
        (None, None)
    };
    if let Some(mut job) = state.job_registry.entry(&id) {
        if let Some(projection) = projection.as_ref() {
            job.batch_id = projection.batch_id.clone();
            job.client_batch_id = projection.client_batch_id.clone();
            job.batch_index = projection.batch_index;
        }
        return Ok(Json(QueueJobEntry { job, work_item }));
    }
    let (Some(projection), Some(window)) = (projection, window) else {
        return Err(ApiError::queue_job_not_found(format!(
            "queue job {id} is not queued on this server"
        )));
    };
    let journal = state.queue_journal.clone();
    let row_id = id.clone();
    let row = spawn_queue_read(move || journal.row(&row_id)).await?;
    let Some(row) = row else {
        return Err(ApiError::queue_job_not_found(format!(
            "queue job {id} is not queued on this server"
        )));
    };
    let position = window
        .rows
        .iter()
        .position(|projected| projected.id == projection.id)
        .unwrap_or(window.rows.len());
    Ok(Json(QueueJobEntry {
        job: job_entry_from_durable_row(projection, &row, position),
        work_item,
    }))
}

/// Project one durable row, adding the settings the payload-free projection
/// deliberately cannot carry. The derivation is the durable feeder's own, so a
/// job read here and the same job after replay describe themselves
/// identically; everything else comes from the projection, so a single-job
/// read and the listing cannot disagree about state, holds, or retryability.
fn job_entry_from_durable_row(
    projection: mold_db::generation_queue::GenerationQueueProjection,
    row: &mold_db::generation_queue::GenerationQueueRow,
    position: usize,
) -> crate::job_registry::JobEntry {
    let metadata = serde_json::from_str::<mold_core::GenerateRequest>(&row.request_json)
        .ok()
        .map(|request| {
            Box::new(mold_core::OutputMetadata::from_generate_request(
                &request,
                request.seed.unwrap_or(0),
                request.scheduler,
                mold_core::build_info::version_string(),
            ))
        });
    let mut entry = job_entry_from_durable_projection(projection, position);
    entry.metadata = metadata;
    entry
}

#[derive(Debug, Default, Deserialize)]
struct QueueListQuery {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueuePageCursor {
    durable: mold_db::generation_queue::QueueProjectionCursor,
    /// Durable rows traversed before this page, held rows included.
    offset: usize,
    /// Rows that can run before this page — what the public `position` counts.
    schedulable_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueuePageRequest {
    limit: usize,
    cursor: Option<QueuePageCursor>,
    explicit: bool,
}

impl QueuePageRequest {
    fn parse(query: QueueListQuery, maximum: usize) -> Result<Self, ApiError> {
        debug_assert!(maximum > 0, "queue capacity is a positive runtime contract");
        if query.limit == Some(0) {
            return Err(invalid_queue_page("limit must be positive"));
        }
        let explicit = query.limit.is_some() || query.cursor.is_some();
        let limit = query.limit.unwrap_or(maximum).min(maximum);
        i64::try_from(limit)
            .map_err(|_| invalid_queue_page("limit is outside SQLite's supported range"))?;
        Ok(Self {
            limit,
            cursor: query
                .cursor
                .as_deref()
                .map(decode_queue_cursor)
                .transpose()?,
            explicit,
        })
    }
}

fn invalid_queue_page(message: impl Into<String>) -> ApiError {
    ApiError::with_code(message, "INVALID_QUEUE_PAGE", StatusCode::BAD_REQUEST)
}

/// Cursor layout: one version byte, then signed big-endian SQLite ordering
/// keys and an unsigned traversal offset. URL-safe base64 keeps every cursor
/// opaque and query-string safe; clients never construct or inspect it.
fn encode_queue_cursor(cursor: QueuePageCursor) -> String {
    const VERSION: u8 = 2;
    let mut bytes = [0_u8; 33];
    bytes[0] = VERSION;
    bytes[1..9].copy_from_slice(&cursor.durable.created_at_ms.to_be_bytes());
    bytes[9..17].copy_from_slice(&cursor.durable.rowid.to_be_bytes());
    bytes[17..25].copy_from_slice(&(cursor.offset as u64).to_be_bytes());
    bytes[25..33].copy_from_slice(&(cursor.schedulable_offset as u64).to_be_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn decode_queue_cursor(raw: &str) -> Result<QueuePageCursor, ApiError> {
    const VERSION: u8 = 2;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(raw)
        .map_err(|_| invalid_queue_page("cursor is malformed"))?;
    let bytes: [u8; 33] = bytes
        .try_into()
        .map_err(|_| invalid_queue_page("cursor is malformed"))?;
    if bytes[0] != VERSION {
        return Err(invalid_queue_page("cursor version is unsupported"));
    }
    let created_at_ms = i64::from_be_bytes(bytes[1..9].try_into().expect("fixed cursor slice"));
    let rowid = i64::from_be_bytes(bytes[9..17].try_into().expect("fixed cursor slice"));
    if rowid <= 0 {
        return Err(invalid_queue_page("cursor is malformed"));
    }
    let offset = usize::try_from(u64::from_be_bytes(
        bytes[17..25].try_into().expect("fixed cursor slice"),
    ))
    .map_err(|_| invalid_queue_page("cursor offset is unsupported on this server"))?;
    let schedulable_offset = usize::try_from(u64::from_be_bytes(
        bytes[25..33].try_into().expect("fixed cursor slice"),
    ))
    .map_err(|_| invalid_queue_page("cursor offset is unsupported on this server"))?;
    if schedulable_offset > offset {
        return Err(invalid_queue_page("cursor is malformed"));
    }
    Ok(QueuePageCursor {
        durable: mold_db::generation_queue::QueueProjectionCursor {
            created_at_ms,
            rowid,
        },
        offset,
        schedulable_offset,
    })
}

async fn spawn_queue_read<T, F>(read: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(read)
        .await
        .map_err(|error| ApiError::internal(format!("queue read task failed: {error}")))?
        .map_err(|error| ApiError::internal(format!("queue read failed: {error:#}")))
}

async fn spawn_queue_mutation<T, F>(mutation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(mutation)
        .await
        .map_err(|error| ApiError::internal(format!("queue mutation task failed: {error}")))?
        .map_err(|error| ApiError::internal(format!("queue mutation failed: {error:#}")))
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
struct QueueListingResponse {
    entries: Vec<crate::job_registry::JobEntry>,
    /// Active jobs that intentionally have no durable row (for example H3,
    /// identity, reference-authority, or oversized requests). Repeated on
    /// every explicit page and bounded by the runtime queue capacity; clients
    /// merge both arrays by job id before reconciling local work.
    #[serde(skip_serializing_if = "Option::is_none")]
    live_only_entries: Option<Vec<crate::job_registry::JobEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<mold_core::QueuePlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    page: Option<mold_core::QueuePage>,
}

impl QueueListingResponse {
    fn legacy(listing: crate::job_registry::QueueListing) -> Self {
        Self {
            entries: listing.entries,
            live_only_entries: None,
            plan: listing.plan,
            page: None,
        }
    }
}

#[utoipa::path(
    get,
    path = "/api/queue/{id}/preview",
    tag = "queue",
    params(("id" = String, Path, description = "Server generation job ID")),
    responses(
        (status = 200, description = "Latest live progress snapshot, or null before the first progress event", body = Option<mold_core::queue_progress::QueueJobProgress>),
        (status = 404, description = "Job is no longer live"),
    )
)]
async fn get_queue_job_preview(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Option<mold_core::queue_progress::QueueJobProgress>>, ApiError> {
    state
        .job_registry
        .progress_snapshot(&id)
        .map(Json)
        .ok_or_else(|| ApiError::queue_job_not_found(format!("queue job {id} is no longer live")))
}

/// Turn a payload-free durable page into public queue rows, overlaying the
/// registry only when that exact durable id is live. The complementary
/// registry-only set is returned separately because those jobs have no
/// `(created_at,rowid)` key and therefore cannot honestly participate in the
/// durable cursor order.
fn project_durable_queue_page(
    durable_rows: Vec<mold_db::generation_queue::GenerationQueueProjection>,
    live_entries: Vec<crate::job_registry::JobEntry>,
    durable_live_ids: &std::collections::HashSet<String>,
    offset: usize,
    schedulable_offset: usize,
) -> Result<
    (
        Vec<crate::job_registry::JobEntry>,
        Vec<crate::job_registry::JobEntry>,
    ),
    ApiError,
> {
    let mut live_by_id = std::collections::HashMap::with_capacity(durable_live_ids.len());
    let mut live_only_entries = Vec::new();
    for mut entry in live_entries {
        if durable_live_ids.contains(&entry.id) {
            live_by_id.insert(entry.id.clone(), entry);
        } else {
            entry.durable = Some(false);
            live_only_entries.push(entry);
        }
    }

    let mut entries = Vec::with_capacity(durable_rows.len());
    for (page_index, row) in durable_rows.into_iter().enumerate() {
        let position = offset
            .checked_add(page_index)
            .ok_or_else(|| invalid_queue_page("queue cursor position is out of range"))?;
        if let Some(mut entry) = live_by_id.remove(&row.id) {
            entry.durable = Some(true);
            entry.replayed = Some(row.replay_seen > 0);
            entry.dispatch_attempts = Some(row.dispatch_attempts);
            entry.batch_id = row.batch_id;
            entry.client_batch_id = row.client_batch_id;
            entry.batch_index = row.batch_index;
            // The durable row is the traversal authority. Keeping the
            // registry's old position here and then offsetting SQLite-only
            // rows by the registry length counted this same job twice.
            entry.position = position;
            entries.push(entry);
            continue;
        }

        entries.push(job_entry_from_durable_projection(row, position));
    }
    // The durable page is traversed by `(created_at, rowid)`, held rows
    // included; the public position only counts rows that can run, so it
    // starts from the runnable rows before this page rather than `offset`.
    crate::job_registry::assign_positions(&mut entries, schedulable_offset);
    Ok((entries, live_only_entries))
}

fn job_entry_from_durable_projection(
    row: mold_db::generation_queue::GenerationQueueProjection,
    position: usize,
) -> crate::job_registry::JobEntry {
    let state = match row.state {
        mold_db::generation_queue::QueueRowState::Queued => {
            crate::job_registry::JobLifecycle::Queued
        }
        mold_db::generation_queue::QueueRowState::Running => {
            // A durable running row without a live registry owner belongs to a
            // prior runtime and is awaiting startup reconciliation.
            crate::job_registry::JobLifecycle::Queued
        }
        mold_db::generation_queue::QueueRowState::Paused => {
            crate::job_registry::JobLifecycle::Paused
        }
        mold_db::generation_queue::QueueRowState::Held => crate::job_registry::JobLifecycle::Held,
    };
    let error = row.held_reason.clone();
    crate::job_registry::JobEntry {
        id: row.id,
        model: row.model,
        state,
        started_at_unix_ms: row.created_at_ms.max(0) as u64,
        position,
        gpu: None,
        target_gpu: row.target_gpu,
        seed_pinned: Some(row.seed_pinned),
        metadata: None,
        durable: Some(true),
        replayed: Some(row.replay_seen > 0),
        dispatch_attempts: Some(row.dispatch_attempts),
        held_reason: row.held_reason,
        error,
        retryable: (state == crate::job_registry::JobLifecycle::Held).then_some(row.retryable),
        batch_id: row.batch_id,
        client_batch_id: row.client_batch_id,
        batch_index: row.batch_index,
    }
}

/// Wrap any present JSON value (including `null`) in `Some`, so a field using
/// this as its `deserialize_with` distinguishes *absent* (`None`) from an
/// explicit `null` (`Some(None)`). Lets a `position`-only PATCH omit
/// `target_gpu` without resetting the lane to Auto.
fn deserialize_some<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct QueuePatchRequest {
    /// Preferred GPU lane. Absent leaves the lane unchanged; `null` means Auto;
    /// a number pins that ordinal. The double `Option` distinguishes absent
    /// from an explicit `null` so a `position`-only reorder never clobbers it.
    #[serde(default, deserialize_with = "deserialize_some")]
    #[schema(value_type = Option<usize>)]
    target_gpu: Option<Option<usize>>,
    /// Preferred stable device ID. Absent leaves the pin unchanged; `null`
    /// means Auto. Stable IDs are the durable API and take the place of
    /// ordinal pins for new clients.
    #[serde(default, deserialize_with = "deserialize_some")]
    #[schema(value_type = Option<String>)]
    hard_pinned_device_id: Option<Option<String>>,
    /// New 0-based index for the job among the queued jobs. Clamped into range
    /// (a large value sends it to the back). Absent means no reorder.
    #[serde(default)]
    position: Option<usize>,
}

#[utoipa::path(
    patch,
    path = "/api/queue/{id}",
    tag = "queue",
    request_body = QueuePatchRequest,
    responses(
        (status = 200, description = "Updated queue entry", body = crate::job_registry::JobEntry),
        (status = 404, description = "Queue job not found"),
        (status = 409, description = "Queue job is already running"),
        (status = 422, description = "Invalid GPU target"),
    )
)]
async fn patch_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<QueuePatchRequest>,
) -> Result<Json<crate::job_registry::JobEntry>, ApiError> {
    if let Some(Some(target)) = req.target_gpu {
        let available = state
            .gpu_pool
            .workers
            .iter()
            .any(|w| w.gpu.ordinal == target);
        if !available {
            return Err(ApiError::validation(format!(
                "gpu:{target} is not available in this server's worker pool"
            )));
        }
    }
    let stable_target_gpu = match req.hard_pinned_device_id.as_ref() {
        Some(Some(id)) => {
            let state_now = current_device_state(&state);
            let device = state_now
                .devices
                .iter()
                .find(|device| device.id == *id)
                .ok_or_else(|| {
                    ApiError::validation(format!("device {id} is not visible on this server"))
                })?;
            Some(Some(device.ordinal.ok_or_else(|| {
                ApiError::validation(format!("device {id} has no schedulable worker ordinal"))
            })?))
        }
        Some(None) => Some(None),
        None => None,
    };
    if let (Some(legacy), Some(stable)) = (req.target_gpu, stable_target_gpu) {
        if legacy != stable {
            return Err(ApiError::validation(
                "target_gpu and hard_pinned_device_id resolve to different devices",
            ));
        }
    }
    let resolved_target_gpu = stable_target_gpu.or(req.target_gpu);

    // SQLite owns jobs beyond the hydrated window. Mutate that authority
    // first and fail the request on any persistence error; returning 200 after
    // only changing the registry would acknowledge a lane/order that a restart
    // silently loses. The DB primitive is one owner/state-fenced IMMEDIATE
    // transaction for target plus position. The durable-transition gate keeps
    // this commit and its later bounded runtime projection ordered with feeder
    // publication and cancellation, but deliberately leaves the scheduler
    // fence free for unrelated grants while SQLite runs.
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let runtime_patch_token = {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        match state.job_registry.begin_queue_patch(&id) {
            Ok(token) => Some(token),
            Err(crate::job_registry::TargetGpuUpdateError::NotFound) => None,
            Err(crate::job_registry::TargetGpuUpdateError::AlreadyRunning) => {
                return Err(ApiError::queue_job_running(format!(
                    "queue job {id} is already running; only queued jobs can be reordered or re-laned"
                )));
            }
        }
    };
    let journal = state.queue_journal.clone();
    let mutation_id = id.clone();
    let requested_position = req.position;
    let requested_device_id = req.hard_pinned_device_id;
    let durable = match spawn_queue_mutation(move || {
        journal.patch_owned_any_queued(
            &mutation_id,
            resolved_target_gpu,
            requested_device_id,
            requested_position,
        )
    })
    .await
    {
        Ok(durable) => durable,
        Err(error) => {
            if let Some(token) = runtime_patch_token {
                let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
                state.job_registry.finish_queue_patch(&id, token);
            }
            return Err(error);
        }
    };

    let durable_entry = match durable {
        mold_db::generation_queue::OwnedQueuedPatchOutcome::Updated {
            position,
            projection,
        } => Some(job_entry_from_durable_projection(projection, position)),
        mold_db::generation_queue::OwnedQueuedPatchOutcome::NotOwned => None,
        mold_db::generation_queue::OwnedQueuedPatchOutcome::NotQueued => {
            if let Some(token) = runtime_patch_token {
                let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
                state.job_registry.finish_queue_patch(&id, token);
            }
            return Err(ApiError::queue_job_running(format!(
                "queue job {id} is no longer queued; only queued jobs can be reordered or re-laned"
            )));
        }
    };

    // A mutation response is also the scheduler acknowledgement: the final
    // lease claim takes this same fence, so no plan built from the old lane or
    // order can grant after this guard is acquired. All SQLite work above has
    // completed before this lock is awaited.
    let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;

    // Hydrated jobs have a second, bounded runtime projection. Apply the same
    // edit only after the durable transaction commits. A durable-only tail row
    // intentionally skips this block and returns its payload-free DB projection.
    if let Some(token) = runtime_patch_token {
        let runtime_result = (|| {
            if !state.job_registry.queue_patch_token_matches(&id, token) {
                return Err(ApiError::queue_job_not_found(format!(
                    "queue job {id} not found"
                )));
            }
            if let Some(target_gpu) = resolved_target_gpu {
                state
                    .job_registry
                    .set_target_gpu(&id, target_gpu)
                    .map_err(|error| match error {
                    crate::job_registry::TargetGpuUpdateError::NotFound => {
                        ApiError::queue_job_not_found(format!("queue job {id} not found"))
                    }
                    crate::job_registry::TargetGpuUpdateError::AlreadyRunning => {
                        ApiError::queue_job_running(format!(
                            "queue job {id} is already running; lane changes only apply to queued jobs"
                        ))
                    }
                    })?;
            }
            if let Some(position) = requested_position {
                state
                    .job_registry
                    .reorder_queued(&id, position)
                    .map_err(|error| match error {
                        crate::job_registry::QueueReorderError::NotFound => {
                            ApiError::queue_job_not_found(format!("queue job {id} not found"))
                        }
                        crate::job_registry::QueueReorderError::AlreadyRunning => {
                            ApiError::queue_job_running(format!(
                            "queue job {id} is already running; only queued jobs can be reordered"
                        ))
                        }
                    })?;
            }
            state
                .job_registry
                .entry(&id)
                .ok_or_else(|| ApiError::queue_job_not_found(format!("queue job {id} not found")))
        })();
        state.job_registry.finish_queue_patch(&id, token);
        return runtime_result.map(Json);
    }

    durable_entry
        .map(Json)
        .ok_or_else(|| ApiError::queue_job_not_found(format!("queue job {id} not found")))
}

/// Cancel a queued or running singleton generation, or an active server-owned
/// batch, by its public parent ID. Running inference is cooperatively stopped
/// at the next model safe point; the request returns as soon as that authority
/// is revoked rather than waiting for GPU teardown.
#[utoipa::path(
    delete,
    path = "/api/queue/{id}",
    tag = "queue",
    params(("id" = String, Path, description = "Queue job id")),
    responses(
        (status = 204, description = "Job cancellation accepted"),
        (status = 404, description = "Queue job not found"),
    )
)]
async fn cancel_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    // Serialize the durable probe, bounded registry revocation, and durable
    // cancellation with feeder publication and PATCH. SQLite can block behind
    // another connection, so the scheduler fence is acquired only for the
    // in-memory lifecycle transition and explicitly dropped before the final
    // DB mutation.
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    match cancel_one_queue_job(&state, &id).await? {
        QueueJobCancelOutcome::Cancelled => Ok(StatusCode::NO_CONTENT),
        QueueJobCancelOutcome::Completing => Err(ApiError::queue_job_not_found(format!(
            "queue job {id} is already completing"
        ))),
        QueueJobCancelOutcome::NotFound => Err(ApiError::queue_job_not_found(format!(
            "queue job {id} not found"
        ))),
    }
}

/// What one row's cancellation did, so the single-job route can answer 404
/// while the batch route simply skips a child nobody can cancel any more.
#[derive(Debug, PartialEq, Eq)]
enum QueueJobCancelOutcome {
    Cancelled,
    Completing,
    NotFound,
}

/// Cancel one queue row. The caller must already hold
/// `lock_durable_transition`, which is what serializes this against feeder
/// publication and PATCH — it is deliberately not taken here so a batch can
/// cancel every child under one transition.
async fn cancel_one_queue_job(
    state: &AppState,
    id: &str,
) -> Result<QueueJobCancelOutcome, ApiError> {
    let journal = state.queue_journal.clone();
    let probe_id = id.to_string();
    let durable_candidate =
        spawn_queue_read(move || journal.owns_cancellable_row(&probe_id)).await?;
    {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        match state.job_registry.cancel_queued(id) {
            Ok(()) => {}
            Err(crate::job_registry::QueuedJobCancelError::AlreadyRunning) => {
                // `cancel_queued` already signalled (or latched) the exact running
                // attempt token while holding the same lifecycle lock used by
                // terminal publication.
            }
            Err(crate::job_registry::QueuedJobCancelError::CompletionClaimed) => {
                return Ok(QueueJobCancelOutcome::Completing);
            }
            // Some retained rows have no registry entry: a held job, and a queued
            // job on a boot with no dispatch owner. This endpoint is the
            // documented way to clear either, and `/api/queue` lists both — so
            // falling straight through to 404 would show an operator work they
            // cannot act on.
            Err(crate::job_registry::QueuedJobCancelError::NotFound) => {
                if !durable_candidate {
                    return Ok(QueueJobCancelOutcome::NotFound);
                }
            }
        }
    }
    // Unconditional, not fence-aware: a cancel that lands during the shutdown
    // drain must not come back after the restart.
    let journal = state.queue_journal.clone();
    let id = id.to_string();
    spawn_queue_mutation(move || journal.cancel_id(&id)).await?;
    Ok(QueueJobCancelOutcome::Cancelled)
}

/// Resume a durable dependency-preparation failure after the dependency or
/// host condition has been corrected. Non-retryable operator holds remain
/// fenced so corrupt media or invalid publication authority cannot be
/// re-executed blindly.
#[utoipa::path(
    post,
    path = "/api/queue/{id}/retry",
    tag = "queue",
    params(("id" = String, Path, description = "Held generation job id")),
    request_body = mold_core::GenerationRetryRequest,
    responses(
        (status = 202, description = "Held job returned to the durable queue"),
        (status = 404, description = "Queue job not found"),
        (status = 409, description = "Job is not a retryable hold"),
    )
)]
async fn retry_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(authority): Json<mold_core::GenerationRetryRequest>,
) -> Result<StatusCode, ApiError> {
    if authority.job_id != id {
        return Err(ApiError::with_code(
            "retry job authority does not match the route",
            "QUEUE_JOB_AUTHORITY_MISMATCH",
            StatusCode::CONFLICT,
        ));
    }
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let journal = state.queue_journal.clone();
    let serving_instance_id = state.instance_id.as_ref().clone();
    match spawn_queue_mutation(move || journal.retry_held(&serving_instance_id, &authority)).await?
    {
        mold_db::generation_batches::OwnedRetry::Retried => Ok(StatusCode::ACCEPTED),
        mold_db::generation_batches::OwnedRetry::NotOwned => Err(ApiError::queue_job_not_found(
            format!("queue job {id} not found"),
        )),
        mold_db::generation_batches::OwnedRetry::AuthorityMismatch => Err(ApiError::with_code(
            format!("queue job {id} retry authority changed"),
            "QUEUE_JOB_AUTHORITY_MISMATCH",
            StatusCode::CONFLICT,
        )),
        mold_db::generation_batches::OwnedRetry::NotHeld => Err(ApiError::with_code(
            format!("queue job {id} is not held"),
            "QUEUE_JOB_NOT_HELD",
            StatusCode::CONFLICT,
        )),
        mold_db::generation_batches::OwnedRetry::NotRetryable => Err(ApiError::with_code(
            format!("queue job {id} requires operator repair and cannot be retried"),
            "QUEUE_JOB_NOT_RETRYABLE",
            StatusCode::CONFLICT,
        )),
    }
}

#[utoipa::path(
    post,
    path = "/api/queue/{id}/pause",
    tag = "queue",
    params(("id" = String, Path, description = "Queued generation job id")),
    responses(
        (status = 204, description = "Only this queued job was paused"),
        (status = 404, description = "Queue job not found"),
        (status = 409, description = "Queue job is already running or held"),
    )
)]
async fn pause_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    set_one_queue_job_paused(&state, &id, true).await
}

#[utoipa::path(
    post,
    path = "/api/queue/{id}/resume",
    tag = "queue",
    params(("id" = String, Path, description = "Paused generation job id")),
    responses(
        (status = 204, description = "Only this paused job was resumed"),
        (status = 404, description = "Queue job not found"),
        (status = 409, description = "Queue job is running or held"),
    )
)]
async fn resume_queue_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    set_one_queue_job_paused(&state, &id, false).await
}

/// Change one row's lifecycle without touching `QueuePause`, the host-wide
/// dispatch gate. The existing PATCH token is the scheduler fence: while the
/// SQLite transition runs, only this hydrated row is excluded from grants.
async fn set_one_queue_job_paused(
    state: &AppState,
    id: &str,
    paused: bool,
) -> Result<StatusCode, ApiError> {
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let runtime_token = {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        match state.job_registry.begin_queue_patch(id) {
            Ok(token) => Some(token),
            Err(crate::job_registry::TargetGpuUpdateError::NotFound) => None,
            Err(crate::job_registry::TargetGpuUpdateError::AlreadyRunning) => {
                return Err(ApiError::queue_job_running(format!(
                    "queue job {id} is already running; only waiting jobs can be paused or resumed"
                )));
            }
        }
    };

    let journal = state.queue_journal.clone();
    let mutation_id = id.to_string();
    let durable =
        match spawn_queue_mutation(move || journal.set_job_paused(&mutation_id, paused)).await {
            Ok(outcome) => outcome,
            Err(error) => {
                if let Some(token) = runtime_token {
                    let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
                    state.job_registry.finish_queue_patch(id, token);
                }
                return Err(error);
            }
        };

    use mold_db::generation_queue::OwnedJobPauseOutcome;
    match durable {
        OwnedJobPauseOutcome::NotEligible => {
            if let Some(token) = runtime_token {
                let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
                state.job_registry.finish_queue_patch(id, token);
            }
            return Err(ApiError::queue_job_running(format!(
                "queue job {id} is not eligible to {}",
                if paused { "pause" } else { "resume" }
            )));
        }
        OwnedJobPauseOutcome::NotOwned if runtime_token.is_none() => {
            return Err(ApiError::queue_job_not_found(format!(
                "queue job {id} not found"
            )));
        }
        OwnedJobPauseOutcome::Updated
        | OwnedJobPauseOutcome::Unchanged
        | OwnedJobPauseOutcome::NotOwned => {}
    }

    if let Some(token) = runtime_token {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        let lifecycle = if paused {
            crate::job_registry::JobLifecycle::Paused
        } else {
            crate::job_registry::JobLifecycle::Queued
        };
        if !state
            .job_registry
            .finish_queue_patch_state(id, token, lifecycle)
        {
            return Err(ApiError::queue_job_not_found(format!(
                "queue job {id} changed while its pause state was being updated"
            )));
        }
    }
    if !paused {
        state.queue_journal.wake_feeder();
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Response of `POST /api/queue/pause` and `POST /api/queue/resume` — the
/// resulting pause state (`true` after pause, `false` after resume).
#[derive(Debug, Serialize, utoipa::ToSchema)]
struct QueuePauseResponse {
    paused: bool,
}

/// Response of `DELETE /api/queue` — how many queued jobs were cancelled.
#[derive(Debug, Serialize, utoipa::ToSchema)]
struct QueueCancelAllResponse {
    cancelled: usize,
}

/// Pause dispatch of new generation jobs. The job currently running on a
/// worker finishes; only the *next* job is held. Idempotent — repeat pauses
/// return `{"paused": true}` without re-emitting the `queue_paused` event.
#[utoipa::path(
    post,
    path = "/api/queue/pause",
    tag = "queue",
    responses(
        (status = 200, description = "Queue dispatch paused", body = QueuePauseResponse),
    )
)]
async fn pause_queue(State(state): State<AppState>) -> Result<Json<QueuePauseResponse>, ApiError> {
    let v2_authoritative = state.scheduled_work.v2_authoritative();
    let changed = if v2_authoritative {
        state
            .scheduled_work
            .set_queue_paused(true)
            .await
            .map_err(ApiError::internal)?
    } else {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        state.queue_pause.pause()
    };
    if changed && !v2_authoritative {
        state.events.publish(mold_core::ServerEvent::QueuePaused);
    }
    Ok(Json(QueuePauseResponse { paused: true }))
}

/// Resume dispatch and return every restart-paused durable job to its queue.
/// Idempotent — repeat resumes return `{"paused": false}` without duplicating
/// work or re-emitting the global `queue_resumed` event.
#[utoipa::path(
    post,
    path = "/api/queue/resume",
    tag = "queue",
    responses(
        (status = 200, description = "Queue dispatch resumed", body = QueuePauseResponse),
    )
)]
async fn resume_queue(State(state): State<AppState>) -> Result<Json<QueuePauseResponse>, ApiError> {
    let v2_authoritative = state.scheduled_work.v2_authoritative();
    let changed = if v2_authoritative {
        state
            .scheduled_work
            .set_queue_paused(false)
            .await
            .map_err(ApiError::internal)?
    } else {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        state.queue_pause.resume()
    };
    if changed && !v2_authoritative {
        state.events.publish(mold_core::ServerEvent::QueueResumed);
    }
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let journal = state.queue_journal.clone();
    let resumed = spawn_queue_mutation(move || journal.resume_all_paused()).await?;
    if resumed.generation_jobs > 0 {
        state.queue_journal.wake_feeder();
    }
    if resumed.chain_jobs > 0 {
        if let Some(handle) = state.chain_jobs.as_ref() {
            handle.kick();
        }
    }
    Ok(Json(QueuePauseResponse { paused: false }))
}

/// Cancel every queued or restart-paused generation job on this host, settling
/// each row's batch child as `cancelled`. Running jobs are left untouched —
/// use `DELETE /api/queue/{id}` for one running singleton, or `DELETE
/// /api/generation-batches/{id}` for one batch's children. The returned count
/// is the number of queued or paused rows removed, preserving the queue API
/// contract.
#[utoipa::path(
    delete,
    path = "/api/queue",
    tag = "queue",
    responses(
        (status = 200, description = "Queued jobs cancelled", body = QueueCancelAllResponse),
    )
)]
async fn cancel_all_queue(
    State(state): State<AppState>,
) -> Result<Json<QueueCancelAllResponse>, ApiError> {
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let live_cancelled = {
        let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
        state.job_registry.cancel_all_queued_ids()
    };
    let live_count = live_cancelled.len();
    let journal = state.queue_journal.clone();
    let durable_only =
        spawn_queue_mutation(move || journal.cancel_all_queued(&live_cancelled)).await?;
    let cancelled = live_count
        .checked_add(durable_only)
        .ok_or_else(|| ApiError::internal("queue cancellation count overflow"))?;
    Ok(Json(QueueCancelAllResponse { cancelled }))
}

// ── /api/history ─────────────────────────────────────────────────────────────

/// Default number of history rows returned when `limit` is omitted.
const HISTORY_DEFAULT_LIMIT: usize = 50;
/// Hard cap on `limit` — matches the legacy 500-entry history bound.
const HISTORY_MAX_LIMIT: usize = 500;

/// 503 error code when the metadata DB is disabled (`MOLD_DB_DISABLE=1`).
const HISTORY_UNAVAILABLE: &str = "HISTORY_UNAVAILABLE";

fn history_db(state: &AppState) -> Result<&mold_db::MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::with_code(
            "prompt history is unavailable because the metadata DB is disabled",
            HISTORY_UNAVAILABLE,
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

#[derive(Debug, Deserialize)]
struct HistoryListQuery {
    /// Case-insensitive substring filter over the prompt text.
    query: Option<String>,
    /// Max rows to return (default 50, capped at 500).
    limit: Option<usize>,
}

#[utoipa::path(
    get,
    path = "/api/history",
    tag = "server",
    params(
        ("query" = Option<String>, Query, description = "Substring filter over prompt text (case-insensitive)"),
        ("limit" = Option<usize>, Query, description = "Max rows to return (default 50, max 500)"),
    ),
    responses(
        (status = 200, description = "Prompt history, newest first", body = mold_core::HistoryListing),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
async fn list_history(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HistoryListQuery>,
) -> Result<Json<mold_core::HistoryListing>, ApiError> {
    let db = history_db(&state)?;
    let limit = params
        .limit
        .unwrap_or(HISTORY_DEFAULT_LIMIT)
        .min(HISTORY_MAX_LIMIT);
    let history = mold_db::PromptHistory::new(db);
    let rows = match params
        .query
        .as_deref()
        .map(str::trim)
        .filter(|q| !q.is_empty())
    {
        Some(query) => history.search(query, limit),
        None => history.recent(limit),
    }
    .map_err(|e| ApiError::internal(format!("failed to read prompt history: {e:#}")))?;
    let entries = rows
        .into_iter()
        .map(|e| mold_core::HistoryEntry {
            prompt: e.prompt,
            model: e.model,
            used_at: e.created_at_ms,
        })
        .collect();
    Ok(Json(mold_core::HistoryListing { entries }))
}

#[derive(Debug, Deserialize)]
struct HistoryDeleteQuery {
    /// When present, trim to the most recent N entries instead of clearing.
    keep: Option<usize>,
}

#[utoipa::path(
    delete,
    path = "/api/history",
    tag = "server",
    params(
        ("keep" = Option<usize>, Query, description = "Keep only the most recent N entries instead of clearing everything"),
    ),
    responses(
        (status = 204, description = "Prompt history cleared (or trimmed)"),
        (status = 503, description = "Metadata DB disabled"),
    )
)]
async fn delete_history(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HistoryDeleteQuery>,
) -> Result<StatusCode, ApiError> {
    let db = history_db(&state)?;
    let history = mold_db::PromptHistory::new(db);
    match params.keep {
        Some(keep) => history.trim_to(keep),
        None => history.clear(),
    }
    .map_err(|e| ApiError::internal(format!("failed to clear prompt history: {e:#}")))?;
    Ok(StatusCode::NO_CONTENT)
}

// ── /api/discovery/peers + /api/capabilities ─────────────────────────────────

/// Return the current DNS-SD cache. The serving instance is omitted by stable
/// UUID so the browser is only offered other machines it can connect to.
#[utoipa::path(
    get,
    path = "/api/discovery/peers",
    tag = "server",
    responses(
        (status = 200, description = "Mold servers visible on the serving host's LAN", body = Vec<mold_core::DiscoveryPeer>),
        (status = 503, description = "DNS-SD browsing is unavailable or disabled"),
    )
)]
async fn discovery_peers(
    State(state): State<AppState>,
) -> Result<Json<Vec<mold_core::DiscoveryPeer>>, ApiError> {
    if !state.discovery.can_browse() {
        return Err(ApiError::with_code(
            "LAN discovery is unavailable on this server",
            "DISCOVERY_UNAVAILABLE",
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    let own_instance_id = state.instance_id.as_str();
    let peers = state
        .discovery
        .peers
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .filter(|peer| {
            !peer.is_this_machine && peer.instance_id.as_deref() != Some(own_instance_id)
        })
        .cloned()
        .collect();
    Ok(Json(peers))
}

/// Report the feature toggles a client needs to render correctly (hide the
/// delete button when delete isn't allowed, etc.). Authentication follows the
/// rest of `/api/*` when an API key is configured.
async fn server_capabilities(
    State(state): State<AppState>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
) -> Json<mold_core::ServerCapabilities> {
    let catalog_available = std::env::var("MOLD_CATALOG_DISABLE")
        .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let config = state.config.read().await;
    let expand_settings = config.expand.clone().with_env_overrides();
    let expand_model_present =
        expand_settings.is_local() && config.manifest_model_is_downloaded(&expand_settings.model);
    let expand = expand_capabilities(&expand_settings, expand_model_present);
    let api_key_auth_enabled = auth_state
        .as_ref()
        .is_some_and(|Extension(state)| state.is_some());
    let device_state = current_device_state(&state);
    let minimax_h3 = advertised_private_h3_capability(
        api_key_auth_enabled,
        &config.resolved_models_dir(),
        &device_state,
    );
    // The two source-controlled compact H3 manifests are ordinary model
    // identities. Runtime availability remains represented by the exact
    // additive task capability above rather than a licensing restriction.
    let model_access = mold_core::model_access_capabilities();

    // A host with no server gallery target cannot promise durability for ANY
    // job: the only delivery is the HTTP response, which by definition does
    // not survive a restart, so `record` declines every admission there. That
    // makes it a systematic over-promise rather than an edge case, and clients
    // read this to decide whether to keep polling a job whose stream died.
    let durable_queue = state.queue_journal.is_enabled() && !state.is_output_disabled(&config);
    // One resolved value, total over all four conjuncts. Advertising a batch
    // limit without conjunct 4 is what let a host with a corrupt admission key
    // promise admission and then 503 every POST.
    let readiness = DurableAdmissionReadiness::resolve(&state, &config);
    // Trash and organization both live in the metadata DB; trash additionally
    // needs somewhere to move bytes to. With the DB disabled, DELETE stays a
    // hard delete and the organization routes answer 501.
    let db_available = state.metadata_db.is_some();
    let gallery = mold_core::GalleryCapabilities {
        can_delete: true,
        trash: Some(mold_core::GalleryTrashCapabilities {
            enabled: db_available && !state.is_output_disabled(&config),
            retention_days: config.gallery.effective_trash_retention_days(),
        }),
        organize: db_available,
        bulk_mutations: db_available,
        media_version: true,
        conditional_get: true,
        row_events: true,
    };
    Json(mold_core::ServerCapabilities {
        generation_profile_v1: true,
        licenses: true,
        gallery,
        catalog: mold_core::CatalogCapabilities {
            available: catalog_available,
            families: mold_catalog::families::active_families()
                .map(|family| family.as_str().to_string())
                .collect::<Vec<_>>(),
            sort: mold_catalog::live::CatalogSort::WIRE_VALUES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        },
        model_access,
        minimax_h3,
        discovery: mold_core::DiscoveryCapabilities {
            can_browse: state.discovery.can_browse(),
        },
        events: mold_core::EventsCapabilities { available: true },
        queue: mold_core::QueueCapabilities {
            can_pause: true,
            can_pause_job: true,
            can_cancel_all: true,
            can_reorder: true,
            stable_device_pins: true,
            cooperative_cancellation: true,
            durable_queue,
            heterogeneous_batch_max_outputs: readiness
                .generation_admitted()
                .then_some(MAX_HETEROGENEOUS_BATCH_OUTPUTS as u32),
        },
        video_upscale: mold_core::VideoUpscaleCapabilities {
            available: db_available
                && !state.is_output_disabled(&config)
                && state
                    .generation_unavailable_reason
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .is_none(),
            gallery_image: db_available
                && !state.is_output_disabled(&config)
                && state
                    .generation_unavailable_reason
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .is_none(),
            contract_version: mold_core::VIDEO_UPSCALE_CONTRACT_VERSION,
            source_library: true,
            source_upload: false,
            input_containers: ["mp4", "mov", "webm"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            output_container: "mp4".into(),
            preserves_primary_audio_when_compatible: true,
            supports_vfr: false,
            supports_hdr: false,
            disclosure: mold_core::VIDEO_UPSCALE_DISCLOSURE.into(),
        },
        durable_media: readiness.advertised_media(),
        reference_uploads: mold_core::ReferenceUploadCapabilities {
            // The request-bound upload protocol derives its authority from an
            // authenticated API-key identity. When server auth is disabled,
            // clients retain validated inline references instead.
            available: api_key_auth_enabled,
            // V2 rebinds the request scope to content-probed canonical
            // descriptors as each upload completes. V1 trusted provisional
            // browser AAC packet arithmetic and is intentionally not offered.
            protocol_version: 2,
            requires_api_key: true,
            session_path: "/api/generate/reference-upload-sessions".to_string(),
            upload_path: "/api/generate/reference-upload".to_string(),
            session_handle_header: crate::reference_uploads::SESSION_HANDLE_HEADER.to_string(),
            upload_handle_header: crate::reference_uploads::UPLOAD_HANDLE_HEADER.to_string(),
            max_file_bytes: crate::reference_uploads::MAX_REFERENCE_UPLOAD_FILE_BYTES,
            max_session_bytes: crate::reference_uploads::MAX_REFERENCE_UPLOAD_SESSION_BYTES,
            max_active_sessions:
                crate::reference_uploads::MAX_REFERENCE_UPLOAD_SESSIONS_PER_IDENTITY as u32,
            session_ttl_ms: crate::reference_uploads::REFERENCE_UPLOAD_SESSION_TTL.as_millis()
                as u64,
        },
        devices: device_capabilities(&state.scheduled_work),
        dispatch: dispatch_capabilities(&state.scheduled_work),
        expand: Some(expand),
        // A build that cannot execute identity conditioning advertises
        // nothing, so a client is told "no" rather than being allowed to send
        // fields this server would drop on the floor.
        identity: if mold_core::identity::identity_runtime_available() {
            mold_core::IdentityCapabilities::advertised()
        } else {
            mold_core::IdentityCapabilities::default()
        },
        mesh: Some(mesh_capabilities()),
    })
}

/// What this build can do with 3-D artifacts.
///
/// `generation` is read from the factory rather than from the presence of the
/// manifests, so a build that ships the family contract without an engine arm
/// says so instead of advertising a model it would refuse after admission.
/// Delivery is advertised unconditionally because a stored `.glb` can be
/// listed, served and exported by any build — including one that cannot
/// generate a new one.
fn mesh_capabilities() -> mold_core::MeshCapabilities {
    let generation = matches!(
        mold_inference::factory_family_availability(mold_core::manifest::HUNYUAN3D_FAMILY),
        Some(mold_inference::FactoryFamilyAvailability::Runnable)
    );
    mold_core::MeshCapabilities {
        generation,
        formats: vec![mold_core::OutputFormat::Glb],
        export_formats: vec![mold_core::OutputFormat::Glb, mold_core::OutputFormat::Obj],
        // Geometry only. Flipped by the PBR paint stage, not before — a user
        // must not discover that a render is untextured after waiting for it.
        textures: false,
    }
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn advertised_private_h3_capability(
    api_key_auth_enabled: bool,
    models_root: &std::path::Path,
    device_state: &mold_core::DeviceState,
) -> Option<mold_core::MiniMaxH3Capability> {
    crate::h3_private_bridge::advertised_h3_private_capability(
        api_key_auth_enabled,
        models_root,
        device_state,
    )
}

#[cfg(any(feature = "h3", feature = "h3-private-uat"))]
fn authenticated_private_h3_model_rows(
    capability: &mold_core::MiniMaxH3Capability,
) -> Vec<mold_core::ModelInfoExtended> {
    crate::h3_private_bridge::authenticated_h3_private_model_rows(capability)
}

#[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
fn authenticated_private_h3_model_rows(
    _capability: &mold_core::MiniMaxH3Capability,
) -> Vec<mold_core::ModelInfoExtended> {
    Vec::new()
}

#[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
fn advertised_private_h3_capability(
    _api_key_auth_enabled: bool,
    _models_root: &std::path::Path,
    _device_state: &mold_core::DeviceState,
) -> Option<mold_core::MiniMaxH3Capability> {
    None
}

#[derive(Debug, Deserialize)]
struct Ltx2ControlAdaptersQuery {
    model: String,
    /// Opt into the `Ltx2CameraControlAvailability` envelope. Absent keeps the
    /// bare-array response older clients parse, byte for byte.
    ///
    /// Deliberately a string: `serde_urlencoded` only accepts `true`/`false`
    /// for a `bool`, so a plain `?detail=1` would 400 the whole request.
    #[serde(default)]
    detail: Option<String>,
}

impl Ltx2ControlAdaptersQuery {
    /// `?detail=1`, `?detail=true`, and a bare `?detail` all opt in.
    fn wants_detail(&self) -> bool {
        matches!(self.detail.as_deref(), Some("1" | "true" | "yes" | ""))
    }
}

/// Either shape of `/api/capabilities/ltx2-camera-controls`, chosen by
/// `?detail=`. `untagged` means the list arm serializes as a bare JSON array,
/// exactly as before.
#[derive(Debug, Serialize, utoipa::ToSchema)]
#[serde(untagged)]
enum Ltx2CameraControlsResponse {
    List(Vec<mold_core::Ltx2CameraControlInfo>),
    Detailed(mold_core::Ltx2CameraControlAvailability),
}

#[utoipa::path(
    get,
    path = "/api/capabilities/ltx2-control-adapters",
    tag = "generation",
    params(("model" = String, Query, description = "Installed LTX-2 model ID")),
    responses(
        (status = 200, description = "Compatible built-in IC-LoRA controls", body = Vec<mold_core::Ltx2ControlAdapterInfo>),
        (status = 422, description = "Model is not a supported distilled LTX-2 profile")
    )
)]
async fn capabilities_ltx2_control_adapters(
    State(state): State<AppState>,
    Query(query): Query<Ltx2ControlAdaptersQuery>,
) -> Result<Json<Vec<mold_core::Ltx2ControlAdapterInfo>>, ApiError> {
    let config = state.config.read().await;
    let effective_config =
        crate::model_manager::resolve_existing_model_authority(&query.model, &config)?
            .map_or_else(|| config.clone(), |authority| authority.config);
    let model_config = effective_config.resolved_model_config(&query.model);
    let profile = mold_core::ltx2_control::control_profile_for_model(&query.model, &model_config)
        .map_err(ApiError::validation)?;
    let models_dir = config.resolved_models_dir();
    let adapters = mold_core::ltx2_control::adapters_for_profile(profile)
        .map(|adapter| {
            let manifest = mold_core::manifest::find_manifest(adapter.download_model)
                .expect("control registry and hidden manifests must stay in sync");
            // `control_artifact_is_complete` already checks every companion
            // relative to the weights' directory, so resolve the weights once
            // and ask once. Asking per manifest file would re-run the whole
            // check N times, and would look for the weights in a companion's
            // directory if the two ever stopped sharing one.
            let weights = manifest
                .files
                .iter()
                .find(|file| file.hf_filename == adapter.hf_filename)
                .expect("control registry and hidden manifests must stay in sync");
            let installed = control_artifact_is_complete(
                adapter,
                &models_dir.join(mold_core::manifest::storage_path(manifest, weights)),
            );
            mold_core::Ltx2ControlAdapterInfo {
                id: adapter.id.to_string(),
                label: adapter.label.to_string(),
                guide: adapter.guide.to_string(),
                size_bytes: adapter.total_size_bytes(),
                installed,
                download_model: adapter.download_model.to_string(),
                download_repo: adapter.hf_repo.to_string(),
                download_filename: adapter.hf_filename.to_string(),
                download_sha256: adapter.sha256.to_string(),
                gated: adapter.gated,
            }
        })
        .collect();
    Ok(Json(adapters))
}

#[utoipa::path(
    get,
    path = "/api/capabilities/ltx2-camera-controls",
    tag = "generation",
    params(
        ("model" = String, Query, description = "Installed LTX-2 model ID"),
        (
            "detail" = Option<String>,
            Query,
            description = "`1`, `true`, or bare to return the availability envelope with the                            host's reason instead of a bare array"
        ),
    ),
    responses(
        (status = 200, description = "Compatible built-in camera controls; a bare array, or an availability envelope for `detail=1`", body = Ltx2CameraControlsResponse),
        (status = 422, description = "Model is not an LTX-2 model")
    )
)]
async fn capabilities_ltx2_camera_controls(
    State(state): State<AppState>,
    Query(query): Query<Ltx2ControlAdaptersQuery>,
) -> Result<Json<Ltx2CameraControlsResponse>, ApiError> {
    let detail = query.wants_detail();
    // The bare array cannot carry a reason, so for an older client the
    // 19B-only case stays an empty list — exactly what it returns today. Every
    // other failure keeps its existing status for that client (unknown
    // architecture still 422s) rather than being silently flattened here. A
    // `detail=1` client gets the reason instead of guessing at server policy.
    let unsupported = |reason: String| -> Result<Json<Ltx2CameraControlsResponse>, ApiError> {
        if detail {
            Ok(Json(Ltx2CameraControlsResponse::Detailed(
                mold_core::Ltx2CameraControlAvailability {
                    controls: Vec::new(),
                    supported: false,
                    unsupported_reason: Some(reason),
                },
            )))
        } else {
            Ok(Json(Ltx2CameraControlsResponse::List(Vec::new())))
        }
    };

    let config = state.config.read().await;
    let effective_config =
        crate::model_manager::resolve_existing_model_authority(&query.model, &config)?
            .map_or_else(|| config.clone(), |authority| authority.config);
    let model_config = effective_config.resolved_model_config(&query.model);
    let profile =
        match mold_core::ltx2_camera::camera_profile_for_model(&query.model, &model_config) {
            Ok(profile) => profile,
            Err(error) if error.contains("published for LTX-2 19B only") => {
                return unsupported(error);
            }
            // An unknown architecture is a legitimate "no presets here"
            // answer, not a client error. It used to 422, which every client
            // caught into an unexplained empty picker.
            Err(error) if detail => return unsupported(error),
            Err(error) => return Err(ApiError::validation(error)),
        };
    let models_dir = config.resolved_models_dir();
    let controls = mold_core::ltx2_camera::camera_controls_for_profile(profile)
        .map(|preset| {
            let manifest = mold_core::manifest::find_manifest(preset.download_model)
                .expect("camera-control registry and hidden manifests must stay in sync");
            let installed = manifest.files.iter().all(|file| {
                let path = models_dir.join(mold_core::manifest::storage_path(manifest, file));
                camera_control_artifact_is_complete(preset, &path)
            });
            mold_core::Ltx2CameraControlInfo {
                id: preset.id.to_string(),
                label: preset.label.to_string(),
                size_bytes: preset.size_bytes,
                installed,
                download_model: preset.download_model.to_string(),
                download_repo: preset.hf_repo.to_string(),
                download_filename: preset.hf_filename.to_string(),
                download_sha256: preset.sha256.to_string(),
            }
        })
        .collect();
    Ok(Json(if detail {
        Ltx2CameraControlsResponse::Detailed(mold_core::Ltx2CameraControlAvailability {
            supported: true,
            unsupported_reason: None,
            controls,
        })
    } else {
        Ltx2CameraControlsResponse::List(controls)
    }))
}

fn device_capabilities(
    handle: &crate::scheduler::ScheduledWorkHandle,
) -> mold_core::DeviceCapabilities {
    let v2_authoritative = handle.v2_authoritative();
    mold_core::DeviceCapabilities {
        available: true,
        lifecycle: v2_authoritative,
        restart_enable: !v2_authoritative,
        stable_pins: true,
        planned_lanes: v2_authoritative,
        learned_eta: v2_authoritative,
    }
}

fn dispatch_capabilities(
    handle: &crate::scheduler::ScheduledWorkHandle,
) -> mold_core::DispatchCapabilities {
    mold_core::DispatchCapabilities {
        modes: ["legacy", "observe", "v2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        active_mode: Some(handle.dispatch_mode().as_str().to_string()),
        v2_authoritative: handle.v2_authoritative(),
        observes_v2_decisions: handle.observes_v2_decisions(),
        request_placement_preview: handle.v2_authoritative()
            && handle.placement_preview_available(),
    }
}

/// Derive the wire capability without constructing an expander or probing an
/// external API. `model_present` is supplied by the caller so the feature and
/// backend permutations remain pure and directly testable.
fn expand_capabilities(
    settings: &mold_core::expand::ExpandSettings,
    model_present: bool,
) -> mold_core::ExpandCapabilities {
    if settings.is_local() {
        mold_core::ExpandCapabilities {
            configured: cfg!(feature = "expand"),
            model_present: Some(model_present),
            backend: mold_core::ExpandBackend::Local,
            remix: true,
            // Naming the model is what lets a client offer to pull it without
            // hard-coding the manifest id; a custom `expand.model` is honoured.
            model: Some(settings.model.trim().to_string()).filter(|m| !m.is_empty()),
        }
    } else {
        mold_core::ExpandCapabilities {
            configured: !settings.backend.trim().is_empty(),
            model_present: None,
            backend: mold_core::ExpandBackend::Api,
            remix: true,
            model: None,
        }
    }
}

// ── /api/capabilities/chain-limits ───────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/capabilities/chain-limits",
    tag = "server",
    params(
        ("model" = String, Query, description = "Model name (e.g. ltx-2-19b-distilled:fp8)"),
        ("fps" = Option<u32>, Query, description = "fps the clips will render at. LTX-2's per-clip cap is a runtime duration, so the returned cap moves with this; defaults to the model's own default fps."),
    ),
    responses(
        (status = 200, description = "Chain limits for the requested model",
         body = crate::chain_limits::ChainLimits),
        (status = 400, description = "Missing required 'model' query parameter"),
        (status = 404, description = "Unknown or unsupported model"),
    )
)]
async fn capabilities_chain_limits(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> axum::response::Response {
    let raw_model = match params.get("model") {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "missing required 'model' query parameter\n",
            )
                .into_response();
        }
    };

    let resolved = mold_core::manifest::resolve_model_name(&raw_model);
    let (family, quant, supports_audio, default_frames, default_fps) =
        if let Some(manifest) = mold_core::manifest::find_manifest(&resolved) {
            let quant = resolved
                .split_once(':')
                .map(|(_, tag)| tag.to_string())
                .unwrap_or_default();
            let family = manifest.family.clone();
            let supports_audio = crate::chain_limits::family_supports_audio(&family);
            let model_config = state.config.read().await.resolved_model_config(&resolved);
            (
                family,
                quant,
                supports_audio,
                model_config.effective_frames(),
                model_config.effective_fps(),
            )
        } else {
            // Installed live-catalog models retain opaque `cv:` / `hf:` ids and
            // therefore cannot be found in the built-in manifest. Resolve them
            // through the same installed-sidecar inventory as `/api/models`;
            // config lookup alone deliberately does not synthesize catalog
            // metadata and would incorrectly return 404 for runnable installs.
            let Some(entry) = model_manager::list_models(&state)
                .await
                .into_iter()
                .find(|entry| entry.downloaded && entry.info.name == raw_model)
            else {
                return (StatusCode::NOT_FOUND, "unknown model\n").into_response();
            };
            let family = entry.info.family;
            let supports_audio = entry
                .supports_audio
                .unwrap_or_else(|| crate::chain_limits::family_supports_audio(&family));
            (
                family,
                String::new(),
                supports_audio,
                entry.defaults.default_frames,
                entry.defaults.default_fps,
            )
        };

    if crate::chain_limits::family_cap(&family).is_none() {
        return (StatusCode::NOT_FOUND, "model is not chain-capable\n").into_response();
    }

    // An explicit `?fps=` wins so a client editing the fps control can ask for
    // the cap it will actually be held to; otherwise use the model's default.
    let fps = params
        .get("fps")
        .and_then(|value| value.parse::<u32>().ok())
        .or(default_fps);

    // Chain limits are model-derived on purpose: a recommendation that moved
    // with transient GPU pressure would make the clip-length options flicker
    // in a picker the SPA caches per model. The VRAM-aware answer lives on
    // `POST /api/generate/chain/validate`, which prices the actual stages.
    let mut limits =
        crate::chain_limits::compute_limits(&raw_model, &family, &quant, default_frames, fps);
    limits.supports_audio = supports_audio;
    Json(limits).into_response()
}

// ── /api/shutdown ─────────────────────────────────────────────────────────────

/// Trigger graceful server shutdown.
///
/// When API key auth is enabled, the auth middleware protects this endpoint.
/// When auth is disabled, only requests from loopback addresses (127.0.0.1, ::1)
/// are accepted to prevent remote shutdown.
#[utoipa::path(
    post,
    path = "/api/shutdown",
    tag = "server",
    responses(
        (status = 200, description = "Shutdown initiated"),
        (status = 403, description = "Forbidden — remote shutdown requires API key auth"),
    )
)]
async fn shutdown_server(State(state): State<AppState>, request: Request) -> impl IntoResponse {
    // When auth is disabled (no AuthState extension or AuthState is None),
    // restrict shutdown to loopback addresses only.
    let auth_enabled = request
        .extensions()
        .get::<crate::auth::AuthState>()
        .is_some_and(|s| s.is_some());

    if !auth_enabled {
        let is_loopback = request
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip().is_loopback())
            .unwrap_or(false);
        if !is_loopback {
            return (
                StatusCode::FORBIDDEN,
                "shutdown requires API key auth or localhost access\n",
            );
        }
    }

    tracing::info!("shutdown requested via API");
    if let Some(tx) = state.shutdown_tx.lock().await.take() {
        let _ = tx.send(());
    }
    (StatusCode::OK, "shutdown initiated\n")
}

// ── /api/gallery ──────────────────────────────────────────────────────────────

const GALLERY_IMPORT_CONTENT_TYPE: &str = "application/vnd.mold.gallery-import";
const GALLERY_IMPORT_HEADER_BYTES: usize = 12;
const MAX_GALLERY_IMPORT_METADATA_BYTES: usize = 1024 * 1024;
const MAX_GALLERY_IMPORT_FILE_BYTES: u64 = 64 * 1024 * 1024 * 1024;

#[derive(Debug, Serialize, utoipa::ToSchema)]
struct GalleryImportResponse {
    filename: String,
}

fn invalid_gallery_import(message: impl Into<String>) -> ApiError {
    ApiError::with_code(
        message,
        "INVALID_GALLERY_IMPORT",
        StatusCode::UNPROCESSABLE_ENTITY,
    )
}

fn gallery_import_too_large(message: impl Into<String>) -> ApiError {
    ApiError::with_code(
        message,
        "GALLERY_IMPORT_TOO_LARGE",
        StatusCode::PAYLOAD_TOO_LARGE,
    )
}

pub(crate) fn validate_gallery_filename(filename: &str) -> Result<(), ApiError> {
    let path = std::path::Path::new(filename);
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || path.components().count() != 1
        || !matches!(
            path.components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err(ApiError::validation(
            "gallery filename must be one normal path component",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GalleryImportDescriptor {
    metadata: mold_core::OutputMetadata,
    metadata_synthetic: bool,
}

struct ParsedGalleryImport {
    descriptor: GalleryImportDescriptor,
    file_len: u64,
    pending_file_bytes: axum::body::Bytes,
    stream: axum::body::BodyDataStream,
}

async fn parse_gallery_import_prefix(
    output_dir: &std::path::Path,
    headers: &HeaderMap,
    body: Body,
) -> Result<ParsedGalleryImport, ApiError> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    if content_type != Some(GALLERY_IMPORT_CONTENT_TYPE) {
        return Err(ApiError::with_code(
            format!("gallery imports require Content-Type {GALLERY_IMPORT_CONTENT_TYPE}"),
            "UNSUPPORTED_GALLERY_IMPORT",
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
        ));
    }
    let maximum_body = MAX_GALLERY_IMPORT_FILE_BYTES
        .saturating_add(MAX_GALLERY_IMPORT_METADATA_BYTES as u64)
        .saturating_add(GALLERY_IMPORT_HEADER_BYTES as u64);
    if headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > maximum_body)
    {
        return Err(gallery_import_too_large(format!(
            "gallery import body exceeds {maximum_body} bytes"
        )));
    }

    let mut stream = body.into_data_stream();
    let mut prefix = Vec::with_capacity(GALLERY_IMPORT_HEADER_BYTES);
    let mut metadata_len: Option<usize> = None;
    let mut file_len: Option<u64> = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|error| invalid_gallery_import(format!("import body failed: {error}")))?;
        let mut offset = 0;
        if prefix.len() < GALLERY_IMPORT_HEADER_BYTES {
            let take = (GALLERY_IMPORT_HEADER_BYTES - prefix.len()).min(chunk.len());
            prefix.extend_from_slice(&chunk[..take]);
            offset += take;
            if prefix.len() == GALLERY_IMPORT_HEADER_BYTES {
                let declared_metadata =
                    u32::from_be_bytes(prefix[..4].try_into().expect("four-byte prefix")) as usize;
                if declared_metadata == 0 {
                    return Err(invalid_gallery_import(
                        "gallery import descriptor cannot be empty",
                    ));
                }
                if declared_metadata > MAX_GALLERY_IMPORT_METADATA_BYTES {
                    return Err(gallery_import_too_large(format!(
                        "gallery import metadata exceeds {MAX_GALLERY_IMPORT_METADATA_BYTES} bytes"
                    )));
                }
                let declared_file =
                    u64::from_be_bytes(prefix[4..].try_into().expect("eight-byte prefix"));
                if declared_file == 0 {
                    return Err(invalid_gallery_import(
                        "gallery import file payload cannot be empty",
                    ));
                }
                if declared_file > MAX_GALLERY_IMPORT_FILE_BYTES {
                    return Err(gallery_import_too_large(format!(
                        "gallery import file exceeds {MAX_GALLERY_IMPORT_FILE_BYTES} bytes"
                    )));
                }
                metadata_len = Some(declared_metadata);
                file_len = Some(declared_file);
                prefix.reserve(declared_metadata);
                let dir = output_dir.to_path_buf();
                tokio::task::spawn_blocking(move || {
                    crate::batch_transaction::preflight_disk_space(&dir, declared_file)
                })
                .await
                .map_err(|error| {
                    ApiError::internal(format!("gallery import preflight task failed: {error}"))
                })?
                .map_err(|error| {
                    ApiError::with_code(
                        format!("gallery import disk preflight failed: {error:#}"),
                        "GALLERY_IMPORT_DISK_FULL",
                        StatusCode::INSUFFICIENT_STORAGE,
                    )
                })?;
            }
        }
        if let Some(declared_metadata) = metadata_len {
            let descriptor_end = GALLERY_IMPORT_HEADER_BYTES + declared_metadata;
            if prefix.len() < descriptor_end {
                let take = (descriptor_end - prefix.len()).min(chunk.len() - offset);
                prefix.extend_from_slice(&chunk[offset..offset + take]);
                offset += take;
            }
            if prefix.len() == descriptor_end {
                let descriptor = serde_json::from_slice(&prefix[GALLERY_IMPORT_HEADER_BYTES..])
                    .map_err(|error| {
                        invalid_gallery_import(format!(
                            "invalid gallery import descriptor: {error}"
                        ))
                    })?;
                return Ok(ParsedGalleryImport {
                    descriptor,
                    file_len: file_len.expect("file length parsed with metadata length"),
                    pending_file_bytes: chunk.slice(offset..),
                    stream,
                });
            }
        }
    }
    Err(invalid_gallery_import(
        "gallery import ended before its descriptor was complete",
    ))
}

async fn stream_gallery_import_file(
    transaction: &crate::batch_transaction::GalleryImportTransaction,
    mut parsed: ParsedGalleryImport,
) -> Result<(GalleryImportDescriptor, u64), ApiError> {
    use tokio::io::AsyncWriteExt as _;

    let staged_path = transaction.staging_path();
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)
        .await
        .map_err(|error| {
            ApiError::internal(format!(
                "failed to create gallery import staging file: {error}"
            ))
        })?;
    let mut written = parsed.pending_file_bytes.len() as u64;
    if written > parsed.file_len {
        return Err(invalid_gallery_import(
            "gallery import contains bytes after its declared file payload",
        ));
    }
    file.write_all(&parsed.pending_file_bytes)
        .await
        .map_err(|error| ApiError::internal(format!("failed to stage gallery import: {error}")))?;
    while let Some(chunk) = parsed.stream.next().await {
        let chunk = chunk
            .map_err(|error| invalid_gallery_import(format!("import body failed: {error}")))?;
        let next = written.saturating_add(chunk.len() as u64);
        if next > parsed.file_len {
            return Err(invalid_gallery_import(
                "gallery import contains bytes after its declared file payload",
            ));
        }
        file.write_all(&chunk).await.map_err(|error| {
            ApiError::internal(format!("failed to stage gallery import: {error}"))
        })?;
        written = next;
    }
    if written != parsed.file_len {
        return Err(invalid_gallery_import(format!(
            "gallery import file ended at {written} bytes; expected {}",
            parsed.file_len
        )));
    }
    file.flush().await.map_err(|error| {
        ApiError::internal(format!(
            "failed to flush gallery import staging file: {error}"
        ))
    })?;
    file.sync_all().await.map_err(|error| {
        ApiError::internal(format!(
            "failed to fsync gallery import staging file: {error}"
        ))
    })?;
    Ok((parsed.descriptor, parsed.file_len))
}

fn files_match(left: &std::path::Path, right: &std::path::Path) -> std::io::Result<bool> {
    use std::io::Read as _;
    if std::fs::metadata(left)?.len() != std::fs::metadata(right)?.len() {
        return Ok(false);
    }
    let mut left = std::io::BufReader::new(std::fs::File::open(left)?);
    let mut right = std::io::BufReader::new(std::fs::File::open(right)?);
    let mut left_buf = [0_u8; 64 * 1024];
    let mut right_buf = [0_u8; 64 * 1024];
    loop {
        let left_read = left.read(&mut left_buf)?;
        let right_read = right.read(&mut right_buf)?;
        if left_read != right_read || left_buf[..left_read] != right_buf[..right_read] {
            return Ok(false);
        }
        if left_read == 0 {
            return Ok(true);
        }
    }
}

/// Stream an already-encoded image/video into the server-owned gallery.
///
/// The fixed binary envelope is `u32 metadata_len`, `u64 file_len`, metadata
/// JSON, then exactly `file_len` bytes. It lets native mirroring preserve
/// metadata and cross-host filename identity without buffering large videos
/// in the server. Authentication is the ordinary API-key middleware; media
/// reads remain on the existing short-lived ticket endpoint.
#[utoipa::path(
    put,
    path = "/api/gallery/import/{filename}",
    tag = "gallery",
    params(("filename" = String, Path, description = "Preferred gallery basename")),
    responses(
        (status = 200, description = "An identical existing file was retained", body = GalleryImportResponse),
        (status = 201, description = "Imported through journaled atomic no-replace publication", body = GalleryImportResponse),
        (status = 409, description = "Identical bytes conflict with existing metadata"),
        (status = 413, description = "Metadata or file exceeds the bounded import envelope"),
        (status = 415, description = "Unsupported import content type"),
        (status = 422, description = "Invalid filename, framing, metadata, or output format"),
    )
)]
async fn import_gallery_file(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(filename): axum::extract::Path<String>,
    body: Body,
) -> Result<(StatusCode, Json<GalleryImportResponse>), ApiError> {
    validate_gallery_filename(&filename)?;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);
    let format = mold_db::metadata_io::format_from_path(std::path::Path::new(&filename))
        .ok_or_else(|| {
            invalid_gallery_import("gallery import must be PNG, JPEG, WebP, GIF, APNG, or MP4")
        })?;
    let parsed_import = parse_gallery_import_prefix(&output_dir, &headers, body).await?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let descriptor = parsed_import.descriptor.clone();
    let mut record = mold_db::GenerationRecord::from_save(
        &output_dir,
        &filename,
        format,
        descriptor.metadata.clone(),
        mold_db::RecordSource::Backfill,
        i64::try_from(timestamp.saturating_mul(1000)).unwrap_or(i64::MAX),
    );
    record.metadata_synthetic = descriptor.metadata_synthetic;
    let parent_id = format!("gallery-import-{}", uuid::Uuid::new_v4());
    let begin_dir = output_dir.clone();
    let requested_filename = filename.clone();
    let mut transaction = tokio::task::spawn_blocking(move || {
        crate::batch_transaction::GalleryImportTransaction::begin(
            &begin_dir,
            &parent_id,
            0,
            serde_json::json!({
                "kind": "gallery_import",
                "requested_filename": requested_filename,
            }),
            record,
        )
    })
    .await
    .map_err(|error| ApiError::internal(format!("gallery import begin task failed: {error}")))?
    .map_err(|error| {
        ApiError::internal(format!("failed to begin atomic gallery import: {error:#}"))
    })?;

    let (descriptor, declared_file_len) =
        match stream_gallery_import_file(&transaction, parsed_import).await {
            Ok(import) => import,
            Err(error) => {
                let _ =
                    tokio::task::spawn_blocking(move || transaction.rollback_unpublished()).await;
                return Err(error);
            }
        };
    let staged_path = transaction.staging_path();
    let descriptor_for_validation = descriptor.clone();
    let validation = tokio::task::spawn_blocking(move || {
        let valid =
            mold_db::metadata_io::is_valid_gallery_file(&staged_path, format, declared_file_len);
        let embedded = mold_db::metadata_io::read_embedded(&staged_path, format);
        let embedded_matches = embedded
            .as_ref()
            .is_none_or(|embedded| embedded == &descriptor_for_validation.metadata);
        (valid, embedded_matches, embedded.is_some())
    })
    .await
    .map_err(|error| {
        ApiError::internal(format!(
            "gallery import metadata validation task failed: {error}"
        ))
    })?;
    if !validation.0 {
        let _ = tokio::task::spawn_blocking(move || transaction.rollback_unpublished()).await;
        return Err(ApiError::with_code(
            "gallery import payload is not a valid gallery media file",
            "INVALID_GALLERY_MEDIA",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    if !validation.1 || (descriptor.metadata_synthetic && validation.2) {
        let _ = tokio::task::spawn_blocking(move || transaction.rollback_unpublished()).await;
        return Err(ApiError::with_code(
            "embedded gallery metadata conflicts with the immutable import descriptor or its synthetic flag",
            "GALLERY_METADATA_CONFLICT",
            StatusCode::CONFLICT,
        ));
    }
    transaction = tokio::task::spawn_blocking(move || {
        if let Err(error) = transaction
            .seal_staged_file()
            .and_then(|()| transaction.mark_prepared())
        {
            let cleanup = transaction.rollback_unpublished();
            return Err(error.context(format!(
                "rolling back failed with: {}",
                cleanup
                    .err()
                    .map(|error| format!("{error:#}"))
                    .unwrap_or_else(|| "clean rollback".to_string())
            )));
        }
        Ok::<_, anyhow::Error>(transaction)
    })
    .await
    .map_err(|error| ApiError::internal(format!("gallery import seal task failed: {error}")))?
    .map_err(|error| {
        ApiError::internal(format!("failed to seal atomic gallery import: {error:#}"))
    })?;

    let gallery_writer = state.gallery_publication_gate.write().await;
    let db = state.metadata_db.clone();
    let events = state.events.clone();
    // An import of a name that is currently in the trash republishes it
    // live. Purge the trashed copy first (bytes, tombstone, row, retirement
    // projection) so the fresh publication never coexists with a row that
    // says "trashed" or with stale `.trash/` bytes; the reconcile and the
    // sweeper both key on the row, so the old copy would otherwise linger.
    {
        let db_for_trash = db.clone();
        let gate_for_trash = state.gallery_publication_gate.clone();
        let media_lifecycle = state.queue_journal.queue_media_lifecycle();
        let dir_for_trash = output_dir.clone();
        let name_for_trash = filename.clone();
        let purged = tokio::task::spawn_blocking(move || -> Result<bool, ApiError> {
            let Some(db) = db_for_trash.as_ref().as_ref() else {
                return Ok(false);
            };
            let trashed = db
                .get(&dir_for_trash, &name_for_trash)
                .map_err(|error| {
                    ApiError::internal(format!(
                        "failed to inspect trashed gallery metadata before import: {error:#}"
                    ))
                })?
                .is_some_and(|row| row.trashed_at_ms.is_some());
            if !trashed {
                return Ok(false);
            }
            crate::gallery_trash::purge_trashed_print_blocking(
                &dir_for_trash,
                &name_for_trash,
                db,
                &gate_for_trash,
                media_lifecycle.as_deref(),
            )?;
            Ok(true)
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("gallery import trash purge task failed: {error}"))
        })?;
        match purged {
            Ok(true) => {
                tracing::info!(file = %filename, "gallery import replaced a trashed print");
            }
            Ok(false) => {}
            Err(error) => {
                let _ =
                    tokio::task::spawn_blocking(move || transaction.rollback_unpublished()).await;
                return Err(error);
            }
        }
    }
    let db_for_existing = db.clone();
    let archive_for_existing = state.gallery_publication_gate.clone();
    let filename_for_task = filename.clone();
    let dir_for_task = output_dir.clone();
    let staged_path = transaction.staging_path();
    enum PreparedGalleryImport {
        Existing(String),
        Transaction {
            transaction: Box<crate::batch_transaction::GalleryImportTransaction>,
            filename: String,
        },
    }
    let prepared = tokio::task::spawn_blocking(move || {
        let desired_path = dir_for_task.join(&filename_for_task);
        let identical = if desired_path.is_file() {
            match files_match(&desired_path, &staged_path) {
                Ok(identical) => identical,
                Err(error) => {
                    let _ = transaction.rollback_unpublished();
                    return Err(ApiError::internal(format!(
                        "failed to compare existing gallery file: {error}"
                    )));
                }
            }
        } else {
            false
        };
        let embedded = mold_db::metadata_io::read_embedded(&staged_path, format);
        if !identical {
            let filename = transaction.manifest().children[0].final_name.clone();
            return Ok(PreparedGalleryImport::Transaction {
                transaction: Box::new(transaction),
                filename,
            });
        }

        let recorded = match db_for_existing
            .as_ref()
            .as_ref()
            .map(|db| db.get(&dir_for_task, &filename_for_task))
            .transpose()
        {
            Ok(recorded) => recorded.flatten(),
            Err(error) => {
                let _ = transaction.rollback_unpublished();
                return Err(ApiError::internal(format!(
                    "failed to inspect existing gallery metadata: {error:#}"
                )));
            }
        };
        let recorded_missing = recorded.is_none();
        let recorded_descriptor =
            recorded.map(|record| (record.metadata, record.metadata_synthetic));
        let archive_index = match archive_for_existing.committed_archive_index(&dir_for_task) {
            Ok(index) => index,
            Err(error) => {
                let _ = transaction.rollback_unpublished();
                return Err(ApiError::internal(format!(
                    "failed to validate committed gallery metadata: {error:#}"
                )));
            }
        };
        let archived_descriptor = archive_index.get(&filename_for_task).map(|entry| {
            (
                entry.record().metadata.clone(),
                entry.record().metadata_synthetic,
            )
        });
        let can_backfill_missing_row = recorded_missing && embedded.is_some();
        let embedded_descriptor = embedded.map(|metadata| (metadata, false));
        let available_descriptors = [
            embedded_descriptor.as_ref(),
            archived_descriptor.as_ref(),
            recorded_descriptor.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if available_descriptors
            .windows(2)
            .any(|pair| pair[0] != pair[1])
        {
            let _ = transaction.rollback_unpublished();
            return Err(ApiError::with_code(
                "existing gallery metadata authorities disagree",
                "GALLERY_METADATA_CONFLICT",
                StatusCode::CONFLICT,
            ));
        }
        let authoritative = embedded_descriptor
            .as_ref()
            .or(archived_descriptor.as_ref())
            .or(recorded_descriptor.as_ref());
        match authoritative {
            Some((metadata, synthetic))
                if metadata == &descriptor.metadata
                    && synthetic == &descriptor.metadata_synthetic => {}
            Some(_) => {
                let _ = transaction.rollback_unpublished();
                return Err(ApiError::with_code(
                    "identical gallery bytes already exist with different metadata",
                    "GALLERY_METADATA_CONFLICT",
                    StatusCode::CONFLICT,
                ));
            }
            None => {
                let _ = transaction.rollback_unpublished();
                return Err(ApiError::with_code(
                    "identical gallery bytes exist but their metadata cannot be verified",
                    "GALLERY_METADATA_CONFLICT",
                    StatusCode::CONFLICT,
                ));
            }
        }
        let metadata = embedded_descriptor
            .map(|(metadata, _)| metadata)
            .or_else(|| recorded_descriptor.map(|(metadata, _)| metadata))
            .unwrap_or_else(|| descriptor.metadata.clone());
        if db_for_existing.as_ref().as_ref().is_some() && can_backfill_missing_row {
            let image = db_for_existing.as_ref().as_ref().and_then(|db| {
                mold_db::persist::record_saved_output_returning(
                    db,
                    &dir_for_task,
                    &filename_for_task,
                    &desired_path,
                    &mold_db::persist::OutputRecordParams {
                        format,
                        metadata: &metadata,
                        source: mold_db::RecordSource::Backfill,
                        generation_time_ms: None,
                        backend: None,
                    },
                )
                .map(|record| Box::new(record.to_gallery_image()))
            });
            events.publish(mold_core::ServerEvent::GalleryAdded {
                filename: filename_for_task.clone(),
                image,
            });
        }
        transaction.rollback_unpublished().map_err(|error| {
            ApiError::internal(format!(
                "failed to retire idempotent gallery import transaction: {error:#}"
            ))
        })?;
        Ok::<_, ApiError>(PreparedGalleryImport::Existing(filename_for_task))
    })
    .await
    .map_err(|error| ApiError::internal(format!("gallery import task failed: {error}")))??;
    drop(gallery_writer);

    let (saved_name, created) = match prepared {
        PreparedGalleryImport::Existing(filename) => (filename, false),
        PreparedGalleryImport::Transaction {
            mut transaction,
            filename,
        } => {
            if let Err(error) = transaction
                .commit(&state.gallery_publication_gate, db.clone())
                .await
            {
                let message = error.to_string();
                if error.entered_committing() {
                    tracing::error!(%message, "atomic gallery import commit is unresolved");
                    drop(error);
                    unreachable!("unresolved commit aborts the process");
                }
                return Err(ApiError::internal(format!(
                    "atomic gallery import commit failed before publication: {message}"
                )));
            }
            let image = db
                .as_ref()
                .as_ref()
                .and_then(|db| db.get(&output_dir, &filename).ok().flatten())
                .map(|record| Box::new(record.to_gallery_image()));
            state.events.publish(mold_core::ServerEvent::GalleryAdded {
                filename: filename.clone(),
                image,
            });
            (filename, true)
        }
    };
    Ok((
        if created {
            StatusCode::CREATED
        } else {
            StatusCode::OK
        },
        Json(GalleryImportResponse {
            filename: saved_name,
        }),
    ))
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub(crate) struct GalleryMediaTokenRequest {
    pub(crate) path: String,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct GalleryMediaTokenResponse {
    pub(crate) token: Option<String>,
    pub(crate) expires_at: Option<u64>,
    pub(crate) auth_required: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum GalleryExportFormat {
    Gif,
    Apng,
    Webp,
}

impl GalleryExportFormat {
    fn output_format(self) -> mold_core::OutputFormat {
        match self {
            Self::Gif => mold_core::OutputFormat::Gif,
            Self::Apng => mold_core::OutputFormat::Apng,
            Self::Webp => mold_core::OutputFormat::Webp,
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Gif => "gif",
            Self::Apng => "png",
            Self::Webp => "webp",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum GalleryGifPlayback {
    #[default]
    Loop,
    Bounce,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum GalleryGifRepeat {
    #[default]
    Forever,
    Once,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub(crate) struct GalleryExportRequest {
    pub(crate) format: GalleryExportFormat,
    #[serde(default)]
    pub(crate) playback: GalleryGifPlayback,
    #[serde(default)]
    pub(crate) repeat: GalleryGifRepeat,
    /// Optional decoded-frame cap. The longest side is resized to this many
    /// pixels while decoding, before frames enter the animation buffer.
    pub(crate) max_dimension: Option<u32>,
    /// Optional target frame rate. The decoder samples without retaining
    /// skipped full-resolution frames.
    pub(crate) fps: Option<u32>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct GalleryExportOptionsResponse {
    pub(crate) formats: Vec<&'static str>,
    pub(crate) gif_playback: [&'static str; 2],
    pub(crate) gif_repeat: [&'static str; 2],
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct PairingSessionResponse {
    pub(crate) token: Option<String>,
    pub(crate) expires_at: Option<u64>,
    pub(crate) auth_required: bool,
    pub(crate) instance_id: String,
    pub(crate) hostname: Option<String>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub(crate) struct PairingClaimRequest {
    pub(crate) token: Option<String>,
    pub(crate) client_name: Option<String>,
    pub(crate) client_kind: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct PairingClaimResponse {
    pub(crate) api_key: Option<String>,
    pub(crate) instance_id: String,
    pub(crate) hostname: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct PairedClientResponse {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) client_kind: String,
    pub(crate) created_at_ms: i64,
    pub(crate) last_used_at_ms: Option<i64>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct PairedClientsResponse {
    pub(crate) auth_required: bool,
    pub(crate) pairing_available: bool,
    pub(crate) clients: Vec<PairedClientResponse>,
}

fn pairing_hostname() -> Option<String> {
    hostname::get()
        .ok()
        .and_then(|value| value.into_string().ok())
        .filter(|value| !value.trim().is_empty())
}

/// Start a two-minute, one-use mobile pairing handoff. The durable key never
/// enters the QR payload; the scanner receives it only after redeeming the
/// high-entropy token against this exact server.
#[utoipa::path(
    post,
    path = "/api/pairing/sessions",
    tag = "server",
    responses(
        (status = 200, description = "Short-lived one-time mobile pairing session", body = PairingSessionResponse),
        (status = 401, description = "API key authentication is required"),
    )
)]
async fn create_pairing_session(
    State(state): State<AppState>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    authority: Option<Extension<crate::auth::PairingAuthority>>,
) -> Result<impl IntoResponse, ApiError> {
    let key_set = auth_state.and_then(|Extension(state)| state);
    let (token, expires_at, auth_required) = match key_set {
        Some(key_set) => {
            let Extension(_) = authority.ok_or_else(|| {
                ApiError::with_code(
                    "only an operator API key can start mobile pairing",
                    "PAIRING_OPERATOR_REQUIRED",
                    StatusCode::FORBIDDEN,
                )
            })?;
            if !key_set.pairing_available() {
                return Err(ApiError::with_code(
                    "paired access is unavailable while the metadata database is disabled",
                    "PAIRING_UNAVAILABLE",
                    StatusCode::SERVICE_UNAVAILABLE,
                ));
            }
            let (token, expires_at) = key_set.issue_pairing_token().map_err(|error| {
                ApiError::internal(format!("failed to create a secure pairing token: {error}"))
            })?;
            (Some(token), Some(expires_at), true)
        }
        None => (None, None, false),
    };
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok((
        headers,
        Json(PairingSessionResponse {
            token,
            expires_at,
            auth_required,
            instance_id: (*state.instance_id).clone(),
            hostname: pairing_hostname(),
        }),
    ))
}

/// Redeem the QR bearer once. This is the sole unauthenticated API route that
/// can return a durable key; its random token is single-use, short-lived, kept
/// only as an HMAC server-side, and the response is explicitly non-cacheable.
#[utoipa::path(
    post,
    path = "/api/pairing/claim",
    tag = "server",
    request_body = PairingClaimRequest,
    responses(
        (status = 200, description = "Pairing credential redeemed", body = PairingClaimResponse),
        (status = 401, description = "Pairing token is invalid, expired, or already used"),
    )
)]
async fn claim_pairing_session(
    State(state): State<AppState>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    Json(request): Json<PairingClaimRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let key_set = auth_state.and_then(|Extension(state)| state);
    let api_key = match key_set {
        Some(key_set) => {
            let token = request
                .token
                .as_deref()
                .filter(|token| !token.is_empty())
                .ok_or_else(|| {
                    ApiError::with_code(
                        "pairing token is missing, expired, or already used",
                        "PAIRING_TOKEN_INVALID",
                        StatusCode::UNAUTHORIZED,
                    )
                })?;
            Some(
                key_set
                    .claim_pairing_token(
                        token,
                        request.client_name.as_deref().unwrap_or("Mold mobile"),
                        request.client_kind.as_deref().unwrap_or("mobile"),
                    )
                    .map_err(|error| {
                        ApiError::internal(format!("failed to create paired access: {error:#}"))
                    })?
                    .ok_or_else(|| {
                        ApiError::with_code(
                            "pairing token is missing, expired, or already used",
                            "PAIRING_TOKEN_INVALID",
                            StatusCode::UNAUTHORIZED,
                        )
                    })?,
            )
        }
        None => None,
    };
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok((
        headers,
        Json(PairingClaimResponse {
            api_key,
            instance_id: (*state.instance_id).clone(),
            hostname: pairing_hostname(),
        }),
    ))
}

#[utoipa::path(
    get,
    path = "/api/pairing/clients",
    tag = "server",
    responses(
        (status = 200, description = "Paired client access grants", body = PairedClientsResponse),
        (status = 403, description = "Operator API key is required"),
    )
)]
async fn list_paired_clients(
    auth_state: Option<Extension<crate::auth::AuthState>>,
    authority: Option<Extension<crate::auth::PairingAuthority>>,
) -> Result<Json<PairedClientsResponse>, ApiError> {
    let key_set = auth_state.and_then(|Extension(state)| state);
    let Some(key_set) = key_set else {
        return Ok(Json(PairedClientsResponse {
            auth_required: false,
            pairing_available: true,
            clients: Vec::new(),
        }));
    };
    if authority.is_none() {
        return Err(ApiError::with_code(
            "only an operator API key can manage paired access",
            "PAIRING_OPERATOR_REQUIRED",
            StatusCode::FORBIDDEN,
        ));
    }
    Ok(Json(PairedClientsResponse {
        auth_required: true,
        pairing_available: key_set.pairing_available(),
        clients: key_set
            .paired_clients()
            .into_iter()
            .map(|client| PairedClientResponse {
                id: client.id,
                name: client.name,
                client_kind: client.client_kind,
                created_at_ms: client.created_at_ms,
                last_used_at_ms: client.last_used_at_ms,
            })
            .collect(),
    }))
}

#[utoipa::path(
    delete,
    path = "/api/pairing/clients/{id}",
    tag = "server",
    params(("id" = String, Path, description = "Paired client grant id")),
    responses(
        (status = 204, description = "Paired client access revoked"),
        (status = 403, description = "Operator API key is required"),
        (status = 404, description = "Paired client was not found"),
    )
)]
async fn revoke_paired_client(
    Path(id): Path<String>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    authority: Option<Extension<crate::auth::PairingAuthority>>,
) -> Result<StatusCode, ApiError> {
    let key_set = auth_state
        .and_then(|Extension(state)| state)
        .ok_or_else(|| {
            ApiError::with_code(
                "authentication is disabled; there is no paired access to revoke",
                "PAIRING_NOT_REQUIRED",
                StatusCode::NOT_FOUND,
            )
        })?;
    if authority.is_none() {
        return Err(ApiError::with_code(
            "only an operator API key can manage paired access",
            "PAIRING_OPERATOR_REQUIRED",
            StatusCode::FORBIDDEN,
        ));
    }
    if key_set
        .revoke_paired_client(&id)
        .map_err(|error| ApiError::internal(format!("failed to revoke paired access: {error:#}")))?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::with_code(
            "paired client was not found",
            "PAIRED_CLIENT_NOT_FOUND",
            StatusCode::NOT_FOUND,
        ))
    }
}

/// Issue a short-lived credential for a browser media element.
///
/// The endpoint itself always uses normal `X-Api-Key` authentication. The
/// auth middleware records only an authenticated marker in request extensions;
/// this handler signs with an independent random per-process secret. The
/// resulting ticket can authenticate GET, HEAD, and Range requests to
/// `/api/gallery/image/:filename` until `expires_at` without exposing the
/// long-lived API key in the URL or making the URL an offline API-key verifier.
#[utoipa::path(
    post,
    path = "/api/gallery/media-token",
    tag = "server",
    request_body = GalleryMediaTokenRequest,
    responses(
        (status = 200, description = "Short-lived gallery media ticket", body = GalleryMediaTokenResponse),
        (status = 401, description = "API key authentication is required"),
        (status = 422, description = "Requested path is not a gallery media path"),
    )
)]
async fn create_gallery_media_token(
    State(state): State<AppState>,
    auth_state: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Json(request): Json<GalleryMediaTokenRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    if !crate::auth::is_gallery_image_path(&request.path) {
        return Err(ApiError::validation(
            "media token path must match /api/gallery/image/:filename",
        ));
    }

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
            let (token, expires_at) = key_set.issue_gallery_media_token(&request.path);
            (Some(token), Some(expires_at), true)
        }
        // Headerless servers need no ticket. Returning an explicit response
        // lets clients with a stale saved key use the ordinary direct URL.
        None => (None, None, false),
    };

    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok((
        headers,
        Json(GalleryMediaTokenResponse {
            token,
            expires_at,
            auth_required,
        }),
    ))
}

#[utoipa::path(
    get,
    path = "/api/gallery/export-options",
    tag = "server",
    responses((status = 200, description = "Available gallery video export formats", body = GalleryExportOptionsResponse))
)]
async fn gallery_export_options() -> Json<GalleryExportOptionsResponse> {
    let formats = if cfg!(feature = "webp") {
        vec!["gif", "apng", "webp"]
    } else {
        vec!["gif", "apng"]
    };
    Json(GalleryExportOptionsResponse {
        formats,
        gif_playback: ["loop", "bounce"],
        gif_repeat: ["forever", "once"],
    })
}

#[utoipa::path(
    post,
    path = "/api/gallery/export/{filename}",
    tag = "server",
    params(("filename" = String, Path, description = "Gallery MP4 filename")),
    request_body = GalleryExportRequest,
    responses(
        (status = 200, description = "Converted animation bytes"),
        (status = 404, description = "Gallery video not found"),
        (status = 422, description = "Unsupported source or export options")
    )
)]
async fn export_gallery_video(
    State(state): State<AppState>,
    Path(filename): Path<String>,
    Json(request): Json<GalleryExportRequest>,
) -> Result<Response, ApiError> {
    // Decoding/quantizing a long animation is CPU and memory intensive. Keep
    // one export in flight per server process so concurrent clients cannot
    // multiply the bounded frame buffer into host-wide memory pressure.
    static EXPORT_PERMIT: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
    let _export_permit = EXPORT_PERMIT
        .acquire()
        .await
        .map_err(|_| ApiError::internal("video export service is unavailable"))?;
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("video output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_default();
    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }
    if !std::path::Path::new(&clean_name)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mp4"))
    {
        return Err(ApiError::validation(
            "only MP4 gallery videos can be exported",
        ));
    }

    let source = output_dir.join(&clean_name);
    if !tokio::fs::metadata(&source)
        .await
        .is_ok_and(|metadata| metadata.is_file())
    {
        return Err(ApiError::not_found("gallery video not found"));
    }

    let output_format = request.format.output_format();
    let bounce = matches!(request.playback, GalleryGifPlayback::Bounce);
    if bounce && !matches!(request.format, GalleryExportFormat::Gif) {
        return Err(ApiError::validation(
            "bounce playback is only supported for GIF exports",
        ));
    }
    let repeat_forever = matches!(request.repeat, GalleryGifRepeat::Forever);
    if request
        .max_dimension
        .is_some_and(|dimension| !(240..=2160).contains(&dimension))
    {
        return Err(ApiError::validation(
            "max_dimension must be between 240 and 2160 pixels",
        ));
    }
    if request.fps.is_some_and(|fps| !(1..=60).contains(&fps)) {
        return Err(ApiError::validation("fps must be between 1 and 60"));
    }
    let target_fps = request.fps;
    let max_dimension = request.max_dimension;
    let bytes = tokio::task::spawn_blocking(move || {
        mold_inference::ltx2::media::export_animation(
            &source,
            output_format,
            bounce,
            repeat_forever,
            target_fps,
            max_dimension,
        )
    })
    .await
    .map_err(|error| ApiError::internal(format!("video export task failed: {error}")))?
    .map_err(|error| ApiError::inference(format!("video export failed: {error:#}")))?;

    let stem = std::path::Path::new(&clean_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("mold-video")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let download_name = format!("{stem}.{}", request.format.extension());
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, output_format.content_type())
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{download_name}\""),
        )
        .header(header::CACHE_CONTROL, "private, no-store")
        .body(Body::from(bytes))
        .expect("static export response headers are valid"))
}

/// `?view=` on `GET /api/gallery`. Absent means the live library.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct GalleryListQuery {
    #[serde(default)]
    pub(crate) view: Option<String>,
    /// Narrow the listing to one print.
    ///
    /// Deliberately a filter on this endpoint rather than a second
    /// `GET /api/gallery/item/:filename` route: the listing's DB read,
    /// committed-archive overlay, organization overlay, and filesystem
    /// fallback are one pipeline with several exits, and a dedicated handler
    /// would have to restate all of it — a second authority that can disagree
    /// with the first about what a print's metadata is.
    ///
    /// It removes the response transfer, not the server-side query: reading
    /// one row's metadata used to mean serializing and shipping the entire
    /// gallery, per artifact.
    #[serde(default)]
    pub(crate) filename: Option<String>,
}

/// Which gallery listing a client asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GalleryView {
    /// Live prints (default) — trashed rows are excluded.
    Library,
    /// Trashed prints only, newest-trashed first.
    Trash,
}

impl GalleryView {
    pub(crate) fn parse(raw: Option<&str>) -> Result<Self, ApiError> {
        match raw.map(str::trim) {
            None | Some("") | Some("library") => Ok(Self::Library),
            Some("trash") => Ok(Self::Trash),
            Some(other) => Err(ApiError::validation(format!(
                "unknown gallery view '{other}'; expected 'library' or 'trash'"
            ))),
        }
    }
}

/// List gallery images from the server's output directory.
///
/// Prefers the SQLite metadata DB when available so listings stay fast on
/// large galleries (no per-request directory walk). Falls back to the
/// filesystem scan when the DB is disabled, can't be opened, or — as a
/// safety net — has no rows for this directory yet (e.g. the reconciliation
/// background task has not finished on first startup).
///
/// Organization state (title, tags, favorite, collections, trash) is folded
/// in AFTER the committed-archive overlay, keyed by filename, so an archive
/// record can never resurrect a trashed print or hide a user's edits.
/// `?view=trash` lists only trashed rows; the DB-less filesystem fallback
/// has no trash index and returns the live scan (library) or nothing
/// (trash).
async fn list_gallery(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<GalleryListQuery>,
) -> Result<Response, ApiError> {
    // Query-time DB recovery is serialized by the DB recovery lock. The
    // shared gallery side excludes an atomic batch publication without
    // needlessly serializing ordinary listings and media reads.
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let view = GalleryView::parse(query.view.as_deref())?;
    // Reject a path-shaped filter rather than silently matching nothing: it
    // is a caller bug, and a quiet empty listing reads as "the print is gone".
    let only = match query.filename.as_deref() {
        None => None,
        Some(name)
            if name.is_empty()
                || std::path::Path::new(name)
                    .file_name()
                    .map(|part| part.to_string_lossy().into_owned())
                    .as_deref()
                    != Some(name) =>
        {
            return Err(ApiError::validation("invalid filename"));
        }
        Some(name) => Some(name.to_string()),
    };
    let only = only.as_deref();
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return gallery_list_response(&headers, Vec::new(), only);
    }
    let output_dir = config.effective_output_dir();
    let retention_days = config.gallery.effective_trash_retention_days();
    drop(config);

    if !output_dir.is_dir() {
        return gallery_list_response(&headers, Vec::new(), only);
    }

    if state.metadata_db.is_some() {
        let db_arc = state.metadata_db.clone();
        let dir = output_dir.clone();
        let gallery_archive = state.gallery_publication_gate.clone();
        let listed = tokio::task::spawn_blocking(move || {
            let Some(db) = db_arc.as_ref().as_ref() else {
                return Ok::<_, anyhow::Error>(None);
            };
            let organization = db
                .organization_for_dir(&dir)
                .map_err(|e| anyhow::anyhow!("gallery organization query failed: {e}"))?;
            match view {
                GalleryView::Trash => {
                    let mut images = db
                        .list_trashed(Some(&dir))?
                        .iter()
                        .map(|row| {
                            let mut image = row.to_gallery_image();
                            crate::gallery_organization::apply_organization(
                                &mut image,
                                organization.get(&row.filename),
                                retention_days,
                            );
                            image
                        })
                        .collect::<Vec<_>>();
                    images.sort_by_key(|image| {
                        std::cmp::Reverse((image.trashed_at.unwrap_or(0), image.timestamp))
                    });
                    Ok(Some(images))
                }
                GalleryView::Library => {
                    let rows = db.list(Some(&dir))?;
                    if rows.is_empty() {
                        return Ok(None);
                    }
                    let archive = gallery_archive.committed_archive_index(&dir)?;
                    let images = archive.overlay_db_gallery(&rows);
                    Ok(Some(crate::gallery_organization::enrich_library_listing(
                        images,
                        &organization,
                        retention_days,
                    )))
                }
            }
        })
        .await
        .map_err(|e| ApiError::internal(format!("gallery DB query failed: {e}")))?
        .map_err(|e| ApiError::internal(format!("gallery DB query failed: {e:#}")))?;
        if let Some(images) = listed {
            return gallery_list_response(&headers, images, only);
        }
    }

    if view == GalleryView::Trash {
        // No trash index without the metadata DB.
        return gallery_list_response(&headers, Vec::new(), only);
    }

    let gallery_archive = state.gallery_publication_gate.clone();
    let images = tokio::task::spawn_blocking(move || {
        let archive_index = gallery_archive.committed_archive_index(&output_dir)?;
        Ok::<_, anyhow::Error>(scan_gallery_dir_with_archive(&output_dir, &archive_index))
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery scan failed: {e}")))?
    .map_err(|e| ApiError::internal(format!("gallery archive validation failed: {e:#}")))?;

    gallery_list_response(&headers, images, only)
}

/// Every listing exit funnels through here, so `?filename=` is applied once
/// and cannot miss a path (the DB view, the archive overlay, the filesystem
/// fallback, and both output-disabled short circuits).
fn gallery_list_response(
    request_headers: &HeaderMap,
    images: Vec<mold_core::GalleryImage>,
    only: Option<&str>,
) -> Result<Response, ApiError> {
    let images = match only {
        Some(filename) => images
            .into_iter()
            .filter(|image| image.filename == filename)
            .collect(),
        None => images,
    };
    let body = serde_json::to_vec(&images)
        .map_err(|error| ApiError::internal(format!("gallery serialization failed: {error}")))?;
    let etag = format!("\"{:x}\"", Sha256::digest(&body));
    let unchanged = request_headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.split(',').any(|candidate| candidate.trim() == etag));

    let mut builder = Response::builder()
        .header(header::ETAG, &etag)
        .header(header::CACHE_CONTROL, "private, no-cache");
    if unchanged {
        builder = builder.status(StatusCode::NOT_MODIFIED);
        return builder
            .body(Body::empty())
            .map_err(|error| ApiError::internal(format!("gallery response failed: {error}")));
    }
    builder
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .map_err(|error| ApiError::internal(format!("gallery response failed: {error}")))
}

/// Serve a gallery file by filename.
///
/// Supports HTTP `Range` requests so `<video>` elements can scrub MP4
/// outputs without downloading the whole clip up front. Partial responses
/// stream straight from disk via `tokio_util::io::ReaderStream` — nothing
/// buffers the full file in server RAM, which matters once a gallery
/// contains multi-GB LTX-2 outputs. Non-range requests still return the
/// whole file (streamed) with `Accept-Ranges: bytes` so the client knows
/// it can seek on subsequent requests.
async fn get_gallery_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<axum::response::Response, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    // Sanitize: prevent directory traversal
    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    // A trashed print still streams: resolve the live path first, then
    // `<dir>/.trash/<name>`, so the trash view and a restore preview render.
    let path = crate::gallery_trash::resolve_gallery_media_source(&output_dir, &clean_name);
    let content_type = content_type_for_filename(&clean_name);
    serve_media_file(&path, &headers, content_type, "image not found").await
}

pub(crate) async fn serve_media_file(
    path: &std::path::Path,
    headers: &HeaderMap,
    content_type: &'static str,
    missing_message: &'static str,
) -> Result<axum::response::Response, ApiError> {
    let meta = match tokio::fs::metadata(path).await {
        Ok(meta) if meta.is_file() => meta,
        _ => return Err(ApiError::not_found(missing_message)),
    };
    let total_len = meta.len();
    let range_header = headers
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok());
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|error| ApiError::internal(format!("failed to open media file: {error}")))?;

    if let Some(raw) = range_header {
        if let Some((start, end)) = parse_byte_range(raw, total_len) {
            return serve_range(file, start, end, total_len, content_type).await;
        }
        return Ok(axum::response::Response::builder()
            .status(StatusCode::RANGE_NOT_SATISFIABLE)
            .header(header::CONTENT_RANGE, format!("bytes */{total_len}"))
            .body(axum::body::Body::empty())
            .unwrap());
    }

    let stream = tokio_util::io::ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, total_len)
        .header(header::CACHE_CONTROL, "private, no-store")
        .body(body)
        .unwrap())
}

/// Parse a `Range: bytes=start-end` header into a concrete (start, end)
/// byte range inclusive on both ends. Returns `None` for unsatisfiable or
/// malformed ranges — the caller translates that into a 416 response.
///
/// Only the single-range form is supported (multipart ranges are vanishingly
/// rare in practice and substantially more complex to implement correctly;
/// browsers for `<video>` always send single ranges).
fn parse_byte_range(header: &str, total_len: u64) -> Option<(u64, u64)> {
    let spec = header.strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (start_s, end_s) = spec.split_once('-')?;
    let start_s = start_s.trim();
    let end_s = end_s.trim();

    if total_len == 0 {
        return None;
    }

    if start_s.is_empty() {
        // Suffix range: `bytes=-N` means "the last N bytes".
        let suffix: u64 = end_s.parse().ok()?;
        if suffix == 0 {
            return None;
        }
        let start = total_len.saturating_sub(suffix);
        return Some((start, total_len - 1));
    }

    let start: u64 = start_s.parse().ok()?;
    if start >= total_len {
        return None;
    }
    let end: u64 = if end_s.is_empty() {
        total_len - 1
    } else {
        end_s.parse().ok()?
    };
    let end = end.min(total_len - 1);
    if end < start {
        return None;
    }
    Some((start, end))
}

/// Emit a `206 Partial Content` response streaming `[start, end]` inclusive
/// from the already-open file handle. `take(len)` bounds the reader so the
/// body terminates exactly at `end + 1` instead of reading the tail.
async fn serve_range(
    mut file: tokio::fs::File,
    start: u64,
    end: u64,
    total_len: u64,
    content_type: &'static str,
) -> Result<axum::response::Response, ApiError> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    file.seek(std::io::SeekFrom::Start(start))
        .await
        .map_err(|e| ApiError::internal(format!("seek failed: {e}")))?;
    let len = end - start + 1;
    let stream = tokio_util::io::ReaderStream::new(file.take(len));
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, len)
        .header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{total_len}"),
        )
        // Gallery media may be authorized by a short-lived bearer ticket, so
        // neither browsers nor intermediaries should retain partial responses.
        .header(header::CACHE_CONTROL, "private, no-store")
        .body(body)
        .unwrap())
}

/// Pick an HTTP Content-Type for a gallery filename. Covers every format
/// `OutputFormat` can emit plus a safe default.
fn content_type_for_filename(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else if lower.ends_with(".apng") {
        "image/apng"
    } else if lower.ends_with(".mp4") {
        "video/mp4"
    } else if lower.ends_with(".wav") {
        "audio/wav"
    } else if lower.ends_with(".glb") {
        "model/gltf-binary"
    } else if lower.ends_with(".obj") {
        "model/obj"
    } else {
        "application/octet-stream"
    }
}

/// Serve a thumbnail for a gallery image. Generated on-demand and cached
/// at ~/.mold/cache/thumbnails/ on the server side.
/// `?size=256|512` and `?fmt=png|jpeg` select a rendition; absent means the
/// historical 256 px PNG, whose cache path and ETag are unchanged.
#[derive(Debug, Default, serde::Deserialize)]
struct ThumbnailQuery {
    size: Option<u32>,
    fmt: Option<String>,
}

/// Names the rendition the server UNDERSTOOD (`256-png`, `512-jpg`), so a
/// client can tell "this server honours `?size`" from an older server that
/// answered its 256 px PNG to every request — a small source legitimately
/// comes back smaller than the tier it asked for, and without this signal a
/// client would misread that as a legacy server.
pub const THUMBNAIL_RENDITION_HEADER: &str = "x-mold-thumbnail-rendition";

async fn get_gallery_thumbnail(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(filename): axum::extract::Path<String>,
    // Not `Option<Query<_>>`: that would swallow a malformed `?size=abc` as
    // the default rendition. Both fields are optional, so an absent query
    // still parses; a malformed one is axum's 400.
    Query(query): Query<ThumbnailQuery>,
) -> Result<Response, ApiError> {
    let variant = crate::thumbnails::ThumbnailVariant::from_query(query.size, query.fmt.as_deref())
        .map_err(ApiError::validation)?;
    let mut response = render_gallery_thumbnail(state, headers, filename, variant).await?;
    if let Ok(value) = header::HeaderValue::from_str(&variant.rendition_label()) {
        response.headers_mut().insert(
            header::HeaderName::from_static(THUMBNAIL_RENDITION_HEADER),
            value,
        );
    }
    Ok(response)
}

async fn render_gallery_thumbnail(
    state: AppState,
    headers: HeaderMap,
    filename: String,
    variant: crate::thumbnails::ThumbnailVariant,
) -> Result<Response, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    let source_path = crate::gallery_trash::resolve_gallery_media_source(&output_dir, &clean_name);
    if !source_path.is_file() {
        return Err(ApiError::not_found(format!(
            "image not found: {clean_name}"
        )));
    }

    let source_metadata = tokio::fs::metadata(&source_path)
        .await
        .map_err(|error| ApiError::internal(format!("failed to stat gallery media: {error}")))?;
    let media_version = file_media_version(&source_metadata);
    let etag = format!("\"thumb-{media_version}{}\"", variant.etag_suffix());

    // Thumbnail cache path: always `.png` regardless of the source extension,
    // so mp4 / gif / apng / webp / jpg all coexist cleanly in the same cache
    // dir and `image.save()` doesn't pick the wrong format from the path.
    let thumb_dir = server_thumbnail_dir();
    let lower = clean_name.to_ascii_lowercase();
    let is_video = lower.ends_with(".mp4");
    // Audio outputs ship a waveform PNG written into the thumbnail cache at
    // save time — there is nothing in a WAV for a raster decoder to read, so
    // a missing cache entry goes straight to the placeholder.
    let is_audio = lower.ends_with(".wav");
    // A mesh is the same shape of problem as audio: its poster PNG is
    // rendered at save time because there is nothing in a glTF buffer for a
    // raster decoder to read. Both therefore share the save-time sidecar name
    // (`<file>.png`) rather than the versioned cache path.
    let is_mesh = crate::thumbnails::is_mesh_filename(&clean_name);
    let thumb_path = if is_audio || is_mesh {
        thumb_dir.join(format!("{clean_name}.png"))
    } else {
        variant.cache_path(&thumb_dir, &clean_name, &media_version)
    };
    if (is_audio || is_mesh) && !thumb_path.is_file() {
        return thumbnail_response(
            &headers,
            "image/svg+xml",
            "public, max-age=300",
            &etag,
            if is_mesh {
                MESH_PLACEHOLDER_SVG.as_bytes().to_vec()
            } else {
                AUDIO_PLACEHOLDER_SVG.as_bytes().to_vec()
            },
        );
    }

    if !thumb_path.is_file() {
        let singleflight = thumbnail_singleflight(&thumb_path);
        let _singleflight_guard = singleflight.lock().await;
        if thumb_path.is_file() {
            let data = tokio::fs::read(&thumb_path)
                .await
                .map_err(|e| ApiError::internal(format!("failed to read thumbnail: {e}")))?;
            let content_type = crate::thumbnails::sniff_content_type(&data).unwrap_or("image/png");
            return thumbnail_response(
                &headers,
                content_type,
                "public, max-age=31536000, immutable",
                &etag,
                data,
            );
        }
        // Generate thumbnail on-demand. Videos go through openh264 for a real
        // first-frame extract (only the first frame is decoded); everything
        // else decodes via the `image` crate. If either path fails, we fall
        // back to serving the source bytes directly — browsers are more
        // lenient about partial / checksum-mismatched images than either
        // decoder, and the SPA would rather show something than a 500.
        let source = source_path.clone();
        let dest = thumb_path.clone();
        let name_for_render = clean_name.clone();
        let gen_result = tokio::task::spawn_blocking(move || {
            let rendered = crate::thumbnails::render_thumbnail(
                &source,
                &name_for_render,
                variant.max_dim,
                variant.format,
            )?;
            write_thumbnail_atomically(&dest, &rendered)
        })
        .await
        .map_err(|e| ApiError::internal(format!("thumbnail generation failed: {e}")))?;

        if let Err(err) = gen_result {
            tracing::warn!(
                file = %clean_name,
                error = %err,
                "thumbnail decode failed; falling back to source bytes"
            );
            // For videos and meshes, the browser can't render the raw bytes
            // as an <img> either, so serving the source doesn't help — and
            // for a mesh it is actively wrong, because it would hand a
            // multi-megabyte glTF buffer to an <img> tag that can only
            // discard it. Fall back to the SVG placeholder instead.
            if is_video || is_mesh {
                return thumbnail_response(
                    &headers,
                    "image/svg+xml",
                    "public, max-age=300",
                    &etag,
                    if is_mesh {
                        MESH_PLACEHOLDER_SVG.as_bytes().to_vec()
                    } else {
                        VIDEO_PLACEHOLDER_SVG.as_bytes().to_vec()
                    },
                );
            }
            let raw = tokio::fs::read(&source_path)
                .await
                .map_err(|e| ApiError::internal(format!("failed to read source: {e}")))?;
            return thumbnail_response(
                &headers,
                content_type_for_filename(&clean_name),
                "public, max-age=300",
                &etag,
                raw,
            );
        }
    }

    let data = tokio::fs::read(&thumb_path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to read thumbnail: {e}")))?;
    // A JPEG-requested tile of a transparent print is stored as PNG under the
    // `.jpg` name, so the type comes from the bytes.
    let content_type = crate::thumbnails::sniff_content_type(&data).unwrap_or("image/png");

    thumbnail_response(
        &headers,
        content_type,
        "public, max-age=31536000, immutable",
        &etag,
        data,
    )
}

// The cache layout and the renderers live in `crate::thumbnails` so the
// desktop app's offline tiles share them; these names stay as thin aliases
// for the route, the warmup, and the trash sweeper.
use crate::thumbnails::{
    file_media_version, versioned_thumbnail_path, AUDIO_PLACEHOLDER_SVG, MESH_PLACEHOLDER_SVG,
    VIDEO_PLACEHOLDER_SVG,
};

fn thumbnail_singleflight(path: &std::path::Path) -> std::sync::Arc<tokio::sync::Mutex<()>> {
    static FLIGHTS: std::sync::LazyLock<
        std::sync::Mutex<
            std::collections::HashMap<std::path::PathBuf, std::sync::Weak<tokio::sync::Mutex<()>>>,
        >,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut flights = FLIGHTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    flights.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = flights.get(path).and_then(std::sync::Weak::upgrade) {
        return lock;
    }
    let lock = std::sync::Arc::new(tokio::sync::Mutex::new(()));
    flights.insert(path.to_path_buf(), std::sync::Arc::downgrade(&lock));
    lock
}

fn thumbnail_response(
    request_headers: &HeaderMap,
    content_type: &'static str,
    cache_control: &'static str,
    etag: &str,
    bytes: Vec<u8>,
) -> Result<Response, ApiError> {
    let unchanged = request_headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.split(',').any(|candidate| candidate.trim() == etag));
    let mut builder = Response::builder()
        .header(header::ETAG, etag)
        .header(header::CACHE_CONTROL, cache_control);
    if unchanged {
        builder = builder.status(StatusCode::NOT_MODIFIED);
        return builder
            .body(Body::empty())
            .map_err(|error| ApiError::internal(format!("thumbnail response failed: {error}")));
    }
    builder
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, bytes.len())
        .body(Body::from(bytes))
        .map_err(|error| ApiError::internal(format!("thumbnail response failed: {error}")))
}

/// Serve a cached animated GIF preview for a gallery video output.
///
/// Looks up `<preview_dir>/<filename>.preview.gif` (default:
/// `~/.mold/cache/previews/`). When present, streams the file back as
/// `image/gif`; otherwise returns 404. This exists so the TUI's remote
/// gallery detail pane can animate video entries the same way it animates
/// local ones — previously it fell through to fetching the raw MP4 over
/// `/api/gallery/image/:filename`, which `image::open` couldn't decode,
/// leaving the panel on `Loading…` forever.
async fn get_gallery_preview(
    State(state): State<AppState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<axum::response::Response, ApiError> {
    let _gallery_reader = state.gallery_publication_gate.read().await;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);

    // Sanitize: prevent directory traversal — the filename must be a bare
    // basename with no separators.
    let clean_name = std::path::Path::new(&filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    if clean_name.is_empty() || clean_name != filename {
        return Err(ApiError::validation("invalid filename"));
    }

    // The preview cache lifecycle is tied to the underlying gallery file:
    // if the MP4 has been deleted (via `DELETE /api/gallery/image/:filename`
    // or an out-of-band `rm`), the sidecar may still be on disk but is
    // orphaned and must not be served.
    // Check the source file first and 404 before touching the cache so a
    // stale `.preview.gif` never leaks deleted content.
    let source_path = crate::gallery_trash::resolve_gallery_media_source(&output_dir, &clean_name);
    if !tokio::fs::metadata(&source_path)
        .await
        .map(|m| m.is_file())
        .unwrap_or(false)
    {
        return Err(ApiError::not_found(format!(
            "image not found: {clean_name}"
        )));
    }

    let preview_path =
        server_preview_gif_dir().join(mold_core::media_paths::preview_gif_filename(&clean_name));
    let meta = match tokio::fs::metadata(&preview_path).await {
        Ok(m) if m.is_file() => m,
        _ => {
            return Err(ApiError::not_found(format!(
                "preview not found: {clean_name}"
            )));
        }
    };
    let total_len = meta.len();

    let file = tokio::fs::File::open(&preview_path)
        .await
        .map_err(|e| ApiError::internal(format!("failed to open preview: {e}")))?;
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/gif")
        .header(header::CONTENT_LENGTH, total_len)
        .header(header::CACHE_CONTROL, "public, max-age=3600")
        .body(body)
        .unwrap())
}

/// Server-side GIF preview cache directory. Mirrors the layout the TUI
/// writes to (`crates/mold-tui/src/thumbnails.rs::preview_dir`) so a
/// single preview.gif authored on either side is reachable via this
/// endpoint.
pub(crate) fn server_preview_gif_dir() -> std::path::PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("previews")
}

/// Server-side thumbnail cache directory.
pub(crate) fn server_thumbnail_dir() -> std::path::PathBuf {
    crate::thumbnails::server_thumbnail_dir()
}

/// Write one rendered tile atomically (temp + rename) so a concurrent reader
/// never sees a half-written PNG.
fn write_thumbnail_atomically(
    dest: &std::path::Path,
    rendered: &crate::thumbnails::RenderedThumbnail,
) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&tmp, &rendered.bytes)?;
    if let Err(error) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error.into());
    }
    Ok(())
}

/// Generate a 256x256 max thumbnail from source image. The result is always
/// written as a PNG regardless of the source format, so callers should pass
/// a `.png`-suffixed `dest` to keep the on-disk cache unambiguous.
fn generate_server_thumbnail(
    source: &std::path::Path,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    let rendered = crate::thumbnails::render_raster_thumbnail(
        source,
        crate::thumbnails::DEFAULT_MAX_DIM,
        crate::thumbnails::ThumbFormat::Png,
    )?;
    write_thumbnail_atomically(dest, &rendered)
}

/// Extract the first frame of an MP4 (and only the first frame — see
/// `mold_inference::ltx2::media::extract_first_frame`) and downscale it to
/// a 256px max PNG.
fn generate_video_thumbnail(
    source: &std::path::Path,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    let rendered = crate::thumbnails::render_video_thumbnail(
        source,
        crate::thumbnails::DEFAULT_MAX_DIM,
        crate::thumbnails::ThumbFormat::Png,
    )?;
    write_thumbnail_atomically(dest, &rendered)
}

/// Pre-generate thumbnails for all gallery images on server startup.
///
/// The directory walk keeps its per-entry publication-gate contract (one
/// read authority per observation, so a publisher can only run between
/// entries), but the DECODES no longer happen inside it: misses are
/// collected newest-first and rendered on a bounded thread pool, each job
/// taking its own read guard. A 1 000-print gallery used to warm serially
/// on one core.
fn warm_gallery_thumbnails(
    output_dir: &std::path::Path,
    thumb_dir: &std::path::Path,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
    after_acquire: &dyn Fn(usize),
    after_release: &dyn Fn(usize),
) {
    let misses = collect_thumbnail_misses(
        output_dir,
        thumb_dir,
        gallery_gate,
        after_acquire,
        after_release,
    );
    if misses.is_empty() {
        return;
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 4);
    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("mold-thumb-warm-{i}"))
        .build()
    {
        Ok(pool) => pool,
        Err(error) => {
            tracing::warn!(%error, "thumbnail warmup pool unavailable; rendering serially");
            for miss in &misses {
                render_warmup_miss(miss, gallery_gate);
            }
            return;
        }
    };
    pool.install(|| {
        use rayon::prelude::*;
        misses
            .par_iter()
            .for_each(|miss| render_warmup_miss(miss, gallery_gate));
    });
}

struct ThumbnailMiss {
    source: std::path::PathBuf,
    filename: String,
    dest: std::path::PathBuf,
    is_video: bool,
    modified: Option<std::time::SystemTime>,
}

fn render_warmup_miss(
    miss: &ThumbnailMiss,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) {
    let _gallery_reader = gallery_gate.blocking_read();
    if miss.dest.is_file() || !miss.source.is_file() {
        return;
    }
    let result = if miss.is_video {
        generate_video_thumbnail(&miss.source, &miss.dest)
    } else {
        generate_server_thumbnail(&miss.source, &miss.dest)
    };
    if let Err(e) = result {
        tracing::warn!(
            "failed to generate thumbnail for {}: {e}",
            miss.source.display()
        );
    }
}

/// The walk half of warmup: which prints lack a default tile, newest first
/// (the ones a Library opens on), observed one entry per read guard.
fn collect_thumbnail_misses(
    output_dir: &std::path::Path,
    thumb_dir: &std::path::Path,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
    after_acquire: &dyn Fn(usize),
    after_release: &dyn Fn(usize),
) -> Vec<ThumbnailMiss> {
    let mut misses = Vec::new();
    let mut walker = walkdir::WalkDir::new(output_dir).max_depth(1).into_iter();
    let mut observation = 0_usize;
    loop {
        // One read authority owns the full observation: directory iterator
        // advance, entry metadata, format classification, and source decode.
        // A publisher can therefore run only before or after one entry, never
        // in the middle of deciding what that entry contains.
        let gallery_reader = gallery_gate.blocking_read();
        after_acquire(observation);
        let entry = walker.next();
        let done = entry.is_none();
        if let Some(Ok(entry)) = entry {
            let path = entry.path();
            if path.is_file() {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase());
                let is_raster = matches!(
                    ext.as_deref(),
                    Some("png" | "jpg" | "jpeg" | "gif" | "apng" | "webp")
                );
                let is_video = matches!(ext.as_deref(), Some("mp4"));
                // `.wav` is deliberately absent: its thumbnail is the waveform
                // PNG written at save time, and there is nothing in the audio
                // bytes for either decoder to render.
                if is_raster || is_video {
                    let filename = path
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let metadata = entry.metadata().ok();
                    let thumb_path = metadata
                        .as_ref()
                        .map(|metadata| {
                            versioned_thumbnail_path(
                                thumb_dir,
                                &filename,
                                &file_media_version(metadata),
                            )
                        })
                        .unwrap_or_else(|| thumb_dir.join(format!("{filename}.png")));
                    if !thumb_path.is_file() {
                        misses.push(ThumbnailMiss {
                            source: path.to_path_buf(),
                            filename,
                            dest: thumb_path,
                            is_video,
                            modified: metadata.and_then(|m| m.modified().ok()),
                        });
                    }
                }
            }
        }
        drop(gallery_reader);
        after_release(observation);
        observation += 1;
        if done {
            break;
        }
    }
    misses.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.filename.cmp(&b.filename))
    });
    misses
}

pub fn spawn_thumbnail_warmup(
    config: &mold_core::Config,
    gallery_gate: crate::batch_transaction::GalleryPublicationGate,
) -> Option<tokio::task::JoinHandle<()>> {
    if !thumbnail_warmup_enabled() {
        tracing::info!("thumbnail warmup disabled; thumbnails will be generated on demand");
        return None;
    }

    let output_dir = config.effective_output_dir();
    Some(tokio::spawn(async move {
        let join = tokio::task::spawn_blocking(move || {
            let thumb_dir = server_thumbnail_dir();
            warm_gallery_thumbnails(&output_dir, &thumb_dir, &gallery_gate, &|_| {}, &|_| {});
            // Tiles of purged or re-rendered prints can only be identified
            // against what is on disk now, so the sweep rides the warmup —
            // under the publication read gate, so a restore moving a print
            // between `.trash/` and the live dir cannot slip between the two
            // directory walks and have its valid tiles swept.
            let _gallery_reader = gallery_gate.blocking_read();
            match crate::thumbnails::sweep_orphans(
                &output_dir,
                &thumb_dir,
                std::time::Duration::from_secs(24 * 60 * 60),
            ) {
                Ok(removed) if removed > 0 => {
                    tracing::info!(removed, "swept orphaned gallery thumbnails")
                }
                Ok(_) => {}
                Err(error) => tracing::debug!(%error, "thumbnail orphan sweep skipped"),
            }
        })
        .await;
        if let Err(error) = join {
            tracing::warn!(%error, "thumbnail warmup task failed");
        }
        tracing::info!("thumbnail warmup complete");
    }))
}

fn thumbnail_warmup_enabled() -> bool {
    std::env::var("MOLD_THUMBNAIL_WARMUP")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

/// Scan a directory for gallery outputs (images + videos).
///
/// Picks up every format `OutputFormat` can emit: png / jpg / jpeg / gif /
/// apng / webp / mp4. For files with no embedded `mold:parameters` chunk
/// (notably gif / webp / mp4), we synthesize a stub `OutputMetadata` from
/// the filename so the UI can still display them alongside annotated items.
///
/// Invalid files are filtered out at scan time rather than surfaced as
/// broken tiles in the UI. "Invalid" here means any of:
/// - below a format-specific size floor (tiny stubs left by abandoned
///   writes, aborted generations, or test harnesses)
/// - no decodable image header (raster formats)
/// - no `ftyp` box at the start of the file (mp4)
///
/// This is a header-only validation, not a full pixel decode, so a file
/// that passes the check can still be corrupt mid-stream (e.g. broken
/// IDAT CRC). Those fall through to the thumbnail endpoint which serves
/// the raw bytes as a last resort.
#[cfg(test)]
fn scan_gallery_dir(dir: &std::path::Path) -> Vec<mold_core::GalleryImage> {
    scan_gallery_dir_with_archive(
        dir,
        &crate::batch_transaction::CommittedArchiveIndex::default(),
    )
}

fn scan_gallery_dir_with_archive(
    dir: &std::path::Path,
    archive_index: &crate::batch_transaction::CommittedArchiveIndex,
) -> Vec<mold_core::GalleryImage> {
    let mut images: Vec<mold_core::GalleryImage> = mold_db::scan::scan_output_dir(dir)
        .filter_map(|item| match item {
            mold_db::scan::ScanItem::Valid(file) => Some(file),
            _ => None,
        })
        .filter_map(|file| {
            if archive_index.is_retired(&file.filename) {
                return None;
            }
            if let Some(entry) = archive_index.get(&file.filename) {
                return Some(entry.record().to_gallery_image());
            }
            let timestamp = file.timestamp_secs();
            let size_bytes = file.size_u64();
            let (metadata, synthetic) = mold_db::metadata_io::read_or_synthesize(
                &file.path,
                file.format,
                &file.filename,
                timestamp,
            );
            Some(mold_core::GalleryImage {
                filename: file.filename,
                metadata,
                timestamp,
                format: Some(file.format),
                size_bytes: Some(size_bytes),
                media_version: Some(format!("{timestamp}:{size_bytes}")),
                metadata_synthetic: synthetic,
                title: None,
                tags: Vec::new(),
                favorite: false,
                collections: Vec::new(),
                trashed_at: None,
                purge_at: None,
            })
        })
        .collect();

    images.sort_by_key(|img| std::cmp::Reverse(img.timestamp));
    images
}

// ── /api/config/model/:name/placement (Agent C, model-ui-overhaul §3) ────────

/// Read the saved per-model placement default so an editor can hydrate its
/// controls before letting the user edit-and-save (without which a save
/// silently clobbers the persisted placement with defaults). Returns the raw
/// persisted value — not the env-overlaid `resolved_placement` — so a `404`
/// faithfully means "nothing saved for this model".
async fn get_model_placement(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<mold_core::types::DevicePlacement>, ApiError> {
    let cfg = state.config.read().await;
    match cfg.models.get(&name).and_then(|mc| mc.placement.clone()) {
        Some(placement) => Ok(Json(placement)),
        None => Err(ApiError::not_found(format!(
            "no placement saved for model '{name}'"
        ))),
    }
}

async fn put_model_placement(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(placement): Json<mold_core::types::DevicePlacement>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_multi_gpu_placement(&state, Some(&placement))?;
    {
        let mut cfg = state.config.write().await;
        cfg.set_model_placement(&name, Some(placement.clone()));
        cfg.save().map_err(|e| {
            tracing::warn!("failed to persist placement to config.toml: {e}");
            ApiError::internal(format!("failed to persist placement to config.toml: {e}"))
        })?;
    }
    Ok(Json(serde_json::json!({
        "ok": true,
        "model": name,
    })))
}

async fn delete_model_placement(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut cfg = state.config.write().await;
    cfg.set_model_placement(&name, None);
    cfg.save().map_err(|e| {
        tracing::warn!("failed to persist placement removal to config.toml: {e}");
        ApiError::internal(format!(
            "failed to persist placement removal to config.toml: {e}"
        ))
    })?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

// ── /api/openapi.json ─────────────────────────────────────────────────────────

async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

// ── /api/docs ─────────────────────────────────────────────────────────────────

async fn scalar_docs() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html")],
        r#"<!DOCTYPE html>
<html>
<head>
  <title>mold API</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <script id="api-reference" data-url="/api/openapi.json"></script>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>"#,
    )
}

// ─── Third-party model licenses ──────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/licenses",
    tag = "models",
    responses((
        status = 200,
        description = "Known third-party model licenses and this server's acceptance state",
        body = mold_core::LicenseListing,
    )),
)]
/// List every third-party model license and whether THIS server has accepted it.
///
/// Acceptance is per Mold data root, so the answer belongs to the host that
/// served it — a multi-host client must ask each one rather than caching a
/// fleet-wide verdict.
async fn list_licenses_endpoint() -> Result<Json<mold_core::LicenseListing>, ApiError> {
    let mold_home = mold_core::Config::mold_dir()
        .ok_or_else(|| ApiError::internal("could not resolve the Mold data directory"))?;
    Ok(Json(mold_core::LicenseListing {
        licenses: mold_core::license_acceptance::license_statuses(&mold_home),
    }))
}

// ─── Downloads UI (Agent A) ──────────────────────────────────────────────────

#[derive(serde::Deserialize, utoipa::ToSchema)]
pub struct CreateDownloadBody {
    pub model: String,
    /// Third-party licenses the user has accepted, each carrying the exact
    /// terms they were shown, recorded in this server's Mold data root before
    /// the pull starts.
    ///
    /// Additive: absent means "accept nothing", which is what every existing
    /// client sends and leaves their behaviour unchanged.
    #[serde(default)]
    pub accept_licenses: Vec<mold_core::LicenseAcceptance>,
}

#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct CreateDownloadResponse {
    pub id: String,
    pub position: usize,
}

#[utoipa::path(
    post,
    path = "/api/downloads",
    tag = "downloads",
    request_body = CreateDownloadBody,
    responses(
        (status = 200, description = "Enqueued; position 0 = will start immediately", body = CreateDownloadResponse),
        (status = 400, description = "Unknown model"),
        (status = 409, description = "Already active or queued; body contains existing id", body = CreateDownloadResponse),
    )
)]
pub async fn create_download(
    State(state): State<AppState>,
    Json(body): Json<CreateDownloadBody>,
) -> axum::response::Response {
    use crate::downloads::{EnqueueError, EnqueueOutcome};
    if let Err(error) = require_server_model_acquisition(&state, &body.model).await {
        return error.into_response();
    }
    if let Err(error) =
        apply_download_license_acceptances(&state, &body.model, &body.accept_licenses).await
    {
        return error.into_response();
    }
    match state.downloads.enqueue(body.model.clone()).await {
        Ok((id, position, EnqueueOutcome::Created)) => (
            StatusCode::OK,
            Json(CreateDownloadResponse { id, position }),
        )
            .into_response(),
        Ok((id, position, EnqueueOutcome::AlreadyPresent)) => (
            StatusCode::CONFLICT,
            Json(CreateDownloadResponse { id, position }),
        )
            .into_response(),
        Err(EnqueueError::ModelActivation(error)) => {
            ApiError::model_activation(error).into_response()
        }
        Err(EnqueueError::UnknownModel(_)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("unknown model '{}'. Run 'mold list' to see available models.", body.model)
            })),
        )
            .into_response(),
        Err(EnqueueError::LockPoisoned) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "download queue state is corrupt" })),
        )
            .into_response(),
    }
}

#[utoipa::path(
    delete,
    path = "/api/downloads/{id}",
    tag = "downloads",
    params(("id" = String, Path, description = "Job id")),
    responses(
        (status = 204, description = "Cancelled"),
        (status = 404, description = "Unknown id"),
    )
)]
pub async fn delete_download(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if state.downloads.cancel(&id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown download id '{id}'") })),
        )
            .into_response()
    }
}

#[utoipa::path(
    get,
    path = "/api/downloads",
    tag = "downloads",
    responses((status = 200, description = "Current queue state"))
)]
pub async fn list_downloads(State(state): State<AppState>) -> axum::response::Response {
    Json(state.downloads.listing().await).into_response()
}

#[utoipa::path(
    get,
    path = "/api/downloads/stream",
    tag = "downloads",
    responses((status = 200, description = "SSE stream of DownloadEvent JSON")),
)]
pub async fn stream_downloads(
    State(state): State<AppState>,
) -> Sse<
    impl futures_core::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse::Event;
    use tokio_stream::wrappers::BroadcastStream;
    use tokio_stream::StreamExt as _;

    // Subscribe BEFORE snapshotting so any event arriving during the
    // snapshot read is queued in the broadcast channel instead of being
    // missed. The first frame we yield is `Snapshot { listing }` —
    // mirrors `/api/resources/stream`'s initial-snapshot pattern so a
    // freshly-mounted SPA paints current state without waiting for the
    // next delta.
    let rx = state.downloads.subscribe();
    let shutdown = state.events.shutdown_token();
    let initial = state.downloads.listing().await;
    let snapshot_event = mold_core::types::DownloadEvent::Snapshot { listing: initial };

    let stream = async_stream::stream! {
        if shutdown.is_cancelled() {
            return;
        }
        let data = serde_json::to_string(&snapshot_event).unwrap_or_else(|_| "{}".to_string());
        yield Ok::<_, std::convert::Infallible>(Event::default().event("download").data(data));

        let mut bs = BroadcastStream::new(rx);
        loop {
            let item = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                item = bs.next() => item,
            };
            let Some(item) = item else { break };
            match item {
                Ok(event) => {
                    let data = serde_json::to_string(&event)
                        .unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("download").data(data));
                }
                // Slow subscribers see lag silently; the snapshot above
                // already carries the full state so we don't need to
                // resync on every drop.
                Err(_lagged) => continue,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

// ── Resource telemetry (Agent B scope) ───────────────────────────────────────

/// `GET /api/resources` — one-shot JSON snapshot from the aggregator cache.
/// Returns 503 if the aggregator has not yet fired (first 1 s after startup
/// and before `spawn_aggregator` has run).
async fn get_resources(State(state): State<AppState>) -> Result<Json<ResourceSnapshot>, ApiError> {
    match state.resources.latest() {
        Some(snap) => Ok(Json(snap)),
        None => Err(ApiError::internal_with_status(
            "resource telemetry not ready",
            StatusCode::SERVICE_UNAVAILABLE,
        )),
    }
}

/// `GET /api/resources/stream` — SSE stream of `ResourceSnapshot` frames.
/// Event name: `snapshot`. Matches the keepalive cadence of `/api/generate/stream`.
async fn get_resources_stream(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>> {
    use tokio_stream::wrappers::BroadcastStream;

    let rx = state.resources.subscribe();
    let shutdown = state.events.shutdown_token();
    // Attach the cached `latest` snapshot as the first frame so clients
    // don't wait up to one full tick for their initial value.
    let initial = state.resources.latest();

    let stream = async_stream::stream! {
        if shutdown.is_cancelled() {
            return;
        }
        if let Some(snap) = initial {
            yield Ok::<_, Infallible>(snapshot_to_sse(&snap));
        }
        let mut bs = BroadcastStream::new(rx);
        loop {
            let item = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                item = bs.next() => item,
            };
            let Some(item) = item else { break };
            match item {
                Ok(snap) => yield Ok(snapshot_to_sse(&snap)),
                // Lag is normal for slow clients — skip dropped frames
                // silently; the next one will catch them up.
                Err(_lagged) => continue,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

fn snapshot_to_sse(snap: &ResourceSnapshot) -> SseEvent {
    match serde_json::to_string(snap) {
        Ok(data) => SseEvent::default().event("snapshot").data(data),
        Err(e) => SseEvent::default()
            .event("error")
            .data(format!("{{\"message\":\"serialize failed: {e}\"}}")),
    }
}

/// `GET /api/events` — SSE stream of server-wide [`mold_core::ServerEvent`]s:
/// job lifecycle, gallery mutations, queue replans, and semantic device
/// lifecycle/health transitions. One connection observes the whole server.
/// The stream opens with an `authority` frame carrying this server's stable
/// `instance_id`. Lifecycle deltas retain their existing `event` event name
/// and exact [`mold_core::ServerEvent`] JSON shape. If the bounded broadcast
/// buffer overruns a slow receiver, `resync_required` reports the gap instead
/// of pretending the delta stream is complete; repair from `GET /api/queue`,
/// `GET /api/devices`, and `GET /api/gallery`. Raw utilization/memory telemetry
/// remains on `/api/resources/stream`. Feature-detect via
/// `capabilities.events.available`.
#[utoipa::path(
    get,
    path = "/api/events",
    tag = "server",
    responses(
        (status = 200, description = "SSE stream of server lifecycle events", content_type = "text/event-stream")
    )
)]
async fn stream_events(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>> {
    use tokio_stream::wrappers::BroadcastStream;

    let rx = state.events.subscribe();
    let shutdown = state.events.shutdown_token();
    let instance_id = state.instance_id.clone();
    let stream = async_stream::stream! {
        if shutdown.is_cancelled() {
            return;
        }
        yield Ok::<_, Infallible>(event_authority_to_sse(instance_id.as_str()));
        let mut bs = BroadcastStream::new(rx);
        loop {
            let item = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                item = bs.next() => item,
            };
            let Some(item) = item else { break };
            match crate::events::classify_delivery(item) {
                crate::events::BroadcastDelivery::Event(event) => {
                    yield Ok::<_, Infallible>(server_event_to_sse(&event));
                }
                crate::events::BroadcastDelivery::ResyncRequired { missed_events } => {
                    yield Ok::<_, Infallible>(event_resync_to_sse(
                        instance_id.as_str(),
                        missed_events,
                    ));
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

#[derive(Serialize)]
struct EventStreamAuthority<'a> {
    instance_id: &'a str,
}

#[derive(Serialize)]
struct EventStreamResync<'a> {
    instance_id: &'a str,
    missed_events: u64,
}

const EVENT_STREAM_RESYNC_NAME: &str = "resync_required";

fn event_authority_to_sse(instance_id: &str) -> SseEvent {
    SseEvent::default().event("authority").data(
        serde_json::to_string(&EventStreamAuthority { instance_id })
            .expect("authority frame serialization cannot fail"),
    )
}

fn event_resync_to_sse(instance_id: &str, missed_events: u64) -> SseEvent {
    SseEvent::default()
        .event(EVENT_STREAM_RESYNC_NAME)
        .data(event_resync_data(instance_id, missed_events))
}

fn event_resync_data(instance_id: &str, missed_events: u64) -> String {
    serde_json::to_string(&EventStreamResync {
        instance_id,
        missed_events,
    })
    .expect("resync frame serialization cannot fail")
}

fn server_event_to_sse(ev: &mold_core::ServerEvent) -> SseEvent {
    match serde_json::to_string(ev) {
        Ok(data) => SseEvent::default().event("event").data(data),
        // json! escapes the error text — quotes/newlines in `e` must not
        // produce an invalid JSON frame.
        Err(e) => SseEvent::default()
            .event("error")
            .data(serde_json::json!({ "message": format!("serialize failed: {e}") }).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    #[test]
    fn event_stream_gap_has_explicit_resync_wire_contract() {
        assert_eq!(EVENT_STREAM_RESYNC_NAME, "resync_required");
        assert_eq!(
            event_resync_data("host-123", 17),
            r#"{"instance_id":"host-123","missed_events":17}"#
        );
    }

    async fn response_json(response: axum::response::Response) -> serde_json::Value {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queue_sql_work_runs_off_the_async_runtime_thread() {
        let runtime_thread = std::thread::current().id();
        let blocking_thread = spawn_queue_read(|| Ok(std::thread::current().id()))
            .await
            .unwrap();
        let mutation_thread = spawn_queue_mutation(|| Ok(std::thread::current().id()))
            .await
            .unwrap();
        assert_ne!(
            blocking_thread, runtime_thread,
            "SQLite queue reads must execute on Tokio's blocking pool"
        );
        assert_ne!(
            mutation_thread, runtime_thread,
            "SQLite queue mutations must execute on Tokio's blocking pool"
        );
    }

    #[test]
    fn h3_models_do_not_publish_a_family_wide_access_restriction() {
        let access = mold_core::model_access_capabilities();
        assert!(access.restrictions.is_empty());
    }

    #[test]
    fn reviewed_h3_manifest_executes_without_authorizing_raw_repository_ids() {
        let manifest =
            mold_core::manifest::find_manifest(mold_core::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P)
                .expect("reviewed H3 Turbo manifest");
        // The raw repository identity is never authorized, whatever this
        // build's runtime answer is; the reviewed manifest's own activation
        // is the build's question and is asserted as such (#1276).
        assert_eq!(
            mold_core::require_registered_manifest_activation(manifest).is_ok(),
            mold_core::minimax_h3::engine_is_built()
        );
        assert!(mold_core::require_model_activation(
            "hf:Comfy-Org/MiniMax-H3",
            Some(mold_core::minimax_h3::FAMILY),
        )
        .is_err());
    }
    /// #787: an absent negative on a wan request materializes the tuned
    /// default so queue/worker metadata record the real uncond; an explicit
    /// value — the empty-string opt-out above all — passes through untouched,
    /// and families without an engine fallback stay absent.
    #[test]
    fn materialize_default_negative_prompt_fills_wan_absence_only() {
        let request = || -> mold_core::GenerateRequest {
            serde_json::from_value(serde_json::json!({
                "prompt": "a cat",
                "model": "wan22-t2v-a14b:q5",
                "width": 832,
                "height": 480,
                "steps": 20,
                "guidance": 3.5,
                "batch_size": 1,
                "strength": 0.75
            }))
            .unwrap()
        };

        let mut absent = request();
        materialize_default_negative_prompt(&mut absent, Some("wan"));
        assert_eq!(
            absent.negative_prompt.as_deref(),
            Some(mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT)
        );

        let mut opted_out = request();
        opted_out.negative_prompt = Some(String::new());
        materialize_default_negative_prompt(&mut opted_out, Some("wan"));
        assert_eq!(opted_out.negative_prompt.as_deref(), Some(""));

        let mut explicit = request();
        explicit.negative_prompt = Some("blurry".into());
        materialize_default_negative_prompt(&mut explicit, Some("wan"));
        assert_eq!(explicit.negative_prompt.as_deref(), Some("blurry"));

        let mut other_family = request();
        materialize_default_negative_prompt(&mut other_family, Some("flux"));
        assert_eq!(other_family.negative_prompt, None);

        let mut unknown_family = request();
        materialize_default_negative_prompt(&mut unknown_family, None);
        assert_eq!(unknown_family.negative_prompt, None);
    }

    /// The materialized request is what `OutputMetadata::from_generate_request`
    /// captures everywhere (queue, workers, batch runtime), so this pins the
    /// truthful-provenance half of #787's acceptance criteria.
    #[test]
    fn materialized_wan_negative_reaches_output_metadata() {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "wan22-t2v-a14b:q5",
            "width": 832,
            "height": 480,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1,
            "strength": 0.75
        }))
        .unwrap();
        materialize_default_negative_prompt(&mut request, Some("wan"));
        let metadata = mold_core::OutputMetadata::from_generate_request(&request, 7, None, "test");
        assert_eq!(
            metadata.negative_prompt.as_deref(),
            Some(mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT)
        );
    }

    fn wan_continuation(model: &str) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a cat keeps walking",
            "model": model,
            "width": 832,
            "height": 480,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1,
            "strength": 0.75,
            "frames": 49,
            "fps": 16,
            "output_format": "mp4",
            "extend_video_path": "/srv/mold/clip.mp4"
        }))
        .unwrap()
    }

    /// Admission's source-image contract gate has to count an extend as
    /// carrying source frames (#783).
    ///
    /// `validate_extend` forbids pairing `extend_video` with `source_image`
    /// or keyframes, so a continuation provably has neither — and a gate that
    /// counted only those refused every Wan I2V extend with "this Wan I2V
    /// checkpoint needs a source image", the exact contract that makes the
    /// checkpoint extend-capable. This drives the real admission helper, so
    /// restoring the inline `source_image.is_some() || keyframes` expression
    /// fails it.
    #[tokio::test]
    async fn admission_reads_a_wan_continuation_as_carrying_source_frames() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::for_tests();
        // No checkpoints on disk: the header probe finds nothing and the
        // manifest's own task classification binds, exactly as it does for a
        // cold tier.
        state.config.write().await.models_dir = temp.path().display().to_string();

        enforce_source_image_capability(
            &state,
            &wan_continuation("wan22-i2v-a14b:q8"),
            Some("wan"),
        )
        .await
        .expect("a Required-source checkpoint is satisfied by the clip being continued");

        // …and the same predicate still refuses a text-to-video checkpoint at
        // admission, instead of letting it die in the engine after the UMT5
        // encode and expert load are paid for.
        let refused = enforce_source_image_capability(
            &state,
            &wan_continuation("wan22-t2v-a14b:q8"),
            Some("wan"),
        )
        .await
        .expect_err("a text-to-video checkpoint cannot accept a continuation");
        assert!(
            refused.error.contains("text-to-video only"),
            "got: {}",
            refused.error
        );

        // The pre-existing carriers are untouched: an ordinary text-to-video
        // render still reads as source-less and is admitted.
        let mut plain = wan_continuation("wan22-t2v-a14b:q8");
        plain.extend_video_path = None;
        enforce_source_image_capability(&state, &plain, Some("wan"))
            .await
            .expect("an ordinary text-to-video render carries no source frames");
    }

    /// The overlap admission materializes is what saved provenance records.
    ///
    /// An installed `cv:` / `hf:` wan checkpoint has no manifest, so metadata
    /// built from an unmaterialized request resolved LTX-2's 17 for a render
    /// that used one frame (#783).
    #[test]
    fn materialized_extend_overlap_reaches_output_metadata() {
        let mut request = wan_continuation("cv:2041121");
        assert_eq!(request.extend_overlap_frames, None);
        mold_core::validation::materialize_extend_overlap_frames(&mut request, Some("wan"));
        let metadata = mold_core::OutputMetadata::from_generate_request(&request, 7, None, "test");
        assert_eq!(
            metadata.extend_overlap_frames,
            Some(mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES)
        );
    }

    /// …and admission is where that materialization actually happens.
    ///
    /// The test above only proves the helper works; this one drives
    /// `prepare_generation` itself, which is the single server-side seam
    /// between a client that named no overlap and every downstream consumer
    /// (queue, worker, saved metadata). Admission still fails afterwards — no
    /// checkpoint is installed — but the mutation is already on the request by
    /// then, so deleting the call at the seam fails this test.
    ///
    /// Both families run through it because the seam has to pass the resolved
    /// family through: a call that hardcoded either constant would satisfy one
    /// assertion and break the other.
    #[tokio::test]
    async fn admission_materializes_the_resolved_familys_extend_carryover() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::for_tests();
        state.config.write().await.models_dir = temp.path().display().to_string();

        // Inline bytes rather than a server-local path: the path form is
        // resolved against the media roots later in admission, and this test
        // is about the seam, not about media resolution.
        let mut wan = wan_continuation("wan22-i2v-a14b:q8");
        wan.extend_video_path = None;
        wan.extend_video = Some(b"\0\0\0\x20ftypisom".to_vec());
        assert_eq!(wan.extend_overlap_frames, None);
        let _ = prepare_generation_inner(&state, &mut wan, None, None).await;
        assert_eq!(
            wan.extend_overlap_frames,
            Some(mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES),
            "admission must write wan's own one-frame carryover into the request"
        );

        let mut ltx2 = wan_continuation("ltx-2-19b-dev:fp8");
        ltx2.extend_video_path = None;
        ltx2.extend_video = Some(b"\0\0\0\x20ftypisom".to_vec());
        assert_eq!(ltx2.extend_overlap_frames, None);
        let _ = prepare_generation_inner(&state, &mut ltx2, None, None).await;
        assert_eq!(
            ltx2.extend_overlap_frames,
            Some(mold_core::validation::DEFAULT_EXTEND_OVERLAP_FRAMES),
            "the same seam must resolve LTX-2's 17 from the resolved family"
        );

        // An explicit overlap is authoritative, and an ordinary render is
        // never handed one — so the seam is a fill-in, not a rewrite.
        let mut explicit = wan_continuation("wan22-i2v-a14b:q8");
        explicit.extend_video_path = None;
        explicit.extend_video = Some(b"\0\0\0\x20ftypisom".to_vec());
        explicit.extend_overlap_frames = Some(5);
        let _ = prepare_generation_inner(&state, &mut explicit, None, None).await;
        assert_eq!(explicit.extend_overlap_frames, Some(5));

        let mut plain = wan_continuation("wan22-t2v-a14b:q8");
        plain.extend_video_path = None;
        let _ = prepare_generation_inner(&state, &mut plain, None, None).await;
        assert_eq!(plain.extend_overlap_frames, None);
    }

    #[tokio::test]
    async fn production_pairing_handlers_issue_claim_and_reject_replay() {
        let mut state = AppState::for_tests();
        let metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        state.metadata_db = metadata_db.clone();
        let key_set = Arc::new(crate::auth::ApiKeySet::new_with_metadata_db(
            std::collections::HashSet::from(["phone-key".to_string()]),
            metadata_db,
            state.instance_id.as_ref().clone(),
        ));
        let auth_state = Some(key_set.clone());
        let created = create_pairing_session(
            State(state.clone()),
            Some(Extension(auth_state.clone())),
            Some(Extension(crate::auth::PairingAuthority)),
        )
        .await
        .unwrap()
        .into_response();
        assert_eq!(created.status(), StatusCode::OK);
        assert_eq!(
            created.headers().get(header::CACHE_CONTROL).unwrap(),
            "no-store"
        );
        let created = response_json(created).await;
        assert_eq!(created["auth_required"], true);
        assert_eq!(created["instance_id"], *state.instance_id);
        let token = created["token"].as_str().unwrap().to_string();

        let claimed = claim_pairing_session(
            State(state.clone()),
            Some(Extension(auth_state.clone())),
            Json(PairingClaimRequest {
                token: Some(token.clone()),
                client_name: Some("Test iPhone".into()),
                client_kind: Some("iphone".into()),
            }),
        )
        .await
        .unwrap()
        .into_response();
        assert_eq!(claimed.status(), StatusCode::OK);
        assert_eq!(
            claimed.headers().get(header::CACHE_CONTROL).unwrap(),
            "no-store"
        );
        let claimed = response_json(claimed).await;
        let paired_key = claimed["api_key"].as_str().unwrap().to_string();
        assert!(paired_key.starts_with("mold_pair_"));
        assert_ne!(claimed["api_key"], "phone-key");
        assert_eq!(claimed["instance_id"], *state.instance_id);
        assert!(key_set.contains(&paired_key));

        let clients = list_paired_clients(
            Some(Extension(auth_state.clone())),
            Some(Extension(crate::auth::PairingAuthority)),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(clients.clients.len(), 1);
        assert_eq!(clients.clients[0].name, "Test iPhone");
        assert_eq!(clients.clients[0].client_kind, "iphone");
        revoke_paired_client(
            Path(clients.clients[0].id.clone()),
            Some(Extension(auth_state.clone())),
            Some(Extension(crate::auth::PairingAuthority)),
        )
        .await
        .unwrap();
        assert!(!key_set.contains(&paired_key));
        assert!(key_set.contains("phone-key"));

        let replay = match claim_pairing_session(
            State(state),
            Some(Extension(auth_state)),
            Json(PairingClaimRequest {
                token: Some(token),
                client_name: None,
                client_kind: None,
            }),
        )
        .await
        {
            Ok(_) => panic!("a consumed pairing token must not be accepted twice"),
            Err(error) => error.into_response(),
        };
        assert_eq!(replay.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response_json(replay).await["code"], "PAIRING_TOKEN_INVALID");
    }

    #[tokio::test]
    async fn production_pairing_handlers_keep_open_hosts_credential_free() {
        let state = AppState::for_tests();
        let created = create_pairing_session(State(state.clone()), Some(Extension(None)), None)
            .await
            .unwrap()
            .into_response();
        let created = response_json(created).await;
        assert_eq!(created["token"], serde_json::Value::Null);
        assert_eq!(created["auth_required"], false);

        let claimed = claim_pairing_session(
            State(state),
            Some(Extension(None)),
            Json(PairingClaimRequest {
                token: None,
                client_name: None,
                client_kind: None,
            }),
        )
        .await
        .unwrap()
        .into_response();
        let claimed = response_json(claimed).await;
        assert_eq!(claimed["api_key"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn control_planning_is_read_only_and_freezes_the_exact_artifact_path() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::for_tests();
        state.config.write().await.models_dir = temp.path().display().to_string();
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test",
            "model": "ltx-2.3-22b-distilled:fp8",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "source_video_path": "/guide.mp4",
            "ic_lora_control": "MOTION_TRACK"
        }))
        .unwrap();
        assert!(state.downloads.listing().await.active_jobs.is_empty());

        let (adapter, path) = plan_builtin_ltx2_control(&state, &mut request)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(adapter.id, "motion-track");
        assert_eq!(request.ic_lora_control.as_deref(), Some("motion-track"));
        assert_eq!(request.pipeline, Some(mold_core::Ltx2PipelineMode::IcLora));
        let manifest = mold_core::manifest::find_manifest(adapter.download_model).unwrap();
        assert_eq!(
            path,
            temp.path().join(mold_core::manifest::storage_path(
                manifest,
                &manifest.files[0]
            ))
        );
        let listing = state.downloads.listing().await;
        assert!(listing.active_jobs.is_empty());
        assert!(listing.queued.is_empty());
    }

    /// The HDR adapter is the multi-file case: weights plus a companion file
    /// of pre-computed prompt embeddings.
    fn multi_file_adapter() -> &'static mold_core::ltx2_control::Ltx2ControlAdapter {
        mold_core::ltx2_control::LTX2_CONTROL_ADAPTERS
            .iter()
            .find(|adapter| adapter.extra_files.len() == 1)
            .expect("the registry must keep at least one multi-file adapter")
    }

    #[test]
    fn a_present_companion_is_not_reported_as_a_pending_download() {
        let adapter = multi_file_adapter();
        let temp = tempfile::tempdir().unwrap();
        let weights = temp.path().join(adapter.hf_filename);
        for file in adapter.files() {
            let path = temp.path().join(file.hf_filename);
            std::fs::write(&path, b"x").unwrap();
            mold_core::download::write_sha256_marker(&path, file.sha256).unwrap();
        }

        assert!(control_missing_files(adapter, &weights).is_empty());
        assert!(control_artifact_is_complete(adapter, &weights));
    }

    #[test]
    fn a_pending_control_download_reports_only_the_bytes_still_missing() {
        let adapter = multi_file_adapter();
        let companion = adapter.extra_files[0];
        let temp = tempfile::tempdir().unwrap();

        // Weights landed; the companion never did. Reporting the adapter's
        // total, or the weights' filename, would both describe a download
        // that is not the one about to run.
        let weights = temp.path().join(adapter.hf_filename);
        std::fs::write(&weights, b"x").unwrap();
        mold_core::download::write_sha256_marker(&weights, adapter.sha256).unwrap();

        let missing = control_missing_files(adapter, &weights);
        assert_eq!(missing.len(), 1, "only the companion is outstanding");
        assert_eq!(missing[0].hf_filename, companion.hf_filename);
        assert_eq!(
            control_pending_download_bytes(adapter, &weights),
            companion.size_bytes,
            "a half-finished pull must not re-report the bytes already on disk"
        );
        assert!(!control_artifact_is_complete(adapter, &weights));
    }

    /// The caller resolves the weights path through the manifest, so it is
    /// authoritative even when its basename is not the registry's `hf_filename`.
    /// Reconstructing the name instead made a present single-file adapter read
    /// as missing, and `materialize_builtin_ltx2_control` then blocked forever
    /// waiting for a download nobody had queued.
    #[test]
    fn a_resolved_weights_path_counts_even_when_its_name_differs() {
        let single_file = mold_core::ltx2_control::LTX2_CONTROL_ADAPTERS
            .iter()
            .find(|adapter| adapter.extra_files.is_empty())
            .expect("the registry must keep a single-file adapter");
        let temp = tempfile::tempdir().unwrap();
        let weights = temp.path().join("resolved-by-manifest.safetensors");
        std::fs::write(&weights, b"installed").unwrap();
        mold_core::download::write_sha256_marker(&weights, single_file.sha256).unwrap();

        assert!(
            control_missing_files(single_file, &weights).is_empty(),
            "the given weights path is the artifact, not a name to rebuild"
        );
        assert!(control_artifact_is_complete(single_file, &weights));
        assert_eq!(control_pending_download_bytes(single_file, &weights), 0);
    }

    #[test]
    fn control_completeness_is_read_from_the_weights_directory_once() {
        let adapter = multi_file_adapter();
        let temp = tempfile::tempdir().unwrap();
        let weights = temp.path().join(adapter.hf_filename);

        // Nothing on disk: every file is outstanding and the reported size is
        // the adapter total.
        assert_eq!(
            control_missing_files(adapter, &weights).len(),
            adapter.files().count()
        );
        assert_eq!(
            control_pending_download_bytes(adapter, &weights),
            adapter.total_size_bytes()
        );
    }

    /// A dimension header that carries non-dimension text is a header a client
    /// can reasonably discard. Keep the two channels separate.
    #[test]
    fn request_warnings_keep_dimension_advisories_in_their_own_channel() {
        let warnings = super::RequestWarnings {
            dimension: Some("dimensions adjusted from 1000x1000 to 1024x1024".to_string()),
            other: vec!["lip-dub takes its length from the reference video".to_string()],
        };

        // Everything, request-specific first, for the general header.
        assert_eq!(
            warnings.all().collect::<Vec<_>>(),
            vec![
                "lip-dub takes its length from the reference video",
                "dimensions adjusted from 1000x1000 to 1024x1024",
            ]
        );
        // The dimension header still means exactly what its docs say.
        assert_eq!(
            warnings.dimension.as_deref(),
            Some("dimensions adjusted from 1000x1000 to 1024x1024")
        );

        let timing_only = super::RequestWarnings {
            dimension: None,
            other: vec!["lip-dub takes its frame rate from the reference video".to_string()],
        };
        assert!(!timing_only.is_empty());
        assert!(
            timing_only.dimension.is_none(),
            "a timing substitution must not be published as a dimension adjustment"
        );
        assert!(super::RequestWarnings::default().is_empty());
    }

    // ── creation-time filing at admission ───────────────────────────────

    fn filed_request(
        tags: &[&str],
        collection: Option<mold_core::CollectionRef>,
    ) -> mold_core::GenerateRequest {
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q4",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 3.5,
        }))
        .unwrap();
        if !tags.is_empty() {
            request.tags = Some(tags.iter().map(|t| t.to_string()).collect());
        }
        request.collection = collection;
        request
    }

    /// An `{id}` reference becomes the `{id, name}` form publication files
    /// by; a `{name}` reference is left exactly as it arrived, because
    /// publication creates it when absent.
    #[test]
    fn admission_resolves_a_collection_id_to_its_name_and_leaves_a_name_alone() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let created = db.create_collection("Smurf Village", None).unwrap();

        let mut by_id = Some(mold_core::CollectionRef::by_id(created.id.clone()));
        assert!(super::resolve_collection_reference(&db, &mut by_id).is_empty());
        assert_eq!(
            by_id,
            Some(mold_core::CollectionRef {
                id: Some(created.id.clone()),
                name: Some("Smurf Village".to_string()),
            })
        );

        let mut by_name = Some(mold_core::CollectionRef::by_name("Somewhere Else"));
        assert!(super::resolve_collection_reference(&db, &mut by_name).is_empty());
        assert_eq!(
            by_name,
            Some(mold_core::CollectionRef::by_name("Somewhere Else")),
            "a name needs no lookup and must not be rewritten"
        );

        let mut none = None;
        assert!(super::resolve_collection_reference(&db, &mut none).is_empty());
        assert_eq!(none, None);
    }

    /// A collection deleted between the client reading the list and pressing
    /// Generate drops the filing with an advisory — never a refusal, because
    /// the print is the expensive artifact and its filing is not.
    #[test]
    fn admission_drops_an_unknown_collection_id_with_an_advisory() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let mut gone = Some(mold_core::CollectionRef::by_id(
            "11111111-2222-3333-4444-555555555555",
        ));
        let warnings = super::resolve_collection_reference(&db, &mut gone);
        assert_eq!(gone, None, "the filing is dropped, not carried forward");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("no longer exists"), "{}", warnings[0]);
        assert!(
            warnings[0].contains("11111111-2222-3333-4444-555555555555"),
            "the advisory must name the collection: {}",
            warnings[0]
        );
    }

    /// A host with no metadata DB has nowhere to file. The print still
    /// generates; the filing is dropped and said so.
    #[tokio::test]
    async fn a_host_without_a_metadata_db_drops_the_filing_with_a_warning() {
        let mut state = AppState::for_tests();
        state.metadata_db = std::sync::Arc::new(None);

        let mut request = filed_request(
            &["smurfs", "village"],
            Some(mold_core::CollectionRef::by_name("Sequences")),
        );
        let warnings = super::resolve_request_filing(&state, &mut request).await;
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("no metadata database"),
            "{}",
            warnings[0]
        );
        assert!(warnings[0].contains("2 tags"), "{}", warnings[0]);
        assert!(warnings[0].contains("collection"), "{}", warnings[0]);
        assert_eq!(request.tags, None, "the filing is dropped from the request");
        assert_eq!(request.collection, None);

        // An unfiled request on the same host is silent — the advisory must
        // not become boilerplate on every print.
        let mut plain = filed_request(&[], None);
        assert!(super::resolve_request_filing(&state, &mut plain)
            .await
            .is_empty());
    }

    /// Admission rewrites the filing into the form that will actually be
    /// applied, so the queue journal, `OutputMetadata`, and the gallery row
    /// all carry one canonical spelling. A direct HTTP caller is the one who
    /// needs this — every first-party client normalizes before sending.
    #[test]
    fn admission_materializes_the_filing_a_raw_http_caller_sent() {
        let mut request = filed_request(&[], None);
        request.tags = Some(vec![
            "  Smurfs  ".into(),
            "smurfs".into(),
            "".into(),
            " village  green ".into(),
        ]);
        request.collection = Some(mold_core::CollectionRef::by_name("  Smurf   Village  "));

        mold_core::validation::materialize_request_organization(&mut request).unwrap();

        assert_eq!(
            request.tags.as_deref(),
            Some(["Smurfs".to_string(), "village green".to_string()].as_slice())
        );
        assert_eq!(
            request.collection,
            Some(mold_core::CollectionRef::by_name("Smurf Village"))
        );
        // The journal stores the admitted request, so a replay files the same
        // print the original run did.
        let journaled: mold_core::GenerateRequest =
            serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
        assert_eq!(journaled.tags, request.tags);
        assert_eq!(journaled.collection, request.collection);
    }

    /// The advisory names what was dropped, in the singular or the plural,
    /// so the user can tell which part of their filing did not land.
    #[test]
    fn the_filing_advisory_names_what_was_requested() {
        assert_eq!(
            super::describe_request_filing(&filed_request(&[], None)),
            None
        );
        assert_eq!(
            super::describe_request_filing(&filed_request(&["one"], None)).as_deref(),
            Some("the requested tag")
        );
        assert_eq!(
            super::describe_request_filing(&filed_request(&["one", "two"], None)).as_deref(),
            Some("the requested 2 tags")
        );
        assert_eq!(
            super::describe_request_filing(&filed_request(
                &[],
                Some(mold_core::CollectionRef::by_name("Sequences"))
            ))
            .as_deref(),
            Some("the requested collection")
        );
        assert_eq!(
            super::describe_request_filing(&filed_request(
                &["one"],
                Some(mold_core::CollectionRef::by_name("Sequences"))
            ))
            .as_deref(),
            Some("the requested collection and tag")
        );
    }

    /// Advisories ride the general `x-mold-request-warning` header, joined
    /// with `; ` and stripped of newlines so the value stays a legal header.
    #[test]
    fn filing_advisories_become_a_single_request_warning_header() {
        assert!(super::request_warning_headers(&[]).is_empty());
        let headers =
            super::request_warning_headers(&["first\nadvisory".to_string(), "second".to_string()]);
        assert_eq!(
            headers.get("x-mold-request-warning").unwrap(),
            "first advisory; second"
        );
    }

    /// The identity extraction decides which of several faces to condition on
    /// while preparing the job's dependencies — long after this handler built
    /// its admission advisories. Before #1223 that decision was a server-side
    /// `tracing::warn!` and the person who handed mold a group photograph was
    /// never told which face it picked.
    #[test]
    fn advisories_the_render_produced_join_the_request_warning_header() {
        let admission = super::RequestWarnings {
            dimension: Some("rounded 1023 up to 1024".to_string()),
            other: vec!["the requested collection was dropped".to_string()],
        };
        let identity =
            "3 faces were detected in the identity image; conditioning on the largest one";

        let merged = super::merge_render_warnings(admission, &[identity.to_string()]);
        let all = merged.all().collect::<Vec<_>>();
        assert!(all.contains(&identity), "{all:?}");
        assert!(
            all.contains(&"the requested collection was dropped"),
            "{all:?}"
        );
        // The dimension advisory keeps its own channel.
        assert_eq!(merged.dimension.as_deref(), Some("rounded 1023 up to 1024"));
        assert_eq!(
            super::request_warning_headers(&merged.all().map(str::to_string).collect::<Vec<_>>())
                .get("x-mold-request-warning")
                .unwrap(),
            "the requested collection was dropped; \
             3 faces were detected in the identity image; conditioning on the largest one; \
             rounded 1023 up to 1024"
        );
    }

    #[test]
    fn deferred_preparation_advisories_join_direct_admission_warnings() {
        let admission = super::RequestWarnings {
            dimension: Some("rounded 1023 up to 1024".to_string()),
            other: vec!["the requested collection was dropped".to_string()],
        };
        let deferred = super::RequestWarnings {
            dimension: None,
            other: vec![
                "lip-dub takes its length from the reference video".to_string(),
                "the requested collection was dropped".to_string(),
            ],
        };

        let merged = super::merge_request_warnings(admission, deferred);
        assert_eq!(
            merged.all().collect::<Vec<_>>(),
            vec![
                "the requested collection was dropped",
                "lip-dub takes its length from the reference video",
                "rounded 1023 up to 1024",
            ]
        );
    }

    #[test]
    fn durable_status_exposes_claimed_cancellation_without_regressing_to_accepted() {
        let detail = mold_db::generation_batches::DurableGenerationBatchDetail {
            batch: mold_db::generation_batches::GenerationBatchRow {
                id: "batch-id".to_string(),
                client_batch_id: "client-id".to_string(),
                owner_uuid: "owner-id".to_string(),
                request_sha256: "receipt".to_string(),
                created_at_ms: 10,
            },
            children: vec![
                mold_db::generation_batches::DurableGenerationBatchChildRow {
                    batch_id: "batch-id".to_string(),
                    job_id: "job-id".to_string(),
                    batch_index: 1,
                    state: "cancelling".to_string(),
                    error: Some("Cancelled".to_string()),
                    retryable: false,
                    error_code: None,
                    updated_at_ms: 20,
                    revision: 3,
                    terminal_error_json: None,
                    result_json: None,
                    completed_at_ms: None,
                },
            ],
        };

        let status = super::generation_batch_status("instance-id", detail);
        assert_eq!(
            status.children[0].state,
            mold_core::GenerationBatchChildState::Cancelling
        );
        assert_eq!(status.children[0].completed_at_ms, None);
    }

    /// A batch child carries its parent's identity advisory, so the same text
    /// can arrive twice. A caller should see it once.
    #[test]
    fn a_repeated_render_advisory_is_not_duplicated() {
        let held = "3 faces were detected in the identity image; conditioning on the largest one";
        let merged = super::merge_render_warnings(
            super::RequestWarnings {
                dimension: None,
                other: vec![held.to_string()],
            },
            &[held.to_string()],
        );
        assert_eq!(merged.all().count(), 1);
    }

    #[tokio::test]
    async fn built_in_control_is_the_first_concrete_lora_at_unit_scale() {
        let state = AppState::for_tests();
        let temp = tempfile::tempdir().unwrap();
        let adapter_path = temp.path().join("control.safetensors");
        std::fs::write(&adapter_path, b"installed").unwrap();
        mold_core::download::write_sha256_marker(&adapter_path, "test").unwrap();
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test",
            "model": "ltx-2-19b-distilled:fp8",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "lora": { "path": "/loras/legacy.safetensors", "scale": 0.6 },
            "loras": [{ "path": "/loras/style.safetensors", "scale": 0.8 }]
        }))
        .unwrap();
        let adapter = mold_core::ltx2_control::resolve_control_adapter(
            mold_core::ltx2_control::Ltx2ControlProfile::Ltx2_19bDistilled,
            "union",
        )
        .unwrap();

        materialize_builtin_ltx2_control(&state, &mut request, adapter, adapter_path.clone())
            .await
            .unwrap();

        assert!(request.lora.is_none());
        let loras = request.loras.unwrap();
        assert_eq!(loras.len(), 3);
        assert_eq!(loras[0].path, adapter_path.to_string_lossy());
        assert_eq!(loras[0].scale, 1.0);
        assert_eq!(loras[1].path, "/loras/legacy.safetensors");
        assert_eq!(loras[2].path, "/loras/style.safetensors");
    }

    struct TrackingUpscaler {
        unloaded: std::sync::Arc<std::sync::atomic::AtomicBool>,
        unloaded_on: std::sync::Arc<std::sync::Mutex<Option<std::thread::ThreadId>>>,
    }

    impl mold_inference::UpscaleEngine for TrackingUpscaler {
        fn upscale(
            &mut self,
            _req: &mold_core::UpscaleRequest,
        ) -> anyhow::Result<mold_core::UpscaleResponse> {
            unreachable!("cleanup test never runs inference")
        }

        fn model_name(&self) -> &str {
            "tracking-upscaler"
        }

        fn is_loaded(&self) -> bool {
            !self.unloaded.load(Ordering::SeqCst)
        }

        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }

        fn unload(&mut self) {
            *self.unloaded_on.lock().unwrap() = Some(std::thread::current().id());
            self.unloaded.store(true, Ordering::SeqCst);
        }

        fn scale_factor(&self) -> u32 {
            4
        }

        fn set_on_progress(&mut self, _callback: mold_inference::progress::ProgressCallback) {}

        fn clear_on_progress(&mut self) {}
    }

    #[tokio::test]
    async fn clearing_upscaler_cache_unloads_before_drop() {
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(queue_tx),
            AppState::empty_gpu_pool_for_test(),
            1,
        );
        let unloaded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let unloaded_on = std::sync::Arc::new(std::sync::Mutex::new(None));
        let runtime_thread = std::thread::current().id();
        *state.upscaler_cache.lock().unwrap() = Some(Box::new(TrackingUpscaler {
            unloaded: unloaded.clone(),
            unloaded_on: unloaded_on.clone(),
        }));

        clear_global_upscaler_cache(&state).await;

        assert!(unloaded.load(Ordering::SeqCst));
        assert_ne!(
            unloaded_on.lock().unwrap().as_ref(),
            Some(&runtime_thread),
            "upscaler teardown must not run on the async runtime thread"
        );
        assert!(state.upscaler_cache.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn generation_placement_normalization_honors_request_over_persisted_defaults() {
        let mut config = mold_core::Config::default();
        config.set_model_placement(
            "flux-dev:q4",
            Some(mold_core::types::DevicePlacement {
                text_encoders: mold_core::types::DeviceRef::Cpu,
                advanced: None,
            }),
        );
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let state = AppState::empty(
            config,
            crate::state::QueueHandle::new(queue_tx),
            AppState::empty_gpu_pool_for_test(),
            1,
        );
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q4",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 1.0,
            "batch_size": 1,
            "strength": 0.75
        }))
        .unwrap();

        normalize_generation_placement(&state, &mut request).await;
        assert_eq!(
            request
                .placement
                .as_ref()
                .expect("persisted placement applied")
                .text_encoders,
            mold_core::types::DeviceRef::Cpu
        );

        request.placement = Some(mold_core::types::DevicePlacement {
            text_encoders: mold_core::types::DeviceRef::Auto,
            advanced: Some(mold_core::types::AdvancedPlacement {
                transformer: mold_core::types::DeviceRef::device(
                    "cuda:0123456789abcdef0123456789abcdef",
                ),
                ..Default::default()
            }),
        });
        normalize_generation_placement(&state, &mut request).await;
        let effective = request.placement.expect("request placement retained");
        assert_eq!(effective.text_encoders, mold_core::types::DeviceRef::Auto);
        assert!(matches!(
            effective
                .advanced
                .expect("request advanced placement")
                .transformer,
            mold_core::types::DeviceRef::Device { .. }
        ));
    }

    #[test]
    fn expand_config_for_request_threads_style() {
        let settings = mold_core::expand::ExpandSettings::default();
        let req = mold_core::ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "flux".to_string(),
            variations: 2,
            style: Some("oil painting".to_string()),
            task: Some(mold_core::ExpandTask::ImageToVideo),
        };
        let config = expand_config_for_request(&settings, &req);
        assert_eq!(config.style.as_deref(), Some("oil painting"));
        assert_eq!(config.model_family, "flux");
        assert_eq!(config.variations, 2);
        assert_eq!(config.task, mold_core::ExpandTask::ImageToVideo);

        let bare = mold_core::ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "flux".to_string(),
            variations: 1,
            style: None,
            task: None,
        };
        let config = expand_config_for_request(&settings, &bare);
        assert!(config.style.is_none());
        assert_eq!(config.task, mold_core::ExpandTask::TextToImage);
    }

    #[test]
    fn expand_variation_admission_allows_large_counts_but_rejects_unsafe_sets() {
        validate_expand_variations(10_000).unwrap();
        assert!(validate_expand_variations(0).is_err());
        assert!(validate_expand_variations(10_001).is_err());
    }

    #[test]
    fn local_expand_capability_reports_feature_and_model_facts_separately() {
        let settings = mold_core::expand::ExpandSettings::default();
        let missing = expand_capabilities(&settings, false);
        let present = expand_capabilities(&settings, true);

        assert_eq!(missing.backend, mold_core::ExpandBackend::Local);
        assert!(missing.remix);
        assert_eq!(missing.configured, cfg!(feature = "expand"));
        assert_eq!(missing.model_present, Some(false));
        assert_eq!(present.configured, cfg!(feature = "expand"));
        assert_eq!(present.model_present, Some(true));
        assert!(present.remix);
        // Clients pull what the host names, never a hard-coded manifest id.
        assert_eq!(present.model.as_deref(), Some(settings.model.as_str()));
        assert_eq!(missing.model.as_deref(), Some(settings.model.as_str()));

        let custom = mold_core::expand::ExpandSettings {
            model: "qwen3-expand:q4".into(),
            ..Default::default()
        };
        assert_eq!(
            expand_capabilities(&custom, true).model.as_deref(),
            Some("qwen3-expand:q4")
        );
    }

    #[test]
    fn api_expand_capability_does_not_claim_external_reachability() {
        let settings = mold_core::expand::ExpandSettings {
            backend: "http://localhost:11434".into(),
            ..Default::default()
        };
        let capability = expand_capabilities(&settings, false);

        assert_eq!(capability.backend, mold_core::ExpandBackend::Api);
        assert!(capability.configured);
        assert_eq!(capability.model_present, None);
        // An API backend runs someone else's weights; there is nothing to pull.
        assert_eq!(capability.model, None);

        let unconfigured = mold_core::expand::ExpandSettings {
            backend: "   ".into(),
            ..Default::default()
        };
        let capability = expand_capabilities(&unconfigured, true);
        assert_eq!(capability.backend, mold_core::ExpandBackend::Api);
        assert!(!capability.configured);
        assert_eq!(capability.model_present, None);
    }

    #[test]
    fn dispatch_capability_reports_authority_without_implying_live_cutover() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let legacy_handle = crate::scheduler::ScheduledWorkHandle::for_mode(
            tx,
            crate::dispatch_mode::DispatchMode::Legacy,
        );
        let legacy = dispatch_capabilities(&legacy_handle);
        assert_eq!(legacy.active_mode.as_deref(), Some("legacy"));
        assert!(!legacy.v2_authoritative);
        assert!(!legacy.observes_v2_decisions);
        let legacy_devices = device_capabilities(&legacy_handle);
        assert!(!legacy_devices.lifecycle);
        assert!(!legacy_devices.planned_lanes);
        assert!(!legacy_devices.learned_eta);

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let observe_handle = crate::scheduler::ScheduledWorkHandle::for_mode(
            tx,
            crate::dispatch_mode::DispatchMode::Observe,
        );
        let observe = dispatch_capabilities(&observe_handle);
        assert_eq!(observe.active_mode.as_deref(), Some("observe"));
        assert!(!observe.v2_authoritative);
        assert!(observe.observes_v2_decisions);
        let observe_devices = device_capabilities(&observe_handle);
        assert!(!observe_devices.lifecycle);
        assert!(!observe_devices.planned_lanes);
        assert!(!observe_devices.learned_eta);

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let v2_handle = crate::scheduler::ScheduledWorkHandle::for_mode(
            tx,
            crate::dispatch_mode::DispatchMode::V2,
        );
        let v2 = dispatch_capabilities(&v2_handle);
        assert_eq!(v2.active_mode.as_deref(), Some("v2"));
        assert!(v2.v2_authoritative);
        assert!(!v2.observes_v2_decisions);
        assert_eq!(v2.modes, ["legacy", "observe", "v2"]);
        let v2_devices = device_capabilities(&v2_handle);
        assert!(v2_devices.lifecycle);
        assert!(v2_devices.planned_lanes);
        assert!(v2_devices.learned_eta);

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let maintenance = crate::scheduler::ScheduledWorkHandle::for_runtime(
            tx,
            crate::dispatch_mode::DispatchMode::V2,
            false,
            false,
        );
        let maintenance = dispatch_capabilities(&maintenance);
        assert_eq!(maintenance.active_mode.as_deref(), Some("v2"));
        assert!(!maintenance.v2_authoritative);
        assert!(!maintenance.observes_v2_decisions);
    }

    #[test]
    fn disk_usage_for_path_picks_longest_matching_mount() {
        let disks = vec![
            (std::path::PathBuf::from("/"), 100, 10),
            (std::path::PathBuf::from("/data"), 500, 50),
        ];
        let usage = disk_usage_for_path(&disks, std::path::Path::new("/data/models")).unwrap();
        assert_eq!(usage.total_bytes, 500);
        assert_eq!(usage.free_bytes, 50);
        // A path outside /data falls back to the root mount.
        let usage = disk_usage_for_path(&disks, std::path::Path::new("/home/u/models")).unwrap();
        assert_eq!(usage.total_bytes, 100);
    }

    #[test]
    fn disk_usage_for_path_component_boundaries_and_no_match() {
        let disks = vec![(std::path::PathBuf::from("/data"), 500, 50)];
        // `/database` is not under the `/data` mount — prefix matching must be
        // per path component, not per byte.
        assert_eq!(
            disk_usage_for_path(&disks, std::path::Path::new("/database/models")),
            None
        );
        // No mount matches at all (e.g. relative path) → None.
        assert_eq!(
            disk_usage_for_path(&disks, std::path::Path::new("relative/models")),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn canonical_models_dir_resolves_symlinks_for_mount_matching() {
        // A symlinked models dir (`ln -s /Volumes/Big/models ~/.mold/models`)
        // must resolve to the target's filesystem: the longest-prefix mount
        // match has to run against the canonical target path, not the
        // symlink's own location.
        let tmp = tempfile::tempdir().unwrap();
        // Canonicalize the tempdir itself so the fake mount table matches on
        // platforms where the temp root is itself a symlink (/tmp, /var).
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let target = root.join("big-volume").join("models");
        std::fs::create_dir_all(&target).unwrap();
        let link = root.join("link-models");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let resolved = canonical_models_dir(&link);
        assert_eq!(resolved, target);

        // With a mount table containing the symlink target's volume, the
        // canonicalized path must pick that mount over the root fallback.
        let disks = vec![
            (std::path::PathBuf::from("/"), 100, 10),
            (root.join("big-volume"), 500, 50),
        ];
        let usage = disk_usage_for_path(&disks, &resolved).unwrap();
        assert_eq!(usage.total_bytes, 500);
        assert_eq!(usage.free_bytes, 50);
    }

    #[test]
    fn canonical_models_dir_falls_back_to_the_literal_path() {
        // A models dir that doesn't exist yet must not panic or change the
        // matching behavior — the raw path is used as-is.
        let missing = std::path::Path::new("/definitely/not/a/real/mold/models/dir");
        assert_eq!(canonical_models_dir(missing), missing.to_path_buf());
    }

    #[test]
    fn clean_error_message_strips_backtrace() {
        let err = anyhow::anyhow!(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")\n\
             \x20  0: candle_core::error::Error::bt\n\
             \x20           at /home/user/.cargo/git/candle/src/error.rs:264:25\n\
             \x20  1: <core::result::Result<O,E> as candle_core::cuda_backend::error::WrapErr<O>>::w\n\
             \x20           at /home/user/.cargo/git/candle/src/cuda_backend/error.rs:60:65"
        );
        let msg = clean_error_message(&err);
        assert_eq!(
            msg,
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        );
    }

    #[test]
    fn clean_error_message_preserves_simple_error() {
        let err = anyhow::anyhow!("model not found: flux-dev:q4");
        let msg = clean_error_message(&err);
        assert_eq!(msg, "model not found: flux-dev:q4");
    }

    #[test]
    fn clean_error_message_preserves_multiline_without_backtrace() {
        let err = anyhow::anyhow!("validation failed\nprompt is empty\nsteps must be > 0");
        let msg = clean_error_message(&err);
        assert_eq!(msg, "validation failed\nprompt is empty\nsteps must be > 0");
    }

    #[test]
    fn clean_error_message_strips_high_numbered_frames() {
        let err = anyhow::anyhow!(
            "some error\n\
             \x20 10: tokio::runtime::task::core::Core<T,S>::poll at /home/user/.cargo/tokio/src/core.rs:375\n\
             \x20 11: std::panicking::catch_unwind at /nix/store/rust/src/panicking.rs:544"
        );
        let msg = clean_error_message(&err);
        assert_eq!(msg, "some error");
    }

    #[test]
    fn clean_error_message_empty_fallback() {
        // An error whose Display starts immediately with a backtrace-like line
        let err = anyhow::anyhow!("0: candle_core::error::Error::bt at /some/path.rs:10:5");
        let msg = clean_error_message(&err);
        // Should fall back to root_cause since all lines look like backtrace
        assert!(!msg.is_empty());
    }

    #[test]
    fn clean_error_message_renders_full_anyhow_chain() {
        // Wrapped errors must surface the root cause; previously the outer
        // `with_context` swallowed everything below it (cv:2739091 truncated
        // checkpoint surfaced as "mmap single-file checkpoint at …" with no
        // hint that the safetensors data was short).
        let root = std::io::Error::new(std::io::ErrorKind::InvalidData, "bytes past end");
        let err: anyhow::Error = anyhow::Error::new(root)
            .context("validate single-file checkpoint at /tmp/foo.safetensors");
        let msg = clean_error_message(&err);
        assert!(
            msg.contains("validate single-file checkpoint") && msg.contains("bytes past end"),
            "expected both context layers in the rendered chain, got: {msg}",
        );
    }

    #[test]
    fn save_image_to_dir_creates_directory_and_writes_file() {
        let dir = std::env::temp_dir().join(format!(
            "mold-save-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!dir.exists());

        let img = mold_core::ImageData {
            data: vec![0x89, 0x50, 0x4E, 0x47], // PNG magic bytes
            format: mold_core::OutputFormat::Png,
            width: 64,
            height: 64,
            index: 0,
        };

        save_image_to_dir(&dir, &img, "test-model:q8", 1);

        assert!(dir.exists(), "directory should be created");
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(files.len(), 1, "should have exactly one file");
        let file = files[0].as_ref().unwrap();
        let filename = file.file_name().to_str().unwrap().to_string();
        assert!(filename.starts_with("mold-test-model-q8-"), "{filename}");
        assert!(filename.ends_with(".png"), "{filename}");
        let contents = std::fs::read(file.path()).unwrap();
        assert_eq!(contents, vec![0x89, 0x50, 0x4E, 0x47]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_image_to_dir_batch_includes_index() {
        let dir = std::env::temp_dir().join(format!(
            "mold-save-batch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let img = mold_core::ImageData {
            data: vec![0xFF, 0xD8], // JPEG magic
            format: mold_core::OutputFormat::Jpeg,
            width: 64,
            height: 64,
            index: 2,
        };

        save_image_to_dir(&dir, &img, "flux-dev", 4);

        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(files.len(), 1);
        let filename = files[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            filename.contains("-2.jpeg"),
            "batch index in name: {filename}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_image_to_dir_invalid_path_logs_warning_no_panic() {
        // Saving to a path that can't be created should not panic
        let img = mold_core::ImageData {
            data: vec![0x00],
            format: mold_core::OutputFormat::Png,
            width: 1,
            height: 1,
            index: 0,
        };
        // /dev/null/impossible can't be created as a directory
        save_image_to_dir(
            std::path::Path::new("/dev/null/impossible"),
            &img,
            "test",
            1,
        );
        // Test passes if no panic occurred
    }

    #[test]
    fn thumbnail_warmup_is_enabled_by_default() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var("MOLD_THUMBNAIL_WARMUP");
        }
        assert!(thumbnail_warmup_enabled());
    }

    #[tokio::test]
    async fn gallery_list_etag_returns_unchanged_without_a_body() {
        let first = gallery_list_response(&HeaderMap::new(), Vec::new(), None).unwrap();
        let etag = first.headers().get(header::ETAG).unwrap().clone();
        assert_eq!(first.status(), StatusCode::OK);

        let mut headers = HeaderMap::new();
        headers.insert(header::IF_NONE_MATCH, etag);
        let second = gallery_list_response(&headers, Vec::new(), None).unwrap();
        assert_eq!(second.status(), StatusCode::NOT_MODIFIED);
        let bytes = axum::body::to_bytes(second.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn thumbnail_cache_identity_changes_with_media_version() {
        let dir = std::path::Path::new("/tmp/thumbs");
        let first = versioned_thumbnail_path(dir, "cat.png", "1-100");
        let second = versioned_thumbnail_path(dir, "cat.png", "2-100");
        assert_ne!(first, second);
        assert_eq!(first, versioned_thumbnail_path(dir, "cat.png", "1-100"));
    }

    #[test]
    fn concurrent_thumbnail_misses_share_one_flight() {
        let path = std::path::Path::new("/tmp/thumbs/singleflight.png");
        let first = thumbnail_singleflight(path);
        let second = thumbnail_singleflight(path);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn thumbnail_warmup_accepts_truthy_env_values() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "1");
        }
        assert!(thumbnail_warmup_enabled());
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "true");
        }
        assert!(thumbnail_warmup_enabled());
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "YES");
        }
        assert!(thumbnail_warmup_enabled());
        unsafe {
            std::env::remove_var("MOLD_THUMBNAIL_WARMUP");
        }
    }

    #[test]
    fn thumbnail_warmup_rejects_falsey_env_values() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "0");
        }
        assert!(!thumbnail_warmup_enabled());
        unsafe {
            std::env::set_var("MOLD_THUMBNAIL_WARMUP", "false");
        }
        assert!(!thumbnail_warmup_enabled());
        unsafe {
            std::env::remove_var("MOLD_THUMBNAIL_WARMUP");
        }
    }

    #[test]
    fn content_type_covers_every_output_format() {
        assert_eq!(content_type_for_filename("a.png"), "image/png");
        assert_eq!(content_type_for_filename("a.PNG"), "image/png");
        assert_eq!(content_type_for_filename("a.jpg"), "image/jpeg");
        assert_eq!(content_type_for_filename("a.jpeg"), "image/jpeg");
        assert_eq!(content_type_for_filename("a.gif"), "image/gif");
        assert_eq!(content_type_for_filename("a.webp"), "image/webp");
        assert_eq!(content_type_for_filename("a.apng"), "image/apng");
        assert_eq!(content_type_for_filename("a.mp4"), "video/mp4");
        assert_eq!(
            content_type_for_filename("a.unknown"),
            "application/octet-stream"
        );
    }
    // ── Gallery validation ───────────────────────────────────────────────
    // The guard-rail pure functions live in `mold_db::metadata_io` (with
    // their own unit tests); these end-to-end scans pin that the server
    // gallery keeps consuming them correctly.

    /// Create a scratch directory unique to this test and delete it on drop.
    /// Using `std::env::temp_dir()` rather than pulling in a `tempfile`
    /// dev-dep for two tests' worth of fixtures.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("mold-gallery-test-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).expect("create tempdir");
            Self(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Encode a noisy PNG in-memory via the `image` crate. The checkerboard
    /// pattern resists zlib compression so the encoded bytes exceed the
    /// gallery size floor — a solid-color PNG of the same dimensions would
    /// compress to ~80 bytes and be filtered out by the size guard.
    fn make_png_bytes(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            let n = (x.wrapping_mul(37) ^ y.wrapping_mul(131)) as u8;
            image::Rgb([n, n.wrapping_add(85), n.wrapping_sub(17)])
        });
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("encode png");
        buf
    }

    #[test]
    fn scan_gallery_dir_filters_invalid_and_keeps_valid() {
        let td = TempDir::new("scan");
        let dir = td.path();

        // A valid PNG large enough to exceed the 256-byte raster size floor.
        std::fs::write(dir.join("mold-model-1000.png"), make_png_bytes(32, 32)).unwrap();

        // Truncated raster that passes size floor but has no valid header.
        let mut junk = vec![0u8; 512];
        junk[..4].copy_from_slice(b"JUNK");
        std::fs::write(dir.join("mold-broken-2000.png"), &junk).unwrap();

        // Tiny raster under the size floor (sub-IHDR).
        std::fs::write(
            dir.join("mold-tiny-3000.png"),
            b"\x89PNG\r\n\x1a\n", // 8 bytes: signature only
        )
        .unwrap();

        // Valid-enough mp4 (ftyp at offset 4) — should survive.
        let mut mp4 = Vec::new();
        mp4.extend_from_slice(&[0x00, 0x00, 0x00, 0x20]);
        mp4.extend_from_slice(b"ftyp");
        mp4.extend_from_slice(b"isom\x00\x00\x02\x00");
        // Pad above the 4096-byte mp4 size floor so it isn't filtered on
        // size alone — the scan still checks ftyp either way.
        mp4.resize(8192, 0);
        std::fs::write(dir.join("mold-ltx-4000.mp4"), &mp4).unwrap();

        // Mp4 extension but no ftyp.
        let bad_mp4 = vec![0u8; 8192];
        std::fs::write(dir.join("mold-no-ftyp-5000.mp4"), &bad_mp4).unwrap();

        // Unsupported extension — ignored entirely.
        std::fs::write(dir.join("random.txt"), b"not an output").unwrap();

        let results = scan_gallery_dir(dir);
        let names: Vec<&str> = results.iter().map(|i| i.filename.as_str()).collect();
        assert!(
            names.contains(&"mold-model-1000.png"),
            "valid PNG should survive: {names:?}"
        );
        assert!(
            names.contains(&"mold-ltx-4000.mp4"),
            "valid MP4 with ftyp should survive: {names:?}"
        );
        assert!(
            !names.contains(&"mold-broken-2000.png"),
            "PNG with no valid header should be filtered: {names:?}"
        );
        assert!(
            !names.contains(&"mold-tiny-3000.png"),
            "under-size PNG stub should be filtered: {names:?}"
        );
        assert!(
            !names.contains(&"mold-no-ftyp-5000.mp4"),
            "MP4 without ftyp should be filtered: {names:?}"
        );
        assert_eq!(names.len(), 2, "only the 2 valid fixtures remain");
    }

    #[test]
    fn solid_black_png_is_filtered_at_scan_time() {
        let td = TempDir::new("black");
        let dir = td.path();

        // A 256×256 solid-black PNG — definitely below the suspect-size
        // threshold (compresses to a few hundred bytes) and every pixel is
        // below the channel ceiling.
        let black = image::RgbImage::from_pixel(256, 256, image::Rgb([0, 0, 0]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(black)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(dir.join("mold-noisy-1000.png"), &buf).unwrap();

        // A normal noisy PNG with the same dimensions — should survive.
        std::fs::write(dir.join("mold-valid-2000.png"), make_png_bytes(256, 256)).unwrap();

        let results = scan_gallery_dir(dir);
        let names: Vec<&str> = results.iter().map(|i| i.filename.as_str()).collect();
        assert!(
            !names.contains(&"mold-noisy-1000.png"),
            "solid-black PNG should be filtered: {names:?}"
        );
        assert!(
            names.contains(&"mold-valid-2000.png"),
            "noisy PNG should survive: {names:?}"
        );
    }
    #[test]
    fn parse_byte_range_handles_common_forms() {
        // `bytes=0-499` — first 500 bytes
        assert_eq!(parse_byte_range("bytes=0-499", 2000), Some((0, 499)));
        // open-ended `bytes=100-` — from byte 100 to EOF
        assert_eq!(parse_byte_range("bytes=100-", 2000), Some((100, 1999)));
        // suffix `bytes=-500` — last 500 bytes
        assert_eq!(parse_byte_range("bytes=-500", 2000), Some((1500, 1999)));
        // end past EOF — clamped to last byte
        assert_eq!(parse_byte_range("bytes=0-9999", 2000), Some((0, 1999)));
        // whole file
        assert_eq!(parse_byte_range("bytes=0-1999", 2000), Some((0, 1999)));
    }

    #[test]
    fn parse_byte_range_rejects_malformed_and_unsatisfiable() {
        assert_eq!(parse_byte_range("bytes=", 1000), None);
        assert_eq!(parse_byte_range("bytes=abc-100", 1000), None);
        // start past EOF
        assert_eq!(parse_byte_range("bytes=2000-", 1000), None);
        // end before start
        assert_eq!(parse_byte_range("bytes=500-100", 1000), None);
        // multi-range not supported
        assert_eq!(parse_byte_range("bytes=0-10,20-30", 1000), None);
        // suffix of 0 bytes is meaningless
        assert_eq!(parse_byte_range("bytes=-0", 1000), None);
        // empty file can't satisfy any range
        assert_eq!(parse_byte_range("bytes=0-10", 0), None);
        // wrong unit prefix
        assert_eq!(parse_byte_range("items=0-10", 1000), None);
    }

    #[test]
    fn scan_populates_real_dimensions_for_synthesized_metadata() {
        // Files without an embedded mold:parameters chunk still get their
        // actual width/height filled in from the header decode — useful for
        // the SPA's aspect-ratio-preserving layout.
        let td = TempDir::new("dims");
        let dir = td.path();
        std::fs::write(dir.join("mold-nometa-1000.png"), make_png_bytes(128, 96)).unwrap();

        let results = scan_gallery_dir(dir);
        assert_eq!(results.len(), 1);
        let entry = &results[0];
        assert!(entry.metadata_synthetic);
        assert_eq!(entry.metadata.width, 128);
        assert_eq!(entry.metadata.height, 96);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn thumbnail_warmup_observes_one_entry_atomically_and_yields_to_writer() {
        let output = TempDir::new("warmup-output");
        let thumbs = TempDir::new("warmup-thumbs");
        std::fs::write(output.path().join("print.png"), make_png_bytes(32, 32)).unwrap();
        let gate = crate::batch_transaction::GalleryPublicationGate::default();
        let worker_gate = gate.clone();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = std::sync::Arc::new(std::sync::Mutex::new(release_rx));
        let (warm_done_tx, warm_done_rx) = std::sync::mpsc::channel();
        let output_path = output.path().to_path_buf();
        let thumbs_path = thumbs.path().to_path_buf();
        let worker = std::thread::spawn(move || {
            let release_rx = release_rx.clone();
            warm_gallery_thumbnails(
                &output_path,
                &thumbs_path,
                &worker_gate,
                &|observation| {
                    if observation == 0 {
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                    }
                },
                &|_| {},
            );
            warm_done_tx.send(()).unwrap();
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();

        let (writer_acquired_tx, writer_acquired_rx) = tokio::sync::oneshot::channel();
        let (writer_release_tx, writer_release_rx) = tokio::sync::oneshot::channel();
        let writer_gate = gate.clone();
        let writer = tokio::spawn(async move {
            let _writer = writer_gate.write().await;
            let _ = writer_acquired_tx.send(());
            let _ = writer_release_rx.await;
        });
        let mut writer_acquired_rx = writer_acquired_rx;
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(25),
                &mut writer_acquired_rx
            )
            .await
            .is_err(),
            "writer observed the directory while one warmup entry was only partially inspected"
        );

        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), &mut writer_acquired_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(
                warm_done_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "warmup reacquired the reader instead of yielding between entries"
        );
        let _ = writer_release_tx.send(());
        writer.await.unwrap();
        worker.join().unwrap();
    }

    #[tokio::test]
    async fn expansion_skipped_for_empty_prompt() {
        // An empty prompt is a deliberate "let the conditioning speak" signal.
        // Handing "" to the expander would let the LLM hallucinate a prompt
        // that then becomes the frozen, recorded prompt.
        let state = AppState::for_tests();
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "   ",
            "model": "ltx-2-19b-distilled:fp8",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "expand": true
        }))
        .unwrap();

        maybe_expand_prompt(&state, &mut request, None, None)
            .await
            .unwrap();

        assert_eq!(request.prompt, "   ");
        assert!(request.original_prompt.is_none());
        // Cleared so scheduler-owned local expansion doesn't re-plan it.
        assert_eq!(request.expand, Some(false));
    }

    #[tokio::test]
    async fn generation_expansion_rejects_h3_before_scheduling() {
        let state = AppState::for_tests();
        state.config.write().await.expand.model = "MiniMax-H3".to_string();
        let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a quiet desert sunrise",
            "model": "flux-schnell:q8",
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1,
            "expand": true
        }))
        .unwrap();

        let error = maybe_expand_prompt(&state, &mut request, None, Some("flux"))
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS);
        assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        assert_eq!(request.prompt, "a quiet desert sunrise");
        assert!(request.original_prompt.is_none());
    }

    /// Both doors a compact H3 request can reach agree on the canvas RULE,
    /// and neither needs a GPU to answer.
    ///
    /// Authenticated private H3 ingress deliberately skips
    /// `validate_request_against_generation_profile`, so its own door —
    /// `validate_h3_private_uat_request` — has to carry the canvas rule, and
    /// every ordinary client reaches the same answer through the profile's
    /// range. That door's canvas rule IS the `validate_reviewed_canvas`
    /// called here: the `h3` feature gates the wrapper, not the rule, and
    /// `mold-ai-core`'s
    /// `private_h3_ingress_admits_every_reviewed_canvas_and_refuses_the_rest`
    /// pins the delegation verbatim. 1024x768 is the interesting positive: a
    /// canonical upstream resolver output no campaign has run, refused by the
    /// old bucket set and admitted by the rule.
    #[test]
    fn both_admission_doors_admit_every_reviewed_h3_canvas_and_refuse_the_rest() {
        let request = |model: &str, width: u32, height: u32| {
            serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
                "prompt": "a red fox in a snowy pine forest at dawn",
                "model": model,
                "width": width,
                "height": height,
                "steps": mold_core::minimax_h3::COMFY_DEFAULT_STEPS,
                "guidance": 0.0,
                "strength": 1.0,
                "seed": 770_021,
                "batch_size": 1,
                "frames": mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES,
                "fps": mold_core::minimax_h3::FIXED_FPS,
                "output_format": "mp4"
            }))
            .unwrap()
        };
        let profile = |model: &str| {
            mold_core::resolve_generation_profile(mold_core::GenerationProfileInput {
                model,
                family: mold_core::minimax_h3::FAMILY,
                sub_family: None,
                default_width: mold_core::minimax_h3::DEFAULT_WIDTH,
                default_height: mold_core::minimax_h3::DEFAULT_HEIGHT,
                default_steps: mold_core::minimax_h3::COMFY_DEFAULT_STEPS,
                default_guidance: 0.0,
                default_frames: Some(mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES),
                default_fps: Some(mold_core::minimax_h3::FIXED_FPS),
                default_negative_prompt: None,
                source_image: Some(mold_core::SourceImageCapability::Required),
                supports_sequence: false,
                supports_extend: false,
                supports_audio: true,
            })
        };

        for model in [
            mold_core::minimax_h3::FL2VA_COMFY,
            mold_core::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
        ] {
            // Every recommended preset, plus canvases no campaign ran that
            // the rule admits.
            let admitted = mold_core::minimax_h3::REVIEWED_COMPACT_CANVASES
                .iter()
                .copied()
                .chain([(1024, 768), (768, 1344), (1024, 576), (512, 1984)]);
            for (width, height) in admitted {
                let mut reviewed = request(model, width, height);
                reviewed.steps = profile(model).default_recipe().unwrap().steps.default;
                mold_core::minimax_h3::validate_reviewed_canvas(&reviewed)
                    .unwrap_or_else(|error| panic!("{model} {width}x{height}: {}", error.message));
                mold_core::validate_request_against_generation_profile(&profile(model), &reviewed)
                    .unwrap_or_else(|error| panic!("{model} {width}x{height}: {error}"));
            }

            // Off the 32 stride, over the compact area ceiling, under the
            // compact axis floor, and outside the family aspect bounds. Each
            // must be refused at BOTH doors.
            for (width, height) in [(1000, 600), (1056, 992), (224, 896), (1600, 288)] {
                let mut off = request(model, width, height);
                off.steps = profile(model).default_recipe().unwrap().steps.default;
                let private = mold_core::minimax_h3::validate_reviewed_canvas(&off).unwrap_err();
                assert_eq!(
                    private.code, "MINIMAX_H3_DIMENSIONS",
                    "{model} {width}x{height}: {}",
                    private.message
                );
                assert!(
                    mold_core::validate_request_against_generation_profile(&profile(model), &off)
                        .is_err(),
                    "{model} {width}x{height} passed the ordinary door"
                );
            }

            // The clip length is a grid now, and both doors follow it.
            for frames in [
                mold_core::minimax_h3::MIN_FRAMES,
                mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES,
                mold_core::minimax_h3::MAX_FRAMES,
            ] {
                let mut clip = request(
                    model,
                    mold_core::minimax_h3::DEFAULT_WIDTH,
                    mold_core::minimax_h3::DEFAULT_HEIGHT,
                );
                clip.steps = profile(model).default_recipe().unwrap().steps.default;
                clip.frames = Some(frames);
                mold_core::validate_request_against_generation_profile(&profile(model), &clip)
                    .unwrap_or_else(|error| panic!("{model} {frames} frames: {error}"));
            }
            for frames in [90, 125, 362] {
                let mut clip = request(
                    model,
                    mold_core::minimax_h3::DEFAULT_WIDTH,
                    mold_core::minimax_h3::DEFAULT_HEIGHT,
                );
                clip.steps = profile(model).default_recipe().unwrap().steps.default;
                clip.frames = Some(frames);
                assert!(
                    mold_core::validate_request_against_generation_profile(&profile(model), &clip)
                        .is_err(),
                    "{model} {frames} frames passed the ordinary door"
                );
            }
        }
    }

    #[tokio::test]
    async fn generation_preflight_gates_nested_models_and_root_relative_artifacts() {
        let state = AppState::for_tests();
        let root = "/Volumes/ExternalStorage/mold-uat/minimax-h3/models";
        state.config.write().await.models_dir = root.to_string();
        let request = || {
            serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
                "prompt": "a quiet desert sunrise",
                "model": "flux-schnell:q8",
                "width": 512,
                "height": 512,
                "steps": 4,
                "guidance": 0.0,
                "batch_size": 1
            }))
            .unwrap()
        };

        let mut ordinary = request();
        ordinary.lora = Some(mold_core::LoraWeight {
            path: format!("{root}/flux/ordinary.safetensors"),
            scale: 1.0,

            expert: None,
        });
        require_server_generation_request_activation(&state, &ordinary, Some("flux"))
            .await
            .unwrap();

        let mut nested_lora = request();
        nested_lora.lora = Some(mold_core::LoraWeight {
            path: format!("{root}/custom/MiniMax-H3/adapter.safetensors"),
            scale: 1.0,

            expert: None,
        });
        let error =
            require_server_generation_request_activation(&state, &nested_lora, Some("flux"))
                .await
                .unwrap_err();
        assert_eq!(error.status, StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS);

        for (control_model, upscale_model) in [
            (Some("MiniMax-H3-FL2VA"), None),
            (None, Some("hf:MiniMaxAI/MiniMax-H3")),
        ] {
            let mut nested_model = request();
            nested_model.control_model = control_model.map(str::to_string);
            nested_model.upscale_model = upscale_model.map(str::to_string);
            let error =
                require_server_generation_request_activation(&state, &nested_model, Some("flux"))
                    .await
                    .unwrap_err();
            assert_eq!(error.status, StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS);
            assert_eq!(error.code, mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED);
        }
    }

    #[test]
    fn prompt_history_skips_empty() {
        let mut state = AppState::for_tests();
        state.metadata_db =
            std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));

        record_prompt_history(&state, "", None, "ltx-2-19b-distilled:fp8");
        record_prompt_history(&state, "  \n ", None, "ltx-2-19b-distilled:fp8");
        let history = mold_db::PromptHistory::new(state.metadata_db.as_ref().as_ref().unwrap());
        assert!(history.recent(10).unwrap().is_empty());

        record_prompt_history(&state, "a red apple", None, "ltx-2-19b-distilled:fp8");
        assert_eq!(history.recent(10).unwrap().len(), 1);
    }
}
