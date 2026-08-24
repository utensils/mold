//! Canonical encrypted-media admission shared by batch and direct routes.

use std::collections::HashSet;
use std::sync::Arc;

use axum::http::StatusCode;

use crate::queue_journal::{MediaBatchJournalAdmission, MediaJournalAdmission};
use crate::queue_media_ingress::{ObserverMode, ObserverRegistration, QueueMediaIngress};
use crate::queue_media_lifecycle::QueueMediaLifecycle;
use crate::queue_media_store::{
    MediaSetRef, QueueMediaOperationFingerprint, QueueMediaOperationReceipt,
};
use crate::routes::{
    ApiError, GenerationPreparationDelivery, PreparedGenerationRoute, RequestWarnings,
};
use crate::state::{AppState, SseCompletionPayload};

pub(crate) const DURABLE_MEDIA_IDENTITY_UNDECIDABLE: &str = "DURABLE_MEDIA_IDENTITY_UNDECIDABLE";
pub(crate) const DURABLE_MEDIA_ADMISSION_CONFLICT: &str = "DURABLE_MEDIA_ADMISSION_CONFLICT";

pub(crate) struct DurableMediaAdmission {
    lifecycle: Arc<QueueMediaLifecycle>,
    ingress: Arc<QueueMediaIngress>,
}

pub(crate) struct DurableAdmissionOutcome {
    pub status_code: StatusCode,
    pub status: mold_core::GenerationBatchStatus,
    pub observer: Option<ObserverRegistration>,
    pub warnings: Option<RequestWarnings>,
}

struct PreparedChild {
    request: mold_core::GenerateRequest,
    route: PreparedGenerationRoute,
}

struct SealedChild {
    id: String,
    model: String,
    request_json: String,
    media_set: Option<MediaSetRef>,
    output_dir: std::path::PathBuf,
    target_gpu: Option<usize>,
    target_device_id: Option<String>,
    completion_payload: SseCompletionPayload,
    seed_pinned: bool,
}

impl DurableMediaAdmission {
    pub(crate) fn new(lifecycle: Arc<QueueMediaLifecycle>, queue_capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            lifecycle,
            ingress: QueueMediaIngress::new(queue_capacity),
        })
    }

    pub(crate) fn ingress(&self) -> &Arc<QueueMediaIngress> {
        &self.ingress
    }

    pub(crate) fn publish_observer(&self, job_id: &str) {
        self.ingress.publish_committed(job_id);
    }

    pub(crate) async fn admit_batch(
        self: &Arc<Self>,
        state: &AppState,
        authenticated: Option<&crate::auth::ApiKeyAuthenticated>,
        mut body: mold_core::GenerationBatchAdmissionRequest,
        observer_mode: Option<ObserverMode>,
        completion_payload: SseCompletionPayload,
    ) -> Result<DurableAdmissionOutcome, ApiError> {
        if uuid::Uuid::parse_str(body.client_batch_id.trim()).is_err() {
            return Err(ApiError::validation("client_batch_id must be a UUID"));
        }
        if body.requests.is_empty() {
            return Err(ApiError::validation(
                "requests must contain at least one child",
            ));
        }
        if observer_mode.is_some() && body.requests.len() != 1 {
            return Err(ApiError::internal(
                "an attached durable observer requires exactly one child",
            ));
        }

        let typed_history = body
            .requests
            .iter()
            .map(|request| {
                (
                    request.prompt.clone(),
                    request.negative_prompt.clone(),
                    request.model.clone(),
                )
            })
            .collect::<Vec<_>>();
        let count = body.requests.len() as u32;
        for (offset, request) in body.requests.iter_mut().enumerate() {
            mold_core::minimax_h3::canonicalize_request_model(request);
            if request.batch_size != 1 {
                return Err(ApiError::validation(format!(
                    "requests[{}].batch_size must be 1",
                    offset + 1
                )));
            }
            request.batch_id = Some(body.client_batch_id.clone());
            request.batch_index = Some(offset as u32 + 1);
            request.batch_count = Some(count);
        }
        let canonical_operation = serde_json::to_vec(&body.requests)
            .map_err(|error| ApiError::internal(format!("batch serialization failed: {error}")))?;
        let fingerprint = QueueMediaOperationFingerprint::sha256_v1(&canonical_operation);

        if let Some(existing) = existing_by_client(state, &body.client_batch_id).await? {
            self.verify_existing(&body.client_batch_id, &fingerprint, &existing)?;
            return Ok(DurableAdmissionOutcome {
                status_code: StatusCode::OK,
                status: crate::routes::generation_batch_status(&state.instance_id, existing),
                observer: None,
                warnings: None,
            });
        }

        for (offset, request) in body.requests.iter().enumerate() {
            durable_media_preflight(request).map_err(|mut error| {
                error.error = format!("requests[{}]: {}", offset + 1, error.error);
                error
            })?;
        }

        let mut prepared = Vec::with_capacity(body.requests.len());
        for (offset, mut request) in body.requests.into_iter().enumerate() {
            let route = crate::routes::prepare_generation_for_delivery(
                state,
                &mut request,
                authenticated,
                GenerationPreparationDelivery::DurableBatch,
            )
            .await
            .map_err(|mut error| {
                error.error = format!("requests[{}]: {}", offset + 1, error.error);
                error
            })?;
            if route.resolved_references.is_some() || request.references.is_some() {
                return Err(typed_refusal(
                    "DURABLE_MEDIA_PRIVATE_AUTHORITY_UNSUPPORTED",
                    "temporary reference authority cannot enter durable media admission",
                ));
            }
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            if route.h3_private_ingress_grant.is_some() {
                return Err(typed_refusal(
                    "DURABLE_MEDIA_PRIVATE_AUTHORITY_UNSUPPORTED",
                    "private MiniMax H3 authority cannot enter durable media admission",
                ));
            }
            prepared.push(PreparedChild { request, route });
        }

        let batch_id = uuid::Uuid::new_v4().to_string();
        let job_ids = (0..prepared.len())
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect::<Vec<_>>();
        let mut sealed = Vec::with_capacity(prepared.len());
        let mut sealed_refs = Vec::new();
        let mut direct_warnings = None;
        for (offset, (prepared, job_id)) in prepared.into_iter().zip(&job_ids).enumerate() {
            let child = (|| -> Result<SealedChild, ApiError> {
                let PreparedChild { request, route } = prepared;
                let PreparedGenerationRoute {
                    output_dir,
                    warnings,
                    preferred_gpu,
                    resolved_references: _,
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                        h3_private_ingress_grant: _,
                } = route;
                if observer_mode.is_some() {
                    direct_warnings = Some(warnings);
                }
                let output_dir = output_dir.ok_or_else(|| {
                    ApiError::validation(format!(
                        "requests[{}]: durable media requires server gallery output",
                        offset + 1
                    ))
                })?;
                let target_gpu = preferred_gpu;
                let target_device_id =
                    crate::queue_journal::stable_device_id_for_ordinal(state, target_gpu);
                let model = request.model.clone();
                let seed_pinned = request.seed.is_some();
                let (request_json, media_set) = if request_has_durable_media(&request) {
                    let extracted = crate::queue_media::extract_request_media(
                        job_id,
                        request,
                        &crate::queue_media::ProcessPrivateAuthorities::none(),
                    )
                    .map_err(|error| extraction_error(offset, error))?;
                    let projection = crate::queue_media::project_request_media(extracted.media())
                        .map_err(|error| extraction_error(offset, error))?;
                    let (request_json, media) = extracted.into_parts();
                    let seal_media = crate::queue_media::into_seal_media(media)
                        .map_err(|error| extraction_error(offset, error))?;
                    let reference = self
                        .lifecycle
                        .seal_v2(job_id, &fingerprint, &projection, seal_media)
                        .map_err(|error| {
                            ApiError::internal(format!(
                                "requests[{}]: encrypted media sealing failed: {error}",
                                offset + 1
                            ))
                        })?;
                    sealed_refs.push(reference.clone());
                    (request_json, Some(reference))
                } else {
                    let request_json = serde_json::to_string(&request).map_err(|error| {
                        ApiError::internal(format!(
                            "requests[{}]: request serialization failed: {error}",
                            offset + 1
                        ))
                    })?;
                    (request_json, None)
                };
                Ok(SealedChild {
                    id: job_id.clone(),
                    model,
                    request_json,
                    media_set,
                    output_dir,
                    target_gpu,
                    target_device_id,
                    completion_payload,
                    seed_pinned,
                })
            })();
            match child {
                Ok(child) => sealed.push(child),
                Err(error) => {
                    self.delete_unpublished(&sealed_refs);
                    return Err(error);
                }
            }
        }

        let receipt = self
            .lifecycle
            .seal_operation_receipt(&body.client_batch_id, &fingerprint)
            .map_err(|error| {
                self.delete_unpublished(&sealed_refs);
                ApiError::internal(format!(
                    "durable media operation receipt could not be sealed: {error}"
                ))
            })?;
        let observer = observer_mode.and_then(|mode| self.ingress.reserve(&job_ids[0], mode));
        let observer_job_id = observer.as_ref().map(|_| job_ids[0].clone());

        let journal = state.queue_journal.clone();
        let batch_id_for_db = batch_id.clone();
        let client_id_for_db = body.client_batch_id.clone();
        let receipt_for_db = receipt.as_str().to_string();
        let sealed_for_db = sealed;
        let observer_for_db = observer_job_id;
        let outcome = tokio::task::spawn_blocking(move || {
            let children = sealed_for_db
                .iter()
                .map(|child| MediaJournalAdmission {
                    id: &child.id,
                    model: &child.model,
                    request_json: &child.request_json,
                    media_set: child.media_set.as_ref(),
                    output_dir: &child.output_dir,
                    target_gpu: child.target_gpu,
                    target_device_id: child.target_device_id.as_deref(),
                    completion_payload: child.completion_payload,
                    seed_pinned: child.seed_pinned,
                })
                .collect::<Vec<_>>();
            journal.record_batch_with_media(MediaBatchJournalAdmission {
                id: &batch_id_for_db,
                client_batch_id: &client_id_for_db,
                operation_receipt: &receipt_for_db,
                children: &children,
                observer_job_id: observer_for_db.as_deref(),
            })
        })
        .await
        .map_err(|error| ApiError::internal(format!("generation batch DB task failed: {error}")))?;
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(message) => {
                // The transaction returned an authoritative failure, so none of
                // these file-first seals can be referenced by committed rows.
                self.delete_unpublished(&sealed_refs);
                return Err(ApiError::with_code(
                    message,
                    "GENERATION_BATCH_NOT_DURABLE",
                    StatusCode::UNPROCESSABLE_ENTITY,
                ));
            }
        };

        match outcome {
            mold_db::generation_batches::GenerationBatchMediaInsertOutcome::Inserted(_) => {
                for (prompt, negative, model) in &typed_history {
                    crate::routes::record_prompt_history(state, prompt, negative.as_deref(), model);
                }
                let detail = existing_by_id(state, &batch_id).await?.ok_or_else(|| {
                    ApiError::internal("generation batch disappeared after durable media admission")
                })?;
                Ok(DurableAdmissionOutcome {
                    status_code: StatusCode::ACCEPTED,
                    status: crate::routes::generation_batch_status(&state.instance_id, detail),
                    observer,
                    warnings: direct_warnings,
                })
            }
            mold_db::generation_batches::GenerationBatchMediaInsertOutcome::Existing {
                detail,
                gc_pending_media_set_ids,
                colliding_media_set_ids,
            } => {
                drop(observer);
                self.cleanup_losers(&sealed_refs, &gc_pending_media_set_ids);
                if !colliding_media_set_ids.is_empty() {
                    return Err(ApiError::with_code(
                        "a durable media set id collided with existing authority",
                        DURABLE_MEDIA_ADMISSION_CONFLICT,
                        StatusCode::INTERNAL_SERVER_ERROR,
                    ));
                }
                let detail = existing_by_id(state, &detail.batch.id)
                    .await?
                    .ok_or_else(|| ApiError::internal("idempotent generation batch disappeared"))?;
                self.verify_existing(&body.client_batch_id, &fingerprint, &detail)?;
                Ok(DurableAdmissionOutcome {
                    status_code: StatusCode::OK,
                    status: crate::routes::generation_batch_status(&state.instance_id, detail),
                    observer: None,
                    warnings: None,
                })
            }
        }
    }

    fn verify_existing(
        &self,
        operation_id: &str,
        fingerprint: &QueueMediaOperationFingerprint,
        detail: &mold_db::generation_batches::DurableGenerationBatchDetail,
    ) -> Result<(), ApiError> {
        let receipt = QueueMediaOperationReceipt::parse(detail.batch.request_sha256.clone())
            .map_err(|_| identity_undecidable())?;
        let existing = self
            .lifecycle
            .open_operation_receipt(operation_id, &receipt)
            .map_err(|_| identity_undecidable())?;
        if existing.constant_time_eq(fingerprint) {
            Ok(())
        } else {
            Err(ApiError::with_code(
                "client_batch_id was already used for a different request",
                "GENERATION_BATCH_IDEMPOTENCY_CONFLICT",
                StatusCode::CONFLICT,
            ))
        }
    }

    fn cleanup_losers(&self, refs: &[MediaSetRef], gc_pending_ids: &[String]) {
        let pending = gc_pending_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        for media_set in refs
            .iter()
            .filter(|set| pending.contains(set.set_id.as_str()))
        {
            let cleanup = self
                .lifecycle
                .candidate_for_ref(media_set.clone())
                .and_then(|candidate| self.lifecycle.cleanup_after_committed_delete(&candidate));
            if let Err(error) = cleanup {
                tracing::warn!(
                    media_set = %media_set.set_id,
                    %error,
                    "idempotency-loser media remains GC-pending"
                );
            }
        }
    }

    fn delete_unpublished(&self, refs: &[MediaSetRef]) {
        for media_set in refs {
            if let Err(error) = self.lifecycle.delete_unpublished(media_set) {
                tracing::warn!(
                    media_set = %media_set.set_id,
                    %error,
                    "unpublished queue media will be reconciled at startup"
                );
            }
        }
    }
}

pub(crate) fn request_has_durable_media(request: &mold_core::GenerateRequest) -> bool {
    request.source_image.is_some()
        || request.source_image_name.is_some()
        || request.id_image.is_some()
        || request.id_image_name.is_some()
        || request.id_images.is_some()
        || request.id_image_names.is_some()
        || request.edit_images.is_some()
        || request.mask_image.is_some()
        || request.control_image.is_some()
        || request.audio_file.is_some()
        || request.audio_file_path.is_some()
        || request.source_video.is_some()
        || request.source_video_path.is_some()
        || request.extend_video.is_some()
        || request.extend_video_path.is_some()
        || request.keyframes.is_some()
}

/// Requests that must enter the durable-media admission boundary, including
/// forms that the protocol refuses before extraction because their authority
/// cannot survive process restart.
pub(crate) fn request_requires_durable_media_admission(
    request: &mold_core::GenerateRequest,
) -> bool {
    request_has_durable_media(request)
        || request.references.is_some()
        || request.hdr_exr_dir.is_some()
        || mold_core::minimax_h3::capability_contract_for_model(&request.model).is_some()
}

fn durable_media_preflight(request: &mold_core::GenerateRequest) -> Result<(), ApiError> {
    if mold_core::minimax_h3::capability_contract_for_model(&request.model).is_some() {
        return Err(typed_refusal(
            "DURABLE_MEDIA_H3_UNSUPPORTED",
            "MiniMax H3 requests require process-private replay authority",
        ));
    }
    if request.references.is_some() {
        return Err(typed_refusal(
            "DURABLE_MEDIA_REFERENCES_UNSUPPORTED",
            "ordered references require temporary replay authority",
        ));
    }
    if request.hdr_exr_dir.is_some() {
        return Err(typed_refusal(
            "DURABLE_MEDIA_HDR_UNSUPPORTED",
            "HDR output authority is not supported by durable media protocol v1",
        ));
    }
    if request_has_durable_media(request) && (request.lora.is_some() || request.loras.is_some()) {
        return Err(typed_refusal(
            "DURABLE_MEDIA_LORA_UNSUPPORTED",
            "media and LoRA inputs cannot share durable media protocol v1",
        ));
    }
    Ok(())
}

fn typed_refusal(code: &'static str, message: &'static str) -> ApiError {
    ApiError::with_code(message, code, StatusCode::UNPROCESSABLE_ENTITY)
}

fn extraction_error(offset: usize, error: crate::queue_media::QueueMediaError) -> ApiError {
    ApiError::with_code(
        format!("requests[{}]: {error}", offset + 1),
        "DURABLE_MEDIA_UNSUPPORTED",
        StatusCode::UNPROCESSABLE_ENTITY,
    )
}

fn identity_undecidable() -> ApiError {
    ApiError::with_code(
        "the stored durable media operation receipt cannot be authenticated",
        DURABLE_MEDIA_IDENTITY_UNDECIDABLE,
        StatusCode::SERVICE_UNAVAILABLE,
    )
}

async fn existing_by_client(
    state: &AppState,
    client_id: &str,
) -> Result<Option<mold_db::generation_batches::DurableGenerationBatchDetail>, ApiError> {
    let journal = state.queue_journal.clone();
    let client_id = client_id.to_string();
    tokio::task::spawn_blocking(move || journal.durable_generation_batch_by_client(&client_id))
        .await
        .map_err(|error| ApiError::internal(format!("generation batch DB task failed: {error}")))?
        .map_err(|error| ApiError::internal(format!("generation batch DB lookup failed: {error}")))
}

async fn existing_by_id(
    state: &AppState,
    batch_id: &str,
) -> Result<Option<mold_db::generation_batches::DurableGenerationBatchDetail>, ApiError> {
    let journal = state.queue_journal.clone();
    let batch_id = batch_id.to_string();
    tokio::task::spawn_blocking(move || journal.durable_generation_batch(&batch_id))
        .await
        .map_err(|error| ApiError::internal(format!("generation batch DB task failed: {error}")))?
        .map_err(|error| ApiError::internal(format!("generation batch DB lookup failed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "media admission",
            "model": "mock-model",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 1,
            "output_format": "png"
        }))
        .unwrap()
    }

    #[test]
    fn media_free_lora_sibling_is_allowed_but_media_plus_lora_is_typed_refused() {
        let mut media_free = request();
        media_free.lora = Some(mold_core::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,
            expert: None,
        });
        assert!(durable_media_preflight(&media_free).is_ok());

        media_free.source_image = Some(vec![1, 2, 3]);
        let refusal = durable_media_preflight(&media_free).unwrap_err();
        assert_eq!(refusal.code, "DURABLE_MEDIA_LORA_UNSUPPORTED");
    }

    #[test]
    fn h3_references_and_hdr_have_stable_protocol_refusals() {
        let mut h3 = request();
        h3.model = mold_core::minimax_h3::FL2VA_COMFY.to_string();
        assert_eq!(
            durable_media_preflight(&h3).unwrap_err().code,
            "DURABLE_MEDIA_H3_UNSUPPORTED"
        );

        let mut references = request();
        references.references = Some(vec![mold_core::GenerationReference::Image {
            media: mold_core::GenerationReferenceAuthority::Inline { data: vec![1] },
            provenance: mold_core::GenerationReferenceProvenance::default(),
            mime_type: "image/png".to_string(),
            width: 1,
            height: 1,
        }]);
        assert_eq!(
            durable_media_preflight(&references).unwrap_err().code,
            "DURABLE_MEDIA_REFERENCES_UNSUPPORTED"
        );

        let mut hdr = request();
        hdr.hdr_exr_dir = Some("private-output".to_string());
        assert_eq!(
            durable_media_preflight(&hdr).unwrap_err().code,
            "DURABLE_MEDIA_HDR_UNSUPPORTED"
        );
    }
}
