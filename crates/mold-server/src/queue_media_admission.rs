//! Canonical encrypted-media admission shared by batch and direct routes.

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::Arc;

use axum::http::StatusCode;
use sha2::{Digest as _, Sha256};

use crate::queue_journal::{MediaBatchJournalAdmission, MediaJournalAdmission};
use crate::queue_media_ingress::{ObserverMode, ObserverRegistration, QueueMediaIngress};
use crate::queue_media_lifecycle::QueueMediaLifecycle;
use crate::queue_media_store::{
    MediaSetRef, QueueMediaOperationFingerprint, QueueMediaOperationReceipt,
};
use crate::routes::{ApiError, RequestWarnings};
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
    /// One slot per admitted child. `None` means admission stayed durable but
    /// the bounded attached-observer registry was full for that child.
    pub observers: Vec<Option<ObserverRegistration>>,
    pub warnings: Option<RequestWarnings>,
}

struct PreparedChild {
    request: mold_core::GenerateRequest,
    output_dir: std::path::PathBuf,
    preferred_gpu: Option<usize>,
    authority: Option<crate::durable_admission_authority::CapturedAuthority>,
}

struct FingerprintWriter(Sha256);

impl FingerprintWriter {
    fn new() -> Self {
        Self(Sha256::new())
    }

    fn finish(self) -> QueueMediaOperationFingerprint {
        QueueMediaOperationFingerprint::from_sha256_v1_digest(self.0.finalize().into())
    }
}

impl std::io::Write for FingerprintWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct SealInput {
    offset: usize,
    id: String,
    request: mold_core::GenerateRequest,
    output_dir: std::path::PathBuf,
    target_gpu: Option<usize>,
    target_device_id: Option<String>,
    completion_payload: SseCompletionPayload,
    admission_authority: Option<Vec<u8>>,
    durable_replacement: Option<crate::queue_media::ProcessPrivateAuthority>,
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
    admission_authority: Option<String>,
}

struct BlockingSealedBatch {
    lifecycle: Arc<QueueMediaLifecycle>,
    children: Vec<SealedChild>,
    media_sets: Vec<MediaSetRef>,
    receipt: Option<QueueMediaOperationReceipt>,
    cleanup_armed: bool,
}

impl BlockingSealedBatch {
    fn disarm_cleanup(&mut self) {
        self.cleanup_armed = false;
    }

    fn cleanup_losers(&self, gc_pending_ids: &[String]) {
        let pending = gc_pending_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        for media_set in self
            .media_sets
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
}

impl Drop for BlockingSealedBatch {
    fn drop(&mut self) {
        if !self.cleanup_armed {
            return;
        }
        for media_set in &self.media_sets {
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

enum BlockingRecordOutcome {
    Inserted,
    Existing {
        batch_id: String,
        colliding_media_set_ids: Vec<String>,
    },
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
        body.client_batch_id = crate::routes::canonical_client_batch_id(&body.client_batch_id)?;
        if body.requests.is_empty() {
            return Err(ApiError::validation(
                "requests must contain at least one child",
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
        for (offset, request) in body.requests.iter_mut().enumerate() {
            mold_core::minimax_h3::canonicalize_request_model(request);
            if request.batch_size != 1 {
                return Err(ApiError::validation(format!(
                    "requests[{}].batch_size must be 1",
                    offset + 1
                )));
            }
        }
        normalize_batch_provenance(&mut body.requests, &body.client_batch_id)?;
        // Protocol-level authority refusals are request facts, not stored
        // operation state. Resolve them before even a read-only SQLite lookup;
        // in particular an HDR output path must never cross the DB boundary.
        durable_media_batch_preflight(&body.requests)?;
        // Hash JSON directly into SHA-256. A batch may contain large source
        // media, so materializing the full canonical JSON would multiply peak
        // admission memory before the encrypted media store can stream it.
        let mut fingerprint = FingerprintWriter::new();
        serde_json::to_writer(&mut fingerprint, &body.requests)
            .map_err(|error| ApiError::internal(format!("batch serialization failed: {error}")))?;
        let output_dir = {
            let config = state.config.read().await;
            if state.is_output_disabled(&config) {
                return Err(ApiError::validation(
                    "durable admission requires server gallery output",
                ));
            }
            config.effective_output_dir()
        };
        let mut prepared = Vec::with_capacity(body.requests.len());
        for (offset, mut request) in body.requests.into_iter().enumerate() {
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            if mold_core::minimax_h3::task_for_model(&request.model).is_some() {
                request.normalise_output_format(Some(mold_core::minimax_h3::FAMILY));
            }
            crate::routes::apply_default_metadata_setting(state, &mut request).await;
            crate::routes::normalize_generation_placement(state, &mut request).await;
            let preferred_gpu =
                crate::routes::validate_multi_gpu_placement(state, request.placement.as_ref())?;
            // Authority binds the exact deterministic request persisted below.
            let authority = crate::durable_admission_authority::capture(
                &request,
                authenticated,
                state.instance_id.as_str(),
            )
            .map_err(|mut error| {
                error.error = format!("requests[{}]: {}", offset + 1, error.error);
                error
            })?;
            let validation = if authority.is_some() {
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                {
                    mold_core::validation::validate_h3_private_uat_request(&request)
                }
                #[cfg(not(any(feature = "h3", feature = "h3-private-uat")))]
                {
                    mold_core::validate_generate_request_fields(&request, None)
                }
            } else {
                mold_core::validate_generate_request_fields(&request, None)
            };
            validation.map_err(|error| {
                ApiError::validation(format!("requests[{}]: {error}", offset + 1))
            })?;
            prepared.push(PreparedChild {
                request,
                output_dir: output_dir.clone(),
                preferred_gpu,
                authority,
            });
        }

        // H3 idempotency is identity-bound: the same client key and request
        // submitted under a different authenticated identity must conflict,
        // not inherit the first caller's durable authority.
        for child in &prepared {
            if let Some(authority) = &child.authority {
                fingerprint
                    .write_all(authority.idempotency_subject_sha256.as_bytes())
                    .map_err(|error| {
                        ApiError::internal(format!("batch fingerprint failed: {error}"))
                    })?;
            }
        }
        let fingerprint = fingerprint.finish();

        if let Some(existing) = existing_by_client(state, &body.client_batch_id).await? {
            self.verify_existing_async(&body.client_batch_id, &fingerprint, &existing)
                .await?;
            if observer_mode.is_some() {
                return Err(ApiError::with_code(
                    "this operation is already durable; reconcile it through the queue status endpoint",
                    "DIRECT_OPERATION_ALREADY_ADMITTED",
                    StatusCode::CONFLICT,
                ));
            }
            return Ok(DurableAdmissionOutcome {
                status_code: StatusCode::OK,
                status: crate::routes::generation_batch_status(&state.instance_id, existing),
                observers: Vec::new(),
                warnings: None,
            });
        }

        let batch_id = uuid::Uuid::new_v4().to_string();
        let job_ids = (0..prepared.len())
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect::<Vec<_>>();
        let mut seal_inputs = Vec::with_capacity(prepared.len());
        let direct_warnings = observer_mode.map(|_| RequestWarnings::default());
        for (offset, (prepared, job_id)) in prepared.into_iter().zip(&job_ids).enumerate() {
            let PreparedChild {
                request,
                output_dir,
                preferred_gpu,
                authority,
            } = prepared;
            let (admission_authority, durable_replacement) = authority
                .map(|authority| (Some(authority.envelope), Some(authority.replaces)))
                .unwrap_or((None, None));
            let target_gpu = preferred_gpu;
            let target_device_id =
                crate::queue_journal::stable_device_id_for_ordinal(state, target_gpu);
            seal_inputs.push(SealInput {
                offset,
                id: job_id.clone(),
                request,
                output_dir,
                target_gpu,
                target_device_id,
                completion_payload,
                admission_authority,
                durable_replacement,
            });
        }

        let lifecycle = Arc::clone(&self.lifecycle);
        let fingerprint_for_seal = fingerprint.clone();
        let operation_id = body.client_batch_id.clone();
        let observers = job_ids
            .iter()
            .map(|job_id| observer_mode.and_then(|mode| self.ingress.reserve(job_id, mode)))
            .collect::<Vec<_>>();
        if observer_mode.is_some() && observers.iter().any(Option::is_none) {
            return Err(ApiError::with_code(
                "direct response capacity is full; retry before the request is admitted",
                "DIRECT_OBSERVER_CAPACITY_EXCEEDED",
                StatusCode::SERVICE_UNAVAILABLE,
            ));
        }
        let observer_job_ids = observers
            .iter()
            .zip(&job_ids)
            .filter_map(|(observer, job_id)| observer.as_ref().map(|_| job_id.clone()))
            .collect::<Vec<_>>();
        let journal = state.queue_journal.clone();
        let batch_id_for_db = batch_id.clone();
        let client_id_for_db = body.client_batch_id.clone();
        let observers_for_db = observer_job_ids;
        // One blocking operation owns extraction, safe-open, hashing,
        // encryption, fsync, file-first cleanup, and the committing DB
        // transaction. Cancellation detaches this operation but cannot strand
        // its armed cleanup guard between two await points.
        let outcome =
            spawn_admission_blocking("media sealing and generation batch DB", move || {
                let mut sealed = seal_batch_blocking(
                    lifecycle,
                    &operation_id,
                    &fingerprint_for_seal,
                    seal_inputs,
                )?;
                let children = sealed
                    .children
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
                        admission_authority: child.admission_authority.as_deref(),
                    })
                    .collect::<Vec<_>>();
                match journal.record_batch_with_media(MediaBatchJournalAdmission {
                    id: &batch_id_for_db,
                    client_batch_id: &client_id_for_db,
                    operation_receipt: sealed
                        .receipt
                        .as_ref()
                        .expect("sealed batch has an operation receipt")
                        .as_str(),
                    children: &children,
                    observer_job_ids: &observers_for_db,
                }) {
                    Ok(
                        mold_db::generation_batches::GenerationBatchMediaInsertOutcome::Inserted(_),
                    ) => {
                        sealed.disarm_cleanup();
                        Ok(BlockingRecordOutcome::Inserted)
                    }
                    Ok(
                        mold_db::generation_batches::GenerationBatchMediaInsertOutcome::Existing {
                            detail,
                            gc_pending_media_set_ids,
                            colliding_media_set_ids,
                        },
                    ) => {
                        sealed.cleanup_losers(&gc_pending_media_set_ids);
                        sealed.disarm_cleanup();
                        Ok(BlockingRecordOutcome::Existing {
                            batch_id: detail.batch.id,
                            colliding_media_set_ids,
                        })
                    }
                    Err(message) => Err(ApiError::with_code(
                        message,
                        "GENERATION_BATCH_NOT_DURABLE",
                        StatusCode::UNPROCESSABLE_ENTITY,
                    )),
                }
            })
            .await??;

        match outcome {
            BlockingRecordOutcome::Inserted => {
                for (prompt, negative, model) in &typed_history {
                    crate::routes::record_prompt_history(state, prompt, negative.as_deref(), model);
                }
                let detail = existing_by_id(state, &batch_id).await?.ok_or_else(|| {
                    ApiError::internal("generation batch disappeared after durable media admission")
                })?;
                Ok(DurableAdmissionOutcome {
                    status_code: StatusCode::ACCEPTED,
                    status: crate::routes::generation_batch_status(&state.instance_id, detail),
                    observers,
                    warnings: direct_warnings,
                })
            }
            BlockingRecordOutcome::Existing {
                batch_id,
                colliding_media_set_ids,
            } => {
                drop(observers);
                if !colliding_media_set_ids.is_empty() {
                    return Err(ApiError::with_code(
                        "a durable media set id collided with existing authority",
                        DURABLE_MEDIA_ADMISSION_CONFLICT,
                        StatusCode::INTERNAL_SERVER_ERROR,
                    ));
                }
                let detail = existing_by_id(state, &batch_id)
                    .await?
                    .ok_or_else(|| ApiError::internal("idempotent generation batch disappeared"))?;
                self.verify_existing_async(&body.client_batch_id, &fingerprint, &detail)
                    .await?;
                if observer_mode.is_some() {
                    return Err(ApiError::with_code(
                        "this operation is already durable; reconcile it through the queue status endpoint",
                        "DIRECT_OPERATION_ALREADY_ADMITTED",
                        StatusCode::CONFLICT,
                    ));
                }
                Ok(DurableAdmissionOutcome {
                    status_code: StatusCode::OK,
                    status: crate::routes::generation_batch_status(&state.instance_id, detail),
                    observers: Vec::new(),
                    warnings: None,
                })
            }
        }
    }

    async fn verify_existing_async(
        &self,
        operation_id: &str,
        fingerprint: &QueueMediaOperationFingerprint,
        detail: &mold_db::generation_batches::DurableGenerationBatchDetail,
    ) -> Result<(), ApiError> {
        let lifecycle = Arc::clone(&self.lifecycle);
        let operation_id = operation_id.to_string();
        let fingerprint = fingerprint.clone();
        let receipt = QueueMediaOperationReceipt::parse(detail.batch.request_sha256.clone())
            .map_err(|_| identity_undecidable())?;
        spawn_admission_blocking("operation receipt verification", move || {
            let existing = lifecycle
                .open_operation_receipt(&operation_id, &receipt)
                .map_err(|_| identity_undecidable())?;
            if existing.constant_time_eq(&fingerprint) {
                Ok(())
            } else {
                Err(ApiError::with_code(
                    "client_batch_id was already used for a different request",
                    "GENERATION_BATCH_IDEMPOTENCY_CONFLICT",
                    StatusCode::CONFLICT,
                ))
            }
        })
        .await?
    }
}

/// Admission identity and logical sibling identity are separate. A client may
/// split one large Batch N across several idempotent operations, so supplied
/// global provenance must survive each operation unchanged. Requests without
/// provenance receive a local operation-scoped group for older callers.
fn normalize_batch_provenance(
    requests: &mut [mold_core::GenerateRequest],
    client_batch_id: &str,
) -> Result<(), ApiError> {
    const MAX_LOGICAL_BATCH_ID_BYTES: usize = 128;
    let supplied = requests.iter().any(|request| {
        request.batch_id.is_some() || request.batch_index.is_some() || request.batch_count.is_some()
    });
    if !supplied {
        let count = u32::try_from(requests.len())
            .map_err(|_| ApiError::validation("requests contains too many children"))?;
        for (offset, request) in requests.iter_mut().enumerate() {
            request.batch_id = Some(client_batch_id.to_string());
            request.batch_index = Some(offset as u32 + 1);
            request.batch_count = Some(count);
        }
        return Ok(());
    }

    let mut logical_id: Option<&str> = None;
    let mut logical_count: Option<u32> = None;
    let mut indexes = std::collections::HashSet::with_capacity(requests.len());
    for (offset, request) in requests.iter().enumerate() {
        let (Some(batch_id), Some(batch_index), Some(batch_count)) = (
            request.batch_id.as_deref(),
            request.batch_index,
            request.batch_count,
        ) else {
            return Err(ApiError::validation(format!(
                "requests[{}] must provide batch_id, batch_index, and batch_count together",
                offset + 1
            )));
        };
        if batch_id.trim().is_empty() {
            return Err(ApiError::validation(format!(
                "requests[{}].batch_id must not be empty",
                offset + 1
            )));
        }
        if batch_id.len() > MAX_LOGICAL_BATCH_ID_BYTES || batch_id.chars().any(char::is_control) {
            return Err(ApiError::validation(format!(
                "requests[{}].batch_id must be at most {MAX_LOGICAL_BATCH_ID_BYTES} bytes and contain no control characters",
                offset + 1
            )));
        }
        if batch_index == 0 || batch_count == 0 || batch_index > batch_count {
            return Err(ApiError::validation(format!(
                "requests[{}] has invalid batch_index/batch_count provenance",
                offset + 1
            )));
        }
        if logical_id.is_some_and(|expected| expected != batch_id)
            || logical_count.is_some_and(|expected| expected != batch_count)
        {
            return Err(ApiError::validation(
                "all requests in one operation must share batch_id and batch_count",
            ));
        }
        if !indexes.insert(batch_index) {
            return Err(ApiError::validation(format!(
                "requests contains duplicate batch_index {batch_index}"
            )));
        }
        logical_id = Some(batch_id);
        logical_count = Some(batch_count);
    }
    Ok(())
}

pub(crate) fn request_has_durable_media(request: &mold_core::GenerateRequest) -> bool {
    crate::queue_media::request_has_extractable_media(request)
}

pub(crate) fn durable_media_preflight(
    request: &mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    if request_has_durable_media(request)
        && !crate::queue_media_store::QueueMediaStore::supports_mixed_hydration()
    {
        return Err(typed_refusal(
            "DURABLE_MEDIA_PLATFORM_UNSUPPORTED",
            "durable request media cannot be hydrated securely on this platform",
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
            "HDR output authority is not supported by the durable media protocol",
        ));
    }
    if request_has_durable_media(request) && (request.lora.is_some() || request.loras.is_some()) {
        return Err(typed_refusal(
            "DURABLE_MEDIA_LORA_UNSUPPORTED",
            "media and LoRA inputs cannot share the durable media protocol",
        ));
    }
    Ok(())
}

fn durable_media_batch_preflight(requests: &[mold_core::GenerateRequest]) -> Result<(), ApiError> {
    let batch_has_media = requests.iter().any(request_has_durable_media);
    for (offset, request) in requests.iter().enumerate() {
        durable_media_preflight(request).map_err(|mut error| {
            error.error = format!("requests[{}]: {}", offset + 1, error.error);
            error
        })?;
        if batch_has_media && (request.lora.is_some() || request.loras.is_some()) {
            return Err(ApiError::with_code(
                format!(
                    "requests[{}]: media batches cannot persist local LoRA authority",
                    offset + 1
                ),
                "DURABLE_MEDIA_LORA_UNSUPPORTED",
                StatusCode::UNPROCESSABLE_ENTITY,
            ));
        }
    }
    Ok(())
}

fn seal_batch_blocking(
    lifecycle: Arc<QueueMediaLifecycle>,
    operation_id: &str,
    fingerprint: &QueueMediaOperationFingerprint,
    inputs: Vec<SealInput>,
) -> Result<BlockingSealedBatch, ApiError> {
    let mut batch = BlockingSealedBatch {
        lifecycle,
        children: Vec::with_capacity(inputs.len()),
        media_sets: Vec::new(),
        receipt: None,
        cleanup_armed: true,
    };
    for input in inputs {
        let SealInput {
            offset,
            id,
            request,
            output_dir,
            target_gpu,
            target_device_id,
            completion_payload,
            admission_authority,
            durable_replacement,
        } = input;
        let model = request.model.clone();
        let seed_pinned = request.seed.is_some();
        let admission_authority = admission_authority
            .as_deref()
            .map(|payload| {
                batch
                    .lifecycle
                    .seal_admission_authority(&id, payload)
                    .map(|authority| authority.as_str().to_owned())
                    .map_err(|error| {
                        ApiError::internal(format!(
                            "requests[{}]: admission authority sealing failed: {error}",
                            offset + 1
                        ))
                    })
            })
            .transpose()?;
        let (request_json, media_set) = if request_has_durable_media(&request) {
            let authorities = crate::queue_media::ProcessPrivateAuthorities::none()
                .with_durable_replacement(durable_replacement);
            let extracted = crate::queue_media::extract_request_media(&id, request, &authorities)
                .map_err(|error| extraction_error(offset, error))?;
            let projection = crate::queue_media::project_request_media(extracted.media())
                .map_err(|error| extraction_error(offset, error))?;
            let (request_json, media) = extracted.into_parts();
            let seal_media = crate::queue_media::into_seal_media(media)
                .map_err(|error| extraction_error(offset, error))?;
            let reference = batch
                .lifecycle
                .seal_v2(&id, fingerprint, &projection, seal_media)
                .map_err(|error| {
                    ApiError::internal(format!(
                        "requests[{}]: encrypted media sealing failed: {error}",
                        offset + 1
                    ))
                })?;
            batch.media_sets.push(reference.clone());
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
        batch.children.push(SealedChild {
            id,
            model,
            request_json,
            media_set,
            output_dir,
            target_gpu,
            target_device_id,
            completion_payload,
            seed_pinned,
            admission_authority,
        });
    }
    batch.receipt = Some(
        batch
            .lifecycle
            .seal_operation_receipt(operation_id, fingerprint)
            .map_err(|error| {
                ApiError::internal(format!(
                    "durable media operation receipt could not be sealed: {error}"
                ))
            })?,
    );
    Ok(batch)
}

async fn spawn_admission_blocking<T, F>(label: &'static str, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| ApiError::internal(format!("{label} task failed: {error}")))
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
    fn chunk_admission_preserves_global_batch_provenance() {
        let mut requests = vec![request(), request()];
        for (offset, request) in requests.iter_mut().enumerate() {
            request.batch_id = Some("logical-batch".to_string());
            request.batch_index = Some(offset as u32 + 65);
            request.batch_count = Some(130);
        }
        normalize_batch_provenance(&mut requests, "operation-id").unwrap();
        assert_eq!(requests[0].batch_id.as_deref(), Some("logical-batch"));
        assert_eq!(requests[0].batch_index, Some(65));
        assert_eq!(requests[1].batch_index, Some(66));
        assert_eq!(requests[1].batch_count, Some(130));
    }

    #[test]
    fn partial_or_duplicate_batch_provenance_is_rejected() {
        let mut partial = vec![request()];
        partial[0].batch_id = Some("logical-batch".to_string());
        assert!(normalize_batch_provenance(&mut partial, "operation-id").is_err());

        let mut duplicate = vec![request(), request()];
        for request in &mut duplicate {
            request.batch_id = Some("logical-batch".to_string());
            request.batch_index = Some(1);
            request.batch_count = Some(2);
        }
        assert!(normalize_batch_provenance(&mut duplicate, "operation-id").is_err());

        let mut boundary = vec![request()];
        boundary[0].batch_id = Some("b".repeat(128));
        boundary[0].batch_index = Some(1);
        boundary[0].batch_count = Some(1);
        normalize_batch_provenance(&mut boundary, "operation-id").unwrap();

        for invalid in ["b".repeat(129), "batch\nheader".to_string()] {
            let mut request_with_invalid_id = vec![request()];
            request_with_invalid_id[0].batch_id = Some(invalid);
            request_with_invalid_id[0].batch_index = Some(1);
            request_with_invalid_id[0].batch_count = Some(1);
            assert!(
                normalize_batch_provenance(&mut request_with_invalid_id, "operation-id").is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn mixed_media_batch_refuses_media_free_lora_before_persistable_bytes_exist() {
        const SENTINEL: &str = "/private/local-adapter-sentinel.safetensors";
        let mut lora_sibling = request();
        lora_sibling.lora = Some(mold_core::LoraWeight {
            path: SENTINEL.to_string(),
            scale: 1.0,
            expert: None,
        });
        assert!(durable_media_batch_preflight(&[lora_sibling.clone()]).is_ok());

        let mut media = request();
        media.source_image = Some(vec![1, 2, 3]);
        let requests = vec![media, lora_sibling];
        let persistable = durable_media_batch_preflight(&requests)
            .map(|()| serde_json::to_vec(&requests).unwrap());
        let refusal = persistable.unwrap_err();
        assert_eq!(refusal.code, "DURABLE_MEDIA_LORA_UNSUPPORTED");
        assert!(!format!("{refusal:?}").contains(SENTINEL));
    }

    #[test]
    fn media_free_ordinary_lora_batch_remains_valid() {
        let mut lora = request();
        lora.lora = Some(mold_core::LoraWeight {
            path: "ordinary-adapter.safetensors".to_string(),
            scale: 1.0,
            expert: None,
        });
        assert!(durable_media_batch_preflight(&[lora]).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn media_plus_lora_remains_typed_refused() {
        let mut request = request();
        request.lora = Some(mold_core::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,
            expert: None,
        });
        request.source_image = Some(vec![1, 2, 3]);
        let refusal = durable_media_preflight(&request).unwrap_err();
        assert_eq!(refusal.code, "DURABLE_MEDIA_LORA_UNSUPPORTED");
    }

    #[cfg(windows)]
    #[test]
    fn windows_inline_media_admission_fails_closed() {
        let mut inline = request();
        inline.source_image = Some(vec![1, 2, 3]);
        assert_eq!(
            durable_media_preflight(&inline).unwrap_err().code,
            "DURABLE_MEDIA_PLATFORM_UNSUPPORTED"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_mixed_hydration_policy_is_enabled() {
        assert!(crate::queue_media_store::QueueMediaStore::supports_mixed_hydration());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn deliberately_blocked_seal_does_not_starve_async_status_work() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let sealing = tokio::spawn(spawn_admission_blocking("test media sealing", move || {
            started_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            17_u8
        }));
        tokio::time::timeout(std::time::Duration::from_secs(2), started_rx)
            .await
            .expect("the blocking seal started")
            .unwrap();

        let status = tokio::time::timeout(std::time::Duration::from_millis(250), async {
            tokio::task::yield_now().await;
            "ready"
        })
        .await
        .expect("the current-thread executor stayed responsive");
        assert_eq!(status, "ready");

        release_tx.send(()).unwrap();
        assert_eq!(sealing.await.unwrap().unwrap(), 17);
    }

    #[test]
    fn h3_is_admissible_while_references_and_hdr_keep_stable_refusals() {
        let mut h3 = request();
        h3.model = mold_core::minimax_h3::FL2VA_COMFY.to_string();
        durable_media_preflight(&h3).expect("H3 uses sealed durable replay authority");

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
