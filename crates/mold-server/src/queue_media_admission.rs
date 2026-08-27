//! Canonical encrypted-media admission shared by batch and direct routes.

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::Arc;

use axum::http::StatusCode;
use hmac::{Hmac, Mac as _};
use sha2::{Digest as _, Sha256};
use subtle::ConstantTimeEq as _;
use zeroize::Zeroizing;

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
    receipt_key: Arc<Zeroizing<[u8; 32]>>,
    owner_uuid: String,
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
    /// The resolver's staged reference media, alive only until it is sealed.
    staged_references: Option<crate::reference_uploads::StagedReferences>,
}

struct FingerprintWriter(Sha256);

const DURABLE_OPERATION_RECEIPT_PREFIX: &str = "generation-v2";

enum DurableReceiptVerification {
    Match,
    Conflict,
    Invalid,
}

fn durable_operation_receipt(
    key: &[u8; 32],
    owner_uuid: &str,
    operation_id: &str,
    fingerprint: &QueueMediaOperationFingerprint,
) -> String {
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let opaque_fingerprint =
        opaque_operation_fingerprint(key, owner_uuid, operation_id, fingerprint);
    let digest = durable_operation_receipt_mac(
        key,
        owner_uuid,
        operation_id,
        fingerprint.version(),
        &opaque_fingerprint,
        &nonce,
    );
    format!(
        "{DURABLE_OPERATION_RECEIPT_PREFIX}.{nonce}.{:04x}.{}.{}",
        fingerprint.version(),
        opaque_fingerprint,
        hex_bytes(&digest)
    )
}

fn opaque_operation_fingerprint(
    key: &[u8; 32],
    owner_uuid: &str,
    operation_id: &str,
    fingerprint: &QueueMediaOperationFingerprint,
) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts a 32-byte key");
    mac.update(b"mold durable generation opaque fingerprint\0");
    update_length_delimited(&mut mac, owner_uuid.as_bytes());
    update_length_delimited(&mut mac, operation_id.as_bytes());
    mac.update(&fingerprint.version().to_be_bytes());
    update_length_delimited(&mut mac, fingerprint.sha256_hex().as_bytes());
    hex_bytes(&mac.finalize().into_bytes())
}

fn update_length_delimited(mac: &mut Hmac<Sha256>, value: &[u8]) {
    mac.update(&(value.len() as u64).to_be_bytes());
    mac.update(value);
}

fn durable_operation_receipt_mac(
    key: &[u8; 32],
    owner_uuid: &str,
    operation_id: &str,
    fingerprint_version: u16,
    opaque_fingerprint: &str,
    nonce: &str,
) -> [u8; 32] {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts a 32-byte key");
    mac.update(b"mold durable generation operation receipt\0");
    update_length_delimited(&mut mac, owner_uuid.as_bytes());
    update_length_delimited(&mut mac, operation_id.as_bytes());
    update_length_delimited(&mut mac, nonce.as_bytes());
    mac.update(&fingerprint_version.to_be_bytes());
    update_length_delimited(&mut mac, opaque_fingerprint.as_bytes());
    mac.finalize().into_bytes().into()
}

fn verify_durable_operation_receipt(
    key: &[u8; 32],
    receipt: &str,
    owner_uuid: &str,
    operation_id: &str,
    fingerprint: &QueueMediaOperationFingerprint,
) -> Option<DurableReceiptVerification> {
    let mut parts = receipt.split('.');
    if parts.next()? != DURABLE_OPERATION_RECEIPT_PREFIX {
        return None;
    }
    let nonce = parts.next()?;
    let stored_version = parts.next()?;
    let stored_fingerprint_token = parts.next()?;
    let received_mac = parts.next()?;
    if parts.next().is_some()
        || nonce.len() != 32
        || !nonce.bytes().all(|byte| byte.is_ascii_hexdigit())
        || stored_version.len() != 4
        || !stored_version.bytes().all(|byte| byte.is_ascii_hexdigit())
        || stored_fingerprint_token.len() != 64
        || !stored_fingerprint_token
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || received_mac.len() != 64
        || !received_mac.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Some(DurableReceiptVerification::Invalid);
    }
    let Ok(version) = u16::from_str_radix(stored_version, 16) else {
        return Some(DurableReceiptVerification::Invalid);
    };
    let expected = hex_bytes(&durable_operation_receipt_mac(
        key,
        owner_uuid,
        operation_id,
        version,
        stored_fingerprint_token,
        nonce,
    ));
    if !bool::from(expected.as_bytes().ct_eq(received_mac.as_bytes())) {
        return Some(DurableReceiptVerification::Invalid);
    }
    let incoming_token = opaque_operation_fingerprint(key, owner_uuid, operation_id, fingerprint);
    Some(
        if version == fingerprint.version()
            && bool::from(
                stored_fingerprint_token
                    .as_bytes()
                    .ct_eq(incoming_token.as_bytes()),
            )
        {
            DurableReceiptVerification::Match
        } else {
            DurableReceiptVerification::Conflict
        },
    )
}

fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

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

#[cfg(test)]
pub(crate) fn test_operation_fingerprint(
    operation_id: &str,
    mut requests: Vec<mold_core::GenerateRequest>,
) -> QueueMediaOperationFingerprint {
    for request in &mut requests {
        mold_core::minimax_h3::canonicalize_request_model(request);
    }
    normalize_batch_provenance(&mut requests, operation_id, false).unwrap();
    let mut writer = FingerprintWriter::new();
    serde_json::to_writer(&mut writer, &requests).unwrap();
    writer.finish()
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
    staged_references: Option<crate::reference_uploads::StagedReferences>,
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
    receipt: Option<String>,
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
    pub(crate) fn new(
        lifecycle: Arc<QueueMediaLifecycle>,
        queue_capacity: usize,
        receipt_evidence_exists: bool,
    ) -> Result<Arc<Self>, crate::queue_media_store::QueueMediaError> {
        let receipt_key = crate::queue_media_store::QueueMediaStore::generation_admission_key(
            lifecycle.mold_home(),
            receipt_evidence_exists,
        )?;
        let owner_uuid = lifecycle.owner_uuid().to_string();
        Ok(Arc::new(Self {
            lifecycle,
            ingress: QueueMediaIngress::new(queue_capacity),
            receipt_key: Arc::new(receipt_key),
            owner_uuid,
        }))
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
        reference_identity: Option<crate::reference_uploads::ReferenceIdentity>,
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
        // A host on its way down admits nothing new. The retention fence is up
        // before the scheduler is cancelled, so this is the moment to stop
        // taking work: accepting here would journal a print whose only
        // delivery is a queue about to be drained, and the caller would get a
        // `202` instead of the `Retry-After` that tells it to come back.
        if state.queue_journal.is_retaining() {
            return Err(crate::routes::ApiError::server_restarting(
                "server is restarting; this generation was not accepted",
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
        let direct_one_shot = observer_mode.is_some() && body.requests.len() == 1;
        normalize_batch_provenance(&mut body.requests, &body.client_batch_id, direct_one_shot)?;
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
        // H3 idempotency is identity-bound: the same client key and request
        // submitted under a different authenticated identity must conflict,
        // not inherit the first caller's durable authority.
        for (offset, request) in body.requests.iter().enumerate() {
            if let Some(subject) = crate::durable_admission_authority::idempotency_subject_sha256(
                request,
                authenticated,
                state.instance_id.as_str(),
            )
            .map_err(|mut error| {
                error.error = format!("requests[{}]: {}", offset + 1, error.error);
                error
            })? {
                fingerprint.write_all(subject.as_bytes()).map_err(|error| {
                    ApiError::internal(format!("batch fingerprint failed: {error}"))
                })?;
            }
        }
        let fingerprint = fingerprint.finish();

        // A duplicate POST is answered BEFORE any child is resolved: a
        // reference upload session is consumed exactly once, so the retry of
        // a lost response must find its batch here rather than burn the
        // session the first attempt already spent.
        if let Some(existing) = existing_by_client(state, &body.client_batch_id).await? {
            self.verify_existing_async(state, &body.client_batch_id, &fingerprint, &existing)
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

        let (output_dir, media_roots) = {
            let config = state.config.read().await;
            if state.is_output_disabled(&config) {
                return Err(ApiError::validation(
                    "durable admission requires server gallery output",
                ));
            }
            (config.effective_output_dir(), config.resolved_media_roots())
        };
        let mut prepared = Vec::with_capacity(body.requests.len());
        for (offset, mut request) in body.requests.into_iter().enumerate() {
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            if mold_core::minimax_h3::task_for_model(&request.model).is_some() {
                request.normalise_output_format(Some(mold_core::minimax_h3::FAMILY));
            }
            // An upload session is bound to the request scope the client
            // created it for, so the scope is frozen here, before this host
            // stamps its own defaults onto the request.
            let reference_scope_sha256 = request
                .references
                .as_ref()
                .map(|_| state.reference_uploads.scope_sha256(&request))
                .transpose()?;
            crate::routes::apply_default_metadata_setting(state, &mut request).await;
            crate::routes::normalize_generation_placement(state, &mut request).await;
            let preferred_gpu =
                crate::routes::validate_multi_gpu_placement(state, request.placement.as_ref())?;
            let private_ingress =
                crate::durable_admission_authority::claims_private_ingress(&request);
            // Model activation is asked BEFORE the row is durably accepted,
            // because its answers are HTTP contracts a client acts on rather
            // than transient conditions worth replaying: `451` for a
            // compliance-gated family, `501 MINIMAX_H3_RUNTIME_UNAVAILABLE`
            // for a row this build can download but not run, `400` for a model
            // nobody has. Deferring them to preparation would turn a
            // documented refusal into an accepted job that quietly holds.
            // Config-only work, exactly like the LTX-2 control contracts
            // resolved above it. A private-H3 ingress carries its own
            // authority and is deliberately skipped, mirroring
            // `prepare_generation`.
            //
            // Asked BEFORE field validation, because activation is a property
            // of the MODEL and does not depend on the request's shape: telling
            // a caller its `strength` is wrong for a checkpoint this build
            // cannot run at all answers the wrong question.
            crate::routes::reject_client_supplied_hdr_output(&request).map_err(|mut error| {
                error.error = format!("requests[{}]: {}", offset + 1, error.error);
                error
            })?;
            if !private_ingress {
                let family = crate::routes::require_server_model_activation(state, &request.model)
                    .await
                    .map_err(|mut error| {
                        error.error = format!("requests[{}]: {}", offset + 1, error.error);
                        error
                    })?;
                crate::routes::require_server_generation_request_activation(
                    state,
                    &request,
                    family.as_deref(),
                )
                .await
                .map_err(|mut error| {
                    error.error = format!("requests[{}]: {}", offset + 1, error.error);
                    error
                })?;
            }
            let validation = if private_ingress {
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
            // Every public reference authority — one-use upload handles,
            // inline bytes, server paths — is consumed here and rewritten to
            // a descriptor. This is the LAST fallible step before capture, so
            // a refused request never spends a session, and the FIRST thing
            // the request is serialized after, so a handle never reaches
            // durable JSON.
            let staged_references = state
                .reference_uploads
                .resolve_request(
                    reference_identity.as_ref(),
                    &mut request,
                    &media_roots,
                    reference_scope_sha256.as_deref(),
                )
                .await
                .map_err(|mut error| {
                    error.error = format!("requests[{}]: {}", offset + 1, error.error);
                    error
                })?;
            // Authority binds the exact deterministic request persisted below
            // — the descriptor form, which is what every later consumer
            // re-hashes against it.
            let authority = crate::durable_admission_authority::capture(
                &request,
                authenticated,
                state.instance_id.as_str(),
            )
            .map_err(|mut error| {
                error.error = format!("requests[{}]: {}", offset + 1, error.error);
                error
            })?;
            prepared.push(PreparedChild {
                request,
                output_dir: output_dir.clone(),
                preferred_gpu,
                authority,
                staged_references,
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
                staged_references,
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
                staged_references,
            });
        }

        let lifecycle = Arc::clone(&self.lifecycle);
        let receipt_key = Arc::clone(&self.receipt_key);
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
        let owner_uuid = self.owner_uuid.clone();
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
                    receipt_key.as_ref(),
                    &owner_uuid,
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
                self.verify_existing_async(state, &body.client_batch_id, &fingerprint, &detail)
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
        state: &AppState,
        operation_id: &str,
        fingerprint: &QueueMediaOperationFingerprint,
        detail: &mold_db::generation_batches::DurableGenerationBatchDetail,
    ) -> Result<(), ApiError> {
        if let Some(result) = verify_durable_operation_receipt(
            self.receipt_key.as_ref(),
            &detail.batch.request_sha256,
            &self.owner_uuid,
            operation_id,
            fingerprint,
        ) {
            return match result {
                DurableReceiptVerification::Match => Ok(()),
                DurableReceiptVerification::Conflict => Err(ApiError::with_code(
                    "client_batch_id was already used for a different request",
                    "GENERATION_BATCH_IDEMPOTENCY_CONFLICT",
                    StatusCode::CONFLICT,
                )),
                DurableReceiptVerification::Invalid => Err(identity_undecidable()),
            };
        }
        let lifecycle = Arc::clone(&self.lifecycle);
        let operation_id = operation_id.to_string();
        let fingerprint = fingerprint.clone();
        let v1_operation_id = operation_id.clone();
        let v1_fingerprint = fingerprint.clone();
        let receipt = QueueMediaOperationReceipt::parse(detail.batch.request_sha256.clone())
            .map_err(|_| identity_undecidable())?;
        spawn_admission_blocking("v1 operation receipt verification", move || {
            let existing = lifecycle
                .open_operation_receipt(&v1_operation_id, &receipt)
                .map_err(|_| identity_undecidable())?;
            if existing.constant_time_eq(&v1_fingerprint) {
                Ok(())
            } else {
                Err(ApiError::with_code(
                    "client_batch_id was already used for a different request",
                    "GENERATION_BATCH_IDEMPOTENCY_CONFLICT",
                    StatusCode::CONFLICT,
                ))
            }
        })
        .await??;

        let replacement = durable_operation_receipt(
            self.receipt_key.as_ref(),
            &self.owner_uuid,
            &operation_id,
            &fingerprint,
        );
        let journal = state.queue_journal.clone();
        let batch_id = detail.batch.id.clone();
        let expected = detail.batch.request_sha256.clone();
        let replacement_for_update = replacement.clone();
        let migrated = spawn_admission_blocking("v1 operation receipt migration", move || {
            journal.replace_generation_batch_receipt(&batch_id, &expected, &replacement_for_update)
        })
        .await?
        .map_err(ApiError::internal)?;
        if migrated {
            return Ok(());
        }

        let current = existing_by_id(state, &detail.batch.id)
            .await?
            .ok_or_else(identity_undecidable)?;
        match verify_durable_operation_receipt(
            self.receipt_key.as_ref(),
            &current.batch.request_sha256,
            &self.owner_uuid,
            &operation_id,
            &fingerprint,
        ) {
            Some(DurableReceiptVerification::Match) => Ok(()),
            Some(DurableReceiptVerification::Conflict) => Err(ApiError::with_code(
                "client_batch_id was already used for a different request",
                "GENERATION_BATCH_IDEMPOTENCY_CONFLICT",
                StatusCode::CONFLICT,
            )),
            _ => Err(identity_undecidable()),
        }
    }
}

/// Admission identity and logical sibling identity are separate. A client may
/// split one large Batch N across several idempotent operations, so supplied
/// global provenance must survive each operation unchanged. Requests without
/// provenance are ordinary siblings and are never stamped: `batch_id` /
/// `batch_index` / `batch_count` ARE the prepared-expansion contract, so only
/// a client that prepared variations supplies them.
fn normalize_batch_provenance(
    requests: &mut [mold_core::GenerateRequest],
    client_batch_id: &str,
    direct_one_shot: bool,
) -> Result<(), ApiError> {
    const MAX_LOGICAL_BATCH_ID_BYTES: usize = 128;
    let supplied = requests.iter().any(|request| {
        request.batch_id.is_some() || request.batch_index.is_some() || request.batch_count.is_some()
    });
    if !supplied {
        // Neither a direct one-shot nor a plain Batch N is a prepared set.
        // `batch_id`/`batch_index`/`batch_count` flow into `OutputMetadata`
        // and the Library, where they mean "this print was a prepared
        // variation" — synthesising them here makes that question
        // unanswerable to every gallery and reuse consumer and changes what a
        // plain `--batch 4` records. Only a caller that prepared variations
        // supplies them, and then they survive unchanged.
        let _ = (client_batch_id, direct_one_shot);
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

pub(crate) fn request_requires_encrypted_durable_media(
    request: &mold_core::GenerateRequest,
) -> bool {
    request_has_durable_media(request)
        || mold_core::minimax_h3::task_for_model(&request.model).is_some()
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
    // `references`, `hdr_exr_dir` and "a LoRA beside conditioning media" were
    // refused here by durable media protocol v1, which could not carry them.
    // None of the three was ever an invariant:
    //
    // * A LoRA is an ordinary request field. `lora.path` names a server-local
    //   adapter exactly as `model` names a checkpoint, it is persisted with
    //   the rest of the request, and dispatch re-validates it — a LoRA that
    //   vanished between admission and replay HOLDS its row by name rather
    //   than rendering without the adapter. Refusing the pair took out every
    //   img2img, inpaint, control and video-source render that used one.
    // * `hdr_exr_dir` is refused for a better, older reason that has nothing
    //   to do with durability: it names an output directory on the inference
    //   machine and an HTTP client may not choose one. That refusal lives in
    //   `routes::reject_client_supplied_hdr_output` and keeps its actionable
    //   "re-run with --local" wording.
    // * Ordered references are a descriptor plus media. The resolver consumes
    //   every one-use handle before the request is serialized and the bytes
    //   seal into the encrypted media set like any other source media, so
    //   the platform check above is the only durability question they raise.
    Ok(())
}

fn durable_media_batch_preflight(requests: &[mold_core::GenerateRequest]) -> Result<(), ApiError> {
    for (offset, request) in requests.iter().enumerate() {
        durable_media_preflight(request).map_err(|mut error| {
            error.error = format!("requests[{}]: {}", offset + 1, error.error);
            error
        })?;
    }
    Ok(())
}

fn seal_batch_blocking(
    lifecycle: Arc<QueueMediaLifecycle>,
    receipt_key: &[u8; 32],
    owner_uuid: &str,
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
            staged_references,
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
            let extracted = crate::queue_media::extract_request_media(
                &id,
                request,
                &authorities,
                staged_references.as_ref(),
            )
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
            // The encrypted set is now the only copy: releasing the staged
            // set returns its quota and unlinks the admission staging.
            drop(staged_references);
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
    batch.receipt = Some(durable_operation_receipt(
        receipt_key,
        owner_uuid,
        operation_id,
        fingerprint,
    ));
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

    #[test]
    fn v2_receipt_hides_raw_fingerprint_and_is_owner_bound() {
        let key = [7_u8; 32];
        let fingerprint = QueueMediaOperationFingerprint::sha256_v1(b"private request bytes");
        let receipt = durable_operation_receipt(&key, "owner-a", "operation-a", &fingerprint);
        let parts = receipt.split('.').collect::<Vec<_>>();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], DURABLE_OPERATION_RECEIPT_PREFIX);
        assert_eq!(parts[1].len(), 32);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 64);
        assert_eq!(parts[4].len(), 64);
        assert!(!receipt.contains(fingerprint.sha256_hex()));
        assert!(matches!(
            verify_durable_operation_receipt(
                &key,
                &receipt,
                "owner-a",
                "operation-a",
                &fingerprint
            ),
            Some(DurableReceiptVerification::Match)
        ));
        assert!(matches!(
            verify_durable_operation_receipt(
                &key,
                &receipt,
                "owner-b",
                "operation-a",
                &fingerprint
            ),
            Some(DurableReceiptVerification::Invalid)
        ));
    }

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
        normalize_batch_provenance(&mut requests, "operation-id", false).unwrap();
        assert_eq!(requests[0].batch_id.as_deref(), Some("logical-batch"));
        assert_eq!(requests[0].batch_index, Some(65));
        assert_eq!(requests[1].batch_index, Some(66));
        assert_eq!(requests[1].batch_count, Some(130));
    }

    #[test]
    fn partial_or_duplicate_batch_provenance_is_rejected() {
        let mut partial = vec![request()];
        partial[0].batch_id = Some("logical-batch".to_string());
        assert!(normalize_batch_provenance(&mut partial, "operation-id", false).is_err());

        let mut duplicate = vec![request(), request()];
        for request in &mut duplicate {
            request.batch_id = Some("logical-batch".to_string());
            request.batch_index = Some(1);
            request.batch_count = Some(2);
        }
        assert!(normalize_batch_provenance(&mut duplicate, "operation-id", false).is_err());

        let mut boundary = vec![request()];
        boundary[0].batch_id = Some("b".repeat(128));
        boundary[0].batch_index = Some(1);
        boundary[0].batch_count = Some(1);
        normalize_batch_provenance(&mut boundary, "operation-id", false).unwrap();

        for invalid in ["b".repeat(129), "batch\nheader".to_string()] {
            let mut request_with_invalid_id = vec![request()];
            request_with_invalid_id[0].batch_id = Some(invalid);
            request_with_invalid_id[0].batch_index = Some(1);
            request_with_invalid_id[0].batch_count = Some(1);
            assert!(normalize_batch_provenance(
                &mut request_with_invalid_id,
                "operation-id",
                false
            )
            .is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn mixed_media_batch_admits_a_lora_sibling() {
        let mut lora_sibling = request();
        lora_sibling.lora = Some(mold_core::LoraWeight {
            path: "/private/local-adapter.safetensors".to_string(),
            scale: 1.0,
            expert: None,
        });
        assert!(durable_media_batch_preflight(&[lora_sibling.clone()]).is_ok());

        let mut media = request();
        media.source_image = Some(vec![1, 2, 3]);
        assert!(durable_media_batch_preflight(&[media, lora_sibling]).is_ok());
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

    /// img2img with an adapter is an ordinary print; the durable store seals
    /// the LoRA record beside the media rather than refusing the pair.
    #[cfg(unix)]
    #[test]
    fn media_plus_lora_is_admitted() {
        let mut request = request();
        request.lora = Some(mold_core::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,
            expert: None,
        });
        request.source_image = Some(vec![1, 2, 3]);
        durable_media_preflight(&request).expect("media and a LoRA share the durable path");
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

    /// Nothing refuses here any more but the platform check: H3 uses sealed
    /// replay authority, ordered references seal like any other media,
    /// `hdr_exr_dir` is refused elsewhere by the older rule that an HTTP
    /// client may not name an output directory on the inference machine, and
    /// a LoRA beside media is an ordinary durable request.
    #[cfg(unix)]
    #[test]
    fn durable_preflight_refuses_nothing_the_encrypted_store_can_carry() {
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
        assert!(request_has_durable_media(&references));
        durable_media_preflight(&references)
            .expect("ordered references are descriptors plus sealed media");

        let mut hdr = request();
        hdr.hdr_exr_dir = Some("private-output".to_string());
        durable_media_preflight(&hdr)
            .expect("hdr_exr_dir is refused by the local-only rule, not by durability");

        let mut media_lora = request();
        media_lora.source_image = Some(vec![1, 2, 3]);
        media_lora.lora = Some(mold_core::LoraWeight {
            path: "adapter.safetensors".to_string(),
            scale: 1.0,
            expert: None,
        });
        durable_media_preflight(&media_lora)
            .expect("a LoRA beside media is an ordinary durable request");
        assert!(durable_media_batch_preflight(&[media_lora]).is_ok());
    }
}
