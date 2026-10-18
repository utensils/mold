//! Fresh-video candidate sessions share ordinary upload identity, lifetime,
//! staging and quota authority. No candidate is consumable before finalize.
use super::*;
use mold_core::video_normalization::*;

#[cfg(test)]
mod tests {
    use super::*;
    fn candidate() -> CandidateSessionRequest {
        serde_json::from_value(serde_json::json!({"protocol_version":1,"request":{"model":"minimax-h3-ref2va","prompt":"test","width":512,"height":512,"steps":4},"reference_candidates":[{"index":1,"type":"fresh_video","kind":"video","original":{"name":"a.mov","mime_type":"video/quicktime","sha256":"a".repeat(64),"byte_length":123}}]})).unwrap()
    }
    #[test]
    fn candidate_authority_is_ordered_fresh_and_bounded() {
        assert!(prepare_candidates(&candidate()).is_ok());
        let mut bad = candidate();
        if let ReferenceCandidate::FreshVideo { index, .. } = &mut bad.reference_candidates[0] {
            *index = 2;
        }
        assert!(prepare_candidates(&bad).is_err());
        let mut bad = candidate();
        bad.request.references = Some(vec![]);
        assert!(prepare_candidates(&bad).is_err());
        let mut bad = candidate();
        if let ReferenceCandidate::FreshVideo { original, .. } = &mut bad.reference_candidates[0] {
            original.byte_length = MAX_REFERENCE_UPLOAD_FILE_BYTES + 1;
        }
        assert!(prepare_candidates(&bad).is_err());
    }
    #[test]
    fn existing_canonical_path_is_preserved_separately_from_candidate_shape_validation() {
        let mut payload = candidate();
        let mut reference = placeholder_video(&OriginalVideo {
            name: Some("existing.mp4".into()),
            mime_type: "video/mp4".into(),
            byte_length: 100,
            sha256: "b".repeat(64),
        });
        if let GenerationReference::Video { media, .. } = &mut reference {
            *media = GenerationReferenceAuthority::ServerPath {
                path: "/allowed/reference.mp4".into(),
            };
        }
        payload
            .reference_candidates
            .push(ReferenceCandidate::Canonical {
                index: 2,
                upload: false,
                reference,
                byte_length: None,
            });
        let prepared = prepare_candidates(&payload).unwrap();
        assert_eq!(prepared.upload_references, vec![1]);
        assert!(matches!(
            prepared.request.references.unwrap()[1].media(),
            GenerationReferenceAuthority::Descriptor
        ));
    }

    #[test]
    fn v2_response_wrapper_preserves_the_original_typed_serialization() {
        let reference = placeholder_video(&OriginalVideo {
            name: None,
            mime_type: "video/mp4".into(),
            byte_length: 100,
            sha256: "a".repeat(64),
        });
        let response = ReferenceUploadCompleteResponse {
            instance_id: "instance-a".into(),
            reference: 1,
            metadata: reference.redacted_metadata(0).unwrap(),
            request_scope_sha256: "f".repeat(64),
            session_complete: true,
        };
        let expected = serde_json::to_vec(&response).unwrap();
        assert_eq!(
            expected,
            serde_json::to_vec(&ReferenceUploadResponseBody::Canonical(response)).unwrap()
        );
    }

    #[test]
    fn duplicate_json_keys_are_refused_at_every_depth() {
        assert!(parse_candidate_body(br#"{"request":{"model":"a","model":"b"}}"#).is_err());
    }
}

pub(super) struct NormalizationSession {
    id: String,
    candidate_scope: String,
    candidates: HashMap<u32, ReferenceCandidate>,
    receipts: HashMap<u32, NormalizationReceipt>,
    pub(super) finalized: bool,
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::with_code(
        message,
        "REFERENCE_NORMALIZATION_INVALID",
        StatusCode::UNPROCESSABLE_ENTITY,
    )
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn placeholder_video(original: &OriginalVideo) -> GenerationReference {
    GenerationReference::Video {
        media: GenerationReferenceAuthority::Descriptor,
        provenance: GenerationReferenceProvenance {
            name: original.name.clone(),
            sha256: Some(original.sha256.clone()),
            crop: None,
        },
        mime_type: "video/mp4".into(),
        width: 32,
        height: 32,
        frame_count: Some(48),
        duration_ms: minimax_h3::MIN_REFERENCE_DURATION_MS,
        fps: 24.0,
        has_audio: false,
        audio_duration_ms: None,
        audio_sample_count: None,
        audio_sample_rate: None,
        audio_channels: None,
    }
}
fn prepare_candidates(
    payload: &CandidateSessionRequest,
) -> Result<ReferenceUploadSessionRequest, ApiError> {
    if payload.protocol_version != VERSION
        || payload.request.references.is_some()
        || payload.request.batch_size > 1
    {
        return Err(invalid(
            "normalization v1 requires one captured request without references",
        ));
    }
    if minimax_h3::task_for_model(&payload.request.model) != Some(minimax_h3::Task::Ref2va) {
        return Err(invalid(
            "fresh video normalization requires an H3 Ref2VA request",
        ));
    }
    if payload.reference_candidates.is_empty()
        || payload.reference_candidates.len() > minimax_h3::MAX_REFERENCE_FILES
    {
        return Err(invalid("invalid reference candidate count"));
    }
    let mut references = Vec::new();
    let mut uploads = Vec::new();
    let mut fresh_count = 0;
    for (position, candidate) in payload.reference_candidates.iter().enumerate() {
        if candidate.index() as usize != position + 1 {
            return Err(invalid(
                "reference candidates must be ordered, unique and contiguous",
            ));
        }
        let index = candidate.index();
        match candidate {
            ReferenceCandidate::Canonical {
                upload,
                reference,
                byte_length,
                ..
            } => {
                if !reference
                    .provenance()
                    .sha256
                    .as_deref()
                    .is_some_and(valid_digest)
                {
                    return Err(invalid("canonical references require an exact SHA-256"));
                }
                if *upload {
                    if !matches!(reference.media(), GenerationReferenceAuthority::Descriptor)
                        || !byte_length
                            .is_some_and(|n| n > 0 && n <= MAX_REFERENCE_UPLOAD_FILE_BYTES)
                    {
                        return Err(invalid(
                            "canonical upload requires a descriptor and bounded byte length",
                        ));
                    }
                    uploads.push(index);
                } else if !matches!(
                    reference.media(),
                    GenerationReferenceAuthority::ServerPath { .. }
                ) || byte_length.is_some()
                {
                    return Err(invalid("non-upload candidates require an existing server path, never a prior upload lease"));
                }
                if let GenerationReferenceAuthority::ServerPath { path } = reference.media() {
                    if path.trim().is_empty()
                        || path.len() > minimax_h3::MAX_REFERENCE_PATH_BYTES
                        || path.contains('\0')
                    {
                        return Err(invalid("existing server reference path is malformed"));
                    }
                }
                references.push(reference_as_descriptor(reference));
            }
            ReferenceCandidate::FreshVideo { kind, original, .. } => {
                if *kind != mold_core::GenerationReferenceKind::Video
                    || !valid_digest(&original.sha256)
                    || original.byte_length == 0
                    || original.byte_length > MAX_REFERENCE_UPLOAD_FILE_BYTES
                    || !matches!(
                        original.mime_type.as_str(),
                        "video/mp4"
                            | "video/quicktime"
                            | "video/x-m4v"
                            | "video/webm"
                            | "video/x-matroska"
                    )
                {
                    return Err(invalid("fresh video requires a supported MIME hint, exact SHA-256 and bounded byte length"));
                }
                fresh_count += 1;
                uploads.push(index);
                // Private validation-only facts, never returned or consumable.
                references.push(placeholder_video(original));
            }
        }
    }
    if fresh_count == 0 {
        return Err(invalid("normalization session requires a fresh video"));
    }
    let mut request = payload.request.clone();
    request.references = Some(references);
    validate_session_descriptors(&request, None)?;
    Ok(ReferenceUploadSessionRequest {
        request,
        upload_references: uploads,
    })
}

// Recursive duplicate-key rejection before typed parsing. Hashing the raw body
// does not resolve ambiguous JSON objects interpreted differently by clients.
fn parse_candidate_body(body: &[u8]) -> Result<CandidateSessionRequest, ApiError> {
    use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
    struct Unique;
    impl<'de> DeserializeSeed<'de> for Unique {
        type Value = serde_json::Value;
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(self)
        }
    }
    impl<'de> Visitor<'de> for Unique {
        type Value = serde_json::Value;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("JSON with unique object keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
            let mut map = serde_json::Map::new();
            while let Some(key) = a.next_key::<String>()? {
                if map.contains_key(&key) {
                    return Err(serde::de::Error::custom("duplicate JSON object key"));
                }
                map.insert(key, a.next_value_seed(Unique)?);
            }
            Ok(map.into())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while let Some(value) = a.next_element_seed(Unique)? {
                values.push(value);
            }
            Ok(values.into())
        }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.into())
        }
        fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(v.into())
        }
        fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
            Ok(v.into())
        }
        fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v.into())
        }
        fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(v.into())
        }
        fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
            serde_json::Number::from_f64(v)
                .map(Into::into)
                .ok_or_else(|| E::custom("non-finite JSON number"))
        }
        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(serde_json::Value::Null)
        }
    }
    let mut decoder = serde_json::Deserializer::from_slice(body);
    let value = Unique
        .deserialize(&mut decoder)
        .map_err(|_| invalid("malformed candidate JSON or duplicate object keys"))?;
    decoder
        .end()
        .map_err(|_| invalid("trailing candidate JSON"))?;
    if value
        .get("request")
        .and_then(|request| request.as_object())
        .is_some_and(|request| request.contains_key("references"))
    {
        return Err(invalid("candidate request must omit references entirely"));
    }
    // Preserve exact-number parsing from the original bytes after uniqueness.
    serde_json::from_slice(body).map_err(|_| invalid("invalid candidate session fields"))
}

#[utoipa::path(
    post, path = "/api/generate/reference-normalization-sessions", tag = "generation",
    request_body = CandidateSessionRequest,
    params(("X-Mold-Reference-Candidate-Sha256" = String, Header, description = "Domain-separated SHA-256 of exact frozen JSON bytes")),
    responses((status = 200, description = "Fresh candidate session, not yet generation authority", body = CandidateSessionResponse), (status = 422, description = "Invalid candidates or fingerprint"))
)]
pub(crate) async fn create_normalization_session(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<impl IntoResponse, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    let fingerprint = candidate_scope_sha256(&body);
    if headers
        .get(CANDIDATE_DIGEST_HEADER)
        .and_then(|value| value.to_str().ok())
        != Some(fingerprint.as_str())
    {
        return Err(invalid("candidate body fingerprint mismatch"));
    }
    let mut payload = parse_candidate_body(&body)?;
    minimax_h3::canonicalize_request_model(&mut payload.request);
    let prepared = prepare_candidates(&payload)?;
    crate::routes::require_server_generation_request_activation(&state, &payload.request, None)
        .await?;
    validate_normalization_request(&prepared.request)?;
    if crate::video_reference_codec::capabilities().await.is_none() {
        return Err(ApiError::generation_unavailable("fresh video normalization tools are unavailable; install ffmpeg with H.264/AAC encoding and the advertised decoders"));
    }
    let response = state
        .reference_uploads
        .create_normalization(
            &identity,
            &state.instance_id,
            payload,
            prepared,
            fingerprint,
        )
        .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)))
}

impl ReferenceUploadStore {
    async fn create_normalization(
        &self,
        identity: &str,
        instance: &str,
        payload: CandidateSessionRequest,
        prepared: ReferenceUploadSessionRequest,
        fingerprint: String,
    ) -> Result<CandidateSessionResponse, ApiError> {
        let response = self.create_session(identity, instance, prepared).await?;
        let digest = self.session_digest(&response.session_handle);
        let id = uuid::Uuid::new_v4().to_string();
        let mut inner = self.inner.lock().await;
        let session = inner
            .sessions
            .get_mut(&digest)
            .ok_or_else(unknown_session)?;
        if let Some(references) = session.request.references.as_mut() {
            for candidate in &payload.reference_candidates {
                if let ReferenceCandidate::Canonical {
                    index,
                    upload: false,
                    reference,
                    ..
                } = candidate
                {
                    references[*index as usize - 1] = reference.clone();
                }
            }
        }
        session.normalization = Some(NormalizationSession {
            id: id.clone(),
            candidate_scope: fingerprint.clone(),
            candidates: payload
                .reference_candidates
                .into_iter()
                .map(|candidate| (candidate.index(), candidate))
                .collect(),
            receipts: HashMap::new(),
            finalized: false,
        });
        Ok(CandidateSessionResponse {
            protocol_version: VERSION,
            instance_id: instance.into(),
            session_id: id,
            candidate_scope_sha256: fingerprint,
            expires_at_ms: response.expires_at_ms,
            session_handle: response.session_handle,
            session_complete: false,
            uploads: response.uploads,
        })
    }

    pub(super) async fn is_normalization_upload(&self, handle: &str) -> bool {
        let inner = self.inner.lock().await;
        inner
            .upload_sessions
            .get(&self.upload_digest(handle))
            .and_then(|key| inner.sessions.get(key))
            .is_some_and(|session| session.normalization.is_some())
    }

    pub(super) async fn upload_normalized(
        &self,
        identity: &str,
        instance: &str,
        handle: &str,
        mime: &str,
        length: u64,
        body: Body,
    ) -> Result<NormalizationUploadResponse, ApiError> {
        self.purge_expired().await;
        let upload_digest = self.upload_digest(handle);
        let (session_digest, path, descriptor, index, original, fresh, reservation, cancel) = {
            let mut inner = self.inner.lock().await;
            let session_digest = inner
                .upload_sessions
                .get(&upload_digest)
                .cloned()
                .ok_or_else(unknown_upload)?;
            let host_bytes = inner
                .reserved_bytes
                .saturating_add(self.resolved_bytes.load(Ordering::Acquire));
            let session = inner
                .sessions
                .get_mut(&session_digest)
                .ok_or_else(unknown_upload)?;
            if session.identity != identity || session.consuming {
                return Err(unknown_upload());
            }
            let normalized = session.normalization.as_ref().ok_or_else(unknown_upload)?;
            if normalized.finalized {
                return Err(invalid("normalization session is already finalized"));
            }
            let slot = session
                .slots
                .get_mut(&upload_digest)
                .ok_or_else(unknown_upload)?;
            if !matches!(slot.state, UploadState::Empty) {
                return Err(ApiError::with_code(
                    "reference upload handle has already been used",
                    "REFERENCE_UPLOAD_ALREADY_USED",
                    StatusCode::CONFLICT,
                ));
            }
            let candidate = normalized
                .candidates
                .get(&slot.reference)
                .ok_or_else(unknown_upload)?;
            let (original, fresh) = match candidate {
                ReferenceCandidate::FreshVideo { original, .. } => (original.clone(), true),
                ReferenceCandidate::Canonical {
                    reference,
                    byte_length: Some(byte_length),
                    upload: true,
                    ..
                } => (
                    OriginalVideo {
                        mime_type: reference_mime_type(reference).into(),
                        name: reference.provenance().name.clone(),
                        sha256: reference
                            .provenance()
                            .sha256
                            .clone()
                            .ok_or_else(unknown_upload)?,
                        byte_length: *byte_length,
                    },
                    false,
                ),
                _ => return Err(unknown_upload()),
            };
            if length != original.byte_length || !mime_matches(&original.mime_type, mime) {
                return Err(invalid(
                    "upload MIME or Content-Length differs from its captured original",
                ));
            }
            let reservation = length
                .checked_add(if fresh {
                    MAX_REFERENCE_UPLOAD_FILE_BYTES
                } else {
                    0
                })
                .ok_or_else(|| invalid("upload quota overflow"))?;
            if session.reserved_bytes.saturating_add(reservation)
                > MAX_REFERENCE_UPLOAD_SESSION_BYTES
                || host_bytes.saturating_add(reservation) > MAX_REFERENCE_UPLOAD_HOST_BYTES
            {
                return Err(ApiError::with_code(
                    "original and canonical video scratch exceed reference quota",
                    "REFERENCE_UPLOAD_QUOTA",
                    StatusCode::PAYLOAD_TOO_LARGE,
                ));
            }
            slot.state = UploadState::Uploading {
                reserved_bytes: reservation,
            };
            session.reserved_bytes += reservation;
            session.active_bytes += reservation;
            self.resolved_bytes.fetch_add(reservation, Ordering::AcqRel);
            (
                session_digest,
                slot.path.clone(),
                slot.descriptor.clone(),
                slot.reference,
                original,
                fresh,
                reservation,
                session.cancel.child_token(),
            )
        };
        let disconnect = cancel.clone().drop_guard();
        let store = self.clone();
        let instance = instance.to_string();
        // Owned task retains scratch quota until child termination/reaping and
        // cleanup, even when HTTP transport drops its handler or session.
        let result = tokio::spawn(async move {
            let quota = ActiveQuota {
                bytes: reservation,
                counter: store.resolved_bytes.clone(),
            };
            let result = store
                .normalize_worker(
                    &session_digest,
                    &upload_digest,
                    &instance,
                    &path,
                    &descriptor,
                    index,
                    &original,
                    fresh,
                    body,
                    &cancel,
                )
                .await;
            let _ = tokio::fs::remove_file(path.with_extension("original")).await;
            if result.is_err() {
                let _ = tokio::fs::remove_file(&path).await;
                let mut inner = store.inner.lock().await;
                if let Some(session) = inner.sessions.get_mut(&session_digest) {
                    session.reserved_bytes = session.reserved_bytes.saturating_sub(reservation);
                    session.active_bytes = session.active_bytes.saturating_sub(reservation);
                    if let Some(slot) = session.slots.get_mut(&upload_digest) {
                        slot.state = UploadState::Empty;
                    }
                }
            }
            drop(quota);
            result
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("reference normalization task failed: {error}"))
        })?;
        disconnect.disarm();
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn normalize_worker(
        &self,
        session_key: &str,
        upload_key: &str,
        instance: &str,
        path: &Path,
        descriptor: &GenerationReference,
        index: u32,
        original: &OriginalVideo,
        fresh: bool,
        body: Body,
        cancel: &CancellationToken,
    ) -> Result<NormalizationUploadResponse, ApiError> {
        let permit = tokio::select! {
            _ = cancel.cancelled() => return Err(invalid("reference normalization cancelled")),
            permit = self.writers.clone().acquire_owned() => permit.map_err(|_| invalid("reference upload service is stopping"))?,
        };
        let input = if fresh {
            path.with_extension("original")
        } else {
            path.to_path_buf()
        };
        let digest = stream_to_private_file(&input, original.byte_length, body, cancel).await?;
        if digest != original.sha256 {
            return Err(invalid(
                "original video SHA-256 differs from captured bytes",
            ));
        }
        // Conversion cannot monopolize the ordinary V2 byte writers. Fresh
        // videos release their stream permit before entering the smaller CPU
        // conversion lane; quota stays owned while waiting and until cleanup.
        let _writer = if fresh {
            drop(permit);
            None
        } else {
            Some(permit)
        };
        let _converter = if fresh {
            Some(tokio::select! {
                _ = cancel.cancelled() => return Err(invalid("reference normalization cancelled")),
                permit = self.converters.clone().acquire_owned() => permit.map_err(|_| invalid("reference conversion service is stopping"))?,
            })
        } else {
            None
        };
        let converted = if fresh {
            crate::video_reference_codec::normalize(&input, path, cancel).await?
        } else {
            false
        };
        if cancel.is_cancelled() {
            return Err(invalid("reference normalization cancelled"));
        }
        let size = tokio::fs::metadata(path)
            .await
            .map_err(|_| invalid("canonical reference is missing"))?
            .len();
        if size == 0 || size > MAX_REFERENCE_UPLOAD_FILE_BYTES {
            return Err(invalid("canonical reference exceeds per-file byte limit"));
        }
        let probe_path = path.to_path_buf();
        let mut expected = descriptor.clone();
        if fresh {
            let provenance = match &mut expected {
                GenerationReference::Video { provenance, .. } => provenance,
                _ => return Err(invalid("fresh normalization slot is not video")),
            };
            provenance.sha256 = None;
        }
        let metadata = tokio::task::spawn_blocking(move || {
            let digest = mold_core::secure_file::sha256_open_file(
                &crate::batch_transaction::open_regular_file_no_follow(&probe_path)
                    .map_err(|_| invalid("cannot open canonical reference"))?,
            )
            .map_err(|_| invalid("cannot hash canonical reference"))?;
            probe_reference(
                &probe_path,
                &expected,
                index,
                &digest,
                ReferenceProbePolicy::CanonicalUpload,
            )
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("canonical reference probe failed: {error}"))
        })??;
        let receipt = NormalizationReceipt {
            version: VERSION,
            kind: metadata.kind,
            original_sha256: digest,
            original_byte_length: original.byte_length,
            original_mime_type: original.mime_type.clone(),
            canonical_sha256: metadata.sha256.clone(),
            canonical_byte_length: size,
            canonical_mime_type: metadata.mime_type.clone(),
            converted,
        };
        if !converted
            && (receipt.original_sha256 != receipt.canonical_sha256
                || receipt.original_byte_length != size)
        {
            return Err(invalid(
                "unchanged reference bytes failed receipt verification",
            ));
        }
        // Original bytes must no longer occupy scratch before their quota can
        // transition to the canonical completed-session charge.
        if fresh && converted {
            tokio::fs::remove_file(&input)
                .await
                .map_err(|_| invalid("cannot retire original video scratch"))?;
        }
        let mut inner = self.inner.lock().await;
        let session = inner
            .sessions
            .get_mut(session_key)
            .ok_or_else(unknown_upload)?;
        if session.cancel.is_cancelled() || cancel.is_cancelled() {
            return Err(invalid("reference normalization cancelled"));
        }
        let slot = session
            .slots
            .get_mut(upload_key)
            .ok_or_else(unknown_upload)?;
        let UploadState::Uploading { reserved_bytes } = slot.state else {
            return Err(unknown_upload());
        };
        slot.descriptor = reference_from_metadata(&metadata);
        slot.state = UploadState::Complete {
            size_bytes: size,
            metadata: Box::new(metadata.clone()),
        };
        session.reserved_bytes = session
            .reserved_bytes
            .saturating_sub(reserved_bytes)
            .saturating_add(size);
        session.active_bytes = session.active_bytes.saturating_sub(reserved_bytes);
        let normalized = session.normalization.as_mut().ok_or_else(unknown_upload)?;
        normalized.receipts.insert(index, receipt.clone());
        let response = NormalizationUploadResponse {
            protocol_version: VERSION,
            instance_id: instance.into(),
            session_id: normalized.id.clone(),
            candidate_scope_sha256: normalized.candidate_scope.clone(),
            reference: index,
            session_complete: false,
            receipt,
            metadata,
        };
        inner.reserved_bytes += size;
        Ok(response)
    }
}
struct ActiveQuota {
    bytes: u64,
    counter: Arc<AtomicU64>,
}
impl Drop for ActiveQuota {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[utoipa::path(
    post, path = "/api/generate/reference-normalization-finalize", tag = "generation",
    request_body = NormalizationFinalizeRequest,
    params(("X-Mold-Reference-Upload-Session" = String, Header, description = "Secret session handle")),
    responses((status = 200, description = "Atomically finalized canonical descriptors and scope", body = NormalizationFinalizeResponse), (status = 422, description = "Incomplete or mismatched receipt set"))
)]
pub(crate) async fn finalize_normalization_session(
    State(state): State<AppState>,
    authenticated: Option<Extension<ApiKeyAuthenticated>>,
    headers: HeaderMap,
    Json(payload): Json<NormalizationFinalizeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let identity = authenticated_identity(authenticated)?;
    let handle = required_secret_header(&headers, SESSION_HANDLE_HEADER)?;
    let captured = {
        let inner = state.reference_uploads.inner.lock().await;
        let session = inner
            .sessions
            .get(&state.reference_uploads.session_digest(handle))
            .ok_or_else(unknown_session)?;
        if session.identity != identity || session.consuming {
            return Err(unknown_session());
        }
        session.request.clone()
    };
    crate::routes::require_server_generation_request_activation(&state, &captured, None).await?;
    let roots = state.config.read().await.resolved_media_roots();
    let response = state
        .reference_uploads
        .finalize_normalization(&identity, &state.instance_id, handle, payload, &roots)
        .await?;
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(response)))
}
impl ReferenceUploadStore {
    async fn finalize_normalization(
        &self,
        identity: &str,
        instance: &str,
        handle: &str,
        payload: NormalizationFinalizeRequest,
        roots: &[PathBuf],
    ) -> Result<NormalizationFinalizeResponse, ApiError> {
        self.purge_expired().await;
        let digest = self.session_digest(handle);
        let (mut request, receipts, cancel) = {
            let inner = self.inner.lock().await;
            let session = inner.sessions.get(&digest).ok_or_else(unknown_session)?;
            if session.identity != identity || session.consuming {
                return Err(unknown_session());
            }
            let normalized = session.normalization.as_ref().ok_or_else(unknown_session)?;
            if payload.protocol_version != VERSION
                || payload.instance_id != instance
                || payload.session_id != normalized.id
                || payload.candidate_scope_sha256 != normalized.candidate_scope
            {
                return Err(invalid("normalization finalization authority mismatch"));
            }
            let mut receipts = normalized
                .receipts
                .iter()
                .map(|(reference, receipt)| FinalizeReceipt {
                    reference: *reference,
                    kind: receipt.kind,
                    original_sha256: receipt.original_sha256.clone(),
                    canonical_sha256: receipt.canonical_sha256.clone(),
                })
                .collect::<Vec<_>>();
            receipts.sort_by_key(|receipt| receipt.reference);
            let mut submitted = payload.receipts.clone();
            submitted.sort_by_key(|receipt| receipt.reference);
            if session.active_bytes != 0
                || session.slots.len() != receipts.len()
                || receipts != submitted
                || !session
                    .slots
                    .values()
                    .all(|slot| matches!(slot.state, UploadState::Complete { .. }))
            {
                return Err(invalid(
                    "finalization requires the exact complete upload receipt set",
                ));
            }
            if normalized.finalized {
                return Ok(NormalizationFinalizeResponse {
                    protocol_version: VERSION,
                    instance_id: instance.into(),
                    session_id: normalized.id.clone(),
                    candidate_scope_sha256: normalized.candidate_scope.clone(),
                    request_scope_sha256: session.scope_sha256.clone(),
                    session_complete: true,
                    references: session
                        .request
                        .references
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(reference_as_descriptor)
                        .collect(),
                    receipts,
                });
            }
            let mut request = session.request.clone();
            let references = request.references.as_mut().ok_or_else(unknown_session)?;
            for slot in session.slots.values() {
                references[slot.reference as usize - 1] = slot.descriptor.clone();
            }
            (request, receipts, session.cancel.clone())
        };
        let roots = roots.to_vec();
        let existing = request
            .references
            .as_deref()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter_map(|(position, reference)| {
                matches!(
                    reference.media(),
                    GenerationReferenceAuthority::ServerPath { .. }
                )
                .then_some((position, reference.clone()))
            })
            .collect::<Vec<_>>();
        let mut canonical_existing = Vec::new();
        for (position, reference) in existing {
            let metadata = self
                .probe_existing_snapshot(
                    identity,
                    &digest,
                    reference,
                    position as u32 + 1,
                    roots.clone(),
                    &cancel,
                )
                .await?;
            canonical_existing.push((position, reference_from_metadata(&metadata)));
        }
        for (position, canonical) in canonical_existing {
            request.references.as_mut().ok_or_else(unknown_session)?[position] = canonical;
        }
        validate_session_descriptors(&request, None)?;
        validate_normalization_request(&request)?;
        let scope = request_scope_sha256(&request)?;
        let mut inner = self.inner.lock().await;
        let session = inner
            .sessions
            .get_mut(&digest)
            .ok_or_else(unknown_session)?;
        if cancel.is_cancelled()
            || session.consuming
            || session.active_bytes != 0
            || session.identity != identity
            || session.expires_at_ms <= unix_time_ms()
        {
            return Err(unknown_session());
        }
        let normalized = session.normalization.as_mut().ok_or_else(unknown_session)?;
        // No slot can change after completion; the exact receipt set remains
        // immutable across this off-executor validation and repeated finalize.
        session.request = request.clone();
        session.scope_sha256 = scope.clone();
        normalized.finalized = true;
        Ok(NormalizationFinalizeResponse {
            protocol_version: VERSION,
            instance_id: instance.into(),
            session_id: normalized.id.clone(),
            candidate_scope_sha256: normalized.candidate_scope.clone(),
            request_scope_sha256: scope,
            session_complete: true,
            references: request
                .references
                .unwrap_or_default()
                .iter()
                .map(reference_as_descriptor)
                .collect(),
            receipts,
        })
    }
}

fn validate_normalization_request(request: &mold_core::GenerateRequest) -> Result<(), ApiError> {
    let mut descriptor_request = request.clone();
    descriptor_request.references = request
        .references
        .as_ref()
        .map(|references| references.iter().map(reference_as_descriptor).collect());
    mold_core::validation::validate_resolved_generate_request_fields(
        &descriptor_request,
        Some(minimax_h3::FAMILY),
    )
    .map_err(ApiError::validation)
}

#[cfg(test)]
mod store_tests {
    use super::*;
    fn candidate() -> CandidateSessionRequest {
        serde_json::from_value(serde_json::json!({"protocol_version":1,"request":{"model":minimax_h3::REF2VA_COMFY,"prompt":"A shot","width":minimax_h3::DEFAULT_WIDTH,"height":minimax_h3::DEFAULT_HEIGHT,"steps":minimax_h3::DEFAULT_STEPS,"frames":minimax_h3::REVIEWED_COMPACT_FRAMES,"fps":24,"output_format":"mp4","guidance":0.0,"strength":1.0},"reference_candidates":[{"index":1,"type":"fresh_video","kind":"video","original":{"name":"reference.m4v","mime_type":"video/x-m4v","sha256":"a".repeat(64),"byte_length":123}}]})).unwrap()
    }
    async fn session(store: &ReferenceUploadStore) -> CandidateSessionResponse {
        let candidate = candidate();
        let prepared = prepare_candidates(&candidate).unwrap();
        store
            .create_normalization("auth-a", "instance-a", candidate, prepared, "f".repeat(64))
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn body_hash_failure_releases_all_scratch_and_shared_quotas() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let session = session(&store).await;
        assert!(store
            .upload_normalized(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "video/x-m4v",
                123,
                Body::from(vec![1; 123])
            )
            .await
            .is_err());
        let inner = store.inner.lock().await;
        assert_eq!(inner.reserved_bytes, 0);
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
        let session = inner
            .sessions
            .get(&store.session_digest(&session.session_handle))
            .unwrap();
        assert_eq!(session.reserved_bytes, 0);
        assert_eq!(session.active_bytes, 0);
        assert!(!session.slots.values().next().unwrap().path.exists());
        assert!(!session
            .slots
            .values()
            .next()
            .unwrap()
            .path
            .with_extension("original")
            .exists());
    }
    #[tokio::test]
    async fn deleting_an_active_session_retains_host_charge_until_worker_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let session = session(&store).await;
        let handle = session.uploads[0].handle.clone();
        let worker_store = store.clone();
        let task = tokio::spawn(async move {
            worker_store
                .upload_normalized(
                    "auth-a",
                    "instance-a",
                    &handle,
                    "video/x-m4v",
                    123,
                    Body::from_stream(futures::stream::pending::<
                        Result<axum::body::Bytes, std::io::Error>,
                    >()),
                )
                .await
        });
        for _ in 0..100 {
            if store.resolved_bytes.load(Ordering::Acquire) > 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            store.resolved_bytes.load(Ordering::Acquire),
            123 + MAX_REFERENCE_UPLOAD_FILE_BYTES
        );
        // Remove only under the store lock so the worker cannot race the
        // assertion proving cancellation does not manufacture free capacity.
        {
            let mut inner = store.inner.lock().await;
            let path =
                remove_session_locked(&mut inner, &store.session_digest(&session.session_handle))
                    .unwrap();
            assert_eq!(inner.reserved_bytes, 0);
            assert_eq!(
                store.resolved_bytes.load(Ordering::Acquire),
                123 + MAX_REFERENCE_UPLOAD_FILE_BYTES
            );
            let _ = std::fs::remove_dir_all(path);
        }
        assert!(task.await.unwrap().is_err());
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
    }
    #[tokio::test]
    async fn completed_puts_still_cannot_generate_until_exact_receipt_finalization() {
        use base64::Engine;
        let dir = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(dir.path().join("cache"));
        let session = session(&store).await;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(include_str!("../../tests/fixtures/reference-video.mp4.b64").trim())
            .unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let key = store.session_digest(&session.session_handle);
        let mut request = {
            let mut inner = store.inner.lock().await;
            let stored = inner.sessions.get_mut(&key).unwrap();
            let slot = stored.slots.values_mut().next().unwrap();
            std::fs::write(&slot.path, &bytes).unwrap();
            let mut descriptor = slot.descriptor.clone();
            if let GenerationReference::Video { provenance, .. } = &mut descriptor {
                provenance.sha256 = Some(digest.clone());
            }
            let metadata = probe_reference(
                &slot.path,
                &descriptor,
                1,
                &digest,
                ReferenceProbePolicy::CanonicalUpload,
            )
            .unwrap();
            slot.descriptor = reference_from_metadata(&metadata);
            slot.state = UploadState::Complete {
                size_bytes: bytes.len() as u64,
                metadata: Box::new(metadata),
            };
            stored.request.references = Some(vec![slot.descriptor.clone()]);
            stored.scope_sha256 = request_scope_sha256(&stored.request).unwrap();
            stored.reserved_bytes = bytes.len() as u64;
            stored.normalization.as_mut().unwrap().receipts.insert(
                1,
                NormalizationReceipt {
                    version: VERSION,
                    kind: mold_core::GenerationReferenceKind::Video,
                    original_sha256: "a".repeat(64),
                    original_byte_length: 123,
                    original_mime_type: "video/x-m4v".into(),
                    canonical_sha256: digest.clone(),
                    canonical_byte_length: bytes.len() as u64,
                    canonical_mime_type: "video/mp4".into(),
                    converted: true,
                },
            );
            let request = stored.request.clone();
            inner.reserved_bytes = bytes.len() as u64;
            request
        };
        if let GenerationReference::Video { media, .. } =
            &mut request.references.as_mut().unwrap()[0]
        {
            *media = GenerationReferenceAuthority::Upload {
                handle: session.uploads[0].handle.clone(),
            };
        }
        assert!(store
            .upload_sources_for_request("auth-a", &request, None)
            .await
            .is_err());
        let receipt = FinalizeReceipt {
            reference: 1,
            kind: mold_core::GenerationReferenceKind::Video,
            original_sha256: "a".repeat(64),
            canonical_sha256: digest,
        };
        let mut payload = NormalizationFinalizeRequest {
            protocol_version: VERSION,
            instance_id: "instance-a".into(),
            session_id: session.session_id.clone(),
            candidate_scope_sha256: session.candidate_scope_sha256.clone(),
            receipts: vec![receipt.clone()],
        };
        payload.receipts.push(receipt);
        assert!(store
            .finalize_normalization(
                "auth-a",
                "instance-a",
                &session.session_handle,
                payload.clone(),
                &[]
            )
            .await
            .is_err());
        assert!(store
            .upload_sources_for_request("auth-a", &request, None)
            .await
            .is_err());
        payload.receipts.pop();
        let finalized = store
            .finalize_normalization(
                "auth-a",
                "instance-a",
                &session.session_handle,
                payload.clone(),
                &[],
            )
            .await
            .unwrap();
        assert!(finalized.session_complete);
        let repeated = store
            .finalize_normalization(
                "auth-a",
                "instance-a",
                &session.session_handle,
                payload,
                &[],
            )
            .await
            .unwrap();
        assert_eq!(
            finalized.request_scope_sha256,
            repeated.request_scope_sha256
        );
        assert!(store
            .upload_sources_for_request("auth-b", &request, None)
            .await
            .is_err());
        assert!(store
            .upload_sources_for_request("auth-a", &request, None)
            .await
            .is_ok());
    }
}

impl ReferenceUploadStore {
    async fn probe_existing_snapshot(
        &self,
        identity: &str,
        session_key: &str,
        reference: GenerationReference,
        index: u32,
        roots: Vec<PathBuf>,
        parent_cancel: &CancellationToken,
    ) -> Result<GenerationReferenceMetadata, ApiError> {
        let cancel = parent_cancel.child_token();
        let disconnect = cancel.clone().drop_guard();
        let store = self.clone();
        let identity = identity.to_string();
        let session_key = session_key.to_string();
        let result = tokio::spawn(async move {
            let _permit = tokio::select! {
                _ = cancel.cancelled() => return Err(invalid("existing reference validation cancelled")),
                permit = store.converters.clone().acquire_owned() => permit.map_err(|_| invalid("reference validation service is stopping"))?,
            };
            let opened_reference = reference.clone();
            let (file, bytes) = tokio::task::spawn_blocking(move || {
                let GenerationReferenceAuthority::ServerPath { path } = opened_reference.media() else { return Err(invalid("existing reference must have server path authority")); };
                let path = mold_core::resolve_server_media_path(path, &roots).map_err(invalid)?;
                let file = crate::batch_transaction::open_regular_file_no_follow(&path).map_err(|_| invalid("existing reference is unavailable"))?;
                let bytes = file.metadata().map_err(|_| invalid("existing reference metadata unavailable"))?.len();
                if bytes == 0 || bytes > MAX_REFERENCE_UPLOAD_FILE_BYTES { return Err(invalid("existing reference exceeds per-file limit")); }
                Ok((file, bytes))
            }).await.map_err(|error| ApiError::internal(format!("existing reference open failed: {error}")))??;
            let (snapshot_path, quota) = {
                let mut inner = store.inner.lock().await;
                let host = inner.reserved_bytes.saturating_add(store.resolved_bytes.load(Ordering::Acquire));
                let session = inner.sessions.get_mut(&session_key).ok_or_else(unknown_session)?;
                if session.identity != identity || session.consuming || cancel.is_cancelled() { return Err(unknown_session()); }
                if session.reserved_bytes.saturating_add(bytes) > MAX_REFERENCE_UPLOAD_SESSION_BYTES || host.saturating_add(bytes) > MAX_REFERENCE_UPLOAD_HOST_BYTES { return Err(ApiError::with_code("existing reference validation exceeds shared scratch quota", "REFERENCE_UPLOAD_QUOTA", StatusCode::PAYLOAD_TOO_LARGE)); }
                session.reserved_bytes += bytes;
                session.active_bytes += bytes;
                store.resolved_bytes.fetch_add(bytes, Ordering::AcqRel);
                (session.dir.join(format!("validate-{index}-{}.media", uuid::Uuid::new_v4())), ActiveQuota { bytes, counter: store.resolved_bytes.clone() })
            };
            let validation_cancel = cancel.clone();
            let result = tokio::task::spawn_blocking(move || {
                struct Snapshot(PathBuf);
                impl Drop for Snapshot { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }
                let snapshot = Snapshot(snapshot_path);
                if validation_cancel.is_cancelled() { return Err(invalid("existing reference validation cancelled")); }
                // Hash and probe the SAME immutable private copy. Reopening a
                // mutable server path after hashing its original descriptor
                // could otherwise bind facts from a replacement file.
                let digest = copy_private_bounded(file, &snapshot.0)?;
                if validation_cancel.is_cancelled() { return Err(invalid("existing reference validation cancelled")); }
                probe_reference(&snapshot.0, &reference, index, &digest, ReferenceProbePolicy::Strict)
            }).await.map_err(|error| ApiError::internal(format!("existing reference validation failed: {error}")));
            {
                let mut inner = store.inner.lock().await;
                if let Some(session) = inner.sessions.get_mut(&session_key) {
                    session.reserved_bytes = session.reserved_bytes.saturating_sub(bytes);
                    session.active_bytes = session.active_bytes.saturating_sub(bytes);
                }
            }
            drop(quota);
            result?
        }).await.map_err(|error| ApiError::internal(format!("existing reference snapshot task failed: {error}")))?;
        disconnect.disarm();
        result
    }
}

#[cfg(test)]
mod writer_tests {
    use super::*;
    #[tokio::test]
    async fn waiting_conversions_do_not_occupy_ordinary_upload_writers() {
        let directory = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(directory.path().join("cache"));
        let converters = store
            .converters
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let original = b"exact original bytes";
        let payload: CandidateSessionRequest = serde_json::from_value(serde_json::json!({"protocol_version":1,"request":{"model":minimax_h3::REF2VA_COMFY,"prompt":"A shot","width":512,"height":512,"steps":50},"reference_candidates":[{"index":1,"type":"fresh_video","kind":"video","original":{"mime_type":"video/quicktime","name":"reference.mov","byte_length":original.len(),"sha256":format!("{:x}",Sha256::digest(original))}}]})).unwrap();
        let prepared = prepare_candidates(&payload).unwrap();
        let session = store
            .create_normalization("auth-a", "instance-a", payload, prepared, "f".repeat(64))
            .await
            .unwrap();
        let handle = session.uploads[0].handle.clone();
        let key = store.session_digest(&session.session_handle);
        let original_path = store
            .inner
            .lock()
            .await
            .sessions
            .get(&key)
            .unwrap()
            .slots
            .values()
            .next()
            .unwrap()
            .path
            .with_extension("original");
        let task_store = store.clone();
        let task = tokio::spawn(async move {
            task_store
                .upload_normalized(
                    "auth-a",
                    "instance-a",
                    &handle,
                    "video/quicktime",
                    original.len() as u64,
                    Body::from(original.to_vec()),
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if tokio::fs::metadata(&original_path)
                    .await
                    .is_ok_and(|metadata| metadata.len() == original.len() as u64)
                    && store.writers.available_permits() == MAX_CONCURRENT_REFERENCE_UPLOADS
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            store.writers.available_permits(),
            MAX_CONCURRENT_REFERENCE_UPLOADS
        );
        assert!(store.resolved_bytes.load(Ordering::Acquire) > MAX_REFERENCE_UPLOAD_FILE_BYTES);
        store
            .cancel_session("auth-a", &session.session_handle)
            .await
            .unwrap();
        assert!(task.await.unwrap().is_err());
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
        assert!(!original_path.exists());
        drop(converters);
    }
}

#[cfg(test)]
mod client_server_tests {
    use super::*;
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };
    #[tokio::test]
    async fn real_store_receipts_and_final_descriptors_bind_the_rust_client_request() {
        use base64::Engine;
        let directory = tempfile::tempdir().unwrap();
        let store = ReferenceUploadStore::at(directory.path().join("cache"));
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(include_str!("../../tests/fixtures/reference-video.mp4.b64").trim())
            .unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let payload: CandidateSessionRequest = serde_json::from_value(serde_json::json!({"protocol_version":1,"request":{"model":minimax_h3::REF2VA_COMFY,"prompt":"A shot","width":minimax_h3::DEFAULT_WIDTH,"height":minimax_h3::DEFAULT_HEIGHT,"steps":minimax_h3::DEFAULT_STEPS,"frames":minimax_h3::REVIEWED_COMPACT_FRAMES,"fps":24,"output_format":"mp4","guidance":0.0,"strength":1.0,"seed":u64::MAX},"reference_candidates":[{"index":1,"type":"fresh_video","kind":"video","original":{"name":"reference.m4v","mime_type":"video/x-m4v","sha256":hash,"byte_length":bytes.len()}}]})).unwrap();
        let fingerprint = candidate_scope_sha256(&serde_json::to_vec(&payload).unwrap());
        let prepared = prepare_candidates(&payload).unwrap();
        let session = store
            .create_normalization(
                "auth-a",
                "instance-a",
                payload.clone(),
                prepared,
                fingerprint.clone(),
            )
            .await
            .unwrap();
        let Some(capability) =
            crate::video_reference_codec::completed_runtime_probe_for_test().await
        else {
            assert!(store
                .upload_normalized(
                    "auth-a",
                    "instance-a",
                    &session.uploads[0].handle,
                    "video/x-m4v",
                    bytes.len() as u64,
                    Body::from(bytes)
                )
                .await
                .is_err());
            assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
            store
                .cancel_session("auth-a", &session.session_handle)
                .await
                .unwrap();
            println!(
                "real_store_runtime=unavailable; fresh normalization refused without publication"
            );
            return;
        };
        let uploaded = store
            .upload_normalized(
                "auth-a",
                "instance-a",
                &session.uploads[0].handle,
                "video/x-m4v",
                bytes.len() as u64,
                Body::from(bytes.clone()),
            )
            .await
            .unwrap();
        assert!(!uploaded.session_complete);
        assert!(!uploaded.receipt.converted);
        assert_eq!(uploaded.receipt.original_mime_type, "video/x-m4v");
        assert_eq!(uploaded.receipt.canonical_mime_type, "video/mp4");
        assert_eq!(
            uploaded.receipt.original_sha256,
            uploaded.receipt.canonical_sha256
        );
        let finalized = store
            .finalize_normalization(
                "auth-a",
                "instance-a",
                &session.session_handle,
                NormalizationFinalizeRequest {
                    protocol_version: VERSION,
                    instance_id: "instance-a".into(),
                    session_id: session.session_id.clone(),
                    candidate_scope_sha256: fingerprint,
                    receipts: vec![FinalizeReceipt {
                        reference: 1,
                        kind: mold_core::GenerationReferenceKind::Video,
                        original_sha256: uploaded.receipt.original_sha256.clone(),
                        canonical_sha256: uploaded.receipt.canonical_sha256.clone(),
                    }],
                },
                &[],
            )
            .await
            .unwrap();
        let server = MockServer::start().await;
        let caps = mold_core::ReferenceUploadCapabilities {
            available: true,
            protocol_version: 2,
            requires_api_key: true,
            session_path: mold_core::reference_upload::SESSION_PATH.into(),
            upload_path: mold_core::reference_upload::UPLOAD_PATH.into(),
            session_handle_header: SESSION_HANDLE_HEADER.into(),
            upload_handle_header: UPLOAD_HANDLE_HEADER.into(),
            max_file_bytes: MAX_REFERENCE_UPLOAD_FILE_BYTES,
            max_session_bytes: MAX_REFERENCE_UPLOAD_SESSION_BYTES,
            max_active_sessions: 4,
            session_ttl_ms: REFERENCE_UPLOAD_SESSION_TTL.as_millis() as u64,
            video_normalization: Some(capability),
        };
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mold_core::ServerCapabilities {
                    reference_uploads: caps,
                    ..Default::default()
                }),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET")).and(path("/api/status")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"version":"test","instance_id":"instance-a","models_loaded":[],"gpu_info":null,"uptime_secs":0}))).mount(&server).await;
        Mock::given(method("POST"))
            .and(path(CANDIDATE_PATH))
            .and(wiremock::matchers::body_bytes(
                serde_json::to_vec(&payload).unwrap(),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(session.clone()))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path(mold_core::reference_upload::UPLOAD_PATH))
            .and(wiremock::matchers::body_bytes(bytes.clone()))
            .respond_with(ResponseTemplate::new(200).set_body_json(uploaded))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(FINALIZE_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(finalized))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path(mold_core::reference_upload::SESSION_PATH))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        let client = mold_core::MoldClient::with_api_key(&server.uri(), "test-key".into());
        let mut lease = client
            .bind_reference_candidates_v1(
                &payload.request,
                payload.reference_candidates,
                vec![mold_core::ReferenceUploadSource::bytes(bytes).unwrap()],
            )
            .await
            .unwrap();
        assert_eq!(lease.request().seed, Some(u64::MAX));
        assert_eq!(
            lease.request_scope_sha256,
            request_scope_sha256(lease.request()).unwrap()
        );
        assert_eq!(
            store
                .inner
                .lock()
                .await
                .sessions
                .get(&store.session_digest(&session.session_handle))
                .unwrap()
                .scope_sha256,
            lease.request_scope_sha256
        );
        lease.cancel().await.unwrap();
        store
            .cancel_session("auth-a", &session.session_handle)
            .await
            .unwrap();
        assert_eq!(store.inner.lock().await.reserved_bytes, 0);
        assert_eq!(store.resolved_bytes.load(Ordering::Acquire), 0);
        println!("real_store_runtime=available client_metadata_and_finalize_contract=true canonical_scope_verified=true");
    }
}
