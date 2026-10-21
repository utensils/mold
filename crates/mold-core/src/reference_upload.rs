//! Shared authenticated MiniMax H3 reference-upload V2 client authority.
//!
//! Every surface uses this module so canonical media probing, request-scope
//! rebinding, one-use completion order, and cleanup cannot drift independently.

use crate::{
    GenerateRequest, GenerationReference, GenerationReferenceAuthority, GenerationReferenceKind,
    GenerationReferenceProvenance, MoldClient, ReferenceUploadCapabilities,
    ReferenceUploadCompleteResponse, ReferenceUploadSessionRequest, ReferenceUploadSessionResponse,
};
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::File,
    time::{SystemTime, UNIX_EPOCH},
};

pub const PROTOCOL_VERSION: u32 = 2;
pub const SESSION_PATH: &str = "/api/generate/reference-upload-sessions";
pub const UPLOAD_PATH: &str = "/api/generate/reference-upload";
pub const SESSION_HANDLE_HEADER: &str = "x-mold-reference-upload-session";
pub const UPLOAD_HANDLE_HEADER: &str = "x-mold-reference-upload";

enum ReferenceUploadBody {
    OpenFile(File),
    Bytes(Vec<u8>),
}

/// One exact upload body. Debug output never reveals bytes or a local path.
pub struct ReferenceUploadSource {
    body: ReferenceUploadBody,
    length: u64,
}

impl std::fmt::Debug for ReferenceUploadSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUploadSource")
            .field("body", &"<redacted>")
            .field("length", &self.length)
            .finish()
    }
}

impl ReferenceUploadSource {
    pub fn open_file(file: File) -> Result<Self> {
        let metadata = file
            .metadata()
            .context("failed to inspect reference file")?;
        ensure!(
            metadata.is_file() && metadata.len() > 0,
            "reference upload source is not a non-empty regular file"
        );
        Ok(Self {
            body: ReferenceUploadBody::OpenFile(file),
            length: metadata.len(),
        })
    }

    pub fn bytes(bytes: Vec<u8>) -> Result<Self> {
        ensure!(!bytes.is_empty(), "reference upload source is empty");
        let length = u64::try_from(bytes.len()).context("reference upload is too large")?;
        Ok(Self {
            body: ReferenceUploadBody::Bytes(bytes),
            length,
        })
    }

    fn sha256(&self) -> Result<String> {
        match &self.body {
            ReferenceUploadBody::OpenFile(file) => crate::secure_file::sha256_open_file(file),
            ReferenceUploadBody::Bytes(bytes) => Ok(format!("{:x}", Sha256::digest(bytes))),
        }
    }
}

/// Fresh one-attempt authority returned after every upload is canonicalized.
pub struct ReferenceUploadLease {
    request: GenerateRequest,
    pub expires_at_ms: u64,
    pub request_scope_sha256: String,
    cancellation: SessionCancellationGuard,
}

impl std::fmt::Debug for ReferenceUploadLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUploadLease")
            .field("request", &"<redacted frozen request>")
            .field("session_handle", &"<redacted>")
            .field("expires_at_ms", &self.expires_at_ms)
            .field("request_scope_sha256", &self.request_scope_sha256)
            .finish()
    }
}

impl ReferenceUploadLease {
    /// Exact request frozen at session creation with only server-canonical
    /// descriptors and their one-use handles rebound afterward.
    pub fn request(&self) -> &GenerateRequest {
        &self.request
    }

    /// Stop best-effort cleanup after the server has accepted/consumed the
    /// request. Calling this before generation would leak the session.
    pub fn mark_consumed(&mut self) {
        self.cancellation.disarm();
    }

    /// A durable admission POST may have claimed this input even if its reply
    /// was lost. Keep the lease alive for exact operation reconciliation;
    /// dropping a local observer must not revoke the server's pending claim.
    pub fn preserve_for_uncertain_admission(&mut self) {
        self.cancellation.disarm();
    }

    /// Explicitly release an unused lease. Drop remains armed when transport
    /// cancellation interrupts this await, so it can schedule one final try.
    pub async fn cancel(&mut self) -> Result<()> {
        self.cancellation.cancel().await
    }
}

struct SessionCancellationGuard {
    client: MoldClient,
    session_handle: Option<String>,
}

impl SessionCancellationGuard {
    fn new(client: MoldClient, session_handle: &str) -> Self {
        Self {
            client,
            session_handle: valid_handle(session_handle).then(|| session_handle.to_string()),
        }
    }

    fn disarm(&mut self) {
        self.session_handle = None;
    }

    async fn cancel(&mut self) -> Result<()> {
        let Some(session_handle) = self.session_handle.clone() else {
            return Ok(());
        };
        self.client
            .cancel_reference_upload_session(&session_handle)
            .await?;
        self.disarm();
        Ok(())
    }
}

impl Drop for SessionCancellationGuard {
    fn drop(&mut self) {
        let Some(session_handle) = self.session_handle.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            // A server-enforced TTL remains the last bound during runtime
            // teardown, when spawning cleanup is no longer possible.
            return;
        };
        let client = self.client.clone();
        runtime.spawn(async move {
            let _ = client
                .cancel_reference_upload_session(&session_handle)
                .await;
        });
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_handle(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= crate::minimax_h3::MAX_REFERENCE_UPLOAD_HANDLE_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b':' | b'.'))
}

fn now_ms() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis();
    u64::try_from(millis).context("system clock is outside the supported range")
}

fn ensure_session_live(expires_at_ms: u64, observed_at_ms: u64) -> Result<()> {
    ensure!(
        expires_at_ms > observed_at_ms,
        "host returned an expired reference-upload session"
    );
    Ok(())
}

fn validate_capabilities(capabilities: &ReferenceUploadCapabilities) -> Result<()> {
    ensure!(
        capabilities.available
            && capabilities.protocol_version == PROTOCOL_VERSION
            && capabilities.requires_api_key,
        "host does not advertise authenticated reference-upload protocol V2"
    );
    ensure!(
        capabilities.session_path == SESSION_PATH
            && capabilities.upload_path == UPLOAD_PATH
            && capabilities
                .session_handle_header
                .eq_ignore_ascii_case(SESSION_HANDLE_HEADER)
            && capabilities
                .upload_handle_header
                .eq_ignore_ascii_case(UPLOAD_HANDLE_HEADER),
        "host advertised an incompatible reference-upload V2 wire contract"
    );
    ensure!(
        capabilities.max_file_bytes > 0
            && capabilities.max_session_bytes >= capabilities.max_file_bytes
            && capabilities.session_ttl_ms > 0,
        "host advertised invalid reference-upload V2 limits"
    );
    Ok(())
}

fn validate_session(
    session: &ReferenceUploadSessionResponse,
    expected_instance_id: &str,
    expected_count: usize,
    now_ms: u64,
) -> Result<Vec<String>> {
    ensure!(
        session.instance_id == expected_instance_id,
        "reference-upload session came from a different Mold instance"
    );
    ensure_session_live(session.expires_at_ms, now_ms)?;
    ensure!(
        valid_sha256(&session.request_scope_sha256)
            && valid_handle(&session.session_handle)
            && session.uploads.len() == expected_count,
        "host returned an incomplete reference-upload session"
    );
    let mut ordered = vec![None; expected_count];
    let mut handles = HashSet::new();
    for slot in &session.uploads {
        let index = usize::try_from(slot.reference)
            .ok()
            .and_then(|reference| reference.checked_sub(1))
            .filter(|index| *index < expected_count)
            .context("host returned an out-of-range reference-upload slot")?;
        ensure!(
            valid_handle(&slot.handle)
                && handles.insert(slot.handle.as_str())
                && ordered[index].replace(slot.handle.clone()).is_none(),
            "host returned an invalid or duplicate reference-upload slot"
        );
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, handle)| {
            handle.with_context(|| format!("host omitted reference-upload slot {}", index + 1))
        })
        .collect()
}

fn canonical_reference(
    provisional: &GenerationReference,
    complete: &ReferenceUploadCompleteResponse,
    expected_instance_id: &str,
    expected_reference: u32,
    expected_session_complete: bool,
    upload_handle: String,
) -> Result<GenerationReference> {
    ensure!(
        complete.instance_id == expected_instance_id
            && complete.reference == expected_reference
            && complete.metadata.index == expected_reference
            && complete.metadata.kind == provisional.kind()
            && valid_sha256(&complete.request_scope_sha256)
            && complete.session_complete == expected_session_complete,
        "reference {expected_reference} upload response did not match its bound V2 session"
    );
    let expected = provisional
        .redacted_metadata(usize::try_from(expected_reference.saturating_sub(1))?)
        .with_context(|| format!("reference {expected_reference} lost digest provenance"))?;
    ensure!(
        complete
            .metadata
            .sha256
            .eq_ignore_ascii_case(&expected.sha256)
            && complete.metadata.name == expected.name,
        "reference {expected_reference} upload response changed immutable provenance"
    );
    let provenance = GenerationReferenceProvenance {
        name: complete.metadata.name.clone(),
        sha256: Some(complete.metadata.sha256.to_ascii_lowercase()),
        crop: provisional.provenance().crop.clone(),
    };
    let descriptor = match provisional {
        GenerationReference::Image { mime_type, .. } => {
            ensure!(
                complete.metadata.kind == GenerationReferenceKind::Image
                    && complete.metadata.mime_type == *mime_type,
                "reference {expected_reference} upload response changed its image type"
            );
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: complete.metadata.mime_type.clone(),
                width: complete
                    .metadata
                    .width
                    .filter(|value| *value > 0)
                    .context("canonical image width is missing")?,
                height: complete
                    .metadata
                    .height
                    .filter(|value| *value > 0)
                    .context("canonical image height is missing")?,
            }
        }
        GenerationReference::NamedImage {
            role, mime_type, ..
        } => {
            ensure!(
                complete.metadata.kind == GenerationReferenceKind::Image
                    && complete.metadata.image_role == Some(*role)
                    && complete.metadata.mime_type == *mime_type,
                "reference {expected_reference} upload response changed its named image type or role"
            );
            GenerationReference::NamedImage {
                role: *role,
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: complete.metadata.mime_type.clone(),
                width: complete
                    .metadata
                    .width
                    .filter(|value| *value > 0)
                    .context("canonical named image width is missing")?,
                height: complete
                    .metadata
                    .height
                    .filter(|value| *value > 0)
                    .context("canonical named image height is missing")?,
            }
        }
        GenerationReference::Video { mime_type, .. } => {
            ensure!(
                mime_type == "video/mp4"
                    && complete.metadata.kind == GenerationReferenceKind::Video
                    && complete.metadata.mime_type == "video/mp4",
                "reference {expected_reference} upload response changed its video type"
            );
            let fps = complete
                .metadata
                .fps
                .filter(|value| value.is_finite() && *value > 0.0)
                .context("canonical video fps is missing")?;
            let (audio_duration_ms, audio_sample_count, audio_sample_rate, audio_channels) =
                if complete.metadata.has_audio {
                    let channels = complete
                        .metadata
                        .audio_channels
                        .filter(|value| matches!(value, 1 | 2))
                        .context("canonical video audio channel count is missing")?;
                    (
                        Some(
                            complete
                                .metadata
                                .audio_duration_ms
                                .filter(|value| *value > 0)
                                .context("canonical video audio duration is missing")?,
                        ),
                        Some(
                            complete
                                .metadata
                                .audio_sample_count
                                .filter(|value| *value > 0)
                                .context("canonical video audio sample count is missing")?,
                        ),
                        Some(
                            complete
                                .metadata
                                .audio_sample_rate
                                .filter(|value| *value > 0)
                                .context("canonical video audio sample rate is missing")?,
                        ),
                        Some(channels),
                    )
                } else {
                    ensure!(
                        complete.metadata.audio_duration_ms.is_none()
                            && complete.metadata.audio_sample_count.is_none()
                            && complete.metadata.audio_sample_rate.is_none()
                            && complete.metadata.audio_channels.is_none(),
                        "canonical silent video unexpectedly contains audio facts"
                    );
                    (None, None, None, None)
                };
            GenerationReference::Video {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: "video/mp4".to_string(),
                width: complete
                    .metadata
                    .width
                    .filter(|value| *value > 0)
                    .context("canonical video width is missing")?,
                height: complete
                    .metadata
                    .height
                    .filter(|value| *value > 0)
                    .context("canonical video height is missing")?,
                frame_count: Some(
                    complete
                        .metadata
                        .frame_count
                        .filter(|value| *value > 0)
                        .context("canonical video frame count is missing")?,
                ),
                duration_ms: complete
                    .metadata
                    .duration_ms
                    .filter(|value| *value > 0)
                    .context("canonical video duration is missing")?,
                fps,
                has_audio: complete.metadata.has_audio,
                audio_duration_ms,
                audio_sample_count,
                audio_sample_rate,
                audio_channels,
            }
        }
        GenerationReference::Audio { mime_type, .. } => {
            let declared = mime_type.trim().to_ascii_lowercase();
            ensure!(
                matches!(
                    declared.as_str(),
                    "audio/wav" | "audio/x-wav" | "audio/wave"
                ) && complete.metadata.kind == GenerationReferenceKind::Audio
                    && complete.metadata.mime_type == "audio/wav",
                "reference {expected_reference} upload response changed its audio type"
            );
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: "audio/wav".to_string(),
                duration_ms: complete
                    .metadata
                    .duration_ms
                    .filter(|value| *value > 0)
                    .context("canonical audio duration is missing")?,
                sample_rate: complete
                    .metadata
                    .sample_rate
                    .filter(|value| *value > 0)
                    .context("canonical audio sample rate is missing")?,
                channels: complete
                    .metadata
                    .channels
                    .filter(|value| matches!(value, 1 | 2))
                    .context("canonical audio channel count is missing")?,
                sample_count: Some(
                    complete
                        .metadata
                        .sample_count
                        .filter(|value| *value > 0)
                        .context("canonical audio sample count is missing")?,
                ),
            }
        }
        GenerationReference::Mesh { format, .. } => {
            ensure!(
                complete.metadata.kind == GenerationReferenceKind::Mesh
                    && complete.metadata.mesh_format == Some(*format),
                "reference {expected_reference} upload response changed its mesh format"
            );
            GenerationReference::Mesh {
                media: GenerationReferenceAuthority::Descriptor,
                provenance,
                mime_type: complete.metadata.mime_type.clone(),
                format: *format,
                byte_length: complete
                    .metadata
                    .byte_length
                    .filter(|value| *value > 0)
                    .context("canonical mesh byte length is missing")?,
                coordinates: complete
                    .metadata
                    .coordinates
                    .context("canonical mesh coordinates are missing")?,
            }
        }
    };
    if matches!(
        &descriptor,
        GenerationReference::Image { .. }
            | GenerationReference::Video { .. }
            | GenerationReference::Audio { .. }
    ) {
        crate::minimax_h3::reference_prepared_shape(&descriptor).map_err(anyhow::Error::new)?;
    }
    Ok(with_authority(
        descriptor,
        GenerationReferenceAuthority::Upload {
            handle: upload_handle,
        },
    ))
}

fn with_authority(
    reference: GenerationReference,
    media: GenerationReferenceAuthority,
) -> GenerationReference {
    match reference {
        GenerationReference::Image {
            provenance,
            mime_type,
            width,
            height,
            ..
        } => GenerationReference::Image {
            media,
            provenance,
            mime_type,
            width,
            height,
        },
        GenerationReference::NamedImage {
            role,
            provenance,
            mime_type,
            width,
            height,
            ..
        } => GenerationReference::NamedImage {
            role,
            media,
            provenance,
            mime_type,
            width,
            height,
        },
        GenerationReference::Video {
            provenance,
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
            ..
        } => GenerationReference::Video {
            media,
            provenance,
            mime_type,
            width,
            height,
            frame_count,
            duration_ms,
            fps,
            has_audio,
            audio_duration_ms,
            audio_sample_count,
            audio_sample_rate,
            audio_channels,
        },
        GenerationReference::Audio {
            provenance,
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
            ..
        } => GenerationReference::Audio {
            media,
            provenance,
            mime_type,
            duration_ms,
            sample_rate,
            channels,
            sample_count,
        },
        GenerationReference::Mesh {
            provenance,
            mime_type,
            format,
            byte_length,
            coordinates,
            ..
        } => GenerationReference::Mesh {
            media,
            provenance,
            mime_type,
            format,
            byte_length,
            coordinates,
        },
    }
}

fn mime_type(reference: &GenerationReference) -> &str {
    match reference {
        GenerationReference::Image { mime_type, .. }
        | GenerationReference::NamedImage { mime_type, .. }
        | GenerationReference::Video { mime_type, .. }
        | GenerationReference::Audio { mime_type, .. }
        | GenerationReference::Mesh { mime_type, .. } => mime_type,
    }
}

fn rebind_canonical_request(
    mut scoped_request: GenerateRequest,
    references: Vec<GenerationReference>,
) -> GenerateRequest {
    scoped_request.references = Some(references);
    scoped_request
}

impl MoldClient {
    /// Create and fill one canonical upload-V2 lease. A returned request may be
    /// submitted exactly once; callers must re-open/re-download original media
    /// to retry after any generation attempt.
    pub async fn bind_reference_uploads_v2(
        &self,
        request: &GenerateRequest,
        sources: Vec<ReferenceUploadSource>,
    ) -> Result<ReferenceUploadLease> {
        ensure!(
            self.has_api_key(),
            "reference uploads require a client configured with a valid API key"
        );
        let (capabilities, status) =
            tokio::try_join!(self.server_capabilities(), self.server_status())?;
        let capabilities = capabilities.reference_uploads;
        validate_capabilities(&capabilities)?;
        let expected_instance_id = status
            .instance_id
            .filter(|value| !value.trim().is_empty())
            .context("host status does not expose an exact Mold instance identity")?;
        let scoped_request = request.clone();
        let descriptors = scoped_request
            .references
            .clone()
            .context("reference uploads require ordered H3 descriptors")?;
        ensure!(
            !descriptors.is_empty() && descriptors.len() == sources.len(),
            "reference descriptor/body count mismatch"
        );
        crate::minimax_h3::validate_reference_descriptors(&descriptors)
            .map_err(anyhow::Error::new)?;

        let mut total_bytes = 0_u64;
        for (index, (descriptor, source)) in descriptors.iter().zip(&sources).enumerate() {
            ensure!(
                source.length <= capabilities.max_file_bytes,
                "reference {} exceeds this host's per-file upload limit",
                index + 1
            );
            total_bytes = total_bytes
                .checked_add(source.length)
                .context("combined reference size overflowed")?;
            ensure!(
                total_bytes <= capabilities.max_session_bytes,
                "ordered references exceed this host's upload-session limit"
            );
            let expected_digest = descriptor
                .content_sha256()
                .with_context(|| format!("reference {} has no content digest", index + 1))?;
            ensure!(
                source.sha256()?.eq_ignore_ascii_case(&expected_digest),
                "reference {} changed after it was probed",
                index + 1
            );
        }

        let upload_references = (1..=descriptors.len())
            .map(|index| u32::try_from(index).context("too many reference uploads"))
            .collect::<Result<Vec<_>>>()?;
        let session = self
            .create_reference_upload_session(&ReferenceUploadSessionRequest {
                request: scoped_request.clone(),
                upload_references,
            })
            .await?;
        // Arm cleanup before the next await. Dropping this future during any
        // upload schedules a best-effort DELETE instead of waiting for TTL.
        let mut cancellation = SessionCancellationGuard::new(self.clone(), &session.session_handle);
        let handles = match validate_session(
            &session,
            &expected_instance_id,
            descriptors.len(),
            now_ms()?,
        ) {
            Ok(handles) => handles,
            Err(error) => {
                let _ = cancellation.cancel().await;
                return Err(error);
            }
        };

        let result: Result<(Vec<GenerationReference>, String)> = async {
            let upload_count = sources.len();
            let mut canonical = Vec::with_capacity(upload_count);
            let mut canonical_scope = session.request_scope_sha256.to_ascii_lowercase();
            for (index, ((descriptor, source), handle)) in
                descriptors.iter().zip(sources).zip(handles).enumerate()
            {
                let reference = u32::try_from(index + 1).context("too many references")?;
                let content_type = mime_type(descriptor);
                let completed = match source.body {
                    ReferenceUploadBody::OpenFile(file) => {
                        self.upload_reference_open_file(&handle, file, content_type)
                            .await
                    }
                    ReferenceUploadBody::Bytes(bytes) => {
                        self.upload_reference_bytes(&handle, bytes, content_type)
                            .await
                    }
                }
                .with_context(|| format!("reference {reference} upload failed"))?;
                let bound = canonical_reference(
                    descriptor,
                    &completed,
                    &expected_instance_id,
                    reference,
                    index + 1 == upload_count,
                    handle,
                )?;
                canonical_scope = completed.request_scope_sha256.to_ascii_lowercase();
                canonical.push(bound);
            }
            crate::minimax_h3::validate_references(&canonical).map_err(anyhow::Error::new)?;
            ensure_session_live(session.expires_at_ms, now_ms()?).context(
                "reference-upload session expired before its canonical request could be returned",
            )?;
            Ok((canonical, canonical_scope))
        }
        .await;

        match result {
            Ok((references, request_scope_sha256)) => {
                let request = rebind_canonical_request(scoped_request, references);
                Ok(ReferenceUploadLease {
                    request,
                    expires_at_ms: session.expires_at_ms,
                    request_scope_sha256,
                    cancellation,
                })
            }
            Err(error) => {
                let _ = cancellation.cancel().await;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_references(references: Vec<GenerationReference>) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "keep this request frozen",
            "model": crate::minimax_h3::REF2VA_COMFY,
            "width": crate::minimax_h3::DEFAULT_WIDTH,
            "height": crate::minimax_h3::DEFAULT_HEIGHT,
            "steps": crate::minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "seed": 77,
            "batch_size": 1,
            "output_format": "mp4",
            "strength": 1.0,
            "frames": crate::minimax_h3::MIN_FRAMES,
            "fps": crate::minimax_h3::FIXED_FPS,
            "enable_audio": true
        }))
        .unwrap();
        request.references = Some(references);
        request
    }

    fn audio_descriptor() -> GenerationReference {
        GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: GenerationReferenceProvenance {
                name: Some("voice.wav".into()),
                sha256: Some("a".repeat(64)),
                crop: None,
            },
            mime_type: "audio/x-wav".into(),
            duration_ms: 1_000,
            sample_rate: 44_100,
            channels: 2,
            sample_count: Some(44_100),
        }
    }

    #[test]
    fn canonical_audio_response_rebinds_provisional_facts() {
        let completed = ReferenceUploadCompleteResponse {
            instance_id: "instance".into(),
            reference: 1,
            metadata: crate::GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Audio,
                index: 1,
                name: Some("voice.wav".into()),
                sha256: "a".repeat(64),
                mime_type: "audio/wav".into(),
                width: None,
                height: None,
                frame_count: None,
                duration_ms: Some(998),
                fps: None,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
                sample_rate: Some(48_000),
                channels: Some(2),
                sample_count: Some(47_904),
                prepared_shape: None,
                crop: None,
                image_role: None,
                mesh_format: None,
                byte_length: None,
                coordinates: None,
            },
            request_scope_sha256: "b".repeat(64),
            session_complete: true,
        };
        let bound = canonical_reference(
            &audio_descriptor(),
            &completed,
            "instance",
            1,
            true,
            "mru_handle".into(),
        )
        .unwrap();
        assert!(matches!(
            bound,
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Upload { .. },
                duration_ms: 998,
                sample_rate: 48_000,
                sample_count: Some(47_904),
                ..
            }
        ));
    }

    #[test]
    fn lease_debug_redacts_frozen_request_and_session_authority() {
        let mut request = request_with_references(vec![with_authority(
            audio_descriptor(),
            GenerationReferenceAuthority::Upload {
                handle: "mru_private_upload".into(),
            },
        )]);
        request.prompt = "private prompt".into();
        request.source_image = Some(vec![17, 42, 99]);
        let lease = ReferenceUploadLease {
            request,
            expires_at_ms: u64::MAX,
            request_scope_sha256: "c".repeat(64),
            cancellation: SessionCancellationGuard::new(
                MoldClient::new("http://127.0.0.1:9"),
                "mrs_private_session",
            ),
        };

        let debug = format!("{lease:?}");
        assert!(debug.contains("<redacted frozen request>"));
        assert!(!debug.contains("private prompt"));
        assert!(!debug.contains("mru_private_upload"));
        assert!(!debug.contains("mrs_private_session"));
        assert!(!debug.contains("17, 42, 99"));
    }

    #[test]
    fn completion_order_and_scope_fail_closed() {
        let mut completed = ReferenceUploadCompleteResponse {
            instance_id: "instance".into(),
            reference: 1,
            metadata: audio_descriptor().redacted_metadata(0).unwrap(),
            request_scope_sha256: "b".repeat(64),
            session_complete: true,
        };
        assert!(canonical_reference(
            &audio_descriptor(),
            &completed,
            "instance",
            1,
            false,
            "mru_handle".into(),
        )
        .is_err());
        completed.session_complete = false;
        completed.request_scope_sha256 = "bad".into();
        assert!(canonical_reference(
            &audio_descriptor(),
            &completed,
            "instance",
            1,
            false,
            "mru_handle".into(),
        )
        .is_err());
    }

    #[test]
    fn canonical_rebind_changes_only_the_reference_authority() {
        let original = request_with_references(vec![audio_descriptor()]);
        let bound_reference = with_authority(
            audio_descriptor(),
            GenerationReferenceAuthority::Upload {
                handle: "mru_handle".into(),
            },
        );
        let rebound = rebind_canonical_request(original.clone(), vec![bound_reference.clone()]);
        let mut expected = original;
        expected.references = Some(vec![bound_reference]);
        assert_eq!(
            serde_json::to_value(rebound).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[test]
    fn final_expiry_check_rejects_a_session_that_elapsed_during_upload() {
        assert!(ensure_session_live(101, 100).is_ok());
        assert!(ensure_session_live(100, 100).is_err());
        assert!(ensure_session_live(99, 100).is_err());
    }

    #[tokio::test]
    async fn unauthenticated_client_fails_before_any_host_or_media_work() {
        let server = wiremock::MockServer::start().await;
        let request = request_with_references(vec![audio_descriptor()]);
        let error = MoldClient::new(&server.uri())
            .bind_reference_uploads_v2(
                &request,
                vec![ReferenceUploadSource::bytes(b"unread media".to_vec()).unwrap()],
            )
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("configured with a valid API key"));
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn dropping_armed_session_guard_schedules_cancellation() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path(SESSION_PATH))
            .and(header("x-api-key", "sekrit"))
            .and(header(SESSION_HANDLE_HEADER, "mrs_abort"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;
        let client = MoldClient::with_api_key(&server.uri(), "sekrit".into());
        drop(SessionCancellationGuard::new(client, "mrs_abort"));
        for _ in 0..20 {
            if !server.received_requests().await.unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

/// Accept only the explicit additive fresh-reference contract. Unsupported
/// versions retain ordinary canonical probing and its actionable errors.
pub fn supports_video_normalization(caps: &ReferenceUploadCapabilities) -> bool {
    use crate::video_normalization::*;
    validate_capabilities(caps).is_ok()
        && caps.video_normalization.as_ref().is_some_and(|value| {
            value.available
                && value.protocol_version == VERSION
                && value
                    .input_roles
                    .iter()
                    .any(|role| role == "reference_video")
                && value.candidate_session_path == CANDIDATE_PATH
                && value.finalize_path == FINALIZE_PATH
                && value
                    .candidate_digest_header
                    .eq_ignore_ascii_case(CANDIDATE_DIGEST_HEADER)
                && value.canonical_mime_type == "video/mp4"
                && value.max_original_file_bytes > 0
                && value.max_canonical_file_bytes > 0
                && value.max_session_bytes >= value.max_original_file_bytes
                && value.max_original_file_bytes <= caps.max_file_bytes
                && value.max_canonical_file_bytes <= caps.max_file_bytes
        })
}

/// Filename hints never prove format; the server performs bounded demux/probe.
pub fn video_mime_hint(name: &str) -> Option<&'static str> {
    match std::path::Path::new(name)
        .extension()?
        .to_str()?
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" => Some("video/mp4"),
        "mov" => Some("video/quicktime"),
        "m4v" => Some("video/x-m4v"),
        "webm" => Some("video/webm"),
        "mkv" => Some("video/x-matroska"),
        _ => None,
    }
}

fn normalized_video_descriptor(
    metadata: &crate::GenerationReferenceMetadata,
) -> Result<GenerationReference> {
    ensure!(
        metadata.kind == GenerationReferenceKind::Video
            && metadata.mime_type == "video/mp4"
            && valid_sha256(&metadata.sha256),
        "normalization returned invalid canonical video identity"
    );
    if metadata.has_audio {
        ensure!(
            metadata
                .audio_channels
                .is_some_and(|channels| matches!(channels, 1 | 2))
                && metadata.audio_sample_rate.is_some_and(|rate| rate > 0)
                && metadata.audio_sample_count.is_some_and(|count| count > 0)
                && metadata
                    .audio_duration_ms
                    .is_some_and(|duration| duration > 0),
            "canonical video soundtrack facts are missing or unsupported"
        );
    } else {
        ensure!(
            metadata.audio_channels.is_none()
                && metadata.audio_sample_rate.is_none()
                && metadata.audio_sample_count.is_none()
                && metadata.audio_duration_ms.is_none(),
            "silent normalized video contains soundtrack facts"
        );
    }
    let descriptor = GenerationReference::Video {
        media: GenerationReferenceAuthority::Descriptor,
        provenance: GenerationReferenceProvenance {
            name: metadata.name.clone(),
            sha256: Some(metadata.sha256.clone()),
            crop: None,
        },
        mime_type: metadata.mime_type.clone(),
        width: metadata.width.context("canonical video width missing")?,
        height: metadata.height.context("canonical video height missing")?,
        frame_count: metadata.frame_count,
        duration_ms: metadata
            .duration_ms
            .context("canonical video duration missing")?,
        fps: metadata.fps.context("canonical video rate missing")?,
        has_audio: metadata.has_audio,
        audio_duration_ms: metadata.audio_duration_ms,
        audio_sample_count: metadata.audio_sample_count,
        audio_sample_rate: metadata.audio_sample_rate,
        audio_channels: metadata.audio_channels,
    };
    crate::minimax_h3::reference_prepared_shape(&descriptor).map_err(anyhow::Error::new)?;
    ensure!(
        descriptor
            .redacted_metadata(metadata.index.saturating_sub(1) as usize)
            .as_ref()
            == Some(metadata),
        "normalization metadata includes inconsistent video facts"
    );
    Ok(descriptor)
}

impl MoldClient {
    /// One frozen candidate attempt, finalized before placement or admission.
    /// This helper accepts fresh local bodies only; existing server authorities
    /// continue through the ordinary canonical request path.
    pub async fn bind_reference_candidates_v1(
        &self,
        request: &GenerateRequest,
        candidates: Vec<crate::video_normalization::ReferenceCandidate>,
        sources: Vec<ReferenceUploadSource>,
    ) -> Result<ReferenceUploadLease> {
        use crate::video_normalization::*;
        // Capture everything before the first await, including private file
        // handles; no later selected host or mutable authoring state is read.
        ensure!(request.references.is_none(), "candidate request must omit canonical references; supply the complete ordered candidate set");
        let scoped_request = request.clone();
        let mut payload = CandidateSessionRequest {
            protocol_version: VERSION,
            request: scoped_request.clone(),
            reference_candidates: candidates,
        };
        ensure!(
            sources.len() == payload.reference_candidates.len() && !sources.is_empty(),
            "candidate/body count mismatch"
        );
        ensure!(
            self.has_api_key(),
            "fresh reference normalization requires API-key authentication"
        );
        payload.request = crate::prompt_text::protect_generate_request_for_wire(&payload.request);
        let body = serde_json::to_vec(&payload)?;
        let fingerprint = candidate_scope_sha256(&body);
        let (caps, status) = tokio::try_join!(self.server_capabilities(), self.server_status())?;
        ensure!(
            supports_video_normalization(&caps.reference_uploads),
            "host does not advertise fresh video normalization v1"
        );
        let capability = caps
            .reference_uploads
            .video_normalization
            .as_ref()
            .context("normalization capability missing")?;
        let instance = status
            .instance_id
            .filter(|value| !value.is_empty())
            .context("host instance identity missing")?;
        let mut originals = Vec::new();
        let mut total = 0_u64;
        let mut fresh = false;
        for (position, (candidate, source)) in payload
            .reference_candidates
            .iter()
            .zip(&sources)
            .enumerate()
        {
            ensure!(
                candidate.index() as usize == position + 1,
                "candidate order is invalid"
            );
            let original = match candidate {
                ReferenceCandidate::FreshVideo { kind, original, .. } => {
                    ensure!(
                        *kind == GenerationReferenceKind::Video
                            && capability.accepted_mime_types.contains(&original.mime_type),
                        "fresh video kind/MIME is unsupported"
                    );
                    fresh = true;
                    original.clone()
                }
                ReferenceCandidate::Canonical {
                    upload: true,
                    reference,
                    byte_length: Some(length),
                    ..
                } if matches!(reference.media(), GenerationReferenceAuthority::Descriptor) => {
                    OriginalVideo {
                        mime_type: mime_type(reference).into(),
                        name: reference.provenance().name.clone(),
                        sha256: reference
                            .content_sha256()
                            .context("canonical reference digest missing")?,
                        byte_length: *length,
                    }
                }
                _ => anyhow::bail!(
                    "fresh local candidate helper refuses existing upload/server authorities"
                ),
            };
            ensure!(
                source.length == original.byte_length
                    && source.length > 0
                    && source.length <= capability.max_original_file_bytes
                    && source.sha256()? == original.sha256,
                "reference original changed after capture or exceeds upload limit"
            );
            total = total
                .checked_add(source.length)
                .context("candidate sizes overflow")?;
            ensure!(
                total <= capability.max_session_bytes,
                "reference candidates exceed session limit"
            );
            originals.push(original);
        }
        ensure!(fresh, "candidate normalization requires a fresh video");
        let session = self
            .create_reference_normalization_session(body, &fingerprint)
            .await?;
        let mut cancellation = SessionCancellationGuard::new(self.clone(), &session.session_handle);
        let result: Result<(Vec<GenerationReference>, String)> = async {
            ensure!(
                session.protocol_version == VERSION
                    && session.instance_id == instance
                    && session.candidate_scope_sha256 == fingerprint
                    && !session.session_complete
                    && !session.session_id.is_empty(),
                "candidate session identity mismatch"
            );
            let handles = validate_session(
                &ReferenceUploadSessionResponse {
                    instance_id: session.instance_id.clone(),
                    expires_at_ms: session.expires_at_ms,
                    request_scope_sha256: fingerprint.clone(),
                    session_handle: session.session_handle.clone(),
                    uploads: session.uploads.clone(),
                },
                &instance,
                originals.len(),
                now_ms()?,
            )?;
            let mut descriptors = Vec::new();
            let mut receipts = Vec::new();
            for (position, (((candidate, original), source), handle)) in payload
                .reference_candidates
                .iter()
                .zip(&originals)
                .zip(sources)
                .zip(&handles)
                .enumerate()
            {
                let reference = position as u32 + 1;
                let body = match source.body {
                    ReferenceUploadBody::OpenFile(mut file) => {
                        use std::io::{Seek, SeekFrom};
                        file.seek(SeekFrom::Start(0))?;
                        reqwest::Body::wrap_stream(tokio_util::io::ReaderStream::new(
                            tokio::fs::File::from_std(file),
                        ))
                    }
                    ReferenceUploadBody::Bytes(bytes) => reqwest::Body::from(bytes),
                };
                let completed = self
                    .upload_normalization_body(
                        handle,
                        &original.mime_type,
                        original.byte_length,
                        body,
                    )
                    .await?;
                let receipt = &completed.receipt;
                let is_fresh = matches!(candidate, ReferenceCandidate::FreshVideo { .. });
                let kind = match candidate {
                    ReferenceCandidate::FreshVideo { kind, .. } => *kind,
                    ReferenceCandidate::Canonical { reference, .. } => reference.kind(),
                };
                ensure!(
                    completed.protocol_version == VERSION
                        && completed.instance_id == instance
                        && completed.session_id == session.session_id
                        && completed.candidate_scope_sha256 == fingerprint
                        && completed.reference == reference
                        && !completed.session_complete
                        && completed.metadata.index == reference,
                    "normalization upload is from a different session/slot"
                );
                ensure!(
                    receipt.version == VERSION
                        && receipt.kind == kind
                        && completed.metadata.kind == kind
                        && receipt.original_sha256 == original.sha256
                        && receipt.original_byte_length == original.byte_length
                        && receipt.original_mime_type == original.mime_type
                        && receipt.canonical_sha256 == completed.metadata.sha256
                        && valid_sha256(&receipt.canonical_sha256)
                        && receipt.canonical_mime_type == completed.metadata.mime_type
                        && receipt.canonical_byte_length > 0
                        && receipt.canonical_byte_length <= capability.max_canonical_file_bytes
                        && completed.metadata.name == original.name,
                    "normalization original/canonical receipt mismatch"
                );
                ensure!(
                    !receipt.converted || is_fresh,
                    "server converted a byte-preserving canonical slot"
                );
                ensure!(
                    receipt.converted
                        || (receipt.original_sha256 == receipt.canonical_sha256
                            && receipt.original_byte_length == receipt.canonical_byte_length),
                    "unchanged normalization receipt changed bytes"
                );
                let descriptor = match candidate {
                    ReferenceCandidate::FreshVideo { .. } => {
                        normalized_video_descriptor(&completed.metadata)?
                    }
                    ReferenceCandidate::Canonical {
                        reference: provisional,
                        ..
                    } => {
                        let complete = ReferenceUploadCompleteResponse {
                            instance_id: instance.clone(),
                            reference,
                            metadata: completed.metadata.clone(),
                            request_scope_sha256: fingerprint.clone(),
                            session_complete: false,
                        };
                        let bound = canonical_reference(
                            provisional,
                            &complete,
                            &instance,
                            reference,
                            false,
                            handle.clone(),
                        )?;
                        with_authority(bound, GenerationReferenceAuthority::Descriptor)
                    }
                };
                receipts.push(FinalizeReceipt {
                    reference,
                    kind,
                    original_sha256: receipt.original_sha256.clone(),
                    canonical_sha256: receipt.canonical_sha256.clone(),
                });
                descriptors.push(descriptor);
            }
            let finalized = self
                .finalize_reference_normalization(
                    &session.session_handle,
                    &NormalizationFinalizeRequest {
                        protocol_version: VERSION,
                        instance_id: instance.clone(),
                        session_id: session.session_id.clone(),
                        candidate_scope_sha256: fingerprint.clone(),
                        receipts: receipts.clone(),
                    },
                )
                .await?;
            ensure!(
                finalized.protocol_version == VERSION
                    && finalized.instance_id == instance
                    && finalized.session_id == session.session_id
                    && finalized.candidate_scope_sha256 == fingerprint
                    && finalized.session_complete
                    && valid_sha256(&finalized.request_scope_sha256)
                    && finalized.receipts == receipts
                    && serde_json::to_value(&finalized.references)?
                        == serde_json::to_value(&descriptors)?,
                "normalization finalization changed the complete canonical receipt set"
            );
            ensure_session_live(session.expires_at_ms, now_ms()?)?;
            let references = descriptors
                .into_iter()
                .zip(handles)
                .map(|(descriptor, handle)| {
                    with_authority(descriptor, GenerationReferenceAuthority::Upload { handle })
                })
                .collect::<Vec<_>>();
            crate::minimax_h3::validate_references(&references).map_err(anyhow::Error::new)?;
            Ok((references, finalized.request_scope_sha256))
        }
        .await;
        match result {
            Ok((references, request_scope_sha256)) => Ok(ReferenceUploadLease {
                request: rebind_canonical_request(scoped_request, references),
                expires_at_ms: session.expires_at_ms,
                request_scope_sha256,
                cancellation,
            }),
            Err(error) => {
                let _ = cancellation.cancel().await;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    use crate::video_normalization::*;
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };
    fn caps() -> ReferenceUploadCapabilities {
        serde_json::from_value(serde_json::json!({"available":true,"protocol_version":2,"requires_api_key":true,"session_path":SESSION_PATH,"upload_path":UPLOAD_PATH,"session_handle_header":SESSION_HANDLE_HEADER,"upload_handle_header":UPLOAD_HANDLE_HEADER,"max_file_bytes":268435456u64,"max_session_bytes":1073741824u64,"max_active_sessions":4,"session_ttl_ms":1800000,"video_normalization":{"available":true,"protocol_version":1,"input_roles":["reference_video"],"candidate_session_path":CANDIDATE_PATH,"finalize_path":FINALIZE_PATH,"candidate_digest_header":CANDIDATE_DIGEST_HEADER,"max_original_file_bytes":268435456u64,"max_canonical_file_bytes":268435456u64,"max_session_bytes":1073741824u64,"accepted_mime_types":["video/quicktime","video/x-m4v"],"accepted_extensions":["mov","m4v"],"accepted_video_codecs":["h264","hevc"],"canonical_mime_type":"video/mp4"}})).unwrap()
    }
    #[test]
    fn additive_capability_is_not_authority_without_the_exact_role_and_protocol() {
        let mut value = caps();
        assert!(supports_video_normalization(&value));
        value.video_normalization.as_mut().unwrap().protocol_version = 9;
        assert!(!supports_video_normalization(&value));
        value = caps();
        value.video_normalization.as_mut().unwrap().input_roles = vec!["source_video".into()];
        assert!(!supports_video_normalization(&value));
        value = caps();
        value.video_normalization = None;
        assert!(!supports_video_normalization(&value));
        assert_eq!(video_mime_hint("reference.MKV"), Some("video/x-matroska"));
        assert_eq!(video_mime_hint("reference.m3u8"), None);
    }
    async fn attempt(tamper: &str, converted: bool) -> Result<ReferenceUploadLease> {
        let server = MockServer::start().await;
        let original = b"exact original bytes".to_vec();
        let original_hash = format!("{:x}", Sha256::digest(&original));
        let canonical_hash = if converted {
            "c".repeat(64)
        } else {
            original_hash.clone()
        };
        let request: GenerateRequest = serde_json::from_value(
            serde_json::json!({"model":"minimax-h3-ref2va:bf16","prompt":"A shot","seed":u64::MAX,"width":512,"height":512,"steps":4}),
        )?;
        let candidate = ReferenceCandidate::FreshVideo {
            index: 1,
            kind: GenerationReferenceKind::Video,
            original: OriginalVideo {
                name: Some("reference.m4v".into()),
                mime_type: "video/x-m4v".into(),
                byte_length: original.len() as u64,
                sha256: original_hash.clone(),
            },
        };
        let payload = CandidateSessionRequest {
            protocol_version: VERSION,
            request: crate::prompt_text::protect_generate_request_for_wire(&request),
            reference_candidates: vec![candidate.clone()],
        };
        let body = serde_json::to_vec(&payload)?;
        let fingerprint = candidate_scope_sha256(&body);
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(crate::ServerCapabilities {
                    reference_uploads: caps(),
                    ..Default::default()
                }),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET")).and(path("/api/status")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"version":"test","instance_id":"instance-a","uptime_secs":0,"models_loaded":[],"gpu_info":null}))).mount(&server).await;
        Mock::given(method("POST")).and(path(CANDIDATE_PATH)).and(wiremock::matchers::body_bytes(body)).and(wiremock::matchers::header(CANDIDATE_DIGEST_HEADER, fingerprint.as_str())).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"protocol_version":1,"instance_id":"instance-a","session_id":"public-session","candidate_scope_sha256":fingerprint,"expires_at_ms":now_ms()?+60000,"session_handle":"mrs_test","session_complete":false,"uploads":[{"reference":1,"handle":"mru_test"}]}))).expect(1).mount(&server).await;
        let metadata = serde_json::json!({"index":1,"kind":"video","name":"reference.m4v","sha256":canonical_hash,"mime_type":"video/mp4","width":32,"height":32,"frame_count":48,"duration_ms":2000,"fps":24.0,"has_audio":false});
        let mut canonical = metadata.clone();
        canonical["media"] = serde_json::json!({"authority":"descriptor"});
        canonical["provenance"] =
            serde_json::json!({"name":"reference.m4v","sha256":canonical_hash});
        let canonical: GenerationReference = serde_json::from_value(canonical)?;
        let metadata = serde_json::to_value(
            canonical
                .redacted_metadata(0)
                .context("test canonical metadata missing")?,
        )?;
        let mut upload = serde_json::json!({"protocol_version":1,"instance_id":"instance-a","session_id":"public-session","candidate_scope_sha256":fingerprint,"reference":1,"session_complete":false,"receipt":{"version":1,"kind":"video","original_sha256":original_hash,"original_byte_length":original.len(),"original_mime_type":"video/x-m4v","canonical_sha256":canonical_hash,"canonical_byte_length":if converted {123}else{original.len()},"canonical_mime_type":"video/mp4","converted":converted},"metadata":metadata});
        match tamper {
            "original" => upload["receipt"]["original_sha256"] = "a".repeat(64).into(),
            "session" => upload["session_id"] = "other".into(),
            "unnegotiated_hash" => upload["receipt"]["converted"] = false.into(),
            _ => {}
        }
        Mock::given(method("PUT"))
            .and(path(UPLOAD_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload))
            .expect(1)
            .mount(&server)
            .await;
        let mut descriptor = metadata;
        descriptor.as_object_mut().unwrap().remove("index");
        descriptor.as_object_mut().unwrap().remove("name");
        descriptor.as_object_mut().unwrap().remove("sha256");
        descriptor["media"] = serde_json::json!({"authority":"descriptor"});
        descriptor["provenance"] =
            serde_json::json!({"name":"reference.m4v","sha256":canonical_hash});
        let receipts = serde_json::json!([{"reference":1,"kind":"video","original_sha256":original_hash,"canonical_sha256":canonical_hash}]);
        let mut finalization = serde_json::json!({"protocol_version":1,"instance_id":"instance-a","session_id":"public-session","candidate_scope_sha256":fingerprint,"request_scope_sha256":"f".repeat(64),"session_complete":true,"references":[descriptor],"receipts":receipts});
        if tamper == "final_descriptor" {
            finalization["references"][0]["width"] = 64.into();
        }
        Mock::given(method("POST"))
            .and(path(FINALIZE_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(finalization))
            .expect(if matches!(tamper, "" | "final_descriptor") {
                1
            } else {
                0
            })
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path(SESSION_PATH))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        let client = MoldClient::with_api_key(&server.uri(), "test-key".into());
        let result = client
            .bind_reference_candidates_v1(
                &request,
                vec![candidate],
                vec![ReferenceUploadSource::bytes(original)?],
            )
            .await;
        // Explicit cleanup while this mock transport is still alive.
        if let Ok(mut lease) = result {
            lease.cancel().await?;
            return Ok(lease);
        }
        result
    }
    #[tokio::test]
    async fn original_receipt_and_explicit_finalize_are_required_before_a_lease() {
        attempt("", true).await.unwrap();
        attempt("", false)
            .await
            .expect("unchanged M4V MIME alias must be accepted");
        for tamper in [
            "original",
            "session",
            "unnegotiated_hash",
            "final_descriptor",
        ] {
            assert!(attempt(tamper, true).await.is_err(), "accepted {tamper}");
        }
    }
}
