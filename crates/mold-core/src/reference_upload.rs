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
