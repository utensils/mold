//! Lossless separation of request media from durable queue JSON.
//!
//! The queue's ordinary JSON row is useful for scheduling and replay, but it
//! must not become a second plaintext media store. This module moves every
//! `GenerateRequest` byte payload, media-local path, provenance filename,
//! content digest, and LoRA/HDR path into non-serializable, job-scoped records.
//! The records expose crate-private parts for a separate encrypted storage
//! layer; this module deliberately defines no storage schema and performs no
//! cross-job content deduplication. Until that layer exists, this is a mapping
//! primitive rather than a claim that queue media survives a restart.
//!
//! Process-private grants are a different class of authority. Resolved H3
//! reference staging owns descriptor/quota lifetimes, and the H3 ingress grant
//! binds authentication, instance, policy, and the exact request. Neither can
//! be reconstructed from media records, so extraction rejects them explicitly.

use std::collections::{BTreeMap, HashMap};

use mold_core::{GenerationReference, KeyframeCondition, LoraWeight};

/// Top-level `GenerateRequest` fields whose values must never enter the
/// durable request JSON. Keep this list in lockstep with `extract_request_media`.
pub const REQUEST_AUTHORITY_JSON_FIELDS: &[&str] = &[
    "source_image",
    "source_image_name",
    "id_image",
    "id_image_name",
    "id_images",
    "id_image_names",
    "edit_images",
    "references",
    "mask_image",
    "control_image",
    "audio_file",
    "audio_file_path",
    "source_video",
    "source_video_path",
    "extend_video",
    "extend_video_path",
    "keyframes",
    "hdr_exr_dir",
    "lora",
    "loras",
];

/// A process-private authority that media extraction cannot make durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessPrivateAuthority {
    /// Private staged paths plus their quota/lifetime owner.
    ResolvedReferenceStaging,
    /// Authenticated, instance- and policy-bound MiniMax H3 ingress proof.
    H3PrivateIngressGrant,
}

/// Explicit inventory supplied by the admission boundary.
///
/// This is intentionally not inferred from `GenerateRequest`: a descriptor-only
/// H3 request does not reveal whether a live `ResolvedReferenceSet` accompanies
/// it, and the private ingress grant never enters the request at all.
#[derive(Debug, Default)]
pub struct ProcessPrivateAuthorities {
    authorities: Vec<ProcessPrivateAuthority>,
}

impl ProcessPrivateAuthorities {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn from_authority(authority: ProcessPrivateAuthority) -> Self {
        Self {
            authorities: vec![authority],
        }
    }

    pub fn from_present(resolved_reference_staging: bool, h3_private_ingress_grant: bool) -> Self {
        let mut authorities = Vec::new();
        if resolved_reference_staging {
            authorities.push(ProcessPrivateAuthority::ResolvedReferenceStaging);
        }
        if h3_private_ingress_grant {
            authorities.push(ProcessPrivateAuthority::H3PrivateIngressGrant);
        }
        Self { authorities }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QueueMediaError {
    #[error("process-private generation authority cannot be durably rehydrated: {0:?}")]
    UnsupportedProcessPrivateAuthority(ProcessPrivateAuthority),
    #[error("request authority must be available before deferred media hydration: {0:?}")]
    UnsupportedPreDispatchAuthority(QueueMediaRole),
    #[error("durable request JSON serialization failed: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("durable request JSON deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("durable request JSON retained prohibited authority field {0}")]
    RequestJsonContainsAuthority(&'static str),
    #[error("queue media records belong to a different job")]
    JobScopeMismatch,
    #[error("queue media records require a non-empty job scope")]
    InvalidJobScope,
    #[error("durable request JSON must be an object")]
    InvalidRequestJsonShape,
    #[error("queue media records are malformed for role {0:?}")]
    MalformedRecords(QueueMediaRole),
    #[error("a media collection is too large to index")]
    CollectionTooLarge,
}

/// Stable semantic role of one extracted request value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueMediaRole {
    SourceImage,
    SourceImageName,
    IdentityImage,
    IdentityImageName,
    IdentityImages,
    IdentityImageNames,
    EditImages,
    References,
    MaskImage,
    ControlImage,
    AudioFile,
    AudioFilePath,
    SourceVideo,
    SourceVideoPath,
    ExtendVideo,
    ExtendVideoPath,
    Keyframes,
    HdrExrDir,
    Lora,
    Loras,
}

/// A scalar, collection-presence marker, or ordered collection item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMediaPosition {
    Scalar,
    Collection,
    Item(u32),
}

pub(crate) enum QueueMediaPayload {
    Presence,
    Bytes(Vec<u8>),
    Text(String),
    Reference(GenerationReference),
    Keyframe(KeyframeCondition),
    Lora(LoraWeight),
}

/// One non-serializable record. Payload access remains crate-private so an
/// eventual encrypted store can consume it without exposing it to API types.
pub struct OpaqueQueueMediaRecord {
    pub(crate) role: QueueMediaRole,
    pub(crate) position: QueueMediaPosition,
    pub(crate) payload: QueueMediaPayload,
}

impl OpaqueQueueMediaRecord {
    pub fn role(&self) -> QueueMediaRole {
        self.role
    }

    pub fn position(&self) -> QueueMediaPosition {
        self.position
    }
}

impl std::fmt::Debug for OpaqueQueueMediaRecord {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpaqueQueueMediaRecord")
            .field("role", &self.role)
            .field("position", &self.position)
            .field("payload", &"<redacted>")
            .finish()
    }
}

/// Media records bound to one queue job. It is intentionally neither `Clone`
/// nor serializable: callers cannot accidentally create a content-addressed
/// cross-job cache or write the payloads as ordinary JSON.
pub struct OpaqueQueueMedia {
    pub(crate) job_id: String,
    pub(crate) records: Vec<OpaqueQueueMediaRecord>,
}

impl OpaqueQueueMedia {
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    pub fn records(&self) -> &[OpaqueQueueMediaRecord] {
        &self.records
    }
}

impl std::fmt::Debug for OpaqueQueueMedia {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpaqueQueueMedia")
            .field("job_id", &self.job_id)
            .field("records", &self.records.len())
            .finish()
    }
}

pub struct ExtractedQueueRequest {
    request_json: String,
    media: OpaqueQueueMedia,
}

impl ExtractedQueueRequest {
    pub fn request_json(&self) -> &str {
        &self.request_json
    }

    pub fn media(&self) -> &OpaqueQueueMedia {
        &self.media
    }

    pub fn into_parts(self) -> (String, OpaqueQueueMedia) {
        (self.request_json, self.media)
    }

    pub fn rehydrate(
        self,
        expected_job_id: &str,
    ) -> Result<mold_core::GenerateRequest, QueueMediaError> {
        rehydrate_request_media(expected_job_id, &self.request_json, self.media)
    }
}

fn scalar(
    records: &mut Vec<OpaqueQueueMediaRecord>,
    role: QueueMediaRole,
    payload: Option<QueueMediaPayload>,
) {
    if let Some(payload) = payload {
        records.push(OpaqueQueueMediaRecord {
            role,
            position: QueueMediaPosition::Scalar,
            payload,
        });
    }
}

fn collection<T>(
    records: &mut Vec<OpaqueQueueMediaRecord>,
    role: QueueMediaRole,
    values: Option<Vec<T>>,
    wrap: impl Fn(T) -> QueueMediaPayload,
) -> Result<(), QueueMediaError> {
    let Some(values) = values else { return Ok(()) };
    records.push(OpaqueQueueMediaRecord {
        role,
        position: QueueMediaPosition::Collection,
        payload: QueueMediaPayload::Presence,
    });
    for (index, value) in values.into_iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| QueueMediaError::CollectionTooLarge)?;
        records.push(OpaqueQueueMediaRecord {
            role,
            position: QueueMediaPosition::Item(index),
            payload: wrap(value),
        });
    }
    Ok(())
}

/// Split one request into JSON-safe settings and job-local opaque records.
///
/// The input is consumed so no second plaintext request copy is retained by
/// this API. Callers must inventory process-private authorities separately;
/// any such authority makes the operation unsupported rather than guessed.
pub fn extract_request_media(
    job_id: impl Into<String>,
    request: mold_core::GenerateRequest,
    process_private: &ProcessPrivateAuthorities,
) -> Result<ExtractedQueueRequest, QueueMediaError> {
    let job_id = job_id.into();
    if job_id.trim().is_empty() {
        return Err(QueueMediaError::InvalidJobScope);
    }
    if let Some(authority) = process_private.authorities.first().copied() {
        return Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
            authority,
        ));
    }
    // The private grant is not carried on GenerateRequest, but its necessity
    // is. Deriving this here prevents a new call site from making an H3 request
    // look durable merely by forgetting to inventory the live grant.
    if mold_core::minimax_h3::capability_contract_for_model(&request.model).is_some() {
        return Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
            ProcessPrivateAuthority::H3PrivateIngressGrant,
        ));
    }
    // At the current server boundary references have already been rewritten
    // to descriptors paired with a live ResolvedReferenceSet. Raw upload and
    // server-path forms are temporary authority too. Until a durable staging
    // owner exists, every form is unsupported rather than replayed as dead
    // authority.
    if request.references.is_some() {
        return Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
            ProcessPrivateAuthority::ResolvedReferenceStaging,
        ));
    }
    // Scheduler V2 resolves exact adapter paths/scales before the worker lease,
    // while durable media hydration is intentionally deferred. HDR likewise
    // carries a local output authority rather than replayable input media.
    // Keep their lossless field mapping covered below, but do not advertise a
    // public durable path until admission owns a safe pre-dispatch projection.
    if request.lora.is_some() {
        return Err(QueueMediaError::UnsupportedPreDispatchAuthority(
            QueueMediaRole::Lora,
        ));
    }
    if request.loras.is_some() {
        return Err(QueueMediaError::UnsupportedPreDispatchAuthority(
            QueueMediaRole::Loras,
        ));
    }
    if request.hdr_exr_dir.is_some() {
        return Err(QueueMediaError::UnsupportedPreDispatchAuthority(
            QueueMediaRole::HdrExrDir,
        ));
    }

    extract_request_fields(job_id, request)
}

/// The field mapper stays separate from the authority preflight so its
/// exhaustive lossless property can be tested for currently unsupported H3
/// reference shapes without making those shapes admissible.
fn extract_request_fields(
    job_id: String,
    request: mold_core::GenerateRequest,
) -> Result<ExtractedQueueRequest, QueueMediaError> {
    // Intentionally exhaustive: adding any GenerateRequest field fails this
    // build until it is classified as retained JSON or extracted authority.
    let mold_core::GenerateRequest {
        prompt,
        negative_prompt,
        model,
        width,
        height,
        steps,
        guidance,
        seed,
        batch_size,
        output_format,
        embed_metadata,
        scheduler,
        cfg_plus,
        source_image,
        source_image_name,
        source_fit,
        id_image,
        id_image_name,
        id_weight,
        id_start_step,
        id_images,
        id_image_names,
        true_cfg,
        cfg_start_step,
        edit_images,
        references,
        strength,
        mask_image,
        control_image,
        control_model,
        control_scale,
        expand,
        original_prompt,
        prompt_transform,
        title,
        tags,
        collection: filing_collection,
        batch_id,
        batch_index,
        batch_count,
        lora,
        frames,
        fps,
        upscale_model,
        gif_preview,
        enable_audio,
        audio_file,
        audio_file_path,
        source_video,
        source_video_path,
        extend_video,
        extend_video_path,
        extend_overlap_frames,
        keyframes,
        hdr_exr_dir,
        hdr_exr_full_float,
        pipeline,
        ic_lora_control,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        guidance_overrides,
        sample_shift,
        distill_strength_high,
        distill_strength_low,
        placement,
    } = request;

    let mut records = Vec::new();
    scalar(
        &mut records,
        QueueMediaRole::SourceImage,
        source_image.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::SourceImageName,
        source_image_name.map(QueueMediaPayload::Text),
    );
    scalar(
        &mut records,
        QueueMediaRole::IdentityImage,
        id_image.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::IdentityImageName,
        id_image_name.map(QueueMediaPayload::Text),
    );
    collection(
        &mut records,
        QueueMediaRole::IdentityImages,
        id_images,
        QueueMediaPayload::Bytes,
    )?;
    collection(
        &mut records,
        QueueMediaRole::IdentityImageNames,
        id_image_names,
        QueueMediaPayload::Text,
    )?;
    collection(
        &mut records,
        QueueMediaRole::EditImages,
        edit_images,
        QueueMediaPayload::Bytes,
    )?;
    collection(
        &mut records,
        QueueMediaRole::References,
        references,
        QueueMediaPayload::Reference,
    )?;
    scalar(
        &mut records,
        QueueMediaRole::MaskImage,
        mask_image.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::ControlImage,
        control_image.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::AudioFile,
        audio_file.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::AudioFilePath,
        audio_file_path.map(QueueMediaPayload::Text),
    );
    scalar(
        &mut records,
        QueueMediaRole::SourceVideo,
        source_video.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::SourceVideoPath,
        source_video_path.map(QueueMediaPayload::Text),
    );
    scalar(
        &mut records,
        QueueMediaRole::ExtendVideo,
        extend_video.map(QueueMediaPayload::Bytes),
    );
    scalar(
        &mut records,
        QueueMediaRole::ExtendVideoPath,
        extend_video_path.map(QueueMediaPayload::Text),
    );
    collection(
        &mut records,
        QueueMediaRole::Keyframes,
        keyframes,
        QueueMediaPayload::Keyframe,
    )?;
    scalar(
        &mut records,
        QueueMediaRole::HdrExrDir,
        hdr_exr_dir.map(QueueMediaPayload::Text),
    );
    scalar(
        &mut records,
        QueueMediaRole::Lora,
        lora.map(QueueMediaPayload::Lora),
    );
    collection(
        &mut records,
        QueueMediaRole::Loras,
        loras,
        QueueMediaPayload::Lora,
    )?;

    let sanitized = mold_core::GenerateRequest {
        prompt,
        negative_prompt,
        model,
        width,
        height,
        steps,
        guidance,
        seed,
        batch_size,
        output_format,
        embed_metadata,
        scheduler,
        cfg_plus,
        source_image: None,
        source_image_name: None,
        source_fit,
        id_image: None,
        id_image_name: None,
        id_weight,
        id_start_step,
        id_images: None,
        id_image_names: None,
        true_cfg,
        cfg_start_step,
        edit_images: None,
        references: None,
        strength,
        mask_image: None,
        control_image: None,
        control_model,
        control_scale,
        expand,
        original_prompt,
        prompt_transform,
        title,
        tags,
        collection: filing_collection,
        batch_id,
        batch_index,
        batch_count,
        lora: None,
        frames,
        fps,
        upscale_model,
        gif_preview,
        enable_audio,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames,
        keyframes: None,
        hdr_exr_dir: None,
        hdr_exr_full_float,
        pipeline,
        ic_lora_control,
        loras: None,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        guidance_overrides,
        sample_shift,
        distill_strength_high,
        distill_strength_low,
        placement,
    };
    let request_json = serde_json::to_string(&sanitized).map_err(QueueMediaError::Serialize)?;
    ensure_json_is_authority_free(&request_json)?;
    Ok(ExtractedQueueRequest {
        request_json,
        media: OpaqueQueueMedia { job_id, records },
    })
}

fn ensure_json_is_authority_free(request_json: &str) -> Result<(), QueueMediaError> {
    let value: serde_json::Value =
        serde_json::from_str(request_json).map_err(QueueMediaError::Deserialize)?;
    let object = value
        .as_object()
        .ok_or(QueueMediaError::InvalidRequestJsonShape)?;
    for field in REQUEST_AUTHORITY_JSON_FIELDS {
        if object.contains_key(*field) {
            return Err(QueueMediaError::RequestJsonContainsAuthority(field));
        }
    }
    Ok(())
}

fn take_role(
    grouped: &mut HashMap<QueueMediaRole, Vec<OpaqueQueueMediaRecord>>,
    role: QueueMediaRole,
) -> Vec<OpaqueQueueMediaRecord> {
    grouped.remove(&role).unwrap_or_default()
}

fn scalar_payload(
    grouped: &mut HashMap<QueueMediaRole, Vec<OpaqueQueueMediaRecord>>,
    role: QueueMediaRole,
) -> Result<Option<QueueMediaPayload>, QueueMediaError> {
    let mut records = take_role(grouped, role);
    match records.len() {
        0 => Ok(None),
        1 if records[0].position == QueueMediaPosition::Scalar => {
            Ok(Some(records.pop().expect("length checked").payload))
        }
        _ => Err(QueueMediaError::MalformedRecords(role)),
    }
}

fn collection_payloads(
    grouped: &mut HashMap<QueueMediaRole, Vec<OpaqueQueueMediaRecord>>,
    role: QueueMediaRole,
) -> Result<Option<Vec<QueueMediaPayload>>, QueueMediaError> {
    let records = take_role(grouped, role);
    if records.is_empty() {
        return Ok(None);
    }
    let mut presence = false;
    let mut items = BTreeMap::new();
    for record in records {
        match (record.position, record.payload) {
            (QueueMediaPosition::Collection, QueueMediaPayload::Presence) if !presence => {
                presence = true;
            }
            (QueueMediaPosition::Item(index), payload) => {
                if items.insert(index, payload).is_some() {
                    return Err(QueueMediaError::MalformedRecords(role));
                }
            }
            _ => return Err(QueueMediaError::MalformedRecords(role)),
        }
    }
    if !presence {
        return Err(QueueMediaError::MalformedRecords(role));
    }
    let mut values = Vec::with_capacity(items.len());
    for (expected, (actual, payload)) in (0_u32..).zip(items) {
        if expected != actual {
            return Err(QueueMediaError::MalformedRecords(role));
        }
        values.push(payload);
    }
    Ok(Some(values))
}

/// Reconstruct the exact request semantics from its JSON-safe settings and
/// the records belonging to the same queue job.
pub fn rehydrate_request_media(
    expected_job_id: &str,
    request_json: &str,
    media: OpaqueQueueMedia,
) -> Result<mold_core::GenerateRequest, QueueMediaError> {
    if media.job_id != expected_job_id {
        return Err(QueueMediaError::JobScopeMismatch);
    }
    ensure_json_is_authority_free(request_json)?;
    let mut request: mold_core::GenerateRequest =
        serde_json::from_str(request_json).map_err(QueueMediaError::Deserialize)?;
    let mut grouped: HashMap<_, Vec<_>> = HashMap::new();
    for record in media.records {
        grouped.entry(record.role).or_default().push(record);
    }

    macro_rules! restore_scalar {
        ($role:expr, $field:ident, $variant:ident) => {
            if let Some(payload) = scalar_payload(&mut grouped, $role)? {
                request.$field = Some(match payload {
                    QueueMediaPayload::$variant(value) => value,
                    _ => return Err(QueueMediaError::MalformedRecords($role)),
                });
            }
        };
    }
    macro_rules! restore_collection {
        ($role:expr, $field:ident, $variant:ident) => {
            if let Some(payloads) = collection_payloads(&mut grouped, $role)? {
                let mut values = Vec::with_capacity(payloads.len());
                for payload in payloads {
                    values.push(match payload {
                        QueueMediaPayload::$variant(value) => value,
                        _ => return Err(QueueMediaError::MalformedRecords($role)),
                    });
                }
                request.$field = Some(values);
            }
        };
    }

    restore_scalar!(QueueMediaRole::SourceImage, source_image, Bytes);
    restore_scalar!(QueueMediaRole::SourceImageName, source_image_name, Text);
    restore_scalar!(QueueMediaRole::IdentityImage, id_image, Bytes);
    restore_scalar!(QueueMediaRole::IdentityImageName, id_image_name, Text);
    restore_collection!(QueueMediaRole::IdentityImages, id_images, Bytes);
    restore_collection!(QueueMediaRole::IdentityImageNames, id_image_names, Text);
    restore_collection!(QueueMediaRole::EditImages, edit_images, Bytes);
    restore_collection!(QueueMediaRole::References, references, Reference);
    restore_scalar!(QueueMediaRole::MaskImage, mask_image, Bytes);
    restore_scalar!(QueueMediaRole::ControlImage, control_image, Bytes);
    restore_scalar!(QueueMediaRole::AudioFile, audio_file, Bytes);
    restore_scalar!(QueueMediaRole::AudioFilePath, audio_file_path, Text);
    restore_scalar!(QueueMediaRole::SourceVideo, source_video, Bytes);
    restore_scalar!(QueueMediaRole::SourceVideoPath, source_video_path, Text);
    restore_scalar!(QueueMediaRole::ExtendVideo, extend_video, Bytes);
    restore_scalar!(QueueMediaRole::ExtendVideoPath, extend_video_path, Text);
    restore_collection!(QueueMediaRole::Keyframes, keyframes, Keyframe);
    restore_scalar!(QueueMediaRole::HdrExrDir, hdr_exr_dir, Text);
    restore_scalar!(QueueMediaRole::Lora, lora, Lora);
    restore_collection!(QueueMediaRole::Loras, loras, Lora);

    if let Some((&role, _)) = grouped.iter().next() {
        return Err(QueueMediaError::MalformedRecords(role));
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_extraction_round_trips_every_request_authority_without_json_leakage() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a patient red fox",
            "model": "mock-with-reference-shaped-fields",
            "width": 512,
            "height": 512,
            "steps": 8,
            "guidance": 3.0,
            "source_image": "c291cmNlLWJ5dGVz",
            "source_image_name": "source-private.png",
            "id_image": "ZmFjZS1ieXRlcw==",
            "id_image_name": "biometric-singular.png",
            "id_images": ["ZmFjZS0x", "ZmFjZS0y"],
            "id_image_names": ["biometric-one.png", "biometric-two.png"],
            "edit_images": ["ZWRpdC0x", "ZWRpdC0y"],
            "references": [
                {
                    "kind": "image",
                    "media": { "authority": "inline", "data": "cmVmLWJ5dGVz" },
                    "provenance": { "name": "reference-secret.png", "sha256": "11".repeat(32) },
                    "mime_type": "image/png",
                    "width": 320,
                    "height": 240
                },
                {
                    "kind": "audio",
                    "media": { "authority": "upload", "handle": "upload-bearer-secret" },
                    "provenance": { "name": "voice-secret.wav", "sha256": "22".repeat(32) },
                    "mime_type": "audio/wav",
                    "duration_ms": 1000,
                    "sample_rate": 48000,
                    "channels": 2,
                    "sample_count": 48000
                },
                {
                    "kind": "video",
                    "media": { "authority": "server_path", "path": "/private/reference.mp4" },
                    "provenance": { "name": "video-secret.mp4", "sha256": "33".repeat(32) },
                    "mime_type": "video/mp4",
                    "width": 640,
                    "height": 360,
                    "frame_count": 25,
                    "duration_ms": 1000,
                    "fps": 24.0
                },
                {
                    "kind": "image",
                    "media": { "authority": "descriptor" },
                    "provenance": { "name": "resolved-secret.png", "sha256": "44".repeat(32) },
                    "mime_type": "image/png",
                    "width": 128,
                    "height": 128
                }
            ],
            "mask_image": "bWFzay1ieXRlcw==",
            "control_image": "Y29udHJvbC1ieXRlcw==",
            "audio_file": "YXVkaW8tYnl0ZXM=",
            "audio_file_path": "/private/audio.wav",
            "source_video": "c291cmNlLXZpZGVv",
            "source_video_path": "/private/source.mp4",
            "extend_video": "ZXh0ZW5kLXZpZGVv",
            "extend_video_path": "/private/extend.mp4",
            "keyframes": [
                { "frame": 0, "image": "a2V5ZnJhbWU=", "name": "keyframe-secret.png" }
            ],
            "hdr_exr_dir": "/private/exr-output",
            "lora": { "path": "/private/singular-lora.safetensors", "scale": 0.5 },
            "loras": [
                { "path": "/private/stack-a.safetensors", "scale": 0.7 },
                { "path": "/private/stack-b.safetensors", "scale": 0.8, "expert": "high" }
            ]
        }))
        .unwrap();
        let expected = serde_json::to_value(&request).unwrap();

        let extracted = extract_request_fields("job-a".to_string(), request).unwrap();

        let durable: serde_json::Value = serde_json::from_str(extracted.request_json()).unwrap();
        let durable = durable.as_object().unwrap();
        for field in REQUEST_AUTHORITY_JSON_FIELDS {
            assert!(
                !durable.contains_key(*field),
                "{field} leaked into durable JSON"
            );
        }
        let json = extracted.request_json();
        for secret in [
            "source-private.png",
            "biometric-singular.png",
            "biometric-one.png",
            "reference-secret.png",
            "resolved-secret.png",
            "upload-bearer-secret",
            "/private/reference.mp4",
            "11".repeat(32).as_str(),
            "44".repeat(32).as_str(),
            "/private/audio.wav",
            "/private/exr-output",
            "/private/singular-lora.safetensors",
        ] {
            assert!(
                !json.contains(secret),
                "secret leaked into durable JSON: {secret}"
            );
        }

        let restored = extracted.rehydrate("job-a").unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), expected);
    }

    #[test]
    fn present_empty_collections_round_trip_without_becoming_absent() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "empty collections are still explicit",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "id_images": [],
            "id_image_names": [],
            "edit_images": [],
            "references": [],
            "keyframes": [],
            "loras": []
        }))
        .unwrap();
        let expected = serde_json::to_value(&request).unwrap();

        let extracted = extract_request_fields("job-empty".to_string(), request).unwrap();
        let restored = extracted.rehydrate("job-empty").unwrap();

        assert_eq!(serde_json::to_value(restored).unwrap(), expected);
    }

    #[test]
    fn records_are_job_scoped_and_not_cross_job_rehydratable() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "job scoped",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": "c2VjcmV0"
        }))
        .unwrap();
        let extracted =
            extract_request_media("job-one", request, &ProcessPrivateAuthorities::none()).unwrap();

        assert!(matches!(
            extracted.rehydrate("job-two"),
            Err(QueueMediaError::JobScopeMismatch)
        ));
    }

    #[test]
    fn process_private_authorities_are_explicitly_unsupported() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "private authority",
            "model": "minimax-h3-ref2va:comfy-pruned-int8",
            "width": 64,
            "height": 64,
            "steps": 1
        }))
        .unwrap();

        for authority in [
            ProcessPrivateAuthority::ResolvedReferenceStaging,
            ProcessPrivateAuthority::H3PrivateIngressGrant,
        ] {
            let result = extract_request_media(
                "job-private",
                request.clone(),
                &ProcessPrivateAuthorities::from_authority(authority),
            );
            assert!(matches!(
                result,
                Err(QueueMediaError::UnsupportedProcessPrivateAuthority(actual)) if actual == authority
            ));
        }
    }

    #[test]
    fn request_itself_fail_closed_classifies_current_h3_and_reference_authority() {
        let base = serde_json::json!({
            "prompt": "derived authority",
            "model": "minimax-h3-ref2va:comfy-pruned-int8",
            "width": 64,
            "height": 64,
            "steps": 1
        });
        let h3: mold_core::GenerateRequest = serde_json::from_value(base.clone()).unwrap();
        assert!(matches!(
            extract_request_media("job-h3", h3, &ProcessPrivateAuthorities::none()),
            Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
                ProcessPrivateAuthority::H3PrivateIngressGrant
            ))
        ));

        let mut reference = base;
        reference["model"] = serde_json::json!("mock");
        reference["references"] = serde_json::json!([{
            "kind": "image",
            "media": { "authority": "upload", "handle": "temporary-handle" },
            "provenance": { "sha256": "55".repeat(32) },
            "mime_type": "image/png",
            "width": 64,
            "height": 64
        }]);
        let reference: mold_core::GenerateRequest = serde_json::from_value(reference).unwrap();
        assert!(matches!(
            extract_request_media(
                "job-reference",
                reference,
                &ProcessPrivateAuthorities::none()
            ),
            Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
                ProcessPrivateAuthority::ResolvedReferenceStaging
            ))
        ));
    }

    #[test]
    fn pre_dispatch_lora_and_hdr_authority_remain_non_durable() {
        let base = serde_json::json!({
            "prompt": "pre-dispatch authority",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1
        });
        for (field, value, expected) in [
            (
                "lora",
                serde_json::json!({ "path": "/private/one.safetensors", "scale": 0.5 }),
                QueueMediaRole::Lora,
            ),
            (
                "loras",
                serde_json::json!([{ "path": "/private/two.safetensors", "scale": 0.7 }]),
                QueueMediaRole::Loras,
            ),
            (
                "hdr_exr_dir",
                serde_json::json!("/private/hdr"),
                QueueMediaRole::HdrExrDir,
            ),
        ] {
            let mut value_request = base.clone();
            value_request[field] = value;
            let request: mold_core::GenerateRequest =
                serde_json::from_value(value_request).unwrap();
            assert!(matches!(
                extract_request_media(
                    format!("job-{field}"),
                    request,
                    &ProcessPrivateAuthorities::none()
                ),
                Err(QueueMediaError::UnsupportedPreDispatchAuthority(actual)) if actual == expected
            ));
        }
    }

    #[test]
    fn job_scope_must_be_non_empty() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "scoped",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1
        }))
        .unwrap();

        assert!(matches!(
            extract_request_media("  ", request, &ProcessPrivateAuthorities::none()),
            Err(QueueMediaError::InvalidJobScope)
        ));
    }

    #[test]
    fn opaque_container_keeps_its_job_binding_through_the_storage_handoff() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "storage seam",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": "c2VjcmV0"
        }))
        .unwrap();
        let expected = serde_json::to_value(&request).unwrap();
        let extracted =
            extract_request_media("job-store", request, &ProcessPrivateAuthorities::none())
                .unwrap();
        let (request_json, media) = extracted.into_parts();
        assert_eq!(media.job_id(), "job-store");

        let restored = rehydrate_request_media("job-store", &request_json, media).unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), expected);
    }

    #[test]
    fn rehydration_rejects_even_null_authority_fields_in_durable_json() {
        let contaminated = serde_json::json!({
            "prompt": "contaminated",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": null
        })
        .to_string();
        let media = OpaqueQueueMedia {
            job_id: "job-contaminated".to_string(),
            records: Vec::new(),
        };

        assert!(matches!(
            rehydrate_request_media("job-contaminated", &contaminated, media),
            Err(QueueMediaError::RequestJsonContainsAuthority(
                "source_image"
            ))
        ));
    }

    #[test]
    fn opaque_record_debug_never_renders_payloads() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "redacted debug",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": "c2VjcmV0LWJ5dGVz",
            "source_image_name": "secret-name.png"
        }))
        .unwrap();
        let extracted =
            extract_request_media("job-debug", request, &ProcessPrivateAuthorities::none())
                .unwrap();

        let debug = format!("{:?}", extracted.media());
        assert!(!debug.contains("secret-bytes"));
        assert!(!debug.contains("secret-name.png"));
        assert!(debug.contains("records"));
    }
}
