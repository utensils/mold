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

#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use mold_core::{GenerationReference, KeyframeCondition, LoraWeight};
use zeroize::Zeroize;

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

/// The authoritative predicate for request fields transported by the
/// encrypted durable-media store. Process-private references and local-only
/// HDR/LoRA authorities are intentionally classified separately.
pub(crate) fn request_has_extractable_media(request: &mold_core::GenerateRequest) -> bool {
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
    durable_replacements: Vec<ProcessPrivateAuthority>,
}

impl ProcessPrivateAuthorities {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn from_authority(authority: ProcessPrivateAuthority) -> Self {
        Self {
            authorities: vec![authority],
            durable_replacements: Vec::new(),
        }
    }

    /// Declare that admission replaced the live H3 grant with its validated,
    /// one-way replay subject. Extraction remains fail-closed for H3 unless
    /// this explicit durable authority is present.
    pub(crate) fn with_durable_replacement(
        mut self,
        authority: Option<ProcessPrivateAuthority>,
    ) -> Self {
        if let Some(authority) = authority {
            self.durable_replacements.push(authority);
        }
        self
    }

    pub fn from_present(resolved_reference_staging: bool, h3_private_ingress_grant: bool) -> Self {
        let mut authorities = Vec::new();
        if resolved_reference_staging {
            authorities.push(ProcessPrivateAuthority::ResolvedReferenceStaging);
        }
        if h3_private_ingress_grant {
            authorities.push(ProcessPrivateAuthority::H3PrivateIngressGrant);
        }
        Self {
            authorities,
            durable_replacements: Vec::new(),
        }
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
    #[error("deferred media cannot overlay a request that already carries authority field {0}")]
    OverlayAuthorityConflict(&'static str),
    #[error("decrypted queue-media record has an invalid role or position label")]
    InvalidStoredRecordLabel,
    #[error("decrypted queue-media text is not UTF-8")]
    InvalidStoredText,
    #[error("decrypted queue-media path is not representable as UTF-8")]
    InvalidStoredPath,
    #[error("queue-media source could not be safely opened: {0}")]
    SealSource(#[source] crate::queue_media_store::QueueMediaError),
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

impl QueueMediaRole {
    fn wire_label(self) -> &'static str {
        match self {
            Self::SourceImage => "source_image",
            Self::SourceImageName => "source_image_name",
            Self::IdentityImage => "identity_image",
            Self::IdentityImageName => "identity_image_name",
            Self::IdentityImages => "identity_images",
            Self::IdentityImageNames => "identity_image_names",
            Self::EditImages => "edit_images",
            Self::References => "references",
            Self::MaskImage => "mask_image",
            Self::ControlImage => "control_image",
            Self::AudioFile => "audio_file",
            Self::AudioFilePath => "audio_file_path",
            Self::SourceVideo => "source_video",
            Self::SourceVideoPath => "source_video_path",
            Self::ExtendVideo => "extend_video",
            Self::ExtendVideoPath => "extend_video_path",
            Self::Keyframes => "keyframes",
            Self::HdrExrDir => "hdr_exr_dir",
            Self::Lora => "lora",
            Self::Loras => "loras",
        }
    }

    fn from_wire_label(value: &str) -> Option<Self> {
        Some(match value {
            "source_image" => Self::SourceImage,
            "source_image_name" => Self::SourceImageName,
            "identity_image" => Self::IdentityImage,
            "identity_image_name" => Self::IdentityImageName,
            "identity_images" => Self::IdentityImages,
            "identity_image_names" => Self::IdentityImageNames,
            "edit_images" => Self::EditImages,
            "references" => Self::References,
            "mask_image" => Self::MaskImage,
            "control_image" => Self::ControlImage,
            "audio_file" => Self::AudioFile,
            "audio_file_path" => Self::AudioFilePath,
            "source_video" => Self::SourceVideo,
            "source_video_path" => Self::SourceVideoPath,
            "extend_video" => Self::ExtendVideo,
            "extend_video_path" => Self::ExtendVideoPath,
            "keyframes" => Self::Keyframes,
            "hdr_exr_dir" => Self::HdrExrDir,
            "lora" => Self::Lora,
            "loras" => Self::Loras,
            _ => return None,
        })
    }

    fn is_path_shaped(self) -> bool {
        matches!(
            self,
            Self::AudioFilePath | Self::SourceVideoPath | Self::ExtendVideoPath
        )
    }
}

impl QueueMediaPosition {
    fn wire_label(self) -> String {
        match self {
            Self::Scalar => "scalar".to_string(),
            Self::Collection => "collection".to_string(),
            Self::Item(index) => format!("item:{index}"),
        }
    }

    fn from_wire_label(value: &str) -> Option<Self> {
        match value {
            "scalar" => Some(Self::Scalar),
            "collection" => Some(Self::Collection),
            _ => value
                .strip_prefix("item:")
                .and_then(|index| index.parse().ok())
                .map(Self::Item),
        }
    }
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
    #[cfg(test)]
    scrub_probe: Option<(usize, Arc<AtomicBool>)>,
}

impl OpaqueQueueMedia {
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    pub fn records(&self) -> &[OpaqueQueueMediaRecord] {
        &self.records
    }

    #[cfg(test)]
    fn with_scrub_probe(mut self, scrubbed: Arc<AtomicBool>) -> Self {
        self.scrub_probe = Some((self.records.len(), scrubbed));
        self
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

impl Drop for OpaqueQueueMedia {
    fn drop(&mut self) {
        scrub_opaque_records(&mut self.records);
        #[cfg(test)]
        if let Some((expected_records, scrubbed)) = &self.scrub_probe {
            scrubbed.store(
                self.records.len() == *expected_records
                    && self
                        .records
                        .iter()
                        .all(|record| payload_is_scrubbed(&record.payload)),
                Ordering::SeqCst,
            );
        }
    }
}

#[cfg(test)]
fn payload_is_scrubbed(payload: &QueueMediaPayload) -> bool {
    fn bytes_are_scrubbed(bytes: &[u8]) -> bool {
        bytes.iter().all(|byte| *byte == 0)
    }

    match payload {
        QueueMediaPayload::Presence => true,
        QueueMediaPayload::Bytes(bytes) => bytes_are_scrubbed(bytes),
        QueueMediaPayload::Text(text) => bytes_are_scrubbed(text.as_bytes()),
        QueueMediaPayload::Keyframe(keyframe) => {
            bytes_are_scrubbed(&keyframe.image)
                && keyframe
                    .name
                    .as_ref()
                    .is_none_or(|name| bytes_are_scrubbed(name.as_bytes()))
        }
        QueueMediaPayload::Reference(reference) => {
            let (media, provenance, mime_type) = match reference {
                GenerationReference::Image {
                    media,
                    provenance,
                    mime_type,
                    ..
                }
                | GenerationReference::Video {
                    media,
                    provenance,
                    mime_type,
                    ..
                }
                | GenerationReference::Audio {
                    media,
                    provenance,
                    mime_type,
                    ..
                } => (media, provenance, mime_type),
            };
            let media_scrubbed = match media {
                mold_core::GenerationReferenceAuthority::Inline { data } => {
                    bytes_are_scrubbed(data)
                }
                mold_core::GenerationReferenceAuthority::Upload { handle } => {
                    bytes_are_scrubbed(handle.as_bytes())
                }
                mold_core::GenerationReferenceAuthority::ServerPath { path } => {
                    bytes_are_scrubbed(path.as_bytes())
                }
                mold_core::GenerationReferenceAuthority::Descriptor => true,
            };
            media_scrubbed
                && provenance
                    .name
                    .as_ref()
                    .is_none_or(|name| bytes_are_scrubbed(name.as_bytes()))
                && provenance
                    .sha256
                    .as_ref()
                    .is_none_or(|digest| bytes_are_scrubbed(digest.as_bytes()))
                && bytes_are_scrubbed(mime_type.as_bytes())
        }
        QueueMediaPayload::Lora(lora) => bytes_are_scrubbed(lora.path.as_bytes()),
    }
}

fn scrub_payload(payload: &mut QueueMediaPayload) {
    match payload {
        QueueMediaPayload::Bytes(bytes) => bytes.zeroize(),
        QueueMediaPayload::Text(text) => text.zeroize(),
        QueueMediaPayload::Keyframe(keyframe) => {
            keyframe.image.zeroize();
            if let Some(name) = &mut keyframe.name {
                name.zeroize();
            }
        }
        QueueMediaPayload::Reference(reference) => scrub_reference(reference),
        QueueMediaPayload::Lora(lora) => lora.path.zeroize(),
        QueueMediaPayload::Presence => {}
    }
}

fn scrub_reference(reference: &mut GenerationReference) {
    let (media, provenance, mime_type) = match reference {
        GenerationReference::Image {
            media,
            provenance,
            mime_type,
            ..
        }
        | GenerationReference::Video {
            media,
            provenance,
            mime_type,
            ..
        }
        | GenerationReference::Audio {
            media,
            provenance,
            mime_type,
            ..
        } => (media, provenance, mime_type),
    };
    match media {
        mold_core::GenerationReferenceAuthority::Inline { data } => data.zeroize(),
        mold_core::GenerationReferenceAuthority::Upload { handle } => handle.zeroize(),
        mold_core::GenerationReferenceAuthority::ServerPath { path } => path.zeroize(),
        mold_core::GenerationReferenceAuthority::Descriptor => {}
    }
    if let Some(name) = &mut provenance.name {
        name.zeroize();
    }
    if let Some(digest) = &mut provenance.sha256 {
        digest.zeroize();
    }
    mime_type.zeroize();
}

fn scrub_opaque_records(records: &mut [OpaqueQueueMediaRecord]) {
    for record in records {
        scrub_payload(&mut record.payload);
    }
}

/// Wipe every request field that the durable queue-media overlay can restore.
///
/// This is intentionally the same exhaustive authority set as
/// `extract_request_fields`. Attempt-scoped runtime guards call it before
/// releasing private staging, and zeroizing request clones call it on every
/// success/error exit from downstream worker ownership.
pub(crate) fn scrub_request_media(request: &mut mold_core::GenerateRequest) {
    fn scrub_bytes(value: &mut Option<Vec<u8>>) {
        if let Some(bytes) = value {
            bytes.zeroize();
        }
        *value = None;
    }

    fn scrub_text(value: &mut Option<String>) {
        if let Some(text) = value {
            text.zeroize();
        }
        *value = None;
    }

    fn scrub_byte_collection(value: &mut Option<Vec<Vec<u8>>>) {
        if let Some(items) = value {
            for item in items.iter_mut() {
                item.zeroize();
            }
            items.clear();
        }
        *value = None;
    }

    fn scrub_text_collection(value: &mut Option<Vec<String>>) {
        if let Some(items) = value {
            for item in items.iter_mut() {
                item.zeroize();
            }
            items.clear();
        }
        *value = None;
    }

    scrub_bytes(&mut request.source_image);
    scrub_text(&mut request.source_image_name);
    scrub_bytes(&mut request.id_image);
    scrub_text(&mut request.id_image_name);
    scrub_byte_collection(&mut request.id_images);
    scrub_text_collection(&mut request.id_image_names);
    scrub_byte_collection(&mut request.edit_images);
    if let Some(references) = &mut request.references {
        for reference in references.iter_mut() {
            scrub_reference(reference);
        }
        references.clear();
    }
    request.references = None;
    scrub_bytes(&mut request.mask_image);
    scrub_bytes(&mut request.control_image);
    scrub_bytes(&mut request.audio_file);
    scrub_text(&mut request.audio_file_path);
    scrub_bytes(&mut request.source_video);
    scrub_text(&mut request.source_video_path);
    scrub_bytes(&mut request.extend_video);
    scrub_text(&mut request.extend_video_path);
    if let Some(keyframes) = &mut request.keyframes {
        for keyframe in keyframes.iter_mut() {
            keyframe.image.zeroize();
            if let Some(name) = &mut keyframe.name {
                name.zeroize();
            }
        }
        keyframes.clear();
    }
    request.keyframes = None;
    scrub_text(&mut request.hdr_exr_dir);
    if let Some(lora) = &mut request.lora {
        lora.path.zeroize();
    }
    request.lora = None;
    if let Some(loras) = &mut request.loras {
        for lora in loras.iter_mut() {
            lora.path.zeroize();
        }
        loras.clear();
    }
    request.loras = None;
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
    if mold_core::minimax_h3::capability_contract_for_model(&request.model).is_some()
        && !process_private
            .durable_replacements
            .contains(&ProcessPrivateAuthority::H3PrivateIngressGrant)
    {
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
        media: OpaqueQueueMedia {
            job_id,
            records,
            #[cfg(test)]
            scrub_probe: None,
        },
    })
}

/// Derive the exact payload-free facts authenticated by the V2 first record.
pub fn project_request_media(
    media: &OpaqueQueueMedia,
) -> Result<crate::queue_media_store::QueueMediaProjection, QueueMediaError> {
    use crate::queue_media_store::{ProjectedImageDimensions, QueueMediaProjection};

    let mut projection = QueueMediaProjection::default();
    let mut identity_count = 0_u32;
    for record in &media.records {
        match (record.role, record.position, &record.payload) {
            (
                QueueMediaRole::SourceImage,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                projection.source_image = true;
            }
            (
                QueueMediaRole::SourceVideo,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                projection.source_video_inline = true;
            }
            (
                QueueMediaRole::SourceVideoPath,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Text(_),
            ) => {
                projection.source_video_path = true;
            }
            (
                QueueMediaRole::ExtendVideo,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                projection.extend_video_inline = true;
            }
            (
                QueueMediaRole::ExtendVideoPath,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Text(_),
            ) => {
                projection.extend_video_path = true;
            }
            (
                QueueMediaRole::IdentityImage,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                identity_count = identity_count
                    .checked_add(1)
                    .ok_or(QueueMediaError::CollectionTooLarge)?;
            }
            (
                QueueMediaRole::IdentityImages,
                QueueMediaPosition::Item(_),
                QueueMediaPayload::Bytes(_),
            ) => {
                identity_count = identity_count
                    .checked_add(1)
                    .ok_or(QueueMediaError::CollectionTooLarge)?;
            }
            (
                QueueMediaRole::EditImages,
                QueueMediaPosition::Item(_),
                QueueMediaPayload::Bytes(bytes),
            ) => {
                projection.edit_image_count = projection
                    .edit_image_count
                    .checked_add(1)
                    .ok_or(QueueMediaError::CollectionTooLarge)?;
                if projection.edit_images.len()
                    < crate::queue_media_store::PROJECTED_EDIT_DIMENSION_SLOTS
                {
                    let dimensions = image::ImageReader::new(std::io::Cursor::new(bytes))
                        .with_guessed_format()
                        .ok()
                        .and_then(|reader| reader.into_dimensions().ok())
                        .map_or(
                            ProjectedImageDimensions::UnreadableHeader,
                            |(width, height)| ProjectedImageDimensions::Known { width, height },
                        );
                    projection.edit_images.push(dimensions);
                }
            }
            (
                QueueMediaRole::Keyframes,
                QueueMediaPosition::Item(_),
                QueueMediaPayload::Keyframe(_),
            ) => {
                projection.keyframe_count = projection
                    .keyframe_count
                    .checked_add(1)
                    .ok_or(QueueMediaError::CollectionTooLarge)?;
            }
            (
                QueueMediaRole::MaskImage,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                projection.mask_image = true;
            }
            (
                QueueMediaRole::ControlImage,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                projection.control_image = true;
            }
            (
                QueueMediaRole::AudioFile,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Bytes(_),
            ) => {
                projection.audio_inline = true;
            }
            (
                QueueMediaRole::AudioFilePath,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Text(_),
            ) => {
                projection.audio_path = true;
            }
            _ => {}
        }
    }
    projection.identity_present = identity_count > 0;
    projection.identity_photograph_count = identity_count;
    Ok(projection)
}

/// Convert opaque extracted records into store inputs without retaining a
/// second plaintext request. Path-shaped roles are safe-opened by the store;
/// all other bytes remain memory-sink records when hydrated.
pub fn into_seal_media(
    mut media: OpaqueQueueMedia,
) -> Result<Vec<crate::queue_media_store::SealMedia>, QueueMediaError> {
    use crate::queue_media_store::SealMedia;

    // Complete every fallible check while `media` still owns all plaintext, so
    // its Drop guard can scrub the full set on error. Serialized keyframes stay
    // zeroizing until their bytes move into the successful result.
    validate_record_topology(&media.records)?;
    for record in &media.records {
        match &record.payload {
            QueueMediaPayload::Reference(_) => {
                return Err(QueueMediaError::UnsupportedPreDispatchAuthority(
                    QueueMediaRole::References,
                ));
            }
            QueueMediaPayload::Lora(_) => {
                return Err(QueueMediaError::UnsupportedPreDispatchAuthority(
                    record.role,
                ));
            }
            _ => {}
        }
    }
    let mut serialized_keyframes = media
        .records
        .iter()
        .map(|record| match &record.payload {
            QueueMediaPayload::Keyframe(value) => serde_json::to_vec(value)
                .map(zeroize::Zeroizing::new)
                .map(Some)
                .map_err(QueueMediaError::Serialize),
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(unix)]
    let opened_paths = media
        .records
        .iter()
        .map(|record| match &record.payload {
            QueueMediaPayload::Text(value) if record.role.is_path_shaped() => {
                SealMedia::preopen_path(std::path::Path::new(value))
                    .map(Some)
                    .map_err(QueueMediaError::SealSource)
            }
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(not(unix))]
    let opened_paths = {
        if media.records.iter().any(|record| {
            record.role.is_path_shaped() && matches!(record.payload, QueueMediaPayload::Text(_))
        }) {
            return Err(QueueMediaError::SealSource(
                crate::queue_media_store::QueueMediaError::SecurityUnavailable(
                    "path-shaped queue media requires Unix safe-open and path scrubbing".into(),
                ),
            ));
        }
        std::iter::repeat_with(|| None::<()>)
            .take(media.records.len())
            .collect::<Vec<_>>()
    };

    let mut sealed = Vec::with_capacity(media.records.len());
    for ((record, serialized_keyframe), opened_path) in std::mem::take(&mut media.records)
        .into_iter()
        .zip(&mut serialized_keyframes)
        .zip(opened_paths)
    {
        let role = record.role.wire_label();
        let position = record.position.wire_label();
        let item = match record.payload {
            QueueMediaPayload::Presence => SealMedia::bytes(role, position, Vec::new()),
            QueueMediaPayload::Bytes(bytes) => SealMedia::bytes(role, position, bytes),
            QueueMediaPayload::Text(value) if record.role.is_path_shaped() => {
                #[cfg(unix)]
                {
                    let mut value = value;
                    value.zeroize();
                    SealMedia::from_preopened_path(
                        role,
                        position,
                        opened_path.expect("path source was pre-opened before plaintext moved"),
                    )
                }
                #[cfg(not(unix))]
                {
                    let mut value = value;
                    value.zeroize();
                    unreachable!("non-Unix path authority was rejected before plaintext moved")
                }
            }
            QueueMediaPayload::Text(value) => SealMedia::bytes(role, position, value.into_bytes()),
            QueueMediaPayload::Keyframe(_) => SealMedia::bytes(
                role,
                position,
                std::mem::take(
                    serialized_keyframe
                        .as_mut()
                        .expect("keyframe serialization was precomputed")
                        .as_mut(),
                ),
            ),
            QueueMediaPayload::Reference(_) | QueueMediaPayload::Lora(_) => {
                unreachable!("unsupported payloads were rejected before moving plaintext")
            }
        };
        sealed.push(item);
    }
    Ok(sealed)
}

pub(crate) fn decrypted_media_into_opaque(
    job_id: &str,
    decrypted: &mut crate::queue_media_store::DecryptedQueueMediaSet,
) -> Result<OpaqueQueueMedia, QueueMediaError> {
    use crate::queue_media_store::DecryptedQueueMediaPayload;

    let mut records = OpaqueRecordsGuard(Vec::with_capacity(decrypted.media.len()));
    while let Some(item) = decrypted.media.last() {
        let role = QueueMediaRole::from_wire_label(&item.role)
            .ok_or(QueueMediaError::InvalidStoredRecordLabel)?;
        let position = QueueMediaPosition::from_wire_label(&item.name)
            .ok_or(QueueMediaError::InvalidStoredRecordLabel)?;
        let item = decrypted.media.pop().expect("last item exists");
        let payload = match item.payload {
            DecryptedQueueMediaPayload::PrivatePath(path) if role.is_path_shaped() => {
                QueueMediaPayload::Text(
                    path.into_os_string()
                        .into_string()
                        .map_err(|_| QueueMediaError::InvalidStoredPath)?,
                )
            }
            DecryptedQueueMediaPayload::PrivatePath(_) => {
                return Err(QueueMediaError::MalformedRecords(role))
            }
            DecryptedQueueMediaPayload::Bytes(bytes)
                if position == QueueMediaPosition::Collection && bytes.is_empty() =>
            {
                QueueMediaPayload::Presence
            }
            DecryptedQueueMediaPayload::Bytes(mut bytes) => match role {
                QueueMediaRole::SourceImage
                | QueueMediaRole::IdentityImage
                | QueueMediaRole::IdentityImages
                | QueueMediaRole::EditImages
                | QueueMediaRole::MaskImage
                | QueueMediaRole::ControlImage
                | QueueMediaRole::AudioFile
                | QueueMediaRole::SourceVideo
                | QueueMediaRole::ExtendVideo => QueueMediaPayload::Bytes(bytes),
                QueueMediaRole::SourceImageName
                | QueueMediaRole::IdentityImageName
                | QueueMediaRole::IdentityImageNames => match String::from_utf8(bytes) {
                    Ok(text) => QueueMediaPayload::Text(text),
                    Err(error) => {
                        let mut bytes = error.into_bytes();
                        bytes.zeroize();
                        return Err(QueueMediaError::InvalidStoredText);
                    }
                },
                QueueMediaRole::Keyframes => {
                    let result = serde_json::from_slice(&bytes);
                    bytes.zeroize();
                    QueueMediaPayload::Keyframe(result.map_err(QueueMediaError::Deserialize)?)
                }
                _ => {
                    bytes.zeroize();
                    return Err(QueueMediaError::MalformedRecords(role));
                }
            },
        };
        records.0.push(OpaqueQueueMediaRecord {
            role,
            position,
            payload,
        });
    }
    records.0.reverse();
    Ok(OpaqueQueueMedia {
        job_id: job_id.to_string(),
        records: records.into_records(),
        #[cfg(test)]
        scrub_probe: None,
    })
}

struct OpaqueRecordsGuard(Vec<OpaqueQueueMediaRecord>);

impl OpaqueRecordsGuard {
    fn into_records(mut self) -> Vec<OpaqueQueueMediaRecord> {
        std::mem::take(&mut self.0)
    }
}

impl Drop for OpaqueRecordsGuard {
    fn drop(&mut self) {
        for record in &mut self.0 {
            match &mut record.payload {
                QueueMediaPayload::Bytes(bytes) => bytes.zeroize(),
                QueueMediaPayload::Text(text) => text.zeroize(),
                QueueMediaPayload::Keyframe(keyframe) => {
                    keyframe.image.zeroize();
                    if let Some(name) = &mut keyframe.name {
                        name.zeroize();
                    }
                }
                QueueMediaPayload::Presence
                | QueueMediaPayload::Reference(_)
                | QueueMediaPayload::Lora(_) => {}
            }
        }
    }
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

fn collection_role(role: QueueMediaRole) -> bool {
    matches!(
        role,
        QueueMediaRole::IdentityImages
            | QueueMediaRole::IdentityImageNames
            | QueueMediaRole::EditImages
            | QueueMediaRole::References
            | QueueMediaRole::Keyframes
            | QueueMediaRole::Loras
    )
}

fn record_payload_matches(record: &OpaqueQueueMediaRecord) -> bool {
    use QueueMediaPayload::{Bytes, Keyframe, Lora, Presence, Reference, Text};

    if collection_role(record.role) && record.position == QueueMediaPosition::Collection {
        return matches!(record.payload, Presence);
    }
    matches!(
        (record.role, record.position, &record.payload),
        (
            QueueMediaRole::SourceImage
                | QueueMediaRole::IdentityImage
                | QueueMediaRole::MaskImage
                | QueueMediaRole::ControlImage
                | QueueMediaRole::AudioFile
                | QueueMediaRole::SourceVideo
                | QueueMediaRole::ExtendVideo,
            QueueMediaPosition::Scalar,
            Bytes(_),
        ) | (
            QueueMediaRole::SourceImageName
                | QueueMediaRole::IdentityImageName
                | QueueMediaRole::AudioFilePath
                | QueueMediaRole::SourceVideoPath
                | QueueMediaRole::ExtendVideoPath
                | QueueMediaRole::HdrExrDir,
            QueueMediaPosition::Scalar,
            Text(_),
        ) | (
            QueueMediaRole::IdentityImages | QueueMediaRole::EditImages,
            QueueMediaPosition::Item(_),
            Bytes(_),
        ) | (
            QueueMediaRole::IdentityImageNames,
            QueueMediaPosition::Item(_),
            Text(_)
        ) | (
            QueueMediaRole::References,
            QueueMediaPosition::Item(_),
            Reference(_)
        ) | (
            QueueMediaRole::Keyframes,
            QueueMediaPosition::Item(_),
            Keyframe(_)
        ) | (QueueMediaRole::Lora, QueueMediaPosition::Scalar, Lora(_))
            | (QueueMediaRole::Loras, QueueMediaPosition::Item(_), Lora(_))
    )
}

fn validate_record_topology(records: &[OpaqueQueueMediaRecord]) -> Result<(), QueueMediaError> {
    let mut grouped: HashMap<QueueMediaRole, Vec<&OpaqueQueueMediaRecord>> = HashMap::new();
    for record in records {
        if !record_payload_matches(record) {
            return Err(QueueMediaError::MalformedRecords(record.role));
        }
        grouped.entry(record.role).or_default().push(record);
    }
    for (role, records) in grouped {
        if !collection_role(role) {
            if records.len() != 1 {
                return Err(QueueMediaError::MalformedRecords(role));
            }
            continue;
        }
        let mut presence = 0_usize;
        let mut items = BTreeMap::new();
        for record in records {
            match record.position {
                QueueMediaPosition::Collection => presence += 1,
                QueueMediaPosition::Item(index) => {
                    if items.insert(index, ()).is_some() {
                        return Err(QueueMediaError::MalformedRecords(role));
                    }
                }
                QueueMediaPosition::Scalar => {
                    return Err(QueueMediaError::MalformedRecords(role));
                }
            }
        }
        if presence != 1
            || (0_u32..)
                .zip(items.keys().copied())
                .any(|(expected, actual)| expected != actual)
        {
            return Err(QueueMediaError::MalformedRecords(role));
        }
    }
    Ok(())
}

/// Reconstruct the exact request semantics from its JSON-safe settings and
/// the records belonging to the same queue job.
pub fn rehydrate_request_media(
    expected_job_id: &str,
    request_json: &str,
    media: OpaqueQueueMedia,
) -> Result<mold_core::GenerateRequest, QueueMediaError> {
    ensure_json_is_authority_free(request_json)?;
    let mut request: mold_core::GenerateRequest =
        serde_json::from_str(request_json).map_err(QueueMediaError::Deserialize)?;
    rehydrate_request_media_into(expected_job_id, &mut request, media)?;
    Ok(request)
}

/// Overlay deferred media onto the scheduler-mutated request. Only extracted
/// media fields are touched; prompt/seed and every frozen plan/private
/// authority field retain their current values.
pub fn rehydrate_request_media_into(
    expected_job_id: &str,
    request: &mut mold_core::GenerateRequest,
    mut media: OpaqueQueueMedia,
) -> Result<(), QueueMediaError> {
    if media.job_id != expected_job_id {
        return Err(QueueMediaError::JobScopeMismatch);
    }
    for (present, field) in [
        (request.source_image.is_some(), "source_image"),
        (request.source_image_name.is_some(), "source_image_name"),
        (request.id_image.is_some(), "id_image"),
        (request.id_image_name.is_some(), "id_image_name"),
        (request.id_images.is_some(), "id_images"),
        (request.id_image_names.is_some(), "id_image_names"),
        (request.edit_images.is_some(), "edit_images"),
        (request.references.is_some(), "references"),
        (request.mask_image.is_some(), "mask_image"),
        (request.control_image.is_some(), "control_image"),
        (request.audio_file.is_some(), "audio_file"),
        (request.audio_file_path.is_some(), "audio_file_path"),
        (request.source_video.is_some(), "source_video"),
        (request.source_video_path.is_some(), "source_video_path"),
        (request.extend_video.is_some(), "extend_video"),
        (request.extend_video_path.is_some(), "extend_video_path"),
        (request.keyframes.is_some(), "keyframes"),
        (request.hdr_exr_dir.is_some(), "hdr_exr_dir"),
        (request.lora.is_some(), "lora"),
        (request.loras.is_some(), "loras"),
    ] {
        if present {
            return Err(QueueMediaError::OverlayAuthorityConflict(field));
        }
    }
    // Validate the complete topology and payload types before moving a single
    // plaintext value into the caller's request. All fallible exits above and
    // here leave `media` armed, whose Drop scrubs every byte payload.
    validate_record_topology(&media.records)?;

    let mut grouped: HashMap<_, Vec<_>> = HashMap::new();
    for record in std::mem::take(&mut media.records) {
        grouped.entry(record.role).or_default().push(record);
    }

    macro_rules! restore_scalar {
        ($role:expr, $field:ident, $variant:ident) => {
            if let Some(mut records) = grouped.remove(&$role) {
                let payload = records.pop().expect("validated scalar record").payload;
                let QueueMediaPayload::$variant(value) = payload else {
                    unreachable!("record payload type was validated before overlay")
                };
                request.$field = Some(value);
            }
        };
    }
    macro_rules! restore_collection {
        ($role:expr, $field:ident, $variant:ident) => {
            if let Some(records) = grouped.remove(&$role) {
                let mut items = BTreeMap::new();
                for record in records {
                    if let QueueMediaPosition::Item(index) = record.position {
                        let QueueMediaPayload::$variant(value) = record.payload else {
                            unreachable!("record payload type was validated before overlay")
                        };
                        items.insert(index, value);
                    }
                }
                request.$field = Some(items.into_values().collect());
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

    debug_assert!(grouped.is_empty(), "every validated role is restored");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    #[cfg(unix)]
    use std::io::Read as _;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(width, height)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

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
    fn projection_classifies_every_prelease_media_fact_without_payload() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "projection",
            "model": "flux2-dev:bf16",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": base64::engine::general_purpose::STANDARD.encode(png(4, 5)),
            "id_image": "ZmFjZQ==",
            "id_images": ["ZmFjZTE=", "ZmFjZTI="],
            "edit_images": [
                base64::engine::general_purpose::STANDARD.encode(png(320, 240)),
                "bm90LWltYWdl"
            ],
            "mask_image": "bWFzaw==",
            "control_image": "Y29udHJvbA==",
            "audio_file": "YXVkaW8=",
            "audio_file_path": "/private/audio.wav",
            "source_video": "dmlkZW8=",
            "source_video_path": "/private/source.mp4",
            "extend_video": "ZXh0ZW5k",
            "extend_video_path": "/private/extend.mp4",
            "keyframes": [{"frame": 0, "image": "a2V5"}]
        }))
        .unwrap();
        let extracted = extract_request_fields("job-projection".into(), request).unwrap();
        let projection = project_request_media(extracted.media()).unwrap();

        assert!(projection.source_image);
        assert!(projection.source_video_inline && projection.source_video_path);
        assert!(projection.extend_video_inline && projection.extend_video_path);
        assert_eq!(projection.keyframe_count, 1);
        assert!(projection.identity_present);
        assert_eq!(projection.identity_photograph_count, 3);
        assert_eq!(projection.edit_image_count, 2);
        assert_eq!(
            projection.edit_images,
            vec![
                crate::queue_media_store::ProjectedImageDimensions::Known {
                    width: 320,
                    height: 240
                },
                crate::queue_media_store::ProjectedImageDimensions::UnreadableHeader
            ]
        );
        assert!(projection.mask_image && projection.control_image);
        assert!(projection.audio_inline && projection.audio_path);
    }

    #[test]
    fn valid_qwen_edit_above_flux_slot_count_projects_total_without_extra_dimensions() {
        let image = png(8, 8);
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "five references",
            "model": "qwen-image-edit:bf16",
            "width": 512,
            "height": 512,
            "steps": 1,
            "batch_size": 1,
            "edit_images": (0..5)
                .map(|_| base64::engine::general_purpose::STANDARD.encode(&image))
                .collect::<Vec<_>>()
        }))
        .unwrap();
        mold_core::validation::validate_generate_request(&request).unwrap();
        let expected = serde_json::to_value(&request).unwrap();
        let extracted = extract_request_fields("job-qwen-five".into(), request).unwrap();
        let projection = project_request_media(extracted.media()).unwrap();

        assert_eq!(projection.edit_image_count, 5);
        assert_eq!(
            projection.edit_images.len(),
            crate::queue_media_store::PROJECTED_EDIT_DIMENSION_SLOTS
        );
        assert_eq!(
            serde_json::to_value(extracted.rehydrate("job-qwen-five").unwrap()).unwrap(),
            expected
        );
    }

    #[cfg(unix)]
    #[test]
    fn seal_conversion_scrubs_every_record_when_a_preopened_path_fails() {
        use std::os::unix::fs::symlink;

        fn record(
            role: QueueMediaRole,
            position: QueueMediaPosition,
            payload: QueueMediaPayload,
        ) -> OpaqueQueueMediaRecord {
            OpaqueQueueMediaRecord {
                role,
                position,
                payload,
            }
        }

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target.mp4");
        let link = temp.path().join("rejected-link.mp4");
        std::fs::write(&target, b"path-secret").unwrap();
        symlink(&target, &link).unwrap();
        let keyframe: KeyframeCondition = serde_json::from_value(serde_json::json!({
            "frame": 0,
            "image": "a2V5ZnJhbWUtc2VjcmV0",
            "name": "keyframe-name-secret.png"
        }))
        .unwrap();
        let scrubbed = Arc::new(AtomicBool::new(false));
        let media = OpaqueQueueMedia {
            job_id: "job-preopen-failure".into(),
            records: vec![
                record(
                    QueueMediaRole::SourceVideoPath,
                    QueueMediaPosition::Scalar,
                    QueueMediaPayload::Text(link.to_string_lossy().into_owned()),
                ),
                record(
                    QueueMediaRole::IdentityImage,
                    QueueMediaPosition::Scalar,
                    QueueMediaPayload::Bytes(b"later-identity-secret".to_vec()),
                ),
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Collection,
                    QueueMediaPayload::Presence,
                ),
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Item(0),
                    QueueMediaPayload::Bytes(b"later-edit-secret".to_vec()),
                ),
                record(
                    QueueMediaRole::Keyframes,
                    QueueMediaPosition::Collection,
                    QueueMediaPayload::Presence,
                ),
                record(
                    QueueMediaRole::Keyframes,
                    QueueMediaPosition::Item(0),
                    QueueMediaPayload::Keyframe(keyframe),
                ),
            ],
            scrub_probe: None,
        }
        .with_scrub_probe(Arc::clone(&scrubbed));

        assert!(matches!(
            into_seal_media(media),
            Err(QueueMediaError::SealSource(
                crate::queue_media_store::QueueMediaError::InsecurePath(_)
            ))
        ));
        assert!(
            scrubbed.load(Ordering::SeqCst),
            "the failing path and every later inline/name/keyframe record must remain under the opaque scrub owner"
        );
    }

    #[cfg(unix)]
    #[test]
    fn seal_conversion_keeps_preopened_paths_aligned_with_mixed_roles() {
        fn record(
            role: QueueMediaRole,
            position: QueueMediaPosition,
            payload: QueueMediaPayload,
        ) -> OpaqueQueueMediaRecord {
            OpaqueQueueMediaRecord {
                role,
                position,
                payload,
            }
        }

        let temp = tempfile::tempdir().unwrap();
        let audio = temp.path().join("audio.wav");
        let video = temp.path().join("video.mp4");
        std::fs::write(&audio, b"audio-path-bytes").unwrap();
        std::fs::write(&video, b"video-path-bytes").unwrap();
        let media = OpaqueQueueMedia {
            job_id: "job-preopen-order".into(),
            records: vec![
                record(
                    QueueMediaRole::AudioFilePath,
                    QueueMediaPosition::Scalar,
                    QueueMediaPayload::Text(audio.to_string_lossy().into_owned()),
                ),
                record(
                    QueueMediaRole::SourceImage,
                    QueueMediaPosition::Scalar,
                    QueueMediaPayload::Bytes(b"inline-between-paths".to_vec()),
                ),
                record(
                    QueueMediaRole::SourceVideoPath,
                    QueueMediaPosition::Scalar,
                    QueueMediaPayload::Text(video.to_string_lossy().into_owned()),
                ),
            ],
            scrub_probe: None,
        };

        let mut sealed = into_seal_media(media).unwrap();
        let mut opened = BTreeMap::new();
        for item in &mut sealed {
            if let crate::queue_media_store::SealMediaSource::OpenFile(file) = &mut item.source {
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).unwrap();
                opened.insert(item.role.clone(), bytes);
            }
        }
        assert_eq!(
            opened.get("audio_file_path").map(Vec::as_slice),
            Some(b"audio-path-bytes".as_slice())
        );
        assert_eq!(
            opened.get("source_video_path").map(Vec::as_slice),
            Some(b"video-path-bytes".as_slice())
        );
    }

    #[test]
    fn overlay_restores_media_without_reverting_scheduler_mutations() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "before preparation",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "seed": 7,
            "source_image": "c291cmNl",
            "id_image": "ZmFjZQ=="
        }))
        .unwrap();
        let extracted = extract_request_fields("job-overlay".into(), request).unwrap();
        let (json, media) = extracted.into_parts();
        let mut current: mold_core::GenerateRequest = serde_json::from_str(&json).unwrap();
        current.prompt = "expanded by scheduler".into();
        current.seed = Some(99);

        rehydrate_request_media_into("job-overlay", &mut current, media).unwrap();
        assert_eq!(current.prompt, "expanded by scheduler");
        assert_eq!(current.seed, Some(99));
        assert_eq!(current.source_image.as_deref(), Some(b"source".as_slice()));
        assert_eq!(current.id_image.as_deref(), Some(b"face".as_slice()));
    }

    #[test]
    fn overlay_rejects_scope_authority_and_malformed_topology_before_mutation() {
        fn request() -> mold_core::GenerateRequest {
            serde_json::from_value(serde_json::json!({
                "prompt": "atomic overlay",
                "model": "mock",
                "width": 64,
                "height": 64,
                "steps": 1,
                "seed": 17
            }))
            .unwrap()
        }
        fn record(
            role: QueueMediaRole,
            position: QueueMediaPosition,
            payload: QueueMediaPayload,
        ) -> OpaqueQueueMediaRecord {
            OpaqueQueueMediaRecord {
                role,
                position,
                payload,
            }
        }

        let malformed = vec![
            vec![
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Collection,
                    QueueMediaPayload::Presence,
                ),
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Item(0),
                    QueueMediaPayload::Bytes(b"first-secret".to_vec()),
                ),
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Item(0),
                    QueueMediaPayload::Bytes(b"duplicate-secret".to_vec()),
                ),
            ],
            vec![
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Collection,
                    QueueMediaPayload::Presence,
                ),
                record(
                    QueueMediaRole::EditImages,
                    QueueMediaPosition::Item(1),
                    QueueMediaPayload::Bytes(b"gapped-secret".to_vec()),
                ),
            ],
            vec![record(
                QueueMediaRole::SourceImage,
                QueueMediaPosition::Scalar,
                QueueMediaPayload::Text("wrong-payload-secret".into()),
            )],
            vec![
                record(
                    QueueMediaRole::SourceImage,
                    QueueMediaPosition::Scalar,
                    QueueMediaPayload::Bytes(b"early-secret".to_vec()),
                ),
                record(
                    QueueMediaRole::ControlImage,
                    QueueMediaPosition::Item(0),
                    QueueMediaPayload::Bytes(b"trailing-secret".to_vec()),
                ),
            ],
        ];
        for records in malformed {
            let mut current = request();
            let before = serde_json::to_value(&current).unwrap();
            let result = rehydrate_request_media_into(
                "job-atomic",
                &mut current,
                OpaqueQueueMedia {
                    job_id: "job-atomic".into(),
                    records,
                    scrub_probe: None,
                },
            );
            assert!(matches!(result, Err(QueueMediaError::MalformedRecords(_))));
            assert_eq!(serde_json::to_value(&current).unwrap(), before);
        }

        let mut wrong_job = request();
        let before = serde_json::to_value(&wrong_job).unwrap();
        assert!(matches!(
            rehydrate_request_media_into(
                "job-atomic",
                &mut wrong_job,
                OpaqueQueueMedia {
                    job_id: "job-other".into(),
                    records: vec![record(
                        QueueMediaRole::SourceImage,
                        QueueMediaPosition::Scalar,
                        QueueMediaPayload::Bytes(b"scope-secret".to_vec()),
                    )],
                    scrub_probe: None,
                },
            ),
            Err(QueueMediaError::JobScopeMismatch)
        ));
        assert_eq!(serde_json::to_value(&wrong_job).unwrap(), before);

        let mut conflict = request();
        conflict.source_image = Some(b"existing-authority".to_vec());
        let before = serde_json::to_value(&conflict).unwrap();
        assert!(matches!(
            rehydrate_request_media_into(
                "job-atomic",
                &mut conflict,
                OpaqueQueueMedia {
                    job_id: "job-atomic".into(),
                    records: vec![record(
                        QueueMediaRole::IdentityImage,
                        QueueMediaPosition::Scalar,
                        QueueMediaPayload::Bytes(b"conflict-secret".to_vec()),
                    )],
                    scrub_probe: None,
                },
            ),
            Err(QueueMediaError::OverlayAuthorityConflict("source_image"))
        ));
        assert_eq!(serde_json::to_value(&conflict).unwrap(), before);
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
            scrub_probe: None,
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
