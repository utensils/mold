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
//! An ordered MiniMax H3 reference is a DESCRIPTOR plus MEDIA. The descriptor
//! (`{authority: "descriptor"}` plus its content digest and probed shape) is an
//! ordinary request setting — it is exactly what gallery metadata already
//! records — so it stays in the durable JSON. The media is the staged file the
//! admission boundary resolved every public authority into; it seals into the
//! private-staging sink and hydrates back to one ordered private path per
//! descriptor. Upload handles, inline bytes, and server paths are consumed
//! before extraction and refused here by index if any survives.
//!
//! The H3 ingress grant is a different class of authority: it binds
//! authentication, instance, policy, and the exact request, and cannot be
//! reconstructed from media records, so extraction demands its sealed
//! replacement explicitly.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use mold_core::{GenerationReference, GenerationReferenceAuthority, KeyframeCondition, LoraWeight};
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
/// encrypted durable-media store. Local-only HDR/LoRA authorities are
/// intentionally classified separately.
pub(crate) fn request_has_extractable_media(request: &mold_core::GenerateRequest) -> bool {
    request.has_durable_media_inputs()
}

/// A process-private authority that media extraction cannot make durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessPrivateAuthority {
    /// Authenticated, instance- and policy-bound MiniMax H3 ingress proof.
    H3PrivateIngressGrant,
}

/// Explicit inventory supplied by the admission boundary.
///
/// This is intentionally not inferred from `GenerateRequest`: the private
/// ingress grant never enters the request at all, so only the boundary that
/// sealed its replay subject can vouch that it exists.
#[derive(Debug, Default)]
pub struct ProcessPrivateAuthorities {
    durable_replacements: Vec<ProcessPrivateAuthority>,
}

impl ProcessPrivateAuthorities {
    pub fn none() -> Self {
        Self::default()
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
    #[error("durable request JSON references[{0}] must be a digest-bearing descriptor")]
    RequestJsonReferenceAuthority(usize),
    #[error("{descriptors} reference descriptors do not match {staged} staged reference files")]
    ReferenceCountMismatch { descriptors: usize, staged: usize },
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
    /// A resolved reference's staged media file. Pre-seal it names the
    /// admission staging the resolver wrote; post-hydration it names the
    /// owner-only private staging the store decrypted into. Either way the
    /// path is process-private and never enters the request.
    StagedPath(PathBuf),
    Keyframe(KeyframeCondition),
    Lora(LoraWeight),
}

fn scrub_path(path: &mut PathBuf) {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt as _;
        let mut bytes = std::mem::take(path).into_os_string().into_vec();
        bytes.zeroize();
    }
    #[cfg(not(unix))]
    {
        *path = PathBuf::new();
    }
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
        QueueMediaPayload::StagedPath(path) => path.as_os_str().is_empty(),
        QueueMediaPayload::Lora(lora) => bytes_are_scrubbed(lora.path.as_bytes()),
    }
}

fn scrub_payload(payload: &mut QueueMediaPayload) {
    match payload {
        QueueMediaPayload::Bytes(bytes) => bytes.zeroize(),
        QueueMediaPayload::Text(text) => text.zeroize(),
        QueueMediaPayload::StagedPath(path) => scrub_path(path),
        QueueMediaPayload::Keyframe(keyframe) => {
            keyframe.image.zeroize();
            if let Some(name) = &mut keyframe.name {
                name.zeroize();
            }
        }
        QueueMediaPayload::Lora(lora) => lora.path.zeroize(),
        QueueMediaPayload::Presence => {}
    }
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
/// success/error exit from downstream worker ownership. Reference DESCRIPTORS
/// are deliberately left alone: they are settings, not media — byte-identical
/// to what `OutputMetadata.references` persists — and no public authority can
/// be present on them past admission.
/// The durable queue's scrub, owned by `mold_core::request_media` so the H3
/// admission identity in `mold-inference` hashes the same persisted form.
pub(crate) fn scrub_request_media(request: &mut mold_core::GenerateRequest) {
    mold_core::request_media::scrub_request_media(request);
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

    /// Reconstruct the request plus the ordered staged reference paths.
    pub fn rehydrate(
        self,
        expected_job_id: &str,
    ) -> Result<(mold_core::GenerateRequest, Vec<PathBuf>), QueueMediaError> {
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
/// `staged` is the resolver's staged media for `request.references`, one file
/// per descriptor in request order; it is required exactly when the request
/// carries a non-empty reference list.
pub fn extract_request_media(
    job_id: impl Into<String>,
    request: mold_core::GenerateRequest,
    process_private: &ProcessPrivateAuthorities,
    staged: Option<&crate::reference_uploads::StagedReferences>,
) -> Result<ExtractedQueueRequest, QueueMediaError> {
    let job_id = job_id.into();
    if job_id.trim().is_empty() {
        return Err(QueueMediaError::InvalidJobScope);
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
    // A LoRA is a server-local adapter path plus a scale. It seals as JSON
    // beside the conditioning media and is rehydrated by the feeder BEFORE
    // `prepare_generation_after_durable_ack`, which is where the scheduler
    // resolves exact adapter paths — so a replayed print plans with the
    // adapter it was admitted with rather than a stripped request. HDR EXR
    // output is a local-only authority the server API refuses at its own
    // boundary, exactly as it did before the durable path; it stays
    // unsupported here as well.
    if request.hdr_exr_dir.is_some() {
        return Err(QueueMediaError::UnsupportedPreDispatchAuthority(
            QueueMediaRole::HdrExrDir,
        ));
    }

    extract_request_fields(job_id, request, staged)
}

/// Refuse any reference that is not a digest-bearing descriptor. This is the
/// pin that keeps one-use upload handles, inline bytes, and server paths out
/// of `mold.db`: extraction asks it before the durable JSON is minted, and
/// hydration asks it again before trusting what came back.
pub(crate) fn ensure_references_are_descriptors(
    references: Option<&[GenerationReference]>,
) -> Result<(), QueueMediaError> {
    for (index, reference) in references.unwrap_or_default().iter().enumerate() {
        let descriptor = matches!(reference.media(), GenerationReferenceAuthority::Descriptor);
        if !descriptor || reference.provenance().sha256.is_none() {
            return Err(QueueMediaError::RequestJsonReferenceAuthority(index));
        }
    }
    Ok(())
}

fn ensure_reference_count(
    references: Option<&[GenerationReference]>,
    staged: usize,
) -> Result<(), QueueMediaError> {
    let descriptors = references.map_or(0, <[GenerationReference]>::len);
    if descriptors != staged {
        return Err(QueueMediaError::ReferenceCountMismatch {
            descriptors,
            staged,
        });
    }
    Ok(())
}

/// The field mapper stays separate from the authority preflight so its
/// exhaustive lossless property can be tested on its own.
fn extract_request_fields(
    job_id: String,
    request: mold_core::GenerateRequest,
    staged: Option<&crate::reference_uploads::StagedReferences>,
) -> Result<ExtractedQueueRequest, QueueMediaError> {
    ensure_references_are_descriptors(request.references.as_deref())?;
    let staged_paths = staged.map(|staged| staged.paths()).unwrap_or_default();
    ensure_reference_count(request.references.as_deref(), staged_paths.len())?;
    // Intentionally exhaustive: adding any GenerateRequest field fails this
    // build until it is classified as retained JSON or extracted authority.
    let mold_core::GenerateRequest {
        // Plain settings, retained in the durable JSON like `enable_audio`.
        // `mesh` carries no media of its own — it is five scalars — so it
        // survives a restart on the request rather than through the encrypted
        // media set, and a replayed job re-renders the same geometry.
        mesh,
        video_only,
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
    // The descriptors stay on the request; only the staged files are records.
    collection(
        &mut records,
        QueueMediaRole::References,
        references.as_ref().map(|_| staged_paths),
        QueueMediaPayload::StagedPath,
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
        references,
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
        video_only,
        mesh,
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
    // Keyframes and LoRA weights are structured records; both seal as their
    // JSON encoding and stay zeroizing until the bytes move into the result.
    let mut serialized_json = media
        .records
        .iter()
        .map(|record| match &record.payload {
            QueueMediaPayload::Keyframe(value) => serde_json::to_vec(value)
                .map(zeroize::Zeroizing::new)
                .map(Some)
                .map_err(QueueMediaError::Serialize),
            QueueMediaPayload::Lora(value) => serde_json::to_vec(value)
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
            QueueMediaPayload::StagedPath(path) => SealMedia::preopen_path(path)
                .map(Some)
                .map_err(QueueMediaError::SealSource),
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(not(unix))]
    let opened_paths = {
        if media.records.iter().any(|record| {
            matches!(record.payload, QueueMediaPayload::StagedPath(_))
                || (record.role.is_path_shaped()
                    && matches!(record.payload, QueueMediaPayload::Text(_)))
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
    for ((record, serialized_record), opened_path) in std::mem::take(&mut media.records)
        .into_iter()
        .zip(&mut serialized_json)
        .zip(opened_paths)
    {
        #[cfg(not(unix))]
        let _ = &opened_path;
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
            QueueMediaPayload::StagedPath(path) => {
                let mut path = path;
                scrub_path(&mut path);
                #[cfg(unix)]
                {
                    SealMedia::from_preopened_path(
                        role,
                        position,
                        opened_path
                            .expect("staged reference was pre-opened before plaintext moved"),
                    )
                }
                #[cfg(not(unix))]
                {
                    unreachable!("non-Unix staged references were rejected before plaintext moved")
                }
            }
            QueueMediaPayload::Keyframe(_) | QueueMediaPayload::Lora(_) => SealMedia::bytes(
                role,
                position,
                std::mem::take(
                    serialized_record
                        .as_mut()
                        .expect("structured record serialization was precomputed")
                        .as_mut(),
                ),
            ),
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
            DecryptedQueueMediaPayload::PrivatePath(path)
                if role == QueueMediaRole::References
                    && matches!(position, QueueMediaPosition::Item(_)) =>
            {
                QueueMediaPayload::StagedPath(path)
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
                QueueMediaRole::Lora | QueueMediaRole::Loras => {
                    let result = serde_json::from_slice(&bytes);
                    bytes.zeroize();
                    QueueMediaPayload::Lora(result.map_err(QueueMediaError::Deserialize)?)
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
        scrub_opaque_records(&mut self.0);
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
    use QueueMediaPayload::{Bytes, Keyframe, Lora, Presence, StagedPath, Text};

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
            StagedPath(_)
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
/// the records belonging to the same queue job. The second value is the
/// ordered staged reference path per `request.references` descriptor.
pub fn rehydrate_request_media(
    expected_job_id: &str,
    request_json: &str,
    media: OpaqueQueueMedia,
) -> Result<(mold_core::GenerateRequest, Vec<PathBuf>), QueueMediaError> {
    ensure_json_is_authority_free(request_json)?;
    let mut request: mold_core::GenerateRequest =
        serde_json::from_str(request_json).map_err(QueueMediaError::Deserialize)?;
    let paths = rehydrate_request_media_into(expected_job_id, &mut request, media)?;
    Ok((request, paths))
}

/// Overlay deferred media onto the scheduler-mutated request. Only extracted
/// media fields are touched; prompt/seed and every frozen plan/private
/// authority field retain their current values. Reference descriptors are
/// settings the request already carries, so their staged files are not
/// overlaid — they are returned in descriptor order for the consumer to bind
/// under its own lease.
pub fn rehydrate_request_media_into(
    expected_job_id: &str,
    request: &mut mold_core::GenerateRequest,
    mut media: OpaqueQueueMedia,
) -> Result<Vec<PathBuf>, QueueMediaError> {
    if media.job_id != expected_job_id {
        return Err(QueueMediaError::JobScopeMismatch);
    }
    ensure_references_are_descriptors(request.references.as_deref())?;
    for (present, field) in [
        (request.source_image.is_some(), "source_image"),
        (request.source_image_name.is_some(), "source_image_name"),
        (request.id_image.is_some(), "id_image"),
        (request.id_image_name.is_some(), "id_image_name"),
        (request.id_images.is_some(), "id_images"),
        (request.id_image_names.is_some(), "id_image_names"),
        (request.edit_images.is_some(), "edit_images"),
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
    let staged = media
        .records
        .iter()
        .filter(|record| {
            record.role == QueueMediaRole::References
                && matches!(record.position, QueueMediaPosition::Item(_))
        })
        .count();
    ensure_reference_count(request.references.as_deref(), staged)?;

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
    let mut reference_paths = BTreeMap::new();
    for record in grouped
        .remove(&QueueMediaRole::References)
        .unwrap_or_default()
    {
        if let (QueueMediaPosition::Item(index), QueueMediaPayload::StagedPath(path)) =
            (record.position, record.payload)
        {
            reference_paths.insert(index, path);
        }
    }
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
    Ok(reference_paths.into_values().collect())
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
                    "media": { "authority": "descriptor" },
                    "provenance": { "name": "resolved.png", "sha256": "44".repeat(32) },
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
        let staging = tempfile::tempdir().unwrap();
        let resolved = staging.path().join("reference-1.media");
        std::fs::write(&resolved, b"resolved-reference-bytes").unwrap();
        let staged = crate::reference_uploads::StagedReferences::from_files_for_test(
            &request,
            vec![resolved],
        );

        let extracted =
            extract_request_fields("job-a".to_string(), request, Some(&staged)).unwrap();

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
            "/private/audio.wav",
            "/private/exr-output",
            "/private/singular-lora.safetensors",
        ] {
            assert!(
                !json.contains(secret),
                "secret leaked into durable JSON: {secret}"
            );
        }

        let (restored, paths) = extracted.rehydrate("job-a").unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), expected);
        assert_eq!(paths, staged.paths());
    }

    fn ref2va_descriptor_request() -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "ordered references",
            "model": mold_core::minimax_h3::REF2VA_COMFY,
            "width": 64,
            "height": 64,
            "steps": 4,
            "seed": 7,
            "references": [
                {
                    "kind": "image",
                    "media": { "authority": "descriptor" },
                    "provenance": { "name": "subject.png", "sha256": "11".repeat(32) },
                    "mime_type": "image/png",
                    "width": 320,
                    "height": 240
                },
                {
                    "kind": "audio",
                    "media": { "authority": "descriptor" },
                    "provenance": { "name": "voice.wav", "sha256": "22".repeat(32) },
                    "mime_type": "audio/wav",
                    "duration_ms": 1000,
                    "sample_rate": 48000,
                    "channels": 2,
                    "sample_count": 48000
                }
            ]
        }))
        .unwrap()
    }

    fn h3_durable_authorities() -> ProcessPrivateAuthorities {
        ProcessPrivateAuthorities::none()
            .with_durable_replacement(Some(ProcessPrivateAuthority::H3PrivateIngressGrant))
    }

    /// A reference is a DESCRIPTOR plus MEDIA. The descriptor is a request
    /// setting and stays in the durable JSON; the media is a staged file that
    /// seals into the private-staging sink and hydrates back to an ordered
    /// private path per descriptor.
    #[cfg(unix)]
    #[test]
    fn references_seal_as_staged_files_and_hydrate_to_private_paths() {
        use crate::queue_media_store::{
            DecryptedQueueMediaPayload, QueueMediaOperationFingerprint, QueueMediaSink,
            QueueMediaStore, SealMediaSource,
        };

        let temp = tempfile::tempdir().unwrap();
        let staging = temp.path().join("resolved-test");
        std::fs::create_dir_all(&staging).unwrap();
        let first = staging.join("reference-1.media");
        let second = staging.join("reference-2.media");
        std::fs::write(&first, b"first-reference-bytes").unwrap();
        std::fs::write(&second, b"second-reference-bytes").unwrap();
        let request = ref2va_descriptor_request();
        let expected = serde_json::to_value(&request).unwrap();
        let staged = crate::reference_uploads::StagedReferences::from_files_for_test(
            &request,
            vec![first, second],
        );

        let extracted = extract_request_media(
            "job-references",
            request,
            &h3_durable_authorities(),
            Some(&staged),
        )
        .unwrap();
        let durable: serde_json::Value = serde_json::from_str(extracted.request_json()).unwrap();
        let descriptors = durable["references"].as_array().unwrap();
        assert_eq!(descriptors.len(), 2);
        for descriptor in descriptors {
            assert_eq!(descriptor["media"]["authority"], "descriptor");
            assert_eq!(
                descriptor["provenance"]["sha256"].as_str().unwrap().len(),
                64
            );
        }
        assert!(
            !extracted.request_json().contains("resolved-test"),
            "staging paths never enter durable JSON"
        );
        assert_eq!(
            extracted
                .media()
                .records()
                .iter()
                .map(|record| (record.role(), record.position()))
                .collect::<Vec<_>>(),
            [
                (QueueMediaRole::References, QueueMediaPosition::Collection),
                (QueueMediaRole::References, QueueMediaPosition::Item(0)),
                (QueueMediaRole::References, QueueMediaPosition::Item(1)),
            ]
        );
        let projection = project_request_media(extracted.media()).unwrap();

        let (request_json, media) = extracted.into_parts();
        let sealed = into_seal_media(media).unwrap();
        let mut staged_items = sealed
            .iter()
            .filter(|item| item.name.starts_with("item:"))
            .collect::<Vec<_>>();
        staged_items.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(staged_items.len(), 2);
        for item in &staged_items {
            assert_eq!(item.role, "references");
            assert_eq!(item.sink, QueueMediaSink::PrivateStaging);
            assert!(matches!(item.source, SealMediaSource::OpenFile(_)));
        }

        std::fs::create_dir_all(temp.path().join("home")).unwrap();
        let store = QueueMediaStore::open(temp.path().join("home"))
            .unwrap()
            .store;
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner-references",
                "job-references",
                &QueueMediaOperationFingerprint::sha256_v1(b"references operation"),
                &projection,
                sealed,
            )
            .unwrap();
        drop(staged);

        let mut decrypted = store.decrypt_mixed(&reference).unwrap();
        assert!(decrypted.media.iter().all(|item| {
            item.role != "references"
                || item.name == "collection"
                || matches!(item.payload, DecryptedQueueMediaPayload::PrivatePath(_))
        }));
        let opaque = decrypted_media_into_opaque("job-references", &mut decrypted).unwrap();
        let mut restored: mold_core::GenerateRequest = serde_json::from_str(&request_json).unwrap();
        let paths = rehydrate_request_media_into("job-references", &mut restored, opaque).unwrap();
        assert_eq!(serde_json::to_value(&restored).unwrap(), expected);
        assert_eq!(paths.len(), 2);
        assert_eq!(
            std::fs::read(&paths[0]).unwrap(),
            b"first-reference-bytes".to_vec()
        );
        assert_eq!(
            std::fs::read(&paths[1]).unwrap(),
            b"second-reference-bytes".to_vec()
        );
    }

    /// Only descriptors may sit in the durable JSON: an inline payload, a
    /// one-use upload handle, a server path, or a descriptor missing its digest
    /// is refused by index rather than written down.
    #[test]
    fn durable_json_references_must_be_digest_bearing_descriptors() {
        let mut inline = ref2va_descriptor_request();
        inline.references.as_mut().unwrap()[1] = mold_core::GenerationReference::Audio {
            media: mold_core::GenerationReferenceAuthority::Inline {
                data: vec![1, 2, 3],
            },
            provenance: mold_core::GenerationReferenceProvenance {
                name: None,
                sha256: Some("22".repeat(32)),
                crop: None,
            },
            mime_type: "audio/wav".to_string(),
            duration_ms: 1000,
            sample_rate: 48_000,
            channels: 2,
            sample_count: Some(48_000),
        };
        assert!(matches!(
            ensure_references_are_descriptors(inline.references.as_deref()),
            Err(QueueMediaError::RequestJsonReferenceAuthority(1))
        ));

        let mut upload = ref2va_descriptor_request();
        if let Some(mold_core::GenerationReference::Image { media, .. }) =
            upload.references.as_mut().unwrap().first_mut()
        {
            *media = mold_core::GenerationReferenceAuthority::Upload {
                handle: "one-use-secret".to_string(),
            };
        }
        assert!(matches!(
            ensure_references_are_descriptors(upload.references.as_deref()),
            Err(QueueMediaError::RequestJsonReferenceAuthority(0))
        ));

        let mut server_path = ref2va_descriptor_request();
        if let Some(mold_core::GenerationReference::Image { media, .. }) =
            server_path.references.as_mut().unwrap().first_mut()
        {
            *media = mold_core::GenerationReferenceAuthority::ServerPath {
                path: "/private/subject.png".to_string(),
            };
        }
        assert!(matches!(
            ensure_references_are_descriptors(server_path.references.as_deref()),
            Err(QueueMediaError::RequestJsonReferenceAuthority(0))
        ));

        let mut undigested = ref2va_descriptor_request();
        if let Some(mold_core::GenerationReference::Image { provenance, .. }) =
            undigested.references.as_mut().unwrap().first_mut()
        {
            provenance.sha256 = None;
        }
        assert!(matches!(
            ensure_references_are_descriptors(undigested.references.as_deref()),
            Err(QueueMediaError::RequestJsonReferenceAuthority(0))
        ));

        assert!(ensure_references_are_descriptors(None).is_ok());
        assert!(ensure_references_are_descriptors(
            ref2va_descriptor_request().references.as_deref()
        )
        .is_ok());

        // Extraction is where the durable JSON is minted, so it asks the same
        // question before a byte is serialized.
        assert!(matches!(
            extract_request_media("job-inline", inline, &h3_durable_authorities(), None),
            Err(QueueMediaError::RequestJsonReferenceAuthority(1))
        ));
    }

    /// Every descriptor owns exactly one staged file. A staged set that does
    /// not line up with the descriptors is a mismatch at extraction and again
    /// at hydration, never a silent truncation.
    #[cfg(unix)]
    #[test]
    fn reference_descriptor_and_staged_counts_must_agree() {
        let temp = tempfile::tempdir().unwrap();
        let only = temp.path().join("reference-1.media");
        std::fs::write(&only, b"only").unwrap();
        let request = ref2va_descriptor_request();
        let short =
            crate::reference_uploads::StagedReferences::from_files_for_test(&request, vec![only]);
        assert!(matches!(
            extract_request_media(
                "job-short",
                request.clone(),
                &h3_durable_authorities(),
                Some(&short)
            ),
            Err(QueueMediaError::ReferenceCountMismatch {
                descriptors: 2,
                staged: 1
            })
        ));
        assert!(matches!(
            extract_request_media("job-none", request.clone(), &h3_durable_authorities(), None),
            Err(QueueMediaError::ReferenceCountMismatch {
                descriptors: 2,
                staged: 0
            })
        ));

        let mut restored = request;
        let media = OpaqueQueueMedia {
            job_id: "job-hydrate".into(),
            records: vec![
                OpaqueQueueMediaRecord {
                    role: QueueMediaRole::References,
                    position: QueueMediaPosition::Collection,
                    payload: QueueMediaPayload::Presence,
                },
                OpaqueQueueMediaRecord {
                    role: QueueMediaRole::References,
                    position: QueueMediaPosition::Item(0),
                    payload: QueueMediaPayload::StagedPath(temp.path().join("00000000.media")),
                },
            ],
            scrub_probe: None,
        };
        assert!(matches!(
            rehydrate_request_media_into("job-hydrate", &mut restored, media),
            Err(QueueMediaError::ReferenceCountMismatch {
                descriptors: 2,
                staged: 1
            })
        ));
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
        let extracted = extract_request_fields("job-projection".into(), request, None).unwrap();
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
        let extracted = extract_request_fields("job-qwen-five".into(), request, None).unwrap();
        let projection = project_request_media(extracted.media()).unwrap();

        assert_eq!(projection.edit_image_count, 5);
        assert_eq!(
            projection.edit_images.len(),
            crate::queue_media_store::PROJECTED_EDIT_DIMENSION_SLOTS
        );
        assert_eq!(
            serde_json::to_value(extracted.rehydrate("job-qwen-five").unwrap().0).unwrap(),
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
        let extracted = extract_request_fields("job-overlay".into(), request, None).unwrap();
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

        let extracted = extract_request_fields("job-empty".to_string(), request, None).unwrap();
        let (restored, paths) = extracted.rehydrate("job-empty").unwrap();
        assert!(paths.is_empty());

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
            extract_request_media("job-one", request, &ProcessPrivateAuthorities::none(), None)
                .unwrap();

        assert!(matches!(
            extracted.rehydrate("job-two"),
            Err(QueueMediaError::JobScopeMismatch)
        ));
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
            extract_request_media("job-h3", h3, &ProcessPrivateAuthorities::none(), None),
            Err(QueueMediaError::UnsupportedProcessPrivateAuthority(
                ProcessPrivateAuthority::H3PrivateIngressGrant
            ))
        ));

        // A reference that still carries public authority never reaches the
        // durable JSON: the resolver consumed every handle before extraction,
        // so one surviving here is refused by index rather than written down.
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
                &ProcessPrivateAuthorities::none(),
                None
            ),
            Err(QueueMediaError::RequestJsonReferenceAuthority(0))
        ));
    }

    #[test]
    fn pre_dispatch_hdr_authority_remains_non_durable() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "pre-dispatch authority",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "hdr_exr_dir": "/private/hdr"
        }))
        .unwrap();
        assert!(matches!(
            extract_request_media("job-hdr", request, &ProcessPrivateAuthorities::none(), None),
            Err(QueueMediaError::UnsupportedPreDispatchAuthority(
                QueueMediaRole::HdrExrDir
            ))
        ));
    }

    /// A LoRA beside conditioning media is the ordinary img2img-with-adapter
    /// print. It used to work only because it fell back to the non-durable
    /// path; the durable path now seals the adapter record as JSON beside the
    /// media and hands it back whole, scale and expert included.
    #[test]
    fn lora_beside_media_seals_as_json_and_rehydrates_losslessly() {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "adapter beside media",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": "c291cmNlLWJ5dGVz",
            "lora": { "path": "/private/one.safetensors", "scale": 0.5 },
            "loras": [
                { "path": "/private/two.safetensors", "scale": 0.7, "expert": "high" },
                { "path": "/private/three.safetensors", "scale": 1.25 }
            ]
        }))
        .unwrap();
        let expected = request.clone();
        let extracted = extract_request_media(
            "job-lora",
            request,
            &ProcessPrivateAuthorities::none(),
            None,
        )
        .unwrap();

        let json: serde_json::Value = serde_json::from_str(extracted.request_json()).unwrap();
        assert!(
            json.get("lora").is_none(),
            "the adapter path never enters the durable JSON"
        );
        assert!(json.get("loras").is_none());
        assert!(!extracted.request_json().contains("safetensors"));

        let (request_json, media) = extracted.into_parts();
        let sealed = into_seal_media(media).unwrap();
        let lora_records = sealed
            .iter()
            .filter(|item| item.role == "lora" || item.role == "loras")
            .count();
        assert_eq!(
            lora_records, 4,
            "one scalar record, one presence record, two items"
        );

        let restored = {
            let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
                "prompt": "adapter beside media",
                "model": "mock",
                "width": 64,
                "height": 64,
                "steps": 1,
                "source_image": "c291cmNlLWJ5dGVz",
                "lora": { "path": "/private/one.safetensors", "scale": 0.5 },
                "loras": [
                    { "path": "/private/two.safetensors", "scale": 0.7, "expert": "high" },
                    { "path": "/private/three.safetensors", "scale": 1.25 }
                ]
            }))
            .unwrap();
            extract_request_media(
                "job-lora",
                request,
                &ProcessPrivateAuthorities::none(),
                None,
            )
            .unwrap()
            .rehydrate("job-lora")
            .unwrap()
            .0
        };
        assert_eq!(
            serde_json::to_value(&restored).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        let _ = request_json;
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
            extract_request_media("  ", request, &ProcessPrivateAuthorities::none(), None),
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
        let extracted = extract_request_media(
            "job-store",
            request,
            &ProcessPrivateAuthorities::none(),
            None,
        )
        .unwrap();
        let (request_json, media) = extracted.into_parts();
        assert_eq!(media.job_id(), "job-store");

        let (restored, _paths) =
            rehydrate_request_media("job-store", &request_json, media).unwrap();
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
        let extracted = extract_request_media(
            "job-debug",
            request,
            &ProcessPrivateAuthorities::none(),
            None,
        )
        .unwrap();

        let debug = format!("{:?}", extracted.media());
        assert!(!debug.contains("secret-bytes"));
        assert!(!debug.contains("secret-name.png"));
        assert!(debug.contains("records"));
    }
}
