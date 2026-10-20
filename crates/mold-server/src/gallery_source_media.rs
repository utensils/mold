//! Authenticated, path-free access to gallery-retained authored media.

use axum::{
    extract::{Extension, Path as AxumPath, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use mold_core::{
    RetainedSourceMediaAvailability as Availability, RetainedSourceMediaInventory as Inventory,
    RetainedSourceMediaMember as Member,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use crate::{routes::ApiError, state::AppState};

const MAX_SOURCE_DOWNLOAD_BYTES: u64 = 512 * 1024 * 1024;
const REUSE_SESSION_TTL_SECS: u64 = 2 * 60;
const MAX_REUSE_SESSION_MEMBERS: usize = 64;
pub(crate) const REUSE_SESSION_HEADER: &str = "x-mold-retained-media-session";
/// Bound subject for a reuse session minted on a host with no API keys.
const ANONYMOUS_REUSE_SUBJECT: &str = "mold.retained-media.anonymous-subject.v1";

pub(crate) fn validate_reuse_batch_cardinality(
    headers: &HeaderMap,
    request_count: usize,
) -> Result<bool, ApiError> {
    if headers.get(REUSE_SESSION_HEADER).is_none() {
        return Ok(false);
    }
    if request_count != 1 {
        return Err(ApiError::with_code(
            "a retained-media reuse session binds exactly one batch child",
            "RETAINED_MEDIA_REUSE_BATCH_AMBIGUOUS",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    Ok(true)
}

/// Whether this CALLER may read the host's retained private media.
///
/// This asks about the caller's authorization, never about the operator's
/// configuration. `AuthState` is `None` whenever `MOLD_API_KEY` is unset, and
/// the earlier `auth.is_some_and(|Extension(s)| s.is_some())` read that as a
/// refusal — so on a default, keyless server every retained-media route
/// answered `unavailable_auth` for prints it never looked at, and the whole
/// feature was dead. A keyless host is open BY POLICY: `require_api_key` passes
/// every request through it, and `DELETE /api/gallery/image/:filename` plus
/// device lifecycle are already open there (`auth.rs`'s
/// `device_patch_is_open_when_server_auth_is_disabled`). Retained source media
/// is not more sensitive than the print bytes and full metadata that same host
/// already serves unauthenticated, so it must not be the one route that refuses.
pub(crate) fn private_media_authorized(
    auth: Option<&Extension<crate::auth::AuthState>>,
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
) -> bool {
    match auth {
        // Enforced: `require_api_key` already rejected unauthenticated callers,
        // so the marker's absence here means a route reached without one.
        Some(Extension(Some(_))) => authenticated.is_some(),
        // Disabled: open by policy, exactly like every other privileged route.
        _ => true,
    }
}

/// Subject a reuse session is bound to.
///
/// An open host has no credential, so it binds one stable anonymous subject —
/// create and hydrate MUST agree or a handle minted there could never be
/// redeemed. The handle's real safety properties are unchanged either way: one
/// use, a 120 s TTL, and binding to `instance_id` + `request_sha256` +
/// `archive_identity_sha256`, which is what stops a DIFFERENT request consuming
/// it. The constant is domain-separated so it can never collide with a real
/// `ApiKeySet::durable_identity` digest.
fn reuse_credential_identity(
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
) -> String {
    match authenticated {
        Some(Extension(marker)) => marker.durable_identity.clone(),
        None => ANONYMOUS_REUSE_SUBJECT.to_string(),
    }
}

/// How a resolved lookup answers the inventory question.
///
/// `members.is_empty()` alone is NOT damage: `downloadable_role` deliberately
/// filters provenance-text roles, so an H3 print retaining only
/// `source_image_name` resolves cleanly with nothing to hand back. Reporting
/// that as `missing_or_corrupt` told the user their media was damaged when it
/// was never downloadable by design.
fn availability_of(resolved: &ResolvedGalleryMedia) -> Availability {
    if resolved.corrupt {
        Availability::UnavailableMissingOrCorrupt
    } else if resolved.legacy || resolved.members.is_empty() {
        Availability::UnavailableLegacy
    } else {
        Availability::Available
    }
}

fn downloadable_role(role: &str) -> bool {
    !matches!(
        role,
        "source_image_name"
            | "identity_image_name"
            | "identity_image_names"
            | "hdr_exr_dir"
            | "lora"
            | "loras"
    )
}

fn reusable_role(role: &str) -> bool {
    if role.starts_with("chain_continuation_tail:") {
        return false;
    }
    if role.starts_with("stage_source:") {
        return role == "stage_source:0";
    }
    !matches!(
        role,
        "matting_processed_source_image" | "matting_processed_references"
    )
}

fn member_id(
    media_set: &crate::queue_media_store::MediaSetRef,
    pin_id: &str,
    index: usize,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"mold.gallery-source-member.v1\0");
    for component in [
        media_set.owner_id.as_bytes(),
        media_set.job_id.as_bytes(),
        media_set.set_id.as_bytes(),
    ] {
        digest.update((component.len() as u64).to_be_bytes());
        digest.update(component);
    }
    digest.update(pin_id.as_bytes());
    digest.update((index as u64).to_le_bytes());
    format!("{:x}", digest.finalize())
}

fn safe_display_name(name: &str, role: &str, index: usize) -> String {
    let basename = std::path::Path::new(name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let cleaned = basename
        .chars()
        .filter(|character| !character.is_control())
        .take(160)
        .collect::<String>();
    if cleaned.trim().is_empty() || cleaned == "scalar" || cleaned.starts_with("item:") {
        format!("{role}-{}", index + 1)
    } else {
        cleaned
    }
}

#[derive(Clone)]
pub(crate) struct ResolvedMember {
    pub(crate) media_set: crate::queue_media_store::MediaSetRef,
    pub(crate) pin_id: String,
    pub(crate) index: usize,
    pub(crate) position: String,
    pub(crate) sink: crate::queue_media_store::QueueMediaSink,
    pub(crate) sha256: String,
    pub(crate) member: Member,
}

pub(crate) struct ResolvedGalleryMedia {
    pub(crate) archive_identity_sha256: String,
    pub(crate) members: Vec<ResolvedMember>,
    pub(crate) legacy: bool,
    pub(crate) corrupt: bool,
}

pub(crate) fn resolve_members(
    state: &AppState,
    filename: &str,
) -> Result<Option<ResolvedGalleryMedia>, ApiError> {
    let config = state.config.blocking_read();
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);
    let lookup = state
        .gallery_publication_gate
        .validated_retained_media_for_item(&output_dir, filename)
        .map_err(|error| ApiError::internal(format!("gallery authority read failed: {error:#}")))?;
    let revision = retained_lookup_revision(&lookup)?;
    let resolved = resolve_lookup(state, lookup)?;
    let current = state
        .gallery_publication_gate
        .validated_retained_media_for_item(&output_dir, filename)
        .map_err(|error| ApiError::internal(format!("gallery authority read failed: {error:#}")))?;
    if retained_lookup_revision(&current)? != revision {
        return Err(ApiError::not_found("retained source owner changed"));
    }
    Ok(resolved)
}

fn retained_lookup_revision(
    lookup: &crate::batch_transaction::ValidatedRetainedMedia,
) -> Result<Option<String>, ApiError> {
    match lookup {
        crate::batch_transaction::ValidatedRetainedMedia::Present { identity, pins, .. } => {
            let bytes = serde_json::to_vec(&(identity, pins))
                .map_err(|_| ApiError::internal("retained owner identity unavailable"))?;
            Ok(Some(format!("{:x}", Sha256::digest(bytes))))
        }
        crate::batch_transaction::ValidatedRetainedMedia::Missing => Ok(None),
        crate::batch_transaction::ValidatedRetainedMedia::Invalid => Ok(Some("invalid".into())),
    }
}

fn resolve_lookup(
    state: &AppState,
    lookup: crate::batch_transaction::ValidatedRetainedMedia,
) -> Result<Option<ResolvedGalleryMedia>, ApiError> {
    let (identity, pins, metadata) = match lookup {
        crate::batch_transaction::ValidatedRetainedMedia::Missing => return Ok(None),
        crate::batch_transaction::ValidatedRetainedMedia::Invalid => {
            return Ok(Some(ResolvedGalleryMedia {
                archive_identity_sha256: String::new(),
                members: Vec::new(),
                legacy: false,
                corrupt: true,
            }))
        }
        crate::batch_transaction::ValidatedRetainedMedia::Present {
            identity,
            pins,
            metadata,
        } => (identity, pins, metadata),
    };
    let mut resolved = resolve_members_from_pins(state, identity, pins)?;
    if let Some(resolved) = resolved.as_mut() {
        for member in &mut resolved.members {
            present_member(member, &metadata);
        }
    }
    Ok(resolved)
}

pub(crate) fn resolve_members_from_pins(
    state: &AppState,
    identity: crate::batch_transaction::ArchivedChildIdentity,
    pins: Vec<crate::batch_transaction::GalleryMediaPin>,
) -> Result<Option<ResolvedGalleryMedia>, ApiError> {
    if pins.is_empty() {
        return Ok(Some(ResolvedGalleryMedia {
            archive_identity_sha256: archive_identity_sha256(&identity)?,
            members: Vec::new(),
            legacy: true,
            corrupt: false,
        }));
    }
    let Some(lifecycle) = state.queue_journal.queue_media_lifecycle() else {
        return Ok(Some(ResolvedGalleryMedia {
            archive_identity_sha256: archive_identity_sha256(&identity)?,
            members: Vec::new(),
            legacy: false,
            corrupt: true,
        }));
    };
    let mut members = Vec::new();
    let mut corrupt = false;
    for pin in pins {
        let manifest = match lifecycle.gallery_manifest(pin.media_set.clone(), pin.pin_id.clone()) {
            Ok(manifest) => manifest,
            Err(error) => {
                tracing::warn!(%error, "gallery retained-media manifest is unavailable");
                corrupt = true;
                continue;
            }
        };
        for (index, entry) in manifest.entries.into_iter().enumerate() {
            if entry.size_bytes == 0 || !downloadable_role(&entry.role) {
                continue;
            }
            members.push(ResolvedMember {
                media_set: pin.media_set.clone(),
                pin_id: pin.pin_id.clone(),
                index,
                position: entry.name.clone(),
                sink: entry.sink,
                sha256: entry.sha256_hex.clone(),
                member: Member {
                    member_id: member_id(&pin.media_set, &pin.pin_id, index),
                    content_sha256: entry.content_sha256.clone(),
                    role: entry.role.clone(),
                    display_name: safe_display_name(&entry.name, &entry.role, index),
                    size_bytes: entry.size_bytes,
                    label: None,
                    media_kind: None,
                    position: None,
                    frame_index: None,
                    image_role: None,
                },
            });
        }
    }
    Ok(Some(ResolvedGalleryMedia {
        archive_identity_sha256: archive_identity_sha256(&identity)?,
        members,
        legacy: false,
        corrupt,
    }))
}

pub(crate) fn archive_identity_sha256(
    identity: &crate::batch_transaction::ArchivedChildIdentity,
) -> Result<String, ApiError> {
    let bytes = serde_json::to_vec(identity).map_err(|error| {
        ApiError::internal(format!("gallery identity encoding failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Display metadata is derived only from the exact owner's archived recipe.
fn present_member(member: &mut ResolvedMember, metadata: &mold_core::OutputMetadata) {
    let position = member
        .position
        .strip_prefix("item:")
        .and_then(|v| v.parse::<u32>().ok());
    let role = member.member.role.as_str();
    let mut kind = "image";
    let mut image_role = None;
    let mut frame_index = None;
    let mut label = match role {
        "source_image" if metadata.frames.is_some_and(|frames| frames > 1) => {
            image_role = Some("start_frame".into());
            "Start frame".into()
        }
        "source_image" => "Source image".to_string(),
        "identity_image" | "identity_images" => "Identity reference".into(),
        "edit_images" => "Reference image".into(),
        "mask_image" => "Mask".into(),
        "control_image" => "Control image".into(),
        "matting_processed_source_image" => "Processed source image".into(),
        "matting_processed_references" => "Processed reference".into(),
        "source_video" | "source_video_path" => {
            kind = "video";
            "Source video".into()
        }
        "extend_video" | "extend_video_path" => {
            kind = "video";
            "Continuation video".into()
        }
        "audio_file" | "audio_file_path" => {
            kind = "audio";
            "Source audio".into()
        }
        "references" => {
            kind = "unknown";
            if let Some(reference) =
                position.and_then(|i| metadata.references.as_ref()?.iter().find(|r| r.index == i))
            {
                kind = match reference.kind {
                    mold_core::GenerationReferenceKind::Image => "image",
                    mold_core::GenerationReferenceKind::Video => "video",
                    mold_core::GenerationReferenceKind::Audio => "audio",
                    mold_core::GenerationReferenceKind::Mesh => "mesh",
                };
                image_role = reference
                    .image_role
                    .and_then(|r| serde_json::to_value(r).ok()?.as_str().map(str::to_string));
            }
            format!(
                "{} reference",
                match kind {
                    "image" => "Image",
                    "video" => "Video",
                    "audio" => "Audio",
                    "mesh" => "Mesh",
                    _ => "Media",
                }
            )
        }
        "keyframes" => {
            frame_index = position.and_then(|i| {
                metadata
                    .keyframes
                    .as_ref()?
                    .get(i as usize)
                    .map(|k| k.frame)
            });
            match frame_index {
                Some(0) => {
                    image_role = Some("start_frame".into());
                    "Start frame".into()
                }
                Some(frame)
                    if metadata
                        .frames
                        .is_some_and(|count| count.checked_sub(1) == Some(frame)) =>
                {
                    image_role = Some("end_frame".into());
                    "End frame".into()
                }
                Some(frame) => format!("Keyframe {frame}"),
                None => position
                    .map(|index| format!("Keyframe {}", u64::from(index) + 1))
                    .unwrap_or_else(|| "Keyframe".into()),
            }
        }
        role if role.starts_with("stage_keyframe:") || role.starts_with("stage_reference:") => {
            let parts = role.split(':').collect::<Vec<_>>();
            let stage_index = parts.get(1).and_then(|value| value.parse::<usize>().ok());
            let item_index = parts.get(2).and_then(|value| value.parse::<usize>().ok());
            let stage = stage_index.and_then(|index| metadata.chain.as_ref()?.stages.get(index));
            if role.starts_with("stage_reference:") {
                kind = match stage
                    .and_then(|stage| stage.generation_references.get(item_index?))
                    .map(|reference| reference.kind)
                {
                    Some(mold_core::GenerationReferenceKind::Image) => "image",
                    Some(mold_core::GenerationReferenceKind::Video) => "video",
                    Some(mold_core::GenerationReferenceKind::Audio) => "audio",
                    _ => "unknown",
                };
            } else {
                frame_index = stage
                    .and_then(|stage| stage.keyframes.get(item_index?))
                    .map(|frame| frame.frame);
            }
            format!(
                "Stage {} {} {}",
                stage_index.map_or(0, |index| index + 1),
                if role.starts_with("stage_keyframe:") {
                    "boundary"
                } else {
                    "reference"
                },
                item_index.map_or(0, |index| index + 1)
            )
        }
        role if role.starts_with("chain_continuation_tail:") => {
            image_role = Some("continuation_frame".into());
            format!(
                "Incoming continuation frame {}",
                role.strip_prefix("chain_continuation_tail:")
                    .and_then(|value| value.parse::<u32>().ok())
                    .map_or(0, |value| u64::from(value) + 1)
            )
        }
        role if role.starts_with("stage_source:") => match role
            .strip_prefix("stage_source:")
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(stage) => format!("Stage {} source", u64::from(stage) + 1),
            None => {
                kind = "unknown";
                "Stage source".into()
            }
        },
        _ => {
            kind = "unknown";
            "Source media".into()
        }
    };
    if role != "keyframes" && !role.starts_with("chain_continuation_tail:") {
        if let Some(index) = position {
            label.push_str(&format!(" {}", u64::from(index) + 1));
        }
    }
    member.member.label = Some(label);
    member.member.media_kind = Some(kind.into());
    member.member.position = position;
    member.member.frame_index = frame_index;
    member.member.image_role = image_role;
}

/// Keep the raw endpoint's structured record intact; only presentation and
/// independent attachments unwrap the keyframe's original image.
pub(crate) fn media_content(role: &str, bytes: Vec<u8>) -> Result<Vec<u8>, ApiError> {
    if role != "keyframes" {
        return Ok(bytes);
    }
    let bytes = zeroize::Zeroizing::new(bytes);
    let keyframe: mold_core::KeyframeCondition = serde_json::from_slice(&bytes).map_err(|_| {
        ApiError::with_code(
            "retained keyframe is malformed",
            "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
            StatusCode::CONFLICT,
        )
    })?;
    if keyframe.image.is_empty() {
        return Err(ApiError::not_found("retained keyframe image unavailable"));
    }
    Ok(keyframe.image)
}

#[derive(Deserialize, utoipa::ToSchema)]
pub(crate) struct BatchInventoryRequest {
    filenames: Vec<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct BatchInventoryResponse {
    instance_id: String,
    inventories: std::collections::BTreeMap<String, Inventory>,
}

fn validate_inventory_filenames(filenames: &[String]) -> Result<(), ApiError> {
    if filenames.len() > 100 {
        return Err(ApiError::with_code(
            "source inventory accepts at most 100 filenames",
            "RETAINED_SOURCE_MEDIA_BATCH_TOO_LARGE",
            StatusCode::UNPROCESSABLE_ENTITY,
        ));
    }
    for filename in filenames {
        crate::routes::validate_gallery_filename(filename)?;
    }
    Ok(())
}

fn private_inventory_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    headers
}

fn inventory_response(resolved: Option<ResolvedGalleryMedia>) -> Inventory {
    match resolved {
        None => Inventory {
            availability: Availability::UnavailableLegacy,
            members: Vec::new(),
        },
        Some(resolved) => {
            let availability = availability_of(&resolved);
            Inventory {
                availability,
                members: if availability == Availability::Available {
                    resolved.members.into_iter().map(|m| m.member).collect()
                } else {
                    Vec::new()
                },
            }
        }
    }
}

#[utoipa::path(post, path = "/api/gallery/source-media/inventory", tag = "gallery",
    request_body = BatchInventoryRequest,
    responses((status = 200, body = BatchInventoryResponse), (status = 401, description = "Authentication required")))]
pub(crate) async fn batch_inventory(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Json(request): Json<BatchInventoryRequest>,
) -> Result<(HeaderMap, Json<BatchInventoryResponse>), ApiError> {
    if !private_media_authorized(auth.as_ref(), authenticated.as_ref()) {
        return Err(ApiError::with_code(
            "API-key authentication required",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    validate_inventory_filenames(&request.filenames)?;
    let instance_id = state.instance_id.to_string();
    let inventories = tokio::task::spawn_blocking(move || {
        let config = state.config.blocking_read();
        if state.is_output_disabled(&config) {
            return Err(ApiError::not_found("image output is disabled"));
        }
        let root = config.effective_output_dir();
        drop(config);
        let lookups = state
            .gallery_publication_gate
            .validated_retained_media_for_items(&root, &request.filenames)
            .map_err(|e| ApiError::internal(format!("gallery authority read failed: {e:#}")))?;
        let revisions = lookups
            .iter()
            .map(retained_lookup_revision)
            .collect::<Result<Vec<_>, _>>()?;
        let mut inventories = request
            .filenames
            .iter()
            .cloned()
            .zip(lookups)
            .map(|(filename, lookup)| {
                Ok((
                    filename,
                    inventory_response(resolve_lookup(&state, lookup)?),
                ))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>, ApiError>>()?;
        // A cold scan can outlive a deletion/transfer. Recheck one fresh authority
        // snapshot before publishing, without holding a publication lock over I/O.
        let current = state
            .gallery_publication_gate
            .validated_retained_media_for_items(&root, &request.filenames)
            .map_err(|e| ApiError::internal(format!("gallery authority read failed: {e:#}")))?;
        for ((filename, previous), current) in request.filenames.iter().zip(revisions).zip(current)
        {
            if retained_lookup_revision(&current)? != previous {
                inventories.insert(
                    filename.clone(),
                    Inventory {
                        availability: Availability::UnavailableMissingOrCorrupt,
                        members: Vec::new(),
                    },
                );
            }
        }
        Ok::<_, ApiError>(inventories)
    })
    .await
    .map_err(|_| ApiError::internal("source inventory task failed"))??;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    Ok((
        headers,
        Json(BatchInventoryResponse {
            instance_id,
            inventories,
        }),
    ))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub(crate) struct CreateReuseSessionRequest {
    /// Exact payload-free request that will consume the retained inputs.
    pub(crate) target_request: mold_core::GenerateRequest,
    /// Opaque member ids from the inventory response.
    pub(crate) member_ids: Vec<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct CreateReuseSessionResponse {
    pub(crate) instance_id: String,
    pub(crate) expires_at: u64,
    pub(crate) request_sha256: String,
    pub(crate) session_handle: String,
}

#[derive(Clone)]
struct ReuseSessionMember {
    member_id: String,
    media_set: crate::queue_media_store::MediaSetRef,
    pin_id: String,
    index: usize,
    role: String,
    position: String,
}

struct ReuseSession {
    instance_id: String,
    credential_identity: String,
    filename: String,
    archive_identity_sha256: String,
    request_sha256: String,
    expires_at: u64,
    members: Vec<ReuseSessionMember>,
}

fn reuse_sessions() -> &'static Mutex<HashMap<[u8; 32], ReuseSession>> {
    static SESSIONS: OnceLock<Mutex<HashMap<[u8; 32], ReuseSession>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn request_sha256(request: &mold_core::GenerateRequest) -> Result<String, ApiError> {
    let bytes = serde_json::to_vec(request)
        .map_err(|error| ApiError::internal(format!("reuse request encoding failed: {error}")))?;
    let mut digest = Sha256::new();
    digest.update(b"mold.gallery-source-reuse-request.v1\0");
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn session_token_hash(token: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mold.gallery-source-reuse-token.v1\0");
    digest.update(token.as_bytes());
    digest.finalize().into()
}

fn issue_session_token() -> Result<(String, [u8; 32]), ApiError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|error| ApiError::internal(format!("reuse session randomness failed: {error}")))?;
    let token = URL_SAFE_NO_PAD.encode(bytes);
    let hash = session_token_hash(&token);
    Ok((token, hash))
}

fn take_reuse_session(
    token_hash: &[u8; 32],
    now: u64,
    instance_id: &str,
    credential_identity: &str,
    request_digest: &str,
) -> Result<ReuseSession, ApiError> {
    let mut sessions = reuse_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sessions.retain(|_, held| held.expires_at > now);
    let held = sessions.get(token_hash).ok_or_else(|| {
        ApiError::with_code(
            "retained-media reuse session is missing, expired, or already consumed",
            "RETAINED_MEDIA_REUSE_INVALID",
            StatusCode::CONFLICT,
        )
    })?;
    if held.instance_id != instance_id
        || held.credential_identity != credential_identity
        || held.request_sha256 != request_digest
    {
        return Err(ApiError::with_code(
            "retained-media reuse session does not match this server, credential, or request",
            "RETAINED_MEDIA_REUSE_SCOPE_MISMATCH",
            StatusCode::CONFLICT,
        ));
    }
    Ok(sessions
        .remove(token_hash)
        .expect("session was just present"))
}

fn selected_members(
    resolved: &ResolvedGalleryMedia,
    requested: &[String],
) -> Result<Vec<ReuseSessionMember>, ApiError> {
    if requested.is_empty() || requested.len() > MAX_REUSE_SESSION_MEMBERS {
        return Err(ApiError::validation(format!(
            "member_ids must contain 1 to {MAX_REUSE_SESSION_MEMBERS} retained members"
        )));
    }
    let mut seen = HashSet::new();
    let mut selected = Vec::with_capacity(requested.len());
    let mut total_bytes = 0_u64;
    for requested_id in requested {
        if !seen.insert(requested_id) {
            return Err(ApiError::validation("member_ids contains a duplicate"));
        }
        let member = resolved
            .members
            .iter()
            .find(|member| member.member.member_id == *requested_id)
            .ok_or_else(|| ApiError::not_found("retained source-media member was not found"))?;
        if !reusable_role(&member.member.role) {
            return Err(ApiError::with_code(
                "this retained media is available for download but cannot be reused as one generation input",
                "RETAINED_MEDIA_REUSE_ROLE_UNSUPPORTED",
                StatusCode::UNPROCESSABLE_ENTITY,
            ));
        }
        total_bytes = total_bytes
            .checked_add(member.member.size_bytes)
            .ok_or_else(|| ApiError::validation("selected retained media is too large"))?;
        if member.member.size_bytes > MAX_SOURCE_DOWNLOAD_BYTES
            || total_bytes > MAX_SOURCE_DOWNLOAD_BYTES
        {
            return Err(ApiError::with_code(
                "selected retained media exceeds the v1 reuse-session limit",
                "RETAINED_SOURCE_MEDIA_TOO_LARGE",
                StatusCode::PAYLOAD_TOO_LARGE,
            ));
        }
        selected.push(ReuseSessionMember {
            member_id: member.member.member_id.clone(),
            media_set: member.media_set.clone(),
            pin_id: member.pin_id.clone(),
            index: member.index,
            role: member.member.role.clone(),
            position: member.position.clone(),
        });
    }
    selected.sort_by_key(|member| (member.pin_id.clone(), member.index));
    Ok(selected)
}

#[utoipa::path(
    post,
    path = "/api/gallery/source-media/{filename}/reuse-sessions",
    tag = "gallery",
    params(("filename" = String, Path, description = "Exact gallery filename")),
    request_body = CreateReuseSessionRequest,
    responses(
        (status = 200, body = CreateReuseSessionResponse),
        (status = 401, description = "API-key authentication is required"),
        (status = 409, description = "Retained media is unavailable")
    )
)]
pub(crate) async fn create_reuse_session(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    AxumPath(filename): AxumPath<String>,
    Json(mut body): Json<CreateReuseSessionRequest>,
) -> Result<Response, ApiError> {
    if !private_media_authorized(auth.as_ref(), authenticated.as_ref()) {
        return Err(ApiError::with_code(
            "retained-media reuse sessions require API-key authentication",
            "RETAINED_MEDIA_REUSE_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    crate::routes::validate_gallery_filename(&filename)?;
    mold_core::minimax_h3::canonicalize_request_model(&mut body.target_request);
    let request_sha256 = request_sha256(&body.target_request)?;
    let lookup_state = state.clone();
    let lookup_filename = filename.clone();
    let resolved =
        tokio::task::spawn_blocking(move || resolve_members(&lookup_state, &lookup_filename))
            .await
            .map_err(|error| ApiError::internal(format!("reuse session lookup failed: {error}")))??
            .filter(|resolved| !resolved.legacy && !resolved.corrupt)
            .ok_or_else(|| {
                ApiError::with_code(
                    "retained source media is unavailable",
                    "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
                    StatusCode::CONFLICT,
                )
            })?;
    let members = selected_members(&resolved, &body.member_ids)?;
    ensure_hydration_target_is_empty(&body.target_request, &members)?;
    let (session_handle, token_hash) = issue_session_token()?;
    let now = unix_timestamp();
    let expires_at = now.saturating_add(REUSE_SESSION_TTL_SECS);
    let credential_identity = reuse_credential_identity(authenticated.as_ref());
    let session = ReuseSession {
        instance_id: state.instance_id.as_ref().clone(),
        credential_identity,
        filename,
        archive_identity_sha256: resolved.archive_identity_sha256,
        request_sha256: request_sha256.clone(),
        expires_at,
        members,
    };
    let mut sessions = reuse_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sessions.retain(|_, held| held.expires_at > now);
    sessions.insert(token_hash, session);
    drop(sessions);
    let mut response = Json(CreateReuseSessionResponse {
        instance_id: state.instance_id.as_ref().clone(),
        expires_at,
        request_sha256,
        session_handle,
    })
    .into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    Ok(response)
}

pub(crate) async fn hydrate_reuse_session(
    state: &AppState,
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
    auth: Option<&Extension<crate::auth::AuthState>>,
    headers: &HeaderMap,
    request: &mut mold_core::GenerateRequest,
) -> Result<(), ApiError> {
    let Some(handle) = headers.get(REUSE_SESSION_HEADER) else {
        return Ok(());
    };
    if !private_media_authorized(auth, authenticated) {
        return Err(ApiError::with_code(
            "retained-media reuse sessions require API-key authentication",
            "RETAINED_MEDIA_REUSE_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    let credential_identity = reuse_credential_identity(authenticated);
    let handle = handle.to_str().map_err(|_| {
        ApiError::with_code(
            "retained-media reuse session handle is malformed",
            "RETAINED_MEDIA_REUSE_INVALID",
            StatusCode::BAD_REQUEST,
        )
    })?;
    let request_digest = request_sha256(request)?;
    let token_hash = session_token_hash(handle);
    let now = unix_timestamp();
    let session = take_reuse_session(
        &token_hash,
        now,
        state.instance_id.as_str(),
        &credential_identity,
        &request_digest,
    )?;

    let lookup_state = state.clone();
    let filename = session.filename.clone();
    let current = tokio::task::spawn_blocking(move || resolve_members(&lookup_state, &filename))
        .await
        .map_err(|error| ApiError::internal(format!("reuse session lookup failed: {error}")))??
        .filter(|resolved| !resolved.legacy && !resolved.corrupt)
        .ok_or_else(|| {
            ApiError::with_code(
                "retained source media is no longer available",
                "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
                StatusCode::CONFLICT,
            )
        })?;
    if current.archive_identity_sha256 != session.archive_identity_sha256 {
        return Err(ApiError::with_code(
            "gallery item identity changed before retained-media reuse",
            "RETAINED_MEDIA_REUSE_ARCHIVE_CHANGED",
            StatusCode::CONFLICT,
        ));
    }
    for selected in &session.members {
        let current_member = current
            .members
            .iter()
            .find(|member| member.member.member_id == selected.member_id)
            .ok_or_else(|| ApiError::not_found("retained source-media member was removed"))?;
        if current_member.media_set != selected.media_set
            || current_member.pin_id != selected.pin_id
            || current_member.index != selected.index
            || current_member.member.role != selected.role
            || current_member.position != selected.position
        {
            return Err(ApiError::with_code(
                "retained source-media member identity changed",
                "RETAINED_MEDIA_REUSE_ARCHIVE_CHANGED",
                StatusCode::CONFLICT,
            ));
        }
    }
    let lifecycle = state.queue_journal.queue_media_lifecycle().ok_or_else(|| {
        ApiError::with_code(
            "retained source media is unavailable",
            "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
            StatusCode::CONFLICT,
        )
    })?;
    let members = session.members;
    let decrypted = tokio::task::spawn_blocking(move || {
        members
            .into_iter()
            .map(|member| {
                lifecycle
                    .gallery_member_bytes(
                        member.media_set.clone(),
                        member.pin_id.clone(),
                        member.index,
                    )
                    .map(|bytes| (member, bytes))
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|error| ApiError::internal(format!("reuse session decrypt task failed: {error}")))?
    .map_err(|error| ApiError::internal(format!("reuse session decrypt failed: {error}")))?;
    hydrate_selected_members(request, decrypted)
}

fn ensure_hydration_target_is_empty(
    request: &mold_core::GenerateRequest,
    members: &[ReuseSessionMember],
) -> Result<(), ApiError> {
    let roles = members
        .iter()
        .map(|member| member.role.as_str())
        .collect::<HashSet<_>>();
    let source_count = members
        .iter()
        .filter(|member| matches!(member.role.as_str(), "source_image" | "stage_source:0"))
        .count();
    let conflict = source_count > 1
        || (source_count > 0 && request.source_image.is_some())
        || (roles.contains("identity_image") && request.id_image.is_some())
        || (roles.contains("identity_images") && request.id_images.is_some())
        || (roles.contains("edit_images") && request.edit_images.is_some())
        || (roles.contains("mask_image") && request.mask_image.is_some())
        || (roles.contains("control_image") && request.control_image.is_some())
        || ((roles.contains("audio_file") || roles.contains("audio_file_path"))
            && (request.audio_file.is_some() || request.audio_file_path.is_some()))
        || ((roles.contains("source_video") || roles.contains("source_video_path"))
            && (request.source_video.is_some() || request.source_video_path.is_some()))
        || ((roles.contains("extend_video") || roles.contains("extend_video_path"))
            && (request.extend_video.is_some() || request.extend_video_path.is_some()))
        || (roles.contains("keyframes") && request.keyframes.is_some());
    if conflict {
        return Err(ApiError::with_code(
            "target_request already contains authority for a selected retained-media role",
            "RETAINED_MEDIA_REUSE_TARGET_CONFLICT",
            StatusCode::CONFLICT,
        ));
    }
    let selected_references = members
        .iter()
        .filter(|member| member.role == "references")
        .count();
    if selected_references > 0 {
        let references = request.references.as_deref().ok_or_else(|| {
            ApiError::validation(
                "target_request.references must describe every selected retained reference",
            )
        })?;
        if references.len() != selected_references
            || references.iter().any(|reference| {
                !matches!(
                    reference.media(),
                    mold_core::GenerationReferenceAuthority::Descriptor
                )
            })
        {
            return Err(ApiError::validation(
                "target_request.references must be descriptor-only and match the selected retained reference count",
            ));
        }
    }
    Ok(())
}

fn hydrate_selected_members(
    request: &mut mold_core::GenerateRequest,
    decrypted: Vec<(ReuseSessionMember, Vec<u8>)>,
) -> Result<(), ApiError> {
    ensure_hydration_target_is_empty(
        request,
        &decrypted
            .iter()
            .map(|(member, _)| member.clone())
            .collect::<Vec<_>>(),
    )?;
    let mut identity_images = Vec::new();
    let mut edit_images = Vec::new();
    let mut references = Vec::new();
    let mut keyframes = Vec::new();
    for (member, bytes) in decrypted {
        match member.role.as_str() {
            "source_image" | "stage_source:0" => request.source_image = Some(bytes),
            "identity_image" => request.id_image = Some(bytes),
            "identity_images" => identity_images.push(bytes),
            "edit_images" => edit_images.push(bytes),
            "references" => references.push(bytes),
            "mask_image" => request.mask_image = Some(bytes),
            "control_image" => request.control_image = Some(bytes),
            "audio_file" | "audio_file_path" => request.audio_file = Some(bytes),
            "source_video" | "source_video_path" => request.source_video = Some(bytes),
            "extend_video" | "extend_video_path" => request.extend_video = Some(bytes),
            "keyframes" => {
                let keyframe = serde_json::from_slice(&bytes).map_err(|error| {
                    ApiError::with_code(
                        format!("retained keyframe is malformed: {error}"),
                        "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
                        StatusCode::CONFLICT,
                    )
                })?;
                keyframes.push(keyframe);
            }
            role => {
                return Err(ApiError::with_code(
                    format!("retained source-media role '{role}' cannot be reused"),
                    "RETAINED_MEDIA_REUSE_ROLE_UNSUPPORTED",
                    StatusCode::UNPROCESSABLE_ENTITY,
                ));
            }
        }
    }
    if !identity_images.is_empty() {
        request.id_images = Some(identity_images);
    }
    if !edit_images.is_empty() {
        request.edit_images = Some(edit_images);
    }
    if !keyframes.is_empty() {
        request.keyframes = Some(keyframes);
    }
    if !references.is_empty() {
        let descriptors = request
            .references
            .as_mut()
            .expect("target topology was checked above");
        for (reference, bytes) in descriptors.iter_mut().zip(references) {
            match reference {
                mold_core::GenerationReference::Image { media, .. }
                | mold_core::GenerationReference::NamedImage { media, .. }
                | mold_core::GenerationReference::Video { media, .. }
                | mold_core::GenerationReference::Audio { media, .. }
                | mold_core::GenerationReference::Mesh { media, .. } => {
                    *media = mold_core::GenerationReferenceAuthority::Inline { data: bytes };
                }
            }
        }
    }
    Ok(())
}

#[utoipa::path(
    get,
    path = "/api/gallery/source-media/{filename}",
    tag = "gallery",
    params(("filename" = String, Path, description = "Exact gallery filename")),
    responses((status = 200, body = mold_core::RetainedSourceMediaInventory))
)]
pub(crate) async fn inventory(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    AxumPath(filename): AxumPath<String>,
) -> Result<(HeaderMap, Json<Inventory>), ApiError> {
    if !private_media_authorized(auth.as_ref(), authenticated.as_ref()) {
        return Ok((
            private_inventory_headers(),
            Json(Inventory {
                availability: Availability::UnavailableAuth,
                members: Vec::new(),
            }),
        ));
    }
    crate::routes::validate_gallery_filename(&filename)?;
    let resolved = tokio::task::spawn_blocking(move || resolve_members(&state, &filename))
        .await
        .map_err(|error| {
            ApiError::internal(format!("source-media inventory task failed: {error}"))
        })??;
    Ok((
        private_inventory_headers(),
        Json(inventory_response(resolved)),
    ))
}

#[utoipa::path(
    get,
    path = "/api/gallery/source-media/{filename}/{member_id}",
    tag = "gallery",
    params(
        ("filename" = String, Path, description = "Exact gallery filename"),
        ("member_id" = String, Path, description = "Opaque inventory member id")
    ),
    responses(
        (status = 200, description = "Original retained media bytes"),
        (status = 401, description = "API-key authentication is required"),
        (status = 404, description = "Gallery item or member is unavailable")
    )
)]
pub(crate) async fn download(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    AxumPath((filename, requested_member)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    download_member(
        state,
        auth,
        authenticated,
        filename,
        requested_member,
        false,
    )
    .await
}

#[utoipa::path(get, path = "/api/gallery/source-media/{filename}/{member_id}/content", tag = "gallery",
    params(("filename" = String, Path), ("member_id" = String, Path)),
    responses((status = 200, description = "Original attachable media bytes"), (status = 404, description = "Member unavailable")))]
pub(crate) async fn content(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    AxumPath((filename, requested_member)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    download_member(state, auth, authenticated, filename, requested_member, true).await
}

async fn download_member(
    state: AppState,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    filename: String,
    requested_member: String,
    extract_content: bool,
) -> Result<Response, ApiError> {
    if !private_media_authorized(auth.as_ref(), authenticated.as_ref()) {
        return Err(ApiError::with_code(
            "retained source media requires API-key authentication",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    crate::routes::validate_gallery_filename(&filename)?;
    let lifecycle = state.queue_journal.queue_media_lifecycle().ok_or_else(|| {
        ApiError::with_code(
            "retained source media is unavailable",
            "RETAINED_SOURCE_MEDIA_UNAVAILABLE",
            StatusCode::CONFLICT,
        )
    })?;
    let state_for_lookup = state.clone();
    let member = tokio::task::spawn_blocking(move || {
        resolve_members(&state_for_lookup, &filename).map(|members| {
            members
                .map(|resolved| resolved.members)
                .unwrap_or_default()
                .into_iter()
                .find(|member| member.member.member_id == requested_member)
        })
    })
    .await
    .map_err(|error| ApiError::internal(format!("source-media lookup task failed: {error}")))??
    .ok_or_else(|| ApiError::not_found("retained source-media member was not found"))?;
    if member.member.size_bytes > MAX_SOURCE_DOWNLOAD_BYTES {
        return Err(ApiError::with_code(
            "retained source-media member exceeds the v1 download limit",
            "RETAINED_SOURCE_MEDIA_TOO_LARGE",
            StatusCode::PAYLOAD_TOO_LARGE,
        ));
    }
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let bytes = lifecycle
            .gallery_member_bytes(member.media_set, member.pin_id, member.index)
            .map_err(|error| ApiError::internal(format!("source-media decrypt failed: {error}")))?;
        if extract_content {
            media_content(&member.member.role, bytes)
        } else {
            Ok(bytes)
        }
    })
    .await
    .map_err(|error| ApiError::internal(format!("source-media decrypt task failed: {error}")))??;
    let mut response = bytes.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    let disposition = format!(
        "attachment; filename=\"{}\"",
        member.member.display_name.replace(['\\', '"'], "_")
    );
    if let Ok(value) = HeaderValue::from_str(&disposition) {
        response
            .headers_mut()
            .insert(header::CONTENT_DISPOSITION, value);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_extracts_keyframe_image_but_preserves_other_bytes() {
        let json = br#"{"frame":12,"image":"YWJj","name":"frame.png"}"#.to_vec();
        assert_eq!(media_content("keyframes", json).unwrap(), b"abc");
        assert_eq!(
            media_content("source_image", b"raw".to_vec()).unwrap(),
            b"raw"
        );
        assert!(media_content("keyframes", b"broken".to_vec()).is_err());
    }

    #[test]
    fn inventory_batch_rejects_oversized_and_path_requests() {
        assert!(validate_inventory_filenames(&vec!["saved.png".into(); 101]).is_err());
        assert!(validate_inventory_filenames(&["../secret.png".into()]).is_err());
        assert!(validate_inventory_filenames(&["saved.png".into()]).is_ok());
    }

    #[tokio::test]
    async fn batch_inventory_auth_and_exact_owner_scope() {
        let (state, _home, gallery) = fixture().await;
        let auth = Some(Extension(Some(std::sync::Arc::new(
            crate::auth::ApiKeySet::new(HashSet::from(["fixture-key".to_string()])),
        ))));
        assert!(batch_inventory(
            State(state.clone()),
            auth,
            None,
            Json(BatchInventoryRequest {
                filenames: vec!["one.png".into()]
            })
        )
        .await
        .is_err());
        assert_eq!(std::fs::read_dir(gallery.path()).unwrap().count(), 0);
        publish(&state, gallery.path(), "one.png").await;
        publish(&state, gallery.path(), "hidden.png").await;
        let (headers, Json(response)) = batch_inventory(
            State(state.clone()),
            None,
            None,
            Json(BatchInventoryRequest {
                filenames: vec!["one.png".into()],
            }),
        )
        .await
        .unwrap();
        assert_eq!(headers[header::CACHE_CONTROL], "private, no-store");
        assert_eq!(response.inventories.len(), 1);
        assert_eq!(
            response.inventories["one.png"].availability,
            Availability::UnavailableLegacy
        );
        std::fs::write(gallery.path().join("one.png"), b"replaced").unwrap();
        let (_, Json(response)) = batch_inventory(
            State(state),
            None,
            None,
            Json(BatchInventoryRequest {
                filenames: vec!["one.png".into()],
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            response.inventories["one.png"].availability,
            Availability::UnavailableMissingOrCorrupt
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn keyframe_content_and_thumbnail_keep_raw_download_compatible() {
        use axum::body::{to_bytes, Body};
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "saved.png").await;
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(8, 8)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let original = png.into_inner();
        let envelope = serde_json::to_vec(&mold_core::KeyframeCondition {
            frame: 0,
            image: original.clone(),
            name: Some("first.png".into()),
        })
        .unwrap();
        let (_, Json(offer)) = crate::gallery_media_transfer::offer(
            State(state.clone()),
            None,
            None,
            AxumPath("saved.png".into()),
        )
        .await
        .unwrap();
        let mut descriptor = serde_json::to_value(offer).unwrap();
        descriptor["members"] = serde_json::json!([{
            "role":"keyframes", "position":"item:0", "sink":"memory",
            "size_bytes":envelope.len(), "sha256":format!("{:x}", Sha256::digest(&envelope))
        }]);
        let encoded = serde_json::to_vec(&descriptor).unwrap();
        let mut body = (encoded.len() as u32).to_be_bytes().to_vec();
        body.extend(encoded);
        body.extend(&envelope);
        let _ = crate::gallery_media_transfer::receive(
            State(state.clone()),
            None,
            None,
            AxumPath("saved.png".into()),
            Body::from(body),
        )
        .await
        .unwrap();
        let (_, Json(response)) = batch_inventory(
            State(state.clone()),
            None,
            None,
            Json(BatchInventoryRequest {
                filenames: vec!["saved.png".into()],
            }),
        )
        .await
        .unwrap();
        let member = &response.inventories["saved.png"].members[0];
        assert_eq!(member.media_kind.as_deref(), Some("image"));
        assert_eq!(member.position, Some(0));
        assert_eq!(
            member.content_sha256.as_deref(),
            Some(format!("{:x}", Sha256::digest(&original)).as_str())
        );
        let args = ("saved.png".to_string(), member.member_id.clone());
        let raw = download(State(state.clone()), None, None, AxumPath(args.clone()))
            .await
            .unwrap();
        assert_eq!(
            to_bytes(raw.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .as_ref(),
            envelope.as_slice()
        );
        let extracted = content(State(state.clone()), None, None, AxumPath(args.clone()))
            .await
            .unwrap();
        assert_eq!(
            to_bytes(extracted.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .as_ref(),
            original.as_slice()
        );
        let thumbnail = crate::gallery_source_thumbnail::get(
            State(state.clone()),
            None,
            None,
            AxumPath(args.clone()),
        )
        .await
        .unwrap();
        let preview = to_bytes(thumbnail.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        assert!(image::load_from_memory(&preview).is_ok());
        let cache_store = state
            .queue_journal
            .queue_media_lifecycle()
            .unwrap()
            .runtime_store()
            .unwrap();
        let scans_before = cache_store
            .gallery_decode_calls
            .load(std::sync::atomic::Ordering::SeqCst);
        let reads_before = cache_store
            .gallery_member_reads
            .load(std::sync::atomic::Ordering::SeqCst);
        let second = crate::gallery_source_thumbnail::get(
            State(state.clone()),
            None,
            None,
            AxumPath(args.clone()),
        )
        .await
        .unwrap();
        let second = to_bytes(second.into_body(), 2 * 1024 * 1024).await.unwrap();
        assert_eq!(preview, second);
        assert_eq!(
            cache_store
                .gallery_decode_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            scans_before
        );
        assert_eq!(
            cache_store
                .gallery_member_reads
                .load(std::sync::atomic::Ordering::SeqCst),
            reads_before
        );
        assert_eq!(
            state
                .queue_journal
                .queue_media_lifecycle()
                .unwrap()
                .runtime_store()
                .unwrap()
                .gallery_thumbnail_renders
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        std::fs::remove_file(gallery.path().join("saved.png")).unwrap();
        assert!(crate::gallery_source_thumbnail::get(
            State(state.clone()),
            None,
            None,
            AxumPath(args.clone())
        )
        .await
        .is_err());
        assert!(content(State(state), None, None, AxumPath(args))
            .await
            .is_err());
    }

    #[test]
    fn presentation_distinguishes_reference_kinds_and_frame_positions() {
        let mut metadata =
            mold_core::OutputMetadata::from_generate_request(&request(), 1, None, "test");
        metadata.frames = Some(25);
        metadata.keyframes = Some(vec![mold_core::KeyframeMetadata {
            frame: 24,
            name: None,
            sha256: "a".repeat(64),
        }]);
        let mut member = ResolvedMember {
            media_set: crate::queue_media_store::MediaSetRef {
                owner_id: "owner".into(),
                job_id: "job".into(),
                set_id: "set".into(),
            },
            pin_id: "pin".into(),
            index: 0,
            position: "item:0".into(),
            sink: crate::queue_media_store::QueueMediaSink::Memory,
            sha256: "sha".into(),
            member: Member {
                member_id: "id".into(),
                content_sha256: None,
                role: "keyframes".into(),
                display_name: "frame".into(),
                size_bytes: 3,
                label: None,
                media_kind: None,
                position: None,
                frame_index: None,
                image_role: None,
            },
        };
        present_member(&mut member, &metadata);
        assert_eq!(member.member.label.as_deref(), Some("End frame"));
        assert_eq!(member.member.frame_index, Some(24));
        member.member.role = "references".into();
        present_member(&mut member, &metadata);
        assert_eq!(member.member.media_kind.as_deref(), Some("unknown"));
        for kind in ["image", "video", "audio", "mesh"] {
            metadata.references = Some(vec![serde_json::from_value(
                serde_json::json!({"kind":kind,"index":0,"sha256":"sha","mime_type":"test"}),
            )
            .unwrap()]);
            present_member(&mut member, &metadata);
            assert_eq!(member.member.media_kind.as_deref(), Some(kind));
        }
        member.member.role = "chain_continuation_tail:4294967295".into();
        present_member(&mut member, &metadata);
        assert_eq!(
            member.member.label.as_deref(),
            Some("Incoming continuation frame 4294967296")
        );
        assert!(!reusable_role(&member.member.role));
        member.member.role = "chain_continuation_tail:0".into();
        member.position = "continuation-tail".into();
        present_member(&mut member, &metadata);
        assert_eq!(member.member.label.as_deref(), Some("Incoming continuation frame 1"));
        assert!(!reusable_role(&member.member.role));
    }
    /// Opt-in authentic, encrypted fixture for browser/native visual acceptance.
    /// MOLD_SOURCE_MEDIA_UAT_HOME must name a new absolute directory under /tmp.
    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "creates an isolated persistent UAT library only when explicitly requested"]
    async fn build_source_media_uat_fixture() {
        use axum::body::Body;
        use mold_core::{OutputFormat, OutputMetadata};
        use mold_db::{GenerationRecord, RecordSource};
        let home = std::path::PathBuf::from(
            std::env::var("MOLD_SOURCE_MEDIA_UAT_HOME").expect("set isolated fixture destination"),
        );
        assert!(home.is_absolute() && home.starts_with("/tmp") && !home.exists());
        std::fs::create_dir_all(&home).unwrap();
        let gallery = home.join("output");
        std::fs::create_dir(&gallery).unwrap();
        let db = std::sync::Arc::new(Some(
            mold_db::MetadataDb::open(&home.join("mold.db")).unwrap(),
        ));
        let instance = crate::instance::resolve_instance_id(db.as_ref().as_ref(), 17681);
        let journal = std::sync::Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(&home),
            &instance,
        ));
        let lifecycle =
            std::sync::Arc::new(crate::queue_media_lifecycle::QueueMediaLifecycle::new(
                db.clone(),
                home.clone(),
                journal.owner_uuid().unwrap().to_string(),
            ));
        journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .unwrap();
        assert!(
            crate::queue_media_startup::reconcile_claimed_owner(&journal, lifecycle.as_ref())
                .unwrap()
                .durable_media_ready
        );
        let mut state = AppState::for_tests();
        state.queue_journal = journal;
        state.metadata_db = db.clone();
        state.config.write().await.output_dir = Some(gallery.to_string_lossy().into());
        let mut pngs = Vec::new();
        for variant in 0..3 {
            let picture = image::RgbImage::from_fn(384, 256, |x, y| {
                let wave = ((x / 24 + y / 24 + variant) % 2) as u8;
                image::Rgb([
                    ((x + variant * 55) % 255) as u8,
                    (y % 255) as u8,
                    70 + wave * 100,
                ])
            });
            let mut encoded = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(picture)
                .write_to(&mut encoded, image::ImageFormat::Png)
                .unwrap();
            pngs.push(encoded.into_inner());
        }
        for (index, name) in ["source-study.png", "keyframe-study.png", "mixed-study.png"]
            .iter()
            .enumerate()
        {
            let mut req = request();
            req.model = "flux2-klein:4b".into();
            req.prompt = format!("Source Media UAT — study {}", index + 1);
            req.width = 384;
            req.height = 256;
            req.frames = Some(25);
            req.keyframes = Some(vec![mold_core::KeyframeCondition {
                frame: 24,
                image: pngs[1].clone(),
                name: Some("Closing frame.png".into()),
            }]);
            let metadata = OutputMetadata::from_generate_request(&req, 1, None, "uat");
            let record = GenerationRecord::from_save(
                std::path::Path::new("unused"),
                *name,
                OutputFormat::Png,
                metadata,
                RecordSource::Server,
                1,
            );
            let mut transaction = crate::batch_transaction::GalleryImportTransaction::begin(
                &gallery,
                &format!("uat-{index}"),
                0,
                serde_json::json!({"kind":"fixture"}),
                record,
            )
            .unwrap();
            std::fs::write(transaction.staging_path(), &pngs[index]).unwrap();
            transaction.seal_staged_file().unwrap();
            transaction.mark_prepared().unwrap();
            transaction
                .commit(&state.gallery_publication_gate, state.metadata_db.clone())
                .await
                .unwrap();
            let keyframe = serde_json::to_vec(&req.keyframes.as_ref().unwrap()[0]).unwrap();
            // A valid tiny PCM WAV supplies a non-image tile without generation.
            let mut wav = b"RIFF".to_vec();
            wav.extend(38_u32.to_le_bytes());
            wav.extend(b"WAVEfmt ");
            wav.extend(16_u32.to_le_bytes());
            wav.extend(1_u16.to_le_bytes());
            wav.extend(1_u16.to_le_bytes());
            wav.extend(8000_u32.to_le_bytes());
            wav.extend(16000_u32.to_le_bytes());
            wav.extend(2_u16.to_le_bytes());
            wav.extend(16_u16.to_le_bytes());
            wav.extend(b"data");
            wav.extend(2_u32.to_le_bytes());
            wav.extend(0_i16.to_le_bytes());
            let mut members = vec![
                ("source_image", "scalar", pngs[0].clone()),
                ("edit_images", "item:0", pngs[2].clone()),
            ];
            if index > 0 {
                members.push(("keyframes", "item:0", keyframe));
            }
            if index > 1 {
                members.extend([
                    ("audio_file", "scalar", wav),
                    ("mask_image", "scalar", pngs[1].clone()),
                    ("stage_source:1", "authored-source", pngs[2].clone()),
                ]);
            }
            let (_, Json(offer)) = crate::gallery_media_transfer::offer(
                State(state.clone()),
                None,
                None,
                AxumPath((*name).into()),
            )
            .await
            .unwrap();
            let mut descriptor = serde_json::to_value(offer).unwrap();
            descriptor["members"] = serde_json::Value::Array(members.iter().map(|(role, position, bytes)| serde_json::json!({"role":role,"position":position,"sink":"memory","size_bytes":bytes.len(),"sha256":format!("{:x}", Sha256::digest(bytes))})).collect());
            let encoded = serde_json::to_vec(&descriptor).unwrap();
            let mut body = (encoded.len() as u32).to_be_bytes().to_vec();
            body.extend(encoded);
            for (_, _, bytes) in members {
                body.extend(bytes);
            }
            let _ = crate::gallery_media_transfer::receive(
                State(state.clone()),
                None,
                None,
                AxumPath((*name).into()),
                Body::from(body),
            )
            .await
            .unwrap();
        }
        let db = db.as_ref().as_ref().unwrap();
        let collection = db
            .create_collection("Source studies", Some("Retained input UAT fixtures"))
            .unwrap();
        db.collection_add(
            &collection.id,
            &gallery,
            &["source-study.png".into(), "keyframe-study.png".into()],
        )
        .unwrap();
        crate::gallery_authority::release_gallery_writer_leases();
        println!("UAT fixture: {} (port 17681)", home.display());
    }
    async fn fixture() -> (AppState, tempfile::TempDir, tempfile::TempDir) {
        let home = tempfile::tempdir().unwrap();
        let gallery = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let journal = std::sync::Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "transfer-test",
        ));
        let lifecycle =
            std::sync::Arc::new(crate::queue_media_lifecycle::QueueMediaLifecycle::new(
                db.clone(),
                home.path().into(),
                journal.owner_uuid().unwrap().to_string(),
            ));
        journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .unwrap();
        assert!(
            crate::queue_media_startup::reconcile_claimed_owner(&journal, lifecycle.as_ref())
                .unwrap()
                .durable_media_ready
        );
        let mut state = AppState::for_tests();
        state.queue_journal = journal;
        state.metadata_db = db;
        state.config.write().await.output_dir = Some(gallery.path().to_string_lossy().into());
        (state, home, gallery)
    }
    async fn publish(state: &AppState, gallery: &std::path::Path, name: &str) {
        use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
        use mold_db::{GenerationRecord, RecordSource};
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({"prompt":"fixture", "model":"test", "width":64,"height":64,"steps":1,"guidance":1.0})).unwrap();
        let record = GenerationRecord::from_save(
            std::path::Path::new("unused"),
            name,
            OutputFormat::Png,
            OutputMetadata::from_generate_request(&request, 1, None, "test"),
            RecordSource::Server,
            1,
        );
        let mut transaction = crate::batch_transaction::GalleryImportTransaction::begin(
            gallery,
            &format!("import-{name}"),
            0,
            serde_json::json!({"kind":"fixture"}),
            record,
        )
        .unwrap();
        std::fs::write(transaction.staging_path(), name.as_bytes()).unwrap();
        transaction.seal_staged_file().unwrap();
        transaction.mark_prepared().unwrap();
        transaction
            .commit(&state.gallery_publication_gate, state.metadata_db.clone())
            .await
            .unwrap();
    }
    fn request() -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "reuse retained source",
            "model": "mock-model",
            "width": 64,
            "height": 64,
            "steps": 1,
            "batch_size": 1,
            "output_format": "png"
        }))
        .unwrap()
    }

    fn selected(role: &str, index: usize) -> ReuseSessionMember {
        ReuseSessionMember {
            member_id: format!("member-{role}-{index}"),
            media_set: crate::queue_media_store::MediaSetRef {
                owner_id: "owner".to_string(),
                job_id: "job".to_string(),
                set_id: "set".to_string(),
            },
            pin_id: "a".repeat(64),
            index,
            role: role.to_string(),
            position: if index == 0 {
                "scalar".to_string()
            } else {
                format!("item:{}", index - 1)
            },
        }
    }

    #[test]
    fn first_chain_stage_restores_its_picture_without_guessing_other_stages() {
        let mut target = request();
        hydrate_selected_members(
            &mut target,
            vec![(selected("stage_source:0", 0), vec![1, 2, 3])],
        )
        .unwrap();
        assert_eq!(target.source_image, Some(vec![1, 2, 3]));
        let mut target = request();
        assert!(hydrate_selected_members(
            &mut target,
            vec![(selected("stage_source:1", 0), vec![1])]
        )
        .is_err());
        assert!(hydrate_selected_members(
            &mut target,
            vec![
                (selected("source_image", 0), vec![1]),
                (selected("stage_source:0", 0), vec![2])
            ]
        )
        .is_err());
        assert!(target.source_image.is_none());
    }

    #[test]
    fn inventory_exposes_binary_roles_but_not_provenance_text() {
        for role in [
            "source_image",
            "identity_image",
            "identity_images",
            "edit_images",
            "references",
            "mask_image",
            "control_image",
            "audio_file_path",
            "source_video_path",
            "extend_video_path",
            "keyframes",
        ] {
            assert!(downloadable_role(role), "{role}");
        }
        for role in [
            "source_image_name",
            "identity_image_name",
            "identity_image_names",
            "hdr_exr_dir",
            "lora",
            "loras",
        ] {
            assert!(!downloadable_role(role), "{role}");
        }
        for role in [
            "matting_processed_source_image",
            "matting_processed_references",
        ] {
            assert!(downloadable_role(role), "{role}");
            assert!(!reusable_role(role), "{role}");
        }
    }

    #[test]
    fn processed_mattes_can_be_downloaded_but_cannot_mint_a_reuse_session() {
        let member = ResolvedMember {
            media_set: crate::queue_media_store::MediaSetRef {
                owner_id: "owner".into(),
                job_id: "job-derived-matting_processed".into(),
                set_id: "set".into(),
            },
            pin_id: "a".repeat(64),
            index: 0,
            position: "front".into(),
            sink: crate::queue_media_store::QueueMediaSink::Memory,
            sha256: "a".repeat(64),
            member: Member {
                member_id: "processed-front".into(),
                content_sha256: None,
                role: "matting_processed_references".into(),
                display_name: "front".into(),
                size_bytes: 3,
                label: None,
                media_kind: None,
                position: None,
                frame_index: None,
                image_role: None,
            },
        };
        let resolved = ResolvedGalleryMedia {
            archive_identity_sha256: "identity".into(),
            members: vec![member],
            legacy: false,
            corrupt: false,
        };
        assert!(selected_members(&resolved, &["processed-front".into()]).is_err());
    }

    #[test]
    fn member_ids_are_item_scoped_and_display_names_are_header_safe() {
        let first_set = crate::queue_media_store::MediaSetRef {
            owner_id: "owner".into(),
            job_id: "job-a".into(),
            set_id: "0".repeat(32),
        };
        let second_set = crate::queue_media_store::MediaSetRef {
            job_id: "job-b".into(),
            ..first_set.clone()
        };
        let first = member_id(&first_set, &"a".repeat(64), 0);
        assert_eq!(first.len(), 64);
        assert_ne!(first, member_id(&first_set, &"a".repeat(64), 1));
        assert_ne!(first, member_id(&first_set, &"b".repeat(64), 0));
        assert_ne!(first, member_id(&second_set, &"a".repeat(64), 0));
        assert_eq!(
            safe_display_name("../secret\nname.png", "source_image", 0),
            "secretname.png"
        );
        assert_eq!(
            safe_display_name("scalar", "source_image", 0),
            "source_image-1"
        );
    }

    #[test]
    fn hydration_restores_binary_and_path_backed_roles_without_server_paths() {
        let mut request = request();
        hydrate_selected_members(
            &mut request,
            vec![
                (selected("source_image", 0), vec![1]),
                (selected("audio_file_path", 1), vec![2]),
                (selected("source_video_path", 2), vec![3]),
                (selected("extend_video_path", 3), vec![4]),
                (selected("identity_images", 4), vec![5]),
                (selected("identity_images", 5), vec![6]),
                (selected("edit_images", 6), vec![7]),
            ],
        )
        .unwrap();
        assert_eq!(request.source_image, Some(vec![1]));
        assert_eq!(request.audio_file, Some(vec![2]));
        assert_eq!(request.source_video, Some(vec![3]));
        assert_eq!(request.extend_video, Some(vec![4]));
        assert_eq!(request.id_images, Some(vec![vec![5], vec![6]]));
        assert_eq!(request.edit_images, Some(vec![vec![7]]));
        assert!(request.audio_file_path.is_none());
        assert!(request.source_video_path.is_none());
        assert!(request.extend_video_path.is_none());
    }

    /// Reuse settings on a Klein reference print restores the whole ORDERED
    /// set. Order is not cosmetic here: the engine numbers references by
    /// position (`sampling::pack_reference_group`, t = 10, 20, ...), so a
    /// restore that dropped or reordered them would re-render a different
    /// image from settings the user was told were the same. The role is
    /// `edit_images` for every family that speaks the reference protocol —
    /// nothing about this path is keyed on the model, which is why Klein needed
    /// no server change to become reusable.
    #[test]
    fn klein_reuse_restores_every_ordered_reference_image() {
        let mut request = request();
        request.model = "flux2-klein:bf16".to_string();
        hydrate_selected_members(
            &mut request,
            vec![
                (selected("edit_images", 0), vec![11]),
                (selected("edit_images", 1), vec![22]),
                (selected("edit_images", 2), vec![33]),
            ],
        )
        .unwrap();

        assert_eq!(
            request.edit_images,
            Some(vec![vec![11], vec![22], vec![33]]),
            "the restored references must keep the order the print was made with"
        );
        // Klein's source relation is exclusive, so a reference restore must not
        // invent a source image beside them.
        assert!(request.source_image.is_none());
        assert!(request.mask_image.is_none());
    }

    #[test]
    fn hydration_refuses_to_override_client_media() {
        let mut request = request();
        request.source_image = Some(vec![9]);
        let error =
            hydrate_selected_members(&mut request, vec![(selected("source_image", 0), vec![1])])
                .unwrap_err();
        assert_eq!(error.code, "RETAINED_MEDIA_REUSE_TARGET_CONFLICT");
        assert_eq!(request.source_image, Some(vec![9]));
    }

    #[test]
    fn request_digest_is_typed_deterministic_and_media_sensitive() {
        let first = request();
        let mut same = serde_json::from_slice::<mold_core::GenerateRequest>(
            &serde_json::to_vec(&first).unwrap(),
        )
        .unwrap();
        assert_eq!(
            request_sha256(&first).unwrap(),
            request_sha256(&same).unwrap()
        );
        same.source_image = Some(vec![1, 2, 3]);
        assert_ne!(
            request_sha256(&first).unwrap(),
            request_sha256(&same).unwrap()
        );
    }

    #[test]
    fn reuse_header_is_single_child_only_on_batch_admission() {
        let mut headers = HeaderMap::new();
        assert!(!validate_reuse_batch_cardinality(&headers, 5).unwrap());
        headers.insert(REUSE_SESSION_HEADER, HeaderValue::from_static("opaque"));
        assert!(validate_reuse_batch_cardinality(&headers, 1).unwrap());
        assert_eq!(
            validate_reuse_batch_cardinality(&headers, 2)
                .unwrap_err()
                .code,
            "RETAINED_MEDIA_REUSE_BATCH_AMBIGUOUS"
        );
    }

    #[test]
    fn reuse_session_is_scope_bound_and_consumed_exactly_once() {
        let token = format!("test-{}", uuid::Uuid::new_v4());
        let token_hash = session_token_hash(&token);
        let now = unix_timestamp();
        reuse_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                token_hash,
                ReuseSession {
                    instance_id: "instance-a".to_string(),
                    credential_identity: "credential-a".to_string(),
                    filename: "print.png".to_string(),
                    archive_identity_sha256: "archive".to_string(),
                    request_sha256: "request".to_string(),
                    expires_at: now + 60,
                    members: Vec::new(),
                },
            );
        assert_eq!(
            take_reuse_session(&token_hash, now, "instance-b", "credential-a", "request")
                .err()
                .unwrap()
                .code,
            "RETAINED_MEDIA_REUSE_SCOPE_MISMATCH"
        );
        take_reuse_session(&token_hash, now, "instance-a", "credential-a", "request").unwrap();
        assert_eq!(
            take_reuse_session(&token_hash, now, "instance-a", "credential-a", "request")
                .err()
                .unwrap()
                .code,
            "RETAINED_MEDIA_REUSE_INVALID"
        );
    }

    // ── Caller authorization ────────────────────────────────────────────────
    // The predicate answers "may THIS CALLER read the host's retained private
    // media", never "did the operator configure API keys". Conflating the two
    // made every retained-media route dead on a default (keyless) server.

    fn keyed_auth() -> Extension<crate::auth::AuthState> {
        Extension(Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(
            std::collections::HashSet::from(["operator-key".to_string()]),
        ))))
    }

    fn authenticated_marker() -> Extension<crate::auth::ApiKeyAuthenticated> {
        Extension(crate::auth::ApiKeyAuthenticated {
            identity: "key:00000000".to_string(),
            durable_identity: "0".repeat(64),
        })
    }

    #[test]
    fn an_open_host_authorizes_every_caller() {
        // `MOLD_API_KEY` unset: `require_api_key` passes every request through
        // and `DELETE /api/gallery/image/:filename` is already open, so
        // retained media must not be the one route that refuses.
        assert!(private_media_authorized(Some(&Extension(None)), None));
        assert!(private_media_authorized(None, None));
    }

    #[test]
    fn a_keyed_host_authorizes_only_an_authenticated_caller() {
        let keyed = keyed_auth();
        let marker = authenticated_marker();
        assert!(private_media_authorized(Some(&keyed), Some(&marker)));
        assert!(!private_media_authorized(Some(&keyed), None));
    }

    #[test]
    fn an_open_host_binds_reuse_sessions_to_one_stable_anonymous_subject() {
        // Create and hydrate must agree, or a handle minted on an open host
        // could never be redeemed on that same host.
        assert_eq!(
            reuse_credential_identity(None),
            reuse_credential_identity(None)
        );
        let marker = authenticated_marker();
        assert_ne!(
            reuse_credential_identity(Some(&marker)),
            reuse_credential_identity(None),
            "an authenticated caller must never share the anonymous subject"
        );
        assert_eq!(
            reuse_credential_identity(Some(&marker)),
            "0".repeat(64),
            "an authenticated caller keeps its own durable identity"
        );
    }

    #[test]
    fn a_clean_manifest_with_no_downloadable_roles_is_not_corruption() {
        // H3 retains `source_image_name`, which `downloadable_role` filters out
        // by design. An empty member list after a CLEAN resolve means "nothing
        // to restore", never "missing or damaged".
        let clean = ResolvedGalleryMedia {
            archive_identity_sha256: "a".repeat(64),
            members: Vec::new(),
            legacy: false,
            corrupt: false,
        };
        assert_eq!(availability_of(&clean), Availability::UnavailableLegacy);

        let damaged = ResolvedGalleryMedia {
            corrupt: true,
            ..clean
        };
        assert_eq!(
            availability_of(&damaged),
            Availability::UnavailableMissingOrCorrupt
        );
    }
}
