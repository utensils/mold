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

fn private_auth_enabled(auth: Option<&Extension<crate::auth::AuthState>>) -> bool {
    auth.is_some_and(|Extension(state)| state.is_some())
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
struct ResolvedMember {
    media_set: crate::queue_media_store::MediaSetRef,
    pin_id: String,
    index: usize,
    position: String,
    member: Member,
}

struct ResolvedGalleryMedia {
    archive_identity_sha256: String,
    members: Vec<ResolvedMember>,
    legacy: bool,
    corrupt: bool,
}

fn resolve_members(
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
    let (identity, pins) = match lookup {
        crate::batch_transaction::ValidatedRetainedMedia::Missing => return Ok(None),
        crate::batch_transaction::ValidatedRetainedMedia::Invalid => {
            return Ok(Some(ResolvedGalleryMedia {
                archive_identity_sha256: String::new(),
                members: Vec::new(),
                legacy: false,
                corrupt: true,
            }));
        }
        crate::batch_transaction::ValidatedRetainedMedia::Present { identity, pins } => {
            (identity, pins)
        }
    };
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
                member: Member {
                    member_id: member_id(&pin.media_set, &pin.pin_id, index),
                    role: entry.role.clone(),
                    display_name: safe_display_name(&entry.name, &entry.role, index),
                    size_bytes: entry.size_bytes,
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

fn archive_identity_sha256(
    identity: &crate::batch_transaction::ArchivedChildIdentity,
) -> Result<String, ApiError> {
    let bytes = serde_json::to_vec(identity).map_err(|error| {
        ApiError::internal(format!("gallery identity encoding failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
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
    if !private_auth_enabled(auth.as_ref()) || authenticated.is_none() {
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
    let credential_identity = authenticated
        .as_ref()
        .expect("checked above")
        .durable_identity
        .clone();
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
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
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
    if !private_auth_enabled(auth) {
        return Err(ApiError::with_code(
            "retained-media reuse sessions require API-key authentication",
            "RETAINED_MEDIA_REUSE_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    let authenticated = authenticated.ok_or_else(|| {
        ApiError::with_code(
            "retained-media reuse session has no authenticated credential",
            "RETAINED_MEDIA_REUSE_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        )
    })?;
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
        &authenticated.durable_identity,
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
    let conflict = (roles.contains("source_image") && request.source_image.is_some())
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
            "source_image" => request.source_image = Some(bytes),
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
                | mold_core::GenerationReference::Video { media, .. }
                | mold_core::GenerationReference::Audio { media, .. } => {
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
    AxumPath(filename): AxumPath<String>,
) -> Result<Json<Inventory>, ApiError> {
    if !private_auth_enabled(auth.as_ref()) {
        return Ok(Json(Inventory {
            availability: Availability::UnavailableAuth,
            members: Vec::new(),
        }));
    }
    crate::routes::validate_gallery_filename(&filename)?;
    let resolved = tokio::task::spawn_blocking(move || resolve_members(&state, &filename))
        .await
        .map_err(|error| {
            ApiError::internal(format!("source-media inventory task failed: {error}"))
        })??;
    Ok(Json(match resolved {
        None => Inventory {
            availability: Availability::UnavailableLegacy,
            members: Vec::new(),
        },
        Some(resolved) if resolved.legacy => Inventory {
            availability: Availability::UnavailableLegacy,
            members: Vec::new(),
        },
        Some(resolved) if resolved.corrupt || resolved.members.is_empty() => Inventory {
            availability: Availability::UnavailableMissingOrCorrupt,
            members: Vec::new(),
        },
        Some(resolved) => Inventory {
            availability: Availability::Available,
            members: resolved
                .members
                .into_iter()
                .map(|member| member.member)
                .collect(),
        },
    }))
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
    AxumPath((filename, requested_member)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    if !private_auth_enabled(auth.as_ref()) {
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
    let bytes = tokio::task::spawn_blocking(move || {
        lifecycle.gallery_member_bytes(member.media_set, member.pin_id, member.index)
    })
    .await
    .map_err(|error| ApiError::internal(format!("source-media decrypt task failed: {error}")))?
    .map_err(|error| ApiError::internal(format!("source-media decrypt failed: {error}")))?;
    let mut response = bytes.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
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
}
