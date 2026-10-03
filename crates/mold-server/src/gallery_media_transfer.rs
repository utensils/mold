//! Exact-output-bound retained-media transfer for library mirrors.
#[cfg(unix)]
use crate::queue_media_store::{GalleryMediaPinRef, MediaSetManifest};
use crate::{queue_media_store::QueueMediaSink, routes::ApiError, state::AppState};
#[cfg(unix)]
use axum::body::{Body, BodyDataStream, Bytes};
use axum::{
    extract::{Extension, Path, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    Json,
};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use sha2::{Digest, Sha256};
use std::collections::HashSet;
#[cfg(unix)]
use tokio::io::AsyncWriteExt;
#[cfg(unix)]
use tokio_stream::StreamExt;
const MAX_MEMBERS: usize = 64;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
#[cfg(unix)]
const MAX_DESCRIPTOR: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TransferMember {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    member_id: Option<String>,
    role: String,
    position: String,
    sink: QueueMediaSink,
    size_bytes: u64,
    sha256: String,
}
#[derive(Serialize, Deserialize)]
pub(crate) struct TransferDescriptor {
    archive_identity_sha256: String,
    members: Vec<TransferMember>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<mold_core::OutputMetadata>,
}
#[derive(Serialize)]
pub(crate) struct TransferResult {
    archive_identity_sha256: String,
    member_count: usize,
}
fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::with_code(
        message,
        "RETAINED_MEDIA_TRANSFER_INVALID",
        StatusCode::BAD_REQUEST,
    )
}
fn conflict() -> ApiError {
    ApiError::with_code(
        "the output or retained-media binding changed",
        "RETAINED_MEDIA_TRANSFER_CONFLICT",
        StatusCode::CONFLICT,
    )
}
fn permitted_role(role: &str) -> bool {
    matches!(
        role,
        "source_image"
            | "identity_image"
            | "identity_images"
            | "edit_images"
            | "references"
            | "mask_image"
            | "control_image"
            | "audio_file"
            | "audio_file_path"
            | "source_video"
            | "source_video_path"
            | "extend_video"
            | "extend_video_path"
            | "keyframes"
            | "matting_processed_source_image"
            | "matting_processed_references"
    )
}
fn validate_members(members: &[TransferMember]) -> Result<(), String> {
    if members.is_empty() || members.len() > MAX_MEMBERS {
        return Err("retained transfer requires 1 to 64 members".into());
    }
    let mut seen = HashSet::new();
    let mut total = 0_u64;
    for member in members {
        let indexed = member
            .position
            .strip_prefix("item:")
            .is_some_and(|v| v.parse::<u32>().is_ok_and(|n| n.to_string() == v));
        let slot = match member.role.as_str() {
            "identity_images" | "edit_images" | "references" | "keyframes" => indexed,
            "matting_processed_references" => matches!(
                member.position.as_str(),
                "front" | "left" | "back" | "right"
            ),
            _ => member.position == "scalar",
        };
        if !permitted_role(&member.role) || !slot {
            return Err("retained transfer role or slot is invalid".into());
        }
        if !seen.insert((&member.role, &member.position)) {
            return Err("retained transfer repeats a role and slot".into());
        }
        if member.size_bytes == 0 || member.size_bytes > MAX_BYTES {
            return Err("retained transfer member exceeds its byte limit".into());
        }
        total = total
            .checked_add(member.size_bytes)
            .filter(|total| *total <= MAX_BYTES)
            .ok_or("retained transfer exceeds its aggregate byte limit")?;
        if member.sha256.len() != 64
            || !member
                .sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err("retained transfer needs a lowercase SHA-256 digest".into());
        }
    }
    Ok(())
}
fn authorize(
    auth: Option<&Extension<crate::auth::AuthState>>,
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
) -> Result<(), ApiError> {
    if crate::gallery_source_media::private_media_authorized(auth, authenticated) {
        Ok(())
    } else {
        Err(ApiError::with_code(
            "retained media requires authentication",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ))
    }
}
#[cfg(unix)]
fn contract(manifest: &MediaSetManifest) -> Vec<TransferMember> {
    manifest
        .entries
        .iter()
        .filter(|entry| entry.size_bytes > 0 && permitted_role(&entry.role))
        .map(|entry| TransferMember {
            member_id: None,
            role: entry.role.clone(),
            position: entry.name.clone(),
            sink: entry.sink,
            size_bytes: entry.size_bytes,
            sha256: entry.sha256_hex.clone(),
        })
        .collect()
}
pub(crate) fn offer_for(state: &AppState, filename: &str) -> Result<TransferDescriptor, ApiError> {
    let config = state.config.blocking_read();
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let output_dir = config.effective_output_dir();
    drop(config);
    let entry = state
        .gallery_publication_gate
        .validated_transfer_entry(&output_dir, filename)
        .map_err(|_| conflict())?
        .ok_or_else(|| ApiError::not_found("gallery output not found"))?;
    let resolved = crate::gallery_source_media::resolve_members_from_pins(
        state,
        entry.identity.clone(),
        entry.retained_media,
    )?
    .ok_or_else(conflict)?;
    if resolved.corrupt {
        return Err(conflict());
    }
    let mut members = Vec::new();
    for member in resolved.members {
        members.push(TransferMember {
            member_id: Some(member.member.member_id),
            role: member.member.role,
            position: member.position,
            sink: member.sink,
            size_bytes: member.member.size_bytes,
            sha256: member.sha256,
        });
    }
    if !members.is_empty() {
        validate_members(&members).map_err(|_| conflict())?;
    }
    Ok(TransferDescriptor {
        archive_identity_sha256: resolved.archive_identity_sha256,
        members,
        output_sha256: Some(entry.identity.checksum_sha256),
        output_size_bytes: Some(entry.identity.size_bytes),
        metadata: Some(entry.record.metadata),
    })
}
#[utoipa::path(
    get, path = "/api/gallery/source-media/{filename}/transfer", tag = "gallery",
    params(("filename" = String, Path, description = "Exact gallery filename")),
    responses((status = 200, description = "Exact archive hash, output SHA-256, byte count, metadata and ordered retained members"), (status = 401, description = "API-key authentication required"), (status = 409, description = "Output or retained media unavailable"))
)]
pub(crate) async fn offer(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path(filename): Path<String>,
) -> Result<(HeaderMap, Json<TransferDescriptor>), ApiError> {
    authorize(auth.as_ref(), authenticated.as_ref())?;
    crate::routes::validate_gallery_filename(&filename)?;
    let _publication = state.gallery_publication_gate.read().await;
    let value = tokio::task::spawn_blocking(move || offer_for(&state, &filename))
        .await
        .map_err(|_| ApiError::internal("transfer offer task failed"))??;
    let mut headers = HeaderMap::new();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok((headers, Json(value)))
}
#[cfg(unix)]
struct FramedBody {
    stream: BodyDataStream,
    pending: Bytes,
}
#[cfg(unix)]
impl FramedBody {
    async fn chunk(&mut self) -> Result<Option<Bytes>, ApiError> {
        if !self.pending.is_empty() {
            return Ok(Some(std::mem::take(&mut self.pending)));
        }
        self.stream
            .next()
            .await
            .transpose()
            .map_err(|_| invalid("transfer body interrupted"))
    }
    async fn exact(&mut self, count: usize) -> Result<Vec<u8>, ApiError> {
        let mut result = Vec::with_capacity(count);
        while result.len() < count {
            let mut chunk = self
                .chunk()
                .await?
                .ok_or_else(|| invalid("transfer body truncated"))?;
            let take = (count - result.len()).min(chunk.len());
            result.extend_from_slice(&chunk.split_to(take));
            self.pending = chunk;
        }
        Ok(result)
    }
}
#[cfg(unix)]
#[utoipa::path(
    put, path = "/api/gallery/source-media/{filename}/transfer", tag = "gallery",
    params(("filename" = String, Path, description = "Exact gallery filename")),
    request_body(content = String, content_type = "application/vnd.mold.retained-media-transfer", description = "Big-endian u32 JSON descriptor length, descriptor, then ordered member bytes; at most 64 members and 512 MiB"),
    responses((status = 200, description = "Exact archive hash and committed member count"), (status = 400, description = "Invalid frame, length or digest"), (status = 401, description = "API-key authentication required"), (status = 409, description = "Output or retained binding changed"))
)]
pub(crate) async fn receive(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path(filename): Path<String>,
    body: Body,
) -> Result<Json<TransferResult>, ApiError> {
    authorize(auth.as_ref(), authenticated.as_ref())?;
    crate::routes::validate_gallery_filename(&filename)?;
    let mut reader = FramedBody {
        stream: body.into_data_stream(),
        pending: Bytes::new(),
    };
    let length =
        u32::from_be_bytes(reader.exact(4).await?.try_into().expect("four bytes")) as usize;
    if length == 0 || length > MAX_DESCRIPTOR {
        return Err(invalid("transfer descriptor exceeds its limit"));
    }
    let descriptor: TransferDescriptor = serde_json::from_slice(&reader.exact(length).await?)
        .map_err(|_| invalid("transfer descriptor is malformed"))?;
    validate_members(&descriptor.members).map_err(invalid)?;
    let lifecycle = state
        .queue_journal
        .queue_media_lifecycle()
        .ok_or_else(|| ApiError::internal("durable media store unavailable"))?;
    let store = lifecycle
        .runtime_store()
        .map_err(|_| ApiError::internal("durable media store unavailable"))?;
    let staging = store
        .transfer_staging()
        .map_err(|_| ApiError::internal("private transfer staging unavailable"))?;
    for (index, member) in descriptor.members.iter().enumerate() {
        let mut file = tokio::fs::File::from_std(
            staging
                .create_file(index)
                .map_err(|_| ApiError::internal("private transfer staging unavailable"))?,
        );
        let mut remaining = member.size_bytes;
        let mut digest = Sha256::new();
        while remaining > 0 {
            let mut chunk = reader
                .chunk()
                .await?
                .ok_or_else(|| invalid("transfer member truncated"))?;
            let take = remaining.min(chunk.len() as u64) as usize;
            let bytes = chunk.split_to(take);
            digest.update(&bytes);
            file.write_all(&bytes)
                .await
                .map_err(|_| ApiError::internal("transfer staging write failed"))?;
            remaining -= take as u64;
            reader.pending = chunk;
        }
        file.flush()
            .await
            .map_err(|_| ApiError::internal("transfer staging flush failed"))?;
        if format!("{:x}", digest.finalize()) != member.sha256 {
            return Err(invalid("transfer member digest mismatch"));
        }
    }
    while let Some(chunk) = reader.chunk().await? {
        if !chunk.is_empty() {
            return Err(invalid("transfer contains trailing bytes"));
        }
    }
    let _publication = state.gallery_publication_gate.write().await;
    let result = tokio::task::spawn_blocking(move || {
        let _staging = staging;
        commit_transfer(&state, &filename, &descriptor, |index, member| {
            let mut media = crate::queue_media_store::SealMedia::path(
                &member.role,
                &member.position,
                _staging.file_path(index),
            )?;
            media.sink = member.sink;
            Ok(media)
        })
    })
    .await
    .map_err(|_| ApiError::internal("transfer commit task failed"))??;
    Ok(Json(result))
}
#[cfg(not(unix))]
#[utoipa::path(
    put, path = "/api/gallery/source-media/{filename}/transfer", tag = "gallery",
    params(("filename" = String, Path, description = "Exact gallery filename")),
    request_body(content = String, content_type = "application/vnd.mold.retained-media-transfer", description = "Big-endian u32 JSON descriptor length, descriptor, then ordered member bytes; at most 64 members and 512 MiB"),
    responses((status = 200, description = "Exact archive hash and committed member count"), (status = 400, description = "Invalid frame, length or digest"), (status = 401, description = "API-key authentication required"), (status = 409, description = "Output or retained binding changed"))
)]
pub(crate) async fn receive() -> Result<Json<TransferResult>, ApiError> {
    Err(ApiError::internal("private transfer staging unavailable"))
}

#[cfg(unix)]
fn normalized(members: &[TransferMember]) -> Vec<TransferMember> {
    members
        .iter()
        .cloned()
        .map(|mut member| {
            member.member_id = None;
            member
        })
        .collect()
}
#[cfg(unix)]
fn commit_transfer(
    state: &AppState,
    filename: &str,
    descriptor: &TransferDescriptor,
    mut media: impl FnMut(
        usize,
        &TransferMember,
    ) -> Result<
        crate::queue_media_store::SealMedia,
        crate::queue_media_store::QueueMediaError,
    >,
) -> Result<TransferResult, ApiError> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _serial = SERIAL
        .lock()
        .map_err(|_| ApiError::internal("transfer store lock unavailable"))?;
    let existing = offer_for(state, filename)?;
    if existing.archive_identity_sha256 != descriptor.archive_identity_sha256 {
        return Err(conflict());
    }
    let wanted = normalized(&descriptor.members);
    if !existing.members.is_empty() {
        if normalized(&existing.members) != wanted {
            return Err(conflict());
        }
        repair_projection(state, filename)?;
        return Ok(TransferResult {
            archive_identity_sha256: existing.archive_identity_sha256,
            member_count: wanted.len(),
        });
    }
    let config = state.config.blocking_read();
    let output_dir = config.effective_output_dir();
    drop(config);
    let identity = match state
        .gallery_publication_gate
        .validated_retained_media_for_item(&output_dir, filename)
        .map_err(|_| conflict())?
    {
        crate::batch_transaction::ValidatedRetainedMedia::Present { identity, .. } => identity,
        _ => return Err(conflict()),
    };
    if crate::gallery_source_media::archive_identity_sha256(&identity)?
        != descriptor.archive_identity_sha256
    {
        return Err(conflict());
    }
    let lifecycle = state
        .queue_journal
        .queue_media_lifecycle()
        .ok_or_else(conflict)?;
    let store = lifecycle.runtime_store().map_err(|_| conflict())?;
    let encoded = serde_json::to_vec(&wanted).map_err(|_| invalid("invalid content contract"))?;
    let fingerprint = crate::queue_media_store::QueueMediaOperationFingerprint::sha256_v1(&encoded);
    let mut identity_digest = Sha256::new();
    identity_digest.update(b"mold-gallery-import-v1\0");
    identity_digest.update(lifecycle.owner_uuid().as_bytes());
    identity_digest.update(&encoded);
    let job = format!("gallery-import-{:x}", identity_digest.finalize());
    let mut candidates = store
        .inspect_gallery_pins()
        .pins
        .into_iter()
        .filter(|pin| {
            pin.media_set.owner_id == lifecycle.owner_uuid() && pin.media_set.job_id == job
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| a.pin_id.cmp(&b.pin_id));
    let mut reused: Option<GalleryMediaPinRef> = None;
    if let Some(candidate) = candidates.into_iter().next() {
        let manifest = store
            .load_from_gallery_pin(&candidate)
            .map_err(|_| conflict())?;
        if contract(&manifest) != wanted {
            return Err(conflict());
        }
        reused = Some(candidate);
    }
    let newly_sealed = reused.is_none();
    let set = if let Some(pin) = &reused {
        pin.media_set.clone()
    } else {
        let payloads = wanted
            .iter()
            .enumerate()
            .map(|(index, member)| media(index, member))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("transfer input unavailable"))?;
        lifecycle
            .seal_v2(&job, &fingerprint, &Default::default(), payloads)
            .map_err(|_| ApiError::internal("transfer encryption failed"))?
    };
    let bound = state
        .gallery_publication_gate
        .bind_transferred_media_for_output(&output_dir, &identity, &set, |pin_id| {
            if let Some(source) = &reused {
                store
                    .pin_gallery_copy(source, pin_id)
                    .map(|_| ())
                    .map_err(Into::into)
            } else {
                store
                    .pin_for_gallery_item(&set, pin_id)
                    .map(|_| ())
                    .map_err(Into::into)
            }
        });
    // Failed binding leaves at most an orphan pin, never archive authority
    // pointing at nonexistent bytes. Startup already reconciles orphan pins.
    if newly_sealed {
        let _ = lifecycle.delete_unpublished(&set);
    }
    bound.map_err(|_| conflict())?;
    repair_projection(state, filename)?;
    Ok(TransferResult {
        archive_identity_sha256: descriptor.archive_identity_sha256.clone(),
        member_count: wanted.len(),
    })
}

#[cfg(unix)]
fn repair_projection(state: &AppState, filename: &str) -> Result<(), ApiError> {
    let config = state.config.blocking_read();
    let output_dir = config.effective_output_dir();
    drop(config);
    let canonical = std::fs::canonicalize(&output_dir).map_err(|_| conflict())?;
    let bindings = state
        .gallery_publication_gate
        .retained_media_for_item(&canonical, filename)
        .map_err(|_| conflict())?
        .ok_or_else(conflict)?
        .1
        .into_iter()
        .map(|binding| mold_db::gallery_media::GalleryMediaBinding {
            output_dir: canonical.to_string_lossy().into_owned(),
            filename: filename.to_string(),
            pin_id: binding.pin_id,
            media_set_id: binding.media_set.set_id,
            owner_uuid: binding.media_set.owner_id,
            job_id: binding.media_set.job_id,
        })
        .collect::<Vec<_>>();
    let lifecycle = state
        .queue_journal
        .queue_media_lifecycle()
        .ok_or_else(conflict)?;
    mold_db::gallery_media::replace_for_item(
        lifecycle.db().map_err(|_| conflict())?,
        &canonical.to_string_lossy(),
        filename,
        &bindings,
    )
    .map_err(|_| ApiError::internal("retained-media projection repair failed"))?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    fn member() -> TransferMember {
        TransferMember {
            member_id: None,
            role: "source_image".into(),
            position: "scalar".into(),
            sink: QueueMediaSink::Memory,
            size_bytes: 3,
            sha256: "a".repeat(64),
        }
    }
    #[test]
    fn transfer_descriptor_rejects_duplicate_slots_and_unbounded_uploads() {
        assert!(validate_members(&[member(), member()]).is_err());
        let mut oversized = member();
        oversized.size_bytes = MAX_BYTES + 1;
        assert!(validate_members(&[oversized]).is_err());
    }
    #[test]
    fn transfer_never_accepts_paths_or_provenance_text_as_slots() {
        let mut value = member();
        value.position = "/private/source.png".into();
        assert!(validate_members(&[value]).is_err());
        let mut value = member();
        value.role = "source_image_name".into();
        assert!(validate_members(&[value]).is_err());
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
    #[tokio::test]
    async fn transferred_sources_survive_origin_removal_and_deduplicate_siblings() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "first.png").await;
        publish(&state, gallery.path(), "second.png").await;
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            let mut value = member();
            value.sha256 = format!("{:x}", Sha256::digest(b"abc"));
            for filename in ["first.png", "second.png"] {
                let mut offer = offer_for(&state, filename).unwrap();
                offer.members = vec![value.clone()];
                commit_transfer(&state, filename, &offer, |_, m| {
                    Ok(crate::queue_media_store::SealMedia::bytes(
                        &m.role,
                        &m.position,
                        b"abc".to_vec(),
                    ))
                })
                .unwrap();
                commit_transfer(&state, filename, &offer, |_, _| {
                    panic!("idempotent transfer must not seal")
                })
                .unwrap();
            }
            let lifecycle = state.queue_journal.queue_media_lifecycle().unwrap();
            let store = lifecycle.runtime_store().unwrap();
            let pins = store.inspect_gallery_pins().pins;
            assert_eq!(pins.len(), 2);
            assert_eq!(pins[0].media_set, pins[1].media_set);
            let mut offer = offer_for(&state, "second.png").unwrap();
            lifecycle
                .release_gallery_pin(pins[0].media_set.clone(), pins[0].pin_id.clone())
                .unwrap();
            assert_eq!(
                lifecycle
                    .gallery_member_bytes(pins[1].media_set.clone(), pins[1].pin_id.clone(), 0)
                    .unwrap(),
                b"abc"
            );
            offer.archive_identity_sha256 = "0".repeat(64);
            assert!(commit_transfer(&state, "second.png", &offer, |_, _| panic!()).is_err());
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    async fn framed_body_rejects_truncation_and_preserves_chunk_boundaries() {
        let mut reader = FramedBody {
            stream: Body::from("abc").into_data_stream(),
            pending: Bytes::new(),
        };
        assert_eq!(reader.exact(2).await.unwrap(), b"ab");
        assert_eq!(reader.exact(1).await.unwrap(), b"c");
        assert!(reader.exact(1).await.is_err());
    }

    fn framed(descriptor: &TransferDescriptor, payload: &[u8]) -> Vec<u8> {
        let encoded = serde_json::to_vec(descriptor).unwrap();
        let mut body = (encoded.len() as u32).to_be_bytes().to_vec();
        body.extend(encoded);
        body.extend(payload);
        body
    }
    #[tokio::test]
    async fn streamed_transfer_rejects_digest_truncation_trailing_and_replaced_output() {
        let (state, home, gallery) = fixture().await;
        publish(&state, gallery.path(), "stream.png").await;
        let snapshot_state = state.clone();
        let mut descriptor =
            tokio::task::spawn_blocking(move || offer_for(&snapshot_state, "stream.png").unwrap())
                .await
                .unwrap();
        let mut value = member();
        value.sha256 = format!("{:x}", Sha256::digest(b"abc"));
        descriptor.members = vec![value];
        for payload in [b"ab".as_slice(), b"abd", b"abcd"] {
            assert!(receive(
                State(state.clone()),
                None,
                None,
                Path("stream.png".into()),
                Body::from(framed(&descriptor, payload))
            )
            .await
            .is_err());
        }
        let interrupted = Body::from_stream(tokio_stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(framed(&descriptor, b"a"))),
            Err(std::io::Error::other("interrupted fixture")),
        ]));
        assert!(receive(
            State(state.clone()),
            None,
            None,
            Path("stream.png".into()),
            interrupted
        )
        .await
        .is_err());
        fn has_transfer_stage(path: &std::path::Path) -> bool {
            std::fs::read_dir(path).unwrap().any(|entry| {
                let entry = entry.unwrap();
                entry.file_name().to_string_lossy().starts_with("transfer-")
                    || (entry.file_type().unwrap().is_dir() && has_transfer_stage(&entry.path()))
            })
        }
        assert!(
            !has_transfer_stage(home.path()),
            "failed or interrupted transfers must erase their private plaintext staging"
        );
        let snapshot_state = state.clone();
        assert!(
            tokio::task::spawn_blocking(move || offer_for(&snapshot_state, "stream.png")
                .unwrap()
                .members
                .is_empty())
            .await
            .unwrap()
        );
        // A downloaded old output cannot acquire a source binding on replacement bytes.
        std::fs::write(gallery.path().join("stream.png"), b"replacement-output").unwrap();
        assert!(receive(
            State(state),
            None,
            None,
            Path("stream.png".into()),
            Body::from(framed(&descriptor, b"abc"))
        )
        .await
        .is_err());
    }
    #[tokio::test]
    async fn streamed_transfer_preserves_order_sinks_and_safe_slots() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "ordered.png").await;
        let snapshot_state = state.clone();
        let mut descriptor =
            tokio::task::spawn_blocking(move || offer_for(&snapshot_state, "ordered.png").unwrap())
                .await
                .unwrap();
        let mut first = member();
        first.role = "references".into();
        first.position = "item:1".into();
        first.sha256 = format!("{:x}", Sha256::digest(b"abc"));
        let mut second = first.clone();
        second.position = "item:0".into();
        second.sink = QueueMediaSink::PrivateStaging;
        second.sha256 = format!("{:x}", Sha256::digest(b"def"));
        descriptor.members = vec![first, second];
        let _ = receive(
            State(state.clone()),
            None,
            None,
            Path("ordered.png".into()),
            Body::from(framed(&descriptor, b"abcdef")),
        )
        .await
        .unwrap();
        tokio::task::spawn_blocking(move || {
            let offer = offer_for(&state, "ordered.png").unwrap();
            assert_eq!(normalized(&offer.members), descriptor.members);
            let resolved = crate::gallery_source_media::resolve_members(&state, "ordered.png")
                .unwrap()
                .unwrap();
            let lifecycle = state.queue_journal.queue_media_lifecycle().unwrap();
            for (member, bytes) in resolved.members.into_iter().zip([b"abc", b"def"]) {
                assert_eq!(
                    lifecycle
                        .gallery_member_bytes(member.media_set, member.pin_id, member.index)
                        .unwrap(),
                    bytes
                );
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn transfer_auth_follows_explicit_server_policy() {
        assert!(authorize(None, None).is_ok());
        let auth = Extension(Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(
            HashSet::from(["secret".into()]),
        ))));
        assert!(authorize(Some(&auth), None).is_err());
    }
    #[tokio::test]
    async fn copied_library_output_retains_source_after_origin_output_and_pin_deletion() {
        let (origin, _origin_home, origin_gallery) = fixture().await;
        let (target, _target_home, target_gallery) = fixture().await;
        publish(&origin, origin_gallery.path(), "copy.png").await;
        publish(&target, target_gallery.path(), "copy.png").await;
        let original = origin.clone();
        let source_offer = tokio::task::spawn_blocking(move || {
            let mut offer = offer_for(&original, "copy.png").unwrap();
            let mut value = member();
            value.sha256 = format!("{:x}", Sha256::digest(b"abc"));
            offer.members = vec![value];
            commit_transfer(&original, "copy.png", &offer, |_, m| {
                Ok(crate::queue_media_store::SealMedia::bytes(
                    &m.role,
                    &m.position,
                    b"abc".to_vec(),
                ))
            })
            .unwrap();
            offer_for(&original, "copy.png").unwrap()
        })
        .await
        .unwrap();
        let destination = target.clone();
        let mut target_offer =
            tokio::task::spawn_blocking(move || offer_for(&destination, "copy.png").unwrap())
                .await
                .unwrap();
        assert_eq!(source_offer.output_sha256, target_offer.output_sha256);
        target_offer.members = source_offer.members;
        let _ = receive(
            State(target.clone()),
            None,
            None,
            Path("copy.png".into()),
            Body::from(framed(&target_offer, b"abc")),
        )
        .await
        .unwrap();
        std::fs::remove_file(origin_gallery.path().join("copy.png")).unwrap();
        tokio::task::spawn_blocking(move || {
            let origin_lifecycle = origin.queue_journal.queue_media_lifecycle().unwrap();
            for pin in origin_lifecycle
                .runtime_store()
                .unwrap()
                .inspect_gallery_pins()
                .pins
            {
                origin_lifecycle
                    .release_gallery_pin(pin.media_set, pin.pin_id)
                    .unwrap();
            }
            let resolved = crate::gallery_source_media::resolve_members(&target, "copy.png")
                .unwrap()
                .unwrap();
            assert!(!resolved.corrupt);
            assert_eq!(resolved.members.len(), 1);
            let member = &resolved.members[0];
            assert_eq!(
                target
                    .queue_journal
                    .queue_media_lifecycle()
                    .unwrap()
                    .gallery_member_bytes(
                        member.media_set.clone(),
                        member.pin_id.clone(),
                        member.index
                    )
                    .unwrap(),
                b"abc"
            );
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    async fn output_replaced_during_seal_never_acquires_binding() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "racing.png").await;
        tokio::task::spawn_blocking(move || {
            let mut descriptor = offer_for(&state, "racing.png").unwrap();
            let mut value = member();
            value.sha256 = format!("{:x}", Sha256::digest(b"abc"));
            descriptor.members = vec![value];
            let result = commit_transfer(&state, "racing.png", &descriptor, |_, m| {
                std::fs::write(gallery.path().join("racing.png"), b"replaced").unwrap();
                Ok(crate::queue_media_store::SealMedia::bytes(
                    &m.role,
                    &m.position,
                    b"abc".to_vec(),
                ))
            });
            assert!(result.is_err());
            assert!(state
                .queue_journal
                .queue_media_lifecycle()
                .unwrap()
                .runtime_store()
                .unwrap()
                .inspect_gallery_pins()
                .pins
                .is_empty());
            assert!(state
                .gallery_publication_gate
                .retained_media_for_item(gallery.path(), "racing.png")
                .unwrap()
                .unwrap()
                .1
                .is_empty());
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    async fn output_replaced_while_pinning_never_acquires_binding() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "pin-race.png").await;
        tokio::task::spawn_blocking(move || {
            let entry = state
                .gallery_publication_gate
                .validated_transfer_entry(gallery.path(), "pin-race.png")
                .unwrap()
                .unwrap();
            let lifecycle = state.queue_journal.queue_media_lifecycle().unwrap();
            let store = lifecycle.runtime_store().unwrap();
            let set = lifecycle
                .seal_v2(
                    "pin-race",
                    &crate::queue_media_store::QueueMediaOperationFingerprint::sha256_v1(
                        b"fixture",
                    ),
                    &Default::default(),
                    vec![crate::queue_media_store::SealMedia::bytes(
                        "source_image",
                        "scalar",
                        b"abc".to_vec(),
                    )],
                )
                .unwrap();
            let result = state
                .gallery_publication_gate
                .bind_transferred_media_for_output(
                    gallery.path(),
                    &entry.identity,
                    &set,
                    |pin_id| {
                        store.pin_for_gallery_item(&set, pin_id)?;
                        std::fs::write(gallery.path().join("pin-race.png"), b"replacement")?;
                        Ok(())
                    },
                );
            assert!(
                result.is_err(),
                "replacement between pinning and authority commit must fail closed"
            );
            assert!(state
                .gallery_publication_gate
                .retained_media_for_item(gallery.path(), "pin-race.png")
                .unwrap()
                .unwrap()
                .1
                .is_empty());
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn transfer_offer_waits_for_gallery_publication_writer() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "gated.png").await;
        let writer = state.gallery_publication_gate.write().await;
        let mut request = tokio::spawn(offer(State(state), None, None, Path("gated.png".into())));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut request)
                .await
                .is_err()
        );
        drop(writer);
        assert!(request.await.unwrap().is_ok());
    }
}
