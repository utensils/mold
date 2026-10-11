//! Authenticated role-aware previews of sealed queue conditioning, independent of model.
use crate::{routes::ApiError, state::AppState};
use axum::{
    extract::{Path, Query, State},
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct QueueInput {
    index: usize,
    label: String,
    preview: bool,
}

#[derive(Default, Deserialize)]
pub(crate) struct PreviewQuery {
    index: Option<usize>,
}

/// Unlike thumbnails, inspection never silently selects a different member.
#[derive(Deserialize)]
pub(crate) struct OriginalQuery {
    index: usize,
}

fn input_label(role: &str, position: &str) -> Option<String> {
    if position == "collection" {
        return None;
    }
    let ordinal = position
        .strip_prefix("item:")
        .and_then(|v| v.parse::<u32>().ok())
        .map(|v| u64::from(v) + 1);
    let label = match role {
        "source_image" => "Source image",
        "identity_image" | "identity_images" => "Identity photo",
        "edit_images" => "Reference image",
        "references" => "Reference",
        "keyframes" => "Keyframe",
        "mask_image" => "Mask",
        "control_image" => "Control image",
        "audio_file" | "audio_file_path" => "Source audio",
        "source_video" | "source_video_path" => "Source video",
        "extend_video" | "extend_video_path" => "Continuation video",
        _ => return None,
    };
    Some(ordinal.map_or_else(|| label.to_string(), |n| format!("{label} {n}")))
}

fn inputs(
    manifest: &crate::queue_media_store::MediaSetManifest,
    request: &serde_json::Value,
) -> Vec<QueueInput> {
    manifest
        .entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let mut label = input_label(&entry.role, &entry.name)?;
            let mut preview = matches!(
                entry.role.as_str(),
                "source_image"
                    | "identity_image"
                    | "identity_images"
                    | "edit_images"
                    | "mask_image"
                    | "control_image"
                    | "keyframes"
            );
            if entry.role == "references" {
                let slot = entry.name.strip_prefix("item:")?.parse::<usize>().ok()?;
                let reference = request.get("references")?.get(slot)?;
                let kind = reference
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("media");
                preview = matches!(kind, "image" | "named_image");
                label = format!("Reference {} · {}", slot + 1, kind.replace('_', " "));
                if let Some(view) = reference.get("role").and_then(|v| v.as_str()) {
                    label.push_str(&format!(" · {view}"));
                }
            }
            Some(QueueInput {
                index,
                label,
                preview: preview && entry.size_bytes <= MAX_SOURCE_BYTES as u64,
            })
        })
        .collect()
}

struct ResolvedInputs {
    lifecycle: std::sync::Arc<crate::queue_media_lifecycle::QueueMediaLifecycle>,
    manifest: crate::queue_media_store::MediaSetManifest,
    inputs: Vec<QueueInput>,
}

fn resolve(state: &AppState, id: &str) -> Result<Option<ResolvedInputs>, ApiError> {
    let row = state
        .queue_journal
        .row(id)
        .map_err(|_| ApiError::internal("queue lookup failed"))?
        .ok_or_else(|| ApiError::not_found("queue job not found"))?;
    let Some(set_id) = row.media_set_id else {
        return Ok(None);
    };
    let lifecycle = state
        .queue_journal
        .queue_media_lifecycle()
        .ok_or_else(|| ApiError::not_found("queue inputs unavailable"))?;
    let media_set = crate::queue_media_store::MediaSetRef {
        owner_id: row.owner_uuid,
        job_id: row.id,
        set_id,
    };
    let manifest = lifecycle
        .input_manifest(&media_set)
        .map_err(|_| ApiError::not_found("queue inputs unavailable"))?;
    let request: serde_json::Value = serde_json::from_str(&row.request_json)
        .map_err(|_| ApiError::not_found("queue inputs unavailable"))?;
    let inputs = inputs(&manifest, &request);
    Ok(Some(ResolvedInputs {
        lifecycle,
        manifest,
        inputs,
    }))
}

async fn permit() -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    static SLOTS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
        std::sync::OnceLock::new();
    SLOTS
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("queue inputs unavailable"))
}

#[utoipa::path(get, path = "/api/queue/{id}/inputs", tag = "queue",
    params(("id" = String, Path, description = "Queue job id")),
    responses((status = 200, description = "Ordered private conditioning descriptors", body = [QueueInput]),
              (status = 404, description = "Inputs unavailable")))]
pub(crate) async fn list(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let permit = permit().await?;
    let items = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        resolve(&state, &id).map(|resolved| resolved.map(|r| r.inputs).unwrap_or_default())
    })
    .await
    .map_err(|_| ApiError::internal("queue input task failed"))??;
    Ok(([(header::CACHE_CONTROL, "private, no-store")], Json(items)).into_response())
}

#[utoipa::path(get, path = "/api/queue/{id}/input-thumbnail", tag = "queue",
    params(("id" = String, Path, description = "Queue job id"), ("index" = Option<usize>, Query, description = "Input descriptor index; omitted preserves the scalar source image")),
    responses((status = 200, description = "Private 320-pixel conditioning preview", content_type = "image/png"),
              (status = 404, description = "Preview unavailable")))]
pub(crate) async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<PreviewQuery>,
) -> Result<Response, ApiError> {
    let permit = permit().await?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let _permit = permit;
        let resolved =
            resolve(&state, &id)?.ok_or_else(|| ApiError::not_found("queue inputs unavailable"))?;
        let input = resolved
            .inputs
            .iter()
            .find(|input| {
                input.preview
                    && match query.index {
                        Some(index) => index == input.index,
                        None => {
                            resolved.manifest.entries[input.index].role == "source_image"
                                && resolved.manifest.entries[input.index].name == "scalar"
                        }
                    }
            })
            .ok_or_else(|| ApiError::not_found("queue input preview unavailable"))?;
        let bytes = resolved
            .lifecycle
            .input_member_bytes(&resolved.manifest, input.index)
            .map_err(|_| ApiError::not_found("queue input preview unavailable"))?;
        let bytes = zeroize::Zeroizing::new(
            preview_pixels(&resolved.manifest.entries[input.index].role, bytes)
                .map_err(|_| ApiError::not_found("queue input preview unavailable"))?,
        );
        render(&bytes).map_err(|_| ApiError::not_found("queue input preview unavailable"))
    })
    .await
    .map_err(|_| ApiError::internal("queue thumbnail task failed"))??;
    Ok((
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "private, no-store"),
        ],
        bytes,
    )
        .into_response())
}

#[utoipa::path(get, path = "/api/queue/{id}/input-original", tag = "queue",
    params(("id" = String, Path, description = "Queue job id"), ("index" = usize, Query, description = "Required exact sealed input descriptor index")),
    responses((status = 200, description = "Private full-sized submitted input media"),
              (status = 400, description = "Missing or invalid index"),
              (status = 401, description = "Authentication required on keyed hosts"),
              (status = 404, description = "Input unavailable, unsupported or over 64 MiB")))]
pub(crate) async fn original(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<OriginalQuery>,
) -> Result<Response, ApiError> {
    let permit = permit().await?;
    let (mime, bytes) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let resolved =
            resolve(&state, &id)?.ok_or_else(|| ApiError::not_found("queue inputs unavailable"))?;
        // The inventory, not preview eligibility, authorizes this exact member.
        // Audio/video and large full-resolution images may have no thumbnail.
        if !resolved
            .inputs
            .iter()
            .any(|input| input.index == query.index)
        {
            return Err(ApiError::not_found("queue input unavailable"));
        }
        let entry = &resolved.manifest.entries[query.index];
        if entry.size_bytes > MAX_SOURCE_BYTES as u64 {
            return Err(ApiError::not_found("queue input exceeds inspection limit"));
        }
        let bytes = resolved
            .lifecycle
            .input_member_bytes(&resolved.manifest, query.index)
            .map_err(|_| ApiError::not_found("queue input unavailable"))?;
        let (mime, content) = original_content(&entry.role, bytes)
            .map_err(|_| ApiError::not_found("queue input format unavailable"))?;
        let mut content = zeroize::Zeroizing::new(content);
        // Deletion or replacement during the authenticated read revokes the offer.
        let current = state
            .queue_journal
            .row(&id)
            .map_err(|_| ApiError::internal("queue lookup failed"))?
            .ok_or_else(|| ApiError::not_found("queue job not found"))?;
        if current.media_set_id.as_deref() != Some(resolved.manifest.media_set.set_id.as_str())
            || current.owner_uuid != resolved.manifest.media_set.owner_id
        {
            return Err(ApiError::not_found("queue input changed"));
        }
        Ok((mime, std::mem::take(&mut *content)))
    })
    .await
    .map_err(|_| ApiError::internal("queue original task failed"))??;
    Ok((
        [
            (header::CONTENT_TYPE, mime),
            (header::CACHE_CONTROL, "private, no-store"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        bytes,
    )
        .into_response())
}

fn original_content(role: &str, bytes: Vec<u8>) -> anyhow::Result<(&'static str, Vec<u8>)> {
    let mut bytes = zeroize::Zeroizing::new(bytes);
    anyhow::ensure!(bytes.len() <= MAX_SOURCE_BYTES, "submitted input too large");
    let mut bytes = zeroize::Zeroizing::new(preview_pixels(role, std::mem::take(&mut *bytes))?);
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len() <= MAX_SOURCE_BYTES,
        "submitted input unavailable"
    );
    let mime = if let Some(format) = mold_core::validation::sniff_image_input_format(&bytes) {
        use mold_core::generation_profile::ImageInputFormat;
        match format {
            ImageInputFormat::Png => "image/png",
            ImageInputFormat::Jpeg => "image/jpeg",
            ImageInputFormat::Webp => "image/webp",
        }
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        "audio/wav"
    } else if bytes.len() >= 12
        && &bytes[4..8] == b"ftyp"
        && matches!(
            &bytes[8..12],
            b"isom"
                | b"iso2"
                | b"iso3"
                | b"iso4"
                | b"iso5"
                | b"iso6"
                | b"iso7"
                | b"iso8"
                | b"iso9"
                | b"mp41"
                | b"mp42"
                | b"avc1"
                | b"dash"
                | b"M4V "
                | b"MSNV"
        )
    {
        "video/mp4"
    } else {
        anyhow::bail!("unsupported submitted media format");
    };
    Ok((mime, std::mem::take(&mut *bytes)))
}

fn preview_pixels(role: &str, bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    if role == "keyframes" {
        let bytes = zeroize::Zeroizing::new(bytes);
        let frame: mold_core::KeyframeCondition = serde_json::from_slice(&bytes)?;
        Ok(frame.image)
    } else {
        Ok(bytes)
    }
}

pub(crate) fn render(bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(bytes.len() <= MAX_SOURCE_BYTES, "source image too large");
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = crate::thumbnails::downscale(&reader.decode()?, 320);
    Ok(crate::thumbnails::encode(&image, crate::thumbnails::ThumbFormat::Png)?.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[tokio::test]
    async fn original_route_returns_exact_ordered_member_and_refuses_missing_authority() {
        use crate::queue_media_store::{QueueMediaStore, SealMedia};
        use axum::{
            body::{to_bytes, Body},
            http::{Request, StatusCode},
            routing::get,
            Router,
        };
        use mold_db::generation_queue::{self, GenerationQueueRow, QueueRowState};
        use mold_db::generation_queue_media::{QueueMediaObligation, QueueMediaObligationState};
        use std::sync::Arc;
        use tower::ServiceExt;
        let home = tempfile::tempdir().unwrap();
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(home.path()),
            "original-test",
        ));
        let owner = journal.owner_uuid().unwrap().to_string();
        let lifecycle = Arc::new(crate::queue_media_lifecycle::QueueMediaLifecycle::new(
            db.clone(),
            home.path().into(),
            owner.clone(),
        ));
        journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .unwrap();
        assert!(
            crate::queue_media_startup::reconcile_claimed_owner(&journal, lifecycle.as_ref())
                .unwrap()
                .durable_media_ready
        );
        let encode = |width| {
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::DynamicImage::new_rgb8(width, 900)
                .write_to(&mut buffer, image::ImageFormat::Png)
                .unwrap();
            buffer.into_inner()
        };
        let first = encode(1400);
        let second = encode(1500);
        let keyframe = serde_json::to_vec(&mold_core::KeyframeCondition {
            frame: 42,
            image: second.clone(),
            name: None,
        })
        .unwrap();
        let wav = b"RIFF1234WAVEdata".to_vec();
        let video = b"\x00\x00\x00\x18ftypisom".to_vec();
        let store = QueueMediaStore::open(home.path()).unwrap().store;
        let set = store
            .seal_v2_with_operation_fingerprint(
                &owner,
                "job",
                &crate::queue_media_store::QueueMediaOperationFingerprint::sha256_v1(
                    b"original-test",
                ),
                &crate::queue_media_store::QueueMediaProjection::default(),
                vec![
                    SealMedia::bytes("edit_images", "item:0", first.clone()),
                    SealMedia::bytes("edit_images", "item:1", second.clone()),
                    SealMedia::bytes("keyframes", "item:0", keyframe),
                    SealMedia::bytes("audio_file", "scalar", wav.clone()),
                    SealMedia::bytes("source_video", "scalar", video.clone()),
                    SealMedia::bytes("source_image_name", "scalar", b"private.png".to_vec()),
                    SealMedia::bytes("references", "item:0", b"unsupported".to_vec()),
                ],
            )
            .unwrap();
        let row = GenerationQueueRow {
            id: "job".into(),
            owner_uuid: owner.clone(),
            state: QueueRowState::Queued,
            model: "test".into(),
            request_json: r#"{"references":[{"kind":"mesh"}]}"#.into(),
            output_dir: "/gallery".into(),
            target_gpu: None,
            target_device_id: None,
            completion_payload: "metadata_only".into(),
            seed_pinned: false,
            dispatch_attempts: 0,
            replay_seen: 0,
            held_reason: None,
            created_at_ms: 1,
            updated_at_ms: 1,
            started_at_ms: None,
            media_set_id: Some(set.set_id.clone()),
            admission_authority: None,
        };
        generation_queue::insert_with_media(
            db.as_ref().as_ref().unwrap(),
            &row,
            &QueueMediaObligation {
                media_set_id: set.set_id,
                owner_uuid: owner,
                state: QueueMediaObligationState::Active,
                created_at_ms: 1,
                updated_at_ms: 1,
            },
        )
        .unwrap();
        let mut state = AppState::for_tests();
        state.queue_journal = journal;
        state.metadata_db = db.clone();
        let app = Router::new()
            .route("/api/queue/:id/input-original", get(original))
            .with_state(state);
        for (index, expected, mime) in [
            (0, first, "image/png"),
            (1, second.clone(), "image/png"),
            (2, second, "image/png"),
            (3, wav, "audio/wav"),
            (4, video, "video/mp4"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::get(format!("/api/queue/job/input-original?index={index}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            if response.status() != StatusCode::OK {
                panic!(
                    "input {index}: {:?}",
                    to_bytes(response.into_body(), 4096).await.unwrap()
                );
            }
            assert_eq!(response.headers()[header::CONTENT_TYPE], mime);
            assert_eq!(
                response.headers()[header::CACHE_CONTROL],
                "private, no-store"
            );
            assert_eq!(
                to_bytes(response.into_body(), MAX_SOURCE_BYTES)
                    .await
                    .unwrap()
                    .as_ref(),
                expected
            );
        }
        for query in ["", "?index=-1", "?index=bad"] {
            let response = app
                .clone()
                .oneshot(
                    Request::get(format!("/api/queue/job/input-original{query}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        for index in [5, 6, 99] {
            let response = app
                .clone()
                .oneshot(
                    Request::get(format!("/api/queue/job/input-original?index={index}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
        }
        generation_queue::delete(db.as_ref().as_ref().unwrap(), "job").unwrap();
        let response = app
            .oneshot(
                Request::get("/api/queue/job/input-original?index=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn original_query_requires_an_exact_explicit_index() {
        assert!(serde_json::from_str::<OriginalQuery>(r#"{}"#).is_err());
        for value in [r#"{"index":-1}"#, r#"{"index":"bad"}"#] {
            assert!(serde_json::from_str::<OriginalQuery>(value).is_err());
        }
        assert_eq!(
            serde_json::from_str::<OriginalQuery>(r#"{"index":2}"#)
                .unwrap()
                .index,
            2
        );
    }

    #[test]
    fn submitted_original_refuses_oversized_and_empty_content() {
        assert!(original_content("source_image", vec![0; MAX_SOURCE_BYTES + 1]).is_err());
        assert!(original_content("source_image", Vec::new()).is_err());
        assert!(original_content(
            "keyframes",
            br#"{"frame":0,"image":[],"name":null}"#.to_vec()
        )
        .is_err());
    }

    #[test]
    fn submitted_original_keeps_full_resolution_and_extracts_keyframes() {
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(1400, 900)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let encoded = encoded.into_inner();
        let (mime, bytes) = original_content("source_image", encoded.clone()).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, encoded);
        assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 1400);
        let frame = mold_core::KeyframeCondition {
            frame: 19,
            image: encoded.clone(),
            name: None,
        };
        assert_eq!(
            original_content("keyframes", serde_json::to_vec(&frame).unwrap())
                .unwrap()
                .1,
            encoded
        );
        assert!(original_content("keyframes", b"broken".to_vec()).is_err());
        assert!(original_content("source_image", b"private.png".to_vec()).is_err());
        assert!(original_content("source_video", b"\x00\x00\x00\x18ftypheic".to_vec()).is_err());
        assert!(original_content("audio_file", b"RIFF1234WAVEdata".to_vec()).is_ok());
        assert!(original_content("source_video", b"\x00\x00\x00\x18ftypisom".to_vec()).is_ok());
    }

    #[test]
    fn ordered_upload_backed_references_keep_kind_and_named_view_and_size_limits() {
        use crate::queue_media_store::{
            MediaManifestEntry, MediaSetManifest, MediaSetRef, QueueMediaSink,
        };
        let entry = |role: &str, name: &str, size| MediaManifestEntry {
            content_sha256: None,
            role: role.into(),
            name: name.into(),
            size_bytes: size,
            sha256_hex: "digest".into(),
            sink: QueueMediaSink::PrivateStaging,
        };
        let manifest = MediaSetManifest {
            media_set: MediaSetRef {
                owner_id: "owner".into(),
                job_id: "job".into(),
                set_id: "set".into(),
            },
            operation_fingerprint: None,
            entries: vec![
                entry("references", "collection", 0),
                entry("references", "item:0", 12),
                entry("references", "item:1", 12),
                entry("references", "item:2", 12),
                entry("edit_images", "item:0", MAX_SOURCE_BYTES as u64 + 1),
                entry("identity_image_name", "scalar", 12),
            ],
        };
        let result = inputs(
            &manifest,
            &serde_json::json!({"references": [{"kind": "named_image", "role": "front"}, {"kind": "audio"}, {"kind": "image"}]}),
        );
        assert_eq!(
            result.iter().map(|i| i.index).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(result[0].label, "Reference 1 · named image · front");
        assert!(result[0].preview);
        assert!(!result[1].preview);
        assert!(result[2].preview);
        assert!(!result[3].preview);
    }

    #[test]
    fn every_image_role_is_discovered_without_model_allowlists() {
        for role in [
            "source_image",
            "identity_image",
            "identity_images",
            "edit_images",
            "mask_image",
            "control_image",
            "references",
            "keyframes",
        ] {
            assert!(input_label(role, "item:0").is_some(), "{role}");
        }
        for role in ["source_image_name", "identity_image_names", "lora", "loras"] {
            assert!(input_label(role, "scalar").is_none(), "{role}");
        }
        assert_eq!(
            input_label("edit_images", "item:1").unwrap(),
            "Reference image 2"
        );
        assert!(input_label("edit_images", "collection").is_none());
    }

    #[test]
    fn boundary_frame_preview_extracts_pixels_from_the_sealed_record() {
        let frame = mold_core::KeyframeCondition {
            frame: 12,
            image: vec![1, 2, 3],
            name: None,
        };
        let bytes = serde_json::to_vec(&frame).unwrap();
        assert_eq!(preview_pixels("keyframes", bytes).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn thumbnail_is_bounded_and_never_returns_original_bytes() {
        let original = image::DynamicImage::new_rgb8(1400, 900);
        let mut source = std::io::Cursor::new(Vec::new());
        original
            .write_to(&mut source, image::ImageFormat::Png)
            .unwrap();
        let thumbnail = render(&source.into_inner()).unwrap();
        let decoded = image::load_from_memory(&thumbnail).unwrap();
        assert_eq!(decoded.width(), 320);
        assert!(decoded.height() <= 320);
    }
    #[test]
    fn non_image_authority_is_not_a_preview() {
        assert!(render(b"private-file-name.png").is_err());
    }
}
