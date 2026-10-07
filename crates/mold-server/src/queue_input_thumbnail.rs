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
    #[test]
    fn ordered_upload_backed_references_keep_kind_and_named_view_and_size_limits() {
        use crate::queue_media_store::{
            MediaManifestEntry, MediaSetManifest, MediaSetRef, QueueMediaSink,
        };
        let entry = |role: &str, name: &str, size| MediaManifestEntry {
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
