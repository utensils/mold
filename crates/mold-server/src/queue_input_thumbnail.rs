//! Private previews of the actual conditioning image sealed for a queued job.
use crate::{routes::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::header,
    response::{IntoResponse, Response},
};

const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

#[utoipa::path(get, path = "/api/queue/{id}/input-thumbnail", tag = "queue",
    params(("id" = String, Path, description = "Queue job id")),
    responses((status = 200, description = "Private 320-pixel source image preview", content_type = "image/png"),
              (status = 404, description = "Job has no available sealed source image")))]
pub(crate) async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    static RENDER_SLOTS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
        std::sync::OnceLock::new();
    let permit = RENDER_SLOTS
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("queue thumbnail unavailable"))?;
    // The route lives behind the same API-key middleware as private gallery media.
    // Only journal-owned sealed media may be read: no caller-supplied path is accepted.
    let journal = state.queue_journal.clone();
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let _permit = permit;
        let row = journal
            .row(&id)
            .map_err(|_| ApiError::internal("queue lookup failed"))?
            .ok_or_else(|| ApiError::not_found("queue job not found"))?;
        let set_id = row
            .media_set_id
            .ok_or_else(|| ApiError::not_found("queue input image unavailable"))?;
        let lifecycle = journal
            .queue_media_lifecycle()
            .ok_or_else(|| ApiError::not_found("queue input image unavailable"))?;
        let media_set = crate::queue_media_store::MediaSetRef {
            owner_id: row.owner_uuid,
            job_id: row.id,
            set_id,
        };
        let bytes = lifecycle
            .input_image_bytes(media_set)
            .map_err(|_| ApiError::not_found("queue input image unavailable"))?;
        let bytes = zeroize::Zeroizing::new(bytes);
        render(&bytes).map_err(|_| ApiError::not_found("queue input image unavailable"))
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
