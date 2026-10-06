//! Authenticated previews of retained conditioning, never provenance paths.
use crate::{routes::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};

#[utoipa::path(get, path = "/api/gallery/source-media/{filename}/{member_id}/thumbnail", tag = "gallery",
    params(("filename" = String, Path, description = "Exact gallery filename"),
           ("member_id" = String, Path, description = "Opaque retained member id")),
    responses((status = 200, description = "Private 320-pixel image preview", content_type = "image/png"),
              (status = 401, description = "API-key authentication required"),
              (status = 404, description = "Image preview unavailable")))]
pub(crate) async fn get(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
    Path((filename, requested_member)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    validate_preview_request(auth.as_ref(), authenticated.as_ref(), &filename)?;
    static SLOTS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
        std::sync::OnceLock::new();
    let permit = SLOTS
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("thumbnail unavailable"))?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let _permit = permit;
        let members = crate::gallery_source_media::resolve_members(&state, &filename)?
            .ok_or_else(|| ApiError::not_found("retained source unavailable"))?;
        let member = members
            .members
            .into_iter()
            .find(|m| m.member.member_id == requested_member)
            .ok_or_else(|| ApiError::not_found("retained member unavailable"))?;
        if member.member.size_bytes > 64 * 1024 * 1024 {
            return Err(ApiError::not_found("retained image exceeds preview limit"));
        }
        let lifecycle = state
            .queue_journal
            .queue_media_lifecycle()
            .ok_or_else(|| ApiError::not_found("retained media unavailable"))?;
        let bytes = lifecycle
            .gallery_member_bytes(member.media_set, member.pin_id, member.index)
            .map_err(|_| ApiError::not_found("retained media unavailable"))?;
        let bytes = zeroize::Zeroizing::new(bytes);
        crate::queue_input_thumbnail::render(&bytes)
            .map_err(|_| ApiError::not_found("retained member has no image preview"))
    })
    .await
    .map_err(|_| ApiError::internal("thumbnail task failed"))??;
    Ok((
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "private, no-store"),
        ],
        bytes,
    )
        .into_response())
}

fn validate_preview_request(
    auth: Option<&Extension<crate::auth::AuthState>>,
    authenticated: Option<&Extension<crate::auth::ApiKeyAuthenticated>>,
    filename: &str,
) -> Result<(), ApiError> {
    if !crate::gallery_source_media::private_media_authorized(auth, authenticated) {
        return Err(ApiError::with_code(
            "retained source media requires API-key authentication",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    crate::routes::validate_gallery_filename(filename)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyed_thumbnail_rejects_an_unauthenticated_caller_before_lookup() {
        let auth = Extension(Some(std::sync::Arc::new(crate::auth::ApiKeySet::new(
            std::collections::HashSet::from(["fixture-key".to_string()]),
        ))));
        let result = validate_preview_request(Some(&auth), None, "fixture.png");
        assert_eq!(
            result.err().unwrap().into_response().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[test]
    fn preview_never_resolves_a_path_outside_the_gallery() {
        let result = validate_preview_request(None, None, "../fixture.png");
        assert_eq!(
            result.err().unwrap().into_response().status(),
            StatusCode::BAD_REQUEST
        );
    }
}
