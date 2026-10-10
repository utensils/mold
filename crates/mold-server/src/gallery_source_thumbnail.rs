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
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let resolved = crate::gallery_source_media::resolve_members(&state, &filename)?
            .ok_or_else(|| ApiError::not_found("retained source unavailable"))?;
        let member = resolved
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
        let store = lifecycle
            .runtime_store()
            .map_err(|_| ApiError::not_found("retained media unavailable"))?;
        let pin = crate::queue_media_store::GalleryMediaPinRef::new(
            member.media_set.clone(),
            member.pin_id.clone(),
        )
        .map_err(|_| ApiError::not_found("retained media unavailable"))?;
        let version = store
            .gallery_pin_version(&pin)
            .map_err(|_| ApiError::not_found("retained media unavailable"))?;
        let key = version.clone().map(|version| (version, member.index));
        let flight = key.as_ref().map(|key| store.gallery_thumbnails.flight(key));
        let _single = flight
            .as_ref()
            .map(|flight| flight.lock().unwrap_or_else(|p| p.into_inner()));
        let cached = key
            .as_ref()
            .and_then(|key| store.gallery_thumbnails.get(key));
        let preview = if let Some(cached) = cached {
            cached
        } else {
            with_preview_payload(
                &store.gallery_preview_payloads,
                || -> Result<_, ApiError> {
                    let bytes = store
                        .read_gallery_member(&pin, member.index, 64 * 1024 * 1024)
                        .map_err(|_| ApiError::not_found("retained media unavailable"))?;
                    let bytes = zeroize::Zeroizing::new(
                        crate::gallery_source_media::media_content(&member.member.role, bytes)?,
                    );
                    // This is separate from the selected-read permit: never nest the gate.
                    let _permit = store.gallery_cold_work.enter();
                    #[cfg(test)]
                    store
                        .gallery_thumbnail_renders
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let preview = crate::queue_input_thumbnail::render(&bytes)
                        .map_err(|_| ApiError::not_found("retained member has no image preview"))?;
                    Ok(std::sync::Arc::new(zeroize::Zeroizing::new(preview)))
                },
            )?
        };
        // A queued/decode operation may outlive its owner. Cache hits are subject
        // to the same fresh publication and exact-pin check as freshly rendered bytes.
        let current = crate::gallery_source_media::resolve_members(&state, &filename)?
            .ok_or_else(|| ApiError::not_found("retained source unavailable"))?;
        if current.archive_identity_sha256 != resolved.archive_identity_sha256
            || !current
                .members
                .iter()
                .any(|m| m.member.member_id == requested_member && m.sha256 == member.sha256)
            || store
                .gallery_pin_version(&pin)
                .map_err(|_| ApiError::not_found("retained media unavailable"))?
                != version
        {
            return Err(ApiError::not_found("retained source changed"));
        }
        if let Some(key) = key {
            store
                .gallery_thumbnails
                .insert(key, preview.clone(), preview.len());
        }
        Ok(preview.as_slice().to_vec())
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

fn with_preview_payload<T>(
    gate: &crate::gallery_source_cache::ColdWork,
    work: impl FnOnce() -> T,
) -> T {
    // Hold capacity before allocating selected plaintext until rendering drops it.
    // This gate is distinct from the shared scan/render gate to avoid nesting it.
    let _payload_lifetime = gate.enter();
    work()
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
    fn gallery_preview_payload_lifetimes_are_bounded_through_render_waits() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let gate = crate::gallery_source_cache::ColdWork::new(2);
        let payloads = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..12 {
                scope.spawn(|| {
                    with_preview_payload(&gate, || {
                        // Selected bytes remain live while the renderer waits.
                        let bytes = zeroize::Zeroizing::new(vec![0u8; 1024]);
                        let count = payloads.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        drop(bytes);
                        payloads.fetch_sub(1, Ordering::SeqCst);
                    })
                });
            }
        });
        assert!(peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(payloads.load(Ordering::SeqCst), 0);
    }

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
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }
}
