//! Read-only, batched retained-authority evidence for incremental library mirrors.
use crate::{routes::ApiError, state::AppState};
use axum::{
    extract::{Extension, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    Json,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Serialize)]
pub(crate) struct Checkpoint {
    protocol_version: u32,
    instance_id: String,
    revisions: BTreeMap<String, String>,
}

fn revision(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) async fn checkpoint(
    State(state): State<AppState>,
    auth: Option<Extension<crate::auth::AuthState>>,
    authenticated: Option<Extension<crate::auth::ApiKeyAuthenticated>>,
) -> Result<(HeaderMap, Json<Checkpoint>), ApiError> {
    if !crate::gallery_source_media::private_media_authorized(auth.as_ref(), authenticated.as_ref())
    {
        return Err(ApiError::with_code(
            "API-key authentication required",
            "RETAINED_SOURCE_MEDIA_AUTH_REQUIRED",
            StatusCode::UNAUTHORIZED,
        ));
    }
    let _publication = state.gallery_publication_gate.read().await;
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(ApiError::not_found("image output is disabled"));
    }
    let root = config.effective_output_dir();
    drop(config);
    let instance_id = state.instance_id.to_string();
    let revisions = tokio::task::spawn_blocking(move || {
        let index = state
            .gallery_publication_gate
            .committed_archive_index(&root)
            .map_err(|e| ApiError::internal(format!("gallery authority read failed: {e:#}")))?;
        let mut revisions = BTreeMap::new();
        for (name, entry) in &index.entries {
            if index.get(name).is_none() {
                continue;
            }
            // Cache proof requires unchanged regular-file facts. Never hash
            // output payloads here: old entries without facts and changed or
            // missing files fall back to exact-output transfer verification.
            let Some(facts) = entry.facts.as_ref() else {
                continue;
            };
            if crate::batch_transaction::ArchiveFileFacts::from_path(&root.join(name))
                .ok()
                .as_ref()
                != Some(facts)
            {
                continue;
            }
            let Some(media) = crate::gallery_source_media::resolve_members_from_pins(
                &state,
                entry.identity.clone(),
                entry.retained_media.clone(),
            )?
            else {
                continue;
            };
            // Missing/corrupt manifests never establish cache trust. Member facts
            // capture late retained repairs independently of output media_version.
            if media.corrupt {
                continue;
            }
            let members: Vec<_> = media
                .members
                .iter()
                .map(|m| (&m.member, &m.position, &m.sink, &m.sha256))
                .collect();
            let bytes = serde_json::to_vec(&(entry, members, media.legacy)).map_err(|e| {
                ApiError::internal(format!("gallery checkpoint serialization failed: {e}"))
            })?;
            revisions.insert(name.clone(), revision(&bytes));
        }
        Ok::<_, ApiError>(revisions)
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery checkpoint task failed: {e}")))??;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    Ok((
        headers,
        Json(Checkpoint {
            protocol_version: 1,
            instance_id,
            revisions,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn checkpoint_requires_auth_on_keyed_hosts_and_refuses_disabled_output() {
        let (state, _home, gallery) = fixture().await;
        let auth = Some(Extension(Some(std::sync::Arc::new(
            crate::auth::ApiKeySet::new(std::collections::HashSet::from([
                "fixture-key".to_string()
            ])),
        ))));
        assert!(checkpoint(State(state.clone()), auth, None).await.is_err());
        // Authorization refusal cannot initialize or change gallery authority.
        assert_eq!(std::fs::read_dir(gallery.path()).unwrap().count(), 0);
        state.config.write().await.output_dir = Some("".into());
        assert!(checkpoint(State(state), None, None).await.is_err());
    }
    #[tokio::test]
    async fn checkpoint_is_stable_and_read_only_for_existing_output() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "saved.png").await;
        let before = std::fs::metadata(gallery.path().join("saved.png"))
            .unwrap()
            .modified()
            .unwrap();
        let first = checkpoint(State(state.clone()), None, None)
            .await
            .unwrap()
            .1
             .0;
        let second = checkpoint(State(state), None, None).await.unwrap().1 .0;
        assert_eq!(first.revisions, second.revisions);
        assert_eq!(first.revisions.len(), 1);
        assert_eq!(
            before,
            std::fs::metadata(gallery.path().join("saved.png"))
                .unwrap()
                .modified()
                .unwrap()
        );
    }
    #[tokio::test]
    async fn changed_output_facts_are_omitted_without_hashing_output() {
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "saved.png").await;
        assert_eq!(
            checkpoint(State(state.clone()), None, None)
                .await
                .unwrap()
                .1
                 .0
                .revisions
                .len(),
            1
        );
        std::fs::write(gallery.path().join("saved.png"), b"different-output").unwrap();
        assert!(checkpoint(State(state), None, None)
            .await
            .unwrap()
            .1
             .0
            .revisions
            .is_empty());
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn late_inputs_change_revision_and_missing_pin_cannot_establish_trust() {
        use axum::{body::Body, extract::Path};
        let (state, _home, gallery) = fixture().await;
        publish(&state, gallery.path(), "saved.png").await;
        let first = checkpoint(State(state.clone()), None, None)
            .await
            .unwrap()
            .1
             .0;
        let (_, Json(offer)) = crate::gallery_media_transfer::offer(
            State(state.clone()),
            None,
            None,
            Path("saved.png".into()),
        )
        .await
        .unwrap();
        let mut descriptor = serde_json::to_value(offer).unwrap();
        descriptor["members"] = serde_json::json!([{
            "role":"source_image", "position":"scalar", "sink":"memory",
            "size_bytes":3, "sha256":format!("{:x}", Sha256::digest(b"abc"))
        }]);
        let encoded = serde_json::to_vec(&descriptor).unwrap();
        let mut body = (encoded.len() as u32).to_be_bytes().to_vec();
        body.extend(encoded);
        body.extend(b"abc");
        let _ = crate::gallery_media_transfer::receive(
            State(state.clone()),
            None,
            None,
            Path("saved.png".into()),
            Body::from(body),
        )
        .await
        .unwrap();
        let second = checkpoint(State(state.clone()), None, None)
            .await
            .unwrap()
            .1
             .0;
        assert_ne!(first.revisions["saved.png"], second.revisions["saved.png"]);
        let lifecycle = state.queue_journal.queue_media_lifecycle().unwrap();
        let pin = lifecycle
            .runtime_store()
            .unwrap()
            .inspect_gallery_pins()
            .pins
            .pop()
            .unwrap();
        lifecycle
            .release_gallery_pin(pin.media_set, pin.pin_id)
            .unwrap();
        assert!(checkpoint(State(state), None, None)
            .await
            .unwrap()
            .1
             .0
            .revisions
            .is_empty());
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
}
