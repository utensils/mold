//! Lease-bound durable queue-media hydration.
//!
//! The scheduler may inspect only the authenticated, payload-free projection.
//! The encrypted bundle remains opaque until a worker owns its only execution
//! slot (or a concrete GPU device lease), at which point this handle overlays
//! authenticated media onto the already-mutated runtime request.

use std::fmt;
use std::sync::Arc;

use crate::queue_media_store::{
    DecryptedQueueMediaSet, MediaSetRef, QueueMediaProjection, QueueMediaStore,
};

#[derive(Clone)]
pub struct DeferredQueueMedia {
    store: Arc<QueueMediaStore>,
    media_set: MediaSetRef,
    projection: QueueMediaProjection,
}

impl fmt::Debug for DeferredQueueMedia {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeferredQueueMedia")
            .field("projection", &self.projection)
            .finish_non_exhaustive()
    }
}

impl DeferredQueueMedia {
    pub fn new(
        store: Arc<QueueMediaStore>,
        media_set: MediaSetRef,
        projection: QueueMediaProjection,
    ) -> Self {
        Self {
            store,
            media_set,
            projection,
        }
    }

    pub fn projection(&self) -> &QueueMediaProjection {
        &self.projection
    }

    pub fn media_set_ref(&self) -> &MediaSetRef {
        &self.media_set
    }

    /// Authenticate and decrypt the complete bundle, then overlay it onto the
    /// request without changing scheduler-mutated non-media fields.
    pub fn hydrate_into(
        &self,
        expected_job_id: &str,
        request: &mut mold_core::GenerateRequest,
    ) -> Result<HydratedQueueMediaLease, DeferredQueueMediaError> {
        if self.media_set.job_id != expected_job_id {
            return Err(DeferredQueueMediaError::hold(
                "queue-media job identity does not match the runtime job",
            ));
        }
        let mut decrypted = self
            .store
            .decrypt_mixed(&self.media_set)
            .map_err(DeferredQueueMediaError::from_store)?;
        let media =
            crate::queue_media::decrypted_media_into_opaque(expected_job_id, &mut decrypted)
                .map_err(DeferredQueueMediaError::from_media)?;
        crate::queue_media::rehydrate_request_media_into(expected_job_id, request, media)
            .map_err(DeferredQueueMediaError::from_media)?;
        Ok(HydratedQueueMediaLease { decrypted })
    }
}

/// Owns every private staged path until the generation attempt finishes.
/// Dropping it removes the private staging tree; memory-only bytes remain on
/// the request and are never materialized to a filesystem path.
pub struct HydratedQueueMediaLease {
    #[allow(dead_code)]
    decrypted: DecryptedQueueMediaSet,
}

impl fmt::Debug for HydratedQueueMediaLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HydratedQueueMediaLease")
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredHydrationDisposition {
    /// Authenticated state is permanently invalid or violates its authority
    /// contract. Preserve it for operator review instead of replaying it.
    Hold,
    /// The store could not be accessed safely now. Retain it for replay.
    Retain,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct DeferredQueueMediaError {
    disposition: DeferredHydrationDisposition,
    message: String,
}

impl DeferredQueueMediaError {
    fn hold(message: impl Into<String>) -> Self {
        Self {
            disposition: DeferredHydrationDisposition::Hold,
            message: message.into(),
        }
    }

    fn retain(message: impl Into<String>) -> Self {
        Self {
            disposition: DeferredHydrationDisposition::Retain,
            message: message.into(),
        }
    }

    pub(crate) fn worker_failure() -> Self {
        Self::retain("durable queue-media hydration worker did not complete safely")
    }

    fn from_store(error: crate::queue_media_store::QueueMediaError) -> Self {
        use crate::queue_media_store::QueueMediaError;

        match error {
            QueueMediaError::Io(_) | QueueMediaError::SecurityUnavailable(_) => Self::retain(
                format!("durable queue media is temporarily unavailable: {error}"),
            ),
            _ => Self::hold(format!(
                "durable queue media failed authentication or structural validation: {error}"
            )),
        }
    }

    fn from_media(error: crate::queue_media::QueueMediaError) -> Self {
        Self::hold(format!(
            "durable queue media does not match its request authority: {error}"
        ))
    }

    pub fn disposition(&self) -> DeferredHydrationDisposition {
        self.disposition
    }

    pub fn public_message(&self) -> &'static str {
        match self.disposition {
            DeferredHydrationDisposition::Hold => {
                "durable queue media could not be authenticated; the job is held for review"
            }
            DeferredHydrationDisposition::Retain => {
                "durable queue media is temporarily unavailable; the job was retained for replay"
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::queue_media::{
        extract_request_media, into_seal_media, project_request_media, ProcessPrivateAuthorities,
    };
    use crate::queue_media_store::QueueMediaOperationFingerprint;

    fn request(path: &std::path::Path) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "original prompt",
            "model": "flux-dev-pulid",
            "width": 512,
            "height": 512,
            "steps": 8,
            "guidance": 3.0,
            "seed": 11,
            "source_image": "c291cmNlLWJ5dGVz",
            "id_image": "aWRlbnRpdHktYnl0ZXM=",
            "id_weight": 0.8,
            "true_cfg": 2.0,
            "edit_images": ["bm90LWFuLWltYWdl"],
            "mask_image": "bWFzay1ieXRlcw==",
            "control_image": "Y29udHJvbC1ieXRlcw==",
            "source_video_path": path.to_string_lossy()
        }))
        .unwrap()
    }

    #[test]
    fn leased_hydration_matches_projection_and_private_paths_are_raii_owned() {
        let home = tempfile::tempdir().unwrap();
        let source_path = home.path().join("input.mp4");
        std::fs::write(&source_path, b"private-video-bytes").unwrap();
        let store = Arc::new(QueueMediaStore::open(home.path()).unwrap().store);
        let extracted = extract_request_media(
            "job-runtime",
            request(&source_path),
            &ProcessPrivateAuthorities::none(),
        )
        .unwrap();
        let projection = project_request_media(extracted.media()).unwrap();
        let (request_json, opaque_media) = extracted.into_parts();
        let mut sanitized: mold_core::GenerateRequest =
            serde_json::from_str(&request_json).unwrap();
        let media = into_seal_media(opaque_media).unwrap();
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner-runtime",
                "job-runtime",
                &QueueMediaOperationFingerprint::sha256_v1(b"runtime operation"),
                &projection,
                &media,
            )
            .unwrap();
        let deferred = DeferredQueueMedia::new(store, reference, projection.clone());

        // Scheduler mutations survive the media overlay.
        sanitized.prompt = "scheduler-mutated prompt".into();
        sanitized.seed = Some(22);
        let lease = deferred
            .hydrate_into("job-runtime", &mut sanitized)
            .unwrap();
        assert_eq!(sanitized.prompt, "scheduler-mutated prompt");
        assert_eq!(sanitized.seed, Some(22));
        assert_eq!(
            sanitized.id_image.as_deref(),
            Some(b"identity-bytes".as_slice())
        );
        let private_path = std::path::PathBuf::from(sanitized.source_video_path.as_ref().unwrap());
        let runtime_root = private_path
            .parent()
            .and_then(std::path::Path::parent)
            .unwrap()
            .to_path_buf();
        drop(deferred);
        assert!(runtime_root.is_dir());
        assert_eq!(
            std::fs::read(&private_path).unwrap(),
            b"private-video-bytes"
        );

        let hydrated = extract_request_media(
            "job-runtime",
            sanitized.clone(),
            &ProcessPrivateAuthorities::none(),
        )
        .unwrap();
        assert_eq!(project_request_media(hydrated.media()).unwrap(), projection);

        drop(lease);
        assert!(!private_path.exists());
        assert!(!runtime_root.exists());
    }

    #[test]
    fn authentication_holds_while_transient_store_access_retains() {
        assert_eq!(
            DeferredQueueMediaError::from_store(
                crate::queue_media_store::QueueMediaError::Authentication,
            )
            .disposition(),
            DeferredHydrationDisposition::Hold,
        );
        assert_eq!(
            DeferredQueueMediaError::from_store(crate::queue_media_store::QueueMediaError::Io(
                std::io::Error::new(std::io::ErrorKind::Interrupted, "temporary"),
            ))
            .disposition(),
            DeferredHydrationDisposition::Retain,
        );
        assert_eq!(
            DeferredQueueMediaError::from_store(
                crate::queue_media_store::QueueMediaError::SecurityUnavailable("temporary".into()),
            )
            .disposition(),
            DeferredHydrationDisposition::Retain,
        );
    }
}
