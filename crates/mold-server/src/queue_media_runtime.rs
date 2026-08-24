//! Lease-bound durable queue-media hydration.
//!
//! The scheduler may inspect only the authenticated, payload-free projection.
//! The encrypted bundle remains opaque until a worker owns its only execution
//! slot (or a concrete GPU device lease), at which point this handle overlays
//! authenticated media onto the already-mutated runtime request.

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use zeroize::Zeroize;

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

/// Attempt-scoped access to a request whose durable media has been hydrated.
///
/// The guard borrows the authoritative request, owns its private staging
/// lease, and scrubs every overlaid media field before the staging tree is
/// released. Workers must route all post-hydration request access through this
/// value; any owned request clone must use [`Self::zeroizing_clone`].
pub struct HydratedQueueMediaRequest<'a> {
    request: &'a mut mold_core::GenerateRequest,
    lease: HydratedQueueMediaLease,
}

/// Uniform read-only request access for worker code after the hydration point.
/// Only the hydrated arm owns a scrub/staging guard; the plain arm preserves
/// legacy non-durable behavior exactly.
pub enum AttemptQueueMediaRequest<'a> {
    Plain(&'a mold_core::GenerateRequest),
    Hydrated(HydratedQueueMediaRequest<'a>),
}

impl<'a> AttemptQueueMediaRequest<'a> {
    pub fn plain(request: &'a mold_core::GenerateRequest) -> Self {
        Self::Plain(request)
    }

    pub fn hydrated(
        request: &'a mut mold_core::GenerateRequest,
        lease: HydratedQueueMediaLease,
    ) -> Self {
        Self::Hydrated(HydratedQueueMediaRequest::new(request, lease))
    }

    pub fn is_hydrated(&self) -> bool {
        matches!(self, Self::Hydrated(_))
    }

    pub fn zeroizing_clone(&self) -> ZeroizingGenerateRequest {
        match self {
            Self::Plain(request) => ZeroizingGenerateRequest {
                request: (*request).clone(),
                #[cfg(test)]
                scrub_probe: None,
            },
            Self::Hydrated(request) => request.zeroizing_clone(),
        }
    }

    pub fn output_metadata(
        &self,
        seed: u64,
        scheduler: Option<mold_core::Scheduler>,
        version: impl Into<String>,
    ) -> mold_core::OutputMetadata {
        match self {
            Self::Plain(request) => {
                mold_core::OutputMetadata::from_generate_request(request, seed, scheduler, version)
            }
            Self::Hydrated(request) => request.output_metadata(seed, scheduler, version),
        }
    }

    /// Remove process-private staging roots from any diagnostic that crosses
    /// into logs, SSE, or a client-visible result. Non-durable requests retain
    /// their existing diagnostics byte-for-byte.
    pub fn redact_staging_paths(&self, message: impl Into<String>) -> String {
        let mut message = message.into();
        let Self::Hydrated(request) = self else {
            return message;
        };
        for path in [
            request.audio_file_path.as_deref(),
            request.source_video_path.as_deref(),
            request.extend_video_path.as_deref(),
            request.hdr_exr_dir.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            let path = std::path::Path::new(path);
            let runtime_root = path.ancestors().find(|ancestor| {
                ancestor
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("runtime-"))
            });
            let sensitive = runtime_root.unwrap_or(path).to_string_lossy();
            if !sensitive.is_empty() {
                message = message.replace(sensitive.as_ref(), "<private-staging>");
            }
        }
        message
    }
}

impl Deref for AttemptQueueMediaRequest<'_> {
    type Target = mold_core::GenerateRequest;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Plain(request) => request,
            Self::Hydrated(request) => request,
        }
    }
}

impl<'a> HydratedQueueMediaRequest<'a> {
    pub fn new(
        request: &'a mut mold_core::GenerateRequest,
        lease: HydratedQueueMediaLease,
    ) -> Self {
        Self { request, lease }
    }

    pub fn zeroizing_clone(&self) -> ZeroizingGenerateRequest {
        ZeroizingGenerateRequest {
            request: self.request.clone(),
            #[cfg(test)]
            scrub_probe: None,
        }
    }

    /// Derive normal output provenance while ensuring process-private staging
    /// paths never enter gallery metadata or completion events.
    pub fn output_metadata(
        &self,
        seed: u64,
        scheduler: Option<mold_core::Scheduler>,
        version: impl Into<String>,
    ) -> mold_core::OutputMetadata {
        let mut request = self.zeroizing_clone();
        let mut metadata =
            mold_core::OutputMetadata::from_generate_request(&request, seed, scheduler, version);
        scrub_metadata_path(&mut metadata.audio_file_path);
        scrub_metadata_path(&mut metadata.source_video_path);
        scrub_metadata_path(&mut metadata.extend_video_path);
        scrub_metadata_path(&mut metadata.hdr_exr_dir);
        // Drop the clone before returning so its media buffers do not survive
        // alongside the intentionally payload-free metadata.
        crate::queue_media::scrub_request_media(&mut request);
        metadata
    }
}

impl Deref for HydratedQueueMediaRequest<'_> {
    type Target = mold_core::GenerateRequest;

    fn deref(&self) -> &Self::Target {
        self.request
    }
}

impl Drop for HydratedQueueMediaRequest<'_> {
    fn drop(&mut self) {
        crate::queue_media::scrub_request_media(self.request);
        // `lease` is deliberately released by field drop only after this Drop
        // body has wiped the staged path strings from the request.
        let _ = &self.lease;
    }
}

/// An owned request copy whose durable media is wiped on every downstream
/// success, error, cancellation, panic unwind, or worker join failure.
pub struct ZeroizingGenerateRequest {
    request: mold_core::GenerateRequest,
    #[cfg(test)]
    scrub_probe: Option<Arc<std::sync::atomic::AtomicBool>>,
}

#[cfg(test)]
impl ZeroizingGenerateRequest {
    fn with_scrub_probe(mut self, scrubbed: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.scrub_probe = Some(scrubbed);
        self
    }
}

impl Deref for ZeroizingGenerateRequest {
    type Target = mold_core::GenerateRequest;

    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

impl DerefMut for ZeroizingGenerateRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.request
    }
}

impl Drop for ZeroizingGenerateRequest {
    fn drop(&mut self) {
        crate::queue_media::scrub_request_media(&mut self.request);
        #[cfg(test)]
        if let Some(scrubbed) = &self.scrub_probe {
            scrubbed.store(
                request_media_is_cleared(&self.request),
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }
}

#[cfg(test)]
fn request_media_is_cleared(request: &mold_core::GenerateRequest) -> bool {
    request.source_image.is_none()
        && request.source_image_name.is_none()
        && request.id_image.is_none()
        && request.id_image_name.is_none()
        && request.id_images.is_none()
        && request.id_image_names.is_none()
        && request.edit_images.is_none()
        && request.references.is_none()
        && request.mask_image.is_none()
        && request.control_image.is_none()
        && request.audio_file.is_none()
        && request.audio_file_path.is_none()
        && request.source_video.is_none()
        && request.source_video_path.is_none()
        && request.extend_video.is_none()
        && request.extend_video_path.is_none()
        && request.keyframes.is_none()
        && request.hdr_exr_dir.is_none()
        && request.lora.is_none()
        && request.loras.is_none()
}

fn scrub_metadata_path(path: &mut Option<String>) {
    if let Some(path) = path {
        path.zeroize();
    }
    *path = None;
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
                media,
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
    fn hydrated_request_guard_scrubs_normal_clone_and_panic_paths_and_redacts_metadata() {
        let home = tempfile::tempdir().unwrap();
        let source_path = home.path().join("input.mp4");
        std::fs::write(&source_path, b"private-video-bytes").unwrap();
        let store = Arc::new(QueueMediaStore::open(home.path()).unwrap().store);
        let extracted = extract_request_media(
            "job-guard",
            request(&source_path),
            &ProcessPrivateAuthorities::none(),
        )
        .unwrap();
        let projection = project_request_media(extracted.media()).unwrap();
        let (request_json, opaque_media) = extracted.into_parts();
        let media = into_seal_media(opaque_media).unwrap();
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner-guard",
                "job-guard",
                &QueueMediaOperationFingerprint::sha256_v1(b"guard operation"),
                &projection,
                media,
            )
            .unwrap();
        let deferred = DeferredQueueMedia::new(store, reference, projection);
        let mut sanitized: mold_core::GenerateRequest =
            serde_json::from_str(&request_json).unwrap();

        let lease = deferred.hydrate_into("job-guard", &mut sanitized).unwrap();
        let private_path = std::path::PathBuf::from(sanitized.source_video_path.as_ref().unwrap());
        let runtime_root = private_path
            .ancestors()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("runtime-"))
            })
            .unwrap()
            .to_path_buf();
        let guard = AttemptQueueMediaRequest::hydrated(&mut sanitized, lease);
        let metadata = guard.output_metadata(11, None, "test");
        assert!(metadata.source_image_sha256.is_some());
        assert!(metadata.id_image_sha256.is_some());
        assert!(metadata.source_video_path.is_none());
        assert!(metadata.audio_file_path.is_none());
        assert!(metadata.extend_video_path.is_none());
        assert!(metadata.hdr_exr_dir.is_none());
        let diagnostic = guard.redact_staging_paths(format!(
            "decoder rejected {} beneath {}",
            private_path.display(),
            runtime_root.display()
        ));
        assert!(!diagnostic.contains(runtime_root.to_string_lossy().as_ref()));
        assert!(diagnostic.contains("<private-staging>"));

        let clone_scrubbed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let clone = guard
            .zeroizing_clone()
            .with_scrub_probe(Arc::clone(&clone_scrubbed));
        drop(clone);
        assert!(clone_scrubbed.load(std::sync::atomic::Ordering::SeqCst));
        assert!(private_path.is_file());
        drop(guard);
        assert!(request_media_is_cleared(&sanitized));
        assert!(!private_path.exists());
        assert!(runtime_root.is_dir());

        let lease = deferred.hydrate_into("job-guard", &mut sanitized).unwrap();
        let panic_path = std::path::PathBuf::from(sanitized.source_video_path.as_ref().unwrap());
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = AttemptQueueMediaRequest::hydrated(&mut sanitized, lease);
            assert!(guard.source_image.is_some());
            panic!("injected worker panic after hydration");
        }));
        assert!(panicked.is_err());
        assert!(request_media_is_cleared(&sanitized));
        assert!(!panic_path.exists());
        drop(deferred);
        assert!(!runtime_root.exists());
    }

    #[test]
    fn plain_attempt_metadata_preserves_non_durable_paths() {
        let request = request(std::path::Path::new("/user/media/source.mp4"));
        let attempt = AttemptQueueMediaRequest::plain(&request);
        let metadata = attempt.output_metadata(11, None, "test");
        assert_eq!(
            metadata.source_video_path.as_deref(),
            Some("/user/media/source.mp4")
        );
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
