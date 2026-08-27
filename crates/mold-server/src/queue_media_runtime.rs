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
        let reference_paths =
            crate::queue_media::rehydrate_request_media_into(expected_job_id, request, media)
                .map_err(DeferredQueueMediaError::from_media)?;
        Ok(HydratedQueueMediaLease {
            decrypted: Arc::new(decrypted),
            reference_paths,
        })
    }
}

/// Owns every private staged path until the generation attempt finishes.
/// Dropping the last holder removes the private staging tree; memory-only
/// bytes remain on the request and are never materialized to a filesystem
/// path. Staged reference files are not overlaid onto the request — they are
/// handed out through [`Self::references`] under this same hold.
pub struct HydratedQueueMediaLease {
    decrypted: Arc<DecryptedQueueMediaSet>,
    /// One private path per `request.references` descriptor, in order.
    reference_paths: Vec<std::path::PathBuf>,
}

impl fmt::Debug for HydratedQueueMediaLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HydratedQueueMediaLease")
            .field("reference_paths", &self.reference_paths.len())
            .finish_non_exhaustive()
    }
}

impl HydratedQueueMediaLease {
    /// The request's ordered references bound to their private staged files.
    ///
    /// `None` for a request with no references. The set shares this lease's
    /// hold, so it — and any admission view minted from it — keeps the
    /// staging alive past the lease itself; every consumer builds its own set
    /// from its own hydration rather than receiving one on the job.
    pub fn references(
        &self,
        request: &mold_core::GenerateRequest,
    ) -> Result<Option<crate::reference_uploads::ResolvedReferenceSet>, DeferredQueueMediaError>
    {
        if request.references.is_none() && self.reference_paths.is_empty() {
            return Ok(None);
        }
        crate::reference_uploads::ResolvedReferenceSet::from_hydrated(
            request,
            self.reference_paths.clone(),
            Arc::clone(&self.decrypted),
        )
        .map(Some)
        .map_err(|error| {
            DeferredQueueMediaError::hold(format!(
                "durable queue media does not match its request authority: {error}"
            ))
        })
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

impl Clone for ZeroizingGenerateRequest {
    fn clone(&self) -> Self {
        Self::from_owned(self.request.clone())
    }
}

impl ZeroizingGenerateRequest {
    pub(crate) fn from_owned(request: mold_core::GenerateRequest) -> Self {
        Self {
            request,
            #[cfg(test)]
            scrub_probe: None,
        }
    }

    /// Return a payload-free copy for durable runtime publication while this
    /// owner continues to guarantee cleanup on cancellation or panic.
    pub(crate) fn scrubbed_clone(&mut self) -> mold_core::GenerateRequest {
        crate::queue_media::scrub_request_media(&mut self.request);
        self.request.clone()
    }
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

/// Seal one request exactly as admission does — extraction, projection, safe
/// open, encryption — into a fresh store under `home`, and hand back the
/// opaque deferred authority a job would carry plus the durable request JSON.
/// Test consumers hydrate it under their own lease exactly like production.
#[cfg(all(test, unix))]
pub(crate) fn seal_request_for_test(
    home: &std::path::Path,
    job_id: &str,
    request: mold_core::GenerateRequest,
    staged: Option<&crate::reference_uploads::StagedReferences>,
) -> (DeferredQueueMedia, String) {
    use crate::queue_media::{
        extract_request_media, into_seal_media, project_request_media, ProcessPrivateAuthorities,
        ProcessPrivateAuthority,
    };
    use crate::queue_media_store::QueueMediaOperationFingerprint;

    std::fs::create_dir_all(home).unwrap();
    let store = Arc::new(QueueMediaStore::open(home).unwrap().store);
    let durable_replacement = mold_core::minimax_h3::capability_contract_for_model(&request.model)
        .map(|_| ProcessPrivateAuthority::H3PrivateIngressGrant);
    let authorities =
        ProcessPrivateAuthorities::none().with_durable_replacement(durable_replacement);
    let extracted = extract_request_media(job_id, request, &authorities, staged).unwrap();
    let projection = project_request_media(extracted.media()).unwrap();
    let (request_json, media) = extracted.into_parts();
    let sealed = into_seal_media(media).unwrap();
    let reference = store
        .seal_v2_with_operation_fingerprint(
            "owner-test",
            job_id,
            &QueueMediaOperationFingerprint::sha256_v1(job_id.as_bytes()),
            &projection,
            sealed,
        )
        .unwrap();
    (
        DeferredQueueMedia::new(store, reference, projection),
        request_json,
    )
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
    fn zeroizing_owned_clones_scrub_on_success_error_and_panic() {
        let source = ZeroizingGenerateRequest::from_owned(request(std::path::Path::new(
            "/private/source.mp4",
        )));
        let run = |mode: &str, scrubbed: Arc<std::sync::atomic::AtomicBool>| {
            let clone = source.clone().with_scrub_probe(scrubbed);
            match mode {
                "success" => {
                    drop(clone);
                    Ok::<(), ()>(())
                }
                "error" => Err(()),
                "panic" => panic!("injected per-device admission panic"),
                _ => unreachable!(),
            }
        };

        for mode in ["success", "error"] {
            let scrubbed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let _ = run(mode, Arc::clone(&scrubbed));
            assert!(scrubbed.load(std::sync::atomic::Ordering::SeqCst));
        }

        let scrubbed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let scrubbed = Arc::clone(&scrubbed);
            || {
                let _ = run("panic", scrubbed);
            }
        }));
        assert!(panicked.is_err());
        assert!(scrubbed.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn zeroizing_owned_request_scrubs_when_deferred_preparation_is_aborted() {
        let scrubbed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let request = ZeroizingGenerateRequest::from_owned(request(std::path::Path::new(
            "/private/deferred.mp4",
        )))
        .with_scrub_probe(Arc::clone(&scrubbed));
        let started = Arc::new(tokio::sync::Notify::new());
        let task = tokio::spawn({
            let started = Arc::clone(&started);
            async move {
                let _request = request;
                started.notify_one();
                std::future::pending::<()>().await;
            }
        });
        started.notified().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(scrubbed.load(std::sync::atomic::Ordering::SeqCst));
    }

    /// The feeder hydrates BEFORE `prepare_generation_after_durable_ack`, so a
    /// sealed LoRA is back on the request when the scheduler resolves adapter
    /// paths. This is the whole-store round trip: extract, seal, decrypt,
    /// overlay — the adapter record comes back exactly as admitted.
    #[test]
    fn leased_hydration_restores_a_lora_sealed_beside_media() {
        let home = tempfile::tempdir().unwrap();
        let store = Arc::new(QueueMediaStore::open(home.path()).unwrap().store);
        let admitted: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "adapter beside media",
            "model": "mock",
            "width": 64,
            "height": 64,
            "steps": 1,
            "source_image": "c291cmNlLWJ5dGVz",
            "lora": { "path": "/private/one.safetensors", "scale": 0.5 },
            "loras": [
                { "path": "/private/two.safetensors", "scale": 0.7, "expert": "high" }
            ]
        }))
        .unwrap();
        let expected_lora = admitted.lora.clone();
        let expected_loras = admitted.loras.clone();
        let extracted = extract_request_media(
            "job-lora",
            admitted,
            &ProcessPrivateAuthorities::none(),
            None,
        )
        .unwrap();
        let projection = project_request_media(extracted.media()).unwrap();
        let (request_json, opaque_media) = extracted.into_parts();
        let mut sanitized: mold_core::GenerateRequest =
            serde_json::from_str(&request_json).unwrap();
        assert!(sanitized.lora.is_none() && sanitized.loras.is_none());
        let media = into_seal_media(opaque_media).unwrap();
        let reference = store
            .seal_v2_with_operation_fingerprint(
                "owner-lora",
                "job-lora",
                &QueueMediaOperationFingerprint::sha256_v1(b"lora operation"),
                &projection,
                media,
            )
            .unwrap();
        let deferred = DeferredQueueMedia::new(store, reference, projection);

        let _lease = deferred.hydrate_into("job-lora", &mut sanitized).unwrap();
        assert_eq!(
            sanitized.source_image.as_deref(),
            Some(b"source-bytes".as_slice())
        );
        assert_eq!(
            serde_json::to_value(&sanitized.lora).unwrap(),
            serde_json::to_value(&expected_lora).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&sanitized.loras).unwrap(),
            serde_json::to_value(&expected_loras).unwrap()
        );
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
            None,
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
            None,
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
            None,
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

    fn ref2va_request_for(files: &[&[u8]]) -> mold_core::GenerateRequest {
        use sha2::{Digest as _, Sha256};

        let references = files
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                serde_json::json!({
                    "kind": "image",
                    "media": { "authority": "descriptor" },
                    "provenance": {
                        "name": format!("reference-{}.png", index + 1),
                        "sha256": format!("{:x}", Sha256::digest(bytes))
                    },
                    "mime_type": "image/png",
                    "width": 1024,
                    "height": 768
                })
            })
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({
            "prompt": "ordered references",
            "model": mold_core::minimax_h3::REF2VA_COMFY,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": 4,
            "guidance": 0.0,
            "seed": 7,
            "batch_size": 1,
            "output_format": "mp4",
            "references": references
        }))
        .unwrap()
    }

    /// The whole point of sealing reference bytes: a set sealed by one
    /// process is hydrated by the next one over the same `MOLD_HOME`, and the
    /// bindings it mints verify the descriptors' digests against the bytes
    /// that came back — nothing from the admitting process survives but the
    /// store.
    #[test]
    fn sealed_references_survive_a_store_reopen_and_bind_by_digest() {
        let home = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let files: [&[u8]; 2] = [b"first-reference-bytes", b"second-reference-bytes"];
        let paths = files
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                let path = staging.path().join(format!("reference-{index}.media"));
                std::fs::write(&path, bytes).unwrap();
                path
            })
            .collect::<Vec<_>>();
        let request = ref2va_request_for(&files);
        let staged =
            crate::reference_uploads::StagedReferences::from_files_for_test(&request, paths);
        let (admitted, request_json) =
            seal_request_for_test(home.path(), "job-restart", request.clone(), Some(&staged));
        drop(staged);
        drop(staging);

        // "Restart": a new store over the same home, the same media-set ref
        // and projection as the journal row carries.
        let reopened = Arc::new(QueueMediaStore::open(home.path()).unwrap().store);
        let deferred = DeferredQueueMedia::new(
            reopened,
            admitted.media_set_ref().clone(),
            admitted.projection().clone(),
        );
        drop(admitted);
        let mut restored: mold_core::GenerateRequest = serde_json::from_str(&request_json).unwrap();
        assert_eq!(
            serde_json::to_value(&restored).unwrap(),
            serde_json::to_value(&request).unwrap(),
            "descriptors are settings and ride the row unchanged"
        );

        let lease = deferred.hydrate_into("job-restart", &mut restored).unwrap();
        let references = lease
            .references(&restored)
            .unwrap()
            .expect("a reference-bearing request hydrates a set");
        assert_eq!(references.entries().len(), 2);
        let bindings = references.inference_bindings(&restored, None).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].metadata().index, 1);
        assert_eq!(bindings[1].metadata().index, 2);
        let private_path = references.entries()[0].path.clone();
        assert!(private_path.is_file());

        // A request with no references hydrates no set at all.
        let mut plain: mold_core::GenerateRequest = serde_json::from_str(&request_json).unwrap();
        plain.references = None;
        assert!(matches!(
            lease.references(&plain),
            Err(error) if error.disposition() == DeferredHydrationDisposition::Hold
        ));

        // The staging is released by the last holder, whichever it is.
        drop(lease);
        assert!(private_path.is_file());
        drop(bindings);
        drop(references);
        assert!(!private_path.exists());
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
