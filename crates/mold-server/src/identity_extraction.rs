//! The one place a face is turned into an identity, and the counter that
//! proves it happens once.
//!
//! #1223's central requirement is a lifetime, not a computation: when the
//! effective `id_weight` is above zero the identity is resolved ONCE, at
//! parent-request admission, before batch fan-out, and every sibling and every
//! denoise step reuses that exact value. The computation itself lives in
//! `mold_inference::identity::extraction`; what lives here is the single call
//! site, the accounting that makes "once" checkable, and the test seam that
//! lets the contract be tested without 2 GB of weights.
//!
//! ## Why here and not in the per-device loop
//!
//! `variant_dependencies::prepare_inputs_for_devices` materializes asset
//! *paths* once per eligible device, because a mixed-capacity host can pick
//! different encoder variants per GPU. The identity is the opposite kind of
//! thing: it is a device-independent 256 KiB value derived from the request's
//! own bytes, identical on every device by construction. So it is resolved
//! once, after that loop, from whichever device's frozen bundle resolved — and
//! stored on `PreparedExecutionInputs`, which the batch parent clones into
//! every child.
//!
//! ## Why the extraction never overlaps the encode peak
//!
//! It runs at admission, before the scheduler has leased a device, so the T5
//! and CLIP encoders it must not compete with do not exist yet. Its ~1.4 GB of
//! host RAM (see `EXTRACTION_HOST_PEAK_BYTES`) is allocated and released
//! before the job is dispatched. That is a stronger guarantee than a scheduled
//! slot: the two peaks cannot coexist rather than being arranged not to.

use std::sync::atomic::{AtomicU64, Ordering};

use mold_core::identity::FrozenIdentityEmbedding;
use mold_core::pulid_assets::PulidPaths;
use mold_core::GenerateRequest;

/// Every identity extraction this process has performed.
///
/// Exists for the contract test rather than for telemetry: `batch_size = 4`
/// across two devices must move this by exactly one. A counter is the only way
/// to state that, because the alternative — asserting the four children agree
/// — passes just as well when the extractor ran four times deterministically.
static EXTRACTIONS: AtomicU64 = AtomicU64::new(0);

/// Reads the extraction counter. Test-only: the counter exists to state the
/// once-per-parent contract, not to report it.
#[cfg(test)]
pub(crate) fn extraction_count() -> u64 {
    EXTRACTIONS.load(Ordering::SeqCst)
}

/// Whether this request resolves an identity at all.
///
/// Weight zero is completely inert — no pull, no decode, no load, no
/// extraction — so the predicate is the effective weight, never the mere
/// presence of the fields. Shared with the asset planner so the two can never
/// disagree about which requests are identity requests.
pub(crate) fn request_resolves_identity(request: &GenerateRequest) -> bool {
    crate::identity_dependencies::request_needs_identity_assets(request)
}

/// Extract and freeze the identity for one parent request.
///
/// Returns `Ok(None)` for every request that does not condition on a face.
/// A request that does, but whose bundle did not resolve, is an error: a print
/// that renders without the face nobody would be told about is exactly what
/// `mold_core::identity` refuses at the contract boundary.
pub(crate) async fn resolve_identity_embedding(
    request: &GenerateRequest,
    paths: Option<&PulidPaths>,
) -> Result<Option<FrozenIdentityEmbedding>, String> {
    if !request_resolves_identity(request) {
        return Ok(None);
    }
    let Some(image) = request.id_image.clone() else {
        return Err(
            "id_image is required when id_weight, id_start_step, or id_image_name is set"
                .to_string(),
        );
    };
    let Some(paths) = paths.cloned() else {
        return Err(
            "this request asks for face-identity conditioning but no PuLID bundle resolved on \
             any eligible device; run `mold pull pulid-flux --accept-license \
             insightface-antelopev2`"
                .to_string(),
        );
    };

    EXTRACTIONS.fetch_add(1, Ordering::SeqCst);

    #[cfg(test)]
    if let Some(stub) = test_stub() {
        return stub(&paths, &image);
    }

    extract_blocking(paths, image).await
}

/// The real extraction, off the async runtime.
///
/// It is CPU-bound for seconds — two ONNX graph decodes, a 609 MB tower load,
/// and a 24-block forward — so it never runs on a reactor thread.
#[cfg(feature = "pulid")]
async fn extract_blocking(
    paths: PulidPaths,
    image: Vec<u8>,
) -> Result<Option<FrozenIdentityEmbedding>, String> {
    let outcome = tokio::task::spawn_blocking(move || {
        mold_inference::identity::extraction::extract_identity_embedding(&paths, &image)
    })
    .await
    .map_err(|error| format!("the identity extractor panicked: {error}"))?
    .map_err(|error| format!("{error:#}"))?;

    if let Some(warning) = &outcome.warning {
        tracing::warn!(target: "mold::identity", "{warning}");
    }
    Ok(Some(outcome.embedding))
}

/// A build without `pulid` never reaches here: the request contract refuses
/// every identity request before admission. The arm exists so the module
/// compiles in the default feature set rather than being conditionally wired
/// into `prepare_inputs_for_devices`, which would be a second place the
/// lifetime could diverge.
#[cfg(not(feature = "pulid"))]
async fn extract_blocking(
    _paths: PulidPaths,
    _image: Vec<u8>,
) -> Result<Option<FrozenIdentityEmbedding>, String> {
    Err(mold_core::identity::IDENTITY_BUILD_UNSUPPORTED.to_string())
}

#[cfg(test)]
type Stub = fn(&PulidPaths, &[u8]) -> Result<Option<FrozenIdentityEmbedding>, String>;

#[cfg(test)]
static TEST_STUB: std::sync::Mutex<Option<Stub>> = std::sync::Mutex::new(None);

#[cfg(test)]
fn test_stub() -> Option<Stub> {
    *TEST_STUB.lock().unwrap_or_else(|error| error.into_inner())
}

/// Substitute a stub extractor for the duration of a test, and reset the
/// counter. The real stack needs ~2 GB of pinned weights; the lifetime
/// contract does not.
#[cfg(test)]
pub(crate) struct StubbedExtractor {
    baseline: u64,
}

#[cfg(test)]
impl StubbedExtractor {
    pub(crate) fn install(stub: Stub) -> Self {
        *TEST_STUB.lock().unwrap_or_else(|error| error.into_inner()) = Some(stub);
        Self {
            baseline: extraction_count(),
        }
    }

    /// Extractions performed since this stub was installed.
    pub(crate) fn extractions(&self) -> u64 {
        extraction_count() - self.baseline
    }
}

#[cfg(test)]
impl Drop for StubbedExtractor {
    fn drop(&mut self) {
        *TEST_STUB.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }
}

/// A frozen embedding built from a stub, for tests that need a concrete value.
#[cfg(test)]
pub(crate) fn stub_embedding(image: &[u8]) -> FrozenIdentityEmbedding {
    let values: Vec<f32> = (0..mold_core::identity::ID_EMBEDDING_VALUES)
        .map(|index| (index % 97) as f32 / 97.0)
        .collect();
    FrozenIdentityEmbedding::new(
        &values,
        mold_core::identity::id_image_sha256(image),
        mold_core::identity::IdentityAssetDigests {
            adapter: "stub-adapter".to_string(),
            vision: "stub-vision".to_string(),
            face_detector: "stub-detector".to_string(),
            face_recognizer: "stub-recognizer".to_string(),
        },
    )
    .expect("the stub embedding is correctly shaped")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id_weight: Option<f64>) -> GenerateRequest {
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a portrait",
            "model": "flux-dev:q8",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 3.5,
            "batch_size": 1,
            "strength": 0.75,
        }))
        .expect("the minimal generate-request wire shape");
        request.id_image = Some(b"pretend-png".to_vec());
        request.id_weight = id_weight;
        request
    }

    fn paths() -> PulidPaths {
        PulidPaths {
            adapter: "/models/shared/pulid/adapter.safetensors".into(),
            vision_encoder_source: "/models/shared/pulid/eva.pt".into(),
            face_detector: "/models/shared/pulid/scrfd.onnx".into(),
            face_recognizer: "/models/shared/pulid/glintr100.onnx".into(),
        }
    }

    fn stub(_: &PulidPaths, image: &[u8]) -> Result<Option<FrozenIdentityEmbedding>, String> {
        Ok(Some(stub_embedding(image)))
    }

    /// The zero-weight rule, at the one gate that could break it: no
    /// extraction is even counted, let alone performed.
    #[tokio::test]
    async fn a_zero_weight_request_performs_no_extraction() {
        let stubbed = StubbedExtractor::install(stub);
        let resolved = resolve_identity_embedding(&request(Some(0.0)), Some(&paths()))
            .await
            .expect("a zero weight is inert, not an error");
        assert!(resolved.is_none());
        assert_eq!(stubbed.extractions(), 0);
    }

    /// A request with no identity fields at all takes the same path.
    #[tokio::test]
    async fn a_request_without_identity_performs_no_extraction() {
        let stubbed = StubbedExtractor::install(stub);
        let mut plain = request(None);
        plain.id_image = None;
        let resolved = resolve_identity_embedding(&plain, Some(&paths()))
            .await
            .expect("nothing to resolve");
        assert!(resolved.is_none());
        assert_eq!(stubbed.extractions(), 0);
    }

    #[tokio::test]
    async fn a_conditioned_request_extracts_exactly_once() {
        let stubbed = StubbedExtractor::install(stub);
        let frozen = resolve_identity_embedding(&request(None), Some(&paths()))
            .await
            .expect("the stub extractor answers")
            .expect("a conditioned request resolves an identity");
        assert_eq!(stubbed.extractions(), 1);
        assert_eq!(
            frozen.source_sha256(),
            mold_core::identity::id_image_sha256(b"pretend-png")
        );
    }

    /// A conditioned request whose bundle never resolved must fail loudly.
    /// Rendering it would produce a print with somebody else's face and no
    /// indication that anything went wrong.
    #[tokio::test]
    async fn a_missing_bundle_is_an_error_not_an_unconditioned_render() {
        let stubbed = StubbedExtractor::install(stub);
        let error = resolve_identity_embedding(&request(None), None)
            .await
            .expect_err("no bundle, no render");
        assert!(error.contains("mold pull pulid-flux"), "{error}");
        assert!(error.contains("insightface-antelopev2"), "{error}");
        assert_eq!(
            stubbed.extractions(),
            0,
            "nothing may be counted for a request that could not run"
        );
    }
}
