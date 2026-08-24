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
//! ## Why inside the lease and not in the per-device loop
//!
//! It used to run in `variant_dependencies::prepare_inputs_for_devices`, at
//! admission, before the scheduler had leased anything — because
//! `candle-onnx` could not place a tensor off the CPU and because a
//! pre-lease phase cannot overlap the T5/CLIP encode peak by construction.
//! #1227 phase 1 removed the first constraint and phase 2 traded the second
//! for its cost: `docs/architecture/pulid-perf.md` §4 measured one extraction
//! at 2,840 ms on the host, of which the EVA02-CLIP tower alone was 79%, and
//! §5 moved the whole phase onto the render's own leased device, where the
//! same tower forward is 92 ms.
//!
//! So this now runs INSIDE the leased job, first, before prompt encode, on the
//! device the frozen plan named — `ProgressPhase::IdentityExtract`. The
//! extraction is built, forwarded, and fully released before
//! `flux::identity::EngineIdentityState` begins the adapter's residency: a
//! strictly earlier, strictly disjoint phase of the same lease, never a third
//! permanent resident.
//!
//! ## Why extractions no longer need a slot of their own
//!
//! [`ExtractionSlot`] used to serialize them behind a bare
//! `tokio::sync::Semaphore` charged against a fresh `ram_snapshot()`,
//! deliberately NOT the scheduler's `HostMemoryLedger`, because "extraction
//! happens strictly BEFORE this job has a lease". That justification expired
//! with the call site. A leased job's memory is the frozen plan's business:
//! the device bytes are charged by
//! `memory_preflight::IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES` and the host
//! bytes by the identity artifacts' own `is_host_only` component roles, both
//! rechecked at dispatch against the granting ledger like every other phase.
//! Two admission-time memory gates that can disagree is worse than one, so the
//! slot is retired rather than run beside the ledger — and serialization comes
//! free with it, because a lease is already exclusive on its device.
//!
//! ## What "exactly once per parent" means now
//!
//! The counter still states it, and the cache is what keeps it true: the first
//! sibling to reach a device extracts, and every other sibling — on any device,
//! in any order — is answered by
//! `mold_inference::identity::extraction`'s per-photograph cache without
//! opening a model. A batch child that was re-prepared by the scheduler still
//! carries its parent's frozen value and never asks at all.

use std::sync::atomic::{AtomicU64, Ordering};

use mold_core::identity::FrozenIdentityEmbedding;
use mold_core::pulid_assets::PulidPaths;
use mold_core::GenerateRequest;
use zeroize::{Zeroize, Zeroizing};

/// The request-owned photographs copied for one extraction attempt.
///
/// This owner is deliberately not cloneable. The request remains the wire
/// authority, but the extractor needs owned bytes that survive for the whole
/// synchronous call. Keeping those copies behind one RAII boundary guarantees
/// they are wiped on every return path, including an extractor panic.
struct OwnedIdentityPhotographs {
    images: Zeroizing<Vec<Vec<u8>>>,
    #[cfg(test)]
    drop_probe: Option<std::sync::Arc<PhotographDropProbe>>,
}

impl OwnedIdentityPhotographs {
    fn from_request(request: &GenerateRequest) -> Self {
        Self {
            images: Zeroizing::new(
                mold_core::identity::identity_images(request)
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect(),
            ),
            #[cfg(test)]
            drop_probe: None,
        }
    }

    fn as_slice(&self) -> &[Vec<u8>] {
        self.images.as_slice()
    }

    fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    #[cfg(test)]
    fn with_drop_probe(
        images: Vec<Vec<u8>>,
        drop_probe: std::sync::Arc<PhotographDropProbe>,
    ) -> Self {
        Self {
            images: Zeroizing::new(images),
            drop_probe: Some(drop_probe),
        }
    }
}

impl Drop for OwnedIdentityPhotographs {
    fn drop(&mut self) {
        // Wipe before the test hook observes the allocation. `Zeroizing` also
        // wipes during field drop, so this explicit pass is intentionally
        // redundant in production and makes the destructor probe exact.
        self.images.zeroize();
        #[cfg(test)]
        if let Some(probe) = &self.drop_probe {
            probe.dropped.store(true, Ordering::SeqCst);
            probe.zeroized.store(
                self.images.iter().flatten().all(|byte| *byte == 0),
                Ordering::SeqCst,
            );
        }
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct PhotographDropProbe {
    dropped: std::sync::atomic::AtomicBool,
    zeroized: std::sync::atomic::AtomicBool,
}

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

/// Why one identity resolution failed, and whose fault it is.
///
/// #1227 phase 2 moved the extraction onto the render's leased GPU, so its
/// failures now reach a worker's reliability counter for the first time. That
/// counter exists to take a sick card out of rotation, and an unusable
/// photograph is not a sick card: three faceless images in a row would
/// otherwise degrade a healthy single-GPU worker for a minute and the person
/// who supplied them would see an unrelated "GPU degraded" story.
///
/// So the attribution travels with the message. `mold_inference`'s
/// `IdentityError::is_user_input` is the authority for the extraction itself —
/// a face that is not there, a payload that will not decode, landmarks that
/// will not fit — and this module adds the request-shape refusals it raises
/// before the extractor is ever entered.
#[derive(Debug, Clone)]
pub struct IdentityFailure {
    /// The message the caller sees.
    pub message: String,
    /// True when the refusal is about the request, reproducible on any device.
    /// Only a `false` may touch the worker's health.
    pub user_input: bool,
}

impl IdentityFailure {
    /// A refusal about the request or the photograph.
    pub fn user_input(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            user_input: true,
        }
    }

    /// A refusal about the machine: a load failure, a kernel fault, a driver
    /// error.
    pub fn device(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            user_input: false,
        }
    }
}

impl std::fmt::Display for IdentityFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for IdentityFailure {}

/// What one parent request's identity resolution produced.
#[derive(Debug, Default, Clone)]
pub struct ResolvedIdentity {
    /// Whether this resolution actually COMPUTED an identity, or whether every
    /// photograph came from the per-photograph cache — including the case
    /// where a batch sibling waited on another sibling's single flight and
    /// took its tokens.
    ///
    /// Two things read it and both would be wrong without it: the
    /// once-per-parent counter, which must not count a sibling that computed
    /// nothing, and `ProgressPhase::IdentityExtract`, which must not teach the
    /// scheduler that this phase costs the two milliseconds a cache hit costs.
    pub extracted: bool,
    /// `None` for every request that does not condition on a face.
    pub embedding: Option<FrozenIdentityEmbedding>,
    /// A caller-facing advisory the extraction produced — today only "several
    /// faces were found, the largest was used".
    ///
    /// It travels with the embedding rather than being logged and forgotten:
    /// the person who supplied a group photograph is the one who needs to know
    /// which face was picked, and a `tracing::warn!` on the server reaches
    /// nobody holding a CLI or a browser.
    pub warning: Option<String>,
}

/// [`resolve_identity_for_lease`], with the batch parent's own pin as the
/// authority.
///
/// This is what the worker calls, and the ordering is the contract: the pin is
/// read BEFORE the resolver and written from it, so a sibling arriving after
/// another has already extracted renders that exact face rather than one it
/// composed itself. The per-photograph cache underneath still does the work of
/// making concurrent siblings compose once; the pin is what makes "one identity
/// per parent" survive the cache's own bounded eviction — sixteen unrelated
/// photographs between two waves of siblings, or a retry after a long gap.
///
/// A pin hit reports `extracted: false`, so it neither counts as an extraction
/// nor teaches the scheduler what this phase costs.
pub fn resolve_pinned_identity_for_lease(
    request: &GenerateRequest,
    paths: Option<&PulidPaths>,
    pin: &crate::execution_plan::IdentityPin,
    backend: mold_core::GpuBackend,
    ordinal: usize,
) -> Result<ResolvedIdentity, IdentityFailure> {
    if let Some(pinned) = pin.get() {
        return Ok(ResolvedIdentity {
            extracted: false,
            embedding: Some(pinned.embedding),
            // The advisory rides the pin, so every sibling reports the face
            // choice the extraction actually made. Post-lease preparation
            // leaves `identity_warning` empty, so a child that took the pin and
            // did not carry this would drop it silently — and the person who
            // supplied a group photograph would see it on one print of four.
            warning: pinned.warning,
        });
    }
    let resolved = resolve_identity_for_lease(request, paths, backend, ordinal)?;
    // Adopt whatever is pinned afterwards, never the local value: a sibling
    // that raced and lost must render the WINNER's face. Keeping its own would
    // leave two siblings of one print conditioned on two identities that differ
    // at the measured device tolerance, and look like it had worked.
    let Some(embedding) = resolved.embedding else {
        return Ok(resolved);
    };
    let pinned = pin.pin(crate::execution_plan::PinnedIdentity {
        embedding,
        warning: resolved.warning,
    });
    Ok(ResolvedIdentity {
        extracted: resolved.extracted,
        embedding: Some(pinned.embedding),
        warning: pinned.warning,
    })
}

/// Extract and freeze the identity for one leased job, on that job's device.
///
/// Returns an empty [`ResolvedIdentity`] for every request that does not
/// condition on a face. A request that does, but whose bundle did not resolve,
/// is an error: a print that renders without the face nobody would be told
/// about is exactly what `mold_core::identity` refuses at the contract
/// boundary.
///
/// Synchronous, and called on the worker thread that owns the lease. It used
/// to be `async` only because it awaited [`ExtractionSlot`]'s semaphore, and
/// that slot is retired — see this module's doc. `backend`/`ordinal` name the
/// device the frozen plan admitted; they are never probed or defaulted,
/// because an extraction that quietly ran somewhere else would be running
/// against a memory grant made for somewhere else.
pub fn resolve_identity_for_lease(
    request: &GenerateRequest,
    paths: Option<&PulidPaths>,
    backend: mold_core::GpuBackend,
    ordinal: usize,
) -> Result<ResolvedIdentity, IdentityFailure> {
    if !request_resolves_identity(request) {
        return Ok(ResolvedIdentity::default());
    }
    // Either wire shape, in request order. The photographs are processed
    // serially, and each one hits or misses the per-photograph cache
    // independently.
    let images = OwnedIdentityPhotographs::from_request(request);
    if images.is_empty() {
        return Err(IdentityFailure::user_input(
            "id_image (or id_images) is required when id_weight, id_start_step, id_image_name, \
             or id_image_names is set",
        ));
    }
    // Which renders need the unconditional identity is a per-family question
    // — FLUX's true-CFG opt-in versus SDXL's always-on classifier-free
    // negative pass — and `mold_core::identity` owns it so this and the
    // engine's own requirement cannot disagree.
    let want_uncond = mold_core::identity::request_needs_unconditional_identity(request);
    let Some(paths) = paths.cloned() else {
        // A missing bundle is a machine that has not been provisioned, not a
        // photograph the caller can fix — but it is equally not a fault the
        // GPU's health counter can act on, and every retry will hit it. Named
        // as request-scoped so a mis-provisioned host is not also degraded.
        return Err(IdentityFailure::user_input(format!(
            "this request asks for face-identity conditioning but no PuLID bundle resolved \
                 on any eligible device; run `mold pull {bundle} --accept-license \
                 insightface-antelopev2`",
            bundle = mold_core::identity::identity_family(&request.model)
                .map(mold_core::identity::IdentityFamily::manifest)
                .unwrap_or(mold_core::manifest::PULID_FLUX_MANIFEST),
        )));
    };

    #[cfg(test)]
    let resolved = match test_stub() {
        Some(stub) => stub(&paths, images.as_slice(), want_uncond)?,
        None => extract_on_device(paths, images.as_slice(), want_uncond, backend, ordinal)?,
    };
    #[cfg(not(test))]
    let resolved = extract_on_device(paths, images.as_slice(), want_uncond, backend, ordinal)?;

    // Counted on what was COMPUTED, not on what was asked for, and through the
    // one arm both paths take. Four siblings of one parent resolve four times
    // and extract once — the first holds the photograph's single flight and
    // the other three take its tokens — so counting resolutions would report
    // four and the once-per-parent contract would stop being checkable exactly
    // when it started mattering.
    if resolved.extracted {
        EXTRACTIONS.fetch_add(1, Ordering::SeqCst);
    }
    Ok(resolved)
}

/// The real extraction, on the leased device.
///
/// Already off the reactor: the caller is the GPU owner thread, which is the
/// only thread allowed to touch this device at all. It is CPU- and GPU-bound
/// for hundreds of milliseconds — two graph decodes, a 609 MB tower
/// materialization, and a 24-block forward — so it never runs anywhere else.
#[cfg(feature = "pulid")]
fn extract_on_device(
    paths: PulidPaths,
    images: &[Vec<u8>],
    want_uncond: bool,
    backend: mold_core::GpuBackend,
    ordinal: usize,
) -> Result<ResolvedIdentity, IdentityFailure> {
    use mold_inference::identity::extraction::ExtractionPlacement;

    let borrowed: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let outcome = mold_inference::identity::extraction::extract_identity_embeddings_at(
        &paths,
        &borrowed,
        want_uncond,
        ExtractionPlacement::Gpu { backend, ordinal },
    )
    // `IdentityError::is_user_input` is the authority: only a `Runtime` arm is
    // about the machine that ran it.
    .map_err(|error| IdentityFailure {
        message: format!("{error:#}"),
        user_input: error.is_user_input(),
    })?;

    if let Some(warning) = &outcome.warning {
        tracing::warn!(target: "mold::identity", "{warning}");
    }
    Ok(ResolvedIdentity {
        extracted: outcome.extracted,
        embedding: Some(outcome.embedding),
        warning: outcome.warning,
    })
}

/// A build without `pulid` never reaches here: the request contract refuses
/// every identity request before admission. The arm exists so the module
/// compiles in the default feature set rather than being conditionally wired
/// into the dispatch path, which would be a second place the lifetime could
/// diverge.
#[cfg(not(feature = "pulid"))]
fn extract_on_device(
    _paths: PulidPaths,
    _images: &[Vec<u8>],
    _want_uncond: bool,
    _backend: mold_core::GpuBackend,
    _ordinal: usize,
) -> Result<ResolvedIdentity, IdentityFailure> {
    Err(IdentityFailure::user_input(
        mold_core::identity::IDENTITY_BUILD_UNSUPPORTED,
    ))
}

#[cfg(test)]
type Stub = fn(&PulidPaths, &[Vec<u8>], bool) -> Result<ResolvedIdentity, IdentityFailure>;

#[cfg(test)]
static TEST_STUB: std::sync::Mutex<Option<Stub>> = std::sync::Mutex::new(None);

#[cfg(test)]
fn test_stub() -> Option<Stub> {
    *TEST_STUB.lock().unwrap_or_else(|error| error.into_inner())
}

/// Serializes every test that observes [`EXTRACTIONS`].
///
/// The counter is process-global on purpose — it is the only way to state
/// "exactly one extraction for this whole request" — which means two tests
/// counting at once would each see the other's work. This is the lock that
/// makes the delta meaningful.
#[cfg(test)]
static STUB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Substitute a stub extractor for the duration of a test and measure the
/// extractions it causes. The real stack needs ~2 GB of pinned weights; the
/// lifetime contract does not.
#[cfg(test)]
pub(crate) struct StubbedExtractor {
    baseline: u64,
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl StubbedExtractor {
    pub(crate) fn install(stub: Stub) -> Self {
        let guard = STUB_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        *TEST_STUB.lock().unwrap_or_else(|error| error.into_inner()) = Some(stub);
        Self {
            baseline: extraction_count(),
            _guard: guard,
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
    stub_embedding_for(std::slice::from_ref(&image.to_vec()), false)
}

/// The set-aware form: records every source in order, and attaches an
/// unconditional half exactly when the request asked for one.
#[cfg(test)]
pub(crate) fn stub_embedding_for(images: &[Vec<u8>], want_uncond: bool) -> FrozenIdentityEmbedding {
    let values: Vec<f32> = (0..mold_core::identity::ID_EMBEDDING_VALUES)
        .map(|index| (index % 97) as f32 / 97.0)
        .collect();
    let embedding = FrozenIdentityEmbedding::from_sources(
        &values,
        images
            .iter()
            .map(|image| mold_core::identity::id_image_sha256(image))
            .collect(),
        mold_core::identity::IdentityAssetDigests {
            adapter: "stub-adapter".to_string(),
            vision: "stub-vision".to_string(),
            face_detector: "stub-detector".to_string(),
            face_recognizer: "stub-recognizer".to_string(),
            face_parser: "stub-parser".to_string(),
        },
    )
    .expect("the stub embedding is correctly shaped");
    if want_uncond {
        embedding
            .with_uncond(&vec![0.25_f32; mold_core::identity::ID_EMBEDDING_VALUES])
            .expect("the stub unconditional embedding is correctly shaped")
    } else {
        embedding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_photographs_zeroize_after(operation: impl FnOnce(&[Vec<u8>])) {
        let probe = std::sync::Arc::new(PhotographDropProbe::default());
        let photographs = OwnedIdentityPhotographs::with_drop_probe(
            vec![
                b"first-private-photo".to_vec(),
                b"second-private-photo".to_vec(),
            ],
            std::sync::Arc::clone(&probe),
        );
        operation(photographs.as_slice());
        drop(photographs);
        assert!(probe.dropped.load(Ordering::SeqCst));
        assert!(probe.zeroized.load(Ordering::SeqCst));
    }

    fn error_with_owned_photographs(
        probe: std::sync::Arc<PhotographDropProbe>,
    ) -> Result<(), IdentityFailure> {
        let photographs =
            OwnedIdentityPhotographs::with_drop_probe(vec![b"error-private-photo".to_vec()], probe);
        assert_eq!(photographs.as_slice().len(), 1);
        Err(IdentityFailure::user_input("deterministic error probe"))?;
        Ok(())
    }

    #[test]
    fn owned_photographs_zeroize_on_success_error_and_panic() {
        assert_photographs_zeroize_after(|images| assert_eq!(images.len(), 2));

        let error_probe = std::sync::Arc::new(PhotographDropProbe::default());
        assert!(error_with_owned_photographs(std::sync::Arc::clone(&error_probe)).is_err());
        assert!(error_probe.dropped.load(Ordering::SeqCst));
        assert!(error_probe.zeroized.load(Ordering::SeqCst));

        let probe = std::sync::Arc::new(PhotographDropProbe::default());
        let photographs = OwnedIdentityPhotographs::with_drop_probe(
            vec![b"panic-private-photo".to_vec()],
            std::sync::Arc::clone(&probe),
        );
        let panicked = std::panic::catch_unwind(move || {
            let _owner = photographs;
            panic!("deterministic destructor probe");
        });
        assert!(panicked.is_err());
        assert!(probe.dropped.load(Ordering::SeqCst));
        assert!(probe.zeroized.load(Ordering::SeqCst));
    }

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
            family: mold_core::identity::IdentityFamily::Flux,
            adapter: "/models/shared/pulid/adapter.safetensors".into(),
            vision_encoder_source: "/models/shared/pulid/eva.pt".into(),
            face_detector: "/models/shared/pulid/scrfd.onnx".into(),
            face_recognizer: "/models/shared/pulid/glintr100.onnx".into(),
            face_parser_source: "/models/shared/pulid/parsing_bisenet.pth".into(),
        }
    }

    fn stub(
        _: &PulidPaths,
        images: &[Vec<u8>],
        want_uncond: bool,
    ) -> Result<ResolvedIdentity, IdentityFailure> {
        Ok(ResolvedIdentity {
            extracted: true,
            embedding: Some(stub_embedding_for(images, want_uncond)),
            warning: None,
        })
    }

    /// The lease every test extracts under. Named rather than probed: the
    /// stubbed extractor never opens a device, and a test that guessed one
    /// would be asserting about this machine.
    const LEASED_BACKEND: mold_core::GpuBackend = mold_core::GpuBackend::Cuda;
    const LEASED_ORDINAL: usize = 0;

    /// The multi-face advisory the real extractor emits, so the plumbing that
    /// carries it out of admission can be tested without a group photograph.
    fn stub_with_warning(
        _: &PulidPaths,
        images: &[Vec<u8>],
        want_uncond: bool,
    ) -> Result<ResolvedIdentity, IdentityFailure> {
        Ok(ResolvedIdentity {
            extracted: true,
            embedding: Some(stub_embedding_for(images, want_uncond)),
            warning: Some(
                "3 faces were detected in the identity image; conditioning on the largest one"
                    .to_string(),
            ),
        })
    }

    /// The zero-weight rule, at the one gate that could break it: no
    /// extraction is even counted, let alone performed.
    #[test]
    fn a_zero_weight_request_performs_no_extraction() {
        let stubbed = StubbedExtractor::install(stub);
        let resolved = resolve_identity_for_lease(
            &request(Some(0.0)),
            Some(&paths()),
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("a zero weight is inert, not an error");
        assert!(resolved.embedding.is_none());
        assert_eq!(stubbed.extractions(), 0);
    }

    /// A request with no identity fields at all takes the same path.
    #[test]
    fn a_request_without_identity_performs_no_extraction() {
        let stubbed = StubbedExtractor::install(stub);
        let mut plain = request(None);
        plain.id_image = None;
        let resolved =
            resolve_identity_for_lease(&plain, Some(&paths()), LEASED_BACKEND, LEASED_ORDINAL)
                .expect("nothing to resolve");
        assert!(resolved.embedding.is_none());
        assert_eq!(stubbed.extractions(), 0);
    }

    #[test]
    fn a_conditioned_request_extracts_exactly_once() {
        let stubbed = StubbedExtractor::install(stub);
        let frozen = resolve_identity_for_lease(
            &request(None),
            Some(&paths()),
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the stub extractor answers")
        .embedding
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
    #[test]
    fn a_missing_bundle_is_an_error_not_an_unconditioned_render() {
        let stubbed = StubbedExtractor::install(stub);
        let error =
            resolve_identity_for_lease(&request(None), None, LEASED_BACKEND, LEASED_ORDINAL)
                .expect_err("no bundle, no render");
        assert!(error.message.contains("mold pull pulid-flux"), "{error}");
        assert!(error.message.contains("insightface-antelopev2"), "{error}");
        assert!(
            error.user_input,
            "a host with no bundle is not a sick GPU: {error}"
        );
        assert_eq!(
            stubbed.extractions(),
            0,
            "nothing may be counted for a request that could not run"
        );
    }

    /// Counts how many stubbed extractions are in flight at once, and the
    /// highest that number ever reached.
    static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
    static PEAK_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

    fn overlap_observing_stub(
        _: &PulidPaths,
        images: &[Vec<u8>],
        want_uncond: bool,
    ) -> Result<ResolvedIdentity, IdentityFailure> {
        let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
        PEAK_IN_FLIGHT.fetch_max(now, Ordering::SeqCst);
        // Long enough that a genuinely concurrent peer would be observed.
        std::thread::sleep(std::time::Duration::from_millis(60));
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        Ok(ResolvedIdentity {
            extracted: true,
            embedding: Some(stub_embedding_for(images, want_uncond)),
            warning: None,
        })
    }

    /// The extractor's own advisory — which of several faces it picked —
    /// must leave this module with the embedding. Logging it server-side, as
    /// it was before #1223, reaches nobody holding a CLI or a browser.
    #[test]
    fn the_multi_face_advisory_travels_with_the_identity() {
        let _stubbed = StubbedExtractor::install(stub_with_warning);
        let resolved = resolve_identity_for_lease(
            &request(None),
            Some(&paths()),
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the stub extractor answers");

        assert!(resolved.embedding.is_some());
        let warning = resolved
            .warning
            .expect("an unforced face choice is reported");
        assert!(warning.contains("faces were detected"), "{warning}");
        assert!(warning.contains("largest"), "{warning}");
    }

    /// A single-face photograph is the ordinary case and says nothing.
    #[test]
    fn an_unambiguous_identity_carries_no_advisory() {
        let _stubbed = StubbedExtractor::install(stub);
        let resolved = resolve_identity_for_lease(
            &request(None),
            Some(&paths()),
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the stub extractor answers");
        assert!(resolved.embedding.is_some());
        assert!(resolved.warning.is_none());
    }

    /// A multi-photograph request is still ONE extraction: one pass over the
    /// set, one device peak, and one frozen identity every sibling reuses. The
    /// counter is the only way to state that — four photographs handled by four
    /// separate resolutions would agree on nothing and cost four peaks.
    #[test]
    fn a_multi_photograph_request_is_one_extraction_and_one_peak() {
        let stubbed = StubbedExtractor::install(overlap_observing_stub);
        IN_FLIGHT.store(0, Ordering::SeqCst);
        PEAK_IN_FLIGHT.store(0, Ordering::SeqCst);

        let mut request = request(None);
        request.id_image = None;
        request.id_images = Some(vec![
            b"pretend-png-one".to_vec(),
            b"pretend-png-two".to_vec(),
            b"pretend-png-three".to_vec(),
        ]);

        let frozen =
            resolve_identity_for_lease(&request, Some(&paths()), LEASED_BACKEND, LEASED_ORDINAL)
                .expect("the stub extractor answers")
                .embedding
                .expect("a conditioned request resolves an identity");

        assert_eq!(
            stubbed.extractions(),
            1,
            "three photographs are one extraction, not three"
        );
        assert_eq!(
            PEAK_IN_FLIGHT.load(Ordering::SeqCst),
            1,
            "a photograph set must never hold more than one device peak"
        );
        assert_eq!(
            frozen.source_sha256s(),
            [
                mold_core::identity::id_image_sha256(b"pretend-png-one"),
                mold_core::identity::id_image_sha256(b"pretend-png-two"),
                mold_core::identity::id_image_sha256(b"pretend-png-three"),
            ],
            "every photograph is recorded, in request order"
        );
    }

    /// The unconditional identity is computed only for a request that actually
    /// runs the negative branch: an ordinary identity render must pay nothing
    /// for a true-CFG feature it is not using.
    #[test]
    fn the_unconditional_identity_is_resolved_only_for_a_true_cfg_request() {
        let _stubbed = StubbedExtractor::install(stub);
        let plain = resolve_identity_for_lease(
            &request(None),
            Some(&paths()),
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the stub extractor answers")
        .embedding
        .expect("an identity");
        assert!(
            !plain.has_uncond(),
            "an ordinary identity render computes no unconditional half"
        );

        let mut cfg = request(None);
        cfg.true_cfg = Some(2.0);
        let branched =
            resolve_identity_for_lease(&cfg, Some(&paths()), LEASED_BACKEND, LEASED_ORDINAL)
                .expect("the stub extractor answers")
                .embedding
                .expect("an identity");
        assert!(branched.has_uncond());
        assert_ne!(
            plain.fingerprint(),
            branched.fingerprint(),
            "a true-CFG plan must never be mistaken for the plain one"
        );

        // An inert scale is not a true-CFG request.
        let mut inert = request(None);
        inert.true_cfg = Some(1.0);
        assert!(!resolve_identity_for_lease(
            &inert,
            Some(&paths()),
            LEASED_BACKEND,
            LEASED_ORDINAL
        )
        .expect("the stub extractor answers")
        .embedding
        .expect("an identity")
        .has_uncond());
    }

    /// A resolution served from the per-photograph cache — or from a peer
    /// sibling's single flight — is not an extraction, and must not be counted
    /// as one. Without this the once-per-parent counter reports the number of
    /// SIBLINGS, which is the number it exists to distinguish from.
    #[test]
    fn a_resolution_that_computed_nothing_is_not_counted() {
        fn cache_served(
            _: &PulidPaths,
            images: &[Vec<u8>],
            want_uncond: bool,
        ) -> Result<ResolvedIdentity, IdentityFailure> {
            Ok(ResolvedIdentity {
                extracted: false,
                embedding: Some(stub_embedding_for(images, want_uncond)),
                warning: None,
            })
        }
        let stubbed = StubbedExtractor::install(cache_served);
        for _ in 0..4 {
            assert!(resolve_identity_for_lease(
                &request(None),
                Some(&paths()),
                LEASED_BACKEND,
                LEASED_ORDINAL
            )
            .expect("a cached resolution still answers")
            .embedding
            .is_some());
        }
        assert_eq!(
            stubbed.extractions(),
            0,
            "four siblings served from the cache are zero extractions"
        );
    }

    /// Every sibling of a batch derives the SAME cache key, because the key is
    /// a pure function of the photograph bytes each child carries plus the
    /// build's own asset digests — nothing request-scoped reaches it.
    ///
    /// This is why the children need no frozen key handed down from the
    /// parent: content addressing already gives them one, and a second copy
    /// travelling in the plan would be an authority that could disagree with
    /// the bytes it claims to describe. What the parent's preparation cannot
    /// supply — and what the single flight does — is making them compute it
    /// once.
    #[test]
    fn every_batch_sibling_derives_the_same_cache_key() {
        let assets = mold_core::identity::IdentityAssetDigests {
            adapter: "a".repeat(64),
            vision: "v".repeat(64),
            face_detector: "d".repeat(64),
            face_recognizer: "r".repeat(64),
            face_parser: "p".repeat(64),
        };
        let parent = request(None);
        let keys: std::collections::BTreeSet<String> = (1..=4u32)
            .map(|index| {
                let mut child = parent.clone();
                child.batch_id = Some("parent".to_string());
                child.batch_index = Some(index);
                child.batch_count = Some(4);
                child.seed = Some(u64::from(index));
                let photo = mold_core::identity::identity_images(&child)
                    .into_iter()
                    .next()
                    .expect("every child carries the parent's photograph");
                mold_core::identity::identity_cache_key(
                    &mold_core::identity::id_image_sha256(photo),
                    &assets,
                )
            })
            .collect();
        assert_eq!(
            keys.len(),
            1,
            "four siblings must resolve to one cache key, not four"
        );
    }

    /// The parent's pin is the authority for "one identity per parent", and
    /// the per-photograph LRU is only an accelerator underneath it.
    ///
    /// Simulated harder than eviction actually is: the stub returns a
    /// DIFFERENT embedding on every call, which is what a re-extraction on
    /// another GPU after the cache dropped the entry amounts to — the two
    /// agree to the measured 3.82e-5 device tolerance and are not equal. Four
    /// siblings must still render one face.
    #[test]
    fn siblings_share_the_pinned_identity_even_when_the_cache_forgets() {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        fn diverging(
            _: &PulidPaths,
            images: &[Vec<u8>],
            want_uncond: bool,
        ) -> Result<ResolvedIdentity, IdentityFailure> {
            // Every call produces a distinguishable identity, so an equal
            // result can only mean the pin was honoured.
            let nth = CALLS.fetch_add(1, Ordering::SeqCst);
            let values: Vec<f32> = (0..mold_core::identity::ID_EMBEDDING_VALUES)
                .map(|index| (index as f32) + (nth as f32) * 1000.0)
                .collect();
            let sources = images
                .iter()
                .map(|bytes| mold_core::identity::id_image_sha256(bytes))
                .collect();
            let mut embedding = mold_core::identity::FrozenIdentityEmbedding::from_sources(
                &values,
                sources,
                mold_core::identity::IdentityAssetDigests {
                    adapter: "stub-adapter".to_string(),
                    vision: "stub-vision".to_string(),
                    face_detector: "stub-detector".to_string(),
                    face_recognizer: "stub-recognizer".to_string(),
                    face_parser: "stub-parser".to_string(),
                },
            )
            .expect("a well-shaped embedding");
            if want_uncond {
                embedding = embedding.with_uncond(&values).expect("an uncond half");
            }
            Ok(ResolvedIdentity {
                extracted: true,
                embedding: Some(embedding),
                warning: None,
            })
        }

        let stubbed = StubbedExtractor::install(diverging);
        CALLS.store(0, Ordering::SeqCst);
        let pin = crate::execution_plan::IdentityPin::default();
        let request = request(None);

        let fingerprints: std::collections::BTreeSet<String> = (0..4)
            .map(|_| {
                resolve_pinned_identity_for_lease(
                    &request,
                    Some(&paths()),
                    &pin,
                    LEASED_BACKEND,
                    LEASED_ORDINAL,
                )
                .expect("every sibling resolves")
                .embedding
                .expect("an identity")
                .fingerprint()
                .to_string()
            })
            .collect();

        assert_eq!(
            fingerprints.len(),
            1,
            "four siblings must render one face, not four"
        );
        assert_eq!(
            stubbed.extractions(),
            1,
            "only the first sibling may extract"
        );

        // A SECOND parent, interleaved, gets its own cell and its own identity
        // — the pin is per parent, not per process.
        let other = crate::execution_plan::IdentityPin::default();
        let second = resolve_pinned_identity_for_lease(
            &request,
            Some(&paths()),
            &other,
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the second parent resolves")
        .embedding
        .expect("an identity");
        assert!(
            !fingerprints.contains(second.fingerprint()),
            "a second parent must not inherit the first parent's pin"
        );
        assert_eq!(stubbed.extractions(), 2);
    }

    /// The multi-face advisory belongs to the identity, so every sibling that
    /// takes the pin reports it — not just the one that extracted.
    ///
    /// Post-lease preparation leaves `identity_warning` empty for a new parent,
    /// so a child reading the pin has no other source for it. Three of four
    /// prints from a group photograph silently losing "the largest one was
    /// used" is a worse failure than it looks: the caller cannot tell which
    /// prints the note applied to.
    #[test]
    fn every_sibling_reports_the_advisory_the_extraction_produced() {
        let stubbed = StubbedExtractor::install(stub_with_warning);
        let pin = crate::execution_plan::IdentityPin::default();
        let request = request(None);

        let mut advisories = Vec::new();
        for _ in 0..4 {
            let resolved = resolve_pinned_identity_for_lease(
                &request,
                Some(&paths()),
                &pin,
                LEASED_BACKEND,
                LEASED_ORDINAL,
            )
            .expect("every sibling resolves");
            advisories.push(resolved.warning);
        }

        assert_eq!(stubbed.extractions(), 1, "only the first sibling extracts");
        for (index, advisory) in advisories.iter().enumerate() {
            let advisory = advisory
                .as_deref()
                .unwrap_or_else(|| panic!("sibling {index} lost the advisory"));
            assert!(advisory.contains("faces were detected"), "{advisory}");
            assert!(advisory.contains("largest"), "{advisory}");
        }
        assert!(
            advisories.windows(2).all(|pair| pair[0] == pair[1]),
            "every sibling must report the SAME advisory: {advisories:?}"
        );
    }

    /// A pin hit is not an extraction, and must not be reported as a phase.
    #[test]
    fn a_pinned_sibling_reports_no_extraction() {
        let stubbed = StubbedExtractor::install(stub);
        let pin = crate::execution_plan::IdentityPin::default();
        let request = request(None);

        let first = resolve_pinned_identity_for_lease(
            &request,
            Some(&paths()),
            &pin,
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the first sibling extracts");
        assert!(first.extracted);

        let second = resolve_pinned_identity_for_lease(
            &request,
            Some(&paths()),
            &pin,
            LEASED_BACKEND,
            LEASED_ORDINAL,
        )
        .expect("the second sibling is pinned");
        assert!(
            !second.extracted,
            "a pinned sibling must not be counted or timed as an extraction"
        );
        assert_eq!(stubbed.extractions(), 1);
        assert_eq!(
            first.embedding.map(|e| e.fingerprint().to_string()),
            second.embedding.map(|e| e.fingerprint().to_string())
        );
    }

    /// The charged peaks are the measurements, not second numbers that can
    /// drift away from them.
    ///
    /// The host figure is still the one `mold_inference` re-exports and
    /// derives from the artifacts' pinned sizes; the device figure is the one
    /// `memory_preflight` adds to a conditioned request's peak estimate, and
    /// `pulid_device_parity.rs` measures it live on CUDA.
    #[cfg(feature = "pulid")]
    #[test]
    fn the_charged_peaks_match_their_measurements() {
        assert_eq!(
            mold_core::identity::EXTRACTION_HOST_PEAK_BYTES,
            mold_inference::identity::extraction::EXTRACTION_HOST_PEAK_BYTES
        );
        assert_eq!(
            crate::memory_preflight::IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES,
            mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES
        );
    }

    /// The gate an extraction is admitted under is the frozen plan's, not a
    /// second one this module owns. #1227 phase 2 retired `ExtractionSlot`'s
    /// semaphore-plus-`ram_snapshot()` because two admission-time memory gates
    /// that can disagree is worse than one; this is the structural check that
    /// it stayed retired.
    #[test]
    fn no_second_memory_gate_lives_in_this_module() {
        // The production half, minus its prose: the module doc explains the
        // retirement by naming what was retired, and a scan that read its own
        // explanation — or this test — would always fail.
        let source = include_str!("identity_extraction.rs");
        let production = source
            .split_once("#[cfg(test)]\nmod tests {")
            .map(|(production, _)| production)
            .expect("this module has a test module");
        let code: String = production
            .lines()
            .map(str::trim_start)
            .filter(|line| !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in ["Semaphore", "ram_snapshot", "available_host_bytes"] {
            assert!(
                !code.contains(banned),
                "`{banned}` is back: the lease's ledger is the one memory gate"
            );
        }
    }
}
