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
) -> Result<ResolvedIdentity, String> {
    if !request_resolves_identity(request) {
        return Ok(ResolvedIdentity::default());
    }
    // Either wire shape, in request order. The photographs are processed
    // serially, and each one hits or misses the per-photograph cache
    // independently.
    let images: Vec<Vec<u8>> = mold_core::identity::identity_images(request)
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    if images.is_empty() {
        return Err(
            "id_image (or id_images) is required when id_weight, id_start_step, id_image_name, \
             or id_image_names is set"
                .to_string(),
        );
    }
    // The unconditional identity is computed only for a request that actually
    // runs the true-CFG negative branch; an ordinary identity render pays
    // nothing for it.
    let want_uncond = mold_core::identity::request_uses_true_cfg(request);
    let Some(paths) = paths.cloned() else {
        return Err(
            "this request asks for face-identity conditioning but no PuLID bundle resolved on \
             any eligible device; run `mold pull pulid-flux --accept-license \
             insightface-antelopev2`"
                .to_string(),
        );
    };

    #[cfg(test)]
    let resolved = match test_stub() {
        Some(stub) => stub(&paths, &images, want_uncond)?,
        None => extract_on_device(paths, images, want_uncond, backend, ordinal)?,
    };
    #[cfg(not(test))]
    let resolved = extract_on_device(paths, images, want_uncond, backend, ordinal)?;

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
    images: Vec<Vec<u8>>,
    want_uncond: bool,
    backend: mold_core::GpuBackend,
    ordinal: usize,
) -> Result<ResolvedIdentity, String> {
    use mold_inference::identity::extraction::ExtractionPlacement;

    let borrowed: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let outcome = mold_inference::identity::extraction::extract_identity_embeddings_at(
        &paths,
        &borrowed,
        want_uncond,
        ExtractionPlacement::Gpu { backend, ordinal },
    )
    .map_err(|error| format!("{error:#}"))?;

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
    _images: Vec<Vec<u8>>,
    _want_uncond: bool,
    _backend: mold_core::GpuBackend,
    _ordinal: usize,
) -> Result<ResolvedIdentity, String> {
    Err(mold_core::identity::IDENTITY_BUILD_UNSUPPORTED.to_string())
}

#[cfg(test)]
type Stub = fn(&PulidPaths, &[Vec<u8>], bool) -> Result<ResolvedIdentity, String>;

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
            face_parser_source: "/models/shared/pulid/parsing_bisenet.pth".into(),
        }
    }

    fn stub(
        _: &PulidPaths,
        images: &[Vec<u8>],
        want_uncond: bool,
    ) -> Result<ResolvedIdentity, String> {
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
    ) -> Result<ResolvedIdentity, String> {
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
        assert!(error.contains("mold pull pulid-flux"), "{error}");
        assert!(error.contains("insightface-antelopev2"), "{error}");
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
    ) -> Result<ResolvedIdentity, String> {
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
        ) -> Result<ResolvedIdentity, String> {
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
