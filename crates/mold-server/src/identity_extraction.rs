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
//!
//! ## Why extractions do not overlap EACH OTHER
//!
//! That argument covers the encoders and says nothing about a second identity
//! request arriving while the first is extracting. Admission is concurrent —
//! two clients, or one client's `/api/generate` racing a prepared batch's
//! `freeze_batch_plan` — and two extractions peak at ~2.8 GB together, which
//! is enough to OOM a 8 GB box that was going to run both renders fine.
//!
//! [`ExtractionSlot`] is that gate. It is deliberately NOT the scheduler's
//! `HostMemoryLedger`: the ledger is private state owned by the planner task
//! and keyed on leases, and extraction happens strictly BEFORE this job has a
//! lease — it is one of the inputs to the plan the ledger will later reserve
//! against. Re-entering the planner from admission to charge a transient CPU
//! peak would invert that ownership. The slot instead does the two things the
//! requirement actually asks for: serialize (so peaks never sum) and refuse
//! (so a host that cannot fit even one says so instead of dying).

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

/// Serializes identity extractions and charges each one against real host
/// headroom.
///
/// One permit, because there is no throughput argument for two: an extraction
/// is CPU-bound for a second or more and holds
/// [`EXTRACTION_HOST_PEAK_BYTES`], so running two concurrently doubles the
/// peak to buy nothing. Queuing is the correct behaviour, not a compromise.
static EXTRACTION_SLOT: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);

/// Bytes one extraction is charged.
///
/// Restated here rather than read from `mold_inference` because that module is
/// behind the `pulid` feature while this gate is not; a build without the
/// feature still admits and refuses requests. `the_charged_peak_matches_the_
/// measurement` pins the two together on the builds that have both.
const EXTRACTION_PEAK_BYTES: u64 = 1_400_000_000;

/// Reads host memory. Injectable so the gate can be tested without a machine
/// that happens to be short on RAM.
#[cfg(test)]
static TEST_AVAILABLE_HOST_BYTES: AtomicU64 = AtomicU64::new(u64::MAX);

fn available_host_bytes() -> Option<u64> {
    #[cfg(test)]
    {
        let forced = TEST_AVAILABLE_HOST_BYTES.load(Ordering::SeqCst);
        if forced != u64::MAX {
            return Some(forced);
        }
    }
    let ram = crate::resources::ram_snapshot();
    // Absent means unknown, and unknown must not refuse: a container that
    // cannot report `MemAvailable` would otherwise lose face identity
    // entirely. The serialization above still holds there.
    ram.available
}

/// Held for the duration of one extraction; releasing it admits the next.
struct ExtractionSlot(#[allow(dead_code)] tokio::sync::SemaphorePermit<'static>);

impl ExtractionSlot {
    /// Wait for the slot, then charge the peak against a FRESH reading.
    ///
    /// The order is the whole point. Sampling first would read headroom that a
    /// peer extraction is about to consume; sampling after the permit is held
    /// means this process has at most one extraction outstanding and the
    /// reading is about this one.
    async fn acquire() -> Result<Self, String> {
        let permit = EXTRACTION_SLOT
            .acquire()
            .await
            .map_err(|_| "the identity extraction slot was closed".to_string())?;
        if let Some(available) = available_host_bytes() {
            if available < EXTRACTION_PEAK_BYTES {
                return Err(format!(
                    "not enough host memory to extract a face identity: this needs about \
                     {needed} MB and {available} MB is available. Free memory or retry when \
                     the host is less busy.",
                    needed = EXTRACTION_PEAK_BYTES / 1_000_000,
                    available = available / 1_000_000,
                ));
            }
        }
        Ok(Self(permit))
    }
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
pub(crate) struct ResolvedIdentity {
    /// `None` for every request that does not condition on a face.
    pub(crate) embedding: Option<FrozenIdentityEmbedding>,
    /// A caller-facing advisory the extraction produced — today only "several
    /// faces were found, the largest was used".
    ///
    /// It travels with the embedding rather than being logged and forgotten:
    /// the person who supplied a group photograph is the one who needs to know
    /// which face was picked, and a `tracing::warn!` on the server reaches
    /// nobody holding a CLI or a browser.
    pub(crate) warning: Option<String>,
}

/// Extract and freeze the identity for one parent request.
///
/// Returns an empty [`ResolvedIdentity`] for every request that does not
/// condition on a face. A request that does, but whose bundle did not resolve,
/// is an error: a print that renders without the face nobody would be told
/// about is exactly what `mold_core::identity` refuses at the contract
/// boundary.
pub(crate) async fn resolve_identity_embedding(
    request: &GenerateRequest,
    paths: Option<&PulidPaths>,
) -> Result<ResolvedIdentity, String> {
    if !request_resolves_identity(request) {
        return Ok(ResolvedIdentity::default());
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

    // Held across the whole extraction, including the stubbed path, so the
    // serialization is the same thing the tests observe.
    let _slot = ExtractionSlot::acquire().await?;

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
async fn extract_blocking(paths: PulidPaths, image: Vec<u8>) -> Result<ResolvedIdentity, String> {
    let outcome = tokio::task::spawn_blocking(move || {
        mold_inference::identity::extraction::extract_identity_embedding(&paths, &image)
    })
    .await
    .map_err(|error| format!("the identity extractor panicked: {error}"))?
    .map_err(|error| format!("{error:#}"))?;

    if let Some(warning) = &outcome.warning {
        tracing::warn!(target: "mold::identity", "{warning}");
    }
    Ok(ResolvedIdentity {
        embedding: Some(outcome.embedding),
        warning: outcome.warning,
    })
}

/// A build without `pulid` never reaches here: the request contract refuses
/// every identity request before admission. The arm exists so the module
/// compiles in the default feature set rather than being conditionally wired
/// into `prepare_inputs_for_devices`, which would be a second place the
/// lifetime could diverge.
#[cfg(not(feature = "pulid"))]
async fn extract_blocking(_paths: PulidPaths, _image: Vec<u8>) -> Result<ResolvedIdentity, String> {
    Err(mold_core::identity::IDENTITY_BUILD_UNSUPPORTED.to_string())
}

#[cfg(test)]
type Stub = fn(&PulidPaths, &[u8]) -> Result<ResolvedIdentity, String>;

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

    fn stub(_: &PulidPaths, image: &[u8]) -> Result<ResolvedIdentity, String> {
        Ok(ResolvedIdentity {
            embedding: Some(stub_embedding(image)),
            warning: None,
        })
    }

    /// The multi-face advisory the real extractor emits, so the plumbing that
    /// carries it out of admission can be tested without a group photograph.
    fn stub_with_warning(_: &PulidPaths, image: &[u8]) -> Result<ResolvedIdentity, String> {
        Ok(ResolvedIdentity {
            embedding: Some(stub_embedding(image)),
            warning: Some(
                "3 faces were detected in the identity image; conditioning on the largest one"
                    .to_string(),
            ),
        })
    }

    /// The zero-weight rule, at the one gate that could break it: no
    /// extraction is even counted, let alone performed.
    #[tokio::test]
    async fn a_zero_weight_request_performs_no_extraction() {
        let stubbed = StubbedExtractor::install(stub);
        let resolved = resolve_identity_embedding(&request(Some(0.0)), Some(&paths()))
            .await
            .expect("a zero weight is inert, not an error");
        assert!(resolved.embedding.is_none());
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
        assert!(resolved.embedding.is_none());
        assert_eq!(stubbed.extractions(), 0);
    }

    #[tokio::test]
    async fn a_conditioned_request_extracts_exactly_once() {
        let stubbed = StubbedExtractor::install(stub);
        let frozen = resolve_identity_embedding(&request(None), Some(&paths()))
            .await
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

    /// Forces the host-memory reading for the duration of a test, and restores
    /// it afterwards even on panic.
    struct ForcedHostMemory;

    impl ForcedHostMemory {
        fn available(bytes: u64) -> Self {
            TEST_AVAILABLE_HOST_BYTES.store(bytes, Ordering::SeqCst);
            Self
        }
    }

    impl Drop for ForcedHostMemory {
        fn drop(&mut self) {
            TEST_AVAILABLE_HOST_BYTES.store(u64::MAX, Ordering::SeqCst);
        }
    }

    /// Counts how many stubbed extractions are in flight at once, and the
    /// highest that number ever reached.
    static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
    static PEAK_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

    fn overlap_observing_stub(
        _: &PulidPaths,
        image: &[u8],
    ) -> Result<ResolvedIdentity, String> {
        let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
        PEAK_IN_FLIGHT.fetch_max(now, Ordering::SeqCst);
        // Long enough that a genuinely concurrent peer would be observed.
        std::thread::sleep(std::time::Duration::from_millis(60));
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        Ok(ResolvedIdentity {
            embedding: Some(stub_embedding(image)),
            warning: None,
        })
    }

    /// Two admissions arriving together must not both allocate the ~1.4 GB
    /// peak. A host with room for exactly one extraction has to serve them one
    /// after the other; summing the peaks is how an otherwise-fine box OOMs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_identity_preparations_serialize_on_the_extraction_slot() {
        let stubbed = StubbedExtractor::install(overlap_observing_stub);
        // Room for one peak, not two — the ledger sized below two peaks.
        let _memory = ForcedHostMemory::available(EXTRACTION_PEAK_BYTES + 200_000_000);
        IN_FLIGHT.store(0, Ordering::SeqCst);
        PEAK_IN_FLIGHT.store(0, Ordering::SeqCst);

        let (one, two) = (request(None), request(None));
        let paths = paths();
        let first = resolve_identity_embedding(&one, Some(&paths));
        let second = resolve_identity_embedding(&two, Some(&paths));
        let (first, second) = tokio::join!(first, second);

        assert!(first.expect("first extraction").embedding.is_some());
        assert!(
            second
                .expect("the queued extraction still runs")
                .embedding
                .is_some(),
            "serializing must queue the second request, not refuse it"
        );
        assert_eq!(stubbed.extractions(), 2);
        assert_eq!(
            PEAK_IN_FLIGHT.load(Ordering::SeqCst),
            1,
            "two extractions must never hold their host peaks at the same time"
        );
    }

    /// A host that cannot fit even one extraction is told so, rather than
    /// being taken down by the allocation.
    #[tokio::test]
    async fn a_host_without_room_for_one_extraction_refuses_instead_of_allocating() {
        let stubbed = StubbedExtractor::install(stub);
        let _memory = ForcedHostMemory::available(EXTRACTION_PEAK_BYTES / 2);

        let error = resolve_identity_embedding(&request(None), Some(&paths()))
            .await
            .expect_err("a host with half the peak available cannot extract");
        assert!(error.contains("not enough host memory"), "{error}");
        assert_eq!(
            stubbed.extractions(),
            0,
            "the refusal must happen before the extractor is entered"
        );
    }

    /// Unknown is not empty. A container that cannot report `MemAvailable`
    /// must keep face identity rather than losing it to a missing reading.
    #[tokio::test]
    async fn an_unreadable_host_reading_does_not_refuse() {
        let stubbed = StubbedExtractor::install(stub);
        let _memory = ForcedHostMemory::available(u64::MAX);

        assert!(resolve_identity_embedding(&request(None), Some(&paths()))
            .await
            .expect("an unknown reading is not a refusal")
            .embedding
            .is_some());
        assert_eq!(stubbed.extractions(), 1);
    }

    /// The extractor's own advisory — which of several faces it picked —
    /// must leave this module with the embedding. Logging it server-side, as
    /// it was before #1223, reaches nobody holding a CLI or a browser.
    #[tokio::test]
    async fn the_multi_face_advisory_travels_with_the_identity() {
        let _stubbed = StubbedExtractor::install(stub_with_warning);
        let resolved = resolve_identity_embedding(&request(None), Some(&paths()))
            .await
            .expect("the stub extractor answers");

        assert!(resolved.embedding.is_some());
        let warning = resolved.warning.expect("an unforced face choice is reported");
        assert!(warning.contains("faces were detected"), "{warning}");
        assert!(warning.contains("largest"), "{warning}");
    }

    /// A single-face photograph is the ordinary case and says nothing.
    #[tokio::test]
    async fn an_unambiguous_identity_carries_no_advisory() {
        let _stubbed = StubbedExtractor::install(stub);
        let resolved = resolve_identity_embedding(&request(None), Some(&paths()))
            .await
            .expect("the stub extractor answers");
        assert!(resolved.embedding.is_some());
        assert!(resolved.warning.is_none());
    }

    /// The charged peak is the measurement, not a second number that can
    /// drift away from it.
    #[cfg(feature = "pulid")]
    #[test]
    fn the_charged_peak_matches_the_measurement() {
        assert_eq!(
            EXTRACTION_PEAK_BYTES,
            mold_inference::identity::extraction::EXTRACTION_HOST_PEAK_BYTES
        );
    }
}
