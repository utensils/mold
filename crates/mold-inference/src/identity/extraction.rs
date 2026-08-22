//! One identity image in, one frozen `[1, 32, 2048]` embedding out.
//!
//! This is the composer #1223 needed and nothing else had: #1222 delivered the
//! face stack (SCRFD → ArcFace → the 512x512 EVA crop), #1229 delivered the
//! EVA02-CLIP tower and the IDFormer, and #1221 delivered the adapter that
//! consumes the result. Each of those was reachable only from its own tests.
//! Here they are run in order, once, and the answer is frozen as plain data.
//!
//! ```text
//!   id_image bytes
//!         |  IdentityExtractor (candle-onnx, CPU)
//!         v
//!   arcface [1, 512] (raw)      eva_crop_512 (RGB)
//!         |                             |  BiSeNetParser -> labels
//!         |                             |  background -> white, face -> grey
//!         |                             |  eva_clip_preprocess (bicubic 336)
//!         |                             v
//!         |                     EvaClipVisionTower
//!         |                       /            \
//!         |     5 x [1, 577, 1024]              [1, 768] L2-normalized
//!         |              |                            |
//!         +--------------|------------- cat ----------+
//!                        |                     [1, 1280]
//!                        v
//!                 IdFormer -> [1, 32, 2048] -> FrozenIdentityEmbedding
//! ```
//!
//! **Everything here takes a device**, as of #1227 phase 2. It used to be
//! hardcoded to `Device::Cpu` for two reasons that have both expired:
//! `candle-onnx` could not place a tensor anywhere else (phase 1 replaced it
//! with resident candle ports, `identity/scrfd_net.rs` and
//! `identity/arcface_net.rs`), and the extraction ran at *admission*, before
//! the scheduler had leased anything, so there was no device to name. Phase 2
//! moved the call site inside the leased job — first, before prompt encode —
//! which is what `docs/architecture/pulid-perf.md` §5 designs and why
//! [`ExtractionPlacement`] exists.
//!
//! What has NOT changed is the drop-and-reload discipline, and it is now
//! load-bearing in a stronger way: the tower, the parser, the two face
//! networks, and the IDFormer are built, forwarded, and fully released BEFORE
//! `flux::identity::EngineIdentityState` begins the adapter's residency for
//! that dispatch. The extraction is a strictly earlier, strictly disjoint
//! phase of the same lease, never a third resident beside the ~1.14 GB adapter
//! and the transformer's own peak. The result is still a device-independent
//! 256 KiB value, which is what lets one extraction serve every sibling of a
//! batch on every device it fans out to.
//!
//! Peak host RAM is the EVA tower — its 609 MB f16 file read into a private
//! buffer, materialized at [`eva_working_dtype`]'s working dtype, plus
//! activations — beside the two ONNX graphs (~278 MB); see
//! [`EXTRACTION_HOST_PEAK_BYTES`]. On a device placement most of that becomes
//! device memory instead, charged by
//! `memory_preflight::IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES`; the private
//! authenticated copy stays on the host either way, because that is what the
//! `VarBuilder` reads from.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use mold_core::identity::{FrozenIdentityEmbedding, IdentityAssetDigests, IdentityFamily};
use mold_core::manifest::ModelComponent;
use mold_core::pulid_assets::PulidPaths;

use crate::encoders::eva_clip_preprocess::{planar_rgb_from_image, preprocess_planar_rgb};
use crate::encoders::eva_clip_vision::{
    EvaClipVisionTower, EMBED_DIM, HIDDEN_STATE_BLOCKS, PROJECTION_DIM, SEQUENCE_LEN,
};
use crate::encoders::pickle_convert::{
    ensure_bisenet_parser_safetensors, ensure_eva_clip_vision_safetensors, BISENET_DERIVED_SHA256,
    EVA_DERIVED_SHA256,
};
use crate::flux::pulid_encoder::IdFormer;

use super::parsing::{apply_pulid_face_mask, BiSeNetParser};
use super::{IdentityError, IdentityExtractor};

/// Host bytes one extraction peaks at, measured on the shipped artifacts.
///
/// Re-exported from [`mold_core::identity`], which is the one authority: the
/// admission gate in `mold-server` charges this number and is compiled WITHOUT
/// the `pulid` feature, so it cannot read a constant that lives here. It used
/// to restate the literal, and the two drifted by a gigabyte the moment this
/// measurement moved — an admission gate under-charging by that much is a host
/// that OOMs instead of queuing.
///
/// The measurement itself is documented on the core constant, and
/// `the_charged_host_peak_covers_every_stage_of_the_extraction` below re-derives
/// it from the artifacts' own pinned sizes, which only this crate can see.
pub use mold_core::identity::EXTRACTION_HOST_PEAK_BYTES;

/// Host bytes each ADDITIONAL identity photograph adds to that peak.
///
/// A multi-photograph extraction does not multiply the peak, because the
/// expensive halves are still built once and dropped in sequence: the tower is
/// loaded once and run N times, then dropped, and the IDFormer is built once
/// afterwards. The only thing that scales with N is the five retained
/// `[1, 577, 1024]` f32 hidden states and the `[1, 768]` projection each
/// photograph contributes, which the IDFormer needs after the tower is gone.
///
/// `5 * 577 * 1024 * 4` = 11,816,960 bytes, plus the projection, rounded up.
///
/// Re-exported from `mold_core::identity` for the same reason
/// [`EXTRACTION_HOST_PEAK_BYTES`] is: `mold_core` composes the device charge's
/// multi-photograph allowance from it and is compiled without the `pulid`
/// feature, so it cannot read a constant that lives here. A second literal
/// there and here is what let the host peak drift by a gigabyte.
/// `the_retained_per_image_charge_covers_the_towers_own_shape` re-derives it
/// from the tower's geometry, which only this crate can see.
pub use mold_core::identity::EXTRACTION_RETAINED_BYTES_PER_IMAGE;

/// Where one extraction runs.
///
/// The server names a placement, never a `candle_core::Device`: `mold-server`
/// deliberately does not depend on candle (CLAUDE.md's crate-boundary rule),
/// and the device a leased job owns is identified by backend and ordinal
/// everywhere else in the scheduler. Resolution goes through
/// [`crate::device::metal_device`] and friends so a Metal ordinal reuses this
/// process's existing device rather than minting a second one with a split
/// identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionPlacement {
    /// The host. Every CPU-only build, the `--local` CLI on a CPU device, and
    /// any job whose lease is not a GPU.
    Host,
    /// The render's own leased GPU (#1227 phase 2).
    Gpu {
        backend: mold_core::GpuBackend,
        ordinal: usize,
    },
}

impl ExtractionPlacement {
    /// Open the device this placement names.
    ///
    /// Never falls back: a placement that cannot be opened is an error the
    /// caller must propagate, exactly as
    /// `device::create_exact_gpu_device` refuses to substitute a backend for
    /// an admitted plan's. Silently demoting to the host would run the
    /// extraction against a memory grant that was made for the GPU.
    pub fn device(&self) -> Result<Device> {
        match self {
            Self::Host => Ok(Device::Cpu),
            Self::Gpu {
                backend: mold_core::GpuBackend::Metal,
                ordinal,
            } => crate::device::metal_device(*ordinal),
            Self::Gpu {
                backend: mold_core::GpuBackend::Cuda,
                ordinal,
            } => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(*ordinal).map_err(|error| {
                        anyhow::anyhow!(
                            "failed to open CUDA device {ordinal} for face extraction: {error}"
                        )
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!(
                        "face extraction was placed on CUDA device {ordinal} but this build has \
                         no CUDA support"
                    )
                }
            }
        }
    }
}

/// Extract, average, compose, and freeze the identity for one request on the
/// device a placement names.
///
/// The entry point the server calls once phase 2 moved extraction inside the
/// leased job. [`extract_identity_embeddings`] is the same thing with an
/// already-opened device, which is what the CLI and the benchmark hold.
pub fn extract_identity_embeddings_at(
    paths: &PulidPaths,
    images: &[&[u8]],
    want_uncond: bool,
    placement: ExtractionPlacement,
) -> std::result::Result<IdentityExtraction, IdentityError> {
    let device = placement
        .device()
        .map_err(|error| IdentityError::Runtime(error.context("placing the face extraction")))?;
    extract_identity_embeddings(paths, images, want_uncond, &device)
}

/// One photograph's cached identity.
#[derive(Clone, Debug)]
struct CachedIdentity {
    /// The final `[1, 32, 2048]` tokens, flattened. 256 KiB.
    tokens: std::sync::Arc<Vec<f32>>,
    /// The advisory that photograph produced, verbatim and unpositioned — a
    /// hit must report "several faces were found" exactly as the extraction
    /// that produced it did, and the "photo N of M" prefix is applied by the
    /// caller because it belongs to the request, not to the photograph.
    warning: Option<String>,
}

/// How many photographs' identities this process keeps.
///
/// `docs/architecture/pulid-perf.md` §2 sizes this for the SESSION's hot set —
/// a batch's siblings, a sequence's per-clip identity, a `Prepare N
/// variations` review, an interactive retry — never for a photo library. Each
/// entry is `ID_EMBEDDING_VALUES` f32 (256 KiB) plus a short warning and a
/// 64-character key, so sixteen of them is about **4.2 MB**: an entry-count cap
/// alone is sufficient because the entry size is fixed, and a byte budget would
/// be complexity buying nothing.
const IDENTITY_CACHE_ENTRIES: usize = 16;

/// The per-photograph identity cache, and the degenerate one-entry memo for
/// the unconditional identity beside it.
///
/// In process, never on disk. `pulid-perf.md` §2 recommends against
/// persistence and the reason is not performance: these values are a biometric
/// derivative, which is why [`FrozenIdentityEmbedding`]'s own `Debug` redacts
/// them, and putting them at rest would introduce a retention-and-deletion
/// story — how long, who may read it, what `mold rm pulid-flux` does to it —
/// that nothing in this codebase's posture toward these bytes anticipates.
/// Recomputation is cheap enough to make that trade obvious.
///
/// A plain `Vec` in most-recent-first order rather than a crate: at sixteen
/// entries a linear scan is faster than a hash lookup and there is nothing to
/// tune.
type IdentityCache = std::sync::Mutex<Vec<(String, CachedIdentity)>>;

fn identity_cache() -> &'static IdentityCache {
    static CACHE: std::sync::OnceLock<IdentityCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// The unconditional identity is a pure function of the adapter checkpoint —
/// `PuLID/pulid/pipeline_flux.py:188-192` runs the IDFormer over
/// `zeros_like(id_cond)` and zeroed hidden states, which depend on no
/// photograph at all. So it is memoized on the adapter digest and the pipeline
/// version, NOT held as an LRU entry under a sentinel key: it is computed once
/// per process and never needs eviction.
///
/// It matters more than its own forward pass costs. Without it, a true-CFG
/// request whose photograph IS cached would still have to open and materialize
/// the 605 MB `pulid_encoder.*` half of the adapter just to produce a tensor
/// that never varies.
type UncondMemo = std::sync::Mutex<Option<(String, std::sync::Arc<Vec<f32>>)>>;

fn uncond_memo() -> &'static UncondMemo {
    static MEMO: std::sync::OnceLock<UncondMemo> = std::sync::OnceLock::new();
    MEMO.get_or_init(|| std::sync::Mutex::new(None))
}

static IDENTITY_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Photographs served from the cache since this process started.
///
/// Exported for the tests that state the contract: a second render of the same
/// face must not re-run a 300 GFLOP tower.
pub fn identity_cache_hit_count() -> u64 {
    IDENTITY_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Empty the cache and the uncond memo. Test-only: both are process-global, so
/// a test that wants to observe a miss has to start from nothing.
pub fn forget_cached_identities() {
    identity_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    *uncond_memo()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

fn cache_get(key: &str) -> Option<CachedIdentity> {
    let mut cache = identity_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let index = cache.iter().position(|(held, _)| held == key)?;
    // Most-recent-first: a hit is promoted, so the cap evicts the least
    // recently USED entry rather than the oldest inserted one.
    let entry = cache.remove(index);
    let value = entry.1.clone();
    cache.insert(0, entry);
    IDENTITY_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(value)
}

fn cache_put(key: String, value: CachedIdentity) {
    let mut cache = identity_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.retain(|(held, _)| held != &key);
    cache.insert(0, (key, value));
    cache.truncate(IDENTITY_CACHE_ENTRIES);
}

/// Every asset digest this build's extraction will record, known before any of
/// them is opened.
///
/// The adapter is family-specific, so the digest that enters the key is the
/// family's own manifest pin — which is what keeps a FLUX identity and an SDXL
/// identity for the SAME photograph two different cache entries. They are
/// different IDFormers producing genuinely different tensors; a key that
/// ignored the family would serve one render the other's face embedding.
///
/// A cache key needs the digests BEFORE the extraction runs, which is why they
/// are resolved from the manifest pins and compiled-in constants here rather
/// than from an [`IdentityAssetDigests`] a request has not produced yet. The
/// two cannot disagree: the loaders refuse any file whose bytes do not hash to
/// the pin, so a post-load digest that differed from this one would have
/// failed the load instead of being recorded.
pub fn pinned_asset_digests(family: IdentityFamily) -> IdentityAssetDigests {
    IdentityAssetDigests {
        adapter: adapter_sha256(family),
        vision: EVA_DERIVED_SHA256.to_string(),
        face_detector: super::onnx_graph::pinned_artifact(ModelComponent::FaceDetector)
            .map(|pin| pin.sha256.to_string())
            .unwrap_or_else(|| "unpinned".to_string()),
        face_recognizer: super::onnx_graph::pinned_artifact(ModelComponent::FaceRecognizer)
            .map(|pin| pin.sha256.to_string())
            .unwrap_or_else(|| "unpinned".to_string()),
        face_parser: BISENET_DERIVED_SHA256.to_string(),
    }
}

/// One lock per cache key, so concurrent callers asking for the SAME
/// photograph compute it once and every other one takes that result.
///
/// The cache alone is not enough and the gap is not theoretical. A
/// `batch_size = 4` parent is prepared before fan-out, its children are
/// dispatched to several GPU worker threads at once, and each child resolves
/// its own identity — so four threads can all miss a cold cache in the window
/// between `cache_get` and `cache_put`, run the whole 300 GFLOP stack four
/// times, and, because they are on different devices, end up with four
/// embeddings that differ at the measured 3.82e-5 device tolerance. Four
/// siblings of one print conditioned on four slightly different faces is not a
/// performance bug.
///
/// So the miss path is single-flight: the first caller holds the key's lock and
/// computes, every other caller BLOCKS on that lock and then re-reads the
/// cache, taking the tokens the winner stored. "One extraction per parent" is
/// therefore true by construction rather than by preparation ordering, and the
/// siblings' embeddings are bit-identical rather than merely equivalent.
///
/// Blocking rather than async because the callers are GPU owner threads, which
/// is also why this is a `std::sync::Mutex`: there is no reactor to yield to,
/// and an extraction holds the device anyway.
///
/// **Locks are always acquired in sorted key order.** A multi-photograph set
/// takes several at once, and two requests sharing a subset of photographs in
/// different orders would otherwise deadlock.
///
/// A failed flight stores nothing, so the key is simply released and the next
/// caller — or a retry — computes it for real. There is deliberately no
/// negative caching: an extraction failure is a torn file or an absent face,
/// both of which a later attempt may legitimately see differently.
type FlightRegistry =
    std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<std::sync::Mutex<()>>>>;

fn flight_registry() -> &'static FlightRegistry {
    static FLIGHTS: std::sync::OnceLock<FlightRegistry> = std::sync::OnceLock::new();
    FLIGHTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// The flight lock for every key in `keys`, deduplicated and SORTED.
fn flights_for(keys: &[String]) -> Vec<std::sync::Arc<std::sync::Mutex<()>>> {
    let mut ordered: Vec<&String> = keys.iter().collect();
    ordered.sort();
    ordered.dedup();
    let mut registry = flight_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    ordered
        .into_iter()
        .map(|key| {
            registry
                .entry(key.clone())
                .or_insert_with(|| std::sync::Arc::new(std::sync::Mutex::new(())))
                .clone()
        })
        .collect()
}

/// Drop registry entries nobody is holding any more.
///
/// Called after the guards AND the local `Arc`s are gone, so a `strong_count`
/// of one means the registry is the only owner and no peer is waiting on it.
/// Skipping this would leak one mutex per distinct photograph for the life of
/// the process.
fn release_flights(keys: &[String]) {
    let mut registry = flight_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for key in keys {
        if registry
            .get(key)
            .is_some_and(|held| std::sync::Arc::strong_count(held) == 1)
        {
            registry.remove(key);
        }
    }
}

/// The flight key for the unconditional identity.
///
/// Not a cache key and never an LRU entry — `pulid-perf.md` §2 is explicit that
/// the uncond is a degenerate memo rather than a keyed entry. This is only a
/// lock name, and it exists for the same reason the per-photograph flights do:
/// the uncond rides the frozen fingerprint through `with_uncond`, so two
/// siblings computing it on two devices would disagree there too.
fn uncond_flight_key(adapter_sha256: &str) -> String {
    format!("mold.identity.uncond.flight/{adapter_sha256}")
}

static IDENTITY_EXTRACTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Photographs (and unconditional identities) this process has actually
/// composed, as opposed to served from the cache.
///
/// This is the counter that makes "exactly once per parent" checkable now that
/// resolution happens per sibling inside each lease: a `batch_size = 4` parent
/// across two devices must move it by exactly one.
pub fn identity_extraction_count() -> u64 {
    IDENTITY_EXTRACTIONS.load(std::sync::atomic::Ordering::Relaxed)
}

/// What one extraction produced.
#[derive(Debug)]
pub struct IdentityExtraction {
    /// The immutable identity every sibling of the parent request reuses.
    pub embedding: FrozenIdentityEmbedding,
    /// A caller-surfacable advisory — today only "several faces were found".
    /// Travels to the client as `x-mold-request-warning`.
    pub warning: Option<String>,
    /// Whether anything was actually computed, or whether every photograph
    /// (and the unconditional identity, when asked for) came from the cache.
    ///
    /// The caller needs this for two things it would otherwise get wrong. The
    /// once-per-parent counter must not count a sibling that waited on a
    /// flight and took the winner's tokens; and
    /// `ProgressPhase::IdentityExtract` must not teach the scheduler that this
    /// phase takes two milliseconds, which is what a cache hit costs and what
    /// would drag `ewma_identity_extract_ms` to a figure no cold request can
    /// meet.
    pub extracted: bool,
}

/// Extract, compose, and freeze the identity for one request.
///
/// Called EXACTLY ONCE per parent request, at admission, before batch fan-out.
/// Everything it loads is released before it returns.
pub fn extract_identity_embedding(
    paths: &PulidPaths,
    image_bytes: &[u8],
    device: &Device,
) -> std::result::Result<IdentityExtraction, IdentityError> {
    extract_identity_embeddings(paths, &[image_bytes], false, device)
}

/// Extract, average, compose, and freeze the identity for one request.
///
/// The multi-photograph form, of which [`extract_identity_embedding`] is the
/// one-element case rather than a second code path. Called EXACTLY ONCE per
/// parent request, at admission, before batch fan-out; everything it loads is
/// released before it returns.
///
/// ## Averaging
///
/// Each photograph is run through the WHOLE pipeline independently — detector,
/// ArcFace, EVA02-CLIP, IDFormer — and the resulting `[1, 32, 2048]` token sets
/// are averaged. The reference is `cubiq/PuLID_ComfyUI`, not
/// `ToTheBeginning/PuLID`, whose pipeline only ever handles a single image
/// (`pulid/pipeline_flux.py:120-194`): `pulid.py:406` appends
/// `pulid_model.get_image_embeds(id_cond, id_vit_hidden)` — the IDFormer's
/// output — once per image, and `pulid.py:415-419` means over them. Averaging
/// the raw ArcFace vectors or the EVA hidden states BEFORE the IDFormer would
/// be a different and untrained composition, so it is not done.
///
/// ## Ordering and refusals
///
/// The mean is order-independent, but the recorded provenance keeps request
/// order. A photograph with no detectable face refuses the whole request and
/// names which one: `PuLID_ComfyUI` warns and silently skips it
/// (`pulid.py:360-373`), which would change the face that renders without
/// changing anything the caller can see.
///
/// ## `want_uncond`
///
/// When the request runs a true-CFG negative branch it also needs the
/// UNCONDITIONAL identity — the IDFormer over all-zero conditioning
/// (`PuLID/pulid/pipeline_flux.py:188-192`). It is a pure function of the
/// adapter weights: it depends on no photograph, which is also why
/// `PuLID_ComfyUI`'s per-image uncond is the same tensor for every image at
/// `noise == 0` and its mean is that tensor (`pulid.py:396-407,416-419`). It is
/// computed only when asked for, so an ordinary identity render pays nothing.
pub fn extract_identity_embeddings(
    paths: &PulidPaths,
    images: &[&[u8]],
    want_uncond: bool,
    device: &Device,
) -> std::result::Result<IdentityExtraction, IdentityError> {
    // The bounded-decode limits are the request contract's, checked before any
    // decoder sees the bytes. Admission has already applied them, but this is
    // a public entry point and the check costs a header read.
    mold_core::identity::validate_id_images(images).map_err(IdentityError::Decode)?;

    let count = images.len();
    let assets = pinned_asset_digests(paths.family);
    let sources: Vec<String> = images
        .iter()
        .map(|bytes| mold_core::identity::id_image_sha256(bytes))
        .collect();
    // Consulted BEFORE anything is opened. A set whose every photograph is
    // cached loads no detector, no recognizer, no parser, no tower, and no
    // adapter — which is the whole point: the value is a pure function of the
    // photograph and the five assets, all of which are known here.
    let keys: Vec<String> = sources
        .iter()
        .map(|sha| mold_core::identity::identity_cache_key(sha, &assets))
        .collect();
    let mut cached: Vec<Option<CachedIdentity>> = keys.iter().map(|key| cache_get(key)).collect();
    // The uncond half is memoized separately on the adapter alone; see
    // [`uncond_memo`].
    let mut uncond = if want_uncond {
        uncond_memo()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .filter(|(digest, _)| digest == &assets.adapter)
            .map(|(_, tokens)| tokens.clone())
    } else {
        None
    };
    let missing: Vec<usize> = (0..count)
        .filter(|index| cached[*index].is_none())
        .collect();

    let mut extracted = false;
    if !missing.is_empty() || (want_uncond && uncond.is_none()) {
        // Everything this call might have to compute, named as flight keys.
        // Taking them all before re-reading the cache is what makes the
        // re-read authoritative: a peer that beat us here has already stored
        // its result and released.
        let mut flight_keys: Vec<String> = missing.iter().map(|&i| keys[i].clone()).collect();
        if want_uncond && uncond.is_none() {
            flight_keys.push(uncond_flight_key(&assets.adapter));
        }
        let arcs = flights_for(&flight_keys);
        let _flights: Vec<std::sync::MutexGuard<'_, ()>> = arcs
            .iter()
            .map(|lock| lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
            .collect();

        let outcome = (|| -> std::result::Result<bool, IdentityError> {
            // Re-read under the flights. Whatever a peer computed while we
            // waited is now visible, and taking ITS value rather than
            // recomputing is what makes two siblings byte-identical rather
            // than merely equivalent within tolerance.
            for &index in &missing {
                if let Some(value) = cache_get(&keys[index]) {
                    cached[index] = Some(value);
                }
            }
            if want_uncond && uncond.is_none() {
                uncond = uncond_memo()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .as_ref()
                    .filter(|(digest, _)| digest == &assets.adapter)
                    .map(|(_, tokens)| tokens.clone());
            }
            let still_missing: Vec<usize> = missing
                .iter()
                .copied()
                .filter(|index| cached[*index].is_none())
                .collect();
            let need_uncond = want_uncond && uncond.is_none();
            if still_missing.is_empty() && !need_uncond {
                return Ok(false);
            }

            // The face stack is loaded ONLY for photographs that still need
            // detecting. An uncond-only miss — the first true-CFG request
            // after an ordinary one, whose photograph is already cached —
            // needs the IDFormer and nothing else, and opening SCRFD and
            // ArcFace for it would place ~278 MB on the device to run neither.
            let mut faces = Vec::with_capacity(still_missing.len());
            let mut detected_warnings = Vec::with_capacity(still_missing.len());
            if !still_missing.is_empty() {
                let extractor = IdentityExtractor::load(paths, device)
                    .context("loading the PuLID face-extraction models")?;
                for &index in &still_missing {
                    // `in_photo_set` names which photograph while PRESERVING the
                    // category. Rebuilding this as a `Runtime` — which is what
                    // the `format!` it replaced did — reports a photograph with
                    // no detectable face as a device fault, and three such
                    // requests degrade a healthy GPU.
                    let features = extractor
                        .extract(images[index])
                        .map_err(|error| error.in_photo_set(index + 1, count))?;
                    detected_warnings.push(features.warning);
                    faces.push((features.arcface.raw, features.eva_crop_512));
                }
            }
            let composed = compose_identity_token_sets(paths, &faces, need_uncond, device)
                .context("composing the PuLID identity embedding")?;
            IDENTITY_EXTRACTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for ((&index, tokens), warning) in still_missing
                .iter()
                .zip(composed.per_image)
                .zip(detected_warnings)
            {
                let value = CachedIdentity {
                    tokens: std::sync::Arc::new(tokens),
                    warning,
                };
                cache_put(keys[index].clone(), value.clone());
                cached[index] = Some(value);
            }
            if let Some(fresh) = composed.uncond {
                let fresh = std::sync::Arc::new(fresh);
                *uncond_memo()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                    Some((assets.adapter.clone(), fresh.clone()));
                uncond = Some(fresh);
            }
            Ok(true)
        })();

        // The guards and the local handles go first, so `release_flights` can
        // tell "the registry is the only owner" from "a peer is still waiting".
        drop(_flights);
        drop(arcs);
        release_flights(&flight_keys);
        extracted = outcome?;
    }

    let mut warnings = Vec::new();
    let mut per_image = Vec::with_capacity(count);
    for (index, entry) in cached.into_iter().enumerate() {
        let entry = entry.context("an identity photograph produced no tokens")?;
        if let Some(warning) = &entry.warning {
            warnings.push(if count == 1 {
                warning.clone()
            } else {
                format!("identity photo {} of {count}: {warning}", index + 1)
            });
        }
        per_image.push(entry.tokens.as_ref().clone());
    }
    let tokens = mold_core::identity::average_identity_tokens(&per_image)
        .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?;

    let embedding = FrozenIdentityEmbedding::from_sources(&tokens, sources, assets)
        .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?;
    let embedding = match uncond {
        Some(uncond) => embedding
            .with_uncond(&uncond)
            .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?,
        None => embedding,
    };

    Ok(IdentityExtraction {
        embedding,
        warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
        extracted,
    })
}

/// The safetensors prefix the IDFormer's weights live under.
///
/// The one family-specific fact about extraction. Upstream instantiates the
/// identical `IDFormer` class in both pipelines — same file
/// (`pulid/encoders_transformer.py`), same defaults, same shapes, confirmed by
/// `id_adapter.latents` `[1, 32, 1024]` and `id_adapter.proj_out`
/// `[1024, 2048]` matching the FLUX golden exactly
/// (`testdata/pulid_sdxl/README.md`) — and stores it under a different leading
/// module name in each checkpoint.
pub fn idformer_prefix(family: IdentityFamily) -> &'static str {
    match family {
        // `pipeline_flux.py:99-109`.
        IdentityFamily::Flux => "pulid_encoder",
        // `pipeline_v1_1.py:151-163`, whose `getattr(self, module)` resolves
        // `id_adapter` to `self.id_adapter = IDFormer()`.
        IdentityFamily::Sdxl => "id_adapter",
    }
}

/// The manifest's pin for the adapter file.
///
/// Recorded rather than re-hashed: `mold pull` verified these 1.1 GB against
/// this exact digest when it wrote them, and re-reading the whole file on every
/// conditioned request would cost more than the extraction it annotates.
fn adapter_sha256(family: IdentityFamily) -> String {
    mold_core::pulid_assets::pulid_manifest_for(family)
        .files
        .iter()
        .find(|file| file.component == ModelComponent::IdentityAdapter)
        .and_then(|file| file.sha256)
        .unwrap_or("unpinned")
        .to_string()
}

/// Which half of [`compose_identity_tokens_observed`] a timing sample belongs
/// to.
///
/// `docs/architecture/pulid-perf.md` §0 recorded that this half of the
/// extraction had never been measured anywhere in the repository, and §4 made
/// measuring it the first deliverable of the implementation phase. The two
/// halves are reported separately because they are separately expensive and
/// separately optimizable: the tower is a 24-block ViT over 577 tokens, the
/// IDFormer is a much smaller cross-attention stack over 32 queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeStage {
    /// Segmenting one aligned crop with the BiSeNet parser, applying PuLID's
    /// mask, and preprocessing the result for the tower (#1225) — everything
    /// that turns one photograph into tower input. Emitted once PER
    /// photograph, even though the parser itself is materialized once, because
    /// it is per-crop work; the materialization falls inside
    /// [`ComposeStage::EvaBuild`]'s window, which subtracts these out.
    ///
    /// Reported apart from `EvaBuild` because it is a whole second network: a
    /// reader comparing the tower's cost to the extraction's would otherwise
    /// be silently attributing the parse to the tower.
    Parse,
    /// Materializing the derived safetensors and building the EVA02-CLIP
    /// vision tower — everything before its forward pass that is not
    /// [`ComposeStage::Parse`]. Separate from [`ComposeStage::EvaForward`]
    /// because it is what the drop-and-reload rule pays for on every request,
    /// and therefore what a residency change would buy back.
    ///
    /// The three stages below DECOMPOSE this one rather than sitting beside
    /// it: each is measured inside this window, so a caller that sums every
    /// stage must count `EvaBuild` alone. `pulid-perf.md` §5 asked for the
    /// split because the two halves of the largest line item in the pipeline
    /// have completely different fixes — one is a 609 MB re-hash, the other is
    /// an f16 -> f32 widening — and 1,268 ms of undifferentiated "build" said
    /// nothing about which to spend effort on.
    EvaBuild,
    /// Materializing and building the BiSeNet face parser. A SUBSET of
    /// [`ComposeStage::EvaBuild`]'s window: the parser is built (and dropped)
    /// before the tower, inside it.
    ParserBuild,
    /// Re-reading the 609 MB derived tower into a private buffer and proving
    /// it against the compiled-in pin. A SUBSET of [`ComposeStage::EvaBuild`].
    EvaAuthenticate,
    /// Constructing the tower's modules from those verified bytes — the
    /// `VarBuilder` pass that materializes every weight at the working dtype.
    /// A SUBSET of [`ComposeStage::EvaBuild`].
    EvaConstruct,
    /// The EVA02-CLIP vision tower's forward pass.
    EvaForward,
    /// Reading the adapter and building the PuLID IDFormer.
    IdFormerBuild,
    /// The IDFormer's forward pass.
    IdFormerForward,
}

/// Run the EVA tower and the IDFormer over one aligned crop, with a per-stage
/// wall-clock observer.
///
/// The one-photograph case of [`compose_identity_token_sets_observed`], and the
/// entry point `pulid_face_probe bench --full` drives — which §4 requires
/// before any performance claim about identity extraction is made. It
/// delegates rather than duplicating, so the benchmark cannot drift from the
/// production path by measuring a copy of it.
pub fn compose_identity_tokens_observed(
    paths: &PulidPaths,
    arcface: &[f32],
    eva_crop_512: &image::RgbImage,
    device: &Device,
    observe: &mut dyn FnMut(ComposeStage, std::time::Duration),
) -> Result<Vec<f32>> {
    let faces = [(arcface.to_vec(), eva_crop_512.clone())];
    let composed = compose_identity_token_sets_observed(paths, &faces, false, device, observe)?;
    composed
        .per_image
        .into_iter()
        .next()
        .context("the composer returned no identity tokens")
}

/// What one composition produced.
pub(crate) struct ComposedIdentityTokens {
    /// One `[32 * 2048]` token set per photograph, in request order.
    pub(crate) per_image: Vec<Vec<f32>>,
    /// The unconditional identity, when it was asked for.
    pub(crate) uncond: Option<Vec<f32>>,
}

/// Run the EVA tower and the IDFormer over every aligned crop.
pub(crate) fn compose_identity_token_sets(
    paths: &PulidPaths,
    faces: &[(Vec<f32>, image::RgbImage)],
    want_uncond: bool,
    device: &Device,
) -> Result<ComposedIdentityTokens> {
    compose_identity_token_sets_observed(paths, faces, want_uncond, device, &mut |_, _| {})
}

/// The one implementation: every photograph, the optional unconditional
/// identity, and the per-stage observer.
///
/// Every crop is parsed and masked first, with the parser resident once and
/// dropped; the tower is then built ONCE and run per photograph, then dropped;
/// the IDFormer is built once afterwards and run per photograph. That ordering
/// is why N photographs do not multiply the host peak: the three large halves
/// still never coexist, and the only thing that scales with N is the retained
/// hidden
/// states — [`EXTRACTION_RETAINED_BYTES_PER_IMAGE`], ~12 MB each against a
/// 2.4 GB peak.
///
/// The observer sees [`ComposeStage::EvaBuild`] and
/// [`ComposeStage::IdFormerBuild`] exactly once each, because they are paid
/// once however many photographs arrive — which is the whole point of building
/// the tower outside the loop. `EvaForward` and `IdFormerForward` are emitted
/// once PER photograph (and once more for the unconditional identity), so a
/// caller measuring a set sees the per-photograph cost rather than a sum it
/// cannot decompose. [`ComposeStage::Parse`] is likewise once per photograph.
/// `pulid_face_probe bench --full` passes one photograph, so its samples are
/// unchanged.
pub(crate) fn compose_identity_token_sets_observed(
    paths: &PulidPaths,
    faces: &[(Vec<f32>, image::RgbImage)],
    want_uncond: bool,
    device: &Device,
    observe: &mut dyn FnMut(ComposeStage, std::time::Duration),
) -> Result<ComposedIdentityTokens> {
    anyhow::ensure!(
        !faces.is_empty() || want_uncond,
        "composing needs at least one face"
    );
    let tower_dtype = eva_working_dtype(device);
    // Everything before the tower's first forward pass is one window, and the
    // per-crop parse work is subtracted out of it below rather than restarting
    // the clock — a restart would silently drop whatever ran before the parse
    // (materializing the tower, decoding the crops) from every stage's
    // account. `parse_elapsed` accumulates because `Parse` is emitted once per
    // photograph.
    let mut stage_started = std::time::Instant::now();
    let mut parse_elapsed = std::time::Duration::ZERO;
    // `pipeline_flux.py:161-174`: every crop is parsed, masked, and only then
    // resized and normalized for the tower. Both models read the SAME `[0, 1]`
    // planar tensor and normalize it differently — ImageNet for the parser,
    // OpenAI CLIP for the tower — so it is built once per crop here and the
    // normalizations stay inside the two consumers.
    //
    // This is a PRE-PASS over every photograph rather than work inside the
    // tower's loop, for the reason the tower itself is built outside that loop:
    // the parser is a second network, and parsing everything first lets it be
    // dropped before the tower is built. The three large halves still never
    // coexist.
    // A set whose every photograph was served from the cache still needs the
    // unconditional identity if this request runs the true-CFG branch, and
    // that value depends on no photograph at all. So the parser, the tower,
    // and their two stages are skipped entirely rather than run over nothing.
    let prepared: Vec<Tensor> = if faces.is_empty() {
        Vec::new()
    } else {
        // The derived artifact arrives as verified private BYTES — never a
        // pathname a loader would resolve a second time, never a shared mapping
        // another writer could edit underneath it. See
        // `pickle_convert::AuthenticatedArtifact`. Scoped to the build that
        // consumes it, so its 53 MB copy is released as soon as the parser owns
        // its tensors.
        let parser = {
            let started = std::time::Instant::now();
            let artifact = ensure_bisenet_parser_safetensors(paths)
                .context("materializing the BiSeNet face parser")?;
            let parser = BiSeNetParser::from_authenticated(&artifact, device)
                .context("building the BiSeNet face parser")?;
            settle(device)?;
            observe(ComposeStage::ParserBuild, started.elapsed());
            parser
        };
        let mut prepared = Vec::with_capacity(faces.len());
        for (_, eva_crop_512) in faces {
            let parse_started = std::time::Instant::now();
            let image = image::DynamicImage::ImageRgb8(eva_crop_512.clone());
            let (mut planar, height, width) = planar_rgb_from_image(&image);
            let labels = parser
                .labels(&planar, height, width)
                .context("parsing the aligned face crop")?;
            apply_pulid_face_mask(&mut planar, &labels).context("masking the aligned face crop")?;
            let pixels = preprocess_planar_rgb(&planar, height, width, device)
                .context("preprocessing the aligned face crop for EVA02-CLIP")?;
            settle(device)?;
            let elapsed = parse_started.elapsed();
            parse_elapsed += elapsed;
            observe(ComposeStage::Parse, elapsed);
            prepared.push(pixels);
        }
        prepared
    };

    // The tower and the IDFormer are built and dropped in sequence, never held
    // together: the tower is 609 MB and the IDFormer's `id_embedding_mapping`
    // alone is a 1280 x 5120 matrix, and this phase is deliberately disjoint
    // from the adapter residency that follows it on the same lease.
    let vision: Vec<(Vec<Tensor>, Tensor)> = if prepared.is_empty() {
        Vec::new()
    } else {
        // As above: verified private bytes, scoped to the build that consumes
        // them, so the tower's 609 MB copy is released as soon as it owns its
        // tensors.
        let tower = {
            let started = std::time::Instant::now();
            let artifact = ensure_eva_clip_vision_safetensors(paths)
                .context("materializing the EVA02-CLIP vision tower")?;
            observe(ComposeStage::EvaAuthenticate, started.elapsed());
            let started = std::time::Instant::now();
            let tower = EvaClipVisionTower::from_authenticated(&artifact, device, tower_dtype)
                .context("building the EVA02-CLIP-L-14-336 vision tower")?;
            settle(device)?;
            observe(ComposeStage::EvaConstruct, started.elapsed());
            tower
        };
        // Everything paid once before the first forward, minus the per-crop
        // parse work the pre-pass already reported. Restarting the clock after
        // the pre-pass instead would silently drop materializing the tower from
        // every stage's account.
        observe(
            ComposeStage::EvaBuild,
            stage_started.elapsed().saturating_sub(parse_elapsed),
        );

        let mut outputs = Vec::with_capacity(faces.len());
        for pixels in &prepared {
            stage_started = std::time::Instant::now();
            let output = tower
                .forward(pixels)
                .context("running the EVA02-CLIP vision tower")?;
            settle(device)?;
            observe(ComposeStage::EvaForward, stage_started.elapsed());
            outputs.push((output.hidden_states, output.cls_projection));
        }
        outputs
    };
    stage_started = std::time::Instant::now();

    // SAFETY: the ordinary mold safetensors mmap contract — the file must not
    // be mutated while the IDFormer holds it.
    //
    // This one IS loaded by pathname, and deliberately: the adapter is a
    // MANIFEST file whose digest the download verified, not an artifact mold
    // derived and hashed moments ago. There is no fresher authentication here
    // to throw away by reopening a name (see `adapter_sha256`, and
    // `pickle_convert::AuthenticatedArtifact` for the case that is different).
    // `pipeline_flux.py:99-109` and `pipeline_v1_1.py:151-163` both split the
    // checkpoint by leading module name; the IDFormer half is `pulid_encoder.*`
    // in the FLUX file and `id_adapter.*` in the SDXL v1.1 file. The two are
    // the SAME class with the same shapes — upstream instantiates one
    // `IDFormer()` in each pipeline — so only the prefix differs.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&paths.adapter),
            DType::F32,
            device,
        )
        .with_context(|| format!("reading the PuLID adapter {}", paths.adapter.display()))?
    };
    let idformer = IdFormer::new(vb.pp(idformer_prefix(paths.family)))
        .context("building the PuLID IDFormer")?;
    settle(device)?;
    observe(ComposeStage::IdFormerBuild, stage_started.elapsed());

    let mut per_image = Vec::with_capacity(faces.len());
    for ((arcface, _), (hidden_states, cls_projection)) in faces.iter().zip(vision.iter()) {
        stage_started = std::time::Instant::now();
        // `pipeline_flux.py:181`: `cat([id_ante_embedding, id_cond_vit])` — the
        // RAW ArcFace output, not the L2-normalized one; only the CLIP
        // projection is normalized, and the tower already did that.
        let arcface = Tensor::from_slice(arcface, (1, arcface.len()), device)
            .context("materializing the ArcFace embedding")?;
        // The tower may have run narrow (see [`eva_working_dtype`]); the
        // IDFormer's own `VarBuilder` is f32, and candle refuses a mixed-dtype
        // `cat` and matmul. Widening HERE rather than inside the tower keeps
        // the narrowing scoped to the tower's arithmetic: the IDFormer half
        // computes in exactly the dtype its committed goldens were captured
        // in, whatever device the tower ran on.
        let cls_projection = cls_projection
            .to_dtype(DType::F32)
            .context("widening the EVA02-CLIP projection for the IDFormer")?;
        let hidden_states = hidden_states
            .iter()
            .map(|hidden| hidden.to_dtype(DType::F32))
            .collect::<candle_core::Result<Vec<_>>>()
            .context("widening the EVA02-CLIP hidden states for the IDFormer")?;
        let id_cond = Tensor::cat(&[&arcface, &cls_projection], 1)
            .context("concatenating the ArcFace and EVA02-CLIP conditions")?;
        per_image.push(run_idformer(&idformer, &id_cond, &hidden_states)?);
        observe(ComposeStage::IdFormerForward, stage_started.elapsed());
    }

    let uncond = if want_uncond {
        stage_started = std::time::Instant::now();
        // `pipeline_flux.py:188-192`: `zeros_like(id_cond)` and a zeroed hidden
        // state per scale. Shaped from the tower's OWN declared geometry rather
        // than from a live output, because `zeros_like` means the values are
        // all zero and a cached photograph set produces no live tensor to copy
        // a shape from. `the_uncond_geometry_matches_the_tower_it_stands_in_for`
        // keeps the constants honest, and the assertion below re-checks them
        // against a tower that actually ran.
        let width = ID_ANTE_DIM + PROJECTION_DIM;
        if let Some((hidden_states, cls_projection)) = vision.first() {
            anyhow::ensure!(
                hidden_states.len() == HIDDEN_STATE_BLOCKS.len()
                    && hidden_states[0].dims() == [1, SEQUENCE_LEN, EMBED_DIM]
                    && cls_projection.dims() == [1, PROJECTION_DIM],
                "the vision tower's output geometry no longer matches the constants the \
                 unconditional identity is shaped from"
            );
        }
        let id_uncond = Tensor::zeros((1, width), DType::F32, device)
            .context("materializing the unconditional identity condition")?;
        let hidden_uncond = (0..HIDDEN_STATE_BLOCKS.len())
            .map(|_| Tensor::zeros((1, SEQUENCE_LEN, EMBED_DIM), DType::F32, device))
            .collect::<candle_core::Result<Vec<_>>>()
            .context("materializing the unconditional vision hidden states")?;
        let uncond = run_idformer(&idformer, &id_uncond, &hidden_uncond)?;
        observe(ComposeStage::IdFormerForward, stage_started.elapsed());
        Some(uncond)
    } else {
        None
    };

    Ok(ComposedIdentityTokens { per_image, uncond })
}

/// The EVA02-CLIP tower's working dtype on `device`.
///
/// The derived tower is stored f16 — it is a straight conversion of the
/// released `EVA02_CLIP_L_336_psz14_s6B.pt`, which BAAI ships in fp16 — so
/// asking a `VarBuilder` for f32 does not merely cost a copy, it costs a
/// widening pass that writes ~1.2 GB. `pulid-perf.md` §4 measured that pass as
/// half of `eva-build`, itself the single largest line item in the extraction,
/// and §5 made removing it phase 2's first target.
///
/// Narrow is also what upstream does: `PuLID/pulid/pipeline_flux.py:60` casts
/// the whole tower to `weight_dtype`, which `PuLID/app_flux.py:45` sets to
/// bfloat16 — strictly fewer mantissa bits than the f16 the file already
/// holds. EVA-CLIP's own factory takes the same route
/// (`PuLID/eva_clip/factory.py:342-344`). So the f16 arm is upstream's
/// behaviour and the f32 arm is mold's CPU concession, not the other way
/// round.
///
/// CPU stays f32 for two reasons and neither is precision: candle has no
/// narrow CPU kernels — an f16 matmul there is emulated element by element —
/// and every committed parity golden for this tower was captured against the
/// f32 CPU path. A CPU-only build must keep costing exactly what it cost
/// before phase 2.
pub fn eva_working_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    }
}

/// Force the device to finish before a stage's clock is read.
///
/// Metal and CUDA enqueue: `tower.forward(...)` returns as soon as the
/// commands are submitted, so an un-synchronized stage boundary attributes the
/// GPU's work to whichever LATER stage happens to block first. The first Metal
/// measurement of this pipeline reported `eva-forward` at 3.2 ms and hid 300
/// GFLOP inside the IDFormer's `to_vec1`, which is a measurement that would
/// have sent phase 3 after the wrong stage.
///
/// The cost is nothing to defend: every stage here consumes the previous
/// stage's output, so the pipeline was already serial, and the final
/// `to_vec1` synchronizes regardless. A failure to synchronize is a real
/// device error and is propagated rather than swallowed — a stage that could
/// not be waited for did not produce a value either.
fn settle(device: &Device) -> Result<()> {
    device
        .synchronize()
        .context("waiting for the identity extraction device")
}

/// Width of the raw ArcFace half of the IDFormer's conditioning vector.
const ID_ANTE_DIM: usize = 512;

fn run_idformer(idformer: &IdFormer, id_cond: &Tensor, hidden: &[Tensor]) -> Result<Vec<f32>> {
    let tokens = idformer
        .forward(id_cond, hidden)
        .context("running the PuLID IDFormer")?;
    let tokens = tokens.i(0).context("the IDFormer returned no batch")?;
    tokens
        .flatten_all()
        .context("flattening the identity tokens")?
        .to_dtype(DType::F32)
        .context("the identity tokens are numeric")?
        .to_vec1::<f32>()
        .context("reading the identity tokens")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity for the SDXL half of `idformer_prefix` (#1228).
    ///
    /// The claim it defends is that `id_adapter.*` and `pulid_encoder.*` hold
    /// the SAME class — upstream instantiates one bare `IDFormer()` in each
    /// pipeline — so mold's single port serves both and only the prefix moves.
    /// A prefix that resolved nothing would build an IDFormer of zeros and
    /// return a plausible-looking tensor, which is exactly why this is a
    /// golden rather than a shape assertion.
    ///
    /// Weight-gated, mirroring `flux::pulid_encoder`'s own goldens:
    ///
    /// ```text
    /// MOLD_TEST_PULID_ASSETS=/path/to/pulid \
    ///   cargo test --release -p mold-ai-inference --features pulid \
    ///     --lib identity::extraction -- --ignored --nocapture
    /// ```
    mod sdxl_idformer_parity {
        use super::*;
        use crate::pulid_fixtures::{
            pulid_asset, scale_relative_error, DeterministicStream, GoldenStats,
            SEED_SDXL_IDFORMER_ID, SEED_SDXL_IDFORMER_VIT,
        };
        use candle_core::Device;

        const GOLDEN_FILE: &str = "idformer_goldens.safetensors";
        /// `capture_idformer_goldens.py`'s `ID_COND_DIM`: 512 ArcFace + 768
        /// EVA02-CLIP-L-14-336 projection. Deliberately NOT 1792 — see that
        /// directory's README, correction 1.
        const ID_COND_DIM: usize = 512 + 768;
        const VIT_TOKENS: usize = 577;
        const VIT_DIM: usize = 1024;
        const SCALES: usize = 5;

        /// The README measures f16 INPUT sensitivity at 6.5e-5 relative; a
        /// port compared against the f32 golden should sit far below that.
        /// FLUX's own `IdFormer` golden lands at 1.5e-7 absolute-vs-peak, so
        /// the budget is set an order of magnitude above that measurement and
        /// two below the input-sensitivity floor: a real regression in the
        /// attention, the softmax widening, or the `proj_out` orientation
        /// moves this by whole percent.
        const TOLERANCE: f32 = 1.0e-5;

        fn load_sdxl_idformer(device: &Device) -> IdFormer {
            let adapter = pulid_asset("pulid_v1.1.safetensors");
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)
                    .unwrap()
            };
            IdFormer::new(vb.pp(idformer_prefix(IdentityFamily::Sdxl))).unwrap()
        }

        fn assert_case(label: &str, output: &Tensor) {
            let stats = GoldenStats::load_sdxl(GOLDEN_FILE, &format!("{label}.stats"));
            stats.assert_matches(&GoldenStats::measure(output), 1e-4, label);

            let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let expected = crate::pulid_fixtures::sdxl_golden(GOLDEN_FILE, label)
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let error = scale_relative_error(&actual, &expected, stats.peak);
            println!("{label}: {error:.3e} of the {} scale", stats.peak);
            assert!(error < TOLERANCE, "{label} drifted by {error}");
        }

        fn single_inputs(device: &Device) -> (Tensor, Vec<Tensor>) {
            // The capture draws `(1, 1, 1280)` from one stream; mold's port
            // takes `[batch, 1280]` because it averages ACROSS photographs
            // after the IDFormer rather than stacking them into it, so the
            // same 1280 values arrive one rank lower.
            let id_cond =
                DeterministicStream::new(SEED_SDXL_IDFORMER_ID).tensor(&[1, ID_COND_DIM], device);
            let hidden = (0..SCALES)
                .map(|index| {
                    DeterministicStream::new(SEED_SDXL_IDFORMER_VIT + index as u64)
                        .tensor(&[1, VIT_TOKENS, VIT_DIM], device)
                })
                .collect();
            (id_cond, hidden)
        }

        #[test]
        #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
        fn the_sdxl_prefix_loads_the_same_idformer_upstream_does() {
            let device = Device::Cpu;
            let idformer = load_sdxl_idformer(&device);
            let (id_cond, hidden) = single_inputs(&device);
            let output = idformer.forward(&id_cond, &hidden).unwrap();
            assert_eq!(
                output.dims(),
                &[
                    1,
                    mold_core::identity::ID_EMBEDDING_TOKENS,
                    mold_core::identity::ID_EMBEDDING_DIM
                ]
            );
            assert_case("idformer.single.output", &output);
        }

        /// The unconditional identity SDXL's negative CFG branch conditions on
        /// (`pipeline_v1_1.py:243-247`). It is NOT a zero tensor — the
        /// IDFormer has biases, LayerNorms, and learned latent queries — so a
        /// port that shortcut it to zeros would render the negative branch
        /// unconditioned and quietly halve the identity in the guided result.
        #[test]
        #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
        fn the_sdxl_unconditional_identity_matches_upstream() {
            let device = Device::Cpu;
            let idformer = load_sdxl_idformer(&device);
            let id_cond = Tensor::zeros((1, ID_COND_DIM), DType::F32, &device).unwrap();
            let hidden: Vec<Tensor> = (0..SCALES)
                .map(|_| Tensor::zeros((1, VIT_TOKENS, VIT_DIM), DType::F32, &device).unwrap())
                .collect();
            let output = idformer.forward(&id_cond, &hidden).unwrap();
            let peak = GoldenStats::load_sdxl(GOLDEN_FILE, "idformer.uncond.output.stats").peak;
            assert!(peak > 1.0, "the unconditional identity is not near zero");
            assert_case("idformer.uncond.output", &output);
        }

        /// The committed `idformer.two_image.output` golden documents
        /// upstream's stacking path (`pipeline_v1_1.py:249-256`), which mold
        /// deliberately does NOT follow: it averages the IDFormer's OUTPUTS
        /// instead (`cubiq/PuLID_ComfyUI`'s `pulid.py:415-419`), which is why
        /// `IdFormer::forward` takes one photograph's conditioning at a time.
        /// Asserting the divergence keeps a future reader from "fixing" the
        /// port to match a golden it was never meant to reproduce.
        #[test]
        fn the_two_image_golden_documents_a_path_mold_does_not_take() {
            let device = Device::Cpu;
            let stacked = Tensor::zeros((1, 2, ID_COND_DIM), DType::F32, &device).unwrap();
            assert!(
                stacked.dims2().is_err(),
                "upstream's stacked id_cond is rank 3; mold's IDFormer takes rank 2 per photograph"
            );
        }
    }

    /// Every test that observes the process-global cache counter must run
    /// alone, for the same reason the memo tests in `pickle_convert` do.
    static CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn digests(vision: &str) -> IdentityAssetDigests {
        IdentityAssetDigests {
            adapter: "a".repeat(64),
            vision: vision.to_string(),
            face_detector: "d".repeat(64),
            face_recognizer: "r".repeat(64),
            face_parser: "p".repeat(64),
        }
    }

    /// The digests a cache key is composed from must be knowable BEFORE any
    /// asset is opened, or there is nothing to look a value up with.
    ///
    /// They are also the digests the extraction records afterwards: the
    /// loaders refuse any file whose bytes do not hash to the pin, so a
    /// post-load digest that differed from this one would have failed the load
    /// rather than been recorded.
    #[test]
    fn the_cache_key_digests_are_all_available_before_anything_is_loaded() {
        let assets = pinned_asset_digests(IdentityFamily::Flux);
        for (label, digest) in [
            ("adapter", &assets.adapter),
            ("vision", &assets.vision),
            ("face_detector", &assets.face_detector),
            ("face_recognizer", &assets.face_recognizer),
            ("face_parser", &assets.face_parser),
        ] {
            assert_eq!(digest.len(), 64, "{label} is not a SHA-256: {digest}");
            assert_ne!(digest, "unpinned", "{label} is unpinned");
        }
    }

    /// Every component of the key must actually change it. A component that
    /// does not is a component that cannot invalidate, which is the whole
    /// failure `pulid-perf.md` §2 enumerates: a repair pull that swapped the
    /// adapter, or a code change to the arithmetic, silently serving the old
    /// answer.
    #[test]
    fn every_key_component_changes_the_key() {
        use mold_core::identity::identity_cache_key;
        let photo = "0".repeat(64);
        let base = identity_cache_key(&photo, &digests("v"));
        assert_ne!(base, identity_cache_key(&"1".repeat(64), &digests("v")));
        assert_ne!(base, identity_cache_key(&photo, &digests("v2")));
        for mutate in [
            |a: &mut IdentityAssetDigests| a.adapter.push('x'),
            |a: &mut IdentityAssetDigests| a.face_detector.push('x'),
            |a: &mut IdentityAssetDigests| a.face_recognizer.push('x'),
            |a: &mut IdentityAssetDigests| a.face_parser.push('x'),
        ] {
            let mut assets = digests("v");
            mutate(&mut assets);
            assert_ne!(base, identity_cache_key(&photo, &assets));
        }
        // Same inputs, same key — a cache whose key was not a pure function of
        // its inputs would never hit.
        assert_eq!(base, identity_cache_key(&photo, &digests("v")));
    }

    /// The one invalidation case that is not structural. A reviewer can check
    /// the constant moved; this checks it is an input at all.
    #[test]
    fn the_cache_key_changes_when_the_pipeline_version_does() {
        use sha2::{Digest, Sha256};
        // Recomposed here rather than called, because the constant cannot be
        // varied at runtime — this is the only way to state "the version is
        // mixed in" as a test rather than as a code reading.
        let compose = |version: u32| {
            let assets = digests("v");
            let mut hasher = Sha256::new();
            hasher.update(b"mold.identity.cache.v1\0");
            hasher.update("0".repeat(64).as_bytes());
            hasher.update(b"\0");
            hasher.update(version.to_le_bytes());
            hasher.update(b"\0");
            for digest in [
                &assets.adapter,
                &assets.vision,
                &assets.face_detector,
                &assets.face_recognizer,
                &assets.face_parser,
            ] {
                hasher.update(digest.as_bytes());
                hasher.update(b"\0");
            }
            format!("{:x}", hasher.finalize())
        };
        assert_eq!(
            compose(mold_core::identity::IDENTITY_PIPELINE_VERSION),
            mold_core::identity::identity_cache_key(&"0".repeat(64), &digests("v")),
            "the shipped key is not this composition"
        );
        assert_ne!(
            compose(mold_core::identity::IDENTITY_PIPELINE_VERSION),
            compose(mold_core::identity::IDENTITY_PIPELINE_VERSION + 1)
        );
    }

    /// A hit returns the stored value and promotes it; a miss is a miss.
    #[test]
    fn the_cache_hits_misses_and_promotes() {
        let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        forget_cached_identities();
        let value = |seed: f32| CachedIdentity {
            tokens: std::sync::Arc::new(vec![seed; 4]),
            warning: None,
        };
        assert!(cache_get("absent").is_none());
        cache_put("a".to_string(), value(1.0));
        cache_put("b".to_string(), value(2.0));
        let before = identity_cache_hit_count();
        assert_eq!(cache_get("a").unwrap().tokens.as_ref(), &vec![1.0; 4]);
        assert_eq!(identity_cache_hit_count(), before + 1);
        // "a" was just used, so it is now the most recent.
        let cache = identity_cache().lock().unwrap();
        assert_eq!(cache[0].0, "a");
        drop(cache);
        // A miss does not count as a hit.
        assert!(cache_get("c").is_none());
        assert_eq!(identity_cache_hit_count(), before + 1);
    }

    /// The cap is on entries, and it evicts the least recently USED one —
    /// which is what makes a batch's siblings, all sharing one photograph,
    /// safe from a burst of unrelated faces.
    #[test]
    fn the_cache_is_bounded_and_evicts_the_least_recently_used_entry() {
        let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        forget_cached_identities();
        for index in 0..IDENTITY_CACHE_ENTRIES {
            cache_put(
                format!("key-{index}"),
                CachedIdentity {
                    tokens: std::sync::Arc::new(vec![index as f32]),
                    warning: None,
                },
            );
        }
        // Touch the oldest so it is no longer the eviction candidate.
        assert!(cache_get("key-0").is_some());
        cache_put(
            "overflow".to_string(),
            CachedIdentity {
                tokens: std::sync::Arc::new(vec![99.0]),
                warning: None,
            },
        );
        assert_eq!(
            identity_cache().lock().unwrap().len(),
            IDENTITY_CACHE_ENTRIES
        );
        assert!(
            cache_get("key-0").is_some(),
            "the touched entry was evicted"
        );
        assert!(cache_get("key-1").is_none(), "the LRU entry survived");
    }

    /// The advisory belongs to the photograph, so a hit must report it exactly
    /// as the extraction that produced it did. A cached "several faces were
    /// found" that went silent on the second render would tell the person who
    /// supplied a group photograph nothing.
    #[test]
    fn a_cached_photograph_keeps_its_advisory() {
        let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        forget_cached_identities();
        cache_put(
            "k".to_string(),
            CachedIdentity {
                tokens: std::sync::Arc::new(vec![0.0]),
                warning: Some("several faces were found".to_string()),
            },
        );
        assert_eq!(
            cache_get("k").unwrap().warning.as_deref(),
            Some("several faces were found")
        );
    }

    /// The unconditional identity depends on the adapter alone
    /// (`pipeline_flux.py:188-192`), so its geometry is the tower's declared
    /// shape rather than a live output — which is what lets a fully cached
    /// photograph set produce it without building a tower at all.
    #[test]
    fn the_uncond_geometry_matches_the_tower_it_stands_in_for() {
        assert_eq!(SEQUENCE_LEN, 577);
        assert_eq!(EMBED_DIM, 1024);
        assert_eq!(PROJECTION_DIM, 768);
        assert_eq!(HIDDEN_STATE_BLOCKS.len(), 5);
        assert_eq!(ID_ANTE_DIM + PROJECTION_DIM, 1280);
    }

    /// The gap the cache alone leaves open: concurrent callers for one key must
    /// compute once, and every other one must take THAT value rather than an
    /// equivalent one it computed itself.
    ///
    /// Exercised on the flight primitive directly rather than through the
    /// extractor, so it needs no weights and can actually run N threads: the
    /// production path's only addition is what it does inside the flight.
    #[test]
    fn concurrent_callers_for_one_key_compute_exactly_once() {
        let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        forget_cached_identities();
        const THREADS: usize = 8;
        let key = "one-photograph".to_string();
        let computed = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

        let seen: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|index| {
                    let key = key.clone();
                    let computed = computed.clone();
                    scope.spawn(move || {
                        if let Some(hit) = cache_get(&key) {
                            return hit.tokens.as_ref().clone();
                        }
                        let arcs = flights_for(std::slice::from_ref(&key));
                        let guards: Vec<_> = arcs
                            .iter()
                            .map(|lock| lock.lock().unwrap_or_else(|e| e.into_inner()))
                            .collect();
                        let value = match cache_get(&key) {
                            Some(hit) => hit,
                            None => {
                                let seq =
                                    computed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                // Every thread would produce a DIFFERENT value,
                                // so an equal result can only mean one compute.
                                let value = CachedIdentity {
                                    tokens: std::sync::Arc::new(vec![
                                        seq as f32 * 1000.0 + index as f32,
                                    ]),
                                    warning: None,
                                };
                                cache_put(key.clone(), value.clone());
                                value
                            }
                        };
                        drop(guards);
                        drop(arcs);
                        release_flights(std::slice::from_ref(&key));
                        value.tokens.as_ref().clone()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert_eq!(
            computed.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "{THREADS} concurrent callers for one key must compute once"
        );
        assert!(
            seen.windows(2).all(|pair| pair[0] == pair[1]),
            "every caller must return the winner's exact value: {seen:?}"
        );
    }

    /// A flight that fails stores nothing and releases the key, so the next
    /// caller computes for real. Negative caching would turn a torn file or a
    /// momentarily faceless frame into a permanent refusal.
    #[test]
    fn a_failed_flight_releases_the_key_so_a_retry_can_run() {
        let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        forget_cached_identities();
        let key = "retry-me".to_string();
        let keys = std::slice::from_ref(&key);

        // A flight that stores nothing, as a failing extraction does.
        {
            let arcs = flights_for(keys);
            let guards: Vec<_> = arcs
                .iter()
                .map(|lock| lock.lock().unwrap_or_else(|e| e.into_inner()))
                .collect();
            drop(guards);
            drop(arcs);
            release_flights(keys);
        }
        assert!(
            !flight_registry().lock().unwrap().contains_key(&key),
            "a settled flight must not leak a mutex per photograph"
        );
        assert!(cache_get(&key).is_none(), "a failure must cache nothing");

        // And the key is takeable again.
        let arcs = flights_for(keys);
        assert_eq!(arcs.len(), 1);
        drop(arcs);
        release_flights(keys);
    }

    /// Flights are taken in sorted order so two requests sharing a subset of
    /// photographs in different orders cannot deadlock. Stated over the
    /// ordering function rather than by racing two threads, because a
    /// deadlock test that passes proves only that it did not deadlock today.
    #[test]
    fn flights_are_taken_in_one_global_order() {
        let _guard = CACHE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let forward = ["b".to_string(), "a".to_string(), "c".to_string()];
        let reverse = ["c".to_string(), "b".to_string(), "a".to_string()];
        let ptrs = |keys: &[String]| -> Vec<usize> {
            let arcs = flights_for(keys);
            let ptrs = arcs
                .iter()
                .map(|arc| std::sync::Arc::as_ptr(arc) as usize)
                .collect();
            drop(arcs);
            ptrs
        };
        let one = ptrs(&forward);
        let two = ptrs(&reverse);
        assert_eq!(one, two, "two orderings of the same set must lock alike");
        release_flights(&forward);

        // Duplicates collapse: a set naming one photograph twice must take one
        // lock, not attempt to lock the same mutex twice and hang.
        let doubled = ["x".to_string(), "x".to_string()];
        assert_eq!(flights_for(&doubled).len(), 1);
        release_flights(&doubled);
    }

    /// The whole extraction, mask included, against upstream's on a real
    /// photograph.
    ///
    /// Every other PuLID fixture pins one stage on a synthetic input. This
    /// one starts from a committed portrait's aligned crop and the RAW
    /// ArcFace embedding #1222 captured beside it, and compares the `[1, 32,
    /// 2048]` value FLUX is actually conditioned on. It is the acceptance pin
    /// #1225 exists to satisfy: a mask that were subtly wrong would pass every
    /// per-stage test above and fail here.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn the_identity_matches_upstream_end_to_end() {
        /// Largest deviation, as a fraction of the golden tensor's own peak.
        /// The whole stack's f32 accumulation, over a real image rather than
        /// the synthetic input the per-stage goldens use.
        /// Measured worst on these four faces: 1.02e-5.
        const RELATIVE_BUDGET: f32 = 5e-5;
        const SEED_PARSE_PROBE: u64 = 0x50554C49_44505253;

        if std::env::var_os("MOLD_TEST_PULID_ASSETS").is_none() {
            eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset");
            return;
        }
        let testdata = crate::pulid_fixtures::testdata_dir();
        let faces = testdata.join("faces");
        let paths = PulidPaths {
            family: IdentityFamily::Flux,
            adapter: crate::pulid_fixtures::pulid_asset("pulid_flux_v0.9.1.safetensors"),
            vision_encoder_source: crate::pulid_fixtures::pulid_asset(
                "EVA02_CLIP_L_336_psz14_s6B.pt",
            ),
            face_detector: crate::pulid_fixtures::pulid_asset("scrfd_10g_bnkps.onnx"),
            face_recognizer: crate::pulid_fixtures::pulid_asset("glintr100.onnx"),
            face_parser_source: crate::pulid_fixtures::pulid_asset("parsing_bisenet.pth"),
        };

        let goldens = candle_core::safetensors::load(
            testdata.join("parse_goldens.safetensors"),
            &Device::Cpu,
        )
        .expect("the #1225 goldens are committed");
        let sources = std::fs::read_to_string(faces.join("sources.json")).unwrap();
        let stems: Vec<String> = sources
            .split("\"file\": \"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .map(|file| file.trim_end_matches(".jpg").to_string())
            .collect();
        assert!(!stems.is_empty());

        for stem in stems {
            let golden_json: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(faces.join(format!("{stem}.golden.json"))).unwrap(),
            )
            .unwrap();
            let arcface: Vec<f32> = golden_json["embedding"]
                .as_array()
                .expect("the #1222 golden carries the raw embedding")
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            let crop = image::open(faces.join(format!("{stem}.eva512.png")))
                .unwrap()
                .to_rgb8();

            let tokens = compose_identity_tokens_observed(
                &paths,
                &arcface,
                &crop,
                &Device::Cpu,
                &mut |_, _| {},
            )
            .unwrap();
            assert_eq!(tokens.len(), 32 * 2048);

            // The identity probe is the fourth draw from the capture script's
            // stream, after the label, masked and preprocess probes.
            let mut stream = crate::pulid_fixtures::DeterministicStream::new(SEED_PARSE_PROBE);
            let plane = (crop.width() * crop.height()) as usize;
            let _ = stream.indices(crate::pulid_fixtures::PROBE_COUNT, plane);
            let _ = stream.indices(crate::pulid_fixtures::PROBE_COUNT, 3 * plane);
            let _ = stream.indices(crate::pulid_fixtures::PROBE_COUNT, 3 * 336 * 336);
            let indices = stream.indices(crate::pulid_fixtures::PROBE_COUNT, tokens.len());

            let expected = goldens[&format!("{stem}.identity.probe")]
                .to_vec1::<f32>()
                .unwrap();
            let stats = goldens[&format!("{stem}.identity.stats")]
                .to_vec1::<f32>()
                .unwrap();
            let peak = stats[4];
            let actual: Vec<f32> = indices.iter().map(|i| tokens[*i as usize]).collect();
            let relative = crate::pulid_fixtures::scale_relative_error(&actual, &expected, peak);
            eprintln!("{stem}: identity relative error {relative:.3e} of peak {peak:.1}");
            assert!(
                relative <= RELATIVE_BUDGET,
                "{stem}: identity relative error {relative}"
            );
        }
    }

    /// The peak is what the host-RAM ledger is charged, so it must stay a
    /// number a reviewer can check against the table in the doc comment rather
    /// than a value that drifted.
    #[test]
    fn the_charged_host_peak_covers_every_stage_of_the_extraction() {
        const SCRFD: u64 = 16_923_827;
        const GLINTR100: u64 = 260_665_334;
        // The derived vision tower's own bytes, read into a private buffer so
        // the digest and the load see the same ones. Transient, but alive
        // while the weights below are being materialized from it.
        const TOWER_VERIFIED_BYTES: u64 = crate::encoders::pickle_convert::EVA_DERIVED.size_bytes;
        // The same weights as f32: the `VarBuilder` widens the f16 file.
        const TOWER_WEIGHTS: u64 = 2 * TOWER_VERIFIED_BYTES;
        // f32 activations for 577 tokens x 1024 across 24 blocks, plus the
        // five retained hidden states.
        const TOWER_ACTIVATIONS: u64 = 180 * 1_000_000;
        let peak = SCRFD + GLINTR100 + TOWER_VERIFIED_BYTES + TOWER_WEIGHTS + TOWER_ACTIVATIONS;
        assert!(
            EXTRACTION_HOST_PEAK_BYTES >= peak,
            "the charged peak {EXTRACTION_HOST_PEAK_BYTES} must cover the measured {peak}"
        );
        assert!(
            EXTRACTION_HOST_PEAK_BYTES < 2 * peak,
            "a charge more than twice the measurement parks hosts that could run this"
        );
    }

    /// The digest recorded for the adapter is the manifest's pin, which is
    /// what the download verified. A silent `unpinned` would make the frozen
    /// fingerprint stop distinguishing bundles.
    #[test]
    fn the_adapter_digest_is_the_manifest_pin() {
        let sha = adapter_sha256(IdentityFamily::Flux);
        assert_eq!(sha.len(), 64, "{sha}");
        assert_ne!(sha, "unpinned");
    }

    /// A multi-photograph extraction must not multiply the peak. The two large
    /// halves are still built once and dropped in sequence; only the retained
    /// hidden states scale with the count, and the whole budget has to stay
    /// inside the charge admission makes.
    #[test]
    fn the_charged_peak_still_covers_the_largest_admissible_photograph_set() {
        let extra =
            EXTRACTION_RETAINED_BYTES_PER_IMAGE * (mold_core::identity::ID_IMAGES_MAX as u64 - 1);
        // The five retained f32 hidden states, from the tower's own shape.
        let measured_per_image: u64 = 5 * 577 * 1024 * 4;
        assert!(
            EXTRACTION_RETAINED_BYTES_PER_IMAGE >= measured_per_image,
            "the per-photograph charge must cover the retained hidden states"
        );
        assert!(
            EXTRACTION_RETAINED_BYTES_PER_IMAGE < 2 * measured_per_image,
            "a charge more than twice the measurement parks hosts that could run this"
        );
        // The same terms the single-photograph test uses, so the two cannot
        // disagree about what the tower stage costs.
        const SCRFD: u64 = 16_923_827;
        const GLINTR100: u64 = 260_665_334;
        const TOWER_VERIFIED_BYTES: u64 = crate::encoders::pickle_convert::EVA_DERIVED.size_bytes;
        const TOWER_WEIGHTS: u64 = 2 * TOWER_VERIFIED_BYTES;
        const TOWER_ACTIVATIONS: u64 = 180 * 1_000_000;
        let peak =
            SCRFD + GLINTR100 + TOWER_VERIFIED_BYTES + TOWER_WEIGHTS + TOWER_ACTIVATIONS + extra;
        assert!(
            EXTRACTION_HOST_PEAK_BYTES >= peak,
            "the charged peak {EXTRACTION_HOST_PEAK_BYTES} must cover {peak} at \
             {} photographs",
            mold_core::identity::ID_IMAGES_MAX
        );
    }

    /// The set validator is what this entry point applies, so a set that is
    /// individually legal but collectively over budget is refused here too —
    /// before two ONNX graph decodes are paid for.
    #[test]
    fn an_over_budget_set_is_refused_before_any_model_loads() {
        let paths = PulidPaths {
            family: IdentityFamily::Flux,
            adapter: "/nonexistent/adapter.safetensors".into(),
            vision_encoder_source: "/nonexistent/eva.pt".into(),
            face_detector: "/nonexistent/scrfd.onnx".into(),
            face_recognizer: "/nonexistent/glintr100.onnx".into(),
            face_parser_source: "/nonexistent/parsing_bisenet.pth".into(),
        };
        let png = b"\x89PNG\r\n\x1a\n".to_vec();
        let too_many: Vec<&[u8]> = vec![png.as_slice(); mold_core::identity::ID_IMAGES_MAX + 1];
        let error =
            extract_identity_embeddings(&paths, &too_many, false, &Device::Cpu).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("at most"), "{rendered}");
        assert!(!rendered.contains("scrfd.onnx"), "{rendered}");

        let error = extract_identity_embeddings(&paths, &[], false, &Device::Cpu).unwrap_err();
        assert!(format!("{error:#}").contains("must not be empty"));
    }

    /// The bounded-decode limits are re-checked at this entry point, before
    /// any model is loaded — an oversized payload must not first pay for two
    /// ONNX graph decodes.
    #[test]
    fn an_invalid_payload_is_refused_before_any_model_loads() {
        let paths = PulidPaths {
            family: IdentityFamily::Flux,
            adapter: "/nonexistent/adapter.safetensors".into(),
            vision_encoder_source: "/nonexistent/eva.pt".into(),
            face_detector: "/nonexistent/scrfd.onnx".into(),
            face_recognizer: "/nonexistent/glintr100.onnx".into(),
            face_parser_source: "/nonexistent/parsing_bisenet.pth".into(),
        };
        let error = extract_identity_embedding(&paths, b"not an image", &Device::Cpu).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("PNG or JPEG"), "{rendered}");
        assert!(
            !rendered.contains("scrfd.onnx"),
            "no model may be opened for a payload the contract already refused: {rendered}"
        );
    }
}
