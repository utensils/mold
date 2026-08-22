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
use mold_core::identity::{FrozenIdentityEmbedding, IdentityAssetDigests};
use mold_core::manifest::ModelComponent;
use mold_core::pulid_assets::PulidPaths;

use crate::encoders::eva_clip_preprocess::{planar_rgb_from_image, preprocess_planar_rgb};
use crate::encoders::eva_clip_vision::EvaClipVisionTower;
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
pub const EXTRACTION_RETAINED_BYTES_PER_IMAGE: u64 = 12_000_000;

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

/// What one extraction produced.
#[derive(Debug)]
pub struct IdentityExtraction {
    /// The immutable identity every sibling of the parent request reuses.
    pub embedding: FrozenIdentityEmbedding,
    /// A caller-surfacable advisory — today only "several faces were found".
    /// Travels to the client as `x-mold-request-warning`.
    pub warning: Option<String>,
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

    let extractor = IdentityExtractor::load(paths, device)
        .context("loading the PuLID face-extraction models")?;

    let count = images.len();
    let mut faces = Vec::with_capacity(count);
    let mut warnings = Vec::new();
    for (index, bytes) in images.iter().enumerate() {
        let features = extractor.extract(bytes).map_err(|error| {
            if count == 1 {
                error
            } else {
                IdentityError::Runtime(anyhow::anyhow!(
                    "identity photo {} of {count}: {error}",
                    index + 1
                ))
            }
        })?;
        if let Some(warning) = features.warning {
            warnings.push(if count == 1 {
                warning
            } else {
                format!("identity photo {} of {count}: {warning}", index + 1)
            });
        }
        faces.push((features.arcface.raw, features.eva_crop_512));
    }

    let composed = compose_identity_token_sets(paths, &faces, want_uncond, device)
        .context("composing the PuLID identity embedding")?;
    let tokens = mold_core::identity::average_identity_tokens(&composed.per_image)
        .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?;

    let assets = IdentityAssetDigests {
        adapter: adapter_sha256(),
        vision: EVA_DERIVED_SHA256.to_string(),
        face_detector: extractor.detector_sha256().to_string(),
        face_recognizer: extractor.recognizer_sha256().to_string(),
        face_parser: BISENET_DERIVED_SHA256.to_string(),
    };
    let sources = images
        .iter()
        .map(|bytes| mold_core::identity::id_image_sha256(bytes))
        .collect();
    let embedding = FrozenIdentityEmbedding::from_sources(&tokens, sources, assets)
        .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?;
    let embedding = match composed.uncond {
        Some(uncond) => embedding
            .with_uncond(&uncond)
            .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?,
        None => embedding,
    };

    Ok(IdentityExtraction {
        embedding,
        warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
    })
}

/// The manifest's pin for the adapter file.
///
/// Recorded rather than re-hashed: `mold pull` verified these 1.1 GB against
/// this exact digest when it wrote them, and re-reading the whole file on every
/// conditioned request would cost more than the extraction it annotates.
fn adapter_sha256() -> String {
    mold_core::pulid_assets::pulid_manifest()
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
    anyhow::ensure!(!faces.is_empty(), "composing needs at least one face");
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
    let prepared: Vec<Tensor> = {
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
            let elapsed = parse_started.elapsed();
            parse_elapsed += elapsed;
            observe(ComposeStage::Parse, elapsed);
            prepared.push(pixels);
        }
        prepared
    };

    // The tower and the IDFormer are built and dropped in sequence, never held
    // together: the tower is 609 MB and the IDFormer's `id_embedding_mapping`
    // alone is a 1280 x 5120 matrix, and admission is the one place in the
    // process where host RAM is not already committed to a render.
    let vision: Vec<(Vec<Tensor>, Tensor)> = {
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
    // `pipeline_flux.py:99-109` splits the checkpoint by leading module name;
    // the IDFormer half is `pulid_encoder.*`.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&paths.adapter),
            DType::F32,
            device,
        )
        .with_context(|| format!("reading the PuLID adapter {}", paths.adapter.display()))?
    };
    let idformer = IdFormer::new(vb.pp("pulid_encoder")).context("building the PuLID IDFormer")?;
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
        // state per scale. Shaped from the first photograph's tensors, which is
        // exactly what `zeros_like` means — the values are all zero, so which
        // photograph they were shaped from cannot matter.
        let (hidden_states, cls_projection) = &vision[0];
        let width = ID_ANTE_DIM + cls_projection.dim(1)?;
        let id_uncond = Tensor::zeros((1, width), DType::F32, device)
            .context("materializing the unconditional identity condition")?;
        // Shaped from the tower's output but built at the IDFormer's dtype,
        // for the same reason the conditional branch widens above.
        let hidden_uncond = hidden_states
            .iter()
            .map(|hidden| Tensor::zeros(hidden.shape(), DType::F32, device))
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
        let sha = adapter_sha256();
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
