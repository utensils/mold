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
//! **Everything here runs on the CPU, deliberately.** `candle-onnx`
//! materializes every initializer on `Device::Cpu` and refuses anything else
//! (`IdentityExtractor::load`), and the two candle halves follow it for a
//! reason that outlives that constraint: this runs at *admission*, before the
//! scheduler has picked a device, so there is no GPU to run it on yet — and
//! that is exactly what satisfies #1223's "a measured slot that does not
//! overlap the T5/CLIP encode peak". The extraction has completed and released
//! its memory before the job is dispatched, so the two peaks cannot coexist by
//! construction rather than by scheduling luck. The result is a
//! device-independent 256 KiB value, which is what lets one extraction serve
//! every sibling of a batch on every device it fans out to.
//!
//! Peak host RAM is the EVA tower (~609 MB f16 mmap'd, widened to f32
//! activations) plus the two ONNX graphs (~278 MB) plus the IDFormer
//! (~330 MB) — [`EXTRACTION_HOST_PEAK_BYTES`]. Admission charges the host-RAM
//! ledger from the artifacts' own sizes through their `is_host_only` component
//! roles rather than from this constant; the constant is the MEASUREMENT those
//! charges are sized against, and the reason
//! `memory_preflight::IDENTITY_VRAM_OVERHEAD_BYTES` no longer counts any of it
//! as device memory.

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
/// | Stage | Bytes |
/// | --- | --- |
/// | SCRFD `scrfd_10g_bnkps.onnx` graph, decoded | 17 MB |
/// | ArcFace `glintr100.onnx` graph, decoded | 261 MB |
/// | BiSeNet face parser, f32 mmap + f32 activations | 53 MB + ~200 MB |
/// | EVA02-CLIP vision tower, f16 mmap + f32 activations | 609 MB + ~180 MB |
/// | IDFormer (`pulid_encoder.*` of the adapter file), f32 | ~330 MB |
///
/// The parser is dropped before the tower is built and the tower before the
/// IDFormer, so the three large stages do not coexist; the peak is the tower
/// stage. Rounded up to a round 1.4 GB so the figure is conservative rather
/// than exact.
///
/// Nothing reads this at runtime — the ledger charges the artifacts themselves
/// — so it is the recorded measurement, kept beside the code that produces it
/// and pinned by a test, rather than an input to a calculation.
pub const EXTRACTION_HOST_PEAK_BYTES: u64 = 1_400_000_000;

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
) -> std::result::Result<IdentityExtraction, IdentityError> {
    extract_identity_embeddings(paths, &[image_bytes], false)
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
) -> std::result::Result<IdentityExtraction, IdentityError> {
    // The bounded-decode limits are the request contract's, checked before any
    // decoder sees the bytes. Admission has already applied them, but this is
    // a public entry point and the check costs a header read.
    mold_core::identity::validate_id_images(images).map_err(IdentityError::Decode)?;

    let extractor = IdentityExtractor::load(paths, &Device::Cpu)
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

    let composed = compose_identity_token_sets(paths, &faces, want_uncond)
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
    EvaBuild,
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
    observe: &mut dyn FnMut(ComposeStage, std::time::Duration),
) -> Result<Vec<f32>> {
    let faces = [(arcface.to_vec(), eva_crop_512.clone())];
    let composed = compose_identity_token_sets_observed(paths, &faces, false, observe)?;
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
) -> Result<ComposedIdentityTokens> {
    compose_identity_token_sets_observed(paths, faces, want_uncond, &mut |_, _| {})
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
/// 1.4 GB peak.
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
    observe: &mut dyn FnMut(ComposeStage, std::time::Duration),
) -> Result<ComposedIdentityTokens> {
    anyhow::ensure!(!faces.is_empty(), "composing needs at least one face");
    let device = Device::Cpu;
    let mut stage_started = std::time::Instant::now();
    let vision_path = ensure_eva_clip_vision_safetensors(paths)
        .context("materializing the EVA02-CLIP vision tower")?;

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
    let mut parse_elapsed = std::time::Duration::ZERO;
    let prepared: Vec<Tensor> = {
        let parser_path = ensure_bisenet_parser_safetensors(paths)
            .context("materializing the BiSeNet face parser")?;
        // SAFETY: the same mmap contract every other mold safetensors loader
        // relies on. These bytes were just authenticated against
        // `BISENET_DERIVED_SHA256`.
        let parser = {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&parser_path),
                    DType::F32,
                    &device,
                )
                .with_context(|| format!("reading the face parser {}", parser_path.display()))?
            };
            BiSeNetParser::new(vb, &device).context("building the BiSeNet face parser")?
        };
        let mut prepared = Vec::with_capacity(faces.len());
        for (_, eva_crop_512) in faces {
            let parse_started = std::time::Instant::now();
            let image = image::DynamicImage::ImageRgb8(eva_crop_512.clone());
            let (mut planar, height, width) = planar_rgb_from_image(&image);
            let labels = parser
                .labels(&planar, height, width)
                .context("parsing the aligned face crop")?;
            apply_pulid_face_mask(&mut planar, &labels)
                .context("masking the aligned face crop")?;
            let pixels = preprocess_planar_rgb(&planar, height, width, &device)
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
        // SAFETY: the same mmap contract every other mold safetensors loader
        // relies on — the file must not be mutated while the tower holds it.
        // These bytes were just authenticated against `DERIVED_SHA256`.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&vision_path),
                DType::F32,
                &device,
            )
            .with_context(|| format!("reading the vision tower {}", vision_path.display()))?
        };
        let tower = EvaClipVisionTower::new(vb, &device)
            .context("building the EVA02-CLIP-L-14-336 vision tower")?;
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

    // SAFETY: as above. `pipeline_flux.py:99-109` splits the checkpoint by
    // leading module name; the IDFormer half is `pulid_encoder.*`.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            std::slice::from_ref(&paths.adapter),
            DType::F32,
            &device,
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
        let arcface = Tensor::from_slice(arcface, (1, arcface.len()), &device)
            .context("materializing the ArcFace embedding")?;
        let id_cond = Tensor::cat(&[&arcface, cls_projection], 1)
            .context("concatenating the ArcFace and EVA02-CLIP conditions")?;
        per_image.push(run_idformer(&idformer, &id_cond, hidden_states)?);
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
        let id_uncond = Tensor::zeros((1, width), DType::F32, &device)
            .context("materializing the unconditional identity condition")?;
        let hidden_uncond = hidden_states
            .iter()
            .map(|hidden| hidden.zeros_like())
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

            let tokens =
                compose_identity_tokens_observed(&paths, &arcface, &crop, &mut |_, _| {}).unwrap();
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
        // The derived vision tower, f16.
        const TOWER: u64 = 609 * 1_000_000;
        // f32 activations for 577 tokens x 1024 across 24 blocks, plus the
        // five retained hidden states.
        const TOWER_ACTIVATIONS: u64 = 180 * 1_000_000;
        let peak = SCRFD + GLINTR100 + TOWER + TOWER_ACTIVATIONS;
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
        const SCRFD: u64 = 16_923_827;
        const GLINTR100: u64 = 260_665_334;
        const TOWER: u64 = 609 * 1_000_000;
        const TOWER_ACTIVATIONS: u64 = 180 * 1_000_000;
        let peak = SCRFD + GLINTR100 + TOWER + TOWER_ACTIVATIONS + extra;
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
        let error = extract_identity_embeddings(&paths, &too_many, false).unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("at most"), "{rendered}");
        assert!(!rendered.contains("scrfd.onnx"), "{rendered}");

        let error = extract_identity_embeddings(&paths, &[], false).unwrap_err();
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
        let error = extract_identity_embedding(&paths, b"not an image").unwrap_err();
        let rendered = format!("{error:#}");
        assert!(rendered.contains("PNG or JPEG"), "{rendered}");
        assert!(
            !rendered.contains("scrfd.onnx"),
            "no model may be opened for a payload the contract already refused: {rendered}"
        );
    }
}
