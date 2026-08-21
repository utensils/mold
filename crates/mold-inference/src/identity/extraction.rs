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
//! (~330 MB); see [`EXTRACTION_HOST_PEAK_BYTES`], which is what admission
//! charges the host-RAM ledger.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use mold_core::identity::{FrozenIdentityEmbedding, IdentityAssetDigests};
use mold_core::manifest::ModelComponent;
use mold_core::pulid_assets::PulidPaths;

use crate::encoders::eva_clip_convert::{ensure_eva_clip_vision_safetensors, DERIVED_SHA256};
use crate::encoders::eva_clip_preprocess::{planar_rgb_from_image, preprocess_planar_rgb};
use crate::encoders::eva_clip_vision::EvaClipVisionTower;
use crate::flux::pulid_encoder::IdFormer;

use super::{IdentityError, IdentityExtractor};

/// Host bytes one extraction peaks at, measured on the shipped artifacts.
///
/// | Stage | Bytes |
/// | --- | --- |
/// | SCRFD `scrfd_10g_bnkps.onnx` graph, decoded | 17 MB |
/// | ArcFace `glintr100.onnx` graph, decoded | 261 MB |
/// | EVA02-CLIP vision tower, f16 mmap + f32 activations | 609 MB + ~180 MB |
/// | IDFormer (`pulid_encoder.*` of the adapter file), f32 | ~330 MB |
///
/// The tower is dropped before the IDFormer is built, so the two large halves
/// do not coexist; the peak is the tower stage. Rounded up to a round 1.4 GB
/// so the ledger is charged a number that is conservative rather than exact.
pub const EXTRACTION_HOST_PEAK_BYTES: u64 = 1_400_000_000;

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
    // The bounded-decode limits are the request contract's, checked before any
    // decoder sees the bytes. Admission has already applied them, but this is
    // a public entry point and the check costs a header read.
    mold_core::identity::validate_id_image_bytes(image_bytes).map_err(IdentityError::Decode)?;

    let extractor = IdentityExtractor::load(paths, &Device::Cpu)
        .context("loading the PuLID face-extraction models")?;
    let features = extractor.extract(image_bytes)?;

    let tokens = compose_identity_tokens(paths, &features.arcface.raw, &features.eva_crop_512)
        .context("composing the PuLID identity embedding")?;

    let assets = IdentityAssetDigests {
        adapter: adapter_sha256(),
        vision: DERIVED_SHA256.to_string(),
        face_detector: extractor.detector_sha256().to_string(),
        face_recognizer: extractor.recognizer_sha256().to_string(),
    };
    let embedding = FrozenIdentityEmbedding::new(
        &tokens,
        mold_core::identity::id_image_sha256(image_bytes),
        assets,
    )
    .map_err(|reason| IdentityError::Runtime(anyhow::anyhow!(reason)))?;

    Ok(IdentityExtraction {
        embedding,
        warning: features.warning,
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

/// Run the EVA tower and the IDFormer over one aligned crop.
///
/// Split out from [`extract_identity_embedding`] so the weight-gated parity
/// test can drive it from a crop it produced by other means.
pub(crate) fn compose_identity_tokens(
    paths: &PulidPaths,
    arcface: &[f32],
    eva_crop_512: &image::RgbImage,
) -> Result<Vec<f32>> {
    let device = Device::Cpu;
    let vision_path = ensure_eva_clip_vision_safetensors(paths)
        .context("materializing the EVA02-CLIP vision tower")?;

    let pixels = {
        let image = image::DynamicImage::ImageRgb8(eva_crop_512.clone());
        let (planar, height, width) = planar_rgb_from_image(&image);
        preprocess_planar_rgb(&planar, height, width, &device)
            .context("preprocessing the aligned face crop for EVA02-CLIP")?
    };

    // The tower and the IDFormer are built and dropped in sequence, never held
    // together: the tower is 609 MB and the IDFormer's `id_embedding_mapping`
    // alone is a 1280 x 5120 matrix, and admission is the one place in the
    // process where host RAM is not already committed to a render.
    let (hidden_states, cls_projection) = {
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
        let output = tower
            .forward(&pixels)
            .context("running the EVA02-CLIP vision tower")?;
        (output.hidden_states, output.cls_projection)
    };

    // `pipeline_flux.py:181`: `cat([id_ante_embedding, id_cond_vit])` — the
    // RAW ArcFace output, not the L2-normalized one; only the CLIP projection
    // is normalized, and the tower already did that.
    let arcface = Tensor::from_slice(arcface, (1, arcface.len()), &device)
        .context("materializing the ArcFace embedding")?;
    let id_cond = Tensor::cat(&[&arcface, &cls_projection], 1)
        .context("concatenating the ArcFace and EVA02-CLIP conditions")?;

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
    let tokens = idformer
        .forward(&id_cond, &hidden_states)
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
