//! ArcFace identity embedding (`glintr100`, iResNet100).
//!
//! Ported from `insightface/python-package/insightface/model_zoo/arcface_onnx.py`.
//! The graph takes a 112x112 aligned crop and returns one 512-d vector; the
//! only preprocessing outside it is the crop and `(x - 127.5) / 127.5`
//! (`arcface_onnx.py:37-40`, `:78-81`).
//!
//! ## Normalization: raw is the contract
//!
//! `ArcFaceONNX.get` stores the **raw** network output on the face
//! (`arcface_onnx.py:63-66`), and PuLID conditions on exactly that value:
//! `pipeline_flux.py:130` reads `face_info['embedding']`, and `:156-158` moves
//! it to the device without normalizing — in visible contrast to the EVA
//! branch three lines later (`:177-178`), which *is* L2-normalized before the
//! two halves are concatenated. Issue #1222's one-line summary says
//! "L2-normalized"; upstream says otherwise, and upstream is what the IDFormer
//! was trained against. [`ArcFaceEmbedding::raw`] is therefore the value that
//! travels to #1229, and [`ArcFaceEmbedding::l2_normalized`] is offered beside
//! it for the cosine-similarity comparisons parity testing needs.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;
use image::RgbImage;

use super::align::{estimate_arcface_norm, Landmarks5};
use super::arcface_net::IResNet100;
use super::warp::warp_affine;

/// The aligned crop size the recognizer takes, `glintr100`'s declared input.
pub const CROP_SIZE: u32 = 112;
/// Embedding width.
pub const EMBEDDING_DIM: usize = 512;
/// `ArcFaceONNX.input_mean` for a non-MXNet export, `arcface_onnx.py:38`.
const INPUT_MEAN: f32 = 127.5;
/// `ArcFaceONNX.input_std` for a non-MXNet export, `arcface_onnx.py:39`.
const INPUT_STD: f32 = 127.5;
/// `cv2.warpAffine(..., borderValue=0.0)`, `face_align.py:26-29`.
const CROP_BORDER: [u8; 3] = [0, 0, 0];

/// A 512-d ArcFace embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct ArcFaceEmbedding {
    /// The network's raw output — what PuLID conditions on.
    pub raw: Vec<f32>,
}

impl ArcFaceEmbedding {
    /// The unit-length embedding, for cosine comparisons.
    pub fn l2_normalized(&self) -> Vec<f32> {
        let norm = self.raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            return self.raw.clone();
        }
        self.raw.iter().map(|v| v / norm).collect()
    }

    /// Cosine similarity against another embedding, `arcface_onnx.py:68-73`.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .raw
            .iter()
            .zip(other.raw.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na = self.raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = other.raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na <= f32::EPSILON || nb <= f32::EPSILON {
            return 0.0;
        }
        dot / (na * nb)
    }
}

/// `face_align.norm_crop`, `face_align.py:26-29`: fit the ArcFace template and
/// warp with a black constant border.
pub fn norm_crop(image: &RgbImage, landmarks: &Landmarks5) -> Result<RgbImage> {
    let m = estimate_arcface_norm(landmarks, CROP_SIZE)
        .map_err(|e| anyhow::anyhow!("ArcFace alignment failed: {e}"))?;
    warp_affine(image, &m, CROP_SIZE, CROP_SIZE, CROP_BORDER)
        .context("the ArcFace alignment transform was not invertible")
}

/// The recognizer: the resident `glintr100` network.
///
/// The `ModelProto` is consumed at construction and dropped — see
/// [`super::arcface_net`] and `docs/architecture/pulid-perf.md` §1. This is
/// where the per-call re-materialization of 261 MB of initializers used to be.
pub struct ArcFaceRecognizer {
    net: IResNet100,
}

impl ArcFaceRecognizer {
    /// Wrap a decoded `glintr100` graph, on the CPU.
    pub fn new(model: ModelProto) -> Result<Self> {
        Self::new_on_device(model, &Device::Cpu)
    }

    /// Wrap a decoded `glintr100` graph, placing its weights on `device`.
    pub fn new_on_device(model: ModelProto, device: &Device) -> Result<Self> {
        let graph = model
            .graph
            .as_ref()
            .context("the ArcFace model carries no graph")?;
        if graph.output.len() != 1 {
            bail!(
                "expected exactly one ArcFace output (arcface_onnx.py:56), got {}",
                graph.output.len()
            );
        }
        Ok(Self {
            net: IResNet100::new(&model, device).context("building the ArcFace network")?,
        })
    }

    /// `cv2.dnn.blobFromImages([img], 1/127.5, (112, 112), (127.5,)*3,
    /// swapRB=True)`, `arcface_onnx.py:78-81`.
    ///
    /// Upstream's crop is BGR and `swapRB` feeds the graph RGB; mold's crop is
    /// already RGB, so there is no swap.
    pub fn blob(crop: &RgbImage) -> Result<Tensor> {
        if crop.dimensions() != (CROP_SIZE, CROP_SIZE) {
            bail!(
                "the ArcFace crop must be {CROP_SIZE}x{CROP_SIZE}, got {}x{}",
                crop.width(),
                crop.height()
            );
        }
        let side = CROP_SIZE as usize;
        let mut data = vec![0f32; 3 * side * side];
        for (x, y, px) in crop.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            for c in 0..3 {
                data[c * side * side + y * side + x] = (px.0[c] as f32 - INPUT_MEAN) / INPUT_STD;
            }
        }
        Tensor::from_vec(data, (1, 3, side, side), &Device::Cpu).map_err(Into::into)
    }

    /// Embed an already-aligned 112x112 crop.
    pub fn embed_crop(&self, crop: &RgbImage) -> Result<ArcFaceEmbedding> {
        let raw = self
            .net
            .forward(&Self::blob(crop)?)
            .context("ArcFace evaluation failed")?;
        if raw.len() != EMBEDDING_DIM {
            bail!(
                "ArcFace returned {} values, expected {EMBEDDING_DIM}",
                raw.len()
            );
        }
        Ok(ArcFaceEmbedding { raw })
    }

    /// Align and embed in one step.
    pub fn embed(&self, image: &RgbImage, landmarks: &Landmarks5) -> Result<ArcFaceEmbedding> {
        self.embed_crop(&norm_crop(image, landmarks)?)
    }

    /// The device this recognizer's weights are resident on.
    ///
    /// Since #1227 the network is an ordinary resident candle module, so this
    /// is a real property of the instance rather than the `candle-onnx`
    /// evaluator's hardcoded `Device::Cpu`. Milestone 1 still only ever builds
    /// it on the CPU — extraction runs at admission, before any device is
    /// leased (`docs/architecture/pulid-perf.md` §1) — and
    /// [`super::IdentityExtractor::load`] keeps asserting that.
    pub fn device(&self) -> &Device {
        self.net.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blob_matches_upstreams_mean_and_std() {
        let crop = RgbImage::from_pixel(CROP_SIZE, CROP_SIZE, image::Rgb([255, 127, 0]));
        let blob = ArcFaceRecognizer::blob(&crop).unwrap();
        assert_eq!(blob.dims(), &[1, 3, 112, 112]);
        let values = blob.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let plane = (CROP_SIZE * CROP_SIZE) as usize;
        assert!((values[0] - 1.0).abs() < 1e-6, "{}", values[0]);
        assert!((values[plane] - (127.0 - 127.5) / 127.5).abs() < 1e-6);
        assert!((values[2 * plane] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_wrongly_sized_crop_is_refused() {
        let crop = RgbImage::new(64, 64);
        let err = ArcFaceRecognizer::blob(&crop).unwrap_err();
        assert!(format!("{err}").contains("112x112"), "{err}");
    }

    #[test]
    fn norm_crop_produces_the_declared_input_size() {
        let image = RgbImage::from_pixel(400, 400, image::Rgb([80, 90, 100]));
        let landmarks: Landmarks5 = [
            [150.0, 180.0],
            [250.0, 180.0],
            [200.0, 230.0],
            [160.0, 280.0],
            [240.0, 280.0],
        ];
        let crop = norm_crop(&image, &landmarks).unwrap();
        assert_eq!(crop.dimensions(), (CROP_SIZE, CROP_SIZE));
    }

    #[test]
    fn cosine_similarity_is_one_for_a_scaled_copy() {
        let a = ArcFaceEmbedding {
            raw: (0..EMBEDDING_DIM).map(|i| i as f32 - 200.0).collect(),
        };
        let b = ArcFaceEmbedding {
            raw: a.raw.iter().map(|v| v * 3.5).collect(),
        };
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_of_a_zero_vector_is_zero_not_nan() {
        let a = ArcFaceEmbedding {
            raw: vec![0.0; EMBEDDING_DIM],
        };
        let b = ArcFaceEmbedding {
            raw: vec![1.0; EMBEDDING_DIM],
        };
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    #[test]
    fn normalizing_preserves_direction_and_reaches_unit_length() {
        let e = ArcFaceEmbedding {
            raw: vec![3.0, 4.0]
                .into_iter()
                .chain(std::iter::repeat_n(0.0, EMBEDDING_DIM - 2))
                .collect(),
        };
        let n = e.l2_normalized();
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        let len: f32 = n.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }
}
