//! Face detection and identity embedding for PuLID-FLUX (#1222).
//!
//! One image in, one identity out: SCRFD finds the face and its five
//! landmarks, a similarity fit lands them on the ArcFace template, `glintr100`
//! embeds the 112x112 crop, and the SAME landmarks produce the 512x512 crop
//! the EVA vision tower conditions on in #1229.
//!
//! Everything here is a port of upstream InsightFace, facexlib, and PuLID —
//! never of mold's own intuition. Each module names its source file and line
//! range. The two ONNX graphs are evaluated by `candle-onnx`; the Step-0
//! decision that qualified it, with the measured latency and the alternatives
//! that were on the table, is recorded in
//! `docs/architecture/pulid-face-extraction.md`.
//!
//! Deliberately NOT implemented here, and named rather than silently skipped:
//! facexlib's RetinaFace detector and its BiSeNet background mask, both of
//! which PuLID applies before the EVA tower
//! (`PuLID/pulid/pipeline_flux.py:145-170`). Issue #1225 owns them; #1222's
//! fidelity gate decides whether the milestone needs them.

pub mod align;
pub mod arcface;
pub mod extraction;
pub mod onnx_graph;
pub mod onnx_inventory;
pub mod scrfd;
pub mod warp;

use std::path::Path;

use anyhow::{Context, Result};
use image::RgbImage;
use mold_core::manifest::ModelComponent;
use mold_core::pulid_assets::PulidPaths;

use align::{estimate_facexlib_512, Landmarks5};
use arcface::{ArcFaceEmbedding, ArcFaceRecognizer};
use scrfd::{DetectedFace, ScrfdDetector};
use warp::warp_affine;

/// The EVA crop's edge length, `facexlib`'s `face_size=512`
/// (`PuLID/pulid/pipeline_flux.py:47-53`).
pub const EVA_CROP_SIZE: u32 = 512;

/// facexlib's `borderValue=(135, 133, 132)` — a neutral grey
/// (`face_restoration_helper.py:258-259`). That tuple is **BGR**, because
/// facexlib works on `cv2`-decoded images; mold warps RGB, so the channels are
/// reversed here. Feeding `(135, 133, 132)` as RGB would tint every out-of-frame
/// border the wrong way.
pub const EVA_CROP_BORDER_RGB: [u8; 3] = [132, 133, 135];

/// Why identity extraction could not produce features.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// The detector found nothing at or above its score threshold.
    #[error("no face was detected in the identity image")]
    NoFaceDetected,
    /// The image could not be decoded.
    #[error("the identity image could not be decoded: {0}")]
    Decode(String),
    /// The landmarks could not be aligned to a template.
    #[error("the detected face could not be aligned: {0}")]
    Alignment(String),
    /// Anything in the model, graph, or evaluator path.
    #[error(transparent)]
    Runtime(#[from] anyhow::Error),
}

/// Everything one identity image yields.
#[derive(Debug, Clone)]
pub struct IdentityFeatures {
    /// The 512-d ArcFace embedding. Raw, as PuLID conditions on it — see
    /// [`arcface::ArcFaceEmbedding`].
    pub arcface: ArcFaceEmbedding,
    /// The 512x512 crop the EVA vision tower takes (#1229).
    pub eva_crop_512: RgbImage,
    /// The five landmarks both crops were fitted from, in source pixels.
    pub landmarks: Landmarks5,
    /// The chosen face's bounding box and score.
    pub face: DetectedFace,
    /// A caller-surfacable advisory, e.g. that several faces were found and
    /// the largest was used. Travels to the client as
    /// `x-mold-request-warning`.
    pub warning: Option<String>,
}

/// The loaded face-extraction stack.
///
/// Loading decodes both graphs eagerly, which is the point: a broken or
/// substituted model must fail at load rather than at the first render.
pub struct IdentityExtractor {
    detector: ScrfdDetector,
    recognizer: ArcFaceRecognizer,
    detector_sha256: String,
    recognizer_sha256: String,
}

impl IdentityExtractor {
    /// Load the detector and recognizer from a resolved PuLID bundle.
    ///
    /// `device` is accepted for symmetry with every other engine and is
    /// asserted, not honoured: `candle-onnx` materializes every tensor on the
    /// CPU (see [`arcface::ArcFaceRecognizer::device`]), so a caller that
    /// believes it placed this on a GPU would be wrong. A non-CPU request is
    /// an explicit error rather than a silent demotion.
    pub fn load(paths: &PulidPaths, device: &candle_core::Device) -> Result<Self> {
        if !device.is_cpu() {
            anyhow::bail!(
                "PuLID face extraction runs on the CPU: candle-onnx materializes every \
                 initializer on Device::Cpu (candle-onnx/src/eval.rs:191-232)"
            );
        }
        Self::from_paths(&paths.face_detector, &paths.face_recognizer)
    }

    /// Load from explicit model paths. Used by tests, which hold the files
    /// without a configured mold home.
    ///
    /// Both graphs are authenticated against the manifest's SHA-256 pins
    /// before they are decoded — the paths may be arbitrary, but the bytes at
    /// them may not. There is deliberately no unverified variant of this
    /// constructor: an extractor built from a graph nobody vouched for is the
    /// thing the pin exists to prevent. Tools that inspect arbitrary graphs
    /// call [`onnx_graph::load_onnx_model`] with `None` instead, and get no
    /// extractor out of it.
    pub fn from_paths(detector: &Path, recognizer: &Path) -> Result<Self> {
        let det = onnx_graph::load_onnx_model(
            detector,
            onnx_graph::pinned_artifact(ModelComponent::FaceDetector),
        )?;
        let rec = onnx_graph::load_onnx_model(
            recognizer,
            onnx_graph::pinned_artifact(ModelComponent::FaceRecognizer),
        )?;
        Ok(Self {
            detector: ScrfdDetector::new(det.model).context("loading the SCRFD detector")?,
            recognizer: ArcFaceRecognizer::new(rec.model)
                .context("loading the ArcFace recognizer")?,
            detector_sha256: det.sha256,
            recognizer_sha256: rec.sha256,
        })
    }

    /// SHA-256 of the detector bytes this extractor decoded.
    pub fn detector_sha256(&self) -> &str {
        &self.detector_sha256
    }

    /// SHA-256 of the recognizer bytes this extractor decoded.
    pub fn recognizer_sha256(&self) -> &str {
        &self.recognizer_sha256
    }

    /// Extract identity features from encoded image bytes.
    ///
    /// The decode is EXIF-oriented, through the crate's single orientation
    /// path ([`crate::img_utils::decode_oriented_srgb`]). A phone photograph
    /// carries its rotation in an EXIF tag rather than in the pixels, so a
    /// plain `load_from_memory` hands SCRFD a sideways face: the detector
    /// either misses it outright or reports landmarks in a frame every
    /// downstream crop then inherits. Upstream orients too — PuLID reads
    /// through `cv2.imread` (`pipeline_flux.py:124`), and OpenCV applies EXIF
    /// orientation unless `IMREAD_IGNORE_ORIENTATION` is set.
    ///
    /// It additionally converts an embedded ICC profile to sRGB, which
    /// `cv2.imread` does not. That is a deliberate improvement, not a parity
    /// gap: an untagged sRGB image — every parity fixture — takes the
    /// identical path, and a tagged one would otherwise have its colors
    /// misread into the embedding.
    pub fn extract(
        &self,
        image_bytes: &[u8],
    ) -> std::result::Result<IdentityFeatures, IdentityError> {
        let image = crate::img_utils::decode_oriented_srgb(image_bytes)
            .map_err(|e| IdentityError::Decode(format!("{e:#}")))?;
        self.extract_rgb(&image)
    }

    /// Extract identity features from an already-decoded RGB image.
    pub fn extract_rgb(
        &self,
        image: &RgbImage,
    ) -> std::result::Result<IdentityFeatures, IdentityError> {
        let faces = self.detector.detect(image)?;
        let (face, warning) = select_face(&faces).ok_or(IdentityError::NoFaceDetected)?;

        let arcface = self
            .recognizer
            .embed(image, &face.landmarks)
            .map_err(IdentityError::Runtime)?;

        let m = estimate_facexlib_512(&face.landmarks)
            .map_err(|e| IdentityError::Alignment(e.to_string()))?;
        let eva_crop_512 =
            warp_affine(image, &m, EVA_CROP_SIZE, EVA_CROP_SIZE, EVA_CROP_BORDER_RGB).ok_or_else(
                || {
                    IdentityError::Alignment(
                        "the EVA crop transform was not invertible".to_string(),
                    )
                },
            )?;

        Ok(IdentityFeatures {
            arcface,
            eva_crop_512,
            landmarks: face.landmarks,
            face,
            warning,
        })
    }
}

/// Pick the face PuLID conditions on, and say so when the choice was not
/// forced.
///
/// `PuLID/pulid/pipeline_flux.py:127-129` sorts by bounding-box area and takes
/// the last, i.e. the largest — not the most confident, and not the most
/// central. Several faces is not an error, but it IS a decision the user did
/// not make, so it returns a warning the caller surfaces through
/// `x-mold-request-warning`.
pub fn select_face(faces: &[DetectedFace]) -> Option<(DetectedFace, Option<String>)> {
    let largest = faces.iter().copied().max_by(|a, b| {
        a.area()
            .partial_cmp(&b.area())
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let warning = (faces.len() > 1).then(|| {
        format!(
            "{} faces were detected in the identity image; conditioning on the largest one",
            faces.len()
        )
    });
    Some((largest, warning))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(bbox: [f32; 4], score: f32) -> DetectedFace {
        DetectedFace {
            bbox,
            score,
            landmarks: [[0.0; 2]; 5],
        }
    }

    #[test]
    fn no_faces_is_none_so_the_caller_can_raise_the_typed_error() {
        assert!(select_face(&[]).is_none());
        // The facade turns that into the typed variant.
        let err = IdentityError::NoFaceDetected;
        assert_eq!(
            err.to_string(),
            "no face was detected in the identity image"
        );
    }

    #[test]
    fn one_face_carries_no_warning() {
        let (chosen, warning) = select_face(&[face([0.0, 0.0, 10.0, 10.0], 0.9)]).unwrap();
        assert_eq!(chosen.bbox, [0.0, 0.0, 10.0, 10.0]);
        assert!(warning.is_none());
    }

    #[test]
    fn several_faces_pick_the_largest_by_area_not_by_score() {
        let small_but_confident = face([0.0, 0.0, 10.0, 10.0], 0.99);
        let large_but_less_confident = face([50.0, 50.0, 150.0, 170.0], 0.61);
        let (chosen, warning) =
            select_face(&[small_but_confident, large_but_less_confident]).unwrap();
        assert_eq!(chosen.bbox, large_but_less_confident.bbox);
        let warning = warning.expect("an unforced choice must be reported");
        assert!(warning.contains("2 faces"), "{warning}");
        assert!(warning.contains("largest"), "{warning}");
    }

    #[test]
    fn the_eva_border_is_facexlibs_bgr_grey_reversed() {
        // facexlib passes (135, 133, 132) to cv2 on a BGR image.
        assert_eq!(EVA_CROP_BORDER_RGB, [132, 133, 135]);
    }

    #[test]
    fn a_gpu_placement_request_is_refused_rather_than_silently_demoted() {
        let paths = PulidPaths {
            adapter: Path::new("/nonexistent/adapter.safetensors").to_path_buf(),
            vision_encoder_source: Path::new("/nonexistent/eva.pt").to_path_buf(),
            face_detector: Path::new("/nonexistent/scrfd.onnx").to_path_buf(),
            face_recognizer: Path::new("/nonexistent/glintr100.onnx").to_path_buf(),
        };
        // A CPU device gets as far as opening the (absent) files; only the
        // device check can fail before that, which is what this pins.
        let cpu_err = match IdentityExtractor::load(&paths, &candle_core::Device::Cpu) {
            Ok(_) => panic!("nonexistent models must not load"),
            Err(e) => e,
        };
        assert!(
            format!("{cpu_err:#}").contains("scrfd.onnx"),
            "a CPU load must reach the file open: {cpu_err:#}"
        );
    }
}
