//! Face detection and identity embedding for PuLID (#1222).
//!
//! Shared by both families: `pipeline_v1_1.py:get_id_embedding` runs the same
//! detector, recognizer, parser, and vision tower as `pipeline_flux.py`'s, and
//! only the IDFormer's checkpoint prefix differs
//! (`extraction::idformer_prefix`).
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
//! The BiSeNet mask PuLID applies to that crop before the tower sees it
//! (`PuLID/pulid/pipeline_flux.py:161-170`) lives in [`parsing`] and is
//! applied by [`extraction`], which owns the crop-to-tower step.
//!
//! Deliberately NOT implemented here, and named rather than silently skipped:
//! facexlib's RetinaFace detector, which upstream uses for the 512 crop's
//! landmarks while taking the ArcFace embedding from InsightFace's SCRFD
//! (`pipeline_flux.py:127-147`). Mold warps both crops from ONE SCRFD
//! detection. #1225 measured that divergence end to end rather than assuming
//! it away; the numbers are in `docs/architecture/pulid-face-extraction.md`.

pub mod align;
pub mod arcface;
pub mod arcface_net;
pub mod extraction;
pub mod onnx_graph;
pub mod onnx_inventory;
pub mod onnx_weights;
pub mod parsing;
pub mod scrfd;
pub mod scrfd_net;
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
    /// A refusal about ONE photograph of a set, naming which.
    ///
    /// A wrapper rather than a re-worded variant, because the category has to
    /// survive: #1227 phase 2 made the category decide whether a failure
    /// touches the GPU's health, and collapsing "no face in photo 2 of 3" into
    /// [`Self::Runtime`] would report a bad photograph as a bad card. The
    /// message is byte-identical to the `format!` this replaced.
    #[error("identity photo {index} of {count}: {source}")]
    Photo {
        /// One-based, as the caller supplied them.
        index: usize,
        count: usize,
        #[source]
        source: Box<IdentityError>,
    },
}

impl IdentityError {
    /// Whether this refusal is about the PHOTOGRAPH the caller supplied rather
    /// than about the machine that tried to read it.
    ///
    /// #1227 phase 2 runs the extraction on the render's leased GPU, so its
    /// failures now reach a worker's reliability counter. Three unusable
    /// photographs must not degrade a healthy card out of rotation for a
    /// minute: a face that is not there, a payload that will not decode, and
    /// landmarks that will not fit the template are all answers about the
    /// input, reproducible on any device, and the caller's to fix.
    ///
    /// [`Self::Runtime`] is the only device-attributable arm — a load failure,
    /// a kernel failure, a driver fault — and is the only one that counts.
    pub fn is_user_input(&self) -> bool {
        match self {
            Self::NoFaceDetected | Self::Decode(_) | Self::Alignment(_) => true,
            Self::Runtime(_) => false,
            // A set does not change whose fault a photograph is.
            Self::Photo { source, .. } => source.is_user_input(),
        }
    }

    /// Name which photograph of a set this refusal is about, preserving its
    /// category. A one-photograph set is the singular form and is returned
    /// unchanged, so its message stays exactly what every surface has shown.
    pub fn in_photo_set(self, index: usize, count: usize) -> Self {
        if count == 1 {
            return self;
        }
        Self::Photo {
            index,
            count,
            source: Box::new(self),
        }
    }
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
    /// `device` is HONOURED as of #1227 phase 2. Phase 1 made both networks
    /// ordinary resident candle modules ([`scrfd_net::ScrfdNet::new`],
    /// [`arcface_net::IResNet100::new`] both place their weights), and phase 2
    /// moved the extraction inside the leased job, so there is finally a device
    /// to name (`docs/architecture/pulid-perf.md` §5). The assertion this used
    /// to carry — "extraction runs at admission, before a device is leased" —
    /// described the call site rather than the arithmetic, and the call site
    /// moved.
    pub fn load(paths: &PulidPaths, device: &candle_core::Device) -> Result<Self> {
        Self::from_paths_on_device(&paths.face_detector, &paths.face_recognizer, device)
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
        Self::from_paths_on_device(detector, recognizer, &candle_core::Device::Cpu)
    }

    /// [`Self::from_paths`], placing both networks on `device`.
    pub fn from_paths_on_device(
        detector: &Path,
        recognizer: &Path,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let det = onnx_graph::load_onnx_model(
            detector,
            onnx_graph::pinned_artifact(ModelComponent::FaceDetector),
        )?;
        let rec = onnx_graph::load_onnx_model(
            recognizer,
            onnx_graph::pinned_artifact(ModelComponent::FaceRecognizer),
        )?;
        Ok(Self {
            detector: ScrfdDetector::new_on_device(det.model, device)
                .context("loading the SCRFD detector")?,
            recognizer: ArcFaceRecognizer::new_on_device(rec.model, device)
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
///
/// Upstream applies that rule to the ArcFace branch only. Its 512 crop comes
/// from a SECOND detector, facexlib's RetinaFace, run with
/// `only_center_face=True` (`pipeline_flux.py:145`), which on a group
/// photograph selects a different person — `get_center_face` minimizes
/// distance to the image centre (`face_restoration_helper.py:152-163`) while
/// `get_largest_face` maximizes area (`:71-89`). The two halves of one
/// identity would then describe two faces, which is upstream's defect and not
/// a behaviour to reproduce. Mold runs ONE detection and hands the same face
/// to both crops; #1225 measured what that costs on a single-face photograph
/// (where the two rules agree by construction) and recorded it in
/// `docs/architecture/pulid-face-extraction.md`.
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

    /// The centre-vs-largest divergence, made concrete.
    ///
    /// Upstream would embed the small central face and crop the large offset
    /// one. Mold picks one face for both, and says which.
    #[test]
    fn one_detection_serves_both_crops_even_when_centre_and_largest_disagree() {
        // A 1000x1000 frame: a small face dead centre, a large one off to the
        // side. `get_center_face` would take the first, `get_largest_face`
        // the second.
        let central = face([460.0, 460.0, 540.0, 540.0], 0.95);
        let large = face([600.0, 100.0, 900.0, 500.0], 0.90);
        assert!(central.area() < large.area());
        let centre_of_frame = [500.0, 500.0];
        let distance = |f: &DetectedFace| {
            let cx = (f.bbox[0] + f.bbox[2]) / 2.0 - centre_of_frame[0];
            let cy = (f.bbox[1] + f.bbox[3]) / 2.0 - centre_of_frame[1];
            (cx * cx + cy * cy).sqrt()
        };
        assert!(
            distance(&central) < distance(&large),
            "the fixture must actually make the two rules disagree"
        );

        let (chosen, warning) = select_face(&[central, large]).unwrap();
        assert_eq!(chosen.bbox, large.bbox, "mold follows the ArcFace rule");
        assert!(warning.is_some(), "an unforced choice is reported");
    }

    #[test]
    fn the_eva_border_is_facexlibs_bgr_grey_reversed() {
        // facexlib passes (135, 133, 132) to cv2 on a BGR image.
        assert_eq!(EVA_CROP_BORDER_RGB, [132, 133, 135]);
    }

    /// Whose fault a refusal is decides whether it touches a GPU's health
    /// counter (#1227 phase 2), so the category has to survive being told
    /// WHICH photograph of a set it came from.
    #[test]
    fn naming_the_photograph_preserves_whose_fault_it_is() {
        // A one-photograph set is the singular form: unchanged, so its message
        // stays exactly what every surface has always shown.
        let single = IdentityError::NoFaceDetected.in_photo_set(1, 1);
        assert!(matches!(single, IdentityError::NoFaceDetected));
        assert_eq!(
            single.to_string(),
            "no face was detected in the identity image"
        );
        assert!(single.is_user_input());

        // A real set names the photograph and keeps the category. The message
        // is byte-identical to the `format!` this replaced.
        let located = IdentityError::NoFaceDetected.in_photo_set(2, 3);
        assert_eq!(
            located.to_string(),
            "identity photo 2 of 3: no face was detected in the identity image"
        );
        assert!(
            located.is_user_input(),
            "a faceless photograph is the caller's, whether it is one of one or one of three"
        );

        for user_input in [
            IdentityError::Decode("not a PNG or JPEG".to_string()),
            IdentityError::Alignment("degenerate landmarks".to_string()),
        ] {
            assert!(user_input.in_photo_set(1, 4).is_user_input());
        }

        // And a device fault stays a device fault.
        let runtime = IdentityError::Runtime(anyhow::anyhow!("CUDA_ERROR_ILLEGAL_ADDRESS"))
            .in_photo_set(3, 4);
        assert!(!runtime.is_user_input());
        assert!(runtime.to_string().contains("identity photo 3 of 4"));
    }

    /// Phase 1's `a_gpu_placement_request_is_refused_rather_than_silently_demoted`
    /// inverted: #1227 phase 2 moved extraction inside the leased job, so a
    /// non-CPU device is now the ordinary case and must reach the same file
    /// open a CPU device does. A refusal here would mean the extraction had
    /// silently stayed on the host after the phase moved.
    #[test]
    fn a_device_placement_request_reaches_the_models_rather_than_being_refused() {
        let paths = PulidPaths {
            family: mold_core::identity::IdentityFamily::Flux,
            adapter: Path::new("/nonexistent/adapter.safetensors").to_path_buf(),
            vision_encoder_source: Path::new("/nonexistent/eva.pt").to_path_buf(),
            face_detector: Path::new("/nonexistent/scrfd.onnx").to_path_buf(),
            face_recognizer: Path::new("/nonexistent/glintr100.onnx").to_path_buf(),
            face_parser_source: Path::new("/nonexistent/parsing_bisenet.pth").to_path_buf(),
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
        // A real accelerator is not available in every build or on every CI
        // runner, so the device arm is asserted only where one exists. Where
        // it does, it must fail on the ABSENT MODEL, never on the placement.
        if let Ok(device) = candle_core::Device::new_metal(0) {
            let err = match IdentityExtractor::load(&paths, &device) {
                Ok(_) => panic!("nonexistent models must not load"),
                Err(err) => err,
            };
            assert!(
                format!("{err:#}").contains("scrfd.onnx"),
                "a device load must reach the file open too: {err:#}"
            );
        }
    }
}
