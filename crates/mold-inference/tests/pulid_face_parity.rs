//! Parity for PuLID face extraction (#1222) against the Python pipeline.
//!
//! Two tiers, deliberately separated so CI stays hermetic:
//!
//! * **Hermetic** — everything that does not need the InsightFace weights: the
//!   Step-0 op gate reading the committed inventory, and the alignment and warp
//!   goldens, which are checkable from the committed landmarks alone.
//! * **Weight-gated** (`#[ignore]`) — the detector and recognizer themselves.
//!   Run with the antelopev2 files present:
//!
//!   ```text
//!   MOLD_TEST_PULID_ASSETS=/path/to/antelopev2 \
//!     cargo test -p mold-ai-inference --features pulid \
//!     --test pulid_face_parity -- --ignored --nocapture
//!   ```
//!
//! The goldens were captured by `testdata/pulid/capture_goldens.py` from the
//! SHA-pinned models; `testdata/pulid/faces/README.md` records the licenses,
//! the capture commit, and every tolerance below with the number that earned
//! it.

#![cfg(feature = "pulid")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image::RgbImage;
use mold_inference::identity::align::{
    estimate_arcface_norm, estimate_facexlib_512, Landmarks5, ARCFACE_DST_112, FACEXLIB_FFHQ_512,
};
use mold_inference::identity::arcface::{norm_crop, ArcFaceEmbedding};
use mold_inference::identity::onnx_inventory::{
    graph_inventory, ignored_attributes, unsupported_by_candle_onnx, GraphInventory,
};
use mold_inference::identity::warp::{warp_affine, Affine2x3};
use mold_inference::identity::{IdentityExtractor, EVA_CROP_BORDER_RGB, EVA_CROP_SIZE};
use serde::Deserialize;

/// Manifest SHA-256 pins, `crates/mold-core/src/manifest.rs`
/// (`pulid_manifests`). The inventory fixture is only evidence about the
/// graphs mold actually ships if it was captured from these exact bytes.
const DETECTOR_SHA256: &str = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91";
const RECOGNIZER_SHA256: &str = "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf";

/// SCRFD's letterbox quantizes to 640 px, so a landmark carries roughly one
/// detector pixel of slack per source pixel of scale. Measured worst case
/// across the four fixtures is recorded in the faces README; the budget is set
/// a little above it so a resampler tweak is a visible regression rather than
/// a flake.
const LANDMARK_TOLERANCE_PX: f64 = 1.0;
/// PuLID's own identity check treats anything below ~0.9 as a different
/// person; the issue's gate is 0.99 and the measured value is far above it.
const ARCFACE_COSINE_FLOOR: f32 = 0.99;
/// Element-wise budget for a 2x3 similarity matrix against skimage and
/// OpenCV. Measured worst case across the fixture set is 1.74e-5 against
/// skimage and 1.14e-5 against `cv2.LMEDS`; the translation terms are hundreds
/// of pixels, so that is ~1e-7 relative and the deviation documented in
/// `identity/align.rs` costs nothing measurable.
const AFFINE_TOLERANCE: f64 = 1e-4;
/// mold evaluates the warp in `f64`; OpenCV uses 5-bit fixed-point
/// interpolation weights. Measured worst case is a mean absolute difference of
/// 0.23/255 with a 99.9th percentile of 2 LSB, on both crops.
const WARP_MEAN_ABS_TOLERANCE: f64 = 0.6;
const WARP_P999_ABS_TOLERANCE: u8 = 4;

#[derive(Debug, Deserialize)]
struct Golden {
    image: String,
    faces_detected: usize,
    bbox: [f64; 4],
    score: f64,
    landmarks: [[f64; 2]; 5],
    m112: [[f64; 3]; 2],
    m512: [[f64; 3]; 2],
    m512_skimage: [[f64; 3]; 2],
    embedding: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct InventoryFixture {
    sha256: String,
    bytes: usize,
    inventory: GraphInventory,
}

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/pulid")
        .canonicalize()
        .expect("testdata/pulid is committed")
}

fn goldens() -> Vec<(Golden, PathBuf)> {
    let faces = testdata().join("faces");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&faces).expect("fixture faces directory") {
        let path = entry.expect("readable fixture entry").path();
        if path.to_string_lossy().ends_with(".golden.json") {
            let golden: Golden =
                serde_json::from_slice(&std::fs::read(&path).expect("readable golden"))
                    .unwrap_or_else(|e| panic!("malformed golden {}: {e}", path.display()));
            out.push((golden, faces.clone()));
        }
    }
    out.sort_by(|a, b| a.0.image.cmp(&b.0.image));
    assert!(
        out.len() >= 3,
        "the parity set needs at least three faces, found {}",
        out.len()
    );
    out
}

fn landmarks_of(golden: &Golden) -> Landmarks5 {
    golden.landmarks
}

fn max_affine_delta(a: &Affine2x3, b: &[[f64; 3]; 2]) -> f64 {
    let mut worst = 0.0f64;
    for r in 0..2 {
        for c in 0..3 {
            worst = worst.max((a[r][c] - b[r][c]).abs());
        }
    }
    worst
}

/// Mean absolute per-channel difference and the 99.9th-percentile absolute
/// difference between two same-sized images.
fn image_delta(a: &RgbImage, b: &RgbImage) -> (f64, u8) {
    assert_eq!(a.dimensions(), b.dimensions(), "golden size mismatch");
    let mut histogram = [0u64; 256];
    let mut total = 0u64;
    let mut count = 0u64;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        for c in 0..3 {
            let d = pa.0[c].abs_diff(pb.0[c]);
            histogram[d as usize] += 1;
            total += d as u64;
            count += 1;
        }
    }
    let cutoff = (count as f64 * 0.999).ceil() as u64;
    let mut running = 0u64;
    let mut p999 = 0u8;
    for (value, hits) in histogram.iter().enumerate() {
        running += hits;
        if running >= cutoff {
            p999 = value as u8;
            break;
        }
    }
    (total as f64 / count as f64, p999)
}

// ---------------------------------------------------------------------------
// Hermetic: Step 0's op gate, read from the committed inventory.
// ---------------------------------------------------------------------------

fn inventory_fixture() -> BTreeMap<String, InventoryFixture> {
    let path = testdata().join("onnx-inventory.json");
    serde_json::from_slice(&std::fs::read(&path).expect("committed onnx inventory"))
        .expect("the inventory fixture parses")
}

#[test]
fn the_inventory_fixture_was_captured_from_the_manifest_pinned_models() {
    let fixture = inventory_fixture();
    assert_eq!(fixture["scrfd_10g_bnkps.onnx"].sha256, DETECTOR_SHA256);
    assert_eq!(fixture["glintr100.onnx"].sha256, RECOGNIZER_SHA256);
    assert_eq!(fixture["scrfd_10g_bnkps.onnx"].bytes, 16_923_827);
    assert_eq!(fixture["glintr100.onnx"].bytes, 260_665_334);
}

/// Step 0's op gate. A candle bump that drops or restricts an op fails here,
/// hermetically, before any weight is downloaded.
#[test]
fn candle_onnx_implements_every_op_in_both_graphs() {
    for (name, entry) in inventory_fixture() {
        let unsupported = unsupported_by_candle_onnx(&entry.inventory);
        assert!(
            unsupported.is_empty(),
            "{name} uses ops candle-onnx cannot run: {}",
            unsupported
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
}

/// The one attribute candle-onnx drops, pinned so it can never grow silently.
#[test]
fn the_only_ignored_attribute_is_scrfds_pooling_ceil_mode() {
    let fixture = inventory_fixture();
    let ignored = ignored_attributes(&fixture["scrfd_10g_bnkps.onnx"].inventory);
    assert_eq!(ignored.len(), 1, "{ignored:?}");
    assert_eq!(ignored[0].op_type, "AveragePool");
    assert_eq!(ignored[0].attribute, "ceil_mode");
    assert!(ignored_attributes(&fixture["glintr100.onnx"].inventory).is_empty());
}

/// The detector's static output rows are what pins `_num_anchors = 2` and the
/// 640 input; a graph that disagreed would decode into the wrong anchors.
#[test]
fn the_detector_declares_the_nine_keypoint_outputs_the_decoder_assumes() {
    let fixture = inventory_fixture();
    let detector = &fixture["scrfd_10g_bnkps.onnx"].inventory;
    assert_eq!(detector.outputs.len(), 9);
    let rows: Vec<i64> = detector.outputs.iter().map(|o| o.dims[0]).collect();
    assert_eq!(
        rows,
        vec![12800, 3200, 800, 12800, 3200, 800, 12800, 3200, 800]
    );
    let widths: Vec<i64> = detector.outputs.iter().map(|o| o.dims[1]).collect();
    assert_eq!(widths, vec![1, 1, 1, 4, 4, 4, 10, 10, 10]);
    let recognizer = &fixture["glintr100.onnx"].inventory;
    assert_eq!(recognizer.outputs.len(), 1);
    assert_eq!(recognizer.outputs[0].dims, vec![1, 512]);
}

// ---------------------------------------------------------------------------
// Hermetic: alignment and warp parity, from the committed landmarks.
// ---------------------------------------------------------------------------

/// mold's Umeyama fit against skimage's, which is what
/// `face_align.estimate_norm` uses.
#[test]
fn the_arcface_fit_matches_skimages_similarity_transform() {
    for (golden, _) in goldens() {
        let m = estimate_arcface_norm(&landmarks_of(&golden), 112).expect("a fittable face");
        let delta = max_affine_delta(&m, &golden.m112);
        assert!(
            delta < AFFINE_TOLERANCE,
            "{}: max|mold - skimage| on m112 = {delta:e}",
            golden.image
        );
    }
}

/// mold's least-squares similarity against BOTH references: skimage's (which
/// it ports) and OpenCV's actual `LMEDS` output (which facexlib calls, and
/// which mold deliberately does not reproduce randomly). The second assertion
/// is what turns the documented deviation into a measured one.
#[test]
fn the_eva_fit_matches_skimage_and_opencvs_lmeds() {
    for (golden, _) in goldens() {
        let m = estimate_facexlib_512(&landmarks_of(&golden)).expect("a fittable face");
        let vs_skimage = max_affine_delta(&m, &golden.m512_skimage);
        println!(
            "{}: max|mold - skimage| = {vs_skimage:e}, max|mold - cv2.LMEDS| = {:e}",
            golden.image,
            max_affine_delta(&m, &golden.m512)
        );
        assert!(
            vs_skimage < AFFINE_TOLERANCE,
            "{}: max|mold - skimage| on m512 = {vs_skimage:e}",
            golden.image
        );
        let vs_lmeds = max_affine_delta(&m, &golden.m512);
        assert!(
            vs_lmeds < AFFINE_TOLERANCE,
            "{}: max|mold - cv2.LMEDS| on m512 = {vs_lmeds:e}",
            golden.image
        );
    }
}

/// The 112 crop mold hands `glintr100`, against `cv2.warpAffine`'s.
#[test]
fn the_arcface_crop_matches_opencvs_warp() {
    for (golden, dir) in goldens() {
        let stem = golden.image.trim_end_matches(".jpg");
        let source = image::open(dir.join(&golden.image))
            .expect("fixture image decodes")
            .to_rgb8();
        let mine = norm_crop(&source, &landmarks_of(&golden)).expect("a warpable face");
        let theirs = image::open(dir.join(format!("{stem}.arcface112.png")))
            .expect("golden 112 crop")
            .to_rgb8();
        let (mean, p999) = image_delta(&mine, &theirs);
        println!("{}: 112 crop mean {mean:.4}, p99.9 {p999}", golden.image);
        assert!(
            mean < WARP_MEAN_ABS_TOLERANCE && p999 <= WARP_P999_ABS_TOLERANCE,
            "{}: 112 crop mean|delta| = {mean:.3}, p99.9 = {p999}",
            golden.image
        );
    }
}

/// The 512 crop #1229 conditions on, against facexlib's.
#[test]
fn the_eva_crop_matches_facexlibs_warp() {
    for (golden, dir) in goldens() {
        let stem = golden.image.trim_end_matches(".jpg");
        let source = image::open(dir.join(&golden.image))
            .expect("fixture image decodes")
            .to_rgb8();
        let m = estimate_facexlib_512(&landmarks_of(&golden)).expect("a fittable face");
        let mine = warp_affine(
            &source,
            &m,
            EVA_CROP_SIZE,
            EVA_CROP_SIZE,
            EVA_CROP_BORDER_RGB,
        )
        .expect("an invertible fit");
        let theirs = image::open(dir.join(format!("{stem}.eva512.png")))
            .expect("golden 512 crop")
            .to_rgb8();
        let (mean, p999) = image_delta(&mine, &theirs);
        println!("{}: 512 crop mean {mean:.4}, p99.9 {p999}", golden.image);
        assert!(
            mean < WARP_MEAN_ABS_TOLERANCE && p999 <= WARP_P999_ABS_TOLERANCE,
            "{}: 512 crop mean|delta| = {mean:.3}, p99.9 = {p999}",
            golden.image
        );
    }
}

/// Both templates are load-bearing constants; a transcription slip would show
/// up as a plausible-looking but subtly wrong crop.
#[test]
fn the_committed_goldens_were_fitted_to_the_templates_mold_ships() {
    for (golden, _) in goldens() {
        let landmarks = landmarks_of(&golden);
        let m112 = estimate_arcface_norm(&landmarks, 112).unwrap();
        let m512 = estimate_facexlib_512(&landmarks).unwrap();
        let residual112 =
            mold_inference::identity::align::fit_residual_rms(&m112, &landmarks, &ARCFACE_DST_112);
        let residual512 = mold_inference::identity::align::fit_residual_rms(
            &m512,
            &landmarks,
            &FACEXLIB_FFHQ_512,
        );
        // A real frontal face lands close to both templates; a wrong template
        // would blow these up long before it changed the matrix comparison.
        // Measured worst case across the fixture set: 6.20 and 14.86 px.
        println!(
            "{}: template rms 112 = {residual112:.4} px, 512 = {residual512:.4} px",
            golden.image
        );
        assert!(residual112 < 9.0, "{}: 112 rms {residual112}", golden.image);
        assert!(
            residual512 < 22.0,
            "{}: 512 rms {residual512}",
            golden.image
        );
    }
}

// ---------------------------------------------------------------------------
// Weight-gated: the graphs themselves.
// ---------------------------------------------------------------------------

fn extractor() -> Option<IdentityExtractor> {
    let dir = PathBuf::from(std::env::var_os("MOLD_TEST_PULID_ASSETS")?);
    Some(
        IdentityExtractor::from_paths(
            &dir.join("scrfd_10g_bnkps.onnx"),
            &dir.join("glintr100.onnx"),
        )
        .expect("the antelopev2 models load"),
    )
}

#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn detection_and_embedding_match_insightface() {
    let extractor = extractor().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    assert_eq!(extractor.detector_sha256(), DETECTOR_SHA256);
    assert_eq!(extractor.recognizer_sha256(), RECOGNIZER_SHA256);

    let mut worst_landmark = 0.0f64;
    let mut worst_cosine = 1.0f32;
    for (golden, dir) in goldens() {
        let bytes = std::fs::read(dir.join(&golden.image)).expect("fixture image");
        let features = extractor
            .extract(&bytes)
            .unwrap_or_else(|e| panic!("{}: {e}", golden.image));

        assert_eq!(
            features.warning.is_some(),
            golden.faces_detected > 1,
            "{}: warning must appear exactly when the choice was unforced",
            golden.image
        );

        for (i, (mine, theirs)) in features
            .landmarks
            .iter()
            .zip(golden.landmarks.iter())
            .enumerate()
        {
            let d = ((mine[0] - theirs[0]).powi(2) + (mine[1] - theirs[1]).powi(2)).sqrt();
            worst_landmark = worst_landmark.max(d);
            assert!(
                d < LANDMARK_TOLERANCE_PX,
                "{}: landmark {i} is {d:.4} px from insightface's",
                golden.image
            );
        }
        for (i, (mine, theirs)) in features
            .face
            .bbox
            .iter()
            .zip(golden.bbox.iter())
            .enumerate()
        {
            let d = (*mine as f64 - theirs).abs();
            assert!(d < 2.0, "{}: bbox[{i}] off by {d:.4} px", golden.image);
        }
        let score_delta = (features.face.score as f64 - golden.score).abs();
        assert!(
            score_delta < 0.02,
            "{}: score off by {score_delta:.5}",
            golden.image
        );

        let reference = ArcFaceEmbedding {
            raw: golden.embedding.iter().map(|v| *v as f32).collect(),
        };
        let cosine = features.arcface.cosine_similarity(&reference);
        worst_cosine = worst_cosine.min(cosine);
        assert!(
            cosine >= ARCFACE_COSINE_FLOOR,
            "{}: ArcFace cosine {cosine:.6} < {ARCFACE_COSINE_FLOOR}",
            golden.image
        );

        assert_eq!(features.eva_crop_512.dimensions(), (512, 512));
        println!(
            "{}: cosine {cosine:.6}, |raw| {:.3} (reference {:.3})",
            golden.image,
            features
                .arcface
                .raw
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt(),
            reference.raw.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
    }
    println!("worst landmark error {worst_landmark:.4} px, worst cosine {worst_cosine:.6}");
}

/// A photograph with no face is a clear typed error, never an empty embedding.
#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn an_image_with_no_face_is_a_typed_error() {
    let extractor = extractor().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    let blank = RgbImage::from_pixel(640, 480, image::Rgb([128, 128, 128]));
    match extractor.extract_rgb(&blank) {
        Err(mold_inference::identity::IdentityError::NoFaceDetected) => {}
        other => panic!("expected NoFaceDetected, got {other:?}"),
    }
}

/// The graphs mold actually ships must produce the committed inventory, not
/// merely one that happens to pass.
#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_committed_inventory_still_describes_the_real_models() {
    let dir = PathBuf::from(
        std::env::var_os("MOLD_TEST_PULID_ASSETS")
            .expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory"),
    );
    let fixture = inventory_fixture();
    for name in ["scrfd_10g_bnkps.onnx", "glintr100.onnx"] {
        let loaded = mold_inference::identity::onnx_graph::load_onnx_model(&dir.join(name))
            .expect("the model loads");
        assert_eq!(loaded.sha256, fixture[name].sha256, "{name} digest drifted");
        let live = graph_inventory(&loaded.model).expect("the graph parses");
        // The load normalizes `Resize`'s empty optional inputs, so compare the
        // op signatures rather than the raw node inputs, which that rewrite
        // does not touch anyway.
        assert_eq!(
            live.signatures, fixture[name].inventory.signatures,
            "{name}'s op inventory drifted from the committed fixture"
        );
    }
}
