//! Parity for the hand-ported face networks (#1227) against the `candle-onnx`
//! path they replace.
//!
//! `docs/architecture/pulid-perf.md` §1 replaced `candle_onnx::simple_eval`
//! with resident `candle-core`/`candle-nn` forward passes
//! (`identity::scrfd_net`, `identity::arcface_net`) to stop re-materializing
//! 278 MB of initializers on every request. The port invents no numerics, so
//! the qualifying question is narrow and answerable exactly: **do the two
//! evaluators agree, tensor for tensor, on the pinned weights?**
//!
//! That is what this file asks. The *upstream* qualification — landmark ≤ 1.0
//! px, bbox ≤ 2.0 px, score ≤ 0.02, ArcFace cosine ≥ 0.99 against InsightFace's
//! own goldens — is unchanged and still lives in `pulid_face_parity.rs`, which
//! after the swap exercises this port rather than `simple_eval`. No new
//! tolerance is invented here; the numbers below are tighter than that gate by
//! design, because two float32 evaluators of the same graph should differ only
//! by summation order.
//!
//! Weight-gated, like every other test that needs the antelopev2 files:
//!
//! ```text
//! MOLD_TEST_PULID_ASSETS=/path/to/antelopev2 \
//!   cargo test -p mold-ai-inference --features pulid \
//!   --test pulid_handport_parity -- --ignored --nocapture
//! ```

#![cfg(feature = "pulid")]

use std::path::{Path, PathBuf};

use candle_core::Device;
use image::RgbImage;
use mold_inference::identity::align::Landmarks5;
use mold_inference::identity::arcface::{norm_crop, ArcFaceEmbedding, ArcFaceRecognizer};
use mold_inference::identity::arcface_net::IResNet100;
use mold_inference::identity::onnx_graph::load_onnx_model;
use mold_inference::identity::scrfd::ScrfdDetector;
use mold_inference::identity::scrfd_net::ScrfdNet;
use mold_inference::identity::warp::letterbox_top_left;
use serde::Deserialize;

/// SCRFD's port carries no batch-norm fold — the export already folded them —
/// and calls the same `candle-core` kernels in the same order, so measured
/// drift across the whole fixture set is **exactly zero** on every one of the
/// nine head tensors. The budget is not zero only because a different
/// platform's `rayon` split could reassociate a sum.
const SCORE_TOLERANCE: f32 = 1e-5;
/// Distance-encoded box and keypoint predictions are in stride units, roughly
/// `[-20, 20]` before scaling. Measured worst case: 0.
const DISTANCE_TOLERANCE: f32 = 1e-4;
/// Raw ArcFace components run to about ±5 with a vector norm near 20. This is
/// the one place the port differs arithmetically: `FoldedBatchNorm` collapses
/// the spec's divide-and-affine into one multiply-add per channel, which moves
/// the last bits. Measured worst case across the fixture set: 2.62e-6.
const EMBEDDING_TOLERANCE: f32 = 1e-4;
/// The upstream gate is 0.99 against InsightFace. Against the evaluator this
/// port replaced, anything below five nines means the graphs diverged.
const SELF_COSINE_FLOOR: f32 = 0.999_99;

#[derive(Debug, Deserialize)]
struct Golden {
    image: String,
    landmarks: [[f64; 2]; 5],
}

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/pulid")
        .canonicalize()
        .expect("crates/mold-inference/testdata/pulid is committed")
}

fn assets() -> Option<PathBuf> {
    std::env::var_os("MOLD_TEST_PULID_ASSETS").map(PathBuf::from)
}

/// Every committed face fixture with its InsightFace landmarks.
fn fixtures() -> Vec<(RgbImage, Landmarks5, String)> {
    let faces = testdata().join("faces");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&faces).expect("fixture faces directory") {
        let path = entry.expect("readable fixture entry").path();
        if !path.to_string_lossy().ends_with(".golden.json") {
            continue;
        }
        let golden: Golden =
            serde_json::from_slice(&std::fs::read(&path).expect("readable golden"))
                .unwrap_or_else(|e| panic!("malformed golden {}: {e}", path.display()));
        let image = image::open(faces.join(&golden.image))
            .expect("fixture image decodes")
            .to_rgb8();
        out.push((image, golden.landmarks, golden.image));
    }
    out.sort_by(|a, b| a.2.cmp(&b.2));
    assert!(out.len() >= 3, "the parity set needs at least three faces");
    out
}

fn max_abs(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: length {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Hermetic: the port is wired in, and refuses the wrong graph.
// ---------------------------------------------------------------------------

/// A detector graph handed the recognizer's weights (or any other graph whose
/// parameter order differs) must fail at LOAD with a shape complaint, never
/// produce a plausible-looking detection from misaligned tensors.
#[test]
fn a_graph_whose_parameters_do_not_match_the_port_is_refused_at_load() {
    let Some(dir) = assets() else {
        eprintln!("skipping: set MOLD_TEST_PULID_ASSETS to run the cross-graph refusal");
        return;
    };
    let recognizer = load_onnx_model(&dir.join("glintr100.onnx"), None).expect("the model loads");
    let rendered = match ScrfdNet::new(&recognizer.model, &Device::Cpu) {
        Ok(_) => panic!("the recognizer graph is not a detector and must not build one"),
        Err(err) => format!("{err:#}"),
    };
    assert!(
        rendered.contains("stem conv") || rendered.contains("shape"),
        "the refusal must name where the graph diverged: {rendered}"
    );
}

// ---------------------------------------------------------------------------
// Weight-gated: the two evaluators, tensor for tensor.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_scrfd_hand_port_matches_candle_onnx_on_every_head_tensor() {
    let dir = assets().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    let loaded = load_onnx_model(&dir.join("scrfd_10g_bnkps.onnx"), None).expect("the model loads");
    let net = ScrfdNet::new(&loaded.model, &Device::Cpu).expect("the hand port builds");

    let mut worst_score = 0.0f32;
    let mut worst_distance = 0.0f32;
    for (image, _, name) in fixtures() {
        let boxed = letterbox_top_left(&image, 640);
        let blob = ScrfdDetector::blob(&boxed.image).expect("the blob builds");
        let mine = net.forward(&blob).expect("the hand port runs");
        let theirs = mold_inference::identity::scrfd_net::reference_forward(&loaded.model, &blob)
            .expect("candle-onnx runs");

        for level in 0..3 {
            worst_score = worst_score.max(max_abs(
                &mine.scores[level],
                &theirs.scores[level],
                "scores",
            ));
            worst_distance = worst_distance.max(max_abs(
                &mine.bboxes[level],
                &theirs.bboxes[level],
                "bboxes",
            ));
            worst_distance = worst_distance.max(max_abs(
                &mine.keypoints[level],
                &theirs.keypoints[level],
                "keypoints",
            ));
        }
        println!("{name}: worst score {worst_score:e}, worst distance {worst_distance:e}");
    }
    assert!(
        worst_score < SCORE_TOLERANCE,
        "score drift {worst_score:e} exceeds {SCORE_TOLERANCE:e}"
    );
    assert!(
        worst_distance < DISTANCE_TOLERANCE,
        "distance drift {worst_distance:e} exceeds {DISTANCE_TOLERANCE:e}"
    );
}

#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_arcface_hand_port_matches_candle_onnx_on_the_raw_embedding() {
    let dir = assets().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    let loaded = load_onnx_model(&dir.join("glintr100.onnx"), None).expect("the model loads");
    let net = IResNet100::new(&loaded.model, &Device::Cpu).expect("the hand port builds");

    let mut worst_delta = 0.0f32;
    let mut worst_cosine = 1.0f32;
    for (image, landmarks, name) in fixtures() {
        let crop = norm_crop(&image, &landmarks).expect("a warpable face");
        let blob = ArcFaceRecognizer::blob(&crop).expect("the blob builds");
        let mine = net.forward(&blob).expect("the hand port runs");
        let theirs = mold_inference::identity::arcface_net::reference_forward(&loaded.model, &blob)
            .expect("candle-onnx runs");
        assert_eq!(mine.len(), 512);

        worst_delta = worst_delta.max(max_abs(&mine, &theirs, "embedding"));
        let cosine =
            ArcFaceEmbedding { raw: mine }.cosine_similarity(&ArcFaceEmbedding { raw: theirs });
        worst_cosine = worst_cosine.min(cosine);
        println!("{name}: worst |delta| {worst_delta:e}, cosine {cosine:.8}");
    }
    assert!(
        worst_delta < EMBEDDING_TOLERANCE,
        "embedding drift {worst_delta:e} exceeds {EMBEDDING_TOLERANCE:e}"
    );
    assert!(
        worst_cosine >= SELF_COSINE_FLOOR,
        "cosine {worst_cosine:.8} against the evaluator this port replaced"
    );
}

/// The whole point of the port: a second evaluation must not re-read the
/// weights. Nothing observable proves that directly, so this pins the property
/// that motivated it — repeated forwards are deterministic AND the graph is no
/// longer needed, so the `ModelProto` can be dropped before the first call.
#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_port_outlives_the_graph_it_was_built_from() {
    let dir = assets().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    let crop = RgbImage::from_pixel(112, 112, image::Rgb([90, 110, 130]));
    let blob = ArcFaceRecognizer::blob(&crop).expect("the blob builds");
    let net = {
        let loaded = load_onnx_model(&dir.join("glintr100.onnx"), None).expect("the model loads");
        IResNet100::new(&loaded.model, &Device::Cpu).expect("the hand port builds")
        // `loaded` — the whole 261 MB `ModelProto` — is dropped here.
    };
    let first = net.forward(&blob).expect("the first forward");
    let second = net.forward(&blob).expect("the second forward");
    assert_eq!(first, second, "resident weights must be deterministic");
    assert!(net.device().is_cpu(), "milestone 1 stays CPU-resident");
}

/// Placement is now a real property of the instance rather than the
/// evaluator's hardcoded `Device::Cpu`, and milestone 1 must still answer CPU
/// through the ordinary constructors.
#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_shipped_constructors_stay_cpu_resident() {
    let dir = assets().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    let detector = load_onnx_model(&dir.join("scrfd_10g_bnkps.onnx"), None).expect("the model");
    let recognizer = load_onnx_model(&dir.join("glintr100.onnx"), None).expect("the model");
    assert!(ScrfdDetector::new(detector.model)
        .expect("the detector builds")
        .device()
        .is_cpu());
    assert!(ArcFaceRecognizer::new(recognizer.model)
        .expect("the recognizer builds")
        .device()
        .is_cpu());
}

// ---------------------------------------------------------------------------
// Weight-gated: the performance property, pinned relatively.
// ---------------------------------------------------------------------------

/// Warmups and measured runs, the `pulid_face_probe` gate protocol.
const GATE_WARMUPS: usize = 5;
const GATE_RUNS: usize = 20;

/// The port must never be SLOWER than the evaluator it replaced.
///
/// This is deliberately a **relative** pin rather than a millisecond ceiling.
/// An absolute number is a property of the machine and the machine's load —
/// `pulid-face-extraction.md` records a p95 that tripled under load average 83
/// — so a committed millisecond threshold either flakes on a busy box or is set
/// so loose it catches nothing. Running both evaluators alternately in one loop
/// makes them share whatever contention exists, and the ratio survives it.
///
/// It is not a hypothetical guard. The first version of `arcface_net` computed
/// the fully-connected layer as `X @ W^T` and so materialized a transpose of
/// the 51 MB weight on **every forward**, which made the "faster" port about a
/// tenth SLOWER than `simple_eval` on the recognizer; this ratio is what caught
/// it.
///
/// The margin is generous — the port only has to not regress — because the
/// measured win is small: see `docs/architecture/pulid-perf.md` §4, which
/// records why re-materialization turned out not to be the cost centre. It is
/// still tight enough to catch the transpose bug, which cost ~16% of the
/// recognizer.
const NO_SLOWER_THAN: f64 = 0.95;

/// The best and median samples.
///
/// The ratio below is taken over the BEST sample of each side, not the p95.
/// Both evaluators are deterministic, so their fastest run is the one least
/// contaminated by another process — and this box routinely sits at load
/// average 15 with peer builds running, where a p95 is whatever stall happened
/// to land in the 19th sample. The median is reported alongside so a reader can
/// see the spread rather than trusting one number.
fn best_and_median(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[0], samples[samples.len() / 2])
}

fn milliseconds(f: impl FnOnce()) -> f64 {
    let started = std::time::Instant::now();
    f();
    started.elapsed().as_secs_f64() * 1000.0
}

#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_resident_port_is_never_slower_than_the_evaluator_it_replaced() {
    let dir = assets().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    let detector = load_onnx_model(&dir.join("scrfd_10g_bnkps.onnx"), None).expect("the model");
    let recognizer = load_onnx_model(&dir.join("glintr100.onnx"), None).expect("the model");
    let scrfd = ScrfdNet::new(&detector.model, &Device::Cpu).expect("the hand port builds");
    let arcface = IResNet100::new(&recognizer.model, &Device::Cpu).expect("the hand port builds");

    let (image, landmarks, _) = fixtures().remove(0);
    let boxed = letterbox_top_left(&image, 640);
    let detect_blob = ScrfdDetector::blob(&boxed.image).expect("the blob builds");
    let crop = norm_crop(&image, &landmarks).expect("a warpable face");
    let embed_blob = ArcFaceRecognizer::blob(&crop).expect("the blob builds");

    let mut port = Vec::new();
    let mut oracle = Vec::new();
    for i in 0..(GATE_WARMUPS + GATE_RUNS) {
        // Alternated inside one iteration so both see the same contention.
        let p = milliseconds(|| {
            scrfd.forward(&detect_blob).expect("the port runs");
            arcface.forward(&embed_blob).expect("the port runs");
        });
        let o = milliseconds(|| {
            mold_inference::identity::scrfd_net::reference_forward(&detector.model, &detect_blob)
                .expect("candle-onnx runs");
            mold_inference::identity::arcface_net::reference_forward(
                &recognizer.model,
                &embed_blob,
            )
            .expect("candle-onnx runs");
        });
        if i >= GATE_WARMUPS {
            port.push(p);
            oracle.push(o);
        }
    }
    let (port_best, port_median) = best_and_median(port);
    let (oracle_best, oracle_median) = best_and_median(oracle);
    let ratio = oracle_best / port_best;
    println!(
        "face stack: port {port_best:.1}/{port_median:.1} ms, candle-onnx \
         {oracle_best:.1}/{oracle_median:.1} ms (best/median), {ratio:.2}x"
    );
    assert!(
        ratio >= NO_SLOWER_THAN,
        "the resident port is {ratio:.2}x the evaluator it replaced \
         ({port_best:.1} ms vs {oracle_best:.1} ms best of {GATE_RUNS}); a resident forward must \
         not cost more than one that re-materializes 278 MB of initializers per call"
    );
}

/// The device seam, exercised end to end rather than only at the blob.
///
/// `place_input`'s unit test proves the input follows the weights; this proves
/// the rest of the network does too — every convolution, the folded batch
/// norms, the PReLUs, and the `Gemm` — because a seam that compiles but cannot
/// complete a forward is not a seam. It is what makes
/// `docs/architecture/pulid-perf.md` §5 implementable rather than aspirational.
///
/// Metal reassociates reductions differently from the CPU, so the comparison is
/// a tolerance rather than bit-identity — deliberately looser than the
/// CPU-vs-`candle-onnx` budgets above, and still far tighter than the 0.99
/// cosine the upstream gate asks for.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requires the antelopev2 ONNX models via MOLD_TEST_PULID_ASSETS"]
fn the_device_seam_completes_a_forward_off_the_cpu() {
    let dir = assets().expect("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    // `metal_device`, never `Device::new_metal`. Two reasons, and the second
    // is the one that bites: mold opens each Metal GPU exactly once per
    // process (a second device for the same GPU is the split-identity bug),
    // and `device::tests::production_code_never_constructs_a_metal_device_directly`
    // scans every `.rs` under `crates/` treating everything before the first
    // `#[cfg(test)]` as production — which, in an integration-test file that
    // has no such marker, is the whole file.
    let Ok(metal) = mold_inference::device::metal_device(0) else {
        eprintln!("skipping: no Metal device on this machine");
        return;
    };
    let recognizer = load_onnx_model(&dir.join("glintr100.onnx"), None).expect("the model loads");
    let on_cpu = IResNet100::new(&recognizer.model, &Device::Cpu).expect("the CPU port builds");
    let on_metal = IResNet100::new(&recognizer.model, &metal).expect("the Metal port builds");
    assert!(on_metal.device().same_device(&metal));

    let (image, landmarks, _) = fixtures().remove(0);
    let crop = norm_crop(&image, &landmarks).expect("a warpable face");
    // The blob is built on the CPU, as production builds it: this call is the
    // regression the finding was about.
    let blob = ArcFaceRecognizer::blob(&crop).expect("the blob builds");

    let host = on_cpu.forward(&blob).expect("the CPU forward");
    let device = on_metal.forward(&blob).expect("the Metal forward");
    let delta = max_abs(&host, &device, "embedding");
    let cosine =
        ArcFaceEmbedding { raw: host }.cosine_similarity(&ArcFaceEmbedding { raw: device });
    println!("metal vs cpu: worst |delta| {delta:e}, cosine {cosine:.8}");
    assert!(cosine >= 0.9999, "metal cosine {cosine:.8} against the CPU");
}
