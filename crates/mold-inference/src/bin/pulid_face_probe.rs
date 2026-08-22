//! Step-0 probe for #1222, extended for #1227: what does one identity
//! extraction cost, stage by stage?
//!
//! ```text
//! pulid_face_probe inventory <dir> [--write <path>]
//! pulid_face_probe bench     <dir> [--warmups 5] [--runs 20]
//!                                  [--compare] [--regress-against halcyon|plato]
//!                                  [--regress-against-full halcyon|plato]
//!                                  [--device cpu|metal]
//!                                  [--full --adapter <path> --eva <path>]
//! pulid_face_probe gate      <graph.onnx>
//! ```
//!
//! `<dir>` holds `scrfd_10g_bnkps.onnx`, `glintr100.onnx`, and
//! `parsing_bisenet.pth`.
//!
//! `gate` runs the same op/attribute gate over ONE arbitrary graph, which is
//! how #1225 qualified (and disqualified) a candidate ONNX export of
//! facexlib's BiSeNet face parser before any of it was ported. It takes a
//! pathname rather than a bundle because the graph under test is not yet a
//! manifest artifact and has no pin to check it against.
//!
//! `inventory` prints (and optionally writes) the op/attribute inventory the
//! hermetic gate test consumes.
//!
//! `bench` measures WARM repeated evaluation, which is the protocol both gates
//! are stated in: neither the `candle-onnx` evaluator this probe originally
//! measured nor the resident hand port that replaced it (#1227) has a cold/warm
//! split to amortize once weights are in memory, so the second call costs what
//! the thousandth does and five warmups are enough.
//!
//! Three things #1227 added, each for a reason `docs/architecture/pulid-perf.md`
//! records:
//!
//! * `--compare` re-measures the retained `candle-onnx` oracle beside the
//!   resident port, on the same machine in the same run, so the speedup is a
//!   measurement rather than a comparison against a number recorded on a
//!   differently loaded box (§4).
//! * `--regress-against` turns §1's "p95 at least 25% faster" acceptance
//!   criterion into a mechanical check against the baselines committed to this
//!   repository, instead of a percentage a reviewer recomputes by hand (§4).
//! * `--device` places the WHOLE extraction — detector, recognizer, parser,
//!   tower, IDFormer — on that device (#1227 phase 2). `cpu` is the default and
//!   is what every committed baseline was measured on.
//! * `--regress-against-full` is `--regress-against` stated over the WHOLE
//!   extraction rather than the face stack, which is what §5 moved the 25%
//!   criterion to once §4 showed the face stack was 13% of the problem.
//! * `--full` additionally measures the EVA02-CLIP tower and the IDFormer,
//!   which §0 found had **no number anywhere in the repository** even though
//!   they run on every conditioned request. It needs the rest of the bundle:
//!   `--adapter` (`pulid_flux_v0.9.1.safetensors`) and `--eva`
//!   (`EVA02_CLIP_L_336_psz14_s6B.pt`).
//!
//! Development-only: `dev-bins` + `pulid`, never in a release recipe.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use image::RgbImage;
use mold_core::pulid_assets::PulidPaths;
use mold_inference::identity::arcface_net::IResNet100;
use mold_inference::identity::extraction::{compose_identity_tokens_observed, ComposeStage};
use mold_inference::identity::onnx_graph::load_onnx_model;
use mold_inference::identity::onnx_inventory::{
    graph_inventory, ignored_attributes, pinned_input_makes_pooling_exact,
    unsupported_by_candle_onnx, GraphInventory,
};
use mold_inference::identity::scrfd_net::ScrfdNet;
use mold_inference::identity::{arcface::ArcFaceRecognizer, scrfd::ScrfdDetector};

/// #1222's Step-0 latency gate, in milliseconds of warm p95 per image.
const LATENCY_BUDGET_MS: f64 = 2000.0;

/// #1222's recorded SCRFD + ArcFace warm p95, `candle-onnx`, per host.
///
/// These are the numbers `docs/architecture/pulid-face-extraction.md` published
/// and `pulid-perf.md` §1 states its acceptance criterion against. They are
/// baselines, not budgets: a build that beats them by less than
/// [`REGRESSION_MARGIN`] fails `--regress-against`.
const BASELINE_HALCYON_P95_MS: f64 = 415.7;
const BASELINE_PLATO_P95_MS: f64 = 1574.5;

/// #1227 phase 1's recorded WHOLE-extraction warm p95 on halcyon.
///
/// `pulid-perf.md` §4's four-stage table, and the number §5 states phase 2's
/// acceptance against: 25% under it is 2,147.9 ms. It is deliberately the
/// phase-1 figure and not a re-measurement — a threshold edited to match the
/// outcome measures nothing (§4's own words about the face-stack pin).
///
/// It predates #1292's BiSeNet mask, which added a stage the phase-1 run never
/// paid. That makes the criterion HARDER, not looser, so it stands unchanged.
const BASELINE_HALCYON_FULL_P95_MS: f64 = 2863.9;

/// plato has no phase-1 whole-extraction measurement — §4 records the skip
/// rather than a number nobody took. `--regress-against-full plato` says so
/// instead of comparing against a fabricated baseline.
const BASELINE_PLATO_FULL_P95_MS: Option<f64> = None;

/// §1's acceptance criterion, "p95 at least 25% faster", as a ratio.
const REGRESSION_MARGIN: f64 = 0.75;

const DETECTOR: &str = "scrfd_10g_bnkps.onnx";
const RECOGNIZER: &str = "glintr100.onnx";
/// facexlib's face parser, as its own release ships it.
const PARSER: &str = "parsing_bisenet.pth";

/// A named measurement host from `pulid-perf.md` §4.
#[derive(Debug, Clone, Copy)]
struct Baseline {
    host: &'static str,
    p95_ms: f64,
    /// The whole-extraction baseline, where §4 recorded one.
    full_p95_ms: Option<f64>,
}

fn baseline_for(name: &str) -> Result<Baseline> {
    match name {
        "halcyon" => Ok(Baseline {
            host: "halcyon",
            p95_ms: BASELINE_HALCYON_P95_MS,
            full_p95_ms: Some(BASELINE_HALCYON_FULL_P95_MS),
        }),
        "plato" => Ok(Baseline {
            host: "plato",
            p95_ms: BASELINE_PLATO_P95_MS,
            full_p95_ms: BASELINE_PLATO_FULL_P95_MS,
        }),
        other => bail!("unknown baseline host `{other}`; pulid-perf.md names halcyon and plato"),
    }
}

/// Resolve `--device`.
///
/// Named rather than probed: a benchmark that silently fell back to the CPU
/// because Metal was unavailable would record a CPU number under a device
/// heading, which is the one mistake a device measurement cannot survive.
fn device_for(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        // Through `device::metal_device`, never `Device::new_metal`: every
        // Metal construction in mold goes through that memo, and a second
        // device for one GPU has a split identity.
        "metal" => mold_inference::device::metal_device(0)
            .context("--device metal needs a Metal device and a build with the `metal` feature"),
        "cuda" => Device::new_cuda(0)
            .context("--device cuda needs a CUDA device and a build with the `cuda` feature"),
        other => bail!("unknown device `{other}`; use cpu, metal, or cuda"),
    }
}

/// Resident-set peak, in bytes.
fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid, fully initialized `rusage`.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    let raw = usage.ru_maxrss as u64;
    // macOS reports bytes; Linux reports kilobytes.
    if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    }
}

/// Nearest-rank percentile of an ascending slice.
///
/// Panics on an empty slice, deliberately. The old `0.0` fallback meant an
/// empty sample set reported a p95 of zero and PASSED the gate — a decision
/// procedure answering "well within budget" from no measurement at all.
/// [`validate_bench_args`] makes the slice non-empty before this is reachable,
/// and this panic is the backstop if that ever stops being true.
fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    assert!(
        !sorted_ms.is_empty(),
        "percentile of an empty sample set must never decide the gate"
    );
    // Nearest-rank, which for 20 samples at p95 is the 19th.
    let rank = ((p / 100.0) * sorted_ms.len() as f64).ceil() as usize;
    sorted_ms[rank.clamp(1, sorted_ms.len()) - 1]
}

/// Sample counts #1222's gate is stated over.
const GATE_WARMUPS: usize = 5;
const GATE_RUNS: usize = 20;
/// Anything past this is a typo, not an intention — at ~0.4 s per run it is
/// already more than a day of benchmarking.
const MAX_RUNS: usize = 10_000;

/// Reject sample counts that cannot produce a verdict, and say so when the
/// counts are legal but not the gate's.
fn validate_bench_args(warmups: usize, runs: usize) -> Result<Option<String>> {
    if runs == 0 {
        bail!("--runs must be at least 1; a gate cannot be decided from no samples");
    }
    if runs > MAX_RUNS {
        bail!("--runs {runs} exceeds the {MAX_RUNS} sanity cap");
    }
    if warmups > MAX_RUNS {
        bail!("--warmups {warmups} exceeds the {MAX_RUNS} sanity cap");
    }
    if runs < GATE_RUNS || warmups < GATE_WARMUPS {
        return Ok(Some(format!(
            "NOTE: {warmups} warmups / {runs} runs is not the #1222 protocol \
             ({GATE_WARMUPS} / {GATE_RUNS}). At this sample count the p95 is \
             dominated by any single scheduler stall, so the verdict below is \
             advisory and must not be recorded as the gate."
        )));
    }
    Ok(None)
}

fn synthetic_face(width: u32, height: u32) -> RgbImage {
    // Deterministic noise: the benchmark measures the graph, not the content,
    // and a constant image would be an unrepresentative cache case.
    let mut image = RgbImage::new(width, height);
    let mut state: u32 = 0x9E37_79B9;
    for px in image.pixels_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        px.0 = [(state >> 24) as u8, (state >> 16) as u8, (state >> 8) as u8];
    }
    image
}

fn inventory_for(dir: &Path, file: &str) -> Result<(GraphInventory, String, usize)> {
    let loaded = load_onnx_model(&dir.join(file), None)?;
    let inventory = graph_inventory(&loaded.model)?;
    Ok((inventory, loaded.sha256, loaded.bytes))
}

/// Print one graph's inventory and its verdict. Returns true when the gate
/// FAILED, so a caller running several graphs can report every one of them
/// before it exits non-zero.
fn report_inventory(label: &str, inventory: &GraphInventory, sha256: &str, bytes: usize) -> bool {
    println!("=== {label}  sha256={sha256}  {bytes} bytes");
    println!("    opset: {:?}", inventory.opset);
    for sig in &inventory.signatures {
        let attrs = sig
            .attributes
            .iter()
            .map(|a| format!("{}={}", a.name, a.value))
            .collect::<Vec<_>>()
            .join(" ");
        println!("    x{:<4} {:<20} {attrs}", sig.count, sig.op_type);
    }
    let unsupported = unsupported_by_candle_onnx(inventory);
    let failed = !unsupported.is_empty();
    if failed {
        println!("    OP GATE: FAIL");
        for item in &unsupported {
            println!("      - {item}");
        }
    } else {
        println!("    OP GATE: pass — every op and attribute is implemented");
    }
    for ignored in ignored_attributes(inventory) {
        println!(
            "    ignored: {}.{}={} — {}",
            ignored.op_type, ignored.attribute, ignored.value, ignored.harmless_because
        );
    }
    failed
}

/// Run the op gate over one arbitrary graph.
///
/// Unpinned by construction: a candidate export is not a manifest artifact, so
/// there is nothing to authenticate it against. `load_onnx_model`'s
/// `UNPINNED_MAX_BYTES` bound still applies, which is why this stays a probe
/// subcommand and never a runtime path.
fn run_gate(graph: &Path) -> Result<()> {
    let loaded = load_onnx_model(graph, None)?;
    let inventory = graph_inventory(&loaded.model)?;
    let label = graph
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| graph.display().to_string());
    if report_inventory(&label, &inventory, &loaded.sha256, loaded.bytes) {
        bail!("the op gate failed; candle-onnx cannot run {label} as-is");
    }
    Ok(())
}

fn run_inventory(dir: &Path, write: Option<&Path>) -> Result<()> {
    let mut all = serde_json::Map::new();
    let mut failed = false;
    for file in [DETECTOR, RECOGNIZER] {
        let (inventory, sha256, bytes) = inventory_for(dir, file)?;
        failed |= report_inventory(file, &inventory, &sha256, bytes);
        all.insert(
            file.to_string(),
            serde_json::json!({ "sha256": sha256, "bytes": bytes, "inventory": inventory }),
        );
    }
    println!(
        "\n640 is an exact pooling input: {}",
        pinned_input_makes_pooling_exact(640)
    );
    if let Some(path) = write {
        let json = serde_json::to_string_pretty(&all)?;
        std::fs::write(path, format!("{json}\n"))?;
        println!("wrote {}", path.display());
    }
    if failed {
        bail!("the op gate failed; candle-onnx cannot run these graphs as-is");
    }
    Ok(())
}

/// What one `bench` invocation was asked to measure.
struct BenchOptions {
    warmups: usize,
    runs: usize,
    /// Also measure the retained `candle-onnx` oracle and report the speedup.
    compare: bool,
    /// Apply §1's "at least 25% faster" criterion against a named host.
    regress_against: Option<Baseline>,
    /// Apply §5's whole-extraction criterion against a named host.
    regress_against_full: Option<Baseline>,
    /// Also measure the EVA tower and the IDFormer.
    full: Option<PulidPaths>,
    /// Where the whole extraction runs.
    device: Device,
}

/// Sorted samples plus the two percentiles every row reports.
fn report(label: &str, mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p95 = percentile(&samples, 95.0);
    println!(
        "{label:<12} n={:<3} min={:8.1} ms  mean={:8.1} ms  p50={:8.1} ms  p95={:8.1} ms  max={:8.1} ms",
        samples.len(),
        samples.first().copied().unwrap_or_default(),
        mean,
        percentile(&samples, 50.0),
        p95,
        samples.last().copied().unwrap_or_default(),
    );
    p95
}

/// Time one closure, in milliseconds.
fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<f64> {
    let t = Instant::now();
    f()?;
    Ok(t.elapsed().as_secs_f64() * 1000.0)
}

fn run_bench(dir: &Path, options: BenchOptions) -> Result<()> {
    let BenchOptions {
        warmups,
        runs,
        compare,
        regress_against,
        regress_against_full,
        full,
        device,
    } = options;
    let advisory = validate_bench_args(warmups, runs)?;
    if let Some(note) = &advisory {
        println!("{note}\n");
    }
    println!(
        "host: {} {} | warmups={warmups} runs={runs} | build={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(debug_assertions) {
            "debug (NOT a valid measurement)"
        } else {
            "release"
        }
    );
    println!("device: {}", device_label(&device));
    println!("load average at start: {}", load_average());
    let rss_start = peak_rss_bytes();

    let load = Instant::now();
    let detector_model = load_onnx_model(&dir.join(DETECTOR), None)?;
    let recognizer_model = load_onnx_model(&dir.join(RECOGNIZER), None)?;
    // The oracle keeps its own copy of each graph, because the shipped
    // constructors consume theirs — which is the point of #1227.
    let oracle = compare.then(|| (detector_model.model.clone(), recognizer_model.model.clone()));
    let detector = ScrfdDetector::new_on_device(detector_model.model, &device)?;
    let recognizer = ArcFaceRecognizer::new_on_device(recognizer_model.model, &device)?;
    println!(
        "decode+construct: {:.1} ms   peak RSS after load: {:.1} MiB",
        load.elapsed().as_secs_f64() * 1000.0,
        peak_rss_bytes() as f64 / 1024.0 / 1024.0
    );

    let source = synthetic_face(1024, 768);
    let crop = synthetic_face(112, 112);
    let eva_crop = synthetic_face(512, 512);
    let arcface_vector = vec![0.05f32; 512];

    let mut detect_ms = Vec::new();
    let mut embed_ms = Vec::new();
    let mut parse_ms = Vec::new();
    let mut parser_build_ms = Vec::new();
    let mut eva_auth_ms = Vec::new();
    let mut eva_construct_ms = Vec::new();
    let mut eva_build_ms = Vec::new();
    let mut eva_forward_ms = Vec::new();
    let mut idformer_build_ms = Vec::new();
    let mut idformer_forward_ms = Vec::new();
    for i in 0..(warmups + runs) {
        let d = timed(|| detector.detect(&source).context("SCRFD evaluation"))?;
        let e = timed(|| recognizer.embed_crop(&crop).context("ArcFace evaluation"))?;
        let mut stages = [0.0f64; 8];
        if let Some(paths) = &full {
            compose_identity_tokens_observed(
                paths,
                &arcface_vector,
                &eva_crop,
                &device,
                &mut |stage, elapsed| {
                    let ms = elapsed.as_secs_f64() * 1000.0;
                    let slot = match stage {
                        ComposeStage::Parse => 0,
                        ComposeStage::EvaBuild => 1,
                        ComposeStage::EvaForward => 2,
                        ComposeStage::IdFormerBuild => 3,
                        ComposeStage::IdFormerForward => 4,
                        ComposeStage::ParserBuild => 5,
                        ComposeStage::EvaAuthenticate => 6,
                        ComposeStage::EvaConstruct => 7,
                    };
                    stages[slot] = ms;
                },
            )
            .context("EVA tower + IDFormer evaluation")?;
        }
        if i >= warmups {
            detect_ms.push(d);
            embed_ms.push(e);
            if full.is_some() {
                parse_ms.push(stages[0]);
                eva_build_ms.push(stages[1]);
                eva_forward_ms.push(stages[2]);
                idformer_build_ms.push(stages[3]);
                idformer_forward_ms.push(stages[4]);
                parser_build_ms.push(stages[5]);
                eva_auth_ms.push(stages[6]);
                eva_construct_ms.push(stages[7]);
            }
        }
    }

    // The per-image total is the sum of the stages that actually ran, sample by
    // sample — never the sum of independently sorted percentiles, which is a
    // different and larger number.
    let mut total: Vec<f64> = detect_ms
        .iter()
        .zip(embed_ms.iter())
        .map(|(d, e)| d + e)
        .collect();
    report("scrfd", detect_ms);
    report("arcface", embed_ms);
    let face_p95 = report("face-stack", total.clone());
    let mut full_p95 = None;
    if full.is_some() {
        // Build and forward are reported apart because they answer different
        // questions: the forward is arithmetic, the build is what the
        // drop-and-reload rule re-pays on every request and is therefore what a
        // residency change (`pulid-perf.md` §3) would buy back.
        report("bisenet", parse_ms.clone());
        report("eva-build", eva_build_ms.clone());
        // The three rows below DECOMPOSE `eva-build` — they are measured
        // inside its window, so they are never added to the total. §5 asked
        // for exactly this split: the authenticate half is a 609 MB re-hash
        // and the construct half is a dtype widening, and 1,268 ms of
        // undifferentiated "build" said nothing about which to attack.
        report("  parser", parser_build_ms.clone());
        report("  eva-auth", eva_auth_ms.clone());
        report("  eva-ctor", eva_construct_ms.clone());
        report("eva-forward", eva_forward_ms.clone());
        report("idformer-build", idformer_build_ms.clone());
        report("idformer-fwd", idformer_forward_ms.clone());
        for (i, sample) in total.iter_mut().enumerate() {
            *sample += parse_ms[i]
                + eva_build_ms[i]
                + eva_forward_ms[i]
                + idformer_build_ms[i]
                + idformer_forward_ms[i];
        }
        full_p95 = Some(report("per-image", total));
    }

    if let Some((detector_graph, recognizer_graph)) = oracle {
        // The oracle is `candle-onnx`, which places every initializer on
        // `Device::Cpu` and refuses anything else — that is §1's whole
        // argument for the hand port. `--compare` is therefore a CPU-only
        // comparison, and running the port arm on an accelerator beside it
        // would report a device speedup as an evaluator speedup.
        debug_assert!(device.is_cpu());
        println!("\n--- paired against the candle-onnx oracle (the path #1227 replaced) ---");
        // Paired and INTERLEAVED, on byte-identical blobs, comparing network to
        // network rather than production entry point to raw graph. Two
        // sequential blocks on a shared machine measure whatever else the box
        // was doing during each block; alternating within one iteration makes
        // both evaluators see the same contention. The blob and letterbox are
        // excluded from BOTH sides because they are common to both.
        let boxed = mold_inference::identity::warp::letterbox_top_left(&source, 640);
        let detect_blob = ScrfdDetector::blob(&boxed.image)?;
        let embed_blob = ArcFaceRecognizer::blob(&crop)?;
        let scrfd_net = ScrfdNet::new(&detector_graph, &Device::Cpu)?;
        let arcface_net = IResNet100::new(&recognizer_graph, &Device::Cpu)?;
        let mut port_detect = Vec::new();
        let mut port_embed = Vec::new();
        let mut oracle_detect = Vec::new();
        let mut oracle_embed = Vec::new();
        for i in 0..(warmups + runs) {
            let pd = timed(|| scrfd_net.forward(&detect_blob))?;
            let od = timed(|| {
                mold_inference::identity::scrfd_net::reference_forward(
                    &detector_graph,
                    &detect_blob,
                )
            })?;
            let pe = timed(|| arcface_net.forward(&embed_blob))?;
            let oe = timed(|| {
                mold_inference::identity::arcface_net::reference_forward(
                    &recognizer_graph,
                    &embed_blob,
                )
            })?;
            if i >= warmups {
                port_detect.push(pd);
                port_embed.push(pe);
                oracle_detect.push(od);
                oracle_embed.push(oe);
            }
        }
        let pair = |port: Vec<f64>, oracle: Vec<f64>| -> (f64, f64) {
            let sum = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            (sum(&port), sum(&oracle))
        };
        let port_total: Vec<f64> = port_detect
            .iter()
            .zip(port_embed.iter())
            .map(|(d, e)| d + e)
            .collect();
        let oracle_total: Vec<f64> = oracle_detect
            .iter()
            .zip(oracle_embed.iter())
            .map(|(d, e)| d + e)
            .collect();
        report("port-scrfd", port_detect.clone());
        report("onnx-scrfd", oracle_detect.clone());
        report("port-arcface", port_embed.clone());
        report("onnx-arcface", oracle_embed.clone());
        let port_p95 = report("port-graph", port_total.clone());
        let oracle_p95 = report("onnx-graph", oracle_total.clone());
        for (label, (port, orc)) in [
            ("scrfd", pair(port_detect, oracle_detect)),
            ("arcface", pair(port_embed, oracle_embed)),
            ("graph", pair(port_total, oracle_total)),
        ] {
            println!(
                "{label:<12} mean speedup {:.2}x  ({orc:.1} ms -> {port:.1} ms)",
                orc / port.max(f64::MIN_POSITIVE)
            );
        }
        println!(
            "graph p95 speedup {:.2}x  ({oracle_p95:.1} ms -> {port_p95:.1} ms)",
            oracle_p95 / port_p95.max(f64::MIN_POSITIVE)
        );
    }

    let rss_end = peak_rss_bytes();
    println!(
        "peak RSS: {:.1} MiB (delta over process start: {:.1} MiB)",
        rss_end as f64 / 1024.0 / 1024.0,
        (rss_end.saturating_sub(rss_start)) as f64 / 1024.0 / 1024.0
    );
    println!("load average at end: {}", load_average());

    if let Some(per_image) = full_p95 {
        // Reported, never gated. #1222's budget was stated over SCRFD + ArcFace
        // and nothing else, so applying it to a measurement that now covers
        // four stages would fail a gate this build was never subject to. What
        // the number IS good for is recorded in `pulid-perf.md` §4: it is the
        // first per-request figure for the whole extraction, and it is what any
        // future budget should be stated over.
        println!(
            "\nWHOLE EXTRACTION: p95 per image = {per_image:.1} ms across four stages. No budget \
             has ever been stated over this; #1222's is the face stack alone."
        );
    }

    let p95 = face_p95;
    let passed = p95 <= LATENCY_BUDGET_MS;
    println!(
        "\n{}: face-stack p95 per image = {p95:.1} ms, budget = {LATENCY_BUDGET_MS:.0} ms -> {}",
        if advisory.is_some() {
            "ADVISORY (not the gate protocol)"
        } else {
            "GATE"
        },
        if passed { "PASS" } else { "FAIL" }
    );
    // Exit non-zero on a failed gate, exactly as `inventory` does for the op
    // gate. A decision procedure that always exits 0 is a decision nothing can
    // act on: CI, a bisect, or a `&&` chain would all read a blown budget as
    // success.
    if !passed {
        bail!("latency gate failed: p95 {p95:.1} ms exceeds the {LATENCY_BUDGET_MS:.0} ms budget");
    }

    if let Some(baseline) = regress_against {
        // The baselines cover SCRFD + ArcFace only, so the comparison is made
        // against the face stack even when `--full` measured more. Comparing a
        // four-stage total to a two-stage baseline would report a regression
        // that is really a wider measurement.
        let ceiling = baseline.p95_ms * REGRESSION_MARGIN;
        let improvement = (1.0 - face_p95 / baseline.p95_ms) * 100.0;
        println!(
            "\nREGRESSION vs {} ({:.1} ms baseline): face-stack p95 {face_p95:.1} ms is \
             {improvement:.1}% faster, ceiling {ceiling:.1} ms -> {}",
            baseline.host,
            baseline.p95_ms,
            if face_p95 <= ceiling { "PASS" } else { "FAIL" }
        );
        if advisory.is_some() {
            bail!("--regress-against needs the gate protocol, not an advisory sample count");
        }
        if face_p95 > ceiling {
            bail!(
                "regression gate failed: face-stack p95 {face_p95:.1} ms exceeds {ceiling:.1} ms \
                 (75% of the {} baseline)",
                baseline.host
            );
        }
    }

    if let Some(baseline) = regress_against_full {
        let Some(per_image) = full_p95 else {
            bail!(
                "--regress-against-full needs --full: there is no whole-extraction p95 without it"
            );
        };
        let Some(recorded) = baseline.full_p95_ms else {
            bail!(
                "no whole-extraction baseline is recorded for {}; pulid-perf.md §4 records the \
                 skip rather than a number nobody took",
                baseline.host
            );
        };
        let ceiling = recorded * REGRESSION_MARGIN;
        let improvement = (1.0 - per_image / recorded) * 100.0;
        println!(
            "\nREGRESSION (whole extraction) vs {} ({recorded:.1} ms baseline): p95 \
             {per_image:.1} ms is {improvement:.1}% faster, ceiling {ceiling:.1} ms -> {}",
            baseline.host,
            if per_image <= ceiling { "PASS" } else { "FAIL" }
        );
        if advisory.is_some() {
            bail!("--regress-against-full needs the gate protocol, not an advisory sample count");
        }
        if per_image > ceiling {
            bail!(
                "regression gate failed: whole-extraction p95 {per_image:.1} ms exceeds \
                 {ceiling:.1} ms (75% of the {} baseline)",
                baseline.host
            );
        }
    }
    Ok(())
}

/// How a device is named in a recorded measurement.
fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda:0",
        Device::Metal(_) => "metal:0",
    }
}

/// The 1-minute load average, so a recorded number carries the conditions it
/// was measured under.
///
/// `pulid-face-extraction.md`'s own cautionary example is a p95 that tripled
/// under load average 83; a benchmark that does not report this invites the
/// same mistake again.
fn load_average() -> String {
    let mut loads = [0f64; 3];
    // SAFETY: `getloadavg` writes at most `nelem` doubles into the buffer.
    let n = unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
    if n < 1 {
        return "unavailable".to_string();
    }
    format!("{:.2}", loads[0])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: pulid_face_probe <inventory|bench> <assets-dir> [--write PATH] \
                 [--warmups N] [--runs N] [--compare] [--regress-against halcyon|plato] \
                 [--regress-against-full halcyon|plato] [--device cpu|metal|cuda] \
                 [--full --adapter PATH --eva PATH]\n       \
                 pulid_face_probe gate <graph.onnx>";
    if args.len() < 3 {
        bail!("{usage}");
    }
    let dir = PathBuf::from(&args[2]);
    let mut write: Option<PathBuf> = None;
    let mut warmups = GATE_WARMUPS;
    let mut runs = GATE_RUNS;
    let mut compare = false;
    let mut regress_against: Option<Baseline> = None;
    let mut regress_against_full: Option<Baseline> = None;
    let mut device_name = "cpu".to_string();
    let mut full = false;
    let mut adapter: Option<PathBuf> = None;
    let mut eva: Option<PathBuf> = None;
    let mut i = 3;
    let value = |i: usize, flag: &str| -> Result<String> {
        args.get(i + 1)
            .cloned()
            .with_context(|| format!("{flag} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--write" => {
                write = Some(PathBuf::from(value(i, "--write")?));
                i += 2;
            }
            "--warmups" => {
                warmups = value(i, "--warmups")?.parse()?;
                i += 2;
            }
            "--runs" => {
                runs = value(i, "--runs")?.parse()?;
                i += 2;
            }
            "--compare" => {
                compare = true;
                i += 1;
            }
            "--regress-against" => {
                regress_against = Some(baseline_for(&value(i, "--regress-against")?)?);
                i += 2;
            }
            "--regress-against-full" => {
                regress_against_full = Some(baseline_for(&value(i, "--regress-against-full")?)?);
                i += 2;
            }
            "--device" => {
                device_name = value(i, "--device")?;
                i += 2;
            }
            "--full" => {
                full = true;
                i += 1;
            }
            "--adapter" => {
                adapter = Some(PathBuf::from(value(i, "--adapter")?));
                i += 2;
            }
            "--eva" => {
                eva = Some(PathBuf::from(value(i, "--eva")?));
                i += 2;
            }
            other => bail!("unknown argument `{other}`\n{usage}"),
        }
    }
    match args[1].as_str() {
        "inventory" => run_inventory(&dir, write.as_deref()),
        "bench" => {
            let full = if full {
                Some(PulidPaths {
                    adapter: adapter.context(
                        "--full needs --adapter <pulid_flux_v0.9.1.safetensors>: the IDFormer \
                         lives in the adapter checkpoint",
                    )?,
                    vision_encoder_source: eva.context(
                        "--full needs --eva <EVA02_CLIP_L_336_psz14_s6B.pt>: the vision tower is \
                         derived from it on first use",
                    )?,
                    face_detector: dir.join(DETECTOR),
                    face_recognizer: dir.join(RECOGNIZER),
                    face_parser_source: dir.join(PARSER),
                })
            } else {
                if adapter.is_some() || eva.is_some() {
                    bail!("--adapter and --eva only mean something with --full");
                }
                None
            };
            let device = device_for(&device_name)?;
            if compare && !device.is_cpu() {
                bail!(
                    "--compare is a CPU-only comparison: the candle-onnx oracle it measures \
                     against places every initializer on Device::Cpu and refuses anything else"
                );
            }
            run_bench(
                &dir,
                BenchOptions {
                    warmups,
                    runs,
                    compare,
                    regress_against,
                    regress_against_full,
                    full,
                    device,
                },
            )
        }
        "gate" => run_gate(&dir),
        other => bail!("unknown subcommand `{other}`\n{usage}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_runs_is_refused_rather_than_measured() {
        let err = validate_bench_args(GATE_WARMUPS, 0).unwrap_err();
        assert!(
            format!("{err}").contains("--runs must be at least 1"),
            "{err}"
        );
    }

    #[test]
    fn absurd_counts_are_refused() {
        assert!(validate_bench_args(GATE_WARMUPS, MAX_RUNS + 1).is_err());
        assert!(validate_bench_args(MAX_RUNS + 1, GATE_RUNS).is_err());
        assert!(validate_bench_args(MAX_RUNS, MAX_RUNS).unwrap().is_none());
    }

    #[test]
    fn the_gate_protocol_carries_no_advisory() {
        assert!(validate_bench_args(GATE_WARMUPS, GATE_RUNS)
            .unwrap()
            .is_none());
        assert!(validate_bench_args(GATE_WARMUPS + 3, GATE_RUNS + 80)
            .unwrap()
            .is_none());
    }

    /// A legal but too-small sample set still runs — it is useful while
    /// iterating — but must never be mistaken for the recorded gate.
    #[test]
    fn a_short_run_is_advisory_and_says_so() {
        let note = validate_bench_args(0, 2).unwrap().expect("an advisory");
        assert!(note.contains("not the #1222 protocol"), "{note}");
        assert!(note.contains("advisory"), "{note}");
        assert!(validate_bench_args(GATE_WARMUPS, GATE_RUNS - 1)
            .unwrap()
            .is_some());
        assert!(validate_bench_args(GATE_WARMUPS - 1, GATE_RUNS)
            .unwrap()
            .is_some());
    }

    #[test]
    fn the_named_baselines_are_the_ones_the_doc_records() {
        assert_eq!(baseline_for("halcyon").unwrap().p95_ms, 415.7);
        assert_eq!(baseline_for("plato").unwrap().p95_ms, 1574.5);
        let err = baseline_for("hal9000").unwrap_err();
        assert!(format!("{err}").contains("halcyon and plato"), "{err}");
    }

    /// §5's acceptance number, stated once. 2,863.9 x 0.75 = 2,147.9 ms is the
    /// figure the doc publishes, and a build that quietly moved either input
    /// would move the acceptance criterion with it.
    #[test]
    fn the_whole_extraction_ceiling_is_the_number_the_doc_publishes() {
        let baseline = baseline_for("halcyon").unwrap().full_p95_ms.unwrap();
        assert_eq!(baseline, 2863.9);
        let ceiling = baseline * REGRESSION_MARGIN;
        assert!((ceiling - 2147.925).abs() < 1e-9, "{ceiling}");
    }

    /// plato has no phase-1 whole-extraction measurement. Asking for one must
    /// say so, not compare against a fabricated number.
    #[test]
    fn a_host_without_a_full_baseline_reports_the_gap() {
        assert!(baseline_for("plato").unwrap().full_p95_ms.is_none());
    }

    #[test]
    fn devices_are_named_rather_than_probed() {
        assert!(device_for("cpu").unwrap().is_cpu());
        let err = device_for("gpu").unwrap_err();
        assert!(format!("{err}").contains("cpu, metal, or cuda"), "{err}");
        assert_eq!(device_label(&Device::Cpu), "cpu");
    }

    /// "At least 25% faster" has to mean one thing. A build landing exactly on
    /// the ceiling passes; a hair over it fails.
    #[test]
    fn the_regression_ceiling_is_three_quarters_of_the_baseline() {
        let ceiling = BASELINE_HALCYON_P95_MS * REGRESSION_MARGIN;
        assert!((ceiling - 311.775).abs() < 1e-9, "{ceiling}");
        assert!(ceiling <= BASELINE_HALCYON_P95_MS * REGRESSION_MARGIN);
        assert!(ceiling + 1e-6 > BASELINE_HALCYON_P95_MS * REGRESSION_MARGIN);
    }

    #[test]
    fn nearest_rank_percentiles_pick_the_documented_sample() {
        let samples: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        // 20 samples at p95 is the 19th.
        assert_eq!(percentile(&samples, 95.0), 19.0);
        assert_eq!(percentile(&samples, 50.0), 10.0);
        assert_eq!(percentile(&samples, 100.0), 20.0);
        assert_eq!(percentile(&[7.0], 95.0), 7.0);
    }

    #[test]
    #[should_panic(expected = "empty sample set")]
    fn an_empty_sample_set_can_never_report_a_percentile() {
        percentile(&[], 95.0);
    }
}
