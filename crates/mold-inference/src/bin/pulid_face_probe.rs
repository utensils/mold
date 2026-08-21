//! Step-0 probe for #1222: can `candle-onnx` run the two InsightFace graphs,
//! and how fast?
//!
//! Two subcommands, both taking a directory holding `scrfd_10g_bnkps.onnx` and
//! `glintr100.onnx`:
//!
//! ```text
//! pulid_face_probe inventory <dir> [--write <path>]
//! pulid_face_probe bench     <dir> [--warmups 5] [--runs 20]
//! ```
//!
//! `inventory` prints (and optionally writes) the op/attribute inventory the
//! hermetic gate test consumes. `bench` measures WARM repeated `simple_eval`,
//! which is the number the decision gate is stated in — `simple_eval`
//! re-materializes every initializer and retains every intermediate on each
//! call (`candle-onnx/src/eval.rs:191-257`), so there is no cold/warm split to
//! amortize and the second call costs what the thousandth does.
//!
//! Development-only: `dev-bins` + `pulid`, never in a release recipe.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use image::RgbImage;
use mold_inference::identity::onnx_graph::load_onnx_model;
use mold_inference::identity::onnx_inventory::{
    graph_inventory, ignored_attributes, pinned_input_makes_pooling_exact,
    unsupported_by_candle_onnx, GraphInventory,
};
use mold_inference::identity::{arcface::ArcFaceRecognizer, scrfd::ScrfdDetector};

/// #1222's Step-0 latency gate, in milliseconds of warm p95 per image.
const LATENCY_BUDGET_MS: f64 = 2000.0;

const DETECTOR: &str = "scrfd_10g_bnkps.onnx";
const RECOGNIZER: &str = "glintr100.onnx";

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

fn report(label: &str, mut samples: Vec<f64>) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "{label:<10} n={:<3} min={:8.1} ms  mean={:8.1} ms  p50={:8.1} ms  p95={:8.1} ms  max={:8.1} ms",
        samples.len(),
        samples.first().copied().unwrap_or_default(),
        mean,
        percentile(&samples, 50.0),
        percentile(&samples, 95.0),
        samples.last().copied().unwrap_or_default(),
    );
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

fn run_inventory(dir: &Path, write: Option<&Path>) -> Result<()> {
    let mut all = serde_json::Map::new();
    let mut failed = false;
    for file in [DETECTOR, RECOGNIZER] {
        let (inventory, sha256, bytes) = inventory_for(dir, file)?;
        println!("=== {file}  sha256={sha256}  {bytes} bytes");
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
        let unsupported = unsupported_by_candle_onnx(&inventory);
        if unsupported.is_empty() {
            println!("    OP GATE: pass — every op and attribute is implemented");
        } else {
            failed = true;
            println!("    OP GATE: FAIL");
            for item in &unsupported {
                println!("      - {item}");
            }
        }
        for ignored in ignored_attributes(&inventory) {
            println!(
                "    ignored: {}.{}={} — {}",
                ignored.op_type, ignored.attribute, ignored.value, ignored.harmless_because
            );
        }
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

fn run_bench(dir: &Path, warmups: usize, runs: usize) -> Result<()> {
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
    let rss_start = peak_rss_bytes();

    let load = Instant::now();
    let detector_model = load_onnx_model(&dir.join(DETECTOR), None)?;
    let recognizer_model = load_onnx_model(&dir.join(RECOGNIZER), None)?;
    let detector = ScrfdDetector::new(detector_model.model)?;
    let recognizer = ArcFaceRecognizer::new(recognizer_model.model)?;
    println!(
        "decode+construct: {:.1} ms   peak RSS after load: {:.1} MiB",
        load.elapsed().as_secs_f64() * 1000.0,
        peak_rss_bytes() as f64 / 1024.0 / 1024.0
    );

    let source = synthetic_face(1024, 768);
    let crop = synthetic_face(112, 112);

    let mut detect_ms = Vec::new();
    let mut embed_ms = Vec::new();
    for i in 0..(warmups + runs) {
        let t = Instant::now();
        let _ = detector.detect(&source).context("SCRFD evaluation")?;
        let d = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let _ = recognizer.embed_crop(&crop).context("ArcFace evaluation")?;
        let e = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmups {
            detect_ms.push(d);
            embed_ms.push(e);
        }
    }
    let total: Vec<f64> = detect_ms
        .iter()
        .zip(embed_ms.iter())
        .map(|(d, e)| d + e)
        .collect();
    report("scrfd", detect_ms);
    report("arcface", embed_ms);
    report("per-image", total.clone());

    let rss_end = peak_rss_bytes();
    println!(
        "peak RSS: {:.1} MiB (delta over process start: {:.1} MiB)",
        rss_end as f64 / 1024.0 / 1024.0,
        (rss_end.saturating_sub(rss_start)) as f64 / 1024.0 / 1024.0
    );
    let mut sorted = total;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = percentile(&sorted, 95.0);
    let passed = p95 <= LATENCY_BUDGET_MS;
    println!(
        "\n{}: p95 per image = {p95:.1} ms, budget = {LATENCY_BUDGET_MS:.0} ms -> {}",
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
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: pulid_face_probe <inventory|bench> <assets-dir> [--write PATH] \
                 [--warmups N] [--runs N]";
    if args.len() < 3 {
        bail!("{usage}");
    }
    let dir = PathBuf::from(&args[2]);
    let mut write: Option<PathBuf> = None;
    let mut warmups = 5usize;
    let mut runs = 20usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--write" => {
                write = Some(PathBuf::from(
                    args.get(i + 1).context("--write needs a path")?,
                ));
                i += 2;
            }
            "--warmups" => {
                warmups = args
                    .get(i + 1)
                    .context("--warmups needs a count")?
                    .parse()?;
                i += 2;
            }
            "--runs" => {
                runs = args.get(i + 1).context("--runs needs a count")?.parse()?;
                i += 2;
            }
            other => bail!("unknown argument `{other}`\n{usage}"),
        }
    }
    match args[1].as_str() {
        "inventory" => run_inventory(&dir, write.as_deref()),
        "bench" => run_bench(&dir, warmups, runs),
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
