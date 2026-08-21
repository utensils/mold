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

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    // Nearest-rank, which for 20 samples at p95 is the 19th.
    let rank = ((p / 100.0) * sorted_ms.len() as f64).ceil() as usize;
    sorted_ms[rank.clamp(1, sorted_ms.len()) - 1]
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
    let loaded = load_onnx_model(&dir.join(file))?;
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
    let detector_model = load_onnx_model(&dir.join(DETECTOR))?;
    let recognizer_model = load_onnx_model(&dir.join(RECOGNIZER))?;
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
    println!(
        "\nGATE: p95 per image = {p95:.1} ms, budget = 2000 ms -> {}",
        if p95 <= 2000.0 { "PASS" } else { "FAIL" }
    );
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
