//! Read-only artifact-resolution benchmark. With no arguments uses sparse
//! temporary fixtures; pass real model paths for a storage-specific baseline.
use mold_core::download::{artifact_hash_work, compute_sha256, installed_artifact_identity};
use std::{path::PathBuf, time::Instant};

fn main() -> anyhow::Result<()> {
    let temporary = tempfile::tempdir()?;
    let mut paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let fixture = paths.is_empty();
    if fixture {
        for (name, bytes) in [("a.bin", 256 * 1024 * 1024), ("b.bin", 128 * 1024 * 1024)] {
            let path = temporary.path().join(name);
            std::fs::File::create(&path)?.set_len(bytes)?;
            paths.push(path);
        }
    }
    let sizes = paths
        .iter()
        .map(|path| path.metadata().map(|m| m.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let started = Instant::now();
    for path in &paths {
        compute_sha256(path)?;
    }
    let baseline_ms = started.elapsed().as_secs_f64() * 1000.;
    let before = artifact_hash_work();
    let started = Instant::now();
    let identities = paths
        .iter()
        .map(|path| installed_artifact_identity(path))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let cold_us = started.elapsed().as_secs_f64() * 1_000_000.;
    let started = Instant::now();
    for _ in 0..100 {
        for (path, expected) in paths.iter().zip(&identities) {
            anyhow::ensure!(
                installed_artifact_identity(path)? == *expected,
                "identity changed"
            );
        }
    }
    let repeat_us = started.elapsed().as_secs_f64() * 1_000_000.;
    let after = artifact_hash_work();
    anyhow::ensure!(before == after, "runtime resolution hashed artifact bytes");
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "fixture": if fixture { "sparse temporary files" } else { "supplied model files" },
            "artifact_bytes": sizes, "full_hash_baseline_ms": baseline_ms,
            "cold_resolution_us": cold_us, "alternating_resolutions": 100 * paths.len(),
            "alternating_resolution_total_us": repeat_us,
            "runtime_hash_attempts": after.0 - before.0, "runtime_hash_bytes": after.1 - before.1,
            "scope": "artifact resolution only; excludes model loading and GPU inference"
        }))?
    );
    Ok(())
}
