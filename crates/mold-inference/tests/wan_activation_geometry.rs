//! Geometry probe against real installed checkpoints.
//!
//! Four of the seven shipped wan tiers are GGUF, and the admission model reads
//! its token grid and per-token slope from this probe. A safetensors-only
//! reader returned `None` for every GGUF tier and silently fell back to the
//! A14B shape — right by luck for A14B, a 4x token over-count for TI2V-5B.
//!
//! Skips when the checkpoint is not installed, so it is a no-op in CI and real
//! coverage on a machine with the models.

use std::path::{Path, PathBuf};

use mold_inference::device::WanVaeGeneration;
use mold_inference::wan::pipeline::activation_geometry;

fn models_dir() -> Option<PathBuf> {
    std::env::var_os("MOLD_MODELS_DIR")
        .map(PathBuf::from)
        .filter(|dir| dir.is_dir())
}

fn first_weight_file(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = walk(dir)
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext == "gguf" || ext == "safetensors")
        })
        // The DiT is the largest file in the tier; LoRAs and configs are not.
        .filter(|path| {
            path.metadata()
                .map(|meta| meta.len() > 1_000_000_000)
                .unwrap_or(false)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

#[test]
fn gguf_and_safetensors_checkpoints_both_yield_geometry() {
    let Some(models) = models_dir() else {
        eprintln!("skipping: MOLD_MODELS_DIR is not set to a directory");
        return;
    };

    // (tier directory, expected dim, expected VAE generation)
    let expected: &[(&str, u64, WanVaeGeneration)] = &[
        ("wan21-t2v-1.3b-bf16", 1536, WanVaeGeneration::V21),
        ("wan22-t2v-a14b-q5", 5120, WanVaeGeneration::V21),
        ("wan22-t2v-a14b-q8", 5120, WanVaeGeneration::V21),
        ("wan22-ti2v-5b-fp16", 3072, WanVaeGeneration::V22),
        ("wan22-ti2v-5b-q8", 3072, WanVaeGeneration::V22),
        ("wan22-ti2v-5b-dmd", 3072, WanVaeGeneration::V22),
    ];

    let mut checked = 0;
    for (tier, dim, vae) in expected {
        let dir = models.join(tier);
        if !dir.is_dir() {
            continue;
        }
        let Some(weights) = first_weight_file(&dir) else {
            continue;
        };
        let geometry = activation_geometry(&weights).unwrap_or_else(|| {
            panic!("no geometry for {tier} at {}", weights.display());
        });
        assert_eq!(geometry.dim, *dim, "{tier} width");
        assert_eq!(geometry.vae, *vae, "{tier} VAE generation");
        assert_eq!(geometry.patch_spatial, 2, "{tier} patch width");
        assert_eq!(geometry.num_heads, dim / 128, "{tier} head count");
        checked += 1;
    }

    if checked == 0 {
        eprintln!(
            "skipping: no wan tiers installed under {}",
            models.display()
        );
    }
}
