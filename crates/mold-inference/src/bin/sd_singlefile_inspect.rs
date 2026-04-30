//! Phase-2 tensor-prefix audit for SD1.5 / SDXL single-file Civitai
//! checkpoints. Walks each input safetensors header and reports the
//! top-level prefix layout the loader will need to consume.
//!
//! Usage:
//!     cargo run -p mold-ai-inference --features dev-bins \
//!         --bin sd_singlefile_inspect -- file1.safetensors [file2 ...]
//!
//! The handoff (`tasks/catalog-expansion-phase-2-handoff.md`) flags
//! Civitai SDXL CLIP-prefix variation as a tensor-loader landmine:
//!     "Not every checkpoint uses A1111's `conditioner.embedders.*`
//!     exactly. Some use `cond_stage_model.*` (SD-style) for CLIP-L."
//! This binary's job is to surface that variation — across 2-4
//! representative files — before tasks 2.4 / 2.5 commit to a key map.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use safetensors::tensor::TensorInfo;
use serde_json::Value;

/// Family-of-interest prefixes. The audit checks each input against
/// every probe so we can spot when a checkpoint deviates from the
/// canonical A1111 layout.
const PROBES: &[(&str, &str)] = &[
    ("unet", "model.diffusion_model"),
    ("vae", "first_stage_model"),
    // SDXL (A1111 / kohya layout)
    (
        "clip_l_sdxl",
        "conditioner.embedders.0.transformer.text_model",
    ),
    ("clip_g_sdxl", "conditioner.embedders.1.model"),
    // SD1.5 (A1111 layout)
    ("clip_l_sd15", "cond_stage_model.transformer.text_model"),
    // Diffusers-style stragglers (rare in A1111 single-file but seen in
    // some Civitai uploads — flag if present so the loader knows).
    ("text_encoder_diffusers", "text_encoder"),
    ("text_encoder_2_diffusers", "text_encoder_2"),
];

fn main() -> Result<()> {
    let inputs: Vec<PathBuf> = env::args().skip(1).map(PathBuf::from).collect();
    if inputs.is_empty() {
        bail!(
            "usage: cargo run -p mold-ai-inference --features dev-bins \\\n        --bin sd_singlefile_inspect -- <file1.safetensors> [file2 ...]"
        );
    }

    let mut summaries = Vec::with_capacity(inputs.len());
    for path in &inputs {
        if !path.is_file() {
            bail!("not a file: {}", path.display());
        }
        let summary =
            inspect(path).with_context(|| format!("failed to inspect {}", path.display()))?;
        print_one(&summary);
        summaries.push(summary);
    }

    if summaries.len() > 1 {
        print_diff_table(&summaries);
    }

    Ok(())
}

#[derive(Debug)]
struct Summary {
    path: PathBuf,
    tensor_count: usize,
    metadata: BTreeMap<String, String>,
    depth1: BTreeMap<String, usize>,
    depth2: BTreeMap<String, usize>,
    /// `(probe_label, hit_count, sample_tensor_name)` — sample is the
    /// first matching tensor name encountered, useful as a sanity
    /// anchor when comparing layouts across files.
    probes: Vec<(String, usize, Option<String>)>,
    dtype_counts: BTreeMap<String, usize>,
}

fn inspect(path: &Path) -> Result<Summary> {
    let (metadata, tensors) = read_safetensors_header(path)?;

    let mut depth1: BTreeMap<String, usize> = BTreeMap::new();
    let mut depth2: BTreeMap<String, usize> = BTreeMap::new();
    let mut dtype_counts: BTreeMap<String, usize> = BTreeMap::new();

    for (name, info) in &tensors {
        *dtype_counts.entry(format!("{:?}", info.dtype)).or_insert(0) += 1;

        let parts: Vec<&str> = name.split('.').collect();
        if let Some(p1) = parts.first() {
            *depth1.entry((*p1).to_string()).or_insert(0) += 1;
        }
        if parts.len() >= 2 {
            *depth2
                .entry(format!("{}.{}", parts[0], parts[1]))
                .or_insert(0) += 1;
        }
    }

    let probes = PROBES
        .iter()
        .map(|(label, prefix)| {
            let prefix_dot = format!("{prefix}.");
            let mut hit = 0usize;
            let mut sample: Option<String> = None;
            for name in tensors.keys() {
                if name.starts_with(&prefix_dot) || name == *prefix {
                    hit += 1;
                    if sample.is_none() {
                        sample = Some(name.clone());
                    }
                }
            }
            ((*label).to_string(), hit, sample)
        })
        .collect();

    Ok(Summary {
        path: path.to_path_buf(),
        tensor_count: tensors.len(),
        metadata,
        depth1,
        depth2,
        probes,
        dtype_counts,
    })
}

fn print_one(s: &Summary) {
    println!();
    println!("=== {} ===", s.path.display());
    println!("tensor_count={}", s.tensor_count);

    if !s.metadata.is_empty() {
        println!("metadata:");
        for (k, v) in &s.metadata {
            // Truncate long values so a stray training-config blob doesn't bury the audit.
            let short: String = v.chars().take(80).collect();
            let suffix = if v.len() > 80 { " …" } else { "" };
            println!("  {k}={short}{suffix}");
        }
    }

    println!("dtype_counts:");
    for (dt, n) in &s.dtype_counts {
        println!("  {dt}: {n}");
    }

    println!("depth-1 prefixes:");
    for (p, n) in &s.depth1 {
        println!("  {p:30}  {n}");
    }

    println!("depth-2 prefixes (top by count, max 30):");
    let mut d2: Vec<_> = s.depth2.iter().collect();
    d2.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (p, n) in d2.iter().take(30) {
        println!("  {p:60}  {n}");
    }

    println!("probes:");
    for (label, hit, sample) in &s.probes {
        let status = if *hit > 0 { "HIT " } else { "miss" };
        let sample = sample.as_deref().unwrap_or("-");
        println!("  {status} {label:26} count={hit:5}  sample={sample}");
    }
}

fn print_diff_table(summaries: &[Summary]) {
    println!();
    println!("=== cross-file probe diff ===");
    let labels: Vec<String> = PROBES.iter().map(|(l, _)| (*l).to_string()).collect();
    let names: Vec<String> = summaries
        .iter()
        .map(|s| {
            s.path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect();

    let label_w = labels.iter().map(|s| s.len()).max().unwrap_or(0).max(6);
    let col_w = names.iter().map(|n| n.len().max(8)).collect::<Vec<_>>();

    print!("{:label_w$}", "probe", label_w = label_w);
    for (i, n) in names.iter().enumerate() {
        print!("  {:w$}", n, w = col_w[i]);
    }
    println!();

    for (i, label) in labels.iter().enumerate() {
        print!("{:label_w$}", label, label_w = label_w);
        for (j, s) in summaries.iter().enumerate() {
            let hit = s.probes.get(i).map(|p| p.1).unwrap_or(0);
            print!("  {:w$}", hit, w = col_w[j]);
        }
        println!();
    }

    let any_disagrees = labels.iter().enumerate().any(|(i, _)| {
        let first = summaries[0].probes.get(i).map(|p| p.1 > 0).unwrap_or(false);
        summaries
            .iter()
            .skip(1)
            .any(|s| s.probes.get(i).map(|p| p.1 > 0).unwrap_or(false) != first)
    });
    if any_disagrees {
        println!();
        println!("⚠  at least one probe is HIT in some files and miss in others — the");
        println!("   single-file loader must branch on this when picking key prefixes.");
    } else {
        println!();
        println!(
            "✓ all probes agree across files — a single key map will work for these checkpoints."
        );
    }

    let mut all_d1 = BTreeSet::new();
    for s in summaries {
        for k in s.depth1.keys() {
            all_d1.insert(k.clone());
        }
    }
    let unanimous: BTreeSet<&String> = all_d1
        .iter()
        .filter(|k| summaries.iter().all(|s| s.depth1.contains_key(*k)))
        .collect();
    let any_only: BTreeSet<&String> = all_d1.iter().filter(|k| !unanimous.contains(*k)).collect();
    if !any_only.is_empty() {
        println!();
        println!("depth-1 prefixes not present in every file:");
        for k in any_only {
            let present_in: Vec<&str> = summaries
                .iter()
                .filter(|s| s.depth1.contains_key(k))
                .map(|s| s.path.file_stem().and_then(|s| s.to_str()).unwrap_or("?"))
                .collect();
            println!("  {k:30}  present_in={:?}", present_in);
        }
    }
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(BTreeMap<String, String>, BTreeMap<String, TensorInfo>)> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .with_context(|| format!("failed to read header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)
        .with_context(|| format!("failed to read header bytes from {}", path.display()))?;
    let header: BTreeMap<String, Value> = serde_json::from_slice(&header_buf)
        .with_context(|| format!("failed to parse header JSON from {}", path.display()))?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            if let Value::Object(entries) = value {
                for (key, value) in entries {
                    metadata.insert(key, value.as_str().unwrap_or_default().to_string());
                }
            }
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value)
            .with_context(|| format!("failed to parse tensor info for {name}"))?;
        tensors.insert(name, info);
    }

    Ok((metadata, tensors))
}
