use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use safetensors::tensor::TensorInfo;
use serde_json::Value;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let checkpoint = args
        .next()
        .map(PathBuf::from)
        .context("usage: cargo run -p mold-ai-inference --bin ltx2_checkpoint_probe -- <checkpoint.safetensors> [pattern ...]")?;
    if !checkpoint.is_file() {
        bail!("checkpoint not found: {}", checkpoint.display());
    }

    let (metadata, tensors) = read_safetensors_header(&checkpoint)?;

    let patterns = args.collect::<Vec<_>>();
    if patterns.is_empty() {
        print_summary(&metadata, &tensors);
        return Ok(());
    }

    for pattern in patterns {
        println!("pattern={pattern}");
        let mut matches = tensors
            .keys()
            .filter(|name| name.contains(&pattern))
            .cloned()
            .collect::<Vec<_>>();
        matches.sort_unstable();
        if matches.is_empty() {
            println!("  no matches");
            continue;
        }
        for name in matches {
            let info = tensors.get(&name).expect("match came from keys()");
            println!("  {name}: dtype={:?} shape={:?}", info.dtype, info.shape);
        }
    }

    Ok(())
}

fn read_safetensors_header(
    path: &PathBuf,
) -> Result<(BTreeMap<String, String>, BTreeMap<String, TensorInfo>)> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
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

fn print_summary(metadata: &BTreeMap<String, String>, tensors: &BTreeMap<String, TensorInfo>) {
    let mut dtype_counts = BTreeMap::new();
    let mut scale_keys = Vec::new();
    let mut interesting = Vec::new();
    let mut prefixes = BTreeSet::new();

    println!("metadata:");
    if metadata.is_empty() {
        println!("  <none>");
    } else {
        for (key, value) in metadata {
            println!("  {key}={value}");
        }
    }

    for (name, info) in tensors {
        *dtype_counts
            .entry(format!("{:?}", info.dtype))
            .or_insert(0usize) += 1;
        if name.contains("weight_scale") || name.contains("input_scale") {
            scale_keys.push(name.to_string());
        }
        if name.contains("patchify_proj")
            || name.contains("proj_in")
            || name.contains("adaln_single")
            || name.contains("time_embed")
            || name.contains(".q_norm")
            || name.contains(".norm_q")
            || name.contains(".k_norm")
            || name.contains(".norm_k")
        {
            interesting.push(name.to_string());
        }
        if let Some(prefix) = name.split('.').next() {
            prefixes.insert(prefix.to_string());
        }
    }

    println!("tensor_count={}", tensors.len());
    println!("dtype_counts:");
    for (dtype, count) in dtype_counts {
        println!("  {dtype}: {count}");
    }
    println!("scale_tensor_count={}", scale_keys.len());
    for name in scale_keys.iter().take(32) {
        if let Some(info) = tensors.get(name) {
            println!("  {name}: dtype={:?} shape={:?}", info.dtype, info.shape);
        }
    }
    if scale_keys.len() > 32 {
        println!("  ... {} more scale tensors", scale_keys.len() - 32);
    }
    println!("interesting_keys:");
    for name in interesting.iter().take(64) {
        if let Some(info) = tensors.get(name) {
            println!("  {name}: dtype={:?} shape={:?}", info.dtype, info.shape);
        }
    }
    if interesting.len() > 64 {
        println!("  ... {} more interesting keys", interesting.len() - 64);
    }
    println!("top_level_prefixes:");
    for prefix in prefixes.into_iter().take(32) {
        println!("  {prefix}");
    }
}
