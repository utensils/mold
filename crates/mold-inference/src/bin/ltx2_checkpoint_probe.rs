use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use safetensors::SafeTensors;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let checkpoint = args
        .next()
        .map(PathBuf::from)
        .context("usage: cargo run -p mold-ai-inference --bin ltx2_checkpoint_probe -- <checkpoint.safetensors> [pattern ...]")?;
    if !checkpoint.is_file() {
        bail!("checkpoint not found: {}", checkpoint.display());
    }

    let bytes = fs::read(&checkpoint)
        .with_context(|| format!("failed to read {}", checkpoint.display()))?;
    let (_header_size, metadata) = SafeTensors::read_metadata(&bytes)
        .with_context(|| format!("failed to read metadata from {}", checkpoint.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse {}", checkpoint.display()))?;

    let patterns = args.collect::<Vec<_>>();
    if patterns.is_empty() {
        print_summary(&metadata, &tensors);
        return Ok(());
    }

    for pattern in patterns {
        println!("pattern={pattern}");
        let mut matches = tensors
            .names()
            .into_iter()
            .filter(|name| name.contains(&pattern))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        if matches.is_empty() {
            println!("  no matches");
            continue;
        }
        for name in matches {
            let view = tensors.tensor(name)?;
            if let Some(scalar) = scalar_preview(&view) {
                println!(
                    "  {name}: dtype={:?} shape={:?} value={scalar}",
                    view.dtype(),
                    view.shape()
                );
            } else {
                println!(
                    "  {name}: dtype={:?} shape={:?}",
                    view.dtype(),
                    view.shape()
                );
            }
        }
    }

    Ok(())
}

fn print_summary(metadata: &safetensors::tensor::Metadata, tensors: &SafeTensors<'_>) {
    let mut dtype_counts = BTreeMap::new();
    let mut scale_keys = Vec::new();
    let mut interesting = Vec::new();

    println!("metadata:");
    match metadata.metadata() {
        Some(metadata) if metadata.is_empty() => println!("  <empty>"),
        Some(metadata) => {
            for (key, value) in metadata.iter() {
                println!("  {key}={value}");
            }
        }
        None => println!("  <none>"),
    }

    for name in tensors.names() {
        if let Ok(view) = tensors.tensor(name) {
            *dtype_counts
                .entry(format!("{:?}", view.dtype()))
                .or_insert(0usize) += 1;
        }
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
    }

    println!("tensor_count={}", tensors.names().len());
    println!("dtype_counts:");
    for (dtype, count) in dtype_counts {
        println!("  {dtype}: {count}");
    }
    println!("scale_tensor_count={}", scale_keys.len());
    for name in scale_keys.iter().take(32) {
        if let Ok(view) = tensors.tensor(name) {
            println!(
                "  {name}: dtype={:?} shape={:?}",
                view.dtype(),
                view.shape()
            );
        }
    }
    if scale_keys.len() > 32 {
        println!("  ... {} more scale tensors", scale_keys.len() - 32);
    }
    println!("interesting_keys:");
    for name in interesting.iter().take(64) {
        if let Ok(view) = tensors.tensor(name) {
            println!(
                "  {name}: dtype={:?} shape={:?}",
                view.dtype(),
                view.shape()
            );
        }
    }
    if interesting.len() > 64 {
        println!("  ... {} more interesting keys", interesting.len() - 64);
    }
}

fn scalar_preview(view: &safetensors::tensor::TensorView<'_>) -> Option<String> {
    if !view.shape().is_empty() {
        return None;
    }
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes: [u8; 4] = view.data().try_into().ok()?;
            Some(format!("{:.8}", f32::from_le_bytes(bytes)))
        }
        safetensors::Dtype::BF16 => {
            let bytes: [u8; 2] = view.data().try_into().ok()?;
            let bits = u16::from_le_bytes(bytes) as u32;
            Some(format!("{:.8}", f32::from_bits(bits << 16)))
        }
        _ => None,
    }
}
