//! Utility for loading safetensors model weights via lazy memory-mapped I/O.
//!
//! Wraps candle's `VarBuilder::from_mmaped_safetensors()` with progress events.
//! Only the safetensors header is parsed upfront — tensor data loads on demand
//! via OS page faults during model construction.

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::progress::ProgressReporter;

/// Load safetensors files via lazy mmap with progress events.
///
/// `component` is the human-readable label (e.g. "FLUX transformer", "VAE").
/// Emits start/complete `WeightLoad` events; actual tensor I/O is deferred
/// to model construction via OS page faults.
pub fn load_safetensors_with_progress<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();

    let bytes_total: u64 = path_refs
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    progress.weight_load(component, 0, bytes_total);

    // Lazy mmap — only parses safetensors header, no tensor I/O.
    // Tensors load on-demand via OS page faults during model construction.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_refs, dtype, device)? };

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}
