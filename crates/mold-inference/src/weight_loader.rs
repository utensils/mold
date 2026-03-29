//! Utility for loading safetensors model weights with byte-level progress reporting.
//!
//! Replaces the opaque `VarBuilder::from_mmaped_safetensors()` with a per-tensor
//! loading loop that emits `WeightLoad` progress events. The returned `VarBuilder`
//! is backed by an in-memory `HashMap` (via `VarBuilder::from_tensors`).

use anyhow::Result;
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

use crate::progress::ProgressReporter;

/// Load safetensors files with per-tensor progress reporting.
///
/// `component` is the human-readable label (e.g. "FLUX transformer", "VAE").
/// Progress events track bytes loaded vs total file size on disk.
pub fn load_safetensors_with_progress<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();

    // Total bytes = sum of file sizes on disk (accurate, includes header/alignment)
    let bytes_total: u64 = path_refs
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    // Open mmap (cheap — sets up page table entries, no I/O yet)
    let st = unsafe { MmapedSafetensors::multi(&path_refs)? };

    // Enumerate all tensors and compute per-tensor byte sizes from shape/dtype
    let tensor_list: Vec<(String, usize)> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let elements: usize = view.shape().iter().product();
            let byte_size = elements * dtype.size_in_bytes();
            (name, byte_size)
        })
        .collect();

    let mut bytes_loaded: u64 = 0;
    let mut last_reported: u64 = 0;
    // Throttle: emit at most ~100 events (every 1% or 50MB, whichever is larger)
    let report_interval = (bytes_total / 100).max(50_000_000);

    progress.weight_load(component, 0, bytes_total);

    let mut tensor_map: HashMap<String, Tensor> = HashMap::with_capacity(tensor_list.len());
    for (name, byte_size) in &tensor_list {
        // Load tensor from mmap (triggers page fault → disk read → device transfer)
        let tensor = st.load(name, device)?;
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };
        tensor_map.insert(name.clone(), tensor);

        bytes_loaded += *byte_size as u64;
        if bytes_loaded - last_reported >= report_interval || bytes_loaded >= bytes_total {
            // Clamp to file-size total to avoid overshoot from dtype size differences
            progress.weight_load(component, bytes_loaded.min(bytes_total), bytes_total);
            last_reported = bytes_loaded;
        }
    }

    Ok(VarBuilder::from_tensors(tensor_map, dtype, device))
}
