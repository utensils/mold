//! Utility for loading safetensors model weights via lazy memory-mapped I/O.
//!
//! Wraps candle's `VarBuilder::from_mmaped_safetensors()` with progress events.
//! Only the safetensors header is parsed upfront — tensor data loads on demand
//! via OS page faults during model construction.

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::progress::ProgressReporter;

/// VarBuilder backend for FP8 safetensors that preserves native dtypes.
///
/// Loads tensors at their on-disk dtype: F8E4M3 weights stay F8E4M3 on GPU,
/// BF16 biases/norms stay BF16. The transformer's `QwenLinear::Fp8` handles
/// per-layer FP8→BF16 dequantization (with optional scale) during forward.
pub(crate) struct NativeFp8Backend {
    inner: candle_core::safetensors::MmapedSafetensors,
}

impl candle_nn::var_builder::SimpleBackend for NativeFp8Backend {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: candle_nn::Init,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Load at native dtype — no casting
        let tensor = self.inner.load(path, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            })?
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        path: &str,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.inner.load(path, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

/// Load FP8 safetensors preserving native dtypes on the target device.
///
/// F8E4M3 weights stay as F8E4M3 in VRAM (~19.5GB for full model).
/// BF16 biases/norms/scales stay as BF16. The transformer's `QwenLinear::Fp8`
/// handles per-layer dequantization during forward (ComfyUI "manual cast" style).
pub fn load_fp8_safetensors<'a>(
    paths: &[impl AsRef<Path>],
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

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs)? };
    let backend = NativeFp8Backend { inner: tensors };
    let vb = VarBuilder::from_backend(Box::new(backend), DType::BF16, device.clone());

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}

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
