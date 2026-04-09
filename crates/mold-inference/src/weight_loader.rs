//! Utility for loading safetensors model weights via lazy memory-mapped I/O.
//!
//! Wraps candle's `VarBuilder::from_mmaped_safetensors()` with progress events.
//! Only the safetensors header is parsed upfront — tensor data loads on demand
//! via OS page faults during model construction.

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use safetensors::tensor::TensorInfo;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
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

fn total_file_bytes(paths: &[impl AsRef<Path>]) -> u64 {
    paths.iter()
        .map(|p| std::fs::metadata(p.as_ref()).map(|m| m.len()).unwrap_or(0))
        .sum()
}

fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, Value>> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    Ok(serde_json::from_slice(&header_buf)?)
}

fn filtered_safetensors_tensor_bytes(
    paths: &[impl AsRef<Path>],
    include_tensor: impl Fn(&str) -> bool,
) -> Result<u64> {
    let mut total = 0u64;
    for path in paths {
        let header = read_safetensors_header(path.as_ref())?;
        for (name, value) in header {
            if name == "__metadata__" || !include_tensor(&name) {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value)?;
            total += info.data_offsets.1.saturating_sub(info.data_offsets.0) as u64;
        }
    }
    Ok(total)
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
    let bytes_total = total_file_bytes(paths);

    progress.weight_load(component, 0, bytes_total);

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs)? };
    let backend = NativeFp8Backend { inner: tensors };
    let vb = VarBuilder::from_backend(Box::new(backend), DType::BF16, device.clone());

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}

fn load_safetensors_with_progress_total<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
    bytes_total: u64,
) -> Result<VarBuilder<'a>> {
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();

    progress.weight_load(component, 0, bytes_total);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_refs, dtype, device)? };

    progress.weight_load(component, bytes_total, bytes_total);

    Ok(vb)
}

/// Load safetensors via lazy mmap but report progress for only the tensors that
/// match `include_tensor`. This is useful when a shared shard set contains a
/// much larger model than the submodule we actually instantiate, such as the
/// Qwen2.5-VL vision tower embedded inside the shared text-encoder shards.
pub fn load_safetensors_with_filtered_progress<'a>(
    paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
    include_tensor: impl Fn(&str) -> bool,
) -> Result<VarBuilder<'a>> {
    let bytes_total =
        filtered_safetensors_tensor_bytes(paths, include_tensor).unwrap_or_else(|_| total_file_bytes(paths));
    load_safetensors_with_progress_total(paths, dtype, device, component, progress, bytes_total)
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
    let bytes_total = total_file_bytes(paths);
    load_safetensors_with_progress_total(paths, dtype, device, component, progress, bytes_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;

    fn temp_file(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-weight-loader-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    #[test]
    fn filtered_safetensors_tensor_bytes_counts_matching_tensors_only() {
        let path = temp_file("visual-bytes");
        let visual_data = vec![0u8; 16];
        let text_data = vec![0u8; 64];
        let mut tensors = HashMap::new();
        tensors.insert(
            "visual.patch_embed.proj.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &visual_data).unwrap(),
        );
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![4, 4], &text_data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let total =
            filtered_safetensors_tensor_bytes(&[path.clone()], |name| name.starts_with("visual."))
                .unwrap();
        assert_eq!(total, visual_data.len() as u64);

        let _ = std::fs::remove_file(path);
    }
}
