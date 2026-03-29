use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::progress::ProgressReporter;

/// A parsed LoRA adapter: pairs of (A, B) weight matrices keyed by layer name.
pub(crate) struct LoraAdapter {
    /// Map from diffusers layer name (without lora_A/lora_B suffix) to (A, B) tensors.
    pub layers: HashMap<String, LoraLayer>,
    pub rank: usize,
}

pub(crate) struct LoraLayer {
    pub a: Tensor,
    pub b: Tensor,
    /// Per-layer alpha (if present in the safetensors file).
    pub alpha: Option<f64>,
}

impl LoraAdapter {
    /// Load a LoRA safetensors file. Tensors are loaded on CPU.
    pub fn load(path: &Path) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
        let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut b_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut alpha_values: HashMap<String, f64> = HashMap::new();
        let mut rank = 0usize;

        for (name, tensor) in &tensors {
            if let Some(layer) = name.strip_suffix(".lora_A.weight") {
                rank = rank.max(tensor.dim(0)?);
                a_tensors.insert(layer.to_string(), tensor.clone());
            } else if let Some(layer) = name.strip_suffix(".lora_B.weight") {
                b_tensors.insert(layer.to_string(), tensor.clone());
            } else if let Some(layer) = name.strip_suffix(".alpha") {
                if let Ok(val) = tensor.to_scalar::<f32>() {
                    alpha_values.insert(layer.to_string(), val as f64);
                }
            }
        }

        let mut layers = HashMap::new();
        for (layer_name, a) in a_tensors {
            if let Some(b) = b_tensors.remove(&layer_name) {
                let alpha = alpha_values.get(&layer_name).copied();
                layers.insert(layer_name, LoraLayer { a, b, alpha });
            }
        }

        if layers.is_empty() {
            bail!("no LoRA A/B pairs found in {}", path.display());
        }

        Ok(Self { layers, rank })
    }
}

/// Describes how a diffusers-format LoRA key maps to a candle model tensor.
enum LoraTarget {
    /// Direct 1:1 mapping: LoRA delta applies to the entire candle tensor.
    Direct { candle_key: String },
    /// Fused mapping: LoRA delta applies to a row slice of the candle tensor.
    FusedSlice {
        candle_key: String,
        /// Which component within the fused tensor (0, 1, 2, ...).
        component: usize,
        /// Total number of equally-sized components in the fused tensor.
        num_components: usize,
    },
}

/// Map a diffusers-format LoRA key to a candle model target.
///
/// Returns None for unrecognized keys (logged as warning, skipped).
fn map_lora_key(diffusers_key: &str) -> Option<LoraTarget> {
    // Strip the "transformer." prefix that LoRA files use
    let key = diffusers_key
        .strip_prefix("transformer.")
        .unwrap_or(diffusers_key);

    // --- Double blocks (transformer_blocks.{i}) ---
    if let Some(rest) = key.strip_prefix("transformer_blocks.") {
        let (idx_str, layer) = rest.split_once('.')?;
        let _idx: usize = idx_str.parse().ok()?;
        let block = format!("double_blocks.{idx_str}");

        return match layer {
            // Image attention QKV (fused into img_attn.qkv): Q=0, K=1, V=2
            "attn.to_q" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.img_attn.qkv.weight"),
                component: 0,
                num_components: 3,
            }),
            "attn.to_k" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.img_attn.qkv.weight"),
                component: 1,
                num_components: 3,
            }),
            "attn.to_v" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.img_attn.qkv.weight"),
                component: 2,
                num_components: 3,
            }),
            // Text attention QKV (fused into txt_attn.qkv): Q=0, K=1, V=2
            "attn.add_q_proj" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.txt_attn.qkv.weight"),
                component: 0,
                num_components: 3,
            }),
            "attn.add_k_proj" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.txt_attn.qkv.weight"),
                component: 1,
                num_components: 3,
            }),
            "attn.add_v_proj" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.txt_attn.qkv.weight"),
                component: 2,
                num_components: 3,
            }),
            // Output projections
            "attn.to_out.0" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_attn.proj.weight"),
            }),
            "attn.to_add_out" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_attn.proj.weight"),
            }),
            "ff.net.0.proj" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_mlp.0.weight"),
            }),
            "ff.net.2" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_mlp.2.weight"),
            }),
            "ff_context.net.0.proj" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_mlp.0.weight"),
            }),
            "ff_context.net.2" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_mlp.2.weight"),
            }),
            "norm1.linear" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.img_mod.lin.weight"),
            }),
            "norm1_context.linear" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.txt_mod.lin.weight"),
            }),
            _ => None,
        };
    }

    // --- Single blocks (single_transformer_blocks.{i}) ---
    if let Some(rest) = key.strip_prefix("single_transformer_blocks.") {
        let (idx_str, layer) = rest.split_once('.')?;
        let _idx: usize = idx_str.parse().ok()?;
        let block = format!("single_blocks.{idx_str}");

        // single_blocks.linear1 fuses: [Q, K, V, MLP_gate, MLP_up]
        // Q/K/V each have hidden_size rows, MLP has mlp_size rows.
        // We use component indices and derive sizes from the actual tensor.
        return match layer {
            "attn.to_q" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 0,
                num_components: 0, // sentinel: use special single-block logic
            }),
            "attn.to_k" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 1,
                num_components: 0,
            }),
            "attn.to_v" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 2,
                num_components: 0,
            }),
            "proj_mlp" => Some(LoraTarget::FusedSlice {
                candle_key: format!("{block}.linear1.weight"),
                component: 3, // MLP starts after Q,K,V
                num_components: 0,
            }),
            "proj_out" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.linear2.weight"),
            }),
            "norm.linear" => Some(LoraTarget::Direct {
                candle_key: format!("{block}.modulation.lin.weight"),
            }),
            _ => None,
        };
    }

    None
}

/// Compute the row offset and size for a fused slice, handling both
/// equal-split (QKV with num_components=3) and single-block linear1
/// (Q,K,V each h_sz, then MLP is the remainder).
fn fused_slice_range(
    base_rows: usize,
    lora_out_dim: usize,
    component: usize,
    num_components: usize,
) -> (usize, usize) {
    if num_components > 0 {
        // Equal split (e.g. QKV fused: each is base_rows / 3)
        let component_size = base_rows / num_components;
        (component * component_size, component_size)
    } else {
        // Single-block linear1: [Q, K, V, MLP]
        // For Q/K/V (components 0-2): lora_out_dim = qkv_dim (e.g. 3072)
        // For MLP (component 3): lora_out_dim = mlp_dim (e.g. 12288)
        // Total: 3*qkv_dim + mlp_dim = base_rows
        if component < 3 {
            // Q/K/V: each has lora_out_dim rows
            (component * lora_out_dim, lora_out_dim)
        } else {
            // MLP: starts after 3*qkv_dim, size = lora_out_dim (= mlp_dim)
            // Derive qkv_dim from: base_rows = 3*qkv_dim + mlp_dim
            let qkv_dim = (base_rows - lora_out_dim) / 3;
            (3 * qkv_dim, lora_out_dim)
        }
    }
}

/// Apply LoRA deltas to base model tensors in-place.
/// Currently unused (superseded by `stream_merge_lora` for FLUX), but retained
/// for future SD1.5/SDXL LoRA support where the UNet loading path differs.
#[allow(dead_code)]
///
/// For direct mappings: `W' = W + scale * (B @ A)`
/// For fused slices: compute delta, then add to the corresponding row slice.
///
/// When a CUDA/Metal device is provided, the matmul (`B @ A`) runs on GPU for
/// speed — LoRA tensors are small (~50-200 MB total) and GPU matmul handles all
/// layers in seconds versus minutes on CPU.  The merged result is kept on CPU
/// (same as the base tensors) so the caller can build a VarBuilder normally.
pub(crate) fn merge_lora_into_tensors(
    base_tensors: &mut HashMap<String, Tensor>,
    adapter: &LoraAdapter,
    scale: f64,
    compute_device: &Device,
    progress: &ProgressReporter,
) -> Result<()> {
    let total = adapter.layers.len();
    let mut applied = 0usize;
    let mut skipped = 0usize;
    let on_gpu = compute_device.is_cuda() || compute_device.is_metal();

    if on_gpu {
        progress.info("Merging LoRA on GPU (fast path)");
    }

    for (i, (diffusers_key, lora_layer)) in adapter.layers.iter().enumerate() {
        if (i + 1) % 100 == 0 || i + 1 == total {
            progress.info(&format!("Merging LoRA layer {}/{total}", i + 1));
        }

        let target = match map_lora_key(diffusers_key) {
            Some(t) => t,
            None => {
                tracing::warn!(key = diffusers_key, "unrecognized LoRA key, skipping");
                skipped += 1;
                continue;
            }
        };

        // Effective scale: if alpha is present, scale = user_scale * alpha / rank
        let effective_scale = match lora_layer.alpha {
            Some(alpha) => scale * alpha / adapter.rank as f64,
            None => scale,
        };

        // Compute delta: B @ A
        // A shape: (rank, in_features), B shape: (out_features, rank)
        // delta shape: (out_features, in_features)
        // When a GPU is available, move A/B there for the matmul then bring delta back.
        let a = lora_layer.a.to_dtype(DType::F32)?.to_device(compute_device)?;
        let b = lora_layer.b.to_dtype(DType::F32)?.to_device(compute_device)?;
        let delta = b.matmul(&a)?;
        let delta = (delta * effective_scale)?.to_device(&Device::Cpu)?;

        match target {
            LoraTarget::Direct { candle_key } => {
                let base = base_tensors
                    .get(&candle_key)
                    .ok_or_else(|| anyhow::anyhow!("base model missing tensor: {candle_key}"))?;
                let original_dtype = base.dtype();
                let base_f32 = base.to_dtype(DType::F32)?;
                let merged = (base_f32 + delta)?;
                base_tensors.insert(candle_key, merged.to_dtype(original_dtype)?);
                applied += 1;
            }
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                let base = base_tensors
                    .get(&candle_key)
                    .ok_or_else(|| anyhow::anyhow!("base model missing tensor: {candle_key}"))?;
                let original_dtype = base.dtype();
                let base_f32 = base.to_dtype(DType::F32)?;
                let base_rows = base_f32.dim(0)?;
                let lora_out_dim = delta.dim(0)?;

                let (offset, size) =
                    fused_slice_range(base_rows, lora_out_dim, component, num_components);

                if offset + size > base_rows {
                    tracing::warn!(
                        key = diffusers_key,
                        offset,
                        size,
                        base_rows,
                        "fused slice out of bounds, skipping"
                    );
                    skipped += 1;
                    continue;
                }

                // Extract slice, add delta, reconstruct
                let slice = base_f32.narrow(0, offset, size)?;
                let updated_slice = (slice + delta)?;

                let mut parts: Vec<Tensor> = Vec::new();
                if offset > 0 {
                    parts.push(base_f32.narrow(0, 0, offset)?);
                }
                parts.push(updated_slice);
                let after = offset + size;
                if after < base_rows {
                    parts.push(base_f32.narrow(0, after, base_rows - after)?);
                }
                let merged = Tensor::cat(&parts, 0)?;
                base_tensors.insert(candle_key, merged.to_dtype(original_dtype)?);
                applied += 1;
            }
        }
    }

    progress.info(&format!(
        "LoRA merged: {applied} layers applied, {skipped} skipped (rank {})",
        adapter.rank
    ));
    tracing::info!(applied, skipped, rank = adapter.rank, "LoRA merge complete");
    Ok(())
}

/// Stream-load safetensors to GPU with LoRA deltas merged inline.
///
/// Streams one tensor at a time from mmap.  Non-LoRA tensors go directly to GPU.
/// LoRA-affected tensors are merged on whichever device has enough headroom:
///
/// - **GPU merge** (fast): when free VRAM > 4× the tensor's F32 size (enough for
///   BF16 base + F32 cast + delta + merged + back to BF16 without fragmentation).
/// - **CPU merge** (safe): when VRAM is tight.  The A×B matmul still runs on GPU
///   (the LoRA matrices are tiny), but the base tensor work stays on CPU and only
///   the final BF16 result is moved to GPU.
///
/// This keeps peak VRAM close to model size regardless of LoRA overhead.
pub(crate) fn stream_merge_lora<'a>(
    transformer_path: &Path,
    adapter: &LoraAdapter,
    scale: f64,
    dtype: DType,
    device: &Device,
    progress: &ProgressReporter,
) -> Result<candle_nn::VarBuilder<'a>> {
    use candle_core::safetensors::MmapedSafetensors;
    use crate::device::free_vram_bytes;

    let on_gpu = device.is_cuda() || device.is_metal();
    let device_label = if device.is_cuda() {
        "GPU"
    } else if device.is_metal() {
        "Metal"
    } else {
        "CPU"
    };

    // Open mmap (cheap, no I/O)
    let st = unsafe {
        MmapedSafetensors::multi(std::slice::from_ref(&transformer_path))?
    };

    // Detect key prefix (model.diffusion_model. or diffusion_model. or none)
    let all_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let prefix = if all_names.iter().any(|n| n == "img_in.weight") {
        ""
    } else if all_names.iter().any(|n| n == "model.diffusion_model.img_in.weight") {
        "model.diffusion_model."
    } else if all_names.iter().any(|n| n == "diffusion_model.img_in.weight") {
        "diffusion_model."
    } else {
        ""
    };

    // Pre-compute: which candle keys have LoRA deltas and what targets they map to.
    // Multiple diffusers keys can target the same candle key (e.g. Q, K, V → fused qkv).
    let mut lora_targets: HashMap<String, Vec<(&str, &LoraLayer, LoraTarget)>> = HashMap::new();
    for (diffusers_key, lora_layer) in &adapter.layers {
        if let Some(target) = map_lora_key(diffusers_key) {
            let candle_key = match &target {
                LoraTarget::Direct { candle_key } => candle_key.clone(),
                LoraTarget::FusedSlice { candle_key, .. } => candle_key.clone(),
            };
            lora_targets
                .entry(candle_key)
                .or_default()
                .push((diffusers_key.as_str(), lora_layer, target));
        } else {
            tracing::warn!(key = diffusers_key.as_str(), "unrecognized LoRA key, skipping");
        }
    }

    // Compute total bytes for progress
    let bytes_total: u64 = std::fs::metadata(transformer_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let mut bytes_loaded: u64 = 0;
    let mut last_reported: u64 = 0;
    let report_interval = (bytes_total / 100).max(50_000_000);

    let component_label = format!("FLUX transformer + LoRA ({device_label})");
    progress.weight_load(&component_label, 0, bytes_total);

    let total_tensors = all_names.len();
    let mut tensor_map: HashMap<String, Tensor> = HashMap::with_capacity(total_tensors);
    let mut applied = 0usize;
    let mut gpu_merges = 0usize;
    let mut cpu_merges = 0usize;

    for (i, raw_name) in all_names.iter().enumerate() {
        // Strip prefix for the canonical key
        let key = raw_name.strip_prefix(prefix).unwrap_or(raw_name).to_string();

        let has_lora = lora_targets.contains_key(&key);

        // Non-LoRA tensors: load directly to GPU (fast path, no intermediates).
        // LoRA tensors: load to CPU initially — merge device chosen below.
        let load_device = if has_lora { &Device::Cpu } else { device };
        let tensor = st.load(raw_name, load_device)?;
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Estimate bytes for progress (shape * dtype size)
        let elements: usize = tensor.dims().iter().product();
        let byte_size = elements * dtype.size_in_bytes();
        bytes_loaded += byte_size as u64;

        // Apply LoRA deltas if any target this tensor
        let tensor = if let Some(targets) = lora_targets.get(&key) {
            // Decide merge device: GPU if enough VRAM for F32 temporaries, else CPU.
            // F32 overhead ≈ 4× the BF16 tensor size (base_f32 + delta + merged + cast back).
            let f32_headroom = (byte_size as u64) * 4;
            let free = if on_gpu { free_vram_bytes().unwrap_or(0) } else { 0 };
            let merge_on_gpu = on_gpu && free > f32_headroom;
            let merge_device = if merge_on_gpu { device } else { &Device::Cpu };

            let mut t = if merge_on_gpu {
                // Move base tensor to GPU for merge
                tensor.to_device(device)?
            } else {
                // Keep on CPU for merge
                tensor
            };

            for (_, lora_layer, target) in targets {
                let effective_scale = match lora_layer.alpha {
                    Some(alpha) => scale * alpha / adapter.rank as f64,
                    None => scale,
                };

                // A×B matmul: always on GPU if available (LoRA matrices are tiny).
                // Delta then moved to the merge device for the base tensor addition.
                let matmul_device = if on_gpu { device } else { &Device::Cpu };
                let a = lora_layer.a.to_dtype(DType::F32)?.to_device(matmul_device)?;
                let b = lora_layer.b.to_dtype(DType::F32)?.to_device(matmul_device)?;
                let delta = b.matmul(&a)?;
                let delta = (delta * effective_scale)?.to_device(merge_device)?;

                t = match target {
                    LoraTarget::Direct { .. } => {
                        let t_f32 = t.to_dtype(DType::F32)?;
                        let merged = (t_f32 + delta)?;
                        applied += 1;
                        merged.to_dtype(dtype)?
                    }
                    LoraTarget::FusedSlice {
                        component,
                        num_components,
                        ..
                    } => {
                        let t_f32 = t.to_dtype(DType::F32)?;
                        let base_rows = t_f32.dim(0)?;
                        let lora_out_dim = delta.dim(0)?;
                        let (offset, size) =
                            fused_slice_range(base_rows, lora_out_dim, *component, *num_components);

                        if offset + size > base_rows {
                            tracing::warn!(
                                offset, size, base_rows,
                                "fused slice out of bounds, skipping"
                            );
                            t
                        } else {
                            let slice = t_f32.narrow(0, offset, size)?;
                            let updated_slice = (slice + delta)?;
                            let mut parts: Vec<Tensor> = Vec::new();
                            if offset > 0 {
                                parts.push(t_f32.narrow(0, 0, offset)?);
                            }
                            parts.push(updated_slice);
                            let after = offset + size;
                            if after < base_rows {
                                parts.push(t_f32.narrow(0, after, base_rows - after)?);
                            }
                            applied += 1;
                            Tensor::cat(&parts, 0)?.to_dtype(dtype)?
                        }
                    }
                };
            }

            if merge_on_gpu {
                gpu_merges += 1;
            } else {
                cpu_merges += 1;
            }

            // If merged on CPU, move final result to GPU
            if !merge_on_gpu && on_gpu {
                t.to_device(device)?
            } else {
                t
            }
        } else {
            tensor
        };

        tensor_map.insert(key, tensor);

        // Progress reporting
        if bytes_loaded - last_reported >= report_interval || i + 1 == total_tensors {
            progress.weight_load(
                &component_label,
                bytes_loaded.min(bytes_total),
                bytes_total,
            );
            last_reported = bytes_loaded;
        }
    }

    let skipped = adapter.layers.len() - applied;
    let merge_detail = if gpu_merges > 0 && cpu_merges > 0 {
        format!(" ({gpu_merges} on {device_label}, {cpu_merges} on CPU)")
    } else if gpu_merges > 0 {
        format!(" (all on {device_label})")
    } else if cpu_merges > 0 && on_gpu {
        format!(" (merged on CPU, loaded to {device_label})")
    } else {
        String::new()
    };
    progress.info(&format!(
        "LoRA: {applied} applied, {skipped} skipped (rank {}){merge_detail}",
        adapter.rank
    ));
    tracing::info!(applied, skipped, gpu_merges, cpu_merges, rank = adapter.rank, "LoRA stream merge complete");

    Ok(candle_nn::VarBuilder::from_tensors(tensor_map, dtype, device))
}

/// Strip a known prefix from all tensor keys in a HashMap.
///
/// FLUX safetensors may store weights under `model.diffusion_model.` or
/// `diffusion_model.` — this normalizes them to root level.
#[allow(dead_code)]
pub(crate) fn strip_tensor_prefix(tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let prefix = if tensors.contains_key("img_in.weight") {
        ""
    } else if tensors.contains_key("model.diffusion_model.img_in.weight") {
        "model.diffusion_model."
    } else if tensors.contains_key("diffusion_model.img_in.weight") {
        "diffusion_model."
    } else {
        ""
    };

    if prefix.is_empty() {
        return tensors;
    }

    tensors
        .into_iter()
        .map(|(k, v)| {
            let stripped = k.strip_prefix(prefix).unwrap_or(&k).to_string();
            (stripped, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_double_block_img_attn_qkv() {
        let key = "transformer.transformer_blocks.5.attn.to_q";
        let target = map_lora_key(key).unwrap();
        match target {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                assert_eq!(candle_key, "double_blocks.5.img_attn.qkv.weight");
                assert_eq!(component, 0);
                assert_eq!(num_components, 3);
            }
            _ => panic!("expected FusedSlice"),
        }

        let key = "transformer.transformer_blocks.5.attn.to_k";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice { component, .. } => assert_eq!(component, 1),
            _ => panic!("expected FusedSlice"),
        }

        let key = "transformer.transformer_blocks.5.attn.to_v";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice { component, .. } => assert_eq!(component, 2),
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_double_block_txt_attn_qkv() {
        let key = "transformer.transformer_blocks.0.attn.add_q_proj";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                assert_eq!(candle_key, "double_blocks.0.txt_attn.qkv.weight");
                assert_eq!(component, 0);
                assert_eq!(num_components, 3);
            }
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_double_block_direct() {
        let cases = [
            (
                "transformer.transformer_blocks.3.attn.to_out.0",
                "double_blocks.3.img_attn.proj.weight",
            ),
            (
                "transformer.transformer_blocks.3.attn.to_add_out",
                "double_blocks.3.txt_attn.proj.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff.net.0.proj",
                "double_blocks.3.img_mlp.0.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff.net.2",
                "double_blocks.3.img_mlp.2.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff_context.net.0.proj",
                "double_blocks.3.txt_mlp.0.weight",
            ),
            (
                "transformer.transformer_blocks.3.ff_context.net.2",
                "double_blocks.3.txt_mlp.2.weight",
            ),
            (
                "transformer.transformer_blocks.3.norm1.linear",
                "double_blocks.3.img_mod.lin.weight",
            ),
            (
                "transformer.transformer_blocks.3.norm1_context.linear",
                "double_blocks.3.txt_mod.lin.weight",
            ),
        ];
        for (lora_key, expected) in cases {
            match map_lora_key(lora_key).unwrap() {
                LoraTarget::Direct { candle_key } => assert_eq!(candle_key, expected),
                _ => panic!("expected Direct for {lora_key}"),
            }
        }
    }

    #[test]
    fn map_single_block_fused() {
        let key = "transformer.single_transformer_blocks.7.attn.to_q";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                ..
            } => {
                assert_eq!(candle_key, "single_blocks.7.linear1.weight");
                assert_eq!(component, 0);
            }
            _ => panic!("expected FusedSlice"),
        }

        let key = "transformer.single_transformer_blocks.7.proj_mlp";
        match map_lora_key(key).unwrap() {
            LoraTarget::FusedSlice { component, .. } => assert_eq!(component, 3),
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_single_block_direct() {
        let key = "transformer.single_transformer_blocks.7.proj_out";
        match map_lora_key(key).unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "single_blocks.7.linear2.weight")
            }
            _ => panic!("expected Direct"),
        }

        let key = "transformer.single_transformer_blocks.7.norm.linear";
        match map_lora_key(key).unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "single_blocks.7.modulation.lin.weight")
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn map_unknown_key_returns_none() {
        assert!(map_lora_key("totally.unknown.key").is_none());
        assert!(map_lora_key("transformer.transformer_blocks.0.unknown_layer").is_none());
    }

    #[test]
    fn fused_slice_range_equal_split() {
        // QKV fused: 9216 rows / 3 = 3072 each
        let (offset, size) = fused_slice_range(9216, 3072, 0, 3);
        assert_eq!((offset, size), (0, 3072));

        let (offset, size) = fused_slice_range(9216, 3072, 1, 3);
        assert_eq!((offset, size), (3072, 3072));

        let (offset, size) = fused_slice_range(9216, 3072, 2, 3);
        assert_eq!((offset, size), (6144, 3072));
    }

    #[test]
    fn fused_slice_range_single_block() {
        // linear1 fuses Q(3072), K(3072), V(3072), MLP(12288) = 21504 total
        // Q component:
        let (offset, size) = fused_slice_range(21504, 3072, 0, 0);
        assert_eq!((offset, size), (0, 3072));

        // K component:
        let (offset, size) = fused_slice_range(21504, 3072, 1, 0);
        assert_eq!((offset, size), (3072, 3072));

        // V component:
        let (offset, size) = fused_slice_range(21504, 3072, 2, 0);
        assert_eq!((offset, size), (6144, 3072));

        // MLP component (lora_out_dim = 12288):
        let (offset, size) = fused_slice_range(21504, 12288, 3, 0);
        assert_eq!((offset, size), (9216, 12288));
    }
}
