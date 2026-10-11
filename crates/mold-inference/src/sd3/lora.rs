//! LoRA support for Stable Diffusion 3 / 3.5 (MMDiT).
//!
//! SD3.5's joint MMDiT transformer is built on top of candle's
//! [`candle_transformers::models::mmdit::model::MMDiT`]. Every weight is
//! loaded through a `VarBuilder`, so we apply LoRA by intercepting
//! `vb.get()` with a custom [`SimpleBackend`](candle_nn::var_builder::SimpleBackend)
//! that merges the low-rank `B @ A * scale` delta into the base tensor on
//! the fly — the same pattern used in `flux/lora.rs` and `flux2/lora.rs`.
//!
//! ## Key naming
//!
//! Diffusers-format SD3 LoRAs name their layers after the
//! `SD3Transformer2DModel` Python class — that's the dominant convention
//! in the wild. Kohya / sd-scripts checkpoints flatten the same path with
//! a `lora_unet_` prefix. We accept both.
//!
//! Diffusers convention for SD3:
//!
//! - `transformer.transformer_blocks.{i}.attn.to_q|to_k|to_v` → x_block QKV fuse
//! - `transformer.transformer_blocks.{i}.attn.add_q_proj|add_k_proj|add_v_proj`
//!   → context_block QKV fuse
//! - `transformer.transformer_blocks.{i}.attn.to_out.0` → x_block.attn.proj
//! - `transformer.transformer_blocks.{i}.attn.to_add_out` → context_block.attn.proj
//! - `transformer.transformer_blocks.{i}.attn2.to_q|to_k|to_v` → x_block.attn2 QKV
//!   (MMDiT-X / SD3.5-medium only)
//! - `transformer.transformer_blocks.{i}.attn2.to_out.0` → x_block.attn2.proj
//! - `transformer.transformer_blocks.{i}.ff.net.0.proj` → x_block.mlp.fc1
//! - `transformer.transformer_blocks.{i}.ff.net.2` → x_block.mlp.fc2
//! - `transformer.transformer_blocks.{i}.ff_context.net.0.proj` → context_block.mlp.fc1
//! - `transformer.transformer_blocks.{i}.ff_context.net.2` → context_block.mlp.fc2
//! - `transformer.transformer_blocks.{i}.norm1.linear` → x_block.adaLN_modulation.1
//! - `transformer.transformer_blocks.{i}.norm1_context.linear`
//!   → context_block.adaLN_modulation.1
//!
//! The candle MMDiT fuses Q/K/V into a single `attn.qkv.weight` tensor
//! with `[Q rows | K rows | V rows]` row-stacked in that order. So
//! `attn.to_q/to_k/to_v` LoRAs land as FusedSlice targets on the same
//! candle key with component indices 0/1/2.

#[cfg(test)]
use crate::lora::{classify_lora_key, LoraDirection, LoraLayer};
#[cfg(test)]
use crate::lora::{clear_parsed_lora_cache_for_test, lock_parsed_lora_cache_tests};
pub(crate) use crate::lora::{get_or_load_adapter, LoraAdapter, LoraDeltaCache};
use crate::lora::{AdapterIdentity, LoraCacheKey};
use crate::progress::ProgressReporter;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// How a diffusers-format LoRA key maps to an SD3 MMDiT tensor.
pub(crate) enum LoraTarget {
    /// 1:1 mapping — the LoRA delta covers the entire candle tensor.
    Direct { candle_key: String },
    /// One component (Q/K/V = 0/1/2) of a fused 3-way QKV weight.
    FusedSlice {
        candle_key: String,
        component: usize,
        num_components: usize,
    },
}

/// Map a LoRA key (diffusers or Kohya format) to an SD3 MMDiT target.
///
/// Returns `None` for unrecognized keys — the caller logs a warning and
/// skips. Text-encoder LoRAs (`lora_te_*`) are intentionally not mapped
/// here; SD3 has CLIP-L + CLIP-G + T5 encoders that don't participate
/// in MMDiT denoise. Encoder LoRAs would need a separate path.
pub(crate) fn map_lora_key(diffusers_key: &str) -> Option<LoraTarget> {
    // Kohya/sd-scripts naming: flatten the diffusers path with `.`→`_`
    // and prepend `lora_unet_`. The transformer's fused `attn.qkv` tensors
    // match candle 1:1, so every Kohya key maps to a single Direct target
    // (Kohya already trains on the full fused output shape).
    if let Some(rest) = diffusers_key.strip_prefix("lora_unet_") {
        return map_kohya_unet_key(rest);
    }

    // Strip the optional `transformer.` prefix (diffusers SD3 LoRAs)
    let key = diffusers_key
        .strip_prefix("transformer.")
        .unwrap_or(diffusers_key);

    let rest = key.strip_prefix("transformer_blocks.")?;
    let (idx_str, layer) = rest.split_once('.')?;
    let _idx: usize = idx_str.parse().ok()?;
    let block = format!("joint_blocks.{idx_str}");

    match layer {
        // --- x_block (image stream) QKV — fused into x_block.attn.qkv.weight ---
        "attn.to_q" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.x_block.attn.qkv.weight"),
            component: 0,
            num_components: 3,
        }),
        "attn.to_k" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.x_block.attn.qkv.weight"),
            component: 1,
            num_components: 3,
        }),
        "attn.to_v" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.x_block.attn.qkv.weight"),
            component: 2,
            num_components: 3,
        }),
        // --- context_block (text stream) QKV — fused into context_block.attn.qkv.weight ---
        "attn.add_q_proj" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.context_block.attn.qkv.weight"),
            component: 0,
            num_components: 3,
        }),
        "attn.add_k_proj" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.context_block.attn.qkv.weight"),
            component: 1,
            num_components: 3,
        }),
        "attn.add_v_proj" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.context_block.attn.qkv.weight"),
            component: 2,
            num_components: 3,
        }),
        // --- Attention output projections ---
        "attn.to_out.0" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.x_block.attn.proj.weight"),
        }),
        "attn.to_add_out" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.context_block.attn.proj.weight"),
        }),
        // --- attn2 (MMDiT-X / SD3.5-medium only; x_block side) ---
        "attn2.to_q" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.x_block.attn2.qkv.weight"),
            component: 0,
            num_components: 3,
        }),
        "attn2.to_k" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.x_block.attn2.qkv.weight"),
            component: 1,
            num_components: 3,
        }),
        "attn2.to_v" => Some(LoraTarget::FusedSlice {
            candle_key: format!("{block}.x_block.attn2.qkv.weight"),
            component: 2,
            num_components: 3,
        }),
        "attn2.to_out.0" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.x_block.attn2.proj.weight"),
        }),
        // --- Feed-forward (x_block) — candle names them mlp.fc1 / mlp.fc2 ---
        "ff.net.0.proj" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.x_block.mlp.fc1.weight"),
        }),
        "ff.net.2" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.x_block.mlp.fc2.weight"),
        }),
        // --- Feed-forward (context_block) ---
        "ff_context.net.0.proj" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.context_block.mlp.fc1.weight"),
        }),
        "ff_context.net.2" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.context_block.mlp.fc2.weight"),
        }),
        // --- AdaLN modulation linears ---
        "norm1.linear" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.x_block.adaLN_modulation.1.weight"),
        }),
        "norm1_context.linear" => Some(LoraTarget::Direct {
            candle_key: format!("{block}.context_block.adaLN_modulation.1.weight"),
        }),
        _ => None,
    }
}

/// Map a Kohya/sd-scripts `lora_unet_*` key (prefix already stripped) to
/// the SD3 candle tensor name.
fn map_kohya_unet_key(rest: &str) -> Option<LoraTarget> {
    let after = rest.strip_prefix("joint_blocks_")?;
    let (idx_str, suffix) = after.split_once('_')?;
    idx_str.parse::<usize>().ok()?;
    let candle_key = match suffix {
        // x_block primary attention
        "x_block_attn_qkv" => format!("joint_blocks.{idx_str}.x_block.attn.qkv.weight"),
        "x_block_attn_proj" => format!("joint_blocks.{idx_str}.x_block.attn.proj.weight"),
        // x_block attn2 (MMDiT-X / medium)
        "x_block_attn2_qkv" => format!("joint_blocks.{idx_str}.x_block.attn2.qkv.weight"),
        "x_block_attn2_proj" => format!("joint_blocks.{idx_str}.x_block.attn2.proj.weight"),
        // x_block MLP
        "x_block_mlp_fc1" => format!("joint_blocks.{idx_str}.x_block.mlp.fc1.weight"),
        "x_block_mlp_fc2" => format!("joint_blocks.{idx_str}.x_block.mlp.fc2.weight"),
        // x_block adaLN — Kohya collapses `adaLN_modulation.1` to `adaLN_modulation_1`
        "x_block_adaLN_modulation_1" => {
            format!("joint_blocks.{idx_str}.x_block.adaLN_modulation.1.weight")
        }
        // context_block primary attention
        "context_block_attn_qkv" => {
            format!("joint_blocks.{idx_str}.context_block.attn.qkv.weight")
        }
        "context_block_attn_proj" => {
            format!("joint_blocks.{idx_str}.context_block.attn.proj.weight")
        }
        // context_block MLP
        "context_block_mlp_fc1" => format!("joint_blocks.{idx_str}.context_block.mlp.fc1.weight"),
        "context_block_mlp_fc2" => format!("joint_blocks.{idx_str}.context_block.mlp.fc2.weight"),
        "context_block_adaLN_modulation_1" => {
            format!("joint_blocks.{idx_str}.context_block.adaLN_modulation.1.weight")
        }
        _ => return None,
    };
    Some(LoraTarget::Direct { candle_key })
}

/// Compute `(row_offset, row_size)` for a fused-slice patch. SD3 only
/// uses the equal-split shape (Q | K | V each `base_rows / 3`), but the
/// helper takes `lora_out_dim` so future asymmetric layouts (if any)
/// stay one edit away.
pub(crate) fn fused_slice_range(
    base_rows: usize,
    _lora_out_dim: usize,
    component: usize,
    num_components: usize,
) -> (usize, usize) {
    let component_size = base_rows / num_components.max(1);
    (component * component_size, component_size)
}

/// A single LoRA patch destined for one candle tensor. Multiple patches
/// per tensor stack additively.
struct LoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: LoraTarget,

    source_layer: String,
    adapter_identity: AdapterIdentity,
}

/// One adapter (including its revision) and requested scale.
pub(crate) struct LoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
}

fn build_patches(specs: &[LoraSpec<'_>]) -> (HashMap<String, Vec<LoraPatch>>, usize) {
    let mut patches: HashMap<String, Vec<LoraPatch>> = HashMap::new();
    let mut skipped = 0usize;
    for spec in specs {
        for (diffusers_key, lora_layer) in &spec.adapter.layers {
            if let Some(target) = map_lora_key(diffusers_key) {
                let candle_key = match &target {
                    LoraTarget::Direct { candle_key } => candle_key.clone(),
                    LoraTarget::FusedSlice { candle_key, .. } => candle_key.clone(),
                };
                let layer_rank = lora_layer.a.dims()[0] as f64;
                let effective_scale = match lora_layer.alpha {
                    Some(alpha) => spec.scale * alpha / layer_rank,
                    None => spec.scale,
                };
                patches.entry(candle_key).or_default().push(LoraPatch {
                    a: lora_layer.a.clone(),
                    b: lora_layer.b.clone(),
                    effective_scale,
                    target,

                    source_layer: diffusers_key.clone(),
                    adapter_identity: spec.adapter.identity.clone(),
                });
            } else {
                tracing::warn!(
                    key = diffusers_key.as_str(),
                    "unrecognized SD3 LoRA key, skipping"
                );
                skipped += 1;
            }
        }
    }
    (patches, skipped)
}

/// `SimpleBackend` that streams base tensors from mmap and applies LoRA
/// deltas on the fly. Same approach as `flux/lora.rs` — no
/// pre-materialised `HashMap`, peak VRAM is the same as the non-LoRA
/// path.
struct LoraBackend {
    st: candle_core::safetensors::MmapedSafetensors,
    /// Prefix to prepend to the requested name (e.g. `model.diffusion_model.`).
    prefix: String,
    patches: HashMap<String, Vec<LoraPatch>>,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

impl candle_nn::var_builder::SimpleBackend for LoraBackend {
    fn get(
        &self,
        _s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.get_unchecked(name, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let raw_key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{name}", self.prefix)
        };

        let tensor = self.st.load(&raw_key, dev)?;
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        if let Some(patches) = self.patches.get(name) {
            let mut t = tensor;
            for patch in patches {
                let cache_key = LoraCacheKey {
                    tensor_name: name.to_string(),
                    source_layer: patch.source_layer.clone(),
                    adapter_identity: patch.adapter_identity.clone(),
                    scale_bits: patch.effective_scale.to_bits(),
                };

                let cached_delta = self.delta_cache.as_ref().and_then(|c| {
                    c.lock()
                        .ok()
                        .and_then(|guard| guard.get(&cache_key).cloned())
                });

                let delta = if let Some(cpu_delta) = cached_delta {
                    cpu_delta.to_device(dev)?
                } else {
                    let a = patch.a.to_dtype(DType::F32)?.to_device(dev)?;
                    let b = patch.b.to_dtype(DType::F32)?.to_device(dev)?;
                    let computed = b.matmul(&a)?;
                    let computed = (&computed * patch.effective_scale)?;
                    if let Some(ref cache) = self.delta_cache {
                        if let Ok(mut guard) = cache.lock() {
                            guard.insert(cache_key, &computed)?;
                        }
                    }
                    computed
                };

                t = match &patch.target {
                    LoraTarget::Direct { .. } => {
                        let t_f32 = t.to_dtype(DType::F32)?;
                        (&t_f32 + &delta)?.to_dtype(dtype)?
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
                                offset,
                                size,
                                base_rows,
                                "SD3 fused slice out of bounds, skipping"
                            );
                            t
                        } else {
                            let slice = t_f32.narrow(0, offset, size)?;
                            let updated_slice = (&slice + &delta)?;
                            let mut parts: Vec<Tensor> = Vec::new();
                            if offset > 0 {
                                parts.push(t_f32.narrow(0, 0, offset)?);
                            }
                            parts.push(updated_slice);
                            let after = offset + size;
                            if after < base_rows {
                                parts.push(t_f32.narrow(0, after, base_rows - after)?);
                            }
                            Tensor::cat(&parts, 0)?.to_dtype(dtype)?
                        }
                    }
                };
            }
            Ok(t)
        } else {
            Ok(tensor)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let raw_key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{name}", self.prefix)
        };
        self.st.get(&raw_key).is_ok()
    }
}

/// Inspect the safetensors to figure out which prefix the file uses for
/// MMDiT tensors. stabilityai's BF16 weights store under
/// `model.diffusion_model.*`; some forks drop to `diffusion_model.*`;
/// city96's GGUF and a few BF16 re-exports use unprefixed names.
fn detect_prefix<I, S>(all_names: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    // `x_embedder.proj.weight` is the patch embedder — unique to MMDiT,
    // always present in every variant.
    const SENTINEL: &str = "x_embedder.proj.weight";
    let mut has_root = false;
    let mut has_diffusion = false;
    let mut has_model_diffusion = false;
    for name in all_names {
        match name.as_ref() {
            n if n == SENTINEL => has_root = true,
            n if n == format!("diffusion_model.{SENTINEL}").as_str() => has_diffusion = true,
            n if n == format!("model.diffusion_model.{SENTINEL}").as_str() => {
                has_model_diffusion = true
            }
            _ => {}
        }
    }
    if has_model_diffusion {
        "model.diffusion_model.".to_string()
    } else if has_diffusion {
        "diffusion_model.".to_string()
    } else if has_root {
        String::new()
    } else {
        // Fall back to the most common convention — stabilityai BF16.
        // The model constructor will error with a clear missing-key
        // message if this is wrong.
        "model.diffusion_model.".to_string()
    }
}

/// Build a LoRA-patching VarBuilder for SD3's BF16 safetensors path.
///
/// The returned builder is positioned at the prefix root — feed it to
/// `MMDiT::new` exactly as the non-LoRA path does (`vb.pp("model.diffusion_model")`
/// adjusted as appropriate). The backend strips/re-adds the on-disk
/// prefix automatically so the model constructor's relative keys keep
/// working.
pub(crate) fn lora_var_builder<'a>(
    transformer_path: &Path,
    specs: &[LoraSpec<'_>],
    dtype: DType,
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<candle_nn::VarBuilder<'a>> {
    use candle_core::safetensors::MmapedSafetensors;

    if specs.is_empty() {
        bail!("lora_var_builder called with no LoraSpecs — caller must provide at least one");
    }

    let st = unsafe { MmapedSafetensors::multi(std::slice::from_ref(&transformer_path))? };

    let all_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let prefix = detect_prefix(&all_names);

    let (patches, skipped) = build_patches(specs);
    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "SD3 LoRA: {n} adapter(s), {total_patches} patches on {patched_keys} tensors, \
         {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let backend = LoraBackend {
        st,
        prefix,
        patches,
        delta_cache,
    };

    Ok(candle_nn::VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

/// Build a quantized VarBuilder for SD3's GGUF path with LoRA deltas
/// merged in. Mirrors `flux::lora::gguf_lora_var_builder` — selectively
/// dequantize each patched tensor to F32 on CPU, apply the delta,
/// re-quantize back to the original GGML dtype on the target device.
pub(crate) fn gguf_lora_var_builder(
    transformer_path: &Path,
    specs: &[LoraSpec<'_>],
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<mold_candle::quantized::VarBuilder> {
    use candle_core::quantized::{gguf_file, QTensor};

    if specs.is_empty() {
        bail!("gguf_lora_var_builder called with no LoraSpecs — caller must provide at least one");
    }

    let mut file = std::fs::File::open(transformer_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    let total_tensors = content.tensor_infos.len();
    let mut data: HashMap<String, Arc<QTensor>> = HashMap::with_capacity(total_tensors);

    let (patches, skipped) = build_patches(specs);
    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "SD3 LoRA (GGUF): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, \
         {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let gguf_bytes_total: u64 = std::fs::metadata(transformer_path)
        .map(|m| m.len())
        .unwrap_or(0);
    progress.weight_load("SD3 MMDiT (GGUF)", 0, gguf_bytes_total);
    for (i, tensor_name) in content.tensor_infos.keys().enumerate() {
        let qtensor = content.tensor(&mut file, tensor_name, device)?;
        data.insert(tensor_name.clone(), Arc::new(qtensor));
        let approx_bytes = gguf_bytes_total * (i as u64 + 1) / total_tensors as u64;
        progress.weight_load(
            "SD3 MMDiT (GGUF)",
            approx_bytes.min(gguf_bytes_total),
            gguf_bytes_total,
        );
    }
    drop(file);

    let on_gpu = device.is_cuda() || device.is_metal();
    let mut applied = 0usize;
    let lora_keys: Vec<String> = patches.keys().cloned().collect();
    let lora_total = lora_keys.len();
    for (i, candle_key) in lora_keys.iter().enumerate() {
        let layer_patches = &patches[candle_key];

        let tensor_key = if data.contains_key(candle_key) {
            candle_key.clone()
        } else {
            tracing::warn!(
                key = candle_key.as_str(),
                "SD3 LoRA target tensor not found in GGUF, skipping"
            );
            continue;
        };

        let orig_dtype = data[&tensor_key].dtype();
        let qtensor = data.remove(&tensor_key).unwrap();
        let mut t = qtensor.dequantize(&Device::Cpu)?;
        drop(qtensor);
        if on_gpu {
            device.synchronize()?;
        }

        for patch in layer_patches {
            let cache_key = LoraCacheKey {
                tensor_name: candle_key.clone(),
                source_layer: patch.source_layer.clone(),
                adapter_identity: patch.adapter_identity.clone(),
                scale_bits: patch.effective_scale.to_bits(),
            };

            let cached = delta_cache.as_ref().and_then(|c| {
                c.lock()
                    .ok()
                    .and_then(|guard| guard.get(&cache_key).cloned())
            });

            let delta = if let Some(cpu_delta) = cached {
                cpu_delta
            } else {
                let matmul_dev = if on_gpu { device } else { &Device::Cpu };
                let a = patch.a.to_dtype(DType::F32)?.to_device(matmul_dev)?;
                let b = patch.b.to_dtype(DType::F32)?.to_device(matmul_dev)?;
                let computed = b.matmul(&a)?;
                let computed = (&computed * patch.effective_scale)?.to_device(&Device::Cpu)?;

                if let Some(ref cache) = delta_cache {
                    if let Ok(mut guard) = cache.lock() {
                        guard.insert(cache_key, &computed)?;
                    }
                }
                computed
            };

            t = match &patch.target {
                LoraTarget::Direct { .. } => (&t + &delta)?,
                LoraTarget::FusedSlice {
                    component,
                    num_components,
                    ..
                } => {
                    let base_rows = t.dim(0)?;
                    let lora_out_dim = delta.dim(0)?;
                    let (offset, size) =
                        fused_slice_range(base_rows, lora_out_dim, *component, *num_components);

                    if offset + size > base_rows {
                        tracing::warn!(
                            offset,
                            size,
                            base_rows,
                            "SD3 GGUF fused slice out of bounds, skipping"
                        );
                        t
                    } else {
                        let slice = t.narrow(0, offset, size)?;
                        let updated_slice = (&slice + &delta)?;
                        let mut parts: Vec<Tensor> = Vec::new();
                        if offset > 0 {
                            parts.push(t.narrow(0, 0, offset)?);
                        }
                        parts.push(updated_slice);
                        let after = offset + size;
                        if after < base_rows {
                            parts.push(t.narrow(0, after, base_rows - after)?);
                        }
                        Tensor::cat(&parts, 0)?
                    }
                }
            };
            applied += 1;
        }

        let patched = mold_candle::quantized::quantize_onto(&t, orig_dtype, device)?;
        drop(t);
        data.insert(tensor_key, Arc::new(patched));

        if (i + 1) % 50 == 0 || i + 1 == lora_total {
            progress.info(&format!(
                "SD3 patching LoRA tensor {}/{}",
                i + 1,
                lora_keys.len()
            ));
        }
    }

    let total_layers: usize = specs.iter().map(|s| s.adapter.layers.len()).sum();
    progress.info(&format!(
        "SD3 LoRA: {applied} applied, {} skipped (max rank {max_rank}, {patched_keys} layers patched)",
        total_layers.saturating_sub(applied),
    ));

    if on_gpu {
        device.synchronize()?;
    }

    Ok(mold_candle::quantized::VarBuilder::from_qtensors(
        data, device,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn audit_delta_cache_invalidates_after_same_path_adapter_replacement() {
        use candle_nn::var_builder::SimpleBackend;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let path = dir.path().join("adapter.safetensors");
        let key = "joint_blocks.0.x_block.attn.qkv.weight";
        candle_core::safetensors::save(
            &HashMap::from([(
                key,
                Tensor::zeros((3, 1), DType::F32, &Device::Cpu).unwrap(),
            )]),
            &base,
        )
        .unwrap();
        let gguf_path = dir.path().join("base.gguf");
        let qt = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((3, 1), DType::F32, &Device::Cpu).unwrap(),
            candle_core::quantized::GgmlDType::F32,
        )
        .unwrap();
        candle_core::quantized::gguf_file::write(
            &mut std::fs::File::create(&gguf_path).unwrap(),
            &[],
            &[(key, &qt)],
        )
        .unwrap();
        let cache = Arc::new(Mutex::new(LoraDeltaCache::new()));
        for value in [2.0f32, 7.0] {
            let replacement = dir.path().join("replacement.safetensors");
            let stem = "transformer_blocks.0.attn.to_q";
            candle_core::safetensors::save(
                &HashMap::from([
                    (
                        format!("{stem}.lora_A.weight"),
                        Tensor::ones((1, 1), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    (
                        format!("{stem}.lora_B.weight"),
                        Tensor::full(value, (1, 1), &Device::Cpu).unwrap(),
                    ),
                ]),
                &replacement,
            )
            .unwrap();
            std::fs::rename(replacement, &path).unwrap();
            let adapter = get_or_load_adapter(&path).unwrap();
            let (patches, _) = build_patches(&[LoraSpec {
                adapter: &adapter,
                scale: 1.0,
            }]);
            let backend = LoraBackend {
                st: unsafe { candle_core::safetensors::MmapedSafetensors::new(&base).unwrap() },
                prefix: String::new(),
                patches,
                delta_cache: Some(cache.clone()),
            };
            let values = backend
                .get_unchecked(key, DType::F32, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(values, vec![value, 0.0, 0.0]);
            // Exercise the GGUF consumer with deltas populated by the dense consumer,
            // and compare the same merge with its cache disabled.
            for delta_cache in [Some(cache.clone()), None] {
                let vb = gguf_lora_var_builder(
                    &gguf_path,
                    &[LoraSpec {
                        adapter: &adapter,
                        scale: 1.0,
                    }],
                    &Device::Cpu,
                    &ProgressReporter::default(),
                    delta_cache,
                )
                .unwrap();
                let merged = vb
                    .get((3, 1), key)
                    .unwrap()
                    .dequantize(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(merged, values);
            }
        }
    }

    #[test]
    fn audit_delta_cache_follows_source_layers_when_patch_order_changes() {
        use candle_nn::var_builder::SimpleBackend;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let key = "joint_blocks.0.x_block.attn.qkv.weight";
        candle_core::safetensors::save(
            &HashMap::from([(
                key,
                Tensor::zeros((3, 1), DType::F32, &Device::Cpu).unwrap(),
            )]),
            &base,
        )
        .unwrap();
        let layers = [("q", 1.0f32), ("k", 2.0), ("v", 3.0)]
            .into_iter()
            .map(|(name, value)| {
                (
                    format!("transformer_blocks.0.attn.to_{name}"),
                    LoraLayer {
                        a: Tensor::ones((1, 1), DType::F32, &Device::Cpu).unwrap(),
                        b: Tensor::full(value, (1, 1), &Device::Cpu).unwrap(),
                        alpha: None,
                    },
                )
            })
            .collect();
        let adapter = LoraAdapter::from_layers(layers, 1);
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 1.0,
        }];
        let cache = Arc::new(Mutex::new(LoraDeltaCache::new()));
        for reverse in [false, true] {
            let (mut patches, _) = build_patches(&specs);
            if reverse {
                patches.get_mut(key).unwrap().reverse();
            }
            let backend = LoraBackend {
                st: unsafe { candle_core::safetensors::MmapedSafetensors::new(&base).unwrap() },
                prefix: String::new(),
                patches,
                delta_cache: Some(cache.clone()),
            };
            let values = backend
                .get_unchecked(key, DType::F32, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                values,
                vec![1.0, 2.0, 3.0],
                "cache identity must follow source Q/K/V, not vector order"
            );
        }
    }

    #[test]
    fn audit_adapter_alpha_accepts_integer_and_vector_scalars() {
        let dir = tempfile::tempdir().unwrap();
        for alpha in [
            Tensor::new(8i64, &Device::Cpu).unwrap(),
            Tensor::new(&[8f32], &Device::Cpu).unwrap(),
        ] {
            let path = dir.path().join("alpha.safetensors");
            let stem = "transformer_blocks.0.attn.to_q";
            candle_core::safetensors::save(
                &HashMap::from([
                    (
                        format!("{stem}.lora_A.weight"),
                        Tensor::ones((16, 1), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    (
                        format!("{stem}.lora_B.weight"),
                        Tensor::ones((1, 16), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    (format!("{stem}.alpha"), alpha),
                ]),
                &path,
            )
            .unwrap();
            let adapter = LoraAdapter::load(&path).unwrap();
            let (patches, _) = build_patches(&[LoraSpec {
                adapter: &adapter,
                scale: 1.0,
            }]);
            assert_eq!(patches.values().next().unwrap()[0].effective_scale, 0.5);
        }
    }

    use super::*;
    use safetensors::tensor::TensorView;

    // ── Key mapping — diffusers convention ───────────────────────────────

    #[test]
    fn map_x_block_qkv_components() {
        for (suffix, component) in [("to_q", 0), ("to_k", 1), ("to_v", 2)] {
            let key = format!("transformer.transformer_blocks.5.attn.{suffix}");
            match map_lora_key(&key).expect("known leaf") {
                LoraTarget::FusedSlice {
                    candle_key,
                    component: c,
                    num_components,
                } => {
                    assert_eq!(candle_key, "joint_blocks.5.x_block.attn.qkv.weight");
                    assert_eq!(c, component);
                    assert_eq!(num_components, 3);
                }
                _ => panic!("expected FusedSlice for {suffix}"),
            }
        }
    }

    #[test]
    fn map_context_block_qkv_components() {
        for (suffix, component) in [("add_q_proj", 0), ("add_k_proj", 1), ("add_v_proj", 2)] {
            let key = format!("transformer.transformer_blocks.0.attn.{suffix}");
            match map_lora_key(&key).expect("known leaf") {
                LoraTarget::FusedSlice {
                    candle_key,
                    component: c,
                    num_components,
                } => {
                    assert_eq!(candle_key, "joint_blocks.0.context_block.attn.qkv.weight");
                    assert_eq!(c, component);
                    assert_eq!(num_components, 3);
                }
                _ => panic!("expected FusedSlice for {suffix}"),
            }
        }
    }

    #[test]
    fn map_attn_output_projections() {
        match map_lora_key("transformer.transformer_blocks.3.attn.to_out.0").unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "joint_blocks.3.x_block.attn.proj.weight")
            }
            _ => panic!("expected Direct"),
        }
        match map_lora_key("transformer.transformer_blocks.3.attn.to_add_out").unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "joint_blocks.3.context_block.attn.proj.weight")
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn map_attn2_mmdit_x_only() {
        // SD3.5-medium MMDiT-X variant — x_block has a second attention
        // tower keyed as `attn2`. Same fuse layout, distinct candle key.
        match map_lora_key("transformer.transformer_blocks.7.attn2.to_q").unwrap() {
            LoraTarget::FusedSlice {
                candle_key,
                component,
                num_components,
            } => {
                assert_eq!(candle_key, "joint_blocks.7.x_block.attn2.qkv.weight");
                assert_eq!(component, 0);
                assert_eq!(num_components, 3);
            }
            _ => panic!("expected FusedSlice"),
        }
        match map_lora_key("transformer.transformer_blocks.7.attn2.to_out.0").unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "joint_blocks.7.x_block.attn2.proj.weight")
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn map_feed_forward_both_streams() {
        let cases = [
            (
                "transformer.transformer_blocks.2.ff.net.0.proj",
                "joint_blocks.2.x_block.mlp.fc1.weight",
            ),
            (
                "transformer.transformer_blocks.2.ff.net.2",
                "joint_blocks.2.x_block.mlp.fc2.weight",
            ),
            (
                "transformer.transformer_blocks.2.ff_context.net.0.proj",
                "joint_blocks.2.context_block.mlp.fc1.weight",
            ),
            (
                "transformer.transformer_blocks.2.ff_context.net.2",
                "joint_blocks.2.context_block.mlp.fc2.weight",
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
    fn map_adaln_modulation_linears() {
        match map_lora_key("transformer.transformer_blocks.4.norm1.linear").unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(
                    candle_key,
                    "joint_blocks.4.x_block.adaLN_modulation.1.weight"
                )
            }
            _ => panic!("expected Direct"),
        }
        match map_lora_key("transformer.transformer_blocks.4.norm1_context.linear").unwrap() {
            LoraTarget::Direct { candle_key } => {
                assert_eq!(
                    candle_key,
                    "joint_blocks.4.context_block.adaLN_modulation.1.weight"
                )
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn map_strips_transformer_prefix_optional() {
        // Some exporters drop the `transformer.` prefix — accept either.
        match map_lora_key("transformer_blocks.0.attn.to_q").unwrap() {
            LoraTarget::FusedSlice { candle_key, .. } => {
                assert_eq!(candle_key, "joint_blocks.0.x_block.attn.qkv.weight")
            }
            _ => panic!("expected FusedSlice"),
        }
    }

    #[test]
    fn map_unknown_keys_return_none() {
        assert!(map_lora_key("totally.unknown.key").is_none());
        assert!(map_lora_key("transformer.unknown_block.0.attn.to_q").is_none());
        assert!(map_lora_key("transformer.transformer_blocks.0.bogus_leaf").is_none());
        // Encoder-LoRA keys (CLIP/T5) — out of scope for the MMDiT path.
        assert!(map_lora_key("text_encoder.layers.0.attn.k_proj").is_none());
    }

    // ── Key mapping — Kohya/sd-scripts convention ────────────────────────

    #[test]
    fn map_kohya_joint_block_all_leaves() {
        let cases = [
            (
                "lora_unet_joint_blocks_0_x_block_attn_qkv",
                "joint_blocks.0.x_block.attn.qkv.weight",
            ),
            (
                "lora_unet_joint_blocks_0_x_block_attn_proj",
                "joint_blocks.0.x_block.attn.proj.weight",
            ),
            (
                "lora_unet_joint_blocks_5_x_block_attn2_qkv",
                "joint_blocks.5.x_block.attn2.qkv.weight",
            ),
            (
                "lora_unet_joint_blocks_5_x_block_attn2_proj",
                "joint_blocks.5.x_block.attn2.proj.weight",
            ),
            (
                "lora_unet_joint_blocks_3_x_block_mlp_fc1",
                "joint_blocks.3.x_block.mlp.fc1.weight",
            ),
            (
                "lora_unet_joint_blocks_3_x_block_mlp_fc2",
                "joint_blocks.3.x_block.mlp.fc2.weight",
            ),
            (
                "lora_unet_joint_blocks_3_x_block_adaLN_modulation_1",
                "joint_blocks.3.x_block.adaLN_modulation.1.weight",
            ),
            (
                "lora_unet_joint_blocks_7_context_block_attn_qkv",
                "joint_blocks.7.context_block.attn.qkv.weight",
            ),
            (
                "lora_unet_joint_blocks_7_context_block_attn_proj",
                "joint_blocks.7.context_block.attn.proj.weight",
            ),
            (
                "lora_unet_joint_blocks_2_context_block_mlp_fc1",
                "joint_blocks.2.context_block.mlp.fc1.weight",
            ),
            (
                "lora_unet_joint_blocks_2_context_block_mlp_fc2",
                "joint_blocks.2.context_block.mlp.fc2.weight",
            ),
            (
                "lora_unet_joint_blocks_2_context_block_adaLN_modulation_1",
                "joint_blocks.2.context_block.adaLN_modulation.1.weight",
            ),
        ];
        for (kohya_key, expected) in cases {
            match map_lora_key(kohya_key).unwrap() {
                LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "{kohya_key}")
                }
                _ => panic!("expected Direct for {kohya_key}"),
            }
        }
    }

    #[test]
    fn map_kohya_unrelated_or_te_returns_none() {
        // Text-encoder LoRAs and unrecognized leaves are skipped (caller
        // logs a warning).
        assert!(map_lora_key("lora_te_text_model_layer_0_attn_q").is_none());
        assert!(map_lora_key("lora_unet_joint_blocks_0_unknown_leaf").is_none());
        assert!(map_lora_key("lora_unet_some_other_block_0_x").is_none());
    }

    // ── fused_slice_range ────────────────────────────────────────────────

    #[test]
    fn fused_slice_range_qkv_equal_split() {
        // SD3.5-large hidden=2432 (depth 38 × head_size 64).
        // qkv weight rows = 3 × hidden = 7296 — split into 3 × 2432.
        let (offset, size) = fused_slice_range(7296, 2432, 0, 3);
        assert_eq!((offset, size), (0, 2432));
        let (offset, size) = fused_slice_range(7296, 2432, 1, 3);
        assert_eq!((offset, size), (2432, 2432));
        let (offset, size) = fused_slice_range(7296, 2432, 2, 3);
        assert_eq!((offset, size), (4864, 2432));
    }

    // ── classify_lora_key — suffix matrix ────────────────────────────────

    #[test]
    fn classify_diffusers_kohya_onetrainer_peft_mochi() {
        assert_eq!(
            classify_lora_key("x.lora_A.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B.weight"),
            Some((LoraDirection::Up, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_down.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_up.weight"),
            Some((LoraDirection::Up, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_linear_layer.down.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_linear_layer.up.weight"),
            Some((LoraDirection::Up, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_A.default.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B.default.weight"),
            Some((LoraDirection::Up, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B"),
            Some((LoraDirection::Up, "x"))
        );
    }

    #[test]
    fn classify_lora_key_unrelated_returns_none() {
        assert_eq!(classify_lora_key("x.weight"), None);
        assert_eq!(classify_lora_key("transformer.embed.weight"), None);
        assert_eq!(classify_lora_key("layer.alpha"), None);
    }

    // ── Adapter loading — synthetic safetensors fixtures ─────────────────

    /// Synthetic diffusers-keyed safetensors targeting a single attn
    /// leaf — enough to prove `LoraAdapter::load` recognises the format.
    fn write_diffusers_fixture(path: &Path) {
        let stem = "transformer.transformer_blocks.0.attn.to_q";
        // A: (rank=2, in=4), B: (out=6, rank=2)
        let down: Vec<f32> = (0..2 * 4).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..6 * 2).map(|i| i as f32 * 0.2).collect();

        let down_bytes: Vec<u8> = down.iter().flat_map(|f| f.to_le_bytes()).collect();
        let up_bytes: Vec<u8> = up.iter().flat_map(|f| f.to_le_bytes()).collect();

        let down_view = TensorView::new(safetensors::Dtype::F32, vec![2, 4], &down_bytes).unwrap();
        let up_view = TensorView::new(safetensors::Dtype::F32, vec![6, 2], &up_bytes).unwrap();

        let entries: Vec<(String, TensorView)> = vec![
            (format!("{stem}.lora_A.weight"), down_view),
            (format!("{stem}.lora_B.weight"), up_view),
        ];
        safetensors::serialize_to_file(entries, &None, path).expect("write safetensors");
    }

    /// Kohya-keyed safetensors with alpha — proves `lora_down/lora_up/
    /// alpha` are picked up and paired together by stem.
    fn write_kohya_fixture(path: &Path) {
        let stem = "lora_unet_joint_blocks_0_x_block_attn_qkv";
        let down: Vec<f32> = (0..2 * 4).map(|i| i as f32 * 0.1).collect();
        let up: Vec<f32> = (0..6 * 2).map(|i| i as f32 * 0.2).collect();
        let alpha: Vec<f32> = vec![16.0];

        let down_bytes: Vec<u8> = down.iter().flat_map(|f| f.to_le_bytes()).collect();
        let up_bytes: Vec<u8> = up.iter().flat_map(|f| f.to_le_bytes()).collect();
        let alpha_bytes: Vec<u8> = alpha.iter().flat_map(|f| f.to_le_bytes()).collect();

        let down_view = TensorView::new(safetensors::Dtype::F32, vec![2, 4], &down_bytes).unwrap();
        let up_view = TensorView::new(safetensors::Dtype::F32, vec![6, 2], &up_bytes).unwrap();
        let alpha_view = TensorView::new(safetensors::Dtype::F32, vec![], &alpha_bytes).unwrap();

        let entries: Vec<(String, TensorView)> = vec![
            (format!("{stem}.lora_down.weight"), down_view),
            (format!("{stem}.lora_up.weight"), up_view),
            (format!("{stem}.alpha"), alpha_view),
        ];
        safetensors::serialize_to_file(entries, &None, path).expect("write safetensors");
    }

    #[test]
    fn load_diffusers_safetensors_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("diffusers.safetensors");
        write_diffusers_fixture(&path);

        let adapter = LoraAdapter::load(&path).expect("diffusers fixture must load");
        assert_eq!(adapter.layers.len(), 1);
        assert_eq!(adapter.rank, 2);
        let stem = "transformer.transformer_blocks.0.attn.to_q";
        let layer = adapter.layers.get(stem).expect("layer present");
        assert_eq!(layer.a.dims(), &[2, 4]);
        assert_eq!(layer.b.dims(), &[6, 2]);
        assert!(layer.alpha.is_none());
    }

    #[test]
    fn load_kohya_safetensors_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kohya.safetensors");
        write_kohya_fixture(&path);

        let adapter = LoraAdapter::load(&path).expect("kohya fixture must load");
        assert_eq!(adapter.layers.len(), 1);
        assert_eq!(adapter.rank, 2);
        let stem = "lora_unet_joint_blocks_0_x_block_attn_qkv";
        let layer = adapter.layers.get(stem).expect("layer present");
        assert_eq!(layer.alpha, Some(16.0));
    }

    // ── build_patches stacking ───────────────────────────────────────────

    fn synthetic_single_layer_adapter(scale_a: f32, scale_b: f32) -> LoraAdapter {
        let device = Device::Cpu;
        // Targets x_block.attn.proj.weight — a direct mapping with no
        // fused slicing, so the build-patches assertion is simple.
        let a = Tensor::full(scale_a, (2, 4), &device).unwrap();
        let b = Tensor::full(scale_b, (4, 2), &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer_blocks.0.attn.to_out.0".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        LoraAdapter::from_layers(layers, 2)
    }

    #[test]
    fn build_patches_stacks_two_adapters_on_same_tensor() {
        let a1 = synthetic_single_layer_adapter(1.0, 1.0);
        let a2 = synthetic_single_layer_adapter(2.0, 3.0);
        let specs = [
            LoraSpec {
                adapter: &a1,
                scale: 0.5,
            },
            LoraSpec {
                adapter: &a2,
                scale: 0.25,
            },
        ];
        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0);
        let stack = patches
            .get("joint_blocks.0.x_block.attn.proj.weight")
            .expect("present");
        assert_eq!(stack.len(), 2);
    }

    #[test]
    fn build_patches_alpha_overrides_scale_with_rank_normalisation() {
        let device = Device::Cpu;
        let a = Tensor::full(1.0f32, (4, 4), &device).unwrap(); // rank=4
        let b = Tensor::full(1.0f32, (4, 4), &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer_blocks.0.attn.to_out.0".to_string(),
            LoraLayer {
                a,
                b,
                alpha: Some(8.0),
            },
        );
        let adapter = LoraAdapter::from_layers(layers, 4);
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 0.5,
        }];
        let (patches, _) = build_patches(&specs);
        let patch = &patches["joint_blocks.0.x_block.attn.proj.weight"][0];
        // effective = scale * alpha / rank = 0.5 * 8 / 4 = 1.0
        assert!((patch.effective_scale - 1.0).abs() < 1e-9);
    }

    // ── Prefix detection ─────────────────────────────────────────────────

    #[test]
    fn detect_prefix_picks_stabilityai() {
        let names = vec!["model.diffusion_model.x_embedder.proj.weight".to_string()];
        assert_eq!(detect_prefix(&names), "model.diffusion_model.");
    }

    #[test]
    fn detect_prefix_picks_diffusion_model() {
        let names = vec!["diffusion_model.x_embedder.proj.weight".to_string()];
        assert_eq!(detect_prefix(&names), "diffusion_model.");
    }

    #[test]
    fn detect_prefix_picks_root() {
        let names = vec!["x_embedder.proj.weight".to_string()];
        assert_eq!(detect_prefix(&names), "");
    }

    // ── End-to-end via LoraBackend: tiny synthetic MMDiT-style tensor ────

    /// Build a synthetic safetensors with a fake `x_block.attn.proj`
    /// tensor under the `model.diffusion_model.` prefix and confirm
    /// that loading through `LoraBackend` reproduces the merged weights
    /// at the requested dtype.
    #[test]
    fn lora_backend_applies_direct_delta_to_base_tensor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base_path = dir.path().join("base.safetensors");

        // Base tensor: 4×4 of zeros, BF16 (the path that mmap'd
        // safetensors will actually load).
        let base_data: Vec<f32> = vec![0.0; 16];
        let base_bytes: Vec<u8> = base_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base_view = TensorView::new(safetensors::Dtype::F32, vec![4, 4], &base_bytes).unwrap();
        let candle_key = "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight";
        let entries: Vec<(String, TensorView)> = vec![(candle_key.to_string(), base_view)];
        safetensors::serialize_to_file(entries, &None, &base_path).expect("write base");

        // LoRA: A=(2,4) of ones, B=(4,2) of ones → B@A = (4,4) of 2's,
        // scaled by 0.5 → merged = 1.0 everywhere.
        let a = Tensor::full(1.0f32, (2, 4), &Device::Cpu).unwrap();
        let b = Tensor::full(1.0f32, (4, 2), &Device::Cpu).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer_blocks.0.attn.to_out.0".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter::from_layers(layers, 2);
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 0.5,
        }];

        let progress = ProgressReporter::default();
        let vb = lora_var_builder(
            &base_path,
            &specs,
            DType::F32,
            &Device::Cpu,
            &progress,
            None,
        )
        .expect("lora_var_builder must build");

        // The vb was built at the on-disk prefix root, so to fetch the
        // relative key we go through the backend directly. (In the
        // real pipeline the model constructor calls `vb.pp("model.
        // diffusion_model").pp("joint_blocks.0")…`, which composes the
        // same relative path).
        let merged = vb
            .get((4, 4), "joint_blocks.0.x_block.attn.proj.weight")
            .expect("merged tensor must load");
        let merged_f32: Vec<f32> = merged.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in merged_f32 {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0 everywhere, got {v}");
        }
    }

    /// Fused-slice path: build a synthetic fused-QKV tensor where the
    /// V slice has a recognisable pattern, attach a LoRA that targets
    /// the V slice, and assert that ONLY the V slice changed.
    #[test]
    fn lora_backend_applies_fused_slice_to_v_component_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base_path = dir.path().join("qkv_base.safetensors");

        // Base tensor: 12×4 split as [Q rows 0..4 | K rows 4..8 | V rows
        // 8..12]. Fill each component with a distinguishable constant.
        let mut base_data: Vec<f32> = Vec::with_capacity(48);
        base_data.extend(std::iter::repeat_n(10.0, 16)); // Q
        base_data.extend(std::iter::repeat_n(20.0, 16)); // K
        base_data.extend(std::iter::repeat_n(30.0, 16)); // V
        let base_bytes: Vec<u8> = base_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base_view = TensorView::new(safetensors::Dtype::F32, vec![12, 4], &base_bytes).unwrap();
        let candle_key = "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight";
        let entries: Vec<(String, TensorView)> = vec![(candle_key.to_string(), base_view)];
        safetensors::serialize_to_file(entries, &None, &base_path).expect("write base");

        // LoRA targeting `attn.to_v` (V slice): A=(2,4) of ones, B=(4,2)
        // of ones → B@A = (4,4) of 2's, scaled by 1.0 → V slice gains 2.
        let a = Tensor::full(1.0f32, (2, 4), &Device::Cpu).unwrap();
        let b = Tensor::full(1.0f32, (4, 2), &Device::Cpu).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer_blocks.0.attn.to_v".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter::from_layers(layers, 2);
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 1.0,
        }];

        let progress = ProgressReporter::default();
        let vb = lora_var_builder(
            &base_path,
            &specs,
            DType::F32,
            &Device::Cpu,
            &progress,
            None,
        )
        .expect("lora_var_builder must build");

        let merged = vb
            .get((12, 4), "joint_blocks.0.x_block.attn.qkv.weight")
            .expect("merged tensor must load");
        let merged_f32: Vec<f32> = merged.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Q rows 0..4 (16 entries) — unchanged.
        for v in &merged_f32[0..16] {
            assert!(
                (v - 10.0).abs() < 1e-5,
                "Q slice must be untouched, got {v}"
            );
        }
        // K rows 4..8 (16 entries) — unchanged.
        for v in &merged_f32[16..32] {
            assert!(
                (v - 20.0).abs() < 1e-5,
                "K slice must be untouched, got {v}"
            );
        }
        // V rows 8..12 (16 entries) — 30 + 2 = 32.
        for v in &merged_f32[32..48] {
            assert!(
                (v - 32.0).abs() < 1e-5,
                "V slice must be merged with LoRA delta (+2), got {v}"
            );
        }
    }

    // ── End-to-end engine smoke: load through MMDiT::new ─────────────────
    //
    // The actual MMDiT requires the full set of weights (patch embedder,
    // pos embedder, timestep embedder, vector embedder, context embedder,
    // 24 or 38 joint blocks, final layer) — synthesising the full weight
    // shape is out of scope at this granularity. The per-tensor merge
    // round-trip tests above prove the SimpleBackend pattern composes
    // correctly with candle's VarBuilder, which is what MMDiT::new
    // ultimately consumes. A full end-to-end engine smoke test would
    // need on-disk SD3 weights and is documented in the integration
    // notes for when one becomes available.
    //
    // If you want a real-LoRA smoke test, point at any SD3.5 LoRA on
    // Civitai (e.g. https://civitai.com/models/879428 — search "SD3.5
    // LoRA") and gate the test behind `#[ignore]`, downloading it once
    // to `target/test-fixtures/sd3-lora.safetensors`.

    // ── parsed_lora_cache ────────────────────────────────────────────────

    #[test]
    fn parsed_lora_cache_hits_on_second_load() {
        let _cache_test_guard = lock_parsed_lora_cache_tests();
        clear_parsed_lora_cache_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("hit.safetensors");
        write_kohya_fixture(&path);

        let first = get_or_load_adapter(&path).expect("first load");
        let second = get_or_load_adapter(&path).expect("second load");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second load must return the same Arc — proof the cache hit, no re-parse"
        );
    }

    #[test]
    fn parsed_lora_cache_invalidates_on_mtime_change() {
        let _cache_test_guard = lock_parsed_lora_cache_tests();
        clear_parsed_lora_cache_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("invalidate.safetensors");
        write_kohya_fixture(&path);

        let first = get_or_load_adapter(&path).expect("first load");

        // Bump mtime past common filesystem mtime resolution (1s on HFS+).
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write_kohya_fixture(&path);

        let second = get_or_load_adapter(&path).expect("second load");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "mtime change must produce a fresh Arc — proof the cache key invalidated"
        );
    }
}
