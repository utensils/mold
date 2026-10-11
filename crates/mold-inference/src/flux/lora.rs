#[cfg(test)]
use crate::lora::{classify_lora_key, read_alpha_scalar, LoraDirection, LoraLayer};
#[cfg(test)]
use crate::lora::{
    clear_parsed_lora_cache_for_test, lock_parsed_lora_cache_tests, unsupported_layout_hint,
};
pub(crate) use crate::lora::{get_or_load_adapter, LoraAdapter, LoraDeltaCache};
use crate::lora::{AdapterIdentity, LoraCacheKey};
use crate::progress::ProgressReporter;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Describes how a diffusers-format LoRA key maps to a candle model tensor.
pub(crate) enum LoraTarget {
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

/// Map a LoRA layer key (diffusers- or Kohya-format) to a candle model target.
///
/// Returns None for unrecognized keys (logged as warning, skipped).
pub(crate) fn map_lora_key(diffusers_key: &str) -> Option<LoraTarget> {
    // Kohya / sd-scripts naming (`lora_unet_*`) — keys carry the FLUX module
    // path with `.` flattened to `_`. The transformer's fused tensors
    // (`img_attn.qkv`, `txt_attn.qkv`, `single_blocks.*.linear1`) match the
    // candle layout 1:1, so every Kohya key maps to a single Direct target —
    // no FusedSlice splitting needed (Kohya already trains a B@A delta of
    // the full fused output shape).
    if let Some(rest) = diffusers_key.strip_prefix("lora_unet_") {
        return map_kohya_unet_key(rest);
    }

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

/// Map a Kohya/sd-scripts FLUX UNet LoRA key (with the `lora_unet_` prefix
/// already stripped) to a candle model target.
///
/// Kohya's flat-naming scheme is unambiguous after `lora_unet_<block>_<idx>_`
/// — what follows is one of a small fixed set of leaves per block kind. We
/// match on that suffix directly rather than try to reverse the
/// `.`→`_` collapse, which is ambiguous (`img_attn` vs `img.attn`).
fn map_kohya_unet_key(rest: &str) -> Option<LoraTarget> {
    if let Some(after) = rest.strip_prefix("double_blocks_") {
        let (idx_str, suffix) = after.split_once('_')?;
        idx_str.parse::<usize>().ok()?;
        let candle_key = match suffix {
            "img_attn_qkv" => format!("double_blocks.{idx_str}.img_attn.qkv.weight"),
            "img_attn_proj" => format!("double_blocks.{idx_str}.img_attn.proj.weight"),
            "img_mlp_0" => format!("double_blocks.{idx_str}.img_mlp.0.weight"),
            "img_mlp_2" => format!("double_blocks.{idx_str}.img_mlp.2.weight"),
            "img_mod_lin" => format!("double_blocks.{idx_str}.img_mod.lin.weight"),
            "txt_attn_qkv" => format!("double_blocks.{idx_str}.txt_attn.qkv.weight"),
            "txt_attn_proj" => format!("double_blocks.{idx_str}.txt_attn.proj.weight"),
            "txt_mlp_0" => format!("double_blocks.{idx_str}.txt_mlp.0.weight"),
            "txt_mlp_2" => format!("double_blocks.{idx_str}.txt_mlp.2.weight"),
            "txt_mod_lin" => format!("double_blocks.{idx_str}.txt_mod.lin.weight"),
            _ => return None,
        };
        return Some(LoraTarget::Direct { candle_key });
    }
    if let Some(after) = rest.strip_prefix("single_blocks_") {
        let (idx_str, suffix) = after.split_once('_')?;
        idx_str.parse::<usize>().ok()?;
        let candle_key = match suffix {
            "linear1" => format!("single_blocks.{idx_str}.linear1.weight"),
            "linear2" => format!("single_blocks.{idx_str}.linear2.weight"),
            "modulation_lin" => format!("single_blocks.{idx_str}.modulation.lin.weight"),
            _ => return None,
        };
        return Some(LoraTarget::Direct { candle_key });
    }
    None
}

/// Compute the row offset and size for a fused slice, handling both
/// equal-split (QKV with num_components=3) and single-block linear1
/// (Q,K,V each h_sz, then MLP is the remainder).
pub(crate) fn fused_slice_range(
    base_rows: usize,
    lora_out_dim: usize,
    component: usize,
    num_components: usize,
) -> (usize, usize) {
    if let Some(component_size) = base_rows.checked_div(num_components) {
        // Equal split (e.g. QKV fused: each is base_rows / 3)
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

/// A `SimpleBackend` that wraps mmap'd safetensors and applies LoRA deltas
/// on-the-fly when the model constructor requests each tensor.
///
/// This is the ComfyUI/InvokeAI approach adapted for candle:
/// - Tensors are loaded lazily from mmap (identical memory profile to non-LoRA)
/// - LoRA deltas are computed and applied per-tensor as `Flux::new()` calls `vb.get()`
/// - Peak VRAM = final model size only (no pre-loaded HashMap)
///
/// The A×B matmul runs on the target device (GPU if available), and the merge
/// (F32 cast + add + cast back) also happens on the target device since we're
/// processing one tensor at a time with plenty of headroom.
struct LoraBackend {
    /// The mmap'd base safetensors.
    st: candle_core::safetensors::MmapedSafetensors,
    /// Key prefix to strip (e.g. "model.diffusion_model.").
    prefix: String,
    /// Pre-computed LoRA patches keyed by canonical tensor name. With
    /// multi-LoRA the inner `Vec` may contain patches from different
    /// adapters; each patch carries its source layer and file revision so
    /// the delta-cache key stays unique per (tensor, adapter, slice).
    patches: HashMap<String, Vec<LoraPatch>>,
    /// Optional CPU-resident cache of pre-computed deltas (shared across rebuilds).
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

/// A single LoRA patch to apply to a base tensor. Multiple patches on the
/// same tensor stack additively: `W' = W + Σ scale_i · B_i @ A_i` — and
/// because each patch carries its own source revision, the delta cache
/// can disambiguate "the cinematic LoRA's contribution to img_in.weight"
/// from "the lighting LoRA's contribution to img_in.weight" without
/// recomputing either.
struct LoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: LoraTarget,
    /// Stable source identity, independent of HashMap iteration order.
    source_layer: String,
    adapter_identity: AdapterIdentity,
}

/// A loaded adapter (including its revision) and requested scale.
pub(crate) struct LoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
}

/// Walk every (adapter, layer) pair across `specs` and turn it into a
/// `LoraPatch` keyed by its target candle tensor. Multiple specs that
/// touch the same tensor accumulate into the same `Vec` so the backend
/// applies them additively. The returned counter is the number of
/// `lora_*` keys we couldn't map (logged by the caller as `skipped`).
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
                    "unrecognized LoRA key, skipping"
                );
                skipped += 1;
            }
        }
    }
    (patches, skipped)
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
        // Resolve the raw key in the safetensors file
        let raw_key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{name}", self.prefix)
        };

        // Load from mmap directly to target device (same as non-LoRA path)
        let tensor = self.st.load(&raw_key, dev)?;
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Apply LoRA patches if any target this tensor
        if let Some(patches) = self.patches.get(name) {
            let mut t = tensor;
            for patch in patches {
                // Source layer plus revision disambiguates fused slices and adapters.
                let cache_key = LoraCacheKey {
                    tensor_name: name.to_string(),
                    source_layer: patch.source_layer.clone(),
                    adapter_identity: patch.adapter_identity.clone(),
                    scale_bits: patch.effective_scale.to_bits(),
                };

                // Try to retrieve from cache (CPU-resident delta)
                let cached_delta = self.delta_cache.as_ref().and_then(|c| {
                    c.lock()
                        .ok()
                        .and_then(|guard| guard.get(&cache_key).cloned())
                });

                let delta = if let Some(cpu_delta) = cached_delta {
                    // Cache hit: move to target device
                    cpu_delta.to_device(dev)?
                } else {
                    // Cache miss: compute delta on target device
                    let a = patch.a.to_dtype(DType::F32)?.to_device(dev)?;
                    let b = patch.b.to_dtype(DType::F32)?.to_device(dev)?;
                    let computed = b.matmul(&a)?;
                    let computed = (&computed * patch.effective_scale)?;

                    // Store on CPU for future rebuilds
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
                        let merged = (&t_f32 + &delta)?;
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
                                offset,
                                size,
                                base_rows,
                                "fused slice out of bounds, skipping"
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
        // Check via trying to load metadata (tensors() lists all names)
        self.st.get(&raw_key).is_ok()
    }
}

/// Build a LoRA-patching VarBuilder that wraps mmap'd safetensors.
///
/// This uses candle's `SimpleBackend` trait to intercept every `vb.get()` call
/// during model construction.  Each tensor is loaded from mmap directly to the
/// target device (GPU), with LoRA deltas applied inline.  Memory profile is
/// identical to the non-LoRA mmap path — no HashMap, no pre-loading.
///
/// Multi-LoRA: pass multiple `LoraSpec`s and they merge additively. The
/// per-patch cache key (tensor, source layer, file revision, scale bits)
/// keeps each adapter's delta independently cacheable.
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

    // Open mmap (cheap, no I/O)
    let st = unsafe { MmapedSafetensors::multi(std::slice::from_ref(&transformer_path))? };

    // Detect key prefix
    let all_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let prefix = if all_names.iter().any(|n| n == "img_in.weight") {
        ""
    } else if all_names
        .iter()
        .any(|n| n == "model.diffusion_model.img_in.weight")
    {
        "model.diffusion_model."
    } else if all_names
        .iter()
        .any(|n| n == "diffusion_model.img_in.weight")
    {
        "diffusion_model."
    } else {
        ""
    };

    // Build patch index: for each candle key, collect all LoRA patches
    // (across every adapter in `specs`).
    let (patches, skipped) = build_patches(specs);

    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA: {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let backend = LoraBackend {
        st,
        prefix: prefix.to_string(),
        patches,
        delta_cache,
    };

    Ok(candle_nn::VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

/// Build a quantized VarBuilder from a GGUF file with LoRA deltas applied.
///
/// Loads the GGUF file into a `HashMap<String, Arc<QTensor>>`, then for each
/// LoRA-targeted tensor: dequantizes to F32 on CPU, applies the LoRA delta,
/// and re-quantizes back to the original GGML dtype (e.g. Q8_0) on the target
/// device.  Non-LoRA tensors stay quantized and untouched.
///
/// By re-quantizing to the original dtype instead of storing as F16/BF16, each
/// patched tensor occupies the same VRAM as its original — no inflation.  The
/// LoRA rank is small (typically 32) so the re-quantization error is negligible.
pub(crate) fn gguf_lora_var_builder(
    transformer_path: &Path,
    specs: &[LoraSpec<'_>],
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<mold_candle::quantized::VarBuilder> {
    use candle_core::quantized::QTensor;
    use std::sync::Arc;

    if specs.is_empty() {
        bail!("gguf_lora_var_builder called with no LoraSpecs — caller must provide at least one");
    }

    let map = mold_candle::gguf_mmap::GgufMmap::open(transformer_path)?;

    // Build patch index (same as safetensors LoRA path) — accumulate
    // patches from every adapter into the same map. The downstream
    // dequant→merge→requant loop already iterates per-tensor patches in
    // sequence, so multi-LoRA stacking is just additional entries.
    let (patches, skipped) = build_patches(specs);

    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA: {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    // Phase 1: Load ALL tensors off the mapping — the exact same CUDA
    // allocation as the non-LoRA path, and now the same transport. The counter
    // reports each tensor's real payload rather than the tensor-count
    // approximation this replaced, because the map knows every size up front.
    let mut data: HashMap<String, Arc<QTensor>> = map.load_all(device, &mut |done, total| {
        progress.weight_load("FLUX transformer (GGUF)", done, total)
    })?;
    drop(map);

    // Phase 2: Patch LoRA-affected tensors in-place.
    // For each target: dequantize the GPU QTensor to F32 on CPU, apply LoRA
    // delta, re-quantize back to the original GGML dtype (e.g. Q8_0), and
    // place the result on GPU.  This keeps each patched tensor at its original
    // quantized size — no VRAM inflation.
    let on_gpu = device.is_cuda() || device.is_metal();
    let mut applied = 0usize;
    let lora_keys: Vec<String> = patches.keys().cloned().collect();
    let lora_total = lora_keys.len();
    for (i, candle_key) in lora_keys.iter().enumerate() {
        let layer_patches = &patches[candle_key];

        // Find the matching tensor key (try with .weight suffix)
        let tensor_key = if data.contains_key(candle_key) {
            candle_key.clone()
        } else {
            // Shouldn't happen if map_lora_key produced correct candle keys
            tracing::warn!(
                key = candle_key.as_str(),
                "LoRA target tensor not found in GGUF, skipping"
            );
            continue;
        };

        // Remember the original quantized dtype so we can re-quantize to it.
        let orig_dtype = data[&tensor_key].dtype();

        // Dequantize to F32 on CPU — keeps GPU clean for other tensors.
        // The original Q8 GPU entry is removed to reclaim its VRAM.
        let qtensor = data.remove(&tensor_key).unwrap();
        let mut t = qtensor.dequantize(&Device::Cpu)?;
        drop(qtensor); // release GPU QTensor VRAM
        if on_gpu {
            device.synchronize()?; // ensure CUDA frees the Q8 allocation
        }

        for patch in layer_patches {
            // Source layer plus revision disambiguates fused slices and adapters.
            let cache_key = LoraCacheKey {
                tensor_name: candle_key.clone(),
                source_layer: patch.source_layer.clone(),
                adapter_identity: patch.adapter_identity.clone(),
                scale_bits: patch.effective_scale.to_bits(),
            };

            // Try cache first, then compute
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

                // Store in cache for future rebuilds
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
                            "fused slice out of bounds, skipping"
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

        // Re-quantize back to the original GGML dtype (e.g. Q8_0) and place
        // on the target device.  `quantize_onto` quantizes the CPU F32 tensor
        // into CPU Q8_0 blocks, then copies the raw bytes to GPU — producing
        // the exact same storage size as the original GGUF-loaded tensor.
        // This avoids the 2x VRAM inflation that storing as F16 would cause.
        let patched = mold_candle::quantized::quantize_onto(&t, orig_dtype, device)?;
        drop(t); // free CPU F32 copy
        data.insert(tensor_key, Arc::new(patched));

        if (i + 1) % 50 == 0 || i + 1 == lora_total {
            progress.info(&format!(
                "Patching LoRA tensor {}/{}",
                i + 1,
                lora_keys.len()
            ));
        }
    }

    let total_layers: usize = specs.iter().map(|s| s.adapter.layers.len()).sum();
    progress.info(&format!(
        "LoRA: {applied} applied, {} skipped (max rank {max_rank}, {patched_keys} layers patched)",
        total_layers.saturating_sub(applied),
    ));

    // Drain pending cuMemFreeAsync from the per-tensor merge loop. Each
    // patched tensor allocates F32 A/B/B@A intermediates on GPU sized like the
    // full weight (~150 MB for a 3072×12288 MLP); their drops queue async
    // frees that don't actually return VRAM to the device until a sync. With
    // 50+ LoRA-affected tensors per adapter and a stack of 2 LoRAs, the queued
    // frees pile up to several GB. Without this sync, denoising starts with a
    // bloated working set and VAE decode at 1024² OOMs even though the kept
    // transformer is the same size as the no-LoRA case.
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
        let key = "double_blocks.0.img_attn.qkv.weight";
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
        let key = "double_blocks.0.img_attn.qkv.weight";
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

    use super::*;

    /// mold has never parsed XLabs-AI's FLUX.1 layout, and the refusal now
    /// says so.
    ///
    /// Their keys are `double_blocks.N.processor.{proj,qkv}_lora{1,2}.{down,up}.weight`.
    /// The `.down.weight` there is preceded by `proj_lora1`, not by `.lora`,
    /// so it matches none of the four supported conventions — verified
    /// directly against the classifier below rather than assumed.
    ///
    /// Before the LoRA sealing fix such an adapter was dropped before it ever
    /// reached this parser, so the render succeeded with no adapter applied.
    /// Now it reaches here and fails, which is the honest answer and a
    /// user-visible change.
    #[test]
    fn an_xlabs_layout_is_unsupported_and_the_refusal_names_it() {
        const XLABS_KEYS: [&str; 4] = [
            "double_blocks.0.processor.proj_lora1.down.weight",
            "double_blocks.0.processor.proj_lora1.up.weight",
            "double_blocks.0.processor.qkv_lora2.down.weight",
            "double_blocks.0.processor.qkv_lora2.up.weight",
        ];

        // The premise: none of these is a LoRA pair tensor to mold.
        for key in XLABS_KEYS {
            assert!(
                classify_lora_key(key).is_none(),
                "{key} must not classify — mold has never supported this layout"
            );
        }

        let owned: Vec<String> = XLABS_KEYS.iter().map(|key| (*key).to_string()).collect();
        let hint = unsupported_layout_hint(owned.iter());
        assert!(
            hint.contains("XLabs"),
            "the refusal must name the layout it saw, got {hint:?}"
        );
        assert!(
            hint.contains("lora_A") && hint.contains("lora_down"),
            "and must name the exports that do work, got {hint:?}"
        );

        // A genuinely empty or unrelated file gets no speculation.
        assert_eq!(unsupported_layout_hint(std::iter::empty()), "");
        let unrelated = ["model.diffusion_model.foo.weight".to_string()];
        assert_eq!(unsupported_layout_hint(unrelated.iter()), "");

        // And the supported conventions still classify, so the hint can never
        // fire for an adapter that would have loaded.
        for key in [
            "transformer.blocks.0.attn.to_q.lora_A.weight",
            "transformer.blocks.0.attn.to_q.lora_down.weight",
            "transformer.blocks.0.attn.to_q.lora_linear_layer.down.weight",
            "transformer.blocks.0.attn.to_q.lora_A.default.weight",
        ] {
            assert!(classify_lora_key(key).is_some(), "{key} must still load");
        }
    }

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

    /// Build a synthetic adapter that touches one direct-mapped tensor
    /// (`transformer_blocks.0.attn.to_out.0`). Shape is (out=4, rank=2)
    /// for B and (rank=2, in=4) for A so the math is small but real.
    fn synthetic_single_layer_adapter(scale_a: f32, scale_b: f32) -> LoraAdapter {
        let device = candle_core::Device::Cpu;
        let a = Tensor::full(scale_a, (2, 4), &device).unwrap();
        let b = Tensor::full(scale_b, (4, 2), &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer_blocks.0.attn.to_out.0".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        LoraAdapter::from_layers(layers, 2)
    }

    /// Two adapters targeting the same tensor must produce two patches
    /// in the same Vec, each carrying its own adapter revision so the
    /// per-patch delta cache stays disambiguated.
    #[test]
    fn build_patches_stacks_multiple_specs_on_same_tensor() {
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
        assert_eq!(skipped, 0, "every test layer maps to a known target");
        let key = "double_blocks.0.img_attn.proj.weight";
        let stack = patches.get(key).expect("target tensor must be patched");
        assert_eq!(
            stack.len(),
            2,
            "both adapters must contribute a patch to the same tensor"
        );
        // Order is the order of the specs; source revisions discriminate
        // them so the delta cache can never confuse one for the other.

        assert!(
            (stack[0].effective_scale - 0.5).abs() < 1e-9,
            "first patch keeps its caller-supplied scale (no alpha override)"
        );
        assert!(
            (stack[1].effective_scale - 0.25).abs() < 1e-9,
            "second patch keeps its caller-supplied scale"
        );
    }

    #[test]
    fn map_kohya_double_block_keys() {
        let cases = [
            (
                "lora_unet_double_blocks_0_img_attn_qkv",
                "double_blocks.0.img_attn.qkv.weight",
            ),
            (
                "lora_unet_double_blocks_5_img_attn_proj",
                "double_blocks.5.img_attn.proj.weight",
            ),
            (
                "lora_unet_double_blocks_10_img_mlp_0",
                "double_blocks.10.img_mlp.0.weight",
            ),
            (
                "lora_unet_double_blocks_10_img_mlp_2",
                "double_blocks.10.img_mlp.2.weight",
            ),
            (
                "lora_unet_double_blocks_3_img_mod_lin",
                "double_blocks.3.img_mod.lin.weight",
            ),
            (
                "lora_unet_double_blocks_7_txt_attn_qkv",
                "double_blocks.7.txt_attn.qkv.weight",
            ),
            (
                "lora_unet_double_blocks_7_txt_attn_proj",
                "double_blocks.7.txt_attn.proj.weight",
            ),
            (
                "lora_unet_double_blocks_2_txt_mlp_0",
                "double_blocks.2.txt_mlp.0.weight",
            ),
            (
                "lora_unet_double_blocks_2_txt_mlp_2",
                "double_blocks.2.txt_mlp.2.weight",
            ),
            (
                "lora_unet_double_blocks_18_txt_mod_lin",
                "double_blocks.18.txt_mod.lin.weight",
            ),
        ];
        for (kohya_key, expected) in cases {
            match map_lora_key(kohya_key).unwrap() {
                LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "kohya key {kohya_key}");
                }
                _ => panic!("expected Direct for kohya key {kohya_key}"),
            }
        }
    }

    #[test]
    fn map_kohya_single_block_keys() {
        let cases = [
            (
                "lora_unet_single_blocks_0_linear1",
                "single_blocks.0.linear1.weight",
            ),
            (
                "lora_unet_single_blocks_9_linear2",
                "single_blocks.9.linear2.weight",
            ),
            (
                "lora_unet_single_blocks_37_modulation_lin",
                "single_blocks.37.modulation.lin.weight",
            ),
        ];
        for (kohya_key, expected) in cases {
            match map_lora_key(kohya_key).unwrap() {
                LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "kohya key {kohya_key}");
                }
                _ => panic!("expected Direct for kohya key {kohya_key}"),
            }
        }
    }

    #[test]
    fn map_kohya_unknown_leaves_returns_none() {
        // Text-encoder LoRAs (`lora_te_*`) and unrecognized leaves are
        // skipped (caller logs a warning) rather than panicking.
        assert!(map_lora_key("lora_te_text_model_layer_0_attn_q").is_none());
        assert!(map_lora_key("lora_unet_double_blocks_0_unknown_leaf").is_none());
        assert!(map_lora_key("lora_unet_single_blocks_0_norm_query").is_none());
        assert!(map_lora_key("lora_unet_unrelated_block_0_x").is_none());
    }

    /// Round-trip `LoraAdapter::load` against a Kohya-shaped safetensors
    /// fixture to prove the suffix matcher accepts `lora_down`/`lora_up`/
    /// `alpha` and pairs them up correctly. Synthetic shapes — the
    /// numbers don't have to match a real FLUX layer, just the down/up
    /// convention.
    #[test]
    fn load_accepts_kohya_lora_down_up_alpha() {
        use safetensors::tensor::TensorView;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kohya.safetensors");
        let layer = "lora_unet_double_blocks_0_img_attn_qkv";

        // (rank=2, in=4) for down, (out=6, rank=2) for up. f32 little-endian
        // raw bytes — `safetensors::serialize` round-trips them as F32
        // tensors that `candle_core::safetensors::load` then parses.
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
            (format!("{layer}.lora_down.weight"), down_view),
            (format!("{layer}.lora_up.weight"), up_view),
            (format!("{layer}.alpha"), alpha_view),
        ];
        safetensors::serialize_to_file(entries, &None, &path).expect("write safetensors");

        let adapter = LoraAdapter::load(&path).expect("kohya safetensors must load");
        assert_eq!(
            adapter.layers.len(),
            1,
            "lora_down/lora_up should be paired into one layer"
        );
        assert_eq!(adapter.rank, 2);
        let lora_layer = adapter.layers.get(layer).expect("layer present");
        assert_eq!(lora_layer.a.dims(), &[2, 4]);
        assert_eq!(lora_layer.b.dims(), &[6, 2]);
        assert_eq!(lora_layer.alpha, Some(16.0));
    }

    /// Regression: an `I64` alpha must be read, not silently dropped.
    ///
    /// The Wan 2.2 Lightning distills ship `.alpha` as `I64` (value 8 against
    /// rank 64). `to_scalar::<f32>()` errors on `I64` storage, and the old
    /// `if let Ok` swallowed it — the layer got `alpha: None`, the scale fell
    /// back to the caller's raw value, and the adapter applied 8x too strong
    /// with no diagnostic.
    #[test]
    fn load_reads_an_i64_alpha_rather_than_dropping_it() {
        use safetensors::tensor::TensorView;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("i64_alpha.safetensors");
        let layer = "lora_unet_double_blocks_0_img_attn_qkv";

        let down_bytes: Vec<u8> = (0..2 * 4).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let up_bytes: Vec<u8> = (0..6 * 2).flat_map(|i| (i as f32).to_le_bytes()).collect();
        // The shipped spelling: a rank-0 I64 scalar.
        let alpha_bytes = 8i64.to_le_bytes().to_vec();

        let entries: Vec<(String, TensorView)> = vec![
            (
                format!("{layer}.lora_down.weight"),
                TensorView::new(safetensors::Dtype::F32, vec![2, 4], &down_bytes).unwrap(),
            ),
            (
                format!("{layer}.lora_up.weight"),
                TensorView::new(safetensors::Dtype::F32, vec![6, 2], &up_bytes).unwrap(),
            ),
            (
                format!("{layer}.alpha"),
                TensorView::new(safetensors::Dtype::I64, vec![], &alpha_bytes).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(entries, &None, &path).expect("write safetensors");

        let adapter = LoraAdapter::load(&path).expect("I64-alpha safetensors must load");
        let lora_layer = adapter.layers.get(layer).expect("layer present");
        assert_eq!(
            lora_layer.alpha,
            Some(8.0),
            "an I64 alpha must survive the read"
        );
        // Rank 2 with alpha 8 means an effective scale of 4x the user's, not 1x.
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 1.0,
        }];
        let (patches, _) = build_patches(&specs);
        let patch = patches.values().next().expect("one patch").first().unwrap();
        assert!((patch.effective_scale - 4.0).abs() < 1e-9);
    }

    /// `read_alpha_scalar` must accept every spelling trainers emit and reject
    /// anything that is not one scalar.
    #[test]
    fn alpha_scalar_reads_are_dtype_and_shape_tolerant() {
        let device = Device::Cpu;
        for (label, tensor) in [
            ("f32 rank-0", Tensor::new(16f32, &device).unwrap()),
            ("f64 rank-0", Tensor::new(16f64, &device).unwrap()),
            (
                "f16 rank-0",
                Tensor::new(16f32, &device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap(),
            ),
            (
                "bf16 rank-0",
                Tensor::new(16f32, &device)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            ),
            ("i64 rank-0", Tensor::new(16i64, &device).unwrap()),
            ("u32 rank-0", Tensor::new(16u32, &device).unwrap()),
            ("f32 shape [1]", Tensor::new(&[16f32], &device).unwrap()),
            ("i64 shape [1]", Tensor::new(&[16i64], &device).unwrap()),
        ] {
            assert_eq!(read_alpha_scalar(&tensor), Some(16.0), "{label}");
        }
        // Not a scalar: refuse rather than silently taking the first element.
        let vector = Tensor::new(&[1f32, 2.0], &device).unwrap();
        assert_eq!(read_alpha_scalar(&vector), None);
    }

    #[test]
    fn build_patches_single_spec_matches_legacy_shape() {
        // Regression: a one-element specs slice must produce the same
        // patch shape as the original single-LoRA path, so existing
        // single-LoRA flows don't change behaviour.
        let a = synthetic_single_layer_adapter(1.0, 1.0);
        let specs = [LoraSpec {
            adapter: &a,
            scale: 0.75,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0);
        let stack = patches
            .get("double_blocks.0.img_attn.proj.weight")
            .expect("present");
        assert_eq!(stack.len(), 1);
    }

    // ── classify_lora_key — multi-format suffix matrix ──────────────────

    #[test]
    fn classify_lora_key_diffusers() {
        assert_eq!(
            classify_lora_key("x.lora_A.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    #[test]
    fn classify_lora_key_kohya() {
        assert_eq!(
            classify_lora_key("x.lora_down.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_up.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    /// OneTrainer's flat-naming scheme inverts the down/up positions
    /// in the dotted key compared to Kohya: the *down* matrix is at
    /// `lora_linear_layer.down.weight`, NOT `lora_linear_layer.up.weight`.
    /// Pin the mapping so a future refactor doesn't silently swap them.
    #[test]
    fn classify_lora_key_onetrainer_inverted_naming() {
        assert_eq!(
            classify_lora_key("x.lora_linear_layer.down.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_linear_layer.up.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    #[test]
    fn classify_lora_key_default_adapter_peft() {
        assert_eq!(
            classify_lora_key("x.lora_A.default.weight"),
            Some((LoraDirection::Down, "x"))
        );
        assert_eq!(
            classify_lora_key("x.lora_B.default.weight"),
            Some((LoraDirection::Up, "x"))
        );
    }

    /// Mochi-style trainer drops the trailing `.weight` segment from
    /// the up matrix only. Down stays as `.lora_A.weight`. The
    /// suffix-list ordering is load-bearing: `.lora_B` comes last so
    /// it doesn't shadow `.lora_B.weight` / `.lora_B.default.weight`.
    #[test]
    fn classify_lora_key_mochi_no_dot_weight() {
        assert_eq!(
            classify_lora_key("x.lora_B"),
            Some((LoraDirection::Up, "x"))
        );
        // Sanity: the down side keeps the canonical form.
        assert_eq!(
            classify_lora_key("x.lora_A.weight"),
            Some((LoraDirection::Down, "x"))
        );
    }

    #[test]
    fn classify_lora_key_unrelated_returns_none() {
        assert_eq!(classify_lora_key("x.weight"), None);
        assert_eq!(classify_lora_key("transformer.embed.weight"), None);
        assert_eq!(classify_lora_key("alpha"), None);
        // `.alpha` scalars are also not LoRA pair tensors — the
        // loader handles them on its own branch.
        assert_eq!(classify_lora_key("layer.alpha"), None);
    }

    /// Stems must come back unmodified — the loader pairs A/B by
    /// stem equality, so any silent normalisation here would break
    /// adapter loading on long module paths.
    #[test]
    fn classify_lora_key_preserves_dotted_stem() {
        assert_eq!(
            classify_lora_key("transformer.transformer_blocks.5.attn.to_q.lora_A.weight"),
            Some((
                LoraDirection::Down,
                "transformer.transformer_blocks.5.attn.to_q",
            ))
        );
    }

    // ── parsed_lora_cache — hit + mtime invalidate ──────────────────────

    /// Build the same fixture as `load_accepts_kohya_lora_down_up_alpha`
    /// but reusable so the cache tests can also drop a real safetensors
    /// file at a path of their choice. Returns a path the caller owns.
    fn write_synthetic_kohya_safetensors(path: &Path) {
        use safetensors::tensor::TensorView;
        let layer = "lora_unet_double_blocks_0_img_attn_qkv";

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
            (format!("{layer}.lora_down.weight"), down_view),
            (format!("{layer}.lora_up.weight"), up_view),
            (format!("{layer}.alpha"), alpha_view),
        ];
        safetensors::serialize_to_file(entries, &None, path).expect("write safetensors");
    }

    /// Loading the same path twice must hand back the exact same
    /// `Arc<LoraAdapter>` — `Arc::ptr_eq` is the strongest possible
    /// hit-test (no parsed-from-disk twin can ever satisfy it).
    #[test]
    fn parsed_lora_cache_is_a_hit_on_second_load() {
        let _cache_test_guard = lock_parsed_lora_cache_tests();
        clear_parsed_lora_cache_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("hit.safetensors");
        write_synthetic_kohya_safetensors(&path);

        let first = get_or_load_adapter(&path).expect("first load");
        let second = get_or_load_adapter(&path).expect("second load");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second load must return the same Arc — proof the cache hit, no re-parse"
        );
    }

    /// When the file mtime changes (e.g. the user re-trains and saves
    /// over the same path), the cache key changes, so the second load
    /// must produce a NEW `Arc` even though the path string is
    /// identical.
    #[test]
    fn parsed_lora_cache_invalidates_on_mtime_change() {
        let _cache_test_guard = lock_parsed_lora_cache_tests();
        clear_parsed_lora_cache_for_test();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("invalidate.safetensors");
        write_synthetic_kohya_safetensors(&path);

        let first = get_or_load_adapter(&path).expect("first load");

        // Bump the file mtime by rewriting the file with fresh
        // contents. We sleep just long enough to exceed common
        // file-system mtime resolutions (HFS+ on older macOS rounds
        // to 1 s) so `metadata.modified()` actually changes.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write_synthetic_kohya_safetensors(&path);

        let second = get_or_load_adapter(&path).expect("second load");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "mtime change must produce a fresh Arc — proof the cache key invalidated"
        );
    }
}
