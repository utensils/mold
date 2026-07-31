//! SD1.5 LoRA support — diffusers + Kohya/sd-scripts naming, BF16/FP16 path.
//!
//! Mirrors `crate::sdxl::lora` (lifted on 2026-05-10) but adapted to SD1.5's
//! UNet topology. SD1.5 and SDXL both use candle's
//! `stable_diffusion::unet_2d::UNet2DConditionModel`, and the Kohya / diffusers
//! key conventions are identical between the two — the only structural delta
//! is **block count** (SD1.5 has 4 down / 4 up blocks, indices 0..=3, with
//! attention only in 0..=2; SDXL has 3 / 3, indices 0..=2, with attention only
//! in 1..=2). The Kohya helpers parse the block index dynamically from the
//! flattened key, so the same routing logic works for either family.
//!
//! Like SDXL, SD1.5 has **no fused QKV** (`attn1`/`attn2` carry separate
//! `to_q` / `to_k` / `to_v` / `to_out.0` linear weights), and **no quantized
//! transformer path** in mold today, so this module ships only a BF16/FP16
//! `LoraBackend` — no `gguf_lora_var_builder`.
//!
//! Two LoRA naming conventions ship in the wild SD1.5 ecosystem:
//!
//! 1. **Kohya / sd-scripts** (the majority of Civitai SD1.5 LoRAs):
//!    `lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight`.
//!    The prefix `lora_unet_` is stripped, then the underscore-flattened module
//!    path is parsed leaf-first.
//!
//! 2. **Diffusers / PEFT canonical**:
//!    `down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.weight`
//!    (often with a leading `unet.` or `transformer.` prefix).
//!
//! Both shapes route through `map_sd15_lora_key(stem)` which emits one or more
//! `Sd15LoraTarget`s pointing at candle UNet keys.
//!
//! # Pipeline integration
//!
//! `SD15Engine` carries a `pending_loras: Vec<LoraWeight>` set per-request from
//! `generate()`. `build_unet_for_strategy` branches on `has_lora`: when set,
//! the underlying mmap'd safetensors (or `SingleFileBackend` for Civitai
//! single-file checkpoints) is wrapped with a `Sd15LoraBackend` that
//! intercepts every `vb.get()` and merges `W' = W + scale·(B @ A)` in F32
//! before casting to the model dtype.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::flux::lora::{get_or_load_adapter, LoraAdapter, LoraDeltaCache};
use crate::progress::ProgressReporter;

// ---------------------------------------------------------------------------
// Public path-hash helper — seed `Sd15LoraSpec`s with a stable per-file id so
// the delta cache can disambiguate adapters in a multi-LoRA stack.
// ---------------------------------------------------------------------------

/// Stable hash of a LoRA file path. Independent copy of the FLUX/SDXL helper
/// of the same name so this module doesn't reach into other-family internals.
pub(crate) fn lora_path_hash(path: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Target descriptors
// ---------------------------------------------------------------------------

/// How a SD1.5 LoRA layer's `B @ A * scale` delta lands on a candle tensor.
///
/// SD1.5 has no fused QKV / single-block / linear1-style fused tensors anywhere
/// in candle's UNet, so `Direct` covers every observed leaf. The variant is
/// kept as an enum (rather than a bare string) for parity with the SDXL/FLUX
/// shape and to leave headroom for any future SD1.5 inpainting-specific
/// variant that might introduce a fused weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Sd15LoraTarget {
    /// Apply the entire delta to the entire tensor (additive merge).
    Direct { candle_key: String },
}

impl Sd15LoraTarget {
    fn candle_key(&self) -> &str {
        match self {
            Self::Direct { candle_key } => candle_key,
        }
    }
}

// ---------------------------------------------------------------------------
// Key mapping: LoRA stem → SD1.5 UNet candle target
// ---------------------------------------------------------------------------

/// Strip the diffusers / PEFT prefixes that wrap real module paths in SD1.5
/// LoRAs. We accept (in order): `transformer.`, `unet.`, `model.diffusion_model.`,
/// `diffusion_model.`. The `lora_unet_` Kohya prefix is handled by the dedicated
/// `map_kohya_sd15_key` arm.
fn strip_diffusers_prefixes(stem: &str) -> &str {
    let s = stem.strip_prefix("transformer.").unwrap_or(stem);
    let s = s.strip_prefix("unet.").unwrap_or(s);
    let s = s.strip_prefix("model.diffusion_model.").unwrap_or(s);
    s.strip_prefix("diffusion_model.").unwrap_or(s)
}

/// Map a LoRA layer stem (with the `.lora_A`/`.lora_down`/etc. suffix already
/// stripped) onto a SD1.5 UNet candle target. Returns an empty `Vec` for
/// unrecognised keys — the caller logs a warning and skips them.
///
/// Recognised input shapes:
/// - Kohya / sd-scripts (`cv:*`-style): `lora_unet_<flattened_module_path>`
/// - Diffusers / PEFT canonical: optional `(transformer|unet|diffusion_model).`
///   prefix, then `(down_blocks|mid_block|up_blocks).<idx>.<rest>`.
pub(crate) fn map_sd15_lora_key(raw_stem: &str) -> Vec<Sd15LoraTarget> {
    if let Some(rest) = raw_stem.strip_prefix("lora_unet_") {
        return match map_kohya_sd15_key(rest) {
            Some(candle_key) => vec![Sd15LoraTarget::Direct { candle_key }],
            None => Vec::new(),
        };
    }
    let stem = strip_diffusers_prefixes(raw_stem);
    if stem.starts_with("down_blocks.")
        || stem.starts_with("mid_block.")
        || stem.starts_with("up_blocks.")
    {
        if let Some(candle_key) = map_diffusers_path(stem) {
            return vec![Sd15LoraTarget::Direct { candle_key }];
        }
    }
    Vec::new()
}

/// Map a diffusers-form SD1.5 UNet module path (e.g.
/// `down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q`) to the matching
/// candle key (which appends `.weight`).
///
/// Accepts the same attention / FF / resnet leaf set as SDXL — SD1.5's UNet is
/// the same `UNet2DConditionModel` candle type and the leaves match 1:1.
fn map_diffusers_path(stem: &str) -> Option<String> {
    // The diffusers path is dotted and matches the candle tensor name exactly;
    // we just need to validate that the leaf is a known LoRA target so we
    // don't pass through `norm1`/`norm2`/etc. by accident.
    let known_leaves = [
        ".attn1.to_q",
        ".attn1.to_k",
        ".attn1.to_v",
        ".attn1.to_out.0",
        ".attn2.to_q",
        ".attn2.to_k",
        ".attn2.to_v",
        ".attn2.to_out.0",
        ".ff.net.0.proj",
        ".ff.net.2",
        ".proj_in",
        ".proj_out",
        ".time_emb_proj",
        // resnet conv leaves — present in some full-fine-tune LoRAs.
        ".conv1",
        ".conv2",
        ".conv_shortcut",
        // up/downsamplers — rare but observed.
        ".downsamplers.0.conv",
        ".upsamplers.0.conv",
    ];
    if known_leaves.iter().any(|leaf| stem.ends_with(leaf)) {
        Some(format!("{stem}.weight"))
    } else {
        None
    }
}

/// Map the Kohya / sd-scripts flattened form (with the `lora_unet_` prefix
/// already stripped) to a SD1.5 candle key. The flattening turns every `.` in
/// the original module path into `_`, so we walk it from the front:
///
/// 1. Block kind: `down_blocks_{i}_`, `mid_block_`, `up_blocks_{i}_`.
/// 2. Inner kind: `attentions_{j}_...`, `resnets_{j}_...`, or for downsamplers
///    `downsamplers_0_conv`, `upsamplers_0_conv`.
///
/// Returns `None` for any leaf the SD1.5 UNet can't host (text-encoder LoRAs
/// (`lora_te_*`), unrecognised sub-trees, etc.) — the caller logs and skips.
fn map_kohya_sd15_key(rest: &str) -> Option<String> {
    if let Some(after) = rest.strip_prefix("down_blocks_") {
        return map_kohya_indexed_block(after, "down_blocks");
    }
    if let Some(after) = rest.strip_prefix("mid_block_") {
        return map_kohya_mid_block(after);
    }
    if let Some(after) = rest.strip_prefix("up_blocks_") {
        return map_kohya_indexed_block(after, "up_blocks");
    }
    // `conv_in`, `conv_out`, `time_embedding_linear_*` are theoretically
    // possible but never appear in observed Civitai SD1.5 LoRAs — left out
    // intentionally so an unexpected match doesn't silently mis-merge.
    None
}

/// Map an indexed-block Kohya tail (the part after `down_blocks_` or
/// `up_blocks_`) to a candle key. `block_kind` is `"down_blocks"` or
/// `"up_blocks"`.
fn map_kohya_indexed_block(rest: &str, block_kind: &str) -> Option<String> {
    // `rest` begins with `{i}_<sub>...`. Pull the block index.
    let (block_idx, after_idx) = rest.split_once('_')?;
    block_idx.parse::<usize>().ok()?;

    if let Some(rest) = after_idx.strip_prefix("attentions_") {
        return map_kohya_attentions(rest, block_kind, block_idx);
    }
    if let Some(rest) = after_idx.strip_prefix("resnets_") {
        return map_kohya_resnets(rest, block_kind, block_idx);
    }
    if let Some(rest) = after_idx.strip_prefix("downsamplers_") {
        // `0_conv` → `down_blocks.{i}.downsamplers.0.conv.weight`.
        let candle = format!(
            "{block_kind}.{block_idx}.downsamplers.{}",
            rest.replace('_', ".")
        );
        return Some(format!("{candle}.weight"));
    }
    if let Some(rest) = after_idx.strip_prefix("upsamplers_") {
        let candle = format!(
            "{block_kind}.{block_idx}.upsamplers.{}",
            rest.replace('_', ".")
        );
        return Some(format!("{candle}.weight"));
    }
    None
}

fn map_kohya_mid_block(rest: &str) -> Option<String> {
    if let Some(rest) = rest.strip_prefix("attentions_") {
        return map_kohya_attentions(rest, "mid_block", "");
    }
    if let Some(rest) = rest.strip_prefix("resnets_") {
        return map_kohya_resnets(rest, "mid_block", "");
    }
    None
}

/// `block_idx_str` is empty for `mid_block` and the numeric index otherwise.
/// The candle prefix is rebuilt as `{block_kind}.{idx}.` (or `{block_kind}.`
/// when `idx` is empty).
fn block_prefix(block_kind: &str, block_idx_str: &str) -> String {
    if block_idx_str.is_empty() {
        format!("{block_kind}.")
    } else {
        format!("{block_kind}.{block_idx_str}.")
    }
}

/// Map an attentions tail like `1_transformer_blocks_0_attn1_to_q` to its
/// candle key. The attention block index is the first integer; the rest is
/// `transformer_blocks_{k}_<leaf>` OR a top-level attention leaf
/// (`proj_in`, `proj_out`).
fn map_kohya_attentions(rest: &str, block_kind: &str, block_idx_str: &str) -> Option<String> {
    let (attn_idx, after_attn) = rest.split_once('_')?;
    attn_idx.parse::<usize>().ok()?;
    let prefix = block_prefix(block_kind, block_idx_str);

    if let Some(leaf) = after_attn.strip_prefix("transformer_blocks_") {
        return map_kohya_transformer_block(leaf, &prefix, attn_idx);
    }
    let candle_leaf = match after_attn {
        "proj_in" => Some("proj_in"),
        "proj_out" => Some("proj_out"),
        _ => None,
    }?;
    Some(format!(
        "{prefix}attentions.{attn_idx}.{candle_leaf}.weight"
    ))
}

/// Map a transformer-block tail `{k}_<leaf>` to a candle key, where `<leaf>`
/// is one of the SD1.5 attention/FF leaves.
fn map_kohya_transformer_block(rest: &str, prefix: &str, attn_idx: &str) -> Option<String> {
    let (tb_idx, leaf) = rest.split_once('_')?;
    tb_idx.parse::<usize>().ok()?;

    let candle_leaf = match leaf {
        // Self-attention (image-image)
        "attn1_to_q" => "attn1.to_q",
        "attn1_to_k" => "attn1.to_k",
        "attn1_to_v" => "attn1.to_v",
        "attn1_to_out_0" => "attn1.to_out.0",
        // Cross-attention (image-text)
        "attn2_to_q" => "attn2.to_q",
        "attn2_to_k" => "attn2.to_k",
        "attn2_to_v" => "attn2.to_v",
        "attn2_to_out_0" => "attn2.to_out.0",
        // Feed-forward (GeGLU project + out-projection)
        "ff_net_0_proj" => "ff.net.0.proj",
        "ff_net_2" => "ff.net.2",
        _ => return None,
    };
    Some(format!(
        "{prefix}attentions.{attn_idx}.transformer_blocks.{tb_idx}.{candle_leaf}.weight"
    ))
}

/// Map a resnets tail like `0_time_emb_proj` to its candle key.
fn map_kohya_resnets(rest: &str, block_kind: &str, block_idx_str: &str) -> Option<String> {
    let (resnet_idx, leaf) = rest.split_once('_')?;
    resnet_idx.parse::<usize>().ok()?;
    let prefix = block_prefix(block_kind, block_idx_str);

    let candle_leaf = match leaf {
        "time_emb_proj" => "time_emb_proj",
        "conv1" => "conv1",
        "conv2" => "conv2",
        "conv_shortcut" => "conv_shortcut",
        _ => return None,
    };
    Some(format!("{prefix}resnets.{resnet_idx}.{candle_leaf}.weight"))
}

// ---------------------------------------------------------------------------
// Patch building — turn every (adapter, layer) pair into per-tensor patches
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Sd15LoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: Sd15LoraTarget,
    lora_path_hash: u64,
}

/// A loaded LoRA + its scale + a stable hash of its file path. The hash is the
/// per-LoRA cache-key component so a multi-LoRA stack keeps each adapter's
/// delta independently cacheable.
pub(crate) struct Sd15LoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
    pub path_hash: u64,
}

fn build_patches(specs: &[Sd15LoraSpec<'_>]) -> (HashMap<String, Vec<Sd15LoraPatch>>, usize) {
    let mut patches: HashMap<String, Vec<Sd15LoraPatch>> = HashMap::new();
    let mut skipped = 0usize;
    for spec in specs {
        for (lora_stem, layer) in &spec.adapter.layers {
            let targets = map_sd15_lora_key(lora_stem);
            if targets.is_empty() {
                tracing::warn!(
                    key = lora_stem.as_str(),
                    "unrecognized SD1.5 LoRA key, skipping"
                );
                skipped += 1;
                continue;
            }
            let layer_rank = layer.a.dims()[0] as f64;
            let effective_scale = match layer.alpha {
                Some(alpha) => spec.scale * alpha / layer_rank,
                None => spec.scale,
            };
            for target in targets {
                let candle_key = target.candle_key().to_string();
                patches.entry(candle_key).or_default().push(Sd15LoraPatch {
                    a: layer.a.clone(),
                    b: layer.b.clone(),
                    effective_scale,
                    target,
                    lora_path_hash: spec.path_hash,
                });
            }
        }
    }
    (patches, skipped)
}

// ---------------------------------------------------------------------------
// Delta computation + apply
// ---------------------------------------------------------------------------

#[derive(Hash, Eq, PartialEq, Clone)]
struct DeltaCacheKey {
    tensor_name: String,
    patch_index: usize,
    lora_path_hash: u64,
    scale_bits: u64,
}

fn compute_delta(patch: &Sd15LoraPatch, target_dev: &Device) -> candle_core::Result<Tensor> {
    let a = patch.a.to_dtype(DType::F32)?.to_device(target_dev)?;
    let b = patch.b.to_dtype(DType::F32)?.to_device(target_dev)?;
    let computed = b.matmul(&a)?;
    &computed * patch.effective_scale
}

/// Apply a `Sd15LoraPatch` to a base tensor in F32 working precision. The
/// caller handles dtype casts; SD1.5 has no fused-slice path, so this is just
/// a tensor add when shapes agree, with a best-effort reshape for rare
/// resnet-targeting LoRAs that store conv weights as Linear `(out, in)` while
/// the candle tensor is `(out, in, 1, 1)`.
fn apply_patch_f32(
    base_f32: &Tensor,
    delta: &Tensor,
    patch: &Sd15LoraPatch,
) -> candle_core::Result<Tensor> {
    match &patch.target {
        Sd15LoraTarget::Direct { .. } => {
            if base_f32.dims() == delta.dims() {
                return base_f32 + delta;
            }
            if base_f32.rank() == 4 && delta.rank() == 2 {
                let (b_rows, b_cols) = (base_f32.dim(0)?, base_f32.dim(1)?);
                if delta.dim(0)? == b_rows && delta.dim(1)? == b_cols {
                    let reshaped = delta.reshape(base_f32.shape())?;
                    return base_f32 + &reshaped;
                }
            }
            tracing::warn!(
                base_dims = ?base_f32.dims(),
                delta_dims = ?delta.dims(),
                "SD1.5 LoRA shape mismatch, skipping merge"
            );
            Ok(base_f32.clone())
        }
    }
}

// ---------------------------------------------------------------------------
// `Sd15LoraBackend` — wraps a `SimpleBackend` and merges LoRAs at vb.get()
// ---------------------------------------------------------------------------

struct Sd15LoraBackend {
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    patches: HashMap<String, Vec<Sd15LoraPatch>>,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

impl Sd15LoraBackend {
    fn merge_into(
        &self,
        name: &str,
        tensor: Tensor,
        target_dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let Some(layer_patches) = self.patches.get(name) else {
            return Ok(tensor);
        };
        let mut merged = tensor.to_dtype(DType::F32)?;
        for (patch_idx, patch) in layer_patches.iter().enumerate() {
            // Per-patch delta-cache key. We don't share storage with the FLUX
            // cache (different key struct), so the `delta_cache` argument is
            // accepted for API parity but currently unused on this backend.
            let _ = DeltaCacheKey {
                tensor_name: name.to_string(),
                patch_index: patch_idx,
                lora_path_hash: patch.lora_path_hash,
                scale_bits: patch.effective_scale.to_bits(),
            };
            let _ = &self.delta_cache;

            let delta = compute_delta(patch, dev)?;
            merged = apply_patch_f32(&merged, &delta, patch)?;
        }
        merged.to_dtype(target_dtype)
    }
}

impl candle_nn::var_builder::SimpleBackend for Sd15LoraBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.inner.get(s, name, h, dtype, dev)?;
        self.merge_into(name, tensor, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.inner.get_unchecked(name, dtype, dev)?;
        self.merge_into(name, tensor, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Wrap an existing `SimpleBackend` (typically a `MmapedSafetensors` or the
/// SD1.5 `SingleFileBackend`) so its `vb.get()` calls return merged-LoRA
/// tensors. The wrapper applies LoRA deltas in F32 and casts back to the
/// requested dtype, so the SD1.5 UNet constructor (`UNet2DConditionModel::new`)
/// loads merged weights without knowing the LoRA exists.
pub(crate) fn wrap_backend_with_lora(
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    specs: &[Sd15LoraSpec<'_>],
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<Box<dyn candle_nn::var_builder::SimpleBackend>> {
    if specs.is_empty() {
        bail!("wrap_backend_with_lora called with no LoraSpecs");
    }
    let (patches, skipped) = build_patches(specs);
    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA (SD1.5): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    Ok(Box::new(Sd15LoraBackend {
        inner,
        patches,
        delta_cache,
    }))
}

/// Load LoRA adapter files into `Arc<LoraAdapter>`s. Reuses the FLUX
/// `get_or_load_adapter` parsed-LoRA cache so adapter files mmap once across
/// requests.
pub(crate) fn load_lora_adapters(
    loras: &[mold_core::LoraWeight],
    progress: &ProgressReporter,
) -> Result<Vec<Arc<LoraAdapter>>> {
    loras
        .iter()
        .map(|w| {
            progress.info("Loading SD1.5 LoRA adapter");
            let adapter = get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "SD1.5 LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect()
}

/// Resolve the effective LoRA list for a request. Mirrors the FLUX / SDXL /
/// Flux.2 helpers: `loras` (plural) wins over `lora` (singular) when both are
/// set, and zero-scale entries are filtered out so they don't trigger a
/// transformer rebuild for nothing.
pub(crate) fn effective_sd15_loras(req: &mold_core::GenerateRequest) -> Vec<mold_core::LoraWeight> {
    /// Threshold below which a LoRA scale is treated as off.
    const ZERO_SCALE_EPS: f64 = 1e-8;

    let raw: Vec<mold_core::LoraWeight> = if let Some(plural) = &req.loras {
        if !plural.is_empty() {
            plural.clone()
        } else {
            req.lora.iter().cloned().collect()
        }
    } else {
        req.lora.iter().cloned().collect()
    };
    raw.into_iter()
        .filter(|w| {
            let keep = w.scale.abs() > ZERO_SCALE_EPS;
            if !keep {
                tracing::debug!(
                    path = w.path.as_str(),
                    scale = w.scale,
                    "dropping zero-scale SD1.5 LoRA"
                );
            }
            keep
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::lora::{classify_lora_key, LoraDirection, LoraLayer};
    use safetensors::tensor::TensorView;

    // ── map_sd15_lora_key — Kohya leaf coverage ─────────────────────────

    /// Every Kohya attention leaf the wild SD1.5 LoRAs ship under a
    /// `down_blocks` attention. Pin them so a refactor can't silently
    /// rename one. SD1.5 has 4 down blocks (0..=3); blocks 0..=2 carry
    /// attention.
    #[test]
    fn kohya_down_blocks_attention_leaves() {
        let cases = [
            // Self-attention (attn1)
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
            ),
            // Cross-attention (attn2)
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
            ),
            // Feed-forward
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj",
                "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
            ),
            (
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_2",
                "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.weight",
            ),
        ];
        for (kohya, expected) in cases {
            let targets = map_sd15_lora_key(kohya);
            assert_eq!(targets.len(), 1, "exactly one target for {kohya}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={kohya}");
                }
            }
        }
    }

    /// SD1.5 has up_blocks at indices 0..=3; only 1..=3 carry attention
    /// (block 0 is the bottom of the U-Net and only has resnets).
    #[test]
    fn kohya_up_blocks_attention_leaves() {
        let cases = [
            (
                "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
                "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            ),
            (
                "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0",
                "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.weight",
            ),
            (
                "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj",
                "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
            ),
            (
                "lora_unet_up_blocks_2_attentions_1_transformer_blocks_0_ff_net_2",
                "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.weight",
            ),
        ];
        for (kohya, expected) in cases {
            let targets = map_sd15_lora_key(kohya);
            assert_eq!(targets.len(), 1, "leaf={kohya}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected);
                }
            }
        }
    }

    /// SD1.5 mid_block has a single transformer-block layer (`layers_per_block=2`
    /// applies to the down/up blocks, not the bottleneck) — same shape as SDXL
    /// at the LoRA mapping level (no numeric block index).
    #[test]
    fn kohya_mid_block_attention_leaves() {
        let cases = [
            (
                "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q",
                "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            ),
            (
                "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_v",
                "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight",
            ),
            (
                "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_2",
                "mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight",
            ),
        ];
        for (kohya, expected) in cases {
            let targets = map_sd15_lora_key(kohya);
            assert_eq!(targets.len(), 1, "leaf={kohya}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected);
                }
            }
        }
    }

    /// SpatialTransformer-level `proj_in` / `proj_out` LoRAs.
    #[test]
    fn kohya_attentions_proj_in_proj_out() {
        for (kohya, expected) in [
            (
                "lora_unet_down_blocks_1_attentions_0_proj_in",
                "down_blocks.1.attentions.0.proj_in.weight",
            ),
            (
                "lora_unet_up_blocks_1_attentions_1_proj_out",
                "up_blocks.1.attentions.1.proj_out.weight",
            ),
            (
                "lora_unet_mid_block_attentions_0_proj_in",
                "mid_block.attentions.0.proj_in.weight",
            ),
        ] {
            let targets = map_sd15_lora_key(kohya);
            assert_eq!(targets.len(), 1, "leaf={kohya}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected);
                }
            }
        }
    }

    /// Resnet `time_emb_proj`, `conv1`, `conv2`, `conv_shortcut` LoRAs.
    /// Block 3 of SD1.5's down_blocks has only resnets (no attentions),
    /// so it's the natural target for a resnet-only LoRA.
    #[test]
    fn kohya_resnet_leaves() {
        for (kohya, expected) in [
            (
                "lora_unet_down_blocks_3_resnets_0_time_emb_proj",
                "down_blocks.3.resnets.0.time_emb_proj.weight",
            ),
            (
                "lora_unet_down_blocks_3_resnets_0_conv1",
                "down_blocks.3.resnets.0.conv1.weight",
            ),
            (
                "lora_unet_down_blocks_3_resnets_0_conv2",
                "down_blocks.3.resnets.0.conv2.weight",
            ),
            (
                "lora_unet_down_blocks_3_resnets_0_conv_shortcut",
                "down_blocks.3.resnets.0.conv_shortcut.weight",
            ),
            (
                "lora_unet_up_blocks_0_resnets_1_time_emb_proj",
                "up_blocks.0.resnets.1.time_emb_proj.weight",
            ),
            (
                "lora_unet_mid_block_resnets_0_time_emb_proj",
                "mid_block.resnets.0.time_emb_proj.weight",
            ),
        ] {
            let targets = map_sd15_lora_key(kohya);
            assert_eq!(targets.len(), 1, "leaf={kohya}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected);
                }
            }
        }
    }

    /// Down/upsampler conv leaves — the per-block residual paths.
    #[test]
    fn kohya_downsampler_upsampler_conv() {
        for (kohya, expected) in [
            (
                "lora_unet_down_blocks_0_downsamplers_0_conv",
                "down_blocks.0.downsamplers.0.conv.weight",
            ),
            (
                "lora_unet_up_blocks_2_upsamplers_0_conv",
                "up_blocks.2.upsamplers.0.conv.weight",
            ),
        ] {
            let targets = map_sd15_lora_key(kohya);
            assert_eq!(targets.len(), 1, "leaf={kohya}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected);
                }
            }
        }
    }

    #[test]
    fn unknown_kohya_leaves_skipped_silently() {
        // Text-encoder LoRAs (`lora_te_*` for SD1.5 — single CLIP-L) and
        // unrecognised sub-trees must not produce a target.
        for key in [
            "lora_te_text_model_encoder_layers_0_self_attn_q_proj",
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj",
            "lora_unet_down_blocks_0_attentions_0_norm",
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_unknown",
            "lora_unet_unknown_leaf",
            "lora_unet_down_blocks_X_attentions_0_proj_in", // bad idx
        ] {
            assert!(
                map_sd15_lora_key(key).is_empty(),
                "expected skip for {key}, got targets"
            );
        }
    }

    // ── Diffusers / PEFT canonical paths ────────────────────────────────

    /// PEFT canonical (`down_blocks.X.Y...`) with optional `unet.` /
    /// `transformer.` / `diffusion_model.` prefix.
    #[test]
    fn diffusers_canonical_attention_leaves() {
        let cases = [
            (
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight",
            ),
            (
                "unet.down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight",
            ),
            (
                "transformer.up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0",
                "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.weight",
            ),
            (
                "diffusion_model.mid_block.attentions.0.transformer_blocks.0.ff.net.2",
                "mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight",
            ),
            (
                "model.diffusion_model.up_blocks.1.attentions.0.proj_in",
                "up_blocks.1.attentions.0.proj_in.weight",
            ),
        ];
        for (peft, expected) in cases {
            let targets = map_sd15_lora_key(peft);
            assert_eq!(targets.len(), 1, "leaf={peft}");
            match &targets[0] {
                Sd15LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected);
                }
            }
        }
    }

    #[test]
    fn diffusers_unknown_leaf_returns_empty() {
        // `norm1` / `norm2` / `norm3` are LayerNorm layers — never LoRA targets.
        assert!(
            map_sd15_lora_key("down_blocks.2.attentions.1.transformer_blocks.0.norm1").is_empty()
        );
        // Top-level UNet weights like `conv_in` aren't LoRA targets either.
        assert!(map_sd15_lora_key("conv_in").is_empty());
        assert!(map_sd15_lora_key("totally.unknown.key").is_empty());
    }

    // ── build_patches / wrap_backend_with_lora — end-to-end math ────────

    /// Synthetic adapter targeting one SD1.5 Kohya leaf. (rank=2, in=4) for
    /// A and (out=6, rank=2) for B.
    fn synthetic_kohya_adapter(layer: &str, fill_a: f32, fill_b: f32) -> LoraAdapter {
        let dev = Device::Cpu;
        let a = Tensor::full(fill_a, (2, 4), &dev).unwrap();
        let b = Tensor::full(fill_b, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(layer.to_string(), LoraLayer { a, b, alpha: None });
        LoraAdapter { layers, rank: 2 }
    }

    #[test]
    fn build_patches_routes_kohya_leaf_to_single_target() {
        let adapter = synthetic_kohya_adapter(
            "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q",
            1.0,
            1.0,
        );
        let specs = [Sd15LoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0xC0FFEE,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0, "leaf must map to a real candle target");
        assert_eq!(patches.len(), 1);
        let key = "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight";
        let bucket = patches.get(key).expect("target tensor must be patched");
        assert_eq!(bucket.len(), 1);
        assert!((bucket[0].effective_scale - 0.5).abs() < 1e-9);
    }

    #[test]
    fn build_patches_alpha_normalises_scale() {
        let mut adapter = synthetic_kohya_adapter(
            "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q",
            1.0,
            1.0,
        );
        // alpha=4, rank=2 → effective scale = user(0.5) * 4 / 2 = 1.0.
        if let Some(layer) = adapter
            .layers
            .get_mut("lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q")
        {
            layer.alpha = Some(4.0);
        }
        let specs = [Sd15LoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0,
        }];
        let (patches, _) = build_patches(&specs);
        let key = "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight";
        let s = patches[key][0].effective_scale;
        assert!(
            (s - 1.0).abs() < 1e-9,
            "expected user(0.5) * alpha(4) / rank(2) = 1.0, got {s}"
        );
    }

    #[test]
    fn build_patches_two_specs_stack_on_same_target() {
        let a1 = synthetic_kohya_adapter(
            "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q",
            1.0,
            1.0,
        );
        let a2 = synthetic_kohya_adapter(
            "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q",
            0.5,
            0.5,
        );
        let specs = [
            Sd15LoraSpec {
                adapter: &a1,
                scale: 1.0,
                path_hash: 0xAA,
            },
            Sd15LoraSpec {
                adapter: &a2,
                scale: 1.0,
                path_hash: 0xBB,
            },
        ];
        let (patches, _) = build_patches(&specs);
        let key = "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight";
        let bucket = &patches[key];
        assert_eq!(bucket.len(), 2, "stack must keep distinct patches");
        assert_eq!(bucket[0].lora_path_hash, 0xAA);
        assert_eq!(bucket[1].lora_path_hash, 0xBB);
    }

    fn write_synthetic_safetensors_with_data(
        path: &Path,
        entries: &[(String, Vec<usize>, Vec<f32>)],
    ) {
        let buffers: Vec<Vec<u8>> = entries
            .iter()
            .map(|(_, _, data)| {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for v in data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .zip(buffers.iter())
            .map(|((k, shape, _), buf)| {
                (
                    k.clone(),
                    TensorView::new(safetensors::Dtype::F32, shape.clone(), buf).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, path).expect("write safetensors");
    }

    /// Build a synthetic base tensor + a synthetic Kohya LoRA, then walk it
    /// through `wrap_backend_with_lora` and verify the merged tensor equals
    /// `base + scale·(B @ A)` everywhere.
    #[test]
    fn end_to_end_kohya_direct_merge_matches_math() {
        use crate::flux::lora::LoraAdapter;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base.safetensors");
        // Base tensor is (6, 4) of constant 1.0.
        let key = "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight";
        write_synthetic_safetensors_with_data(
            &path,
            &[(key.to_string(), vec![6, 4], vec![1.0; 24])],
        );

        // Adapter: B (6, 2) of 0.5, A (2, 4) of 1.0 → B@A = (6, 4) of 1.0.
        // With scale = 1.0 → merged = 1 + 1 = 2.0 everywhere.
        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 4), &dev).unwrap();
        let b = Tensor::full(0.5f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [Sd15LoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0xFEED,
        }];

        let st =
            unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path]).expect("mmap") };
        struct MmapBackend {
            st: candle_core::safetensors::MmapedSafetensors,
        }
        impl candle_nn::var_builder::SimpleBackend for MmapBackend {
            fn get(
                &self,
                _s: candle_core::Shape,
                name: &str,
                _h: candle_nn::Init,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                let t = self.st.load(name, dev)?;
                if t.dtype() != dtype {
                    t.to_dtype(dtype)
                } else {
                    Ok(t)
                }
            }
            fn get_unchecked(
                &self,
                name: &str,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                let t = self.st.load(name, dev)?;
                if t.dtype() != dtype {
                    t.to_dtype(dtype)
                } else {
                    Ok(t)
                }
            }
            fn contains_tensor(&self, name: &str) -> bool {
                self.st.get(name).is_ok()
            }
        }
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(MmapBackend { st });

        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");

        let merged = wrapped.get_unchecked(key, DType::F32, &dev).expect("get");
        let vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
            "expected 2.0 (= 1 + B@A·scale) everywhere, got {vals:?}"
        );
    }

    /// Diffusers-form LoRA must merge identically to the Kohya form.
    #[test]
    fn end_to_end_diffusers_direct_merge_matches_math() {
        use crate::flux::lora::LoraAdapter;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base_diffusers.safetensors");
        let key = "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight";
        write_synthetic_safetensors_with_data(
            &path,
            &[(key.to_string(), vec![4, 8], vec![0.0; 32])],
        );

        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(1.0f32, (4, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        // Diffusers stem (with `transformer.` prefix to exercise strip).
        layers.insert(
            "transformer.up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [Sd15LoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0xBEEF,
        }];

        let st =
            unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path]).expect("mmap") };
        struct MmapBackend {
            st: candle_core::safetensors::MmapedSafetensors,
        }
        impl candle_nn::var_builder::SimpleBackend for MmapBackend {
            fn get(
                &self,
                _s: candle_core::Shape,
                name: &str,
                _h: candle_nn::Init,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                let t = self.st.load(name, dev)?;
                if t.dtype() != dtype {
                    t.to_dtype(dtype)
                } else {
                    Ok(t)
                }
            }
            fn get_unchecked(
                &self,
                name: &str,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                let t = self.st.load(name, dev)?;
                if t.dtype() != dtype {
                    t.to_dtype(dtype)
                } else {
                    Ok(t)
                }
            }
            fn contains_tensor(&self, name: &str) -> bool {
                self.st.get(name).is_ok()
            }
        }
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(MmapBackend { st });
        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");

        // B@A = (4,2)·(2,8) of ones → entry value 2.0; merged = 0 + 2 = 2.0.
        let merged = wrapped.get_unchecked(key, DType::F32, &dev).expect("get");
        let vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
            "expected 2.0 (= 0 + 2.0 from B@A·scale), got {vals:?}"
        );
    }

    // ── classify_lora_key — both Kohya and PEFT canonical accepted ──────

    #[test]
    fn classify_kohya_and_peft_suffixes_for_sd15_layers() {
        // Kohya: down/up = A/B
        assert_eq!(
            classify_lora_key(
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight"
            ),
            Some((
                LoraDirection::Down,
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q"
            ))
        );
        assert_eq!(
            classify_lora_key(
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight"
            ),
            Some((
                LoraDirection::Up,
                "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q"
            ))
        );
        // PEFT canonical
        assert_eq!(
            classify_lora_key(
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.weight"
            ),
            Some((
                LoraDirection::Down,
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q"
            ))
        );
        assert_eq!(
            classify_lora_key(
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.weight"
            ),
            Some((
                LoraDirection::Up,
                "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q"
            ))
        );
    }

    #[test]
    fn lora_path_hash_is_deterministic() {
        let h1 = lora_path_hash("/a/b/c.safetensors");
        let h2 = lora_path_hash("/a/b/c.safetensors");
        let h3 = lora_path_hash("/a/b/d.safetensors");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    // ── effective_sd15_loras — request normalisation ────────────────────

    fn req_with_loras(
        lora: Option<mold_core::LoraWeight>,
        loras: Option<Vec<mold_core::LoraWeight>>,
    ) -> mold_core::GenerateRequest {
        mold_core::GenerateRequest {
            guidance_overrides: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "sd15".to_string(),
            width: 512,
            height: 512,
            steps: 30,
            guidance: 7.0,
            seed: Some(1),
            batch_size: 1,
            output_format: None,
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        }
    }

    #[test]
    fn effective_loras_plural_wins_over_singular() {
        let plural = vec![
            mold_core::LoraWeight {
                path: "/a.safetensors".into(),
                scale: 0.8,
            },
            mold_core::LoraWeight {
                path: "/b.safetensors".into(),
                scale: 0.4,
            },
        ];
        let req = req_with_loras(
            Some(mold_core::LoraWeight {
                path: "/legacy.safetensors".into(),
                scale: 1.0,
            }),
            Some(plural.clone()),
        );
        let resolved = effective_sd15_loras(&req);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].path, "/a.safetensors");
        assert_eq!(resolved[1].path, "/b.safetensors");
    }

    #[test]
    fn effective_loras_legacy_singular_falls_through() {
        let req = req_with_loras(
            Some(mold_core::LoraWeight {
                path: "/legacy.safetensors".into(),
                scale: 0.7,
            }),
            None,
        );
        let resolved = effective_sd15_loras(&req);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].path, "/legacy.safetensors");
    }

    #[test]
    fn effective_loras_drops_zero_scale_entries() {
        let req = req_with_loras(
            None,
            Some(vec![
                mold_core::LoraWeight {
                    path: "/active.safetensors".into(),
                    scale: 0.5,
                },
                mold_core::LoraWeight {
                    path: "/off.safetensors".into(),
                    scale: 0.0,
                },
            ]),
        );
        let resolved = effective_sd15_loras(&req);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].path, "/active.safetensors");
    }

    // ── LoraAdapter::load against a synthetic Kohya SD1.5 fixture ───────

    /// Pin that `LoraAdapter::load` (shared with FLUX/Flux.2/SDXL) accepts the
    /// Kohya SD1.5 key shape end-to-end: `*.lora_down.weight`, `*.lora_up.weight`,
    /// `*.alpha` round-trip through the safetensors parser into one paired
    /// layer.
    #[test]
    fn lora_adapter_load_accepts_kohya_sd15_layer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sd15_kohya.safetensors");
        let layer = "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q";

        // Synthetic shapes: down=(2,4), up=(6,2), alpha=16.
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

        let adapter = LoraAdapter::load(&path).expect("SD1.5 kohya safetensors must load");
        assert_eq!(adapter.layers.len(), 1);
        assert_eq!(adapter.rank, 2);
        let lora_layer = adapter.layers.get(layer).expect("paired layer present");
        assert_eq!(lora_layer.a.dims(), &[2, 4]);
        assert_eq!(lora_layer.b.dims(), &[6, 2]);
        assert_eq!(lora_layer.alpha, Some(16.0));
    }

    /// End-to-end: synthetic UNet config → small base tensor → small synthetic
    /// LoRA → wrap_backend_with_lora → forward (vb.get) → assert deterministic
    /// merged output. This is the SD1.5-counterpart of the SDXL E2E coverage,
    /// scaled down so the test fits the unit budget.
    #[test]
    fn end_to_end_synthetic_lora_against_unet_attn_leaf() {
        use crate::flux::lora::LoraAdapter;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("synth.safetensors");

        // Pick a representative SD1.5 attention leaf — block 0, attentions 0,
        // transformer block 0 attn1.to_q. Cross-attention dim is 768 in SD1.5
        // but we use a tiny shape (8,8) here to keep the test cheap; the
        // routing logic is shape-agnostic.
        let key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight";
        let base_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        write_synthetic_safetensors_with_data(
            &path,
            &[(key.to_string(), vec![8, 8], base_data.clone())],
        );

        // Adapter: A=(rank=4, in=8) of 0.1, B=(out=8, rank=4) of 0.1 →
        // B@A = (8,8) where each entry = 4 * 0.1 * 0.1 = 0.04.
        let dev = Device::Cpu;
        let a = Tensor::full(0.1f32, (4, 8), &dev).unwrap();
        let b = Tensor::full(0.1f32, (8, 4), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 4 };
        let specs = [Sd15LoraSpec {
            adapter: &adapter,
            scale: 2.0, // user-side scale → effective = 2.0 (no alpha)
            path_hash: 0xDEADBEEF,
        }];

        let st =
            unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path]).expect("mmap") };
        struct MmapBackend {
            st: candle_core::safetensors::MmapedSafetensors,
        }
        impl candle_nn::var_builder::SimpleBackend for MmapBackend {
            fn get(
                &self,
                _s: candle_core::Shape,
                name: &str,
                _h: candle_nn::Init,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                let t = self.st.load(name, dev)?;
                if t.dtype() != dtype {
                    t.to_dtype(dtype)
                } else {
                    Ok(t)
                }
            }
            fn get_unchecked(
                &self,
                name: &str,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                let t = self.st.load(name, dev)?;
                if t.dtype() != dtype {
                    t.to_dtype(dtype)
                } else {
                    Ok(t)
                }
            }
            fn contains_tensor(&self, name: &str) -> bool {
                self.st.get(name).is_ok()
            }
        }
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(MmapBackend { st });
        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");

        // Each entry should be base + 2.0 * 0.04 = base + 0.08.
        let merged = wrapped.get_unchecked(key, DType::F32, &dev).expect("get");
        let merged_vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (m, b)) in merged_vals.iter().zip(base_data.iter()).enumerate() {
            let expected = b + 0.08;
            assert!(
                (m - expected).abs() < 1e-5,
                "idx={i}: expected {expected}, got {m}"
            );
        }
    }

    /// `wrap_backend_with_lora` must error when called with no specs — empty
    /// stacks are dropped at the request layer (`effective_sd15_loras`), so
    /// hitting this guard means the caller forgot the gate.
    #[test]
    fn wrap_backend_with_lora_rejects_empty_specs() {
        struct StubBackend;
        impl candle_nn::var_builder::SimpleBackend for StubBackend {
            fn get(
                &self,
                _s: candle_core::Shape,
                _name: &str,
                _h: candle_nn::Init,
                _dtype: DType,
                _dev: &Device,
            ) -> candle_core::Result<Tensor> {
                unreachable!("stub")
            }
            fn get_unchecked(
                &self,
                _name: &str,
                _dtype: DType,
                _dev: &Device,
            ) -> candle_core::Result<Tensor> {
                unreachable!("stub")
            }
            fn contains_tensor(&self, _name: &str) -> bool {
                false
            }
        }
        let progress = ProgressReporter::default();
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(StubBackend);
        let result = wrap_backend_with_lora(inner, &[], &progress, None);
        assert!(
            result.is_err(),
            "wrap_backend_with_lora must reject empty specs"
        );
    }
}
