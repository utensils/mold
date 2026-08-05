//! Z-Image LoRA support.
//!
//! Z-Image has a curious architecture choice that simplifies LoRA wiring
//! compared to FLUX / Flux.2: there is exactly **one** candle key-space the
//! transformer ever sees — the diffusers / split-Q-K-V layout that
//! `candle_transformers::models::z_image::ZImageTransformer2DModel`
//! consumes. The GGUF "quantized" path actually dequantises into BF16 dense
//! tensors via [`super::gguf_dense::load_gguf_dense_transformer`] before the
//! transformer is constructed — the *fused* `attention.qkv` weight is split
//! into three `attention.to_q` / `attention.to_k` / `attention.to_v`
//! tensors at load time, never reaches the model.
//!
//! What that means for LoRAs:
//!
//! * Civitai-style Z-Image LoRAs (cv:2904324 is the canonical reference)
//!   target the **BFL-native fused** `diffusion_model.layers.{i}.attention.qkv`
//!   layer. Their `B` is shape `[3 * dim, rank]`. We must **splat** each
//!   third of `B @ A` onto the corresponding split `to_q` / `to_k` / `to_v`
//!   tensor — exactly the asymmetry flux2's `Splat` patch type was built
//!   for, just with a single key-space rather than two.
//! * Trainers that natively emit split `attention.to_q` / `to_k` / `to_v`
//!   layers map to `Direct` patches, one tensor each.
//! * Non-attention leaves (`attention.out`, `feed_forward.w{1,2,3}`,
//!   `adaLN_modulation.0`) map to `Direct` regardless of LoRA naming.
//!
//! The BF16 path wraps the dense `VarBuilder` backend with
//! [`ZImageLoraBackend`]. The GGUF path patches affected quantized tensors
//! in place and hands the result to the quantized transformer, avoiding a
//! full dense copy of the checkpoint.
//!
//! # Recognised LoRA-key shapes
//!
//! | Form | Example |
//! |---|---|
//! | Kohya / sd-scripts | `lora_unet_layers_0_attention_qkv.lora_down.weight` |
//! | PEFT canonical, BFL prefix | `diffusion_model.layers.0.attention.qkv.lora_A.weight` |
//! | PEFT canonical, root | `layers.0.attention.qkv.lora_A.weight` |
//! | PEFT default-adapter | `diffusion_model.layers.0.attention.qkv.lora_A.default.weight` |
//! | OneTrainer | `layers.0.attention.qkv.lora_linear_layer.down.weight` |
//!
//! Suffix detection is delegated to [`crate::flux::lora::classify_lora_key`]
//! — the same matrix as FLUX so we never need to re-invent that table.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use mold_candle::quantized::VarBuilder as QuantizedVarBuilder;

use crate::flux::lora::{get_or_load_adapter, LoraAdapter, LoraDeltaCache};
use crate::progress::ProgressReporter;

// ---------------------------------------------------------------------------
// Path-hash helper — used to seed per-adapter cache keys.
// ---------------------------------------------------------------------------

pub(crate) fn lora_path_hash(path: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Target descriptors — how a LoRA delta lands on a candle tensor.
// ---------------------------------------------------------------------------

/// How a LoRA layer's `B @ A * scale` delta lands on a candle tensor.
///
/// The patch type carries the *full* delta in both arms; the difference is
/// whether we add all of it (`Direct`) or slice a row-band out of it first
/// (`Splat`). The `row_size == 0` sentinel encodes "split the delta into
/// equal thirds and take the `row_offset`-th third" — the resolved
/// `(offset, size)` is filled in once `B`'s row count is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ZImageLoraTarget {
    Direct {
        candle_key: String,
    },
    Splat {
        candle_key: String,
        row_offset: usize,
        row_size: usize,
    },
}

impl ZImageLoraTarget {
    fn candle_key(&self) -> &str {
        match self {
            Self::Direct { candle_key } => candle_key,
            Self::Splat { candle_key, .. } => candle_key,
        }
    }
}

// ---------------------------------------------------------------------------
// Key mapping: LoRA stem → candle target(s).
// ---------------------------------------------------------------------------

fn strip_known_prefixes(stem: &str) -> &str {
    let s = stem.strip_prefix("model.").unwrap_or(stem);
    let s = s.strip_prefix("diffusion_model.").unwrap_or(s);
    s.strip_prefix("transformer.").unwrap_or(s)
}

/// Map a LoRA layer stem (suffix already stripped by `classify_lora_key`)
/// to one or more candle targets. Returns an empty `Vec` for stems we don't
/// recognise — the caller logs and skips.
pub(crate) fn map_zimage_lora_key(raw_stem: &str) -> Vec<ZImageLoraTarget> {
    // ── Kohya / sd-scripts: lora_unet_<module-with-underscores> ──────────
    if let Some(rest) = raw_stem.strip_prefix("lora_unet_") {
        return map_kohya(rest);
    }

    // ── PEFT canonical: optional `transformer.` / `diffusion_model.` /
    //    `model.` prefix, then dotted BFL module path ─────────────────────
    let stem = strip_known_prefixes(raw_stem);

    for block_name in BLOCK_NAMES {
        let prefix = format!("{block_name}.");
        if let Some(rest) = stem.strip_prefix(&prefix) {
            return map_block_dotted(block_name, rest);
        }
    }
    Vec::new()
}

/// The three top-level block lists in `ZImageTransformer2DModel`.
///
/// Trainers in the wild target every one of these — cv:2904324 only
/// includes `layers.*`, but the candle module reserves names for all
/// three.
const BLOCK_NAMES: &[&str] = &["layers", "noise_refiner", "context_refiner"];

fn map_kohya(rest: &str) -> Vec<ZImageLoraTarget> {
    // Kohya path: `<block>_<idx>_<flattened module dots become underscores>`.
    // `block` is one of `layers`, `noise_refiner`, `context_refiner`.
    for block_name in BLOCK_NAMES {
        let prefix = format!("{block_name}_");
        if let Some(after) = rest.strip_prefix(&prefix) {
            let (idx_str, leaf_us) = match after.split_once('_') {
                Some(p) => p,
                None => return Vec::new(),
            };
            if idx_str.parse::<usize>().is_err() {
                return Vec::new();
            }
            return map_block_leaf(block_name, idx_str, leaf_us);
        }
    }
    Vec::new()
}

/// `rest` here is `<idx>.<dotted leaf>` (e.g. `0.attention.qkv`).
fn map_block_dotted(block_name: &str, rest: &str) -> Vec<ZImageLoraTarget> {
    let (idx, leaf) = match rest.split_once('.') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx.parse::<usize>().is_err() {
        return Vec::new();
    }
    // Flatten dotted leaf to the Kohya underscore form so we only maintain
    // one table.
    let kohya_leaf = leaf.replace('.', "_");
    map_block_leaf(block_name, idx, &kohya_leaf)
}

/// Dispatch the underscore-form leaf onto a Z-Image candle target.
///
/// The block name is preserved verbatim — `layers.{i}.…`,
/// `noise_refiner.{i}.…`, `context_refiner.{i}.…` are valid candle paths
/// for all blocks. The leaves we recognise:
///
/// | LoRA leaf (underscores) | Candle target |
/// |---|---|
/// | `attention_qkv` | three Splat patches on `attention.to_q/to_k/to_v` |
/// | `attention_to_q` / `_to_k` / `_to_v` | Direct on the matching tensor |
/// | `attention_out` | Direct on `attention.to_out.0` |
/// | `attention_to_out_0` | Direct on `attention.to_out.0` |
/// | `feed_forward_w1` / `_w2` / `_w3` | Direct on the matching FF weight |
/// | `adaLN_modulation_0` | Direct on `adaLN_modulation.0` (only for blocks with modulation) |
fn map_block_leaf(block: &str, idx: &str, leaf_us: &str) -> Vec<ZImageLoraTarget> {
    match leaf_us {
        // Fused QKV — splat across the three split tensors.
        "attention_qkv" => vec![
            ZImageLoraTarget::Splat {
                candle_key: format!("{block}.{idx}.attention.to_q.weight"),
                row_offset: 0,
                row_size: 0,
            },
            ZImageLoraTarget::Splat {
                candle_key: format!("{block}.{idx}.attention.to_k.weight"),
                row_offset: 1,
                row_size: 0,
            },
            ZImageLoraTarget::Splat {
                candle_key: format!("{block}.{idx}.attention.to_v.weight"),
                row_offset: 2,
                row_size: 0,
            },
        ],
        // Already-split projections — Direct.
        "attention_to_q" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.attention.to_q.weight"),
        }],
        "attention_to_k" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.attention.to_k.weight"),
        }],
        "attention_to_v" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.attention.to_v.weight"),
        }],
        // Output projection — both BFL `attention.out` and diffusers
        // `attention.to_out.0` collapse to the same candle tensor.
        "attention_out" | "attention_to_out_0" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.attention.to_out.0.weight"),
        }],
        // SwiGLU feed-forward.
        "feed_forward_w1" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.feed_forward.w1.weight"),
        }],
        "feed_forward_w2" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.feed_forward.w2.weight"),
        }],
        "feed_forward_w3" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.feed_forward.w3.weight"),
        }],
        // adaLN modulation (only present on `layers.*` and `noise_refiner.*`;
        // `context_refiner.*` blocks have no modulation, so a LoRA targeting
        // adaLN_modulation on a context refiner will resolve to a candle
        // key that doesn't exist — the apply path logs and skips).
        "adaLN_modulation_0" => vec![ZImageLoraTarget::Direct {
            candle_key: format!("{block}.{idx}.adaLN_modulation.0.weight"),
        }],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Patch building — turn every (adapter, layer) pair into per-tensor patches.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ZImageLoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: ZImageLoraTarget,
    /// Retained for future per-adapter delta caching (mirrors flux2's
    /// API for consistency). Currently unused — Z-Image's transformer
    /// rebuild cadence is low enough that re-computing `B @ A` on each
    /// load is cheap. Tests rely on the field being populated.
    #[allow(dead_code)]
    lora_path_hash: u64,
    /// Resolved `(offset, size)` for `Splat` targets. Filled at patch-build
    /// time once `B`'s row count is known.
    resolved_rows: Option<(usize, usize)>,
}

/// A loaded LoRA + its scale + a stable hash of its file path.
pub(crate) struct ZImageLoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
    pub path_hash: u64,
}

fn resolve_rows(target: &ZImageLoraTarget, b_rows: usize) -> Option<(usize, usize)> {
    match target {
        ZImageLoraTarget::Direct { .. } => None,
        ZImageLoraTarget::Splat {
            row_size,
            row_offset,
            ..
        } => {
            if *row_size == 0 {
                let third = b_rows / 3;
                Some((row_offset * third, third))
            } else {
                Some((*row_offset, *row_size))
            }
        }
    }
}

fn build_patches(specs: &[ZImageLoraSpec<'_>]) -> (HashMap<String, Vec<ZImageLoraPatch>>, usize) {
    let mut patches: HashMap<String, Vec<ZImageLoraPatch>> = HashMap::new();
    let mut skipped = 0usize;
    for spec in specs {
        for (lora_stem, layer) in &spec.adapter.layers {
            let targets = map_zimage_lora_key(lora_stem);
            if targets.is_empty() {
                tracing::warn!(
                    key = lora_stem.as_str(),
                    "unrecognized Z-Image LoRA key, skipping"
                );
                skipped += 1;
                continue;
            }
            let rank = layer.a.dims()[0] as f64;
            let effective_scale = match layer.alpha {
                Some(alpha) => spec.scale * alpha / rank,
                None => spec.scale,
            };
            let b_rows = layer.b.dims().first().copied().unwrap_or(0);
            for target in targets {
                let resolved_rows = resolve_rows(&target, b_rows);
                let candle_key = target.candle_key().to_string();
                patches
                    .entry(candle_key)
                    .or_default()
                    .push(ZImageLoraPatch {
                        a: layer.a.clone(),
                        b: layer.b.clone(),
                        effective_scale,
                        target,
                        lora_path_hash: spec.path_hash,
                        resolved_rows,
                    });
            }
        }
    }
    (patches, skipped)
}

// ---------------------------------------------------------------------------
// Delta computation + apply.
// ---------------------------------------------------------------------------

fn compute_delta(patch: &ZImageLoraPatch, target_dev: &Device) -> candle_core::Result<Tensor> {
    let a = patch.a.to_dtype(DType::F32)?.to_device(target_dev)?;
    let b = patch.b.to_dtype(DType::F32)?.to_device(target_dev)?;
    let computed = b.matmul(&a)?;
    &computed * patch.effective_scale
}

/// Add `delta_full` (or the appropriate row-slice of it) to an F32 base.
fn apply_patch_f32(
    base_f32: &Tensor,
    delta_full: &Tensor,
    patch: &ZImageLoraPatch,
) -> candle_core::Result<Tensor> {
    match &patch.target {
        ZImageLoraTarget::Direct { .. } => base_f32 + delta_full,
        ZImageLoraTarget::Splat { .. } => {
            let (offset, size) = patch
                .resolved_rows
                .expect("Splat patch must have resolved_rows");
            let delta_slice = delta_full.narrow(0, offset, size)?;
            let base_rows = base_f32.dim(0)?;
            if base_rows != size {
                tracing::warn!(
                    base_rows,
                    delta_rows = size,
                    "Z-Image LoRA Splat: base row count != delta row count, skipping"
                );
                return Ok(base_f32.clone());
            }
            base_f32 + &delta_slice
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ZImageGgufLoraTarget {
    Direct {
        tensor_key: String,
    },
    FusedSlice {
        tensor_key: String,
        component: usize,
        num_components: usize,
    },
}

impl ZImageGgufLoraTarget {
    fn tensor_key(&self) -> &str {
        match self {
            Self::Direct { tensor_key } | Self::FusedSlice { tensor_key, .. } => tensor_key,
        }
    }

    #[cfg(test)]
    fn component(&self) -> Option<(usize, usize)> {
        match self {
            Self::Direct { .. } => None,
            Self::FusedSlice {
                component,
                num_components,
                ..
            } => Some((*component, *num_components)),
        }
    }
}

fn attention_qkv_component(candle_key: &str) -> Option<(String, usize)> {
    for (needle, component) in [
        (".attention.to_q.weight", 0),
        (".attention.to_k.weight", 1),
        (".attention.to_v.weight", 2),
    ] {
        if let Some(prefix) = candle_key.strip_suffix(needle) {
            return Some((format!("{prefix}.attention.qkv.weight"), component));
        }
    }
    None
}

fn map_gguf_lora_target(target: &ZImageLoraTarget) -> Option<ZImageGgufLoraTarget> {
    let candle_key = target.candle_key();
    if let Some((tensor_key, component)) = attention_qkv_component(candle_key) {
        return Some(ZImageGgufLoraTarget::FusedSlice {
            tensor_key,
            component,
            num_components: 3,
        });
    }

    let tensor_key = candle_key.replace(".attention.to_out.0.weight", ".attention.out.weight");
    Some(ZImageGgufLoraTarget::Direct { tensor_key })
}

fn gguf_patch_delta(delta_full: &Tensor, patch: &ZImageLoraPatch) -> candle_core::Result<Tensor> {
    match &patch.target {
        ZImageLoraTarget::Direct { .. } => Ok(delta_full.clone()),
        ZImageLoraTarget::Splat { .. } => {
            let (offset, size) = patch
                .resolved_rows
                .expect("Splat patch must have resolved_rows");
            delta_full.narrow(0, offset, size)
        }
    }
}

fn apply_gguf_patch_f32(
    base_f32: &Tensor,
    delta_full: &Tensor,
    patch: &ZImageLoraPatch,
    target: &ZImageGgufLoraTarget,
) -> candle_core::Result<Tensor> {
    match target {
        ZImageGgufLoraTarget::Direct { .. } => {
            let delta = gguf_patch_delta(delta_full, patch)?;
            base_f32 + &delta
        }
        ZImageGgufLoraTarget::FusedSlice {
            component,
            num_components,
            ..
        } => {
            let delta = gguf_patch_delta(delta_full, patch)?;
            let base_rows = base_f32.dim(0)?;
            let slice_rows = base_rows / num_components;
            let offset = component * slice_rows;
            if offset + slice_rows > base_rows || delta.dim(0)? != slice_rows {
                tracing::warn!(
                    offset,
                    slice_rows,
                    base_rows,
                    delta_rows = delta.dim(0).unwrap_or(0),
                    "Z-Image GGUF LoRA fused slice shape mismatch, skipping"
                );
                return Ok(base_f32.clone());
            }

            let slice = base_f32.narrow(0, offset, slice_rows)?;
            let updated_slice = (&slice + &delta)?;
            let mut parts = Vec::new();
            if offset > 0 {
                parts.push(base_f32.narrow(0, 0, offset)?);
            }
            parts.push(updated_slice);
            let after = offset + slice_rows;
            if after < base_rows {
                parts.push(base_f32.narrow(0, after, base_rows - after)?);
            }
            Tensor::cat(&parts, 0)
        }
    }
}

// ---------------------------------------------------------------------------
// `LoraBackend` — wraps a `SimpleBackend` and merges LoRAs at `vb.get()`.
// ---------------------------------------------------------------------------

struct ZImageLoraBackend {
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    patches: HashMap<String, Vec<ZImageLoraPatch>>,
    // Reserved for future delta caching (mirrors flux2's API for consistency
    // when refactoring; currently unused — Z-Image's transformer rebuild
    // cadence is low enough that re-computing `B @ A` on each construction
    // is cheap).
    _delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

impl ZImageLoraBackend {
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
        for patch in layer_patches.iter() {
            let delta_full = compute_delta(patch, dev)?;
            merged = apply_patch_f32(&merged, &delta_full, patch)?;
        }
        merged.to_dtype(target_dtype)
    }
}

impl candle_nn::var_builder::SimpleBackend for ZImageLoraBackend {
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
// Public entry points.
// ---------------------------------------------------------------------------

/// Wrap an existing `SimpleBackend` so its `vb.get()` calls deliver
/// LoRA-merged tensors. The wrapper applies deltas in F32 and casts back
/// to the requested dtype.
pub(crate) fn wrap_backend_with_lora(
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    specs: &[ZImageLoraSpec<'_>],
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
        "LoRA (Z-Image): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    Ok(Box::new(ZImageLoraBackend {
        inner,
        patches,
        _delta_cache: delta_cache,
    }))
}

/// Build `ZImageLoraSpec`s by loading every adapter through the shared
/// parsed-LoRA cache (`crate::flux::lora::get_or_load_adapter`). Returns
/// the `Arc`s so the caller can hold them for the lifetime of the spec
/// slice.
pub(crate) fn load_lora_adapters(
    loras: &[mold_core::LoraWeight],
    progress: &ProgressReporter,
) -> Result<Vec<Arc<LoraAdapter>>> {
    loras
        .iter()
        .map(|w| {
            progress.info("Loading Z-Image LoRA adapter");
            let adapter = get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "Z-Image LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect()
}

/// Build a quantized GGUF VarBuilder with Z-Image LoRA deltas merged into
/// affected tensors. This keeps the transformer quantized instead of
/// dequantizing the whole checkpoint into dense BF16 tensors first.
pub(crate) fn gguf_lora_var_builder(
    transformer_path: &Path,
    specs: &[ZImageLoraSpec<'_>],
    device: &Device,
    progress: &ProgressReporter,
) -> Result<QuantizedVarBuilder> {
    use candle_core::quantized::{gguf_file, QTensor};

    if specs.is_empty() {
        bail!("gguf_lora_var_builder called with no LoraSpecs");
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
        "Z-Image LoRA (GGUF): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let gguf_bytes_total = std::fs::metadata(transformer_path)
        .map(|m| m.len())
        .unwrap_or(0);
    progress.weight_load("Z-Image transformer (GGUF)", 0, gguf_bytes_total);
    for (i, tensor_name) in content.tensor_infos.keys().enumerate() {
        let qtensor = content.tensor(&mut file, tensor_name, device)?;
        data.insert(tensor_name.clone(), Arc::new(qtensor));
        let approx_bytes = gguf_bytes_total * (i as u64 + 1) / total_tensors as u64;
        progress.weight_load(
            "Z-Image transformer (GGUF)",
            approx_bytes.min(gguf_bytes_total),
            gguf_bytes_total,
        );
    }
    drop(file);

    let mut native_patches: HashMap<String, Vec<(ZImageGgufLoraTarget, ZImageLoraPatch)>> =
        HashMap::new();
    for layer_patches in patches.values() {
        for patch in layer_patches {
            if let Some(target) = map_gguf_lora_target(&patch.target) {
                native_patches
                    .entry(target.tensor_key().to_string())
                    .or_default()
                    .push((target, patch.clone()));
            }
        }
    }

    let on_gpu = device.is_cuda() || device.is_metal();
    let mut applied = 0usize;
    let native_keys: Vec<String> = native_patches.keys().cloned().collect();
    let native_total = native_keys.len();
    for (i, tensor_key) in native_keys.iter().enumerate() {
        let Some(layer_patches) = native_patches.get(tensor_key) else {
            continue;
        };
        let Some(qtensor) = data.remove(tensor_key) else {
            tracing::warn!(
                key = tensor_key.as_str(),
                "Z-Image GGUF LoRA target tensor not found, skipping"
            );
            continue;
        };

        let orig_dtype = qtensor.dtype();
        let mut t = qtensor.dequantize(&Device::Cpu)?;
        drop(qtensor);
        if on_gpu {
            device.synchronize()?;
        }

        for (target, patch) in layer_patches.iter() {
            let matmul_dev = if on_gpu { device } else { &Device::Cpu };
            let delta_full = compute_delta(patch, matmul_dev)?.to_device(&Device::Cpu)?;
            t = apply_gguf_patch_f32(&t, &delta_full, patch, target)?;
            applied += 1;
        }

        let patched = mold_candle::quantized::quantize_onto(&t, orig_dtype, device)?;
        drop(t);
        data.insert(tensor_key.clone(), Arc::new(patched));

        if (i + 1) % 16 == 0 || i + 1 == native_total {
            progress.info(&format!(
                "Z-Image LoRA GGUF merge: {}/{} tensors",
                i + 1,
                native_total,
            ));
        }
    }

    let total_layers: usize = specs.iter().map(|s| s.adapter.layers.len()).sum();
    progress.info(&format!(
        "Z-Image LoRA (GGUF): {applied} applied, {} skipped (max rank {max_rank}, {patched_keys} layers patched)",
        total_layers.saturating_sub(applied),
    ));

    if on_gpu {
        device.synchronize()?;
    }

    Ok(QuantizedVarBuilder::from_qtensors(data, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::lora::LoraLayer;
    use safetensors::tensor::TensorView;

    // ── map_zimage_lora_key — leaf coverage ────────────────────────────

    /// Pin every leaf observed in real Z-Image LoRAs (cv:2904324). A
    /// refactor that silently drops one of these breaks every adapter in
    /// the wild.
    #[test]
    fn kohya_layers_leaves_recognised() {
        let cases: &[(&str, &str)] = &[
            ("attention_out", "layers.0.attention.to_out.0.weight"),
            ("feed_forward_w1", "layers.0.feed_forward.w1.weight"),
            ("feed_forward_w2", "layers.0.feed_forward.w2.weight"),
            ("feed_forward_w3", "layers.0.feed_forward.w3.weight"),
            ("adaLN_modulation_0", "layers.0.adaLN_modulation.0.weight"),
        ];
        for (leaf, expected) in cases {
            let key = format!("lora_unet_layers_0_{leaf}");
            let targets = map_zimage_lora_key(&key);
            assert_eq!(targets.len(), 1, "exactly one target for {key}");
            match &targets[0] {
                ZImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf}");
                }
                _ => panic!("expected Direct for {leaf}"),
            }
        }
    }

    /// The fused-QKV Kohya leaf splits into three Splat targets in
    /// canonical Q→K→V order.
    #[test]
    fn kohya_attention_qkv_splits_into_three_splat_targets() {
        let targets = map_zimage_lora_key("lora_unet_layers_3_attention_qkv");
        assert_eq!(targets.len(), 3, "QKV splits into to_q/to_k/to_v");
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "layers.3.attention.to_q.weight",
                "layers.3.attention.to_k.weight",
                "layers.3.attention.to_v.weight",
            ],
            "ordering is Q→K→V (component index 0/1/2)"
        );
        // Verify each carries the right component index.
        for (i, t) in targets.iter().enumerate() {
            match t {
                ZImageLoraTarget::Splat {
                    row_offset,
                    row_size,
                    ..
                } => {
                    assert_eq!(*row_offset, i, "component index");
                    assert_eq!(*row_size, 0, "row_size sentinel for thirds-split");
                }
                _ => panic!("expected Splat for component {i}"),
            }
        }
    }

    /// Pre-split Kohya leaves (`attention_to_q` / `_to_k` / `_to_v`) map to
    /// Direct patches — no splat needed when the trainer already separated
    /// the Q/K/V projections.
    #[test]
    fn kohya_pre_split_attention_leaves_resolve_direct() {
        for (leaf, expected) in [
            ("attention_to_q", "layers.0.attention.to_q.weight"),
            ("attention_to_k", "layers.0.attention.to_k.weight"),
            ("attention_to_v", "layers.0.attention.to_v.weight"),
        ] {
            let key = format!("lora_unet_layers_0_{leaf}");
            let targets = map_zimage_lora_key(&key);
            assert_eq!(targets.len(), 1, "leaf={leaf}");
            match &targets[0] {
                ZImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf}");
                }
                _ => panic!("expected Direct for {leaf}"),
            }
        }
    }

    #[test]
    fn gguf_lora_maps_split_attention_to_fused_qkv_slice() {
        let target = ZImageLoraTarget::Direct {
            candle_key: "layers.7.attention.to_k.weight".into(),
        };

        let mapped = map_gguf_lora_target(&target).expect("mapped GGUF target");

        assert_eq!(mapped.tensor_key(), "layers.7.attention.qkv.weight");
        assert_eq!(mapped.component(), Some((1, 3)));
    }

    /// PEFT canonical with the `diffusion_model.` prefix — the form
    /// cv:2904324 actually ships.
    #[test]
    fn peft_canonical_diffusion_model_prefix_resolves() {
        let stem = "diffusion_model.layers.0.attention.qkv";
        let targets = map_zimage_lora_key(stem);
        assert_eq!(
            targets.len(),
            3,
            "PEFT canonical fused-QKV must splat into three"
        );
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "layers.0.attention.to_q.weight",
                "layers.0.attention.to_k.weight",
                "layers.0.attention.to_v.weight",
            ]
        );
    }

    /// The exact PEFT-canonical key shape cv:2904324 ships for every
    /// `attention.out` layer. Must collapse to candle's `to_out.0`.
    #[test]
    fn peft_canonical_attention_out_maps_to_to_out_0() {
        let stem = "diffusion_model.layers.0.attention.out";
        let targets = map_zimage_lora_key(stem);
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            ZImageLoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "layers.0.attention.to_out.0.weight");
            }
            _ => panic!("expected Direct for attention.out"),
        }
    }

    /// The exact PEFT-canonical adaLN key cv:2904324 ships.
    #[test]
    fn peft_canonical_adaln_modulation_resolves() {
        let stem = "diffusion_model.layers.0.adaLN_modulation.0";
        let targets = map_zimage_lora_key(stem);
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            ZImageLoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "layers.0.adaLN_modulation.0.weight");
            }
            _ => panic!("expected Direct for adaLN_modulation.0"),
        }
    }

    #[test]
    fn peft_canonical_strips_optional_model_and_transformer_prefix() {
        for stem in [
            "model.diffusion_model.layers.7.feed_forward.w1",
            "transformer.layers.7.feed_forward.w1",
            "layers.7.feed_forward.w1",
        ] {
            let targets = map_zimage_lora_key(stem);
            assert_eq!(targets.len(), 1, "stem={stem}");
            match &targets[0] {
                ZImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, "layers.7.feed_forward.w1.weight");
                }
                _ => panic!("expected Direct for {stem}"),
            }
        }
    }

    /// Both refiner block families must resolve — the wild has trainers
    /// that target every block list.
    #[test]
    fn noise_and_context_refiner_blocks_recognised() {
        for block in ["noise_refiner", "context_refiner"] {
            let stem = format!("diffusion_model.{block}.0.feed_forward.w2");
            let targets = map_zimage_lora_key(&stem);
            assert_eq!(targets.len(), 1);
            match &targets[0] {
                ZImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, &format!("{block}.0.feed_forward.w2.weight"));
                }
                _ => panic!("expected Direct"),
            }
        }
    }

    #[test]
    fn refiner_qkv_splat_uses_refiner_block_name() {
        let targets = map_zimage_lora_key("lora_unet_noise_refiner_2_attention_qkv");
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "noise_refiner.2.attention.to_q.weight",
                "noise_refiner.2.attention.to_k.weight",
                "noise_refiner.2.attention.to_v.weight",
            ]
        );
    }

    #[test]
    fn unknown_leaf_returns_empty_vec() {
        assert!(map_zimage_lora_key("lora_unet_layers_0_unknown_thing").is_empty());
        assert!(map_zimage_lora_key("lora_te_text_model_layer_0_attn_q").is_empty());
        assert!(map_zimage_lora_key("garbage").is_empty());
    }

    // ── resolve_rows ────────────────────────────────────────────────────

    #[test]
    fn resolve_rows_thirds_for_splat_with_zero_size() {
        let target = ZImageLoraTarget::Splat {
            candle_key: "x".into(),
            row_offset: 1,
            row_size: 0,
        };
        let rows = resolve_rows(&target, 3 * 8).unwrap();
        assert_eq!(rows, (8, 8), "component 1 of 3 in a 24-row B");
    }

    #[test]
    fn resolve_rows_direct_target_is_none() {
        let target = ZImageLoraTarget::Direct {
            candle_key: "x".into(),
        };
        assert!(resolve_rows(&target, 12).is_none());
    }

    // ── apply_patch_f32 — math correctness ──────────────────────────────

    #[test]
    fn apply_patch_direct_adds_full_delta() {
        let dev = Device::Cpu;
        let base = Tensor::full(2.0f32, (4, 3), &dev).unwrap();
        let delta = Tensor::full(0.5f32, (4, 3), &dev).unwrap();
        let patch = ZImageLoraPatch {
            a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            effective_scale: 1.0,
            target: ZImageLoraTarget::Direct {
                candle_key: "x".into(),
            },
            lora_path_hash: 0,
            resolved_rows: None,
        };
        let merged = apply_patch_f32(&base, &delta, &patch).unwrap();
        let vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 2.5).abs() < 1e-6),
            "Direct merge expected base + delta = 2.5 everywhere, got {vals:?}",
        );
    }

    /// Z-Image's load-bearing test: a fused-QKV delta must hit the *right*
    /// third of each split tensor.
    #[test]
    fn apply_patch_splat_uses_correct_third_of_delta() {
        let dev = Device::Cpu;
        // delta_full's three thirds are filled with constants 0.1 / 0.2 / 0.3.
        // Q takes rows 0..h, K rows h..2h, V rows 2h..3h.
        let h = 3;
        let in_dim = 2;
        let mut delta_data = Vec::with_capacity(3 * h * in_dim);
        for v in [0.1f32, 0.2, 0.3] {
            for _ in 0..(h * in_dim) {
                delta_data.push(v);
            }
        }
        let delta_full = Tensor::from_vec(delta_data, (3 * h, in_dim), &dev).unwrap();

        let base = Tensor::zeros((h, in_dim), DType::F32, &dev).unwrap();
        for (component, expected) in [(0, 0.1f32), (1, 0.2), (2, 0.3)] {
            let mut patch = ZImageLoraPatch {
                a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                effective_scale: 1.0,
                target: ZImageLoraTarget::Splat {
                    candle_key: "x".into(),
                    row_offset: component,
                    row_size: 0,
                },
                lora_path_hash: 0,
                resolved_rows: None,
            };
            patch.resolved_rows = resolve_rows(&patch.target, 3 * h);
            let merged = apply_patch_f32(&base, &delta_full, &patch).unwrap();
            let vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                vals.iter().all(|v| (v - expected).abs() < 1e-6),
                "component {component}: expected {expected} everywhere, got {vals:?}",
            );
        }
    }

    /// `apply_patch_f32` for Splat must refuse a base whose row count
    /// disagrees with the delta third (corrupt LoRA / wrong target).
    #[test]
    fn apply_patch_splat_skips_when_dim_mismatches() {
        let dev = Device::Cpu;
        let h = 3;
        let in_dim = 2;
        let delta_full = Tensor::full(0.7f32, (3 * h, in_dim), &dev).unwrap();
        let wrong_base = Tensor::full(5.0f32, (h + 1, in_dim), &dev).unwrap();
        let mut patch = ZImageLoraPatch {
            a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            effective_scale: 1.0,
            target: ZImageLoraTarget::Splat {
                candle_key: "x".into(),
                row_offset: 0,
                row_size: 0,
            },
            lora_path_hash: 0,
            resolved_rows: None,
        };
        patch.resolved_rows = resolve_rows(&patch.target, 3 * h);
        let merged = apply_patch_f32(&wrong_base, &delta_full, &patch).unwrap();
        // Skip path returns the base unchanged.
        let vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| (v - 5.0).abs() < 1e-6));
    }

    // ── build_patches — adapter wiring ──────────────────────────────────

    fn synthetic_kohya_adapter(layer: &str, b_rows: usize) -> LoraAdapter {
        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 4), &dev).unwrap();
        let b = Tensor::full(1.0f32, (b_rows, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(layer.to_string(), LoraLayer { a, b, alpha: None });
        LoraAdapter { layers, rank: 2 }
    }

    #[test]
    fn build_patches_fused_qkv_records_three_splat_buckets() {
        let adapter = synthetic_kohya_adapter("lora_unet_layers_0_attention_qkv", 6);
        let specs = [ZImageLoraSpec {
            adapter: &adapter,
            scale: 0.7,
            path_hash: 0xCAFE,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0);
        assert_eq!(patches.len(), 3);
        for k in [
            "layers.0.attention.to_q.weight",
            "layers.0.attention.to_k.weight",
            "layers.0.attention.to_v.weight",
        ] {
            assert!(patches.contains_key(k), "missing {k}");
            let bucket = &patches[k];
            assert_eq!(bucket.len(), 1);
            // resolved_rows = (component * (6/3), 2) — six rows split into thirds.
            assert_eq!(bucket[0].resolved_rows.unwrap().1, 2);
        }
    }

    #[test]
    fn build_patches_alpha_normalises_scale() {
        let dev = Device::Cpu;
        let mut adapter = synthetic_kohya_adapter("lora_unet_layers_0_attention_out", 4);
        adapter
            .layers
            .get_mut("lora_unet_layers_0_attention_out")
            .unwrap()
            .alpha = Some(4.0);
        let _dev_use = dev;
        let specs = [ZImageLoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0,
        }];
        let (patches, _) = build_patches(&specs);
        let bucket = &patches["layers.0.attention.to_out.0.weight"];
        let s = bucket[0].effective_scale;
        assert!(
            (s - 1.0).abs() < 1e-9,
            "effective scale = user(0.5) * alpha(4) / rank(2) = 1.0, got {s}"
        );
    }

    #[test]
    fn build_patches_skips_unknown_keys() {
        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 4), &dev).unwrap();
        let b = Tensor::full(1.0f32, (8, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_garbage_42_unknown".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [ZImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert!(patches.is_empty());
        assert_eq!(skipped, 1);
    }

    #[test]
    fn build_patches_two_specs_stack_on_same_target() {
        let a1 = synthetic_kohya_adapter("lora_unet_layers_0_feed_forward_w1", 4);
        let a2 = synthetic_kohya_adapter("lora_unet_layers_0_feed_forward_w1", 4);
        let specs = [
            ZImageLoraSpec {
                adapter: &a1,
                scale: 1.0,
                path_hash: 0xAA,
            },
            ZImageLoraSpec {
                adapter: &a2,
                scale: 1.0,
                path_hash: 0xBB,
            },
        ];
        let (patches, _) = build_patches(&specs);
        let bucket = &patches["layers.0.feed_forward.w1.weight"];
        assert_eq!(bucket.len(), 2);
        assert_eq!(bucket[0].lora_path_hash, 0xAA);
        assert_eq!(bucket[1].lora_path_hash, 0xBB);
    }

    // ── End-to-end via SimpleBackend wrapper ────────────────────────────

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

    /// End-to-end Splat: a fused-QKV adapter must land its three thirds on
    /// the three split candle tensors.
    #[test]
    fn end_to_end_fused_qkv_splat_lands_on_three_tensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base.safetensors");

        let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for k in [
            "layers.0.attention.to_q.weight",
            "layers.0.attention.to_k.weight",
            "layers.0.attention.to_v.weight",
        ] {
            entries.push((k.to_string(), vec![2, 8], vec![0.0; 16]));
        }
        write_synthetic_safetensors_with_data(&path, &entries);

        // B = 6×2 of ones, A = 2×8 of ones. B@A = 6×8 of 2. Split into
        // three 2×8 thirds, each constant 2. Base = 0 → merged = 2.
        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(1.0f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_layers_0_attention_qkv".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [ZImageLoraSpec {
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

        for k in [
            "layers.0.attention.to_q.weight",
            "layers.0.attention.to_k.weight",
            "layers.0.attention.to_v.weight",
        ] {
            let t = wrapped.get_unchecked(k, DType::F32, &dev).expect("get");
            let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
                "{k}: expected constant 2.0 (= 0 + B@A row-third), got {vals:?}",
            );
        }
    }

    /// End-to-end Direct: a leaf-named LoRA (`feed_forward_w1`) merges via
    /// the additive `base + B@A·scale` path with no slicing.
    #[test]
    fn end_to_end_direct_merge_on_feed_forward() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base_direct.safetensors");
        write_synthetic_safetensors_with_data(
            &path,
            &[(
                "layers.0.feed_forward.w1.weight".to_string(),
                vec![6, 8],
                vec![1.0; 48],
            )],
        );

        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(0.5f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_layers_0_feed_forward_w1".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [ZImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
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

        // B@A entry = 2 * 0.5 * 1 = 1. Merged = base(1) + delta(1) = 2.
        let t = wrapped
            .get_unchecked("layers.0.feed_forward.w1.weight", DType::F32, &dev)
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
            "expected 2.0 (= 1 + 1), got {vals:?}",
        );
    }

    /// End-to-end via `DenseVarBuilder::from_tensors` (the form the GGUF
    /// path ultimately constructs after dequantising): the LoRA wrapper
    /// behaves identically regardless of the inner backend impl.
    #[test]
    fn end_to_end_dense_var_builder_path_picks_up_lora() {
        let dev = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "layers.0.feed_forward.w1.weight".to_string(),
            Tensor::full(1.0f32, (4, 4), &dev).unwrap(),
        );

        let a = Tensor::full(1.0f32, (2, 4), &dev).unwrap();
        let b = Tensor::full(0.5f32, (4, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_layers_0_feed_forward_w1".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [ZImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];

        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(tensors);
        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");

        let t = wrapped
            .get_unchecked("layers.0.feed_forward.w1.weight", DType::F32, &dev)
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // B@A per entry = 2 * 0.5 * 1 = 1. base 1 + 1 = 2.
        assert!(vals.iter().all(|v| (v - 2.0).abs() < 1e-5));
    }

    /// Unrelated tensors must pass through unchanged.
    #[test]
    fn unrelated_tensors_pass_through_untouched() {
        let dev = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "unrelated.weight".to_string(),
            Tensor::full(7.0f32, (3, 3), &dev).unwrap(),
        );
        let adapter = synthetic_kohya_adapter("lora_unet_layers_0_feed_forward_w1", 4);
        let specs = [ZImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(tensors);
        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");
        let t = wrapped
            .get_unchecked("unrelated.weight", DType::F32, &dev)
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| (v - 7.0).abs() < 1e-6));
    }

    // ── wrap_backend_with_lora — input validation ───────────────────────

    #[test]
    fn wrap_backend_with_no_specs_returns_error() {
        let dev = Device::Cpu;
        let empty: HashMap<String, Tensor> = HashMap::new();
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(empty);
        let progress = ProgressReporter::default();
        let _ = dev;
        match wrap_backend_with_lora(inner, &[], &progress, None) {
            Ok(_) => panic!("expected error for empty spec list"),
            Err(e) => assert!(
                e.to_string().contains("no LoraSpecs"),
                "expected 'no LoraSpecs' message, got: {e}"
            ),
        }
    }

    #[test]
    fn lora_path_hash_is_deterministic_and_distinguishes() {
        let h1 = lora_path_hash("/a/b/c.safetensors");
        let h2 = lora_path_hash("/a/b/c.safetensors");
        let h3 = lora_path_hash("/a/b/d.safetensors");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
