//! Qwen-Image LoRA support.
//!
//! Qwen-Image is a 60-block dual-stream transformer with joint attention.
//! Both the BF16 official checkpoint and the GGUF quantized checkpoint
//! use **split** Q/K/V projections — the candle module exposes `to_q`
//! / `to_k` / `to_v` (image stream) and `add_q_proj` / `add_k_proj`
//! / `add_v_proj` (text stream). There is no fused QKV anywhere in
//! mold's qwen_image module.
//!
//! What that means for LoRAs:
//!
//! * **Native split** trainers (the diffusers / PEFT canonical layout)
//!   emit one `lora_A`/`lora_B` pair per Q/K/V projection. These map
//!   to `Direct` patches, one tensor each.
//! * **Fused** trainers — sometimes shipped on Civitai for
//!   compatibility with diffusers' fused-attention path — emit a
//!   single `attn.qkv` (or `attn.add_qkv_proj`) layer whose `B` is
//!   `[3·dim, rank]`. These map to three `Splat` patches that slice
//!   the appropriate third of `B @ A` onto the matching split tensor.
//!
//! Two on-disk naming axes complicate things further:
//!
//! 1. **FFN naming.** The official BF16 checkpoint ships
//!    `transformer_blocks.{i}.ff.net.0.proj.weight` while the
//!    ComfyUI / FP8 / GGUF re-export uses
//!    `transformer_blocks.{i}.img_mlp.net.0.proj.weight` (and a
//!    sibling `txt_mlp` for the text stream). Both candle paths exist
//!    inside the model — the constructor probes for `img_mlp.net.0.proj.weight`
//!    and instantiates the appropriate path.
//! 2. **AdaLN modulation naming.** Same split: BF16 uses
//!    `transformer_blocks.{i}.norm1.linear.weight` /
//!    `norm1_context.linear.weight`; ComfyUI/GGUF uses
//!    `transformer_blocks.{i}.img_mod.1.weight` /
//!    `txt_mod.1.weight`.
//!
//! Rather than probe the base checkpoint before building patches,
//! we emit **both** candidate candle keys. The
//! [`SimpleBackend`]-wrapping `LoraBackend` only fires a patch when
//! the requested tensor name matches; entries for a sibling naming
//! style are dormant no-ops. This keeps the mapper pure and the
//! patch table small (~2× the rows touched, still well under
//! the 60-block × ~12-leaf upper bound).
//!
//! Single-stream (image) leaves we recognise:
//!
//! | LoRA leaf | Candle target(s) |
//! |---|---|
//! | `attn.to_q` / `attn.to_k` / `attn.to_v` | Direct on `attn.{to_q,to_k,to_v}.weight` |
//! | `attn.qkv` (fused) | three Splat on `attn.{to_q,to_k,to_v}.weight` |
//! | `attn.to_out.0` / `attn.to_out_0` | Direct on `attn.to_out.0.weight` |
//! | `ff.net.0.proj` / `ff.net.2` | Direct + ComfyUI sibling (`img_mlp.net.{0.proj,2}.weight`) |
//! | `norm1.linear` | Direct + ComfyUI sibling (`img_mod.1.weight`) |
//!
//! Text-stream:
//!
//! | LoRA leaf | Candle target(s) |
//! |---|---|
//! | `attn.add_q_proj` / `attn.add_k_proj` / `attn.add_v_proj` | Direct |
//! | `attn.add_qkv_proj` (fused, rare) | three Splat on add_{q,k,v}_proj |
//! | `attn.to_add_out` | Direct |
//! | `ff_context.net.0.proj` / `ff_context.net.2` | Direct + ComfyUI sibling (`txt_mlp.net.{0.proj,2}.weight`) |
//! | `norm1_context.linear` | Direct + ComfyUI sibling (`txt_mod.1.weight`) |
//!
//! Suffix detection (Diffusers / PEFT canonical / Kohya / OneTrainer /
//! PEFT default-adapter / Mochi-edge) is delegated to
//! [`crate::flux::lora::classify_lora_key`].
//!
//! On-disk LoRAs in the wild: none of the mold maintainers have a
//! verified Qwen-Image LoRA on disk at time of writing. A real-LoRA
//! smoke test is gated behind `#[ignore]` until one ships — see the
//! `civitai_smoke` test below for the placeholder URL.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::flux::lora::{get_or_load_adapter, LoraAdapter, LoraDeltaCache};
use crate::progress::ProgressReporter;

// ---------------------------------------------------------------------------
// Path-hash helper.
// ---------------------------------------------------------------------------

pub(crate) fn lora_path_hash(path: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Target descriptors.
// ---------------------------------------------------------------------------

/// How a LoRA layer's `B @ A * scale` delta lands on a candle tensor.
///
/// `Direct` adds the entire delta to the matched tensor. `Splat` slices
/// a row-band out of the delta first — for fused-QKV LoRAs, where one
/// `B` carries Q | K | V stacked along the row axis. `row_size == 0` is
/// a sentinel that means "split the delta into equal thirds and take
/// the `row_offset`-th third"; the resolved `(offset, size)` is filled
/// in once `B`'s row count is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QwenImageLoraTarget {
    Direct {
        candle_key: String,
    },
    Splat {
        candle_key: String,
        row_offset: usize,
        row_size: usize,
    },
}

impl QwenImageLoraTarget {
    fn candle_key(&self) -> &str {
        match self {
            Self::Direct { candle_key } => candle_key,
            Self::Splat { candle_key, .. } => candle_key,
        }
    }
}

// ---------------------------------------------------------------------------
// Key mapping.
// ---------------------------------------------------------------------------

fn strip_known_prefixes(stem: &str) -> &str {
    let s = stem.strip_prefix("model.").unwrap_or(stem);
    let s = s.strip_prefix("diffusion_model.").unwrap_or(s);
    s.strip_prefix("transformer.").unwrap_or(s)
}

/// Map a LoRA layer stem (suffix already stripped by [`classify_lora_key`])
/// to one or more candle targets. Returns an empty `Vec` for stems we
/// don't recognise — the caller logs and skips.
///
/// [`classify_lora_key`]: crate::flux::lora::classify_lora_key
pub(crate) fn map_qwen_image_lora_key(raw_stem: &str) -> Vec<QwenImageLoraTarget> {
    // ── Kohya / sd-scripts: `lora_unet_<flattened-with-underscores>` ──
    if let Some(rest) = raw_stem.strip_prefix("lora_unet_") {
        return map_kohya(rest);
    }

    // ── PEFT canonical: optional `transformer.` / `diffusion_model.` /
    //    `model.` prefix, then dotted module path. ────────────────────
    let stem = strip_known_prefixes(raw_stem);
    let prefix = "transformer_blocks.";
    let rest = match stem.strip_prefix(prefix) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let (idx, leaf) = match rest.split_once('.') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx.parse::<usize>().is_err() {
        return Vec::new();
    }
    // Flatten the dotted leaf to underscore form so we maintain one
    // dispatch table for both Kohya and dotted forms.
    let kohya_leaf = leaf.replace('.', "_");
    map_block_leaf(idx, &kohya_leaf)
}

fn map_kohya(rest: &str) -> Vec<QwenImageLoraTarget> {
    // Kohya: `transformer_blocks_<idx>_<leaf-with-underscores>`. Some
    // trainers drop the `transformer_` and emit `blocks_<idx>_<leaf>`;
    // accept both.
    let after = rest
        .strip_prefix("transformer_blocks_")
        .or_else(|| rest.strip_prefix("blocks_"));
    let after = match after {
        Some(a) => a,
        None => return Vec::new(),
    };
    let (idx_str, leaf_us) = match after.split_once('_') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx_str.parse::<usize>().is_err() {
        return Vec::new();
    }
    map_block_leaf(idx_str, leaf_us)
}

/// Dispatch the underscore-form leaf onto qwen-image candle target(s).
///
/// Most leaves emit one `Direct` patch. Leaves that have BF16-vs-ComfyUI
/// naming asymmetries (FFN, AdaLN modulation) emit two `Direct` patches
/// — one for each candle key — so the same LoRA wrapper works on both
/// checkpoint formats without a probe step. `attn.qkv` / `attn.add_qkv_proj`
/// emit three `Splat` patches each.
fn map_block_leaf(idx: &str, leaf_us: &str) -> Vec<QwenImageLoraTarget> {
    let block = format!("transformer_blocks.{idx}");
    match leaf_us {
        // ── Image-stream attention: split Q / K / V (Direct) ──────────
        "attn_to_q" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.to_q.weight"),
        }],
        "attn_to_k" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.to_k.weight"),
        }],
        "attn_to_v" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.to_v.weight"),
        }],
        // ── Image-stream attention: fused QKV (Splat × 3) ─────────────
        "attn_qkv" => vec![
            QwenImageLoraTarget::Splat {
                candle_key: format!("{block}.attn.to_q.weight"),
                row_offset: 0,
                row_size: 0,
            },
            QwenImageLoraTarget::Splat {
                candle_key: format!("{block}.attn.to_k.weight"),
                row_offset: 1,
                row_size: 0,
            },
            QwenImageLoraTarget::Splat {
                candle_key: format!("{block}.attn.to_v.weight"),
                row_offset: 2,
                row_size: 0,
            },
        ],
        // ── Image-stream attention output ─────────────────────────────
        "attn_to_out_0" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.to_out.0.weight"),
        }],
        // ── Text-stream attention: split add Q / K / V (Direct) ──────
        "attn_add_q_proj" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.add_q_proj.weight"),
        }],
        "attn_add_k_proj" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.add_k_proj.weight"),
        }],
        "attn_add_v_proj" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.add_v_proj.weight"),
        }],
        // ── Text-stream attention: fused add QKV (Splat × 3, rare) ────
        "attn_add_qkv_proj" => vec![
            QwenImageLoraTarget::Splat {
                candle_key: format!("{block}.attn.add_q_proj.weight"),
                row_offset: 0,
                row_size: 0,
            },
            QwenImageLoraTarget::Splat {
                candle_key: format!("{block}.attn.add_k_proj.weight"),
                row_offset: 1,
                row_size: 0,
            },
            QwenImageLoraTarget::Splat {
                candle_key: format!("{block}.attn.add_v_proj.weight"),
                row_offset: 2,
                row_size: 0,
            },
        ],
        // ── Text-stream attention output ──────────────────────────────
        "attn_to_add_out" => vec![QwenImageLoraTarget::Direct {
            candle_key: format!("{block}.attn.to_add_out.weight"),
        }],
        // ── Image-stream FF: emit BF16 + ComfyUI candidates ───────────
        "ff_net_0_proj" => vec![
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.ff.net.0.proj.weight"),
            },
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.img_mlp.net.0.proj.weight"),
            },
        ],
        "ff_net_2" => vec![
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.ff.net.2.weight"),
            },
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.img_mlp.net.2.weight"),
            },
        ],
        // ── Text-stream FF: emit BF16 + ComfyUI candidates ────────────
        "ff_context_net_0_proj" => vec![
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.ff_context.net.0.proj.weight"),
            },
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.txt_mlp.net.0.proj.weight"),
            },
        ],
        "ff_context_net_2" => vec![
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.ff_context.net.2.weight"),
            },
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.txt_mlp.net.2.weight"),
            },
        ],
        // ── Image-stream AdaLN modulation: BF16 + ComfyUI siblings ────
        "norm1_linear" => vec![
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.norm1.linear.weight"),
            },
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.img_mod.1.weight"),
            },
        ],
        // ── Text-stream AdaLN modulation: BF16 + ComfyUI siblings ─────
        "norm1_context_linear" => vec![
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.norm1_context.linear.weight"),
            },
            QwenImageLoraTarget::Direct {
                candle_key: format!("{block}.txt_mod.1.weight"),
            },
        ],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Patch building.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QwenImageLoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: QwenImageLoraTarget,
    /// Stable hash of the source LoRA path; used as part of the delta
    /// cache key so two LoRAs at the same scale on the same tensor
    /// don't collide. Currently retained for future delta caching;
    /// tests rely on the field being populated.
    #[allow(dead_code)]
    lora_path_hash: u64,
    /// Resolved `(offset, size)` for `Splat` targets. Filled at patch-
    /// build time once `B`'s row count is known.
    resolved_rows: Option<(usize, usize)>,
}

/// A loaded LoRA + its scale + a stable hash of its file path.
pub(crate) struct QwenImageLoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
    pub path_hash: u64,
}

fn resolve_rows(target: &QwenImageLoraTarget, b_rows: usize) -> Option<(usize, usize)> {
    match target {
        QwenImageLoraTarget::Direct { .. } => None,
        QwenImageLoraTarget::Splat {
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

fn build_patches(
    specs: &[QwenImageLoraSpec<'_>],
) -> (HashMap<String, Vec<QwenImageLoraPatch>>, usize) {
    let mut patches: HashMap<String, Vec<QwenImageLoraPatch>> = HashMap::new();
    let mut skipped = 0usize;
    for spec in specs {
        for (lora_stem, layer) in &spec.adapter.layers {
            let targets = map_qwen_image_lora_key(lora_stem);
            if targets.is_empty() {
                tracing::warn!(
                    key = lora_stem.as_str(),
                    "unrecognized Qwen-Image LoRA key, skipping"
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
                    .push(QwenImageLoraPatch {
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

fn compute_delta(patch: &QwenImageLoraPatch, target_dev: &Device) -> candle_core::Result<Tensor> {
    let a = patch.a.to_dtype(DType::F32)?.to_device(target_dev)?;
    let b = patch.b.to_dtype(DType::F32)?.to_device(target_dev)?;
    let computed = b.matmul(&a)?;
    &computed * patch.effective_scale
}

/// Add `delta_full` (or the appropriate row-slice of it) to an F32 base.
fn apply_patch_f32(
    base_f32: &Tensor,
    delta_full: &Tensor,
    patch: &QwenImageLoraPatch,
) -> candle_core::Result<Tensor> {
    match &patch.target {
        QwenImageLoraTarget::Direct { .. } => base_f32 + delta_full,
        QwenImageLoraTarget::Splat { .. } => {
            let (offset, size) = patch
                .resolved_rows
                .expect("Splat patch must have resolved_rows");
            let delta_slice = delta_full.narrow(0, offset, size)?;
            let base_rows = base_f32.dim(0)?;
            if base_rows != size {
                tracing::warn!(
                    base_rows,
                    delta_rows = size,
                    "Qwen-Image LoRA Splat: base row count != delta row count, skipping"
                );
                return Ok(base_f32.clone());
            }
            base_f32 + &delta_slice
        }
    }
}

// ---------------------------------------------------------------------------
// `LoraBackend` — wraps a `SimpleBackend` and merges LoRAs at `vb.get()`.
// ---------------------------------------------------------------------------

struct QwenImageLoraBackend {
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    patches: HashMap<String, Vec<QwenImageLoraPatch>>,
    // Reserved for future delta caching (mirrors flux2/zimage's API for
    // consistency when refactoring; currently unused — Qwen-Image's
    // transformer rebuild cadence is low enough that re-computing
    // `B @ A` on each construction is cheap).
    _delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

impl QwenImageLoraBackend {
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

impl candle_nn::var_builder::SimpleBackend for QwenImageLoraBackend {
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
/// LoRA-merged tensors. The wrapper applies deltas in F32 and casts
/// back to the requested dtype.
pub(crate) fn wrap_backend_with_lora(
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    specs: &[QwenImageLoraSpec<'_>],
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
        "LoRA (Qwen-Image): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    Ok(Box::new(QwenImageLoraBackend {
        inner,
        patches,
        _delta_cache: delta_cache,
    }))
}

/// Build a LoRA-patching `VarBuilder` for the **GGUF** transformer
/// path. Selectively dequantises every patched tensor to F32 on CPU,
/// merges `B @ A · scale` (or the appropriate row-slice for Splat
/// patches), and re-quantises back to the original GGML dtype on the
/// target device. Mirrors `flux::lora::gguf_lora_var_builder` and
/// `sd3::lora::gguf_lora_var_builder`.
pub(crate) fn gguf_lora_var_builder(
    transformer_path: &Path,
    specs: &[QwenImageLoraSpec<'_>],
    device: &Device,
    progress: &ProgressReporter,
    _delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<candle_transformers::quantized_var_builder::VarBuilder> {
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
        "LoRA (Qwen-Image GGUF): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let gguf_bytes_total: u64 = std::fs::metadata(transformer_path)
        .map(|m| m.len())
        .unwrap_or(0);
    progress.weight_load("Qwen-Image transformer (GGUF)", 0, gguf_bytes_total);
    for (i, tensor_name) in content.tensor_infos.keys().enumerate() {
        let qtensor = content.tensor(&mut file, tensor_name, device)?;
        data.insert(tensor_name.clone(), Arc::new(qtensor));
        let approx_bytes = gguf_bytes_total * (i as u64 + 1) / total_tensors as u64;
        progress.weight_load(
            "Qwen-Image transformer (GGUF)",
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

        // GGUF may not contain a key we emit (e.g. the BF16-only
        // sibling of an FFN/AdaLN dual emit). Skipping silently is
        // correct — the partner key picks up the patch.
        let tensor_key = if data.contains_key(candle_key) {
            candle_key.clone()
        } else {
            tracing::debug!(
                key = candle_key.as_str(),
                "Qwen-Image LoRA target tensor not found in GGUF, skipping"
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

        for patch in layer_patches.iter() {
            let matmul_dev = if on_gpu { device } else { &Device::Cpu };
            let a = patch.a.to_dtype(DType::F32)?.to_device(matmul_dev)?;
            let b = patch.b.to_dtype(DType::F32)?.to_device(matmul_dev)?;
            let delta_full = (b.matmul(&a)? * patch.effective_scale)?.to_device(&Device::Cpu)?;

            t = match &patch.target {
                QwenImageLoraTarget::Direct { .. } => (&t + &delta_full)?,
                QwenImageLoraTarget::Splat { .. } => {
                    let (offset, size) = patch
                        .resolved_rows
                        .expect("Splat patch must have resolved_rows");
                    let delta_slice = delta_full.narrow(0, offset, size)?;
                    let base_rows = t.dim(0)?;
                    if base_rows != size {
                        tracing::warn!(
                            base_rows,
                            delta_rows = size,
                            "Qwen-Image GGUF Splat: base row count != delta row count, skipping"
                        );
                        t
                    } else {
                        (&t + &delta_slice)?
                    }
                }
            };
        }

        let merged_q = QTensor::quantize(&t, orig_dtype)?;
        data.insert(tensor_key, Arc::new(merged_q));
        applied += 1;
        if i % 16 == 0 {
            progress.info(&format!(
                "Qwen-Image LoRA GGUF merge: {}/{} tensors",
                applied, lora_total,
            ));
        }
    }

    if on_gpu {
        device.synchronize()?;
    }

    Ok(candle_transformers::quantized_var_builder::VarBuilder::from_qtensors(data, device))
}

/// Build [`QwenImageLoraSpec`]s by loading every adapter through the
/// shared parsed-LoRA cache (`crate::flux::lora::get_or_load_adapter`).
/// Returns the `Arc`s so the caller can hold them for the lifetime of
/// the spec slice.
pub(crate) fn load_lora_adapters(
    loras: &[mold_core::LoraWeight],
    progress: &ProgressReporter,
) -> Result<Vec<Arc<LoraAdapter>>> {
    loras
        .iter()
        .map(|w| {
            progress.info("Loading Qwen-Image LoRA adapter");
            let adapter = get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "Qwen-Image LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::lora::LoraLayer;
    use safetensors::tensor::TensorView;

    // ── map_qwen_image_lora_key — Direct (split) attention leaves ─────────

    /// Pin every split-attention leaf for the image stream.
    #[test]
    fn peft_canonical_split_image_attn_leaves_resolve_direct() {
        for (leaf_dotted, expected) in [
            ("attn.to_q", "transformer_blocks.0.attn.to_q.weight"),
            ("attn.to_k", "transformer_blocks.0.attn.to_k.weight"),
            ("attn.to_v", "transformer_blocks.0.attn.to_v.weight"),
            ("attn.to_out.0", "transformer_blocks.0.attn.to_out.0.weight"),
        ] {
            let stem = format!("transformer_blocks.0.{leaf_dotted}");
            let targets = map_qwen_image_lora_key(&stem);
            assert_eq!(targets.len(), 1, "leaf={leaf_dotted}");
            match &targets[0] {
                QwenImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf_dotted}");
                }
                _ => panic!("expected Direct for {leaf_dotted}"),
            }
        }
    }

    /// Pin every split-attention leaf for the text stream.
    #[test]
    fn peft_canonical_split_text_attn_leaves_resolve_direct() {
        for (leaf_dotted, expected) in [
            (
                "attn.add_q_proj",
                "transformer_blocks.0.attn.add_q_proj.weight",
            ),
            (
                "attn.add_k_proj",
                "transformer_blocks.0.attn.add_k_proj.weight",
            ),
            (
                "attn.add_v_proj",
                "transformer_blocks.0.attn.add_v_proj.weight",
            ),
            (
                "attn.to_add_out",
                "transformer_blocks.0.attn.to_add_out.weight",
            ),
        ] {
            let stem = format!("transformer_blocks.0.{leaf_dotted}");
            let targets = map_qwen_image_lora_key(&stem);
            assert_eq!(targets.len(), 1, "leaf={leaf_dotted}");
            match &targets[0] {
                QwenImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf_dotted}");
                }
                _ => panic!("expected Direct for {leaf_dotted}"),
            }
        }
    }

    // ── map_qwen_image_lora_key — fused QKV (Splat × 3) ─────────────────

    /// Image-stream fused QKV must splat into three Splat targets in
    /// canonical Q→K→V order with the thirds-split sentinel.
    #[test]
    fn fused_image_qkv_splits_into_three_splat_targets() {
        let targets = map_qwen_image_lora_key("transformer_blocks.3.attn.qkv");
        assert_eq!(targets.len(), 3);
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "transformer_blocks.3.attn.to_q.weight",
                "transformer_blocks.3.attn.to_k.weight",
                "transformer_blocks.3.attn.to_v.weight",
            ]
        );
        for (i, t) in targets.iter().enumerate() {
            match t {
                QwenImageLoraTarget::Splat {
                    row_offset,
                    row_size,
                    ..
                } => {
                    assert_eq!(*row_offset, i, "component index Q→K→V");
                    assert_eq!(*row_size, 0, "thirds-split sentinel");
                }
                _ => panic!("expected Splat for component {i}"),
            }
        }
    }

    /// Text-stream fused add-QKV (rare but observed in some trainers
    /// that fuse both streams) must splat onto add_{q,k,v}_proj.
    #[test]
    fn fused_text_add_qkv_splits_into_three_add_proj_splat_targets() {
        let targets = map_qwen_image_lora_key("transformer_blocks.7.attn.add_qkv_proj");
        assert_eq!(targets.len(), 3);
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "transformer_blocks.7.attn.add_q_proj.weight",
                "transformer_blocks.7.attn.add_k_proj.weight",
                "transformer_blocks.7.attn.add_v_proj.weight",
            ]
        );
    }

    // ── map_qwen_image_lora_key — FF + AdaLN dual emit ────────────────────

    /// Image-stream FF must emit BOTH the BF16 (`ff.net.*`) and ComfyUI
    /// (`img_mlp.net.*`) candle keys so the same wrapper covers either
    /// checkpoint format without a probe step.
    #[test]
    fn image_ff_emits_both_bf16_and_comfyui_candle_keys() {
        for (leaf, bf16_key, comfy_key) in [
            (
                "ff.net.0.proj",
                "transformer_blocks.5.ff.net.0.proj.weight",
                "transformer_blocks.5.img_mlp.net.0.proj.weight",
            ),
            (
                "ff.net.2",
                "transformer_blocks.5.ff.net.2.weight",
                "transformer_blocks.5.img_mlp.net.2.weight",
            ),
        ] {
            let stem = format!("transformer_blocks.5.{leaf}");
            let targets = map_qwen_image_lora_key(&stem);
            assert_eq!(targets.len(), 2, "leaf={leaf} dual emit");
            let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
            assert!(keys.contains(&bf16_key), "missing BF16 key for {leaf}");
            assert!(keys.contains(&comfy_key), "missing ComfyUI key for {leaf}");
        }
    }

    /// Text-stream FF mirrors the dual-emit pattern (`ff_context` vs
    /// `txt_mlp`).
    #[test]
    fn text_ff_emits_both_bf16_and_comfyui_candle_keys() {
        let stem = "transformer_blocks.5.ff_context.net.0.proj";
        let targets = map_qwen_image_lora_key(stem);
        assert_eq!(targets.len(), 2);
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert!(keys.contains(&"transformer_blocks.5.ff_context.net.0.proj.weight"));
        assert!(keys.contains(&"transformer_blocks.5.txt_mlp.net.0.proj.weight"));
    }

    /// AdaLN modulation: BF16 (`norm1.linear`) vs ComfyUI (`img_mod.1`).
    #[test]
    fn image_adaln_modulation_emits_both_bf16_and_comfyui_keys() {
        let stem = "transformer_blocks.0.norm1.linear";
        let targets = map_qwen_image_lora_key(stem);
        assert_eq!(targets.len(), 2);
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert!(keys.contains(&"transformer_blocks.0.norm1.linear.weight"));
        assert!(keys.contains(&"transformer_blocks.0.img_mod.1.weight"));
    }

    /// Text-stream AdaLN modulation mirrors the dual-emit pattern.
    #[test]
    fn text_adaln_modulation_emits_both_bf16_and_comfyui_keys() {
        let stem = "transformer_blocks.0.norm1_context.linear";
        let targets = map_qwen_image_lora_key(stem);
        assert_eq!(targets.len(), 2);
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert!(keys.contains(&"transformer_blocks.0.norm1_context.linear.weight"));
        assert!(keys.contains(&"transformer_blocks.0.txt_mod.1.weight"));
    }

    // ── Kohya / sd-scripts naming ────────────────────────────────────────

    /// Pin the Kohya (`lora_unet_*`) split-attention leaves that the
    /// majority of Civitai LoRAs would ship.
    #[test]
    fn kohya_lora_unet_split_image_attn_leaves_resolve() {
        for (leaf_us, expected) in [
            ("attn_to_q", "transformer_blocks.0.attn.to_q.weight"),
            ("attn_to_k", "transformer_blocks.0.attn.to_k.weight"),
            ("attn_to_v", "transformer_blocks.0.attn.to_v.weight"),
            ("attn_to_out_0", "transformer_blocks.0.attn.to_out.0.weight"),
        ] {
            let key = format!("lora_unet_transformer_blocks_0_{leaf_us}");
            let targets = map_qwen_image_lora_key(&key);
            assert_eq!(targets.len(), 1, "leaf={leaf_us}");
            match &targets[0] {
                QwenImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf_us}");
                }
                _ => panic!("expected Direct for {leaf_us}"),
            }
        }
    }

    /// Some Kohya trainers emit the abbreviated form `lora_unet_blocks_*`
    /// (dropping the `transformer_` prefix). Accept both.
    #[test]
    fn kohya_abbreviated_blocks_prefix_resolves() {
        let key = "lora_unet_blocks_0_attn_to_q";
        let targets = map_qwen_image_lora_key(key);
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            QwenImageLoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "transformer_blocks.0.attn.to_q.weight");
            }
            _ => panic!("expected Direct"),
        }
    }

    /// Kohya fused-QKV must splat across the three split-Q/K/V tensors.
    #[test]
    fn kohya_fused_qkv_splits_into_three_splat() {
        let targets = map_qwen_image_lora_key("lora_unet_transformer_blocks_3_attn_qkv");
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "transformer_blocks.3.attn.to_q.weight",
                "transformer_blocks.3.attn.to_k.weight",
                "transformer_blocks.3.attn.to_v.weight",
            ]
        );
    }

    // ── Prefix stripping ─────────────────────────────────────────────────

    #[test]
    fn peft_canonical_strips_optional_prefixes() {
        for stem in [
            "model.diffusion_model.transformer_blocks.7.attn.to_q",
            "diffusion_model.transformer_blocks.7.attn.to_q",
            "transformer.transformer_blocks.7.attn.to_q",
            "transformer_blocks.7.attn.to_q",
        ] {
            let targets = map_qwen_image_lora_key(stem);
            assert_eq!(targets.len(), 1, "stem={stem}");
            match &targets[0] {
                QwenImageLoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, "transformer_blocks.7.attn.to_q.weight");
                }
                _ => panic!("expected Direct for {stem}"),
            }
        }
    }

    // ── Negative coverage ────────────────────────────────────────────────

    #[test]
    fn unknown_or_te_keys_return_empty_vec() {
        // Non-attention leaf inside a real block still returns empty —
        // the dispatch is a closed set.
        assert!(map_qwen_image_lora_key("transformer_blocks.0.unknown_thing").is_empty());
        // Text-encoder LoRAs (lora_te_*) are not the transformer's
        // concern.
        assert!(map_qwen_image_lora_key("lora_te_text_model_layer_0_attn_q").is_empty());
        // Garbage.
        assert!(map_qwen_image_lora_key("garbage").is_empty());
        // Right prefix, missing index segment.
        assert!(map_qwen_image_lora_key("transformer_blocks").is_empty());
        assert!(map_qwen_image_lora_key("transformer_blocks.notanindex.attn.to_q").is_empty());
    }

    // ── resolve_rows ─────────────────────────────────────────────────────

    #[test]
    fn resolve_rows_thirds_for_splat_with_zero_size() {
        let target = QwenImageLoraTarget::Splat {
            candle_key: "x".into(),
            row_offset: 1,
            row_size: 0,
        };
        let rows = resolve_rows(&target, 3 * 8).unwrap();
        assert_eq!(rows, (8, 8), "component 1 of 3 in a 24-row B");
    }

    #[test]
    fn resolve_rows_direct_target_is_none() {
        let target = QwenImageLoraTarget::Direct {
            candle_key: "x".into(),
        };
        assert!(resolve_rows(&target, 12).is_none());
    }

    // ── apply_patch_f32 — math correctness ───────────────────────────────

    #[test]
    fn apply_patch_direct_adds_full_delta() {
        let dev = Device::Cpu;
        let base = Tensor::full(2.0f32, (4, 3), &dev).unwrap();
        let delta = Tensor::full(0.5f32, (4, 3), &dev).unwrap();
        let patch = QwenImageLoraPatch {
            a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            effective_scale: 1.0,
            target: QwenImageLoraTarget::Direct {
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

    /// Load-bearing: a fused-QKV delta must hit the *right* third of
    /// each split tensor (Q = rows 0..h, K = rows h..2h, V = rows 2h..3h).
    #[test]
    fn apply_patch_splat_uses_correct_third_of_delta() {
        let dev = Device::Cpu;
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
            let mut patch = QwenImageLoraPatch {
                a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                effective_scale: 1.0,
                target: QwenImageLoraTarget::Splat {
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

    /// `apply_patch_f32` Splat must refuse a base whose row count
    /// disagrees with the delta third (corrupt LoRA / wrong target).
    #[test]
    fn apply_patch_splat_skips_when_dim_mismatches() {
        let dev = Device::Cpu;
        let h = 3;
        let in_dim = 2;
        let delta_full = Tensor::full(0.7f32, (3 * h, in_dim), &dev).unwrap();
        let wrong_base = Tensor::full(5.0f32, (h + 1, in_dim), &dev).unwrap();
        let mut patch = QwenImageLoraPatch {
            a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            effective_scale: 1.0,
            target: QwenImageLoraTarget::Splat {
                candle_key: "x".into(),
                row_offset: 0,
                row_size: 0,
            },
            lora_path_hash: 0,
            resolved_rows: None,
        };
        patch.resolved_rows = resolve_rows(&patch.target, 3 * h);
        let merged = apply_patch_f32(&wrong_base, &delta_full, &patch).unwrap();
        let vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| (v - 5.0).abs() < 1e-6));
    }

    // ── build_patches — adapter wiring ───────────────────────────────────

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
        let adapter = synthetic_kohya_adapter("lora_unet_transformer_blocks_0_attn_qkv", 6);
        let specs = [QwenImageLoraSpec {
            adapter: &adapter,
            scale: 0.7,
            path_hash: 0xCAFE,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert_eq!(skipped, 0);
        assert_eq!(patches.len(), 3);
        for k in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ] {
            assert!(patches.contains_key(k), "missing {k}");
            let bucket = &patches[k];
            assert_eq!(bucket.len(), 1);
            // resolved_rows = (component * (6/3), 2).
            assert_eq!(bucket[0].resolved_rows.unwrap().1, 2);
        }
    }

    #[test]
    fn build_patches_alpha_normalises_scale() {
        let mut adapter =
            synthetic_kohya_adapter("lora_unet_transformer_blocks_0_attn_to_out_0", 4);
        adapter
            .layers
            .get_mut("lora_unet_transformer_blocks_0_attn_to_out_0")
            .unwrap()
            .alpha = Some(4.0);
        let specs = [QwenImageLoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0,
        }];
        let (patches, _) = build_patches(&specs);
        let bucket = &patches["transformer_blocks.0.attn.to_out.0.weight"];
        let s = bucket[0].effective_scale;
        assert!(
            (s - 1.0).abs() < 1e-9,
            "effective scale = user(0.5) * alpha(4) / rank(2) = 1.0, got {s}",
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
        let specs = [QwenImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];
        let (patches, skipped) = build_patches(&specs);
        assert!(patches.is_empty());
        assert_eq!(skipped, 1);
    }

    /// Multi-LoRA stacking: two specs targeting the same tensor produce
    /// two patches in the same bucket, each tagged with its own
    /// `lora_path_hash` so the delta cache keys don't collide.
    #[test]
    fn build_patches_two_specs_stack_on_same_target() {
        let a1 = synthetic_kohya_adapter("lora_unet_transformer_blocks_0_attn_to_q", 4);
        let a2 = synthetic_kohya_adapter("lora_unet_transformer_blocks_0_attn_to_q", 4);
        let specs = [
            QwenImageLoraSpec {
                adapter: &a1,
                scale: 1.0,
                path_hash: 0xAA,
            },
            QwenImageLoraSpec {
                adapter: &a2,
                scale: 1.0,
                path_hash: 0xBB,
            },
        ];
        let (patches, _) = build_patches(&specs);
        let bucket = &patches["transformer_blocks.0.attn.to_q.weight"];
        assert_eq!(bucket.len(), 2);
        assert_eq!(bucket[0].lora_path_hash, 0xAA);
        assert_eq!(bucket[1].lora_path_hash, 0xBB);
    }

    /// Dual-emit FF: a single LoRA layer creates two patch buckets (one
    /// per candle key candidate). Whichever the underlying VarBuilder
    /// actually serves wins; the other is dormant.
    #[test]
    fn build_patches_ff_dual_emit_produces_two_buckets() {
        let adapter = synthetic_kohya_adapter("lora_unet_transformer_blocks_0_ff_net_0_proj", 4);
        let specs = [QwenImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];
        let (patches, _) = build_patches(&specs);
        assert!(patches.contains_key("transformer_blocks.0.ff.net.0.proj.weight"));
        assert!(patches.contains_key("transformer_blocks.0.img_mlp.net.0.proj.weight"));
        assert_eq!(patches.len(), 2);
    }

    // ── End-to-end via SimpleBackend wrapper ─────────────────────────────

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

    /// End-to-end Splat: a fused-QKV adapter must land its three thirds
    /// on the three split candle tensors.
    #[test]
    fn end_to_end_fused_qkv_splat_lands_on_three_tensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base.safetensors");

        let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for k in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ] {
            entries.push((k.to_string(), vec![2, 8], vec![0.0; 16]));
        }
        write_synthetic_safetensors_with_data(&path, &entries);

        // B = 6×2 of ones, A = 2×8 of ones. B@A = 6×8 of 2. Three thirds
        // each constant 2. Base = 0 → merged = 2.
        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(1.0f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_transformer_blocks_0_attn_qkv".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [QwenImageLoraSpec {
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
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ] {
            let t = wrapped.get_unchecked(k, DType::F32, &dev).expect("get");
            let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
                "{k}: expected constant 2.0 (= 0 + B@A row-third), got {vals:?}",
            );
        }
    }

    /// End-to-end Direct: a leaf-named LoRA (`attn.to_q`) merges via the
    /// additive `base + B@A·scale` path with no slicing.
    #[test]
    fn end_to_end_direct_merge_on_attn_to_q() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base_direct.safetensors");
        write_synthetic_safetensors_with_data(
            &path,
            &[(
                "transformer_blocks.0.attn.to_q.weight".to_string(),
                vec![6, 8],
                vec![1.0; 48],
            )],
        );

        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(0.5f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_transformer_blocks_0_attn_to_q".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [QwenImageLoraSpec {
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
            .get_unchecked("transformer_blocks.0.attn.to_q.weight", DType::F32, &dev)
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
            "expected 2.0 (= 1 + 1), got {vals:?}",
        );
    }

    /// Dual-emit FF leaves: the LoRA wrapper picks the candle key that
    /// the inner backend actually serves. Tests both axes — BF16
    /// (`ff.net.*`) and ComfyUI (`img_mlp.net.*`) — succeed with a
    /// single `lora_unet_..._ff_net_0_proj` key.
    #[test]
    fn end_to_end_ff_dual_emit_resolves_either_naming() {
        for inner_key in [
            "transformer_blocks.0.ff.net.0.proj.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir
                .path()
                .join(format!("base-{}.safetensors", inner_key.len()));
            write_synthetic_safetensors_with_data(
                &path,
                &[(inner_key.to_string(), vec![4, 4], vec![0.0; 16])],
            );

            let dev = Device::Cpu;
            let a = Tensor::full(1.0f32, (2, 4), &dev).unwrap();
            let b = Tensor::full(1.0f32, (4, 2), &dev).unwrap();
            let mut layers = HashMap::new();
            layers.insert(
                "lora_unet_transformer_blocks_0_ff_net_0_proj".to_string(),
                LoraLayer { a, b, alpha: None },
            );
            let adapter = LoraAdapter { layers, rank: 2 };
            let specs = [QwenImageLoraSpec {
                adapter: &adapter,
                scale: 1.0,
                path_hash: 0,
            }];

            let st = unsafe {
                candle_core::safetensors::MmapedSafetensors::multi(&[path]).expect("mmap")
            };
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
            let inner: Box<dyn candle_nn::var_builder::SimpleBackend> =
                Box::new(MmapBackend { st });
            let progress = ProgressReporter::default();
            let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");

            // B@A entry = 2. base(0) + 2 = 2.
            let t = wrapped
                .get_unchecked(inner_key, DType::F32, &dev)
                .expect("get");
            let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
                "{inner_key}: expected 2.0 (= 0 + B@A), got {vals:?}",
            );
        }
    }

    /// VAE-tiling sanity check: the LoRA wrapper targets the
    /// transformer only — VAE keys must pass through unchanged
    /// regardless of `MOLD_VAE_TILED`. This is a contract test on the
    /// patch table: there are no VAE entries, so any VAE tensor name
    /// returned by the inner backend bypasses the merge path.
    #[test]
    fn vae_tensors_pass_through_lora_wrapper_unchanged() {
        let dev = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "vae.decoder.up_blocks.0.resnets.0.conv1.weight".to_string(),
            Tensor::full(7.0f32, (3, 3), &dev).unwrap(),
        );
        let adapter = synthetic_kohya_adapter("lora_unet_transformer_blocks_0_attn_to_q", 4);
        let specs = [QwenImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(tensors);
        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");
        let t = wrapped
            .get_unchecked(
                "vae.decoder.up_blocks.0.resnets.0.conv1.weight",
                DType::F32,
                &dev,
            )
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 7.0).abs() < 1e-6),
            "VAE tensor must pass through unchanged regardless of MOLD_VAE_TILED",
        );
    }

    /// End-to-end via `DenseVarBuilder::from_tensors` — the GGUF path
    /// ultimately constructs this shape after dequantising. Confirms the
    /// LoRA wrapper behaves identically regardless of the inner backend
    /// impl.
    #[test]
    fn end_to_end_dense_var_builder_path_picks_up_lora() {
        let dev = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "transformer_blocks.0.attn.to_q.weight".to_string(),
            Tensor::full(1.0f32, (4, 4), &dev).unwrap(),
        );

        let a = Tensor::full(1.0f32, (2, 4), &dev).unwrap();
        let b = Tensor::full(0.5f32, (4, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_transformer_blocks_0_attn_to_q".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [QwenImageLoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0,
        }];

        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(tensors);
        let progress = ProgressReporter::default();
        let wrapped = wrap_backend_with_lora(inner, &specs, &progress, None).expect("wrap");

        let t = wrapped
            .get_unchecked("transformer_blocks.0.attn.to_q.weight", DType::F32, &dev)
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // B@A per entry = 2 * 0.5 * 1 = 1. base 1 + 1 = 2.
        assert!(vals.iter().all(|v| (v - 2.0).abs() < 1e-5));
    }

    // ── wrap_backend_with_lora — input validation ────────────────────────

    #[test]
    fn wrap_backend_with_no_specs_returns_error() {
        let empty: HashMap<String, Tensor> = HashMap::new();
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = Box::new(empty);
        let progress = ProgressReporter::default();
        match wrap_backend_with_lora(inner, &[], &progress, None) {
            Ok(_) => panic!("expected error for empty spec list"),
            Err(e) => assert!(
                e.to_string().contains("no LoraSpecs"),
                "expected 'no LoraSpecs' message, got: {e}",
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

    /// Real-LoRA smoke test placeholder. Qwen-Image LoRAs in the wild
    /// are still rare — when one is verified, the URL goes here and the
    /// `#[ignore]` lifts. Currently parked as a documentation marker.
    ///
    /// Civitai search anchor: <https://civitai.com/?type=lora&baseModel=Qwen-Image>
    #[test]
    #[ignore = "no on-disk Qwen-Image LoRA available; document a Civitai URL when one ships"]
    fn civitai_smoke_placeholder() {}
}
