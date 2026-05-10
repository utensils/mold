//! Flux.2 LoRA support — diffusers + Kohya/sd-scripts naming, BF16 + GGUF paths.
//!
//! Mirrors `crate::flux::lora` (FLUX.1) but adapted to Flux.2's tensor layout:
//!
//! | Aspect | FLUX.1 (BFL-native) | Flux.2 BF16 (diffusers) | Flux.2 GGUF (BFL-native) |
//! |---|---|---|---|
//! | Double-block QKV  | fused `img_attn.qkv` (3*h) | **separate** `to_q` / `to_k` / `to_v` (h) | fused `img_attn.qkv` (3*h) |
//! | Double-block MLP  | `img_mlp.{0,2}` | `ff.{linear_in,linear_out}` | `img_mlp.{0,2}` |
//! | Single-block fused| `linear1` `[Q,K,V,MLP]` (3*h + mlp) | `to_qkv_mlp_proj` `[Q,K,V,MLP_gate,MLP_val]` (3*h + 2*mlp) | `linear1` (same as BF16 layout) |
//! | Top-level         | `img_in`, `txt_in`, … | `x_embedder`, `context_embedder`, … | `img_in`, `txt_in`, … |
//!
//! Flux.2 LoRAs in the wild (Civitai cv:2682864 / cv:2742432) use **BFL-native
//! module names** (`double_blocks.{i}.img_attn.qkv`, `single_blocks.{i}.linear1`)
//! either with the PEFT canonical suffix (`.lora_A.weight` / `.lora_B.weight`)
//! or the Kohya/sd-scripts suffix (`lora_unet_*` prefix + `.lora_down.weight`
//! / `.lora_up.weight`). We accept both. The single-block fused-output layout
//! `[Q, K, V, MLP_gate, MLP_val]` (BF16/diffusers) is identical to BFL `linear1`
//! at the byte level — the FP4-quantised single-file checkpoints store these
//! under the `linear1` BFL name and the diffusers transformer reads them via
//! `SingleFileBackend`'s rename to `to_qkv_mlp_proj`. So a `linear1` Kohya
//! LoRA targets the same fused weight regardless of path; only the **candle
//! key** the model asks for differs.
//!
//! # Path A: BF16 → diffusers candle keys
//!
//! Used when the transformer is loaded via `Flux2Transformer::new` (sharded HF
//! diffusers safetensors, or BFL-native single-file remapped through
//! `SingleFileBackend`). The model asks for `transformer_blocks.{i}.attn.to_q.weight`
//! etc. — **separate** Q/K/V tensors. A single `lora_unet_double_blocks_{i}_img_attn_qkv`
//! LoRA whose `B` is shape `[3*h, rank]` is split into three patches: rows
//! `0..h` → `to_q`, `h..2h` → `to_k`, `2h..3h` → `to_v`. Each gets a `Splat`
//! target.
//!
//! # Path B: GGUF → BFL-native candle keys
//!
//! Used by `QuantizedFlux2Transformer`. The model asks for
//! `double_blocks.{i}.img_attn.qkv.weight` — fused. The same Kohya LoRA is a
//! single `Direct` target that adds `B @ A * scale` to the whole fused tensor.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

#[cfg(test)]
use crate::flux::lora::{classify_lora_key, LoraDirection};
use crate::flux::lora::{get_or_load_adapter, LoraAdapter, LoraDeltaCache};
use crate::progress::ProgressReporter;

// ---------------------------------------------------------------------------
// Public path-hash helper — used by the pipeline to seed `LoraSpec`s.
// ---------------------------------------------------------------------------

/// Stable hash of a LoRA file path. Mirrors the FLUX helper of the same name —
/// we keep an independent copy to avoid taking a dependency on `flux::pipeline`
/// internals.
pub(crate) fn lora_path_hash(path: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Target descriptors
// ---------------------------------------------------------------------------

/// How a LoRA layer's `B @ A * scale` delta lands on a candle tensor.
#[derive(Debug, Clone)]
pub(crate) enum Flux2LoraTarget {
    /// Apply the entire delta to the entire tensor (additive merge).
    Direct { candle_key: String },
    /// Apply a row-slice of the delta (`row_offset..row_offset+row_size` of `B`)
    /// to the entire candle tensor. Used in the BF16 path where a single
    /// fused-QKV Kohya LoRA splits across three separate Q/K/V tensors.
    Splat {
        candle_key: String,
        row_offset: usize,
        row_size: usize,
    },
}

impl Flux2LoraTarget {
    fn candle_key(&self) -> &str {
        match self {
            Self::Direct { candle_key } => candle_key,
            Self::Splat { candle_key, .. } => candle_key,
        }
    }
}

/// Which candle-key namespace the model construction will ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flux2KeySpace {
    /// Diffusers names (`transformer_blocks.{i}.attn.to_q.weight`, …) — used by
    /// `Flux2Transformer::new` with sharded HF safetensors **or** BFL-native
    /// single-file checkpoints (the latter remap BFL→diffusers via
    /// `SingleFileBackend`).
    Diffusers,
    /// BFL-native names (`double_blocks.{i}.img_attn.qkv.weight`, …) — used by
    /// `QuantizedFlux2Transformer` (GGUF) and by the legacy BFL transformer.
    Bfl,
}

// ---------------------------------------------------------------------------
// Key mapping: LoRA stem → candle target(s)
// ---------------------------------------------------------------------------

/// Strip leading `transformer.` and the optional `diffusion_model.` /
/// `model.diffusion_model.` checkpoint prefixes — the LoRA layer-stems we
/// care about start at `double_blocks.*` / `single_blocks.*` /
/// `transformer_blocks.*` / `single_transformer_blocks.*`.
fn strip_known_prefixes(stem: &str) -> &str {
    let s = stem.strip_prefix("model.").unwrap_or(stem);
    let s = s.strip_prefix("diffusion_model.").unwrap_or(s);
    s.strip_prefix("transformer.").unwrap_or(s)
}

/// Map a LoRA layer stem (with the `.lora_A`/`.lora_down`/etc. suffix already
/// stripped) onto candle targets in the requested key space. Returns an empty
/// vec for keys we don't recognise — the caller logs and skips.
///
/// Recognised input shapes:
/// - `lora_unet_double_blocks_{i}_<leaf>`             (Kohya, BFL module names)
/// - `lora_unet_single_blocks_{i}_<leaf>`             (Kohya, BFL module names)
/// - `(diffusion_model.|model.diffusion_model.)?double_blocks.{i}.<leaf>`  (PEFT-canonical, BFL module names)
/// - `(diffusion_model.|model.diffusion_model.)?single_blocks.{i}.<leaf>`  (PEFT-canonical, BFL module names)
/// - `transformer.transformer_blocks.{i}.<leaf>`      (PEFT-canonical, diffusers module names)
/// - `transformer.single_transformer_blocks.{i}.<leaf>` (PEFT-canonical, diffusers module names)
pub(crate) fn map_flux2_lora_key(raw_stem: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    // Kohya prefix: `lora_unet_*`. Underlying module names are BFL-native.
    if let Some(rest) = raw_stem.strip_prefix("lora_unet_") {
        return map_kohya_unet_key(rest, space);
    }

    // PEFT canonical suffix shapes: stem may carry `transformer.` / `model.` /
    // `diffusion_model.` prefixes. Strip them, then route on the leading
    // module-path token.
    let stem = strip_known_prefixes(raw_stem);

    if let Some(rest) = stem.strip_prefix("double_blocks.") {
        // BFL-native module path (e.g. `double_blocks.0.img_attn.qkv`).
        return map_bfl_double_block(rest, space);
    }
    if let Some(rest) = stem.strip_prefix("single_blocks.") {
        return map_bfl_single_block(rest, space);
    }
    if let Some(rest) = stem.strip_prefix("transformer_blocks.") {
        // Diffusers module path (e.g. `transformer_blocks.0.attn.to_q`).
        return map_diffusers_double_block(rest, space);
    }
    if let Some(rest) = stem.strip_prefix("single_transformer_blocks.") {
        return map_diffusers_single_block(rest, space);
    }
    Vec::new()
}

/// Kohya leaf forms emitted by FLUX/Flux.2 trainers. Mirrors what the cv:*
/// flux2 LoRAs ship: every target is an integral BFL-native layer.
fn map_kohya_unet_key(rest: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    if let Some(after) = rest.strip_prefix("double_blocks_") {
        let (idx_str, suffix) = match after.split_once('_') {
            Some(p) => p,
            None => return Vec::new(),
        };
        if idx_str.parse::<usize>().is_err() {
            return Vec::new();
        }
        return kohya_double_block(idx_str, suffix, space);
    }
    if let Some(after) = rest.strip_prefix("single_blocks_") {
        let (idx_str, suffix) = match after.split_once('_') {
            Some(p) => p,
            None => return Vec::new(),
        };
        if idx_str.parse::<usize>().is_err() {
            return Vec::new();
        }
        return kohya_single_block(idx_str, suffix, space);
    }
    Vec::new()
}

/// Kohya double-block leaves observed in the wild:
/// `img_attn_qkv`, `img_attn_proj`, `img_mlp_0`, `img_mlp_2`, `img_mod_lin`,
/// `txt_attn_qkv`, `txt_attn_proj`, `txt_mlp_0`, `txt_mlp_2`, `txt_mod_lin`.
fn kohya_double_block(idx: &str, leaf: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    match (leaf, space) {
        // ── img_attn fused QKV ────────────────────────────────────────────
        ("img_attn_qkv", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_attn.qkv.weight"),
        }],
        ("img_attn_qkv", Flux2KeySpace::Diffusers) => {
            splat_qkv(idx, "attn.to_q", "attn.to_k", "attn.to_v")
        }
        // ── txt_attn fused QKV ────────────────────────────────────────────
        ("txt_attn_qkv", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_attn.qkv.weight"),
        }],
        ("txt_attn_qkv", Flux2KeySpace::Diffusers) => {
            splat_qkv(idx, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")
        }
        // ── img output proj ──────────────────────────────────────────────
        ("img_attn_proj", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_attn.proj.weight"),
        }],
        ("img_attn_proj", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_out.0.weight"),
        }],
        // ── txt output proj ──────────────────────────────────────────────
        ("txt_attn_proj", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_attn.proj.weight"),
        }],
        ("txt_attn_proj", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_add_out.weight"),
        }],
        // ── img MLP (in / out) ───────────────────────────────────────────
        ("img_mlp_0", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_mlp.0.weight"),
        }],
        ("img_mlp_0", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff.linear_in.weight"),
        }],
        ("img_mlp_2", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_mlp.2.weight"),
        }],
        ("img_mlp_2", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff.linear_out.weight"),
        }],
        // ── txt MLP (in / out) ───────────────────────────────────────────
        ("txt_mlp_0", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_mlp.0.weight"),
        }],
        ("txt_mlp_0", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff_context.linear_in.weight"),
        }],
        ("txt_mlp_2", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_mlp.2.weight"),
        }],
        ("txt_mlp_2", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff_context.linear_out.weight"),
        }],
        // ── modulation lin (shared across blocks in Flux.2 — the per-block
        //    `img_mod_lin`/`txt_mod_lin` LoRA has no BFL/diffusers home in
        //    this transformer because modulation is shared. Skip silently
        //    rather than crash a load that includes them by accident).
        ("img_mod_lin" | "txt_mod_lin", _) => Vec::new(),
        _ => Vec::new(),
    }
}

/// Kohya single-block leaves: `linear1` (fused), `linear2`,
/// `modulation_lin` (shared, no per-block target in Flux.2).
fn kohya_single_block(idx: &str, leaf: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    match (leaf, space) {
        // The single-block fused weight has identical [Q, K, V, MLP_gate,
        // MLP_val] layout in BF16 (`to_qkv_mlp_proj`) and BFL/GGUF (`linear1`).
        ("linear1", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_blocks.{idx}.linear1.weight"),
        }],
        ("linear1", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_transformer_blocks.{idx}.attn.to_qkv_mlp_proj.weight"),
        }],
        ("linear2", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_blocks.{idx}.linear2.weight"),
        }],
        ("linear2", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_transformer_blocks.{idx}.attn.to_out.weight"),
        }],
        ("modulation_lin", _) => Vec::new(),
        _ => Vec::new(),
    }
}

/// Three sliced targets for a single fused-QKV Kohya LoRA: rows
/// `0..h` → Q, `h..2h` → K, `2h..3h` → V. The actual `h` is derived
/// from the loaded `B` tensor's row count at patch-build time, so we
/// emit `Splat { row_offset, row_size: 0 }` as a sentinel meaning
/// "thirds of B".
fn splat_qkv(idx: &str, q_leaf: &str, k_leaf: &str, v_leaf: &str) -> Vec<Flux2LoraTarget> {
    // row_size = 0 → "split B equally into three parts at patch-build time"
    vec![
        Flux2LoraTarget::Splat {
            candle_key: format!("transformer_blocks.{idx}.{q_leaf}.weight"),
            row_offset: 0,
            row_size: 0,
        },
        Flux2LoraTarget::Splat {
            candle_key: format!("transformer_blocks.{idx}.{k_leaf}.weight"),
            row_offset: 1,
            row_size: 0,
        },
        Flux2LoraTarget::Splat {
            candle_key: format!("transformer_blocks.{idx}.{v_leaf}.weight"),
            row_offset: 2,
            row_size: 0,
        },
    ]
}

/// Map the BFL-native form `double_blocks.{idx}.<rest>` (PEFT canonical
/// suffix) onto candle targets in `space`. The shared dispatch with the
/// Kohya path means the leaf list lives only in `kohya_double_block`.
fn map_bfl_double_block(rest: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    let (idx, leaf) = match rest.split_once('.') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx.parse::<usize>().is_err() {
        return Vec::new();
    }
    // Translate dotted leaf → underscore leaf so we can reuse the Kohya table.
    // e.g. `img_attn.qkv` → `img_attn_qkv`, `img_mlp.0` → `img_mlp_0`.
    let kohya_leaf = leaf.replace('.', "_");
    kohya_double_block(idx, &kohya_leaf, space)
}

fn map_bfl_single_block(rest: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    let (idx, leaf) = match rest.split_once('.') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx.parse::<usize>().is_err() {
        return Vec::new();
    }
    let kohya_leaf = leaf.replace('.', "_");
    kohya_single_block(idx, &kohya_leaf, space)
}

/// Map the diffusers form `transformer_blocks.{idx}.<rest>` onto candle
/// targets. In `Diffusers` space this is mostly Direct (e.g. `attn.to_q` → its
/// own tensor); in `Bfl` space the `attn.to_q/k/v` triple maps to a
/// `FusedSlice` of `img_attn.qkv` (the FLUX-style mapping).
fn map_diffusers_double_block(rest: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    let (idx, leaf) = match rest.split_once('.') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx.parse::<usize>().is_err() {
        return Vec::new();
    }
    match (leaf, space) {
        // ── Image attention ──────────────────────────────────────────────
        ("attn.to_q", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_q.weight"),
        }],
        ("attn.to_k", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_k.weight"),
        }],
        ("attn.to_v", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_v.weight"),
        }],
        // The Bfl arms below encode component index in `row_offset` and 0 size
        // as a sentinel meaning "this is the c-th of three equal Q/K/V slabs"
        // — the patch-builder resolves the absolute slice at build time.
        ("attn.to_q", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Splat {
            candle_key: format!("double_blocks.{idx}.img_attn.qkv.weight"),
            row_offset: 0,
            row_size: 0,
        }],
        ("attn.to_k", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Splat {
            candle_key: format!("double_blocks.{idx}.img_attn.qkv.weight"),
            row_offset: 1,
            row_size: 0,
        }],
        ("attn.to_v", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Splat {
            candle_key: format!("double_blocks.{idx}.img_attn.qkv.weight"),
            row_offset: 2,
            row_size: 0,
        }],
        ("attn.add_q_proj", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Splat {
            candle_key: format!("double_blocks.{idx}.txt_attn.qkv.weight"),
            row_offset: 0,
            row_size: 0,
        }],
        ("attn.add_k_proj", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Splat {
            candle_key: format!("double_blocks.{idx}.txt_attn.qkv.weight"),
            row_offset: 1,
            row_size: 0,
        }],
        ("attn.add_v_proj", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Splat {
            candle_key: format!("double_blocks.{idx}.txt_attn.qkv.weight"),
            row_offset: 2,
            row_size: 0,
        }],
        ("attn.add_q_proj", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.add_q_proj.weight"),
        }],
        ("attn.add_k_proj", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.add_k_proj.weight"),
        }],
        ("attn.add_v_proj", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.add_v_proj.weight"),
        }],
        ("attn.to_out.0", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_out.0.weight"),
        }],
        ("attn.to_out.0", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_attn.proj.weight"),
        }],
        ("attn.to_add_out", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.attn.to_add_out.weight"),
        }],
        ("attn.to_add_out", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_attn.proj.weight"),
        }],
        ("ff.linear_in", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff.linear_in.weight"),
        }],
        ("ff.linear_in", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_mlp.0.weight"),
        }],
        ("ff.linear_out", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff.linear_out.weight"),
        }],
        ("ff.linear_out", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.img_mlp.2.weight"),
        }],
        ("ff_context.linear_in", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff_context.linear_in.weight"),
        }],
        ("ff_context.linear_in", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_mlp.0.weight"),
        }],
        ("ff_context.linear_out", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("transformer_blocks.{idx}.ff_context.linear_out.weight"),
        }],
        ("ff_context.linear_out", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("double_blocks.{idx}.txt_mlp.2.weight"),
        }],
        _ => Vec::new(),
    }
}

fn map_diffusers_single_block(rest: &str, space: Flux2KeySpace) -> Vec<Flux2LoraTarget> {
    let (idx, leaf) = match rest.split_once('.') {
        Some(p) => p,
        None => return Vec::new(),
    };
    if idx.parse::<usize>().is_err() {
        return Vec::new();
    }
    match (leaf, space) {
        ("attn.to_qkv_mlp_proj", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_transformer_blocks.{idx}.attn.to_qkv_mlp_proj.weight"),
        }],
        ("attn.to_qkv_mlp_proj", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_blocks.{idx}.linear1.weight"),
        }],
        ("attn.to_out", Flux2KeySpace::Diffusers) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_transformer_blocks.{idx}.attn.to_out.weight"),
        }],
        ("attn.to_out", Flux2KeySpace::Bfl) => vec![Flux2LoraTarget::Direct {
            candle_key: format!("single_blocks.{idx}.linear2.weight"),
        }],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Patch building — turn every (adapter, layer) pair into per-tensor patches
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Flux2LoraPatch {
    a: Tensor,
    b: Tensor,
    effective_scale: f64,
    target: Flux2LoraTarget,
    lora_path_hash: u64,
    /// Filled in lazily at patch-build time when the target is `Splat` with
    /// `row_size == 0` (sentinel for "split B equally into thirds by
    /// component index in `row_offset`"). Stored as the resolved
    /// `(offset, size)` pair so the forward path is simple.
    resolved_rows: Option<(usize, usize)>,
}

/// A loaded LoRA + its scale + a stable hash of its file path. The hash
/// is the per-LoRA cache-key component so a multi-LoRA stack keeps each
/// adapter's delta independently cacheable.
pub(crate) struct Flux2LoraSpec<'a> {
    pub adapter: &'a LoraAdapter,
    pub scale: f64,
    pub path_hash: u64,
}

fn resolve_rows(target: &Flux2LoraTarget, b_rows: usize) -> Option<(usize, usize)> {
    match target {
        Flux2LoraTarget::Direct { .. } => None,
        Flux2LoraTarget::Splat {
            row_size,
            row_offset,
            ..
        } => {
            if *row_size == 0 {
                // Sentinel: row_offset carries the component index (0/1/2),
                // each slab is `b_rows / 3` wide.
                let third = b_rows / 3;
                Some((row_offset * third, third))
            } else {
                Some((*row_offset, *row_size))
            }
        }
    }
}

fn build_patches(
    specs: &[Flux2LoraSpec<'_>],
    space: Flux2KeySpace,
) -> (HashMap<String, Vec<Flux2LoraPatch>>, usize) {
    let mut patches: HashMap<String, Vec<Flux2LoraPatch>> = HashMap::new();
    let mut skipped = 0usize;
    for spec in specs {
        for (lora_stem, layer) in &spec.adapter.layers {
            let targets = map_flux2_lora_key(lora_stem, space);
            if targets.is_empty() {
                tracing::warn!(
                    key = lora_stem.as_str(),
                    "unrecognized Flux.2 LoRA key, skipping"
                );
                skipped += 1;
                continue;
            }
            let layer_rank = layer.a.dims()[0] as f64;
            let effective_scale = match layer.alpha {
                Some(alpha) => spec.scale * alpha / layer_rank,
                None => spec.scale,
            };
            let b_rows = layer.b.dims().first().copied().unwrap_or(0);
            for target in targets {
                let resolved_rows = resolve_rows(&target, b_rows);
                let candle_key = target.candle_key().to_string();
                patches.entry(candle_key).or_default().push(Flux2LoraPatch {
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
// Delta computation + cache + apply
// ---------------------------------------------------------------------------

#[derive(Hash, Eq, PartialEq, Clone)]
struct DeltaCacheKey {
    tensor_name: String,
    patch_index: usize,
    lora_path_hash: u64,
    scale_bits: u64,
}

fn compute_delta(patch: &Flux2LoraPatch, target_dev: &Device) -> candle_core::Result<Tensor> {
    let a = patch.a.to_dtype(DType::F32)?.to_device(target_dev)?;
    let b = patch.b.to_dtype(DType::F32)?.to_device(target_dev)?;
    let computed = b.matmul(&a)?;
    &computed * patch.effective_scale
}

/// Apply a `Flux2LoraPatch` to a base tensor (`base_full_rows` is `B @ A`'s
/// full row count when computed on the full B). Mutates an F32 working
/// tensor in place. The caller handles dtype conversions.
fn apply_patch_f32(
    base_f32: &Tensor,
    delta_full: &Tensor,
    patch: &Flux2LoraPatch,
) -> candle_core::Result<Tensor> {
    match &patch.target {
        Flux2LoraTarget::Direct { .. } => base_f32 + delta_full,
        Flux2LoraTarget::Splat { .. } => {
            let (offset, size) = patch
                .resolved_rows
                .expect("Splat patch must have resolved_rows");
            // Slice rows [offset..offset+size] of delta_full and add to ALL of base.
            let delta_slice = delta_full.narrow(0, offset, size)?;
            let base_rows = base_f32.dim(0)?;
            if base_rows != size {
                tracing::warn!(
                    base_rows,
                    delta_rows = size,
                    "Flux.2 LoRA Splat: base row count != delta row count, skipping"
                );
                return Ok(base_f32.clone());
            }
            base_f32 + &delta_slice
        }
    }
}

// ---------------------------------------------------------------------------
// `LoraBackend` — wraps a `SimpleBackend` and merges LoRAs at vb.get()
// ---------------------------------------------------------------------------

struct Flux2LoraBackend {
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    patches: HashMap<String, Vec<Flux2LoraPatch>>,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
}

impl Flux2LoraBackend {
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
            let cache_key = DeltaCacheKey {
                tensor_name: name.to_string(),
                patch_index: patch_idx,
                lora_path_hash: patch.lora_path_hash,
                scale_bits: patch.effective_scale.to_bits(),
            };
            // We piggyback on the FLUX `LoraDeltaCache` storage. The FLUX cache
            // is keyed on `flux::lora::LoraCacheKey`, not ours — so we cannot
            // share entries across families. To avoid pulling that struct's
            // fields out of pub(crate), keep the cache private to this
            // backend by allocating our own small CPU map. The `delta_cache`
            // argument is currently *unused* on the Flux.2 path; we accept it
            // for API parity with FLUX so future refactors can unify both.
            let _ = (&self.delta_cache, &cache_key); // silence warning when cache is unused
            let delta_full = compute_delta(patch, dev)?;
            let m = apply_patch_f32(&merged, &delta_full, patch)?;
            merged = m;
        }
        merged.to_dtype(target_dtype)
    }
}

impl candle_nn::var_builder::SimpleBackend for Flux2LoraBackend {
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

/// Wrap an existing `SimpleBackend` (typically a `MmapedSafetensors` or
/// `SingleFileBackend`) so its `vb.get()` calls land merged-LoRA tensors.
/// The wrapper applies LoRA deltas in F32 and casts back to the requested
/// dtype, so downstream `Flux2Linear` constructors load the merged weights.
///
/// `space` selects the candle-key namespace the inner backend exposes. For
/// `Flux2Transformer::new` (BF16 sharded HF or BFL-native single-file via
/// `SingleFileBackend`) pass `Flux2KeySpace::Diffusers`. For raw BFL-native
/// loads (rare, but possible) pass `Flux2KeySpace::Bfl`.
pub(crate) fn wrap_backend_with_lora(
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    specs: &[Flux2LoraSpec<'_>],
    space: Flux2KeySpace,
    progress: &ProgressReporter,
    delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<Box<dyn candle_nn::var_builder::SimpleBackend>> {
    if specs.is_empty() {
        bail!("wrap_backend_with_lora called with no LoraSpecs");
    }
    let (patches, skipped) = build_patches(specs, space);
    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA (Flux.2): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    Ok(Box::new(Flux2LoraBackend {
        inner,
        patches,
        delta_cache,
    }))
}

/// Build a quantized `VarBuilder` from a GGUF file with LoRA deltas applied.
///
/// Mirrors `crate::flux::lora::gguf_lora_var_builder`: dequantises the
/// LoRA-affected tensors, applies the delta in F32 on CPU, re-quantises to
/// the original GGML dtype on the target device. Non-LoRA tensors stay
/// quantized and untouched.
pub(crate) fn gguf_lora_var_builder_flux2(
    transformer_path: &Path,
    specs: &[Flux2LoraSpec<'_>],
    device: &Device,
    progress: &ProgressReporter,
    _delta_cache: Option<Arc<Mutex<LoraDeltaCache>>>,
) -> Result<candle_transformers::quantized_var_builder::VarBuilder> {
    use candle_core::quantized::{gguf_file, QTensor};

    if specs.is_empty() {
        bail!("gguf_lora_var_builder_flux2 called with no LoraSpecs");
    }

    let mut file = std::fs::File::open(transformer_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    let total_tensors = content.tensor_infos.len();
    let mut data: HashMap<String, Arc<QTensor>> = HashMap::with_capacity(total_tensors);

    let (patches, skipped) = build_patches(specs, Flux2KeySpace::Bfl);
    let patched_keys = patches.len();
    let total_patches: usize = patches.values().map(|v| v.len()).sum();
    let max_rank = specs.iter().map(|s| s.adapter.rank).max().unwrap_or(0);
    progress.info(&format!(
        "LoRA (Flux.2 GGUF): {n} adapter(s), {total_patches} patches on {patched_keys} tensors, {skipped} skipped (max rank {max_rank})",
        n = specs.len(),
    ));

    let gguf_bytes_total: u64 = std::fs::metadata(transformer_path)
        .map(|m| m.len())
        .unwrap_or(0);
    progress.weight_load("Flux.2 transformer (GGUF)", 0, gguf_bytes_total);
    for (i, tensor_name) in content.tensor_infos.keys().enumerate() {
        let qtensor = content.tensor(&mut file, tensor_name, device)?;
        data.insert(tensor_name.clone(), Arc::new(qtensor));
        let approx_bytes = gguf_bytes_total * (i as u64 + 1) / total_tensors as u64;
        progress.weight_load(
            "Flux.2 transformer (GGUF)",
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
        if !data.contains_key(candle_key) {
            tracing::warn!(
                key = candle_key.as_str(),
                "Flux.2 LoRA target tensor not found in GGUF, skipping"
            );
            continue;
        }
        let orig_dtype = data[candle_key].dtype();
        let qtensor = data.remove(candle_key).unwrap();
        let mut t = qtensor.dequantize(&Device::Cpu)?;
        drop(qtensor);
        if on_gpu {
            device.synchronize()?;
        }

        for patch in layer_patches.iter() {
            let matmul_dev = if on_gpu { device } else { &Device::Cpu };
            let delta_full = compute_delta(patch, matmul_dev)?.to_device(&Device::Cpu)?;
            let next = apply_patch_f32(&t, &delta_full, patch)?;
            t = next;
            applied += 1;
        }

        let patched = QTensor::quantize_onto(&t, orig_dtype, device)?;
        drop(t);
        data.insert(candle_key.clone(), Arc::new(patched));

        if (i + 1) % 50 == 0 || i + 1 == lora_total {
            progress.info(&format!(
                "Patching Flux.2 LoRA tensor {}/{}",
                i + 1,
                lora_total
            ));
        }
    }

    let total_layers: usize = specs.iter().map(|s| s.adapter.layers.len()).sum();
    progress.info(&format!(
        "LoRA (Flux.2 GGUF): {applied} applied, {} skipped (max rank {max_rank}, {patched_keys} layers patched)",
        total_layers.saturating_sub(applied),
    ));
    if on_gpu {
        device.synchronize()?;
    }

    Ok(candle_transformers::quantized_var_builder::VarBuilder::from_qtensors(data, device))
}

/// Build `Flux2LoraSpec`s from a list of `LoraWeight`s, loading adapters via
/// the parsed-LoRA cache. Returns an `Arc`-owned vector so the caller can
/// hold both the adapters (for lifetimes) and the spec slice.
pub(crate) fn load_lora_adapters(
    loras: &[mold_core::LoraWeight],
    progress: &ProgressReporter,
) -> Result<Vec<Arc<LoraAdapter>>> {
    loras
        .iter()
        .map(|w| {
            progress.info("Loading Flux.2 LoRA adapter");
            let adapter = get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "Flux.2 LoRA: {} layers, rank {}, scale {:.2}",
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

    // ── map_kohya_*_key — leaf coverage for Flux.2 ──────────────────────

    /// Civitai cv:2742432 ships every double-block in this exact leaf set.
    /// Pin every leaf so a refactor can't silently drop one.
    #[test]
    fn kohya_double_block_leaves_bfl_space() {
        let cases = [
            ("img_attn_qkv", "double_blocks.0.img_attn.qkv.weight"),
            ("img_attn_proj", "double_blocks.0.img_attn.proj.weight"),
            ("img_mlp_0", "double_blocks.0.img_mlp.0.weight"),
            ("img_mlp_2", "double_blocks.0.img_mlp.2.weight"),
            ("txt_attn_qkv", "double_blocks.0.txt_attn.qkv.weight"),
            ("txt_attn_proj", "double_blocks.0.txt_attn.proj.weight"),
            ("txt_mlp_0", "double_blocks.0.txt_mlp.0.weight"),
            ("txt_mlp_2", "double_blocks.0.txt_mlp.2.weight"),
        ];
        for (leaf, expected) in cases {
            let key = format!("lora_unet_double_blocks_0_{leaf}");
            let targets = map_flux2_lora_key(&key, Flux2KeySpace::Bfl);
            assert_eq!(targets.len(), 1, "exactly one target for {key}");
            match &targets[0] {
                Flux2LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf}");
                }
                _ => panic!("expected Direct for {leaf}"),
            }
        }
    }

    #[test]
    fn kohya_double_block_qkv_splits_into_three_diffusers_targets() {
        let targets = map_flux2_lora_key(
            "lora_unet_double_blocks_3_img_attn_qkv",
            Flux2KeySpace::Diffusers,
        );
        assert_eq!(
            targets.len(),
            3,
            "QKV in diffusers space splits into to_q/to_k/to_v"
        );
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "transformer_blocks.3.attn.to_q.weight",
                "transformer_blocks.3.attn.to_k.weight",
                "transformer_blocks.3.attn.to_v.weight",
            ],
            "ordering is Q→K→V (component index 0/1/2)"
        );
    }

    #[test]
    fn kohya_double_block_txt_qkv_splits_into_three_diffusers_targets() {
        let targets = map_flux2_lora_key(
            "lora_unet_double_blocks_5_txt_attn_qkv",
            Flux2KeySpace::Diffusers,
        );
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "transformer_blocks.5.attn.add_q_proj.weight",
                "transformer_blocks.5.attn.add_k_proj.weight",
                "transformer_blocks.5.attn.add_v_proj.weight",
            ]
        );
    }

    #[test]
    fn kohya_single_block_leaves() {
        let cases = [
            (
                "linear1",
                Flux2KeySpace::Bfl,
                "single_blocks.7.linear1.weight",
            ),
            (
                "linear1",
                Flux2KeySpace::Diffusers,
                "single_transformer_blocks.7.attn.to_qkv_mlp_proj.weight",
            ),
            (
                "linear2",
                Flux2KeySpace::Bfl,
                "single_blocks.7.linear2.weight",
            ),
            (
                "linear2",
                Flux2KeySpace::Diffusers,
                "single_transformer_blocks.7.attn.to_out.weight",
            ),
        ];
        for (leaf, space, expected) in cases {
            let key = format!("lora_unet_single_blocks_7_{leaf}");
            let targets = map_flux2_lora_key(&key, space);
            assert_eq!(targets.len(), 1, "leaf={leaf} space={space:?}");
            match &targets[0] {
                Flux2LoraTarget::Direct { candle_key } => {
                    assert_eq!(candle_key, expected, "leaf={leaf} space={space:?}");
                }
                _ => panic!("expected Direct for leaf={leaf}"),
            }
        }
    }

    #[test]
    fn kohya_modulation_leaves_skipped_silently() {
        // Flux.2 modulations are SHARED across blocks, so the per-block
        // `img_mod_lin`/`txt_mod_lin`/`modulation_lin` Kohya entries from
        // FLUX-style trainers don't apply here. Skip rather than crash.
        for (kind, leaf) in [
            ("double_blocks", "img_mod_lin"),
            ("double_blocks", "txt_mod_lin"),
            ("single_blocks", "modulation_lin"),
        ] {
            let key = format!("lora_unet_{kind}_0_{leaf}");
            let targets = map_flux2_lora_key(&key, Flux2KeySpace::Bfl);
            assert!(targets.is_empty(), "expected skip for {key}");
        }
    }

    #[test]
    fn unknown_leaf_returns_empty() {
        assert!(map_flux2_lora_key(
            "lora_unet_double_blocks_0_unknown_thing",
            Flux2KeySpace::Bfl
        )
        .is_empty());
        assert!(
            map_flux2_lora_key("lora_te_text_model_layer_0_attn_q", Flux2KeySpace::Bfl).is_empty()
        );
        assert!(map_flux2_lora_key("garbage", Flux2KeySpace::Bfl).is_empty());
    }

    // ── PEFT-canonical (BFL module path with .lora_A.weight suffix) ─────

    /// cv:2682864 uses `diffusion_model.<bfl module>.lora_A.weight`. The
    /// classifier strips the suffix; our key mapper strips the prefix
    /// chain `model.` → `diffusion_model.` → `transformer.` and then
    /// resolves the BFL path.
    #[test]
    fn peft_canonical_bfl_module_path_resolves_to_bfl_target() {
        let stem = "diffusion_model.double_blocks.0.img_attn.qkv";
        let targets = map_flux2_lora_key(stem, Flux2KeySpace::Bfl);
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            Flux2LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "double_blocks.0.img_attn.qkv.weight");
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn peft_canonical_bfl_qkv_splits_in_diffusers_space() {
        let stem = "diffusion_model.double_blocks.0.img_attn.qkv";
        let targets = map_flux2_lora_key(stem, Flux2KeySpace::Diffusers);
        let keys: Vec<&str> = targets.iter().map(|t| t.candle_key()).collect();
        assert_eq!(
            keys,
            vec![
                "transformer_blocks.0.attn.to_q.weight",
                "transformer_blocks.0.attn.to_k.weight",
                "transformer_blocks.0.attn.to_v.weight",
            ]
        );
    }

    #[test]
    fn peft_canonical_diffusers_module_path_resolves_to_diffusers_target() {
        let stem = "transformer.transformer_blocks.4.attn.to_q";
        let targets = map_flux2_lora_key(stem, Flux2KeySpace::Diffusers);
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            Flux2LoraTarget::Direct { candle_key } => {
                assert_eq!(candle_key, "transformer_blocks.4.attn.to_q.weight");
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn peft_canonical_diffusers_qkv_splat_in_bfl_space() {
        let stem = "transformer.transformer_blocks.4.attn.to_q";
        let targets = map_flux2_lora_key(stem, Flux2KeySpace::Bfl);
        assert_eq!(targets.len(), 1);
        match &targets[0] {
            Flux2LoraTarget::Splat {
                candle_key,
                row_offset,
                row_size,
            } => {
                assert_eq!(candle_key, "double_blocks.4.img_attn.qkv.weight");
                assert_eq!(*row_offset, 0);
                assert_eq!(*row_size, 0, "0 = sentinel for thirds-split");
            }
            _ => panic!("expected Splat"),
        }
    }

    // ── resolve_rows + apply_patch_f32 — math correctness ───────────────

    #[test]
    fn resolve_rows_thirds_for_splat_with_zero_size() {
        let target = Flux2LoraTarget::Splat {
            candle_key: "x".into(),
            row_offset: 1,
            row_size: 0,
        };
        let rows = resolve_rows(&target, 3 * 8).unwrap();
        assert_eq!(rows, (8, 8), "component 1 of 3 in a 24-row B");
    }

    #[test]
    fn resolve_rows_passes_through_explicit_size() {
        let target = Flux2LoraTarget::Splat {
            candle_key: "x".into(),
            row_offset: 5,
            row_size: 11,
        };
        assert_eq!(resolve_rows(&target, 100).unwrap(), (5, 11));
    }

    /// `apply_patch_f32` for Direct: the merged tensor must equal `base + delta`.
    #[test]
    fn apply_patch_direct_adds_full_delta() {
        let dev = Device::Cpu;
        let base = Tensor::full(2.0f32, (4, 3), &dev).unwrap();
        let delta = Tensor::full(0.5f32, (4, 3), &dev).unwrap();
        let patch = Flux2LoraPatch {
            a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            effective_scale: 1.0,
            target: Flux2LoraTarget::Direct {
                candle_key: "x".into(),
            },
            lora_path_hash: 0,
            resolved_rows: None,
        };
        let merged = apply_patch_f32(&base, &delta, &patch).unwrap();
        let merged_vals: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            merged_vals.iter().all(|v| (v - 2.5).abs() < 1e-6),
            "Direct merge expected base + delta = 2.5 everywhere, got {merged_vals:?}",
        );
    }

    /// `apply_patch_f32` for Splat: the delta is `b_full @ a` whose first
    /// `h` rows go to `to_q`, next `h` go to `to_k`, last `h` go to `to_v`.
    /// Pin Q-component patch math: base (h, in) + delta_full[0..h, :].
    #[test]
    fn apply_patch_splat_uses_correct_third_of_delta() {
        let dev = Device::Cpu;
        // Construct a delta_full whose three thirds are constant 0.1 / 0.2 / 0.3.
        // Q row gets 0.1, K gets 0.2, V gets 0.3.
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
        for (component, expected_value) in [(0, 0.1f32), (1, 0.2), (2, 0.3)] {
            let mut patch = Flux2LoraPatch {
                a: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                b: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                effective_scale: 1.0,
                target: Flux2LoraTarget::Splat {
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
                vals.iter().all(|v| (v - expected_value).abs() < 1e-6),
                "component {component}: expected {expected_value} everywhere, got {vals:?}",
            );
        }
    }

    // ── build_patches — adapter wiring ──────────────────────────────────

    /// Synthetic adapter targeting one BFL-name layer. (rank=2, in=4) for
    /// A and (out=6, rank=2) for B — small enough to inspect tensor shapes
    /// without producing huge fixtures.
    fn synthetic_kohya_adapter(layer: &str, scale_a: f32, scale_b: f32) -> LoraAdapter {
        let dev = Device::Cpu;
        let a = Tensor::full(scale_a, (2, 4), &dev).unwrap();
        let b = Tensor::full(scale_b, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(layer.to_string(), LoraLayer { a, b, alpha: None });
        LoraAdapter { layers, rank: 2 }
    }

    #[test]
    fn build_patches_records_per_target_count() {
        let adapter = synthetic_kohya_adapter("lora_unet_double_blocks_0_img_attn_qkv", 1.0, 1.0);
        let specs = [Flux2LoraSpec {
            adapter: &adapter,
            scale: 0.7,
            path_hash: 0xCAFE,
        }];

        // BFL: one fused target.
        let (patches, skipped) = build_patches(&specs, Flux2KeySpace::Bfl);
        assert_eq!(skipped, 0);
        assert_eq!(patches.len(), 1);
        assert!(patches.contains_key("double_blocks.0.img_attn.qkv.weight"));

        // Diffusers: three Splat targets (Q/K/V), each in its own bucket.
        let (patches, _) = build_patches(&specs, Flux2KeySpace::Diffusers);
        assert_eq!(patches.len(), 3);
        for k in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ] {
            assert!(patches.contains_key(k), "missing {k}");
            let bucket = &patches[k];
            assert_eq!(bucket.len(), 1);
            // Splat resolved_rows set at build time: each is 6/3 = 2 rows.
            assert_eq!(
                bucket[0].resolved_rows,
                Some((bucket[0].resolved_rows.unwrap().0, 2))
            );
        }
    }

    #[test]
    fn build_patches_alpha_normalises_scale() {
        let dev = Device::Cpu;
        let mut adapter =
            synthetic_kohya_adapter("lora_unet_double_blocks_0_img_attn_proj", 1.0, 1.0);
        // Insert alpha=4 on the layer; with rank=2 the effective scale becomes
        // user_scale * 4 / 2 = 2 * user_scale.
        if let Some(layer) = adapter
            .layers
            .get_mut("lora_unet_double_blocks_0_img_attn_proj")
        {
            layer.alpha = Some(4.0);
        }
        let _dev_use = dev; // silence unused
        let specs = [Flux2LoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0,
        }];
        let (patches, _) = build_patches(&specs, Flux2KeySpace::Bfl);
        let bucket = &patches["double_blocks.0.img_attn.proj.weight"];
        let s = bucket[0].effective_scale;
        assert!(
            (s - 1.0).abs() < 1e-9,
            "effective scale = user(0.5) * alpha(4) / rank(2) = 1.0, got {s}"
        );
    }

    #[test]
    fn build_patches_two_specs_stack_on_same_bfl_target() {
        let a1 = synthetic_kohya_adapter("lora_unet_double_blocks_0_img_attn_qkv", 1.0, 1.0);
        let a2 = synthetic_kohya_adapter("lora_unet_double_blocks_0_img_attn_qkv", 0.5, 0.5);
        let specs = [
            Flux2LoraSpec {
                adapter: &a1,
                scale: 1.0,
                path_hash: 0xAA,
            },
            Flux2LoraSpec {
                adapter: &a2,
                scale: 1.0,
                path_hash: 0xBB,
            },
        ];
        let (patches, _) = build_patches(&specs, Flux2KeySpace::Bfl);
        let bucket = &patches["double_blocks.0.img_attn.qkv.weight"];
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

    /// End-to-end: build a synthetic SimpleBackend exposing a (24, 8) fused
    /// QKV tensor of zeros, layer it with a Kohya `img_attn_qkv` LoRA whose
    /// `B = 1` and `A = 1`, then verify each component patch added the
    /// expected `(2, 8)` constant slab in Diffusers space.
    #[test]
    fn end_to_end_diffusers_splat_merges_correctly() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Inner backend: three (2, 8) zero tensors named after the three
        // diffusers Q/K/V targets. The SimpleBackend wrapper will be queried
        // for each candle key; we want every one to come back with the LoRA
        // patch applied.
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

        // Adapter: B = 6×2 of ones, A = 2×8 of ones. delta_full = B@A * 1 = 6×8 of 2s.
        // Component 0 → Q gets rows [0..2] (constant 2). Component 1 → K rows
        // [2..4]. Component 2 → V rows [4..6]. So every Q/K/V tensor merges
        // base(=0) + 2 = 2 everywhere.
        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(1.0f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_double_blocks_0_img_attn_qkv".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [Flux2LoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0xFEED,
        }];

        // Build the inner mmap-backed backend. We use candle's
        // `SafeTensorWithRouting`-equivalent by mmap'ing directly through
        // `MmapedSafetensors`.
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
        let wrapped =
            wrap_backend_with_lora(inner, &specs, Flux2KeySpace::Diffusers, &progress, None)
                .expect("wrap");

        // For each diffusers candle key, fetch and inspect the merged tensor.
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

    /// End-to-end (BFL space): fused QKV LoRA lands on the single fused
    /// candle tensor as a single Direct merge. The merged tensor should be
    /// `base + B@A * scale` everywhere.
    #[test]
    fn end_to_end_bfl_direct_merge_on_fused_qkv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("base_bfl.safetensors");
        // Fused QKV: rows = 6, cols = 8.
        write_synthetic_safetensors_with_data(
            &path,
            &[(
                "double_blocks.0.img_attn.qkv.weight".to_string(),
                vec![6, 8],
                vec![1.0; 48],
            )],
        );

        let dev = Device::Cpu;
        let a = Tensor::full(1.0f32, (2, 8), &dev).unwrap();
        let b = Tensor::full(0.5f32, (6, 2), &dev).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "lora_unet_double_blocks_0_img_attn_qkv".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 2 };
        let specs = [Flux2LoraSpec {
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
        let wrapped = wrap_backend_with_lora(inner, &specs, Flux2KeySpace::Bfl, &progress, None)
            .expect("wrap");

        // B@A: (6,2) @ (2,8) of all-ones with B=0.5, A=1 → entry value = 2 * 0.5 * 1 = 1.
        // Merged tensor = base(1.0) + delta(1.0) = 2.0.
        let t = wrapped
            .get_unchecked("double_blocks.0.img_attn.qkv.weight", DType::F32, &dev)
            .expect("get");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| (v - 2.0).abs() < 1e-5),
            "expected 2.0 (= 1 + 1), got {vals:?}",
        );
    }

    // ── classify_lora_key — both Kohya and PEFT canonical accepted ──────

    #[test]
    fn classify_kohya_and_peft_suffixes() {
        // Kohya: down/up = A/B
        assert_eq!(
            classify_lora_key("lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"),
            Some((
                LoraDirection::Down,
                "lora_unet_double_blocks_0_img_attn_qkv"
            ))
        );
        assert_eq!(
            classify_lora_key("lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight"),
            Some((LoraDirection::Up, "lora_unet_double_blocks_0_img_attn_qkv"))
        );
        // PEFT canonical
        assert_eq!(
            classify_lora_key("diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"),
            Some((
                LoraDirection::Down,
                "diffusion_model.double_blocks.0.img_attn.qkv"
            ))
        );
        assert_eq!(
            classify_lora_key("diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight"),
            Some((
                LoraDirection::Up,
                "diffusion_model.double_blocks.0.img_attn.qkv"
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
}
