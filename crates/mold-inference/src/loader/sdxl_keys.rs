//! SDXL A1111 → diffusers key rename pass (phase 2.5).
//!
//! Bridges Civitai single-file SDXL checkpoints (LDM tensor naming)
//! to the diffusers naming candle's
//! `stable_diffusion::{unet_2d::UNet2DConditionModel, vae::AutoEncoderKL,
//! clip::ClipTextTransformer}` constructors expect.
//!
//! Layout per the phase-2.2 audit
//! (`tasks/catalog-expansion-phase-2-tensor-audit.md`):
//!
//! - **UNet** at `model.diffusion_model.*` — 3 stages
//!   (`block_out_channels = [320, 640, 1280]`). Down stage 0 is resnet-only;
//!   stages 1+2 carry `Transformer2DModel` blocks (2 transformer layers in
//!   stage 1, 10 in stage 2). Up side mirrors with the order flipped.
//! - **VAE** at `first_stage_model.*` — identical to SD1.5 (audit point 3),
//!   so the rename pass is shared via [`crate::loader::vae_keys::apply_vae_rename`].
//! - **CLIP-L** at `conditioner.embedders.0.transformer.text_model.*` —
//!   inner HF CLIP layout, just a different outer prefix from SD1.5.
//! - **CLIP-G** at `conditioner.embedders.1.model.*` — OpenCLIP layout
//!   (`transformer.resblocks.{i}.{ln_1, ln_2, attn.{in_proj_weight,
//!   in_proj_bias, out_proj}, mlp.{c_fc, c_proj}}`, plus
//!   `token_embedding`, `positional_embedding`, `ln_final`,
//!   `text_projection`). The fused `attn.in_proj_weight` /
//!   `attn.in_proj_bias` slabs split row-wise into the diffusers
//!   `self_attn.{q,k,v}_proj.{weight,bias}` triple — that's the
//!   [`RenameOutput::FusedSlice`] case.
//!
//! Naming reference: every diffusers key in the table below comes from a
//! `vs.pp("…")` chain in
//! `~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/
//! src/models/stable_diffusion/{unet_2d,clip,vae}.rs`. Confirmed against
//! the depth-2 audit dump for Pony / Juggernaut / generic SDXL.

use crate::loader::single_file::SingleFileBundle;
use crate::loader::vae_keys::apply_vae_rename;
use std::collections::BTreeMap;
use thiserror::Error;

/// Output of a CLIP-G rename — most A1111 keys map 1:1 to a diffusers
/// key, but the OpenCLIP `attn.in_proj_weight` / `attn.in_proj_bias`
/// slabs are fused QKV that split row-wise into three separate
/// diffusers tensors. The future `SimpleBackend` slices the underlying
/// mmap'd tensor based on `(component, num_components)`, mirroring the
/// `LoraTarget::FusedSlice` precedent at
/// `crates/mold-inference/src/flux/lora.rs:104-112`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenameOutput {
    /// 1:1 rename — the candle constructor reads the named tensor whole.
    Direct(String),
    /// Slice rename — the candle constructor reads one of `num_components`
    /// equally-sized row-wise slices from a fused source tensor.
    FusedSlice {
        /// Diffusers-side key the candle constructor will request.
        diffusers_key: String,
        /// Axis to slice along. `0` is row-wise (the `attn.in_proj_*`
        /// case); held open as a field so future fusions on column or
        /// inner axes don't churn the public API.
        axis: usize,
        /// Which component within the fused source (0, 1, …, num_components-1).
        component: usize,
        /// Total number of equally-sized components in the fused source.
        num_components: usize,
    },
}

/// Diffusers→a1111 lookup for the four SDXL components. UNet / VAE / CLIP-L
/// resolve directly to the original A1111 key inside the mmap'd
/// safetensors; CLIP-G additionally carries a [`RenameOutput`] so the
/// future `SimpleBackend` knows whether to slice the source tensor.
#[derive(Debug, Default, Clone)]
pub struct SdxlRemap {
    pub unet: BTreeMap<String, String>,
    pub vae: BTreeMap<String, String>,
    pub clip_l: BTreeMap<String, String>,
    /// Diffusers→`(a1111_key, RenameOutput)`. The string is the original
    /// CLIP-G A1111 key (so the backend can mmap the source tensor); the
    /// `RenameOutput` tells the backend how to project that source onto
    /// the diffusers key (`Direct` or `FusedSlice`).
    pub clip_g: BTreeMap<String, (String, RenameOutput)>,
    /// A1111 keys with no rename rule. Per the audit, these are tolerated:
    /// log + drop, not error.
    pub unmapped: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RemapError {
    // No fatal cases yet — rename rules tolerate stragglers via `unmapped`.
    // The variant slot is held open so the public signature stays
    // `Result<_, RemapError>` and downstream consumers can absorb future
    // hard-fails (e.g. malformed `attn.in_proj_weight` shape) without API
    // churn.
    #[error("placeholder — no fatal rename failures in SDXL yet")]
    #[allow(dead_code)]
    Placeholder,
}

/// Translate a SDXL A1111 UNet key to its diffusers equivalent.
///
/// Returns `None` for keys that don't start with `model.diffusion_model.`
/// or that name an outer position with no rename rule. Inner attention
/// payloads (`transformer_blocks.{i}.attn{1,2}.{to_q,to_k,to_v,to_out.0}.…`,
/// `norm.…`) are passed through verbatim — the rename only rewrites the
/// outer block envelope.
pub fn apply_sdxl_unet_rename(a1111_key: &str) -> Option<String> {
    let inner = a1111_key.strip_prefix("model.diffusion_model.")?;

    // Top-level constants — same naming as SD1.5.
    if let Some(out) = match inner {
        "time_embed.0.weight" => Some("time_embedding.linear_1.weight"),
        "time_embed.0.bias" => Some("time_embedding.linear_1.bias"),
        "time_embed.2.weight" => Some("time_embedding.linear_2.weight"),
        "time_embed.2.bias" => Some("time_embedding.linear_2.bias"),
        "out.0.weight" => Some("conv_norm_out.weight"),
        "out.0.bias" => Some("conv_norm_out.bias"),
        "out.2.weight" => Some("conv_out.weight"),
        "out.2.bias" => Some("conv_out.bias"),
        // SDXL adds `label_emb` for the size/crop conditioning vector
        // that candle's UNet calls `add_embedding`.
        "label_emb.0.0.weight" => Some("add_embedding.linear_1.weight"),
        "label_emb.0.0.bias" => Some("add_embedding.linear_1.bias"),
        "label_emb.0.2.weight" => Some("add_embedding.linear_2.weight"),
        "label_emb.0.2.bias" => Some("add_embedding.linear_2.bias"),
        _ => None,
    } {
        return Some(out.to_string());
    }

    if let Some(rest) = inner.strip_prefix("input_blocks.") {
        let (block_idx, rest) = split_idx(rest)?;
        let (sub_idx, suffix) = split_idx(rest)?;
        return rename_unet_input_block(block_idx, sub_idx, suffix);
    }

    if let Some(rest) = inner.strip_prefix("middle_block.") {
        let (sub_idx, suffix) = split_idx(rest)?;
        return rename_unet_middle_block(sub_idx, suffix);
    }

    if let Some(rest) = inner.strip_prefix("output_blocks.") {
        let (block_idx, rest) = split_idx(rest)?;
        let (sub_idx, suffix) = split_idx(rest)?;
        return rename_unet_output_block(block_idx, sub_idx, suffix);
    }

    None
}

/// Translate a SDXL A1111 CLIP-L key to its diffusers equivalent.
///
/// SDXL CLIP-L lives at `conditioner.embedders.0.transformer.text_model.*` —
/// stripping the leading `conditioner.embedders.0.transformer.` lands keys
/// at `text_model.*`, exactly what
/// `candle_transformers::models::stable_diffusion::clip::ClipTextTransformer::new`
/// asks for (`vs.pp("text_model")` chain at `clip.rs:350`).
pub fn apply_sdxl_clip_l_rename(a1111_key: &str) -> Option<String> {
    let stripped = a1111_key.strip_prefix("conditioner.embedders.0.transformer.")?;
    if !stripped.starts_with("text_model.") && stripped != "text_model" {
        return None;
    }
    Some(stripped.to_string())
}

/// Translate a SDXL A1111 CLIP-G key to its diffusers equivalent.
///
/// CLIP-G uses OpenCLIP's layout under `conditioner.embedders.1.model.*`.
/// Two distinct rename surfaces:
///
/// 1. **Layout rename** — `transformer.resblocks.{i}.{ln_1, ln_2,
///    attn.out_proj, mlp.c_fc, mlp.c_proj}` → `text_model.encoder.layers.{i}.{
///    layer_norm1, layer_norm2, self_attn.out_proj, mlp.fc1, mlp.fc2}`,
///    plus the outer `token_embedding`, `positional_embedding`, `ln_final`,
///    and `text_projection` keys. Returns [`RenameOutput::Direct`].
///
/// 2. **Fused QKV split** — `transformer.resblocks.{i}.attn.in_proj_{weight,bias}`
///    is a fused row-wise stack `[3*d, d]` (or `[3*d]` for the bias) that
///    splits into the diffusers `self_attn.{q,k,v}_proj.{weight,bias}`
///    triple. Three calls return three `FusedSlice` outputs sharing the
///    same source A1111 key with `component ∈ {0, 1, 2}` and `num_components: 3`.
///
/// Reference: diffusers `convert_open_clip_checkpoint.py`.
pub fn apply_sdxl_clip_g_rename(a1111_key: &str) -> Option<Vec<RenameOutput>> {
    let inner = a1111_key.strip_prefix("conditioner.embedders.1.model.")?;

    // Outer constants — token / positional embeddings, final layer norm,
    // text projection. OpenCLIP names: `token_embedding.weight`,
    // `positional_embedding`, `ln_final.{weight,bias}`, `text_projection`
    // (the projection ships as a raw tensor, no `.weight` suffix).
    if let Some(direct) = match inner {
        "token_embedding.weight" => Some("text_model.embeddings.token_embedding.weight"),
        "positional_embedding" => Some("text_model.embeddings.position_embedding.weight"),
        "ln_final.weight" => Some("text_model.final_layer_norm.weight"),
        "ln_final.bias" => Some("text_model.final_layer_norm.bias"),
        "text_projection" => Some("text_projection.weight"),
        _ => None,
    } {
        return Some(vec![RenameOutput::Direct(direct.to_string())]);
    }

    if let Some(rest) = inner.strip_prefix("transformer.resblocks.") {
        let (layer_idx, suffix) = split_idx(rest)?;
        return rename_clip_g_resblock(layer_idx, suffix);
    }

    None
}

/// Apply the SDXL rename rules to every key in the bundle, producing
/// component-keyed diffusers→source maps the engine can hand to a
/// custom `SimpleBackend`. Unmapped keys land in `SdxlRemap::unmapped`
/// and are logged (not errored) per the audit's "tolerate stray
/// tensors" rule.
pub fn build_sdxl_remap(bundle: &SingleFileBundle) -> Result<SdxlRemap, RemapError> {
    let mut out = SdxlRemap::default();

    apply_into(
        &bundle.unet_keys,
        &mut out.unet,
        &mut out.unmapped,
        apply_sdxl_unet_rename,
    );
    apply_into(
        &bundle.vae_keys,
        &mut out.vae,
        &mut out.unmapped,
        apply_vae_rename,
    );
    apply_into(
        &bundle.clip_l_keys,
        &mut out.clip_l,
        &mut out.unmapped,
        apply_sdxl_clip_l_rename,
    );

    if let Some(clip_g_keys) = bundle.clip_g_keys.as_ref() {
        for key in clip_g_keys {
            match apply_sdxl_clip_g_rename(key) {
                Some(outputs) => {
                    for output in outputs {
                        let diffusers_key = match &output {
                            RenameOutput::Direct(k) => k.clone(),
                            RenameOutput::FusedSlice { diffusers_key, .. } => diffusers_key.clone(),
                        };
                        out.clip_g.insert(diffusers_key, (key.clone(), output));
                    }
                }
                None => out.unmapped.push(key.clone()),
            }
        }
    }

    Ok(out)
}

fn apply_into<F>(
    src: &[String],
    dst: &mut BTreeMap<String, String>,
    unmapped: &mut Vec<String>,
    rename: F,
) where
    F: Fn(&str) -> Option<String>,
{
    for key in src {
        match rename(key) {
            Some(diffusers_key) => {
                dst.insert(diffusers_key, key.clone());
            }
            None => unmapped.push(key.clone()),
        }
    }
}

// ─── inner helpers ────────────────────────────────────────────────────────

/// Split `"<int>.<rest>"` into `(<int>, <rest>)`.
fn split_idx(s: &str) -> Option<(usize, &str)> {
    let (head, tail) = s.split_once('.')?;
    let idx: usize = head.parse().ok()?;
    Some((idx, tail))
}

/// SDXL UNet down-side outer block layout:
///
/// | block_idx | sub_idx | meaning                                   |
/// |-----------|---------|-------------------------------------------|
/// | 0         | 0       | conv_in                                   |
/// | 1, 2      | 0       | down_blocks.0.resnets.{0,1} (no attn)     |
/// | 3         | 0       | down_blocks.0.downsamplers.0 (op → conv)  |
/// | 4, 5      | 0       | down_blocks.1.resnets.{0,1}               |
/// | 4, 5      | 1       | down_blocks.1.attentions.{0,1}            |
/// | 6         | 0       | down_blocks.1.downsamplers.0              |
/// | 7, 8      | 0       | down_blocks.2.resnets.{0,1}               |
/// | 7, 8      | 1       | down_blocks.2.attentions.{0,1}            |
///
/// SDXL has 3 stages and no downsampler at the bottom — input_blocks
/// stops at index 8.
fn rename_unet_input_block(block_idx: usize, sub_idx: usize, suffix: &str) -> Option<String> {
    if block_idx == 0 && sub_idx == 0 {
        return Some(format!("conv_in.{suffix}"));
    }

    // Stage 0: input_blocks 1, 2 = resnets; 3 = downsampler. No attentions.
    if (1..=3).contains(&block_idx) {
        let in_stage = block_idx - 1; // 0, 1, 2
        return match (in_stage, sub_idx) {
            (0, 0) | (1, 0) => Some(format!(
                "down_blocks.0.resnets.{in_stage}.{}",
                rename_resnet_inner(suffix)?
            )),
            (2, 0) => Some(format!(
                "down_blocks.0.downsamplers.0.{}",
                rename_downsampler_inner(suffix)?
            )),
            _ => None,
        };
    }

    // Stages 1, 2: input_blocks {4,5} / {7,8} = [resnet, attention];
    // input_blocks 6 = downsampler for stage 1; stage 2 has no downsampler.
    if (4..=8).contains(&block_idx) {
        let stage_idx = if block_idx <= 6 { 1 } else { 2 };
        let stage_base = if stage_idx == 1 { 4 } else { 7 };
        let in_stage = block_idx - stage_base; // 0, 1 (resnet pair) or 2 (downsampler stage 1)

        if stage_idx == 1 && in_stage == 2 {
            // input_blocks.6.0 = down_blocks.1.downsamplers.0
            return match sub_idx {
                0 => Some(format!(
                    "down_blocks.1.downsamplers.0.{}",
                    rename_downsampler_inner(suffix)?
                )),
                _ => None,
            };
        }

        return match (in_stage, sub_idx) {
            (0, 0) | (1, 0) => Some(format!(
                "down_blocks.{stage_idx}.resnets.{in_stage}.{}",
                rename_resnet_inner(suffix)?
            )),
            (0, 1) | (1, 1) => Some(format!(
                "down_blocks.{stage_idx}.attentions.{in_stage}.{suffix}",
            )),
            _ => None,
        };
    }

    None
}

/// Same shape as SD1.5: middle_block = [resnet, attention, resnet] →
/// mid_block.{resnets, attentions}.
fn rename_unet_middle_block(sub_idx: usize, suffix: &str) -> Option<String> {
    match sub_idx {
        0 => Some(format!(
            "mid_block.resnets.0.{}",
            rename_resnet_inner(suffix)?
        )),
        1 => Some(format!("mid_block.attentions.0.{suffix}")),
        2 => Some(format!(
            "mid_block.resnets.1.{}",
            rename_resnet_inner(suffix)?
        )),
        _ => None,
    }
}

/// SDXL UNet up-side outer block layout (3 stages reversed from down):
///
/// | block_idx | sub_idx | meaning                                          |
/// |-----------|---------|--------------------------------------------------|
/// | 0, 1, 2   | 0       | up_blocks.0.resnets.{0,1,2}                      |
/// | 0, 1, 2   | 1       | up_blocks.0.attentions.{0,1,2}                   |
/// | 2         | 2       | up_blocks.0.upsamplers.0                         |
/// | 3, 4, 5   | 0       | up_blocks.1.resnets.{0,1,2}                      |
/// | 3, 4, 5   | 1       | up_blocks.1.attentions.{0,1,2}                   |
/// | 5         | 2       | up_blocks.1.upsamplers.0                         |
/// | 6, 7, 8   | 0       | up_blocks.2.resnets.{0,1,2} (no attn, no upsamp) |
///
/// Three resnets per stage on the up-side per diffusers convention
/// (layers_per_block + 1 = 3); top stage has no upsampler.
fn rename_unet_output_block(block_idx: usize, sub_idx: usize, suffix: &str) -> Option<String> {
    let stage_idx = block_idx / 3;
    let resnet_idx = block_idx % 3;

    match stage_idx {
        // Stages 0, 1: resnet at .0, attention at .1, optional upsampler at .2.
        0 | 1 => match sub_idx {
            0 => Some(format!(
                "up_blocks.{stage_idx}.resnets.{resnet_idx}.{}",
                rename_resnet_inner(suffix)?
            )),
            1 => Some(format!(
                "up_blocks.{stage_idx}.attentions.{resnet_idx}.{suffix}",
            )),
            2 if resnet_idx == 2 => Some(format!(
                "up_blocks.{stage_idx}.upsamplers.0.{}",
                rename_upsampler_inner(suffix)?
            )),
            _ => None,
        },
        // Stage 2 (top, lowest-resolution feature map): resnets only, no
        // attentions, no upsampler.
        2 => match sub_idx {
            0 => Some(format!(
                "up_blocks.2.resnets.{resnet_idx}.{}",
                rename_resnet_inner(suffix)?
            )),
            _ => None,
        },
        _ => None,
    }
}

/// LDM ResnetBlock inner naming → diffusers ResnetBlock2D. Identical to
/// SD1.5 — UNet ResnetBlock layout doesn't change between families.
fn rename_resnet_inner(suffix: &str) -> Option<String> {
    Some(match suffix {
        "in_layers.0.weight" => "norm1.weight".to_string(),
        "in_layers.0.bias" => "norm1.bias".to_string(),
        "in_layers.2.weight" => "conv1.weight".to_string(),
        "in_layers.2.bias" => "conv1.bias".to_string(),
        "emb_layers.1.weight" => "time_emb_proj.weight".to_string(),
        "emb_layers.1.bias" => "time_emb_proj.bias".to_string(),
        "out_layers.0.weight" => "norm2.weight".to_string(),
        "out_layers.0.bias" => "norm2.bias".to_string(),
        "out_layers.3.weight" => "conv2.weight".to_string(),
        "out_layers.3.bias" => "conv2.bias".to_string(),
        "skip_connection.weight" => "conv_shortcut.weight".to_string(),
        "skip_connection.bias" => "conv_shortcut.bias".to_string(),
        _ => return None,
    })
}

/// LDM Downsample (uses `op` as the conv name) → diffusers (uses `conv`).
fn rename_downsampler_inner(suffix: &str) -> Option<String> {
    suffix
        .strip_prefix("op.")
        .map(|tail| format!("conv.{tail}"))
}

/// LDM Upsample uses `conv` already — same as diffusers `upsamplers.0.conv`.
fn rename_upsampler_inner(suffix: &str) -> Option<String> {
    Some(suffix.to_string())
}

/// CLIP-G OpenCLIP `transformer.resblocks.{layer_idx}.{suffix}` →
/// diffusers `text_model.encoder.layers.{layer_idx}.{suffix}` (with
/// inner suffix renames + fused QKV split).
fn rename_clip_g_resblock(layer_idx: usize, suffix: &str) -> Option<Vec<RenameOutput>> {
    let layer = format!("text_model.encoder.layers.{layer_idx}");

    // Layer norms: ln_1 / ln_2 → layer_norm1 / layer_norm2.
    if let Some(direct) = match suffix {
        "ln_1.weight" => Some(format!("{layer}.layer_norm1.weight")),
        "ln_1.bias" => Some(format!("{layer}.layer_norm1.bias")),
        "ln_2.weight" => Some(format!("{layer}.layer_norm2.weight")),
        "ln_2.bias" => Some(format!("{layer}.layer_norm2.bias")),
        _ => None,
    } {
        return Some(vec![RenameOutput::Direct(direct)]);
    }

    // Output projection — direct rename, no slicing.
    if let Some(direct) = match suffix {
        "attn.out_proj.weight" => Some(format!("{layer}.self_attn.out_proj.weight")),
        "attn.out_proj.bias" => Some(format!("{layer}.self_attn.out_proj.bias")),
        _ => None,
    } {
        return Some(vec![RenameOutput::Direct(direct)]);
    }

    // MLP fc1 / fc2 — OpenCLIP's `c_fc` / `c_proj` map to diffusers' `fc1` / `fc2`.
    if let Some(direct) = match suffix {
        "mlp.c_fc.weight" => Some(format!("{layer}.mlp.fc1.weight")),
        "mlp.c_fc.bias" => Some(format!("{layer}.mlp.fc1.bias")),
        "mlp.c_proj.weight" => Some(format!("{layer}.mlp.fc2.weight")),
        "mlp.c_proj.bias" => Some(format!("{layer}.mlp.fc2.bias")),
        _ => None,
    } {
        return Some(vec![RenameOutput::Direct(direct)]);
    }

    // Fused QKV slabs — `attn.in_proj_weight` shape `[3*d, d]` and
    // `attn.in_proj_bias` shape `[3*d]` both split row-wise into the
    // diffusers `self_attn.{q,k,v}_proj.{weight,bias}` triple.
    let (kind, lookup) = match suffix {
        "attn.in_proj_weight" => ("weight", true),
        "attn.in_proj_bias" => ("bias", true),
        _ => ("", false),
    };
    if lookup {
        return Some(vec![
            RenameOutput::FusedSlice {
                diffusers_key: format!("{layer}.self_attn.q_proj.{kind}"),
                axis: 0,
                component: 0,
                num_components: 3,
            },
            RenameOutput::FusedSlice {
                diffusers_key: format!("{layer}.self_attn.k_proj.{kind}"),
                axis: 0,
                component: 1,
                num_components: 3,
            },
            RenameOutput::FusedSlice {
                diffusers_key: format!("{layer}.self_attn.v_proj.{kind}"),
                axis: 0,
                component: 2,
                num_components: 3,
            },
        ]);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Round 1 — pure rename rules, no I/O ─────────────────────────────

    #[test]
    fn unet_input_block_0_to_conv_in() {
        // Same as SD1.5: input_blocks.0.0 = the outermost conv.
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.input_blocks.0.0.weight").as_deref(),
            Some("conv_in.weight"),
        );
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.input_blocks.0.0.bias").as_deref(),
            Some("conv_in.bias"),
        );
    }

    #[test]
    fn unet_input_block_stage_0_resnet_only() {
        // SDXL diverges from SD1.5 here: stage 0 is resnet-only (no
        // attentions), so input_blocks.{1,2}.0 = resnets, input_blocks.3.0
        // = downsampler. Confirm both the resnet and the downsampler.
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.input_blocks.1.0.in_layers.0.weight",)
                .as_deref(),
            Some("down_blocks.0.resnets.0.norm1.weight"),
        );
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.input_blocks.3.0.op.weight").as_deref(),
            Some("down_blocks.0.downsamplers.0.conv.weight"),
        );
        // input_blocks.{1,2}.1.* must NOT be a thing — stage 0 has no attentions.
        assert!(apply_sdxl_unet_rename(
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        )
        .is_none());
    }

    #[test]
    fn unet_input_block_stage_2_attention_with_transformer_layer_5() {
        // SDXL stage 2 packs 10 transformer layers per attention block —
        // verify the rename preserves the inner `transformer_blocks.5.*`
        // suffix verbatim. block_idx 7 = down_blocks.2 first slot.
        assert_eq!(
            apply_sdxl_unet_rename(
                "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_q.weight",
            )
            .as_deref(),
            Some("down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_q.weight"),
        );
    }

    #[test]
    fn unet_output_block_top_stage_no_attention() {
        // SDXL up-stage 2 (top, low-res) is resnet-only, mirroring down-stage 0.
        // output_blocks.6.0 → up_blocks.2.resnets.0; .1.* must not exist.
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.output_blocks.6.0.in_layers.0.weight",)
                .as_deref(),
            Some("up_blocks.2.resnets.0.norm1.weight"),
        );
        assert!(apply_sdxl_unet_rename(
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight",
        )
        .is_none());
        // Stage 1 upsampler still lives at output_blocks.5.2.
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.output_blocks.5.2.conv.weight")
                .as_deref(),
            Some("up_blocks.1.upsamplers.0.conv.weight"),
        );
    }

    #[test]
    fn unet_label_emb_to_add_embedding() {
        // SDXL conditions on a size/crop vector via `label_emb` (LDM) =
        // `add_embedding` (diffusers). SD1.5 has no equivalent — this is
        // exclusive to the SDXL rename surface.
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.label_emb.0.0.weight").as_deref(),
            Some("add_embedding.linear_1.weight"),
        );
        assert_eq!(
            apply_sdxl_unet_rename("model.diffusion_model.label_emb.0.2.bias").as_deref(),
            Some("add_embedding.linear_2.bias"),
        );
    }

    #[test]
    fn clip_l_strip_new_prefix() {
        // SDXL CLIP-L outer prefix differs from SD1.5
        // (`conditioner.embedders.0.transformer.*` vs
        // `cond_stage_model.transformer.*`); inner layout is identical.
        assert_eq!(
            apply_sdxl_clip_l_rename(
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            )
            .as_deref(),
            Some("text_model.encoder.layers.0.self_attn.q_proj.weight"),
        );
    }

    #[test]
    fn clip_g_resblock_layer_norm_direct() {
        // OpenCLIP `ln_1` / `ln_2` → diffusers `layer_norm1` / `layer_norm2`.
        let outputs = apply_sdxl_clip_g_rename(
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight",
        )
        .expect("ln_1 must rename");
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0],
            RenameOutput::Direct("text_model.encoder.layers.0.layer_norm1.weight".to_string()),
        );
    }

    #[test]
    fn clip_g_attn_in_proj_weight_splits_q_k_v() {
        // `attn.in_proj_weight` shape [3d, d] → three FusedSlice outputs
        // with axis=0, num_components=3, component ∈ {0, 1, 2}.
        let outputs = apply_sdxl_clip_g_rename(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
        )
        .expect("in_proj_weight must rename");
        assert_eq!(outputs.len(), 3, "Q/K/V must produce three slice outputs");
        let expected: Vec<RenameOutput> = vec![
            RenameOutput::FusedSlice {
                diffusers_key: "text_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
                axis: 0,
                component: 0,
                num_components: 3,
            },
            RenameOutput::FusedSlice {
                diffusers_key: "text_model.encoder.layers.0.self_attn.k_proj.weight".to_string(),
                axis: 0,
                component: 1,
                num_components: 3,
            },
            RenameOutput::FusedSlice {
                diffusers_key: "text_model.encoder.layers.0.self_attn.v_proj.weight".to_string(),
                axis: 0,
                component: 2,
                num_components: 3,
            },
        ];
        assert_eq!(outputs, expected);
    }

    #[test]
    fn clip_g_attn_in_proj_bias_splits_q_k_v() {
        // The bias slab follows the same row-wise split as the weight.
        let outputs = apply_sdxl_clip_g_rename(
            "conditioner.embedders.1.model.transformer.resblocks.7.attn.in_proj_bias",
        )
        .expect("in_proj_bias must rename");
        assert_eq!(outputs.len(), 3);
        for (i, comp) in outputs.iter().enumerate() {
            match comp {
                RenameOutput::FusedSlice {
                    diffusers_key,
                    axis,
                    component,
                    num_components,
                } => {
                    let expected_letter = ["q", "k", "v"][i];
                    assert_eq!(
                        diffusers_key,
                        &format!(
                            "text_model.encoder.layers.7.self_attn.{expected_letter}_proj.bias"
                        ),
                    );
                    assert_eq!(*axis, 0);
                    assert_eq!(*component, i);
                    assert_eq!(*num_components, 3);
                }
                _ => panic!("expected FusedSlice for in_proj_bias, got {comp:?}"),
            }
        }
    }

    #[test]
    fn clip_g_mlp_c_fc_renames_to_fc1() {
        let outputs = apply_sdxl_clip_g_rename(
            "conditioner.embedders.1.model.transformer.resblocks.3.mlp.c_fc.weight",
        )
        .expect("mlp.c_fc must rename");
        assert_eq!(
            outputs,
            vec![RenameOutput::Direct(
                "text_model.encoder.layers.3.mlp.fc1.weight".to_string()
            )],
        );
    }

    #[test]
    fn clip_g_text_projection_to_text_projection_weight() {
        // OpenCLIP ships `text_projection` as a raw tensor (no `.weight`);
        // diffusers expects `text_projection.weight`.
        let outputs = apply_sdxl_clip_g_rename("conditioner.embedders.1.model.text_projection")
            .expect("text_projection must rename");
        assert_eq!(
            outputs,
            vec![RenameOutput::Direct("text_projection.weight".to_string())],
        );
    }

    #[test]
    fn unrecognized_keys_return_none() {
        assert!(apply_sdxl_unet_rename("denoiser.sigmas").is_none());
        assert!(apply_sdxl_unet_rename(
            "model.diffusion_model.input_blocks.99.0.in_layers.0.weight"
        )
        .is_none());
        assert!(apply_sdxl_clip_l_rename("conditioner.unrelated.thing").is_none());
        assert!(apply_sdxl_clip_g_rename("conditioner.embedders.1.model.unknown.thing").is_none());
        // Defensive: SD1.5's CLIP-L prefix must NOT match the SDXL CLIP-L renamer.
        assert!(apply_sdxl_clip_l_rename(
            "cond_stage_model.transformer.text_model.final_layer_norm.weight",
        )
        .is_none());
    }

    // ─── Round 2 — build_sdxl_remap integration with synthetic safetensors ─

    #[test]
    fn build_sdxl_remap_routes_keys_per_component_with_clip_g_fused_split() {
        use crate::loader::single_file::{load, SingleFileBundle};
        use mold_catalog::families::Family;
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;
        use std::path::PathBuf;

        let path: PathBuf = std::env::temp_dir().join(format!(
            "mold-loader-sdxl-remap-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        // Synthesise a tiny SDXL-shaped fixture: 3-5 keys per component
        // including the CLIP-G `attn.in_proj_weight` slab that exercises
        // the fused QKV split.
        let keys: &[&str] = &[
            // UNet (5)
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_q.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.5.2.conv.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            // VAE (3) — same naming as SD1.5 (audit point 3).
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "first_stage_model.decoder.up.3.block.1.conv1.weight",
            "first_stage_model.quant_conv.weight",
            // CLIP-L (3)
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
            // CLIP-G (4 — including the fused QKV slab)
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight",
            "conditioner.embedders.1.model.text_projection",
            "conditioner.embedders.1.model.token_embedding.weight",
            // Stray (filtered upstream into unknown_keys)
            "denoiser.sigmas",
        ];

        let f32_zero = 0.0f32.to_le_bytes().to_vec();
        let buffers: Vec<Vec<u8>> = keys.iter().map(|_| f32_zero.clone()).collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, buf) in keys.iter().zip(buffers.iter()) {
            tensors.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, &path).unwrap();

        let bundle: SingleFileBundle = load(&path, Family::Sdxl).expect("load partition");
        let remap = build_sdxl_remap(&bundle).expect("build remap");

        // UNet — every renamed key resolves back to the original A1111 key.
        assert_eq!(
            remap.unet.get("conv_in.weight").map(|s| s.as_str()),
            Some("model.diffusion_model.input_blocks.0.0.weight"),
        );
        assert_eq!(
            remap
                .unet
                .get("down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_q.weight")
                .map(|s| s.as_str()),
            Some("model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_q.weight",),
        );
        assert_eq!(
            remap
                .unet
                .get("up_blocks.1.upsamplers.0.conv.weight")
                .map(|s| s.as_str()),
            Some("model.diffusion_model.output_blocks.5.2.conv.weight"),
        );
        assert_eq!(
            remap
                .unet
                .get("add_embedding.linear_1.weight")
                .map(|s| s.as_str()),
            Some("model.diffusion_model.label_emb.0.0.weight"),
        );

        // VAE — shared rename helper handles encoder + decoder + quant_conv.
        assert!(remap
            .vae
            .contains_key("encoder.down_blocks.0.resnets.0.norm1.weight"));
        assert!(remap
            .vae
            .contains_key("decoder.up_blocks.0.resnets.1.conv1.weight"));
        assert!(remap.vae.contains_key("quant_conv.weight"));

        // CLIP-L — prefix-stripped pass-through.
        assert!(remap
            .clip_l
            .contains_key("text_model.encoder.layers.0.self_attn.q_proj.weight"));
        assert!(remap
            .clip_l
            .contains_key("text_model.embeddings.token_embedding.weight"));
        assert!(remap
            .clip_l
            .contains_key("text_model.final_layer_norm.weight"));

        // CLIP-G fused QKV: the single `attn.in_proj_weight` slab expands
        // into THREE diffusers entries — q_proj, k_proj, v_proj — sharing
        // the same source A1111 key with component ∈ {0, 1, 2}.
        let q = remap
            .clip_g
            .get("text_model.encoder.layers.0.self_attn.q_proj.weight")
            .expect("q_proj must be present after fused split");
        assert_eq!(
            q.0.as_str(),
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
        );
        match &q.1 {
            RenameOutput::FusedSlice {
                axis,
                component,
                num_components,
                ..
            } => {
                assert_eq!(*axis, 0);
                assert_eq!(*component, 0);
                assert_eq!(*num_components, 3);
            }
            _ => panic!("q_proj must be FusedSlice, got {:?}", q.1),
        }
        // K and V share the source but differ on `component`.
        let k = remap
            .clip_g
            .get("text_model.encoder.layers.0.self_attn.k_proj.weight")
            .expect("k_proj");
        let v = remap
            .clip_g
            .get("text_model.encoder.layers.0.self_attn.v_proj.weight")
            .expect("v_proj");
        assert_eq!(k.0, q.0);
        assert_eq!(v.0, q.0);
        match (&k.1, &v.1) {
            (
                RenameOutput::FusedSlice { component: kc, .. },
                RenameOutput::FusedSlice { component: vc, .. },
            ) => {
                assert_eq!(*kc, 1);
                assert_eq!(*vc, 2);
            }
            _ => panic!("k_proj / v_proj must both be FusedSlice"),
        }

        // CLIP-G direct renames also land — ln_1, text_projection, token_embedding.
        assert!(remap
            .clip_g
            .contains_key("text_model.encoder.layers.0.layer_norm1.weight"));
        assert!(remap.clip_g.contains_key("text_projection.weight"));
        assert!(remap
            .clip_g
            .contains_key("text_model.embeddings.token_embedding.weight"));

        // No unmapped keys — the single stray (`denoiser.sigmas`) is
        // filtered upstream into bundle.unknown_keys, never reaches the
        // remap.
        assert!(
            remap.unmapped.is_empty(),
            "expected no unmapped keys, got {:?}",
            remap.unmapped
        );

        let _ = std::fs::remove_file(path);
    }
}
