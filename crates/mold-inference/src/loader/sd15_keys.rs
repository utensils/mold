//! SD1.5 A1111 → diffusers key rename pass (phase 2.4).
//!
//! Bridges the original LDM tensor key naming used in Civitai single-file
//! checkpoints to the diffusers naming candle's
//! `stable_diffusion::{unet_2d::UNet2DConditionModel, vae::AutoEncoderKL,
//! clip::ClipTextTransformer}` constructors expect.
//!
//! Three pure-data rename helpers — one per component (UNet / VAE / CLIP-L)
//! — plus a `build_sd15_remap` integration that takes a `SingleFileBundle`
//! and returns the diffusers→a1111 lookup the SimpleBackend will index by.
//!
//! Naming reference: every diffusers key in the table below comes from a
//! `vs.pp("…")` chain in
//! `~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/
//! src/models/stable_diffusion/{unet_2d,unet_2d_blocks,vae,clip,resnet,
//! attention,embeddings}.rs`. Verified against the depth-2 audit dump in
//! `tasks/catalog-expansion-phase-2-tensor-audit.md` (DreamShaper 8 / SD 1.5).

use crate::loader::single_file::SingleFileBundle;
use std::collections::BTreeMap;
use thiserror::Error;

/// Diffusers→a1111 lookup for the three SD1.5 components, ready to feed
/// a `SimpleBackend` that translates each `vb.get(diffusers_key)` to the
/// original A1111 key in the mmap'd safetensors.
#[derive(Debug, Default, Clone)]
pub struct Sd15Remap {
    pub unet: BTreeMap<String, String>,
    pub vae: BTreeMap<String, String>,
    pub clip_l: BTreeMap<String, String>,
    /// A1111 keys with no rename rule. Per the audit, these are tolerated:
    /// log + drop, not error. (`denoiser.sigmas` is the canonical example,
    /// but it lands in `SingleFileBundle::unknown_keys` upstream and never
    /// reaches the remap.)
    pub unmapped: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RemapError {
    // No fatal cases yet — rename rules tolerate stragglers via `unmapped`.
    // The variant slot is held open so the public signature stays
    // `Result<_, RemapError>` and downstream code doesn't churn when 2.5
    // adds an SDXL CLIP-G OpenCLIP→HF pass that *can* fail (e.g. malformed
    // `attn.in_proj_weight` shape).
    #[error("placeholder — no fatal rename failures in SD1.5 yet")]
    #[allow(dead_code)]
    Placeholder,
}

/// Translate a SD1.5 A1111 UNet key to its diffusers equivalent.
///
/// Returns `None` if `a1111_key` does not start with `model.diffusion_model.`
/// or names an outer block / inner field that has no rename rule. Callers
/// must tolerate `None` — the audit's contract is "log + drop", not error.
pub fn apply_sd15_unet_rename(a1111_key: &str) -> Option<String> {
    let inner = a1111_key.strip_prefix("model.diffusion_model.")?;

    // Top-level constants
    if let Some(out) = match inner {
        "time_embed.0.weight" => Some("time_embedding.linear_1.weight"),
        "time_embed.0.bias" => Some("time_embedding.linear_1.bias"),
        "time_embed.2.weight" => Some("time_embedding.linear_2.weight"),
        "time_embed.2.bias" => Some("time_embedding.linear_2.bias"),
        "out.0.weight" => Some("conv_norm_out.weight"),
        "out.0.bias" => Some("conv_norm_out.bias"),
        "out.2.weight" => Some("conv_out.weight"),
        "out.2.bias" => Some("conv_out.bias"),
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

/// Translate a SD1.5 A1111 VAE key to its diffusers equivalent.
pub fn apply_sd15_vae_rename(a1111_key: &str) -> Option<String> {
    let inner = a1111_key.strip_prefix("first_stage_model.")?;

    // Top-level constants — quant convs are 1×1 convs at the latent boundary
    // and keep identical naming between LDM and diffusers.
    if matches!(
        inner,
        "quant_conv.weight" | "quant_conv.bias" | "post_quant_conv.weight" | "post_quant_conv.bias"
    ) {
        return Some(inner.to_string());
    }

    if let Some(rest) = inner.strip_prefix("encoder.") {
        return rename_vae_half(rest, VaeHalf::Encoder).map(|s| format!("encoder.{s}"));
    }
    if let Some(rest) = inner.strip_prefix("decoder.") {
        return rename_vae_half(rest, VaeHalf::Decoder).map(|s| format!("decoder.{s}"));
    }

    None
}

/// Translate a SD1.5 A1111 CLIP-L key to its diffusers equivalent.
///
/// SD1.5 CLIP-L lives at `cond_stage_model.transformer.text_model.*` —
/// stripping the leading `cond_stage_model.transformer.` lands keys at
/// `text_model.*`, which is exactly what
/// `candle_transformers::models::stable_diffusion::clip::ClipTextTransformer`
/// asks for (see `clip.rs:350`: `let vs = vs.pp("text_model");`).
pub fn apply_sd15_clip_l_rename(a1111_key: &str) -> Option<String> {
    let stripped = a1111_key.strip_prefix("cond_stage_model.transformer.")?;
    if !stripped.starts_with("text_model.") && stripped != "text_model" {
        return None;
    }
    Some(stripped.to_string())
}

/// Apply the SD1.5 rename rules to every key in the bundle, producing
/// component-keyed diffusers→a1111 maps the engine can hand to a custom
/// `SimpleBackend`. Unmapped keys land in `Sd15Remap::unmapped` and are
/// logged (not errored) per the audit's "tolerate stray tensors" rule.
pub fn build_sd15_remap(bundle: &SingleFileBundle) -> Result<Sd15Remap, RemapError> {
    let mut out = Sd15Remap::default();

    apply_into(
        &bundle.unet_keys,
        &mut out.unet,
        &mut out.unmapped,
        apply_sd15_unet_rename,
    );
    apply_into(
        &bundle.vae_keys,
        &mut out.vae,
        &mut out.unmapped,
        apply_sd15_vae_rename,
    );
    apply_into(
        &bundle.clip_l_keys,
        &mut out.clip_l,
        &mut out.unmapped,
        apply_sd15_clip_l_rename,
    );

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

fn rename_unet_input_block(block_idx: usize, sub_idx: usize, suffix: &str) -> Option<String> {
    // SD1.5 input_blocks layout (LDM → diffusers):
    //   0    : conv_in
    //   1, 2 : down_blocks.0.{resnets|attentions}.{0|1}
    //   3    : down_blocks.0.downsamplers.0
    //   4, 5 : down_blocks.1.{resnets|attentions}.{0|1}
    //   6    : down_blocks.1.downsamplers.0
    //   7, 8 : down_blocks.2.{resnets|attentions}.{0|1}
    //   9    : down_blocks.2.downsamplers.0
    //  10,11 : down_blocks.3.resnets.{0|1}    (no attentions in stage 3)
    if block_idx == 0 && sub_idx == 0 {
        return Some(format!("conv_in.{suffix}"));
    }

    let stage_idx = (block_idx - 1) / 3; // 1..=3 → 0, 4..=6 → 1, …
    let in_stage = (block_idx - 1) % 3; //  0,1,2 (resnet, resnet, downsampler)

    if stage_idx == 3 {
        // Bottom stage: only two resnets, no attentions, no downsampler.
        // input_blocks.10.0 → resnets.0 ; input_blocks.11.0 → resnets.1
        let resnet_idx = block_idx - 10;
        if sub_idx == 0 {
            return Some(format!(
                "down_blocks.3.resnets.{resnet_idx}.{}",
                rename_resnet_inner(suffix)?
            ));
        }
        return None;
    }

    match (in_stage, sub_idx) {
        (0, 0) | (1, 0) => Some(format!(
            "down_blocks.{stage_idx}.resnets.{in_stage}.{}",
            rename_resnet_inner(suffix)?
        )),
        (0, 1) | (1, 1) => Some(format!(
            "down_blocks.{stage_idx}.attentions.{in_stage}.{suffix}",
        )),
        (2, 0) => Some(format!(
            "down_blocks.{stage_idx}.downsamplers.0.{}",
            rename_downsampler_inner(suffix)?
        )),
        _ => None,
    }
}

fn rename_unet_middle_block(sub_idx: usize, suffix: &str) -> Option<String> {
    // middle_block: [resnet, attention, resnet] → mid_block.{resnets,attentions}
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

fn rename_unet_output_block(block_idx: usize, sub_idx: usize, suffix: &str) -> Option<String> {
    // SD1.5 output_blocks layout (LDM → diffusers):
    //   0,1,2  : up_blocks.0.resnets.{0,1,2}    (no attentions in top stage)
    //      .2.1: up_blocks.0.upsamplers.0
    //   3,4,5  : up_blocks.1.{resnets|attentions}.{0,1,2}
    //      .5.2: up_blocks.1.upsamplers.0
    //   6,7,8  : up_blocks.2.{resnets|attentions}.{0,1,2}
    //      .8.2: up_blocks.2.upsamplers.0
    //   9,10,11: up_blocks.3.{resnets|attentions}.{0,1,2}    (no upsampler at bottom)
    let stage_idx = block_idx / 3;
    let resnet_idx = block_idx % 3;

    if stage_idx == 0 {
        // Top stage: resnets only, then upsampler at output_blocks.2.1.
        match sub_idx {
            0 => Some(format!(
                "up_blocks.0.resnets.{resnet_idx}.{}",
                rename_resnet_inner(suffix)?
            )),
            1 if block_idx == 2 => Some(format!(
                "up_blocks.0.upsamplers.0.{}",
                rename_upsampler_inner(suffix)?
            )),
            _ => None,
        }
    } else {
        // Stages 1, 2, 3: resnet at .0, attention at .1, optional upsampler at .2.
        match sub_idx {
            0 => Some(format!(
                "up_blocks.{stage_idx}.resnets.{resnet_idx}.{}",
                rename_resnet_inner(suffix)?
            )),
            1 => Some(format!(
                "up_blocks.{stage_idx}.attentions.{resnet_idx}.{suffix}",
            )),
            2 if resnet_idx == 2 && stage_idx != 3 => Some(format!(
                "up_blocks.{stage_idx}.upsamplers.0.{}",
                rename_upsampler_inner(suffix)?
            )),
            _ => None,
        }
    }
}

/// LDM ResnetBlock inner naming → diffusers ResnetBlock2D.
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

#[derive(Copy, Clone)]
enum VaeHalf {
    Encoder,
    Decoder,
}

/// VAE encoder/decoder rename. Inputs are the suffix after stripping
/// `first_stage_model.{encoder|decoder}.` and outputs are the diffusers
/// suffix without the leading `encoder.`/`decoder.` segment.
fn rename_vae_half(suffix: &str, half: VaeHalf) -> Option<String> {
    // Edge cells common to both halves.
    if let Some(out) = match suffix {
        "conv_in.weight" => Some("conv_in.weight"),
        "conv_in.bias" => Some("conv_in.bias"),
        "conv_out.weight" => Some("conv_out.weight"),
        "conv_out.bias" => Some("conv_out.bias"),
        "norm_out.weight" => Some("conv_norm_out.weight"),
        "norm_out.bias" => Some("conv_norm_out.bias"),
        _ => None,
    } {
        return Some(out.to_string());
    }

    if let Some(rest) = suffix.strip_prefix("mid.") {
        return rename_vae_mid(rest);
    }

    match half {
        VaeHalf::Encoder => {
            if let Some(rest) = suffix.strip_prefix("down.") {
                let (stage, rest) = split_idx(rest)?;
                if let Some(rest) = rest.strip_prefix("block.") {
                    let (block, tail) = split_idx(rest)?;
                    return Some(format!(
                        "down_blocks.{stage}.resnets.{block}.{}",
                        rename_resnet_inner_vae(tail)?
                    ));
                }
                if let Some(tail) = rest.strip_prefix("downsample.") {
                    return Some(format!("down_blocks.{stage}.downsamplers.0.{tail}"));
                }
            }
            None
        }
        VaeHalf::Decoder => {
            if let Some(rest) = suffix.strip_prefix("up.") {
                let (ldm_stage, rest) = split_idx(rest)?;
                // LDM stores decoder up.0 as the highest resolution stage,
                // diffusers reverses it: `up_blocks.0` is the lowest. With 4
                // VAE stages, diffusers_stage = 3 - ldm_stage.
                let diff_stage = 3usize.checked_sub(ldm_stage)?;
                if let Some(rest) = rest.strip_prefix("block.") {
                    let (block, tail) = split_idx(rest)?;
                    return Some(format!(
                        "up_blocks.{diff_stage}.resnets.{block}.{}",
                        rename_resnet_inner_vae(tail)?
                    ));
                }
                if let Some(tail) = rest.strip_prefix("upsample.") {
                    return Some(format!("up_blocks.{diff_stage}.upsamplers.0.{tail}"));
                }
            }
            None
        }
    }
}

fn rename_vae_mid(suffix: &str) -> Option<String> {
    // mid.block_{1,2}.* → mid_block.resnets.{0,1}.*
    if let Some(rest) = suffix.strip_prefix("block_1.") {
        return Some(format!(
            "mid_block.resnets.0.{}",
            rename_resnet_inner_vae(rest)?
        ));
    }
    if let Some(rest) = suffix.strip_prefix("block_2.") {
        return Some(format!(
            "mid_block.resnets.1.{}",
            rename_resnet_inner_vae(rest)?
        ));
    }
    // mid.attn_1.{q,k,v,proj_out,norm} → mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}
    if let Some(rest) = suffix.strip_prefix("attn_1.") {
        return Some(format!(
            "mid_block.attentions.0.{}",
            rename_vae_mid_attn(rest)?
        ));
    }
    None
}

fn rename_vae_mid_attn(suffix: &str) -> Option<String> {
    Some(match suffix {
        "q.weight" => "to_q.weight".to_string(),
        "q.bias" => "to_q.bias".to_string(),
        "k.weight" => "to_k.weight".to_string(),
        "k.bias" => "to_k.bias".to_string(),
        "v.weight" => "to_v.weight".to_string(),
        "v.bias" => "to_v.bias".to_string(),
        "proj_out.weight" => "to_out.0.weight".to_string(),
        "proj_out.bias" => "to_out.0.bias".to_string(),
        "norm.weight" => "group_norm.weight".to_string(),
        "norm.bias" => "group_norm.bias".to_string(),
        _ => return None,
    })
}

/// VAE resnets share the LDM ResnetBlock layout (norm1/conv1/norm2/conv2,
/// optional `nin_shortcut` instead of `skip_connection`), but they have no
/// time embedding — so `emb_layers.*` doesn't appear here.
fn rename_resnet_inner_vae(suffix: &str) -> Option<String> {
    Some(match suffix {
        "norm1.weight" => "norm1.weight".to_string(),
        "norm1.bias" => "norm1.bias".to_string(),
        "conv1.weight" => "conv1.weight".to_string(),
        "conv1.bias" => "conv1.bias".to_string(),
        "norm2.weight" => "norm2.weight".to_string(),
        "norm2.bias" => "norm2.bias".to_string(),
        "conv2.weight" => "conv2.weight".to_string(),
        "conv2.bias" => "conv2.bias".to_string(),
        "nin_shortcut.weight" => "conv_shortcut.weight".to_string(),
        "nin_shortcut.bias" => "conv_shortcut.bias".to_string(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Round 1 — pure rename rules, no I/O ─────────────────────────────

    #[test]
    fn unet_input_block_0_to_conv_in() {
        // SD1.5 input_blocks.0 is the outermost conv, not a resnet.
        // Verified against candle's `unet_2d.rs:120`:
        //   `let conv_in = conv2d(in_channels, b_channels, 3, conv_cfg, vs.pp("conv_in"))?;`
        assert_eq!(
            apply_sd15_unet_rename("model.diffusion_model.input_blocks.0.0.weight").as_deref(),
            Some("conv_in.weight"),
        );
        assert_eq!(
            apply_sd15_unet_rename("model.diffusion_model.input_blocks.0.0.bias").as_deref(),
            Some("conv_in.bias"),
        );
    }

    #[test]
    fn unet_middle_block_attention_transformer_block() {
        // middle_block.0 = resnet, .1 = attention (Transformer2DModel), .2 = resnet.
        // Inner attention naming is identical (audit point 7) — the rename
        // is just the outer `middle_block.1` → `mid_block.attentions.0`.
        assert_eq!(
            apply_sd15_unet_rename(
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            )
            .as_deref(),
            Some("mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight"),
        );
    }

    #[test]
    fn unet_output_block_with_upsampler() {
        // SD1.5 output_blocks.2 = [resnet, upsampler] (top stage has no
        // attentions). The upsampler is at sub-index 1, not 2 — the
        // handoff's example was off by one for SD15. Confirm both the
        // resnet at .2.0 and the upsampler at .2.1.
        assert_eq!(
            apply_sd15_unet_rename("model.diffusion_model.output_blocks.2.0.in_layers.0.weight",)
                .as_deref(),
            Some("up_blocks.0.resnets.2.norm1.weight"),
        );
        assert_eq!(
            apply_sd15_unet_rename("model.diffusion_model.output_blocks.2.1.conv.weight")
                .as_deref(),
            Some("up_blocks.0.upsamplers.0.conv.weight"),
        );
        // Stage 1's upsampler lives at output_blocks.5.2 (resnet, attention,
        // upsampler). That's the case the handoff was likely thinking of.
        assert_eq!(
            apply_sd15_unet_rename("model.diffusion_model.output_blocks.5.2.conv.weight")
                .as_deref(),
            Some("up_blocks.1.upsamplers.0.conv.weight"),
        );
    }

    #[test]
    fn vae_encoder_down_block_resnet_norm1() {
        assert_eq!(
            apply_sd15_vae_rename("first_stage_model.encoder.down.0.block.0.norm1.weight",)
                .as_deref(),
            Some("encoder.down_blocks.0.resnets.0.norm1.weight"),
        );
    }

    #[test]
    fn clip_l_text_model_self_attn_q_proj() {
        // SD1.5 CLIP-L: strip `cond_stage_model.transformer.` and pass through.
        assert_eq!(
            apply_sd15_clip_l_rename(
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            )
            .as_deref(),
            Some("text_model.encoder.layers.0.self_attn.q_proj.weight"),
        );
    }

    #[test]
    fn unrecognized_keys_return_none() {
        // Defensive: unknown prefixes / sub-paths return None so callers
        // route them through `Sd15Remap::unmapped` (log + drop) instead of
        // ending up in a component map with a bogus diffusers name.
        assert!(apply_sd15_unet_rename("denoiser.sigmas").is_none());
        assert!(apply_sd15_unet_rename("model.diffusion_model.input_blocks.0.0.unknown").is_some());
        // input_blocks beyond the SD1.5 grid (12+) have no rule.
        assert!(apply_sd15_unet_rename(
            "model.diffusion_model.input_blocks.99.0.in_layers.0.weight"
        )
        .is_none());
        assert!(apply_sd15_vae_rename("first_stage_model.unknown.thing").is_none());
        assert!(apply_sd15_clip_l_rename("cond_stage_model.unrelated.thing").is_none());
    }

    // ─── Round 2 — build_sd15_remap integration with synthetic safetensors ─

    #[test]
    fn build_sd15_remap_routes_keys_per_component() {
        use crate::loader::single_file::{load, SingleFileBundle};
        use mold_catalog::families::Family;
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;
        use std::path::PathBuf;

        let path: PathBuf = std::env::temp_dir().join(format!(
            "mold-loader-sd15-remap-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        // Synthesise a tiny single-file with 3 keys per component plus one
        // stray (`denoiser.sigmas`) that 2.3's loader filters into
        // `unknown_keys` before the remap ever sees it.
        let keys: &[&str] = &[
            // UNet (3)
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.5.2.conv.weight",
            // VAE (3)
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "first_stage_model.decoder.up.3.block.1.conv1.weight",
            "first_stage_model.quant_conv.weight",
            // CLIP-L (3)
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
            "cond_stage_model.transformer.text_model.final_layer_norm.weight",
            // Stray (filtered upstream into unknown_keys, never reaches build_sd15_remap)
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

        let bundle: SingleFileBundle = load(&path, Family::Sd15).expect("load partition");
        let remap = build_sd15_remap(&bundle).expect("build remap");

        // UNet: every diffusers key resolves back to the original A1111 key.
        assert_eq!(
            remap.unet.get("conv_in.weight").map(|s| s.as_str()),
            Some("model.diffusion_model.input_blocks.0.0.weight"),
        );
        assert_eq!(
            remap
                .unet
                .get("mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight")
                .map(|s| s.as_str()),
            Some("model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"),
        );
        assert_eq!(
            remap
                .unet
                .get("up_blocks.1.upsamplers.0.conv.weight")
                .map(|s| s.as_str()),
            Some("model.diffusion_model.output_blocks.5.2.conv.weight"),
        );

        // VAE: encoder + decoder + quant_conv all resolve. candle's
        // `AutoEncoderKL::new` walks `vs.pp("encoder")` / `vs.pp("decoder")`
        // (vae.rs:345/351), so the diffusers keys in the remap include the
        // `encoder.`/`decoder.` segment.
        assert!(remap
            .vae
            .contains_key("encoder.down_blocks.0.resnets.0.norm1.weight"));
        // decoder.up.3 (LDM) → decoder.up_blocks.0 (diffusers, reversed)
        assert!(remap
            .vae
            .contains_key("decoder.up_blocks.0.resnets.1.conv1.weight"));
        assert!(remap.vae.contains_key("quant_conv.weight"));

        // CLIP-L: all three pass through with the prefix stripped.
        assert!(remap
            .clip_l
            .contains_key("text_model.encoder.layers.0.self_attn.q_proj.weight"));
        assert!(remap
            .clip_l
            .contains_key("text_model.embeddings.token_embedding.weight"));
        assert!(remap
            .clip_l
            .contains_key("text_model.final_layer_norm.weight"));

        // Sd15Remap.unmapped should be empty for this fixture: every key
        // partitioned into a component bucket has a rename rule, and the
        // single stray (`denoiser.sigmas`) was filtered upstream.
        assert!(
            remap.unmapped.is_empty(),
            "expected no unmapped keys, got {:?}",
            remap.unmapped
        );

        let _ = std::fs::remove_file(path);
    }
}
