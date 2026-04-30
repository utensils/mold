//! Shared VAE A1111 → diffusers key rename pass.
//!
//! Phase-2.2 audit point 3 confirmed SD1.5 and SDXL ship the **identical**
//! VAE weights — same 248 keys, same shapes, same naming. So both family
//! loaders can share a single rename helper. Factored out of
//! `sd15_keys::apply_sd15_vae_rename` during phase 2.5 when the SDXL
//! loader landed.
//!
//! Diffusers naming reference:
//! `~/.cargo/registry/src/index.crates.io-*/candle-transformers-mold-0.9.10/
//! src/models/stable_diffusion/vae.rs` — every key here corresponds to a
//! `vs.pp("…")` chain in candle's `AutoEncoderKL::new`.

/// Translate a Civitai single-file VAE key (`first_stage_model.*` LDM
/// layout) to its diffusers equivalent.
///
/// Returns `None` for any key that does not start with
/// `first_stage_model.` or that names a stage/sub-path with no rename
/// rule. Callers must tolerate `None` per the audit's "log + drop"
/// contract.
pub fn apply_vae_rename(a1111_key: &str) -> Option<String> {
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

/// Split `"<int>.<rest>"` into `(<int>, <rest>)`.
fn split_idx(s: &str) -> Option<(usize, &str)> {
    let (head, tail) = s.split_once('.')?;
    let idx: usize = head.parse().ok()?;
    Some((idx, tail))
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

    #[test]
    fn encoder_down_block_resnet_norm1() {
        // Identical to the SD1.5 case: encoder.down.0.block.0.norm1 →
        // encoder.down_blocks.0.resnets.0.norm1.
        assert_eq!(
            apply_vae_rename("first_stage_model.encoder.down.0.block.0.norm1.weight").as_deref(),
            Some("encoder.down_blocks.0.resnets.0.norm1.weight"),
        );
    }

    #[test]
    fn decoder_up_stage_reversal() {
        // LDM decoder.up.3 is the *highest* resolution stage; diffusers
        // reverses to decoder.up_blocks.0. SDXL VAE has the same 4-stage
        // depth as SD1.5 (`block_out_channels = [128, 256, 512, 512]`).
        assert_eq!(
            apply_vae_rename("first_stage_model.decoder.up.3.block.1.conv1.weight").as_deref(),
            Some("decoder.up_blocks.0.resnets.1.conv1.weight"),
        );
    }

    #[test]
    fn quant_conv_pass_through() {
        assert_eq!(
            apply_vae_rename("first_stage_model.quant_conv.weight").as_deref(),
            Some("quant_conv.weight"),
        );
        assert_eq!(
            apply_vae_rename("first_stage_model.post_quant_conv.bias").as_deref(),
            Some("post_quant_conv.bias"),
        );
    }

    #[test]
    fn mid_attn_proj_out_renames_to_to_out() {
        // mid.attn_1.proj_out → mid_block.attentions.0.to_out.0
        assert_eq!(
            apply_vae_rename("first_stage_model.encoder.mid.attn_1.proj_out.weight").as_deref(),
            Some("encoder.mid_block.attentions.0.to_out.0.weight"),
        );
    }

    #[test]
    fn unknown_returns_none() {
        assert!(apply_vae_rename("first_stage_model.unknown.thing").is_none());
        assert!(apply_vae_rename("not_a_vae_prefix.thing").is_none());
    }
}
