//! Common Kohya naming for SD1.5 and SDXL. Dotted-family allowlists stay local.

/// Map the Kohya / sd-scripts flattened form (with the `lora_unet_` prefix
/// already stripped) to a SD1.5/SDXL candle key. The flattening turns every `.` in
/// the original module path into `_`, so we walk it from the front:
///
/// 1. Block kind: `down_blocks_{i}_`, `mid_block_`, `up_blocks_{i}_`.
/// 2. Inner kind: `attentions_{j}_...`, `resnets_{j}_...`, or for downsamplers
///    `downsamplers_0_conv`, `upsamplers_0_conv`.
///
/// Returns `None` for any leaf the SD1.5/SDXL UNet can't host (text-encoder LoRAs
/// (`lora_te_*`), unrecognised sub-trees, etc.) — the caller logs and skips.
pub(crate) fn map_kohya_unet_key(rest: &str) -> Option<String> {
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
    // possible but never appear in observed Civitai SD1.5/SDXL LoRAs — left out
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
/// is one of the SD1.5/SDXL attention/FF leaves.
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
