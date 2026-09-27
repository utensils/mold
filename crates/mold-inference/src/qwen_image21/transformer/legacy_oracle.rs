//! A FROZEN copy of the v0.32 text-to-image forward (`transformer.rs` at
//! `c1dfa872`), kept only as a test oracle.
//!
//! The layout-generalized forward must reproduce this bit for bit on every
//! text-to-image input, uncached and cached, because archived v0.32 seeds
//! must render the same bytes under `MOLD_ATTN=math`. Do not "fix" or refactor
//! this file: its value is that it does not move. It reads the live module's
//! weights through the parent's private fields.

use crate::qwen_image21::linear::Q21Linear;
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};

use super::{Attention, PrefixKv, QwenImage21Transformer, TransformerBlock};
use crate::qwen_image21::QwenImage21TextConditioning;

pub(super) enum LegacyCache<'a> {
    Disabled,
    Extract(&'a mut Vec<PrefixKv>),
    Reuse(&'a [PrefixKv]),
}

enum LegacyLayerCache<'a> {
    Disabled,
    Extract(&'a mut Vec<PrefixKv>),
    Reuse(&'a PrefixKv),
}

fn target_attention(
    attn: &Attention,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let scale = (1.0 / (attn.head_dim as f64).sqrt()) as f32;
    if attn.dispatch.fused_target
        && q.device().is_metal()
        && bias.is_none()
        && matches!(attn.head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
    {
        return candle_nn::ops::sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            None,
            false,
            scale,
            1.0,
        )
        .map_err(Into::into);
    }
    crate::attention::attention_with_bias(q, k, v, scale, bias).map_err(Into::into)
}

fn t2i_prefix_bias(valid_tokens: &[Vec<bool>], dtype: DType, device: &Device) -> Result<Tensor> {
    let batch = valid_tokens.len();
    let text_len = valid_tokens.first().map_or(0, Vec::len);
    let mut values = Vec::with_capacity(batch * text_len * text_len);
    for row in valid_tokens {
        for query in 0..text_len {
            for (key, valid) in row.iter().enumerate() {
                values.push(if key <= query && *valid {
                    0.0
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
    }
    Tensor::from_vec(values, (batch, 1, text_len, text_len), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn t2i_target_bias(
    valid_tokens: &[Vec<bool>],
    target_tokens: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let batch = valid_tokens.len();
    let text_len = valid_tokens.first().map_or(0, Vec::len);
    if valid_tokens
        .iter()
        .all(|row| row.iter().all(|value| *value))
    {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(batch * (text_len + target_tokens));
    for row in valid_tokens {
        values.extend(
            row.iter()
                .map(|valid| if *valid { 0.0 } else { f32::NEG_INFINITY }),
        );
        values.extend(std::iter::repeat_n(0.0, target_tokens));
    }
    Ok(Some(
        Tensor::from_vec(values, (batch, 1, 1, text_len + target_tokens), device)?
            .to_dtype(dtype)?,
    ))
}

fn attention_forward_t2i(
    attn: &Attention,
    hidden_states: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    valid_tokens: &[Vec<bool>],
    cache: LegacyLayerCache<'_>,
) -> Result<Tensor> {
    let (batch, sequence, inner) = hidden_states.dims3()?;
    let text_len = valid_tokens.first().map_or(0, Vec::len);
    let cached = matches!(cache, LegacyLayerCache::Reuse(_));
    let target_tokens = if cached {
        sequence
    } else {
        sequence - text_len
    };

    let (q, k, v) = if attn.fused_ops && hidden_states.device().is_metal() {
        let project = |linear: &Q21Linear, weight: &Tensor| -> Result<Tensor> {
            let xs = linear.forward(hidden_states)?;
            let normalized = candle_nn::ops::rms_norm(
                &xs.reshape((batch * sequence * attn.heads, attn.head_dim))?,
                weight,
                attn.eps as f32,
            )?
            .reshape((batch, sequence, attn.heads, attn.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
            candle_nn::rotary_emb::rope_i(
                &normalized.to_dtype(DType::F32)?,
                &rope_cos.to_dtype(DType::F32)?.contiguous()?,
                &rope_sin.to_dtype(DType::F32)?.contiguous()?,
            )?
            .to_dtype(hidden_states.dtype())
            .map_err(Into::into)
        };
        (
            project(&attn.to_q, &attn.norm_q)?,
            project(&attn.to_k, &attn.norm_k)?,
            attn.to_v
                .forward(hidden_states)?
                .reshape((batch, sequence, attn.heads, attn.head_dim))?
                .transpose(1, 2)?
                .contiguous()?,
        )
    } else {
        let project = |linear: &Q21Linear| -> Result<Tensor> {
            linear
                .forward(hidden_states)?
                .reshape((batch, sequence, attn.heads, attn.head_dim))?
                .transpose(1, 2)
                .map_err(Into::into)
        };
        let q = attn.normalize_heads(&project(&attn.to_q)?, &attn.norm_q)?;
        let k = attn.normalize_heads(&project(&attn.to_k)?, &attn.norm_k)?;
        let v = project(&attn.to_v)?;
        let q = crate::wan::model::rope::apply_rope(&q.transpose(1, 2)?, rope_cos, rope_sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = crate::wan::model::rope::apply_rope(&k.transpose(1, 2)?, rope_cos, rope_sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.contiguous()?;
        (q, k, v)
    };

    match cache {
        LegacyLayerCache::Extract(layers) => {
            layers.push(PrefixKv {
                key: k.narrow(2, 0, text_len)?.force_contiguous()?,
                value: v.narrow(2, 0, text_len)?.force_contiguous()?,
            });
        }
        LegacyLayerCache::Reuse(prefix) => {
            let k = Tensor::cat(&[&prefix.key, &k], 2)?;
            let v = Tensor::cat(&[&prefix.value, &v], 2)?;
            let bias = t2i_target_bias(valid_tokens, target_tokens, q.dtype(), q.device())?;
            let context = target_attention(attn, &q, &k, &v, bias.as_ref())?;
            return attn
                .to_out
                .forward(&context.transpose(1, 2)?.reshape((batch, sequence, inner))?);
        }
        LegacyLayerCache::Disabled => {}
    }

    let prefix_q = q.narrow(2, 0, text_len)?.contiguous()?;
    let prefix_k = k.narrow(2, 0, text_len)?.contiguous()?;
    let prefix_v = v.narrow(2, 0, text_len)?.contiguous()?;
    let prefix_bias = t2i_prefix_bias(valid_tokens, q.dtype(), q.device())?;
    let prefix = crate::attention::attention_with_bias(
        &prefix_q,
        &prefix_k,
        &prefix_v,
        (1.0 / (attn.head_dim as f64).sqrt()) as f32,
        Some(&prefix_bias),
    )?;
    let target_q = q.narrow(2, text_len, target_tokens)?.contiguous()?;
    let target_bias = t2i_target_bias(valid_tokens, target_tokens, q.dtype(), q.device())?;
    let target = target_attention(attn, &target_q, &k, &v, target_bias.as_ref())?;
    let context = Tensor::cat(&[&prefix, &target], 2)?;
    attn.to_out
        .forward(&context.transpose(1, 2)?.reshape((batch, sequence, inner))?)
}

fn block_forward_t2i(
    block: &TransformerBlock,
    hidden_states: &Tensor,
    modulation: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    valid_tokens: &[Vec<bool>],
    cache: LegacyLayerCache<'_>,
) -> Result<Tensor> {
    let dim = hidden_states.dim(D::Minus1)?;
    let mod1 = modulation.narrow(D::Minus1, 0, 2 * dim)?;
    let mod2 = modulation.narrow(D::Minus1, 2 * dim, 2 * dim)?;
    let (normalized, gate) =
        TransformerBlock::modulate(block.norm1.forward(hidden_states)?, &mod1)?;
    let attn = attention_forward_t2i(
        &block.attn,
        &normalized,
        rope_cos,
        rope_sin,
        valid_tokens,
        cache,
    )?;
    let hidden_states = (hidden_states + gate.tanh()?.broadcast_mul(&attn)?)?;
    let (normalized, gate) =
        TransformerBlock::modulate(block.norm2.forward(&hidden_states)?, &mod2)?;
    let hidden_states = (&hidden_states
        + gate
            .tanh()?
            .broadcast_mul(&block.mlp.forward(&normalized)?)?)?;
    if hidden_states.dtype() == DType::F16 {
        return hidden_states
            .clamp(-65_504.0f32, 65_504.0f32)
            .map_err(Into::into);
    }
    Ok(hidden_states)
}

/// The v0.32 `t2i_rope`, verbatim.
pub(super) fn t2i_rope(
    transformer: &QwenImage21Transformer,
    text_len: usize,
    latent_height: usize,
    latent_width: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let cfg = &transformer.cfg;
    let target_len = latent_height * latent_width;
    let mut coords = Vec::with_capacity(text_len + target_len);
    for index in 0..text_len {
        let position = index as i32;
        coords.push([position, position, position]);
    }
    let frame = text_len as i32;
    let h_start = -(latent_height as i32 - latent_height as i32 / 2);
    let w_start = -(latent_width as i32 - latent_width as i32 / 2);
    for height in 0..latent_height {
        for width in 0..latent_width {
            coords.push([frame, h_start + height as i32, w_start + width as i32]);
        }
    }
    let mut cos = Vec::with_capacity(coords.len() * (cfg.attention_head_dim / 2));
    let mut sin = Vec::with_capacity(coords.len() * (cfg.attention_head_dim / 2));
    for coordinate in coords {
        for (axis, axis_dim) in cfg.axes_dims_rope.iter().copied().enumerate() {
            for index in (0..axis_dim).step_by(2) {
                let frequency = 1.0 / 10_000.0f64.powf(index as f64 / axis_dim as f64);
                let angle = coordinate[axis] as f64 * frequency;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    let sequence = text_len + target_len;
    let dtype = if device.is_metal() { DType::F32 } else { dtype };
    Ok((
        Tensor::from_vec(cos, (sequence, cfg.attention_head_dim / 2), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (sequence, cfg.attention_head_dim / 2), device)?.to_dtype(dtype)?,
    ))
}

/// The v0.32 `forward_with_cache`, verbatim apart from reading the parent's
/// fields through `transformer`.
pub(super) fn forward_with_cache(
    transformer: &QwenImage21Transformer,
    latents: &Tensor,
    timestep: f64,
    conditioning: &QwenImage21TextConditioning,
    latent_height: usize,
    latent_width: usize,
    mut cache: LegacyCache<'_>,
) -> Result<Tensor> {
    let cached = matches!(cache, LegacyCache::Reuse(_));
    let (batch, target_tokens, _) = latents.dims3()?;
    let text_len = conditioning.sequence_length();
    let text = conditioning
        .embeddings
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let target = transformer.img_in.forward(latents)?;
    let mut hidden_states = if cached {
        target
    } else {
        Tensor::cat(&[&transformer.txt_in.forward(&text)?, &target], 1)?
    };
    let (rope_cos, rope_sin) = t2i_rope(
        transformer,
        text_len,
        latent_height,
        latent_width,
        latents.dtype(),
        latents.device(),
    )?;
    let (rope_cos, rope_sin) = if cached {
        (
            rope_cos.narrow(0, text_len, target_tokens)?.contiguous()?,
            rope_sin.narrow(0, text_len, target_tokens)?.contiguous()?,
        )
    } else {
        (rope_cos, rope_sin)
    };
    let mut timesteps = vec![timestep; batch];
    timesteps.push(0.0);
    let temb =
        transformer
            .time_text_embed
            .forward(&timesteps, latents.dtype(), latents.device())?;
    let modulation = transformer
        .modulation
        .forward(&candle_nn::Activation::Silu.forward(&temb)?)?;
    let inner = transformer.cfg.inner_dim();
    let real_row = modulation.narrow(0, 0, batch)?.unsqueeze(1)?;
    let real = real_row.broadcast_as((batch, target_tokens, 4 * inner))?;
    let zero = modulation
        .narrow(0, batch, 1)?
        .unsqueeze(1)?
        .broadcast_as((batch, text_len, 4 * inner))?;
    let per_token_modulation = if cached {
        if latents.device().is_metal() && transformer.compact_modulation {
            real_row
        } else {
            real
        }
    } else {
        Tensor::cat(&[&zero, &real], 1)?
    };
    for (index, block) in transformer.blocks.iter().enumerate() {
        let layer_cache = match &mut cache {
            LegacyCache::Extract(layers) => LegacyLayerCache::Extract(layers),
            LegacyCache::Reuse(layers) => LegacyLayerCache::Reuse(&layers[index]),
            LegacyCache::Disabled => LegacyLayerCache::Disabled,
        };
        hidden_states = block_forward_t2i(
            block,
            &hidden_states,
            &per_token_modulation,
            &rope_cos,
            &rope_sin,
            &conditioning.valid_tokens,
            layer_cache,
        )?;
    }
    let target_hidden =
        hidden_states.narrow(1, if cached { 0 } else { text_len }, target_tokens)?;
    let target_temb = temb.narrow(0, 0, batch)?;
    transformer
        .proj_out
        .forward(&transformer.norm_out.forward(&target_hidden, &target_temb)?)
}
