// LTX Video Transformer — ported from FerrisMind/candle-video (Apache 2.0)
// Original: https://github.com/FerrisMind/candle-video
// Adapted as a Mold-owned model with the following changes:
// - Removed t2v_pipeline trait dependencies (standalone model)
// - Removed flash-attn feature gates
// - Simplified for integration with mold inference engine

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Output wrapper
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Transformer2DModelOutput {
    pub sample: Tensor,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LtxVideoTransformer3DModelConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,
    pub num_layers: usize,
    pub qk_norm: String,
    pub norm_elementwise_affine: bool,
    pub norm_eps: f64,
    pub caption_channels: usize,
    pub attention_bias: bool,
    pub attention_out_bias: bool,
}

impl Default for LtxVideoTransformer3DModelConfig {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 32,
            attention_head_dim: 64,
            cross_attention_dim: 2048,
            num_layers: 28,
            qk_norm: "rms_norm_across_heads".to_string(),
            norm_elementwise_affine: false,
            norm_eps: 1e-6,
            caption_channels: 4096,
            attention_bias: true,
            attention_out_bias: true,
        }
    }
}

impl LtxVideoTransformer3DModelConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

// ---------------------------------------------------------------------------
// LayerNorm without affine parameters (elementwise_affine=False)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let last_dim = xs.dim(D::Minus1)?;
        let mean = (xs.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let xc = xs.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let denom = (var + self.eps)?.sqrt()?;
        xc.broadcast_div(&denom)
    }
}

// ---------------------------------------------------------------------------
// RMSNorm with optional affine weight
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, elementwise_affine: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if elementwise_affine {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let dim = xs_f32.dim(D::Minus1)? as f64;
        let ms = xs_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let ys_f32 = xs_f32.broadcast_div(&denom)?;
        let mut ys = ys_f32.to_dtype(dtype)?;
        if let Some(w) = &self.weight {
            let rank = ys.rank();
            let mut shape = vec![1usize; rank];
            shape[rank - 1] = w.dims1()?;
            let w = w.reshape(shape)?;
            ys = ys.broadcast_mul(&w)?;
        }
        Ok(ys)
    }
}

// ---------------------------------------------------------------------------
// GELU (approximate) — F32 upcast for numerical stability
// ---------------------------------------------------------------------------

pub fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_cube = x_f32.sqr()?.broadcast_mul(&x_f32)?;
    let inner = x_f32.broadcast_add(&x_cube.affine(0.044715, 0.0)?)?;
    let scale = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let tanh_input = inner.affine(scale as f64, 0.0)?;
    let tanh_out = tanh_input.tanh()?;
    let gelu = x_f32
        .broadcast_mul(&tanh_out.affine(1.0, 1.0)?)?
        .affine(0.5, 0.0)?;
    gelu.to_dtype(x.dtype())
}

// ---------------------------------------------------------------------------
// GeluProjection (Linear + GELU approximate)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GeluProjection {
    proj: nn::Linear,
}

impl GeluProjection {
    fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(xs)?;
        gelu_approximate(&x)
    }
}

impl Module for GeluProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// FeedForward (GELU projection + linear)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FeedForward {
    net_0: GeluProjection,
    net_2: nn::Linear,
}

impl FeedForward {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * 4;
        let net_0 = GeluProjection::new(dim, hidden, vb.pp("net.0"))?;
        let net_2 = nn::linear(hidden, dim, vb.pp("net.2"))?;
        Ok(Self { net_0, net_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.net_0.forward(xs)?;
        self.net_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// PixArtAlphaTextProjection
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PixArtAlphaTextProjection {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl PixArtAlphaTextProjection {
    pub fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_features, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = gelu_approximate(&x)?;
        self.linear_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// TimestepEmbedding (two linear layers + SiLU)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = x.silu()?;
        self.linear_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// PixArtAlphaCombinedTimestepSizeEmbeddings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    timestep_embedder: TimestepEmbedding,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder =
            TimestepEmbedding::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self { timestep_embedder })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let timesteps_proj = get_timestep_embedding(timestep, 256, true)?;
        self.timestep_embedder.forward(&timesteps_proj)
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNormSingle
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: nn::Linear,
}

impl AdaLayerNormSingle {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(dim, vb.pp("emb"))?;
        let linear = nn::linear(dim, 6 * dim, vb.pp("linear"))?;
        Ok(Self { emb, linear })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded_timestep = self.emb.forward(timestep)?;
        let x = embedded_timestep.silu()?;
        let x = self.linear.forward(&x)?;
        Ok((x, embedded_timestep))
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal timestep embedding (DDPM-style)
// ---------------------------------------------------------------------------

fn get_timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
) -> Result<Tensor> {
    let device = timesteps.device();
    let original_dtype = timesteps.dtype();
    let dtype = DType::F32;

    let n = timesteps.dim(0)?;
    let half = embedding_dim / 2;

    let t = timesteps.to_dtype(dtype)?;
    let t = t.unsqueeze(1)?;

    let inv_freq: Vec<_> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / (half as f32)))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
    let freqs = t.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let sin = freqs.sin()?;
    let cos = freqs.cos()?;

    let emb = if flip_sin_to_cos {
        Tensor::cat(&[cos, sin], D::Minus1)?
    } else {
        Tensor::cat(&[sin, cos], D::Minus1)?
    };

    if embedding_dim % 2 == 1 {
        let pad = Tensor::zeros((n, 1), dtype, device)?;
        Tensor::cat(&[emb, pad], D::Minus1)?.to_dtype(original_dtype)
    } else {
        emb.to_dtype(original_dtype)
    }
}

// ---------------------------------------------------------------------------
// Rotary position embedding helpers — F32 upcast for stability
// ---------------------------------------------------------------------------

pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let (b, s, c) = x_f32.dims3()?;
    if c % 2 != 0 {
        candle::bail!("apply_rotary_emb expects last dim even, got {c}");
    }
    let half = c / 2;

    let x2 = x_f32.reshape((b, s, half, 2))?;
    let x_real = x2.i((.., .., .., 0))?;
    let x_imag = x2.i((.., .., .., 1))?;

    let x_rot = Tensor::stack(&[x_imag.neg()?, x_real.clone()], D::Minus1)?.reshape((b, s, c))?;

    let out = x_f32
        .broadcast_mul(&cos)?
        .broadcast_add(&x_rot.broadcast_mul(&sin)?)?;
    out.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// LtxVideoRotaryPosEmbed — 3D spatio-temporal RoPE
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LtxVideoRotaryPosEmbed {
    dim: usize,
    base_num_frames: usize,
    base_height: usize,
    base_width: usize,
    patch_size: usize,
    patch_size_t: usize,
    theta: f64,
}

impl LtxVideoRotaryPosEmbed {
    pub fn new(
        dim: usize,
        base_num_frames: usize,
        base_height: usize,
        base_width: usize,
        patch_size: usize,
        patch_size_t: usize,
        theta: f64,
    ) -> Self {
        Self {
            dim,
            base_num_frames,
            base_height,
            base_width,
            patch_size,
            patch_size_t,
            theta,
        }
    }

    fn prepare_video_coords(
        &self,
        batch_size: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        device: &Device,
    ) -> Result<Tensor> {
        let dtype = DType::F32;

        let grid_h = Tensor::arange(0u32, height as u32, device)?.to_dtype(dtype)?;
        let grid_w = Tensor::arange(0u32, width as u32, device)?.to_dtype(dtype)?;
        let grid_f = Tensor::arange(0u32, num_frames as u32, device)?.to_dtype(dtype)?;

        let f = grid_f
            .reshape((num_frames, 1, 1))?
            .broadcast_as((num_frames, height, width))?;
        let h = grid_h
            .reshape((1, height, 1))?
            .broadcast_as((num_frames, height, width))?;
        let w = grid_w
            .reshape((1, 1, width))?
            .broadcast_as((num_frames, height, width))?;

        let mut grid = Tensor::stack(&[f, h, w], 0)?;
        grid = grid
            .unsqueeze(0)?
            .broadcast_as((batch_size, 3, num_frames, height, width))?;

        if let Some((sf, sh, sw)) = rope_interpolation_scale {
            let f_scale = (sf * self.patch_size_t as f64 / self.base_num_frames as f64) as f32;
            let h_scale = (sh * self.patch_size as f64 / self.base_height as f64) as f32;
            let w_scale = (sw * self.patch_size as f64 / self.base_width as f64) as f32;

            let gf = grid
                .i((.., 0..1, .., .., ..))?
                .affine(f_scale as f64, 0.0)?;
            let gh = grid
                .i((.., 1..2, .., .., ..))?
                .affine(h_scale as f64, 0.0)?;
            let gw = grid
                .i((.., 2..3, .., .., ..))?
                .affine(w_scale as f64, 0.0)?;
            grid = Tensor::cat(&[gf, gh, gw], 1)?;
        }

        let seq = num_frames * height * width;
        let grid = grid
            .reshape((batch_size, 3, seq))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(grid)
    }

    /// Returns (cos, sin), both shaped [B, seq, dim].
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let device = hidden_states.device();
        let batch_size = hidden_states.dim(0)?;

        let grid = if let Some(coords) = video_coords {
            // Normalize external video coordinates by base sizes (matching Python reference).
            let base_f = self.base_num_frames as f64;
            let base_h = self.base_height as f64;
            let base_w = self.base_width as f64;
            let cf = coords
                .i((.., .., 0))?
                .to_dtype(DType::F32)?
                .affine(1.0 / base_f, 0.0)?;
            let ch = coords
                .i((.., .., 1))?
                .to_dtype(DType::F32)?
                .affine(1.0 / base_h, 0.0)?;
            let cw = coords
                .i((.., .., 2))?
                .to_dtype(DType::F32)?
                .affine(1.0 / base_w, 0.0)?;
            Tensor::stack(&[cf, ch, cw], candle::D::Minus1)?
        } else {
            // Fall back to internally-computed coordinates (normalized by base sizes).
            self.prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale,
                device,
            )?
        };

        let steps = self.dim / 6;
        let dtype = DType::F32;

        let lin = if steps <= 1 {
            Tensor::zeros((1,), dtype, device)?
        } else {
            let idx = Tensor::arange(0u32, steps as u32, device)?.to_dtype(dtype)?;
            idx.affine(1.0 / ((steps - 1) as f64), 0.0)?
        };

        let theta_ln = (self.theta.ln()) as f32;
        let freqs = (lin.affine(theta_ln as f64, 0.0)?).exp()?;
        let freqs = freqs.affine(std::f64::consts::PI / 2.0, 0.0)?;

        let grid = grid.to_dtype(dtype)?;
        let grid_scaled = grid.unsqueeze(D::Minus1)?.affine(2.0, -1.0)?;
        let freqs = grid_scaled.broadcast_mul(&freqs.reshape((1, 1, 1, steps))?)?;
        let freqs = freqs
            .transpose(D::Minus1, D::Minus2)?
            .contiguous()?
            .flatten_from(2)?;

        fn repeat_interleave_2(t: &Tensor) -> Result<Tensor> {
            let t_unsq = t.unsqueeze(D::Minus1)?;
            let t_rep = Tensor::cat(&[t_unsq.clone(), t_unsq], D::Minus1)?;
            let shape = t.dims();
            let new_last = shape[shape.len() - 1] * 2;
            let mut new_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
            new_shape.push(new_last);
            t_rep.reshape(new_shape)
        }

        let mut cos = repeat_interleave_2(&freqs.cos()?)?;
        let mut sin = repeat_interleave_2(&freqs.sin()?)?;

        let rem = self.dim % 6;
        if rem != 0 {
            let (b, seq, _) = cos.dims3()?;
            let cos_pad = Tensor::ones((b, seq, rem), dtype, device)?;
            let sin_pad = Tensor::zeros((b, seq, rem), dtype, device)?;
            cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
            sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
        }

        Ok((cos, sin))
    }
}

// ---------------------------------------------------------------------------
// LtxAttention — multi-head attention with RoPE + QK RMSNorm
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct LtxAttention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    inner_kv_dim: usize,
    cross_attention_dim: usize,

    norm_q: RmsNorm,
    norm_k: RmsNorm,

    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,

    to_out: nn::Linear,
    dropout: nn::Dropout,
}

impl LtxAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_dim: usize,
        heads: usize,
        kv_heads: usize,
        dim_head: usize,
        dropout: f64,
        bias: bool,
        cross_attention_dim: Option<usize>,
        out_bias: bool,
        qk_norm: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        if qk_norm != "rms_norm_across_heads" {
            candle::bail!("Only 'rms_norm_across_heads' is supported as qk_norm.");
        }

        let inner_dim = dim_head * heads;
        let inner_kv_dim = dim_head * kv_heads;
        let cross_attention_dim = cross_attention_dim.unwrap_or(query_dim);

        let norm_q = RmsNorm::new(inner_dim, 1e-5, true, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(inner_kv_dim, 1e-5, true, vb.pp("norm_k"))?;

        let to_q = nn::linear_b(query_dim, inner_dim, bias, vb.pp("to_q"))?;
        let to_k = nn::linear_b(cross_attention_dim, inner_kv_dim, bias, vb.pp("to_k"))?;
        let to_v = nn::linear_b(cross_attention_dim, inner_kv_dim, bias, vb.pp("to_v"))?;

        let to_out = nn::linear_b(inner_dim, query_dim, out_bias, vb.pp("to_out").pp("0"))?;
        let dropout = nn::Dropout::new(dropout as f32);

        Ok(Self {
            heads,
            head_dim: dim_head,
            inner_dim,
            inner_kv_dim,
            cross_attention_dim,
            norm_q,
            norm_k,
            to_q,
            to_k,
            to_v,
            to_out,
            dropout,
        })
    }

    fn prepare_attention_mask(
        &self,
        attention_mask: &Tensor,
        q_len: usize,
        k_len: usize,
    ) -> Result<Tensor> {
        match attention_mask.rank() {
            2 => {
                let (b, kk) = attention_mask.dims2()?;
                if kk != k_len {
                    candle::bail!(
                        "Expected attention_mask [B,k_len]=[{},{}], got [{},{}]",
                        b,
                        k_len,
                        b,
                        kk
                    );
                }
                let mask_f = attention_mask.to_dtype(DType::F32)?;
                let mask = ((mask_f.affine(-1.0, 1.0))?.affine(-10000.0, 0.0))?;
                let m = mask.unsqueeze(1)?.unsqueeze(1)?;
                m.broadcast_as((b, self.heads, q_len, k_len))?.contiguous()
            }
            3 => {
                let (b, one, kk) = attention_mask.dims3()?;
                if one != 1 || kk != k_len {
                    candle::bail!(
                        "Expected attention_mask [B,1,k_len]=[{},1,{}], got [{},{},{}]",
                        b,
                        k_len,
                        b,
                        one,
                        kk
                    );
                }
                let m = attention_mask.unsqueeze(2)?;
                m.broadcast_as((b, self.heads, q_len, k_len))?.contiguous()
            }
            4 => Ok(attention_mask.clone()),
            other => candle::bail!("Unsupported attention_mask rank {other}"),
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let enc = encoder_hidden_states.unwrap_or(hidden_states);
        let (_, k_len, _) = enc.dims3()?;

        let attn_mask = if let Some(mask) = attention_mask {
            Some(self.prepare_attention_mask(mask, q_len, k_len)?)
        } else {
            None
        };

        let mut q = self.to_q.forward(hidden_states)?;
        let mut k = self.to_k.forward(enc)?;
        let v = self.to_v.forward(enc)?;

        q = self.norm_q.forward(&q)?;
        k = self.norm_k.forward(&k)?;

        if let Some((cos, sin)) = image_rotary_emb {
            q = apply_rotary_emb(&q, cos, sin)?;
            k = apply_rotary_emb(&k, cos, sin)?;
        }

        let q = q.reshape((b, q_len, self.heads, self.head_dim))?;
        let k = k.reshape((b, k_len, self.heads, self.head_dim))?;
        let v = v.reshape((b, k_len, self.heads, self.head_dim))?;

        let dtype = q.dtype();
        let scale = 1f32 / (self.head_dim as f32).sqrt();

        // Manual attention path — F32 upcast for softmax stability
        let q_f32 = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

        let att = q_f32.matmul(&k_f32.transpose(D::Minus1, D::Minus2)?)?;
        let att = (att * (scale as f64))?;

        let att = match &attn_mask {
            Some(mask) => att.broadcast_add(&mask.to_dtype(DType::F32)?)?,
            None => att,
        };

        let (b_sz, h_sz, q_l, k_l) = att.dims4()?;
        let att = att.reshape((b_sz * h_sz * q_l, k_l))?;
        let att = nn::ops::softmax(&att, D::Minus1)?;
        let att = att.reshape((b_sz, h_sz, q_l, k_l))?;

        let out_f32 = att.matmul(&v_f32)?;
        let out = out_f32.to_dtype(dtype)?;

        let out = out.transpose(1, 2)?.contiguous()?;
        let out = out.reshape((b, q_len, self.inner_dim))?;

        let out = self.to_out.forward(&out)?;
        self.dropout.forward(&out, false)
    }
}

// ---------------------------------------------------------------------------
// LtxVideoTransformerBlock
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LtxVideoTransformerBlock {
    norm1: RmsNorm,
    attn1: LtxAttention,
    norm2: RmsNorm,
    attn2: LtxAttention,
    ff: FeedForward,
    scale_shift_table: Tensor,
}

impl LtxVideoTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_attention_heads: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        qk_norm: &str,
        attention_bias: bool,
        attention_out_bias: bool,
        eps: f64,
        elementwise_affine: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm1"))?;
        let attn1 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            None,
            attention_out_bias,
            qk_norm,
            vb.pp("attn1"),
        )?;
        let norm2 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm2"))?;
        let attn2 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            Some(cross_attention_dim),
            attention_out_bias,
            qk_norm,
            vb.pp("attn2"),
        )?;

        let ff = FeedForward::new(dim, vb.pp("ff"))?;
        let scale_shift_table = vb.get((6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            ff,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = hidden_states.dim(0)?;
        let norm_hidden = self.norm1.forward(hidden_states)?;

        let (b_temb, temb_last) = temb.dims2()?;
        if b_temb != b {
            candle::bail!(
                "temb batch size {} mismatch hidden_states batch size {}",
                b_temb,
                b
            );
        }

        if temb_last % 6 != 0 {
            candle::bail!("temb last dim must be divisible by 6, got {temb_last}");
        }
        let dim = temb_last / 6;
        let t = 1;
        let temb_reshaped = temb.reshape((b, t, 6, dim))?;

        let table = self
            .scale_shift_table
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, t, 6, dim))?;
        let ada = table.broadcast_add(&temb_reshaped)?;

        let shift_msa = ada.i((.., .., 0, ..))?;
        let scale_msa = ada.i((.., .., 1, ..))?;
        let gate_msa = ada.i((.., .., 2, ..))?;
        let shift_mlp = ada.i((.., .., 3, ..))?;
        let scale_mlp = ada.i((.., .., 4, ..))?;
        let gate_mlp = ada.i((.., .., 5, ..))?;

        let norm_hidden = {
            let one = Tensor::ones_like(&scale_msa)?;
            let s = one.broadcast_add(&scale_msa)?;
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hidden_states.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_msa.dim(1)? == 1 {
                shift_msa.broadcast_as((b, hidden_states.dim(1)?, shift_msa.dim(2)?))?
            } else {
                shift_msa
            };
            norm_hidden.broadcast_mul(&s)?.broadcast_add(&sh)?
        };

        let attn1 = self
            .attn1
            .forward(&norm_hidden, None, None, image_rotary_emb)?;
        let gate_msa = if gate_msa.dim(1)? == 1 {
            gate_msa.broadcast_as((b, hidden_states.dim(1)?, gate_msa.dim(2)?))?
        } else {
            gate_msa
        };
        let mut hs = hidden_states.broadcast_add(&attn1.broadcast_mul(&gate_msa)?)?;

        let attn2 = self.attn2.forward(
            &hs,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            None,
        )?;
        hs = hs.broadcast_add(&attn2)?;

        let norm2 = self.norm2.forward(&hs)?;
        let norm2 = {
            let one = Tensor::ones_like(&scale_mlp)?;
            let s = one.broadcast_add(&scale_mlp)?;
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hs.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_mlp.dim(1)? == 1 {
                shift_mlp.broadcast_as((b, hs.dim(1)?, shift_mlp.dim(2)?))?
            } else {
                shift_mlp
            };
            norm2.broadcast_mul(&s)?.broadcast_add(&sh)?
        };
        let ff = self.ff.forward(&norm2)?;
        let gate_mlp = if gate_mlp.dim(1)? == 1 {
            gate_mlp.broadcast_as((b, hs.dim(1)?, gate_mlp.dim(2)?))?
        } else {
            gate_mlp
        };
        hs = hs.broadcast_add(&ff.broadcast_mul(&gate_mlp)?)?;

        Ok(hs)
    }
}

// ---------------------------------------------------------------------------
// LtxVideoTransformer3DModel — top-level model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LtxVideoTransformer3DModel {
    proj_in: nn::Linear,
    scale_shift_table: Tensor,
    time_embed: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    rope: LtxVideoRotaryPosEmbed,
    transformer_blocks: Vec<LtxVideoTransformerBlock>,
    norm_out: LayerNormNoParams,
    proj_out: nn::Linear,
    config: LtxVideoTransformer3DModelConfig,
    skip_block_list: Vec<usize>,
    /// Timestep scaling factor (1000.0 for LTX Video, matching Python's
    /// `timestep_scale_multiplier`). Applied to the input timestep before
    /// computing the sinusoidal embedding.
    timestep_scale_multiplier: f64,
}

impl LtxVideoTransformer3DModel {
    pub fn new(config: &LtxVideoTransformer3DModelConfig, vb: VarBuilder) -> Result<Self> {
        let out_channels = if config.out_channels == 0 {
            config.in_channels
        } else {
            config.out_channels
        };
        let inner_dim = config.num_attention_heads * config.attention_head_dim;

        let proj_in = nn::linear(config.in_channels, inner_dim, vb.pp("proj_in"))?;
        let scale_shift_table = vb.get((2, inner_dim), "scale_shift_table")?;

        let time_embed = AdaLayerNormSingle::new(inner_dim, vb.pp("time_embed"))?;
        let caption_projection = PixArtAlphaTextProjection::new(
            config.caption_channels,
            inner_dim,
            vb.pp("caption_projection"),
        )?;

        let rope = LtxVideoRotaryPosEmbed::new(
            inner_dim,
            20,
            2048,
            2048,
            config.patch_size,
            config.patch_size_t,
            10000.0,
        );

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            transformer_blocks.push(LtxVideoTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.cross_attention_dim,
                &config.qk_norm,
                config.attention_bias,
                config.attention_out_bias,
                config.norm_eps,
                config.norm_elementwise_affine,
                vb.pp("transformer_blocks").pp(layer_idx.to_string()),
            )?);
        }

        let norm_out = LayerNormNoParams::new(1e-6);
        let proj_out = nn::linear(inner_dim, out_channels, vb.pp("proj_out"))?;

        Ok(Self {
            proj_in,
            scale_shift_table,
            time_embed,
            caption_projection,
            rope,
            transformer_blocks,
            norm_out,
            proj_out,
            config: config.clone(),
            skip_block_list: Vec::new(),
            timestep_scale_multiplier: 1000.0,
        })
    }

    pub fn config(&self) -> &LtxVideoTransformer3DModelConfig {
        &self.config
    }

    pub fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.skip_block_list = list;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
    ) -> Result<Tensor> {
        let model_dtype = self.proj_in.weight().dtype();
        let hidden_states = hidden_states.to_dtype(model_dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(model_dtype)?;

        let hidden_states = self.proj_in.forward(&hidden_states)?;

        // Scale timestep by multiplier (1000.0) before computing sinusoidal embedding.
        // The model receives raw sigma (0-1) and needs to map to embedding range (0-1000).
        let timestep = timestep
            .flatten_all()?
            .to_dtype(model_dtype)?
            .affine(self.timestep_scale_multiplier, 0.0)?;

        let (temb, embedded_timestep) = self.time_embed.forward(&timestep)?;

        let encoder_hidden_states = self.caption_projection.forward(&encoder_hidden_states)?;

        let encoder_attention_mask = if let Some(mask) = encoder_attention_mask {
            if mask.rank() == 2 {
                let mask_f = mask.to_dtype(hidden_states.dtype())?;
                let bias = (mask_f.affine(-1.0, 1.0)? * (-10000.0))?;
                Some(bias.unsqueeze(1)?)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };
        let encoder_attention_mask = encoder_attention_mask.as_ref();

        let (cos, sin) = self.rope.forward(
            &hidden_states,
            num_frames,
            height,
            width,
            rope_interpolation_scale,
            video_coords,
        )?;

        let mut hidden_states = hidden_states;
        let image_rotary_emb = Some((&cos, &sin));

        for (index, block) in self.transformer_blocks.iter().enumerate() {
            if self.skip_block_list.contains(&index) {
                continue;
            }

            hidden_states = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                image_rotary_emb,
                encoder_attention_mask,
            )?;
        }

        // Final modulation
        let b = hidden_states.dim(0)?;
        let inner_dim = hidden_states.dim(2)?;

        let table = self.scale_shift_table.to_dtype(embedded_timestep.dtype())?;
        let table = table.unsqueeze(0)?.unsqueeze(0)?;
        let emb = embedded_timestep.unsqueeze(1)?.unsqueeze(2)?;
        let scale_shift = table.broadcast_add(&emb)?;

        let shift = scale_shift.i((.., .., 0, ..))?;
        let scale = scale_shift.i((.., .., 1, ..))?;

        let mut hidden_states = self.norm_out.forward(&hidden_states)?;

        let one = Tensor::ones_like(&scale)?;
        let ss = one.broadcast_add(&scale)?;

        let s_dim = hidden_states.dim(1)?;
        let ss = ss.broadcast_as((b, s_dim, inner_dim))?;
        let sh = shift.broadcast_as((b, s_dim, inner_dim))?;

        hidden_states = hidden_states.broadcast_mul(&ss)?.broadcast_add(&sh)?;

        self.proj_out.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn test_config_defaults() {
        let config = LtxVideoTransformer3DModelConfig::default();
        assert_eq!(config.in_channels, 128);
        assert_eq!(config.out_channels, 128);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.attention_head_dim, 64);
        assert_eq!(config.inner_dim(), 2048);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.qk_norm, "rms_norm_across_heads");
    }

    #[test]
    fn test_model_compiles_with_zeros() -> candle::Result<()> {
        let device = Device::Cpu;
        let config = LtxVideoTransformer3DModelConfig {
            num_layers: 2,
            attention_head_dim: 16,
            num_attention_heads: 2,
            cross_attention_dim: 32,
            caption_channels: 32,
            in_channels: 32,
            out_channels: 32,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LtxVideoTransformer3DModel::new(&config, vb.pp("transformer"))?;

        assert_eq!(model.config().num_layers, 2);
        assert_eq!(model.config().inner_dim(), 32);
        assert!(model.skip_block_list.is_empty());

        Ok(())
    }

    #[test]
    fn test_skip_block_list() -> candle::Result<()> {
        let device = Device::Cpu;
        let config = LtxVideoTransformer3DModelConfig {
            num_layers: 3,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model = LtxVideoTransformer3DModel::new(&config, vb.pp("transformer"))?;

        assert!(model.skip_block_list.is_empty());
        model.set_skip_block_list(vec![1]);
        assert_eq!(model.skip_block_list, vec![1]);

        Ok(())
    }
}
