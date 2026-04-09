// LTX-2 video transformer — adapted from candle-transformers-mold's LTX Video model.
// This keeps the proven video-only denoiser structure but patches the positional
// embedding path and config surface to match the native LTX-2 checkpoints.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use nn::{Module, VarBuilder};

use super::rope::LtxRopeType;

// ---------------------------------------------------------------------------
// Output wrapper
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Ltx2VideoTransformer3DModelConfig {
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
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: Vec<usize>,
    pub use_middle_indices_grid: bool,
    pub rope_type: LtxRopeType,
    pub double_precision_rope: bool,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_positional_embedding_max_pos: Vec<usize>,
    pub av_ca_timestep_scale_multiplier: f64,
    pub cross_attention_adaln: bool,
}

impl Default for Ltx2VideoTransformer3DModelConfig {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 32,
            attention_head_dim: 128,
            cross_attention_dim: 4096,
            num_layers: 48,
            qk_norm: "rms_norm".to_string(),
            norm_elementwise_affine: false,
            norm_eps: 1e-6,
            caption_channels: 3840,
            attention_bias: true,
            attention_out_bias: true,
            positional_embedding_theta: 10_000.0,
            positional_embedding_max_pos: vec![20, 2048, 2048],
            use_middle_indices_grid: true,
            rope_type: LtxRopeType::Split,
            double_precision_rope: false,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_cross_attention_dim: 2048,
            audio_positional_embedding_max_pos: vec![20],
            av_ca_timestep_scale_multiplier: 1.0,
            cross_attention_adaln: false,
        }
    }
}

impl Ltx2VideoTransformer3DModelConfig {
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
// FP8-aware linear
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum LtxLinear {
    Standard(nn::Linear),
    Fp8 {
        weight: Tensor,
        scale: Option<Tensor>,
        input_scale: Option<Tensor>,
        bias: Option<Tensor>,
    },
}

impl LtxLinear {
    fn load(in_dim: usize, out_dim: usize, has_bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let scale = vb.get_unchecked("weight_scale").ok();
        let input_scale = vb.get_unchecked("input_scale").ok();
        let bias = if has_bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        if weight.dtype() == DType::F8E4M3 || scale.is_some() || input_scale.is_some() {
            Ok(Self::Fp8 {
                weight,
                scale,
                input_scale,
                bias,
            })
        } else {
            Ok(Self::Standard(nn::Linear::new(weight, bias)))
        }
    }

    fn weight_dtype(&self) -> DType {
        match self {
            Self::Standard(linear) => linear.weight().dtype(),
            Self::Fp8 { weight, .. } => weight.dtype(),
        }
    }
}

impl Module for LtxLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(linear) => linear.forward(xs),
            Self::Fp8 {
                weight,
                scale,
                input_scale: _input_scale,
                bias,
            } => {
                let dtype = match xs.dtype() {
                    DType::F8E4M3 => DType::BF16,
                    other => other,
                };
                let xs = if xs.dtype() == dtype {
                    xs.clone()
                } else {
                    xs.to_dtype(dtype)?
                };
                let mut weight = weight.to_dtype(dtype)?;
                if let Some(scale) = scale {
                    weight = weight.broadcast_mul(&scale.to_dtype(dtype)?)?;
                }
                let weight_t = weight.t()?;
                let out = match *xs.dims() {
                    [batch0, batch1, tokens, hidden] => xs
                        .reshape((batch0 * batch1 * tokens, hidden))?
                        .matmul(&weight_t)?
                        .reshape((batch0, batch1, tokens, ()))?,
                    [batch, tokens, hidden] => xs
                        .reshape((batch * tokens, hidden))?
                        .matmul(&weight_t)?
                        .reshape((batch, tokens, ()))?,
                    _ => xs.matmul(&weight_t)?,
                };
                match bias {
                    Some(bias) => out.broadcast_add(&bias.to_dtype(dtype)?),
                    None => Ok(out),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GeluProjection (Linear + GELU approximate)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GeluProjection {
    proj: LtxLinear,
}

impl GeluProjection {
    fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let proj = LtxLinear::load(dim_in, dim_out, true, vb.pp("proj"))?;
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
    net_2: LtxLinear,
}

impl FeedForward {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * 4;
        let net_0 = GeluProjection::new(dim, hidden, vb.pp("net.0"))?;
        let net_2 = LtxLinear::load(hidden, dim, true, vb.pp("net.2"))?;
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
        Self::new_with_out_features(in_features, hidden_size, hidden_size, vb)
    }

    pub fn new_with_out_features(
        in_features: usize,
        hidden_size: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_1 = nn::linear(in_features, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(hidden_size, out_features, vb.pp("linear_2"))?;
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
        Self::new_with_coefficient(dim, 6, vb)
    }

    pub fn new_with_coefficient(
        dim: usize,
        embedding_coefficient: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(dim, vb.pp("emb"))?;
        let linear = nn::linear(dim, embedding_coefficient * dim, vb.pp("linear"))?;
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

fn apply_interleaved_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let (b, s, c) = x_f32.dims3()?;
    if c % 2 != 0 {
        candle_core::bail!("apply_rotary_emb expects last dim even, got {c}");
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

fn apply_split_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let (batch, seq, inner_dim) = x.dims3()?;
    if inner_dim != heads * head_dim {
        candle_core::bail!(
            "split rotary input dimension mismatch: expected {}, got {}",
            heads * head_dim,
            inner_dim
        );
    }
    if head_dim % 2 != 0 {
        candle_core::bail!("split rotary requires an even head_dim, got {head_dim}");
    }

    let x = x
        .reshape((batch, seq, heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let x = x.reshape((batch, heads, seq, 2, head_dim / 2))?;
    let first = x.i((.., .., .., 0..1, ..))?;
    let second = x.i((.., .., .., 1..2, ..))?;
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(3)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(3)?;

    let first_out = first
        .broadcast_mul(&cos)?
        .broadcast_sub(&second.broadcast_mul(&sin)?)?;
    let second_out = second
        .broadcast_mul(&cos)?
        .broadcast_add(&first.broadcast_mul(&sin)?)?;
    Tensor::cat(&[first_out, second_out], 3)?
        .reshape((batch, heads, seq, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, seq, inner_dim))?
        .to_dtype(dtype)
}

pub fn apply_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_type: LtxRopeType,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    match rope_type {
        LtxRopeType::Interleaved => apply_interleaved_rotary_emb(x, cos, sin),
        LtxRopeType::Split => apply_split_rotary_emb(x, cos, sin, heads, head_dim),
    }
}

// ---------------------------------------------------------------------------
// LTX-2 rotary embedding — supports both interleaved and split RoPE.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Ltx2VideoRotaryPosEmbed {
    dim: usize,
    theta: f64,
    max_pos: Vec<usize>,
    use_middle_indices_grid: bool,
    num_attention_heads: usize,
    rope_type: LtxRopeType,
    double_precision_rope: bool,
}

impl Ltx2VideoRotaryPosEmbed {
    pub fn new(
        dim: usize,
        theta: f64,
        max_pos: Vec<usize>,
        use_middle_indices_grid: bool,
        num_attention_heads: usize,
        rope_type: LtxRopeType,
        double_precision_rope: bool,
    ) -> Self {
        Self {
            dim,
            theta,
            max_pos,
            use_middle_indices_grid,
            num_attention_heads,
            rope_type,
            double_precision_rope,
        }
    }

    fn fractional_positions(&self, indices_grid: &Tensor) -> Result<Tensor> {
        let (batch, pos_dims, seq, _bounds) = indices_grid.dims4()?;
        if pos_dims != self.max_pos.len() {
            candle_core::bail!(
                "rotary position dims mismatch: expected {}, got {}",
                self.max_pos.len(),
                pos_dims
            );
        }
        let grid = if self.use_middle_indices_grid {
            let starts = indices_grid.narrow(3, 0, 1)?;
            let ends = indices_grid.narrow(3, 1, 1)?;
            starts.broadcast_add(&ends)?.affine(0.5, 0.0)?
        } else {
            indices_grid.narrow(3, 0, 1)?
        }
        .squeeze(3)?
        .to_dtype(DType::F32)?;

        let mut normalized = Vec::with_capacity(pos_dims);
        for (dim, max_pos) in self.max_pos.iter().enumerate() {
            normalized.push(grid.i((.., dim, ..))?.affine(1.0 / *max_pos as f64, 0.0)?);
        }
        Tensor::stack(&normalized, 2)?
            .reshape((batch, seq, pos_dims))
            .map_err(Into::into)
    }

    fn base_indices(&self, device: &Device, position_dims: usize) -> Result<Tensor> {
        let steps = self.dim / (2 * position_dims);
        if steps == 0 {
            candle_core::bail!(
                "rotary dimension {} is too small for {} positional dims",
                self.dim,
                position_dims
            );
        }
        if steps == 1 {
            Tensor::zeros((1,), DType::F32, device)
        } else {
            let denom = (steps - 1) as f64;
            let values: Vec<f32> = (0..steps)
                .map(|index| {
                    let ratio = index as f64 / denom;
                    let power = if self.double_precision_rope {
                        self.theta.powf(ratio)
                    } else {
                        (self.theta as f32).powf(ratio as f32) as f64
                    };
                    (power * std::f64::consts::PI / 2.0) as f32
                })
                .collect();
            Tensor::from_vec(values, (steps,), device)
        }
    }

    fn repeat_interleave_2(freqs: &Tensor) -> Result<Tensor> {
        let freq_unsq = freqs.unsqueeze(D::Minus1)?;
        let duplicated = Tensor::cat(&[freq_unsq.clone(), freq_unsq], D::Minus1)?;
        let shape = freqs.dims();
        let mut new_shape = shape[..shape.len() - 1].to_vec();
        new_shape.push(shape[shape.len() - 1] * 2);
        duplicated.reshape(new_shape)
    }

    pub fn forward(&self, hidden_states: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = hidden_states.device();
        let position_dims = positions.dim(1)?;
        let indices = self.base_indices(device, position_dims)?;
        let fractional = self.fractional_positions(positions)?;
        let scaled = fractional.unsqueeze(D::Minus1)?.affine(2.0, -1.0)?;
        let freqs = indices
            .reshape((1, 1, 1, indices.dim(0)?))?
            .broadcast_mul(&scaled)?
            .transpose(2, 3)?
            .contiguous()?
            .flatten_from(2)?;

        match self.rope_type {
            LtxRopeType::Interleaved => {
                let mut cos = Self::repeat_interleave_2(&freqs.cos()?)?;
                let mut sin = Self::repeat_interleave_2(&freqs.sin()?)?;
                let rem = self.dim % (2 * position_dims);
                if rem != 0 {
                    let (batch, seq, _) = cos.dims3()?;
                    let cos_pad = Tensor::ones((batch, seq, rem), DType::F32, device)?;
                    let sin_pad = Tensor::zeros((batch, seq, rem), DType::F32, device)?;
                    cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
                    sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
                }
                Ok((
                    cos.to_dtype(hidden_states.dtype())?,
                    sin.to_dtype(hidden_states.dtype())?,
                ))
            }
            LtxRopeType::Split => {
                let expected = self.dim / 2;
                let current = freqs.dim(D::Minus1)?;
                let pad_size = expected.saturating_sub(current);
                let mut cos = freqs.cos()?;
                let mut sin = freqs.sin()?;
                if pad_size != 0 {
                    let (batch, seq, _) = cos.dims3()?;
                    let cos_pad = Tensor::ones((batch, seq, pad_size), DType::F32, device)?;
                    let sin_pad = Tensor::zeros((batch, seq, pad_size), DType::F32, device)?;
                    cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
                    sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
                }
                let (batch, seq, _) = cos.dims3()?;
                let cos = cos
                    .reshape((
                        batch,
                        seq,
                        self.num_attention_heads,
                        expected / self.num_attention_heads,
                    ))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let sin = sin
                    .reshape((
                        batch,
                        seq,
                        self.num_attention_heads,
                        expected / self.num_attention_heads,
                    ))?
                    .transpose(1, 2)?
                    .contiguous()?;
                Ok((
                    cos.to_dtype(hidden_states.dtype())?,
                    sin.to_dtype(hidden_states.dtype())?,
                ))
            }
        }
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
    rope_type: LtxRopeType,
    norm_eps: f64,

    norm_q: RmsNorm,
    norm_k: RmsNorm,

    to_q: LtxLinear,
    to_k: LtxLinear,
    to_v: LtxLinear,

    to_out: LtxLinear,
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
        rope_type: LtxRopeType,
        vb: VarBuilder,
    ) -> Result<Self> {
        if qk_norm != "rms_norm_across_heads" && qk_norm != "rms_norm" {
            candle_core::bail!(
                "Only 'rms_norm' and 'rms_norm_across_heads' are supported as qk_norm."
            );
        }

        let inner_dim = dim_head * heads;
        let inner_kv_dim = dim_head * kv_heads;
        let cross_attention_dim = cross_attention_dim.unwrap_or(query_dim);

        let norm_eps = 1e-6;
        let norm_q = RmsNorm::new(inner_dim, norm_eps, true, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(inner_kv_dim, norm_eps, true, vb.pp("norm_k"))?;

        let to_q = LtxLinear::load(query_dim, inner_dim, bias, vb.pp("to_q"))?;
        let to_k = LtxLinear::load(cross_attention_dim, inner_kv_dim, bias, vb.pp("to_k"))?;
        let to_v = LtxLinear::load(cross_attention_dim, inner_kv_dim, bias, vb.pp("to_v"))?;

        let to_out = LtxLinear::load(inner_dim, query_dim, out_bias, vb.pp("to_out").pp("0"))?;
        let dropout = nn::Dropout::new(dropout as f32);

        Ok(Self {
            heads,
            head_dim: dim_head,
            inner_dim,
            inner_kv_dim,
            cross_attention_dim,
            rope_type,
            norm_eps,
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
                    candle_core::bail!(
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
                    candle_core::bail!(
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
            other => candle_core::bail!("Unsupported attention_mask rank {other}"),
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        key_rotary_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let is_self_attention = encoder_hidden_states.is_none();
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
            if is_self_attention {
                q = apply_rotary_emb(&q, cos, sin, self.rope_type, self.heads, self.head_dim)?;
                k = apply_rotary_emb(&k, cos, sin, self.rope_type, self.heads, self.head_dim)?;
            } else if let Some((k_cos, k_sin)) = key_rotary_emb {
                q = apply_rotary_emb(&q, cos, sin, self.rope_type, self.heads, self.head_dim)?;
                k = apply_rotary_emb(&k, k_cos, k_sin, self.rope_type, self.heads, self.head_dim)?;
            }
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
    norm3: RmsNorm,
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
        rope_type: LtxRopeType,
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
            rope_type,
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
            rope_type,
            vb.pp("attn2"),
        )?;
        let norm3 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm3"))?;

        let ff = FeedForward::new(dim, vb.pp("ff"))?;
        let scale_shift_table = vb.get((6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
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
            candle_core::bail!(
                "temb batch size {} mismatch hidden_states batch size {}",
                b_temb,
                b
            );
        }

        if temb_last % 6 != 0 {
            candle_core::bail!("temb last dim must be divisible by 6, got {temb_last}");
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
            .forward(&norm_hidden, None, None, image_rotary_emb, None)?;
        let gate_msa = if gate_msa.dim(1)? == 1 {
            gate_msa.broadcast_as((b, hidden_states.dim(1)?, gate_msa.dim(2)?))?
        } else {
            gate_msa
        };
        let mut hs = hidden_states.broadcast_add(&attn1.broadcast_mul(&gate_msa)?)?;

        let norm2 = self.norm2.forward(&hs)?;

        let attn2 = self.attn2.forward(
            &norm2,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            None,
            None,
        )?;
        hs = hs.broadcast_add(&attn2)?;

        let norm3 = self.norm3.forward(&hs)?;
        let norm3 = {
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
            norm3.broadcast_mul(&s)?.broadcast_add(&sh)?
        };
        let ff = self.ff.forward(&norm3)?;
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
// LTX-2 video transformer — top-level model
// ---------------------------------------------------------------------------

enum TransformerBlockSource {
    Eager(Vec<LtxVideoTransformerBlock>),
    Streaming(VarBuilder<'static>),
}

pub struct Ltx2VideoTransformer3DModel {
    proj_in: nn::Linear,
    scale_shift_table: Tensor,
    time_embed: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    rope: Ltx2VideoRotaryPosEmbed,
    transformer_blocks: TransformerBlockSource,
    norm_out: LayerNormNoParams,
    proj_out: nn::Linear,
    config: Ltx2VideoTransformer3DModelConfig,
    skip_block_list: Vec<usize>,
    /// Timestep scaling factor (1000.0 for LTX Video, matching Python's
    /// `timestep_scale_multiplier`). Applied to the input timestep before
    /// computing the sinusoidal embedding.
    timestep_scale_multiplier: f64,
}

impl Ltx2VideoTransformer3DModel {
    pub fn new(config: &Ltx2VideoTransformer3DModelConfig, vb: VarBuilder) -> Result<Self> {
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

        let rope = Ltx2VideoRotaryPosEmbed::new(
            inner_dim,
            config.positional_embedding_theta,
            config.positional_embedding_max_pos.clone(),
            config.use_middle_indices_grid,
            config.num_attention_heads,
            config.rope_type,
            config.double_precision_rope,
        );

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            transformer_blocks.push(LtxVideoTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.cross_attention_dim,
                &config.qk_norm,
                config.rope_type,
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
            transformer_blocks: TransformerBlockSource::Eager(transformer_blocks),
            norm_out,
            proj_out,
            config: config.clone(),
            skip_block_list: Vec::new(),
            timestep_scale_multiplier: 1000.0,
        })
    }

    pub fn new_streaming(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder<'static>,
    ) -> Result<Self> {
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

        let rope = Ltx2VideoRotaryPosEmbed::new(
            inner_dim,
            config.positional_embedding_theta,
            config.positional_embedding_max_pos.clone(),
            config.use_middle_indices_grid,
            config.num_attention_heads,
            config.rope_type,
            config.double_precision_rope,
        );

        let norm_out = LayerNormNoParams::new(1e-6);
        let proj_out = nn::linear(inner_dim, out_channels, vb.pp("proj_out"))?;

        Ok(Self {
            proj_in,
            scale_shift_table,
            time_embed,
            caption_projection,
            rope,
            transformer_blocks: TransformerBlockSource::Streaming(vb.pp("transformer_blocks")),
            norm_out,
            proj_out,
            config: config.clone(),
            skip_block_list: Vec::new(),
            timestep_scale_multiplier: 1000.0,
        })
    }

    pub fn config(&self) -> &Ltx2VideoTransformer3DModelConfig {
        &self.config
    }

    pub fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.skip_block_list = list;
    }

    fn streaming_block(
        &self,
        blocks_vb: VarBuilder<'static>,
        index: usize,
    ) -> Result<LtxVideoTransformerBlock> {
        let inner_dim = self.config.num_attention_heads * self.config.attention_head_dim;
        LtxVideoTransformerBlock::new(
            inner_dim,
            self.config.num_attention_heads,
            self.config.attention_head_dim,
            self.config.cross_attention_dim,
            &self.config.qk_norm,
            self.config.rope_type,
            self.config.attention_bias,
            self.config.attention_out_bias,
            self.config.norm_eps,
            self.config.norm_elementwise_affine,
            blocks_vb.pp(index.to_string()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_block(
        &self,
        index: usize,
        block: &LtxVideoTransformerBlock,
        hidden_states: Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.skip_block_list.contains(&index) {
            return Ok(hidden_states);
        }

        let original_hidden_states = if skip_layer_mask.is_some() {
            Some(hidden_states.clone())
        } else {
            None
        };

        let mut hidden_states = block.forward(
            &hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            encoder_attention_mask,
        )?;

        if let (Some(mask), Some(orig)) = (skip_layer_mask, original_hidden_states) {
            let m = mask.narrow(0, index, 1)?.flatten_all()?;
            let batch = hidden_states.dim(0)?;
            let m = m.reshape((batch, 1, 1))?.to_dtype(hidden_states.dtype())?;
            let one_minus_m = m.affine(-1.0, 1.0)?;
            hidden_states = hidden_states
                .broadcast_mul(&one_minus_m)?
                .broadcast_add(&orig.broadcast_mul(&m)?)?;
        }

        Ok(hidden_states)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        _num_frames: usize,
        _height: usize,
        _width: usize,
        _rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let weight_dtype = self.proj_in.weight().dtype();
        let compute_dtype = match weight_dtype {
            DType::F8E4M3 => DType::BF16,
            _ => weight_dtype,
        };
        let hidden_states = hidden_states.to_dtype(compute_dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(compute_dtype)?;

        let hidden_states = self.proj_in.forward(&hidden_states)?;

        // Scale timestep by multiplier (1000.0) before computing sinusoidal embedding.
        // The model receives raw sigma (0-1) and needs to map to embedding range (0-1000).
        let timestep = timestep
            .flatten_all()?
            .to_dtype(compute_dtype)?
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

        let video_coords = video_coords.ok_or_else(|| {
            candle_core::Error::msg("LTX-2 video transformer requires explicit positional bounds")
        })?;
        let (cos, sin) = self.rope.forward(&hidden_states, video_coords)?;

        let mut hidden_states = hidden_states;
        let image_rotary_emb = Some((&cos, &sin));

        match &self.transformer_blocks {
            TransformerBlockSource::Eager(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    hidden_states = self.apply_block(
                        index,
                        block,
                        hidden_states,
                        &encoder_hidden_states,
                        &temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        skip_layer_mask,
                    )?;
                }
            }
            TransformerBlockSource::Streaming(blocks_vb) => {
                for index in 0..self.config.num_layers {
                    let block = self.streaming_block(blocks_vb.clone(), index)?;
                    hidden_states = self.apply_block(
                        index,
                        &block,
                        hidden_states,
                        &encoder_hidden_states,
                        &temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        skip_layer_mask,
                    )?;
                    if hidden_states.device().is_cuda() {
                        hidden_states.device().synchronize()?;
                    }
                    drop(block);
                }
            }
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

fn rms_norm_tensor(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs_f32 = xs.to_dtype(DType::F32)?;
    let dim = xs_f32.dim(D::Minus1)? as f64;
    let ms = xs_f32
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0 / dim, 0.0)?;
    let denom = ms.affine(1.0, eps)?.sqrt()?;
    xs_f32.broadcast_div(&denom)?.to_dtype(dtype)
}

fn broadcast_to_tokens(values: &Tensor, tokens: usize) -> Result<Tensor> {
    if values.dim(1)? == 1 {
        values.broadcast_as((values.dim(0)?, tokens, values.dim(2)?))
    } else {
        Ok(values.clone())
    }
}

fn modulate_tokens(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let scale = broadcast_to_tokens(scale, x.dim(1)?)?;
    let shift = broadcast_to_tokens(shift, x.dim(1)?)?;
    let one = Tensor::ones_like(&scale)?;
    x.broadcast_mul(&one.broadcast_add(&scale)?)?
        .broadcast_add(&shift)
}

fn gate_tokens(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&broadcast_to_tokens(gate, x.dim(1)?)?)
}

#[derive(Clone, Debug)]
struct LtxPreparedModality {
    x: Tensor,
    context: Tensor,
    context_mask: Option<Tensor>,
    timesteps: Tensor,
    embedded_timestep: Tensor,
    rope: (Tensor, Tensor),
    cross_rope: Option<(Tensor, Tensor)>,
    cross_scale_shift_timestep: Option<Tensor>,
    cross_gate_timestep: Option<Tensor>,
    prompt_timestep: Option<Tensor>,
}

#[derive(Clone, Debug)]
struct LtxAvTransformerBlock {
    video_attn1: LtxAttention,
    video_attn2: LtxAttention,
    video_ff: FeedForward,
    video_scale_shift_table: Tensor,
    audio_attn1: LtxAttention,
    audio_attn2: LtxAttention,
    audio_ff: FeedForward,
    audio_scale_shift_table: Tensor,
    audio_to_video_attn: LtxAttention,
    video_to_audio_attn: LtxAttention,
    scale_shift_table_a2v_ca_audio: Tensor,
    scale_shift_table_a2v_ca_video: Tensor,
    norm_eps: f64,
    cross_attention_adaln: bool,
    prompt_scale_shift_table: Option<Tensor>,
    audio_prompt_scale_shift_table: Option<Tensor>,
}

impl LtxAvTransformerBlock {
    fn new(config: &Ltx2VideoTransformer3DModelConfig, vb: VarBuilder) -> Result<Self> {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;

        let video_attn1 = LtxAttention::new(
            video_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            config.attention_head_dim,
            0.0,
            config.attention_bias,
            None,
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            vb.pp("attn1"),
        )?;
        let video_attn2 = LtxAttention::new(
            video_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            config.attention_head_dim,
            0.0,
            config.attention_bias,
            Some(config.cross_attention_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            vb.pp("attn2"),
        )?;
        let video_ff = FeedForward::new(video_dim, vb.pp("ff"))?;
        let video_scale_shift_table = vb.get(
            (if config.cross_attention_adaln { 9 } else { 6 }, video_dim),
            "scale_shift_table",
        )?;

        let audio_attn1 = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            None,
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            vb.pp("audio_attn1"),
        )?;
        let audio_attn2 = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(config.audio_cross_attention_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            vb.pp("audio_attn2"),
        )?;
        let audio_ff = FeedForward::new(audio_dim, vb.pp("audio_ff"))?;
        let audio_scale_shift_table = vb.get(
            (if config.cross_attention_adaln { 9 } else { 6 }, audio_dim),
            "audio_scale_shift_table",
        )?;

        let audio_to_video_attn = LtxAttention::new(
            video_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(audio_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            vb.pp("audio_to_video_attn"),
        )?;
        let video_to_audio_attn = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(video_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            vb.pp("video_to_audio_attn"),
        )?;
        let scale_shift_table_a2v_ca_audio =
            vb.get((5, audio_dim), "scale_shift_table_a2v_ca_audio")?;
        let scale_shift_table_a2v_ca_video =
            vb.get((5, video_dim), "scale_shift_table_a2v_ca_video")?;

        let prompt_scale_shift_table = if config.cross_attention_adaln {
            Some(vb.get((2, video_dim), "prompt_scale_shift_table")?)
        } else {
            None
        };
        let audio_prompt_scale_shift_table = if config.cross_attention_adaln {
            Some(vb.get((2, audio_dim), "audio_prompt_scale_shift_table")?)
        } else {
            None
        };

        Ok(Self {
            video_attn1,
            video_attn2,
            video_ff,
            video_scale_shift_table,
            audio_attn1,
            audio_attn2,
            audio_ff,
            audio_scale_shift_table,
            audio_to_video_attn,
            video_to_audio_attn,
            scale_shift_table_a2v_ca_audio,
            scale_shift_table_a2v_ca_video,
            norm_eps: config.norm_eps,
            cross_attention_adaln: config.cross_attention_adaln,
            prompt_scale_shift_table,
            audio_prompt_scale_shift_table,
        })
    }

    fn add_ada_values(
        scale_shift_table: &Tensor,
        timestep: &Tensor,
        count: usize,
    ) -> Result<Tensor> {
        let batch = timestep.dim(0)?;
        let tokens = timestep.dim(1)?;
        let dim = scale_shift_table.dim(1)?;
        let table = scale_shift_table
            .to_dtype(timestep.dtype())?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch, tokens, count, dim))?;
        table.broadcast_add(&timestep.reshape((batch, tokens, count, dim))?)
    }

    fn get_ada_triplet(
        &self,
        scale_shift_table: &Tensor,
        timestep: &Tensor,
        start_index: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let ada = Self::add_ada_values(scale_shift_table, timestep, scale_shift_table.dim(0)?)?;
        Ok((
            ada.i((.., .., start_index, ..))?,
            ada.i((.., .., start_index + 1, ..))?,
            ada.i((.., .., start_index + 2, ..))?,
        ))
    }

    fn get_cross_ada_values(
        &self,
        scale_shift_table: &Tensor,
        scale_shift_timestep: &Tensor,
        gate_timestep: &Tensor,
        start_index: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let scale_shift =
            Self::add_ada_values(&scale_shift_table.i((0..4, ..))?, scale_shift_timestep, 4)?;
        let gate = Self::add_ada_values(&scale_shift_table.i((4..5, ..))?, gate_timestep, 1)?;
        Ok((
            scale_shift.i((.., .., start_index, ..))?,
            scale_shift.i((.., .., start_index + 1, ..))?,
            gate.i((.., .., 0, ..))?,
        ))
    }

    fn apply_text_cross_attention(
        &self,
        x: &Tensor,
        context: &Tensor,
        attn: &LtxAttention,
        scale_shift_table: &Tensor,
        prompt_scale_shift_table: Option<&Tensor>,
        timestep: &Tensor,
        prompt_timestep: Option<&Tensor>,
        context_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.cross_attention_adaln {
            let prompt_scale_shift_table = prompt_scale_shift_table.ok_or_else(|| {
                candle_core::Error::msg("cross-attention AdaLN requires prompt scale-shift weights")
            })?;
            let prompt_timestep = prompt_timestep.ok_or_else(|| {
                candle_core::Error::msg("cross-attention AdaLN requires prompt timestep embeddings")
            })?;
            let (shift_q, scale_q, gate) = self.get_ada_triplet(scale_shift_table, timestep, 6)?;
            let attn_input =
                modulate_tokens(&rms_norm_tensor(x, self.norm_eps)?, &scale_q, &shift_q)?;
            let prompt = Self::add_ada_values(prompt_scale_shift_table, prompt_timestep, 2)?;
            let shift_kv = prompt.i((.., .., 0, ..))?;
            let scale_kv = prompt.i((.., .., 1, ..))?;
            let context = modulate_tokens(context, &scale_kv, &shift_kv)?;
            return gate_tokens(
                &attn.forward(&attn_input, Some(&context), context_mask, None, None)?,
                &gate,
            );
        }
        attn.forward(
            &rms_norm_tensor(x, self.norm_eps)?,
            Some(context),
            context_mask,
            None,
            None,
        )
    }

    fn forward(
        &self,
        video: &LtxPreparedModality,
        audio: &LtxPreparedModality,
    ) -> Result<(Tensor, Tensor)> {
        let (v_shift_msa, v_scale_msa, v_gate_msa) =
            self.get_ada_triplet(&self.video_scale_shift_table, &video.timesteps, 0)?;
        let mut vx = video.x.broadcast_add(&gate_tokens(
            &self.video_attn1.forward(
                &modulate_tokens(
                    &rms_norm_tensor(&video.x, self.norm_eps)?,
                    &v_scale_msa,
                    &v_shift_msa,
                )?,
                None,
                None,
                Some((&video.rope.0, &video.rope.1)),
                None,
            )?,
            &v_gate_msa,
        )?)?;
        vx = vx.broadcast_add(&self.apply_text_cross_attention(
            &vx,
            &video.context,
            &self.video_attn2,
            &self.video_scale_shift_table,
            self.prompt_scale_shift_table.as_ref(),
            &video.timesteps,
            video.prompt_timestep.as_ref(),
            video.context_mask.as_ref(),
        )?)?;

        let (a_shift_msa, a_scale_msa, a_gate_msa) =
            self.get_ada_triplet(&self.audio_scale_shift_table, &audio.timesteps, 0)?;
        let mut ax = audio.x.broadcast_add(&gate_tokens(
            &self.audio_attn1.forward(
                &modulate_tokens(
                    &rms_norm_tensor(&audio.x, self.norm_eps)?,
                    &a_scale_msa,
                    &a_shift_msa,
                )?,
                None,
                None,
                Some((&audio.rope.0, &audio.rope.1)),
                None,
            )?,
            &a_gate_msa,
        )?)?;
        ax = ax.broadcast_add(&self.apply_text_cross_attention(
            &ax,
            &audio.context,
            &self.audio_attn2,
            &self.audio_scale_shift_table,
            self.audio_prompt_scale_shift_table.as_ref(),
            &audio.timesteps,
            audio.prompt_timestep.as_ref(),
            audio.context_mask.as_ref(),
        )?)?;

        let vx_norm3 = rms_norm_tensor(&vx, self.norm_eps)?;
        let ax_norm3 = rms_norm_tensor(&ax, self.norm_eps)?;

        let video_cross_scale_shift_timestep =
            video.cross_scale_shift_timestep.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "video cross scale-shift timestep missing for AV transformer",
                )
            })?;
        let video_cross_gate_timestep = video.cross_gate_timestep.as_ref().ok_or_else(|| {
            candle_core::Error::msg("video cross gate timestep missing for AV transformer")
        })?;
        let audio_cross_scale_shift_timestep =
            audio.cross_scale_shift_timestep.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "audio cross scale-shift timestep missing for AV transformer",
                )
            })?;
        let audio_cross_gate_timestep = audio.cross_gate_timestep.as_ref().ok_or_else(|| {
            candle_core::Error::msg("audio cross gate timestep missing for AV transformer")
        })?;
        let video_cross_rope = video.cross_rope.as_ref().ok_or_else(|| {
            candle_core::Error::msg("video cross positional embeddings missing for AV transformer")
        })?;
        let audio_cross_rope = audio.cross_rope.as_ref().ok_or_else(|| {
            candle_core::Error::msg("audio cross positional embeddings missing for AV transformer")
        })?;

        let (v_ca_scale, v_ca_shift, v_gate) = self.get_cross_ada_values(
            &self.scale_shift_table_a2v_ca_video,
            video_cross_scale_shift_timestep,
            video_cross_gate_timestep,
            0,
        )?;
        let (a_ca_scale, a_ca_shift, _) = self.get_cross_ada_values(
            &self.scale_shift_table_a2v_ca_audio,
            audio_cross_scale_shift_timestep,
            audio_cross_gate_timestep,
            0,
        )?;
        let vx_scaled = modulate_tokens(&vx_norm3, &v_ca_scale, &v_ca_shift)?;
        let ax_scaled = modulate_tokens(&ax_norm3, &a_ca_scale, &a_ca_shift)?;
        vx = vx.broadcast_add(&gate_tokens(
            &self.audio_to_video_attn.forward(
                &vx_scaled,
                Some(&ax_scaled),
                None,
                Some((&video_cross_rope.0, &video_cross_rope.1)),
                Some((&audio_cross_rope.0, &audio_cross_rope.1)),
            )?,
            &v_gate,
        )?)?;

        let (a_ca_scale, a_ca_shift, a_gate) = self.get_cross_ada_values(
            &self.scale_shift_table_a2v_ca_audio,
            audio_cross_scale_shift_timestep,
            audio_cross_gate_timestep,
            2,
        )?;
        let (v_ca_scale, v_ca_shift, _) = self.get_cross_ada_values(
            &self.scale_shift_table_a2v_ca_video,
            video_cross_scale_shift_timestep,
            video_cross_gate_timestep,
            2,
        )?;
        let ax_scaled = modulate_tokens(&ax_norm3, &a_ca_scale, &a_ca_shift)?;
        let vx_scaled = modulate_tokens(&vx_norm3, &v_ca_scale, &v_ca_shift)?;
        ax = ax.broadcast_add(&gate_tokens(
            &self.video_to_audio_attn.forward(
                &ax_scaled,
                Some(&vx_scaled),
                None,
                Some((&audio_cross_rope.0, &audio_cross_rope.1)),
                Some((&video_cross_rope.0, &video_cross_rope.1)),
            )?,
            &a_gate,
        )?)?;

        let (v_shift_mlp, v_scale_mlp, v_gate_mlp) =
            self.get_ada_triplet(&self.video_scale_shift_table, &video.timesteps, 3)?;
        vx = vx.broadcast_add(&gate_tokens(
            &self.video_ff.forward(&modulate_tokens(
                &rms_norm_tensor(&vx, self.norm_eps)?,
                &v_scale_mlp,
                &v_shift_mlp,
            )?)?,
            &v_gate_mlp,
        )?)?;

        let (a_shift_mlp, a_scale_mlp, a_gate_mlp) =
            self.get_ada_triplet(&self.audio_scale_shift_table, &audio.timesteps, 3)?;
        ax = ax.broadcast_add(&gate_tokens(
            &self.audio_ff.forward(&modulate_tokens(
                &rms_norm_tensor(&ax, self.norm_eps)?,
                &a_scale_mlp,
                &a_shift_mlp,
            )?)?,
            &a_gate_mlp,
        )?)?;

        Ok((vx, ax))
    }
}

enum AvTransformerBlockSource {
    Eager(Vec<LtxAvTransformerBlock>),
    Streaming(VarBuilder<'static>),
}

pub struct Ltx2AvTransformer3DModel {
    patchify_proj: nn::Linear,
    adaln_single: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    scale_shift_table: Tensor,
    norm_out: LayerNormNoParams,
    proj_out: nn::Linear,
    audio_patchify_proj: nn::Linear,
    audio_adaln_single: AdaLayerNormSingle,
    audio_caption_projection: PixArtAlphaTextProjection,
    audio_scale_shift_table: Tensor,
    audio_norm_out: LayerNormNoParams,
    audio_proj_out: nn::Linear,
    av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle,
    av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle,
    av_ca_a2v_gate_adaln_single: AdaLayerNormSingle,
    av_ca_v2a_gate_adaln_single: AdaLayerNormSingle,
    video_rope: Ltx2VideoRotaryPosEmbed,
    audio_rope: Ltx2VideoRotaryPosEmbed,
    cross_rope: Ltx2VideoRotaryPosEmbed,
    transformer_blocks: AvTransformerBlockSource,
    config: Ltx2VideoTransformer3DModelConfig,
}

impl Ltx2AvTransformer3DModel {
    pub fn new_streaming(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder<'static>,
    ) -> Result<Self> {
        let video_dim = config.inner_dim();
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        let cross_max = config
            .positional_embedding_max_pos
            .first()
            .copied()
            .unwrap_or(20)
            .max(
                config
                    .audio_positional_embedding_max_pos
                    .first()
                    .copied()
                    .unwrap_or(20),
            );

        Ok(Self {
            patchify_proj: nn::linear(config.in_channels, video_dim, vb.pp("patchify_proj"))?,
            adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                6,
                vb.pp("adaln_single"),
            )?,
            caption_projection: PixArtAlphaTextProjection::new_with_out_features(
                config.caption_channels,
                video_dim,
                video_dim,
                vb.pp("caption_projection"),
            )?,
            scale_shift_table: vb.get((2, video_dim), "scale_shift_table")?,
            norm_out: LayerNormNoParams::new(config.norm_eps),
            proj_out: nn::linear(video_dim, config.out_channels, vb.pp("proj_out"))?,
            audio_patchify_proj: nn::linear(
                config.audio_in_channels,
                audio_dim,
                vb.pp("audio_patchify_proj"),
            )?,
            audio_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                6,
                vb.pp("audio_adaln_single"),
            )?,
            audio_caption_projection: PixArtAlphaTextProjection::new_with_out_features(
                config.caption_channels,
                audio_dim,
                audio_dim,
                vb.pp("audio_caption_projection"),
            )?,
            audio_scale_shift_table: vb.get((2, audio_dim), "audio_scale_shift_table")?,
            audio_norm_out: LayerNormNoParams::new(config.norm_eps),
            audio_proj_out: nn::linear(
                audio_dim,
                config.audio_out_channels,
                vb.pp("audio_proj_out"),
            )?,
            av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                4,
                vb.pp("av_ca_video_scale_shift_adaln_single"),
            )?,
            av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                4,
                vb.pp("av_ca_audio_scale_shift_adaln_single"),
            )?,
            av_ca_a2v_gate_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                video_dim,
                1,
                vb.pp("av_ca_a2v_gate_adaln_single"),
            )?,
            av_ca_v2a_gate_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                1,
                vb.pp("av_ca_v2a_gate_adaln_single"),
            )?,
            video_rope: Ltx2VideoRotaryPosEmbed::new(
                video_dim,
                config.positional_embedding_theta,
                config.positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            audio_rope: Ltx2VideoRotaryPosEmbed::new(
                audio_dim,
                config.positional_embedding_theta,
                config.audio_positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            cross_rope: Ltx2VideoRotaryPosEmbed::new(
                config.audio_cross_attention_dim,
                config.positional_embedding_theta,
                vec![cross_max],
                true,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            transformer_blocks: AvTransformerBlockSource::Streaming(vb.pp("transformer_blocks")),
            config: config.clone(),
        })
    }

    fn streaming_block(
        &self,
        blocks_vb: VarBuilder<'static>,
        index: usize,
    ) -> Result<LtxAvTransformerBlock> {
        LtxAvTransformerBlock::new(&self.config, blocks_vb.pp(index.to_string()))
    }

    fn prepare_context_mask(mask: Option<&Tensor>, dtype: DType) -> Result<Option<Tensor>> {
        match mask {
            Some(mask) if mask.rank() == 2 => Ok(Some(
                (mask.to_dtype(dtype)?.affine(-1.0, 1.0)? * (-10000.0))?.unsqueeze(1)?,
            )),
            Some(mask) => Ok(Some(mask.clone())),
            None => Ok(None),
        }
    }

    fn temporal_cross_positions(&self, positions: &Tensor, expected_dims: usize) -> Result<Tensor> {
        let dims = positions.dim(1)?;
        if dims == expected_dims {
            return Ok(positions.clone());
        }
        if dims < 1 {
            candle_core::bail!("expected at least one positional dimension, got {dims}");
        }
        positions.i((.., 0..1, .., ..))
    }

    fn prepare_modality(
        &self,
        latent: &Tensor,
        context: &Tensor,
        context_mask: Option<&Tensor>,
        timestep: &Tensor,
        positions: &Tensor,
        patchify_proj: &nn::Linear,
        adaln_single: &AdaLayerNormSingle,
        caption_projection: &PixArtAlphaTextProjection,
        rope: &Ltx2VideoRotaryPosEmbed,
    ) -> Result<LtxPreparedModality> {
        let x = patchify_proj.forward(latent)?;
        let (timesteps, embedded_timestep) = adaln_single.forward(&timestep.flatten_all()?)?;
        let batch = x.dim(0)?;
        let timesteps = timesteps.reshape((batch, 1, timesteps.dim(1)?))?;
        let embedded_timestep = embedded_timestep.reshape((batch, 1, embedded_timestep.dim(1)?))?;
        let context = caption_projection.forward(context)?;
        let rope = rope.forward(&x, positions)?;
        Ok(LtxPreparedModality {
            x,
            context,
            context_mask: Self::prepare_context_mask(context_mask, latent.dtype())?,
            timesteps,
            embedded_timestep,
            rope,
            cross_rope: None,
            cross_scale_shift_timestep: None,
            cross_gate_timestep: None,
            prompt_timestep: None,
        })
    }

    fn prepare_cross_attention_timestep(
        adaln: &AdaLayerNormSingle,
        timestep: &Tensor,
        scale: f64,
        batch: usize,
    ) -> Result<Tensor> {
        let (output, _) = adaln.forward(&timestep.flatten_all()?.affine(scale, 0.0)?)?;
        Ok(output.reshape((batch, 1, output.dim(1)?))?)
    }

    fn process_output(
        scale_shift_table: &Tensor,
        norm_out: &LayerNormNoParams,
        proj_out: &nn::Linear,
        x: &Tensor,
        embedded_timestep: &Tensor,
    ) -> Result<Tensor> {
        let tokens = x.dim(1)?;
        let table = scale_shift_table
            .to_dtype(embedded_timestep.dtype())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scale_shift = table.broadcast_add(&embedded_timestep.unsqueeze(2)?)?;
        let shift = scale_shift.i((.., .., 0, ..))?;
        let scale = scale_shift.i((.., .., 1, ..))?;
        let x = norm_out.forward(x)?;
        let scale = broadcast_to_tokens(&scale, tokens)?;
        let shift = broadcast_to_tokens(&shift, tokens)?;
        let one = Tensor::ones_like(&scale)?;
        proj_out.forward(
            &x.broadcast_mul(&one.broadcast_add(&scale)?)?
                .broadcast_add(&shift)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        video_hidden_states: &Tensor,
        audio_hidden_states: &Tensor,
        video_encoder_hidden_states: &Tensor,
        audio_encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        video_encoder_attention_mask: Option<&Tensor>,
        audio_encoder_attention_mask: Option<&Tensor>,
        video_positions: &Tensor,
        audio_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let compute_dtype = match self.patchify_proj.weight().dtype() {
            DType::F8E4M3 => DType::BF16,
            other => other,
        };
        let timestep = timestep.to_dtype(compute_dtype)?.affine(1000.0, 0.0)?;
        let mut video = self.prepare_modality(
            &video_hidden_states.to_dtype(compute_dtype)?,
            &video_encoder_hidden_states.to_dtype(compute_dtype)?,
            video_encoder_attention_mask,
            &timestep,
            video_positions,
            &self.patchify_proj,
            &self.adaln_single,
            &self.caption_projection,
            &self.video_rope,
        )?;
        let mut audio = self.prepare_modality(
            &audio_hidden_states.to_dtype(compute_dtype)?,
            &audio_encoder_hidden_states.to_dtype(compute_dtype)?,
            audio_encoder_attention_mask,
            &timestep,
            audio_positions,
            &self.audio_patchify_proj,
            &self.audio_adaln_single,
            &self.audio_caption_projection,
            &self.audio_rope,
        )?;
        let batch = video.x.dim(0)?;
        let av_scale = self.config.av_ca_timestep_scale_multiplier / 1000.0;
        let video_cross_positions = self.temporal_cross_positions(video_positions, 1)?;
        let audio_cross_positions = self.temporal_cross_positions(audio_positions, 1)?;
        video.cross_rope = Some(self.cross_rope.forward(&video.x, &video_cross_positions)?);
        audio.cross_rope = Some(self.cross_rope.forward(&audio.x, &audio_cross_positions)?);
        video.cross_scale_shift_timestep = Some(Self::prepare_cross_attention_timestep(
            &self.av_ca_video_scale_shift_adaln_single,
            &timestep,
            1.0,
            batch,
        )?);
        audio.cross_scale_shift_timestep = Some(Self::prepare_cross_attention_timestep(
            &self.av_ca_audio_scale_shift_adaln_single,
            &timestep,
            1.0,
            batch,
        )?);
        video.cross_gate_timestep = Some(Self::prepare_cross_attention_timestep(
            &self.av_ca_a2v_gate_adaln_single,
            &timestep,
            av_scale,
            batch,
        )?);
        audio.cross_gate_timestep = Some(Self::prepare_cross_attention_timestep(
            &self.av_ca_v2a_gate_adaln_single,
            &timestep,
            av_scale,
            batch,
        )?);

        match &self.transformer_blocks {
            AvTransformerBlockSource::Eager(blocks) => {
                for block in blocks {
                    let (vx, ax) = block.forward(&video, &audio)?;
                    video.x = vx;
                    audio.x = ax;
                }
            }
            AvTransformerBlockSource::Streaming(blocks_vb) => {
                for index in 0..self.config.num_layers {
                    let block = self.streaming_block(blocks_vb.clone(), index)?;
                    let (vx, ax) = block.forward(&video, &audio)?;
                    video.x = vx;
                    audio.x = ax;
                    if video.x.device().is_cuda() {
                        video.x.device().synchronize()?;
                    }
                }
            }
        }

        Ok((
            Self::process_output(
                &self.scale_shift_table,
                &self.norm_out,
                &self.proj_out,
                &video.x,
                &video.embedded_timestep,
            )?,
            Self::process_output(
                &self.audio_scale_shift_table,
                &self.audio_norm_out,
                &self.audio_proj_out,
                &audio.x,
                &audio.embedded_timestep,
            )?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{LtxAttention, LtxRopeType};

    fn attention_var_builder(dim: usize) -> VarBuilder<'static> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        let mut identity = vec![0.0f32; dim * dim];
        for idx in 0..dim {
            identity[idx * dim + idx] = 1.0;
        }
        for name in ["to_q", "to_k", "to_v", "to_out.0"] {
            tensors.insert(
                format!("{name}.weight"),
                Tensor::from_vec(identity.clone(), (dim, dim), &device).unwrap(),
            );
            tensors.insert(
                format!("{name}.bias"),
                Tensor::zeros(dim, DType::F32, &device).unwrap(),
            );
        }
        tensors.insert(
            "norm_q.weight".to_string(),
            Tensor::ones(dim, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "norm_k.weight".to_string(),
            Tensor::ones(dim, DType::F32, &device).unwrap(),
        );
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    }

    #[test]
    fn text_cross_attention_ignores_video_rope_without_key_rotary_inputs() {
        let attention = LtxAttention::new(
            4,
            1,
            1,
            4,
            0.0,
            true,
            Some(4),
            true,
            "rms_norm",
            LtxRopeType::Interleaved,
            attention_var_builder(4),
        )
        .unwrap();
        let hidden_states = Tensor::new(
            &[[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let encoder_hidden_states = Tensor::new(
            &[[
                [0.5f32, 1.0, 1.5, 2.0],
                [2.0, 1.5, 1.0, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]],
            &Device::Cpu,
        )
        .unwrap();
        let cos = Tensor::new(
            &[[[1.0f32, 0.5, -1.0, 0.25], [0.25, -1.0, 0.5, 1.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let sin = Tensor::new(
            &[[
                [0.0f32, 0.8660254, 0.0, -0.9689124],
                [0.9689124, 0.0, -0.8660254, 0.0],
            ]],
            &Device::Cpu,
        )
        .unwrap();

        let baseline = attention
            .forward(
                &hidden_states,
                Some(&encoder_hidden_states),
                None,
                None,
                None,
            )
            .unwrap();
        let with_video_rope = attention
            .forward(
                &hidden_states,
                Some(&encoder_hidden_states),
                None,
                Some((&cos, &sin)),
                None,
            )
            .unwrap();

        assert_eq!(
            baseline.to_vec3::<f32>().unwrap(),
            with_video_rope.to_vec3::<f32>().unwrap()
        );
    }
}
