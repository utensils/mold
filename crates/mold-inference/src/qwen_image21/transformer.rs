//! Qwen Image 2.1's 32-block causal-condition diffusion transformer.
//!
//! The checkpoint is deliberately implemented separately from
//! `qwen_image::transformer`: 2.1 consumes unpatched 64-channel latents,
//! conditions with Qwen3-VL's 4096-wide final pre-norm states, and uses a
//! single stream whose text prefix is causal while the target-image block is
//! bidirectional.  Treating it as the older 60-block dual-stream model would
//! load neither its tensors nor its attention topology correctly.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use std::path::PathBuf;

use super::QwenImage21TextConditioning;

/// Official `transformer/config.json` geometry for Qwen/Qwen-Image-2.1.
#[derive(Debug, Clone)]
pub(crate) struct QwenImage21TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_in_dim: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub mlp_ratio: usize,
    pub axes_dims_rope: [usize; 3],
    pub eps: f64,
}

impl QwenImage21TransformerConfig {
    pub(crate) fn official() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            context_in_dim: 4096,
            num_attention_heads: 32,
            attention_head_dim: 128,
            num_layers: 32,
            mlp_ratio: 3,
            axes_dims_rope: [16, 56, 56],
            eps: 1e-6,
        }
    }

    fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.in_channels > 0,
            "Qwen Image 2.1 in_channels must be positive"
        );
        anyhow::ensure!(
            self.out_channels > 0,
            "Qwen Image 2.1 out_channels must be positive"
        );
        anyhow::ensure!(
            self.context_in_dim > 0,
            "Qwen Image 2.1 context_in_dim must be positive"
        );
        anyhow::ensure!(
            self.num_attention_heads > 0,
            "Qwen Image 2.1 needs attention heads"
        );
        anyhow::ensure!(
            self.attention_head_dim > 0,
            "Qwen Image 2.1 needs a head dimension"
        );
        anyhow::ensure!(
            self.num_layers > 0,
            "Qwen Image 2.1 needs transformer layers"
        );
        anyhow::ensure!(
            self.mlp_ratio > 0,
            "Qwen Image 2.1 MLP ratio must be positive"
        );
        anyhow::ensure!(
            self.axes_dims_rope.iter().sum::<usize>() == self.attention_head_dim,
            "Qwen Image 2.1 RoPE axes {:?} do not sum to head dim {}",
            self.axes_dims_rope,
            self.attention_head_dim
        );
        anyhow::ensure!(
            self.axes_dims_rope.iter().all(|axis| axis % 2 == 0),
            "Qwen Image 2.1 RoPE axes must be even"
        );
        Ok(())
    }
}

/// Zero-centered RMS norm used by `txt_in.text_norm`.
///
/// The checkpoint stores `scale - 1`, so the effective learned multiplier is
/// `weight + 1`; it is not the ordinary Qwen3 RMSNorm used by the text model.
struct ZeroCenterRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ZeroCenterRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let f32_x = xs.to_dtype(DType::F32)?;
        let rms = (f32_x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let scale = (self.weight.to_dtype(DType::F32)? + 1.0)?;
        f32_x
            .broadcast_div(&rms)?
            .broadcast_mul(&scale)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

/// Affine-free LayerNorm evaluated in FP32, matching PyTorch LayerNorm's
/// accumulation for the BF16 inference path.
struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    fn new(eps: f64) -> Self {
        Self { eps }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let f32_x = xs.to_dtype(DType::F32)?;
        let mean = f32_x.mean_keepdim(D::Minus1)?;
        let centered = f32_x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

struct TextProjection {
    text_norm: ZeroCenterRmsNorm,
    in_layer: Linear,
    out_layer: Linear,
}

impl TextProjection {
    fn new(cfg: &QwenImage21TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            text_norm: ZeroCenterRmsNorm::new(cfg.context_in_dim, cfg.eps, vb.pp("text_norm"))?,
            in_layer: candle_nn::linear_no_bias(cfg.context_in_dim, inner, vb.pp("in_layer"))?,
            out_layer: candle_nn::linear_no_bias(inner, inner, vb.pp("out_layer"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_layer
            .forward(
                &candle_nn::Activation::GeluPytorchTanh
                    .forward(&self.in_layer.forward(&self.text_norm.forward(xs)?)?)?,
            )
            .map_err(Into::into)
    }
}

struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
    inner_dim: usize,
}

impl TimestepEmbedder {
    fn new(inner_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            linear_1: candle_nn::linear_no_bias(256, inner_dim, vb.pp("linear_1"))?,
            linear_2: candle_nn::linear_no_bias(inner_dim, inner_dim, vb.pp("linear_2"))?,
            inner_dim,
        })
    }

    fn temporal_embedding(timesteps: &[f64], dtype: DType, device: &Device) -> Result<Tensor> {
        const DIM: usize = 256;
        const HALF: usize = DIM / 2;
        const MAX_PERIOD: f64 = 10_000.0;
        let mut values = Vec::with_capacity(timesteps.len() * DIM);
        for &timestep in timesteps {
            // The reference's temporal projector multiplies the normalized
            // `[0, 1]` timestep by 1000 before applying the sinusoid.
            let timestep = timestep * 1000.0;
            for index in 0..HALF {
                let frequency = (-MAX_PERIOD.ln() * index as f64 / HALF as f64).exp();
                values.push((timestep * frequency).cos() as f32);
            }
            for index in 0..HALF {
                let frequency = (-MAX_PERIOD.ln() * index as f64 / HALF as f64).exp();
                values.push((timestep * frequency).sin() as f32);
            }
        }
        Tensor::from_vec(values, (timesteps.len(), DIM), device)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }

    fn forward(&self, timesteps: &[f64], dtype: DType, device: &Device) -> Result<Tensor> {
        let embedding = Self::temporal_embedding(timesteps, dtype, device)?;
        let embedding = self.linear_1.forward(&embedding)?;
        let embedding = candle_nn::Activation::Silu.forward(&embedding)?;
        let embedding = self.linear_2.forward(&embedding)?;
        anyhow::ensure!(
            embedding.dim(D::Minus1)? == self.inner_dim,
            "Qwen Image 2.1 timestep embedder returned the wrong width"
        );
        Ok(embedding)
    }
}

struct SwiGlu {
    proj: Linear,
    gate_layer: Linear,
    out: Linear,
}

impl SwiGlu {
    fn new(dim: usize, mlp_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            proj: candle_nn::linear_no_bias(dim, mlp_dim, vb.pp("proj"))?,
            gate_layer: candle_nn::linear_no_bias(dim, mlp_dim, vb.pp("gate_layer"))?,
            out: candle_nn::linear_no_bias(mlp_dim, dim, vb.pp("out"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_layer.forward(xs)?)?;
        self.out
            .forward(&(gate * self.proj.forward(xs)?)?)
            .map_err(Into::into)
    }
}

struct PrefixKv {
    key: Tensor,
    value: Tensor,
}

enum PrefixCache<'a> {
    Disabled,
    Extract(&'a mut Vec<PrefixKv>),
    Reuse(&'a [PrefixKv]),
}

enum LayerCache<'a> {
    Disabled,
    Extract(&'a mut Vec<PrefixKv>),
    Reuse(&'a PrefixKv),
}

struct Attention {
    fused_target: bool,
    fused_ops: bool,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn new(cfg: &QwenImage21TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            fused_target: crate::attention::metal_fast_path_enabled(),
            fused_ops: crate::attention::metal_fast_path_enabled(),
            to_q: candle_nn::linear_no_bias(inner, inner, vb.pp("to_q"))?,
            to_k: candle_nn::linear_no_bias(inner, inner, vb.pp("to_k"))?,
            to_v: candle_nn::linear_no_bias(inner, inner, vb.pp("to_v"))?,
            to_out: candle_nn::linear_no_bias(inner, inner, vb.pp("to_out").pp("0"))?,
            norm_q: vb.pp("norm_q").get(cfg.attention_head_dim, "weight")?,
            norm_k: vb.pp("norm_k").get(cfg.attention_head_dim, "weight")?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.eps,
        })
    }

    /// Image queries are non-causal and usually unmasked. Keep the small
    /// causal prefix and padded batches on math rather than expanding a mask
    /// to the full score matrix or changing its negative-infinity semantics.
    fn target_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        if self.fused_target
            && q.device().is_metal()
            && bias.is_none()
            && matches!(self.head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
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

    fn normalize_heads(&self, xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let (batch, heads, sequence, head_dim) = xs.dims4()?;
        anyhow::ensure!(
            heads == self.heads && head_dim == self.head_dim,
            "Qwen Image 2.1 attention head shape mismatch"
        );
        let flat = xs.flatten(0, 2)?;
        let weight = if weight.dtype() == flat.dtype() {
            weight.clone()
        } else {
            weight.to_dtype(flat.dtype())?
        };
        candle_nn::ops::rms_norm(&flat, &weight, self.eps as f32)?
            .reshape((batch, heads, sequence, head_dim))
            .map_err(Into::into)
    }

    fn t2i_prefix_bias(
        valid_tokens: &[Vec<bool>],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let batch = valid_tokens.len();
        let text_len = valid_tokens.first().map_or(0, Vec::len);
        anyhow::ensure!(text_len > 0, "Qwen Image 2.1 text conditioning is empty");
        let mut values = Vec::with_capacity(batch * text_len * text_len);
        for (batch_index, row) in valid_tokens.iter().enumerate() {
            anyhow::ensure!(
                row.len() == text_len,
                "Qwen Image 2.1 text-mask row {batch_index} has {}, expected {text_len}",
                row.len()
            );
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

    /// Target rows see every target token and every *valid* text token.
    /// Return `None` when every text row is valid so the attention dispatcher
    /// remains eligible for its no-bias optimized path.
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
        for (batch_index, row) in valid_tokens.iter().enumerate() {
            anyhow::ensure!(
                row.len() == text_len,
                "Qwen Image 2.1 text-mask row {batch_index} has {}, expected {text_len}",
                row.len()
            );
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

    fn forward_t2i(
        &self,
        hidden_states: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        valid_tokens: &[Vec<bool>],
        cache: LayerCache<'_>,
    ) -> Result<Tensor> {
        let (batch, sequence, inner) = hidden_states.dims3()?;
        let text_len = valid_tokens.first().map_or(0, Vec::len);
        let cached = matches!(cache, LayerCache::Reuse(_));
        anyhow::ensure!(
            text_len > 0 && (cached || text_len < sequence),
            "Qwen Image 2.1 joint sequence needs non-empty text and target-image blocks"
        );
        let target_tokens = if cached {
            sequence
        } else {
            sequence - text_len
        };
        anyhow::ensure!(
            inner == self.heads * self.head_dim,
            "Qwen Image 2.1 attention inner width mismatch"
        );

        let (q, k, v) = if self.fused_ops && hidden_states.device().is_metal() {
            // Normalize while BSHD is contiguous; flattening BHSD first copies
            // the whole projection. RoPE still accumulates in F32 as upstream.
            let project = |linear: &Linear, weight: &Tensor| -> Result<Tensor> {
                let xs = linear.forward(hidden_states)?;
                let normalized = candle_nn::ops::rms_norm(
                    &xs.reshape((batch * sequence * self.heads, self.head_dim))?,
                    weight,
                    self.eps as f32,
                )?
                .reshape((batch, sequence, self.heads, self.head_dim))?
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
                project(&self.to_q, &self.norm_q)?,
                project(&self.to_k, &self.norm_k)?,
                self.to_v
                    .forward(hidden_states)?
                    .reshape((batch, sequence, self.heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
            )
        } else {
            let project = |linear: &Linear| -> Result<Tensor> {
                linear
                    .forward(hidden_states)?
                    .reshape((batch, sequence, self.heads, self.head_dim))?
                    .transpose(1, 2)
                    .map_err(Into::into)
            };
            let q = self.normalize_heads(&project(&self.to_q)?, &self.norm_q)?;
            let k = self.normalize_heads(&project(&self.to_k)?, &self.norm_k)?;
            let v = project(&self.to_v)?;

            // Wan's public RoPE helper has exactly the interleaved complex-pair
            // layout that Qwen Image 2.1's `apply_rotary_emb_qwen(...,
            // use_real=False)` uses.  Its input is BSHD; attention below is BHSD.
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
            LayerCache::Extract(layers) => {
                // A contiguous view can still retain the full joint allocation
                // (notably with one head). Copy only the immutable prefix.
                layers.push(PrefixKv {
                    key: k.narrow(2, 0, text_len)?.force_contiguous()?,
                    value: v.narrow(2, 0, text_len)?.force_contiguous()?,
                });
            }
            LayerCache::Reuse(prefix) => {
                let k = Tensor::cat(&[&prefix.key, &k], 2)?;
                let v = Tensor::cat(&[&prefix.value, &v], 2)?;
                let bias =
                    Self::t2i_target_bias(valid_tokens, target_tokens, q.dtype(), q.device())?;
                let context = self.target_attention(&q, &k, &v, bias.as_ref())?;
                return self
                    .to_out
                    .forward(&context.transpose(1, 2)?.reshape((batch, sequence, inner))?)
                    .map_err(Into::into);
            }
            LayerCache::Disabled => {}
        }

        // The prefix's own attention is causal, whereas target-image rows are
        // fully bidirectional within their block and see the complete prefix.
        // Splitting those two calls is algebraically identical to the model's
        // block-causal mask and avoids ever allocating an N-by-N score matrix
        // for the tiny text prefix.
        // Candle Metal matmul requires the physical tensor extent to match
        // the view it receives. A `narrow` retains the full joint sequence's
        // backing allocation (the target is a suffix after text), so make
        // each independently-attended segment contiguous before dispatch.
        let prefix_q = q.narrow(2, 0, text_len)?.contiguous()?;
        let prefix_k = k.narrow(2, 0, text_len)?.contiguous()?;
        let prefix_v = v.narrow(2, 0, text_len)?.contiguous()?;
        let prefix_bias = Self::t2i_prefix_bias(valid_tokens, q.dtype(), q.device())?;
        let prefix = crate::attention::attention_with_bias(
            &prefix_q,
            &prefix_k,
            &prefix_v,
            (1.0 / (self.head_dim as f64).sqrt()) as f32,
            Some(&prefix_bias),
        )?;

        let target_q = q.narrow(2, text_len, target_tokens)?.contiguous()?;
        let target_bias =
            Self::t2i_target_bias(valid_tokens, target_tokens, q.dtype(), q.device())?;
        let target = self.target_attention(&target_q, &k, &v, target_bias.as_ref())?;
        let context = Tensor::cat(&[&prefix, &target], 2)?;
        self.to_out
            .forward(&context.transpose(1, 2)?.reshape((batch, sequence, inner))?)
            .map_err(Into::into)
    }
}

struct TransformerBlock {
    norm1: LayerNormNoParams,
    attn: Attention,
    norm2: LayerNormNoParams,
    mlp: SwiGlu,
}

impl TransformerBlock {
    fn new(cfg: &QwenImage21TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            norm1: LayerNormNoParams::new(cfg.eps),
            attn: Attention::new(cfg, vb.pp("attn"))?,
            norm2: LayerNormNoParams::new(cfg.eps),
            mlp: SwiGlu::new(inner, inner * cfg.mlp_ratio, vb.pp("img_mlp"))?,
        })
    }

    fn modulate(normalized: Tensor, parameters: &Tensor) -> Result<(Tensor, Tensor)> {
        let width = parameters.dim(D::Minus1)?;
        anyhow::ensure!(
            width % 2 == 0,
            "Qwen Image 2.1 modulation width must be even"
        );
        let half = width / 2;
        let scale = parameters.narrow(D::Minus1, 0, half)?;
        let gate = parameters.narrow(D::Minus1, half, half)?;
        Ok((normalized.broadcast_mul(&(scale + 1.0)?)?, gate))
    }

    fn forward_t2i(
        &self,
        hidden_states: &Tensor,
        modulation: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        valid_tokens: &[Vec<bool>],
        cache: LayerCache<'_>,
    ) -> Result<Tensor> {
        let dim = hidden_states.dim(D::Minus1)?;
        anyhow::ensure!(
            modulation.dim(D::Minus1)? == 4 * dim,
            "Qwen Image 2.1 modulation width does not match hidden width"
        );
        let mod1 = modulation.narrow(D::Minus1, 0, 2 * dim)?;
        let mod2 = modulation.narrow(D::Minus1, 2 * dim, 2 * dim)?;

        let (normalized, gate) = Self::modulate(self.norm1.forward(hidden_states)?, &mod1)?;
        let attn = self
            .attn
            .forward_t2i(&normalized, rope_cos, rope_sin, valid_tokens, cache)?;
        let hidden_states = (hidden_states + gate.tanh()?.broadcast_mul(&attn)?)?;

        let (normalized, gate) = Self::modulate(self.norm2.forward(&hidden_states)?, &mod2)?;
        let hidden_states = (&hidden_states
            + gate
                .tanh()?
                .broadcast_mul(&self.mlp.forward(&normalized)?)?)?;
        if hidden_states.dtype() == DType::F16 {
            return hidden_states
                .clamp(-65_504.0f32, 65_504.0f32)
                .map_err(Into::into);
        }
        Ok(hidden_states)
    }
}

struct AdaFinalNorm {
    norm: LayerNormNoParams,
    linear: Linear,
}

impl AdaFinalNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            norm: LayerNormNoParams::new(eps),
            linear: candle_nn::linear_no_bias(dim, dim, vb.pp("linear"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let scale = self
            .linear
            .forward(&candle_nn::Activation::Silu.forward(conditioning)?)?;
        self.norm
            .forward(hidden_states)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)
            .map_err(Into::into)
    }
}

/// Dense Qwen Image 2.1 transformer loaded from its two official safetensor
/// shards.  This first native path intentionally supports text-to-image only;
/// image-conditioned generation additionally needs Qwen3-VL's vision tower
/// and is rejected by the engine rather than silently treating an image as
/// text-only conditioning.
pub(crate) struct QwenImage21Transformer {
    compact_modulation: bool,
    cfg: QwenImage21TransformerConfig,
    img_in: Linear,
    time_text_embed: TimestepEmbedder,
    txt_in: TextProjection,
    modulation: Linear,
    blocks: Vec<TransformerBlock>,
    norm_out: AdaFinalNorm,
    proj_out: Linear,
}

/// One conditioning branch of one denoise request. The borrow ties the cache
/// to its exact transformer and immutable prompt; nothing survives the request.
pub(crate) struct PreparedT2i<'a> {
    transformer: &'a QwenImage21Transformer,
    conditioning: &'a QwenImage21TextConditioning,
    height: usize,
    width: usize,
    layers: Vec<PrefixKv>,
}

impl PreparedT2i<'_> {
    pub(crate) fn forward(&mut self, latents: &Tensor, timestep: f64) -> Result<Tensor> {
        // Keep the retained allocation within the admission budget. Longer
        // prompts still render in full; they simply recompute their prefix.
        if self.conditioning.sequence_length() > super::PREFIX_CACHE_MAX_TOKENS {
            return self.transformer.forward_with_cache(
                latents,
                timestep,
                self.conditioning,
                self.height,
                self.width,
                PrefixCache::Disabled,
            );
        }
        if self.layers.is_empty() {
            let mut layers = Vec::with_capacity(self.transformer.blocks.len());
            let result = self.transformer.forward_with_cache(
                latents,
                timestep,
                self.conditioning,
                self.height,
                self.width,
                PrefixCache::Extract(&mut layers),
            )?;
            // Publish only a complete, successful prefill.
            self.layers = layers;
            Ok(result)
        } else {
            let key = &self.layers[0].key;
            anyhow::ensure!(
                key.dtype() == latents.dtype() && key.device().same_device(latents.device()),
                "Qwen Image 2.1 cached denoise cannot change device or dtype"
            );
            self.transformer.forward_with_cache(
                latents,
                timestep,
                self.conditioning,
                self.height,
                self.width,
                PrefixCache::Reuse(&self.layers),
            )
        }
    }
}

impl QwenImage21Transformer {
    pub(crate) fn prepare_t2i<'a>(
        &'a self,
        conditioning: &'a QwenImage21TextConditioning,
        height: usize,
        width: usize,
    ) -> PreparedT2i<'a> {
        PreparedT2i {
            transformer: self,
            conditioning,
            height,
            width,
            layers: Vec::new(),
        }
    }

    pub(crate) fn load(
        paths: &[PathBuf],
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            paths,
            dtype,
            device,
            "Qwen Image 2.1 transformer",
            progress,
        )?;
        Self::from_var_builder(QwenImage21TransformerConfig::official(), vb)
    }

    pub(crate) fn from_var_builder(
        cfg: QwenImage21TransformerConfig,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        cfg.validate()?;
        let inner = cfg.inner_dim();
        let img_in = candle_nn::linear_no_bias(cfg.in_channels, inner, vb.pp("img_in"))?;
        let time_text_embed =
            TimestepEmbedder::new(inner, vb.pp("time_text_embed").pp("timestep_embedder"))?;
        let txt_in = TextProjection::new(&cfg, vb.pp("txt_in"))?;
        let modulation = candle_nn::linear_no_bias(inner, 4 * inner, vb.pp("modulation").pp("1"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for index in 0..cfg.num_layers {
            blocks.push(TransformerBlock::new(
                &cfg,
                vb.pp("transformer_blocks").pp(index),
            )?);
        }
        let norm_out = AdaFinalNorm::new(inner, cfg.eps, vb.pp("norm_out"))?;
        let proj_out = candle_nn::linear_no_bias(inner, cfg.out_channels, vb.pp("proj_out"))?;
        Ok(Self {
            compact_modulation: crate::attention::metal_fast_path_enabled(),
            cfg,
            img_in,
            time_text_embed,
            txt_in,
            modulation,
            blocks,
            norm_out,
            proj_out,
        })
    }

    /// Build 3-axis RoPE for a text prefix followed by one target image block.
    ///
    /// The frame coordinate advances across text and freezes at the text cursor
    /// for the image. Height and width are centred around zero inside the image
    /// block. This is the upstream `QwenImage21Rope` rule specialized only to
    /// the text-to-image layout.
    fn t2i_rope(
        &self,
        text_len: usize,
        latent_height: usize,
        latent_width: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        anyhow::ensure!(
            latent_height > 0 && latent_width > 0,
            "Qwen Image 2.1 latent dimensions must be positive"
        );
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

        let mut cos = Vec::with_capacity(coords.len() * (self.cfg.attention_head_dim / 2));
        let mut sin = Vec::with_capacity(coords.len() * (self.cfg.attention_head_dim / 2));
        for coordinate in coords {
            for (axis, axis_dim) in self.cfg.axes_dims_rope.iter().copied().enumerate() {
                for index in (0..axis_dim).step_by(2) {
                    let frequency = 1.0 / 10_000.0f64.powf(index as f64 / axis_dim as f64);
                    let angle = coordinate[axis] as f64 * frequency;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        let sequence = text_len + target_len;
        // Qualify the Metal BF16 candidate with full-precision rotary tables.
        // Other backends retain their existing rounding boundary.
        let dtype = if device.is_metal() { DType::F32 } else { dtype };
        Ok((
            Tensor::from_vec(cos, (sequence, self.cfg.attention_head_dim / 2), device)?
                .to_dtype(dtype)?,
            Tensor::from_vec(sin, (sequence, self.cfg.attention_head_dim / 2), device)?
                .to_dtype(dtype)?,
        ))
    }

    /// Denoise a packed `[B, H*W, 64]` text-to-image latent tensor.
    ///
    /// `timestep` is normalized to `[0, 1]`, matching the Diffusers transformer's
    /// call (`scheduler_timestep / 1000`).
    #[cfg(test)]
    fn forward_t2i(
        &self,
        latents: &Tensor,
        timestep: f64,
        conditioning: &QwenImage21TextConditioning,
        latent_height: usize,
        latent_width: usize,
    ) -> Result<Tensor> {
        self.forward_with_cache(
            latents,
            timestep,
            conditioning,
            latent_height,
            latent_width,
            PrefixCache::Disabled,
        )
    }

    fn forward_with_cache(
        &self,
        latents: &Tensor,
        timestep: f64,
        conditioning: &QwenImage21TextConditioning,
        latent_height: usize,
        latent_width: usize,
        mut cache: PrefixCache<'_>,
    ) -> Result<Tensor> {
        let cached = matches!(cache, PrefixCache::Reuse(_));
        let (batch, target_tokens, latent_channels) = latents.dims3()?;
        anyhow::ensure!(
            latent_channels == self.cfg.in_channels,
            "Qwen Image 2.1 expected {} latent channels, got {latent_channels}",
            self.cfg.in_channels
        );
        anyhow::ensure!(
            target_tokens == latent_height.saturating_mul(latent_width),
            "Qwen Image 2.1 packed latent length {target_tokens} does not match {latent_height}x{latent_width}"
        );
        anyhow::ensure!(
            conditioning.batch_size() == batch,
            "Qwen Image 2.1 text batch {} does not match latent batch {batch}",
            conditioning.batch_size()
        );
        anyhow::ensure!(
            conditioning.image_slots.iter().all(|row| row.iter().all(|slot| !*slot)),
            "Qwen Image 2.1 image conditioning is not implemented yet; use text-to-image without reference media"
        );

        let text_len = conditioning.sequence_length();
        anyhow::ensure!(text_len > 0, "Qwen Image 2.1 text conditioning is empty");
        anyhow::ensure!(
            conditioning.valid_tokens.len() == batch
                && conditioning
                    .valid_tokens
                    .iter()
                    .all(|row| row.len() == text_len && row[0]),
            "Qwen Image 2.1 requires a right-padded text mask with a valid first token"
        );
        let text = conditioning
            .embeddings
            .to_device(latents.device())?
            .to_dtype(latents.dtype())?;
        let (text_batch, checked_text_len, text_dim) = text.dims3()?;
        anyhow::ensure!(
            text_batch == batch
                && checked_text_len == text_len
                && text_dim == self.cfg.context_in_dim,
            "Qwen Image 2.1 text embedding shape does not match its checkpoint contract"
        );

        let target = self.img_in.forward(latents)?;
        let mut hidden_states = if cached {
            target
        } else {
            Tensor::cat(&[&self.txt_in.forward(&text)?, &target], 1)?
        };
        let (rope_cos, rope_sin) = self.t2i_rope(
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

        // `causal_condition`: text positions take a dedicated t=0 modulation
        // row, while target image positions take each sample's real timestep.
        let mut timesteps = vec![timestep; batch];
        timesteps.push(0.0);
        let temb = self
            .time_text_embed
            .forward(&timesteps, latents.dtype(), latents.device())?;
        let modulation = self
            .modulation
            .forward(&candle_nn::Activation::Silu.forward(&temb)?)?;
        let inner = self.cfg.inner_dim();
        let real_row = modulation.narrow(0, 0, batch)?.unsqueeze(1)?;
        let real = real_row.broadcast_as((batch, target_tokens, 4 * inner))?;
        let zero = modulation
            .narrow(0, batch, 1)?
            .unsqueeze(1)?
            .broadcast_as((batch, text_len, 4 * inner))?;
        let per_token_modulation = if cached {
            // Every target position shares a timestep. Keep the row compact
            // so each block's scale and tanh execute once per feature, rather
            // than once per image token. Broadcast only at the residual ops.
            if latents.device().is_metal() && self.compact_modulation {
                real_row
            } else {
                real
            }
        } else {
            Tensor::cat(&[&zero, &real], 1)?
        };

        for (index, block) in self.blocks.iter().enumerate() {
            let layer_cache = match &mut cache {
                PrefixCache::Extract(layers) => LayerCache::Extract(layers),
                PrefixCache::Reuse(layers) => LayerCache::Reuse(&layers[index]),
                PrefixCache::Disabled => LayerCache::Disabled,
            };
            hidden_states = block.forward_t2i(
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
        self.proj_out
            .forward(&self.norm_out.forward(&target_hidden, &target_temb)?)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_config() -> QwenImage21TransformerConfig {
        QwenImage21TransformerConfig {
            in_channels: 4,
            out_channels: 4,
            context_in_dim: 8,
            num_attention_heads: 1,
            attention_head_dim: 8,
            num_layers: 1,
            mlp_ratio: 2,
            axes_dims_rope: [2, 2, 4],
            eps: 1e-6,
        }
    }

    fn add_tensor(
        map: &mut HashMap<String, Tensor>,
        name: impl Into<String>,
        shape: impl Into<candle_core::Shape>,
    ) {
        let shape = shape.into();
        let count = shape.elem_count();
        let data = (0..count)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
            .collect::<Vec<_>>();
        map.insert(
            name.into(),
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap(),
        );
    }

    fn tiny_transformer() -> QwenImage21Transformer {
        tiny_transformer_on(tiny_config(), &Device::Cpu)
    }

    fn tiny_transformer_on(
        cfg: QwenImage21TransformerConfig,
        device: &Device,
    ) -> QwenImage21Transformer {
        let inner = cfg.inner_dim();
        let mut map = HashMap::new();
        add_tensor(&mut map, "img_in.weight", (inner, cfg.in_channels));
        add_tensor(
            &mut map,
            "time_text_embed.timestep_embedder.linear_1.weight",
            (inner, 256),
        );
        add_tensor(
            &mut map,
            "time_text_embed.timestep_embedder.linear_2.weight",
            (inner, inner),
        );
        add_tensor(&mut map, "txt_in.text_norm.weight", cfg.context_in_dim);
        add_tensor(
            &mut map,
            "txt_in.in_layer.weight",
            (inner, cfg.context_in_dim),
        );
        add_tensor(&mut map, "txt_in.out_layer.weight", (inner, inner));
        add_tensor(&mut map, "modulation.1.weight", (4 * inner, inner));
        add_tensor(&mut map, "norm_out.linear.weight", (inner, inner));
        add_tensor(&mut map, "proj_out.weight", (cfg.out_channels, inner));
        for index in 0..cfg.num_layers {
            let prefix = format!("transformer_blocks.{index}");
            add_tensor(
                &mut map,
                format!("{prefix}.attn.norm_q.weight"),
                cfg.attention_head_dim,
            );
            add_tensor(
                &mut map,
                format!("{prefix}.attn.norm_k.weight"),
                cfg.attention_head_dim,
            );
            for name in ["to_q", "to_k", "to_v"] {
                add_tensor(
                    &mut map,
                    format!("{prefix}.attn.{name}.weight"),
                    (inner, inner),
                );
            }
            add_tensor(
                &mut map,
                format!("{prefix}.attn.to_out.0.weight"),
                (inner, inner),
            );
            add_tensor(
                &mut map,
                format!("{prefix}.img_mlp.proj.weight"),
                (inner * cfg.mlp_ratio, inner),
            );
            add_tensor(
                &mut map,
                format!("{prefix}.img_mlp.gate_layer.weight"),
                (inner * cfg.mlp_ratio, inner),
            );
            add_tensor(
                &mut map,
                format!("{prefix}.img_mlp.out.weight"),
                (inner, inner * cfg.mlp_ratio),
            );
        }
        let map = map
            .into_iter()
            .map(|(name, value)| (name, value.to_device(device).unwrap()))
            .collect();
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
        QwenImage21Transformer::from_var_builder(cfg, vb).unwrap()
    }

    #[test]
    fn official_config_matches_checkpoint_geometry() {
        let cfg = QwenImage21TransformerConfig::official();
        assert_eq!(cfg.in_channels, 64);
        assert_eq!(cfg.out_channels, 64);
        assert_eq!(cfg.context_in_dim, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.attention_head_dim, 128);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.axes_dims_rope, [16, 56, 56]);
        cfg.validate().unwrap();
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        let error = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let peak = expected
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Metal may select a different GEMM kernel for the shorter sequence;
        // the one-head fixture measures up to 4.52e-5 (CPU is exact).
        assert!(
            error < 1e-4,
            "cached prediction max error {error}, peak {peak}"
        );
    }

    fn cache_parity(device: &Device, heads: usize) {
        let mut cfg = tiny_config();
        cfg.num_layers = 3;
        cfg.num_attention_heads = heads;
        let transformer = tiny_transformer_on(cfg, device);
        let make_conditioning = |length, seed| QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(seed, &[2, length, 8], device, DType::F32)
                .unwrap(),
            valid_tokens: vec![
                (0..length).map(|i| i < length - 1).collect(),
                vec![true; length],
            ],
            image_slots: vec![vec![false; length]; 2],
        };
        let positive = make_conditioning(3, 21);
        let negative = make_conditioning(5, 22);
        let mut positive_cache = transformer.prepare_t2i(&positive, 2, 2);
        let mut negative_cache = transformer.prepare_t2i(&negative, 2, 2);
        let mut first_keys = Vec::new();
        for (step, time) in [1.0, 0.7, 0.3, 0.01].iter().enumerate() {
            let latents =
                crate::engine::seeded_randn(23 + step as u64, &[2, 4, 4], device, DType::F32)
                    .unwrap();
            for (conditioning, cache) in [
                (&positive, &mut positive_cache),
                (&negative, &mut negative_cache),
            ] {
                let expected = transformer
                    .forward_t2i(&latents, *time, conditioning, 2, 2)
                    .unwrap();
                let actual = cache.forward(&latents, *time).unwrap();
                assert_close(&actual, &expected);
                assert_eq!(cache.layers.len(), 3);
                for layer in &cache.layers {
                    assert_eq!(
                        layer.key.dims(),
                        &[2, heads, conditioning.sequence_length(), 8]
                    );
                    assert!(layer.key.is_contiguous());
                    assert!(layer.value.is_contiguous());
                }
            }
            let keys: Vec<_> = positive_cache
                .layers
                .iter()
                .map(|layer| layer.key.id())
                .collect();
            if step == 0 {
                first_keys = keys;
            } else {
                assert_eq!(keys, first_keys);
            }
        }
        assert!(transformer.prepare_t2i(&positive, 2, 2).layers.is_empty());
        let invalid = Tensor::zeros((2, 5, 4), DType::F32, device).unwrap();
        let mut fresh = transformer.prepare_t2i(&positive, 2, 2);
        assert!(fresh.forward(&invalid, 1.0).is_err());
        assert!(fresh.layers.is_empty());
        assert!(positive_cache.forward(&invalid, 1.0).is_err());
    }

    #[test]
    fn prefix_cache_matches_full_forward_across_steps_and_cfg_branches() {
        for heads in [1, 2] {
            cache_parity(&Device::Cpu, heads);
        }
    }

    #[test]
    fn single_head_cache_owns_only_prefix_storage() {
        let transformer = tiny_transformer();
        let conditioning = QwenImage21TextConditioning {
            embeddings: Tensor::ones((1, 3, 8), DType::F32, &Device::Cpu).unwrap(),
            valid_tokens: vec![vec![true; 3]],
            image_slots: vec![vec![false; 3]],
        };
        let latents = Tensor::ones((1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let mut cache = transformer.prepare_t2i(&conditioning, 2, 2);
        cache.forward(&latents, 1.0).unwrap();
        for tensor in [&cache.layers[0].key, &cache.layers[0].value] {
            let (storage, layout) = tensor.storage_and_layout();
            let candle_core::Storage::Cpu(storage) = &*storage else {
                panic!("expected CPU")
            };
            assert_eq!(storage.as_slice::<f32>().unwrap().len(), 3 * 8);
            assert_eq!(layout.start_offset(), 0);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn prefix_cache_matches_full_forward_on_metal() {
        let device = crate::device::metal_device(0).unwrap();
        for heads in [1, 2] {
            cache_parity(&device, heads);
        }
    }

    #[test]
    fn long_prefix_falls_back_without_truncation_or_retention() {
        let transformer = tiny_transformer();
        let length = super::super::PREFIX_CACHE_MAX_TOKENS + 1;
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(21, &[1, length, 8], &Device::Cpu, DType::F32)
                .unwrap(),
            valid_tokens: vec![vec![true; length]],
            image_slots: vec![vec![false; length]],
        };
        let latents = Tensor::ones((1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let mut cache = transformer.prepare_t2i(&conditioning, 2, 2);
        assert_close(
            &cache.forward(&latents, 0.5).unwrap(),
            &transformer
                .forward_t2i(&latents, 0.5, &conditioning, 2, 2)
                .unwrap(),
        );
        assert!(cache.layers.is_empty());
    }

    #[test]
    fn prefix_cache_budget_is_additive_and_bounds_both_float32_cfg_branches() {
        use crate::device::{activation_bytes, ActivationFamily};
        let bound = super::super::prefix_cache_budget_bytes(1);
        assert_eq!(bound, 1_073_741_824);
        for size in [32, 1024, 2048] {
            for dtype_bytes in [2, 4] {
                for batch in [1, 2] {
                    let before = activation_bytes(
                        size,
                        size,
                        batch,
                        dtype_bytes,
                        ActivationFamily::QwenImageDit,
                    );
                    let after = activation_bytes(
                        size,
                        size,
                        batch,
                        dtype_bytes,
                        ActivationFamily::QwenImage21Dit,
                    );
                    assert_eq!(after - before, bound * u64::from(batch));
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn compact_modulation_is_exact_across_cached_steps() {
        let device = crate::device::metal_device(0).unwrap();
        let mut reference = tiny_transformer_on(tiny_config(), &device);
        reference.compact_modulation = false;
        let mut compact = tiny_transformer_on(tiny_config(), &device);
        compact.compact_modulation = true;
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(31, &[2, 3, 8], &device, DType::F32).unwrap(),
            valid_tokens: vec![vec![true, true, false], vec![true; 3]],
            image_slots: vec![vec![false; 3]; 2],
        };
        let mut expected = reference.prepare_t2i(&conditioning, 2, 2);
        let mut actual = compact.prepare_t2i(&conditioning, 2, 2);
        for time in [1.0, 0.7, 0.3] {
            let latents = crate::engine::seeded_randn(32, &[2, 4, 4], &device, DType::F32).unwrap();
            let diff = (actual.forward(&latents, time).unwrap()
                - expected.forward(&latents, time).unwrap())
            .unwrap();
            assert_eq!(
                diff.abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap(),
                0.0
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_projection_and_rope_preserve_cached_forward() {
        let device = crate::device::metal_device(0).unwrap();
        let mut cfg = tiny_config();
        cfg.num_layers = 3;
        let mut reference = tiny_transformer_on(cfg.clone(), &device);
        for block in &mut reference.blocks {
            block.attn.fused_ops = false;
        }
        let mut optimized = tiny_transformer_on(cfg, &device);
        for block in &mut optimized.blocks {
            block.attn.fused_ops = true;
        }
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(25, &[2, 3, 8], &device, DType::F32).unwrap(),
            valid_tokens: vec![vec![true, true, false], vec![true; 3]],
            image_slots: vec![vec![false; 3]; 2],
        };
        let mut expected = reference.prepare_t2i(&conditioning, 2, 2);
        let mut actual = optimized.prepare_t2i(&conditioning, 2, 2);
        for (step, time) in [1.0, 0.7, 0.3].into_iter().enumerate() {
            let latents =
                crate::engine::seeded_randn(26 + step as u64, &[2, 4, 4], &device, DType::F32)
                    .unwrap();
            assert_close(
                &actual.forward(&latents, time).unwrap(),
                &expected.forward(&latents, time).unwrap(),
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_target_attention_matches_math_with_rectangular_keys() {
        let device = crate::device::metal_device(0).unwrap();
        let mut transformer = tiny_transformer_on(tiny_config(), &device);
        let attn = &mut transformer.blocks[0].attn;
        attn.head_dim = 128;
        attn.fused_target = true;
        for dtype in [DType::F32, DType::BF16] {
            let q = crate::engine::seeded_randn(21, &[2, 2, 17, 128], &device, dtype).unwrap();
            let k = crate::engine::seeded_randn(22, &[2, 2, 29, 128], &device, dtype).unwrap();
            let v = crate::engine::seeded_randn(23, &[2, 2, 29, 128], &device, dtype).unwrap();
            let actual = attn.target_attention(&q, &k, &v, None).unwrap();
            attn.fused_target = false;
            let expected = attn.target_attention(&q, &k, &v, None).unwrap();
            attn.fused_target = true;
            let error = (actual.to_dtype(DType::F32).unwrap()
                - expected.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
            assert!(
                error < if dtype == DType::F32 { 1e-5 } else { 0.02 },
                "{dtype:?}: {error}"
            );
            // A real padded batch must preserve the math mask semantics.
            let bias = Tensor::from_vec(
                (0..58)
                    .map(|i| if i % 29 == 28 { f32::NEG_INFINITY } else { 0.0 })
                    .collect::<Vec<_>>(),
                (2, 1, 1, 29),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let actual = attn.target_attention(&q, &k, &v, Some(&bias)).unwrap();
            let expected =
                crate::attention::attention_with_bias(&q, &k, &v, 1.0 / 128f32.sqrt(), Some(&bias))
                    .unwrap();
            let error = (actual - expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(error, 0.0);
        }
    }

    /// Opt-in real-checkpoint parity and timing probe. No downloads or writes.
    /// QWEN_IMAGE21_MODEL_ROOT points at an existing mold models directory.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires installed Qwen Image 2.1 weights and an idle Metal GPU"]
    fn official_prefix_cache_metal_parity_and_timing() -> Result<()> {
        use crate::progress::ProgressReporter;
        use std::time::Instant;
        let root = PathBuf::from(std::env::var("QWEN_IMAGE21_MODEL_ROOT")?);
        let device = crate::device::metal_device(0)?;
        let progress = ProgressReporter::default();
        let shared = root.join("shared/qwen-image21");
        let text_paths = (1..=4)
            .map(|i| shared.join(format!("text_encoder/model-{i:05}-of-00004.safetensors")))
            .collect::<Vec<_>>();
        let mut encoder = crate::encoders::qwen3::Qwen3Encoder::load_bf16(
            &text_paths,
            &shared.join("processor/tokenizer.json"),
            &device,
            DType::F32,
            &crate::encoders::qwen3_bf16::Qwen3BF16Config::qwen3_image_21_text_encoder(),
            &progress,
        )?;
        let prompt = "A small red ceramic teapot on a sunlit wooden windowsill, editorial product photograph, soft morning shadows";
        let repeats =
            std::env::var("QWEN_IMAGE21_BENCH_REPEAT").map_or(Ok(1usize), |s| s.parse())?;
        let prompt = std::iter::repeat_n(prompt, repeats)
            .collect::<Vec<_>>()
            .join(". ");
        let conditioning = super::super::encode_t2i_prompts(&mut encoder, &[prompt])?;
        anyhow::ensure!(
            conditioning.sequence_length() <= super::super::PREFIX_CACHE_MAX_TOKENS,
            "benchmark prompt exceeds cache retention bound"
        );
        drop(encoder);
        let paths = (1..=2).map(|i| root.join(format!("qwen-image-2.1-bf16/transformer/diffusion_pytorch_model-{i:05}-of-00002.safetensors"))).collect::<Vec<_>>();
        let transformer = QwenImage21Transformer::load(&paths, &device, DType::F32, &progress)?;
        let mut cache = transformer.prepare_t2i(&conditioning, 64, 64);
        let mut scheduler = super::super::scheduler::QwenImage21Scheduler::new(
            40,
            4096,
            super::super::scheduler::QwenShiftPolicy::DynamicResolution,
        );
        let mut latents = crate::engine::seeded_randn(210001, &[1, 4096, 64], &device, DType::F32)?;
        let mut uncached_seconds = 0.0;
        let mut cached_seconds = 0.0;
        for step in 0..4 {
            let time = scheduler.current_timestep() / 1000.0;
            // Reverse the pair order on alternate steps to reduce order bias.
            let mut uncached = None;
            let mut cached = None;
            for cached_first in [step % 2 == 0, step % 2 != 0] {
                device.synchronize()?;
                let start = Instant::now();
                let output = if cached_first {
                    cache.forward(&latents, time)?
                } else {
                    transformer.forward_t2i(&latents, time, &conditioning, 64, 64)?
                };
                device.synchronize()?;
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("step={step} cached={cached_first} elapsed_seconds={elapsed:.4}");
                if step > 0 {
                    if cached_first {
                        cached_seconds += elapsed;
                    } else {
                        uncached_seconds += elapsed;
                    }
                }
                if cached_first {
                    cached = Some(output);
                } else {
                    uncached = Some(output);
                }
            }
            let expected = uncached.unwrap();
            let actual = cached.unwrap();
            let diff = (&actual - &expected)?;
            let max_error = diff.abs()?.max_all()?.to_scalar::<f32>()?;
            let relative_rms = (diff.sqr()?.mean_all()?.to_scalar::<f32>()?
                / expected.sqr()?.mean_all()?.to_scalar::<f32>()?)
            .sqrt();
            eprintln!("step={step} max_error={max_error} relative_rms={relative_rms}");
            anyhow::ensure!(
                max_error < 1e-3 && relative_rms < 1e-4,
                "cached real-checkpoint prediction diverged"
            );
            latents = scheduler.step(&expected, &latents)?;
        }
        eprintln!("prefix_tokens={} steady_uncached_seconds={uncached_seconds:.4} steady_cached_seconds={cached_seconds:.4} speedup={:.4}", conditioning.sequence_length(), uncached_seconds/cached_seconds);
        Ok(())
    }

    #[test]
    fn t2i_rope_centers_target_image_coordinates() {
        let transformer = tiny_transformer();
        let (cos, sin) = transformer
            .t2i_rope(3, 2, 2, DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(cos.dims(), &[7, 4]);
        assert_eq!(sin.dims(), &[7, 4]);
        // Target's first axis is frozen at the text cursor (3); h/w axes
        // differ across the target grid, so the complete rows must not all be
        // identical even though every target token shares its frame position.
        assert_ne!(
            cos.get(3).unwrap().to_vec1::<f32>().unwrap(),
            cos.get(4).unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_rope_keeps_f32_tables_for_bf16_latents() {
        let transformer = tiny_transformer();
        let device = crate::device::metal_device(0).unwrap();
        let (expected_cos, expected_sin) = transformer
            .t2i_rope(3, 2, 2, DType::F32, &Device::Cpu)
            .unwrap();
        let (cos, sin) = transformer.t2i_rope(3, 2, 2, DType::BF16, &device).unwrap();
        for (actual, expected) in [(cos, expected_cos), (sin, expected_sin)] {
            assert_eq!(actual.dtype(), DType::F32);
            assert_eq!(
                actual
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap(),
                expected.to_vec2::<f32>().unwrap()
            );
        }
        let (cos, sin) = transformer
            .t2i_rope(3, 2, 2, DType::BF16, &Device::Cpu)
            .unwrap();
        assert_eq!(cos.dtype(), DType::BF16);
        assert_eq!(sin.dtype(), DType::BF16);
    }

    #[test]
    fn tiny_t2i_forward_preserves_packed_latent_shape() {
        let transformer = tiny_transformer();
        let latents = Tensor::randn(0f32, 1.0, (2, 4, 4), &Device::Cpu).unwrap();
        let conditioning = QwenImage21TextConditioning {
            embeddings: Tensor::randn(0f32, 1.0, (2, 3, 8), &Device::Cpu).unwrap(),
            valid_tokens: vec![vec![true, true, false], vec![true; 3]],
            image_slots: vec![vec![false; 3], vec![false; 3]],
        };
        let output = transformer
            .forward_t2i(&latents, 1.0, &conditioning, 2, 2)
            .unwrap();
        assert_eq!(output.dims(), &[2, 4, 4]);
        assert!(output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn t2i_target_bias_only_masks_padded_text_keys() {
        let bias = Attention::t2i_target_bias(
            &[vec![true, false], vec![true, true]],
            4,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
        .unwrap();
        let rows = bias
            .squeeze(1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(rows[0][0], 0.0);
        assert_eq!(rows[0][1], f32::NEG_INFINITY);
        assert!(rows[0][2..].iter().all(|value| *value == 0.0));
        assert!(rows[1].iter().all(|value| *value == 0.0));
    }
}

#[cfg(all(test, feature = "metal"))]
mod performance_tests;
