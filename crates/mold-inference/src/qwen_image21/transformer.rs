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
use candle_nn::VarBuilder;
use std::path::PathBuf;

use super::attention::SegmentDispatch;
use super::exec_path::{Qwen21ExecPath, Qwen21RequestShape};
use super::layout::RopeAngles;
use super::layout::{BlockCausalPlan, QwenImage21JointLayout};
use super::linear::{Q21GateUp, Q21Linear, Q21Vb, Q21WeightSource};
use super::{PrefixCacheDecision, QwenImage21TextConditioning};

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

/// Every bias-free projection in the checkpoint is built here, and only here,
/// so swapping the linear representation (quantized storage, a LoRA adapter
/// slot) is one change rather than one per module.
///
/// The arm is the checkpoint's (`qwen_image21::linear`): the BF16 shards give
/// the Dense arm, bit-identical to `candle_nn::linear_no_bias`; GGUF, Comfy
/// INT8 ConvRot and torchao FP8 tiers give their quantized arms.
fn linear(in_dim: usize, out_dim: usize, vb: Q21Vb<'_>) -> Result<Q21Linear> {
    vb.linear(in_dim, out_dim)
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
    fn new(dim: usize, eps: f64, vb: Q21Vb<'_>) -> Result<Self> {
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
    in_layer: Q21Linear,
    out_layer: Q21Linear,
}

impl TextProjection {
    fn new(cfg: &QwenImage21TransformerConfig, vb: Q21Vb<'_>) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            text_norm: ZeroCenterRmsNorm::new(cfg.context_in_dim, cfg.eps, vb.pp("text_norm"))?,
            in_layer: linear(cfg.context_in_dim, inner, vb.pp("in_layer"))?,
            out_layer: linear(inner, inner, vb.pp("out_layer"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_layer.forward(
            &candle_nn::Activation::GeluPytorchTanh
                .forward(&self.in_layer.forward(&self.text_norm.forward(xs)?)?)?,
        )
    }
}

struct TimestepEmbedder {
    linear_1: Q21Linear,
    linear_2: Q21Linear,
    inner_dim: usize,
}

impl TimestepEmbedder {
    fn new(inner_dim: usize, vb: Q21Vb<'_>) -> Result<Self> {
        Ok(Self {
            linear_1: linear(256, inner_dim, vb.pp("linear_1"))?,
            linear_2: linear(inner_dim, inner_dim, vb.pp("linear_2"))?,
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
    /// `gate_layer` and `proj`, split (diffusers) or fused gate-first
    /// (ComfyUI / GGUF) as the checkpoint stores them.
    gate_up: Q21GateUp,
    out: Q21Linear,
}

impl SwiGlu {
    fn new(dim: usize, mlp_dim: usize, vb: Q21Vb<'_>) -> Result<Self> {
        Ok(Self {
            gate_up: vb.gate_up(dim, mlp_dim)?,
            out: linear(mlp_dim, dim, vb.pp("out"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // `silu(gate_layer(x)) * proj(x)` (`transformer_qwenimage21.py:212`).
        self.out.forward(&self.gate_up.swiglu(xs)?)
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
    dispatch: SegmentDispatch,
    fused_ops: bool,
    to_q: Q21Linear,
    to_k: Q21Linear,
    to_v: Q21Linear,
    to_out: Q21Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn new(cfg: &QwenImage21TransformerConfig, vb: Q21Vb<'_>) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            dispatch: SegmentDispatch {
                attention: super::exec_path::Qwen21ExecPath::metal(
                    crate::attention::metal_fast_path_enabled(),
                )
                .attention,
                head_dim: cfg.attention_head_dim,
            },
            fused_ops: crate::attention::metal_fast_path_enabled(),
            to_q: linear(inner, inner, vb.pp("to_q"))?,
            to_k: linear(inner, inner, vb.pp("to_k"))?,
            to_v: linear(inner, inner, vb.pp("to_v"))?,
            to_out: linear(inner, inner, vb.pp("to_out").pp("0"))?,
            norm_q: vb.pp("norm_q").get(cfg.attention_head_dim, "weight")?,
            norm_k: vb.pp("norm_k").get(cfg.attention_head_dim, "weight")?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.eps,
        })
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

    /// One attention layer over a joint sequence (or, on a cached step, over
    /// the target block alone) under the block-causal `plan`.
    ///
    /// `prefix_len` is the number of leading joint positions a prefill
    /// retains; on a cached step `hidden_states` holds only the target block
    /// and the retained prefix K/V is prepended before `plan` runs.
    fn forward_block_causal(
        &self,
        hidden_states: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        plan: &BlockCausalPlan,
        prefix_len: usize,
        cache: LayerCache<'_>,
    ) -> Result<Tensor> {
        let (batch, sequence, inner) = hidden_states.dims3()?;
        anyhow::ensure!(
            inner == self.heads * self.head_dim,
            "Qwen Image 2.1 attention inner width mismatch"
        );

        let (q, k, v) = if self.fused_ops {
            // Normalize while BSHD is contiguous; flattening BHSD first copies
            // the whole projection. RoPE still accumulates in F32 as upstream
            // (`apply_rotary_emb_qwen` rotates in float32). Metal's shipped
            // path and the CUDA fast path (`Qwen21ExecPath::fused_projection`).
            // `mold_candle::qk_norm_rope` IS this composite (rms_norm, BHSD
            // transpose, F32 rope_i, narrow) off CUDA, and one fused kernel,
            // bitwise the composite, on CUDA.
            let (cos, sin) = (
                rope_cos.to_dtype(DType::F32)?,
                rope_sin.to_dtype(DType::F32)?,
            );
            let project = |linear: &Q21Linear, weight: &Tensor| -> Result<Tensor> {
                let xs = linear.forward(hidden_states)?.reshape((
                    batch,
                    sequence,
                    self.heads,
                    self.head_dim,
                ))?;
                let weight = if weight.dtype() == xs.dtype() {
                    weight.clone()
                } else {
                    weight.to_dtype(xs.dtype())?
                };
                Ok(mold_candle::qk_norm_rope::rms_norm_rope_i(
                    &xs,
                    &weight,
                    &cos,
                    &sin,
                    self.eps as f32,
                )?)
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
            let project = |linear: &Q21Linear| -> Result<Tensor> {
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

        let (k, v) = match cache {
            LayerCache::Extract(layers) => {
                // A contiguous view can still retain the full joint allocation
                // (notably with one head). Copy only the immutable prefix
                // (`transformer_qwenimage21.py:340-349` clones for the same
                // reason).
                layers.push(PrefixKv {
                    key: k.narrow(2, 0, prefix_len)?.force_contiguous()?,
                    value: v.narrow(2, 0, prefix_len)?.force_contiguous()?,
                });
                (k, v)
            }
            LayerCache::Reuse(prefix) => (
                Tensor::cat(&[&prefix.key, &k], 2)?,
                Tensor::cat(&[&prefix.value, &v], 2)?,
            ),
            LayerCache::Disabled => (k, v),
        };
        let context = self.dispatch.attend(&q, &k, &v, plan)?;
        self.to_out
            .forward(&context.transpose(1, 2)?.reshape((batch, sequence, inner))?)
    }
}

struct TransformerBlock {
    norm1: LayerNormNoParams,
    attn: Attention,
    norm2: LayerNormNoParams,
    mlp: SwiGlu,
    /// `Qwen21ExecPath::fused_adaln`: fold each LayerNorm and its `1 + scale`
    /// into `candle_nn::ops::layer_norm(x, 1 + scale, 0)` on compact rows.
    fused_adaln: bool,
    eps: f64,
}

/// One timestep's modulation, precomputed ONCE per forward for all blocks
/// (every block reads the same `modulation` output): `1 + scale` and
/// `tanh(gate)` for both halves, each `[B, 1, D]`. Upstream's
/// `_modulated_norm` / `_gated_residual` apply the same arithmetic per
/// block; hoisting it is value-identical.
struct ModRow {
    scale1: Tensor,
    gate1: Tensor,
    scale2: Tensor,
    gate2: Tensor,
}

impl ModRow {
    /// `row` is `[B, 1, 4D]` in checkpoint order `(scale1, gate1, scale2,
    /// gate2)`.
    fn new(row: &Tensor, dim: usize) -> Result<Self> {
        Ok(Self {
            scale1: (row.narrow(D::Minus1, 0, dim)? + 1.0)?,
            gate1: row.narrow(D::Minus1, dim, dim)?.tanh()?,
            scale2: (row.narrow(D::Minus1, 2 * dim, dim)? + 1.0)?,
            gate2: row.narrow(D::Minus1, 3 * dim, dim)?.tanh()?,
        })
    }
}

/// How a block reads its modulation.
enum BlockModulation {
    /// `[B, N, 4D]` (or a broadcastable `[B, 1, 4D]`): scale and tanh are
    /// evaluated per block. v0.32's arithmetic, and Metal's shipped path.
    PerToken(Tensor),
    /// Compact rows, one per contiguous run of the block's hidden rows:
    /// `[(prefix_len, t=0 row), (target, real row)]` on a prefill, one real
    /// row on a cached step.
    Rows(Vec<(usize, ModRow)>),
}

impl TransformerBlock {
    fn new(cfg: &QwenImage21TransformerConfig, vb: Q21Vb<'_>) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            norm1: LayerNormNoParams::new(cfg.eps),
            attn: Attention::new(cfg, vb.pp("attn"))?,
            norm2: LayerNormNoParams::new(cfg.eps),
            mlp: SwiGlu::new(inner, inner * cfg.mlp_ratio, vb.pp("img_mlp"))?,
            fused_adaln: false,
            eps: cfg.eps,
        })
    }

    /// `norm(x) * (1 + scale)` over compact rows. With `fused_adaln` each run
    /// is one fused LayerNorm whose affine weight is the row: every batch row
    /// of a forward shares its timestep, so row 0 is the row of all of them.
    /// Without it, the hand-written F32 LayerNorm then a broadcast multiply.
    fn modulated_norm(
        &self,
        norm: &LayerNormNoParams,
        xs: &Tensor,
        rows: &[(usize, ModRow)],
        scale: fn(&ModRow) -> &Tensor,
    ) -> Result<Tensor> {
        let mut parts = Vec::with_capacity(rows.len());
        let mut start = 0;
        for (len, row) in rows {
            let x = if rows.len() == 1 {
                xs.clone()
            } else {
                xs.narrow(1, start, *len)?
            };
            start += len;
            let scale = scale(row);
            parts.push(if self.fused_adaln {
                let alpha = scale.get(0)?.flatten_all()?.contiguous()?;
                let beta = alpha.zeros_like()?;
                candle_nn::ops::layer_norm(&x.contiguous()?, &alpha, &beta, self.eps as f32)?
            } else {
                norm.forward(&x)?.broadcast_mul(scale)?
            });
        }
        concat_rows(parts)
    }

    /// `xs + tanh(gate) * ys` over compact rows.
    fn gated_residual(
        xs: &Tensor,
        ys: &Tensor,
        rows: &[(usize, ModRow)],
        gate: fn(&ModRow) -> &Tensor,
    ) -> Result<Tensor> {
        let mut parts = Vec::with_capacity(rows.len());
        let mut start = 0;
        for (len, row) in rows {
            let y = if rows.len() == 1 {
                ys.clone()
            } else {
                ys.narrow(1, start, *len)?
            };
            start += len;
            parts.push(gate(row).broadcast_mul(&y)?);
        }
        Ok((xs + concat_rows(parts)?)?)
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

    #[allow(clippy::too_many_arguments)]
    fn forward_block_causal(
        &self,
        hidden_states: &Tensor,
        modulation: &BlockModulation,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        plan: &BlockCausalPlan,
        prefix_len: usize,
        cache: LayerCache<'_>,
    ) -> Result<Tensor> {
        let modulation = match modulation {
            BlockModulation::PerToken(modulation) => modulation,
            BlockModulation::Rows(rows) => {
                let normalized =
                    self.modulated_norm(&self.norm1, hidden_states, rows, |row| &row.scale1)?;
                let attn = self.attn.forward_block_causal(
                    &normalized,
                    rope_cos,
                    rope_sin,
                    plan,
                    prefix_len,
                    cache,
                )?;
                let hidden_states =
                    Self::gated_residual(hidden_states, &attn, rows, |row| &row.gate1)?;
                let normalized =
                    self.modulated_norm(&self.norm2, &hidden_states, rows, |row| &row.scale2)?;
                let mlp = self.mlp.forward(&normalized)?;
                let hidden_states =
                    Self::gated_residual(&hidden_states, &mlp, rows, |row| &row.gate2)?;
                if hidden_states.dtype() == DType::F16 {
                    return hidden_states
                        .clamp(-65_504.0f32, 65_504.0f32)
                        .map_err(Into::into);
                }
                return Ok(hidden_states);
            }
        };
        let dim = hidden_states.dim(D::Minus1)?;
        anyhow::ensure!(
            modulation.dim(D::Minus1)? == 4 * dim,
            "Qwen Image 2.1 modulation width does not match hidden width"
        );
        let mod1 = modulation.narrow(D::Minus1, 0, 2 * dim)?;
        let mod2 = modulation.narrow(D::Minus1, 2 * dim, 2 * dim)?;

        let (normalized, gate) = Self::modulate(self.norm1.forward(hidden_states)?, &mod1)?;
        let attn = self.attn.forward_block_causal(
            &normalized,
            rope_cos,
            rope_sin,
            plan,
            prefix_len,
            cache,
        )?;
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

/// Concatenate per-run results along the sequence axis (a single run passes
/// through untouched).
fn concat_rows(mut parts: Vec<Tensor>) -> Result<Tensor> {
    if parts.len() == 1 {
        return Ok(parts.pop().expect("one part"));
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

struct AdaFinalNorm {
    norm: LayerNormNoParams,
    linear: Q21Linear,
}

impl AdaFinalNorm {
    fn new(dim: usize, eps: f64, vb: Q21Vb<'_>) -> Result<Self> {
        Ok(Self {
            norm: LayerNormNoParams::new(eps),
            linear: linear(dim, dim, vb.pp("linear"))?,
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
/// shards. It runs any [`QwenImage21JointLayout`]: text-to-image is the layout
/// with no condition images, and a reference-conditioned request lays its
/// condition latents into the text stream where the Qwen3-VL image slots were.
pub(crate) struct QwenImage21Transformer {
    exec: Qwen21ExecPath,
    /// The checkpoint's tier, named by the per-step finiteness guard.
    tier: String,
    /// Whether a GGUF linear may be on candle's QMatMul arm.
    qmatmul_guard: bool,
    cfg: QwenImage21TransformerConfig,
    img_in: Q21Linear,
    time_text_embed: TimestepEmbedder,
    txt_in: TextProjection,
    modulation: Q21Linear,
    blocks: Vec<TransformerBlock>,
    norm_out: AdaFinalNorm,
    proj_out: Q21Linear,
}

/// Rotary tables of one branch, built once per request rather than on every
/// step: the full joint sequence for the prefill, and the target rows for
/// cached steps.
struct BranchRope {
    dtype: DType,
    angles: RopeAngles,
    full: (Tensor, Tensor),
    target: (Tensor, Tensor),
}

/// One conditioning branch of one denoise request: its layout, its condition
/// latents (shared by both CFG branches), and its prefix-cache decision. The
/// borrow ties the cache to its exact transformer and immutable prompt;
/// nothing survives the request.
pub(crate) struct PreparedBranch<'a> {
    transformer: &'a QwenImage21Transformer,
    conditioning: &'a QwenImage21TextConditioning,
    layout: QwenImage21JointLayout,
    cond_latents: Option<Tensor>,
    cache_policy: PrefixCacheDecision,
    /// The request this branch renders, for the rounding boundaries the exec
    /// path decides per request ([`Qwen21ExecPath::rope_angles`]).
    request: Qwen21RequestShape,
    rope: Option<BranchRope>,
    layers: Vec<PrefixKv>,
}

impl PreparedBranch<'_> {
    /// Render this branch for `request` (the pipeline's
    /// [`Qwen21RequestShape::of`]). Without it a branch assumes v0.32 bytes
    /// exist exactly when it has no condition images.
    pub(crate) fn for_request(mut self, request: Qwen21RequestShape) -> Self {
        self.request = request;
        self.rope = None;
        self
    }

    #[cfg(test)]
    pub(crate) fn layout(&self) -> &QwenImage21JointLayout {
        &self.layout
    }

    /// Predict the flow for `latents` `[B, target_tokens, 64]` at normalized
    /// `timestep`.
    pub(crate) fn forward(&mut self, latents: &Tensor, timestep: f64) -> Result<Tensor> {
        ensure_branch_rope(
            &mut self.rope,
            &self.layout,
            self.transformer.cfg.axes_dims_rope,
            self.transformer.exec.rope_angles(self.request),
            self.transformer
                .rope_table_dtype(latents.dtype(), &self.layout),
            latents.device(),
        )?;
        let rope = self.rope.as_ref().expect("rope tables were just built");
        let forward = |cache| {
            self.transformer.forward_layout(
                latents,
                self.cond_latents.as_ref(),
                timestep,
                self.conditioning,
                &self.layout,
                rope,
                cache,
            )
        };
        if self.cache_policy == PrefixCacheDecision::Recompute {
            return forward(PrefixCache::Disabled);
        }
        if self.layers.is_empty() {
            let mut layers = Vec::with_capacity(self.transformer.blocks.len());
            let result = forward(PrefixCache::Extract(&mut layers))?;
            // Publish only a complete, successful prefill.
            self.layers = layers;
            Ok(result)
        } else {
            let key = &self.layers[0].key;
            anyhow::ensure!(
                key.dtype() == latents.dtype() && key.device().same_device(latents.device()),
                "Qwen Image 2.1 cached denoise cannot change device or dtype"
            );
            forward(PrefixCache::Reuse(&self.layers))
        }
    }
}

/// Build (or rebuild, when the dtype or device changed) a branch's rotary
/// tables from its layout. Caching them is value-identical: the tables are a
/// pure function of the layout.
fn ensure_branch_rope(
    rope: &mut Option<BranchRope>,
    layout: &QwenImage21JointLayout,
    axes_dims: [usize; 3],
    angles: RopeAngles,
    dtype: DType,
    device: &Device,
) -> Result<()> {
    let stale = rope.as_ref().is_none_or(|rope| {
        rope.dtype != dtype || rope.angles != angles || !rope.full.0.device().same_device(device)
    });
    if !stale {
        return Ok(());
    }
    let (cos, sin) =
        QwenImage21JointLayout::rope_tables(layout.rope(), axes_dims, angles, dtype, device)?;
    let prefix = layout.prefix_len();
    let target = layout.target_tokens();
    let target_tables = (
        cos.narrow(0, prefix, target)?.contiguous()?,
        sin.narrow(0, prefix, target)?.contiguous()?,
    );
    *rope = Some(BranchRope {
        dtype,
        angles,
        full: (cos, sin),
        target: target_tables,
    });
    Ok(())
}

impl QwenImage21Transformer {
    /// Prepare one branch of a request.
    ///
    /// `cond_latents` are the packed, normalized condition-image latents
    /// `[B, Σ h·w, 64]` in caller order (`pipeline_qwenimage21.py:476`); they
    /// must be present exactly when the layout has condition images.
    pub(crate) fn prepare<'a>(
        &'a self,
        conditioning: &'a QwenImage21TextConditioning,
        layout: QwenImage21JointLayout,
        cond_latents: Option<Tensor>,
        cache_policy: PrefixCacheDecision,
    ) -> Result<PreparedBranch<'a>> {
        anyhow::ensure!(
            layout.text_len() == conditioning.sequence_length(),
            "Qwen Image 2.1 layout covers {} text rows but the conditioning has {}",
            layout.text_len(),
            conditioning.sequence_length()
        );
        match &cond_latents {
            Some(latents) => {
                let (batch, tokens, channels) = latents.dims3()?;
                anyhow::ensure!(
                    batch == conditioning.batch_size()
                        && tokens == layout.condition_tokens()
                        && channels == self.cfg.in_channels,
                    "Qwen Image 2.1 condition latents {:?} do not match the layout's {} tokens",
                    latents.dims(),
                    layout.condition_tokens()
                );
            }
            None => anyhow::ensure!(
                layout.condition_tokens() == 0,
                "Qwen Image 2.1 layout has condition images but no condition latents"
            ),
        }
        let layout_has_no_condition = layout.condition_tokens() == 0;
        Ok(PreparedBranch {
            transformer: self,
            conditioning,
            layout,
            cond_latents,
            cache_policy,
            request: Qwen21RequestShape {
                has_v032_bytes: layout_has_no_condition,
            },
            rope: None,
            layers: Vec::new(),
        })
    }

    /// Prepare a text-to-image branch.
    #[cfg(test)]
    pub(crate) fn prepare_t2i<'a>(
        &'a self,
        conditioning: &'a QwenImage21TextConditioning,
        latent_height: usize,
        latent_width: usize,
        cache_policy: PrefixCacheDecision,
    ) -> Result<PreparedBranch<'a>> {
        anyhow::ensure!(
            conditioning
                .image_slots
                .iter()
                .all(|row| row.iter().all(|slot| !*slot)),
            "Qwen Image 2.1 conditioning carries image slots; prepare it with its reference layout"
        );
        let layout = QwenImage21JointLayout::text_to_image(
            &conditioning.valid_tokens,
            (latent_height, latent_width),
        )?;
        self.prepare(conditioning, layout, None, cache_policy)
    }

    /// Load any published tier. The format comes from the checkpoint header
    /// (`artifact_format::probe_qwen_image21_transformer`): the BF16 shards,
    /// Comfy INT8 ConvRot, torchao FP8, or a GGUF (leejet or unsloth).
    pub(crate) fn load(
        paths: &[PathBuf],
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let source = Q21WeightSource::open(
            paths,
            device,
            dtype,
            super::linear::qmatmul_enabled(),
            progress,
        )?;
        let transformer = Self::from_source(QwenImage21TransformerConfig::official(), &source)?;
        progress.info(&format!(
            "Qwen Image 2.1 transformer tier: {} (block linears: {:?})",
            source.tier_label(),
            transformer.blocks[0].attn.to_q.kind()
        ));
        Ok(transformer)
    }

    /// Move every weight to `device` in place — the transformer's park to
    /// host RAM for a 2K VAE decode and its restore afterwards
    /// (`text_encoder_residency::TransformerDecode::ParkHost`). No reload
    /// from disk: quantized storage makes a byte-exact round trip.
    /// Install one bypass stack per linear from `registry`, keyed by the
    /// checkpoint's own `<module>.weight` names (`qwen_image21::lora` maps a
    /// LoRA onto them); every linear the registry does not name is cleared.
    /// `None` clears every adapter. Base weights are never touched, so a new
    /// request's LoRA set is an adapter swap, never a rebuild.
    pub(crate) fn install_lora(
        &mut self,
        registry: Option<&crate::flux::lora_bypass::LoraRegistry>,
    ) -> Result<()> {
        let stack = |module: &str| -> Vec<crate::flux::lora_bypass::LinearLoraAdapter> {
            registry
                .map(|registry| registry.adapters_for(&format!("{module}.weight")).to_vec())
                .unwrap_or_default()
        };
        self.img_in.set_adapters(stack("img_in"))?;
        self.txt_in
            .in_layer
            .set_adapters(stack("txt_in.in_layer"))?;
        self.txt_in
            .out_layer
            .set_adapters(stack("txt_in.out_layer"))?;
        self.time_text_embed
            .linear_1
            .set_adapters(stack("time_text_embed.timestep_embedder.linear_1"))?;
        self.time_text_embed
            .linear_2
            .set_adapters(stack("time_text_embed.timestep_embedder.linear_2"))?;
        self.modulation.set_adapters(stack("modulation.1"))?;
        self.norm_out
            .linear
            .set_adapters(stack("norm_out.linear"))?;
        self.proj_out.set_adapters(stack("proj_out"))?;
        for (index, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("transformer_blocks.{index}");
            block
                .attn
                .to_q
                .set_adapters(stack(&format!("{prefix}.attn.to_q")))?;
            block
                .attn
                .to_k
                .set_adapters(stack(&format!("{prefix}.attn.to_k")))?;
            block
                .attn
                .to_v
                .set_adapters(stack(&format!("{prefix}.attn.to_v")))?;
            block
                .attn
                .to_out
                .set_adapters(stack(&format!("{prefix}.attn.to_out.0")))?;
            block.mlp.gate_up.set_adapters(
                stack(&format!("{prefix}.img_mlp.gate_layer")),
                stack(&format!("{prefix}.img_mlp.proj")),
            )?;
            block
                .mlp
                .out
                .set_adapters(stack(&format!("{prefix}.img_mlp.out")))?;
        }
        Ok(())
    }
    pub(crate) fn move_to_device(&mut self, device: &Device) -> Result<()> {
        self.img_in = self.img_in.to_device(device)?;
        self.time_text_embed.linear_1 = self.time_text_embed.linear_1.to_device(device)?;
        self.time_text_embed.linear_2 = self.time_text_embed.linear_2.to_device(device)?;
        self.txt_in.text_norm.weight = self.txt_in.text_norm.weight.to_device(device)?;
        self.txt_in.in_layer = self.txt_in.in_layer.to_device(device)?;
        self.txt_in.out_layer = self.txt_in.out_layer.to_device(device)?;
        self.modulation = self.modulation.to_device(device)?;
        for block in &mut self.blocks {
            let attn = &mut block.attn;
            attn.to_q = attn.to_q.to_device(device)?;
            attn.to_k = attn.to_k.to_device(device)?;
            attn.to_v = attn.to_v.to_device(device)?;
            attn.to_out = attn.to_out.to_device(device)?;
            attn.norm_q = attn.norm_q.to_device(device)?;
            attn.norm_k = attn.norm_k.to_device(device)?;
            block.mlp.gate_up = block.mlp.gate_up.to_device(device)?;
            block.mlp.out = block.mlp.out.to_device(device)?;
        }
        self.norm_out.linear = self.norm_out.linear.to_device(device)?;
        self.proj_out = self.proj_out.to_device(device)?;
        Ok(())
    }

    /// Fail a denoise step whose prediction is not finite, naming this
    /// checkpoint's tier and whether the QMatMul switch shaped it.
    pub(crate) fn ensure_finite(&self, prediction: &Tensor, step: usize) -> Result<()> {
        super::linear::ensure_finite_prediction(prediction, step, &self.tier, self.qmatmul_guard)
    }

    /// Build from a dense `VarBuilder` (synthetic tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_var_builder(
        cfg: QwenImage21TransformerConfig,
        vb: VarBuilder<'static>,
    ) -> Result<Self> {
        Self::from_source(cfg, &Q21WeightSource::from_var_builder(vb))
    }

    pub(crate) fn from_source(
        cfg: QwenImage21TransformerConfig,
        source: &Q21WeightSource,
    ) -> Result<Self> {
        let vb = source.root();
        cfg.validate()?;
        let inner = cfg.inner_dim();
        let img_in = linear(cfg.in_channels, inner, vb.pp("img_in"))?;
        let time_text_embed =
            TimestepEmbedder::new(inner, vb.pp("time_text_embed").pp("timestep_embedder"))?;
        let txt_in = TextProjection::new(&cfg, vb.pp("txt_in"))?;
        let modulation = linear(inner, 4 * inner, vb.pp("modulation").pp("1"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for index in 0..cfg.num_layers {
            blocks.push(TransformerBlock::new(
                &cfg,
                vb.pp("transformer_blocks").pp(index),
            )?);
        }
        let norm_out = AdaFinalNorm::new(inner, cfg.eps, vb.pp("norm_out"))?;
        let proj_out = linear(inner, cfg.out_channels, vb.pp("proj_out"))?;
        // Resolved from where the weights landed (the text norm is always a
        // dense tensor on the load device).
        let exec = Qwen21ExecPath::resolve(txt_in.text_norm.weight.device());
        let mut transformer = Self {
            exec,
            tier: source.tier_label(),
            qmatmul_guard: source.qmatmul_guard(),
            cfg,
            img_in,
            time_text_embed,
            txt_in,
            modulation,
            blocks,
            norm_out,
            proj_out,
        };
        transformer.set_exec_path(exec);
        Ok(transformer)
    }

    /// The execution path this transformer runs.
    pub(crate) fn exec_path(&self) -> Qwen21ExecPath {
        self.exec
    }

    /// Put an execution path into effect on every block. The engine resolves
    /// it once at load; tests and the CUDA harness build any combination.
    pub(crate) fn set_exec_path(&mut self, exec: Qwen21ExecPath) {
        self.exec = exec;
        for block in &mut self.blocks {
            block.attn.fused_ops = exec.fused_projection;
            block.attn.dispatch.attention = exec.attention;
            block.fused_adaln = exec.fused_adaln;
        }
    }

    /// The dtype of this transformer's rotary tables for `latent_dtype`
    /// working storage: F32 on the fast path (and always on Metal, which
    /// `rope_tables` itself enforces), the working dtype on the v0.32 path —
    /// except for a layout with condition images, which v0.32 never rendered
    /// and so has no bytes to keep: upstream's tables are float32 there too
    /// (`transformer_qwenimage21.py:673-675`).
    fn rope_table_dtype(&self, latent_dtype: DType, layout: &QwenImage21JointLayout) -> DType {
        if self.exec.f32_rope_tables || layout.condition_tokens() > 0 {
            DType::F32
        } else {
            latent_dtype
        }
    }

    /// Assemble the joint hidden states (`transformer_qwenimage21.py:905-923`):
    /// `txt_in` over the Qwen3-VL rows, `img_in` over `[cond…, target]`, and
    /// one `index_select` that drops each image slot's text row and lays the
    /// latents in its place. Text-to-image's gather is the identity, which is
    /// exactly the concatenation it has always been.
    fn assemble_joint(
        &self,
        text: &Tensor,
        cond_latents: Option<&Tensor>,
        latents: &Tensor,
        layout: &QwenImage21JointLayout,
    ) -> Result<Tensor> {
        let text = self.txt_in.forward(text)?;
        let images = match cond_latents {
            Some(cond) => self.img_in.forward(&Tensor::cat(
                &[
                    &cond
                        .to_device(latents.device())?
                        .to_dtype(latents.dtype())?,
                    latents,
                ],
                1,
            )?)?,
            None => self.img_in.forward(latents)?,
        };
        let joint = Tensor::cat(&[&text, &images], 1)?;
        if layout.gather_is_identity() {
            return Ok(joint);
        }
        let index = Tensor::from_slice(
            layout.gather_index(),
            layout.gather_index().len(),
            latents.device(),
        )?;
        Ok(joint.index_select(&index, 1)?)
    }

    /// Denoise a packed `[B, target_tokens, 64]` latent under `layout`.
    ///
    /// `timestep` is normalized to `[0, 1]`, matching the Diffusers
    /// transformer's call (`scheduler_timestep / 1000`). On a cached step
    /// (`PrefixCache::Reuse`) only the target block runs through the blocks,
    /// against the retained prefix K/V.
    #[allow(clippy::too_many_arguments)]
    fn forward_layout(
        &self,
        latents: &Tensor,
        cond_latents: Option<&Tensor>,
        timestep: f64,
        conditioning: &QwenImage21TextConditioning,
        layout: &QwenImage21JointLayout,
        rope: &BranchRope,
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
            target_tokens == layout.target_tokens(),
            "Qwen Image 2.1 packed latent length {target_tokens} does not match the layout's {} target tokens",
            layout.target_tokens()
        );
        anyhow::ensure!(
            conditioning.batch_size() == batch,
            "Qwen Image 2.1 text batch {} does not match latent batch {batch}",
            conditioning.batch_size()
        );
        let text_len = conditioning.sequence_length();
        anyhow::ensure!(
            text_len == layout.text_len(),
            "Qwen Image 2.1 text conditioning has {text_len} rows but the layout reads {}",
            layout.text_len()
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

        let prefix_len = layout.prefix_len();
        let mut hidden_states = if cached {
            self.img_in.forward(latents)?
        } else {
            self.assemble_joint(&text, cond_latents, latents, layout)?
        };
        let (rope_cos, rope_sin) = if cached { &rope.target } else { &rope.full };
        let plan = layout.attention_plan(cached);

        // `causal_condition` (`transformer_qwenimage21.py:238-254, 926-937`):
        // every prefix position — text AND condition image — takes a
        // dedicated t=0 modulation row, while target positions take each
        // sample's real timestep. The target is always the last block, so the
        // per-token rows are `[t=0 prefix; real target]`.
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
        let zero_row = modulation
            .narrow(0, batch, 1)?
            .unsqueeze(1)?
            .broadcast_as((batch, 1, 4 * inner))?;
        let per_token_modulation = if self.exec.fused_adaln
            || (self.exec.compact_modulation && !latents.device().is_metal())
        {
            // The CUDA fast path: scale and tanh once per forward, compact
            // `[B, 1, D]` rows applied per run (ComfyUI's `_modulated_norm` /
            // `_gated_residual` on narrowed views).
            let real = ModRow::new(&real_row, inner)?;
            BlockModulation::Rows(if cached {
                vec![(target_tokens, real)]
            } else {
                vec![
                    (prefix_len, ModRow::new(&zero_row.contiguous()?, inner)?),
                    (target_tokens, real),
                ]
            })
        } else if cached {
            // Every target position shares a timestep. Metal keeps the row
            // compact so each block's scale and tanh execute once per
            // feature rather than once per image token; v0.32 broadcasts.
            BlockModulation::PerToken(if self.exec.compact_modulation {
                real_row
            } else {
                real_row.broadcast_as((batch, target_tokens, 4 * inner))?
            })
        } else {
            BlockModulation::PerToken(Tensor::cat(
                &[
                    &zero_row.broadcast_as((batch, prefix_len, 4 * inner))?,
                    &real_row.broadcast_as((batch, target_tokens, 4 * inner))?,
                ],
                1,
            )?)
        };

        for (index, block) in self.blocks.iter().enumerate() {
            let layer_cache = match &mut cache {
                PrefixCache::Extract(layers) => LayerCache::Extract(layers),
                PrefixCache::Reuse(layers) => LayerCache::Reuse(&layers[index]),
                PrefixCache::Disabled => LayerCache::Disabled,
            };
            hidden_states = block.forward_block_causal(
                &hidden_states,
                &per_token_modulation,
                rope_cos,
                rope_sin,
                &plan,
                prefix_len,
                layer_cache,
            )?;
        }
        // Only the target rows are kept (`pipeline_qwenimage21.py:784`), so
        // `norm_out`/`proj_out` run on the target alone.
        let target_hidden =
            hidden_states.narrow(1, if cached { 0 } else { prefix_len }, target_tokens)?;
        let target_temb = temb.narrow(0, 0, batch)?;
        self.proj_out
            .forward(&self.norm_out.forward(&target_hidden, &target_temb)?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    impl QwenImage21Transformer {
        /// Uncached text-to-image forward through the layout path.
        fn forward_t2i(
            &self,
            latents: &Tensor,
            timestep: f64,
            conditioning: &QwenImage21TextConditioning,
            latent_height: usize,
            latent_width: usize,
        ) -> Result<Tensor> {
            let layout = QwenImage21JointLayout::text_to_image(
                &conditioning.valid_tokens,
                (latent_height, latent_width),
            )?;
            self.forward_uncached(latents, None, timestep, conditioning, &layout)
        }

        /// Uncached forward of any layout.
        fn forward_uncached(
            &self,
            latents: &Tensor,
            cond_latents: Option<&Tensor>,
            timestep: f64,
            conditioning: &QwenImage21TextConditioning,
            layout: &QwenImage21JointLayout,
        ) -> Result<Tensor> {
            let mut rope = None;
            // `prepare`'s default request: v0.32 bytes exist exactly when the
            // layout has no condition images.
            let request = Qwen21RequestShape {
                has_v032_bytes: layout.condition_tokens() == 0,
            };
            ensure_branch_rope(
                &mut rope,
                layout,
                self.cfg.axes_dims_rope,
                self.exec.rope_angles(request),
                self.rope_table_dtype(latents.dtype(), layout),
                latents.device(),
            )?;
            self.forward_layout(
                latents,
                cond_latents,
                timestep,
                conditioning,
                layout,
                rope.as_ref().unwrap(),
                PrefixCache::Disabled,
            )
        }
    }

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
        tiny_transformer_dtype(cfg, device, DType::F32)
    }

    fn tiny_transformer_dtype(
        cfg: QwenImage21TransformerConfig,
        device: &Device,
        dtype: DType,
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
        let vb = VarBuilder::from_tensors(map, dtype, device);
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

    fn max_relative_error(actual: &Tensor, expected: &Tensor) -> f32 {
        let actual = actual.to_dtype(DType::F32).unwrap();
        let expected = expected.to_dtype(DType::F32).unwrap();
        let error = (&actual - &expected)
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
        error / peak.max(f32::MIN_POSITIVE)
    }

    /// Run `candidate` and `reference` through the same prefill and cached
    /// steps (a padded CFG-shaped batch) and hand every prediction pair to
    /// `check`.
    fn compare_paths(
        reference: &QwenImage21Transformer,
        candidate: &QwenImage21Transformer,
        device: &Device,
        dtype: DType,
        check: impl Fn(&Tensor, &Tensor, f64),
    ) {
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(41, &[2, 5, 8], device, dtype).unwrap(),
            valid_tokens: vec![vec![true, true, true, true, false], vec![true; 5]],
            image_slots: vec![vec![false; 5]; 2],
        };
        let mut expected = reference
            .prepare_t2i(&conditioning, 2, 4, PrefixCacheDecision::Retain)
            .unwrap();
        let mut actual = candidate
            .prepare_t2i(&conditioning, 2, 4, PrefixCacheDecision::Retain)
            .unwrap();
        for (step, time) in [1.0, 0.7, 0.3].into_iter().enumerate() {
            let latents =
                crate::engine::seeded_randn(42 + step as u64, &[2, 8, 4], device, dtype).unwrap();
            check(
                &actual.forward(&latents, time).unwrap(),
                &expected.forward(&latents, time).unwrap(),
                time,
            );
        }
    }

    /// A5: hoisting `1 + scale` / `tanh(gate)` out of the blocks and applying
    /// compact per-run rows is the SAME arithmetic as v0.32's per-token
    /// broadcast, prefill and cached steps alike, bit for bit.
    #[test]
    fn compact_rows_are_bitwise_the_legacy_modulation() {
        let mut cfg = tiny_config();
        cfg.num_layers = 3;
        let mut reference = tiny_transformer_on(cfg.clone(), &Device::Cpu);
        reference.set_exec_path(Qwen21ExecPath::legacy());
        let mut compact = tiny_transformer_on(cfg, &Device::Cpu);
        compact.set_exec_path(Qwen21ExecPath {
            compact_modulation: true,
            ..Qwen21ExecPath::legacy()
        });
        compare_paths(&reference, &compact, &Device::Cpu, DType::F32, |a, e, t| {
            let diff = (a - e).unwrap().abs().unwrap().max_all().unwrap();
            assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0, "t={t}");
        });
    }

    /// A6: the fused `layer_norm(x, 1 + scale, 0)` (CPU reference kernel)
    /// agrees with the hand-written F32 LayerNorm and multiply.
    #[test]
    fn fused_adaln_matches_the_hand_layer_norm() {
        let mut cfg = tiny_config();
        cfg.num_layers = 3;
        let mut reference = tiny_transformer_on(cfg.clone(), &Device::Cpu);
        reference.set_exec_path(Qwen21ExecPath::legacy());
        let mut fused = tiny_transformer_on(cfg, &Device::Cpu);
        fused.set_exec_path(Qwen21ExecPath {
            compact_modulation: true,
            fused_adaln: true,
            ..Qwen21ExecPath::legacy()
        });
        compare_paths(&reference, &fused, &Device::Cpu, DType::F32, |a, e, t| {
            let error = max_relative_error(a, e);
            assert!(error < 1e-5, "t={t}: {error}");
        });
    }

    /// The engine resolves its path at construction: CPU is legacy.
    #[test]
    fn a_cpu_transformer_resolves_the_legacy_path() {
        assert!(tiny_transformer().exec_path().is_legacy());
    }

    /// Every CUDA fast-path knob, alone and together, against the v0.32
    /// forward evaluated in F32 on the same weights: BF16 rounding apart,
    /// the same prediction, prefill and cached, on a padded batch. Skips
    /// without a CUDA device (CI has none).
    #[cfg(feature = "cuda")]
    #[test]
    fn every_cuda_fast_path_knob_matches_the_legacy_forward() {
        use super::super::exec_path::TargetAttention;
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let cfg = QwenImage21TransformerConfig {
            in_channels: 4,
            out_channels: 4,
            context_in_dim: 8,
            num_attention_heads: 2,
            attention_head_dim: 64,
            num_layers: 2,
            mlp_ratio: 2,
            axes_dims_rope: [16, 24, 24],
            eps: 1e-6,
        };
        let mut reference = tiny_transformer_dtype(cfg.clone(), &device, DType::F32);
        reference.set_exec_path(Qwen21ExecPath::legacy());
        let legacy = Qwen21ExecPath::legacy();
        // BF16 itself moves this toy model by a few percent; each knob may
        // add no more than that rounding again. `baseline` holds the v0.32
        // BF16 path's own error against F32, per step.
        let mut baseline = Vec::new();
        for (name, path) in [
            ("legacy-bf16", legacy),
            (
                "flash",
                Qwen21ExecPath {
                    attention: TargetAttention::FastStill,
                    ..legacy
                },
            ),
            (
                "projection",
                Qwen21ExecPath {
                    fused_projection: true,
                    f32_rope_tables: true,
                    ..legacy
                },
            ),
            (
                "compact",
                Qwen21ExecPath {
                    compact_modulation: true,
                    ..legacy
                },
            ),
            (
                "adaln",
                Qwen21ExecPath {
                    compact_modulation: true,
                    fused_adaln: true,
                    ..legacy
                },
            ),
            ("fast", Qwen21ExecPath::cuda_fast()),
        ] {
            let mut candidate = tiny_transformer_dtype(cfg.clone(), &device, DType::BF16);
            candidate.set_exec_path(path);
            // Same inputs for both: BF16 values, the reference widened to F32.
            let conditioning = QwenImage21TextConditioning {
                embeddings: crate::engine::seeded_randn(41, &[2, 5, 8], &device, DType::BF16)
                    .unwrap(),
                valid_tokens: vec![vec![true, true, true, true, false], vec![true; 5]],
                image_slots: vec![vec![false; 5]; 2],
            };
            let reference_conditioning = QwenImage21TextConditioning {
                embeddings: conditioning.embeddings.to_dtype(DType::F32).unwrap(),
                valid_tokens: conditioning.valid_tokens.clone(),
                image_slots: conditioning.image_slots.clone(),
            };
            let mut expected = reference
                .prepare_t2i(&reference_conditioning, 2, 4, PrefixCacheDecision::Retain)
                .unwrap();
            let mut actual = candidate
                .prepare_t2i(&conditioning, 2, 4, PrefixCacheDecision::Retain)
                .unwrap();
            for (step, time) in [1.0, 0.7, 0.3].into_iter().enumerate() {
                let latents =
                    crate::engine::seeded_randn(42 + step as u64, &[2, 8, 4], &device, DType::BF16)
                        .unwrap();
                let got = actual.forward(&latents, time).unwrap();
                assert_eq!(got.dtype(), DType::BF16);
                let want = expected
                    .forward(&latents.to_dtype(DType::F32).unwrap(), time)
                    .unwrap();
                let error = max_relative_error(&got, &want);
                if name == "legacy-bf16" {
                    baseline.push(error);
                }
                let bound = 2.0 * baseline[step] + 5e-3;
                assert!(error <= bound, "{name} t={time}: {error} > {bound}");
            }
        }
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
        let mut positive_cache = transformer
            .prepare_t2i(&positive, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
        let mut negative_cache = transformer
            .prepare_t2i(&negative, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
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
        assert!(transformer
            .prepare_t2i(&positive, 2, 2, PrefixCacheDecision::Retain)
            .unwrap()
            .layers
            .is_empty());
        let invalid = Tensor::zeros((2, 5, 4), DType::F32, device).unwrap();
        let mut fresh = transformer
            .prepare_t2i(&positive, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
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
        let mut cache = transformer
            .prepare_t2i(&conditioning, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
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
    fn recompute_decision_matches_full_forward_without_retention() {
        let transformer = tiny_transformer();
        let length = super::super::LEGACY_PREFIX_CACHE_TOKENS + 1;
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(21, &[1, length, 8], &Device::Cpu, DType::F32)
                .unwrap(),
            valid_tokens: vec![vec![true; length]],
            image_slots: vec![vec![false; length]],
        };
        let latents = Tensor::ones((1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let mut cache = transformer
            .prepare_t2i(&conditioning, 2, 2, PrefixCacheDecision::Recompute)
            .unwrap();
        for _ in 0..2 {
            assert_eq!(
                flat(&cache.forward(&latents, 0.5).unwrap()),
                flat(
                    &transformer
                        .forward_t2i(&latents, 0.5, &conditioning, 2, 2)
                        .unwrap()
                )
            );
            assert!(cache.layers.is_empty());
        }
    }
    #[test]
    fn prefix_cache_budget_is_additive_and_bounds_both_float32_cfg_branches() {
        use crate::device::{activation_bytes, qwen_image21_activation_bytes, ActivationFamily};
        let bound = super::super::prefix_cache_budget_bytes(1);
        assert_eq!(bound, 1_073_741_824);
        let backend = crate::attention::AttentionBackend::resolve_effective_for(
            crate::attention::AttentionPolicy::FastStill,
        );
        for size in [32u32, 1024, 2048] {
            for dtype_bytes in [2, 4] {
                for batch in [1, 2] {
                    let joint = u64::from(size) * u64::from(size) / 256
                        + super::super::LEGACY_PREFIX_CACHE_TOKENS as u64;
                    let workspace =
                        qwen_image21_activation_bytes(joint, batch, dtype_bytes, backend);
                    let total = activation_bytes(
                        size,
                        size,
                        batch,
                        dtype_bytes,
                        ActivationFamily::QwenImage21Dit,
                    );
                    assert_eq!(total - workspace, bound * u64::from(batch));
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn compact_modulation_is_exact_across_cached_steps() {
        let device = crate::device::metal_device(0).unwrap();
        let mut reference = tiny_transformer_on(tiny_config(), &device);
        reference.set_exec_path(Qwen21ExecPath::metal(false));
        let mut compact = tiny_transformer_on(tiny_config(), &device);
        compact.set_exec_path(Qwen21ExecPath {
            compact_modulation: true,
            ..Qwen21ExecPath::metal(false)
        });
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(31, &[2, 3, 8], &device, DType::F32).unwrap(),
            valid_tokens: vec![vec![true, true, false], vec![true; 3]],
            image_slots: vec![vec![false; 3]; 2],
        };
        let mut expected = reference
            .prepare_t2i(&conditioning, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
        let mut actual = compact
            .prepare_t2i(&conditioning, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
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
        reference.set_exec_path(Qwen21ExecPath::metal(false));
        let mut optimized = tiny_transformer_on(cfg, &device);
        optimized.set_exec_path(Qwen21ExecPath {
            fused_projection: true,
            ..Qwen21ExecPath::metal(false)
        });
        let conditioning = QwenImage21TextConditioning {
            embeddings: crate::engine::seeded_randn(25, &[2, 3, 8], &device, DType::F32).unwrap(),
            valid_tokens: vec![vec![true, true, false], vec![true; 3]],
            image_slots: vec![vec![false; 3]; 2],
        };
        let mut expected = reference
            .prepare_t2i(&conditioning, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
        let mut actual = optimized
            .prepare_t2i(&conditioning, 2, 2, PrefixCacheDecision::Retain)
            .unwrap();
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
        use super::super::layout::{AttentionSegment, SegmentMask};
        let device = crate::device::metal_device(0).unwrap();
        use super::super::exec_path::TargetAttention;
        let mut dispatch = SegmentDispatch {
            attention: TargetAttention::MetalSdpa,
            head_dim: 128,
        };
        for dtype in [DType::F32, DType::BF16] {
            let q = crate::engine::seeded_randn(21, &[2, 2, 17, 128], &device, dtype).unwrap();
            let k = crate::engine::seeded_randn(22, &[2, 2, 29, 128], &device, dtype).unwrap();
            let v = crate::engine::seeded_randn(23, &[2, 2, 29, 128], &device, dtype).unwrap();
            dispatch.attention = TargetAttention::MetalSdpa;
            let actual = dispatch.full_unbiased(&q, &k, &v).unwrap();
            dispatch.attention = TargetAttention::Legacy;
            let expected = dispatch.full_unbiased(&q, &k, &v).unwrap();
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
            // A real padded batch must preserve the math mask semantics even
            // with the fused path enabled.
            dispatch.attention = TargetAttention::MetalSdpa;
            let key_valid: Vec<Vec<bool>> =
                (0..2).map(|_| (0..29).map(|i| i != 28).collect()).collect();
            let plan = BlockCausalPlan {
                segments: vec![AttentionSegment {
                    q_start: 0,
                    q_len: 17,
                    kv_len: 29,
                    mask: SegmentMask::Full,
                }],
                key_valid: Some(key_valid.into()),
            };
            let actual = dispatch.attend(&q, &k, &v, &plan).unwrap();
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
            conditioning.sequence_length() <= super::super::LEGACY_PREFIX_CACHE_TOKENS,
            "benchmark prompt exceeds cache retention bound"
        );
        drop(encoder);
        let paths = (1..=2).map(|i| root.join(format!("qwen-image-2.1-bf16/transformer/diffusion_pytorch_model-{i:05}-of-00002.safetensors"))).collect::<Vec<_>>();
        let transformer = QwenImage21Transformer::load(&paths, &device, DType::F32, &progress)?;
        let mut cache = transformer
            .prepare_t2i(&conditioning, 64, 64, PrefixCacheDecision::Retain)
            .unwrap();
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

    /// A layout with condition images has no v0.32 bytes, so even the legacy
    /// path builds float32 tables for it (upstream's dtype); a text-to-image
    /// layout keeps the working dtype there.
    #[test]
    fn condition_layouts_take_f32_rope_tables_on_every_path() {
        let transformer = tiny_transformer();
        assert!(transformer.exec_path().is_legacy());
        let t2i = QwenImage21JointLayout::text_to_image(&[vec![true; 3]], (2, 2)).unwrap();
        let slots = [false, false, false, true, true, false, false, false];
        let referenced =
            QwenImage21JointLayout::build(&slots, &[vec![true; 8]], &[(2, 4)], (4, 4)).unwrap();
        assert_eq!(transformer.rope_table_dtype(DType::BF16, &t2i), DType::BF16);
        assert_eq!(
            transformer.rope_table_dtype(DType::BF16, &referenced),
            DType::F32
        );
    }

    /// The two angle arithmetics are genuinely different tables, so the
    /// per-request choice is observable.
    #[test]
    fn upstream_and_v032_angles_differ() {
        let layout = QwenImage21JointLayout::text_to_image(&[vec![true; 40]], (8, 8)).unwrap();
        let axes = tiny_transformer().cfg.axes_dims_rope;
        let build = |angles| {
            let (cos, _) = QwenImage21JointLayout::rope_tables(
                layout.rope(),
                axes,
                angles,
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            flat(&cos)
        };
        assert_ne!(build(RopeAngles::Upstream), build(RopeAngles::V032));
    }

    /// U3: the text-to-image layout's rotary tables equal v0.32's
    /// `t2i_rope`, bit for bit, at every working dtype.
    #[test]
    fn t2i_layout_rope_equals_the_legacy_t2i_rope() {
        let transformer = tiny_transformer();
        for (text_len, height, width) in [(3, 2, 2), (7, 4, 6), (1, 6, 2)] {
            let valid = vec![vec![true; text_len]];
            let layout = QwenImage21JointLayout::text_to_image(&valid, (height, width)).unwrap();
            for dtype in [DType::F32, DType::BF16] {
                let (cos, sin) = QwenImage21JointLayout::rope_tables(
                    layout.rope(),
                    transformer.cfg.axes_dims_rope,
                    RopeAngles::V032,
                    dtype,
                    &Device::Cpu,
                )
                .unwrap();
                let (legacy_cos, legacy_sin) = legacy_oracle::t2i_rope(
                    &transformer,
                    text_len,
                    height,
                    width,
                    dtype,
                    &Device::Cpu,
                )
                .unwrap();
                for (actual, expected) in [(cos, legacy_cos), (sin, legacy_sin)] {
                    assert_eq!(actual.dtype(), expected.dtype());
                    assert_eq!(flat(&actual), flat(&expected));
                }
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_rope_keeps_f32_tables_for_bf16_latents() {
        let transformer = tiny_transformer();
        let device = crate::device::metal_device(0).unwrap();
        let layout = QwenImage21JointLayout::text_to_image(&[vec![true; 3]], (2, 2)).unwrap();
        let axes = transformer.cfg.axes_dims_rope;
        let (expected_cos, expected_sin) = QwenImage21JointLayout::rope_tables(
            layout.rope(),
            axes,
            RopeAngles::V032,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let (cos, sin) = QwenImage21JointLayout::rope_tables(
            layout.rope(),
            axes,
            RopeAngles::V032,
            DType::BF16,
            &device,
        )
        .unwrap();
        for (actual, expected) in [(cos, expected_cos), (sin, expected_sin)] {
            assert_eq!(actual.dtype(), DType::F32);
            assert_eq!(
                flat(&actual.to_device(&Device::Cpu).unwrap()),
                flat(&expected)
            );
        }
        let (cos, sin) = QwenImage21JointLayout::rope_tables(
            layout.rope(),
            axes,
            RopeAngles::V032,
            DType::BF16,
            &Device::Cpu,
        )
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

    fn flat(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn padded_and_unpadded(length: usize, device: &Device) -> Vec<QwenImage21TextConditioning> {
        vec![
            QwenImage21TextConditioning {
                embeddings: crate::engine::seeded_randn(41, &[2, length, 8], device, DType::F32)
                    .unwrap(),
                valid_tokens: vec![
                    (0..length).map(|i| i + 2 < length).collect(),
                    vec![true; length],
                ],
                image_slots: vec![vec![false; length]; 2],
            },
            QwenImage21TextConditioning {
                embeddings: crate::engine::seeded_randn(42, &[1, length, 8], device, DType::F32)
                    .unwrap(),
                valid_tokens: vec![vec![true; length]],
                image_slots: vec![vec![false; length]],
            },
        ]
    }

    /// The layout-generalized text-to-image forward is BITWISE the frozen
    /// v0.32 forward, uncached and across cached steps, padded or not, in
    /// F32 and BF16 — archived seeds keep their bytes.
    #[test]
    fn t2i_forward_is_bitwise_the_frozen_legacy_forward() {
        // Candle's CPU matmul has no BF16 kernel; F16 exercises the same
        // half-precision rounding boundaries. The accelerator variants below
        // cover BF16 on CUDA and Metal.
        legacy_parity(&Device::Cpu, &[DType::F32, DType::F16]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn t2i_forward_is_bitwise_the_frozen_legacy_forward_on_cuda() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        legacy_parity(&device, &[DType::F32, DType::BF16]);
        eprintln!("CUDA F32 and BF16 legacy parity: bitwise");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn t2i_forward_is_bitwise_the_frozen_legacy_forward_on_metal() {
        let device = crate::device::metal_device(0).unwrap();
        legacy_parity(&device, &[DType::F32, DType::BF16]);
    }

    fn legacy_parity(device: &Device, dtypes: &[DType]) {
        let device = device.clone();
        for &dtype in dtypes {
            for heads in [1, 2] {
                let mut cfg = tiny_config();
                cfg.num_layers = 3;
                cfg.num_attention_heads = heads;
                let mut transformer = tiny_transformer_dtype(cfg, &device, dtype);
                // The oracle is v0.32's forward; off Metal that is the legacy
                // path, which CUDA now reaches only through MOLD_ATTN=math.
                if !device.is_metal() {
                    transformer.set_exec_path(Qwen21ExecPath::legacy());
                }
                for conditioning in padded_and_unpadded(5, &device) {
                    let conditioning = conditioning.to_device_dtype(&device, dtype).unwrap();
                    let batch = conditioning.batch_size();
                    let mut branch = transformer
                        .prepare_t2i(&conditioning, 2, 4, PrefixCacheDecision::Retain)
                        .unwrap();
                    let mut legacy_layers: Vec<PrefixKv> = Vec::new();
                    for (step, time) in [1.0, 0.6, 0.2].into_iter().enumerate() {
                        let latents = crate::engine::seeded_randn(
                            50 + step as u64,
                            &[batch, 8, 4],
                            &device,
                            dtype,
                        )
                        .unwrap();
                        let uncached = transformer
                            .forward_t2i(&latents, time, &conditioning, 2, 4)
                            .unwrap();
                        let legacy_uncached = legacy_oracle::forward_with_cache(
                            &transformer,
                            &latents,
                            time,
                            &conditioning,
                            2,
                            4,
                            legacy_oracle::LegacyCache::Disabled,
                        )
                        .unwrap();
                        assert_eq!(
                            flat(&uncached),
                            flat(&legacy_uncached),
                            "{dtype:?} uncached"
                        );

                        let cached = branch.forward(&latents, time).unwrap();
                        let legacy_cached = if legacy_layers.is_empty() {
                            let mut layers = Vec::new();
                            let out = legacy_oracle::forward_with_cache(
                                &transformer,
                                &latents,
                                time,
                                &conditioning,
                                2,
                                4,
                                legacy_oracle::LegacyCache::Extract(&mut layers),
                            )
                            .unwrap();
                            legacy_layers = layers;
                            out
                        } else {
                            legacy_oracle::forward_with_cache(
                                &transformer,
                                &latents,
                                time,
                                &conditioning,
                                2,
                                4,
                                legacy_oracle::LegacyCache::Reuse(&legacy_layers),
                            )
                            .unwrap()
                        };
                        assert_eq!(
                            flat(&cached),
                            flat(&legacy_cached),
                            "{dtype:?} cached step {step}"
                        );
                    }
                }
            }
        }
    }

    /// One reference image (2x4 latents = two slots) between text rows, with
    /// positive and negative prompts of different lengths and padding.
    struct ReferenceCase {
        positive: QwenImage21TextConditioning,
        negative: QwenImage21TextConditioning,
        cond_latents: Tensor,
    }

    fn reference_case(device: &Device) -> ReferenceCase {
        let conditioning = |slots: Vec<bool>, pad: [usize; 2], seed| {
            let length = slots.len();
            QwenImage21TextConditioning {
                embeddings: crate::engine::seeded_randn(seed, &[2, length, 8], device, DType::F32)
                    .unwrap(),
                valid_tokens: pad
                    .iter()
                    .map(|pad| (0..length).map(|i| i + pad < length).collect())
                    .collect(),
                image_slots: vec![slots; 2],
            }
        };
        ReferenceCase {
            positive: conditioning(
                vec![false, false, true, true, false, false, false],
                [1, 0],
                61,
            ),
            negative: conditioning(
                vec![false, true, true, false, false, false, false, false, false],
                [2, 1],
                62,
            ),
            cond_latents: crate::engine::seeded_randn(63, &[2, 8, 4], device, DType::F32).unwrap(),
        }
    }

    fn reference_layout(conditioning: &QwenImage21TextConditioning) -> QwenImage21JointLayout {
        QwenImage21JointLayout::build(
            &conditioning.image_slots[0],
            &conditioning.valid_tokens,
            &[(2, 4)],
            (2, 2),
        )
        .unwrap()
    }

    /// A deliberately naive, layout-free forward written from upstream's
    /// transformer (`transformer_qwenimage21.py`): repeat/overwrite the image
    /// slots row by row (`:905-923`), label blocks from the shapes
    /// (`:808-850`), lay RoPE out with `QwenImage21Rope.forward`'s cursor walk
    /// (`:677-710`), select modulation rows per token (`:238-254`), and
    /// attend densely under `mask_mod` (`:292-296`).
    fn naive_forward(
        transformer: &QwenImage21Transformer,
        latents: &Tensor,
        cond_latents: &Tensor,
        timestep: f64,
        conditioning: &QwenImage21TextConditioning,
        cond_shapes: &[(usize, usize)],
        target: (usize, usize),
    ) -> Tensor {
        let device = latents.device();
        let batch = latents.dim(0).unwrap();
        let slots = &conditioning.image_slots[0];
        let text = transformer
            .txt_in
            .forward(&conditioning.embeddings)
            .unwrap();
        let images = transformer
            .img_in
            .forward(&Tensor::cat(&[cond_latents, latents], 1).unwrap())
            .unwrap();
        // Joint position -> Some(VL text row) | None (image token).
        let mut joint: Vec<Option<usize>> = Vec::new();
        for (row, slot) in slots.iter().enumerate() {
            if *slot {
                joint.extend([None; 4]);
            } else {
                joint.push(Some(row));
            }
        }
        let target_tokens = target.0 * target.1;
        joint.extend(std::iter::repeat_n(None, target_tokens));
        let total = joint.len();
        let mut rows = Vec::new();
        let mut image_row = 0;
        for position in &joint {
            match position {
                Some(row) => rows.push(text.narrow(1, *row, 1).unwrap()),
                None => {
                    rows.push(images.narrow(1, image_row, 1).unwrap());
                    image_row += 1;
                }
            }
        }
        let mut hidden = Tensor::cat(&rows, 1).unwrap();

        let shapes: Vec<(usize, usize)> = cond_shapes
            .iter()
            .copied()
            .chain(std::iter::once(target))
            .collect();
        let image_positions: Vec<usize> = (0..total).filter(|&p| joint[p].is_none()).collect();
        let mut image_ids = vec![-1i64; total];
        let mut cursor_id = 0;
        for (block, (h, w)) in shapes.iter().enumerate() {
            for _ in 0..h * w {
                image_ids[image_positions[cursor_id]] = block as i64;
                cursor_id += 1;
            }
        }
        let target_mask: Vec<bool> = (0..total)
            .map(|p| image_ids[p] == shapes.len() as i64 - 1)
            .collect();

        // QwenImage21Rope.forward.
        let is_image: Vec<bool> = joint.iter().map(Option::is_none).collect();
        let (mut frame, mut image_h, mut image_w) = (Vec::new(), Vec::new(), Vec::new());
        let (mut cursor, mut position) = (0usize, 0i32);
        for &(h, w) in &shapes {
            let block_start = (cursor..total).find(|&p| is_image[p]).unwrap();
            let text_len = block_start - cursor;
            frame.extend((0..text_len).map(|i| position + i as i32));
            position += text_len as i32;
            cursor = block_start + h * w;
            frame.extend(std::iter::repeat_n(position, h * w));
            position += h.max(w) as i32;
            let (h, w) = (h as i32, w as i32);
            for y in -(h - h / 2)..h / 2 {
                for x in -(w - w / 2)..w / 2 {
                    image_h.push(y);
                    image_w.push(x);
                }
            }
        }
        frame.extend((0..total - cursor).map(|i| position + i as i32));
        let mut coords: Vec<[i32; 3]> = frame.iter().map(|&f| [f, f, f]).collect();
        for (index, &p) in image_positions.iter().enumerate() {
            coords[p][1] = image_h[index];
            coords[p][2] = image_w[index];
        }
        // The CPU transformer under test takes the legacy path, whose angle
        // arithmetic is v0.32's on every layout.
        let (cos, sin) = QwenImage21JointLayout::rope_tables(
            &coords,
            transformer.cfg.axes_dims_rope,
            RopeAngles::V032,
            DType::F32,
            device,
        )
        .unwrap();

        // Dense mask; joint text positions map in order onto VL text rows.
        let mut bias = Vec::new();
        for row in 0..batch {
            let valid: Vec<bool> = joint
                .iter()
                .map(|p| p.is_none_or(|r| conditioning.valid_tokens[row][r]))
                .collect();
            for q in 0..total {
                for kv in 0..total {
                    let same = image_ids[q] >= 0 && image_ids[q] == image_ids[kv];
                    bias.push(if (q >= kv || same) && valid[kv] {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        }
        let bias = Tensor::from_vec(bias, (batch, 1, total, total), device).unwrap();

        let mut timesteps = vec![timestep; batch];
        timesteps.push(0.0);
        let temb = transformer
            .time_text_embed
            .forward(&timesteps, DType::F32, device)
            .unwrap();
        let modulation = transformer
            .modulation
            .forward(&candle_nn::Activation::Silu.forward(&temb).unwrap())
            .unwrap();
        let per_token = Tensor::stack(
            &(0..batch)
                .map(|b| {
                    Tensor::cat(
                        &target_mask
                            .iter()
                            .map(|&is_target| {
                                modulation
                                    .narrow(0, if is_target { b } else { batch }, 1)
                                    .unwrap()
                            })
                            .collect::<Vec<_>>(),
                        0,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();

        let heads = transformer.cfg.num_attention_heads;
        let head_dim = transformer.cfg.attention_head_dim;
        let inner = transformer.cfg.inner_dim();
        for block in &transformer.blocks {
            let mod1 = per_token.narrow(D::Minus1, 0, 2 * inner).unwrap();
            let mod2 = per_token.narrow(D::Minus1, 2 * inner, 2 * inner).unwrap();
            let (normalized, gate) =
                TransformerBlock::modulate(block.norm1.forward(&hidden).unwrap(), &mod1).unwrap();
            let attn = &block.attn;
            let project = |linear: &Q21Linear| {
                linear
                    .forward(&normalized)
                    .unwrap()
                    .reshape((batch, total, heads, head_dim))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
            };
            let rope = |x: Tensor| {
                crate::wan::model::rope::apply_rope(&x.transpose(1, 2).unwrap(), &cos, &sin)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap()
            };
            let q = rope(
                attn.normalize_heads(&project(&attn.to_q), &attn.norm_q)
                    .unwrap(),
            );
            let k = rope(
                attn.normalize_heads(&project(&attn.to_k), &attn.norm_k)
                    .unwrap(),
            );
            let v = project(&attn.to_v).contiguous().unwrap();
            let context = crate::attention::attention_with_bias(
                &q,
                &k,
                &v,
                (1.0 / (head_dim as f64).sqrt()) as f32,
                Some(&bias),
            )
            .unwrap();
            let attn_out = attn
                .to_out
                .forward(
                    &context
                        .transpose(1, 2)
                        .unwrap()
                        .reshape((batch, total, inner))
                        .unwrap(),
                )
                .unwrap();
            hidden = (&hidden + gate.tanh().unwrap().broadcast_mul(&attn_out).unwrap()).unwrap();
            let (normalized, gate) =
                TransformerBlock::modulate(block.norm2.forward(&hidden).unwrap(), &mod2).unwrap();
            hidden = (&hidden
                + gate
                    .tanh()
                    .unwrap()
                    .broadcast_mul(&block.mlp.forward(&normalized).unwrap())
                    .unwrap())
            .unwrap();
        }
        let target_hidden = hidden
            .narrow(1, total - target_tokens, target_tokens)
            .unwrap();
        transformer
            .proj_out
            .forward(
                &transformer
                    .norm_out
                    .forward(&target_hidden, &temb.narrow(0, 0, batch).unwrap())
                    .unwrap(),
            )
            .unwrap()
    }

    /// U5: a forward with a condition block equals the naive dense oracle.
    #[test]
    fn condition_block_forward_matches_the_naive_dense_oracle() {
        let device = Device::Cpu;
        for heads in [1, 2] {
            let mut cfg = tiny_config();
            cfg.num_layers = 3;
            cfg.num_attention_heads = heads;
            let transformer = tiny_transformer_on(cfg, &device);
            let case = reference_case(&device);
            for conditioning in [&case.positive, &case.negative] {
                let layout = reference_layout(conditioning);
                for time in [1.0, 0.4] {
                    let latents =
                        crate::engine::seeded_randn(70, &[2, 4, 4], &device, DType::F32).unwrap();
                    let actual = transformer
                        .forward_uncached(
                            &latents,
                            Some(&case.cond_latents),
                            time,
                            conditioning,
                            &layout,
                        )
                        .unwrap();
                    let expected = naive_forward(
                        &transformer,
                        &latents,
                        &case.cond_latents,
                        time,
                        conditioning,
                        &[(2, 4)],
                        (2, 2),
                    );
                    assert_close(&actual, &expected);
                }
            }
        }
    }

    /// U5: prefix-cache parity with a condition block, across steps and both
    /// CFG branches (which have different text lengths but share the
    /// condition latents).
    #[test]
    fn condition_block_cache_matches_full_forward_across_steps_and_branches() {
        let device = Device::Cpu;
        let mut cfg = tiny_config();
        cfg.num_layers = 3;
        cfg.num_attention_heads = 2;
        let transformer = tiny_transformer_on(cfg, &device);
        let case = reference_case(&device);
        let mut branches: Vec<_> = [&case.positive, &case.negative]
            .into_iter()
            .map(|conditioning| {
                transformer
                    .prepare(
                        conditioning,
                        reference_layout(conditioning),
                        Some(case.cond_latents.clone()),
                        PrefixCacheDecision::Retain,
                    )
                    .unwrap()
            })
            .collect();
        for (step, time) in [1.0, 0.7, 0.3, 0.01].into_iter().enumerate() {
            let latents =
                crate::engine::seeded_randn(80 + step as u64, &[2, 4, 4], &device, DType::F32)
                    .unwrap();
            for branch in &mut branches {
                let expected = transformer
                    .forward_uncached(
                        &latents,
                        Some(&case.cond_latents),
                        time,
                        branch.conditioning,
                        branch.layout(),
                    )
                    .unwrap();
                let actual = branch.forward(&latents, time).unwrap();
                assert_close(&actual, &expected);
                let prefix = branch.layout().prefix_len();
                assert_eq!(branch.layers.len(), 3);
                for layer in &branch.layers {
                    // Text AND condition-image tokens are retained.
                    assert_eq!(layer.key.dims(), &[2, 2, prefix, 8]);
                }
            }
        }
        assert_eq!(branches[0].layout().prefix_len(), 5 + 8);
        assert_eq!(branches[1].layout().prefix_len(), 7 + 8);
    }

    /// U6: condition-image rows modulate from the t=0 row, so the prefix a
    /// prefill retains does not depend on the step's timestep.
    #[test]
    fn condition_rows_take_the_t0_modulation_row() {
        let device = Device::Cpu;
        let transformer = tiny_transformer();
        let case = reference_case(&device);
        let extract = |time: f64| {
            let mut branch = transformer
                .prepare(
                    &case.positive,
                    reference_layout(&case.positive),
                    Some(case.cond_latents.clone()),
                    PrefixCacheDecision::Retain,
                )
                .unwrap();
            let latents = crate::engine::seeded_randn(90, &[2, 4, 4], &device, DType::F32).unwrap();
            branch.forward(&latents, time).unwrap();
            flat(&branch.layers[0].key)
        };
        assert_eq!(extract(1.0), extract(0.25));
    }

    #[test]
    fn prepare_refuses_mismatched_condition_latents() {
        let device = Device::Cpu;
        let transformer = tiny_transformer();
        let case = reference_case(&device);
        let layout = reference_layout(&case.positive);
        assert!(transformer
            .prepare(
                &case.positive,
                layout.clone(),
                None,
                PrefixCacheDecision::Retain
            )
            .is_err());
        let short = case.cond_latents.narrow(1, 0, 4).unwrap();
        assert!(transformer
            .prepare(
                &case.positive,
                layout,
                Some(short),
                PrefixCacheDecision::Retain
            )
            .is_err());
        assert!(transformer
            .prepare_t2i(&case.positive, 2, 2, PrefixCacheDecision::Retain)
            .is_err());
    }
}

#[cfg(test)]
mod legacy_oracle;

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod performance_tests;
