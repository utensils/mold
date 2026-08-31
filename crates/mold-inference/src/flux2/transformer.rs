//! Flux.2 Klein-4B Transformer (diffusers weight format)
//!
//! Architecture: DoubleStreamBlock + SingleStreamBlock (same as FLUX.1) but with:
//! - `in_channels`: 128 (patchified latent_channels=32 * 2x2)
//! - `axes_dims_rope`: 4D [32, 32, 32, 32]
//! - `joint_attention_dim`: 7680 (Qwen3 hidden_size=2560, stacked 3x)
//! - `mlp_ratio`: 3.0, `rope_theta`: 2000
//! - Shared modulation across all blocks (not per-block)
//! - All linear layers bias=False
//! - 5 double + 20 single blocks for Klein-4B
//!
//! Loads from HuggingFace diffusers `Flux2Transformer2DModel` safetensors format.

use crate::adaptive_offload::{
    plan_adaptive_residency, AdaptiveResidencyPlan, ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
};
use crate::progress::ProgressReporter;
use candle_core::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// Linear (BF16 + FP8 + NVFP4 streaming)
// ---------------------------------------------------------------------------

/// Linear layer supporting BF16 (`Standard`), FP8 manual-cast (`Fp8`), and
/// NVFP4 streaming-dequant (`Nvfp4Streaming`).
///
/// `Standard` mirrors `candle_nn::Linear`. `Fp8` mirrors
/// `qwen_image::transformer::QwenLinear::Fp8` — F8E4M3 weights resident on the
/// model device, cast to the activation dtype at forward.
///
/// `Nvfp4Streaming` keeps the packed FP4 + FP8 block scales mmap'd on CPU and
/// dequantizes lazily on first forward into a BF16 weight, also cached on
/// CPU. Subsequent forwards copy the cached BF16 weight to the activation
/// device for matmul (no re-dequant). For sliced fused QKV the cache is shared
/// across `to_q`/`to_k`/`to_v` via `Arc<OnceLock<Tensor>>` so the FP4 →
/// BF16 dequant runs exactly once per fused source. Memory: ~18 GB BF16 +
/// ~5.6 GB packed source on CPU for Klein-9B; per-forward GPU peak is one
/// layer's BF16 weight (≈ 64-200 MB). This is what makes Klein-9B fit on
/// a 24 GB 3090.
///
/// Auto-detection in `load_with_bias`:
///   `vb.contains_tensor("weight.nvfp4_packed")` → `Nvfp4Streaming`
///   `vb.get(...).dtype() == F8E4M3`              → `Fp8`
///   otherwise                                     → `Standard`
#[derive(Debug, Clone)]
pub(crate) enum Flux2Linear {
    Standard(candle_nn::Linear),
    Fp8 {
        weight: Tensor,
        scale: Option<Tensor>,
        bias: Option<Tensor>,
    },
    Nvfp4Streaming {
        /// Packed FP4 nibbles, U8 `[N_full, K/2]` on CPU.
        packed: Tensor,
        /// FP8-E4M3 per-block scales `[N_full, K/16]` on CPU.
        block_scales: Tensor,
        /// Per-tensor F32 scalar (as `f32` to skip a per-forward host read).
        tensor_scale: f32,
        /// Output dim *after* slicing — what the caller's matmul expects.
        out_dim: usize,
        /// Input dim K (matches `block_scales.dim(1) * 16`). Stored for
        /// post-load introspection; the forward path derives K from `packed`.
        #[allow(dead_code)]
        in_dim: usize,
        /// Optional `(axis, component, num_components)` slice descriptor.
        /// `None` for unfused layers; `Some((0, c, 3))` for sliced QKV.
        slice: Option<(usize, usize, usize)>,
        /// Bias on the model device (NVFP4 layers in cv:2759597 have none).
        bias: Option<Tensor>,
        /// Lazy CPU BF16 cache of the FULL (un-sliced) dequanted weight,
        /// shape `[N_full, K]`. Sliced QKV variants share this cache via
        /// `Arc<OnceLock>`, so the FP4 → BF16 dequant happens exactly once
        /// per fused source even when three sliced linears reference it.
        cache: Arc<OnceLock<Tensor>>,
    },
}

impl Flux2Linear {
    fn load_with_bias(
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // NVFP4 streaming path: probe for the sub-key the backend emits for
        // every NVFP4-quantised layer. If present, all three components plus
        // the optional slice-meta marker live alongside it.
        if vb.contains_tensor("weight.nvfp4_packed") {
            // Explicit-dtype lookups: each NVFP4 sub-key has a native dtype
            // distinct from the VarBuilder's default (BF16). `get_unchecked`
            // would request the default and ask the backend to cast — which
            // breaks for U8 (packed), F8E4M3 (block scales), F32 (tensor
            // scale), and U32 (slice meta).
            let packed = vb.get_unchecked_dtype("weight.nvfp4_packed", DType::U8)?;
            let block_scales =
                vb.get_unchecked_dtype("weight.nvfp4_block_scales", DType::F8E4M3)?;
            let tensor_scale_t = vb.get_unchecked_dtype("weight.nvfp4_tensor_scale", DType::F32)?;
            // The backend already returns these on CPU; defensive `to_device`
            // keeps the contract local to this constructor in case the
            // backend's invariant changes.
            let cpu = candle_core::Device::Cpu;
            let packed = packed.to_device(&cpu)?;
            let block_scales = block_scales.to_device(&cpu)?;
            let tensor_scale: f32 = tensor_scale_t.to_dtype(DType::F32)?.to_scalar()?;

            let slice = if vb.contains_tensor("weight.nvfp4_slice_meta") {
                let meta = vb
                    .get_unchecked_dtype("weight.nvfp4_slice_meta", DType::U32)?
                    .to_device(&cpu)?;
                let v: Vec<u32> = meta.flatten_all()?.to_vec1()?;
                if v.len() != 3 {
                    candle_core::bail!(
                        "NVFP4 slice meta tensor must have length 3, got {}",
                        v.len()
                    );
                }
                Some((v[0] as usize, v[1] as usize, v[2] as usize))
            } else {
                None
            };

            // Validate shapes against the requested (out_dim, in_dim) given
            // any slicing — catches a bad rename table early.
            let packed_dims = packed.dims();
            if packed_dims.len() != 2 {
                candle_core::bail!("NVFP4 packed weight must be rank 2, got {:?}", packed_dims,);
            }
            let n_full = packed_dims[0];
            let k_half = packed_dims[1];
            let k = k_half * 2;
            if k != in_dim {
                candle_core::bail!(
                    "NVFP4: in_dim mismatch — checkpoint K={}, module expected {}",
                    k,
                    in_dim,
                );
            }
            let expected_n_full = match slice {
                Some((_, _, n_components)) => out_dim * n_components,
                None => out_dim,
            };
            if n_full != expected_n_full {
                candle_core::bail!(
                    "NVFP4: out_dim mismatch — checkpoint N_full={}, module expected {} (out_dim={}, slice={:?})",
                    n_full,
                    expected_n_full,
                    out_dim,
                    slice,
                );
            }

            let bias = if has_bias {
                vb.get_unchecked("bias").ok()
            } else {
                None
            };

            return Ok(Self::Nvfp4Streaming {
                packed,
                block_scales,
                tensor_scale,
                out_dim,
                in_dim,
                slice,
                bias,
                cache: Arc::new(OnceLock::new()),
            });
        }

        let weight = vb.get((out_dim, in_dim), "weight")?;
        if weight.dtype() == DType::F8E4M3 {
            // Every producer writes the scale as F32 (BFL's `weight_scale`,
            // ComfyUI's `scale_weight` — the single-file backend normalizes
            // the spelling), so asking for F32 is the identity cast on every
            // shipped checkpoint and keeps the stored value intact through
            // load. `forward` still multiplies in the working dtype, like
            // every other weight in the block.
            let scale = vb
                .get_unchecked_dtype("scale_weight", DType::F32)
                .or_else(|_| vb.get_unchecked("scale_weight"))
                .ok();
            let bias = if has_bias {
                vb.get_unchecked("bias").ok()
            } else {
                None
            };
            Ok(Self::Fp8 {
                weight,
                scale,
                bias,
            })
        } else {
            let bias = if has_bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(Self::Standard(candle_nn::Linear::new(weight, bias)))
        }
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        match self {
            Self::Standard(linear) => Ok(Self::Standard(linear_to_device(linear, device)?)),
            Self::Fp8 {
                weight,
                scale,
                bias,
            } => Ok(Self::Fp8 {
                weight: weight.to_device(device)?,
                scale: scale.as_ref().map(|t| t.to_device(device)).transpose()?,
                bias: bias.as_ref().map(|t| t.to_device(device)).transpose()?,
            }),
            Self::Nvfp4Streaming { .. } => {
                candle_core::bail!("Flux.2 block offload does not support NVFP4 streaming layers")
            }
        }
    }
}

fn linear_to_device(linear: &Linear, device: &candle_core::Device) -> Result<Linear> {
    let weight = linear.weight().to_device(device)?;
    let bias = linear
        .bias()
        .map(|bias| bias.to_device(device))
        .transpose()?;
    Ok(Linear::new(weight, bias))
}

fn layer_norm_to_device(norm: &LayerNorm, device: &candle_core::Device) -> Result<LayerNorm> {
    let weight = norm.weight().to_device(device)?;
    match norm.bias() {
        Some(bias) => Ok(LayerNorm::new(weight, bias.to_device(device)?, 1e-6)),
        None => Ok(LayerNorm::new_no_bias(weight, 1e-6)),
    }
}

fn rms_norm_to_device(norm: &RmsNorm, device: &candle_core::Device) -> Result<RmsNorm> {
    let inner = norm.clone().into_inner();
    Ok(RmsNorm::new(inner.weight().to_device(device)?, 1e-6))
}

fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

fn flux2_linear_bytes(linear: &Flux2Linear) -> usize {
    match linear {
        Flux2Linear::Standard(linear) => {
            tensor_bytes(linear.weight()) + linear.bias().map(tensor_bytes).unwrap_or(0)
        }
        Flux2Linear::Fp8 {
            weight,
            scale,
            bias,
        } => {
            // The FP8 slab is 1 byte per parameter at rest, but `forward`
            // materializes a full working-dtype copy of it on every call, and
            // this figure drives the block-offload residency plan. Charging
            // the at-rest bytes would let the planner keep twice as many
            // blocks resident as the forward actually fits.
            tensor_bytes(weight) * (DType::BF16.size_in_bytes() / DType::F8E4M3.size_in_bytes())
                + scale.as_ref().map(tensor_bytes).unwrap_or(0)
                + bias.as_ref().map(tensor_bytes).unwrap_or(0)
        }
        Flux2Linear::Nvfp4Streaming {
            packed,
            block_scales,
            bias,
            cache,
            ..
        } => {
            tensor_bytes(packed)
                + tensor_bytes(block_scales)
                + bias.as_ref().map(tensor_bytes).unwrap_or(0)
                + cache.get().map(tensor_bytes).unwrap_or(0)
        }
    }
}

fn layer_norm_bytes(norm: &LayerNorm) -> usize {
    tensor_bytes(norm.weight()) + norm.bias().map(tensor_bytes).unwrap_or(0)
}

fn rms_norm_bytes(norm: &RmsNorm) -> usize {
    tensor_bytes(norm.clone().into_inner().weight())
}

impl Module for Flux2Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(x),
            Self::Fp8 {
                weight,
                scale,
                bias,
            } => {
                let dtype = x.dtype();
                // A per-tensor scale is linear in the matmul, so it rides on
                // the OUTPUT — which is orders of magnitude smaller than the
                // weight — instead of costing a second full-size pass over the
                // dequantized slab. Mirrors the Qwen-Image FP8 rule. Anything
                // with per-row structure still scales the weight.
                let scalar_scale = scale.as_ref().filter(|s| s.rank() == 0);
                let w = weight.to_dtype(dtype)?;
                let w = match scale {
                    Some(s) if scalar_scale.is_none() => w.broadcast_mul(&s.to_dtype(dtype)?)?,
                    _ => w,
                };
                let w = w.t()?;
                let out = match *x.dims() {
                    [b1, b2, m, k] => {
                        x.reshape((b1 * b2 * m, k))?
                            .matmul(&w)?
                            .reshape((b1, b2, m, ()))?
                    }
                    [bsize, m, k] => {
                        x.reshape((bsize * m, k))?
                            .matmul(&w)?
                            .reshape((bsize, m, ()))?
                    }
                    _ => x.matmul(&w)?,
                };
                let out = match scalar_scale {
                    Some(s) => out.broadcast_mul(&s.to_dtype(dtype)?)?,
                    None => out,
                };
                match bias {
                    Some(b) => out.broadcast_add(&b.to_dtype(dtype)?),
                    None => Ok(out),
                }
            }
            Self::Nvfp4Streaming {
                packed,
                block_scales,
                tensor_scale,
                out_dim,
                slice,
                bias,
                cache,
                ..
            } => {
                let _backend = crate::nvfp4::resolve_nvfp4_backend(x.device())?;
                // First forward: dequant FULL weight to BF16 on CPU and stash
                // it. Subsequent forwards skip straight to the slice + DMA.
                // OnceLock::get_or_try_init isn't stable yet; emulate it.
                let bf16_full = match cache.get() {
                    Some(t) => t,
                    None => {
                        let dequanted = crate::nvfp4::dequant_nvfp4_to_bf16_cpu(
                            packed,
                            block_scales,
                            *tensor_scale,
                        )?;
                        // Race-safe set: if another thread won, we drop ours
                        // and use theirs. Either way the value cached is the
                        // same dequant of the same source.
                        let _ = cache.set(dequanted);
                        cache.get().expect("cache populated above")
                    }
                };

                // Slice if needed. `bf16_full` lives on CPU; narrow is a
                // view, no copy.
                let bf16_sliced_cpu = match slice {
                    Some((axis, component, _n_components)) => {
                        bf16_full.narrow(*axis, component * out_dim, *out_dim)?
                    }
                    None => bf16_full.clone(),
                };

                let dtype = x.dtype();
                let w_dev = bf16_sliced_cpu.to_device(x.device())?.to_dtype(dtype)?;
                let w = w_dev.t()?;

                let out = match *x.dims() {
                    [b1, b2, m, k] => {
                        x.reshape((b1 * b2 * m, k))?
                            .matmul(&w)?
                            .reshape((b1, b2, m, ()))?
                    }
                    [bsize, m, k] => {
                        x.reshape((bsize * m, k))?
                            .matmul(&w)?
                            .reshape((bsize, m, ()))?
                    }
                    _ => x.matmul(&w)?,
                };

                match bias {
                    Some(b) => out.broadcast_add(&b.to_dtype(dtype)?),
                    None => Ok(out),
                }
            }
        }
    }
}

/// Convenience: load a bias-free Flux2Linear from `vb`. Matches the
/// `candle_nn::linear_no_bias` ergonomics it replaces.
fn flux2_linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Flux2Linear> {
    Flux2Linear::load_with_bias(in_dim, out_dim, false, vb)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Flux.2 transformer configuration.
#[derive(Debug, Clone)]
pub struct Flux2Config {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub guidance_embed: bool,
}

impl Flux2Config {
    /// Configuration for the full FLUX.2 [dev] checkpoint.
    pub fn dev() -> Self {
        Self {
            in_channels: 128,
            vec_in_dim: 0,
            context_in_dim: 15360,
            hidden_size: 6144,
            mlp_ratio: 3.0,
            num_heads: 48,
            depth: 8,
            depth_single_blocks: 48,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
            guidance_embed: true,
        }
    }

    /// Configuration for Flux.2 Klein-4B (Apache 2.0, distilled).
    pub fn klein() -> Self {
        Self {
            in_channels: 128,
            vec_in_dim: 0,
            context_in_dim: 7680,
            hidden_size: 3072,
            mlp_ratio: 3.0,
            num_heads: 24,
            depth: 5,
            depth_single_blocks: 20,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
            guidance_embed: false,
        }
    }

    /// Configuration for Flux.2 Klein-9B (Non-Commercial, distilled).
    /// Larger Qwen3 encoder (hidden_size=4096, joint_attention_dim=12288).
    pub fn klein_9b() -> Self {
        Self {
            in_channels: 128,
            vec_in_dim: 0,
            context_in_dim: 12288, // 4096 * 3 (Qwen3 hidden_size stacked 3x)
            hidden_size: 4096,
            mlp_ratio: 3.0,
            num_heads: 32,
            depth: 8,
            depth_single_blocks: 24,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
            guidance_embed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn layer_norm(dim: usize, vb: &VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

pub(crate) fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // Single dispatch point — FlashAttention / SDPA / math is selected at
    // process start via `MOLD_ATTN` and the `flash-attn` cargo feature.
    crate::attention::attention_default_scale(q, k, v)
}

pub(crate) fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let freqs = pos
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

pub(crate) fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let output_dtype = x.dtype();
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x
        .to_dtype(DType::F32)?
        .reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?
        .reshape(dims.to_vec())?
        .to_dtype(output_dtype)
}

pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    x.transpose(1, 2)?.flatten_from(2)
}

pub(crate) fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        candle_core::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// N-dimensional RoPE embedder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct EmbedNd {
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    pub(crate) fn new(theta: usize, axes_dim: Vec<usize>) -> Self {
        Self { theta, axes_dim }
    }
}

impl candle_core::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            emb.push(rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?)
        }
        Tensor::cat(&emb, 2)?.unsqueeze(1)
    }
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// MLP embedder for timestep/guidance conditioning.
#[derive(Debug, Clone)]
struct MlpEmbedder {
    in_layer: Flux2Linear,
    out_layer: Flux2Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        // Diffusers names: linear_1 / linear_2
        let in_layer = flux2_linear_no_bias(in_sz, h_sz, vb.pp("linear_1"))?;
        let out_layer = flux2_linear_no_bias(h_sz, h_sz, vb.pp("linear_2"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            in_layer: self.in_layer.to_device(device)?,
            out_layer: self.out_layer.to_device(device)?,
        })
    }
}

impl candle_core::Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: Flux2Linear,
}

impl Modulation1 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = flux2_linear_no_bias(dim, 3 * dim, vb.pp("linear"))?;
        Ok(Self { lin })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            lin: self.lin.to_device(device)?,
        })
    }

    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2 {
    lin: Flux2Linear,
}

impl Modulation2 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = flux2_linear_no_bias(dim, 6 * dim, vb.pp("linear"))?;
        Ok(Self { lin })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            lin: self.lin.to_device(device)?,
        })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        Ok((
            ModulationOut {
                shift: ys[0].clone(),
                scale: ys[1].clone(),
                gate: ys[2].clone(),
            },
            ModulationOut {
                shift: ys[3].clone(),
                scale: ys[4].clone(),
                gate: ys[5].clone(),
            },
        ))
    }
}

/// SwiGLU MLP (double-stream blocks).
#[derive(Debug, Clone)]
struct Mlp {
    lin1: Flux2Linear,
    lin2: Flux2Linear,
    mlp_sz: usize,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = flux2_linear_no_bias(in_sz, mlp_sz * 2, vb.pp("linear_in"))?;
        let lin2 = flux2_linear_no_bias(mlp_sz, in_sz, vb.pp("linear_out"))?;
        Ok(Self { lin1, lin2, mlp_sz })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            lin1: self.lin1.to_device(device)?,
            lin2: self.lin2.to_device(device)?,
            mlp_sz: self.mlp_sz,
        })
    }
}

fn mlp_bytes(mlp: &Mlp) -> usize {
    flux2_linear_bytes(&mlp.lin1) + flux2_linear_bytes(&mlp.lin2)
}

impl candle_core::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.apply(&self.lin1)?;
        let gate = x.narrow(D::Minus1, 0, self.mlp_sz)?.silu()?;
        let val = x.narrow(D::Minus1, self.mlp_sz, self.mlp_sz)?;
        (gate * val)?.apply(&self.lin2)
    }
}

// ---------------------------------------------------------------------------
// DoubleStreamBlock — joint image+text attention (diffusers naming)
// ---------------------------------------------------------------------------

/// Separate Q/K/V attention for double-stream blocks (diffusers format).
#[derive(Debug, Clone)]
struct DoubleAttention {
    to_q: Flux2Linear,
    to_k: Flux2Linear,
    to_v: Flux2Linear,
    to_out: Flux2Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    num_heads: usize,
}

impl DoubleAttention {
    /// Load image-side attention from `attn.to_q/k/v`, `attn.to_out.0`, `attn.norm_q/k`.
    fn new_img(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            to_q: flux2_linear_no_bias(dim, dim, vb.pp("to_q"))?,
            to_k: flux2_linear_no_bias(dim, dim, vb.pp("to_k"))?,
            to_v: flux2_linear_no_bias(dim, dim, vb.pp("to_v"))?,
            to_out: flux2_linear_no_bias(dim, dim, vb.pp("to_out").pp("0"))?,
            norm_q: RmsNorm::new(vb.get(head_dim, "norm_q.weight")?, 1e-6),
            norm_k: RmsNorm::new(vb.get(head_dim, "norm_k.weight")?, 1e-6),
            num_heads,
        })
    }

    /// Load text-side attention from `attn.add_q_proj`, `attn.to_add_out`, `attn.norm_added_q/k`.
    fn new_txt(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            to_q: flux2_linear_no_bias(dim, dim, vb.pp("add_q_proj"))?,
            to_k: flux2_linear_no_bias(dim, dim, vb.pp("add_k_proj"))?,
            to_v: flux2_linear_no_bias(dim, dim, vb.pp("add_v_proj"))?,
            to_out: flux2_linear_no_bias(dim, dim, vb.pp("to_add_out"))?,
            norm_q: RmsNorm::new(vb.get(head_dim, "norm_added_q.weight")?, 1e-6),
            norm_k: RmsNorm::new(vb.get(head_dim, "norm_added_k.weight")?, 1e-6),
            num_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, l, _) = xs.dims3()?;
        let q = xs
            .apply(&self.to_q)?
            .reshape((b, l, self.num_heads, ()))?
            .transpose(1, 2)?
            .apply(&self.norm_q)?;
        let k = xs
            .apply(&self.to_k)?
            .reshape((b, l, self.num_heads, ()))?
            .transpose(1, 2)?
            .apply(&self.norm_k)?;
        let v = xs
            .apply(&self.to_v)?
            .reshape((b, l, self.num_heads, ()))?
            .transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            to_q: self.to_q.to_device(device)?,
            to_k: self.to_k.to_device(device)?,
            to_v: self.to_v.to_device(device)?,
            to_out: self.to_out.to_device(device)?,
            norm_q: rms_norm_to_device(&self.norm_q, device)?,
            norm_k: rms_norm_to_device(&self.norm_k, device)?,
            num_heads: self.num_heads,
        })
    }
}

fn double_attention_bytes(attention: &DoubleAttention) -> usize {
    flux2_linear_bytes(&attention.to_q)
        + flux2_linear_bytes(&attention.to_k)
        + flux2_linear_bytes(&attention.to_v)
        + flux2_linear_bytes(&attention.to_out)
        + rms_norm_bytes(&attention.norm_q)
        + rms_norm_bytes(&attention.norm_k)
}

#[derive(Debug, Clone)]
struct DoubleStreamBlock {
    img_norm1: LayerNorm,
    img_attn: DoubleAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_attn: DoubleAttention,
    txt_norm1: LayerNorm,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let attn_vb = vb.pp("attn");
        Ok(Self {
            img_norm1: layer_norm(h_sz, &vb)?,
            img_attn: DoubleAttention::new_img(h_sz, cfg.num_heads, attn_vb.clone())?,
            img_norm2: layer_norm(h_sz, &vb)?,
            img_mlp: Mlp::new(h_sz, mlp_sz, vb.pp("ff"))?,
            txt_attn: DoubleAttention::new_txt(h_sz, cfg.num_heads, attn_vb)?,
            txt_norm1: layer_norm(h_sz, &vb)?,
            txt_norm2: layer_norm(h_sz, &vb)?,
            txt_mlp: Mlp::new(h_sz, mlp_sz, vb.pp("ff_context"))?,
        })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            img_norm1: layer_norm_to_device(&self.img_norm1, device)?,
            img_attn: self.img_attn.to_device(device)?,
            img_norm2: layer_norm_to_device(&self.img_norm2, device)?,
            img_mlp: self.img_mlp.to_device(device)?,
            txt_attn: self.txt_attn.to_device(device)?,
            txt_norm1: layer_norm_to_device(&self.txt_norm1, device)?,
            txt_norm2: layer_norm_to_device(&self.txt_norm2, device)?,
            txt_mlp: self.txt_mlp.to_device(device)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod1: &ModulationOut,
        img_mod2: &ModulationOut,
        txt_mod1: &ModulationOut,
        txt_mod2: &ModulationOut,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let img_modulated = img_mod1.scale_shift(&img.apply(&self.img_norm1)?)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt_mod1.scale_shift(&txt.apply(&self.txt_norm1)?)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&img_attn_out.apply(&self.img_attn.to_out)?))?;
        let img = (&img
            + img_mod2.gate(
                &img_mod2
                    .scale_shift(&img.apply(&self.img_norm2)?)?
                    .apply(&self.img_mlp)?,
            )?)?;

        let txt = (txt + txt_mod1.gate(&txt_attn_out.apply(&self.txt_attn.to_out)?))?;
        let txt = (&txt
            + txt_mod2.gate(
                &txt_mod2
                    .scale_shift(&txt.apply(&self.txt_norm2)?)?
                    .apply(&self.txt_mlp)?,
            )?)?;

        Ok((img, txt))
    }
}

fn double_stream_block_bytes(block: &DoubleStreamBlock) -> usize {
    layer_norm_bytes(&block.img_norm1)
        + double_attention_bytes(&block.img_attn)
        + layer_norm_bytes(&block.img_norm2)
        + mlp_bytes(&block.img_mlp)
        + double_attention_bytes(&block.txt_attn)
        + layer_norm_bytes(&block.txt_norm1)
        + layer_norm_bytes(&block.txt_norm2)
        + mlp_bytes(&block.txt_mlp)
}

// ---------------------------------------------------------------------------
// SingleStreamBlock (diffusers naming)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SingleStreamBlock {
    linear1: Flux2Linear,
    linear2: Flux2Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    pre_norm: LayerNorm,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let attn_vb = vb.pp("attn");
        // Fused: QKV (3*h_sz) + SwiGLU (2*mlp_sz) → to_qkv_mlp_proj
        let linear1 =
            flux2_linear_no_bias(h_sz, h_sz * 3 + mlp_sz * 2, attn_vb.pp("to_qkv_mlp_proj"))?;
        // Output: attn (h_sz) + mlp (mlp_sz) → to_out
        let linear2 = flux2_linear_no_bias(h_sz + mlp_sz, h_sz, attn_vb.pp("to_out"))?;
        Ok(Self {
            linear1,
            linear2,
            norm_q: RmsNorm::new(attn_vb.get(head_dim, "norm_q.weight")?, 1e-6),
            norm_k: RmsNorm::new(attn_vb.get(head_dim, "norm_k.weight")?, 1e-6),
            pre_norm: layer_norm(h_sz, &vb)?,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            linear1: self.linear1.to_device(device)?,
            linear2: self.linear2.to_device(device)?,
            norm_q: rms_norm_to_device(&self.norm_q, device)?,
            norm_k: rms_norm_to_device(&self.norm_k, device)?,
            pre_norm: layer_norm_to_device(&self.pre_norm, device)?,
            h_sz: self.h_sz,
            mlp_sz: self.mlp_sz,
            num_heads: self.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, mod_out: &ModulationOut, pe: &Tensor) -> Result<Tensor> {
        let x_mod = mod_out.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = x_mod.apply(&self.linear1)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.apply(&self.norm_q)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.apply(&self.norm_k)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp_portion = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz * 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let mlp_gate = mlp_portion.narrow(D::Minus1, 0, self.mlp_sz)?.silu()?;
        let mlp_val = mlp_portion.narrow(D::Minus1, self.mlp_sz, self.mlp_sz)?;
        let mlp_out = (mlp_gate * mlp_val)?;
        let output = Tensor::cat(&[attn, mlp_out], 2)?.apply(&self.linear2)?;
        xs + mod_out.gate(&output)
    }
}

fn single_stream_block_bytes(block: &SingleStreamBlock) -> usize {
    flux2_linear_bytes(&block.linear1)
        + flux2_linear_bytes(&block.linear2)
        + rms_norm_bytes(&block.norm_q)
        + rms_norm_bytes(&block.norm_k)
        + layer_norm_bytes(&block.pre_norm)
}

// ---------------------------------------------------------------------------
// LastLayer — final projection (diffusers: proj_out + norm_out)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LastLayer {
    norm_final: LayerNorm,
    linear: Flux2Linear,
    ada_ln_modulation: Flux2Linear,
}

impl LastLayer {
    fn new(h_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm(h_sz, &vb)?,
            linear: flux2_linear_no_bias(h_sz, out_c, vb.pp("proj_out"))?,
            ada_ln_modulation: flux2_linear_no_bias(
                h_sz,
                2 * h_sz,
                vb.pp("norm_out").pp("linear"),
            )?,
        })
    }

    fn to_device(&self, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm_to_device(&self.norm_final, device)?,
            linear: self.linear.to_device(device)?,
            ada_ln_modulation: self.ada_ln_modulation.to_device(device)?,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        // Diffusers `AdaLayerNormContinuous` convention: scale first, shift second.
        // Diffusers checkpoints store this ordering directly. BFL-native
        // single-file checkpoints store (shift, scale) but `SingleFileBackend`
        // applies `SwapHalves` when loading `norm_out.linear.weight` so the
        // weight always arrives here in diffusers (scale, shift) order.
        let (scale, shift) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        xs.apply(&self.linear)
    }
}

// ---------------------------------------------------------------------------
// Flux2Transformer — full model (diffusers format)
// ---------------------------------------------------------------------------

/// Flux.2 transformer (BF16 safetensors, diffusers naming).
///
/// Key difference from FLUX.1: modulation is shared across all blocks.
#[derive(Debug, Clone)]
pub struct Flux2Transformer {
    img_in: Flux2Linear,
    txt_in: Flux2Linear,
    time_in: MlpEmbedder,
    vector_in: Option<MlpEmbedder>,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    // Shared modulation (NOT per-block)
    double_mod_img: Modulation2,
    double_mod_txt: Modulation2,
    single_mod: Modulation1,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flux2StreamingBlock {
    Double(usize),
    Single(usize),
}

pub(crate) fn flux2_streaming_block_plan(cfg: &Flux2Config) -> Vec<Flux2StreamingBlock> {
    let mut blocks = Vec::with_capacity(cfg.depth + cfg.depth_single_blocks);
    blocks.extend((0..cfg.depth).map(Flux2StreamingBlock::Double));
    blocks.extend((0..cfg.depth_single_blocks).map(Flux2StreamingBlock::Single));
    blocks
}

enum DoubleBlockSlot {
    Resident(DoubleStreamBlock),
    Streamed(DoubleStreamBlock),
}

enum SingleBlockSlot {
    Resident(SingleStreamBlock),
    Streamed(SingleStreamBlock),
}

fn is_probable_cuda_oom(err: &candle_core::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("cuda_error_out_of_memory")
        || msg.contains("out of memory")
        || msg.contains("memory allocation")
}

fn materialize_flux2_block_slots(
    double_blocks: &[DoubleStreamBlock],
    single_blocks: &[SingleStreamBlock],
    plan: &AdaptiveResidencyPlan,
    device: &candle_core::Device,
) -> Result<(Vec<DoubleBlockSlot>, Vec<SingleBlockSlot>)> {
    let mut double_slots = Vec::with_capacity(double_blocks.len());
    for (i, block) in double_blocks.iter().enumerate() {
        if plan.resident.get(i).copied().unwrap_or(false) {
            double_slots.push(DoubleBlockSlot::Resident(block.to_device(device)?));
        } else {
            double_slots.push(DoubleBlockSlot::Streamed(block.clone()));
        }
    }

    let single_offset = double_blocks.len();
    let mut single_slots = Vec::with_capacity(single_blocks.len());
    for (i, block) in single_blocks.iter().enumerate() {
        if plan
            .resident
            .get(single_offset + i)
            .copied()
            .unwrap_or(false)
        {
            single_slots.push(SingleBlockSlot::Resident(block.to_device(device)?));
        } else {
            single_slots.push(SingleBlockSlot::Streamed(block.clone()));
        }
    }

    Ok((double_slots, single_slots))
}

pub(crate) struct OffloadedFlux2Transformer {
    block_plan: Vec<Flux2StreamingBlock>,
    img_in: Flux2Linear,
    txt_in: Flux2Linear,
    time_in: MlpEmbedder,
    vector_in: Option<MlpEmbedder>,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double_mod_img: Modulation2,
    double_mod_txt: Modulation2,
    single_mod: Modulation1,
    double_blocks: Vec<DoubleBlockSlot>,
    single_blocks: Vec<SingleBlockSlot>,
    final_layer: LastLayer,
    device: candle_core::Device,
}

impl OffloadedFlux2Transformer {
    pub(crate) fn new(
        cfg: &Flux2Config,
        cpu_vb: VarBuilder,
        device: &candle_core::Device,
        gpu_ordinal: usize,
        activation_budget: u64,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        let block_plan = flux2_streaming_block_plan(cfg);
        let dense = Flux2Transformer::new(cfg, cpu_vb)?;
        Self::from_dense(
            dense,
            block_plan,
            device,
            gpu_ordinal,
            activation_budget,
            progress,
        )
    }

    fn from_dense(
        dense: Flux2Transformer,
        block_plan: Vec<Flux2StreamingBlock>,
        device: &candle_core::Device,
        gpu_ordinal: usize,
        activation_budget: u64,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        let Flux2Transformer {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_mod_img,
            double_mod_txt,
            single_mod,
            double_blocks,
            single_blocks,
            final_layer,
        } = dense;

        let img_in = img_in.to_device(device)?;
        let txt_in = txt_in.to_device(device)?;
        let time_in = time_in.to_device(device)?;
        let vector_in = vector_in
            .as_ref()
            .map(|embedder| embedder.to_device(device))
            .transpose()?;
        let guidance_in = guidance_in
            .as_ref()
            .map(|embedder| embedder.to_device(device))
            .transpose()?;
        let double_mod_img = double_mod_img.to_device(device)?;
        let double_mod_txt = double_mod_txt.to_device(device)?;
        let single_mod = single_mod.to_device(device)?;
        let final_layer = final_layer.to_device(device)?;

        let mut block_sizes = Vec::with_capacity(double_blocks.len() + single_blocks.len());
        block_sizes.extend(double_blocks.iter().map(double_stream_block_bytes));
        block_sizes.extend(single_blocks.iter().map(single_stream_block_bytes));

        let free_vram = crate::device::usable_free_vram_bytes(gpu_ordinal).unwrap_or(0);
        let mut plan = plan_adaptive_residency(
            &block_sizes,
            free_vram,
            // No weights outside the block set on this path.
            0,
            activation_budget,
            ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
        );

        let (double_blocks, single_blocks, plan) = loop {
            match materialize_flux2_block_slots(&double_blocks, &single_blocks, &plan, device) {
                Ok((double_slots, single_slots)) => break (double_slots, single_slots, plan),
                Err(err)
                    if device.is_cuda()
                        && plan.resident_count() > 0
                        && is_probable_cuda_oom(&err) =>
                {
                    progress.info(&format!(
                        "Flux.2 adaptive offload: resident allocation OOM at {} resident blocks; \
                         retrying with fewer resident blocks",
                        plan.resident_count()
                    ));
                    if let Err(sync_err) = device.synchronize() {
                        tracing::warn!(
                            "Flux.2 adaptive offload: synchronize after OOM failed: {sync_err}"
                        );
                    }
                    if !plan.demote_largest_resident(&block_sizes) {
                        return Err(err);
                    }
                }
                Err(err) => return Err(err),
            }
        };

        progress.info(&format!(
            "Flux.2 adaptive offload: {} resident / {} streamed blocks \
             (resident {:.2} GB, streamed {:.2} GB per denoise pass, reserve {:.2} GB)",
            plan.resident_count(),
            plan.streamed_count(),
            plan.resident_bytes as f64 / 1_000_000_000.0,
            plan.streamed_bytes as f64 / 1_000_000_000.0,
            plan.reserved_bytes() as f64 / 1_000_000_000.0,
        ));

        Ok(Self {
            block_plan,
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_mod_img,
            double_mod_txt,
            single_mod,
            double_blocks,
            single_blocks,
            final_layer,
            device: device.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 || img.rank() != 3 {
            candle_core::bail!("expected rank 3, got txt={} img={}", txt.rank(), img.rank())
        }
        let device = &self.device;
        let dtype = img.dtype();
        let img = img.to_device(device)?;
        let txt = txt.to_device(device)?;
        let img_ids = img_ids.to_device(device)?;
        let txt_ids = txt_ids.to_device(device)?;
        let timesteps = timesteps.to_device(device)?;
        let y = y.to_device(device)?;
        let guidance = guidance.map(|g| g.to_device(device)).transpose()?;
        let pe = {
            let ids = Tensor::cat(&[&txt_ids, &img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let mut vec_ = timestep_embedding(&timesteps, 256, dtype)?.apply(&self.time_in)?;

        if let (Some(g_in), Some(guidance)) = (self.guidance_in.as_ref(), guidance.as_ref()) {
            vec_ = (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?;
        }
        if let Some(vec_in) = self.vector_in.as_ref() {
            vec_ = (vec_ + y.apply(vec_in))?;
        }

        let (img_mod1, img_mod2) = self.double_mod_img.forward(&vec_)?;
        let (txt_mod1, txt_mod2) = self.double_mod_txt.forward(&vec_)?;
        debug_assert_eq!(
            self.block_plan.len(),
            self.double_blocks.len() + self.single_blocks.len()
        );

        for block in &self.double_blocks {
            match block {
                DoubleBlockSlot::Resident(block) => {
                    (img, txt) = block
                        .forward(&img, &txt, &img_mod1, &img_mod2, &txt_mod1, &txt_mod2, &pe)?;
                }
                DoubleBlockSlot::Streamed(block) => {
                    let block = block.to_device(device)?;
                    (img, txt) = block
                        .forward(&img, &txt, &img_mod1, &img_mod2, &txt_mod1, &txt_mod2, &pe)?;
                    device.synchronize()?;
                    drop(block);
                }
            }
        }

        let single_mod = self.single_mod.forward(&vec_)?;
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single_blocks {
            match block {
                SingleBlockSlot::Resident(block) => {
                    img = block.forward(&img, &single_mod, &pe)?;
                }
                SingleBlockSlot::Streamed(block) => {
                    let block = block.to_device(device)?;
                    img = block.forward(&img, &single_mod, &pe)?;
                    device.synchronize()?;
                    drop(block);
                }
            }
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

impl Flux2Transformer {
    pub fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let img_in = flux2_linear_no_bias(cfg.in_channels, cfg.hidden_size, vb.pp("x_embedder"))?;
        let txt_in = flux2_linear_no_bias(
            cfg.context_in_dim,
            cfg.hidden_size,
            vb.pp("context_embedder"),
        )?;

        let time_in = MlpEmbedder::new(
            256,
            cfg.hidden_size,
            vb.pp("time_guidance_embed").pp("timestep_embedder"),
        )?;

        let vector_in = if cfg.vec_in_dim > 0 {
            Some(MlpEmbedder::new(
                cfg.vec_in_dim,
                cfg.hidden_size,
                vb.pp("vector_in"),
            )?)
        } else {
            None
        };

        let guidance_in = if cfg.guidance_embed {
            Some(MlpEmbedder::new(
                256,
                cfg.hidden_size,
                vb.pp("time_guidance_embed").pp("guidance_embedder"),
            )?)
        } else {
            None
        };

        // Shared modulation layers
        let double_mod_img =
            Modulation2::new(cfg.hidden_size, vb.pp("double_stream_modulation_img"))?;
        let double_mod_txt =
            Modulation2::new(cfg.hidden_size, vb.pp("double_stream_modulation_txt"))?;
        let single_mod = Modulation1::new(cfg.hidden_size, vb.pp("single_stream_modulation"))?;

        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("transformer_blocks");
        for idx in 0..cfg.depth {
            double_blocks.push(DoubleStreamBlock::new(cfg, vb_d.pp(idx))?);
        }

        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_transformer_blocks");
        for idx in 0..cfg.depth_single_blocks {
            single_blocks.push(SingleStreamBlock::new(cfg, vb_s.pp(idx))?);
        }

        let final_layer = LastLayer::new(cfg.hidden_size, cfg.in_channels, vb.clone())?;
        let pe_embedder = EmbedNd::new(cfg.theta, cfg.axes_dim.to_vec());

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_mod_img,
            double_mod_txt,
            single_mod,
            double_blocks,
            single_blocks,
            final_layer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 || img.rank() != 3 {
            candle_core::bail!("expected rank 3, got txt={} img={}", txt.rank(), img.rank())
        }
        let dtype = img.dtype();
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;
        let mut vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;

        if let (Some(g_in), Some(guidance)) = (self.guidance_in.as_ref(), guidance) {
            vec_ = (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?;
        }
        if let Some(vec_in) = self.vector_in.as_ref() {
            vec_ = (vec_ + y.apply(vec_in))?;
        }

        // Shared modulation: compute once, reuse for all blocks
        let (img_mod1, img_mod2) = self.double_mod_img.forward(&vec_)?;
        let (txt_mod1, txt_mod2) = self.double_mod_txt.forward(&vec_)?;

        for block in &self.double_blocks {
            (img, txt) =
                block.forward(&img, &txt, &img_mod1, &img_mod2, &txt_mod1, &txt_mod2, &pe)?;
        }

        let single_mod = self.single_mod.forward(&vec_)?;
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single_blocks {
            img = block.forward(&img, &single_mod, &pe)?;
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

// ---------------------------------------------------------------------------
// Wrapper enum for BF16 and GGUF quantized
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
pub(crate) enum Flux2TransformerWrapper {
    BF16(Flux2Transformer),
    Offloaded(OffloadedFlux2Transformer),
    Quantized(super::quantized_transformer::QuantizedFlux2Transformer),
}

/// The unconditional branch of a classifier-free-guided FLUX.2 render.
///
/// Only the undistilled [klein] base checkpoints use one: they carry no
/// guidance embedding, so `guidance` can only reach the render as a real CFG
/// scale over a second forward. `diffusers`'
/// `pipeline_flux2_klein.py:747-753,862-875` is the contract — the negative
/// prompt defaults to `""`, both branches see the identical latent and ids,
/// and the combination is `neg + scale * (pos - neg)`.
pub(crate) struct Flux2CfgBranch<'a> {
    /// `guidance_scale`; the branch is constructed only when it exceeds 1.
    pub scale: f64,
    /// The negative prompt's encoder hidden states.
    pub txt: &'a Tensor,
    pub txt_ids: &'a Tensor,
}

impl Flux2TransformerWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        reference: Option<(&Tensor, &Tensor)>,
        txt: &Tensor,
        txt_ids: &Tensor,
        vec_: &Tensor,
        timesteps: &[f64],
        guidance: f64,
        progress: &crate::progress::ProgressReporter,
        inpaint_ctx: Option<&crate::img_utils::InpaintContext>,
        preview: Option<&crate::latent_preview::LatentPreviewer>,
        cfg: Option<&Flux2CfgBranch<'_>>,
    ) -> anyhow::Result<Tensor> {
        use crate::progress::ProgressEvent;
        use std::time::Instant;

        let b_sz = img.dim(0)?;
        let dev = img.device();
        let guidance_tensor = Tensor::full(guidance as f32, b_sz, dev)?;
        let mut img = img.clone();
        let total_steps = timesteps.len().saturating_sub(1);

        for (step, window) in timesteps.windows(2).enumerate() {
            progress.checkpoint()?;
            let step_start = Instant::now();
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;

            let (model_img, model_img_ids) = if let Some((reference_img, reference_ids)) = reference
            {
                (
                    Tensor::cat(&[&img, reference_img], 1)?,
                    Tensor::cat(&[img_ids, reference_ids], 1)?,
                )
            } else {
                (img.clone(), img_ids.clone())
            };
            let pred = self.forward_once(
                &model_img,
                &model_img_ids,
                txt,
                txt_ids,
                &t_vec,
                vec_,
                &guidance_tensor,
            )?;
            // The unconditional branch. `None` leaves `pred` exactly what the
            // single-forward path produced — the same conditioning, the same
            // call — so a distilled render is byte-identical to one made
            // before this branch existed.
            let pred = match cfg {
                Some(branch) => {
                    let neg = self.forward_once(
                        &model_img,
                        &model_img_ids,
                        branch.txt,
                        branch.txt_ids,
                        &t_vec,
                        vec_,
                        &guidance_tensor,
                    )?;
                    // `pipeline_flux2_klein.py:875`:
                    // `noise_pred = neg + guidance_scale * (noise_pred - neg)`
                    (&neg + ((&pred - &neg)? * branch.scale)?)?
                }
                None => pred,
            };
            let pred = pred.narrow(1, 0, img.dim(1)?)?;
            img = (img + &pred * (t_prev - t_curr))?;

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ctx) = inpaint_ctx {
                img = crate::img2img::apply_flow_match_inpaint(&img, ctx, *t_prev)?;
            }

            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: total_steps,
                elapsed: step_start.elapsed(),
            });
            if let Some(previewer) = preview {
                if previewer.due(step + 1, total_steps) {
                    // Preview the predicted clean image (flow matching:
                    // x0 = x_t - t*v), not the still-noisy working latent —
                    // composition is visible from the first step.
                    match &img - &(&pred * *t_prev)? {
                        Ok(x0_est) => {
                            previewer.maybe_emit(progress, &x0_est, step + 1, total_steps)
                        }
                        Err(e) => tracing::warn!("skipping denoise preview: {e}"),
                    }
                }
            }
        }
        progress.checkpoint()?;
        Ok(img)
    }

    /// One transformer forward, routed to whichever of the three arms this is.
    ///
    /// Extracted so the conditional pass and the CFG unconditional pass are
    /// the SAME call rather than two hand-kept copies of a three-arm match:
    /// the two must differ only in their text conditioning.
    #[allow(clippy::too_many_arguments)]
    fn forward_once(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        t_vec: &Tensor,
        vec_: &Tensor,
        guidance: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let pred = match self {
            Self::BF16(m) => m.forward(img, img_ids, txt, txt_ids, t_vec, vec_, Some(guidance))?,
            Self::Offloaded(m) => {
                m.forward(img, img_ids, txt, txt_ids, t_vec, vec_, Some(guidance))?
            }
            Self::Quantized(m) => {
                m.forward(img, img_ids, txt, txt_ids, t_vec, vec_, Some(guidance))?
            }
        };
        Ok(pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Denoise the tiny synthetic transformer for one step, optionally with an
    /// unconditional branch. Returns the resulting latent as f32.
    fn denoise_once(
        wrapper: &Flux2TransformerWrapper,
        guidance: f64,
        cfg: Option<(&Tensor, &Tensor, f64)>,
    ) -> Vec<f32> {
        use crate::flux2::quantized_transformer::test_support::spread;
        let device = candle_core::Device::Cpu;
        let txt = spread((2, 6), 3.2).reshape((1, 2, 6)).unwrap();
        let txt_ids = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();
        denoise_once_with_text(wrapper, guidance, &txt, &txt_ids, cfg)
    }

    /// As `denoise_once`, with the positive conditioning supplied.
    fn denoise_once_with_text(
        wrapper: &Flux2TransformerWrapper,
        guidance: f64,
        txt: &Tensor,
        txt_ids: &Tensor,
        cfg: Option<(&Tensor, &Tensor, f64)>,
    ) -> Vec<f32> {
        use crate::flux2::quantized_transformer::test_support::spread;
        let device = candle_core::Device::Cpu;
        let img = spread((3, 4), 3.1).reshape((1, 3, 4)).unwrap();
        let img_ids = Tensor::zeros((1, 3, 4), DType::F32, &device).unwrap();
        let vec_ = Tensor::zeros((1, 1), DType::F32, &device).unwrap();
        let branch = cfg.map(|(txt, txt_ids, scale)| Flux2CfgBranch {
            scale,
            txt,
            txt_ids,
        });
        let progress = crate::progress::ProgressReporter::default();
        wrapper
            .denoise(
                &img,
                &img_ids,
                None,
                txt,
                txt_ids,
                &vec_,
                &[1.0, 0.0],
                guidance,
                &progress,
                None,
                None,
                branch.as_ref(),
            )
            .expect("denoise")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// A render that asks for no unconditional branch must take exactly the
    /// path it took before the branch existed — same conditioning, same single
    /// forward. Pinned by construction: `forward_once` is the only caller.
    #[test]
    fn a_render_without_a_cfg_branch_is_unchanged_by_a_scale_of_one() {
        use crate::flux2::quantized_transformer::test_support::{
            spread, tiny_cfg, tiny_transformer,
        };
        let cfg = tiny_cfg(false);
        let wrapper = Flux2TransformerWrapper::Quantized(tiny_transformer(&cfg));
        let device = candle_core::Device::Cpu;

        let plain = denoise_once(&wrapper, 1.0, None);
        // The SAME conditioning on both branches at scale 1.0:
        // neg + 1.0 * (pos - neg) == pos.
        let txt = spread((2, 6), 3.2).reshape((1, 2, 6)).unwrap();
        let txt_ids = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();
        let inert = denoise_once(&wrapper, 1.0, Some((&txt, &txt_ids, 1.0)));

        assert_eq!(plain.len(), inert.len());
        for (a, b) in plain.iter().zip(&inert) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// The combination is `neg + scale * (pos - neg)`
    /// (`pipeline_flux2_klein.py:875`), and this pins it exactly rather than
    /// settling for "the render changed". Over one step the update is
    /// `img - pred`, so a conditional-only run, an unconditional-only run and
    /// a guided run are related by that same lerp — with the sign inverted,
    /// or the branches swapped, the identity fails.
    #[test]
    fn the_cfg_combination_is_the_upstream_lerp_between_the_two_branches() {
        use crate::flux2::quantized_transformer::test_support::{
            spread, tiny_cfg, tiny_transformer,
        };
        let cfg = tiny_cfg(false);
        let wrapper = Flux2TransformerWrapper::Quantized(tiny_transformer(&cfg));
        let device = candle_core::Device::Cpu;
        // The positive conditioning `denoise_once` uses, and a different
        // negative one.
        let pos_txt = spread((2, 6), 3.2).reshape((1, 2, 6)).unwrap();
        let neg_txt = spread((2, 6), 4.9).reshape((1, 2, 6)).unwrap();
        let ids = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();

        let scale = 3.0_f32;
        // Conditional only, unconditional only, and the guided render.
        let conditional = denoise_once(&wrapper, 1.0, None);
        let unconditional = denoise_once_with_text(&wrapper, 1.0, &neg_txt, &ids, None);
        let guided = denoise_once(&wrapper, 1.0, Some((&neg_txt, &ids, scale as f64)));

        assert!(
            conditional
                .iter()
                .zip(&unconditional)
                .any(|(a, b)| (a - b).abs() > 1e-5),
            "the two conditionings must actually differ, or the identity is vacuous"
        );
        for ((got, uncond), cond) in guided.iter().zip(&unconditional).zip(&conditional) {
            let want = uncond + scale * (cond - uncond);
            assert!(
                (got - want).abs() < 1e-3,
                "guided {got} != neg + {scale} * (pos - neg) = {want}",
            );
        }
        // And the positive conditioning is the one that was guided toward,
        // not the negative: this is what an accidental swap breaks.
        assert!(pos_txt
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(neg_txt.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .any(|(a, b)| (a - b).abs() > 1e-6),);
    }

    #[test]
    fn klein_config_dimensions() {
        let cfg = Flux2Config::klein();
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.hidden_size / cfg.num_heads, 128); // head_dim
        assert_eq!(cfg.depth, 5);
        assert_eq!(cfg.depth_single_blocks, 20);
        assert_eq!(cfg.axes_dim, vec![32, 32, 32, 32]);
        assert_eq!(cfg.theta, 2000);
        assert!(!cfg.guidance_embed); // distilled
    }

    #[test]
    fn klein_mlp_sizes() {
        let cfg = Flux2Config::klein();
        let h_sz = cfg.hidden_size; // 3072
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize; // 9216
        assert_eq!(mlp_sz, 9216);
        // Double-stream MLP: lin1 = (h_sz, 2*mlp_sz), lin2 = (mlp_sz, h_sz)
        assert_eq!(h_sz * 3 + mlp_sz * 2, 27648); // single fused projection
        assert_eq!(h_sz + mlp_sz, 12288); // single output projection
    }

    #[test]
    fn klein_context_dim_matches_qwen3() {
        let cfg = Flux2Config::klein();
        // Qwen3 hidden_size=2560, stacked 3 layers = 7680
        assert_eq!(cfg.context_in_dim, 7680);
        assert_eq!(cfg.context_in_dim, 2560 * 3);
    }

    #[test]
    fn klein_9b_config_dimensions() {
        let cfg = Flux2Config::klein_9b();
        assert_eq!(cfg.in_channels, 128);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.hidden_size / cfg.num_heads, 128); // head_dim
        assert_eq!(cfg.depth, 8);
        assert_eq!(cfg.depth_single_blocks, 24);
        assert_eq!(cfg.context_in_dim, 12288);
        assert_eq!(cfg.context_in_dim, 4096 * 3); // Qwen3 hidden_size=4096, stacked 3x
        assert!(!cfg.guidance_embed); // distilled
    }

    #[test]
    fn flux2_streaming_block_plan_preserves_reference_order() {
        let mut cfg = Flux2Config::klein();
        cfg.depth = 2;
        cfg.depth_single_blocks = 3;

        assert_eq!(
            flux2_streaming_block_plan(&cfg),
            vec![
                Flux2StreamingBlock::Double(0),
                Flux2StreamingBlock::Double(1),
                Flux2StreamingBlock::Single(0),
                Flux2StreamingBlock::Single(1),
                Flux2StreamingBlock::Single(2),
            ]
        );
    }

    #[test]
    fn timestep_embedding_shape() {
        let dev = candle_core::Device::Cpu;
        let t = Tensor::full(0.5f32, 2, &dev).unwrap();
        let emb = timestep_embedding(&t, 256, DType::F32).unwrap();
        assert_eq!(emb.dims(), &[2, 256]);
    }

    #[test]
    fn rope_4d_shape() {
        let dev = candle_core::Device::Cpu;
        let pos = Tensor::zeros((1, 16), DType::F32, &dev).unwrap();
        let r = rope(&pos, 32, 2000).unwrap();
        assert_eq!(r.dims(), &[1, 16, 16, 2, 2]);
    }

    #[test]
    fn test_timestep_embedding_dtype_preserved() {
        let dev = candle_core::Device::Cpu;
        let t = Tensor::full(0.5f32, 2, &dev).unwrap();
        let emb = timestep_embedding(&t, 128, DType::BF16).unwrap();
        assert_eq!(emb.dtype(), DType::BF16);
        assert_eq!(emb.dims(), &[2, 128]);
    }

    #[test]
    fn test_timestep_embedding_values_bounded() {
        let dev = candle_core::Device::Cpu;
        let t = Tensor::full(0.7f32, 1, &dev).unwrap();
        let emb = timestep_embedding(&t, 64, DType::F32).unwrap();
        let flat = emb.flatten_all().unwrap();
        let vals: Vec<f32> = flat.to_vec1().unwrap();
        for v in &vals {
            assert!(
                *v >= -1.0 && *v <= 1.0,
                "embedding value {v} outside [-1, 1] (sin/cos bounds)"
            );
        }
    }

    #[test]
    fn test_rope_odd_dim_fails() {
        let dev = candle_core::Device::Cpu;
        let pos = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let result = rope(&pos, 33, 2000);
        assert!(result.is_err(), "rope with odd dim should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("odd"),
            "error should mention 'odd', got: {err_msg}"
        );
    }

    #[test]
    fn flux2_linear_standard_bf16_forward() {
        // Sanity: the BF16 (Standard) branch behaves like a stock Linear.
        // weight = [[1.0, 2.0], [3.0, 4.0]]  → out = x @ weight.t()
        // bias=None. Check via Module::forward.
        let dev = candle_core::Device::Cpu;
        let weight = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).unwrap();
        let lin = Flux2Linear::Standard(candle_nn::Linear::new(weight, None));
        let x = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let out = lin.forward(&x).unwrap();
        // x[0] * weight.t() col 0 = 1*1 + 0*2 = 1; col 1 = 1*3 + 0*4 = 3
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1.0, 3.0]);
    }

    #[test]
    fn flux2_linear_fp8_forward_matches_bf16_reference() {
        // FP8 path: stored W is FP8(2.0). Input x = [[1, 1, 1, 1]].
        // x @ W.t() with W = ones·2 (out=2, in=4) → [[8, 8]]. No sidecar
        // scale — community FP8 conversions without an NVFP4 ancestor.
        // Tests run F32 → F32 because candle CPU has no BF16 matmul.
        let dev = candle_core::Device::Cpu;
        let weight = Tensor::from_vec(vec![2.0f32; 8], (2, 4), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let lin = Flux2Linear::Fp8 {
            weight,
            scale: None,
            bias: None,
        };
        let x = Tensor::from_vec(vec![1.0f32; 4], (1, 4), &dev).unwrap();
        let out = lin.forward(&x).unwrap();
        assert_eq!(
            out.dtype(),
            DType::F32,
            "FP8 forward must preserve activation dtype",
        );
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for x in &v {
            assert!((x - 8.0).abs() < 1e-3, "got {x}, want 8.0");
        }
    }

    /// BFL's own FP8 conversions store `weight / weight_scale` and ship the
    /// scale beside it. Ignoring it does not blur the render, it destroys it:
    /// a scale of 0.002 makes every weight ~500x too large.
    #[test]
    fn flux2_linear_fp8_applies_a_scalar_dequantization_scale() {
        let dev = candle_core::Device::Cpu;
        let weight = Tensor::from_vec(vec![2.0f32; 8], (2, 4), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let scale = Tensor::new(0.25f32, &dev).unwrap();
        assert_eq!(scale.rank(), 0, "per-tensor scale is a scalar");
        let lin = Flux2Linear::Fp8 {
            weight,
            scale: Some(scale),
            bias: None,
        };
        let x = Tensor::from_vec(vec![1.0f32; 4], (1, 4), &dev).unwrap();
        let v: Vec<f32> = lin
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // 4 inputs x (2.0 * 0.25) = 2.0 per output row.
        for got in &v {
            assert!((got - 2.0).abs() < 1e-3, "got {got}, want 2.0");
        }
    }

    /// The scalar scale rides on the output for cost, so it must still agree
    /// with the weight-side arithmetic a per-row scale takes.
    #[test]
    fn flux2_linear_fp8_scalar_and_per_row_scales_agree() {
        let dev = candle_core::Device::Cpu;
        let raw: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let weight = Tensor::from_vec(raw, (2, 4), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();

        let scalar = Flux2Linear::Fp8 {
            weight: weight.clone(),
            scale: Some(Tensor::new(0.5f32, &dev).unwrap()),
            bias: None,
        };
        // The same 0.5, expressed per output row — this takes the weight-side
        // branch instead.
        let per_row = Flux2Linear::Fp8 {
            weight,
            scale: Some(Tensor::from_vec(vec![0.5f32; 2], (2, 1), &dev).unwrap()),
            bias: None,
        };

        let a: Vec<f32> = scalar
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = per_row
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(a.len(), b.len());
        for (got, want) in a.iter().zip(&b) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn flux2_linear_fp8_basic_matmul() {
        // Simplest FP8 forward smoke test: weight [[3.0]], bias=None,
        // x=[[2.0]]. Out = 2*3 = 6.
        let dev = candle_core::Device::Cpu;
        let weight = Tensor::from_vec(vec![3.0f32], (1, 1), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let lin = Flux2Linear::Fp8 {
            weight,
            scale: None,
            bias: None,
        };
        let x = Tensor::from_vec(vec![2.0f32], (1, 1), &dev).unwrap();
        let out = lin.forward(&x).unwrap();
        let v: f32 = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((v - 6.0).abs() < 1e-3);
    }

    #[test]
    fn flux2_linear_fp8_applies_bias_after_matmul() {
        // weight = [[1.0]], bias = [10.0], x = [[3]]. x@w.t()+bias = 13.
        let dev = candle_core::Device::Cpu;
        let weight = Tensor::from_vec(vec![1.0f32], (1, 1), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let bias = Tensor::from_vec(vec![10.0f32], 1, &dev).unwrap();
        let lin = Flux2Linear::Fp8 {
            weight,
            scale: None,
            bias: Some(bias),
        };
        let x = Tensor::from_vec(vec![3.0f32], (1, 1), &dev).unwrap();
        let out = lin.forward(&x).unwrap();
        let v: f32 = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((v - 13.0).abs() < 1e-2);
    }

    #[test]
    fn flux2_linear_fp8_applies_sidecar_scale_at_forward() {
        // Verify the NVFP4 sidecar tensor_scale path: weight stored at FP8(2.0)
        // pre-scale, scale_weight = [0.5] (1-D, length 1). Forward must
        // broadcast-multiply every weight element by 0.5 → effective weight
        // 1.0, so x @ W.t() with x=[1,1,1,1] and W=ones [2,4] yields [4,4].
        let dev = candle_core::Device::Cpu;
        let weight = Tensor::from_vec(vec![2.0f32; 8], (2, 4), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let scale = Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap();
        let lin = Flux2Linear::Fp8 {
            weight,
            scale: Some(scale),
            bias: None,
        };
        let x = Tensor::from_vec(vec![1.0f32; 4], (1, 4), &dev).unwrap();
        let out = lin.forward(&x).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for x in &v {
            assert!(
                (x - 4.0).abs() < 1e-3,
                "got {x}, want 4.0 (sidecar scale 0.5 applied to FP8(2.0))",
            );
        }
    }

    /// Build the canonical 1-block NVFP4 fixture used by the streaming tests:
    /// N rows × 16 columns of weight = +1.0 (E2M1 nibble 0b0010), one block
    /// scale per row at E4M3 = 1.0 (byte 0x38), per-tensor scale applied at
    /// forward time. block_scales are stored in cuBLAS SWIZZLE_32_4_4 tiled
    /// layout to match what ComfyUI's NVFP4 converter writes: padded to
    /// (roundup(N, 128), roundup(1, 4)) = (128, 4).
    /// Returns (packed U8 [N, 8], block_scales F8E4M3 [128, 4]).
    fn nvfp4_unit_fixture(n_rows: usize) -> (Tensor, Tensor) {
        use crate::nvfp4::swizzle_block_scales;
        use candle_core::Device;
        let dev = Device::Cpu;
        let packed_bytes = vec![0x22u8; n_rows * 8];
        let packed = Tensor::from_vec(packed_bytes, (n_rows, 8), &dev).unwrap();
        // Natural block-scale layout: [n_rows, 1], all 1.0.
        let natural_scales: Vec<f32> = vec![1.0f32; n_rows];
        let swizzled = swizzle_block_scales(&natural_scales, n_rows, 1).unwrap();
        let padded_rows = n_rows.div_ceil(128) * 128;
        let padded_cols = 4;
        let scales_f32 = Tensor::from_vec(swizzled, (padded_rows, padded_cols), &dev).unwrap();
        let block_scales = scales_f32.to_dtype(DType::F8E4M3).unwrap();
        (packed, block_scales)
    }

    #[test]
    fn flux2_linear_nvfp4_streaming_round_trip_matches_standard() {
        // Streaming path must produce the same output as a Standard linear
        // built from the dequanted weight. Tolerance is BF16-level (~1e-3).
        // 4 output rows × 16 input columns. Each weight element = +1.0
        // (FP4 1.0 × block_scale 1.0). Per-tensor scale = 0.25 → effective
        // weight = 0.25. Input = ones. Out = sum(weights row) = 16 * 0.25 = 4.
        let dev = candle_core::Device::Cpu;
        let n_full = 4;
        let k = 16;
        let tensor_scale = 0.25f32;
        let (packed, block_scales) = nvfp4_unit_fixture(n_full);

        let streaming = Flux2Linear::Nvfp4Streaming {
            packed: packed.clone(),
            block_scales: block_scales.clone(),
            tensor_scale,
            out_dim: n_full,
            in_dim: k,
            slice: None,
            bias: None,
            cache: Arc::new(OnceLock::new()),
        };

        // Reference: full BF16 dequanted weight (all 0.25), as a Standard
        // candle Linear.
        let ref_w = Tensor::from_vec(vec![tensor_scale; n_full * k], (n_full, k), &dev).unwrap();
        let ref_lin = Flux2Linear::Standard(candle_nn::Linear::new(ref_w, None));

        let x = Tensor::from_vec(vec![1.0f32; k], (1, k), &dev).unwrap();
        let out_streaming = streaming.forward(&x).unwrap();
        let out_ref = ref_lin.forward(&x).unwrap();

        let s: Vec<f32> = out_streaming
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let r: Vec<f32> = out_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(s.len(), r.len());
        for (i, (a, b)) in s.iter().zip(r.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "streaming[{i}]={a}, reference={b} — diverged beyond BF16 tolerance",
            );
            assert!(
                (a - 4.0).abs() < 1e-2,
                "streaming[{i}]={a}, want 4.0 (sum of 16 × 0.25)",
            );
        }
    }

    #[test]
    fn flux2_linear_nvfp4_streaming_caches_bf16() {
        // After one forward, the OnceLock cache must be populated. This
        // ensures the second forward bypasses the f32 dequant pass.
        let dev = candle_core::Device::Cpu;
        let n_full = 2;
        let k = 16;
        let (packed, block_scales) = nvfp4_unit_fixture(n_full);
        let cache = Arc::new(OnceLock::new());
        let streaming = Flux2Linear::Nvfp4Streaming {
            packed,
            block_scales,
            tensor_scale: 1.0,
            out_dim: n_full,
            in_dim: k,
            slice: None,
            bias: None,
            cache: cache.clone(),
        };
        assert!(cache.get().is_none(), "cache empty before first forward");

        let x = Tensor::from_vec(vec![1.0f32; k], (1, k), &dev).unwrap();
        let _ = streaming.forward(&x).unwrap();
        assert!(
            cache.get().is_some(),
            "cache must be populated after first forward",
        );
        let cached = cache.get().unwrap();
        assert_eq!(cached.dtype(), DType::BF16);
        assert_eq!(cached.dims(), &[n_full, k]);
    }

    #[test]
    fn flux2_linear_nvfp4_streaming_slice_q_k_v_share_cache() {
        // Three sliced linears (`to_q`/`to_k`/`to_v`) sharing one
        // `Arc<OnceLock<Tensor>>` populate the cache exactly once and each
        // produces output matching an independently-built reference Linear
        // sliced from the same dequanted weight.
        //
        // Build a 3*N=6 row fused weight (Q/K/V each contribute 2 rows of
        // 16 columns). Q rows = 0.25 (all FP4 1.0, scale 1.0, t_scale 0.25).
        // The reference Linear is sliced from a manually-constructed BF16
        // weight matching what dequant produces.
        let dev = candle_core::Device::Cpu;
        let out_dim = 2;
        let n_full = out_dim * 3; // 6 rows for fused QKV
        let k = 16;
        let tensor_scale = 0.25f32;
        let (packed, block_scales) = nvfp4_unit_fixture(n_full);

        let shared_cache = Arc::new(OnceLock::new());
        let mut linears = Vec::with_capacity(3);
        for component in 0..3 {
            linears.push(Flux2Linear::Nvfp4Streaming {
                packed: packed.clone(),
                block_scales: block_scales.clone(),
                tensor_scale,
                out_dim,
                in_dim: k,
                slice: Some((0, component, 3)),
                bias: None,
                cache: shared_cache.clone(),
            });
        }
        assert!(
            shared_cache.get().is_none(),
            "shared cache empty before any forward",
        );

        let x = Tensor::from_vec(vec![1.0f32; k], (1, k), &dev).unwrap();
        // Forward Q. After this, the shared cache is populated; K and V both
        // hit the warm cache.
        let out_q = linears[0].forward(&x).unwrap();
        assert!(
            shared_cache.get().is_some(),
            "Q-forward must populate cache"
        );
        let cached_after_q = shared_cache.get().unwrap().clone();
        let _out_k = linears[1].forward(&x).unwrap();
        let cached_after_k = shared_cache.get().unwrap().clone();
        // The cached pointer is identical (OnceLock::set is no-op once set).
        // Compare via F32 round-trip to avoid pulling in the half crate.
        let after_q_data: Vec<f32> = cached_after_q
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let after_k_data: Vec<f32> = cached_after_k
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            after_q_data, after_k_data,
            "shared cache must be unchanged after subsequent forwards",
        );

        // Per-component reference: each component is identical here (all
        // weights = 0.25), so each Q/K/V forward must yield 16 × 0.25 = 4.0.
        for (component, lin) in linears.iter().enumerate() {
            let out = lin.forward(&x).unwrap();
            let v: Vec<f32> = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(v.len(), out_dim);
            for (i, &x) in v.iter().enumerate() {
                assert!(
                    (x - 4.0).abs() < 1e-2,
                    "component {component} out[{i}] = {x}, want 4.0",
                );
            }
        }
        // Sanity: Q output values match expectations.
        let q_v: Vec<f32> = out_q
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &x) in q_v.iter().enumerate() {
            assert!((x - 4.0).abs() < 1e-2, "Q[{i}]={x}");
        }
    }

    #[test]
    fn test_klein_config_vec_in_dim_zero() {
        let cfg = Flux2Config::klein();
        // Klein uses timestep-only conditioning (no pooled text embeddings),
        // so vec_in_dim must be 0 to skip the vector_in MLP embedder.
        assert_eq!(
            cfg.vec_in_dim, 0,
            "Klein vec_in_dim must be 0 (no pooled text vector)"
        );
        // Confirm the constructor logic: vec_in_dim == 0 means vector_in is None.
        // This is the architectural invariant enforced in Flux2Transformer::new().
        assert!(
            cfg.vec_in_dim == 0,
            "vec_in_dim > 0 would create an unused MlpEmbedder for Klein"
        );
        // Also verify that guidance_embed is false (distilled model, no CFG).
        assert!(
            !cfg.guidance_embed,
            "Klein is a distilled model; guidance_embed must be false"
        );
    }

    /// Regression test for the LastLayer shift/scale ordering bug (commit c0c2b80).
    ///
    /// `AdaLayerNormContinuous` — the diffusers norm — outputs (scale, shift):
    /// `x = norm(x) * (1 + scale) + shift`.
    ///
    /// `LastLayer::forward` must use chunks[0]=scale, chunks[1]=shift.
    ///
    /// Strategy: use xs=zeros so `norm(xs)=0` and the output reduces to
    /// `shift` only. If chunks were swapped, the output would be `scale`
    /// instead, which is distinct from `shift` in our fixture.
    ///
    /// `ada_ln_modulation` weight (2·h_sz × h_sz): constructed so that
    /// - rows 0..h_sz (scale half) produce `scale_val` per element
    /// - rows h_sz..2·h_sz (shift half) produce `shift_val` per element
    ///
    /// For a vec of all-ones and weight `w` per row, `silu(vec)·w.T` gives
    /// `h_sz * silu(1) * w_row` per output.  Set `w_row = target / (h_sz * silu(1))`.
    #[test]
    fn last_layer_forward_uses_diffusers_scale_then_shift_ordering() {
        use candle_core::Device;
        use candle_nn::VarBuilder;
        use std::collections::HashMap;

        let dev = Device::Cpu;
        let h_sz = 2usize;
        let out_c = 2usize; // proj_out: h_sz → h_sz (square for identity weight)

        let scale_val = 3.0f32;
        let shift_val = 0.5f32;

        // silu(1.0) ≈ 0.7311.  With vec = ones (h_sz=2 elements), the ada_ln
        // output for row i is: h_sz * silu(1) * w_per_element.
        // So w_per_element = target / (h_sz * silu(1)).
        let silu_one = 0.731_058_6f32; // silu(1) = 1 / (1 + e^-1)
        let dot_factor = h_sz as f32 * silu_one;
        let w_scale = scale_val / dot_factor;
        let w_shift = shift_val / dot_factor;

        // ada_ln weight: (2·h_sz, h_sz) = (4, 2).
        // Rows 0..1 are the SCALE half (diffusers first half → chunks[0]).
        // Rows 2..3 are the SHIFT half (diffusers second half → chunks[1]).
        let ada_weight: Vec<f32> = vec![
            w_scale, w_scale, // row 0 → scale[0]
            w_scale, w_scale, // row 1 → scale[1]
            w_shift, w_shift, // row 2 → shift[0]
            w_shift, w_shift, // row 3 → shift[1]
        ];

        // proj_out (out_c × h_sz): identity so output = xs_mod unchanged.
        let proj_weight = vec![1.0f32, 0.0, 0.0, 1.0];

        let mut map: HashMap<String, candle_core::Tensor> = HashMap::new();
        map.insert(
            "norm_out.linear.weight".to_string(),
            Tensor::from_vec(ada_weight, (2 * h_sz, h_sz), &dev).unwrap(),
        );
        map.insert(
            "proj_out.weight".to_string(),
            Tensor::from_vec(proj_weight, (out_c, h_sz), &dev).unwrap(),
        );
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        let layer = LastLayer::new(h_sz, out_c, vb).unwrap();

        // xs: all-zeros → norm(xs)=0 → output = 0*(1+scale) + shift = shift.
        let xs = Tensor::zeros((1, 1, h_sz), DType::F32, &dev).unwrap();
        // vec: all-ones → silu → matmul → scale_val in scale half, shift_val in shift half.
        let vec_ = Tensor::ones((1, h_sz), DType::F32, &dev).unwrap();

        let out = layer.forward(&xs, &vec_).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals.len(), out_c);

        // Correct ordering (scale, shift) → output = norm(0)*(1+scale_val) + shift_val = shift_val.
        // Wrong ordering (shift, scale) → output = norm(0)*(1+shift_val) + scale_val = scale_val.
        // These differ (0.5 vs 3.0) so the assertion unambiguously catches the regression.
        let tol = 0.08; // BF16 + silu rounding headroom
        assert!(
            (vals[0] - shift_val).abs() < tol,
            "LastLayer output[0]={:.4}: expected shift={shift_val:.4} \
             (diffusers scale-then-shift ordering). \
             Got scale={scale_val:.4} instead? The c0c2b80 regression is present.",
            vals[0],
        );
        assert!(
            (vals[1] - shift_val).abs() < tol,
            "LastLayer output[1]={:.4}: expected shift={shift_val:.4}",
            vals[1],
        );
    }
}
