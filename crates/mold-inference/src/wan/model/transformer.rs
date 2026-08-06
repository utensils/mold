//! The Wan DiT — a 3-D diffusion transformer over patchified video latents.
//!
//! Upstream is `Wan2.1/wan/modules/model.py`, with Wan 2.2's per-token timestep
//! branch (`Wan2.2/wan/modules/model.py:237-247, 460-469`) folded in because
//! TI2V-5B needs it. The load target is the **original** checkpoint layout
//! (`blocks.{i}.self_attn.*`, `ffn.0`/`ffn.2`, per-block `modulation`, ...),
//! which is what Comfy-Org repacks and the Wan-AI releases ship.
//!
//! Diffusers' `WanTransformer3DModel` is the same network under different
//! names, and its rename table swaps two of them: the original calls its norms
//! in the order `norm1, norm3, norm2`, so diffusers renames `norm3 <-> norm2`
//! to make the call order match the numbering. This port keeps the **original**
//! names, so here `norm3` is the affine cross-attention norm and `norm2` is the
//! parameter-free feed-forward norm. See [`diffusers_to_original`].
//!
//! Precision follows upstream rather than being uniformly promoted: patch
//! embedding, every normalization, the AdaLN modulation arithmetic, RoPE, and
//! the gated residual accumulations run in F32; the projections and attention
//! run in the model dtype.

use candle_core::{bail, DType, Device, Result, Tensor, D};
use candle_nn::{ops, Module, VarBuilder};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::wan::model::fp8;
use crate::wan::model::rope::{apply_rope, WanRope, WAN_ROPE_THETA};

/// Positions Wan precomputes rotation tables for (`model.py:478-485`).
const DEFAULT_ROPE_MAX_SEQ_LEN: usize = 1024;

/// Weight init used when a [`VarBuilder`] has nothing on disk (tests over a
/// `VarMap`). File-backed builders ignore these.
const TEST_WEIGHT_INIT: candle_nn::Init = candle_nn::Init::Randn {
    mean: 0.0,
    stdev: 0.02,
};

/// A biased linear projection, however it is stored.
///
/// `candle_nn::Linear` and `mold_candle::quantized_nn::Linear` both implement
/// `Module`, so the forward path is identical and only construction differs.
/// Holding the trait object rather than an enum keeps every call site in this
/// file unchanged between the plain and quantized builds.
pub(crate) type WanLinear = Arc<dyn Module + Send + Sync>;

/// Where a Wan transformer reads its weights.
///
/// GGUF is not merely a different container: a Q5_K A14B expert is ~10 GB
/// against ~28 GB at BF16, and that only holds if the quantized blocks stay
/// quantized in memory and dequantize per matmul. So the quantized arm builds
/// `QMatMul`-backed linears rather than dequantizing at load. fp8-scaled
/// safetensors work the same way at 1 byte per parameter — see
/// [`crate::wan::model::fp8`].
///
/// Non-linear tensors — norm gains, `modulation`, `patch_embedding` — are
/// dequantized eagerly. In the shipped QuantStack files they are already F16 or
/// F32 (only the 400 projection weights are quantized), so this costs nothing.
pub(crate) enum WanWeights<'a> {
    Plain(VarBuilder<'a>),
    /// GGUF. `loras` is empty unless an adapter is active, in which case each
    /// patched projection gains a parallel low-rank branch — see
    /// [`crate::wan::lora::WanQuantizedLoraLinear`] for why a merge is not an
    /// option on a quantized weight.
    Quantized {
        vb: mold_candle::quantized::VarBuilder,
        loras: Arc<crate::wan::lora::WanLoraRegistry>,
    },
    /// fp8-scaled safetensors. The builder is backed by
    /// [`crate::weight_loader::NativeFp8Backend`], so tensors arrive at their
    /// on-disk dtype and the per-module dequantization happens in the linear.
    Fp8 {
        vb: VarBuilder<'a>,
        marker: fp8::ScaledFp8Marker,
    },
}

impl<'a> WanWeights<'a> {
    pub fn plain(vb: VarBuilder<'a>) -> Self {
        Self::Plain(vb)
    }

    pub fn quantized(vb: mold_candle::quantized::VarBuilder) -> Self {
        Self::quantized_with_loras(vb, Arc::new(Default::default()))
    }

    pub fn quantized_with_loras(
        vb: mold_candle::quantized::VarBuilder,
        loras: Arc<crate::wan::lora::WanLoraRegistry>,
    ) -> Self {
        Self::Quantized { vb, loras }
    }

    pub fn fp8(vb: VarBuilder<'a>, marker: fp8::ScaledFp8Marker) -> Self {
        Self::Fp8 { vb, marker }
    }

    fn pp(&self, segment: impl std::fmt::Display) -> Self {
        match self {
            Self::Plain(vb) => Self::Plain(vb.pp(segment.to_string())),
            Self::Quantized { vb, loras } => Self::Quantized {
                vb: vb.pp(segment.to_string()),
                loras: loras.clone(),
            },
            Self::Fp8 { vb, marker } => Self::Fp8 {
                vb: vb.pp(segment.to_string()),
                marker: *marker,
            },
        }
    }

    fn device(&self) -> Device {
        match self {
            Self::Plain(vb) => vb.device().clone(),
            Self::Quantized { vb, .. } => vb.device().clone(),
            Self::Fp8 { vb, .. } => vb.device().clone(),
        }
    }

    /// Model dtype. The fp8 arm keeps the caller's dtype: that is the dtype
    /// its weights dequantize *into*, so it is the activation dtype for the
    /// whole model.
    ///
    /// The quantized arm runs BF16 activations on GPU: candle's `QMatMul`
    /// dequantizes into F32 *inside* each linear, but carrying F32 between
    /// ops was measured to OOM a 24 GB card — ~3.4 GB of activations at a
    /// mere 4.7k tokens, extrapolating past the card at the 20k+ tokens a
    /// 49-frame 480p render needs. Each quantized linear casts its own IO at
    /// the boundary (see `CastBoundary`), so the F32 stays transient and
    /// per-linear. CPU keeps F32 because candle's CPU backend has no BF16
    /// matmul — the same constraint that moved the CPU-parked encoder to F32.
    fn dtype(&self) -> DType {
        match self {
            Self::Plain(vb) => vb.dtype(),
            Self::Quantized { vb, .. } => {
                if vb.device().is_cpu() {
                    DType::F32
                } else {
                    DType::BF16
                }
            }
            Self::Fp8 { vb, .. } => vb.dtype(),
        }
    }

    /// The fp8 marker this source was built from, for the load-time report.
    fn quantization(&self) -> Option<fp8::ScaledFp8Marker> {
        match self {
            Self::Fp8 { marker, .. } => Some(*marker),
            _ => None,
        }
    }

    fn linear(&self, in_dim: usize, out_dim: usize) -> Result<WanLinear> {
        match self {
            Self::Plain(vb) => Ok(Arc::new(candle_nn::linear(in_dim, out_dim, vb.clone())?)),
            Self::Quantized { vb, loras } => {
                let inner = crate::wan::lora::attach_quantized_branches(
                    loras,
                    &vb.key("weight"),
                    mold_candle::quantized_nn::linear(in_dim, out_dim, vb.clone())?,
                    vb.device(),
                )?;
                Ok(CastBoundary::wrap(inner, self.dtype()))
            }
            Self::Fp8 { vb, .. } => fp8::load_linear(vb, in_dim, out_dim, self.dtype()),
        }
    }

    /// Fetch a non-linear tensor, dequantizing when the source is GGUF.
    fn tensor<S: Into<candle_core::Shape>>(
        &self,
        shape: S,
        name: &str,
        init: candle_nn::Init,
    ) -> Result<Tensor> {
        let shape = shape.into();
        match self {
            // The fp8 backend shape-checks and hands back the on-disk dtype;
            // every consumer of a non-linear tensor casts to F32 anyway, so
            // keeping the file's own F32 is both cheaper and more faithful
            // than a round trip through the compute dtype.
            Self::Plain(vb) | Self::Fp8 { vb, .. } => vb.get_with_hints(shape, name, init),
            Self::Quantized { vb, loras } => {
                let tensor = vb.get_no_shape(name)?.dequantize(&self.device())?;
                if tensor.shape() != &shape {
                    bail!(
                        "Wan DiT: GGUF tensor `{name}` is {:?} but the config wants {shape:?}",
                        tensor.shape()
                    );
                }
                // A non-linear tensor is already dequantized, so an adapter
                // targeting one merges here exactly as it would on the plain
                // path. The shipped distills touch none of these, but a
                // checkpoint-specific adapter that did must not be applied on
                // one weight source and silently skipped on the other.
                loras.merge_into(&vb.key(name), tensor)
            }
        }
    }
}

/// IO-dtype boundary around a quantized linear: the surrounding graph runs in
/// the model dtype (BF16 on GPU), while `QMatMul` and the Lightning adapter
/// branch compute in F32 internally. Casting at this boundary keeps every F32
/// tensor transient and scoped to one linear call — carrying F32 *between*
/// ops is what blew past 24 GB at 20k+ tokens (measured: ~3.4 GB of
/// activations at 4.7k tokens on the 9-frame probe). A no-op when the model
/// dtype is already F32 (CPU).
struct CastBoundary {
    inner: WanLinear,
    io: DType,
}

impl CastBoundary {
    fn wrap(inner: WanLinear, io: DType) -> WanLinear {
        if io == DType::F32 {
            return inner;
        }
        Arc::new(Self { inner, io })
    }
}

impl Module for CastBoundary {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.inner
            .forward(&xs.to_dtype(DType::F32)?)?
            .to_dtype(self.io)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WanTransformerConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// Latent channels in. 16 for 2.1/2.2 T2V, 36 for I2V (16 noise + 4 mask +
    /// 16 conditioning image), 48 for TI2V-5B.
    pub in_dim: usize,
    pub out_dim: usize,
    pub text_dim: usize,
    /// Width of the sinusoidal timestep embedding before the MLP.
    pub freq_dim: usize,
    pub patch_size: (usize, usize, usize),
    pub eps: f64,
    pub rope_max_seq_len: usize,
}

impl WanTransformerConfig {
    fn base(dim: usize, ffn_dim: usize, num_heads: usize, num_layers: usize) -> Self {
        Self {
            dim,
            ffn_dim,
            num_heads,
            num_layers,
            in_dim: 16,
            out_dim: 16,
            text_dim: 4096,
            freq_dim: 256,
            patch_size: (1, 2, 2),
            eps: 1e-6,
            rope_max_seq_len: DEFAULT_ROPE_MAX_SEQ_LEN,
        }
    }

    /// Wan 2.1 T2V-1.3B (`configs/wan_t2v_1_3B.py:20-29`).
    pub fn t2v_1_3b() -> Self {
        Self::base(1536, 8960, 12, 30)
    }

    /// Wan 2.1 T2V-14B and both Wan 2.2 A14B experts
    /// (`configs/wan_t2v_14B.py:20-29`).
    pub fn t2v_14b() -> Self {
        Self::base(5120, 13824, 40, 40)
    }

    /// Wan 2.1 I2V-14B and Wan 2.2 I2V-A14B: identical to T2V-14B except the
    /// 36-channel input. The 2.1 CLIP cross-attention branch (`k_img`/`v_img`)
    /// is deliberately not modelled — 2.2 dropped it and conditions purely
    /// through the channel concat.
    pub fn i2v_14b() -> Self {
        Self {
            in_dim: 36,
            ..Self::t2v_14b()
        }
    }

    /// Wan 2.2 TI2V-5B (`configs/wan_ti2v_5B.py:20-29`). Pairs with the 2.2
    /// VAE, hence 48 latent channels in and out.
    pub fn ti2v_5b() -> Self {
        Self {
            in_dim: 48,
            out_dim: 48,
            ..Self::base(3072, 14336, 24, 30)
        }
    }

    /// Small config for tests; mirrors the production topology at toy widths.
    pub fn tiny(dim: usize, num_heads: usize, num_layers: usize) -> Self {
        Self {
            dim,
            ffn_dim: dim * 2,
            num_heads,
            num_layers,
            in_dim: dim,
            out_dim: dim,
            text_dim: dim * 2,
            freq_dim: dim,
            patch_size: (1, 2, 2),
            eps: 1e-6,
            rope_max_seq_len: 32,
        }
    }

    /// 128 for every production variant.
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    /// Latent elements folded into one token.
    pub fn patch_elems(&self) -> usize {
        self.patch_size.0 * self.patch_size.1 * self.patch_size.2
    }

    fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.num_heads == 0 || !self.dim.is_multiple_of(self.num_heads) {
            bail!(
                "Wan DiT: dim {} must be a positive multiple of num_heads {}",
                self.dim,
                self.num_heads
            );
        }
        if !self.freq_dim.is_multiple_of(2) {
            bail!("Wan DiT: freq_dim {} must be even", self.freq_dim);
        }
        let (pt, ph, pw) = self.patch_size;
        if pt == 0 || ph == 0 || pw == 0 {
            bail!("Wan DiT: patch size {:?} must be positive", self.patch_size);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// checkpoint key naming
// ---------------------------------------------------------------------------

/// Module-path pairs, `(diffusers, original)`, for names outside the blocks.
const TOP_LEVEL_RENAMES: &[(&str, &str)] = &[
    ("patch_embedding", "patch_embedding"),
    (
        "condition_embedder.time_embedder.linear_1",
        "time_embedding.0",
    ),
    (
        "condition_embedder.time_embedder.linear_2",
        "time_embedding.2",
    ),
    (
        "condition_embedder.text_embedder.linear_1",
        "text_embedding.0",
    ),
    (
        "condition_embedder.text_embedder.linear_2",
        "text_embedding.2",
    ),
    ("condition_embedder.time_proj", "time_projection.1"),
    ("proj_out", "head.head"),
    ("scale_shift_table", "head.modulation"),
];

/// Module-path pairs, `(diffusers, original)`, for the per-block tail after
/// `blocks.{i}.`. Note `norm2 <-> norm3`: this is the swap diffusers' converter
/// performs with a placeholder (`convert_wan_to_diffusers.py:45-48`).
const BLOCK_RENAMES: &[(&str, &str)] = &[
    ("scale_shift_table", "modulation"),
    ("norm2", "norm3"),
    ("norm3", "norm2"),
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2", "ffn.2"),
    ("attn1.to_q", "self_attn.q"),
    ("attn1.to_k", "self_attn.k"),
    ("attn1.to_v", "self_attn.v"),
    ("attn1.to_out.0", "self_attn.o"),
    ("attn1.norm_q", "self_attn.norm_q"),
    ("attn1.norm_k", "self_attn.norm_k"),
    ("attn2.to_q", "cross_attn.q"),
    ("attn2.to_k", "cross_attn.k"),
    ("attn2.to_v", "cross_attn.v"),
    ("attn2.to_out.0", "cross_attn.o"),
    ("attn2.norm_q", "cross_attn.norm_q"),
    ("attn2.norm_k", "cross_attn.norm_k"),
];

/// Split a parameter name into its module path and leaf (`weight` / `bias`).
/// Bare parameters (`modulation`, `scale_shift_table`) have no leaf.
fn split_leaf(name: &str) -> (&str, Option<&str>) {
    for leaf in ["weight", "bias"] {
        if let Some(path) = name.strip_suffix(leaf) {
            if let Some(path) = path.strip_suffix('.') {
                return (path, Some(leaf));
            }
        }
    }
    (name, None)
}

fn rejoin(path: &str, leaf: Option<&str>) -> String {
    match leaf {
        Some(leaf) => format!("{path}.{leaf}"),
        None => path.to_string(),
    }
}

fn lookup(
    table: &'static [(&'static str, &'static str)],
    key: &str,
    forward: bool,
) -> Option<&'static str> {
    table.iter().find_map(|(diffusers, original)| {
        let (from, to) = if forward {
            (diffusers, original)
        } else {
            (original, diffusers)
        };
        (*from == key).then_some(*to)
    })
}

fn translate(name: &str, forward: bool) -> Option<String> {
    let (path, leaf) = split_leaf(name);
    if let Some(rest) = path.strip_prefix("blocks.") {
        let (index, tail) = rest.split_once('.')?;
        index.parse::<usize>().ok()?;
        let mapped = lookup(BLOCK_RENAMES, tail, forward)?;
        return Some(rejoin(&format!("blocks.{index}.{mapped}"), leaf));
    }
    Some(rejoin(lookup(TOP_LEVEL_RENAMES, path, forward)?, leaf))
}

/// Map a diffusers `WanTransformer3DModel` parameter name to the original
/// checkpoint name this module loads. Returns `None` for names outside the
/// modelled subset (the I2V CLIP adapter and VACE blocks).
pub(crate) fn diffusers_to_original(name: &str) -> Option<String> {
    translate(name, true)
}

/// The inverse of [`diffusers_to_original`].
pub(crate) fn original_to_diffusers(name: &str) -> Option<String> {
    translate(name, false)
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

/// `WanRMSNorm` (`model.py:73-89`) over the last dimension, normalized in F32.
#[derive(Debug, Clone)]
struct WanRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl WanRmsNorm {
    fn load(vb: &WanWeights<'_>, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.tensor(dim, "weight", candle_nn::Init::Const(1.0))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let variance = x32.sqr()?.mean_keepdim(D::Minus1)?;
        x32.broadcast_div(&variance.affine(1.0, self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    }
}

/// LayerNorm over the last dimension, always evaluated in F32 (diffusers'
/// `FP32LayerNorm`; upstream casts with `.float()` at every call site).
/// Returns F32 — callers decide when to come back down.
///
/// Only the cross-attention norm carries affine parameters; `norm1` and
/// `norm2` are parameter-free (`model.py:258-262`).
#[derive(Debug, Clone)]
struct WanLayerNorm {
    affine: Option<(Tensor, Tensor)>,
    eps: f64,
}

impl WanLayerNorm {
    fn plain(eps: f64) -> Self {
        Self { affine: None, eps }
    }

    fn affine(vb: &WanWeights<'_>, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.tensor(dim, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.tensor(dim, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            affine: Some((weight, bias)),
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&variance.affine(1.0, self.eps)?.sqrt()?)?;
        match &self.affine {
            Some((weight, bias)) => normed
                .broadcast_mul(&weight.to_dtype(DType::F32)?)?
                .broadcast_add(&bias.to_dtype(DType::F32)?),
            None => Ok(normed),
        }
    }
}

/// Self- or cross-attention. Both carry biased q/k/v/o projections and RMS
/// norms on q and k applied to the **full inner dimension before the head
/// split** (`qk_norm = "rms_norm_across_heads"`, `model.py:127-128`).
#[derive(Clone)]
struct WanAttention {
    q: WanLinear,
    k: WanLinear,
    v: WanLinear,
    o: WanLinear,
    norm_q: WanRmsNorm,
    norm_k: WanRmsNorm,
    heads: usize,
    head_dim: usize,
}

impl WanAttention {
    fn load(vb: &WanWeights<'_>, dim: usize, heads: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            q: vb.pp("q").linear(dim, dim)?,
            k: vb.pp("k").linear(dim, dim)?,
            v: vb.pp("v").linear(dim, dim)?,
            o: vb.pp("o").linear(dim, dim)?,
            norm_q: WanRmsNorm::load(&vb.pp("norm_q"), dim, eps)?,
            norm_k: WanRmsNorm::load(&vb.pp("norm_k"), dim, eps)?,
            heads,
            head_dim: dim / heads,
        })
    }

    /// `context` is `None` for self-attention (k/v come from `x`, and RoPE
    /// applies). Cross-attention passes the text embedding and no RoPE.
    fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (batch, tokens, _) = x.dims3()?;
        let kv_source = context.unwrap_or(x);
        let (_, kv_tokens, _) = kv_source.dims3()?;

        let split = |t: Tensor, len: usize| -> Result<Tensor> {
            t.reshape((batch, len, self.heads, self.head_dim))
        };
        let q = split(self.norm_q.forward(&self.q.forward(x)?)?, tokens)?;
        let k = split(self.norm_k.forward(&self.k.forward(kv_source)?)?, kv_tokens)?;
        let v = split(self.v.forward(kv_source)?, kv_tokens)?;

        let (q, k) = match rope {
            Some((cos, sin)) => (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?),
            None => (q, k),
        };

        // `crate::attention::attention` wants [batch, heads, seq, head_dim].
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = crate::attention::attention(&q, &k, &v, scale)?;

        let merged = attn.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            tokens,
            self.heads * self.head_dim,
        ))?;
        self.o.forward(&merged)
    }
}

/// `Linear -> GELU(tanh) -> Linear` (`model.py:271-273`).
#[derive(Clone)]
struct WanFeedForward {
    up: WanLinear,
    down: WanLinear,
}

impl WanFeedForward {
    fn load(vb: &WanWeights<'_>, dim: usize, ffn_dim: usize) -> Result<Self> {
        Ok(Self {
            up: vb.pp("0").linear(dim, ffn_dim)?,
            down: vb.pp("2").linear(ffn_dim, dim)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(x)?.gelu()?)
    }
}

/// The six AdaLN parameters for one block, each `[batch, n, dim]` with `n` of
/// 1 (scalar timestep) or `tokens` (per-token, Wan 2.2 TI2V) so both broadcast
/// against `[batch, tokens, dim]`.
struct BlockModulation {
    shift_attn: Tensor,
    scale_attn: Tensor,
    gate_attn: Tensor,
    shift_ffn: Tensor,
    scale_ffn: Tensor,
    gate_ffn: Tensor,
}

/// Pull entry `index` out of a `[batch, n, chunks, dim]` table.
fn chunk_at(table: &Tensor, index: usize) -> Result<Tensor> {
    table.narrow(2, index, 1)?.squeeze(2)
}

#[derive(Clone)]
struct WanBlock {
    norm1: WanLayerNorm,
    self_attn: WanAttention,
    /// The cross-attention norm — affine, and named `norm3` upstream even
    /// though it is called second.
    norm3: WanLayerNorm,
    cross_attn: WanAttention,
    norm2: WanLayerNorm,
    ffn: WanFeedForward,
    /// `[1, 6, dim]`, added to the shared time projection.
    modulation: Tensor,
}

impl WanBlock {
    fn load(vb: &WanWeights<'_>, config: &WanTransformerConfig) -> Result<Self> {
        let dim = config.dim;
        Ok(Self {
            norm1: WanLayerNorm::plain(config.eps),
            self_attn: WanAttention::load(&vb.pp("self_attn"), dim, config.num_heads, config.eps)?,
            norm3: WanLayerNorm::affine(&vb.pp("norm3"), dim, config.eps)?,
            cross_attn: WanAttention::load(
                &vb.pp("cross_attn"),
                dim,
                config.num_heads,
                config.eps,
            )?,
            norm2: WanLayerNorm::plain(config.eps),
            ffn: WanFeedForward::load(&vb.pp("ffn"), dim, config.ffn_dim)?,
            modulation: vb.tensor((1, 6, dim), "modulation", TEST_WEIGHT_INIT)?,
        })
    }

    /// `timestep_proj` is `[batch, n, 6, dim]` in F32.
    fn modulation(&self, timestep_proj: &Tensor) -> Result<BlockModulation> {
        let dim = self.modulation.dim(2)?;
        let table = self
            .modulation
            .to_dtype(DType::F32)?
            .reshape((1, 1, 6, dim))?
            .broadcast_add(timestep_proj)?;
        Ok(BlockModulation {
            shift_attn: chunk_at(&table, 0)?,
            scale_attn: chunk_at(&table, 1)?,
            gate_attn: chunk_at(&table, 2)?,
            shift_ffn: chunk_at(&table, 3)?,
            scale_ffn: chunk_at(&table, 4)?,
            gate_ffn: chunk_at(&table, 5)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        timestep_proj: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dtype = x.dtype();
        let m = self.modulation(timestep_proj)?;

        // 1. Self-attention, shift/scale in, gate out.
        let normed = self
            .norm1
            .forward(x)?
            .broadcast_mul(&m.scale_attn.affine(1.0, 1.0)?)?
            .broadcast_add(&m.shift_attn)?
            .to_dtype(dtype)?;
        let attn = self.self_attn.forward(&normed, None, Some((cos, sin)))?;
        let x = x
            .to_dtype(DType::F32)?
            .add(&attn.to_dtype(DType::F32)?.broadcast_mul(&m.gate_attn)?)?
            .to_dtype(dtype)?;

        // 2. Cross-attention on the text embedding — no modulation, no gate,
        //    plain residual (`model.py:307-310`).
        let normed = self.norm3.forward(&x)?.to_dtype(dtype)?;
        let attn = self.cross_attn.forward(&normed, Some(context), None)?;
        let x = x.add(&attn)?;

        // 3. Feed-forward, shift/scale in, gate out.
        let normed = self
            .norm2
            .forward(&x)?
            .broadcast_mul(&m.scale_ffn.affine(1.0, 1.0)?)?
            .broadcast_add(&m.shift_ffn)?
            .to_dtype(dtype)?;
        let ff = self.ffn.forward(&normed)?;
        x.to_dtype(DType::F32)?
            .add(&ff.to_dtype(DType::F32)?.broadcast_mul(&m.gate_ffn)?)?
            .to_dtype(dtype)
    }
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

/// `sinusoidal_embedding_1d` (`model.py:18-28`): the cosine half first, then
/// the sine half (diffusers spells this `flip_sin_to_cos=True`).
///
/// Evaluated in `f64` on the host like upstream. That is exact for the scalar
/// timestep every variant except TI2V uses; a per-token vector at 720p is
/// ~75k entries, which is a visible but not dominant host cost.
fn sinusoidal_embedding(values: &[f64], dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut out = Vec::with_capacity(values.len() * dim);
    for value in values {
        let angle = |i: usize| value * 10000f64.powf(-(i as f64) / half as f64);
        for i in 0..half {
            out.push(angle(i).cos() as f32);
        }
        for i in 0..half {
            out.push(angle(i).sin() as f32);
        }
    }
    Tensor::from_vec(out, (values.len(), dim), device)
}

#[derive(Clone)]
pub(crate) struct WanTransformer {
    config: WanTransformerConfig,
    /// The checkpoint's 5-D Conv3d weight flattened to `[in*patch, dim]`.
    /// Kernel equals stride, so the convolution is exactly a patch unfold
    /// followed by a matmul — and the unfold produces the `(f, h, w)` token
    /// order RoPE assumes. Kept in F32 because ComfyUI runs this layer as
    /// `patch_embedding(x.float()).to(dtype)`.
    patch_weight: Tensor,
    patch_bias: Tensor,
    time_embedding_in: WanLinear,
    time_embedding_out: WanLinear,
    time_projection: WanLinear,
    text_embedding_in: WanLinear,
    text_embedding_out: WanLinear,
    blocks: Vec<WanBlock>,
    head_norm: WanLayerNorm,
    /// `[1, 2, dim]`.
    head_modulation: Tensor,
    head: WanLinear,
    rope: WanRope,
    dtype: DType,
    /// Set when the weights came from an fp8-scaled checkpoint, so the engine
    /// can report which precision path a run is actually on.
    quantization: Option<fp8::ScaledFp8Marker>,
}

impl WanTransformer {
    /// Load from safetensors carrying the original key layout. A
    /// `model.diffusion_model.` prefix (full-pipeline checkpoints) is stripped.
    pub fn from_safetensors(
        paths: &[std::path::PathBuf],
        config: WanTransformerConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_safetensors_with_loras(paths, config, device, dtype, &Default::default())
    }

    /// Load with a LoRA stack merged into the weights as they are read.
    ///
    /// The merge happens inside a `SimpleBackend` rather than after
    /// construction, so no code in this module changes and the merged weight is
    /// the only copy that ever exists — there is no separate base tensor to
    /// keep resident. That is the same shape as FLUX's `LoraBackend`.
    pub fn from_safetensors_with_loras(
        paths: &[std::path::PathBuf],
        config: WanTransformerConfig,
        device: &Device,
        dtype: DType,
        loras: &crate::wan::lora::WanLoraRegistry,
    ) -> Result<Self> {
        // Header-only, and first: an e5m2 checkpoint has to be refused before
        // anything memory-maps a dtype candle cannot represent.
        let scaled_fp8 = fp8::probe_scaled_fp8(paths)?;
        if let Some(checkpoint) = &scaled_fp8 {
            let first = paths.first().map(PathBuf::as_path).unwrap_or(Path::new(""));
            checkpoint.ensure_supported(first)?;
            if !loras.is_empty() {
                bail!(
                    "{} is fp8-scaled, and mold merges LoRAs into bf16 and GGUF checkpoints \
                     only — an fp8 merge would dequantize every targeted weight, add the \
                     delta, and re-round it to three mantissa bits. Use the bf16 or GGUF \
                     tier for adapters, or a checkpoint that ships the distill already \
                     merged.",
                    first.display()
                );
            }
        }

        // The shipped Comfy-Org repacks prefix every DiT key; bare exports and
        // the GGUF layout do not. Probe once and reuse for both paths.
        let probe = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
        let prefix = if probe.contains_tensor("patch_embedding.weight") {
            String::new()
        } else {
            "model.diffusion_model.".to_string()
        };
        drop(probe);

        if let Some(checkpoint) = scaled_fp8 {
            // Native dtypes all the way in: the weights stay fp8 and the
            // linears dequantize per call.
            let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths) }?;
            let backend = crate::weight_loader::NativeFp8Backend::from_mmap(mmap);
            let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
            let vb = if prefix.is_empty() {
                vb
            } else {
                vb.pp("model.diffusion_model")
            };
            return Self::from_weights(&WanWeights::fp8(vb, checkpoint.marker), config);
        }

        if loras.is_empty() {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
            let vb = if prefix.is_empty() {
                vb
            } else {
                vb.pp("model.diffusion_model")
            };
            return Self::from_var_builder(vb, config);
        }

        let backend =
            unsafe { crate::wan::lora::WanLoraBackend::new(paths, prefix, loras.clone())? };
        backend
            .ensure_lora_targets_present(
                paths.first().map(PathBuf::as_path).unwrap_or(Path::new("")),
            )
            .map_err(|err| candle_core::Error::Msg(format!("{err:#}")))?;
        let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
        Self::from_var_builder(vb, config)
    }

    /// Load from a GGUF checkpoint (the QuantStack A14B layout).
    ///
    /// Quantized blocks stay quantized: a Q5_K expert is ~10 GB against ~28 GB
    /// dequantized, and two of them have to be resident-or-parked for the A14B
    /// tier. `QMatMul` dequantizes per matmul instead.
    ///
    /// GGUF keys are **bare** (`blocks.0.self_attn.q.weight`) — the
    /// `model.diffusion_model.` prefix the Comfy-Org safetensors repacks carry
    /// is absent, so no prefix handling applies here.
    pub fn from_gguf(path: &Path, config: WanTransformerConfig, device: &Device) -> Result<Self> {
        Self::from_gguf_with_loras(path, config, device, &Default::default())
    }

    /// Load a GGUF checkpoint with a LoRA stack applied.
    ///
    /// The adapters do not merge into the weights the way they do on the two
    /// safetensors paths — a quantized weight cannot take a delta without being
    /// requantized, which costs minutes per expert and rounds most of a distill
    /// away. Each patched projection gets a parallel low-rank branch instead;
    /// [`crate::wan::lora::WanQuantizedLoraLinear`] carries the measurements.
    pub fn from_gguf_with_loras(
        path: &Path,
        config: WanTransformerConfig,
        device: &Device,
        loras: &crate::wan::lora::WanLoraRegistry,
    ) -> Result<Self> {
        let vb = mold_candle::quantized::VarBuilder::from_gguf(path, device)?;
        if loras.is_empty() {
            return Self::from_weights(&WanWeights::quantized(vb), config);
        }
        loras
            .ensure_targets_present(|name| vb.contains_key(name), path)
            .map_err(|err| candle_core::Error::Msg(format!("{err:#}")))?;
        Self::from_weights(
            &WanWeights::quantized_with_loras(vb, Arc::new(loras.clone())),
            config,
        )
    }

    /// Convenience for a single-file checkpoint.
    pub fn from_safetensors_file(
        path: &Path,
        config: WanTransformerConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_safetensors(&[path.to_path_buf()], config, device, dtype)
    }

    pub fn from_var_builder(vb: VarBuilder, config: WanTransformerConfig) -> Result<Self> {
        Self::from_weights(&WanWeights::plain(vb), config)
    }

    /// Build from either weight source.
    pub fn from_weights(vb: &WanWeights<'_>, config: WanTransformerConfig) -> Result<Self> {
        config.validate()?;
        let dtype = vb.dtype();
        let dim = config.dim;
        let (pt, ph, pw) = config.patch_size;

        // Always F32, independent of the model dtype. In the GGUF layout this
        // is the file's only F32 tensor, stored 5-D — the quantizers made the
        // same call.
        let vb_patch = vb.pp("patch_embedding");
        let patch_weight = vb_patch
            .tensor((dim, config.in_dim, pt, ph, pw), "weight", TEST_WEIGHT_INIT)?
            .to_dtype(DType::F32)?
            .reshape((dim, config.in_dim * config.patch_elems()))?
            .t()?
            .contiguous()?;
        let patch_bias = vb_patch
            .tensor(dim, "bias", candle_nn::Init::Const(0.0))?
            .to_dtype(DType::F32)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(WanBlock::load(&vb.pp(format!("blocks.{i}")), &config)?);
        }

        let head_modulation = vb
            .pp("head")
            .tensor((1, 2, dim), "modulation", TEST_WEIGHT_INIT)?;

        Ok(Self {
            time_embedding_in: vb
                .pp("time_embedding")
                .pp("0")
                .linear(config.freq_dim, dim)?,
            time_embedding_out: vb.pp("time_embedding").pp("2").linear(dim, dim)?,
            time_projection: vb.pp("time_projection").pp("1").linear(dim, 6 * dim)?,
            text_embedding_in: vb
                .pp("text_embedding")
                .pp("0")
                .linear(config.text_dim, dim)?,
            text_embedding_out: vb.pp("text_embedding").pp("2").linear(dim, dim)?,
            blocks,
            head_norm: WanLayerNorm::plain(config.eps),
            head_modulation,
            head: vb
                .pp("head")
                .pp("head")
                .linear(dim, config.out_dim * config.patch_elems())?,
            rope: WanRope::new(config.head_dim(), config.rope_max_seq_len, WAN_ROPE_THETA)?,
            patch_weight,
            patch_bias,
            config,
            dtype,
            quantization: vb.quantization(),
        })
    }

    pub fn config(&self) -> &WanTransformerConfig {
        &self.config
    }

    /// The fp8 marker the loaded checkpoint declared, if any.
    pub fn quantization(&self) -> Option<fp8::ScaledFp8Marker> {
        self.quantization
    }

    /// Fold `[b, c, f, h, w]` into `[b, tokens, dim]` in `(f, h, w)` order.
    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = x.dims5()?;
        let (pt, ph, pw) = self.config.patch_size;
        let (gf, gh, gw) = (frames / pt, height / ph, width / pw);
        let folded = x
            .to_dtype(DType::F32)?
            .reshape(&[batch, channels, gf, pt, gh, ph, gw, pw])?
            // -> [b, gf, gh, gw, c, pt, ph, pw]; channel-major inside a patch,
            // which is the order Conv3d flattens its kernel in.
            .permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?
            .contiguous()?
            .reshape((batch, gf * gh * gw, channels * pt * ph * pw))?;
        folded
            .broadcast_matmul(&self.patch_weight)?
            .broadcast_add(&self.patch_bias.reshape((1, 1, self.config.dim))?)
    }

    /// Inverse of [`Self::patchify`] over the head's `out_dim * patch` output.
    fn unpatchify(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (batch, _, _) = x.dims3()?;
        let (gf, gh, gw) = grid;
        let (pt, ph, pw) = self.config.patch_size;
        let out_dim = self.config.out_dim;
        x.reshape(&[batch, gf, gh, gw, pt, ph, pw, out_dim])?
            .permute(vec![0, 7, 1, 4, 2, 5, 3, 6])?
            .contiguous()?
            .reshape((batch, out_dim, gf * pt, gh * ph, gw * pw))
    }

    /// Timestep -> `(temb [b, n, dim], timestep_proj [b, n, 6, dim])`, both
    /// F32. `n` is 1 for a `[b]` timestep and `tokens` for a `[b, tokens]` one.
    fn embed_timestep(&self, timestep: &Tensor, batch: usize) -> Result<(Tensor, Tensor)> {
        let per_token = match timestep.rank() {
            1 => None,
            2 => Some(timestep.dim(1)?),
            other => {
                bail!("Wan DiT: timestep must be [batch] or [batch, tokens], got rank {other}")
            }
        };
        if timestep.dim(0)? != batch {
            bail!(
                "Wan DiT: timestep batch {} does not match latent batch {batch}",
                timestep.dim(0)?
            );
        }
        let values: Vec<f64> = timestep
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect();
        let n = per_token.unwrap_or(1);

        let sinusoid = sinusoidal_embedding(&values, self.config.freq_dim, timestep.device())?
            .to_dtype(self.dtype)?;
        let temb = self
            .time_embedding_out
            .forward(&ops::silu(&self.time_embedding_in.forward(&sinusoid)?)?)?;
        let projection = self.time_projection.forward(&ops::silu(&temb)?)?;

        let temb = temb
            .to_dtype(DType::F32)?
            .reshape((batch, n, self.config.dim))?;
        let projection =
            projection
                .to_dtype(DType::F32)?
                .reshape((batch, n, 6, self.config.dim))?;
        Ok((temb, projection))
    }

    /// `x`: `[batch, in_dim, frames, height, width]`. `timestep`: `[batch]`, or
    /// `[batch, tokens]` for Wan 2.2 TI2V's per-token schedule. `context`:
    /// `[batch, text_len, text_dim]`, already encoded by UMT5.
    ///
    /// Returns the predicted flow at `[batch, out_dim, frames, height, width]`.
    pub fn forward(&self, x: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        let rope = self.rope_freqs_for(x)?;
        self.forward_with_rope(x, timestep, context, &rope)
    }

    /// Build the rotation tables for a latent's patch grid.
    ///
    /// Split out so a denoise loop can hoist this above the step loop: the
    /// tables depend only on the grid shape, which is fixed for the whole
    /// sampling run, and at 720p they cost ~38 MB plus a few million
    /// `f64`->`f32` conversions per call.
    pub fn rope_freqs_for(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let grid = self.patch_grid(x)?;
        self.rope.freqs(grid.0, grid.1, grid.2, x.device())
    }

    fn patch_grid(&self, x: &Tensor) -> Result<(usize, usize, usize)> {
        let (_, channels, frames, height, width) = x.dims5()?;
        if channels != self.config.in_dim {
            bail!(
                "Wan DiT: expected {} latent channels, got {channels}",
                self.config.in_dim
            );
        }
        let (pt, ph, pw) = self.config.patch_size;
        if !frames.is_multiple_of(pt) || !height.is_multiple_of(ph) || !width.is_multiple_of(pw) {
            bail!(
                "Wan DiT: {frames}x{height}x{width} is not divisible by the patch size {:?}",
                self.config.patch_size
            );
        }
        Ok((frames / pt, height / ph, width / pw))
    }

    /// [`Self::forward`] with the rotation tables supplied by the caller.
    pub fn forward_with_rope(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        rope: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let grid = self.patch_grid(x)?;
        let (cos, sin) = (&rope.0, &rope.1);
        let expected_tokens = grid.0 * grid.1 * grid.2;
        if cos.dim(0)? != expected_tokens {
            bail!(
                "Wan DiT: rotation tables cover {} tokens but this latent has {expected_tokens}",
                cos.dim(0)?
            );
        }
        let mut hidden = self.patchify(x)?.to_dtype(self.dtype)?;
        let (temb, timestep_proj) = self.embed_timestep(timestep, batch)?;

        let context = self.text_embedding_out.forward(
            &self
                .text_embedding_in
                .forward(&context.to_dtype(self.dtype)?)?
                .gelu()?,
        )?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &context, &timestep_proj, cos, sin)?;
        }

        // Head: a 2-entry modulation table over the *unprojected* time
        // embedding, not the 6-entry projection the blocks use.
        let table = self
            .head_modulation
            .to_dtype(DType::F32)?
            .reshape((1, 1, 2, self.config.dim))?
            .broadcast_add(&temb.unsqueeze(2)?)?;
        let shift = chunk_at(&table, 0)?;
        let scale = chunk_at(&table, 1)?;
        let normed = self
            .head_norm
            .forward(&hidden)?
            .broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        self.unpatchify(&self.head.forward(&normed)?, grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use std::collections::HashMap;

    /// Every diffusers `WanTransformer3DModel` parameter in the fixture, with
    /// its shape. Weights are synthesized from these names, then loaded under
    /// the original checkpoint names.
    #[allow(clippy::type_complexity)]
    const GOLDEN_PARAMS: &[(&str, &[usize])] = &[
        ("blocks.0.attn1.norm_k.weight", &[16]),
        ("blocks.0.attn1.norm_q.weight", &[16]),
        ("blocks.0.attn1.to_k.bias", &[16]),
        ("blocks.0.attn1.to_k.weight", &[16, 16]),
        ("blocks.0.attn1.to_out.0.bias", &[16]),
        ("blocks.0.attn1.to_out.0.weight", &[16, 16]),
        ("blocks.0.attn1.to_q.bias", &[16]),
        ("blocks.0.attn1.to_q.weight", &[16, 16]),
        ("blocks.0.attn1.to_v.bias", &[16]),
        ("blocks.0.attn1.to_v.weight", &[16, 16]),
        ("blocks.0.attn2.norm_k.weight", &[16]),
        ("blocks.0.attn2.norm_q.weight", &[16]),
        ("blocks.0.attn2.to_k.bias", &[16]),
        ("blocks.0.attn2.to_k.weight", &[16, 16]),
        ("blocks.0.attn2.to_out.0.bias", &[16]),
        ("blocks.0.attn2.to_out.0.weight", &[16, 16]),
        ("blocks.0.attn2.to_q.bias", &[16]),
        ("blocks.0.attn2.to_q.weight", &[16, 16]),
        ("blocks.0.attn2.to_v.bias", &[16]),
        ("blocks.0.attn2.to_v.weight", &[16, 16]),
        ("blocks.0.ffn.net.0.proj.bias", &[32]),
        ("blocks.0.ffn.net.0.proj.weight", &[32, 16]),
        ("blocks.0.ffn.net.2.bias", &[16]),
        ("blocks.0.ffn.net.2.weight", &[16, 32]),
        ("blocks.0.norm2.bias", &[16]),
        ("blocks.0.norm2.weight", &[16]),
        ("blocks.0.scale_shift_table", &[1, 6, 16]),
        ("blocks.1.attn1.norm_k.weight", &[16]),
        ("blocks.1.attn1.norm_q.weight", &[16]),
        ("blocks.1.attn1.to_k.bias", &[16]),
        ("blocks.1.attn1.to_k.weight", &[16, 16]),
        ("blocks.1.attn1.to_out.0.bias", &[16]),
        ("blocks.1.attn1.to_out.0.weight", &[16, 16]),
        ("blocks.1.attn1.to_q.bias", &[16]),
        ("blocks.1.attn1.to_q.weight", &[16, 16]),
        ("blocks.1.attn1.to_v.bias", &[16]),
        ("blocks.1.attn1.to_v.weight", &[16, 16]),
        ("blocks.1.attn2.norm_k.weight", &[16]),
        ("blocks.1.attn2.norm_q.weight", &[16]),
        ("blocks.1.attn2.to_k.bias", &[16]),
        ("blocks.1.attn2.to_k.weight", &[16, 16]),
        ("blocks.1.attn2.to_out.0.bias", &[16]),
        ("blocks.1.attn2.to_out.0.weight", &[16, 16]),
        ("blocks.1.attn2.to_q.bias", &[16]),
        ("blocks.1.attn2.to_q.weight", &[16, 16]),
        ("blocks.1.attn2.to_v.bias", &[16]),
        ("blocks.1.attn2.to_v.weight", &[16, 16]),
        ("blocks.1.ffn.net.0.proj.bias", &[32]),
        ("blocks.1.ffn.net.0.proj.weight", &[32, 16]),
        ("blocks.1.ffn.net.2.bias", &[16]),
        ("blocks.1.ffn.net.2.weight", &[16, 32]),
        ("blocks.1.norm2.bias", &[16]),
        ("blocks.1.norm2.weight", &[16]),
        ("blocks.1.scale_shift_table", &[1, 6, 16]),
        ("condition_embedder.text_embedder.linear_1.bias", &[16]),
        (
            "condition_embedder.text_embedder.linear_1.weight",
            &[16, 32],
        ),
        ("condition_embedder.text_embedder.linear_2.bias", &[16]),
        (
            "condition_embedder.text_embedder.linear_2.weight",
            &[16, 16],
        ),
        ("condition_embedder.time_embedder.linear_1.bias", &[16]),
        (
            "condition_embedder.time_embedder.linear_1.weight",
            &[16, 16],
        ),
        ("condition_embedder.time_embedder.linear_2.bias", &[16]),
        (
            "condition_embedder.time_embedder.linear_2.weight",
            &[16, 16],
        ),
        ("condition_embedder.time_proj.bias", &[96]),
        ("condition_embedder.time_proj.weight", &[96, 16]),
        ("patch_embedding.bias", &[16]),
        ("patch_embedding.weight", &[16, 16, 1, 2, 2]),
        ("proj_out.bias", &[64]),
        ("proj_out.weight", &[64, 16]),
        ("scale_shift_table", &[1, 2, 16]),
    ];

    /// Hidden states after block 0, `[1, 8, 16]` flattened.
    const GOLDEN_BLOCK_0: &[f64] = &[
        0.02714613452553749,
        0.032741762697696686,
        0.025569695979356766,
        0.004228069446980953,
        -0.0221966952085495,
        -0.037281155586242676,
        -0.03186117857694626,
        -0.012180414982140064,
        0.009960420429706573,
        0.026649879291653633,
        0.03440403565764427,
        0.028897713869810104,
        0.007746624294668436,
        -0.021220915019512177,
        -0.04037526994943619,
        -0.036996740847826004,
        0.026789387688040733,
        0.037321485579013824,
        0.032189883291721344,
        0.008703813888132572,
        -0.02269519865512848,
        -0.042438164353370667,
        -0.0384175106883049,
        -0.015990091487765312,
        0.011305442079901695,
        0.03229910135269165,
        0.040786366909742355,
        0.03197810426354408,
        0.005577489268034697,
        -0.027268270030617714,
        -0.046476949006319046,
        -0.039296574890613556,
        0.02971804514527321,
        0.03319671005010605,
        0.023623093962669373,
        0.001108909142203629,
        -0.02451547607779503,
        -0.03733629360795021,
        -0.029612859711050987,
        -0.009054803289473057,
        0.011989302933216095,
        0.026300132274627686,
        0.03189520165324211,
        0.025816379114985466,
        0.006041578948497772,
        -0.020471645519137383,
        -0.03764871880412102,
        -0.03401070833206177,
        0.0259560514241457,
        0.03441790118813515,
        0.029051022604107857,
        0.0073144142515957355,
        -0.02145698107779026,
        -0.03935444727540016,
        -0.035439085215330124,
        -0.014998731203377247,
        0.009682501666247845,
        0.029087718576192856,
        0.03801712021231651,
        0.03140166774392128,
        0.007558038923889399,
        -0.02398284152150154,
        -0.04396286979317665,
        -0.03914470225572586,
        0.032526545226573944,
        0.035459090024232864,
        0.02391219325363636,
        -0.000759810907766223,
        -0.027359575033187866,
        -0.03936029225587845,
        -0.029532400891184807,
        -0.0069254012778401375,
        0.014823387376964092,
        0.02804797887802124,
        0.031450167298316956,
        0.02346045896410942,
        0.003264809027314186,
        -0.021913645789027214,
        -0.036847591400146484,
        -0.03146708756685257,
        0.02704794891178608,
        0.032773636281490326,
        0.025711363181471825,
        0.004390031564980745,
        -0.022116869688034058,
        -0.03733382746577263,
        -0.03201323747634888,
        -0.012335550040006638,
        0.009900212287902832,
        0.026722634211182594,
        0.0345638133585453,
        0.029043488204479218,
        0.007786205038428307,
        -0.02131255716085434,
        -0.04054006561636925,
        -0.03713073581457138,
        0.03431088477373123,
        0.03851352259516716,
        0.026307130232453346,
        -0.0005392711027525365,
        -0.02945239096879959,
        -0.04244452342391014,
        -0.03165606036782265,
        -0.00674798246473074,
        0.017190590500831604,
        0.031108776107430458,
        0.0332687646150589,
        0.022887442260980606,
        0.0006626985850743949,
        -0.024899717420339584,
        -0.038331288844347,
        -0.030507804825901985,
        0.029575053602457047,
        0.033126622438430786,
        0.023670051246881485,
        0.0012433180818334222,
        -0.024378644302487373,
        -0.03728340193629265,
        -0.029677610844373703,
        -0.009196006692945957,
        0.011860810220241547,
        0.026265539228916168,
        0.031976472586393356,
        0.025962095707654953,
        0.0061595868319272995,
        -0.02045596018433571,
        -0.03774508833885193,
        -0.034158527851104736,
    ];
    /// Hidden states after block 1, `[1, 8, 16]` flattened.
    const GOLDEN_BLOCK_1: &[f64] = &[
        0.043472398072481155,
        0.049289267510175705,
        0.036398839205503464,
        0.0016588476719334722,
        -0.03993077948689461,
        -0.06040225550532341,
        -0.04590022936463356,
        -0.010788703337311745,
        0.022739943116903305,
        0.04343966394662857,
        0.05017533153295517,
        0.0387626476585865,
        0.005262129940092564,
        -0.03831605985760689,
        -0.06354718655347824,
        -0.05165065452456474,
        0.04311542585492134,
        0.05386921390891075,
        0.04301895573735237,
        0.006134563125669956,
        -0.04042947292327881,
        -0.0655587837100029,
        -0.05245696380734444,
        -0.014598306268453598,
        0.024085089564323425,
        0.04908875748515129,
        0.05655767768621445,
        0.041843172162771225,
        0.0030929017812013626,
        -0.04436368867754936,
        -0.06964831054210663,
        -0.05395087972283363,
        0.046044811606407166,
        0.049743711948394775,
        0.034452393651008606,
        -0.0014602456940338016,
        -0.04224912077188492,
        -0.060458485037088394,
        -0.043650995939970016,
        -0.0076632569544017315,
        0.02476854994893074,
        0.04309019073843956,
        0.04766646400094032,
        0.035680998116731644,
        0.00355729553848505,
        -0.0375661663711071,
        -0.06082190200686455,
        -0.04866373538970947,
        0.04228202626109123,
        0.05096568539738655,
        0.03988008201122284,
        0.004745151847600937,
        -0.03919132053852081,
        -0.062474917620420456,
        -0.04947865754365921,
        -0.013606925494968891,
        0.022462181746959686,
        0.04587734863162041,
        0.053788427263498306,
        0.0412667877972126,
        0.005073421634733677,
        -0.0410783477127552,
        -0.0671340599656105,
        -0.05379911884665489,
        0.04885377362370491,
        0.052005618810653687,
        0.03474164754152298,
        -0.0033289087004959583,
        -0.04509282857179642,
        -0.06248347461223602,
        -0.04356970638036728,
        -0.0055340067483484745,
        0.02760237827897072,
        0.044838301837444305,
        0.047221384942531586,
        0.03332480043172836,
        0.0007807155488990247,
        -0.03900760039687157,
        -0.060021933168172836,
        -0.04611929506063461,
        0.043374188244342804,
        0.04932115972042084,
        0.036540497094392776,
        0.0018208068795502186,
        -0.03985097259283066,
        -0.06045488268136978,
        -0.04605232551693916,
        -0.010943831875920296,
        0.02267974615097046,
        0.043512407690286636,
        0.0503351092338562,
        0.03890843316912651,
        0.005301701836287975,
        -0.03840772807598114,
        -0.06371193379163742,
        -0.051784686744213104,
        0.05063830688595772,
        0.05505985766649246,
        0.03713664785027504,
        -0.003108344739302993,
        -0.04718548059463501,
        -0.06556811183691025,
        -0.04569302126765251,
        -0.0053566498681902885,
        0.0299694761633873,
        0.04789920523762703,
        0.04903996363282204,
        0.032751668244600296,
        -0.0018213156145066023,
        -0.041993435472249985,
        -0.061506111174821854,
        -0.045159678906202316,
        0.045901793986558914,
        0.0496736504137516,
        0.03449933975934982,
        -0.0013258397812023759,
        -0.0421123132109642,
        -0.060405537486076355,
        -0.04371579363942146,
        -0.007804451044648886,
        0.024640072137117386,
        0.04305558279156685,
        0.04774773493409157,
        0.035826727747917175,
        0.0036752931773662567,
        -0.037550512701272964,
        -0.06091820448637009,
        -0.04881160333752632,
    ];
    const GOLDEN_BLOCK_OUTPUTS: &[&[f64]] = &[GOLDEN_BLOCK_0, GOLDEN_BLOCK_1];

    /// Final output, `[1, 16, 2, 4, 4]` flattened.
    const GOLDEN_OUTPUT: &[f64] = &[
        0.07076235860586166,
        -0.011110587045550346,
        0.06678541004657745,
        -0.007043812423944473,
        -0.023730847984552383,
        0.02337062731385231,
        -0.027819637209177017,
        0.027413252741098404,
        0.07970508188009262,
        -0.02025090716779232,
        0.06563816219568253,
        -0.005890287458896637,
        -0.014545334503054619,
        0.014293091371655464,
        -0.028960207477211952,
        0.028521854430437088,
        0.08792239427566528,
        -0.028688035905361176,
        0.07039324194192886,
        -0.010734288021922112,
        -0.006029079668223858,
        0.00583970732986927,
        -0.02410806342959404,
        0.023742467164993286,
        0.09128914773464203,
        -0.03214418888092041,
        0.07923939824104309,
        -0.019773811101913452,
        -0.0025411415845155716,
        0.0023781433701515198,
        -0.015025902539491653,
        0.014769110828638077,
        0.24809513986110687,
        -0.21758344769477844,
        0.24822230637073517,
        -0.21718278527259827,
        0.22879919409751892,
        -0.2417272925376892,
        0.2278774231672287,
        -0.24029971659183502,
        0.24784258008003235,
        -0.21851655840873718,
        0.2481098473072052,
        -0.2169232964515686,
        0.23090246319770813,
        -0.24496562778949738,
        0.22747519612312317,
        -0.23976150155067444,
        0.2473198026418686,
        -0.21909351646900177,
        0.2480982542037964,
        -0.21753785014152527,
        0.23256953060626984,
        -0.24769501388072968,
        0.22870568931102753,
        -0.24158738553524017,
        0.2471105456352234,
        -0.219334676861763,
        0.24786429107189178,
        -0.21847625076770782,
        0.23325705528259277,
        -0.24881742894649506,
        0.2308007925748825,
        -0.2448042631149292,
        0.04122637212276459,
        -0.05590026080608368,
        0.04525495320558548,
        -0.05980435758829117,
        0.11402611434459686,
        -0.1438997983932495,
        0.11774064600467682,
        -0.1473628133535385,
        0.03218108043074608,
        -0.04713878035545349,
        0.046356525272130966,
        -0.06085251271724701,
        0.10569454729557037,
        -0.13613706827163696,
        0.11871790885925293,
        -0.14825290441513062,
        0.023751530796289444,
        -0.03893590718507767,
        0.04159673675894737,
        -0.05625804513692856,
        0.09785514324903488,
        -0.12879183888435364,
        0.114365354180336,
        -0.14421483874320984,
        0.02029982954263687,
        -0.03557766228914261,
        0.03265559300780296,
        -0.04759952425956726,
        0.09464634209871292,
        -0.12578599154949188,
        0.1061338409781456,
        -0.13654756546020508,
        -0.2352745682001114,
        0.21576064825057983,
        -0.2337660938501358,
        0.2137749046087265,
        -0.17011100053787231,
        0.16746073961257935,
        -0.16768106818199158,
        0.16462716460227966,
        -0.23869441449642181,
        0.22025099396705627,
        -0.2332063466310501,
        0.21308980882167816,
        -0.17559696733951569,
        0.1738508641719818,
        -0.16688208281993866,
        0.1637275516986847,
        -0.24159418046474457,
        0.22415843605995178,
        -0.235127255320549,
        0.21556980907917023,
        -0.18044692277908325,
        0.17956244945526123,
        -0.169879749417305,
        0.1671929657459259,
        -0.24278634786605835,
        0.22576303780078888,
        -0.23852351307868958,
        0.22002364695072174,
        -0.1824372410774231,
        0.18190528452396393,
        -0.1753169596195221,
        0.17352284491062164,
        -0.15400272607803345,
        0.15410056710243225,
        -0.1574188470840454,
        0.15719841420650482,
        -0.1615368127822876,
        0.2100430727005005,
        -0.1642647385597229,
        0.21235564351081848,
        -0.14634592831134796,
        0.14716222882270813,
        -0.15829315781593323,
        0.1579684317111969,
        -0.1554325819015503,
        0.20487478375434875,
        -0.16491763293743134,
        0.212880477309227,
        -0.13909371197223663,
        0.1405458152294159,
        -0.15431329607963562,
        0.15438085794448853,
        -0.1495622992515564,
        0.19984853267669678,
        -0.1617821604013443,
        0.2102493941783905,
        -0.13612602651119232,
        0.13783904910087585,
        -0.1467510461807251,
        0.14753064513206482,
        -0.1471615880727768,
        0.1977938860654831,
        -0.15575818717479706,
        0.20515212416648865,
        0.15027469396591187,
        -0.1578538864850998,
        0.14737923443317413,
        -0.15461038053035736,
        0.12510694563388824,
        -0.06916539371013641,
        0.12156941741704941,
        -0.0653928816318512,
        0.15680329501628876,
        -0.16516129672527313,
        0.1464645266532898,
        -0.1536126583814621,
        0.13307127356529236,
        -0.07765386998653412,
        0.12050534784793854,
        -0.0642801970243454,
        0.1626475304365158,
        -0.17175504565238953,
        0.15000133216381073,
        -0.15754924714565277,
        0.14030462503433228,
        -0.08540618419647217,
        0.12477606534957886,
        -0.06881383061408997,
        0.16504457592964172,
        -0.17445863783359528,
        0.15646791458129883,
        -0.16478434205055237,
        0.14326965808868408,
        -0.08858321607112885,
        0.13265906274318695,
        -0.07721322774887085,
        0.19789128005504608,
        -0.2359178364276886,
        0.2001318335533142,
        -0.2376987636089325,
        0.22225850820541382,
        -0.21636734902858734,
        0.2235502302646637,
        -0.21714822947978973,
        0.1928851455450058,
        -0.23194634914398193,
        0.20063474774360657,
        -0.2380637228488922,
        0.21938790380954742,
        -0.21464547514915466,
        0.22377105057239532,
        -0.21722128987312317,
        0.18800576031208038,
        -0.22800707817077637,
        0.19809085130691528,
        -0.23607441782951355,
        0.21645444631576538,
        -0.21276672184467316,
        0.22236953675746918,
        -0.2164309322834015,
        0.18601131439208984,
        -0.22639799118041992,
        0.19315412640571594,
        -0.2321617305278778,
        0.21525755524635315,
        -0.2120020091533661,
        0.2195461392402649,
        -0.21474388241767883,
        -0.07365060597658157,
        0.03961174190044403,
        -0.06984548270702362,
        0.03564513474702835,
        -0.04026396572589874,
        0.001584470272064209,
        -0.036202020943164825,
        -0.002505081705749035,
        -0.08221174031496048,
        0.04853159189224243,
        -0.0687265694141388,
        0.03449925780296326,
        -0.04939378798007965,
        0.010772048495709896,
        -0.03504828363656998,
        -0.0036474373191595078,
        -0.09003706276416779,
        0.05672474205493927,
        -0.0732962116599083,
        0.03924350440502167,
        -0.057818152010440826,
        0.01928715780377388,
        -0.03988802433013916,
        0.0012070946395397186,
        -0.09324388951063156,
        0.06008163094520569,
        -0.08176715672016144,
        0.04806719720363617,
        -0.061269134283065796,
        0.022774677723646164,
        -0.04891733080148697,
        0.010291464626789093,
        -0.21636271476745605,
        0.23537516593933105,
        -0.21705834567546844,
        0.23554565012454987,
        -0.2567875385284424,
        0.22625420987606049,
        -0.256430059671402,
        0.22537469863891602,
        -0.21483251452445984,
        0.23502524197101593,
        -0.21710701286792755,
        0.23544535040855408,
        -0.25762373208999634,
        0.22826257348060608,
        -0.256182461977005,
        0.22498396039009094,
        -0.21313028037548065,
        0.23441246151924133,
        -0.21641841530799866,
        0.23538225889205933,
        -0.2581106126308441,
        0.22984105348587036,
        -0.25674593448638916,
        0.2261645793914795,
        -0.21243782341480255,
        0.23416629433631897,
        -0.21492095291614532,
        0.23505207896232605,
        -0.2583148777484894,
        0.23049229383468628,
        -0.2575885057449341,
        0.228165864944458,
        0.007013374008238316,
        0.052978456020355225,
        0.002925807610154152,
        0.05701429396867752,
        -0.0864100456237793,
        0.08415032178163528,
        -0.09032685309648514,
        0.08788278698921204,
        0.016195787116885185,
        0.04391653090715408,
        0.0017871484160423279,
        0.05811943858861923,
        -0.07761971652507782,
        0.07577816396951675,
        -0.0913800597190857,
        0.08886649459600449,
        0.02471223473548889,
        0.035474590957164764,
        0.006636364385485649,
        0.0533495768904686,
        -0.06939305365085602,
        0.06790393590927124,
        -0.08676909655332565,
        0.0844913125038147,
        0.028200199827551842,
        0.03201776370406151,
        0.01571529358625412,
        0.044391825795173645,
        -0.06602499634027481,
        0.06468082219362259,
        -0.0780818834900856,
        0.07621949166059494,
        0.2402205616235733,
        -0.20178990066051483,
        0.2392566055059433,
        -0.20032177865505219,
        0.20559664070606232,
        -0.21070370078086853,
        0.2036488950252533,
        -0.20830875635147095,
        0.24241851270198822,
        -0.20511920750141144,
        0.23884296417236328,
        -0.19977279007434845,
        0.21000178158283234,
        -0.21611124277114868,
        0.20297369360923767,
        -0.20751862227916718,
        0.2441740334033966,
        -0.2079339176416397,
        0.2401231974363327,
        -0.20164631307125092,
        0.21382881700992584,
        -0.2208867371082306,
        0.20540925860404968,
        -0.2104756087064743,
        0.2448977380990982,
        -0.20909127593040466,
        0.24231187999248505,
        -0.20495305955410004,
        0.21540051698684692,
        -0.22284658253192902,
        0.20977891981601715,
        -0.21583537757396698,
        0.10144960135221481,
        -0.11346099525690079,
        0.10514578223228455,
        -0.11690077930688858,
        0.16766054928302765,
        -0.19204704463481903,
        0.1707865446805954,
        -0.19480715692043304,
        0.09315957129001617,
        -0.10575080662965775,
        0.10611650347709656,
        -0.11778303235769272,
        0.16065874695777893,
        -0.18587039411067963,
        0.17156562209129333,
        -0.19547003507614136,
        0.08535586297512054,
        -0.09845167398452759,
        0.10178706794977188,
        -0.11377381533384323,
        0.1539858877658844,
        -0.17993508279323578,
        0.16794350743293762,
        -0.19229543209075928,
        0.08216173946857452,
        -0.09546474367380142,
        0.09359676390886307,
        -0.10615863651037216,
        0.15125596523284912,
        -0.1775076687335968,
        0.16103042662143707,
        -0.18619972467422485,
        -0.20332029461860657,
        0.17645390331745148,
        -0.20085567235946655,
        0.17358924448490143,
        -0.12481887638568878,
        0.11715918779373169,
        -0.12160195410251617,
        0.11364361643791199,
        -0.2088841199874878,
        0.1829136312007904,
        -0.20004788041114807,
        0.1726820319890976,
        -0.13206681609153748,
        0.12507449090480804,
        -0.12061043083667755,
        0.11258433759212494,
        -0.21380797028541565,
        0.1886918544769287,
        -0.20308592915534973,
        0.1761833280324936,
        -0.138603076338768,
        0.132259801030159,
        -0.12451660633087158,
        0.1168302595615387,
        -0.215828537940979,
        0.19106194376945496,
        -0.20859995484352112,
        0.18258190155029297,
        -0.14128315448760986,
        0.1352051943540573,
        -0.131693035364151,
        0.12466492503881454,
        -0.20149189233779907,
        0.19536767899990082,
        -0.20418742299079895,
        0.19764436781406403,
        -0.19667837023735046,
        0.23814426362514496,
        -0.19849824905395508,
        0.2394770085811615,
        -0.19546082615852356,
        0.19028019905090332,
        -0.20483019948005676,
        0.19815827906131744,
        -0.19261930882930756,
        0.2351813167333603,
        -0.19887477159500122,
        0.23970980942249298,
        -0.18965628743171692,
        0.18532706797122955,
        -0.20173421502113342,
        0.19557064771652222,
        -0.18860022723674774,
        0.23216328024864197,
        -0.1968386173248291,
        0.23825910687446594,
        -0.1872825026512146,
        0.18330241739749908,
        -0.19578266143798828,
        0.19055333733558655,
        -0.1869584321975708,
        0.23093172907829285,
        -0.19283923506736755,
        0.2353443205356598,
        0.09904423356056213,
        -0.10216130316257477,
        0.09548517316579819,
        -0.09837227314710617,
        0.06539230793714523,
        -0.007148256525397301,
        0.0614364892244339,
        -0.003091590479016304,
        0.10705667734146118,
        -0.11068658530712128,
        0.09441643953323364,
        -0.09725640714168549,
        0.0742882788181305,
        -0.016266562044620514,
        0.06029210984706879,
        -0.0019377805292606354,
        0.1143372654914856,
        -0.11847583949565887,
        0.09871143847703934,
        -0.10180830210447311,
        0.0824563205242157,
        -0.02467721328139305,
        0.0650249794125557,
        -0.006772706285119057,
        0.11732158064842224,
        -0.12166795879602432,
        0.10664185881614685,
        -0.11024396121501923,
        0.08580298721790314,
        -0.028122618794441223,
        0.07382521778345108,
        -0.01579079031944275,
        0.22482754290103912,
        -0.25480297207832336,
        0.22607803344726562,
        -0.25554123520851135,
        0.2328149676322937,
        -0.21919742226600647,
        0.23302873969078064,
        -0.21888312697410583,
        0.2220495492219925,
        -0.25317683815956116,
        0.226286843419075,
        -0.2556021213531494,
        0.23236775398254395,
        -0.21993659436702728,
        0.23294062912464142,
        -0.21864748001098633,
        0.219200998544693,
        -0.25138622522354126,
        0.22493471205234528,
        -0.25486260652542114,
        0.23166495561599731,
        -0.22033336758613586,
        0.23282603919506073,
        -0.219159796833992,
        0.2180389165878296,
        -0.2506575882434845,
        0.22220300137996674,
        -0.2532702684402466,
        0.23138196766376495,
        -0.22050069272518158,
        0.23239965736865997,
        -0.21990644931793213,
    ];

    /// Observed agreement with diffusers is 4.5e-8 -- f32 round-off. The brief
    /// allowed 1e-4, but the fixture's 0.02-scale weights make several
    /// sub-paths contribute only weakly to the output, so a loose bound lets
    /// real mistakes through: swapping the time embedding's cos/sin halves
    /// moves the output by just 6e-6. Bound it two orders below that, still
    /// ~20x above the observed noise.
    const GOLDEN_TOLERANCE: f64 = 1e-6;

    const TINY_DIM: usize = 16;
    const TINY_HEADS: usize = 2;
    const TINY_LAYERS: usize = 2;

    fn tiny_config() -> WanTransformerConfig {
        // Exactly the fixture's config: dim 16, 2 heads (head_dim 8), 2 layers,
        // ffn 32, in/out 16, text_dim 32, freq_dim 16, rope_max_seq_len 32.
        WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(TINY_DIM, TINY_HEADS, TINY_LAYERS)
        }
    }

    /// The fixture's deterministic parameter filling:
    /// `w[i] = 0.02 * sin(0.7 * i + offset(name))` with
    /// `offset(name) = (sum of utf8 bytes) % 97 * 0.1`, row-major, f64 then f32.
    fn synth(name: &str, shape: &[usize]) -> Tensor {
        let offset = (name.bytes().map(u64::from).sum::<u64>() % 97) as f64 * 0.1;
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|i| (0.02 * (0.7 * i as f64 + offset).sin()) as f32)
            .collect();
        Tensor::from_vec(values, shape, &Device::Cpu).expect("synthesized tensor")
    }

    fn golden_model() -> WanTransformer {
        let mut tensors = HashMap::new();
        for (name, shape) in GOLDEN_PARAMS {
            let original = diffusers_to_original(name)
                .unwrap_or_else(|| panic!("no original name for diffusers key {name}"));
            tensors.insert(original, synth(name, shape));
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        WanTransformer::from_var_builder(vb, tiny_config()).expect("golden model loads")
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// A tiny model at weight scale 1.0 rather than the fixture's 0.02, with
    /// one named original-layout parameter optionally shifted.
    ///
    /// The golden fixture cannot see the attention internals: at 0.02-scale
    /// weights the pre-softmax logits sit around 6e-4, so the softmax is
    /// effectively uniform and the attention output is `mean(v)` no matter what
    /// q and k did. Scaling the weights up puts the logits at O(1) and makes
    /// the branch observable, which is enough to prove a parameter is wired in
    /// even without a reference to compare against.
    fn sensitivity_model(tweak: Option<&str>) -> WanTransformer {
        let mut tensors = HashMap::new();
        for (name, shape) in GOLDEN_PARAMS {
            let original = diffusers_to_original(name).expect("renameable");
            let mut tensor = synth(name, shape).affine(50.0, 0.0).expect("scaled");
            if tweak == Some(original.as_str()) {
                tensor = tensor.affine(1.0, 0.25).expect("shifted");
            }
            tensors.insert(original, tensor);
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        WanTransformer::from_var_builder(vb, tiny_config()).expect("sensitivity model loads")
    }

    fn sensitivity_forward(tweak: Option<&str>) -> Vec<f32> {
        let model = sensitivity_model(tweak);
        let x = synth("input.hidden_states", &[1, 16, 2, 4, 4]);
        let context = synth("input.encoder_hidden_states", &[1, 6, 32]);
        let timestep = Tensor::from_vec(vec![500f32], 1, &Device::Cpu).unwrap();
        values(&model.forward(&x, &timestep, &context).unwrap())
    }

    /// Every parameter the golden fixture cannot exercise still has to be
    /// reachable. Shifting each one must move the output.
    #[test]
    fn attention_norms_are_actually_wired_in() {
        let base = sensitivity_forward(None);
        assert!(
            base.iter().all(|v| v.is_finite()),
            "sensitivity model diverged"
        );
        for key in [
            // Dropping the q/k RMS norms is invisible to the golden fixture.
            "blocks.0.self_attn.norm_q.weight",
            "blocks.0.self_attn.norm_k.weight",
            "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.cross_attn.norm_k.weight",
            // So is running cross-attention off `norm2` instead of `norm3`,
            // which is exactly the mistake the diffusers rename swap invites.
            "blocks.0.norm3.weight",
            "blocks.0.norm3.bias",
        ] {
            let tweaked = sensitivity_forward(Some(key));
            let err = max_abs_diff_f32(&base, &tweaked);
            // Most of these move the output by >1e-3; cross-attention's q norm
            // has the least leverage at 5e-5, still ~500x over f32 noise.
            assert!(
                err > 1e-5,
                "shifting {key} changed the output by only {err:e}"
            );
        }
    }

    /// Goldens are held at the fixture's own f64 precision; the model runs in
    /// f32, so widening the result to compare is the honest direction.
    fn max_abs_diff(got: &[f32], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len(), "length mismatch");
        got.iter()
            .zip(want)
            .map(|(a, b)| (f64::from(*a) - b).abs())
            .fold(0.0f64, f64::max)
    }

    /// Same, for comparing two of our own f32 runs against each other.
    fn max_abs_diff_f32(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "length mismatch");
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Full-model parity against diffusers 0.38.0. Every weight is synthesized
    /// under its *diffusers* name and inserted under its *original* name, so a
    /// wrong rename (the `norm2`/`norm3` swap especially) fails here just as
    /// loudly as wrong math.
    #[test]
    fn golden_parity_with_diffusers() {
        let model = golden_model();
        let x = synth("input.hidden_states", &[1, 16, 2, 4, 4]);
        let context = synth("input.encoder_hidden_states", &[1, 6, 32]);
        let timestep = Tensor::from_vec(vec![500f32], 1, &Device::Cpu).unwrap();

        // Re-run the block stack by hand so the per-block probes can be read.
        let (batch, _, frames, height, width) = x.dims5().unwrap();
        let grid = (frames, height / 2, width / 2);
        let (cos, sin) = model
            .rope
            .freqs(grid.0, grid.1, grid.2, &Device::Cpu)
            .unwrap();
        let mut hidden = model.patchify(&x).unwrap();
        let (temb, timestep_proj) = model.embed_timestep(&timestep, batch).unwrap();
        let ctx = model
            .text_embedding_out
            .forward(
                &model
                    .text_embedding_in
                    .forward(&context)
                    .unwrap()
                    .gelu()
                    .unwrap(),
            )
            .unwrap();

        let mut worst = 0.0f64;
        for (i, block) in model.blocks.iter().enumerate() {
            hidden = block
                .forward(&hidden, &ctx, &timestep_proj, &cos, &sin)
                .unwrap();
            let err = max_abs_diff(&values(&hidden), GOLDEN_BLOCK_OUTPUTS[i]);
            assert!(err < GOLDEN_TOLERANCE, "block {i} max error {err:e}");
            worst = worst.max(err);
        }
        let _ = temb;

        let out = model.forward(&x, &timestep, &context).unwrap();
        assert_eq!(out.dims(), &[1, 16, 2, 4, 4]);
        let err = max_abs_diff(&values(&out), GOLDEN_OUTPUT);
        assert!(err < GOLDEN_TOLERANCE, "final max error {err:e}");
        worst = worst.max(err);
        assert!(worst < GOLDEN_TOLERANCE, "worst error {worst:e}");
    }

    /// A `[batch, tokens]` timestep whose entries are all equal must reproduce
    /// the scalar path. This is the branch Wan 2.2 TI2V drives with a distinct
    /// timestep on the conditioning frame.
    #[test]
    fn per_token_timestep_matches_the_scalar_path() {
        let model = golden_model();
        let x = synth("input.hidden_states", &[1, 16, 2, 4, 4]);
        let context = synth("input.encoder_hidden_states", &[1, 6, 32]);
        let scalar = Tensor::from_vec(vec![500f32], 1, &Device::Cpu).unwrap();
        // 2 frames x 2 x 2 patches = 8 tokens.
        let per_token = Tensor::from_vec(vec![500f32; 8], (1, 8), &Device::Cpu).unwrap();

        let a = model.forward(&x, &scalar, &context).unwrap();
        let b = model.forward(&x, &per_token, &context).unwrap();
        assert_eq!(a.dims(), b.dims());
        let err = max_abs_diff_f32(&values(&a), &values(&b));
        // Observed 1.5e-8 -- f32 rounding, not a different code path.
        assert!(err < 1e-6, "per-token vs scalar max error {err:e}");
    }

    /// A per-token timestep that is *not* uniform must actually change the
    /// output, otherwise the branch is silently ignoring its extra axis.
    #[test]
    fn per_token_timestep_is_not_ignored() {
        let model = golden_model();
        let x = synth("input.hidden_states", &[1, 16, 2, 4, 4]);
        let context = synth("input.encoder_hidden_states", &[1, 6, 32]);
        // Half the tokens held at t = 0 while the rest denoise at t = 999 is
        // the shape TI2V drives: a clean conditioning frame beside noisy ones.
        let low = Tensor::from_vec(vec![0f32; 8], (1, 8), &Device::Cpu).unwrap();
        let high = Tensor::from_vec(vec![999f32; 8], (1, 8), &Device::Cpu).unwrap();
        let mut split = vec![999f32; 8];
        split[..4].fill(0.0);
        let split = Tensor::from_vec(split, (1, 8), &Device::Cpu).unwrap();

        let low = values(&model.forward(&x, &low, &context).unwrap());
        let high = values(&model.forward(&x, &high, &context).unwrap());
        let split = values(&model.forward(&x, &split, &context).unwrap());

        // The toy weights are 0.02-scale so the timestep's absolute influence
        // is small, but it sits four orders of magnitude above the 1.5e-8 at
        // which a uniform vector reproduces the scalar path.
        for (other, label) in [(&low, "all-zero"), (&high, "all-999")] {
            let err = max_abs_diff_f32(&split, other);
            assert!(
                err > 1e-5,
                "a split timestep vector matched the {label} one ({err:e}) - the per-token \
                 axis is being collapsed"
            );
        }
    }

    #[test]
    fn rename_table_round_trips_every_fixture_name() {
        for (name, _) in GOLDEN_PARAMS {
            let original = diffusers_to_original(name)
                .unwrap_or_else(|| panic!("no original name for {name}"));
            let back = original_to_diffusers(&original)
                .unwrap_or_else(|| panic!("no diffusers name for {original}"));
            assert_eq!(&back, name, "round trip via {original}");
        }
        // The swap is the part most likely to be transcribed backwards.
        assert_eq!(
            diffusers_to_original("blocks.7.norm2.weight").unwrap(),
            "blocks.7.norm3.weight"
        );
        assert_eq!(
            diffusers_to_original("blocks.7.norm3.bias").unwrap(),
            "blocks.7.norm2.bias"
        );
        // Bare parameters keep their lack of a leaf.
        assert_eq!(
            diffusers_to_original("scale_shift_table").unwrap(),
            "head.modulation"
        );
        assert_eq!(
            diffusers_to_original("blocks.0.scale_shift_table").unwrap(),
            "blocks.0.modulation"
        );
        // Out-of-scope names are declined, not mangled.
        assert!(diffusers_to_original("condition_embedder.image_embedder.norm1.weight").is_none());
        assert!(diffusers_to_original("vace_blocks.0.proj.weight").is_none());
    }

    fn key_set(varmap: &VarMap) -> Vec<String> {
        let mut keys: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }

    /// The loader must request exactly the original checkpoint layout,
    /// including the 5-D `patch_embedding.weight`.
    #[test]
    fn loads_the_original_checkpoint_key_layout() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        WanTransformer::from_var_builder(vb, tiny_config())
            .expect("tiny model materializes from a VarMap");
        let keys = key_set(&varmap);
        let has = |k: &str| keys.iter().any(|s| s == k);

        for key in [
            "patch_embedding.weight",
            "patch_embedding.bias",
            "text_embedding.0.weight",
            "text_embedding.2.bias",
            "time_embedding.0.weight",
            "time_embedding.2.bias",
            "time_projection.1.weight",
            "time_projection.1.bias",
            "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.q.bias",
            "blocks.0.self_attn.o.bias",
            "blocks.0.self_attn.norm_q.weight",
            "blocks.0.self_attn.norm_k.weight",
            "blocks.0.cross_attn.v.weight",
            "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.norm3.weight",
            "blocks.0.norm3.bias",
            "blocks.0.ffn.0.weight",
            "blocks.0.ffn.2.bias",
            "blocks.0.modulation",
            "blocks.1.modulation",
            "head.head.weight",
            "head.head.bias",
            "head.modulation",
        ] {
            assert!(has(key), "missing {key}");
        }

        // norm1 and norm2 are parameter-free; norm3 is the affine one.
        assert!(!has("blocks.0.norm1.weight"));
        assert!(!has("blocks.0.norm2.weight"));
        // Diffusers names must not leak into the load path.
        assert!(!keys.iter().any(|k| k.contains("scale_shift_table")));
        assert!(!keys.iter().any(|k| k.contains("attn1.")));
        assert!(!keys.iter().any(|k| k.contains("ffn.net.")));
        // Only the declared layer count is requested.
        assert!(!keys.iter().any(|k| k.starts_with("blocks.2.")));

        let data = varmap.data().lock().unwrap();
        assert_eq!(
            data["patch_embedding.weight"].as_tensor().dims(),
            &[TINY_DIM, TINY_DIM, 1, 2, 2],
            "patch_embedding must be read as the checkpoint's 5-D Conv3d weight"
        );
        assert_eq!(
            data["blocks.0.modulation"].as_tensor().dims(),
            &[1, 6, TINY_DIM]
        );
        assert_eq!(
            data["head.modulation"].as_tensor().dims(),
            &[1, 2, TINY_DIM]
        );
    }

    /// candle's GGUF reader already converts ggml's fastest-varying-first dims
    /// into torch order (`gguf_file.rs`'s `dimensions.reverse()`), so a loader
    /// must NOT reverse them again. Pin that, because the whole quantized path
    /// silently transposes if it ever changes.
    #[test]
    fn candle_presents_gguf_dims_in_torch_order() {
        use candle_core::quantized::{gguf_file, QTensor};

        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dims.gguf");

        // A [3, 5] torch weight: out_features 3, in_features 5.
        let weight = Tensor::arange(0f32, 15f32, &device)
            .unwrap()
            .reshape((3, 5))
            .unwrap();
        let q = QTensor::quantize(&weight, candle_core::quantized::GgmlDType::F32).unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        let arch = gguf_file::Value::String("wan".to_string());
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &[("w", &q)]).unwrap();
        drop(file);

        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        assert_eq!(
            content.tensor_infos["w"].shape.dims(),
            &[3, 5],
            "candle must hand back torch order, not ggml's reversed [5, 3]"
        );
        let back = content.tensor(&mut file, "w", &device).unwrap();
        assert_eq!(back.shape().dims(), &[3, 5]);
    }

    /// Full GGUF round trip, 5-D patch embedding included.
    ///
    /// Every Wan GGUF carries one 5-D tensor — `patch_embedding.weight`,
    /// `[dim, in_dim, pt, ph, pw]` — which candle refused until the compat
    /// branch raised `GGUF_MAX_TENSOR_DIMS` to 5 (`3a49a729`). This writes a
    /// complete tiny model, reads it back through `from_gguf`, and requires the
    /// forward to match the safetensors path that produced the weights.
    ///
    /// Quantizing at F32 is lossless, so any divergence is a naming, dim-order,
    /// or dispatch bug rather than quantization error.
    #[test]
    fn gguf_round_trip_reproduces_the_safetensors_forward() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        use candle_nn::{VarBuilder as NnVarBuilder, VarMap};

        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny_wan.gguf");

        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        let plain = WanTransformer::from_var_builder(
            NnVarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();

        // Every tensor the model asked for, under its bare GGUF name — the
        // shipped files carry no `model.diffusion_model.` prefix.
        let quantized: Vec<(String, QTensor)> = {
            let data = varmap.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| {
                    (
                        name.clone(),
                        QTensor::quantize(var.as_tensor(), GgmlDType::F32).unwrap(),
                    )
                })
                .collect()
        };
        assert!(
            quantized
                .iter()
                .any(|(name, q)| name == "patch_embedding.weight" && q.shape().rank() == 5),
            "the fixture must exercise the 5-D patch embedding"
        );

        let refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, q)| (name.as_str(), q))
            .collect();
        let arch = gguf_file::Value::String("wan".to_string());
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs).unwrap();
        drop(file);

        let from_gguf = WanTransformer::from_gguf(&path, config.clone(), &device).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, config.in_dim, 1, 4, 4), &device).unwrap();
        let timestep = Tensor::from_vec(vec![500f32], 1, &device).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 4, config.text_dim), &device).unwrap();

        let want = plain.forward(&x, &timestep, &context).unwrap();
        let got = from_gguf.forward(&x, &timestep, &context).unwrap();
        assert_eq!(want.dims(), got.dims());

        let want_v: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let err = want_v
            .iter()
            .zip(&got_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            err < 1e-4,
            "GGUF forward diverged from safetensors by {err:e}"
        );

        // Per-token timesteps travel the same path.
        let per_token = Tensor::from_vec(vec![500f32; 4], (1, 4), &device).unwrap();
        let want = plain.forward(&x, &per_token, &context).unwrap();
        let got = from_gguf.forward(&x, &per_token, &context).unwrap();
        let err = want
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(got.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-4, "per-token GGUF forward diverged by {err:e}");
    }

    /// The quantized source serves every tensor class the real files contain,
    /// and refuses a shape the config did not ask for.
    #[test]
    fn gguf_quantized_source_serves_every_tensor_class() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        use candle_nn::{VarBuilder as NnVarBuilder, VarMap};

        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("classes.gguf");

        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            NnVarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();
        let quantized: Vec<(String, QTensor)> = {
            let data = varmap.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| {
                    (
                        name.clone(),
                        QTensor::quantize(var.as_tensor(), GgmlDType::F32).unwrap(),
                    )
                })
                .collect()
        };
        let refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, q)| (name.as_str(), q))
            .collect();
        let arch = gguf_file::Value::String("wan".to_string());
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs).unwrap();
        drop(file);

        let vb = mold_candle::quantized::VarBuilder::from_gguf(&path, &device).unwrap();
        let source = WanWeights::quantized(vb);
        let dim = config.dim;

        // 2-D linear through QMatMul.
        let linear = source.pp("blocks.0.self_attn.q").linear(dim, dim).unwrap();
        let probe = Tensor::randn(0f32, 1f32, (1, 4, dim), &device).unwrap();
        assert_eq!(linear.forward(&probe).unwrap().dims(), &[1, 4, dim]);

        // 1-D norm gain, 3-D modulation, 5-D patch embedding.
        assert_eq!(
            source
                .pp("blocks.0.self_attn.norm_q")
                .tensor(dim, "weight", candle_nn::Init::Const(1.0))
                .unwrap()
                .dims(),
            &[dim]
        );
        assert_eq!(
            source
                .pp("blocks.0")
                .tensor((1, 6, dim), "modulation", TEST_WEIGHT_INIT)
                .unwrap()
                .dims(),
            &[1, 6, dim],
            "3-D modulation must come back in torch order"
        );
        let (pt, ph, pw) = config.patch_size;
        assert_eq!(
            source
                .pp("patch_embedding")
                .tensor((dim, config.in_dim, pt, ph, pw), "weight", TEST_WEIGHT_INIT)
                .unwrap()
                .dims(),
            &[dim, config.in_dim, pt, ph, pw],
            "5-D patch embedding must come back in torch order"
        );

        // A shape the config does not expect is refused, not silently reshaped.
        assert!(source
            .pp("blocks.0")
            .tensor((1, 2, dim), "modulation", TEST_WEIGHT_INIT)
            .is_err());
    }

    /// A GGUF missing a tensor the config demands must fail at load, naming the
    /// key — not produce a partly-initialised model.
    #[test]
    fn gguf_load_reports_a_missing_tensor() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};

        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("incomplete.gguf");
        let q = QTensor::quantize(
            &Tensor::zeros((16, 16), DType::F32, &device).unwrap(),
            GgmlDType::F32,
        )
        .unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        let arch = gguf_file::Value::String("wan".to_string());
        gguf_file::write(
            &mut file,
            &[("general.architecture", &arch)],
            &[("blocks.0.self_attn.q.weight", &q)],
        )
        .unwrap();
        drop(file);

        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let error = match WanTransformer::from_gguf(&path, config, &device) {
            Ok(_) => panic!("an incomplete GGUF must not load"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("cannot find tensor"),
            "unexpected error: {error}"
        );
    }

    /// The `e4m3` maximum finite magnitude. Upstream picks the per-tensor scale
    /// so the largest weight lands on it (`comfy/float.py`'s stochastic
    /// rounding operates on `weight / scale_weight`).
    const FP8_E4M3_MAX: f32 = 448.0;

    /// True for the tensors the shipped fp8 files actually quantize: the ten
    /// block projections. `norm_q`/`norm_k`/`norm3` gains and every bias stay
    /// unquantized, as do the five top-level projections.
    fn is_quantized_projection(name: &str) -> bool {
        name.starts_with("blocks.") && name.ends_with(".weight") && !name.contains("norm")
    }

    /// Write a tiny fp8-scaled checkpoint plus the F32 checkpoint that holds
    /// exactly the weights it dequantizes to, and return both paths.
    ///
    /// The reference is the *same-dequantized* baseline, not the pre-quantized
    /// one: an fp8 forward cannot reproduce the original F32 weights, so
    /// comparing against them would only measure e4m3's 3-bit mantissa. What
    /// has to hold is that the loader reproduces `weight * scale_weight`.
    fn write_fp8_pair(
        dir: &Path,
        varmap: &VarMap,
        marker_elements: usize,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let device = Device::Cpu;
        let mut quantized: HashMap<String, Tensor> = HashMap::new();
        let mut reference: HashMap<String, Tensor> = HashMap::new();
        let mut quantized_count = 0usize;

        for (name, var) in varmap.data().lock().unwrap().iter() {
            let tensor = var.as_tensor().clone();
            if !is_quantized_projection(name) {
                quantized.insert(name.clone(), tensor.clone());
                reference.insert(name.clone(), tensor);
                continue;
            }
            let peak = tensor
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let scale = peak / FP8_E4M3_MAX;
            let fp8 = tensor
                .affine(1.0 / f64::from(scale), 0.0)
                .unwrap()
                .to_dtype(DType::F8E4M3)
                .unwrap();
            reference.insert(
                name.clone(),
                fp8.to_dtype(DType::F32)
                    .unwrap()
                    .affine(f64::from(scale), 0.0)
                    .unwrap(),
            );
            quantized.insert(name.clone(), fp8);
            quantized.insert(
                format!("{}.scale_weight", name.trim_end_matches(".weight")),
                Tensor::from_vec(vec![scale], 1, &device).unwrap(),
            );
            quantized_count += 1;
        }
        assert_eq!(
            quantized_count, 20,
            "the fixture must quantize the ten projections of both blocks"
        );

        quantized.insert(
            "scaled_fp8".to_string(),
            Tensor::zeros(marker_elements, DType::F8E4M3, &device).unwrap(),
        );

        let fp8_path = dir.join("scaled_fp8.safetensors");
        let reference_path = dir.join("dequantized.safetensors");
        candle_core::safetensors::save(&quantized, &fp8_path).unwrap();
        candle_core::safetensors::save(&reference, &reference_path).unwrap();
        (fp8_path, reference_path)
    }

    /// Full fp8-scaled round trip: a checkpoint whose block projections are
    /// e4m3 with per-tensor scales must forward identically to one holding the
    /// dequantized weights outright.
    ///
    /// This also exercises the mixed layout the shipped experts have — the
    /// head, both embedding MLPs, and the time projection stay F32 in the same
    /// file, so a loader that assumed "fp8 file ⇒ every weight is fp8" fails
    /// here.
    #[test]
    fn fp8_scaled_forward_matches_the_dequantized_weights() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();

        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();

        // 2 elements: the mode both Kijai A14B experts declare.
        let (fp8_path, reference_path) = write_fp8_pair(dir.path(), &varmap, 2);

        let quantized =
            WanTransformer::from_safetensors(&[fp8_path], config.clone(), &device, dtype).unwrap();
        let reference =
            WanTransformer::from_safetensors(&[reference_path], config.clone(), &device, dtype)
                .unwrap();

        assert_eq!(
            quantized.quantization().map(|marker| marker.to_string()),
            Some("e4m3, full-precision matmul".to_string()),
            "the marker has to survive the load — it is the only record of which \
             matmul path the checkpoint was calibrated for"
        );
        assert!(
            reference.quantization().is_none(),
            "a checkpoint with no marker is not fp8-scaled"
        );

        let x = Tensor::randn(0f32, 1f32, (1, config.in_dim, 1, 4, 4), &device).unwrap();
        let timestep = Tensor::from_vec(vec![500f32], 1, &device).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 4, config.text_dim), &device).unwrap();

        let want = reference.forward(&x, &timestep, &context).unwrap();
        let got = quantized.forward(&x, &timestep, &context).unwrap();
        assert_eq!(want.dims(), got.dims());

        let err = max_abs_diff_f32(&values(&got), &values(&want));
        assert!(
            err < 1e-5,
            "fp8 forward diverged from its own dequantized weights by {err:e} — that is a \
             loader bug, not quantization error"
        );

        // The fixture is only meaningful if fp8 actually perturbed the weights;
        // otherwise the comparison above would pass on a no-op quantizer.
        let untouched = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config,
        )
        .unwrap();
        let original = untouched.forward(&x, &timestep, &context).unwrap();
        assert!(
            max_abs_diff_f32(&values(&got), &values(&original)) > 0.0,
            "e4m3 rounding must move the output; a fixture it cannot perturb proves nothing"
        );
    }

    /// The marker's element count decides the matmul path, and a `[0]` marker —
    /// what `umt5_xxl_fp8_e4m3fn_scaled` ships — must not be read as "absent".
    #[test]
    fn an_empty_marker_still_marks_the_checkpoint_as_fp8_scaled() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();
        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();

        let (fp8_path, _) = write_fp8_pair(dir.path(), &varmap, 0);
        let loaded = WanTransformer::from_safetensors(&[fp8_path], config, &device, dtype).unwrap();
        assert_eq!(
            loaded.quantization().map(|marker| marker.to_string()),
            Some("e4m3, native fp8 matmul".to_string())
        );
    }

    /// An adapter over an fp8 checkpoint must be refused, not dropped. Merging
    /// would mean re-rounding every patched weight to three mantissa bits.
    #[test]
    fn an_fp8_checkpoint_refuses_a_lora_stack() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();
        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();
        let (fp8_path, _) = write_fp8_pair(dir.path(), &varmap, 2);

        let adapter = dir.path().join("adapter.safetensors");
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight".to_string(),
            Tensor::zeros((2, config.dim), dtype, &device).unwrap(),
        );
        tensors.insert(
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight".to_string(),
            Tensor::zeros((config.dim, 2), dtype, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &adapter).unwrap();
        let registry = crate::wan::lora::WanLoraRegistry::load(&[mold_core::LoraWeight {
            path: adapter.to_string_lossy().to_string(),
            scale: 1.0,
        }])
        .unwrap();
        assert!(!registry.is_empty());

        let error = match WanTransformer::from_safetensors_with_loras(
            &[fp8_path],
            config,
            &device,
            dtype,
            &registry,
        ) {
            Ok(_) => panic!("an fp8 checkpoint must not silently drop an adapter"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("fp8-scaled"), "{error}");
        assert!(error.contains("GGUF"), "{error}");
    }

    #[test]
    fn production_configs_match_upstream() {
        let cases: [(
            WanTransformerConfig,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        ); 4] = [
            (WanTransformerConfig::t2v_1_3b(), 1536, 8960, 12, 30, 16, 16),
            (WanTransformerConfig::t2v_14b(), 5120, 13824, 40, 40, 16, 16),
            (WanTransformerConfig::i2v_14b(), 5120, 13824, 40, 40, 36, 16),
            (WanTransformerConfig::ti2v_5b(), 3072, 14336, 24, 30, 48, 48),
        ];
        for (config, dim, ffn, heads, layers, in_dim, out_dim) in cases {
            assert_eq!(config.dim, dim);
            assert_eq!(config.ffn_dim, ffn);
            assert_eq!(config.num_heads, heads);
            assert_eq!(config.num_layers, layers);
            assert_eq!(config.in_dim, in_dim);
            assert_eq!(config.out_dim, out_dim);
            // Every production variant lands on head_dim 128.
            assert_eq!(config.head_dim(), 128, "head_dim for dim {dim}");
            assert_eq!(config.freq_dim, 256);
            assert_eq!(config.text_dim, 4096);
            assert_eq!(config.patch_size, (1, 2, 2));
            assert_eq!(config.eps, 1e-6);
            config.validate().unwrap();
        }
    }

    #[test]
    fn forward_rejects_mismatched_inputs() {
        let model = golden_model();
        let context = synth("input.encoder_hidden_states", &[1, 6, 32]);
        let timestep = Tensor::from_vec(vec![500f32], 1, &Device::Cpu).unwrap();

        // Wrong channel count.
        let bad = Tensor::zeros((1, 8, 2, 4, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(model.forward(&bad, &timestep, &context).is_err());
        // Odd spatial extent against a patch size of 2.
        let bad = Tensor::zeros((1, 16, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(model.forward(&bad, &timestep, &context).is_err());
        // Timestep batch disagreement.
        let x = synth("input.hidden_states", &[1, 16, 2, 4, 4]);
        let bad_t = Tensor::from_vec(vec![500f32, 500.0], 2, &Device::Cpu).unwrap();
        assert!(model.forward(&x, &bad_t, &context).is_err());
        // Rank-3 timestep.
        let bad_t = Tensor::zeros((1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(model.forward(&x, &bad_t, &context).is_err());
    }

    #[test]
    fn config_validation_rejects_degenerate_shapes() {
        let mut config = WanTransformerConfig::t2v_1_3b();
        config.num_heads = 7;
        assert!(config.validate().is_err());

        let mut config = WanTransformerConfig::t2v_1_3b();
        config.freq_dim = 255;
        assert!(config.validate().is_err());

        let mut config = WanTransformerConfig::t2v_1_3b();
        config.patch_size = (1, 0, 2);
        assert!(config.validate().is_err());
    }

    /// Batch and token counts must flow through unchanged, and the output must
    /// carry `out_dim` channels rather than `in_dim`.
    #[test]
    fn shape_contract_holds_for_asymmetric_inputs() {
        let mut config = tiny_config();
        config.out_dim = 8;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = WanTransformer::from_var_builder(vb, config).unwrap();

        let x = Tensor::randn(0f32, 1f32, (2, 16, 3, 6, 4), &Device::Cpu).unwrap();
        let context = Tensor::randn(0f32, 1f32, (2, 5, 32), &Device::Cpu).unwrap();
        let timestep = Tensor::from_vec(vec![900f32, 100.0], 2, &Device::Cpu).unwrap();
        let out = model.forward(&x, &timestep, &context).unwrap();
        assert_eq!(out.dims(), &[2, 8, 3, 6, 4]);
    }
}
