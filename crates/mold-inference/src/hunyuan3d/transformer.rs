//! The Hunyuan3D 2.0 shape DiT: a flow-matching transformer over a 1-D latent
//! token sequence.
//!
//! # Why this is not a new architecture
//!
//! Upstream's `Hunyuan3Dv2` imports FLUX's blocks verbatim —
//! `comfy/ldm/hunyuan3d/model.py:1-9` pulls `DoubleStreamBlock`,
//! `SingleStreamBlock`, `LastLayer`, `MLPEmbedder` and `timestep_embedding`
//! straight out of `comfy.ldm.flux.layers`. The weight names in the shipped
//! checkpoints confirm it: `model.double_blocks.N.img_attn.qkv.weight`,
//! `model.single_blocks.N.linear1.weight`, `model.final_layer.adaLN_modulation.1.*`
//! are FLUX's names.
//!
//! Three things differ, and all three matter:
//!
//! 1. **`pe` is `None`.** There is no positional encoding at all — the latent
//!    is an unordered *set* of tokens (a "vecset"), not a raster, so there is
//!    no grid to embed. Every attention call here is plain SDPA.
//! 2. **The timestep is inverted** (`timestep = 1.0 - timestep`) and the
//!    output is negated (`* -1.0`), which together flip the flow direction
//!    relative to FLUX. `model.py:79` and `:148`.
//! 3. **`max_period` is 1000, not 10000.** Upstream's own comment at
//!    `model.py:37` says they meant to set `time_factor` and set `max_period`
//!    instead. It is baked into the trained weights, so it is reproduced, not
//!    fixed.
//!
//! Candle ships FLUX blocks, but their constructors are private and their
//! `forward` takes a required `pe` tensor, so they cannot express `pe = None`.
//! Per `docs/architecture/candle-extension.md` a change expressible with the
//! public API does not belong in the fork, so the blocks are rebuilt here over
//! `candle_nn` — the same call the quantized Flux.2 transformer made
//! (`crates/mold-inference/src/flux2/quantized_transformer.rs`).
//!
//! # Shapes
//!
//! The latent is `[B, 64, L]` on the way in and out — channels *before*
//! length, which is why `forward` opens and closes with a `movedim`. `L` is
//! 3072 for every 2.0 tier. Conditioning is `[B, T, 1536]`: the DINOv2-giant
//! last hidden state, CLS token included.

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, RmsNorm, VarBuilder};

use crate::attention::{attention_for, AttentionPolicy};

/// Geometry of one shape-DiT checkpoint.
///
/// Every field is readable from the checkpoint itself, exactly as
/// `comfy/model_detection.py:784-797` does it, so [`Config::from_state_dict`]
/// is the authority and these presets are only a convenience for tests and
/// for naming what upstream ships.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub in_channels: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub qkv_bias: bool,
    /// Distilled tiers carry a guidance embedding and run at CFG 1.0; the
    /// undistilled 2.0 base does not and runs a real guided branch.
    pub guidance_embed: bool,
}

impl Config {
    /// `mlp_ratio` is 4.0 in every published tier, and is not stored in the
    /// checkpoint in a form worth detecting.
    pub const MLP_RATIO: f64 = 4.0;
    /// Upstream's `self.max_period`. See the module doc: this is upstream's
    /// bug, preserved because the weights were trained with it.
    pub const MAX_PERIOD: f64 = 1000.0;
    /// `time_factor` in `comfy/ldm/flux/layers.py:29`, left at its default.
    pub const TIME_FACTOR: f64 = 1000.0;
    /// Width of the sinusoidal timestep embedding fed to `time_in`.
    pub const TIME_EMBED_DIM: usize = 256;

    /// `hunyuan3d-dit-v2-0/config.yaml` — the undistilled 1.1B tier.
    pub fn v2_0() -> Self {
        Self {
            in_channels: 64,
            context_in_dim: 1536,
            hidden_size: 1024,
            num_heads: 16,
            depth: 16,
            depth_single_blocks: 32,
            qkv_bias: true,
            guidance_embed: false,
        }
    }

    /// `hunyuan3d-dit-v2-0-turbo/config.yaml` — same geometry, distilled.
    pub fn v2_0_turbo() -> Self {
        Self {
            guidance_embed: true,
            ..Self::v2_0()
        }
    }

    /// `hunyuan3d-dit-v2-mini-turbo/config.yaml` — the 0.6B tier.
    pub fn v2_0_mini_turbo() -> Self {
        Self {
            depth: 8,
            depth_single_blocks: 16,
            guidance_embed: true,
            ..Self::v2_0_turbo()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn mlp_size(&self) -> usize {
        (self.hidden_size as f64 * Self::MLP_RATIO) as usize
    }

    /// Recover the geometry from the tensors themselves.
    ///
    /// Mirrors `comfy/model_detection.py:784-797`: the two `Linear` shapes
    /// give every width, and the block counts come from how many indices the
    /// prefixes carry. Detecting rather than trusting a filename is what lets
    /// a community re-quantization or a repack load without a manifest entry
    /// describing its internals.
    ///
    /// `keys` is any iterator over the *un-prefixed* tensor names (i.e. with
    /// the checkpoint's `model.` prefix already stripped).
    pub fn from_state_dict<'a>(
        latent_in_shape: (usize, usize),
        cond_in_in_dim: usize,
        keys: impl Iterator<Item = &'a str>,
    ) -> Self {
        let (hidden_size, in_channels) = latent_in_shape;
        let mut depth = 0usize;
        let mut depth_single_blocks = 0usize;
        let mut guidance_embed = false;
        for key in keys {
            if let Some(index) = block_index(key, "double_blocks.") {
                depth = depth.max(index + 1);
            } else if let Some(index) = block_index(key, "single_blocks.") {
                depth_single_blocks = depth_single_blocks.max(index + 1);
            } else if key.starts_with("guidance_in.") {
                guidance_embed = true;
            }
        }
        Self {
            in_channels,
            context_in_dim: cond_in_in_dim,
            hidden_size,
            // Fixed at 16 in every tier; `model_detection.py:792` hard-codes
            // it too, because the checkpoint does not record it.
            num_heads: 16,
            depth,
            depth_single_blocks,
            qkv_bias: true,
            guidance_embed,
        }
    }
}

/// `double_blocks.7.img_attn.qkv.weight` -> `Some(7)` for prefix
/// `"double_blocks."`.
fn block_index(key: &str, prefix: &str) -> Option<usize> {
    key.strip_prefix(prefix)?
        .split('.')
        .next()?
        .parse::<usize>()
        .ok()
}

/// Sinusoidal timestep embedding.
///
/// `comfy/ldm/flux/layers.py:29-48`, called from `model.py:82` with
/// `dim = 256` and `max_period = self.max_period`. Note the concatenation
/// order is `[cos, sin]`, not the more common `[sin, cos]`.
pub fn timestep_embedding(t: &Tensor, dim: usize, max_period: f64, dtype: DType) -> Result<Tensor> {
    if dim % 2 != 0 {
        candle_core::bail!("timestep embedding dim must be even, got {dim}");
    }
    let half = dim / 2;
    let device = t.device();
    let t = (t * Config::TIME_FACTOR)?;
    let arange = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-max_period.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
}

/// A `LayerNorm` with neither weight nor bias, matching PyTorch's
/// `elementwise_affine=False`. FLUX's `img_norm1`/`img_norm2`/`pre_norm`/
/// `norm_final` are all of this shape and store no tensors, which is why the
/// checkpoint dump has no keys for them.
fn affineless_layer_norm(size: usize) -> LayerNorm {
    LayerNorm::new_no_bias(
        Tensor::ones(size, DType::F32, &candle_core::Device::Cpu).unwrap(),
        1e-6,
    )
}

/// Per-head query/key RMS norms. The stored parameter is called `scale`, not
/// `weight`, so it is fetched by hand.
#[derive(Debug, Clone)]
struct QkNorm {
    query: RmsNorm,
    key: RmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let query = RmsNorm::new(vb.get(head_dim, "query_norm.scale")?, 1e-6);
        let key = RmsNorm::new(vb.get(head_dim, "key_norm.scale")?, 1e-6);
        Ok(Self { query, key })
    }
}

/// One `(shift, scale, gate)` triple produced by an adaLN modulation head.
struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.0)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

/// `Linear(hidden, n * hidden)` behind a SiLU, chunked into `n / 3`
/// modulation triples.
#[derive(Debug, Clone)]
struct Modulation {
    lin: Linear,
    chunks: usize,
}

impl Modulation {
    fn new(dim: usize, triples: usize, vb: VarBuilder) -> Result<Self> {
        let chunks = triples * 3;
        Ok(Self {
            lin: candle_nn::linear(dim, chunks * dim, vb.pp("lin"))?,
            chunks,
        })
    }

    fn forward(&self, vec: &Tensor) -> Result<Vec<ModulationOut>> {
        let parts = vec
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(self.chunks, D::Minus1)?;
        if parts.len() != self.chunks {
            candle_core::bail!("modulation produced {} chunks", parts.len());
        }
        Ok(parts
            .chunks_exact(3)
            .map(|triple| ModulationOut {
                shift: triple[0].clone(),
                scale: triple[1].clone(),
                gate: triple[2].clone(),
            })
            .collect())
    }
}

/// Plain scaled-dot-product attention over `[B, H, L, Dh]`, flattened back to
/// `[B, L, H*Dh]`.
///
/// `AttentionPolicy::Image` is deliberate: the shape DiT's sequence is 3072
/// unordered tokens with a head dim of 64, which is the image families'
/// regime, not a video DiT's. It must also match
/// `crate::attention::policy_for_family("hunyuan3d")`, because
/// `FrozenEngineConfig` records that answer and execution-plan equivalence is
/// built from it — freezing one policy and running another would describe
/// different arithmetic from the one that executes.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale = (1.0 / (head_dim as f64).sqrt()) as f32;
    let out = attention_for(AttentionPolicy::Image, q, k, v, scale)
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
    let (b, h, l, dh) = out.dims4()?;
    out.transpose(1, 2)?.reshape((b, l, h * dh))
}

#[derive(Debug, Clone)]
struct SelfAttention {
    qkv: Linear,
    norm: QkNorm,
    proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: candle_nn::linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?,
            norm: QkNorm::new(dim / num_heads, vb.pp("norm"))?,
            proj: candle_nn::linear(dim, dim, vb.pp("proj"))?,
            num_heads,
        })
    }

    /// Projects to `[B, H, L, Dh]` triples with the q/k norms already applied.
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
        Ok((q.apply(&self.norm.query)?, k.apply(&self.norm.key)?, v))
    }
}

/// `Linear -> GELU(tanh) -> Linear`, stored under the indices `0` and `2`
/// because upstream builds it as an `nn.Sequential`.
#[derive(Debug, Clone)]
struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn new(dim: usize, mlp_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: candle_nn::linear(dim, mlp_size, vb.pp("0"))?,
            lin2: candle_nn::linear(mlp_size, dim, vb.pp("2"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

/// A two-stream block: latent tokens and conditioning tokens keep separate
/// weights but attend jointly over the concatenated sequence.
#[derive(Debug, Clone)]
struct DoubleStreamBlock {
    img_mod: Modulation,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp = cfg.mlp_size();
        Ok(Self {
            img_mod: Modulation::new(h, 2, vb.pp("img_mod"))?,
            img_norm1: affineless_layer_norm(h),
            img_attn: SelfAttention::new(h, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?,
            img_norm2: affineless_layer_norm(h),
            img_mlp: Mlp::new(h, mlp, vb.pp("img_mlp"))?,
            txt_mod: Modulation::new(h, 2, vb.pp("txt_mod"))?,
            txt_norm1: affineless_layer_norm(h),
            txt_attn: SelfAttention::new(h, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"))?,
            txt_norm2: affineless_layer_norm(h),
            txt_mlp: Mlp::new(h, mlp, vb.pp("txt_mlp"))?,
        })
    }

    fn forward(&self, img: &Tensor, txt: &Tensor, vec: &Tensor) -> Result<(Tensor, Tensor)> {
        let img_mod = self.img_mod.forward(vec)?;
        let txt_mod = self.txt_mod.forward(vec)?;

        let img_modulated = img_mod[0].scale_shift(&img.apply(&self.img_norm1)?)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;
        let txt_modulated = txt_mod[0].scale_shift(&txt.apply(&self.txt_norm1)?)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        // Conditioning first, then the latent — the order the split below and
        // the `cat((txt, img))` in `forward` both depend on.
        let q = Tensor::cat(&[txt_q, img_q], 2)?.contiguous()?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?.contiguous()?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?.contiguous()?;
        let attn = attention(&q, &k, &v)?;

        let txt_len = txt.dim(1)?;
        let txt_attn = attn.narrow(1, 0, txt_len)?;
        let img_attn = attn.narrow(1, txt_len, attn.dim(1)? - txt_len)?;

        let img = (img + img_mod[0].gate(&img_attn.apply(&self.img_attn.proj)?)?)?;
        let img_mlp = img_mod[1]
            .scale_shift(&img.apply(&self.img_norm2)?)?
            .apply(&self.img_mlp)?;
        let img = (&img + img_mod[1].gate(&img_mlp)?)?;

        let txt = (txt + txt_mod[0].gate(&txt_attn.apply(&self.txt_attn.proj)?)?)?;
        let txt_mlp = txt_mod[1]
            .scale_shift(&txt.apply(&self.txt_norm2)?)?
            .apply(&self.txt_mlp)?;
        let txt = (&txt + txt_mod[1].gate(&txt_mlp)?)?;

        Ok((img, txt))
    }
}

/// A fused block over the already-concatenated sequence: attention and MLP
/// share one input projection and one output projection.
#[derive(Debug, Clone)]
struct SingleStreamBlock {
    linear1: Linear,
    linear2: Linear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation,
    hidden_size: usize,
    mlp_size: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp = cfg.mlp_size();
        Ok(Self {
            linear1: candle_nn::linear(h, h * 3 + mlp, vb.pp("linear1"))?,
            linear2: candle_nn::linear(h + mlp, h, vb.pp("linear2"))?,
            norm: QkNorm::new(cfg.head_dim(), vb.pp("norm"))?,
            pre_norm: affineless_layer_norm(h),
            modulation: Modulation::new(h, 1, vb.pp("modulation"))?,
            hidden_size: h,
            mlp_size: mlp,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let modulation = self.modulation.forward(vec)?;
        let m = &modulation[0];
        let projected = m
            .scale_shift(&xs.apply(&self.pre_norm)?)?
            .apply(&self.linear1)?;

        let qkv = projected.narrow(D::Minus1, 0, 3 * self.hidden_size)?;
        let (b, l, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv
            .i((.., .., 0))?
            .transpose(1, 2)?
            .contiguous()?
            .apply(&self.norm.query)?;
        let k = qkv
            .i((.., .., 1))?
            .transpose(1, 2)?
            .contiguous()?
            .apply(&self.norm.key)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;

        let mlp = projected.narrow(D::Minus1, 3 * self.hidden_size, self.mlp_size)?;
        let attn = attention(&q, &k, &v)?;
        let output = Tensor::cat(&[attn, mlp.gelu()?], 2)?.apply(&self.linear2)?;
        xs + m.gate(&output)?
    }
}

/// adaLN-Zero output head. `p_sz` is 1 here — there is no patchification,
/// each token maps straight back to one latent vector.
#[derive(Debug, Clone)]
struct LastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl LastLayer {
    fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: affineless_layer_norm(hidden_size),
            linear: candle_nn::linear(hidden_size, out_channels, vb.pp("linear"))?,
            ada_ln_modulation: candle_nn::linear(
                hidden_size,
                2 * hidden_size,
                vb.pp("adaLN_modulation.1"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        xs.apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?
            .apply(&self.linear)
    }
}

/// `Linear -> SiLU -> Linear`, upstream's `MLPEmbedder`.
#[derive(Debug, Clone)]
struct MlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            in_layer: candle_nn::linear(in_dim, hidden, vb.pp("in_layer"))?,
            out_layer: candle_nn::linear(hidden, hidden, vb.pp("out_layer"))?,
        })
    }
}

impl Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}

/// The shape DiT.
#[derive(Debug, Clone)]
pub struct Hunyuan3dDit {
    latent_in: Linear,
    cond_in: Linear,
    time_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
    cfg: Config,
}

impl Hunyuan3dDit {
    /// `vb` must already be scoped to the checkpoint's `model.` prefix.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let latent_in = candle_nn::linear(cfg.in_channels, h, vb.pp("latent_in"))?;
        let cond_in = candle_nn::linear(cfg.context_in_dim, h, vb.pp("cond_in"))?;
        let time_in = MlpEmbedder::new(Config::TIME_EMBED_DIM, h, vb.pp("time_in"))?;
        let guidance_in = if cfg.guidance_embed {
            Some(MlpEmbedder::new(
                Config::TIME_EMBED_DIM,
                h,
                vb.pp("guidance_in"),
            )?)
        } else {
            None
        };
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        for index in 0..cfg.depth {
            double_blocks.push(DoubleStreamBlock::new(
                cfg,
                vb.pp(format!("double_blocks.{index}")),
            )?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        for index in 0..cfg.depth_single_blocks {
            single_blocks.push(SingleStreamBlock::new(
                cfg,
                vb.pp(format!("single_blocks.{index}")),
            )?);
        }
        let final_layer = LastLayer::new(h, cfg.in_channels, vb.pp("final_layer"))?;
        Ok(Self {
            latent_in,
            cond_in,
            time_in,
            guidance_in,
            double_blocks,
            single_blocks,
            final_layer,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// One denoising step.
    ///
    /// - `latents`: `[B, in_channels, L]` — channels before length.
    /// - `timestep`: `[B]`, in `[0, 1]`.
    /// - `context`: `[B, T, context_in_dim]` DINOv2 hidden states.
    /// - `guidance`: `[B]`, only consulted when the checkpoint carries a
    ///   guidance embedding. Passing `Some` to a checkpoint without one is
    ///   silently ignored, matching `model.py:84-86`.
    ///
    /// Returns `[B, in_channels, L]`.
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        // `[B, C, L]` -> `[B, L, C]`. `model.py:78` writes it as
        // `x.movedim(-1, -2)`; on a rank-3 tensor that is exactly a swap of
        // the trailing two axes, which is what candle spells `transpose`.
        let xs = latents.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        // `model.py:79`. The trained flow runs backwards relative to FLUX.
        let timestep = (timestep.affine(-1.0, 1.0))?;

        let dtype = xs.dtype();
        let mut img = xs.apply(&self.latent_in)?;
        let mut vec = self.time_in.forward(&timestep_embedding(
            &timestep,
            Config::TIME_EMBED_DIM,
            Config::MAX_PERIOD,
            dtype,
        )?)?;
        if let (Some(guidance_in), Some(guidance)) = (&self.guidance_in, guidance) {
            let embedded =
                timestep_embedding(guidance, Config::TIME_EMBED_DIM, Config::MAX_PERIOD, dtype)?;
            vec = (vec + guidance_in.forward(&embedded)?)?;
        }
        let mut txt = context.apply(&self.cond_in)?;

        for block in &self.double_blocks {
            let (next_img, next_txt) = block.forward(&img, &txt, &vec)?;
            img = next_img;
            txt = next_txt;
        }

        let txt_len = txt.dim(1)?;
        let mut fused = Tensor::cat(&[&txt, &img], 1)?.contiguous()?;
        for block in &self.single_blocks {
            fused = block.forward(&fused, &vec)?;
        }
        let img = fused.narrow(1, txt_len, fused.dim(1)? - txt_len)?;

        // `model.py:148`: back to `[B, C, L]`, and negated.
        let out = self.final_layer.forward(&img, &vec)?;
        out.transpose(D::Minus2, D::Minus1)?.contiguous()? * -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn presets_match_upstream_config_yaml() {
        let base = Config::v2_0();
        assert_eq!(base.depth, 16);
        assert_eq!(base.depth_single_blocks, 32);
        assert_eq!(base.hidden_size, 1024);
        assert_eq!(base.context_in_dim, 1536);
        assert_eq!(base.in_channels, 64);
        assert!(!base.guidance_embed, "2.0 base runs a real guided branch");

        let mini = Config::v2_0_mini_turbo();
        assert_eq!(mini.depth, 8);
        assert_eq!(mini.depth_single_blocks, 16);
        assert!(mini.guidance_embed, "distilled tiers carry the embedding");
        assert_eq!(mini.hidden_size, base.hidden_size);
    }

    #[test]
    fn derived_widths_match_the_checkpoint_shapes() {
        let cfg = Config::v2_0();
        // `model.single_blocks.N.linear1.weight` is [7168, 1024].
        assert_eq!(cfg.hidden_size * 3 + cfg.mlp_size(), 7168);
        // `model.single_blocks.N.linear2.weight` is [1024, 5120].
        assert_eq!(cfg.hidden_size + cfg.mlp_size(), 5120);
        // `model.double_blocks.N.img_mod.lin.weight` is [6144, 1024].
        assert_eq!(cfg.hidden_size * 6, 6144);
        // `model.double_blocks.N.img_attn.norm.query_norm.scale` is [64].
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn state_dict_detection_recovers_the_geometry() {
        let mut keys: Vec<String> = Vec::new();
        for index in 0..16 {
            keys.push(format!("double_blocks.{index}.img_attn.qkv.weight"));
        }
        for index in 0..32 {
            keys.push(format!("single_blocks.{index}.linear1.weight"));
        }
        keys.push("latent_in.weight".to_string());
        let detected = Config::from_state_dict((1024, 64), 1536, keys.iter().map(String::as_str));
        assert_eq!(detected, Config::v2_0());

        keys.push("guidance_in.in_layer.weight".to_string());
        let distilled = Config::from_state_dict((1024, 64), 1536, keys.iter().map(String::as_str));
        assert!(distilled.guidance_embed);
        assert_eq!(distilled, Config::v2_0_turbo());
    }

    #[test]
    fn timestep_embedding_uses_upstreams_max_period_and_cos_first() {
        let device = Device::Cpu;
        let t = Tensor::new(&[0.0f32], &device).unwrap();
        let emb = timestep_embedding(&t, 8, Config::MAX_PERIOD, DType::F32).unwrap();
        assert_eq!(emb.dims(), &[1, 8]);
        let values = emb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // t = 0 => args = 0 => cos = 1 for the first half, sin = 0 for the
        // second. This is the check that pins the [cos, sin] order.
        for value in &values[..4] {
            assert!(
                (value - 1.0).abs() < 1e-6,
                "cos half should be 1, got {value}"
            );
        }
        for value in &values[4..] {
            assert!(value.abs() < 1e-6, "sin half should be 0, got {value}");
        }
    }

    #[test]
    fn timestep_embedding_rejects_an_odd_width() {
        let device = Device::Cpu;
        let t = Tensor::new(&[0.5f32], &device).unwrap();
        assert!(timestep_embedding(&t, 7, Config::MAX_PERIOD, DType::F32).is_err());
    }

    fn tiny_config(guidance_embed: bool) -> Config {
        Config {
            in_channels: 8,
            context_in_dim: 12,
            hidden_size: 16,
            num_heads: 2,
            depth: 1,
            depth_single_blocks: 1,
            qkv_bias: true,
            guidance_embed,
        }
    }

    fn build(cfg: &Config) -> (Hunyuan3dDit, Device) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Hunyuan3dDit::new(cfg, vb).expect("tiny model builds");
        (model, device)
    }

    #[test]
    fn forward_preserves_the_channels_before_length_layout() {
        let cfg = tiny_config(false);
        let (model, device) = build(&cfg);
        let latents = Tensor::randn(0f32, 1.0, (2, cfg.in_channels, 5), &device).unwrap();
        let timestep = Tensor::new(&[0.3f32, 0.7], &device).unwrap();
        let context = Tensor::randn(0f32, 1.0, (2, 3, cfg.context_in_dim), &device).unwrap();

        let out = model.forward(&latents, &timestep, &context, None).unwrap();
        assert_eq!(out.dims(), &[2, cfg.in_channels, 5]);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "output must be finite"
        );
    }

    #[test]
    fn guidance_embedding_changes_the_output_only_when_the_checkpoint_has_one() {
        let device = Device::Cpu;
        let latents = Tensor::randn(0f32, 1.0, (1, 8, 4), &device).unwrap();
        let timestep = Tensor::new(&[0.5f32], &device).unwrap();
        let context = Tensor::randn(0f32, 1.0, (1, 2, 12), &device).unwrap();
        let guidance = Tensor::new(&[5.0f32], &device).unwrap();

        // Without the embedding, `guidance` is ignored rather than an error.
        let (plain, _) = build(&tiny_config(false));
        let a = plain.forward(&latents, &timestep, &context, None).unwrap();
        let b = plain
            .forward(&latents, &timestep, &context, Some(&guidance))
            .unwrap();
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            a, b,
            "a checkpoint without guidance_in must ignore guidance"
        );

        // With it, the value has to reach the output.
        let (distilled, _) = build(&tiny_config(true));
        let c = distilled
            .forward(&latents, &timestep, &context, None)
            .unwrap();
        let d = distilled
            .forward(&latents, &timestep, &context, Some(&guidance))
            .unwrap();
        let c = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let d = d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            c.iter().zip(&d).any(|(x, y)| (x - y).abs() > 1e-6),
            "guidance must change the output when guidance_in exists"
        );
    }
}
