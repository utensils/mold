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

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};

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
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
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
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

pub(crate) fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
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
    in_layer: Linear,
    out_layer: Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        // Diffusers names: linear_1 / linear_2
        let in_layer = candle_nn::linear_no_bias(in_sz, h_sz, vb.pp("linear_1"))?;
        let out_layer = candle_nn::linear_no_bias(h_sz, h_sz, vb.pp("linear_2"))?;
        Ok(Self {
            in_layer,
            out_layer,
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
    lin: Linear,
}

impl Modulation1 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = candle_nn::linear_no_bias(dim, 3 * dim, vb.pp("linear"))?;
        Ok(Self { lin })
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
    lin: Linear,
}

impl Modulation2 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = candle_nn::linear_no_bias(dim, 6 * dim, vb.pp("linear"))?;
        Ok(Self { lin })
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
    lin1: Linear,
    lin2: Linear,
    mlp_sz: usize,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = candle_nn::linear_no_bias(in_sz, mlp_sz * 2, vb.pp("linear_in"))?;
        let lin2 = candle_nn::linear_no_bias(mlp_sz, in_sz, vb.pp("linear_out"))?;
        Ok(Self { lin1, lin2, mlp_sz })
    }
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
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    num_heads: usize,
}

impl DoubleAttention {
    /// Load image-side attention from `attn.to_q/k/v`, `attn.to_out.0`, `attn.norm_q/k`.
    fn new_img(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            to_q: candle_nn::linear_no_bias(dim, dim, vb.pp("to_q"))?,
            to_k: candle_nn::linear_no_bias(dim, dim, vb.pp("to_k"))?,
            to_v: candle_nn::linear_no_bias(dim, dim, vb.pp("to_v"))?,
            to_out: candle_nn::linear_no_bias(dim, dim, vb.pp("to_out").pp("0"))?,
            norm_q: RmsNorm::new(vb.get(head_dim, "norm_q.weight")?, 1e-6),
            norm_k: RmsNorm::new(vb.get(head_dim, "norm_k.weight")?, 1e-6),
            num_heads,
        })
    }

    /// Load text-side attention from `attn.add_q_proj`, `attn.to_add_out`, `attn.norm_added_q/k`.
    fn new_txt(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            to_q: candle_nn::linear_no_bias(dim, dim, vb.pp("add_q_proj"))?,
            to_k: candle_nn::linear_no_bias(dim, dim, vb.pp("add_k_proj"))?,
            to_v: candle_nn::linear_no_bias(dim, dim, vb.pp("add_v_proj"))?,
            to_out: candle_nn::linear_no_bias(dim, dim, vb.pp("to_add_out"))?,
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

// ---------------------------------------------------------------------------
// SingleStreamBlock (diffusers naming)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SingleStreamBlock {
    linear1: Linear,
    linear2: Linear,
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
            candle_nn::linear_no_bias(h_sz, h_sz * 3 + mlp_sz * 2, attn_vb.pp("to_qkv_mlp_proj"))?;
        // Output: attn (h_sz) + mlp (mlp_sz) → to_out
        let linear2 = candle_nn::linear_no_bias(h_sz + mlp_sz, h_sz, attn_vb.pp("to_out"))?;
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

// ---------------------------------------------------------------------------
// LastLayer — final projection (diffusers: proj_out + norm_out)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl LastLayer {
    fn new(h_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm(h_sz, &vb)?,
            linear: candle_nn::linear_no_bias(h_sz, out_c, vb.pp("proj_out"))?,
            ada_ln_modulation: candle_nn::linear_no_bias(
                h_sz,
                2 * h_sz,
                vb.pp("norm_out").pp("linear"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        // AdaLayerNormContinuous: scale first, shift second (differs from modulation order)
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
    img_in: Linear,
    txt_in: Linear,
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

impl Flux2Transformer {
    pub fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let img_in =
            candle_nn::linear_no_bias(cfg.in_channels, cfg.hidden_size, vb.pp("x_embedder"))?;
        let txt_in = candle_nn::linear_no_bias(
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
    Quantized(super::quantized_transformer::QuantizedFlux2Transformer),
}

impl Flux2TransformerWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        vec_: &Tensor,
        timesteps: &[f64],
        guidance: f64,
        progress: &crate::progress::ProgressReporter,
        inpaint_ctx: Option<&crate::img_utils::InpaintContext>,
    ) -> anyhow::Result<Tensor> {
        use crate::progress::ProgressEvent;
        use std::time::Instant;

        let b_sz = img.dim(0)?;
        let dev = img.device();
        let guidance_tensor = Tensor::full(guidance as f32, b_sz, dev)?;
        let mut img = img.clone();
        let total_steps = timesteps.len().saturating_sub(1);

        for (step, window) in timesteps.windows(2).enumerate() {
            let step_start = Instant::now();
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;

            let pred = match self {
                Self::BF16(m) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
                Self::Quantized(m) => m.forward(
                    &img,
                    img_ids,
                    txt,
                    txt_ids,
                    &t_vec,
                    vec_,
                    Some(&guidance_tensor),
                )?,
            };
            img = (img + pred * (t_prev - t_curr))?;

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ctx) = inpaint_ctx {
                let t = *t_prev;
                // Re-noise original latents to current timestep (flow-matching schedule)
                let noised_original = ((&ctx.original_latents * (1.0 - t))? + (&ctx.noise * t)?)?;
                // mask=1 -> repaint (use denoised), mask=0 -> preserve (use noised original)
                img = ((&ctx.mask * &img)? + (&(1.0 - &ctx.mask)? * &noised_original)?)?;
            }

            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: total_steps,
                elapsed: step_start.elapsed(),
            });
        }
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
