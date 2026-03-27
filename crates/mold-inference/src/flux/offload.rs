//! Block-level GPU offloading for FLUX transformers.
//!
//! Streams transformer blocks one at a time between CPU and GPU during each
//! denoising step. Reduces peak VRAM from ~24GB (full BF16 dev model) to ~2-4GB
//! at the cost of 3-5x slower inference.
//!
//! Self-contained: defines its own block types and forward logic so no patches
//! to candle-transformers are needed.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};

use crate::progress::ProgressReporter;

// Re-export Config and EmbedNd — these are public types with public constructors.
use candle_transformers::models::flux::model::{Config, EmbedNd};

// ── Reimplemented candle-internal helpers ────────────────────────────────────

fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        anyhow::bail!("{dim} is odd");
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(candle_core::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(candle_core::DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
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
    Ok(attn_scores.reshape(batch_dims)?)
}

#[allow(dead_code)]
fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        anyhow::bail!("dim {dim} is odd");
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
    Ok(out.reshape((b, n, d, 2, 2))?)
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    Ok((fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())?)
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    Ok(x.transpose(1, 2)?.flatten_from(2)?)
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

// ── Device-transfer helpers ──────────────────────────────────────────────────

fn linear_to_device(l: &Linear, dev: &Device) -> Result<Linear> {
    let w = l.weight().to_device(dev)?;
    let b = l.bias().map(|b| b.to_device(dev)).transpose()?;
    Ok(Linear::new(w, b))
}

fn layer_norm_to_device(ln: &LayerNorm, dev: &Device) -> Result<LayerNorm> {
    let w = ln.weight().to_device(dev)?;
    match ln.bias() {
        Some(b) => Ok(LayerNorm::new(w, b.to_device(dev)?, 1e-6)),
        None => Ok(LayerNorm::new_no_bias(w, 1e-6)),
    }
}

fn rms_norm_to_device(rn: &RmsNorm, dev: &Device) -> Result<RmsNorm> {
    let inner = rn.clone().into_inner();
    Ok(RmsNorm::new(inner.weight().to_device(dev)?, 1e-6))
}

// ── Self-contained block types ───────────────────────────────────────────────

struct Modulation1 {
    lin: Linear,
}

impl Modulation1 {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin: candle_nn::linear(dim, 3 * dim, vb.pp("lin"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            lin: linear_to_device(&self.lin, dev)?,
        })
    }
    fn forward(&self, vec_: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        Ok((ys[0].clone(), ys[1].clone(), ys[2].clone()))
    }
}

struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin: candle_nn::linear(dim, 6 * dim, vb.pp("lin"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            lin: linear_to_device(&self.lin, dev)?,
        })
    }
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        vec_: &Tensor,
    ) -> Result<((Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        Ok((
            (ys[0].clone(), ys[1].clone(), ys[2].clone()),
            (ys[3].clone(), ys[4].clone(), ys[5].clone()),
        ))
    }
}

struct SelfAttention {
    qkv: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn load(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = candle_nn::linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let query_norm = vb.get(head_dim, "norm.query_norm.scale")?;
        let key_norm = vb.get(head_dim, "norm.key_norm.scale")?;
        let proj = candle_nn::linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            query_norm: RmsNorm::new(query_norm, 1e-6),
            key_norm: RmsNorm::new(key_norm, 1e-6),
            proj,
            num_heads,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            qkv: linear_to_device(&self.qkv, dev)?,
            query_norm: rms_norm_to_device(&self.query_norm, dev)?,
            key_norm: rms_norm_to_device(&self.key_norm, dev)?,
            proj: linear_to_device(&self.proj, dev)?,
            num_heads: self.num_heads,
        })
    }
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = xs.apply(&self.qkv)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.query_norm)?;
        let k = k.apply(&self.key_norm)?;
        Ok((q, k, v))
    }
}

struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn load(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: candle_nn::linear(in_sz, mlp_sz, vb.pp("0"))?,
            lin2: candle_nn::linear(mlp_sz, in_sz, vb.pp("2"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            lin1: linear_to_device(&self.lin1, dev)?,
            lin2: linear_to_device(&self.lin2, dev)?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)?)
    }
}

/// FLUX double-stream block — processes image and text streams in parallel
/// with cross-attention.
pub(crate) struct DoubleBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleBlock {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
        Ok(Self {
            img_mod: Modulation2::load(h, vb.pp("img_mod"))?,
            img_norm1: layer_norm(h, vb.pp("img_norm1"))?,
            img_attn: SelfAttention::load(h, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?,
            img_norm2: layer_norm(h, vb.pp("img_norm2"))?,
            img_mlp: Mlp::load(h, mlp_sz, vb.pp("img_mlp"))?,
            txt_mod: Modulation2::load(h, vb.pp("txt_mod"))?,
            txt_norm1: layer_norm(h, vb.pp("txt_norm1"))?,
            txt_attn: SelfAttention::load(h, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"))?,
            txt_norm2: layer_norm(h, vb.pp("txt_norm2"))?,
            txt_mlp: Mlp::load(h, mlp_sz, vb.pp("txt_mlp"))?,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            img_mod: self.img_mod.to_device(dev)?,
            img_norm1: layer_norm_to_device(&self.img_norm1, dev)?,
            img_attn: self.img_attn.to_device(dev)?,
            img_norm2: layer_norm_to_device(&self.img_norm2, dev)?,
            img_mlp: self.img_mlp.to_device(dev)?,
            txt_mod: self.txt_mod.to_device(dev)?,
            txt_norm1: layer_norm_to_device(&self.txt_norm1, dev)?,
            txt_attn: self.txt_attn.to_device(dev)?,
            txt_norm2: layer_norm_to_device(&self.txt_norm2, dev)?,
            txt_mlp: self.txt_mlp.to_device(dev)?,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let ((img_s1, img_sc1, img_g1), (img_s2, img_sc2, img_g2)) = self.img_mod.forward(vec_)?;
        let ((txt_s1, txt_sc1, txt_g1), (txt_s2, txt_sc2, txt_g2)) = self.txt_mod.forward(vec_)?;

        // QKV for both streams
        let img_modulated = img
            .apply(&self.img_norm1)?
            .broadcast_mul(&(&img_sc1 + 1.)?)?
            .broadcast_add(&img_s1)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt
            .apply(&self.txt_norm1)?
            .broadcast_mul(&(&txt_sc1 + 1.)?)?
            .broadcast_add(&txt_s1)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        // Cross-attention
        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;
        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn_out = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn_out = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        // Image residual
        let img = (img + img_g1.broadcast_mul(&img_attn_out.apply(&self.img_attn.proj)?)?)?;
        let img_ff = img
            .apply(&self.img_norm2)?
            .broadcast_mul(&(&img_sc2 + 1.)?)?
            .broadcast_add(&img_s2)?;
        let img = (&img + img_g2.broadcast_mul(&self.img_mlp.forward(&img_ff)?)?)?;

        // Text residual
        let txt = (txt + txt_g1.broadcast_mul(&txt_attn_out.apply(&self.txt_attn.proj)?)?)?;
        let txt_ff = txt
            .apply(&self.txt_norm2)?
            .broadcast_mul(&(&txt_sc2 + 1.)?)?
            .broadcast_add(&txt_s2)?;
        let txt = (&txt + txt_g2.broadcast_mul(&self.txt_mlp.forward(&txt_ff)?)?)?;

        Ok((img, txt))
    }
}

/// FLUX single-stream block — processes combined image+text stream.
pub(crate) struct SingleBlock {
    linear1: Linear,
    linear2: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleBlock {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h / cfg.num_heads;
        Ok(Self {
            linear1: candle_nn::linear(h, h * 3 + mlp_sz, vb.pp("linear1"))?,
            linear2: candle_nn::linear(h + mlp_sz, h, vb.pp("linear2"))?,
            query_norm: {
                let w = vb.get(head_dim, "norm.query_norm.scale")?;
                RmsNorm::new(w, 1e-6)
            },
            key_norm: {
                let w = vb.get(head_dim, "norm.key_norm.scale")?;
                RmsNorm::new(w, 1e-6)
            },
            pre_norm: layer_norm(h, vb.pp("pre_norm"))?,
            modulation: Modulation1::load(h, vb.pp("modulation"))?,
            h_sz: h,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            linear1: linear_to_device(&self.linear1, dev)?,
            linear2: linear_to_device(&self.linear2, dev)?,
            query_norm: rms_norm_to_device(&self.query_norm, dev)?,
            key_norm: rms_norm_to_device(&self.key_norm, dev)?,
            pre_norm: layer_norm_to_device(&self.pre_norm, dev)?,
            modulation: self.modulation.to_device(dev)?,
            h_sz: self.h_sz,
            mlp_sz: self.mlp_sz,
            num_heads: self.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let (shift, scale, gate) = self.modulation.forward(vec_)?;
        let x_mod = xs
            .apply(&self.pre_norm)?
            .broadcast_mul(&(&scale + 1.)?)?
            .broadcast_add(&shift)?;
        let x_mod = x_mod.apply(&self.linear1)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.query_norm)?;
        let k = k.apply(&self.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output = Tensor::cat(&[attn, mlp.gelu()?], 2)?.apply(&self.linear2)?;
        Ok((xs + gate.broadcast_mul(&output)?)?)
    }
}

/// Last layer: AdaLN modulation → linear projection.
struct FinalLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl FinalLayer {
    fn load(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm(h_sz, vb.pp("norm_final"))?,
            linear: candle_nn::linear(h_sz, p_sz * p_sz * out_c, vb.pp("linear"))?,
            ada_ln_modulation: candle_nn::linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm_final: layer_norm_to_device(&self.norm_final, dev)?,
            linear: linear_to_device(&self.linear, dev)?,
            ada_ln_modulation: linear_to_device(&self.ada_ln_modulation, dev)?,
        })
    }
    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        Ok(xs.apply(&self.linear)?)
    }
}

// ── Main offloaded transformer ───────────────────────────────────────────────

/// BF16 FLUX transformer with blocks on CPU, streamed to GPU one at a time.
pub(crate) struct OffloadedFluxTransformer {
    // Stem layers on GPU permanently (~50MB)
    img_in: Linear,
    txt_in: Linear,
    time_in: StemMlpEmbedder,
    vector_in: StemMlpEmbedder,
    guidance_in: Option<StemMlpEmbedder>,
    pe_embedder: EmbedNd,
    final_layer: FinalLayer,
    // Blocks on CPU
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    gpu_device: Device,
}

impl OffloadedFluxTransformer {
    /// Load the full FLUX transformer from safetensors on CPU, then move stem to GPU.
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        gpu_device: &Device,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        progress.info("Loading transformer blocks on CPU…");

        // Load stem on CPU, then move to GPU.
        // We use our own StemMlpEmbedder (2 Linears + SiLU) since candle's
        // MlpEmbedder has private fields and no to_device method.
        let img_in = linear_to_device(
            &candle_nn::linear(cfg.in_channels, cfg.hidden_size, vb.pp("img_in"))?,
            gpu_device,
        )?;
        let txt_in = linear_to_device(
            &candle_nn::linear(cfg.context_in_dim, cfg.hidden_size, vb.pp("txt_in"))?,
            gpu_device,
        )?;
        let time_in =
            StemMlpEmbedder::load(256, cfg.hidden_size, vb.pp("time_in"))?.to_device(gpu_device)?;
        let vector_in = StemMlpEmbedder::load(cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"))?
            .to_device(gpu_device)?;
        let guidance_in = if cfg.guidance_embed {
            Some(
                StemMlpEmbedder::load(256, cfg.hidden_size, vb.pp("guidance_in"))?
                    .to_device(gpu_device)?,
            )
        } else {
            None
        };

        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());

        let final_layer =
            FinalLayer::load(cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"))?
                .to_device(gpu_device)?;

        // Load blocks on CPU
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.depth {
            double_blocks.push(DoubleBlock::load(cfg, vb_d.pp(idx))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            single_blocks.push(SingleBlock::load(cfg, vb_s.pp(idx))?);
        }

        progress.info(&format!(
            "Offloading: {} double + {} single blocks on CPU, stem on GPU",
            double_blocks.len(),
            single_blocks.len(),
        ));

        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            final_layer,
            double_blocks,
            single_blocks,
            gpu_device: gpu_device.clone(),
        })
    }

    /// Run the full FLUX forward pass with block-level streaming.
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
        let dtype = img.dtype();

        // Positional encoding
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };

        // Stem projections (on GPU)
        let mut txt = txt.apply(&self.txt_in)?;
        let mut img = img.apply(&self.img_in)?;

        // Timestep + guidance + vector embedding
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // Double blocks: stream each from CPU → GPU
        for (i, block) in self.double_blocks.iter().enumerate() {
            let gpu_block = block.to_device(&self.gpu_device)?;
            (img, txt) = gpu_block.forward(&img, &txt, &vec_, &pe)?;
            self.gpu_device.synchronize()?;
            drop(gpu_block);
            tracing::trace!("double block {i} done");
        }

        // Single blocks: stream each from CPU → GPU
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        let txt_len = txt.dim(1)?;
        for (i, block) in self.single_blocks.iter().enumerate() {
            let gpu_block = block.to_device(&self.gpu_device)?;
            img = gpu_block.forward(&img, &vec_, &pe)?;
            self.gpu_device.synchronize()?;
            drop(gpu_block);
            tracing::trace!("single block {i} done");
        }

        // Final layer (on GPU)
        let img = img.i((.., txt_len..))?;
        self.final_layer.forward(&img, &vec_)
    }
}

/// Simple MLP embedder (2 Linears + SiLU) for stem layers.
/// Reimplemented here because candle's `MlpEmbedder` has private fields.
struct StemMlpEmbedder {
    in_layer: Linear,
    out_layer: Linear,
}

impl StemMlpEmbedder {
    fn load(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            in_layer: candle_nn::linear(in_sz, h_sz, vb.pp("in_layer"))?,
            out_layer: candle_nn::linear(h_sz, h_sz, vb.pp("out_layer"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            in_layer: linear_to_device(&self.in_layer, dev)?,
            out_layer: linear_to_device(&self.out_layer, dev)?,
        })
    }
}

impl Module for StemMlpEmbedder {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.in_layer)?.silu()?.apply(&self.out_layer)
    }
}
