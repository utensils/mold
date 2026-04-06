//! Block-level GPU offloading for the Qwen-Image BF16 transformer.
//!
//! Streams 60 transformer blocks one at a time between CPU and GPU during each
//! denoising step. Reduces peak VRAM from ~38GB (full BF16) to ~4-6GB at the
//! cost of 3-5x slower inference. This enables native 1328×1328 generation on
//! 24GB cards.
//!
//! Self-contained: defines its own block types with `to_device()` methods,
//! following the same pattern as `flux/offload.rs`.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use candle_transformers::models::z_image::transformer::apply_rotary_emb;

use super::quantized_transformer::QwenRopeEmbedder;
use super::transformer::{QwenImageConfig, MAX_PERIOD};
use crate::progress::ProgressReporter;

// ── Device-transfer helpers ──────────────────────────────────────────────────

fn linear_to_device(l: &Linear, dev: &Device) -> Result<Linear> {
    let w = l.weight().to_device(dev)?;
    let b = l.bias().map(|b| b.to_device(dev)).transpose()?;
    Ok(Linear::new(w, b))
}

fn rms_norm_to_device(rn: &candle_nn::RmsNorm, dev: &Device) -> Result<candle_nn::RmsNorm> {
    // candle_nn::RmsNorm stores a single weight tensor + eps
    // We clone to avoid moving out of the reference, then transfer
    let cloned = rn.clone();
    let w = cloned.into_inner().weight().to_device(dev)?;
    Ok(candle_nn::RmsNorm::new(w, 1e-6))
}

fn load_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::RmsNorm> {
    Ok(candle_nn::rms_norm(size, eps, vb)?)
}

// ── Parameterless LayerNorm ──────────────────────────────────────────────────

/// LayerNorm without learnable parameters. No device transfer needed (no weights).
#[derive(Debug, Clone)]
struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoParams {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)
    }
}

// ── FeedForward (SwiGLU) ─────────────────────────────────────────────────────

struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn load(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(dim, hidden_dim, vb.pp("w1"))?,
            w2: linear_no_bias(hidden_dim, dim, vb.pp("w2"))?,
            w3: linear_no_bias(dim, hidden_dim, vb.pp("w3"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            w1: linear_to_device(&self.w1, dev)?,
            w2: linear_to_device(&self.w2, dev)?,
            w3: linear_to_device(&self.w3, dev)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = x.apply(&self.w1)?.silu()?;
        let x3 = x.apply(&self.w3)?;
        Ok((x1 * x3)?.apply(&self.w2)?)
    }
}

// ── Joint Attention ──────────────────────────────────────────────────────────

struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    add_out_proj: Linear,
    norm_q: candle_nn::RmsNorm,
    norm_k: candle_nn::RmsNorm,
    norm_added_q: candle_nn::RmsNorm,
    norm_added_k: candle_nn::RmsNorm,
    n_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn load(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim;
        let text_dim = cfg.joint_attention_dim;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.attention_head_dim;
        let qkv_dim = n_heads * head_dim;

        Ok(Self {
            to_q: linear_no_bias(dim, qkv_dim, vb.pp("to_q"))?,
            to_k: linear_no_bias(dim, qkv_dim, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim, qkv_dim, vb.pp("to_v"))?,
            to_out: linear_no_bias(qkv_dim, dim, vb.pp("to_out_0"))?,
            add_q_proj: linear_no_bias(text_dim, qkv_dim, vb.pp("add_q_proj"))?,
            add_k_proj: linear_no_bias(text_dim, qkv_dim, vb.pp("add_k_proj"))?,
            add_v_proj: linear_no_bias(text_dim, qkv_dim, vb.pp("add_v_proj"))?,
            add_out_proj: linear_no_bias(qkv_dim, text_dim, vb.pp("to_add_out"))?,
            norm_q: load_rms_norm(head_dim, 1e-6, vb.pp("norm_q"))?,
            norm_k: load_rms_norm(head_dim, 1e-6, vb.pp("norm_k"))?,
            norm_added_q: load_rms_norm(head_dim, 1e-6, vb.pp("norm_added_q"))?,
            norm_added_k: load_rms_norm(head_dim, 1e-6, vb.pp("norm_added_k"))?,
            n_heads,
            head_dim,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            to_q: linear_to_device(&self.to_q, dev)?,
            to_k: linear_to_device(&self.to_k, dev)?,
            to_v: linear_to_device(&self.to_v, dev)?,
            to_out: linear_to_device(&self.to_out, dev)?,
            add_q_proj: linear_to_device(&self.add_q_proj, dev)?,
            add_k_proj: linear_to_device(&self.add_k_proj, dev)?,
            add_v_proj: linear_to_device(&self.add_v_proj, dev)?,
            add_out_proj: linear_to_device(&self.add_out_proj, dev)?,
            norm_q: rms_norm_to_device(&self.norm_q, dev)?,
            norm_k: rms_norm_to_device(&self.norm_k, dev)?,
            norm_added_q: rms_norm_to_device(&self.norm_added_q, dev)?,
            norm_added_k: rms_norm_to_device(&self.norm_added_k, dev)?,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
        })
    }

    fn apply_qk_norm(&self, x: &Tensor, norm: &candle_nn::RmsNorm) -> candle_core::Result<Tensor> {
        let (b, seq, heads, head_dim) = x.dims4()?;
        let flat = x.reshape((b * seq * heads, head_dim))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, seq, heads, head_dim))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        txt_mask: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        img_seq_len: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (b, _, _) = img_hidden.dims3()?;

        let q_img = img_hidden.apply(&self.to_q)?;
        let k_img = img_hidden.apply(&self.to_k)?;
        let v_img = img_hidden.apply(&self.to_v)?;
        let q_txt = txt_hidden.apply(&self.add_q_proj)?;
        let k_txt = txt_hidden.apply(&self.add_k_proj)?;
        let v_txt = txt_hidden.apply(&self.add_v_proj)?;

        let txt_seq_len = txt_hidden.dim(1)?;

        let q_img = q_img.reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let k_img = k_img.reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let v_img = v_img.reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let q_txt = q_txt.reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let k_txt = k_txt.reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let v_txt = v_txt.reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;

        let q_img = self.apply_qk_norm(&q_img, &self.norm_q)?;
        let k_img = self.apply_qk_norm(&k_img, &self.norm_k)?;
        let q_txt = self.apply_qk_norm(&q_txt, &self.norm_added_q)?;
        let k_txt = self.apply_qk_norm(&k_txt, &self.norm_added_k)?;

        let q_img = apply_rotary_emb(&q_img, img_cos, img_sin)?;
        let k_img = apply_rotary_emb(&k_img, img_cos, img_sin)?;
        let q_txt = apply_rotary_emb(&q_txt, txt_cos, txt_sin)?;
        let k_txt = apply_rotary_emb(&k_txt, txt_cos, txt_sin)?;

        // [text, image] order
        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let img_mask = Tensor::ones((b, img_seq_len), DType::U8, q.device())?;
        let key_mask = Tensor::cat(&[txt_mask, &img_mask], 1)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let on_true = key_mask.zeros_like()?.to_dtype(q.dtype())?;
        let on_false = Tensor::new(f32::NEG_INFINITY, q.device())?
            .broadcast_as(key_mask.shape())?
            .to_dtype(q.dtype())?;
        let key_mask = key_mask.where_cond(&on_true, &on_false)?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        attn_weights = attn_weights.broadcast_add(&key_mask)?;
        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn = attn_weights.matmul(&v)?;

        let total_seq = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((b, total_seq, ()))?;

        let txt_attn = attn.narrow(1, 0, txt_seq_len)?;
        let img_attn = attn.narrow(1, txt_seq_len, img_seq_len)?;

        let img_out = img_attn.apply(&self.to_out)?;
        let txt_out = txt_attn.apply(&self.add_out_proj)?.broadcast_mul(
            &txt_mask
                .unsqueeze(D::Minus1)?
                .to_dtype(txt_hidden.dtype())?,
        )?;

        Ok((img_out, txt_out))
    }
}

// ── Transformer Block ────────────────────────────────────────────────────────

struct OffloadedQwenBlock {
    norm1: LayerNormNoParams,
    norm1_context: LayerNormNoParams,
    attn: JointAttention,
    ff: FeedForward,
    ff_context: FeedForward,
    norm2: LayerNormNoParams,
    norm2_context: LayerNormNoParams,
    adaln_modulation: Linear,
    adaln_context_modulation: Linear,
}

impl OffloadedQwenBlock {
    fn load(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim;
        let text_dim = cfg.joint_attention_dim;
        let hidden_dim = cfg.hidden_dim();

        Ok(Self {
            norm1: LayerNormNoParams::new(cfg.norm_eps),
            norm1_context: LayerNormNoParams::new(cfg.norm_eps),
            attn: JointAttention::load(cfg, vb.pp("attn"))?,
            ff: FeedForward::load(dim, hidden_dim, vb.pp("ff"))?,
            ff_context: FeedForward::load(text_dim, text_dim * 4, vb.pp("ff_context"))?,
            norm2: LayerNormNoParams::new(cfg.norm_eps),
            norm2_context: LayerNormNoParams::new(cfg.norm_eps),
            adaln_modulation: linear(dim, 6 * dim, vb.pp("norm1").pp("linear"))?,
            adaln_context_modulation: linear(
                dim,
                6 * text_dim,
                vb.pp("norm1_context").pp("linear"),
            )?,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm1: self.norm1.clone(),
            norm1_context: self.norm1_context.clone(),
            attn: self.attn.to_device(dev)?,
            ff: self.ff.to_device(dev)?,
            ff_context: self.ff_context.to_device(dev)?,
            norm2: self.norm2.clone(),
            norm2_context: self.norm2_context.clone(),
            adaln_modulation: linear_to_device(&self.adaln_modulation, dev)?,
            adaln_context_modulation: linear_to_device(&self.adaln_context_modulation, dev)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_hidden: &Tensor,
        txt_hidden: &Tensor,
        txt_mask: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let img_seq_len = img_hidden.dim(1)?;

        // AdaLN modulation (6 params per stream)
        let img_mod = temb.silu()?.apply(&self.adaln_modulation)?.unsqueeze(1)?;
        let img_chunks = img_mod.chunk(6, D::Minus1)?;

        let txt_mod = temb
            .silu()?
            .apply(&self.adaln_context_modulation)?
            .unsqueeze(1)?;
        let txt_chunks = txt_mod.chunk(6, D::Minus1)?;

        // Attention
        let img_attn_in = self
            .norm1
            .forward(img_hidden)?
            .broadcast_mul(&(&img_chunks[1] + 1.0)?)?
            .broadcast_add(&img_chunks[0])?;
        let txt_attn_in = self
            .norm1_context
            .forward(txt_hidden)?
            .broadcast_mul(&(&txt_chunks[1] + 1.0)?)?
            .broadcast_add(&txt_chunks[0])?;

        let (img_attn, txt_attn) = self.attn.forward(
            &img_attn_in,
            &txt_attn_in,
            txt_mask,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            img_seq_len,
        )?;

        // Gate + residual
        let img_hidden = (img_hidden + img_chunks[2].broadcast_mul(&img_attn)?)?;
        let txt_dtype = txt_hidden.dtype();
        let txt_hidden = (txt_hidden + txt_chunks[2].broadcast_mul(&txt_attn)?)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?.to_dtype(txt_dtype)?)?;

        // Feedforward
        let img_mlp_in = self
            .norm2
            .forward(&img_hidden)?
            .broadcast_mul(&(&img_chunks[4] + 1.0)?)?
            .broadcast_add(&img_chunks[3])?;
        let img_ff = self.ff.forward(&img_mlp_in)?;
        let img_hidden = (img_hidden + img_chunks[5].broadcast_mul(&img_ff)?)?;

        let txt_mlp_in = self
            .norm2_context
            .forward(&txt_hidden)?
            .broadcast_mul(&(&txt_chunks[4] + 1.0)?)?
            .broadcast_add(&txt_chunks[3])?;
        let txt_ff = self.ff_context.forward(&txt_mlp_in)?;
        let txt_dtype = txt_hidden.dtype();
        let txt_hidden = (txt_hidden + txt_chunks[5].broadcast_mul(&txt_ff)?)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?.to_dtype(txt_dtype)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

// ── Timestep Embedding ───────────────────────────────────────────────────────

const FREQUENCY_EMBEDDING_SIZE: usize = 256;

struct TimestepProjEmbeddings {
    linear1: Linear,
    linear2: Linear,
}

impl TimestepProjEmbeddings {
    fn load(inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(FREQUENCY_EMBEDDING_SIZE, inner_dim, vb.pp("linear_1"))?,
            linear2: linear(inner_dim, inner_dim, vb.pp("linear_2"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            linear1: linear_to_device(&self.linear1, dev)?,
            linear2: linear_to_device(&self.linear2, dev)?,
        })
    }
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let dtype = self.linear1.weight().dtype();
        let half = FREQUENCY_EMBEDDING_SIZE / 2;
        let freqs = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let t_freq = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
        Ok(t_freq.apply(&self.linear1)?.silu()?.apply(&self.linear2)?)
    }
}

// ── Output Layer ─────────────────────────────────────────────────────────────

struct OutputLayer {
    norm_final: LayerNormNoParams,
    linear: Linear,
    adaln_linear: Linear,
}

impl OutputLayer {
    fn load(inner_dim: usize, out_channels: usize, patch_size: usize, vb: VarBuilder) -> Result<Self> {
        let output_dim = patch_size * patch_size * out_channels;
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            linear: linear(inner_dim, output_dim, vb.pp("proj_out"))?,
            adaln_linear: linear(inner_dim, 2 * inner_dim, vb.pp("norm_out").pp("linear"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm_final: self.norm_final.clone(),
            linear: linear_to_device(&self.linear, dev)?,
            adaln_linear: linear_to_device(&self.adaln_linear, dev)?,
        })
    }
    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let mod_params = temb.silu()?.apply(&self.adaln_linear)?;
        let chunks = mod_params.chunk(2, D::Minus1)?;
        // AdaLayerNormContinuous: scale = chunk[0], shift = chunk[1]
        let scale = chunks[0].unsqueeze(1)?;
        let shift = chunks[1].unsqueeze(1)?;
        let x = self
            .norm_final
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        Ok(x.apply(&self.linear)?)
    }
}

// ── Main offloaded transformer ───────────────────────────────────────────────

/// BF16 Qwen-Image transformer with blocks on CPU, streamed to GPU one at a time.
pub(crate) struct OffloadedQwenImageTransformer {
    // Stem layers on GPU permanently
    time_embed: TimestepProjEmbeddings,
    img_in: Linear,
    txt_in: Linear,
    txt_norm: candle_nn::RmsNorm,
    output_layer: OutputLayer,
    rope_embedder: QwenRopeEmbedder,
    cfg: QwenImageConfig,
    // 60 blocks on CPU
    blocks: Vec<OffloadedQwenBlock>,
    gpu_device: Device,
}

impl OffloadedQwenImageTransformer {
    /// Load the full transformer from BF16 safetensors on CPU, then move stem to GPU.
    pub fn load(
        vb: VarBuilder,
        cfg: &QwenImageConfig,
        gpu_device: &Device,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        progress.info("Loading transformer blocks on CPU for offloading…");

        // Stem layers: load on CPU, move to GPU
        let time_embed =
            TimestepProjEmbeddings::load(cfg.inner_dim, vb.pp("time_text_embed"))?
                .to_device(gpu_device)?;
        let img_in = linear_to_device(
            &linear(cfg.in_channels, cfg.inner_dim, vb.pp("x_embedder"))?,
            gpu_device,
        )?;
        let txt_in = linear_to_device(
            &linear(
                cfg.joint_attention_dim,
                cfg.joint_attention_dim,
                vb.pp("context_embedder"),
            )?,
            gpu_device,
        )?;
        let txt_norm = rms_norm_to_device(
            &load_rms_norm(cfg.joint_attention_dim, cfg.norm_eps, vb.pp("txt_norm"))?,
            gpu_device,
        )?;
        let output_layer = OutputLayer::load(
            cfg.inner_dim,
            cfg.out_channels,
            cfg.patch_size,
            vb.clone(),
        )?
        .to_device(gpu_device)?;

        // RoPE embedder (frequency tables on CPU, sliced per-forward to GPU)
        let cpu_device = vb.device();
        let rope_embedder =
            QwenRopeEmbedder::new(10000.0, cfg.axes_dims_rope.clone(), cpu_device, vb.dtype())?;

        // Load 60 blocks on CPU
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(OffloadedQwenBlock::load(cfg, vb_blocks.pp(i))?);
        }

        progress.info(&format!(
            "Offloading: {} blocks on CPU, stem on GPU",
            blocks.len(),
        ));

        Ok(Self {
            time_embed,
            img_in,
            txt_in,
            txt_norm,
            output_layer,
            rope_embedder,
            cfg: cfg.clone(),
            blocks,
            gpu_device: gpu_device.clone(),
        })
    }

    /// Run the full forward pass with block-level streaming.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let device = &self.gpu_device;
        let (_b, _c, h, w) = x.dims4()?;
        let patch_size = self.cfg.patch_size;

        // 1. Timestep embedding (on GPU)
        let temb = self.time_embed.forward(t)?;

        // 2. Pack latents: (B, C, H, W) -> (B, (H/p)*(W/p), C*p*p)
        let hp = h / patch_size;
        let wp = w / patch_size;
        let x_packed = x
            .reshape((_b, _c, hp, patch_size, wp, patch_size))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((_b, hp * wp, _c * patch_size * patch_size))?
            .contiguous()?;
        let mut img = x_packed.apply(&self.img_in)?;

        // 3. Text embedding (on GPU)
        let txt_normed = self.txt_norm.forward(encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;

        // 4. RoPE
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_embedder
                .forward(1, h_tokens, w_tokens, txt_seq_len, device)?;

        // 5. Stream blocks CPU → GPU
        for (i, block) in self.blocks.iter().enumerate() {
            let gpu_block = block.to_device(device)?;
            (img, txt) = gpu_block.forward(
                &img,
                &txt,
                encoder_attention_mask,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
            )?;
            device.synchronize()?;
            drop(gpu_block);
            tracing::trace!("qwen block {i} done");
        }

        // 6. Output layer (on GPU)
        let img_out = self.output_layer.forward(&img, &temb)?;

        // 7. Unpack latents: (B, (H/p)*(W/p), C*p*p) -> (B, C, H, W)
        let x_out = img_out
            .reshape((_b, hp, wp, self.cfg.out_channels, patch_size, patch_size))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((_b, self.cfg.out_channels, h, w))?
            .contiguous()?;
        Ok(x_out)
    }
}
