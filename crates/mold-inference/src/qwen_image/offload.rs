//! Block-level GPU offloading for the Qwen-Image BF16/FP8 transformer.
//!
//! Streams 60 transformer blocks one at a time between CPU and GPU during each
//! denoising step. Reduces peak VRAM from ~38GB (full BF16) to ~4-6GB at the
//! cost of 3-5x slower inference. This enables native 1328×1328 generation on
//! 24GB cards.
//!
//! Self-contained: defines its own block types with `to_device()` methods,
//! following the same pattern as `flux/offload.rs`.
//!
//! Key names match the official diffusers/ComfyUI safetensors format:
//! `img_in`, `txt_in`, `txt_norm`, `time_text_embed.timestep_embedder`,
//! `transformer_blocks.N.img_mod.1`, `transformer_blocks.N.attn.to_out.0`,
//! `transformer_blocks.N.img_mlp.net.{0.proj,2}`, etc.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::z_image::transformer::apply_rotary_emb;

use super::quantized_transformer::{
    build_edit_modulation_index, select_modulation_params, QwenRopeEmbedder,
};
use super::transformer::{QwenImageConfig, MAX_PERIOD};
use crate::progress::ProgressReporter;

// ── Device-transfer helpers ──────────────────────────────────────────────────

fn linear_to_device(l: &Linear, dev: &Device) -> Result<Linear> {
    let w = l.weight().to_device(dev)?;
    let b = l.bias().map(|b| b.to_device(dev)).transpose()?;
    Ok(Linear::new(w, b))
}

fn rms_norm_to_device(
    rn: &candle_nn::RmsNorm,
    eps: f64,
    dev: &Device,
) -> Result<candle_nn::RmsNorm> {
    let cloned = rn.clone();
    let w = cloned.into_inner().weight().to_device(dev)?;
    Ok(candle_nn::RmsNorm::new(w, eps))
}

fn load_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::RmsNorm> {
    Ok(candle_nn::rms_norm(size, eps, vb)?)
}

// ── Parameterless LayerNorm ──────────────────────────────────────────────────

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

// ── GELU MLP (diffusers format: net.0.proj + net.2) ──────────────────────────

struct GeluMlp {
    proj: Linear, // net.0.proj — GELU gate projection
    out: Linear,  // net.2 — output projection
}

impl GeluMlp {
    fn load(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb_net = vb.pp("net");
        Ok(Self {
            proj: linear(in_dim, hidden_dim, vb_net.pp("0").pp("proj"))?,
            out: linear(hidden_dim, in_dim, vb_net.pp("2"))?,
        })
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            proj: linear_to_device(&self.proj, dev)?,
            out: linear_to_device(&self.out, dev)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.apply(&self.proj)?
            .apply(&candle_nn::Activation::GeluPytorchTanh)?
            .apply(&self.out)?)
    }
}

// ── Joint Attention ──────────────────────────────────────────────────────────

struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear, // safetensors: attn.to_out.0
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    add_out_proj: Linear, // safetensors: attn.to_add_out
    norm_q: candle_nn::RmsNorm,
    norm_k: candle_nn::RmsNorm,
    norm_added_q: candle_nn::RmsNorm,
    norm_added_k: candle_nn::RmsNorm,
    n_heads: usize,
    head_dim: usize,
    norm_eps: f64,
}

impl JointAttention {
    fn load(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.attention_head_dim;
        let qkv_dim = n_heads * head_dim;
        // After txt_in projection, text is inner_dim (3072), not joint_attention_dim
        let text_proj_dim = dim;

        Ok(Self {
            to_q: linear(dim, qkv_dim, vb.pp("to_q"))?,
            to_k: linear(dim, qkv_dim, vb.pp("to_k"))?,
            to_v: linear(dim, qkv_dim, vb.pp("to_v"))?,
            to_out: linear(qkv_dim, dim, vb.pp("to_out").pp("0"))?,
            add_q_proj: linear(text_proj_dim, qkv_dim, vb.pp("add_q_proj"))?,
            add_k_proj: linear(text_proj_dim, qkv_dim, vb.pp("add_k_proj"))?,
            add_v_proj: linear(text_proj_dim, qkv_dim, vb.pp("add_v_proj"))?,
            add_out_proj: linear(qkv_dim, text_proj_dim, vb.pp("to_add_out"))?,
            norm_q: load_rms_norm(head_dim, 1e-6, vb.pp("norm_q"))?,
            norm_k: load_rms_norm(head_dim, 1e-6, vb.pp("norm_k"))?,
            norm_added_q: load_rms_norm(head_dim, 1e-6, vb.pp("norm_added_q"))?,
            norm_added_k: load_rms_norm(head_dim, 1e-6, vb.pp("norm_added_k"))?,
            n_heads,
            head_dim,
            norm_eps: cfg.norm_eps,
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
            norm_q: rms_norm_to_device(&self.norm_q, self.norm_eps, dev)?,
            norm_k: rms_norm_to_device(&self.norm_k, self.norm_eps, dev)?,
            norm_added_q: rms_norm_to_device(&self.norm_added_q, self.norm_eps, dev)?,
            norm_added_k: rms_norm_to_device(&self.norm_added_k, self.norm_eps, dev)?,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            norm_eps: self.norm_eps,
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
        let txt_out = txt_attn.apply(&self.add_out_proj)?;

        Ok((img_out, txt_out))
    }
}

// ── Transformer Block ────────────────────────────────────────────────────────

struct OffloadedQwenBlock {
    norm1: LayerNormNoParams,
    norm1_context: LayerNormNoParams,
    attn: JointAttention,
    img_mlp: GeluMlp, // safetensors: img_mlp.net.{0.proj,2}
    txt_mlp: GeluMlp, // safetensors: txt_mlp.net.{0.proj,2}
    norm2: LayerNormNoParams,
    norm2_context: LayerNormNoParams,
    img_mod: Linear, // safetensors: img_mod.1
    txt_mod: Linear, // safetensors: txt_mod.1
}

impl OffloadedQwenBlock {
    fn load(cfg: &QwenImageConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.inner_dim;
        // After txt_in: text embeddings are inner_dim (3072), not joint_attention_dim
        let text_dim = dim;

        Ok(Self {
            norm1: LayerNormNoParams::new(cfg.norm_eps),
            norm1_context: LayerNormNoParams::new(cfg.norm_eps),
            attn: JointAttention::load(cfg, vb.pp("attn"))?,
            img_mlp: GeluMlp::load(dim, dim * 4, vb.pp("img_mlp"))?,
            txt_mlp: GeluMlp::load(text_dim, text_dim * 4, vb.pp("txt_mlp"))?,
            norm2: LayerNormNoParams::new(cfg.norm_eps),
            norm2_context: LayerNormNoParams::new(cfg.norm_eps),
            // Safetensors key: img_mod.1.weight (sequential module index 1)
            img_mod: linear(dim, 6 * dim, vb.pp("img_mod").pp("1"))?,
            txt_mod: linear(dim, 6 * text_dim, vb.pp("txt_mod").pp("1"))?,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm1: self.norm1.clone(),
            norm1_context: self.norm1_context.clone(),
            attn: self.attn.to_device(dev)?,
            img_mlp: self.img_mlp.to_device(dev)?,
            txt_mlp: self.txt_mlp.to_device(dev)?,
            norm2: self.norm2.clone(),
            norm2_context: self.norm2_context.clone(),
            img_mod: linear_to_device(&self.img_mod, dev)?,
            txt_mod: linear_to_device(&self.txt_mod, dev)?,
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
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let img_seq_len = img_hidden.dim(1)?;

        // AdaLN modulation (6 params per stream)
        let img_mod = temb.silu()?.apply(&self.img_mod)?;
        let img_mod = if let Some(modulate_index) = modulate_index {
            select_modulation_params(&img_mod, modulate_index)?
        } else {
            img_mod.unsqueeze(1)?
        };
        let img_chunks = img_mod.chunk(6, D::Minus1)?;

        let txt_temb = if modulate_index.is_some() {
            temb.narrow(0, 0, txt_hidden.dim(0)?)?
        } else {
            temb.clone()
        };
        let txt_mod = txt_temb.silu()?.apply(&self.txt_mod)?.unsqueeze(1)?;
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

        // Gate + residual (matching ComfyUI: y + gate * x, no mask multiplication)
        let img_hidden = (img_hidden + img_chunks[2].broadcast_mul(&img_attn)?)?;
        let txt_hidden = (txt_hidden + txt_chunks[2].broadcast_mul(&txt_attn)?)?;

        // Feedforward
        let img_mlp_in = self
            .norm2
            .forward(&img_hidden)?
            .broadcast_mul(&(&img_chunks[4] + 1.0)?)?
            .broadcast_add(&img_chunks[3])?;
        let img_ff = self.img_mlp.forward(&img_mlp_in)?;
        let img_hidden = (img_hidden + img_chunks[5].broadcast_mul(&img_ff)?)?;

        let txt_mlp_in = self
            .norm2_context
            .forward(&txt_hidden)?
            .broadcast_mul(&(&txt_chunks[4] + 1.0)?)?
            .broadcast_add(&txt_chunks[3])?;
        let txt_ff = self.txt_mlp.forward(&txt_mlp_in)?;
        let txt_hidden = (txt_hidden + txt_chunks[5].broadcast_mul(&txt_ff)?)?;

        // Return (text, image) to match ComfyUI block output order
        Ok((txt_hidden, img_hidden))
    }
}

// ── Timestep Embedding ───────────────────────────────────────────────────────

const FREQUENCY_EMBEDDING_SIZE: usize = 256;

struct TimestepProjEmbeddings {
    linear1: Linear,
    linear2: Linear,
}

impl TimestepProjEmbeddings {
    /// Load from safetensors key: time_text_embed.timestep_embedder.linear_{1,2}
    fn load(inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("timestep_embedder");
        Ok(Self {
            linear1: linear(FREQUENCY_EMBEDDING_SIZE, inner_dim, vb.pp("linear_1"))?,
            linear2: linear(inner_dim, inner_dim, vb.pp("linear_2"))?,
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
    proj_out: Linear,     // safetensors: proj_out
    adaln_linear: Linear, // safetensors: norm_out.linear
}

impl OutputLayer {
    fn load(
        inner_dim: usize,
        out_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let output_dim = patch_size * patch_size * out_channels;
        Ok(Self {
            norm_final: LayerNormNoParams::new(1e-6),
            proj_out: linear(inner_dim, output_dim, vb.pp("proj_out"))?,
            adaln_linear: linear(inner_dim, 2 * inner_dim, vb.pp("norm_out").pp("linear"))?,
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
        Ok(x.apply(&self.proj_out)?)
    }
}

// ── Main offloaded transformer ───────────────────────────────────────────────

/// Where a transformer block lives — GPU (no transfer needed) or CPU (streamed per step).
enum BlockResidency {
    Gpu(OffloadedQwenBlock),
    Cpu(OffloadedQwenBlock),
}

/// BF16/FP8 Qwen-Image transformer with dynamic GPU/CPU block placement.
///
/// After loading, measures free VRAM and moves as many blocks to GPU as fit.
/// During each denoising step, GPU-resident blocks execute in-place (zero transfer
/// cost) while CPU-resident blocks stream one at a time. This maximizes GPU
/// utilization instead of leaving VRAM idle during denoising.
pub(crate) struct OffloadedQwenImageTransformer {
    // Stem layers on GPU permanently
    time_embed: TimestepProjEmbeddings,
    img_in: Linear,               // safetensors: img_in
    txt_in: Linear,               // safetensors: txt_in
    txt_norm: candle_nn::RmsNorm, // safetensors: txt_norm
    output_layer: OutputLayer,
    rope_embedder: QwenRopeEmbedder,
    cfg: QwenImageConfig,
    // Blocks: either GPU-resident or CPU-resident
    blocks: Vec<BlockResidency>,
    gpu_device: Device,
    gpu_resident_count: usize,
}

impl OffloadedQwenImageTransformer {
    /// Load the transformer with dynamic GPU/CPU block placement.
    ///
    /// Loads as many blocks directly on GPU as VRAM allows, with remaining
    /// blocks loaded on CPU for per-step streaming.
    pub fn load(
        gpu_vb: VarBuilder,
        cpu_vb: VarBuilder,
        cfg: &QwenImageConfig,
        gpu_device: &Device,
        gpu_ordinal: usize,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        progress.info("Loading transformer with dynamic GPU/CPU placement…");

        // Stem layers: load directly on GPU
        let time_embed = TimestepProjEmbeddings::load(cfg.inner_dim, gpu_vb.pp("time_text_embed"))?;
        let img_in = linear(cfg.in_channels, cfg.inner_dim, gpu_vb.pp("img_in"))?;
        let txt_in = linear(cfg.joint_attention_dim, cfg.inner_dim, gpu_vb.pp("txt_in"))?;
        let txt_norm = load_rms_norm(cfg.joint_attention_dim, cfg.norm_eps, gpu_vb.pp("txt_norm"))?;
        let output_layer = OutputLayer::load(
            cfg.inner_dim,
            cfg.out_channels,
            cfg.patch_size,
            gpu_vb.clone(),
        )?;

        // RoPE embedder (frequency tables on CPU, sliced per-forward to GPU)
        let rope_embedder = QwenRopeEmbedder::new(
            10000.0,
            cfg.axes_dims_rope.clone(),
            &Device::Cpu,
            DType::F32,
        )?;

        // Measure free VRAM after stem layers and decide how many blocks fit
        gpu_device.synchronize()?;
        let free_vram = crate::device::free_vram_bytes(gpu_ordinal).unwrap_or(0);
        const VRAM_HEADROOM: u64 = 4_500_000_000; // 4.5GB for attention + activations + CUDA overhead
        let vram_budget = free_vram.saturating_sub(VRAM_HEADROOM);

        // Load first block on CPU to measure actual size
        let first_block = OffloadedQwenBlock::load(cfg, cpu_vb.pp("transformer_blocks").pp(0))?;
        let block_size = Self::block_size_bytes(&first_block);
        let max_gpu_blocks = vram_budget.checked_div(block_size).unwrap_or(0) as usize;
        let max_gpu_blocks = max_gpu_blocks.min(cfg.num_layers);

        progress.info(&format!(
            "Block size: {} MB, VRAM budget: {} MB → {} of {} blocks on GPU",
            block_size / (1024 * 1024),
            vram_budget / (1024 * 1024),
            max_gpu_blocks,
            cfg.num_layers,
        ));

        // Place first block
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        if max_gpu_blocks > 0 {
            // Re-load directly on GPU for GPU-resident blocks
            let gpu_block = OffloadedQwenBlock::load(cfg, gpu_vb.pp("transformer_blocks").pp(0))?;
            blocks.push(BlockResidency::Gpu(gpu_block));
            drop(first_block); // discard CPU copy
        } else {
            blocks.push(BlockResidency::Cpu(first_block));
        }

        // Load remaining blocks — GPU-direct until budget exhausted, then CPU
        for i in 1..cfg.num_layers {
            if i < max_gpu_blocks {
                let block = OffloadedQwenBlock::load(cfg, gpu_vb.pp("transformer_blocks").pp(i))?;
                blocks.push(BlockResidency::Gpu(block));
            } else {
                let block = OffloadedQwenBlock::load(cfg, cpu_vb.pp("transformer_blocks").pp(i))?;
                blocks.push(BlockResidency::Cpu(block));
            }
            if (i + 1) % 10 == 0 || i + 1 == cfg.num_layers {
                progress.info(&format!(
                    "Loaded {}/{} blocks ({} GPU, {} CPU)",
                    i + 1,
                    cfg.num_layers,
                    (i + 1).min(max_gpu_blocks),
                    (i + 1).saturating_sub(max_gpu_blocks),
                ));
            }
        }
        let gpu_count = max_gpu_blocks;

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
            gpu_resident_count: gpu_count,
        })
    }

    /// Compute actual block size in bytes by summing all weight tensors.
    fn block_size_bytes(block: &OffloadedQwenBlock) -> u64 {
        let lb = |l: &Linear| -> u64 {
            let w = (l.weight().elem_count() * l.weight().dtype().size_in_bytes()) as u64;
            let b = l
                .bias()
                .map(|b| (b.elem_count() * b.dtype().size_in_bytes()) as u64)
                .unwrap_or(0);
            w + b
        };
        let rb = |r: &candle_nn::RmsNorm| -> u64 {
            let w = r.clone().into_inner().weight().clone();
            (w.elem_count() * w.dtype().size_in_bytes()) as u64
        };
        lb(&block.img_mod)
            + lb(&block.txt_mod)
            + lb(&block.attn.to_q)
            + lb(&block.attn.to_k)
            + lb(&block.attn.to_v)
            + lb(&block.attn.to_out)
            + lb(&block.attn.add_q_proj)
            + lb(&block.attn.add_k_proj)
            + lb(&block.attn.add_v_proj)
            + lb(&block.attn.add_out_proj)
            + rb(&block.attn.norm_q)
            + rb(&block.attn.norm_k)
            + rb(&block.attn.norm_added_q)
            + rb(&block.attn.norm_added_k)
            + lb(&block.img_mlp.proj)
            + lb(&block.img_mlp.out)
            + lb(&block.txt_mlp.proj)
            + lb(&block.txt_mlp.out)
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

        // 3. Text embedding (on GPU): norm then project
        let txt_normed = self.txt_norm.forward(encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;

        // 4. RoPE (cast to computation dtype — RoPE tables may be F32 from CPU)
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let compute_dtype = x.dtype();
        let (img_cos, img_sin, txt_cos, txt_sin) = {
            let (ic, is, tc, ts) =
                self.rope_embedder
                    .forward(1, h_tokens, w_tokens, txt_seq_len, device)?;
            (
                ic.to_dtype(compute_dtype)?,
                is.to_dtype(compute_dtype)?,
                tc.to_dtype(compute_dtype)?,
                ts.to_dtype(compute_dtype)?,
            )
        };

        // 5. Execute blocks — GPU-resident run in-place, CPU blocks stream
        tracing::debug!(
            gpu_resident = self.gpu_resident_count,
            cpu_streaming = self.blocks.len() - self.gpu_resident_count,
            "denoising step"
        );
        //    Block returns (text, image) — matching ComfyUI convention
        for (i, residency) in self.blocks.iter().enumerate() {
            match residency {
                BlockResidency::Gpu(block) => {
                    // Already on GPU — execute directly, no transfer
                    (txt, img) = block.forward(
                        &img,
                        &txt,
                        encoder_attention_mask,
                        &temb,
                        &img_cos,
                        &img_sin,
                        &txt_cos,
                        &txt_sin,
                        None,
                    )?;
                }
                BlockResidency::Cpu(block) => {
                    // Stream CPU → GPU, execute, drop GPU copy
                    let gpu_block = block.to_device(device)?;
                    (txt, img) = gpu_block.forward(
                        &img,
                        &txt,
                        encoder_attention_mask,
                        &temb,
                        &img_cos,
                        &img_sin,
                        &txt_cos,
                        &txt_sin,
                        None,
                    )?;
                    device.synchronize()?;
                    drop(gpu_block);
                }
            }
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

    pub fn forward_packed(
        &self,
        packed_hidden_states: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
        img_shapes: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        let device = &self.gpu_device;
        let batch = packed_hidden_states.dim(0)?;
        let mut timestep = t.clone();
        let modulate_index = if self.cfg.zero_cond_t {
            timestep = Tensor::cat(&[&timestep, &(timestep.zeros_like()?)], 0)?;
            Some(build_edit_modulation_index(img_shapes, batch, device)?)
        } else {
            None
        };

        let temb = self.time_embed.forward(&timestep)?;
        let mut img = packed_hidden_states.apply(&self.img_in)?;
        let txt_normed = self.txt_norm.forward(encoder_hidden_states)?;
        let mut txt = txt_normed.apply(&self.txt_in)?;

        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let compute_dtype = packed_hidden_states.dtype();
        let (img_cos, img_sin, txt_cos, txt_sin) = {
            let (ic, is, tc, ts) =
                self.rope_embedder
                    .forward_shapes(img_shapes, txt_seq_len, device)?;
            (
                ic.to_dtype(compute_dtype)?,
                is.to_dtype(compute_dtype)?,
                tc.to_dtype(compute_dtype)?,
                ts.to_dtype(compute_dtype)?,
            )
        };

        for residency in &self.blocks {
            match residency {
                BlockResidency::Gpu(block) => {
                    (txt, img) = block.forward(
                        &img,
                        &txt,
                        encoder_attention_mask,
                        &temb,
                        &img_cos,
                        &img_sin,
                        &txt_cos,
                        &txt_sin,
                        modulate_index.as_ref(),
                    )?;
                }
                BlockResidency::Cpu(block) => {
                    let gpu_block = block.to_device(device)?;
                    (txt, img) = gpu_block.forward(
                        &img,
                        &txt,
                        encoder_attention_mask,
                        &temb,
                        &img_cos,
                        &img_sin,
                        &txt_cos,
                        &txt_sin,
                        modulate_index.as_ref(),
                    )?;
                    device.synchronize()?;
                    drop(gpu_block);
                }
            }
        }

        let out_temb = if self.cfg.zero_cond_t {
            temb.narrow(0, 0, batch)?
        } else {
            temb
        };
        self.output_layer.forward(&img, &out_temb)
    }
}
