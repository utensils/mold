//! Qwen-Image Transformer (QwenImageTransformer2DModel).
//!
//! This is architecturally similar to Z-Image's transformer but with key differences:
//! - 60 identical dual-stream blocks (no separate noise_refiner/context_refiner)
//! - `QwenTimestepProjEmbeddings` for timestep embedding
//! - `img_in` / `txt_in` linear projections for input embedding
//! - `txt_norm` (RMSNorm) applied to text encoder output
//! - Output via AdaLN + projection
//! - 3D RoPE with axes_dims_rope=[16, 56, 56]
//! - inner_dim = 24 heads * 128 head_dim = 3072
//! - joint_attention_dim = 3584 (matches text encoder hidden_size)
//!
//! We reuse Z-Image's building blocks (RopeEmbedder, apply_rotary_emb, patchify/unpatchify,
//! FeedForward, QkNorm, etc.) since the core components are identical.

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear, linear_no_bias, VarBuilder};
use candle_transformers::models::with_tracing::RmsNorm;
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, FeedForward, RopeEmbedder,
};

/// Qwen-Image transformer configuration.
#[derive(Debug, Clone)]
pub(crate) struct QwenImageConfig {
    /// Number of attention heads (24).
    pub num_attention_heads: usize,
    /// Dimension per attention head (128).
    pub attention_head_dim: usize,
    /// Inner dimension = num_attention_heads * attention_head_dim (3072).
    pub inner_dim: usize,
    /// Text encoder output dimension (3584).
    pub joint_attention_dim: usize,
    /// Number of transformer blocks (60).
    pub num_layers: usize,
    /// Input channels from VAE latent (64 = 16 * patch_size^2).
    pub in_channels: usize,
    /// Output channels per patch element (16).
    pub out_channels: usize,
    /// Spatial patch size (2).
    pub patch_size: usize,
    /// 3D RoPE axis dimensions [16, 56, 56].
    pub axes_dims_rope: Vec<usize>,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl Default for QwenImageConfig {
    fn default() -> Self {
        Self::qwen_image_2512()
    }
}

impl QwenImageConfig {
    /// Create configuration for Qwen-Image-2512.
    pub fn qwen_image_2512() -> Self {
        let num_attention_heads = 24;
        let attention_head_dim = 128;
        Self {
            num_attention_heads,
            attention_head_dim,
            inner_dim: num_attention_heads * attention_head_dim, // 3072
            joint_attention_dim: 3584,
            num_layers: 60,
            in_channels: 64,  // after patchify: 16 * 2 * 2
            out_channels: 16, // VAE latent channels
            patch_size: 2,
            axes_dims_rope: vec![16, 56, 56],
            norm_eps: 1e-6,
        }
    }

    /// Hidden dimension for FFN: int(inner_dim / 3 * 8) = 8192 for inner_dim=3072.
    pub fn hidden_dim(&self) -> usize {
        (self.inner_dim / 3) * 8
    }
}

// ==================== Timestep Projection Embedding ====================

/// QwenTimestepProjEmbeddings: sinusoidal timestep embedding projected through MLP.
///
/// Matches diffusers `QwenTimestepProjEmbeddings`:
///   1. Sinusoidal encoding of timestep -> frequency_embedding_size
///   2. Linear -> SiLU -> Linear -> inner_dim
#[derive(Debug, Clone)]
struct TimestepProjEmbeddings {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    frequency_embedding_size: usize,
}

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
pub(crate) const MAX_PERIOD: f64 = 10000.0;

impl TimestepProjEmbeddings {
    fn new(inner_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let linear1 = linear(FREQUENCY_EMBEDDING_SIZE, inner_dim, vb.pp("linear_1"))?;
        let linear2 = linear(inner_dim, inner_dim, vb.pp("linear_2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn timestep_embedding(
        &self,
        t: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        let half = self.frequency_embedding_size / 2;
        let freqs = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        embedding.to_dtype(dtype)
    }

    fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let device = t.device();
        let dtype = self.linear1.weight().dtype();
        let t_freq = self.timestep_embedding(t, device, dtype)?;
        t_freq.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

// ==================== Joint Attention Block ====================

/// Qwen-Image joint attention with separate Q/K/V for image and text streams.
///
/// Each block processes both image and text through shared attention:
/// 1. Separate norm + Q/K/V projections for image and text
/// 2. Concatenate Q/K/V along sequence dimension
/// 3. Unified attention computation
/// 4. Split output back to image and text
/// 5. Separate output projections
#[derive(Debug, Clone)]
struct JointAttention {
    // Image stream projections
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    // Text stream projections
    add_q_proj: candle_nn::Linear,
    add_k_proj: candle_nn::Linear,
    add_v_proj: candle_nn::Linear,
    add_out_proj: candle_nn::Linear,
    // QK normalization
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    // Dimensions
    n_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = cfg.inner_dim;
        let text_dim = cfg.joint_attention_dim;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.attention_head_dim;
        let qkv_dim = n_heads * head_dim; // 3072

        // Image projections
        let to_q = linear_no_bias(dim, qkv_dim, vb.pp("to_q"))?;
        let to_k = linear_no_bias(dim, qkv_dim, vb.pp("to_k"))?;
        let to_v = linear_no_bias(dim, qkv_dim, vb.pp("to_v"))?;
        let to_out = linear_no_bias(qkv_dim, dim, vb.pp("to_out_0"))?;

        // Text projections
        let add_q_proj = linear_no_bias(text_dim, qkv_dim, vb.pp("add_q_proj"))?;
        let add_k_proj = linear_no_bias(text_dim, qkv_dim, vb.pp("add_k_proj"))?;
        let add_v_proj = linear_no_bias(text_dim, qkv_dim, vb.pp("add_v_proj"))?;
        let add_out_proj = linear_no_bias(qkv_dim, text_dim, vb.pp("to_add_out"))?;

        // QK normalization
        let norm_q = RmsNorm::new(head_dim, 1e-6, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(head_dim, 1e-6, vb.pp("norm_k"))?;
        let norm_added_q = RmsNorm::new(head_dim, 1e-6, vb.pp("norm_added_q"))?;
        let norm_added_k = RmsNorm::new(head_dim, 1e-6, vb.pp("norm_added_k"))?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            add_q_proj,
            add_k_proj,
            add_v_proj,
            add_out_proj,
            norm_q,
            norm_k,
            norm_added_q,
            norm_added_k,
            n_heads,
            head_dim,
        })
    }

    /// Joint attention forward pass.
    ///
    /// Returns (image_output, text_output).
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

        // Image Q/K/V
        let q_img = img_hidden.apply(&self.to_q)?;
        let k_img = img_hidden.apply(&self.to_k)?;
        let v_img = img_hidden.apply(&self.to_v)?;

        // Text Q/K/V
        let q_txt = txt_hidden.apply(&self.add_q_proj)?;
        let k_txt = txt_hidden.apply(&self.add_k_proj)?;
        let v_txt = txt_hidden.apply(&self.add_v_proj)?;

        let txt_seq_len = txt_hidden.dim(1)?;

        // Reshape to (B, seq, heads, head_dim)
        let q_img = q_img.reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let k_img = k_img.reshape((b, img_seq_len, self.n_heads, self.head_dim))?;
        let v_img = v_img.reshape((b, img_seq_len, self.n_heads, self.head_dim))?;

        let q_txt = q_txt.reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let k_txt = k_txt.reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;
        let v_txt = v_txt.reshape((b, txt_seq_len, self.n_heads, self.head_dim))?;

        // QK normalization (applied per-head: flatten B*seq*heads, apply norm, reshape)
        let q_img = self.apply_qk_norm(&q_img, &self.norm_q)?;
        let k_img = self.apply_qk_norm(&k_img, &self.norm_k)?;
        let q_txt = self.apply_qk_norm(&q_txt, &self.norm_added_q)?;
        let k_txt = self.apply_qk_norm(&k_txt, &self.norm_added_k)?;

        // Apply RoPE to image Q/K
        let q_img = apply_rotary_emb(&q_img, img_cos, img_sin)?;
        let k_img = apply_rotary_emb(&k_img, img_cos, img_sin)?;
        let q_txt = apply_rotary_emb(&q_txt, txt_cos, txt_sin)?;
        let k_txt = apply_rotary_emb(&k_txt, txt_cos, txt_sin)?;

        // Concatenate image + text along sequence dimension
        let q = Tensor::cat(&[&q_img, &q_txt], 1)?; // (B, img+txt, heads, head_dim)
        let k = Tensor::cat(&[&k_img, &k_txt], 1)?;
        let v = Tensor::cat(&[&v_img, &v_txt], 1)?;

        // Transpose to (B, heads, seq, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let img_mask = Tensor::ones((b, img_seq_len), DType::U8, q.device())?;
        let key_mask = Tensor::cat(&[&img_mask, txt_mask], 1)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let on_true = key_mask.zeros_like()?.to_dtype(q.dtype())?;
        let on_false = Tensor::new(f32::NEG_INFINITY, q.device())?
            .broadcast_as(key_mask.shape())?
            .to_dtype(q.dtype())?;
        let key_mask = key_mask.where_cond(&on_true, &on_false)?;
        let attn = self.attention_dispatch(&q, &k, &v, scale, q.device(), Some(&key_mask))?;

        // Reshape: (B, heads, total_seq, head_dim) -> (B, total_seq, inner_dim)
        let total_seq = img_seq_len + txt_seq_len;
        let attn = attn.transpose(1, 2)?.reshape((b, total_seq, ()))?;

        // Split back to image and text
        let img_attn = attn.narrow(1, 0, img_seq_len)?;
        let txt_attn = attn.narrow(1, img_seq_len, txt_seq_len)?;

        // Output projections
        let img_out = img_attn.apply(&self.to_out)?;
        let txt_out = txt_attn.apply(&self.add_out_proj)?.broadcast_mul(
            &txt_mask
                .unsqueeze(D::Minus1)?
                .to_dtype(txt_hidden.dtype())?,
        )?;

        Ok((img_out, txt_out))
    }

    /// Apply QK normalization per head.
    fn apply_qk_norm(&self, x: &Tensor, norm: &RmsNorm) -> candle_core::Result<Tensor> {
        let (b, seq, heads, head_dim) = x.dims4()?;
        let flat = x.reshape((b * seq * heads, head_dim))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, seq, heads, head_dim))
    }

    /// Attention dispatch: use platform-optimal implementation.
    fn attention_dispatch(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        device: &Device,
        key_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        if device.is_metal() {
            candle_nn::ops::sdpa(q, k, v, None, false, scale as f32, 1.0)
        } else {
            // Basic attention for CUDA/CPU
            let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(mask) = key_mask {
                attn_weights = attn_weights.broadcast_add(mask)?;
            }
            attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(v)
        }
    }
}

// ==================== QwenImageTransformerBlock ====================

/// A single dual-stream transformer block for Qwen-Image.
///
/// Each block has:
/// - Separate AdaLN norms for image and text streams
/// - Joint attention across both streams
/// - Separate feedforward networks for image and text
/// - AdaLN modulation from timestep embedding (image stream only)
#[derive(Debug, Clone)]
struct QwenImageTransformerBlock {
    // Image stream norms
    norm1: RmsNorm,
    norm1_context: RmsNorm,
    // Joint attention
    attn: JointAttention,
    // Feedforward
    ff: FeedForward,
    ff_context: FeedForward,
    // Post-attention norms
    norm2: RmsNorm,
    norm2_context: RmsNorm,
    // AdaLN modulation for image stream (scale_msa, gate_msa, scale_mlp, gate_mlp)
    adaln_modulation: candle_nn::Linear,
    // AdaLN modulation for text stream
    adaln_context_modulation: candle_nn::Linear,
}

impl QwenImageTransformerBlock {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = cfg.inner_dim;
        let text_dim = cfg.joint_attention_dim;
        let hidden_dim = cfg.hidden_dim();

        let norm1 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("norm1"))?;
        let norm1_context = RmsNorm::new(text_dim, cfg.norm_eps, vb.pp("norm1_context"))?;

        let attn = JointAttention::new(cfg, vb.pp("attn"))?;

        let ff = FeedForward::new(dim, hidden_dim, vb.pp("ff"))?;
        let ff_context = FeedForward::new(text_dim, text_dim * 4, vb.pp("ff_context"))?;

        let norm2 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("norm2"))?;
        let norm2_context = RmsNorm::new(text_dim, cfg.norm_eps, vb.pp("norm2_context"))?;

        // AdaLN: 6 modulation values (scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp, shift_mlp)
        // but we use 4 (scale + gate for attention and FFN)
        let adaln_modulation = linear(dim, 4 * dim, vb.pp("norm1").pp("linear"))?;
        let adaln_context_modulation =
            linear(dim, 4 * text_dim, vb.pp("norm1_context").pp("linear"))?;

        Ok(Self {
            norm1,
            norm1_context,
            attn,
            ff,
            ff_context,
            norm2,
            norm2_context,
            adaln_modulation,
            adaln_context_modulation,
        })
    }

    /// Forward pass through one transformer block.
    ///
    /// `img_hidden`: (B, img_seq, inner_dim)
    /// `txt_hidden`: (B, txt_seq, joint_attention_dim)
    /// `temb`: (B, inner_dim) timestep embedding
    /// `cos`, `sin`: RoPE embeddings for image positions
    ///
    /// Returns (img_hidden, txt_hidden).
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
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let img_seq_len = img_hidden.dim(1)?;

        // --- AdaLN modulation ---
        let img_mod = temb.apply(&self.adaln_modulation)?.unsqueeze(1)?;
        let img_chunks = img_mod.chunk(4, D::Minus1)?;
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = (
            &img_chunks[0],
            &img_chunks[1],
            &img_chunks[2],
            &img_chunks[3],
        );

        let txt_mod = temb.apply(&self.adaln_context_modulation)?.unsqueeze(1)?;
        let txt_chunks = txt_mod.chunk(4, D::Minus1)?;
        let (c_scale_msa, c_gate_msa, c_scale_mlp, c_gate_mlp) = (
            &txt_chunks[0],
            &txt_chunks[1],
            &txt_chunks[2],
            &txt_chunks[3],
        );

        // --- Attention ---
        // Image: norm + scale
        let img_normed = self.norm1.forward(img_hidden)?;
        let img_scaled = img_normed.broadcast_mul(&(scale_msa + 1.0)?)?;

        // Text: norm + scale
        let txt_normed = self.norm1_context.forward(txt_hidden)?;
        let txt_scaled = txt_normed.broadcast_mul(&(c_scale_msa + 1.0)?)?;

        // Joint attention
        let (img_attn, txt_attn) = self.attn.forward(
            &img_scaled,
            &txt_scaled,
            txt_mask,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            img_seq_len,
        )?;

        // Gate + residual
        let img_hidden = (img_hidden + gate_msa.tanh()?.broadcast_mul(&img_attn)?)?;
        let txt_dtype = txt_hidden.dtype();
        let txt_hidden = (txt_hidden + c_gate_msa.tanh()?.broadcast_mul(&txt_attn)?)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?.to_dtype(txt_dtype)?)?;

        // --- Feedforward ---
        // Image: norm + scale + FF + gate + residual
        let img_normed = self.norm2.forward(&img_hidden)?;
        let img_scaled = img_normed.broadcast_mul(&(scale_mlp + 1.0)?)?;
        let img_ff = self.ff.forward(&img_scaled)?;
        let img_hidden = (img_hidden + gate_mlp.tanh()?.broadcast_mul(&img_ff)?)?;

        // Text: norm + scale + FF + gate + residual
        let txt_normed = self.norm2_context.forward(&txt_hidden)?;
        let txt_scaled = txt_normed.broadcast_mul(&(c_scale_mlp + 1.0)?)?;
        let txt_ff = self.ff_context.forward(&txt_scaled)?;
        let txt_dtype = txt_hidden.dtype();
        let txt_hidden = (txt_hidden + c_gate_mlp.tanh()?.broadcast_mul(&txt_ff)?)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?.to_dtype(txt_dtype)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

// ==================== Output Layer ====================

/// Final output layer: AdaLN normalization + linear projection.
#[derive(Debug, Clone)]
struct OutputLayer {
    norm_final: RmsNorm,
    linear: candle_nn::Linear,
    adaln_linear: candle_nn::Linear,
}

impl OutputLayer {
    fn new(
        inner_dim: usize,
        out_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let output_dim = patch_size * patch_size * out_channels;
        let norm_final = RmsNorm::new(inner_dim, 1e-6, vb.pp("norm_out"))?;
        let proj_out = linear(inner_dim, output_dim, vb.pp("proj_out"))?;
        let adaln_linear = linear(inner_dim, inner_dim, vb.pp("norm_out").pp("linear"))?;

        Ok(Self {
            norm_final,
            linear: proj_out,
            adaln_linear,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> candle_core::Result<Tensor> {
        let scale = temb.silu()?.apply(&self.adaln_linear)?;
        let scale = (scale + 1.0)?.unsqueeze(1)?;
        let x = self.norm_final.forward(x)?.broadcast_mul(&scale)?;
        x.apply(&self.linear)
    }
}

// ==================== QwenImageTransformer2DModel ====================

/// Qwen-Image Transformer 2D Model.
///
/// Full transformer with:
/// - Timestep embedding (sinusoidal + MLP)
/// - Patch embedding (img_in) and text embedding (txt_in, txt_norm)
/// - 60 dual-stream transformer blocks with joint attention
/// - 3D RoPE positional encoding
/// - Output AdaLN + projection
#[derive(Debug, Clone)]
pub(crate) struct QwenImageTransformer2DModel {
    /// Timestep embedding
    time_embed: TimestepProjEmbeddings,
    /// Patch (image) input projection: patch_dim -> inner_dim
    img_in: candle_nn::Linear,
    /// Text input projection: joint_attention_dim -> joint_attention_dim (identity dim)
    txt_in: candle_nn::Linear,
    /// Text encoder output normalization
    txt_norm: RmsNorm,
    /// Transformer blocks
    blocks: Vec<QwenImageTransformerBlock>,
    /// RoPE embedder for 3D positional encoding
    rope_embedder: RopeEmbedder,
    /// Output layer
    output_layer: OutputLayer,
    /// Configuration
    cfg: QwenImageConfig,
}

impl QwenImageTransformer2DModel {
    pub fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        // Timestep embedding
        let time_embed = TimestepProjEmbeddings::new(cfg.inner_dim, vb.pp("time_text_embed"))?;

        // Patch embedding: in_channels (64) -> inner_dim (3072)
        let img_in = linear(cfg.in_channels, cfg.inner_dim, vb.pp("x_embedder"))?;

        // Text input projection: joint_attention_dim -> joint_attention_dim
        let txt_in = linear(
            cfg.joint_attention_dim,
            cfg.joint_attention_dim,
            vb.pp("context_embedder"),
        )?;

        // Text normalization
        let txt_norm = RmsNorm::new(cfg.joint_attention_dim, cfg.norm_eps, vb.pp("txt_norm"))?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(QwenImageTransformerBlock::new(cfg, vb_blocks.pp(i))?);
        }

        // 3D RoPE embedder
        // axes_lens = axes_dims_rope values used as max sequence lengths per axis
        let axes_lens = vec![2048, 2048, 2048];
        let rope_embedder = RopeEmbedder::new(
            10000.0, // rope_theta
            cfg.axes_dims_rope.clone(),
            axes_lens,
            device,
            dtype,
        )?;

        // Output layer
        let output_layer =
            OutputLayer::new(cfg.inner_dim, cfg.out_channels, cfg.patch_size, vb.clone())?;

        Ok(Self {
            time_embed,
            img_in,
            txt_in,
            txt_norm,
            blocks,
            rope_embedder,
            output_layer,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Latent tensor (B, C, H, W) where C=16
    /// * `t` - Timestep tensor (B,) — raw timestep values from scheduler
    /// * `encoder_hidden_states` - Text encoder output (B, text_len, 3584)
    ///
    /// # Returns
    /// Noise prediction tensor (B, C, H, W)
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let device = x.device();
        let (_b, _c, h, w) = x.dims4()?;
        let patch_size = self.cfg.patch_size;

        // 1. Timestep embedding -> (B, inner_dim)
        let temb = self.time_embed.forward(t)?;

        // 2. Patchify image: (B, C, H, W) -> (B, C, 1, H, W) -> patchify -> (B, num_patches, patch_dim)
        let x_5d = x.unsqueeze(2)?;
        let (x_patches, orig_size) = patchify(&x_5d, patch_size, 1)?;
        let img_hidden = x_patches.apply(&self.img_in)?; // (B, img_seq, inner_dim)

        // 3. Text embedding: norm + project
        let txt_normed = self.txt_norm.forward(encoder_hidden_states)?;
        let txt_mask = encoder_attention_mask
            .to_device(device)?
            .to_dtype(txt_normed.dtype())?;
        let txt_hidden = txt_normed
            .apply(&self.txt_in)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?)?;

        // 4. Create position IDs and RoPE embeddings for image tokens
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let img_pos_ids = create_coordinate_grid((1, h_tokens, w_tokens), (0, 0, 0), device)?;
        let (img_cos, img_sin) = self.rope_embedder.forward(&img_pos_ids)?;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let txt_offset = h_tokens.max(w_tokens) as u32;
        let mut txt_coords = Vec::with_capacity(txt_seq_len * 3);
        for i in 0..txt_seq_len {
            let pos = txt_offset + i as u32;
            txt_coords.push(pos);
            txt_coords.push(pos);
            txt_coords.push(pos);
        }
        let txt_pos_ids = Tensor::from_vec(txt_coords, (txt_seq_len, 3), device)?;
        let (txt_cos, txt_sin) = self.rope_embedder.forward(&txt_pos_ids)?;

        // 5. Process through all transformer blocks
        let mut img = img_hidden;
        let mut txt = txt_hidden;
        for block in &self.blocks {
            let (new_img, new_txt) = block.forward(
                &img,
                &txt,
                encoder_attention_mask,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
            )?;
            img = new_img;
            txt = new_txt;
        }

        // 6. Output layer (image only)
        let img_out = self.output_layer.forward(&img, &temb)?;

        // 7. Unpatchify: (B, num_patches, patch_dim) -> (B, C, 1, H, W) -> squeeze -> (B, C, H, W)
        let x_out = unpatchify(&img_out, orig_size, patch_size, 1, self.cfg.out_channels)?;
        x_out.squeeze(2)
    }
}
