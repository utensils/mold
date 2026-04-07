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
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::RmsNorm;
use candle_transformers::models::z_image::transformer::apply_rotary_emb;

use super::quantized_transformer::QwenRopeEmbedder;

// ==================== FP8 Linear (per-layer dequant) ====================

/// Linear layer supporting both standard BF16 and FP8 with per-layer dequantization.
///
/// For FP8 models, weights stay as F8E4M3 in VRAM (~1 byte/param). On each
/// forward call, the weight is cast to the activation dtype (BF16), optionally
/// multiplied by a scale factor, used for matmul, and the transient BF16 copy
/// is immediately freed. This matches ComfyUI's "manual_cast" FP8 inference.
#[derive(Debug, Clone)]
enum QwenLinear {
    Standard(candle_nn::Linear),
    Fp8 {
        weight: Tensor,
        scale: Option<Tensor>,
        bias: Option<Tensor>,
    },
}

impl QwenLinear {
    /// Load a linear layer, auto-detecting FP8 vs standard from weight dtype.
    fn load(
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        if weight.dtype() == DType::F8E4M3 {
            let scale = vb.get_unchecked("scale_weight").ok();
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
}

impl Module for QwenLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(x),
            Self::Fp8 {
                weight,
                scale,
                bias,
            } => {
                let dtype = x.dtype();
                let w = weight.to_dtype(dtype)?;
                let w = match scale {
                    Some(s) => w.broadcast_mul(&s.to_dtype(dtype)?)?,
                    None => w,
                };
                // Handle multi-dim inputs like nn::Linear (reshape → matmul → reshape back)
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
                match bias {
                    Some(b) => out.broadcast_add(&b.to_dtype(dtype)?),
                    None => Ok(out),
                }
            }
        }
    }
}

// ==================== Feed Forward ====================

/// Feed-forward network supporting both official (SwiGLU w1/w2/w3) and
/// ComfyUI/diffusers (GELU net.0.proj + net.2) tensor naming.
/// Uses `QwenLinear` for FP8-aware per-layer dequantization.
#[derive(Debug, Clone)]
enum FeedForward {
    SwiGlu {
        w1: QwenLinear,
        w2: QwenLinear,
        w3: QwenLinear,
    },
    Gelu {
        proj: QwenLinear,
        out: QwenLinear,
    },
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        if vb.contains_tensor("net.0.proj.weight") {
            let has_bias = vb.contains_tensor("net.0.proj.bias");
            let proj =
                QwenLinear::load(dim, hidden_dim, has_bias, vb.pp("net").pp("0").pp("proj"))?;
            let out = QwenLinear::load(hidden_dim, dim, has_bias, vb.pp("net").pp("2"))?;
            Ok(Self::Gelu { proj, out })
        } else {
            let w1 = QwenLinear::load(dim, hidden_dim, false, vb.pp("w1"))?;
            let w2 = QwenLinear::load(hidden_dim, dim, false, vb.pp("w2"))?;
            let w3 = QwenLinear::load(dim, hidden_dim, false, vb.pp("w3"))?;
            Ok(Self::SwiGlu { w1, w2, w3 })
        }
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::SwiGlu { w1, w2, w3 } => {
                let gate = w1.forward(x)?.silu()?;
                let x = (gate * w3.forward(x)?)?;
                w2.forward(&x)
            }
            Self::Gelu { proj, out } => {
                let x = proj
                    .forward(x)?
                    .apply(&candle_nn::Activation::GeluPytorchTanh)?;
                out.forward(&x)
            }
        }
    }
}

// ==================== Layer Norm (No Params) ====================

/// Standard LayerNorm without learnable parameters.
///
/// Matches `nn.LayerNorm(dim, elementwise_affine=False)` in PyTorch.
/// The Qwen-Image model uses this for block-level normalization, with
/// scale/shift provided externally via AdaLN modulation.
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
    linear1: QwenLinear,
    linear2: QwenLinear,
    frequency_embedding_size: usize,
}

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
pub(crate) const MAX_PERIOD: f64 = 10000.0;

impl TimestepProjEmbeddings {
    fn new(inner_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let vb = if vb.contains_tensor("timestep_embedder.linear_1.weight") {
            vb.pp("timestep_embedder")
        } else {
            vb
        };
        let has_bias = vb.contains_tensor("linear_1.bias");
        let linear1 = QwenLinear::load(
            FREQUENCY_EMBEDDING_SIZE,
            inner_dim,
            has_bias,
            vb.pp("linear_1"),
        )?;
        let linear2 = QwenLinear::load(inner_dim, inner_dim, has_bias, vb.pp("linear_2"))?;
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

    fn forward(&self, t: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        let device = t.device();
        let t_freq = self.timestep_embedding(t, device, dtype)?;
        self.linear1
            .forward(&t_freq)?
            .silu()
            .and_then(|x| self.linear2.forward(&x))
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
    to_q: QwenLinear,
    to_k: QwenLinear,
    to_v: QwenLinear,
    to_out: QwenLinear,
    add_q_proj: QwenLinear,
    add_k_proj: QwenLinear,
    add_v_proj: QwenLinear,
    add_out_proj: QwenLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
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

        let has_bias = vb.contains_tensor("to_q.bias");
        let to_q = QwenLinear::load(dim, qkv_dim, has_bias, vb.pp("to_q"))?;
        let to_k = QwenLinear::load(dim, qkv_dim, has_bias, vb.pp("to_k"))?;
        let to_v = QwenLinear::load(dim, qkv_dim, has_bias, vb.pp("to_v"))?;
        let to_out_key = if vb.contains_tensor("to_out.0.weight") {
            "to_out.0"
        } else {
            "to_out_0"
        };
        let to_out = QwenLinear::load(qkv_dim, dim, has_bias, vb.pp(to_out_key))?;

        let add_q_proj = QwenLinear::load(text_dim, qkv_dim, has_bias, vb.pp("add_q_proj"))?;
        let add_k_proj = QwenLinear::load(text_dim, qkv_dim, has_bias, vb.pp("add_k_proj"))?;
        let add_v_proj = QwenLinear::load(text_dim, qkv_dim, has_bias, vb.pp("add_v_proj"))?;
        let add_out_proj = QwenLinear::load(qkv_dim, text_dim, has_bias, vb.pp("to_add_out"))?;

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

        // Concatenate in [text, image] order (matches diffusers QwenDoubleStreamAttnProcessor2_0)
        let q = Tensor::cat(&[&q_txt, &q_img], 1)?;
        let k = Tensor::cat(&[&k_txt, &k_img], 1)?;
        let v = Tensor::cat(&[&v_txt, &v_img], 1)?;

        // Transpose to (B, heads, seq, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let img_mask = Tensor::ones((b, img_seq_len), DType::U8, q.device())?;
        // Key mask order: [text, image] to match concatenation
        let key_mask = Tensor::cat(&[txt_mask, &img_mask], 1)?
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

        // Split in [text, image] order
        let txt_attn = attn.narrow(1, 0, txt_seq_len)?;
        let img_attn = attn.narrow(1, txt_seq_len, img_seq_len)?;

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
/// - Separate AdaLN norms for image and text streams (parameterless LayerNorm)
/// - Joint attention across both streams
/// - Separate feedforward networks for image and text
/// - AdaLN modulation from timestep embedding: 6 params per stream (shift, scale, gate × 2)
#[derive(Debug, Clone)]
struct QwenImageTransformerBlock {
    // Image stream norms (no learnable params — scale/shift from AdaLN modulation)
    norm1: LayerNormNoParams,
    norm1_context: LayerNormNoParams,
    // Joint attention
    attn: JointAttention,
    // Feedforward
    ff: FeedForward,
    ff_context: FeedForward,
    // Post-attention norms (no learnable params)
    norm2: LayerNormNoParams,
    norm2_context: LayerNormNoParams,
    // AdaLN modulation: 6 values (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    adaln_modulation: QwenLinear,
    // AdaLN modulation for text stream
    adaln_context_modulation: QwenLinear,
}

impl QwenImageTransformerBlock {
    fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = cfg.inner_dim;
        let text_dim = cfg.joint_attention_dim;
        let is_comfyui = vb.contains_tensor("img_mlp.net.0.proj.weight");
        // FP8/ComfyUI uses 4x expansion; BF16 official uses int(dim/3)*8.
        let hidden_dim = if is_comfyui {
            dim * 4
        } else {
            cfg.hidden_dim()
        };

        // Block norms: parameterless LayerNorm (scale/shift come from AdaLN modulation)
        let norm1 = LayerNormNoParams::new(cfg.norm_eps);
        let norm1_context = LayerNormNoParams::new(cfg.norm_eps);

        let attn = JointAttention::new(cfg, vb.pp("attn"))?;

        let ff_key = if is_comfyui { "img_mlp" } else { "ff" };
        let ff_ctx_key = if is_comfyui { "txt_mlp" } else { "ff_context" };
        let ff = FeedForward::new(dim, hidden_dim, vb.pp(ff_key))?;
        let ff_context = FeedForward::new(text_dim, text_dim * 4, vb.pp(ff_ctx_key))?;

        let norm2 = LayerNormNoParams::new(cfg.norm_eps);
        let norm2_context = LayerNormNoParams::new(cfg.norm_eps);

        // AdaLN: 6 modulation values per stream (shift, scale, gate for attention + MLP)
        // FP8/ComfyUI uses "img_mod.1"/"txt_mod.1"; BF16 official uses "norm1.linear"/"norm1_context.linear".
        let has_bias =
            vb.contains_tensor("img_mod.1.bias") || vb.contains_tensor("norm1.linear.bias");
        let (adaln_modulation, adaln_context_modulation) = if vb.contains_tensor("img_mod.1.weight")
        {
            (
                QwenLinear::load(dim, 6 * dim, has_bias, vb.pp("img_mod").pp("1"))?,
                QwenLinear::load(dim, 6 * text_dim, has_bias, vb.pp("txt_mod").pp("1"))?,
            )
        } else {
            (
                QwenLinear::load(dim, 6 * dim, has_bias, vb.pp("norm1").pp("linear"))?,
                QwenLinear::load(
                    dim,
                    6 * text_dim,
                    has_bias,
                    vb.pp("norm1_context").pp("linear"),
                )?,
            )
        };

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

        // --- AdaLN modulation (6 params: shift, scale, gate for attention + MLP) ---
        let img_mod = temb.silu()?.apply(&self.adaln_modulation)?.unsqueeze(1)?;
        let img_chunks = img_mod.chunk(6, D::Minus1)?;
        let (
            img_shift_msa,
            img_scale_msa,
            img_gate_msa,
            img_shift_mlp,
            img_scale_mlp,
            img_gate_mlp,
        ) = (
            &img_chunks[0],
            &img_chunks[1],
            &img_chunks[2],
            &img_chunks[3],
            &img_chunks[4],
            &img_chunks[5],
        );

        let txt_mod = temb
            .silu()?
            .apply(&self.adaln_context_modulation)?
            .unsqueeze(1)?;
        let txt_chunks = txt_mod.chunk(6, D::Minus1)?;
        let (
            txt_shift_msa,
            txt_scale_msa,
            txt_gate_msa,
            txt_shift_mlp,
            txt_scale_mlp,
            txt_gate_mlp,
        ) = (
            &txt_chunks[0],
            &txt_chunks[1],
            &txt_chunks[2],
            &txt_chunks[3],
            &txt_chunks[4],
            &txt_chunks[5],
        );

        // --- Attention ---
        // Image: norm + scale + shift
        let img_attn_in = self
            .norm1
            .forward(img_hidden)?
            .broadcast_mul(&(img_scale_msa + 1.0)?)?
            .broadcast_add(img_shift_msa)?;

        // Text: norm + scale + shift
        let txt_attn_in = self
            .norm1_context
            .forward(txt_hidden)?
            .broadcast_mul(&(txt_scale_msa + 1.0)?)?
            .broadcast_add(txt_shift_msa)?;

        // Joint attention
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

        // Gate + residual (no tanh on gate)
        let img_hidden = (img_hidden + img_gate_msa.broadcast_mul(&img_attn)?)?;
        let txt_dtype = txt_hidden.dtype();
        let txt_hidden = (txt_hidden + txt_gate_msa.broadcast_mul(&txt_attn)?)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?.to_dtype(txt_dtype)?)?;

        // --- Feedforward ---
        // Image: norm + scale + shift + FF + gate + residual
        let img_mlp_in = self
            .norm2
            .forward(&img_hidden)?
            .broadcast_mul(&(img_scale_mlp + 1.0)?)?
            .broadcast_add(img_shift_mlp)?;
        let img_ff = self.ff.forward(&img_mlp_in)?;
        let img_hidden = (img_hidden + img_gate_mlp.broadcast_mul(&img_ff)?)?;

        // Text: norm + scale + shift + FF + gate + residual
        let txt_mlp_in = self
            .norm2_context
            .forward(&txt_hidden)?
            .broadcast_mul(&(txt_scale_mlp + 1.0)?)?
            .broadcast_add(txt_shift_mlp)?;
        let txt_ff = self.ff_context.forward(&txt_mlp_in)?;
        let txt_dtype = txt_hidden.dtype();
        let txt_hidden = (txt_hidden + txt_gate_mlp.broadcast_mul(&txt_ff)?)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?.to_dtype(txt_dtype)?)?;

        Ok((img_hidden, txt_hidden))
    }
}

// ==================== Output Layer ====================

/// Final output layer: AdaLN normalization (shift + scale) + linear projection.
#[derive(Debug, Clone)]
struct OutputLayer {
    norm_final: LayerNormNoParams,
    linear: QwenLinear,
    adaln_linear: QwenLinear,
}

impl OutputLayer {
    fn new(
        inner_dim: usize,
        out_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let output_dim = patch_size * patch_size * out_channels;
        let norm_final = LayerNormNoParams::new(1e-6);
        let has_bias = vb.contains_tensor("proj_out.bias");
        let proj_out = QwenLinear::load(inner_dim, output_dim, has_bias, vb.pp("proj_out"))?;
        let adaln_linear = QwenLinear::load(
            inner_dim,
            2 * inner_dim,
            has_bias,
            vb.pp("norm_out").pp("linear"),
        )?;

        Ok(Self {
            norm_final,
            linear: proj_out,
            adaln_linear,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> candle_core::Result<Tensor> {
        let mod_params = temb.silu()?.apply(&self.adaln_linear)?;
        let chunks = mod_params.chunk(2, D::Minus1)?;
        // AdaLayerNormContinuous: scale = chunk[0], shift = chunk[1]
        // (opposite of block-level modulation which uses shift, scale, gate order)
        let scale = chunks[0].unsqueeze(1)?;
        let shift = chunks[1].unsqueeze(1)?;
        let x = self
            .norm_final
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
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
    img_in: QwenLinear,
    /// Text input projection: joint_attention_dim -> joint_attention_dim (identity dim)
    txt_in: QwenLinear,
    /// Text encoder output normalization
    txt_norm: RmsNorm,
    /// Transformer blocks
    blocks: Vec<QwenImageTransformerBlock>,
    /// RoPE embedder for 3D positional encoding (centered positions, scale_rope=True)
    rope_embedder: QwenRopeEmbedder,
    /// Output layer
    output_layer: OutputLayer,
    /// Configuration
    cfg: QwenImageConfig,
}

impl QwenImageTransformer2DModel {
    pub fn new(cfg: &QwenImageConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        // Detect FP8/ComfyUI format vs BF16 official format.
        // FP8 projects text from 3584 → 3072 (inner_dim) and uses inner_dim throughout blocks.
        // BF16 keeps text at 3584 (joint_attention_dim) throughout.
        let is_comfyui = vb.contains_tensor("img_in.weight");
        let block_text_dim = if is_comfyui {
            cfg.inner_dim
        } else {
            cfg.joint_attention_dim
        };

        // Timestep embedding
        let time_embed = TimestepProjEmbeddings::new(cfg.inner_dim, vb.pp("time_text_embed"))?;

        // Patch embedding: in_channels (64) -> inner_dim (3072)
        let img_in_key = if is_comfyui { "img_in" } else { "x_embedder" };
        let has_stem_bias = vb.contains_tensor(&format!("{img_in_key}.bias"));
        let img_in = QwenLinear::load(
            cfg.in_channels,
            cfg.inner_dim,
            has_stem_bias,
            vb.pp(img_in_key),
        )?;

        // Text input projection.
        let (txt_in_key, txt_in_in) = if is_comfyui {
            ("txt_in", cfg.joint_attention_dim) // 3584 → 3072
        } else {
            ("context_embedder", cfg.joint_attention_dim) // 3584 → 3584
        };
        let txt_in = QwenLinear::load(txt_in_in, block_text_dim, has_stem_bias, vb.pp(txt_in_key))?;

        // Text normalization
        let txt_norm = RmsNorm::new(cfg.joint_attention_dim, cfg.norm_eps, vb.pp("txt_norm"))?;

        // Transformer blocks
        // For blocks, use block_text_dim as the effective joint_attention_dim.
        let mut block_cfg = cfg.clone();
        block_cfg.joint_attention_dim = block_text_dim;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            blocks.push(QwenImageTransformerBlock::new(&block_cfg, vb_blocks.pp(i))?);
        }

        // 3D RoPE embedder with Qwen centered positions (scale_rope=True).
        // Uses negative+positive frequency tables for height/width (centered),
        // positive-only for temporal axis.
        let rope_embedder =
            QwenRopeEmbedder::new(10000.0, cfg.axes_dims_rope.clone(), device, dtype)?;

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
    /// * `t` - Timestep tensor (B,) — Qwen pre-scaled sigma values (`sigma * 1000`)
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
        let temb = self
            .time_embed
            .forward(t, crate::engine::gpu_dtype(device))?;

        // 2. Pack latents like diffusers `_pack_latents`:
        //    (B, C, H, W) -> (B, (H/p)*(W/p), C*p*p)
        let hp = h / patch_size;
        let wp = w / patch_size;
        let x_packed = x
            .reshape((_b, _c, hp, patch_size, wp, patch_size))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((_b, hp * wp, _c * patch_size * patch_size))?
            .contiguous()?;
        let img_hidden = x_packed.apply(&self.img_in)?;

        // 3. Text embedding: norm + project
        let txt_normed = self.txt_norm.forward(encoder_hidden_states)?;
        let txt_mask = encoder_attention_mask
            .to_device(device)?
            .to_dtype(txt_normed.dtype())?;
        let txt_hidden = txt_normed
            .apply(&self.txt_in)?
            .broadcast_mul(&txt_mask.unsqueeze(D::Minus1)?)?;

        // 4. RoPE embeddings: centered positions for image (scale_rope=True),
        //    offset-based for text.
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope_embedder
                .forward(1, h_tokens, w_tokens, txt_seq_len, device)?;

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

        // 7. Unpack latents like diffusers `_unpack_latents`:
        //    (B, (H/p)*(W/p), out_channels*p*p) -> (B, out_channels, H, W)
        let x_out = img_out
            .reshape((_b, hp, wp, self.cfg.out_channels, patch_size, patch_size))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((_b, self.cfg.out_channels, h, w))?
            .contiguous()?;
        Ok(x_out)
    }
}
