//! Quantized (GGUF) Z-Image Transformer
//!
//! Mirrors `candle_transformers::models::z_image::transformer::ZImageTransformer2DModel`
//! but uses quantized layer types from `candle_transformers::quantized_nn`.
//!
//! Key differences from the BF16 version:
//! - Uses `quantized_nn::Linear` / `quantized_nn::RmsNorm` (dequantize on forward)
//! - Fused QKV projection (single `[dim, 3*dim]` weight instead of separate Q/K/V)
//! - Different tensor naming (GGUF convention vs safetensors convention)

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::RmsNorm as CandleRmsNorm;
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, Config, LayerNormNoParams,
    RopeEmbedder, ADALN_EMBED_DIM, FREQUENCY_EMBEDDING_SIZE, MAX_PERIOD,
};
use candle_transformers::quantized_nn::{self, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;

// ==================== TimestepEmbedder ====================

struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn new(out_size: usize, mid_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 =
            quantized_nn::linear(FREQUENCY_EMBEDDING_SIZE, mid_size, vb.pp("mlp").pp("0"))?;
        let linear2 = quantized_nn::linear(mid_size, out_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn timestep_embedding(&self, t: &Tensor, device: &Device) -> Result<Tensor> {
        let half = self.frequency_embedding_size / 2;
        let freqs = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        // Keep F32 — quantized linears dequantize to F32 internally
        Ok(embedding)
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let t_freq = self.timestep_embedding(t, device)?;
        t_freq.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

// ==================== FeedForward (SwiGLU) ====================

struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = quantized_nn::linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = quantized_nn::linear_no_bias(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = quantized_nn::linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = x.apply(&self.w1)?.silu()?;
        let x3 = x.apply(&self.w3)?;
        (x1 * x3)?.apply(&self.w2)
    }
}

// ==================== QkNorm ====================

struct QkNorm {
    norm_q: CandleRmsNorm,
    norm_k: CandleRmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // GGUF names: q_norm.weight, k_norm.weight
        let norm_q_w = vb.get(head_dim, "q_norm.weight")?.dequantize(vb.device())?;
        let norm_q = CandleRmsNorm::new(norm_q_w, eps);
        let norm_k_w = vb.get(head_dim, "k_norm.weight")?.dequantize(vb.device())?;
        let norm_k = CandleRmsNorm::new(norm_k_w, eps);
        Ok(Self { norm_q, norm_k })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q = self.norm_q.forward(q)?;
        let k = self.norm_k.forward(k)?;
        Ok((q, k))
    }
}

// ==================== ZImageAttention ====================

/// Z-Image attention with fused QKV (GGUF layout), QK normalization, and 3D RoPE.
struct ZImageAttention {
    qkv: Linear,
    out: Linear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
    use_accelerated_attn: bool,
}

impl ZImageAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        // GGUF: fused QKV [dim, 3*n_heads*head_dim]
        let qkv = quantized_nn::linear_no_bias(dim, 3 * n_heads * head_dim, vb.pp("qkv"))?;
        // GGUF: "out" instead of "to_out.0"
        let out = quantized_nn::linear_no_bias(n_heads * head_dim, dim, vb.pp("out"))?;

        let qk_norm = if cfg.qk_norm {
            Some(QkNorm::new(head_dim, 1e-5, vb.clone())?)
        } else {
            None
        };

        Ok(Self {
            qkv,
            out,
            qk_norm,
            n_heads,
            head_dim,
            use_accelerated_attn: cfg.use_accelerated_attn,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        // Fused QKV projection → split into Q, K, V
        let qkv = hidden_states.apply(&self.qkv)?;
        let qkv = qkv.reshape((b, seq_len, 3, self.n_heads, self.head_dim))?;
        let q = qkv.i((.., .., 0))?;
        let k = qkv.i((.., .., 1))?;
        let v = qkv.i((.., .., 2))?;

        // Apply QK norm
        let (q, k) = if let Some(ref norm) = self.qk_norm {
            norm.forward(&q, &k)?
        } else {
            (q, k)
        };

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // Transpose for attention: (B, n_heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let device = hidden_states.device();

        let context = self.attention_dispatch(&q, &k, &v, attention_mask, scale, device)?;

        // Reshape back: (B, n_heads, seq_len, head_dim) → (B, seq_len, dim)
        let context = context.transpose(1, 2)?.reshape((b, seq_len, ()))?;

        context.apply(&self.out)
    }

    fn attention_dispatch(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
        device: &Device,
    ) -> Result<Tensor> {
        if !self.use_accelerated_attn {
            return self.attention_basic(q, k, v, mask, scale);
        }
        if device.is_cuda() {
            self.attention_cuda(q, k, v, mask, scale)
        } else if device.is_metal() {
            self.attention_metal(q, k, v, mask, scale)
        } else {
            self.attention_basic(q, k, v, mask, scale)
        }
    }

    fn attention_cuda(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        // Quantized models use basic attention on CUDA (no flash-attn support)
        self.attention_basic(q, k, v, mask, scale)
    }

    fn attention_metal(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let sdpa_mask = self.prepare_sdpa_mask(mask, q)?;
        candle_nn::ops::sdpa(q, k, v, sdpa_mask.as_ref(), false, scale as f32, 1.0)
    }

    fn attention_basic(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            let m = m.unsqueeze(1)?.unsqueeze(2)?;
            let m = m.to_dtype(attn_weights.dtype())?;
            let m = ((m - 1.0)? * 1e9)?;
            attn_weights = attn_weights.broadcast_add(&m)?;
        }
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_probs.matmul(v)
    }

    fn prepare_sdpa_mask(&self, mask: Option<&Tensor>, q: &Tensor) -> Result<Option<Tensor>> {
        match mask {
            Some(m) => {
                let (b, _, seq_len, _) = q.dims4()?;
                let m = m.unsqueeze(1)?.unsqueeze(2)?;
                let m = m.to_dtype(q.dtype())?;
                let m = ((m - 1.0)? * 1e9)?;
                let m = m.broadcast_as((b, self.n_heads, seq_len, seq_len))?;
                Ok(Some(m))
            }
            None => Ok(None),
        }
    }
}

// ==================== ZImageTransformerBlock ====================

struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: quantized_nn::RmsNorm,
    attention_norm2: quantized_nn::RmsNorm,
    ffn_norm1: quantized_nn::RmsNorm,
    ffn_norm2: quantized_nn::RmsNorm,
    adaln_modulation: Option<Linear>,
}

impl ZImageTransformerBlock {
    fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();

        let attention = ZImageAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?;

        let attention_norm1 =
            quantized_nn::RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm1"))?;
        let attention_norm2 =
            quantized_nn::RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm2"))?;
        let ffn_norm1 = quantized_nn::RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?;
        let ffn_norm2 = quantized_nn::RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?;

        let adaln_modulation = if modulation {
            let adaln_dim = dim.min(ADALN_EMBED_DIM);
            Some(quantized_nn::linear(
                adaln_dim,
                4 * dim,
                vb.pp("adaLN_modulation").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            attention,
            feed_forward,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln_modulation,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(ref adaln) = self.adaln_modulation {
            let adaln_input = adaln_input.expect("adaln_input required when modulation=true");
            let modulation = adaln_input.apply(adaln)?.unsqueeze(1)?;
            let chunks = modulation.chunk(4, D::Minus1)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
                (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

            let gate_msa = gate_msa.tanh()?;
            let gate_mlp = gate_mlp.tanh()?;
            let scale_msa = (scale_msa + 1.0)?;
            let scale_mlp = (scale_mlp + 1.0)?;

            // Attention block
            let normed = self.attention_norm1.forward(x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward(&scaled, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            // FFN block
            let normed = self.ffn_norm1.forward(&x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + gate_mlp.broadcast_mul(&ffn_out)?
        } else {
            // Without modulation
            let normed = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + attn_out)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + ffn_out
        }
    }
}

// ==================== FinalLayer ====================

struct FinalLayer {
    norm_final: LayerNormNoParams,
    linear: Linear,
    adaln_silu: Linear,
}

impl FinalLayer {
    fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = LayerNormNoParams::new(1e-6);
        let linear = quantized_nn::linear(hidden_size, out_channels, vb.pp("linear"))?;
        let adaln_dim = hidden_size.min(ADALN_EMBED_DIM);
        let adaln_silu =
            quantized_nn::linear(adaln_dim, hidden_size, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self {
            norm_final,
            linear,
            adaln_silu,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = c.silu()?.apply(&self.adaln_silu)?;
        let scale = (scale + 1.0)?.unsqueeze(1)?;
        let x = self.norm_final.forward(x)?.broadcast_mul(&scale)?;
        x.apply(&self.linear)
    }
}

// ==================== QuantizedZImageTransformer2DModel ====================

/// Quantized (GGUF) Z-Image Transformer, matching the BF16 forward signature.
pub struct QuantizedZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: quantized_nn::RmsNorm,
    cap_embedder_linear: Linear,
    x_embedder: Linear,
    final_layer: FinalLayer,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

impl QuantizedZImageTransformer2DModel {
    pub fn new(cfg: &Config, _dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        // TimestepEmbedder
        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"))?;

        // Caption embedder — GGUF: cap_embedder.0 (RmsNorm), cap_embedder.1 (Linear)
        let cap_embedder_norm = quantized_nn::RmsNorm::new(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear =
            quantized_nn::linear(cfg.cap_feat_dim, cfg.dim, vb.pp("cap_embedder").pp("1"))?;

        // Patch embedder — GGUF: x_embedder (not all_x_embedder.2-1)
        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder = quantized_nn::linear(patch_dim, cfg.dim, vb.pp("x_embedder"))?;

        // Final layer — GGUF: final_layer (not all_final_layer.2-1)
        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer = FinalLayer::new(cfg.dim, out_channels, vb.pp("final_layer"))?;

        // Noise refiner (with modulation)
        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("noise_refiner").pp(i),
            )?);
        }

        // Context refiner (without modulation)
        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                cfg,
                false,
                vb.pp("context_refiner").pp(i),
            )?);
        }

        // Main layers (with modulation)
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("layers").pp(i),
            )?);
        }

        // RoPE embedder (no weights — pure computation)
        // Use F32 for quantized path since all internal computation is F32
        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            &device,
            DType::F32,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass — identical signature to `ZImageTransformer2DModel::forward()`.
    ///
    /// All computation happens in F32 internally (quantized layers dequantize to F32).
    /// The output is cast back to the input dtype.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let out_dtype = x.dtype();
        let device = x.device();

        // Cast inputs to F32 for quantized layers
        let x = x.to_dtype(DType::F32)?;
        let t = t.to_dtype(DType::F32)?;
        let cap_feats = cap_feats.to_dtype(DType::F32)?;

        let (b, _c, f, h, w) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];

        // 1. Timestep embedding
        let t_scaled = (&t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;

        // 2. Patchify and embed image
        let (x_patches, orig_size) = patchify(&x, patch_size, f_patch_size)?;
        let mut x = x_patches.apply(&self.x_embedder)?;
        let img_seq_len = x.dim(1)?;

        // 3. Create image position IDs
        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let text_len = cap_feats.dim(1)?;

        let x_pos_ids =
            create_coordinate_grid((f_tokens, h_tokens, w_tokens), (text_len + 1, 0, 0), device)?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;

        // 4. Caption embedding
        let cap_normed = self.cap_embedder_norm.forward(&cap_feats)?;
        let mut cap = cap_normed.apply(&self.cap_embedder_linear)?;

        // 5. Create caption position IDs
        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        // 6. Create attention masks
        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, device)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;

        // 7. Noise refiner (process image with modulation)
        for layer in &self.noise_refiner {
            x = layer.forward(&x, Some(&x_attn_mask), &x_cos, &x_sin, Some(&adaln_input))?;
        }

        // 8. Context refiner (process text without modulation)
        for layer in &self.context_refiner {
            cap = layer.forward(&cap, Some(&cap_attn_mask), &cap_cos, &cap_sin, None)?;
        }

        // 9. Concatenate image and text
        let unified = Tensor::cat(&[&x, &cap], 1)?;

        // 10. Create unified position IDs and attention mask
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        // 11. Main transformer layers
        let mut unified = unified;
        for layer in &self.layers {
            unified = layer.forward(
                &unified,
                Some(&unified_attn_mask),
                &unified_cos,
                &unified_sin,
                Some(&adaln_input),
            )?;
        }

        // 12. Final layer (only on image portion)
        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;

        // 13. Unpatchify and cast back to original dtype
        let result = unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            self.cfg.in_channels,
        )?;
        result.to_dtype(out_dtype)
    }
}
