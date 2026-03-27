//! Native BF16 Qwen3-4B encoder for safetensors weights.
//!
//! Provides `forward_with_layers()` for multi-layer hidden state extraction,
//! required by Flux.2 Klein (layers 9, 18, 27 → 7680-dim stacked embeddings).
//!
//! This replaces the upstream `ZImageTextEncoder` which only returns the
//! penultimate layer and has no multi-layer extraction support.
//!
//! Architecture: 36 layers, 32 Q heads, 8 KV heads (GQA 4:1), 2560 hidden,
//! 128 head_dim, SwiGLU MLP (9728 intermediate), RoPE theta=1e6, RMSNorm eps=1e-6.
//!
//! HuggingFace safetensors weight names:
//! - `model.embed_tokens.weight`
//! - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! - `model.layers.{i}.self_attn.{q,k}_norm.weight`
//! - `model.layers.{i}.input_layernorm.weight`
//! - `model.layers.{i}.post_attention_layernorm.weight`
//! - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

// ── Architecture constants ──────────────────────────────────────────────────

const NUM_HIDDEN_LAYERS: usize = 36;
const NUM_ATTENTION_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const HIDDEN_SIZE: usize = 2560;
const INTERMEDIATE_SIZE: usize = 9728;
const ROPE_THETA: f64 = 1_000_000.0;
const RMS_NORM_EPS: f64 = 1e-6;
const VOCAB_SIZE: usize = 151936;
const MAX_POSITION_EMBEDDINGS: usize = 40960;
const KV_REPEAT: usize = NUM_ATTENTION_HEADS / NUM_KV_HEADS; // 4

// ── Rotary Embedding ────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, device: &Device) -> Result<Self> {
        let dim = HEAD_DIM;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / ROPE_THETA.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, MAX_POSITION_EMBEDDINGS as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_POSITION_EMBEDDINGS, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to q, k tensors of shape (B, H, L, D).
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── GQA repeat_kv ───────────────────────────────────────────────────────────

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, n_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))
        .map_err(Into::into)
}

// ── RmsNorm (per-head variant for Q/K norms) ────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32).map_err(Into::into)
    }
}

// ── SwiGLU MLP ──────────────────────────────────────────────────────────────

struct SwiGluMlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl SwiGluMlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, INTERMEDIATE_SIZE, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&xs.apply(&self.gate_proj)?)?;
        let up = xs.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj).map_err(Into::into)
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: std::sync::Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(rotary_emb: std::sync::Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let q_proj = candle_nn::linear_no_bias(
            HIDDEN_SIZE,
            NUM_ATTENTION_HEADS * HEAD_DIM,
            vb.pp("q_proj"),
        )?;
        let k_proj =
            candle_nn::linear_no_bias(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(
            NUM_ATTENTION_HEADS * HEAD_DIM,
            HIDDEN_SIZE,
            vb.pp("o_proj"),
        )?;
        let q_norm = RmsNorm::new(HEAD_DIM, RMS_NORM_EPS, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(HEAD_DIM, RMS_NORM_EPS, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to (B, L, H, D) then transpose to (B, H, L, D)
        let q = q
            .reshape((b, l, NUM_ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = k.reshape((b, l, NUM_KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;
        let v = v.reshape((b, l, NUM_KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;

        // Per-head RMSNorm (flatten batch+heads, norm, reshape back)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, NUM_ATTENTION_HEADS, l, HEAD_DIM))?;
        let k = k_flat.reshape((b, NUM_KV_HEADS, l, HEAD_DIM))?;

        // RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, 0)?;

        // GQA repeat KV
        let k = repeat_kv(k, KV_REPEAT)?.contiguous()?;
        let v = repeat_kv(v, KV_REPEAT)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn_weights.matmul(&v)?;

        // Output projection
        ctx.transpose(1, 2)?
            .reshape((b, l, NUM_ATTENTION_HEADS * HEAD_DIM))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: SwiGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: std::sync::Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, vb.pp("self_attn"))?;
        let mlp = SwiGluMlp::new(vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(HIDDEN_SIZE, RMS_NORM_EPS, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::new(HIDDEN_SIZE, RMS_NORM_EPS, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, mask)?;
        let xs = (xs + h)?;
        let h = self.post_attention_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        (xs + h).map_err(Into::into)
    }
}

// ── Bf16Qwen3Encoder ───────────────────────────────────────────────────────

/// Native BF16 Qwen3-4B encoder with multi-layer hidden state extraction.
pub(crate) struct Bf16Qwen3Encoder {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    device: Device,
    dtype: DType,
}

impl Bf16Qwen3Encoder {
    /// Load from HuggingFace safetensors files.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(VOCAB_SIZE, HIDDEN_SIZE, vb_model.pp("embed_tokens"))?;

        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(dtype, &device)?);

        let vb_layers = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        for i in 0..NUM_HIDDEN_LAYERS {
            layers.push(DecoderLayer::new(rotary_emb.clone(), vb_layers.pp(i))?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            device,
            dtype,
        })
    }

    /// Create causal attention mask.
    fn causal_mask(&self, b: usize, tgt: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| (0..tgt).map(move |j| if j <= i { 0.0 } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt), &self.device)?
            .to_dtype(self.dtype)
            .map_err(Into::into)
    }

    /// Encode text, returning second-to-last layer hidden states (no final norm).
    /// Compatible with the upstream `ZImageTextEncoder::forward` behavior.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l)?)
        };

        let target_layer = NUM_HIDDEN_LAYERS - 2; // layer 34

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
            if i == target_layer {
                return Ok(hidden_states);
            }
        }

        Ok(hidden_states)
    }

    /// Run forward pass and collect hidden states from specific layers.
    /// Returns (B, seq_len, num_layers * hidden_size).
    /// Used by Flux.2 Klein which stacks layers 9, 18, 27 → 7680-dim embeddings.
    pub fn forward_with_layers(
        &self,
        input_ids: &Tensor,
        layer_indices: &[usize],
    ) -> Result<Tensor> {
        if layer_indices.is_empty() {
            anyhow::bail!("layer_indices must not be empty");
        }
        let max_layer = layer_indices.iter().copied().max().unwrap_or(0);
        if max_layer >= self.layers.len() {
            anyhow::bail!(
                "layer index {max_layer} out of bounds (model has {} layers)",
                self.layers.len()
            );
        }

        let (b, l) = input_ids.dims2()?;
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l)?)
        };

        let n_run = max_layer + 1;
        let mut collected: Vec<Tensor> = Vec::with_capacity(layer_indices.len());

        for (i, layer) in self.layers[..n_run].iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
            if layer_indices.contains(&i) {
                collected.push(hidden_states.clone());
            }
        }

        // Stack: (B, num_layers, seq, hidden) → permute → (B, seq, num_layers * hidden)
        let stacked = Tensor::stack(&collected, 1)?;
        let (b, _n, s, h) = stacked.dims4()?;
        Ok(stacked
            .permute((0, 2, 1, 3))?
            .reshape((b, s, collected.len() * h))?)
    }
}
