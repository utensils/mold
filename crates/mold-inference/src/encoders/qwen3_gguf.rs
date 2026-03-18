//! Quantized Qwen3-4B encoder loader for GGUF files (llama.cpp standard naming).
//!
//! Implements the Qwen3-4B architecture used as the Z-Image text encoder.
//! Architecture: 36 layers, 32 Q heads, 8 KV heads (GQA 4:1), 2560 hidden, 128 head_dim,
//! SwiGLU MLP (9728 intermediate), RoPE theta=1e6, RMSNorm eps=1e-6.
//! Returns second-to-last layer output (layer 34 of 36), no final norm.
//!
//! GGUF tensor names (llama.cpp standard):
//! - `token_embd.weight`
//! - `blk.{i}.attn_norm.weight`, `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`,
//!   `blk.{i}.attn_v.weight`, `blk.{i}.attn_output.weight`
//! - `blk.{i}.attn_q_norm.weight`, `blk.{i}.attn_k_norm.weight`
//! - `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`,
//!   `blk.{i}.ffn_down.weight`

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ── Qwen3-4B architecture constants ──────────────────────────────────────────

/// Default layer count for Qwen3-4B (read from GGUF metadata if available).
const DEFAULT_N_LAYERS: usize = 36;
const N_HEADS: usize = 32; // Q heads
const N_KV_HEADS: usize = 8; // K/V heads (GQA 4:1)
const HEAD_DIM: usize = 128;
const ROPE_THETA: f64 = 1_000_000.0;
const RMS_NORM_EPS: f64 = 1e-6;
/// Return output after this many layers (second-to-last = 35 layers of 36).
const N_RETURN_LAYERS: usize = 35;
/// GQA repeat factor: each KV head serves this many Q heads.
const KV_REPEAT: usize = N_HEADS / N_KV_HEADS; // 4

// ── RMS Layer Norm ───────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        xs.broadcast_mul(&self.weight).map_err(Into::into)
    }
}

// ── RoPE (Rotary Position Embeddings) ────────────────────────────────────────

fn compute_rope(seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half_dim = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0f32 / (ROPE_THETA as f32).powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;
    let freqs = positions.matmul(&inv_freq)?; // (seq_len, half_dim)
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Apply rotary embeddings to a tensor of shape (batch, heads, seq_len, head_dim).
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq_len, _d) = x.dims4()?;
    let half = HEAD_DIM / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    // cos/sin: (seq_len, half_dim) → (1, 1, seq_len, half_dim)
    let cos = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&out1, &out2], D::Minus1).map_err(Into::into)
}

// ── Causal Attention Mask ────────────────────────────────────────────────────

fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

// ── GQA repeat_kv ────────────────────────────────────────────────────────────

/// Repeat KV heads to match Q head count for grouped-query attention.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, n_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))
        .map_err(Into::into)
}

// ── SwiGLU FFN ───────────────────────────────────────────────────────────────

struct SwiGluFFN {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl SwiGluFFN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_out = candle_nn::Activation::Silu.forward(&self.gate.forward(xs)?)?;
        let up_out = self.up.forward(xs)?;
        self.down.forward(&(gate_out * up_out)?).map_err(Into::into)
    }
}

// ── Qwen3 Self-Attention ─────────────────────────────────────────────────────

struct Qwen3Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Qwen3Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Project Q/K/V
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, seq_len, N_HEADS, HEAD_DIM))?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, seq_len, N_KV_HEADS, HEAD_DIM))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, seq_len, N_KV_HEADS, HEAD_DIM))?;

        // Per-head Q/K norms (applied before RoPE)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to (batch, heads, seq, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Apply RoPE to Q and K
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // GQA: repeat KV heads to match Q head count
        let k = repeat_kv(&k, KV_REPEAT)?;
        let v = repeat_kv(&v, KV_REPEAT)?;

        // Scaled dot-product attention with causal mask
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        // Reshape back: (B, heads, seq, head_dim) → (B, seq, hidden_dim)
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, N_HEADS * HEAD_DIM))?;

        self.o_proj.forward(&attn_output).map_err(Into::into)
    }
}

// ── Qwen3 Encoder Block ─────────────────────────────────────────────────────

struct Qwen3Block {
    attn_norm: RmsNorm,
    self_attn: Qwen3Attention,
    ffn_norm: RmsNorm,
    ffn: SwiGluFFN,
}

impl Qwen3Block {
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with pre-norm and residual
        let normed = self.attn_norm.forward(xs)?;
        let attn_output = self.self_attn.forward(&normed, cos, sin, mask)?;
        let xs = (xs + attn_output)?;

        // FFN with pre-norm and residual
        let normed = self.ffn_norm.forward(&xs)?;
        let ffn_output = self.ffn.forward(&normed)?;
        (xs + ffn_output).map_err(Into::into)
    }
}

// ── GgufQwen3Encoder ─────────────────────────────────────────────────────────

/// Quantized Qwen3-4B encoder loaded from a GGUF file with llama.cpp standard names.
pub(crate) struct GgufQwen3Encoder {
    embedding: candle_nn::Embedding,
    blocks: Vec<Qwen3Block>,
}

impl GgufQwen3Encoder {
    /// Load from a GGUF file.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Load all tensors
        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::new();
        for name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, name, device)?;
            tensors.insert(name.clone(), Arc::new(tensor));
        }

        let get = |name: &str| -> Result<Arc<QTensor>> {
            tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))
        };

        // Embedding (dequantize to float)
        let emb_tensor = get("token_embd.weight")?;
        let emb_weights = emb_tensor.dequantize(device)?;
        let d_model = emb_weights.dim(1)?;
        let embedding = candle_nn::Embedding::new(emb_weights, d_model);

        // Read layer count from metadata, default to 36 (Qwen3-4B)
        let n_layers = content
            .metadata
            .get("qwen3.block_count")
            .or_else(|| content.metadata.get("llama.block_count"))
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(DEFAULT_N_LAYERS);

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("blk.{i}");

            // Q/K/V/O projections
            let q_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_q.weight"))?)?;
            let k_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_k.weight"))?)?;
            let v_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_v.weight"))?)?;
            let o_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_output.weight"))?)?;

            // Per-head Q/K norms (weight shape: head_dim)
            let q_norm_w = get(&format!("{prefix}.attn_q_norm.weight"))?.dequantize(device)?;
            let q_norm = RmsNorm {
                weight: q_norm_w,
                eps: RMS_NORM_EPS,
            };
            let k_norm_w = get(&format!("{prefix}.attn_k_norm.weight"))?.dequantize(device)?;
            let k_norm = RmsNorm {
                weight: k_norm_w,
                eps: RMS_NORM_EPS,
            };

            // Attention + FFN norms
            let attn_norm_w = get(&format!("{prefix}.attn_norm.weight"))?.dequantize(device)?;
            let attn_norm = RmsNorm {
                weight: attn_norm_w,
                eps: RMS_NORM_EPS,
            };
            let ffn_norm_w = get(&format!("{prefix}.ffn_norm.weight"))?.dequantize(device)?;
            let ffn_norm = RmsNorm {
                weight: ffn_norm_w,
                eps: RMS_NORM_EPS,
            };

            // SwiGLU FFN
            let gate = QMatMul::from_weights(get(&format!("{prefix}.ffn_gate.weight"))?)?;
            let up = QMatMul::from_weights(get(&format!("{prefix}.ffn_up.weight"))?)?;
            let down = QMatMul::from_weights(get(&format!("{prefix}.ffn_down.weight"))?)?;

            let self_attn = Qwen3Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
            };

            let ffn = SwiGluFFN { gate, up, down };

            blocks.push(Qwen3Block {
                attn_norm,
                self_attn,
                ffn_norm,
                ffn,
            });
        }

        Ok(Self { embedding, blocks })
    }

    /// Run the Qwen3 encoder forward pass.
    /// Returns the second-to-last layer output (no final norm).
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;

        let mut xs = self.embedding.forward(input_ids)?;

        // Compute RoPE sin/cos for this sequence length
        let (cos, sin) = compute_rope(seq_len, xs.device())?;

        // Compute causal attention mask
        let mask = causal_mask(seq_len, xs.dtype(), xs.device())?;

        // Run through layers, stop at second-to-last (return layer N_RETURN_LAYERS-1)
        let n_run = N_RETURN_LAYERS.min(self.blocks.len());
        for block in self.blocks[..n_run].iter_mut() {
            xs = block.forward(&xs, &cos, &sin, &mask)?;
        }

        Ok(xs)
    }

    /// Run forward pass and collect hidden states from specific layers.
    /// Returns outputs stacked and reshaped: (B, seq_len, num_layers * hidden_size).
    /// Used by Flux.2 Klein which needs layers 9, 18, 27 stacked to 7680-dim.
    pub fn forward_with_layers(
        &mut self,
        input_ids: &Tensor,
        layer_indices: &[usize],
    ) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        let mut xs = self.embedding.forward(input_ids)?;
        let (cos, sin) = compute_rope(seq_len, xs.device())?;
        let mask = causal_mask(seq_len, xs.dtype(), xs.device())?;

        let max_layer = layer_indices.iter().copied().max().unwrap_or(0);
        let n_run = (max_layer + 1).min(self.blocks.len());
        let mut collected: Vec<Tensor> = Vec::with_capacity(layer_indices.len());

        for (i, block) in self.blocks[..n_run].iter_mut().enumerate() {
            xs = block.forward(&xs, &cos, &sin, &mask)?;
            if layer_indices.contains(&i) {
                collected.push(xs.clone());
            }
        }

        // Stack along dim 2 and reshape: (B, num_layers, seq, hidden) → (B, seq, num_layers * hidden)
        let stacked = Tensor::stack(&collected, 1)?;
        let (b, _n, s, h) = stacked.dims4()?;
        Ok(stacked
            .permute((0, 2, 1, 3))?
            .reshape((b, s, collected.len() * h))?)
    }
}
