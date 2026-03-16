//! Custom T5 encoder loader for GGUF files using the GGUF standard tensor naming convention.
//!
//! The city96 T5 GGUF files use GGUF standard names (e.g. `enc.blk.0.attn_q.weight`,
//! `token_embd.weight`) rather than PyTorch-style names that candle's `quantized_t5`
//! expects. This module loads the GGUF tensors by their standard names and assembles
//! the T5 encoder model manually.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::Activation;
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// T5 RMS layer norm (no bias, no mean subtraction).
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

/// Gated FFN: gate(x) * up(x) → down
struct T5GatedFFN {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    act: Activation,
}

impl T5GatedFFN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden_gelu = self.act.forward(&self.gate.forward(xs)?)?;
        let hidden_linear = self.up.forward(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        self.down.forward(&xs).map_err(Into::into)
    }
}

/// T5 self-attention (encoder only, no KV cache, no causal mask).
struct T5SelfAttention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<candle_nn::Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
}

impl T5SelfAttention {
    fn forward(&mut self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self
            .q
            .forward(xs)?
            .reshape((b, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, seq_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;

        let scores = q.matmul(&k.t()?)?;

        let (scores, position_bias) = match position_bias {
            Some(pb) => (scores.broadcast_add(pb)?, pb.clone()),
            None => match &self.relative_attention_bias {
                None => {
                    return Err(anyhow::anyhow!(
                        "no relative attention bias and none provided"
                    ));
                }
                Some(rel_attn_bias) => {
                    let pb = self.compute_bias(seq_len, rel_attn_bias, scores.device())?;
                    let pb = pb.to_dtype(scores.dtype())?;
                    (scores.broadcast_add(&pb)?, pb)
                }
            },
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let v = v.contiguous()?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, self.n_heads * self.d_kv))?;
        let output = self.o.forward(&attn_output)?;
        Ok((output, position_bias))
    }

    /// Compute relative position bias using scalar CPU computation (same as candle's implementation).
    fn compute_bias(
        &self,
        seq_len: usize,
        rel_attn_bias: &candle_nn::Embedding,
        device: &Device,
    ) -> Result<Tensor> {
        let num_buckets = self.relative_attention_num_buckets as u32 / 2;
        let max_exact = num_buckets / 2;
        let relative_position: Vec<Vec<u32>> = (0..seq_len as u32)
            .map(|i| {
                (0..seq_len as u32)
                    .map(|j| {
                        if i < j {
                            // j > i: positive direction
                            if j - i < max_exact {
                                j - i + num_buckets
                            } else {
                                let b = f32::log(
                                    (j - i) as f32 / max_exact as f32,
                                    self.relative_attention_max_distance as f32 / max_exact as f32,
                                ) * (num_buckets - max_exact) as f32;
                                u32::min(
                                    max_exact + num_buckets + b as u32,
                                    self.relative_attention_num_buckets as u32 - 1,
                                )
                            }
                        } else if i - j < max_exact {
                            i - j
                        } else {
                            let b = f32::log(
                                (i - j) as f32 / max_exact as f32,
                                self.relative_attention_max_distance as f32 / max_exact as f32,
                            ) * (num_buckets - max_exact) as f32;
                            max_exact + b as u32
                        }
                    })
                    .collect()
            })
            .collect();
        let relative_buckets = Tensor::new(relative_position, device)?;
        let position_bias = rel_attn_bias
            .forward(&relative_buckets)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
        Ok(position_bias)
    }
}

/// One T5 encoder block: self-attention + FFN, each with pre-norm and residual.
struct T5EncoderBlock {
    attn_norm: T5LayerNorm,
    self_attn: T5SelfAttention,
    ffn_norm: T5LayerNorm,
    ffn: T5GatedFFN,
}

impl T5EncoderBlock {
    fn forward(&mut self, xs: &Tensor, position_bias: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        // Self-attention with pre-norm and residual
        let normed = self.attn_norm.forward(xs)?;
        let (attn_output, position_bias) = self.self_attn.forward(&normed, position_bias)?;
        let xs = (xs + attn_output)?;

        // FFN with pre-norm and residual
        let normed = self.ffn_norm.forward(&xs)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let xs = (xs + ffn_output)?;

        Ok((xs, position_bias))
    }
}

/// Quantized T5-XXL encoder loaded from a GGUF file with standard tensor names.
pub(crate) struct GgufT5Encoder {
    embedding: candle_nn::Embedding,
    blocks: Vec<T5EncoderBlock>,
    final_norm: T5LayerNorm,
}

impl GgufT5Encoder {
    /// Load from a GGUF file using GGUF standard tensor naming.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Load all tensors into a HashMap
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

        // Embedding
        let emb_tensor = get("token_embd.weight")?;
        let emb_weights = emb_tensor.dequantize(device)?;
        let d_model = emb_weights.dim(1)?;
        let embedding = candle_nn::Embedding::new(emb_weights, d_model);

        // Read layer count from metadata, default to 24 (T5-XXL)
        let n_layers = content
            .metadata
            .get("t5encoder.block_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(24);

        let num_heads = 64usize; // T5-XXL
        let d_kv = 64usize;
        let eps = 1e-6f64;
        let rel_attn_buckets = 32usize;
        let rel_attn_max_dist = 128usize;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("enc.blk.{i}");

            // Self-attention
            let q = QMatMul::from_weights(get(&format!("{prefix}.attn_q.weight"))?)?;
            let k = QMatMul::from_weights(get(&format!("{prefix}.attn_k.weight"))?)?;
            let v = QMatMul::from_weights(get(&format!("{prefix}.attn_v.weight"))?)?;
            let o = QMatMul::from_weights(get(&format!("{prefix}.attn_o.weight"))?)?;

            // Only block 0 has relative attention bias
            let relative_attention_bias = if i == 0 {
                let rel_b = get(&format!("{prefix}.attn_rel_b.weight"))?;
                let rel_weights = rel_b.dequantize(device)?;
                let emb_dim = rel_weights.dim(1)?;
                Some(candle_nn::Embedding::new(rel_weights, emb_dim))
            } else {
                None
            };

            let attn_norm_w = get(&format!("{prefix}.attn_norm.weight"))?.dequantize(device)?;
            let attn_norm = T5LayerNorm {
                weight: attn_norm_w,
                variance_epsilon: eps,
            };

            let self_attn = T5SelfAttention {
                q,
                k,
                v,
                o,
                n_heads: num_heads,
                d_kv,
                relative_attention_bias,
                relative_attention_num_buckets: rel_attn_buckets,
                relative_attention_max_distance: rel_attn_max_dist,
            };

            // FFN (gated GeGLU for T5 v1.1)
            let gate = QMatMul::from_weights(get(&format!("{prefix}.ffn_gate.weight"))?)?;
            let up = QMatMul::from_weights(get(&format!("{prefix}.ffn_up.weight"))?)?;
            let down = QMatMul::from_weights(get(&format!("{prefix}.ffn_down.weight"))?)?;
            let ffn = T5GatedFFN {
                gate,
                up,
                down,
                act: Activation::NewGelu,
            };

            let ffn_norm_w = get(&format!("{prefix}.ffn_norm.weight"))?.dequantize(device)?;
            let ffn_norm = T5LayerNorm {
                weight: ffn_norm_w,
                variance_epsilon: eps,
            };

            blocks.push(T5EncoderBlock {
                attn_norm,
                self_attn,
                ffn_norm,
                ffn,
            });
        }

        // Final layer norm
        let final_norm_w = get("enc.output_norm.weight")?.dequantize(device)?;
        let final_norm = T5LayerNorm {
            weight: final_norm_w,
            variance_epsilon: eps,
        };

        Ok(Self {
            embedding,
            blocks,
            final_norm,
        })
    }

    /// Run the T5 encoder forward pass.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let mut xs = self.embedding.forward(input_ids)?;
        let mut position_bias: Option<Tensor> = None;

        for block in &mut self.blocks {
            let (new_xs, new_pb) = block.forward(&xs, position_bias.as_ref())?;
            xs = new_xs;
            position_bias = Some(new_pb);
        }

        self.final_norm.forward(&xs)
    }
}
