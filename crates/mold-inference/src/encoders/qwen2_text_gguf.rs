//! Quantized Qwen2.5-VL text encoder loader for GGUF files.
//!
//! Qwen-Image only needs the language-model text stack from Qwen2.5-VL, not the
//! multimodal projector or vision tower. This loader reads the GGUF language
//! tensors directly and returns last hidden states without the final RMSNorm,
//! matching the upstream diffusers Qwen-Image pipeline.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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

fn compute_rope(
    head_dim: usize,
    rope_theta: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0f32 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;
    let positions: Vec<f32> = (0..context_length).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (context_length, 1), device)?;
    let freqs = positions.matmul(&inv_freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (_b, _h, seq_len, _d) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let cos = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&out1, &out2], D::Minus1).map_err(Into::into)
}

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

struct Qwen2Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    kv_repeat: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Qwen2Attention {
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;
        if let Some(bias) = &self.q_bias {
            q = q.broadcast_add(bias)?;
        }
        if let Some(bias) = &self.k_bias {
            k = k.broadcast_add(bias)?;
        }
        if let Some(bias) = &self.v_bias {
            v = v.broadcast_add(bias)?;
        }

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = apply_rotary_emb(&q, cos, sin, self.head_dim)?;
        let k = apply_rotary_emb(&k, cos, sin, self.head_dim)?;

        let k = repeat_kv(&k, self.kv_repeat)?;
        let v = repeat_kv(&v, self.kv_repeat)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? * scale)?;
        let scores = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, self.hidden_size))?;
        self.o_proj.forward(&attn_output).map_err(Into::into)
    }
}

struct Qwen2Block {
    attn_norm: RmsNorm,
    self_attn: Qwen2Attention,
    ffn_norm: RmsNorm,
    ffn: SwiGluFFN,
}

impl Qwen2Block {
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.attn_norm.forward(xs)?;
        let attn_output = self.self_attn.forward(&normed, cos, sin, mask)?;
        let xs = (xs + attn_output)?;

        let normed = self.ffn_norm.forward(&xs)?;
        let ffn_output = self.ffn.forward(&normed)?;
        (xs + ffn_output).map_err(Into::into)
    }
}

pub(crate) struct GgufQwen2TextEncoder {
    embedding: candle_nn::Embedding,
    blocks: Vec<Qwen2Block>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    dtype: DType,
}

impl GgufQwen2TextEncoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

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
        let get_opt = |name: &str| -> Option<Arc<QTensor>> { tensors.get(name).cloned() };

        let md_usize = |keys: &[&str]| -> Option<usize> {
            keys.iter().find_map(|key| {
                content.metadata.get(*key).and_then(|value| {
                    value
                        .to_u32()
                        .ok()
                        .map(|v| v as usize)
                        .or_else(|| value.to_u64().ok().map(|v| v as usize))
                })
            })
        };
        let md_f64 = |keys: &[&str]| -> Option<f64> {
            keys.iter().find_map(|key| {
                content.metadata.get(*key).and_then(|value| {
                    value
                        .to_f64()
                        .ok()
                        .or_else(|| value.to_f32().ok().map(|v| v as f64))
                })
            })
        };

        let embedding_weight = get("token_embd.weight")?.dequantize(device)?;
        let hidden_size = embedding_weight.dim(1)?;
        let embedding = candle_nn::Embedding::new(embedding_weight, hidden_size);

        let num_heads = md_usize(&[
            "qwen2vl.attention.head_count",
            "qwen2.attention.head_count",
            "llama.attention.head_count",
        ])
        .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata: attention head count"))?;
        let num_kv_heads = md_usize(&[
            "qwen2vl.attention.head_count_kv",
            "qwen2.attention.head_count_kv",
            "llama.attention.head_count_kv",
        ])
        .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata: attention kv head count"))?;
        let block_count = md_usize(&[
            "qwen2vl.block_count",
            "qwen2.block_count",
            "llama.block_count",
        ])
        .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata: block count"))?;
        let context_length = md_usize(&[
            "qwen2vl.context_length",
            "qwen2.context_length",
            "llama.context_length",
        ])
        .unwrap_or(128_000);
        let rms_norm_eps = md_f64(&[
            "qwen2vl.attention.layer_norm_rms_epsilon",
            "qwen2.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_epsilon",
        ])
        .unwrap_or(1e-6);
        let rope_theta = md_f64(&[
            "qwen2vl.rope.freq_base",
            "qwen2.rope.freq_base",
            "llama.rope.freq_base",
        ])
        .unwrap_or(1_000_000.0);

        let head_dim = hidden_size / num_heads;
        let kv_repeat = num_heads / num_kv_heads;
        let (cos, sin) = compute_rope(head_dim, rope_theta, context_length, device)?;

        let mut blocks = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let prefix = format!("blk.{i}");

            let q_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_q.weight"))?)?;
            let k_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_k.weight"))?)?;
            let v_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_v.weight"))?)?;
            let o_proj = QMatMul::from_weights(get(&format!("{prefix}.attn_output.weight"))?)?;

            let q_bias = get_opt(&format!("{prefix}.attn_q.bias"))
                .map(|t| t.dequantize(device))
                .transpose()?;
            let k_bias = get_opt(&format!("{prefix}.attn_k.bias"))
                .map(|t| t.dequantize(device))
                .transpose()?;
            let v_bias = get_opt(&format!("{prefix}.attn_v.bias"))
                .map(|t| t.dequantize(device))
                .transpose()?;

            let q_norm = get_opt(&format!("{prefix}.attn_q_norm.weight"))
                .map(|t| {
                    t.dequantize(device).map(|weight| RmsNorm {
                        weight,
                        eps: rms_norm_eps,
                    })
                })
                .transpose()?;
            let k_norm = get_opt(&format!("{prefix}.attn_k_norm.weight"))
                .map(|t| {
                    t.dequantize(device).map(|weight| RmsNorm {
                        weight,
                        eps: rms_norm_eps,
                    })
                })
                .transpose()?;

            let attn_norm = RmsNorm {
                weight: get(&format!("{prefix}.attn_norm.weight"))?.dequantize(device)?,
                eps: rms_norm_eps,
            };
            let ffn_norm = RmsNorm {
                weight: get(&format!("{prefix}.ffn_norm.weight"))?.dequantize(device)?,
                eps: rms_norm_eps,
            };

            let ffn = SwiGluFFN {
                gate: QMatMul::from_weights(get(&format!("{prefix}.ffn_gate.weight"))?)?,
                up: QMatMul::from_weights(get(&format!("{prefix}.ffn_up.weight"))?)?,
                down: QMatMul::from_weights(get(&format!("{prefix}.ffn_down.weight"))?)?,
            };

            let self_attn = Qwen2Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_bias,
                k_bias,
                v_bias,
                q_norm,
                k_norm,
                num_heads,
                num_kv_heads,
                kv_repeat,
                head_dim,
                hidden_size,
            };

            blocks.push(Qwen2Block {
                attn_norm,
                self_attn,
                ffn_norm,
                ffn,
            });
        }

        Ok(Self {
            embedding,
            blocks,
            cos,
            sin,
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
            .map_err(Into::into)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = attn_mask.dims2()?;
        let mut mask = Vec::with_capacity(b_sz);
        for b in 0..b_sz {
            let token_mask = attn_mask.i((b, ..))?.expand((1, 1, seq_len, seq_len))?;
            mask.push(token_mask);
        }
        let pad_mask = Tensor::cat(&mask.iter().collect::<Vec<_>>(), 0)?;
        let on_true = pad_mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(pad_mask.shape())?
            .to_dtype(self.dtype)?;
        let pad_mask = pad_mask.where_cond(&on_true, &on_false)?;
        let causal_mask = self.prepare_causal_attention_mask(b_sz, seq_len, 0)?;
        causal_mask.broadcast_add(&pad_mask).map_err(Into::into)
    }

    pub fn forward_last_hidden(
        &mut self,
        input_ids: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len) = input_ids.dims2()?;
        let attention_mask = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b, seq_len, 0)?)
                }
            }
        };

        let mut xs = self.embedding.forward(input_ids)?;
        for block in &self.blocks {
            xs = block.forward(&xs, &self.cos, &self.sin, attention_mask.as_ref())?;
        }
        Ok(xs)
    }
}
