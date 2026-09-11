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
    /// The checkpoint the blocks were built from, retained so parking is a
    /// device move rather than a re-read from disk.
    ///
    /// Retaining it is very nearly free: `QMatMul::from_weights` keeps the
    /// `Arc<QTensor>` it was given, so every block weight in this map is the
    /// same allocation the model is already using. The ONE exception is
    /// `token_embd.weight`, which is dequantized once at load and then never
    /// touched again — [`GgufQwen3Encoder::from_tensors`] therefore relocates
    /// that single entry to the host, so the retained map adds no device bytes
    /// at all.
    retained: GgufCheckpoint,
}

/// A parked GGUF checkpoint: every tensor on the host, plus the header
/// metadata the architecture half needs to rebuild.
pub(crate) type GgufParked = GgufCheckpoint;

/// The checkpoint as the architecture half consumes it.
type GgufCheckpoint = (
    HashMap<String, Arc<QTensor>>,
    HashMap<String, gguf_file::Value>,
);

/// Read every tensor off one memory mapping, with the header metadata the
/// architecture half needs.
fn read_tensors(path: &Path, device: &Device) -> Result<GgufCheckpoint> {
    let map = mold_candle::gguf_mmap::GgufMmap::open(path)?;
    let metadata = map.content().metadata.clone();
    let tensors = map.load_all(device, &mut |_, _| {})?;
    Ok((tensors, metadata))
}

impl GgufQwen3Encoder {
    /// Load from a GGUF file.
    ///
    /// Split in two so the transport and the architecture are separable: the
    /// read is one memory mapping (see `mold_candle::gguf_mmap`), and
    /// [`Self::from_tensors`] is the part that knows llama.cpp's naming.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let (tensors, metadata) = read_tensors(path, device)?;
        Self::from_tensors(tensors, metadata, device)
    }

    /// Move the whole checkpoint to host RAM and hand it back, leaving nothing
    /// on the device.
    ///
    /// Byte-exact: `wan::block_offload::qtensor_to_device` serializes the
    /// quantized blocks through `QTensor::data` and reconstructs them with
    /// `qtensor_from_ggml`, so an unparked encoder is bit-for-bit the one that
    /// was parked — which is the property `qwen3_gguf_park_unpark_is_byte_identical`
    /// pins. This is the same mechanism #1044 gave Qwen-Image's Qwen2 encoder;
    /// the "GGUF is device-tied and cannot park" carve-out this replaces was a
    /// scoping decision, never a limitation.
    pub fn park_to_cpu(&self) -> Result<GgufParked> {
        let (tensors, metadata) = &self.retained;
        let mut parked = HashMap::with_capacity(tensors.len());
        for (name, tensor) in tensors {
            parked.insert(
                name.clone(),
                crate::wan::block_offload::qtensor_to_device(tensor, &Device::Cpu)?,
            );
        }
        Ok((parked, metadata.clone()))
    }

    /// Rebuild on `device` from a parked checkpoint.
    pub fn from_parked(parked: &GgufParked, device: &Device) -> Result<Self> {
        let (tensors, metadata) = parked;
        let mut restored = HashMap::with_capacity(tensors.len());
        for (name, tensor) in tensors {
            restored.insert(
                name.clone(),
                crate::wan::block_offload::qtensor_to_device(tensor, device)?,
            );
        }
        Self::from_tensors(restored, metadata.clone(), device)
    }

    fn from_tensors(
        tensors: HashMap<String, Arc<QTensor>>,
        metadata: HashMap<String, gguf_file::Value>,
        device: &Device,
    ) -> Result<Self> {
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
        let n_layers = metadata
            .get("qwen3.block_count")
            .or_else(|| metadata.get("llama.block_count"))
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

        // The embedding's quantized source has done its only job. Relocating
        // it to the host is what keeps the retained map free of device bytes
        // the model is not already holding.
        let mut retained = tensors;
        if let Some(embed) = retained.get("token_embd.weight") {
            let host = crate::wan::block_offload::qtensor_to_device(embed, &Device::Cpu)?;
            retained.insert("token_embd.weight".to_string(), host);
        }

        Ok(Self {
            embedding,
            blocks,
            retained: (retained, metadata),
        })
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
        if layer_indices.is_empty() {
            anyhow::bail!("layer_indices must not be empty");
        }
        let max_layer = layer_indices.iter().copied().max().unwrap_or(0);
        if max_layer >= self.blocks.len() {
            anyhow::bail!(
                "layer index {max_layer} out of bounds (model has {} layers)",
                self.blocks.len()
            );
        }

        let (_batch, seq_len) = input_ids.dims2()?;
        let mut xs = self.embedding.forward(input_ids)?;
        let (cos, sin) = compute_rope(seq_len, xs.device())?;
        let mask = causal_mask(seq_len, xs.dtype(), xs.device())?;

        let n_run = max_layer + 1;
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

impl GgufQwen3Encoder {
    /// [`Self::park_to_cpu`] with the same-device short circuit removed.
    ///
    /// Split out for exactly the reason `wan::block_offload::rebuild_on` is:
    /// `qtensor_to_device` hands back the input `Arc` when the target is the
    /// device the tensor is already on, so a CPU-only test written against the
    /// production path would compare a tensor with itself and prove nothing
    /// about the byte path — which is the whole correctness claim.
    #[cfg(test)]
    fn park_rebuilt(&self) -> Result<GgufParked> {
        let (tensors, metadata) = &self.retained;
        let mut parked = HashMap::with_capacity(tensors.len());
        for (name, tensor) in tensors {
            parked.insert(
                name.clone(),
                crate::wan::block_offload::rebuild_on(tensor, &Device::Cpu)?,
            );
        }
        Ok((parked, metadata.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    const TEST_VOCAB: usize = 32;
    const TEST_FFN: usize = 256;
    /// The block width is fixed by the architecture constants above
    /// (`N_HEADS * HEAD_DIM`), so a "tiny" fixture can shrink the vocabulary
    /// and the FFN but not this.
    const TEST_DIM: usize = N_HEADS * HEAD_DIM;

    /// A deterministic quantized tensor of the given shape.
    fn q(shape: (usize, usize), seed: f32) -> Arc<QTensor> {
        let count = shape.0 * shape.1;
        let values: Vec<f32> = (0..count)
            .map(|i| ((i as f32) * 0.001 + seed).sin())
            .collect();
        let dense = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&dense, GgmlDType::Q8_0).unwrap())
    }

    fn norm(width: usize, seed: f32) -> Arc<QTensor> {
        let values: Vec<f32> = (0..width)
            .map(|i| 1.0 + ((i as f32) * 0.01 + seed).cos() * 0.1)
            .collect();
        let dense = Tensor::from_vec(values, (1, width), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&dense, GgmlDType::F32).unwrap())
    }

    /// A one-block checkpoint carrying every tensor `from_tensors` reads.
    fn synthetic_checkpoint() -> GgufCheckpoint {
        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::new();
        tensors.insert("token_embd.weight".into(), q((TEST_VOCAB, TEST_DIM), 0.1));
        let kv_width = N_KV_HEADS * HEAD_DIM;
        tensors.insert("blk.0.attn_q.weight".into(), q((TEST_DIM, TEST_DIM), 0.2));
        tensors.insert("blk.0.attn_k.weight".into(), q((kv_width, TEST_DIM), 0.3));
        tensors.insert("blk.0.attn_v.weight".into(), q((kv_width, TEST_DIM), 0.4));
        tensors.insert(
            "blk.0.attn_output.weight".into(),
            q((TEST_DIM, TEST_DIM), 0.5),
        );
        tensors.insert("blk.0.attn_q_norm.weight".into(), norm(HEAD_DIM, 0.6));
        tensors.insert("blk.0.attn_k_norm.weight".into(), norm(HEAD_DIM, 0.7));
        tensors.insert("blk.0.attn_norm.weight".into(), norm(TEST_DIM, 0.8));
        tensors.insert("blk.0.ffn_norm.weight".into(), norm(TEST_DIM, 0.9));
        tensors.insert("blk.0.ffn_gate.weight".into(), q((TEST_FFN, TEST_DIM), 1.0));
        tensors.insert("blk.0.ffn_up.weight".into(), q((TEST_FFN, TEST_DIM), 1.1));
        tensors.insert("blk.0.ffn_down.weight".into(), q((TEST_DIM, TEST_FFN), 1.2));

        let mut metadata = HashMap::new();
        metadata.insert("qwen3.block_count".to_string(), gguf_file::Value::U32(1));
        (tensors, metadata)
    }

    /// The premise of the GGUF park: the bytes that come back are the bytes
    /// that went out, and the encoder built from them renders identically.
    ///
    /// A dequantize/re-quantize round trip would pass a loose tolerance check
    /// and still make a render depend on whether the encoder happened to be
    /// parked, so this asserts raw storage equality and then BIT equality of
    /// the forward output — not closeness.
    ///
    /// This is the test that retires the "GGUF: device-tied QTensors don't
    /// survive a CPU round-trip" carve-out. They do, through exactly the
    /// mechanism #1044 already gave Qwen-Image's Qwen2 encoder.
    #[test]
    fn qwen3_gguf_park_unpark_is_byte_identical() {
        let (tensors, metadata) = synthetic_checkpoint();
        let mut original =
            GgufQwen3Encoder::from_tensors(tensors.clone(), metadata.clone(), &Device::Cpu)
                .expect("the synthetic checkpoint carries every tensor the loader reads");

        let parked = original.park_rebuilt().expect("park");
        assert_eq!(
            parked.0.len(),
            tensors.len(),
            "a park must keep every tensor the checkpoint carried"
        );
        for (name, before) in &tensors {
            let after = parked.0.get(name).expect("parked set keeps the name");
            assert_eq!(before.dtype(), after.dtype(), "{name} changed quantization");
            assert_eq!(before.shape(), after.shape(), "{name} changed shape");
            assert_eq!(
                before.data().unwrap().as_ref(),
                after.data().unwrap().as_ref(),
                "{name} is not byte-identical after a park/unpark cycle"
            );
        }

        let mut restored =
            GgufQwen3Encoder::from_parked(&parked, &Device::Cpu).expect("unpark rebuilds");

        let ids = Tensor::from_vec(vec![1u32, 5, 9, 2], (1, 4), &Device::Cpu).unwrap();
        let before = original.forward(&ids).unwrap();
        let after = restored.forward(&ids).unwrap();
        assert_eq!(
            before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "a parked encoder must render bit-identically to the one that was parked"
        );
    }

    /// Retaining the checkpoint is not a second copy of the model.
    ///
    /// Every matmul weight in the map is the very `Arc` its `QMatMul` holds,
    /// so it costs nothing. The tensors that are NOT shared are the ones the
    /// loader dequantizes once and never looks at again: the token embedding,
    /// which `from_tensors` therefore relocates to the host because it is the
    /// only large one, and the four RMS norm weights per block, which are a
    /// handful of kilobytes each. What this pins is the bound — the retained
    /// map must never hold a MEANINGFUL unshared allocation on the device,
    /// because that is the failure this design exists to avoid.
    #[test]
    fn the_retained_checkpoint_is_not_a_second_copy_of_the_model() {
        const NEGLIGIBLE: usize = 1024 * 1024;

        let (tensors, metadata) = synthetic_checkpoint();
        let model = GgufQwen3Encoder::from_tensors(tensors, metadata, &Device::Cpu).unwrap();
        let (retained, _) = &model.retained;

        let embedding = retained
            .get("token_embd.weight")
            .expect("the embedding stays in the map so an unpark can rebuild from it");
        assert!(
            embedding.device().is_cpu(),
            "the embedding's quantized source belongs on the host after the dequantize"
        );

        let mut unshared_device_bytes = 0usize;
        for (name, tensor) in retained {
            if name == "token_embd.weight" || Arc::strong_count(tensor) > 1 {
                continue;
            }
            unshared_device_bytes += tensor.storage_size_in_bytes();
        }
        assert!(
            unshared_device_bytes < NEGLIGIBLE,
            "the retained map holds {unshared_device_bytes} device bytes nothing else is using"
        );
    }
}
