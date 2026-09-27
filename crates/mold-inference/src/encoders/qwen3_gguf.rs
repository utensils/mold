//! Quantized Qwen3 language-model loader for GGUF files (llama.cpp standard naming).
//!
//! Serves three checkpoints: the Qwen3-4B Z-Image / Klein-4B encoder, the
//! Qwen3-8B Klein-9B encoder, and the language half of Qwen3-VL-8B-Instruct
//! (Qwen Image 2.1). All three share one block: 32 Q heads, 8 KV heads (GQA
//! 4:1), 128 head_dim, SwiGLU MLP, per-head Q/K RMS norms before RoPE. What
//! differs is read from the GGUF's own metadata under its
//! `general.architecture` key (`qwen3.*` or `qwen3vl.*`): the block count, the
//! RoPE base (1e6 for Qwen3, 5e6 for Qwen3-VL — `qwen3vl.rope.freq_base`), the
//! RMS epsilon and the head counts. A file carrying none of them keeps the
//! historical Qwen3-4B constants, so every existing checkpoint builds exactly
//! as before.
//!
//! GGUF tensor names (llama.cpp standard):
//! - `token_embd.weight`
//! - `blk.{i}.attn_norm.weight`, `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`,
//!   `blk.{i}.attn_v.weight`, `blk.{i}.attn_output.weight`
//! - `blk.{i}.attn_q_norm.weight`, `blk.{i}.attn_k_norm.weight`
//! - `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`,
//!   `blk.{i}.ffn_down.weight`
//!
//! llama.cpp's Qwen2/3/3-VL converters keep HuggingFace's Q/K row order (NEOX
//! rope, `rotate_half`), so the half-split rotation below is the checkpoint's
//! own; the weight-gated parity test pins each dequantized tensor to the BF16
//! shards, which is what proves no permutation happened.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::qwen3_vl_inject::{self, VisualInjection};

// ── Architecture ─────────────────────────────────────────────────────────────

/// Default layer count for Qwen3-4B (read from GGUF metadata if available).
const DEFAULT_N_LAYERS: usize = 36;
const DEFAULT_N_HEADS: usize = 32; // Q heads
const DEFAULT_N_KV_HEADS: usize = 8; // K/V heads (GQA 4:1)
const DEFAULT_HEAD_DIM: usize = 128;
const DEFAULT_ROPE_THETA: f64 = 1_000_000.0;
const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
/// Return output after this many layers (second-to-last = 35 layers of 36).
const N_RETURN_LAYERS: usize = 35;
/// Query rows per attention chunk: bounds the score tile to
/// `heads x chunk x keys` however long the (multimodal) sequence is.
const ATTENTION_QUERY_CHUNK: usize = 1024;

/// The per-checkpoint half of the architecture, from GGUF metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GgufQwen3Arch {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for GgufQwen3Arch {
    fn default() -> Self {
        Self {
            n_layers: DEFAULT_N_LAYERS,
            n_heads: DEFAULT_N_HEADS,
            n_kv_heads: DEFAULT_N_KV_HEADS,
            head_dim: DEFAULT_HEAD_DIM,
            rope_theta: DEFAULT_ROPE_THETA,
            rms_norm_eps: DEFAULT_RMS_NORM_EPS,
        }
    }
}

impl GgufQwen3Arch {
    /// Read the architecture from `metadata`, under `general.architecture`
    /// (llama.cpp writes `qwen3` or `qwen3vl`), falling back to `qwen3.*` and
    /// `llama.*` and then to the Qwen3-4B defaults field by field.
    pub(crate) fn from_metadata(metadata: &HashMap<String, gguf_file::Value>) -> Result<Self> {
        let arch = match metadata.get("general.architecture") {
            Some(gguf_file::Value::String(arch)) => Some(arch.clone()),
            _ => None,
        };
        let prefixes = arch
            .iter()
            .map(String::as_str)
            .chain(["qwen3", "llama"])
            .collect::<Vec<_>>();
        let find = |suffix: &str| {
            prefixes
                .iter()
                .find_map(|prefix| metadata.get(&format!("{prefix}.{suffix}")))
        };
        let integer = |suffix: &str, default: usize| -> Result<usize> {
            Ok(match find(suffix) {
                None => default,
                Some(value) => value
                    .to_u64()
                    .map(|v| v as usize)
                    .or_else(|_| value.to_u32().map(|v| v as usize))?,
            })
        };
        let float = |suffix: &str, default: f64| -> Result<f64> {
            Ok(match find(suffix) {
                None => default,
                Some(gguf_file::Value::F32(v)) => f64::from(*v),
                Some(gguf_file::Value::F64(v)) => *v,
                Some(other) => anyhow::bail!("GGUF {suffix} is not a float: {other:?}"),
            })
        };
        let n_heads = integer("attention.head_count", DEFAULT_N_HEADS)?;
        let arch = Self {
            n_layers: integer("block_count", DEFAULT_N_LAYERS)?,
            n_heads,
            n_kv_heads: integer("attention.head_count_kv", DEFAULT_N_KV_HEADS)?,
            head_dim: integer("attention.key_length", DEFAULT_HEAD_DIM)?,
            rope_theta: float("rope.freq_base", DEFAULT_ROPE_THETA)?,
            rms_norm_eps: float("attention.layer_norm_rms_epsilon", DEFAULT_RMS_NORM_EPS)?,
        };
        anyhow::ensure!(
            arch.n_layers > 0
                && arch.n_kv_heads > 0
                && arch.n_heads.is_multiple_of(arch.n_kv_heads)
                && arch.head_dim.is_multiple_of(2),
            "GGUF Qwen3 architecture is inconsistent: {arch:?}"
        );
        Ok(arch)
    }

    fn kv_repeat(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

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

fn inv_freq(arch: &GgufQwen3Arch) -> Vec<f32> {
    let half_dim = arch.head_dim / 2;
    (0..half_dim)
        .map(|i| 1.0f32 / (arch.rope_theta as f32).powf(2.0 * i as f32 / arch.head_dim as f32))
        .collect()
}

/// `(cos, sin)` for the shared positions `0..seq_len`, each `(seq_len, half)`.
fn compute_rope(arch: &GgufQwen3Arch, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half_dim = arch.head_dim / 2;
    let inv_freq = Tensor::from_vec(inv_freq(arch), (1, half_dim), device)?;
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;
    let freqs = positions.matmul(&inv_freq)?; // (seq_len, half_dim)
    Ok((freqs.cos()?, freqs.sin()?))
}

/// `(cos, sin)` for per-row positions, each `(batch, seq_len, half)` — the
/// left-padded Qwen3-VL prompt batches restart every row's positions at zero
/// (`Bf16Qwen3Encoder::rope_positions_for_attention`).
fn compute_rope_rows(
    arch: &GgufQwen3Arch,
    rows: &[Vec<usize>],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = arch.head_dim / 2;
    let seq_len = rows.first().map_or(0, Vec::len);
    let freqs = inv_freq(arch);
    let mut angles = Vec::with_capacity(rows.len() * seq_len * half_dim);
    for row in rows {
        anyhow::ensure!(row.len() == seq_len, "ragged Qwen3 position rows");
        for &position in row {
            angles.extend(freqs.iter().map(|freq| position as f32 * freq));
        }
    }
    let angles = Tensor::from_vec(angles, (rows.len(), seq_len, half_dim), device)?;
    Ok((angles.cos()?, angles.sin()?))
}

/// Apply rotary embeddings to `(batch, heads, seq_len, head_dim)`. `cos`/`sin`
/// are `(seq, half)` shared by every row, or `(batch, seq, half)` per row.
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq_len, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let (cos, sin) = if cos.rank() == 2 {
        // (seq_len, half_dim) → (1, 1, seq_len, half_dim)
        (
            cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?,
            sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?,
        )
    } else {
        // (batch, seq_len, half_dim) → (batch, 1, seq_len, half_dim)
        (cos.unsqueeze(1)?, sin.unsqueeze(1)?)
    };
    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;
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

/// The attention mask a forward applies.
#[derive(Clone, Copy)]
enum AttentionMask<'a> {
    /// A caller-built additive `(B, 1, L, L)` mask (the text paths: plain
    /// causal, or causal + key padding for left-padded prompt batches).
    Additive(&'a Tensor),
    /// Plain causal attention with the bias built ON the device per query
    /// chunk, each chunk reading only the keys it can see — the multimodal
    /// forward, whose sequence (ten 1024-px references are ~10k tokens) would
    /// otherwise need a full `L x L` host mask (~424 MB of F32 at 10k). The
    /// BF16 twin is `Bf16Qwen3Encoder`'s `forward_causal_with_tables`.
    Causal,
}

/// The additive causal bias for query rows `start..start + rows` against keys
/// `0..start + rows`, `(1, 1, rows, start + rows)` in `dtype`, built on
/// `device`: key `j` is visible to query `start + r` iff `j <= start + r`.
fn causal_chunk_bias(start: usize, rows: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let keys = start + rows;
    let key_index = Tensor::arange(0u32, keys as u32, device)?.reshape((1, keys))?;
    let query_index = Tensor::arange(start as u32, keys as u32, device)?.reshape((rows, 1))?;
    let visible = key_index.broadcast_le(&query_index)?;
    let blocked = Tensor::full(f32::NEG_INFINITY, (rows, keys), device)?.to_dtype(dtype)?;
    let zero = Tensor::zeros((rows, keys), dtype, device)?;
    Ok(visible
        .where_cond(&zero, &blocked)?
        .reshape((1, 1, rows, keys))?)
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
    arch: GgufQwen3Arch,
    /// Query rows per attention chunk ([`ATTENTION_QUERY_CHUNK`]; tests
    /// override it to exercise several chunks on a short prompt).
    query_chunk: usize,
}

impl Qwen3Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttentionMask<'_>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let arch = self.arch;

        // Project Q/K/V
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, seq_len, arch.n_heads, arch.head_dim))?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, seq_len, arch.n_kv_heads, arch.head_dim))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, seq_len, arch.n_kv_heads, arch.head_dim))?;

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
        let k = repeat_kv(&k, arch.kv_repeat())?;
        let v = repeat_kv(&v, arch.kv_repeat())?.contiguous()?;

        // Scaled dot-product attention with the additive mask, one bounded
        // block of query rows at a time. A single chunk is exactly the
        // unchunked computation, so every prompt under the chunk size runs
        // the historical arithmetic.
        let scale = 1.0 / (arch.head_dim as f64).sqrt();
        let k_t = k.t()?;
        let query_chunk = self.query_chunk.max(1);
        let mut chunks = Vec::with_capacity(seq_len.div_ceil(query_chunk));
        for start in (0..seq_len).step_by(query_chunk) {
            let rows = query_chunk.min(seq_len - start);
            let q = if rows == seq_len {
                q.clone()
            } else {
                q.narrow(2, start, rows)?
            };
            let (scores, v) = match mask {
                AttentionMask::Additive(mask) => {
                    let mask = if rows == seq_len {
                        mask.clone()
                    } else {
                        mask.narrow(2, start, rows)?
                    };
                    ((q.matmul(&k_t)? * scale)?.broadcast_add(&mask)?, v.clone())
                }
                AttentionMask::Causal => {
                    // Only keys `0..start + rows` are visible to this chunk.
                    let keys = start + rows;
                    let k_seen = k_t.narrow(3, 0, keys)?;
                    let v_seen = v.narrow(2, 0, keys)?;
                    let scores = (q.matmul(&k_seen)? * scale)?;
                    let scores = if rows > 1 {
                        scores.broadcast_add(&causal_chunk_bias(
                            start,
                            rows,
                            scores.dtype(),
                            scores.device(),
                        )?)?
                    } else {
                        scores
                    };
                    (scores, v_seen)
                }
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
            chunks.push(attn_weights.matmul(&v)?);
        }
        let attn_output = if chunks.len() == 1 {
            chunks.pop().expect("one chunk")
        } else {
            Tensor::cat(&chunks, 2)?
        };

        // Reshape back: (B, heads, seq, head_dim) → (B, seq, hidden_dim)
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, arch.n_heads * arch.head_dim))?;

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
        mask: AttentionMask<'_>,
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
    arch: GgufQwen3Arch,
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

        // Block count, head geometry, RoPE base and epsilon from the file's own
        // metadata (`qwen3.*`, `qwen3vl.*`), Qwen3-4B defaults otherwise.
        let arch = GgufQwen3Arch::from_metadata(&metadata)?;

        let mut blocks = Vec::with_capacity(arch.n_layers);
        for i in 0..arch.n_layers {
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
                eps: arch.rms_norm_eps,
            };
            let k_norm_w = get(&format!("{prefix}.attn_k_norm.weight"))?.dequantize(device)?;
            let k_norm = RmsNorm {
                weight: k_norm_w,
                eps: arch.rms_norm_eps,
            };

            // Attention + FFN norms
            let attn_norm_w = get(&format!("{prefix}.attn_norm.weight"))?.dequantize(device)?;
            let attn_norm = RmsNorm {
                weight: attn_norm_w,
                eps: arch.rms_norm_eps,
            };
            let ffn_norm_w = get(&format!("{prefix}.ffn_norm.weight"))?.dequantize(device)?;
            let ffn_norm = RmsNorm {
                weight: ffn_norm_w,
                eps: arch.rms_norm_eps,
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
                arch,
                query_chunk: ATTENTION_QUERY_CHUNK,
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
            arch,
            retained: (retained, metadata),
        })
    }

    /// The architecture this checkpoint declared.
    #[allow(dead_code)] // read by the multimodal conditioning encoder and tests
    pub(crate) fn arch(&self) -> GgufQwen3Arch {
        self.arch
    }

    /// Run every decoder layer and return the final hidden states BEFORE the
    /// model's final RMSNorm — the state Qwen Image 2.1 consumes (diffusers
    /// `pipeline_qwenimage21.py` takes `hidden_states[-1]` pre-norm), and the
    /// GGUF twin of `Bf16Qwen3Encoder::forward_final_pre_norm_with_attention`.
    ///
    /// `attention[b][key]` marks the real positions of batch row `b` in a
    /// left-padded prompt batch. With it, the mask is the BF16 encoder's
    /// causal + key-padding mask (including its unmask-unattended rule) and
    /// each row's RoPE positions restart at zero after its pads, exactly as
    /// transformers' `get_rope_index` derives them. `None` is plain causal
    /// over positions `0..L`.
    pub(crate) fn forward_final_pre_norm_with_attention(
        &mut self,
        input_ids: &Tensor,
        attention: Option<&[Vec<bool>]>,
    ) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let mut xs = self.embedding.forward(input_ids)?;
        let (cos, sin, mask) = match attention {
            Some(rows) => {
                anyhow::ensure!(
                    rows.len() == batch,
                    "Qwen3 attention mask mismatch: {batch} batch row(s) expected, got {}",
                    rows.len()
                );
                let positions = super::qwen3_bf16::Bf16Qwen3Encoder::rope_positions_for_attention(
                    rows, seq_len,
                )?;
                let (cos, sin) = compute_rope_rows(&self.arch, &positions, xs.device())?;
                let mask = super::qwen3_bf16::Bf16Qwen3Encoder::batch_attention_mask(
                    rows,
                    seq_len,
                    xs.dtype(),
                    xs.device(),
                )?;
                (cos, sin, mask)
            }
            None => {
                let (cos, sin) = compute_rope(&self.arch, seq_len, xs.device())?;
                (cos, sin, causal_mask(seq_len, xs.dtype(), xs.device())?)
            }
        };
        for block in self.blocks.iter_mut() {
            xs = block.forward(&xs, &cos, &sin, AttentionMask::Additive(&mask))?;
        }
        Ok(xs)
    }

    /// The multimodal forward: `<|image_pad|>` rows replaced by the vision
    /// merger's output, interleaved MRoPE from three position axes, and
    /// DeepStack features added after the first layers — transformers
    /// `modeling_qwen3_vl.py` (`Qwen3VLModel.forward`, `:299-314`, `:861-883`).
    ///
    /// Batch-1, like the BF16 twin: upstream encodes the positive and negative
    /// prompts in separate calls. `mrope` holds the T/H/W position of every
    /// token (text tokens carry the same value on all three axes). Returns
    /// the final hidden states before the final RMSNorm.
    #[allow(dead_code)] // driven by the reference-image conditioning encoder
    pub(crate) fn forward_multimodal_final_pre_norm(
        &mut self,
        input_ids: &Tensor,
        visual: Option<VisualInjection>,
        mrope: &[Vec<u32>; 3],
    ) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        anyhow::ensure!(batch == 1, "Qwen3-VL multimodal forward is batch-1");
        anyhow::ensure!(
            mrope.iter().all(|axis| axis.len() == seq_len),
            "Qwen3-VL MRoPE positions cover {} tokens, the input has {seq_len}",
            mrope[0].len()
        );
        let mut xs = self.embedding.forward(input_ids)?;
        let hidden = xs.dim(2)?;
        if let Some(visual) = &visual {
            visual.validate(seq_len, hidden, self.blocks.len())?;
            xs = qwen3_vl_inject::inject_visual_rows(&xs, visual)?;
        }
        let (cos, sin) = qwen3_vl_inject::mrope_cos_sin(
            mrope,
            self.arch.head_dim,
            self.arch.rope_theta,
            qwen3_vl_inject::QWEN3_VL_MROPE_SECTIONS,
            xs.device(),
        )?;
        for (layer, block) in self.blocks.iter_mut().enumerate() {
            xs = block.forward(&xs, &cos, &sin, AttentionMask::Causal)?;
            if let Some(visual) = &visual {
                xs = qwen3_vl_inject::apply_deepstack(&xs, visual, layer)?;
            }
        }
        Ok(xs)
    }

    /// Run the Qwen3 encoder forward pass.
    /// Returns the second-to-last layer output (no final norm).
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;

        let mut xs = self.embedding.forward(input_ids)?;

        // Compute RoPE sin/cos for this sequence length
        let (cos, sin) = compute_rope(&self.arch, seq_len, xs.device())?;

        // Compute causal attention mask
        let mask = causal_mask(seq_len, xs.dtype(), xs.device())?;

        // Run through layers, stop at second-to-last (return layer N_RETURN_LAYERS-1)
        let n_run = N_RETURN_LAYERS.min(self.blocks.len());
        for block in self.blocks[..n_run].iter_mut() {
            xs = block.forward(&xs, &cos, &sin, AttentionMask::Additive(&mask))?;
        }

        Ok(xs)
    }

    /// Run forward pass and collect hidden states from specific layers.
    /// Returns outputs stacked and reshaped: (B, seq_len, num_layers * hidden_size).
    /// Used by Flux.2 Klein which needs layers 9, 18, 27 stacked to 7680-dim.
    ///
    /// `attention` names the real (non-pad) positions of a fixed-width
    /// prompt. When present the causal mask additionally excludes the padded
    /// KEYS, which is the mask BFL hands the Qwen3 language model
    /// (`flux2/text_encoder.py:408-416`).
    pub fn forward_with_layers(
        &mut self,
        input_ids: &Tensor,
        layer_indices: &[usize],
        attention: Option<&[bool]>,
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
        let (cos, sin) = compute_rope(&self.arch, seq_len, xs.device())?;
        let mask = match attention {
            Some(attention) => {
                if attention.len() != seq_len {
                    anyhow::bail!(
                        "Qwen3 attention mask covers {} positions but the input has {seq_len}",
                        attention.len()
                    );
                }
                super::qwen3::causal_padding_mask(attention, xs.dtype(), xs.device())?
            }
            None => causal_mask(seq_len, xs.dtype(), xs.device())?,
        };

        let n_run = max_layer + 1;
        let mut collected: Vec<Tensor> = Vec::with_capacity(layer_indices.len());

        for (i, block) in self.blocks[..n_run].iter_mut().enumerate() {
            xs = block.forward(&xs, &cos, &sin, AttentionMask::Additive(&mask))?;
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
    /// Override the attention's query-chunk size (every block).
    #[cfg(test)]
    fn set_query_chunk(&mut self, rows: usize) {
        for block in &mut self.blocks {
            block.self_attn.query_chunk = rows;
        }
    }

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
    /// (`DEFAULT_N_HEADS * DEFAULT_HEAD_DIM`), so a "tiny" fixture can shrink the vocabulary
    /// and the FFN but not this.
    const TEST_DIM: usize = DEFAULT_N_HEADS * DEFAULT_HEAD_DIM;

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
        let kv_width = DEFAULT_N_KV_HEADS * DEFAULT_HEAD_DIM;
        tensors.insert("blk.0.attn_q.weight".into(), q((TEST_DIM, TEST_DIM), 0.2));
        tensors.insert("blk.0.attn_k.weight".into(), q((kv_width, TEST_DIM), 0.3));
        tensors.insert("blk.0.attn_v.weight".into(), q((kv_width, TEST_DIM), 0.4));
        tensors.insert(
            "blk.0.attn_output.weight".into(),
            q((TEST_DIM, TEST_DIM), 0.5),
        );
        tensors.insert(
            "blk.0.attn_q_norm.weight".into(),
            norm(DEFAULT_HEAD_DIM, 0.6),
        );
        tensors.insert(
            "blk.0.attn_k_norm.weight".into(),
            norm(DEFAULT_HEAD_DIM, 0.7),
        );
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

    fn to_rows(t: &Tensor) -> Vec<Vec<f32>> {
        t.squeeze(0).unwrap().to_vec2::<f32>().unwrap()
    }

    fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
            .fold(0.0, f32::max)
    }

    fn tiny_encoder() -> GgufQwen3Encoder {
        let (tensors, metadata) = synthetic_checkpoint();
        GgufQwen3Encoder::from_tensors(tensors, metadata, &Device::Cpu).unwrap()
    }

    /// [`tiny_encoder`] with every weight stored as F32. A quantized
    /// `QMatMul` on the CPU re-quantizes its ACTIVATION to Q8_K, so a
    /// last-ulp change in the attention output can flip an activation block's
    /// rounding and move a row by percent; comparing two attention schedules
    /// needs linears that pass an ulp through as an ulp.
    fn dense_tiny_encoder() -> GgufQwen3Encoder {
        let (tensors, metadata) = synthetic_checkpoint();
        let dense = tensors
            .into_iter()
            .map(|(name, tensor)| {
                let values = tensor.dequantize(&Device::Cpu).unwrap();
                let widened = QTensor::quantize(&values, GgmlDType::F32).unwrap();
                (name, Arc::new(widened))
            })
            .collect();
        GgufQwen3Encoder::from_tensors(dense, metadata, &Device::Cpu).unwrap()
    }

    /// Qwen3-VL-8B's own GGUF header (`Qwen/Qwen3-VL-8B-Instruct-GGUF`):
    /// `general.architecture = qwen3vl` and every hyperparameter under that
    /// prefix, including the 5e6 RoPE base the stock Qwen3 files do not use.
    #[test]
    fn qwen3vl_metadata_sets_the_architecture() {
        let mut metadata = HashMap::new();
        let s = |v: &str| gguf_file::Value::String(v.to_string());
        metadata.insert("general.architecture".to_string(), s("qwen3vl"));
        metadata.insert("qwen3vl.block_count".into(), gguf_file::Value::U32(36));
        metadata.insert(
            "qwen3vl.attention.head_count".into(),
            gguf_file::Value::U32(32),
        );
        metadata.insert(
            "qwen3vl.attention.head_count_kv".into(),
            gguf_file::Value::U32(8),
        );
        metadata.insert(
            "qwen3vl.attention.key_length".into(),
            gguf_file::Value::U32(128),
        );
        metadata.insert(
            "qwen3vl.rope.freq_base".into(),
            gguf_file::Value::F32(5_000_000.0),
        );
        metadata.insert(
            "qwen3vl.attention.layer_norm_rms_epsilon".into(),
            gguf_file::Value::F32(1e-6),
        );
        let arch = GgufQwen3Arch::from_metadata(&metadata).unwrap();
        assert_eq!(arch.n_layers, 36);
        assert_eq!(arch.rope_theta, 5_000_000.0);
        assert_eq!(arch.rms_norm_eps as f32, 1e-6);

        // A stock Qwen3 file (only `qwen3.block_count`) keeps every default,
        // and an empty header is exactly the historical Qwen3-4B constants.
        let mut stock = HashMap::new();
        stock.insert("qwen3.block_count".into(), gguf_file::Value::U32(1));
        let stock = GgufQwen3Arch::from_metadata(&stock).unwrap();
        assert_eq!(
            stock,
            GgufQwen3Arch {
                n_layers: 1,
                ..GgufQwen3Arch::default()
            }
        );
        assert_eq!(stock.rope_theta, 1_000_000.0);

        // An inconsistent header is refused rather than built.
        let mut bad = HashMap::new();
        bad.insert(
            "qwen3.attention.head_count_kv".into(),
            gguf_file::Value::U32(7),
        );
        assert!(GgufQwen3Arch::from_metadata(&bad).is_err());
    }

    /// The RoPE base is live: the same weights under a different
    /// `rope.freq_base` produce different hidden states.
    #[test]
    fn the_rope_base_reaches_the_forward() {
        let (tensors, mut metadata) = synthetic_checkpoint();
        let mut base =
            GgufQwen3Encoder::from_tensors(tensors.clone(), metadata.clone(), &Device::Cpu)
                .unwrap();
        metadata.insert(
            "qwen3.rope.freq_base".into(),
            gguf_file::Value::F32(5_000_000.0),
        );
        let mut vl = GgufQwen3Encoder::from_tensors(tensors, metadata, &Device::Cpu).unwrap();
        assert_eq!(vl.arch().rope_theta, 5_000_000.0);
        let ids = Tensor::from_vec(vec![1u32, 5, 9, 2, 7], (1, 5), &Device::Cpu).unwrap();
        let a = to_rows(
            &base
                .forward_final_pre_norm_with_attention(&ids, None)
                .unwrap(),
        );
        let b = to_rows(
            &vl.forward_final_pre_norm_with_attention(&ids, None)
                .unwrap(),
        );
        assert_eq!(max_diff(&a[..1], &b[..1]), 0.0, "position 0 is theta-free");
        assert!(
            max_diff(&a, &b) > 1e-6,
            "later positions must move with theta"
        );
    }

    /// The pre-norm forward runs EVERY layer: on the synthetic one-block
    /// checkpoint it is the output of layer 0, which `forward_with_layers`
    /// exposes independently.
    #[test]
    fn the_pre_norm_forward_is_the_last_layers_output() {
        let mut encoder = tiny_encoder();
        let ids = Tensor::from_vec(vec![3u32, 1, 4, 1, 5], (1, 5), &Device::Cpu).unwrap();
        let pre_norm = encoder
            .forward_final_pre_norm_with_attention(&ids, None)
            .unwrap();
        let layer0 = encoder.forward_with_layers(&ids, &[0], None).unwrap();
        assert_eq!(max_diff(&to_rows(&pre_norm), &to_rows(&layer0)), 0.0);
        // An all-real mask is the plain causal forward.
        let all = vec![vec![true; 5]];
        let masked = encoder
            .forward_final_pre_norm_with_attention(&ids, Some(&all))
            .unwrap();
        assert!(max_diff(&to_rows(&pre_norm), &to_rows(&masked)) < 1e-6);
    }

    /// Qwen Image 2.1 left-pads its prompt batches. The real tokens of a
    /// padded row must see exactly what an unpadded run of the same tokens
    /// sees: pad keys masked, and RoPE positions restarting at zero.
    #[test]
    fn a_left_padded_row_matches_the_unpadded_prompt() {
        let mut encoder = tiny_encoder();
        let prompt = [7u32, 2, 9];
        let plain = Tensor::from_vec(prompt.to_vec(), (1, 3), &Device::Cpu).unwrap();
        let expected = to_rows(
            &encoder
                .forward_final_pre_norm_with_attention(&plain, None)
                .unwrap(),
        );
        let padded = Tensor::from_vec(vec![0u32, 0, 7, 2, 9], (1, 5), &Device::Cpu).unwrap();
        let rows = vec![vec![false, false, true, true, true]];
        let actual = to_rows(
            &encoder
                .forward_final_pre_norm_with_attention(&padded, Some(&rows))
                .unwrap(),
        );
        assert!(
            max_diff(&actual[2..], &expected) < 1e-4,
            "padded row diverged by {}",
            max_diff(&actual[2..], &expected)
        );
        // Batched: a second row with no padding is its own unpadded forward.
        let batch =
            Tensor::from_vec(vec![0u32, 0, 7, 2, 9, 4, 4, 7, 2, 9], (2, 5), &Device::Cpu).unwrap();
        let rows = vec![
            vec![false, false, true, true, true],
            vec![true, true, true, true, true],
        ];
        let out = encoder
            .forward_final_pre_norm_with_attention(&batch, Some(&rows))
            .unwrap();
        let first = out.narrow(0, 0, 1).unwrap();
        assert!(max_diff(&to_rows(&first)[2..], &expected) < 1e-4);
        assert!(encoder
            .forward_final_pre_norm_with_attention(&batch, Some(&rows[..1]))
            .is_err());
    }

    /// With no visual rows and equal T/H/W positions, the multimodal forward
    /// is the text forward.
    #[test]
    fn a_text_only_multimodal_forward_is_the_text_forward() {
        let mut encoder = tiny_encoder();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (1, 6), &Device::Cpu).unwrap();
        let text = to_rows(
            &encoder
                .forward_final_pre_norm_with_attention(&ids, None)
                .unwrap(),
        );
        let positions: Vec<u32> = (0..6).collect();
        let mrope = [positions.clone(), positions.clone(), positions];
        let multimodal = to_rows(
            &encoder
                .forward_multimodal_final_pre_norm(&ids, None, &mrope)
                .unwrap(),
        );
        assert!(max_diff(&text, &multimodal) < 1e-5);
    }

    /// Visual rows and DeepStack are causal: nothing before the first image
    /// pad moves, and the pad rows themselves do.
    #[test]
    fn visual_injection_moves_only_the_image_rows_and_what_follows() {
        let mut encoder = tiny_encoder();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 3, 3, 4], (1, 6), &Device::Cpu).unwrap();
        let t: Vec<u32> = vec![0, 1, 2, 2, 2, 3];
        let h: Vec<u32> = vec![0, 1, 2, 2, 3, 3];
        let w: Vec<u32> = vec![0, 1, 2, 3, 2, 3];
        let mrope = [t, h, w];
        let plain = to_rows(
            &encoder
                .forward_multimodal_final_pre_norm(&ids, None, &mrope)
                .unwrap(),
        );
        let rows = |salt: f32| {
            Tensor::from_vec(
                (0..3 * TEST_DIM)
                    .map(|i| ((i as f32) * 0.01 + salt).sin() * 0.1)
                    .collect::<Vec<_>>(),
                (3, TEST_DIM),
                &Device::Cpu,
            )
            .unwrap()
        };
        let visual = VisualInjection {
            positions: vec![2, 3, 4],
            embeds: rows(0.3),
            deepstack: vec![rows(0.7)],
        };
        let injected = to_rows(
            &encoder
                .forward_multimodal_final_pre_norm(&ids, Some(visual.clone()), &mrope)
                .unwrap(),
        );
        assert_eq!(max_diff(&plain[..2], &injected[..2]), 0.0);
        assert!(max_diff(&plain[2..], &injected[2..]) > 1e-4);
        // More DeepStack features than layers is refused.
        let too_many = VisualInjection {
            deepstack: vec![rows(0.1), rows(0.2)],
            ..visual
        };
        assert!(encoder
            .forward_multimodal_final_pre_norm(&ids, Some(too_many), &mrope)
            .is_err());
    }

    /// The multimodal forward's per-chunk causal bias is exactly the rows
    /// `start..start + rows` of the full causal mask, cut at the last key
    /// those rows can see — never the full `L x L` host mask.
    #[test]
    fn the_per_chunk_causal_bias_is_the_full_masks_visible_block() {
        let (seq, start, rows) = (11, 4, 3);
        let full = causal_mask(seq, DType::F32, &Device::Cpu).unwrap();
        let expected = full
            .narrow(2, start, rows)
            .unwrap()
            .narrow(3, 0, start + rows)
            .unwrap();
        let bias = causal_chunk_bias(start, rows, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(bias.dims(), [1, 1, rows, start + rows]);
        assert_eq!(
            bias.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // The keys past the chunk are all masked in the full mask, so
        // dropping them changes no softmax.
        let past = full
            .narrow(2, start, rows)
            .unwrap()
            .narrow(3, start + rows, seq - start - rows)
            .unwrap();
        assert!(past
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == f32::NEG_INFINITY));
    }

    /// Every row of `many` within 1e-4 of its own peak in `one`. A chunk's
    /// GEMM may take a different CPU kernel from the whole-sequence one (a
    /// last-ulp score difference), and the sin-patterned synthetic weights sum
    /// coherently over 4096 lanes, amplifying that ulp to ~1e-4 of a row's
    /// peak; a masking or chunk-boundary error moves a row by O(1).
    fn assert_rows_agree(path: &str, one: &[Vec<f32>], many: &[Vec<f32>]) {
        assert_eq!(one.len(), many.len());
        for (row, (a, b)) in one.iter().zip(many).enumerate() {
            let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let diff = max_diff(std::slice::from_ref(a), std::slice::from_ref(b));
            assert!(
                diff <= 1e-4 * peak.max(1.0),
                "{path} path: chunked row {row} diverges by {diff} (peak {peak})"
            );
        }
    }

    /// Chunking the attention's query rows is the same arithmetic, for EVERY
    /// row: a prompt split into several chunks (the last one ragged) against
    /// the same prompt run as one chunk, on the text path and on the
    /// multimodal path (whose causal bias is built per chunk).
    #[test]
    fn chunked_attention_matches_one_chunk() {
        let seq = 3 * 16 + 5;
        let ids = Tensor::from_vec(
            (0..seq as u32)
                .map(|i| (i * 7 + 3) % TEST_VOCAB as u32)
                .collect::<Vec<_>>(),
            (1, seq),
            &Device::Cpu,
        )
        .unwrap();
        let mut whole = dense_tiny_encoder();
        whole.set_query_chunk(usize::MAX);
        let mut chunked = dense_tiny_encoder();
        chunked.set_query_chunk(16);

        let one = to_rows(
            &whole
                .forward_final_pre_norm_with_attention(&ids, None)
                .unwrap(),
        );
        let many = to_rows(
            &chunked
                .forward_final_pre_norm_with_attention(&ids, None)
                .unwrap(),
        );
        assert_eq!(many.len(), seq);
        assert_rows_agree("text", &one, &many);

        // Unequal T/H/W axes and a visual block that straddles a chunk edge.
        let t: Vec<u32> = (0..seq as u32)
            .map(|i| if (10..22).contains(&i) { 10 } else { i })
            .collect();
        let h: Vec<u32> = (0..seq as u32)
            .map(|i| {
                if (10..22).contains(&i) {
                    10 + (i - 10) / 4
                } else {
                    i
                }
            })
            .collect();
        let w: Vec<u32> = (0..seq as u32)
            .map(|i| {
                if (10..22).contains(&i) {
                    10 + (i - 10) % 4
                } else {
                    i
                }
            })
            .collect();
        let mrope = [t, h, w];
        let visual = VisualInjection {
            positions: (10..22).collect(),
            embeds: Tensor::from_vec(
                (0..12 * TEST_DIM)
                    .map(|i| ((i as f32) * 0.013).sin() * 0.1)
                    .collect::<Vec<_>>(),
                (12, TEST_DIM),
                &Device::Cpu,
            )
            .unwrap(),
            deepstack: vec![],
        };
        let one = to_rows(
            &whole
                .forward_multimodal_final_pre_norm(&ids, Some(visual.clone()), &mrope)
                .unwrap(),
        );
        let many = to_rows(
            &chunked
                .forward_multimodal_final_pre_norm(&ids, Some(visual), &mrope)
                .unwrap(),
        );
        assert_eq!(many.len(), seq);
        assert_rows_agree("multimodal", &one, &many);
    }
}
