//! Quantized Gemma 3 12B encoder loader for GGUF files (llama.cpp `gemma3-text` arch).
//!
//! Mirrors [`crate::ltx2::text::encoder::GemmaHiddenStateEncoder`] in shape contract
//! so [`crate::ltx2::text::prompt_encoder::NativePromptEncoder`] can plug it in
//! behind the [`crate::ltx2::text::GemmaHiddenStateBackend`] enum. Returns
//! `Vec<Tensor>` of hidden states with `len() == num_hidden_layers + 1`:
//! embedding × √hidden_size, then every layer's output (except the last), then
//! the final RMSNorm output.
//!
//! Gemma 3 quirks vs. Qwen3 (the closest sibling in
//! `crates/mold-inference/src/encoders/qwen3_gguf.rs`):
//! - `xs * sqrt(hidden_size)` immediately after embedding lookup
//! - RMSNorm applies `(weight + 1.0)` instead of `weight`
//! - **Four** norms per block: `attn_norm` (input_layernorm), `post_attention_norm`,
//!   `ffn_norm` (pre_feedforward_layernorm), `post_ffw_norm`
//! - Sliding-window attention every layer where `(i + 1) % sliding_window_pattern != 0`,
//!   with a different RoPE base (`rope_local_base_freq`) for sliding layers vs.
//!   `rope_theta` × global linear scaling factor (8) for full-attention layers
//! - Per-head Q/K norms (dim = head_dim)
//! - GeluPytorchTanh activation (not Silu)
//!
//! GGUF tensor names (verified against `google/gemma-3-12b-it-qat-q4_0-gguf`):
//! - `token_embd.weight` (F16 in published file; loader handles any GGML dtype)
//! - `output_norm.weight` (F32)
//! - `blk.{i}.attn_norm.weight` / `post_attention_norm.weight` (F32)
//! - `blk.{i}.ffn_norm.weight` / `post_ffw_norm.weight` (F32)
//! - `blk.{i}.attn_q.weight` / `attn_k.weight` / `attn_v.weight` / `attn_output.weight` (Q4_0)
//! - `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` (F32)
//! - `blk.{i}.ffn_gate.weight` / `ffn_up.weight` / `ffn_down.weight` (Q4_0)

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{Activation, Module};
use candle_transformers::models::with_tracing::QMatMul;

use super::gemma::PromptTokens;

// ── Gemma 3 12B architecture constants (verified against the GGUF metadata) ──

/// `sliding_window_pattern` is not present in the GGUF metadata. The official
/// Gemma 3 12B config sets it to 6 — every 6th layer uses full attention with
/// global RoPE; the others use sliding-window attention with local RoPE. Same
/// value as the BF16 path's `ltx_gemma_config()`.
const SLIDING_WINDOW_PATTERN: usize = 6;

/// `rope_local_base_freq` is also not present in GGUF metadata. The Gemma 3
/// reference uses 10 000 for sliding-window layers vs. `rope.freq_base`
/// (typically 1e6) for full-attention layers. Same value as
/// `ltx_gemma_config().rope_local_base_freq`.
const ROPE_LOCAL_BASE_FREQ: f64 = 10_000.0;

/// Saturating value for masked-out attention scores; matches `MASK_NEGATIVE`
/// in `super::encoder` so cross-backend numerics line up.
const MASK_NEGATIVE: f32 = -1e30;

/// Read at load-time from `gemma3.*` metadata keys. Lifetime-tied to
/// [`GgufGemmaEncoder`] so the forward path doesn't re-parse the file.
#[derive(Debug, Clone)]
struct GgufGemmaConfig {
    block_count: usize,
    embedding_length: usize,
    attention_head_count: usize,
    attention_head_count_kv: usize,
    attention_key_length: usize,
    attention_sliding_window: usize,
    rms_norm_eps: f64,
    rope_freq_base: f64,
    rope_scaling_factor: f64,
    context_length: usize,
}

impl GgufGemmaConfig {
    fn from_metadata(content: &gguf_file::Content) -> Result<Self> {
        let metadata = &content.metadata;
        let get_u32 = |key: &str| -> Result<u32> {
            metadata
                .get(key)
                .and_then(|v| match v {
                    gguf_file::Value::U32(n) => Some(*n),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("missing or non-u32 GGUF metadata key '{key}'"))
        };
        let get_f32 = |key: &str| -> Result<f32> {
            metadata
                .get(key)
                .and_then(|v| match v {
                    gguf_file::Value::F32(x) => Some(*x),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("missing or non-f32 GGUF metadata key '{key}'"))
        };

        Ok(Self {
            block_count: get_u32("gemma3.block_count")? as usize,
            embedding_length: get_u32("gemma3.embedding_length")? as usize,
            attention_head_count: get_u32("gemma3.attention.head_count")? as usize,
            attention_head_count_kv: get_u32("gemma3.attention.head_count_kv")? as usize,
            attention_key_length: get_u32("gemma3.attention.key_length")? as usize,
            attention_sliding_window: get_u32("gemma3.attention.sliding_window")? as usize,
            rms_norm_eps: get_f32("gemma3.attention.layer_norm_rms_epsilon")? as f64,
            rope_freq_base: get_f32("gemma3.rope.freq_base")? as f64,
            rope_scaling_factor: get_f32("gemma3.rope.scaling.factor")? as f64,
            context_length: get_u32("gemma3.context_length")? as usize,
        })
    }

    fn head_dim(&self) -> usize {
        self.attention_key_length
    }

    fn num_kv_groups(&self) -> usize {
        self.attention_head_count / self.attention_head_count_kv
    }
}

// ── RMSNorm with the (weight + 1.0) Gemma rule ───────────────────────────────

#[derive(Debug)]
struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    fn from_qtensor(qt: Arc<QTensor>, device: &Device, eps: f64) -> Result<Self> {
        let weight = qt.dequantize(device)?;
        Ok(Self { weight, eps })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let input_dtype = xs.dtype();
        let internal = match input_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden = xs.dim(D::Minus1)?;
        let xs_f = xs.to_dtype(internal)?;
        let variance = (xs_f.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let normed = xs_f.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed
            .to_dtype(input_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

// ── Rotary embeddings (precomputed; one set per RoPE flavour) ────────────────

#[derive(Debug)]
struct GemmaRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl GemmaRotaryEmbedding {
    /// `freq_base` is `rope_freq_base` for global layers and
    /// `ROPE_LOCAL_BASE_FREQ` for sliding-window layers; `scaling_factor` is
    /// `rope_scaling_factor` for global layers and `1.0` for sliding-window
    /// layers (matching `super::encoder::RotaryEmbedding::new`).
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        freq_base: f64,
        scaling_factor: f64,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| ((1.0 / freq_base.powf(i as f64 / head_dim as f64)) / scaling_factor) as f32)
            .collect();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, _heads, seq, _) = q.dims4()?;
        let (pos_batch, pos_seq) = position_ids.dims2()?;
        if pos_batch != batch || pos_seq != seq {
            bail!(
                "Gemma3 GGUF rotary position_ids shape mismatch: expected [{batch}, {seq}], got [{pos_batch}, {pos_seq}]"
            );
        }
        let position_ids = position_ids.to_dtype(DType::U32)?.flatten_all()?;
        let cos =
            self.cos
                .index_select(&position_ids, 0)?
                .reshape((batch, seq, self.cos.dim(1)?))?;
        let sin =
            self.sin
                .index_select(&position_ids, 0)?
                .reshape((batch, seq, self.sin.dim(1)?))?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

// ── Attention block ──────────────────────────────────────────────────────────

#[derive(Debug)]
struct GgufGemmaAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl GgufGemmaAttention {
    fn forward(
        &self,
        xs: &Tensor,
        cos_sin: &GemmaRotaryEmbedding,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = cos_sin.apply(&q, &k, position_ids)?;
        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = match attention_mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        Ok(probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)?)
    }
}

// ── FFN with GeluPytorchTanh-gated SwiGLU ───────────────────────────────────

#[derive(Debug)]
struct GgufGemmaFfn {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl GgufGemmaFfn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_out = Activation::GeluPytorchTanh.forward(&self.gate.forward(xs)?)?;
        let up_out = self.up.forward(xs)?;
        self.down.forward(&(gate_out * up_out)?).map_err(Into::into)
    }
}

// ── Decoder block (four norms; sliding-window per index) ─────────────────────

#[derive(Debug)]
struct GgufGemmaBlock {
    attn_norm: GemmaRmsNorm,           // ← input_layernorm
    post_attention_norm: GemmaRmsNorm, // applied to attention output before residual add
    ffn_norm: GemmaRmsNorm,            // ← pre_feedforward_layernorm
    post_ffw_norm: GemmaRmsNorm,       // applied to FFN output before residual add
    self_attn: GgufGemmaAttention,
    ffn: GgufGemmaFfn,
    /// `Some(window)` for sliding-window layers, `None` for global-attention layers.
    sliding_window: Option<usize>,
}

impl GgufGemmaBlock {
    fn forward(
        &self,
        xs: &Tensor,
        rotary_global: &GemmaRotaryEmbedding,
        rotary_local: &GemmaRotaryEmbedding,
        full_mask: Option<&Tensor>,
        sliding_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (rotary, mask) = if self.sliding_window.is_some() {
            (rotary_local, sliding_mask)
        } else {
            (rotary_global, full_mask)
        };

        let residual = xs;
        let normed = self.attn_norm.forward(xs)?;
        let attn_out = self
            .self_attn
            .forward(&normed, rotary, mask, position_ids)?;
        let attn_out = self.post_attention_norm.forward(&attn_out)?;
        let xs = (residual + attn_out)?;

        let residual = &xs;
        let normed = self.ffn_norm.forward(&xs)?;
        let ffn_out = self.ffn.forward(&normed)?;
        let ffn_out = self.post_ffw_norm.forward(&ffn_out)?;
        Ok((residual + ffn_out)?)
    }
}

// ── Top-level encoder ────────────────────────────────────────────────────────

pub struct GgufGemmaEncoder {
    cfg: GgufGemmaConfig,
    embedding: candle_nn::Embedding,
    output_norm: GemmaRmsNorm,
    blocks: Vec<GgufGemmaBlock>,
    rotary_global: GemmaRotaryEmbedding,
    rotary_local: GemmaRotaryEmbedding,
    device: Device,
}

impl GgufGemmaEncoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("failed to open Gemma3 GGUF '{}'", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to parse Gemma3 GGUF '{}'", path.display()))?;
        let cfg = GgufGemmaConfig::from_metadata(&content)?;

        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::new();
        for name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, name, device).with_context(|| {
                format!(
                    "failed to read tensor '{name}' from Gemma3 GGUF '{}'",
                    path.display()
                )
            })?;
            tensors.insert(name.clone(), Arc::new(tensor));
        }

        let take = |name: &str| -> Result<Arc<QTensor>> {
            tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("Gemma3 GGUF missing tensor '{name}'"))
        };

        // Embedding (dequantize once on the target device)
        let emb_q = take("token_embd.weight")?;
        let emb = emb_q.dequantize(device)?;
        let d_model = emb.dim(1)?;
        let embedding = candle_nn::Embedding::new(emb, d_model);

        // Final RMSNorm
        let output_norm =
            GemmaRmsNorm::from_qtensor(take("output_norm.weight")?, device, cfg.rms_norm_eps)?;

        // RoPE precomputations — one per flavour
        let max_seq = cfg.context_length.max(1);
        let head_dim = cfg.head_dim();
        let rotary_global = GemmaRotaryEmbedding::new(
            head_dim,
            max_seq,
            cfg.rope_freq_base,
            cfg.rope_scaling_factor,
            device,
        )?;
        let rotary_local =
            GemmaRotaryEmbedding::new(head_dim, max_seq, ROPE_LOCAL_BASE_FREQ, 1.0, device)?;

        // Per-block weights
        let mut blocks = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let prefix = format!("blk.{i}");
            let q_proj = QMatMul::from_weights(take(&format!("{prefix}.attn_q.weight"))?)?;
            let k_proj = QMatMul::from_weights(take(&format!("{prefix}.attn_k.weight"))?)?;
            let v_proj = QMatMul::from_weights(take(&format!("{prefix}.attn_v.weight"))?)?;
            let o_proj = QMatMul::from_weights(take(&format!("{prefix}.attn_output.weight"))?)?;

            let q_norm = GemmaRmsNorm::from_qtensor(
                take(&format!("{prefix}.attn_q_norm.weight"))?,
                device,
                cfg.rms_norm_eps,
            )?;
            let k_norm = GemmaRmsNorm::from_qtensor(
                take(&format!("{prefix}.attn_k_norm.weight"))?,
                device,
                cfg.rms_norm_eps,
            )?;
            let attn_norm = GemmaRmsNorm::from_qtensor(
                take(&format!("{prefix}.attn_norm.weight"))?,
                device,
                cfg.rms_norm_eps,
            )?;
            let post_attention_norm = GemmaRmsNorm::from_qtensor(
                take(&format!("{prefix}.post_attention_norm.weight"))?,
                device,
                cfg.rms_norm_eps,
            )?;
            let ffn_norm = GemmaRmsNorm::from_qtensor(
                take(&format!("{prefix}.ffn_norm.weight"))?,
                device,
                cfg.rms_norm_eps,
            )?;
            let post_ffw_norm = GemmaRmsNorm::from_qtensor(
                take(&format!("{prefix}.post_ffw_norm.weight"))?,
                device,
                cfg.rms_norm_eps,
            )?;

            let gate = QMatMul::from_weights(take(&format!("{prefix}.ffn_gate.weight"))?)?;
            let up = QMatMul::from_weights(take(&format!("{prefix}.ffn_up.weight"))?)?;
            let down = QMatMul::from_weights(take(&format!("{prefix}.ffn_down.weight"))?)?;

            // Sliding-window layers are every layer where (i+1) % pattern != 0;
            // every 6th layer (i+1 == 6, 12, 18, ...) uses full attention.
            let uses_sliding = !(i + 1).is_multiple_of(SLIDING_WINDOW_PATTERN);
            let sliding_window = uses_sliding.then_some(cfg.attention_sliding_window);

            blocks.push(GgufGemmaBlock {
                attn_norm,
                post_attention_norm,
                ffn_norm,
                post_ffw_norm,
                self_attn: GgufGemmaAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    num_heads: cfg.attention_head_count,
                    num_kv_heads: cfg.attention_head_count_kv,
                    num_kv_groups: cfg.num_kv_groups(),
                    head_dim,
                },
                ffn: GgufGemmaFfn { gate, up, down },
                sliding_window,
            });
        }

        Ok(Self {
            cfg,
            embedding,
            output_norm,
            blocks,
            rotary_global,
            rotary_local,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Hidden size of the loaded encoder. Used by future code paths (PR B's
    /// residency tracking) to size CPU-park buffers without re-reading the
    /// GGUF metadata.
    #[allow(dead_code)]
    pub fn hidden_size(&self) -> usize {
        self.cfg.embedding_length
    }

    /// Layer count of the loaded encoder. Same future-use rationale as
    /// [`Self::hidden_size`].
    #[allow(dead_code)]
    pub fn num_hidden_layers(&self) -> usize {
        self.cfg.block_count
    }

    pub fn encode_prompt_tokens(
        &self,
        tokens: &PromptTokens,
    ) -> Result<super::encoder::GemmaHiddenStates> {
        let input_ids = Tensor::new(tokens.input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.attention_mask.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden_states = self.forward_hidden_states(&input_ids, &attention_mask)?;
        Ok(super::encoder::GemmaHiddenStates {
            hidden_states,
            attention_mask,
        })
    }

    pub fn forward_hidden_states(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let (batch, seq) = input_ids.dims2()?;
        let mut xs = self.embedding.forward(input_ids)?;
        xs = (xs * (self.cfg.embedding_length as f64).sqrt())?;
        let mut hidden_states = Vec::with_capacity(self.cfg.block_count + 1);
        hidden_states.push(xs.clone());

        let position_ids = build_position_ids(attention_mask)?;
        let dtype = xs.dtype();
        let full_mask = build_attention_mask(attention_mask, None, dtype, &self.device)?;
        let sliding_mask = build_attention_mask(
            attention_mask,
            Some(self.cfg.attention_sliding_window),
            dtype,
            &self.device,
        )?;

        let last_layer_index = self.cfg.block_count.saturating_sub(1);
        for (index, block) in self.blocks.iter().enumerate() {
            xs = block
                .forward(
                    &xs,
                    &self.rotary_global,
                    &self.rotary_local,
                    Some(&full_mask),
                    Some(&sliding_mask),
                    &position_ids,
                )
                .with_context(|| format!("Gemma3 GGUF block {index} failed"))?;
            if index != last_layer_index {
                hidden_states.push(xs.clone());
            }
        }

        xs = self
            .output_norm
            .forward(&xs)
            .context("Gemma3 GGUF final RMSNorm failed")?;
        hidden_states.push(xs);

        if hidden_states
            .iter()
            .any(|state| state.dims3().ok() != Some((batch, seq, self.cfg.embedding_length)))
        {
            bail!("Gemma3 GGUF encoder produced inconsistent hidden-state shapes");
        }
        Ok(hidden_states)
    }
}

// ── Mask & position helpers (mirror super::encoder) ──────────────────────────

fn build_position_ids(attention_mask: &Tensor) -> Result<Tensor> {
    let (batch, seq) = attention_mask.dims2()?;
    let device = attention_mask.device().clone();
    let mut ids: Vec<u32> = Vec::with_capacity(batch * seq);
    for _row in 0..batch {
        for position in 0..seq {
            ids.push(position as u32);
        }
    }
    Tensor::from_vec(ids, (batch, seq), &device).map_err(Into::into)
}

fn build_attention_mask(
    attention_mask: &Tensor,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq) = attention_mask.dims2()?;
    let key_mask = attention_mask
        .to_dtype(DType::F32)?
        .reshape((batch, 1, 1, seq))?;
    let invalid_keys = (key_mask.ones_like()? - &key_mask)?.affine(MASK_NEGATIVE as f64, 0.0)?;
    let causal = build_causal_mask(seq, sliding_window, device)?;
    Ok(causal.broadcast_add(&invalid_keys)?.to_dtype(dtype)?)
}

fn build_causal_mask(seq: usize, sliding_window: Option<usize>, device: &Device) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(seq * seq);
    for query in 0..seq {
        for key in 0..seq {
            let is_future = key > query;
            let outside_window = sliding_window.is_some_and(|window| key + window < query);
            mask.push(if is_future || outside_window {
                MASK_NEGATIVE
            } else {
                0.0
            });
        }
    }
    Tensor::from_vec(mask, (1, 1, seq, seq), device).map_err(Into::into)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use std::io::Cursor;

    /// Tiny synthetic config that fits Q4_0's 32-element block size.
    /// Real Gemma 3 12B has hidden=3840, head_dim=256, n_layers=48 — too large
    /// for a unit test. Q4_0 requires inner dim divisible by 32, so hidden=32,
    /// head_dim=8, intermediate=32 are the smallest values that quantize.
    struct TinyCfg {
        block_count: usize,
        embedding_length: usize,
        feed_forward_length: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        rope_theta: f32,
        rope_scaling: f32,
    }

    impl TinyCfg {
        fn small(block_count: usize) -> Self {
            Self {
                block_count,
                embedding_length: 32,
                feed_forward_length: 32,
                n_heads: 2,
                n_kv_heads: 1,
                head_dim: 8,
                sliding_window: 4,
                rope_theta: 10_000.0,
                rope_scaling: 8.0,
            }
        }
    }

    fn quantize_or_fail(t: &Tensor, dtype: GgmlDType) -> Arc<QTensor> {
        Arc::new(QTensor::quantize(t, dtype).expect("quantize"))
    }

    fn build_synthetic_gguf(cfg: &TinyCfg, vocab: usize, embed_init: Tensor) -> Vec<u8> {
        let device = Device::Cpu;
        let zeros_norm = || Tensor::zeros(cfg.embedding_length, DType::F32, &device).unwrap();
        let zeros_head_norm = || Tensor::zeros(cfg.head_dim, DType::F32, &device).unwrap();

        let metadata: Vec<(&str, gguf_file::Value)> = vec![
            (
                "gemma3.block_count",
                gguf_file::Value::U32(cfg.block_count as u32),
            ),
            (
                "gemma3.embedding_length",
                gguf_file::Value::U32(cfg.embedding_length as u32),
            ),
            (
                "gemma3.feed_forward_length",
                gguf_file::Value::U32(cfg.feed_forward_length as u32),
            ),
            (
                "gemma3.attention.head_count",
                gguf_file::Value::U32(cfg.n_heads as u32),
            ),
            (
                "gemma3.attention.head_count_kv",
                gguf_file::Value::U32(cfg.n_kv_heads as u32),
            ),
            (
                "gemma3.attention.key_length",
                gguf_file::Value::U32(cfg.head_dim as u32),
            ),
            (
                "gemma3.attention.sliding_window",
                gguf_file::Value::U32(cfg.sliding_window as u32),
            ),
            (
                "gemma3.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F32(1e-6),
            ),
            (
                "gemma3.rope.freq_base",
                gguf_file::Value::F32(cfg.rope_theta),
            ),
            (
                "gemma3.rope.scaling.factor",
                gguf_file::Value::F32(cfg.rope_scaling),
            ),
            ("gemma3.context_length", gguf_file::Value::U32(64)),
        ];

        // Hold owned QTensors so references stay alive until write completes.
        let mut owned: Vec<(String, Arc<QTensor>)> = Vec::new();

        owned.push((
            "token_embd.weight".to_string(),
            quantize_or_fail(&embed_init, GgmlDType::F32),
        ));
        let _ = vocab; // shape carried by `embed_init`.
        owned.push((
            "output_norm.weight".to_string(),
            quantize_or_fail(&zeros_norm(), GgmlDType::F32),
        ));

        for i in 0..cfg.block_count {
            let prefix = format!("blk.{i}");
            // Linear weights (output_dim, input_dim) — Q4_0 needs inner dim % 32 == 0.
            let q_w = Tensor::zeros(
                (cfg.n_heads * cfg.head_dim, cfg.embedding_length),
                DType::F32,
                &device,
            )
            .unwrap();
            let kv_w = Tensor::zeros(
                (cfg.n_kv_heads * cfg.head_dim, cfg.embedding_length),
                DType::F32,
                &device,
            )
            .unwrap();
            let o_w = Tensor::zeros(
                (cfg.embedding_length, cfg.n_heads * cfg.head_dim),
                DType::F32,
                &device,
            )
            .unwrap();
            let gate_w = Tensor::zeros(
                (cfg.feed_forward_length, cfg.embedding_length),
                DType::F32,
                &device,
            )
            .unwrap();
            let up_w = Tensor::zeros(
                (cfg.feed_forward_length, cfg.embedding_length),
                DType::F32,
                &device,
            )
            .unwrap();
            let down_w = Tensor::zeros(
                (cfg.embedding_length, cfg.feed_forward_length),
                DType::F32,
                &device,
            )
            .unwrap();

            owned.push((
                format!("{prefix}.attn_q.weight"),
                quantize_or_fail(&q_w, GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.attn_k.weight"),
                quantize_or_fail(&kv_w, GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.attn_v.weight"),
                quantize_or_fail(&kv_w, GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.attn_output.weight"),
                quantize_or_fail(&o_w, GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.ffn_gate.weight"),
                quantize_or_fail(&gate_w, GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.ffn_up.weight"),
                quantize_or_fail(&up_w, GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.ffn_down.weight"),
                quantize_or_fail(&down_w, GgmlDType::F32),
            ));

            owned.push((
                format!("{prefix}.attn_q_norm.weight"),
                quantize_or_fail(&zeros_head_norm(), GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.attn_k_norm.weight"),
                quantize_or_fail(&zeros_head_norm(), GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.attn_norm.weight"),
                quantize_or_fail(&zeros_norm(), GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.post_attention_norm.weight"),
                quantize_or_fail(&zeros_norm(), GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.ffn_norm.weight"),
                quantize_or_fail(&zeros_norm(), GgmlDType::F32),
            ));
            owned.push((
                format!("{prefix}.post_ffw_norm.weight"),
                quantize_or_fail(&zeros_norm(), GgmlDType::F32),
            ));
        }

        let metadata_refs: Vec<(&str, &gguf_file::Value)> =
            metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensor_refs: Vec<(&str, &QTensor)> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_ref()))
            .collect();

        let mut buf = Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &metadata_refs, &tensor_refs).expect("write gguf");
        buf.into_inner()
    }

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut tmp = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .expect("tempfile");
        tmp.write_all(bytes).expect("write tempfile");
        tmp.flush().expect("flush");
        tmp
    }

    /// Embedding lookup must scale by `√hidden_size` before going through any
    /// other layers — it's the first user-visible Gemma quirk that diverges
    /// from the Qwen3 reference.
    #[test]
    fn embedding_scales_by_sqrt_hidden_size() {
        let cfg = TinyCfg::small(0);
        let vocab = 4;
        // Embedding row 1 = [1, 0, 0, ...]; rows 0/2/3 zero.
        let mut emb = vec![0f32; vocab * cfg.embedding_length];
        emb[cfg.embedding_length] = 1.0; // token id 1, feature 0
        let embed_init =
            Tensor::from_vec(emb, (vocab, cfg.embedding_length), &Device::Cpu).unwrap();

        let bytes = build_synthetic_gguf(&cfg, vocab, embed_init);
        let tmp = write_temp_gguf(&bytes);

        let encoder = GgufGemmaEncoder::load(tmp.path(), &Device::Cpu).expect("load");
        let input_ids = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
        let attention_mask = Tensor::new(&[[1u8]], &Device::Cpu).unwrap();
        let hidden_states = encoder
            .forward_hidden_states(&input_ids, &attention_mask)
            .expect("forward");

        // Block count is 0 → loop body skipped, so we have [embedding, final_norm(embedding)].
        assert_eq!(hidden_states.len(), 2);

        let scaled = hidden_states[0].to_vec3::<f32>().unwrap();
        let expected = (cfg.embedding_length as f64).sqrt() as f32;
        assert!(
            (scaled[0][0][0] - expected).abs() < 1e-3,
            "expected feature 0 = √{} ≈ {}, got {}",
            cfg.embedding_length,
            expected,
            scaled[0][0][0]
        );
        for (f, value) in scaled[0][0].iter().enumerate().skip(1) {
            assert!(value.abs() < 1e-4, "feature {f} not zero");
        }
    }

    /// All-zero linear weights + zero norm weights mean every block forward is
    /// the residual-only identity (attn output = 0, ffn output = 0). Two-layer
    /// run must produce 3 hidden states all equal to the scaled embedding.
    #[test]
    fn multi_block_forward_with_zero_weights_is_identity_on_residual() {
        let cfg = TinyCfg::small(2);
        let vocab = 4;
        let mut emb = vec![0f32; vocab * cfg.embedding_length];
        emb[cfg.embedding_length] = 1.0;
        let embed_init =
            Tensor::from_vec(emb, (vocab, cfg.embedding_length), &Device::Cpu).unwrap();

        let bytes = build_synthetic_gguf(&cfg, vocab, embed_init);
        let tmp = write_temp_gguf(&bytes);

        let encoder = GgufGemmaEncoder::load(tmp.path(), &Device::Cpu).expect("load");
        let input_ids = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
        let attention_mask = Tensor::new(&[[1u8]], &Device::Cpu).unwrap();
        let hidden_states = encoder
            .forward_hidden_states(&input_ids, &attention_mask)
            .expect("forward");

        // Contract: len() == num_hidden_layers + 1
        assert_eq!(hidden_states.len(), cfg.block_count + 1);
        // Every state has the encoder's hidden_size as last dim
        for (i, state) in hidden_states.iter().enumerate() {
            let dims = state.dims3().expect("3D");
            assert_eq!(dims, (1, 1, cfg.embedding_length), "state {i} dims");
        }

        // With zero linear weights every block forward becomes
        // `residual + post_norm(self_attn(...) = 0)` → residual unchanged.
        // The scaled embedding is already RMS-1 (single feature = √32 over
        // 32 elements; mean(x²) = 1 → norm divides by 1 and `(weight+1)=1`),
        // so output_norm is also a no-op. Final state = scaled embedding.
        let expected = (cfg.embedding_length as f64).sqrt() as f32;
        let final_state = hidden_states.last().unwrap().to_vec3::<f32>().unwrap();
        assert!(
            (final_state[0][0][0] - expected).abs() < 1e-3,
            "expected √{} ≈ {} at feature 0, got {}",
            cfg.embedding_length,
            expected,
            final_state[0][0][0]
        );
        for (f, value) in final_state[0][0].iter().enumerate().skip(1) {
            assert!(value.abs() < 1e-3, "feature {f} drifted");
        }
    }

    /// Sliding-window layer assignment must match the BF16 reference
    /// (`super::encoder` uses `(i + 1) % sliding_window_pattern != 0`). With
    /// pattern=6: layers 0–4 sliding, layer 5 global, 6–10 sliding, 11 global,
    /// etc. Layer 47 is sliding (47+1=48; 48 % 6 == 0 → global). Pin this
    /// in a test so a future refactor can't silently flip it.
    #[test]
    fn sliding_window_assignment_matches_bf16_reference() {
        for i in 0..48usize {
            let uses_sliding = !(i + 1).is_multiple_of(SLIDING_WINDOW_PATTERN);
            let expected_global = (i + 1) % SLIDING_WINDOW_PATTERN == 0;
            assert_eq!(uses_sliding, !expected_global, "layer {i}");
        }
        // Layer 5 is global; layer 47 is global (48 % 6 == 0). Spot-check.
        assert!(6usize.is_multiple_of(SLIDING_WINDOW_PATTERN));
        assert!(48usize.is_multiple_of(SLIDING_WINDOW_PATTERN));
        assert!(!1usize.is_multiple_of(SLIDING_WINDOW_PATTERN));
    }

    /// Development-only: dump every tensor name in the Gemma 3 12B Q4 GGUF so
    /// we can reconcile against the table in `docs/superpowers/plans/
    /// 2026-05-09-ltx2-gemma-q4-gguf-and-parking.md`. The exact names depend on
    /// the llama.cpp converter version that produced the file; verifying once
    /// before writing the loader avoids tensor-not-found bugs at runtime.
    ///
    /// Run with:
    /// ```sh
    /// cargo test -p mold-ai-inference --lib \
    ///   ltx2::text::gemma3_gguf::tests::dump_tensor_names \
    ///   -- --ignored --nocapture
    /// ```
    /// The file is expected at
    /// `$HOME/.mold/models/shared/gemma3-12b-q4-gguf/gemma-3-12b-it-q4_0.gguf`;
    /// developers verifying a different path should edit this test locally
    /// before invoking it.
    #[test]
    #[ignore = "requires google/gemma-3-12b-it-qat-q4_0-gguf on disk"]
    fn dump_tensor_names() {
        let path = format!(
            "{}/.mold/models/shared/gemma3-12b-q4-gguf/gemma-3-12b-it-q4_0.gguf",
            std::env::var("HOME").expect("HOME must be set")
        );
        let mut file = File::open(&path).expect("open gguf");
        let content = gguf_file::Content::read(&mut file).expect("read gguf");

        println!("=== METADATA ===");
        let mut meta_keys: Vec<_> = content.metadata.keys().collect();
        meta_keys.sort();
        for key in meta_keys {
            let value = &content.metadata[key];
            let summary = match value {
                gguf_file::Value::U32(v) => format!("U32({v})"),
                gguf_file::Value::F32(v) => format!("F32({v})"),
                gguf_file::Value::String(v) => {
                    let trimmed: String = v.chars().take(80).collect();
                    format!("String({trimmed:?})")
                }
                gguf_file::Value::Array(items) => format!("Array(len={})", items.len()),
                other => format!("{other:?}"),
            };
            println!("{key} = {summary}");
        }

        println!("\n=== TENSOR NAMES (sorted, layer 0/1/47 + non-block) ===");
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in &names {
            let prefix_ok = name.starts_with("blk.0.")
                || name.starts_with("blk.1.")
                || name.starts_with("blk.47.")
                || !name.starts_with("blk.");
            if !prefix_ok {
                continue;
            }
            let info = &content.tensor_infos[name.as_str()];
            println!(
                "{name}  shape={:?}  dtype={:?}",
                info.shape.dims(),
                info.ggml_dtype
            );
        }
        println!("\n=== TOTAL TENSORS: {} ===", content.tensor_infos.len());
    }
}
