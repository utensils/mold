//! Native BF16 Qwen3 encoder for safetensors weights.
//!
//! Supports Qwen3-4B (hidden_size=2560, used by Klein-4B and Z-Image),
//! Qwen3-8B (hidden_size=4096, used by Klein-9B), and the Qwen3-VL language
//! submodel of Qwen/Qwen-Image-2.1 (hidden_size=4096, rope_theta=5e6,
//! tensors rooted at `model.language_model`) via `Qwen3BF16Config`.
//!
//! Provides `forward_with_layers()` for multi-layer hidden state extraction,
//! required by Flux.2 Klein (layers 9, 18, 27 → stacked embeddings), and
//! `forward_final_pre_norm()` which runs every decoder layer and returns the
//! final pre-final-RMSNorm hidden states consumed by the Qwen Image 2.1
//! Diffusers reference.
//!
//! Shared architecture (4B/8B and the Image 2.1 language submodel):
//! 36 layers, 32 Q heads, 8 KV heads (GQA 4:1), 128 head_dim, RMSNorm eps=1e-6.
//! Differs per variant: hidden_size, intermediate_size, RoPE theta, max
//! positions, and the safetensors tensor root.
//!
//! HuggingFace safetensors weight names (stock Qwen3, root `model`; the
//! Qwen-Image-2.1 language submodel uses the same suffixes under the
//! `model.language_model` root):
//! - `model.embed_tokens.weight`
//! - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! - `model.layers.{i}.self_attn.{q,k}_norm.weight`
//! - `model.layers.{i}.input_layernorm.weight`
//! - `model.layers.{i}.post_attention_layernorm.weight`
//! - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

// ── Per-variant configuration ───────────────────────────────────────────────

/// Full architecture configuration for the BF16 Qwen3 loader.
///
/// Legacy Qwen3-4B/8B defaults (shared: 36 layers, 32/8 heads, head_dim 128,
/// RoPE theta 1e6, RMSNorm eps 1e-6, vocab 151936, max positions 40960,
/// tensor root `model`) are produced by [`Self::qwen3_4b`] / [`Self::qwen3_8b`]
/// and must not change; the Qwen-Image-2.1 language submodel overrides theta,
/// max positions, and the tensor root via [`Self::qwen3_image_21_text_encoder`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen3BF16Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    /// Dotted prefix of the safetensors tensor names, e.g. `model` for stock
    /// Qwen3 checkpoints or `model.language_model` for the Qwen3-VL language
    /// submodel shipped inside Qwen/Qwen-Image-2.1.
    pub tensor_root: &'static str,
}

impl Qwen3BF16Config {
    /// Shared stock-Qwen3 geometry (everything except the two MLP/hidden sizes).
    fn legacy(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 151_936,
            max_position_embeddings: 40_960,
            tensor_root: "model",
        }
    }

    /// Qwen3-4B: used by Flux.2 Klein-4B and Z-Image.
    pub fn qwen3_4b() -> Self {
        Self::legacy(2560, 9728)
    }

    /// Qwen3-8B: used by Flux.2 Klein-9B.
    pub fn qwen3_8b() -> Self {
        Self::legacy(4096, 12288)
    }

    /// Qwen/Qwen-Image-2.1 official Qwen3-VL language submodel
    /// (per the checkpoint's `config.json`, `text_config`):
    /// hidden 4096, intermediate 12288, 36 layers, 32 Q / 8 KV heads,
    /// head_dim 128, vocab 151936, RMS eps 1e-6, RoPE theta 5e6,
    /// 262144 max positions, tensors rooted at `model.language_model`.
    ///
    /// Used by the native Qwen Image 2.1 pipeline.
    pub fn qwen3_image_21_text_encoder() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 151_936,
            max_position_embeddings: 262_144,
            tensor_root: "model.language_model",
        }
    }

    /// Reject configs the loader would otherwise mis-index on.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.num_hidden_layers > 0, "num_hidden_layers must be > 0");
        anyhow::ensure!(
            self.num_attention_heads > 0,
            "num_attention_heads must be > 0"
        );
        anyhow::ensure!(self.num_kv_heads > 0, "num_kv_heads must be > 0");
        anyhow::ensure!(
            self.num_attention_heads.is_multiple_of(self.num_kv_heads),
            "num_attention_heads ({}) must be a multiple of num_kv_heads ({})",
            self.num_attention_heads,
            self.num_kv_heads
        );
        anyhow::ensure!(self.head_dim > 0, "head_dim must be > 0");
        anyhow::ensure!(self.hidden_size > 0, "hidden_size must be > 0");
        anyhow::ensure!(self.intermediate_size > 0, "intermediate_size must be > 0");
        anyhow::ensure!(self.vocab_size > 0, "vocab_size must be > 0");
        anyhow::ensure!(
            self.max_position_embeddings > 0,
            "max_position_embeddings must be > 0"
        );
        anyhow::ensure!(
            !self.tensor_root.is_empty(),
            "tensor_root must not be empty"
        );
        Ok(())
    }
}

/// Descend a dotted tensor root (e.g. `model.language_model`) one path
/// segment at a time so the safetensors key join matches the checkpoint.
fn root_builder<'a>(vb: VarBuilder<'a>, root: &str) -> VarBuilder<'a> {
    root.split('.')
        .filter(|seg| !seg.is_empty())
        .fold(vb, |acc, seg| acc.pp(seg))
}

// ── Rotary Embedding ────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Qwen3BF16Config, dtype: DType, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply one shared contiguous RoPE position range to q, k tensors of
    /// shape `(B, H, L, D)`.
    ///
    /// Stock Qwen3 callers use this path. Qwen3-VL's left-padded prompt
    /// batches instead use [`Self::apply_positions`]: its valid tokens start
    /// at position zero per sample, not at their physical offset in the
    /// padded tensor.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Apply per-row RoPE positions to `(B, H, L, D)` Q/K tensors.
    ///
    /// `Qwen3VLForConditionalGeneration.get_rope_index` derives positions
    /// from `attention_mask.cumsum(-1) - 1` and assigns padded positions the
    /// harmless value `1`.  A left-padded prompt therefore has valid tokens
    /// at `0..N`, even though they live at physical columns `P..P+N`.  The
    /// ordinary shared-offset kernel above cannot express that batch-specific
    /// layout; Candle's RoPE does support a `[B, L, D/2]` cos/sin table.
    fn apply_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _, sequence, _) = q.dims4()?;
        anyhow::ensure!(
            positions.len() == batch,
            "Qwen3 RoPE position rows mismatch: expected {batch}, got {}",
            positions.len()
        );
        let mut cos_rows = Vec::with_capacity(batch);
        let mut sin_rows = Vec::with_capacity(batch);
        for (row_index, row) in positions.iter().enumerate() {
            anyhow::ensure!(
                row.len() == sequence,
                "Qwen3 RoPE position row {row_index} has {}, expected {sequence}",
                row.len()
            );
            anyhow::ensure!(
                row.iter()
                    .all(|position| *position < self.cos.dim(0).unwrap_or(0)),
                "Qwen3 RoPE position row {row_index} exceeds the configured position table"
            );
            let positions = Tensor::from_vec(
                row.iter()
                    .map(|position| u32::try_from(*position))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                sequence,
                q.device(),
            )?;
            cos_rows.push(self.cos.index_select(&positions, 0)?);
            sin_rows.push(self.sin.index_select(&positions, 0)?);
        }
        let cos_refs = cos_rows.iter().collect::<Vec<_>>();
        let sin_refs = sin_rows.iter().collect::<Vec<_>>();
        let cos = Tensor::stack(&cos_refs, 0)?.contiguous()?;
        let sin = Tensor::stack(&sin_refs, 0)?.contiguous()?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
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
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                hidden_size,
                intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(
                intermediate_size,
                hidden_size,
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
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_repeat: usize,
}

impl Attention {
    fn new(
        cfg: &Qwen3BF16Config,
        rotary_emb: std::sync::Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;
        let q_norm = RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_attention_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            kv_repeat: cfg.num_attention_heads / cfg.num_kv_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        positions: Option<&[Vec<usize>]>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let (h_q, h_kv, d) = (self.num_attention_heads, self.num_kv_heads, self.head_dim);

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to (B, L, H, D) then transpose to (B, H, L, D)
        let q = q.reshape((b, l, h_q, d))?.transpose(1, 2)?;
        let k = k.reshape((b, l, h_kv, d))?.transpose(1, 2)?;
        let v = v.reshape((b, l, h_kv, d))?.transpose(1, 2)?;

        // Per-head RMSNorm (flatten batch+heads, norm, reshape back)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, h_q, l, d))?;
        let k = k_flat.reshape((b, h_kv, l, d))?;

        // RoPE
        let (q, k) = match positions {
            Some(positions) => self.rotary_emb.apply_positions(&q, &k, positions)?,
            None => self.rotary_emb.apply(&q, &k, 0)?,
        };

        // GQA repeat KV
        let k = repeat_kv(k, self.kv_repeat)?.contiguous()?;
        let v = repeat_kv(v, self.kv_repeat)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn_weights.matmul(&v)?;

        // Output projection
        ctx.transpose(1, 2)?
            .reshape((b, l, h_q * d))?
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
    fn new(
        cfg: &Qwen3BF16Config,
        rotary_emb: std::sync::Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?;
        let mlp = SwiGluMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        positions: Option<&[Vec<usize>]>,
    ) -> Result<Tensor> {
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, mask, positions)?;
        let xs = (xs + h)?;
        let h = self.post_attention_layernorm.forward(&xs)?;
        let h = self.mlp.forward(&h)?;
        (xs + h).map_err(Into::into)
    }
}

// ── Bf16Qwen3Encoder ───────────────────────────────────────────────────────

/// Native BF16 Qwen3 encoder with multi-layer hidden state extraction.
///
/// Supports Qwen3-4B (hidden_size=2560), Qwen3-8B (hidden_size=4096), and the
/// Qwen-Image-2.1 language submodel (nested `model.language_model` tensor
/// root, rope_theta=5e6) via `Qwen3BF16Config`.
pub(crate) struct Bf16Qwen3Encoder {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    device: Device,
    dtype: DType,
}

impl Bf16Qwen3Encoder {
    /// Load from HuggingFace safetensors files under `cfg.tensor_root`.
    pub fn load(cfg: &Qwen3BF16Config, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let vb_model = root_builder(vb, cfg.tensor_root);

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(cfg, dtype, &device)?);

        let vb_layers = vb_model.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary_emb.clone(), vb_layers.pp(i))?);
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

        let target_layer = self.layers.len().saturating_sub(2); // layer 34 for the 36-layer stock models

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref(), None)?;
            if i == target_layer {
                return Ok(hidden_states);
            }
        }

        Ok(hidden_states)
    }

    /// Run all decoder layers and return the final hidden states BEFORE the
    /// model's final RMSNorm (which the consuming pipeline applies itself).
    ///
    /// This is the state the Qwen Image 2.1 Diffusers reference explicitly
    /// consumes; unlike [`Self::forward`] (penultimate layer, unchanged
    /// legacy semantics) no layer is skipped.
    #[allow(dead_code)] // retained as the unmasked native-encoder API
    pub fn forward_final_pre_norm(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_final_pre_norm_with_attention(input_ids, None)
    }

    /// [`Self::forward_final_pre_norm`] with a per-batch-row key visibility
    /// mask.
    ///
    /// `attention[b][key]` marks the real (non-pad) positions of batch row `b`
    /// in a fixed-width prompt. When present, the additive causal mask also
    /// excludes padded KEYS, matching the mask the Diffusers reference hands
    /// the Qwen3 language model, and is applied even at `L == 1`. With `None`
    /// the legacy causal-mask behavior (omitted at `L == 1`) is preserved.
    pub(crate) fn forward_final_pre_norm_with_attention(
        &self,
        input_ids: &Tensor,
        attention: Option<&[Vec<bool>]>,
    ) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let mask = match attention {
            Some(rows) => {
                if rows.len() != b {
                    anyhow::bail!(
                        "Qwen3 attention mask mismatch: {b} batch row(s) expected, got {}",
                        rows.len()
                    );
                }
                for (row, flags) in rows.iter().enumerate() {
                    if flags.len() != l {
                        anyhow::bail!(
                            "Qwen3 attention mask mismatch: batch row {row} covers {} \
                             positions but the input has {l}",
                            flags.len()
                        );
                    }
                }
                Some(Self::batch_attention_mask(
                    rows,
                    l,
                    self.dtype,
                    &self.device,
                )?)
            }
            None if l == 1 => None,
            None => Some(self.causal_mask(b, l)?),
        };

        // Qwen3-VL's text-only `get_rope_index` derives each row's positions
        // from the attention mask. This matters only for the 2.1 left-padded
        // prompt batches; existing stock-Qwen3 callers keep their historical
        // shared 0..L positions.
        let positions = attention
            .map(|rows| Self::rope_positions_for_attention(rows, l))
            .transpose()?;

        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref(), positions.as_deref())?;
        }

        Ok(hidden_states)
    }

    /// Additive `(B, 1, L, L)` causal + key-padding mask.
    ///
    /// A key contributes (`0`) only if it is at or before the query position
    /// and its `rows[b][key]` flag is true; every other slot is
    /// `f32::NEG_INFINITY`. When a query would otherwise have no visible key
    /// at all, its entire row is unmasked instead. This is Transformers'
    /// `unmask_unattended` safety rule for SDPA: the query belongs to padding
    /// and is discarded by the consumer, but letting softmax see an all-`-inf`
    /// row creates NaNs which can poison later layers before that discard.
    fn batch_attention_mask(
        rows: &[Vec<bool>],
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = rows
            .iter()
            .flat_map(|flags| {
                (0..seq_len).flat_map(move |i| {
                    let has_causal_key = flags.iter().take(i + 1).any(|valid| *valid);
                    (0..seq_len).map(move |j| {
                        if !has_causal_key || (j <= i && flags[j]) {
                            0.0
                        } else {
                            minf
                        }
                    })
                })
            })
            .collect();
        Tensor::from_slice(&mask, (rows.len(), 1, seq_len, seq_len), device)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }

    /// Qwen3-VL's text-only MRoPE coordinates after left/right padding.
    ///
    /// This is the exact `attention_mask.cumsum(-1) - 1` rule from
    /// `Qwen3VLForConditionalGeneration.get_rope_index`, including its
    /// explicit pad coordinate of one. When all three MRoPE axes carry these
    /// same text coordinates, the interleaving reduces to normal Qwen3 RoPE;
    /// what remains load-bearing is that each row's valid first token starts
    /// at zero rather than at a left-pad offset.
    fn rope_positions_for_attention(rows: &[Vec<bool>], seq_len: usize) -> Result<Vec<Vec<usize>>> {
        rows.iter()
            .enumerate()
            .map(|(row_index, flags)| {
                anyhow::ensure!(
                    flags.len() == seq_len,
                    "Qwen3 attention mask mismatch: batch row {row_index} covers {} positions but the input has {seq_len}",
                    flags.len()
                );
                let mut next_position = 0usize;
                Ok(flags
                    .iter()
                    .map(|valid| {
                        if *valid {
                            let position = next_position;
                            next_position += 1;
                            position
                        } else {
                            // Transformers sets padded slots to one after
                            // cumsum. Those queries are discarded later, but
                            // retaining the value keeps their execution
                            // numerically aligned with the reference.
                            1
                        }
                    })
                    .collect())
            })
            .collect()
    }

    /// Run forward pass and collect hidden states from specific layers.
    /// Returns (B, seq_len, num_layers * hidden_size).
    /// Used by Flux.2 Klein which stacks layers 9, 18, 27 → 7680-dim embeddings.
    ///
    /// `attention` names the real (non-pad) positions of a fixed-width
    /// prompt. When present the causal mask additionally excludes the padded
    /// KEYS, which is the mask BFL hands the Qwen3 language model
    /// (`flux2/text_encoder.py:408-416`).
    pub fn forward_with_layers(
        &self,
        input_ids: &Tensor,
        layer_indices: &[usize],
        attention: Option<&[bool]>,
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

        let mask = match attention {
            Some(attention) => {
                if attention.len() != l {
                    anyhow::bail!(
                        "Qwen3 attention mask covers {} positions but the input has {l}",
                        attention.len()
                    );
                }
                Some(super::qwen3::causal_padding_mask(
                    attention,
                    self.dtype,
                    &self.device,
                )?)
            }
            None if l == 1 => None,
            None => Some(self.causal_mask(b, l)?),
        };

        let n_run = max_layer + 1;
        let mut collected: Vec<Tensor> = Vec::with_capacity(layer_indices.len());

        for (i, layer) in self.layers[..n_run].iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref(), None)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use std::collections::HashMap;

    #[test]
    fn legacy_configs_keep_every_stock_default() {
        for cfg in [Qwen3BF16Config::qwen3_4b(), Qwen3BF16Config::qwen3_8b()] {
            assert_eq!(cfg.num_hidden_layers, 36);
            assert_eq!(cfg.num_attention_heads, 32);
            assert_eq!(cfg.num_kv_heads, 8);
            assert_eq!(cfg.head_dim, 128);
            assert_eq!(cfg.rope_theta, 1_000_000.0);
            assert_eq!(cfg.rms_norm_eps, 1e-6);
            assert_eq!(cfg.vocab_size, 151_936);
            assert_eq!(cfg.max_position_embeddings, 40_960);
            assert_eq!(cfg.tensor_root, "model");
            cfg.validate().unwrap();
        }
        assert_eq!(Qwen3BF16Config::qwen3_4b().hidden_size, 2560);
        assert_eq!(Qwen3BF16Config::qwen3_4b().intermediate_size, 9728);
        assert_eq!(Qwen3BF16Config::qwen3_8b().hidden_size, 4096);
        assert_eq!(Qwen3BF16Config::qwen3_8b().intermediate_size, 12288);
    }

    #[test]
    fn image_21_text_encoder_matches_official_config() {
        let cfg = Qwen3BF16Config::qwen3_image_21_text_encoder();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        // The three fields that deviate from stock Qwen3:
        assert_eq!(cfg.rope_theta, 5_000_000.0);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        assert_eq!(cfg.tensor_root, "model.language_model");
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_rejects_broken_configs() {
        let mut cfg = Qwen3BF16Config::qwen3_4b();
        cfg.num_kv_heads = 0;
        assert!(cfg.validate().is_err());
        let mut cfg = Qwen3BF16Config::qwen3_4b();
        cfg.num_attention_heads = 30; // not a multiple of 8
        assert!(cfg.validate().is_err());
        let mut cfg = Qwen3BF16Config::qwen3_4b();
        cfg.tensor_root = "";
        assert!(cfg.validate().is_err());
    }

    // ── Tiny synthetic model for loader-wiring tests ────────────────────────

    fn tiny_cfg(root: &'static str) -> Qwen3BF16Config {
        Qwen3BF16Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            max_position_embeddings: 16,
            tensor_root: root,
        }
    }

    fn tiny_weights(root: &str) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let h = 16usize;
        let inter = 32usize;
        let q_proj = 4 * 8; // num_attention_heads * head_dim
        let kv_proj = 2 * 8; // num_kv_heads * head_dim
        let one = |dims: (usize, usize)| Tensor::ones(dims, DType::F32, &dev).unwrap();
        let ones = |n: usize| Tensor::ones((n,), DType::F32, &dev).unwrap();

        let mut w = HashMap::new();
        w.insert(format!("{root}.embed_tokens.weight"), one((32, h)));
        for i in 0..2 {
            let p = format!("{root}.layers.{i}");
            w.insert(format!("{p}.self_attn.q_proj.weight"), one((q_proj, h)));
            w.insert(format!("{p}.self_attn.k_proj.weight"), one((kv_proj, h)));
            w.insert(format!("{p}.self_attn.v_proj.weight"), one((kv_proj, h)));
            w.insert(format!("{p}.self_attn.o_proj.weight"), one((h, q_proj)));
            w.insert(format!("{p}.self_attn.q_norm.weight"), ones(8));
            w.insert(format!("{p}.self_attn.k_norm.weight"), ones(8));
            w.insert(format!("{p}.input_layernorm.weight"), ones(h));
            w.insert(format!("{p}.post_attention_layernorm.weight"), ones(h));
            w.insert(format!("{p}.mlp.gate_proj.weight"), one((inter, h)));
            w.insert(format!("{p}.mlp.up_proj.weight"), one((inter, h)));
            w.insert(format!("{p}.mlp.down_proj.weight"), one((h, inter)));
        }
        w
    }

    #[test]
    fn loads_nested_language_model_root() {
        let cfg = tiny_cfg("model.language_model");
        let vb = VarBuilder::from_tensors(
            tiny_weights("model.language_model"),
            DType::F32,
            &Device::Cpu,
        );
        let enc = Bf16Qwen3Encoder::load(&cfg, vb).unwrap();

        let ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();

        // Final pre-final-norm state: all layers run, full hidden width.
        let final_pre = enc.forward_final_pre_norm(&ids).unwrap();
        assert_eq!(final_pre.dims3().unwrap(), (1, 4, 16));
        assert!(final_pre
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));

        // Legacy penultimate semantics still hold on the same weights.
        let pen = enc.forward(&ids).unwrap();
        assert_eq!(pen.dims3().unwrap(), (1, 4, 16));
    }

    #[test]
    fn stock_root_rejects_language_model_prefixed_weights() {
        // A `model`-rooted config must NOT find tensors stored under
        // `model.language_model`, and vice versa the nested config needs the
        // nested keys (proven by the failing load below).
        let cfg = tiny_cfg("model");
        let vb = VarBuilder::from_tensors(
            tiny_weights("model.language_model"),
            DType::F32,
            &Device::Cpu,
        );
        assert!(Bf16Qwen3Encoder::load(&cfg, vb).is_err());

        let cfg = tiny_cfg("model.language_model");
        let vb = VarBuilder::from_tensors(tiny_weights("model"), DType::F32, &Device::Cpu);
        assert!(Bf16Qwen3Encoder::load(&cfg, vb).is_err());
    }

    #[test]
    fn rotary_derives_theta_and_max_length_from_config() {
        let dev = Device::Cpu;
        let base = tiny_cfg("model");
        let mut hi = base;
        hi.rope_theta = 5_000_000.0;
        hi.max_position_embeddings = 128;

        let a = RotaryEmbedding::new(&base, DType::F32, &dev).unwrap();
        let b = RotaryEmbedding::new(&hi, DType::F32, &dev).unwrap();

        // Max length flows through to the table size.
        assert_eq!(a.cos.dim(0).unwrap(), 16);
        assert_eq!(b.cos.dim(0).unwrap(), 128);

        // Lowest-frequency column (inv_freq[0] = 1.0) is theta-independent;
        // the next column must shift with theta.
        let a0: f32 = a.cos.get(1).unwrap().get(0).unwrap().to_vec0().unwrap();
        let b0: f32 = b.cos.get(1).unwrap().get(0).unwrap().to_vec0().unwrap();
        assert!((a0 - b0).abs() < 1e-6);
        let a1: f32 = a.cos.get(1).unwrap().get(1).unwrap().to_vec0().unwrap();
        let b1: f32 = b.cos.get(1).unwrap().get(1).unwrap().to_vec0().unwrap();
        assert!((a1 - b1).abs() > 1e-4);
    }

    // ── Mask-aware final-pre-norm route (Qwen Image 2.1 left padding) ───────

    fn tiny_encoder(root: &'static str) -> Bf16Qwen3Encoder {
        let cfg = tiny_cfg(root);
        let vb = VarBuilder::from_tensors(tiny_weights(root), DType::F32, &Device::Cpu);
        Bf16Qwen3Encoder::load(&cfg, vb).unwrap()
    }

    /// Two left-padded rows in one batch: row 0 has 2 leading pads, row 1 has
    /// 1. The batched mask must be (2, 1, L, L) and each row's `-inf` pattern
    ///    must follow its own pad layout, not a shared one.
    #[test]
    fn batch_attention_mask_excludes_each_rows_left_pads() {
        let rows: Vec<Vec<bool>> = vec![
            vec![false, false, true, true, true],
            vec![false, true, true, true, true],
        ];
        let mask =
            Bf16Qwen3Encoder::batch_attention_mask(&rows, 5, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[2, 1, 5, 5]);

        let r0 = mask.i((0, 0)).unwrap().to_vec2::<f32>().unwrap();
        // Row 0's two leading pads are invisible to every REAL query in row
        // 0. Fully padded query rows themselves are intentionally unmasked
        // so their softmax stays finite (they are discarded later).
        for (query, row) in r0.iter().enumerate().skip(2) {
            assert_eq!(row[0], f32::NEG_INFINITY);
            assert_eq!(row[1], f32::NEG_INFINITY);
            // ...while the causal floor still holds: query 2 sees only key 2.
            assert_eq!(row[2], if 2 <= query { 0.0 } else { f32::NEG_INFINITY });
            assert_eq!(row[3], if 3 <= query { 0.0 } else { f32::NEG_INFINITY });
            assert_eq!(row[4], if 4 <= query { 0.0 } else { f32::NEG_INFINITY });
        }

        let r1 = mask.i((1, 0)).unwrap().to_vec2::<f32>().unwrap();
        // Row 1 has a different left-pad count: key 0 is masked but key 1 is
        // visible from query 1 onward — the rows are NOT interchangeable.
        for row in r1.iter().skip(1) {
            assert_eq!(row[0], f32::NEG_INFINITY);
        }
        assert_eq!(r1[1][1], 0.0);
        assert_eq!(r1[4][1], 0.0);
    }

    #[test]
    fn qwen3_vl_rope_positions_restart_after_left_padding() {
        let positions = Bf16Qwen3Encoder::rope_positions_for_attention(
            &[
                vec![false, false, true, true, true],
                vec![false, true, true, true, true],
            ],
            5,
        )
        .unwrap();
        // This is `attention_mask.cumsum(-1) - 1`, with transformers'
        // `masked_fill_(attention_mask == 0, 1)` applied afterwards. The
        // first genuine token must be position zero in BOTH rows.
        assert_eq!(positions[0], vec![1, 1, 0, 1, 2]);
        assert_eq!(positions[1], vec![1, 0, 1, 2, 3]);
    }

    /// The mask route runs even at L == 1, where the no-mask route skips the
    /// mask entirely (a single-position key mask is vacuously causal, but a
    /// left-padded... zero-valid-key row must still be expressible).
    #[test]
    fn mask_route_applies_at_single_position() {
        let enc = tiny_encoder("model");
        let ids = Tensor::from_slice(&[7u32], (1, 1), &Device::Cpu).unwrap();

        let valid = enc
            .forward_final_pre_norm_with_attention(&ids, Some(&[vec![true]]))
            .unwrap();
        assert_eq!(valid.dims3().unwrap(), (1, 1, 16));
        assert!(valid
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));

        let masked_out = enc
            .forward_final_pre_norm_with_attention(&ids, Some(&[vec![false]]))
            .unwrap();
        assert_eq!(masked_out.dims3().unwrap(), (1, 1, 16));
        assert!(masked_out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// Two batches with different left-pad positions run as one forward pass.
    #[test]
    fn mask_route_runs_two_differently_padded_rows() {
        let enc = tiny_encoder("model.language_model");
        let ids =
            Tensor::from_slice(&[1u32, 1, 4, 5, 6, 1, 7, 8, 9, 10], (2, 5), &Device::Cpu).unwrap();
        let rows: Vec<Vec<bool>> = vec![
            vec![false, false, true, true, true],
            vec![false, true, true, true, true],
        ];

        let out = enc
            .forward_final_pre_norm_with_attention(&ids, Some(&rows))
            .unwrap();
        assert_eq!(out.dims3().unwrap(), (2, 5, 16));
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// Malformed masks are rejected against input_ids' (B, L), with errors
    /// naming the mismatch.
    #[test]
    fn mask_route_rejects_malformed_masks() {
        let enc = tiny_encoder("model");
        let ids = Tensor::from_slice(&[3u32, 4, 5, 6], (2, 2), &Device::Cpu).unwrap();

        let err = enc
            .forward_final_pre_norm_with_attention(&ids, Some(&[vec![true, true]]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Qwen3 attention mask mismatch") && err.contains("batch row"),
            "batch-count rejection must name the mismatch: {err}"
        );

        let err = enc
            .forward_final_pre_norm_with_attention(
                &ids,
                Some(&[vec![true, true], vec![true, true, true]]),
            )
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Qwen3 attention mask mismatch") && err.contains("positions"),
            "row-width rejection must name the mismatch and the offending row: {err}"
        );
        assert!(
            err.contains("row 1"),
            "the failing row is identified: {err}"
        );
    }

    /// The no-mask route keeps its exact previous behavior: the public entry
    /// point is now a delegation, and None must be bit-identical to what
    /// forward_final_pre_norm produced (pure causal, mask omitted at L == 1).
    #[test]
    fn no_mask_route_is_preserved() {
        let enc = tiny_encoder("model");
        let ids = Tensor::from_slice(&[2u32, 3, 4, 5], (1, 4), &Device::Cpu).unwrap();

        let via_public = enc.forward_final_pre_norm(&ids).unwrap();
        let via_none = enc
            .forward_final_pre_norm_with_attention(&ids, None)
            .unwrap();
        assert_eq!(
            via_public.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            via_none.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // A fully-valid per-row mask is equivalent to the pure causal mask.
        let all_valid = enc
            .forward_final_pre_norm_with_attention(&ids, Some(&[vec![true; 4]]))
            .unwrap();
        assert_eq!(
            via_public.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            all_valid.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // L == 1 on the no-mask route still runs (mask omitted there).
        let one = Tensor::from_slice(&[9u32], (1, 1), &Device::Cpu).unwrap();
        assert_eq!(
            enc.forward_final_pre_norm(&one).unwrap().dims3().unwrap(),
            (1, 1, 16)
        );
    }
}
