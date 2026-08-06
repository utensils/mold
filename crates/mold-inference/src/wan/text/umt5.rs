//! UMT5-XXL encoder for Wan 2.1/2.2.
//!
//! UMT5's distinguishing trait vs the T5-XXL stack mold already ships for
//! FLUX (`encoders/t5.rs` + `encoders/t5_gguf.rs`) is `shared_pos = False`:
//! **every layer owns its relative attention bias** and recomputes the
//! position bias from its own embedding, where T5 computes it once in block 0
//! and threads it through (upstream: `tmp/Wan2.1/wan/modules/t5.py:167-168`).
//! candle's `t5.rs` hard-codes the shared-bias layout, so this module is a
//! self-contained stack, following the `encoders/t5_gguf.rs` precedent.
//!
//! Two weight sources, one module tree:
//! - BF16/FP16 safetensors in HF T5 naming (`encoder.block.{i}.layer.0...`),
//!   the Comfy-Org `umt5_xxl_fp16.safetensors` repack (verified against its
//!   header: 24 blocks x 10 tensors, per-block `relative_attention_bias` of
//!   shape `[32, 64]`, `shared.weight` of `[256384, 4096]`).
//! - GGUF in city96 naming (`enc.blk.{i}.attn_rel_b.weight` present at every
//!   block for UMT5 exports such as `city96/umt5-xxl-encoder-gguf`).
//!
//! The output contract is load-bearing for Wan (upstream `t5.py:506-513`):
//! the encoder runs with the attention mask, and positions at or beyond each
//! prompt's true length are **zeroed** — the DiT must see embeddings of the
//! zero vector for padding, not T5 outputs for pad tokens.

use anyhow::{anyhow, Context, Result};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Wan's fixed text context length (`text_len` in every upstream config).
pub(crate) const WAN_TEXT_LEN: usize = 512;

/// UMT5 sentinel ids (`tmp/Wan2.1/wan/modules/tokenizers.py`: T5 convention).
const UMT5_PAD_ID: u32 = 0;
const UMT5_EOS_ID: u32 = 1;

/// Upstream tokenizes with `clean='whitespace'`
/// (`tmp/Wan2.1/wan/modules/tokenizers.py:49-73`): HTML-unescape twice,
/// collapse all whitespace runs to single spaces, and strip. Multiline or
/// copied-from-web prompts otherwise tokenize differently than they did in
/// training. Deliberate deviation: upstream also runs `ftfy.fix_text`
/// (mojibake repair); that transforms already-broken encodings, which we
/// accept as-is rather than carrying a transliteration table.
fn canonicalize_prompt(text: &str) -> String {
    fn unescape_html_once(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.char_indices();
        while let Some((i, c)) = chars.next() {
            if c != '&' {
                out.push(c);
                continue;
            }
            let rest = &s[i..];
            let Some(end) = rest[..rest.len().min(12)].find(';') else {
                out.push(c);
                continue;
            };
            let entity = &rest[1..end];
            let replacement = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                _ => entity
                    .strip_prefix("#x")
                    .or_else(|| entity.strip_prefix("#X"))
                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                    .or_else(|| entity.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                    .and_then(char::from_u32),
            };
            match replacement {
                Some(ch) => {
                    out.push(ch);
                    // Skip the consumed entity (chars is char_indices over s).
                    for _ in 0..entity.len() + 1 {
                        chars.next();
                    }
                }
                None => out.push(c),
            }
        }
        out
    }
    let unescaped = unescape_html_once(&unescape_html_once(text));
    let mut out = String::with_capacity(unescaped.len());
    let mut in_space = true; // leading whitespace is stripped
    for c in unescaped.chars() {
        if c.is_whitespace() {
            if !in_space {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(c);
            in_space = false;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Fit tokenizer output into the fixed window, preserving EOS. HF truncation
/// reserves the final slot for `</s>`; naively truncating after the
/// post-processor has appended it would drop EOS and treat 512 content
/// tokens as valid, changing the conditioning for long prompts.
fn fit_ids_to_window(mut ids: Vec<u32>) -> (Vec<u32>, usize) {
    if ids.len() > WAN_TEXT_LEN {
        ids.truncate(WAN_TEXT_LEN);
        ids[WAN_TEXT_LEN - 1] = UMT5_EOS_ID;
    }
    let len = ids.len();
    ids.resize(WAN_TEXT_LEN, UMT5_PAD_ID);
    (ids, len)
}

/// Additive key-mask value. Applied in f32 before softmax; large enough to
/// zero the weight without producing NaN on rows that keep valid keys (query
/// rows are never fully masked because every prompt has at least one token).
const MASK_NEG: f32 = -1e9;

/// UMT5 geometry. A struct rather than constants so tests can build tiny
/// models; [`UMt5Config::xxl`] is the only production configuration.
#[derive(Debug, Clone)]
pub(crate) struct UMt5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub eps: f64,
}

impl UMt5Config {
    /// UMT5-XXL as shipped by every Wan checkpoint
    /// (`tmp/Wan2.1/wan/modules/t5.py:456-469`).
    pub fn xxl() -> Self {
        Self {
            vocab_size: 256_384,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10_240,
            num_heads: 64,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            eps: 1e-6,
        }
    }
}

/// A linear projection backed by either plain tensors or a quantized GGUF
/// matmul (same split as `qwen_image`'s `QwenLinear`).
enum UmtLinear {
    Plain(candle_nn::Linear),
    Quant(QMatMul),
}

impl UmtLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => Ok(l.forward(xs)?),
            Self::Quant(q) => Ok(q.forward(xs)?),
        }
    }
}

/// T5-style RMS norm: no mean subtraction, no bias, computed in f32.
struct UmtLayerNorm {
    weight: Tensor,
    eps: f64,
}

impl UmtLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?)
    }
}

/// Gated FFN (T5 v1.1 / UMT5): `down(act(wi_0(x)) * wi_1(x))`.
struct UmtGatedFfn {
    gate: UmtLinear,
    up: UmtLinear,
    down: UmtLinear,
}

impl UmtGatedFfn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Upstream implements a tanh-approximated GELU by hand
        // (`t5.py:46-50`); NewGelu is that same approximation.
        let gated = candle_nn::Activation::NewGelu.forward(&self.gate.forward(xs)?)?;
        let linear = self.up.forward(xs)?;
        self.down.forward(&gated.broadcast_mul(&linear)?)
    }
}

/// Self-attention with a **per-layer** relative attention bias. T5
/// convention: no 1/sqrt(d) scaling.
struct UmtSelfAttention {
    q: UmtLinear,
    k: UmtLinear,
    v: UmtLinear,
    o: UmtLinear,
    n_heads: usize,
    d_kv: usize,
    /// This layer's own bias table, `[num_buckets, num_heads]`. Never shared
    /// and never optional — that is the UMT5 difference.
    relative_attention_bias: candle_nn::Embedding,
}

impl UmtSelfAttention {
    /// `relative_buckets` is the `[S, S]` bucket-index tensor, computed once
    /// per forward and shared by every layer (the bucket arithmetic is
    /// position-only; only the embedding lookup is per-layer).
    /// `key_mask` is an additive `[B, 1, 1, S]` f32 mask (0 valid, -1e9 pad).
    fn forward(
        &self,
        xs: &Tensor,
        relative_buckets: &Tensor,
        key_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let shape = (b, seq_len, self.n_heads, self.d_kv);
        let q = self
            .q
            .forward(xs)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(xs)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;

        // Bias and mask are added in f32 for numerical stability; upstream
        // runs the whole encoder under bf16 autocast but T5's unscaled logits
        // plus a -1e9 mask overflow f16, so the additive step stays f32.
        let mut scores = q.matmul(&k.t()?)?.to_dtype(DType::F32)?;
        let bias = self
            .relative_attention_bias
            .forward(relative_buckets)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;
        scores = scores.broadcast_add(&bias)?;
        if let Some(mask) = key_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        let out =
            attn.matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, seq_len, self.n_heads * self.d_kv))?;
        self.o.forward(&out)
    }
}

struct UmtBlock {
    attn_norm: UmtLayerNorm,
    self_attn: UmtSelfAttention,
    ffn_norm: UmtLayerNorm,
    ffn: UmtGatedFfn,
}

impl UmtBlock {
    fn forward(
        &self,
        xs: &Tensor,
        relative_buckets: &Tensor,
        key_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.attn_norm.forward(xs)?;
        let xs = (xs
            + self
                .self_attn
                .forward(&normed, relative_buckets, key_mask)?)?;
        let normed = self.ffn_norm.forward(&xs)?;
        Ok((&xs + self.ffn.forward(&normed)?)?)
    }
}

/// The UMT5 encoder stack.
pub(crate) struct UMt5Encoder {
    embedding: candle_nn::Embedding,
    blocks: Vec<UmtBlock>,
    final_norm: UmtLayerNorm,
    config: UMt5Config,
    device: Device,
}

/// Bidirectional relative-position bucket, HF `_relative_position_bucket`
/// with `bidirectional=True`. Kept as a free function so the golden test can
/// pin it against hand-computed values.
fn relative_position_bucket(
    query: usize,
    key: usize,
    num_buckets: usize,
    max_distance: usize,
) -> u32 {
    let half = (num_buckets / 2) as u32;
    let max_exact = half / 2;
    let (offset, distance) = if key > query {
        (half, (key - query) as u32)
    } else {
        (0, (query - key) as u32)
    };
    let bucket = if distance < max_exact {
        distance
    } else {
        let log_ratio = f32::ln(distance as f32 / max_exact as f32)
            / f32::ln(max_distance as f32 / max_exact as f32);
        let large = max_exact + (log_ratio * (half - max_exact) as f32) as u32;
        u32::min(large, half - 1)
    };
    offset + bucket
}

impl UMt5Encoder {
    /// Load from HF-named safetensors (Comfy-Org repack or a diffusers
    /// `text_encoder/` shard set). Accepts both bare keys and the full
    ///-checkpoint `text_encoders.umt5xxl.transformer.` prefix.
    pub fn from_safetensors(
        paths: &[PathBuf],
        config: UMt5Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
        let vb = if vb.contains_tensor("shared.weight") {
            vb
        } else {
            vb.pp("text_encoders.umt5xxl.transformer")
        };
        Self::from_var_builder(vb, config, device)
    }

    /// Build the stack from a VarBuilder rooted at the HF T5 layout. Split
    /// out so tests can construct tiny models from in-memory tensors.
    pub fn from_var_builder(vb: VarBuilder, config: UMt5Config, device: &Device) -> Result<Self> {
        let embedding = candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("shared"))
            .context("UMT5: missing shared.weight token embedding")?;
        let linear = |vb: VarBuilder, in_d: usize, out_d: usize, name: &str| -> Result<UmtLinear> {
            Ok(UmtLinear::Plain(
                candle_nn::linear_no_bias(in_d, out_d, vb)
                    .with_context(|| format!("UMT5: missing {name}"))?,
            ))
        };
        // `get_with_hints` so a VarMap-backed builder (tests) materializes
        // sane values — Const(1) is the identity for an RMS-norm gain. File-
        // backed builders ignore the init entirely.
        let norm = |vb: VarBuilder, name: &str| -> Result<UmtLayerNorm> {
            Ok(UmtLayerNorm {
                weight: vb
                    .get_with_hints(config.d_model, "weight", candle_nn::Init::Const(1.0))
                    .with_context(|| format!("UMT5: missing {name}"))?,
                eps: config.eps,
            })
        };

        let inner = config.num_heads * config.d_kv;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let vb_b = vb.pp(format!("encoder.block.{i}"));
            let vb_attn = vb_b.pp("layer.0.SelfAttention");
            // Per-layer bias is required, not optional: silently falling back
            // to a shared table would load mT5/T5 checkpoints as plausible
            // garbage.
            let rel_bias_weight = vb_attn
                .pp("relative_attention_bias")
                .get_with_hints(
                    (config.relative_attention_num_buckets, config.num_heads),
                    "weight",
                    candle_nn::Init::Randn {
                        mean: 0.0,
                        stdev: 0.5,
                    },
                )
                .with_context(|| {
                    format!(
                        "UMT5: encoder.block.{i} has no relative_attention_bias — every UMT5 \
                         layer owns one (shared_pos=False); this checkpoint looks like plain \
                         T5/mT5, which Wan cannot use"
                    )
                })?;
            let self_attn = UmtSelfAttention {
                q: linear(vb_attn.pp("q"), config.d_model, inner, "SelfAttention.q")?,
                k: linear(vb_attn.pp("k"), config.d_model, inner, "SelfAttention.k")?,
                v: linear(vb_attn.pp("v"), config.d_model, inner, "SelfAttention.v")?,
                o: linear(vb_attn.pp("o"), inner, config.d_model, "SelfAttention.o")?,
                n_heads: config.num_heads,
                d_kv: config.d_kv,
                relative_attention_bias: candle_nn::Embedding::new(
                    rel_bias_weight,
                    config.num_heads,
                ),
            };
            let vb_ffn = vb_b.pp("layer.1.DenseReluDense");
            let ffn = UmtGatedFfn {
                gate: linear(vb_ffn.pp("wi_0"), config.d_model, config.d_ff, "wi_0")?,
                up: linear(vb_ffn.pp("wi_1"), config.d_model, config.d_ff, "wi_1")?,
                down: linear(vb_ffn.pp("wo"), config.d_ff, config.d_model, "wo")?,
            };
            blocks.push(UmtBlock {
                attn_norm: norm(vb_b.pp("layer.0.layer_norm"), "layer.0.layer_norm")?,
                self_attn,
                ffn_norm: norm(vb_b.pp("layer.1.layer_norm"), "layer.1.layer_norm")?,
                ffn,
            });
        }
        let final_norm = norm(
            vb.pp("encoder.final_layer_norm"),
            "encoder.final_layer_norm",
        )?;
        Ok(Self {
            embedding,
            blocks,
            final_norm,
            config,
            device: device.clone(),
        })
    }

    /// Load from a city96-named GGUF (`enc.blk.{i}.*`, per-layer
    /// `attn_rel_b`). Mirrors `encoders/t5_gguf.rs`, minus the shared-bias
    /// threading.
    pub fn from_gguf(path: &Path, device: &Device) -> Result<Self> {
        let config = UMt5Config::xxl();
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("UMT5: cannot open GGUF at {}", path.display()))?;
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
                .ok_or_else(|| anyhow!("UMT5 GGUF: missing tensor {name}"))
        };

        let emb = get("token_embd.weight")?.dequantize(device)?;
        if emb.dim(0)? != config.vocab_size {
            return Err(anyhow!(
                "UMT5 GGUF: token_embd vocab is {} but UMT5-XXL has {} — this is a plain T5 \
                 encoder GGUF, which Wan cannot use",
                emb.dim(0)?,
                config.vocab_size
            ));
        }
        let embedding = candle_nn::Embedding::new(emb, config.d_model);

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("enc.blk.{i}");
            let rel = get(&format!("{p}.attn_rel_b.weight")).with_context(|| {
                format!(
                    "UMT5 GGUF: enc.blk.{i} has no attn_rel_b — every UMT5 layer owns one; \
                     this export looks like plain T5"
                )
            })?;
            let rel_weights = rel.dequantize(device)?;
            let self_attn = UmtSelfAttention {
                q: UmtLinear::Quant(QMatMul::from_weights(get(&format!("{p}.attn_q.weight"))?)?),
                k: UmtLinear::Quant(QMatMul::from_weights(get(&format!("{p}.attn_k.weight"))?)?),
                v: UmtLinear::Quant(QMatMul::from_weights(get(&format!("{p}.attn_v.weight"))?)?),
                o: UmtLinear::Quant(QMatMul::from_weights(get(&format!("{p}.attn_o.weight"))?)?),
                n_heads: config.num_heads,
                d_kv: config.d_kv,
                relative_attention_bias: candle_nn::Embedding::new(rel_weights, config.num_heads),
            };
            let ffn = UmtGatedFfn {
                gate: UmtLinear::Quant(QMatMul::from_weights(get(&format!(
                    "{p}.ffn_gate.weight"
                ))?)?),
                up: UmtLinear::Quant(QMatMul::from_weights(get(&format!("{p}.ffn_up.weight"))?)?),
                down: UmtLinear::Quant(QMatMul::from_weights(get(&format!(
                    "{p}.ffn_down.weight"
                ))?)?),
            };
            blocks.push(UmtBlock {
                attn_norm: UmtLayerNorm {
                    weight: get(&format!("{p}.attn_norm.weight"))?.dequantize(device)?,
                    eps: config.eps,
                },
                self_attn,
                ffn_norm: UmtLayerNorm {
                    weight: get(&format!("{p}.ffn_norm.weight"))?.dequantize(device)?,
                    eps: config.eps,
                },
                ffn,
            });
        }
        let final_norm = UmtLayerNorm {
            weight: get("enc.output_norm.weight")?.dequantize(device)?,
            eps: config.eps,
        };
        Ok(Self {
            embedding,
            blocks,
            final_norm,
            config,
            device: device.clone(),
        })
    }

    /// The `[S, S]` bucket-index tensor every layer shares. The arithmetic is
    /// position-only; each layer applies its own embedding to it.
    fn relative_buckets(&self, seq_len: usize) -> Result<Tensor> {
        let rows: Vec<Vec<u32>> = (0..seq_len)
            .map(|i| {
                (0..seq_len)
                    .map(|j| {
                        relative_position_bucket(
                            i,
                            j,
                            self.config.relative_attention_num_buckets,
                            self.config.relative_attention_max_distance,
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(Tensor::new(rows, &self.device)?)
    }

    /// Forward over padded `input_ids` `[B, S]` with per-row true `lengths`.
    ///
    /// Returns `[B, S, d_model]` where positions at or beyond each row's
    /// length are exactly zero — Wan's trim-then-zero-pad contract.
    pub fn forward(&self, input_ids: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        if lengths.len() != batch {
            return Err(anyhow!(
                "UMT5: {} lengths for a batch of {batch}",
                lengths.len()
            ));
        }
        let relative_buckets = self.relative_buckets(seq_len)?;
        // Additive key mask [B, 1, 1, S] and multiplicative output mask
        // [B, S, 1], both from the same lengths so they cannot disagree.
        let mut key_rows = Vec::with_capacity(batch);
        let mut keep_rows = Vec::with_capacity(batch);
        for &len in lengths {
            let len = len.min(seq_len);
            let key_row: Vec<f32> = (0..seq_len)
                .map(|pos| if pos >= len { MASK_NEG } else { 0.0 })
                .collect();
            let keep_row: Vec<f32> = (0..seq_len)
                .map(|pos| if pos >= len { 0.0 } else { 1.0 })
                .collect();
            key_rows.push(key_row);
            keep_rows.push(keep_row);
        }
        let key_mask = Tensor::new(key_rows, &self.device)?
            .reshape((batch, 1, 1, seq_len))?
            .to_dtype(DType::F32)?;
        let keep = Tensor::new(keep_rows, &self.device)?.reshape((batch, seq_len, 1))?;

        let mut xs = self.embedding.forward(input_ids)?;
        for block in &self.blocks {
            xs = block.forward(&xs, &relative_buckets, Some(&key_mask))?;
        }
        let xs = self.final_norm.forward(&xs)?;
        // Zero the padded positions: the DiT re-embeds zeros for padding, so
        // leaking pad-token encoder states here degrades every generation.
        Ok(xs.broadcast_mul(&keep.to_dtype(xs.dtype())?)?)
    }
}

/// Disk-loading + retention wrapper around [`UMt5Encoder`], mirroring the
/// `encoders/t5.rs` surface the engines already use: tokenize, encode with
/// the zero-pad contract, and `drop_weights`/`reload` so the transformer gets
/// the VRAM during denoise. (CPU parking via `MOLD_KEEP_TE_RAM` follows when
/// the engine wires residency in the wan/04 layer.)
pub(crate) struct WanTextEncoder {
    model: Option<UMt5Encoder>,
    pub tokenizer: Arc<Tokenizer>,
    device: Device,
    dtype: DType,
    encoder_paths: Vec<PathBuf>,
    pub is_quantized: bool,
}

impl WanTextEncoder {
    /// Auto-detects `.gguf` vs safetensors from the first path's extension.
    pub fn load_with_tokenizer(
        encoder_paths: &[PathBuf],
        device: &Device,
        dtype: DType,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self> {
        let first = encoder_paths
            .first()
            .ok_or_else(|| anyhow!("UMT5: no encoder weight files supplied"))?;
        let is_quantized = first.extension().is_some_and(|e| e == "gguf");
        let model = if is_quantized {
            UMt5Encoder::from_gguf(first, device)?
        } else {
            UMt5Encoder::from_safetensors(encoder_paths, UMt5Config::xxl(), device, dtype)?
        };
        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            dtype,
            encoder_paths: encoder_paths.to_vec(),
            is_quantized,
        })
    }

    /// Tokenize to the fixed 512-token window and encode.
    ///
    /// Returns `[B, 512, 4096]` with zeroed padding. Prompts are
    /// canonicalized the way upstream's tokenizer wrapper does
    /// (`clean='whitespace'`), and truncation preserves the trailing EOS.
    pub fn encode(&self, prompts: &[&str]) -> Result<Tensor> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("UMT5 weights are dropped; call reload() first"))?;
        let mut ids = Vec::with_capacity(prompts.len());
        let mut lengths = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let cleaned = canonicalize_prompt(prompt);
            let encoding = self
                .tokenizer
                .encode(cleaned.as_str(), true)
                .map_err(|e| anyhow!("UMT5 tokenization failed: {e}"))?;
            let (row, len) = fit_ids_to_window(encoding.get_ids().to_vec());
            lengths.push(len);
            ids.push(row);
        }
        let input_ids = Tensor::new(ids, &self.device)?;
        model.forward(&input_ids, &lengths)
    }

    /// Free the weights (VRAM and host RAM). `reload()` reads from disk.
    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    pub fn reload(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        self.model = Some(if self.is_quantized {
            UMt5Encoder::from_gguf(&self.encoder_paths[0], &self.device)?
        } else {
            UMt5Encoder::from_safetensors(
                &self.encoder_paths,
                UMt5Config::xxl(),
                &self.device,
                self.dtype,
            )?
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny_config() -> UMt5Config {
        UMt5Config {
            vocab_size: 32,
            d_model: 16,
            d_kv: 4,
            d_ff: 32,
            num_heads: 4,
            num_layers: 2,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            eps: 1e-6,
        }
    }

    fn tiny_model(config: &UMt5Config) -> UMt5Encoder {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Materialize every tensor the loader requires by getting it through
        // the VarMap (init to small randn so activations are finite).
        UMt5Encoder::from_var_builder(vb, config.clone(), &device)
            .expect("tiny UMT5 constructs from a VarMap")
    }

    /// Upstream's `clean='whitespace'` contract: double HTML-unescape,
    /// collapse whitespace runs, strip ends.
    #[test]
    fn prompt_canonicalization_matches_upstream_cleaning() {
        assert_eq!(
            canonicalize_prompt("  a\n\nlong\t prompt  "),
            "a long prompt"
        );
        // Double-unescape: "&amp;amp;" -> "&amp;" -> "&".
        assert_eq!(canonicalize_prompt("cats &amp;amp; dogs"), "cats & dogs");
        assert_eq!(
            canonicalize_prompt("&lt;b&gt;bold&lt;/b&gt; &#39;quoted&#39; &#x41;"),
            "<b>bold</b> 'quoted' A"
        );
        // Unknown entities and bare ampersands pass through untouched.
        assert_eq!(
            canonicalize_prompt("R&D &unknown; done"),
            "R&D &unknown; done"
        );
        // Non-breaking spaces collapse with the rest of the whitespace.
        assert_eq!(canonicalize_prompt("a\u{a0}b"), "a b");
    }

    /// HF-style truncation reserves the final slot for EOS; the window is
    /// padded with PAD=0 and the reported length covers EOS.
    #[test]
    fn truncation_preserves_eos_in_the_window() {
        // Short input: untouched, padded, length = content + EOS.
        let mut short: Vec<u32> = vec![7, 8, 9];
        short.push(UMT5_EOS_ID);
        let (row, len) = fit_ids_to_window(short);
        assert_eq!(len, 4);
        assert_eq!(row.len(), WAN_TEXT_LEN);
        assert_eq!(row[3], UMT5_EOS_ID);
        assert!(row[4..].iter().all(|&id| id == UMT5_PAD_ID));

        // Overlength input: truncated to the window with EOS forced last.
        let mut long: Vec<u32> = (10..700).collect();
        long.push(UMT5_EOS_ID);
        let (row, len) = fit_ids_to_window(long);
        assert_eq!(len, WAN_TEXT_LEN);
        assert_eq!(row[WAN_TEXT_LEN - 1], UMT5_EOS_ID);
        assert_eq!(row[WAN_TEXT_LEN - 2], 10 + (WAN_TEXT_LEN as u32) - 2);
    }

    /// Pin the bidirectional bucket function against hand-computed HF values:
    /// 32 buckets split 16 forward / 16 backward, max_exact 8, log-spaced to
    /// max_distance 128.
    #[test]
    fn relative_bucket_math_matches_reference_values() {
        let bucket = |q, k| relative_position_bucket(q, k, 32, 128);
        assert_eq!(bucket(0, 0), 0);
        assert_eq!(bucket(1, 0), 1); // looking back 1
        assert_eq!(bucket(0, 1), 17); // looking forward 1 = 16 + 1
        assert_eq!(bucket(7, 0), 7); // last exact backward bucket
        assert_eq!(bucket(0, 7), 23);
        // First log bucket: distance 8 -> ln(1)/ln(16)*8 = 0 -> bucket 8.
        assert_eq!(bucket(8, 0), 8);
        assert_eq!(bucket(0, 8), 24);
        // Deep log bucket: distance 100 -> 8 + floor(ln(12.5)/ln(16)*8) = 15.
        assert_eq!(bucket(100, 0), 15);
        assert_eq!(bucket(0, 100), 31);
        // Distances at/past max_distance clamp to the last bucket.
        assert_eq!(bucket(511, 0), 15);
        assert_eq!(bucket(0, 511), 31);
    }

    /// The trim-then-zero-pad contract: padded positions come back exactly
    /// zero, real positions do not.
    #[test]
    fn forward_zeroes_padded_positions() {
        let config = tiny_config();
        let model = tiny_model(&config);
        let seq = 8usize;
        let ids = Tensor::zeros((2, seq), DType::U32, &Device::Cpu).unwrap();
        let out = model.forward(&ids, &[3, 5]).unwrap();
        assert_eq!(out.dims(), &[2, seq, config.d_model]);
        let out: Vec<Vec<Vec<f32>>> = out.to_vec3().unwrap();
        for (row, &len) in out.iter().zip([3usize, 5].iter()) {
            for (pos, vals) in row.iter().enumerate() {
                let sum: f32 = vals.iter().map(|v| v.abs()).sum();
                if pos >= len {
                    assert_eq!(sum, 0.0, "padded position {pos} must be zeroed");
                } else {
                    assert!(sum > 0.0, "real position {pos} must carry signal");
                }
            }
        }
    }

    /// Padding tokens must not influence real positions: the same prompt
    /// encoded alone and encoded beside a longer batch row must agree.
    /// This is what the additive key mask exists for.
    #[test]
    fn key_mask_isolates_rows_from_their_padding() {
        let config = tiny_config();
        let model = tiny_model(&config);
        let device = Device::Cpu;
        // Row of ids [1,2,3] padded with a *non-zero* junk token to prove the
        // mask (not the pad id) does the isolation.
        let short = Tensor::new(vec![vec![1u32, 2, 3, 9, 9, 9]], &device).unwrap();
        let clean = Tensor::new(vec![vec![1u32, 2, 3, 4, 5, 6]], &device).unwrap();
        let with_junk = model.forward(&short, &[3]).unwrap();
        let with_other_tail = model.forward(&clean, &[3]).unwrap();
        let a: Vec<Vec<Vec<f32>>> = with_junk.to_vec3().unwrap();
        let b: Vec<Vec<Vec<f32>>> = with_other_tail.to_vec3().unwrap();
        for pos in 0..3 {
            for (x, y) in a[0][pos].iter().zip(b[0][pos].iter()) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "masked tail leaked into real position {pos}: {x} vs {y}"
                );
            }
        }
    }

    /// UMT5 loading must refuse a checkpoint whose blocks lack their own
    /// relative_attention_bias (i.e. plain T5/mT5), naming the block. Built
    /// against a real safetensors file because a VarMap-backed builder
    /// fabricates missing tensors on `get`.
    #[test]
    fn missing_per_layer_bias_is_a_hard_error() {
        let config = tiny_config();
        let device = Device::Cpu;
        // Materialize a complete tiny model through a VarMap, then write
        // every tensor EXCEPT block 1's bias to a safetensors file.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        UMt5Encoder::from_var_builder(vb, config.clone(), &device).unwrap();
        let dropped = "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight";
        let tensors: std::collections::HashMap<String, Tensor> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.as_str() != dropped)
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t5-not-umt5.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();

        let err = UMt5Encoder::from_safetensors(&[path], config, &device, DType::F32)
            .err()
            .expect("a checkpoint without per-layer bias must be refused");
        let text = format!("{err:#}");
        assert!(
            text.contains("encoder.block.1"),
            "must name the block: {text}"
        );
        assert!(
            text.contains("shared_pos"),
            "must explain the UMT5 difference: {text}"
        );
    }
}
