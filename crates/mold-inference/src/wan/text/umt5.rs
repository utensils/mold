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
        Self::from_checkpoint_var_builder(vb, config, device)
    }

    /// Build from a VarBuilder over raw checkpoint tensor names, probing for
    /// the two layouts in the wild before delegating to
    /// [`Self::from_var_builder`].
    ///
    /// Shared by the mmap path and the CPU-park path: both hand over the
    /// checkpoint's own key names, so the prefix probe has to live in one
    /// place or an unparked encoder silently misses every weight.
    pub fn from_checkpoint_var_builder(
        vb: VarBuilder,
        config: UMt5Config,
        device: &Device,
    ) -> Result<Self> {
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
/// the zero-pad contract, `drop_weights`/`reload` so the transformer gets the
/// VRAM during denoise, and `park_to_cpu`/`unpark_to_gpu` so `MOLD_KEEP_TE_RAM`
/// can keep the weights in host RAM between requests instead of re-reading
/// 11.4 GB from disk every generation.
pub(crate) struct WanTextEncoder {
    model: Option<UMt5Encoder>,
    pub tokenizer: Arc<Tokenizer>,
    device: Device,
    dtype: DType,
    encoder_paths: Vec<PathBuf>,
    pub is_quantized: bool,
    /// Parameters parked on host RAM at the checkpoint's own dtype. `None`
    /// when the encoder has never parked, or on the GGUF path, whose
    /// device-tied `QTensor` storage parks by dropping instead.
    parked_tensors: Option<HashMap<String, Tensor>>,
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
            parked_tensors: None,
        })
    }

    /// The exact weight files this encoder was built from.
    pub fn encoder_paths(&self) -> &[PathBuf] {
        &self.encoder_paths
    }

    /// Whether a retained encoder can serve a render planned for these
    /// weights on this device at this dtype.
    ///
    /// All three have to match. The variant resolver re-measures free VRAM
    /// every request, so consecutive renders of the same model can legitimately
    /// land on a different GGUF tier or move between GPU and CPU — reusing a
    /// retained encoder across any of those changes would silently ignore the
    /// decision that was just made.
    pub fn matches(&self, paths: &[PathBuf], device: &Device, dtype: DType) -> bool {
        self.encoder_paths == paths && self.device.same_device(device) && self.dtype == dtype
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

    /// Park the parameters on host RAM and free the compute device.
    ///
    /// The parked copy keeps the checkpoint's own dtype — F16 for the shipped
    /// `umt5_xxl_fp16.safetensors`, so ~11.4 GB of host RAM, not the ~22.7 GB
    /// an F32 CPU-compute copy would take. Widening to the compute dtype
    /// happens inside the VarBuilder on the way back out, which is why
    /// unparking to a CPU device (where candle needs F32) is still correct.
    ///
    /// The first park after a load reads the safetensors from disk once; every
    /// later cycle is pure RAM. GGUF encoders park by dropping — their
    /// `QTensor` storage is device-tied and not walkable — so they keep paying
    /// the reload, which is already far cheaper than the FP16 path's.
    /// No-op when already parked.
    pub fn park_to_cpu(&mut self) -> Result<()> {
        if self.is_parked() {
            self.model = None;
            return Ok(());
        }
        if self.is_quantized {
            self.drop_weights();
            return Ok(());
        }
        self.parked_tensors = Some(crate::encoders::park::load_tensors_to_cpu(
            &self.encoder_paths,
        )?);
        self.model = None;
        Ok(())
    }

    /// Rebuild on this encoder's own device from the parked tensors, falling
    /// back to a disk `reload()` when nothing is parked (GGUF, or a first
    /// request). No-op when the model is already resident.
    pub fn unpark(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let Some(parked) = self.parked_tensors.as_ref() else {
            return self.reload();
        };
        let vb = crate::encoders::park::varbuilder_from_parked(parked, self.dtype, &self.device);
        self.model = Some(UMt5Encoder::from_checkpoint_var_builder(
            vb,
            UMt5Config::xxl(),
            &self.device,
        )?);
        Ok(())
    }

    /// Whether the parameters are currently on host RAM rather than the
    /// compute device. Always false on the GGUF path.
    pub fn is_parked(&self) -> bool {
        self.model.is_none() && self.parked_tensors.is_some()
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

    /// Unparking has to reproduce the mmap load exactly.
    ///
    /// The parked map carries the checkpoint's own key names, which for the
    /// shipped `umt5_xxl_fp16.safetensors` repack are under
    /// `text_encoders.umt5xxl.transformer.`. That prefix probe lives in
    /// `from_checkpoint_var_builder` precisely so both paths share it — an
    /// unpark that skipped it would find no weights, and one that applied it
    /// unconditionally would break the bare-key layout. Pin the equivalence on
    /// output, not on key lists.
    #[test]
    fn an_unparked_encoder_matches_the_mmap_load_under_the_repack_prefix() {
        let config = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        // Root the builder at the repack prefix so the saved file carries the
        // prefixed names, exactly like the shipped encoder.
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device)
            .pp("text_encoders.umt5xxl.transformer");
        UMt5Encoder::from_var_builder(vb, config.clone(), &device).unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-umt5-park-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        varmap.save(&path).unwrap();

        let files = vec![path.clone()];
        let from_disk =
            UMt5Encoder::from_safetensors(&files, config.clone(), &device, DType::F32).unwrap();
        let parked = crate::encoders::park::load_tensors_to_cpu(&files).unwrap();
        let unparked = UMt5Encoder::from_checkpoint_var_builder(
            crate::encoders::park::varbuilder_from_parked(&parked, DType::F32, &device),
            config,
            &device,
        )
        .unwrap();
        let _ = std::fs::remove_file(&path);

        let ids = Tensor::new(vec![vec![1u32, 2, 3, 4, 5, 6]], &device).unwrap();
        let disk: Vec<Vec<Vec<f32>>> = from_disk.forward(&ids, &[6]).unwrap().to_vec3().unwrap();
        let ram: Vec<Vec<Vec<f32>>> = unparked.forward(&ids, &[6]).unwrap().to_vec3().unwrap();
        assert_eq!(
            disk, ram,
            "an unparked encoder must be bit-identical to the mmap load"
        );
        // Guard against the assertion passing on two all-zero stacks.
        assert!(disk.iter().flatten().flatten().any(|v| *v != 0.0));
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
    const GOLDEN_UMT5_HIDDEN: &[f64] = &[
        -2.286623126148028,
        0.08165793117651009,
        0.8817346190432662,
        1.3896423543986116,
        -0.518777871488305,
        -1.0657380793590383,
        0.7141045310450046,
        0.4061820443291086,
        0.3241592418309952,
        -1.4380963371016682,
        -0.33741209906055614,
        0.7555795954296612,
        1.1629191400836394,
        0.10439399936858468,
        -1.7362018760394013,
        0.7664945792052074,
        1.4723705010888328,
        1.0423428770760943,
        -1.6738266633739427,
        -1.3251639839970761,
        0.815963079630597,
        0.5804893941838879,
        -0.06696654642239723,
        -1.2995056001639624,
        0.35951166860103956,
        1.0450372422346101,
        0.8283199630307094,
        -0.8779964803162763,
        -1.6825067524825212,
        0.6878628998233219,
        0.7650394341252135,
        0.35550640151615115,
        -2.3489567708733823,
        -0.025240081293367002,
        0.6331281441887902,
        1.370746832007162,
        -0.34933622167238243,
        -1.0873459582151237,
        0.768536965017508,
        0.35437002783110844,
        0.4716609266053665,
        -1.4390771049381186,
        -0.42979699507600383,
        0.5992513723684159,
        1.0663936487753283,
        0.32953572195713676,
        -1.7494636925193834,
        0.8547779181705956,
        1.5158514908229546,
        1.2265499347202482,
        -1.7075708784768344,
        -1.0845784812414558,
        0.6424836889947346,
        0.7348092867099523,
        -0.22732030300204398,
        -1.3397878616416874,
        0.3001053116819827,
        0.999769484195646,
        1.0221417530715244,
        -0.9419853865432857,
        -1.4508862492662158,
        0.5193532941371808,
        0.8782218183125895,
        0.23680596731294412,
        -2.3340209238642977,
        -0.19637814364302722,
        0.5920061522398015,
        1.2952686759454635,
        -0.3346265201082465,
        -1.1157517535438652,
        0.7949325299651021,
        0.46014374023099275,
        0.5194430636425296,
        -1.4123574302243582,
        -0.551581812818522,
        0.5469847868912694,
        1.0134774038609522,
        0.3350775710765286,
        -1.743503919261813,
        0.8555813035184562,
        1.574895534554837,
        1.2834559245231292,
        -1.630972630625162,
        -0.7209034528027046,
        0.5830462175126028,
        0.961879051820335,
        -0.4983944632902303,
        -1.3854826087545211,
        0.13176655641155594,
        0.9599235111727131,
        1.1581382643439049,
        -0.9634439665910499,
        -1.0711980172056008,
        0.42985398515750606,
        1.1232598722952332,
        -0.054894209334905804,
        -2.227339682183593,
        -0.4552479855618804,
        0.6045660004393026,
        1.1116798590760248,
        -0.4658122100093724,
        -1.178339856614483,
        0.8202711787269672,
        0.7292857217391789,
        0.5745016644684847,
        -1.3345694253581388,
        -0.724979769124978,
        0.4946180630612168,
        0.9207308366905756,
        0.15577053758284343,
        -1.7523045287631227,
        0.8211934106234211,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        1.0958056980457749,
        2.159147395632299,
        -1.3962713568476943,
        -0.6232414600086722,
        -0.32267582278878093,
        1.0503959827337224,
        -0.08552452396558716,
        -1.3104009009767774,
        -0.040603733533837845,
        0.4711505593795457,
        1.7032909932188076,
        -0.6986200899157649,
        -0.9216261077372759,
        -0.3812945110254508,
        0.854388075312769,
        0.6601674412792413,
        -2.1628644978434712,
        -0.5534435920665044,
        -0.22021686350021008,
        1.2217726184660718,
        0.3061547639155639,
        -1.2711386582415523,
        0.6780365737397673,
        0.21683542216275306,
        1.0661687933645165,
        -1.1454138703380632,
        -0.8078585839516945,
        -0.004476930564280237,
        0.6767495463731154,
        1.104138551282234,
        -1.800086163152704,
        0.7381916309850388,
        0.4741094209495858,
        2.5261241629266022,
        -1.004376211972894,
        -0.1037600921528403,
        -1.0347249134104872,
        1.0729065175835817,
        0.08778397725612191,
        -1.0609391766956644,
        -0.2132656729766828,
        -0.08462109003386212,
        1.9311985251147512,
        -0.4415559512937484,
        -0.34008193293787825,
        -0.9803381309912125,
        0.6048831692848099,
        0.9584666429834012,
        -1.8134119433699103,
        -0.6817379771729846,
        -0.7670443565969387,
        0.9586711686847575,
        0.8081386068464526,
        -1.2873670695337975,
        0.5765867953309054,
        -0.06349290664415688,
        1.346014413355481,
        -0.8126724622367228,
        -0.8841416811110756,
        -0.34916618216480777,
        0.2633717524044926,
        1.6163788377364914,
        -1.6952543818908485,
        0.640168743891773,
        -1.252248537866779,
        2.040638387133284,
        0.9787506183830491,
        0.3789079316993384,
        -1.4923209262905077,
        -0.17144519613865225,
        0.875621122515841,
        -0.2006365969014517,
        -0.4462260476068002,
        -1.0814973367162233,
        1.165254804000755,
        1.0352439346614963,
        0.3879881810407025,
        -1.2349857506702073,
        -0.9758224079446072,
        1.3961371549920925,
        0.6938589597051997,
        -0.857807536375452,
        -2.0620770355314604,
        -0.25812547149869536,
        1.7078326161659798,
        0.041053563751784226,
        -0.4107270662641113,
        -0.7335016453127214,
        1.070273269119735,
        0.7737577626431279,
        -0.5161636880690219,
        -1.4300181299313193,
        -0.8419621645064435,
        1.8919993627223526,
        0.4132382715678535,
        -0.3839603902023529,
        -1.6302830569585531,
        1.1625742993040677,
        1.5839989217073764,
        0.5712251045800827,
        -1.3237785897136094,
        -0.6671293083011866,
        0.8900690997614761,
        0.3145956289048813,
        -0.4102081482707864,
        -1.2083409210096923,
        0.46340237539202056,
        1.3539312299053583,
        0.702609448863778,
        -1.0547748233439889,
        -1.3788016901205442,
        1.076278753596701,
        1.2675731391014233,
        -0.432544976499588,
        -1.9937529591581058,
        -0.7245654084818084,
        1.5351091464448767,
        0.46292387798350304,
        -0.4608621947732619,
        -0.8966768301578563,
        0.6998769604885625,
        1.0493053138046549,
        -0.12508246886313285,
        -1.3591202438249468,
        -1.1561110365694143,
        1.4927427219925917,
        0.9277732293918753,
        -0.35789379520286496,
        -1.8114461539618851,
        0.669129524498727,
        1.6442147475882443,
        0.8035239716390786,
        -1.165191491475589,
        -0.8570476842197765,
        0.8069249597498183,
        0.525425453489597,
        -0.28121626893396917,
        -1.2677568992376793,
        0.10613659286095681,
        1.3181302735640932,
        0.9260381916337018,
        -0.8453604217954358,
        -1.5177753312058972,
        0.8528018071868526,
        1.4300412782374565,
        0.448563720416635,
        -1.6074510948045322,
        -1.5632185707875093,
        1.2023229873431969,
        0.46337799517997236,
        0.05116636842177209,
        -1.1386844560752931,
        0.3775583300394828,
        1.1235412333126595,
        0.36670732543456264,
        -0.8478944126097285,
        -1.8660271371032948,
        1.0266904944416486,
        0.8081722255902194,
        0.3357380169856438,
        -2.2933397273786835,
        0.21279129046452191,
        0.6805572185060934,
        1.5875472841073905,
        -0.38248515377641024,
        -1.1034595383073307,
        0.5923662237608186,
        0.20045879582290174,
        0.4650761031960844,
        -1.4246788144452245,
        -0.23856431193712582,
        0.6366792704665069,
        1.229580790960863,
        0.3201423229007269,
        -1.8150039414770116,
        0.6747923959966357,
        1.1941586359130214,
        1.1388605860011158,
        -1.4285999186994813,
        -1.7324746377106202,
        0.805599194354785,
        0.3116144935048012,
        0.3991228123901275,
        -1.2646683346463485,
        0.3517961577116471,
        0.9333542427648505,
        0.7362033242875268,
        -0.565713061320146,
        -2.055537123200845,
        0.699969200767537,
        0.41964486388615013,
        0.8603578375719532,
        -2.3480190441395554,
        0.06901434083223497,
        0.41247562039415486,
        1.5770112643513903,
        -0.15645014578102626,
        -1.104984960323155,
        0.6321323351051327,
        0.13106156387587528,
        0.6073137382722888,
        -1.415523104741309,
        -0.35168735602900403,
        0.46415863862557416,
        1.1357029681538098,
        0.5986095528251743,
        -1.7906903639915122,
        0.7421974670501565,
        1.294758827884262,
        1.4326598703108229,
        -1.5643711362256199,
        -1.4462911251312127,
        0.52984613024427,
        0.538693605844263,
        0.18825443737720787,
        -1.32530751287899,
        0.31126654766606954,
        0.8928242588724378,
        1.0357651967637742,
        -0.7137897506067139,
        -1.7919623909130553,
        0.43591329493881703,
        0.5814173444398665,
        0.7344618953871013,
        -2.33994149907136,
        -0.24017064947039857,
        0.1512074684079953,
        1.413533692837369,
        0.048975737983661526,
        -1.1515102529389951,
        0.7128771296380433,
        0.19584394791785786,
        0.7788060874855894,
        -1.347758573564396,
        -0.5858904365161505,
        0.2739519348061131,
        0.9386385472632818,
        0.8244533451235295,
        -1.7689065975848213,
        0.8135131269276137,
        1.2686553092937027,
        1.8933491815617967,
        -1.5881318438122682,
        -0.8482392944887865,
        0.02475373964407783,
        0.8863895842613324,
        -0.10843834594693093,
        -1.363428783807965,
        0.15597623304230393,
        0.6889820167249401,
        1.4917899129700651,
        -0.8187003605999152,
        -1.2026196389896289,
        -0.036374195228333656,
        0.7905433348442757,
        0.5599255032903022,
        -2.2750246144628186,
        -0.40767447875006635,
        -0.012414690478725031,
        1.3263376119315025,
        0.15960109121968533,
        -1.2072208936166546,
        0.7072537930326486,
        0.227119785155991,
        0.9089906844974983,
        -1.265958480452668,
        -0.7056934748991778,
        0.1472055635911508,
        0.8252042828962722,
        0.9456239525547573,
        -1.7829416621213012,
        0.7873065581584867,
        1.0372188477978013,
        2.219220428515087,
        -1.347743266165042,
        -0.6064064559360749,
        -0.41239960406536136,
        1.0573435675146194,
        -0.04425222856205382,
        -1.2842659264938368,
        -0.06939178554898959,
        0.4155234281743073,
        1.738517649233695,
        -0.6596572040376348,
        -0.8909385587406946,
        -0.46531216843647855,
        0.8307307161628716,
        0.7221848041975957,
        -1.7946642850079166,
        -0.6980752758647776,
        -0.7920668217916204,
        0.9702400793282779,
        0.8349734141233512,
        -1.2762279188314039,
        0.5537218626666397,
        -0.07544467490732382,
        1.3529317979585072,
        -0.798577514393438,
        -0.8888814158502357,
        -0.3713064512626468,
        0.2682987811318727,
        1.642506053516281,
        -1.6746207795027588,
        0.6136042309705373,
        -0.4339750182923295,
        2.6084383344346422,
        -0.1740577824122811,
        0.25726737619660767,
        -1.4977651674534633,
        0.6216355595666979,
        0.493752311305821,
        -0.7101600534664291,
        -0.3426517856258203,
        -0.6899957093130576,
        1.8064989288228352,
        0.1959859500379604,
        0.10645897357987798,
        -1.3019441843492918,
        -0.1118198704782602,
        1.3293009894481083,
        -0.7683114602261326,
        -1.0680816616028117,
        -1.6013655027505678,
        0.5700018971559885,
        1.4852699655077863,
        -0.8769039342662219,
        0.04643222127514994,
        -0.3670106547859442,
        1.50659804595977,
        -0.07144203921439175,
        -0.9503115833337805,
        -1.0360312604808577,
        -0.19600878424349985,
        2.111848364971643,
        -0.9094533311459624,
        0.0348114573848941,
        -1.3028967650754661,
        1.9934288608846853,
        1.0106345105471792,
        0.4242891056555411,
        -1.4844803812867209,
        -0.22261028680203715,
        0.873786514574415,
        -0.17264995185180534,
        -0.4243001776699045,
        -1.1051742881272169,
        1.1228446260988254,
        1.0523893052528428,
        0.4287933543584226,
        -1.2119871414152548,
        -1.0333275385532388,
        1.3751315748014608,
        0.7426777525130955,
        -0.7942294600134782,
        -2.068575111244906,
        -0.3465904452341395,
        1.7088713958272381,
        0.08444812138298842,
        -0.37918126428206694,
        -0.7713103862585965,
        1.032778018141834,
        0.8005925068204978,
        -0.4716490957601191,
        -1.417090149094643,
        -0.9198043202710084,
        1.8714703250922038,
        0.46338973344824597,
        -0.33143577272811475,
        -1.8151322258456752,
        0.6861620004746227,
        1.6360045768914724,
        0.8106847398911651,
        -1.1628808133219357,
        -0.8567754031929911,
        0.8064095276696068,
        0.5119606617855762,
        -0.2773850467815773,
        -1.2704414637367045,
        0.1173390358661136,
        1.3158512252216834,
        0.9267371181774696,
        -0.8388162819504698,
        -1.5222745874851145,
        0.8573096936775546,
    ];
    const UMT5_XXL_UNIT_IDS: &[u32] = &[
        320, 88210, 4062, 273, 56209, 346, 102701, 702, 312, 147946, 313, 14605, 21006, 367, 89680,
        274,
    ];
    const UMT5_XXL_RAW_REPEATS: usize = 60;
    const UMT5_XXL_RAW_LEN: usize = 961;
    const UMT5_XXL_WINDOW_TAIL: &[u32] = &[312, 147946, 313, 14605, 21006, 367, 89680, 1];
    const UMT5_XXL_WINDOW_SUM: u64 = 16874159;

    // ---------------------------------------------------------------------
    // Golden parity with HF transformers `UMT5EncoderModel` (#789).
    //
    // Weights, ids, and expected hidden states are shared with the capture
    // script `tmp/wan-research/gen-wan-umt5-goldens.py`, which runs the tiny
    // config through transformers (the same architecture upstream's
    // `tmp/Wan2.1/wan/modules/t5.py` implements, per-layer relative
    // attention bias included) in float64 over the same float32-cast
    // synthesized weights, asserts the reference's padded positions are
    // nonzero, then applies Wan's trim-then-zero-pad contract
    // (`t5.py:506-513`) before emitting `wan-umt5-golden.json`.
    //
    // Sequence length 24 exercises both the exact (< 8) and the logarithmic
    // (>= 8) relative-distance buckets in both directions, so a bucket
    // asymmetry — swapping the forward/backward offset halves, or breaking
    // the log ramp — moves these values. Mutation spot-check recorded when
    // the fixture landed: swapping the bucket offset branches moves the
    // golden by 8.9e-3; the existing unit tests on
    // `relative_position_bucket` pin the arithmetic itself at exact values.
    // ---------------------------------------------------------------------

    /// The capture script's deterministic fill: layer-norm gains
    /// `1 + 0.25*sin(0.5*i + off)`, everything else `0.2*sin(0.7*i + off)`,
    /// `off(name) = (sum of utf8 bytes % 97) * 0.1`, row-major, f64 -> f32.
    fn synth_param(name: &str, count: usize) -> Vec<f32> {
        let off = (name.bytes().map(u64::from).sum::<u64>() % 97) as f64 * 0.1;
        (0..count)
            .map(|i| {
                let i = i as f64;
                let v = if name.contains("layer_norm") {
                    1.0 + 0.25 * (0.5 * i + off).sin()
                } else {
                    0.2 * (0.7 * i + off).sin()
                };
                v as f32
            })
            .collect()
    }

    /// Materialize the tiny config once through a VarMap to learn the exact
    /// (name, shape) set mold's loader reads, then rebuild from synthesized
    /// tensors under those names. The capture script asserts transformers'
    /// `state_dict()` carries exactly this key set, so a naming divergence
    /// fails capture instead of silently misaligning offsets.
    fn golden_model(config: &UMt5Config) -> UMt5Encoder {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        UMt5Encoder::from_var_builder(vb, config.clone(), &device)
            .expect("tiny UMT5 constructs from a VarMap");
        let mut tensors = std::collections::HashMap::new();
        for (name, var) in varmap.data().lock().unwrap().iter() {
            let dims = var.dims().to_vec();
            let count: usize = dims.iter().product();
            let tensor = Tensor::from_vec(synth_param(name, count), dims, &device).unwrap();
            tensors.insert(name.clone(), tensor);
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        UMt5Encoder::from_var_builder(vb, config.clone(), &device)
            .expect("golden UMT5 constructs from synthesized tensors")
    }

    /// Batch 2 x seq 24 with junk tokens continuing past row 0's true length
    /// of 7, so the fixture proves the additive key mask (not the pad id)
    /// does the isolation: `id[b][s] = (s*7 + b*3 + 1) % 32`.
    const GOLDEN_UMT5_SEQ: usize = 24;
    const GOLDEN_UMT5_LENGTHS: [usize; 2] = [7, 24];

    /// Observed agreement with the f64 reference is ~5e-7 at worst; 5e-6
    /// keeps ~10x headroom while sitting three orders below the mutation
    /// signal (9e-3).
    const UMT5_GOLDEN_TOLERANCE: f64 = 5e-6;

    #[test]
    fn forward_matches_the_transformers_golden() {
        let config = tiny_config();
        let model = golden_model(&config);
        let ids: Vec<Vec<u32>> = (0..2u32)
            .map(|b| {
                (0..GOLDEN_UMT5_SEQ as u32)
                    .map(|s| (s * 7 + b * 3 + 1) % config.vocab_size as u32)
                    .collect()
            })
            .collect();
        let input_ids = Tensor::new(ids, &Device::Cpu).unwrap();
        let out = model.forward(&input_ids, &GOLDEN_UMT5_LENGTHS).unwrap();
        assert_eq!(out.dims(), &[2, GOLDEN_UMT5_SEQ, config.d_model]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.len(), GOLDEN_UMT5_HIDDEN.len());
        let mut worst = 0.0f64;
        for (i, (g, w)) in got.iter().zip(GOLDEN_UMT5_HIDDEN.iter()).enumerate() {
            let err = (f64::from(*g) - w).abs();
            assert!(
                err < UMT5_GOLDEN_TOLERANCE,
                "hidden[{i}]: got {g}, want {w}, err {err:e}"
            );
            worst = worst.max(err);
        }
        assert!(worst < UMT5_GOLDEN_TOLERANCE, "worst error {worst:e}");
    }

    /// The 512-token window against the real `google/umt5-xxl` tokenizer:
    /// the capture script tokenizes one sentence repeated 60 times (verified
    /// strictly periodic, so the raw stream reconstructs from one unit),
    /// then pins upstream's `padding='max_length', truncation=True,
    /// max_length=512` window (`tmp/Wan2.1/wan/modules/tokenizers.py:54-58`)
    /// against `fit_ids_to_window`. HF truncation trims content to 511 and
    /// keeps EOS last; a naive truncate-after-postprocess drops EOS.
    #[test]
    fn window_fitting_matches_the_real_umt5_tokenizer() {
        let mut raw = Vec::with_capacity(UMT5_XXL_RAW_LEN);
        for _ in 0..UMT5_XXL_RAW_REPEATS {
            raw.extend_from_slice(UMT5_XXL_UNIT_IDS);
        }
        raw.push(UMT5_EOS_ID);
        assert_eq!(raw.len(), UMT5_XXL_RAW_LEN);
        assert!(raw.len() > WAN_TEXT_LEN, "fixture must overflow the window");

        let (row, len) = fit_ids_to_window(raw.clone());
        assert_eq!(len, WAN_TEXT_LEN);
        assert_eq!(row.len(), WAN_TEXT_LEN);
        // HF's window is trim-to-511 + EOS; ours must be byte-identical.
        assert_eq!(&row[..WAN_TEXT_LEN - 1], &raw[..WAN_TEXT_LEN - 1]);
        assert_eq!(&row[WAN_TEXT_LEN - 8..], UMT5_XXL_WINDOW_TAIL);
        assert_eq!(
            row.iter().map(|&v| u64::from(v)).sum::<u64>(),
            UMT5_XXL_WINDOW_SUM
        );
    }
}
