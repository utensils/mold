//! Quantized Qwen2.5-VL text encoder loader for GGUF files.
//!
//! Qwen-Image only needs the language-model text stack from Qwen2.5-VL, not the
//! multimodal projector or vision tower. This loader reads the GGUF language
//! tensors directly and returns last hidden states without the final RMSNorm,
//! matching the upstream diffusers Qwen-Image pipeline.
//!
//! ## Why the weight map is retained (#1044)
//!
//! [`GgufQwen2Weights`] is kept on the encoder rather than dropped at the end
//! of `load`. It is the park/unpark source of truth: a cold prompt used to pay
//! a measured **35.1 s** GGUF disk reload on every request because the only way
//! back from `drop_weights()` was to re-read the file. Retaining the map costs
//! nothing while resident **as long as** the quantized entries are the very
//! `Arc<QTensor>` the blocks' [`QMatMul`]s hold, so they are shared rather
//! than duplicated.
//!
//! That is a property of candle's policy, not a law: `QMatMul::from_arc`
//! dequantizes an `F32`/`F16`/`BF16` GGUF tensor (and, under
//! `CANDLE_DEQUANTIZE_ALL`, everything) into a dense `Tensor` and drops the
//! `Arc`, which would make the retained map a second, full, device-resident
//! copy of the weights. So the encoder *measures* the sharing after building
//! ([`GgufQwen2Weights::is_shared_with_built_modules`]) instead of asserting
//! it, and simply does not retain a map it cannot retain for free — that
//! encoder falls back to the old drop-and-reload rather than silently doubling
//! its VRAM. Every shipped `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` variant is a
//! k-quant, so today this only guards the path; nothing in `read()` restricts
//! which GGUF a per-model component path may point at.
//!
//! The map is deliberately split in two. Tensors that back a `QMatMul` stay
//! quantized and share their `Arc`; tensors that are dequantized at load
//! (the embedding table, the attention biases, the RMSNorm weights) are held
//! **plain**, because their quantized source is never read again and keeping
//! it would pin a second copy of a 150k-row embedding table on the device.
//!
//! Moving the quantized half between host and device is a raw byte memcpy via
//! `wan::block_offload`'s [`qtensor_to_device`] / [`rebuild_on`] — never a
//! dequantize/re-quantize round trip, which is lossy and would make a render
//! depend on whether the encoder happened to be parked. That module is the one
//! authority for the byte path; see its header for the full argument.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::with_tracing::QMatMul;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::wan::block_offload::{qtensor_to_device, rebuild_on};

// Qwen-Image tokenization pads to TOKENIZER_WINDOW + template strip prefix,
// so the GGUF path must support sequences comfortably above 1024 tokens.
const MAX_ROPE_POSITIONS: usize = 2048;

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

/// Scalar shape/metadata read out of the GGUF header once.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GgufQwen2Params {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    block_count: usize,
    rope_positions: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
}

impl GgufQwen2Params {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// The encoder's complete weight set, wherever it currently lives.
///
/// This is the park/unpark source of truth (#1044): [`GgufQwen2Weights::to_device`]
/// moves it host↔device without changing a byte, and [`GgufQwen2Weights::build`]
/// reconstructs the runnable modules from it. See the module header for why the
/// quantized and dequantized halves are stored separately.
pub(crate) struct GgufQwen2Weights {
    params: GgufQwen2Params,
    /// Tensors that back a [`QMatMul`]. The blocks hold `Arc` clones of these,
    /// so retaining the map while resident costs no extra memory.
    quant: HashMap<String, Arc<QTensor>>,
    /// Tensors dequantized at load time. Their quantized source is dropped.
    plain: HashMap<String, Tensor>,
}

/// The runnable modules built from a resident [`GgufQwen2Weights`].
struct GgufQwen2Modules {
    embedding: candle_nn::Embedding,
    blocks: Vec<Qwen2Block>,
    cos: Tensor,
    sin: Tensor,
}

fn read_tensor(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<Arc<QTensor>> {
    if !content.tensor_infos.contains_key(name) {
        anyhow::bail!("missing tensor: {name}");
    }
    Ok(Arc::new(content.tensor(file, name, device)?))
}

fn read_tensor_opt(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<Option<Arc<QTensor>>> {
    if !content.tensor_infos.contains_key(name) {
        return Ok(None);
    }
    read_tensor(content, file, name, device).map(Some)
}

/// The seven per-block projections that stay quantized and run through a
/// [`QMatMul`]. Everything else a block needs is dequantized at load.
const BLOCK_QUANT_LEAVES: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

impl GgufQwen2Weights {
    /// Read every tensor this encoder needs out of `path` onto `device`.
    fn read(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

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

        let mut quant: HashMap<String, Arc<QTensor>> = HashMap::new();
        let mut plain: HashMap<String, Tensor> = HashMap::new();

        // Dequantized immediately and never read quantized again, so the
        // quantized source is dropped rather than parked alongside it.
        let embedding_weight = read_tensor(&content, &mut file, TOKEN_EMBD, device)?
            .dequantize(device)
            .map_err(anyhow::Error::from)?;
        let hidden_size = embedding_weight.dim(1)?;
        plain.insert(TOKEN_EMBD.to_string(), embedding_weight);

        for i in 0..block_count {
            let prefix = format!("blk.{i}");
            for leaf in BLOCK_QUANT_LEAVES {
                let name = format!("{prefix}.{leaf}.weight");
                let tensor = read_tensor(&content, &mut file, &name, device)?;
                quant.insert(name, tensor);
            }
            for leaf in ["attn_norm", "ffn_norm"] {
                let name = format!("{prefix}.{leaf}.weight");
                let tensor = read_tensor(&content, &mut file, &name, device)?.dequantize(device)?;
                plain.insert(name, tensor);
            }
            for name in [
                format!("{prefix}.attn_q.bias"),
                format!("{prefix}.attn_k.bias"),
                format!("{prefix}.attn_v.bias"),
                format!("{prefix}.attn_q_norm.weight"),
                format!("{prefix}.attn_k_norm.weight"),
            ] {
                if let Some(tensor) = read_tensor_opt(&content, &mut file, &name, device)? {
                    plain.insert(name, tensor.dequantize(device)?);
                }
            }
        }

        Ok(Self {
            params: GgufQwen2Params {
                hidden_size,
                num_heads,
                num_kv_heads,
                block_count,
                rope_positions: context_length.min(MAX_ROPE_POSITIONS),
                rms_norm_eps,
                rope_theta,
            },
            quant,
            plain,
        })
    }

    /// Total bytes this weight set occupies wherever it currently lives.
    pub(crate) fn size_in_bytes(&self) -> u64 {
        let quant: u64 = self
            .quant
            .values()
            .map(|t| t.storage_size_in_bytes() as u64)
            .sum();
        let plain: u64 = self
            .plain
            .values()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum();
        quant + plain
    }

    /// Move the whole set to `device`, losslessly.
    pub(crate) fn to_device(&self, device: &Device) -> Result<Self> {
        self.relocate(device, false)
    }

    /// [`Self::to_device`] with the same-device short circuit removed — see
    /// `wan::block_offload::rebuild_on` for why that distinction has to be
    /// reachable: CI has no GPU, and a same-device move hands back the input
    /// `Arc`, so a park/unpark test written against `to_device` would compare
    /// a tensor with itself.
    #[cfg(test)]
    fn rebuilt_on(&self, device: &Device) -> Result<Self> {
        self.relocate(device, true)
    }

    fn relocate(&self, device: &Device, force: bool) -> Result<Self> {
        let mut quant = HashMap::with_capacity(self.quant.len());
        for (name, tensor) in &self.quant {
            let moved = if force {
                rebuild_on(tensor, device)?
            } else {
                qtensor_to_device(tensor, device)?
            };
            quant.insert(name.clone(), moved);
        }
        let mut plain = HashMap::with_capacity(self.plain.len());
        for (name, tensor) in &self.plain {
            plain.insert(name.clone(), tensor.to_device(device)?);
        }
        Ok(Self {
            params: self.params,
            quant,
            plain,
        })
    }

    /// Whether the modules just built from this set share its quantized
    /// tensors, i.e. whether retaining the map is actually free.
    ///
    /// Measured through the `Arc` strong counts rather than by re-deriving
    /// candle's dequantize policy, so an upstream change cannot silently turn
    /// the module header's claim into a VRAM doubling. Must be called while
    /// the built [`GgufQwen2Modules`] is still alive, and only on a map no
    /// one else holds a clone of — every other holder inflates the count and
    /// reads as sharing.
    fn is_shared_with_built_modules(&self) -> bool {
        self.quant
            .values()
            .all(|tensor| Arc::strong_count(tensor) > 1)
    }

    fn quant_tensor(&self, name: &str) -> Result<Arc<QTensor>> {
        self.quant
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))
    }

    fn plain_tensor(&self, name: &str) -> Result<Tensor> {
        self.plain
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))
    }

    /// Reconstruct the runnable modules. `device` must be where the weights
    /// currently live; only the RoPE cache is built fresh.
    fn build(&self, device: &Device) -> Result<GgufQwen2Modules> {
        let params = self.params;
        let head_dim = params.head_dim();
        let (cos, sin) = compute_rope(head_dim, params.rope_theta, params.rope_positions, device)?;

        let embedding_weight = self.plain_tensor(TOKEN_EMBD)?;
        let embedding = candle_nn::Embedding::new(embedding_weight, params.hidden_size);

        let mut blocks = Vec::with_capacity(params.block_count);
        for i in 0..params.block_count {
            let prefix = format!("blk.{i}");
            let quant_matmul = |leaf: &str| -> Result<QMatMul> {
                Ok(QMatMul::from_weights(
                    self.quant_tensor(&format!("{prefix}.{leaf}.weight"))?,
                )?)
            };
            let optional = |name: String| -> Option<Tensor> { self.plain.get(&name).cloned() };
            let norm = |name: String| -> Option<RmsNorm> {
                self.plain.get(&name).cloned().map(|weight| RmsNorm {
                    weight,
                    eps: params.rms_norm_eps,
                })
            };

            let self_attn = Qwen2Attention {
                q_proj: quant_matmul("attn_q")?,
                k_proj: quant_matmul("attn_k")?,
                v_proj: quant_matmul("attn_v")?,
                o_proj: quant_matmul("attn_output")?,
                q_bias: optional(format!("{prefix}.attn_q.bias")),
                k_bias: optional(format!("{prefix}.attn_k.bias")),
                v_bias: optional(format!("{prefix}.attn_v.bias")),
                q_norm: norm(format!("{prefix}.attn_q_norm.weight")),
                k_norm: norm(format!("{prefix}.attn_k_norm.weight")),
                num_heads: params.num_heads,
                num_kv_heads: params.num_kv_heads,
                kv_repeat: params.kv_repeat(),
                head_dim,
                hidden_size: params.hidden_size,
            };

            blocks.push(Qwen2Block {
                attn_norm: RmsNorm {
                    weight: self.plain_tensor(&format!("{prefix}.attn_norm.weight"))?,
                    eps: params.rms_norm_eps,
                },
                self_attn,
                ffn_norm: RmsNorm {
                    weight: self.plain_tensor(&format!("{prefix}.ffn_norm.weight"))?,
                    eps: params.rms_norm_eps,
                },
                ffn: SwiGluFFN {
                    gate: quant_matmul("ffn_gate")?,
                    up: quant_matmul("ffn_up")?,
                    down: quant_matmul("ffn_down")?,
                },
            });
        }

        Ok(GgufQwen2Modules {
            embedding,
            blocks,
            cos,
            sin,
        })
    }
}

const TOKEN_EMBD: &str = "token_embd.weight";

pub(crate) struct GgufQwen2TextEncoder {
    /// The park/unpark source of truth, retained only while the built modules
    /// share its quantized tensors. `None` means this encoder cannot park for
    /// free and falls back to drop-and-reload — see the module header.
    weights: Option<GgufQwen2Weights>,
    modules: GgufQwen2Modules,
    device: Device,
    dtype: DType,
}

impl GgufQwen2TextEncoder {
    fn forward_last_hidden_from_embeddings(
        &mut self,
        xs: Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
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

        let mut xs = xs;
        for block in &self.modules.blocks {
            xs = block.forward(
                &xs,
                &self.modules.cos,
                &self.modules.sin,
                attention_mask.as_ref(),
            )?;
        }
        Ok(xs)
    }

    /// Cold load: read the GGUF off disk onto `device` and build the modules.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let weights = GgufQwen2Weights::read(path, device)?;
        Self::from_weights(weights, device)
    }

    /// Warm load: rebuild from an already-read weight set, moving it to
    /// `device` first. This is the unpark path — no disk I/O (#1044).
    pub fn from_weights(weights: GgufQwen2Weights, device: &Device) -> Result<Self> {
        let weights = {
            let moved = weights.to_device(device)?;
            // A same-device move hands back the input `Arc`s, and a shadowed
            // binding would keep the source map alive to the end of this
            // function — inflating the strong counts the sharing check reads.
            drop(weights);
            moved
        };
        let modules = weights.build(device)?;
        // Retaining a map the modules did not share would be a second full
        // device-resident copy of the weights, which is worse than the reload
        // it avoids. Drop it and let this encoder take the disk path.
        let weights = weights.is_shared_with_built_modules().then_some(weights);
        if weights.is_none() {
            tracing::debug!(
                "GGUF Qwen2 encoder dequantized at build; not retaining the weight map (park unavailable)"
            );
        }
        Ok(Self {
            weights,
            modules,
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    /// Park: byte-move the weight set to host RAM and hand it back. The
    /// device-resident modules are released with `self`, so the VRAM the
    /// `QMatMul`s held is freed as soon as the caller drops the return of
    /// this call's consumed receiver.
    ///
    /// `Ok(None)` means this encoder never retained a map (the build
    /// dequantized), so there is nothing to park and the caller must drop.
    pub fn park_to_cpu(self) -> Result<Option<GgufQwen2Weights>> {
        self.weights
            .map(|weights| weights.to_device(&Device::Cpu))
            .transpose()
    }

    /// Bytes the retained weight set occupies on its current device, or `0`
    /// when no map is retained.
    pub fn weights_size_in_bytes(&self) -> u64 {
        self.weights
            .as_ref()
            .map_or(0, GgufQwen2Weights::size_in_bytes)
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
        let xs = self.modules.embedding.forward(input_ids)?;
        self.forward_last_hidden_from_embeddings(xs, attn_mask)
    }

    pub fn forward_last_hidden_with_image_embeds(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[(usize, usize)],
        image_embeds: &[Tensor],
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = self.modules.embedding.forward(input_ids)?;
        for ((start, end), embeds) in image_spans.iter().zip(image_embeds.iter()) {
            if embeds.dim(0)? != end - start {
                anyhow::bail!(
                    "image embedding length {} did not match placeholder span {}",
                    embeds.dim(0)?,
                    end - start
                );
            }
            let embeds = embeds.to_device(&self.device)?.to_dtype(self.dtype)?;
            xs = xs.slice_assign(
                &[0..1, *start..*end, 0..embeds.dim(1)?],
                &embeds.unsqueeze(0)?,
            )?;
        }
        self.forward_last_hidden_from_embeddings(xs, attn_mask)
    }
}

#[cfg(test)]
use candle_core::quantized::GgmlDType;

#[cfg(test)]
/// A tiny synthetic Qwen2 weight set, built straight as
/// [`GgufQwen2Weights`] rather than through a temporary GGUF file: the
/// park path never touches the file again, so the file adds nothing to
/// what is under test and a lot to the fixture.
///
/// Shapes are the smallest that satisfy the family's own constraints —
/// `head_dim` even for RoPE, and every quantized matmul's inner dimension
/// a multiple of 32 for Q4_0's block size.
///
/// `quant_dtype` is a parameter because candle's `QMatMul::from_arc`
/// dequantizes F32/F16/BF16 and shares everything else, and the sharing is
/// exactly what the retained weight map's cost claim rests on.
pub(crate) fn synthetic_weights(block_count: usize, quant_dtype: GgmlDType) -> GgufQwen2Weights {
    const HIDDEN: usize = 64;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const FFN: usize = 128;
    const VOCAB: usize = 16;
    let device = Device::Cpu;
    let head_dim = HIDDEN / HEADS;
    let kv_width = KV_HEADS * head_dim;

    let quantize = |rows: usize, cols: usize, seed: f64| -> Arc<QTensor> {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f64 * 0.017 + seed).sin() as f32)
            .collect();
        let tensor = Tensor::from_vec(values, (rows, cols), &device).unwrap();
        Arc::new(QTensor::quantize(&tensor, quant_dtype).unwrap())
    };
    let plain_tensor = |rows: usize, cols: usize, seed: f64| -> Tensor {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f64 * 0.031 + seed).cos() as f32)
            .collect();
        Tensor::from_vec(values, (rows, cols), &device).unwrap()
    };

    let mut quant = HashMap::new();
    let mut plain = HashMap::new();
    plain.insert(TOKEN_EMBD.to_string(), plain_tensor(VOCAB, HIDDEN, 0.5));

    for i in 0..block_count {
        let prefix = format!("blk.{i}");
        let seed = i as f64;
        quant.insert(
            format!("{prefix}.attn_q.weight"),
            quantize(HIDDEN, HIDDEN, seed + 1.0),
        );
        quant.insert(
            format!("{prefix}.attn_k.weight"),
            quantize(kv_width, HIDDEN, seed + 2.0),
        );
        quant.insert(
            format!("{prefix}.attn_v.weight"),
            quantize(kv_width, HIDDEN, seed + 3.0),
        );
        quant.insert(
            format!("{prefix}.attn_output.weight"),
            quantize(HIDDEN, HIDDEN, seed + 4.0),
        );
        quant.insert(
            format!("{prefix}.ffn_gate.weight"),
            quantize(FFN, HIDDEN, seed + 5.0),
        );
        quant.insert(
            format!("{prefix}.ffn_up.weight"),
            quantize(FFN, HIDDEN, seed + 6.0),
        );
        quant.insert(
            format!("{prefix}.ffn_down.weight"),
            quantize(HIDDEN, FFN, seed + 7.0),
        );
        plain.insert(
            format!("{prefix}.attn_norm.weight"),
            plain_tensor(1, HIDDEN, seed + 8.0).squeeze(0).unwrap(),
        );
        plain.insert(
            format!("{prefix}.ffn_norm.weight"),
            plain_tensor(1, HIDDEN, seed + 9.0).squeeze(0).unwrap(),
        );
        plain.insert(
            format!("{prefix}.attn_q.bias"),
            plain_tensor(1, HIDDEN, seed + 10.0).squeeze(0).unwrap(),
        );
        plain.insert(
            format!("{prefix}.attn_k.bias"),
            plain_tensor(1, kv_width, seed + 11.0).squeeze(0).unwrap(),
        );
        plain.insert(
            format!("{prefix}.attn_v.bias"),
            plain_tensor(1, kv_width, seed + 12.0).squeeze(0).unwrap(),
        );
    }

    GgufQwen2Weights {
        params: GgufQwen2Params {
            hidden_size: HIDDEN,
            num_heads: HEADS,
            num_kv_heads: KV_HEADS,
            block_count,
            rope_positions: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        },
        quant,
        plain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The premise of the whole GGUF park path (#1044): the bytes that come
    /// back are the bytes that went out. A dequantize/re-quantize round trip
    /// would pass a loose tolerance check and still make a render depend on
    /// whether the encoder happened to be parked, so this asserts raw storage
    /// equality rather than closeness.
    ///
    /// Deliberately drives `rebuilt_on`, not `to_device`: CI has no GPU, and a
    /// same-device move returns the input `Arc` untouched, so a test written
    /// against `to_device` would compare a tensor with itself and pass no
    /// matter how broken the byte path was. Same reasoning as
    /// `wan::block_offload::parking_a_block_is_byte_identical`.
    #[test]
    fn parking_the_gguf_encoder_is_byte_identical() {
        let weights = synthetic_weights(2, GgmlDType::Q4_0);
        let parked = weights.rebuilt_on(&Device::Cpu).unwrap();

        assert_eq!(
            parked.quant.len(),
            weights.quant.len(),
            "a park must keep every quantized tensor"
        );
        for (name, before) in &weights.quant {
            let after = parked.quant.get(name).expect("parked set keeps the name");
            assert_eq!(before.dtype(), after.dtype(), "{name} changed quantization");
            assert_eq!(before.shape(), after.shape(), "{name} changed shape");
            assert_eq!(
                before.data().unwrap().as_ref(),
                after.data().unwrap().as_ref(),
                "{name} is not byte-identical after a park/unpark cycle"
            );
        }
        for (name, before) in &weights.plain {
            let after = parked.plain.get(name).expect("parked set keeps the name");
            assert_eq!(
                before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "{name} changed value after a park/unpark cycle"
            );
        }
    }

    /// The consequence that actually matters: an encoder rebuilt from parked
    /// weights produces bit-identical hidden states. This is what lets the
    /// pipeline skip the 35.1 s reload without changing a render.
    #[test]
    fn an_unparked_gguf_encoder_produces_identical_hidden_states() {
        let device = Device::Cpu;
        let weights = synthetic_weights(2, GgmlDType::Q4_0);
        let input_ids = Tensor::from_vec(vec![3u32, 7, 1, 9, 4], (1, 5), &device).unwrap();

        let mut before =
            GgufQwen2TextEncoder::from_weights(weights.rebuilt_on(&device).unwrap(), &device)
                .unwrap();
        let hidden_before = before.forward_last_hidden(&input_ids, None).unwrap();

        // Park (bytes to host) and unpark (bytes back), then run again.
        let parked = before
            .park_to_cpu()
            .unwrap()
            .expect("a k-quant encoder retains its weight map");
        let mut after =
            GgufQwen2TextEncoder::from_weights(parked.rebuilt_on(&device).unwrap(), &device)
                .unwrap();
        let hidden_after = after.forward_last_hidden(&input_ids, None).unwrap();

        assert_eq!(hidden_before.dims(), hidden_after.dims());
        assert_eq!(
            hidden_before
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            hidden_after
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            "a park/unpark cycle must not change the encoder's output"
        );
    }

    /// Retaining the weight map is only free because the blocks share the very
    /// same `Arc<QTensor>`. If `build` ever copied instead, a resident encoder
    /// would silently double its quantized footprint on the device.
    #[test]
    fn building_shares_the_quantized_tensors_rather_than_copying_them() {
        let device = Device::Cpu;
        let weights = synthetic_weights(1, GgmlDType::Q4_0);
        let before = Arc::strong_count(weights.quant.get("blk.0.attn_q.weight").unwrap());
        let _modules = weights.build(&device).unwrap();
        let after = Arc::strong_count(weights.quant.get("blk.0.attn_q.weight").unwrap());
        assert!(
            after > before,
            "the built QMatMul must hold an Arc clone, not a copy"
        );
        assert!(
            weights.is_shared_with_built_modules(),
            "a k-quant build shares every quantized tensor"
        );
    }

    /// The other half of that claim, which the Q4_0 fixture cannot reach:
    /// `QMatMul::from_arc` dequantizes an F16 GGUF tensor into a dense
    /// `Tensor` and drops the `Arc`, so retaining the map would be a second
    /// full copy of the weights on the device. The encoder must notice and
    /// keep no map at all — a park that doubles VRAM is worse than the disk
    /// reload it avoids.
    #[test]
    fn an_f16_gguf_build_does_not_retain_a_duplicate_weight_map() {
        let device = Device::Cpu;
        let weights = synthetic_weights(1, GgmlDType::F16);
        let modules = weights.build(&device).unwrap();
        assert!(
            !weights.is_shared_with_built_modules(),
            "candle dequantizes F16, so nothing shares the quantized Arc"
        );
        drop(modules);

        let encoder = GgufQwen2TextEncoder::from_weights(weights, &device).unwrap();
        assert_eq!(
            encoder.weights_size_in_bytes(),
            0,
            "an encoder that cannot share must retain no map"
        );
        assert!(
            encoder.park_to_cpu().unwrap().is_none(),
            "with no retained map there is nothing to park"
        );
    }

    /// The park's host-RAM accounting has to count both halves of the map, or
    /// a 7B encoder looks free.
    #[test]
    fn size_in_bytes_counts_quantized_and_plain_halves() {
        let weights = synthetic_weights(1, GgmlDType::Q4_0);
        let quant: u64 = weights
            .quant
            .values()
            .map(|t| t.storage_size_in_bytes() as u64)
            .sum();
        let plain: u64 = weights
            .plain
            .values()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum();
        assert!(quant > 0 && plain > 0);
        assert_eq!(weights.size_in_bytes(), quant + plain);
    }

    #[test]
    fn rope_cache_covers_qwen_image_padded_sequence_window() {
        let device = Device::Cpu;
        let head_dim = 64;
        let seq_len = 1056;
        let (cos, sin) = compute_rope(head_dim, 1_000_000.0, MAX_ROPE_POSITIONS, &device).unwrap();
        let x = Tensor::zeros((1, 2, seq_len, head_dim), DType::F32, &device).unwrap();
        let rotated = apply_rotary_emb(&x, &cos, &sin, head_dim).unwrap();
        assert_eq!(rotated.dims4().unwrap(), (1, 2, seq_len, head_dim));
    }
}
