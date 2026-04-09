//! Qwen2.5-VL text encoder adapter for Qwen-Image-2512.
//!
//! Qwen-Image uses the Qwen2.5-VL text stack, but its image transformer expects
//! the same conditioning layout as the upstream Diffusers pipeline:
//! - chat-style prompt formatting
//! - tokenizer-side truncation at 1024 post-template tokens
//! - prompt embeddings truncated to 512 tokens after template stripping
//! - last hidden states (no final RMSNorm)

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::qwen2_text_gguf::GgufQwen2TextEncoder;
use super::qwen2_vision::{Qwen2VisionConfig, Qwen2VisionModel};

const TOKENIZER_WINDOW: usize = 1024;
const MAX_SEQUENCE_LENGTH: usize = 512;
const SYSTEM_PROMPT: &str = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

fn format_qwen_image_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn template_strip_index(tokens: &[u32]) -> usize {
    const IM_START: u32 = 151644;
    const USER: u32 = 872;
    const NEWLINE: u32 = 198;

    let mut im_start_count = 0;
    let mut strip_at = 0;
    for (idx, &token) in tokens.iter().enumerate() {
        if token == IM_START {
            im_start_count += 1;
            if im_start_count == 2 {
                strip_at = idx;
                break;
            }
        }
    }

    if tokens.get(strip_at + 1) == Some(&USER) && tokens.get(strip_at + 2) == Some(&NEWLINE) {
        strip_at += 3;
    }

    strip_at
}

fn window_qwen_image_tokens(
    mut input_ids: Vec<u32>,
    strip_idx: usize,
    pad_id: u32,
) -> (Vec<u32>, usize) {
    let total_window = TOKENIZER_WINDOW + strip_idx;
    if input_ids.len() > total_window {
        input_ids.truncate(total_window);
    }
    let valid_len = input_ids
        .len()
        .saturating_sub(strip_idx)
        .min(TOKENIZER_WINDOW)
        .min(MAX_SEQUENCE_LENGTH);
    input_ids.resize(total_window, pad_id);
    (input_ids, valid_len)
}

fn expand_image_pad_tokens(
    input_ids: &[u32],
    image_pad_id: u32,
    image_token_counts: &[usize],
) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
    let mut expanded = Vec::with_capacity(
        input_ids.len()
            + image_token_counts
                .iter()
                .sum::<usize>()
                .saturating_sub(image_token_counts.len()),
    );
    let mut spans = Vec::with_capacity(image_token_counts.len());
    let mut image_idx = 0usize;
    for &token in input_ids {
        if token == image_pad_id {
            let Some(&count) = image_token_counts.get(image_idx) else {
                bail!("multimodal prompt contained more <|image_pad|> tokens than input images");
            };
            let start = expanded.len();
            expanded.extend(std::iter::repeat_n(image_pad_id, count));
            spans.push((start, expanded.len()));
            image_idx += 1;
        } else {
            expanded.push(token);
        }
    }
    if image_idx != image_token_counts.len() {
        bail!(
            "multimodal prompt referenced {} images but only {} <|image_pad|> placeholders were found",
            image_token_counts.len(),
            image_idx
        );
    }
    Ok((expanded, spans))
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen2TextEncoderConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: candle_transformers::models::with_tracing::Linear,
    up_proj: candle_transformers::models::with_tracing::Linear,
    down_proj: candle_transformers::models::with_tracing::Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Qwen2TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = candle_transformers::models::with_tracing::linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("gate_proj"),
        )?;
        let up_proj = candle_transformers::models::with_tracing::linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("up_proj"),
        )?;
        let down_proj = candle_transformers::models::with_tracing::linear_no_bias(
            intermediate_sz,
            hidden_sz,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: candle_nn::Activation::Silu,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: candle_transformers::models::with_tracing::Linear,
    k_proj: candle_transformers::models::with_tracing::Linear,
    v_proj: candle_transformers::models::with_tracing::Linear,
    o_proj: candle_transformers::models::with_tracing::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Qwen2TextEncoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = candle_transformers::models::with_tracing::linear(
            hidden_sz,
            num_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_transformers::models::with_tracing::linear(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_transformers::models::with_tracing::linear(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_transformers::models::with_tracing::linear_no_bias(
            num_heads * head_dim,
            hidden_sz,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let key_states =
            candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: candle_transformers::models::with_tracing::RmsNorm,
    post_attention_layernorm: candle_transformers::models::with_tracing::RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Qwen2TextEncoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = candle_transformers::models::with_tracing::RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_transformers::models::with_tracing::RmsNorm::new(
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
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        (residual + xs).map_err(Into::into)
    }
}

pub(crate) struct Bf16Qwen2TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    sliding_window: usize,
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl Bf16Qwen2TextModel {
    fn new(cfg: &Qwen2TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            sliding_window: cfg.max_position_embeddings,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
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

    fn forward_last_hidden(
        &self,
        input_ids: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, 0)?)
                }
            }
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        // Return the LAST hidden layer output (hidden_states[-1]),
        // matching diffusers pipeline_qwenimage.py line 210.
        // Previously used penultimate (layer N-2) which was wrong.
        let target_layer = self.layers.len().saturating_sub(1);
        for (idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, attention_mask.as_ref(), 0)?;
            if idx == target_layer {
                return Ok(xs);
            }
        }
        anyhow::bail!("Qwen2 text model has too few layers")
    }

    fn forward_last_hidden_with_image_embeds(
        &self,
        input_ids: &Tensor,
        image_spans: &[(usize, usize)],
        image_embeds: &[Tensor],
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, 0)?)
                }
            }
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for ((start, end), embeds) in image_spans.iter().zip(image_embeds.iter()) {
            if embeds.dim(0)? != end - start {
                bail!(
                    "image embedding length {} did not match placeholder span {}",
                    embeds.dim(0)?,
                    end - start
                );
            }
            let embeds = embeds.to_device(&self.device)?.to_dtype(self.dtype)?;
            xs = xs.slice_assign(
                &[0..1, *start..*end, 0..self.hidden_size],
                &embeds.unsqueeze(0)?,
            )?;
        }

        let target_layer = self.layers.len().saturating_sub(1);
        for (idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, attention_mask.as_ref(), 0)?;
            if idx == target_layer {
                return Ok(xs);
            }
        }
        bail!("Qwen2 text model has too few layers")
    }
}

/// Qwen2.5 text encoder configuration for Qwen-Image-2512.
pub(crate) struct Qwen2TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Default for Qwen2TextEncoderConfig {
    fn default() -> Self {
        Self::qwen_image()
    }
}

impl Qwen2TextEncoderConfig {
    pub fn qwen_image() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 128000,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

/// Loaded Qwen2.5 text encoder.
pub(crate) enum Qwen2TextModel {
    Bf16(Bf16Qwen2TextModel),
    Quantized(GgufQwen2TextEncoder),
}

impl Qwen2TextModel {
    fn forward_last_hidden(
        &mut self,
        input_ids: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16(model) => model.forward_last_hidden(input_ids, attn_mask),
            Self::Quantized(model) => model.forward_last_hidden(input_ids, attn_mask),
        }
    }

    fn forward_last_hidden_with_image_embeds(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[(usize, usize)],
        image_embeds: &[Tensor],
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16(model) => model.forward_last_hidden_with_image_embeds(
                input_ids,
                image_spans,
                image_embeds,
                attn_mask,
            ),
            Self::Quantized(model) => model.forward_last_hidden_with_image_embeds(
                input_ids,
                image_spans,
                image_embeds,
                attn_mask,
            ),
        }
    }
}

pub(crate) struct Qwen2TextEncoder {
    pub model: Option<Qwen2TextModel>,
    vision: Option<Qwen2VisionModel>,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub on_gpu: bool,
    pub is_quantized: bool,
    encoder_paths: Vec<PathBuf>,
    vision_encoder_paths: Vec<PathBuf>,
    dtype: DType,
    config: Qwen2TextEncoderConfig,
}

impl Qwen2TextEncoder {
    fn load_tokenizer(tokenizer_path: &PathBuf) -> Result<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2.5 tokenizer: {e}"))
    }

    fn build_vision(vb: VarBuilder) -> Result<Qwen2VisionModel> {
        Qwen2VisionModel::new(&Qwen2VisionConfig::qwen25_vl(), vb.pp("visual"))
    }

    fn load_vision_from_paths(
        encoder_paths: &[PathBuf],
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Qwen2VisionModel> {
        let vb = crate::weight_loader::load_safetensors_with_filtered_progress(
            encoder_paths,
            dtype,
            device,
            "Qwen2.5-VL vision encoder",
            progress,
            |name| name.starts_with("visual."),
        )?;
        Self::build_vision(vb)
    }

    pub fn prepare_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        enable_vision: bool,
    ) -> Result<Self> {
        let config = Qwen2TextEncoderConfig::qwen_image();
        let tokenizer = Self::load_tokenizer(tokenizer_path)?;
        let on_gpu = crate::device::is_gpu(device);
        Ok(Self {
            model: None,
            vision: None,
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized: false,
            encoder_paths: encoder_paths.to_vec(),
            vision_encoder_paths: if enable_vision {
                encoder_paths.to_vec()
            } else {
                Vec::new()
            },
            dtype,
            config,
        })
    }

    pub fn prepare_gguf(
        gguf_path: &Path,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        vision_encoder_paths: &[PathBuf],
    ) -> Result<Self> {
        let config = Qwen2TextEncoderConfig::qwen_image();
        let tokenizer = Self::load_tokenizer(tokenizer_path)?;
        let on_gpu = crate::device::is_gpu(device);
        Ok(Self {
            model: None,
            vision: None,
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized: true,
            encoder_paths: vec![gguf_path.to_path_buf()],
            vision_encoder_paths: vision_encoder_paths.to_vec(),
            dtype,
            config,
        })
    }

    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        enable_vision: bool,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            encoder_paths,
            dtype,
            device,
            "Qwen2.5-VL encoder",
            progress,
        )?;
        let mut encoder =
            Self::prepare_bf16(encoder_paths, tokenizer_path, device, dtype, enable_vision)?;
        if enable_vision {
            encoder.vision = Some(Self::build_vision(vb.clone())?);
        }
        encoder.model = Some(Qwen2TextModel::Bf16(Bf16Qwen2TextModel::new(
            &encoder.config,
            vb,
        )?));
        Ok(encoder)
    }

    pub fn load_gguf(
        gguf_path: &Path,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        vision_encoder_paths: &[PathBuf],
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let mut encoder = Self::prepare_gguf(
            gguf_path,
            tokenizer_path,
            device,
            dtype,
            vision_encoder_paths,
        )?;
        encoder.model = Some(Qwen2TextModel::Quantized(GgufQwen2TextEncoder::load(
            gguf_path, device,
        )?));
        if !encoder.vision_encoder_paths.is_empty() {
            encoder.vision = Some(Self::load_vision_from_paths(
                &encoder.vision_encoder_paths,
                device,
                dtype,
                progress,
            )?);
        }
        Ok(encoder)
    }

    fn encode_ids_from_formatted(&self, formatted: &str) -> Result<(Vec<u32>, usize, usize)> {
        let input_ids = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|e| anyhow::anyhow!("Qwen2.5 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        let strip_idx = template_strip_index(&input_ids);

        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 tokenizer missing <|endoftext|>"))?;

        let (input_ids, valid_len) = window_qwen_image_tokens(input_ids, strip_idx, pad_id);
        Ok((input_ids, strip_idx, valid_len))
    }

    fn encode_ids(&self, prompt: &str) -> Result<(Vec<u32>, usize, usize)> {
        let formatted = format_qwen_image_prompt(prompt);
        self.encode_ids_from_formatted(&formatted)
    }

    fn encode_token_window(
        &mut self,
        tokens: Vec<u32>,
        strip_idx: usize,
        valid_len: usize,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 model unavailable (weights dropped)"))?;
        let total_window = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, total_window), &self.device)?;
        let mut mask = vec![0u8; total_window];
        for value in &mut mask[..(strip_idx + valid_len)] {
            *value = 1;
        }
        let attn_mask = Tensor::from_vec(mask, (1, total_window), &self.device)?;

        let emb = model
            .forward_last_hidden(&input_ids, Some(&attn_mask))?
            .narrow(1, strip_idx, valid_len)?;

        let text_mask = Tensor::ones((1, valid_len), DType::U8, &self.device)?;

        Ok((
            emb.to_device(target_device)?.to_dtype(target_dtype)?,
            text_mask.to_device(target_device)?,
            valid_len,
        ))
    }

    /// Returns variable-length embeddings and a matching mask after stripping
    /// the system prefix, matching the upstream diffusers pipeline.
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let (tokens, strip_idx, valid_len) = self.encode_ids(prompt)?;
        self.encode_token_window(tokens, strip_idx, valid_len, target_device, target_dtype)
    }

    pub fn encode_formatted_multimodal(
        &mut self,
        formatted_prompt: &str,
        images: &[Vec<u8>],
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let image_pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|image_pad|>")
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 tokenizer missing <|image_pad|>"))?;
        let input_ids = self
            .tokenizer
            .encode(formatted_prompt, false)
            .map_err(|e| anyhow::anyhow!("Qwen2.5 multimodal tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 model unavailable (weights dropped)"))?;
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 vision encoder was not loaded"))?;

        let image_embeds = images
            .iter()
            .map(|image| vision.encode_image_bytes(image, &self.device, self.dtype))
            .collect::<Result<Vec<_>>>()?;
        let image_token_counts = image_embeds
            .iter()
            .map(|embeds| embeds.dim(0))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let (expanded_ids, image_spans) =
            expand_image_pad_tokens(&input_ids, image_pad_id, &image_token_counts)?;
        let strip_idx = template_strip_index(&expanded_ids);
        if expanded_ids.len().saturating_sub(strip_idx) > TOKENIZER_WINDOW {
            bail!(
                "Qwen2.5 multimodal prompt exceeded the {} token window after image expansion",
                TOKENIZER_WINDOW
            );
        }
        let valid_len = expanded_ids.len().saturating_sub(strip_idx);
        let input_ids = Tensor::from_vec(expanded_ids, (1, strip_idx + valid_len), &self.device)?;
        let attention_mask = Tensor::ones((1, strip_idx + valid_len), DType::U8, &self.device)?;
        let hidden_states = model
            .forward_last_hidden_with_image_embeds(
                &input_ids,
                &image_spans,
                &image_embeds,
                Some(&attention_mask),
            )?
            .narrow(1, strip_idx, valid_len)?;
        let attention_mask = Tensor::ones((1, valid_len), DType::U8, &self.device)?;
        Ok((
            hidden_states
                .to_device(target_device)?
                .to_dtype(target_dtype)?,
            attention_mask.to_device(target_device)?,
            valid_len,
        ))
    }

    pub fn drop_weights(&mut self) {
        self.model = None;
        self.vision = None;
    }

    pub fn reload(&mut self, progress: &crate::progress::ProgressReporter) -> Result<()> {
        self.model = if self.is_quantized {
            Some(Qwen2TextModel::Quantized(GgufQwen2TextEncoder::load(
                &self.encoder_paths[0],
                &self.device,
            )?))
        } else {
            let vb = crate::weight_loader::load_safetensors_with_progress(
                &self.encoder_paths,
                self.dtype,
                &self.device,
                "Qwen2.5-VL encoder",
                progress,
            )?;
            if !self.vision_encoder_paths.is_empty() {
                self.vision = Some(Self::build_vision(vb.clone())?);
            }
            Some(Qwen2TextModel::Bf16(Bf16Qwen2TextModel::new(
                &self.config,
                vb,
            )?))
        };
        if self.is_quantized && !self.vision_encoder_paths.is_empty() {
            self.vision = Some(Self::load_vision_from_paths(
                &self.vision_encoder_paths,
                &self.device,
                self.dtype,
                progress,
            )?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_image_prompt_format_includes_chat_template() {
        let formatted = format_qwen_image_prompt("a red apple");
        assert!(formatted.starts_with("<|im_start|>system\n"));
        assert!(formatted.contains(SYSTEM_PROMPT));
        assert!(formatted.contains("<|im_start|>user\na red apple<|im_end|>"));
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn template_strip_index_skips_second_im_start_user_prefix() {
        let tokens = vec![1, 2, 151644, 999, 151644, 872, 198, 77, 88];
        assert_eq!(template_strip_index(&tokens), 7);
    }

    #[test]
    fn window_qwen_image_tokens_truncates_to_1024_and_caps_valid_len_at_512() {
        let strip_idx = 4;
        let input_ids = (0..2_000).collect::<Vec<u32>>();
        let (windowed, valid_len) = window_qwen_image_tokens(input_ids, strip_idx, 42);
        assert_eq!(windowed.len(), TOKENIZER_WINDOW + strip_idx);
        assert_eq!(valid_len, MAX_SEQUENCE_LENGTH);
        assert_eq!(windowed[TOKENIZER_WINDOW + strip_idx - 1], 1027);
    }

    #[test]
    fn window_qwen_image_tokens_pads_short_sequences_after_template_strip() {
        let strip_idx = 3;
        let input_ids = vec![10, 11, 12, 13, 14];
        let (windowed, valid_len) = window_qwen_image_tokens(input_ids, strip_idx, 99);
        assert_eq!(valid_len, 2);
        assert_eq!(windowed.len(), TOKENIZER_WINDOW + strip_idx);
        assert_eq!(&windowed[..5], &[10, 11, 12, 13, 14]);
        assert!(windowed[5..].iter().all(|&id| id == 99));
    }

    #[test]
    fn expand_image_pad_tokens_repeats_each_placeholder_with_span_tracking() {
        let (expanded, spans) = expand_image_pad_tokens(&[1, 9, 2, 9, 3], 9, &[4, 2]).unwrap();
        assert_eq!(expanded, vec![1, 9, 9, 9, 9, 2, 9, 9, 3]);
        assert_eq!(spans, vec![(1, 5), (6, 8)]);
    }
}
