//! Qwen2.5-VL text encoder adapter for Qwen-Image-2512.
//!
//! Qwen-Image uses the Qwen2.5-VL text stack, but its image transformer expects
//! the same conditioning layout as the upstream Diffusers pipeline:
//! - chat-style prompt formatting
//! - padded text conditioning to a fixed 1024-token window
//! - penultimate hidden states (no final RMSNorm)

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;

const TEXT_WINDOW: usize = 1024;
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

#[derive(Debug, Clone)]
pub(crate) struct Qwen2TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Qwen2TextModel {
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
        let mask = Tensor::cat(&mask.iter().collect::<Vec<_>>(), 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false).map_err(Into::into)
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
pub(crate) struct Qwen2TextEncoder {
    pub model: Option<Qwen2TextModel>,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub on_gpu: bool,
    encoder_paths: Vec<PathBuf>,
    dtype: DType,
    config: Qwen2TextEncoderConfig,
}

impl Qwen2TextEncoder {
    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let config = Qwen2TextEncoderConfig::qwen_image();
        let vb = crate::weight_loader::load_safetensors_with_progress(
            encoder_paths,
            dtype,
            device,
            "Qwen2.5-VL encoder",
            progress,
        )?;
        let model = Qwen2TextModel::new(&config, vb)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2.5 tokenizer: {e}"))?;
        let on_gpu = crate::device::is_gpu(device);
        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            encoder_paths: encoder_paths.to_vec(),
            dtype,
            config,
        })
    }

    fn encode_ids(&self, prompt: &str) -> Result<(Vec<u32>, usize, usize)> {
        let formatted = format_qwen_image_prompt(prompt);
        let mut input_ids = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|e| anyhow::anyhow!("Qwen2.5 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        let strip_idx = template_strip_index(&input_ids);
        let total_window = TEXT_WINDOW + strip_idx;

        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 tokenizer missing <|endoftext|>"))?;

        if input_ids.len() > total_window {
            input_ids.truncate(total_window);
        }
        let valid_len = input_ids.len().saturating_sub(strip_idx).min(TEXT_WINDOW);
        input_ids.resize(total_window, pad_id);
        Ok((input_ids, strip_idx, valid_len))
    }

    /// Returns fixed-width embeddings and a matching mask after stripping the
    /// system prefix. The output sequence length remains the model's expected
    /// 1024 tokens even for short prompts.
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, Tensor, usize)> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 model unavailable (weights dropped)"))?;
        let (tokens, strip_idx, valid_len) = self.encode_ids(prompt)?;
        let total_window = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, total_window), &self.device)?;
        let mut mask = vec![0u8; total_window];
        for value in &mut mask[..(strip_idx + valid_len)] {
            *value = 1;
        }
        let attn_mask = Tensor::from_vec(mask, (1, total_window), &self.device)?;

        let emb = model
            .forward_last_hidden(&input_ids, Some(&attn_mask))?
            .narrow(1, strip_idx, TEXT_WINDOW)?;

        let mut text_mask = vec![0u8; TEXT_WINDOW];
        for value in &mut text_mask[..valid_len] {
            *value = 1;
        }
        let text_mask = Tensor::from_vec(text_mask, (1, TEXT_WINDOW), &self.device)?;
        let emb = emb.broadcast_mul(&text_mask.to_dtype(emb.dtype())?.unsqueeze(D::Minus1)?)?;

        Ok((
            emb.to_device(target_device)?.to_dtype(target_dtype)?,
            text_mask.to_device(target_device)?,
            valid_len,
        ))
    }

    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    pub fn reload(&mut self, progress: &crate::progress::ProgressReporter) -> Result<()> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            &self.encoder_paths,
            self.dtype,
            &self.device,
            "Qwen2.5-VL encoder",
            progress,
        )?;
        self.model = Some(Qwen2TextModel::new(&self.config, vb)?);
        Ok(())
    }
}
