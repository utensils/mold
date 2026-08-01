//! Streamed Mistral3 language encoder for FLUX.2 [dev].
//!
//! The upstream checkpoint contains a 40-layer, 24B-class language model, but
//! FLUX conditioning consumes only hidden states after layers 10, 20, and 30.
//! This implementation mmaps the eight shards containing that exact prefix and
//! materializes one decoder layer at a time on the target device. Later layers,
//! the final norm, vision tower, projector, and LM head are intentionally absent.
//! This matches BFL's `Mistral3SmallEmbedder::forward`, which passes text-only
//! chat-template inputs to the language model; editing references enter the
//! denoiser separately through `prepare_reference` VAE latent tokens. See
//! `src/flux2/text_encoder.py` and `src/flux2/sampling.py` in BFL's `flux2`
//! reference repository.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

const VOCAB_SIZE: usize = 131_072;
const HIDDEN_SIZE: usize = 5_120;
const INTERMEDIATE_SIZE: usize = 32_768;
const NUM_ATTENTION_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const KV_REPEAT: usize = NUM_ATTENTION_HEADS / NUM_KV_HEADS;
const ROPE_THETA: f64 = 1_000_000_000.0;
const RMS_NORM_EPS: f64 = 1e-5;
const MAX_LENGTH: usize = 512;
const PAD_TOKEN_ID: u32 = 11;
const CAPTURE_LAYERS: [usize; 3] = [9, 19, 29];

/// Largest simultaneously resident weight allocation in the streamed encoder:
/// the 131072 x 5120 token embedding table. Decoder layers are smaller.
pub(crate) fn streamed_peak_weight_bytes(dtype: DType) -> u64 {
    (VOCAB_SIZE as u64)
        .saturating_mul(HIDDEN_SIZE as u64)
        .saturating_mul(crate::device::dtype_bytes(dtype) as u64)
}

const SYSTEM_PROMPT: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.";

fn format_prompt(prompt: &str) -> String {
    let prompt = prompt.replace("[IMG]", "");
    format!("<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{prompt}[/INST]")
}

struct RmsNorm {
    weight: Tensor,
}

impl RmsNorm {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(HIDDEN_SIZE, "weight")?,
        })
    }

    /// Match Transformers' Mistral RMSNorm: variance and normalization are
    /// performed in float32, then cast back before applying the learned scale.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + RMS_NORM_EPS)?.sqrt()?)?;
        xs.to_dtype(dtype)?
            .broadcast_mul(&self.weight)
            .map_err(Into::into)
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, device: &Device) -> Result<Self> {
        let inv_freq = (0..HEAD_DIM)
            .step_by(2)
            .map(|i| 1f32 / ROPE_THETA.powf(i as f64 / HEAD_DIM as f64) as f32)
            .collect::<Vec<_>>();
        let inv_freq = Tensor::from_vec(inv_freq, (1, HEAD_DIM / 2), device)?;
        let positions = Tensor::arange(0u32, MAX_LENGTH as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_LENGTH, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?,
        ))
    }
}

fn repeat_kv(x: Tensor) -> Result<Tensor> {
    let (batch, heads, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((batch, heads, KV_REPEAT, seq, dim))?
        .reshape((batch, heads * KV_REPEAT, seq, dim))
        .map_err(Into::into)
}

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                NUM_ATTENTION_HEADS * HEAD_DIM,
                vb.pp("q_proj"),
            )?,
            k_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                NUM_KV_HEADS * HEAD_DIM,
                vb.pp("k_proj"),
            )?,
            v_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                NUM_KV_HEADS * HEAD_DIM,
                vb.pp("v_proj"),
            )?,
            o_proj: candle_nn::linear_no_bias(
                NUM_ATTENTION_HEADS * HEAD_DIM,
                HIDDEN_SIZE,
                vb.pp("o_proj"),
            )?,
            rotary,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, seq, NUM_ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((batch, seq, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((batch, seq, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary.apply(&q, &k)?;
        let k = repeat_kv(k)?.contiguous()?;
        let v = repeat_kv(v)?.contiguous()?;
        let scores = (q.matmul(&k.transpose(2, 3)?)? / (HEAD_DIM as f64).sqrt())?;
        let scores = scores.broadcast_add(mask)?;
        let context = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        context
            .transpose(1, 2)?
            .reshape((batch, seq, NUM_ATTENTION_HEADS * HEAD_DIM))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, INTERMEDIATE_SIZE, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&xs.apply(&self.gate_proj)?)?;
        (gate * xs.apply(&self.up_proj)?)?
            .apply(&self.down_proj)
            .map_err(Into::into)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(vb.pp("input_layernorm"))?,
            self_attn: Attention::new(rotary, vb.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::new(vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = (residual + self.self_attn.forward(&xs, mask)?)?;
        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        (residual + self.mlp.forward(&hidden)?).map_err(Into::into)
    }
}

pub(crate) struct Mistral3Encoder {
    encoder_paths: Vec<PathBuf>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    dtype: DType,
}

impl Mistral3Encoder {
    pub fn load(
        encoder_paths: &[PathBuf],
        tokenizer: Arc<Tokenizer>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if encoder_paths.len() != 8 {
            anyhow::bail!(
                "FLUX.2 [dev] requires the first 8 Mistral3 runtime shards (model-00001 through model-00008), found {} configured shard(s)",
                encoder_paths.len()
            );
        }
        Ok(Self {
            encoder_paths: encoder_paths.to_vec(),
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn encode(
        &self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, usize)> {
        let formatted = format_prompt(prompt);
        let encoding = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|error| anyhow::anyhow!("Mistral3 tokenization failed: {error}"))?;
        let token_count = encoding.len().min(MAX_LENGTH);
        let mut tokens = encoding.get_ids()[..token_count].to_vec();
        tokens.resize(MAX_LENGTH, PAD_TOKEN_ID);
        let attention = (0..MAX_LENGTH)
            .map(|index| index < token_count)
            .collect::<Vec<_>>();

        let vb = crate::weight_loader::load_safetensors_with_progress(
            &self.encoder_paths,
            self.dtype,
            &self.device,
            "FLUX.2 [dev] Mistral3 encoder",
            &crate::progress::ProgressReporter::default(),
        )?
        .pp("language_model")
        .pp("model");

        let input_ids = Tensor::from_vec(tokens, (1, MAX_LENGTH), &self.device)?;
        let mut hidden = {
            let embedding = candle_nn::embedding(VOCAB_SIZE, HIDDEN_SIZE, vb.pp("embed_tokens"))?;
            embedding.forward(&input_ids)?
        };
        self.device.synchronize()?;

        let mask = causal_padding_mask(&attention, self.dtype, &self.device)?;
        let rotary = Arc::new(RotaryEmbedding::new(self.dtype, &self.device)?);
        let mut captured = Vec::with_capacity(CAPTURE_LAYERS.len());
        for layer_index in 0..=*CAPTURE_LAYERS.last().unwrap() {
            let layer = DecoderLayer::new(rotary.clone(), vb.pp("layers").pp(layer_index))
                .with_context(|| format!("loading Mistral3 decoder layer {layer_index}"))?;
            hidden = layer
                .forward(&hidden, &mask)
                .with_context(|| format!("running Mistral3 decoder layer {layer_index}"))?;
            self.device.synchronize()?;
            drop(layer);
            if CAPTURE_LAYERS.contains(&layer_index) {
                captured.push(hidden.clone());
            }
        }
        let output = Tensor::cat(&captured, D::Minus1)?
            .to_device(target_device)?
            .to_dtype(target_dtype)?;
        Ok((output, token_count))
    }
}

fn causal_padding_mask(attention: &[bool], dtype: DType, device: &Device) -> Result<Tensor> {
    if attention.len() != MAX_LENGTH {
        anyhow::bail!("Mistral3 attention mask must contain exactly {MAX_LENGTH} positions");
    }
    let values = (0..MAX_LENGTH)
        .flat_map(|query| {
            (0..MAX_LENGTH).map(move |key| {
                if key <= query && attention[key] {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, MAX_LENGTH, MAX_LENGTH), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn prompt_matches_official_flux2_dev_template_and_removes_image_markers() {
        assert_eq!(
            format_prompt("put [IMG] beside [IMG] the tree"),
            format!(
                "<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]put  beside  the tree[/INST]"
            )
        );
    }

    #[test]
    fn hidden_state_indices_are_after_layers_ten_twenty_and_thirty() {
        assert_eq!(CAPTURE_LAYERS, [9, 19, 29]);
    }

    #[test]
    fn streamed_peak_accounts_for_the_resolved_weight_dtype() {
        assert_eq!(
            streamed_peak_weight_bytes(DType::F32),
            streamed_peak_weight_bytes(DType::BF16) * 2
        );
    }

    #[test]
    fn attention_mask_is_causal_and_excludes_padding_keys() {
        let mut attention = vec![false; MAX_LENGTH];
        attention[..3].fill(true);
        let mask = causal_padding_mask(&attention, DType::F32, &Device::Cpu).unwrap();

        assert_eq!(
            mask.i((0, 0, 2, 0)).unwrap().to_scalar::<f32>().unwrap(),
            0.0
        );
        assert_eq!(
            mask.i((0, 0, 2, 2)).unwrap().to_scalar::<f32>().unwrap(),
            0.0
        );
        assert!(mask
            .i((0, 0, 1, 2))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .is_infinite());
        assert!(mask
            .i((0, 0, 511, 3))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .is_infinite());
    }
}
