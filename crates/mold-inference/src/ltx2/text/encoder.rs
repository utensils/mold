#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Embedding, Linear, Module, VarBuilder};
pub use candle_transformers::models::gemma3::Config as GemmaConfig;

use super::gemma::{GemmaAssets, PromptTokens};

const MASK_NEGATIVE: f32 = -1e30;

pub fn ltx_gemma_config() -> GemmaConfig {
    GemmaConfig {
        attention_bias: false,
        head_dim: 256,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size: 3840,
        intermediate_size: 15_360,
        num_attention_heads: 16,
        num_hidden_layers: 48,
        num_key_value_heads: 8,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_local_base_freq: 10_000.0,
        vocab_size: 262_208,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: 256,
        sliding_window: 1024,
        sliding_window_pattern: 6,
        max_position_embeddings: 131_072,
    }
}

pub fn discover_weight_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(root)
        .with_context(|| format!("failed to read Gemma asset root '{}'", root.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    (name == "model.safetensors"
                        || (name.starts_with("model-") && name.ends_with(".safetensors")))
                        && path.is_file()
                })
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        bail!(
            "Gemma asset root '{}' is missing model*.safetensors weights",
            root.display()
        );
    }
    Ok(files)
}

fn map_gemma_weight_key(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("model.") {
        format!("language_model.model.{rest}")
    } else {
        name.to_string()
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let input_dtype = xs.dtype();
        let internal_dtype = match input_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            dtype => dtype,
        };
        let hidden = xs.dim(D::Minus1)?;
        let xs = xs.to_dtype(internal_dtype)?;
        let variance = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed
            .to_dtype(input_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        cfg: &GemmaConfig,
        device: &Device,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let rope_freq = if sliding_window.is_some() {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|index| (1f64 / rope_freq.powf(index as f64 / dim as f64)) as f32)
            .collect::<Vec<_>>();
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &GemmaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?,
            down_proj: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        Ok((lhs * rhs)?.apply(&self.down_proj)?)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: RotaryEmbedding,
}

impl Attention {
    fn new(cfg: &GemmaConfig, sliding_window: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        Ok(Self {
            q_proj: linear(
                cfg.hidden_size,
                num_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear(
                cfg.hidden_size,
                num_kv_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear(
                cfg.hidden_size,
                num_kv_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear(
                num_heads * cfg.head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim: cfg.head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb: RotaryEmbedding::new(vb.dtype(), cfg, vb.device(), sliding_window)?,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
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
        let (q, k) = self.rotary_emb.apply(&q, &k)?;
        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = match self.attn_logit_softcapping {
            Some(scale) => ((scores / scale)?.tanh()? * scale)?,
            None => scores,
        };
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

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}

impl DecoderLayer {
    fn new(cfg: &GemmaConfig, vb: VarBuilder, sliding_window: Option<usize>) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, sliding_window, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            pre_feedforward_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            sliding_window,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        Ok((residual + xs)?)
    }
}

#[derive(Debug, Clone)]
pub struct GemmaHiddenStates {
    pub hidden_states: Vec<Tensor>,
    pub attention_mask: Tensor,
}

#[derive(Clone)]
pub struct GemmaHiddenStateEncoder {
    embed_tokens: Embedding,
    layer_source: GemmaLayerSource,
    hidden_size: usize,
    device: Device,
    dtype: DType,
    sliding_window: usize,
}

#[derive(Clone)]
enum GemmaLayerSource {
    Eager(Vec<DecoderLayer>),
    Streaming {
        cfg: GemmaConfig,
        layers_vb: VarBuilder<'static>,
    },
}

impl GemmaHiddenStateEncoder {
    pub fn new(cfg: &GemmaConfig, vb: VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, model_vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for index in 0..cfg.num_hidden_layers {
            let uses_sliding = (index + 1) % cfg.sliding_window_pattern > 0;
            layers.push(DecoderLayer::new(
                cfg,
                layers_vb.pp(index),
                uses_sliding.then_some(cfg.sliding_window),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layer_source: GemmaLayerSource::Eager(layers),
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn new_streaming(cfg: &GemmaConfig, vb: VarBuilder<'static>) -> Result<Self> {
        let model_vb = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, model_vb.pp("embed_tokens"))?;
        Ok(Self {
            embed_tokens,
            layer_source: GemmaLayerSource::Streaming {
                cfg: cfg.clone(),
                layers_vb: model_vb.pp("layers"),
            },
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            sliding_window: cfg.sliding_window,
        })
    }

    pub fn load_from_assets(assets: &GemmaAssets, device: &Device, dtype: DType) -> Result<Self> {
        let weights = discover_weight_files(&assets.root)?;
        let vb: VarBuilder<'static> =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? };
        let vb = vb.rename_f(map_gemma_weight_key);
        Self::new_streaming(&ltx_gemma_config(), vb)
    }

    pub fn load_from_root(root: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let assets = GemmaAssets::discover(root)?;
        Self::load_from_assets(&assets, device, dtype)
    }

    pub fn encode_prompt_tokens(&mut self, tokens: &PromptTokens) -> Result<GemmaHiddenStates> {
        let input_ids = Tensor::new(tokens.input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.attention_mask.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden_states = self.forward_hidden_states(&input_ids, &attention_mask)?;
        Ok(GemmaHiddenStates {
            hidden_states,
            attention_mask,
        })
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let (batch, seq) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = (xs * (self.hidden_size as f64).sqrt())?;
        let mut hidden_states = Vec::with_capacity(self.layer_count() + 1);
        hidden_states.push(xs.clone());

        let full_mask = build_attention_mask(attention_mask, None, self.dtype, self.device())?;
        let sliding_mask = build_attention_mask(
            attention_mask,
            Some(self.sliding_window),
            self.dtype,
            self.device(),
        )?;

        match &self.layer_source {
            GemmaLayerSource::Eager(layers) => {
                for (index, layer) in layers.iter().enumerate() {
                    let mask = if layer.sliding_window.is_some() {
                        Some(&sliding_mask)
                    } else {
                        Some(&full_mask)
                    };
                    xs = layer
                        .forward(&xs, mask)
                        .with_context(|| format!("Gemma eager decoder layer {index} failed"))?;
                    hidden_states.push(xs.clone());
                }
            }
            GemmaLayerSource::Streaming { cfg, layers_vb } => {
                for index in 0..cfg.num_hidden_layers {
                    let layer = Self::streaming_layer(cfg, layers_vb.clone(), index)?;
                    let mask = if layer.sliding_window.is_some() {
                        Some(&sliding_mask)
                    } else {
                        Some(&full_mask)
                    };
                    xs = layer
                        .forward(&xs, mask)
                        .with_context(|| format!("Gemma streaming decoder layer {index} failed"))?;
                    hidden_states.push(xs.clone());
                }
            }
        }

        if hidden_states
            .iter()
            .any(|state| state.dims3().ok() != Some((batch, seq, self.hidden_size)))
        {
            bail!("Gemma hidden-state encoder produced inconsistent hidden-state shapes");
        }
        Ok(hidden_states)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn layer_count(&self) -> usize {
        match &self.layer_source {
            GemmaLayerSource::Eager(layers) => layers.len(),
            GemmaLayerSource::Streaming { cfg, .. } => cfg.num_hidden_layers,
        }
    }

    fn streaming_layer(
        cfg: &GemmaConfig,
        layers_vb: VarBuilder<'static>,
        index: usize,
    ) -> Result<DecoderLayer> {
        let uses_sliding = (index + 1) % cfg.sliding_window_pattern > 0;
        DecoderLayer::new(
            cfg,
            layers_vb.pp(index),
            uses_sliding.then_some(cfg.sliding_window),
        )
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Activation, VarBuilder};

    use super::{build_attention_mask, GemmaConfig, GemmaHiddenStateEncoder};

    fn tiny_config() -> GemmaConfig {
        GemmaConfig {
            attention_bias: false,
            head_dim: 4,
            hidden_activation: Activation::GeluPytorchTanh,
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            vocab_size: 16,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: 4,
            sliding_window: 4,
            sliding_window_pattern: 2,
            max_position_embeddings: 32,
        }
    }

    fn zero_gemma_var_builder(cfg: &GemmaConfig) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap(),
        );
        for layer in 0..cfg.num_hidden_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let (rows, cols) = match name {
                    "self_attn.q_proj" => (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                    "self_attn.k_proj" | "self_attn.v_proj" => {
                        (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
                    }
                    "self_attn.o_proj" => (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
                    "mlp.gate_proj" | "mlp.up_proj" => (cfg.intermediate_size, cfg.hidden_size),
                    "mlp.down_proj" => (cfg.hidden_size, cfg.intermediate_size),
                    _ => unreachable!(),
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for name in [
                "self_attn.q_norm",
                "self_attn.k_norm",
                "input_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_attention_layernorm",
            ] {
                let dim = if name.contains("q_norm") || name.contains("k_norm") {
                    cfg.head_dim
                } else {
                    cfg.hidden_size
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    #[test]
    fn hidden_state_encoder_emits_embedding_plus_each_layer() {
        let cfg = tiny_config();
        let mut encoder = GemmaHiddenStateEncoder::new(&cfg, zero_gemma_var_builder(&cfg)).unwrap();
        let input_ids = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap();
        let attention_mask = Tensor::new(&[[0u8, 0, 1, 1]], &Device::Cpu).unwrap();

        let hidden_states = encoder
            .forward_hidden_states(&input_ids, &attention_mask)
            .unwrap();

        assert_eq!(hidden_states.len(), cfg.num_hidden_layers + 1);
        for state in &hidden_states {
            assert_eq!(state.dims3().unwrap(), (1, 4, cfg.hidden_size));
        }
    }

    #[test]
    fn streaming_hidden_state_encoder_emits_embedding_plus_each_layer() {
        let cfg = tiny_config();
        let vb: VarBuilder<'static> = zero_gemma_var_builder(&cfg);
        let mut encoder = GemmaHiddenStateEncoder::new_streaming(&cfg, vb).unwrap();
        let input_ids = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu).unwrap();
        let attention_mask = Tensor::new(&[[0u8, 0, 1, 1]], &Device::Cpu).unwrap();

        let hidden_states = encoder
            .forward_hidden_states(&input_ids, &attention_mask)
            .unwrap();

        assert_eq!(hidden_states.len(), cfg.num_hidden_layers + 1);
        for state in &hidden_states {
            assert_eq!(state.dims3().unwrap(), (1, 4, cfg.hidden_size));
        }
    }

    #[test]
    fn additive_attention_mask_blocks_padding_and_future_tokens() {
        let mask = Tensor::new(&[[0u8, 1, 1]], &Device::Cpu).unwrap();
        let additive = build_attention_mask(&mask, None, DType::F32, &Device::Cpu).unwrap();
        let values = additive.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values[0] < -1e20);
        assert_eq!(values[4], 0.0);
        assert!(values[2] < -1e20);
        assert_eq!(values[8], 0.0);
    }
}
