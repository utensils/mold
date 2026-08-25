//! Gemma 4 Unified text-only hidden-state encoder for LTX-2.5.
//!
//! The published text encoder is a packed safetensors file with flattened
//! `model.*` language-model keys. Vision and audio towers are intentionally not
//! materialized here: LTX prompt conditioning asks the unified model for text
//! hidden states only.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Embedding, Linear, Module, VarBuilder};

use super::encoder::{build_attention_mask, build_position_ids, GemmaHiddenStates};
use super::gemma::{GemmaAssets, PromptTokens};

const HIDDEN_SIZE: usize = 3_840;
const INTERMEDIATE_SIZE: usize = 15_360;
const NUM_LAYERS: usize = 48;
const NUM_HEADS: usize = 16;
const LOCAL_KV_HEADS: usize = 8;
const GLOBAL_KV_HEADS: usize = 1;
const LOCAL_HEAD_DIM: usize = 256;
const GLOBAL_HEAD_DIM: usize = 512;
const SLIDING_WINDOW: usize = 1_024;
const SLIDING_PATTERN: usize = 6;
const VOCAB_SIZE: usize = 262_144;

#[derive(Debug, Clone, Copy)]
struct Gemma4Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    local_kv_heads: usize,
    global_kv_heads: usize,
    local_head_dim: usize,
    global_head_dim: usize,
    sliding_window: usize,
    sliding_pattern: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    global_rope_theta: f64,
    local_rope_theta: f64,
    partial_rotary_factor: f64,
}

impl Gemma4Config {
    const fn ltx_12b() -> Self {
        Self {
            hidden_size: HIDDEN_SIZE,
            intermediate_size: INTERMEDIATE_SIZE,
            num_layers: NUM_LAYERS,
            num_heads: NUM_HEADS,
            local_kv_heads: LOCAL_KV_HEADS,
            global_kv_heads: GLOBAL_KV_HEADS,
            local_head_dim: LOCAL_HEAD_DIM,
            global_head_dim: GLOBAL_HEAD_DIM,
            sliding_window: SLIDING_WINDOW,
            sliding_pattern: SLIDING_PATTERN,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: 1e-6,
            global_rope_theta: 1_000_000.0,
            local_rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
        }
    }

    fn layer_is_sliding(self, index: usize) -> bool {
        !(index + 1).is_multiple_of(self.sliding_pattern)
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

fn unit_rms_norm(xs: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let input_dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let hidden = xs.dim(D::Minus1)? as f64;
    let variance = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    let denominator = (variance + eps)?.sqrt()?;
    xs.broadcast_div(&denominator)?.to_dtype(input_dtype)
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        unit_rms_norm(xs, self.eps)?.broadcast_mul(&(&self.weight + 1.0)?)
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
        device: &Device,
        head_dim: usize,
        theta: f64,
        rotary_dims: usize,
        max_length: usize,
    ) -> Result<Self> {
        let mut inv_freq = Vec::with_capacity(head_dim / 2);
        for index in (0..head_dim).step_by(2) {
            if index < rotary_dims {
                inv_freq.push((1.0 / theta.powf(index as f64 / head_dim as f64)) as f32);
            } else {
                inv_freq.push(0.0);
            }
        }
        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, max_length as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_length, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: frequencies.sin()?,
            cos: frequencies.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, _, seq, _) = q.dims4()?;
        let ids = position_ids.to_dtype(DType::U32)?.flatten_all()?;
        let cos = self
            .cos
            .index_select(&ids, 0)?
            .reshape((batch, seq, self.cos.dim(1)?))?;
        let sin = self
            .sin
            .index_select(&ids, 0)?
            .reshape((batch, seq, self.sin.dim(1)?))?;
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
}

impl Mlp {
    fn new(cfg: Gemma4Config, vb: VarBuilder) -> Result<Self> {
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
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let gate = self
            .gate_proj
            .forward(xs)?
            .apply(&Activation::GeluPytorchTanh)?;
        (gate * self.up_proj.forward(xs)?)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn new(cfg: Gemma4Config, sliding: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = if sliding {
            cfg.local_head_dim
        } else {
            cfg.global_head_dim
        };
        let num_kv_heads = if sliding {
            cfg.local_kv_heads
        } else {
            cfg.global_kv_heads
        };
        let rotary_dims = if sliding {
            head_dim
        } else {
            (cfg.partial_rotary_factor * head_dim as f64) as usize
        };
        let theta = if sliding {
            cfg.local_rope_theta
        } else {
            cfg.global_rope_theta
        };
        Ok(Self {
            q_proj: linear(
                cfg.hidden_size,
                cfg.num_heads * head_dim,
                false,
                vb.pp("q_proj"),
            )?,
            k_proj: linear(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                false,
                vb.pp("k_proj"),
            )?,
            v_proj: sliding
                .then(|| {
                    linear(
                        cfg.hidden_size,
                        num_kv_heads * head_dim,
                        false,
                        vb.pp("v_proj"),
                    )
                })
                .transpose()?,
            o_proj: linear(
                cfg.num_heads * head_dim,
                cfg.hidden_size,
                false,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            rotary: RotaryEmbedding::new(
                vb.dtype(),
                vb.device(),
                head_dim,
                theta,
                rotary_dims,
                SLIDING_WINDOW,
            )?,
            num_heads: cfg.num_heads,
            num_kv_heads,
            head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let raw_k =
            self.k_proj
                .forward(xs)?
                .reshape((batch, seq, self.num_kv_heads, self.head_dim))?;
        let v = match &self.v_proj {
            Some(v_proj) => {
                v_proj
                    .forward(xs)?
                    .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            }
            None => raw_k.clone(),
        };
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&raw_k)?.transpose(1, 2)?;
        let v = unit_rms_norm(&v, self.eps)?.transpose(1, 2)?;
        let (q, k) = self.rotary.apply(&q, &k, position_ids)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?;
        let scores = q.matmul(&k.transpose(2, 3)?)?.broadcast_add(mask)?;
        let probs =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    layer_scalar: Tensor,
    sliding: bool,
}

impl DecoderLayer {
    fn new(cfg: Gemma4Config, index: usize, vb: VarBuilder) -> Result<Self> {
        let sliding = cfg.layer_is_sliding(index);
        Ok(Self {
            attention: Attention::new(cfg, sliding, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
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
            layer_scalar: vb.get(1, "layer_scalar")?,
            sliding,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let attended =
            self.attention
                .forward(&self.input_layernorm.forward(xs)?, mask, position_ids)?;
        let xs = (residual + self.post_attention_layernorm.forward(&attended)?)?;
        let residual = &xs;
        let ff = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&xs)?)?;
        let xs = (residual + self.post_feedforward_layernorm.forward(&ff)?)?;
        xs.broadcast_mul(&self.layer_scalar).map_err(Into::into)
    }
}

pub struct Gemma4HiddenStateEncoder {
    cfg: Gemma4Config,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers_vb: VarBuilder<'static>,
    device: Device,
    dtype: DType,
}

impl Gemma4HiddenStateEncoder {
    pub fn load_from_assets(assets: &GemmaAssets, device: &Device, dtype: DType) -> Result<Self> {
        let path = assets.packed_weights.as_ref().ok_or_else(|| {
            anyhow::anyhow!("LTX-2.5 Gemma 4 requires a packed text-encoder safetensors file")
        })?;
        mold_core::ltx25_probe::probe_ltx25_gemma(path).with_context(|| {
            format!("incompatible LTX-2.5 Gemma 4 encoder '{}'", path.display())
        })?;
        let vb: VarBuilder<'static> = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(path), dtype, device)?
        };
        Self::new_streaming(Gemma4Config::ltx_12b(), vb)
    }

    fn new_streaming(cfg: Gemma4Config, vb: VarBuilder<'static>) -> Result<Self> {
        let model = vb.pp("model");
        Ok(Self {
            cfg,
            embed_tokens: candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                model.pp("embed_tokens"),
            )?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, model.pp("norm"))?,
            layers_vb: model.pp("layers"),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn encode_prompt_tokens(&mut self, tokens: &PromptTokens) -> Result<GemmaHiddenStates> {
        let ids = Tensor::new(tokens.input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.attention_mask.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden_states = self.forward_hidden_states(&ids, &attention_mask)?;
        Ok(GemmaHiddenStates {
            hidden_states,
            attention_mask,
        })
    }

    fn forward_hidden_states(&self, ids: &Tensor, attention_mask: &Tensor) -> Result<Vec<Tensor>> {
        let (batch, seq) = ids.dims2()?;
        if seq > self.cfg.sliding_window {
            bail!(
                "Gemma 4 prompt length {seq} exceeds supported maximum {}",
                self.cfg.sliding_window
            );
        }
        let mut xs = (self.embed_tokens.forward(ids)? * (self.cfg.hidden_size as f64).sqrt())?;
        let mut hidden_states = Vec::with_capacity(self.cfg.num_layers + 1);
        hidden_states.push(xs.clone());
        let positions = build_position_ids(attention_mask)?;
        let full_mask = build_attention_mask(attention_mask, None, self.dtype, &self.device)?;
        let sliding_mask = build_attention_mask(
            attention_mask,
            Some(self.cfg.sliding_window),
            self.dtype,
            &self.device,
        )?;
        for index in 0..self.cfg.num_layers {
            let layer = DecoderLayer::new(self.cfg, index, self.layers_vb.pp(index))?;
            let mask = if layer.sliding {
                &sliding_mask
            } else {
                &full_mask
            };
            xs = layer
                .forward(&xs, mask, &positions)
                .with_context(|| format!("Gemma 4 decoder layer {index} failed"))?;
            if index + 1 < self.cfg.num_layers {
                hidden_states.push(xs.clone());
            }
        }
        hidden_states.push(self.norm.forward(&xs)?);
        if hidden_states
            .iter()
            .any(|state| state.dims3().ok() != Some((batch, seq, self.cfg.hidden_size)))
        {
            bail!("Gemma 4 produced inconsistent hidden-state shapes");
        }
        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{Gemma4Config, Gemma4HiddenStateEncoder};

    #[test]
    fn ltx_12b_layer_pattern_and_dimensions_match_upstream() {
        let cfg = Gemma4Config::ltx_12b();
        assert_eq!((cfg.hidden_size, cfg.intermediate_size), (3_840, 15_360));
        assert_eq!((cfg.num_layers, cfg.num_heads), (48, 16));
        assert_eq!((cfg.local_head_dim, cfg.global_head_dim), (256, 512));
        assert!((0..5).all(|index| cfg.layer_is_sliding(index)));
        assert!(!cfg.layer_is_sliding(5));
        assert!(cfg.layer_is_sliding(6));
        assert!(!cfg.layer_is_sliding(47));
    }

    fn tiny_config() -> Gemma4Config {
        Gemma4Config {
            hidden_size: 4,
            intermediate_size: 8,
            num_layers: 2,
            num_heads: 2,
            local_kv_heads: 1,
            global_kv_heads: 1,
            local_head_dim: 2,
            global_head_dim: 2,
            sliding_window: 8,
            sliding_pattern: 2,
            vocab_size: 8,
            rms_norm_eps: 1e-6,
            global_rope_theta: 1_000_000.0,
            local_rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
        }
    }

    fn zero_linear(tensors: &mut HashMap<String, Tensor>, name: String, rows: usize, cols: usize) {
        tensors.insert(
            format!("{name}.weight"),
            Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap(),
        );
    }

    fn tiny_var_builder(cfg: Gemma4Config) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        let mut embeddings = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        embeddings[3 * cfg.hidden_size..4 * cfg.hidden_size].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        tensors.insert(
            "model.embed_tokens.weight".into(),
            Tensor::from_vec(embeddings, (cfg.vocab_size, cfg.hidden_size), &Device::Cpu).unwrap(),
        );
        for layer in 0..cfg.num_layers {
            let prefix = format!("model.layers.{layer}");
            let sliding = cfg.layer_is_sliding(layer);
            let head_dim = if sliding {
                cfg.local_head_dim
            } else {
                cfg.global_head_dim
            };
            let kv_heads = if sliding {
                cfg.local_kv_heads
            } else {
                cfg.global_kv_heads
            };
            zero_linear(
                &mut tensors,
                format!("{prefix}.self_attn.q_proj"),
                cfg.num_heads * head_dim,
                cfg.hidden_size,
            );
            zero_linear(
                &mut tensors,
                format!("{prefix}.self_attn.k_proj"),
                kv_heads * head_dim,
                cfg.hidden_size,
            );
            if sliding {
                zero_linear(
                    &mut tensors,
                    format!("{prefix}.self_attn.v_proj"),
                    kv_heads * head_dim,
                    cfg.hidden_size,
                );
            }
            zero_linear(
                &mut tensors,
                format!("{prefix}.self_attn.o_proj"),
                cfg.hidden_size,
                cfg.num_heads * head_dim,
            );
            zero_linear(
                &mut tensors,
                format!("{prefix}.mlp.gate_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
            );
            zero_linear(
                &mut tensors,
                format!("{prefix}.mlp.up_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
            );
            zero_linear(
                &mut tensors,
                format!("{prefix}.mlp.down_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
            );
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                tensors.insert(
                    format!("{prefix}.{norm}.weight"),
                    Tensor::zeros(cfg.hidden_size, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for norm in ["q_norm", "k_norm"] {
                tensors.insert(
                    format!("{prefix}.self_attn.{norm}.weight"),
                    Tensor::zeros(head_dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            tensors.insert(
                format!("{prefix}.layer_scalar"),
                Tensor::ones(1, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        tensors.insert(
            "model.norm.weight".into(),
            Tensor::zeros(cfg.hidden_size, DType::F32, &Device::Cpu).unwrap(),
        );
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    #[test]
    fn tiny_conditioning_shape_and_values_match_gemma4_residual_contract() {
        let cfg = tiny_config();
        let encoder = Gemma4HiddenStateEncoder::new_streaming(cfg, tiny_var_builder(cfg)).unwrap();
        let ids = Tensor::new(&[[3u32]], &Device::Cpu).unwrap();
        let mask = Tensor::new(&[[1u8]], &Device::Cpu).unwrap();
        let states = encoder.forward_hidden_states(&ids, &mask).unwrap();

        assert_eq!(states.len(), cfg.num_layers + 1);
        assert!(states.iter().all(|state| state.dims() == [1, 1, 4]));
        assert_eq!(
            states[0].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            states[1].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0, 4.0, 6.0, 8.0]
        );

        let final_values = states[2].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let denominator = (30.0f32 + cfg.rms_norm_eps as f32).sqrt();
        let expected = [2.0, 4.0, 6.0, 8.0].map(|value| value / denominator);
        for (actual, expected) in final_values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
}
