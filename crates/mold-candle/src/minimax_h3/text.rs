use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, linear_b, rms_norm, Activation, Embedding, Linear, Module, RmsNorm, VarBuilder,
};

use super::comfy_quant::{H3ComfyInt8TensorwiseEmbedding, H3ComfyNvfp4AwqLinear};
use super::config::{H3ConditionerConfig, H3_SELECTED_LANGUAGE_LAYERS};
use super::model::ConditionerCheckpoint;
use super::H3_COMFY_PORTABLE_ROW_CHUNK;

#[derive(Clone, Debug)]
pub(super) struct Qwen3VlTextDimensions {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    max_position_embeddings: usize,
    mrope_section: [usize; 3],
}

impl Qwen3VlTextDimensions {
    pub(super) fn from_h3(config: &H3ConditionerConfig) -> Self {
        let text = &config.text_config;
        Self {
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            num_layers: H3_SELECTED_LANGUAGE_LAYERS,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
            max_position_embeddings: text.max_position_embeddings,
            mrope_section: text.rope_scaling.mrope_section,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.num_layers == 0
            || self.hidden_size == 0
            || self.head_dim == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.mrope_section.iter().sum::<usize>() != self.head_dim / 2
        {
            candle::bail!("invalid H3 Qwen3-VL text dimensions: {self:?}");
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn tiny() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 12,
            intermediate_size: 24,
            num_layers: 3,
            num_attention_heads: 3,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            mrope_section: [1, 1, 0],
        }
    }
}

enum Qwen3VlEmbedding {
    Dense(Embedding),
    Int8Tensorwise {
        embedding: H3ComfyInt8TensorwiseEmbedding,
        output_dtype: DType,
        device: Device,
    },
}

impl Qwen3VlEmbedding {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.forward(input_ids),
            Self::Int8Tensorwise {
                embedding,
                output_dtype,
                device,
            } => embedding.forward(input_ids, *output_dtype, device),
        }
    }
}

enum Qwen3VlLinear {
    Dense(Linear),
    Nvfp4(H3ComfyNvfp4AwqLinear),
}

impl Qwen3VlLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            Self::Nvfp4(linear) => {
                linear.forward_dequantized(input, None, input.dtype(), H3_COMFY_PORTABLE_ROW_CHUNK)
            }
        }
    }
}

pub(super) struct Qwen3VlNvfp4LayerWeights {
    pub(super) q_proj: H3ComfyNvfp4AwqLinear,
    pub(super) k_proj: H3ComfyNvfp4AwqLinear,
    pub(super) v_proj: H3ComfyNvfp4AwqLinear,
    pub(super) o_proj: H3ComfyNvfp4AwqLinear,
    pub(super) gate_proj: H3ComfyNvfp4AwqLinear,
    pub(super) up_proj: H3ComfyNvfp4AwqLinear,
    pub(super) down_proj: H3ComfyNvfp4AwqLinear,
}

pub(super) struct Qwen3VlNvfp4Weights {
    pub(super) embed_tokens: H3ComfyInt8TensorwiseEmbedding,
    pub(super) layers: Vec<Qwen3VlNvfp4LayerWeights>,
}

struct Mlp {
    gate_proj: Qwen3VlLinear,
    up_proj: Qwen3VlLinear,
    down_proj: Qwen3VlLinear,
}

impl Mlp {
    fn new(config: &Qwen3VlTextDimensions, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: Qwen3VlLinear::Dense(linear_b(
                config.hidden_size,
                config.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?),
            up_proj: Qwen3VlLinear::Dense(linear_b(
                config.hidden_size,
                config.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?),
            down_proj: Qwen3VlLinear::Dense(linear_b(
                config.intermediate_size,
                config.hidden_size,
                false,
                vb.pp("down_proj"),
            )?),
        })
    }

    fn new_nvfp4(
        gate_proj: H3ComfyNvfp4AwqLinear,
        up_proj: H3ComfyNvfp4AwqLinear,
        down_proj: H3ComfyNvfp4AwqLinear,
    ) -> Self {
        Self {
            gate_proj: Qwen3VlLinear::Nvfp4(gate_proj),
            up_proj: Qwen3VlLinear::Nvfp4(up_proj),
            down_proj: Qwen3VlLinear::Nvfp4(down_proj),
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(input)?.apply(&Activation::Silu)?;
        self.down_proj
            .forward(&(gate * self.up_proj.forward(input)?)?)
    }
}

struct MultimodalRotaryEmbedding {
    inv_freq: Tensor,
    sections: [usize; 3],
    maximum_positions: usize,
}

impl MultimodalRotaryEmbedding {
    fn new(config: &Qwen3VlTextDimensions, vb: &VarBuilder) -> Result<Self> {
        let inv_freq = (0..config.head_dim)
            .step_by(2)
            .map(|index| {
                1.0_f32 / (config.rope_theta as f32).powf(index as f32 / config.head_dim as f32)
            })
            .collect::<Vec<_>>();
        let half = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (half,), vb.device())?,
            sections: config.mrope_section,
            maximum_positions: config.max_position_embeddings,
        })
    }

    fn cos_sin(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (axes, _batch, sequence) = position_ids.dims3()?;
        if axes != 3 {
            candle::bail!("Qwen multimodal RoPE needs three position axes, got {axes}");
        }
        let max_position = position_ids
            .to_device(&candle::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .max()
            .unwrap_or(0) as usize;
        if max_position >= self.maximum_positions {
            candle::bail!(
                "Qwen multimodal RoPE position {} exceeds configured maximum {}",
                max_position,
                self.maximum_positions
            );
        }
        let positions = position_ids.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
        let frequencies = positions.broadcast_mul(&self.inv_freq.reshape((1, 1, 1, ()))?)?;
        let axes = [frequencies.i(0)?, frequencies.i(1)?, frequencies.i(2)?];
        let half = self.inv_freq.dim(0)?;
        let mut columns = Vec::with_capacity(half);
        for index in 0..half {
            let axis = if index % 3 == 1 && index < self.sections[1] * 3 {
                1
            } else if index % 3 == 2 && index < self.sections[2] * 3 {
                2
            } else {
                0
            };
            columns.push(axes[axis].narrow(D::Minus1, index, 1)?);
        }
        let refs = columns.iter().collect::<Vec<_>>();
        let interleaved = Tensor::cat(&refs, D::Minus1)?;
        debug_assert_eq!(interleaved.dim(1)?, sequence);
        let doubled = Tensor::cat(&[&interleaved, &interleaved], D::Minus1)?;
        Ok((doubled.cos()?, doubled.sin()?))
    }
}

fn rotate_half(input: &Tensor) -> Result<Tensor> {
    let hidden = input.dim(D::Minus1)?;
    let first = input.narrow(D::Minus1, 0, hidden / 2)?;
    let second = input.narrow(D::Minus1, hidden / 2, hidden / 2)?;
    Tensor::cat(&[&second.neg()?, &first], D::Minus1)
}

fn apply_rotary(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let input = input.to_dtype(DType::F32)?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    (input.broadcast_mul(&cos)? + rotate_half(&input)?.broadcast_mul(&sin)?)?.to_dtype(dtype)
}

struct Attention {
    q_proj: Qwen3VlLinear,
    k_proj: Qwen3VlLinear,
    v_proj: Qwen3VlLinear,
    o_proj: Qwen3VlLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    kv_groups: usize,
    scale: f64,
}

impl Attention {
    fn new(config: &Qwen3VlTextDimensions, vb: VarBuilder) -> Result<Self> {
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        Ok(Self {
            q_proj: Qwen3VlLinear::Dense(linear_b(
                config.hidden_size,
                q_width,
                false,
                vb.pp("q_proj"),
            )?),
            k_proj: Qwen3VlLinear::Dense(linear_b(
                config.hidden_size,
                kv_width,
                false,
                vb.pp("k_proj"),
            )?),
            v_proj: Qwen3VlLinear::Dense(linear_b(
                config.hidden_size,
                kv_width,
                false,
                vb.pp("v_proj"),
            )?),
            o_proj: Qwen3VlLinear::Dense(linear_b(
                q_width,
                config.hidden_size,
                false,
                vb.pp("o_proj"),
            )?),
            q_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kv_groups: config.num_attention_heads / config.num_key_value_heads,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    fn new_nvfp4(
        config: &Qwen3VlTextDimensions,
        vb: VarBuilder,
        q_proj: H3ComfyNvfp4AwqLinear,
        k_proj: H3ComfyNvfp4AwqLinear,
        v_proj: H3ComfyNvfp4AwqLinear,
        o_proj: H3ComfyNvfp4AwqLinear,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Qwen3VlLinear::Nvfp4(q_proj),
            k_proj: Qwen3VlLinear::Nvfp4(k_proj),
            v_proj: Qwen3VlLinear::Nvfp4(v_proj),
            o_proj: Qwen3VlLinear::Nvfp4(o_proj),
            q_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kv_groups: config.num_attention_heads / config.num_key_value_heads,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        causal_mask: &Tensor,
    ) -> Result<Tensor> {
        let dtype = input.dtype();
        let (batch, sequence, _) = input.dims3()?;
        let q = self
            .q_proj
            .forward(input)?
            .reshape((batch, sequence, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .apply(&self.q_norm)?;
        let k = self
            .k_proj
            .forward(input)?
            .reshape((batch, sequence, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .apply(&self.k_norm)?;
        let v = self
            .v_proj
            .forward(input)?
            .reshape((batch, sequence, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = apply_rotary(&q, cos, sin)?.contiguous()?;
        let k = apply_rotary(&k, cos, sin)?.contiguous()?;
        let k = candle_transformers::utils::repeat_kv(k, self.kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.kv_groups)?.contiguous()?;
        let scores = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?.to_dtype(DType::F32)?;
        let scores = scores.broadcast_add(causal_mask)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        let output = probabilities
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, sequence, self.num_heads * self.head_dim))?
            .to_dtype(dtype)?;
        self.o_proj.forward(&output)
    }
}

struct DecoderLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(config: &Qwen3VlTextDimensions, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Attention::new(config, vb.pp("self_attn"))?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn new_nvfp4(
        config: &Qwen3VlTextDimensions,
        vb: VarBuilder,
        weights: Qwen3VlNvfp4LayerWeights,
    ) -> Result<Self> {
        let Qwen3VlNvfp4LayerWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
        } = weights;
        Ok(Self {
            attention: Attention::new_nvfp4(
                config,
                vb.pp("self_attn"),
                q_proj,
                k_proj,
                v_proj,
                o_proj,
            )?,
            mlp: Mlp::new_nvfp4(gate_proj, up_proj, down_proj),
            input_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        causal_mask: &Tensor,
    ) -> Result<Tensor> {
        let attention =
            self.attention
                .forward(&input.apply(&self.input_layernorm)?, cos, sin, causal_mask)?;
        let hidden = (input + attention)?;
        let mlp = self
            .mlp
            .forward(&hidden.apply(&self.post_attention_layernorm)?)?;
        hidden + mlp
    }
}

pub(super) struct Qwen3VlTextEncoder {
    embed_tokens: Qwen3VlEmbedding,
    layers: Vec<DecoderLayer>,
    rotary: MultimodalRotaryEmbedding,
    hidden_size: usize,
}

impl Qwen3VlTextEncoder {
    pub(super) fn new(config: &Qwen3VlTextDimensions, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = MultimodalRotaryEmbedding::new(config, &vb)?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            layers.push(DecoderLayer::new(config, vb.pp("layers").pp(index))?);
        }
        Ok(Self {
            embed_tokens: Qwen3VlEmbedding::Dense(embed_tokens),
            layers,
            rotary,
            hidden_size: config.hidden_size,
        })
    }

    pub(super) fn new_nvfp4(
        config: &Qwen3VlTextDimensions,
        vb: VarBuilder,
        weights: Qwen3VlNvfp4Weights,
    ) -> Result<Self> {
        config.validate()?;
        if weights.layers.len() != config.num_layers {
            candle::bail!(
                "H3 NVFP4 Qwen supplied {} layers, expected exactly {}",
                weights.layers.len(),
                config.num_layers
            );
        }
        if weights.embed_tokens.vocabulary() != config.vocab_size
            || weights.embed_tokens.hidden_size() != config.hidden_size
        {
            candle::bail!(
                "H3 NVFP4 Qwen embedding is {}x{}, expected {}x{}",
                weights.embed_tokens.vocabulary(),
                weights.embed_tokens.hidden_size(),
                config.vocab_size,
                config.hidden_size
            );
        }
        let rotary = MultimodalRotaryEmbedding::new(config, &vb)?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for (index, weights) in weights.layers.into_iter().enumerate() {
            layers.push(DecoderLayer::new_nvfp4(
                config,
                vb.pp("layers").pp(index),
                weights,
            )?);
        }
        Ok(Self {
            embed_tokens: Qwen3VlEmbedding::Int8Tensorwise {
                embedding: weights.embed_tokens,
                output_dtype: vb.dtype(),
                device: vb.device().clone(),
            },
            layers,
            rotary,
            hidden_size: config.hidden_size,
        })
    }

    pub(super) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(super) fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub(super) fn forward_embeds(
        &self,
        mut hidden: Tensor,
        position_ids: &Tensor,
        visual_indices: &[usize],
        deepstack: Option<&[Tensor]>,
        checkpoint: &mut dyn FnMut(ConditionerCheckpoint) -> Result<()>,
    ) -> Result<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        if width != self.hidden_size || batch != 1 {
            candle::bail!(
                "H3 text embeddings have shape {:?}, expected (1, sequence, {})",
                hidden.dims(),
                self.hidden_size
            );
        }
        if deepstack.is_some_and(|features| features.len() > self.layers.len()) {
            candle::bail!("H3 DeepStack has more feature layers than language layers");
        }
        let (cos, sin) = self.rotary.cos_sin(position_ids)?;
        let causal = causal_mask(sequence, hidden.device())?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &causal)?;
            if let Some(features) = deepstack.and_then(|features| features.get(index)) {
                hidden = inject_deepstack(hidden, visual_indices, features, self.hidden_size)?;
            }
            checkpoint(ConditionerCheckpoint::LanguageLayer {
                completed: index + 1,
                total: self.layers.len(),
            })?;
        }
        // Intentionally no final RMS norm: H3 consumes hidden_states[50], the
        // unnormalized state immediately after decoder layer 49.
        Ok(hidden)
    }
}

fn causal_mask(sequence: usize, device: &candle::Device) -> Result<Tensor> {
    let values = (0..sequence)
        .flat_map(|row| {
            (0..sequence).map(
                move |column| {
                    if column > row {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                },
            )
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, sequence, sequence), device)
}

fn inject_deepstack(
    mut hidden: Tensor,
    visual_indices: &[usize],
    features: &Tensor,
    width: usize,
) -> Result<Tensor> {
    if features.dims() != [visual_indices.len(), width] {
        candle::bail!(
            "H3 DeepStack features have shape {:?}, expected ({}, {})",
            features.dims(),
            visual_indices.len(),
            width
        );
    }
    let features = features
        .to_device(hidden.device())?
        .to_dtype(hidden.dtype())?;
    for (row, &sequence_index) in visual_indices.iter().enumerate() {
        let current = hidden.narrow(1, sequence_index, 1)?;
        let addition = features.narrow(0, row, 1)?.unsqueeze(0)?;
        hidden = hidden.slice_assign(
            &[0..1, sequence_index..sequence_index + 1, 0..width],
            &(current + addition)?,
        )?;
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarMap;
    use std::collections::HashMap;

    fn round_up(value: usize, multiple: usize) -> usize {
        value.div_ceil(multiple) * multiple
    }

    fn synthetic_nvfp4(
        out_features: usize,
        in_features: usize,
        with_awq: bool,
    ) -> H3ComfyNvfp4AwqLinear {
        let padded_out = round_up(out_features, 16);
        let padded_in = round_up(in_features, 16);
        let scale_rows = round_up(padded_out, 128);
        let scale_columns = round_up(padded_in / 16, 4);
        // 0x11 is two +0.5 E2M1 values. With unit block scales and a
        // 1/16 tensor scale this represents a dense 1/32 matrix exactly.
        let packed = Tensor::from_vec(
            vec![0x11_u8; padded_out * padded_in / 2],
            (padded_out, padded_in / 2),
            &Device::Cpu,
        )
        .unwrap();
        let block_scales = Tensor::ones((scale_rows, scale_columns), DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let tensor_scale = Tensor::new(0.0625_f32, &Device::Cpu).unwrap();
        let awq = with_awq.then(|| Tensor::ones((in_features,), DType::F32, &Device::Cpu).unwrap());
        H3ComfyNvfp4AwqLinear::new_with_optional_awq(
            packed,
            block_scales,
            tensor_scale,
            awq,
            out_features,
            in_features,
        )
        .unwrap()
    }

    fn synthetic_nvfp4_weights(config: &Qwen3VlTextDimensions) -> Qwen3VlNvfp4Weights {
        let mut raw_embedding = Vec::with_capacity(config.vocab_size * config.hidden_size);
        for row in 0..config.vocab_size {
            for column in 0..config.hidden_size {
                let value = ((row * config.hidden_size + column) % 7) as i8 - 3;
                raw_embedding.push(value as u8);
            }
        }
        let embed_tokens = H3ComfyInt8TensorwiseEmbedding::new(
            Tensor::from_vec(
                raw_embedding,
                (config.vocab_size, config.hidden_size),
                &Device::Cpu,
            )
            .unwrap(),
            Tensor::ones((config.vocab_size, 1), DType::F32, &Device::Cpu).unwrap(),
        )
        .unwrap();
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let layers = (0..config.num_layers)
            .map(|_| Qwen3VlNvfp4LayerWeights {
                q_proj: synthetic_nvfp4(q_width, config.hidden_size, false),
                k_proj: synthetic_nvfp4(kv_width, config.hidden_size, false),
                v_proj: synthetic_nvfp4(kv_width, config.hidden_size, false),
                o_proj: synthetic_nvfp4(config.hidden_size, q_width, true),
                gate_proj: synthetic_nvfp4(config.intermediate_size, config.hidden_size, false),
                up_proj: synthetic_nvfp4(config.intermediate_size, config.hidden_size, false),
                down_proj: synthetic_nvfp4(config.hidden_size, config.intermediate_size, true),
            })
            .collect();
        Qwen3VlNvfp4Weights {
            embed_tokens,
            layers,
        }
    }

    fn synthetic_dense_weights(config: &Qwen3VlTextDimensions) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        let embedding = (0..config.vocab_size * config.hidden_size)
            .map(|index| (index % 7) as f32 - 3.0)
            .collect::<Vec<_>>();
        tensors.insert(
            "embed_tokens.weight".into(),
            Tensor::from_vec(
                embedding,
                (config.vocab_size, config.hidden_size),
                &Device::Cpu,
            )
            .unwrap(),
        );
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        for layer in 0..config.num_layers {
            let prefix = format!("layers.{layer}");
            for (suffix, out_features, in_features) in [
                ("self_attn.q_proj.weight", q_width, config.hidden_size),
                ("self_attn.k_proj.weight", kv_width, config.hidden_size),
                ("self_attn.v_proj.weight", kv_width, config.hidden_size),
                ("self_attn.o_proj.weight", config.hidden_size, q_width),
                (
                    "mlp.gate_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                (
                    "mlp.up_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                (
                    "mlp.down_proj.weight",
                    config.hidden_size,
                    config.intermediate_size,
                ),
            ] {
                tensors.insert(
                    format!("{prefix}.{suffix}"),
                    Tensor::ones((out_features, in_features), DType::F32, &Device::Cpu)
                        .unwrap()
                        .affine(0.03125, 0.0)
                        .unwrap(),
                );
            }
            for (suffix, width) in [
                ("self_attn.q_norm.weight", config.head_dim),
                ("self_attn.k_norm.weight", config.head_dim),
                ("input_layernorm.weight", config.hidden_size),
                ("post_attention_layernorm.weight", config.hidden_size),
            ] {
                tensors.insert(
                    format!("{prefix}.{suffix}"),
                    Tensor::ones((width,), DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        tensors
    }

    #[test]
    fn tiny_encoder_returns_full_unnormalized_sequence_without_tail_keys() {
        let config = Qwen3VlTextDimensions::tiny();
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        let encoder = Qwen3VlTextEncoder::new(&config, vb).unwrap();
        let keys = vars
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(encoder.layer_count(), 3);
        assert!(!keys.iter().any(|key| key.ends_with(".norm.weight")));
        assert!(!keys.iter().any(|key| key.starts_with("lm_head")));

        let ids = Tensor::new(&[[1_u32, 2, 3]], &Device::Cpu).unwrap();
        let positions = Tensor::new(
            &[[[0_u32, 1, 2]], [[0_u32, 1, 2]], [[0_u32, 1, 2]]],
            &Device::Cpu,
        )
        .unwrap();
        let embeds = encoder.embed_tokens(&ids).unwrap();
        let mut checkpoints = Vec::new();
        let output = encoder
            .forward_embeds(embeds, &positions, &[], None, &mut |checkpoint| {
                checkpoints.push(checkpoint);
                Ok(())
            })
            .unwrap();
        assert_eq!(output.dims(), [1, 3, 12]);
        assert_eq!(
            checkpoints,
            vec![
                ConditionerCheckpoint::LanguageLayer {
                    completed: 1,
                    total: 3
                },
                ConditionerCheckpoint::LanguageLayer {
                    completed: 2,
                    total: 3
                },
                ConditionerCheckpoint::LanguageLayer {
                    completed: 3,
                    total: 3
                },
            ]
        );
    }

    #[test]
    fn multimodal_rope_interleaves_axes_in_fp32() {
        let config = Qwen3VlTextDimensions::tiny();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let rotary = MultimodalRotaryEmbedding::new(&config, &vb).unwrap();
        let positions = Tensor::new(&[[[2_u32]], [[3_u32]], [[4_u32]]], &Device::Cpu).unwrap();
        let (cos, sin) = rotary.cos_sin(&positions).unwrap();
        assert_eq!(cos.dtype(), DType::F32);
        assert_eq!(sin.dtype(), DType::F32);
        assert_eq!(cos.dims(), [1, 1, 4]);
        let values = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 2.0_f32.cos()).abs() < 1e-6);
        assert!((values[1] - (3.0_f32 / 100.0).cos()).abs() < 1e-6);
    }

    #[test]
    fn mixed_int8_nvfp4_stack_matches_dense_numerical_oracle() {
        let config = Qwen3VlTextDimensions::tiny();
        let dense_weights = synthetic_dense_weights(&config);
        let dense = Qwen3VlTextEncoder::new(
            &config,
            VarBuilder::from_tensors(dense_weights.clone(), DType::F32, &Device::Cpu),
        )
        .unwrap();
        let quantized = Qwen3VlTextEncoder::new_nvfp4(
            &config,
            VarBuilder::from_tensors(dense_weights, DType::F32, &Device::Cpu),
            synthetic_nvfp4_weights(&config),
        )
        .unwrap();
        let ids = Tensor::new(&[[1_u32, 7, 19]], &Device::Cpu).unwrap();
        let positions = Tensor::new(
            &[[[0_u32, 1, 2]], [[0_u32, 1, 2]], [[0_u32, 1, 2]]],
            &Device::Cpu,
        )
        .unwrap();

        let dense_embeds = dense.embed_tokens(&ids).unwrap();
        let quantized_embeds = quantized.embed_tokens(&ids).unwrap();
        let dense_output = dense
            .forward_embeds(dense_embeds.clone(), &positions, &[], None, &mut |_| Ok(()))
            .unwrap();
        let quantized_output = quantized
            .forward_embeds(quantized_embeds.clone(), &positions, &[], None, &mut |_| {
                Ok(())
            })
            .unwrap();

        for (role, dense, quantized) in [
            ("embedding", dense_embeds, quantized_embeds),
            (
                "unnormalized hidden-state contract",
                dense_output,
                quantized_output,
            ),
        ] {
            let dense = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let quantized = quantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let max_difference = dense
                .iter()
                .zip(&quantized)
                .map(|(left, right)| (left - right).abs())
                .fold(0_f32, f32::max);
            assert!(max_difference <= 2e-5, "{role} drifted by {max_difference}");
        }
    }

    #[test]
    fn mixed_stack_rejects_any_layer_tail_or_truncation() {
        let config = Qwen3VlTextDimensions::tiny();
        for supplied in [config.num_layers - 1, config.num_layers + 1] {
            let mut weights = synthetic_nvfp4_weights(&config);
            weights.layers.truncate(supplied.min(config.num_layers));
            if supplied > config.num_layers {
                weights.layers.push(Qwen3VlNvfp4LayerWeights {
                    q_proj: synthetic_nvfp4(12, 12, false),
                    k_proj: synthetic_nvfp4(4, 12, false),
                    v_proj: synthetic_nvfp4(4, 12, false),
                    o_proj: synthetic_nvfp4(12, 12, true),
                    gate_proj: synthetic_nvfp4(24, 12, false),
                    up_proj: synthetic_nvfp4(24, 12, false),
                    down_proj: synthetic_nvfp4(12, 24, true),
                });
            }
            let error = Qwen3VlTextEncoder::new_nvfp4(
                &config,
                VarBuilder::zeros(DType::F32, &Device::Cpu),
                weights,
            )
            .err()
            .expect("mismatched layer count must fail");
            assert!(error.to_string().contains("expected exactly 3"));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_bf16_text_stack_executes_on_cuda_with_fp32_rotary_and_softmax() {
        let device = Device::new_cuda(0).unwrap();
        let config = Qwen3VlTextDimensions::tiny();
        let encoder =
            Qwen3VlTextEncoder::new(&config, VarBuilder::zeros(DType::BF16, &device)).unwrap();
        let ids = Tensor::new(&[[1_u32, 2, 3]], &device).unwrap();
        let positions = Tensor::new(
            &[[[0_u32, 1, 2]], [[0_u32, 1, 2]], [[0_u32, 1, 2]]],
            &device,
        )
        .unwrap();
        let output = encoder
            .forward_embeds(
                encoder.embed_tokens(&ids).unwrap(),
                &positions,
                &[],
                None,
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(output.dims(), [1, 3, 12]);
        assert_eq!(output.dtype(), DType::BF16);
        assert!(output
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
}
