#![allow(dead_code)]

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear_b, Activation, Linear, Module, VarBuilder};

use crate::ltx2::model::{video_transformer::apply_rotary_emb, LtxRopeType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingSide {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct Projection {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Projection {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn out_features(&self) -> Result<usize> {
        self.weight
            .dims2()
            .map(|(rows, _)| rows)
            .map_err(Into::into)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        let (out, in_features) = self.weight.dims2()?;
        if hidden != in_features {
            bail!("projection input dimension mismatch: expected {in_features}, got {hidden}");
        }
        let compute_dtype =
            if xs.device().is_cpu() && matches!(self.weight.dtype(), DType::BF16 | DType::F16) {
                DType::F32
            } else {
                self.weight.dtype()
            };
        let xs = if xs.dtype() == compute_dtype {
            xs.clone()
        } else {
            xs.to_dtype(compute_dtype)?
        };
        let weight = if self.weight.dtype() == compute_dtype {
            self.weight.clone()
        } else {
            self.weight.to_dtype(compute_dtype)?
        };
        let ys = xs
            .reshape((batch * seq, hidden))?
            .matmul(&weight.transpose(0, 1)?)?;
        let ys = if let Some(bias) = &self.bias {
            let bias = if bias.dtype() == compute_dtype {
                bias.clone()
            } else {
                bias.to_dtype(compute_dtype)?
            };
            ys.broadcast_add(&bias)?
        } else {
            ys
        };
        Ok(ys.reshape((batch, seq, out))?)
    }
}

#[derive(Debug, Clone)]
pub struct FeatureExtractorV1 {
    aggregate_embed: Projection,
    is_av: bool,
}

impl FeatureExtractorV1 {
    pub fn new(aggregate_embed: Projection, is_av: bool) -> Self {
        Self {
            aggregate_embed,
            is_av,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &[Tensor],
        attention_mask: &Tensor,
        padding_side: PaddingSide,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let encoded = stack_hidden_states(hidden_states)?;
        let normed = norm_and_concat_padded_batch(&encoded, attention_mask, padding_side)?;
        let features = self.aggregate_embed.forward(&normed)?;
        let audio = if self.is_av {
            Some(features.clone())
        } else {
            None
        };
        Ok((features, audio))
    }
}

#[derive(Debug, Clone)]
pub struct FeatureExtractorV2 {
    video_aggregate_embed: Projection,
    audio_aggregate_embed: Option<Projection>,
    embedding_dim: usize,
}

impl FeatureExtractorV2 {
    pub fn new(
        video_aggregate_embed: Projection,
        audio_aggregate_embed: Option<Projection>,
        embedding_dim: usize,
    ) -> Self {
        Self {
            video_aggregate_embed,
            audio_aggregate_embed,
            embedding_dim,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &[Tensor],
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let encoded = stack_hidden_states(hidden_states)?;
        let normed = norm_and_concat_per_token_rms(&encoded, attention_mask)?;
        let video = self.video_aggregate_embed.forward(&rescale_norm(
            &normed,
            self.video_aggregate_embed.out_features()?,
            self.embedding_dim,
        )?)?;
        let audio = self
            .audio_aggregate_embed
            .as_ref()
            .map(|projection| {
                projection.forward(
                    &rescale_norm(
                        &normed,
                        projection.out_features().unwrap(),
                        self.embedding_dim,
                    )
                    .unwrap(),
                )
            })
            .transpose()?;
        Ok((video, audio))
    }
}

pub fn stack_hidden_states(hidden_states: &[Tensor]) -> Result<Tensor> {
    let refs = hidden_states.iter().collect::<Vec<_>>();
    Ok(Tensor::stack(&refs, D::Minus1)?)
}

pub fn norm_and_concat_per_token_rms(
    encoded_text: &Tensor,
    attention_mask: &Tensor,
) -> Result<Tensor> {
    let encoded = encoded_text.to_dtype(DType::F32)?;
    let variance = encoded.sqr()?.mean_keepdim(2)?;
    let normed = encoded.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
    let (batch, seq, hidden, layers) = normed.dims4()?;
    let normed = normed.reshape((batch, seq, hidden * layers))?;
    let mask = attention_mask
        .to_dtype(DType::F32)?
        .reshape((batch, seq, 1))?;
    normed.broadcast_mul(&mask).map_err(Into::into)
}

pub fn replace_padded_with_registers(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    registers: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = hidden_states.device().clone();
    let output_dtype =
        if device.is_cpu() && matches!(hidden_states.dtype(), DType::BF16 | DType::F16) {
            DType::F32
        } else {
            hidden_states.dtype()
        };
    let hidden_states = hidden_states
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let attention_mask = attention_mask.to_device(&Device::Cpu)?.to_vec2::<u8>()?;
    let registers = registers
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let batch = hidden_states.len();
    let seq = hidden_states.first().map(Vec::len).unwrap_or(0);
    let dim = registers.first().map(Vec::len).unwrap_or(0);
    if registers.is_empty() {
        bail!("register replacement requires at least one learnable register");
    }

    let mut packed = Vec::with_capacity(batch * seq * dim);
    for (batch_hidden, batch_mask) in hidden_states.iter().zip(attention_mask.iter()) {
        let mut valid = batch_hidden
            .iter()
            .zip(batch_mask.iter())
            .filter(|(_, mask)| **mask != 0)
            .map(|(token, _)| token.clone())
            .collect::<Vec<_>>();
        let pad = seq.saturating_sub(valid.len());
        for index in 0..pad {
            valid.push(registers[index % registers.len()].clone());
        }
        for token in valid {
            packed.extend(token);
        }
    }

    let binary_mask = vec![1u8; batch * seq];
    Ok((
        Tensor::from_vec(packed, (batch, seq, dim), &device)?.to_dtype(output_dtype)?,
        Tensor::from_vec(binary_mask, (batch, seq), &device)?,
    ))
}

#[derive(Debug, Clone)]
pub enum EmbeddingsFeatureExtractor {
    V1(FeatureExtractorV1),
    V2(FeatureExtractorV2),
}

impl EmbeddingsFeatureExtractor {
    pub fn forward(
        &self,
        hidden_states: &[Tensor],
        attention_mask: &Tensor,
        padding_side: PaddingSide,
    ) -> Result<(Tensor, Option<Tensor>)> {
        match self {
            Self::V1(extractor) => extractor.forward(hidden_states, attention_mask, padding_side),
            Self::V2(extractor) => extractor.forward(hidden_states, attention_mask),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingsProcessorOutput {
    pub video_encoding: Tensor,
    pub audio_encoding: Option<Tensor>,
    pub attention_mask: Tensor,
}

#[derive(Debug, Clone)]
pub struct EmbeddingsProcessor {
    feature_extractor: EmbeddingsFeatureExtractor,
    video_connector: Embeddings1DConnector,
    audio_connector: Option<Embeddings1DConnector>,
}

impl EmbeddingsProcessor {
    pub fn new(
        feature_extractor: EmbeddingsFeatureExtractor,
        video_connector: Embeddings1DConnector,
        audio_connector: Option<Embeddings1DConnector>,
    ) -> Self {
        Self {
            feature_extractor,
            video_connector,
            audio_connector,
        }
    }

    pub fn create_embeddings(
        &self,
        video_features: &Tensor,
        audio_features: Option<&Tensor>,
        additive_attention_mask: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>, Tensor)> {
        if self.audio_connector.is_some() && audio_features.is_none() {
            bail!("audio connector is configured but no audio features were provided");
        }
        if self.audio_connector.is_none() && audio_features.is_some() {
            bail!("audio features were provided but no audio connector is configured");
        }

        let (video_encoded, video_mask) = self
            .video_connector
            .forward(video_features, additive_attention_mask)?;
        let (video_encoded, binary_mask) = to_binary_mask(&video_encoded, &video_mask)?;

        let audio_encoded = match (&self.audio_connector, audio_features) {
            (Some(connector), Some(features)) => {
                Some(connector.forward(features, additive_attention_mask)?.0)
            }
            _ => None,
        };

        Ok((
            video_encoded,
            audio_encoded,
            binary_mask.squeeze(D::Minus1)?,
        ))
    }

    pub fn process_hidden_states(
        &self,
        hidden_states: &[Tensor],
        attention_mask: &Tensor,
        padding_side: PaddingSide,
    ) -> Result<EmbeddingsProcessorOutput> {
        let (video_features, audio_features) =
            self.feature_extractor
                .forward(hidden_states, attention_mask, padding_side)?;
        let additive_mask = convert_to_additive_mask(attention_mask, video_features.dtype())?;
        let (video_encoding, audio_encoding, binary_mask) =
            self.create_embeddings(&video_features, audio_features.as_ref(), &additive_mask)?;
        Ok(EmbeddingsProcessorOutput {
            video_encoding,
            audio_encoding,
            attention_mask: binary_mask,
        })
    }
}

pub fn convert_to_additive_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let (batch, seq) = attention_mask.dims2()?;
    let mask = attention_mask
        .to_dtype(DType::F32)?
        .reshape((batch, 1, 1, seq))?;
    let invalid = (mask.ones_like()? - &mask)?;
    invalid
        .affine(-1e30f64, 0.0)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

pub fn to_binary_mask(encoded: &Tensor, encoded_mask: &Tensor) -> Result<(Tensor, Tensor)> {
    let binary_mask =
        additive_mask_to_binary(encoded_mask)?.reshape((encoded.dim(0)?, encoded.dim(1)?, 1))?;
    Ok((
        encoded.broadcast_mul(&binary_mask.to_dtype(encoded.dtype())?)?,
        binary_mask,
    ))
}

fn additive_mask_to_binary(mask: &Tensor) -> Result<Tensor> {
    match mask.rank() {
        4 => {
            let (batch, _heads, _query, seq) = mask.dims4()?;
            let mask = mask.narrow(1, 0, 1)?.narrow(2, 0, 1)?;
            let values = mask.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let binary = values
                .into_iter()
                .map(|value| u8::from(value > -1.0))
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(binary, (batch, seq), mask.device())?)
        }
        2 => Ok(mask.clone()),
        rank => bail!("unsupported attention mask rank {rank}; expected [B, T] or [B, 1, 1, T]"),
    }
}

#[derive(Debug, Clone)]
struct ConnectorAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    dim_head: usize,
    positional_embedding_theta: f64,
    positional_embedding_max_pos: Vec<usize>,
    rope_type: LtxRopeType,
    double_precision_rope: bool,
}

impl ConnectorAttention {
    fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        positional_embedding_theta: f64,
        positional_embedding_max_pos: Vec<usize>,
        rope_type: LtxRopeType,
        double_precision_rope: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;
        Ok(Self {
            q_proj: linear_b(dim, inner_dim, true, vb.pp("to_q"))?,
            k_proj: linear_b(dim, inner_dim, true, vb.pp("to_k"))?,
            v_proj: linear_b(dim, inner_dim, true, vb.pp("to_v"))?,
            out_proj: linear_b(inner_dim, dim, true, vb.pp("to_out").pp("0"))?,
            heads,
            dim_head,
            positional_embedding_theta,
            positional_embedding_max_pos,
            rope_type,
            double_precision_rope,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        let inner_dim = self.heads * self.dim_head;
        let q = scale_free_rms_norm(&self.q_proj.forward(xs)?, 1e-6)?;
        let k = scale_free_rms_norm(&self.k_proj.forward(xs)?, 1e-6)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((batch, seq, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq, self.heads, self.dim_head))?
            .transpose(1, 2)?;

        let (cos, sin) = connector_rotary_emb_cache(
            xs.device(),
            xs.dtype(),
            seq,
            self.heads,
            self.dim_head,
            self.positional_embedding_theta,
            &self.positional_embedding_max_pos,
            self.rope_type,
            self.double_precision_rope,
        )?;
        let q = apply_rotary_emb(
            &q.transpose(1, 2)?.contiguous()?.reshape((batch, seq, inner_dim))?,
            &cos,
            &sin,
            self.rope_type,
            self.heads,
            self.dim_head,
        )?
        .reshape((batch, seq, self.heads, self.dim_head))?
        .transpose(1, 2)?
        .contiguous()?;
        let k = apply_rotary_emb(
            &k.transpose(1, 2)?.contiguous()?.reshape((batch, seq, inner_dim))?,
            &cos,
            &sin,
            self.rope_type,
            self.heads,
            self.dim_head,
        )?
        .reshape((batch, seq, self.heads, self.dim_head))?
        .transpose(1, 2)?
        .contiguous()?;
        let v = v.contiguous()?;

        let scale = 1f64 / (self.dim_head as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = match attention_mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs.contiguous()?.matmul(&v)?;
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, inner_dim))?;
        let output = self.out_proj.forward(&context)?;
        if output.dim(D::Minus1)? != dim {
            bail!("connector attention output dimension mismatch");
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct ConnectorFeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl ConnectorFeedForward {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = dim * 4;
        Ok(Self {
            proj_in: linear_b(dim, inner_dim, true, vb.pp("net").pp("0").pp("proj"))?,
            proj_out: linear_b(inner_dim, dim, true, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.proj_in.forward(xs)?;
        let hidden = Activation::GeluPytorchTanh.forward(&hidden)?;
        self.proj_out.forward(&hidden).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct BasicTransformerBlock1D {
    attn1: ConnectorAttention,
    ff: ConnectorFeedForward,
}

impl BasicTransformerBlock1D {
    fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        positional_embedding_theta: f64,
        positional_embedding_max_pos: Vec<usize>,
        rope_type: LtxRopeType,
        double_precision_rope: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            attn1: ConnectorAttention::new(
                dim,
                heads,
                dim_head,
                positional_embedding_theta,
                positional_embedding_max_pos,
                rope_type,
                double_precision_rope,
                vb.pp("attn1"),
            )?,
            ff: ConnectorFeedForward::new(dim, vb.pp("ff"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attn_input = scale_free_rms_norm(hidden_states, 1e-6)?;
        let hidden_states = (self.attn1.forward(&attn_input, attention_mask)? + hidden_states)?;
        let ff_input = scale_free_rms_norm(&hidden_states, 1e-6)?;
        Ok(self.ff.forward(&ff_input)?.broadcast_add(&hidden_states)?)
    }
}

#[derive(Debug, Clone)]
pub struct Embeddings1DConnector {
    transformer_1d_blocks: Vec<BasicTransformerBlock1D>,
    learnable_registers: Option<Tensor>,
    positional_embedding_theta: f64,
    positional_embedding_max_pos: Vec<usize>,
    rope_type: LtxRopeType,
    double_precision_rope: bool,
}

impl Embeddings1DConnector {
    pub fn new(
        num_attention_heads: usize,
        attention_head_dim: usize,
        num_layers: usize,
        positional_embedding_theta: f64,
        positional_embedding_max_pos: Vec<usize>,
        rope_type: LtxRopeType,
        double_precision_rope: bool,
        num_learnable_registers: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = num_attention_heads * attention_head_dim;
        let blocks = (0..num_layers)
            .map(|index| {
                BasicTransformerBlock1D::new(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    positional_embedding_theta,
                    positional_embedding_max_pos.clone(),
                    rope_type,
                    double_precision_rope,
                    vb.pp("transformer_1d_blocks").pp(index),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let learnable_registers = num_learnable_registers
            .map(|count| vb.get((count, inner_dim), "learnable_registers"))
            .transpose()?;
        Ok(Self {
            transformer_1d_blocks: blocks,
            learnable_registers,
            positional_embedding_theta,
            positional_embedding_max_pos,
            rope_type,
            double_precision_rope,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut hidden_states = hidden_states.clone();
        let mut attention_mask = attention_mask.clone();

        if let Some(registers) = &self.learnable_registers {
            let binary_mask = additive_mask_to_binary(&attention_mask)?;
            let (packed, packed_mask) =
                replace_padded_with_registers(&hidden_states, &binary_mask, registers)?;
            hidden_states = packed;
            attention_mask = convert_to_additive_mask(&packed_mask, hidden_states.dtype())?;
        }

        for block in &self.transformer_1d_blocks {
            hidden_states = block.forward(&hidden_states, Some(&attention_mask))?;
        }
        hidden_states = scale_free_rms_norm(&hidden_states, 1e-6)?;
        Ok((hidden_states, attention_mask))
    }

    pub fn positional_embedding_theta(&self) -> f64 {
        self.positional_embedding_theta
    }
}

fn scale_free_rms_norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let hidden_size = xs.dim(D::Minus1)?;
    let xs = xs.to_dtype(DType::F32)?;
    let variance = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    xs.broadcast_div(&(variance + eps)?.sqrt()?)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn connector_rotary_emb_cache(
    device: &Device,
    dtype: DType,
    seq_len: usize,
    heads: usize,
    dim_head: usize,
    theta: f64,
    positional_embedding_max_pos: &[usize],
    rope_type: LtxRopeType,
    double_precision_rope: bool,
) -> Result<(Tensor, Tensor)> {
    let position_dims = positional_embedding_max_pos.len();
    if position_dims == 0 {
        bail!("connector rotary embedding requires at least one positional dimension");
    }
    let steps = (heads * dim_head) / (2 * position_dims);
    if steps == 0 {
        bail!("connector rotary embedding dimension is too small");
    }

    let indices = if steps == 1 {
        Tensor::zeros((1,), DType::F32, device)?
    } else {
        let denom = (steps - 1) as f64;
        let values = (0..steps)
            .map(|index| {
                let ratio = index as f64 / denom;
                let power = if double_precision_rope {
                    theta.powf(ratio)
                } else {
                    (theta as f32).powf(ratio as f32) as f64
                };
                (power * std::f64::consts::PI / 2.0) as f32
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (steps,), device)?
    };

    let positions = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((1, seq_len, 1))?;
    let max_pos = positional_embedding_max_pos[0] as f64;
    let fractional = positions.affine(1.0 / max_pos, 0.0)?;
    let scaled = fractional.unsqueeze(D::Minus1)?.affine(2.0, -1.0)?;
    let freqs = indices
        .reshape((1, 1, 1, steps))?
        .broadcast_mul(&scaled)?
        .transpose(2, 3)?
        .contiguous()?
        .flatten_from(2)?;

    match rope_type {
        LtxRopeType::Interleaved => {
            let freq_unsq = freqs.unsqueeze(D::Minus1)?;
            let cos = Tensor::cat(&[freq_unsq.clone(), freq_unsq], D::Minus1)?
                .reshape((1, seq_len, freqs.dim(D::Minus1)? * 2))?
                .cos()?;
            let sin = Tensor::cat(&[freqs.unsqueeze(D::Minus1)?, freqs.unsqueeze(D::Minus1)?], D::Minus1)?
                .reshape((1, seq_len, freqs.dim(D::Minus1)? * 2))?
                .sin()?;
            Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
        }
        LtxRopeType::Split => {
            let expected = (heads * dim_head) / 2;
            let current = freqs.dim(D::Minus1)?;
            let pad_size = expected.saturating_sub(current);
            let mut cos = freqs.cos()?;
            let mut sin = freqs.sin()?;
            if pad_size != 0 {
                let cos_pad = Tensor::ones((1, seq_len, pad_size), DType::F32, device)?;
                let sin_pad = Tensor::zeros((1, seq_len, pad_size), DType::F32, device)?;
                cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
                sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
            }
            let cos = cos
                .reshape((1, seq_len, heads, expected / heads))?
                .transpose(1, 2)?
                .contiguous()?;
            let sin = sin
                .reshape((1, seq_len, heads, expected / heads))?
                .transpose(1, 2)?
                .contiguous()?;
            Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
        }
    }
}

fn norm_and_concat_padded_batch(
    encoded_text: &Tensor,
    attention_mask: &Tensor,
    padding_side: PaddingSide,
) -> Result<Tensor> {
    let device = encoded_text.device().clone();
    let encoded = encoded_text.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let attention_mask = attention_mask.to_device(&Device::Cpu)?.to_vec2::<u8>()?;
    let (batch, seq, hidden, layers) = encoded.dims4()?;
    let flat = encoded.flatten_all()?.to_vec1::<f32>()?;
    let index =
        |b: usize, t: usize, d: usize, l: usize| (((b * seq + t) * hidden + d) * layers) + l;

    let mut output = Vec::with_capacity(batch * seq * hidden * layers);
    for (batch_index, batch_mask) in attention_mask.iter().enumerate() {
        let sequence_length = batch_mask.iter().filter(|mask| **mask != 0).count();
        let valid_positions = match padding_side {
            PaddingSide::Right => (0..sequence_length).collect::<Vec<_>>(),
            PaddingSide::Left => ((seq - sequence_length)..seq).collect::<Vec<_>>(),
        };

        let mut sum = vec![0.0f32; layers];
        let mut min = vec![f32::INFINITY; layers];
        let mut max = vec![f32::NEG_INFINITY; layers];
        for &position in &valid_positions {
            for feature in 0..hidden {
                for layer_index in 0..layers {
                    let value = flat[index(batch_index, position, feature, layer_index)];
                    sum[layer_index] += value;
                    min[layer_index] = min[layer_index].min(value);
                    max[layer_index] = max[layer_index].max(value);
                }
            }
        }
        let denom = (sequence_length.max(1) * hidden) as f32;
        let means = sum.iter().map(|value| *value / denom).collect::<Vec<_>>();
        let ranges = min
            .iter()
            .zip(max.iter())
            .map(|(min, max)| (max - min).max(1e-6))
            .collect::<Vec<_>>();

        for position in 0..seq {
            let is_valid = batch_mask[position] != 0;
            for feature in 0..hidden {
                for layer_index in 0..layers {
                    let value = flat[index(batch_index, position, feature, layer_index)];
                    let normalized = if is_valid {
                        8.0 * (value - means[layer_index]) / ranges[layer_index]
                    } else {
                        0.0
                    };
                    output.push(normalized);
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, seq, hidden * layers),
        &device,
    )?)
}

fn rescale_norm(xs: &Tensor, target_dim: usize, source_dim: usize) -> Result<Tensor> {
    Ok((xs * ((target_dim as f64 / source_dim as f64).sqrt()))?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{
        convert_to_additive_mask, replace_padded_with_registers, Embeddings1DConnector,
        EmbeddingsFeatureExtractor, EmbeddingsProcessor, FeatureExtractorV1, FeatureExtractorV2,
        PaddingSide, Projection,
    };
    use crate::ltx2::model::LtxRopeType;

    fn projection(in_features: usize, out_features: usize) -> Projection {
        let device = Device::Cpu;
        let mut weight = vec![0.0f32; out_features * in_features];
        for row in 0..out_features {
            for col in 0..in_features {
                if row == col % out_features {
                    weight[row * in_features + col] = 1.0;
                }
            }
        }
        Projection::new(
            Tensor::from_vec(weight, (out_features, in_features), &device).unwrap(),
            None,
        )
    }

    #[test]
    fn projection_forward_uses_cpu_safe_compute_dtype_for_bf16_weights() {
        let xs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let weight = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let bias = Tensor::zeros(2, DType::BF16, &Device::Cpu).unwrap();
        let projection = Projection::new(weight, Some(bias));

        let ys = projection.forward(&xs).unwrap();

        assert_eq!(ys.dtype(), DType::F32);
        assert_eq!(
            ys.to_vec3::<f32>().unwrap(),
            vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]
        );
    }

    fn zero_connector_var_builder(
        dim: usize,
        num_layers: usize,
        num_registers: Option<usize>,
    ) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        for layer in 0..num_layers {
            for linear_name in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"] {
                tensors.insert(
                    format!("transformer_1d_blocks.{layer}.{linear_name}.weight"),
                    Tensor::zeros((dim, dim), DType::F32, &Device::Cpu).unwrap(),
                );
                tensors.insert(
                    format!("transformer_1d_blocks.{layer}.{linear_name}.bias"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            tensors.insert(
                format!("transformer_1d_blocks.{layer}.ff.net.0.proj.weight"),
                Tensor::zeros((dim * 4, dim), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("transformer_1d_blocks.{layer}.ff.net.0.proj.bias"),
                Tensor::zeros(dim * 4, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("transformer_1d_blocks.{layer}.ff.net.2.weight"),
                Tensor::zeros((dim, dim * 4), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("transformer_1d_blocks.{layer}.ff.net.2.bias"),
                Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        if let Some(count) = num_registers {
            tensors.insert(
                "learnable_registers".to_string(),
                Tensor::zeros((count, dim), DType::F32, &Device::Cpu).unwrap(),
            );
        }
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    #[test]
    fn feature_extractor_v1_produces_video_context_shape() {
        let device = Device::Cpu;
        let hidden_state_0 = Tensor::ones((1, 4, 3), DType::F32, &device).unwrap();
        let hidden_state_1 = Tensor::ones((1, 4, 3), DType::F32, &device)
            .unwrap()
            .affine(2.0, 0.0)
            .unwrap();
        let mask = Tensor::new(&[[0u8, 1, 1, 1]], &device).unwrap();
        let extractor = FeatureExtractorV1::new(projection(6, 4), false);
        let (video, audio) = extractor
            .forward(&[hidden_state_0, hidden_state_1], &mask, PaddingSide::Left)
            .unwrap();

        assert_eq!(video.dims3().unwrap(), (1, 4, 4));
        assert!(audio.is_none());
    }

    #[test]
    fn feature_extractor_v2_produces_video_and_audio_context_shapes() {
        let device = Device::Cpu;
        let hidden_state_0 = Tensor::ones((1, 3, 2), DType::F32, &device).unwrap();
        let hidden_state_1 = Tensor::ones((1, 3, 2), DType::F32, &device)
            .unwrap()
            .affine(3.0, 0.0)
            .unwrap();
        let mask = Tensor::new(&[[1u8, 1, 0]], &device).unwrap();
        let extractor = FeatureExtractorV2::new(projection(4, 5), Some(projection(4, 6)), 4);
        let (video, audio) = extractor
            .forward(&[hidden_state_0, hidden_state_1], &mask)
            .unwrap();

        assert_eq!(video.dims3().unwrap(), (1, 3, 5));
        assert_eq!(audio.unwrap().dims3().unwrap(), (1, 3, 6));
    }

    #[test]
    fn register_replacement_packs_valid_tokens_and_fills_padding() {
        let device = Device::Cpu;
        let hidden_states = Tensor::new(
            &[[[10.0f32, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]],
            &device,
        )
        .unwrap();
        let mask = Tensor::new(&[[0u8, 0, 1, 1]], &device).unwrap();
        let registers = Tensor::new(&[[100.0f32, 7.0], [200.0, 8.0]], &device).unwrap();

        let (packed, packed_mask) =
            replace_padded_with_registers(&hidden_states, &mask, &registers).unwrap();
        assert_eq!(
            packed.to_vec3::<f32>().unwrap(),
            vec![vec![
                vec![30.0, 3.0],
                vec![40.0, 4.0],
                vec![100.0, 7.0],
                vec![200.0, 8.0]
            ]]
        );
        assert_eq!(packed_mask.to_vec2::<u8>().unwrap(), vec![vec![1, 1, 1, 1]]);
    }

    #[test]
    fn register_replacement_uses_cpu_safe_compute_dtype_for_bf16_inputs() {
        let device = Device::Cpu;
        let hidden_states = Tensor::new(
            &[[[10.0f32, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]],
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let mask = Tensor::new(&[[0u8, 0, 1, 1]], &device).unwrap();
        let registers = Tensor::new(&[[100.0f32, 7.0], [200.0, 8.0]], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let (packed, packed_mask) =
            replace_padded_with_registers(&hidden_states, &mask, &registers).unwrap();

        assert_eq!(packed.dtype(), DType::F32);
        assert_eq!(
            packed.to_vec3::<f32>().unwrap(),
            vec![vec![
                vec![30.0, 3.0],
                vec![40.0, 4.0],
                vec![100.0, 7.0],
                vec![200.0, 8.0]
            ]]
        );
        assert_eq!(packed_mask.to_vec2::<u8>().unwrap(), vec![vec![1, 1, 1, 1]]);
    }

    #[test]
    fn additive_mask_conversion_marks_invalid_positions() {
        let mask = Tensor::new(&[[1u8, 0, 1]], &Device::Cpu).unwrap();
        let additive = convert_to_additive_mask(&mask, DType::F32).unwrap();
        let values = additive.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[0], 0.0);
        assert!(values[1] < -1e20);
        assert_eq!(values[2], 0.0);
    }

    #[test]
    fn embeddings_processor_runs_video_connector_and_preserves_binary_mask() {
        let device = Device::Cpu;
        let hidden_state_0 = Tensor::ones((1, 3, 2), DType::F32, &device).unwrap();
        let hidden_state_1 = Tensor::ones((1, 3, 2), DType::F32, &device)
            .unwrap()
            .affine(2.0, 0.0)
            .unwrap();
        let attention_mask = Tensor::new(&[[1u8, 1, 0]], &device).unwrap();
        let extractor =
            EmbeddingsFeatureExtractor::V1(FeatureExtractorV1::new(projection(4, 4), false));
        let connector = Embeddings1DConnector::new(
            1,
            4,
            1,
            10_000.0,
            vec![32],
            LtxRopeType::Split,
            true,
            None,
            zero_connector_var_builder(4, 1, None),
        )
        .unwrap();
        let processor = EmbeddingsProcessor::new(extractor, connector, None);

        let output = processor
            .process_hidden_states(
                &[hidden_state_0, hidden_state_1],
                &attention_mask,
                PaddingSide::Right,
            )
            .unwrap();

        assert_eq!(output.video_encoding.dims3().unwrap(), (1, 3, 4));
        assert!(output.audio_encoding.is_none());
        assert_eq!(
            output.attention_mask.to_vec2::<u8>().unwrap(),
            vec![vec![1, 1, 0]]
        );
    }

    #[test]
    fn embeddings_processor_runs_dual_connectors_for_audio_video() {
        let device = Device::Cpu;
        let hidden_state_0 = Tensor::ones((1, 3, 2), DType::F32, &device).unwrap();
        let hidden_state_1 = Tensor::ones((1, 3, 2), DType::F32, &device)
            .unwrap()
            .affine(3.0, 0.0)
            .unwrap();
        let attention_mask = Tensor::new(&[[1u8, 1, 0]], &device).unwrap();
        let extractor = EmbeddingsFeatureExtractor::V2(FeatureExtractorV2::new(
            projection(4, 4),
            Some(projection(4, 6)),
            4,
        ));
        let video_connector = Embeddings1DConnector::new(
            1,
            4,
            1,
            10_000.0,
            vec![32],
            LtxRopeType::Split,
            true,
            None,
            zero_connector_var_builder(4, 1, None),
        )
        .unwrap();
        let audio_connector = Embeddings1DConnector::new(
            3,
            2,
            1,
            10_000.0,
            vec![32],
            LtxRopeType::Split,
            true,
            None,
            zero_connector_var_builder(6, 1, None),
        )
        .unwrap();
        let processor = EmbeddingsProcessor::new(extractor, video_connector, Some(audio_connector));

        let output = processor
            .process_hidden_states(
                &[hidden_state_0, hidden_state_1],
                &attention_mask,
                PaddingSide::Left,
            )
            .unwrap();

        assert_eq!(output.video_encoding.dims3().unwrap(), (1, 3, 4));
        assert_eq!(output.audio_encoding.unwrap().dims3().unwrap(), (1, 3, 6));
    }
}
