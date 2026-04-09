#![allow(dead_code)]

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};

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
        self.weight.dims2().map(|(rows, _)| rows).map_err(Into::into)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        let (out, in_features) = self.weight.dims2()?;
        if hidden != in_features {
            bail!(
                "projection input dimension mismatch: expected {in_features}, got {hidden}"
            );
        }
        let ys = xs
            .reshape((batch * seq, hidden))?
            .matmul(&self.weight.transpose(0, 1)?)?;
        let ys = if let Some(bias) = &self.bias {
            ys.broadcast_add(bias)?
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
                    &rescale_norm(&normed, projection.out_features().unwrap(), self.embedding_dim)
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
    let mask = attention_mask.to_dtype(DType::F32)?.reshape((batch, seq, 1))?;
    normed.broadcast_mul(&mask).map_err(Into::into)
}

pub fn replace_padded_with_registers(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    registers: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = hidden_states.device().clone();
    let hidden_states = hidden_states.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
    let attention_mask = attention_mask.to_device(&Device::Cpu)?.to_vec2::<u8>()?;
    let registers = registers.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
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
        Tensor::from_vec(packed, (batch, seq, dim), &device)?,
        Tensor::from_vec(binary_mask, (batch, seq), &device)?,
    ))
}

fn norm_and_concat_padded_batch(
    encoded_text: &Tensor,
    attention_mask: &Tensor,
    padding_side: PaddingSide,
) -> Result<Tensor> {
    let device = encoded_text.device().clone();
    let encoded = encoded_text.to_device(&Device::Cpu)?;
    let attention_mask = attention_mask.to_device(&Device::Cpu)?.to_vec2::<u8>()?;
    let (batch, seq, hidden, layers) = encoded.dims4()?;
    let flat = encoded.flatten_all()?.to_vec1::<f32>()?;
    let index = |b: usize, t: usize, d: usize, l: usize| (((b * seq + t) * hidden + d) * layers) + l;

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

    Ok(Tensor::from_vec(output, (batch, seq, hidden * layers), &device)?)
}

fn rescale_norm(xs: &Tensor, target_dim: usize, source_dim: usize) -> Result<Tensor> {
    Ok((xs * ((target_dim as f64 / source_dim as f64).sqrt()))?)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        replace_padded_with_registers, FeatureExtractorV1, FeatureExtractorV2, PaddingSide,
        Projection,
    };

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
        let (video, audio) = extractor.forward(&[hidden_state_0, hidden_state_1], &mask).unwrap();

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
}
