#![allow(dead_code)]

use anyhow::{bail, Result};
use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct TransformerModalityInputs {
    pub latent: Tensor,
    pub context: Tensor,
    pub positions: Tensor,
    pub attention_mask: Option<Tensor>,
    pub enabled: bool,
}

impl TransformerModalityInputs {
    pub fn latent_shape(&self) -> Result<(usize, usize, usize)> {
        self.latent.dims3().map_err(Into::into)
    }

    pub fn context_shape(&self) -> Result<(usize, usize, usize)> {
        self.context.dims3().map_err(Into::into)
    }

    pub fn validate(&self, modality_name: &str) -> Result<(usize, usize, usize)> {
        let (batch, tokens, channels) = self.latent.dims3()?;
        let (context_batch, _context_tokens, _context_dim) = self.context.dims3()?;
        let (position_batch, _position_dims, position_tokens, _bounds) = self.positions.dims4()?;
        if batch != context_batch || batch != position_batch {
            bail!(
                "{modality_name} transformer inputs must share the same batch size across latent, context, and positions"
            );
        }
        if tokens != position_tokens {
            bail!(
                "{modality_name} positions token count ({position_tokens}) must match latent tokens ({tokens})"
            );
        }
        if let Some(mask) = &self.attention_mask {
            let (mask_batch, mask_tokens, other_tokens) = mask.dims3()?;
            if mask_batch != batch || mask_tokens != tokens || other_tokens != tokens {
                bail!("{modality_name} self-attention mask must be shaped [batch, tokens, tokens]");
            }
        }
        Ok((batch, tokens, channels))
    }
}

#[derive(Debug, Clone, Default)]
pub struct TransformerInputContract {
    pub video: Option<TransformerModalityInputs>,
    pub audio: Option<TransformerModalityInputs>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerOutputContract {
    pub video_shape: Option<(usize, usize, usize)>,
    pub audio_shape: Option<(usize, usize, usize)>,
}

impl TransformerInputContract {
    pub fn validate(&self) -> Result<TransformerOutputContract> {
        if self.video.is_none() && self.audio.is_none() {
            bail!("dual-stream transformer requires at least one enabled modality");
        }

        let video_shape = self
            .video
            .as_ref()
            .map(|video| video.validate("video"))
            .transpose()?;
        let audio_shape = self
            .audio
            .as_ref()
            .map(|audio| audio.validate("audio"))
            .transpose()?;

        if let (Some((video_batch, _, _)), Some((audio_batch, _, _))) = (video_shape, audio_shape) {
            if video_batch != audio_batch {
                bail!("video and audio transformer branches must share the same batch size");
            }
        }

        Ok(TransformerOutputContract {
            video_shape,
            audio_shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{TransformerInputContract, TransformerModalityInputs};

    fn modality(tokens: usize, channels: usize, position_dims: usize) -> TransformerModalityInputs {
        let device = Device::Cpu;
        TransformerModalityInputs {
            latent: Tensor::zeros((1, tokens, channels), candle_core::DType::F32, &device).unwrap(),
            context: Tensor::zeros((1, 5, 16), candle_core::DType::F32, &device).unwrap(),
            positions: Tensor::zeros(
                (1, position_dims, tokens, 2),
                candle_core::DType::F32,
                &device,
            )
            .unwrap(),
            attention_mask: Some(
                Tensor::zeros((1, tokens, tokens), candle_core::DType::F32, &device).unwrap(),
            ),
            enabled: true,
        }
    }

    #[test]
    fn transformer_shape_contract_accepts_video_only_inputs() {
        let contract = TransformerInputContract {
            video: Some(modality(32, 128, 3)),
            audio: None,
        };
        let output = contract.validate().unwrap();
        assert_eq!(output.video_shape, Some((1, 32, 128)));
        assert_eq!(output.audio_shape, None);
    }

    #[test]
    fn transformer_shape_contract_accepts_audio_only_inputs() {
        let contract = TransformerInputContract {
            video: None,
            audio: Some(modality(48, 64, 1)),
        };
        let output = contract.validate().unwrap();
        assert_eq!(output.video_shape, None);
        assert_eq!(output.audio_shape, Some((1, 48, 64)));
    }

    #[test]
    fn transformer_shape_contract_accepts_dual_stream_inputs() {
        let contract = TransformerInputContract {
            video: Some(modality(32, 128, 3)),
            audio: Some(modality(48, 64, 1)),
        };
        let output = contract.validate().unwrap();
        assert_eq!(output.video_shape, Some((1, 32, 128)));
        assert_eq!(output.audio_shape, Some((1, 48, 64)));
    }
}
