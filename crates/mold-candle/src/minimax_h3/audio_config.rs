//! MiniMax H3 audio-VAE configuration and immutable media geometry.

use candle::{bail, DType, Result};
use serde::{Deserialize, Serialize};

pub const SAMPLE_RATE: usize = 32_000;
pub const SAMPLES_PER_LATENT: usize = 800;
pub const LATENT_ROWS_PER_SECOND: usize = SAMPLE_RATE / SAMPLES_PER_LATENT;
pub const LATENT_CHANNELS: usize = 32;

pub const LATENTS_MEAN: [f32; LATENT_CHANNELS] = [
    -0.020211687,
    0.38764665,
    -0.043_982_796,
    -0.28591514,
    0.08179686,
    -0.3578264,
    0.04062381,
    -0.015525345,
    -0.22336248,
    0.18210068,
    0.2941779,
    -0.07901168,
    -0.056815073,
    -0.36990282,
    -0.31616315,
    0.5905951,
    -0.05213957,
    0.01367316,
    -0.03691648,
    0.09732661,
    -0.33946624,
    -0.30685678,
    -0.24504599,
    -0.034698524,
    0.028680323,
    -0.2121778,
    -0.16782631,
    0.3221288,
    -0.12230559,
    0.43566048,
    -0.05025992,
    0.39792582,
];

pub const LATENTS_STD: [f32; LATENT_CHANNELS] = [
    1.6895524,
    2.7626374,
    1.7945344,
    1.6801682,
    1.6390227,
    2.7788298,
    1.765909,
    1.6199758,
    2.633_652_4,
    1.8539357,
    2.5056498,
    1.8110192,
    1.9579657,
    1.6685498,
    1.492_247,
    3.2986703,
    1.9491805,
    1.8720003,
    1.833408,
    1.648807,
    1.6176958,
    1.913_145,
    1.5695245,
    1.694366,
    1.8318421,
    1.554_063_8,
    1.9344931,
    1.5991982,
    1.718046,
    1.6307219,
    1.8661226,
    1.5613768,
];

/// The immutable, user-visible media contract. This intentionally carries no
/// model path or activation bit: capability registration remains fail-closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMaxH3AudioContract {
    pub sample_rate: usize,
    pub channels: usize,
    pub samples_per_latent: usize,
    pub latent_rows_per_second: usize,
    pub latent_channels: usize,
}

impl Default for MiniMaxH3AudioContract {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            channels: 2,
            samples_per_latent: SAMPLES_PER_LATENT,
            latent_rows_per_second: LATENT_ROWS_PER_SECOND,
            latent_channels: LATENT_CHANNELS,
        }
    }
}

/// DAC encoder plus BigVGAN decoder topology.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub latent_channels: usize,
    pub attention_heads: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub sample_rate: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Default for AudioVaeConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 4, 5, 5],
            latent_dim: 2048,
            latent_channels: LATENT_CHANNELS,
            attention_heads: 8,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: vec![9, 9, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            sample_rate: SAMPLE_RATE,
            latents_mean: LATENTS_MEAN.to_vec(),
            latents_std: LATENTS_STD.to_vec(),
        }
    }
}

impl AudioVaeConfig {
    /// Small weight-free topology for backend and shape conformance tests.
    /// Production never selects this configuration.
    #[cfg(test)]
    #[doc(hidden)]
    pub(super) fn tiny() -> Self {
        Self {
            encoder_dim: 2,
            encoder_rates: vec![2, 2],
            latent_dim: 8,
            latent_channels: 2,
            attention_heads: 2,
            decoder_dim: 8,
            decoder_rates: vec![2, 2],
            decoder_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            sample_rate: 16,
            latents_mean: vec![0.25, -0.5],
            latents_std: vec![2.0, 0.5],
        }
    }

    pub fn hop_length(&self) -> Result<usize> {
        checked_product(&self.encoder_rates, "encoder rates")
    }

    pub fn latent_rows_per_second(&self) -> Result<usize> {
        let hop = self.hop_length()?;
        if !self.sample_rate.is_multiple_of(hop) {
            bail!(
                "audio sample rate {} is not divisible by hop length {hop}",
                self.sample_rate
            )
        }
        Ok(self.sample_rate / hop)
    }

    pub fn padded_samples(&self, samples: usize) -> Result<usize> {
        if samples == 0 {
            bail!("MiniMax H3 audio input must contain at least one sample")
        }
        let hop = self.hop_length()?;
        samples
            .checked_add(hop - 1)
            .map(|value| value / hop * hop)
            .ok_or_else(|| candle::Error::Msg("audio right-padding overflow".into()))
    }

    pub fn validate_execution_dtype(&self, dtype: DType) -> Result<()> {
        if dtype != DType::F32 {
            bail!("MiniMax H3 audio VAE is FP32-only; requested execution dtype {dtype:?}")
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.encoder_rates.is_empty() || self.decoder_rates.is_empty() {
            bail!("MiniMax H3 audio encoder and decoder rates must be non-empty")
        }
        if self.encoder_rates.contains(&0) || self.decoder_rates.contains(&0) {
            bail!("MiniMax H3 audio rates must be non-zero")
        }
        let hop = self.hop_length()?;
        let decoder_hop = checked_product(&self.decoder_rates, "decoder rates")?;
        if decoder_hop != hop {
            bail!("MiniMax H3 audio decoder hop {decoder_hop} does not match encoder hop {hop}")
        }
        if self.decoder_rates.len() != self.decoder_kernel_sizes.len() {
            bail!("MiniMax H3 audio decoder rates and kernels must have equal lengths")
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len()
            || self.resblock_kernel_sizes.is_empty()
        {
            bail!("MiniMax H3 audio resblock kernels and dilation tables must be non-empty and aligned")
        }
        if self
            .resblock_dilation_sizes
            .iter()
            .any(|dilations| dilations.is_empty() || dilations.contains(&0))
        {
            bail!("MiniMax H3 audio resblock dilations must be non-empty and non-zero")
        }
        if self.latent_channels == 0
            || self.latent_dim == 0
            || self.attention_heads == 0
            || !self.latent_dim.is_multiple_of(self.attention_heads)
        {
            bail!("MiniMax H3 audio latent and attention dimensions are invalid")
        }
        let head_dim = self.latent_dim / self.attention_heads;
        if !head_dim.is_multiple_of(self.latent_channels) {
            bail!(
                "MiniMax H3 audio attention head width {head_dim} must pool exactly to {} latent channels",
                self.latent_channels
            )
        }
        let expected_latent_dim = self
            .encoder_dim
            .checked_mul(1usize << self.encoder_rates.len())
            .ok_or_else(|| candle::Error::Msg("audio encoder width overflow".into()))?;
        if expected_latent_dim != self.latent_dim {
            bail!(
                "MiniMax H3 audio encoder ends at width {expected_latent_dim}, expected latent_dim {}",
                self.latent_dim
            )
        }
        if !self
            .decoder_dim
            .is_multiple_of(1usize << self.decoder_rates.len())
        {
            bail!("MiniMax H3 audio decoder width cannot halve at every stage")
        }
        if self.latents_mean.len() != self.latent_channels
            || self.latents_std.len() != self.latent_channels
            || self
                .latents_std
                .iter()
                .any(|std| !std.is_finite() || *std <= 0.0)
            || self.latents_mean.iter().any(|mean| !mean.is_finite())
        {
            bail!("MiniMax H3 audio latent normalization vectors are invalid")
        }
        if self.sample_rate == 0 || !self.sample_rate.is_multiple_of(hop) {
            bail!("MiniMax H3 audio sample rate must be a non-zero multiple of the hop")
        }
        Ok(())
    }

    /// Conservative boundary-tensor watermark used only by tiny synthetic
    /// tests. It is intentionally not a production admission estimate.
    #[doc(hidden)]
    pub fn tiny_test_boundary_peak_bytes(&self, batch: usize, samples: usize) -> Result<usize> {
        let padded = self.padded_samples(samples)?;
        let rows = padded / self.hop_length()?;
        let waveform = batch
            .checked_mul(2)
            .and_then(|v| v.checked_mul(padded))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle::Error::Msg("tiny audio memory accounting overflow".into()))?;
        let encoder = batch
            .checked_mul(2)
            .and_then(|v| v.checked_mul(self.latent_dim))
            .and_then(|v| v.checked_mul(rows))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle::Error::Msg("tiny audio memory accounting overflow".into()))?;
        Ok(waveform.max(encoder))
    }
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| candle::Error::Msg(format!("MiniMax H3 audio {label} product overflow")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_contract_is_exact() -> Result<()> {
        let config = AudioVaeConfig::default();
        config.validate()?;
        assert_eq!(config.encoder_rates, [2, 4, 4, 5, 5]);
        assert_eq!(config.decoder_rates, [5, 5, 2, 2, 2, 2, 2]);
        assert_eq!(config.decoder_kernel_sizes, [9, 9, 4, 4, 4, 4, 4]);
        assert_eq!(config.hop_length()?, SAMPLES_PER_LATENT);
        assert_eq!(config.latent_rows_per_second()?, LATENT_ROWS_PER_SECOND);
        assert_eq!(config.latent_channels, LATENT_CHANNELS);
        assert_eq!(MiniMaxH3AudioContract::default().channels, 2);
        Ok(())
    }

    #[test]
    fn padding_is_right_only_and_rejects_empty_input() -> Result<()> {
        let config = AudioVaeConfig::default();
        assert_eq!(config.padded_samples(1)?, 800);
        assert_eq!(config.padded_samples(800)?, 800);
        assert_eq!(config.padded_samples(801)?, 1600);
        assert!(config.padded_samples(0).is_err());
        Ok(())
    }

    #[test]
    fn execution_dtype_fails_closed() {
        let config = AudioVaeConfig::default();
        assert!(config.validate_execution_dtype(DType::F32).is_ok());
        assert!(config.validate_execution_dtype(DType::BF16).is_err());
        assert!(config.validate_execution_dtype(DType::F16).is_err());
    }

    #[test]
    fn tiny_memory_watermark_is_scoped_and_bounded() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        config.validate()?;
        let peak = config.tiny_test_boundary_peak_bytes(1, 7)?;
        assert_eq!(peak, 128);
        assert!(peak < 1024);
        Ok(())
    }
}
