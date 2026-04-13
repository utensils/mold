#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::{group_norm, ops, Conv2d, Conv2dConfig, GroupNorm, Module, VarBuilder};
use serde::Deserialize;
use serde_json::Value;

use super::video_vae::PerChannelRmsNorm;
use super::{AudioLatentShape, AudioPatchifier};

const LATENT_DOWNSAMPLE_FACTOR: usize = 4;

fn silu(x: &Tensor) -> Result<Tensor> {
    ops::silu(x).map_err(Into::into)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioNormType {
    Group,
    Pixel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioCausalityAxis {
    None,
    Width,
    WidthCompatibility,
    Height,
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioDecoderConfig {
    pub ch: usize,
    pub out_ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub resolution: usize,
    pub z_channels: usize,
    pub norm_type: AudioNormType,
    pub causality_axis: AudioCausalityAxis,
    pub dropout: f64,
    pub mid_block_add_attention: bool,
    pub sample_rate: usize,
    pub mel_hop_length: usize,
    pub is_causal: bool,
    pub mel_bins: usize,
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioEncoderConfig {
    pub ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_resolutions: Vec<usize>,
    pub resolution: usize,
    pub z_channels: usize,
    pub double_z: bool,
    pub norm_type: AudioNormType,
    pub causality_axis: AudioCausalityAxis,
    pub dropout: f64,
    pub mid_block_add_attention: bool,
    pub resamp_with_conv: bool,
    pub in_channels: usize,
    pub sample_rate: usize,
    pub mel_hop_length: usize,
    pub n_fft: usize,
    pub is_causal: bool,
    pub mel_bins: usize,
}

impl Ltx2AudioDecoderConfig {
    pub fn load(checkpoint_path: &Path) -> Result<Self> {
        let config_json = read_checkpoint_config_json(checkpoint_path)?;
        let checkpoint: CheckpointConfig =
            serde_json::from_str(&config_json).with_context(|| {
                format!(
                    "failed to parse LTX-2 checkpoint config metadata from {}",
                    checkpoint_path.display()
                )
            })?;
        let ddconfig = checkpoint.audio_vae.model.params.ddconfig;
        let preprocessing = checkpoint.audio_vae.preprocessing;
        Ok(Self {
            ch: ddconfig.ch,
            out_ch: ddconfig.out_ch,
            ch_mult: ddconfig.ch_mult,
            num_res_blocks: ddconfig.num_res_blocks,
            attn_resolutions: ddconfig.attn_resolutions,
            resolution: ddconfig.resolution,
            z_channels: ddconfig.z_channels,
            norm_type: ddconfig.norm_type,
            causality_axis: ddconfig.causality_axis,
            dropout: ddconfig.dropout,
            mid_block_add_attention: ddconfig.mid_block_add_attention,
            sample_rate: checkpoint.audio_vae.model.params.sampling_rate,
            mel_hop_length: preprocessing.stft.hop_length,
            is_causal: preprocessing.stft.causal,
            mel_bins: preprocessing.mel.n_mel_channels,
        })
    }
}

impl Ltx2AudioEncoderConfig {
    pub fn load(checkpoint_path: &Path) -> Result<Self> {
        let config_json = read_checkpoint_config_json(checkpoint_path)?;
        let checkpoint: CheckpointConfig =
            serde_json::from_str(&config_json).with_context(|| {
                format!(
                    "failed to parse LTX-2 checkpoint config metadata from {}",
                    checkpoint_path.display()
                )
            })?;
        let ddconfig = checkpoint.audio_vae.model.params.ddconfig;
        let preprocessing = checkpoint.audio_vae.preprocessing;
        Ok(Self {
            ch: ddconfig.ch,
            ch_mult: ddconfig.ch_mult,
            num_res_blocks: ddconfig.num_res_blocks,
            attn_resolutions: ddconfig.attn_resolutions,
            resolution: ddconfig.resolution,
            z_channels: ddconfig.z_channels,
            double_z: ddconfig.double_z.unwrap_or(true),
            norm_type: ddconfig.norm_type,
            causality_axis: ddconfig.causality_axis,
            dropout: ddconfig.dropout,
            mid_block_add_attention: ddconfig.mid_block_add_attention,
            resamp_with_conv: ddconfig.resamp_with_conv.unwrap_or(true),
            in_channels: ddconfig.in_channels,
            sample_rate: checkpoint.audio_vae.model.params.sampling_rate,
            mel_hop_length: preprocessing.stft.hop_length,
            n_fft: preprocessing.stft.filter_length.unwrap_or(1024),
            is_causal: preprocessing.stft.causal,
            mel_bins: preprocessing.mel.n_mel_channels,
        })
    }
}

#[derive(Debug, Deserialize)]
struct CheckpointConfig {
    audio_vae: CheckpointAudioVae,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioVae {
    model: CheckpointAudioVaeModel,
    preprocessing: CheckpointAudioPreprocessing,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioVaeModel {
    params: CheckpointAudioVaeParams,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioVaeParams {
    ddconfig: CheckpointAudioDdConfig,
    sampling_rate: usize,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioDdConfig {
    mel_bins: usize,
    z_channels: usize,
    resolution: usize,
    in_channels: usize,
    out_ch: usize,
    ch: usize,
    ch_mult: Vec<usize>,
    num_res_blocks: usize,
    attn_resolutions: Vec<usize>,
    double_z: Option<bool>,
    dropout: f64,
    resamp_with_conv: Option<bool>,
    mid_block_add_attention: bool,
    norm_type: AudioNormType,
    causality_axis: AudioCausalityAxis,
}

#[derive(Debug, Deserialize)]
struct CheckpointAudioPreprocessing {
    stft: CheckpointStftConfig,
    mel: CheckpointMelConfig,
}

#[derive(Debug, Deserialize)]
struct CheckpointStftConfig {
    hop_length: usize,
    causal: bool,
    filter_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CheckpointMelConfig {
    n_mel_channels: usize,
}

pub(crate) fn read_checkpoint_config_json(checkpoint_path: &Path) -> Result<String> {
    let mut file = File::open(checkpoint_path)
        .with_context(|| format!("failed to open {}", checkpoint_path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).with_context(|| {
        format!(
            "failed to read safetensors header length from {}",
            checkpoint_path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).with_context(|| {
        format!(
            "failed to read safetensors header bytes from {}",
            checkpoint_path.display()
        )
    })?;
    let header: HashMap<String, Value> =
        serde_json::from_slice(&header_buf).with_context(|| {
            format!(
                "failed to parse safetensors header JSON from {}",
                checkpoint_path.display()
            )
        })?;
    let metadata = header
        .get("__metadata__")
        .and_then(Value::as_object)
        .context("safetensors metadata did not contain '__metadata__'")?;
    metadata
        .get("config")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .context("safetensors metadata did not contain a 'config' entry")
}

#[derive(Debug, Clone)]
struct AudioPerChannelStatistics {
    mean: Tensor,
    std: Tensor,
}

impl AudioPerChannelStatistics {
    fn load(audio_vb: VarBuilder, features: usize) -> Result<Self> {
        let stats_vb = audio_vb.pp("per_channel_statistics");
        let mean = stats_vb
            .get(features, "mean-of-means")
            .context("failed to load audio per-channel mean statistics")?;
        let std = stats_vb
            .get(features, "std-of-means")
            .context("failed to load audio per-channel std statistics")?;
        Ok(Self { mean, std })
    }

    fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, features) = x.dims3()?;
        let mean = self
            .mean
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        let std = self
            .std
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        x.broadcast_mul(&std)?
            .broadcast_add(&mean)
            .map_err(Into::into)
    }

    fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, features) = x.dims3()?;
        let mean = self
            .mean
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        let std = self
            .std
            .reshape((1, 1, features))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        x.broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
enum AudioNorm {
    Group(GroupNorm),
    Pixel(PerChannelRmsNorm),
}

impl AudioNorm {
    fn load(channels: usize, norm_type: AudioNormType, vb: VarBuilder) -> Result<Self> {
        Ok(match norm_type {
            AudioNormType::Group => Self::Group(group_norm(32, channels, 1e-6, vb)?),
            AudioNormType::Pixel => Self::Pixel(PerChannelRmsNorm::new(1, 1e-6)),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Group(norm) => norm.forward(x).map_err(Into::into),
            Self::Pixel(norm) => norm.forward(x).map_err(Into::into),
        }
    }
}

#[derive(Debug, Clone)]
struct AudioCausalConv2d {
    conv: Conv2d,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl AudioCausalConv2d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = kernel_size.saturating_sub(1);
        let (pad_left, pad_right, pad_top, pad_bottom) = match causality_axis {
            AudioCausalityAxis::None => (pad / 2, pad - (pad / 2), pad / 2, pad - (pad / 2)),
            AudioCausalityAxis::Width | AudioCausalityAxis::WidthCompatibility => {
                (pad, 0, pad / 2, pad - (pad / 2))
            }
            AudioCausalityAxis::Height => (pad / 2, pad - (pad / 2), pad, 0),
        };
        let conv_vb = vb.pp("conv");
        let weight = conv_vb.get(
            (out_channels, in_channels, kernel_size, kernel_size),
            "weight",
        )?;
        let bias = conv_vb.get(out_channels, "bias").ok();
        let conv = Conv2d::new(
            weight,
            bias,
            Conv2dConfig {
                stride,
                ..Default::default()
            },
        );
        Ok(Self {
            conv,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = x
            .pad_with_zeros(3, self.pad_left, self.pad_right)?
            .pad_with_zeros(2, self.pad_top, self.pad_bottom)?;
        padded.apply(&self.conv).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct AudioResnetBlock {
    in_channels: usize,
    out_channels: usize,
    norm1: AudioNorm,
    conv1: AudioCausalConv2d,
    norm2: AudioNorm,
    conv2: AudioCausalConv2d,
    nin_shortcut: Option<AudioCausalConv2d>,
}

impl AudioResnetBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        norm_type: AudioNormType,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = AudioNorm::load(in_channels, norm_type, vb.pp("norm1"))?;
        let conv1 = AudioCausalConv2d::new(
            in_channels,
            out_channels,
            3,
            1,
            causality_axis,
            vb.pp("conv1"),
        )?;
        let norm2 = AudioNorm::load(out_channels, norm_type, vb.pp("norm2"))?;
        let conv2 = AudioCausalConv2d::new(
            out_channels,
            out_channels,
            3,
            1,
            causality_axis,
            vb.pp("conv2"),
        )?;
        let nin_shortcut = if in_channels != out_channels {
            Some(AudioCausalConv2d::new(
                in_channels,
                out_channels,
                1,
                1,
                causality_axis,
                vb.pp("nin_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            in_channels,
            out_channels,
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.norm1.forward(x)?;
        h = silu(&h)?;
        h = self.conv1.forward(&h)?;
        h = self.norm2.forward(&h)?;
        h = silu(&h)?;
        h = self.conv2.forward(&h)?;
        let residual = match &self.nin_shortcut {
            Some(shortcut) => shortcut.forward(x)?,
            None => x.clone(),
        };
        residual.add(&h).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct AudioMidBlock {
    block_1: AudioResnetBlock,
    block_2: AudioResnetBlock,
}

impl AudioMidBlock {
    fn load(
        channels: usize,
        norm_type: AudioNormType,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            block_1: AudioResnetBlock::load(
                channels,
                channels,
                norm_type,
                causality_axis,
                vb.pp("block_1"),
            )?,
            block_2: AudioResnetBlock::load(
                channels,
                channels,
                norm_type,
                causality_axis,
                vb.pp("block_2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block_1.forward(x)?;
        self.block_2.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct AudioDownsample {
    conv: Conv2d,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

impl AudioDownsample {
    fn load(
        in_channels: usize,
        with_conv: bool,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        if !with_conv {
            bail!("audio VAE downsample without convolution is not implemented for native LTX-2");
        }
        let (pad_left, pad_right, pad_top, pad_bottom) = match causality_axis {
            AudioCausalityAxis::None => (0, 1, 0, 1),
            AudioCausalityAxis::Width => (2, 0, 0, 1),
            AudioCausalityAxis::Height => (0, 1, 2, 0),
            AudioCausalityAxis::WidthCompatibility => (1, 0, 0, 1),
        };
        let conv_vb = vb.pp("conv");
        let weight = conv_vb.get((in_channels, in_channels, 3, 3), "weight")?;
        let bias = conv_vb.get(in_channels, "bias").ok();
        let conv = Conv2d::new(
            weight,
            bias,
            Conv2dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        Ok(Self {
            conv,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = x
            .pad_with_zeros(3, self.pad_left, self.pad_right)?
            .pad_with_zeros(2, self.pad_top, self.pad_bottom)?;
        padded.apply(&self.conv).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct AudioUpsample {
    causality_axis: AudioCausalityAxis,
    conv: AudioCausalConv2d,
}

impl AudioUpsample {
    fn load(
        in_channels: usize,
        causality_axis: AudioCausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            causality_axis,
            conv: AudioCausalConv2d::new(
                in_channels,
                in_channels,
                3,
                1,
                causality_axis,
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let mut x = x.upsample_nearest2d(h * 2, w * 2)?;
        x = self.conv.forward(&x)?;
        match self.causality_axis {
            AudioCausalityAxis::None | AudioCausalityAxis::WidthCompatibility => Ok(x),
            AudioCausalityAxis::Height => {
                let (_, _, up_h, _) = x.dims4()?;
                x.narrow(2, 1, up_h.saturating_sub(1)).map_err(Into::into)
            }
            AudioCausalityAxis::Width => {
                let (_, _, _, up_w) = x.dims4()?;
                x.narrow(3, 1, up_w.saturating_sub(1)).map_err(Into::into)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct AudioDecoderStage {
    blocks: Vec<AudioResnetBlock>,
    upsample: Option<AudioUpsample>,
}

#[derive(Debug, Clone)]
struct AudioEncoderStage {
    blocks: Vec<AudioResnetBlock>,
    downsample: Option<AudioDownsample>,
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioEncoder {
    pub config: Ltx2AudioEncoderConfig,
    per_channel_statistics: AudioPerChannelStatistics,
    patchifier: AudioPatchifier,
    conv_in: AudioCausalConv2d,
    down: Vec<AudioEncoderStage>,
    mid: AudioMidBlock,
    norm_out: AudioNorm,
    conv_out: AudioCausalConv2d,
}

impl Ltx2AudioEncoder {
    pub fn load_from_checkpoint(
        checkpoint_path: &Path,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let config = Ltx2AudioEncoderConfig::load(checkpoint_path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[PathBuf::from(checkpoint_path)], dtype, device)
        }
        .with_context(|| format!("failed to mmap {}", checkpoint_path.display()))?;
        Self::new(config, vb)
    }

    pub fn new(config: Ltx2AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        if !config.attn_resolutions.is_empty() {
            bail!("audio VAE encoder attention blocks are not implemented for native LTX-2");
        }
        if config.mid_block_add_attention {
            bail!("audio VAE encoder mid-block attention is not implemented for native LTX-2");
        }
        if config.dropout != 0.0 {
            bail!("audio VAE encoder dropout != 0.0 is not implemented for native LTX-2");
        }

        let audio_vb = vb.pp("audio_vae");
        let encoder_vb = audio_vb.pp("encoder");
        let per_channel_statistics = AudioPerChannelStatistics::load(audio_vb.clone(), config.ch)?;
        let patchifier = AudioPatchifier::new(
            config.sample_rate,
            config.mel_hop_length,
            LATENT_DOWNSAMPLE_FACTOR,
            config.is_causal,
            0,
        );
        let conv_in = AudioCausalConv2d::new(
            config.in_channels,
            config.ch,
            3,
            1,
            config.causality_axis,
            encoder_vb.pp("conv_in"),
        )?;

        let num_resolutions = config.ch_mult.len();
        let mut down = Vec::with_capacity(num_resolutions);
        let mut block_in = config.ch;
        for level in 0..num_resolutions {
            let block_out = config.ch * config.ch_mult[level];
            let mut blocks = Vec::with_capacity(config.num_res_blocks);
            for block_idx in 0..config.num_res_blocks {
                let block = AudioResnetBlock::load(
                    block_in,
                    block_out,
                    config.norm_type,
                    config.causality_axis,
                    encoder_vb.pp(format!("down.{level}.block.{block_idx}")),
                )?;
                block_in = block_out;
                blocks.push(block);
            }
            let downsample = if level + 1 < num_resolutions {
                Some(AudioDownsample::load(
                    block_in,
                    config.resamp_with_conv,
                    config.causality_axis,
                    encoder_vb.pp(format!("down.{level}.downsample")),
                )?)
            } else {
                None
            };
            down.push(AudioEncoderStage { blocks, downsample });
        }

        let mid = AudioMidBlock::load(
            block_in,
            config.norm_type,
            config.causality_axis,
            encoder_vb.pp("mid"),
        )?;
        let norm_out = AudioNorm::load(block_in, config.norm_type, encoder_vb.pp("norm_out"))?;
        let conv_out = AudioCausalConv2d::new(
            block_in,
            if config.double_z {
                2 * config.z_channels
            } else {
                config.z_channels
            },
            3,
            1,
            config.causality_axis,
            encoder_vb.pp("conv_out"),
        )?;

        Ok(Self {
            config,
            per_channel_statistics,
            patchifier,
            conv_in,
            down,
            mid,
            norm_out,
            conv_out,
        })
    }

    pub fn encode(&self, spectrogram: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(spectrogram)?;
        for stage in &self.down {
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if let Some(downsample) = stage.downsample.as_ref() {
                h = downsample.forward(&h)?;
            }
        }
        h = self.mid.forward(&h)?;
        h = self.norm_out.forward(&h)?;
        h = silu(&h)?;
        h = self.conv_out.forward(&h)?;
        self.normalize_latents(&h)
    }

    fn normalize_latents(&self, latent_output: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, mel_bins) = latent_output.dims4()?;
        let means = if self.config.double_z {
            latent_output.narrow(1, 0, self.config.z_channels.min(channels))?
        } else {
            latent_output.clone()
        };
        let (_, mean_channels, _, _) = means.dims4()?;
        let latent_shape = AudioLatentShape {
            batch,
            channels: mean_channels,
            frames,
            mel_bins,
        };
        let latent_patched = self.patchifier.patchify(&means)?;
        let latent_normalized = self.per_channel_statistics.normalize(&latent_patched)?;
        self.patchifier
            .unpatchify(&latent_normalized, latent_shape)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2AudioDecoder {
    pub config: Ltx2AudioDecoderConfig,
    per_channel_statistics: AudioPerChannelStatistics,
    patchifier: AudioPatchifier,
    conv_in: AudioCausalConv2d,
    mid: AudioMidBlock,
    up: Vec<AudioDecoderStage>,
    norm_out: AudioNorm,
    conv_out: AudioCausalConv2d,
}

impl Ltx2AudioDecoder {
    pub fn load_from_checkpoint(
        checkpoint_path: &Path,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let config = Ltx2AudioDecoderConfig::load(checkpoint_path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[PathBuf::from(checkpoint_path)], dtype, device)
        }
        .with_context(|| format!("failed to mmap {}", checkpoint_path.display()))?;
        Self::new(config, vb)
    }

    pub fn new(config: Ltx2AudioDecoderConfig, vb: VarBuilder) -> Result<Self> {
        if !config.attn_resolutions.is_empty() {
            bail!("audio VAE attention blocks are not implemented for native LTX-2");
        }
        if config.mid_block_add_attention {
            bail!("audio VAE mid-block attention is not implemented for native LTX-2");
        }
        if config.dropout != 0.0 {
            bail!("audio VAE dropout != 0.0 is not implemented for native LTX-2");
        }

        let audio_vb = vb.pp("audio_vae");
        let decoder_vb = audio_vb.pp("decoder");
        let per_channel_statistics = AudioPerChannelStatistics::load(audio_vb.clone(), config.ch)?;
        let patchifier = AudioPatchifier::new(
            config.sample_rate,
            config.mel_hop_length,
            LATENT_DOWNSAMPLE_FACTOR,
            config.is_causal,
            0,
        );

        let base_block_channels = config.ch * config.ch_mult[config.ch_mult.len() - 1];
        let conv_in = AudioCausalConv2d::new(
            config.z_channels,
            base_block_channels,
            3,
            1,
            config.causality_axis,
            decoder_vb.pp("conv_in"),
        )?;
        let mid = AudioMidBlock::load(
            base_block_channels,
            config.norm_type,
            config.causality_axis,
            decoder_vb.pp("mid"),
        )?;

        let num_resolutions = config.ch_mult.len();
        let mut block_in = base_block_channels;
        let mut up = (0..num_resolutions)
            .map(|_| None)
            .collect::<Vec<Option<AudioDecoderStage>>>();
        for level in (0..num_resolutions).rev() {
            let block_out = config.ch * config.ch_mult[level];
            let mut blocks = Vec::with_capacity(config.num_res_blocks + 1);
            for block_idx in 0..(config.num_res_blocks + 1) {
                let block = AudioResnetBlock::load(
                    block_in,
                    block_out,
                    config.norm_type,
                    config.causality_axis,
                    decoder_vb.pp(format!("up.{level}.block.{block_idx}")),
                )?;
                block_in = block_out;
                blocks.push(block);
            }
            let upsample = if level != 0 {
                Some(AudioUpsample::load(
                    block_in,
                    config.causality_axis,
                    decoder_vb.pp(format!("up.{level}.upsample")),
                )?)
            } else {
                None
            };
            up[level] = Some(AudioDecoderStage { blocks, upsample });
        }
        let up = up
            .into_iter()
            .map(|stage| stage.context("audio decoder stage was not initialized"))
            .collect::<Result<Vec<_>>>()?;

        let norm_out = AudioNorm::load(block_in, config.norm_type, decoder_vb.pp("norm_out"))?;
        let conv_out = AudioCausalConv2d::new(
            block_in,
            config.out_ch,
            3,
            1,
            config.causality_axis,
            decoder_vb.pp("conv_out"),
        )?;

        Ok(Self {
            config,
            per_channel_statistics,
            patchifier,
            conv_in,
            mid,
            up,
            norm_out,
            conv_out,
        })
    }

    pub fn decode(&self, sample: &Tensor) -> Result<Tensor> {
        let (sample, target_shape) = self.denormalize_latents(sample)?;
        let mut h = self.conv_in.forward(&sample)?;
        h = self.mid.forward(&h)?;
        for level in (0..self.up.len()).rev() {
            let stage = &self.up[level];
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if let Some(upsample) = stage.upsample.as_ref() {
                h = upsample.forward(&h)?;
            }
        }
        h = self.norm_out.forward(&h)?;
        h = silu(&h)?;
        h = self.conv_out.forward(&h)?;
        adjust_decoded_output_shape(&h, target_shape)
    }

    fn denormalize_latents(&self, sample: &Tensor) -> Result<(Tensor, AudioLatentShape)> {
        let (batch, channels, frames, mel_bins) = sample.dims4()?;
        let latent_shape = AudioLatentShape {
            batch,
            channels,
            frames,
            mel_bins,
        };
        let sample_patched = self.patchifier.patchify(sample)?;
        let sample_denormalized = self.per_channel_statistics.denormalize(&sample_patched)?;
        let sample = self
            .patchifier
            .unpatchify(&sample_denormalized, latent_shape)?;
        let target_shape = decoder_target_shape(
            latent_shape,
            self.config.out_ch,
            self.config.mel_bins,
            self.config.causality_axis,
        );
        Ok((sample, target_shape))
    }
}

fn decoder_target_shape(
    latent_shape: AudioLatentShape,
    out_channels: usize,
    mel_bins: usize,
    causality_axis: AudioCausalityAxis,
) -> AudioLatentShape {
    let mut target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR;
    if causality_axis != AudioCausalityAxis::None {
        target_frames = target_frames
            .saturating_sub(LATENT_DOWNSAMPLE_FACTOR - 1)
            .max(1);
    }
    AudioLatentShape {
        batch: latent_shape.batch,
        channels: out_channels,
        frames: target_frames,
        mel_bins,
    }
}

fn adjust_decoded_output_shape(
    decoded_output: &Tensor,
    target_shape: AudioLatentShape,
) -> Result<Tensor> {
    let (batch, channels, current_time, current_freq) = decoded_output.dims4()?;
    let cropped = decoded_output
        .narrow(0, 0, batch.min(target_shape.batch))?
        .narrow(1, 0, channels.min(target_shape.channels))?
        .narrow(2, 0, current_time.min(target_shape.frames))?
        .narrow(3, 0, current_freq.min(target_shape.mel_bins))?;
    let (_, _, cropped_time, cropped_freq) = cropped.dims4()?;
    let padded = cropped
        .pad_with_zeros(3, 0, target_shape.mel_bins.saturating_sub(cropped_freq))?
        .pad_with_zeros(2, 0, target_shape.frames.saturating_sub(cropped_time))?
        .pad_with_zeros(
            1,
            0,
            target_shape
                .channels
                .saturating_sub(channels.min(target_shape.channels)),
        )?;
    let (_, _, padded_time, padded_freq) = padded.dims4()?;
    padded
        .narrow(1, 0, target_shape.channels)?
        .narrow(2, 0, padded_time.min(target_shape.frames))?
        .narrow(3, 0, padded_freq.min(target_shape.mel_bins))
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{
        adjust_decoded_output_shape, decoder_target_shape, AudioCausalConv2d, AudioCausalityAxis,
        AudioDownsample, AudioNormType, AudioPerChannelStatistics, Ltx2AudioDecoder,
        Ltx2AudioDecoderConfig, Ltx2AudioEncoder, Ltx2AudioEncoderConfig,
    };
    use crate::ltx2::model::AudioLatentShape;

    fn vb_from_tensors(tensors: Vec<(&str, Tensor)>) -> VarBuilder<'static> {
        let map = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect::<HashMap<_, _>>();
        VarBuilder::from_tensors(map, candle_core::DType::F32, &Device::Cpu)
    }

    #[test]
    fn audio_per_channel_statistics_denormalize_patchified_latents() {
        let device = Device::Cpu;
        let stats = AudioPerChannelStatistics {
            mean: Tensor::from_vec(vec![1f32, 2.0], 2, &device).unwrap(),
            std: Tensor::from_vec(vec![2f32, 4.0], 2, &device).unwrap(),
        };
        let x = Tensor::from_vec(vec![3f32, 5.0], (1, 1, 2), &device).unwrap();
        let denorm = stats.denormalize(&x).unwrap();
        assert_eq!(denorm.to_vec3::<f32>().unwrap(), [[[7.0, 22.0]]]);
    }

    #[test]
    fn audio_per_channel_statistics_normalize_patchified_latents() {
        let device = Device::Cpu;
        let stats = AudioPerChannelStatistics {
            mean: Tensor::from_vec(vec![1f32, 2.0], 2, &device).unwrap(),
            std: Tensor::from_vec(vec![2f32, 4.0], 2, &device).unwrap(),
        };
        let x = Tensor::from_vec(vec![7f32, 22.0], (1, 1, 2), &device).unwrap();
        let norm = stats.normalize(&x).unwrap();
        assert_eq!(norm.to_vec3::<f32>().unwrap(), [[[3.0, 5.0]]]);
    }

    #[test]
    fn causal_height_conv2d_uses_top_only_padding() {
        let device = Device::Cpu;
        let vb = vb_from_tensors(vec![
            (
                "conv.weight",
                Tensor::ones((1, 1, 3, 3), candle_core::DType::F32, &device).unwrap(),
            ),
            (
                "conv.bias",
                Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
            ),
        ]);
        let conv = AudioCausalConv2d::new(1, 1, 3, 1, AudioCausalityAxis::Height, vb).unwrap();
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 1, 3, 1), &device).unwrap();
        let y = conv.forward(&x).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 3.0, 6.0]
        );
    }

    #[test]
    fn causal_height_downsample_uses_top_only_padding() {
        let device = Device::Cpu;
        let vb = vb_from_tensors(vec![
            (
                "conv.weight",
                Tensor::ones((1, 1, 3, 3), candle_core::DType::F32, &device).unwrap(),
            ),
            (
                "conv.bias",
                Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
            ),
        ]);
        let downsample = AudioDownsample::load(1, true, AudioCausalityAxis::Height, vb).unwrap();
        let x = Tensor::from_vec(
            vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0],
            (1, 1, 3, 2),
            &device,
        )
        .unwrap();
        let y = downsample.forward(&x).unwrap();
        assert_eq!(y.dims4().unwrap(), (1, 1, 2, 1));
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![11.0, 66.0]
        );
    }

    #[test]
    fn decoder_target_shape_applies_causal_latent_trim() {
        let target = decoder_target_shape(
            AudioLatentShape {
                batch: 1,
                channels: 8,
                frames: 4,
                mel_bins: 16,
            },
            2,
            64,
            AudioCausalityAxis::Height,
        );
        assert_eq!(target.frames, 13);
        assert_eq!(target.mel_bins, 64);
        assert_eq!(target.channels, 2);
    }

    #[test]
    fn adjust_decoded_output_shape_crops_and_pads_to_target() {
        let device = Device::Cpu;
        let decoded = Tensor::from_vec(
            vec![
                1f32, 2.0, 3.0, 4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (1, 2, 2, 3),
            &device,
        )
        .unwrap();
        let adjusted = adjust_decoded_output_shape(
            &decoded,
            AudioLatentShape {
                batch: 1,
                channels: 2,
                frames: 3,
                mel_bins: 4,
            },
        )
        .unwrap();
        assert_eq!(adjusted.dims4().unwrap(), (1, 2, 3, 4));
    }

    #[test]
    fn audio_decoder_forward_respects_causal_target_shape() {
        let device = Device::Cpu;
        let config = Ltx2AudioDecoderConfig {
            ch: 2,
            out_ch: 2,
            ch_mult: vec![1],
            num_res_blocks: 1,
            attn_resolutions: vec![],
            resolution: 4,
            z_channels: 1,
            norm_type: AudioNormType::Pixel,
            causality_axis: AudioCausalityAxis::Height,
            dropout: 0.0,
            mid_block_add_attention: false,
            sample_rate: 16_000,
            mel_hop_length: 160,
            is_causal: true,
            mel_bins: 4,
        };
        let zero1 = |len| Tensor::zeros(len, DType::F32, &device).unwrap();
        let zero4 = |shape| Tensor::zeros(shape, DType::F32, &device).unwrap();
        let vb = vb_from_tensors(vec![
            ("audio_vae.per_channel_statistics.mean-of-means", zero1(2)),
            (
                "audio_vae.per_channel_statistics.std-of-means",
                Tensor::ones(2, DType::F32, &device).unwrap(),
            ),
            ("audio_vae.decoder.conv_in.conv.weight", zero4((2, 1, 3, 3))),
            ("audio_vae.decoder.conv_in.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_1.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_1.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_1.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_1.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_2.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_2.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.mid.block_2.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.mid.block_2.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.0.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.0.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.0.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.0.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.1.conv1.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.1.conv1.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.up.0.block.1.conv2.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.up.0.block.1.conv2.conv.bias", zero1(2)),
            (
                "audio_vae.decoder.conv_out.conv.weight",
                zero4((2, 2, 3, 3)),
            ),
            ("audio_vae.decoder.conv_out.conv.bias", zero1(2)),
        ]);
        let decoder = Ltx2AudioDecoder::new(config, vb).unwrap();
        let latent = Tensor::zeros((1, 1, 3, 2), candle_core::DType::F32, &device).unwrap();
        let decoded = decoder.decode(&latent).unwrap();
        assert_eq!(decoded.dims4().unwrap(), (1, 2, 9, 4));
    }

    #[test]
    fn audio_encoder_forward_emits_normalized_latent_shape() {
        let device = Device::Cpu;
        let config = Ltx2AudioEncoderConfig {
            ch: 8,
            ch_mult: vec![1],
            num_res_blocks: 1,
            attn_resolutions: vec![],
            resolution: 4,
            z_channels: 2,
            double_z: true,
            norm_type: AudioNormType::Pixel,
            causality_axis: AudioCausalityAxis::Height,
            dropout: 0.0,
            mid_block_add_attention: false,
            resamp_with_conv: true,
            in_channels: 2,
            sample_rate: 16_000,
            mel_hop_length: 160,
            n_fft: 1024,
            is_causal: true,
            mel_bins: 4,
        };
        let zero1 = |len| Tensor::zeros(len, DType::F32, &device).unwrap();
        let zero4 = |shape| Tensor::zeros(shape, DType::F32, &device).unwrap();
        let vb = vb_from_tensors(vec![
            ("audio_vae.per_channel_statistics.mean-of-means", zero1(8)),
            (
                "audio_vae.per_channel_statistics.std-of-means",
                Tensor::ones(8, DType::F32, &device).unwrap(),
            ),
            ("audio_vae.encoder.conv_in.conv.weight", zero4((8, 2, 3, 3))),
            ("audio_vae.encoder.conv_in.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.down.0.block.0.conv1.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.down.0.block.0.conv1.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.down.0.block.0.conv2.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.down.0.block.0.conv2.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_1.conv1.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_1.conv1.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_1.conv2.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_1.conv2.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_2.conv1.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_2.conv1.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.mid.block_2.conv2.conv.weight",
                zero4((8, 8, 3, 3)),
            ),
            ("audio_vae.encoder.mid.block_2.conv2.conv.bias", zero1(8)),
            (
                "audio_vae.encoder.conv_out.conv.weight",
                zero4((4, 8, 3, 3)),
            ),
            ("audio_vae.encoder.conv_out.conv.bias", zero1(4)),
        ]);
        let encoder = Ltx2AudioEncoder::new(config, vb).unwrap();
        let spectrogram = Tensor::zeros((1, 2, 9, 4), candle_core::DType::F32, &device).unwrap();
        let encoded = encoder.encode(&spectrogram).unwrap();
        assert_eq!(encoded.dims4().unwrap(), (1, 2, 9, 4));
    }
}
