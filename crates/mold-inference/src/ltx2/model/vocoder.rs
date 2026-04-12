#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};
use serde::Deserialize;

use super::audio_vae::read_checkpoint_config_json;

#[derive(Debug, Clone, Deserialize)]
pub struct Ltx2VocoderConfig {
    pub vocoder: Ltx2GeneratorConfig,
    pub bwe: Option<Ltx2BweConfig>,
    pub output_sample_rate: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ltx2GeneratorConfig {
    pub upsample_initial_channel: usize,
    pub resblock: String,
    pub upsample_rates: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub stereo: bool,
    #[serde(default = "default_use_tanh_at_final")]
    pub use_tanh_at_final: bool,
    #[serde(default = "default_activation")]
    pub activation: String,
    #[serde(default)]
    pub use_bias_at_final: bool,
    #[serde(default = "default_apply_final_activation")]
    pub apply_final_activation: bool,
    #[serde(default = "default_output_sampling_rate")]
    pub output_sampling_rate: usize,
}

fn default_use_tanh_at_final() -> bool {
    true
}

fn default_apply_final_activation() -> bool {
    true
}

fn default_activation() -> String {
    "snake".to_string()
}

fn default_output_sampling_rate() -> usize {
    24_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ltx2BweConfig {
    #[serde(flatten)]
    pub generator: Ltx2GeneratorConfig,
    pub input_sampling_rate: usize,
    pub output_sampling_rate: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub win_size: usize,
    pub num_mels: usize,
}

#[derive(Debug, Deserialize)]
struct CheckpointConfig {
    vocoder: CheckpointVocoderLayout,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CheckpointVocoderLayout {
    Legacy(Ltx2GeneratorConfig),
    Nested {
        vocoder: Ltx2GeneratorConfig,
        bwe: Ltx2BweConfig,
    },
}

impl Ltx2VocoderConfig {
    pub fn load(checkpoint_path: &Path) -> Result<Self> {
        let config_json = read_checkpoint_config_json(checkpoint_path)?;
        let checkpoint: CheckpointConfig =
            serde_json::from_str(&config_json).with_context(|| {
                format!(
                    "failed to parse LTX-2 vocoder config metadata from {}",
                    checkpoint_path.display()
                )
            })?;
        match checkpoint.vocoder {
            CheckpointVocoderLayout::Legacy(vocoder) => Ok(Self {
                output_sample_rate: vocoder.output_sampling_rate,
                vocoder,
                bwe: None,
            }),
            CheckpointVocoderLayout::Nested { vocoder, bwe } => Ok(Self {
                output_sample_rate: bwe.output_sampling_rate,
                vocoder,
                bwe: Some(bwe),
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get(channels, "alpha")?,
            beta: vb.get(channels, "beta")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let channels = self.alpha.dims1()?;
        let alpha = self
            .alpha
            .reshape((1, channels, 1))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .exp()?;
        let beta = self
            .beta
            .reshape((1, channels, 1))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .exp()?;
        let sin_sq = x.broadcast_mul(&alpha)?.sin()?.sqr()?;
        x.broadcast_add(&sin_sq.broadcast_div(&beta)?)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct CheckpointUpSample1d {
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl CheckpointUpSample1d {
    fn load(filter: Tensor, ratio: usize) -> Result<Self> {
        let kernel_size = filter.dims3()?.2;
        let pad = kernel_size / ratio - 1;
        let pad_left = pad * ratio + (kernel_size - ratio) / 2;
        let pad_right = pad * ratio + (kernel_size - ratio + 1) / 2;
        Ok(Self {
            ratio,
            pad,
            pad_left,
            pad_right,
            filter,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _) = x.dims3()?;
        let x = replicate_pad_1d(x, self.pad, self.pad)?;
        let filter = self
            .filter
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .broadcast_as((channels, 1, self.filter.dims3()?.2))?
            .contiguous()?;
        let y = x.conv_transpose1d(&filter, 0, 0, self.ratio, 1, channels)?;
        let y = y.affine(self.ratio as f64, 0.0)?;
        let out_len = y.dims3()?.2;
        y.narrow(
            2,
            self.pad_left,
            out_len.saturating_sub(self.pad_left + self.pad_right),
        )
        .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct CheckpointDownSample1d {
    ratio: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl CheckpointDownSample1d {
    fn load(filter: Tensor, ratio: usize) -> Result<Self> {
        let kernel_size = filter.dims3()?.2;
        let even = kernel_size % 2 == 0;
        let pad_left = kernel_size / 2 - usize::from(even);
        let pad_right = kernel_size / 2;
        Ok(Self {
            ratio,
            pad_left,
            pad_right,
            filter,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _) = x.dims3()?;
        let x = replicate_pad_1d(x, self.pad_left, self.pad_right)?;
        let filter = self
            .filter
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .broadcast_as((channels, 1, self.filter.dims3()?.2))?
            .contiguous()?;
        x.conv1d(&filter, 0, self.ratio, 1, channels)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct Activation1d {
    act: SnakeBeta,
    upsample: CheckpointUpSample1d,
    downsample: CheckpointDownSample1d,
}

impl Activation1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let upsample_filter = vb.pp("upsample").get((1, 1, 12), "filter")?;
        let downsample_filter = vb
            .pp("downsample")
            .pp("lowpass")
            .get((1, 1, 12), "filter")?;
        Ok(Self {
            act: SnakeBeta::load(channels, vb.pp("act"))?,
            upsample: CheckpointUpSample1d::load(upsample_filter, 2)?,
            downsample: CheckpointDownSample1d::load(downsample_filter, 2)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.upsample.forward(x)?;
        let x = self.act.forward(&x)?;
        self.downsample.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct AmpBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    acts1: Vec<Activation1d>,
    acts2: Vec<Activation1d>,
}

impl AmpBlock1 {
    fn load(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut acts1 = Vec::with_capacity(3);
        let mut acts2 = Vec::with_capacity(3);
        for (idx, dilation) in dilations.iter().enumerate() {
            let padding = get_padding(kernel_size, *dilation);
            convs1.push(load_conv1d(
                channels,
                channels,
                kernel_size,
                padding,
                *dilation,
                vb.pp(format!("convs1.{idx}")),
            )?);
            convs2.push(load_conv1d(
                channels,
                channels,
                kernel_size,
                get_padding(kernel_size, 1),
                1,
                vb.pp(format!("convs2.{idx}")),
            )?);
            acts1.push(Activation1d::load(channels, vb.pp(format!("acts1.{idx}")))?);
            acts2.push(Activation1d::load(channels, vb.pp(format!("acts2.{idx}")))?);
        }
        Ok(Self {
            convs1,
            convs2,
            acts1,
            acts2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for idx in 0..self.convs1.len() {
            let mut xt = self.acts1[idx].forward(&x)?;
            xt = xt.apply(&self.convs1[idx])?;
            xt = self.acts2[idx].forward(&xt)?;
            xt = xt.apply(&self.convs2[idx])?;
            x = x.add(&xt)?;
        }
        Ok(x)
    }
}

const LRELU_SLOPE: f64 = 0.1;

fn leaky_relu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::Activation::LeakyRelu(LRELU_SLOPE).forward(x)?)
}

#[derive(Debug, Clone)]
struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn load(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        for (idx, dilation) in dilations.iter().copied().enumerate() {
            convs1.push(load_conv1d(
                channels,
                channels,
                kernel_size,
                get_padding(kernel_size, dilation),
                dilation,
                vb.pp(format!("convs1.{idx}")),
            )?);
            convs2.push(load_conv1d(
                channels,
                channels,
                kernel_size,
                get_padding(kernel_size, 1),
                1,
                vb.pp(format!("convs2.{idx}")),
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for idx in 0..self.convs1.len() {
            let mut xt = leaky_relu(&x)?;
            xt = xt.apply(&self.convs1[idx])?;
            xt = leaky_relu(&xt)?;
            xt = xt.apply(&self.convs2[idx])?;
            x = x.add(&xt)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
enum VocoderResBlock {
    Plain(ResBlock1),
    Amp(AmpBlock1),
}

impl VocoderResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(block) => block.forward(x),
            Self::Amp(block) => block.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
enum VocoderPostActivation {
    LeakyRelu,
    AntiAliased(Activation1d),
}

impl VocoderPostActivation {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::LeakyRelu => leaky_relu(x),
            Self::AntiAliased(activation) => activation.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
struct BigVganGenerator {
    config: Ltx2GeneratorConfig,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<VocoderResBlock>,
    act_post: VocoderPostActivation,
    conv_post: Conv1d,
    num_kernels: usize,
}

impl BigVganGenerator {
    fn load(config: Ltx2GeneratorConfig, vb: VarBuilder) -> Result<Self> {
        if !config.stereo {
            bail!("native LTX-2 vocoder currently supports stereo checkpoints only");
        }
        let uses_amp = match config.resblock.as_str() {
            "1" => false,
            "AMP1" => true,
            other => bail!("native LTX-2 vocoder does not support resblock='{other}'"),
        };
        if uses_amp && config.activation != "snakebeta" {
            bail!(
                "native LTX-2 AMP vocoder currently supports activation='snakebeta', found '{}'",
                config.activation
            );
        }

        let num_kernels = config.resblock_kernel_sizes.len();
        let conv_pre = load_conv1d(
            128,
            config.upsample_initial_channel,
            7,
            3,
            1,
            vb.pp("conv_pre"),
        )?;
        let mut ups = Vec::with_capacity(config.upsample_rates.len());
        let mut resblocks = Vec::with_capacity(config.upsample_rates.len() * num_kernels);
        for (idx, (stride, kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_channels = config.upsample_initial_channel / (1usize << idx);
            let out_channels = config.upsample_initial_channel / (1usize << (idx + 1));
            ups.push(load_conv_transpose1d(
                in_channels,
                out_channels,
                *kernel_size,
                (kernel_size - stride) / 2,
                *stride,
                vb.pp(format!("ups.{idx}")),
            )?);
            for (block_idx, (kernel_size, dilations)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let block_vb = vb.pp(format!("resblocks.{}", idx * num_kernels + block_idx));
                resblocks.push(if uses_amp {
                    VocoderResBlock::Amp(AmpBlock1::load(
                        out_channels,
                        *kernel_size,
                        dilations,
                        block_vb,
                    )?)
                } else {
                    VocoderResBlock::Plain(ResBlock1::load(
                        out_channels,
                        *kernel_size,
                        dilations,
                        block_vb,
                    )?)
                });
            }
        }
        let final_channels =
            config.upsample_initial_channel / (1usize << config.upsample_rates.len());
        let act_post = if uses_amp {
            VocoderPostActivation::AntiAliased(Activation1d::load(
                final_channels,
                vb.pp("act_post"),
            )?)
        } else {
            VocoderPostActivation::LeakyRelu
        };
        let conv_post = load_conv1d_optional_bias(
            final_channels,
            2,
            7,
            3,
            1,
            config.use_bias_at_final,
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            config,
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
            num_kernels,
        })
    }

    fn forward(&self, mel_spec: &Tensor) -> Result<Tensor> {
        let mel_spec = mel_spec.to_dtype(DType::F32)?;
        let (batch, channels, time, mel_bins) = mel_spec.dims4()?;
        if channels != 2 {
            bail!("native LTX-2 vocoder expects stereo mel inputs, got {channels} channels");
        }
        let mut x = mel_spec
            .transpose(2, 3)?
            .reshape((batch, channels * mel_bins, time))?;
        x = x.apply(&self.conv_pre)?;
        for (up_idx, up) in self.ups.iter().enumerate() {
            if self.config.resblock == "1" {
                x = leaky_relu(&x)?;
            }
            x = x.apply(up)?;
            let start = up_idx * self.num_kernels;
            let end = start + self.num_kernels;
            let mut acc: Option<Tensor> = None;
            for resblock in &self.resblocks[start..end] {
                let out = resblock.forward(&x)?;
                acc = Some(match acc {
                    Some(prev) => prev.add(&out)?,
                    None => out,
                });
            }
            x = acc
                .context("vocoder stage did not emit any residual block outputs")?
                .affine(1.0 / self.num_kernels as f64, 0.0)?;
        }
        x = self.act_post.forward(&x)?;
        x = x.apply(&self.conv_post)?;
        if self.config.apply_final_activation {
            if self.config.use_tanh_at_final {
                x = x.tanh()?;
            } else {
                x = x.clamp(-1f32, 1f32)?;
            }
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct StftFn {
    forward_basis: Tensor,
    inverse_basis: Tensor,
    hop_length: usize,
    win_length: usize,
}

impl StftFn {
    fn load(
        filter_length: usize,
        hop_length: usize,
        win_length: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            forward_basis: vb.get(
                (((filter_length / 2) + 1) * 2, 1, filter_length),
                "forward_basis",
            )?,
            inverse_basis: vb.get(
                (((filter_length / 2) + 1) * 2, 1, filter_length),
                "inverse_basis",
            )?,
            hop_length,
            win_length,
        })
    }

    fn magnitude_spectrogram(&self, y: &Tensor) -> Result<Tensor> {
        let y = match y.rank() {
            2 => y.unsqueeze(1)?,
            3 => y.clone(),
            rank => bail!("expected [B, T] or [B, 1, T] waveform input, got rank {rank}"),
        };
        let left_pad = self.win_length.saturating_sub(self.hop_length);
        let y = y.pad_with_zeros(2, left_pad, 0)?;
        let basis = self
            .forward_basis
            .to_device(y.device())?
            .to_dtype(y.dtype())?;
        let spec = y.conv1d(&basis, 0, self.hop_length, 1, 1)?;
        let n_freqs = spec.dims3()?.1 / 2;
        let real = spec.narrow(1, 0, n_freqs)?;
        let imag = spec.narrow(1, n_freqs, n_freqs)?;
        real.sqr()?.add(&imag.sqr()?)?.sqrt().map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct MelStft {
    mel_basis: Tensor,
    stft_fn: StftFn,
}

impl MelStft {
    fn load(
        filter_length: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            mel_basis: vb.get((n_mels, filter_length / 2 + 1), "mel_basis")?,
            stft_fn: StftFn::load(filter_length, hop_length, win_length, vb.pp("stft_fn"))?,
        })
    }

    fn compute_log_mel(&self, audio: &Tensor) -> Result<Tensor> {
        let (batch, channels, samples) = audio.dims3()?;
        let flat = audio.reshape((batch * channels, samples))?;
        let magnitude = self.stft_fn.magnitude_spectrogram(&flat)?;
        let (_, n_freqs, frames) = magnitude.dims3()?;
        let mel_basis = self
            .mel_basis
            .to_device(magnitude.device())?
            .to_dtype(magnitude.dtype())?
            .reshape((1, self.mel_basis.dims2()?.0, n_freqs))?;
        let mel = mel_basis.broadcast_matmul(&magnitude)?;
        let log_mel = mel.clamp(1e-5f32, f32::MAX)?.log()?;
        log_mel
            .reshape((batch, channels, self.mel_basis.dims2()?.0, frames))
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct HannResampler {
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl HannResampler {
    fn new(ratio: usize, device: &candle_core::Device) -> Result<Self> {
        let rolloff = 0.99f64;
        let lowpass_filter_width = 6f64;
        let width = (lowpass_filter_width / rolloff).ceil() as usize;
        let kernel_size = 2 * width * ratio + 1;
        let pad = width;
        let pad_left = 2 * width * ratio;
        let pad_right = kernel_size - ratio;
        let mut filter = Vec::with_capacity(kernel_size);
        for idx in 0..kernel_size {
            let time_axis = (idx as f64 / ratio as f64 - width as f64) * rolloff;
            let time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width);
            let window = (time_clamped * PI / lowpass_filter_width / 2.0)
                .cos()
                .powi(2);
            filter.push((sinc(time_axis) * window * rolloff / ratio as f64) as f32);
        }
        Ok(Self {
            ratio,
            pad,
            pad_left,
            pad_right,
            filter: Tensor::from_vec(filter, (1, 1, kernel_size), device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _) = x.dims3()?;
        let x = replicate_pad_1d(x, self.pad, self.pad)?;
        let filter = self
            .filter
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .broadcast_as((channels, 1, self.filter.dims3()?.2))?
            .contiguous()?;
        let y = x.conv_transpose1d(&filter, 0, 0, self.ratio, 1, channels)?;
        let y = y.affine(self.ratio as f64, 0.0)?;
        let out_len = y.dims3()?.2;
        y.narrow(
            2,
            self.pad_left,
            out_len.saturating_sub(self.pad_left + self.pad_right),
        )
        .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct Ltx2VocoderWithBwe {
    pub config: Ltx2VocoderConfig,
    vocoder: BigVganGenerator,
    bwe_generator: Option<BigVganGenerator>,
    mel_stft: Option<MelStft>,
    resampler: Option<HannResampler>,
}

impl Ltx2VocoderWithBwe {
    pub fn load_from_checkpoint(
        checkpoint_path: &Path,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let config = Ltx2VocoderConfig::load(checkpoint_path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[PathBuf::from(checkpoint_path)],
                DType::F32,
                device,
            )
        }
        .with_context(|| format!("failed to mmap {}", checkpoint_path.display()))?;
        Self::new(config, vb, device)
    }

    fn new(
        config: Ltx2VocoderConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let root_vb = vb.pp("vocoder");
        let (vocoder, bwe_generator, mel_stft, resampler) = if let Some(bwe) = config.bwe.as_ref() {
            let vocoder = BigVganGenerator::load(config.vocoder.clone(), root_vb.pp("vocoder"))?;
            let bwe_generator =
                BigVganGenerator::load(bwe.generator.clone(), root_vb.pp("bwe_generator"))?;
            let mel_stft = MelStft::load(
                bwe.n_fft,
                bwe.hop_length,
                bwe.win_size,
                bwe.num_mels,
                root_vb.pp("mel_stft"),
            )?;
            let ratio = bwe.output_sampling_rate / bwe.input_sampling_rate;
            if ratio * bwe.input_sampling_rate != bwe.output_sampling_rate {
                bail!("native LTX-2 BWE resampler requires an integer sample-rate ratio");
            }
            (
                vocoder,
                Some(bwe_generator),
                Some(mel_stft),
                Some(HannResampler::new(ratio, device)?),
            )
        } else {
            (
                BigVganGenerator::load(config.vocoder.clone(), root_vb)?,
                None,
                None,
                None,
            )
        };
        Ok(Self {
            config,
            vocoder,
            bwe_generator,
            mel_stft,
            resampler,
        })
    }

    pub fn forward(&self, mel_spec: &Tensor) -> Result<Tensor> {
        let mut x = self.vocoder.forward(mel_spec)?;
        let Some(bwe) = self.config.bwe.as_ref() else {
            return Ok(x);
        };
        let length_low_rate = x.dims3()?.2;
        let output_length = length_low_rate * bwe.output_sampling_rate / bwe.input_sampling_rate;
        let remainder = length_low_rate % bwe.hop_length;
        if remainder != 0 {
            x = x.pad_with_zeros(2, 0, bwe.hop_length - remainder)?;
        }
        let mel = self
            .mel_stft
            .as_ref()
            .context("native LTX-2 BWE checkpoint is missing mel STFT state")?
            .compute_log_mel(&x)?;
        let mel_for_bwe = mel.transpose(2, 3)?;
        let residual = self
            .bwe_generator
            .as_ref()
            .context("native LTX-2 BWE checkpoint is missing BWE generator weights")?
            .forward(&mel_for_bwe)?;
        let skip = self
            .resampler
            .as_ref()
            .context("native LTX-2 BWE checkpoint is missing the BWE resampler")?
            .forward(&x)?;
        let residual_len = residual.dims3()?.2;
        let skip_len = skip.dims3()?.2;
        if residual_len != skip_len {
            bail!("native LTX-2 vocoder residual/skip length mismatch: residual={residual_len}, skip={skip_len}");
        }
        let mixed = residual.add(&skip)?;
        mixed
            .clamp(-1f32, 1f32)?
            .narrow(2, 0, residual_len.min(output_length))
            .map_err(Into::into)
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn replicate_pad_1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let (_, _, len) = x.dims3()?;
    let mut parts = Vec::new();
    if left != 0 {
        let first = x.narrow(2, 0, 1)?.repeat((1, 1, left))?;
        parts.push(first);
    }
    parts.push(x.clone());
    if right != 0 {
        let last = x
            .narrow(2, len.saturating_sub(1), 1)?
            .repeat((1, 1, right))?;
        parts.push(last);
    }
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 2).map_err(Into::into)
}

fn load_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    load_conv1d_optional_bias(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        dilation,
        true,
        vb,
    )
}

fn load_conv1d_optional_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    use_bias: bool,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = if use_bias {
        vb.get(out_channels, "bias").ok()
    } else {
        None
    };
    Ok(Conv1d::new(
        weight,
        bias,
        Conv1dConfig {
            padding,
            dilation,
            ..Default::default()
        },
    ))
}

fn load_conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight = vb.get((in_channels, out_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias").ok();
    Ok(ConvTranspose1d::new(
        weight,
        bias,
        ConvTranspose1dConfig {
            padding,
            stride,
            ..Default::default()
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{
        default_apply_final_activation, BigVganGenerator, HannResampler, Ltx2BweConfig,
        Ltx2GeneratorConfig, Ltx2VocoderConfig, Ltx2VocoderWithBwe, MelStft,
    };

    fn vb_from_tensors(tensors: Vec<(&str, Tensor)>) -> VarBuilder<'static> {
        let map = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect::<HashMap<_, _>>();
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    fn unit_filter(device: &Device, size: usize) -> Tensor {
        let mut values = vec![0f32; size];
        values[size / 2] = 1.0;
        Tensor::from_vec(values, (1, 1, size), device).unwrap()
    }

    fn tiny_generator_config() -> Ltx2GeneratorConfig {
        Ltx2GeneratorConfig {
            upsample_initial_channel: 8,
            resblock: "AMP1".to_string(),
            upsample_rates: vec![2],
            resblock_kernel_sizes: vec![3],
            upsample_kernel_sizes: vec![4],
            resblock_dilation_sizes: vec![vec![1, 1, 1]],
            stereo: true,
            use_tanh_at_final: false,
            activation: "snakebeta".to_string(),
            use_bias_at_final: true,
            apply_final_activation: false,
            output_sampling_rate: 16_000,
        }
    }

    fn tiny_plain_generator_config() -> Ltx2GeneratorConfig {
        Ltx2GeneratorConfig {
            upsample_initial_channel: 8,
            resblock: "1".to_string(),
            upsample_rates: vec![2],
            resblock_kernel_sizes: vec![3],
            upsample_kernel_sizes: vec![4],
            resblock_dilation_sizes: vec![vec![1, 1, 1]],
            stereo: true,
            use_tanh_at_final: true,
            activation: "snake".to_string(),
            use_bias_at_final: true,
            apply_final_activation: true,
            output_sampling_rate: 24_000,
        }
    }

    fn tiny_generator_tensors(
        device: &Device,
        prefix: &str,
        upsample_initial_channel: usize,
        post_channels: usize,
        upsample_kernel_size: usize,
    ) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        let zero1 = |len| Tensor::zeros(len, DType::F32, device).unwrap();
        let zero3 = |shape| Tensor::zeros(shape, DType::F32, device).unwrap();
        let out_channels = upsample_initial_channel / 2;
        tensors.insert(
            format!("{prefix}.conv_pre.weight"),
            zero3((upsample_initial_channel, 128, 7)),
        );
        tensors.insert(
            format!("{prefix}.conv_pre.bias"),
            zero1(upsample_initial_channel),
        );
        tensors.insert(
            format!("{prefix}.ups.0.weight"),
            zero3((upsample_initial_channel, out_channels, upsample_kernel_size)),
        );
        tensors.insert(format!("{prefix}.ups.0.bias"), zero1(out_channels));
        for conv_set in ["convs1", "convs2"] {
            for idx in 0..3 {
                tensors.insert(
                    format!("{prefix}.resblocks.0.{conv_set}.{idx}.weight"),
                    zero3((out_channels, out_channels, 3)),
                );
                tensors.insert(
                    format!("{prefix}.resblocks.0.{conv_set}.{idx}.bias"),
                    zero1(out_channels),
                );
            }
        }
        for act_set in ["acts1", "acts2"] {
            for idx in 0..3 {
                tensors.insert(
                    format!("{prefix}.resblocks.0.{act_set}.{idx}.act.alpha"),
                    zero1(out_channels),
                );
                tensors.insert(
                    format!("{prefix}.resblocks.0.{act_set}.{idx}.act.beta"),
                    zero1(out_channels),
                );
                tensors.insert(
                    format!("{prefix}.resblocks.0.{act_set}.{idx}.upsample.filter"),
                    unit_filter(device, 12),
                );
                tensors.insert(
                    format!("{prefix}.resblocks.0.{act_set}.{idx}.downsample.lowpass.filter"),
                    unit_filter(device, 12),
                );
            }
        }
        tensors.insert(format!("{prefix}.act_post.act.alpha"), zero1(post_channels));
        tensors.insert(format!("{prefix}.act_post.act.beta"), zero1(post_channels));
        tensors.insert(
            format!("{prefix}.act_post.upsample.filter"),
            unit_filter(device, 12),
        );
        tensors.insert(
            format!("{prefix}.act_post.downsample.lowpass.filter"),
            unit_filter(device, 12),
        );
        tensors.insert(
            format!("{prefix}.conv_post.weight"),
            zero3((2, post_channels, 7)),
        );
        tensors.insert(format!("{prefix}.conv_post.bias"), zero1(2));

        tensors
    }

    #[test]
    fn stft_mel_projection_emits_expected_shape() {
        let device = Device::Cpu;
        let vb = vb_from_tensors(vec![
            (
                "mel_basis",
                Tensor::ones((3, 3), DType::F32, &device).unwrap(),
            ),
            (
                "stft_fn.forward_basis",
                Tensor::ones((6, 1, 4), DType::F32, &device).unwrap(),
            ),
            (
                "stft_fn.inverse_basis",
                Tensor::zeros((6, 1, 4), DType::F32, &device).unwrap(),
            ),
        ]);
        let mel = MelStft::load(4, 2, 4, 3, vb).unwrap();
        let audio = Tensor::zeros((2, 2, 8), DType::F32, &device).unwrap();
        let out = mel.compute_log_mel(&audio).unwrap();
        assert_eq!(out.dims4().unwrap(), (2, 2, 3, 4));
    }

    #[test]
    fn generator_forward_emits_stereo_waveform_shape() {
        let device = Device::Cpu;
        let config = tiny_generator_config();
        let vb = VarBuilder::from_tensors(
            tiny_generator_tensors(&device, "generator", 8, 4, 4),
            DType::F32,
            &device,
        );
        let generator = BigVganGenerator::load(config, vb.pp("generator")).unwrap();
        let mel = Tensor::zeros((1, 2, 4, 64), DType::F32, &device).unwrap();
        let waveform = generator.forward(&mel).unwrap();
        assert_eq!(waveform.dims3().unwrap(), (1, 2, 8));
    }

    #[test]
    fn legacy_generator_forward_emits_stereo_waveform_shape() {
        let device = Device::Cpu;
        let config = tiny_plain_generator_config();
        let vb = VarBuilder::from_tensors(
            tiny_generator_tensors(&device, "generator", 8, 4, 4),
            DType::F32,
            &device,
        );
        let generator = BigVganGenerator::load(config, vb.pp("generator")).unwrap();
        let mel = Tensor::zeros((1, 2, 4, 64), DType::F32, &device).unwrap();
        let waveform = generator.forward(&mel).unwrap();
        assert_eq!(waveform.dims3().unwrap(), (1, 2, 8));
    }

    #[test]
    fn vocoder_with_bwe_upsamples_output_length() {
        let device = Device::Cpu;
        let config = Ltx2VocoderConfig {
            vocoder: tiny_generator_config(),
            bwe: Some(Ltx2BweConfig {
                generator: Ltx2GeneratorConfig {
                    upsample_initial_channel: 4,
                    upsample_rates: vec![3],
                    upsample_kernel_sizes: vec![3],
                    resblock_kernel_sizes: vec![3],
                    resblock_dilation_sizes: vec![vec![1, 1, 1]],
                    stereo: true,
                    use_tanh_at_final: false,
                    activation: "snakebeta".to_string(),
                    use_bias_at_final: true,
                    apply_final_activation: false,
                    resblock: "AMP1".to_string(),
                    output_sampling_rate: 48_000,
                },
                input_sampling_rate: 16_000,
                output_sampling_rate: 48_000,
                hop_length: 1,
                n_fft: 1,
                win_size: 1,
                num_mels: 64,
            }),
            output_sample_rate: 48_000,
        };
        let mut tensors = tiny_generator_tensors(&device, "vocoder.vocoder", 8, 4, 4);
        tensors.extend(tiny_generator_tensors(
            &device,
            "vocoder.bwe_generator",
            4,
            2,
            3,
        ));
        tensors.insert(
            "vocoder.mel_stft.mel_basis".to_string(),
            Tensor::ones((64, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "vocoder.mel_stft.stft_fn.forward_basis".to_string(),
            Tensor::ones((2, 1, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "vocoder.mel_stft.stft_fn.inverse_basis".to_string(),
            Tensor::zeros((2, 1, 1), DType::F32, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let vocoder = Ltx2VocoderWithBwe::new(config, vb, &device).unwrap();
        let mel = Tensor::zeros((1, 2, 4, 64), DType::F32, &device).unwrap();
        let waveform = vocoder.forward(&mel).unwrap();
        assert_eq!(waveform.dims3().unwrap(), (1, 2, 24));
    }

    #[test]
    fn legacy_vocoder_keeps_generator_output_length() {
        let device = Device::Cpu;
        let config = Ltx2VocoderConfig {
            vocoder: tiny_plain_generator_config(),
            bwe: None,
            output_sample_rate: 24_000,
        };
        let vb = VarBuilder::from_tensors(
            tiny_generator_tensors(&device, "vocoder", 8, 4, 4),
            DType::F32,
            &device,
        );
        let vocoder = Ltx2VocoderWithBwe::new(config, vb, &device).unwrap();
        let mel = Tensor::zeros((1, 2, 4, 64), DType::F32, &device).unwrap();
        let waveform = vocoder.forward(&mel).unwrap();
        assert_eq!(waveform.dims3().unwrap(), (1, 2, 8));
        assert_eq!(vocoder.config.output_sample_rate, 24_000);
    }

    #[test]
    fn hann_resampler_ratio_three_expands_length() {
        let device = Device::Cpu;
        let resampler = HannResampler::new(3, &device).unwrap();
        let x = Tensor::zeros((1, 2, 8), DType::F32, &device).unwrap();
        let y = resampler.forward(&x).unwrap();
        assert_eq!(y.dims3().unwrap(), (1, 2, 24));
    }

    #[test]
    fn generator_default_apply_final_activation_is_true() {
        assert!(default_apply_final_activation());
    }
}
