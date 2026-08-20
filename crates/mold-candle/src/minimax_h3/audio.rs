//! MiniMax H3 DAC/BigVGAN audio autoencoder primitives.
//!
//! This module is intentionally a model primitive, not an inference engine.
//! It performs no downloads and is not reachable from Mold's model factory.

use std::f64::consts::PI;

use candle::{bail, DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, VarBuilder,
};

use super::audio_config::{AudioVaeConfig, SAMPLE_RATE};
use super::audio_weights::AudioTensorLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioVaePhase {
    EncodePreprocess,
    EncodeDac,
    EncodeProjection,
    EncodePosterior,
    DecodeProjection,
    DecodeBigVgan,
    DecodeOutput,
}

pub trait AudioVaeCancellation: Send + Sync {
    fn is_cancelled(&self, phase: AudioVaePhase) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeverCancel;

impl AudioVaeCancellation for NeverCancel {
    fn is_cancelled(&self, _phase: AudioVaePhase) -> bool {
        false
    }
}

fn cancellation_boundary(
    cancellation: &dyn AudioVaeCancellation,
    phase: AudioVaePhase,
) -> Result<()> {
    if cancellation.is_cancelled(phase) {
        bail!("MiniMax H3 audio VAE cancelled at {phase:?}")
    }
    Ok(())
}

/// Association survives the audio boundary without introducing pipeline or
/// packed-reference policy into this model crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioSoundtrackAssociation {
    Generated,
    StandaloneReference { reference_index: usize },
    VideoSoundtrack { reference_index: usize },
}

/// Typed `[batch, stereo=2, samples]` FP32 waveform.
#[derive(Clone, Debug)]
pub struct StereoWaveform {
    samples: Tensor,
    association: AudioSoundtrackAssociation,
}

impl StereoWaveform {
    pub fn new(
        samples: Tensor,
        sample_rate: usize,
        association: AudioSoundtrackAssociation,
    ) -> Result<Self> {
        require_f32(&samples, "stereo waveform")?;
        let (_batch, channels, samples_len) = samples.dims3()?;
        if channels != 2 || samples_len == 0 {
            bail!(
                "MiniMax H3 stereo waveform must have shape [batch, 2, samples>0], got {:?}",
                samples.dims()
            )
        }
        if sample_rate != SAMPLE_RATE {
            bail!("MiniMax H3 audio waveform must be {SAMPLE_RATE} Hz, got {sample_rate} Hz")
        }
        Ok(Self {
            samples,
            association,
        })
    }

    #[doc(hidden)]
    pub(super) fn new_for_config(
        samples: Tensor,
        _config: &AudioVaeConfig,
        association: AudioSoundtrackAssociation,
    ) -> Result<Self> {
        require_f32(&samples, "stereo waveform")?;
        let (_batch, channels, samples_len) = samples.dims3()?;
        if channels != 2 || samples_len == 0 {
            bail!("stereo waveform must have shape [batch, 2, samples>0]")
        }
        Ok(Self {
            samples,
            association,
        })
    }

    pub fn samples(&self) -> &Tensor {
        &self.samples
    }

    pub fn association(&self) -> AudioSoundtrackAssociation {
        self.association
    }

    /// Channel-major batch packing used by all three pinned implementations:
    /// `[B, 2, L] -> [B*2, 1, L]` with `B0-L, B0-R, B1-L, B1-R` order.
    pub fn pack_mono_batch(&self) -> Result<Tensor> {
        let (batch, channels, samples) = self.samples.dims3()?;
        self.samples.reshape((batch * channels, 1, samples))
    }

    pub fn from_mono_batch(
        mono: Tensor,
        batch: usize,
        sample_rate: usize,
        association: AudioSoundtrackAssociation,
    ) -> Result<Self> {
        require_f32(&mono, "mono batch")?;
        let (mono_batch, channels, samples) = mono.dims3()?;
        if channels != 1 || mono_batch != batch * 2 || samples == 0 {
            bail!(
                "MiniMax H3 mono batch must have shape [{}, 1, samples>0], got {:?}",
                batch * 2,
                mono.dims()
            )
        }
        Self::new(mono.reshape((batch, 2, samples))?, sample_rate, association)
    }

    fn from_mono_batch_for_config(
        mono: Tensor,
        batch: usize,
        config: &AudioVaeConfig,
        association: AudioSoundtrackAssociation,
    ) -> Result<Self> {
        require_f32(&mono, "mono batch")?;
        let (mono_batch, channels, samples) = mono.dims3()?;
        if channels != 1 || mono_batch != batch * 2 || samples == 0 {
            bail!("mono batch does not match the MiniMax H3 stereo contract")
        }
        Self::new_for_config(mono.reshape((batch, 2, samples))?, config, association)
    }
}

/// Typed normalized `[batch, latent_channels, stereo=2, rows]` tensor.
#[derive(Clone, Debug)]
pub struct StereoLatents {
    normalized: Tensor,
}

impl StereoLatents {
    pub fn new(normalized: Tensor, config: &AudioVaeConfig) -> Result<Self> {
        require_f32(&normalized, "normalized stereo audio latents")?;
        let (_batch, channels, stereo, rows) = normalized.dims4()?;
        if channels != config.latent_channels || stereo != 2 || rows == 0 {
            bail!(
                "MiniMax H3 stereo latents must have shape [batch, {}, 2, rows>0], got {:?}",
                config.latent_channels,
                normalized.dims()
            )
        }
        Ok(Self { normalized })
    }

    pub fn normalized(&self) -> &Tensor {
        &self.normalized
    }

    pub fn pack_mono_batch(&self) -> Result<Tensor> {
        let (batch, channels, stereo, rows) = self.normalized.dims4()?;
        self.normalized
            .permute((0, 2, 1, 3))?
            .reshape((batch * stereo, channels, rows))
    }

    pub fn from_mono_batch(packed: Tensor, batch: usize, config: &AudioVaeConfig) -> Result<Self> {
        require_f32(&packed, "packed normalized audio latents")?;
        let (mono_batch, channels, rows) = packed.dims3()?;
        if mono_batch != batch * 2 || channels != config.latent_channels || rows == 0 {
            bail!("packed audio latents do not match the stereo/config contract")
        }
        let normalized = packed
            .reshape((batch, 2, channels, rows))?
            .permute((0, 2, 1, 3))?;
        Self::new(normalized, config)
    }
}

#[derive(Clone, Debug)]
pub struct AudioPosterior {
    pub mean: Tensor,
    pub logs: Tensor,
}

pub enum AudioPosteriorSelection<'a> {
    Mode,
    Sample { noise: &'a Tensor },
}

impl AudioPosterior {
    pub fn new(mean: Tensor, logs: Tensor) -> Result<Self> {
        require_f32(&mean, "audio posterior mean")?;
        require_f32(&logs, "audio posterior log standard deviation")?;
        if mean.dims() != logs.dims() {
            bail!("MiniMax H3 audio posterior mean/logs shapes differ")
        }
        Ok(Self { mean, logs })
    }

    pub fn mode(&self) -> Tensor {
        self.mean.clone()
    }

    pub fn select(&self, selection: AudioPosteriorSelection<'_>) -> Result<Tensor> {
        match selection {
            AudioPosteriorSelection::Mode => Ok(self.mode()),
            AudioPosteriorSelection::Sample { noise } => {
                require_f32(noise, "audio posterior noise")?;
                if noise.dims() != self.mean.dims() {
                    bail!("MiniMax H3 audio posterior noise shape differs from the posterior")
                }
                self.mean
                    .broadcast_add(&self.logs.exp()?.broadcast_mul(noise)?)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), "alpha")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        require_f32(x, "Snake1d input")?;
        let phase = x.broadcast_mul(&self.alpha)?;
        let correction = phase
            .sin()?
            .sqr()?
            .broadcast_div(&self.alpha.affine(1.0, 1e-9)?)?;
        x.broadcast_add(&correction)
    }
}

#[derive(Clone, Debug)]
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
        require_f32(x, "SnakeBeta input")?;
        let alpha = self.alpha.exp()?.reshape((1, self.alpha.dims1()?, 1))?;
        let beta = self.beta.exp()?.reshape((1, self.beta.dims1()?, 1))?;
        let correction = x
            .broadcast_mul(&alpha)?
            .sin()?
            .sqr()?
            .broadcast_div(&beta.affine(1.0, 1e-9)?)?;
        x.broadcast_add(&correction)
    }
}

pub fn snake_value(x: f32, alpha: f32) -> f32 {
    x + (alpha * x).sin().powi(2) / (alpha + 1e-9)
}

pub fn snake_beta_value(x: f32, log_alpha: f32, log_beta: f32) -> f32 {
    let alpha = log_alpha.exp();
    let beta = log_beta.exp();
    x + (alpha * x).sin().powi(2) / (beta + 1e-9)
}

/// Exact alias-free-torch Kaiser-windowed sinc recipe used by BigVGAN.
pub fn kaiser_sinc_coefficients(
    cutoff: f64,
    half_width: f64,
    kernel_size: usize,
) -> Result<Vec<f32>> {
    if !(0.0..=0.5).contains(&cutoff) || half_width <= 0.0 || kernel_size < 2 {
        bail!("invalid MiniMax H3 Kaiser-sinc filter parameters")
    }
    let half_size = kernel_size / 2;
    let attenuation = 2.285 * (half_size - 1) as f64 * PI * (4.0 * half_width) + 7.95;
    let beta = if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation >= 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    };
    let denom = bessel_i0(beta);
    let mut filter = Vec::with_capacity(kernel_size);
    for index in 0..kernel_size {
        let window_position = 2.0 * index as f64 / (kernel_size - 1) as f64 - 1.0;
        let window = bessel_i0(beta * (1.0 - window_position * window_position).sqrt()) / denom;
        let time = if kernel_size.is_multiple_of(2) {
            index as f64 - half_size as f64 + 0.5
        } else {
            index as f64 - half_size as f64
        };
        filter.push((2.0 * cutoff * window * sinc(2.0 * cutoff * time)) as f32);
    }
    let sum = filter.iter().map(|value| *value as f64).sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        bail!("MiniMax H3 Kaiser-sinc filter has a non-finite/zero sum")
    }
    for value in &mut filter {
        *value = (*value as f64 / sum) as f32;
    }
    Ok(filter)
}

fn sinc(value: f64) -> f64 {
    if value == 0.0 {
        1.0
    } else {
        (PI * value).sin() / (PI * value)
    }
}

fn bessel_i0(value: f64) -> f64 {
    // Convergent definition used at initialization only. The H3 kernel's beta
    // needs fewer than 30 terms for f64 machine precision.
    let y = value * value / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=64 {
        term *= y / (k * k) as f64;
        sum += term;
        if term <= f64::EPSILON * sum {
            break;
        }
    }
    sum
}

#[derive(Clone, Debug)]
struct UpSample1d {
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl UpSample1d {
    fn load(ratio: usize, vb: VarBuilder) -> Result<Self> {
        let kernel_size = 12usize;
        let filter = vb.get((1, 1, kernel_size), "filter")?;
        let pad = kernel_size / ratio - 1;
        Ok(Self {
            ratio,
            pad,
            pad_left: pad * ratio + (kernel_size - ratio) / 2,
            pad_right: pad * ratio + (kernel_size - ratio).div_ceil(2),
            filter,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        require_f32(x, "alias-free upsample input")?;
        let (_, channels, _) = x.dims3()?;
        let x = replicate_pad_1d(x, self.pad, self.pad)?.contiguous()?;
        let filter = materialize_group_filter(&self.filter, channels, x.device())?;
        let y = x
            .conv_transpose1d(&filter, 0, 0, self.ratio, 1, channels)?
            .affine(self.ratio as f64, 0.0)?;
        let output_len = y.dim(2)?;
        y.narrow(
            2,
            self.pad_left,
            output_len.saturating_sub(self.pad_left + self.pad_right),
        )
    }
}

#[derive(Clone, Debug)]
struct DownSample1d {
    ratio: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl DownSample1d {
    fn load(ratio: usize, vb: VarBuilder) -> Result<Self> {
        let kernel_size = 12usize;
        Ok(Self {
            ratio,
            pad_left: kernel_size / 2 - usize::from(kernel_size.is_multiple_of(2)),
            pad_right: kernel_size / 2,
            filter: vb.pp("lowpass").get((1, 1, kernel_size), "filter")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        require_f32(x, "alias-free downsample input")?;
        let (_, channels, _) = x.dims3()?;
        let x = replicate_pad_1d(x, self.pad_left, self.pad_right)?.contiguous()?;
        let filter = materialize_group_filter(&self.filter, channels, x.device())?;
        x.conv1d(&filter, 0, self.ratio, 1, channels)
    }
}

#[derive(Clone, Debug)]
struct Activation1d {
    activation: SnakeBeta,
    upsample: UpSample1d,
    downsample: DownSample1d,
}

impl Activation1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            activation: SnakeBeta::load(channels, vb.pp("act"))?,
            upsample: UpSample1d::load(2, vb.pp("upsample"))?,
            downsample: DownSample1d::load(2, vb.pp("downsample"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.downsample
            .forward(&self.activation.forward(&self.upsample.forward(x)?)?)
    }
}

#[derive(Clone, Debug)]
struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn load(
        dim: usize,
        dilation: usize,
        layout: AudioTensorLayout,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            snake1: Snake1d::load(dim, vb.pp("block.0"))?,
            conv1: load_conv1d(
                dim,
                dim,
                7,
                ((7 - 1) * dilation) / 2,
                1,
                dilation,
                true,
                layout,
                vb.pp("block.1"),
            )?,
            snake2: Snake1d::load(dim, vb.pp("block.2"))?,
            conv2: load_conv1d(dim, dim, 1, 0, 1, 1, true, layout, vb.pp("block.3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self.conv2.forward(
            &self
                .snake2
                .forward(&self.conv1.forward(&self.snake1.forward(x)?)?)?,
        )?;
        let pad = (x.dim(2)? - residual.dim(2)?) / 2;
        let shortcut = if pad > 0 {
            x.narrow(2, pad, x.dim(2)? - 2 * pad)?
        } else {
            x.clone()
        };
        shortcut.add(&residual)
    }
}

#[derive(Clone, Debug)]
struct EncoderBlock {
    residuals: Vec<ResidualUnit>,
    snake: Snake1d,
    downsample: Conv1d,
}

impl EncoderBlock {
    fn load(dim: usize, stride: usize, layout: AudioTensorLayout, vb: VarBuilder) -> Result<Self> {
        let half = dim / 2;
        Ok(Self {
            residuals: [1usize, 3, 9]
                .iter()
                .enumerate()
                .map(|(index, dilation)| {
                    ResidualUnit::load(half, *dilation, layout, vb.pp(format!("block.{index}")))
                })
                .collect::<Result<Vec<_>>>()?,
            snake: Snake1d::load(half, vb.pp("block.3"))?,
            downsample: load_conv1d(
                half,
                dim,
                2 * stride,
                stride.div_ceil(2),
                stride,
                1,
                true,
                layout,
                vb.pp("block.4"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for residual in &self.residuals {
            x = residual.forward(&x)?;
        }
        self.downsample.forward(&self.snake.forward(&x)?)
    }
}

#[derive(Clone, Debug)]
struct DacEncoder {
    conv_in: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake_out: Snake1d,
    conv_out: Conv1d,
}

impl DacEncoder {
    fn load(config: &AudioVaeConfig, layout: AudioTensorLayout, vb: VarBuilder) -> Result<Self> {
        let conv_in = load_conv1d(
            1,
            config.encoder_dim,
            7,
            3,
            1,
            1,
            true,
            layout,
            vb.pp("block.0"),
        )?;
        let mut dim = config.encoder_dim;
        let mut blocks = Vec::with_capacity(config.encoder_rates.len());
        for (index, stride) in config.encoder_rates.iter().copied().enumerate() {
            dim *= 2;
            blocks.push(EncoderBlock::load(
                dim,
                stride,
                layout,
                vb.pp(format!("block.{}", index + 1)),
            )?);
        }
        let final_index = config.encoder_rates.len() + 1;
        Ok(Self {
            conv_in,
            blocks,
            snake_out: Snake1d::load(dim, vb.pp(format!("block.{final_index}")))?,
            conv_out: load_conv1d(
                dim,
                config.latent_dim,
                3,
                1,
                1,
                1,
                true,
                layout,
                vb.pp(format!("block.{}", final_index + 1)),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.conv_out.forward(&self.snake_out.forward(&x)?)
    }
}

#[derive(Clone, Debug)]
struct GeGluMlp {
    norm: LayerNorm,
    w0: Linear,
    w1: Linear,
    w2: Linear,
}

impl GeGluMlp {
    fn load(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: candle_nn::layer_norm(in_features, 1e-5, vb.pp("norm"))?,
            w0: candle_nn::linear(in_features, hidden_features, vb.pp("w0"))?,
            w1: candle_nn::linear(in_features, hidden_features, vb.pp("w1"))?,
            w2: candle_nn::linear(hidden_features, in_features, vb.pp("w2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&self.w0.forward(&x)?)?;
        self.w2.forward(&gate.broadcast_mul(&self.w1.forward(&x)?)?)
    }
}

#[derive(Clone, Debug)]
struct CausalAttention {
    qkv: Linear,
    q_bias: Tensor,
    zero_k_bias: Tensor,
    v_bias: Tensor,
    proj: Linear,
    heads: usize,
    head_dim: usize,
    out_dim: usize,
}

impl CausalAttention {
    fn load(in_dim: usize, out_dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: candle_nn::linear_no_bias(in_dim, in_dim * 3, vb.pp("qkv"))?,
            q_bias: vb.get(in_dim, "q_bias")?,
            zero_k_bias: vb.get(in_dim, "zero_k_bias")?,
            v_bias: vb.get(in_dim, "v_bias")?,
            proj: candle_nn::linear(out_dim, out_dim, vb.pp("proj"))?,
            heads,
            head_dim: in_dim / heads,
            out_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        require_f32(x, "causal attention input")?;
        let (batch, rows, in_dim) = x.dims3()?;
        let bias = Tensor::cat(&[&self.q_bias, &self.zero_k_bias, &self.v_bias], 0)?;
        let qkv = self.qkv.forward(x)?.broadcast_add(&bias)?;
        let qkv = qkv
            .reshape((batch, rows, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let query = qkv.narrow(0, 0, 1)?.squeeze(0)?.contiguous()?;
        let key = qkv.narrow(0, 1, 1)?.squeeze(0)?.contiguous()?;
        let value = qkv.narrow(0, 2, 1)?.squeeze(0)?.contiguous()?;
        let scores = query
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let mask = causal_mask(rows, x.device())?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let attended = probabilities.matmul(&value)?;
        // Rank-4 `[batch, heads, rows, head_dim]` averaged over the head axis:
        // candle's Metal reduction is wrong for a rank > 3 tensor reduced over
        // a non-final dim, and this head mean is the same shape class as the
        // LTX-2 per-token RMS that silently rescaled a prompt embedding. Fold
        // to rank 3 around the reduced axis first — a no-op reshape on CUDA and
        // CPU. See `crate::metal_reduce`.
        let mean_heads = crate::metal_reduce::mean_stable(&attended, 1)?;
        let pool = self.head_dim / self.out_dim;
        let pooled = mean_heads
            .reshape((batch, rows, self.out_dim, pool))?
            .sum(D::Minus1)?
            .affine(1.0 / pool as f64, 0.0)?;
        if in_dim != self.heads * self.head_dim {
            bail!("MiniMax H3 audio causal attention dimensions drifted")
        }
        self.proj.forward(&pooled)
    }
}

#[derive(Clone, Debug)]
struct AttnProjection {
    norm1: LayerNorm,
    attention: CausalAttention,
    proj: Linear,
    norm3: LayerNorm,
    norm2: LayerNorm,
    mlp: GeGluMlp,
}

impl AttnProjection {
    fn load(in_dim: usize, out_dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(in_dim, 1e-5, vb.pp("norm1"))?,
            attention: CausalAttention::load(in_dim, out_dim, heads, vb.pp("attn"))?,
            proj: candle_nn::linear(in_dim, out_dim, vb.pp("proj"))?,
            norm3: candle_nn::layer_norm(in_dim, 1e-5, vb.pp("norm3"))?,
            norm2: candle_nn::layer_norm(out_dim, 1e-5, vb.pp("norm2"))?,
            mlp: GeGluMlp::load(out_dim, out_dim * 2, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self
            .proj
            .forward(&self.norm3.forward(x)?)?
            .add(&self.attention.forward(&self.norm1.forward(x)?)?)?;
        residual.add(&self.mlp.forward(&self.norm2.forward(&residual)?)?)
    }
}

#[derive(Clone, Debug)]
struct AmpBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations: Vec<Activation1d>,
}

impl AmpBlock {
    fn load(
        channels: usize,
        kernel: usize,
        dilations: &[usize],
        layout: AudioTensorLayout,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut activations = Vec::with_capacity(dilations.len() * 2);
        for (index, dilation) in dilations.iter().copied().enumerate() {
            convs1.push(load_conv1d(
                channels,
                channels,
                kernel,
                (kernel * dilation - dilation) / 2,
                1,
                dilation,
                true,
                layout,
                vb.pp(format!("convs1.{index}")),
            )?);
            convs2.push(load_conv1d(
                channels,
                channels,
                kernel,
                (kernel - 1) / 2,
                1,
                1,
                true,
                layout,
                vb.pp(format!("convs2.{index}")),
            )?);
        }
        for index in 0..(dilations.len() * 2) {
            activations.push(Activation1d::load(
                channels,
                vb.pp(format!("activations.{index}")),
            )?);
        }
        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for index in 0..self.convs1.len() {
            let residual =
                self.convs2[index].forward(&self.activations[2 * index + 1].forward(
                    &self.convs1[index].forward(&self.activations[2 * index].forward(&x)?)?,
                )?)?;
            x = x.add(&residual)?;
        }
        Ok(x)
    }
}

#[derive(Clone, Debug)]
struct BigVganDecoder {
    conv_pre: Conv1d,
    upsamplers: Vec<ConvTranspose1d>,
    resblocks: Vec<AmpBlock>,
    num_kernels: usize,
    activation_post: Activation1d,
    conv_post: Conv1d,
}

impl BigVganDecoder {
    fn load(config: &AudioVaeConfig, layout: AudioTensorLayout, vb: VarBuilder) -> Result<Self> {
        let conv_pre = load_conv1d(
            config.latent_dim,
            config.decoder_dim,
            7,
            3,
            1,
            1,
            true,
            layout,
            vb.pp("conv_pre"),
        )?;
        let num_kernels = config.resblock_kernel_sizes.len();
        let mut upsamplers = Vec::with_capacity(config.decoder_rates.len());
        let mut resblocks = Vec::with_capacity(config.decoder_rates.len() * num_kernels);
        for (stage, (rate, kernel)) in config
            .decoder_rates
            .iter()
            .zip(&config.decoder_kernel_sizes)
            .enumerate()
        {
            let in_channels = config.decoder_dim / (1usize << stage);
            let out_channels = config.decoder_dim / (1usize << (stage + 1));
            upsamplers.push(load_conv_transpose1d(
                in_channels,
                out_channels,
                *kernel,
                (kernel - rate) / 2,
                *rate,
                true,
                layout,
                vb.pp(format!("ups.{stage}.0")),
            )?);
            for (kernel_index, (res_kernel, dilations)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(&config.resblock_dilation_sizes)
                .enumerate()
            {
                resblocks.push(AmpBlock::load(
                    out_channels,
                    *res_kernel,
                    dilations,
                    layout,
                    vb.pp(format!("resblocks.{}", stage * num_kernels + kernel_index)),
                )?);
            }
        }
        let final_channels = config.decoder_dim / (1usize << config.decoder_rates.len());
        Ok(Self {
            conv_pre,
            upsamplers,
            resblocks,
            num_kernels,
            activation_post: Activation1d::load(final_channels, vb.pp("activation_post"))?,
            conv_post: load_conv1d(
                final_channels,
                1,
                7,
                3,
                1,
                1,
                false,
                layout,
                vb.pp("conv_post"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(x)?;
        for (stage, upsampler) in self.upsamplers.iter().enumerate() {
            x = upsampler.forward(&x)?;
            let mut sum: Option<Tensor> = None;
            for block in &self.resblocks[stage * self.num_kernels..(stage + 1) * self.num_kernels] {
                let output = block.forward(&x)?;
                sum = Some(match sum {
                    Some(previous) => previous.add(&output)?,
                    None => output,
                });
            }
            x = sum
                .ok_or_else(|| candle::Error::Msg("BigVGAN stage has no AMP blocks".into()))?
                .affine(1.0 / self.num_kernels as f64, 0.0)?;
        }
        self.conv_post
            .forward(&self.activation_post.forward(&x)?)?
            .clamp(-1f32, 1f32)
    }
}

#[derive(Clone, Debug)]
pub struct AudioVae {
    config: AudioVaeConfig,
    encoder: DacEncoder,
    pre_block: AttnProjection,
    mean_proj: Conv1d,
    logs_proj: Conv1d,
    dec_in_proj: Conv1d,
    decoder: BigVganDecoder,
    latent_mean: Tensor,
    latent_std: Tensor,
}

impl AudioVae {
    #[cfg(test)]
    pub(super) fn load(
        config: AudioVaeConfig,
        layout: AudioTensorLayout,
        requested_dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::load_with_observer(
            config,
            layout,
            requested_dtype,
            vb,
            &mut |_completed, _total| Ok(()),
        )
    }

    pub(super) fn load_with_observer(
        config: AudioVaeConfig,
        layout: AudioTensorLayout,
        requested_dtype: DType,
        vb: VarBuilder,
        observer: &mut dyn FnMut(usize, usize) -> Result<()>,
    ) -> Result<Self> {
        const LOAD_STAGES: usize = 6;
        config.validate()?;
        config.validate_execution_dtype(requested_dtype)?;
        if vb.dtype() != DType::F32 {
            bail!("MiniMax H3 audio VarBuilder must expose F32 tensors")
        }
        observer(0, LOAD_STAGES)?;
        let latent_mean = Tensor::from_vec(
            config.latents_mean.clone(),
            (1, config.latent_channels, 1),
            vb.device(),
        )?;
        let latent_std = Tensor::from_vec(
            config.latents_std.clone(),
            (1, config.latent_channels, 1),
            vb.device(),
        )?;
        observer(1, LOAD_STAGES)?;
        let encoder = DacEncoder::load(&config, layout, vb.pp("encoder"))?;
        observer(2, LOAD_STAGES)?;
        let pre_block = AttnProjection::load(
            config.latent_dim,
            config.latent_channels,
            config.attention_heads,
            vb.pp("pre_block"),
        )?;
        observer(3, LOAD_STAGES)?;
        let mean_proj = load_plain_conv1d(
            config.latent_channels,
            config.latent_channels,
            1,
            0,
            1,
            1,
            true,
            vb.pp("mean_proj"),
        )?;
        // Loaded even though H3 selects the posterior mode. This makes a
        // truncated/missing stochastic head fail strict construction.
        let logs_proj = load_plain_conv1d(
            config.latent_channels,
            config.latent_channels,
            1,
            0,
            1,
            1,
            true,
            vb.pp("logs_proj"),
        )?;
        observer(4, LOAD_STAGES)?;
        let dec_in_proj = load_plain_conv1d(
            config.latent_channels,
            config.latent_dim,
            1,
            0,
            1,
            1,
            true,
            vb.pp("dec_in_proj"),
        )?;
        observer(5, LOAD_STAGES)?;
        let decoder = BigVganDecoder::load(&config, layout, vb.pp("decoder"))?;
        observer(6, LOAD_STAGES)?;
        Ok(Self {
            config,
            encoder,
            pre_block,
            mean_proj,
            logs_proj,
            dec_in_proj,
            decoder,
            latent_mean,
            latent_std,
        })
    }

    pub fn config(&self) -> &AudioVaeConfig {
        &self.config
    }

    pub fn normalize_mono_latents(&self, latents: &Tensor) -> Result<Tensor> {
        require_f32(latents, "denormalized audio latents")?;
        let (_batch, channels, _rows) = latents.dims3()?;
        if channels != self.config.latent_channels {
            bail!("audio latent channel count does not match the VAE")
        }
        latents
            .broadcast_sub(&self.latent_mean)?
            .broadcast_div(&self.latent_std)
    }

    pub fn denormalize_mono_latents(&self, normalized: &Tensor) -> Result<Tensor> {
        require_f32(normalized, "normalized audio latents")?;
        let (_batch, channels, _rows) = normalized.dims3()?;
        if channels != self.config.latent_channels {
            bail!("audio latent channel count does not match the VAE")
        }
        normalized
            .broadcast_mul(&self.latent_std)?
            .broadcast_add(&self.latent_mean)
    }

    pub fn encode_posterior(
        &self,
        waveform: &StereoWaveform,
        cancellation: &dyn AudioVaeCancellation,
    ) -> Result<(AudioPosterior, usize)> {
        self.encode_posterior_with_checkpoint(waveform, &mut |phase| {
            cancellation_boundary(cancellation, phase)
        })
    }

    fn encode_posterior_with_checkpoint(
        &self,
        waveform: &StereoWaveform,
        checkpoint: &mut dyn FnMut(AudioVaePhase) -> Result<()>,
    ) -> Result<(AudioPosterior, usize)> {
        checkpoint(AudioVaePhase::EncodePreprocess)?;
        let (batch, _, samples) = waveform.samples().dims3()?;
        let padded = self.config.padded_samples(samples)?;
        let mut mono = waveform.pack_mono_batch()?;
        if padded > samples {
            mono = mono.pad_with_zeros(2, 0, padded - samples)?;
        }
        checkpoint(AudioVaePhase::EncodeDac)?;
        let encoded = self.encoder.forward(&mono)?;
        checkpoint(AudioVaePhase::EncodeProjection)?;
        let projected = self
            .pre_block
            .forward(&encoded.transpose(1, 2)?)?
            .transpose(1, 2)?;
        checkpoint(AudioVaePhase::EncodePosterior)?;
        let mean = self.mean_proj.forward(&projected)?;
        let logs = self.logs_proj.forward(&projected)?;
        Ok((AudioPosterior::new(mean, logs)?, batch))
    }

    /// H3-authoritative encoding: posterior mode, then per-channel normalize.
    pub fn encode(
        &self,
        waveform: &StereoWaveform,
        cancellation: &dyn AudioVaeCancellation,
    ) -> Result<StereoLatents> {
        self.encode_with_checkpoint(waveform, &mut |phase| {
            cancellation_boundary(cancellation, phase)
        })
    }

    /// Encode with a fallible callback at each named audio safe point.
    ///
    /// This developer-only seam lets a private Ref2VA runtime retain the
    /// original request-scoped cancellation error instead of reducing it to
    /// the string-only [`AudioVaeCancellation`] transport.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[doc(hidden)]
    pub fn encode_with_phase_checkpoint(
        &self,
        waveform: &StereoWaveform,
        checkpoint: &mut dyn FnMut(AudioVaePhase) -> Result<()>,
    ) -> Result<StereoLatents> {
        self.encode_with_checkpoint(waveform, checkpoint)
    }

    fn encode_with_checkpoint(
        &self,
        waveform: &StereoWaveform,
        checkpoint: &mut dyn FnMut(AudioVaePhase) -> Result<()>,
    ) -> Result<StereoLatents> {
        let (posterior, batch) = self.encode_posterior_with_checkpoint(waveform, checkpoint)?;
        let normalized = self.normalize_mono_latents(&posterior.mode())?;
        StereoLatents::from_mono_batch(normalized, batch, &self.config)
    }

    pub fn decode(
        &self,
        latents: &StereoLatents,
        association: AudioSoundtrackAssociation,
        cancellation: &dyn AudioVaeCancellation,
    ) -> Result<StereoWaveform> {
        self.decode_with_checkpoint(latents, association, &mut |phase| {
            cancellation_boundary(cancellation, phase)
        })
    }

    /// Decode with a fallible callback at each named audio safe point.
    ///
    /// This developer-only seam lets the private H3 pipeline preserve its
    /// request-scoped cancellation and phase-accounting authority without
    /// placing a non-`Send` pipeline checkpoint behind the public
    /// [`AudioVaeCancellation`] trait.
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    #[doc(hidden)]
    pub fn decode_with_phase_checkpoint(
        &self,
        latents: &StereoLatents,
        association: AudioSoundtrackAssociation,
        checkpoint: &mut dyn FnMut(AudioVaePhase) -> Result<()>,
    ) -> Result<StereoWaveform> {
        self.decode_with_checkpoint(latents, association, checkpoint)
    }

    fn decode_with_checkpoint(
        &self,
        latents: &StereoLatents,
        association: AudioSoundtrackAssociation,
        checkpoint: &mut dyn FnMut(AudioVaePhase) -> Result<()>,
    ) -> Result<StereoWaveform> {
        checkpoint(AudioVaePhase::DecodeProjection)?;
        let (batch, _, _, _) = latents.normalized().dims4()?;
        let packed = latents.pack_mono_batch()?;
        let denormalized = self.denormalize_mono_latents(&packed)?;
        let projected = self.dec_in_proj.forward(&denormalized)?;
        checkpoint(AudioVaePhase::DecodeBigVgan)?;
        let decoded = self.decoder.forward(&projected)?;
        checkpoint(AudioVaePhase::DecodeOutput)?;
        if self.config.sample_rate == SAMPLE_RATE {
            StereoWaveform::from_mono_batch(decoded, batch, self.config.sample_rate, association)
        } else {
            StereoWaveform::from_mono_batch_for_config(decoded, batch, &self.config, association)
        }
    }
}

fn require_f32(tensor: &Tensor, label: &str) -> Result<()> {
    if tensor.dtype() != DType::F32 {
        bail!("MiniMax H3 {label} must be F32, got {:?}", tensor.dtype())
    }
    Ok(())
}

fn causal_mask(rows: usize, device: &Device) -> Result<Tensor> {
    let values = (0..rows)
        .flat_map(|query| {
            (0..rows).map(move |key| {
                if key <= query {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, rows, rows), device)
}

fn replicate_pad_1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let (_, _, length) = x.dims3()?;
    if length == 0 {
        bail!("cannot replicate-pad an empty audio tensor")
    }
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        parts.push(x.narrow(2, 0, 1)?.repeat((1, 1, left))?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(2, length - 1, 1)?.repeat((1, 1, right))?);
    }
    Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)
}

fn materialize_group_filter(filter: &Tensor, channels: usize, device: &Device) -> Result<Tensor> {
    require_f32(filter, "alias-free filter")?;
    let base = filter.to_device(device)?.contiguous()?;
    Tensor::cat(&(0..channels).map(|_| &base).collect::<Vec<_>>(), 0)
}

#[allow(clippy::too_many_arguments)]
fn load_plain_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel), "weight")?;
    let bias = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(
        weight,
        bias,
        Conv1dConfig {
            padding,
            stride,
            dilation,
            ..Default::default()
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn load_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    bias: bool,
    layout: AudioTensorLayout,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = load_weight_norm(
        (out_channels, in_channels, kernel),
        out_channels,
        layout,
        vb.clone(),
    )?;
    let bias = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(
        weight,
        bias,
        Conv1dConfig {
            padding,
            stride,
            dilation,
            ..Default::default()
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn load_conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    bias: bool,
    layout: AudioTensorLayout,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight = load_weight_norm(
        (in_channels, out_channels, kernel),
        in_channels,
        layout,
        vb.clone(),
    )?;
    let bias = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
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

fn load_weight_norm(
    shape: (usize, usize, usize),
    norm_channels: usize,
    layout: AudioTensorLayout,
    vb: VarBuilder,
) -> Result<Tensor> {
    match layout {
        AudioTensorLayout::ComfyFolded => vb.get(shape, "weight"),
        AudioTensorLayout::OfficialWeightNorm => {
            let g = vb.get((norm_channels, 1, 1), "weight_g")?;
            let v = vb.get(shape, "weight_v")?;
            let norm = v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
            v.broadcast_mul(&g.broadcast_div(&norm)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use candle::{Device, Tensor};

    use super::*;

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "index {index}: {left} != {right} within {tolerance}"
            );
        }
    }

    #[test]
    fn stereo_pack_order_round_trips() -> Result<()> {
        let samples = Tensor::from_vec(
            vec![1f32, 2., 3., 10., 20., 30., 4., 5., 6., 40., 50., 60.],
            (2, 2, 3),
            &Device::Cpu,
        )?;
        let waveform = StereoWaveform::new(
            samples,
            SAMPLE_RATE,
            AudioSoundtrackAssociation::VideoSoundtrack { reference_index: 3 },
        )?;
        assert_eq!(
            waveform
                .pack_mono_batch()?
                .flatten_all()?
                .to_vec1::<f32>()?,
            [1., 2., 3., 10., 20., 30., 4., 5., 6., 40., 50., 60.]
        );
        let round_trip = StereoWaveform::from_mono_batch(
            waveform.pack_mono_batch()?,
            2,
            SAMPLE_RATE,
            waveform.association(),
        )?;
        assert_eq!(
            round_trip.samples().to_vec3::<f32>()?,
            waveform.samples().to_vec3::<f32>()?
        );
        assert_eq!(round_trip.association(), waveform.association());
        Ok(())
    }

    #[test]
    fn posterior_mode_is_default_authority_and_sampling_is_explicit() -> Result<()> {
        let mean = Tensor::from_vec(vec![1f32, -2.], (1, 1, 2), &Device::Cpu)?;
        let logs = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
        let posterior = AudioPosterior::new(mean, logs)?;
        assert_eq!(
            posterior
                .select(AudioPosteriorSelection::Mode)?
                .to_vec3::<f32>()?,
            vec![vec![vec![1., -2.]]]
        );
        let noise = Tensor::from_vec(vec![0.5f32, -0.25], (1, 1, 2), &Device::Cpu)?;
        assert_eq!(
            posterior
                .select(AudioPosteriorSelection::Sample { noise: &noise })?
                .to_vec3::<f32>()?,
            vec![vec![vec![1.5, -2.25]]]
        );
        Ok(())
    }

    #[test]
    fn snake_formulas_preserve_phase_polarity_and_amplitude_contract() {
        assert!((snake_value(0.25, 2.0) - (0.25 + (0.5f32).sin().powi(2) / 2.0)).abs() < 1e-7);
        assert!((snake_beta_value(-0.5, 0.0, 0.0) - (-0.5 + (-0.5f32).sin().powi(2))).abs() < 1e-7);
        assert!(snake_beta_value(0.75, 0.0, 0.0) > 0.75);
        assert!(snake_beta_value(-0.75, 0.0, 0.0) > -0.75);
    }

    #[test]
    fn kaiser_filter_matches_pinned_coefficients_and_unity_sum() -> Result<()> {
        let filter = kaiser_sinc_coefficients(0.25, 0.3, 12)?;
        let expected = [
            0.0020289664,
            0.009389464,
            -0.025543464,
            -0.057657376,
            0.12857261,
            0.4432098,
            0.4432098,
            0.12857261,
            -0.057657376,
            -0.025543464,
            0.009389464,
            0.0020289664,
        ];
        assert_close(&filter, &expected, 2e-6);
        assert!((filter.iter().sum::<f32>() - 1.0).abs() < 2e-6);
        for index in 0..filter.len() {
            assert!((filter[index] - filter[filter.len() - 1 - index]).abs() < 1e-7);
        }
        Ok(())
    }

    fn attention_builder(device: &Device) -> Result<VarBuilder<'static>> {
        let mut tensors = HashMap::new();
        let mut qkv = vec![0f32; 12 * 4];
        for block in 0..3 {
            for index in 0..4 {
                qkv[(block * 4 + index) * 4 + index] = 1.0;
            }
        }
        tensors.insert("qkv.weight".into(), Tensor::from_vec(qkv, (12, 4), device)?);
        tensors.insert("q_bias".into(), Tensor::zeros(4, DType::F32, device)?);
        tensors.insert("zero_k_bias".into(), Tensor::zeros(4, DType::F32, device)?);
        tensors.insert("v_bias".into(), Tensor::zeros(4, DType::F32, device)?);
        tensors.insert("proj.weight".into(), Tensor::eye(2, DType::F32, device)?);
        tensors.insert("proj.bias".into(), Tensor::zeros(2, DType::F32, device)?);
        Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
    }

    #[test]
    fn causal_attention_rejects_future_leakage() -> Result<()> {
        let attention = CausalAttention::load(4, 2, 2, attention_builder(&Device::Cpu)?)?;
        let base = Tensor::from_vec(
            vec![1f32, 0., 0., 1., 0.5, 0.25, -0.5, 0.75, 1., 1., 1., 1.],
            (1, 3, 4),
            &Device::Cpu,
        )?;
        let perturbed = Tensor::from_vec(
            vec![
                1f32, 0., 0., 1., 0.5, 0.25, -0.5, 0.75, 100., -100., 80., -80.,
            ],
            (1, 3, 4),
            &Device::Cpu,
        )?;
        let base_out = attention
            .forward(&base)?
            .narrow(1, 0, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let changed_out = attention
            .forward(&perturbed)?
            .narrow(1, 0, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&base_out, &changed_out, 1e-6);
        Ok(())
    }

    struct CancelAt {
        phase: AudioVaePhase,
        seen: Mutex<Vec<AudioVaePhase>>,
    }

    impl AudioVaeCancellation for CancelAt {
        fn is_cancelled(&self, phase: AudioVaePhase) -> bool {
            self.seen.lock().unwrap().push(phase);
            phase == self.phase
        }
    }

    fn tiny_vae(device: &Device) -> Result<AudioVae> {
        let config = AudioVaeConfig::tiny();
        AudioVae::load(
            config,
            AudioTensorLayout::ComfyFolded,
            DType::F32,
            VarBuilder::zeros(DType::F32, device),
        )
    }

    #[test]
    fn official_weight_norm_and_comfy_folded_weights_agree() -> Result<()> {
        let device = Device::Cpu;
        let v = Tensor::from_vec(vec![3f32, 4., 0., 0., 0., 12., 5., 0.], (2, 2, 2), &device)?;
        let g = Tensor::from_vec(vec![10f32, 13.], (2, 1, 1), &device)?;
        let expected = vec![6f32, 8., 0., 0., 0., 12., 5., 0.];
        let mut official = HashMap::new();
        official.insert("weight_g".into(), g);
        official.insert("weight_v".into(), v);
        let folded = load_weight_norm(
            (2, 2, 2),
            2,
            AudioTensorLayout::OfficialWeightNorm,
            VarBuilder::from_tensors(official, DType::F32, &device),
        )?;
        assert_close(&folded.flatten_all()?.to_vec1::<f32>()?, &expected, 1e-6);

        let mut comfy = HashMap::new();
        comfy.insert(
            "weight".into(),
            Tensor::from_vec(expected.clone(), (2, 2, 2), &device)?,
        );
        let direct = load_weight_norm(
            (2, 2, 2),
            2,
            AudioTensorLayout::ComfyFolded,
            VarBuilder::from_tensors(comfy, DType::F32, &device),
        )?;
        assert_close(&direct.flatten_all()?.to_vec1::<f32>()?, &expected, 0.0);
        Ok(())
    }

    #[test]
    fn tiny_encode_decode_shapes_normalization_and_cancellation() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        let vae = tiny_vae(&Device::Cpu)?;
        let waveform = StereoWaveform::new_for_config(
            Tensor::from_vec(
                (0..14).map(|v| v as f32 / 16.0).collect(),
                (1, 2, 7),
                &Device::Cpu,
            )?,
            &config,
            AudioSoundtrackAssociation::StandaloneReference { reference_index: 1 },
        )?;
        let latents = vae.encode(&waveform, &NeverCancel)?;
        assert_eq!(latents.normalized().dims4()?, (1, 2, 2, 2));
        let decoded = vae.decode(
            &latents,
            AudioSoundtrackAssociation::Generated,
            &NeverCancel,
        )?;
        assert_eq!(decoded.samples().dims3()?, (1, 2, 8));
        assert!(decoded
            .samples()
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|sample| (-1.0..=1.0).contains(sample)));

        let raw = Tensor::from_vec(vec![0.25f32, 2.25, -0.5, 0.0], (1, 2, 2), &Device::Cpu)?;
        let normalized = vae.normalize_mono_latents(&raw)?;
        let round_trip = vae.denormalize_mono_latents(&normalized)?;
        assert_close(
            &raw.flatten_all()?.to_vec1::<f32>()?,
            &round_trip.flatten_all()?.to_vec1::<f32>()?,
            1e-6,
        );

        for phase in [
            AudioVaePhase::EncodePreprocess,
            AudioVaePhase::EncodeDac,
            AudioVaePhase::EncodeProjection,
            AudioVaePhase::EncodePosterior,
        ] {
            let cancellation = CancelAt {
                phase,
                seen: Mutex::new(Vec::new()),
            };
            assert!(vae.encode(&waveform, &cancellation).is_err());
            assert!(cancellation.seen.lock().unwrap().contains(&phase));
        }
        for phase in [
            AudioVaePhase::DecodeProjection,
            AudioVaePhase::DecodeBigVgan,
            AudioVaePhase::DecodeOutput,
        ] {
            let cancellation = CancelAt {
                phase,
                seen: Mutex::new(Vec::new()),
            };
            assert!(vae
                .decode(
                    &latents,
                    AudioSoundtrackAssociation::Generated,
                    &cancellation,
                )
                .is_err());
            assert!(cancellation.seen.lock().unwrap().contains(&phase));
        }
        #[cfg(feature = "h3-private-uat")]
        {
            let encode_phases = [
                AudioVaePhase::EncodePreprocess,
                AudioVaePhase::EncodeDac,
                AudioVaePhase::EncodeProjection,
                AudioVaePhase::EncodePosterior,
            ];
            let mut phases = Vec::new();
            let checkpoint_latents = vae.encode_with_phase_checkpoint(&waveform, &mut |phase| {
                phases.push(phase);
                Ok(())
            })?;
            assert_eq!(checkpoint_latents.normalized().dims4()?, (1, 2, 2, 2));
            assert_eq!(phases, encode_phases);
            assert_eq!(
                waveform.association(),
                AudioSoundtrackAssociation::StandaloneReference { reference_index: 1 }
            );

            for cancelled_phase in [
                AudioVaePhase::EncodePreprocess,
                AudioVaePhase::EncodeProjection,
                AudioVaePhase::EncodePosterior,
            ] {
                let mut seen = Vec::new();
                let error = vae
                    .encode_with_phase_checkpoint(&waveform, &mut |phase| {
                        seen.push(phase);
                        if phase == cancelled_phase {
                            bail!("synthetic pipeline cancellation at {phase:?}")
                        }
                        Ok(())
                    })
                    .unwrap_err();
                assert_eq!(seen.last(), Some(&cancelled_phase));
                assert!(error
                    .to_string()
                    .contains("synthetic pipeline cancellation"));
                assert_eq!(
                    waveform.association(),
                    AudioSoundtrackAssociation::StandaloneReference { reference_index: 1 }
                );
            }

            let soundtrack = StereoWaveform::new_for_config(
                waveform.samples().clone(),
                &config,
                AudioSoundtrackAssociation::VideoSoundtrack { reference_index: 4 },
            )?;
            vae.encode_with_phase_checkpoint(&soundtrack, &mut |_phase| Ok(()))?;
            assert_eq!(
                soundtrack.association(),
                AudioSoundtrackAssociation::VideoSoundtrack { reference_index: 4 }
            );

            let mut phases = Vec::new();
            vae.decode_with_phase_checkpoint(
                &latents,
                AudioSoundtrackAssociation::Generated,
                &mut |phase| {
                    phases.push(phase);
                    Ok(())
                },
            )?;
            assert_eq!(
                phases,
                [
                    AudioVaePhase::DecodeProjection,
                    AudioVaePhase::DecodeBigVgan,
                    AudioVaePhase::DecodeOutput,
                ]
            );

            let error = vae
                .decode_with_phase_checkpoint(
                    &latents,
                    AudioSoundtrackAssociation::Generated,
                    &mut |phase| {
                        if phase == AudioVaePhase::DecodeBigVgan {
                            bail!("synthetic pipeline cancellation")
                        }
                        Ok(())
                    },
                )
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("synthetic pipeline cancellation"));
        }
        Ok(())
    }

    #[test]
    fn non_f32_inputs_fail_closed() -> Result<()> {
        let config = AudioVaeConfig::tiny();
        let bad = Tensor::zeros((1, 2, 8), DType::BF16, &Device::Cpu)?;
        assert!(StereoWaveform::new_for_config(
            bad,
            &config,
            AudioSoundtrackAssociation::Generated
        )
        .is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_cpu_cuda_parity() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let input = (0..16)
            .map(|index| (index as f32 * 0.17).sin() * 0.5)
            .collect::<Vec<_>>();
        let cpu_vae = tiny_nonzero_vae(&Device::Cpu)?;
        let cuda_vae = tiny_nonzero_vae(&cuda)?;
        let cpu_wave = StereoWaveform::new_for_config(
            Tensor::from_vec(input.clone(), (1, 2, 8), &Device::Cpu)?,
            cpu_vae.config(),
            AudioSoundtrackAssociation::Generated,
        )?;
        let cuda_wave = StereoWaveform::new_for_config(
            Tensor::from_vec(input, (1, 2, 8), &cuda)?,
            cuda_vae.config(),
            AudioSoundtrackAssociation::Generated,
        )?;
        let cpu = cpu_vae
            .encode(&cpu_wave, &NeverCancel)?
            .normalized()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gpu = cuda_vae
            .encode(&cuda_wave, &NeverCancel)?
            .normalized()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&cpu, &gpu, 2e-4);

        let cpu_latents = cpu_vae.encode(&cpu_wave, &NeverCancel)?;
        let cuda_latents = cuda_vae.encode(&cuda_wave, &NeverCancel)?;
        let cpu_decoded = cpu_vae
            .decode(
                &cpu_latents,
                AudioSoundtrackAssociation::Generated,
                &NeverCancel,
            )?
            .samples()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gpu_decoded = cuda_vae
            .decode(
                &cuda_latents,
                AudioSoundtrackAssociation::Generated,
                &NeverCancel,
            )?
            .samples()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_close(&cpu_decoded, &gpu_decoded, 3e-4);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn tiny_nonzero_vae(device: &Device) -> Result<AudioVae> {
        use super::super::audio_weights::audio_tensor_specs;

        let config = AudioVaeConfig::tiny();
        let specs = audio_tensor_specs(&config, AudioTensorLayout::ComfyFolded)?;
        let filter = kaiser_sinc_coefficients(0.25, 0.3, 12)?;
        let mut tensors = HashMap::new();
        for (name, spec) in specs {
            let elements = spec.shape.iter().product::<usize>();
            let values = if name.ends_with("upsample.filter")
                || name.ends_with("downsample.lowpass.filter")
            {
                filter.clone()
            } else if name.ends_with(".act.alpha") || name.ends_with(".act.beta") {
                vec![0.0; elements]
            } else if name.ends_with(".alpha")
                || matches!(
                    name.as_str(),
                    "pre_block.norm1.weight"
                        | "pre_block.norm2.weight"
                        | "pre_block.norm3.weight"
                        | "pre_block.mlp.norm.weight"
                )
            {
                vec![1.0; elements]
            } else if name == "latents_mean" {
                config.latents_mean.clone()
            } else if name == "latents_std" {
                config.latents_std.clone()
            } else if name.ends_with("weight") {
                (0..elements)
                    .map(|index| ((index % 11) as f32 - 5.0) * 0.002)
                    .collect()
            } else {
                vec![0.0; elements]
            };
            tensors.insert(name, Tensor::from_vec(values, spec.shape, device)?);
        }
        AudioVae::load(
            config,
            AudioTensorLayout::ComfyFolded,
            DType::F32,
            VarBuilder::from_tensors(tensors, DType::F32, device),
        )
    }
}
