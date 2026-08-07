#![allow(clippy::too_many_arguments)]

use super::visual_condition::{
    denormalize_imagenet, denormalize_latents, ConditionEncodeMode, DiagonalGaussianDistribution,
    ImageNetPixels, SignedRgbPixels, Uint8RgbPixels, UnitRgbPixels, H3_LATENTS_MEAN,
    H3_LATENTS_STD,
};
use super::visual_geometry::{SpatialTilePlan, VisualTemporalGeometry};
use candle::{bail, DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{ops, Conv2d, Conv2dConfig, GroupNorm, Init, VarBuilder};
use serde::{Deserialize, Serialize};

const TEST_WEIGHT_INIT: Init = Init::Randn {
    mean: 0.0,
    stdev: 0.02,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MiniMaxH3VisualVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub spatial_downsample_factors: Vec<usize>,
    pub temporal_downsample_factors: Vec<usize>,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub spatial_padding_mode: String,
    pub decoder_num_layers: usize,
    pub decoder_num_attention_heads: usize,
    pub decoder_attention_head_dim: usize,
    pub decoder_num_register_tokens: usize,
    pub decoder_ffn_mult: usize,
    pub decoder_rope_theta: f64,
    pub decoder_rope_dim_ratio: f64,
    pub decoder_norm_eps: f64,
    pub clip_length: usize,
    pub token_drop: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub tile_size: usize,
    pub tile_overlap_min: usize,
    pub tiling: bool,
}

impl Default for MiniMaxH3VisualVaeConfig {
    fn default() -> Self {
        Self::production()
    }
}

impl MiniMaxH3VisualVaeConfig {
    pub fn production() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 24,
            block_out_channels: vec![128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_downsample_factors: vec![2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: vec![1, 2, 2, 1, 1, 1],
            norm_num_groups: 32,
            norm_eps: 1e-6,
            spatial_padding_mode: "reflect".into(),
            decoder_num_layers: 36,
            decoder_num_attention_heads: 32,
            decoder_attention_head_dim: 64,
            decoder_num_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            latents_mean: H3_LATENTS_MEAN.to_vec(),
            latents_std: H3_LATENTS_STD.to_vec(),
            tile_size: 256,
            tile_overlap_min: 64,
            tiling: true,
        }
    }

    #[cfg(test)]
    pub(crate) fn tiny_for_tests() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 2,
            block_out_channels: vec![4, 8],
            layers_per_block: 1,
            spatial_downsample_factors: vec![2, 2],
            temporal_downsample_factors: vec![2, 2],
            norm_num_groups: 2,
            norm_eps: 1e-6,
            spatial_padding_mode: "reflect".into(),
            decoder_num_layers: 1,
            decoder_num_attention_heads: 1,
            decoder_attention_head_dim: 8,
            decoder_num_register_tokens: 2,
            decoder_ffn_mult: 2,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            latents_mean: vec![0.25, -0.5],
            latents_std: vec![1.5, 0.75],
            tile_size: 32,
            tile_overlap_min: 8,
            tiling: true,
        }
    }

    pub fn spatial_compression_ratio(&self) -> usize {
        self.spatial_downsample_factors
            .iter()
            .fold(1usize, |ratio, factor| ratio.saturating_mul(*factor))
    }

    pub fn temporal_compression_ratio(&self) -> usize {
        self.temporal_downsample_factors
            .iter()
            .fold(1usize, |ratio, factor| ratio.saturating_mul(*factor))
    }

    pub fn temporal_geometry(&self) -> VisualTemporalGeometry {
        VisualTemporalGeometry {
            clip_length: self.clip_length,
            token_drop: self.token_drop,
            temporal_compression: self.temporal_compression_ratio(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let levels = self.block_out_channels.len();
        if levels == 0
            || self.spatial_downsample_factors.len() != levels
            || self.temporal_downsample_factors.len() != levels
        {
            bail!(
                "H3 visual VAE channel and downsample schedules must be nonempty and equal length"
            )
        }
        if self.in_channels != 3 || self.out_channels != 3 || self.latent_channels == 0 {
            bail!("H3 visual VAE requires RGB input/output and nonzero latent channels")
        }
        if self.layers_per_block == 0 || self.norm_num_groups == 0 {
            bail!("H3 visual VAE residual depth and norm groups must be positive")
        }
        if !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
            || !self.decoder_norm_eps.is_finite()
            || self.decoder_norm_eps <= 0.0
        {
            bail!("H3 visual VAE normalization epsilons must be finite and positive")
        }
        if self.spatial_padding_mode != "reflect" {
            bail!(
                "H3 f16t4d24 requires reflect spatial padding, got {:?}",
                self.spatial_padding_mode
            )
        }
        if self
            .block_out_channels
            .iter()
            .any(|channels| *channels == 0 || *channels % self.norm_num_groups != 0)
        {
            bail!("H3 visual VAE channels must be nonzero and divisible by norm groups")
        }
        if self
            .spatial_downsample_factors
            .iter()
            .chain(self.temporal_downsample_factors.iter())
            .any(|factor| !matches!(*factor, 1 | 2))
        {
            bail!("H3 visual VAE downsample factors must be 1 or 2")
        }
        checked_compression_ratio(&self.spatial_downsample_factors, "spatial")?;
        checked_compression_ratio(&self.temporal_downsample_factors, "temporal")?;
        self.temporal_geometry().validate()?;
        if self.decoder_num_layers == 0
            || self.decoder_num_attention_heads == 0
            || self.decoder_attention_head_dim == 0
            || self.decoder_ffn_mult == 0
        {
            bail!("H3 visual VAE decoder dimensions must be positive")
        }
        let decoder_dim = self
            .decoder_num_attention_heads
            .checked_mul(self.decoder_attention_head_dim)
            .ok_or_else(|| candle::Error::Msg("H3 decoder width overflow".into()))?;
        let decoder_inner = decoder_dim
            .checked_mul(self.decoder_ffn_mult)
            .ok_or_else(|| candle::Error::Msg("H3 decoder feed-forward width overflow".into()))?;
        decoder_inner
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("H3 decoder gated width overflow".into()))?;
        self.latent_channels
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("H3 posterior width overflow".into()))?;
        self.out_channels
            .checked_mul(self.temporal_compression_ratio())
            .and_then(|volume| volume.checked_mul(self.spatial_compression_ratio()))
            .and_then(|volume| volume.checked_mul(self.spatial_compression_ratio()))
            .ok_or_else(|| candle::Error::Msg("H3 decoder patch volume overflow".into()))?;
        let rope_dim =
            (self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio) as usize;
        if rope_dim == 0
            || rope_dim > self.decoder_attention_head_dim
            || !rope_dim.is_multiple_of(6)
            || !self.decoder_rope_dim_ratio.is_finite()
            || self.decoder_rope_dim_ratio <= 0.0
            || !self.decoder_rope_theta.is_finite()
            || self.decoder_rope_theta <= 0.0
        {
            bail!("H3 visual VAE rotary dimension must be positive, head-bounded, and divisible by six")
        }
        if self.latents_mean.len() != self.latent_channels
            || self.latents_std.len() != self.latent_channels
            || self.latents_mean.iter().any(|value| !value.is_finite())
            || self
                .latents_std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("H3 visual VAE latent statistics do not match latent_channels")
        }
        SpatialTilePlan::new(
            self.tile_size,
            self.tile_size,
            self.tile_overlap_min,
            self.spatial_compression_ratio(),
        )?;
        Ok(())
    }

    pub fn validate_production_contract(&self) -> Result<()> {
        self.validate()?;
        if self != &Self::production() {
            bail!("H3 f16t4d24 artifact metadata does not match the released production contract")
        }
        Ok(())
    }
}

fn checked_compression_ratio(factors: &[usize], label: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |ratio, factor| {
        ratio
            .checked_mul(*factor)
            .ok_or_else(|| candle::Error::Msg(format!("H3 {label} compression ratio overflow")))
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightLayout {
    #[default]
    Diffusers,
    OfficialComfySource,
}

/// FP32 parameter storage is invariant. The released CUDA decode executes
/// autocast-eligible projections/convolutions in FP16 while norms and rotary
/// construction stay FP32. CPU and Metal remain FP32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeComputePolicy {
    #[default]
    Reference,
    FullF32,
}

impl DecodeComputePolicy {
    pub fn effective_dtype(self, device: &Device) -> DType {
        match self {
            Self::Reference if device.is_cuda() => DType::F16,
            Self::Reference | Self::FullF32 => DType::F32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualVaePhase {
    EncodeChunk,
    EncodeTile,
    DecodeChunk,
    DecodeTile,
    DecoderBlock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualVaeEvent {
    pub phase: VisualVaePhase,
    pub completed: usize,
    pub total: usize,
}

pub trait VisualVaeObserver {
    /// Returning an error cancels immediately at the named safe point.
    fn checkpoint(&mut self, event: VisualVaeEvent) -> Result<()>;
}

#[derive(Default)]
pub struct NoopVisualVaeObserver;

impl VisualVaeObserver for NoopVisualVaeObserver {
    fn checkpoint(&mut self, _event: VisualVaeEvent) -> Result<()> {
        Ok(())
    }
}

pub trait DecodeSink {
    /// Receives contiguous `[B,3,T,H,W]` unit-range FP32 chunks in temporal
    /// order. A runtime can encode/mux them immediately without retaining a
    /// second full decoded video.
    fn write(&mut self, frame_offset: usize, frames: &Tensor) -> Result<()>;
}

#[derive(Clone, Debug)]
struct MixedLinear {
    stored_weight: Tensor,
    stored_bias: Option<Tensor>,
    compute_weight: Option<Tensor>,
    compute_bias: Option<Tensor>,
    compute_dtype: DType,
}

impl MixedLinear {
    fn new(weight: Tensor, bias: Option<Tensor>, compute_dtype: DType) -> Result<Self> {
        let stored_weight = weight.to_dtype(DType::F32)?;
        let stored_bias = match bias {
            Some(bias) => Some(bias.to_dtype(DType::F32)?),
            None => None,
        };
        let compute_weight = if compute_dtype == DType::F32 {
            None
        } else {
            Some(stored_weight.to_dtype(compute_dtype)?)
        };
        let compute_bias = if compute_dtype == DType::F32 {
            None
        } else {
            stored_bias
                .as_ref()
                .map(|bias| bias.to_dtype(compute_dtype))
                .transpose()?
        };
        Ok(Self {
            stored_weight,
            stored_bias,
            compute_weight,
            compute_bias,
            compute_dtype,
        })
    }

    fn load(vb: VarBuilder, input: usize, output: usize, compute_dtype: DType) -> Result<Self> {
        let weight = vb.get_with_hints((output, input), "weight", TEST_WEIGHT_INIT)?;
        let bias = vb.get_with_hints(output, "bias", Init::Const(0.0))?;
        Self::new(weight, Some(bias), compute_dtype)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (weight, bias) = match &self.compute_weight {
            Some(weight) => (weight, self.compute_bias.as_ref()),
            None => (&self.stored_weight, self.stored_bias.as_ref()),
        };
        let input = input.to_dtype(self.compute_dtype)?;
        let mut output_shape = input.dims().to_vec();
        let input_dim = output_shape
            .last()
            .copied()
            .ok_or_else(|| candle::Error::Msg("H3 linear input must have rank >= 1".into()))?;
        if input_dim != weight.dim(1)? {
            bail!(
                "H3 linear input width {input_dim} does not match weight width {}",
                weight.dim(1)?
            )
        }
        let rows = input.elem_count() / input_dim;
        let output = input.reshape((rows, input_dim))?.matmul(&weight.t()?)?;
        output_shape.pop();
        output_shape.push(weight.dim(0)?);
        let output = output.reshape(output_shape.as_slice())?;
        match bias {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }

    fn stored_dtype(&self) -> DType {
        self.stored_weight.dtype()
    }
}

#[derive(Clone, Debug)]
struct Fp32Vector {
    stored: Tensor,
}

impl Fp32Vector {
    fn new(stored: Tensor) -> Result<Self> {
        Ok(Self {
            stored: stored.to_dtype(DType::F32)?,
        })
    }

    fn value(&self) -> &Tensor {
        &self.stored
    }
}

#[derive(Clone, Debug)]
struct PointwiseConv3d {
    linear: MixedLinear,
}

impl PointwiseConv3d {
    fn load(vb: VarBuilder, input: usize, output: usize, compute_dtype: DType) -> Result<Self> {
        let weight = vb
            .get_with_hints((output, input, 1, 1, 1), "weight", TEST_WEIGHT_INIT)?
            .reshape((output, input))?;
        let bias = vb.get_with_hints(output, "bias", Init::Const(0.0))?;
        Ok(Self {
            linear: MixedLinear::new(weight, Some(bias), compute_dtype)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = input.dims5()?;
        let output_channels = self.linear.stored_weight.dim(0)?;
        self.linear
            .forward(
                &input
                    .permute((0, 2, 3, 4, 1))?
                    .contiguous()?
                    .reshape((batch * frames * height * width, channels))?,
            )?
            .reshape((batch, frames, height, width, output_channels))?
            .permute((0, 4, 1, 2, 3))
    }
}

fn cat_tensors(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
    let refs = tensors.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, dim)
}

fn map_frames<F>(input: &Tensor, mut function: F) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let frames = input.dim(2)?;
    let mut output = Vec::with_capacity(frames);
    for frame in 0..frames {
        output.push(function(&input.narrow(2, frame, 1)?.squeeze(2)?.contiguous()?)?.unsqueeze(2)?);
    }
    cat_tensors(&output, 2)
}

#[derive(Clone, Debug)]
struct H3CausalConv3d {
    temporal_kernel: usize,
    temporal_stride: usize,
    spatial_padding: usize,
    slices: Vec<Conv2d>,
    bias: Tensor,
}

impl H3CausalConv3d {
    fn load(
        vb: VarBuilder,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        temporal_stride: usize,
        spatial_stride: usize,
        spatial_padding: usize,
    ) -> Result<Self> {
        let weight = vb.get_with_hints(
            (output_channels, input_channels, kernel, kernel, kernel),
            "weight",
            TEST_WEIGHT_INIT,
        )?;
        let bias = vb.get_with_hints(output_channels, "bias", Init::Const(0.0))?;
        let mut slices = Vec::with_capacity(kernel);
        for temporal in 0..kernel {
            slices.push(Conv2d::new(
                weight.i((.., .., temporal, .., ..))?.contiguous()?,
                None,
                Conv2dConfig {
                    stride: spatial_stride,
                    ..Default::default()
                },
            ));
        }
        Ok(Self {
            temporal_kernel: kernel,
            temporal_stride,
            spatial_padding,
            slices,
            bias,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, _frames, height, width) = input.dims5()?;
        if input.dtype() != DType::F32 {
            bail!("H3 causal encoder convolutions require FP32 activations")
        }
        let front = self.temporal_kernel.saturating_sub(1);
        let input = if front == 0 {
            input.clone()
        } else {
            Tensor::cat(
                &[
                    &Tensor::zeros(
                        (batch, channels, front, height, width),
                        DType::F32,
                        input.device(),
                    )?,
                    input,
                ],
                2,
            )?
        };
        let padded_frames = input.dim(2)?;
        if padded_frames < self.temporal_kernel {
            bail!("H3 causal convolution received too few temporal frames")
        }
        let output_frames = (padded_frames - self.temporal_kernel) / self.temporal_stride + 1;
        let mut outputs = Vec::with_capacity(output_frames);
        for output_frame in 0..output_frames {
            let base = output_frame * self.temporal_stride;
            let mut accumulated: Option<Tensor> = None;
            for kernel_frame in 0..self.temporal_kernel {
                let frame = input
                    .narrow(2, base + kernel_frame, 1)?
                    .squeeze(2)?
                    .contiguous()?;
                let frame = reflect_pad_4d(&frame, self.spatial_padding, self.spatial_padding)?;
                let projected = frame.apply(&self.slices[kernel_frame])?;
                accumulated = Some(match accumulated {
                    Some(previous) => previous.add(&projected)?,
                    None => projected,
                });
            }
            outputs.push(
                accumulated
                    .ok_or_else(|| {
                        candle::Error::Msg("H3 causal convolution has no kernel slices".into())
                    })?
                    .unsqueeze(2)?,
            );
        }
        let output = cat_tensors(&outputs, 2)?;
        output.broadcast_add(&self.bias.reshape((1, self.bias.dim(0)?, 1, 1, 1))?)
    }
}

fn reflect_pad_4d(input: &Tensor, pad_h: usize, pad_w: usize) -> Result<Tensor> {
    let (_, _, height, width) = input.dims4()?;
    if (pad_h > 0 && height <= pad_h) || (pad_w > 0 && width <= pad_w) {
        bail!(
            "H3 reflect padding requires spatial dimensions larger than padding, got {height}x{width} pad={pad_h}x{pad_w}"
        )
    }
    let mut output = input.clone();
    if pad_w > 0 {
        let left = output.narrow(3, 1, pad_w)?.contiguous()?.flip(&[3])?;
        let right = output
            .narrow(3, width - pad_w - 1, pad_w)?
            .contiguous()?
            .flip(&[3])?;
        output = Tensor::cat(&[&left, &output, &right], 3)?;
    }
    if pad_h > 0 {
        let top = output.narrow(2, 1, pad_h)?.contiguous()?.flip(&[2])?;
        let bottom = output
            .narrow(2, height - pad_h - 1, pad_h)?
            .contiguous()?
            .flip(&[2])?;
        output = Tensor::cat(&[&top, &output, &bottom], 2)?;
    }
    Ok(output)
}

fn pad_bottom_right_reflect(input: &Tensor) -> Result<Tensor> {
    let (_, _, _, height, width) = input.dims5()?;
    if height < 2 || width < 2 {
        bail!("H3 stride-2 reflect padding requires at least 2x2 input")
    }
    let right = input.narrow(4, width - 2, 1)?.contiguous()?;
    let with_right = Tensor::cat(&[input, &right], 4)?;
    let bottom = with_right.narrow(3, height - 2, 1)?.contiguous()?;
    Tensor::cat(&[&with_right, &bottom], 3)
}

#[derive(Clone, Debug)]
struct TemporalIsolatedGroupNorm {
    norm: GroupNorm,
}

impl TemporalIsolatedGroupNorm {
    fn load(vb: VarBuilder, groups: usize, channels: usize, eps: f64) -> Result<Self> {
        let weight = vb.get_with_hints(channels, "weight", Init::Const(1.0))?;
        let bias = vb.get_with_hints(channels, "bias", Init::Const(0.0))?;
        Ok(Self {
            norm: GroupNorm::new(weight, bias, channels, groups, eps)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        map_frames(input, |frame| frame.apply(&self.norm))
    }
}

#[derive(Clone, Debug)]
struct H3ResnetBlock3d {
    norm1: TemporalIsolatedGroupNorm,
    conv1: H3CausalConv3d,
    norm2: TemporalIsolatedGroupNorm,
    conv2: H3CausalConv3d,
    shortcut: Option<H3CausalConv3d>,
}

impl H3ResnetBlock3d {
    fn load(
        vb: VarBuilder,
        input_channels: usize,
        output_channels: usize,
        groups: usize,
        eps: f64,
        shortcut_name: &str,
    ) -> Result<Self> {
        Ok(Self {
            norm1: TemporalIsolatedGroupNorm::load(vb.pp("norm1"), groups, input_channels, eps)?,
            conv1: H3CausalConv3d::load(
                vb.pp("conv1"),
                input_channels,
                output_channels,
                3,
                1,
                1,
                1,
            )?,
            norm2: TemporalIsolatedGroupNorm::load(vb.pp("norm2"), groups, output_channels, eps)?,
            conv2: H3CausalConv3d::load(
                vb.pp("conv2"),
                output_channels,
                output_channels,
                3,
                1,
                1,
                1,
            )?,
            shortcut: if input_channels == output_channels {
                None
            } else {
                Some(H3CausalConv3d::load(
                    vb.pp(shortcut_name),
                    input_channels,
                    output_channels,
                    1,
                    1,
                    1,
                    0,
                )?)
            },
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self
            .conv1
            .forward(&ops::silu(&self.norm1.forward(input)?)?)?;
        let hidden = self
            .conv2
            .forward(&ops::silu(&self.norm2.forward(&hidden)?)?)?;
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input)?,
            None => input.clone(),
        };
        hidden.add(&residual)
    }
}

#[derive(Clone, Debug)]
struct H3Downsample3d {
    spatial_stride: usize,
    conv: H3CausalConv3d,
}

impl H3Downsample3d {
    fn load(
        vb: VarBuilder,
        channels: usize,
        temporal_stride: usize,
        spatial_stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            spatial_stride,
            conv: H3CausalConv3d::load(
                vb,
                channels,
                channels,
                3,
                temporal_stride,
                spatial_stride,
                0,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = if self.spatial_stride == 2 {
            pad_bottom_right_reflect(input)?
        } else {
            input.clone()
        };
        self.conv.forward(&input)
    }
}

#[derive(Clone, Debug)]
struct H3EncoderLevel {
    blocks: Vec<H3ResnetBlock3d>,
    downsample: Option<H3Downsample3d>,
}

#[derive(Clone, Debug)]
struct H3Encoder3d {
    conv_in: H3CausalConv3d,
    levels: Vec<H3EncoderLevel>,
    norm_out: TemporalIsolatedGroupNorm,
    conv_out: H3CausalConv3d,
}

impl H3Encoder3d {
    fn load(
        config: &MiniMaxH3VisualVaeConfig,
        vb: VarBuilder,
        layout: WeightLayout,
    ) -> Result<Self> {
        let conv_in = H3CausalConv3d::load(
            vb.pp("encoder.conv_in"),
            config.in_channels,
            config.block_out_channels[0],
            3,
            1,
            1,
            1,
        )?;
        let mut levels = Vec::with_capacity(config.block_out_channels.len());
        for level in 0..config.block_out_channels.len() {
            let output_channels = config.block_out_channels[level];
            let input_channels = if level == 0 {
                output_channels
            } else {
                config.block_out_channels[level - 1]
            };
            let mut blocks = Vec::with_capacity(config.layers_per_block);
            for block in 0..config.layers_per_block {
                let block_input = if block == 0 {
                    input_channels
                } else {
                    output_channels
                };
                let prefix = match layout {
                    WeightLayout::Diffusers => {
                        format!("encoder.down_blocks.{level}.resnets.{block}")
                    }
                    WeightLayout::OfficialComfySource => {
                        format!("encoder.down.{level}.block.{block}")
                    }
                };
                blocks.push(H3ResnetBlock3d::load(
                    vb.pp(prefix),
                    block_input,
                    output_channels,
                    config.norm_num_groups,
                    config.norm_eps,
                    match layout {
                        WeightLayout::Diffusers => "conv_shortcut",
                        WeightLayout::OfficialComfySource => "nin_shortcut",
                    },
                )?);
            }
            let downsample = if config.spatial_downsample_factors[level]
                * config.temporal_downsample_factors[level]
                > 1
            {
                let prefix = match layout {
                    WeightLayout::Diffusers => {
                        format!("encoder.down_blocks.{level}.downsamplers.0.conv")
                    }
                    WeightLayout::OfficialComfySource => {
                        format!("encoder.down.{level}.downsample.conv")
                    }
                };
                Some(H3Downsample3d::load(
                    vb.pp(prefix),
                    output_channels,
                    config.temporal_downsample_factors[level],
                    config.spatial_downsample_factors[level],
                )?)
            } else {
                None
            };
            levels.push(H3EncoderLevel { blocks, downsample });
        }
        let final_channels =
            config.block_out_channels.last().copied().ok_or_else(|| {
                candle::Error::Msg("H3 visual VAE channel schedule is empty".into())
            })?;
        Ok(Self {
            conv_in,
            levels,
            norm_out: TemporalIsolatedGroupNorm::load(
                vb.pp("encoder.norm_out"),
                config.norm_num_groups,
                final_channels,
                config.norm_eps,
            )?,
            conv_out: H3CausalConv3d::load(
                vb.pp("encoder.conv_out"),
                final_channels,
                config.latent_channels * 2,
                3,
                1,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = self.conv_in.forward(input)?;
        for level in &self.levels {
            for block in &level.blocks {
                hidden = block.forward(&hidden)?;
            }
            if let Some(downsample) = &level.downsample {
                hidden = downsample.forward(&hidden)?;
            }
        }
        self.conv_out
            .forward(&ops::silu(&self.norm_out.forward(&hidden)?)?)
    }
}

#[derive(Clone, Debug)]
struct Fp32RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl Fp32RmsNorm {
    fn load_affine(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: Some(
                vb.get_with_hints(dim, "weight", Init::Const(1.0))?
                    .to_dtype(DType::F32)?,
            ),
            eps,
        })
    }

    fn no_affine(eps: f64) -> Self {
        Self { weight: None, eps }
    }

    fn forward(&self, input: &Tensor, output_dtype: DType) -> Result<Tensor> {
        let input32 = input.to_dtype(DType::F32)?;
        let variance = input32.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = input32.broadcast_div(&variance.affine(1.0, self.eps)?.sqrt()?)?;
        let normalized = match &self.weight {
            Some(weight) => normalized.broadcast_mul(weight)?,
            None => normalized,
        };
        normalized.to_dtype(output_dtype)
    }
}

#[derive(Clone, Debug)]
struct Fp32LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl Fp32LayerNorm {
    fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get_with_hints(dim, "weight", Init::Const(1.0))?
                .to_dtype(DType::F32)?,
            bias: vb
                .get_with_hints(dim, "bias", Init::Const(0.0))?
                .to_dtype(DType::F32)?,
            eps,
        })
    }

    fn forward(&self, input: &Tensor, output_dtype: DType) -> Result<Tensor> {
        let input = input.to_dtype(DType::F32)?;
        let mean = input.mean_keepdim(D::Minus1)?;
        let centered = input.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered
            .broadcast_div(&variance.affine(1.0, self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?
            .to_dtype(output_dtype)
    }
}

#[derive(Clone, Debug)]
struct H3RotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl H3RotaryEmbedding {
    fn new(dim: usize, theta: f64) -> Result<Self> {
        if !dim.is_multiple_of(6) {
            bail!("H3 decoder rotary dimension {dim} must be divisible by six")
        }
        let pairs_per_axis = dim / 6;
        let inv_freq = (0..pairs_per_axis)
            .map(|index| {
                let exponent = index as f64 * 6.0 / dim as f64;
                theta.powf(-exponent) as f32
            })
            .collect();
        Ok(Self { inv_freq })
    }

    fn cos_sin(
        &self,
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
        suffix: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let mut positions = Vec::with_capacity((frames * height * width + suffix) * 3);
        for temporal in 0..frames {
            let t = 2.0 * ((temporal as f32 + 0.5) / frames as f32) - 1.0;
            for y in 0..height {
                let h = 2.0 * ((y as f32 + 0.5) / height as f32) - 1.0;
                for x in 0..width {
                    let w = 2.0 * ((x as f32 + 0.5) / width as f32) - 1.0;
                    positions.extend_from_slice(&[t, h, w]);
                }
            }
        }
        positions.resize(positions.len() + suffix * 3, 0.0);
        let sequence = frames * height * width + suffix;
        let mut angles = Vec::with_capacity(sequence * 3 * self.inv_freq.len() * 2);
        for token in 0..sequence {
            let coordinates = &positions[token * 3..token * 3 + 3];
            let mut half = Vec::with_capacity(3 * self.inv_freq.len());
            for coordinate in coordinates {
                for frequency in &self.inv_freq {
                    half.push(2.0 * std::f32::consts::PI * coordinate * frequency);
                }
            }
            angles.extend_from_slice(&half);
            angles.extend_from_slice(&half);
        }
        let angle = Tensor::from_vec(angles, (1, 1, sequence, 6 * self.inv_freq.len()), device)?;
        let cos = angle.cos()?.to_dtype(dtype)?.repeat((batch, 1, 1, 1))?;
        let sin = angle.sin()?.to_dtype(dtype)?.repeat((batch, 1, 1, 1))?;
        Ok((cos, sin))
    }
}

#[derive(Clone, Debug)]
struct H3Attention {
    qkv: MixedLinear,
    qkv_layout: WeightLayout,
    output: MixedLinear,
    q_norm: Fp32RmsNorm,
    k_norm: Fp32RmsNorm,
    heads: usize,
    head_dim: usize,
    compute_dtype: DType,
}

impl H3Attention {
    fn load(
        vb: VarBuilder,
        layout: WeightLayout,
        dim: usize,
        heads: usize,
        head_dim: usize,
        eps: f64,
        compute_dtype: DType,
    ) -> Result<Self> {
        let qkv = match layout {
            WeightLayout::Diffusers => {
                let mut weights = Vec::with_capacity(3);
                let mut biases = Vec::with_capacity(3);
                for projection in ["to_q", "to_k", "to_v"] {
                    weights.push(vb.pp(projection).get_with_hints(
                        (dim, dim),
                        "weight",
                        TEST_WEIGHT_INIT,
                    )?);
                    biases.push(
                        vb.pp(projection)
                            .get_with_hints(dim, "bias", Init::Const(0.0))?,
                    );
                }
                MixedLinear::new(
                    cat_tensors(&weights, 0)?,
                    Some(cat_tensors(&biases, 0)?),
                    compute_dtype,
                )?
            }
            WeightLayout::OfficialComfySource => {
                MixedLinear::load(vb.pp("to_qkv"), dim, dim * 3, compute_dtype)?
            }
        };
        let output_prefix = match layout {
            WeightLayout::Diffusers => "to_out.0",
            WeightLayout::OfficialComfySource => "to_out",
        };
        Ok(Self {
            qkv,
            qkv_layout: layout,
            output: MixedLinear::load(vb.pp(output_prefix), dim, dim, compute_dtype)?,
            q_norm: Fp32RmsNorm::no_affine(eps),
            k_norm: Fp32RmsNorm::no_affine(eps),
            heads,
            head_dim,
            compute_dtype,
        })
    }

    fn forward(&self, input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (batch, sequence, dim) = input.dims3()?;
        let qkv = self.qkv.forward(input)?;
        let (query, key, value) =
            split_qkv_layout(&qkv, self.qkv_layout, self.heads, self.head_dim)?;
        let mut query = self.q_norm.forward(&query, self.compute_dtype)?;
        let mut key = self.k_norm.forward(&key, self.compute_dtype)?;
        let value = value.to_dtype(self.compute_dtype)?;
        let rotary_dim = cos.dim(D::Minus1)?;
        query = apply_rope(&query, cos, sin, rotary_dim)?;
        key = apply_rope(&key, cos, sin, rotary_dim)?;
        let scores = query
            .matmul(&key.transpose(2, 3)?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attention = ops::softmax_last_dim(&scores)?;
        let hidden = attention
            .matmul(&value)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, sequence, dim))?;
        self.output.forward(&hidden)
    }
}

fn split_qkv_layout(
    qkv: &Tensor,
    layout: WeightLayout,
    heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, sequence, width) = qkv.dims3()?;
    let dim = heads
        .checked_mul(head_dim)
        .ok_or_else(|| candle::Error::Msg("H3 attention width overflow".into()))?;
    let fused_dim = dim
        .checked_mul(3)
        .ok_or_else(|| candle::Error::Msg("H3 fused QKV width overflow".into()))?;
    if width != fused_dim {
        bail!("H3 fused QKV width {width} does not match 3 * {dim}")
    }
    let project = |tensor: Tensor| -> Result<Tensor> {
        tensor
            .reshape((batch, sequence, heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    };
    match layout {
        WeightLayout::Diffusers => Ok((
            project(qkv.narrow(2, 0, dim)?)?,
            project(qkv.narrow(2, dim, dim)?)?,
            project(qkv.narrow(2, 2 * dim, dim)?)?,
        )),
        WeightLayout::OfficialComfySource => {
            let fused_head_dim = head_dim
                .checked_mul(3)
                .ok_or_else(|| candle::Error::Msg("H3 fused head width overflow".into()))?;
            let qkv = qkv.reshape((batch, sequence, heads, fused_head_dim))?;
            Ok((
                qkv.narrow(3, 0, head_dim)?
                    .permute((0, 2, 1, 3))?
                    .contiguous()?,
                qkv.narrow(3, head_dim, head_dim)?
                    .permute((0, 2, 1, 3))?
                    .contiguous()?,
                qkv.narrow(3, 2 * head_dim, head_dim)?
                    .permute((0, 2, 1, 3))?
                    .contiguous()?,
            ))
        }
    }
}

fn apply_rope(input: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<Tensor> {
    let head_dim = input.dim(D::Minus1)?;
    if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        bail!("H3 rotary width is invalid for attention head")
    }
    let rotary = input.narrow(D::Minus1, 0, rotary_dim)?;
    let half = rotary_dim / 2;
    let first = rotary.narrow(D::Minus1, 0, half)?;
    let second = rotary.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    let rotary = rotary
        .broadcast_mul(cos)?
        .add(&rotated.broadcast_mul(sin)?)?;
    if rotary_dim == head_dim {
        Ok(rotary)
    } else {
        Tensor::cat(
            &[
                &rotary,
                &input.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?,
            ],
            D::Minus1,
        )
    }
}

#[derive(Clone, Debug)]
struct H3FeedForward {
    input: MixedLinear,
    output: MixedLinear,
    inner: usize,
    gate_first: bool,
}

impl H3FeedForward {
    fn load(
        vb: VarBuilder,
        layout: WeightLayout,
        dim: usize,
        mult: usize,
        compute_dtype: DType,
    ) -> Result<Self> {
        let inner = dim
            .checked_mul(mult)
            .ok_or_else(|| candle::Error::Msg("H3 feed-forward width overflow".into()))?;
        let gated_inner = inner
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("H3 gated feed-forward width overflow".into()))?;
        let (input_name, output_name) = match layout {
            WeightLayout::Diffusers => ("net.0.proj", "net.2"),
            WeightLayout::OfficialComfySource => ("w1", "w2"),
        };
        Ok(Self {
            input: MixedLinear::load(vb.pp(input_name), dim, gated_inner, compute_dtype)?,
            output: MixedLinear::load(vb.pp(output_name), inner, dim, compute_dtype)?,
            inner,
            gate_first: layout == WeightLayout::OfficialComfySource,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let projected = self.input.forward(input)?;
        let (gate, value) = split_gate_value(&projected, self.inner, self.gate_first)?;
        self.output.forward(&ops::silu(&gate)?.mul(&value)?)
    }
}

fn split_gate_value(
    projected: &Tensor,
    inner: usize,
    gate_first: bool,
) -> Result<(Tensor, Tensor)> {
    let expected = inner
        .checked_mul(2)
        .ok_or_else(|| candle::Error::Msg("H3 gated feed-forward width overflow".into()))?;
    if projected.dim(D::Minus1)? != expected {
        bail!("H3 feed-forward projection width does not match its inner dimension")
    }
    let first = projected.narrow(D::Minus1, 0, inner)?;
    let second = projected.narrow(D::Minus1, inner, inner)?;
    Ok(if gate_first {
        (first, second)
    } else {
        (second, first)
    })
}

#[derive(Clone, Debug)]
struct H3TransformerBlock {
    norm1: Fp32RmsNorm,
    attention: H3Attention,
    scale1: Fp32Vector,
    norm2: Fp32RmsNorm,
    feed_forward: H3FeedForward,
    scale2: Fp32Vector,
}

impl H3TransformerBlock {
    fn load(
        vb: VarBuilder,
        layout: WeightLayout,
        dim: usize,
        heads: usize,
        head_dim: usize,
        ffn_mult: usize,
        eps: f64,
        compute_dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            norm1: Fp32RmsNorm::load_affine(vb.pp("norm1"), dim, eps)?,
            attention: H3Attention::load(
                vb.pp("attn"),
                layout,
                dim,
                heads,
                head_dim,
                eps,
                compute_dtype,
            )?,
            scale1: Fp32Vector::new(vb.get_with_hints(dim, "scale1", Init::Const(0.0))?)?,
            norm2: Fp32RmsNorm::load_affine(vb.pp("norm2"), dim, eps)?,
            feed_forward: H3FeedForward::load(vb.pp("ff"), layout, dim, ffn_mult, compute_dtype)?,
            scale2: Fp32Vector::new(vb.get_with_hints(dim, "scale2", Init::Const(0.0))?)?,
        })
    }

    fn forward(&self, input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Autocast lowers the projection sites, but the released learned
        // scales are FP32. Their pointwise products promote the residual
        // stream back to FP32 after every branch.
        let norm = self.norm1.forward(input, DType::F32)?;
        let attention = self
            .attention
            .forward(&norm, cos, sin)?
            .to_dtype(DType::F32)?
            .broadcast_mul(self.scale1.value())?;
        let hidden = input.to_dtype(DType::F32)?.add(&attention)?;
        let norm = self.norm2.forward(&hidden, DType::F32)?;
        hidden.add(
            &self
                .feed_forward
                .forward(&norm)?
                .to_dtype(DType::F32)?
                .broadcast_mul(self.scale2.value())?,
        )
    }
}

#[derive(Clone, Debug)]
struct H3VitDecoder3d {
    projection_in: MixedLinear,
    register_tokens: Fp32Vector,
    blocks: Vec<H3TransformerBlock>,
    norm_out: Fp32LayerNorm,
    projection_out: MixedLinear,
    rope: H3RotaryEmbedding,
    num_register_tokens: usize,
    heads: usize,
    head_dim: usize,
    patch_size: usize,
    patch_size_t: usize,
    out_channels: usize,
    compute_dtype: DType,
}

impl H3VitDecoder3d {
    fn load(
        config: &MiniMaxH3VisualVaeConfig,
        vb: VarBuilder,
        layout: WeightLayout,
        compute_dtype: DType,
    ) -> Result<Self> {
        let dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
        let projection_in_name = match layout {
            WeightLayout::Diffusers => "proj_in",
            WeightLayout::OfficialComfySource => "x_embedder",
        };
        let mut blocks = Vec::with_capacity(config.decoder_num_layers);
        for block in 0..config.decoder_num_layers {
            blocks.push(H3TransformerBlock::load(
                vb.pp(format!("transformer_blocks.{block}")),
                layout,
                dim,
                config.decoder_num_attention_heads,
                config.decoder_attention_head_dim,
                config.decoder_ffn_mult,
                config.decoder_norm_eps,
                compute_dtype,
            )?);
        }
        let patch_size = config.spatial_compression_ratio();
        let patch_size_t = config.temporal_compression_ratio();
        let output = config.out_channels * patch_size_t * patch_size * patch_size;
        Ok(Self {
            projection_in: MixedLinear::load(
                vb.pp(projection_in_name),
                config.latent_channels,
                dim,
                compute_dtype,
            )?,
            register_tokens: Fp32Vector::new(vb.get_with_hints(
                (1, config.decoder_num_register_tokens, dim),
                "register_tokens",
                TEST_WEIGHT_INIT,
            )?)?,
            blocks,
            norm_out: Fp32LayerNorm::load(vb.pp("norm_out"), dim, config.decoder_norm_eps)?,
            projection_out: MixedLinear::load(vb.pp("proj_out"), dim, output, compute_dtype)?,
            rope: H3RotaryEmbedding::new(
                (config.decoder_attention_head_dim as f64 * config.decoder_rope_dim_ratio) as usize,
                config.decoder_rope_theta,
            )?,
            num_register_tokens: config.decoder_num_register_tokens,
            heads: config.decoder_num_attention_heads,
            head_dim: config.decoder_attention_head_dim,
            patch_size,
            patch_size_t,
            out_channels: config.out_channels,
            compute_dtype,
        })
    }

    fn forward(&self, input: &Tensor, observer: &mut dyn VisualVaeObserver) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = input.dims5()?;
        let patches = frames * height * width;
        let mut hidden = self.projection_in.forward(
            &input
                .permute((0, 2, 3, 4, 1))?
                .contiguous()?
                .reshape((batch, patches, channels))?,
        )?;
        // `proj_in` is autocast-eligible, then concatenation with the released
        // FP32 register parameters promotes the decoder residual stream.
        hidden = hidden.to_dtype(DType::F32)?;
        let registers = self.register_tokens.value().repeat((batch, 1, 1))?;
        let zero = Tensor::zeros(
            (batch, 1, self.heads * self.head_dim),
            DType::F32,
            input.device(),
        )?;
        hidden = Tensor::cat(&[&hidden, &registers, &zero], 1)?;
        let (cos, sin) = self.rope.cos_sin(
            batch,
            frames,
            height,
            width,
            self.num_register_tokens + 1,
            input.device(),
            self.compute_dtype,
        )?;
        for (index, block) in self.blocks.iter().enumerate() {
            observer.checkpoint(VisualVaeEvent {
                phase: VisualVaePhase::DecoderBlock,
                completed: index,
                total: self.blocks.len(),
            })?;
            hidden = block.forward(&hidden, &cos, &sin)?;
        }
        observer.checkpoint(VisualVaeEvent {
            phase: VisualVaePhase::DecoderBlock,
            completed: self.blocks.len(),
            total: self.blocks.len(),
        })?;
        let hidden = self
            .projection_out
            .forward(&self.norm_out.forward(&hidden, DType::F32)?)?
            .narrow(1, 0, patches)?;
        let output_order: &[usize] = &[0, 4, 1, 5, 2, 6, 3, 7];
        hidden
            .reshape(&[
                batch,
                frames,
                height,
                width,
                self.out_channels,
                self.patch_size_t,
                self.patch_size,
                self.patch_size,
            ])?
            .permute(output_order)?
            .contiguous()?
            .reshape((
                batch,
                self.out_channels,
                frames * self.patch_size_t,
                height * self.patch_size,
                width * self.patch_size,
            ))
    }
}

#[derive(Clone, Debug)]
pub struct MiniMaxH3VisualVae {
    config: MiniMaxH3VisualVaeConfig,
    encoder: H3Encoder3d,
    quant_conv: PointwiseConv3d,
    post_quant_conv: PointwiseConv3d,
    decoder: H3VitDecoder3d,
    compute_dtype: DType,
}

impl MiniMaxH3VisualVae {
    pub fn new(
        config: MiniMaxH3VisualVaeConfig,
        vb: VarBuilder,
        layout: WeightLayout,
        decode_policy: DecodeComputePolicy,
    ) -> Result<Self> {
        config.validate()?;
        if vb.dtype() != DType::F32 {
            bail!(
                "H3 visual VAE VarBuilder must expose FP32 stored weights, got {:?}",
                vb.dtype()
            )
        }
        let compute_dtype = decode_policy.effective_dtype(vb.device());
        let encoder = H3Encoder3d::load(&config, vb.clone(), layout)?;
        let quant_conv = PointwiseConv3d::load(
            vb.pp("quant_conv"),
            config.latent_channels * 2,
            config.latent_channels * 2,
            DType::F32,
        )?;
        let post_quant_conv = PointwiseConv3d::load(
            vb.pp("post_quant_conv"),
            config.latent_channels,
            config.latent_channels,
            compute_dtype,
        )?;
        let decoder = H3VitDecoder3d::load(&config, vb.pp("decoder"), layout, compute_dtype)?;
        Ok(Self {
            config,
            encoder,
            quant_conv,
            post_quant_conv,
            decoder,
            compute_dtype,
        })
    }

    pub fn config(&self) -> &MiniMaxH3VisualVaeConfig {
        &self.config
    }

    pub fn stored_weight_dtype(&self) -> DType {
        self.decoder.projection_in.stored_dtype()
    }

    pub fn decode_compute_dtype(&self) -> DType {
        self.compute_dtype
    }

    pub fn encode_condition_u8(
        &self,
        pixels: Uint8RgbPixels,
        mode: ConditionEncodeMode,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        self.encode_condition_imagenet(pixels.into_imagenet()?, mode, observer)
    }

    pub fn encode_condition_unit(
        &self,
        pixels: UnitRgbPixels,
        mode: ConditionEncodeMode,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        self.encode_condition_imagenet(pixels.into_imagenet()?, mode, observer)
    }

    pub fn encode_condition_signed(
        &self,
        pixels: SignedRgbPixels,
        mode: ConditionEncodeMode,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        self.encode_condition_imagenet(pixels.into_imagenet()?, mode, observer)
    }

    fn encode_condition_imagenet(
        &self,
        pixels: ImageNetPixels,
        mode: ConditionEncodeMode,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        let moments = self.encode_moments(&pixels.into_tensor(), observer)?;
        DiagonalGaussianDistribution::from_moments(&moments)?.condition_latents(
            mode,
            &self.config.latents_mean,
            &self.config.latents_std,
        )
    }

    fn encode_moments(
        &self,
        pixels: &Tensor,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = pixels.dims5()?;
        if batch != 1 || channels != 3 || pixels.dtype() != DType::F32 {
            bail!("H3 encoder expects ImageNet-normalized FP32 [1,3,T,H,W]")
        }
        let spatial_ratio = self.config.spatial_compression_ratio();
        if height % spatial_ratio != 0 || width % spatial_ratio != 0 {
            bail!(
                "H3 encoder dimensions {width}x{height} must be divisible by spatial compression {spatial_ratio}"
            )
        }
        let plan = self.config.temporal_geometry().encode_plan(frames)?;
        if plan.single_frame {
            return self.encode_clip(pixels, observer);
        }
        let mut chunks = Vec::with_capacity(plan.chunks.len());
        for (index, chunk) in plan.chunks.iter().enumerate() {
            observer.checkpoint(VisualVaeEvent {
                phase: VisualVaePhase::EncodeChunk,
                completed: index,
                total: plan.chunks.len(),
            })?;
            let mut clip = pixels.narrow(2, chunk.input_start, chunk.valid_frames)?;
            if chunk.repeated_tail_frames > 0 {
                let tail = clip.narrow(2, chunk.valid_frames - 1, 1)?.repeat((
                    1,
                    1,
                    chunk.repeated_tail_frames,
                    1,
                    1,
                ))?;
                clip = Tensor::cat(&[&clip, &tail], 2)?;
            }
            chunks.push(self.encode_clip(&clip, observer)?);
        }
        observer.checkpoint(VisualVaeEvent {
            phase: VisualVaePhase::EncodeChunk,
            completed: plan.chunks.len(),
            total: plan.chunks.len(),
        })?;
        let moments = cat_tensors(&chunks, 2)?;
        moments.narrow(2, 0, plan.latent_frames)
    }

    fn encode_clip(&self, pixels: &Tensor, observer: &mut dyn VisualVaeObserver) -> Result<Tensor> {
        if !self.config.tiling {
            return self
                .quant_conv
                .forward(&self.encoder.forward(pixels)?)
                .and_then(|tensor| tensor.to_dtype(DType::F32));
        }
        let height_plan = SpatialTilePlan::new(
            pixels.dim(3)?,
            self.config.tile_size,
            self.config.tile_overlap_min,
            self.config.spatial_compression_ratio(),
        )?;
        let width_plan = SpatialTilePlan::new(
            pixels.dim(4)?,
            self.config.tile_size,
            self.config.tile_overlap_min,
            self.config.spatial_compression_ratio(),
        )?;
        let total = height_plan.starts.len() * width_plan.starts.len();
        let mut rows = Vec::with_capacity(height_plan.starts.len());
        let mut completed = 0;
        for (&y, &tile_height) in height_plan.starts.iter().zip(&height_plan.lengths) {
            let mut row = Vec::with_capacity(width_plan.starts.len());
            for (&x, &tile_width) in width_plan.starts.iter().zip(&width_plan.lengths) {
                observer.checkpoint(VisualVaeEvent {
                    phase: VisualVaePhase::EncodeTile,
                    completed,
                    total,
                })?;
                let tile = pixels.narrow(3, y, tile_height)?.narrow(4, x, tile_width)?;
                row.push(self.quant_conv.forward(&self.encoder.forward(&tile)?)?);
                completed += 1;
            }
            rows.push(row);
        }
        observer.checkpoint(VisualVaeEvent {
            phase: VisualVaePhase::EncodeTile,
            completed,
            total,
        })?;
        let ratio = self.config.spatial_compression_ratio();
        stitch_tiles(
            rows,
            &height_plan
                .overlaps
                .iter()
                .map(|overlap| overlap / ratio)
                .collect::<Vec<_>>(),
            &width_plan
                .overlaps
                .iter()
                .map(|overlap| overlap / ratio)
                .collect::<Vec<_>>(),
        )
    }

    pub fn decode_normalized(
        &self,
        latents: &Tensor,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        let mut sink = CollectDecodeSink::default();
        self.decode_normalized_to_sink(latents, &mut sink, observer)?;
        sink.finish()
    }

    pub fn decode_normalized_to_sink(
        &self,
        normalized_latents: &Tensor,
        sink: &mut dyn DecodeSink,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<()> {
        let (batch, channels, latent_frames, height, width) = normalized_latents.dims5()?;
        if batch == 0 || channels != self.config.latent_channels || height == 0 || width == 0 {
            bail!("H3 decoder latent shape does not match the visual VAE contract")
        }
        if !matches!(
            normalized_latents.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) {
            bail!("H3 decoder latents must use a floating-point dtype")
        }
        let latents = denormalize_latents(
            normalized_latents,
            &self.config.latents_mean,
            &self.config.latents_std,
        )?
        .to_dtype(self.compute_dtype)?;
        let plan = self.config.temporal_geometry().decode_plan(latent_frames)?;
        if latent_frames == 1 {
            let decoded = self.decode_clip(&latents, observer)?;
            let last = decoded.narrow(2, decoded.dim(2)? - 1, 1)?;
            sink.write(0, &denormalize_imagenet(&last)?)?;
            return Ok(());
        }
        let padded = if plan.repeated_tail_tokens > 0 {
            let tail = latents.narrow(2, latent_frames - 1, 1)?.repeat((
                1,
                1,
                plan.repeated_tail_tokens,
                1,
                1,
            ))?;
            Tensor::cat(&[&latents, &tail], 2)?
        } else {
            latents
        };
        let chunk_decoded_frames = plan.tokens_per_chunk * self.config.temporal_compression_ratio();
        let split_count = usize::from(self.config.token_drop > 0) + 1;
        let mut overlap: Option<Tensor> = None;
        let mut written = 0;
        for (index, chunk) in plan.chunks.iter().enumerate() {
            observer.checkpoint(VisualVaeEvent {
                phase: VisualVaePhase::DecodeChunk,
                completed: index,
                total: plan.chunks.len(),
            })?;
            let clip = padded.narrow(2, chunk.latent_start, chunk.latent_frames)?;
            let decoded = self.decode_clip(&clip, observer)?;
            for split in 0..split_count {
                let start = split * chunk_decoded_frames;
                let end = (start + chunk_decoded_frames).min(decoded.dim(2)?);
                if end <= start + plan.frame_pre_padding {
                    continue;
                }
                let mut part = decoded.narrow(2, start, end - start)?.narrow(
                    2,
                    plan.frame_pre_padding,
                    end - start - plan.frame_pre_padding,
                )?;
                if split == 0 {
                    if let Some(previous) = overlap.take() {
                        part = blend(&previous, &part, plan.frame_overlap, 2)?;
                    }
                    write_limited(
                        sink,
                        &denormalize_imagenet(&part)?,
                        &mut written,
                        plan.output_frames,
                    )?;
                } else {
                    overlap = Some(part.contiguous()?);
                }
            }
        }
        if let Some(overlap) = overlap {
            write_limited(
                sink,
                &denormalize_imagenet(&overlap)?,
                &mut written,
                plan.output_frames,
            )?;
        }
        observer.checkpoint(VisualVaeEvent {
            phase: VisualVaePhase::DecodeChunk,
            completed: plan.chunks.len(),
            total: plan.chunks.len(),
        })?;
        if written != plan.output_frames {
            bail!(
                "H3 streamed decode wrote {written} frames but planned {}",
                plan.output_frames
            )
        }
        Ok(())
    }

    fn decode_clip(
        &self,
        latents: &Tensor,
        observer: &mut dyn VisualVaeObserver,
    ) -> Result<Tensor> {
        if !self.config.tiling {
            return self
                .decoder
                .forward(&self.post_quant_conv.forward(latents)?, observer);
        }
        let ratio = self.config.spatial_compression_ratio();
        let pixel_height = latents
            .dim(3)?
            .checked_mul(ratio)
            .ok_or_else(|| candle::Error::Msg("H3 decoded height overflow".into()))?;
        let pixel_width = latents
            .dim(4)?
            .checked_mul(ratio)
            .ok_or_else(|| candle::Error::Msg("H3 decoded width overflow".into()))?;
        let height_plan = SpatialTilePlan::new(
            pixel_height,
            self.config.tile_size,
            self.config.tile_overlap_min,
            ratio,
        )?;
        let width_plan = SpatialTilePlan::new(
            pixel_width,
            self.config.tile_size,
            self.config.tile_overlap_min,
            ratio,
        )?;
        let total = height_plan.starts.len() * width_plan.starts.len();
        let mut canvas: Option<Tensor> = None;
        let mut row_tails: Vec<Tensor> = Vec::new();
        let mut output_y = 0;
        let mut completed = 0;
        for (row_index, (&y, &tile_height)) in height_plan
            .starts
            .iter()
            .zip(&height_plan.lengths)
            .enumerate()
        {
            let mut new_row_tails = Vec::with_capacity(width_plan.starts.len());
            let mut left_tail: Option<Tensor> = None;
            let mut output_x = 0;
            let mut written_height = 0;
            for (column_index, (&x, &tile_width)) in width_plan
                .starts
                .iter()
                .zip(&width_plan.lengths)
                .enumerate()
            {
                observer.checkpoint(VisualVaeEvent {
                    phase: VisualVaePhase::DecodeTile,
                    completed,
                    total,
                })?;
                let tile = latents.narrow(3, y / ratio, tile_height / ratio)?.narrow(
                    4,
                    x / ratio,
                    tile_width / ratio,
                )?;
                let mut tile = self
                    .decoder
                    .forward(&self.post_quant_conv.forward(&tile)?, observer)?;

                // Preserve raw tails before blending, exactly as ComfyUI's
                // preallocated tiled decoder does. No full decoded tile row is
                // retained after its cropped body is copied into the canvas.
                if row_index < height_plan.starts.len() - 1 {
                    let overlap = height_plan.overlaps[row_index];
                    new_row_tails.push(
                        tile.narrow(3, tile.dim(3)? - overlap, overlap)?
                            .contiguous()?,
                    );
                }
                let next_left_tail = if column_index < width_plan.starts.len() - 1 {
                    let overlap = width_plan.overlaps[column_index];
                    Some(
                        tile.narrow(4, tile.dim(4)? - overlap, overlap)?
                            .contiguous()?,
                    )
                } else {
                    None
                };
                if row_index > 0 {
                    tile = blend(
                        &row_tails[column_index],
                        &tile,
                        height_plan.overlaps[row_index - 1],
                        3,
                    )?;
                }
                if column_index > 0 {
                    tile = blend(
                        left_tail.as_ref().ok_or_else(|| {
                            candle::Error::Msg("H3 decode left-tail state is missing".into())
                        })?,
                        &tile,
                        width_plan.overlaps[column_index - 1],
                        4,
                    )?;
                }
                left_tail = next_left_tail;
                if row_index < height_plan.starts.len() - 1 {
                    tile = tile.narrow(3, 0, tile.dim(3)? - height_plan.overlaps[row_index])?;
                }
                if column_index < width_plan.starts.len() - 1 {
                    tile = tile.narrow(4, 0, tile.dim(4)? - width_plan.overlaps[column_index])?;
                }
                let tile = tile.contiguous()?;
                written_height = tile.dim(3)?;
                let output = match &canvas {
                    Some(canvas) => canvas,
                    None => canvas.insert(Tensor::zeros(
                        (
                            tile.dim(0)?,
                            tile.dim(1)?,
                            tile.dim(2)?,
                            pixel_height,
                            pixel_width,
                        ),
                        tile.dtype(),
                        tile.device(),
                    )?),
                };
                copy_spatial_tile(output, &tile, output_y, output_x)?;
                output_x += tile.dim(4)?;
                completed += 1;
            }
            if output_x != pixel_width {
                bail!("H3 tiled decode row wrote {output_x}px, expected {pixel_width}px")
            }
            row_tails = new_row_tails;
            output_y += written_height;
        }
        observer.checkpoint(VisualVaeEvent {
            phase: VisualVaePhase::DecodeTile,
            completed,
            total,
        })?;
        if output_y != pixel_height {
            bail!("H3 tiled decode wrote {output_y}px, expected {pixel_height}px")
        }
        canvas.ok_or_else(|| candle::Error::Msg("H3 tiled decode produced no tiles".into()))
    }
}

fn copy_spatial_tile(canvas: &Tensor, tile: &Tensor, y: usize, x: usize) -> Result<()> {
    let (batch, channels, frames, canvas_height, canvas_width) = canvas.dims5()?;
    let (tile_batch, tile_channels, tile_frames, tile_height, tile_width) = tile.dims5()?;
    if (batch, channels, frames) != (tile_batch, tile_channels, tile_frames)
        || y + tile_height > canvas_height
        || x + tile_width > canvas_width
    {
        bail!("H3 decoded tile does not fit its output canvas")
    }
    let flat_canvas = canvas.flatten_all()?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            for frame in 0..frames {
                for row in 0..tile_height {
                    let destination = (((batch_index * channels + channel) * frames + frame)
                        * canvas_height
                        + y
                        + row)
                        * canvas_width
                        + x;
                    let source = tile
                        .i((batch_index, channel, frame, row, ..))?
                        .contiguous()?
                        .flatten_all()?;
                    flat_canvas.slice_set(&source, 0, destination)?;
                }
            }
        }
    }
    Ok(())
}

fn blend(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    if extent == 0 {
        return Ok(b.clone());
    }
    let extent = extent.min(a.dim(dim)?).min(b.dim(dim)?);
    let positions = Tensor::arange(0u32, extent as u32, b.device())?.to_dtype(b.dtype())?;
    let mut shape = vec![1usize; b.rank()];
    shape[dim] = extent;
    let weight_a = positions
        .affine(-1.0 / extent as f64, 1.0)?
        .reshape(shape.clone())?;
    let weight_b = positions.affine(1.0 / extent as f64, 0.0)?.reshape(shape)?;
    let blended = a
        .narrow(dim, a.dim(dim)? - extent, extent)?
        .broadcast_mul(&weight_a)?
        .add(&b.narrow(dim, 0, extent)?.broadcast_mul(&weight_b)?)?;
    if extent == b.dim(dim)? {
        Ok(blended)
    } else {
        Tensor::cat(
            &[&blended, &b.narrow(dim, extent, b.dim(dim)? - extent)?],
            dim,
        )
    }
}

fn stitch_tiles(
    rows: Vec<Vec<Tensor>>,
    height_overlaps: &[usize],
    width_overlaps: &[usize],
) -> Result<Tensor> {
    if rows.is_empty() || rows[0].is_empty() {
        bail!("H3 tile stitch requires at least one tile")
    }
    let mut result_rows = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let mut result_row = Vec::with_capacity(row.len());
        for (column_index, raw_tile) in row.iter().enumerate() {
            let mut tile = raw_tile.clone();
            if row_index > 0 {
                tile = blend(
                    &rows[row_index - 1][column_index],
                    &tile,
                    height_overlaps[row_index - 1],
                    3,
                )?;
            }
            if column_index > 0 {
                tile = blend(
                    &row[column_index - 1],
                    &tile,
                    width_overlaps[column_index - 1],
                    4,
                )?;
            }
            if row_index < rows.len() - 1 {
                tile = tile.narrow(3, 0, tile.dim(3)? - height_overlaps[row_index])?;
            }
            if column_index < row.len() - 1 {
                tile = tile.narrow(4, 0, tile.dim(4)? - width_overlaps[column_index])?;
            }
            result_row.push(tile);
        }
        result_rows.push(cat_tensors(&result_row, 4)?);
    }
    cat_tensors(&result_rows, 3)
}

fn write_limited(
    sink: &mut dyn DecodeSink,
    frames: &Tensor,
    written: &mut usize,
    total: usize,
) -> Result<()> {
    let count = frames.dim(2)?.min(total.saturating_sub(*written));
    if count > 0 {
        sink.write(*written, &frames.narrow(2, 0, count)?)?;
        *written += count;
    }
    Ok(())
}

#[derive(Default)]
struct CollectDecodeSink {
    parts: Vec<Tensor>,
    next_offset: usize,
}

impl DecodeSink for CollectDecodeSink {
    fn write(&mut self, frame_offset: usize, frames: &Tensor) -> Result<()> {
        if frame_offset != self.next_offset {
            bail!(
                "H3 decode sink expected frame offset {}, got {frame_offset}",
                self.next_offset
            )
        }
        self.next_offset += frames.dim(2)?;
        self.parts.push(frames.clone());
        Ok(())
    }
}

impl CollectDecodeSink {
    fn finish(self) -> Result<Tensor> {
        if self.parts.is_empty() {
            bail!("H3 decode produced no frames")
        }
        cat_tensors(&self.parts, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn build(config: MiniMaxH3VisualVaeConfig) -> (MiniMaxH3VisualVae, VarMap) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = MiniMaxH3VisualVae::new(
            config,
            vb,
            WeightLayout::Diffusers,
            DecodeComputePolicy::Reference,
        )
        .unwrap();
        (model, varmap)
    }

    #[test]
    fn tiny_encoder_executes_the_causal_shape_contract() {
        let config = MiniMaxH3VisualVaeConfig::tiny_for_tests();
        let (model, _) = build(config.clone());
        let pixels = Tensor::zeros((1, 3, 22, 32, 32), DType::F32, &Device::Cpu).unwrap();
        let moments = model
            .encode_moments(&pixels, &mut NoopVisualVaeObserver)
            .unwrap();
        assert_eq!(moments.dims5().unwrap(), (1, 4, 7, 8, 8));
        assert_eq!(model.stored_weight_dtype(), DType::F32);
        assert_eq!(model.decode_compute_dtype(), DType::F32);
    }

    #[test]
    fn converted_and_original_qkv_and_ff_layouts_resolve_identically() {
        let diffusers_qkv = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            (1, 1, 12),
            &Device::Cpu,
        )
        .unwrap();
        let original_qkv = Tensor::from_vec(
            vec![1f32, 2., 5., 6., 9., 10., 3., 4., 7., 8., 11., 12.],
            (1, 1, 12),
            &Device::Cpu,
        )
        .unwrap();
        let diffusers = split_qkv_layout(&diffusers_qkv, WeightLayout::Diffusers, 2, 2).unwrap();
        let original =
            split_qkv_layout(&original_qkv, WeightLayout::OfficialComfySource, 2, 2).unwrap();
        for (converted, source) in [
            (&diffusers.0, &original.0),
            (&diffusers.1, &original.1),
            (&diffusers.2, &original.2),
        ] {
            assert_eq!(
                converted.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                source.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }

        let original_ff =
            Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 4), &Device::Cpu).unwrap();
        let diffusers_ff =
            Tensor::from_vec(vec![3f32, 4., 1., 2.], (1, 1, 4), &Device::Cpu).unwrap();
        let original = split_gate_value(&original_ff, 2, true).unwrap();
        let converted = split_gate_value(&diffusers_ff, 2, false).unwrap();
        assert_eq!(
            original.0.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            converted.0.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            original.1.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            converted.1.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn tiny_decoder_executes_released_temporal_frame_counts() {
        let mut config = MiniMaxH3VisualVaeConfig::tiny_for_tests();
        config.tiling = false;
        let (model, _) = build(config);
        for (latent_frames, pixel_frames) in [(1, 1), (2, 5), (7, 22), (12, 39)] {
            let latents =
                Tensor::zeros((1, 2, latent_frames, 2, 2), DType::F32, &Device::Cpu).unwrap();
            let decoded = model
                .decode_normalized(&latents, &mut NoopVisualVaeObserver)
                .unwrap();
            assert_eq!(
                decoded.dims5().unwrap(),
                (1, 3, pixel_frames, 8, 8),
                "latent frame count {latent_frames}"
            );
        }
    }

    #[test]
    fn comfy_style_tiled_decode_streams_into_the_exact_canvas() {
        #[derive(Default)]
        struct CountTiles(usize);
        impl VisualVaeObserver for CountTiles {
            fn checkpoint(&mut self, event: VisualVaeEvent) -> Result<()> {
                if event.phase == VisualVaePhase::DecodeTile && event.completed < event.total {
                    self.0 += 1;
                }
                Ok(())
            }
        }

        let (model, _) = build(MiniMaxH3VisualVaeConfig::tiny_for_tests());
        let latents = Tensor::zeros((1, 2, 2, 9, 9), DType::F32, &Device::Cpu).unwrap();
        let mut observer = CountTiles::default();
        let decoded = model.decode_normalized(&latents, &mut observer).unwrap();
        assert_eq!(decoded.dims5().unwrap(), (1, 3, 5, 36, 36));
        assert_eq!(observer.0, 4);
    }

    #[test]
    fn tiled_stitch_reconstructs_identical_overlap_samples_without_seams() {
        let global = Tensor::arange(0u32, 36, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, 1, 1, 6, 6))
            .unwrap();
        let rows = vec![
            vec![
                global.narrow(3, 0, 4).unwrap().narrow(4, 0, 4).unwrap(),
                global.narrow(3, 0, 4).unwrap().narrow(4, 2, 4).unwrap(),
            ],
            vec![
                global.narrow(3, 2, 4).unwrap().narrow(4, 0, 4).unwrap(),
                global.narrow(3, 2, 4).unwrap().narrow(4, 2, 4).unwrap(),
            ],
        ];
        let stitched = stitch_tiles(rows, &[2], &[2]).unwrap();
        assert_eq!(
            stitched.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            global.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn decode_stream_is_cancellable_at_a_named_safe_point() {
        struct Cancel;
        impl VisualVaeObserver for Cancel {
            fn checkpoint(&mut self, event: VisualVaeEvent) -> Result<()> {
                if event.phase == VisualVaePhase::DecodeChunk {
                    bail!("synthetic cancellation")
                }
                Ok(())
            }
        }
        let (model, _) = build(MiniMaxH3VisualVaeConfig::tiny_for_tests());
        let latents = Tensor::zeros((1, 2, 7, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let error = model
            .decode_normalized(&latents, &mut Cancel)
            .unwrap_err()
            .to_string();
        assert!(error.contains("synthetic cancellation"));
    }

    #[test]
    fn dtype_policy_separates_storage_from_backend_compute() {
        assert_eq!(
            DecodeComputePolicy::Reference.effective_dtype(&Device::Cpu),
            DType::F32
        );
        assert_eq!(
            DecodeComputePolicy::FullF32.effective_dtype(&Device::Cpu),
            DType::F32
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiny_decoder_cpu_cuda_consistency() {
        use std::collections::HashMap;

        let config = MiniMaxH3VisualVaeConfig::tiny_for_tests();
        let (cpu_model, varmap) = build(config.clone());
        let cuda = Device::new_cuda(0).expect("CUDA test requires a device");
        let tensors = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| Ok((name.clone(), var.as_tensor().to_device(&cuda)?)))
            .collect::<Result<HashMap<_, _>>>()
            .unwrap();
        let cuda_model = MiniMaxH3VisualVae::new(
            config,
            VarBuilder::from_tensors(tensors, DType::F32, &cuda),
            WeightLayout::Diffusers,
            DecodeComputePolicy::Reference,
        )
        .unwrap();
        assert_eq!(cuda_model.decode_compute_dtype(), DType::F16);
        assert_eq!(
            cuda_model.decoder.register_tokens.value().dtype(),
            DType::F32
        );
        assert_eq!(
            cuda_model.decoder.blocks[0].scale1.value().dtype(),
            DType::F32
        );
        assert_eq!(
            cuda_model.decoder.blocks[0]
                .attention
                .qkv
                .compute_weight
                .as_ref()
                .expect("CUDA reference policy keeps an FP16 compute copy")
                .dtype(),
            DType::F16
        );
        // 9x9 latents map to 36x36 pixels, forcing the 32px/8px-overlap
        // synthetic configuration through four streamed decode tiles.
        let latents = Tensor::zeros((1, 2, 2, 9, 9), DType::F32, &Device::Cpu).unwrap();
        let cpu = cpu_model
            .decode_normalized(&latents, &mut NoopVisualVaeObserver)
            .unwrap();
        let gpu = cuda_model
            .decode_normalized(
                &latents.to_device(&cuda).unwrap(),
                &mut NoopVisualVaeObserver,
            )
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let max_error = cpu
            .sub(&gpu)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(gpu.dims5().unwrap(), (1, 3, 5, 36, 36));
        assert!(max_error < 0.08, "CPU/CUDA max error {max_error}");
    }
}
