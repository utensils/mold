//! Image-specialized Qwen Image 2.1 VAE decoder.
//!
//! The upstream class is a one-frame specialization of a causal video VAE.
//! Its `QwenImage21CausalConv3d` immediately squeezes the singleton temporal
//! dimension, and the decode path feeds its first (and only) frame through the
//! `first_chunk` branch.  This module implements that exact image path with
//! 2-D Candle operations; it intentionally does not pretend to support the
//! VAE's video streaming caches.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};
use std::path::Path;

use super::QWEN_IMAGE_21_LATENT_CHANNELS;

const DECODER_BASE_DIM: usize = 144;
const DIM_MULT: [usize; 5] = [1, 2, 4, 8, 8];
const NUM_RES_BLOCKS: usize = 2;
const VAE_ATTN_CHUNK_ROWS: usize = 1024;

// These are checkpoint parameters, not approximations of mathematical
// constants (one happens to be close to `FRAC_PI_6`).
#[allow(clippy::approx_constant)]
const LATENTS_MEAN: [f64; QWEN_IMAGE_21_LATENT_CHANNELS] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];

const LATENTS_STD: [f64; QWEN_IMAGE_21_LATENT_CHANNELS] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904,
    2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075,
    3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
    3.7662, 3.6746, 3.4563, 3.8161,
];

/// Qwen's channel-first RMS normalizer.
///
/// `F.normalize(x, dim=1) * sqrt(channels) * gamma` simplifies to
/// `x / RMS_channel(x) * gamma`; `gamma` is stored as `[C, 1, 1]` for image
/// attention and `[C, 1, 1, 1]` for residual/decoder feature tensors.
struct RmsNorm2d {
    gamma: Tensor,
}

impl RmsNorm2d {
    fn image(channels: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gamma: vb.get((channels, 1, 1), "gamma")?,
        })
    }

    fn feature(channels: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gamma: vb
                .get((channels, 1, 1, 1), "gamma")?
                .reshape((channels, 1, 1))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let f32_x = xs.to_dtype(DType::F32)?;
        let channels = xs.dim(1)?;
        // Upstream calls `F.normalize(x, dim=1) * sqrt(channels)`, not a
        // conventional epsilon-RMSNorm. For nonzero vectors that is exactly
        // `x / sqrt(mean(x²))`; `F.normalize` only clamps a zero/denormal
        // vector at its 1e-12 norm floor. Keeping the reduction in F32
        // preserves the checkpoint's BF16 path without injecting a 1e-6
        // scale change into every decoder residual block.
        let floor = 1e-12f64 / (channels as f64).sqrt();
        let rms = f32_x
            .sqr()?
            .mean_keepdim(1)?
            .sqrt()?
            .clamp(floor, f32::INFINITY)?;
        f32_x
            .broadcast_div(&rms)?
            .broadcast_mul(&self.gamma.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

struct AttentionBlock2d {
    norm: RmsNorm2d,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl AttentionBlock2d {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm2d::image(dim, vb.pp("norm"))?,
            to_qkv: conv2d(dim, 3 * dim, 1, Default::default(), vb.pp("to_qkv"))?,
            proj: conv2d(dim, dim, 1, Default::default(), vb.pp("proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = xs.dims4()?;
        let normalized = self.norm.forward(xs)?;
        let qkv = self
            .to_qkv
            .forward(&normalized)?
            .reshape((batch, 1, 3 * channels, height * width))?
            .transpose(2, 3)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let q = chunks[0].contiguous()?;
        let k = chunks[1].contiguous()?;
        let v = chunks[2].contiguous()?;
        let scale = (1.0 / (channels as f64).sqrt()) as f32;
        let tokens = height * width;
        let attended = if tokens > VAE_ATTN_CHUNK_ROWS {
            crate::attention::math_attention_with_chunk(&q, &k, &v, scale, VAE_ATTN_CHUNK_ROWS)?
        } else {
            crate::attention::math_attention(&q, &k, &v, scale)?
        };
        let attended = attended
            .transpose(2, 3)?
            .reshape((batch, channels, height, width))?;
        Ok((self.proj.forward(&attended)? + xs)?)
    }
}

struct ResidualBlock2d {
    norm1: RmsNorm2d,
    conv1: Conv2d,
    norm2: RmsNorm2d,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResidualBlock2d {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            norm1: RmsNorm2d::feature(in_dim, vb.pp("norm1"))?,
            conv1: conv2d(in_dim, out_dim, 3, conv_cfg, vb.pp("conv1"))?,
            norm2: RmsNorm2d::feature(out_dim, vb.pp("norm2"))?,
            conv2: conv2d(out_dim, out_dim, 3, conv_cfg, vb.pp("conv2"))?,
            shortcut: (in_dim != out_dim)
                .then(|| {
                    conv2d(
                        in_dim,
                        out_dim,
                        1,
                        Default::default(),
                        vb.pp("conv_shortcut"),
                    )
                })
                .transpose()?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        let hidden = self
            .conv1
            .forward(&candle_nn::Activation::Silu.forward(&self.norm1.forward(xs)?)?)?;
        let hidden = self
            .conv2
            .forward(&candle_nn::Activation::Silu.forward(&self.norm2.forward(&hidden)?)?)?;
        Ok((shortcut + hidden)?)
    }
}

struct MidBlock2d {
    resnet0: ResidualBlock2d,
    attention: AttentionBlock2d,
    resnet1: ResidualBlock2d,
}

impl MidBlock2d {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            resnet0: ResidualBlock2d::new(dim, dim, vb.pp("resnets").pp("0"))?,
            attention: AttentionBlock2d::new(dim, vb.pp("attentions").pp("0"))?,
            resnet1: ResidualBlock2d::new(dim, dim, vb.pp("resnets").pp("1"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.resnet1
            .forward(&self.attention.forward(&self.resnet0.forward(xs)?)?)
    }
}

/// The residual decoder's parameter-free `DupUp3D` shortcut after the VAE
/// folds its only temporal frame away.
///
/// `first_chunk=true` is load-bearing: it selects the last temporal duplicate
/// from the causal VAE's virtual upsampled frame pair, leaving one image frame
/// while preserving its exact channel grouping.
fn duplicate_upsample_shortcut(
    xs: &Tensor,
    out_channels: usize,
    temporal_factor: usize,
    first_chunk: bool,
) -> Result<Tensor> {
    let (batch, in_channels, height, width) = xs.dims4()?;
    let spatial_factor = 2usize;
    let factor = temporal_factor * spatial_factor * spatial_factor;
    anyhow::ensure!(
        (out_channels * factor).is_multiple_of(in_channels),
        "Qwen Image 2.1 VAE shortcut cannot map {in_channels} channels to {out_channels}"
    );
    let repeats = out_channels * factor / in_channels;
    let duplicated = xs
        .unsqueeze(2)?
        .broadcast_as((batch, in_channels, repeats, height, width))?
        .reshape(&[
            batch,
            out_channels,
            temporal_factor,
            spatial_factor,
            spatial_factor,
            height,
            width,
        ])?
        // Candle's `permute` tuple implementation intentionally stops at
        // six axes.  Three pairwise swaps express the 7-D DupUp3D layout
        // without flattening or copying the grouped channel/space axes.
        .transpose(3, 5)?
        .transpose(4, 5)?
        .transpose(5, 6)?
        .reshape((
            batch,
            out_channels,
            temporal_factor,
            height * spatial_factor,
            width * spatial_factor,
        ))?;
    let temporal_index = if first_chunk { temporal_factor - 1 } else { 0 };
    duplicated
        .narrow(2, temporal_index, 1)?
        .reshape((
            batch,
            out_channels,
            height * spatial_factor,
            width * spatial_factor,
        ))
        .map_err(Into::into)
}

struct ResidualUpBlock2d {
    resnets: Vec<ResidualBlock2d>,
    upsampler: Option<Conv2d>,
    out_dim: usize,
    temporal_upsample: bool,
}

impl ResidualUpBlock2d {
    fn new(
        in_dim: usize,
        out_dim: usize,
        upsample: bool,
        temporal_upsample: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(NUM_RES_BLOCKS + 1);
        let mut current_dim = in_dim;
        for index in 0..=NUM_RES_BLOCKS {
            resnets.push(ResidualBlock2d::new(
                current_dim,
                out_dim,
                vb.pp("resnets").pp(index),
            )?);
            current_dim = out_dim;
        }
        let upsampler = if upsample {
            Some(conv2d(
                out_dim,
                out_dim,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("upsampler").pp("resample").pp("1"),
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            upsampler,
            out_dim,
            temporal_upsample,
        })
    }

    fn forward(&self, xs: &Tensor, first_chunk: bool) -> Result<Tensor> {
        let shortcut_input = xs.clone();
        let mut hidden = xs.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward(&hidden)?;
        }
        let Some(upsampler) = &self.upsampler else {
            return Ok(hidden);
        };
        let (_, _, height, width) = hidden.dims4()?;
        let hidden = upsampler.forward(&hidden.upsample_nearest2d(height * 2, width * 2)?)?;
        let shortcut = duplicate_upsample_shortcut(
            &shortcut_input,
            self.out_dim,
            if self.temporal_upsample { 2 } else { 1 },
            first_chunk,
        )?;
        Ok((hidden + shortcut)?)
    }
}

struct Decoder2d {
    conv_in: Conv2d,
    mid_block: MidBlock2d,
    up_blocks: Vec<ResidualUpBlock2d>,
    norm_out: RmsNorm2d,
    conv_out: Conv2d,
}

impl Decoder2d {
    fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let dimensions = [
            DECODER_BASE_DIM * DIM_MULT[4],
            DECODER_BASE_DIM * DIM_MULT[4],
            DECODER_BASE_DIM * DIM_MULT[3],
            DECODER_BASE_DIM * DIM_MULT[2],
            DECODER_BASE_DIM * DIM_MULT[1],
            DECODER_BASE_DIM * DIM_MULT[0],
        ];
        let temporal_upsample = [true, true, true, false];
        let mut up_blocks = Vec::with_capacity(DIM_MULT.len());
        for index in 0..DIM_MULT.len() {
            let upsample = index != DIM_MULT.len() - 1;
            up_blocks.push(ResidualUpBlock2d::new(
                dimensions[index],
                dimensions[index + 1],
                upsample,
                upsample && temporal_upsample[index],
                vb.pp("up_blocks").pp(index),
            )?);
        }
        let final_dim = *dimensions.last().expect("decoder dimensions are non-empty");
        Ok(Self {
            conv_in: conv2d(
                QWEN_IMAGE_21_LATENT_CHANNELS,
                dimensions[0],
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv_in"),
            )?,
            mid_block: MidBlock2d::new(dimensions[0], vb.pp("mid_block"))?,
            up_blocks,
            norm_out: RmsNorm2d::feature(final_dim, vb.pp("norm_out"))?,
            conv_out: conv2d(
                final_dim,
                4,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv_out"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.mid_block.forward(&self.conv_in.forward(xs)?)?;
        for block in &self.up_blocks {
            hidden = block.forward(&hidden, true)?;
        }
        self.conv_out
            .forward(&candle_nn::Activation::Silu.forward(&self.norm_out.forward(&hidden)?)?)
            .map_err(Into::into)
    }
}

/// Decoder-only native Qwen Image 2.1 VAE.
pub(crate) struct QwenImage21Vae {
    post_quant_conv: Conv2d,
    decoder: Decoder2d,
    latents_mean: Tensor,
    latents_std: Tensor,
}

impl QwenImage21Vae {
    pub(crate) fn load(
        path: &Path,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            &[path],
            dtype,
            device,
            "Qwen Image 2.1 VAE",
            progress,
        )?;
        Self::from_var_builder(vb, device, dtype)
    }

    pub(crate) fn from_var_builder(
        vb: VarBuilder<'_>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let post_quant_conv = conv2d(
            QWEN_IMAGE_21_LATENT_CHANNELS,
            QWEN_IMAGE_21_LATENT_CHANNELS,
            1,
            Default::default(),
            vb.pp("post_quant_conv"),
        )?;
        let decoder = Decoder2d::new(vb.pp("decoder"))?;
        let mean = Tensor::from_vec(
            LATENTS_MEAN
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            (1, QWEN_IMAGE_21_LATENT_CHANNELS, 1, 1),
            device,
        )?
        .to_dtype(dtype)?;
        let std = Tensor::from_vec(
            LATENTS_STD
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            (1, QWEN_IMAGE_21_LATENT_CHANNELS, 1, 1),
            device,
        )?
        .to_dtype(dtype)?;
        Ok(Self {
            post_quant_conv,
            decoder,
            latents_mean: mean,
            latents_std: std,
        })
    }

    /// Decode normalized `[B, H*W, 64]` diffusion latents into clamped RGBA
    /// samples `[B, 4, H*16, W*16]`.
    pub(crate) fn decode_packed(
        &self,
        latents: &Tensor,
        latent_height: usize,
        latent_width: usize,
    ) -> Result<Tensor> {
        let (batch, tokens, channels) = latents.dims3()?;
        anyhow::ensure!(
            channels == QWEN_IMAGE_21_LATENT_CHANNELS,
            "Qwen Image 2.1 VAE expected {QWEN_IMAGE_21_LATENT_CHANNELS} latent channels, got {channels}"
        );
        anyhow::ensure!(
            tokens == latent_height.saturating_mul(latent_width),
            "Qwen Image 2.1 VAE latent token count {tokens} does not match {latent_height}x{latent_width}"
        );
        let latents =
            latents
                .transpose(1, 2)?
                .reshape((batch, channels, latent_height, latent_width))?;
        let latents = latents
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?;
        self.decoder
            .forward(&self.post_quant_conv.forward(&latents)?)?
            .clamp(-1.0f32, 1.0f32)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_statistics_match_the_official_64_channel_config() {
        assert_eq!(LATENTS_MEAN.len(), QWEN_IMAGE_21_LATENT_CHANNELS);
        assert_eq!(LATENTS_STD.len(), QWEN_IMAGE_21_LATENT_CHANNELS);
        assert!(LATENTS_STD.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn first_chunk_shortcut_preserves_one_temporal_image_frame() {
        let input =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &Device::Cpu).unwrap();
        // One input channel -> eight output channels, so the 2x2 spatial and
        // 2x temporal duplicate layout divides exactly.
        let output = duplicate_upsample_shortcut(&input, 1, 2, true).unwrap();
        assert_eq!(output.dims(), &[1, 1, 4, 4]);
        assert!(output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn decoder_channel_ladder_is_the_21_checkpoint_ladder() {
        let dimensions = [1152, 1152, 1152, 576, 288, 144];
        assert_eq!(DECODER_BASE_DIM * DIM_MULT[4], dimensions[0]);
        assert_eq!(DECODER_BASE_DIM * DIM_MULT[0], *dimensions.last().unwrap());
    }
}
