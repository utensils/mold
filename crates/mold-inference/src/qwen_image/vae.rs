//! Qwen-Image VAE encoder and decoder for single-image inference.
//!
//! The upstream Qwen-Image VAE is a 3D causal autoencoder fine-tuned from Wan VAE.
//! For still-image generation (`T = 1`), the encoder/decoder can be specialized to
//! a 2D path: the causal 3D convolutions only see the current frame and the temporal
//! upsample/downsample paths are inactive on the first frame.

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

/// Per-channel latent normalization constants from the upstream VAE config.
const LATENTS_MEAN: [f64; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const LATENTS_STD: [f64; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

const BLOCK_OUT_CHANNELS: [usize; 4] = [96, 192, 384, 384];
const LATENT_CHANNELS: usize = 16;
const NUM_RES_BLOCKS: usize = 2;

/// Load a 5D causal conv3d weight as a 2D conv by extracting the last temporal slice.
///
/// For single-frame (T=1) inference with causal padding, only the last temporal kernel
/// slice contributes (the first slices operate on zero-padded frames). This is exact,
/// not an approximation.
fn load_3d_conv_as_2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    // Last temporal slice: for T=1 with causal padding (2 zero frames + 1 real frame),
    // only kernel[t=kernel_size-1] is applied to the real frame.
    let ws = ws.i((.., .., kernel_size - 1, .., ..))?.contiguous()?;
    let bias = vb.get(out_channels, "bias").ok();
    Ok(Conv2d::new(
        ws,
        bias,
        Conv2dConfig {
            padding,
            ..Default::default()
        },
    ))
}

fn load_3d_conv1x1_as_2d(
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get((out_channels, in_channels, 1, 1, 1), "weight")?;
    let ws = ws.i((.., .., 0, .., ..))?.contiguous()?;
    let bias = vb.get(out_channels, "bias").ok();
    Ok(Conv2d::new(ws, bias, Default::default()))
}

/// Channel-wise RMS normalization matching the Wan VAE's `RMS_norm`.
///
/// The Python reference uses `F.normalize(x, dim=1) * scale * gamma` where
/// `scale = sqrt(dim)`. Since `F.normalize` divides by the L2 norm
/// `sqrt(sum(x^2, dim=1))` = `sqrt(C) * sqrt(mean(x^2, dim=1))`, the `sqrt(C)`
/// from `scale` cancels with the `sqrt(C)` in the denominator. The net effect
/// is simply `x / RMS(x) * gamma` — NO extra sqrt(C) scaling.
#[derive(Debug, Clone)]
struct QwenImageRmsNorm2d {
    gamma: Tensor,
}

impl QwenImageRmsNorm2d {
    fn for_image(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((channels, 1, 1), "gamma")?;
        Ok(Self { gamma })
    }

    fn for_feature(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb
            .get((channels, 1, 1, 1), "gamma")?
            .reshape((channels, 1, 1))?;
        Ok(Self { gamma })
    }
}

impl Module for QwenImageRmsNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rms = (x.sqr()?.mean_keepdim(1)? + 1e-6)?.sqrt()?;
        x.broadcast_div(&rms)?.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug, Clone)]
struct QwenImageAttentionBlock2d {
    norm: QwenImageRmsNorm2d,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl QwenImageAttentionBlock2d {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: QwenImageRmsNorm2d::for_image(dim, vb.pp("norm"))?,
            to_qkv: conv2d(dim, dim * 3, 1, Default::default(), vb.pp("to_qkv"))?,
            proj: conv2d(dim, dim, 1, Default::default(), vb.pp("proj"))?,
        })
    }
}

impl Module for QwenImageAttentionBlock2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, c, h, w) = x.dims4()?;
        let x = x.apply(&self.norm)?;
        let qkv = x.apply(&self.to_qkv)?;
        let qkv = qkv.reshape((b, 1, c * 3, h * w))?.transpose(2, 3)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];
        let scale = 1.0 / (c as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let x = attn.matmul(v)?;
        let x = x.transpose(2, 3)?.reshape((b, c, h, w))?;
        x.apply(&self.proj)? + residual
    }
}

#[derive(Debug, Clone)]
struct QwenImageResidualBlock2d {
    norm1: QwenImageRmsNorm2d,
    conv1: Conv2d,
    norm2: QwenImageRmsNorm2d,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl QwenImageResidualBlock2d {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: QwenImageRmsNorm2d::for_feature(in_dim, vb.pp("norm1"))?,
            conv1: load_3d_conv_as_2d(in_dim, out_dim, 3, 1, vb.pp("conv1"))?,
            norm2: QwenImageRmsNorm2d::for_feature(out_dim, vb.pp("norm2"))?,
            conv2: load_3d_conv_as_2d(out_dim, out_dim, 3, 1, vb.pp("conv2"))?,
            conv_shortcut: if in_dim != out_dim {
                Some(load_3d_conv1x1_as_2d(
                    in_dim,
                    out_dim,
                    vb.pp("conv_shortcut"),
                )?)
            } else {
                None
            },
        })
    }
}

impl Module for QwenImageResidualBlock2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = match &self.conv_shortcut {
            Some(conv) => x.apply(conv)?,
            None => x.clone(),
        };
        let h = x
            .apply(&self.norm1)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv1)?
            .apply(&self.norm2)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv2)?;
        residual + h
    }
}

#[derive(Debug, Clone)]
struct QwenImageMidBlock2d {
    resnet0: QwenImageResidualBlock2d,
    attention: QwenImageAttentionBlock2d,
    resnet1: QwenImageResidualBlock2d,
}

impl QwenImageMidBlock2d {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            resnet0: QwenImageResidualBlock2d::new(channels, channels, vb.pp("resnets").pp("0"))?,
            attention: QwenImageAttentionBlock2d::new(channels, vb.pp("attentions").pp("0"))?,
            resnet1: QwenImageResidualBlock2d::new(channels, channels, vb.pp("resnets").pp("1"))?,
        })
    }
}

impl Module for QwenImageMidBlock2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.resnet0)?
            .apply(&self.attention)?
            .apply(&self.resnet1)
    }
}

/// Spatial upsample: nearest 2x → Conv2d.
///
/// The time_conv weights in the safetensors are for video temporal upsampling
/// and are correctly skipped for single-frame (T=1) image generation —
/// the Wan VAE only uses time_conv when feat_cache is available (streaming video).
#[derive(Debug, Clone)]
struct QwenImageUpsample2d {
    conv: Conv2d,
}

impl QwenImageUpsample2d {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(
                in_dim,
                out_dim,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("resample").pp("1"),
            )?,
        })
    }
}

impl Module for QwenImageUpsample2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        x.upsample_nearest2d(h * 2, w * 2)?.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct QwenImageUpBlock2d {
    resnets: Vec<QwenImageResidualBlock2d>,
    upsample: Option<QwenImageUpsample2d>,
}

impl QwenImageUpBlock2d {
    fn new(in_dim: usize, out_dim: usize, add_upsample: bool, vb: VarBuilder) -> Result<Self> {
        let mut resnets = Vec::with_capacity(NUM_RES_BLOCKS + 1);
        let mut current_dim = in_dim;
        for i in 0..=NUM_RES_BLOCKS {
            resnets.push(QwenImageResidualBlock2d::new(
                current_dim,
                out_dim,
                vb.pp("resnets").pp(i),
            )?);
            current_dim = out_dim;
        }
        Ok(Self {
            resnets,
            upsample: if add_upsample {
                Some(QwenImageUpsample2d::new(
                    out_dim,
                    out_dim / 2,
                    vb.pp("upsamplers").pp("0"),
                )?)
            } else {
                None
            },
        })
    }
}

impl Module for QwenImageUpBlock2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = x.apply(resnet)?;
        }
        if let Some(us) = &self.upsample {
            x = x.apply(us)?;
        }
        Ok(x)
    }
}

/// Spatial downsample: asymmetric zero-pad → stride-2 Conv2d.
///
/// Mirrors `QwenImageUpsample2d` but in the downsampling direction.
/// The time_conv weights in the safetensors are for video temporal downsampling
/// and are correctly skipped for single-frame (T=1) image generation.
#[derive(Debug, Clone)]
struct QwenImageDownsample2d {
    conv: Conv2d,
}

impl QwenImageDownsample2d {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        // The encoder's resample weights are already 2D (not 3D like other encoder
        // convolutions), so we use a standard stride-2 conv2d instead of
        // load_3d_conv_as_2d_stride2.
        let cfg = Conv2dConfig {
            stride: 2,
            padding: 0, // asymmetric padding applied in forward()
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(in_dim, out_dim, 3, cfg, vb.pp("resample").pp("1"))?,
        })
    }
}

impl Module for QwenImageDownsample2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Asymmetric padding: pad right and bottom by 1 to match PyTorch stride-2
        // conv behavior with kernel_size=3 (same as Flux VAE Downsample).
        let x = x.pad_with_zeros(D::Minus1, 0, 1)?;
        let x = x.pad_with_zeros(D::Minus2, 0, 1)?;
        x.apply(&self.conv)
    }
}

/// Encoder block: either a ResNet or a Downsample (flat layout from Wan VAE).
///
/// The Wan VAE encoder uses flat `down_blocks.{0-10}` indexing where each block
/// is either a single ResNet or a Downsample operation, NOT the nested
/// `down_blocks.{N}.resnets.{M}` grouping used by diffusers.
#[derive(Debug, Clone)]
enum QwenImageEncoderBlock {
    ResNet(QwenImageResidualBlock2d),
    Downsample(QwenImageDownsample2d),
}

impl Module for QwenImageEncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::ResNet(r) => r.forward(x),
            Self::Downsample(d) => d.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
struct QwenImageEncoder2d {
    conv_in: Conv2d,
    blocks: Vec<QwenImageEncoderBlock>,
    mid_block: QwenImageMidBlock2d,
    norm_out: QwenImageRmsNorm2d,
    conv_out: Conv2d,
}

impl QwenImageEncoder2d {
    fn new(vb: VarBuilder) -> Result<Self> {
        // Wan VAE encoder flat block layout (channels differ from decoder):
        //   conv_in: 3 → 96
        //   0,1  → ResNet (96→96)
        //   2    → Downsample (96→96)
        //   3,4  → ResNet (96→192, block 3 has conv_shortcut)
        //   5    → Downsample (192→192)
        //   6,7  → ResNet (192→384, block 6 has conv_shortcut)
        //   8    → Downsample (384→384)
        //   9,10 → ResNet (384→384)
        //   mid_block: 384
        let db = vb.pp("down_blocks");
        let blocks = vec![
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(96, 96, db.pp("0"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(96, 96, db.pp("1"))?),
            QwenImageEncoderBlock::Downsample(QwenImageDownsample2d::new(96, 96, db.pp("2"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(96, 192, db.pp("3"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(192, 192, db.pp("4"))?),
            QwenImageEncoderBlock::Downsample(QwenImageDownsample2d::new(192, 192, db.pp("5"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(192, 384, db.pp("6"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(384, 384, db.pp("7"))?),
            QwenImageEncoderBlock::Downsample(QwenImageDownsample2d::new(384, 384, db.pp("8"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(384, 384, db.pp("9"))?),
            QwenImageEncoderBlock::ResNet(QwenImageResidualBlock2d::new(384, 384, db.pp("10"))?),
        ];
        Ok(Self {
            conv_in: load_3d_conv_as_2d(3, 96, 3, 1, vb.pp("conv_in"))?,
            blocks,
            mid_block: QwenImageMidBlock2d::new(384, vb.pp("mid_block"))?,
            norm_out: QwenImageRmsNorm2d::for_feature(384, vb.pp("norm_out"))?,
            conv_out: load_3d_conv_as_2d(384, 2 * LATENT_CHANNELS, 3, 1, vb.pp("conv_out"))?,
        })
    }
}

impl Module for QwenImageEncoder2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.apply(&self.conv_in)?;
        for block in &self.blocks {
            x = x.apply(block)?;
        }
        x.apply(&self.mid_block)?
            .apply(&self.norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct QwenImageDecoder2d {
    conv_in: Conv2d,
    mid_block: QwenImageMidBlock2d,
    up_blocks: Vec<QwenImageUpBlock2d>,
    norm_out: QwenImageRmsNorm2d,
    conv_out: Conv2d,
}

impl QwenImageDecoder2d {
    fn new(vb: VarBuilder) -> Result<Self> {
        let dims = [
            BLOCK_OUT_CHANNELS[3],
            BLOCK_OUT_CHANNELS[3],
            BLOCK_OUT_CHANNELS[2],
            BLOCK_OUT_CHANNELS[1],
            BLOCK_OUT_CHANNELS[0],
        ];
        let mut up_blocks = Vec::with_capacity(BLOCK_OUT_CHANNELS.len());
        for i in 0..BLOCK_OUT_CHANNELS.len() {
            let mut in_dim = dims[i];
            if i > 0 {
                in_dim /= 2;
            }
            let out_dim = dims[i + 1];
            let add_upsample = i < BLOCK_OUT_CHANNELS.len() - 1;
            up_blocks.push(QwenImageUpBlock2d::new(
                in_dim,
                out_dim,
                add_upsample,
                vb.pp("up_blocks").pp(i),
            )?);
        }
        Ok(Self {
            conv_in: load_3d_conv_as_2d(16, BLOCK_OUT_CHANNELS[3], 3, 1, vb.pp("conv_in"))?,
            mid_block: QwenImageMidBlock2d::new(BLOCK_OUT_CHANNELS[3], vb.pp("mid_block"))?,
            up_blocks,
            norm_out: QwenImageRmsNorm2d::for_feature(BLOCK_OUT_CHANNELS[0], vb.pp("norm_out"))?,
            conv_out: load_3d_conv_as_2d(BLOCK_OUT_CHANNELS[0], 3, 3, 1, vb.pp("conv_out"))?,
        })
    }
}

impl Module for QwenImageDecoder2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.apply(&self.conv_in)?.apply(&self.mid_block)?;
        for block in &self.up_blocks {
            x = x.apply(block)?;
        }
        x.apply(&self.norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

/// Qwen-Image VAE with per-channel latent normalization.
pub(crate) struct QwenImageVae {
    encoder: QwenImageEncoder2d,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    decoder: QwenImageDecoder2d,
    latents_mean: Tensor,
    latents_std: Tensor,
}

impl QwenImageVae {
    pub fn load(
        vae_path: &std::path::Path,
        device: &candle_core::Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            &[vae_path],
            dtype,
            device,
            "Qwen-Image VAE",
            progress,
        )
        .map_err(candle_core::Error::msg)?;
        let encoder = QwenImageEncoder2d::new(vb.pp("encoder"))?;
        let quant_conv = load_3d_conv1x1_as_2d(
            2 * LATENT_CHANNELS,
            2 * LATENT_CHANNELS,
            vb.pp("quant_conv"),
        )?;
        let post_quant_conv =
            load_3d_conv1x1_as_2d(LATENT_CHANNELS, LATENT_CHANNELS, vb.pp("post_quant_conv"))?;
        let decoder = QwenImageDecoder2d::new(vb.pp("decoder"))?;

        let mean_vec: Vec<f32> = LATENTS_MEAN.iter().map(|&v| v as f32).collect();
        let std_vec: Vec<f32> = LATENTS_STD.iter().map(|&v| v as f32).collect();
        let latents_mean =
            Tensor::from_vec(mean_vec, (1, LATENT_CHANNELS, 1, 1), device)?.to_dtype(dtype)?;
        let latents_std =
            Tensor::from_vec(std_vec, (1, LATENT_CHANNELS, 1, 1), device)?.to_dtype(dtype)?;

        Ok(Self {
            encoder,
            quant_conv,
            post_quant_conv,
            decoder,
            latents_mean,
            latents_std,
        })
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let denormed = latents
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?;
        let denormed = denormed.apply(&self.post_quant_conv)?;
        denormed.apply(&self.decoder)
    }

    /// Encode a pixel-space image [1, 3, H, W] (in [-1, 1]) to normalized latents.
    ///
    /// Applies the encoder, quant_conv, diagonal Gaussian sampling (mean + logvar → z),
    /// then per-channel normalization (inverse of decode's denormalization):
    ///   `normed = (z - mean) / std`
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.encoder)?.apply(&self.quant_conv)?;

        // Diagonal Gaussian: split mean/logvar along channel dim, sample z
        let c2 = h.dim(1)?;
        let c = c2 / 2;
        let mean = h.narrow(1, 0, c)?;
        let logvar = h.narrow(1, c, c)?.clamp(-30.0, 20.0)?;
        let std = (&logvar * 0.5)?.exp()?;
        let z = (&mean + &std.broadcast_mul(&mean.randn_like(0., 1.)?)?)?;

        // Per-channel normalization (inverse of decode's denormalization)
        let normed = z
            .broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)?;

        Ok(normed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_no_spurious_scaling() {
        // RMS norm should compute x / RMS(x) * gamma — no sqrt(C) factor.
        // For a constant tensor, RMS = |x|, so output = sign(x) * gamma.
        let dev = candle_core::Device::Cpu;
        let gamma = Tensor::ones((4, 1, 1), DType::F32, &dev).unwrap();
        let norm = QwenImageRmsNorm2d { gamma };

        // Input: [1, 4, 1, 1] all 2.0 → RMS per channel = 2.0 → output = 1.0
        let x = Tensor::full(2.0f32, (1, 4, 1, 1), &dev).unwrap();
        let out = norm.forward(&x).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(
                (v - 1.0).abs() < 0.01,
                "expected ~1.0 but got {v} (would be ~2.0 with spurious sqrt(C))"
            );
        }
    }

    #[test]
    fn rms_norm_gamma_broadcast() {
        // Verify gamma broadcasts correctly over (B, C, H, W)
        let dev = candle_core::Device::Cpu;
        let gamma_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = Tensor::from_vec(gamma_vals, (4, 1, 1), &dev).unwrap();
        let norm = QwenImageRmsNorm2d { gamma };

        let x = Tensor::full(1.0f32, (1, 4, 2, 2), &dev).unwrap();
        let out = norm.forward(&x).unwrap();
        // RMS of [1,1,1,1] over channel dim = 1.0, so out = x * gamma
        // Channel 0: 1.0, Channel 1: 2.0, Channel 2: 3.0, Channel 3: 4.0
        let c0 = out.i((0, 0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        let c1 = out.i((0, 1, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        assert!((c0 - 1.0).abs() < 0.01, "channel 0: expected 1.0, got {c0}");
        assert!((c1 - 2.0).abs() < 0.01, "channel 1: expected 2.0, got {c1}");
    }

    #[test]
    fn latent_denormalization_formula() {
        // Verify: denormed = latents * std + mean
        let dev = candle_core::Device::Cpu;
        let mean = Tensor::from_vec(vec![0.5f32, -0.5], (1, 2, 1, 1), &dev).unwrap();
        let std = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2, 1, 1), &dev).unwrap();
        let latents = Tensor::full(1.0f32, (1, 2, 1, 1), &dev).unwrap();
        let result = latents
            .broadcast_mul(&std)
            .unwrap()
            .broadcast_add(&mean)
            .unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 2.5).abs() < 1e-6, "1.0 * 2.0 + 0.5 = 2.5");
        assert!((vals[1] - 2.5).abs() < 1e-6, "1.0 * 3.0 + (-0.5) = 2.5");
    }

    #[test]
    fn test_latent_constants_length() {
        assert_eq!(
            LATENTS_MEAN.len(),
            16,
            "LATENTS_MEAN must have 16 elements (one per latent channel)"
        );
        assert_eq!(
            LATENTS_STD.len(),
            16,
            "LATENTS_STD must have 16 elements (one per latent channel)"
        );
    }

    #[test]
    fn test_latent_std_all_positive() {
        for (i, &val) in LATENTS_STD.iter().enumerate() {
            assert!(
                val > 0.0,
                "LATENTS_STD[{i}] = {val} is not positive; zero or negative std would cause division issues in denormalization"
            );
        }
    }

    #[test]
    fn test_block_out_channels_architecture() {
        // The decoder uses BLOCK_OUT_CHANNELS to define the up-block channel progression.
        // It must have exactly 4 elements with an ascending-then-plateau pattern.
        assert_eq!(BLOCK_OUT_CHANNELS.len(), 4, "expected 4 decoder stages");
        assert_eq!(BLOCK_OUT_CHANNELS, [96, 192, 384, 384]);

        // Channels should increase monotonically (non-strictly at the plateau)
        for i in 1..BLOCK_OUT_CHANNELS.len() {
            assert!(
                BLOCK_OUT_CHANNELS[i] >= BLOCK_OUT_CHANNELS[i - 1],
                "block_out_channels should be non-decreasing: [{}]={} < [{}]={}",
                i,
                BLOCK_OUT_CHANNELS[i],
                i - 1,
                BLOCK_OUT_CHANNELS[i - 1]
            );
        }
    }
}
