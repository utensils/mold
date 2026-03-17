//! Qwen-Image VAE decoder for single-image inference.
//!
//! The upstream Qwen-Image VAE is a 3D causal autoencoder fine-tuned from Wan VAE.
//! For still-image generation (`T = 1`), the decoder can be specialized to a 2D path:
//! the causal 3D convolutions only see the current frame and the temporal upsample
//! path is inactive on the first frame. This lets us decode image latents without
//! porting the full video encoder stack.

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
const NUM_RES_BLOCKS: usize = 2;

fn load_3d_conv_as_2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (out_channels, in_channels, kernel_size, kernel_size, kernel_size),
        "weight",
    )?;
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

fn load_3d_conv1x1_as_2d(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Conv2d> {
    let ws = vb.get((out_channels, in_channels, 1, 1, 1), "weight")?;
    let ws = ws.i((.., .., 0, .., ..))?.contiguous()?;
    let bias = vb.get(out_channels, "bias").ok();
    Ok(Conv2d::new(ws, bias, Default::default()))
}

#[derive(Debug, Clone)]
struct QwenImageRmsNorm2d {
    gamma: Tensor,
    scale: f64,
}

impl QwenImageRmsNorm2d {
    fn for_image(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((channels, 1, 1), "gamma")?;
        Ok(Self {
            gamma,
            scale: (channels as f64).sqrt(),
        })
    }

    fn for_feature(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((channels, 1, 1, 1), "gamma")?.reshape((channels, 1, 1))?;
        Ok(Self {
            gamma,
            scale: (channels as f64).sqrt(),
        })
    }
}

impl Module for QwenImageRmsNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rms = (x.sqr()?.mean_keepdim(1)? + 1e-6)?.sqrt()?;
        let x = x.broadcast_div(&rms)?;
        let x = (x * self.scale)?;
        x.broadcast_mul(&self.gamma)
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
                Some(load_3d_conv1x1_as_2d(in_dim, out_dim, vb.pp("conv_shortcut"))?)
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
                Some(QwenImageUpsample2d::new(out_dim, out_dim / 2, vb.pp("upsamplers").pp("0"))?)
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

/// Qwen-Image VAE decoder with per-channel latent denormalization.
pub(crate) struct QwenImageVae {
    post_quant_conv: Conv2d,
    decoder: QwenImageDecoder2d,
    latents_mean: Tensor,
    latents_std: Tensor,
}

impl QwenImageVae {
    pub fn load(vae_path: &std::path::Path, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path], dtype, device)? };
        let post_quant_conv = load_3d_conv1x1_as_2d(16, 16, vb.pp("post_quant_conv"))?;
        let decoder = QwenImageDecoder2d::new(vb.pp("decoder"))?;

        let mean_vec: Vec<f32> = LATENTS_MEAN.iter().map(|&v| v as f32).collect();
        let std_vec: Vec<f32> = LATENTS_STD.iter().map(|&v| v as f32).collect();
        let latents_mean = Tensor::from_vec(mean_vec, (1, 16, 1, 1), device)?.to_dtype(dtype)?;
        let latents_std = Tensor::from_vec(std_vec, (1, 16, 1, 1), device)?.to_dtype(dtype)?;

        Ok(Self {
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
}
