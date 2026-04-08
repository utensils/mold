//! Flux.2 VAE (`AutoencoderKLFlux2`) — diffusers weight format
//!
//! Standard AutoencoderKL encoder + decoder with:
//! - `latent_channels`: 32, `block_out_channels`: [128, 256, 512, 512]
//! - `use_quant_conv` / `use_post_quant_conv`: true
//! - BatchNorm2d latent normalization/denormalization (running_mean/running_var)
//! - 2x2 patchify/unpatchify around the normalization step
//!
//! Loads from HuggingFace diffusers format (e.g. `encoder.down_blocks.0`, `decoder.mid_block.resnets.0`).

use candle_core::{Result, Tensor, D};
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, Linear, Module, VarBuilder};

/// Flux.2 VAE configuration.
#[derive(Debug, Clone)]
pub struct Flux2VaeConfig {
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    /// Number of patchified channels: latent_channels * patch_h * patch_w.
    pub patchified_channels: usize,
    pub batch_norm_eps: f64,
}

impl Flux2VaeConfig {
    pub fn klein() -> Self {
        Self {
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 32,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
            patchified_channels: 32 * 2 * 2, // 128
            batch_norm_eps: 0.0001,
        }
    }
}

// ---------------------------------------------------------------------------
// Building blocks (diffusers naming)
// ---------------------------------------------------------------------------

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

/// Diffusers attention block using Linear layers (not Conv2d).
#[derive(Debug, Clone)]
struct AttnBlock {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    group_norm: GroupNorm,
}

impl AttnBlock {
    fn new(in_c: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            to_q: candle_nn::linear(in_c, in_c, vb.pp("to_q"))?,
            to_k: candle_nn::linear(in_c, in_c, vb.pp("to_k"))?,
            to_v: candle_nn::linear(in_c, in_c, vb.pp("to_v"))?,
            to_out: candle_nn::linear(in_c, in_c, vb.pp("to_out").pp("0"))?,
            group_norm: group_norm(num_groups, in_c, 1e-6, vb.pp("group_norm"))?,
        })
    }
}

impl Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let (b, c, h, w) = xs.dims4()?;
        let xs = xs.apply(&self.group_norm)?;
        // Reshape to (B, H*W, C) for linear attention
        let xs = xs.flatten_from(2)?.transpose(1, 2)?; // (B, H*W, C)
        let q = xs.apply(&self.to_q)?;
        let k = xs.apply(&self.to_k)?;
        let v = xs.apply(&self.to_v)?;
        let q = q.unsqueeze(1)?; // (B, 1, H*W, C)
        let k = k.unsqueeze(1)?;
        let v = v.unsqueeze(1)?;
        let xs = scaled_dot_product_attention(&q, &k, &v)?;
        let xs = xs.squeeze(1)?.apply(&self.to_out)?; // (B, H*W, C)
        let xs = xs.transpose(1, 2)?.reshape((b, c, h, w))?; // (B, C, H, W)
        xs + residual
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(in_c: usize, out_c: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("conv_shortcut"),
            )?)
        };
        Ok(Self {
            norm1: group_norm(num_groups, in_c, 1e-6, vb.pp("norm1"))?,
            conv1: conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?,
            norm2: group_norm(num_groups, out_c, 1e-6, vb.pp("norm2"))?,
            conv2: conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?,
            conv_shortcut,
        })
    }
}

impl Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.norm1)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv1)?
            .apply(&self.norm2)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv2)?;
        match self.conv_shortcut.as_ref() {
            None => xs + h,
            Some(c) => xs.apply(c)? + h,
        }
    }
}

#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn new(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv2d(
            in_c,
            in_c,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(h * 2, w * 2)?.apply(&self.conv)
    }
}

// ---------------------------------------------------------------------------
// Encoder (diffusers naming: down_blocks, mid_block, conv_norm_out)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv2d(
            in_c,
            in_c,
            3,
            Conv2dConfig {
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }
}

impl Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Asymmetric padding: pad right=1, bottom=1 (same as diffusers)
        let xs = xs.pad_with_zeros(D::Minus1, 0, 1)?; // width: right
        let xs = xs.pad_with_zeros(D::Minus2, 0, 1)?; // height: bottom
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone)]
struct DownBlock {
    resnets: Vec<ResnetBlock>,
    downsample: Option<Downsample>,
}

#[derive(Debug, Clone)]
struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_block_1: ResnetBlock,
    mid_attn_1: AttnBlock,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    fn new(cfg: &Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let ch_mult = &cfg.block_out_channels;

        let conv_in = conv2d(cfg.out_channels, ch_mult[0], 3, conv_cfg, vb.pp("conv_in"))?;

        // Down blocks (diffusers: down_blocks.{i}.resnets.{j}, down_blocks.{i}.downsamplers.0)
        let mut down_blocks = Vec::with_capacity(ch_mult.len());
        let mut block_in = ch_mult[0];
        let vb_d = vb.pp("down_blocks");
        for (i_level, &block_out) in ch_mult.iter().enumerate() {
            let vb_block = vb_d.pp(i_level);
            let vb_r = vb_block.pp("resnets");
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            for i_block in 0..cfg.layers_per_block {
                let b =
                    ResnetBlock::new(block_in, block_out, cfg.norm_num_groups, vb_r.pp(i_block))?;
                resnets.push(b);
                block_in = block_out;
            }
            let downsample = if i_level != ch_mult.len() - 1 {
                Some(Downsample::new(
                    block_in,
                    vb_block.pp("downsamplers").pp("0"),
                )?)
            } else {
                None
            };
            down_blocks.push(DownBlock {
                resnets,
                downsample,
            });
        }

        // Mid block: diffusers names mid_block.resnets.0/1 and mid_block.attentions.0
        let mid_vb = vb.pp("mid_block");
        let mid_block_1 = ResnetBlock::new(
            block_in,
            block_in,
            cfg.norm_num_groups,
            mid_vb.pp("resnets").pp("0"),
        )?;
        let mid_attn_1 = AttnBlock::new(
            block_in,
            cfg.norm_num_groups,
            mid_vb.pp("attentions").pp("0"),
        )?;
        let mid_block_2 = ResnetBlock::new(
            block_in,
            block_in,
            cfg.norm_num_groups,
            mid_vb.pp("resnets").pp("1"),
        )?;

        // Diffusers: conv_norm_out (not norm_out)
        let norm_out = group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("conv_norm_out"))?;
        // Output: 2 * latent_channels for mean + logvar (DiagonalGaussian)
        let conv_out = conv2d(
            block_in,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.apply(&self.conv_in)?;
        for block in &self.down_blocks {
            for r in &block.resnets {
                h = h.apply(r)?;
            }
            if let Some(ds) = block.downsample.as_ref() {
                h = h.apply(ds)?;
            }
        }
        h.apply(&self.mid_block_1)?
            .apply(&self.mid_attn_1)?
            .apply(&self.mid_block_2)?
            .apply(&self.norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ---------------------------------------------------------------------------
// Decoder (diffusers naming: mid_block, up_blocks, conv_norm_out)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Upsample>,
}

#[derive(Debug, Clone)]
struct Decoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: AttnBlock,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    up_blocks: Vec<UpBlock>,
}

impl Decoder {
    fn new(cfg: &Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let ch_mult = &cfg.block_out_channels;
        let mut block_in = *ch_mult.last().unwrap_or(&ch_mult[0]);

        let conv_in = conv2d(cfg.latent_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;

        // Mid block: diffusers names mid_block.resnets.0/1 and mid_block.attentions.0
        let mid_vb = vb.pp("mid_block");
        let mid_block_1 = ResnetBlock::new(
            block_in,
            block_in,
            cfg.norm_num_groups,
            mid_vb.pp("resnets").pp("0"),
        )?;
        let mid_attn_1 = AttnBlock::new(
            block_in,
            cfg.norm_num_groups,
            mid_vb.pp("attentions").pp("0"),
        )?;
        let mid_block_2 = ResnetBlock::new(
            block_in,
            block_in,
            cfg.norm_num_groups,
            mid_vb.pp("resnets").pp("1"),
        )?;

        // Up blocks (diffusers: up_blocks.{i}.resnets.{j}, up_blocks.{i}.upsamplers.0)
        let mut up_blocks = Vec::with_capacity(ch_mult.len());
        let vb_u = vb.pp("up_blocks");
        for (i_level, &block_out) in ch_mult.iter().enumerate().rev() {
            let vb_block = vb_u.pp(ch_mult.len() - 1 - i_level);
            let vb_r = vb_block.pp("resnets");
            let mut resnets = Vec::with_capacity(cfg.layers_per_block + 1);
            for i_block in 0..=cfg.layers_per_block {
                let b =
                    ResnetBlock::new(block_in, block_out, cfg.norm_num_groups, vb_r.pp(i_block))?;
                resnets.push(b);
                block_in = block_out;
            }
            let upsample = if i_level != 0 {
                Some(Upsample::new(block_in, vb_block.pp("upsamplers").pp("0"))?)
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsample });
        }
        up_blocks.reverse();

        // Diffusers: conv_norm_out (not norm_out)
        let norm_out = group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(block_in, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
            up_blocks,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.conv_in)?;
        let mut h = h
            .apply(&self.mid_block_1)?
            .apply(&self.mid_attn_1)?
            .apply(&self.mid_block_2)?;
        for block in self.up_blocks.iter().rev() {
            for r in &block.resnets {
                h = h.apply(r)?;
            }
            if let Some(us) = block.upsample.as_ref() {
                h = h.apply(us)?;
            }
        }
        h.apply(&self.norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

// ---------------------------------------------------------------------------
// Flux2AutoEncoder
// ---------------------------------------------------------------------------

/// Flux.2 VAE (AutoencoderKLFlux2).
///
/// Encodes RGB images to 32-channel latents and decodes them back, using
/// BatchNorm2d latent normalization/denormalization on patchified latents (128 channels).
#[derive(Debug, Clone)]
pub struct Flux2AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Option<Conv2d>,
    post_quant_conv: Option<Conv2d>,
    bn_running_mean: Tensor,
    bn_running_std: Tensor,
    latent_channels: usize,
}

impl Flux2AutoEncoder {
    pub fn new(cfg: &Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;

        let quant_conv = if cfg.use_quant_conv {
            Some(conv2d(
                2 * cfg.latent_channels,
                2 * cfg.latent_channels,
                1,
                Default::default(),
                vb.pp("quant_conv"),
            )?)
        } else {
            None
        };

        let post_quant_conv = if cfg.use_post_quant_conv {
            Some(conv2d(
                cfg.latent_channels,
                cfg.latent_channels,
                1,
                Default::default(),
                vb.pp("post_quant_conv"),
            )?)
        } else {
            None
        };

        // BatchNorm running statistics for latent normalization/denormalization
        let bn_vb = vb.pp("bn");
        let bn_running_mean = bn_vb
            .get(cfg.patchified_channels, "running_mean")?
            .reshape((1, cfg.patchified_channels, 1, 1))?;
        let bn_running_var = bn_vb.get(cfg.patchified_channels, "running_var")?;
        let bn_running_std = (bn_running_var + cfg.batch_norm_eps)?.sqrt()?.reshape((
            1,
            cfg.patchified_channels,
            1,
            1,
        ))?;

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            bn_running_mean,
            bn_running_std,
            latent_channels: cfg.latent_channels,
        })
    }

    /// Encode pixel-space image to latent space.
    ///
    /// Input: (B, 3, H, W) in [0, 1] → Output: (B, 32, H/8, W/8)
    ///
    /// Pipeline:
    /// 1. Encoder conv/resnet/attention → (B, 64, H/8, W/8)  [2 * latent_channels]
    /// 2. Optional quant_conv
    /// 3. DiagonalGaussian: split mean/logvar, sample z = mean + std * eps
    /// 4. BN normalization: patchify → (z - mean) / std → unpatchify
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.encoder)?;

        // Optional quant_conv (applied before splitting mean/logvar)
        let h = if let Some(qc) = &self.quant_conv {
            h.apply(qc)?
        } else {
            h
        };

        // DiagonalGaussian: split into mean and logvar, then sample
        let (_b, c2, _h_dim, _w_dim) = h.dims4()?;
        let c = c2 / 2;
        let mean = h.narrow(1, 0, c)?;
        let logvar = h.narrow(1, c, c)?.clamp(-30.0, 20.0)?;
        let std = (&logvar * 0.5)?.exp()?;
        let z = (&mean + &std * mean.randn_like(0.0, 1.0)?)?;

        // BN normalization (reverse of decode's denormalization):
        // Patchify → normalize → unpatchify
        let (b, c, h_s, w_s) = z.dims4()?;
        let z = z
            .reshape((b, c, h_s / 2, 2, w_s / 2, 2))?
            .permute((0, 1, 3, 5, 2, 4))?
            .reshape((b, c * 4, h_s / 2, w_s / 2))?;
        let z = z
            .broadcast_sub(&self.bn_running_mean)?
            .broadcast_div(&self.bn_running_std)?;
        let z = z
            .reshape((b, self.latent_channels, 4, h_s / 2, w_s / 2))?
            .permute((0, 1, 3, 4, 2))?
            .reshape((b, self.latent_channels, h_s / 2, w_s / 2, 2, 2))?
            .permute((0, 1, 2, 4, 3, 5))?
            .reshape((b, self.latent_channels, h_s, w_s))?;

        Ok(z)
    }

    /// Decode latents to pixel space.
    ///
    /// Input: (B, 32, H, W) → Output: (B, 3, H*8, W*8) in [-1, 1]
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;

        // Patchify: (B, 32, H, W) → (B, 128, H/2, W/2)
        let xs = xs
            .reshape((b, c, h / 2, 2, w / 2, 2))?
            .permute((0, 1, 3, 5, 2, 4))?
            .reshape((b, c * 4, h / 2, w / 2))?;

        // BatchNorm denormalization: latents * sqrt(var + eps) + mean
        let xs = xs
            .broadcast_mul(&self.bn_running_std)?
            .broadcast_add(&self.bn_running_mean)?;

        // Unpatchify: (B, 128, H/2, W/2) → (B, 32, H, W)
        let xs = xs
            .reshape((b, self.latent_channels, 4, h / 2, w / 2))?
            .permute((0, 1, 3, 4, 2))?
            .reshape((b, self.latent_channels, h / 2, w / 2, 2, 2))?
            .permute((0, 1, 2, 4, 3, 5))?
            .reshape((b, self.latent_channels, h, w))?;

        let xs = if let Some(pqc) = &self.post_quant_conv {
            xs.apply(pqc)?
        } else {
            xs
        };

        xs.apply(&self.decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn klein_vae_config() {
        let cfg = Flux2VaeConfig::klein();
        assert_eq!(cfg.latent_channels, 32);
        assert_eq!(cfg.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(cfg.patchified_channels, 128); // 32 * 2 * 2
        assert!(cfg.use_quant_conv);
        assert!(cfg.use_post_quant_conv);
    }

    #[test]
    fn patchify_unpatchify_roundtrip() {
        let dev = Device::Cpu;
        let b = 1;
        let c = 32;
        let h = 8;
        let w = 8;
        let xs = Tensor::randn(0f32, 1., (b, c, h, w), &dev).unwrap();

        // Patchify: (1, 32, 8, 8) → (1, 128, 4, 4)
        let patched = xs
            .reshape((b, c, h / 2, 2, w / 2, 2))
            .unwrap()
            .permute((0, 1, 3, 5, 2, 4))
            .unwrap()
            .reshape((b, c * 4, h / 2, w / 2))
            .unwrap();
        assert_eq!(patched.dims(), &[1, 128, 4, 4]);

        // Unpatchify: (1, 128, 4, 4) → (1, 32, 8, 8)
        let unpatched = patched
            .reshape((b, c, 4, h / 2, w / 2))
            .unwrap()
            .permute((0, 1, 3, 4, 2))
            .unwrap()
            .reshape((b, c, h / 2, w / 2, 2, 2))
            .unwrap()
            .permute((0, 1, 2, 4, 3, 5))
            .unwrap()
            .reshape((b, c, h, w))
            .unwrap();
        assert_eq!(unpatched.dims(), &[1, 32, 8, 8]);

        // Values should match
        let diff = (xs - unpatched)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "patchify/unpatchify roundtrip failed: diff={diff}"
        );
    }

    #[test]
    fn bn_denormalization_formula() {
        let dev = Device::Cpu;
        // Simulate: denormed = latents * sqrt(var + eps) + mean
        let mean = Tensor::from_vec(vec![1.0f32, -1.0], (1, 2, 1, 1), &dev).unwrap();
        let var = Tensor::from_vec(vec![4.0f32, 9.0], (1, 2), &dev).unwrap();
        let eps = 0.0001;
        let std = (var + eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();

        let latents = Tensor::ones((1, 2, 1, 1), candle_core::DType::F32, &dev).unwrap();
        let result = latents
            .broadcast_mul(&std)
            .unwrap()
            .broadcast_add(&mean)
            .unwrap();

        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Channel 0: 1.0 * sqrt(4.0001) + 1.0 ≈ 3.0
        assert!((vals[0] - 3.0).abs() < 0.01, "ch0: {}", vals[0]);
        // Channel 1: 1.0 * sqrt(9.0001) + (-1.0) ≈ 2.0
        assert!((vals[1] - 2.0).abs() < 0.01, "ch1: {}", vals[1]);
    }

    #[test]
    fn test_klein_decoder_channel_progression() {
        let cfg = Flux2VaeConfig::klein();
        // Must have exactly 4 stages
        assert_eq!(cfg.block_out_channels.len(), 4, "expected 4 decoder stages");
        // Last two channels must be equal (the plateau at the deepest resolution)
        assert_eq!(
            cfg.block_out_channels[2], cfg.block_out_channels[3],
            "last two block_out_channels should be equal: {} != {}",
            cfg.block_out_channels[2], cfg.block_out_channels[3]
        );
    }

    #[test]
    fn test_patchify_non_square() {
        // Patchify with non-square spatial dims: (1, 32, 8, 16) → (1, 128, 4, 8)
        let dev = Device::Cpu;
        let b = 1usize;
        let c = 32usize;
        let h = 8usize;
        let w = 16usize;
        let xs = Tensor::randn(0f32, 1., (b, c, h, w), &dev).unwrap();

        let patched = xs
            .reshape((b, c, h / 2, 2, w / 2, 2))
            .unwrap()
            .permute((0, 1, 3, 5, 2, 4))
            .unwrap()
            .reshape((b, c * 4, h / 2, w / 2))
            .unwrap();

        assert_eq!(
            patched.dims(),
            &[1, 128, 4, 8],
            "non-square patchify should produce (1, 128, H/2, W/2)"
        );
    }

    #[test]
    fn test_bn_denormalization_zero_variance() {
        // When variance is 0, eps prevents division by zero: sqrt(0 + eps) > 0
        let dev = Device::Cpu;
        let eps = 0.0001f64;
        let var = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &dev).unwrap();
        let std = (var + eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let mean = Tensor::zeros((1, 2, 1, 1), candle_core::DType::F32, &dev).unwrap();

        let latents = Tensor::ones((1, 2, 1, 1), candle_core::DType::F32, &dev).unwrap();
        let result = latents
            .broadcast_mul(&std)
            .unwrap()
            .broadcast_add(&mean)
            .unwrap();

        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected_std = (eps as f32).sqrt(); // sqrt(0.0001) = 0.01
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "channel {i} produced non-finite value with zero variance"
            );
            assert!(
                (v - expected_std).abs() < 1e-6,
                "channel {i}: expected {expected_std}, got {v}"
            );
        }
    }
}
