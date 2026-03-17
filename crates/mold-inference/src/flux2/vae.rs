//! Flux.2 VAE (`AutoencoderKLFlux2`)
//!
//! Same basic encoder/decoder architecture as FLUX.1's VAE (ResnetBlock + AttnBlock)
//! but with different channel configuration:
//! - `latent_channels`: 32 (vs 16)
//! - `block_out_channels`: [128, 256, 512, 512]
//! - `use_quant_conv` / `use_post_quant_conv`: true (unlike FLUX.1 which doesn't have these)
//! - `patch_size`: [2, 2] built into the VAE (not handled here, done in sampling state)
//!
//! The decoder is the primary path for inference (latent -> pixels).
//! We reuse candle's existing FLUX autoencoder architecture since the building blocks
//! (ResnetBlock, AttnBlock, DownBlock, UpBlock) are identical — only the config differs.

use candle_core::{Result, Tensor, D};
use candle_nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm, Module, VarBuilder};

/// Flux.2 VAE configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Flux2VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    /// Scale factor for latent space normalization.
    pub scale_factor: f64,
    /// Shift factor for latent space normalization.
    pub shift_factor: f64,
}

impl Flux2VaeConfig {
    /// Configuration for Flux.2 Klein-4B VAE.
    pub fn klein() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 32,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
            // These values need to be calibrated from the HF model config.
            // Using FLUX.1 defaults as a starting point; they may need adjustment.
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

// ---------------------------------------------------------------------------
// Building blocks (same architecture as FLUX.1 VAE)
// ---------------------------------------------------------------------------

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

#[derive(Debug, Clone)]
struct AttnBlock {
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
    norm: GroupNorm,
}

impl AttnBlock {
    fn new(in_c: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        let q = conv2d(in_c, in_c, 1, Default::default(), vb.pp("q"))?;
        let k = conv2d(in_c, in_c, 1, Default::default(), vb.pp("k"))?;
        let v = conv2d(in_c, in_c, 1, Default::default(), vb.pp("v"))?;
        let proj_out = conv2d(in_c, in_c, 1, Default::default(), vb.pp("proj_out"))?;
        let norm = group_norm(num_groups, in_c, 1e-6, vb.pp("norm"))?;
        Ok(Self {
            q,
            k,
            v,
            proj_out,
            norm,
        })
    }
}

impl Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let init_xs = xs;
        let xs = xs.apply(&self.norm)?;
        let q = xs.apply(&self.q)?;
        let k = xs.apply(&self.k)?;
        let v = xs.apply(&self.v)?;
        let (b, c, h, w) = q.dims4()?;
        let q = q.flatten_from(2)?.t()?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.t()?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.t()?.unsqueeze(1)?;
        let xs = scaled_dot_product_attention(&q, &k, &v)?;
        let xs = xs.squeeze(1)?.t()?.reshape((b, c, h, w))?;
        xs.apply(&self.proj_out)? + init_xs
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    nin_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(in_c: usize, out_c: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(num_groups, in_c, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(num_groups, out_c, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?;
        let nin_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("nin_shortcut"),
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
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
        match self.nin_shortcut.as_ref() {
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
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(in_c, in_c, 3, conv_cfg, vb.pp("conv"))?;
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
// Decoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct UpBlock {
    block: Vec<ResnetBlock>,
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
    up: Vec<UpBlock>,
}

impl Decoder {
    fn new(cfg: &Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let ch_mult = &cfg.block_out_channels;
        let base_ch = ch_mult[0]; // 128
        let mut block_in = *ch_mult.last().unwrap_or(&base_ch);

        let conv_in = conv2d(cfg.latent_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;
        let mid_block_1 = ResnetBlock::new(
            block_in,
            block_in,
            cfg.norm_num_groups,
            vb.pp("mid.block_1"),
        )?;
        let mid_attn_1 = AttnBlock::new(block_in, cfg.norm_num_groups, vb.pp("mid.attn_1"))?;
        let mid_block_2 = ResnetBlock::new(
            block_in,
            block_in,
            cfg.norm_num_groups,
            vb.pp("mid.block_2"),
        )?;

        // Build up blocks in reverse order (matching the encoder's down blocks)
        let mut up = Vec::with_capacity(ch_mult.len());
        let vb_u = vb.pp("up");
        for (i_level, &block_out) in ch_mult.iter().enumerate().rev() {
            let vb_u = vb_u.pp(i_level);
            let vb_b = vb_u.pp("block");
            let mut block = Vec::with_capacity(cfg.layers_per_block + 1);
            for i_block in 0..=cfg.layers_per_block {
                let b =
                    ResnetBlock::new(block_in, block_out, cfg.norm_num_groups, vb_b.pp(i_block))?;
                block.push(b);
                block_in = block_out;
            }
            let upsample = if i_level != 0 {
                Some(Upsample::new(block_in, vb_u.pp("upsample"))?)
            } else {
                None
            };
            up.push(UpBlock { block, upsample });
        }
        up.reverse();

        let norm_out = group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("norm_out"))?;
        let conv_out = conv2d(block_in, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
            up,
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
        for block in self.up.iter().rev() {
            for b in block.block.iter() {
                h = h.apply(b)?
            }
            if let Some(us) = block.upsample.as_ref() {
                h = h.apply(us)?
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
/// Decodes 32-channel latents to RGB images. Uses quant_conv/post_quant_conv
/// layers and has a different latent channel count than FLUX.1.
#[derive(Debug, Clone)]
pub struct Flux2AutoEncoder {
    decoder: Decoder,
    /// Optional post-quantization conv (latent_channels -> latent_channels).
    post_quant_conv: Option<Conv2d>,
    scale_factor: f64,
    shift_factor: f64,
}

impl Flux2AutoEncoder {
    pub fn new(cfg: &Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;

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

        Ok(Self {
            decoder,
            post_quant_conv,
            scale_factor: cfg.scale_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    /// Decode latents to pixel space.
    ///
    /// Input: (B, 32, H, W) latent tensor
    /// Output: (B, 3, H*8, W*8) RGB image in [-1, 1]
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        // Undo scale+shift normalization
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;

        // Apply post_quant_conv if present
        let xs = if let Some(pqc) = &self.post_quant_conv {
            xs.apply(pqc)?
        } else {
            xs
        };

        xs.apply(&self.decoder)
    }
}
