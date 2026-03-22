//! ControlNet model for SD1.5.
//!
//! Implements the ControlNet architecture: a copy of the UNet encoder half
//! with zero-initialized convolution outputs at each skip connection.
//! The outputs are injected into the UNet via `forward_with_additional_residuals`.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn as nn;
use candle_transformers::models::stable_diffusion::embeddings::{TimestepEmbedding, Timesteps};
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModelConfig,
};
use candle_transformers::models::stable_diffusion::unet_2d_blocks::*;
use candle_transformers::models::with_tracing::{conv2d, Conv2d};

/// ControlNet conditioning embedding — projects a 3-channel control image
/// to the UNet's block channel dimension via a series of convolutions.
///
/// Architecture: 3 -> 16 -> 32 -> 96 -> 256 -> block_channels (e.g. 320)
struct ControlNetConditioningEmbedding {
    blocks: Vec<Conv2d>,
}

impl ControlNetConditioningEmbedding {
    fn new(
        vs: nn::VarBuilder,
        conditioning_channels: usize,
        block_channels: usize,
    ) -> Result<Self> {
        let channels = [16, 32, 96, 256];
        let mut blocks = Vec::new();

        // First conv: 3 -> 16
        let cfg_pad = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        blocks.push(conv2d(
            conditioning_channels,
            channels[0],
            3,
            cfg_pad,
            vs.pp("blocks.0"),
        )?);

        // Middle convs with stride 2: 16->32, 32->96, 96->256
        for i in 0..3 {
            let cfg_stride = nn::Conv2dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            };
            blocks.push(conv2d(
                channels[i],
                channels[i],
                3,
                cfg_pad,
                vs.pp(format!("blocks.{}", 2 * i + 1)),
            )?);
            blocks.push(conv2d(
                channels[i],
                channels[i + 1],
                3,
                cfg_stride,
                vs.pp(format!("blocks.{}", 2 * i + 2)),
            )?);
        }

        // Final conv: 256 -> block_channels
        blocks.push(conv2d(
            channels[3],
            block_channels,
            3,
            cfg_pad,
            vs.pp(format!("blocks.{}", 2 * 3 + 1)),
        )?);

        Ok(Self { blocks })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h)?;
            // Apply SiLU after every conv except the last
            if i < self.blocks.len() - 1 {
                h = nn::ops::silu(&h)?;
            }
        }
        Ok(h)
    }
}

/// Down block types mirroring the UNet (re-exported from candle).
enum ControlNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

/// ControlNet model: UNet encoder half with zero convolution outputs.
pub struct ControlNetModel {
    controlnet_cond_embedding: ControlNetConditioningEmbedding,
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<ControlNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    /// Zero-initialized 1x1 convolutions, one per down block residual.
    controlnet_down_blocks: Vec<Conv2d>,
    /// Zero-initialized 1x1 convolution for the mid block output.
    controlnet_mid_block: Conv2d,
}

impl ControlNetModel {
    /// Load a ControlNet model from safetensors weights.
    pub fn load<P: AsRef<std::path::Path>>(
        weights_path: P,
        device: &Device,
        dtype: DType,
        config: UNet2DConditionModelConfig,
    ) -> anyhow::Result<Self> {
        let vs =
            unsafe { nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };
        let model = Self::new(vs, config)?;
        Ok(model)
    }

    fn new(vs: nn::VarBuilder, config: UNet2DConditionModelConfig) -> Result<Self> {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;

        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(4, b_channels, 3, conv_cfg, vs.pp("conv_in"))?;

        let time_proj = Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vs.pp("time_embedding"), b_channels, time_embed_dim)?;

        // Conditioning embedding for the control image
        let controlnet_cond_embedding = ControlNetConditioningEmbedding::new(
            vs.pp("controlnet_cond_embedding"),
            3,
            b_channels,
        )?;

        // Build down blocks (same architecture as UNet encoder)
        let vs_db = vs.pp("down_blocks");
        let mut down_blocks = Vec::new();
        let mut zero_conv_channels = Vec::new();

        // Track channels for zero convs: conv_in output
        zero_conv_channels.push(b_channels);

        for i in 0..n_blocks {
            let BlockConfig {
                out_channels,
                use_cross_attn,
                attention_head_dim,
            } = config.blocks[i];

            let sliced_attention_size = match config.sliced_attention_size {
                Some(0) => Some(attention_head_dim / 2),
                _ => config.sliced_attention_size,
            };

            let in_channels = if i > 0 {
                config.blocks[i - 1].out_channels
            } else {
                b_channels
            };

            let db_cfg = DownBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: config.norm_eps,
                resnet_groups: config.norm_num_groups,
                add_downsample: i < n_blocks - 1,
                downsample_padding: config.downsample_padding,
                ..Default::default()
            };

            // Each resnet produces one residual
            for _ in 0..config.layers_per_block {
                zero_conv_channels.push(out_channels);
            }
            // Downsample produces one more residual
            if i < n_blocks - 1 {
                zero_conv_channels.push(out_channels);
            }

            if let Some(transformer_layers_per_block) = use_cross_attn {
                let ca_cfg = CrossAttnDownBlock2DConfig {
                    downblock: db_cfg,
                    attn_num_head_channels: attention_head_dim,
                    cross_attention_dim: config.cross_attention_dim,
                    sliced_attention_size,
                    use_linear_projection: config.use_linear_projection,
                    transformer_layers_per_block,
                };
                let block = CrossAttnDownBlock2D::new(
                    vs_db.pp(i.to_string()),
                    in_channels,
                    out_channels,
                    Some(time_embed_dim),
                    false, // use_flash_attn
                    ca_cfg,
                )?;
                down_blocks.push(ControlNetDownBlock::CrossAttn(block));
            } else {
                let block = DownBlock2D::new(
                    vs_db.pp(i.to_string()),
                    in_channels,
                    out_channels,
                    Some(time_embed_dim),
                    db_cfg,
                )?;
                down_blocks.push(ControlNetDownBlock::Basic(block));
            }
        }

        // Mid block
        let mid_transformer_layers_per_block = match config.blocks.last() {
            None => 1,
            Some(block) => block.use_cross_attn.unwrap_or(1),
        };
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: bl_attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            use_linear_projection: config.use_linear_projection,
            transformer_layers_per_block: mid_transformer_layers_per_block,
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            vs.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            false, // use_flash_attn
            mid_cfg,
        )?;

        // Zero convolutions for down block residuals
        let zero_cfg = nn::Conv2dConfig::default(); // 1x1 convolution, no padding
        let vs_zero = vs.pp("controlnet_down_blocks");
        let controlnet_down_blocks = zero_conv_channels
            .iter()
            .enumerate()
            .map(|(i, &ch)| conv2d(ch, ch, 1, zero_cfg, vs_zero.pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;

        // Zero convolution for mid block
        let controlnet_mid_block = conv2d(
            bl_channels,
            bl_channels,
            1,
            zero_cfg,
            vs.pp("controlnet_mid_block"),
        )?;

        Ok(Self {
            controlnet_cond_embedding,
            conv_in,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            controlnet_down_blocks,
            controlnet_mid_block,
        })
    }

    /// Forward pass: returns (down_block_residuals, mid_block_residual).
    ///
    /// These are injected into the UNet via `forward_with_additional_residuals`.
    pub fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        controlnet_cond: &Tensor,
        conditioning_scale: f64,
    ) -> anyhow::Result<(Vec<Tensor>, Tensor)> {
        let (bsize, _channels, _height, _width) = xs.dims4()?;
        let device = xs.device();

        // 1. Time embedding
        let emb = (Tensor::ones(bsize, xs.dtype(), device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;

        // 2. Pre-process: conv_in + conditioning embedding
        let xs = self.conv_in.forward(xs)?;
        let cond = self.controlnet_cond_embedding.forward(controlnet_cond)?;
        let mut xs = (xs + cond)?;

        // 3. Down blocks
        let mut down_block_res_samples = vec![xs.clone()];
        for down_block in &self.down_blocks {
            let (_xs, res_xs) = match down_block {
                ControlNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb))?,
                ControlNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))?
                }
            };
            down_block_res_samples.extend(res_xs);
            xs = _xs;
        }

        // 4. Mid block
        let mid = self
            .mid_block
            .forward(&xs, Some(&emb), Some(encoder_hidden_states))?;

        // 5. Apply zero convolutions and scale
        let mut controlnet_down_samples = Vec::with_capacity(down_block_res_samples.len());
        for (sample, zero_conv) in down_block_res_samples
            .iter()
            .zip(&self.controlnet_down_blocks)
        {
            let out = zero_conv.forward(sample)?;
            controlnet_down_samples.push((out * conditioning_scale)?);
        }

        let mid_out = self.controlnet_mid_block.forward(&mid)?;
        let mid_out = (mid_out * conditioning_scale)?;

        Ok((controlnet_down_samples, mid_out))
    }
}
