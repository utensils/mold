//! Stable Diffusion component configuration owned by Mold.
//!
//! Candle's aggregate `StableDiffusionConfig` intentionally keeps component
//! fields private. Mold constructs components independently for single-file
//! loading, so these factories keep that application policy local rather than
//! requiring accessors in the framework.

use candle_transformers::models::stable_diffusion::{unet_2d, vae};

pub fn sd15_unet() -> unet_2d::UNet2DConditionModelConfig {
    let block = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    unet_2d::UNet2DConditionModelConfig {
        blocks: vec![
            block(320, Some(1), 8),
            block(640, Some(1), 8),
            block(1280, Some(1), 8),
            block(1280, None, 8),
        ],
        center_input_sample: false,
        cross_attention_dim: 768,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: false,
    }
}

pub fn sdxl_unet() -> unet_2d::UNet2DConditionModelConfig {
    let block = |out_channels, use_cross_attn, attention_head_dim| unet_2d::BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    unet_2d::UNet2DConditionModelConfig {
        blocks: vec![
            block(320, None, 5),
            block(640, Some(2), 10),
            block(1280, Some(10), 20),
        ],
        center_input_sample: false,
        cross_attention_dim: 2048,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

pub fn vae() -> vae::AutoEncoderKLConfig {
    vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_factories_match_candle_model_families() {
        let sd15 = sd15_unet();
        assert_eq!(sd15.cross_attention_dim, 768);
        assert_eq!(sd15.blocks.len(), 4);

        let sdxl = sdxl_unet();
        assert_eq!(sdxl.cross_attention_dim, 2048);
        assert_eq!(sdxl.blocks.len(), 3);

        assert_eq!(vae().latent_channels, 4);
    }
}
