//! Qwen-Image VAE (AutoencoderKLQwenImage) with per-channel latent normalization.
//!
//! The Qwen-Image VAE is architecturally the same as a standard diffusers
//! AutoencoderKL but with per-channel normalization applied to the latent space:
//!   encode: (latent - latents_mean) / latents_std
//!   decode: latent * latents_std + latents_mean
//!
//! The underlying convnet uses base_dim=96, dim_mult=[1,2,4,4], z_dim=16.
//! We map this to the Z-Image VaeConfig format (block_out_channels=[96,192,384,384]).
//!
//! Note: This wrapper delegates the actual conv/attention computation to candle's
//! Z-Image AutoEncoderKL since the diffusers format is compatible. The Qwen-Image
//! VAE adds the per-channel normalization on top.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{AutoEncoderKL, VaeConfig};

/// Per-channel latent normalization constants from `vae/config.json`.
const LATENTS_MEAN: [f64; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const LATENTS_STD: [f64; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

/// Qwen-Image VAE configuration.
///
/// Maps Qwen-Image's `base_dim=96, dim_mult=[1,2,4,4], z_dim=16`
/// to the diffusers VaeConfig format.
fn qwen_image_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 16,
        block_out_channels: vec![96, 192, 384, 384], // base_dim * dim_mult
        layers_per_block: 2,                         // num_res_blocks
        // Qwen-Image VAE uses per-channel normalization instead of scale/shift,
        // so set scale_factor=1.0 and shift_factor=0.0 to disable the built-in
        // normalization in AutoEncoderKL.
        scaling_factor: 1.0,
        shift_factor: 0.0,
        norm_num_groups: 32,
    }
}

/// Qwen-Image VAE with per-channel latent normalization.
pub(crate) struct QwenImageVae {
    inner: AutoEncoderKL,
    /// Per-channel mean tensor: (1, 16, 1, 1)
    latents_mean: Tensor,
    /// Per-channel std tensor: (1, 16, 1, 1)
    latents_std: Tensor,
}

impl QwenImageVae {
    /// Load VAE weights from safetensors.
    pub fn load(vae_path: &std::path::Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg = qwen_image_vae_config();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path], dtype, device)? };
        let inner = AutoEncoderKL::new(&cfg, vb)?;

        // Create per-channel normalization tensors: (1, 16, 1, 1) for broadcasting
        let mean_vec: Vec<f32> = LATENTS_MEAN.iter().map(|&v| v as f32).collect();
        let std_vec: Vec<f32> = LATENTS_STD.iter().map(|&v| v as f32).collect();
        let latents_mean = Tensor::from_vec(mean_vec, (1, 16, 1, 1), device)?.to_dtype(dtype)?;
        let latents_std = Tensor::from_vec(std_vec, (1, 16, 1, 1), device)?.to_dtype(dtype)?;

        Ok(Self {
            inner,
            latents_mean,
            latents_std,
        })
    }

    /// Decode latents to image.
    ///
    /// Applies per-channel denormalization before passing to the decoder:
    ///   latent_denorm = latent * latents_std + latents_mean
    ///
    /// Returns image tensor of shape (B, 3, H, W) in range [-1, 1].
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        // Denormalize: latent * std + mean
        let denormed = latents
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?;
        Ok(self.inner.decode(&denormed)?)
    }
}
