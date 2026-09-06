//! Paint VAE adapter. Published PyTorch checkpoints are read by Candle's Rust
//! tensor parser; Python is used only to produce qualification fixtures.
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use mold_candle::stable_diffusion::vae::{AutoEncoderKL, DiagonalGaussianDistribution};

pub const LATENT_SCALE: f64 = 0.18215;
pub struct PaintVae {
    model: AutoEncoderKL,
    dtype: DType,
    device: Device,
}
impl PaintVae {
    pub fn load(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        ensure!(
            matches!(dtype, DType::F32 | DType::F16),
            "paint VAE requires float32 or float16"
        );
        let model = super::paint_weights::load_pth_exact(path, dtype, device, |vb| {
            AutoEncoderKL::new_with_numerics(
                vb,
                3,
                3,
                mold_candle::stable_diffusion::vae(),
                mold_candle::stable_diffusion::vae::VaeNumerics::Diffusers,
            )
        })?;
        Ok(Self {
            model,
            dtype,
            device: device.clone(),
        })
    }
    /// Pixels are NCHW in [-1,1]. Posterior noise is explicit and belongs to
    /// the pipeline RNG sequence, exactly as Tencent pipeline.py:163-169.
    pub fn encode_with_noise(&self, pixels: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = pixels.dims4()?;
        ensure!(
            batch > 0
                && channels == 3
                && height > 0
                && width > 0
                && height.is_multiple_of(8)
                && width.is_multiple_of(8),
            "invalid paint VAE pixel shape"
        );
        ensure!(
            noise.dims() == [batch, 4, height / 8, width / 8],
            "paint VAE posterior noise shape differs from latent shape"
        );
        let pixels = pixels.to_device(&self.device)?.to_dtype(self.dtype)?;
        let noise = noise.to_device(&self.device)?.to_dtype(self.dtype)?;
        let moments = self.model.encode_moments(&pixels)?;
        Ok(
            (DiagonalGaussianDistribution::new_clamped(&moments)?.sample_with_noise(&noise)?
                * LATENT_SCALE)?,
        )
    }
    /// Decode scaled diffusion latents to NCHW pixels in the VAE's [-1,1] range.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = latents.dims4()?;
        ensure!(
            batch > 0 && channels == 4 && height > 0 && width > 0,
            "invalid paint VAE latent shape"
        );
        let latents = (latents.to_device(&self.device)?.to_dtype(self.dtype)? / LATENT_SCALE)?;
        Ok(self.model.decode(&latents)?)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA, installed paint VAE .bin and retained oracle"]
    fn pretrained_paint_vae_matches_diffusers() -> Result<()> {
        let checkpoint = std::env::var("MOLD_PAINT_VAE_CHECKPOINT")?;
        let fixture = std::env::var("MOLD_PAINT_VAE_ORACLE")?;
        let output = std::path::PathBuf::from(std::env::var("MOLD_PAINT_VAE_RESULT")?);
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let tensors = candle_core::safetensors::load(fixture, &device)?;
        let dtype = tensors["pixels"].dtype();
        let model = PaintVae::load(Path::new(&checkpoint), dtype, &device)?;
        let latents = model.encode_with_noise(&tensors["pixels"], &tensors["noise"])?;
        let decoded = model.decode(&latents)?;
        let mut stages = std::collections::HashMap::new();
        let moments =
            model
                .model
                .encode_moments_with_observer(&tensors["pixels"], |name, value| {
                    stages.insert(name.to_string(), value.clone());
                    Ok(())
                })?;
        candle_core::safetensors::save(&stages, output.join("encoder.safetensors"))?;
        let posterior = DiagonalGaussianDistribution::new_clamped(&moments)?;
        let decoded_reference_latents = model.decode(&tensors["sampled"])?;
        candle_core::safetensors::save(
            &std::collections::HashMap::from([
                ("sampled".to_string(), latents.clone()),
                ("decoded".to_string(), decoded.clone()),
                ("mean".to_string(), posterior.mode()?),
                ("std".to_string(), posterior.std().clone()),
                (
                    "decoded_reference_latents".to_string(),
                    decoded_reference_latents,
                ),
            ]),
            output.join("actual.safetensors"),
        )?;
        let (latent_tolerance, decode_tolerance, rms_tolerance) = if dtype == DType::F16 {
            (0.01, 0.05, 0.005)
        } else {
            (0.0001, 0.001, 0.0001)
        };
        for (name, actual, tolerance) in [
            ("sampled", latents, latent_tolerance),
            ("decoded", decoded, decode_tolerance),
        ] {
            let delta = (actual.to_dtype(DType::F32)? - tensors[name].to_dtype(DType::F32)?)?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            anyhow::ensure!(delta.iter().all(|x| x.is_finite()), "nonfinite VAE {name}");
            let max = delta.iter().copied().fold(0., f32::max);
            let rms = (delta.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                / delta.len() as f64)
                .sqrt();
            eprintln!("paint VAE {dtype:?} {name}: max_abs={max}, rms={rms}");
            anyhow::ensure!(
                max < tolerance && rms < rms_tolerance,
                "paint VAE {name} diverges"
            );
        }
        Ok(())
    }
}
