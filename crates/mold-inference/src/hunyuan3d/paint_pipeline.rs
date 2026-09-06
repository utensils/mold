//! Staged paint inference with explicit noise and observable tensor boundaries.

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use super::{
    dino2::{Dinov2Config, Dinov2Model},
    paint_denoiser::{PaintDenoiser, PaintInputs},
    paint_pixels::{materials_from_pixels, normalize_views, PaintMaterials},
    paint_vae::PaintVae,
};

/// Prepared images and the four independent draws from the caller's RNG.
/// Appearance is DINO-normalized NCHW; other images are BVCHW in [0,1].
/// Geometry images may remain F16 when reference pixels/model are F32.
pub struct PaintRequest {
    pub appearance: Tensor,
    pub reference: Tensor,
    pub normal: Tensor,
    pub position: Tensor,
    pub reference_noise: Tensor,
    pub normal_noise: Tensor,
    pub position_noise: Tensor,
    pub initial_noise: Tensor,
}

impl PaintRequest {
    fn reference_for_model(&self, dtype: DType) -> Result<Tensor> {
        // Reference pixels are cast before normalization (pipeline.py:223);
        // PIL geometry conditions deliberately keep their input precision.
        Ok(self.reference.to_dtype(dtype)?)
    }
    /// Check every tensor before opening a checkpoint or executing a network.
    pub fn validate(&self) -> Result<(usize, usize)> {
        let (_, views, _, height, _) = self.normal.dims5()?;
        normalize_views(&self.normal)?;
        normalize_views(&self.position)?;
        normalize_views(&self.reference)?;
        ensure!(
            self.position.dims() == self.normal.dims(),
            "paint geometry views differ"
        );
        ensure!(
            self.reference.dims() == [1, 1, 3, height, height],
            "paint requires one matching reference image"
        );
        ensure!(
            self.appearance.dims() == [1, 3, 224, 224],
            "invalid paint appearance pixels"
        );
        let size = height / 8;
        ensure!(
            self.reference_noise.dims() == [1, 1, 4, size, size],
            "invalid paint reference noise"
        );
        for noise in [&self.normal_noise, &self.position_noise] {
            ensure!(
                noise.dims() == [1, views, 4, size, size],
                "invalid paint geometry noise"
            );
        }
        ensure!(
            self.initial_noise.dims() == [2 * views, 4, size, size],
            "invalid paint diffusion noise"
        );
        for tensor in [
            &self.appearance,
            &self.reference_noise,
            &self.normal_noise,
            &self.position_noise,
            &self.initial_noise,
        ] {
            ensure!(
                matches!(tensor.dtype(), DType::F16 | DType::F32),
                "paint tensors require float16 or float32"
            );
            ensure!(
                tensor
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .is_finite(),
                "nonfinite paint input"
            );
        }
        Ok((views, size))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintStage {
    Appearance,
    Reference,
    Normal,
    Position,
    Denoise,
    Decode,
}

/// A stage starts with no tensor; completed boundaries expose their tensor for
/// retained qualification evidence. Denoise reports initial noise at step0 and
/// the completed sample at steps1..15. Returning an error cancels further work.
pub struct PaintEvent<'a> {
    pub stage: PaintStage,
    pub step: usize,
    pub total: usize,
    pub tensor: Option<&'a Tensor>,
}

fn report(
    callback: &mut impl FnMut(PaintEvent<'_>) -> Result<()>,
    stage: PaintStage,
    step: usize,
    total: usize,
    tensor: Option<&Tensor>,
) -> Result<()> {
    callback(PaintEvent {
        stage,
        step,
        total,
        tensor,
    })
}

pub struct PaintCheckpoints<'a> {
    pub dino: &'a Path,
    pub vae: &'a Path,
    pub unet: &'a Path,
}

impl PaintCheckpoints<'_> {
    /// Tencent multiview_utils.py:78-103 and pipeline.py:219-265,717-728.
    /// Each lexical model scope ends before the next network is loaded. VAE
    /// encoding and decoding deliberately reload the same checkpoint around
    /// denoising, avoiding simultaneous VAE/UNet residency.
    pub fn run(
        &self,
        input: &PaintRequest,
        dtype: DType,
        device: &Device,
        mut callback: impl FnMut(PaintEvent<'_>) -> Result<()>,
    ) -> Result<PaintMaterials> {
        ensure!(
            matches!(dtype, DType::F16 | DType::F32),
            "paint requires float16 or float32"
        );
        let (views, _) = input.validate()?;
        report(&mut callback, PaintStage::Appearance, 0, 1, None)?;
        let dino = {
            let vb = crate::weight_loader::load_safetensors_with_progress(
                &[self.dino],
                dtype,
                device,
                "paint DINO",
                &crate::progress::ProgressReporter::default(),
            )?;
            let model = Dinov2Model::new(&Dinov2Config::paint_giant(), vb)?;
            model.forward(&input.appearance.to_device(device)?.to_dtype(dtype)?)?
        };
        report(&mut callback, PaintStage::Appearance, 1, 1, Some(&dino))?;
        report(&mut callback, PaintStage::Reference, 0, 1, None)?;
        let (reference, normal, position) = {
            let model = PaintVae::load(self.vae, dtype, device)?;
            let reference = model.encode_views_with_noise(
                &input.reference_for_model(dtype)?,
                &input.reference_noise,
            )?;
            report(&mut callback, PaintStage::Reference, 1, 1, Some(&reference))?;
            report(&mut callback, PaintStage::Normal, 0, 1, None)?;
            let normal = model.encode_views_with_noise(&input.normal, &input.normal_noise)?;
            report(&mut callback, PaintStage::Normal, 1, 1, Some(&normal))?;
            report(&mut callback, PaintStage::Position, 0, 1, None)?;
            let position = model.encode_views_with_noise(&input.position, &input.position_noise)?;
            report(&mut callback, PaintStage::Position, 1, 1, Some(&position))?;
            (reference, normal, position)
        };
        report(&mut callback, PaintStage::Denoise, 0, 15, None)?;
        let latents = {
            let model = PaintDenoiser::load(self.unet, dtype, device)?;
            let conditioning = PaintInputs {
                normal,
                position,
                reference,
                dino,
                position_maps: input.position.to_device(device)?,
            };
            model.denoise(
                &conditioning,
                &input.initial_noise.to_device(device)?.to_dtype(dtype)?,
                |step, total, sample| {
                    report(
                        &mut callback,
                        PaintStage::Denoise,
                        step,
                        total,
                        Some(sample),
                    )
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
                },
            )?
        };
        report(&mut callback, PaintStage::Decode, 0, 1, None)?;
        let pixels = {
            let model = PaintVae::load(self.vae, dtype, device)?;
            model.decode(&latents)?
        };
        report(&mut callback, PaintStage::Decode, 1, 1, Some(&pixels))?;
        materials_from_pixels(&pixels, views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn request() -> anyhow::Result<PaintRequest> {
        let image = Tensor::zeros((1, 1, 3, 64, 64), DType::F16, &Device::Cpu)?;
        let noise = Tensor::zeros((1, 1, 4, 8, 8), DType::F32, &Device::Cpu)?;
        Ok(PaintRequest {
            appearance: Tensor::zeros((1, 3, 224, 224), DType::F32, &Device::Cpu)?,
            reference: image.to_dtype(DType::F32)?,
            normal: image.clone(),
            position: image,
            reference_noise: noise.clone(),
            normal_noise: noise.clone(),
            position_noise: noise,
            initial_noise: Tensor::zeros((2, 4, 8, 8), DType::F32, &Device::Cpu)?,
        })
    }

    #[test]
    fn paint_pipeline_validates_all_noise_before_loading() -> anyhow::Result<()> {
        let mut input = request()?;
        assert_eq!(input.validate()?, (1, 8));
        input.position_noise = Tensor::zeros((1, 2, 4, 8, 8), DType::F32, &Device::Cpu)?;
        assert!(input.validate().is_err());
        let mut input = request()?;
        input.initial_noise = Tensor::full(f32::NAN, (2, 4, 8, 8), &Device::Cpu)?;
        assert!(input.validate().is_err());
        let mut input = request()?;
        input.reference = Tensor::full(2f32, (1, 1, 3, 64, 64), &Device::Cpu)?;
        assert!(input.validate().is_err());
        Ok(())
    }

    #[test]
    fn paint_pipeline_cancel_precedes_checkpoint_loading() -> anyhow::Result<()> {
        let missing = std::path::Path::new("/missing-paint-checkpoint");
        let checkpoints = PaintCheckpoints {
            dino: missing,
            vae: missing,
            unet: missing,
        };
        let mut events = Vec::new();
        let result = checkpoints.run(&request()?, DType::F32, &Device::Cpu, |event| {
            events.push(event.stage);
            anyhow::bail!("test cancelled")
        });
        assert_eq!(result.err().unwrap().to_string(), "test cancelled");
        assert_eq!(events, [PaintStage::Appearance]);
        Ok(())
    }

    #[test]
    fn paint_pipeline_reference_cast_precedes_normalization() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-pixels.safetensors"),
            &Device::Cpu,
        )?;
        let mut input = request()?;
        input.reference = fixture["f32.input"].clone();
        let actual =
            normalize_views(&input.reference_for_model(DType::F16)?)?.to_dtype(DType::F32)?;
        let expected = fixture["f16.normalized"].to_dtype(DType::F32)?;
        let delta = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(delta, 0.);
        assert_eq!(input.reference_for_model(DType::F16)?.dtype(), DType::F16);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA, installed paint models and full Tencent pipeline oracle"]
    fn pretrained_paint_pipeline_matches_tencent() -> anyhow::Result<()> {
        use std::{collections::HashMap, path::PathBuf};
        let oracle = PathBuf::from(std::env::var("MOLD_PAINT_PIPELINE_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_PIPELINE_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let dino = PathBuf::from(std::env::var("MOLD_PAINT_DINO_CHECKPOINT")?);
        let vae = PathBuf::from(std::env::var("MOLD_PAINT_VAE_CHECKPOINT")?);
        let unet = PathBuf::from(std::env::var("MOLD_PAINT_UNET_WEIGHTS")?);
        let _scope = crate::conv_policy::ConvScope::for_family("hunyuan3d");
        let device = Device::new_cuda(0)?;
        let fixture =
            candle_core::safetensors::load(oracle.join("pipeline.safetensors"), &Device::Cpu)?;
        let input = PaintRequest {
            appearance: fixture["input.appearance"].clone(),
            reference: fixture["input.reference"].clone(),
            normal: fixture["input.normal"].clone(),
            position: fixture["input.position"].clone(),
            reference_noise: fixture["input.reference_noise"].clone(),
            normal_noise: fixture["input.normal_noise"].clone(),
            position_noise: fixture["input.position_noise"].clone(),
            initial_noise: fixture["input.initial_noise"].clone(),
        };
        let mut failures = Vec::new();
        let mut measurements = Vec::new();
        let start = std::time::Instant::now();
        let materials = PaintCheckpoints { dino: &dino, vae: &vae, unet: &unet }.run(
            &input, DType::F16, &device, |event| {
                let Some(actual) = event.tensor else {
                    eprintln!("starting {:?}", event.stage);
                    return Ok(());
                };
                let name = match event.stage {
                    PaintStage::Appearance => "expected.appearance".to_string(),
                    PaintStage::Reference => "expected.reference".to_string(),
                    PaintStage::Normal => "expected.normal".to_string(),
                    PaintStage::Position => "expected.position".to_string(),
                    PaintStage::Decode => "expected.decode".to_string(),
                    PaintStage::Denoise if event.step == 0 => "input.initial_noise".to_string(),
                    PaintStage::Denoise => format!("expected.denoise.{:02}", event.step),
                };
                let actual = actual.to_device(&Device::Cpu)?;
                candle_core::safetensors::save(&HashMap::from([(name.clone(), actual.clone())]), output.join(format!("{name}.safetensors")))?;
                let delta = (actual.to_dtype(DType::F32)? - fixture[&name].to_dtype(DType::F32)?)?;
                let max = delta.abs()?.max_all()?.to_scalar::<f32>()?;
                let rms = delta.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
                let (max_bound, rms_bound) = match event.stage {
                    PaintStage::Appearance => (0.2, 0.03),
                    PaintStage::Reference | PaintStage::Normal | PaintStage::Position => (0.01, 0.005),
                    PaintStage::Denoise => (0.02, 0.002),
                    PaintStage::Decode => (0.05, 0.005),
                };
                eprintln!("{name}: max={max} rms={rms}");
                if !max.is_finite() || !rms.is_finite() || max > max_bound || rms > rms_bound {
                    failures.push(name.clone());
                }
                measurements.push(serde_json::json!({"name":name,"max":max,"rms":rms,"max_bound":max_bound,"rms_bound":rms_bound}));
                Ok(())
            },
        )?;
        let expected_materials =
            materials_from_pixels(&fixture["expected.decode"], input.normal.dim(1)?)?;
        for (role, images, expected_images) in [
            ("albedo", materials.albedo, expected_materials.albedo),
            (
                "mr",
                materials.metallic_roughness,
                expected_materials.metallic_roughness,
            ),
        ] {
            for (index, image) in images.iter().enumerate() {
                image.save(output.join(format!("{role}-{index:02}.png")))?;
                let expected =
                    image::open(oracle.join(format!("{role}-{index:02}.png")))?.to_rgb8();
                // Check production-size conversion and material ordering against
                // the actual upstream PIL files, independently of tensor error.
                ensure!(
                    expected_images[index] == expected,
                    "upstream material conversion differs for {role}-{index}"
                );
                ensure!(
                    image.dimensions() == expected.dimensions(),
                    "material dimensions differ"
                );
                let differences: Vec<f64> = image
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .collect();
                let max = differences.iter().copied().fold(0f64, f64::max);
                let rms = (differences.iter().map(|v| v * v).sum::<f64>()
                    / differences.len() as f64)
                    .sqrt();
                // Propagate the existing decode limits through x/2+.5 and
                // scaling by255, with at most two bytes for half/uint8 rounding.
                let max_bound = 0.05 * 127.5 + 2.;
                let rms_bound = 0.005 * 127.5 + 2.;
                let name = format!("{role}-{index:02}.png");
                if max > max_bound || rms > rms_bound {
                    failures.push(name.clone());
                }
                measurements.push(serde_json::json!({"name":name,"max":max,"rms":rms,"max_bound":max_bound,"rms_bound":rms_bound}));
            }
        }
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "seconds": start.elapsed().as_secs_f64(), "measurements": measurements, "failures": failures,
            }))?,
        )?;
        ensure!(
            failures.is_empty(),
            "paint pipeline parity failed: {failures:?}"
        );
        Ok(())
    }
}
