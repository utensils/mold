//! CPU material blending in UV texture space.

use anyhow::{ensure, Result};

/// RGB values in the material's original encoding. Albedo remains sRGB and
/// metallic/roughness remains data: neither stream is gamma transformed here.
pub struct BakedTexture {
    pub size: u32,
    pub colors: Vec<[f32; 3]>,
    pub trusted: Vec<bool>,
}

/// Streaming equivalent of Tencent MeshRender.py:1353-1380. Only accumulated
/// color and weight remain resident between views, instead of six full maps.
pub struct TextureBaker {
    size: u32,
    sums: Vec<[f32; 3]>,
    weights: Vec<f32>,
    views: usize,
    valid: bool,
}

#[cfg(test)]
fn cosine_weight(cosine: f32, weight: f32) -> f32 {
    // PyTorch 2.5.1 PowKernel.cu only specializes exponents 2 and 3. Tencent's
    // tensor ** 4 therefore takes the general floating-point pow path.
    weight * cosine.powf(4.)
}

pub(super) fn cosine_weights(
    cosine: &[f32],
    weight: f32,
    device: &candle_core::Device,
) -> Result<Vec<f32>> {
    Ok(
        candle_core::Tensor::from_vec(cosine.to_vec(), cosine.len(), device)?
            .powf(4.)?
            .affine(f64::from(weight), 0.)?
            .to_device(&candle_core::Device::Cpu)?
            .to_vec1::<f32>()?,
    )
}

impl TextureBaker {
    pub fn new(size: u32, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&size),
            "paint texture size must be 1 through 4096"
        );
        checkpoint()?;
        let pixels = size as usize * size as usize;
        let sums = vec![[0.; 3]; pixels];
        checkpoint()?;
        let weights = vec![0.; pixels];
        checkpoint()?;
        Ok(Self {
            size,
            sums,
            weights,
            views: 0,
            valid: true,
        })
    }

    /// Tencent's strict >99% overlap uses positive accumulated weights, even
    /// below the final1e-8 trust threshold. Validation failures preserve state;
    /// cancellation during mutation invalidates the session, so partial color
    /// can never be finalized or used by a later view.
    pub fn add_view(
        &mut self,
        colors: &[[f32; 3]],
        cosine: &[f32],
        weight: f32,
        device: &candle_core::Device,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        ensure!(self.valid, "paint bake was cancelled during accumulation");
        ensure!(
            self.views < 6,
            "paint bake accepts at most six ordered views"
        );
        ensure!(
            colors.len() == self.sums.len() && cosine.len() == colors.len(),
            "paint projection dimensions differ from texture"
        );
        ensure!(
            weight.is_finite() && weight >= 0.,
            "invalid paint camera weight"
        );
        checkpoint()?;
        for (index, (color, &cosine)) in colors.iter().zip(cosine).enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            ensure!(
                color
                    .iter()
                    .all(|v| v.is_finite() && (0. ..=1.).contains(v)),
                "paint colors must be finite unit values"
            );
            ensure!(
                cosine.is_finite() && cosine >= 0.,
                "invalid paint cosine map"
            );
        }
        checkpoint()?;
        // Tencent applies tensor ** 4 on the active accelerator. This matters
        // numerically: CUDA's libdevice powf is not bit-identical to host powf.
        let weighted = cosine_weights(cosine, weight, device)?;
        checkpoint()?;
        let mut visible = 0usize;
        let mut painted = 0usize;
        for (index, (&value, &previous)) in weighted.iter().zip(&self.weights).enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            ensure!(value.is_finite(), "paint cosine weight overflow");
            if value > 0. {
                visible += 1;
                painted += usize::from(previous > 0.);
            }
        }
        checkpoint()?;
        // An empty view's Torch0/0 is NaN, so its >.99 is false.
        if visible > 0 && (painted as f32 / visible as f32) > 0.99 {
            self.views += 1;
            return Ok(false);
        }
        for (index, (&value, &previous)) in weighted.iter().zip(&self.weights).enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            ensure!(
                (value + previous).is_finite(),
                "paint accumulated weight overflow"
            );
        }
        self.valid = false;
        for (index, ((sum, accumulated), (color, weight))) in self
            .sums
            .iter_mut()
            .zip(&mut self.weights)
            .zip(colors.iter().zip(weighted))
            .enumerate()
        {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            for channel in 0..3 {
                sum[channel] += color[channel] * weight;
            }
            *accumulated += weight;
        }
        checkpoint()?;
        self.valid = true;
        self.views += 1;
        Ok(true)
    }

    pub fn finish(mut self, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<BakedTexture> {
        ensure!(self.valid, "paint bake was cancelled during accumulation");
        checkpoint()?;
        let mut trusted = Vec::with_capacity(self.weights.len());
        for (index, (color, &weight)) in self.sums.iter_mut().zip(&self.weights).enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            for value in color {
                *value /= weight.max(1e-8);
            }
            trusted.push(weight > 1e-8);
        }
        checkpoint()?;
        Ok(BakedTexture {
            size: self.size,
            colors: self.sums,
            trusted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fourth_power_uses_torch_cuda_general_pow_path() {
        assert_eq!(
            super::cosine_weight(
                std::hint::black_box(f32::from_bits(1_062_387_645)),
                std::hint::black_box(1.),
            )
            .to_bits(),
            1_055_599_141
        );
    }
    #[test]
    fn paint_bake_matches_tencent_overlap_and_trust() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-bake.safetensors"),
            &Device::Cpu,
        )?;
        for case in ["overlap", "tiny"] {
            let colors = fixture[&format!("{case}.colors")]
                .flatten_all()?
                .to_vec1::<f32>()?;
            let cosine = fixture[&format!("{case}.cosine")]
                .flatten_all()?
                .to_vec1::<f32>()?;
            let weights = fixture[&format!("{case}.weights")].to_vec1::<f32>()?;
            for count in 1..=4 {
                let mut baker = TextureBaker::new(10, &mut || Ok(()))?;
                for index in 0..count {
                    let colors: Vec<[f32; 3]> = colors[index * 300..(index + 1) * 300]
                        .chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]])
                        .collect();
                    baker.add_view(
                        &colors,
                        &cosine[index * 100..(index + 1) * 100],
                        weights[index],
                        &Device::Cpu,
                        &mut || Ok(()),
                    )?;
                }
                let output = baker.finish(&mut || Ok(()))?;
                let expected = fixture[&format!("{case}.texture.{count}")]
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let max = output
                    .colors
                    .iter()
                    .flatten()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max <= 1e-7, "{case} count{count}: {max}");
                let trust: Vec<u8> = output
                    .trusted
                    .iter()
                    .map(|&value| u8::from(value))
                    .collect();
                assert_eq!(
                    trust,
                    fixture[&format!("{case}.trust.{count}")]
                        .flatten_all()?
                        .to_vec1::<u8>()?,
                    "{case} count{count}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn paint_bake_rejects_invalid_views_without_changing_prior_result() -> anyhow::Result<()> {
        assert!(TextureBaker::new(0, &mut || Ok(())).is_err());
        assert!(TextureBaker::new(4097, &mut || Ok(())).is_err());
        let mut baker = TextureBaker::new(1, &mut || Ok(()))?;
        assert!(baker.add_view(&[[0.2, 0.4, 0.6]], &[1.], 1., &Device::Cpu, &mut || Ok(()))?);
        assert!(baker
            .add_view(&[[f32::NAN, 0., 0.]], &[1.], 1., &Device::Cpu, &mut || Ok(
                ()
            ))
            .is_err());
        assert!(baker
            .add_view(&[], &[], 1., &Device::Cpu, &mut || Ok(()))
            .is_err());
        assert!(baker
            .add_view(&[[0.; 3]], &[f32::INFINITY], 1., &Device::Cpu, &mut || Ok(
                ()
            ))
            .is_err());
        assert!(baker
            .add_view(&[[0.; 3]], &[1.], -1., &Device::Cpu, &mut || Ok(()))
            .is_err());
        let result = baker.finish(&mut || Ok(()))?;
        assert_eq!(result.colors, [[0.2, 0.4, 0.6]]);
        assert_eq!(result.trusted, [true]);
        Ok(())
    }

    #[test]
    fn paint_bake_cancel_never_exposes_partial_accumulation() -> anyhow::Result<()> {
        let colors = vec![[0.5; 3]; 128 * 128];
        let cosine = vec![1.; 128 * 128];
        let mut cancellations = 0;
        let mut invalidated = 0;
        for stop in 1..32 {
            let mut baker = TextureBaker::new(128, &mut || Ok(()))?;
            let mut calls = 0;
            let result = baker.add_view(&colors, &cosine, 1., &Device::Cpu, &mut || {
                calls += 1;
                if calls == stop {
                    anyhow::bail!("cancel bake")
                }
                Ok(())
            });
            if result.is_ok() {
                break;
            }
            cancellations += 1;
            if let Ok(output) = baker.finish(&mut || Ok(())) {
                assert!(output.colors.iter().all(|color| *color == [0.; 3]));
                assert!(output.trusted.iter().all(|v| !v));
            } else {
                invalidated += 1;
            }
        }
        assert!(cancellations > 1 && invalidated > 0);
        let baker = TextureBaker::new(1, &mut || Ok(()))?;
        assert!(baker
            .finish(&mut || anyhow::bail!("cancel finalize"))
            .is_err());
        Ok(())
    }
}
