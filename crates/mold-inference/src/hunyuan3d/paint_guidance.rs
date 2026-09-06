//! Tencent paint three-branch guidance, pipeline.py:663-688 at 82920d64.
//! The two updates remain separate: folding them changes half-precision output.
use candle_core::{DType, Device, Result, Tensor};

/// Flat [three branches * two materials * views,4,H,W] to [two materials * views,4,H,W].
/// The published paint pipeline fixes guidance to three, with zero rescaling.
/// Its default call does not forward camera azimuths, so every view has weight1.
pub fn paint_guidance(
    prediction: &Tensor,
    views: usize,
    azimuths: Option<&[f64]>,
) -> Result<Tensor> {
    let (rows, channels, height, width) = prediction.dims4()?;
    if !(1..=6).contains(&views)
        || rows != 6 * views
        || channels != 4
        || height == 0
        || width == 0
        || !matches!(prediction.dtype(), DType::F32 | DType::F16)
        || azimuths.is_some_and(|values| {
            values.len() != views
                || values
                    .iter()
                    .any(|v| !v.is_finite() || !(0. ..=360.).contains(v))
        })
    {
        candle_core::bail!("invalid paint guidance shape, dtype or camera azimuths")
    }
    let weights = (0..views)
        .map(|index| {
            let angle = azimuths.map_or(0., |values| values[index]);
            if angle < 90. {
                angle / 90. + 1.
            } else if angle < 330. {
                2.
            } else {
                -angle / 90. + 5.
            }
        })
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(weights, (1, views), &Device::Cpu)?
        .to_dtype(prediction.dtype())?
        .repeat((2, 1))?
        .reshape((2 * views, 1, 1, 1))?
        .to_device(prediction.device())?;
    let scale = weight
        .to_dtype(DType::F32)?
        .affine(3., 0.)?
        .to_dtype(prediction.dtype())?;
    let unconditioned = prediction.narrow(0, 0, 2 * views)?;
    let reference = prediction.narrow(0, 2 * views, 2 * views)?;
    let full = prediction.narrow(0, 4 * views, 2 * views)?;
    let reference_delta = (&reference - &unconditioned)?;
    let full_delta = (full - &reference)?;
    let guided = (unconditioned + reference_delta.broadcast_mul(&scale)?)?;
    guided + full_delta.broadcast_mul(&scale)?
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paint_guidance_matches_tencent() -> Result<()> {
        compare(&Device::Cpu)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn paint_guidance_matches_tencent_cuda() -> Result<()> {
        compare(&Device::new_cuda(0)?)
    }
    fn compare(device: &Device) -> Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-guidance.safetensors"),
            device,
        )?;
        for label in ["f32", "f16"] {
            for views in [1, 2, 6] {
                for mode in ["default", "azimuth"] {
                    let key = format!("{label}.{views}.{mode}");
                    let angles = [0., 45., 90., 300., 330., 359.];
                    let actual = paint_guidance(
                        &fixture[&format!("{key}.input")],
                        views,
                        if mode == "azimuth" {
                            Some(&angles[..views])
                        } else {
                            None
                        },
                    )?;
                    let expected = &fixture[&format!("{key}.expected")];
                    let delta = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
                    let maximum = delta.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(maximum < 1e-6, "{key}: max={maximum}");
                }
            }
        }
        Ok(())
    }
}
