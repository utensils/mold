//! Tensor/image boundaries of Tencent paint conditioning and material decode.

use anyhow::{ensure, Result};
use candle_core::{DType, Tensor};
use image::RgbImage;

/// Paint's two material streams, retaining the same view order in both.
/// MR bytes are data channels; they are not converted through an sRGB curve.
pub struct PaintMaterials {
    pub albedo: Vec<RgbImage>,
    pub metallic_roughness: Vec<RgbImage>,
}

/// Tencent pipeline.py:160-169. Normalize in the INPUT dtype, before a caller
/// casts to the VAE dtype. PIL conditions are half even for an F32 VAE.
/// Keeping subtraction and multiplication separate preserves that rounding.
pub fn normalize_views(images: &Tensor) -> Result<Tensor> {
    let (batch, views, channels, height, width) = images.dims5()?;
    ensure!(
        batch == 1
            && (1..=6).contains(&views)
            && channels == 3
            && height == width
            && (64..=512).contains(&height)
            && height.is_power_of_two()
            && matches!(images.dtype(), DType::F16 | DType::F32),
        "invalid paint pixel dimensions or dtype"
    );
    let check = images.to_dtype(DType::F32)?;
    ensure!(
        check.sum_all()?.to_scalar::<f32>()?.is_finite()
            && check.min_all()?.to_scalar::<f32>()? >= 0.
            && check.max_all()?.to_scalar::<f32>()? <= 1.,
        "paint pixels must be finite and in [0,1]"
    );
    let flat = images.reshape((batch * views, channels, height, width))?;
    Ok(((flat - 0.5)? * 2.)?)
}

/// Diffusers 0.30 image_processor.py:105-116,144-164: denormalize in model
/// dtype, clamp, convert to float32, multiply by255, then ties-to-even rounding.
/// The upstream pipeline returns all albedo views followed by all MR views.
pub fn materials_from_pixels(pixels: &Tensor, views: usize) -> Result<PaintMaterials> {
    let (batch, channels, height, width) = pixels.dims4()?;
    ensure!(
        (1..=6).contains(&views)
            && batch == 2 * views
            && channels == 3
            && (1..=512).contains(&height)
            && (1..=512).contains(&width)
            && matches!(pixels.dtype(), DType::F16 | DType::F32),
        "invalid decoded paint pixels"
    );
    // Images must reach host memory for export anyway. One bulk transfer also
    // lets us reject every nonfinite value before clamping could hide it.
    let values = pixels
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite decoded paint pixels"
    );
    let half = pixels.dtype() == DType::F16;
    let plane = height * width;
    let mut images = Vec::with_capacity(batch);
    for image_index in 0..batch {
        let mut bytes = vec![0u8; 3 * plane];
        for pixel_index in 0..plane {
            for channel in 0..3 {
                let mut value = values[(image_index * 3 + channel) * plane + pixel_index] / 2.;
                if half {
                    value = half::f16::from_f32(value).to_f32();
                }
                value += 0.5;
                if half {
                    value = half::f16::from_f32(value).to_f32();
                }
                bytes[pixel_index * 3 + channel] =
                    (value.clamp(0., 1.) * 255.).round_ties_even() as u8;
            }
        }
        images.push(
            RgbImage::from_raw(width as u32, height as u32, bytes).expect("validated RGB extent"),
        );
    }
    let metallic_roughness = images.split_off(views);
    Ok(PaintMaterials {
        albedo: images,
        metallic_roughness,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    fn fixture() -> candle_core::Result<std::collections::HashMap<String, Tensor>> {
        candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-pixels.safetensors"),
            &Device::Cpu,
        )
    }
    #[test]
    fn paint_view_normalization_matches_tencent() -> anyhow::Result<()> {
        let tensors = fixture()?;
        for name in ["f32", "f16", "f16_to_f32"] {
            let input = &tensors[&format!("{name}.input")];
            let before = input
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let actual = normalize_views(input)?;
            let expected = &tensors[&format!("{name}.normalized")];
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "{name}"
            );
            assert_eq!(
                input
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
                before
            );
        }
        Ok(())
    }
    #[test]
    fn paint_material_conversion_matches_diffusers() -> anyhow::Result<()> {
        let tensors = fixture()?;
        for name in ["f32", "f16"] {
            let actual = materials_from_pixels(&tensors[&format!("{name}.decoded")], 2)?;
            assert_eq!(actual.albedo.len(), 2);
            assert_eq!(actual.metallic_roughness.len(), 2);
            let bytes: Vec<u8> = actual
                .albedo
                .iter()
                .chain(&actual.metallic_roughness)
                .flat_map(|image| image.as_raw().iter().copied())
                .collect();
            assert_eq!(
                bytes,
                tensors[&format!("{name}.rgb")]
                    .flatten_all()?
                    .to_vec1::<u8>()?,
                "{name}"
            );
        }
        Ok(())
    }
    #[test]
    fn paint_pixel_boundaries_reject_invalid_inputs() -> anyhow::Result<()> {
        let device = Device::Cpu;
        for shape in [
            (1, 0, 3, 64, 64),
            (2, 1, 3, 64, 64),
            (1, 1, 4, 64, 64),
            (1, 1, 3, 63, 64),
        ] {
            assert!(normalize_views(&Tensor::zeros(shape, DType::F32, &device)?).is_err());
        }
        for value in [-0.1f32, 1.1, f32::NAN, f32::INFINITY] {
            let pixels = Tensor::full(value, (1, 1, 3, 64, 64), &device)?;
            assert!(normalize_views(&pixels).is_err());
        }
        let decoded = Tensor::zeros((4, 3, 8, 16), DType::F32, &device)?;
        assert!(materials_from_pixels(&decoded, 1).is_err());
        assert!(materials_from_pixels(&decoded, 0).is_err());
        let decoded = Tensor::full(f32::INFINITY, (2, 3, 8, 8), &device)?;
        assert!(materials_from_pixels(&decoded, 1).is_err());
        Ok(())
    }
}
