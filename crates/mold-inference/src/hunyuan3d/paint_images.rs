//! Image preparation for the paint reference and ordered geometry views.

use crate::pillow_resize::{resize, Filter};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, RgbaImage};

/// CPU tensors ready for the staged paint runner. Geometry is always half,
/// reference pixels use model precision, and DINO normalization stays float32.
pub struct PaintImages {
    pub appearance: Tensor,
    pub reference: Tensor,
    pub normal: Tensor,
    pub position: Tensor,
}

impl PaintImages {
    /// Tencent textureGenPipeline.py:136-145 then multiview_utils.py:69-103.
    /// RGB callers supply alpha255; source transparency is composited AFTER
    /// the first resize. Normal and position view order is preserved.
    pub fn prepare(
        appearance: &RgbaImage,
        normal: &[RgbImage],
        position: &[RgbImage],
        size: u32,
        dtype: DType,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        ensure!(
            (1..=6).contains(&normal.len()) && normal.len() == position.len(),
            "invalid paint view counts"
        );
        ensure!(
            (64..=512).contains(&size) && size.is_power_of_two(),
            "invalid paint view size"
        );
        ensure!(
            matches!(dtype, DType::F16 | DType::F32),
            "invalid paint image precision"
        );
        ensure!(
            normal
                .iter()
                .chain(position)
                .all(|image| image.width() > 0 && image.height() > 0),
            "empty paint condition image"
        );
        checkpoint()?;
        let appearance = resize_over_white(appearance, 512, 512, checkpoint)?;
        let appearance = resize(&appearance, size, size, Filter::Bicubic, checkpoint)?;
        checkpoint()?;
        let dino = super::dino2::preprocess_paint(&appearance)?;
        checkpoint()?;
        let reference = unit_views(std::slice::from_ref(&appearance), dtype)?;
        let mut prepare_views = |images: &[RgbImage]| -> Result<Tensor> {
            let resized = images
                .iter()
                .map(|image| resize(image, size, size, Filter::Bicubic, checkpoint))
                .collect::<Result<Vec<_>>>()?;
            checkpoint()?;
            unit_views(&resized, DType::F16)
        };
        let normal = prepare_views(normal)?;
        let position = prepare_views(position)?;
        checkpoint()?;
        Ok(Self {
            appearance: dino,
            reference,
            normal,
            position,
        })
    }
}

fn unit_views(images: &[RgbImage], dtype: DType) -> Result<Tensor> {
    let width = images[0].width() as usize;
    let height = images[0].height() as usize;
    let plane = width * height;
    let mut pixels = vec![0f32; images.len() * 3 * plane];
    for (view, image) in images.iter().enumerate() {
        for (pixel, rgb) in image.pixels().enumerate() {
            for channel in 0..3 {
                pixels[(view * 3 + channel) * plane + pixel] = f32::from(rgb[channel]) / 255.;
            }
        }
    }
    Ok(
        Tensor::from_vec(pixels, (1, images.len(), 3, height, width), &Device::Cpu)?
            .to_dtype(dtype)?,
    )
}

fn divide_255(value: u32) -> u8 {
    let value = value + 128;
    (((value >> 8) + value) >> 8) as u8
}

fn resize_over_white(
    image: &RgbaImage,
    width: u32,
    height: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbImage> {
    ensure!(
        image.width() > 0 && image.height() > 0 && width > 0 && height > 0,
        "empty paint appearance image"
    );
    ensure!(
        u64::from(image.width()) * u64::from(image.height()) * 6 <= 512 * 1024 * 1024,
        "paint appearance exceeds preprocessing buffer budget"
    );
    checkpoint()?;
    // Pillow Image.py:2401 returns a copy before premultiplication when size
    // is unchanged. Preserve this branch: low-alpha round trips lose bytes.
    if image.dimensions() == (width, height) {
        let mut output = RgbImage::new(width, height);
        for (y, row) in image.rows().enumerate() {
            checkpoint()?;
            for (x, rgba) in row.enumerate() {
                let alpha = u32::from(rgba[3]);
                output.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb(std::array::from_fn(|c| {
                        divide_255(255 * (255 - alpha) + u32::from(rgba[c]) * alpha)
                    })),
                );
            }
        }
        return Ok(output);
    }
    // Pillow Convert.c:421-450 and Resample.c: each premultiplied color and
    // alpha channel uses the same bicubic U8 filter, then integer unpremultiply.
    let mut colors = RgbImage::new(image.width(), image.height());
    let mut alpha = RgbImage::new(image.width(), image.height());
    for (y, row) in image.rows().enumerate() {
        checkpoint()?;
        for (x, rgba) in row.enumerate() {
            let a = u32::from(rgba[3]);
            colors.put_pixel(
                x as u32,
                y as u32,
                image::Rgb(std::array::from_fn(|c| divide_255(u32::from(rgba[c]) * a))),
            );
            alpha.put_pixel(x as u32, y as u32, image::Rgb([rgba[3]; 3]));
        }
    }
    let mut colors = resize(&colors, width, height, Filter::Bicubic, checkpoint)?;
    let alpha = resize(&alpha, width, height, Filter::Bicubic, checkpoint)?;
    for y in 0..height {
        checkpoint()?;
        for x in 0..width {
            let a = u32::from(alpha.get_pixel(x, y)[0]);
            for color in &mut colors.get_pixel_mut(x, y).0 {
                let straight = if a == 0 || a == 255 {
                    u32::from(*color)
                } else {
                    (255 * u32::from(*color) / a).min(255)
                };
                *color = divide_255(255 * (255 - a) + straight * a);
            }
        }
    }
    Ok(colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn paint_rgba_resize_composition_matches_pillow() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-images.safetensors"),
            &Device::Cpu,
        )?;
        for name in ["down", "up", "same"] {
            let input = &fixture[&format!("{name}.input")];
            let (height, width, _) = input.dims3()?;
            let image = image::RgbaImage::from_raw(
                width as u32,
                height as u32,
                input.flatten_all()?.to_vec1::<u8>()?,
            )
            .unwrap();
            let expected = &fixture[&format!("{name}.expected")];
            let (height, width, _) = expected.dims3()?;
            let actual = resize_over_white(&image, width as u32, height as u32, &mut || Ok(()))?;
            assert_eq!(
                actual.as_raw(),
                &expected.flatten_all()?.to_vec1::<u8>()?,
                "{name}"
            );
        }
        Ok(())
    }

    #[test]
    fn paint_images_validate_and_cancel_before_preprocessing() -> anyhow::Result<()> {
        let image = RgbaImage::new(9, 7);
        let views = [RgbImage::new(8, 8)];
        let mut callbacks = 0;
        let error = PaintImages::prepare(&image, &views, &views, 64, DType::F32, &mut || {
            callbacks += 1;
            anyhow::bail!("cancel image preparation")
        })
        .err()
        .unwrap();
        assert_eq!(callbacks, 1);
        assert_eq!(error.to_string(), "cancel image preparation");
        for size in [0, 32, 63, 65, 1024] {
            assert!(
                PaintImages::prepare(&image, &views, &views, size, DType::F32, &mut || Ok(()))
                    .is_err()
            );
        }
        assert!(PaintImages::prepare(&image, &views, &[], 64, DType::F32, &mut || Ok(())).is_err());
        assert!(
            PaintImages::prepare(&image, &views, &views, 64, DType::BF16, &mut || Ok(())).is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires retained six-view Tencent source images and pipeline fixture; CPU only"]
    fn pretrained_paint_image_preparation_matches_tencent() -> anyhow::Result<()> {
        use std::path::PathBuf;
        let oracle = PathBuf::from(std::env::var("MOLD_PAINT_PIPELINE_ORACLE")?);
        let conditions = PathBuf::from(std::env::var("MOLD_PAINT_CONDITION_IMAGES")?);
        let image = image::open(std::env::var("MOLD_PAINT_APPEARANCE_IMAGE")?)?;
        let appearance = match std::env::var("MOLD_PAINT_APPEARANCE_MODE").as_deref() {
            Ok("rgba") => image.to_rgba8(),
            Ok("rgb") | Err(_) => image::DynamicImage::ImageRgb8(image.to_rgb8()).to_rgba8(),
            Ok(mode) => anyhow::bail!("unsupported appearance mode {mode}"),
        };
        let output = PathBuf::from(std::env::var("MOLD_PAINT_IMAGES_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let normal = (0..6)
            .map(|index| {
                Ok(image::open(conditions.join(format!("condition-{index:02}.png")))?.to_rgb8())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let position = (6..12)
            .map(|index| {
                Ok(image::open(conditions.join(format!("condition-{index:02}.png")))?.to_rgb8())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let actual = PaintImages::prepare(
            &appearance,
            &normal,
            &position,
            512,
            DType::F16,
            &mut || Ok(()),
        )?;
        let fixture =
            candle_core::safetensors::load(oracle.join("pipeline.safetensors"), &Device::Cpu)?;
        for (name, actual) in [
            ("appearance", actual.appearance.to_dtype(DType::F16)?),
            ("reference", actual.reference),
            ("normal", actual.normal),
            ("position", actual.position),
        ] {
            let name = format!("input.{name}");
            candle_core::safetensors::save(
                &std::collections::HashMap::from([(name.clone(), actual.clone())]),
                output.join(format!("{name}.safetensors")),
            )?;
            let delta = (actual.to_dtype(DType::F32)? - fixture[&name].to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            eprintln!("{name}: max={delta}");
            assert_eq!(delta, 0., "{name}");
        }
        Ok(())
    }
}
