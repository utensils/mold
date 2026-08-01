//! Image decoding and preprocessing utilities for img2img.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, RgbImage};

/// Normalization range for source images before VAE encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeRange {
    /// [-1, 1] for SD1.5, SDXL, FLUX, Flux.2, Qwen-Image, and SD3.
    MinusOneToOne,
    /// [0, 1] for Z-Image and VQ/VAE families that natively operate in that range.
    ZeroToOne,
}

/// Decode PNG/JPEG bytes into a [1, 3, H, W] tensor normalized to the specified range,
/// resized to target dimensions.
pub fn decode_source_image(
    bytes: &[u8],
    target_w: u32,
    target_h: u32,
    range: NormalizeRange,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode source image: {e}"))?;

    let img = img
        .resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3)
        .to_rgb8();

    rgb_image_to_tensor(img, range, device, dtype)
}

/// Decode and preprocess one FLUX.2 Dev reference exactly like BFL's
/// `default_prep`: reject sides below 64 px and aspect ratios above 8:1,
/// downscale only when the image exceeds its pixel cap, then center-crop both
/// dimensions down to a multiple of 16. References are never upscaled.
pub fn decode_flux2_reference_image(
    bytes: &[u8],
    max_pixels: u64,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| anyhow::anyhow!("failed to decode FLUX.2 reference image: {error}"))?;
    let image = preprocess_flux2_reference(image, max_pixels)?;
    rgb_image_to_tensor(image, NormalizeRange::MinusOneToOne, device, dtype)
}

fn preprocess_flux2_reference(image: DynamicImage, max_pixels: u64) -> Result<RgbImage> {
    const MIN_SIDE: u32 = 64;
    const MAX_ASPECT_RATIO: u32 = 8;
    const DIMENSION_MULTIPLE: u32 = 16;

    let mut image = image.to_rgb8();
    let (width, height) = image.dimensions();
    if width < MIN_SIDE || height < MIN_SIDE {
        bail!(
            "FLUX.2 reference images require both sides to be at least {MIN_SIDE}px (got {width}x{height})"
        );
    }
    let (long, short) = if width >= height {
        (width, height)
    } else {
        (height, width)
    };
    if u64::from(long) > u64::from(short) * u64::from(MAX_ASPECT_RATIO) {
        bail!(
            "FLUX.2 reference images support aspect ratios up to {MAX_ASPECT_RATIO}:1 (got {width}x{height})"
        );
    }

    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        let scale = (max_pixels as f64 / pixels as f64).sqrt();
        let scaled_width = ((width as f64 * scale) as u32).max(1);
        let scaled_height = ((height as f64 * scale) as u32).max(1);
        image = image::imageops::resize(
            &image,
            scaled_width,
            scaled_height,
            image::imageops::FilterType::Lanczos3,
        );
    }

    let crop_width = image.width() / DIMENSION_MULTIPLE * DIMENSION_MULTIPLE;
    let crop_height = image.height() / DIMENSION_MULTIPLE * DIMENSION_MULTIPLE;
    let left = (image.width() - crop_width) / 2;
    let top = (image.height() - crop_height) / 2;
    Ok(image::imageops::crop_imm(&image, left, top, crop_width, crop_height).to_image())
}

fn rgb_image_to_tensor(
    img: RgbImage,
    range: NormalizeRange,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();

    // raw is HWC u8, convert to f32 [0, 1]
    let data: Vec<f32> = raw.iter().map(|&v| v as f32 / 255.0).collect();

    // Reshape to [H, W, 3] then permute to [3, H, W]
    let tensor = Tensor::from_vec(data, (h, w, 3), &Device::Cpu)?;
    let tensor = tensor.permute((2, 0, 1))?; // [3, H, W]

    // Normalize to desired range
    let tensor = match range {
        NormalizeRange::MinusOneToOne => {
            // [0,1] -> [-1,1]: x * 2 - 1
            ((tensor * 2.0)? - 1.0)?
        }
        NormalizeRange::ZeroToOne => tensor,
    };

    // Add batch dimension: [1, 3, H, W]
    let tensor = tensor.unsqueeze(0)?;
    // Cast to target dtype and move to device
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    Ok(tensor)
}

/// Decode a mask image (PNG/JPEG) into a [1, 1, latent_h, latent_w] tensor with values in [0, 1].
/// White (255) = 1.0 = repaint region, Black (0) = 0.0 = preserve region.
/// RGB images are converted to grayscale.
pub fn decode_mask_image(
    bytes: &[u8],
    latent_height: usize,
    latent_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode mask image: {e}"))?;

    let img = img.resize_exact(
        latent_width as u32,
        latent_height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let gray = img.to_luma8();

    let data: Vec<f32> = gray.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

    let tensor = Tensor::from_vec(data, (1, 1, latent_height, latent_width), &Device::Cpu)?;
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    Ok(tensor)
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;

    fn encode_test_png(pixel: [u8; 3]) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(1, 1, Rgb(pixel));
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, ImageFormat::Png)
            .expect("encode PNG");
        out.into_inner()
    }

    fn decode_values(bytes: &[u8], range: NormalizeRange) -> Vec<f32> {
        decode_source_image(bytes, 1, 1, range, &Device::Cpu, DType::F32)
            .expect("decode source image")
            .flatten_all()
            .expect("flatten decoded tensor")
            .to_vec1::<f32>()
            .expect("decoded values")
    }

    #[test]
    fn zero_to_one_normalization_preserves_unit_interval() {
        let bytes = encode_test_png([0, 128, 255]);
        let values = decode_values(&bytes, NormalizeRange::ZeroToOne);

        assert_eq!(values.len(), 3);
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[1] - (128.0 / 255.0)).abs() < 1e-6);
        assert!((values[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn minus_one_to_one_normalization_centers_and_scales_pixels() {
        let bytes = encode_test_png([0, 128, 255]);
        let values = decode_values(&bytes, NormalizeRange::MinusOneToOne);

        assert_eq!(values.len(), 3);
        assert!((values[0] + 1.0).abs() < 1e-6);
        assert!((values[1] - ((128.0 / 255.0) * 2.0 - 1.0)).abs() < 1e-6);
        assert!((values[2] - 1.0).abs() < 1e-6);
    }

    fn encode_solid_png(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(width, height, Rgb([32, 64, 96]));
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, ImageFormat::Png)
            .expect("encode PNG");
        out.into_inner()
    }

    #[test]
    fn flux2_reference_prep_never_upscales_and_center_crops_to_sixteen() {
        let bytes = encode_solid_png(513, 527);
        let tensor = decode_flux2_reference_image(
            &bytes,
            mold_core::validation::FLUX2_DEV_SINGLE_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 512, 512]);
    }

    #[test]
    fn flux2_reference_prep_downscales_only_above_the_cap() {
        let bytes = encode_solid_png(4096, 1024);
        let tensor = decode_flux2_reference_image(
            &bytes,
            mold_core::validation::FLUX2_DEV_MULTI_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 512, 2048]);
    }

    #[test]
    fn flux2_reference_prep_rejects_small_sides_and_extreme_aspects() {
        let small = encode_solid_png(512, 63);
        assert!(decode_flux2_reference_image(
            &small,
            mold_core::validation::FLUX2_DEV_SINGLE_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err()
        .to_string()
        .contains("at least 64px"));

        let wide = encode_solid_png(1024, 64);
        assert!(decode_flux2_reference_image(
            &wide,
            mold_core::validation::FLUX2_DEV_SINGLE_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err()
        .to_string()
        .contains("up to 8:1"));
    }
}

/// Context for inpainting: holds pre-computed tensors needed during the denoising loop.
pub struct InpaintContext {
    /// VAE-encoded original latents (unnoised).
    pub original_latents: Tensor,
    /// Mask tensor [1, 1, latent_h, latent_w] with values in [0, 1].
    /// 1.0 = repaint, 0.0 = preserve.
    pub mask: Tensor,
    /// Noise tensor matching latent shape, for re-noising the original at each step.
    pub noise: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a tiny 4x4 red PNG for testing.
    fn tiny_png() -> Vec<u8> {
        let img = image::RgbImage::from_fn(4, 4, |_, _| image::Rgb([255, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_source_image_shape() {
        let png = tiny_png();
        let tensor = decode_source_image(
            &png,
            8,
            8,
            NormalizeRange::ZeroToOne,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 8, 8]);
    }

    #[test]
    fn decode_source_image_minus_one_to_one_range() {
        let png = tiny_png();
        let tensor = decode_source_image(
            &png,
            4,
            4,
            NormalizeRange::MinusOneToOne,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        // Red channel should be ~1.0 (was 255/255=1.0, then 1.0*2-1=1.0)
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min >= -1.0 - 0.01);
        assert!(max <= 1.0 + 0.01);
    }

    #[test]
    fn decode_source_image_zero_to_one_range() {
        let png = tiny_png();
        let tensor = decode_source_image(
            &png,
            4,
            4,
            NormalizeRange::ZeroToOne,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min >= 0.0 - 0.01);
        assert!(max <= 1.0 + 0.01);
    }

    #[test]
    fn decode_source_image_resize() {
        let png = tiny_png(); // 4x4 source
        let tensor = decode_source_image(
            &png,
            16,
            16,
            NormalizeRange::ZeroToOne,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 16, 16]);
    }
    // ── Mask decoding tests ───────────────────────────────────────────────

    /// Create a 4x4 white PNG (mask = repaint everywhere).
    fn white_mask_png() -> Vec<u8> {
        let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Create a 4x4 black PNG (mask = preserve everywhere).
    fn black_mask_png() -> Vec<u8> {
        let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_mask_shape() {
        let mask = white_mask_png();
        let tensor = decode_mask_image(&mask, 8, 8, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 8, 8]);
    }

    #[test]
    fn decode_mask_white_is_one() {
        let mask = white_mask_png();
        let tensor = decode_mask_image(&mask, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min > 0.99, "white mask should be ~1.0, got {min}");
    }

    #[test]
    fn decode_mask_black_is_zero() {
        let mask = black_mask_png();
        let tensor = decode_mask_image(&mask, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max < 0.01, "black mask should be ~0.0, got {max}");
    }

    #[test]
    fn decode_mask_rgb_converted_to_grayscale() {
        // Red RGB image -- grayscale luminance of pure red is ~76/255 ~ 0.3
        let rgb = tiny_png(); // 4x4 red
        let tensor = decode_mask_image(&rgb, 4, 4, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 4, 4]);
        let val = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(
            val > 0.1 && val < 0.5,
            "red -> grayscale should be ~0.3, got {val}"
        );
    }
}
