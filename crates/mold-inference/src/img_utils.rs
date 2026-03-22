//! Image decoding and preprocessing utilities for img2img.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Normalization range for source images before VAE encoding.
pub enum NormalizeRange {
    /// [-1, 1] for SD1.5 and SDXL (UNet-based models).
    MinusOneToOne,
    /// [0, 1] for FLUX, SD3, Z-Image, Flux.2, Qwen-Image (flow-matching models).
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

    let img = img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();

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

/// Decode a mask image (PNG/JPEG bytes) into a [1, 1, latent_h, latent_w] tensor in [0, 1].
/// White (255) = 1.0 = repaint region; black (0) = 0.0 = preserve region.
pub fn decode_mask_image(
    bytes: &[u8],
    latent_height: usize,
    latent_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode mask image: {e}"))?;

    // Convert to grayscale, resize to latent dimensions
    let img = img.into_luma8();
    let img = image::imageops::resize(
        &img,
        latent_width as u32,
        latent_height as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();

    // Normalize to [0, 1]
    let data: Vec<f32> = raw.iter().map(|&v| v as f32 / 255.0).collect();

    // Shape: [1, 1, H, W]
    let tensor = Tensor::from_vec(data, (1, 1, h, w), &Device::Cpu)?;
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    Ok(tensor)
}

/// Context for inpainting: holds pre-encoded original latents, the mask, and noise for re-noising.
pub struct InpaintContext {
    /// VAE-encoded original image latents (unnoised).
    pub original_latents: Tensor,
    /// Mask tensor [1, 1, latent_h, latent_w] in [0, 1]. 1.0 = repaint, 0.0 = preserve.
    pub mask: Tensor,
    /// Noise tensor for re-noising the original latents at each step.
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

    /// Create a 4x4 white PNG (all 255) for mask testing.
    fn white_png() -> Vec<u8> {
        let img = image::RgbImage::from_fn(4, 4, |_, _| image::Rgb([255, 255, 255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Create a 4x4 black PNG (all 0) for mask testing.
    fn black_png() -> Vec<u8> {
        let img = image::RgbImage::from_fn(4, 4, |_, _| image::Rgb([0, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_mask_shape() {
        let png = white_png();
        let tensor = decode_mask_image(&png, 8, 8, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 8, 8]);
    }

    #[test]
    fn decode_mask_white_is_one() {
        let png = white_png();
        let tensor = decode_mask_image(&png, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min > 0.99, "white mask should be ~1.0, got {min}");
    }

    #[test]
    fn decode_mask_black_is_zero() {
        let png = black_png();
        let tensor = decode_mask_image(&png, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max < 0.01, "black mask should be ~0.0, got {max}");
    }

    #[test]
    fn decode_mask_rgb_converted_to_grayscale() {
        // A colored image should be converted to grayscale before processing
        let png = tiny_png(); // red image
        let tensor = decode_mask_image(&png, 4, 4, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 4, 4]);
        // Red in grayscale is ~76/255 ≈ 0.3
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(
            max > 0.1 && max < 0.5,
            "red in grayscale should be ~0.3, got {max}"
        );
    }
}
