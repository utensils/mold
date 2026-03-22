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
}
