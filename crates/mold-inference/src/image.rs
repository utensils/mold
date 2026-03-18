use anyhow::{bail, Result};
use candle_core::Tensor;
use mold_core::OutputFormat;

/// Encode a candle tensor [3, H, W] of u8 values into PNG or JPEG bytes.
pub(crate) fn encode_image(
    img: &Tensor,
    format: OutputFormat,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        bail!("expected 3 channels, got {c}");
    }
    let _ = (h, w); // dims used implicitly via from_raw

    let img_data = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let rgb_image = image::RgbImage::from_raw(width, height, img_data)
        .ok_or_else(|| anyhow::anyhow!("failed to create image from tensor data"))?;

    let mut buf = std::io::Cursor::new(Vec::new());
    match format {
        OutputFormat::Png => rgb_image.write_to(&mut buf, image::ImageFormat::Png)?,
        OutputFormat::Jpeg => rgb_image.write_to(&mut buf, image::ImageFormat::Jpeg)?,
    }

    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Build a 3xHxW solid-red tensor (R=255, G=0, B=0).
    fn solid_red_tensor(h: usize, w: usize) -> Tensor {
        let mut data = vec![0u8; 3 * h * w];
        // Channel 0 (R) = 255, channels 1 and 2 stay 0
        for i in 0..(h * w) {
            data[i] = 255;
        }
        Tensor::from_vec(data, (3, h, w), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
    }

    #[test]
    fn test_encode_png_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4).unwrap();
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_encode_jpeg_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4).unwrap();
        assert!(bytes.len() >= 2);
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_encode_wrong_channels_fails() {
        // 4-channel tensor should be rejected
        let data = vec![0u8; 4 * 4 * 4];
        let tensor = Tensor::from_vec(data, (4, 4, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        let result = encode_image(&tensor, OutputFormat::Png, 4, 4);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("expected 3 channels"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_encode_single_pixel() {
        let tensor = solid_red_tensor(1, 1);
        let bytes = encode_image(&tensor, OutputFormat::Png, 1, 1).unwrap();
        // Must be a valid PNG
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_encode_both_formats_succeed() {
        let tensor = solid_red_tensor(4, 4);
        let png = encode_image(&tensor, OutputFormat::Png, 4, 4).unwrap();
        let jpeg = encode_image(&tensor, OutputFormat::Jpeg, 4, 4).unwrap();
        assert!(!png.is_empty(), "PNG output should not be empty");
        assert!(!jpeg.is_empty(), "JPEG output should not be empty");
        // Formats should differ in content
        assert_ne!(png, jpeg, "PNG and JPEG outputs should differ");
    }
}
