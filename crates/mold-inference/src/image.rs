use anyhow::{Result, bail};
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
