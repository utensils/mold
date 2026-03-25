use anyhow::{bail, Result};
use candle_core::Tensor;
use mold_core::{GenerateRequest, OutputFormat, OutputMetadata, Scheduler};

const MOLD_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn build_output_metadata(
    req: &GenerateRequest,
    seed: u64,
    scheduler: Option<Scheduler>,
) -> Option<OutputMetadata> {
    if !req.embed_metadata.unwrap_or(true) {
        return None;
    }

    Some(OutputMetadata::from_generate_request(
        req,
        seed,
        scheduler,
        MOLD_VERSION,
    ))
}

pub(crate) fn update_output_metadata_size(
    metadata: &mut Option<OutputMetadata>,
    width: u32,
    height: u32,
) {
    if let Some(metadata) = metadata {
        metadata.width = width;
        metadata.height = height;
    }
}

/// Encode a candle tensor [3, H, W] of u8 values into PNG or JPEG bytes.
pub(crate) fn encode_image(
    img: &Tensor,
    format: OutputFormat,
    width: u32,
    height: u32,
    metadata: Option<&OutputMetadata>,
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
        OutputFormat::Png => write_png(&rgb_image, &mut buf, metadata)?,
        OutputFormat::Jpeg => rgb_image.write_to(&mut buf, image::ImageFormat::Jpeg)?,
    }

    Ok(buf.into_inner())
}

fn write_png(
    rgb_image: &image::RgbImage,
    writer: &mut std::io::Cursor<Vec<u8>>,
    metadata: Option<&OutputMetadata>,
) -> Result<()> {
    let mut encoder = png::Encoder::new(writer, rgb_image.width(), rgb_image.height());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    if let Some(metadata) = metadata {
        encoder.add_itxt_chunk("mold:prompt".to_string(), metadata.prompt.clone())?;
        encoder.add_itxt_chunk("mold:model".to_string(), metadata.model.clone())?;
        encoder.add_text_chunk("mold:seed".to_string(), metadata.seed.to_string())?;
        encoder.add_text_chunk("mold:steps".to_string(), metadata.steps.to_string())?;
        encoder.add_text_chunk("mold:guidance".to_string(), metadata.guidance.to_string())?;
        encoder.add_text_chunk("mold:width".to_string(), metadata.width.to_string())?;
        encoder.add_text_chunk("mold:height".to_string(), metadata.height.to_string())?;
        if let Some(strength) = metadata.strength {
            encoder.add_text_chunk("mold:strength".to_string(), strength.to_string())?;
        }
        if let Some(scheduler) = metadata.scheduler {
            encoder.add_text_chunk("mold:scheduler".to_string(), scheduler.to_string())?;
        }
        encoder.add_itxt_chunk("mold:version".to_string(), metadata.version.clone())?;
        encoder.add_itxt_chunk(
            "mold:parameters".to_string(),
            serde_json::to_string(metadata)?,
        )?;
    }

    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(rgb_image.as_raw())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::io::Cursor;

    /// Build a 3xHxW solid-red tensor (R=255, G=0, B=0).
    fn solid_red_tensor(h: usize, w: usize) -> Tensor {
        let mut data = vec![0u8; 3 * h * w];
        // Channel 0 (R) = 255, channels 1 and 2 stay 0
        for value in data.iter_mut().take(h * w) {
            *value = 255;
        }
        Tensor::from_vec(data, (3, h, w), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
    }

    #[test]
    fn test_encode_png_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4, None).unwrap();
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_encode_jpeg_valid_tensor() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, None).unwrap();
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
        let result = encode_image(&tensor, OutputFormat::Png, 4, 4, None);
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
        let bytes = encode_image(&tensor, OutputFormat::Png, 1, 1, None).unwrap();
        // Must be a valid PNG
        assert!(bytes.len() >= 4);
        assert_eq!(&bytes[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn test_encode_both_formats_succeed() {
        let tensor = solid_red_tensor(4, 4);
        let png = encode_image(&tensor, OutputFormat::Png, 4, 4, None).unwrap();
        let jpeg = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, None).unwrap();
        assert!(!png.is_empty(), "PNG output should not be empty");
        assert!(!jpeg.is_empty(), "JPEG output should not be empty");
        // Formats should differ in content
        assert_ne!(png, jpeg, "PNG and JPEG outputs should differ");
    }

    fn decode_png_info(bytes: &[u8]) -> png::Info<'static> {
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let out_size = reader.output_buffer_size().unwrap();
        let mut buf = vec![0; out_size];
        reader.next_frame(&mut buf).unwrap();
        reader.info().clone()
    }

    #[test]
    fn test_encode_png_with_metadata_chunks() {
        let tensor = solid_red_tensor(4, 4);
        let metadata = OutputMetadata {
            prompt: "hello \u{2603}".to_string(),
            model: "flux-schnell:q8".to_string(),
            seed: 42,
            steps: 4,
            guidance: 0.0,
            width: 4,
            height: 4,
            strength: None,
            scheduler: None,
            version: "0.1.0".to_string(),
        };

        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4, Some(&metadata)).unwrap();
        let info = decode_png_info(&bytes);

        assert!(info
            .utf8_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:prompt"
                && chunk.get_text().unwrap() == "hello \u{2603}"));
        assert!(info
            .utf8_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:model"
                && chunk.get_text().unwrap() == "flux-schnell:q8"));
        assert!(info
            .utf8_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:parameters"
                && chunk
                    .get_text()
                    .unwrap()
                    .contains("\"model\":\"flux-schnell:q8\"")));
        assert!(info
            .uncompressed_latin1_text
            .iter()
            .any(|chunk| chunk.keyword == "mold:seed" && chunk.text == "42"));
    }

    #[test]
    fn test_encode_png_without_metadata_chunks() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Png, 4, 4, None).unwrap();
        let info = decode_png_info(&bytes);
        assert!(info.uncompressed_latin1_text.is_empty());
        assert!(info.utf8_text.is_empty());
    }

    #[test]
    fn test_build_output_metadata_respects_opt_out() {
        let req = GenerateRequest {
            prompt: "a cat".to_string(),
            model: "flux-schnell:q8".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: Some(false),
            scheduler: None,
            source_image: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
        };

        assert!(build_output_metadata(&req, 42, None).is_none());
    }

    #[test]
    fn test_update_output_metadata_size_overrides_dimensions() {
        let mut metadata = Some(OutputMetadata {
            prompt: "a cat".to_string(),
            model: "wuerstchen-v2:fp16".to_string(),
            seed: 42,
            steps: 30,
            guidance: 0.0,
            width: 1024,
            height: 1024,
            strength: None,
            scheduler: None,
            version: "0.1.0".to_string(),
        });

        update_output_metadata_size(&mut metadata, 1008, 1008);

        let metadata = metadata.unwrap();
        assert_eq!(metadata.width, 1008);
        assert_eq!(metadata.height, 1008);
    }
}
