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
        OutputFormat::Jpeg => write_jpeg(&rgb_image, &mut buf, metadata)?,
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
        if let Some(ref neg) = metadata.negative_prompt {
            encoder.add_itxt_chunk("mold:negative_prompt".to_string(), neg.clone())?;
        }
        if let Some(ref original) = metadata.original_prompt {
            encoder.add_itxt_chunk("mold:original_prompt".to_string(), original.clone())?;
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

fn write_jpeg(
    rgb_image: &image::RgbImage,
    writer: &mut std::io::Cursor<Vec<u8>>,
    metadata: Option<&OutputMetadata>,
) -> Result<()> {
    rgb_image.write_to(writer, image::ImageFormat::Jpeg)?;

    let Some(metadata) = metadata else {
        return Ok(());
    };

    let jpeg_bytes = writer.get_ref().clone();
    // JPEG must start with SOI (0xFFD8)
    if jpeg_bytes.len() < 2 || jpeg_bytes[0] != 0xFF || jpeg_bytes[1] != 0xD8 {
        return Ok(());
    }

    let mut out = Vec::with_capacity(jpeg_bytes.len() + 4096);
    out.extend_from_slice(&jpeg_bytes[..2]); // SOI marker

    // Inject COM marker with JSON parameters (read by exiftool, identify, ffprobe)
    let json = serde_json::to_string(metadata)?;
    let comment = format!("mold:parameters {json}");
    write_jpeg_com_marker(&mut out, comment.as_bytes());

    // Inject XMP APP1 marker (read by Photoshop, Lightroom, GIMP, exiftool -xmp:all)
    let xmp = build_xmp_packet(metadata);
    write_jpeg_xmp_marker(&mut out, &xmp);

    // Append rest of JPEG data (everything after SOI)
    out.extend_from_slice(&jpeg_bytes[2..]);

    writer.get_mut().clear();
    writer.get_mut().extend_from_slice(&out);
    writer.set_position(out.len() as u64);
    Ok(())
}

/// Write a JPEG COM (comment) marker segment.
/// Truncates payload to 65533 bytes (JPEG segment limit: 65535 - 2 for length field).
fn write_jpeg_com_marker(out: &mut Vec<u8>, data: &[u8]) {
    const MAX_PAYLOAD: usize = 65533;
    let data = if data.len() > MAX_PAYLOAD {
        tracing::warn!(
            "JPEG COM marker truncated from {} to {MAX_PAYLOAD} bytes",
            data.len()
        );
        &data[..MAX_PAYLOAD]
    } else {
        data
    };
    let len = (data.len() + 2) as u16;
    out.push(0xFF);
    out.push(0xFE); // COM marker
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(data);
}

/// Build an XMP packet containing generation metadata as RDF/XML.
fn build_xmp_packet(metadata: &OutputMetadata) -> Vec<u8> {
    use std::fmt::Write;
    let mut xmp = String::with_capacity(1024);
    xmp.push_str(r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>"#);
    xmp.push_str(r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">"#);
    xmp.push_str(r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#);
    xmp.push_str(r#"<rdf:Description rdf:about="" xmlns:mold="https://github.com/utensils/mold">"#);
    let _ = write!(
        xmp,
        "<mold:prompt>{}</mold:prompt>",
        xml_escape(&metadata.prompt)
    );
    let _ = write!(
        xmp,
        "<mold:model>{}</mold:model>",
        xml_escape(&metadata.model)
    );
    let _ = write!(xmp, "<mold:seed>{}</mold:seed>", metadata.seed);
    let _ = write!(xmp, "<mold:steps>{}</mold:steps>", metadata.steps);
    let _ = write!(xmp, "<mold:guidance>{}</mold:guidance>", metadata.guidance);
    let _ = write!(xmp, "<mold:width>{}</mold:width>", metadata.width);
    let _ = write!(xmp, "<mold:height>{}</mold:height>", metadata.height);
    if let Some(strength) = metadata.strength {
        let _ = write!(xmp, "<mold:strength>{strength}</mold:strength>");
    }
    if let Some(scheduler) = metadata.scheduler {
        let _ = write!(xmp, "<mold:scheduler>{scheduler}</mold:scheduler>");
    }
    if let Some(ref neg) = metadata.negative_prompt {
        let _ = write!(
            xmp,
            "<mold:negativePrompt>{}</mold:negativePrompt>",
            xml_escape(neg)
        );
    }
    if let Some(ref original) = metadata.original_prompt {
        let _ = write!(
            xmp,
            "<mold:originalPrompt>{}</mold:originalPrompt>",
            xml_escape(original)
        );
    }
    let _ = write!(
        xmp,
        "<mold:version>{}</mold:version>",
        xml_escape(&metadata.version)
    );
    let json = serde_json::to_string(metadata).expect("metadata already serialized in COM marker");
    let _ = write!(
        xmp,
        "<mold:parameters>{}</mold:parameters>",
        xml_escape(&json)
    );
    xmp.push_str("</rdf:Description></rdf:RDF></x:xmpmeta>");
    xmp.push_str(r#"<?xpacket end="w"?>"#);
    xmp.into_bytes()
}

/// Write a JPEG APP1 marker with the standard XMP namespace prefix.
/// Skips the marker entirely if the payload exceeds the 65535-byte segment limit.
///
/// Per JPEG spec, the segment length field (u16) counts itself (2 bytes) plus the
/// payload (namespace + XMP data). The 0xFF 0xE1 marker bytes are NOT included in
/// the length field. So `total` = 2 + namespace + xmp, and max is 0xFFFF.
fn write_jpeg_xmp_marker(out: &mut Vec<u8>, xmp_data: &[u8]) {
    let namespace = b"http://ns.adobe.com/xap/1.0/\0";
    let total = namespace.len() + xmp_data.len() + 2; // +2 for the length field itself
    if total > 0xFFFF {
        tracing::warn!("XMP packet too large for JPEG APP1 marker ({total} bytes), skipping");
        return;
    }
    let total_len = total as u16;
    out.push(0xFF);
    out.push(0xE1); // APP1 marker
    out.extend_from_slice(&total_len.to_be_bytes());
    out.extend_from_slice(namespace);
    out.extend_from_slice(xmp_data);
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
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
            negative_prompt: None,
            original_prompt: None,
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
            negative_prompt: None,
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
            expand: None,
            original_prompt: None,
        };

        assert!(build_output_metadata(&req, 42, None).is_none());
    }

    #[test]
    fn test_update_output_metadata_size_overrides_dimensions() {
        let mut metadata = Some(OutputMetadata {
            prompt: "a cat".to_string(),
            negative_prompt: None,
            original_prompt: None,
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

    // ── JPEG metadata tests ───────────────────────────────────────────────

    fn test_metadata() -> OutputMetadata {
        OutputMetadata {
            prompt: "hello world".to_string(),
            negative_prompt: None,
            original_prompt: None,
            model: "flux-schnell:q8".to_string(),
            seed: 42,
            steps: 4,
            guidance: 0.0,
            width: 4,
            height: 4,
            strength: None,
            scheduler: None,
            version: "0.1.0".to_string(),
        }
    }

    /// Find the first JPEG COM (0xFFFE) marker and return its payload.
    fn find_jpeg_comment(bytes: &[u8]) -> Option<Vec<u8>> {
        let mut i = 2; // skip SOI
        while i + 3 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if marker == 0xFE {
                // COM marker
                return Some(bytes[i + 4..i + 2 + len].to_vec());
            }
            i += 2 + len;
        }
        None
    }

    /// Find the first JPEG APP1 XMP marker and return the XMP payload (after namespace).
    fn find_jpeg_xmp(bytes: &[u8]) -> Option<String> {
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let mut i = 2; // skip SOI
        while i + 3 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if marker == 0xE1 {
                let payload = &bytes[i + 4..i + 2 + len];
                if payload.starts_with(namespace) {
                    let xmp_bytes = &payload[namespace.len()..];
                    return String::from_utf8(xmp_bytes.to_vec()).ok();
                }
            }
            i += 2 + len;
        }
        None
    }

    #[test]
    fn test_encode_jpeg_with_metadata_has_comment() {
        let tensor = solid_red_tensor(4, 4);
        let metadata = test_metadata();
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, Some(&metadata)).unwrap();

        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "should be valid JPEG");
        let comment = find_jpeg_comment(&bytes).expect("should have COM marker");
        let comment_str = String::from_utf8(comment).unwrap();
        assert!(
            comment_str.starts_with("mold:parameters "),
            "comment should start with mold:parameters: {comment_str}"
        );
        let json_str = &comment_str["mold:parameters ".len()..];
        let parsed: OutputMetadata = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.prompt, "hello world");
        assert_eq!(parsed.model, "flux-schnell:q8");
        assert_eq!(parsed.seed, 42);
    }

    #[test]
    fn test_encode_jpeg_with_metadata_has_xmp() {
        let tensor = solid_red_tensor(4, 4);
        let metadata = test_metadata();
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, Some(&metadata)).unwrap();

        let xmp = find_jpeg_xmp(&bytes).expect("should have XMP APP1 marker");
        assert!(xmp.contains("mold:prompt"), "XMP should contain prompt");
        assert!(
            xmp.contains("hello world"),
            "XMP should contain prompt text"
        );
        assert!(xmp.contains("mold:seed"), "XMP should contain seed element");
        assert!(xmp.contains("<mold:seed>42</mold:seed>"), "seed value");
        assert!(
            xmp.contains("xmlns:mold=\"https://github.com/utensils/mold\""),
            "XMP should have mold namespace"
        );
        assert!(
            xmp.contains("mold:parameters"),
            "XMP should contain parameters JSON"
        );
    }

    #[test]
    fn test_encode_jpeg_without_metadata_no_extra_markers() {
        let tensor = solid_red_tensor(4, 4);
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 4, 4, None).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
        assert!(
            find_jpeg_comment(&bytes).is_none(),
            "no COM marker without metadata"
        );
        assert!(
            find_jpeg_xmp(&bytes).is_none(),
            "no XMP marker without metadata"
        );
    }

    #[test]
    fn test_encode_jpeg_metadata_roundtrip() {
        let tensor = solid_red_tensor(8, 8);
        let metadata = OutputMetadata {
            prompt: "a cat & a dog <br>".to_string(),
            negative_prompt: None,
            original_prompt: None,
            model: "sdxl-turbo:fp16".to_string(),
            seed: 99999,
            steps: 25,
            guidance: 7.5,
            width: 8,
            height: 8,
            strength: Some(0.6),
            scheduler: Some(mold_core::Scheduler::EulerAncestral),
            version: "0.2.0".to_string(),
        };
        let bytes = encode_image(&tensor, OutputFormat::Jpeg, 8, 8, Some(&metadata)).unwrap();

        // Roundtrip via COM JSON
        let comment = find_jpeg_comment(&bytes).unwrap();
        let json_str = String::from_utf8(comment).unwrap();
        let json_str = &json_str["mold:parameters ".len()..];
        let parsed: OutputMetadata = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed, metadata);

        // XMP should have XML-escaped special characters
        let xmp = find_jpeg_xmp(&bytes).unwrap();
        assert!(
            xmp.contains("a cat &amp; a dog &lt;br&gt;"),
            "prompt should be XML-escaped in XMP: {xmp}"
        );
        assert!(
            xmp.contains("<mold:strength>0.6</mold:strength>"),
            "strength should be present"
        );
        assert!(
            xmp.contains("<mold:scheduler>euler-ancestral</mold:scheduler>"),
            "scheduler should be present"
        );
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("hello"), "hello");
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape(r#"say "hi""#), "say &quot;hi&quot;");
        assert_eq!(xml_escape("a < b & c > d"), "a &lt; b &amp; c &gt; d");
    }
}
