//! CPU-only preparation of fresh image inputs for transport and role ingestion.
//! Processing budgets belong to engines. Already admitted inputs keep exact bytes.
use image::{ImageDecoder, ImageEncoder};
use std::io::Cursor;

/// Bounded original download/read envelope, independent of encoded request budgets.
pub const MAX_INGEST_BYTES: usize = 64 * 1024 * 1024;

/// A transport/role envelope, never a model processing resolution.
#[derive(Clone, Copy)]
pub struct InputImageLimits {
    pub max_bytes: usize,
    pub max_axis: u32,
    pub max_pixels: u64,
}

/// Bound fresh authoring bytes, preserving aspect, alpha and the colour profile.
/// Do not apply this to immutable uploaded/retained authorities or paired masks.
pub fn prepare_image(bytes: &[u8], limits: InputImageLimits) -> Result<Vec<u8>, String> {
    if limits.max_bytes == 0 || limits.max_axis == 0 || limits.max_pixels == 0 {
        return Err("image input envelope must be positive".into());
    }
    if bytes.len() > MAX_INGEST_BYTES {
        return Err("image exceeds the 64 MiB ingestion limit".into());
    }
    if crate::validation::sniff_image_input_format(bytes).is_none() {
        return Err("image must be a PNG, JPEG, or WebP image".into());
    }
    let (width, height) = crate::reference_image::oriented_dimensions(bytes)?;
    crate::reference_image::validate_reference_image_dimensions("Image input", width, height)?;
    let pixels = u64::from(width) * u64::from(height);
    if bytes.len() <= limits.max_bytes
        && width.max(height) <= limits.max_axis
        && pixels <= limits.max_pixels
    {
        return Ok(bytes.to_vec());
    }
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| format!("unreadable image: {error}"))?;
    reader.limits(crate::reference_image::reference_decode_limits());
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| format!("unreadable image: {error}"))?;
    let orientation = decoder
        .orientation()
        .map_err(|error| format!("unreadable image orientation: {error}"))?;
    let icc = decoder
        .icc_profile()
        .map_err(|error| format!("unreadable image colour profile: {error}"))?;
    let mut decoded = image::DynamicImage::from_decoder(decoder)
        .map_err(|error| format!("unreadable image: {error}"))?;
    decoded.apply_orientation(orientation);
    let scale = (f64::from(limits.max_axis) / f64::from(width.max(height)))
        .min((limits.max_pixels as f64 / pixels as f64).sqrt())
        .min(1.0);
    let mut target_width = (f64::from(width) * scale).floor().max(1.0) as u32;
    let mut target_height = (f64::from(height) * scale).floor().max(1.0) as u32;
    loop {
        let resized = resize_with_alpha(&decoded, target_width, target_height);
        let mut prepared = Vec::new();
        let mut encoder = image::codecs::png::PngEncoder::new(&mut prepared);
        if let Some(profile) = &icc {
            encoder
                .set_icc_profile(profile.clone())
                .map_err(|error| format!("cannot preserve image colour profile: {error}"))?;
        }
        // Orientation has been applied to pixels; do not copy an EXIF rotation tag.
        encoder
            .write_image(
                resized.as_bytes(),
                target_width,
                target_height,
                resized.color().into(),
            )
            .map_err(|error| format!("cannot encode image input: {error}"))?;
        if prepared.len() <= limits.max_bytes {
            return Ok(prepared);
        }
        if target_width == 1 && target_height == 1 {
            return Err("image cannot fit the input byte budget".into());
        }
        target_width = ((target_width as f64 * 0.75).floor() as u32).max(1);
        target_height = ((target_height as f64 * 0.75).floor() as u32).max(1);
    }
}

// Resample alpha in premultiplied space to keep transparent edge colours out
// of visible pixels. No colour-space conversion occurs; ICC still describes RGB.
fn resize_with_alpha(image: &image::DynamicImage, width: u32, height: u32) -> image::DynamicImage {
    if !image.color().has_alpha() {
        return image.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    }
    let mut rgba = image.to_rgba16();
    for pixel in rgba.pixels_mut() {
        for channel in 0..3 {
            pixel.0[channel] =
                (u64::from(pixel.0[channel]) * u64::from(pixel.0[3]) / 65_535) as u16;
        }
    }
    let mut resized =
        image::imageops::resize(&rgba, width, height, image::imageops::FilterType::Lanczos3);
    for pixel in resized.pixels_mut() {
        let alpha = pixel.0[3];
        for channel in 0..3 {
            pixel.0[channel] = if alpha > 0 {
                (u64::from(pixel.0[channel]) * 65_535 / u64::from(alpha)).min(65_535) as u16
            } else {
                0
            };
        }
        pixel.0[3] = alpha;
    }
    let result = image::DynamicImage::ImageRgba16(resized);
    match image.color() {
        image::ColorType::La8 => image::DynamicImage::ImageLumaA8(result.to_luma_alpha8()),
        image::ColorType::La16 => image::DynamicImage::ImageLumaA16(result.to_luma_alpha16()),
        image::ColorType::Rgba16 => image::DynamicImage::ImageRgba16(result.to_rgba16()),
        _ => image::DynamicImage::ImageRgba8(result.to_rgba8()),
    }
}

/// Identity's face detector has a narrower ingestion envelope than general images.
pub fn prepare_identity_image(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !matches!(
        crate::validation::sniff_image_input_format(bytes),
        Some(crate::ImageInputFormat::Png | crate::ImageInputFormat::Jpeg)
    ) {
        return Err("id_image must be a PNG or JPEG image".into());
    }
    let limits = crate::identity::ID_IMAGE_LIMITS;
    prepare_image(
        bytes,
        InputImageLimits {
            max_bytes: limits.max_encoded_bytes,
            max_axis: limits.max_axis_pixels,
            max_pixels: limits.max_decoded_pixels,
        },
    )
}

/// Bound a multi-photo identity set as well as each photograph.
pub fn prepare_identity_group(images: &mut [Vec<u8>]) -> Result<(), String> {
    if images.is_empty() || images.len() > crate::identity::ID_IMAGES_MAX {
        return Ok(());
    }
    let total_bytes = images
        .iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes.len()))
        .ok_or("identity byte count overflowed")?;
    let total_pixels = images.iter().try_fold(0u64, |total, bytes| {
        let (width, height) = crate::reference_image::oriented_dimensions(bytes)?;
        crate::reference_image::validate_reference_image_dimensions(
            "Identity photo",
            width,
            height,
        )?;
        total
            .checked_add(u64::from(width) * u64::from(height))
            .ok_or_else(|| "identity pixel count overflowed".to_string())
    })?;
    let role = crate::identity::ID_IMAGE_LIMITS;
    let max_bytes = if total_bytes > crate::identity::ID_IMAGES_TOTAL_ENCODED_BYTES_MAX {
        role.max_encoded_bytes
            .min(crate::identity::ID_IMAGES_TOTAL_ENCODED_BYTES_MAX / images.len())
    } else {
        role.max_encoded_bytes
    };
    let max_pixels = if total_pixels > crate::identity::ID_IMAGES_TOTAL_DECODED_PIXELS_MAX {
        role.max_decoded_pixels
            .min(crate::identity::ID_IMAGES_TOTAL_DECODED_PIXELS_MAX / images.len() as u64)
    } else {
        role.max_decoded_pixels
    };
    let prepared = images
        .iter()
        .map(|bytes| {
            if !matches!(
                crate::validation::sniff_image_input_format(bytes),
                Some(crate::ImageInputFormat::Png | crate::ImageInputFormat::Jpeg)
            ) {
                return Err("id_image must be a PNG or JPEG image".into());
            }
            prepare_image(
                bytes,
                InputImageLimits {
                    max_bytes,
                    max_axis: role.max_axis_pixels,
                    max_pixels,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (image, prepared) in images.iter_mut().zip(prepared) {
        *image = prepared;
    }
    Ok(())
}

/// Fit an oversized ordered-reference transport group without imposing model caps.
/// Small groups remain byte-identical, including originals above processing size.
pub fn prepare_reference_images(images: &mut [Vec<u8>]) -> Result<(), String> {
    let total = images
        .iter()
        .try_fold(0usize, |total, image| total.checked_add(image.len()))
        .ok_or("reference byte count overflowed")?;
    if total <= 32 * 1024 * 1024 {
        return Ok(());
    }
    let prepared = images
        .iter()
        .map(|bytes| {
            prepare_image(
                bytes,
                InputImageLimits {
                    max_bytes: 2 * 1024 * 1024,
                    max_axis: crate::reference_image::REFERENCE_IMAGE_MAX_SIDE,
                    max_pixels: crate::reference_image::REFERENCE_IMAGE_MAX_PIXELS,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (image, prepared) in images.iter_mut().zip(prepared) {
        *image = prepared;
    }
    Ok(())
}

/// Normalize only freshly supplied identity bytes, before admission seals/digests.
/// Retained media has no inline bytes here and remains immutable.
pub fn prepare_identity_inputs(request: &mut crate::GenerateRequest) -> Result<(), String> {
    if request.id_image.is_some() && request.id_images.is_some() {
        return Ok(());
    }
    if request
        .id_images
        .as_ref()
        .is_some_and(|images| images.len() > crate::identity::ID_IMAGES_MAX)
    {
        return Ok(());
    }
    let singular = request
        .id_image
        .as_ref()
        .map(|bytes| prepare_identity_image(bytes))
        .transpose()?;
    let plural = request
        .id_images
        .as_ref()
        .map(|images| {
            let mut prepared = images.clone();
            prepare_identity_group(&mut prepared)?;
            Ok::<_, String>(prepared)
        })
        .transpose()?;
    if let (Some(original), Some(prepared)) = (&request.id_image, &singular) {
        if original != prepared {
            if let Some(name) = &mut request.id_image_name {
                *name = png_name(name);
            }
        }
    }
    if let (Some(originals), Some(prepared), Some(names)) =
        (&request.id_images, &plural, &mut request.id_image_names)
    {
        for ((original, prepared), name) in originals.iter().zip(prepared).zip(names) {
            if original != prepared {
                *name = png_name(name);
            }
        }
    }
    request.id_image = singular;
    request.id_images = plural;
    Ok(())
}

/// Name transformed authoring bytes by their actual PNG container.
pub fn png_name(name: &str) -> String {
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    format!("{stem}.png")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn png(width: u32, height: u32) -> Vec<u8> {
        let pixels = image::RgbaImage::from_pixel(width, height, image::Rgba([42, 60, 80, 90]));
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(
                pixels.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgba8,
            )
            .unwrap();
        bytes
    }
    #[test]
    fn small_originals_are_byte_identical() {
        let bytes = png(1280, 853);
        assert_eq!(prepare_identity_image(&bytes).unwrap(), bytes);
    }
    #[test]
    fn ingestion_resize_is_down_only_proportional_and_keeps_alpha() {
        let bytes = png(120, 80);
        let prepared = prepare_image(
            &bytes,
            InputImageLimits {
                max_bytes: 1_000_000,
                max_axis: 60,
                max_pixels: 2400,
            },
        )
        .unwrap();
        let decoded = image::load_from_memory(&prepared).unwrap().into_rgba8();
        assert_eq!(decoded.dimensions(), (60, 40));
        assert_eq!(decoded.get_pixel(20, 20).0[3], 90);
    }
    #[test]
    fn unsafe_or_unreadable_inputs_are_refused() {
        assert!(prepare_identity_image(b"not an image").is_err());
    }
    #[test]
    fn oriented_resize_does_not_reapply_exif() {
        let bytes =
            include_bytes!("../testdata/reference_orientation/landscape_96x48_orientation6.jpg");
        let prepared = prepare_image(
            bytes,
            InputImageLimits {
                max_bytes: 1_000_000,
                max_axis: 48,
                max_pixels: 10_000,
            },
        )
        .unwrap();
        assert_eq!(
            crate::reference_image::oriented_dimensions(&prepared),
            Ok((24, 48))
        );
    }
    #[test]
    fn transparent_edge_colours_do_not_bleed() {
        let pixels = image::RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgba([255, 0, 0, 255])
            } else {
                image::Rgba([0, 0, 255, 0])
            }
        });
        let resized =
            resize_with_alpha(&image::DynamicImage::ImageRgba8(pixels), 1, 1).into_rgba8();
        assert!(resized.get_pixel(0, 0).0[0] >= 250);
        assert!(resized.get_pixel(0, 0).0[2] <= 2);
    }
    #[test]
    #[ignore = "large CPU fixture; run explicitly for image ingestion changes"]
    fn a_48mp_phone_photo_fits_identity_ingestion() {
        let pixels = image::GrayImage::from_pixel(8000, 6000, image::Luma([80]));
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(pixels.as_raw(), 8000, 6000, image::ExtendedColorType::L8)
            .unwrap();
        assert!(crate::identity::validate_id_image_bytes(&bytes).is_err());
        let prepared = prepare_identity_image(&bytes).unwrap();
        crate::identity::validate_id_image_bytes(&prepared).unwrap();
        let (width, height) = crate::reference_image::oriented_dimensions(&prepared).unwrap();
        assert!(u64::from(width) * u64::from(height) <= 32_000_000);
        assert!((f64::from(width) / f64::from(height) - 4.0 / 3.0).abs() < 0.001);
    }
    #[test]
    fn resizing_retains_profile_and_does_not_convert_pixel_colours() {
        let pixels = image::RgbImage::from_pixel(120, 80, image::Rgb([42, 60, 80]));
        let profile = b"test-profile-preserved-verbatim".to_vec();
        let mut bytes = Vec::new();
        let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
        encoder.set_icc_profile(profile.clone()).unwrap();
        encoder
            .write_image(pixels.as_raw(), 120, 80, image::ExtendedColorType::Rgb8)
            .unwrap();
        let prepared = prepare_image(
            &bytes,
            InputImageLimits {
                max_bytes: 1_000_000,
                max_axis: 60,
                max_pixels: 2400,
            },
        )
        .unwrap();
        let mut decoder = image::ImageReader::new(Cursor::new(&prepared))
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        assert_eq!(decoder.icc_profile().unwrap(), Some(profile));
        let decoded = image::DynamicImage::from_decoder(decoder)
            .unwrap()
            .into_rgb8();
        assert_eq!(decoded.get_pixel(20, 20).0, [42, 60, 80]);
    }
    #[test]
    fn fresh_identity_preparation_does_not_touch_other_media_authorities() {
        let mut request: crate::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test", "model": "flux-dev", "width": 256, "height": 256,
            "steps": 4, "guidance": 1.0, "batch_size": 1
        }))
        .unwrap();
        request.source_image = Some(vec![1, 2, 3]);
        request.edit_images = Some(vec![png(1280, 853)]);
        request.id_image = Some(png(8193, 100));
        request.references = Some(vec![crate::GenerationReference::Image {
            media: crate::GenerationReferenceAuthority::Upload {
                handle: "immutable-handle".into(),
            },
            provenance: crate::GenerationReferenceProvenance {
                name: Some("original.jpg".into()),
                sha256: Some("a".repeat(64)),
                crop: None,
            },
            mime_type: "image/jpeg".into(),
            width: 8000,
            height: 6000,
        }]);
        let original = request.clone();
        prepare_identity_inputs(&mut request).unwrap();
        assert_ne!(request.id_image, original.id_image);
        crate::identity::validate_id_image_bytes(request.id_image.as_ref().unwrap()).unwrap();
        assert_eq!(request.source_image, original.source_image);
        assert_eq!(request.edit_images, original.edit_images);
        assert_eq!(
            serde_json::to_value(&request.references).unwrap(),
            serde_json::to_value(&original.references).unwrap()
        );
    }
    #[test]
    fn fresh_plural_identity_preparation_fits_the_whole_set_before_validation() {
        let mut request: crate::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test", "model": "flux-dev", "width": 256, "height": 256,
            "steps": 4, "guidance": 1.0, "batch_size": 1
        }))
        .unwrap();
        let mut original = png(120, 80);
        original.resize(17 * 1024 * 1024, 0);
        request.id_images = Some(vec![original.clone(), original]);
        let photos = request.id_images.as_ref().unwrap();
        assert!(crate::identity::validate_id_images(
            &photos.iter().map(Vec::as_slice).collect::<Vec<_>>()
        )
        .is_err());
        prepare_identity_inputs(&mut request).unwrap();
        let photos = request.id_images.as_ref().unwrap();
        crate::identity::validate_id_images(&photos.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .unwrap();
        assert!(
            photos.iter().map(Vec::len).sum::<usize>()
                <= crate::identity::ID_IMAGES_TOTAL_ENCODED_BYTES_MAX
        );
    }

    #[test]
    fn admission_accepts_originals_above_each_reference_processing_budget() {
        for (family, model) in [
            ("qwen-image21", "qwen-image-2.1:bf16"),
            ("flux2", "flux2-klein:bf16"),
            ("qwen-image-edit", "qwen-image-edit:bf16"),
        ] {
            let mut request: crate::GenerateRequest = serde_json::from_value(serde_json::json!({
                "prompt": "test", "model": model, "width": 1024, "height": 1024,
                "steps": 4, "guidance": 1.0, "batch_size": 1
            }))
            .unwrap();
            request.edit_images = Some(vec![png(1280, 853), png(1600, 900)]);
            let profile = crate::generation_profile::reference_images_for_recipe(family, model);
            crate::generation_profile::validate_edit_images_against(&profile, model, &request)
                .unwrap();
        }
    }

    #[test]
    fn reference_transport_is_checked_after_normalization() {
        let mut large = png(120, 80);
        large.resize(17 * 1024 * 1024, 0);
        let mut images = vec![large.clone(), large];
        prepare_reference_images(&mut images).unwrap();
        assert!(images.iter().all(|bytes| bytes.len() < 2 * 1024 * 1024));
        assert_eq!(
            crate::reference_image::oriented_dimensions(&images[0]),
            Ok((120, 80))
        );
    }
}
