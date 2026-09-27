//! Header facts and bounds for ordered reference images
//! (`GenerateRequest.edit_images`).
//!
//! Every surface that sizes something from a reference — the CLI's, MCP's and
//! Discord's `canvas: last-reference` default, the server's dimension
//! advisory, admission's size bounds — reads the reference through
//! [`oriented_dimensions`], so they all see the picture the ENGINE sees.
//!
//! The engines apply the EXIF `Orientation` tag when they decode a reference
//! (`mold_inference::img_utils::decode_oriented_srgb_rgba`), so a portrait
//! phone photo stored as landscape pixels plus `Orientation = 6` conditions
//! the right way up. Upstream diffusers (`pipeline_qwenimage21.py:653-660`
//! at `e0abab83b`) opens it with PIL and reads `img.size` unrotated; mold
//! deliberately differs there, and a canvas authority that read the raw
//! header would size a landscape canvas for a portrait reference.
//!
//! Both sides use the `image` crate's own EXIF reader
//! (`ImageDecoder::orientation`): JPEG `APP1 Exif`, PNG `eXIf`, WebP `EXIF`.
//! Studio's `studio/lib/imageDimensions.ts` mirrors that reader byte for byte.

use std::io::Cursor;

/// Largest reference side the engines resample: the Pillow-compatible resize
/// (`mold_inference::pillow_resize`) refuses anything above it, so admitting a
/// larger side would only fail later, after the queue.
pub const REFERENCE_IMAGE_MAX_SIDE: u32 = 16_384;

/// Largest reference area, in pixels — the figure the durable reference door
/// already uses (`minimax_h3::MAX_REFERENCE_IMAGE_PIXELS`). A decoded RGBA
/// reference is four bytes a pixel, so this bounds one decode at ~400 MB.
pub const REFERENCE_IMAGE_MAX_PIXELS: u64 = 100_000_000;

/// Largest long-to-short side ratio. The Qwen2-VL-family processor refuses
/// anything above it (`transformers` `image_processing_qwen2_vl.py:74-77`,
/// `smart_resize`: "absolute aspect ratio must be smaller than 200"), and no
/// mold reference recipe conditions on a more extreme picture.
pub const REFERENCE_IMAGE_MAX_ASPECT: u32 = 200;

/// The one decode-limit policy for reference pixels, shared by every engine
/// decode of a reference and by alpha detection.
///
/// The per-side bound is enforced by the decoder itself
/// (`ImageDecoder::set_limits`). `image` 0.25 does NOT apply `max_alloc` to
/// `DynamicImage::from_decoder`, so a decoder also checks the area with
/// [`validate_reference_image_dimensions`] before reading pixels; the
/// allocation bound here is what the format decoders (PNG, WebP) apply to
/// their own buffers — eight bytes a pixel, a 16-bit RGBA decode of the
/// largest admitted area.
pub fn reference_decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(REFERENCE_IMAGE_MAX_SIDE);
    limits.max_image_height = Some(REFERENCE_IMAGE_MAX_SIDE);
    limits.max_alloc = Some(REFERENCE_IMAGE_MAX_PIXELS * 8);
    limits
}

/// Refuse a reference no engine can condition on, or whose decode alone would
/// be a denial of service (a few hundred bytes of PNG can declare 60000x60000,
/// ~14 GB of pixels). `subject` names the reference in the message, as in
/// "Reference 2". Orientation does not change any of the three bounds, so stored or
/// upright dimensions answer the same.
pub fn validate_reference_image_dimensions(
    subject: &str,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err(format!("{subject} has no pixels ({width}x{height})."));
    }
    if width > REFERENCE_IMAGE_MAX_SIDE || height > REFERENCE_IMAGE_MAX_SIDE {
        return Err(format!(
            "{subject} is {width}x{height}; each side must be at most \
             {REFERENCE_IMAGE_MAX_SIDE} pixels. Resize it and try again."
        ));
    }
    if u64::from(width) * u64::from(height) > REFERENCE_IMAGE_MAX_PIXELS {
        return Err(format!(
            "{subject} is {width}x{height}; it must be at most \
             {REFERENCE_IMAGE_MAX_PIXELS} pixels in total. Resize it and try again."
        ));
    }
    let (long, short) = (width.max(height), width.min(height));
    if u64::from(long) > u64::from(short) * u64::from(REFERENCE_IMAGE_MAX_ASPECT) {
        return Err(format!(
            "{subject} is {width}x{height}; its aspect ratio must be at most \
             {REFERENCE_IMAGE_MAX_ASPECT}:1. Crop it and try again."
        ));
    }
    Ok(())
}

/// The dimensions an encoded image has once its EXIF orientation is applied,
/// read from the container header without decoding pixels.
///
/// Orientations 5–8 (every transposing one) swap width and height; 1–4 keep
/// them. A missing or unreadable EXIF block is orientation 1, exactly as the
/// engine's decoder treats it.
pub fn oriented_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    use image::ImageDecoder;
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| format!("unreadable image: {error}"))?;
    // Header reads only need the metadata allocation bound; the side bound is
    // reported by `validate_reference_image_dimensions` with a real sentence
    // rather than the decoder's generic "limits exceeded".
    reader.limits(image::Limits::default());
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| format!("unreadable image header: {error}"))?;
    let (width, height) = decoder.dimensions();
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    Ok(oriented(width, height, orientation))
}

/// `(width, height)` after `orientation` is applied.
pub fn oriented(width: u32, height: u32, orientation: image::metadata::Orientation) -> (u32, u32) {
    use image::metadata::Orientation;
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (height, width),
        _ => (width, height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;

    /// A little-endian TIFF block whose IFD0 holds only `Orientation`.
    fn exif_orientation(value: u16) -> Vec<u8> {
        let mut exif = b"II*\0".to_vec();
        exif.extend_from_slice(&8u32.to_le_bytes());
        exif.extend_from_slice(&1u16.to_le_bytes());
        exif.extend_from_slice(&0x0112u16.to_le_bytes());
        exif.extend_from_slice(&3u16.to_le_bytes());
        exif.extend_from_slice(&1u32.to_le_bytes());
        exif.extend_from_slice(&value.to_le_bytes());
        exif.extend_from_slice(&[0, 0]);
        exif.extend_from_slice(&0u32.to_le_bytes());
        exif
    }

    fn landscape_rgb() -> image::RgbImage {
        image::RgbImage::from_pixel(40, 24, image::Rgb([90, 120, 150]))
    }

    fn jpeg_with_orientation(orientation: Option<u16>) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new(&mut bytes);
        if let Some(value) = orientation {
            encoder.set_exif_metadata(exif_orientation(value)).unwrap();
        }
        let image = landscape_rgb();
        encoder
            .write_image(image.as_raw(), 40, 24, image::ExtendedColorType::Rgb8)
            .unwrap();
        bytes
    }

    fn png_with_orientation(orientation: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
        encoder
            .set_exif_metadata(exif_orientation(orientation))
            .unwrap();
        let image = landscape_rgb();
        encoder
            .write_image(image.as_raw(), 40, 24, image::ExtendedColorType::Rgb8)
            .unwrap();
        bytes
    }

    fn webp_with_orientation(orientation: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut bytes);
        encoder
            .set_exif_metadata(exif_orientation(orientation))
            .unwrap();
        let image = landscape_rgb();
        encoder
            .write_image(image.as_raw(), 40, 24, image::ExtendedColorType::Rgb8)
            .unwrap();
        bytes
    }

    #[test]
    fn every_transposing_orientation_swaps_the_sides_in_every_container() {
        for value in 1..=8u16 {
            let expected = if value >= 5 { (24, 40) } else { (40, 24) };
            assert_eq!(
                oriented_dimensions(&jpeg_with_orientation(Some(value))),
                Ok(expected),
                "jpeg orientation {value}"
            );
            assert_eq!(
                oriented_dimensions(&png_with_orientation(value)),
                Ok(expected),
                "png orientation {value}"
            );
            assert_eq!(
                oriented_dimensions(&webp_with_orientation(value)),
                Ok(expected),
                "webp orientation {value}"
            );
        }
        assert_eq!(
            oriented_dimensions(&jpeg_with_orientation(None)),
            Ok((40, 24))
        );
    }

    /// Files written by Pillow (big-endian `MM` EXIF, as cameras and PIL
    /// write it) rather than by the `image` encoders the test above uses. The
    /// same files pin Studio's reader (`studio/lib/imageDimensions.test.ts`).
    #[test]
    fn pillow_written_fixtures_read_upright() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/reference_orientation");
        for value in 1..=8 {
            let bytes =
                std::fs::read(dir.join(format!("landscape_96x48_orientation{value}.jpg"))).unwrap();
            let expected = if value >= 5 { (48, 96) } else { (96, 48) };
            assert_eq!(oriented_dimensions(&bytes), Ok(expected), "jpeg {value}");
        }
        for container in ["png", "webp"] {
            let bytes =
                std::fs::read(dir.join(format!("landscape_96x48_orientation6.{container}")))
                    .unwrap();
            assert_eq!(oriented_dimensions(&bytes), Ok((48, 96)), "{container}");
        }
    }

    #[test]
    fn the_bounds_refuse_oversized_and_extreme_references_by_name() {
        assert!(validate_reference_image_dimensions("Reference 1", 16_384, 6_000).is_ok());
        assert!(validate_reference_image_dimensions("Reference 1", 2, 400).is_ok());
        let side = validate_reference_image_dimensions("Reference 2", 16_385, 64).unwrap_err();
        assert!(
            side.contains("Reference 2") && side.contains("16384"),
            "{side}"
        );
        let area = validate_reference_image_dimensions("Reference 1", 12_000, 9_000).unwrap_err();
        assert!(area.contains("100000000"), "{area}");
        let aspect = validate_reference_image_dimensions("Reference 1", 2, 401).unwrap_err();
        assert!(aspect.contains("200:1"), "{aspect}");
        assert!(validate_reference_image_dimensions("Reference 1", 0, 10).is_err());
    }

    #[test]
    fn the_decode_limits_bound_every_side_at_the_admitted_maximum() {
        let limits = reference_decode_limits();
        assert_eq!(limits.max_image_width, Some(REFERENCE_IMAGE_MAX_SIDE));
        assert_eq!(limits.max_image_height, Some(REFERENCE_IMAGE_MAX_SIDE));
        assert_eq!(limits.max_alloc, Some(REFERENCE_IMAGE_MAX_PIXELS * 8));
    }

    #[test]
    fn an_unreadable_header_is_an_error_not_a_size() {
        assert!(oriented_dimensions(b"not an image").is_err());
        assert!(oriented_dimensions(&[0x89, b'P', b'N', b'G']).is_err());
    }
}
