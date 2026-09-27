//! Header facts for ordered reference images
//! (`GenerateRequest.edit_images`).
//!
//! Every surface that sizes something from a reference — the CLI's, MCP's and
//! Discord's `canvas: last-reference` default, the server's dimension
//! advisory — reads the reference through
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
    fn an_unreadable_header_is_an_error_not_a_size() {
        assert!(oriented_dimensions(b"not an image").is_err());
        assert!(oriented_dimensions(&[0x89, b'P', b'N', b'G']).is_err());
    }
}
