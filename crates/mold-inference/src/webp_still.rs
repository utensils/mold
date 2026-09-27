//! Still (single-frame, non-animated) WebP encoding.
//!
//! Every still recipe advertises WebP whenever the binary links the `webp`
//! feature, but the only WebP encoder mold carried was `webp-animation`'s
//! `WebPAnimEncoder`, which always writes an `ANIM`/`ANMF` container: a
//! one-frame animation that viewers, and mold's own media classification,
//! read as a video. This module is libwebp's *simple* encoding API
//! (`WebPEncodeRGB` / `WebPEncodeRGBA`, `src/webp/encode.h`), which emits a
//! plain `VP8 ` bitstream — or `VP8X` + `ALPH` + `VP8 ` when the image carries
//! alpha. The colour planes are lossy at [`WEBP_STILL_QUALITY`]; libwebp's
//! simple API compresses the alpha plane losslessly (its default
//! `alpha_quality` is 100 and `alpha_compression` is 1), so a transparent
//! cut-out keeps its exact mask.
//!
//! `libwebp-sys2` 0.2 is the same crate `webp-animation` already links, so
//! no new native code enters the build.

use anyhow::Result;

/// libwebp quality factor for still output (0 = smallest, 100 = best).
pub(crate) const WEBP_STILL_QUALITY: f32 = 90.0;

/// Encode an RGB image as a lossy still WebP (`VP8 ` bitstream, no `ANIM`).
#[cfg(feature = "webp")]
pub(crate) fn encode_webp_still_rgb(image: &image::RgbImage, quality: f32) -> Result<Vec<u8>> {
    encode(
        image.as_raw(),
        image.width(),
        image.height(),
        3,
        quality,
        libwebp_sys::WebPEncodeRGB,
    )
}

/// Encode an RGBA image as a still WebP with lossy colour and lossless alpha.
#[cfg(feature = "webp")]
pub(crate) fn encode_webp_still_rgba(image: &image::RgbaImage, quality: f32) -> Result<Vec<u8>> {
    encode(
        image.as_raw(),
        image.width(),
        image.height(),
        4,
        quality,
        libwebp_sys::WebPEncodeRGBA,
    )
}

#[cfg(feature = "webp")]
type SimpleEncodeFn = unsafe extern "C" fn(
    *const u8,
    std::os::raw::c_int,
    std::os::raw::c_int,
    std::os::raw::c_int,
    std::os::raw::c_float,
    *mut *mut u8,
) -> usize;

#[cfg(feature = "webp")]
fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    quality: f32,
    encode_fn: SimpleEncodeFn,
) -> Result<Vec<u8>> {
    use std::os::raw::c_int;

    // WebP's bitstream stores each dimension in 14 bits.
    const WEBP_MAX_DIMENSION: u32 = 16383;
    if width == 0 || height == 0 || width > WEBP_MAX_DIMENSION || height > WEBP_MAX_DIMENSION {
        anyhow::bail!(
            "WebP stills must be between 1 and {WEBP_MAX_DIMENSION} pixels on each side, got {width}x{height}"
        );
    }
    let stride = width
        .checked_mul(channels)
        .ok_or_else(|| anyhow::anyhow!("WebP row stride overflows"))?;
    let expected = (stride as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow::anyhow!("WebP buffer size overflows"))?;
    if pixels.len() != expected {
        anyhow::bail!(
            "WebP still buffer holds {} bytes, expected {expected} for {width}x{height}x{channels}",
            pixels.len()
        );
    }

    let mut output: *mut u8 = std::ptr::null_mut();
    // SAFETY: `pixels` is a live, correctly-sized (checked above) buffer of
    // `height` rows of `stride` bytes; libwebp only reads it. On success it
    // writes a heap pointer it allocated into `output`, which we copy out of
    // and release with `WebPFree` exactly once.
    let size = unsafe {
        encode_fn(
            pixels.as_ptr(),
            width as c_int,
            height as c_int,
            stride as c_int,
            quality,
            &mut output,
        )
    };
    if output.is_null() {
        anyhow::bail!("libwebp failed to encode a {width}x{height} still");
    }
    // SAFETY: libwebp reports `size` valid bytes at `output` (0 on failure,
    // in which case the slice is empty and we still free the allocation).
    let bytes = unsafe { std::slice::from_raw_parts(output, size) }.to_vec();
    // SAFETY: `output` was allocated by libwebp and is freed once here.
    unsafe { libwebp_sys::WebPFree(output.cast()) };
    if bytes.is_empty() {
        anyhow::bail!("libwebp failed to encode a {width}x{height} still");
    }
    Ok(bytes)
}

/// Without the `webp` feature the generation profile never advertises WebP;
/// a request that reaches the encoder anyway gets a named reason rather than
/// a silent PNG.
#[cfg(not(feature = "webp"))]
pub(crate) fn encode_webp_still_rgb(_image: &image::RgbImage, _quality: f32) -> Result<Vec<u8>> {
    anyhow::bail!("WebP output requires the 'webp' feature")
}

#[cfg(not(feature = "webp"))]
pub(crate) fn encode_webp_still_rgba(_image: &image::RgbaImage, _quality: f32) -> Result<Vec<u8>> {
    anyhow::bail!("WebP output requires the 'webp' feature")
}

/// Whether a RIFF/WEBP buffer contains an `ANIM` chunk (an animation
/// container rather than a still).
#[cfg(all(test, feature = "webp"))]
pub(crate) fn has_anim_chunk(bytes: &[u8]) -> bool {
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let fourcc = &bytes[offset..offset + 4];
        if fourcc == b"ANIM" || fourcc == b"ANMF" {
            return true;
        }
        let len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8 + len + (len & 1);
    }
    false
}

#[cfg(all(test, feature = "webp"))]
mod tests {
    use super::*;

    fn gradient_rgb(width: u32, height: u32) -> image::RgbImage {
        image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 5) as u8, ((x + y) * 3) as u8])
        })
    }

    fn cutout_rgba(width: u32, height: u32) -> image::RgbaImage {
        image::RgbaImage::from_fn(width, height, |x, y| {
            let inside =
                x >= width / 4 && x < 3 * width / 4 && y >= height / 4 && y < 3 * height / 4;
            if inside {
                image::Rgba([200, 40, 30, 255])
            } else if x == 0 {
                image::Rgba([10, 20, 30, 128])
            } else {
                image::Rgba([0, 0, 0, 0])
            }
        })
    }

    fn first_chunk(bytes: &[u8]) -> &[u8] {
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WEBP");
        &bytes[12..16]
    }

    #[test]
    fn rgb_still_round_trips_and_is_not_animated() {
        let source = gradient_rgb(40, 24);
        let bytes = encode_webp_still_rgb(&source, WEBP_STILL_QUALITY).unwrap();
        assert_eq!(first_chunk(&bytes), b"VP8 ");
        assert!(!has_anim_chunk(&bytes));
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (40, 24));
        assert!(!decoded.color().has_alpha());
    }

    #[test]
    fn rgba_still_keeps_its_alpha_losslessly() {
        let source = cutout_rgba(32, 32);
        let bytes = encode_webp_still_rgba(&source, WEBP_STILL_QUALITY).unwrap();
        assert_eq!(first_chunk(&bytes), b"VP8X");
        assert!(!has_anim_chunk(&bytes));
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert!(decoded.color().has_alpha());
        let decoded = decoded.to_rgba8();
        let alpha = |image: &image::RgbaImage| -> Vec<u8> {
            image.pixels().map(|pixel| pixel[3]).collect()
        };
        assert_eq!(alpha(&decoded), alpha(&source), "alpha is lossless");
    }

    #[test]
    fn empty_images_are_refused() {
        let empty = image::RgbImage::new(0, 0);
        assert!(encode_webp_still_rgb(&empty, WEBP_STILL_QUALITY).is_err());
    }
}
