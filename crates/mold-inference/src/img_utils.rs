//! Image decoding and preprocessing utilities for img2img.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, ImageDecoder, RgbImage};
use std::io::Cursor;

/// Normalization range for source images before VAE encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeRange {
    /// [-1, 1] for SD1.5, SDXL, FLUX, Flux.2, Qwen-Image, SD3, and Z-Image.
    MinusOneToOne,
    /// [0, 1] for ControlNet hint images and Wuerstchen's VQ-GAN encoder.
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

    let img = img
        .resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3)
        .to_rgb8();

    rgb_image_to_tensor(img, range, device, dtype)
}

/// Decode and preprocess one FLUX.2 Dev reference exactly like BFL's
/// `default_prep`: reject sides below 64 px and aspect ratios above 8:1,
/// downscale only when the image exceeds its pixel cap, then center-crop both
/// dimensions down to a multiple of 16. References are never upscaled.
pub fn decode_flux2_reference_image(
    bytes: &[u8],
    max_pixels: u64,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| anyhow::anyhow!("failed to decode FLUX.2 reference image: {error}"))?;
    let image = preprocess_flux2_reference(image, max_pixels)?;
    rgb_image_to_tensor(image, NormalizeRange::MinusOneToOne, device, dtype)
}

fn preprocess_flux2_reference(image: DynamicImage, max_pixels: u64) -> Result<RgbImage> {
    const MIN_SIDE: u32 = 64;
    const MAX_ASPECT_RATIO: u32 = 8;
    const DIMENSION_MULTIPLE: u32 = 16;

    let mut image = image.to_rgb8();
    let (width, height) = image.dimensions();
    if width < MIN_SIDE || height < MIN_SIDE {
        bail!(
            "FLUX.2 reference images require both sides to be at least {MIN_SIDE}px (got {width}x{height})"
        );
    }
    let (long, short) = if width >= height {
        (width, height)
    } else {
        (height, width)
    };
    if u64::from(long) > u64::from(short) * u64::from(MAX_ASPECT_RATIO) {
        bail!(
            "FLUX.2 reference images support aspect ratios up to {MAX_ASPECT_RATIO}:1 (got {width}x{height})"
        );
    }

    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        let scale = (max_pixels as f64 / pixels as f64).sqrt();
        let scaled_width = ((width as f64 * scale) as u32).max(1);
        let scaled_height = ((height as f64 * scale) as u32).max(1);
        image = image::imageops::resize(
            &image,
            scaled_width,
            scaled_height,
            image::imageops::FilterType::Lanczos3,
        );
    }

    let crop_width = image.width() / DIMENSION_MULTIPLE * DIMENSION_MULTIPLE;
    let crop_height = image.height() / DIMENSION_MULTIPLE * DIMENSION_MULTIPLE;
    let left = (image.width() - crop_width) / 2;
    let top = (image.height() - crop_height) / 2;
    Ok(image::imageops::crop_imm(&image, left, top, crop_width, crop_height).to_image())
}

fn rgb_image_to_tensor(
    img: RgbImage,
    range: NormalizeRange,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
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

/// Decode a mask image (PNG/JPEG) into a [1, 1, latent_h, latent_w] tensor with values in [0, 1].
/// White (255) = 1.0 = repaint region, Black (0) = 0.0 = preserve region.
/// RGB images are converted to grayscale.
pub fn decode_mask_image(
    bytes: &[u8],
    latent_height: usize,
    latent_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode mask image: {e}"))?;

    let img = img.resize_exact(
        latent_width as u32,
        latent_height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let gray = img.to_luma8();

    let data: Vec<f32> = gray.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

    let tensor = Tensor::from_vec(data, (1, 1, latent_height, latent_width), &Device::Cpu)?;
    let tensor = tensor.to_dtype(dtype)?.to_device(device)?;

    Ok(tensor)
}

// ---------------------------------------------------------------------------
// Oriented sRGB decode
//
// Shared by LTX-2 still conditioning (`ltx2::preprocess`, which re-exports it)
// and PuLID identity extraction (`identity`). One orientation path in the
// crate on purpose: a second table is a second chance to disagree about which
// way up a photograph is.
// ---------------------------------------------------------------------------

/// Decode image bytes into an upright sRGB image: EXIF orientation is
/// applied, alpha is flattened, and an embedded ICC profile is converted
/// to sRGB (malformed profiles warn and assume sRGB, mirroring upstream
/// decode.py:164-166).
pub fn decode_oriented_srgb(bytes: &[u8]) -> Result<RgbImage> {
    decode_oriented_srgb_with_limits(bytes, image::Limits::default())
}

/// [`decode_oriented_srgb`] with caller-supplied decoder limits.
///
/// H3 accepts user media through several durable and local entry points. They
/// all need the same EXIF/ICC semantics without giving up the family's strict
/// dimension and allocation bounds merely to share this decoder.
pub fn decode_oriented_srgb_with_limits(bytes: &[u8], limits: image::Limits) -> Result<RgbImage> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("failed to sniff source image format")?;
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .context("failed to decode source image")?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let icc = decoder.icc_profile().unwrap_or_default();
    let mut decoded =
        DynamicImage::from_decoder(decoder).context("failed to decode source image")?;
    // Upstream order (decode.py:143-163): orient, flatten to RGB, then
    // convert color to sRGB.
    decoded.apply_orientation(orientation);
    // A grayscale source with a Gray ICC profile must be transformed in
    // its own layout — moxcms (correctly) refuses an Rgb-layout transform
    // from a Gray profile, and pre-flattening to RGB would turn the
    // conversion into the assume-sRGB fallback (codex review, PR #1072).
    let is_grayscale = matches!(
        decoded.color(),
        image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::L16
            | image::ColorType::La16
    );
    let rgb = decoded.to_rgb8();
    Ok(match icc {
        Some(profile) if !profile.is_empty() => {
            let layout = if is_grayscale {
                moxcms::Layout::Gray
            } else {
                moxcms::Layout::Rgb
            };
            convert_icc_to_srgb(rgb, &profile, layout)
        }
        _ => rgb,
    })
}

/// Read the dimensions the upright decoded pixels will have without
/// allocating the image.
///
/// The reader must already have a format (normally via
/// `with_guessed_format`). Keeping this beside the decode path means every H3
/// descriptor agrees with the pixels after EXIF orientation is applied.
pub fn oriented_image_dimensions<R>(
    mut reader: image::ImageReader<R>,
    limits: image::Limits,
) -> Result<(u32, u32)>
where
    R: std::io::BufRead + std::io::Seek,
{
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .context("failed to decode source image")?;
    let (width, height) = decoder.dimensions();
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    Ok(match orientation {
        image::metadata::Orientation::Rotate90
        | image::metadata::Orientation::Rotate270
        | image::metadata::Orientation::Rotate90FlipH
        | image::metadata::Orientation::Rotate270FlipH => (height, width),
        _ => (width, height),
    })
}

/// [`decode_oriented_srgb`] that KEEPS the alpha channel.
///
/// Same orientation and ICC handling; the colour transform still runs on the
/// three colour planes, and alpha is carried through untouched because it is
/// not a colour and no profile describes it.
///
/// This exists for Hunyuan3D, where alpha is not decoration but the subject
/// mask: `hunyuan3d::dino2::letterbox_square` crops and centres on the
/// non-zero alpha bounding box, so flattening first makes a background-removed
/// cutout — the input the docs recommend as the BEST one — indistinguishable
/// from a full opaque frame, and conditions the vision tower on the whole
/// canvas including the black transparent pixels.
pub fn decode_oriented_srgb_rgba(bytes: &[u8]) -> Result<image::RgbaImage> {
    decode_oriented_srgb_rgba_with_limits(bytes, image::Limits::no_limits())
}

pub fn decode_oriented_srgb_rgba_with_limits(
    bytes: &[u8],
    limits: image::Limits,
) -> Result<image::RgbaImage> {
    decode_oriented_srgb_rgba_inner(bytes, limits, |_, _| Ok(()), |decoded| decoded.to_rgba8())
}

/// Decode one ordered reference image (`GenerateRequest.edit_images`) the one
/// way both of its readers need it: the Qwen Image 2.1 engine's conditioning
/// (`qwen_image21::reference::prepare_reference`) and the output-alpha rule
/// (`image::encoded_image_has_alpha`). One decoder and one limit policy, so
/// the answer "does this reference carry alpha" is a fact about the very
/// pixels the engine conditions on.
///
/// - Bounded: `mold_core::reference_image::reference_decode_limits` (the
///   decoder refuses a side past the admitted maximum) and the admission
///   area/aspect bounds, checked on the header BEFORE any pixel is decoded.
///   Admission refuses the same bytes first; this is the engine's own floor
///   for a caller that reached it some other way.
/// - Upright: the EXIF orientation is applied. Upstream opens the reference
///   with PIL (`pipeline_qwenimage21.py:653-660` at `e0abab83b`), which does
///   NOT, so a phone photo would be conditioned sideways; mold deliberately
///   differs, and every canvas authority reads the oriented size
///   (`mold_core::reference_image::oriented_dimensions`).
/// - sRGB: an embedded ICC profile is converted to sRGB (alpha untouched).
///   PIL ignores ICC profiles entirely, so this too is a deliberate
///   divergence: a Display-P3 phone photo conditions on the colours it shows,
///   not on its raw P3 code values read as sRGB.
/// - 8-bit exactly as Pillow converts: see [`pillow_rgba8`].
pub fn decode_reference_rgba(bytes: &[u8]) -> Result<image::RgbaImage> {
    decode_oriented_srgb_rgba_inner(
        bytes,
        mold_core::reference_image::reference_decode_limits(),
        |width, height| {
            mold_core::reference_image::validate_reference_image_dimensions(
                "A reference image",
                width,
                height,
            )
            .map_err(anyhow::Error::msg)
        },
        pillow_rgba8,
    )
}

/// PIL's `Image.open(...).convert("RGBA")` from a decoded image, which is what
/// upstream hands the reference path (`pipeline_qwenimage21.py:653-654`).
///
/// The 8-bit cases are the `image` crate's own conversion. The 16-bit PNG
/// cases are NOT: `image` rescales with `x / 257` (rounded), while Pillow's
/// PNG plugin unpacks 16-bit RGB, RGBA and LA by taking the HIGH byte
/// (`PngImagePlugin.py` `_MODES` `RGB;16B`/`RGBA;16B`/`LA;16B`, unpacked by
/// `Unpack.c` `unpackRGB16B`/`unpackRGBA16B`, i.e. `x >> 8`), and opens 16-bit
/// grayscale as `I;16`, whose `convert("RGBA")` clamps
/// (`Convert.c` `I16_RGB`: `in[1] == 0 ? in[0] : 255`, i.e. `min(x, 255)`).
/// Measured against Pillow 12.3.0: alpha `0xFF00` is 255 there and 254 under
/// `x / 257`, which flips "has alpha below 255" for a reference that is
/// opaque to upstream.
pub fn pillow_rgba8(decoded: DynamicImage) -> image::RgbaImage {
    let high = |value: u16| (value >> 8) as u8;
    match decoded {
        DynamicImage::ImageRgb16(image) => {
            image::RgbaImage::from_fn(image.width(), image.height(), |x, y| {
                let [r, g, b] = image.get_pixel(x, y).0;
                image::Rgba([high(r), high(g), high(b), 255])
            })
        }
        DynamicImage::ImageRgba16(image) => {
            image::RgbaImage::from_fn(image.width(), image.height(), |x, y| {
                image::Rgba(image.get_pixel(x, y).0.map(high))
            })
        }
        DynamicImage::ImageLumaA16(image) => {
            image::RgbaImage::from_fn(image.width(), image.height(), |x, y| {
                let [l, a] = image.get_pixel(x, y).0;
                let l = high(l);
                image::Rgba([l, l, l, high(a)])
            })
        }
        DynamicImage::ImageLuma16(image) => {
            image::RgbaImage::from_fn(image.width(), image.height(), |x, y| {
                let l = image.get_pixel(x, y).0[0].min(255) as u8;
                image::Rgba([l, l, l, 255])
            })
        }
        other => other.to_rgba8(),
    }
}

fn decode_oriented_srgb_rgba_inner(
    bytes: &[u8],
    limits: image::Limits,
    check_dimensions: impl FnOnce(u32, u32) -> Result<()>,
    to_rgba8: impl FnOnce(DynamicImage) -> image::RgbaImage,
) -> Result<image::RgbaImage> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("failed to sniff source image format")?;
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .context("failed to decode source image")?;
    let (width, height) = decoder.dimensions();
    check_dimensions(width, height)?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let icc = decoder.icc_profile().unwrap_or_default();
    let mut decoded =
        DynamicImage::from_decoder(decoder).context("failed to decode source image")?;
    // DELIBERATE DIVERGENCE: PIL's `open` leaves the EXIF orientation
    // unapplied, so upstream conditions on a phone photo sideways. Every
    // canvas authority reads this same oriented size
    // (`mold_core::reference_image::oriented_dimensions`).
    decoded.apply_orientation(orientation);
    let is_grayscale = matches!(
        decoded.color(),
        image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::L16
            | image::ColorType::La16
    );
    let mut rgba = to_rgba8(decoded);
    // DELIBERATE DIVERGENCE from the PIL-based upstreams this decoder serves
    // (Qwen Image 2.1's `pipeline_qwenimage21.py:653-660`, Hunyuan3D's
    // loaders): PIL's `open`/`convert` never applies an embedded ICC profile,
    // so upstream reads a Display-P3 or Adobe RGB file's code values as if
    // they were sRGB. mold converts them, like the EXIF orientation above,
    // so a wide-gamut photo conditions on the colours it actually shows.
    if let Some(profile) = icc {
        if !profile.is_empty() {
            let layout = if is_grayscale {
                moxcms::Layout::Gray
            } else {
                moxcms::Layout::Rgb
            };
            let (width, height) = rgba.dimensions();
            let mut rgb = RgbImage::new(width, height);
            for (target, source) in rgb.pixels_mut().zip(rgba.pixels()) {
                *target = image::Rgb([source.0[0], source.0[1], source.0[2]]);
            }
            let converted = convert_icc_to_srgb(rgb, &profile, layout);
            for (target, source) in rgba.pixels_mut().zip(converted.pixels()) {
                target.0[0] = source.0[0];
                target.0[1] = source.0[1];
                target.0[2] = source.0[2];
            }
        }
    }
    Ok(rgba)
}

/// Convert `rgb` from the embedded `profile` to sRGB, reading the source
/// pixels in `source_layout` (`Gray` collapses the flattened RGB back to
/// one channel — the three are identical for a decoded grayscale image).
/// Any failure — malformed profile, unsupported connection — logs and
/// returns the pixels unchanged (assume-sRGB), exactly upstream's
/// fallback.
pub(crate) fn convert_icc_to_srgb(
    rgb: RgbImage,
    profile: &[u8],
    source_layout: moxcms::Layout,
) -> RgbImage {
    let source = match moxcms::ColorProfile::new_from_slice(profile) {
        Ok(profile) => profile,
        Err(error) => {
            tracing::warn!("ignoring malformed embedded ICC profile: {error}");
            return rgb;
        }
    };
    let srgb = moxcms::ColorProfile::new_srgb();
    let options = moxcms::TransformOptions {
        rendering_intent: moxcms::RenderingIntent::Perceptual,
        ..Default::default()
    };
    let transform =
        match source.create_transform_8bit(source_layout, &srgb, moxcms::Layout::Rgb, options) {
            Ok(transform) => transform,
            Err(error) => {
                tracing::warn!("cannot convert embedded ICC profile to sRGB: {error}");
                return rgb;
            }
        };
    let (width, height) = rgb.dimensions();
    let src_rgb = rgb.into_raw();
    let src: std::borrow::Cow<'_, [u8]> = match source_layout {
        moxcms::Layout::Gray => src_rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| px[0])
            .collect::<Vec<_>>()
            .into(),
        _ => (&src_rgb).into(),
    };
    let mut dst = vec![0u8; width as usize * height as usize * 3];
    if let Err(error) = transform.transform(&src, &mut dst) {
        tracing::warn!("ICC-to-sRGB conversion failed: {error}");
        return RgbImage::from_raw(width, height, src_rgb)
            .expect("source buffer matches dimensions");
    }
    RgbImage::from_raw(width, height, dst).expect("destination buffer matches dimensions")
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;

    fn encode_test_png(pixel: [u8; 3]) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(1, 1, Rgb(pixel));
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, ImageFormat::Png)
            .expect("encode PNG");
        out.into_inner()
    }

    fn decode_values(bytes: &[u8], range: NormalizeRange) -> Vec<f32> {
        decode_source_image(bytes, 1, 1, range, &Device::Cpu, DType::F32)
            .expect("decode source image")
            .flatten_all()
            .expect("flatten decoded tensor")
            .to_vec1::<f32>()
            .expect("decoded values")
    }

    #[test]
    fn zero_to_one_normalization_preserves_unit_interval() {
        let bytes = encode_test_png([0, 128, 255]);
        let values = decode_values(&bytes, NormalizeRange::ZeroToOne);

        assert_eq!(values.len(), 3);
        assert!((values[0] - 0.0).abs() < 1e-6);
        assert!((values[1] - (128.0 / 255.0)).abs() < 1e-6);
        assert!((values[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn minus_one_to_one_normalization_centers_and_scales_pixels() {
        let bytes = encode_test_png([0, 128, 255]);
        let values = decode_values(&bytes, NormalizeRange::MinusOneToOne);

        assert_eq!(values.len(), 3);
        assert!((values[0] + 1.0).abs() < 1e-6);
        assert!((values[1] - ((128.0 / 255.0) * 2.0 - 1.0)).abs() < 1e-6);
        assert!((values[2] - 1.0).abs() < 1e-6);
    }

    fn encode_solid_png(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(width, height, Rgb([32, 64, 96]));
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, ImageFormat::Png)
            .expect("encode PNG");
        out.into_inner()
    }

    #[test]
    fn flux2_reference_prep_never_upscales_and_center_crops_to_sixteen() {
        let bytes = encode_solid_png(513, 527);
        let tensor = decode_flux2_reference_image(
            &bytes,
            mold_core::validation::FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 512, 512]);
    }

    #[test]
    fn flux2_reference_prep_downscales_only_above_the_cap() {
        let bytes = encode_solid_png(4096, 1024);
        let tensor = decode_flux2_reference_image(
            &bytes,
            mold_core::validation::FLUX2_MULTI_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 512, 2048]);
    }

    #[test]
    fn flux2_reference_prep_rejects_small_sides_and_extreme_aspects() {
        let small = encode_solid_png(512, 63);
        assert!(decode_flux2_reference_image(
            &small,
            mold_core::validation::FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err()
        .to_string()
        .contains("at least 64px"));

        let wide = encode_solid_png(1024, 64);
        assert!(decode_flux2_reference_image(
            &wide,
            mold_core::validation::FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err()
        .to_string()
        .contains("up to 8:1"));
    }
}

/// Context for inpainting: holds pre-computed tensors needed during the denoising loop.
pub struct InpaintContext {
    /// VAE-encoded original latents (unnoised).
    pub original_latents: Tensor,
    /// Mask tensor [1, 1, latent_h, latent_w] with values in [0, 1].
    /// 1.0 = repaint, 0.0 = preserve.
    pub mask: Tensor,
    /// Noise tensor matching latent shape, for re-noising the original at each step.
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
    // ── Mask decoding tests ───────────────────────────────────────────────

    /// Create a 4x4 white PNG (mask = repaint everywhere).
    fn white_mask_png() -> Vec<u8> {
        let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Create a 4x4 black PNG (mask = preserve everywhere).
    fn black_mask_png() -> Vec<u8> {
        let img = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_mask_shape() {
        let mask = white_mask_png();
        let tensor = decode_mask_image(&mask, 8, 8, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 8, 8]);
    }

    #[test]
    fn decode_mask_white_is_one() {
        let mask = white_mask_png();
        let tensor = decode_mask_image(&mask, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(min > 0.99, "white mask should be ~1.0, got {min}");
    }

    #[test]
    fn decode_mask_black_is_zero() {
        let mask = black_mask_png();
        let tensor = decode_mask_image(&mask, 4, 4, &Device::Cpu, DType::F32).unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max < 0.01, "black mask should be ~0.0, got {max}");
    }

    #[test]
    fn decode_mask_rgb_converted_to_grayscale() {
        // Red RGB image -- grayscale luminance of pure red is ~76/255 ~ 0.3
        let rgb = tiny_png(); // 4x4 red
        let tensor = decode_mask_image(&rgb, 4, 4, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 4, 4]);
        let val = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(
            val > 0.1 && val < 0.5,
            "red -> grayscale should be ~0.3, got {val}"
        );
    }
}

#[cfg(test)]
mod reference_decode_tests {
    use super::*;
    use image::{ImageBuffer, Luma, LumaA, Rgb, Rgba};

    /// The 16-bit samples measured against Pillow 12.3.0, and what
    /// `Image.open(png).convert("RGBA")` made of each: the high byte for
    /// RGB, RGBA and LA, a clamp for grayscale (`I;16`).
    const SAMPLES: [u16; 10] = [
        0x0000, 0x00ff, 0x0100, 0x7fff, 0x8000, 0xff00, 0xff7f, 0xff80, 0xfeff, 0xffff,
    ];
    const PIL_HIGH_BYTE: [u8; 10] = [0, 0, 1, 127, 128, 255, 255, 255, 254, 255];
    const PIL_I16_CLAMP: [u8; 10] = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    fn png<P: image::PixelWithColorType>(image: &ImageBuffer<P, Vec<P::Subpixel>>) -> Vec<u8>
    where
        [P::Subpixel]: image::EncodableLayout,
    {
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn sixteen_bit_pngs_convert_exactly_as_pillow_does() {
        let rgba = ImageBuffer::from_fn(10, 1, |x, _| Rgba([SAMPLES[x as usize]; 4]));
        let rgb = ImageBuffer::from_fn(10, 1, |x, _| Rgb([SAMPLES[x as usize]; 3]));
        let la = ImageBuffer::from_fn(10, 1, |x, _| LumaA([SAMPLES[x as usize]; 2]));
        let l = ImageBuffer::from_fn(10, 1, |x, _| Luma([SAMPLES[x as usize]]));
        let channel = |bytes: Vec<u8>, index: usize| -> Vec<u8> {
            decode_reference_rgba(&bytes)
                .unwrap()
                .pixels()
                .map(|pixel| pixel.0[index])
                .collect()
        };
        assert_eq!(channel(png(&rgba), 0), PIL_HIGH_BYTE);
        assert_eq!(channel(png(&rgba), 3), PIL_HIGH_BYTE);
        assert_eq!(channel(png(&rgb), 1), PIL_HIGH_BYTE);
        assert_eq!(channel(png(&rgb), 3), [255; 10]);
        assert_eq!(channel(png(&la), 2), PIL_HIGH_BYTE);
        assert_eq!(channel(png(&la), 3), PIL_HIGH_BYTE);
        assert_eq!(channel(png(&l), 0), PIL_I16_CLAMP);
        assert_eq!(channel(png(&l), 3), [255; 10]);
    }

    #[test]
    fn eight_bit_references_decode_unchanged() {
        let image =
            image::RgbaImage::from_fn(6, 4, |x, y| Rgba([x as u8 * 40, y as u8 * 60, 7, 200]));
        assert_eq!(decode_reference_rgba(&png(&image)).unwrap(), image);
    }

    /// The decoder refuses a reference past the admitted bounds from its
    /// header, before a single pixel is allocated.
    #[test]
    fn an_oversized_reference_is_refused_before_it_is_decoded() {
        let mut bytes = png(&image::RgbaImage::new(1, 1));
        // Rewrite IHDR to 60000x60000 and fix its CRC.
        bytes[16..20].copy_from_slice(&60_000u32.to_be_bytes());
        bytes[20..24].copy_from_slice(&60_000u32.to_be_bytes());
        let crc = crc32(&bytes[12..29]);
        bytes[29..33].copy_from_slice(&crc.to_be_bytes());
        let error = format!("{:#}", decode_reference_rgba(&bytes).unwrap_err());
        assert!(
            error.contains("decode") || error.contains("16384"),
            "{error}"
        );
        // Inside the side bound but past the area bound.
        bytes[16..20].copy_from_slice(&12_000u32.to_be_bytes());
        bytes[20..24].copy_from_slice(&9_000u32.to_be_bytes());
        let crc = crc32(&bytes[12..29]);
        bytes[29..33].copy_from_slice(&crc.to_be_bytes());
        let error = format!("{:#}", decode_reference_rgba(&bytes).unwrap_err());
        assert!(error.contains("100000000"), "{error}");
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }
}
