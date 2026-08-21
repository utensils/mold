//! Upstream-parity preprocessing for LTX-2 still-image conditioning.
//!
//! Official LTX-2 does not feed a pristine still to the VAE
//! (`ltx_pipelines/utils/media_io/decode.py:46-79` @ fd4ded7). It
//! 1. decodes with EXIF orientation applied and embedded ICC profiles
//!    converted to sRGB (`decode_image`, decode.py:139-170),
//! 2. round-trips the still through a one-frame H.264/YUV420 encode at the
//!    checkpoint generation's training CRF, at native resolution
//!    (`preprocess`, decode.py:386-435),
//! 3. aspect-preserving fill-resizes and center-crops to the requested
//!    canvas (`resize_and_center_crop`, resize.py:41-73), and
//! 4. normalizes to `[-1, 1]`.
//!
//! The order is load-bearing: compression happens at native resolution,
//! fitting afterwards. This module is LTX-2 specific on purpose — other
//! families keep `img_utils::decode_source_image` semantics.
//!
//! CRF is an x264 rate-distortion concept the bundled openh264 encoder
//! cannot express; the honest stand-in is a constant quantizer (the
//! `QpRange` clamped to a single value) on the one IDR frame, gated by
//! golden fixtures captured from upstream's libx264 round-trip (see
//! `testdata/preprocess/README.md`). Saved metadata records the codec
//! actually used — never a CRF claim.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use mold_core::ltx2_preprocess::Ltx2ImagePreprocessingProfile;

/// Codec identity recorded in provenance metadata for the constant-QP
/// H.264 round-trip. Deliberately names openh264 and the quantizer, not
/// "CRF": the two are different rate-control models.
pub(crate) fn roundtrip_codec_label(profile: &Ltx2ImagePreprocessingProfile) -> String {
    format!("openh264-cqp{}", profile.image_crf)
}

/// Fit policy identity recorded in provenance metadata.
pub(crate) const FIT_POLICY_LABEL: &str = "fill-center-crop";

/// EXIF-oriented, ICC-corrected decode.
///
/// Lived here until #1222 needed the same upright decode for identity
/// images; it now sits in [`crate::img_utils`] so there is exactly one
/// orientation path in the crate, and is re-exported under its original
/// name so every LTX-2 call site and citation still reads the same.
pub(crate) use crate::img_utils::decode_oriented_srgb;

/// An image resized/cropped in float space. Upstream never re-quantizes
/// after its bilinear resize (the float tensor flows straight into
/// normalization), so the fitted result stays `f32` in the 0..=255 domain.
pub(crate) struct FittedImage {
    /// HWC, row-major, 0..=255 domain.
    pub(crate) data: Vec<f32>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Aspect-preserving fill resize + center crop, exactly upstream
/// `resize_and_center_crop` (resize.py:60-70): `scale = max(th/sh, tw/sw)`,
/// new dims `ceil(src * scale)` (the ceil avoids negative crop offsets),
/// bilinear interpolation with torch `align_corners=False` semantics,
/// then a `floor((new - target) / 2)` crop.
pub(crate) fn resize_fill_center_crop(
    rgb: &RgbImage,
    target_w: u32,
    target_h: u32,
) -> Result<FittedImage> {
    anyhow::ensure!(
        target_w > 0 && target_h > 0,
        "target dimensions must be non-zero"
    );
    let (src_w, src_h) = rgb.dimensions();
    anyhow::ensure!(src_w > 0 && src_h > 0, "source image is empty");
    let scale = f64::max(
        f64::from(target_h) / f64::from(src_h),
        f64::from(target_w) / f64::from(src_w),
    );
    let new_h = (f64::from(src_h) * scale).ceil() as u32;
    let new_w = (f64::from(src_w) * scale).ceil() as u32;
    let resized = bilinear_resize_f32(rgb, new_w, new_h);
    let crop_top = (new_h - target_h) / 2;
    let crop_left = (new_w - target_w) / 2;
    let mut data = Vec::with_capacity(target_w as usize * target_h as usize * 3);
    for y in 0..target_h {
        let src_row = (crop_top + y) as usize;
        let start = (src_row * new_w as usize + crop_left as usize) * 3;
        data.extend_from_slice(&resized[start..start + target_w as usize * 3]);
    }
    Ok(FittedImage {
        data,
        width: target_w,
        height: target_h,
    })
}

/// Bilinear resample with torch `align_corners=False` coordinate mapping:
/// `src = (dst + 0.5) * (src_len / dst_len) - 0.5`, edge-clamped. Output
/// is HWC f32 in the 0..=255 domain. The `image` crate's Triangle filter
/// is deliberately not used — its window handling is not torch-parity.
fn bilinear_resize_f32(rgb: &RgbImage, new_w: u32, new_h: u32) -> Vec<f32> {
    let (src_w, src_h) = rgb.dimensions();
    let src = rgb.as_raw();
    if (new_w, new_h) == (src_w, src_h) {
        return src.iter().map(|&v| f32::from(v)).collect();
    }
    let scale_x = src_w as f64 / new_w as f64;
    let scale_y = src_h as f64 / new_h as f64;
    let sample_axis = |dst: u32, scale: f64, src_len: u32| -> (usize, usize, f32) {
        let pos = (f64::from(dst) + 0.5) * scale - 0.5;
        let floor = pos.floor();
        let frac = (pos - floor) as f32;
        let lo = (floor.max(0.0) as usize).min(src_len as usize - 1);
        let hi = ((floor + 1.0).max(0.0) as usize).min(src_len as usize - 1);
        (lo, hi, frac)
    };
    let mut out = Vec::with_capacity(new_w as usize * new_h as usize * 3);
    let stride = src_w as usize * 3;
    for y in 0..new_h {
        let (y0, y1, fy) = sample_axis(y, scale_y, src_h);
        for x in 0..new_w {
            let (x0, x1, fx) = sample_axis(x, scale_x, src_w);
            for c in 0..3 {
                let p00 = f32::from(src[y0 * stride + x0 * 3 + c]);
                let p01 = f32::from(src[y0 * stride + x1 * 3 + c]);
                let p10 = f32::from(src[y1 * stride + x0 * 3 + c]);
                let p11 = f32::from(src[y1 * stride + x1 * 3 + c]);
                let top = p00 + (p01 - p00) * fx;
                let bottom = p10 + (p11 - p10) * fx;
                out.push(top + (bottom - top) * fy);
            }
        }
    }
    out
}

/// One-frame H.264/YUV420 round-trip at constant quantizer `qp`, matching
/// upstream `preprocess` (decode.py:413-435): `qp == 0` and images with a
/// side below 2 px pass through unchanged; dimensions are floored to even
/// by dropping the last row/column (decode.py:391-393) before the 4:2:0
/// encode. Color conversion is a mold-owned rounded BT.601 limited-range
/// pair rather than the openh264 crate's truncating helpers: truncation
/// biases every sample dark by up to ~4 codes, which is a systematic
/// conditioning shift upstream's swscale path does not have.
pub(crate) fn h264_roundtrip(rgb: &RgbImage, qp: u8) -> Result<RgbImage> {
    use openh264::encoder::{BitRate, EncoderConfig, FrameRate, QpRange, RateControlMode};
    use openh264::formats::YUVSlices;

    if qp == 0 {
        return Ok(rgb.clone());
    }
    let (width, height) = rgb.dimensions();
    if width < 2 || height < 2 {
        return Ok(rgb.clone());
    }
    let even_w = width / 2 * 2;
    let even_h = height / 2 * 2;
    let cropped;
    let rgb = if (even_w, even_h) == (width, height) {
        rgb
    } else {
        cropped = image::imageops::crop_imm(rgb, 0, 0, even_w, even_h).to_image();
        &cropped
    };

    // A `QpRange` clamped to one value pins the quantizer regardless of
    // the rate-control target; the generous bitrate and disabled frame
    // skipping keep the RC from interfering with the single IDR frame.
    let config = EncoderConfig::new()
        .max_frame_rate(FrameRate::from_hz(1.0))
        .bitrate(BitRate::from_bps(50_000_000))
        .rate_control_mode(RateControlMode::Quality)
        .skip_frames(false)
        .qp(QpRange::new(qp, qp))
        .profile(openh264::encoder::Profile::High)
        .vui(openh264::encoder::VuiConfig::bt601());
    let api = openh264::OpenH264API::from_source();
    let mut encoder = openh264::encoder::Encoder::with_api_config(api, config)
        .context("failed to create the H.264 conditioning encoder")?;
    let (y_plane, u_plane, v_plane) = rgb_to_yuv420_bt601(rgb);
    let (w, h) = (even_w as usize, even_h as usize);
    let yuv = YUVSlices::new((&y_plane, &u_plane, &v_plane), (w, h), (w, w / 2, w / 2));
    let bitstream = encoder
        .encode(&yuv)
        .context("failed to encode the conditioning frame")?
        .to_vec();

    let api = openh264::OpenH264API::from_source();
    let mut decoder = openh264::decoder::Decoder::with_api_config(
        api,
        openh264::decoder::DecoderConfig::new().flush_after_decode(openh264::decoder::Flush::Flush),
    )
    .context("failed to create the H.264 conditioning decoder")?;
    let decoded = match decoder
        .decode(&bitstream)
        .context("failed to decode the conditioning frame")?
    {
        Some(frame) => frame_to_rgb(&frame, even_w, even_h)?,
        None => {
            let mut flushed = decoder.flush_remaining().context(
                "H.264 conditioning decoder produced no frame for the encoded conditioning image",
            )?;
            let frame = flushed.pop().context(
                "H.264 conditioning decoder produced no frame for the encoded conditioning image",
            )?;
            frame_to_rgb(&frame, even_w, even_h)?
        }
    };
    Ok(decoded)
}

/// Rounded BT.601 limited-range RGB → planar YUV 4:2:0. Chroma is computed
/// from the 2×2 box-averaged RGB block, matching swscale's default
/// subsampling closely enough that flat colors survive within a code.
fn rgb_to_yuv420_bt601(rgb: &RgbImage) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);
    let src = rgb.as_raw();
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; (w / 2) * (h / 2)];
    let mut v_plane = vec![0u8; (w / 2) * (h / 2)];
    let clamp = |v: f32| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for row in 0..h {
        for col in 0..w {
            let i = (row * w + col) * 3;
            let (r, g, b) = (
                f32::from(src[i]),
                f32::from(src[i + 1]),
                f32::from(src[i + 2]),
            );
            y_plane[row * w + col] = clamp(0.256_788 * r + 0.504_129 * g + 0.097_906 * b + 16.0);
        }
    }
    for row in 0..h / 2 {
        for col in 0..w / 2 {
            let mut sums = [0.0f32; 3];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let i = ((row * 2 + dy) * w + col * 2 + dx) * 3;
                sums[0] += f32::from(src[i]);
                sums[1] += f32::from(src[i + 1]);
                sums[2] += f32::from(src[i + 2]);
            }
            let (r, g, b) = (sums[0] / 4.0, sums[1] / 4.0, sums[2] / 4.0);
            u_plane[row * (w / 2) + col] =
                clamp(-0.148_223 * r - 0.290_993 * g + 0.439_216 * b + 128.0);
            v_plane[row * (w / 2) + col] =
                clamp(0.439_216 * r - 0.367_788 * g - 0.071_427 * b + 128.0);
        }
    }
    (y_plane, u_plane, v_plane)
}

/// Rounded BT.601 limited-range planar YUV 4:2:0 → RGB, reading the
/// decoder's strided planes directly (each 2×2 block shares one chroma
/// sample). Inverse of [`rgb_to_yuv420_bt601`] up to codec quantization.
fn frame_to_rgb(frame: &openh264::decoder::DecodedYUV<'_>, w: u32, h: u32) -> Result<RgbImage> {
    use openh264::formats::YUVSource;
    let (dec_w, dec_h) = frame.dimensions();
    anyhow::ensure!(
        dec_w >= w as usize && dec_h >= h as usize,
        "decoded conditioning frame ({dec_w}x{dec_h}) smaller than the encoded input ({w}x{h})"
    );
    let (stride_y, stride_u, stride_v) = frame.strides();
    let (y_plane, u_plane, v_plane) = (frame.y(), frame.u(), frame.v());
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; w * h * 3];
    let clamp = |v: f32| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for row in 0..h {
        for col in 0..w {
            let y = f32::from(y_plane[row * stride_y + col]) - 16.0;
            let u = f32::from(u_plane[(row / 2) * stride_u + col / 2]) - 128.0;
            let v = f32::from(v_plane[(row / 2) * stride_v + col / 2]) - 128.0;
            let i = (row * w + col) * 3;
            out[i] = clamp(1.164_384 * y + 1.596_027 * v);
            out[i + 1] = clamp(1.164_384 * y - 0.391_762 * u - 0.812_968 * v);
            out[i + 2] = clamp(1.164_384 * y + 2.017_232 * u);
        }
    }
    RgbImage::from_raw(w as u32, h as u32, out).context("decoded conditioning frame size mismatch")
}

/// Full native-resolution preprocessing: oriented sRGB decode, then the
/// generation's H.264 round-trip. Fitting (resize/crop/normalize) happens
/// separately per render stage via [`fit_conditioning_image`] because the
/// two LTX-2 stages condition at different resolutions while the
/// compression step is resolution-independent.
pub(crate) fn preprocess_conditioning_image(
    bytes: &[u8],
    profile: &Ltx2ImagePreprocessingProfile,
) -> Result<RgbImage> {
    let decoded = decode_oriented_srgb(bytes)?;
    h264_roundtrip(&decoded, profile.image_crf)
}

/// [`preprocess_conditioning_image`] for a staged file, memoized so a
/// two-stage render (whose stages re-derive conditioning at their own
/// resolutions) decodes and round-trips each source once. Staged paths
/// live in per-request work directories, so `(path, crf)` is unique per
/// request; the cache holds the last few native-resolution images only.
pub(crate) fn cached_native_conditioning_image(
    path: &str,
    profile: &Ltx2ImagePreprocessingProfile,
) -> Result<std::sync::Arc<RgbImage>> {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex, OnceLock};

    type CacheEntry = ((String, u8), Arc<RgbImage>);
    static CACHE: OnceLock<Mutex<VecDeque<CacheEntry>>> = OnceLock::new();
    const CACHE_CAP: usize = 4;

    let key = (path.to_string(), profile.image_crf);
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    if let Some((_, image)) = cache
        .lock()
        .expect("conditioning image cache poisoned")
        .iter()
        .find(|(entry_key, _)| *entry_key == key)
    {
        return Ok(image.clone());
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read staged LTX-2 conditioning image '{path}'"))?;
    let image = Arc::new(
        preprocess_conditioning_image(&bytes, profile)
            .with_context(|| format!("failed to preprocess LTX-2 conditioning image '{path}'"))?,
    );
    let mut cache = cache.lock().expect("conditioning image cache poisoned");
    cache.push_back((key, image.clone()));
    while cache.len() > CACHE_CAP {
        cache.pop_front();
    }
    Ok(image)
}

/// Fit a preprocessed native-resolution image to the conditioning canvas
/// and normalize to `[-1, 1]` (upstream `x / 127.5 - 1`), producing the
/// `[1, 3, H, W]` tensor the VAE encodes.
pub(crate) fn fit_conditioning_image(
    native: &RgbImage,
    target_w: u32,
    target_h: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let fitted = resize_fill_center_crop(native, target_w, target_h)?;
    let data: Vec<f32> = fitted.data.iter().map(|&v| v / 127.5 - 1.0).collect();
    let tensor = Tensor::from_vec(
        data,
        (fitted.height as usize, fitted.width as usize, 3),
        &Device::Cpu,
    )?
    .permute((2, 0, 1))?
    .unsqueeze(0)?;
    Ok(tensor.to_dtype(dtype)?.to_device(device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    // The ICC fallback tests below still exercise this directly; it moved to
    // `img_utils` with `decode_oriented_srgb` but its behaviour is LTX-2's
    // upstream-parity requirement, so the coverage stays here.
    use crate::img_utils::convert_icc_to_srgb;
    use mold_core::ltx2_preprocess::{ltx2_image_preprocessing_profile, Ltx2Generation};

    const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ltx2/testdata/preprocess/");

    fn fixture_bytes(name: &str) -> Vec<u8> {
        std::fs::read(format!("{TESTDATA}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn fixture_u8(name: &str, h: usize, w: usize) -> Vec<u8> {
        let bytes = fixture_bytes(name);
        assert_eq!(bytes.len(), h * w * 3, "{name} shape mismatch");
        bytes
    }

    fn fixture_f32(name: &str, h: usize, w: usize) -> Vec<f32> {
        let bytes = fixture_bytes(name);
        assert_eq!(bytes.len(), h * w * 3 * 4, "{name} shape mismatch");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn mean_abs_u8(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .sum::<f64>()
            / a.len() as f64
    }

    fn max_abs_u8(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x.abs_diff(y))
            .max()
            .unwrap_or(0)
    }

    fn psnr_u8(a: &[u8], b: &[u8]) -> f64 {
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = f64::from(x) - f64::from(y);
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    // ── resize: torch align_corners=False parity ────────────────────────

    #[test]
    fn resize_fill_center_crop_matches_upstream_bilinear_goldens() {
        let cases: &[(&str, &str, u32, u32)] = &[
            (
                "gradient_96x64.png",
                "resized_gradient_96x64_to_64x64.bin",
                64,
                64,
            ),
            (
                "photo_like_128x96.png",
                "resized_photo_like_128x96_to_96x96.bin",
                96,
                96,
            ),
            (
                "oddsize_97x63.png",
                "resized_oddsize_97x63_to_64x48.bin",
                64,
                48,
            ),
        ];
        for (input, golden, tw, th) in cases {
            let rgb = image::load_from_memory(&fixture_bytes(input))
                .unwrap()
                .to_rgb8();
            let fitted = resize_fill_center_crop(&rgb, *tw, *th).unwrap();
            let expected = fixture_f32(golden, *th as usize, *tw as usize);
            let max_err = fitted
                .data
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err <= 1e-2,
                "{golden}: max deviation {max_err} exceeds float-rounding tolerance"
            );
        }
    }

    #[test]
    fn resize_never_stretches_aspect() {
        // A 2:1 source fitted to a square must center-crop, never squash:
        // the output must be exactly the middle columns of the source.
        let mut rgb = RgbImage::new(8, 4);
        for (x, _, pixel) in rgb.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x * 30) as u8, 0, 0]);
        }
        let fitted = resize_fill_center_crop(&rgb, 4, 4).unwrap();
        for y in 0..4usize {
            for x in 0..4usize {
                let expected = ((x + 2) * 30) as f32;
                let got = fitted.data[(y * 4 + x) * 3];
                assert_eq!(got, expected, "pixel ({x},{y}) not a pure center crop");
            }
        }
    }

    // ── decode: EXIF orientation + ICC-to-sRGB ──────────────────────────

    #[test]
    fn exif_orientation_6_jpeg_is_rotated_upright() {
        let decoded = decode_oriented_srgb(&fixture_bytes("portrait_exif6.jpg")).unwrap();
        assert_eq!(decoded.dimensions(), (64, 96), "orientation not applied");
        let expected = fixture_u8("decoded_portrait_exif6.bin", 96, 64);
        let mean = mean_abs_u8(decoded.as_raw(), &expected);
        let max = max_abs_u8(decoded.as_raw(), &expected);
        eprintln!("[measure] exif6: mean-abs {mean:.3} max {max}");
        // Different JPEG decoders (zune-jpeg vs libjpeg-turbo) may differ by
        // a code or two per sample; orientation errors differ by ~100s.
        assert!(mean <= 2.0, "mean-abs {mean}");
    }

    #[test]
    fn display_p3_png_converts_to_srgb_within_tolerance() {
        let bytes = fixture_bytes("display_p3.png");
        let decoded = decode_oriented_srgb(&bytes).unwrap();
        let expected = fixture_u8("decoded_display_p3.bin", 64, 64);
        let mean = mean_abs_u8(decoded.as_raw(), &expected);
        let max = max_abs_u8(decoded.as_raw(), &expected);
        eprintln!("[measure] p3: mean-abs {mean:.3} max {max}");
        // moxcms vs lcms2 round differently; skipping the conversion
        // entirely mis-measures by mean-abs ~13 (fixture README).
        assert!(mean <= 3.0, "mean-abs {mean} — ICC conversion drifted");
        // And prove the conversion actually ran: the raw (untransformed)
        // pixels are much further from the sRGB golden.
        let untransformed = image::load_from_memory(&bytes).unwrap().to_rgb8();
        let untransformed_mean = mean_abs_u8(untransformed.as_raw(), &expected);
        assert!(
            untransformed_mean > 8.0,
            "fixture no longer distinguishes converted from unconverted ({untransformed_mean})"
        );
    }

    #[test]
    fn grayscale_source_with_gray_icc_profile_is_transformed_not_skipped() {
        // Codex review (PR #1072): a Gray-profile image flattened to RGB
        // before the transform made moxcms reject the Rgb layout and fall
        // back to assume-sRGB — silently skipping the conversion.
        let gray_profile = moxcms::ColorProfile::new_gray_with_gamma(1.0)
            .encode()
            .unwrap();
        let img = image::GrayImage::from_pixel(8, 8, image::Luma([128]));
        let mut png = Vec::new();
        {
            let mut encoder = image::codecs::png::PngEncoder::new(&mut png);
            image::ImageEncoder::set_icc_profile(&mut encoder, gray_profile).unwrap();
            image::ImageEncoder::write_image(
                encoder,
                img.as_raw(),
                8,
                8,
                image::ExtendedColorType::L8,
            )
            .unwrap();
        }
        let decoded = decode_oriented_srgb(&png).unwrap();
        assert_eq!(decoded.dimensions(), (8, 8));
        // A gamma-1.0 (linear) gray profile maps mid-gray 128 (~0.502
        // linear) to a noticeably brighter sRGB value (~0.737 → ~188); an
        // untransformed fallback would leave it at exactly 128.
        let px = decoded.get_pixel(4, 4).0;
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
        assert!(
            px[0] > 150,
            "gray ICC profile was not applied (pixel stayed {})",
            px[0]
        );
    }

    #[test]
    fn malformed_icc_profile_falls_back_to_srgb() {
        let rgb = image::load_from_memory(&fixture_bytes("gradient_96x64.png"))
            .unwrap()
            .to_rgb8();
        let out = convert_icc_to_srgb(rgb.clone(), b"not an icc profile", moxcms::Layout::Rgb);
        assert_eq!(out.as_raw(), rgb.as_raw());
    }

    // ── H.264 round-trip ────────────────────────────────────────────────

    #[test]
    fn crf_zero_and_tiny_images_bypass_roundtrip() {
        let rgb = image::load_from_memory(&fixture_bytes("gradient_96x64.png"))
            .unwrap()
            .to_rgb8();
        let identity = h264_roundtrip(&rgb, 0).unwrap();
        assert_eq!(identity.as_raw(), rgb.as_raw());
        let tiny = RgbImage::new(1, 8);
        let tiny_out = h264_roundtrip(&tiny, 33).unwrap();
        assert_eq!(tiny_out.as_raw(), tiny.as_raw());
    }

    #[test]
    fn odd_dimension_input_is_floored_even_before_roundtrip() {
        let rgb = image::load_from_memory(&fixture_bytes("oddsize_97x63.png"))
            .unwrap()
            .to_rgb8();
        let out = h264_roundtrip(&rgb, 33).unwrap();
        // Upstream drops the last row/column (decode.py:391-393): 97x63 → 96x62.
        assert_eq!(out.dimensions(), (96, 62));
    }

    #[test]
    fn h264_roundtrip_flat_colors_survive_chroma_conversion() {
        // Flat frames are almost free for the codec, so what remains is the
        // RGB→YUV420→RGB path itself. A coefficient/range bug (full-range vs
        // BT.601 limited) shifts flat mid-tones by tens of codes.
        for color in [
            [128u8, 128, 128],
            [200, 60, 60],
            [60, 200, 60],
            [60, 60, 200],
        ] {
            let rgb = RgbImage::from_pixel(64, 64, image::Rgb(color));
            let out = h264_roundtrip(&rgb, 20).unwrap();
            let mean = mean_abs_u8(out.as_raw(), rgb.as_raw());
            eprintln!("[measure] flat {color:?}: mean-abs {mean:.3}");
            assert!(mean <= 3.0, "flat {color:?} shifted by {mean}");
        }
    }

    #[test]
    #[ignore = "measurement sweep"]
    fn qp_sweep_measurement() {
        for (input, golden, h, w) in [
            (
                "gradient_96x64.png",
                "crf33_gradient_96x64.bin",
                64usize,
                96usize,
            ),
            (
                "photo_like_128x96.png",
                "crf33_photo_like_128x96.bin",
                96,
                128,
            ),
        ] {
            let original = image::load_from_memory(&fixture_bytes(input))
                .unwrap()
                .to_rgb8();
            let upstream = fixture_u8(golden, h, w);
            for qp in [32u8, 33, 34, 35, 36, 37, 38] {
                let out = h264_roundtrip(&original, qp).unwrap();
                eprintln!(
                    "[sweep] {input} qp={qp}: vs-orig {:.2} dB / {:.3}; vs-upstream {:.2} dB / {:.3}",
                    psnr_u8(out.as_raw(), original.as_raw()),
                    mean_abs_u8(out.as_raw(), original.as_raw()),
                    psnr_u8(out.as_raw(), &upstream),
                    mean_abs_u8(out.as_raw(), &upstream),
                );
            }
        }
    }

    #[test]
    fn qp33_roundtrip_lands_in_upstream_crf33_degradation_envelope() {
        // libx264 CRF 33 measured against the originals (fixture README):
        // 25.14 dB PSNR / 10.11 mean-abs on the photo-like fixture. The
        // openh264 constant-QP-33 stand-in must degrade photo-statistics
        // content comparably — measured at 24.71 dB / 10.69 (0.43 dB from
        // upstream), gated at ±2 dB. The synthetic gradient fixture is
        // deliberately NOT envelope-gated: x264's psychovisual RD spends
        // extra distortion on smooth gradients (measured gap ~4.9 dB) in a
        // way openh264 has no equivalent for; it keeps the weaker gates
        // below so a regression to no-op or garbage still fails.
        let photo = image::load_from_memory(&fixture_bytes("photo_like_128x96.png"))
            .unwrap()
            .to_rgb8();
        let photo_upstream = fixture_u8("crf33_photo_like_128x96.bin", 96, 128);
        let photo_mold = h264_roundtrip(&photo, 33).unwrap();
        let mold_psnr = psnr_u8(photo_mold.as_raw(), photo.as_raw());
        let cross_psnr = psnr_u8(photo_mold.as_raw(), &photo_upstream);
        eprintln!(
            "[measure] photo_like: mold-vs-original {mold_psnr:.2} dB \
             (upstream 25.14 dB); mold-vs-upstream {cross_psnr:.2} dB"
        );
        const UPSTREAM_PHOTO_PSNR: f64 = 25.14;
        assert!(
            (mold_psnr - UPSTREAM_PHOTO_PSNR).abs() <= 2.0,
            "photo-like degradation {mold_psnr:.2} dB outside the upstream \
             CRF-33 envelope {UPSTREAM_PHOTO_PSNR:.2} dB"
        );
        // The two round-tripped results must describe the same signal —
        // closer to each other than either is to the original.
        assert!(
            cross_psnr >= UPSTREAM_PHOTO_PSNR + 1.0,
            "mold round-trip is not converging on upstream's ({cross_psnr:.2} dB)"
        );

        // Gradient: compression must actually happen and stay correlated
        // with upstream's output, but no envelope claim (psy-rd caveat).
        let gradient = image::load_from_memory(&fixture_bytes("gradient_96x64.png"))
            .unwrap()
            .to_rgb8();
        let gradient_upstream = fixture_u8("crf33_gradient_96x64.bin", 64, 96);
        let gradient_mold = h264_roundtrip(&gradient, 33).unwrap();
        assert!(
            mean_abs_u8(gradient_mold.as_raw(), gradient.as_raw()) > 0.5,
            "gradient round-trip was a no-op"
        );
        let gradient_cross = psnr_u8(gradient_mold.as_raw(), &gradient_upstream);
        eprintln!("[measure] gradient: mold-vs-upstream {gradient_cross:.2} dB");
        assert!(
            gradient_cross >= 23.0,
            "gradient round-trip diverged from upstream ({gradient_cross:.2} dB)"
        );
    }

    // ── composition + provenance ────────────────────────────────────────

    #[test]
    fn profile_crf_flows_into_roundtrip_and_labels() {
        let profile = ltx2_image_preprocessing_profile(Ltx2Generation::V2);
        assert_eq!(roundtrip_codec_label(&profile), "openh264-cqp33");
        assert_eq!(FIT_POLICY_LABEL, "fill-center-crop");
        let bytes = fixture_bytes("gradient_96x64.png");
        let preprocessed = preprocess_conditioning_image(&bytes, &profile).unwrap();
        let pristine = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert!(
            mean_abs_u8(preprocessed.as_raw(), pristine.as_raw()) > 0.5,
            "profile CRF did not reach the round-trip"
        );
    }

    #[test]
    fn fit_conditioning_image_normalizes_to_unit_range() {
        let rgb = image::load_from_memory(&fixture_bytes("gradient_96x64.png"))
            .unwrap()
            .to_rgb8();
        let tensor = fit_conditioning_image(&rgb, 32, 32, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 32, 32]);
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(
            min >= -1.0 - 1e-4 && max <= 1.0 + 1e-4,
            "range [{min}, {max}]"
        );
    }

    #[test]
    fn cached_native_conditioning_image_reuses_the_decoded_image() {
        let profile = ltx2_image_preprocessing_profile(Ltx2Generation::V2);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.png");
        std::fs::write(&path, fixture_bytes("gradient_96x64.png")).unwrap();
        let path = path.to_string_lossy().to_string();
        let first = cached_native_conditioning_image(&path, &profile).unwrap();
        let second = cached_native_conditioning_image(&path, &profile).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
    }
}
