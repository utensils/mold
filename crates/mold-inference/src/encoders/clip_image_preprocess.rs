//! Image preprocessing for the CLIP-ViT-H-14 vision tower.
//!
//! Upstream is `transformers` 4.57.3
//! `models/clip/image_processing_clip.py`, driven with the defaults
//! `h94/IP-Adapter`'s `models/image_encoder/preprocessor_config.json` records
//! (`CLIPImageProcessor`, `shortest_edge: 224`, `crop_size: 224`, bicubic).
//! `preprocess` (`:326-336`) applies four steps in this order and mold applies
//! the same four:
//!
//! ```text
//! :328  resize          shortest edge -> 224, bicubic, aspect preserved
//! :331  center_crop     224 x 224
//! :334  rescale         x 1/255
//! :337  normalize       (x - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD
//! ```
//!
//! It is reached from `diffusers`
//! `pipelines/stable_diffusion/pipeline_stable_diffusion.py:517-518`
//! (`self.feature_extractor(image, return_tensors="pt").pixel_values`), which
//! is the IP-Adapter call site.
//!
//! ## This is not EVA's preprocessing
//!
//! [`super::eva_clip_preprocess`] serves PuLID's EVA02-CLIP tower, which
//! **stretches** the whole frame to a 336 square and never crops
//! (`PuLID/pulid/pipeline_flux.py:173`, a bare `resize` to `image_size`). CLIP
//! preserves the aspect ratio, matches the SHORTEST edge, and then throws away
//! the middle-out remainder of the long one. Feeding one tower the other's
//! pixels produces a perfectly well-shaped, differently-framed embedding. The
//! two modules share the normalization statistics and the resampling kernel and
//! nothing else.
//!
//! ## Why the resize is integer arithmetic
//!
//! `image_transforms.resize` (`:356-391`) does not resample the array at all:
//! it converts to a **PIL image** and calls `Image.resize`. The input at that
//! point is still `uint8` — `do_rescale` runs two steps later — so Pillow
//! resamples in 8-bit and every intermediate is quantized and CLAMPED back into
//! `[0, 255]`. Bicubic overshoots at edges, so the clamp is not a rounding
//! detail: measured against Pillow 12.3.0 on a high-frequency 512 -> 224
//! downscale, resampling the same weights in `f32` and rounding at the end
//! differs by up to **18/255**, and on natural imagery by 1/255 nearly
//! everywhere. So this module ports Pillow's fixed-point path
//! (`src/libImaging/Resample.c`, `precompute_coeffs` /
//! `normalize_coeffs_8bpc` / `ImagingResampleHorizontal_8bpc`) rather than
//! reusing the float resize next door, and reproduces Pillow **bit-for-bit** on
//! every case it was checked against.
//!
//! `transformers` 5.x added a torchvision-backed fast processor and made it the
//! default; its resampling numerics have not been compared here. The PIL
//! processor is the one this port tracks, because it is what the version
//! `diffusers`' IP-Adapter runs against resolves to and what 5.x still exposes
//! as `CLIPImageProcessorPil`.
//!
//! The weight computation itself is Pillow's `precompute_coeffs`, which is the
//! same function `super::eva_clip_preprocess::axis_taps` implements for aten's
//! antialiased bicubic — support widened by the downscale ratio, Keys cubic
//! with `a = -0.5`, renormalized to sum to one. That module's copy is private
//! to it and its output is `f32`, so this is a deliberate mirror rather than a
//! call: the shared part is [`axis_taps`], and the difference is everything
//! after it.

// The IP-Adapter pipeline that consumes this module lands separately; until
// that consumer exists every item here is reachable only from tests, so the
// dead-code lint would otherwise force either a premature `pub` surface or a
// stub caller. This mirrors `eva_clip_preprocess`'s note for the same reason.
#![allow(dead_code)]

use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};

use super::openclip_vision::IMAGE_SIZE;

/// Channels the tower reads. Alpha is dropped, never composited: PIL's
/// `convert_to_rgb` (`image_transforms.py:781-798`) is a plain
/// `Image.convert("RGB")`, which discards the alpha band.
const CHANNELS: usize = 3;

/// Shortest edge the resize targets (`preprocessor_config.json`,
/// `size.shortest_edge`). Equal to the crop and to the tower's input edge for
/// this checkpoint, but they are three different settings upstream and are kept
/// as three names here.
pub(crate) const SHORTEST_EDGE: usize = IMAGE_SIZE;
/// Centre-crop edge (`crop_size`).
pub(crate) const CROP_SIZE: usize = IMAGE_SIZE;

/// `OPENAI_CLIP_MEAN` (`transformers/image_utils.py`), which reads
/// `(0.48145466, 0.4578275, 0.40821073)`.
///
/// Byte-identical to [`super::eva_clip_preprocess::CLIP_MEAN`] — OpenAI's CLIP
/// statistics are the same numbers under both names, and a test asserts they
/// stay that way. They are restated here rather than imported because the two
/// modules cite different upstreams and a future checkpoint could move one.
pub(crate) const OPENAI_CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// `OPENAI_CLIP_STD`, `(0.26862954, 0.26130258, 0.27577711)`. The last two are
/// written in their shortest f32-exact form because the extra digits round to
/// the same `f32` and clippy rejects them; the values are unchanged.
pub(crate) const OPENAI_CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Keys cubic coefficient. Pillow's `BICUBIC` filter is `a = -0.5`
/// (`Resample.c`, `bicubic_filter`), the same variant torchvision's
/// antialiased bicubic uses and NOT the `-0.75` of PyTorch's plain bicubic.
const CUBIC_A: f64 = -0.5;
/// Bicubic reaches two source samples either side before the support is scaled.
const CUBIC_SUPPORT: f64 = 2.0;
/// Pillow's `PRECISION_BITS`, `32 - 8 - 2` (`Resample.c`). The coefficients are
/// rounded onto this fixed-point grid and the accumulator is shifted back down
/// by it.
const PRECISION_BITS: u32 = 22;
/// `ss = 1 << (PRECISION_BITS - 1)` — the round-to-nearest bias Pillow seeds
/// each accumulator with before the arithmetic shift truncates.
const ROUNDING_BIAS: i64 = 1 << (PRECISION_BITS - 1);

/// `bicubic_filter` — the Keys convolution kernel.
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * CUBIC_A) * x + 8.0 * CUBIC_A) * x - 4.0 * CUBIC_A
    } else {
        0.0
    }
}

/// One output sample's source window and its normalized weights.
#[derive(Debug, Clone)]
struct Taps {
    start: usize,
    weights: Vec<f64>,
}

/// The separable weight table for one axis, matching Pillow's
/// `precompute_coeffs`: the support widens by the downscale ratio (that
/// widening *is* the antialiasing), the filter argument is divided by the same
/// ratio, and the weights are renormalized to sum to one so truncated edge
/// windows stay unbiased.
fn axis_taps(input: usize, output: usize) -> Vec<Taps> {
    let scale = input as f64 / output as f64;
    // `filterscale = scale; if (filterscale < 1.0) filterscale = 1.0;`
    let filter_scale = scale.max(1.0);
    let support = CUBIC_SUPPORT * filter_scale;
    let inverse = 1.0 / filter_scale;
    (0..output)
        .map(|index| {
            let center = scale * (index as f64 + 0.5);
            // Both bounds truncate toward zero exactly as the C casts do; the
            // lower one is then clamped at 0 and the upper at `input`.
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let mut weights: Vec<f64> = (start..end)
                .map(|source| cubic((source as f64 - center + 0.5) * inverse))
                .collect();
            let total: f64 = weights.iter().sum();
            if total != 0.0 {
                for weight in &mut weights {
                    *weight /= total;
                }
            }
            Taps { start, weights }
        })
        .collect()
}

/// One output sample's window with the weights on Pillow's fixed-point grid.
#[derive(Debug, Clone)]
struct FixedTaps {
    start: usize,
    weights: Vec<i32>,
}

/// `normalize_coeffs_8bpc` — round each coefficient to `PRECISION_BITS`,
/// AWAY FROM ZERO. `(int)(0.5 + w * (1 << bits))` for a positive weight and
/// `(int)(-0.5 + ...)` for a negative one; Rust's `f64::round` is the same
/// half-away-from-zero rule, and the sign split exists in C only because a cast
/// truncates.
fn quantize(taps: &[Taps]) -> Vec<FixedTaps> {
    let unit = f64::from(1u32 << PRECISION_BITS);
    taps.iter()
        .map(|taps| FixedTaps {
            start: taps.start,
            weights: taps
                .weights
                .iter()
                .map(|weight| (weight * unit).round() as i32)
                .collect(),
        })
        .collect()
}

/// `clip8` — arithmetic shift back down, then clamp into a byte. Pillow does
/// the clamp with a lookup table; the table is a clamp.
fn clip8(accumulator: i64) -> u8 {
    (accumulator >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow's two-pass 8-bit resample of an interleaved RGB buffer: horizontal
/// first into a `u8` intermediate, then vertical. The intermediate really is
/// bytes — that second quantization is part of the reference and is why this
/// cannot be folded into one separable pass in a wider type.
fn resize_bicubic_u8(
    pixels: &[u8],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<u8> {
    let columns = quantize(&axis_taps(width, out_width));
    let mut horizontal = vec![0_u8; height * out_width * CHANNELS];
    for row in 0..height {
        let source = row * width * CHANNELS;
        let target = row * out_width * CHANNELS;
        for (index, taps) in columns.iter().enumerate() {
            for channel in 0..CHANNELS {
                let mut accumulator = ROUNDING_BIAS;
                for (offset, weight) in taps.weights.iter().enumerate() {
                    let sample = pixels[source + (taps.start + offset) * CHANNELS + channel];
                    accumulator += i64::from(sample) * i64::from(*weight);
                }
                horizontal[target + index * CHANNELS + channel] = clip8(accumulator);
            }
        }
    }

    let rows = quantize(&axis_taps(height, out_height));
    let mut resized = vec![0_u8; out_height * out_width * CHANNELS];
    for (index, taps) in rows.iter().enumerate() {
        let target = index * out_width * CHANNELS;
        for column in 0..out_width {
            for channel in 0..CHANNELS {
                let mut accumulator = ROUNDING_BIAS;
                for (offset, weight) in taps.weights.iter().enumerate() {
                    let row = taps.start + offset;
                    let sample = horizontal[(row * out_width + column) * CHANNELS + channel];
                    accumulator += i64::from(sample) * i64::from(*weight);
                }
                resized[target + (column * CHANNELS) + channel] = clip8(accumulator);
            }
        }
    }
    resized
}

/// `get_resize_output_image_size(..., default_to_square=False)`
/// (`image_transforms.py:302-309`): the shortest edge becomes
/// `shortest_edge` and the longest is scaled by the same ratio, TRUNCATED —
/// `int(requested_new_short * long / short)`, not rounded.
fn resize_output_size(height: usize, width: usize, shortest_edge: usize) -> (usize, usize) {
    let (short, long) = if width <= height {
        (width, height)
    } else {
        (height, width)
    };
    let new_long = (shortest_edge as f64 * long as f64 / short as f64) as usize;
    if width <= height {
        (new_long, shortest_edge)
    } else {
        (shortest_edge, new_long)
    }
}

/// `BaseImageProcessor.center_crop` -> `image_transforms.center_crop`
/// (`:498-513`): floor-divide the slack, take the crop from there.
///
/// Upstream also has a zero-padding branch for a crop larger than the image
/// (`:515-536`). It is unreachable here and the `ensure!` says so rather than
/// implementing it: the resize always leaves the short axis at exactly
/// `SHORTEST_EDGE` and the long one at `>= SHORTEST_EDGE`, and `CROP_SIZE`
/// equals `SHORTEST_EDGE` for this checkpoint.
fn center_crop(pixels: &[u8], height: usize, width: usize, crop: usize) -> Result<Vec<u8>> {
    ensure!(
        height >= crop && width >= crop,
        "a {height}x{width} image cannot supply a {crop}x{crop} centre crop"
    );
    let top = (height - crop) / 2;
    let left = (width - crop) / 2;
    let mut cropped = vec![0_u8; crop * crop * CHANNELS];
    for row in 0..crop {
        let source = ((top + row) * width + left) * CHANNELS;
        let target = row * crop * CHANNELS;
        cropped[target..target + crop * CHANNELS]
            .copy_from_slice(&pixels[source..source + crop * CHANNELS]);
    }
    Ok(cropped)
}

/// Run the whole `CLIPImageProcessor` pipeline over an interleaved
/// (HWC) 8-bit RGB buffer.
///
/// The result is `[1, 3, 224, 224]` f32 on `device`, planar and normalized,
/// ready for [`super::openclip_vision::OpenClipVisionTower::forward`].
pub(crate) fn preprocess_rgb8(
    pixels: &[u8],
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    ensure!(
        height > 0 && width > 0 && pixels.len() == CHANNELS * height * width,
        "expected {height} x {width} x {CHANNELS} interleaved samples, got {}",
        pixels.len()
    );
    let (out_height, out_width) = resize_output_size(height, width, SHORTEST_EDGE);
    // `Image.resize` short-circuits an identity resize
    // (`PIL/Image.py`, `if self.size == size ... return self.copy()`); the
    // tap table is the identity there anyway, so this is a shortcut and not a
    // second behaviour.
    let resized = if (out_height, out_width) == (height, width) {
        pixels.to_vec()
    } else {
        resize_bicubic_u8(pixels, height, width, out_height, out_width)
    };
    let cropped = center_crop(&resized, out_height, out_width, CROP_SIZE)?;

    // `rescale` then `normalize`, both in f32, and the interleaved buffer
    // becomes planar because `data_format` defaults to `ChannelDimension.FIRST`
    // (`image_processing_clip.py:214`).
    let plane = CROP_SIZE * CROP_SIZE;
    let mut planar = vec![0.0_f32; CHANNELS * plane];
    for (index, sample) in cropped.iter().enumerate() {
        let channel = index % CHANNELS;
        let pixel = index / CHANNELS;
        let scaled = f32::from(*sample) / 255.0;
        planar[channel * plane + pixel] =
            (scaled - OPENAI_CLIP_MEAN[channel]) / OPENAI_CLIP_STD[channel];
    }
    Ok(Tensor::from_vec(
        planar,
        (1, CHANNELS, CROP_SIZE, CROP_SIZE),
        device,
    )?)
}

/// Decode an sRGB image into the interleaved 8-bit RGB buffer
/// [`preprocess_rgb8`] expects.
///
/// `to_rgb8` drops alpha and replicates grey, which is what
/// `Image.convert("RGB")` does for the same inputs.
pub(crate) fn rgb8_from_image(image: &image::DynamicImage) -> (Vec<u8>, usize, usize) {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    (rgb.into_raw(), height, width)
}

/// Convenience wrapper: decode and preprocess in one call.
pub(crate) fn preprocess_image(image: &image::DynamicImage, device: &Device) -> Result<Tensor> {
    let (pixels, height, width) = rgb8_from_image(image);
    preprocess_rgb8(&pixels, height, width, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    /// The deterministic source both oracle captures below were taken from.
    /// Channel formulas differ per channel on purpose: a channel swap or an
    /// HWC/CHW transpose cannot survive a comparison against them.
    fn synthetic(height: usize, width: usize) -> Vec<u8> {
        let mut pixels = vec![0_u8; height * width * CHANNELS];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * CHANNELS;
                pixels[offset] = ((x * 7 + y * 3) % 256) as u8;
                pixels[offset + 1] = ((x * x + y * 5) % 256) as u8;
                pixels[offset + 2] = ((x * 11 + y * y) % 256) as u8;
            }
        }
        pixels
    }

    /// These are OpenAI's CLIP statistics verbatim, and they are the same
    /// numbers PuLID's EVA tower normalizes with. A transposed or rounded copy
    /// shifts every embedding slightly and nothing else notices.
    #[test]
    fn the_normalization_constants_are_openais() {
        assert_eq!(OPENAI_CLIP_MEAN, [0.481_454_66, 0.457_827_5, 0.408_210_73]);
        assert_eq!(OPENAI_CLIP_STD, [0.268_629_54, 0.261_302_6, 0.275_777_1]);
        // The literals above are the f32 values of the published f64 ones.
        assert_eq!(OPENAI_CLIP_MEAN[0], 0.48145466_f64 as f32);
        assert_eq!(OPENAI_CLIP_STD[1], 0.26130258_f64 as f32);
        assert_eq!(OPENAI_CLIP_STD[2], 0.27577711_f64 as f32);
        // `OPENAI_CLIP_*` and eva_clip's `OPENAI_DATASET_*` are one set of
        // numbers under two upstream names. If a checkpoint ever moves one,
        // this is where that shows up.
        assert_eq!(
            OPENAI_CLIP_MEAN,
            super::super::eva_clip_preprocess::CLIP_MEAN
        );
        assert_eq!(OPENAI_CLIP_STD, super::super::eva_clip_preprocess::CLIP_STD);
        // Distinct per channel — a single scalar would silently pass a shape
        // test.
        assert_ne!(OPENAI_CLIP_MEAN[0], OPENAI_CLIP_MEAN[2]);
        assert_ne!(OPENAI_CLIP_STD[0], OPENAI_CLIP_STD[2]);
    }

    /// Shortest edge to 224, aspect preserved, long edge truncated.
    /// The five cases were read off `get_resize_output_image_size` and
    /// confirmed against Pillow.
    #[test]
    fn the_resize_matches_the_shortest_edge_rule() {
        // Landscape: height is short, so height becomes 224.
        assert_eq!(resize_output_size(200, 320, 224), (224, 358));
        assert_eq!(resize_output_size(480, 640, 224), (224, 298));
        // Portrait: width is short.
        assert_eq!(resize_output_size(320, 200, 224), (358, 224));
        // Square passes through, upscale and downscale alike.
        assert_eq!(resize_output_size(224, 224, 224), (224, 224));
        assert_eq!(resize_output_size(100, 100, 224), (224, 224));
        // int(224 * 640 / 480) = int(298.66) = 298, not 299. Rounding here
        // shifts the crop window by half a pixel on every landscape photo.
        assert_eq!(
            resize_output_size(480, 640, 224).1,
            (224.0 * 640.0 / 480.0) as usize
        );
        // Both axes always clear the crop, which is what makes upstream's
        // zero-padding branch unreachable.
        for (height, width) in [(200, 320), (320, 200), (1, 4000), (4000, 1), (224, 224)] {
            let (out_height, out_width) = resize_output_size(height, width, 224);
            assert!(
                out_height >= 224 && out_width >= 224,
                "{out_height}x{out_width}"
            );
        }
    }

    /// The tap table for a 4 -> 2 downscale, computed by hand from the Keys
    /// kernel.
    ///
    /// `scale = 2`, so `filterscale = 2`, `support = 4` and the filter
    /// argument is halved. Output 0 has `center = 2 * 0.5 = 1`, so the window
    /// is `[max(0, int(1 - 4 + 0.5)), min(4, int(1 + 4 + 0.5))) = [0, 4)` and
    /// the four offsets are `(s - 1 + 0.5) / 2` = -0.25, 0.25, 0.75, 1.25:
    ///
    /// ```text
    /// cubic(0.25) = (1.5*0.25 - 2.5) * 0.0625 + 1        =  0.8671875
    /// cubic(0.75) = (1.5*0.75 - 2.5) * 0.5625 + 1        =  0.2265625
    /// cubic(1.25) = ((1.25-5)*-0.5*1.25 - 4) * 1.25 + 2  = -0.0703125
    /// sum         = 2*0.8671875 + 0.2265625 - 0.0703125  =  1.890625
    /// ```
    #[test]
    fn the_downscale_taps_are_the_hand_computed_keys_weights() {
        let taps = axis_taps(4, 2);
        assert_eq!(taps.len(), 2);
        assert_eq!(taps[0].start, 0);
        let total = 1.890625_f64;
        let expected = [
            0.8671875 / total,
            0.8671875 / total,
            0.2265625 / total,
            -0.0703125 / total,
        ];
        assert_eq!(taps[0].weights.len(), expected.len());
        for (actual, expected) in taps[0].weights.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
        }
        // The negative lobe is the whole point of a cubic; a filter that
        // clamped it to zero would blur instead of resample.
        assert!(taps[0].weights[3] < 0.0);
        // Every window is a partition of unity, or a flat image would not
        // survive the resize.
        for (input, output) in [(4_usize, 2_usize), (512, 224), (224, 224), (100, 224)] {
            for taps in axis_taps(input, output) {
                let total: f64 = taps.weights.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-12,
                    "{input}->{output} weights sum to {total}"
                );
                assert!(taps.start + taps.weights.len() <= input, "window overruns");
            }
        }
    }

    /// Downscaling widens the filter support; upscaling does not. That
    /// asymmetry is the antialiasing.
    #[test]
    fn downscaling_widens_the_support() {
        let widest = |input, output| {
            axis_taps(input, output)
                .iter()
                .map(|taps| taps.weights.len())
                .max()
                .unwrap()
        };
        assert!(widest(512, 224) > widest(100, 224));
        assert_eq!(widest(100, 224), 4, "an upscale reads the usual four taps");
    }

    /// The Keys coefficient is -0.5. -0.75 (PyTorch's plain bicubic) is a
    /// different, wrong filter here; pin the kernel so a "cleanup" cannot swap
    /// it.
    #[test]
    fn the_cubic_kernel_is_the_minus_one_half_variant() {
        assert!((cubic(0.0) - 1.0).abs() < 1e-12);
        assert!(cubic(1.0).abs() < 1e-12);
        assert!(cubic(2.0).abs() < 1e-12);
        assert!(cubic(3.0).abs() < 1e-12);
        // f(0.5) = 0.5625 for a = -0.5; it is 0.5859375 for a = -0.75.
        assert!((cubic(0.5) - 0.5625).abs() < 1e-12, "{}", cubic(0.5));
        assert!((cubic(1.5) + 0.0625).abs() < 1e-12, "{}", cubic(1.5));
    }

    /// An identity resize is exactly the identity, and a flat image survives
    /// any resize exactly.
    ///
    /// Both follow from the fixed-point arithmetic rather than from luck: the
    /// identity table is `[0, 1, 0, 0]`, and for a constant `c` the
    /// accumulator is `2^21 + c * sum(k)` with `sum(k)` within a few units of
    /// `2^22`, which shifts back to `c` for every byte value.
    #[test]
    fn a_flat_image_and_an_identity_resize_are_exact() {
        let identity = resize_bicubic_u8(&synthetic(6, 4), 6, 4, 6, 4);
        assert_eq!(identity, synthetic(6, 4));

        for value in [0_u8, 1, 127, 254, 255] {
            let flat = vec![value; 32 * 48 * CHANNELS];
            let resized = resize_bicubic_u8(&flat, 32, 48, 224, 336);
            assert!(
                resized.iter().all(|sample| *sample == value),
                "a flat {value} image drifted"
            );
        }
    }

    /// Bit-exact against Pillow 12.3.0 on a 4x6 -> 2x3 downscale, captured
    /// from `Image.fromarray(src, "RGB").resize((3, 2), Image.BICUBIC)`.
    ///
    /// A float resize with the same weights does NOT reproduce these bytes;
    /// see the module note. This case is small enough to read: the source is
    /// three smooth ramps, so nothing here is riding on the clamp, and the
    /// values it does pin are the fixed-point rounding and the
    /// horizontal-then-vertical pass order.
    #[test]
    fn the_resize_is_bit_exact_against_pillow() {
        let source = synthetic(4, 6);
        assert_eq!(
            source,
            vec![
                0, 0, 0, 7, 1, 11, 14, 4, 22, 21, 9, 33, 28, 16, 44, 35, 25, 55, 3, 5, 1, 10, 6,
                12, 17, 9, 23, 24, 14, 34, 31, 21, 45, 38, 30, 56, 6, 10, 4, 13, 11, 15, 20, 14,
                26, 27, 19, 37, 34, 26, 48, 41, 35, 59, 9, 15, 9, 16, 16, 20, 23, 19, 31, 30, 24,
                42, 37, 31, 53, 44, 40, 64,
            ],
            "the synthetic source drifted from the one the oracle was run on"
        );
        let resized = resize_bicubic_u8(&source, 4, 6, 2, 3);
        assert_eq!(
            resized,
            vec![6, 3, 7, 20, 10, 29, 33, 23, 50, 11, 12, 12, 25, 19, 34, 38, 32, 55]
        );
    }

    /// End-to-end against `CLIPImageProcessor` itself: a 32x48 source resizes
    /// to 224x336, crops at `left = (336 - 224) / 2 = 56`, rescales and
    /// normalizes. The probes were captured by running the processor
    /// (`CLIPImageProcessorPil`, Pillow 12.3.0) over this exact source with
    /// `h94/IP-Adapter`'s settings.
    ///
    /// One tensor pins the shortest-edge rule, the crop offset, the rescale,
    /// the per-channel normalization and the planar layout at once — the
    /// probes span all three channel planes (0, 50176 and 100352 are the first
    /// element of each).
    #[test]
    fn preprocessing_matches_the_clip_image_processor() {
        let source = synthetic(32, 48);
        let tensor = preprocess_rgb8(&source, 32, 48, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, CHANNELS, CROP_SIZE, CROP_SIZE]);
        let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (index, expected) in [
            (0_usize, -1.018_546_f32),
            (1, -1.003_947_5),
            (223, -1.500_294),
            (224, -1.018_546),
            (5000, 0.149_328_26),
            (50175, -0.142_640_35),
            (50176, -0.911_662_1),
            (100352, -0.299_954_27),
            (120000, 1.690_855),
            (150527, 0.254_628_27),
        ] {
            let actual = flat[index];
            assert!(
                (actual - expected).abs() < 1e-5,
                "probe {index}: {actual} vs {expected}"
            );
        }
    }

    /// A non-square input is CROPPED, not stretched.
    ///
    /// The source is 112 tall and 448 wide with three flat vertical bands, so
    /// the resize takes it to 224 x 896 and the crop keeps `[336, 560)` — the
    /// middle band only. A stretch would have kept the outer bands and put
    /// them at the edges, which is what makes the two distinguishable from the
    /// output alone.
    #[test]
    fn a_non_square_input_is_cropped_rather_than_stretched() {
        const HEIGHT: usize = 112;
        const WIDTH: usize = 448;
        let bands = [0_u8, 128, 255];
        let mut source = vec![0_u8; HEIGHT * WIDTH * CHANNELS];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let value = bands[(x * 3) / WIDTH];
                for channel in 0..CHANNELS {
                    source[(y * WIDTH + x) * CHANNELS + channel] = value;
                }
            }
        }
        let tensor = preprocess_rgb8(&source, HEIGHT, WIDTH, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, CHANNELS, CROP_SIZE, CROP_SIZE]);

        // Only the middle band survives, so every column of the red plane is
        // the normalized value of 128 and none of them is the normalized 0 or
        // 255 the outer bands would contribute.
        let row = tensor.i((0, 0, 100, ..)).unwrap().to_vec1::<f32>().unwrap();
        let middle = (128.0 / 255.0 - OPENAI_CLIP_MEAN[0]) / OPENAI_CLIP_STD[0];
        let black = (0.0 - OPENAI_CLIP_MEAN[0]) / OPENAI_CLIP_STD[0];
        let white = (1.0 - OPENAI_CLIP_MEAN[0]) / OPENAI_CLIP_STD[0];
        for (column, value) in row.iter().enumerate() {
            assert!(
                (value - middle).abs() < 1e-4,
                "column {column} is {value}, expected the middle band {middle}"
            );
        }
        // ...and the assertion above is only meaningful because a stretch
        // would have been visibly different.
        assert!((black - middle).abs() > 1.0 && (white - middle).abs() > 1.0);

        // The crop window itself, stated directly.
        let (out_height, out_width) = resize_output_size(HEIGHT, WIDTH, SHORTEST_EDGE);
        assert_eq!((out_height, out_width), (224, 896));
        assert_eq!((out_width - CROP_SIZE) / 2, 336);
    }

    /// A portrait input crops top-and-bottom by the same rule.
    #[test]
    fn a_portrait_input_crops_the_long_axis() {
        let source = synthetic(448, 112);
        let (out_height, out_width) = resize_output_size(448, 112, SHORTEST_EDGE);
        assert_eq!((out_height, out_width), (896, 224));
        assert_eq!((out_height - CROP_SIZE) / 2, 336);
        let tensor = preprocess_rgb8(&source, 448, 112, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, CHANNELS, CROP_SIZE, CROP_SIZE]);
    }

    /// The crop takes the middle, and an odd slack floors rather than rounds
    /// (`image_transforms.py:503-506` is explicit about it).
    #[test]
    fn the_centre_crop_floors_an_odd_slack() {
        // A 3 x 4 image whose every pixel names its own (row, column), so the
        // crop window is readable straight off the output.
        let mut pixels = vec![0_u8; 3 * 4 * CHANNELS];
        for row in 0..3_usize {
            for column in 0..4_usize {
                let offset = (row * 4 + column) * CHANNELS;
                pixels[offset] = row as u8;
                pixels[offset + 1] = column as u8;
                pixels[offset + 2] = 9;
            }
        }
        // Even slack on the rows: top = 0. Odd slack on the columns: the
        // remainder floors, so left = (4 - 3) / 2 = 0 and columns 0, 1, 2
        // survive — NOT 1, 2, 3, which is what rounding up would keep.
        let cropped = center_crop(&pixels, 3, 4, 3).unwrap();
        assert_eq!(
            cropped,
            vec![
                0, 0, 9, 0, 1, 9, 0, 2, 9, // row 0, columns 0..3
                1, 0, 9, 1, 1, 9, 1, 2, 9, // row 1
                2, 0, 9, 2, 1, 9, 2, 2, 9, // row 2
            ]
        );

        // Slack 2 takes the middle: left = 1.
        let mut wide = vec![0_u8; 3 * 5 * CHANNELS];
        for row in 0..3_usize {
            for column in 0..5_usize {
                let offset = (row * 5 + column) * CHANNELS;
                wide[offset] = column as u8;
                wide[offset + 1] = column as u8;
                wide[offset + 2] = column as u8;
            }
        }
        let cropped = center_crop(&wide, 3, 5, 3).unwrap();
        assert_eq!(&cropped[0..9], &[1, 1, 1, 2, 2, 2, 3, 3, 3]);

        // Upstream's zero-padding branch is unreachable here, so a crop that
        // would need it is an error rather than a silently padded frame.
        let error = center_crop(&pixels, 3, 4, 8).unwrap_err().to_string();
        assert!(error.contains("centre crop"), "unexpected error: {error}");
    }

    #[test]
    fn a_mismatched_buffer_is_refused() {
        let error = preprocess_rgb8(&[0; 10], 4, 4, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("interleaved samples"),
            "unexpected error: {error}"
        );
    }

    /// The decode path hands back interleaved RGB at the image's own size and
    /// drops alpha rather than compositing it, matching `convert("RGB")`.
    #[test]
    fn the_decoder_produces_interleaved_rgb() {
        let mut rgba = image::RgbaImage::new(3, 2);
        rgba.put_pixel(0, 0, image::Rgba([10, 20, 30, 0]));
        rgba.put_pixel(2, 1, image::Rgba([40, 50, 60, 255]));
        let (pixels, height, width) = rgb8_from_image(&image::DynamicImage::ImageRgba8(rgba));
        assert_eq!((height, width), (2, 3));
        assert_eq!(pixels.len(), 2 * 3 * CHANNELS);
        assert_eq!(&pixels[0..3], &[10, 20, 30], "a zero alpha must not matte");
        assert_eq!(&pixels[15..18], &[40, 50, 60]);
    }
}
