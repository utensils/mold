//! Image preprocessing for the EVA02-CLIP-L-14-336 tower.
//!
//! Upstream is `ToTheBeginning/PuLID` at commit
//! `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa`, `pulid/pipeline_flux.py`:
//!
//! ```text
//! :161  input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
//! :173  face_features_image = resize(face_features_image,
//!                                    self.clip_vision_model.image_size,
//!                                    InterpolationMode.BICUBIC)
//! :174  face_features_image = normalize(face_features_image,
//!                                       eva_transform_mean, eva_transform_std)
//! ```
//!
//! Three details are load-bearing and none of them is visible from the call
//! site:
//!
//! 1. The resize runs on the **float** tensor in `[0, 1]`, not on `u8` pixels,
//!    and its output is not clamped — a real 512 -> 336 downscale of this
//!    fixture lands at `[-0.0119, 1.0090]`.
//! 2. `torchvision.transforms.functional.resize` defaults to
//!    `antialias=True` for tensors, so the filter support is scaled by the
//!    downsampling ratio. The `image` crate's `FilterType::CatmullRom` is the
//!    same cubic family but a different `a`, so it is not a substitute.
//! 3. That cubic's `a` is **-0.5**, not the -0.75 PyTorch's *non*-antialiased
//!    bicubic uses. Verified against `torchvision` directly: -0.5 reproduces
//!    it to 1.5e-5 in f32, -0.75 is off by 6.4e-2.

// The PuLID pipeline that consumes this module lands with the FLUX
// integration (milestone "PuLID-FLUX: functional"); issue #1229 delivers the
// encoders and their parity coverage on their own. Until that consumer exists
// every item here is reachable only from tests, so the dead-code lint would
// otherwise force either a premature `pub` surface or a stub caller.
#![allow(dead_code)]

use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};

use super::eva_clip_vision::IMAGE_SIZE;

/// `OPENAI_DATASET_MEAN` (`eva_clip/constants.py:1`).
pub(crate) const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// `OPENAI_DATASET_STD` (`eva_clip/constants.py:2`), which reads
/// `(0.26862954, 0.26130258, 0.27577711)`. The last two are written here in
/// their shortest f32-exact form because the extra digits round to the same
/// `f32` and clippy rejects them; the values are unchanged.
pub(crate) const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Keys cubic coefficient. See the module note: this is the antialiased
/// filter's `a`, and it is not the same as PyTorch's plain-bicubic `a`.
const CUBIC_A: f64 = -0.5;
/// Bicubic reaches two source samples either side.
const CUBIC_SUPPORT: f64 = 2.0;

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
struct Taps {
    start: usize,
    weights: Vec<f32>,
}

/// The separable weight table for one axis, matching aten's
/// `_compute_index_ranges_weights`: the support widens by the downscale ratio
/// (that widening *is* the antialiasing), and the weights are renormalized to
/// sum to one so edge windows stay unbiased.
fn axis_taps(input: usize, output: usize) -> Vec<Taps> {
    let scale = input as f64 / output as f64;
    let (support, inverse) = if scale >= 1.0 {
        (CUBIC_SUPPORT * scale, 1.0 / scale)
    } else {
        (CUBIC_SUPPORT, 1.0)
    };
    (0..output)
        .map(|index| {
            let center = scale * (index as f64 + 0.5);
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
            Taps {
                start,
                weights: weights.into_iter().map(|w| w as f32).collect(),
            }
        })
        .collect()
}

/// Antialiased bicubic resize of a planar CHW f32 buffer to `output` square.
fn resize_bicubic(
    pixels: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output: usize,
) -> Vec<f32> {
    let columns = axis_taps(width, output);
    let mut horizontal = vec![0.0_f32; channels * height * output];
    for channel in 0..channels {
        for row in 0..height {
            let source = (channel * height + row) * width;
            let target = (channel * height + row) * output;
            for (index, taps) in columns.iter().enumerate() {
                let mut sum = 0.0_f32;
                for (offset, weight) in taps.weights.iter().enumerate() {
                    sum += pixels[source + taps.start + offset] * weight;
                }
                horizontal[target + index] = sum;
            }
        }
    }

    let rows = axis_taps(height, output);
    let mut resized = vec![0.0_f32; channels * output * output];
    for channel in 0..channels {
        for (index, taps) in rows.iter().enumerate() {
            for column in 0..output {
                let mut sum = 0.0_f32;
                for (offset, weight) in taps.weights.iter().enumerate() {
                    let row = taps.start + offset;
                    sum += horizontal[(channel * height + row) * output + column] * weight;
                }
                resized[(channel * output + index) * output + column] = sum;
            }
        }
    }
    resized
}

/// Resize to 336 square and apply the OpenAI CLIP normalization.
///
/// `pixels` is planar CHW f32 in `[0, 1]` — the shape
/// `pipeline_flux.py:161`'s `img2tensor(...) / 255.0` produces. The result is
/// `[1, 3, 336, 336]` f32 on `device`, ready for
/// [`super::eva_clip_vision::EvaClipVisionTower::forward`].
pub(crate) fn preprocess_planar_rgb(
    pixels: &[f32],
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    const CHANNELS: usize = 3;
    ensure!(
        height > 0 && width > 0 && pixels.len() == CHANNELS * height * width,
        "expected {CHANNELS} x {height} x {width} planar samples, got {}",
        pixels.len()
    );
    let mut resized = resize_bicubic(pixels, CHANNELS, height, width, IMAGE_SIZE);
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    for channel in 0..CHANNELS {
        let (mean, std) = (CLIP_MEAN[channel], CLIP_STD[channel]);
        for value in &mut resized[channel * plane..(channel + 1) * plane] {
            *value = (*value - mean) / std;
        }
    }
    Ok(Tensor::from_vec(
        resized,
        (1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        device,
    )?)
}

/// Decode an sRGB image into the planar `[0, 1]` CHW buffer
/// [`preprocess_planar_rgb`] expects.
pub(crate) fn planar_rgb_from_image(image: &image::DynamicImage) -> (Vec<f32>, usize, usize) {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let mut planar = vec![0.0_f32; 3 * height * width];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let offset = y as usize * width + x as usize;
        for channel in 0..3 {
            planar[channel * height * width + offset] = pixel.0[channel] as f32 / 255.0;
        }
    }
    (planar, height, width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulid_fixtures::{
        golden, max_errors, testdata_dir, DeterministicStream, SEED_IMAGE,
    };
    use candle_core::IndexOp;

    /// These are the OpenAI CLIP statistics verbatim. A transposed or rounded
    /// copy shifts every embedding slightly and nothing else notices.
    #[test]
    fn the_normalization_constants_are_upstreams() {
        assert_eq!(CLIP_MEAN, [0.481_454_66, 0.457_827_5, 0.408_210_73]);
        // The literals below are the f32 values of upstream's
        // `(0.26862954, 0.26130258, 0.27577711)`.
        assert_eq!(CLIP_STD, [0.268_629_54, 0.261_302_6, 0.275_777_1]);
        assert_eq!(CLIP_STD[1], 0.26130258_f64 as f32);
        assert_eq!(CLIP_STD[2], 0.27577711_f64 as f32);
        // Distinct per channel — a single scalar would silently pass a shape
        // test.
        assert_ne!(CLIP_MEAN[0], CLIP_MEAN[2]);
        assert_ne!(CLIP_STD[0], CLIP_STD[2]);
    }

    /// Every output sample must be a convex-ish combination summing to one, or
    /// a flat image would not survive the resize.
    #[test]
    fn resize_weights_sum_to_one_and_preserve_a_flat_image() {
        for (input, output) in [(512_usize, 336_usize), (336, 336), (100, 336)] {
            for taps in axis_taps(input, output) {
                let total: f32 = taps.weights.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "{input}->{output} weights sum to {total}"
                );
                assert!(taps.start + taps.weights.len() <= input, "window overruns");
            }
        }
        let flat = vec![0.25_f32; 3 * 64 * 64];
        let resized = resize_bicubic(&flat, 3, 64, 64, IMAGE_SIZE);
        let (absolute, _) = max_errors(&resized, &vec![0.25_f32; resized.len()]);
        assert!(absolute < 1e-6, "flat image drifted by {absolute}");
    }

    /// Downscaling widens the filter support; upscaling does not. That
    /// asymmetry is the antialiasing, so it gets its own assertion.
    #[test]
    fn downscaling_widens_the_support() {
        let down = axis_taps(512, 336);
        let up = axis_taps(100, 336);
        let widest_down = down.iter().map(|t| t.weights.len()).max().unwrap();
        let widest_up = up.iter().map(|t| t.weights.len()).max().unwrap();
        assert!(widest_down > widest_up, "{widest_down} vs {widest_up}");
        assert_eq!(widest_up, 4, "an upscale reads the usual four taps");
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

    /// Full preprocessing parity against the committed torchvision golden.
    /// Hermetic: the fixture image is committed and no model weights are read.
    #[test]
    fn preprocessing_matches_torchvision() {
        let path = testdata_dir().join("input_pattern.png");
        let image = image::open(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let (planar, height, width) = planar_rgb_from_image(&image);
        assert_eq!((height, width), (512, 512));
        let tensor = preprocess_planar_rgb(&planar, height, width, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, IMAGE_SIZE, IMAGE_SIZE]);

        let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let indices = DeterministicStream::new(SEED_IMAGE ^ 0x1)
            .indices(crate::pulid_fixtures::PROBE_COUNT, flat.len());
        let actual: Vec<f32> = indices.iter().map(|&i| flat[i as usize]).collect();
        let expected = golden("preprocess.probe").to_vec1::<f32>().unwrap();
        let (absolute, _) = max_errors(&actual, &expected);
        assert!(absolute < 1e-4, "preprocess probe drifted by {absolute}");

        // A whole row of the green channel: a channel swap or an HWC/CHW
        // transpose cannot survive this even though the probe above might.
        let row = tensor.i((0, 1, 168, ..)).unwrap().to_vec1::<f32>().unwrap();
        let expected_row = golden("preprocess.row_g_168").to_vec1::<f32>().unwrap();
        let (absolute, _) = max_errors(&row, &expected_row);
        assert!(absolute < 1e-4, "green row drifted by {absolute}");
    }

    #[test]
    fn a_mismatched_buffer_is_refused() {
        let error = preprocess_planar_rgb(&[0.0; 10], 4, 4, &Device::Cpu).unwrap_err();
        assert!(
            error.to_string().contains("planar samples"),
            "unexpected error: {error}"
        );
    }
}
