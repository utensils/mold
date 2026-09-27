//! Pure Rust Pillow-compatible RGB and RGBA preprocessing, shared by model families.
//! Reference: Pillow 12.3.0 bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d,
//! src/libImaging/Resample.c. Attribution is in THIRD_PARTY_NOTICES.md.
use anyhow::{anyhow, bail, ensure, Context, Result};
use image::{RgbImage, RgbaImage};

#[derive(Clone, Copy)]
pub(crate) enum Filter {
    Lanczos,
    Bicubic,
}
impl Filter {
    fn support(self) -> f64 {
        match self {
            Self::Lanczos => 3.,
            Self::Bicubic => 2.,
        }
    }
    fn sample(self, x: f64) -> f64 {
        match self {
            Self::Lanczos => pillow_lanczos(x),
            Self::Bicubic => {
                let x = x.abs();
                if x < 1. {
                    (1.5 * x - 2.5) * x * x + 1.
                } else if x < 2. {
                    (((x - 5.) * x + 8.) * x - 4.) * (-0.5)
                } else {
                    0.
                }
            }
        }
    }
}
fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    let size = values.iter().try_fold(1usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or_else(|| anyhow!("{label} size overflow"))
    })?;
    ensure!(
        size <= 512 * 1024 * 1024,
        "{label} exceeds the 512 MiB buffer budget"
    );
    Ok(size)
}

const PILLOW_RESAMPLE_PRECISION_BITS: u32 = 22;

struct PillowResampleCoefficients {
    start: usize,
    values: Vec<i32>,
}

fn pillow_lanczos(x: f64) -> f64 {
    fn sinc(mut x: f64) -> f64 {
        if x == 0.0 {
            return 1.0;
        }
        x *= std::f64::consts::PI;
        x.sin() / x
    }

    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow's U8 LANCZOS coefficient generation and fixed-point rounding.
///
/// H3's pinned preprocessing authority uses `PIL.Image.Resampling.LANCZOS`.
/// The `image` crate's similarly named Lanczos3 filter uses different edge and
/// quantization rules, which changes the endpoint tensor before seed-42 VAE
/// sampling. See Diffusers
/// `src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:134-158` at
/// `9c6a68c32b3b2a64db91800b624d33cec6e25ab8` and Pillow 12.3.0
/// (`bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d`)
/// `src/libImaging/Resample.c:65-87,183-284,344-363,446-463`. The Pillow
/// attribution and license are preserved in `THIRD_PARTY_NOTICES.md`.
fn pillow_resample_coefficients(
    input: usize,
    output: usize,
    filter: Filter,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<PillowResampleCoefficients>> {
    if input == 0 || output == 0 {
        bail!("Pillow-compatible resize dimensions must be non-zero");
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let coefficient_scale = f64::from(1_u32 << PILLOW_RESAMPLE_PRECISION_BITS);

    (0..output)
        .map(|destination| {
            checkpoint()?;
            let center = (destination as f64 + 0.5) * scale;
            // Match Pillow's C casts, which truncate toward zero before
            // clamping the bounds to the source image.
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as isize).clamp(0, input as isize) as usize;
            if end <= start {
                bail!("Pillow-compatible resize produced an empty filter window");
            }
            let mut weights = (start..end)
                .map(|source| filter.sample((source as f64 - center + 0.5) / filter_scale))
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
            let values = weights
                .into_iter()
                .map(|weight| {
                    let scaled = weight * coefficient_scale;
                    if weight < 0.0 {
                        (scaled - 0.5) as i32
                    } else {
                        (scaled + 0.5) as i32
                    }
                })
                .collect();
            Ok(PillowResampleCoefficients { start, values })
        })
        .collect()
}

fn pillow_resample_channel(samples: impl Iterator<Item = (u8, i32)>) -> u8 {
    let accumulator = samples.fold(
        1_i64 << (PILLOW_RESAMPLE_PRECISION_BITS - 1),
        |total, (sample, coefficient)| total + i64::from(sample) * i64::from(coefficient),
    );
    (accumulator >> PILLOW_RESAMPLE_PRECISION_BITS).clamp(0, 255) as u8
}

pub(crate) fn resize(
    source: &RgbImage,
    width: u32,
    height: u32,
    filter: Filter,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbImage> {
    let output = resample_interleaved(
        source.as_raw(),
        (source.width(), source.height()),
        3,
        (width, height),
        filter,
        checkpoint,
    )?;
    RgbImage::from_raw(width, height, output)
        .ok_or_else(|| anyhow!("Pillow-compatible resize produced an invalid RGB image"))
}

/// Pillow's separable 8-bit resample (`Resample.c`, horizontal pass then
/// vertical) over an interleaved `channels`-band image. Every band is filtered
/// independently with the same fixed-point coefficients, which is what the
/// 3- and 4-band `image32` paths do.
fn resample_interleaved(
    source_bytes: &[u8],
    (source_width, source_height): (u32, u32),
    channels: usize,
    (width, height): (u32, u32),
    filter: Filter,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    ensure!(
        source_width > 0 && source_height > 0 && width > 0 && height > 0,
        "resize dimensions must be nonzero"
    );
    ensure!(
        [source_width, source_height, width, height]
            .into_iter()
            .all(|n| n <= 16384),
        "resize dimensions exceed 16384"
    );
    let source_width = usize::try_from(source_width).context("source width does not fit usize")?;
    let source_height =
        usize::try_from(source_height).context("source height does not fit usize")?;
    let target_width = usize::try_from(width).context("target width does not fit usize")?;
    let target_height = usize::try_from(height).context("target height does not fit usize")?;
    ensure!(
        source_bytes.len() == source_width * source_height * channels,
        "Pillow-compatible resize source has the wrong byte length"
    );

    let horizontal = if source_width == target_width {
        source_bytes.to_vec()
    } else {
        let coefficients =
            pillow_resample_coefficients(source_width, target_width, filter, checkpoint)?;
        let output_len = checked_product(
            &[target_width, source_height, channels],
            "Pillow-compatible horizontal resize",
        )?;
        let mut output = vec![0_u8; output_len];
        for y in 0..source_height {
            checkpoint()?;
            for (x, filter) in coefficients.iter().enumerate() {
                for channel in 0..channels {
                    output[(y * target_width + x) * channels + channel] =
                        pillow_resample_channel(filter.values.iter().enumerate().map(
                            |(offset, &coefficient)| {
                                (
                                    source_bytes[(y * source_width + filter.start + offset)
                                        * channels
                                        + channel],
                                    coefficient,
                                )
                            },
                        ));
                }
            }
        }
        output
    };

    if source_height == target_height {
        return Ok(horizontal);
    }
    let coefficients =
        pillow_resample_coefficients(source_height, target_height, filter, checkpoint)?;
    let output_len = checked_product(
        &[target_width, target_height, channels],
        "Pillow-compatible vertical resize",
    )?;
    let mut output = vec![0_u8; output_len];
    for (y, filter) in coefficients.iter().enumerate() {
        checkpoint()?;
        for x in 0..target_width {
            for channel in 0..channels {
                output[(y * target_width + x) * channels + channel] =
                    pillow_resample_channel(filter.values.iter().enumerate().map(
                        |(offset, &coefficient)| {
                            (
                                horizontal[((filter.start + offset) * target_width + x) * channels
                                    + channel],
                                coefficient,
                            )
                        },
                    ));
            }
        }
    }
    Ok(output)
}

/// `SHIFTFORDIV255` (`libImaging/ImagingUtils.h:17`): an exact `/255` of a
/// value already biased by 128.
fn shift_for_div255(value: u32) -> u32 {
    ((value >> 8) + value) >> 8
}

/// `MULDIV255(a, b)` (`ImagingUtils.h:20`): `a * b / 255`, rounded.
fn mul_div255(a: u8, b: u8) -> u8 {
    shift_for_div255(u32::from(a) * u32::from(b) + 128) as u8
}

/// `Image.resize` on an RGBA image (Pillow 12.3.0 `PIL/Image.py:2401-2410`):
/// an unchanged size returns a copy untouched; otherwise the image is
/// converted to premultiplied `RGBa` (`libImaging/Convert.c:421-431`,
/// `rgbA2rgba`), resampled on all four bands, and converted back
/// (`Convert.c:436-452`, `rgba2rgbA`: alpha 0 or 255 passes the colour
/// through, anything else divides by alpha with integer truncation and
/// clips). Colour under `α = 0` therefore becomes 0, which is what the Qwen
/// Image 2.1 VAE is trained to read.
pub(crate) fn resize_rgba_premultiplied(
    source: &RgbaImage,
    width: u32,
    height: u32,
    filter: Filter,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbaImage> {
    if source.dimensions() == (width, height) {
        return Ok(source.clone());
    }
    let premultiplied: Vec<u8> = source
        .as_raw()
        .chunks_exact(4)
        .flat_map(|pixel| {
            let alpha = pixel[3];
            [
                mul_div255(pixel[0], alpha),
                mul_div255(pixel[1], alpha),
                mul_div255(pixel[2], alpha),
                alpha,
            ]
        })
        .collect();
    let mut resized = resample_interleaved(
        &premultiplied,
        source.dimensions(),
        4,
        (width, height),
        filter,
        checkpoint,
    )?;
    for pixel in resized.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        if alpha != 0 && alpha != 255 {
            for channel in &mut pixel[..3] {
                *channel = (255 * u32::from(*channel) / alpha).min(255) as u8;
            }
        }
    }
    RgbaImage::from_raw(width, height, resized)
        .ok_or_else(|| anyhow!("Pillow-compatible resize produced an invalid RGBA image"))
}

/// `Image.new("RGB", size, white).paste(img, mask=img.getchannel("A"))`:
/// `paste_mask_L` (`libImaging/Paste.c:127-170`) blends every colour byte as
/// `BLEND(a, 255, c) = DIV255(255 * (255 - a) + c * a)`
/// (`ImagingUtils.h:22-24`). This is the copy Qwen Image 2.1 hands its
/// vision tower (`pipeline_qwenimage21.py:266-271`).
///
/// It is also the ONE flatten for an output container with no alpha (a JPEG
/// render that kept alpha, `image::encode_rgba_image`). mold had a second
/// copy there, written as straight-alpha round-to-nearest,
/// `(c * a + 255 * (255 - a) + 127) / 255`; Pillow's `DIV255(x + 128)` is
/// that same function on every `(c, a)` — pinned exhaustively by
/// `pillow_div255_blend_is_round_to_nearest` — so the two were unified rather
/// than kept as look-alikes that could drift apart.
pub(crate) fn composite_over_white(source: &RgbaImage) -> RgbImage {
    let (width, height) = source.dimensions();
    let bytes = source
        .as_raw()
        .chunks_exact(4)
        .flat_map(|pixel| {
            let alpha = u32::from(pixel[3]);
            let blend = |channel: u8| {
                shift_for_div255(255 * (255 - alpha) + u32::from(channel) * alpha + 128) as u8
            };
            [blend(pixel[0]), blend(pixel[1]), blend(pixel[2])]
        })
        .collect();
    RgbImage::from_raw(width, height, bytes).expect("an RGBA image's RGB copy has matching size")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        width: u32,
        height: u32,
        target_width: u32,
        target_height: u32,
        source: Vec<u8>,
        expected: Vec<u8>,
    }

    #[test]
    fn bicubic_matches_executable_pillow_pixels() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../tests/fixtures/hunyuan3d/pillow-bicubic.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let source = RgbImage::from_raw(case.width, case.height, case.source).unwrap();
            let actual = resize(
                &source,
                case.target_width,
                case.target_height,
                Filter::Bicubic,
                &mut || Ok(()),
            )
            .unwrap();
            assert_eq!(actual.into_raw(), case.expected);
        }
    }

    fn fixture_image(
        tensors: &std::collections::HashMap<String, candle_core::Tensor>,
        name: &str,
    ) -> (u32, u32, Vec<u8>) {
        let tensor = &tensors[name];
        let (height, width, _) = tensor.dims3().unwrap();
        (
            width as u32,
            height as u32,
            tensor.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
        )
    }

    /// U8: Pillow 12.3.0 LANCZOS resizes (down / up / odd) of RGBA — which
    /// Pillow premultiplies — of RGB, and of RGB converted to opaque RGBA,
    /// plus their white composites, byte for byte
    /// (`testdata/qwen_image21/pillow_resize.safetensors`).
    #[test]
    fn rgba_lanczos_and_white_composite_match_executable_pillow_bytes() {
        let tensors = candle_core::safetensors::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("testdata/qwen_image21/pillow_resize.safetensors"),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let (width, height, rgba) = fixture_image(&tensors, "src_rgba");
        let src_rgba = RgbaImage::from_raw(width, height, rgba).unwrap();
        let (width, height, rgb) = fixture_image(&tensors, "src_rgb");
        let src_rgb = RgbImage::from_raw(width, height, rgb).unwrap();
        let rgb_as_rgba = image::DynamicImage::ImageRgb8(src_rgb.clone()).to_rgba8();
        for case in ["down", "up", "odd"] {
            let (target_width, target_height, expected) =
                fixture_image(&tensors, &format!("rgba_{case}"));
            let actual = resize_rgba_premultiplied(
                &src_rgba,
                target_width,
                target_height,
                Filter::Lanczos,
                &mut || Ok(()),
            )
            .unwrap();
            assert_eq!(actual.as_raw(), &expected, "rgba_{case}");
            let (_, _, white) = fixture_image(&tensors, &format!("rgba_{case}_white"));
            assert_eq!(
                composite_over_white(&actual).as_raw(),
                &white,
                "white {case}"
            );

            let (_, _, expected) = fixture_image(&tensors, &format!("rgb_{case}"));
            let actual = resize(
                &src_rgb,
                target_width,
                target_height,
                Filter::Lanczos,
                &mut || Ok(()),
            )
            .unwrap();
            assert_eq!(actual.as_raw(), &expected, "rgb_{case}");

            let (_, _, expected) = fixture_image(&tensors, &format!("rgb_as_rgba_{case}"));
            let actual = resize_rgba_premultiplied(
                &rgb_as_rgba,
                target_width,
                target_height,
                Filter::Lanczos,
                &mut || Ok(()),
            )
            .unwrap();
            assert_eq!(actual.as_raw(), &expected, "rgb_as_rgba_{case}");
        }
    }

    #[test]
    fn pillow_div255_blend_is_round_to_nearest() {
        for alpha in 0..=255u8 {
            for colour in 0..=255u8 {
                let composited = composite_over_white(&RgbaImage::from_pixel(
                    1,
                    1,
                    image::Rgba([colour, colour, colour, alpha]),
                ));
                let (a, c) = (u32::from(alpha), u32::from(colour));
                let nearest = ((c * a + 255 * (255 - a) + 127) / 255) as u8;
                assert_eq!(composited.get_pixel(0, 0).0[0], nearest, "c={c} a={a}");
            }
        }
    }

    #[test]
    fn same_size_rgba_is_an_untouched_copy() {
        // Pillow returns `self.copy()` before converting to RGBa, so colour
        // under alpha 0 survives an identity resize.
        let image = RgbaImage::from_raw(2, 1, vec![200, 10, 20, 0, 1, 2, 3, 128]).unwrap();
        let resized =
            resize_rgba_premultiplied(&image, 2, 1, Filter::Lanczos, &mut || Ok(())).unwrap();
        assert_eq!(resized, image);
        assert_eq!(
            composite_over_white(&image).as_raw(),
            &vec![255, 255, 255, 128, 128, 129]
        );
    }
}
