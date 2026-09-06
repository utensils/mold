//! Pure Rust Pillow-compatible RGB preprocessing, shared by model families.
//! Reference: Pillow 12.3.0 bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d,
//! src/libImaging/Resample.c. Attribution is in THIRD_PARTY_NOTICES.md.
use anyhow::{anyhow, bail, ensure, Context, Result};
use image::RgbImage;

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
    ensure!(
        source.width() > 0 && source.height() > 0 && width > 0 && height > 0,
        "resize dimensions must be nonzero"
    );
    ensure!(
        [source.width(), source.height(), width, height]
            .into_iter()
            .all(|n| n <= 16384),
        "resize dimensions exceed 16384"
    );
    let source_width =
        usize::try_from(source.width()).context("source width does not fit usize")?;
    let source_height =
        usize::try_from(source.height()).context("source height does not fit usize")?;
    let target_width = usize::try_from(width).context("target width does not fit usize")?;
    let target_height = usize::try_from(height).context("target height does not fit usize")?;
    let source_bytes = source.as_raw();

    let horizontal = if source_width == target_width {
        source_bytes.clone()
    } else {
        let coefficients =
            pillow_resample_coefficients(source_width, target_width, filter, checkpoint)?;
        let output_len = checked_product(
            &[target_width, source_height, 3],
            "Pillow-compatible horizontal resize",
        )?;
        let mut output = vec![0_u8; output_len];
        for y in 0..source_height {
            checkpoint()?;
            for (x, filter) in coefficients.iter().enumerate() {
                for channel in 0..3 {
                    output[(y * target_width + x) * 3 + channel] =
                        pillow_resample_channel(filter.values.iter().enumerate().map(
                            |(offset, &coefficient)| {
                                (
                                    source_bytes
                                        [(y * source_width + filter.start + offset) * 3 + channel],
                                    coefficient,
                                )
                            },
                        ));
                }
            }
        }
        output
    };

    let output = if source_height == target_height {
        horizontal
    } else {
        let coefficients =
            pillow_resample_coefficients(source_height, target_height, filter, checkpoint)?;
        let output_len = checked_product(
            &[target_width, target_height, 3],
            "Pillow-compatible vertical resize",
        )?;
        let mut output = vec![0_u8; output_len];
        for (y, filter) in coefficients.iter().enumerate() {
            checkpoint()?;
            for x in 0..target_width {
                for channel in 0..3 {
                    output[(y * target_width + x) * 3 + channel] =
                        pillow_resample_channel(filter.values.iter().enumerate().map(
                            |(offset, &coefficient)| {
                                (
                                    horizontal[((filter.start + offset) * target_width + x) * 3
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

    RgbImage::from_raw(width, height, output)
        .ok_or_else(|| anyhow!("Pillow-compatible resize produced an invalid RGB image"))
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
}
