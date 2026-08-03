//! OpenEXR output for the LTX-2 HDR pipeline.
//!
//! HDR renders carry scene-referred linear light well outside `[0, 1]`, which
//! no 8-bit container can hold. Upstream writes one EXR per frame with sRGB /
//! Rec.709 primaries so a compositor reads the grade correctly
//! (`packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py:734-766`).
//!
//! Pure Rust via the `exr` crate, which `image` already pulls in — no system
//! OpenEXR, nothing to cross-compile for the CUDA architecture matrix.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use exr::meta::attribute::Chromaticities;
use exr::prelude::*;

/// One frame of scene-referred linear RGB.
#[derive(Debug)]
pub(crate) struct HdrFrame {
    pub width: usize,
    pub height: usize,
    /// Interleaved RGB, `width * height * 3` samples. May be negative: LogC3's
    /// toe decodes pure black below zero, and that is written verbatim.
    pub rgb: Vec<f32>,
}

impl HdrFrame {
    fn sample(&self, x: usize, y: usize) -> (f32, f32, f32) {
        let base = (y * self.width + x) * 3;
        (self.rgb[base], self.rgb[base + 1], self.rgb[base + 2])
    }
}

/// Sample precision for the written file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExrPrecision {
    /// 16-bit float. Upstream's default: half the size, and still far more
    /// range than the grade needs.
    Half,
    Full,
}

/// sRGB / Rec.709 primaries with a D65 white point, matching upstream's
/// `chromaticities` attribute (`media_io.py:752`). Without it a compositor
/// has to guess the gamut, and the grade shifts.
fn srgb_chromaticities() -> Chromaticities {
    Chromaticities {
        red: Vec2(0.64, 0.33),
        green: Vec2(0.30, 0.60),
        blue: Vec2(0.15, 0.06),
        white: Vec2(0.3127, 0.3290),
    }
}

/// Write one frame to `path`.
pub(crate) fn write_exr_frame(
    path: &Path,
    frame: &HdrFrame,
    precision: ExrPrecision,
) -> Result<()> {
    let expected = frame.width * frame.height * 3;
    anyhow::ensure!(
        frame.rgb.len() == expected,
        "HDR frame has {} samples, expected {expected} for {}x{}",
        frame.rgb.len(),
        frame.width,
        frame.height
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    // `SpecificChannels::rgb` names the channels R/G/B in that order, which is
    // what every compositor expects to find.
    let channels = match precision {
        ExrPrecision::Half => SpecificChannels::rgb(|Vec2(x, y): Vec2<usize>| {
            let (r, g, b) = frame.sample(x, y);
            (f16::from_f32(r), f16::from_f32(g), f16::from_f32(b))
        }),
        ExrPrecision::Full => {
            // Two arms rather than a boxed closure: the sample type is part of
            // the channel type, so this is where the precision is decided.
            return write_full(path, frame);
        }
    };

    let mut image = Image::from_channels((frame.width, frame.height), channels);
    image.layer_data.encoding.compression = Compression::ZIP16;
    image.attributes.chromaticities = Some(srgb_chromaticities());

    image
        .write()
        .to_file(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn write_full(path: &Path, frame: &HdrFrame) -> Result<()> {
    let channels = SpecificChannels::rgb(|Vec2(x, y): Vec2<usize>| frame.sample(x, y));
    let mut image = Image::from_channels((frame.width, frame.height), channels);
    image.layer_data.encoding.compression = Compression::ZIP16;
    image.attributes.chromaticities = Some(srgb_chromaticities());
    image
        .write()
        .to_file(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Per-frame filename inside an EXR sequence directory: `frame_00000.exr`,
/// zero-padded to five digits (`hdr_ic_lora.py:724`).
pub(crate) fn exr_frame_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("frame_{index:05}.exr"))
}

/// Write a whole sequence, returning the paths written in order.
/// Write a whole sequence at once.
///
/// Test-only: production streams frames out during decode via
/// `write_exr_frame`, because buffering a clip's worth of `f32` samples costs
/// gigabytes at LTX-2's larger shapes. This keeps the round-trip tests
/// readable without reintroducing that buffer on a real render.
#[cfg(test)]
pub(crate) fn write_exr_sequence(
    dir: &Path,
    frames: &[HdrFrame],
    precision: ExrPrecision,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let mut written = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let path = exr_frame_path(dir, index);
        write_exr_frame(&path, frame, precision)?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        width: usize,
        height: usize,
        fill: impl Fn(usize, usize) -> (f32, f32, f32),
    ) -> HdrFrame {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = fill(x, y);
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
        HdrFrame { width, height, rgb }
    }

    /// The values that make HDR *HDR* are the ones outside `[0, 1]`: highlights
    /// above 1.0 and LogC3's negative toe. Both must survive the round trip, or
    /// the export is just an expensive LDR file.
    #[test]
    fn out_of_range_values_survive_a_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.exr");
        let source = frame(4, 4, |x, _| match x {
            0 => (-0.017_289_9, -0.017_289_9, -0.017_289_9), // LogC3 black
            1 => (0.0, 0.5, 1.0),
            2 => (12.5, 30.0, 55.1), // specular highlights
            _ => (0.18, 0.18, 0.18), // mid grey
        });

        write_exr_frame(&path, &source, ExrPrecision::Full).unwrap();
        let read = read_first_rgba_layer_from_file(
            &path,
            |resolution, _| {
                vec![(0f32, 0f32, 0f32, 0f32); resolution.width() * resolution.height()]
            },
            |pixels, position, (r, g, b, a): (f32, f32, f32, f32)| {
                pixels[position.y() * 4 + position.x()] = (r, g, b, a);
            },
        )
        .unwrap();

        let pixels = &read.layer_data.channel_data.pixels;
        assert!(pixels[0].0 < 0.0, "negative toe was clamped away");
        assert!((pixels[0].0 - (-0.017_289_9)).abs() < 1e-6);
        assert!((pixels[2].2 - 55.1).abs() < 1e-4, "highlight was clipped");
        assert!((pixels[3].0 - 0.18).abs() < 1e-6);
    }

    /// Half is the default because it keeps the *range* — the thing HDR is
    /// for — at half the sample width. It is not automatically a smaller file:
    /// on smooth data ZIP16 can compress f32 better than f16, so this asserts
    /// the sample type and the surviving range rather than a byte count.
    #[test]
    fn half_precision_keeps_the_hdr_range() {
        let dir = tempfile::tempdir().unwrap();
        let half = dir.path().join("half.exr");
        let source = frame(32, 32, |x, y| {
            (x as f32 * 0.5, y as f32 * 0.25, (x + y) as f32 * 0.125)
        });

        write_exr_frame(&half, &source, ExrPrecision::Half).unwrap();

        let meta = MetaData::read_from_file(&half, false).unwrap();
        let sample_types: Vec<_> = meta.headers[0]
            .channels
            .list
            .iter()
            .map(|channel| channel.sample_type)
            .collect();
        assert!(
            sample_types.iter().all(|t| *t == SampleType::F16),
            "half output must write 16-bit samples, got {sample_types:?}"
        );

        // Half still carries the range; it just carries less precision.
        let read = read_first_rgba_layer_from_file(
            &half,
            |resolution, _| {
                vec![(0f32, 0f32, 0f32, 0f32); resolution.width() * resolution.height()]
            },
            |pixels, position, (r, g, b, a): (f32, f32, f32, f32)| {
                pixels[position.y() * 32 + position.x()] = (r, g, b, a);
            },
        )
        .unwrap();
        let last = read.layer_data.channel_data.pixels[31 * 32 + 31];
        assert!((last.0 - 15.5).abs() < 0.05);
    }

    /// Compositors read the gamut from this attribute; without it they guess.
    #[test]
    fn written_files_declare_srgb_primaries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chroma.exr");
        write_exr_frame(
            &path,
            &frame(2, 2, |_, _| (0.5, 0.5, 0.5)),
            ExrPrecision::Half,
        )
        .unwrap();

        let meta = MetaData::read_from_file(&path, false).unwrap();
        let chromaticities = meta
            .headers
            .first()
            .and_then(|header| header.shared_attributes.chromaticities)
            .expect("chromaticities must be written");
        assert!((chromaticities.red.0 - 0.64).abs() < 1e-6);
        assert!((chromaticities.white.0 - 0.3127).abs() < 1e-6);
        assert!((chromaticities.white.1 - 0.3290).abs() < 1e-6);
    }

    #[test]
    fn sequence_filenames_are_zero_padded_to_five_digits() {
        let dir = Path::new("/tmp/seq");
        assert_eq!(exr_frame_path(dir, 0), dir.join("frame_00000.exr"));
        assert_eq!(exr_frame_path(dir, 42), dir.join("frame_00042.exr"));
        assert_eq!(exr_frame_path(dir, 160), dir.join("frame_00160.exr"));
    }

    #[test]
    fn a_sequence_writes_every_frame_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("shot_exr");
        let frames: Vec<_> = (0..3)
            .map(|i| frame(2, 2, move |_, _| (i as f32, 0.0, 0.0)))
            .collect();

        let written = write_exr_sequence(&out, &frames, ExrPrecision::Half).unwrap();
        assert_eq!(written.len(), 3);
        for (index, path) in written.iter().enumerate() {
            assert_eq!(path, &exr_frame_path(&out, index));
            assert!(path.is_file(), "{} was not written", path.display());
        }
    }

    #[test]
    fn a_mis_sized_frame_is_rejected_rather_than_written_short() {
        let dir = tempfile::tempdir().unwrap();
        let bad = HdrFrame {
            width: 4,
            height: 4,
            rgb: vec![0.0; 10],
        };
        let err = write_exr_frame(&dir.path().join("bad.exr"), &bad, ExrPrecision::Half)
            .expect_err("a short buffer must not be written");
        assert!(format!("{err}").contains("expected 48"), "got: {err}");
    }
}
