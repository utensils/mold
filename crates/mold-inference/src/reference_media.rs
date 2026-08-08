//! Family-neutral reference-media decoding and normalization helpers.
//!
//! Reference ingress has already opened, bounded, and hashed each regular
//! file. These helpers preserve that authority by copying the open descriptor
//! into private decoder storage; they never reopen a user-controlled path.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::{anyhow, bail, Context, Result};
use image::{ImageReader, RgbImage};
use tempfile::{Builder, NamedTempFile};

use crate::ltx2::{media, DecodedAudio};

const COPY_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(crate) struct DecodedReferenceVideo {
    pub frames: Vec<RgbImage>,
    pub fps: f64,
    pub audio: Option<DecodedAudio>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NormalizedStereoAudio {
    pub sample_rate: u32,
    pub channels: [Vec<f32>; 2],
}

impl NormalizedStereoAudio {
    pub fn samples_per_channel(&self) -> usize {
        self.channels[0].len()
    }
}

pub(crate) fn decode_image_from_open_file(
    file: &File,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbImage> {
    let bytes = read_open_file(file, checkpoint)?;
    checkpoint()?;
    let mut reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("unknown reference image format")?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(mold_core::minimax_h3::MAX_REFERENCE_DIMENSION);
    limits.max_image_height = Some(mold_core::minimax_h3::MAX_REFERENCE_DIMENSION);
    limits.max_alloc = Some(
        mold_core::minimax_h3::MAX_REFERENCE_IMAGE_PIXELS
            .checked_mul(4)
            .context("reference image decode limit overflowed")?,
    );
    reader.limits(limits);
    let image = reader
        .decode()
        .context("reference image decode failed")?
        .to_rgb8();
    checkpoint()?;
    Ok(image)
}

pub(crate) fn decode_video_from_open_file(
    file: &File,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<DecodedReferenceVideo> {
    let staged = stage_open_file(file, ".mp4", checkpoint)?;
    let (metadata, frames) =
        media::decode_video_frames_from_path_with_checkpoint(staged.path(), checkpoint)?;
    checkpoint()?;
    let audio = if metadata.has_audio {
        DecodedAudio::from_mp4_file(staged.path(), None)?
    } else {
        None
    };
    checkpoint()?;
    Ok(DecodedReferenceVideo {
        frames,
        fps: f64::from(metadata.fps),
        audio,
    })
}

pub(crate) fn decode_audio_from_open_file(
    file: &File,
    mime_type: &str,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<DecodedAudio> {
    let suffix = audio_suffix(mime_type);
    let staged = stage_open_file(file, suffix, checkpoint)?;
    checkpoint()?;
    let decoded = DecodedAudio::from_file(staged.path(), None)?
        .ok_or_else(|| anyhow!("reference audio did not contain a decodable audio stream"))?;
    checkpoint()?;
    Ok(decoded)
}

/// Reproduce ffmpeg's CFR source-frame assignment at a fixed output rate.
///
/// Source frame `i` owns destination slots
/// `floor(i * target/source + .5)..floor((i+1) * target/source + .5)`.
/// The caller supplies the already-frozen (and possibly duration-truncated)
/// output count, so preprocessing cannot silently change admission geometry.
#[cfg(test)]
pub(crate) fn normalize_cfr_frames(
    source: &[RgbImage],
    source_fps: f64,
    target_fps: u32,
    output_frames: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<RgbImage>> {
    let indices = cfr_source_indices(
        source.len(),
        source_fps,
        target_fps,
        output_frames,
        checkpoint,
    )?;
    Ok(indices
        .into_iter()
        .map(|index| source[index].clone())
        .collect())
}

pub(crate) fn cfr_source_indices(
    source_frames: usize,
    source_fps: f64,
    target_fps: u32,
    output_frames: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<usize>> {
    if source_frames == 0 || !source_fps.is_finite() || source_fps <= 0.0 || target_fps == 0 {
        bail!("reference CFR normalization requires frames and positive finite rates");
    }
    if output_frames == 0 {
        bail!("reference CFR normalization requires a nonzero output frame count");
    }
    let scale = f64::from(target_fps) / source_fps;
    let terminal = ((source_frames as f64 * scale) + 0.5).floor() as usize;
    if output_frames > terminal {
        bail!(
            "reference CFR normalization requested {output_frames} frames but the decoded timeline produces only {terminal}"
        );
    }

    let mut normalized = Vec::with_capacity(output_frames);
    for index in 0..source_frames {
        checkpoint()?;
        let start = ((index as f64 * scale) + 0.5).floor() as usize;
        let end = ((((index + 1) as f64 * scale) + 0.5).floor() as usize).min(output_frames);
        if end <= start {
            continue;
        }
        for _ in start..end {
            checkpoint()?;
            normalized.push(index);
        }
        if normalized.len() == output_frames {
            break;
        }
    }
    if normalized.len() != output_frames {
        bail!(
            "reference CFR normalization produced {} frames, expected {output_frames}",
            normalized.len()
        );
    }
    Ok(normalized)
}

/// Truncate in the native sample domain, remix every input channel to stereo,
/// then resample exactly once to the frozen output rate and sample count.
pub(crate) fn normalize_audio(
    decoded: &DecodedAudio,
    maximum_native_samples: usize,
    target_sample_rate: u32,
    expected_samples: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<NormalizedStereoAudio> {
    if decoded.sample_rate == 0 || decoded.channels.is_empty() || target_sample_rate == 0 {
        bail!("reference audio normalization requires positive rates and decoded channels");
    }
    let source_samples = decoded.sample_count();
    if source_samples == 0
        || decoded
            .channels
            .iter()
            .any(|channel| channel.len() != source_samples)
    {
        bail!("reference audio channels must have one nonempty, shared sample count");
    }
    let native_samples = source_samples.min(maximum_native_samples);
    if native_samples == 0 || expected_samples == 0 {
        bail!("reference audio normalization resolved an empty duration");
    }
    checkpoint()?;
    let stereo = remix_stereo(&decoded.channels, native_samples, checkpoint)?;
    let channels = if decoded.sample_rate == target_sample_rate as usize
        && native_samples == expected_samples
    {
        stereo
    } else {
        [
            resample_channel(
                &stereo[0],
                decoded.sample_rate,
                target_sample_rate as usize,
                expected_samples,
                checkpoint,
            )?,
            resample_channel(
                &stereo[1],
                decoded.sample_rate,
                target_sample_rate as usize,
                expected_samples,
                checkpoint,
            )?,
        ]
    };
    checkpoint()?;
    Ok(NormalizedStereoAudio {
        sample_rate: target_sample_rate,
        channels,
    })
}

fn remix_stereo(
    channels: &[Vec<f32>],
    sample_count: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<[Vec<f32>; 2]> {
    if channels.len() == 1 {
        let mono = channels[0][..sample_count].to_vec();
        return Ok([mono.clone(), mono]);
    }
    if channels.len() == 2 {
        return Ok([
            channels[0][..sample_count].to_vec(),
            channels[1][..sample_count].to_vec(),
        ]);
    }

    // Preserve the established left/right pair and include every additional
    // channel deterministically. Even-numbered channels feed left, odd feed
    // right; each side is averaged to avoid gain growth for wide layouts.
    let left_count = channels.len().div_ceil(2) as f32;
    let right_count = (channels.len() / 2) as f32;
    let mut left = Vec::with_capacity(sample_count);
    let mut right = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        if sample.is_multiple_of(4_096) {
            checkpoint()?;
        }
        let mut left_sum = 0.0_f32;
        let mut right_sum = 0.0_f32;
        for (channel_index, channel) in channels.iter().enumerate() {
            if channel_index.is_multiple_of(2) {
                left_sum += channel[sample];
            } else {
                right_sum += channel[sample];
            }
        }
        left.push(left_sum / left_count);
        right.push(right_sum / right_count);
    }
    Ok([left, right])
}

fn resample_channel(
    source: &[f32],
    source_rate: usize,
    target_rate: usize,
    expected_samples: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(expected_samples);
    for target_index in 0..expected_samples {
        if target_index.is_multiple_of(4_096) {
            checkpoint()?;
        }
        let source_position = target_index as f64 * source_rate as f64 / target_rate as f64;
        let left = (source_position.floor() as usize).min(source.len() - 1);
        let right = (left + 1).min(source.len() - 1);
        let fraction = (source_position - source_position.floor()) as f32;
        output.push(source[left] + (source[right] - source[left]) * fraction);
    }
    Ok(output)
}

fn read_open_file(file: &File, checkpoint: &mut dyn FnMut() -> Result<()>) -> Result<Vec<u8>> {
    let mut file = file
        .try_clone()
        .context("failed to duplicate opened reference media")?;
    file.seek(SeekFrom::Start(0))
        .context("failed to rewind opened reference media")?;
    let capacity =
        usize::try_from(file.metadata()?.len()).context("reference media is too large")?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut chunk = [0_u8; COPY_CHUNK_BYTES];
    loop {
        checkpoint()?;
        let read = file
            .read(&mut chunk)
            .context("failed to read opened reference media")?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    Ok(bytes)
}

fn stage_open_file(
    file: &File,
    suffix: &str,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<NamedTempFile> {
    let mut source = file
        .try_clone()
        .context("failed to duplicate opened reference media")?;
    source
        .seek(SeekFrom::Start(0))
        .context("failed to rewind opened reference media")?;
    let mut staged = Builder::new()
        .prefix("mold-reference-")
        .suffix(suffix)
        .tempfile()
        .context("failed to create private reference decoder storage")?;
    let mut chunk = [0_u8; COPY_CHUNK_BYTES];
    loop {
        checkpoint()?;
        let read = source
            .read(&mut chunk)
            .context("failed to read opened reference media")?;
        if read == 0 {
            break;
        }
        staged
            .as_file_mut()
            .write_all(&chunk[..read])
            .context("failed to stage opened reference media")?;
    }
    staged
        .as_file_mut()
        .flush()
        .context("failed to flush private reference decoder storage")?;
    checkpoint()?;
    Ok(staged)
}

fn audio_suffix(mime_type: &str) -> &'static str {
    match mime_type.to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => ".wav",
        "audio/flac" | "audio/x-flac" => ".flac",
        "audio/mpeg" | "audio/mp3" => ".mp3",
        "audio/ogg" | "application/ogg" => ".ogg",
        "audio/mp4" | "audio/x-m4a" | "video/mp4" => ".m4a",
        _ => ".audio",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(value: u8) -> RgbImage {
        RgbImage::from_pixel(2, 2, image::Rgb([value, 0, 0]))
    }

    #[test]
    fn cfr_normalization_downsamples_and_upsamples_by_source_assignment() {
        let source = (0..5).map(frame).collect::<Vec<_>>();
        let down = normalize_cfr_frames(&source, 30.0, 24, 4, &mut || Ok(())).unwrap();
        assert_eq!(
            down.iter()
                .map(|frame| frame[(0, 0)][0])
                .collect::<Vec<_>>(),
            vec![0, 1, 3, 4]
        );

        let up = normalize_cfr_frames(&source[..2], 12.0, 24, 4, &mut || Ok(())).unwrap();
        assert_eq!(
            up.iter().map(|frame| frame[(0, 0)][0]).collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
    }

    #[test]
    fn arbitrary_channel_audio_is_truncated_remixed_and_resampled_exactly() {
        let decoded = DecodedAudio {
            sample_rate: 16_000,
            channels: vec![vec![1.0; 8], vec![2.0; 8], vec![3.0; 8], vec![4.0; 8]],
        };
        let normalized = normalize_audio(&decoded, 4, 32_000, 8, &mut || Ok(())).unwrap();
        assert_eq!(normalized.sample_rate, 32_000);
        assert_eq!(normalized.samples_per_channel(), 8);
        assert_eq!(normalized.channels[0], vec![2.0; 8]);
        assert_eq!(normalized.channels[1], vec![3.0; 8]);
    }

    #[test]
    fn long_audio_work_polls_cancellation() {
        let decoded = DecodedAudio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 20_000]; 6],
        };
        let mut polls = 0;
        let error = normalize_audio(&decoded, 20_000, 32_000, 13_334, &mut || {
            polls += 1;
            if polls == 4 {
                Err(anyhow::Error::new(crate::progress::InferenceCancelled))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(polls, 4);
        assert!(crate::progress::is_inference_cancelled(&error));
    }
}
