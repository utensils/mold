//! Family-neutral reference-media decoding and normalization helpers.
//!
//! Reference ingress has already opened, bounded, and hashed each regular
//! file. These helpers preserve that authority by copying the open descriptor
//! into private decoder storage; they never reopen a user-controlled path.

use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::{anyhow, bail, Context, Result};
use image::{ImageReader, RgbImage};
use sha2::{Digest, Sha256};
use tempfile::{Builder, NamedTempFile};

use crate::engine::GenerationReferenceBinding;
use crate::ltx2::{media, DecodedAudio};

const COPY_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(crate) struct DecodedReferenceVideo {
    pub frames: Vec<RgbImage>,
    pub source_width: u32,
    pub source_height: u32,
    pub fps: f64,
    pub source_frame_count: u32,
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

pub(crate) fn decode_image_from_binding(
    binding: &GenerationReferenceBinding,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbImage> {
    let bytes = read_verified_binding(binding, checkpoint)?;
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

pub(crate) fn decode_video_from_binding(
    binding: &GenerationReferenceBinding,
    selected_source_indices: &[usize],
    frame_transform: &mut dyn FnMut(RgbImage) -> Result<RgbImage>,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<DecodedReferenceVideo> {
    let staged = stage_verified_binding(binding, ".mp4", checkpoint)?;
    let (metadata, frames) = media::decode_video_frames_selected_from_path_with_checkpoint(
        staged.path(),
        selected_source_indices,
        frame_transform,
        checkpoint,
    )?;
    checkpoint()?;
    let audio = if metadata.has_audio {
        DecodedAudio::from_mp4_file_with_checkpoint(staged.path(), None, checkpoint)?
    } else {
        None
    };
    checkpoint()?;
    Ok(DecodedReferenceVideo {
        frames,
        source_width: metadata.width,
        source_height: metadata.height,
        fps: f64::from(metadata.fps),
        source_frame_count: metadata
            .frames
            .context("decoded reference video lost its source frame count")?,
        audio,
    })
}

pub(crate) fn decode_audio_from_binding(
    binding: &GenerationReferenceBinding,
    mime_type: &str,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<DecodedAudio> {
    let suffix = audio_suffix(mime_type);
    let staged = stage_verified_binding(binding, suffix, checkpoint)?;
    checkpoint()?;
    let decoded = DecodedAudio::from_file_with_checkpoint(staged.path(), None, checkpoint)?
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

/// Truncate in the native sample domain, upmix mono to stereo, then resample
/// exactly once to the frozen output rate and sample count.
pub(crate) fn normalize_audio(
    decoded: &DecodedAudio,
    maximum_native_samples: usize,
    target_sample_rate: u32,
    expected_samples: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<NormalizedStereoAudio> {
    if decoded.sample_rate == 0
        || !matches!(decoded.channels.len(), 1 | 2)
        || target_sample_rate == 0
    {
        bail!("reference audio normalization requires positive rates and mono/stereo audio");
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
    let stereo = match decoded.channels.as_slice() {
        [mono] => {
            let mono = mono[..native_samples].to_vec();
            [mono.clone(), mono]
        }
        [left, right] => [
            left[..native_samples].to_vec(),
            right[..native_samples].to_vec(),
        ],
        _ => unreachable!("channel count was validated above"),
    };
    let channels = if decoded.sample_rate == target_sample_rate as usize
        && native_samples == expected_samples
    {
        stereo
    } else {
        [
            torchaudio_hann_resample_channel(
                &stereo[0],
                decoded.sample_rate,
                target_sample_rate as usize,
                expected_samples,
                checkpoint,
            )?,
            torchaudio_hann_resample_channel(
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

/// Pure-Rust expression of torchaudio's default `sinc_interp_hann` resampler.
///
/// H3's pinned Diffusers setup calls `torchaudio.transforms.Resample` with its
/// defaults. Keep the same reduced-rate polyphase kernel, six-zero-crossing
/// Hann window, 0.99 rolloff, zero padding, and ceiling output length. The
/// reference is torchaudio 2.8.0 `functional.py`; attribution is preserved in
/// `THIRD_PARTY_NOTICES.md`.
fn torchaudio_hann_resample_channel(
    source: &[f32],
    source_rate: usize,
    target_rate: usize,
    expected_samples: usize,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<f32>> {
    const LOWPASS_FILTER_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;

    if source.is_empty() || source_rate == 0 || target_rate == 0 {
        bail!("reference audio resampling requires samples and positive rates");
    }
    let divisor = greatest_common_divisor(source_rate, target_rate);
    let source_rate = source_rate / divisor;
    let target_rate = target_rate / divisor;
    let calculated_samples = source
        .len()
        .checked_mul(target_rate)
        .context("reference audio resample length overflowed")?
        .div_ceil(source_rate);
    if calculated_samples != expected_samples {
        bail!(
            "reference audio resample length changed: calculated {calculated_samples}, expected {expected_samples}"
        );
    }

    let base_rate = source_rate.min(target_rate) as f64 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * source_rate as f64 / base_rate).ceil() as usize;
    let kernel_len = width
        .checked_mul(2)
        .and_then(|length| length.checked_add(source_rate))
        .context("reference audio resample kernel length overflowed")?;
    let scale = base_rate / source_rate as f64;
    let mut kernels = Vec::with_capacity(target_rate);
    for phase in 0..target_rate {
        checkpoint()?;
        let mut kernel = Vec::with_capacity(kernel_len);
        for offset in 0..kernel_len {
            let index = offset as f64 - width as f64;
            let mut time =
                (-(phase as f64) / target_rate as f64 + index / source_rate as f64) * base_rate;
            time = time.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (time * std::f64::consts::PI / LOWPASS_FILTER_WIDTH / 2.0)
                .cos()
                .powi(2);
            let angle = time * std::f64::consts::PI;
            let sinc = if angle == 0.0 {
                1.0
            } else {
                angle.sin() / angle
            };
            kernel.push((sinc * window * scale) as f32);
        }
        kernels.push(kernel);
    }

    let mut output = Vec::with_capacity(expected_samples);
    for output_index in 0..expected_samples {
        if output_index.is_multiple_of(4_096) {
            checkpoint()?;
        }
        let frame = output_index / target_rate;
        let phase = output_index % target_rate;
        let input_start = frame
            .checked_mul(source_rate)
            .context("reference audio resample cursor overflowed")?;
        let mut sample = 0.0_f32;
        for (kernel_index, &coefficient) in kernels[phase].iter().enumerate() {
            let padded_index = input_start
                .checked_add(kernel_index)
                .context("reference audio resample index overflowed")?;
            let Some(source_index) = padded_index.checked_sub(width) else {
                continue;
            };
            if source_index >= source.len() {
                continue;
            }
            sample += source[source_index] * coefficient;
        }
        output.push(sample);
    }
    Ok(output)
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn read_verified_binding(
    binding: &GenerationReferenceBinding,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    let expected_bytes = binding.verified_bytes();
    let capacity = usize::try_from(expected_bytes).context("reference media is too large")?;
    let mut bytes = Vec::with_capacity(capacity);
    copy_and_verify_binding(binding, &mut bytes, checkpoint)?;
    Ok(bytes)
}

fn copy_and_verify_binding(
    binding: &GenerationReferenceBinding,
    destination: &mut dyn Write,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    let expected_bytes = binding.verified_bytes();
    let mut file = binding
        .file()
        .try_clone()
        .context("failed to duplicate opened reference media")?;
    file.seek(SeekFrom::Start(0))
        .context("failed to rewind opened reference media")?;
    let mut chunk = [0_u8; COPY_CHUNK_BYTES];
    let mut digest = Sha256::new();
    let mut observed_bytes = 0_u64;
    loop {
        checkpoint()?;
        let read = file
            .read(&mut chunk)
            .context("failed to read opened reference media")?;
        if read == 0 {
            break;
        }
        observed_bytes = observed_bytes
            .checked_add(u64::try_from(read)?)
            .context("reference media byte count overflowed")?;
        if observed_bytes > expected_bytes {
            bail!("opened reference media grew after digest verification");
        }
        digest.update(&chunk[..read]);
        destination
            .write_all(&chunk[..read])
            .context("failed to copy opened reference media")?;
    }
    if observed_bytes != expected_bytes {
        bail!("opened reference media changed size after digest verification");
    }
    let observed_digest = format!("{:x}", digest.finalize());
    if !observed_digest.eq_ignore_ascii_case(&binding.metadata().sha256) {
        bail!("opened reference media changed after digest verification");
    }
    Ok(())
}

fn stage_verified_binding(
    binding: &GenerationReferenceBinding,
    suffix: &str,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<NamedTempFile> {
    let mut staged = Builder::new()
        .prefix("mold-reference-")
        .suffix(suffix)
        .tempfile()
        .context("failed to create private reference decoder storage")?;
    copy_and_verify_binding(binding, staged.as_file_mut(), checkpoint)?;
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
    use mold_core::{GenerationReferenceKind, GenerationReferenceMetadata};

    fn binding(bytes: &[u8]) -> (NamedTempFile, GenerationReferenceBinding) {
        let mut staged = NamedTempFile::new().unwrap();
        staged.write_all(bytes).unwrap();
        staged.flush().unwrap();
        let metadata = GenerationReferenceMetadata {
            kind: GenerationReferenceKind::Image,
            index: 1,
            name: Some("reference.png".into()),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            mime_type: "image/png".into(),
            width: Some(1),
            height: Some(1),
            frame_count: None,
            duration_ms: None,
            fps: None,
            has_audio: false,
            audio_duration_ms: None,
            audio_sample_count: None,
            audio_sample_rate: None,
            audio_channels: None,
            sample_rate: None,
            channels: None,
            sample_count: None,
            prepared_shape: None,
        };
        let opened = staged.reopen().unwrap();
        let binding = GenerationReferenceBinding::from_opened_file(
            metadata,
            opened,
            u64::try_from(bytes.len()).unwrap(),
            None,
        )
        .unwrap();
        (staged, binding)
    }

    fn frame(value: u8) -> RgbImage {
        RgbImage::from_pixel(2, 2, image::Rgb([value, 0, 0]))
    }

    #[test]
    fn decoder_copy_rejects_same_length_inode_mutation_after_admission() {
        let (mut staged, binding) = binding(b"original");
        staged.as_file_mut().seek(SeekFrom::Start(0)).unwrap();
        staged.as_file_mut().write_all(b"tampered").unwrap();
        staged.as_file_mut().flush().unwrap();

        let error = read_verified_binding(&binding, &mut || Ok(())).unwrap_err();
        assert!(error.to_string().contains("changed after digest"));
    }

    #[test]
    fn decoder_copy_rejects_inode_growth_after_admission() {
        let (mut staged, binding) = binding(b"original");
        staged.as_file_mut().seek(SeekFrom::End(0)).unwrap();
        staged.as_file_mut().write_all(b"-growth").unwrap();
        staged.as_file_mut().flush().unwrap();

        let error = read_verified_binding(&binding, &mut || Ok(())).unwrap_err();
        assert!(error.to_string().contains("grew after digest"));
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
    fn mono_audio_is_truncated_upmixed_and_hann_resampled_exactly() {
        let decoded = DecodedAudio {
            sample_rate: 8_000,
            channels: vec![vec![0.0, 1.0, 0.0, -1.0]],
        };
        let normalized = normalize_audio(&decoded, 4, 16_000, 8, &mut || Ok(())).unwrap();
        assert_eq!(normalized.sample_rate, 16_000);
        assert_eq!(normalized.samples_per_channel(), 8);
        assert_eq!(normalized.channels[0], normalized.channels[1]);
        let expected = [
            0.004_270_599,
            0.545_218_17,
            0.997_540_24,
            0.807_425_9,
            0.0,
            -0.807_425_9,
            -0.997_540_24,
            -0.545_218_17,
        ];
        for (actual, expected) in normalized.channels[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn audio_with_more_than_two_channels_fails_closed() {
        let decoded = DecodedAudio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 8]; 3],
        };
        let error = normalize_audio(&decoded, 8, 32_000, 16, &mut || Ok(())).unwrap_err();
        assert!(error.to_string().contains("mono/stereo"));
    }

    #[test]
    fn long_audio_work_polls_cancellation() {
        let decoded = DecodedAudio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 20_000]; 2],
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
