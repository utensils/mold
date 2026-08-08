//! Family-neutral audio/video container helpers.
//!
//! Model pipelines produce decoded interleaved PCM and an H.264-only MP4.
//! This module owns the shared AAC encode/mux boundary, including exact
//! rational timeline alignment and cancellation-safe publication. It must not
//! contain model-family policy or model-specific error messages.

use anyhow::{bail, Context, Result};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;

#[cfg(feature = "mp4")]
use anyhow::anyhow;
#[cfg(feature = "mp4")]
use fdk_aac::enc::{
    AudioObjectType as FdkAudioObjectType, BitRate as FdkBitRate, ChannelMode as FdkChannelMode,
    Encoder as FdkEncoder, EncoderParams as FdkEncoderParams, Transport as FdkTransport,
};
#[cfg(feature = "mp4")]
use mp4_rs::{
    AacConfig, AudioObjectType, AvcConfig, Bytes, ChannelConfig, MediaConfig, MediaType, Mp4Config,
    Mp4Reader, Mp4Sample, Mp4Writer, SampleFreqIndex, TrackConfig, TrackType,
};
#[cfg(feature = "mp4")]
use std::io::Cursor;

/// How decoded PCM is conformed before AAC encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTimelinePolicy {
    /// Preserve the supplied PCM and declare every encoded AAC packet at the
    /// codec's full frame length, including the zero-padded final packet.
    ///
    /// This is the legacy LTX-2 container contract retained by the existing
    /// compatibility wrappers and call sites.
    Preserve,
    /// Trim or zero-pad PCM to the exact duration declared by the video track.
    ///
    /// The target is computed once with integer rational arithmetic, so a
    /// non-integral samples-per-frame ratio never accumulates per-frame drift.
    MatchVideo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AacMuxOptions {
    pub timeline: AudioTimelinePolicy,
}

impl AacMuxOptions {
    /// Preserve the pre-extraction LTX-2 mux contract.
    pub const fn legacy_compatible() -> Self {
        Self {
            timeline: AudioTimelinePolicy::Preserve,
        }
    }

    /// Conform PCM and its MP4 timeline to the exact video duration.
    pub const fn match_video() -> Self {
        Self {
            timeline: AudioTimelinePolicy::MatchVideo,
        }
    }
}

/// Observable points where a caller may stop an in-progress mux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvMuxPhase {
    Validate,
    CopyVideo,
    EncodeAudio,
    Finalize,
    Commit,
}

impl fmt::Display for AvMuxPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Validate => "validation",
            Self::CopyVideo => "video copy",
            Self::EncodeAudio => "audio encode",
            Self::Finalize => "container finalization",
            Self::Commit => "output commit",
        })
    }
}

/// Stable typed cancellation signal for callers that distinguish cancellation
/// from a failed encode.
#[derive(Debug, thiserror::Error)]
#[error("A/V mux cancelled during {phase}")]
pub struct AvMuxCancelled {
    pub phase: AvMuxPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AacMuxReport {
    pub sample_rate: u32,
    pub channels: u16,
    pub source_samples_per_channel: u64,
    pub output_samples_per_channel: u64,
    pub padded_samples_per_channel: u64,
    pub trimmed_samples_per_channel: u64,
    pub video_duration_ticks: u64,
    pub video_timescale: u32,
}

#[derive(Debug)]
pub struct MuxedMp4 {
    pub bytes: Vec<u8>,
    pub report: AacMuxReport,
}

/// Round a media timeline to the nearest whole PCM sample without accumulating
/// a per-frame error.
///
/// The returned sample count differs from the exact rational duration by at
/// most half a sample. All arithmetic is checked and performed in `u128` so
/// long clips cannot overflow intermediate multiplication.
pub fn aligned_sample_count(duration_ticks: u64, timescale: u32, sample_rate: u32) -> Result<u64> {
    if duration_ticks == 0 {
        bail!("A/V timeline duration must be greater than zero");
    }
    if timescale == 0 {
        bail!("A/V timeline timescale must be greater than zero");
    }
    if sample_rate == 0 {
        bail!("PCM sample rate must be greater than zero");
    }
    let denominator = u128::from(timescale);
    let numerator = u128::from(duration_ticks)
        .checked_mul(u128::from(sample_rate))
        .context("A/V timeline sample-count multiplication overflowed")?;
    let rounded = numerator
        .checked_add(denominator / 2)
        .context("A/V timeline sample-count rounding overflowed")?
        / denominator;
    u64::try_from(rounded).context("A/V timeline sample count exceeded u64")
}

/// Compatibility wrapper used by existing pipelines. New pipelines that need
/// cancellation telemetry should call [`mux_aac_track_to_mp4_bytes`].
pub fn attach_aac_track_to_mp4_bytes(
    video_only_mp4: &[u8],
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>> {
    Ok(mux_aac_track_to_mp4_bytes(
        video_only_mp4,
        samples,
        sample_rate,
        channels,
        AacMuxOptions::legacy_compatible(),
        &|_| false,
    )?
    .bytes)
}

/// Compatibility path wrapper. Output publication is atomic within the target
/// directory; cancellation or any encode failure removes the staged tempfile
/// and leaves an existing destination untouched.
pub fn attach_aac_track_from_f32_interleaved(
    input_mp4: &Path,
    out_path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    mux_aac_track_from_f32_interleaved(
        input_mp4,
        out_path,
        samples,
        sample_rate,
        channels,
        AacMuxOptions::legacy_compatible(),
        &|_| false,
    )?;
    Ok(())
}

/// Encode decoded PCM as AAC and attach it to an in-memory H.264 MP4.
///
/// `cancelled` is sampled at bounded container and codec boundaries. A caller
/// may map every phase to one job cancellation token or record the phase for
/// telemetry and tests.
#[cfg(feature = "mp4")]
pub fn mux_aac_track_to_mp4_bytes(
    video_only_mp4: &[u8],
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    options: AacMuxOptions,
    cancelled: &(dyn Fn(AvMuxPhase) -> bool + Send + Sync),
) -> Result<MuxedMp4> {
    check_cancelled(cancelled, AvMuxPhase::Validate)?;
    validate_pcm(samples, sample_rate, channels)?;
    let mut reader =
        Mp4Reader::read_header(Cursor::new(video_only_mp4), video_only_mp4.len() as u64)
            .context("failed to parse MP4 container for A/V mux")?;
    let video = find_h264_track(&reader)?;
    let source_samples_per_channel = u64::try_from(samples.len() / usize::from(channels))
        .context("PCM sample count exceeded u64")?;
    let output_samples_per_channel = match options.timeline {
        AudioTimelinePolicy::MatchVideo => {
            aligned_sample_count(video.duration_ticks, video.timescale, sample_rate)?
        }
        AudioTimelinePolicy::Preserve => source_samples_per_channel,
    };
    if output_samples_per_channel == 0 {
        bail!("A/V mux audio timeline resolved to zero samples");
    }
    let report = AacMuxReport {
        sample_rate,
        channels,
        source_samples_per_channel,
        output_samples_per_channel,
        padded_samples_per_channel: output_samples_per_channel
            .saturating_sub(source_samples_per_channel),
        trimmed_samples_per_channel: source_samples_per_channel
            .saturating_sub(output_samples_per_channel),
        video_duration_ticks: video.duration_ticks,
        video_timescale: video.timescale,
    };

    tracing::debug!(
        sample_rate,
        channels,
        source_samples_per_channel,
        output_samples_per_channel,
        padded_samples_per_channel = report.padded_samples_per_channel,
        trimmed_samples_per_channel = report.trimmed_samples_per_channel,
        "encoding synchronized AAC track"
    );

    let pcm = aligned_pcm_i16(samples, channels, output_samples_per_channel)?;
    let bitrate = recommended_aac_bitrate(sample_rate, channels);
    let encoder = FdkEncoder::new(FdkEncoderParams {
        bit_rate: FdkBitRate::Cbr(bitrate),
        sample_rate,
        transport: FdkTransport::Raw,
        channels: fdk_channel_mode(channels)?,
        audio_object_type: FdkAudioObjectType::Mpeg4LowComplexity,
    })
    .map_err(|error| anyhow!("failed to create AAC encoder: {error}"))?;
    let encoder_info = encoder
        .info()
        .map_err(|error| anyhow!("failed to query AAC encoder: {error}"))?;
    if encoder_info.inputChannels != u32::from(channels) {
        bail!(
            "AAC encoder configured {} input channels for {channels}-channel PCM",
            encoder_info.inputChannels
        );
    }

    let mut writer = Mp4Writer::write_start(Cursor::new(Vec::new()), &mp4_config()?)
        .context("failed to start MP4 writer for A/V mux")?;
    writer
        .add_track(&video_track_config(&video))
        .context("failed to add video track to A/V mux output")?;
    writer
        .add_track(&TrackConfig {
            track_type: TrackType::Audio,
            timescale: sample_rate,
            language: "und".to_string(),
            media_conf: MediaConfig::AacConfig(AacConfig {
                bitrate,
                profile: AudioObjectType::AacLowComplexity,
                freq_index: sample_freq_index(sample_rate)?,
                chan_conf: channel_config(channels)?,
            }),
        })
        .context("failed to add AAC track to MP4")?;

    for sample_id in 1..=video.sample_count {
        check_cancelled(cancelled, AvMuxPhase::CopyVideo)?;
        let Some(sample) = reader.read_sample(video.track_id, sample_id)? else {
            continue;
        };
        writer
            .write_sample(1, &sample)
            .with_context(|| format!("failed to copy video sample {sample_id} into A/V MP4"))?;
    }

    let samples_per_channel = usize::try_from(encoder_info.frameLength)
        .context("AAC encoder returned a negative frame length")?;
    let input_channels = usize::try_from(encoder_info.inputChannels)
        .context("AAC encoder returned a negative channel count")?;
    let frame_samples = samples_per_channel
        .checked_mul(input_channels)
        .context("AAC encoder frame size overflowed")?;
    if frame_samples == 0 {
        bail!("AAC encoder returned an empty input frame size");
    }
    let mut out_buf = vec![0u8; encoder_info.maxOutBufBytes as usize];
    let mut start_time = 0u64;
    let mut remaining = output_samples_per_channel;
    let mut offset = 0usize;
    while offset < pcm.len() {
        check_cancelled(cancelled, AvMuxPhase::EncodeAudio)?;
        let end = (offset + frame_samples).min(pcm.len());
        let mut frame = vec![0i16; frame_samples];
        frame[..(end - offset)].copy_from_slice(&pcm[offset..end]);
        let encoded = encoder
            .encode(&frame, &mut out_buf)
            .map_err(|error| anyhow!("failed to encode AAC frame: {error}"))?;
        if encoded.input_consumed != frame_samples {
            bail!(
                "AAC encoder consumed {} of {frame_samples} input samples",
                encoded.input_consumed
            );
        }
        if encoded.output_size == 0 {
            bail!("AAC encoder produced an empty frame");
        }
        let duration = match options.timeline {
            AudioTimelinePolicy::Preserve => samples_per_channel as u64,
            AudioTimelinePolicy::MatchVideo => remaining.min(samples_per_channel as u64),
        };
        let duration =
            u32::try_from(duration).context("AAC sample duration exceeded MP4's u32 range")?;
        writer
            .write_sample(
                2,
                &Mp4Sample {
                    start_time,
                    duration,
                    rendering_offset: 0,
                    is_sync: true,
                    bytes: Bytes::copy_from_slice(&out_buf[..encoded.output_size]),
                },
            )
            .context("failed to write AAC sample")?;
        start_time = start_time
            .checked_add(u64::from(duration))
            .context("AAC start time overflowed")?;
        remaining = remaining.saturating_sub(u64::from(duration));
        offset = end;
    }
    if remaining != 0 {
        bail!("AAC encoder ended with {remaining} timeline samples unwritten");
    }

    check_cancelled(cancelled, AvMuxPhase::Finalize)?;
    writer
        .write_end()
        .context("failed to finalize AAC MP4 output")?;
    let bytes = writer.into_writer().into_inner();
    tracing::debug!(
        audio_samples_per_channel = start_time,
        output_bytes = bytes.len(),
        "synchronized AAC track encoded"
    );
    Ok(MuxedMp4 { bytes, report })
}

#[cfg(not(feature = "mp4"))]
pub fn mux_aac_track_to_mp4_bytes(
    _video_only_mp4: &[u8],
    _samples: &[f32],
    _sample_rate: u32,
    _channels: u16,
    _options: AacMuxOptions,
    _cancelled: &(dyn Fn(AvMuxPhase) -> bool + Send + Sync),
) -> Result<MuxedMp4> {
    bail!("AAC/MP4 muxing requires the 'mp4' feature")
}

/// Path-based cancellable mux with atomic destination publication.
pub fn mux_aac_track_from_f32_interleaved(
    input_mp4: &Path,
    out_path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    options: AacMuxOptions,
    cancelled: &(dyn Fn(AvMuxPhase) -> bool + Send + Sync),
) -> Result<AacMuxReport> {
    let input = fs::read(input_mp4)
        .with_context(|| format!("failed to read A/V mux input {}", input_mp4.display()))?;
    let muxed =
        mux_aac_track_to_mp4_bytes(&input, samples, sample_rate, channels, options, cancelled)?;
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to stage A/V mux output in {}", parent.display()))?;
    staged
        .write_all(&muxed.bytes)
        .with_context(|| format!("failed to stage A/V mux output for {}", out_path.display()))?;
    staged.as_file().sync_all().with_context(|| {
        format!(
            "failed to sync staged A/V mux output for {}",
            out_path.display()
        )
    })?;
    check_cancelled(cancelled, AvMuxPhase::Commit)?;
    staged
        .persist(out_path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to publish A/V mux output {}", out_path.display()))?;
    Ok(muxed.report)
}

fn check_cancelled(
    cancelled: &(dyn Fn(AvMuxPhase) -> bool + Send + Sync),
    phase: AvMuxPhase,
) -> Result<()> {
    if cancelled(phase) {
        return Err(AvMuxCancelled { phase }.into());
    }
    Ok(())
}

#[cfg(feature = "mp4")]
fn validate_pcm(samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if samples.is_empty() {
        bail!("A/V mux received an empty PCM track");
    }
    if channels == 0 {
        bail!("A/V mux PCM channel count must be greater than zero");
    }
    if !samples.len().is_multiple_of(usize::from(channels)) {
        bail!(
            "A/V mux PCM sample count {} is not divisible by {channels} channels",
            samples.len()
        );
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        bail!("A/V mux PCM track contains a non-finite sample");
    }
    let _ = sample_freq_index(sample_rate)?;
    let _ = fdk_channel_mode(channels)?;
    Ok(())
}

#[cfg(feature = "mp4")]
#[derive(Debug)]
struct H264TrackInfo {
    track_id: u32,
    width: u32,
    height: u32,
    sample_count: u32,
    duration_ticks: u64,
    timescale: u32,
    language: String,
    seq_param_set: Vec<u8>,
    pic_param_set: Vec<u8>,
}

#[cfg(feature = "mp4")]
fn find_h264_track<R: std::io::Read + std::io::Seek>(
    reader: &Mp4Reader<R>,
) -> Result<H264TrackInfo> {
    let (track_id, track) = reader
        .tracks()
        .iter()
        .find(|(_, track)| matches!(track.track_type(), Ok(TrackType::Video)))
        .ok_or_else(|| anyhow!("A/V mux input MP4 did not contain a video track"))?;
    if track.media_type()? != MediaType::H264 {
        bail!(
            "A/V mux requires H.264/AVC video, found {}",
            track.media_type()?
        );
    }
    let width = track.width() as u32;
    let height = track.height() as u32;
    if width == 0 || height == 0 || width > u16::MAX.into() || height > u16::MAX.into() {
        bail!("A/V mux input has invalid video dimensions {width}x{height}");
    }
    let timescale = track.timescale();
    let duration_ticks = track.trak.mdia.mdhd.duration;
    if timescale == 0 || duration_ticks == 0 {
        bail!("A/V mux input video has an invalid timeline");
    }
    Ok(H264TrackInfo {
        track_id: *track_id,
        width,
        height,
        sample_count: track.sample_count(),
        duration_ticks,
        timescale,
        language: track.language().to_string(),
        seq_param_set: track.sequence_parameter_set()?.to_vec(),
        pic_param_set: track.picture_parameter_set()?.to_vec(),
    })
}

#[cfg(feature = "mp4")]
fn video_track_config(video: &H264TrackInfo) -> TrackConfig {
    TrackConfig {
        track_type: TrackType::Video,
        timescale: video.timescale,
        language: video.language.clone(),
        media_conf: MediaConfig::AvcConfig(AvcConfig {
            width: video.width as u16,
            height: video.height as u16,
            seq_param_set: video.seq_param_set.clone(),
            pic_param_set: video.pic_param_set.clone(),
        }),
    }
}

#[cfg(feature = "mp4")]
fn mp4_config() -> Result<Mp4Config> {
    Ok(Mp4Config {
        major_brand: "isom".parse()?,
        minor_version: 0x200,
        compatible_brands: vec![
            "isom".parse()?,
            "iso2".parse()?,
            "avc1".parse()?,
            "mp41".parse()?,
        ],
        timescale: 1_000,
    })
}

#[cfg(feature = "mp4")]
fn recommended_aac_bitrate(sample_rate: u32, channels: u16) -> u32 {
    match (sample_rate, channels) {
        (_, 1) => 96_000,
        (48_000, 2) => 192_000,
        (_, 2) => 160_000,
        _ => 128_000,
    }
}

#[cfg(feature = "mp4")]
fn fdk_channel_mode(channels: u16) -> Result<FdkChannelMode> {
    Ok(match channels {
        1 => FdkChannelMode::Mono,
        2 => FdkChannelMode::Stereo,
        _ => bail!("AAC mux supports mono or stereo PCM, got {channels} channels"),
    })
}

#[cfg(feature = "mp4")]
fn sample_freq_index(sample_rate: u32) -> Result<SampleFreqIndex> {
    Ok(match sample_rate {
        96_000 => SampleFreqIndex::Freq96000,
        88_200 => SampleFreqIndex::Freq88200,
        64_000 => SampleFreqIndex::Freq64000,
        48_000 => SampleFreqIndex::Freq48000,
        44_100 => SampleFreqIndex::Freq44100,
        32_000 => SampleFreqIndex::Freq32000,
        24_000 => SampleFreqIndex::Freq24000,
        22_050 => SampleFreqIndex::Freq22050,
        16_000 => SampleFreqIndex::Freq16000,
        12_000 => SampleFreqIndex::Freq12000,
        11_025 => SampleFreqIndex::Freq11025,
        8_000 => SampleFreqIndex::Freq8000,
        7_350 => SampleFreqIndex::Freq7350,
        _ => bail!("AAC mux does not support a {sample_rate} Hz sample rate"),
    })
}

#[cfg(feature = "mp4")]
fn channel_config(channels: u16) -> Result<ChannelConfig> {
    Ok(match channels {
        1 => ChannelConfig::Mono,
        2 => ChannelConfig::Stereo,
        _ => bail!("AAC mux supports mono or stereo PCM, got {channels} channels"),
    })
}

#[cfg(feature = "mp4")]
fn aligned_pcm_i16(samples: &[f32], channels: u16, target_per_channel: u64) -> Result<Vec<i16>> {
    let target_per_channel = usize::try_from(target_per_channel)
        .context("aligned PCM sample count exceeded this platform's address space")?;
    let target_len = target_per_channel
        .checked_mul(usize::from(channels))
        .context("aligned interleaved PCM sample count overflowed")?;
    let mut output = Vec::with_capacity(target_len);
    output.extend(
        samples
            .iter()
            .take(target_len)
            .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16),
    );
    output.resize(target_len, 0);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_frame_grids_align_to_32khz_with_less_than_one_sample_skew() {
        for (frames, expected) in [(124u64, 165_333u64), (243, 324_000), (362, 482_667)] {
            let actual = aligned_sample_count(frames, 24, 32_000).unwrap();
            assert_eq!(actual, expected);
            let skew_numerator = (u128::from(actual) * 24).abs_diff(u128::from(frames) * 32_000);
            assert!(skew_numerator < 24, "{frames} frames skewed by >= 1 sample");
        }
    }

    #[test]
    fn invalid_timelines_fail_closed() {
        assert!(aligned_sample_count(0, 24, 32_000).is_err());
        assert!(aligned_sample_count(124, 0, 32_000).is_err());
        assert!(aligned_sample_count(124, 24, 0).is_err());
    }

    #[cfg(feature = "mp4")]
    mod mp4_tests {
        use super::*;
        use image::{ImageBuffer, Rgb, RgbImage};
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::ltx_video::video_enc;

        fn sample_video(frames: usize, fps: u32) -> Vec<u8> {
            let frames: Vec<RgbImage> = (0..frames)
                .map(|index| ImageBuffer::from_pixel(32, 32, Rgb([index as u8, 64, 192])))
                .collect();
            video_enc::encode_mp4(&frames, fps).unwrap()
        }

        fn stereo_pcm(samples_per_channel: usize, sample_rate: u32) -> Vec<f32> {
            (0..samples_per_channel)
                .flat_map(|index| {
                    let t = index as f32 / sample_rate as f32;
                    [
                        (t * std::f32::consts::TAU * 440.0).sin() * 0.2,
                        (t * std::f32::consts::TAU * 660.0).sin() * 0.2,
                    ]
                })
                .collect()
        }

        fn audio_track(bytes: &[u8]) -> (u64, u32, ChannelConfig, Vec<u32>) {
            let mut reader =
                Mp4Reader::read_header(Cursor::new(bytes), bytes.len() as u64).unwrap();
            let (track_id, duration, timescale, channels, sample_count) = {
                let (track_id, track) = reader
                    .tracks()
                    .iter()
                    .find(|(_, track)| matches!(track.track_type(), Ok(TrackType::Audio)))
                    .unwrap();
                (
                    *track_id,
                    track.trak.mdia.mdhd.duration,
                    track.timescale(),
                    track.channel_config().unwrap(),
                    track.sample_count(),
                )
            };
            let sample_durations = (1..=sample_count)
                .map(|sample_id| {
                    reader
                        .read_sample(track_id, sample_id)
                        .unwrap()
                        .unwrap()
                        .duration
                })
                .collect();
            (duration, timescale, channels, sample_durations)
        }

        fn video_track_signature(bytes: &[u8]) -> (u64, u32, u32, u16, u16, Vec<u8>, Vec<u8>) {
            let reader = Mp4Reader::read_header(Cursor::new(bytes), bytes.len() as u64).unwrap();
            let track = reader
                .tracks()
                .values()
                .find(|track| matches!(track.track_type(), Ok(TrackType::Video)))
                .unwrap();
            (
                track.trak.mdia.mdhd.duration,
                track.timescale(),
                track.sample_count(),
                track.width(),
                track.height(),
                track.sequence_parameter_set().unwrap().to_vec(),
                track.picture_parameter_set().unwrap().to_vec(),
            )
        }

        #[test]
        fn muxes_32khz_stereo_and_pads_to_the_exact_video_timeline() {
            let video = sample_video(4, 24);
            let input_samples = 4_000;
            let muxed = mux_aac_track_to_mp4_bytes(
                &video,
                &stereo_pcm(input_samples, 32_000),
                32_000,
                2,
                AacMuxOptions::match_video(),
                &|_| false,
            )
            .unwrap();

            assert_eq!(muxed.report.source_samples_per_channel, 4_000);
            assert_eq!(muxed.report.output_samples_per_channel, 5_333);
            assert_eq!(muxed.report.padded_samples_per_channel, 1_333);
            assert_eq!(muxed.report.trimmed_samples_per_channel, 0);
            let (duration, timescale, channels, packet_durations) = audio_track(&muxed.bytes);
            assert_eq!(duration, 5_333);
            assert_eq!(timescale, 32_000);
            assert_eq!(channels, ChannelConfig::Stereo);
            assert_eq!(packet_durations.len(), 6);
            assert_eq!(packet_durations, [1_024, 1_024, 1_024, 1_024, 1_024, 213]);
            assert_eq!(
                video_track_signature(&muxed.bytes),
                video_track_signature(&video)
            );
        }

        #[test]
        fn compatibility_wrapper_preserves_legacy_pcm_and_full_packet_timeline() {
            let video = sample_video(4, 24);
            let muxed =
                attach_aac_track_to_mp4_bytes(&video, &stereo_pcm(2_500, 32_000), 32_000, 2)
                    .unwrap();

            let (duration, timescale, channels, packet_durations) = audio_track(&muxed);
            assert_eq!(timescale, 32_000);
            assert_eq!(channels, ChannelConfig::Stereo);
            assert_eq!(duration, 3_072);
            assert_eq!(packet_durations, [1_024, 1_024, 1_024]);
            assert_eq!(video_track_signature(&muxed), video_track_signature(&video));
        }

        #[test]
        fn trims_long_pcm_to_the_exact_video_timeline() {
            let video = sample_video(4, 24);
            let muxed = mux_aac_track_to_mp4_bytes(
                &video,
                &stereo_pcm(8_000, 32_000),
                32_000,
                2,
                AacMuxOptions::match_video(),
                &|_| false,
            )
            .unwrap();
            assert_eq!(muxed.report.output_samples_per_channel, 5_333);
            assert_eq!(muxed.report.trimmed_samples_per_channel, 2_667);
            assert_eq!(audio_track(&muxed.bytes).0, 5_333);
        }

        #[test]
        fn rejects_malformed_pcm_before_parsing_video() {
            let cases = [
                (vec![], 32_000, 2, "empty PCM"),
                (vec![0.0], 32_000, 0, "channel count"),
                (vec![0.0, 0.0, 0.0], 32_000, 2, "not divisible"),
                (vec![f32::NAN, 0.0], 32_000, 2, "non-finite"),
                (vec![0.0, 0.0], 32_001, 2, "sample rate"),
                (vec![0.0, 0.0, 0.0], 32_000, 3, "mono or stereo"),
            ];
            for (samples, sample_rate, channels, expected) in cases {
                let error = mux_aac_track_to_mp4_bytes(
                    b"not an mp4",
                    &samples,
                    sample_rate,
                    channels,
                    AacMuxOptions::match_video(),
                    &|_| false,
                )
                .unwrap_err();
                assert!(
                    format!("{error:#}").contains(expected),
                    "unexpected error for {expected}: {error:#}"
                );
            }
        }

        #[test]
        fn malformed_mp4_does_not_publish_an_output() {
            let dir = tempfile::tempdir().unwrap();
            let input = dir.path().join("broken.mp4");
            let output = dir.path().join("output.mp4");
            fs::write(&input, b"not an mp4").unwrap();
            let error = mux_aac_track_from_f32_interleaved(
                &input,
                &output,
                &[0.0, 0.0],
                32_000,
                2,
                AacMuxOptions::match_video(),
                &|_| false,
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains("parse MP4"));
            assert!(!output.exists());
        }

        #[test]
        fn cancellation_during_encode_or_commit_preserves_output_without_tempfiles() {
            for cancel_phase in [AvMuxPhase::EncodeAudio, AvMuxPhase::Commit] {
                let dir = tempfile::tempdir().unwrap();
                let input = dir.path().join("video.mp4");
                let output = dir.path().join("output.mp4");
                fs::write(&input, sample_video(4, 24)).unwrap();
                fs::write(&output, b"existing output").unwrap();
                let checks = AtomicUsize::new(0);
                let error = mux_aac_track_from_f32_interleaved(
                    &input,
                    &output,
                    &stereo_pcm(5_333, 32_000),
                    32_000,
                    2,
                    AacMuxOptions::match_video(),
                    &|phase| {
                        checks.fetch_add(1, Ordering::Relaxed);
                        phase == cancel_phase
                    },
                )
                .unwrap_err();
                let cancellation = error.downcast_ref::<AvMuxCancelled>().unwrap();
                assert_eq!(cancellation.phase, cancel_phase);
                assert_eq!(fs::read(&output).unwrap(), b"existing output");
                assert!(checks.load(Ordering::Relaxed) > 1);
                let names: Vec<_> = fs::read_dir(dir.path())
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name())
                    .collect();
                assert_eq!(names.len(), 2, "staged tempfile leaked: {names:?}");
            }
        }
    }
}
