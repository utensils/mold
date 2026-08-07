#![allow(clippy::too_many_arguments)]

use anyhow::{anyhow, bail, Context, Result};
use image::{GenericImage, Rgb, RgbImage};
use mold_core::OutputFormat;
use std::fs;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;

use mp4_rs::{
    AvcConfig, ChannelConfig, MediaConfig, MediaType, Mp4Config, Mp4Reader, Mp4Writer, TrackConfig,
    TrackType,
};
use openh264::decoder::{Decoder, DecoderConfig, Flush};
use openh264::formats::YUVSource;

use crate::ltx_video::video_enc;

// Compatibility exports for downstream callers. The implementation and all
// new call sites live at the family-neutral `crate::av_media` boundary.
pub use crate::av_media::{attach_aac_track_from_f32_interleaved, attach_aac_track_to_mp4_bytes};

#[derive(Debug, Default)]
pub struct ProbeMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frames: Option<u32>,
    pub duration_ms: Option<u64>,
    pub has_audio: bool,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedAudioProbe {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_per_channel: u64,
}

/// Decode a staged MP4 soundtrack and report the exact PCM authority used by
/// Ref2VA preprocessing. Container duration alone is deliberately not enough:
/// codec delay and packet boundaries can change the decoded sample count.
pub fn probe_decoded_mp4_audio_file(path: &Path) -> Result<Option<DecodedAudioProbe>> {
    let Some(decoded) = super::model::DecodedAudio::from_mp4_file(path, None)? else {
        return Ok(None);
    };
    Ok(Some(DecodedAudioProbe {
        sample_rate: u32::try_from(decoded.sample_rate)
            .context("decoded reference audio sample rate exceeded u32")?,
        channels: u16::try_from(decoded.channel_count())
            .context("decoded reference audio channel count exceeded u16")?,
        samples_per_channel: u64::try_from(decoded.sample_count())
            .context("decoded reference audio sample count exceeded u64")?,
    }))
}

#[derive(Debug)]
#[allow(dead_code)]
struct VideoTrackInfo {
    track_id: u32,
    width: u32,
    height: u32,
    fps: u32,
    frames: Option<u32>,
    duration_ms: Option<u64>,
    #[allow(dead_code)]
    timescale: u32,
    #[allow(dead_code)]
    language: String,
    #[allow(dead_code)]
    seq_param_set: Vec<u8>,
    #[allow(dead_code)]
    pic_param_set: Vec<u8>,
}

#[derive(Debug)]
struct DecodedVideo {
    metadata: ProbeMetadata,
    frames: Vec<RgbImage>,
}

fn read_mp4(input_video: &Path) -> Result<Mp4Reader<BufReader<fs::File>>> {
    let file = fs::File::open(input_video)
        .with_context(|| format!("failed to open {}", input_video.display()))?;
    let size = file.metadata()?.len();
    Mp4Reader::read_header(BufReader::new(file), size).with_context(|| {
        format!(
            "failed to parse MP4 container from {}",
            input_video.display()
        )
    })
}

fn read_mp4_file(file: fs::File) -> Result<Mp4Reader<BufReader<fs::File>>> {
    let size = file.metadata()?.len();
    Mp4Reader::read_header(BufReader::new(file), size).context("failed to parse MP4 container")
}

fn read_mp4_slice(bytes: &[u8]) -> Result<Mp4Reader<Cursor<&[u8]>>> {
    Mp4Reader::read_header(Cursor::new(bytes), bytes.len() as u64)
        .context("failed to parse MP4 container from the request body")
}

fn find_video_track<R: Read + Seek>(reader: &Mp4Reader<R>) -> Result<VideoTrackInfo> {
    let (track_id, track) = reader
        .tracks()
        .iter()
        .find(|(_, track)| matches!(track.track_type(), Ok(TrackType::Video)))
        .ok_or_else(|| anyhow!("MP4 did not contain a video track"))?;

    if track.media_type()? != MediaType::H264 {
        bail!(
            "unsupported LTX-2 video codec: expected H.264/AVC, found {}",
            track.media_type()?
        );
    }

    let width = track.width() as u32;
    let height = track.height() as u32;
    let fps = track.frame_rate().round() as u32;
    if width == 0 || height == 0 {
        bail!("MP4 video track did not contain valid dimensions");
    }
    if fps == 0 {
        bail!("MP4 video track did not contain a valid frame rate");
    }

    Ok(VideoTrackInfo {
        track_id: *track_id,
        width,
        height,
        fps,
        frames: Some(track.sample_count()),
        duration_ms: Some(track.duration().as_millis() as u64),
        timescale: track.timescale(),
        language: track.language().to_string(),
        seq_param_set: track.sequence_parameter_set()?.to_vec(),
        pic_param_set: track.picture_parameter_set()?.to_vec(),
    })
}

fn audio_metadata<R: Read + Seek>(
    reader: &Mp4Reader<R>,
) -> Result<(bool, Option<u32>, Option<u32>)> {
    if let Some(track) = reader
        .tracks()
        .values()
        .find(|track| matches!(track.track_type(), Ok(TrackType::Audio)))
    {
        let sample_rate = track.sample_freq_index().ok().map(|value| value.freq());
        let channels = track.channel_config().ok().map(channel_config_channels);
        Ok((true, sample_rate, channels))
    } else {
        Ok((false, None, None))
    }
}

fn channel_config_channels(config: ChannelConfig) -> u32 {
    match config {
        ChannelConfig::Mono => 1,
        ChannelConfig::Stereo => 2,
        ChannelConfig::Three => 3,
        ChannelConfig::Four => 4,
        ChannelConfig::Five => 5,
        ChannelConfig::FiveOne => 6,
        ChannelConfig::SevenOne => 8,
    }
}

fn annex_b_convert_packet(
    packet: &[u8],
    length_size: u8,
    sps: &[Vec<u8>],
    pps: &[Vec<u8>],
    new_idr: &mut bool,
    sps_seen: &mut bool,
    pps_seen: &mut bool,
    out: &mut Vec<u8>,
) {
    out.clear();
    let mut stream = packet;

    while stream.len() >= length_size as usize {
        let mut nal_size = 0usize;
        for _ in 0..length_size {
            nal_size = (nal_size << 8) | usize::from(stream[0]);
            stream = &stream[1..];
        }

        if nal_size == 0 || nal_size > stream.len() {
            break;
        }

        let nal = &stream[..nal_size];
        stream = &stream[nal_size..];
        if nal.is_empty() {
            continue;
        }

        let nal_type = nal[0] & 0x1F;
        match nal_type {
            7 => *sps_seen = true,
            8 => *pps_seen = true,
            5 => {
                if !*new_idr && nal.get(1).is_some_and(|value| value & 0x80 != 0) {
                    *new_idr = true;
                }
                if *new_idr && !*sps_seen && !*pps_seen {
                    *new_idr = false;
                    for unit in sps {
                        out.extend([0, 0, 1]);
                        out.extend(unit);
                    }
                    for unit in pps {
                        out.extend([0, 0, 1]);
                        out.extend(unit);
                    }
                } else if *new_idr && *sps_seen && !*pps_seen {
                    for unit in pps {
                        out.extend([0, 0, 1]);
                        out.extend(unit);
                    }
                }
            }
            _ => {}
        }

        out.extend([0, 0, 1]);
        out.extend(nal);

        if !*new_idr && nal_type == 1 {
            *new_idr = true;
            *sps_seen = false;
            *pps_seen = false;
        }
    }
}

fn decode_video(input_video: &Path) -> Result<DecodedVideo> {
    decode_video_bounded(input_video, None, None, None)
}

const MAX_ANIMATION_EXPORT_RGB_BYTES: u64 = 256 * 1024 * 1024;

fn export_dimensions(width: u32, height: u32, max_dimension: Option<u32>) -> (u32, u32) {
    let Some(limit) = max_dimension else {
        return (width, height);
    };
    let longest = width.max(height);
    if longest <= limit {
        return (width, height);
    }
    let scale = limit as f64 / longest as f64;
    (
        ((width as f64 * scale).round() as u32).max(1),
        ((height as f64 * scale).round() as u32).max(1),
    )
}

fn estimated_export_rgb_bytes(
    width: u32,
    height: u32,
    frame_count: u32,
    source_fps: u32,
    target_fps: u32,
    max_dimension: Option<u32>,
) -> u64 {
    let (width, height) = export_dimensions(width, height, max_dimension);
    let sampled_frames = u64::from(frame_count)
        .saturating_mul(u64::from(target_fps))
        .div_ceil(u64::from(source_fps.max(1)));
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(3)
        .saturating_mul(sampled_frames)
}

fn should_sample_export_frame(frame_index: u32, source_fps: u32, target_fps: u32) -> bool {
    if frame_index == 0 || target_fps >= source_fps {
        return true;
    }
    let current = u64::from(frame_index) * u64::from(target_fps) / u64::from(source_fps);
    let previous = u64::from(frame_index - 1) * u64::from(target_fps) / u64::from(source_fps);
    current > previous
}

fn decode_video_bounded(
    input_video: &Path,
    target_fps: Option<u32>,
    max_dimension: Option<u32>,
    max_rgb_bytes: Option<u64>,
) -> Result<DecodedVideo> {
    let mut reader = read_mp4(input_video)?;
    let video = find_video_track(&reader)?;
    let (has_audio, audio_sample_rate, audio_channels) = audio_metadata(&reader)?;
    let export_fps = target_fps.unwrap_or(video.fps).min(video.fps).max(1);
    if let (Some(limit), Some(frame_count)) = (max_rgb_bytes, video.frames) {
        anyhow::ensure!(
            estimated_export_rgb_bytes(
                video.width,
                video.height,
                frame_count,
                video.fps,
                export_fps,
                max_dimension,
            ) <= limit,
            "animation export exceeds the safe decoded-frame budget; choose a smaller size or frame rate"
        );
    }

    let track = reader
        .tracks()
        .get(&video.track_id)
        .ok_or_else(|| anyhow!("video track {} disappeared during decode", video.track_id))?;
    let avcc = &track
        .trak
        .mdia
        .minf
        .stbl
        .stsd
        .avc1
        .as_ref()
        .ok_or_else(|| anyhow!("video track is missing avcC configuration"))?
        .avcc;

    let sps: Vec<Vec<u8>> = avcc
        .sequence_parameter_sets
        .iter()
        .map(|unit| unit.bytes.clone())
        .collect();
    let pps: Vec<Vec<u8>> = avcc
        .picture_parameter_sets
        .iter()
        .map(|unit| unit.bytes.clone())
        .collect();
    let length_size = avcc.length_size_minus_one + 1;

    let mut decoder = Decoder::with_api_config(
        openh264::OpenH264API::from_source(),
        DecoderConfig::new().flush_after_decode(Flush::NoFlush),
    )
    .context("failed to create H.264 decoder for LTX-2 media output")?;

    let mut converted = Vec::new();
    let estimated_frames = video
        .frames
        .unwrap_or_default()
        .saturating_mul(export_fps)
        .div_ceil(video.fps.max(1)) as usize;
    let mut frames = Vec::with_capacity(estimated_frames);
    let mut decoded_index = 0_u32;
    let mut retained_rgb_bytes = 0_u64;
    let mut new_idr = true;
    let mut sps_seen = false;
    let mut pps_seen = false;

    for sample_id in 1..=video.frames.unwrap_or_default() {
        let Some(sample) = reader.read_sample(video.track_id, sample_id)? else {
            continue;
        };
        annex_b_convert_packet(
            &sample.bytes,
            length_size,
            &sps,
            &pps,
            &mut new_idr,
            &mut sps_seen,
            &mut pps_seen,
            &mut converted,
        );
        if converted.is_empty() {
            continue;
        }

        if let Some(image) = decoder
            .decode(&converted)
            .context("failed to decode H.264 frame from MP4")?
        {
            if should_sample_export_frame(decoded_index, video.fps, export_fps) {
                let mut rgb = vec![0; image.rgb8_len()];
                image.write_rgb8(&mut rgb);
                let frame =
                    RgbImage::from_raw(video.width, video.height, rgb).ok_or_else(|| {
                        anyhow!("decoded H.264 frame size did not match track dimensions")
                    })?;
                let frame = resize_export_frame(frame, max_dimension);
                retained_rgb_bytes = retained_rgb_bytes
                    .saturating_add(u64::from(frame.width()) * u64::from(frame.height()) * 3);
                if let Some(limit) = max_rgb_bytes {
                    anyhow::ensure!(
                        retained_rgb_bytes <= limit,
                        "animation export exceeds the safe decoded-frame budget; choose a smaller size or frame rate"
                    );
                }
                frames.push(frame);
            }
            decoded_index += 1;
        }
    }

    for image in decoder
        .flush_remaining()
        .context("failed to flush delayed H.264 frames")?
    {
        if should_sample_export_frame(decoded_index, video.fps, export_fps) {
            let mut rgb = vec![0; image.rgb8_len()];
            image.write_rgb8(&mut rgb);
            let frame = RgbImage::from_raw(video.width, video.height, rgb).ok_or_else(|| {
                anyhow!("flushed H.264 frame size did not match track dimensions")
            })?;
            let frame = resize_export_frame(frame, max_dimension);
            retained_rgb_bytes = retained_rgb_bytes
                .saturating_add(u64::from(frame.width()) * u64::from(frame.height()) * 3);
            if let Some(limit) = max_rgb_bytes {
                anyhow::ensure!(
                    retained_rgb_bytes <= limit,
                    "animation export exceeds the safe decoded-frame budget; choose a smaller size or frame rate"
                );
            }
            frames.push(frame);
        }
        decoded_index += 1;
    }

    if frames.is_empty() {
        bail!(
            "no decodable video frames were found in {}",
            input_video.display()
        );
    }

    Ok(DecodedVideo {
        metadata: ProbeMetadata {
            width: frames[0].width(),
            height: frames[0].height(),
            fps: export_fps,
            frames: Some(frames.len() as u32),
            duration_ms: video.duration_ms,
            has_audio,
            audio_sample_rate,
            audio_channels,
        },
        frames,
    })
}

fn resize_export_frame(frame: RgbImage, max_dimension: Option<u32>) -> RgbImage {
    let (width, height) = export_dimensions(frame.width(), frame.height(), max_dimension);
    if (width, height) == (frame.width(), frame.height()) {
        return frame;
    }
    image::imageops::resize(&frame, width, height, image::imageops::FilterType::Triangle)
}

pub(crate) fn decode_video_frames(input_video: &Path) -> Result<(ProbeMetadata, Vec<RgbImage>)> {
    let video = decode_video(input_video)?;
    Ok((video.metadata, video.frames))
}

pub fn decode_video_frames_from_path(input: &Path) -> Result<(ProbeMetadata, Vec<RgbImage>)> {
    decode_video_frames(input)
}

#[allow(dead_code)]
fn video_only_track_config(video: &VideoTrackInfo) -> TrackConfig {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn copy_video_only_mp4(input_mp4: &Path, out_path: &Path) -> Result<()> {
    let mut reader = read_mp4(input_mp4)?;
    let video = find_video_track(&reader)?;
    let mut writer = Mp4Writer::write_start(Cursor::new(Vec::new()), &mp4_config()?)
        .context("failed to start video-only MP4 writer")?;
    writer
        .add_track(&video_only_track_config(&video))
        .context("failed to add video track to output MP4")?;

    for sample_id in 1..=video.frames.unwrap_or_default() {
        let Some(sample) = reader.read_sample(video.track_id, sample_id)? else {
            continue;
        };
        writer
            .write_sample(1, &sample)
            .with_context(|| format!("failed to copy video sample {sample_id} into silent MP4"))?;
    }

    writer
        .write_end()
        .context("failed to finalize video-only MP4")?;
    let bytes = writer.into_writer().into_inner();
    fs::write(out_path, bytes)
        .with_context(|| format!("failed to write {}", out_path.display()))?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn transcode_output(
    input_mp4: &Path,
    output_format: OutputFormat,
    out_path: &Path,
) -> Result<()> {
    match output_format {
        OutputFormat::Mp4 => {
            fs::copy(input_mp4, out_path)?;
        }
        OutputFormat::Gif => {
            let video = decode_video(input_mp4)?;
            let encoded = video_enc::encode_gif(&video.frames, video.metadata.fps)?;
            fs::write(out_path, encoded)?;
        }
        OutputFormat::Apng => {
            let video = decode_video(input_mp4)?;
            let encoded = video_enc::encode_apng(&video.frames, video.metadata.fps, None)?;
            fs::write(out_path, encoded)?;
        }
        OutputFormat::Webp => {
            #[cfg(feature = "webp")]
            {
                let video = decode_video(input_mp4)?;
                let encoded = video_enc::encode_webp(&video.frames, video.metadata.fps)?;
                fs::write(out_path, encoded)?;
            }
            #[cfg(not(feature = "webp"))]
            {
                bail!("WebP output requires the 'webp' feature");
            }
        }
        other => bail!("{other:?} is not supported for LTX-2 video output"),
    }
    Ok(())
}

/// Convert an existing gallery MP4 into a browser-friendly animation.
///
/// This is intentionally independent from generation: callers can create
/// multiple exports without mutating the authoritative gallery print.
pub fn export_animation(
    input_mp4: &Path,
    output_format: OutputFormat,
    gif_bounce: bool,
    gif_repeat_forever: bool,
    target_fps: Option<u32>,
    max_dimension: Option<u32>,
) -> Result<Vec<u8>> {
    let video = decode_video_bounded(
        input_mp4,
        target_fps,
        max_dimension,
        Some(MAX_ANIMATION_EXPORT_RGB_BYTES),
    )?;
    match output_format {
        OutputFormat::Gif => video_enc::encode_gif_with_options(
            &video.frames,
            video.metadata.fps,
            gif_bounce,
            gif_repeat_forever,
        ),
        OutputFormat::Apng => {
            anyhow::ensure!(
                !gif_bounce,
                "bounce playback is only supported for GIF exports"
            );
            video_enc::encode_apng(&video.frames, video.metadata.fps, None)
        }
        OutputFormat::Webp => {
            anyhow::ensure!(
                !gif_bounce,
                "bounce playback is only supported for GIF exports"
            );
            #[cfg(feature = "webp")]
            {
                video_enc::encode_webp(&video.frames, video.metadata.fps)
            }
            #[cfg(not(feature = "webp"))]
            {
                bail!("WebP export requires a mold build with the webp feature")
            }
        }
        other => bail!("{other:?} is not a supported animation export format"),
    }
}

#[allow(dead_code)]
pub(crate) fn strip_audio_track(input_mp4: &Path, out_path: &Path) -> Result<()> {
    copy_video_only_mp4(input_mp4, out_path)
}

pub fn extract_thumbnail(input_video: &Path, output_png: &Path) -> Result<()> {
    let video = decode_video(input_video)?;
    let thumbnail = video_enc::first_frame_png(&video.frames)?;
    fs::write(output_png, thumbnail)?;
    Ok(())
}

pub fn extract_gif_preview(input_video: &Path, output_gif: &Path) -> Result<()> {
    let video = decode_video(input_video)?;
    let gif = video_enc::encode_gif(&video.frames, video.metadata.fps)?;
    fs::write(output_gif, gif)?;
    Ok(())
}

pub fn write_contact_sheet(input_video: &Path, output_png: &Path) -> Result<()> {
    let video = decode_video(input_video)?;
    let sheet = contact_sheet_image(&video.frames)?;
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(sheet)
        .write_to(&mut buf, image::ImageFormat::Png)
        .context("failed to encode contact sheet PNG")?;
    fs::write(output_png, buf.into_inner())
        .with_context(|| format!("failed to write {}", output_png.display()))?;
    Ok(())
}

pub fn probe_video(input_video: &Path) -> Result<ProbeMetadata> {
    probe_reader(read_mp4(input_video)?)
}

/// Probe an already safely-opened regular file. Server ingress uses this
/// overload so a pathname cannot be exchanged for a symlink between allowlist
/// resolution and media parsing.
pub fn probe_video_file(file: fs::File) -> Result<ProbeMetadata> {
    probe_reader(read_mp4_file(file)?)
}

/// [`probe_video`] for a clip that has not been written to disk yet — the
/// server needs a lip-dub reference's frame count and rate before it stages
/// the inline `source_video` body anywhere.
///
/// Borrows the body rather than copying it: an inline `source_video` can be
/// 64 MiB, and the header parse only ever seeks around it.
pub fn probe_video_bytes(bytes: &[u8]) -> Result<ProbeMetadata> {
    probe_reader(read_mp4_slice(bytes)?)
}

fn probe_reader<R: Read + Seek>(reader: Mp4Reader<R>) -> Result<ProbeMetadata> {
    let video = find_video_track(&reader)?;
    let (has_audio, audio_sample_rate, audio_channels) = audio_metadata(&reader)?;
    Ok(ProbeMetadata {
        width: video.width,
        height: video.height,
        fps: video.fps,
        frames: video.frames,
        duration_ms: video.duration_ms,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

pub(crate) fn encode_wav_f32_interleaved(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>> {
    if channels == 0 {
        bail!("cannot encode a WAV with zero channels");
    }
    let bytes_per_sample = 4u16;
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .context("WAV block alignment overflowed")?;
    let byte_rate = sample_rate
        .checked_mul(block_align as u32)
        .context("WAV byte rate overflowed")?;
    let data_size = std::mem::size_of_val(samples) as u32;
    let riff_size = 36u32
        .checked_add(data_size)
        .context("WAV RIFF size overflowed")?;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(out)
}

/// Encode interleaved f32 samples as a canonical 16-bit PCM WAV.
///
/// Deliberately not the 32-bit-float variant above: that one exists to hand a
/// lossless intermediate to the AAC muxer, while this is the artifact users
/// download and browsers play. WebKit — which renders both the desktop app and
/// iPhone — does not decode float WAV reliably, and 16-bit PCM is the one
/// format every `<audio>` element handles.
///
/// Samples are clamped to `[-1, 1]` before scaling, so a vocoder overshoot
/// clips rather than wrapping to the opposite polarity.
pub(crate) fn encode_wav_i16_interleaved(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>> {
    if channels == 0 {
        bail!("cannot encode a WAV with zero channels");
    }
    let bytes_per_sample = 2u16;
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .context("WAV block alignment overflowed")?;
    let byte_rate = sample_rate
        .checked_mul(block_align as u32)
        .context("WAV byte rate overflowed")?;
    let data_size = u32::try_from(samples.len() * bytes_per_sample as usize)
        .context("WAV data chunk exceeds the 4 GiB RIFF limit")?;
    let riff_size = 36u32
        .checked_add(data_size)
        .context("WAV RIFF size overflowed")?;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    // WAVE_FORMAT_PCM
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = (clamped * i16::MAX as f32).round();
        out.extend_from_slice(&(scaled as i16).to_le_bytes());
    }
    Ok(out)
}

/// Peak-envelope waveform PNG for an audio-only gallery tile.
///
/// Rendered here, where the samples already are, so web, desktop, iPhone and
/// the TUI all get a legible thumbnail from the one artifact instead of each
/// inventing a glyph. Monochrome on transparent: the gallery grid supplies the
/// surface colour in whichever theme is active.
pub(crate) fn render_waveform_thumbnail_png(
    samples: &[f32],
    channels: u16,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    use image::{ImageEncoder, Rgba, RgbaImage};

    if width == 0 || height == 0 {
        bail!("waveform thumbnail dimensions must be non-zero");
    }
    let channels = channels.max(1) as usize;
    let frames = samples.len() / channels;
    let mut image = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
    let ink = Rgba([0xE8, 0xE8, 0xEA, 0xFF]);
    let mid = (height - 1) as f32 / 2.0;

    for column in 0..width {
        // Peak envelope over the frames this column covers. Averaging would
        // flatten transients into a smear; the peak is what a waveform is.
        let (mut low, mut high) = (0.0f32, 0.0f32);
        if frames > 0 {
            let start = (column as u64 * frames as u64 / width as u64) as usize;
            let end =
                (((column as u64 + 1) * frames as u64 / width as u64) as usize).max(start + 1);
            for frame in start..end.min(frames) {
                for channel in 0..channels {
                    let sample = samples[frame * channels + channel].clamp(-1.0, 1.0);
                    low = low.min(sample);
                    high = high.max(sample);
                }
            }
        }
        // Always paint at least the centre line so silence reads as "an audio
        // file that is silent" rather than as a failed thumbnail.
        let top = (mid - high * mid).round().clamp(0.0, (height - 1) as f32) as u32;
        let bottom = (mid - low * mid).round().clamp(0.0, (height - 1) as f32) as u32;
        for y in top.min(bottom)..=top.max(bottom) {
            image.put_pixel(column, y, ink);
        }
    }

    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            image.as_raw(),
            width,
            height,
            image::ExtendedColorType::Rgba8,
        )
        .context("failed to encode the LTX-2 waveform thumbnail")?;
    Ok(png)
}

fn contact_sheet_image(frames: &[RgbImage]) -> Result<RgbImage> {
    anyhow::ensure!(!frames.is_empty(), "no frames for contact sheet");

    let frame_width = frames[0].width();
    let frame_height = frames[0].height();
    let columns = ((frames.len() as f64).sqrt().ceil() as u32).max(1);
    let rows = (frames.len() as u32).div_ceil(columns);
    let gutter = 8u32;
    let margin = 12u32;
    let sheet_width = columns * frame_width + (columns.saturating_sub(1) * gutter) + margin * 2;
    let sheet_height = rows * frame_height + (rows.saturating_sub(1) * gutter) + margin * 2;
    let mut sheet = RgbImage::from_pixel(sheet_width, sheet_height, Rgb([18, 18, 18]));

    for (index, frame) in frames.iter().enumerate() {
        let index = index as u32;
        let row = index / columns;
        let col = index % columns;
        let x = margin + col * (frame_width + gutter);
        let y = margin + row * (frame_height + gutter);
        sheet
            .copy_from(frame, x, y)
            .with_context(|| format!("failed to place frame {} in contact sheet", index))?;
    }

    Ok(sheet)
}

#[cfg(all(test, feature = "mp4"))]
mod tests {
    use super::*;
    #[cfg(feature = "mp4")]
    use image::{ImageBuffer, Rgb};
    #[cfg(feature = "mp4")]
    use mp4_rs::{
        AacConfig, AudioObjectType, ChannelConfig, Mp4Config, Mp4Reader, Mp4Sample, Mp4Writer,
        SampleFreqIndex,
    };
    #[cfg(feature = "mp4")]
    use tempfile::tempdir;

    #[test]
    fn export_sampling_preserves_requested_non_divisible_frame_rates() {
        assert_eq!(
            (0..30)
                .filter(|index| should_sample_export_frame(*index, 30, 24))
                .count(),
            24
        );
        assert_eq!(
            (0..60)
                .filter(|index| should_sample_export_frame(*index, 60, 24))
                .count(),
            24
        );
    }

    #[test]
    fn export_budget_rejects_unsafe_original_but_allows_bounded_default() {
        let original = estimated_export_rgb_bytes(1920, 1088, 417, 24, 24, None);
        let bounded = estimated_export_rgb_bytes(1920, 1088, 417, 24, 12, Some(720));
        assert!(original > MAX_ANIMATION_EXPORT_RGB_BYTES);
        assert!(bounded <= MAX_ANIMATION_EXPORT_RGB_BYTES);
    }

    #[cfg(feature = "mp4")]
    use crate::ltx_video::video_enc;

    #[cfg(feature = "mp4")]
    fn sample_frames() -> Vec<RgbImage> {
        let colors = [[255, 64, 32], [32, 192, 255], [16, 224, 96], [240, 224, 64]];
        colors
            .into_iter()
            .map(|rgb| ImageBuffer::from_pixel(64, 64, Rgb(rgb)))
            .collect()
    }

    #[cfg(feature = "mp4")]
    fn write_mp4(frames: &[RgbImage], fps: u32) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
        let dir = tempdir()?;
        let path = dir.path().join("video.mp4");
        fs::write(&path, video_enc::encode_mp4(frames, fps)?)?;
        Ok((dir, path))
    }

    #[cfg(feature = "mp4")]
    fn write_mp4_with_dummy_aac(
        source_mp4: &Path,
    ) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
        let bytes = fs::read(source_mp4)?;
        let size = bytes.len() as u64;
        let mut reader = Mp4Reader::read_header(Cursor::new(bytes), size)?;
        let video = find_video_track(&reader)?;

        let mut writer = Mp4Writer::write_start(
            Cursor::new(Vec::new()),
            &Mp4Config {
                major_brand: "isom".parse()?,
                minor_version: 0x200,
                compatible_brands: vec![
                    "isom".parse()?,
                    "iso2".parse()?,
                    "avc1".parse()?,
                    "mp41".parse()?,
                ],
                timescale: 1_000,
            },
        )?;
        writer.add_track(&video_only_track_config(&video))?;
        writer.add_track(&TrackConfig {
            track_type: TrackType::Audio,
            timescale: 48_000,
            language: "und".to_string(),
            media_conf: MediaConfig::AacConfig(AacConfig {
                bitrate: 128_000,
                profile: AudioObjectType::AacLowComplexity,
                freq_index: SampleFreqIndex::Freq48000,
                chan_conf: ChannelConfig::Stereo,
            }),
        })?;

        for sample_id in 1..=video.frames.unwrap_or_default() {
            if let Some(sample) = reader.read_sample(video.track_id, sample_id)? {
                writer.write_sample(1, &sample)?;
            }
        }

        writer.write_sample(
            2,
            &Mp4Sample {
                start_time: 0,
                duration: 1_024,
                rendering_offset: 0,
                is_sync: true,
                bytes: mp4_rs::Bytes::from_static(&[0x21, 0x10, 0x56, 0xE5]),
            },
        )?;
        writer.write_end()?;

        let dir = tempdir()?;
        let path = dir.path().join("video-audio.mp4");
        fs::write(&path, writer.into_writer().into_inner())?;
        Ok((dir, path))
    }

    #[cfg(feature = "mp4")]
    fn sample_audio_track(samples_per_channel: usize) -> Vec<f32> {
        let mut samples = Vec::with_capacity(samples_per_channel * 2);
        for idx in 0..samples_per_channel {
            let t = idx as f32 / 48_000.0;
            let left = (t * std::f32::consts::TAU * 440.0).sin() * 0.25;
            let right = (t * std::f32::consts::TAU * 660.0).sin() * 0.25;
            samples.push(left);
            samples.push(right);
        }
        samples
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn probe_video_reads_native_mp4_metadata() {
        let frames = sample_frames();
        let (_dir, path) = write_mp4(&frames, 12).unwrap();
        let metadata = probe_video(&path).unwrap();
        assert_eq!(metadata.width, 64);
        assert_eq!(metadata.height, 64);
        assert_eq!(metadata.fps, 12);
        assert_eq!(metadata.frames, Some(4));
        assert!(!metadata.has_audio);
        assert!(metadata.duration_ms.is_some());
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn strip_audio_track_removes_audio_metadata() {
        let frames = sample_frames();
        let (_source_dir, source_path) = write_mp4(&frames, 10).unwrap();
        let (_audio_dir, audio_path) = write_mp4_with_dummy_aac(&source_path).unwrap();
        let stripped_dir = tempdir().unwrap();
        let stripped_path = stripped_dir.path().join("silent.mp4");

        let before = probe_video(&audio_path).unwrap();
        assert!(before.has_audio);
        assert_eq!(before.audio_sample_rate, Some(48_000));
        assert_eq!(before.audio_channels, Some(2));

        strip_audio_track(&audio_path, &stripped_path).unwrap();
        let after = probe_video(&stripped_path).unwrap();
        assert!(!after.has_audio);
        assert_eq!(after.audio_sample_rate, None);
        assert_eq!(after.audio_channels, None);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn ltx2_compat_mux_preserves_legacy_pcm_and_full_aac_packets() {
        let frames = sample_frames();
        let (_dir, source_path) = write_mp4(&frames, 12).unwrap();
        let out_dir = tempdir().unwrap();
        let out_path = out_dir.path().join("video-audio.mp4");

        attach_aac_track_from_f32_interleaved(
            &source_path,
            &out_path,
            &sample_audio_track(2_050),
            48_000,
            2,
        )
        .unwrap();

        let metadata = probe_video(&out_path).unwrap();
        assert!(metadata.has_audio);
        assert_eq!(metadata.audio_sample_rate, Some(48_000));
        assert_eq!(metadata.audio_channels, Some(2));

        let bytes = fs::read(&out_path).unwrap();
        let size = bytes.len() as u64;
        let mut reader = Mp4Reader::read_header(Cursor::new(bytes), size).unwrap();
        let (audio_track_id, duration, sample_count) = {
            let (track_id, track) = reader
                .tracks()
                .iter()
                .find(|(_, track)| matches!(track.track_type(), Ok(TrackType::Audio)))
                .unwrap();
            (
                *track_id,
                track.trak.mdia.mdhd.duration,
                track.sample_count(),
            )
        };
        let sample_durations: Vec<_> = (1..=sample_count)
            .map(|sample_id| {
                reader
                    .read_sample(audio_track_id, sample_id)
                    .unwrap()
                    .unwrap()
                    .duration
            })
            .collect();
        assert_eq!(duration, 3_072);
        assert_eq!(sample_durations, [1_024, 1_024, 1_024]);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn native_transcode_and_preview_outputs_are_generated_without_shellouts() {
        let frames = sample_frames();
        let (_dir, source_path) = write_mp4(&frames, 8).unwrap();
        let out_dir = tempdir().unwrap();
        let gif_path = out_dir.path().join("out.gif");
        let apng_path = out_dir.path().join("out.png");
        let thumb_path = out_dir.path().join("thumb.png");
        let preview_path = out_dir.path().join("preview.gif");
        let contact_sheet_path = out_dir.path().join("contact-sheet.png");

        transcode_output(&source_path, OutputFormat::Gif, &gif_path).unwrap();
        transcode_output(&source_path, OutputFormat::Apng, &apng_path).unwrap();
        extract_thumbnail(&source_path, &thumb_path).unwrap();
        extract_gif_preview(&source_path, &preview_path).unwrap();
        write_contact_sheet(&source_path, &contact_sheet_path).unwrap();

        assert_eq!(&fs::read(&gif_path).unwrap()[..6], b"GIF89a");
        assert_eq!(&fs::read(&apng_path).unwrap()[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&fs::read(&thumb_path).unwrap()[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&fs::read(&preview_path).unwrap()[..6], b"GIF89a");
        assert_eq!(
            &fs::read(&contact_sheet_path).unwrap()[..8],
            b"\x89PNG\r\n\x1a\n"
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn contact_sheet_layout_covers_all_frames() {
        let frames = sample_frames();
        let sheet = contact_sheet_image(&frames).unwrap();

        assert_eq!(sheet.width(), 160);
        assert_eq!(sheet.height(), 160);
        assert_eq!(*sheet.get_pixel(12, 12), Rgb([255, 64, 32]));
        assert_eq!(*sheet.get_pixel(84, 12), Rgb([32, 192, 255]));
        assert_eq!(*sheet.get_pixel(12, 84), Rgb([16, 224, 96]));
        assert_eq!(*sheet.get_pixel(84, 84), Rgb([240, 224, 64]));
    }

    #[test]
    fn wav_encoder_writes_ieee_float_stereo_header() {
        let wav = encode_wav_f32_interleaved(&[0.25, -0.25, 0.5, -0.5], 48_000, 2).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 3);
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            48_000
        );
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 16);
    }
}

/// Audio-artifact helpers. Deliberately outside the `mp4`-gated module above:
/// text-to-audio ships in a default build, so its encoder and thumbnail must
/// be covered without the AAC feature.
#[cfg(test)]
mod audio_tests {
    use super::{encode_wav_i16_interleaved, render_waveform_thumbnail_png};

    fn read_u16(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([bytes[at], bytes[at + 1]])
    }

    fn read_u32(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    /// 16-bit PCM, not the 32-bit float variant: WebKit renders both the
    /// desktop app and iPhone, and its float-WAV support is patchy.
    #[test]
    fn wav_i16_encoder_writes_canonical_pcm_stereo_header() {
        let wav = encode_wav_i16_interleaved(&[0.25, -0.25, 0.5, -0.5], 48_000, 2).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(read_u32(&wav, 16), 16, "PCM fmt chunk is 16 bytes");
        assert_eq!(read_u16(&wav, 20), 1, "WAVE_FORMAT_PCM");
        assert_eq!(read_u16(&wav, 22), 2, "channels");
        assert_eq!(read_u32(&wav, 24), 48_000, "sample rate");
        assert_eq!(read_u32(&wav, 28), 48_000 * 4, "byte rate");
        assert_eq!(read_u16(&wav, 32), 4, "block align");
        assert_eq!(read_u16(&wav, 34), 16, "bits per sample");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(read_u32(&wav, 40), 8, "four i16 samples");
        assert_eq!(wav.len(), 44 + 8);

        let first = i16::from_le_bytes([wav[44], wav[45]]);
        assert_eq!(first, (0.25 * i16::MAX as f32).round() as i16);
    }

    /// A vocoder overshoot must clip, not wrap: `(1.5 * 32767) as i16`
    /// saturates in Rust but the *negative* overshoot is what flips polarity
    /// in a naive cast, turning a loud peak into a loud opposite peak.
    #[test]
    fn wav_i16_encoder_clamps_out_of_range_samples() {
        let wav = encode_wav_i16_interleaved(&[1.5, -1.5], 24_000, 1).unwrap();
        assert_eq!(i16::from_le_bytes([wav[44], wav[45]]), i16::MAX);
        assert_eq!(i16::from_le_bytes([wav[46], wav[47]]), -i16::MAX);
    }

    #[test]
    fn wav_i16_encoder_rejects_zero_channels() {
        assert!(encode_wav_i16_interleaved(&[0.0], 48_000, 0).is_err());
    }

    #[test]
    fn waveform_thumbnail_is_a_decodable_png_of_the_requested_size() {
        let samples: Vec<f32> = (0..4_800)
            .map(|index| (index as f32 * 0.01).sin() * 0.8)
            .collect();
        let png = render_waveform_thumbnail_png(&samples, 2, 320, 180).unwrap();
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), 320);
        assert_eq!(decoded.height(), 180);
    }

    /// Silence must still render something. A blank tile is indistinguishable
    /// from a failed thumbnail, and "the file is silent" is exactly the thing
    /// a user needs the gallery to tell them.
    #[test]
    fn waveform_thumbnail_draws_a_centre_line_for_silence() {
        use image::GenericImageView;

        let png = render_waveform_thumbnail_png(&[0.0; 1_000], 1, 64, 33).unwrap();
        let decoded = image::load_from_memory(&png).unwrap();
        let opaque = (0..decoded.width())
            .filter(|x| decoded.get_pixel(*x, 16).0[3] > 0)
            .count();
        assert_eq!(opaque as u32, decoded.width());
    }

    #[test]
    fn waveform_thumbnail_rejects_zero_dimensions() {
        assert!(render_waveform_thumbnail_png(&[0.0], 1, 0, 32).is_err());
        assert!(render_waveform_thumbnail_png(&[0.0], 1, 32, 0).is_err());
    }
}
