#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use anyhow::{anyhow, bail, Context, Result};
use image::{GenericImage, Rgb, RgbImage};
use mold_core::OutputFormat;
use std::fs;
use std::io::Cursor;
use std::path::Path;

#[cfg(feature = "mp4")]
use mp4_rs::{AacConfig, AudioObjectType, Bytes, Mp4Sample, SampleFreqIndex};
use mp4_rs::{
    AvcConfig, ChannelConfig, MediaConfig, MediaType, Mp4Config, Mp4Reader, Mp4Writer, TrackConfig,
    TrackType,
};
use openh264::decoder::{Decoder, DecoderConfig, Flush};
use openh264::formats::YUVSource;

use crate::ltx_video::video_enc;

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

#[derive(Debug)]
struct VideoTrackInfo {
    track_id: u32,
    width: u32,
    height: u32,
    fps: u32,
    frames: Option<u32>,
    duration_ms: Option<u64>,
    timescale: u32,
    language: String,
    seq_param_set: Vec<u8>,
    pic_param_set: Vec<u8>,
}

#[derive(Debug)]
struct DecodedVideo {
    metadata: ProbeMetadata,
    frames: Vec<RgbImage>,
}

fn read_mp4(input_video: &Path) -> Result<Mp4Reader<Cursor<Vec<u8>>>> {
    let bytes = fs::read(input_video)
        .with_context(|| format!("failed to read {}", input_video.display()))?;
    Mp4Reader::read_header(Cursor::new(bytes), fs::metadata(input_video)?.len()).with_context(
        || {
            format!(
                "failed to parse MP4 container from {}",
                input_video.display()
            )
        },
    )
}

fn find_video_track(reader: &Mp4Reader<Cursor<Vec<u8>>>) -> Result<VideoTrackInfo> {
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

fn audio_metadata(reader: &Mp4Reader<Cursor<Vec<u8>>>) -> Result<(bool, Option<u32>, Option<u32>)> {
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
    let mut reader = read_mp4(input_video)?;
    let video = find_video_track(&reader)?;
    let (has_audio, audio_sample_rate, audio_channels) = audio_metadata(&reader)?;

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
    let mut frames = Vec::with_capacity(video.frames.unwrap_or_default() as usize);
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
            let mut rgb = vec![0; image.rgb8_len()];
            image.write_rgb8(&mut rgb);
            let frame = RgbImage::from_raw(video.width, video.height, rgb).ok_or_else(|| {
                anyhow!("decoded H.264 frame size did not match track dimensions")
            })?;
            frames.push(frame);
        }
    }

    for image in decoder
        .flush_remaining()
        .context("failed to flush delayed H.264 frames")?
    {
        let mut rgb = vec![0; image.rgb8_len()];
        image.write_rgb8(&mut rgb);
        let frame = RgbImage::from_raw(video.width, video.height, rgb)
            .ok_or_else(|| anyhow!("flushed H.264 frame size did not match track dimensions"))?;
        frames.push(frame);
    }

    if frames.is_empty() {
        bail!(
            "no decodable video frames were found in {}",
            input_video.display()
        );
    }

    Ok(DecodedVideo {
        metadata: ProbeMetadata {
            width: video.width,
            height: video.height,
            fps: video.fps,
            frames: Some(frames.len() as u32),
            duration_ms: video.duration_ms,
            has_audio,
            audio_sample_rate,
            audio_channels,
        },
        frames,
    })
}

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

#[cfg(feature = "mp4")]
pub(crate) fn attach_placeholder_aac_track(input_mp4: &Path, out_path: &Path) -> Result<()> {
    let mut reader = read_mp4(input_mp4)?;
    let video = find_video_track(&reader)?;
    let mut writer = Mp4Writer::write_start(Cursor::new(Vec::new()), &mp4_config()?)
        .context("failed to start MP4 writer for AAC mux")?;
    writer
        .add_track(&video_only_track_config(&video))
        .context("failed to add video track to AAC mux output")?;
    writer
        .add_track(&TrackConfig {
            track_type: TrackType::Audio,
            timescale: 48_000,
            language: "und".to_string(),
            media_conf: MediaConfig::AacConfig(AacConfig {
                bitrate: 128_000,
                profile: AudioObjectType::AacLowComplexity,
                freq_index: SampleFreqIndex::Freq48000,
                chan_conf: ChannelConfig::Stereo,
            }),
        })
        .context("failed to add AAC audio track to MP4")?;

    for sample_id in 1..=video.frames.unwrap_or_default() {
        let Some(sample) = reader.read_sample(video.track_id, sample_id)? else {
            continue;
        };
        writer
            .write_sample(1, &sample)
            .with_context(|| format!("failed to copy video sample {sample_id} into AAC MP4"))?;
    }

    let duration_ms = video.duration_ms.unwrap_or_else(|| {
        let frames = video.frames.unwrap_or(1) as u64;
        (frames * 1000).div_ceil(video.fps.max(1) as u64)
    });
    let target_audio_samples = ((duration_ms * 48_000).div_ceil(1000)).max(1_024);
    let mut start_time = 0u64;
    while start_time < target_audio_samples {
        writer
            .write_sample(
                2,
                &Mp4Sample {
                    start_time,
                    duration: 1_024,
                    rendering_offset: 0,
                    is_sync: true,
                    bytes: Bytes::from_static(&[0x21, 0x10, 0x56, 0xE5]),
                },
            )
            .context("failed to write placeholder AAC sample")?;
        start_time += 1_024;
    }

    writer
        .write_end()
        .context("failed to finalize AAC MP4 output")?;
    fs::write(out_path, writer.into_writer().into_inner())
        .with_context(|| format!("failed to write {}", out_path.display()))?;
    Ok(())
}

#[cfg(not(feature = "mp4"))]
pub(crate) fn attach_placeholder_aac_track(_input_mp4: &Path, _out_path: &Path) -> Result<()> {
    bail!("MP4 output requires the 'mp4' feature");
}

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
    let reader = read_mp4(input_video)?;
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
}
