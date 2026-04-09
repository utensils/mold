use anyhow::{bail, Context, Result};
use mold_core::OutputFormat;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Default)]
pub(crate) struct ProbeMetadata {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) fps: u32,
    pub(crate) frames: Option<u32>,
    pub(crate) duration_ms: Option<u64>,
    pub(crate) has_audio: bool,
    pub(crate) audio_sample_rate: Option<u32>,
    pub(crate) audio_channels: Option<u32>,
}

fn run_ffmpeg<I, S>(args: I, context_message: &str) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let status = Command::new("ffmpeg")
        .args(args)
        .status()
        .with_context(|| format!("failed to run ffmpeg for {context_message}"))?;
    if !status.success() {
        bail!("ffmpeg failed while {context_message}");
    }
    Ok(())
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
            run_ffmpeg(
                [
                    "-y",
                    "-i",
                    input_mp4.to_string_lossy().as_ref(),
                    out_path.to_string_lossy().as_ref(),
                ],
                "encoding GIF",
            )?;
        }
        OutputFormat::Apng => {
            run_ffmpeg(
                [
                    "-y",
                    "-i",
                    input_mp4.to_string_lossy().as_ref(),
                    "-plays",
                    "0",
                    out_path.to_string_lossy().as_ref(),
                ],
                "encoding APNG",
            )?;
        }
        OutputFormat::Webp => {
            run_ffmpeg(
                [
                    "-y",
                    "-i",
                    input_mp4.to_string_lossy().as_ref(),
                    "-loop",
                    "0",
                    out_path.to_string_lossy().as_ref(),
                ],
                "encoding WebP",
            )?;
        }
        other => bail!("{other:?} is not supported for LTX-2 video output"),
    }
    Ok(())
}

pub(crate) fn strip_audio_track(input_mp4: &Path, out_path: &Path) -> Result<()> {
    run_ffmpeg(
        [
            "-y",
            "-i",
            input_mp4.to_string_lossy().as_ref(),
            "-an",
            "-c:v",
            "copy",
            out_path.to_string_lossy().as_ref(),
        ],
        "stripping audio track",
    )
}

pub(crate) fn extract_thumbnail(input_video: &Path, output_png: &Path) -> Result<()> {
    run_ffmpeg(
        [
            "-y",
            "-i",
            input_video.to_string_lossy().as_ref(),
            "-update",
            "1",
            "-frames:v",
            "1",
            output_png.to_string_lossy().as_ref(),
        ],
        "extracting thumbnail",
    )
}

pub(crate) fn extract_gif_preview(input_video: &Path, output_gif: &Path) -> Result<()> {
    run_ffmpeg(
        [
            "-y",
            "-i",
            input_video.to_string_lossy().as_ref(),
            output_gif.to_string_lossy().as_ref(),
        ],
        "encoding GIF preview",
    )
}

pub(crate) fn probe_video(input_video: &Path) -> Result<ProbeMetadata> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            input_video.to_string_lossy().as_ref(),
        ])
        .output()
        .context("failed to run ffprobe")?;
    if !output.status.success() {
        bail!("ffprobe failed for {}", input_video.display());
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    parse_probe_metadata(value)
}

pub(crate) fn parse_probe_metadata(value: serde_json::Value) -> Result<ProbeMetadata> {
    let mut metadata = ProbeMetadata::default();
    let streams = value["streams"].as_array().cloned().unwrap_or_default();
    for stream in streams {
        match stream["codec_type"].as_str() {
            Some("video") => {
                metadata.width = stream["width"].as_u64().unwrap_or_default() as u32;
                metadata.height = stream["height"].as_u64().unwrap_or_default() as u32;
                metadata.frames = stream["nb_frames"]
                    .as_str()
                    .and_then(|value| value.parse().ok())
                    .or_else(|| {
                        stream["nb_read_frames"]
                            .as_str()
                            .and_then(|value| value.parse().ok())
                    });
                metadata.fps = stream["r_frame_rate"]
                    .as_str()
                    .and_then(parse_ffprobe_fps)
                    .or_else(|| stream["avg_frame_rate"].as_str().and_then(parse_ffprobe_fps))
                    .unwrap_or_default();
            }
            Some("audio") => {
                metadata.has_audio = true;
                metadata.audio_sample_rate = stream["sample_rate"]
                    .as_str()
                    .and_then(|value| value.parse().ok());
                metadata.audio_channels = stream["channels"].as_u64().map(|value| value as u32);
            }
            _ => {}
        }
    }
    metadata.duration_ms = value["format"]["duration"]
        .as_str()
        .and_then(|value| value.parse::<f64>().ok())
        .map(|seconds| (seconds * 1000.0).round() as u64);

    if metadata.width == 0 || metadata.height == 0 {
        bail!("ffprobe did not return valid video dimensions");
    }
    if metadata.fps == 0 {
        bail!("ffprobe did not return a valid video frame rate");
    }

    Ok(metadata)
}

pub(crate) fn parse_ffprobe_fps(value: &str) -> Option<u32> {
    let (num, den) = value.split_once('/')?;
    let num: f64 = num.parse().ok()?;
    let den: f64 = den.parse().ok()?;
    if den == 0.0 {
        return None;
    }
    Some((num / den).round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_ffprobe_fps_rounds_fraction() {
        assert_eq!(parse_ffprobe_fps("24/1"), Some(24));
        assert_eq!(parse_ffprobe_fps("30000/1001"), Some(30));
    }

    #[test]
    fn parse_probe_metadata_rejects_missing_video_dimensions() {
        let err = parse_probe_metadata(json!({
            "streams": [{
                "codec_type": "video",
                "r_frame_rate": "24/1"
            }],
            "format": { "duration": "1.0" }
        }))
        .unwrap_err();
        assert!(err.to_string().contains("valid video dimensions"));
    }

    #[test]
    fn parse_probe_metadata_uses_avg_frame_rate_fallback() {
        let metadata = parse_probe_metadata(json!({
            "streams": [{
                "codec_type": "video",
                "width": 960,
                "height": 576,
                "avg_frame_rate": "12/1",
                "nb_frames": "17"
            }],
            "format": { "duration": "1.42" }
        }))
        .unwrap();
        assert_eq!(metadata.width, 960);
        assert_eq!(metadata.height, 576);
        assert_eq!(metadata.fps, 12);
        assert_eq!(metadata.frames, Some(17));
        assert_eq!(metadata.duration_ms, Some(1420));
    }
}
