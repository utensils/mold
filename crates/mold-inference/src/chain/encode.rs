//! Shared chain-output frame encoder.
//!
//! One home for the "stitched frames → video container" match that the
//! CLI local-chain path and the server chain routes previously each
//! hand-rolled (near-verbatim twins). Warnings are returned as data so
//! each caller keeps its own sink — `status!`/stderr on the CLI,
//! `tracing::warn!` on the server.

use anyhow::Result;
use image::RgbImage;
use mold_core::OutputFormat;

use crate::audio::NativeAudioTrack;
use crate::ltx_video::video_enc;

/// Decode an APNG byte stream back to raw RGB frames.
///
/// Stage renderers that drive a family's ordinary generation path get video
/// bytes back, not frames. APNG is the round-trip container because it is
/// lossless — every pixel survives — and the encode/decode costs tens of
/// milliseconds against a multi-second denoise.
///
/// Shared rather than duplicated: `ltx-video` and `wan` both render stages
/// this way, and two copies of an image round-trip is two places for a
/// channel-order or alpha-handling bug to differ.
pub fn decode_apng_to_rgb_frames(apng_bytes: &[u8]) -> Result<Vec<RgbImage>> {
    use image::AnimationDecoder;
    let cursor = std::io::Cursor::new(apng_bytes);
    let decoder = image::codecs::png::PngDecoder::new(cursor)
        .map_err(|e| anyhow::anyhow!("failed to open APNG bytes: {e}"))?;
    let apng = decoder
        .apng()
        .map_err(|e| anyhow::anyhow!("decoded PNG is not animated: {e}"))?;
    let mut out = Vec::new();
    for frame in apng.into_frames() {
        let frame = frame.map_err(|e| anyhow::anyhow!("APNG frame decode failed: {e}"))?;
        let rgba = frame.into_buffer();
        let (w, h) = rgba.dimensions();
        let mut rgb_data = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for px in rgba.pixels() {
            rgb_data.extend_from_slice(&px.0[..3]);
        }
        let rgb = RgbImage::from_raw(w, h, rgb_data)
            .ok_or_else(|| anyhow::anyhow!("failed to construct RgbImage from APNG frame"))?;
        out.push(rgb);
    }
    Ok(out)
}

/// Non-fatal conditions surfaced while encoding chain output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainEncodeWarning {
    /// MP4 requested but built without the `mp4` feature — fell back to APNG.
    Mp4FeatureMissing,
    /// WebP chain output is unsupported — fell back to APNG.
    WebpUnsupported,
    /// Audio was supplied but the carrier format has no audio track.
    AudioDropped { format: OutputFormat },
    /// Best-effort GIF preview encode failed (preview left empty).
    GifPreviewFailed(String),
}

impl ChainEncodeWarning {
    pub fn message(&self) -> String {
        match self {
            Self::Mp4FeatureMissing => {
                "MP4 requested but this binary was built without the `mp4` feature — \
                 falling back to APNG"
                    .to_string()
            }
            Self::WebpUnsupported => {
                "WebP chain output is not supported yet — falling back to APNG".to_string()
            }
            Self::AudioDropped { format } => {
                format!("chain audio dropped: {format} output has no audio track carrier")
            }
            Self::GifPreviewFailed(e) => format!("chain gif preview encode failed: {e}"),
        }
    }
}

/// Encoded chain output plus everything the caller needs to describe it.
pub struct EncodedChainVideo {
    pub bytes: Vec<u8>,
    /// Actual container after fallbacks (e.g. WebP request → APNG).
    pub format: OutputFormat,
    /// Best-effort GIF preview; empty when encoding failed (warned).
    pub gif_preview: Vec<u8>,
    pub warnings: Vec<ChainEncodeWarning>,
}

/// Encode stitched chain frames to the requested container.
///
/// MP4 is feature-gated: without `mp4` the encoder falls back to APNG so
/// every build still produces a usable animation. When `audio` is `Some`
/// and the output is MP4, the audio is muxed in as an AAC track via the
/// same path the single-clip pipeline uses; formats with no audio
/// carrier drop it with a warning.
pub fn encode_chain_frames(
    frames: &[RgbImage],
    fps: u32,
    requested: OutputFormat,
    audio: Option<&NativeAudioTrack>,
) -> Result<EncodedChainVideo> {
    let mut warnings = Vec::new();

    // Always produce a GIF preview for gallery/terminal UIs. Non-fatal.
    let gif_preview = match video_enc::encode_gif(frames, fps) {
        Ok(b) => b,
        Err(e) => {
            warnings.push(ChainEncodeWarning::GifPreviewFailed(format!("{e:#}")));
            Vec::new()
        }
    };

    let audio_dropped = |warnings: &mut Vec<ChainEncodeWarning>, format| {
        if audio.is_some() {
            warnings.push(ChainEncodeWarning::AudioDropped { format });
        }
    };

    let (bytes, format) = match requested {
        OutputFormat::Mp4 => {
            #[cfg(feature = "mp4")]
            {
                let video_only = video_enc::encode_mp4(frames, fps)?;
                let muxed = match audio {
                    Some(track) => crate::av_media::attach_aac_track_to_mp4_bytes(
                        &video_only,
                        &track.interleaved_samples,
                        track.sample_rate,
                        track.channels,
                    )?,
                    None => video_only,
                };
                (muxed, OutputFormat::Mp4)
            }
            #[cfg(not(feature = "mp4"))]
            {
                warnings.push(ChainEncodeWarning::Mp4FeatureMissing);
                audio_dropped(&mut warnings, OutputFormat::Apng);
                (
                    video_enc::encode_apng(frames, fps, None)?,
                    OutputFormat::Apng,
                )
            }
        }
        OutputFormat::Apng => {
            audio_dropped(&mut warnings, OutputFormat::Apng);
            (
                video_enc::encode_apng(frames, fps, None)?,
                OutputFormat::Apng,
            )
        }
        OutputFormat::Gif => {
            audio_dropped(&mut warnings, OutputFormat::Gif);
            (video_enc::encode_gif(frames, fps)?, OutputFormat::Gif)
        }
        // WebP fallback: animated-WebP chain output would bind callers to
        // another optional dep; fall back to APNG for now.
        OutputFormat::Webp => {
            warnings.push(ChainEncodeWarning::WebpUnsupported);
            audio_dropped(&mut warnings, OutputFormat::Apng);
            (
                video_enc::encode_apng(frames, fps, None)?,
                OutputFormat::Apng,
            )
        }
        other => anyhow::bail!("{other:?} is not a video output format for chain generation"),
    };

    Ok(EncodedChainVideo {
        bytes,
        format,
        gif_preview,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frames(n: usize) -> Vec<RgbImage> {
        (0..n)
            .map(|i| RgbImage::from_pixel(16, 16, image::Rgb([i as u8 * 40, 80, 120])))
            .collect()
    }

    fn test_audio() -> NativeAudioTrack {
        NativeAudioTrack {
            interleaved_samples: vec![0.0f32; 4800],
            sample_rate: 48_000,
            channels: 2,
        }
    }

    #[test]
    fn apng_request_encodes_apng_with_preview_and_no_warnings() {
        let out = encode_chain_frames(&test_frames(2), 10, OutputFormat::Apng, None).unwrap();
        assert_eq!(out.format, OutputFormat::Apng);
        assert_eq!(&out.bytes[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(!out.gif_preview.is_empty());
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn gif_request_with_audio_warns_audio_dropped() {
        let audio = test_audio();
        let out =
            encode_chain_frames(&test_frames(2), 10, OutputFormat::Gif, Some(&audio)).unwrap();
        assert_eq!(out.format, OutputFormat::Gif);
        assert_eq!(&out.bytes[..6], b"GIF89a");
        assert!(out.warnings.contains(&ChainEncodeWarning::AudioDropped {
            format: OutputFormat::Gif
        }));
    }

    #[test]
    fn webp_request_falls_back_to_apng_with_warning() {
        let audio = test_audio();
        let out =
            encode_chain_frames(&test_frames(2), 10, OutputFormat::Webp, Some(&audio)).unwrap();
        assert_eq!(out.format, OutputFormat::Apng);
        assert!(out.warnings.contains(&ChainEncodeWarning::WebpUnsupported));
        assert!(out.warnings.contains(&ChainEncodeWarning::AudioDropped {
            format: OutputFormat::Apng
        }));
    }

    #[cfg(not(feature = "mp4"))]
    #[test]
    fn mp4_request_without_feature_falls_back_to_apng() {
        let out = encode_chain_frames(&test_frames(2), 10, OutputFormat::Mp4, None).unwrap();
        assert_eq!(out.format, OutputFormat::Apng);
        assert!(out
            .warnings
            .contains(&ChainEncodeWarning::Mp4FeatureMissing));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mp4_request_with_feature_encodes_mp4_and_muxes_audio() {
        let audio = test_audio();
        let out =
            encode_chain_frames(&test_frames(2), 10, OutputFormat::Mp4, Some(&audio)).unwrap();
        assert_eq!(out.format, OutputFormat::Mp4);
        assert!(out.warnings.is_empty());
        // ftyp box shows up in the first bytes of a faststart MP4.
        assert_eq!(&out.bytes[4..8], b"ftyp");
    }

    #[test]
    fn still_image_format_is_rejected() {
        assert!(encode_chain_frames(&test_frames(1), 10, OutputFormat::Png, None).is_err());
    }
}
