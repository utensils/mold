use anyhow::{bail, Result};
use candle_core::Tensor;
use image::RgbImage;
use mold_core::{GenerateRequest, TimeRange};
use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StagedImage {
    pub(crate) path: String,
    pub(crate) frame: u32,
    pub(crate) strength: f32,
}

/// Pre-encoded latent block used as conditioning input, bypassing the
/// staged-image path's VAE encode. Populated by the render-chain
/// orchestrator when handing a motion-tail off between stages; empty for
/// every non-chain caller today.
///
/// Tensor shape must be `[batch=1, channels=128, T_latent, H/32, W/32]`
/// to match the LTX-2 video VAE output. The runtime patchifies it directly
/// into conditioning tokens.
#[derive(Debug, Clone)]
pub(crate) struct StagedLatent {
    pub(crate) latents: Tensor,
    /// Starting pixel frame for this latent block. `0` routes the tokens
    /// through `StageVideoConditioning::replacements`; non-zero values
    /// build a `VideoTokenAppendCondition` like the keyframe image path.
    pub(crate) frame: u32,
    /// Replacement/append strength. `1.0` for chain motion-tail carryover
    /// (hard-overwrite), matching the keyframe image strength convention.
    pub(crate) strength: f32,
    /// Optional RGB frame that the runtime re-encodes through the video VAE
    /// and swaps in for the first latent frame of `latents` before
    /// patchifying. Used by the chain orchestrator to fix a semantic
    /// mismatch: raw tail latents come from the emitting clip's
    /// *continuation* slots (each encoding 8 pixel frames), but when
    /// replayed at `frame: 0` of the next clip they land in the VAE's
    /// *causal* slot (which decodes to a single pixel frame). Encoding the
    /// last decoded RGB frame as a proper causal-first latent removes that
    /// mismatch and makes `clip_N+1`'s first pixel frame visually match
    /// `clip_N`'s last pixel frame.
    pub(crate) causal_first_frame_rgb: Option<RgbImage>,
}

/// Conditioning inputs staged for a single run. Carries both disk-backed
/// files (images, audio, reference video — existing single-clip flow) and
/// in-memory latent blocks (chain carryover — new, empty for non-chain
/// callers).
///
/// Not `PartialEq` because `StagedLatent` wraps a `candle_core::Tensor`
/// which doesn't implement meaningful structural equality. Existing tests
/// only compare individual fields so this is safe to drop.
#[derive(Debug, Clone)]
pub(crate) struct StagedConditioning {
    pub(crate) images: Vec<StagedImage>,
    pub(crate) latents: Vec<StagedLatent>,
    pub(crate) audio_path: Option<String>,
    pub(crate) video_path: Option<String>,
}

fn infer_staged_extension<'a>(data: &[u8], default_ext: &'a str) -> &'a str {
    if data.starts_with(&[0x89, b'P', b'N', b'G']) {
        "png"
    } else if data.starts_with(&[0xFF, 0xD8]) {
        "jpg"
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WAVE") {
        "wav"
    } else if data.get(4..8) == Some(b"ftyp") {
        "mp4"
    } else if data.starts_with(b"OggS") {
        "ogg"
    } else if data.starts_with(b"fLaC") {
        "flac"
    } else if data.starts_with(b"ID3")
        || data
            .get(..2)
            .is_some_and(|header| header[0] == 0xFF && (header[1] & 0xE0) == 0xE0)
    {
        "mp3"
    } else {
        default_ext
    }
}

pub(crate) fn stage_input_file(
    dir: &Path,
    stem: &str,
    data: &[u8],
    default_ext: &str,
) -> Result<PathBuf> {
    let ext = infer_staged_extension(data, default_ext);
    let path = dir.join(format!("{stem}.{ext}"));
    fs::write(&path, data)?;
    Ok(path)
}

pub(crate) fn stage_conditioning(
    req: &GenerateRequest,
    work_dir: &Path,
) -> Result<StagedConditioning> {
    let mut images = Vec::new();
    if let Some(source_image) = &req.source_image {
        let path = stage_input_file(work_dir, "source-image", source_image, "png")?;
        images.push(StagedImage {
            path: path.to_string_lossy().to_string(),
            frame: 0,
            strength: req.strength as f32,
        });
    }
    if let Some(keyframes) = &req.keyframes {
        for (index, keyframe) in keyframes.iter().enumerate() {
            let path = stage_input_file(
                work_dir,
                &format!("keyframe-{index:02}"),
                &keyframe.image,
                "png",
            )?;
            images.push(StagedImage {
                path: path.to_string_lossy().to_string(),
                frame: keyframe.frame,
                strength: 1.0,
            });
        }
    }

    let audio_path = req
        .audio_file
        .as_ref()
        .map(|bytes| stage_input_file(work_dir, "conditioning-audio", bytes, "wav"))
        .transpose()?
        .map(|path| path.to_string_lossy().to_string());

    let video_path = req
        .source_video
        .as_ref()
        .map(|bytes| stage_input_file(work_dir, "source-video", bytes, "mp4"))
        .transpose()?
        .map(|path| path.to_string_lossy().to_string());

    Ok(StagedConditioning {
        images,
        latents: Vec::new(),
        audio_path,
        video_path,
    })
}

#[allow(dead_code)]
pub(crate) fn retake_frame_window(
    range: &TimeRange,
    fps: u32,
    total_frames: u32,
) -> Result<RangeInclusive<u32>> {
    if fps == 0 {
        bail!("retake frame window requires fps > 0");
    }
    if total_frames == 0 {
        bail!("retake frame window requires total_frames > 0");
    }

    let start = (range.start_seconds * fps as f32).floor().max(0.0) as u32;
    let end_exclusive = (range.end_seconds * fps as f32).ceil().max(0.0) as u32;
    if end_exclusive == 0 {
        bail!("retake frame window does not cover any frames");
    }
    if start >= total_frames {
        bail!("retake start time is outside the available video duration");
    }

    let end = end_exclusive.saturating_sub(1).min(total_frames - 1);
    if end < start {
        bail!("retake frame window does not cover any frames");
    }
    Ok(start..=end)
}

#[allow(dead_code)]
pub(crate) fn retake_temporal_mask(
    range: &TimeRange,
    fps: u32,
    total_frames: u32,
) -> Result<Vec<f32>> {
    let active = retake_frame_window(range, fps, total_frames)?;
    Ok((0..total_frames)
        .map(|frame| if active.contains(&frame) { 1.0 } else { 0.0 })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::KeyframeCondition;

    fn req() -> GenerateRequest {
        GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(42),
            batch_size: 1,
            output_format: mold_core::OutputFormat::Mp4,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: Some(17),
            fps: Some(12),
            upscale_model: None,
            gif_preview: false,
            enable_audio: Some(true),
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        }
    }

    #[test]
    fn retake_frame_window_maps_seconds_to_frame_bounds() {
        let range = TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.25,
        };
        let frames = retake_frame_window(&range, 8, 33).unwrap();
        assert_eq!(frames, 8..=17);
    }

    #[test]
    fn retake_frame_window_clamps_to_available_frames() {
        let range = TimeRange {
            start_seconds: 2.0,
            end_seconds: 5.0,
        };
        let frames = retake_frame_window(&range, 12, 30).unwrap();
        assert_eq!(frames, 24..=29);
    }

    #[test]
    fn retake_temporal_mask_marks_only_requested_window() {
        let range = TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.25,
        };
        let mask = retake_temporal_mask(&range, 8, 20).unwrap();
        assert_eq!(mask.len(), 20);
        assert!(mask[..8].iter().all(|value| *value == 0.0));
        assert!(mask[8..18].iter().all(|value| *value == 1.0));
        assert!(mask[18..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn stage_conditioning_leaves_latents_empty_for_non_chain_callers() {
        // Single-clip callers build `StagedConditioning` via this function;
        // the `latents` field (used by the render-chain orchestrator to inject
        // pre-encoded motion-tail tokens) must stay empty so existing runs
        // keep routing conditioning through the image path with VAE encode.
        let work_dir = tempfile::tempdir().unwrap();
        let mut req = req();
        req.source_image = Some(fake_png_bytes());
        req.keyframes = Some(vec![KeyframeCondition {
            frame: 8,
            image: fake_png_bytes(),
        }]);
        req.source_video = Some(fake_mp4_bytes());
        req.audio_file = Some(fake_wav_bytes());

        let staged = stage_conditioning(&req, work_dir.path()).unwrap();
        assert!(
            staged.latents.is_empty(),
            "non-chain callers must leave latents empty",
        );
    }

    #[test]
    fn stage_conditioning_stages_source_image_as_frame_zero_replacement() {
        let work_dir = tempfile::tempdir().unwrap();
        let mut req = req();
        req.source_image = Some(fake_png_bytes());
        req.strength = 0.42;

        let staged = stage_conditioning(&req, work_dir.path()).unwrap();
        assert_eq!(staged.images.len(), 1);
        assert_eq!(staged.images[0].frame, 0);
        assert_eq!(staged.images[0].strength, 0.42);
        assert!(staged.images[0].path.ends_with("source-image.png"));
    }

    #[test]
    fn stage_conditioning_preserves_keyframe_targets() {
        let work_dir = tempfile::tempdir().unwrap();
        let mut req = req();
        req.keyframes = Some(vec![
            KeyframeCondition {
                frame: 8,
                image: fake_png_bytes(),
            },
            KeyframeCondition {
                frame: 16,
                image: fake_png_bytes(),
            },
        ]);

        let staged = stage_conditioning(&req, work_dir.path()).unwrap();
        assert_eq!(staged.images.len(), 2);
        assert_eq!(staged.images[0].frame, 8);
        assert_eq!(staged.images[1].frame, 16);
        assert!(staged.images.iter().all(|image| image.strength == 1.0));
    }

    #[test]
    fn stage_conditioning_keeps_audio_and_reference_video_paths() {
        let work_dir = tempfile::tempdir().unwrap();
        let mut req = req();
        req.audio_file = Some(fake_wav_bytes());
        req.source_video = Some(fake_mp4_bytes());

        let staged = stage_conditioning(&req, work_dir.path()).unwrap();
        assert!(staged
            .audio_path
            .as_deref()
            .is_some_and(|path| path.ends_with("conditioning-audio.wav")));
        assert!(staged
            .video_path
            .as_deref()
            .is_some_and(|path| path.ends_with("source-video.mp4")));
    }

    #[test]
    fn stage_conditioning_infers_mp4_audio_extension_from_container_bytes() {
        let work_dir = tempfile::tempdir().unwrap();
        let mut req = req();
        req.audio_file = Some(fake_mp4_bytes());

        let staged = stage_conditioning(&req, work_dir.path()).unwrap();
        assert!(staged
            .audio_path
            .as_deref()
            .is_some_and(|path| path.ends_with("conditioning-audio.mp4")));
    }

    fn fake_png_bytes() -> Vec<u8> {
        vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
    }

    fn fake_wav_bytes() -> Vec<u8> {
        b"RIFFtestWAVEfmt ".to_vec()
    }

    fn fake_mp4_bytes() -> Vec<u8> {
        vec![0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p']
    }
}
