//! Chain stitch planner.
//!
//! Takes the orchestrator's per-stage frame vectors and a parallel list of
//! boundary transitions, and assembles a single output `Vec<RgbImage>`
//! honouring the per-boundary rule:
//! - `Smooth`: drop leading `motion_tail_frames` of the incoming clip.
//! - `Cut`: concatenate as-is.
//! - `Fade`: replace trailing `fade_len` of prior + leading `fade_len` of
//!   incoming with a single blended block of `fade_len` frames.

use image::RgbImage;
use mold_core::TransitionMode;

use crate::audio::NativeAudioTrack;

pub struct StitchPlan {
    pub clips: Vec<Vec<RgbImage>>,
    /// Transition on the incoming side of each boundary.
    /// `boundaries.len() == clips.len() - 1`.
    pub boundaries: Vec<TransitionMode>,
    /// Per-boundary fade length in pixel frames. For non-fade boundaries
    /// the value is ignored. `fade_lens.len() == clips.len() - 1`.
    pub fade_lens: Vec<u32>,
    pub motion_tail_frames: u32,
}

impl StitchPlan {
    /// Assemble the final stitched frame vector. Consumes `self.clips`.
    pub fn assemble(mut self) -> Result<Vec<RgbImage>, StitchError> {
        if self.clips.is_empty() {
            return Err(StitchError::NoClips);
        }
        let expected_boundaries = self.clips.len() - 1;
        if self.boundaries.len() != expected_boundaries {
            return Err(StitchError::BoundaryMismatch {
                clips: self.clips.len(),
                boundaries: self.boundaries.len(),
            });
        }
        if self.fade_lens.len() != expected_boundaries {
            return Err(StitchError::FadeLenMismatch);
        }

        // Validate each boundary's lengths up front so we fail before any work.
        for (i, &t) in self.boundaries.iter().enumerate() {
            let prior = &self.clips[i];
            let next = &self.clips[i + 1];
            match t {
                TransitionMode::Smooth => {
                    let need = self.motion_tail_frames as usize;
                    if next.len() < need {
                        return Err(StitchError::ClipTooShortForTrim {
                            stage: i + 1,
                            have: next.len(),
                            need,
                        });
                    }
                }
                TransitionMode::Cut => {}
                TransitionMode::Fade => {
                    let fl = self.fade_lens[i] as usize;
                    if prior.len() < fl || next.len() < fl {
                        return Err(StitchError::ClipTooShortForFade {
                            stage: i + 1,
                            fade_len: fl,
                        });
                    }
                }
            }
        }

        // Stage 0 goes in whole; trim/blend on each incoming boundary.
        let mut out: Vec<RgbImage> = Vec::new();
        let mut clips = std::mem::take(&mut self.clips).into_iter();
        let first = clips.next().unwrap();
        out.extend(first);

        for (i, next_clip) in clips.enumerate() {
            match self.boundaries[i] {
                TransitionMode::Smooth => {
                    let drop = self.motion_tail_frames as usize;
                    out.extend(next_clip.into_iter().skip(drop));
                }
                TransitionMode::Cut => {
                    out.extend(next_clip);
                }
                TransitionMode::Fade => {
                    let fl = self.fade_lens[i];
                    let fl_usize = fl as usize;
                    // Pull the trailing fade_len frames off `out` (they're
                    // the tail of the prior clip now that it's been pushed).
                    let tail_start = out.len() - fl_usize;
                    let tail: Vec<RgbImage> = out.drain(tail_start..).collect();
                    let blended = fade_boundary(&tail, &next_clip, fl);
                    out.extend(blended);
                    // Append the post-fade remainder of next_clip.
                    out.extend(next_clip.into_iter().skip(fl_usize));
                }
            }
        }
        Ok(out)
    }
}

/// Per-stage EXR sidecar window on the stitched timeline.
///
/// Stage `i` renders `frames_i` local frames; the sidecar writes local frames
/// `[skip_leading, skip_leading + write_count)` to global indices
/// `[start_index, start_index + write_count)`. Windows across a chain tile
/// the stitched timeline exactly once — no gaps, no index written twice —
/// which is the contract [`plan_exr_stage_windows`] guarantees and the tests
/// pin against [`StitchPlan::assemble`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExrStageWindow {
    /// Local frames to skip at the head of this stage's render (the Smooth
    /// motion-tail duplicate of the prior stage's delivered tail).
    pub skip_leading: u32,
    /// Global index of the first EXR frame this stage writes.
    pub start_index: u32,
    /// Number of EXR frames this stage writes.
    pub write_count: u32,
}

/// Plan the per-stage EXR windows for a chained render.
///
/// `boundaries[i]` is the transition on the incoming side of stage `i + 1`,
/// exactly as [`StitchPlan`] consumes it. `total_frames_cap` clamps the tail
/// of the sequence when the caller trims the stitched video to a target
/// length; stages entirely past the cap get `write_count == 0`.
///
/// `Fade` is refused, never approximated: the stitched fade block is a u8
/// crossfade of two clips built after the lossy 8-bit conversion, so those
/// frames exist in no stage's linear f32 tensor and an EXR sequence
/// containing either source clip's frames would disagree with the video
/// across every fade boundary.
pub fn plan_exr_stage_windows(
    stage_frames: &[u32],
    boundaries: &[TransitionMode],
    motion_tail_frames: u32,
    total_frames_cap: Option<u32>,
) -> Result<Vec<ExrStageWindow>, StitchError> {
    if stage_frames.is_empty() {
        return Err(StitchError::NoClips);
    }
    if boundaries.len() != stage_frames.len() - 1 {
        return Err(StitchError::BoundaryMismatch {
            clips: stage_frames.len(),
            boundaries: boundaries.len(),
        });
    }
    if let Some(idx) = boundaries
        .iter()
        .position(|&transition| transition == TransitionMode::Fade)
    {
        return Err(StitchError::FadeUnsupportedForSidecar { stage: idx + 1 });
    }

    let cap = total_frames_cap.unwrap_or(u32::MAX);
    let mut windows = Vec::with_capacity(stage_frames.len());
    let mut start = 0u32;
    for (idx, &frames) in stage_frames.iter().enumerate() {
        let skip_leading = if idx == 0 {
            0
        } else {
            match boundaries[idx - 1] {
                TransitionMode::Smooth => motion_tail_frames,
                TransitionMode::Cut => 0,
                TransitionMode::Fade => unreachable!("fade boundaries are rejected above"),
            }
        };
        if frames < skip_leading {
            return Err(StitchError::ClipTooShortForTrim {
                stage: idx,
                have: frames as usize,
                need: skip_leading as usize,
            });
        }
        let delivered = frames - skip_leading;
        let write_count = delivered.min(cap.saturating_sub(start));
        windows.push(ExrStageWindow {
            skip_leading,
            start_index: start,
            write_count,
        });
        start += write_count;
    }
    Ok(windows)
}

#[derive(Debug, thiserror::Error)]
pub enum StitchError {
    #[error("stitch plan has no clips")]
    NoClips,
    #[error(
        "stage {stage} enters on a fade boundary, whose blended frames exist in no stage's \
         linear tensor; an EXR sidecar cannot be combined with fade transitions"
    )]
    FadeUnsupportedForSidecar { stage: usize },
    #[error("stitch plan has {clips} clips but {boundaries} boundaries (expected {})", clips.saturating_sub(1))]
    BoundaryMismatch { clips: usize, boundaries: usize },
    #[error("fade_lens length does not match boundaries length")]
    FadeLenMismatch,
    #[error("stage {stage} has {have} frames, needs at least {need} for motion-tail trim")]
    ClipTooShortForTrim {
        stage: usize,
        have: usize,
        need: usize,
    },
    #[error("stage {stage} is shorter than fade_len {fade_len}")]
    ClipTooShortForFade { stage: usize, fade_len: usize },
    #[error(
        "stage {stage} produced no audio but earlier stages did; chain audio must be all-or-nothing"
    )]
    AudioMissingOnContinuation { stage: usize },
    #[error(
        "stage {stage} audio format mismatch (sample_rate={sample_rate} channels={channels}, \
         expected sample_rate={expected_sample_rate} channels={expected_channels})"
    )]
    AudioMismatchedFormat {
        stage: usize,
        sample_rate: u32,
        channels: u16,
        expected_sample_rate: u32,
        expected_channels: u16,
    },
    #[error("stage {stage} audio has {have} samples, needs at least {need} for {kind}")]
    AudioClipTooShort {
        stage: usize,
        have: usize,
        need: usize,
        kind: &'static str,
    },
}

/// Convert a count of pixel frames into samples for a track at the given
/// sample rate and frame rate. Floors any fractional remainder; caller is
/// responsible for picking sample-rate × fps combinations that divide
/// cleanly (the LTX-2 vocoder emits 48_000 Hz at 24 fps, which is an exact
/// 2_000 samples per frame).
fn pixel_frames_to_samples(pixel_frames: u32, sample_rate: u32, fps: u32) -> usize {
    if fps == 0 {
        return 0;
    }
    ((sample_rate as u64 * pixel_frames as u64) / fps as u64) as usize
}

/// Stitch per-stage audio tracks into a single concatenated track using the
/// same per-boundary Smooth/Cut/Fade rules as the video stitch. Returns
/// `None` when no stage produced audio (the v1 default for video-only
/// chains). `audio_clips.len()` must match the chain stage count;
/// `boundaries.len()` and `fade_lens.len()` must equal `audio_clips.len() - 1`.
///
/// All stages must share the same `sample_rate` and `channels` — this holds
/// trivially for chains that use a single LTX-2 model (single VAE / single
/// vocoder), and any divergence indicates a renderer bug worth surfacing.
pub fn stitch_audio_clips(
    audio_clips: &[Option<NativeAudioTrack>],
    boundaries: &[TransitionMode],
    fade_lens: &[u32],
    motion_tail_frames: u32,
    fps: u32,
) -> Result<Option<NativeAudioTrack>, StitchError> {
    if audio_clips.is_empty() {
        return Ok(None);
    }
    if audio_clips.iter().all(Option::is_none) {
        return Ok(None);
    }
    if audio_clips.len() != boundaries.len() + 1 {
        return Err(StitchError::BoundaryMismatch {
            clips: audio_clips.len(),
            boundaries: boundaries.len(),
        });
    }
    if fade_lens.len() != boundaries.len() {
        return Err(StitchError::FadeLenMismatch);
    }

    // Format consistency. The first Some sets the contract; every subsequent
    // Some must match. Missing tracks on continuations are a hard error
    // (caller bug, not a valid mute).
    let first = audio_clips[0]
        .as_ref()
        .ok_or(StitchError::AudioMissingOnContinuation { stage: 0 })?;
    let sample_rate = first.sample_rate;
    let channels = first.channels;
    for (idx, slot) in audio_clips.iter().enumerate().skip(1) {
        let track = slot
            .as_ref()
            .ok_or(StitchError::AudioMissingOnContinuation { stage: idx })?;
        if track.sample_rate != sample_rate || track.channels != channels {
            return Err(StitchError::AudioMismatchedFormat {
                stage: idx,
                sample_rate: track.sample_rate,
                channels: track.channels,
                expected_sample_rate: sample_rate,
                expected_channels: channels,
            });
        }
    }

    let channels_usize = channels as usize;
    let mut out: Vec<f32> = first.interleaved_samples.clone();
    for (i, boundary) in boundaries.iter().enumerate() {
        // Both unwraps are safe: the format check above already proved every
        // continuation slot is Some.
        let next = audio_clips[i + 1].as_ref().unwrap();
        match boundary {
            TransitionMode::Cut => {
                out.extend_from_slice(&next.interleaved_samples);
            }
            TransitionMode::Smooth => {
                let drop_samples =
                    pixel_frames_to_samples(motion_tail_frames, sample_rate, fps) * channels_usize;
                if next.interleaved_samples.len() < drop_samples {
                    return Err(StitchError::AudioClipTooShort {
                        stage: i + 1,
                        have: next.interleaved_samples.len(),
                        need: drop_samples,
                        kind: "smooth motion-tail trim",
                    });
                }
                out.extend_from_slice(&next.interleaved_samples[drop_samples..]);
            }
            TransitionMode::Fade => {
                let fade_samples =
                    pixel_frames_to_samples(fade_lens[i], sample_rate, fps) * channels_usize;
                if out.len() < fade_samples {
                    return Err(StitchError::AudioClipTooShort {
                        stage: i,
                        have: out.len(),
                        need: fade_samples,
                        kind: "fade trailing crossfade",
                    });
                }
                if next.interleaved_samples.len() < fade_samples {
                    return Err(StitchError::AudioClipTooShort {
                        stage: i + 1,
                        have: next.interleaved_samples.len(),
                        need: fade_samples,
                        kind: "fade leading crossfade",
                    });
                }
                let prior_start = out.len() - fade_samples;
                // Linear crossfade in-place over `fade_samples` slots: the
                // trailing prior block is overwritten with `prior * (1-t) +
                // next_leading * t`.
                if fade_samples > 0 {
                    let denom = (fade_samples / channels_usize.max(1)).max(1) as f32;
                    for s in 0..fade_samples {
                        // `t` advances per audio frame, not per sample, so
                        // both channels at the same time index get the same
                        // weight.
                        let frame_index = (s / channels_usize.max(1)) as f32;
                        let t = frame_index / denom;
                        let prior_val = out[prior_start + s];
                        let next_val = next.interleaved_samples[s];
                        out[prior_start + s] = prior_val * (1.0 - t) + next_val * t;
                    }
                }
                out.extend_from_slice(&next.interleaved_samples[fade_samples..]);
            }
        }
    }

    Ok(Some(NativeAudioTrack {
        interleaved_samples: out,
        sample_rate,
        channels,
    }))
}

/// Cross-fade the last `fade_len` frames of clip N with the first
/// `fade_len` frames of clip N+1. Returns a new `Vec<RgbImage>` of length
/// `fade_len` with pixel-wise linear interpolation:
///
/// ```text
/// out[i] = (1 - alpha) * tail[tail.len() - fade_len + i]
///        +      alpha  * head[i]
/// where alpha = i / fade_len
/// ```
///
/// Panics if `tail.len() < fade_len` or `head.len() < fade_len`.
pub fn fade_boundary(tail: &[RgbImage], head: &[RgbImage], fade_len: u32) -> Vec<RgbImage> {
    let n = fade_len as usize;
    assert!(
        tail.len() >= n && head.len() >= n,
        "tail and head must have length >= fade_len"
    );
    let tail_slice = &tail[tail.len() - n..];
    let head_slice = &head[..n];
    let (w, h) = tail_slice[0].dimensions();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let alpha = i as f32 / fade_len as f32;
        let inv = 1.0 - alpha;
        let mut blended = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = tail_slice[i].get_pixel(x, y).0;
                let b = head_slice[i].get_pixel(x, y).0;
                let mixed = [
                    (a[0] as f32 * inv + b[0] as f32 * alpha) as u8,
                    (a[1] as f32 * inv + b[1] as f32 * alpha) as u8,
                    (a[2] as f32 * inv + b[2] as f32 * alpha) as u8,
                ];
                blended.put_pixel(x, y, image::Rgb(mixed));
            }
        }
        out.push(blended);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn track(samples: Vec<f32>, channels: u16, sample_rate: u32) -> NativeAudioTrack {
        NativeAudioTrack {
            interleaved_samples: samples,
            sample_rate,
            channels,
        }
    }

    /// Stage 0 carries audio, stage 1 does not. Mixed-audio chains should
    /// fail loudly rather than silently dropping audio that the AV engine
    /// actually emitted — a missing track on a continuation indicates a
    /// renderer bug, not a valid mute.
    #[test]
    fn stitch_audio_mixed_some_and_none_errors() {
        let clips = vec![Some(track(vec![1.0, 2.0, 3.0, 4.0], 1, 48_000)), None];
        let err = stitch_audio_clips(
            &clips,
            &[TransitionMode::Cut],
            &[0],
            0,
            48_000 / 100, // fps that yields 100 samples/frame for tidy math
        )
        .expect_err("mixed Some/None must fail");
        assert!(
            matches!(err, StitchError::AudioMissingOnContinuation { stage: 1 }),
            "expected AudioMissingOnContinuation, got {err:?}",
        );
    }

    #[test]
    fn stitch_audio_all_none_returns_none() {
        let clips: Vec<Option<NativeAudioTrack>> = vec![None, None, None];
        let out = stitch_audio_clips(
            &clips,
            &[TransitionMode::Smooth, TransitionMode::Cut],
            &[0, 0],
            17,
            24,
        )
        .expect("all-None must succeed");
        assert!(out.is_none(), "no audio anywhere → no output track");
    }

    #[test]
    fn stitch_audio_single_stage_returns_unchanged() {
        let samples: Vec<f32> = (0..200).map(|n| n as f32).collect();
        let clips = vec![Some(track(samples.clone(), 2, 48_000))];
        let out = stitch_audio_clips(&clips, &[], &[], 17, 24)
            .expect("single-stage must succeed")
            .expect("single-stage must produce a track");
        assert_eq!(out.interleaved_samples, samples);
        assert_eq!(out.sample_rate, 48_000);
        assert_eq!(out.channels, 2);
    }

    #[test]
    fn stitch_audio_cut_concatenates_hard() {
        // Cut = no overlap, no crossfade. 1ch / 48kHz / 24fps → 2000 samples
        // per pixel frame; 4 frames per stage → 8000 samples per stage.
        let stage0: Vec<f32> = (0..8_000).map(|n| n as f32).collect();
        let stage1: Vec<f32> = (0..8_000).map(|n| (n + 100_000) as f32).collect();
        let clips = vec![
            Some(track(stage0.clone(), 1, 48_000)),
            Some(track(stage1.clone(), 1, 48_000)),
        ];
        let out = stitch_audio_clips(&clips, &[TransitionMode::Cut], &[0], 17, 24)
            .expect("cut must succeed")
            .expect("cut must produce a track");
        assert_eq!(out.interleaved_samples.len(), 16_000);
        assert_eq!(&out.interleaved_samples[..8_000], stage0.as_slice());
        assert_eq!(&out.interleaved_samples[8_000..], stage1.as_slice());
    }

    #[test]
    fn stitch_audio_smooth_drops_motion_tail_samples_from_continuations() {
        // Smooth = drop leading samples on continuation matching
        // motion_tail_frames worth of audio. 48000/24 = 2000 samples/frame;
        // motion_tail=17 → drop 34000 samples per continuation.
        // Stage size: 50 frames × 2000 = 100_000 samples.
        let stage0: Vec<f32> = (0..100_000).map(|n| n as f32).collect();
        let stage1: Vec<f32> = (0..100_000).map(|n| (n + 1_000_000) as f32).collect();
        let clips = vec![
            Some(track(stage0.clone(), 1, 48_000)),
            Some(track(stage1.clone(), 1, 48_000)),
        ];
        let out = stitch_audio_clips(&clips, &[TransitionMode::Smooth], &[0], 17, 24)
            .expect("smooth must succeed")
            .expect("smooth must produce a track");
        // 100_000 (stage 0) + (100_000 - 34_000) = 166_000
        assert_eq!(out.interleaved_samples.len(), 166_000);
        assert_eq!(&out.interleaved_samples[..100_000], stage0.as_slice());
        // First sample after stage 0 is stage1[34_000].
        assert_eq!(out.interleaved_samples[100_000], 1_034_000.0);
    }

    #[test]
    fn stitch_audio_fade_crossfades_linearly_over_fade_frames() {
        // Fade = drop trailing fade_len samples from prior + leading
        // fade_len samples from incoming, replace with linear crossfade of
        // fade_len samples.
        // 48000/24 = 2000 samples/frame; fade_frames=2 → 4000 sample
        // crossfade. Use a constant signal per stage to verify the blend
        // hits exactly the midpoint.
        let stage0: Vec<f32> = vec![10.0; 8_000];
        let stage1: Vec<f32> = vec![20.0; 8_000];
        let clips = vec![
            Some(track(stage0, 1, 48_000)),
            Some(track(stage1, 1, 48_000)),
        ];
        let out = stitch_audio_clips(&clips, &[TransitionMode::Fade], &[2], 17, 24)
            .expect("fade must succeed")
            .expect("fade must produce a track");
        // Net per-boundary: clip + clip - fade_samples = 8_000 + 8_000 - 4_000 = 12_000
        assert_eq!(out.interleaved_samples.len(), 12_000);
        // First (8_000 - 4_000) = 4_000 samples are stage0's prefix at 10.
        assert!(out.interleaved_samples[..4_000].iter().all(|&s| s == 10.0));
        // The crossfade region of 4000 samples runs from 10 → 20 linearly.
        // Middle of crossfade should be ~15.
        let mid = out.interleaved_samples[4_000 + 2_000];
        assert!(
            (mid - 15.0).abs() < 0.5,
            "crossfade midpoint should be ~15, got {mid}",
        );
        // Last 4_000 samples are stage1's suffix at 20.
        assert!(out.interleaved_samples[8_000..].iter().all(|&s| s == 20.0));
    }

    #[test]
    fn stitch_audio_rejects_mismatched_sample_rate() {
        let clips = vec![
            Some(track(vec![1.0; 100], 1, 48_000)),
            Some(track(vec![1.0; 100], 1, 44_100)),
        ];
        let err = stitch_audio_clips(&clips, &[TransitionMode::Cut], &[0], 0, 24)
            .expect_err("sample-rate mismatch must fail");
        assert!(
            matches!(err, StitchError::AudioMismatchedFormat { .. }),
            "expected AudioMismatchedFormat, got {err:?}",
        );
    }

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for px in img.pixels_mut() {
            *px = image::Rgb(rgb);
        }
        img
    }

    fn clip(len: usize, rgb: [u8; 3]) -> Vec<RgbImage> {
        (0..len).map(|_| solid(2, 2, rgb)).collect()
    }

    #[test]
    fn fade_length_matches_requested_fade_len() {
        let tail = vec![solid(4, 4, [255, 0, 0]); 8];
        let head = vec![solid(4, 4, [0, 255, 0]); 8];
        let blended = fade_boundary(&tail, &head, 8);
        assert_eq!(blended.len(), 8);
    }

    #[test]
    fn fade_first_frame_matches_tail() {
        let tail = vec![solid(2, 2, [200, 0, 0]); 4];
        let head = vec![solid(2, 2, [0, 200, 0]); 4];
        let blended = fade_boundary(&tail, &head, 4);
        // alpha[0] = 0/4 = 0 → pure tail
        let p = blended[0].get_pixel(0, 0);
        assert_eq!(p.0, [200, 0, 0]);
    }

    #[test]
    fn fade_last_frame_is_close_to_head() {
        let tail = vec![solid(2, 2, [200, 0, 0]); 4];
        let head = vec![solid(2, 2, [0, 200, 0]); 4];
        let blended = fade_boundary(&tail, &head, 4);
        // alpha[3] = 3/4 = 0.75 → 0.25*tail + 0.75*head = [50, 150, 0]
        let p = blended[3].get_pixel(0, 0);
        assert!((p.0[0] as i32 - 50).abs() <= 2, "R was {}", p.0[0]);
        assert!((p.0[1] as i32 - 150).abs() <= 2, "G was {}", p.0[1]);
    }

    #[test]
    #[should_panic(expected = "tail and head must have length >= fade_len")]
    fn fade_shorter_than_fade_len_panics() {
        let tail = vec![solid(1, 1, [0, 0, 0]); 2];
        let head = vec![solid(1, 1, [0, 0, 0]); 2];
        let _ = fade_boundary(&tail, &head, 4);
    }

    #[test]
    fn all_smooth_drops_motion_tail() {
        let plan = StitchPlan {
            clips: vec![clip(97, [0, 0, 0]); 3],
            boundaries: vec![TransitionMode::Smooth, TransitionMode::Smooth],
            fade_lens: vec![0, 0],
            motion_tail_frames: 25,
        };
        let out = plan.assemble().unwrap();
        assert_eq!(out.len(), 97 + 72 + 72);
    }

    #[test]
    fn all_cut_keeps_everything() {
        let plan = StitchPlan {
            clips: vec![clip(97, [0, 0, 0]); 3],
            boundaries: vec![TransitionMode::Cut, TransitionMode::Cut],
            fade_lens: vec![0, 0],
            motion_tail_frames: 25,
        };
        let out = plan.assemble().unwrap();
        assert_eq!(out.len(), 97 * 3);
    }

    #[test]
    fn fade_boundary_consumes_2x_fade_len_net() {
        let plan = StitchPlan {
            clips: vec![clip(97, [255, 0, 0]), clip(97, [0, 255, 0])],
            boundaries: vec![TransitionMode::Fade],
            fade_lens: vec![8],
            motion_tail_frames: 25,
        };
        let out = plan.assemble().unwrap();
        // 97 + (97 - 8) = 186
        assert_eq!(out.len(), 186);
    }

    #[test]
    fn exr_windows_all_smooth_uniform_chain() {
        let windows = plan_exr_stage_windows(
            &[97, 97, 97],
            &[TransitionMode::Smooth, TransitionMode::Smooth],
            17,
            None,
        )
        .unwrap();
        assert_eq!(
            windows,
            vec![
                ExrStageWindow {
                    skip_leading: 0,
                    start_index: 0,
                    write_count: 97,
                },
                ExrStageWindow {
                    skip_leading: 17,
                    start_index: 97,
                    write_count: 80,
                },
                ExrStageWindow {
                    skip_leading: 17,
                    start_index: 177,
                    write_count: 80,
                },
            ],
        );
    }

    #[test]
    fn exr_windows_cap_clamps_the_tail() {
        let windows =
            plan_exr_stage_windows(&[97, 97], &[TransitionMode::Smooth], 17, Some(121)).unwrap();
        assert_eq!(
            windows[1],
            ExrStageWindow {
                skip_leading: 17,
                start_index: 97,
                write_count: 24,
            },
        );
        // A stage entirely past the cap writes nothing.
        let windows = plan_exr_stage_windows(
            &[97, 97, 97],
            &[TransitionMode::Smooth, TransitionMode::Smooth],
            17,
            Some(97),
        )
        .unwrap();
        assert_eq!(windows[1].write_count, 0);
        assert_eq!(windows[2].write_count, 0);
        assert_eq!(windows[2].start_index, 97);
    }

    #[test]
    fn exr_windows_exact_fit_last_stage_covers_the_total_exactly() {
        // The HDR chain path sizes its last stage to the exact remainder:
        // 121 total @ clip 97 / tail 17 → stages [97, 41]. Windows must sum
        // to exactly 121 with the last stage delivering 24.
        let windows =
            plan_exr_stage_windows(&[97, 41], &[TransitionMode::Smooth], 17, Some(121)).unwrap();
        assert_eq!(
            windows,
            vec![
                ExrStageWindow {
                    skip_leading: 0,
                    start_index: 0,
                    write_count: 97,
                },
                ExrStageWindow {
                    skip_leading: 17,
                    start_index: 97,
                    write_count: 24,
                },
            ],
        );
    }

    #[test]
    fn exr_windows_cut_boundaries_skip_nothing() {
        let windows = plan_exr_stage_windows(&[97, 97], &[TransitionMode::Cut], 17, None).unwrap();
        assert_eq!(windows[1].skip_leading, 0);
        assert_eq!(windows[1].start_index, 97);
        assert_eq!(windows[1].write_count, 97);
    }

    #[test]
    fn exr_windows_zero_tail_smooth_equals_cut() {
        let smooth = plan_exr_stage_windows(&[25, 25], &[TransitionMode::Smooth], 0, None).unwrap();
        let cut = plan_exr_stage_windows(&[25, 25], &[TransitionMode::Cut], 0, None).unwrap();
        assert_eq!(smooth, cut);
    }

    #[test]
    fn exr_windows_reject_fade_naming_the_boundary() {
        let err = plan_exr_stage_windows(
            &[97, 97, 97],
            &[TransitionMode::Smooth, TransitionMode::Fade],
            17,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, StitchError::FadeUnsupportedForSidecar { stage: 2 }),
            "expected FadeUnsupportedForSidecar at stage 2, got {err:?}",
        );
    }

    /// The windows must tile `[0, N)` exactly once, where `N` is the length
    /// of the stitched video `StitchPlan::assemble` produces for the same
    /// shape (truncated to the cap). This is the "EXR count equals the
    /// stitched frame count and no index is written twice" requirement from
    /// issue #688, proven at the arithmetic level over a grid of Smooth/Cut
    /// mixes, tails, and caps.
    #[test]
    fn exr_windows_tile_the_stitched_timeline_exactly_once() {
        use std::collections::BTreeSet;
        let tails = [0u32, 9, 17];
        let shapes: [&[u32]; 4] = [&[97], &[97, 97], &[97, 41, 97], &[25, 33, 25, 97]];
        let boundary_choices = [TransitionMode::Smooth, TransitionMode::Cut];
        for tail in tails {
            for stage_frames in shapes {
                let boundary_count = stage_frames.len() - 1;
                // Enumerate every Smooth/Cut assignment for the boundaries.
                for mask in 0..(1u32 << boundary_count) {
                    let boundaries: Vec<TransitionMode> = (0..boundary_count)
                        .map(|bit| boundary_choices[((mask >> bit) & 1) as usize])
                        .collect();
                    let clips: Vec<Vec<RgbImage>> = stage_frames
                        .iter()
                        .map(|&frames| clip(frames as usize, [0, 0, 0]))
                        .collect();
                    let stitched = StitchPlan {
                        clips,
                        boundaries: boundaries.clone(),
                        fade_lens: vec![0; boundary_count],
                        motion_tail_frames: tail,
                    }
                    .assemble()
                    .unwrap()
                    .len() as u32;
                    for cap in [
                        None,
                        Some(stitched),
                        Some(stitched.saturating_sub(5)),
                        Some(1),
                    ] {
                        let windows =
                            plan_exr_stage_windows(stage_frames, &boundaries, tail, cap).unwrap();
                        let expected = cap.map_or(stitched, |cap| cap.min(stitched));
                        let mut seen = BTreeSet::new();
                        for window in &windows {
                            for offset in 0..window.write_count {
                                assert!(
                                    seen.insert(window.start_index + offset),
                                    "index {} written twice (tail {tail}, stages {stage_frames:?}, \
                                     boundaries {boundaries:?}, cap {cap:?})",
                                    window.start_index + offset,
                                );
                            }
                        }
                        assert_eq!(
                            seen.len() as u32,
                            expected,
                            "window total != stitched length (tail {tail}, stages \
                             {stage_frames:?}, boundaries {boundaries:?}, cap {cap:?})",
                        );
                        assert_eq!(seen.first().copied(), (expected > 0).then_some(0));
                        assert_eq!(
                            seen.last().copied(),
                            expected.checked_sub(1),
                            "windows must be contiguous from zero",
                        );
                    }
                }
            }
        }
    }

    /// The planner's per-stage delivery must agree with mold-core's
    /// `stage_contributed_frames` — the arithmetic home the server persists
    /// as `frames_emitted` — for every Smooth/Cut chain. Three consumers,
    /// one answer.
    #[test]
    fn exr_windows_agree_with_stage_contributed_frames() {
        let stage_frames: &[u32] = &[97, 41, 97, 25];
        let boundaries = [
            TransitionMode::Smooth,
            TransitionMode::Cut,
            TransitionMode::Smooth,
        ];
        let tail = 17;
        let windows = plan_exr_stage_windows(stage_frames, &boundaries, tail, None).unwrap();
        for (idx, window) in windows.iter().enumerate() {
            let transition = if idx == 0 {
                TransitionMode::Smooth
            } else {
                boundaries[idx - 1]
            };
            let expected = mold_core::chain::stage_contributed_frames(
                idx,
                stage_frames[idx],
                transition,
                // Fade attribution is the only case where the NEXT boundary
                // matters; the sidecar planner rejects fades outright.
                None,
                None,
                tail,
            );
            assert_eq!(
                window.write_count, expected,
                "stage {idx} delivery disagrees with stage_contributed_frames",
            );
        }
    }

    #[test]
    fn mismatched_boundaries_errors() {
        let plan = StitchPlan {
            clips: vec![clip(97, [0, 0, 0]); 3],
            boundaries: vec![TransitionMode::Smooth], // expected 2
            fade_lens: vec![0, 0],
            motion_tail_frames: 25,
        };
        assert!(matches!(
            plan.assemble().unwrap_err(),
            StitchError::BoundaryMismatch { .. }
        ));
    }
}
