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

use crate::ltx2::media::fade_boundary;

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

#[derive(Debug, thiserror::Error)]
pub enum StitchError {
    #[error("stitch plan has no clips")]
    NoClips,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

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
