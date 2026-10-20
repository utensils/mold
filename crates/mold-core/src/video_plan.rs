//! Constant-space planning for a requested delivered video duration.

use crate::error::{MoldError, Result};

/// Concrete server-resolved generation bounds; clients must not infer these
/// from a model name. Both smooth carry and crossfade consume overlap frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClipEnvelope {
    pub maximum_frames: u32,
    pub minimum_frames: u32,
    pub frame_step: u32,
    pub frame_offset: u32,
    pub overlap_frames: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedClip {
    pub frames: u32,
    pub incoming_overlap_frames: u32,
    pub end_trim_frames: u32,
}

impl PlannedClip {
    pub fn delivered_frames(self) -> u64 {
        u64::from(self.frames - self.incoming_overlap_frames - self.end_trim_frames)
    }
}

/// A lazy plan remains the same size for a ten-second or day-long video.
/// The executor may materialize bounded parts without allocating every stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoDurationPlan {
    envelope: ClipEnvelope,
    added_frames: u64,
    prefix_frames: u64,
    clip_count: u64,
    last_clip: PlannedClip,
}

impl VideoDurationPlan {
    pub fn new(added_frames: u64, prefix_frames: u64, envelope: ClipEnvelope) -> Result<Self> {
        let invalid = |message: &str| MoldError::Validation(message.into());
        let ClipEnvelope {
            maximum_frames: cap,
            minimum_frames: min,
            frame_step: step,
            frame_offset: offset,
            overlap_frames: overlap,
        } = envelope;
        if step == 0
            || offset >= step
            || min == 0
            || min > cap
            || cap < offset
            || (cap - offset) % step != 0
            || min < offset
            || (min - offset) % step != 0
            || overlap >= cap
        {
            return Err(invalid("invalid video generation frame envelope"));
        }
        if added_frames == 0 {
            return Err(invalid("video duration must be positive"));
        }
        if prefix_frames > 0 && prefix_frames < u64::from(overlap) {
            return Err(invalid(
                "source video is too short for the required overlap",
            ));
        }
        prefix_frames
            .checked_add(added_frames)
            .ok_or_else(|| invalid("video duration exceeds the supported frame representation"))?;
        let first_overlap = if prefix_frames > 0 { overlap } else { 0 };
        let first_capacity = u64::from(cap - first_overlap);
        let subsequent_capacity = u64::from(cap - overlap);
        let clip_count = if added_frames <= first_capacity {
            1
        } else {
            1 + (added_frames - first_capacity).div_ceil(subsequent_capacity)
        };
        // This product is bounded by added_frames, even for u64::MAX input.
        let before_last = if clip_count == 1 {
            0
        } else {
            first_capacity + (clip_count - 2) * subsequent_capacity
        };
        let incoming = if clip_count == 1 {
            first_overlap
        } else {
            overlap
        };
        let needed = added_frames - before_last + u64::from(incoming);
        let needed = needed.max(u64::from(min));
        let raw = u64::from(offset)
            + needed
                .saturating_sub(u64::from(offset))
                .div_ceil(u64::from(step))
                * u64::from(step);
        let frames =
            u32::try_from(raw).map_err(|_| invalid("planned clip frame count overflowed"))?;
        if frames > cap {
            return Err(invalid("planned clip exceeds the single-generation limit"));
        }
        let end_trim_frames = frames
            - incoming
            - u32::try_from(added_frames - before_last)
                .map_err(|_| invalid("planned clip duration overflowed"))?;
        Ok(Self {
            envelope,
            added_frames,
            prefix_frames,
            clip_count,
            last_clip: PlannedClip {
                frames,
                incoming_overlap_frames: incoming,
                end_trim_frames,
            },
        })
    }

    pub fn clip_count(&self) -> u64 {
        self.clip_count
    }
    pub fn output_frames(&self) -> u64 {
        self.prefix_frames + self.added_frames
    }
    pub fn clip(&self, index: u64) -> Option<PlannedClip> {
        if index >= self.clip_count {
            return None;
        }
        if index == self.clip_count - 1 {
            return Some(self.last_clip);
        }
        Some(PlannedClip {
            frames: self.envelope.maximum_frames,
            incoming_overlap_frames: if index == 0 && self.prefix_frames == 0 {
                0
            } else {
                self.envelope.overlap_frames
            },
            end_trim_frames: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thirty_seconds_accounts_for_motion_overlap_and_grid_rounding() {
        let plan = VideoDurationPlan::new(
            720,
            0,
            ClipEnvelope {
                maximum_frames: 97,
                minimum_frames: 9,
                frame_step: 8,
                frame_offset: 1,
                overlap_frames: 17,
            },
        )
        .unwrap();
        assert_eq!(plan.clip_count(), 9);
        assert_eq!(
            plan.clip(8).unwrap(),
            PlannedClip {
                frames: 81,
                incoming_overlap_frames: 17,
                end_trim_frames: 1,
            }
        );
        assert_eq!(
            (0..plan.clip_count())
                .map(|i| plan.clip(i).unwrap().delivered_frames())
                .sum::<u64>(),
            720
        );
    }

    #[test]
    fn h3_grid_and_existing_prefix_deliver_exact_added_duration() {
        let plan = VideoDurationPlan::new(
            720,
            107,
            ClipEnvelope {
                maximum_frames: 107,
                minimum_frames: 22,
                frame_step: 17,
                frame_offset: 5,
                overlap_frames: 1,
            },
        )
        .unwrap();
        assert_eq!(plan.output_frames(), 827);
        assert_eq!(plan.clip_count(), 7);
        assert_eq!(
            (0..plan.clip_count())
                .map(|i| plan.clip(i).unwrap().delivered_frames())
                .sum::<u64>(),
            720
        );
    }

    #[test]
    fn long_videos_are_lazy_and_do_not_have_a_stage_count_limit() {
        let plan = VideoDurationPlan::new(
            24 * 60 * 60 * 24,
            0,
            ClipEnvelope {
                maximum_frames: 97,
                minimum_frames: 9,
                frame_step: 8,
                frame_offset: 1,
                overlap_frames: 17,
            },
        )
        .unwrap();
        assert!(plan.clip_count() > 16);
        assert!(plan.clip(plan.clip_count() - 1).is_some());
        assert!(plan.clip(plan.clip_count()).is_none());
    }

    #[test]
    fn every_duration_has_valid_grid_clips_and_exact_delivery() {
        for envelope in [
            ClipEnvelope {
                maximum_frames: 97,
                minimum_frames: 9,
                frame_step: 8,
                frame_offset: 1,
                overlap_frames: 17,
            },
            ClipEnvelope {
                maximum_frames: 81,
                minimum_frames: 5,
                frame_step: 4,
                frame_offset: 1,
                overlap_frames: 1,
            },
            ClipEnvelope {
                maximum_frames: 107,
                minimum_frames: 22,
                frame_step: 17,
                frame_offset: 5,
                overlap_frames: 1,
            },
            ClipEnvelope {
                maximum_frames: 107,
                minimum_frames: 22,
                frame_step: 17,
                frame_offset: 5,
                overlap_frames: 8,
            },
            ClipEnvelope {
                maximum_frames: 257,
                minimum_frames: 9,
                frame_step: 8,
                frame_offset: 1,
                overlap_frames: 8,
            },
        ] {
            for prefix in [0, 107] {
                for requested in 1..=3_000 {
                    let plan = VideoDurationPlan::new(requested, prefix, envelope).unwrap();
                    let mut delivered = 0;
                    for index in 0..plan.clip_count() {
                        let clip = plan.clip(index).unwrap();
                        assert!((envelope.minimum_frames..=envelope.maximum_frames)
                            .contains(&clip.frames));
                        assert_eq!(clip.frames % envelope.frame_step, envelope.frame_offset);
                        assert!(clip.delivered_frames() > 0);
                        delivered += clip.delivered_frames();
                    }
                    assert_eq!(delivered, requested);
                    assert_eq!(plan.output_frames(), prefix + requested);
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_or_nonprogressing_envelopes_and_overflow() {
        let valid = ClipEnvelope {
            maximum_frames: 97,
            minimum_frames: 9,
            frame_step: 8,
            frame_offset: 1,
            overlap_frames: 17,
        };
        assert!(VideoDurationPlan::new(0, 0, valid).is_err());
        assert!(VideoDurationPlan::new(1, 1, valid).is_err());
        assert!(VideoDurationPlan::new(u64::MAX, 17, valid).is_err());
        assert!(VideoDurationPlan::new(
            720,
            0,
            ClipEnvelope {
                overlap_frames: 97,
                ..valid
            }
        )
        .is_err());
        assert!(VideoDurationPlan::new(
            720,
            0,
            ClipEnvelope {
                frame_step: 0,
                ..valid
            }
        )
        .is_err());
    }
}
