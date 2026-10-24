//! Scene-aware lazy scheduling. Lifetime work is never expanded into clip arrays.

use crate::error::{MoldError, Result};
use crate::video_job::VideoJobRequest;
use crate::video_plan::{ClipEnvelope, PlannedClip, VideoDurationPlan};

#[derive(Debug, Clone)]
struct SceneSchedule {
    prompt: String,
    plan: VideoDurationPlan,
    first_clip: u64,
    first_part: u64,
    part_count: u64,
}

#[derive(Debug, Clone)]
pub struct VideoJobSchedule {
    scenes: Vec<SceneSchedule>,
    added_frames: u64,
    clip_count: u64,
    part_count: u64,
}

/// One clip per child retains the qualified single-generation memory envelope;
/// the next conditioning tail can come from the actual trimmed visible pixels.
#[derive(Debug, Clone)]
pub struct VideoJobPartPlan {
    pub prompt: String,
    pub scene_index: usize,
    pub first_clip: u64,
    pub clips: Vec<PlannedClip>,
}

pub fn duration_frames(duration_ms: u64, fps: u32) -> Result<u64> {
    if fps == 0 {
        return Err(MoldError::Validation("fps must be positive".into()));
    }
    u64::try_from((u128::from(duration_ms) * u128::from(fps)).div_ceil(1000))
        .map_err(|_| MoldError::Validation("duration exceeds the representable frame count".into()))
}

impl VideoJobSchedule {
    pub fn new(
        request: &VideoJobRequest,
        envelope: ClipEnvelope,
        fps: u32,
        prefix_frames: u64,
    ) -> Result<Self> {
        request.validate_shape().map_err(MoldError::Validation)?;
        let scenes: Vec<(&str, u64)> = match &request.scenes {
            Some(scenes) => scenes
                .iter()
                .map(|scene| (scene.prompt.as_str(), scene.duration_ms))
                .collect(),
            None => vec![(request.generation.prompt.as_str(), request.duration_ms)],
        };
        let mut elapsed_ms = 0u64;
        let mut added_frames = 0u64;
        let mut clip_count = 0u64;
        let mut part_count = 0u64;
        let mut schedules = Vec::with_capacity(scenes.len());
        for (prompt, duration_ms) in scenes {
            elapsed_ms = elapsed_ms
                .checked_add(duration_ms)
                .ok_or_else(|| MoldError::Validation("scene duration overflow".into()))?;
            let boundary = duration_frames(elapsed_ms, fps)?;
            let delivered = boundary - added_frames;
            if delivered == 0 {
                return Err(MoldError::Validation(
                    "each scene must deliver at least one frame at the selected fps".into(),
                ));
            }
            let prefix = prefix_frames.checked_add(added_frames).ok_or_else(|| {
                MoldError::Validation(
                    "source plus duration exceeds the frame representation".into(),
                )
            })?;
            let plan = VideoDurationPlan::new(delivered, prefix, envelope)?;
            let parts = plan.clip_count();
            schedules.push(SceneSchedule {
                prompt: prompt.to_owned(),
                first_clip: clip_count,
                first_part: part_count,
                part_count: parts,
                plan: plan.clone(),
            });
            added_frames = boundary;
            clip_count = clip_count
                .checked_add(plan.clip_count())
                .ok_or_else(|| MoldError::Validation("clip count overflow".into()))?;
            part_count = part_count
                .checked_add(parts)
                .ok_or_else(|| MoldError::Validation("part count overflow".into()))?;
        }
        Ok(Self {
            scenes: schedules,
            added_frames,
            clip_count,
            part_count,
        })
    }
    pub fn added_frames(&self) -> u64 {
        self.added_frames
    }
    pub fn clip_count(&self) -> u64 {
        self.clip_count
    }
    pub fn part_count(&self) -> u64 {
        self.part_count
    }
    /// How many clips each authored scene (or the single prompt) occupies, in
    /// order. More than one means that prompt is rendered again per clip.
    pub fn scene_clip_counts(&self) -> impl Iterator<Item = u64> + '_ {
        self.scenes.iter().map(|scene| scene.plan.clip_count())
    }
    /// New frames each scene delivers, in order.
    pub fn scene_added_frames(&self) -> impl Iterator<Item = u64> + '_ {
        self.scenes.iter().map(|scene| scene.plan.added_frames())
    }
    pub fn part(&self, index: u64) -> Option<VideoJobPartPlan> {
        let (scene_index, scene) = self.scenes.iter().enumerate().find(|(_, scene)| {
            index >= scene.first_part && index - scene.first_part < scene.part_count
        })?;
        let first = index - scene.first_part;
        let count = 1;
        let clips = (first..first + count)
            .map(|index| scene.plan.clip(index))
            .collect::<Option<Vec<_>>>()?;
        Some(VideoJobPartPlan {
            prompt: scene.prompt.clone(),
            scene_index,
            first_clip: scene.first_clip + first,
            clips,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_job::{VideoJobRequest, VideoJobScene};
    fn request() -> VideoJobRequest {
        serde_json::from_value(serde_json::json!({"generation":{"prompt":"walking","model":"ltx-2-19b-distilled:fp8","width":512,"height":512,"steps":4},"duration_ms":30000})).unwrap()
    }
    fn envelope() -> crate::video_plan::ClipEnvelope {
        crate::video_plan::ClipEnvelope {
            maximum_frames: 97,
            minimum_frames: 1,
            frame_step: 8,
            frame_offset: 1,
            overlap_frames: 17,
        }
    }
    #[test]
    fn one_prompt_day_long_plan_has_only_bounded_parts_and_exact_frames() {
        let mut request = request();
        request.duration_ms = 86400000;
        let schedule = VideoJobSchedule::new(&request, envelope(), 24, 0).unwrap();
        assert!(schedule.part_count() > 16);
        assert_eq!(schedule.added_frames(), 2073600);
        let first = schedule.part(0).unwrap();
        assert_eq!(first.clips.len(), 1);
        assert_eq!(schedule.part_count(), schedule.clip_count());
        let last = schedule.part(schedule.part_count() - 1).unwrap();
        assert!(last.clips.len() <= 16);
        assert_eq!(
            last.first_clip + last.clips.len() as u64,
            schedule.clip_count()
        );
    }
    #[test]
    fn scenes_use_cumulative_frame_boundaries_without_rounding_drift() {
        let mut request = request();
        request.duration_ms = 66;
        request.scenes = Some(vec![
            VideoJobScene {
                prompt: "one".into(),
                duration_ms: 33,
            },
            VideoJobScene {
                prompt: "two".into(),
                duration_ms: 33,
            },
        ]);
        let schedule = VideoJobSchedule::new(&request, envelope(), 24, 240).unwrap();
        assert_eq!(schedule.added_frames(), 2);
        assert_eq!(schedule.part_count(), 2);
        let first = schedule.part(0).unwrap();
        let second = schedule.part(1).unwrap();
        assert_eq!(first.prompt, "one");
        assert_eq!(second.prompt, "two");
        assert_eq!(first.clips[0].delivered_frames(), 1);
        assert_eq!(second.clips[0].delivered_frames(), 1);
        assert_eq!(second.first_clip, 1);
    }
    #[test]
    fn represented_duration_overflow_and_invisible_scenes_refuse_honestly() {
        let mut request = request();
        request.duration_ms = u64::MAX;
        assert!(VideoJobSchedule::new(&request, envelope(), u32::MAX, 0).is_err());
        request.duration_ms = 2;
        request.scenes = Some(vec![
            VideoJobScene {
                prompt: "one".into(),
                duration_ms: 1,
            },
            VideoJobScene {
                prompt: "two".into(),
                duration_ms: 1,
            },
        ]);
        assert!(VideoJobSchedule::new(&request, envelope(), 24, 0).is_err());
    }
}
