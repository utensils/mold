//! Per-family chained-generation capability descriptors.
//!
//! Replaces the hardcoded family match that previously lived in
//! `mold-server`'s `chain_limits.rs`. A family appears in chain limits (and
//! is accepted by the chain endpoints) iff it returns `Some` here. New
//! video families add one arm and inherit the whole chain surface.

/// How a chain-capable family carries context across stage boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarryoverKind {
    /// The tail of each clip conditions the next clip (motion-tail latent
    /// pin via re-encoded RGB tail frames). Smooth transitions are real.
    TemporalHandoff,
    /// Stages render independently; the stitch layer concatenates clips.
    /// No temporal context crosses the boundary, so subjects can drift.
    IndependentClips,
}

/// Per-clip pixel-frame cap for `ltx-video`. Its ceiling is a flat frame
/// count, not a runtime duration, so it does not move with fps.
pub const LTX_VIDEO_FRAMES_PER_CLIP_CAP: u32 = 97;

/// Static chain capability for one model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainCapability {
    /// Per-clip pixel-frame cap at the family's default fps. A chain clip is
    /// an ordinary generation, so this tracks the family's single-request
    /// frame ceiling; for LTX-2 that ceiling is a duration, so callers who
    /// know the clip's fps should use [`frames_per_clip_cap_at_fps`].
    pub frames_per_clip_cap: u32,
    pub carryover: CarryoverKind,
    /// Whether the family has an audio decode path. The chain endpoint
    /// rejects `enable_audio: true` when false.
    pub supports_audio: bool,
    /// Set when the per-clip cap is a runtime duration in seconds rather than
    /// a frame count; `None` means `frames_per_clip_cap` is fps-independent.
    pub runtime_seconds_cap: Option<u32>,
}

/// Capability lookup by canonical family string
/// (`mold_core::manifest::resolve_family`). `None` = not chain-capable.
pub fn capability_for_family(family: &str) -> Option<ChainCapability> {
    match family {
        "ltx2" => Some(ChainCapability {
            frames_per_clip_cap: mold_core::validation::max_frames_for_family("ltx2")
                .unwrap_or(LTX_VIDEO_FRAMES_PER_CLIP_CAP),
            carryover: CarryoverKind::TemporalHandoff,
            supports_audio: true,
            runtime_seconds_cap: mold_core::validation::max_runtime_seconds_for_family("ltx2"),
        }),
        "ltx-video" => Some(ChainCapability {
            frames_per_clip_cap: LTX_VIDEO_FRAMES_PER_CLIP_CAP,
            carryover: CarryoverKind::IndependentClips,
            supports_audio: false,
            runtime_seconds_cap: None,
        }),
        _ => None,
    }
}

/// Per-clip pixel-frame cap for `family` at `fps`. `None` = not chain-capable.
///
/// A chain clip is denoised as one generation, so it is bound by exactly the
/// same ceiling `validate_generate_request` applies to a single request.
pub fn frames_per_clip_cap_at_fps(family: &str, fps: u32) -> Option<u32> {
    let capability = capability_for_family(family)?;
    Some(match capability.runtime_seconds_cap {
        Some(_) => mold_core::validation::ltx2_max_frames_at_fps(fps),
        None => capability.frames_per_clip_cap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx2_has_temporal_handoff_with_audio() {
        // Covers both LTX-2 19B and LTX-2.3 22B — both resolve to
        // family="ltx2" via mold_core::manifest::resolve_family.
        let cap = capability_for_family("ltx2").expect("ltx2 is chain-capable");
        assert_eq!(cap.carryover, CarryoverKind::TemporalHandoff);
        assert!(cap.supports_audio);
        assert_eq!(cap.runtime_seconds_cap, Some(20));
    }

    /// A chain clip is one generation, so its cap must be the same ceiling the
    /// single-request validator applies — including LTX-2's fps dependence.
    /// The old flat 97 silently capped clips at a quarter of the real budget.
    #[test]
    fn ltx2_clip_cap_follows_the_single_request_ceiling() {
        for fps in [8u32, 12, 24, 25, 30] {
            assert_eq!(
                frames_per_clip_cap_at_fps("ltx2", fps),
                mold_core::validation::max_frames_for_family_at_fps("ltx2", fps),
                "clip cap must not diverge from the generate validator at {fps} fps",
            );
        }
        assert!(frames_per_clip_cap_at_fps("ltx2", 24).unwrap() > 97);
    }

    #[test]
    fn ltx_video_is_independent_clips_without_audio() {
        // LTX-Video has no latent handoff: stages render independently and
        // the stitch layer concatenates clips (subjects may drift between
        // clips). It is also video-only.
        let cap = capability_for_family("ltx-video").expect("ltx-video is chain-capable");
        assert_eq!(cap.frames_per_clip_cap, LTX_VIDEO_FRAMES_PER_CLIP_CAP);
        assert_eq!(cap.carryover, CarryoverKind::IndependentClips);
        assert!(!cap.supports_audio);
        assert_eq!(cap.runtime_seconds_cap, None);
    }

    /// LTX-Video's cap is a plain frame count and must stay put when fps moves.
    #[test]
    fn ltx_video_clip_cap_is_fps_independent() {
        for fps in [8u32, 24, 30, 60] {
            assert_eq!(
                frames_per_clip_cap_at_fps("ltx-video", fps),
                Some(LTX_VIDEO_FRAMES_PER_CLIP_CAP),
            );
        }
        assert_eq!(frames_per_clip_cap_at_fps("flux", 24), None);
    }

    #[test]
    fn non_video_families_are_not_chain_capable() {
        for family in ["flux", "sdxl", "sd15", "qwen-image", "zimage", ""] {
            assert!(
                capability_for_family(family).is_none(),
                "{family} must not be chain-capable",
            );
        }
    }
}
