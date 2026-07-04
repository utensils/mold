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

/// Static chain capability for one model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainCapability {
    /// Hardcoded per-clip pixel-frame cap (model constraint, not hardware).
    pub frames_per_clip_cap: u32,
    pub carryover: CarryoverKind,
    /// Whether the family has an audio decode path. The chain endpoint
    /// rejects `enable_audio: true` when false.
    pub supports_audio: bool,
}

/// Capability lookup by canonical family string
/// (`mold_core::manifest::resolve_family`). `None` = not chain-capable.
pub fn capability_for_family(family: &str) -> Option<ChainCapability> {
    match family {
        "ltx2" => Some(ChainCapability {
            frames_per_clip_cap: 97,
            carryover: CarryoverKind::TemporalHandoff,
            supports_audio: true,
        }),
        "ltx-video" => Some(ChainCapability {
            frames_per_clip_cap: 97,
            carryover: CarryoverKind::IndependentClips,
            supports_audio: false,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx2_has_temporal_handoff_with_audio() {
        // Covers both LTX-2 19B and LTX-2.3 22B — both resolve to
        // family="ltx2" via mold_core::manifest::resolve_family.
        let cap = capability_for_family("ltx2").expect("ltx2 is chain-capable");
        assert_eq!(cap.frames_per_clip_cap, 97);
        assert_eq!(cap.carryover, CarryoverKind::TemporalHandoff);
        assert!(cap.supports_audio);
    }

    #[test]
    fn ltx_video_is_independent_clips_without_audio() {
        // LTX-Video has no latent handoff: stages render independently and
        // the stitch layer concatenates clips (subjects may drift between
        // clips). It is also video-only.
        let cap = capability_for_family("ltx-video").expect("ltx-video is chain-capable");
        assert_eq!(cap.frames_per_clip_cap, 97);
        assert_eq!(cap.carryover, CarryoverKind::IndependentClips);
        assert!(!cap.supports_audio);
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
