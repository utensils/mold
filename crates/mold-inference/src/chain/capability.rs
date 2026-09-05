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

/// Wan's per-clip pixel-frame cap.
///
/// A chain clip is one ordinary generation, so the cap is the family's own
/// single-request ceiling — the flat global maximum, which sits on the `4k+1`
/// grid so a client clamping to it can still submit it. Unlike LTX-2 this is
/// not a duration: wan's temporal RoPE is indexed by latent frame against a
/// 1024-entry table, so memory binds long before time does.
pub const WAN_FRAMES_PER_CLIP_CAP: u32 = mold_core::validation::MAX_FRAMES_GLOBAL;

/// How a wan checkpoint carries context across a clip boundary.
///
/// Wan has **no latent motion tail**. The natural smooth handoff is last-frame
/// *image* conditioning, which only an image-conditioned checkpoint can accept
/// — so chain eligibility rides the checkpoint, never the family. A 1.3B or
/// A14B T2V checkpoint has no conditioning channel at all and can only ever
/// concatenate independent clips.
///
/// The classification is not a second opinion: it is a total function of the
/// source-image contract `wan::pipeline::source_image_capability` already
/// derives from the checkpoint's own headers, so the two cannot drift.
///
/// - `Required` — the 36-channel mask+latent concat (A14B I2V). Clip N's last
///   frame seeds clip N+1.
/// - `Optional` — the TI2V-5B latent inpaint. Leading latent frames are pinned
///   to the re-encoded tail of the previous clip.
/// - `Unsupported` — text-to-video only. Independent clips, the honest
///   `ltx-video` precedent: Cut and Crossfade, never a fake "Continue motion".
pub fn wan_carryover(source_image: mold_core::SourceImageCapability) -> CarryoverKind {
    match source_image {
        mold_core::SourceImageCapability::Required | mold_core::SourceImageCapability::Optional => {
            CarryoverKind::TemporalHandoff
        }
        mold_core::SourceImageCapability::Unsupported => CarryoverKind::IndependentClips,
    }
}

/// Chain capability for one concrete model.
///
/// Families whose every checkpoint behaves alike fall through to
/// [`capability_for_family`]. Wan does not: `source_image` carries the
/// checkpoint's own conditioning contract (from `/api/models`, or from the
/// engine's header read), and `None` means it has not been classified — which
/// must be read as "unknown", so the conservative independent-clip carryover
/// applies rather than a handoff the checkpoint may not support.
pub fn capability_for_model(
    family: &str,
    source_image: Option<mold_core::SourceImageCapability>,
) -> Option<ChainCapability> {
    let base = capability_for_family(family)?;
    if family != "wan" {
        return Some(base);
    }
    Some(ChainCapability {
        carryover: source_image.map_or(CarryoverKind::IndependentClips, wan_carryover),
        ..base
    })
}

/// Capability lookup by canonical family string
/// (`mold_core::manifest::resolve_family`). `None` = not chain-capable.
///
/// For wan this is the **floor**, not the answer: the family-level view cannot
/// know which checkpoint is selected, so it reports the carryover every wan
/// checkpoint can honour. Callers holding a concrete model must use
/// [`capability_for_model`].
pub fn capability_for_family(family: &str) -> Option<ChainCapability> {
    match family {
        "wan" => Some(ChainCapability {
            frames_per_clip_cap: WAN_FRAMES_PER_CLIP_CAP,
            carryover: CarryoverKind::IndependentClips,
            // The family has no audio decode path at all — asked of the one
            // core authority so a chain and a one-shot cannot disagree about
            // which families render sound.
            supports_audio: mold_core::generation_profile::family_emits_audio("wan"),
            runtime_seconds_cap: None,
        }),
        "ltx2" => Some(ChainCapability {
            frames_per_clip_cap: mold_core::validation::max_frames_for_family("ltx2")
                .unwrap_or(LTX_VIDEO_FRAMES_PER_CLIP_CAP),
            carryover: CarryoverKind::TemporalHandoff,
            supports_audio: mold_core::generation_profile::family_emits_audio("ltx2"),
            runtime_seconds_cap: mold_core::validation::max_runtime_seconds_for_family("ltx2"),
        }),
        "ltx-video" => Some(ChainCapability {
            frames_per_clip_cap: LTX_VIDEO_FRAMES_PER_CLIP_CAP,
            carryover: CarryoverKind::IndependentClips,
            supports_audio: mold_core::generation_profile::family_emits_audio("ltx-video"),
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
        // Grid-snapped: a clip cap that is not itself a valid frame count
        // sends a client that clamps to it straight into a 422.
        Some(_) => mold_core::validation::ltx2_max_frames_on_grid_at_fps(fps),
        None => capability.frames_per_clip_cap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wan's chain carryover is a property of the checkpoint, not the family.
    ///
    /// The family view cannot know which checkpoint is selected, so it must
    /// report the carryover *every* wan checkpoint can honour — a T2V-only
    /// 1.3B has no conditioning channel at all, so promising a handoff at the
    /// family level would offer "Continue motion" on a model that concatenates
    /// unrelated clips.
    #[test]
    fn wan_carryover_follows_the_checkpoint_not_the_family() {
        use mold_core::SourceImageCapability::{Optional, Required, Unsupported};

        let family = capability_for_family("wan").expect("wan is chain-capable");
        assert_eq!(family.carryover, CarryoverKind::IndependentClips);
        assert!(!family.supports_audio, "wan has no audio decode path");
        assert_eq!(
            family.runtime_seconds_cap, None,
            "wan's cap is not a duration"
        );

        // The two image-conditioned contracts carry context; text-to-video
        // cannot. This is the same classification `/api/models` advertises as
        // `source_image`, so a checkpoint can never offer a seam its
        // conditioning contract would reject.
        assert_eq!(wan_carryover(Optional), CarryoverKind::TemporalHandoff);
        assert_eq!(wan_carryover(Required), CarryoverKind::TemporalHandoff);
        assert_eq!(wan_carryover(Unsupported), CarryoverKind::IndependentClips);

        for (source, expected) in [
            (Some(Optional), CarryoverKind::TemporalHandoff),
            (Some(Required), CarryoverKind::TemporalHandoff),
            (Some(Unsupported), CarryoverKind::IndependentClips),
            // Unclassified is "unknown", never an assumed handoff.
            (None, CarryoverKind::IndependentClips),
        ] {
            assert_eq!(
                capability_for_model("wan", source).unwrap().carryover,
                expected,
                "{source:?}"
            );
        }

        // Other families ignore the argument entirely — their checkpoints do
        // not vary this way, and passing a wan-shaped hint must not rewrite
        // LTX-2's real motion-tail handoff.
        for source in [None, Some(Unsupported), Some(Required)] {
            assert_eq!(
                capability_for_model("ltx2", source).unwrap().carryover,
                CarryoverKind::TemporalHandoff,
            );
            assert_eq!(
                capability_for_model("ltx-video", source).unwrap().carryover,
                CarryoverKind::IndependentClips,
            );
        }
        assert!(capability_for_model("flux", None).is_none());
    }

    /// A chain clip is one generation, so wan's per-clip cap is exactly the
    /// ceiling `validate_generate_request` applies to a single wan request —
    /// and unlike LTX-2's it does not move with fps.
    #[test]
    fn wan_clip_cap_is_the_single_request_ceiling_at_every_fps() {
        assert_eq!(
            WAN_FRAMES_PER_CLIP_CAP,
            mold_core::validation::max_frames_for_family("wan").unwrap(),
        );
        // On the family's own 4k+1 grid, so a client clamping to the
        // advertised maximum can still submit it.
        assert_eq!((WAN_FRAMES_PER_CLIP_CAP - 1) % 4, 0);
        for fps in [16, 24, 30] {
            assert_eq!(
                frames_per_clip_cap_at_fps("wan", fps),
                Some(WAN_FRAMES_PER_CLIP_CAP),
                "wan's cap is a frame count, not a duration",
            );
        }
    }

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
