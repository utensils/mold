//! Chain-limits computation for the `/api/capabilities/chain-limits` route.
//!
//! Two different limits meet here and must not be conflated:
//!
//! * the **family's single-request ceiling** at a given fps
//!   (`family_cap_at_fps` → `mold_core::validation::max_frames_for_family_at_fps`).
//!   LTX-2's is a 20 s runtime budget — 481 frames at 24 fps — and it is what
//!   chain *admission* enforces per stage;
//! * the **model's routing clip size** (`mold_core::chain::routing_clip_frames`),
//!   which is what one generation actually renders when mold auto-chains: 97
//!   for the LTX families, a per-checkpoint VRAM envelope for wan.
//!
//! `frames_per_clip_cap` is the smaller of the two, because that is the clip a
//! client can author and expect to render as one clip. Advertising the family
//! ceiling alone let a Studio composer offer a single 481-frame LTX-2 clip
//! that the one-shot path would have split into five.
//!
//! The hardware-derived recommended value is `min(cap, model default)`, snapped
//! down onto the family's own frame grid.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceSupport {
    pub supported: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChainLimits {
    pub model: String,
    /// Largest clip this model renders as ONE generation at `fps`:
    /// `min(family single-request ceiling at fps, the model's routing clip
    /// size)`.
    ///
    /// It is NOT the family's ceiling. For LTX-2 the routing clip size (97)
    /// normally binds, so this does not move with fps — but the family's 20 s
    /// duration budget still wins wherever it lands below 97 (fps <= 4), which
    /// is why `frames_per_clip_runtime_seconds` stays advertised and why the
    /// value must be read rather than recomputed from either half alone.
    pub frames_per_clip_cap: u32,
    /// fps `frames_per_clip_cap` was computed at (additive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Per-clip runtime budget in seconds when the family's single-request
    /// limit is a duration (additive; currently LTX-2 / LTX-2.3). This is the
    /// family ceiling `seconds * fps + 4`, which chain admission enforces —
    /// it is an upper bound on `frames_per_clip_cap`, never a replacement for
    /// it, because the model's routing clip size usually binds first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames_per_clip_runtime_seconds: Option<u32>,
    pub frames_per_clip_recommended: u32,
    pub max_stages: u32,
    pub max_total_frames: u32,
    pub fade_frames_max: u32,
    pub transition_modes: Vec<String>,
    pub quantization_family: String,
    /// Whether this model's family has an audio decode path. The SPA reads
    /// this to decide whether to show the chain-level "Generate audio"
    /// toggle; the chain endpoint refuses `enable_audio: true` upstream
    /// when this is false. Single source of truth: `mold_inference::chain::capability_for_family`.
    pub supports_audio: bool,
    /// Whether the model's effective runtime pipeline can render sequence
    /// stages. Every LTX-2 pipeline `select_pipeline` chooses now can, so in
    /// practice this tracks whether the family chains at all — it stays
    /// per-model because that is where a future incompatible pipeline would
    /// have to be caught.
    pub supports_sequence: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_unsupported_reason: Option<String>,
}

/// The family's SINGLE-REQUEST ceiling at `fps` — the value chain admission
/// enforces per stage. Keyed by the family string returned by
/// `mold_core::manifest::resolve_family`. `None` = not chain-capable.
///
/// LTX-2's is a runtime duration, so it moves with fps. This is deliberately
/// not what `/api/capabilities/chain-limits` advertises as
/// `frames_per_clip_cap`: see the module doc.
pub fn family_cap_at_fps(family: &str, fps: u32) -> Option<u32> {
    mold_inference::chain::frames_per_clip_cap_at_fps(family, fps)
}

/// `family_cap_at_fps` at the chain default fps. Callers that only need the
/// chain-capable / not-chain-capable answer can use this.
pub fn family_cap(family: &str) -> Option<u32> {
    family_cap_at_fps(family, DEFAULT_CHAIN_FPS)
}

/// fps assumed when a caller has no request-level fps. Matches
/// `mold_core::chain`'s `default_fps`.
pub const DEFAULT_CHAIN_FPS: u32 = 24;

/// Whether a chain-capable family also has an audio path. The chain handler
/// rejects requests with `enable_audio: true` when this returns false, so
/// users get a clear upfront error instead of silently-dropped audio.
pub fn family_supports_audio(family: &str) -> bool {
    mold_inference::chain::capability_for_family(family).is_some_and(|c| c.supports_audio)
}

/// Resolve the sequence renderer's model-specific pipeline gate.
pub fn sequence_support(model: &str, family: &str, has_spatial_upscaler: bool) -> SequenceSupport {
    if family_cap(family).is_none() {
        return SequenceSupport {
            supported: false,
            reason: Some(format!("{family} models do not render video sequences")),
        };
    }
    // Every pipeline `select_pipeline` can choose for an LTX-2 checkpoint now
    // renders sequence clips, two-stage included, so the only thing that
    // decides support is whether the family chains at all. `model` and
    // `has_spatial_upscaler` stay in the signature: they are how a checkpoint
    // is classified, and the seam is worth keeping if a future pipeline is
    // again chain-incapable.
    let _ = (model, has_spatial_upscaler);
    SequenceSupport {
        supported: true,
        reason: None,
    }
}

/// Compute the chain-limits response for a resolved model name.
///
/// `family` is the canonical family string (e.g. "ltx2").
/// `quant` is the quantization slug ("fp8", "fp16", "q8", ...).
/// `default_frames` is the model's own default frame count (manifest or
/// catalog sidecar) and drives the recommended per-clip frames.
/// `fps` is the frame rate the clips will render at. It matters because
/// LTX-2's family ceiling is a runtime duration; the advertised cap only moves
/// with fps where that ceiling drops below the model's routing clip size.
pub fn compute_limits(
    model: &str,
    family: &str,
    quant: &str,
    default_frames: Option<u32>,
    fps: Option<u32>,
) -> ChainLimits {
    let fps = fps.filter(|value| *value > 0).unwrap_or(DEFAULT_CHAIN_FPS);
    // The advertised per-clip cap is the size ONE generation renders, which is
    // the tighter of two independent limits:
    //
    //   * the family's single-request ceiling at this fps — LTX-2's is a 20 s
    //     runtime budget, so it moves with fps and is grid-snapped;
    //   * the model's own routing clip size — the VRAM envelope / shipped clip
    //     default the one-shot auto-chain router splits work into.
    //
    // Advertising only the family ceiling let a composer author a single
    // 481-frame LTX-2 clip that `mold run` would have rendered as five.
    let family_cap = family_cap_at_fps(family, fps).unwrap_or(97);
    let cap = mold_core::chain::routing_clip_frames(family, model)
        .map_or(family_cap, |routing| family_cap.min(routing));
    // Recommend the model's own default frame count (LTX-Video ships 25,
    // LTX-2 ships 97, wan 81 or 121) so new clips start at what the model
    // actually runs; clamp to the advertised per-clip cap and snap down onto
    // the family's own grid. Without a model default, fall back to the cap.
    //
    // The grid comes from the family, never a constant: wan is `4k+1` where
    // the LTX families are `8k+1`, so a hardcoded 8 recommended an off-grid
    // clip count that the validator then rejected with a 422.
    let step = mold_core::validation::frame_step_for_family(family).unwrap_or(8);
    let offset = mold_core::validation::frame_offset_for_family(family).unwrap_or(1);
    let snap_down = |frames: u32| {
        if frames <= offset {
            return frames;
        }
        frames - ((frames - offset) % step)
    };
    // The floor is the first on-grid clip at or above one step — 9 on the LTX
    // grid, 5 on wan's — so it cannot itself be off-grid.
    let floor = step + offset;
    let recommended = default_frames
        .map(|frames| snap_down(frames.min(cap)))
        .unwrap_or(cap)
        .max(floor)
        .min(cap);

    const MAX_STAGES: u32 = 16;
    let canonical_model = mold_core::manifest::resolve_model_name(model);
    let has_spatial_upscaler =
        mold_core::manifest::find_manifest(&canonical_model).is_some_and(|manifest| {
            manifest
                .files
                .iter()
                .any(|file| file.component == mold_core::manifest::ModelComponent::SpatialUpscaler)
        });
    let sequence = sequence_support(&canonical_model, family, has_spatial_upscaler);
    ChainLimits {
        model: model.to_string(),
        frames_per_clip_cap: cap,
        fps: Some(fps),
        frames_per_clip_runtime_seconds: mold_inference::chain::capability_for_family(family)
            .and_then(|capability| capability.runtime_seconds_cap),
        frames_per_clip_recommended: recommended,
        max_stages: MAX_STAGES,
        max_total_frames: cap * MAX_STAGES,
        fade_frames_max: 32,
        transition_modes: vec!["smooth".into(), "cut".into(), "fade".into()],
        quantization_family: quant.to_string(),
        supports_audio: family_supports_audio(family),
        supports_sequence: sequence.supported,
        sequence_unsupported_reason: sequence.reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An LTX-2 clip is denoised as one generation, so its cap is the family's
    /// single-request ceiling — a 20s duration, not the old flat 97.
    #[test]
    fn ltx2_cap_follows_the_duration_budget() {
        // ltx2 family covers both v2 19B and v2.3 22B (dev and distilled);
        // both resolve to family="ltx2" via `resolve_family`.
        assert_eq!(
            family_cap("ltx2"),
            mold_core::validation::max_frames_for_family_at_fps("ltx2", DEFAULT_CHAIN_FPS),
        );
        assert_eq!(family_cap_at_fps("ltx2", 12), Some(241));
        assert_eq!(family_cap_at_fps("ltx2", 24), Some(481));
    }

    #[test]
    fn ltx_video_cap_is_97_at_every_fps() {
        // LTX-Video uses the img2vid-less fallback and publishes a flat frame
        // ceiling rather than a duration, so fps must not move its cap.
        assert_eq!(family_cap("ltx-video"), Some(97));
        assert_eq!(family_cap_at_fps("ltx-video", 12), Some(97));
        assert_eq!(family_cap_at_fps("ltx-video", 30), Some(97));
    }

    #[test]
    fn audio_capability_is_ltx2_only() {
        // Only the LTX-2 / LTX-2.3 AV transformer has an audio decode path.
        // LTX-Video is video-only; FLUX/SDXL aren't in chain support at all.
        assert!(family_supports_audio("ltx2"));
        assert!(!family_supports_audio("ltx-video"));
        assert!(!family_supports_audio("flux"));
        assert!(!family_supports_audio(""));
    }

    #[test]
    fn unknown_family_has_no_cap() {
        assert_eq!(family_cap("flux"), None);
        assert_eq!(family_cap("sdxl"), None);
    }

    /// The recommended per-clip frames must follow the model's own default
    /// frame count (LTX-Video ships 25, LTX-2 ships 97) instead of the
    /// family cap, so new clips default to what the model actually runs.
    #[test]
    fn recommended_uses_model_default_frames() {
        let ltx_video = compute_limits("ltx-video-0.9.6:bf16", "ltx-video", "bf16", Some(25), None);
        assert_eq!(ltx_video.frames_per_clip_cap, 97);
        assert_eq!(ltx_video.frames_per_clip_recommended, 25);

        let ltx2 = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", Some(97), None);
        assert_eq!(ltx2.frames_per_clip_recommended, 97);

        // No model default → fall back to the advertised per-clip cap.
        let unknown = compute_limits("cv:123", "ltx2", "", None, None);
        assert_eq!(
            unknown.frames_per_clip_recommended,
            unknown.frames_per_clip_cap,
        );
        assert_eq!(
            unknown.frames_per_clip_cap,
            mold_core::chain::LTX2_DEFAULT_CLIP_FRAMES,
        );

        // Off-grid defaults snap DOWN onto the 8n+1 grid.
        let off_grid = compute_limits("cv:456", "ltx-video", "", Some(30), None);
        assert_eq!(off_grid.frames_per_clip_recommended, 25);

        // Defaults above the cap clamp to the cap. The binding cap here is the
        // per-model routing clip size, not the family's 241-frame duration
        // budget at 12 fps.
        let oversized = compute_limits("cv:789", "ltx2", "", Some(500), Some(12));
        assert_eq!(oversized.frames_per_clip_cap, 97);
        assert_eq!(oversized.frames_per_clip_recommended, 97);
    }

    /// The advertised per-clip cap is the size ONE generation renders, not the
    /// family's single-request ceiling.
    ///
    /// The duration-budget work re-derived this from the family budget, so
    /// `ltx-2-19b-distilled:fp8` advertised 481 frames at 24 fps — a clip the
    /// one-shot auto-chain router would have split into five, and which the
    /// Studio composers happily offered as one clip.
    #[test]
    fn ltx2_clip_cap_is_the_per_model_routing_size_not_the_family_budget() {
        for fps in [12u32, 16, 24, 30] {
            let limits = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(fps));
            assert_eq!(
                limits.frames_per_clip_cap,
                mold_core::chain::LTX2_DEFAULT_CLIP_FRAMES,
                "ltx2 @ {fps} fps must advertise the routing clip size",
            );
            assert!(
                limits.frames_per_clip_cap < family_cap_at_fps("ltx2", fps).unwrap(),
                "the routing clip size must stay under the family budget at {fps} fps",
            );
            // The duration budget stays advertised so clients can still tell
            // the two apart.
            assert_eq!(limits.frames_per_clip_runtime_seconds, Some(20));
            assert_eq!(limits.fps, Some(fps));
        }
    }

    /// Where the family's duration budget is SMALLER than the routing clip
    /// size, the budget still binds — and the advertised value stays on the
    /// family's own grid so a client clamped to it is submittable.
    #[test]
    fn a_duration_budget_below_the_routing_size_still_binds() {
        // 20 s at 1 fps is 24 raw frames, which snaps to 17 on the 8k+1 grid.
        let limits = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(1));
        assert_eq!(limits.frames_per_clip_cap, 17);
        assert_eq!(
            limits.frames_per_clip_cap,
            family_cap_at_fps("ltx2", 1).unwrap(),
        );
        assert_eq!((limits.frames_per_clip_cap - 1) % 8, 0);
        assert!(limits.frames_per_clip_recommended <= limits.frames_per_clip_cap);
    }

    /// Wan's routing clip size is per checkpoint: the two-expert A14B pair
    /// measures near the 24 GB envelope well before the single-expert 5B does.
    #[test]
    fn wan_clip_cap_is_the_checkpoints_own_routing_size() {
        let ti2v = compute_limits("wan22-ti2v-5b:fp16", "wan", "fp16", Some(121), Some(24));
        assert_eq!(ti2v.frames_per_clip_cap, 121);
        assert_eq!(ti2v.frames_per_clip_recommended, 121);

        // The A14B tiers record their own measured envelope (81 for Q5/Q4,
        // 73 for Q8), which is above the 53-frame family floor.
        let a14b = compute_limits("wan22-t2v-a14b:q5", "wan", "q5", Some(81), Some(16));
        assert_eq!(a14b.frames_per_clip_cap, 81);
        assert_eq!(a14b.frames_per_clip_recommended, 81);

        // An opaque catalog A14B checkpoint has no manifest to read, so the
        // floor is the answer.
        let opaque = compute_limits("cv:900-a14b", "wan", "", Some(257), Some(16));
        assert_eq!(opaque.frames_per_clip_cap, 53);
        assert!(opaque.frames_per_clip_recommended <= 53);
        // Both sit on wan's own 4k+1 grid.
        for limits in [&ti2v, &a14b, &opaque] {
            assert_eq!((limits.frames_per_clip_cap - 1) % 4, 0);
        }
    }

    /// LTX-Video's chain capability already publishes a flat 97-frame clip
    /// cap, so the routing size cannot move it.
    #[test]
    fn ltx_video_clip_cap_stays_97() {
        for fps in [12u32, 24, 30] {
            let limits =
                compute_limits("ltx-video-0.9.6:bf16", "ltx-video", "bf16", None, Some(fps));
            assert_eq!(limits.frames_per_clip_cap, 97);
        }
    }

    /// The CLI's auto-chain router and this endpoint must not drift: they read
    /// the same `mold_core::chain` authority.
    #[test]
    fn advertised_cap_matches_the_routers_routing_clip_size() {
        for (model, family, fps) in [
            ("ltx-2-19b-distilled:fp8", "ltx2", 24u32),
            ("ltx-2.3-22b-dev:fp8", "ltx2", 24),
            ("cv:3143864", "ltx2", 24),
            ("wan22-ti2v-5b:fp16", "wan", 24),
            ("wan22-t2v-a14b:q5", "wan", 16),
            ("ltx-video-0.9.6:bf16", "ltx-video", 30),
        ] {
            let limits = compute_limits(model, family, "", None, Some(fps));
            let routing = mold_core::chain::routing_clip_frames(family, model).unwrap();
            let family_cap = family_cap_at_fps(family, fps).unwrap();
            assert_eq!(
                limits.frames_per_clip_cap,
                routing.min(family_cap),
                "{model} @ {fps} fps",
            );
            assert_eq!(limits.max_total_frames, limits.frames_per_clip_cap * 16);
            assert!(limits.frames_per_clip_recommended <= limits.frames_per_clip_cap);
        }
    }

    /// `mold-core` cannot depend on `mold-inference`, so LTX-Video's clip size
    /// exists in both crates. Pin them together: a drift would advertise a cap
    /// admission does not enforce.
    #[test]
    fn ltx_video_routing_size_matches_the_engines_clip_cap() {
        assert_eq!(
            mold_core::chain::LTX_VIDEO_DEFAULT_CLIP_FRAMES,
            mold_inference::chain::LTX_VIDEO_FRAMES_PER_CLIP_CAP,
        );
    }

    /// The 1552-frame total the web fallback still carries is 97 × 16 — it is
    /// the per-model clip size times the stage cap, not the family budget.
    #[test]
    fn ltx2_max_total_frames_is_the_routing_size_times_the_stage_cap() {
        let limits = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(24));
        assert_eq!(limits.max_stages, 16);
        assert_eq!(limits.max_total_frames, 1552);
    }

    /// The recommendation must land on the family's own grid (#783).
    ///
    /// Wan is `4k+1` where the LTX families are `8k+1`. The snap was a
    /// hardcoded 8, so wan's shipped defaults came back off-grid and a client
    /// that started a clip at the recommendation got a 422 from the validator
    /// that owns the real rule.
    #[test]
    fn recommended_clip_frames_land_on_the_family_grid() {
        let step = mold_core::validation::frame_step_for_family("wan").unwrap();
        let offset = mold_core::validation::frame_offset_for_family("wan").unwrap();
        assert_eq!((step, offset), (4, 1));

        for (model, default_frames) in [
            ("wan21-t2v-1.3b:bf16", 81u32),
            ("wan22-ti2v-5b:fp16", 121),
            ("wan22-t2v-a14b:q5", 53),
        ] {
            let limits = compute_limits(model, "wan", "q5", Some(default_frames), Some(16));
            assert_eq!(
                limits.frames_per_clip_recommended, default_frames,
                "{model}: an on-grid default must survive untouched",
            );
        }

        // Off-grid defaults snap DOWN onto 4k+1, not onto 8k+1: 80 -> 77 here,
        // where the old hardcoded step produced 73.
        let off_grid = compute_limits("cv:900", "wan", "", Some(80), Some(16));
        assert_eq!(off_grid.frames_per_clip_recommended, 77);

        // Every recommendation is submittable, at every grid this covers.
        for (family, fps, default_frames) in [
            ("wan", 16, Some(80u32)),
            ("wan", 24, Some(121)),
            ("wan", 16, None),
            ("ltx-video", 24, Some(30)),
            ("ltx2", 24, None),
        ] {
            let limits = compute_limits("m", family, "", default_frames, Some(fps));
            let step = mold_core::validation::frame_step_for_family(family).unwrap();
            let offset = mold_core::validation::frame_offset_for_family(family).unwrap();
            assert_eq!(
                (limits.frames_per_clip_recommended - offset) % step,
                0,
                "{family} @ {fps}fps recommended {} off the {step}k+{offset} grid",
                limits.frames_per_clip_recommended,
            );
            assert!(limits.frames_per_clip_recommended <= limits.frames_per_clip_cap);
        }
    }

    /// The response must echo the fps it used and carry the duration budget so
    /// clients can still tell the per-model clip size and the family's
    /// single-request ceiling apart.
    #[test]
    fn compute_limits_advertises_the_fps_it_used() {
        let at_24 = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(24));
        assert_eq!(at_24.fps, Some(24));
        assert_eq!(at_24.frames_per_clip_cap, 97);
        assert_eq!(at_24.frames_per_clip_runtime_seconds, Some(20));

        let at_12 = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(12));
        assert_eq!(at_12.frames_per_clip_cap, 97);

        // A zero/absent fps falls back to the chain default rather than
        // collapsing the cap to a single frame.
        let fallback = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(0));
        assert_eq!(fallback.fps, Some(DEFAULT_CHAIN_FPS));
        assert_eq!(fallback.frames_per_clip_cap, 97);

        // ltx-video has no duration budget to advertise.
        let video = compute_limits("ltx-video-0.9.6:bf16", "ltx-video", "bf16", None, Some(30));
        assert_eq!(video.frames_per_clip_runtime_seconds, None);
        assert_eq!(video.frames_per_clip_cap, 97);
    }

    #[test]
    fn compute_limits_for_distilled() {
        let lim = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, None);
        let cap = mold_core::chain::LTX2_DEFAULT_CLIP_FRAMES;
        assert_eq!(lim.frames_per_clip_cap, cap);
        assert_eq!(lim.frames_per_clip_recommended, cap);
        assert_eq!(lim.max_stages, 16);
        assert_eq!(lim.max_total_frames, cap * 16);
        assert_eq!(
            lim.transition_modes,
            vec!["smooth".to_string(), "cut".into(), "fade".into()]
        );
        assert!(
            lim.supports_audio,
            "ltx2 family has the AV transformer + audio VAE / vocoder path",
        );
        assert!(
            lim.supports_sequence,
            "the distilled pipeline is supported by sequence rendering",
        );
        assert!(lim.sequence_unsupported_reason.is_none());
    }

    #[test]
    fn ltx2_dev_with_spatial_upscaler_is_sequence_compatible() {
        // A dev checkpoint plus a spatial upscaler selects the two-stage
        // pipeline, which now renders sequence clips.
        let support = sequence_support("ltx-2.3-22b-dev:fp8", "ltx2", true);
        assert!(support.supported);
        assert!(support.reason.is_none());
    }

    #[test]
    fn non_chain_families_still_report_an_actionable_reason() {
        // The seam has to stay alive: it is the only thing that tells the UI
        // why a Sequence picker is empty for a still-image family.
        let support = sequence_support("flux-dev:q4", "flux", false);
        assert!(!support.supported);
        assert!(
            support
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("flux")),
            "the reason must name the family, got: {:?}",
            support.reason,
        );
    }

    #[test]
    fn ltx2_dev_legacy_alias_is_sequence_compatible() {
        let limits = compute_limits("ltx-2.3-22b-dev-fp8", "ltx2", "fp8", None, None);
        assert!(limits.supports_sequence);
        assert!(limits.sequence_unsupported_reason.is_none());
    }

    #[test]
    fn ltx2_catalog_checkpoint_without_upscaler_uses_supported_one_stage_pipeline() {
        let support = sequence_support("cv:3143864", "ltx2", false);
        assert!(support.supported);
        assert!(support.reason.is_none());
    }

    #[test]
    fn compute_limits_for_ltx_video_has_no_audio() {
        // LTX-Video is video-only; the SPA must hide the audio toggle and the
        // chain endpoint will reject `enable_audio: true` upstream regardless.
        let lim = compute_limits(
            "ltx-video-0.9.7-distilled:fp8",
            "ltx-video",
            "fp8",
            None,
            None,
        );
        assert!(
            !lim.supports_audio,
            "ltx-video has no audio path — toggle must stay off",
        );
    }
}
