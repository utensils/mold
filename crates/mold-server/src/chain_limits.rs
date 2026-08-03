//! Chain-limits computation for the `/api/capabilities/chain-limits` route.
//!
//! The model's hardcoded per-clip cap is the primary constraint; the
//! hardware-derived recommended value is `min(cap, free_vram_adjusted)` and
//! is inert for distilled LTX-2 today because 97 is model-capped.

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
    /// Per-clip cap at `fps`. For families with a `frames_per_clip_runtime_seconds`
    /// budget this is derived, not fixed — recompute it when the user changes fps.
    pub frames_per_clip_cap: u32,
    /// fps `frames_per_clip_cap` was computed at (additive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Per-clip runtime budget in seconds when the family's real limit is a
    /// duration (additive; currently LTX-2 / LTX-2.3). Clients derive the cap
    /// at another fps as `seconds * fps + 4`, clamped by the server.
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

/// Per-clip cap for a family at `fps`. Keyed by the family string returned by
/// `mold_core::manifest::resolve_family`. `None` = not chain-capable.
///
/// LTX-2's per-clip cap is a runtime duration, so it moves with fps; passing
/// the clip's real fps is what lets a sequence use clips longer than the old
/// flat 97.
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
/// `fps` is the frame rate the clips will render at; LTX-2's per-clip cap is
/// a runtime duration, so the advertised cap moves with it.
pub fn compute_limits(
    model: &str,
    family: &str,
    quant: &str,
    default_frames: Option<u32>,
    fps: Option<u32>,
) -> ChainLimits {
    let fps = fps.filter(|value| *value > 0).unwrap_or(DEFAULT_CHAIN_FPS);
    let cap = family_cap_at_fps(family, fps).unwrap_or(97);
    // Recommend the model's own default frame count (LTX-Video ships 25,
    // LTX-2 ships 97) so new clips start at what the model actually runs;
    // clamp to the family cap and snap down onto the 8n+1 grid. Without a
    // model default, fall back to the cap (old behavior).
    let recommended = default_frames
        .map(|frames| {
            let clamped = frames.min(cap);
            if clamped > 1 {
                clamped - ((clamped - 1) % 8)
            } else {
                clamped
            }
        })
        .unwrap_or(cap)
        .max(9)
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

        // No model default → fall back to the family cap (old behavior).
        let unknown = compute_limits("cv:123", "ltx2", "", None, None);
        assert_eq!(
            unknown.frames_per_clip_recommended,
            family_cap("ltx2").unwrap()
        );

        // Off-grid defaults snap DOWN onto the 8n+1 grid.
        let off_grid = compute_limits("cv:456", "ltx-video", "", Some(30), None);
        assert_eq!(off_grid.frames_per_clip_recommended, 25);

        // Defaults above the cap clamp to the cap — 500 is over budget at the
        // chain default 24 fps only once the absolute guard is applied, so use
        // a low fps where the duration budget clearly binds.
        let oversized = compute_limits("cv:789", "ltx2", "", Some(500), Some(12));
        assert_eq!(oversized.frames_per_clip_cap, 241);
        assert_eq!(oversized.frames_per_clip_recommended, 241);
    }

    /// The advertised cap must move with the fps the clips will render at, and
    /// carry the duration budget so clients can recompute it themselves.
    #[test]
    fn compute_limits_advertises_the_fps_it_used() {
        let at_24 = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(24));
        assert_eq!(at_24.fps, Some(24));
        assert_eq!(at_24.frames_per_clip_cap, 481);
        assert_eq!(at_24.frames_per_clip_runtime_seconds, Some(20));

        let at_12 = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(12));
        assert_eq!(at_12.frames_per_clip_cap, 241);

        // A zero/absent fps falls back to the chain default rather than
        // collapsing the cap to a single frame.
        let fallback = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, Some(0));
        assert_eq!(fallback.fps, Some(DEFAULT_CHAIN_FPS));
        assert_eq!(fallback.frames_per_clip_cap, 481);

        // ltx-video has no duration budget to advertise.
        let video = compute_limits("ltx-video-0.9.6:bf16", "ltx-video", "bf16", None, Some(30));
        assert_eq!(video.frames_per_clip_runtime_seconds, None);
        assert_eq!(video.frames_per_clip_cap, 97);
    }

    #[test]
    fn compute_limits_for_distilled() {
        let lim = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", None, None);
        let cap = family_cap("ltx2").unwrap();
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
