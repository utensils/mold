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
    pub frames_per_clip_cap: u32,
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
    /// stages. This is model-specific: an LTX-2 dev checkpoint with its
    /// spatial upscaler selects TwoStage, which render-chain v1 cannot run.
    pub supports_sequence: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_unsupported_reason: Option<String>,
}

/// Per-model-family hardcoded caps. Keyed by the family string returned by
/// `mold_core::manifest::resolve_family`.
pub fn family_cap(family: &str) -> Option<u32> {
    mold_inference::chain::capability_for_family(family).map(|c| c.frames_per_clip_cap)
}

/// Whether a chain-capable family also has an audio path. The chain handler
/// rejects requests with `enable_audio: true` when this returns false, so
/// users get a clear upfront error instead of silently-dropped audio.
pub fn family_supports_audio(family: &str) -> bool {
    mold_inference::chain::capability_for_family(family).is_some_and(|c| c.supports_audio)
}

/// Resolve the sequence renderer's model-specific pipeline gate. LTX-2
/// distilled checkpoints use Distilled, while non-distilled checkpoints use
/// TwoStage when a spatial upscaler is present and OneStage otherwise.
pub fn sequence_support(model: &str, family: &str, has_spatial_upscaler: bool) -> SequenceSupport {
    if family_cap(family).is_none() {
        return SequenceSupport {
            supported: false,
            reason: Some(format!("{family} models do not render video sequences")),
        };
    }
    if family != "ltx2" || model.contains("distilled") || !has_spatial_upscaler {
        return SequenceSupport {
            supported: true,
            reason: None,
        };
    }
    SequenceSupport {
        supported: false,
        reason: Some(
            "This checkpoint selects the two-stage LTX-2 pipeline, which is for single videos. \
             Sequences currently require a distilled LTX-2 checkpoint or a one-stage catalog checkpoint."
                .into(),
        ),
    }
}

/// Compute the chain-limits response for a resolved model name.
///
/// `family` is the canonical family string (e.g. "ltx2").
/// `quant` is the quantization slug ("fp8", "fp16", "q8", ...).
/// `free_vram_bytes` is the current free VRAM on the primary GPU.
pub fn compute_limits(model: &str, family: &str, quant: &str, free_vram_bytes: u64) -> ChainLimits {
    let cap = family_cap(family).unwrap_or(97);
    // Hardware-derived recommended: for distilled LTX-2, 97 is already
    // the binding constraint. Reserve the derivation scaffolding for
    // future non-distilled models.
    let _ = free_vram_bytes; // suppress unused for now; D wires this up
    let recommended = cap;

    const MAX_STAGES: u32 = 16;
    let has_spatial_upscaler = mold_core::manifest::find_manifest(model).is_some_and(|manifest| {
        manifest
            .files
            .iter()
            .any(|file| file.component == mold_core::manifest::ModelComponent::SpatialUpscaler)
    });
    let sequence = sequence_support(model, family, has_spatial_upscaler);
    ChainLimits {
        model: model.to_string(),
        frames_per_clip_cap: cap,
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

    #[test]
    fn ltx2_cap_is_97() {
        // ltx2 family covers both v2 19B and v2.3 22B (dev and distilled);
        // both resolve to family="ltx2" via `resolve_family`.
        assert_eq!(family_cap("ltx2"), Some(97));
    }

    #[test]
    fn ltx_video_cap_is_97() {
        // LTX-Video uses the img2vid-less fallback; same per-clip cap as
        // ltx2 because the chain endpoint stitches independent clips at the
        // pixel level once the cap is hit.
        assert_eq!(family_cap("ltx-video"), Some(97));
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

    #[test]
    fn compute_limits_for_distilled() {
        let lim = compute_limits("ltx-2-19b-distilled:fp8", "ltx2", "fp8", 8_000_000_000);
        assert_eq!(lim.frames_per_clip_cap, 97);
        assert_eq!(lim.frames_per_clip_recommended, 97);
        assert_eq!(lim.max_stages, 16);
        assert_eq!(lim.max_total_frames, 97 * 16);
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
    fn ltx2_dev_with_spatial_upscaler_is_not_sequence_compatible() {
        let support = sequence_support("ltx-2.3-22b-dev:fp8", "ltx2", true);
        assert!(!support.supported);
        assert!(
            support
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("two-stage")),
            "the UI needs an actionable pipeline-specific reason",
        );
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
        let lim = compute_limits("ltx-video-0.9.7-distilled:fp8", "ltx-video", "fp8", 0);
        assert!(
            !lim.supports_audio,
            "ltx-video has no audio path — toggle must stay off",
        );
    }
}
