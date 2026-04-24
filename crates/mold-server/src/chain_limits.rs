//! Chain-limits computation for the `/api/capabilities/chain-limits` route.
//!
//! The model's hardcoded per-clip cap is the primary constraint; the
//! hardware-derived recommended value is `min(cap, free_vram_adjusted)` and
//! is inert for distilled LTX-2 today because 97 is model-capped.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

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
}

/// Per-model-family hardcoded caps. Keyed by the family string returned by
/// `mold_core::manifest::resolve_family`.
pub fn family_cap(family: &str) -> Option<u32> {
    match family {
        "ltx2" => Some(97),
        _ => None,
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
    ChainLimits {
        model: model.to_string(),
        frames_per_clip_cap: cap,
        frames_per_clip_recommended: recommended,
        max_stages: MAX_STAGES,
        max_total_frames: cap * MAX_STAGES,
        fade_frames_max: 32,
        transition_modes: vec!["smooth".into(), "cut".into(), "fade".into()],
        quantization_family: quant.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx2_cap_is_97() {
        assert_eq!(family_cap("ltx2"), Some(97));
    }

    #[test]
    fn unknown_family_has_no_cap() {
        assert_eq!(family_cap("flux"), None);
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
    }
}
