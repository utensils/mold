//! LTX-2 checkpoint-generation identity and image-conditioning
//! preprocessing profiles.
//!
//! Upstream LTX-2 re-compresses every still-image conditioning input
//! through a one-frame H.264/YUV420 round-trip so the tensor the VAE sees
//! matches the compressed-video-frame distribution the model was trained
//! on. The compression level is a property of the checkpoint *generation*
//! (upstream `ltx_pipelines/utils/constants.py`: CRF 33 for LTX-2/2.3,
//! CRF 18 for the future 2.4), so the profile must be resolved from
//! authoritative model identity and must fail closed for generations this
//! build does not know — a guessed CRF silently degrades conditioning.
//!
//! This module is the single authority shared by admission, the engine,
//! and diagnostics; `mold-inference`'s `preset_for_model_with_hint`
//! delegates its 2.0-vs-2.3 split here (a contract test pins the two
//! together).

use serde::{Deserialize, Serialize};

/// A known LTX-2 checkpoint generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ltx2Generation {
    /// LTX-2 2.0/2.1/2.2 (the 19B lineage).
    V2,
    /// LTX-2.3 (the 22B lineage).
    V2_3,
}

impl Ltx2Generation {
    pub fn label(self) -> &'static str {
        match self {
            Self::V2 => "LTX-2",
            Self::V2_3 => "LTX-2.3",
        }
    }
}

/// The still-image conditioning preprocessing contract for a checkpoint
/// generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ltx2ImagePreprocessingProfile {
    pub generation: Ltx2Generation,
    /// H.264 compression level the conditioning image is round-tripped
    /// at, matching upstream's per-generation CRF constant. `0` means
    /// lossless/no round-trip (upstream parity).
    pub image_crf: u8,
}

/// Upstream `DEFAULT_IMAGE_CRF` — LTX-2 and LTX-2.3 were both trained
/// against CRF 33 conditioning (constants.py:36, 80-88 @ fd4ded7).
const LTX2_IMAGE_CRF: u8 = 33;

/// Resolve the checkpoint generation from the model name and, failing
/// that, the safetensors `__metadata__.model_version` header hint.
///
/// Returns `None` for anything unrecognised — including a future
/// `model_version: "2.4.x"`, which upstream maps to CRF 18 but which this
/// build has no runnable preset for. Callers that need a preprocessing
/// profile must treat `None` as fail-closed rather than guessing.
pub fn ltx2_generation(
    model_name: &str,
    model_version_hint: Option<&str>,
) -> Option<Ltx2Generation> {
    // A name-embedded version marker (`ltx-2`, `ltx-2.3`, `ltx2.3`, …) is
    // parsed as a COMPLETE minor component, never an unrestricted
    // substring: `ltx-2.30` names an unknown generation and must fail
    // closed rather than be swallowed by an `ltx-2.3` prefix match, and
    // an explicit `ltx-2.0`/`ltx-2.1`/`ltx-2.2` is the supported V2
    // lineage rather than an unknown future one.
    for marker in ["ltx-2", "ltx2"] {
        let Some(idx) = model_name.find(marker) else {
            continue;
        };
        let rest = &model_name[idx + marker.len()..];
        let Some(after_dot) = rest.strip_prefix('.') else {
            // Bare `ltx-2-19b` style names are the V2 lineage; a bare
            // `ltx2` (no dash, no version) is too weak a marker — fall
            // through to the metadata hint.
            if marker == "ltx-2" {
                return Some(Ltx2Generation::V2);
            }
            continue;
        };
        let digits: String = after_dot
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            // `ltx-2.` followed by a non-digit — treat the dot as part of
            // the surrounding name, not a version separator.
            if marker == "ltx-2" {
                return Some(Ltx2Generation::V2);
            }
            continue;
        }
        return generation_for_minor(digits.parse().ok()?);
    }
    let version = model_version_hint?;
    let rest = version.strip_prefix("2.")?;
    let digits = rest.split(['.', '-', '+']).next().unwrap_or("");
    generation_for_minor(digits.parse().ok()?)
}

/// The generation a `2.<minor>` version component belongs to. Unknown
/// minors (2.4 and beyond) return `None` — fail closed.
fn generation_for_minor(minor: u32) -> Option<Ltx2Generation> {
    match minor {
        0..=2 => Some(Ltx2Generation::V2),
        3 => Some(Ltx2Generation::V2_3),
        _ => None,
    }
}

/// The preprocessing profile for a known generation. Both currently
/// supported generations were trained against CRF 33.
pub fn ltx2_image_preprocessing_profile(
    generation: Ltx2Generation,
) -> Ltx2ImagePreprocessingProfile {
    Ltx2ImagePreprocessingProfile {
        generation,
        image_crf: LTX2_IMAGE_CRF,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx2_generation_resolves_known_names_and_hints() {
        assert_eq!(ltx2_generation("ltx-2-19b", None), Some(Ltx2Generation::V2));
        assert_eq!(
            ltx2_generation("ltx-2-19b-distilled:fp8", None),
            Some(Ltx2Generation::V2)
        );
        assert_eq!(
            ltx2_generation("ltx-2.3-22b", None),
            Some(Ltx2Generation::V2_3)
        );
        assert_eq!(
            ltx2_generation("ltx-2.3-22b-distilled:q8", None),
            Some(Ltx2Generation::V2_3)
        );
        // Companion naming without the dash.
        assert_eq!(
            ltx2_generation("ltx2.3-vae", None),
            Some(Ltx2Generation::V2_3)
        );
        // Catalog IDs carry no family marker; the safetensors
        // `model_version` header hint decides.
        assert_eq!(
            ltx2_generation("cv:2752735", Some("2.3.0")),
            Some(Ltx2Generation::V2_3)
        );
        assert_eq!(
            ltx2_generation("hf:someone/some-repo", Some("2.0.1")),
            Some(Ltx2Generation::V2)
        );
        assert_eq!(
            ltx2_generation("cv:1", Some("2.1.0")),
            Some(Ltx2Generation::V2)
        );
        assert_eq!(
            ltx2_generation("cv:1", Some("2.2")),
            Some(Ltx2Generation::V2)
        );
    }

    #[test]
    fn ltx2_generation_fails_closed_on_unknown() {
        // Upstream 2.4 exists (CRF 18) but this build cannot load it —
        // it must not inherit either known profile.
        assert_eq!(ltx2_generation("cv:1", Some("2.4.0")), None);
        assert_eq!(ltx2_generation("cv:1", Some("3.0")), None);
        assert_eq!(ltx2_generation("cv:1", Some("2.30")), None);
        assert_eq!(ltx2_generation("cv:1", Some("garbage")), None);
        assert_eq!(ltx2_generation("cv:1", None), None);
        assert_eq!(ltx2_generation("some-model", None), None);
        // A future name marker is not folded into V2 by substring.
        assert_eq!(ltx2_generation("ltx-2.4-24b", None), None);
        // A version marker is a complete minor component: `2.30` is not
        // `2.3` (codex review, PR #1071).
        assert_eq!(ltx2_generation("ltx-2.30-22b", None), None);
        assert_eq!(ltx2_generation("ltx2.30-vae", None), None);
    }

    #[test]
    fn explicit_v2_minor_names_resolve_to_v2() {
        // Codex review (PR #1071): `ltx-2.0`/`2.1`/`2.2` name the
        // supported V2 lineage explicitly and must not fail closed.
        for name in ["ltx-2.0-19b", "ltx-2.1-19b:fp8", "ltx-2.2-19b"] {
            assert_eq!(
                ltx2_generation(name, None),
                Some(Ltx2Generation::V2),
                "{name}"
            );
        }
    }

    #[test]
    fn profile_maps_both_generations_to_crf_33() {
        for generation in [Ltx2Generation::V2, Ltx2Generation::V2_3] {
            let profile = ltx2_image_preprocessing_profile(generation);
            assert_eq!(profile.generation, generation);
            assert_eq!(profile.image_crf, 33);
        }
    }

    #[test]
    fn profile_serde_round_trips() {
        let profile = ltx2_image_preprocessing_profile(Ltx2Generation::V2_3);
        let json = serde_json::to_string(&profile).unwrap();
        assert_eq!(json, r#"{"generation":"v2_3","image_crf":33}"#);
        let back: Ltx2ImagePreprocessingProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(back, profile);
    }
}
