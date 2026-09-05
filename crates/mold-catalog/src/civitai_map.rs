//! Civitai `baseModel` string → mold `(Family, FamilyRole, sub_family)`.
//!
//! `CIVITAI_BASE_MODELS` is the union of known mappings and explicit drops
//! — it must stay synchronized: every entry either maps to `Some(...)` via
//! `map_base_model` or appears in `CIVITAI_DROPS`. The
//! `civitai_map_completeness` integration test enforces this invariant.

use crate::entry::{Bundling, Kind};
use crate::families::Family;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
pub enum FamilyRoleResult {
    Foundation,
    Finetune,
}

pub fn map_base_model(
    base_model: &str,
) -> Option<(Family, crate::entry::FamilyRole, Option<String>)> {
    use crate::entry::FamilyRole::*;
    use Family::*;
    Some(match base_model {
        // SD1.x
        "SD 1.4" | "SD 1.5" | "SD 1.5 LCM" | "SD 1.5 Hyper" => (Sd15, Finetune, None),

        // SDXL family (architecture-compatible variants)
        "SDXL 1.0" | "SDXL Lightning" | "SDXL Hyper" => (Sdxl, Finetune, None),
        "Pony" => (Sdxl, Finetune, Some("pony".into())),
        "Pony V7" => (Sdxl, Finetune, Some("pony-v7".into())),
        "Illustrious" => (Sdxl, Finetune, Some("illustrious".into())),
        "NoobAI" => (Sdxl, Finetune, Some("noobai".into())),

        // FLUX 1.x
        "Flux.1 S" => (Flux, Finetune, Some("flux1-s".into())),
        "Flux.1 D" => (Flux, Finetune, Some("flux1-d".into())),
        "Flux.1 Krea" => (Flux, Finetune, Some("flux1-krea".into())),
        "Flux.1 Kontext" => (Flux, Finetune, Some("flux1-kontext".into())),

        // FLUX 2
        "Flux.2 D" => (Flux2, Finetune, Some("flux2-d".into())),
        "Flux.2 Klein 9B" | "Flux.2 Klein 9B-base" => (Flux2, Finetune, Some("klein-9b".into())),
        "Flux.2 Klein 4B" | "Flux.2 Klein 4B-base" => (Flux2, Finetune, Some("klein-4b".into())),

        // Z-Image
        "ZImageTurbo" => (ZImage, Finetune, Some("turbo".into())),
        "ZImageBase" => (ZImage, Finetune, Some("base".into())),

        // LTX
        "LTXV" => (LtxVideo, Finetune, None),
        "LTXV2" => (Ltx2, Finetune, Some("v2".into())),
        "LTXV 2.3" => (Ltx2, Finetune, Some("v2.3".into())),

        // Wan. Only the checkpoints this build can install *and* run as a
        // complete model are mapped; the rest stay in `CIVITAI_DROPS` below
        // with the reason. The sub family carries the variant because one
        // `Family::Wan` spans very different runtime shapes.
        "Wan Video 1.3B t2v" => (Wan, Finetune, Some("wan21-t2v-1.3b".into())),
        "Wan Video 14B t2v" => (Wan, Finetune, Some("wan21-t2v-14b".into())),
        "Wan Video 2.2 TI2V-5B" => (Wan, Finetune, Some("wan22-ti2v-5b".into())),
        // A14B is a *pair*: two complete transformers the sampler alternates
        // between at a fixed timestep. Civitai publishes the high- and
        // low-noise experts as separate model versions, so `wan_a14b`
        // pairs the sibling versions into one two-file recipe (primary =
        // high-noise, `low-noise-transformer` role = low-noise) and the
        // normalizer marks any version whose counterpart cannot be
        // identified with confidence as unsupported — never a silent
        // single-expert install. `synthesize_intent` fails closed on the
        // same invariant for defense in depth.
        "Wan Video 2.2 T2V-A14B" => (Wan, Finetune, Some("wan22-t2v-a14b".into())),
        "Wan Video 2.2 I2V-A14B" => (Wan, Finetune, Some("wan22-i2v-a14b".into())),

        // Qwen
        "Qwen" | "Qwen 2" => (QwenImage, Finetune, None),

        _ => return None,
    })
}

/// Refine Civitai's shared Qwen base-model bucket using the model and version
/// names carried by each result. Civitai exposes both Qwen-Image and
/// Qwen-Image-Edit under `baseModel=Qwen`; keeping [`map_base_model`] generic
/// preserves that upstream query while this post-normalization step selects
/// the runtime family that will be written to the sidecar.
pub fn refine_family_from_names(
    family: Family,
    item_name: &str,
    version_name: Option<&str>,
) -> Family {
    if family == Family::QwenImage
        && [Some(item_name), version_name]
            .into_iter()
            .flatten()
            .any(|name| looks_like_qwen_image_edit(name) || looks_like_image_edit(name))
    {
        Family::QwenImageEdit
    } else {
        family
    }
}

fn looks_like_qwen_image_edit(value: &str) -> bool {
    let normalized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    words.iter().enumerate().any(|(index, word)| {
        let compact_suffix = word.strip_prefix("qwenimageedit");
        if compact_suffix.is_some_and(|suffix| suffix.chars().all(|ch| ch.is_ascii_digit())) {
            return true;
        }
        if *word != "qwen" {
            return false;
        }
        let mut rest = &words[index + 1..];
        if rest.first() == Some(&"image") {
            rest = &rest[1..];
        }
        while rest
            .first()
            .is_some_and(|word| word.chars().all(|ch| ch.is_ascii_digit()))
        {
            rest = &rest[1..];
        }
        rest.first() == Some(&"edit")
    })
}

fn looks_like_image_edit(value: &str) -> bool {
    let normalized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    words.iter().enumerate().any(|(index, word)| {
        if *word != "image" {
            return false;
        }
        let mut rest = &words[index + 1..];
        while rest
            .first()
            .is_some_and(|word| word.chars().all(|ch| ch.is_ascii_digit()))
        {
            rest = &rest[1..];
        }
        rest.first() == Some(&"edit")
    })
}

/// Civitai base-model strings we explicitly drop. mold has no engine for
/// these architectures, so surfacing them in the catalog would just tease
/// users with un-runnable downloads.
pub const CIVITAI_DROPS: &[&str] = &[
    "SD 2.0",
    "SD 2.1",
    "AuraFlow",
    "Chroma",
    "CogVideoX",
    "Ernie",
    "Grok",
    "HiDream",
    "Hunyuan 1",
    "Hunyuan Video",
    "Kolors",
    "Lumina",
    "Mochi",
    "PixArt a",
    "PixArt E",
    // Wan 2.1 image-to-video conditions through a CLIP-vision cross-attention
    // branch (`k_img`/`v_img`) that mold's DiT does not implement, and the
    // engine refuses those checkpoints by name. Mapping them would offer a
    // multi-gigabyte download that cannot generate.
    "Wan Video 14B i2v 480p",
    "Wan Video 14B i2v 720p",
    // Wan 2.5 and 2.7 are later architectures with no mold engine at all.
    "Wan Video 2.5 T2V",
    "Wan Video 2.5 I2V",
    "Wan Image 2.7",
    "Wan Video 2.7",
    "Anima",
    "Other",
    "Upscaler",
];

/// Every Civitai base-model string we know about — union of mapped + dropped.
/// The completeness test asserts these two sets are disjoint and exhaust this list.
pub const CIVITAI_BASE_MODELS: &[&str] = &[
    "SD 1.4",
    "SD 1.5",
    "SD 1.5 LCM",
    "SD 1.5 Hyper",
    "SDXL 1.0",
    "SDXL Lightning",
    "SDXL Hyper",
    "Pony",
    "Pony V7",
    "Illustrious",
    "NoobAI",
    "Flux.1 S",
    "Flux.1 D",
    "Flux.1 Krea",
    "Flux.1 Kontext",
    "Flux.2 D",
    "Flux.2 Klein 9B",
    "Flux.2 Klein 9B-base",
    "Flux.2 Klein 4B",
    "Flux.2 Klein 4B-base",
    "ZImageTurbo",
    "ZImageBase",
    "LTXV",
    "LTXV2",
    "LTXV 2.3",
    "Qwen",
    "Qwen 2",
    "SD 2.0",
    "SD 2.1",
    "AuraFlow",
    "Chroma",
    "CogVideoX",
    "Ernie",
    "Grok",
    "HiDream",
    "Hunyuan 1",
    "Hunyuan Video",
    "Kolors",
    "Lumina",
    "Mochi",
    "PixArt a",
    "PixArt E",
    "Wan Video 1.3B t2v",
    "Wan Video 14B t2v",
    "Wan Video 14B i2v 480p",
    "Wan Video 14B i2v 720p",
    "Wan Video 2.2 TI2V-5B",
    "Wan Video 2.2 I2V-A14B",
    "Wan Video 2.2 T2V-A14B",
    "Wan Video 2.5 T2V",
    "Wan Video 2.5 I2V",
    "Wan Image 2.7",
    "Wan Video 2.7",
    "Anima",
    "Other",
    "Upscaler",
];

/// Whether this build can install and run a catalog row.
pub fn supported_for(family: Family, bundling: Bundling, kind: Kind) -> bool {
    use Family::*;
    use Kind::*;
    if family == MinimaxH3 {
        // Family taxonomy is registered for exact classification and policy
        // enforcement, but there is no runnable H3 engine in this build.
        return false;
    }
    match kind {
        // Supporting assets and adapters are installable when the family has
        // a compatible runtime. They are not standalone generation models, so
        // don't inherit checkpoint runnability rules.
        Lora => matches!(
            family,
            Flux | Flux2 | Sd15 | Sdxl | Sd3 | ZImage | Ltx2 | Wan | QwenImage | QwenImageEdit
        ),
        Vae | TextEncoder | Tokenizer | Clip => true,
        ControlNet => matches!(family, Sd15 | Sdxl),
        Checkpoint => supported_for_checkpoint(family, bundling),
    }
}

fn supported_for_checkpoint(family: Family, bundling: Bundling) -> bool {
    // SD3.5 is runnable from a complete separated Diffusers repository. A
    // bare bundled transformer needs the three text encoders and VAE that the
    // catalog does not yet synthesize as companions, so do not advertise that
    // shape as runnable.
    family != Family::Sd3 || bundling == Bundling::Separated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd3_support_requires_a_complete_separated_repository() {
        assert!(supported_for(
            Family::Sd3,
            Bundling::Separated,
            Kind::Checkpoint
        ));
        assert!(!supported_for(
            Family::Sd3,
            Bundling::SingleFile,
            Kind::Checkpoint
        ));
        assert!(supported_for(Family::Sd3, Bundling::SingleFile, Kind::Lora));
    }

    #[test]
    fn qwen_edit_name_refinement_accepts_ecosystem_spellings_without_prefix_matches() {
        for name in [
            "Qwen Image Edit",
            "QWEN_IMAGE_EDIT",
            "QwenImageEdit2511",
            "QWEN-EDIT",
            "Qwen-image_2511_Edit",
        ] {
            assert_eq!(
                refine_family_from_names(Family::QwenImage, name, None),
                Family::QwenImageEdit,
                "{name}"
            );
        }
        for name in ["Qwen Image", "Qwen Image Editorial", "Qwen Image Editable"] {
            assert_eq!(
                refine_family_from_names(Family::QwenImage, name, None),
                Family::QwenImage,
                "{name}"
            );
        }
    }

    #[test]
    fn qwen_edit_catalog_shapes_are_supported() {
        for kind in [Kind::Checkpoint, Kind::Lora] {
            assert!(supported_for(
                Family::QwenImageEdit,
                Bundling::SingleFile,
                kind
            ));
        }
    }
}
