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

        // Wan. Only the checkpoints this build can actually run are mapped;
        // the rest stay in `CIVITAI_DROPS` below with the reason. The sub
        // family carries the variant because one `Family::Wan` spans three
        // very different runtime shapes — single-expert 2.1, the 2.2 5B, and
        // the 2.2 two-expert A14B pair.
        "Wan Video 1.3B t2v" => (Wan, Finetune, Some("wan21-t2v-1.3b".into())),
        "Wan Video 14B t2v" => (Wan, Finetune, Some("wan21-t2v-14b".into())),
        "Wan Video 2.2 TI2V-5B" => (Wan, Finetune, Some("wan22-ti2v-5b".into())),
        "Wan Video 2.2 T2V-A14B" => (Wan, Finetune, Some("wan22-t2v-a14b".into())),
        "Wan Video 2.2 I2V-A14B" => (Wan, Finetune, Some("wan22-i2v-a14b".into())),

        // Qwen
        "Qwen" | "Qwen 2" => (QwenImage, Finetune, None),

        _ => return None,
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
    match kind {
        // Supporting assets and adapters are installable when the family has
        // a compatible runtime. They are not standalone generation models, so
        // don't inherit checkpoint runnability rules.
        Lora => matches!(
            family,
            Flux | Flux2 | Sd15 | Sdxl | ZImage | Ltx2 | Wan | QwenImage
        ),
        Vae | TextEncoder | Tokenizer | Clip => true,
        ControlNet => matches!(family, Sd15 | Sdxl),
        Checkpoint => supported_for_checkpoint(family, bundling),
    }
}

fn supported_for_checkpoint(_family: Family, _bundling: Bundling) -> bool {
    true
}
