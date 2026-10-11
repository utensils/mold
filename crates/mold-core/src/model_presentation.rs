//! How a model is NAMED to a person — the one authority every surface reads.
//!
//! A manifest id (`minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p`) is an
//! address, not a name. Each curated checkpoint is presented as:
//!
//! - **title** — the model itself, without precision: `FLUX.1 Dev`,
//!   `Qwen Image 2.1`, `MiniMax H3 Frames`.
//! - **variant** — what tells siblings apart: `Q4`, `FP8`, `Turbo 4-step 768p`.
//! - **summary** — one line on why you would pick it over its siblings.
//! - **family label** — the section it sits under: `FLUX.1`, `MiniMax H3`.
//!
//! `display_name` is `title variant`, the short alias `/api/models` has always
//! carried. The server publishes all four on every manifest row, and
//! `generate_generation_profiles` writes the same table into the TypeScript
//! and Swift clients, so web, desktop, the phones and the native apps name a
//! model identically even for an id no connected machine reports. Catalog
//! installs (Civitai, Hugging Face) are not ours to rename: they keep their
//! provider title and have no presentation here.

use crate::manifest::ModelManifest;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelPresentation {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_label: Option<String>,
}

impl ModelPresentation {
    /// The short alias: title, then variant. Never a request or file id.
    pub fn display_name(&self) -> String {
        match &self.variant {
            Some(variant) => format!("{} {variant}", self.title),
            None => self.title.clone(),
        }
    }
}

/// The plain name of a model family, for section headers and family-only
/// contexts. `None` for a family this build does not know, so callers keep
/// their own fallback (the slug) rather than inventing a name.
pub fn family_label(family: &str) -> Option<&'static str> {
    Some(match family {
        "flux" => "FLUX.1",
        "flux2" => "FLUX.2",
        "sd15" => "SD 1.5",
        "sd3" | "sd3.5" => "SD 3.5",
        "sdxl" => "SDXL",
        "z-image" => "Z-Image",
        "qwen-image" | "qwen_image" => "Qwen Image",
        "qwen-image-edit" => "Qwen Image Edit",
        "qwen-image21" => "Qwen Image 2.1",
        "wuerstchen" | "wuerstchen-v2" => "Wuerstchen",
        "ltx-video" | "ltx_video" => "LTX Video",
        "ltx2" => "LTX-2",
        "wan" => "Wan Video",
        "minimax-h3" | "minimax_h3" | "minimaxh3" => "MiniMax H3",
        "hunyuan3d" => "Hunyuan3D",
        "controlnet" => "ControlNet",
        "qwen3-expand" => "Prompt expansion",
        "upscaler" => "Upscaler",
        _ => return None,
    })
}

/// Every family label this build knows, keyed by the canonical slug, for the
/// generated client tables.
pub const FAMILY_SLUGS: &[&str] = &[
    "flux",
    "flux2",
    "sd15",
    "sd3",
    "sd3.5",
    "sdxl",
    "z-image",
    "qwen-image",
    "qwen-image-edit",
    "qwen-image21",
    "wuerstchen",
    "ltx-video",
    "ltx2",
    "wan",
    "minimax-h3",
    "hunyuan3d",
    "controlnet",
    "qwen3-expand",
    "upscaler",
];

/// The presentation of one curated checkpoint.
pub fn presentation(manifest: &ModelManifest) -> ModelPresentation {
    let family_label = family_label(&manifest.family).map(str::to_string);
    if manifest.family == crate::minimax_h3::FAMILY {
        if let Some(mut h3) = minimax_h3_presentation(manifest) {
            h3.family_label = family_label;
            return h3;
        }
    }
    if let Some(&(_, title, summary)) = AUTHORED.iter().find(|(name, ..)| *name == manifest.name) {
        return ModelPresentation {
            title: title.to_string(),
            variant: None,
            summary: Some(summary.to_string()),
            family_label,
        };
    }
    let (head, tail) = split_description(&manifest.description);
    let (title, mut variant) = split_precision(head);
    let mut title = normalize_title(&title);
    if manifest.name.starts_with("qwen-image-2.1-turbo:") {
        title.push_str(" Turbo");
    }
    // Siblings whose descriptions differ only after the dash: the dash text is
    // what tells them apart, so it joins the variant.
    if let Some(extra) = variant_extra(&manifest.name) {
        variant = Some(match variant {
            Some(precision) => format!("{extra} {precision}"),
            None => extra.to_string(),
        });
    }
    let tail = tail.or_else(|| {
        manifest
            .name
            .starts_with("qwen-image-2.1-turbo:")
            .then(|| "6-step distill of Qwen Image 2.1: much faster, slightly softer".to_string())
    });
    let summary = tail
        .map(clean_summary)
        .filter(|summary| !summary.is_empty())
        .or_else(|| {
            variant
                .as_deref()
                .and_then(precision_summary)
                .map(str::to_string)
        });
    ModelPresentation {
        title,
        variant,
        summary,
        family_label,
    }
}

/// Checkpoints whose description is a sentence rather than `Title — trade-off`
/// are named here outright.
const AUTHORED: &[(&str, &str, &str)] = &[
    (
        "controlnet-canny-sd15:fp16",
        "ControlNet Canny",
        "Follows the edges of a reference picture (SD 1.5)",
    ),
    (
        "controlnet-depth-sd15:fp16",
        "ControlNet Depth",
        "Follows the depth of a reference picture (SD 1.5)",
    ),
    (
        "controlnet-openpose-sd15:fp16",
        "ControlNet OpenPose",
        "Follows the body pose in a reference picture (SD 1.5)",
    ),
];

/// The presentation of a manifest id, or `None` for anything we did not curate.
pub fn presentation_for_id(model: &str) -> Option<ModelPresentation> {
    crate::manifest::find_manifest(model).map(presentation)
}

/// `"Title — trade-off"` with an optional `"Title: detail"` lead-in, which
/// joins the trade-off rather than the title.
fn split_description(description: &str) -> (&str, Option<String>) {
    let (head, tail) = match description.split_once(" — ") {
        Some((head, tail)) => (head.trim(), Some(tail.trim().to_string())),
        None => (description.trim(), None),
    };
    match head.split_once(": ") {
        Some((title, detail)) => {
            let tail = match tail {
                Some(tail) => format!("{detail} — {tail}"),
                None => detail.to_string(),
            };
            (title.trim(), Some(tail))
        }
        None => (head, tail),
    }
}

fn is_precision(word: &str) -> bool {
    let upper = word.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "FP8" | "FP8-SCALED" | "FP16" | "FP32" | "BF16" | "INT8" | "NVFP4" | "CONVROT"
    ) || (upper.len() >= 2
        && upper.starts_with('Q')
        && upper.as_bytes()[1].is_ascii_digit()
        && upper[2..]
            .chars()
            .all(|c| c == '_' || c.is_ascii_alphanumeric()))
}

/// Pull precision words (anywhere in the title) out into the variant, and
/// drop packaging noise: `GGUF`, `transformer (7.7 GB)`, `+ Viggle …`,
/// parenthetical vendors.
fn split_precision(head: &str) -> (String, Option<String>) {
    let head = head.split_once(" + ").map_or(head, |(title, _)| title);
    let head = head
        .split_once(" transformer")
        .map_or(head, |(title, _)| title);
    let mut title = Vec::new();
    let mut precision = Vec::new();
    let mut in_parenthetical = false;
    for word in head.split_whitespace() {
        if word.starts_with('(') {
            in_parenthetical = !word.ends_with(')');
            continue;
        }
        if in_parenthetical {
            in_parenthetical = !word.ends_with(')');
            continue;
        }
        if word.eq_ignore_ascii_case("GGUF") {
            continue;
        }
        if is_precision(word) {
            precision.push(if word.eq_ignore_ascii_case("ConvRot") {
                "ConvRot".to_string()
            } else if word.eq_ignore_ascii_case("FP8-scaled") {
                "FP8".to_string()
            } else {
                word.to_ascii_uppercase()
            });
        } else {
            title.push(word);
        }
    }
    let variant = (!precision.is_empty()).then(|| precision.join(" "));
    (title.join(" "), variant)
}

fn normalize_title(title: &str) -> String {
    let title = title
        .replace("Flux.2", "FLUX.2")
        .replace("[dev]", "Dev")
        .replace("Qwen-Image-Edit-", "Qwen Image Edit ")
        .replace("Qwen-Image-", "Qwen Image ")
        .replace("Qwen-Image", "Qwen Image")
        .replace("Klein-4B", "Klein 4B")
        .replace("Klein-9B", "Klein 9B");
    title
        .split_whitespace()
        .map(|word| match word {
            "dev" => "Dev",
            "distilled" => "Distilled",
            "mini" => "Mini",
            other => other,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The few siblings whose titles AND precision are identical: their step
/// schedule or VAE is the difference, so it is named.
fn variant_extra(model: &str) -> Option<&'static str> {
    if model.ends_with(":bf16-conv") {
        return Some("Conv VAE");
    }
    match model {
        "qwen-image-lightning:fp8" => Some("4-step"),
        "qwen-image-lightning:fp8-8step" => Some("8-step"),
        _ => None,
    }
}

fn clean_summary(summary: String) -> String {
    // Download sizes are shown beside the name already; drop "(7.7 GB)",
    // "(32MB)", "(23.8GB transformer)".
    let mut out = String::with_capacity(summary.len());
    let mut rest = summary.as_str();
    while let Some(open) = rest.find('(') {
        let Some(close) = rest[open..].find(')') else {
            break;
        };
        let inner = &rest[open + 1..open + close];
        let is_size = inner.split_whitespace().next().is_some_and(|first| {
            let first = first.trim_end_matches("GB").trim_end_matches("MB");
            first.parse::<f64>().is_ok()
        }) && (inner.contains("GB") || inner.contains("MB"));
        if is_size {
            out.push_str(rest[..open].trim_end());
        } else {
            out.push_str(&rest[..open + close + 1]);
        }
        rest = &rest[open + close + 1..];
    }
    out.push_str(rest);
    let out = out.trim().trim_end_matches(',').trim().to_string();
    let mut chars = out.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => out,
    }
}

/// Why pick this precision, for a checkpoint whose description says nothing
/// beyond its name.
fn precision_summary(variant: &str) -> Option<&'static str> {
    let precision = variant.split_whitespace().last()?;
    Some(match precision {
        "BF16" | "FP16" | "FP32" => "Full precision: best quality, largest download",
        "FP8" | "INT8" | "ConvRot" | "Q8" | "Q8_0" => "Near full quality at about half the size",
        p if p.starts_with("Q6") || p.starts_with("Q5") => "Good balance of quality and size",
        p if p.starts_with("Q4") || p.starts_with("Q3") => {
            "Smaller download, slightly lower quality"
        }
        p if p.starts_with("Q2") => "Smallest download; visibly lower quality",
        "NVFP4" => "Compact 4-bit weights for the newest NVIDIA GPUs",
        _ => return None,
    })
}

fn minimax_h3_presentation(manifest: &ModelManifest) -> Option<ModelPresentation> {
    use crate::minimax_h3::{self, Layout, Task};
    let task = minimax_h3::task_for_model(&manifest.name)?;
    let (title, what) = match task {
        Task::Fl2va => (
            "MiniMax H3 Frames",
            "First (and last) frame to a clip with sound",
        ),
        Task::Ref2va => ("MiniMax H3 References", "References to a clip with sound"),
    };
    let base_steps = minimax_h3::COMFY_DEFAULT_STEPS;
    let (variant, tradeoff) = if let Some(tier) = minimax_h3::REVIEWED_TURBO_MANIFEST_TIERS
        .iter()
        .find(|tier| tier.model == manifest.name)
    {
        let mut tradeoff = format!("{} steps, much faster, a little softer", tier.steps);
        if tier.display_label.contains("768p") {
            tradeoff.push_str(", tuned for 768p");
        }
        if tier.display_label.contains("rank") {
            tradeoff.push_str(", lighter adapter");
        }
        (Some(tier.display_label.to_string()), tradeoff)
    } else if let Some(tier) = minimax_h3::distilled_base_tier_for_model(&manifest.name) {
        (
            Some(tier.display_label.to_string()),
            format!("{} steps, fastest, some loss of detail", tier.steps),
        )
    } else {
        match minimax_h3::layout_for_model(&manifest.name) {
            Some(Layout::OfficialBf16) => (
                Some("BF16".to_string()),
                "Official full precision, best quality, most memory".to_string(),
            ),
            Some(Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq) => (
                Some("NVFP4".to_string()),
                "Compact 4-bit weights for the newest NVIDIA GPUs".to_string(),
            ),
            _ => (None, format!("{base_steps} steps, best detail, slowest")),
        }
    };
    Some(ModelPresentation {
        title: title.to_string(),
        variant,
        summary: Some(format!("{what} · {tradeoff}")),
        family_label: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{find_manifest, known_manifests};
    use std::collections::HashMap;

    fn shown(model: &str) -> ModelPresentation {
        presentation(find_manifest(model).unwrap_or_else(|| panic!("{model} is a manifest id")))
    }

    #[test]
    fn splits_the_model_from_its_precision_and_says_why() {
        let p = shown("flux-dev:q4");
        assert_eq!(p.title, "FLUX.1 Dev");
        assert_eq!(p.variant.as_deref(), Some("Q4"));
        assert_eq!(p.summary.as_deref(), Some("Smaller/faster, good quality"));
        assert_eq!(p.family_label.as_deref(), Some("FLUX.1"));
        assert_eq!(p.display_name(), "FLUX.1 Dev Q4");
    }

    #[test]
    fn drops_packaging_noise_and_sizes() {
        let p = shown("qwen-image-2.1:q8");
        assert_eq!(p.title, "Qwen Image 2.1");
        assert_eq!(p.variant.as_deref(), Some("Q8_0"));
        // No authored trade-off: the precision says why.
        assert_eq!(
            p.summary.as_deref(),
            Some("Near full quality at about half the size")
        );
        let p = shown("flux-dev:bf16");
        assert_eq!(p.summary.as_deref(), Some("Full quality, full precision"));
    }

    #[test]
    fn names_minimax_h3_by_what_it_does_not_its_task_code() {
        let p = shown("minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p");
        assert_eq!(p.title, "MiniMax H3 Frames");
        assert_eq!(p.variant.as_deref(), Some("Turbo 4-step 768p"));
        let summary = p.summary.unwrap();
        assert!(summary.starts_with("First (and last) frame"), "{summary}");
        assert!(summary.contains("tuned for 768p"), "{summary}");
        assert_eq!(p.family_label.as_deref(), Some("MiniMax H3"));
        assert_eq!(
            shown("minimax-h3-ref2va:comfy-pruned-int8").title,
            "MiniMax H3 References"
        );
    }

    #[test]
    fn every_visible_model_has_a_unique_short_name_and_a_reason_to_pick_it() {
        let mut seen: HashMap<String, String> = HashMap::new();
        for manifest in known_manifests().iter().filter(|m| !m.hidden) {
            let p = presentation(manifest);
            assert!(!p.title.is_empty(), "{} has no title", manifest.name);
            assert!(
                p.summary.as_deref().is_some_and(|s| !s.is_empty()),
                "{} has no summary",
                manifest.name
            );
            assert!(
                p.display_name().len() <= 48,
                "{} is still long: {}",
                manifest.name,
                p.display_name()
            );
            if let Some(other) = seen.insert(p.display_name(), manifest.name.clone()) {
                panic!(
                    "{} and {other} share the name {}",
                    manifest.name,
                    p.display_name()
                );
            }
        }
    }

    #[test]
    fn unknown_families_have_no_label_so_callers_keep_their_fallback() {
        assert_eq!(family_label("companion"), None);
        assert_eq!(family_label(""), None);
        for slug in FAMILY_SLUGS {
            assert!(family_label(slug).is_some(), "{slug}");
        }
    }

    #[test]
    fn wan_family_label_matches_the_shared_surface_parity_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/wan/surface-parity-v1.json"
        )))
        .expect("fixture parses");
        let family = fixture["family"].as_str().expect("family");
        let label = fixture["family_label"].as_str().expect("family_label");
        assert_eq!(family_label(family), Some(label));
    }
}
