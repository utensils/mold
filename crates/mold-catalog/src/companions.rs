//! Curated canonical-companion registry.
//!
//! Single-file Civitai checkpoints (FLUX, SDXL, etc.) routinely strip
//! their text encoders + VAE to keep download size manageable. Without a
//! finite, mold-curated set of "canonical companions", every Civitai
//! entry would either have to ship its own T5 reference or trust an
//! arbitrary repo. By committing this registry, mold ships *one* T5,
//! *one* CLIP-L, etc., and any single-file checkpoint that demands
//! something exotic gets `engine_phase: 99` (visible-but-unsupported).

use crate::entry::{Bundling, CompanionRef, Kind, Source};
use crate::families::Family;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Companion {
    pub canonical_name: &'static str,
    pub kind: Kind,
    pub family_scope: &'static [Family],
    pub source: Source,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub size_bytes: u64,
}

pub static COMPANIONS: &[Companion] = &[
    Companion {
        canonical_name: "t5-v1_1-xxl",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Flux2, Family::LtxVideo, Family::Ltx2],
        source: Source::Hf,
        repo: "city96/t5-v1_1-xxl-encoder-bf16",
        files: &["t5xxl_*.safetensors"],
        size_bytes: 9_500_000_000,
    },
    Companion {
        canonical_name: "clip-l",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Flux2, Family::Sd15, Family::Sdxl],
        source: Source::Hf,
        repo: "openai/clip-vit-large-patch14",
        files: &[
            "model.safetensors",
            "config.json",
            "tokenizer*.json",
            "vocab.json",
            "merges.txt",
        ],
        size_bytes: 1_700_000_000,
    },
    Companion {
        canonical_name: "clip-g",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Sdxl],
        source: Source::Hf,
        repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        files: &[
            "open_clip_pytorch_model.safetensors",
            "open_clip_config.json",
        ],
        size_bytes: 5_700_000_000,
    },
    Companion {
        canonical_name: "sdxl-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Sdxl],
        source: Source::Hf,
        repo: "madebyollin/sdxl-vae-fp16-fix",
        files: &["sdxl_vae.safetensors"],
        size_bytes: 335_000_000,
    },
    Companion {
        canonical_name: "sd-vae-ft-mse",
        kind: Kind::Vae,
        family_scope: &[Family::Sd15],
        source: Source::Hf,
        repo: "stabilityai/sd-vae-ft-mse",
        files: &["diffusion_pytorch_model.safetensors", "config.json"],
        size_bytes: 335_000_000,
    },
    Companion {
        canonical_name: "flux-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Flux, Family::Flux2],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.1-dev",
        files: &["ae.safetensors"],
        size_bytes: 335_000_000,
    },
    // Reserved canonical for Z-Image. The exact text-encoder repo is
    // finalized when phase-4 single-file loader lands; the canonical
    // NAME is committed now so phase-1 entries can reference it without
    // rewrites.
    Companion {
        canonical_name: "z-image-te",
        kind: Kind::TextEncoder,
        family_scope: &[Family::ZImage],
        source: Source::Hf,
        repo: "Tongyi-MAI/Z-Image-Turbo",
        files: &["text_encoder/*"],
        size_bytes: 4_400_000_000,
    },
];

pub fn companion_by_name(name: &str) -> Option<&'static Companion> {
    COMPANIONS.iter().find(|c| c.canonical_name == name)
}

/// Returns the canonical-companion names a given (family, bundling) needs.
/// Empty for `Bundling::Separated` because diffusers HF entries are
/// self-contained.
pub fn companions_for(family: Family, bundling: Bundling) -> Vec<CompanionRef> {
    if matches!(bundling, Bundling::Separated) {
        return Vec::new();
    }
    let mut out = Vec::new();
    match family {
        Family::Flux | Family::Flux2 => {
            push(&mut out, "t5-v1_1-xxl");
            push(&mut out, "clip-l");
            push(&mut out, "flux-vae");
        }
        Family::Sd15 => {
            push(&mut out, "clip-l");
            push(&mut out, "sd-vae-ft-mse");
        }
        Family::Sdxl => {
            push(&mut out, "clip-l");
            push(&mut out, "clip-g");
            push(&mut out, "sdxl-vae");
        }
        Family::ZImage => {
            push(&mut out, "z-image-te");
        }
        Family::LtxVideo | Family::Ltx2 => {
            push(&mut out, "t5-v1_1-xxl");
        }
        // Single-file for these is `engine_phase: 99` — no companions.
        Family::QwenImage | Family::Wuerstchen => {}
    }
    out
}

fn push(out: &mut Vec<CompanionRef>, name: &'static str) {
    out.push(name.to_string());
}
