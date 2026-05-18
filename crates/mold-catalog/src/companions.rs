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
        family_scope: &[Family::Flux, Family::LtxVideo, Family::Ltx2],
        source: Source::Hf,
        repo: "city96/t5-v1_1-xxl-encoder-bf16",
        files: &["t5xxl_*.safetensors"],
        size_bytes: 9_500_000_000,
    },
    Companion {
        canonical_name: "clip-l",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Sd15, Family::Sdxl],
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
        // Repo file is `open_clip_model.safetensors` (10.16 GB, OpenCLIP
        // format); the older `open_clip_pytorch_model.safetensors` name
        // never existed and 404'd on every SDXL Civitai pull.
        files: &["open_clip_model.safetensors", "open_clip_config.json"],
        size_bytes: 10_158_382_892,
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
        family_scope: &[Family::Flux],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.1-dev",
        files: &["ae.safetensors"],
        size_bytes: 335_000_000,
    },
    // Flux.2 Klein-4B text encoder + tokenizer (Qwen3 4B, two shards). The
    // single-file Civitai Flux.2 checkpoints strip text-encoder weights to
    // keep download size manageable, so we pull Qwen3 from the Apache-2.0
    // Klein-4B repo. Used for sub_family=klein-4b only.
    Companion {
        canonical_name: "flux2-te",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux2],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.2-klein-4B",
        files: &[
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "tokenizer/tokenizer.json",
        ],
        size_bytes: 8_056_404_646,
    },
    // Flux.2 Klein-9B text encoder + tokenizer (Qwen3 8B, four shards).
    // Hosted in the gated Klein-9B repo — single-file Civitai Klein-9B
    // fine-tunes need this encoder and the user must have a BFL token.
    // Used for sub_family ∈ {klein-9b, flux2-d}.
    Companion {
        canonical_name: "flux2-te-9b",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux2],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.2-klein-9B",
        files: &[
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "tokenizer/tokenizer.json",
        ],
        size_bytes: 16_392_938_478,
    },
    // Flux.2 Klein VAE (~168 MB). Distinct from `flux-vae` (FLUX.1's 335 MB
    // ae.safetensors) — Flux.2 ships a smaller VAE and the engine refuses to
    // load FLUX.1's ae as a substitute. The same VAE is used by both Klein-4B
    // and Klein-9B; we pull from the ungated Klein-4B repo.
    Companion {
        canonical_name: "flux2-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Flux2],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.2-klein-4B",
        files: &["vae/diffusion_pytorch_model.safetensors"],
        size_bytes: 168_120_878,
    },
    // Z-Image shared runtime assets (Qwen3 BF16 shards + tokenizer + VAE).
    // Civitai single-file primaries are transformer-only.
    Companion {
        canonical_name: "z-image-te",
        kind: Kind::TextEncoder,
        family_scope: &[Family::ZImage],
        source: Source::Hf,
        repo: "Tongyi-MAI/Z-Image-Turbo",
        files: &[
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "tokenizer/tokenizer.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
        size_bytes: 8_224_071_556,
    },
    // VAE for LTX-Video single-file catalog entries. Civitai LTX-Video
    // fine-tunes are transformer-only, so the VAE must be pulled
    // separately from the same repo mold uses for manifest-based models.
    Companion {
        canonical_name: "ltx-video-vae",
        kind: Kind::Vae,
        family_scope: &[Family::LtxVideo],
        source: Source::Hf,
        repo: "Lightricks/LTX-Video-0.9.5",
        files: &["vae/diffusion_pytorch_model.safetensors"],
        size_bytes: 2_493_855_612,
    },
    // Gemma 3 12B text encoder for LTX-2 single-file catalog entries.
    // Civitai LTX-2 fine-tunes (e.g. cv:2752735, cv:2781713) bundle the
    // transformer + VAE but not the Gemma TE — without this companion the
    // runtime bails with `LTX-2 requires Gemma text encoder files to be
    // available`. Same gated repo + file set the manifest LTX-2 models use,
    // so a user with the Gemma TE already installed for a manifest LTX-2
    // model gets cv:* installs essentially for free (HF cache hits).
    Companion {
        canonical_name: "ltx2-te",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Ltx2],
        source: Source::Hf,
        repo: "google/gemma-3-12b-it-qat-q4_0-unquantized",
        files: &[
            "config.json",
            "generation_config.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
            "model.safetensors.index.json",
            "added_tokens.json",
            "chat_template.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ],
        // Sum of the five .safetensors shards (24_374_793_024 B); the JSON /
        // tokenizer files round to ~38 MB combined and don't move the needle.
        size_bytes: 24_374_793_024,
    },
];

pub fn companion_by_name(name: &str) -> Option<&'static Companion> {
    COMPANIONS.iter().find(|c| c.canonical_name == name)
}

/// Returns the canonical-companion names a given (family, sub_family,
/// bundling, kind) needs. Empty for `Bundling::Separated` because diffusers HF
/// entries are self-contained, and empty for `Kind::Lora` regardless of
/// bundling because LoRAs are self-contained patches that ride on whatever
/// base model is already loaded.
///
/// `sub_family` distinguishes encoder size for families where the same
/// `Family` value covers multiple architectures — currently just Flux.2,
/// where `klein-4b` uses Qwen3 4B and `klein-9b` / `flux2-d` use Qwen3 8B.
pub fn companions_for(
    family: Family,
    sub_family: Option<&str>,
    bundling: Bundling,
    kind: Kind,
) -> Vec<CompanionRef> {
    // LoRAs, ControlNets, and standalone VAEs/text-encoders never pull
    // companions — they slot into an existing pipeline rather than booting
    // their own. Only Kind::Checkpoint participates in the companion graph.
    if !matches!(kind, Kind::Checkpoint) {
        return Vec::new();
    }
    if matches!(bundling, Bundling::Separated) {
        return Vec::new();
    }
    let mut out = Vec::new();
    match family {
        Family::Flux => {
            push(&mut out, "t5-v1_1-xxl");
            push(&mut out, "clip-l");
            push(&mut out, "flux-vae");
        }
        Family::Flux2 => {
            // Flux.2 uses Qwen3 (not T5+CLIP-L) and a Klein-specific VAE.
            // Klein-4B has a 4B Qwen3 encoder (ungated, Apache-2.0);
            // Klein-9B and Flux.2-Dev share the gated 8B encoder. Default
            // to the 9B encoder when sub_family is unknown — it works for
            // the common case (most Civitai fine-tunes are Klein-9B-based).
            match sub_family {
                Some("klein-4b") => push(&mut out, "flux2-te"),
                _ => push(&mut out, "flux2-te-9b"),
            }
            push(&mut out, "flux2-vae");
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
        Family::LtxVideo => {
            push(&mut out, "t5-v1_1-xxl");
            // Civitai LTX-Video checkpoints are transformer-only; VAE
            // comes from the companion.
            push(&mut out, "ltx-video-vae");
        }
        Family::Ltx2 => {
            // LTX-2 combined checkpoints bundle the VAE — no VAE companion.
            // The text encoder is Gemma 3 12B (gated), not T5: the LTX-2
            // runtime in `mold-inference` reads `paths.text_encoder_files`
            // and rejects the load if it's empty (`gemma_root` in
            // `ltx2/assets.rs`). Pulling t5-v1_1-xxl here would download
            // ~9.5 GB of unused weights and still fail the runtime check.
            push(&mut out, "ltx2-te");
        }
        // Single-file for these is `engine_phase: 99` — no companions.
        Family::QwenImage | Family::Wuerstchen => {}
    }
    out
}

fn push(out: &mut Vec<CompanionRef>, name: &'static str) {
    out.push(name.to_string());
}
