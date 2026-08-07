//! Curated canonical-companion registry.
//!
//! Single-file Civitai checkpoints (FLUX, SDXL, etc.) routinely strip
//! their text encoders + VAE to keep download size manageable. Without a
//! finite, mold-curated set of "canonical companions", every Civitai
//! entry would either have to ship its own T5 reference or trust an
//! arbitrary repo. By committing this registry, mold ships *one* T5,
//! *one* CLIP-L, etc., and any single-file checkpoint that demands
//! something exotic is marked unsupported by this build.

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
    // LTX-2 and LTX-2.3 use different video-VAE architectures. Keep
    // separate companions so catalog fine-tunes cannot silently decode with
    // an incompatible latent space.
    Companion {
        canonical_name: "ltx2-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Ltx2],
        source: Source::Hf,
        repo: "Lightricks/LTX-2",
        files: &["vae/diffusion_pytorch_model.safetensors"],
        size_bytes: 2_444_982_370,
    },
    Companion {
        canonical_name: "ltx2.3-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Ltx2],
        source: Source::Hf,
        repo: "Kijai/LTX2.3_comfy",
        files: &["vae/LTX23_video_vae_bf16.safetensors"],
        size_bytes: 1_452_258_578,
    },
    // LTX-2.3 diffusion-only / quantized exports omit the large Gemma hidden-
    // state projection matrices even when they retain the video/audio
    // connector blocks. This is a separate input to ComfyUI's text encoder
    // loader and must accompany those transformer-only checkpoints.
    Companion {
        canonical_name: "ltx2.3-text-projection",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Ltx2],
        source: Source::Hf,
        repo: "Kijai/LTX2.3_comfy",
        files: &["text_encoders/ltx-2.3_text_projection_bf16.safetensors"],
        size_bytes: 2_312_149_072,
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
    // Wan's UMT5-XXL encoder and tokenizer. Every Wan checkpoint in the wild
    // — Comfy-Org repack, QuantStack GGUF, Civitai fine-tune — is
    // transformer-only; the encoder is shipped once and shared. Repo, files,
    // and sizes match the `wan-umt5` manifest companion exactly so a user who
    // already pulled a manifest Wan model gets cv:/hf: installs from cache.
    Companion {
        canonical_name: "wan-umt5",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Wan],
        source: Source::Hf,
        repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        files: &["split_files/text_encoders/umt5_xxl_fp16.safetensors"],
        size_bytes: 11_366_399_385,
    },
    // The two Wan VAEs are different architectures, not versions of one: 2.1
    // is 16-channel at 8x8 spatial, 2.2 is 48-channel at 16x16. The DiT's
    // input channel count is validated against the VAE's latent width at load,
    // so pairing the wrong one fails rather than degrades — but it fails after
    // a multi-gigabyte download, which is why the two are separate companions.
    Companion {
        canonical_name: "wan21-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Wan],
        source: Source::Hf,
        repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        files: &["split_files/vae/wan_2.1_vae.safetensors"],
        size_bytes: 253_815_318,
    },
    Companion {
        canonical_name: "wan22-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Wan],
        source: Source::Hf,
        repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        files: &["split_files/vae/wan2.2_vae.safetensors"],
        size_bytes: 1_409_400_960,
    },
    // Shared Qwen-Image runtime assets. Civitai Qwen single-file
    // checkpoints are transformer-only exports; the VAE, Qwen2.5-VL text
    // encoder shards, and tokenizer come from the base HF release.
    Companion {
        canonical_name: "qwen-image-runtime",
        kind: Kind::TextEncoder,
        family_scope: &[Family::QwenImage],
        source: Source::Hf,
        repo: "Qwen/Qwen-Image",
        files: &[
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "tokenizer/tokenizer.json",
        ],
        size_bytes: 16_845_253_155,
    },
    // Shared Wuerstchen runtime assets excluding the Stage C prior. A
    // single-file catalog primary is treated as the prior transformer, and
    // this companion supplies Stage B decoder, VQGAN, both CLIP encoders,
    // and tokenizers from the manifest Wuerstchen v2 layout.
    Companion {
        canonical_name: "wuerstchen-runtime",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Wuerstchen],
        source: Source::Hf,
        repo: "warp-ai/wuerstchen",
        files: &[
            "decoder/diffusion_pytorch_model.safetensors",
            "vqgan/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
            "prior/text_encoder/model.safetensors",
            "prior/tokenizer/tokenizer.json",
        ],
        size_bytes: 8_483_788_558,
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
            // The text encoder is Gemma 3 12B (gated), not T5: the LTX-2
            // runtime in `mold-inference` reads `paths.text_encoder_files`
            // and rejects the load if it's empty (`gemma_root` in
            // `ltx2/assets.rs`). Pulling t5-v1_1-xxl here would download
            // ~9.5 GB of unused weights and still fail the runtime check.
            push(&mut out, "ltx2-te");
            match sub_family {
                Some("v2") => push(&mut out, "ltx2-vae"),
                // Civitai identifies current checkpoints as `v2.3`. Default
                // unknown single-file rows to the current architecture.
                _ => {
                    push(&mut out, "ltx2.3-vae");
                    push(&mut out, "ltx2.3-text-projection");
                }
            }
        }
        Family::Wan => {
            push(&mut out, "wan-umt5");
            // Which VAE is not a version question, and the naming actively
            // misleads: **Wan 2.2's A14B pair uses the 2.1 VAE**, the same
            // 16-channel file the 1.3B uses. Only TI2V-5B carries the
            // 48-channel 2.2 VAE. Keying off the `wan22-` prefix would hand
            // A14B a VAE whose latent width its DiT rejects.
            match sub_family {
                Some("wan22-ti2v-5b") => push(&mut out, "wan22-vae"),
                // Everything else — 2.1 at either size, both A14B experts,
                // and unknown community fine-tunes, which are overwhelmingly
                // 14B-class — takes the 2.1 VAE.
                _ => push(&mut out, "wan21-vae"),
            }
        }
        // H3 catalog installs remain policy-gated and have no independently
        // runnable companion graph. Its hidden core manifests own the exact
        // FL2VA/Ref2VA component layouts.
        Family::MinimaxH3 => {}
        Family::QwenImage => {
            push(&mut out, "qwen-image-runtime");
        }
        Family::Wuerstchen => {
            push(&mut out, "wuerstchen-runtime");
        }
    }
    out
}

fn push(out: &mut Vec<CompanionRef>, name: &'static str) {
    out.push(name.to_string());
}
