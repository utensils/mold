use mold_catalog::companions::{companion_by_name, companions_for, COMPANIONS};
use mold_catalog::entry::{Bundling, Kind};
use mold_catalog::families::Family;

#[test]
fn flux_single_file_needs_t5_and_clip_l() {
    let names = companions_for(Family::Flux, None, Bundling::SingleFile, Kind::Checkpoint);
    assert!(names.contains(&"t5-v1_1-xxl".to_string()));
    assert!(names.contains(&"clip-l".to_string()));
}

#[test]
fn sdxl_single_file_needs_two_clips_and_vae() {
    let names = companions_for(Family::Sdxl, None, Bundling::SingleFile, Kind::Checkpoint);
    assert!(names.contains(&"clip-l".to_string()));
    assert!(names.contains(&"clip-g".to_string()));
    assert!(names.contains(&"sdxl-vae".to_string()));
}

#[test]
fn flux2_klein_4b_uses_qwen3_4b_companion() {
    let names = companions_for(
        Family::Flux2,
        Some("klein-4b"),
        Bundling::SingleFile,
        Kind::Checkpoint,
    );
    assert!(names.contains(&"flux2-te".to_string()));
    assert!(names.contains(&"flux2-vae".to_string()));
    assert!(!names.contains(&"flux2-te-9b".to_string()));
}

#[test]
fn flux2_klein_9b_uses_qwen3_8b_companion() {
    let names = companions_for(
        Family::Flux2,
        Some("klein-9b"),
        Bundling::SingleFile,
        Kind::Checkpoint,
    );
    assert!(names.contains(&"flux2-te-9b".to_string()));
    assert!(names.contains(&"flux2-vae".to_string()));
    assert!(!names.contains(&"flux2-te".to_string()));
}

#[test]
fn flux2_dev_uses_qwen3_8b_companion() {
    let names = companions_for(
        Family::Flux2,
        Some("flux2-d"),
        Bundling::SingleFile,
        Kind::Checkpoint,
    );
    assert!(names.contains(&"flux2-te-9b".to_string()));
    assert!(names.contains(&"flux2-vae".to_string()));
}

#[test]
fn flux2_unknown_subfamily_defaults_to_9b_encoder() {
    // Most Civitai Flux.2 fine-tunes are 9B-based; default to the 9B
    // encoder when sub_family is missing so unknown rows fail open rather
    // than loading an undersized text encoder.
    let names = companions_for(Family::Flux2, None, Bundling::SingleFile, Kind::Checkpoint);
    assert!(names.contains(&"flux2-te-9b".to_string()));
}

#[test]
fn ltx2_single_file_pulls_gemma_te_not_t5() {
    // LTX-2 / LTX-2.3 use Gemma 3 12B as the text encoder, not T5. Civitai
    // single-file LTX-2 checkpoints (e.g. cv:2752735, cv:2781713) bundle
    // the transformer + VAE but ship no encoder, and the runtime
    // (`ltx2/assets.rs::gemma_root`) reads the parent of the first entry
    // in `paths.text_encoder_files` — a missing or T5-shaped encoder fails
    // the load with `LTX-2 requires Gemma text encoder files to be
    // available`. Pin the correct companion so a future scope-creep edit
    // doesn't silently swap T5 back in.
    let names = companions_for(Family::Ltx2, None, Bundling::SingleFile, Kind::Checkpoint);
    assert!(
        names.contains(&"ltx2-te".to_string()),
        "LTX-2 must pull the Gemma TE companion; got {names:?}"
    );
    assert!(
        !names.contains(&"t5-v1_1-xxl".to_string()),
        "LTX-2 must NOT pull T5 (unused, ~9.5 GB wasted download); got {names:?}"
    );
}

#[test]
fn separated_bundling_has_no_companions() {
    // Diffusers HF entries are self-contained; companions only matter for
    // single-file checkpoints that don't bundle their text encoders.
    assert!(companions_for(Family::Flux, None, Bundling::Separated, Kind::Checkpoint).is_empty());
    assert!(companions_for(Family::Sdxl, None, Bundling::Separated, Kind::Checkpoint).is_empty());
}

#[test]
fn lora_kind_never_pulls_companions() {
    // LoRAs are self-contained safetensors patches — they are merged into a
    // base model that itself supplies T5/CLIP/VAE. Pulling companions for
    // them would silently waste tens of GB of disk per LoRA install and
    // confuse the install-completion check (`catalog_entry_installed`).
    for family in [
        Family::Flux,
        Family::Sdxl,
        Family::Sd15,
        Family::Flux2,
        Family::ZImage,
        Family::LtxVideo,
        Family::Ltx2,
    ] {
        for bundling in [Bundling::SingleFile, Bundling::Separated] {
            let names = companions_for(family, None, bundling, Kind::Lora);
            assert!(
                names.is_empty(),
                "Kind::Lora must have no companions (family={family:?}, bundling={bundling:?}); got {names:?}",
            );
        }
    }
}

#[test]
fn every_canonical_name_resolves() {
    for c in COMPANIONS {
        let resolved = companion_by_name(c.canonical_name).expect("resolves");
        assert_eq!(resolved.canonical_name, c.canonical_name);
    }
}

#[test]
fn z_image_te_canonical_is_committed() {
    // The exact text-encoder repo for Z-Image phase-4 is finalized when
    // single-file loaders land, but the canonical NAME is committed now
    // so phase-1 entries can reference it without rewrites.
    let c = companion_by_name("z-image-te").expect("z-image-te");
    assert_eq!(c.kind, Kind::TextEncoder);
    assert!(c.family_scope.contains(&Family::ZImage));
}
