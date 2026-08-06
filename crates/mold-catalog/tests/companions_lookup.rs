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
fn ltx2_single_file_pulls_gemma_te_and_matching_vae_not_t5() {
    // LTX-2 / LTX-2.3 use Gemma 3 12B as the text encoder, not T5. Civitai
    // single-file LTX-2 checkpoints (e.g. cv:2752735, cv:2781713) bundle
    // the transformer + VAE but ship no encoder, and the runtime
    // (`ltx2/assets.rs::gemma_root`) reads the parent of the first entry
    // in `paths.text_encoder_files` — a missing or T5-shaped encoder fails
    // the load with `LTX-2 requires Gemma text encoder files to be
    // available`. Pin the correct companion so a future scope-creep edit
    // doesn't silently swap T5 back in.
    let names = companions_for(
        Family::Ltx2,
        Some("v2.3"),
        Bundling::SingleFile,
        Kind::Checkpoint,
    );
    assert!(
        names.contains(&"ltx2-te".to_string()),
        "LTX-2 must pull the Gemma TE companion; got {names:?}"
    );
    assert!(
        !names.contains(&"t5-v1_1-xxl".to_string()),
        "LTX-2 must NOT pull T5 (unused, ~9.5 GB wasted download); got {names:?}"
    );
    assert!(
        names.contains(&"ltx2.3-vae".to_string()),
        "LTX-2.3 transformer-only checkpoints need the matching video VAE; got {names:?}"
    );
    assert!(
        names.contains(&"ltx2.3-text-projection".to_string()),
        "LTX-2.3 diffusion-only checkpoints need the Gemma projection; got {names:?}"
    );
}

#[test]
fn ltx2_v2_single_file_uses_v2_vae() {
    let names = companions_for(
        Family::Ltx2,
        Some("v2"),
        Bundling::SingleFile,
        Kind::Checkpoint,
    );
    assert!(names.contains(&"ltx2-vae".to_string()), "got {names:?}");
    assert!(!names.contains(&"ltx2.3-vae".to_string()), "got {names:?}");
    assert!(
        !names.contains(&"ltx2.3-text-projection".to_string()),
        "got {names:?}"
    );
}

/// Wan's two VAEs are different architectures, and the sub-family names are
/// actively misleading about which is which: **the 2.2 A14B pair uses the 2.1
/// VAE**, the same 16-channel file the 1.3B uses. Only TI2V-5B takes the
/// 48-channel 2.2 VAE. A `starts_with("wan22")` rule would hand both A14B
/// experts a VAE their DiT rejects, after a 10 GB download.
#[test]
fn wan_a14b_takes_the_2_1_vae_despite_its_2_2_name() {
    let names_for = |sub: Option<&str>| {
        companions_for(Family::Wan, sub, Bundling::SingleFile, Kind::Checkpoint)
    };

    for sub in [
        Some("wan22-t2v-a14b"),
        Some("wan22-i2v-a14b"),
        Some("wan21-t2v-1.3b"),
        Some("wan21-t2v-14b"),
        // Unknown community fine-tunes are overwhelmingly 14B-class.
        None,
    ] {
        let names = names_for(sub);
        assert!(
            names.contains(&"wan21-vae".to_string()),
            "{sub:?} must take the 2.1 VAE; got {names:?}"
        );
        assert!(
            !names.contains(&"wan22-vae".to_string()),
            "{sub:?} must not take the 2.2 VAE; got {names:?}"
        );
    }

    let five_b = names_for(Some("wan22-ti2v-5b"));
    assert!(five_b.contains(&"wan22-vae".to_string()), "got {five_b:?}");
    assert!(!five_b.contains(&"wan21-vae".to_string()), "got {five_b:?}");
}

/// Every Wan checkpoint in the wild is transformer-only, so the UMT5 encoder
/// always rides along — and it must be UMT5, not the T5 the LTX-Video arm
/// pulls. UMT5 has per-layer relative attention bias that candle's shared-bias
/// T5 cannot express, which is why mold ships a separate encoder at all.
#[test]
fn wan_single_file_pulls_umt5_not_t5() {
    let names = companions_for(Family::Wan, None, Bundling::SingleFile, Kind::Checkpoint);
    assert!(
        names.contains(&"wan-umt5".to_string()),
        "Wan must pull its UMT5 encoder; got {names:?}"
    );
    assert!(
        !names.contains(&"t5-v1_1-xxl".to_string()),
        "Wan must NOT pull T5 (~9.5 GB of weights its engine cannot load); got {names:?}"
    );
}

#[test]
fn wan_companions_match_the_manifest_companions() {
    // The catalog registry and mold-core's `wan-umt5` / `wan21-vae` /
    // `wan22-vae` manifests must name the same repo and file, or a cv:/hf:
    // install re-downloads assets a manifest pull already placed on disk
    // instead of hitting the shared/wan/ cache.
    let umt5 = companion_by_name("wan-umt5").expect("wan-umt5");
    assert_eq!(umt5.kind, Kind::TextEncoder);
    assert_eq!(umt5.repo, "Comfy-Org/Wan_2.1_ComfyUI_repackaged");
    assert!(umt5
        .files
        .contains(&"split_files/text_encoders/umt5_xxl_fp16.safetensors"));

    let vae21 = companion_by_name("wan21-vae").expect("wan21-vae");
    assert_eq!(vae21.kind, Kind::Vae);
    assert!(vae21
        .files
        .contains(&"split_files/vae/wan_2.1_vae.safetensors"));

    let vae22 = companion_by_name("wan22-vae").expect("wan22-vae");
    assert_eq!(vae22.kind, Kind::Vae);
    assert!(vae22
        .files
        .contains(&"split_files/vae/wan2.2_vae.safetensors"));

    for companion in [umt5, vae21, vae22] {
        assert!(companion.family_scope.contains(&Family::Wan));
    }
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
    let c = companion_by_name("z-image-te").expect("z-image-te");
    assert_eq!(c.kind, Kind::TextEncoder);
    assert!(c.family_scope.contains(&Family::ZImage));
    assert_eq!(c.repo, "Tongyi-MAI/Z-Image-Turbo");
    assert!(c
        .files
        .contains(&"text_encoder/model-00001-of-00003.safetensors"));
    assert!(c.files.contains(&"tokenizer/tokenizer.json"));
    assert!(c.files.contains(&"vae/diffusion_pytorch_model.safetensors"));
}
