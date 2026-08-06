use mold_catalog::civitai_map::{
    map_base_model, supported_for, CIVITAI_BASE_MODELS, CIVITAI_DROPS,
};
use mold_catalog::entry::{Bundling, Kind};
use mold_catalog::families::Family;

#[test]
fn every_known_base_model_maps_or_is_explicitly_dropped() {
    for key in CIVITAI_BASE_MODELS {
        let mapped = map_base_model(key).is_some();
        let dropped = CIVITAI_DROPS.contains(key);
        assert!(
            mapped ^ dropped,
            "civitai baseModel '{key}' must either map to a Family OR be in CIVITAI_DROPS — never both, never neither"
        );
    }
}

#[test]
fn pony_keeps_sub_family() {
    let (fam, _role, sub) = map_base_model("Pony").unwrap();
    assert_eq!(fam, Family::Sdxl);
    assert_eq!(sub, Some("pony".to_string()));
}

/// The `sub_family` a Wan row carries is a string contract between three
/// modules that never reference each other: `civitai_map` emits it,
/// `companions_for` picks the VAE from it, and `runtime_defaults_for_family`
/// picks the frame timing from it. A typo in the producer does not fail — the
/// row quietly takes the family default, which is right for four of the five
/// mappings and wrong for the fifth. This closes that loop end to end.
#[test]
fn wan_base_models_reach_the_right_vae_and_timing() {
    use mold_catalog::companions::companions_for;
    use mold_catalog::defaults::runtime_defaults_for_family;

    for (base_model, want_vae, want_frames, want_fps) in [
        ("Wan Video 1.3B t2v", "wan21-vae", 81, 16),
        ("Wan Video 14B t2v", "wan21-vae", 81, 16),
        ("Wan Video 2.2 T2V-A14B", "wan21-vae", 81, 16),
        ("Wan Video 2.2 I2V-A14B", "wan21-vae", 81, 16),
        // The one that differs, and the reason this test exists.
        ("Wan Video 2.2 TI2V-5B", "wan22-vae", 121, 24),
    ] {
        let (family, _role, sub) = map_base_model(base_model)
            .unwrap_or_else(|| panic!("{base_model} must map to a family"));
        assert_eq!(family, Family::Wan, "{base_model}");

        let companions = companions_for(
            family,
            sub.as_deref(),
            Bundling::SingleFile,
            Kind::Checkpoint,
        );
        assert!(
            companions.contains(&want_vae.to_string()),
            "{base_model} (sub_family {sub:?}) must pull {want_vae}; got {companions:?}"
        );
        assert!(
            companions.contains(&"wan-umt5".to_string()),
            "{base_model} must pull the UMT5 encoder; got {companions:?}"
        );

        let defaults = runtime_defaults_for_family(family.as_str(), sub.as_deref());
        assert_eq!(
            (defaults.frames, defaults.fps),
            (Some(want_frames), Some(want_fps)),
            "{base_model} (sub_family {sub:?}) timing"
        );
    }

    // The Wan checkpoints this build cannot run must stay dropped: 2.1 I2V
    // needs a CLIP-vision branch mold's DiT does not implement, and 2.5/2.7
    // are later architectures with no engine at all. Mapping any of them would
    // offer a multi-gigabyte install that cannot generate.
    for base_model in [
        "Wan Video 14B i2v 480p",
        "Wan Video 14B i2v 720p",
        "Wan Video 2.5 T2V",
        "Wan Video 2.5 I2V",
        "Wan Image 2.7",
        "Wan Video 2.7",
    ] {
        assert!(
            map_base_model(base_model).is_none(),
            "{base_model} has no mold engine and must not map"
        );
        assert!(CIVITAI_DROPS.contains(&base_model), "{base_model}");
    }
}

#[test]
fn unknown_strings_drop_silently() {
    assert!(map_base_model("Some Future Model 9000").is_none());
}

#[test]
fn checkpoints_are_supported_for_both_bundle_layouts() {
    for fam in [Family::Flux, Family::Sdxl, Family::Sd15, Family::ZImage] {
        assert!(supported_for(fam, Bundling::Separated, Kind::Checkpoint));
        assert!(supported_for(fam, Bundling::SingleFile, Kind::Checkpoint));
    }
}

#[test]
fn support_is_a_direct_capability() {
    assert!(supported_for(
        Family::Sd15,
        Bundling::SingleFile,
        Kind::ControlNet
    ));
    assert!(!supported_for(
        Family::Flux,
        Bundling::SingleFile,
        Kind::ControlNet
    ));
    assert!(supported_for(
        Family::Ltx2,
        Bundling::SingleFile,
        Kind::Checkpoint
    ));
}
