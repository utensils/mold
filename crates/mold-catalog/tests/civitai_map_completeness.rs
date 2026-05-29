use mold_catalog::civitai_map::{
    engine_phase_for, map_base_model, CIVITAI_BASE_MODELS, CIVITAI_DROPS,
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

#[test]
fn unknown_strings_drop_silently() {
    assert!(map_base_model("Some Future Model 9000").is_none());
}

#[test]
fn engine_phase_classifies_separated_as_one() {
    for fam in [Family::Flux, Family::Sdxl, Family::Sd15, Family::ZImage] {
        assert_eq!(
            engine_phase_for(fam, Bundling::Separated, Kind::Checkpoint),
            1
        );
    }
}

#[test]
fn engine_phase_classifies_single_file_correctly() {
    assert_eq!(
        engine_phase_for(Family::Sd15, Bundling::SingleFile, Kind::Checkpoint),
        2
    );
    assert_eq!(
        engine_phase_for(Family::Sdxl, Bundling::SingleFile, Kind::Checkpoint),
        2
    );
    assert_eq!(
        engine_phase_for(Family::Flux, Bundling::SingleFile, Kind::Checkpoint),
        3
    );
    // Flux.2 is wired through the catalog bridge — phase 1 (runnable).
    assert_eq!(
        engine_phase_for(Family::Flux2, Bundling::SingleFile, Kind::Checkpoint),
        1
    );
    assert_eq!(
        engine_phase_for(Family::ZImage, Bundling::SingleFile, Kind::Checkpoint),
        4
    );
    assert_eq!(
        engine_phase_for(Family::LtxVideo, Bundling::SingleFile, Kind::Checkpoint),
        5
    );
    assert_eq!(
        engine_phase_for(Family::Ltx2, Bundling::SingleFile, Kind::Checkpoint),
        5
    );
    assert_eq!(
        engine_phase_for(Family::QwenImage, Bundling::SingleFile, Kind::Checkpoint),
        1
    );
    assert_eq!(
        engine_phase_for(Family::Wuerstchen, Bundling::SingleFile, Kind::Checkpoint),
        1
    );
    assert_eq!(
        engine_phase_for(Family::Sd15, Bundling::SingleFile, Kind::ControlNet),
        1
    );
    assert_eq!(
        engine_phase_for(Family::Sdxl, Bundling::SingleFile, Kind::ControlNet),
        1
    );
}

#[test]
fn engine_phase_never_uses_legacy_unsupported_sentinel() {
    const LEGACY_UNSUPPORTED_SENTINEL: u8 = 100 - 1;
    for family in mold_catalog::families::ALL_FAMILIES {
        for bundling in [Bundling::Separated, Bundling::SingleFile] {
            for kind in [
                Kind::Checkpoint,
                Kind::Lora,
                Kind::Vae,
                Kind::TextEncoder,
                Kind::Tokenizer,
                Kind::Clip,
                Kind::ControlNet,
            ] {
                assert_ne!(
                    engine_phase_for(*family, bundling, kind),
                    LEGACY_UNSUPPORTED_SENTINEL
                );
            }
        }
    }
}
