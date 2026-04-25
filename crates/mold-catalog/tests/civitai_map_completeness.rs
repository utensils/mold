use mold_catalog::civitai_map::{
    engine_phase_for, map_base_model, CIVITAI_BASE_MODELS, CIVITAI_DROPS,
};
use mold_catalog::entry::Bundling;
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
        assert_eq!(engine_phase_for(fam, Bundling::Separated), 1);
    }
}

#[test]
fn engine_phase_classifies_single_file_correctly() {
    assert_eq!(engine_phase_for(Family::Sd15, Bundling::SingleFile), 2);
    assert_eq!(engine_phase_for(Family::Sdxl, Bundling::SingleFile), 2);
    assert_eq!(engine_phase_for(Family::Flux, Bundling::SingleFile), 3);
    assert_eq!(engine_phase_for(Family::Flux2, Bundling::SingleFile), 3);
    assert_eq!(engine_phase_for(Family::ZImage, Bundling::SingleFile), 4);
    assert_eq!(engine_phase_for(Family::LtxVideo, Bundling::SingleFile), 5);
    assert_eq!(engine_phase_for(Family::Ltx2, Bundling::SingleFile), 5);
    // QwenImage / Wuerstchen single-file is out of scope and gets the 99 sentinel.
    assert_eq!(
        engine_phase_for(Family::QwenImage, Bundling::SingleFile),
        99
    );
    assert_eq!(
        engine_phase_for(Family::Wuerstchen, Bundling::SingleFile),
        99
    );
}
