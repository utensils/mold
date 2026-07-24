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
