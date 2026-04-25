use mold_catalog::companions::{companion_by_name, companions_for, COMPANIONS};
use mold_catalog::entry::{Bundling, Kind};
use mold_catalog::families::Family;

#[test]
fn flux_single_file_needs_t5_and_clip_l() {
    let names = companions_for(Family::Flux, Bundling::SingleFile);
    assert!(names.contains(&"t5-v1_1-xxl".to_string()));
    assert!(names.contains(&"clip-l".to_string()));
}

#[test]
fn sdxl_single_file_needs_two_clips_and_vae() {
    let names = companions_for(Family::Sdxl, Bundling::SingleFile);
    assert!(names.contains(&"clip-l".to_string()));
    assert!(names.contains(&"clip-g".to_string()));
    assert!(names.contains(&"sdxl-vae".to_string()));
}

#[test]
fn separated_bundling_has_no_companions() {
    // Diffusers HF entries are self-contained; companions only matter for
    // single-file checkpoints that don't bundle their text encoders.
    assert!(companions_for(Family::Flux, Bundling::Separated).is_empty());
    assert!(companions_for(Family::Sdxl, Bundling::Separated).is_empty());
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
