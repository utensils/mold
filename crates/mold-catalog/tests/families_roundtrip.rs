use mold_catalog::families::{Family, ALL_FAMILIES};

#[test]
fn every_family_round_trips_through_strings() {
    for fam in ALL_FAMILIES {
        let s = fam.as_str();
        let back = Family::from_str(s).expect("from_str");
        assert_eq!(*fam, back, "round trip failed for {s}");
    }
}

#[test]
fn manifest_strings_are_stable() {
    // These are the strings the existing manifest.rs writes into
    // ModelManifest.family. Changing them is a breaking change for
    // everything that already keys off `family`. This test pins them.
    let cases = [
        (Family::Flux, "flux"),
        (Family::Flux2, "flux2"),
        (Family::Sd15, "sd15"),
        (Family::Sdxl, "sdxl"),
        (Family::ZImage, "z-image"),
        (Family::LtxVideo, "ltx-video"),
        (Family::Ltx2, "ltx2"),
        (Family::QwenImage, "qwen-image"),
        (Family::Wuerstchen, "wuerstchen"),
    ];
    for (fam, s) in cases {
        assert_eq!(fam.as_str(), s, "{:?} → {s}", fam);
        assert_eq!(Family::from_str(s).unwrap(), fam, "{s} → {:?}", fam);
    }
}

#[test]
fn unknown_family_string_returns_err() {
    assert!(Family::from_str("not-a-family").is_err());
    assert!(Family::from_str("").is_err());
}
