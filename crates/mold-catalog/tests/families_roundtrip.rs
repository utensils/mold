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
        (Family::Wan, "wan"),
        (Family::QwenImage, "qwen-image"),
        (Family::Wuerstchen, "wuerstchen"),
    ];
    for (fam, s) in cases {
        assert_eq!(fam.as_str(), s, "{:?} → {s}", fam);
        assert_eq!(Family::from_str(s).unwrap(), fam, "{s} → {:?}", fam);
    }
}

/// `is_video` and the runtime-defaults table are two statements of the same
/// fact, read by different parts of the catalog: the predicate decides a row's
/// `Modality`, and the defaults decide whether it carries `frames`/`fps`. A
/// family that is video by one and not the other produces a row that either
/// shows duration controls it cannot honour, or hides controls it needs.
#[test]
fn video_families_agree_with_the_runtime_defaults_table() {
    use mold_catalog::defaults::runtime_defaults_for_family;

    for family in ALL_FAMILIES {
        let defaults = runtime_defaults_for_family(family.as_str(), None);
        assert_eq!(
            family.is_video(),
            defaults.frames.is_some(),
            "{family} disagrees: is_video()={}, frames={:?}",
            family.is_video(),
            defaults.frames
        );
        assert_eq!(
            defaults.frames.is_some(),
            defaults.fps.is_some(),
            "{family} must carry both frames and fps, or neither"
        );
    }

    // The three that are video today, named explicitly so removing one from
    // `is_video` fails here rather than quietly reclassifying its catalog rows.
    assert!(Family::Wan.is_video());
    assert!(Family::Ltx2.is_video());
    assert!(Family::LtxVideo.is_video());
    assert!(!Family::Flux.is_video());
}

#[test]
fn unknown_family_string_returns_err() {
    assert!(Family::from_str("not-a-family").is_err());
    assert!(Family::from_str("").is_err());
}
