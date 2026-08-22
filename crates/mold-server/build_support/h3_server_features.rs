//! Canonical feature set for the private MiniMax H3 campaign server build.
//!
//! This lives inside `mold-server` on purpose. A published `.crate` contains
//! only files under its own crate directory, so a build script that reached
//! into `../mold-inference/` would compile in the workspace and then fail
//! `cargo publish`'s verification build. `crates/mold-inference/build_support/
//! h3_runtime_code_identity.rs` keeps the same list because it stamps the set
//! into the runtime-code identity hash; `scripts/tests/
//! minimax-h3-private-uat-release-contract.sh` fails if the two drift.

#![allow(dead_code)]

use std::collections::BTreeSet;

pub const CANONICAL_H3_SERVER_FEATURES: &[&str] = &[
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
    "CARGO_FEATURE_H3_PRIVATE_UAT",
    "CARGO_FEATURE_MP4",
    "CARGO_FEATURE_NVML",
];

/// The reviewed CUDA H3 build.
///
/// `CARGO_FEATURE_H3_CUDA` is REQUIRED, not merely allowed. `h3` alone stopped
/// implying `cuda` in #1164 so that an Apple Silicon build could be expressed
/// at all, and the same change stopped it implying the SM89 attention kernel.
/// A published recipe still written as `cuda,h3` would therefore build a
/// CUDA H3 binary with no H3 FlashAttention kernel — a graph this fence used
/// to accept, leaving `scripts/verify-h3-release-exclusion.sh` to reject the
/// artifact after it was built. Requiring the edge moves that detection to
/// the build script, before any artifact exists.
pub const PUBLIC_H3_SERVER_FEATURES: &[&str] = &[
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_H3",
    "CARGO_FEATURE_H3_CUDA",
    "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
    "CARGO_FEATURE_MP4",
    "CARGO_FEATURE_NVML",
];

/// The reviewed Apple Silicon H3 build (#1164).
///
/// It is a separate exact set rather than a relaxation of the CUDA one, for
/// the same reason that one exists: a reviewed build graph is the artifact,
/// and "CUDA optional" would admit a build with neither device. `NVML` is
/// absent because it is NVIDIA telemetry with nothing to read on a Mac, and
/// `CUDA` is absent because it cannot compile there at all.
pub const PUBLIC_H3_METAL_SERVER_FEATURES: &[&str] = &[
    "CARGO_FEATURE_H3",
    "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
    "CARGO_FEATURE_METAL",
    "CARGO_FEATURE_MP4",
];

/// Established server capabilities that do not alter H3 model execution.
///
/// Public release surfaces compose H3 with these ordinary server features.
/// Keep the list explicit so a new feature cannot silently enter the reviewed
/// public H3 build graph.
pub const PUBLIC_H3_ORTHOGONAL_FEATURES: &[&str] = &[
    "CARGO_FEATURE_EXPAND",
    "CARGO_FEATURE_MDNS",
    "CARGO_FEATURE_METRICS",
    // PuLID face identity (#1223). Orthogonal in the strongest sense: it is
    // qualified for two FLUX checkpoints and `mold_core::identity` refuses it
    // for every other model, so it cannot reach an H3 render at all. It is in
    // this list rather than the reviewed sets because it entered every
    // shipping recipe, and this fence is an exact allowlist — omitting it
    // would fail the SM89 release build at its build script.
    "CARGO_FEATURE_PULID",
    "CARGO_FEATURE_WEBP",
];

pub fn validate_canonical_h3_server_features() -> Result<(), String> {
    if std::env::var_os("CARGO_FEATURE_H3").is_none()
        && std::env::var_os("CARGO_FEATURE_H3_PRIVATE_UAT").is_none()
    {
        return Ok(());
    }
    let actual = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| key.starts_with("CARGO_FEATURE_") && key != "CARGO_FEATURE_DEFAULT")
        .collect::<Vec<_>>();
    validate_canonical_h3_server_feature_keys(&actual)
}

pub fn validate_canonical_h3_server_feature_keys(actual: &[String]) -> Result<(), String> {
    let actual = actual.iter().cloned().collect::<BTreeSet<_>>();
    let public = actual.contains("CARGO_FEATURE_H3");
    let metal =
        public && actual.contains("CARGO_FEATURE_METAL") && !actual.contains("CARGO_FEATURE_CUDA");
    let expected = if metal {
        PUBLIC_H3_METAL_SERVER_FEATURES
    } else if public {
        PUBLIC_H3_SERVER_FEATURES
    } else {
        CANONICAL_H3_SERVER_FEATURES
    }
    .iter()
    .map(|key| (*key).to_string())
    .collect::<BTreeSet<_>>();
    let allowed = if public {
        expected
            .iter()
            .cloned()
            .chain(
                PUBLIC_H3_ORTHOGONAL_FEATURES
                    .iter()
                    .map(|key| (*key).to_string()),
            )
            .collect::<BTreeSet<_>>()
    } else {
        expected.clone()
    };
    if !expected.is_subset(&actual) || !actual.is_subset(&allowed) {
        return Err(format!(
            "H3 server features differ from the reviewed build: required {expected:?}, allowed {allowed:?}, actual {actual:?}"
        ));
    }
    Ok(())
}

pub fn canonical_h3_server_feature_rerun_keys() -> impl Iterator<Item = &'static str> {
    CANONICAL_H3_SERVER_FEATURES
        .iter()
        .copied()
        .chain(PUBLIC_H3_SERVER_FEATURES.iter().copied())
        .chain(PUBLIC_H3_METAL_SERVER_FEATURES.iter().copied())
        .chain(PUBLIC_H3_ORTHOGONAL_FEATURES.iter().copied())
        .chain(["CARGO_FEATURE_DEFAULT"])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn public_h3_accepts_only_reviewed_orthogonal_features() {
        let mut desktop = features(PUBLIC_H3_SERVER_FEATURES);
        desktop.extend(features(&["CARGO_FEATURE_EXPAND", "CARGO_FEATURE_MDNS"]));
        validate_canonical_h3_server_feature_keys(&desktop).unwrap();

        let mut unknown = desktop;
        unknown.push("CARGO_FEATURE_TEST_SUPPORT".into());
        assert!(validate_canonical_h3_server_feature_keys(&unknown).is_err());
    }

    /// The regression this exists for: `cuda,h3` is a complete-looking CUDA
    /// H3 graph that silently omits the SM89 attention kernel. It must fail
    /// here, not at artifact verification.
    #[test]
    fn a_cuda_h3_recipe_without_the_shipping_edge_is_refused() {
        let mut without_edge = features(PUBLIC_H3_SERVER_FEATURES);
        without_edge.retain(|feature| feature != "CARGO_FEATURE_H3_CUDA");
        assert!(validate_canonical_h3_server_feature_keys(&without_edge).is_err());
    }

    /// The Metal set is not a back door into a kernel-less CUDA build: the
    /// selector only reaches it when CUDA is absent entirely.
    #[test]
    fn the_metal_set_cannot_stand_in_for_a_cuda_build() {
        let mut metal_plus_cuda = features(PUBLIC_H3_METAL_SERVER_FEATURES);
        metal_plus_cuda.push("CARGO_FEATURE_CUDA".into());
        assert!(validate_canonical_h3_server_feature_keys(&metal_plus_cuda).is_err());
    }

    #[test]
    fn public_h3_rejects_missing_or_private_authority_edges() {
        let mut missing = features(PUBLIC_H3_SERVER_FEATURES);
        missing.retain(|feature| feature != "CARGO_FEATURE_NVML");
        assert!(validate_canonical_h3_server_feature_keys(&missing).is_err());

        let mut crossed = features(PUBLIC_H3_SERVER_FEATURES);
        crossed.push("CARGO_FEATURE_H3_PRIVATE_UAT".into());
        assert!(validate_canonical_h3_server_feature_keys(&crossed).is_err());
    }

    /// The Apple Silicon build is its own reviewed set: the same orthogonal
    /// server features are allowed, but NVML and CUDA are not silently
    /// required, and a build carrying neither device still fails.
    #[test]
    fn public_metal_h3_is_its_own_reviewed_set() {
        let metal = features(PUBLIC_H3_METAL_SERVER_FEATURES);
        validate_canonical_h3_server_feature_keys(&metal).unwrap();

        let mut with_orthogonal = metal.clone();
        with_orthogonal.extend(features(&["CARGO_FEATURE_EXPAND", "CARGO_FEATURE_MDNS"]));
        validate_canonical_h3_server_feature_keys(&with_orthogonal).unwrap();

        // Neither device is not a build anyone reviewed.
        let mut deviceless = metal.clone();
        deviceless.retain(|feature| feature != "CARGO_FEATURE_METAL");
        assert!(validate_canonical_h3_server_feature_keys(&deviceless).is_err());

        // NVML has nothing to read on a Mac and is not part of the set.
        let mut with_nvml = metal;
        with_nvml.push("CARGO_FEATURE_NVML".into());
        assert!(validate_canonical_h3_server_feature_keys(&with_nvml).is_err());

        // A CUDA build still has to be the CUDA set, even if Metal is also on.
        let mut both = features(PUBLIC_H3_SERVER_FEATURES);
        both.push("CARGO_FEATURE_METAL".into());
        assert!(validate_canonical_h3_server_feature_keys(&both).is_err());
    }

    #[test]
    fn private_h3_campaign_remains_exact() {
        let canonical = features(CANONICAL_H3_SERVER_FEATURES);
        validate_canonical_h3_server_feature_keys(&canonical).unwrap();

        let mut widened = canonical;
        widened.push("CARGO_FEATURE_EXPAND".into());
        assert!(validate_canonical_h3_server_feature_keys(&widened).is_err());
    }
}
