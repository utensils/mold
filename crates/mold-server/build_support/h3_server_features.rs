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

pub const PUBLIC_H3_SERVER_FEATURES: &[&str] = &[
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_H3",
    "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
    "CARGO_FEATURE_MP4",
    "CARGO_FEATURE_NVML",
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
    let expected = if public {
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

    #[test]
    fn public_h3_rejects_missing_or_private_authority_edges() {
        let mut missing = features(PUBLIC_H3_SERVER_FEATURES);
        missing.retain(|feature| feature != "CARGO_FEATURE_NVML");
        assert!(validate_canonical_h3_server_feature_keys(&missing).is_err());

        let mut crossed = features(PUBLIC_H3_SERVER_FEATURES);
        crossed.push("CARGO_FEATURE_H3_PRIVATE_UAT".into());
        assert!(validate_canonical_h3_server_feature_keys(&crossed).is_err());
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
