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

pub fn validate_canonical_h3_server_features() -> Result<(), String> {
    if std::env::var_os("CARGO_FEATURE_H3_PRIVATE_UAT").is_none() {
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
    let expected = CANONICAL_H3_SERVER_FEATURES
        .iter()
        .map(|key| (*key).to_string())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "private H3 server features differ from the canonical campaign build: expected {expected:?}, actual {actual:?}"
        ));
    }
    Ok(())
}

pub fn canonical_h3_server_feature_rerun_keys() -> impl Iterator<Item = &'static str> {
    CANONICAL_H3_SERVER_FEATURES
        .iter()
        .copied()
        .chain(["CARGO_FEATURE_DEFAULT"])
}
