//! The reviewed H3 build graph is an exact allowlist, and a shipping recipe
//! that grows a feature must be admitted to it deliberately.
//!
//! `build.rs` enforces `h3_server_features` at build time, which means a recipe
//! that violates it fails *while producing a release artifact* — on a CUDA
//! runner, minutes into a 40-minute job. These tests run the same validator
//! against the exact feature sets `flake.nix` and `.github/workflows/
//! release.yml` publish, so the collision is found here instead.
//!
//! #1223 is why this file exists: adding `pulid` to every shipping recipe put
//! `CARGO_FEATURE_PULID` into the SM89 H3 build's environment, which the
//! allowlist would have rejected.

#[path = "../build_support/h3_server_features.rs"]
mod h3_server_features;

use h3_server_features::validate_canonical_h3_server_feature_keys as validate;

/// `mold-ai`'s `h3-cuda,preview,discord,expand,tui,webp,mp4,metrics,mdns,pulid`
/// as it reaches `mold-ai-server`. Features `mold-ai` does not forward to the
/// server (`preview`, `discord`, `tui`) are deliberately absent.
fn shipping_sm89_recipe() -> Vec<String> {
    [
        "CARGO_FEATURE_CUDA",
        "CARGO_FEATURE_EXPAND",
        "CARGO_FEATURE_H3",
        "CARGO_FEATURE_H3_CUDA",
        "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
        "CARGO_FEATURE_MDNS",
        "CARGO_FEATURE_METRICS",
        "CARGO_FEATURE_MP4",
        "CARGO_FEATURE_NVML",
        "CARGO_FEATURE_PULID",
        "CARGO_FEATURE_WEBP",
    ]
    .iter()
    .map(|key| (*key).to_string())
    .collect()
}

/// The Apple Silicon H3 route (#1164): no CUDA, no NVML.
fn shipping_metal_recipe() -> Vec<String> {
    [
        "CARGO_FEATURE_EXPAND",
        "CARGO_FEATURE_H3",
        "CARGO_FEATURE_H3_PRIVATE_BRIDGE",
        "CARGO_FEATURE_MDNS",
        "CARGO_FEATURE_METAL",
        "CARGO_FEATURE_METRICS",
        "CARGO_FEATURE_MP4",
        "CARGO_FEATURE_PULID",
        "CARGO_FEATURE_WEBP",
    ]
    .iter()
    .map(|key| (*key).to_string())
    .collect()
}

#[test]
fn the_shipping_sm89_recipe_passes_the_reviewed_build_fence() {
    validate(&shipping_sm89_recipe())
        .unwrap_or_else(|error| panic!("the published SM89 recipe must build: {error}"));
}

#[test]
fn the_shipping_metal_recipe_passes_the_reviewed_build_fence() {
    validate(&shipping_metal_recipe())
        .unwrap_or_else(|error| panic!("the published Apple Silicon recipe must build: {error}"));
}

/// The allowlist still bites. `pulid` was admitted deliberately; an
/// unreviewed feature must not slip in behind it.
#[test]
fn an_unreviewed_feature_is_still_rejected() {
    let mut recipe = shipping_sm89_recipe();
    recipe.push("CARGO_FEATURE_SOMETHING_UNREVIEWED".to_string());
    let error = validate(&recipe).expect_err("an unreviewed axis must be refused");
    assert!(error.contains("differ from the reviewed build"), "{error}");
}

/// And dropping a REQUIRED edge is still rejected — `h3-cuda` in particular,
/// which is what keeps a `cuda,h3` recipe from shipping H3 with no attention
/// kernel.
#[test]
fn dropping_the_required_h3_cuda_edge_is_still_rejected() {
    let recipe: Vec<String> = shipping_sm89_recipe()
        .into_iter()
        .filter(|key| key != "CARGO_FEATURE_H3_CUDA")
        .collect();
    validate(&recipe).expect_err("h3-cuda is required, not merely allowed");
}
