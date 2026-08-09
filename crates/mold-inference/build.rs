#[path = "build_support/h3_runtime_code_identity.rs"]
mod h3_runtime_code_identity;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_H3_PRIVATE_UAT");
    if std::env::var_os("CARGO_FEATURE_H3_PRIVATE_UAT").is_none() {
        return;
    }
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
    );
    let workspace_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("mold-inference must be nested under the workspace crates directory");
    let inputs = h3_runtime_code_identity::collect_runtime_inputs(workspace_root)
        .expect("failed to enumerate private H3 runtime identity inputs");
    let environment = h3_runtime_code_identity::collect_build_environment()
        .expect("failed to capture private H3 runtime build environment");
    let identity = h3_runtime_code_identity::identity_for_workspace_and_environment(
        workspace_root,
        &inputs,
        &environment,
    )
    .expect("failed to hash private H3 runtime identity inputs");
    for input in &inputs {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    for key in h3_runtime_code_identity::build_environment_rerun_keys() {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rustc-env=MOLD_H3_RUNTIME_CODE_IDENTITY_SHA256={identity}");
}
