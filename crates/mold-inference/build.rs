#[path = "build_support/h3_runtime_code_identity.rs"]
mod h3_runtime_code_identity;

use std::path::PathBuf;

fn main() {
    #[cfg(feature = "mesh-texture")]
    {
        for file in ["xatlas.cpp", "xatlas.h", "bridge.cpp"] {
            println!("cargo:rerun-if-changed=vendor/xatlas/{file}");
        }
        cc::Build::new()
            .cpp(true)
            .std("c++11")
            .opt_level(3)
            .file("vendor/xatlas/xatlas.cpp")
            .file("vendor/xatlas/bridge.cpp")
            .include("vendor/xatlas")
            .warnings(false)
            .compile("mold_xatlas");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_H3");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_H3_PRIVATE_UAT");
    if std::env::var_os("CARGO_FEATURE_H3").is_none()
        && std::env::var_os("CARGO_FEATURE_H3_PRIVATE_UAT").is_none()
    {
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
    // A toolkit replaced in place changes no tracked source, so watch the
    // native compilers themselves or the cached identity would outlive them.
    let (_, native_toolchain) = h3_runtime_code_identity::collect_native_toolchain_identity()
        .expect("failed to bind the private H3 native toolchain identity");
    for binary in &native_toolchain {
        println!("cargo:rerun-if-changed={}", binary.display());
    }
    for key in h3_runtime_code_identity::build_environment_rerun_keys() {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rustc-env=MOLD_H3_RUNTIME_CODE_IDENTITY_SHA256={identity}");
}
