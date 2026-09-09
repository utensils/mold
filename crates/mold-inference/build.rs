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
            // The pinned oracle, xatlas-python 0.0.9, compiles this exact
            // revision through CMake with `CMAKE_CXX_STANDARD 17` and
            // `CMAKE_BUILD_TYPE=Release`, i.e. `-O3 -DNDEBUG`. Both halves
            // matter and neither changes a valid result.
            //
            // NDEBUG resolves `XA_DEBUG` to 0 (xatlas.cpp:56-60). Without it
            // all 155 `XA_DEBUG_ASSERT` sites compile in: two branches on
            // every `Array::operator[]`, and a redundant `sqrtf` inside every
            // `normalize()` for its `isNormalized` check. Measured on four
            // retained Hunyuan3D meshes (226k-455k triangles) that is ~17% of
            // the unwrap, for byte-identical UVs.
            //
            // C++17 is also what makes the single-threaded scheduler below
            // compile: `TaskGroupHandle` carries a default member initializer,
            // so `destroyGroup({ i })` (xatlas.cpp:3310 in the pinned
            // revision) is aggregate initialization from C++14 onward and
            // ill-formed in C++11.
            .std("c++17")
            .opt_level(3)
            .define("NDEBUG", None)
            // Run xatlas on the calling thread. Upstream's own switch: the
            // `#ifndef` guard is at xatlas.cpp:71-73 in the pinned revision
            // and the inline scheduler it selects is at :3304-3400 there.
            //
            // The threaded `TaskScheduler` sizes its pool at
            // `hardware_concurrency() - 1` with no way to cap it, and waits in
            // a bare `while (group.ref > 0) std::this_thread::yield();`
            // (:3235-3236 upstream). This is a MEASUREMENT, not a claim that
            // the phases are serial — chart parameterization does fan out per
            // chart. On four retained Hunyuan3D meshes (226k-455k triangles)
            // on a 128-core host: 127 threads and 3.06 cores consumed, more
            // system time than user, against 1.00 core here for 7% more wall
            // clock and byte-identical UVs. Two cores of pure spin on a host
            // that is also serving GPU work costs more than 7% of a stage the
            // paint face budget already cut to seconds.
            .define("XA_MULTITHREADED", "0")
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
