#[path = "../build_support/h3_runtime_code_identity.rs"]
mod h3_runtime_code_identity;

use std::fs;
use std::path::Path;

const PRIVATE_SERVER_PATH: &str = "crates/mold-inference/src/minimax_h3/private_server.rs";

fn identity(private_server: &str, runtime: &str) -> String {
    h3_runtime_code_identity::identity_for_entries(&[
        (
            PRIVATE_SERVER_PATH.to_string(),
            private_server.as_bytes().to_vec(),
        ),
        (
            "crates/mold-inference/src/minimax_h3/private_runtime.rs".to_string(),
            runtime.as_bytes().to_vec(),
        ),
    ])
    .expect("synthetic runtime identity")
}

#[test]
fn runtime_identity_changes_for_code_but_not_reviewed_allowlist_values() {
    let empty = r#"const REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256: &[&str] = &[];
fn runtime_owner() { consume(); }
"#;
    let reviewed = r#"const REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256: &[&str] = &[
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
];
fn runtime_owner() { consume(); }
"#;
    let base = identity(empty, "fn execute() { denoise(); }");
    assert_eq!(base, identity(reviewed, "fn execute() { denoise(); }"));
    assert_ne!(base, identity(empty, "fn execute() { denoise_twice(); }"));
    assert_ne!(
        base,
        identity(
            "const REVIEWED_RUNTIME_QUALIFICATION_RECORD_SHA256: &[&str] = &[];\nfn runtime_owner() { consume_twice(); }\n",
            "fn execute() { denoise(); }",
        )
    );
}

#[test]
fn runtime_identity_changes_for_a_build_axis() {
    let entries = [("crates/mold-server/src/lib.rs".into(), b"runtime".to_vec())];
    let linux = [("TARGET".into(), "x86_64-unknown-linux-gnu".into())];
    let macos = [("TARGET".into(), "aarch64-apple-darwin".into())];
    let linux_identity =
        h3_runtime_code_identity::identity_for_entries_and_environment(&entries, &linux).unwrap();
    let macos_identity =
        h3_runtime_code_identity::identity_for_entries_and_environment(&entries, &macos).unwrap();
    assert_ne!(linux_identity, macos_identity);
}

#[test]
fn canonical_private_server_features_reject_an_extra_axis() {
    let canonical = h3_runtime_code_identity::CANONICAL_H3_SERVER_FEATURES
        .iter()
        .map(|feature| (*feature).to_string())
        .collect::<Vec<_>>();
    h3_runtime_code_identity::validate_canonical_h3_server_feature_keys(&canonical).unwrap();

    let mut extra = canonical;
    extra.push("CARGO_FEATURE_METRICS".into());
    let error =
        h3_runtime_code_identity::validate_canonical_h3_server_feature_keys(&extra).unwrap_err();
    assert!(error.contains("canonical campaign build"), "{error}");
}

#[test]
fn runtime_identity_binds_the_native_cuda_toolchain() {
    // An in-place toolkit upgrade leaves CUDA_HOME, rustc -vV, and every source
    // input untouched while changing the compiled executable, so the reported
    // compiler versions must reach the identity and the binaries must be
    // watched for replacement.
    let (entries, watched) = h3_runtime_code_identity::native_toolchain_identity_for(
        true,
        Some(Path::new(env!("CARGO")).to_path_buf()),
        Path::new(env!("CARGO")).to_path_buf(),
    )
    .expect("synthetic toolchain identity");
    let keys = entries
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    assert!(keys.contains(&"MOLD_H3_NVCC_VERSION"), "{keys:?}");
    assert!(keys.contains(&"MOLD_H3_HOST_COMPILER_VERSION"), "{keys:?}");
    // Two toolkits can print the same version banner from different prefixes,
    // so which one was selected is part of the identity.
    assert!(keys.contains(&"MOLD_H3_NVCC_PATH"), "{keys:?}");
    assert!(keys.contains(&"MOLD_H3_HOST_COMPILER_PATH"), "{keys:?}");
    assert_eq!(watched.len(), 2, "{watched:?}");

    let inputs = [("crates/mold-server/src/lib.rs".to_string(), b"x".to_vec())];
    let older = [(
        "MOLD_H3_NVCC_VERSION".to_string(),
        "release 12.8".to_string(),
    )];
    let newer = [(
        "MOLD_H3_NVCC_VERSION".to_string(),
        "release 12.9".to_string(),
    )];
    assert_ne!(
        h3_runtime_code_identity::identity_for_entries_and_environment(&inputs, &older).unwrap(),
        h3_runtime_code_identity::identity_for_entries_and_environment(&inputs, &newer).unwrap(),
    );
}

#[test]
fn cuda_campaign_builds_reject_an_unresolvable_toolkit() {
    let error = h3_runtime_code_identity::native_toolchain_identity_for(true, None, "cc".into())
        .unwrap_err();
    assert!(error.contains("nvcc"), "{error}");

    let error = h3_runtime_code_identity::native_toolchain_identity_for(
        true,
        Some("/nonexistent/bin/nvcc".into()),
        "cc".into(),
    )
    .unwrap_err();
    assert!(error.contains("nvcc"), "{error}");
}

#[test]
fn cuda_compiler_resolution_mirrors_cudaforge() {
    // Naming a different nvcc than the one that compiled the kernels would be
    // worse than naming none, so the precedence must match
    // `cudaforge-0.1.6/src/toolkit.rs:108-135` exactly: NVCC, then PATH, then
    // CUDA_HOME, then CUDA_PATH's Windows `nvcc.exe`.
    let present = |_: &Path| true;
    let missing = |_: &Path| false;
    let resolve = h3_runtime_code_identity::resolve_cuda_compiler_from;

    assert_eq!(
        resolve(
            Some("/override/nvcc".into()),
            Some("/path/nvcc".into()),
            Some("/home".into()),
            None,
            &present,
        ),
        Some(Path::new("/override/nvcc").to_path_buf()),
        "the NVCC override outranks PATH",
    );
    assert_eq!(
        resolve(
            Some("/override/nvcc".into()),
            Some("/path/nvcc".into()),
            Some("/home".into()),
            None,
            &missing,
        ),
        Some(Path::new("/path/nvcc").to_path_buf()),
        "an NVCC override that does not exist falls through to PATH, not to CUDA_HOME",
    );
    assert_eq!(
        resolve(None, None, Some("/home".into()), None, &present),
        Some(Path::new("/home/bin/nvcc").to_path_buf()),
        "CUDA_HOME is the third choice, not the first",
    );
    assert_eq!(
        resolve(None, None, None, Some("/win".into()), &present),
        Some(Path::new("/win/bin/nvcc.exe").to_path_buf()),
    );
    // cudaforge's fifth step sweeps standard prefixes, so a /usr/local/cuda
    // install that is not on PATH still compiles the kernels and must still be
    // identified rather than failing the campaign build.
    assert_eq!(
        resolve(None, None, None, None, &|path: &Path| path
            == Path::new("/usr/local/cuda/bin/nvcc")),
        Some(Path::new("/usr/local/cuda/bin/nvcc").to_path_buf()),
    );
    assert_eq!(
        resolve(None, None, Some("/home".into()), None, &missing),
        None,
        "an unresolvable toolkit fails closed rather than guessing",
    );
}

#[cfg(unix)]
#[test]
fn directory_valued_ccbin_resolves_to_its_compiler() {
    use std::os::unix::fs::PermissionsExt;

    // `nvcc --compiler-bindir` documents a *directory*, so running the value
    // itself would abort a campaign build that nvcc compiles happily.
    let bindir = tempfile::tempdir().unwrap();
    let compiler = bindir.path().join("gcc");
    fs::write(&compiler, "#!/bin/sh\n").unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(
        h3_runtime_code_identity::resolve_host_compiler(bindir.path().to_path_buf()),
        compiler,
    );

    // A plain executable path is passed through untouched.
    assert_eq!(
        h3_runtime_code_identity::resolve_host_compiler(compiler.clone()),
        compiler,
    );

    // A non-executable file of the right name is not what cargo would run.
    let inert = tempfile::tempdir().unwrap();
    fs::write(inert.path().join("cc"), "not a program\n").unwrap();
    assert_eq!(
        h3_runtime_code_identity::resolve_host_compiler(inert.path().to_path_buf()),
        inert.path(),
        "a directory without a runnable compiler keeps the configured value",
    );
}

fn synthetic_workspace(root: &Path) {
    for relative in ["Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, format!("input={relative}\n")).unwrap();
    }
    for crate_name in [
        "mold-core",
        "mold-catalog",
        "mold-db",
        "mold-scheduler",
        "mold-candle",
        "mold-inference",
        "mold-server",
    ] {
        let crate_root = root.join("crates").join(crate_name);
        fs::create_dir_all(crate_root.join("src")).unwrap();
        fs::write(crate_root.join("Cargo.toml"), "[package]\nname='fixture'\n").unwrap();
        fs::write(crate_root.join("src/lib.rs"), "pub fn execute() {}\n").unwrap();
    }
    fs::write(
        root.join("crates/mold-server/Cargo.toml"),
        "[package]\nname='fixture'\n[dependencies]\ncore={path='../mold-core'}\ncatalog={path='../mold-catalog'}\ndb={path='../mold-db'}\ninference={path='../mold-inference'}\nscheduler={path='../mold-scheduler'}\n",
    )
    .unwrap();
    fs::write(
        root.join("crates/mold-inference/Cargo.toml"),
        "[package]\nname='fixture'\n[dependencies]\ncore={path='../mold-core'}\ncatalog={path='../mold-catalog'}\ncandle={path='../mold-candle'}\n",
    )
    .unwrap();
    fs::write(root.join("crates/mold-core/build.rs"), "fn main() {}\n").unwrap();
    fs::write(
        root.join("crates/mold-server/src/kernel.ptx"),
        ".version 8.0\n",
    )
    .unwrap();
}

fn workspace_identity(root: &Path) -> String {
    let inputs = h3_runtime_code_identity::collect_runtime_inputs(root).unwrap();
    h3_runtime_code_identity::identity_for_workspace(root, &inputs).unwrap()
}

#[test]
fn runtime_identity_covers_build_scripts_and_non_rust_inputs() {
    let workspace = tempfile::tempdir().unwrap();
    synthetic_workspace(workspace.path());
    let base = workspace_identity(workspace.path());

    fs::write(
        workspace.path().join("crates/mold-core/build.rs"),
        "fn main() { configure(); }\n",
    )
    .unwrap();
    let build_script_changed = workspace_identity(workspace.path());
    assert_ne!(base, build_script_changed);

    fs::write(
        workspace.path().join("crates/mold-server/src/kernel.ptx"),
        ".version 8.1\n",
    )
    .unwrap();
    assert_ne!(build_script_changed, workspace_identity(workspace.path()));
}

#[cfg(unix)]
#[test]
fn runtime_identity_rejects_symbolic_link_inputs() {
    use std::os::unix::fs::symlink;

    let workspace = tempfile::tempdir().unwrap();
    synthetic_workspace(workspace.path());
    symlink(
        "lib.rs",
        workspace
            .path()
            .join("crates/mold-server/src/linked-runtime.rs"),
    )
    .unwrap();

    let error = h3_runtime_code_identity::collect_runtime_inputs(workspace.path()).unwrap_err();
    assert!(error.contains("symbolic link"), "{error}");
}
