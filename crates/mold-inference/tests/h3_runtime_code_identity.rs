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
