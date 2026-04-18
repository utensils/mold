//! Stage the web gallery SPA so `rust-embed` can bake it into the binary.
//!
//! `rust-embed` expands its `#[folder = "..."]` attribute at macro-expansion
//! time, so the folder it points at must exist when the crate is compiled.
//! We resolve one of three paths, in order, and stamp the winning path into
//! `MOLD_EMBED_WEB_DIR` for `src/web_ui.rs`:
//!
//! 1. `$MOLD_WEB_DIST` — set by the Nix flake to a `mold-web` derivation that
//!    contains the built Vite output (`index.html` + `assets/`). This is how
//!    `nix build` produces a single binary with the real SPA embedded.
//! 2. `<crate>/../../web/dist` — the repo-relative Vite output. Populated by
//!    `cd web && bun run build`; useful for local release builds outside Nix.
//! 3. A stub generated in `$OUT_DIR/web-stub/` containing only a marker file
//!    (`__mold_placeholder`) and a minimal `index.html`. The server detects
//!    the marker and serves the existing inline "mold is running" placeholder
//!    instead, so `cargo build` in a fresh checkout still produces a working
//!    binary — just without the gallery UI.
//!
//! The binary always embeds something; runtime code never has to reason about
//! an empty embed.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PLACEHOLDER_MARKER: &str = "__mold_placeholder";
const PLACEHOLDER_INDEX_HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>mold</title></head>
<body>placeholder</body></html>
"#;

fn main() {
    println!("cargo:rerun-if-env-changed=MOLD_WEB_DIST");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Always watch the repo-relative `web/dist/index.html` — even when the
    // current resolution lands on the placeholder stub — so a later
    // `cd web && bun run build` invalidates this build script and rebakes
    // the real SPA on the next `cargo build`. Without this, a fresh
    // checkout that builds Rust before the SPA would stay stuck on the
    // placeholder until some unrelated input changes.
    let repo_dist_index = crate_dir
        .join("..")
        .join("..")
        .join("web")
        .join("dist")
        .join("index.html");
    println!("cargo:rerun-if-changed={}", repo_dist_index.display());

    let resolved = resolve_web_dist(&crate_dir, &out_dir);
    // Also watch the winning embed source itself so edits to the real
    // `dist/` (or, in the stub path, edits to the stub we wrote) trigger
    // re-runs.
    println!("cargo:rerun-if-changed={}", resolved.display());
    // Stamp the absolute path so `#[folder = "$MOLD_EMBED_WEB_DIR"]` resolves
    // deterministically regardless of which directory cargo invokes rustc from.
    println!("cargo:rustc-env=MOLD_EMBED_WEB_DIR={}", resolved.display());
}

fn resolve_web_dist(crate_dir: &Path, out_dir: &Path) -> PathBuf {
    if let Ok(dist) = env::var("MOLD_WEB_DIST") {
        let path = PathBuf::from(&dist);
        if path.join("index.html").is_file() {
            return path;
        }
        println!(
            "cargo:warning=MOLD_WEB_DIST={} has no index.html; falling back to stub",
            dist
        );
    }

    let repo_dist = crate_dir.join("..").join("..").join("web").join("dist");
    if repo_dist.join("index.html").is_file() {
        // Canonicalize so the rerun-if-changed path is stable across invocations.
        return repo_dist.canonicalize().unwrap_or(repo_dist);
    }

    write_stub(out_dir)
}

fn write_stub(out_dir: &Path) -> PathBuf {
    let stub = out_dir.join("web-stub");
    fs::create_dir_all(&stub).expect("create web stub dir");
    fs::write(stub.join("index.html"), PLACEHOLDER_INDEX_HTML).expect("write stub index.html");
    fs::write(stub.join(PLACEHOLDER_MARKER), b"1").expect("write stub marker");
    stub
}
