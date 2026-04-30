//! Resolves the catalog shard directory and stamps it into
//! `MOLD_CATALOG_DIR`. Mirrors `mold-server/build.rs`'s stub-fallback
//! pattern: if `data/catalog/flux.json` is missing (e.g. a pristine
//! checkout that hasn't yet vendored the shards), emit empty stubs into
//! `$OUT_DIR/catalog-stub/` so `cargo build` succeeds.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const FAMILIES: &[&str] = &[
    "flux",
    "flux2",
    "sd15",
    "sdxl",
    "z-image",
    "ltx-video",
    "ltx2",
    "qwen-image",
    "wuerstchen",
];

fn stub(family: &str) -> String {
    format!(
        r#"{{
  "$schema": "mold.catalog.v1",
  "family": "{family}",
  "generated_at": "1970-01-01T00:00:00Z",
  "scanner_version": "0.0",
  "entries": []
}}
"#
    )
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let real = crate_dir.join("data/catalog");
    println!("cargo:rerun-if-changed={}", real.display());

    let resolved = if FAMILIES
        .iter()
        .all(|f| real.join(format!("{f}.json")).is_file())
    {
        real
    } else {
        emit_stub(&out_dir)
    };

    println!("cargo:rustc-env=MOLD_CATALOG_DIR={}", resolved.display());
}

fn emit_stub(out_dir: &Path) -> PathBuf {
    let dir = out_dir.join("catalog-stub");
    fs::create_dir_all(&dir).expect("create stub dir");
    for family in FAMILIES {
        let path = dir.join(format!("{family}.json"));
        if !path.exists() {
            fs::write(&path, stub(family)).expect("write stub shard");
        }
    }
    dir
}
