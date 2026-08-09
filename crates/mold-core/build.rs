use std::process::Command;

/// Keep in sync with `build_info::SHORT_SHA_LENGTH`, which pins the display
/// width so chrome layouts cannot be shifted by the exact SHA's length.
const SHORT_SHA_LENGTH: usize = 7;

fn main() {
    // Try git SHA from environment first (Nix builds pass MOLD_GIT_SHA),
    // then fall back to running git.
    let sha = std::env::var("MOLD_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        });

    // Try build date from environment first (Nix builds pass MOLD_BUILD_DATE),
    // then fall back to git commit date.
    let date = std::env::var("MOLD_BUILD_DATE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            Command::new("git")
                .args(["log", "-1", "--format=%cs"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        });

    // Human-readable surfaces abbreviate. `MOLD_GIT_SHA` is exact because the
    // private H3 campaign identity needs an unambiguous commit, but a 40-char
    // SHA in a status line is 33 columns of chrome nobody reads, and the TUI
    // tab strip right-aligns this string onto the row holding the tab labels.
    let short_sha = if sha == "unknown" {
        sha.clone()
    } else {
        sha.chars().take(SHORT_SHA_LENGTH).collect::<String>()
    };

    println!("cargo:rustc-env=MOLD_GIT_SHA={sha}");
    println!("cargo:rustc-env=MOLD_GIT_SHA_SHORT={short_sha}");
    println!("cargo:rustc-env=MOLD_BUILD_DATE={date}");

    // Build a full version string as a compile-time constant for clap's
    // #[command(version = ...)] which requires &'static str.
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let full_version = if sha == "unknown" {
        version
    } else {
        format!("{version} ({short_sha} {date})")
    };
    println!("cargo:rustc-env=MOLD_FULL_VERSION={full_version}");

    // Rerun when HEAD changes (new commit, branch switch, etc.)
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
    println!("cargo:rerun-if-changed=../../.git/packed-refs");
    println!("cargo:rerun-if-env-changed=MOLD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=MOLD_BUILD_DATE");
}
