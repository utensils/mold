use std::process::Command;

fn main() {
    // Try git SHA from environment first (Nix builds pass MOLD_GIT_SHA),
    // then fall back to running git.
    let sha = std::env::var("MOLD_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
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

    println!("cargo:rustc-env=MOLD_GIT_SHA={sha}");
    println!("cargo:rustc-env=MOLD_BUILD_DATE={date}");

    // Rerun when HEAD changes (new commit, branch switch, etc.)
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
    println!("cargo:rerun-if-changed=../../.git/packed-refs");
    println!("cargo:rerun-if-env-changed=MOLD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=MOLD_BUILD_DATE");
}
