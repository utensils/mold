//! Pinned crate-boundary check — `mold-discord` MUST NOT transitively
//! depend on `mold-catalog`. Catches accidental dep churn.
//!
//! Implemented as an `#[ignore]`d test so the regular `cargo test --workspace`
//! is unaffected; the workspace CI gate runs it explicitly.

use std::process::Command;

#[test]
#[ignore]
fn discord_does_not_inherit_mold_catalog() {
    let out = Command::new("cargo")
        .args(["tree", "-p", "mold-ai-discord", "-e", "normal"])
        .output()
        .expect("cargo tree");
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        !body.contains("mold-ai-catalog"),
        "mold-ai-discord must not transitively depend on mold-ai-catalog\n\n{body}"
    );
}
