//! Adversarial corpus for `mold chain validate`. Each known-bad TOML
//! should fail with a descriptive error; the known-good corner cases
//! (newlines in prompts) should succeed.

mod common;

use common::TestEnv;
use predicates::prelude::*;

fn run_validate(env: &TestEnv, filename: &str) -> assert_cmd::assert::Assert {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("adversarial")
        .join(filename);
    assert!(path.exists(), "corpus file missing: {}", path.display());
    env.cmd().args(["chain", "validate"]).arg(&path).assert()
}

#[test]
fn rejects_unknown_schema() {
    run_validate(&TestEnv::new(), "v99_schema.toml")
        .failure()
        .stderr(predicate::str::contains("schema"));
}

#[test]
fn rejects_over_max_stages() {
    run_validate(&TestEnv::new(), "seventeen_stages.toml")
        .failure()
        .stderr(predicate::str::contains("stages").or(predicate::str::contains("16")));
}

#[test]
fn rejects_reserved_loras() {
    run_validate(&TestEnv::new(), "reserved_loras.toml")
        .failure()
        .stderr(
            predicate::str::contains("loras")
                .or(predicate::str::contains("sub-project B"))
                .or(predicate::str::contains("reserved")),
        );
}

#[test]
fn accepts_newline_prompt() {
    run_validate(&TestEnv::new(), "newline_prompt.toml").success();
}

#[test]
fn rejects_non_8kplus1_frames() {
    run_validate(&TestEnv::new(), "frames_not_8k_plus_1.toml")
        .failure()
        .stderr(predicate::str::contains("8k+1").or(predicate::str::contains("frames")));
}
