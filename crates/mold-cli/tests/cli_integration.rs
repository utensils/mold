//! Blackbox CLI integration tests for the `mold` binary.
//!
//! Each test uses a [`TestEnv`] that creates an isolated temp directory with
//! its own `MOLD_HOME` and `MOLD_MODELS_DIR`, preventing tests from reading
//! the host machine's real config or model files.
//!
//! These tests run in CI without GPU access — they only exercise commands
//! that work with the filesystem, config, and manifest data.

mod common;

use common::TestEnv;
use predicates::prelude::*;

// ── mold version ──────────────────────────────────────────────────────────

#[test]
fn version_subcommand_prints_version() {
    let env = TestEnv::new();
    env.cmd()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("mold "));
}

#[test]
fn version_flag_prints_version() {
    let env = TestEnv::new();
    env.cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("mold "));
}

#[test]
fn version_flag_matches_subcommand() {
    let env = TestEnv::new();

    let flag_output = env.cmd().arg("--version").output().unwrap();
    let sub_output = env.cmd().arg("version").output().unwrap();

    let flag_str = String::from_utf8_lossy(&flag_output.stdout);
    let sub_str = String::from_utf8_lossy(&sub_output.stdout);

    // Both should contain the same version number (strip "mold " prefix)
    let flag_ver = flag_str.trim().trim_start_matches("mold ");
    let sub_ver = sub_str.trim().trim_start_matches("mold ");
    assert_eq!(
        flag_ver, sub_ver,
        "--version and version subcommand should match"
    );
}

#[test]
fn unknown_subcommand_fails() {
    let env = TestEnv::new();
    env.cmd().arg("nonexistent-subcommand").assert().failure();
}

// ── mold default ──────────────────────────────────────────────────────────

#[test]
fn default_shows_fallback_model() {
    let env = TestEnv::new();
    env.cmd()
        .arg("default")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein"));
}

#[test]
fn default_set_persists_to_config() {
    let env = TestEnv::new();
    env.cmd()
        .args(["default", "flux-dev:q4"])
        .assert()
        .success();

    // Verify it was persisted
    let config_path = env.home.join("config.toml");
    let content = std::fs::read_to_string(&config_path).unwrap();
    assert!(
        content.contains("flux-dev:q4"),
        "config should contain the new default: {content}"
    );
}

#[test]
fn default_rejects_unknown_model() {
    let env = TestEnv::new();
    env.cmd()
        .args(["default", "totally-fake-model:q99"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown model"));
}

#[test]
fn default_env_var_override() {
    let env = TestEnv::new();
    env.cmd()
        .env("MOLD_DEFAULT_MODEL", "flux-dev:q8")
        .arg("default")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux-dev:q8"));
}

// ── mold config ───────────────────────────────────────────────────────────

#[test]
fn config_list_outputs_settings() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("default_model"))
        .stdout(predicate::str::contains("server_port"));
}

#[test]
fn config_list_json_is_valid() {
    let env = TestEnv::new();
    let output = env
        .cmd()
        .args(["config", "list", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON: {e}\noutput: {stdout}"));
    assert!(parsed.is_object(), "should be a JSON object");
}

#[test]
fn config_get_server_port() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "get", "server_port"])
        .assert()
        .success()
        .stdout(predicate::str::contains("7680"));
}

#[test]
fn config_get_raw_outputs_bare_value() {
    let env = TestEnv::new();
    let output = env
        .cmd()
        .args(["config", "get", "server_port", "--raw"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "7680");
}

#[test]
fn config_set_persists_value() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "set", "server_port", "8080"])
        .assert()
        .success();

    // Verify the value was saved
    env.cmd()
        .args(["config", "get", "server_port", "--raw"])
        .assert()
        .success()
        .stdout(predicate::str::is_match("8080").unwrap());
}

#[test]
fn config_path_outputs_valid_path() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("config.toml"));
}

// ── mold stats ────────────────────────────────────────────────────────────

#[test]
fn stats_empty_models_dir() {
    let env = TestEnv::new();
    env.cmd().arg("stats").assert().success().stdout(
        predicate::str::contains("0 models").or(predicate::str::contains("Models directory")),
    );
}

#[test]
fn stats_json_is_valid() {
    let env = TestEnv::new();
    let output = env.cmd().args(["stats", "--json"]).output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON: {e}\noutput: {stdout}"));
    assert!(parsed.is_object(), "should be a JSON object");
}

#[test]
fn stats_with_populated_model() {
    let env = TestEnv::new();
    env.populate_manifest_model("flux2-klein:q4");

    env.cmd()
        .arg("stats")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein:q4"))
        .stdout(predicate::str::contains("1 model"));
}

// ── mold list ─────────────────────────────────────────────────────────────

#[test]
fn list_shows_available_to_pull() {
    let env = TestEnv::new();
    env.cmd()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("Available to pull"));
}

#[test]
fn list_shows_column_headers_when_models_installed() {
    let env = TestEnv::new();
    env.populate_manifest_model("flux2-klein:q4");

    env.cmd()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("NAME"))
        .stdout(predicate::str::contains("FAMILY"));
}

#[test]
fn list_with_populated_model_shows_installed() {
    let env = TestEnv::new();
    env.populate_manifest_model("flux2-klein:q4");

    env.cmd()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein:q4"));
}

#[test]
fn list_no_models_shows_message() {
    let env = TestEnv::new();
    env.cmd()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("No models configured"));
}

#[test]
fn list_upscaler_models_shown_as_installed() {
    // Regression test for #184 — upscaler models were shown as "cached"
    // in "Available to pull" instead of in the installed section.
    let env = TestEnv::new();
    env.populate_manifest_model("real-esrgan-x4plus:fp16");

    let output = env.cmd().arg("list").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // The model should appear BEFORE the "Available to pull" section
    let available_pos = stdout.find("Available to pull");
    let model_pos = stdout.find("real-esrgan-x4plus:fp16");

    assert!(model_pos.is_some(), "upscaler should appear in output");
    if let (Some(mp), Some(ap)) = (model_pos, available_pos) {
        assert!(
            mp < ap,
            "upscaler model should appear in installed section (before 'Available to pull')"
        );
    }
}

// ── mold info ─────────────────────────────────────────────────────────────

#[test]
fn info_overview_shows_paths() {
    let env = TestEnv::new();
    env.cmd()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("Models"))
        .stdout(predicate::str::contains("mold"));
}

#[test]
fn info_unknown_model_errors() {
    let env = TestEnv::new();
    env.cmd()
        .args(["info", "totally-fake-model:q99"])
        .assert()
        .failure();
}

#[test]
fn info_known_model_shows_details() {
    let env = TestEnv::new();
    env.populate_manifest_model("flux2-klein:q4");

    env.cmd()
        .args(["info", "flux2-klein:q4"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein:q4"));
}

// ── mold rm ───────────────────────────────────────────────────────────────

#[test]
fn rm_unknown_model_errors() {
    let env = TestEnv::new();
    env.cmd()
        .args(["rm", "--force", "totally-fake-model:q99"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not installed"));
}

#[test]
fn rm_removes_manifest_model() {
    // Regression for #190 — mold rm couldn't remove manifest-backed models
    let env = TestEnv::new();
    env.populate_manifest_model("flux2-klein:q4");

    // Verify it's listed as installed first
    env.cmd()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein:q4"));

    // Remove it
    env.cmd()
        .args(["rm", "--force", "flux2-klein:q4"])
        .assert()
        .success();
}

#[test]
fn rm_preserves_shared_files_when_sibling_exists() {
    let env = TestEnv::new();
    // Populate two FLUX models that share VAE/T5/CLIP
    env.populate_manifest_model("flux2-klein:q4");
    env.populate_manifest_model("flux2-klein:q6");

    // Remove one
    env.cmd()
        .args(["rm", "--force", "flux2-klein:q4"])
        .assert()
        .success();

    // The sibling should still be listed
    env.cmd()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein:q6"));
}

// ── mold clean ────────────────────────────────────────────────────────────

#[test]
fn clean_dry_run_default() {
    let env = TestEnv::new();
    env.cmd()
        .arg("clean")
        .assert()
        .success()
        .stdout(predicate::str::contains("Nothing to clean").or(predicate::str::contains("clean")));
}

#[test]
fn clean_detects_stale_pulling_marker() {
    let env = TestEnv::new();
    // Create a stale .pulling marker
    let marker = env.models.join(".pulling-fake-model");
    std::fs::write(&marker, "stale").unwrap();
    // Set modification time to the past
    let old_time = filetime::FileTime::from_unix_time(0, 0);
    filetime::set_file_mtime(&marker, old_time).unwrap();

    env.cmd().arg("clean").assert().success();
}

// ── mold completions ──────────────────────────────────────────────────────

#[test]
fn completions_bash_outputs_script() {
    let env = TestEnv::new();
    env.cmd()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("complete").or(predicate::str::contains("COMPREPLY")));
}

#[test]
fn completions_zsh_outputs_script() {
    let env = TestEnv::new();
    env.cmd()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

// ── mold run (error paths, no GPU needed) ─────────────────────────────────

#[test]
fn run_missing_image_file_errors() {
    let env = TestEnv::new();
    env.cmd()
        .args(["run", "a cat", "--image", "/nonexistent/photo.png"])
        .assert()
        .failure();
}

#[test]
fn run_mask_requires_image_flag() {
    let env = TestEnv::new();
    // Create a real mask file so the error is about --mask requiring --image,
    // not about the file not existing.
    let mask = env.home.join("mask.png");
    std::fs::write(&mask, b"stub").unwrap();
    env.cmd()
        .args(["run", "a cat", "--mask"])
        .arg(&mask)
        .assert()
        .failure();
}

// ── mold pull (error paths) ───────────────────────────────────────────────

#[test]
fn pull_unknown_model_errors() {
    let env = TestEnv::new();
    env.cmd()
        .args(["pull", "totally-fake-model:q99"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown").or(predicate::str::contains("Unknown")));
}

// ── mold update ──────────────────────────────────────────────────────────

#[test]
fn update_help_text() {
    let env = TestEnv::new();
    env.cmd()
        .args(["update", "--help"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("--check")
                .and(predicate::str::contains("--force"))
                .and(predicate::str::contains("--version")),
        );
}

#[test]
fn update_appears_in_main_help() {
    let env = TestEnv::new();
    env.cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("update"));
}

#[test]
fn update_check_runs_without_panic() {
    // Verifies `mold update --check` runs to completion without panicking.
    // Outcome depends on network: success with "up to date" / "available",
    // or failure with a connection error. Either is acceptable — panics are not.
    let env = TestEnv::new();
    let output = env
        .cmd()
        .args(["update", "--check"])
        .timeout(std::time::Duration::from_secs(15))
        .output()
        .expect("failed to run mold update --check");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should contain meaningful output, not a panic backtrace
    assert!(
        !stderr.contains("panicked at"),
        "mold update --check panicked: {stderr}"
    );
    // Should print current version regardless of outcome
    assert!(
        stderr.contains("Current version"),
        "expected 'Current version' in stderr: {stderr}"
    );
}

// ── mold run --script --dry-run ───────────────────────────────────────────

#[test]
fn dry_run_prints_stage_summary() {
    let script = r#"
schema = "mold.chain.v1"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
steps = 8
guidance = 3.0
strength = 1.0
motion_tail_frames = 25
output_format = "mp4"

[[stage]]
prompt = "first scene"
frames = 97

[[stage]]
prompt = "second scene"
frames = 49
"#;
    let env = TestEnv::new();
    let path = env.home.join("chain.toml");
    std::fs::write(&path, script).unwrap();

    env.cmd()
        .args(["run", "--script", path.to_str().unwrap(), "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 stages"))
        .stdout(predicate::str::contains("first scene"))
        .stdout(predicate::str::contains("second scene"));
}

// ── mold run --prompt sugar ────────────────────────────────────────────────

#[test]
fn repeated_prompt_flag_yields_chain() {
    let env = TestEnv::new();
    env.cmd()
        .args([
            "run",
            "ltx-2-19b-distilled:fp8",
            "--prompt",
            "first scene",
            "--prompt",
            "second scene",
            "--prompt",
            "third scene",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("3 stages"))
        .stdout(predicate::str::contains("first scene"))
        .stdout(predicate::str::contains("second scene"))
        .stdout(predicate::str::contains("third scene"));
}

// ── mold chain validate ──────────────────────────────────────────────────

#[test]
fn chain_validate_reports_ok_for_valid_script() {
    let script = r#"
schema = "mold.chain.v1"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
steps = 8
guidance = 3.0
strength = 1.0
motion_tail_frames = 25
output_format = "mp4"

[[stage]]
prompt = "only stage"
frames = 97
"#;
    let env = TestEnv::new();
    let path = env.home.join("chain.toml");
    std::fs::write(&path, script).unwrap();

    env.cmd()
        .args(["chain", "validate", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"))
        .stdout(predicate::str::contains("1 stages"));
}

#[test]
fn chain_validate_errors_on_bad_schema() {
    let script = r#"
schema = "mold.chain.v99"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
steps = 8
guidance = 3.0
strength = 1.0
motion_tail_frames = 4
output_format = "mp4"

[[stage]]
prompt = "stage"
frames = 97
"#;
    let env = TestEnv::new();
    let path = env.home.join("bad_schema.toml");
    std::fs::write(&path, script).unwrap();

    env.cmd()
        .args(["chain", "validate", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("schema"));
}
