//! Integration tests for the `mold runpod` CLI surface.
//!
//! Uses the `TestEnv` harness to isolate config + state. These tests only
//! cover clap parsing, help rendering, config round-trips, and error paths
//! that do not hit the network. Live RunPod interaction is covered by the
//! wiremock tests in `crates/mold-core/tests/runpod_client.rs`.

mod common;

use common::TestEnv;
use predicates::prelude::*;

#[test]
fn runpod_help_lists_all_subcommands() {
    let env = TestEnv::new();
    env.cmd()
        .args(["runpod", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("doctor"))
        .stdout(predicate::str::contains("gpus"))
        .stdout(predicate::str::contains("datacenters"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("get"))
        .stdout(predicate::str::contains("create"))
        .stdout(predicate::str::contains("stop"))
        .stdout(predicate::str::contains("start"))
        .stdout(predicate::str::contains("delete"))
        .stdout(predicate::str::contains("connect"))
        .stdout(predicate::str::contains("logs"))
        .stdout(predicate::str::contains("usage"))
        .stdout(predicate::str::contains("run"));
}

#[test]
fn runpod_create_help_documents_flags() {
    let env = TestEnv::new();
    env.cmd()
        .args(["runpod", "create", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--gpu"))
        .stdout(predicate::str::contains("--dc"))
        .stdout(predicate::str::contains("--cloud"))
        .stdout(predicate::str::contains("--volume"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--hf-token"))
        .stdout(predicate::str::contains("--network-volume"))
        .stdout(predicate::str::contains("--image-tag"));
}

#[test]
fn runpod_run_help_documents_flags() {
    let env = TestEnv::new();
    env.cmd()
        .args(["runpod", "run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--output-dir"))
        .stdout(predicate::str::contains("--keep"))
        .stdout(predicate::str::contains("--seed"))
        .stdout(predicate::str::contains("--wait-timeout"));
}

#[test]
fn runpod_doctor_fails_without_api_key() {
    let env = TestEnv::new();
    env.cmd()
        .env_remove("RUNPOD_API_KEY")
        .args(["runpod", "doctor"])
        .assert()
        .failure()
        .stdout(predicate::str::contains("api key not set"));
}

#[test]
fn runpod_config_keys_roundtrip_via_cli() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "set", "runpod.api_key", "test-key-value"])
        .assert()
        .success();
    env.cmd()
        .args(["config", "get", "runpod.api_key"])
        .assert()
        .success()
        // Redacted in display — real value never echoed.
        .stdout(predicate::str::contains("<set>"))
        .stdout(predicate::str::contains("test-key-value").not());
    env.cmd()
        .args(["config", "set", "runpod.auto_teardown", "true"])
        .assert()
        .success();
    env.cmd()
        .args(["config", "get", "runpod.auto_teardown", "--raw"])
        .assert()
        .success()
        .stdout(predicate::str::contains("true"));
}

#[test]
fn runpod_config_rejects_negative_cost_alert() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "set", "runpod.cost_alert_usd", "-5"])
        .assert()
        .failure();
}

#[test]
fn runpod_config_rejects_invalid_bool() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "set", "runpod.auto_teardown", "maybe"])
        .assert()
        .failure();
}

#[test]
fn runpod_config_clears_optional_fields_with_none() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "set", "runpod.default_gpu", "RTX 5090"])
        .assert()
        .success();
    env.cmd()
        .args(["config", "set", "runpod.default_gpu", "none"])
        .assert()
        .success();
    env.cmd()
        .args(["config", "get", "runpod.default_gpu"])
        .assert()
        .success()
        .stdout(predicate::str::contains("(not set)"));
}

#[test]
fn runpod_config_list_includes_runpod_section() {
    let env = TestEnv::new();
    env.cmd()
        .args(["config", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("runpod"));
}

#[test]
fn runpod_delete_runs_without_confirmation() {
    // Delete takes no confirmation — passing an explicit pod id is enough
    // signal. With a fake API key + unreachable host this will fail the
    // network call, but the CLI should NOT bail out at the confirm step.
    let env = TestEnv::new();
    env.cmd()
        .env("RUNPOD_API_KEY", "fake-test-key")
        .args(["runpod", "delete", "fake-pod-id"])
        .assert()
        .stdout(predicate::str::contains("cancelled").not());
}

#[test]
fn runpod_connect_prints_export_line() {
    let env = TestEnv::new();
    env.cmd()
        .env("RUNPOD_API_KEY", "fake-test-key")
        .args(["runpod", "connect", "abc123"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "export MOLD_HOST=https://abc123-7680.proxy.runpod.net",
        ));
}
