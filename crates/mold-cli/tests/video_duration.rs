//! CLI planning must not submit work, even when a generation server is reachable.
mod common;

use common::TestEnv;
use predicates::prelude::*;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test(flavor = "multi_thread")]
async fn h3_long_duration_dry_run_never_submits_or_expands() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
    env.cmd()
        .args([
            "run",
            "minimax-h3-fl2va:fasth3-v2-int8",
            "A tracking shot through a garden",
            "--duration",
            "45",
            "--dry-run",
            "--expand",
            "--host",
            &server.uri(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("4 stages"))
        .stdout(predicate::str::contains("1088 (45.33s @ 24fps)"));
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests.iter().all(|request| request.method == "GET"),
        "dry run wrote to the server"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn ltx_duration_dry_run_never_submits_or_expands() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
    env.cmd()
        .args([
            "run",
            "ltx-2-19b-distilled:fp8",
            "A tracking shot through a garden",
            "--duration",
            "10",
            "--dry-run",
            "--expand",
            "--host",
            &server.uri(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("stages"));
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests.iter().all(|request| request.method == "GET"),
        "dry run wrote to the server"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn chain_limits_preserves_the_server_contract() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/capabilities/chain-limits"))
        .and(query_param("model", "minimax-h3-fl2va:fasth3-v2-int8"))
        .and(query_param("fps", "24"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "max_stages": 16, "future": { "field": 7 }
        })))
        .expect(1)
        .mount(&server)
        .await;
    env.cmd()
        .args([
            "chain",
            "limits",
            "minimax-h3-fl2va:fasth3-v2-int8",
            "--fps",
            "24",
            "--json",
            "--host",
            &server.uri(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"max_stages\": 16"))
        .stdout(predicate::str::contains("\"field\": 7"));
}

#[test]
fn h3_long_duration_refuses_wan_sampling_controls() {
    let env = TestEnv::new();
    for (flag, value) in [
        ("--sample-shift", "3"),
        ("--distill-strength", "1.5"),
        ("--sample-solver", "unipc"),
    ] {
        env.cmd()
            .args([
                "run",
                "minimax-h3-fl2va:fasth3-v2-int8",
                "A tracking shot",
                "--duration",
                "45",
                "--dry-run",
                flag,
                value,
            ])
            .assert()
            .failure();
    }
}
