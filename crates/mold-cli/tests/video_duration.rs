//! CLI planning must not submit work, even when a generation server is reachable.
mod common;

use common::TestEnv;
use predicates::prelude::*;
use wiremock::matchers::{body_partial_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn parent_plan(server: &MockServer, model: &str, duration_ms: u64, overlap: u32) {
    let capabilities = mold_core::ServerCapabilities {
        video_jobs: Some(mold_core::video_job::VideoJobCapabilities {
            supported: true,
            literal_prompts: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    Mock::given(method("GET"))
        .and(path("/api/capabilities"))
        .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/status"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "version":"test", "models_loaded":[], "gpu_info":null,
            "uptime_secs":1, "instance_id":"captured-installation"
        })))
        .mount(server)
        .await;
    // Use each concrete recipe's real short window/canvas, not an impossible
    // generic H3 envelope for the LTX fixture.
    let recipe = mold_core::manifest::find_manifest(model).unwrap();
    let fps = recipe.defaults.fps.unwrap();
    let cap = recipe.defaults.frames.unwrap();
    let frames = duration_ms * u64::from(fps) / 1000;
    let clips = 1 + frames
        .saturating_sub(u64::from(cap))
        .div_ceil(u64::from(cap - overlap));
    Mock::given(method("POST"))
        .and(path("/api/video-jobs/validate"))
        .and(body_partial_json(
            serde_json::json!({"duration_ms":duration_ms.to_string()}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "validation_revision":"read-only-proof", "model":model,
            "width":recipe.defaults.width, "height":recipe.defaults.height,
            "fps":fps, "single_clip_max_frames":cap,
            "overlap_frames":overlap, "join_mode":"continuous", "clip_count":clips.to_string(),
            "part_count":clips.to_string(), "added_frames":frames.to_string(), "source_frames":"0",
            "total_frames":frames.to_string(), "added_duration_ms":duration_ms.to_string(),
            "total_duration_ms":duration_ms.to_string(), "warnings":[]
        })))
        .expect(1)
        .mount(server)
        .await;
}

async fn assert_read_only_plan(server: &MockServer) {
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests.iter().all(|request| request.method == "GET"
            || (request.method == "POST" && request.url.path() == "/api/video-jobs/validate")),
        "dry run must not create, expand, upload or mutate a job"
    );
    let validations: Vec<_> = requests
        .iter()
        .filter(|request| request.url.path() == "/api/video-jobs/validate")
        .collect();
    assert_eq!(validations.len(), 1);
    let request: serde_json::Value = serde_json::from_slice(&validations[0].body).unwrap();
    assert_eq!(
        request["generation"]["prompt"],
        "A tracking shot through a garden"
    );
    assert_ne!(request["generation"]["expand"], true);
    assert_eq!(
        validations[0]
            .headers
            .get("x-mold-destination-instance")
            .unwrap(),
        "captured-installation"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn h3_long_duration_dry_run_never_submits_or_expands() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
    parent_plan(&server, "minimax-h3-fl2va:fasth3-v2-int8", 45000, 1).await;
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
        .stderr(predicate::str::contains("9 clips joined into one video"));
    assert_read_only_plan(&server).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn scenes_file_sets_the_length_and_sends_each_scene_prompt() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
    parent_plan(&server, "minimax-h3-fl2va:fasth3-v2-int8", 26000, 1).await;
    let scenes = env.home.join("scenes.toml");
    std::fs::write(
        &scenes,
        "[[scene]]\nseconds = 12\nprompt = \"\"\"\nA tracking shot through a garden\n\"\"\"\n\n[[scene]]\nseconds = 14\nprompt = \"The gardener turns around\"\n",
    )
    .unwrap();
    env.cmd()
        .args(["run", "minimax-h3-fl2va:fasth3-v2-int8", "--scenes"])
        .arg(&scenes)
        .args(["--dry-run", "--host", &server.uri()])
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "Scene 2: 12.0-26.0 s  The gardener turns around",
        ));
    let requests = server.received_requests().await.unwrap();
    let validation = requests
        .iter()
        .find(|request| request.url.path() == "/api/video-jobs/validate")
        .unwrap();
    let request: serde_json::Value = serde_json::from_slice(&validation.body).unwrap();
    assert_eq!(request["duration_ms"], "26000");
    assert_eq!(request["scenes"][0]["duration_ms"], "12000");
    assert_eq!(request["scenes"][1]["prompt"], "The gardener turns around");
    assert_eq!(
        request["generation"]["prompt"],
        request["scenes"][0]["prompt"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn scenes_file_refuses_a_contradicting_duration_or_a_second_prompt() {
    let env = TestEnv::new();
    let scenes = env.home.join("scenes.toml");
    std::fs::write(&scenes, "[[scene]]\nseconds = 6\nprompt = 'one'\n").unwrap();
    env.cmd()
        .args([
            "run",
            "minimax-h3-fl2va:fasth3-v2-int8",
            "--duration",
            "9",
            "--scenes",
        ])
        .arg(&scenes)
        .args(["--dry-run", "--host", "http://127.0.0.1:9"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("the scenes add up to 6 s"));
    env.cmd()
        .args([
            "run",
            "minimax-h3-fl2va:fasth3-v2-int8",
            "a prompt",
            "--scenes",
        ])
        .arg(&scenes)
        .args(["--dry-run", "--host", "http://127.0.0.1:9"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("pass only the model"));
    env.cmd()
        .args([
            "run",
            "minimax-h3-fl2va:fasth3-v2-int8",
            "--prompt",
            "a",
            "--prompt",
            "b",
            "--scenes",
        ])
        .arg(&scenes)
        .args(["--dry-run", "--host", "http://127.0.0.1:9"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

#[tokio::test(flavor = "multi_thread")]
async fn ltx_duration_dry_run_never_submits_or_expands() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
    parent_plan(&server, "ltx-2-19b-distilled:fp8", 10000, 17).await;
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
        .stderr(predicate::str::contains("3 clips joined into one video"));
    assert_read_only_plan(&server).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn duration_ignores_a_long_local_frame_default_for_legacy_routing() {
    let env = TestEnv::new();
    let mut config = mold_core::Config::default();
    config.models.insert(
        "ltx-2-19b-distilled:fp8".into(),
        mold_core::ModelConfig {
            default_frames: Some(1001),
            ..Default::default()
        },
    );
    env.write_config(&toml::to_string(&config).unwrap());
    let server = MockServer::start().await;
    parent_plan(&server, "ltx-2-19b-distilled:fp8", 10000, 17).await;
    env.cmd()
        .args([
            "run",
            "ltx-2-19b-distilled:fp8",
            "A tracking shot through a garden",
            "--duration",
            "10",
            "--dry-run",
            "--host",
            &server.uri(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("3 clips joined into one video"));
    assert_read_only_plan(&server).await;
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

#[tokio::test]
async fn h3_long_duration_refuses_wan_sampling_controls() {
    let env = TestEnv::new();
    let server = MockServer::start().await;
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
                "--host",
                &server.uri(),
                flag,
                value,
            ])
            .assert()
            .failure()
            .stderr(predicate::str::contains("Wan-only controls"));
    }
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "invalid controls must fail before host discovery or media preparation"
    );
}
