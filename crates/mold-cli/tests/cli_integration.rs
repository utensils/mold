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

#[tokio::test(flavor = "multi_thread")]
async fn loopback_admin_commands_do_not_hide_live_server_http_errors() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    for status in [401, 500] {
        let env = TestEnv::new();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/status"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;

        for command in ["ps", "info"] {
            env.cmd()
                .env("MOLD_HOST", server.uri())
                .arg(command)
                .assert()
                .failure()
                .stderr(predicate::str::contains("server status"));
        }
    }
}

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

fn library_capabilities(organize: bool, bulk_mutations: bool, trash: bool) -> serde_json::Value {
    serde_json::json!({
        "gallery": {
            "can_delete": true,
            "organize": organize,
            "bulk_mutations": bulk_mutations,
            "trash": trash.then_some(serde_json::json!({
                "enabled": true,
                "retention_days": 30
            }))
        },
        "catalog": { "available": false, "families": [] }
    })
}

fn library_row(filename: &str, timestamp: u64, tags: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "filename": filename,
        "metadata": {
            "prompt": "night owl",
            "model": "flux-dev:q4",
            "seed": 7,
            "steps": 20,
            "guidance": 3.5,
            "width": 1024,
            "height": 1024,
            "output_format": "png",
            "version": "test"
        },
        "timestamp": timestamp,
        "format": "png",
        "tags": tags,
        "favorite": true,
        "collections": []
    })
}

#[tokio::test]
async fn library_list_json_is_pure_and_uses_the_same_filtered_page() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/gallery"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            library_row("older.png", 1, &["owl"]),
            library_row("newer.png", 2, &["owl", "night"])
        ])))
        .mount(&server)
        .await;

    let output = env
        .cmd()
        .env("MOLD_HOST", server.uri())
        .args([
            "library", "list", "--tag", "night", "--limit", "1", "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.stdout.contains(&0x1b),
        "JSON stdout contains ANSI bytes"
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["total"], 1);
    assert_eq!(json["items"][0]["filename"], "newer.png");
}

/// A listing teaches the shell what it saw.
///
/// Tags, collections, gallery filenames and the machines you talk to all live
/// on a server, and a completer cannot ask one — so the commands that already
/// fetch them write `$MOLD_HOME/completion-cache.json`, and the completers
/// read that. This drives the real binary, so dropping the refresh call in
/// `commands::library` fails it.
#[tokio::test]
async fn library_list_teaches_the_shell_what_it_just_listed() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/gallery"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            library_row("older.png", 1, &["owl"]),
            library_row("newer.png", 2, &["owl", "night"])
        ])))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/gallery/collections"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!([{
                "id": "c1",
                "name": "Winter Scenes",
                "slug": "winter-scenes",
                "count": 2,
                "created_at": 1,
                "updated_at": 2
            }])),
        )
        .mount(&server)
        .await;

    let cache_path = env.home.join("completion-cache.json");
    assert!(
        !cache_path.exists(),
        "a fresh Mold home carries no completion cache"
    );

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["library", "list"])
        .assert()
        .success();

    let cache = read_completion_cache(&env);
    let strings = |key: &str| cache_strings(&cache, key);
    let filenames = strings("filenames");
    assert!(
        filenames.contains(&"newer.png".to_string()),
        "{filenames:?}"
    );
    assert!(
        filenames.contains(&"older.png".to_string()),
        "{filenames:?}"
    );
    let tags = strings("tags");
    assert!(tags.contains(&"night".to_string()), "{tags:?}");
    assert!(tags.contains(&"owl".to_string()), "{tags:?}");
    let collections = strings("collections");
    assert!(
        collections.contains(&"Winter Scenes".to_string()),
        "{collections:?}"
    );
    assert!(
        collections.contains(&"winter-scenes".to_string()),
        "a NAME-OR-SLUG positional takes either: {collections:?}"
    );
    assert_eq!(
        strings("hosts"),
        vec![server.uri().trim_end_matches('/').to_string()],
        "the machine that answered is the one worth completing"
    );
}

/// The completion cache one command just wrote, as JSON.
fn read_completion_cache(env: &TestEnv) -> serde_json::Value {
    let path = env.home.join("completion-cache.json");
    serde_json::from_slice(&std::fs::read(&path).expect("the command writes the cache"))
        .expect("the cache is JSON")
}

/// One list out of the cache document.
fn cache_strings(cache: &serde_json::Value, key: &str) -> Vec<String> {
    cache[key]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|value| value.as_str().unwrap_or_default().to_string())
        .collect()
}

/// A command that never reaches a server leaves no cache behind, so Tab on a
/// machine that has only ever failed to connect completes nothing.
#[tokio::test]
async fn a_refused_listing_records_no_machine() {
    let env = TestEnv::new();
    env.cmd()
        .env("MOLD_HOST", "http://127.0.0.1:1")
        .args(["library", "list"])
        .assert()
        .failure();
    assert!(
        !env.home.join("completion-cache.json").exists(),
        "an unreachable machine is not a completion candidate"
    );
}

#[tokio::test]
async fn library_tag_add_uses_replay_safe_bulk_mutation_when_advertised() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/capabilities"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(library_capabilities(true, true, true)),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/gallery/mutations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "operation_id": "server-receipt",
            "changed": 2,
            "revision": 1
        })))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args([
            "library",
            "tag",
            "add",
            "a.png",
            "b.png",
            "--tag",
            "night owls",
        ])
        .assert()
        .success();

    let requests = server.received_requests().await.unwrap();
    let request = requests
        .iter()
        .find(|request| request.url.path() == "/api/gallery/mutations")
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["filenames"], serde_json::json!(["a.png", "b.png"]));
    assert_eq!(body["add_tags"], serde_json::json!(["night owls"]));
    assert!(uuid::Uuid::parse_str(body["operation_id"].as_str().unwrap()).is_ok());
}

#[tokio::test]
async fn library_tag_add_falls_back_to_legacy_organize_route() {
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/capabilities"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(library_capabilities(true, false, true)),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/gallery/organize"))
        .and(body_json(serde_json::json!({
            "filenames": ["a.png"],
            "add_tags": ["owl"]
        })))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["library", "tag", "add", "a.png", "--tag", "owl"])
        .assert()
        .success();
}

#[tokio::test]
async fn library_trash_refuses_a_host_without_recoverable_trash() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/capabilities"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(library_capabilities(true, false, false)),
        )
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["library", "trash", "keep-me.png"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no files were deleted"));

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1, "refusal must not send a DELETE/POST");
}

#[tokio::test]
async fn library_collection_remove_resolves_slug_and_sends_membership_change() {
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/capabilities"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(library_capabilities(true, true, true)),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/gallery/collections"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!([{
                "id": "collection-id",
                "name": "Night Owls",
                "slug": "night-owls",
                "hidden": false,
                "count": 2,
                "created_at": 1,
                "updated_at": 2
            }])),
        )
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path("/api/gallery/collections/collection-id/items"))
        .and(body_json(serde_json::json!({
            "add": [],
            "remove": ["odd # 100%.png"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "collection-id",
            "name": "Night Owls",
            "slug": "night-owls",
            "hidden": false,
            "count": 1,
            "created_at": 1,
            "updated_at": 3
        })))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args([
            "library",
            "collection",
            "remove",
            "night-owls",
            "odd # 100%.png",
        ])
        .assert()
        .success();
}

#[tokio::test]
async fn library_show_video_uses_thumbnail_when_animated_preview_is_missing() {
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    let mut row = library_row("clip.mp4", 2, &["motion"]);
    row["format"] = serde_json::json!("mp4");
    row["metadata"]["output_format"] = serde_json::json!("mp4");
    Mock::given(method("GET"))
        .and(path("/api/gallery"))
        .and(query_param("filename", "clip.mp4"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([row])))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/gallery/preview/clip.mp4"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/gallery/thumbnail/clip.mp4"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"not-a-real-png"))
        .expect(1)
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["library", "show", "clip.mp4", "--preview"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Filename: clip.mp4"));
}

#[tokio::test]
async fn library_global_tag_delete_requires_yes_without_a_tty() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/capabilities"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(library_capabilities(true, true, true)),
        )
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["library", "tag", "delete", "owl"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("pass --yes"));

    assert_eq!(server.received_requests().await.unwrap().len(), 1);
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

/// Regression: hidden companion manifests (e.g. `flux2-te`) share text
/// encoder + tokenizer files with their parent FLUX.2 models. Populating a
/// parent like `flux2-klein:q4` makes those shared paths exist on disk, so
/// `manifest_model_is_downloaded` returns true for `flux2-te` too. Stats
/// must filter to `visible_manifests()` so the companion isn't double-
/// counted as a separately-installed model.
#[test]
fn stats_does_not_double_count_hidden_companions() {
    let env = TestEnv::new();
    env.populate_manifest_model("flux2-klein:q4");

    env.cmd()
        .arg("stats")
        .assert()
        .success()
        .stdout(predicate::str::contains("flux2-klein:q4"))
        .stdout(predicate::str::contains("(1 models)").or(predicate::str::contains("(1 model)")))
        .stdout(predicate::str::contains("flux2-te").not());
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

// ── standalone server administration ────────────────────────────────────

#[test]
fn gpu_list_json_falls_back_to_local_inventory_when_loopback_server_is_stopped() {
    let env = TestEnv::new();
    let output = env
        .cmd()
        .env("MOLD_HOST", "http://127.0.0.1:9")
        .args(["gpu", "list", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(body["devices"].is_array());
    assert_eq!(body["plan_version"], 0);
    assert!(String::from_utf8_lossy(&output.stderr).contains("local runtime inventory"));
}

#[test]
fn unload_is_idempotent_when_loopback_server_is_stopped() {
    let env = TestEnv::new();
    env.cmd()
        .env("MOLD_HOST", "http://127.0.0.1:9")
        .arg("unload")
        .assert()
        .success()
        .stdout(predicate::str::contains("no server-loaded model to unload"));
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

#[test]
fn run_qwen_image_edit_rejects_batch_before_remote_generation() {
    let env = TestEnv::new();
    let image = env.home.join("edit.png");
    let img = image::RgbImage::from_fn(16, 16, |_, _| image::Rgb([255, 0, 0]));
    img.save(&image).unwrap();

    env.cmd()
        .args([
            "run",
            "qwen-image-edit-2511:q4",
            "replace the background",
            "--image",
        ])
        .arg(&image)
        .args(["--batch", "2", "--output", "out.png"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "qwen-image-edit only supports --batch 1",
        ));
}

/// `mold run <wan I2V checkpoint> --extend clip.mp4` must reach dispatch.
///
/// An extend carries its source frames in the clip it continues, and
/// `validate_extend` forbids pairing `--extend` with an image or keyframes —
/// so a preflight that counted only those saw every continuation as
/// source-less and refused it with "this Wan I2V checkpoint needs a source
/// image", the exact contract that makes the checkpoint extend-capable
/// (#783). This drives the real binary, so restoring the inline `has_source`
/// expression at the call site fails it.
#[test]
fn run_extend_satisfies_a_wan_i2v_checkpoints_source_contract() {
    let env = TestEnv::new();
    let clip = env.home.join("clip.mp4");
    std::fs::write(&clip, b"\0\0\0\x20ftypisom").unwrap();

    // The run still fails — nothing is downloaded and the fake host is
    // unreachable — but it must not fail *on the contract*.
    env.cmd()
        .args([
            "run",
            "wan22-i2v-a14b:q8",
            "a cat keeps walking",
            "--extend",
        ])
        .arg(&clip)
        .args(["--local", "--frames", "49", "--output", "out.mp4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("needs a source image").not());

    // The same checkpoint with no source at all is still refused, so the
    // assertion above is not vacuous.
    env.cmd()
        .args([
            "run",
            "wan22-i2v-a14b:q8",
            "a cat keeps walking",
            "--local",
            "--frames",
            "49",
            "--output",
            "out.mp4",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("needs a source image"));
}

/// A continuation aimed at a family with no continuation path is refused for
/// *that*, not for source frames it never supplied (#783).
#[test]
fn run_extend_on_a_family_without_a_continuation_path_names_extend() {
    let env = TestEnv::new();
    let clip = env.home.join("clip.mp4");
    std::fs::write(&clip, b"\0\0\0\x20ftypisom").unwrap();

    env.cmd()
        .args([
            "run",
            "ltx-video-0.9.8-13b-distilled:bf16",
            "a cat keeps walking",
            "--extend",
        ])
        .arg(&clip)
        .args(["--local", "--output", "out.mp4"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--extend is only supported for LTX-2 / LTX-2.3 and Wan models",
        ))
        .stderr(predicate::str::contains("source image").not());
}

/// The overlap the CLI *sends* is the family's, not LTX-2's (#783).
///
/// `mold run` materializes `extend_overlap_frames` into the request it builds,
/// so a wan continuation that named no overlap carries 1 — the frame the
/// continuation is seeded with — rather than inheriting a family-blind 17 that
/// clears wan's `4k+1` grid check and then fails inside the engine, after the
/// expert load has been paid for. The recorded value is also what saved
/// provenance reports, and an installed `cv:` / `hf:` wan checkpoint has no
/// manifest for metadata to resolve a family from later.
///
/// This drives the real binary against a mock host, so deleting the
/// materialization call in `commands::generate::run` fails it.
#[tokio::test]
async fn run_extend_sends_the_familys_own_carryover_overlap() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let clip = env.home.join("clip.mp4");
    std::fs::write(&clip, b"\0\0\0\x20ftypisom").unwrap();

    let server = MockServer::start().await;
    // Refuse the render outright: this test is about the request the CLI
    // composes, and a 422 is a hard error, so nothing falls back to local
    // inference or tries to pull a checkpoint.
    Mock::given(method("POST"))
        .and(path("/api/generate/stream"))
        .respond_with(
            ResponseTemplate::new(422).set_body_json(serde_json::json!({"error": "mock refusal"})),
        )
        .mount(&server)
        .await;

    let overlap_of = |model: &str| {
        env.cmd()
            .args(["run", model, "a cat keeps walking", "--extend"])
            .arg(&clip)
            .args([
                "--host",
                &server.uri(),
                "--frames",
                "49",
                "--output",
                "out.mp4",
            ])
            .assert()
            .failure();
    };

    overlap_of("wan22-i2v-a14b:q8");
    overlap_of("ltx-2-19b-dev:fp8");

    let sent: Vec<serde_json::Value> = server
        .received_requests()
        .await
        .expect("the mock server records requests")
        .iter()
        .filter(|request| request.url.path() == "/api/generate/stream")
        .map(|request| serde_json::from_slice(&request.body).expect("the CLI posts JSON"))
        .collect();
    assert_eq!(sent.len(), 2, "one generate request per run");
    assert_eq!(
        sent[0]["extend_overlap_frames"],
        serde_json::json!(mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES),
        "wan's continuation carries its own one-frame carryover"
    );
    assert_eq!(
        sent[1]["extend_overlap_frames"],
        serde_json::json!(mold_core::validation::DEFAULT_EXTEND_OVERLAP_FRAMES),
        "the same seam resolves LTX-2's 17 from the resolved family"
    );
}

/// `mold run --fit` sends the CANVAS the user asked for and the picture
/// resampled onto it, plus the provenance the apps record.
///
/// Without `--fit` a source image decides the canvas, so this is the one flag
/// that reverses that rule. Driving the real binary against a mock host pins
/// all three halves at once: the request's `width`/`height`, the decoded
/// `source_image` dimensions, and the `source_fit` object
/// `parseSourceFitPolicy` reads back.
#[tokio::test]
async fn run_fit_resamples_the_source_onto_the_requested_canvas() {
    use base64::Engine as _;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let source = env.home.join("wide.png");
    let mut wide = image::RgbaImage::new(64, 16);
    for (x, _y, pixel) in wide.enumerate_pixels_mut() {
        *pixel = image::Rgba([if x < 32 { 255 } else { 0 }, 0, 0, 255]);
    }
    wide.save(&source).unwrap();

    let server = MockServer::start().await;
    // Refuse the render: the request body is the whole subject, and a 422 is
    // a hard error so nothing falls back to local inference.
    Mock::given(method("POST"))
        .and(path("/api/generate/stream"))
        .respond_with(
            ResponseTemplate::new(422).set_body_json(serde_json::json!({"error": "mock refusal"})),
        )
        .mount(&server)
        .await;

    env.cmd()
        .args(["run", "flux-dev:q4", "a cat", "--image"])
        .arg(&source)
        .args([
            "--fit",
            "crop-fill",
            "--width",
            "128",
            "--height",
            "128",
            "--host",
            &server.uri(),
            "--output",
            "out.png",
        ])
        .assert()
        .failure();

    let sent: Vec<serde_json::Value> = server
        .received_requests()
        .await
        .expect("the mock server records requests")
        .iter()
        .filter(|request| request.url.path() == "/api/generate/stream")
        .map(|request| serde_json::from_slice(&request.body).expect("the CLI posts JSON"))
        .collect();
    assert_eq!(sent.len(), 1, "one generate request");
    assert_eq!(sent[0]["width"], serde_json::json!(128));
    assert_eq!(sent[0]["height"], serde_json::json!(128));
    assert_eq!(
        sent[0]["source_fit"],
        serde_json::json!({ "mode": "crop-fill" }),
        "the policy rides the request as the apps record it"
    );
    let encoded = sent[0]["source_image"]
        .as_str()
        .expect("the source image rides as base64");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("base64 body");
    let fitted = image::load_from_memory(&bytes).expect("a decodable image");
    assert_eq!(
        (fitted.width(), fitted.height()),
        (128, 128),
        "the fitted source is exactly the requested canvas"
    );
}

/// The two browser-only policies are refused by name, with the reason and the
/// nearest thing the terminal can do.
#[test]
fn run_fit_refuses_the_policies_that_need_a_mask_or_an_upscaler() {
    let env = TestEnv::new();
    let source = env.home.join("tiny.png");
    image::RgbaImage::new(4, 4).save(&source).unwrap();

    env.cmd()
        .args(["run", "flux-dev:q4", "a cat", "--image"])
        .arg(&source)
        .args(["--fit", "pad-repaint"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("pad-repaint").and(predicate::str::contains("pad-fit")));

    env.cmd()
        .args(["run", "flux-dev:q4", "a cat", "--image"])
        .arg(&source)
        .args(["--fit", "upscale-then-fit"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mold upscale"));
}

/// `--fit` without a picture is a clap-level refusal, so no model is resolved
/// and no request is composed.
#[test]
fn run_fit_requires_an_image() {
    let env = TestEnv::new();
    env.cmd()
        .args(["run", "flux-dev:q4", "a cat", "--fit", "crop-fill"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--image"));
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
                .and(predicate::str::contains("--nightly"))
                .and(predicate::str::contains("--version")),
        );
}

#[test]
fn update_nightly_conflicts_with_exact_version() {
    let env = TestEnv::new();
    env.cmd()
        .args(["update", "--nightly", "--version", "v0.23.3"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
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

// ── mold run --prompt flag conflicts ──────────────────────────────────────

#[test]
fn positional_plus_prompt_flag_errors() {
    let env = TestEnv::new();
    env.cmd()
        .args([
            "run",
            "ltx-2-19b-distilled:fp8",
            "my positional prompt",
            "--prompt",
            "also a flag prompt",
            "--dry-run",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot combine"));
}

#[test]
fn positional_alone_still_works() {
    // Sanity: the rejection must NOT trip when only positional is given.
    let env = TestEnv::new();
    env.cmd()
        .args(["run", "a positional-only prompt", "--dry-run"])
        .assert()
        // May fail downstream for unrelated reasons (unknown model, etc.)
        // but the stderr must NOT contain "cannot combine".
        .stderr(predicate::str::contains("cannot combine").not());
}

#[test]
fn prompt_flag_alone_still_works() {
    // Sanity: --prompt alone must not trip the rejection.
    let env = TestEnv::new();
    env.cmd()
        .args([
            "run",
            "ltx-2-19b-distilled:fp8",
            "--prompt",
            "lonely flag prompt",
            "--dry-run",
        ])
        .assert()
        .stderr(predicate::str::contains("cannot combine").not());
}

// This blackbox status test intentionally uses a build without Metal so it
// never opens a GPU; native probes are a separate, read-only qualification.
#[cfg(not(feature = "metal"))]
#[tokio::test(flavor = "multi_thread")]
async fn metal_memory_status_is_local_and_skips_config_and_database() {
    let env = TestEnv::new();
    let server = wiremock::MockServer::start().await;
    let missing_home = env.home.join("offline-drive");
    let result = env
        .cmd()
        .env("MOLD_HOME", &missing_home)
        .env("MOLD_HOST", server.uri())
        .args(["system", "metal-memory", "status", "--json"])
        .assert()
        .success();
    let output: serde_json::Value = serde_json::from_slice(&result.get_output().stdout).unwrap();
    assert_eq!(output["scope"], "local_machine");
    assert!(
        !missing_home.exists(),
        "status must not initialize config or DB"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn metal_memory_cli_rejects_invalid_limits_without_initializing_home() {
    let env = TestEnv::new();
    let missing_home = env.home.join("offline-drive");
    for value in ["0", "-1", "4294967296", "16384;whoami"] {
        env.cmd()
            .env("MOLD_HOME", &missing_home)
            .args(["system", "metal-memory", "set", value])
            .assert()
            .failure();
    }
    assert!(!missing_home.exists());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_memory_cli_refuses_unprivileged_mutation_before_config() {
    // Never attempt a write in a root test environment.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let env = TestEnv::new();
    let missing_home = env.home.join("offline-drive");
    env.cmd()
        .env("MOLD_HOME", &missing_home)
        .args(["system", "metal-memory", "set", "16384", "--persist"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires root"));
    assert!(!missing_home.exists());
}

// ── mold mesh-workflow ────────────────────────────────────────────────────

/// The models listing every `mesh-workflow create` reads for stage defaults.
fn models_listing() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "hunyuan3d-2.1:fp16",
            "family": "hunyuan3d",
            "size_gb": 12.0,
            "is_loaded": false,
            "last_used": null,
            "hf_repo": "tencent/Hunyuan3D-2.1",
            "downloaded": true,
            "default_steps": 30,
            "default_guidance": 5.0,
            "default_width": 1024,
            "default_height": 1024,
            "description": "Hunyuan3D 2.1 shape"
        },
        {
            "name": "flux-schnell:q8",
            "family": "flux",
            "size_gb": 12.0,
            "is_loaded": false,
            "last_used": null,
            "hf_repo": "black-forest-labs/FLUX.1-schnell",
            "downloaded": true,
            "default_steps": 4,
            "default_guidance": 0.0,
            "default_width": 1024,
            "default_height": 1024,
            "description": "FLUX Schnell"
        }
    ])
}

fn mesh_workflow_summary(id: &str, state: &str) -> serde_json::Value {
    serde_json::json!({
        "contract_version": 1,
        "id": id,
        "state": state,
        "mode": "mesh_roundtrip",
        "stage_count": 2,
        "current_stage": 2,
        "output_filename": "mold-h3-1700000000000.glb",
        "created_at_ms": 1_700_000_000_000i64,
        "updated_at_ms": 1_700_000_000_001i64
    })
}

/// A roundtrip run puts the mode tag, a canvasless GLB stage and one inline
/// mesh reference on the wire — the shape the 3-D Studio submits.
#[tokio::test(flavor = "multi_thread")]
async fn mesh_workflow_create_submits_the_studios_own_request_shape() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(models_listing()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/mesh-workflows"))
        .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
            "job_id": "mw-1"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mesh = env.home.join("chair.glb");
    std::fs::write(&mesh, b"glTF binary bytes").unwrap();
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args([
            "mesh-workflow",
            "create",
            "--model",
            "hunyuan3d-2.1:fp16",
            "--mesh",
        ])
        .arg(&mesh)
        .args(["--octree", "320", "--target-faces", "40000"])
        .assert()
        .success()
        .stdout(predicate::str::contains("mw-1"))
        .stdout(predicate::str::contains("shape"));

    let posted = server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .find(|request| request.url.path() == "/api/mesh-workflows")
        .expect("the workflow was submitted");
    let body: serde_json::Value = serde_json::from_slice(&posted.body).unwrap();
    assert_eq!(body["mode"], "mesh_roundtrip");
    let request = &body["roundtrip_request"];
    assert_eq!(request["width"], 0);
    assert_eq!(request["height"], 0);
    assert_eq!(request["output_format"], "glb");
    assert_eq!(request["batch_size"], 1);
    assert_eq!(request["prompt"], "");
    assert_eq!(request["mesh"]["texture"], false);
    assert_eq!(request["mesh"]["octree_resolution"], 320);
    assert_eq!(request["mesh"]["target_faces"], 40000);
    let reference = &request["references"][0];
    assert_eq!(reference["kind"], "mesh");
    assert_eq!(reference["format"], "glb");
    assert_eq!(reference["mime_type"], "model/gltf-binary");
    assert_eq!(reference["byte_length"], 17);
    assert_eq!(reference["coordinates"]["up_axis"], "y");
    assert_eq!(reference["provenance"]["name"], "chair.glb");
    assert!(reference["media"]["data"].is_string());
}

/// A workflow is durable on one machine, so `--local` is refused by name
/// rather than quietly rendering something else.
#[tokio::test(flavor = "multi_thread")]
async fn mesh_workflow_create_refuses_local_by_name() {
    let env = TestEnv::new();
    let mesh = env.home.join("chair.glb");
    std::fs::write(&mesh, b"glTF").unwrap();
    env.cmd()
        .env("MOLD_HOST", "http://127.0.0.1:1")
        .args(["mesh-workflow", "create", "--mesh"])
        .arg(&mesh)
        .arg("--local")
        .assert()
        .failure()
        .stderr(predicate::str::contains("durable on one machine"))
        .stderr(predicate::str::contains("mold run"));
}

/// Two inputs naming two different workflows is a question, and it is asked
/// before anything reaches the network.
#[tokio::test(flavor = "multi_thread")]
async fn mesh_workflow_create_refuses_contradictory_inputs_before_any_request() {
    use wiremock::MockServer;

    let env = TestEnv::new();
    let server = MockServer::start().await;
    let mesh = env.home.join("chair.glb");
    std::fs::write(&mesh, b"glTF").unwrap();
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["mesh-workflow", "create", "--prompt", "a fox", "--mesh"])
        .arg(&mesh)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--mode"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

/// The listing, the detail, and the three lifecycle verbs.
#[tokio::test(flavor = "multi_thread")]
async fn mesh_workflow_list_show_and_lifecycle_talk_to_the_server() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/mesh-workflows"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jobs": [mesh_workflow_summary("mw-1", "completed")]
        })))
        .mount(&server)
        .await;
    let mut detail = mesh_workflow_summary("mw-1", "completed");
    detail["request"] = serde_json::json!({
        "mode": "mesh_roundtrip",
        "roundtrip_request": {
            "prompt": "",
            "model": "hunyuan3d-2.1:fp16",
            "width": 0,
            "height": 0,
            "steps": 30,
            "guidance": 5.0,
            "batch_size": 1,
            "strength": 0.75,
            "control_scale": 1.0,
            "hdr_exr_full_float": false,
            "gif_preview": false,
            "output_format": "glb"
        }
    });
    detail["stages"] = serde_json::json!([
        {
            "index": 0,
            "kind": "shape",
            "state": "completed",
            "artifacts": [{
                "role": "final_glb",
                "relative_path": "stage-0/mesh.glb",
                "media_type": "model/gltf-binary",
                "sha256": "abc",
                "byte_length": 2048
            }]
        },
        { "index": 1, "kind": "finalize", "state": "completed", "artifacts": [] }
    ]);
    Mock::given(method("GET"))
        .and(path("/api/mesh-workflows/mw-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(detail))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/mesh-workflows/mw-1/resume"))
        .respond_with(ResponseTemplate::new(202))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/mesh-workflows/mw-1/cancel"))
        .respond_with(ResponseTemplate::new(202))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/api/mesh-workflows/mw-1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["mesh-workflow", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("mw-1"))
        .stdout(predicate::str::contains("mesh_roundtrip"));

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["mesh-workflow", "show", "mw-1"])
        .assert()
        .success()
        .stdout(predicate::str::contains("final_glb"))
        .stdout(predicate::str::contains("stage-0/mesh.glb"))
        .stdout(predicate::str::contains("mold-h3-1700000000000.glb"));

    for verb in ["resume", "cancel", "delete"] {
        env.cmd()
            .env("MOLD_HOST", server.uri())
            .args(["mesh-workflow", verb, "mw-1"])
            .assert()
            .success()
            .stdout(predicate::str::contains("mw-1"));
    }
}

/// Deleting an unsettled workflow surfaces the SERVER's own wording, which is
/// the sentence that says what to do instead.
#[tokio::test(flavor = "multi_thread")]
async fn mesh_workflow_delete_surfaces_the_servers_refusal() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/api/mesh-workflows/mw-2"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "cancel or wait for the mesh workflow before deleting it"
        })))
        .mount(&server)
        .await;
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["mesh-workflow", "delete", "mw-2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cancel or wait"));
}

// ── mold search ───────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn search_renders_the_hosts_page_and_names_the_install_command() {
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/catalog/search"))
        .and(query_param("q", "flux"))
        .and(query_param("kind", "lora"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "entries": [{
                "id": "cv:827325",
                "name": "Flux Skin Texture",
                "author": "someone",
                "family": "flux",
                "kind": "lora",
                "size_bytes": 167_938_890u64,
                "download_count": 2_400_000u64,
                "installed": false
            }],
            "page": 1,
            "page_size": 20,
            "total": 1,
            "provider_errors": [
                { "source": "civitai", "message": "rate limited", "retry_after_seconds": 30 }
            ]
        })))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["search", "flux", "--kind", "lora"])
        .assert()
        .success()
        .stdout(predicate::str::contains("cv:827325"))
        .stdout(predicate::str::contains("Flux Skin Texture"))
        .stdout(predicate::str::contains("2.4M"))
        .stdout(predicate::str::contains("mold pull <id>"))
        // A failed provider is a warning beside the rows, on stderr, so a
        // piped search still carries only results.
        .stderr(predicate::str::contains("civitai"))
        .stderr(predicate::str::contains("rate limited"));
}

#[tokio::test(flavor = "multi_thread")]
async fn search_json_prints_the_page_verbatim() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/catalog/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "entries": [{
                "id": "hf:black-forest-labs/FLUX.1-dev",
                "name": "FLUX.1 [dev]",
                "family": "flux",
                "kind": "checkpoint"
            }],
            "page": 1,
            "page_size": 20,
            "total": 1
        })))
        .mount(&server)
        .await;

    let result = env
        .cmd()
        .env("MOLD_HOST", server.uri())
        .args(["search", "flux", "--json"])
        .assert()
        .success();
    let page: serde_json::Value =
        serde_json::from_slice(&result.get_output().stdout).expect("valid JSON page");
    assert_eq!(page["entries"][0]["id"], "hf:black-forest-labs/FLUX.1-dev");
    assert_eq!(page["total"], 1);
}

/// The host's own refusal of a sort it does not accept is what the user reads.
#[tokio::test(flavor = "multi_thread")]
async fn search_surfaces_an_unknown_filter_from_the_host() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/catalog/search"))
        .respond_with(ResponseTemplate::new(400).set_body_string("unknown family: nonesuch"))
        .mount(&server)
        .await;
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["search", "flux", "--family", "nonesuch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown family: nonesuch"));
}

// ── mold downloads ────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn downloads_list_shows_active_queued_and_finished_rows() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/downloads"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "active_jobs": [{
                "id": "dl-1",
                "model": "flux-dev:q4",
                "status": "active",
                "files_done": 1,
                "files_total": 4,
                "bytes_done": 500,
                "bytes_total": 1000,
                "current_file": "transformer.gguf"
            }],
            "queued": [{
                "id": "dl-2",
                "model": "sdxl-turbo:fp16",
                "status": "queued",
                "files_done": 0,
                "files_total": 0,
                "bytes_done": 0,
                "bytes_total": 0
            }],
            "history": []
        })))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("dl-1"))
        .stdout(predicate::str::contains("flux-dev:q4"))
        .stdout(predicate::str::contains("50%"))
        .stdout(predicate::str::contains("transformer.gguf"))
        .stdout(predicate::str::contains("dl-2"));
}

/// The download listing teaches the shell its ids, so `mold downloads cancel
/// <TAB>` offers what `mold downloads list` just showed.
///
/// A completer cannot ask a server, so the commands that already fetched the
/// data write `$MOLD_HOME/completion-cache.json`. Driving the real binary
/// means dropping the refresh call fails this.
#[tokio::test(flavor = "multi_thread")]
async fn downloads_list_teaches_the_shell_its_ids() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/downloads"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "active_jobs": [{
                "id": "dl-active",
                "model": "flux-dev:q4",
                "status": "active",
                "files_done": 1,
                "files_total": 4,
                "bytes_done": 500,
                "bytes_total": 1000
            }],
            "queued": [{
                "id": "dl-queued",
                "model": "sdxl-turbo:fp16",
                "status": "queued",
                "files_done": 0,
                "files_total": 0,
                "bytes_done": 0,
                "bytes_total": 0
            }],
            "history": [{
                "id": "dl-done",
                "model": "z-image:q8",
                "status": "completed",
                "files_done": 2,
                "files_total": 2,
                "bytes_done": 10,
                "bytes_total": 10
            }]
        })))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "list"])
        .assert()
        .success();

    let cache = read_completion_cache(&env);
    let ids = cache_strings(&cache, "download_ids");
    for id in ["dl-active", "dl-queued", "dl-done"] {
        assert!(ids.contains(&id.to_string()), "{ids:?}");
    }
    assert_eq!(
        cache_strings(&cache, "hosts"),
        vec![server.uri().trim_end_matches('/').to_string()]
    );
}

/// The workflow listing teaches the shell its ids, so every other
/// `mold mesh-workflow` verb completes.
#[tokio::test(flavor = "multi_thread")]
async fn mesh_workflow_list_teaches_the_shell_its_ids() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/mesh-workflows"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jobs": [
                mesh_workflow_summary("mw-1", "completed"),
                mesh_workflow_summary("mw-2", "running")
            ]
        })))
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["mesh-workflow", "list"])
        .assert()
        .success();

    let cache = read_completion_cache(&env);
    let ids = cache_strings(&cache, "workflow_ids");
    assert!(ids.contains(&"mw-1".to_string()), "{ids:?}");
    assert!(ids.contains(&"mw-2".to_string()), "{ids:?}");
    assert_eq!(
        cache_strings(&cache, "hosts"),
        vec![server.uri().trim_end_matches('/').to_string()]
    );
}

/// A 409 names the download already doing the work rather than failing.
#[tokio::test(flavor = "multi_thread")]
async fn downloads_add_reports_a_conflict_as_the_job_already_running() {
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/downloads"))
        .and(body_json(serde_json::json!({ "model": "flux-dev:q4" })))
        .respond_with(ResponseTemplate::new(409).set_body_json(serde_json::json!({
            "id": "dl-7",
            "position": 3
        })))
        .mount(&server)
        .await;
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "add", "flux-dev:q4"])
        .assert()
        .success()
        .stdout(predicate::str::contains("already in the queue"))
        .stdout(predicate::str::contains("dl-7"));
}

/// A catalog id belongs to the other download door, and nothing is sent.
#[tokio::test(flavor = "multi_thread")]
async fn downloads_add_refuses_a_catalog_id_and_points_at_pull() {
    use wiremock::MockServer;

    let env = TestEnv::new();
    let server = MockServer::start().await;
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "add", "cv:827325"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("mold pull cv:827325"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn downloads_cancel_deletes_the_job_by_id() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/api/downloads/dl-1"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "cancel", "dl-1"])
        .assert()
        .success()
        .stdout(predicate::str::contains("dl-1"));
}

/// The watcher opens with the server's snapshot, then prints one line per
/// event until the stream ends.
#[tokio::test(flavor = "multi_thread")]
async fn downloads_watch_opens_with_the_snapshot_and_follows_the_queue() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    let body = format!(
        "event: download\ndata: {}\n\nevent: download\ndata: {}\n\nevent: download\ndata: {}\n\n",
        serde_json::json!({
            "type": "snapshot",
            "listing": { "active_jobs": [], "queued": [], "history": [] }
        }),
        serde_json::json!({
            "type": "started", "id": "dl-1", "files_total": 2, "bytes_total": 1048576
        }),
        serde_json::json!({ "type": "job_done", "id": "dl-1", "model": "flux-dev:q4" }),
    );
    Mock::given(method("GET"))
        .and(path("/api/downloads/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "watch"])
        .assert()
        .success()
        .stdout(predicate::str::contains("0 transferring"))
        .stdout(predicate::str::contains("started dl-1"))
        .stdout(predicate::str::contains("flux-dev:q4 ready"));
}

// ── mold list --json ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn list_json_prints_the_rows_the_api_serves() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(models_listing()))
        .mount(&server)
        .await;

    let result = env
        .cmd()
        .env("MOLD_HOST", server.uri())
        .args(["list", "--json"])
        .assert()
        .success();
    let rows: serde_json::Value =
        serde_json::from_slice(&result.get_output().stdout).expect("valid JSON rows");
    assert_eq!(rows[0]["name"], "hunyuan3d-2.1:fp16");
    assert_eq!(rows[0]["default_steps"], 30);
}

/// A blank id reaches no route at all.
///
/// An empty path segment resolves to the COLLECTION route, so every lifecycle
/// verb was answered by a listing (or a 404 from the wrong handler) and the
/// user read an error about that answer instead of about the blank argument.
#[tokio::test(flavor = "multi_thread")]
async fn a_blank_id_is_refused_before_any_request() {
    use wiremock::MockServer;

    let env = TestEnv::new();
    let server = MockServer::start().await;
    for verb in ["show", "events", "resume", "cancel", "delete"] {
        env.cmd()
            .env("MOLD_HOST", server.uri())
            .args(["mesh-workflow", verb, ""])
            .assert()
            .failure()
            .stderr(predicate::str::contains("workflow id is required"));
    }
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["downloads", "cancel", "  "])
        .assert()
        .failure()
        .stderr(predicate::str::contains("download id is required"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

/// With no `--model`, a roundtrip picks the 2.1 shape checkpoint, which is the
/// only tier that can run one.
#[tokio::test(flavor = "multi_thread")]
async fn a_roundtrip_defaults_to_the_only_checkpoint_that_can_run_it() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let env = TestEnv::new();
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(models_listing()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/mesh-workflows"))
        .respond_with(
            ResponseTemplate::new(202).set_body_json(serde_json::json!({ "job_id": "mw-9" })),
        )
        .mount(&server)
        .await;

    let mesh = env.home.join("armchair.glb");
    std::fs::write(&mesh, b"glTF binary bytes").unwrap();
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args(["mesh-workflow", "create", "--mesh"])
        .arg(&mesh)
        .assert()
        .success()
        .stdout(predicate::str::contains("mw-9"));

    let posted = server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .find(|request| request.url.path() == "/api/mesh-workflows")
        .expect("the workflow was submitted");
    let body: serde_json::Value = serde_json::from_slice(&posted.body).unwrap();
    assert_eq!(body["mode"], "mesh_roundtrip");
    assert_eq!(body["roundtrip_request"]["model"], "hunyuan3d-2.1:fp16");
}

/// Naming a checkpoint the mode cannot run says which flag fixes it, and
/// nothing is submitted.
#[tokio::test(flavor = "multi_thread")]
async fn a_roundtrip_on_the_wrong_checkpoint_names_the_flag() {
    use wiremock::MockServer;

    let env = TestEnv::new();
    let server = MockServer::start().await;
    let mesh = env.home.join("armchair.glb");
    std::fs::write(&mesh, b"glTF").unwrap();
    env.cmd()
        .env("MOLD_HOST", server.uri())
        .args([
            "mesh-workflow",
            "create",
            "--model",
            "hunyuan3d-mini-turbo:fp16",
            "--mesh",
        ])
        .arg(&mesh)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--model hunyuan3d-2.1:fp16"));
    assert!(server.received_requests().await.unwrap().is_empty());
}
