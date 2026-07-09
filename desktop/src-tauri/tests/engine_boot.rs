//! Boots the embedded mold engine (CPU build — no metal feature in tests)
//! on an ephemeral port and exercises the wire contract the webview relies
//! on: health, capabilities, API-key auth, and graceful shutdown.

use std::time::Duration;

use mold_desktop_lib::server;

#[tokio::test(flavor = "multi_thread")]
async fn engine_boots_authenticates_and_shuts_down() {
    let models_dir = tempfile::tempdir().unwrap();
    let state_dir = tempfile::tempdir().unwrap();
    // Isolate from the user's real mold.db and enforce a known API key.
    std::env::set_var("MOLD_DB_PATH", state_dir.path().join("mold.db"));
    std::env::set_var("MOLD_API_KEY", "desktop-test-key");

    let engine = server::start_engine(
        models_dir.path().to_path_buf(),
        mold_core::types::GpuSelection::All,
    )
    .expect("engine spawns");
    let base = engine.base_url();

    assert!(
        server::wait_healthy(&base, Duration::from_secs(30)).await,
        "engine did not become healthy"
    );
    assert!(server::is_mold_server(&base).await);

    let client = reqwest::Client::new();

    // Auth: /api/models is protected when MOLD_API_KEY is set…
    let unauthorized = client
        .get(format!("{base}/api/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), reqwest::StatusCode::UNAUTHORIZED);

    // …and answers with the key attached.
    let authorized = client
        .get(format!("{base}/api/models"))
        .header("X-Api-Key", "desktop-test-key")
        .send()
        .await
        .unwrap();
    assert!(authorized.status().is_success());

    // Graceful shutdown over loopback.
    let shutdown = client
        .post(format!("{base}/api/shutdown"))
        .header("X-Api-Key", "desktop-test-key")
        .send()
        .await
        .unwrap();
    assert!(shutdown.status().is_success());
    engine.join(Duration::from_secs(10));
}
