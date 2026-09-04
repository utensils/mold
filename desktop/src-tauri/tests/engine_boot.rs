//! Boots the embedded mold engine (CPU build — no metal feature in tests)
//! on an ephemeral port and exercises the wire contract the webview relies
//! on: health, capabilities, API-key auth, and graceful shutdown.

use std::time::Duration;

use mold_desktop_lib::server;

#[tokio::test(flavor = "multi_thread")]
async fn engine_boots_authenticates_and_shuts_down() {
    let models_dir = tempfile::tempdir().unwrap();
    let state_dir = tempfile::tempdir().unwrap();
    // Isolate from the user's real database and gallery. The server waits for
    // its finite gallery observers during shutdown, so inheriting a populated
    // output directory makes this boot-contract test depend on runner state.
    std::env::set_var("MOLD_DB_PATH", state_dir.path().join("mold.db"));
    std::env::set_var("MOLD_OUTPUT_DIR", state_dir.path().join("output"));
    std::env::set_var("MOLD_API_KEY", "desktop-test-key");

    let port = server::allocate_port("127.0.0.1").unwrap();
    let mut engine = server::start_engine(
        "127.0.0.1",
        port,
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
    assert!(server::accepts_api_key(&base, "desktop-test-key").await);
    assert!(!server::accepts_api_key(&base, "different-key").await);

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

    // Liveness tracks the engine thread: alive while serving…
    assert!(engine.is_alive());

    // Graceful shutdown over loopback.
    let shutdown = client
        .post(format!("{base}/api/shutdown"))
        .header("X-Api-Key", "desktop-test-key")
        .send()
        .await
        .unwrap();
    assert!(shutdown.status().is_success());

    // …and reported dead once the thread exits, so the connection state
    // machine knows to restart instead of handing out a dead base URL.
    //
    // The bound is the server's own shutdown contract, not a smaller number:
    // `run_server` waits up to `DEFAULT_SHUTDOWN_ABORT_SECS` for GPU owners
    // and then stops waiting, so an engine that exits inside that budget is
    // behaving. A 15 s bound here failed on slow macOS runners at 15.6-16 s
    // (main runs 33616295687, 33820692419, 33823114932) with no server hang
    // behind it. The elapsed time is printed either way so a run that drifts
    // toward the budget is visible before it fails.
    let budget = Duration::from_secs(mold_server::DEFAULT_SHUTDOWN_ABORT_SECS + 15);
    let started = std::time::Instant::now();
    let exited = engine.join(budget);
    let elapsed = started.elapsed();
    eprintln!("engine shutdown took {elapsed:.1?} (bound {budget:?})");
    assert!(
        exited,
        "engine thread did not exit within {budget:?} (waited {elapsed:.1?})"
    );
    assert!(!engine.is_alive(), "engine thread did not exit");
}
