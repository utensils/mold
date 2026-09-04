//! Boots the embedded mold engine (CPU build — no metal feature in tests)
//! on an ephemeral port and exercises the wire contract the webview relies
//! on: health, capabilities, API-key auth, and graceful shutdown.

use std::io::Write;
use std::time::Duration;

use mold_desktop_lib::server;

#[tokio::test(flavor = "multi_thread")]
async fn engine_boots_authenticates_and_shuts_down() {
    // The server's own log, straight to stderr. Shutdown awaits a dozen
    // background owners in sequence; when one stalls, this is what names it.
    // `writeln!` rather than the print macros because libtest's capture hook
    // only intercepts those, and a passing-but-slow run must stay readable.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("mold_server=debug,warn")
        .try_init();

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

    // Drop the client — and with it the connection pool — BEFORE waiting on
    // the thread. Axum's graceful shutdown drains live connections, and this
    // client's pooled keep-alive connection is one, so holding it here makes
    // the test the reason the server cannot finish. That is exactly what the
    // intermittent CI failures were: the whole shutdown sequence ran within
    // 2 ms of this test giving up and unwinding, never before it.
    drop(shutdown);
    drop(client);

    // …and reported dead once the thread exits, so the connection state
    // machine knows to restart instead of handing out a dead base URL.
    //
    // Embedded shutdown has no enforced deadline of its own: the hard-exit
    // deadline is off for an embedded server (it would take the whole app
    // down), so this bound is the only thing that catches a stalled phase.
    // It is the app's own stop budget with headroom for a loaded runner —
    // `stop_local_engine` gives the engine 10 s before it tells the user
    // gallery authority is stuck with the server, so a shutdown anywhere near
    // this bound is already broken for users, not merely slow here.
    let budget = Duration::from_secs(60);
    let started = std::time::Instant::now();
    let exited = engine.join(budget);
    let elapsed = started.elapsed();
    // Not `eprintln!`: libtest captures the print macros and shows them only
    // for a FAILING test, so a run drifting toward the bound would say
    // nothing. Writing to the stderr handle bypasses that capture.
    let _ = writeln!(
        std::io::stderr(),
        "engine shutdown took {elapsed:.1?} (bound {budget:?}, app stop budget 10s)"
    );
    assert!(
        exited,
        "engine thread did not exit within {budget:?} (waited {elapsed:.1?})"
    );
    assert!(!engine.is_alive(), "engine thread did not exit");
}
