//! Boots the embedded mold engine (CPU build — no metal feature in tests)
//! on an ephemeral port and exercises the wire contract the webview relies
//! on: health, capabilities, API-key auth, and graceful shutdown.

use std::io::Write;
use std::time::Duration;

use mold_desktop_lib::server;
use tokio::io::AsyncWriteExt;

const API_KEY: &str = "desktop-test-key";

/// The app's own stop budget: `stop_local_engine` waits this long before it
/// tells the user gallery authority is stuck with the server. Any shutdown
/// that reaches it is already broken for users, not merely slow here.
const APP_STOP_BUDGET: Duration = Duration::from_secs(10);

/// The engine reads its environment while it boots, and this binary runs its
/// tests on parallel threads, so each boot holds this while the variables
/// are set and until the engine has read them.
static BOOT_ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct BootedEngine {
    engine: server::EngineHandle,
    base: String,
    _models_dir: tempfile::TempDir,
    _state_dir: tempfile::TempDir,
}

async fn boot() -> BootedEngine {
    // The server's own log, straight to stderr. Shutdown awaits a dozen
    // background owners in sequence; when one stalls, this is what names it.
    // `writeln!` rather than the print macros because libtest's capture hook
    // only intercepts those, and a passing-but-slow run must stay readable.
    // `RUST_LOG` overrides the filter so a local reproduction can widen it
    // without editing the test.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("mold_server=debug,warn"));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .try_init();

    let models_dir = tempfile::tempdir().unwrap();
    let state_dir = tempfile::tempdir().unwrap();
    let guard = BOOT_ENV.lock().await;
    // Isolate from the user's real database and gallery. The server waits for
    // its finite gallery observers during shutdown, so inheriting a populated
    // output directory makes this boot-contract test depend on runner state.
    std::env::set_var("MOLD_DB_PATH", state_dir.path().join("mold.db"));
    std::env::set_var("MOLD_OUTPUT_DIR", state_dir.path().join("output"));
    std::env::set_var("MOLD_API_KEY", API_KEY);

    let port = server::allocate_port("127.0.0.1").unwrap();
    let engine = server::start_engine(
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
    drop(guard);

    BootedEngine {
        engine,
        base,
        _models_dir: models_dir,
        _state_dir: state_dir,
    }
}

/// Request a graceful stop over loopback and wait for the engine thread,
/// reporting how long it took whether or not it made the bound.
async fn shutdown_and_join(booted: &mut BootedEngine, bound: Duration) -> Duration {
    let client = reqwest::Client::new();
    let shutdown = client
        .post(format!("{}/api/shutdown", booted.base))
        .header("X-Api-Key", API_KEY)
        .send()
        .await
        .unwrap();
    assert!(shutdown.status().is_success());
    // Nothing of this client may outlive the request: a response this test
    // holds is a response the engine is still serving.
    let _ = shutdown.bytes().await;
    drop(client);

    // Embedded shutdown has no process-ending deadline of its own (that would
    // take the whole app down), so this bound is what catches a stalled
    // phase. It is the app's own stop budget with headroom for a loaded
    // runner: a shutdown anywhere near it is already broken for users.
    let started = std::time::Instant::now();
    let exited = booted.engine.join(bound);
    let elapsed = started.elapsed();
    // Not `eprintln!`: libtest captures the print macros and shows them only
    // for a FAILING test, so a run drifting toward the bound would say
    // nothing. Writing to the stderr handle bypasses that capture.
    let _ = writeln!(
        std::io::stderr(),
        "engine shutdown took {elapsed:.1?} (bound {bound:?}, app stop budget {APP_STOP_BUDGET:?})"
    );
    assert!(
        exited,
        "engine thread did not exit within {bound:?} (waited {elapsed:.1?})"
    );
    assert!(!booted.engine.is_alive(), "engine thread did not exit");
    elapsed
}

#[tokio::test(flavor = "multi_thread")]
async fn engine_boots_authenticates_and_shuts_down() {
    let mut booted = boot().await;
    let base = booted.base.clone();

    assert!(server::is_mold_server(&base).await);
    assert!(server::accepts_api_key(&base, API_KEY).await);
    assert!(!server::accepts_api_key(&base, "different-key").await);

    let client = reqwest::Client::new();

    // Auth: /api/models is protected when MOLD_API_KEY is set…
    let unauthorized = client
        .get(format!("{base}/api/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), reqwest::StatusCode::UNAUTHORIZED);
    let _ = unauthorized.bytes().await;

    // …and answers with the key attached.
    let authorized = client
        .get(format!("{base}/api/models"))
        .header("X-Api-Key", API_KEY)
        .send()
        .await
        .unwrap();
    assert!(authorized.status().is_success());
    // READ the body. It is the whole model catalog, over a megabyte, and a
    // response this test leaves unread is a response the engine is still
    // writing: once the kernel's loopback buffers are full the server sits
    // in the middle of that write, the request is in flight, and graceful
    // shutdown waits for a client that will never read another byte. Whether
    // the buffers happen to hold the whole body is up to the kernel's window
    // autotuning, which is exactly the coin flip that skipped the desktop
    // nightly for days: every shutdown step then ran within 2 ms of this test
    // giving up, because unwinding is what finally closed the socket.
    let models = authorized.bytes().await.unwrap();
    assert!(
        serde_json::from_slice::<serde_json::Value>(&models).is_ok(),
        "/api/models is JSON"
    );
    drop(client);

    // Liveness tracks the engine thread: alive while serving…
    assert!(booted.engine.is_alive());

    // …and reported dead once the thread exits, so the connection state
    // machine knows to restart instead of handing out a dead base URL.
    shutdown_and_join(&mut booted, Duration::from_secs(60)).await;
}

/// A client that stops reading — or, here, stops writing — must not be able
/// to hold the engine open. The webview is a client the app does not control:
/// a paused video element stops draining its socket, and a request whose body
/// never finishes is a request the server is still serving. Graceful
/// shutdown waits for in-flight requests by design; the embedded engine
/// bounds that wait, because its host has a stop budget and cannot end the
/// process to enforce it.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_that_stalls_cannot_hold_the_engine_open() {
    let mut booted = boot().await;
    let addr = booted.base.trim_start_matches("http://").to_string();

    // A request that never completes: the JSON extractor awaits 4096 bytes
    // that are never sent, so this connection stays in flight until the
    // server gives up on it. Held for the whole shutdown.
    let mut stalled = tokio::net::TcpStream::connect(&addr).await.unwrap();
    stalled
        .write_all(
            format!(
                "POST /api/generate/estimate HTTP/1.1\r\nHost: {addr}\r\nX-Api-Key: {API_KEY}\r\n\
                 Content-Type: application/json\r\nContent-Length: 4096\r\n\r\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    stalled.flush().await.unwrap();

    let elapsed = shutdown_and_join(&mut booted, APP_STOP_BUDGET).await;
    assert!(
        elapsed >= server::HTTP_DRAIN_GRACE,
        "the stalled request was cut before its grace ({elapsed:.1?} < {:?})",
        server::HTTP_DRAIN_GRACE
    );
    drop(stalled);
}
