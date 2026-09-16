//! The C ABI the native macOS app links against.
//!
//! mold's engine is `mold_server::run_server`, an async function already built
//! to be embedded in a process it does not own: `HARD_SHUTDOWN_EXIT` defaults
//! off so it RETURNS rather than ending the host, and `bound_http_drain` exists
//! for a host that cannot end itself. The Tauri app does exactly this on Metal
//! today; this crate is that arrangement behind five plain C functions.
//!
//! Deliberately narrow. Nothing complex crosses the boundary: the app talks to
//! the engine over HTTP on loopback, which is the SAME wire contract it uses
//! for a machine across the network. Marshalling `GenerateRequest` through FFI
//! would be rebuilding HTTP without the HTTP.

use std::ffi::{c_char, CStr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// How long the HTTP side may take to drain before the engine thread gives up.
/// The app owns the process, so an overrun must not wedge quitting.
const HTTP_DRAIN_GRACE: Duration = Duration::from_secs(2);

static ENGINE: OnceLock<Mutex<Option<std::thread::JoinHandle<()>>>> = OnceLock::new();
static ALIVE: AtomicBool = AtomicBool::new(false);
static BOOTSTRAPPED: AtomicBool = AtomicBool::new(false);

fn engine() -> &'static Mutex<Option<std::thread::JoinHandle<()>>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

/// Borrows a C string. Returns `None` for null or non-UTF-8.
unsafe fn str_arg<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

/// Prepares process-wide state the engine needs, once.
///
/// Every mold binary does this same preamble before `run_server`. It is
/// separate from `start` because several pieces of it are one-shot for the life
/// of the process -- the models-dir override is a `OnceLock` and tracing
/// installs a global subscriber -- so the engine starts at most once per
/// process and changing the models directory means relaunching the app.
///
/// Returns 0 on success, 1 on failure.
///
/// # Safety
/// Every pointer must be null or a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn mold_engine_bootstrap(
    mold_home: *const c_char,
    api_key: *const c_char,
    log_dir: *const c_char,
) -> i32 {
    if BOOTSTRAPPED.swap(true, Ordering::SeqCst) {
        return 0;
    }

    // Must be set before anything reads a config, and before any thread spawns:
    // the auth layer reads MOLD_API_KEY from the environment, and `setenv` is
    // not safe once other threads are running.
    if let Some(home) = unsafe { str_arg(mold_home) } {
        if !home.is_empty() {
            unsafe { std::env::set_var("MOLD_HOME", home) };
        }
    }
    if let Some(key) = unsafe { str_arg(api_key) } {
        if !key.is_empty() {
            unsafe { std::env::set_var("MOLD_API_KEY", key) };
        }
    }

    // A saved MOLD_HOME pointing at an offline external drive is recoverable
    // rather than fatal for an app: fall back to defaults and carry on, which
    // is what the desktop app does too.
    let _ = mold_core::Config::ensure_saved_mold_dir_available();

    // One-shot config.toml -> DB migration, plus the DB overlay that every
    // later `Config::load_or_default()` depends on.
    mold_db::config_sync::install_config_post_load_hook();

    let config = mold_core::Config::load_or_default();
    let logs = unsafe { str_arg(log_dir) }
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| config.resolved_log_dir());
    // File-only: a windowed app has no console to write to, and stdout here
    // would be discarded anyway.
    let guard = mold_server::logging::init_tracing_file_only(&config.logging, "info", logs);
    // The guard flushes on drop, so it has to outlive every later call.
    std::mem::forget(guard);

    0
}

/// Reserves a loopback port and returns it, or 0 on failure.
///
/// The probe is dropped before the engine rebinds, so there is a window where
/// another process could take it. mold's own desktop app has the same race for
/// the same reason: `run_server` does not report the port it bound. The fix
/// upstream is `run_server_with_listener`.
#[no_mangle]
pub extern "C" fn mold_engine_alloc_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|probe| probe.local_addr())
        .map(|addr| addr.port())
        .unwrap_or(0)
}

/// Starts the engine on its own thread. Returns 0 if it was started.
///
/// # Safety
/// Pointers must be null or valid NUL-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn mold_engine_start(
    bind: *const c_char,
    port: u16,
    models_dir: *const c_char,
) -> i32 {
    let mut slot = match engine().lock() {
        Ok(slot) => slot,
        Err(_) => return 1,
    };
    if slot.is_some() {
        return 0; // Already running.
    }

    let bind = unsafe { str_arg(bind) }.unwrap_or("127.0.0.1").to_string();
    let config = mold_core::Config::load_or_default();
    let models = unsafe { str_arg(models_dir) }
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| config.resolved_models_dir());
    if std::fs::create_dir_all(&models).is_err() {
        return 1;
    }
    let gpu_selection = config.gpu_selection();
    let queue_size = config.queue_size();

    // NOT `allow_hard_shutdown_exit()`: that lets the engine end the process
    // when its shutdown budget expires, which here would take the whole app
    // down over a slow engine stop.
    mold_server::bound_http_drain(HTTP_DRAIN_GRACE);

    let handle = std::thread::Builder::new()
        .name("mold-engine".into())
        .spawn(move || {
            ALIVE.store(true, Ordering::SeqCst);
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .thread_name("mold-engine-worker")
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(_) => {
                    ALIVE.store(false, Ordering::SeqCst);
                    return;
                }
            };
            let result = runtime.block_on(mold_server::run_server(
                &bind,
                port,
                models,
                gpu_selection,
                queue_size,
            ));
            if let Err(error) = result {
                tracing_error(&error);
            }
            // Plain `drop`, not `shutdown_timeout`: a GPU worker mid-render
            // holds a blocking thread, and cutting the runtime out from under
            // it is how you lose a job that was about to finish.
            drop(runtime);
            ALIVE.store(false, Ordering::SeqCst);
        });

    match handle {
        Ok(handle) => {
            *slot = Some(handle);
            0
        }
        Err(_) => 1,
    }
}

fn tracing_error(error: &anyhow::Error) {
    eprintln!("mold engine stopped: {error:#}");
}

/// Whether the engine thread is still running.
#[no_mangle]
pub extern "C" fn mold_engine_is_alive() -> bool {
    ALIVE.load(Ordering::SeqCst)
}

/// Waits for the engine thread to finish, up to `timeout_ms`.
///
/// Stopping is a `POST /api/shutdown` from the app -- the server's shutdown
/// trigger is not exposed to an embedder any other way -- so this only waits
/// for the thread that request already asked to stop.
#[no_mangle]
pub extern "C" fn mold_engine_join(timeout_ms: u64) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
    while ALIVE.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    if ALIVE.load(Ordering::SeqCst) {
        return false;
    }
    if let Ok(mut slot) = engine().lock() {
        if let Some(handle) = slot.take() {
            let _ = handle.join();
        }
    }
    true
}
