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

/// The origin the embedded engine allows, which is deliberately one no browser
/// page can ever present.
///
/// `build_cors_layer` (`crates/mold-server/src/lib.rs`) falls back to
/// `CorsLayer::permissive()` -- `Access-Control-Allow-Origin: *` -- for an
/// empty or absent `MOLD_CORS_ORIGIN`, so an engine started without one is
/// READABLE by any page that finds its port. A non-empty value takes the
/// restrictive arm, which is `AllowOrigin::exact`: tower-http stores it as
/// `OriginInner::Const` and echoes it verbatim as the `Access-Control-Allow-Origin`
/// header (`tower-http-0.6.8/src/cors/allow_origin.rs:40-42`), so the check that
/// matters is the BROWSER's -- it lets a cross-origin read through only when
/// that header is `*` or byte-equal to the requesting page's own origin.
///
/// A serialized origin is `scheme "://" host [":" port]`, or the literal
/// `null`; none of those can contain a space. This value contains two, so no
/// page can ever be granted by it, while it still parses as a `HeaderValue`
/// (space is inside the allowed 0x20..=0x7E range) -- a malformed value is a
/// HARD startup error, not a fallback, so it has to.
///
/// The app itself is unaffected: `URLSession` sends no `Origin` header and
/// enforces no CORS, and the local `MoldHost` carries the API key.
const EMBEDDED_CORS_ORIGIN: &str = "mold-embedded-engine no browser origin";

static ENGINE: OnceLock<Mutex<Option<std::thread::JoinHandle<()>>>> = OnceLock::new();
static ALIVE: AtomicBool = AtomicBool::new(false);
static BOOTSTRAPPED: AtomicBool = AtomicBool::new(false);

fn engine() -> &'static Mutex<Option<std::thread::JoinHandle<()>>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

/// Clears `ALIVE` however the engine thread ends.
///
/// It used to be cleared by a store at the BOTTOM of the thread body, which a
/// panic anywhere inside `run_server` skips: `mold_engine_is_alive` then
/// answered true for the life of the process, `mold_engine_join` always burned
/// its whole timeout, and the app kept "This Mac" in the machine list pointing
/// at a port nothing was listening on (review 05-M1).
struct AliveGuard;

impl Drop for AliveGuard {
    fn drop(&mut self) {
        ALIVE.store(false, Ordering::SeqCst);
    }
}

/// Runs `body` so a panic can never cross the C ABI.
///
/// An unwind through an `extern "C"` frame aborts the process -- for an app
/// that is a crash report with no engine in it. Every entry point answers its
/// own failure value instead.
fn guarded<T>(what: &'static str, fallback: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            tracing::error!("the mold engine panicked in {what}");
            fallback
        }
    }
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
    guarded("bootstrap", 1, || unsafe {
        bootstrap(mold_home, api_key, log_dir)
    })
}

unsafe fn bootstrap(
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
    // Set unconditionally and NOT taken from the caller: the embedded engine
    // is never served to a browser, so there is no origin an embedder could
    // legitimately name. An operator who has set one already gets theirs.
    if std::env::var_os("MOLD_CORS_ORIGIN").is_none_or(|value| value.is_empty()) {
        unsafe { std::env::set_var("MOLD_CORS_ORIGIN", EMBEDDED_CORS_ORIGIN) };
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
    guarded("alloc_port", 0, || {
        std::net::TcpListener::bind(("127.0.0.1", 0))
            .and_then(|probe| probe.local_addr())
            .map(|addr| addr.port())
            .unwrap_or(0)
    })
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
    guarded("start", 1, || unsafe { start(bind, port, models_dir) })
}

unsafe fn start(bind: *const c_char, port: u16, models_dir: *const c_char) -> i32 {
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

    // Set HERE, not at the top of the thread body: `mold_engine_join` polls
    // this flag, and in the window between the spawn returning and the thread
    // being scheduled the old placement made `join` see a dead engine and fall
    // straight into an UNBOUNDED `handle.join()` (review 05-M9).
    ALIVE.store(true, Ordering::SeqCst);

    let handle = std::thread::Builder::new()
        .name("mold-engine".into())
        .spawn(move || {
            // Cleared however this thread ends -- return, `Err`, or unwind.
            let _alive = AliveGuard;
            guarded("run_server", (), || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .thread_name("mold-engine-worker")
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        tracing::error!(%error, "the mold engine could not build its runtime");
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
                    // Through tracing, which `bootstrap` has pointed at a
                    // file: a windowed app has no console, so an `eprintln!`
                    // here went nowhere and the terminal error was lost.
                    tracing::error!(error = format!("{error:#}"), "the mold engine stopped");
                }
                // Plain `drop`, not `shutdown_timeout`: a GPU worker
                // mid-render holds a blocking thread, and cutting the runtime
                // out from under it is how you lose a job that was about to
                // finish.
                drop(runtime);
            });
        });

    match handle {
        Ok(handle) => {
            *slot = Some(handle);
            0
        }
        Err(error) => {
            ALIVE.store(false, Ordering::SeqCst);
            tracing::error!(%error, "the mold engine thread could not be spawned");
            1
        }
    }
}

/// Whether the engine thread is still running.
#[no_mangle]
pub extern "C" fn mold_engine_is_alive() -> bool {
    guarded("is_alive", false, || ALIVE.load(Ordering::SeqCst))
}

/// Waits for the engine thread to finish, up to `timeout_ms`.
///
/// Stopping is a `POST /api/shutdown` from the app -- the server's shutdown
/// trigger is not exposed to an embedder any other way -- so this only waits
/// for the thread that request already asked to stop.
#[no_mangle]
pub extern "C" fn mold_engine_join(timeout_ms: u64) -> bool {
    guarded("join", false, || join(Duration::from_millis(timeout_ms)))
}

fn join(timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while ALIVE.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    if ALIVE.load(Ordering::SeqCst) {
        // Still draining. The lease is this process's claim on the gallery,
        // and the app is about to end the process, so hand it back now rather
        // than leave a file that says a server is publishing (review 05-M7).
        mold_server::gallery_authority::release_gallery_writer_leases();
        return false;
    }
    let handle = match engine().lock() {
        Ok(mut slot) => slot.take(),
        Err(_) => None,
    };
    let Some(handle) = handle else { return true };
    // `ALIVE` falls in a Drop guard, which runs BEFORE the thread's own
    // teardown finishes, so this final join is not instantaneous and must
    // carry its own bound -- it is on the quit path, where an unbounded wait
    // hangs the app instead of the budget it promised (review 05-M9).
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    joined_within(handle, remaining.max(JOIN_TAIL))
}

/// The floor for the final `handle.join()`, so a caller whose whole budget has
/// already elapsed still gives the thread a moment to leave.
const JOIN_TAIL: Duration = Duration::from_millis(250);

/// `std::thread::JoinHandle` has no timed join, so the wait happens on a
/// reaper thread and this side waits on a channel.
fn joined_within(handle: std::thread::JoinHandle<()>, timeout: Duration) -> bool {
    let (done, waited) = std::sync::mpsc::channel();
    let reaper = std::thread::Builder::new()
        .name("mold-engine-join".into())
        .spawn(move || {
            let _ = handle.join();
            let _ = done.send(());
        });
    if reaper.is_err() {
        return false;
    }
    waited.recv_timeout(timeout).is_ok()
}

#[cfg(test)]
mod tests {
    use super::{guarded, joined_within, AliveGuard, ALIVE, EMBEDDED_CORS_ORIGIN};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    /// `ALIVE` is process-wide, so the tests that move it take turns.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// **Fails today**: `ALIVE.store(false)` sat at the BOTTOM of the thread
    /// body, which an unwind skips -- `mold_engine_is_alive` then answered
    /// true for the life of the process.
    #[test]
    fn a_panicking_engine_thread_still_reports_dead() {
        let _serial = SERIAL.lock().unwrap_or_else(|poison| poison.into_inner());
        ALIVE.store(true, Ordering::SeqCst);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::thread::spawn(|| {
            let _alive = AliveGuard;
            panic!("the engine fell over");
        })
        .join();
        std::panic::set_hook(previous);
        assert!(outcome.is_err());
        assert!(!ALIVE.load(Ordering::SeqCst));
    }

    /// **Fails today**: a panic crossing an `extern "C"` frame aborts the
    /// process, so an engine that fell over took the app's crash report with
    /// it instead of answering a failure.
    #[test]
    fn a_panic_inside_an_entry_point_becomes_its_failure_value() {
        let _serial = SERIAL.lock().unwrap_or_else(|poison| poison.into_inner());
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let answered = guarded("test", 7, || panic!("boom"));
        std::panic::set_hook(previous);
        assert_eq!(answered, 7);
    }

    /// **Fails today**: the final `handle.join()` carried no bound at all, on
    /// the quit path.
    #[test]
    fn the_final_join_gives_up_rather_than_hanging_the_app() {
        let slow = std::thread::spawn(|| std::thread::sleep(Duration::from_secs(30)));
        let started = Instant::now();
        assert!(!joined_within(slow, Duration::from_millis(120)));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// **Fails today**: nothing set `MOLD_CORS_ORIGIN`, so `build_cors_layer`
    /// took its `CorsLayer::permissive()` arm and any page that found the
    /// port could READ every reply.
    #[test]
    fn the_embedded_cors_origin_is_one_no_page_can_present() {
        // The server parses it into a `HeaderValue` and treats a failure as a
        // startup error, so this is the half that must not regress.
        let parsed = EMBEDDED_CORS_ORIGIN
            .parse::<http::HeaderValue>()
            .expect("the embedded origin must parse, or the engine refuses to start");
        assert_eq!(parsed.as_bytes(), EMBEDDED_CORS_ORIGIN.as_bytes());

        // A serialized origin is `scheme://host[:port]` or `null`. None of
        // those can hold a space, so a browser can never match this.
        assert!(EMBEDDED_CORS_ORIGIN.contains(' '));
        assert_ne!(EMBEDDED_CORS_ORIGIN, "*");
        assert_ne!(EMBEDDED_CORS_ORIGIN, "null");
        assert!(!EMBEDDED_CORS_ORIGIN.contains("://"));
    }
}
