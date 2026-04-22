//! Test-only helpers for isolating state that lives in env vars and
//! `$MOLD_HOME`.
//!
//! **Usage**: every test that touches `MOLD_HOME`, `MOLD_DB_PATH`, or
//! anything that reads `Config::mold_dir()` transitively must carry
//! `#[serial(mold_env)]` from the `serial_test` crate. That includes
//! all callers of [`with_isolated_env`] plus tests like the thumbnails
//! cache-path check that compute paths via `Config::mold_dir()` even
//! though they don't call the helper themselves. The shared key ensures
//! cargo's parallel runner serialises them against each other.

use std::path::{Path, PathBuf};
use std::sync::Once;

/// Runs before any test body that imports this module. Forces
/// `MOLD_DB_DISABLE=1` at process init so *non-isolated* tests never
/// touch the metadata DB — the source of cross-test races was
/// `make_settings_test_app` (and tests that trigger background
/// `save_session` paths) opening MOLD_DB_PATH while an `#[serial]`
/// test was mid-write against the same file. Isolated tests go through
/// [`with_isolated_env`] below, which removes the var for their body.
static INIT_DISABLE_DB: Once = Once::new();
fn ensure_db_disabled_by_default() {
    INIT_DISABLE_DB.call_once(|| {
        if std::env::var("MOLD_DB_DISABLE").is_err() {
            std::env::set_var("MOLD_DB_DISABLE", "1");
        }
    });
}

/// Process-global lock taken by every test that reads or writes
/// `MOLD_HOME` / `MOLD_DB_PATH` / `MOLD_DB_DISABLE`. Callers:
///
/// - [`with_isolated_env`] (reshapes the env for one test body).
/// - [`enter_test_scope`] (read-only callers like `make_settings_test_app`
///   that just want to snapshot the env under a consistent state).
///
/// The crate's `serial_test::serial(mold_env)` annotations are a
/// belt-and-braces mechanism layered on top — a poisoned lock is fine;
/// we only need exclusive access for the duration of the critical
/// section.
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Grab the shared env lock. Used by helpers that read env vars but
/// don't need to mutate them (e.g. `make_settings_test_app` opening the
/// DB via `PromptHistory::load`). Dropping the returned guard releases.
pub(crate) fn enter_test_scope() -> std::sync::MutexGuard<'static, ()> {
    ENV_MUTEX.lock().unwrap_or_else(|p| p.into_inner())
}

/// Force the metadata DB off for any test that hasn't explicitly opted
/// into isolation via [`with_isolated_env`]. Call this once early in
/// `make_settings_test_app` so a Config-backed `App` constructed in a
/// non-annotated test can't race MOLD_DB_PATH with a parallel
/// `#[serial]` test's write.
pub(crate) fn disable_db_for_non_isolated_tests() {
    ensure_db_disabled_by_default();
}

/// Run `f` with `MOLD_HOME`, `MOLD_DB_PATH`, and `MOLD_DB_DISABLE`
/// scoped to a fresh temp directory. Takes [`ENV_MUTEX`] for the whole
/// body so concurrent non-annotated tests can't observe or mutate the
/// env mid-flight.
pub(crate) fn with_isolated_env<F: FnOnce(&Path)>(f: F) {
    ensure_db_disabled_by_default();
    let guard = ENV_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let tmp = unique_tmp_dir();
    std::fs::create_dir_all(&tmp).unwrap();

    let prev_home = std::env::var("MOLD_HOME").ok();
    let prev_dbpath = std::env::var("MOLD_DB_PATH").ok();
    let prev_disabled = std::env::var("MOLD_DB_DISABLE").ok();
    std::env::set_var("MOLD_HOME", &tmp);
    std::env::set_var("MOLD_DB_PATH", tmp.join("mold.db"));
    std::env::remove_var("MOLD_DB_DISABLE");

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&tmp)));

    match prev_home {
        Some(v) => std::env::set_var("MOLD_HOME", v),
        None => std::env::remove_var("MOLD_HOME"),
    }
    match prev_dbpath {
        Some(v) => std::env::set_var("MOLD_DB_PATH", v),
        None => std::env::remove_var("MOLD_DB_PATH"),
    }
    match prev_disabled {
        Some(v) => std::env::set_var("MOLD_DB_DISABLE", v),
        None => std::env::remove_var("MOLD_DB_DISABLE"),
    }
    let _ = std::fs::remove_dir_all(&tmp);
    drop(guard);
    if let Err(payload) = res {
        std::panic::resume_unwind(payload);
    }
}

fn unique_tmp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "mold-tui-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}
