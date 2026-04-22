//! Test-only helpers for isolating state that lives in env vars and
//! `$MOLD_HOME`. All tests that touch `TuiSession`, `PromptHistory`, or
//! `apply_theme_preset` share the same lock so cargo's parallel test
//! runner can't interleave two tests onto the same temp directory.

use std::path::{Path, PathBuf};

/// Single crate-wide lock for tests that mutate `MOLD_HOME` /
/// `MOLD_DB_PATH` / `MOLD_DB_DISABLE`. A poisoned lock is fine — we just
/// want exclusive access during the body of each test.
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` with `MOLD_HOME`, `MOLD_DB_PATH`, and `MOLD_DB_DISABLE` scoped
/// to a fresh temp directory. Takes [`ENV_LOCK`] for the duration so
/// concurrent tests can't step on each other. Restores the original env
/// + removes the temp dir on exit, even on panic.
pub(crate) fn with_isolated_env<F: FnOnce(&Path)>(f: F) {
    let guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tmp = unique_tmp_dir();
    let _ = std::fs::remove_dir_all(&tmp);
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
    if let Some(v) = prev_disabled {
        std::env::set_var("MOLD_DB_DISABLE", v);
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
