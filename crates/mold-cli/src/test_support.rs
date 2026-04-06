/// Mutex for serializing tests that mutate environment variables.
///
/// Tests that set `MOLD_HOME`, `MOLD_MODELS_DIR`, or other env vars must
/// hold this lock to prevent concurrent tests from seeing stale values.
/// Without this, tests pass in CI (clean env) but fail on developer machines
/// that have real models installed.
///
/// # Usage
/// ```ignore
/// use crate::test_support::ENV_LOCK;
/// let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
/// std::env::set_var("MOLD_MODELS_DIR", &tmp);
/// // ... test logic ...
/// std::env::remove_var("MOLD_MODELS_DIR");
/// ```
pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
