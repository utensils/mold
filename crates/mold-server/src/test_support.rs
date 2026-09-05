//! Lightweight in-process test client for catalog route integration tests.
//! Avoids the full hyper boot — uses `tower::ServiceExt::oneshot` directly.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// Take the one process-wide environment lock.
///
/// Every test that reads or writes a `MOLD_*`-style env var goes through
/// this guard — readers too, because `Config::resolved_models_dir` and
/// friends let the env var beat the struct field. One domain means one
/// panicking holder poisons the lock for every other test in the binary,
/// so the guard is poison-tolerant here, once: the env table is exactly as
/// consistent after a panic as before it (each holder restores what it set
/// on drop), and cascading a single flake into every env test is what the
/// per-caller `unwrap()` did.
#[cfg(test)]
pub fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Env vars that redirect the model store / mold home away from a test's own
/// tempdir. `Config::resolved_models_dir` lets `MOLD_MODELS_DIR` beat the
/// struct field by design, so a developer's direnv
/// (`MOLD_MODELS_DIR=/storage-fast/mold/models`) silently points a test's
/// `Ltx25ModelPaths::resolve` / catalog scan at the PRODUCTION store — and a
/// fixture writer then overwrites real weights: on 2026-08-28 the
/// split-audio fixture in `model_manager.rs` overwrote the real
/// 364,866,540-byte LTX-2.5 audio VAE with its 194-byte stub. Twice.
#[cfg(test)]
const MOLD_STORE_ENV_VARS: &[&str] = &["MOLD_MODELS_DIR", "MOLD_HOME", "MOLD_OUTPUT_DIR"];

/// Holds [`env_lock`] with every store-redirecting `MOLD_*` var removed, and
/// restores the saved values on drop (panic included). Any test that builds
/// a `Config` and then resolves model paths through it — read OR write —
/// must hold this; a bare `env_lock()` only serializes the damage.
#[cfg(test)]
pub struct HermeticStoreEnv {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

#[cfg(test)]
pub fn hermetic_store_env() -> HermeticStoreEnv {
    let lock = env_lock();
    let saved = MOLD_STORE_ENV_VARS
        .iter()
        .map(|name| {
            let previous = std::env::var_os(name);
            std::env::remove_var(name);
            (*name, previous)
        })
        .collect();
    HermeticStoreEnv { _lock: lock, saved }
}

#[cfg(test)]
impl Drop for HermeticStoreEnv {
    fn drop(&mut self) {
        for (name, previous) in self.saved.drain(..) {
            match previous {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

pub struct TestResponse {
    pub status: StatusCode,
    pub body: String,
}

pub struct TestApp {
    router: axum::Router,
}

impl TestApp {
    /// Build an empty AppState (catalog endpoints proxy live HF/Civitai;
    /// tests that exercise live behaviour point catalog_live_civitai_base
    /// at a wiremock instance via `with_civitai_base`).
    pub async fn with_seeded_catalog() -> Self {
        let state = crate::state::AppState::for_tests();
        let router = crate::routes::create_router(state);
        Self { router }
    }

    pub async fn with_civitai_base(base: impl Into<String>) -> Self {
        let state = crate::state::AppState::for_tests().with_civitai_base(base);
        let router = crate::routes::create_router(state);
        Self { router }
    }

    pub async fn get(&self, uri: &str) -> TestResponse {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        TestResponse { status, body }
    }

    pub async fn post_json(&self, uri: &str, body: &str) -> TestResponse {
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = self.router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        TestResponse { status, body }
    }
}
