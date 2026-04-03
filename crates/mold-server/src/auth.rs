use axum::{
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::collections::HashSet;
use std::sync::Arc;
use subtle::ConstantTimeEq;
use tracing::warn;

/// Shared auth state — `None` means authentication is disabled.
pub type AuthState = Option<Arc<ApiKeySet>>;

/// Set of valid API keys loaded from `MOLD_API_KEY`.
pub struct ApiKeySet {
    keys: HashSet<String>,
}

impl ApiKeySet {
    pub fn new(keys: HashSet<String>) -> Self {
        Self { keys }
    }

    pub fn contains(&self, candidate: &str) -> bool {
        self.keys
            .iter()
            .any(|k| k.as_bytes().ct_eq(candidate.as_bytes()).into())
    }
}

#[derive(Debug, Serialize)]
struct AuthError {
    error: String,
    code: String,
}

/// Load API keys from the `MOLD_API_KEY` environment variable.
///
/// Formats:
/// - Single key: `MOLD_API_KEY=my-secret`
/// - Comma-separated: `MOLD_API_KEY=key1,key2,key3`
/// - File reference: `MOLD_API_KEY=@/path/to/keys.txt` (one key per line)
///
/// Returns `None` when the variable is unset or empty (auth disabled).
pub fn load_api_keys() -> anyhow::Result<AuthState> {
    let raw = match std::env::var("MOLD_API_KEY") {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(None),
    };

    let keys = if let Some(path) = raw.strip_prefix('@') {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read API key file {path}: {e}"))?;
        contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect::<HashSet<_>>()
    } else {
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect::<HashSet<_>>()
    };

    if keys.is_empty() {
        return Ok(None);
    }

    tracing::info!(num_keys = keys.len(), "API key authentication enabled");
    Ok(Some(Arc::new(ApiKeySet { keys })))
}

/// Paths that are exempt from API key authentication.
const EXEMPT_PATHS: &[&str] = &["/health", "/api/docs", "/api/openapi.json"];

/// Axum middleware that enforces API key authentication.
pub async fn require_api_key(request: Request, next: Next) -> Response {
    // Auth state is stored as an extension by the layer setup in lib.rs.
    let auth_state = request.extensions().get::<AuthState>().cloned();

    let key_set = match auth_state.as_ref().and_then(|s| s.as_ref()) {
        Some(ks) => ks,
        None => return next.run(request).await, // Auth disabled
    };

    // Exempt certain paths (health checks, docs).
    let path = request.uri().path();
    if EXEMPT_PATHS.contains(&path) {
        return next.run(request).await;
    }

    // Check the X-Api-Key header.
    match request.headers().get("x-api-key") {
        Some(value) => {
            let candidate = value.to_str().unwrap_or("");
            if key_set.contains(candidate) {
                next.run(request).await
            } else {
                warn!("rejected request with invalid API key");
                unauthorized("invalid API key")
            }
        }
        None => {
            warn!(path = %path, "rejected request without API key");
            unauthorized("missing X-Api-Key header")
        }
    }
}

fn unauthorized(msg: &str) -> Response {
    let body = AuthError {
        error: msg.to_string(),
        code: "UNAUTHORIZED".to_string(),
    };
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

/// Injects the `AuthState` as a request extension so the middleware can access it.
pub async fn inject_auth_state(
    axum::extract::State(auth): axum::extract::State<AuthState>,
    mut request: Request,
    next: Next,
) -> Response {
    request.extensions_mut().insert(auth);
    next.run(request).await
}

// ── CORS header exposure ────────────────────────────────────────────────────

/// Header name for API key authentication (needed for CORS `Access-Control-Allow-Headers`).
pub fn api_key_header_name() -> HeaderValue {
    HeaderValue::from_static("x-api-key")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize env var mutations across parallel test threads.
    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        &LOCK
    }

    #[test]
    fn parse_single_key() {
        let _lock = env_lock().lock().unwrap();
        unsafe { std::env::set_var("MOLD_API_KEY", "secret123") };
        let state = load_api_keys().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        let ks = state.as_ref().unwrap();
        assert!(ks.contains("secret123"));
        assert!(!ks.contains("wrong"));
    }

    #[test]
    fn parse_comma_separated_keys() {
        let _lock = env_lock().lock().unwrap();
        unsafe { std::env::set_var("MOLD_API_KEY", "key1,key2, key3 ") };
        let state = load_api_keys().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        let ks = state.as_ref().unwrap();
        assert!(ks.contains("key1"));
        assert!(ks.contains("key2"));
        assert!(ks.contains("key3"));
        assert!(!ks.contains("key4"));
    }

    #[test]
    fn parse_file_keys() {
        let _lock = env_lock().lock().unwrap();
        let dir = std::env::temp_dir().join("mold_test_keys");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("keys.txt");
        std::fs::write(&path, "alpha\n# comment\n\nbeta\n").unwrap();
        let env_val = format!("@{}", path.display());
        unsafe { std::env::set_var("MOLD_API_KEY", &env_val) };
        let state = load_api_keys().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        let _ = std::fs::remove_file(&path);
        let ks = state.as_ref().unwrap();
        assert!(ks.contains("alpha"));
        assert!(ks.contains("beta"));
        assert!(!ks.contains("# comment"));
    }

    #[test]
    fn empty_env_returns_none() {
        let _lock = env_lock().lock().unwrap();
        unsafe { std::env::set_var("MOLD_API_KEY", "") };
        let state = load_api_keys().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        assert!(state.is_none());
    }

    #[test]
    fn unset_env_returns_none() {
        let _lock = env_lock().lock().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        let state = load_api_keys().unwrap();
        assert!(state.is_none());
    }

    #[test]
    fn constant_time_comparison_rejects_wrong_key() {
        let ks = ApiKeySet::new(HashSet::from(["correct-key".to_string()]));
        assert!(ks.contains("correct-key"));
        assert!(!ks.contains("wrong-key"));
        assert!(!ks.contains(""));
    }
}
