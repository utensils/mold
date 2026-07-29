use axum::{
    extract::Request,
    http::{HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::Sha256;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;
use tracing::warn;

const GALLERY_MEDIA_TOKEN_CONTEXT: &[u8] = b"mold-gallery-media-v2\nGET\n";
pub(crate) const GALLERY_MEDIA_TOKEN_TTL_SECS: u64 = 15 * 60;
const GALLERY_SIGNING_SECRET_BYTES: usize = 32;

type HmacSha256 = Hmac<Sha256>;

/// Shared auth state — `None` means authentication is disabled.
pub type AuthState = Option<Arc<ApiKeySet>>;

/// Set of valid API keys loaded from `MOLD_API_KEY`.
pub struct ApiKeySet {
    keys: HashSet<String>,
    gallery_signing_secret: [u8; GALLERY_SIGNING_SECRET_BYTES],
}

impl ApiKeySet {
    pub fn new(keys: HashSet<String>) -> Self {
        Self::try_new(keys).expect("OS randomness is required for gallery media authentication")
    }

    fn try_new(keys: HashSet<String>) -> Result<Self, getrandom::Error> {
        let mut gallery_signing_secret = [0_u8; GALLERY_SIGNING_SECRET_BYTES];
        getrandom::fill(&mut gallery_signing_secret)?;
        Ok(Self {
            keys,
            gallery_signing_secret,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_with_gallery_signing_secret(
        keys: HashSet<String>,
        gallery_signing_secret: [u8; GALLERY_SIGNING_SECRET_BYTES],
    ) -> Self {
        Self {
            keys,
            gallery_signing_secret,
        }
    }

    pub fn contains(&self, candidate: &str) -> bool {
        // Check ALL keys unconditionally to avoid leaking which key matched
        // via timing side-channel (`.any()` would short-circuit on first match).
        let candidate_bytes = candidate.as_bytes();
        let mut found = subtle::Choice::from(0u8);
        for k in &self.keys {
            found |= k.as_bytes().ct_eq(candidate_bytes);
        }
        found.into()
    }

    fn audit_identity(&self, candidate: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(&self.gallery_signing_secret)
            .expect("HMAC-SHA256 accepts keys of any length");
        mac.update(b"mold-api-key-audit-v1\n");
        mac.update(candidate.as_bytes());
        let digest = mac.finalize().into_bytes();
        let mut identity = String::from("key:");
        for byte in &digest[..8] {
            use std::fmt::Write as _;
            write!(&mut identity, "{byte:02x}").expect("writing to String cannot fail");
        }
        identity
    }

    fn validates_gallery_media_token(
        &self,
        token: &str,
        media_path: &str,
        expires_at: u64,
        now: u64,
    ) -> bool {
        if now >= expires_at {
            return false;
        }

        let Ok(signature) = URL_SAFE_NO_PAD.decode(token) else {
            return false;
        };
        if signature.len() != 32 {
            return false;
        }

        let mac = gallery_media_mac(&self.gallery_signing_secret, media_path, expires_at);
        mac.verify_slice(&signature).is_ok()
    }

    pub(crate) fn issue_gallery_media_token(&self, media_path: &str) -> (String, u64) {
        let expires_at = unix_timestamp().saturating_add(GALLERY_MEDIA_TOKEN_TTL_SECS);
        (
            sign_gallery_media_token(&self.gallery_signing_secret, media_path, expires_at),
            expires_at,
        )
    }

    #[cfg(test)]
    pub(crate) fn sign_gallery_media_token_for_tests(
        &self,
        media_path: &str,
        expires_at: u64,
    ) -> String {
        sign_gallery_media_token(&self.gallery_signing_secret, media_path, expires_at)
    }
}

/// Marker proving normal API-key authentication succeeded for this request.
/// The matched API key itself deliberately never leaves the middleware.
#[derive(Clone)]
pub(crate) struct ApiKeyAuthenticated {
    /// Process-local opaque audit label. It is derived with the server's
    /// random signing secret, so neither the API key nor an offline-comparable
    /// digest enters logs.
    pub(crate) identity: String,
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
    let key_set = ApiKeySet::try_new(keys)
        .map_err(|error| anyhow::anyhow!("failed to generate gallery signing secret: {error}"))?;
    Ok(Some(Arc::new(key_set)))
}

/// Paths that are exempt from API key authentication.
const EXEMPT_PATHS: &[&str] = &["/health", "/api/docs", "/api/openapi.json"];

/// Axum middleware that enforces API key authentication.
pub async fn require_api_key(request: Request, next: Next) -> Response {
    let mut request = request;

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

    // Browser media elements cannot attach an X-Api-Key header to their own
    // streaming and Range requests. Accept a short-lived signed ticket only
    // for GET/HEAD reads of the full-size gallery media route; all other paths and
    // methods continue through normal API-key authentication below.
    if is_gallery_image_read(&request) {
        if let Some((token, expires_at)) = gallery_media_ticket_query(&request) {
            let now = unix_timestamp();
            if key_set.validates_gallery_media_token(token, path, expires_at, now) {
                return next.run(request).await;
            }
            if request.headers().get("x-api-key").is_none() {
                warn!(path = %path, "rejected request with invalid or expired gallery media token");
                return unauthorized("invalid or expired gallery media token");
            }
        }
    }

    // Check the X-Api-Key header.
    match request.headers().get("x-api-key") {
        Some(value) => {
            let candidate = value.to_str().unwrap_or("");
            if key_set.contains(candidate) {
                let identity = key_set.audit_identity(candidate);
                request
                    .extensions_mut()
                    .insert(ApiKeyAuthenticated { identity });
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

fn is_gallery_image_read(request: &Request) -> bool {
    if request.method() != Method::GET && request.method() != Method::HEAD {
        return false;
    }

    is_gallery_image_path(request.uri().path())
}

pub(crate) fn is_gallery_image_path(path: &str) -> bool {
    if path.contains('?') || path.contains('#') {
        return false;
    }

    path.strip_prefix("/api/gallery/image/")
        .is_some_and(|filename| !filename.is_empty() && !filename.contains('/'))
}

fn gallery_media_ticket_query(request: &Request) -> Option<(&str, u64)> {
    let mut token = None;
    let mut expires_at = None;

    for pair in request.uri().query()?.split('&') {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        match name {
            "media_token" if token.is_none() => token = Some(value),
            "expires" if expires_at.is_none() => expires_at = value.parse::<u64>().ok(),
            // Reject duplicate credential fields instead of letting a proxy
            // and the application disagree about which one is authoritative.
            "media_token" | "expires" => return None,
            _ => {}
        }
    }

    Some((token?, expires_at?))
}

fn gallery_media_mac(signing_secret: &[u8], media_path: &str, expires_at: u64) -> HmacSha256 {
    let mut mac =
        HmacSha256::new_from_slice(signing_secret).expect("HMAC-SHA256 accepts keys of any length");
    mac.update(GALLERY_MEDIA_TOKEN_CONTEXT);
    mac.update(media_path.as_bytes());
    mac.update(b"\n");
    mac.update(expires_at.to_string().as_bytes());
    mac
}

fn sign_gallery_media_token(signing_secret: &[u8], media_path: &str, expires_at: u64) -> String {
    let signature = gallery_media_mac(signing_secret, media_path, expires_at)
        .finalize()
        .into_bytes();
    URL_SAFE_NO_PAD.encode(signature)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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
    use tower::ServiceExt;

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

    #[test]
    fn gallery_media_token_is_url_safe_and_valid_until_expiry() {
        const MEDIA_PATH: &str = "/api/gallery/image/clip.mp4";
        let ks = ApiKeySet::new_with_gallery_signing_secret(
            HashSet::from(["correct-key".to_string()]),
            [0x42; GALLERY_SIGNING_SECRET_BYTES],
        );
        let token = ks.sign_gallery_media_token_for_tests(MEDIA_PATH, 1_900);

        assert_eq!(token.len(), 43);
        assert!(token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'));
        assert!(ks.validates_gallery_media_token(&token, MEDIA_PATH, 1_900, 1_000));
        assert!(!ks.validates_gallery_media_token(&token, MEDIA_PATH, 1_900, 1_900));
    }

    #[test]
    fn gallery_media_token_rejects_tampering_and_other_signing_secrets() {
        const MEDIA_PATH: &str = "/api/gallery/image/clip.mp4";
        let ks = ApiKeySet::new_with_gallery_signing_secret(
            HashSet::from(["correct-key".to_string()]),
            [0x42; GALLERY_SIGNING_SECRET_BYTES],
        );
        let token = ks.sign_gallery_media_token_for_tests(MEDIA_PATH, 1_900);
        let mut tampered = token.clone();
        tampered.replace_range(..1, if token.starts_with('A') { "B" } else { "A" });

        assert!(!ks.validates_gallery_media_token(&tampered, MEDIA_PATH, 1_900, 1_000));
        assert!(!ks.validates_gallery_media_token(
            &sign_gallery_media_token(&[0x24; GALLERY_SIGNING_SECRET_BYTES], MEDIA_PATH, 1_900,),
            MEDIA_PATH,
            1_900,
            1_000,
        ));
        assert!(!ks.validates_gallery_media_token("not+url/safe", MEDIA_PATH, 1_900, 1_000,));
        assert!(!ks.validates_gallery_media_token(
            &token,
            "/api/gallery/image/other.mp4",
            1_900,
            1_000,
        ));
    }

    #[test]
    fn gallery_media_token_cannot_be_reproduced_from_a_weak_api_key() {
        const WEAK_API_KEY: &str = "password";
        const MEDIA_PATH: &str = "/api/gallery/image/clip.mp4";
        let ks = ApiKeySet::new_with_gallery_signing_secret(
            HashSet::from([WEAK_API_KEY.to_string()]),
            [0x42; GALLERY_SIGNING_SECRET_BYTES],
        );
        let token = ks.sign_gallery_media_token_for_tests(MEDIA_PATH, 1_900);
        let api_key_derived = sign_gallery_media_token(WEAK_API_KEY.as_bytes(), MEDIA_PATH, 1_900);
        let other_process =
            sign_gallery_media_token(&[0x24; GALLERY_SIGNING_SECRET_BYTES], MEDIA_PATH, 1_900);

        assert_ne!(token, api_key_derived);
        assert_ne!(token, other_process);
        assert!(!ks.validates_gallery_media_token(&api_key_derived, MEDIA_PATH, 1_900, 1_000,));
        assert!(!ks.validates_gallery_media_token(&other_process, MEDIA_PATH, 1_900, 1_000,));
        assert!(ks.contains(WEAK_API_KEY));
    }

    fn protected_test_app(auth_state: AuthState) -> axum::Router {
        axum::Router::new()
            .route(
                "/api/devices/:id",
                axum::routing::patch(|| async { axum::http::StatusCode::OK }),
            )
            .layer(axum::middleware::from_fn(require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                auth_state,
                inject_auth_state,
            ))
    }

    #[tokio::test]
    async fn configured_key_is_required_for_device_patch_even_with_loopback_connect_info() {
        let auth = Some(Arc::new(ApiKeySet::new(HashSet::from([
            "correct-key".to_string()
        ]))));
        let mut request = Request::patch("/api/devices/cuda:test")
            .body(axum::body::Body::empty())
            .unwrap();
        request.extensions_mut().insert(axum::extract::ConnectInfo(
            "127.0.0.1:12345".parse::<std::net::SocketAddr>().unwrap(),
        ));

        let missing = protected_test_app(auth.clone())
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(missing.status(), axum::http::StatusCode::UNAUTHORIZED);

        let accepted = protected_test_app(auth)
            .oneshot(
                Request::patch("/api/devices/cuda:test")
                    .header("x-api-key", "correct-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn device_patch_is_open_when_server_auth_is_disabled() {
        let response = protected_test_app(None)
            .oneshot(
                Request::patch("/api/devices/cuda:test")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }
}
