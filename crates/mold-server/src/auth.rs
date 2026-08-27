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
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;
use tracing::warn;

const GALLERY_MEDIA_TOKEN_CONTEXT: &[u8] = b"mold-gallery-media-v2\nGET\n";
pub(crate) const GALLERY_MEDIA_TOKEN_TTL_SECS: u64 = 15 * 60;
const GALLERY_SIGNING_SECRET_BYTES: usize = 32;
const PAIRING_TOKEN_BYTES: usize = 32;
pub(crate) const PAIRING_TOKEN_TTL_SECS: u64 = 2 * 60;
const MAX_PAIRING_SESSIONS: usize = 32;

type HmacSha256 = Hmac<Sha256>;

/// Shared auth state — `None` means authentication is disabled.
pub type AuthState = Option<Arc<ApiKeySet>>;

/// Set of valid API keys loaded from `MOLD_API_KEY`.
pub struct ApiKeySet {
    operator_keys: HashSet<String>,
    gallery_signing_secret: [u8; GALLERY_SIGNING_SECRET_BYTES],
    pairing_sessions: Mutex<HashMap<[u8; 32], PairingSession>>,
    paired_clients: Mutex<HashMap<[u8; 32], PairedClientAccess>>,
    metadata_db: Arc<Option<mold_db::MetadataDb>>,
    server_instance_id: Arc<String>,
}

struct PairingSession {
    expires_at: u64,
}

#[derive(Clone)]
struct PairedClientAccess {
    id: String,
    name: String,
    client_kind: String,
    created_at_ms: i64,
    last_used_at_ms: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthenticationKind {
    Operator,
    PairedClient,
}

impl ApiKeySet {
    pub fn new(keys: HashSet<String>) -> Self {
        Self::try_new(keys, Arc::new(None), Arc::new(String::new()))
            .expect("OS randomness is required for gallery media authentication")
    }

    fn try_new(
        keys: HashSet<String>,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
        server_instance_id: Arc<String>,
    ) -> Result<Self, getrandom::Error> {
        let mut gallery_signing_secret = [0_u8; GALLERY_SIGNING_SECRET_BYTES];
        getrandom::fill(&mut gallery_signing_secret)?;
        let paired_clients = load_paired_clients(&metadata_db, &server_instance_id);
        Ok(Self {
            operator_keys: keys,
            gallery_signing_secret,
            pairing_sessions: Mutex::new(HashMap::new()),
            paired_clients: Mutex::new(paired_clients),
            metadata_db,
            server_instance_id,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_with_gallery_signing_secret(
        keys: HashSet<String>,
        gallery_signing_secret: [u8; GALLERY_SIGNING_SECRET_BYTES],
    ) -> Self {
        Self {
            operator_keys: keys,
            gallery_signing_secret,
            pairing_sessions: Mutex::new(HashMap::new()),
            paired_clients: Mutex::new(HashMap::new()),
            metadata_db: Arc::new(None),
            server_instance_id: Arc::new(String::new()),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_with_metadata_db(
        keys: HashSet<String>,
        metadata_db: Arc<Option<mold_db::MetadataDb>>,
        server_instance_id: impl Into<String>,
    ) -> Self {
        Self::try_new(keys, metadata_db, Arc::new(server_instance_id.into())).unwrap()
    }

    pub fn contains(&self, candidate: &str) -> bool {
        self.authenticate(candidate).is_some()
    }

    fn authenticate(&self, candidate: &str) -> Option<AuthenticationKind> {
        // Check ALL keys unconditionally to avoid leaking which key matched
        // via timing side-channel (`.any()` would short-circuit on first match).
        let candidate_bytes = candidate.as_bytes();
        let mut found = subtle::Choice::from(0u8);
        for k in &self.operator_keys {
            found |= k.as_bytes().ct_eq(candidate_bytes);
        }
        if bool::from(found) {
            return Some(AuthenticationKind::Operator);
        }

        let candidate_hash: [u8; 32] = Sha256::digest(candidate_bytes).into();
        let now_ms = unix_timestamp_ms();
        let mut matched = None;
        let mut clients = self
            .paired_clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (hash, client) in clients.iter_mut() {
            if bool::from(hash.ct_eq(&candidate_hash)) {
                matched = Some(client);
            }
        }
        let client = matched?;
        if client
            .last_used_at_ms
            .is_none_or(|last_used| now_ms.saturating_sub(last_used) >= 60_000)
        {
            client.last_used_at_ms = Some(now_ms);
            if let Some(db) = self.metadata_db.as_ref() {
                if let Err(error) = mold_db::paired_clients::PairedClients::new(db).touch(
                    &self.server_instance_id,
                    &client.id,
                    now_ms,
                ) {
                    warn!(error = %format!("{error:#}"), client_id = %client.id, "failed to persist paired client activity");
                }
            }
        }
        Some(AuthenticationKind::PairedClient)
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

    fn durable_identity(candidate: &str) -> String {
        let mut digest = Sha256::new();
        digest.update(b"mold-api-key-durable-subject-v1\0");
        digest.update((candidate.len() as u64).to_le_bytes());
        digest.update(candidate.as_bytes());
        format!("{:x}", digest.finalize())
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

    /// Create a one-time, short-lived handoff authorizing a new paired grant.
    /// Only the HMAC of the random token is retained; the bearer value exists
    /// solely in the no-store response and QR code.
    pub(crate) fn issue_pairing_token(&self) -> Result<(String, u64), getrandom::Error> {
        let mut token_bytes = [0_u8; PAIRING_TOKEN_BYTES];
        getrandom::fill(&mut token_bytes)?;
        let token = URL_SAFE_NO_PAD.encode(token_bytes);
        let token_hash = self.pairing_token_hash(&token);
        let now = unix_timestamp();
        let expires_at = now.saturating_add(PAIRING_TOKEN_TTL_SECS);
        let mut sessions = self
            .pairing_sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sessions.retain(|_, session| session.expires_at > now);
        if sessions.len() >= MAX_PAIRING_SESSIONS {
            if let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, session)| session.expires_at)
                .map(|(hash, _)| *hash)
            {
                sessions.remove(&oldest);
            }
        }
        sessions.insert(token_hash, PairingSession { expires_at });
        Ok((token, expires_at))
    }

    /// Consume a pairing token exactly once and mint a distinct durable grant.
    /// The operator credential that opened Settings is never copied to the
    /// paired client, so this grant can be revoked independently.
    pub(crate) fn claim_pairing_token(
        &self,
        token: &str,
        client_name: &str,
        client_kind: &str,
    ) -> anyhow::Result<Option<String>> {
        let token_hash = self.pairing_token_hash(token);
        let now = unix_timestamp();
        let mut sessions = self
            .pairing_sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sessions.retain(|_, session| session.expires_at > now);
        let session = sessions
            .remove(&token_hash)
            .filter(|session| session.expires_at > now);
        drop(sessions);
        if session.is_none() {
            return Ok(None);
        }
        let Some(db) = self.metadata_db.as_ref() else {
            anyhow::bail!("paired access requires the Mold metadata database");
        };

        let mut key_bytes = [0_u8; 32];
        getrandom::fill(&mut key_bytes)?;
        let credential = format!("mold_pair_{}", URL_SAFE_NO_PAD.encode(key_bytes));
        let credential_hash: [u8; 32] = Sha256::digest(credential.as_bytes()).into();
        let id = format!("pair_{}", URL_SAFE_NO_PAD.encode(&credential_hash[..12]));
        let created_at_ms = unix_timestamp_ms();
        let access = PairedClientAccess {
            id: id.clone(),
            name: normalized_client_name(client_name),
            client_kind: normalized_client_kind(client_kind),
            created_at_ms,
            last_used_at_ms: None,
        };
        mold_db::paired_clients::PairedClients::new(db).insert(
            &mold_db::paired_clients::PairedClient {
                id,
                server_instance_id: self.server_instance_id.as_ref().clone(),
                name: access.name.clone(),
                client_kind: access.client_kind.clone(),
                credential_hash,
                created_at_ms,
                last_used_at_ms: None,
            },
        )?;
        self.paired_clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(credential_hash, access);
        Ok(Some(credential))
    }

    pub(crate) fn pairing_available(&self) -> bool {
        self.metadata_db.is_some()
    }

    pub(crate) fn paired_clients(&self) -> Vec<PairedClientSummary> {
        let mut clients = self
            .paired_clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .cloned()
            .map(|client| PairedClientSummary {
                id: client.id,
                name: client.name,
                client_kind: client.client_kind,
                created_at_ms: client.created_at_ms,
                last_used_at_ms: client.last_used_at_ms,
            })
            .collect::<Vec<_>>();
        clients.sort_by_key(|client| std::cmp::Reverse(client.created_at_ms));
        clients
    }

    pub(crate) fn revoke_paired_client(&self, id: &str) -> anyhow::Result<bool> {
        let Some(db) = self.metadata_db.as_ref() else {
            anyhow::bail!("paired access requires the Mold metadata database");
        };
        let revoked =
            mold_db::paired_clients::PairedClients::new(db).revoke(&self.server_instance_id, id)?;
        if revoked {
            self.paired_clients
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .retain(|_, client| client.id != id);
        }
        Ok(revoked)
    }

    fn pairing_token_hash(&self, token: &str) -> [u8; 32] {
        // Reusing the process-random gallery secret is intentional: the
        // versioned context below gives pairing an independent HMAC domain.
        let mut mac = HmacSha256::new_from_slice(&self.gallery_signing_secret)
            .expect("HMAC-SHA256 accepts keys of any length");
        mac.update(b"mold-mobile-pairing-v1\n");
        mac.update(token.as_bytes());
        mac.finalize().into_bytes().into()
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

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PairedClientSummary {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) client_kind: String,
    pub(crate) created_at_ms: i64,
    pub(crate) last_used_at_ms: Option<i64>,
}

fn load_paired_clients(
    metadata_db: &Arc<Option<mold_db::MetadataDb>>,
    server_instance_id: &str,
) -> HashMap<[u8; 32], PairedClientAccess> {
    let Some(db) = metadata_db.as_ref() else {
        return HashMap::new();
    };
    match mold_db::paired_clients::PairedClients::new(db).list(server_instance_id) {
        Ok(clients) => clients
            .into_iter()
            .map(|client| {
                (
                    client.credential_hash,
                    PairedClientAccess {
                        id: client.id,
                        name: client.name,
                        client_kind: client.client_kind,
                        created_at_ms: client.created_at_ms,
                        last_used_at_ms: client.last_used_at_ms,
                    },
                )
            })
            .collect(),
        Err(error) => {
            warn!(error = %format!("{error:#}"), "failed to load paired clients");
            HashMap::new()
        }
    }
}

fn normalized_client_name(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        "Mold mobile".to_string()
    } else {
        value.chars().take(80).collect()
    }
}

fn normalized_client_kind(value: &str) -> String {
    match value.trim() {
        "iphone" => "iphone",
        "ipad" => "ipad",
        "android" => "android",
        _ => "mobile",
    }
    .to_string()
}

/// Marker proving normal API-key authentication succeeded for this request.
/// The matched API key itself deliberately never leaves the middleware.
#[derive(Clone)]
pub(crate) struct ApiKeyAuthenticated {
    /// Process-local opaque audit label. It is derived with the server's
    /// random signing secret, so neither the API key nor an offline-comparable
    /// digest enters logs.
    pub(crate) identity: String,
    /// Restart-stable credential subject used only inside encrypted durable
    /// admission authority. It is never logged or returned on the wire.
    #[cfg_attr(not(any(feature = "h3", feature = "h3-private-uat")), allow(dead_code))]
    #[cfg_attr(feature = "h3", allow(dead_code))]
    pub(crate) durable_identity: String,
}

/// Present only when the request used an operator-configured credential.
/// Paired credentials deliberately cannot mint or manage other grants.
#[derive(Clone)]
pub(crate) struct PairingAuthority;

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
    load_api_keys_with_db(Arc::new(None), Arc::new(String::new()))
}

pub(crate) fn load_api_keys_with_db(
    metadata_db: Arc<Option<mold_db::MetadataDb>>,
    server_instance_id: Arc<String>,
) -> anyhow::Result<AuthState> {
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
    let key_set = ApiKeySet::try_new(keys, metadata_db, server_instance_id)
        .map_err(|error| anyhow::anyhow!("failed to generate gallery signing secret: {error}"))?;
    Ok(Some(Arc::new(key_set)))
}

/// Paths that are exempt from API key authentication.
const EXEMPT_PATHS: &[&str] = &[
    "/health",
    "/api/docs",
    "/api/openapi.json",
    "/api/pairing/claim",
];

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
    // for GET/HEAD reads of an explicitly supported media route; all other
    // paths and methods continue through normal API-key authentication below.
    if is_ticketed_media_read(&request) {
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
            let candidate = value.to_str().unwrap_or("").to_string();
            if let Some(kind) = key_set.authenticate(&candidate) {
                let identity = key_set.audit_identity(&candidate);
                let durable_identity = ApiKeySet::durable_identity(&candidate);
                request.extensions_mut().insert(ApiKeyAuthenticated {
                    identity,
                    durable_identity,
                });
                if kind == AuthenticationKind::Operator {
                    request.extensions_mut().insert(PairingAuthority);
                }
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

fn is_ticketed_media_read(request: &Request) -> bool {
    if request.method() != Method::GET && request.method() != Method::HEAD {
        return false;
    }

    is_ticketable_media_path(request.uri().path())
}

pub(crate) fn is_gallery_image_path(path: &str) -> bool {
    if path.contains('?') || path.contains('#') {
        return false;
    }

    path.strip_prefix("/api/gallery/image/")
        .is_some_and(|filename| !filename.is_empty() && !filename.contains('/'))
}

pub(crate) fn is_chain_stage_media_path(path: &str) -> bool {
    if path.contains('?') || path.contains('#') {
        return false;
    }
    let Some(rest) = path.strip_prefix("/api/chain-jobs/") else {
        return false;
    };
    let mut parts = rest.split('/');
    matches!(
        (
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next(),
            parts.next()
        ),
        (Some(id), Some("stages"), Some(idx), Some("media"), None)
            if !id.is_empty()
                && !id.contains(['\\', '.'])
                && idx.parse::<u32>().is_ok()
    )
}

pub(crate) fn is_ticketable_media_path(path: &str) -> bool {
    is_gallery_image_path(path) || is_chain_stage_media_path(path)
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

fn unix_timestamp_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
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
    use crate::test_support::env_lock;
    use tower::ServiceExt;

    #[test]
    fn parse_single_key() {
        let _lock = env_lock();
        unsafe { std::env::set_var("MOLD_API_KEY", "secret123") };
        let state = load_api_keys().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        let ks = state.as_ref().unwrap();
        assert!(ks.contains("secret123"));
        assert!(!ks.contains("wrong"));
    }

    #[test]
    fn parse_comma_separated_keys() {
        let _lock = env_lock();
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
    fn chain_stage_media_path_is_exact_and_traversal_safe() {
        assert!(is_chain_stage_media_path(
            "/api/chain-jobs/01JBR55-TEST/stages/0/media"
        ));
        assert!(is_chain_stage_media_path(
            "/api/chain-jobs/01JBR55-TEST/stages/4294967295/media"
        ));
        for invalid in [
            "/api/chain-jobs/01JBR55-TEST/stages/0/preview",
            "/api/chain-jobs/01JBR55-TEST/stages/not-a-number/media",
            "/api/chain-jobs/../stages/0/media",
            "/api/chain-jobs/01JBR55-TEST/stages/0/media/extra",
            "/api/chain-jobs/01JBR55-TEST/stages/0/media?media_token=x",
        ] {
            assert!(!is_chain_stage_media_path(invalid), "{invalid}");
        }
    }

    #[test]
    fn parse_file_keys() {
        let _lock = env_lock();
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
        let _lock = env_lock();
        unsafe { std::env::set_var("MOLD_API_KEY", "") };
        let state = load_api_keys().unwrap();
        unsafe { std::env::remove_var("MOLD_API_KEY") };
        assert!(state.is_none());
    }

    #[test]
    fn unset_env_returns_none() {
        let _lock = env_lock();
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
    fn stage_media_token_is_bound_to_one_exact_stage_path() {
        const MEDIA_PATH: &str = "/api/chain-jobs/01JBR55-TEST/stages/2/media";
        let ks = ApiKeySet::new_with_gallery_signing_secret(
            HashSet::from(["correct-key".to_string()]),
            [0x42; GALLERY_SIGNING_SECRET_BYTES],
        );
        let token = ks.sign_gallery_media_token_for_tests(MEDIA_PATH, 1_900);

        assert!(ks.validates_gallery_media_token(&token, MEDIA_PATH, 1_900, 1_000));
        assert!(!ks.validates_gallery_media_token(
            &token,
            "/api/chain-jobs/01JBR55-TEST/stages/3/media",
            1_900,
            1_000,
        ));
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

    #[test]
    fn pairing_token_is_random_url_safe_and_single_use() {
        let ks = ApiKeySet::new_with_metadata_db(
            HashSet::from(["phone-key".to_string()]),
            Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap())),
            "server-a",
        );
        let (token, expires_at) = ks.issue_pairing_token().unwrap();

        assert_eq!(token.len(), 43);
        assert!(token
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'));
        assert!(expires_at > unix_timestamp());
        let paired = ks
            .claim_pairing_token(&token, "James's iPhone", "iphone")
            .unwrap()
            .unwrap();
        assert!(paired.starts_with("mold_pair_"));
        assert_ne!(paired, "phone-key");
        assert!(ks.contains(&paired));
        assert_eq!(ks.paired_clients()[0].name, "James's iPhone");
        assert_eq!(
            ks.claim_pairing_token(&token, "Replay", "iphone").unwrap(),
            None
        );
        assert_eq!(
            ks.claim_pairing_token("not-the-token", "Unknown", "mobile")
                .unwrap(),
            None
        );
        let id = ks.paired_clients()[0].id.clone();
        assert!(ks.revoke_paired_client(&id).unwrap());
        assert!(!ks.contains(&paired));
        assert!(ks.contains("phone-key"));
    }

    #[test]
    fn pairing_preserves_android_client_identity() {
        let ks = ApiKeySet::new_with_metadata_db(
            HashSet::from(["operator-key".to_string()]),
            Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap())),
            "server-a",
        );
        let (token, _) = ks.issue_pairing_token().unwrap();

        ks.claim_pairing_token(&token, "Mold on Android", "android")
            .unwrap()
            .unwrap();

        let client = &ks.paired_clients()[0];
        assert_eq!(client.name, "Mold on Android");
        assert_eq!(client.client_kind, "android");
    }

    #[test]
    fn paired_credentials_are_scoped_to_one_server_instance() {
        let metadata_db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let operators = HashSet::from(["operator-key".to_string()]);
        let server_a =
            ApiKeySet::new_with_metadata_db(operators.clone(), metadata_db.clone(), "server-a");
        let (token, _) = server_a.issue_pairing_token().unwrap();
        let paired = server_a
            .claim_pairing_token(&token, "James's iPhone", "iphone")
            .unwrap()
            .unwrap();

        let server_b =
            ApiKeySet::new_with_metadata_db(operators.clone(), metadata_db.clone(), "server-b");
        let restarted_a =
            ApiKeySet::new_with_metadata_db(operators, metadata_db.clone(), "server-a");
        assert!(!server_b.contains(&paired));
        assert!(restarted_a.contains(&paired));

        let client_id = restarted_a.paired_clients()[0].id.clone();
        assert!(restarted_a.revoke_paired_client(&client_id).unwrap());
        let restarted_after_revoke = ApiKeySet::new_with_metadata_db(
            HashSet::from(["operator-key".to_string()]),
            metadata_db,
            "server-a",
        );
        assert!(!restarted_after_revoke.contains(&paired));
    }

    #[test]
    fn expired_pairing_token_is_rejected_and_removed() {
        let ks = ApiKeySet::new_with_metadata_db(
            HashSet::from(["phone-key".to_string()]),
            Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap())),
            "server-a",
        );
        let (token, _) = ks.issue_pairing_token().unwrap();
        let token_hash = ks.pairing_token_hash(&token);
        ks.pairing_sessions
            .lock()
            .unwrap()
            .get_mut(&token_hash)
            .unwrap()
            .expires_at = unix_timestamp();

        assert_eq!(
            ks.claim_pairing_token(&token, "Expired", "mobile").unwrap(),
            None
        );
        assert!(!ks
            .pairing_sessions
            .lock()
            .unwrap()
            .contains_key(&token_hash));
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

    fn pairing_test_app(auth_state: AuthState) -> axum::Router {
        axum::Router::new()
            .route(
                "/api/pairing/sessions",
                axum::routing::post(
                    |authority: Option<axum::extract::Extension<PairingAuthority>>| async move {
                        if authority.is_some() {
                            StatusCode::OK
                        } else {
                            StatusCode::FORBIDDEN
                        }
                    },
                ),
            )
            .route(
                "/api/pairing/claim",
                axum::routing::post(|| async { StatusCode::OK }),
            )
            .layer(axum::middleware::from_fn(require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                auth_state,
                inject_auth_state,
            ))
    }

    #[tokio::test]
    async fn pairing_creation_requires_auth_but_claim_uses_the_one_time_token() {
        let auth = Some(Arc::new(ApiKeySet::new(HashSet::from([
            "correct-key".to_string()
        ]))));
        let missing = pairing_test_app(auth.clone())
            .oneshot(
                Request::post("/api/pairing/sessions")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);

        let created = pairing_test_app(auth.clone())
            .oneshot(
                Request::post("/api/pairing/sessions")
                    .header("x-api-key", "correct-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);

        let claimed = pairing_test_app(auth)
            .oneshot(
                Request::post("/api/pairing/claim")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(claimed.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn paired_credentials_cannot_mint_more_credentials() {
        let key_set = Arc::new(ApiKeySet::new_with_metadata_db(
            HashSet::from(["operator-key".to_string()]),
            Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap())),
            "server-a",
        ));
        let (token, _) = key_set.issue_pairing_token().unwrap();
        let paired = key_set
            .claim_pairing_token(&token, "Mold on iPhone", "iphone")
            .unwrap()
            .unwrap();
        let auth = Some(key_set);

        let denied = pairing_test_app(auth)
            .oneshot(
                Request::post("/api/pairing/sessions")
                    .header("x-api-key", paired)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
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
