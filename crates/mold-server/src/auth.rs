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
type HmacSha256 = Hmac<Sha256>;

/// Shared auth state — `None` means authentication is disabled.
pub type AuthState = Option<Arc<ApiKeySet>>;

/// Set of valid API keys loaded from `MOLD_API_KEY`.
pub struct ApiKeySet {
    pub(crate) secure: Option<Arc<crate::secure_pairing::SecurePairing>>,
    secure_mutations: Mutex<()>,
    operator_keys: HashSet<String>,
    gallery_signing_secret: [u8; GALLERY_SIGNING_SECRET_BYTES],
    paired_clients: Mutex<HashMap<[u8; 32], PairedClientAccess>>,
    metadata_db: Arc<Option<mold_db::MetadataDb>>,
    server_instance_id: Arc<String>,
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
            secure: None,
            secure_mutations: Mutex::new(()),
            operator_keys: keys,
            gallery_signing_secret,
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
            secure: None,
            secure_mutations: Mutex::new(()),
            operator_keys: keys,
            gallery_signing_secret,
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
        let _mutation = self
            .secure_mutations
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let revoked =
            mold_db::paired_clients::PairedClients::new(db).revoke(&self.server_instance_id, id)?;
        if revoked {
            if let Some(secure) = &self.secure {
                secure.revoke(id);
            }
            self.paired_clients
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .retain(|_, client| client.id != id);
        }
        Ok(revoked)
    }

    pub(crate) fn claim_secure_pairing(
        &self,
        context: &mold_secure::RequestContext,
        proof: &str,
        client_name: &str,
        client_kind: &str,
    ) -> anyhow::Result<(String, mold_secure::DeviceGrant)> {
        let secure = self
            .secure
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("secure pairing unavailable"))?;
        let db = self
            .metadata_db
            .as_ref()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paired access requires metadata database"))?;
        let _mutation = self
            .secure_mutations
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let invite = context
            .invite
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("pairing invitation required"))?;
        // Invitations do not reserve device slots. Recheck under the mutation
        // lock so concurrent claims cannot exceed the connector policy limit.
        // Leave a waiting invitation usable if capacity is released later.
        anyhow::ensure!(
            secure.has_device_capacity(),
            "paired device capacity reached"
        );
        anyhow::ensure!(
            secure.consume_invite(
                invite,
                &context.binding,
                &mold_secure::decode_hex(&context.handshake_hash)?,
                proof
            )?,
            "pairing invitation already used or proof invalid"
        );
        let mut bytes = [0; 32];
        getrandom::fill(&mut bytes)?;
        let credential = format!("mold_pair_{}", URL_SAFE_NO_PAD.encode(bytes));
        let credential_hash: [u8; 32] = Sha256::digest(credential.as_bytes()).into();
        let id = format!("pair_{}", URL_SAFE_NO_PAD.encode(&credential_hash[..12]));
        let mut grant_id = [0; 16];
        getrandom::fill(&mut grant_id)?;
        let device = mold_db::secure_devices::SecureDevice {
            client_id: id.clone(),
            server_instance_id: self.server_instance_id.as_ref().clone(),
            grant_id: crate::connections::hex(&grant_id),
            signing_key: context.binding.signing_key.clone(),
            noise_key: context.binding.noise_key.clone(),
            credential_hash,
            expires_at: unix_timestamp() + crate::secure_pairing::GRANT_TTL,
            epoch: secure.epoch(),
        };
        let access = PairedClientAccess {
            id: id.clone(),
            name: normalized_client_name(client_name),
            client_kind: normalized_client_kind(client_kind),
            created_at_ms: unix_timestamp_ms(),
            last_used_at_ms: None,
        };
        mold_db::secure_devices::SecureDevices::new(db).insert(
            &mold_db::paired_clients::PairedClient {
                id,
                server_instance_id: self.server_instance_id.as_ref().clone(),
                name: access.name.clone(),
                client_kind: access.client_kind.clone(),
                credential_hash,
                created_at_ms: access.created_at_ms,
                last_used_at_ms: None,
            },
            &device,
        )?;
        let certificate = secure.certificate(&device)?;
        self.paired_clients
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(credential_hash, access);
        secure.insert_device(device);
        Ok((credential, certificate))
    }
    pub(crate) fn renew_secure_pairing(
        &self,
        context: &mold_secure::RequestContext,
    ) -> anyhow::Result<mold_secure::DeviceGrant> {
        let secure = self
            .secure
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("secure pairing unavailable"))?;
        let db = self
            .metadata_db
            .as_ref()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paired access requires metadata database"))?;
        let _mutation = self
            .secure_mutations
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut device = secure
            .device_for_context(context)
            .ok_or_else(|| anyhow::anyhow!("paired device revoked"))?;
        device.expires_at = unix_timestamp() + crate::secure_pairing::GRANT_TTL;
        anyhow::ensure!(
            mold_db::secure_devices::SecureDevices::new(db).renew(
                &self.server_instance_id,
                &device.grant_id,
                device.expires_at
            )?,
            "paired device revoked"
        );
        let certificate = secure.certificate(&device)?;
        secure.insert_device(device);
        Ok(certificate)
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
    // Protocol-1 tokens have no device key binding and are deliberately retired.
    let secure = match mold_db::secure_devices::SecureDevices::new(db).list(server_instance_id) {
        Ok(devices) => devices
            .into_iter()
            .map(|d| (d.client_id, d.credential_hash))
            .collect::<HashSet<_>>(),
        Err(error) => {
            warn!(error = %format!("{error:#}"), "failed to load secure paired clients");
            return HashMap::new();
        }
    };
    match mold_db::paired_clients::PairedClients::new(db).list(server_instance_id) {
        Ok(clients) => clients
            .into_iter()
            .filter(|client| secure.contains(&(client.id.clone(), client.credential_hash)))
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

pub(crate) fn normalized_client_kind(value: &str) -> String {
    match value.trim() {
        "iphone" => "iphone",
        "ipad" => "ipad",
        "android" => "android",
        "browser" => "browser",
        "web" => "web",
        "desktop" => "desktop",
        "cli" => "cli",
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
    let Some(keys) = load_operator_keys()? else {
        return Ok(None);
    };

    tracing::info!(num_keys = keys.len(), "API key authentication enabled");
    let mut key_set = ApiKeySet::try_new(keys, metadata_db, server_instance_id)
        .map_err(|error| anyhow::anyhow!("failed to generate gallery signing secret: {error}"))?;
    if let Some(db) = key_set.metadata_db.as_ref() {
        let home = mold_core::Config::mold_dir()
            .ok_or_else(|| anyhow::anyhow!("secure identity requires MOLD_HOME"))?;
        let identity = crate::secure_pairing::load_host_identity(&home)?;
        let devices =
            mold_db::secure_devices::SecureDevices::new(db).list(&key_set.server_instance_id)?;
        key_set.secure = Some(Arc::new(crate::secure_pairing::SecurePairing::new(
            identity, devices,
        )));
    }
    Ok(Some(Arc::new(key_set)))
}

// Parse operator configuration without creating a host identity. Bootstrap must
// validate this boundary before importing its first paired client.
pub(crate) fn load_operator_keys() -> anyhow::Result<Option<HashSet<String>>> {
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

    Ok((!keys.is_empty()).then_some(keys))
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

    // Only a real loopback socket may carry plaintext API traffic. The secure
    // WebSocket endpoint is mounted outside this middleware. Fail closed even
    // with auth disabled or if future service wiring omits peer metadata.
    let local = request
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .is_some_and(|peer| peer.0.ip().is_loopback());
    if !local {
        if is_browser_bootstrap_read(request.method(), request.uri().path()) {
            return (
                StatusCode::SEE_OTHER,
                [
                    (
                        axum::http::header::LOCATION,
                        "https://utensils.io/mold/studio/",
                    ),
                    (axum::http::header::CACHE_CONTROL, "no-store"),
                ],
            )
                .into_response();
        }
        return (
            StatusCode::UPGRADE_REQUIRED,
            [(axum::http::header::CACHE_CONTROL, "no-store")],
            Json(AuthError {
                error: "A secure Mold connection is required".into(),
                code: "SECURE_CONNECTION_REQUIRED".into(),
            }),
        )
            .into_response();
    }

    // Auth state is stored as an extension by the layer setup in lib.rs.
    let auth_state = request.extensions().get::<AuthState>().cloned();

    let key_set = match auth_state.as_ref().and_then(|s| s.as_ref()) {
        Some(ks) => ks,
        None => return next.run(request).await, // Auth disabled
    };

    let secure_header = request.headers().get("x-mold-secure-peer").cloned();
    if let Some(value) = secure_header {
        let peer = request
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>();
        if !peer.is_some_and(|peer| peer.0.ip().is_loopback()) {
            return unauthorized("secure peer proof requires local connector");
        }
        let context = value
            .to_str()
            .ok()
            .filter(|s| s.len() <= 4096)
            .and_then(|s| URL_SAFE_NO_PAD.decode(s).ok())
            .and_then(|bytes| serde_json::from_slice::<mold_secure::RequestContext>(&bytes).ok());
        let Some(context) = context else {
            return unauthorized("invalid secure peer proof");
        };
        let Some(secure) = &key_set.secure else {
            return unauthorized("secure pairing unavailable");
        };
        let target = request
            .uri()
            .path_and_query()
            .map(|p| p.as_str())
            .unwrap_or("/");
        if secure
            .verify_request(&context, request.method().as_str(), target)
            .is_err()
        {
            return unauthorized("invalid or revoked secure peer proof");
        }
        request.headers_mut().remove("x-mold-secure-peer");
        request.extensions_mut().insert(context);
    }
    if request.uri().path() == "/api/pairing/secure/check" {
        if request.method() != Method::GET || request.uri().query().is_some() {
            return unauthorized("invalid secure control route");
        }
        let local = request
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .is_some_and(|peer| peer.0.ip().is_loopback());
        let context = request
            .headers()
            .get("x-mold-secure-control")
            .and_then(|v| v.to_str().ok())
            .filter(|s| s.len() <= 1024)
            .and_then(|s| URL_SAFE_NO_PAD.decode(s).ok())
            .and_then(|bytes| serde_json::from_slice::<mold_secure::ControlContext>(&bytes).ok());
        if !local
            || !context.is_some_and(|c| {
                key_set
                    .secure
                    .as_ref()
                    .is_some_and(|secure| secure.verify_control(&c).is_ok())
            })
        {
            return unauthorized("invalid secure control proof");
        }
        request.headers_mut().remove("x-mold-secure-control");
        return next.run(request).await;
    }
    if request.uri().path() == "/api/pairing/claim"
        && request
            .extensions()
            .get::<mold_secure::RequestContext>()
            .is_none()
    {
        return unauthorized("encrypted pairing connection required");
    }

    // Exempt certain paths (health checks, docs).
    let path = request.uri().path();
    if EXEMPT_PATHS.contains(&path) || is_browser_bootstrap_read(request.method(), path) {
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
                if let Some(secure) = &key_set.secure {
                    let hash: [u8; 32] = Sha256::digest(candidate.as_bytes()).into();
                    let context = request.extensions().get::<mold_secure::RequestContext>();
                    // An encrypted device may use only its bound per-device key.
                    // A copied key cannot escape that binding via direct HTTP.
                    if context.is_some_and(|ctx| !secure.credential_matches(ctx, &hash))
                        || (secure.credential_bound(&hash) && context.is_none())
                    {
                        return unauthorized("device credential does not match secure peer");
                    }
                }
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

/// Only the public browser shell and its bundle may load before API authentication.
/// Do not exempt an arbitrary non-API prefix: unmatched routes remain protected.
fn is_browser_bootstrap_read(method: &Method, path: &str) -> bool {
    if method != Method::GET && method != Method::HEAD {
        return false;
    }
    let path = if path == "/" {
        path
    } else {
        path.strip_suffix('/').unwrap_or(path)
    };
    if matches!(
        path,
        "/" | "/index.html"
            | "/create"
            | "/queue"
            | "/library"
            | "/models"
            | "/machines"
            | "/settings"
            | "/logo.png"
            | "/favicon.ico"
            | "/favicon.png"
            | "/favicon.svg"
            | "/apple-touch-icon.png"
    ) {
        return true;
    }
    let safe_segment = |segment: &str| {
        !segment.is_empty()
            && segment != "."
            && segment != ".."
            && segment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    if let Some(asset) = path.strip_prefix("/assets/") {
        let public_extension = asset.rsplit_once('.').is_some_and(|(_, extension)| {
            matches!(
                extension,
                "js" | "css"
                    | "ttf"
                    | "otf"
                    | "woff"
                    | "woff2"
                    | "png"
                    | "jpg"
                    | "jpeg"
                    | "svg"
                    | "webp"
                    | "gif"
                    | "ico"
                    | "avif"
                    | "wasm"
            )
        });
        return public_extension && asset.split('/').all(safe_segment);
    }
    path.strip_prefix("/machines/").is_some_and(safe_segment)
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

#[cfg(test)]
pub(crate) fn test_loopback_peer(
) -> axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>> {
    axum::Extension(axum::extract::ConnectInfo(
        "127.0.0.1:1000".parse().unwrap(),
    ))
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
    fn startup_rejects_legacy_paired_credentials_without_secure_device_binding() {
        let db = Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        mold_db::paired_clients::PairedClients::new(db.as_ref().as_ref().unwrap())
            .insert(&mold_db::paired_clients::PairedClient {
                id: "legacy".into(),
                server_instance_id: "host".into(),
                name: "old phone".into(),
                client_kind: "iphone".into(),
                credential_hash: Sha256::digest(b"old-key").into(),
                created_at_ms: 0,
                last_used_at_ms: None,
            })
            .unwrap();
        let keys = ApiKeySet::new_with_metadata_db(HashSet::from(["operator".into()]), db, "host");
        assert!(!keys.contains("old-key"));
        assert!(keys.paired_clients().is_empty());
        assert!(keys.contains("operator"));
    }

    #[tokio::test]
    async fn authenticated_server_allows_only_browser_bootstrap_reads() {
        use axum::{
            body::Body,
            http::{Method, Request, StatusCode},
        };
        let auth = Some(Arc::new(ApiKeySet::new(HashSet::from([
            "secret".to_string()
        ]))));
        let app = axum::Router::new()
            .fallback(|| async { StatusCode::OK })
            .layer(axum::middleware::from_fn(require_api_key))
            .layer(crate::auth::test_loopback_peer())
            .layer(axum::middleware::from_fn_with_state(
                auth,
                inject_auth_state,
            ));
        for path in [
            "/",
            "/index.html",
            "/favicon.ico",
            "/favicon.svg",
            "/apple-touch-icon.png",
            "/create/",
            "/machines/origin/",
            "/create",
            "/library",
            "/machines/origin",
            "/assets/index-a.js",
            "/logo.png",
        ] {
            for method in [Method::GET, Method::HEAD] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method(method)
                            .uri(path)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK, "{path}");
            }
        }
        for path in [
            "/api/status",
            "/api/status/",
            "/api/queue/private-job/input-thumbnail",
            "/api/queue/private-job/inputs",
            "/api/queue/private-job/input-thumbnail?index=1",
            "/api/queue/private-job/input-original?index=1",
            "/api/gallery/source-media/private.png/member/thumbnail",
            "/api/shutdown",
            "/secrets.json",
            "/.env",
            "/api/unknown",
            "/unknown",
            "/assets/../api/status",
            "/assets//secret",
            "/assets/%2e%2e/api/status",
            "/assets/.env",
            "/assets/private.json",
        ] {
            let response = app
                .clone()
                .oneshot(Request::get(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        }
        let response = app
            .clone()
            .oneshot(
                Request::get("/api/queue/private-job/input-original?index=1")
                    .header("x-api-key", "secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        for path in ["/", "/create", "/assets/index-a.js"] {
            let response = app
                .clone()
                .oneshot(Request::post(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        }
    }

    #[test]
    fn browser_bootstrap_matches_concrete_web_router_paths() {
        let router = include_str!("../../../web/src/router.ts");
        let mut checked = 0;
        for line in router.lines() {
            let Some((_, path)) = line.split_once("path: ") else {
                continue;
            };
            let path = path.trim_start_matches('"');
            let path = path.split('"').next().unwrap();
            if path.contains("pathMatch") {
                continue;
            }
            let path = path.replace(":id", "render-machine");
            assert!(is_browser_bootstrap_read(&Method::GET, &path), "{path}");
            assert!(is_browser_bootstrap_read(&Method::HEAD, &path), "{path}");
            assert!(!is_browser_bootstrap_read(&Method::POST, &path), "{path}");
            checked += 1;
        }
        assert!(
            checked >= 8,
            "the router source must still contain concrete paths"
        );
    }

    fn protected_test_app(auth_state: AuthState) -> axum::Router {
        axum::Router::new()
            .route(
                "/api/devices/:id",
                axum::routing::patch(|| async { axum::http::StatusCode::OK }),
            )
            .layer(axum::middleware::from_fn(require_api_key))
            .layer(crate::auth::test_loopback_peer())
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
            .layer(crate::auth::test_loopback_peer())
            .layer(axum::middleware::from_fn_with_state(
                auth_state,
                inject_auth_state,
            ))
    }

    #[tokio::test]
    async fn pairing_creation_requires_auth_and_claim_requires_encrypted_connection() {
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
        assert_eq!(claimed.status(), StatusCode::UNAUTHORIZED);
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
