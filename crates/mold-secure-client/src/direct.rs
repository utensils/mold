//! Explicit direct transport contracts. A proxy is never a Regular endpoint.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionMode {
    #[default]
    Secure,
    Regular,
}

pub fn regular_origin(value: &str) -> Result<String> {
    ensure!(
        value.len() <= 2048 && !value.chars().any(|c| c.is_whitespace() || c.is_control()),
        "invalid direct origin"
    );
    let url = reqwest::Url::parse(value)?;
    ensure!(
        matches!(url.scheme(), "http" | "https")
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.path() == "/"
            && url.query().is_none()
            && url.fragment().is_none()
            && url.port() != Some(0),
        "Regular mode requires a credential-free HTTP(S) origin without a path, query or fragment"
    );
    ensure!(
        !is_managed_proxy(&url),
        "the Mold proxy always requires a Secure connection"
    );
    Ok(url.origin().ascii_serialization())
}

pub fn is_managed_proxy(url: &reqwest::Url) -> bool {
    let host = url.host_str().unwrap_or_default().trim_end_matches('.');
    host == "mold-link.urandom.io" || host.ends_with(".mold-link.urandom.io")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regular_origins_match_cross_client_contract() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../studio/api/fixtures/direct-origins-v1.json"
        ))
        .unwrap();
        for case in fixture["accepted"].as_array().unwrap() {
            assert_eq!(
                regular_origin(case["input"].as_str().unwrap()).unwrap(),
                case["origin"].as_str().unwrap()
            );
        }
        for case in fixture["rejected"].as_array().unwrap() {
            let input = case.as_str().unwrap();
            assert!(regular_origin(input).is_err(), "{input:?}");
        }
    }
}

#[cfg(test)]
mod regular_authority_tests {
    use super::*;
    #[test]
    fn regular_registration_requires_distinct_credential_and_proxy_never_regular() {
        let owner = uuid::Uuid::new_v4().to_string();
        let host = crate::prepare_bootstrap("modes", "desktop")
            .unwrap()
            .profile(&format!("https://{}.invalid", uuid::Uuid::new_v4()))
            .unwrap();
        crate::register_ephemeral(&owner, host.clone()).unwrap();
        assert!(register_regular_ephemeral(
            &owner,
            RegularHost {
                origin: host.origin.clone(),
                api_key: Some(host.api_key.clone())
            }
        )
        .is_err());
        register_regular_ephemeral(
            &owner,
            RegularHost {
                origin: host.origin.clone(),
                api_key: Some("distinct-direct-key".into()),
            },
        )
        .unwrap();
        assert!(crate::has_ephemeral_authority(&owner, &host));
        assert!(register_regular_ephemeral(
            &owner,
            RegularHost {
                origin: "https://abc.mold-link.urandom.io".into(),
                api_key: None
            }
        )
        .is_err());
        forget_regular_ephemeral(&owner, None);
        crate::forget_ephemeral(&owner, None);
    }
    #[tokio::test]
    async fn regular_dispatch_strips_internal_headers_and_cookies_without_following_redirects() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let owner = uuid::Uuid::new_v4().to_string();
        register_regular_ephemeral(
            &owner,
            RegularHost {
                origin: origin.clone(),
                api_key: Some("direct-cookie-test".into()),
            },
        )
        .unwrap();
        let captured = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = vec![0; 8192];
            let count = socket.read(&mut bytes).await.unwrap();
            socket.write_all(b"HTTP/1.1 302 Found\r\nLocation: https://refuse-redirect.invalid\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
            String::from_utf8(bytes[..count].to_vec())
                .unwrap()
                .to_lowercase()
        });
        let client = reqwest::Client::new();
        let request = client
            .get(format!("{origin}/api/status"))
            .header("x-mold-connection-mode", "regular")
            .header("cookie", "fixture-cookie")
            .header("x-mold-secure-peer", "fixture-proof")
            .header("x-mold-secure-control", "fixture-control")
            .build()
            .unwrap();
        let response = crate::execute_http(&client, request).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        let request = captured.await.unwrap();
        assert!(request.contains("x-api-key: direct-cookie-test"));
        for header in [
            "cookie:",
            "x-mold-connection-mode:",
            "x-mold-secure-peer:",
            "x-mold-secure-control:",
        ] {
            assert!(!request.contains(header));
        }
        forget_regular_ephemeral(&owner, None);
    }
    #[tokio::test]
    async fn regular_dispatch_never_exports_known_secure_authorization() {
        let owner = uuid::Uuid::new_v4().to_string();
        let secure = crate::prepare_bootstrap("headers", "desktop")
            .unwrap()
            .profile(&format!("https://{}.invalid", uuid::Uuid::new_v4()))
            .unwrap();
        crate::register_ephemeral(&owner, secure.clone()).unwrap();
        crate::forget_ephemeral(&owner, None);
        let origin = format!("https://{}.invalid", uuid::Uuid::new_v4());
        register_regular_ephemeral(
            &owner,
            RegularHost {
                origin: origin.clone(),
                api_key: Some("regular".into()),
            },
        )
        .unwrap();
        let client = reqwest::Client::new();
        let request = client
            .get(format!("{origin}/api/status"))
            .header("x-mold-connection-mode", "regular")
            .header("authorization", "Bearer harmless-fixture")
            .header("authorization", format!("Bearer {}", secure.api_key))
            .build()
            .unwrap();
        let error = crate::execute_http(&client, request).await.unwrap_err();
        assert!(error.to_string().contains("Secure credentials"));
        let joined = client
            .get(format!("{origin}/api/status"))
            .header("x-mold-connection-mode", "regular")
            .header(
                "authorization",
                format!("Bearer harmless, Bearer {}", secure.api_key),
            )
            .build()
            .unwrap();
        assert!(crate::execute_http(&client, joined)
            .await
            .unwrap_err()
            .to_string()
            .contains("Secure credentials"));
        forget_regular_ephemeral(&owner, None);
        crate::forget_ephemeral(&owner, None);
    }
    #[tokio::test]
    async fn regular_dispatch_requires_request_local_mode_and_registered_credential() {
        let origin = format!("https://{}.invalid", uuid::Uuid::new_v4());
        let mut request = reqwest::Client::new()
            .get(format!("{origin}/api/status"))
            .build()
            .unwrap();
        request
            .headers_mut()
            .insert("x-mold-connection-mode", "regular".parse().unwrap());
        let error = crate::execute_http(&reqwest::Client::new(), request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Regular credential"));
    }
}

/// Direct credentials are distinct from pinned Noise authority, even at one origin.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RegularHost {
    pub origin: String,
    pub api_key: Option<String>,
}
impl RegularHost {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            regular_origin(&self.origin)? == self.origin,
            "invalid Regular origin"
        );
        if let Some(key) = &self.api_key {
            anyhow::ensure!(
                !key.is_empty()
                    && key.len() <= 4096
                    && reqwest::header::HeaderValue::from_str(key).is_ok(),
                "invalid Regular credential"
            );
            anyhow::ensure!(
                !crate::store::is_known_secure_credential(key)?,
                "Secure credentials cannot authorize Regular HTTP"
            );
        }
        Ok(())
    }
}
fn regular_registry(
) -> std::sync::MutexGuard<'static, std::collections::BTreeMap<(String, String), RegularHost>> {
    static REGISTRY: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<(String, String), RegularHost>>,
    > = std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}
pub fn register_regular_ephemeral(owner: &str, host: RegularHost) -> Result<()> {
    host.validate()?;
    let mut registry = regular_registry();
    let key = (owner.to_owned(), host.origin.clone());
    anyhow::ensure!(
        registry.contains_key(&key) || registry.len() < 128,
        "native Regular host limit exceeded"
    );
    anyhow::ensure!(
        !registry
            .iter()
            .any(|((other, origin), saved)| other != owner
                && origin == &host.origin
                && saved.api_key != host.api_key),
        "conflicting Regular credential for this origin"
    );
    registry.insert(key, host);
    Ok(())
}
pub fn forget_regular_ephemeral(owner: &str, origin: Option<&str>) {
    regular_registry().retain(|(o, u), _| o != owner || origin.is_some_and(|v| v != u));
}
fn regular_path(origin: &str) -> std::path::PathBuf {
    use sha2::{Digest, Sha256};
    crate::default_store().join(format!(
        "regular-{}.json",
        hex::encode(Sha256::digest(origin.as_bytes()))
    ))
}
pub fn save_regular_host(host: &RegularHost) -> Result<()> {
    host.validate()?;
    let bytes = zeroize::Zeroizing::new(serde_json::to_vec(host)?);
    mold_secure::identity_file::write_private_file(&regular_path(&host.origin), &bytes, true)?;
    select_connection_mode(&host.origin, ConnectionMode::Regular)
}
pub fn load_regular_host(origin: &str) -> Result<RegularHost> {
    if let Some(host) = regular_registry()
        .iter()
        .find(|((_, u), _)| u == origin)
        .map(|(_, h)| h.clone())
    {
        return Ok(host);
    }
    let bytes = crate::read_private(&regular_path(origin))
        .context("No Regular credential for this host; explicitly pair or enter a direct key")?;
    let host: RegularHost = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(host.origin == origin, "Regular credential origin mismatch");
    host.validate()?;
    Ok(host)
}
fn selection_path(origin: &str) -> std::path::PathBuf {
    regular_path(origin).with_extension("mode")
}
pub fn select_connection_mode(origin: &str, mode: ConnectionMode) -> Result<()> {
    mold_secure::identity_file::write_private_file(
        &selection_path(origin),
        &serde_json::to_vec(&mode)?,
        true,
    )?;
    Ok(())
}
pub fn configured_connection_mode(origin: &str) -> Result<ConnectionMode> {
    if let Ok(value) = std::env::var("MOLD_CONNECTION_MODE") {
        return match value.as_str() {
            "secure" => Ok(ConnectionMode::Secure),
            "regular" => Ok(ConnectionMode::Regular),
            _ => anyhow::bail!("MOLD_CONNECTION_MODE must be secure or regular"),
        };
    }
    let path = selection_path(origin);
    if !path.try_exists()? {
        return Ok(ConnectionMode::Secure);
    }
    Ok(serde_json::from_slice(&crate::read_private(&path)?)?)
}

impl ConnectionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Secure => "secure",
            Self::Regular => "regular",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegularInvite {
    #[serde(rename = "type")]
    kind: String,
    version: u8,
    mode: ConnectionMode,
    base_url: String,
    instance_id: String,
    name: Option<String>,
    pairing_token: String,
    expires_at: u64,
}
pub async fn pair_regular(value: serde_json::Value, name: &str) -> Result<RegularHost> {
    let invite: RegularInvite = serde_json::from_value(value)?;
    anyhow::ensure!(
        invite.kind == "mold.mobile-pairing"
            && invite.version == 2
            && invite.mode == ConnectionMode::Regular,
        "invalid Regular invitation"
    );
    let origin = regular_origin(&invite.base_url)?;
    anyhow::ensure!(
        origin == invite.base_url
            && !invite.instance_id.is_empty()
            && invite.pairing_token.len() == 64
            && invite
                .pairing_token
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid Regular invitation authority"
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    anyhow::ensure!(now < invite.expires_at, "Regular invitation expired");
    let _display_name = invite.name;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let mut response = client.post(format!("{origin}/api/pairing/regular/claim")).json(&serde_json::json!({"token":invite.pairing_token,"client_name":name,"client_kind":"cli"})).send().await?.error_for_status()?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        anyhow::ensure!(
            bytes.len() + chunk.len() <= 16384,
            "invalid Regular claim size"
        );
        bytes.extend_from_slice(&chunk);
    }
    let claim: serde_json::Value = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        claim["mode"] == "regular"
            && claim["base_url"] == origin
            && claim["instance_id"] == invite.instance_id,
        "Regular claim authority mismatch"
    );
    let host = RegularHost {
        origin,
        api_key: Some(
            claim["api_key"]
                .as_str()
                .context("missing Regular credential")?
                .to_owned(),
        ),
    };
    host.validate()?;
    Ok(host)
}
