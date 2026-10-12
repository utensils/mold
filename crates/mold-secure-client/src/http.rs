//! One dispatch boundary for native Mold HTTP consumers. A remote origin always
//! needs the application's pinned pairing; requests are never replayed.
use crate::{PairedHost, SecureClient};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
struct Cached {
    fingerprint: [u8; 32],
    client: SecureClient,
}
fn cache() -> std::sync::MutexGuard<'static, HashMap<String, Cached>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Cached>>> = OnceLock::new();
    CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
fn fingerprint(host: &PairedHost) -> [u8; 32] {
    let mut hash = Sha256::new();
    for value in [
        &host.namespace,
        &host.host_key,
        &host.host_noise_key,
        &host.device_identity,
        &host.api_key,
        &host.grant.id,
        &host.websocket,
    ] {
        hash.update((value.len() as u64).to_be_bytes());
        hash.update(value.as_bytes());
    }
    hash.update(host.grant.epoch.to_be_bytes());
    hash.finalize().into()
}
pub fn forget_connection(origin: &str) {
    if let Some(entry) = cache().remove(origin) {
        entry.client.close();
    }
}
pub fn is_private_engine(url: &reqwest::Url) -> bool {
    url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    })
}
pub async fn execute_http(
    client: &reqwest::Client,
    mut request: reqwest::Request,
) -> Result<reqwest::Response> {
    let origin = request.url().origin().ascii_serialization();
    let mode = request.headers_mut().remove("x-mold-connection-mode");
    if let Some(mode) = mode {
        match mode.to_str()? {
            "secure" => {}
            "regular" => {
                crate::regular_origin(&origin)?;
                for value in request.headers().get_all("authorization") {
                    let value = value.to_str()?;
                    anyhow::ensure!(
                        !crate::store::is_known_secure_credential(value)?,
                        "Secure credentials cannot authorize Regular HTTP"
                    );
                    let mut parts = value.split_ascii_whitespace();
                    let scheme = parts.next().unwrap_or_default();
                    if scheme.eq_ignore_ascii_case("bearer") {
                        let candidate = parts.next().unwrap_or_default();
                        anyhow::ensure!(
                            !crate::store::is_known_secure_credential(candidate)?,
                            "Secure credentials cannot authorize Regular HTTP"
                        );
                    }
                }
                request.headers_mut().remove("cookie");
                request.headers_mut().remove("x-mold-secure-peer");
                request.headers_mut().remove("x-mold-secure-control");
                let host = crate::load_regular_host(&origin)?;
                host.validate()?;
                for provided in request.headers().get_all("x-api-key") {
                    anyhow::ensure!(
                        Some(provided.to_str()?) == host.api_key.as_deref(),
                        "Regular credential does not match the selected host"
                    );
                }
                if let Some(key) = host.api_key {
                    request.headers_mut().insert("x-api-key", key.parse()?);
                }
                // A direct credential must never follow a redirect to another authority.
                let direct = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()?;
                return direct
                    .execute(request)
                    .await
                    .context("Regular Mold request failed");
            }
            _ => anyhow::bail!("invalid Mold connection mode"),
        }
    }
    let paired = crate::pairing_exists(&origin);
    if is_private_engine(request.url()) && !paired {
        return client
            .execute(request)
            .await
            .context("local Mold request failed");
    }
    // Re-read authority before every dispatch. Cache hits cannot resurrect a
    // forgotten/replaced identity, and an unreadable store fails closed.
    let host = crate::load_pairing(&origin)?;
    let fingerprint = fingerprint(&host);
    let secure = {
        let mut entries = cache();
        if entries
            .get(&origin)
            .is_some_and(|e| e.fingerprint != fingerprint)
        {
            if let Some(old) = entries.remove(&origin) {
                old.client.close();
            }
        }
        if !entries.contains_key(&origin) {
            if entries.len() >= 128 {
                anyhow::bail!("too many active secure hosts")
            }
            entries.insert(
                origin.clone(),
                Cached {
                    fingerprint,
                    client: SecureClient::new(host)?,
                },
            );
        }
        entries[&origin].client.clone()
    };
    // Timeout before response headers, if requested by the existing consumer.
    // Body backpressure/cancellation remains owned by the response stream.
    if let Some(timeout) = request.timeout().copied() {
        tokio::time::timeout(timeout, secure.send(request))
            .await
            .context("secure request timed out; its outcome may be unknown")?
    } else {
        secure.send(request).await
    }
}
pub trait RequestBuilderExt {
    fn secure_send(self) -> impl std::future::Future<Output = Result<reqwest::Response>> + Send;
}
impl RequestBuilderExt for reqwest::RequestBuilder {
    async fn secure_send(self) -> Result<reqwest::Response> {
        let (client, request) = self.build_split();
        let mut request = request?;
        if !request.headers().contains_key("x-mold-connection-mode") {
            let origin = request.url().origin().ascii_serialization();
            let mode = crate::configured_connection_mode(&origin)?;
            request.headers_mut().insert(
                "x-mold-connection-mode",
                match mode {
                    crate::ConnectionMode::Secure => "secure",
                    crate::ConnectionMode::Regular => "regular",
                }
                .parse()?,
            );
        }
        execute_http(&client, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn remote_unpaired_never_dispatches_plaintext() {
        let error = reqwest::Client::new()
            .post("https://unpaired.invalid/api/generate")
            .header("x-api-key", "fixture-secret")
            .secure_send()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("No secure pairing"));
    }
    #[test]
    fn only_loopback_is_private_engine() {
        for s in [
            "http://localhost:7680",
            "http://127.0.0.1:7680",
            "http://[::1]:7680",
        ] {
            assert!(is_private_engine(&reqwest::Url::parse(s).unwrap()));
        }
        for s in ["http://192.168.1.5:7680", "http://localhost.evil.invalid"] {
            assert!(!is_private_engine(&reqwest::Url::parse(s).unwrap()));
        }
    }
}
