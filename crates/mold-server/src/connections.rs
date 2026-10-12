//! Host-authorized connection advertisements; every remote route verifies pinned Noise identity.
use axum::{extract::State, http::header, response::IntoResponse, Extension, Json};
use serde::Serialize;
use std::net::{IpAddr, SocketAddr};
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, utoipa::ToSchema, PartialEq, Eq)]
pub(crate) struct ConnectionEndpoint {
    pub url: String,
    pub kind: String,
}
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LocalNetworkState {
    pub enabled: bool,
    pub epoch: u64,
}
#[derive(Clone, Default)]
pub(crate) struct RegularNetworkStatus {
    pub epoch: u64,
    pub address: Option<SocketAddr>,
    pub error: Option<String>,
}
pub struct ConnectionAddresses {
    configuration: RwLock<Option<(SocketAddr, Option<String>)>>,
    managed_relay_origin: RwLock<Option<String>>,
    secure_websocket_url: RwLock<Option<String>>,
    secure_policy: RwLock<Option<tokio::sync::watch::Receiver<mold_relay::secure::HostPolicy>>>,
    local_network_enabled: tokio::sync::watch::Sender<LocalNetworkState>,
    local_network_listener: RwLock<Option<SocketAddr>>,
    regular_network_enabled: tokio::sync::watch::Sender<LocalNetworkState>,
    regular_network_status: tokio::sync::watch::Sender<RegularNetworkStatus>,
    regular_api_enabled: std::sync::atomic::AtomicBool,
}
impl Default for ConnectionAddresses {
    fn default() -> Self {
        Self {
            configuration: RwLock::default(),
            managed_relay_origin: RwLock::default(),
            secure_websocket_url: RwLock::default(),
            secure_policy: RwLock::default(),
            local_network_enabled: tokio::sync::watch::channel(LocalNetworkState::default()).0,
            local_network_listener: RwLock::default(),
            regular_network_enabled: tokio::sync::watch::channel(LocalNetworkState::default()).0,
            regular_network_status: tokio::sync::watch::channel(RegularNetworkStatus::default()).0,
            regular_api_enabled: Default::default(),
        }
    }
}
impl ConnectionAddresses {
    pub fn set_regular_network_enabled(&self, enabled: bool) {
        self.regular_network_enabled.send_if_modified(|state| {
            if state.enabled == enabled {
                return false;
            }
            state.enabled = enabled;
            state.epoch = state.epoch.wrapping_add(1);
            true
        });
        if !enabled {
            self.regular_network_status
                .send_replace(RegularNetworkStatus::default());
        }
    }
    pub fn regular_network_endpoints(&self) -> Vec<String> {
        let state = *self.regular_network_enabled.borrow();
        let status = self.regular_network_status.borrow();
        if !state.enabled || state.epoch != status.epoch {
            return Vec::new();
        }
        let Some(bound) = status.address else {
            return Vec::new();
        };
        let interfaces = if_addrs::get_if_addrs()
            .unwrap_or_default()
            .into_iter()
            .map(|i| i.ip())
            .collect::<Vec<_>>();
        advertised_endpoints(bound, &interfaces, None)
            .into_iter()
            .map(|e| e.url)
            .collect()
    }
    pub async fn regular_network_ready(
        &self,
        timeout: std::time::Duration,
    ) -> anyhow::Result<Vec<String>> {
        let mut status = self.regular_network_status.subscribe();
        let mut enabled = self.regular_network_enabled.subscribe();
        let epoch = enabled.borrow().epoch;
        tokio::time::timeout(timeout, async {
            loop {
                anyhow::ensure!(
                    enabled.borrow().enabled && enabled.borrow().epoch == epoch,
                    "Regular access was disabled or replaced"
                );
                let current = status.borrow_and_update().clone();
                if current.epoch == epoch {
                    if let Some(error) = current.error {
                        anyhow::bail!("{error}")
                    }
                    if current.address.is_some() {
                        let endpoints = self.regular_network_endpoints();
                        anyhow::ensure!(
                            !endpoints.is_empty(),
                            "No direct network address is available"
                        );
                        return Ok(endpoints);
                    }
                }
                tokio::select! {
                    value = status.changed() => value?,
                    value = enabled.changed() => value?,
                }
            }
        })
        .await
        .map_err(|_| anyhow::anyhow!("Regular access did not become ready in time"))?
    }
    pub(crate) fn watch_regular_network(&self) -> tokio::sync::watch::Receiver<LocalNetworkState> {
        self.regular_network_enabled.subscribe()
    }
    pub(crate) fn publish_regular_network(&self, status: RegularNetworkStatus) {
        let enabled = self.regular_network_enabled.borrow();
        if enabled.enabled && enabled.epoch == status.epoch {
            self.regular_network_status.send_replace(status);
        }
    }
    pub(crate) fn set_regular_api_enabled(&self, enabled: bool) {
        self.regular_api_enabled
            .store(enabled, std::sync::atomic::Ordering::Release);
    }
    pub(crate) fn regular_api_available(&self) -> bool {
        self.regular_api_enabled
            .load(std::sync::atomic::Ordering::Acquire)
            || !self.regular_network_endpoints().is_empty()
    }
    /// In-process connectors observe pairing and revocation before the handler replies.
    pub fn watch_secure_policy(
        &self,
    ) -> Option<tokio::sync::watch::Receiver<mold_relay::secure::HostPolicy>> {
        self.secure_policy
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
    pub(crate) fn set_secure_policy(
        &self,
        policy: Option<tokio::sync::watch::Receiver<mold_relay::secure::HostPolicy>>,
    ) {
        *self
            .secure_policy
            .write()
            .unwrap_or_else(|p| p.into_inner()) = policy;
    }

    pub fn secure_websocket_url(&self) -> Option<String> {
        self.secure_websocket_url
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Discovery is only an address hint. Client trust always comes from the
    /// independently paired host key, including if the cloud changes this URL.
    pub fn set_secure_websocket_url(&self, value: Option<&str>) -> anyhow::Result<()> {
        let validated = value
            .map(|value| {
                let url = url::Url::parse(value)?;
                anyhow::ensure!(
                    value.len() <= 2048
                        && url.scheme() == "wss"
                        && url.host_str().is_some()
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.query().is_none()
                        && url.fragment().is_none()
                        && url.port() != Some(0),
                    "secure relay requires a credential-free WSS URL"
                );
                Ok::<_, anyhow::Error>(url.to_string())
            })
            .transpose()?;
        *self
            .secure_websocket_url
            .write()
            .unwrap_or_else(|p| p.into_inner()) = validated;
        Ok(())
    }

    /// Native Remote Access opt-in. The server opens a second listener using
    /// the same authenticated router; the private engine address never moves.
    pub fn set_local_network_enabled(&self, enabled: bool) {
        self.local_network_enabled.send_if_modified(|state| {
            if state.enabled == enabled {
                return false;
            }
            state.enabled = enabled;
            state.epoch = state.epoch.wrapping_add(1);
            true
        });
        if !enabled {
            self.publish_local_network_listener(None);
        }
    }

    pub(crate) fn watch_local_network(&self) -> tokio::sync::watch::Receiver<LocalNetworkState> {
        self.local_network_enabled.subscribe()
    }

    pub(crate) fn local_network_listener(&self) -> Option<SocketAddr> {
        *self
            .local_network_listener
            .read()
            .unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn publish_local_network_listener(&self, address: Option<(SocketAddr, u64)>) {
        let mut current = self
            .local_network_listener
            .write()
            .unwrap_or_else(|p| p.into_inner());
        let state = self.local_network_enabled.borrow();
        *current = address
            .filter(|(_, epoch)| state.enabled && state.epoch == *epoch)
            .map(|(address, _)| address);
    }
    /// App-owned enrollment may advertise or withdraw a credential-free HTTPS
    /// origin without changing process environment or the bound listener.
    pub fn set_managed_relay_origin(&self, origin: Option<&str>) -> anyhow::Result<()> {
        let validated = origin.map(validate_public_url).transpose()?;
        *self
            .managed_relay_origin
            .write()
            .unwrap_or_else(|p| p.into_inner()) = validated;
        Ok(())
    }

    pub(crate) fn configure(
        &self,
        bound: SocketAddr,
        public_url: Option<&str>,
    ) -> anyhow::Result<()> {
        let public_url = public_url.map(validate_public_url).transpose()?;
        *self
            .configuration
            .write()
            .unwrap_or_else(|p| p.into_inner()) = Some((bound, public_url));
        Ok(())
    }
    pub(crate) fn endpoints(&self) -> Vec<ConnectionEndpoint> {
        let configured = self
            .configuration
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let Some((bound, public)) = configured else {
            return Vec::new();
        };
        let interfaces = if_addrs::get_if_addrs()
            .unwrap_or_default()
            .into_iter()
            .map(|i| i.ip())
            .collect::<Vec<_>>();
        let managed = self
            .managed_relay_origin
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let mut endpoints = advertised_endpoints(bound, &interfaces, public.as_deref());
        if let Some(bound) = self.local_network_listener() {
            for endpoint in advertised_endpoints(bound, &interfaces, None) {
                if !endpoints
                    .iter()
                    .any(|existing| existing.url == endpoint.url)
                {
                    endpoints.push(endpoint);
                }
            }
            endpoints.truncate(8);
        }
        if let Some(url) = managed {
            if !endpoints.iter().any(|endpoint| endpoint.url == url) {
                endpoints.insert(
                    0,
                    ConnectionEndpoint {
                        url,
                        kind: "relay".into(),
                    },
                );
                endpoints.truncate(8);
            }
        }
        endpoints
    }
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn validate_public_url(value: &str) -> anyhow::Result<String> {
    let url = url::Url::parse(value)
        .map_err(|_| anyhow::anyhow!("MOLD_PUBLIC_URL must be an HTTPS origin"))?;
    if value.len() > 2048
        || url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
        || url.port() == Some(0)
        || value
            .split('/')
            .nth(2)
            .is_some_and(|authority| authority.contains('@'))
    {
        anyhow::bail!(
            "MOLD_PUBLIC_URL must be an HTTPS origin without credentials, path, query or fragment"
        );
    }
    Ok(url.origin().ascii_serialization())
}
fn advertised_endpoints(
    bound: SocketAddr,
    interfaces: &[IpAddr],
    public: Option<&str>,
) -> Vec<ConnectionEndpoint> {
    let mut endpoints = Vec::new();
    if let Some(url) = public {
        endpoints.push(ConnectionEndpoint {
            url: url.into(),
            kind: "relay".into(),
        });
    }
    let scoped = matches!(bound, SocketAddr::V6(address) if address.scope_id() != 0);
    let mut local = interfaces
        .iter()
        .copied()
        .filter(|ip| {
            !scoped
                && bound.port() != 0
                && bound.is_ipv4() == ip.is_ipv4()
                && (bound.ip().is_unspecified() || bound.ip() == *ip)
                && usable(*ip)
        })
        .map(|ip| ConnectionEndpoint {
            url: format!("http://{}", SocketAddr::new(ip, bound.port())),
            kind: if tailscale(ip) { "tailscale" } else { "lan" }.into(),
        })
        .collect::<Vec<_>>();
    local.sort_by_key(|e| (e.kind != "tailscale", e.url.clone()));
    local.dedup_by(|a, b| a.url == b.url);
    endpoints.extend(local);
    endpoints.truncate(8);
    endpoints
}
fn tailscale(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.octets()[0] == 100 && (64..128).contains(&ip.octets()[1]),
        IpAddr::V6(ip) => ip.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0],
    }
}
fn usable(ip: IpAddr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return false;
    }
    match ip {
        IpAddr::V4(ip) => !ip.is_link_local() && !ip.is_broadcast() && ip.octets()[0] != 0,
        IpAddr::V6(ip) => {
            !ip.is_unicast_link_local()
                && ip.to_ipv4_mapped().is_none_or(|v4| usable(IpAddr::V4(v4)))
        }
    }
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct ConnectionAddressResponse {
    version: u8,
    instance_id: String,
    endpoints: Vec<ConnectionEndpoint>,
}
#[utoipa::path(get, path = "/api/connection-addresses", tag = "server", responses((status = 200, body = ConnectionAddressResponse), (status = 401, description = "Authentication required")))]
pub(crate) async fn addresses(
    State(state): State<crate::state::AppState>,
    authority: Option<Extension<crate::auth::PairingAuthority>>,
) -> impl IntoResponse {
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(ConnectionAddressResponse {
            version: 1,
            instance_id: state.instance_id.to_string(),
            endpoints: if authority.is_some() {
                Vec::new()
            } else {
                state.connection_addresses.endpoints()
            },
        }),
    )
}
#[cfg(test)]
mod tests {
    #[test]
    fn managed_origin_updates_the_shared_catalog_and_withdraws() {
        let addresses = ConnectionAddresses::default();
        addresses
            .configure("127.0.0.1:7680".parse().unwrap(), None)
            .unwrap();
        addresses
            .set_managed_relay_origin(Some("https://host.example.com"))
            .unwrap();
        assert_eq!(addresses.endpoints()[0].url, "https://host.example.com");
        assert!(addresses
            .set_managed_relay_origin(Some("http://host.example.com"))
            .is_err());
        assert_eq!(addresses.endpoints().len(), 1);
        addresses.set_managed_relay_origin(None).unwrap();
        assert!(addresses.endpoints().is_empty());
    }

    #[test]
    fn managed_withdrawal_preserves_explicit_operator_origin() {
        let addresses = std::sync::Arc::new(ConnectionAddresses::default());
        let consumer = addresses.clone();
        addresses
            .configure(
                "127.0.0.1:7680".parse().unwrap(),
                Some("https://operator.example.com"),
            )
            .unwrap();
        addresses
            .set_managed_relay_origin(Some("https://managed.example.com"))
            .unwrap();
        assert_eq!(consumer.endpoints().len(), 2);
        addresses.set_managed_relay_origin(None).unwrap();
        assert_eq!(consumer.endpoints()[0].url, "https://operator.example.com");
    }

    #[tokio::test]
    async fn retired_connection_probe_has_no_route() {
        use tower::ServiceExt;
        let response = crate::routes::create_router(crate::state::AppState::for_tests())
            .oneshot(
                axum::http::Request::post("/api/connection-probe")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
    }

    use super::*;
    #[test]
    fn public_url_requires_https_origin_without_credentials_or_path() {
        assert_eq!(
            validate_public_url("https://mold-link.urandom.io").unwrap(),
            "https://mold-link.urandom.io"
        );
        for bad in [
            "http://host",
            "https://user:password@host",
            "https://host/path",
            "https://host/?secret=x",
            "https://host/#x",
            "https://host:0",
            "https://host/../x",
        ] {
            assert!(validate_public_url(bad).is_err(), "{bad}");
        }
    }
    #[test]
    fn candidates_match_actual_bind_and_filter_unusable_interfaces() {
        let ips: Vec<IpAddr> = [
            "127.0.0.1",
            "0.0.0.0",
            "169.254.1.1",
            "224.0.0.1",
            "192.168.1.20",
            "100.80.1.2",
            "::1",
            "fe80::1",
            "ff02::1",
            "fd7a:115c:a1e0::123",
            "fd00::1",
        ]
        .iter()
        .map(|ip| ip.parse().unwrap())
        .collect();
        let v4 = advertised_endpoints(
            "0.0.0.0:7680".parse().unwrap(),
            &ips,
            Some("https://relay.example"),
        );
        assert_eq!(
            v4.iter().map(|e| (&*e.url, &*e.kind)).collect::<Vec<_>>(),
            [
                ("https://relay.example", "relay"),
                ("http://100.80.1.2:7680", "tailscale"),
                ("http://192.168.1.20:7680", "lan")
            ]
        );
        let specific = advertised_endpoints("192.168.1.20:7680".parse().unwrap(), &ips, None);
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].url, "http://192.168.1.20:7680");
        assert!(advertised_endpoints("127.0.0.1:7680".parse().unwrap(), &ips, None).is_empty());
        assert!(advertised_endpoints("10.1.1.1:7680".parse().unwrap(), &ips, None).is_empty());
        let v6 = advertised_endpoints("[::]:7680".parse().unwrap(), &ips, None);
        assert_eq!(v6.len(), 2);
        assert!(v6
            .iter()
            .any(|e| e.url == "http://[fd7a:115c:a1e0::123]:7680" && e.kind == "tailscale"));
        let lots: Vec<_> = (1..30).map(|n| IpAddr::from([10, 0, 0, n])).collect();
        assert_eq!(
            advertised_endpoints(
                "0.0.0.0:7680".parse().unwrap(),
                &lots,
                Some("https://relay.example")
            )
            .len(),
            8
        );
    }
}
