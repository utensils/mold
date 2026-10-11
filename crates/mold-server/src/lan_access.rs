//! Opt-in LAN access exposes only the pinned encrypted transport.

use crate::{ConnectionAddresses, secure_pairing::SecurePairing};
use axum::{extract::ConnectInfo, Extension, Router};
use hyper_util::{rt::TokioIo, service::TowerToHyperService};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{net::TcpListener, task::JoinSet};
use tokio_util::sync::CancellationToken;

pub(crate) struct LanAccess {
    cancel: CancellationToken,
    addresses: Arc<ConnectionAddresses>,
    task: Option<tokio::task::JoinHandle<()>>,
}
impl LanAccess {
    pub(crate) async fn shutdown(mut self) {
        self.addresses.set_local_network_enabled(false);
        self.cancel.cancel();
        if let Some(task) = self.task.take() { let _ = task.await; }
    }
}
impl Drop for LanAccess {
    fn drop(&mut self) {
        self.addresses.set_local_network_enabled(false);
        self.cancel.cancel();
    }
}

pub(crate) fn start(
    secure: Option<Arc<SecurePairing>>,
    target: SocketAddr,
    addresses: Arc<ConnectionAddresses>,
    txt: Vec<(String, String)>,
) -> LanAccess {
    let cancel = CancellationToken::new();
    let task = secure.map(|secure| tokio::spawn(run(secure, target, addresses.clone(), cancel.clone(), txt)));
    LanAccess { cancel, addresses, task }
}

async fn run(
    secure: Arc<SecurePairing>,
    target: SocketAddr,
    addresses: Arc<ConnectionAddresses>,
    cancel: CancellationToken,
    txt: Vec<(String, String)>,
) {
    let mut enabled = addresses.watch_local_network();
    loop {
        while !enabled.borrow_and_update().enabled {
            tokio::select! {
                _ = cancel.cancelled() => return,
                changed = enabled.changed() => if changed.is_err() { return; },
            }
        }
        let epoch = enabled.borrow().epoch;
        // Separate from the private engine port. The OS selects a free port;
        // Bonjour and the authenticated catalog publish its actual address.
        let listener = match TcpListener::bind(("0.0.0.0", 0)).await {
            Ok(listener) => listener,
            Err(error) => {
                tracing::warn!(%error, "local network access could not listen");
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = enabled.changed() => {},
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                }
                continue;
            }
        };
        let Ok(bound) = listener.local_addr() else { continue; };
        // Stop can race the asynchronous bind. Never republish after opt-out.
        if cancel.is_cancelled() || !enabled.borrow().enabled || enabled.borrow().epoch != epoch { continue; }
        addresses.publish_local_network_listener(Some((bound, epoch)));
        #[cfg(feature = "mdns")]
        let advertisement = if crate::mdns::enabled_from_env() {
            crate::mdns::register("0.0.0.0", bound.port(), txt.clone()).ok()
        } else { None };
        #[cfg(not(feature = "mdns"))]
        let _ = &txt;
        let session_stop = cancel.child_token();
        let app = secure_router(secure.clone(), target, session_stop.clone());
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = enabled.changed() => {
                    if changed.is_err() || !enabled.borrow_and_update().enabled || enabled.borrow().epoch != epoch { break; }
                }
                _ = connections.join_next(), if !connections.is_empty() => {},
                accepted = listener.accept() => {
                    let Ok((socket, peer)) = accepted else { continue; };
                    // Bound open/slow readers; no request body is buffered here.
                    if connections.len() >= 64 { continue; }
                    let _ = socket.set_nodelay(true);
                    let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
                    connections.spawn(async move {
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(TokioIo::new(socket), service).with_upgrades().await;
                    });
                }
            }
        }
        session_stop.cancel();
        addresses.publish_local_network_listener(None);
        drop(listener);
        // Stop closes existing SSE/media/upload connections as well as the
        // listener. Dropping an axum serve future alone leaves them alive.
        connections.shutdown().await;
        #[cfg(feature = "mdns")]
        if let Some(advertisement) = advertisement { advertisement.shutdown(); }
        if cancel.is_cancelled() { return; }
    }
}

/// An explicitly bound LAN address does not also bind loopback. Give its secure
/// adapters an independently scoped private API target with an actual OS port.
pub(crate) struct PrivateBackend {
    pub address: SocketAddr,
    cancel: CancellationToken,
    task: Option<tokio::task::JoinHandle<()>>,
}
impl PrivateBackend {
    pub async fn start(app: Router) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let address = listener.local_addr()?;
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    _ = stop.cancelled() => break,
                    _ = connections.join_next(), if !connections.is_empty() => {},
                    accepted = listener.accept() => {
                        let Ok((socket, peer)) = accepted else { break; };
                        if !peer.ip().is_loopback() || connections.len() >= 64 { continue; }
                        let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
                        connections.spawn(async move {
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(TokioIo::new(socket), service).await;
                        });
                    }
                }
            }
            connections.shutdown().await;
        });
        Ok(Self { address, cancel, task: Some(task) })
    }
    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() { let _ = task.await; }
    }
}
impl Drop for PrivateBackend {
    fn drop(&mut self) { self.cancel.cancel(); }
}

/// The API remains on the authenticated loopback boundary. LAN and cloud use
/// one authority and the same per-device revocation policy.
pub(crate) fn secure_router(secure: Arc<SecurePairing>, target: SocketAddr, cancel: CancellationToken) -> Router {
    mold_relay::secure::lan_router(mold_relay::secure::HostConfig {
        identity: secure.identity.clone(), target, policy: secure.watch_policy(),
    }, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use tokio::io::AsyncReadExt;

    async fn published(addresses: &ConnectionAddresses) -> SocketAddr {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if let Some(address) = addresses.local_network_listener() {
                    return SocketAddr::new("127.0.0.1".parse().unwrap(), address.port());
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }).await.unwrap()
    }

    #[tokio::test]
    async fn opt_in_exposes_only_secure_upgrade_and_stop_closes_upgraded_sockets() {
        let addresses = Arc::new(ConnectionAddresses::default());
        let secure = Arc::new(SecurePairing::new(mold_secure::Identity::generate().unwrap(), Vec::new()));
        let guard = start(Some(secure), "127.0.0.1:1".parse().unwrap(), addresses.clone(), Vec::new());
        assert!(addresses.local_network_listener().is_none());
        addresses.set_local_network_enabled(true);
        let address = published(&addresses).await;
        let client = reqwest::Client::new();
        assert_eq!(client.get(format!("http://{address}/api/status")).send().await.unwrap().status(), StatusCode::NOT_FOUND);
        let response = client.get(format!("http://{address}/_mold/secure?v=3"))
            .header("connection", "upgrade").header("upgrade", "websocket")
            .header("sec-websocket-version", "13").header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
            .send().await.unwrap();
        assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
        let mut upgraded = response.upgrade().await.unwrap();
        addresses.set_local_network_enabled(false);
        addresses.set_local_network_enabled(true);
        tokio::time::timeout(Duration::from_secs(1), async {
            let mut bytes = [0; 32];
            while matches!(upgraded.read(&mut bytes).await, Ok(n) if n > 0) {}
        }).await.expect("Stop must close detached WebSocket upgrade tasks");
        published(&addresses).await;
        assert!(client.get(format!("http://{address}/api/status")).send().await.is_err());
        guard.shutdown().await;
        assert!(addresses.local_network_listener().is_none());
    }

    #[tokio::test]
    async fn explicit_lan_backend_uses_private_ephemeral_port_and_closes_stalled_requests() {
        let app = Router::new().route("/api/status", axum::routing::get(|ConnectInfo(peer): ConnectInfo<SocketAddr>| async move {
            assert!(peer.ip().is_loopback()); "private"
        })).route("/stalled", axum::routing::get(|| async {
            std::future::pending::<&'static str>().await
        }));
        let backend = PrivateBackend::start(app).await.unwrap();
        let address = backend.address;
        assert!(address.ip().is_loopback());
        assert_ne!(address.port(), 0);
        let client = reqwest::Client::new();
        assert_eq!(client.get(format!("http://{address}/api/status")).send().await.unwrap().text().await.unwrap(), "private");
        let stalled = tokio::spawn(async move { reqwest::get(format!("http://{address}/stalled")).await });
        tokio::time::sleep(Duration::from_millis(25)).await;
        backend.shutdown().await;
        assert!(tokio::time::timeout(Duration::from_secs(1), stalled).await.unwrap().unwrap().is_err());
        assert!(client.get(format!("http://{address}/api/status")).send().await.is_err());
    }

    #[tokio::test]
    async fn an_unauthenticated_engine_never_opens_lan_access() {
        let addresses = Arc::new(ConnectionAddresses::default());
        addresses.set_local_network_enabled(true);
        let _guard = start(None, "127.0.0.1:1".parse().unwrap(), addresses.clone(), Vec::new());
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(addresses.local_network_listener().is_none());
    }
}
