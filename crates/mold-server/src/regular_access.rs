//! Explicit embedded ordinary HTTP listener, independent of secure LAN/cloud.
use crate::{connections::RegularNetworkStatus, ConnectionAddresses};
use axum::{extract::ConnectInfo, Extension, Router};
use hyper_util::{rt::TokioIo, service::TowerToHyperService};
use std::sync::Arc;
use tokio::{net::TcpListener, task::JoinSet};
use tokio_util::sync::CancellationToken;

pub(crate) struct RegularAccess {
    stop: CancellationToken,
    addresses: Arc<ConnectionAddresses>,
    task: Option<tokio::task::JoinHandle<()>>,
}
impl RegularAccess {
    pub async fn shutdown(mut self) {
        self.addresses.set_regular_network_enabled(false);
        self.stop.cancel();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}
impl Drop for RegularAccess {
    fn drop(&mut self) {
        self.addresses.set_regular_network_enabled(false);
        self.stop.cancel();
    }
}
pub(crate) fn start(app: Router, addresses: Arc<ConnectionAddresses>) -> RegularAccess {
    let stop = CancellationToken::new();
    let task = tokio::spawn(run(
        app.layer(Extension(crate::auth::RegularApiAllowed)),
        addresses.clone(),
        stop.clone(),
    ));
    RegularAccess {
        stop,
        addresses,
        task: Some(task),
    }
}
async fn run(app: Router, addresses: Arc<ConnectionAddresses>, stop: CancellationToken) {
    let mut enabled = addresses.watch_regular_network();
    loop {
        while !enabled.borrow_and_update().enabled {
            tokio::select! { _ = stop.cancelled() => return, changed = enabled.changed() => if changed.is_err() { return } }
        }
        let epoch = enabled.borrow().epoch;
        let listener = match TcpListener::bind(("0.0.0.0", 0)).await {
            Ok(listener) => listener,
            Err(_) => {
                addresses.publish_regular_network(RegularNetworkStatus {
                    epoch,
                    error: Some("Could not bind the Regular access listener".into()),
                    address: None,
                });
                tokio::select! { _ = stop.cancelled() => return, _ = enabled.changed() => {} }
                continue;
            }
        };
        let Ok(address) = listener.local_addr() else {
            return;
        };
        if stop.is_cancelled() || !enabled.borrow().enabled || enabled.borrow().epoch != epoch {
            continue;
        }
        addresses.publish_regular_network(RegularNetworkStatus {
            epoch,
            address: Some(address),
            error: None,
        });
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                _ = stop.cancelled() => break,
                changed = enabled.changed() => if changed.is_err() || !enabled.borrow_and_update().enabled || enabled.borrow().epoch != epoch { break },
                _ = connections.join_next(), if !connections.is_empty() => {},
                accepted = listener.accept() => {
                    let Ok((socket,peer)) = accepted else { break };
                    if connections.len() >= 64 { continue }
                    let _ = socket.set_nodelay(true);
                    let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
                    connections.spawn(async move { let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(socket),service).await; });
                }
            }
        }
        drop(listener);
        connections.shutdown().await;
        addresses.publish_regular_network(RegularNetworkStatus {
            epoch,
            address: None,
            error: None,
        });
        if stop.is_cancelled() {
            return;
        }
    }
}
