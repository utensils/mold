//! GPU-independent end-to-end encrypted host connector. No legacy proxy mode.
pub mod aws;
pub mod secure;
use anyhow::{bail, Context, Result};
#[cfg(feature = "cli")]
use clap::{Args, Subcommand};
#[cfg(feature = "cli")]
use std::sync::Arc;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};
#[cfg(feature = "cli")]
use tokio_util::sync::CancellationToken;
fn ensure_tls_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
}
#[cfg(feature = "cli")]
#[derive(Subcommand)]
pub enum RelayAction {
    /// Connect this authenticated host to the opaque Lambda relay using its server identity.
    Connect(ConnectArgs),
}
#[cfg(feature = "cli")]
#[derive(Args)]
pub struct ConnectArgs {
    /// WSS URL discovered from the trusted relay; never accepts credentials or query values.
    #[arg(long)]
    pub relay_url: String,
    /// Existing server identity. Defaults to $MOLD_HOME/secure/host.identity.
    #[arg(long,env="MOLD_SECURE_IDENTITY_FILE",value_hint=clap::ValueHint::FilePath)]
    pub identity_file: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:7680")]
    pub target: SocketAddr,
    /// Permit ws:// on numeric loopback only for local protocol fixtures.
    #[arg(long)]
    pub allow_insecure_loopback: bool,
}
#[cfg(feature = "cli")]
impl RelayAction {
    pub async fn run(self) -> Result<()> {
        let shutdown = CancellationToken::new();
        let signal = shutdown.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            signal.cancel();
        });
        match self {
            Self::Connect(args) => {
                let path = args.identity_file.unwrap_or_else(default_identity_path);
                let identity = Arc::new(read_identity(&path)?);
                drop(authenticated_target(args.target).await?);
                let policy =
                    secure::watch_policy(identity.clone(), args.target, shutdown.clone()).await?;
                secure::connect(
                    &args.relay_url,
                    secure::HostConfig {
                        identity,
                        target: args.target,
                        policy,
                    },
                    shutdown,
                    args.allow_insecure_loopback,
                )
                .await
            }
        }
    }
}
pub fn default_identity_path() -> PathBuf {
    std::env::var_os("MOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(".mold")
        })
        .join("secure/host.identity")
}
/// Load the server-created identity; never create a different cloud identity.
pub fn read_identity(path: &Path) -> Result<mold_secure::Identity> {
    mold_secure::identity_file::read_host_identity(path)
        .context("cannot read server secure identity; start an upgraded server first")
}
#[cfg(feature = "cli")]
pub async fn run(action: RelayAction) -> Result<()> {
    action.run().await
}
/// Only numeric loopback sockets are valid targets and listeners.
pub fn validate_target(target: SocketAddr) -> Result<()> {
    if !target.ip().is_loopback() {
        bail!("relay sockets must use a numeric loopback address");
    }
    Ok(())
}
pub fn validate_endpoint(endpoint: &str, allow_loopback_ws: bool) -> Result<url::Url> {
    let url = url::Url::parse(endpoint).map_err(|_| anyhow::anyhow!("invalid relay endpoint"))?;
    let loopback = url
        .host_str()
        .and_then(|h| h.trim_matches(['[', ']']).parse::<std::net::IpAddr>().ok())
        .is_some_and(|ip| ip.is_loopback());
    if !(url.scheme() == "wss" || (url.scheme() == "ws" && allow_loopback_ws && loopback))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        bail!("relay endpoint must be a credential-free WSS origin (explicit loopback WS is allowed for development)");
    }
    Ok(url)
}
/// Probe and reuse the exact target TCP connection, fencing listener changes.
/// Fail closed unless a bounded HTTP/1.1 401 has unambiguous framing and permits
/// connection reuse. Mold's authentication response uses Content-Length.
pub async fn authenticated_target(target: SocketAddr) -> Result<TcpStream> {
    validate_target(target)?;
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut socket = TcpStream::connect(target).await.context("target unavailable")?;
        socket.write_all(format!("GET /api/status HTTP/1.1\r\nHost: {target}\r\nConnection: keep-alive\r\nAccept-Encoding: identity\r\n\r\n").as_bytes()).await.context("target auth probe failed")?;
        let mut header = Vec::new();
        let mut byte = [0];
        while !header.ends_with(b"\r\n\r\n") {
            if header.len() >= 16 * 1024 { bail!("target auth response headers exceed limit"); }
            socket.read_exact(&mut byte).await.context("target auth probe failed")?;
            header.push(byte[0]);
        }
        let header = std::str::from_utf8(&header).context("invalid target auth response")?;
        let mut lines = header.split("\r\n");
        let status = lines.next().unwrap_or_default().split_whitespace().collect::<Vec<_>>();
        if status.len() < 2 || status[0] != "HTTP/1.1" || status[1] != "401" { bail!("target must require authentication: unauthenticated status probe must return HTTP/1.1 401"); }
        let mut length = None;
        for line in lines.filter(|line| !line.is_empty()) {
            if line.starts_with([' ', '\t']) { bail!("invalid target auth response framing"); }
            let Some((name, value)) = line.split_once(':') else { bail!("invalid target auth response framing"); };
            let value = value.trim();
            if name.eq_ignore_ascii_case("transfer-encoding") || (name.eq_ignore_ascii_case("connection") && value.split(',').any(|v| v.trim().eq_ignore_ascii_case("close"))) { bail!("target auth response cannot reuse connection"); }
            if name.eq_ignore_ascii_case("content-length") {
                if length.is_some() || value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) { bail!("ambiguous target auth response framing"); }
                let n = value.parse::<usize>().context("invalid target auth response length")?;
                if n > 64 * 1024 { bail!("target auth response body exceeds limit"); }
                length = Some(n);
            }
        }
        let length = length.context("target auth response requires Content-Length")?;
        let mut body = vec![0; length];
        socket.read_exact(&mut body).await.context("target auth probe failed")?;
        Ok(socket)
    }).await.context("target authentication timed out")?
}
