//! Optional, bounded TCP-over-WebSocket relay. No inference dependencies.
pub mod aws;
use anyhow::{bail, Context, Result};
use axum::{
    extract::{
        ws::{Message as AxMessage, WebSocketUpgrade},
        Path, State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use clap::{Args, Subcommand};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot, Mutex, Semaphore},
};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{client::IntoClientRequest, Message},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const CHUNK: usize = 32 * 1024;
const LIMIT: usize = 64;
const ATTACH: Duration = Duration::from_secs(15);
const HEARTBEAT: Duration = Duration::from_secs(15);
const LIVENESS: Duration = Duration::from_secs(45);
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(3600);
/// Application-byte inactivity limit; heartbeat traffic does not extend it.
#[derive(Clone, Copy)]
pub struct RelayOptions {
    pub idle_timeout: Duration,
}
impl Default for RelayOptions {
    fn default() -> Self {
        Self {
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}
impl RelayOptions {
    fn validate(self) -> Result<Self> {
        if self.idle_timeout.is_zero() || self.idle_timeout > Duration::from_secs(86400) {
            bail!("relay idle timeout must be positive and at most 24 hours");
        }
        Ok(self)
    }
}

#[derive(Subcommand)]
pub enum RelayAction {
    Serve(ServeArgs),
    Connect(ConnectArgs),
}
#[derive(Args)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:7682")]
    pub data_bind: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:7681")]
    pub control_bind: SocketAddr,
    #[arg(long, env = "MOLD_RELAY_TOKEN_FILE", value_hint = clap::ValueHint::FilePath)]
    pub token_file: Option<PathBuf>,
    /// Close streams without application bytes for this long (heartbeats excluded).
    #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT.as_secs(), value_parser = clap::value_parser!(u64).range(1..=86400))]
    pub idle_timeout_secs: u64,
}
#[derive(Args)]
pub struct ConnectArgs {
    /// WSS origin; credentials and query strings are forbidden.
    #[arg(long)]
    pub relay_url: String,
    #[arg(long, default_value = "127.0.0.1:7680")]
    pub target: SocketAddr,
    #[arg(long, env = "MOLD_RELAY_TOKEN_FILE", value_hint = clap::ValueHint::FilePath)]
    pub token_file: Option<PathBuf>,
    /// Close streams without application bytes for this long (heartbeats excluded).
    #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT.as_secs(), value_parser = clap::value_parser!(u64).range(1..=86400))]
    pub idle_timeout_secs: u64,
    #[arg(long, value_enum, default_value_t = RelayTransport::Direct)]
    pub transport: RelayTransport,
    #[arg(long)]
    pub allow_insecure_loopback: bool,
}
#[derive(Clone, Copy, clap::ValueEnum)]
pub enum RelayTransport {
    Direct,
    Aws,
}
impl RelayAction {
    pub async fn run(self) -> Result<()> {
        let shutdown = CancellationToken::new();
        let signal = shutdown.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            signal.cancel();
        });
        match self {
            Self::Serve(args) => {
                validate_target(args.data_bind)?;
                validate_target(args.control_bind)?;
                let token = load_token(args.token_file.as_deref())?;
                serve_with_options(
                    TcpListener::bind(args.data_bind).await?,
                    TcpListener::bind(args.control_bind).await?,
                    token,
                    shutdown,
                    RelayOptions {
                        idle_timeout: Duration::from_secs(args.idle_timeout_secs),
                    },
                )
                .await
            }
            Self::Connect(args) => {
                let token = load_token(args.token_file.as_deref())?;
                let options = RelayOptions {
                    idle_timeout: Duration::from_secs(args.idle_timeout_secs),
                };
                match args.transport {
                    RelayTransport::Direct => {
                        connect_with_options(
                            &args.relay_url,
                            args.target,
                            token,
                            args.allow_insecure_loopback,
                            shutdown,
                            options,
                        )
                        .await
                    }
                    RelayTransport::Aws => {
                        aws::connect(
                            &args.relay_url,
                            args.target,
                            token,
                            args.allow_insecure_loopback,
                            shutdown,
                            options,
                        )
                        .await
                    }
                }
            }
        }
    }
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
#[cfg(unix)]
pub fn read_token(path: &std::path::Path) -> Result<String> {
    let file = std::fs::File::open(path).context("cannot open relay token file")?;
    let metadata = file.metadata().context("cannot inspect relay token file")?;
    if !metadata.is_file() {
        bail!("relay token must be a regular owner-only file");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.mode() & 0o077 != 0 || metadata.uid() != rustix::process::geteuid().as_raw() {
            bail!("relay token file must have owner-only permissions");
        }
    }
    use std::io::Read;
    let mut token = String::new();
    file.take(4097)
        .read_to_string(&mut token)
        .context("cannot read relay token file")?;
    validate_token(&token)
}
#[cfg(not(unix))]
pub fn read_token(_path: &std::path::Path) -> Result<String> {
    bail!("token file ownership cannot be verified on this platform; use MOLD_RELAY_TOKEN")
}
fn validate_token(token: &str) -> Result<String> {
    let token = token.trim();
    if token.len() < 32 || token.len() > 4096 || !token.bytes().all(|b| b.is_ascii_graphic()) {
        bail!("relay token must contain 32 to 4096 printable non-space ASCII bytes");
    }
    Ok(token.to_owned())
}
fn load_token(path: Option<&std::path::Path>) -> Result<String> {
    if let Some(path) = path {
        return read_token(path);
    }
    let token = std::env::var("MOLD_RELAY_TOKEN").map_err(|_| {
        anyhow::anyhow!("provide --token-file, MOLD_RELAY_TOKEN_FILE, or MOLD_RELAY_TOKEN")
    })?;
    validate_token(&token)
}

struct Session {
    id: Uuid,
    notices: mpsc::Sender<Uuid>,
    cancel: CancellationToken,
    pending: HashMap<Uuid, oneshot::Sender<axum::extract::ws::WebSocket>>,
}
struct Gateway {
    token: String,
    session: Mutex<Option<Session>>,
    capacity: Arc<Semaphore>,
}
fn authorized(headers: &HeaderMap, token: &str) -> bool {
    // Compare without a mismatch-dependent early return.
    let Some(value) = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
    else {
        return false;
    };
    let mut mismatch = value.len() ^ token.len();
    for (a, b) in value.bytes().zip(token.bytes()) {
        mismatch |= usize::from(a ^ b);
    }
    mismatch == 0
}
async fn control(
    State(state): State<Arc<Gateway>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    if !authorized(&headers, &state.token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut slot = state.session.lock().await;
    if slot.is_some() {
        return StatusCode::CONFLICT.into_response();
    }
    let id = Uuid::new_v4();
    let cancel = CancellationToken::new();
    let (tx, mut rx) = mpsc::channel::<Uuid>(LIMIT);
    *slot = Some(Session {
        id,
        notices: tx,
        cancel: cancel.clone(),
        pending: HashMap::new(),
    });
    drop(slot);
    ws.max_message_size(CHUNK).max_frame_size(CHUNK).on_failed_upgrade({
        let state = state.clone(); let cancel = cancel.clone();
        move |_| { cancel.cancel(); tokio::spawn(async move { clear_session(&state, id).await; }); }
    }).on_upgrade(move |mut socket| async move {
        if !send_bounded(&mut socket, AxMessage::Text(id.to_string().into()), &cancel).await { clear_session(&state, id).await; return; }
        let mut heartbeat = tokio::time::interval(HEARTBEAT);
        let mut last = tokio::time::Instant::now();
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                notice = rx.recv() => match notice {
                    Some(stream) => if !send_bounded(&mut socket, AxMessage::Text(stream.to_string().into()), &cancel).await { break; },
                    None => break,
                },
                message = socket.recv() => match message {
                    Some(Ok(AxMessage::Pong(_))) | Some(Ok(AxMessage::Ping(_))) => last = tokio::time::Instant::now(),
                    Some(Ok(AxMessage::Text(text))) => {
                        let Some(stream) = text.strip_prefix("reject ").and_then(|text| Uuid::parse_str(text).ok()) else { break; };
                        let mut slot = state.session.lock().await;
                        if let Some(session) = slot.as_mut().filter(|s| s.id == id) { session.pending.remove(&stream); }
                    },
                    _ => break,
                },
                _ = heartbeat.tick() => {
                    if last.elapsed() > LIVENESS || !send_bounded(&mut socket, AxMessage::Ping(Vec::new().into()), &cancel).await { break; }
                }
            }
        }
        cancel.cancel();
        clear_session(&state, id).await;
    }).into_response()
}
async fn clear_session(state: &Gateway, id: Uuid) {
    let mut slot = state.session.lock().await;
    if slot.as_ref().is_some_and(|s| s.id == id) {
        if let Some(session) = slot.take() {
            session.cancel.cancel();
        }
    }
}
async fn data(
    State(state): State<Arc<Gateway>>,
    Path((session_id, stream_id)): Path<(Uuid, Uuid)>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    if !authorized(&headers, &state.token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut slot = state.session.lock().await;
    let Some(session) = slot.as_mut().filter(|s| s.id == session_id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(pending) = session.pending.remove(&stream_id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    ws.max_message_size(CHUNK)
        .max_frame_size(CHUNK)
        .on_upgrade(move |socket| async move {
            let _ = pending.send(socket);
        })
        .into_response()
}
async fn health(State(state): State<Arc<Gateway>>) -> &'static str {
    if state.session.lock().await.is_some() {
        "online\n"
    } else {
        "offline\n"
    }
}
/// Run with prebound loopback listeners, allowing ephemeral ports for verification.
pub async fn serve(
    data_listener: TcpListener,
    control_listener: TcpListener,
    token: String,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_with_options(
        data_listener,
        control_listener,
        token,
        shutdown,
        RelayOptions::default(),
    )
    .await
}
/// Run with an explicit inactivity policy (also used by integration fixtures).
pub async fn serve_with_options(
    data_listener: TcpListener,
    control_listener: TcpListener,
    token: String,
    shutdown: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    let options = options.validate()?;
    let token = validate_token(&token)?;
    validate_target(data_listener.local_addr()?)?;
    validate_target(control_listener.local_addr()?)?;
    let state = Arc::new(Gateway {
        token,
        session: Mutex::new(None),
        capacity: Arc::new(Semaphore::new(LIMIT)),
    });
    let router = Router::new()
        .route("/_mold/relay/control", get(control))
        .route("/_mold/relay/data/{session}/{stream}", get(data))
        .route("/_mold/relay/health", get(health))
        .with_state(state.clone());
    let signal = shutdown.clone();
    let server = tokio::spawn(async move {
        axum::serve(control_listener, router)
            .with_graceful_shutdown(signal.cancelled_owned())
            .await
    });
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            incoming = accept_with_retry(|| data_listener.accept(), &shutdown) => {
                let Some((mut tcp, _)) = incoming else { break; };
                let Ok(permit) = state.capacity.clone().try_acquire_owned() else {
                    let _ = tcp.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await; continue;
                };
                let state = state.clone(); let shutdown = shutdown.clone();
                tokio::spawn(proxy_incoming(tcp, state, shutdown, options, permit));
            }
        }
    }
    if let Some(session) = state.session.lock().await.take() {
        session.cancel.cancel();
    }
    server.await??;
    Ok(())
}

async fn proxy_incoming(
    mut tcp: TcpStream,
    state: Arc<Gateway>,
    shutdown: CancellationToken,
    options: RelayOptions,
    permit: tokio::sync::OwnedSemaphorePermit,
) {
    let _permit = permit;
    let stream = Uuid::new_v4();
    let (tx, rx) = oneshot::channel();
    let mut slot = state.session.lock().await;
    let Some(session) = slot.as_mut() else {
        drop(slot);
        let _ = tcp.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
        return;
    };
    let id = session.id;
    let cancel = session.cancel.clone();
    session.pending.insert(stream, tx);
    if session.notices.try_send(stream).is_err() {
        session.pending.remove(&stream);
        drop(slot);
        unavailable(&mut tcp).await;
        return;
    }
    drop(slot);
    let socket = tokio::select! {
        _ = cancel.cancelled() => None,
        _ = shutdown.cancelled() => None,
        result = tokio::time::timeout(ATTACH, rx) => result.ok().and_then(Result::ok),
    };
    let mut slot = state.session.lock().await;
    if let Some(s) = slot.as_mut().filter(|s| s.id == id) {
        s.pending.remove(&stream);
    }
    drop(slot);
    if let Some(socket) = socket {
        let transport = socket.map(|result| result.map(ax_to_wire));
        let transport =
            transport.with(|message: Wire| async { Ok::<_, axum::Error>(wire_to_ax(message)) });
        let _ = bridge(
            tcp,
            Box::pin(transport),
            cancel.child_token(),
            options.idle_timeout,
        )
        .await;
    } else {
        let _ = tcp.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
    }
}

#[derive(Debug)]
enum Wire {
    Bytes(Vec<u8>),
    Eof,
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Close,
    Invalid,
}
fn ax_to_wire(message: AxMessage) -> Wire {
    match message {
        AxMessage::Binary(b) => Wire::Bytes(b.to_vec()),
        AxMessage::Text(t) if t == "eof" => Wire::Eof,
        AxMessage::Ping(b) => Wire::Ping(b.to_vec()),
        AxMessage::Pong(b) => Wire::Pong(b.to_vec()),
        AxMessage::Close(_) => Wire::Close,
        _ => Wire::Invalid,
    }
}
fn wire_to_ax(message: Wire) -> AxMessage {
    match message {
        Wire::Bytes(b) => AxMessage::Binary(b.into()),
        Wire::Eof => AxMessage::Text("eof".into()),
        Wire::Ping(b) => AxMessage::Ping(b.into()),
        Wire::Pong(b) => AxMessage::Pong(b.into()),
        _ => AxMessage::Close(None),
    }
}
fn ws_to_wire(message: Message) -> Wire {
    match message {
        Message::Binary(b) => Wire::Bytes(b.to_vec()),
        Message::Text(t) if t == "eof" => Wire::Eof,
        Message::Ping(b) => Wire::Ping(b.to_vec()),
        Message::Pong(b) => Wire::Pong(b.to_vec()),
        Message::Close(_) => Wire::Close,
        _ => Wire::Invalid,
    }
}
fn wire_to_ws(message: Wire) -> Message {
    match message {
        Wire::Bytes(b) => Message::Binary(b.into()),
        Wire::Eof => Message::Text("eof".into()),
        Wire::Ping(b) => Message::Ping(b.into()),
        Wire::Pong(b) => Message::Pong(b.into()),
        _ => Message::Close(None),
    }
}
async fn bridge<W, E>(
    tcp: TcpStream,
    websocket: W,
    cancel: CancellationToken,
    idle_timeout: Duration,
) -> Result<()>
where
    W: Sink<Wire, Error = E> + Stream<Item = std::result::Result<Wire, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    let (mut read, mut write) = tcp.into_split();
    let (mut sink, mut source) = websocket.split();
    let (events_tx, mut events_rx) = mpsc::channel::<Wire>(8);
    let (activity, mut activity_rx) = tokio::sync::watch::channel(tokio::time::Instant::now());
    let outgoing = async {
        let mut buffer = vec![0; CHUNK];
        let mut local_eof = false;
        let mut peer_eof = false;
        let mut heartbeat = tokio::time::interval(HEARTBEAT);
        loop {
            let message = tokio::select! {
                _ = cancel.cancelled() => return Ok::<_, anyhow::Error>(()),
                result = read.read(&mut buffer), if !local_eof => {
                    let n = result?;
                    if n == 0 { local_eof = true; Wire::Eof }
                    else { activity.send_replace(tokio::time::Instant::now()); Wire::Bytes(buffer[..n].to_vec()) }
                },
                event = events_rx.recv() => match event {
                    Some(Wire::Eof) => { peer_eof = true; if local_eof { cancel.cancel(); return Ok(()); } continue; },
                    Some(event) => event,
                    None => return Ok(()),
                },
                _ = heartbeat.tick() => Wire::Ping(Vec::new()),
            };
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                result = tokio::time::timeout(LIVENESS, sink.send(message)) => { result.context("relay write timed out")??; }
            }
            if local_eof && peer_eof {
                cancel.cancel();
                return Ok(());
            }
        }
    };
    let incoming = async {
        let mut peer_eof = false;
        loop {
            let message = tokio::select! {
                _ = cancel.cancelled() => return Ok::<_, anyhow::Error>(()),
                message = tokio::time::timeout(LIVENESS, source.next()) => message.context("relay heartbeat timed out")?,
            };
            match message {
                Some(Ok(Wire::Bytes(bytes))) if !peer_eof && bytes.len() <= CHUNK => {
                    tokio::select! {
                        _ = cancel.cancelled() => return Ok(()),
                        result = tokio::time::timeout(LIVENESS, write.write_all(&bytes)) => { result.context("target write timed out")??; if !bytes.is_empty() { activity.send_replace(tokio::time::Instant::now()); } }
                    }
                }
                Some(Ok(Wire::Eof)) if !peer_eof => {
                    peer_eof = true;
                    write.shutdown().await?;
                    let _ = events_tx.send(Wire::Eof).await;
                }
                Some(Ok(Wire::Ping(bytes))) => {
                    let _ = events_tx.try_send(Wire::Pong(bytes));
                }
                Some(Ok(Wire::Pong(_))) => {}
                _ => {
                    cancel.cancel();
                    return Ok(());
                }
            }
        }
    };
    let idle = async {
        loop {
            let deadline = *activity_rx.borrow_and_update() + idle_timeout;
            tokio::select! {
                _ = cancel.cancelled() => return Ok::<_, anyhow::Error>(()),
                _ = tokio::time::sleep_until(deadline) => { if activity_rx.borrow().elapsed() >= idle_timeout { cancel.cancel(); return Ok(()); } },
                changed = activity_rx.changed() => { if changed.is_err() { return Ok(()); } },
            }
        }
    };
    tokio::try_join!(outgoing, incoming, idle)?;
    Ok(())
}

fn ws_request(
    url: &url::Url,
    token: &str,
) -> Result<tokio_tungstenite::tungstenite::http::Request<()>> {
    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|_| anyhow::anyhow!("invalid relay request"))?;
    request.headers_mut().insert(
        "authorization",
        format!("Bearer {token}")
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid relay credential"))?,
    );
    Ok(request)
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
/// Maintain an outbound session. No requests are replayed across reconnects.
pub async fn connect(
    endpoint: &str,
    target: SocketAddr,
    token: String,
    allow_loopback_ws: bool,
    shutdown: CancellationToken,
) -> Result<()> {
    connect_with_options(
        endpoint,
        target,
        token,
        allow_loopback_ws,
        shutdown,
        RelayOptions::default(),
    )
    .await
}
/// Connect with an explicit inactivity policy.
pub async fn connect_with_options(
    endpoint: &str,
    target: SocketAddr,
    token: String,
    allow_loopback_ws: bool,
    shutdown: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    let options = options.validate()?;
    let token = validate_token(&token)?;
    let origin = validate_endpoint(endpoint, allow_loopback_ws)?;
    validate_target(target)?;
    // Fail before enrolling if authentication is disabled.
    drop(authenticated_target(target).await?);
    let mut backoff = Duration::from_secs(5);
    loop {
        if shutdown.is_cancelled() {
            return Ok(());
        }
        let started = tokio::time::Instant::now();
        let outcome = connector_session(
            &origin,
            target,
            &token,
            shutdown.clone(),
            options,
            Arc::new(Semaphore::new(LIMIT)),
        )
        .await;
        if started.elapsed() >= Duration::from_secs(60) {
            backoff = Duration::from_secs(5);
        }
        if let Err(error) = outcome {
            if error.to_string() == "relay enrollment rejected" {
                return Err(error);
            }
        }
        tokio::select! { _ = shutdown.cancelled() => return Ok(()), _ = tokio::time::sleep(backoff) => {} }
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}
async fn connector_session(
    origin: &url::Url,
    target: SocketAddr,
    token: &str,
    shutdown: CancellationToken,
    options: RelayOptions,
    capacity: Arc<Semaphore>,
) -> Result<()> {
    let mut control_url = origin.clone();
    control_url.set_path("/_mold/relay/control");
    let (mut socket, _) = tokio::time::timeout(ATTACH, connect_async_with_config(ws_request(&control_url, token)?, Some(websocket_config()), false)).await.map_err(|_| anyhow::anyhow!("relay enrollment timed out"))?.map_err(|error| {
        if matches!(&error, tokio_tungstenite::tungstenite::Error::Http(response) if response.status() == 401 || response.status() == 403) { anyhow::anyhow!("relay enrollment rejected") } else { anyhow::anyhow!("relay enrollment unavailable") }
    })?;
    // The session fence is delivered once before any open notices.
    let session = match tokio::time::timeout(ATTACH, socket.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => {
            Uuid::parse_str(&text).map_err(|_| anyhow::anyhow!("invalid relay session"))?
        }
        _ => bail!("relay session unavailable"),
    };
    let cancel = CancellationToken::new();
    let mut tasks = tokio::task::JoinSet::new();
    let mut heartbeat = tokio::time::interval(HEARTBEAT);
    let mut last = tokio::time::Instant::now();
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = tasks.join_next(), if !tasks.is_empty() => {},
            message = socket.next() => {
                last = tokio::time::Instant::now();
                match message {
                    Some(Ok(Message::Text(text))) => {
                        let Ok(stream) = Uuid::parse_str(&text) else { break; };
                        let Ok(permit) = capacity.clone().try_acquire_owned() else {
                            if !send_bounded(&mut socket, Message::Text(format!("reject {stream}").into()), &shutdown).await { break; }
                            continue;
                        };
                        let mut url = origin.clone(); url.set_path(&format!("/_mold/relay/data/{session}/{stream}"));
                        let token = token.to_owned(); let cancel = cancel.child_token();
                        tasks.spawn(async move {
                            let _permit = permit;
                            let attached = tokio::select! {
                                _ = cancel.cancelled() => return,
                                result = tokio::time::timeout(ATTACH, async {
                                    let tcp = authenticated_target(target).await?;
                                    let (socket, _) = connect_async_with_config(ws_request(&url, &token)?, Some(websocket_config()), false).await.map_err(|_| anyhow::anyhow!("relay stream unavailable"))?;
                                    Ok::<_, anyhow::Error>((tcp, socket))
                                }) => result,
                            };
                            if let Ok(Ok((tcp, socket))) = attached {
                                let transport = socket.map(|result| result.map(ws_to_wire));
                                let transport = transport.with(|message: Wire| async { Ok::<_, tokio_tungstenite::tungstenite::Error>(wire_to_ws(message)) });
                                let _ = bridge(tcp, Box::pin(transport), cancel, options.idle_timeout).await;
                            }
                        });
                    },
                    Some(Ok(Message::Ping(bytes))) => { if !send_bounded(&mut socket, Message::Pong(bytes), &shutdown).await { break; } },
                    Some(Ok(Message::Pong(_))) => {},
                    _ => break,
                }
            },
            _ = heartbeat.tick() => {
                if last.elapsed() > LIVENESS { break; }
                if !send_bounded(&mut socket, Message::Ping(Vec::new().into()), &shutdown).await { break; }
            }
        }
    }
    cancel.cancel();
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    Ok(())
}

/// Run a relay subcommand shared by the standalone and full Mold binaries.
pub async fn run(action: RelayAction) -> Result<()> {
    action.run().await
}

fn websocket_config() -> tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
    tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(CHUNK))
        .max_frame_size(Some(CHUNK))
        .write_buffer_size(CHUNK)
        .max_write_buffer_size(CHUNK * 4)
}

async fn send_bounded<S, M>(sink: &mut S, message: M, cancel: &CancellationToken) -> bool
where
    S: Sink<M> + Unpin,
{
    tokio::select! {
        _ = cancel.cancelled() => false,
        result = tokio::time::timeout(LIVENESS, sink.send(message)) => matches!(result, Ok(Ok(()))),
    }
}

async fn unavailable(tcp: &mut TcpStream) {
    let _ = tokio::time::timeout(
        LIVENESS,
        tcp.write_all(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ),
    )
    .await;
}

async fn accept_with_retry<F, Fut>(
    mut accept: F,
    cancel: &CancellationToken,
) -> Option<(TcpStream, SocketAddr)>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::io::Result<(TcpStream, SocketAddr)>>,
{
    loop {
        let result =
            tokio::select! { _ = cancel.cancelled() => return None, result = accept() => result };
        if let Ok(incoming) = result {
            return Some(incoming);
        }
        // Descriptor exhaustion and aborted connections need a bounded retry,
        // rather than taking down a healthy control listener or spinning.
        tokio::select! { _ = cancel.cancelled() => return None, _ = tokio::time::sleep(Duration::from_millis(250)) => {} }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use std::{
        pin::Pin,
        task::{Context as TaskContext, Poll},
    };
    struct TestSocket {
        incoming: mpsc::UnboundedReceiver<Wire>,
        outgoing: mpsc::UnboundedSender<Wire>,
    }
    impl Stream for TestSocket {
        type Item = std::result::Result<Wire, std::io::Error>;
        fn poll_next(
            mut self: Pin<&mut Self>,
            cx: &mut TaskContext<'_>,
        ) -> Poll<Option<Self::Item>> {
            self.incoming.poll_recv(cx).map(|item| item.map(Ok))
        }
    }
    impl Sink<Wire> for TestSocket {
        type Error = std::io::Error;
        fn poll_ready(
            self: Pin<&mut Self>,
            _: &mut TaskContext<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn start_send(self: Pin<&mut Self>, item: Wire) -> Result<(), Self::Error> {
            self.outgoing
                .send(item)
                .map_err(|_| std::io::ErrorKind::BrokenPipe.into())
        }
        fn poll_flush(
            self: Pin<&mut Self>,
            _: &mut TaskContext<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn poll_close(
            self: Pin<&mut Self>,
            _: &mut TaskContext<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }
    async fn bridge_fixture() -> (
        TcpStream,
        TcpStream,
        TestSocket,
        mpsc::UnboundedSender<Wire>,
        mpsc::UnboundedReceiver<Wire>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        (
            client,
            server,
            TestSocket {
                incoming: input_rx,
                outgoing: output_tx,
            },
            input_tx,
            output_rx,
        )
    }
    #[test]
    fn default_idle_budget_allows_one_hour_synchronous_work() {
        assert_eq!(
            RelayOptions::default().idle_timeout,
            Duration::from_secs(3600)
        );
    }
    #[tokio::test]
    async fn exhausted_notice_queue_returns_http_unavailable() {
        let (mut client, server, _socket, _input, _output) = bridge_fixture().await;
        let (notice_tx, _notice_rx) = mpsc::channel(1);
        notice_tx.try_send(Uuid::new_v4()).unwrap();
        let cancel = CancellationToken::new();
        let state = Arc::new(Gateway {
            token: String::new(),
            capacity: Arc::new(Semaphore::new(1)),
            session: Mutex::new(Some(Session {
                id: Uuid::new_v4(),
                notices: notice_tx,
                cancel: cancel.clone(),
                pending: HashMap::new(),
            })),
        });
        let permit = state.capacity.clone().acquire_owned().await.unwrap();
        let task = tokio::spawn(proxy_incoming(
            server,
            state,
            cancel,
            RelayOptions::default(),
            permit,
        ));
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 503"),
            "full notice queue silently dropped HTTP connection"
        );
        task.await.unwrap();
    }
    #[tokio::test]
    async fn transient_accept_failures_retry_with_backoff_and_shutdown_is_prompt() {
        let (_client, server, _socket, _input, _output) = bridge_fixture().await;
        let address = server.peer_addr().unwrap();
        let mut outcomes = std::collections::VecDeque::from([
            Err(std::io::Error::from(std::io::ErrorKind::ConnectionAborted)),
            Err(std::io::Error::from_raw_os_error(24)),
            Ok((server, address)),
        ]);
        let started = tokio::time::Instant::now();
        let received = accept_with_retry(
            || {
                let item = outcomes.pop_front().unwrap();
                async { item }
            },
            &CancellationToken::new(),
        )
        .await;
        assert!(received.is_some(), "accept error stopped listener");
        assert!(
            started.elapsed() >= Duration::from_millis(400),
            "failed accept retries spun without backoff"
        );
        let cancel = CancellationToken::new();
        let signal = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            signal.cancel();
        });
        assert!(tokio::time::timeout(
            Duration::from_millis(100),
            accept_with_retry(
                || async { Err(std::io::Error::from_raw_os_error(24)) },
                &cancel
            )
        )
        .await
        .unwrap()
        .is_none());
    }
    #[tokio::test]
    async fn capacity_rejects_only_the_new_stream_and_keeps_control_alive() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = url::Url::parse(&format!("ws://{}", listener.local_addr().unwrap())).unwrap();
        let stream = Uuid::new_v4();
        let session = Uuid::new_v4();
        let (result_tx, result_rx) = oneshot::channel();
        let result_tx = Arc::new(Mutex::new(Some(result_tx)));
        let app = Router::new().route(
            "/_mold/relay/control",
            get(move |ws: WebSocketUpgrade| {
                let result_tx = result_tx.clone();
                async move {
                    ws.on_upgrade(move |mut socket| async move {
                        socket
                            .send(AxMessage::Text(session.to_string().into()))
                            .await
                            .unwrap();
                        socket
                            .send(AxMessage::Text(stream.to_string().into()))
                            .await
                            .unwrap();
                        let rejected = loop {
                            match socket.recv().await {
                                Some(Ok(AxMessage::Text(text))) => {
                                    break text == format!("reject {stream}")
                                }
                                Some(Ok(AxMessage::Ping(_))) | Some(Ok(AxMessage::Pong(_))) => {}
                                _ => break false,
                            }
                        };
                        if rejected {
                            socket
                                .send(AxMessage::Ping(b"alive".to_vec().into()))
                                .await
                                .unwrap();
                            let alive = loop {
                                match socket.recv().await {
                                    Some(Ok(AxMessage::Pong(bytes)))
                                        if bytes.as_ref() == b"alive" =>
                                    {
                                        break true
                                    }
                                    Some(Ok(AxMessage::Ping(_))) | Some(Ok(AxMessage::Pong(_))) => {
                                    }
                                    _ => break false,
                                }
                            };
                            if let Some(tx) = result_tx.lock().await.take() {
                                let _ = tx.send(alive);
                            }
                        } else if let Some(tx) = result_tx.lock().await.take() {
                            let _ = tx.send(false);
                        }
                        let _ = socket.close().await;
                    })
                }
            }),
        );
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        });
        let task = tokio::spawn(connector_session_owned(
            origin,
            "127.0.0.1:1".parse().unwrap(),
            shutdown.clone(),
            Arc::new(Semaphore::new(0)),
        ));
        assert!(
            tokio::time::timeout(Duration::from_secs(2), result_rx)
                .await
                .unwrap()
                .unwrap(),
            "capacity overflow killed control session"
        );
        task.await.unwrap().unwrap();
        shutdown.cancel();
    }
    async fn connector_session_owned(
        origin: url::Url,
        target: SocketAddr,
        shutdown: CancellationToken,
        capacity: Arc<Semaphore>,
    ) -> Result<()> {
        connector_session(
            &origin,
            target,
            "0123456789012345678901234567890123456789",
            shutdown,
            RelayOptions::default(),
            capacity,
        )
        .await
    }
    #[tokio::test]
    async fn application_bytes_in_either_direction_extend_idle_deadline() {
        for outgoing in [true, false] {
            let (mut client, server, socket, input, _output) = bridge_fixture().await;
            let task = tokio::spawn(bridge(
                server,
                socket,
                CancellationToken::new(),
                Duration::from_millis(100),
            ));
            for _ in 0..3 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                if outgoing {
                    client.write_all(b"progress").await.unwrap();
                } else {
                    input.send(Wire::Bytes(b"progress".to_vec())).unwrap();
                }
                tokio::task::yield_now().await;
                assert!(
                    !task.is_finished(),
                    "application progress did not extend idle deadline"
                );
            }
            tokio::time::timeout(Duration::from_millis(300), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        }
    }
    #[tokio::test]
    async fn duplicate_eof_or_bytes_after_eof_fail_closed() {
        for invalid in [Wire::Eof, Wire::Bytes(b"invalid after EOF".to_vec())] {
            let (_client, server, socket, input, _output) = bridge_fixture().await;
            let task = tokio::spawn(bridge(
                server,
                socket,
                CancellationToken::new(),
                Duration::from_secs(300),
            ));
            input.send(Wire::Eof).unwrap();
            input.send(invalid).unwrap();
            tokio::time::timeout(Duration::from_millis(200), task)
                .await
                .expect("invalid post-EOF frame kept tunnel open")
                .unwrap()
                .unwrap();
        }
    }
    #[tokio::test]
    async fn heartbeats_do_not_extend_application_idle_deadline() {
        let (_client, server, socket, input, _output) = bridge_fixture().await;
        let task = tokio::spawn(bridge(
            server,
            socket,
            CancellationToken::new(),
            Duration::from_millis(50),
        ));
        let heartbeat = tokio::spawn(async move {
            loop {
                if input.send(Wire::Pong(Vec::new())).is_err()
                    || input.send(Wire::Bytes(Vec::new())).is_err()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        tokio::time::timeout(Duration::from_millis(200), task)
            .await
            .expect("heartbeat traffic kept idle stream alive")
            .unwrap()
            .unwrap();
        heartbeat.abort();
    }
    #[tokio::test]
    async fn half_closed_stream_still_handles_ping_and_close() {
        let (_client, server, socket, input, mut output) = bridge_fixture().await;
        let cancel = CancellationToken::new();
        let task = tokio::spawn(bridge(server, socket, cancel, Duration::from_secs(300)));
        input.send(Wire::Eof).unwrap();
        input.send(Wire::Ping(b"after-eof".to_vec())).unwrap();
        loop {
            let message = tokio::time::timeout(Duration::from_millis(200), output.recv())
                .await
                .expect("half-close stopped polling WS")
                .unwrap();
            if matches!(message,Wire::Pong(ref bytes) if bytes==b"after-eof") {
                break;
            }
        }
        input.send(Wire::Close).unwrap();
        tokio::time::timeout(Duration::from_millis(200), task)
            .await
            .expect("WS Close did not stop half-closed tunnel")
            .unwrap()
            .unwrap();
    }
}
