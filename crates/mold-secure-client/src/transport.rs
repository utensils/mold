use anyhow::{bail, Context, Result};
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use mold_secure::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::{mpsc, oneshot, Mutex, Semaphore};
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
type Socket = crate::object_socket::ObjectSocket;
#[derive(Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub url: String,
    pub kind: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct PairedHost {
    pub protocol: u8,
    pub origin: String,
    pub websocket: String,
    #[serde(default)]
    pub endpoints: Vec<Endpoint>,
    pub namespace: String,
    pub host_key: String,
    pub host_noise_key: String,
    pub host_binding_signature: String,
    pub device_identity: String,
    pub api_key: String,
    pub grant: DeviceGrant,
}
impl PairedHost {
    pub fn validate(&self) -> Result<()> {
        if self.protocol != 3 {
            bail!("Pair this host again: end-to-end encrypted protocol3 is required")
        }
        let host = decode_hex::<32>(&self.host_key)?;
        if namespace(&host) != self.namespace {
            bail!("invalid pinned host identity")
        }
        Binding {
            signing_key: self.host_key.clone(),
            noise_key: self.host_noise_key.clone(),
            signature: self.host_binding_signature.clone(),
        }
        .bytes()?;
        let device = Identity::from_bytes(&decode_hex::<64>(&self.device_identity)?)?;
        self.grant
            .verify(&host, &device.public_binding(), now(), self.grant.epoch)?;
        validate_websocket(&self.websocket)?;
        let origin = reqwest::Url::parse(&self.origin)?;
        if origin.origin().ascii_serialization() != self.origin
            || !origin.username().is_empty()
            || origin.password().is_some()
        {
            bail!("invalid paired origin")
        }
        Ok(())
    }
}
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn validate_websocket(value: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(value)?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.query().is_some()
    {
        bail!("secure endpoint must not contain credentials or query")
    }
    let lan = url.host_str().is_some_and(|s| {
        s == "localhost"
            || s.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok()
    });
    if url.scheme() != "wss" && !(url.scheme() == "ws" && lan) {
        bail!("secure endpoint requires WSS or pinned LAN WS")
    }
    url.set_query(Some("v=3"));
    Ok(url)
}
#[derive(Clone)]
pub struct SecureClient {
    host: Arc<std::sync::RwLock<PairedHost>>,
    renewal: Arc<Mutex<()>>,
    stop: CancellationToken,
    connection: Arc<Mutex<Option<Arc<Connection>>>>,
}
struct Connection {
    commands: mpsc::Sender<Command>,
    dispatch: Mutex<()>,
    next: AtomicU32,
    alive: AtomicBool,
    cancel: CancellationToken,
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
struct StreamState {
    head: Option<oneshot::Sender<Result<Headers>>>,
    body: crate::BodySender,
    upload_credit: Arc<Semaphore>,
    receive_credit: ReceiveCredit,
    cancel: CancellationToken,
    head_seen: bool,
}
impl StreamState {
    fn fail(mut self, message: &'static str) {
        self.cancel.cancel();
        let error = anyhow::anyhow!(message);
        if let Some(head) = self.head.take() {
            let _ = head.send(Err(error));
        } else {
            let _ = self.body.try_send(Err(error));
        }
    }
}
// Parent cancellation can make the timer ready before the connection-stop arm.
// Only an authenticated End may close a live response without an error.
fn cancel_stream(streams: &mut BTreeMap<u32, StreamState>, id: u32) {
    if let Some(state) = streams.remove(&id) {
        state.fail("secure request cancelled; mutation outcome may be unknown");
    }
}
enum Command {
    Start {
        open: Open,
        id: u32,
        state: StreamState,
    },
    Record(Record),
    Consumed(u32, u32),
}
struct ResponseGuard {
    cancel: CancellationToken,
    _connection: Arc<Connection>,
}
impl Drop for ResponseGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
impl SecureClient {
    pub fn new(host: PairedHost) -> Result<Self> {
        host.validate()?;
        Ok(Self {
            host: Arc::new(std::sync::RwLock::new(host)),
            renewal: Default::default(),
            stop: CancellationToken::new(),
            connection: Default::default(),
        })
    }
    pub fn close(&self) {
        self.stop.cancel();
    }
    async fn connection(&self) -> Result<Arc<Connection>> {
        if self.stop.is_cancelled() {
            bail!("secure host was forgotten or replaced")
        }
        let mut saved = self.connection.lock().await;
        if let Some(c) = saved.as_ref().filter(|c| c.alive.load(Ordering::Acquire)) {
            return Ok(c.clone());
        }
        let host = self.snapshot();
        let (socket, session, wire) = tokio::select! { _=self.stop.cancelled()=>bail!("secure host was forgotten or replaced"), result=admit_best(&host,None)=>result? };
        let (tx, rx) = mpsc::channel(64);
        let connection = Arc::new(Connection {
            commands: tx,
            dispatch: Mutex::new(()),
            next: AtomicU32::new(1),
            alive: AtomicBool::new(true),
            cancel: self.stop.child_token(),
        });
        let weak = Arc::downgrade(&connection);
        let stop = connection.cancel.clone();
        tokio::spawn(async move {
            let _ = drive(socket, session, wire, rx, stop).await;
            if let Some(c) = weak.upgrade() {
                c.alive.store(false, Ordering::Release)
            }
        });
        *saved = Some(connection.clone());
        Ok(connection)
    }
    fn snapshot(&self) -> PairedHost {
        self.host.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
    async fn renew_if_due(&self) -> Result<()> {
        if self.snapshot().grant.expires_at > now() + 7 * 86400 {
            return Ok(());
        }
        let _renewal = self.renewal.lock().await;
        let mut host = self.snapshot();
        if host.grant.expires_at > now() + 7 * 86400 {
            return Ok(());
        }
        let request = reqwest::Client::new()
            .post(format!("{}/api/pairing/secure/renew", host.origin))
            .build()?;
        let value =
            bounded_authority_json(self.send_inner(request).await?.error_for_status()?).await?;
        let grant: DeviceGrant = serde_json::from_value(value["grant"].clone())?;
        if grant.id != host.grant.id || grant.epoch != host.grant.epoch {
            bail!("secure grant renewal changed identity")
        }
        host.grant = grant;
        host.validate()?;
        crate::store::save_renewed_pairing(&host)?;
        *self.host.write().unwrap_or_else(|e| e.into_inner()) = host;
        Ok(())
    }
    pub async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response> {
        self.renew_if_due()
            .await
            .context("secure grant renewal failed before request dispatch")?;
        self.send_inner(request).await
    }
    async fn send_inner(&self, mut request: reqwest::Request) -> Result<reqwest::Response> {
        let host = self.snapshot();
        let connection = self.connection().await?;
        // Hold through Start enqueue: another task cannot publish a newer ID
        // while this request is validating or waiting for dispatch capacity.
        let dispatch = connection.dispatch.lock().await;
        let id = connection.next.fetch_add(2, Ordering::Relaxed);
        if id == 0 || id > u32::MAX - 2 {
            bail!("secure stream IDs exhausted")
        }
        if request.url().origin().ascii_serialization() != host.origin {
            bail!("secure request origin mismatch")
        }
        let length = request
            .body()
            .and_then(|b| b.as_bytes())
            .map(|v| v.len() as u64)
            .or_else(|| {
                request
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
            });
        let mut headers = request
            .headers()
            .iter()
            .filter(|(k, _)| *k != http::header::HOST)
            .map(|(k, v)| Ok((k.to_string(), v.to_str()?.to_string())))
            .collect::<Result<Vec<_>>>()?;
        headers.retain(|(k, _)| {
            !k.eq_ignore_ascii_case("x-api-key") && !k.eq_ignore_ascii_case("authorization")
        });
        headers.push(("x-api-key".into(), host.api_key.clone()));
        let path = format!(
            "{}{}",
            request.url().path(),
            request
                .url()
                .query()
                .map(|q| format!("?{q}"))
                .unwrap_or_default()
        );
        let open = Open {
            method: request.method().as_str().into(),
            path,
            headers,
            length,
        };
        Record::Open {
            stream_id: id,
            request: open.clone(),
        }
        .validate()?;
        let (headtx, headrx) = oneshot::channel();
        let (bodytx, bodyrx) = crate::body_channel();
        let credit = Arc::new(Semaphore::new(STREAM_WINDOW as usize));
        let cancel = connection.cancel.child_token();
        connection
            .commands
            .send(Command::Start {
                open,
                id,
                state: StreamState {
                    head: Some(headtx),
                    body: bodytx,
                    upload_credit: credit.clone(),
                    receive_credit: ReceiveCredit::default(),
                    cancel: cancel.clone(),
                    head_seen: false,
                },
            })
            .await
            .context("secure connection closed before dispatch")?;
        drop(dispatch);
        let commands = connection.commands.clone();
        let upload_cancel = cancel.clone();
        let body = request.body_mut().take();
        tokio::spawn(async move {
            let result = tokio::select! {_=upload_cancel.cancelled()=>Ok(()),r=upload(id,body,credit,commands.clone())=>r};
            if result.is_err() {
                upload_cancel.cancel();
            }
        });
        let guard = ResponseGuard {
            cancel: cancel.clone(),
            _connection: connection.clone(),
        };
        let head = if let Some(duration) = request.timeout() {
            tokio::time::timeout(*duration, headrx)
                .await
                .context("secure response timed out; mutation outcome may be unknown")??
        } else {
            headrx
                .await
                .context("secure response lost; mutation outcome may be unknown")?
        }?;
        let commands = connection.commands.clone();
        let body = futures_util::stream::unfold(
            (bodyrx, commands, guard),
            move |(mut receiver, commands, guard)| async move {
                let bytes = receiver.recv().await?;
                if let Ok(bytes) = &bytes {
                    if commands
                        .send(Command::Consumed(id, bytes.len() as u32))
                        .await
                        .is_err()
                    {
                        return Some((
                            Err(anyhow::anyhow!("secure connection lost")),
                            (receiver, commands, guard),
                        ));
                    }
                }
                Some((bytes, (receiver, commands, guard)))
            },
        );
        let mut response = http::Response::builder().status(head.status);
        for (name, value) in head.headers {
            response = response.header(name, value);
        }
        Ok(reqwest::Response::from(
            response.body(reqwest::Body::wrap_stream(body))?,
        ))
    }
}
async fn upload(
    id: u32,
    body: Option<reqwest::Body>,
    credit: Arc<Semaphore>,
    commands: mpsc::Sender<Command>,
) -> Result<()> {
    let mut total = 0u64;
    if let Some(body) = body {
        let stream = body.into_data_stream();
        tokio::pin!(stream);
        while let Some(bytes) = stream.next().await {
            let bytes = bytes?;
            total += bytes.len() as u64;
            if total > MAX_UPLOAD {
                bail!("secure upload exceeds64MiB")
            }
            for chunk in bytes.chunks(DATA_CHUNK) {
                let permit = credit.acquire_many(chunk.len() as u32).await?;
                permit.forget();
                commands
                    .send(Command::Record(Record::Data {
                        stream_id: id,
                        bytes: chunk.to_vec(),
                    }))
                    .await?;
            }
        }
    }
    commands
        .send(Command::Record(Record::End { stream_id: id }))
        .await?;
    Ok(())
}
#[derive(Clone, Serialize, Deserialize)]
pub struct PairingInvite {
    #[serde(rename = "version", alias = "protocol")]
    pub protocol: u8,
    #[serde(rename = "base_url", alias = "origin")]
    pub origin: String,
    #[serde(default, rename = "websocket_url", alias = "websocket")]
    pub websocket: Option<String>,
    #[serde(default)]
    pub endpoints: Vec<Endpoint>,
    pub namespace: String,
    pub host_key: String,
    pub host_noise_key: String,
    pub host_binding_signature: String,
    pub invite_seed: String,
    pub invite: Invitation,
}
async fn read_json(socket: &mut Socket) -> Result<serde_json::Value> {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(15), socket.next())
            .await?
            .context("secure socket closed")??;
        match message {
            Message::Text(text) => {
                if text.len() > 24576 {
                    bail!("secure frame too large")
                }
                let value: serde_json::Value = serde_json::from_str(&text)?;
                if value["v"] != 3 {
                    bail!("secure protocol3 required")
                }
                return Ok(value);
            }
            Message::Ping(data) => socket.send(Message::Pong(data)).await?,
            Message::Pong(_) => {}
            _ => bail!("secure socket closed"),
        }
    }
}
fn direct_endpoint(origin: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(origin)?;
    if url.origin().ascii_serialization() != origin {
        bail!("invalid direct endpoint")
    }
    let scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        _ => bail!("invalid direct endpoint scheme"),
    };
    url.set_scheme(scheme)
        .map_err(|_| anyhow::anyhow!("invalid endpoint"))?;
    url.set_path("/_mold/secure");
    validate_websocket(url.as_str())?;
    Ok(url.to_string())
}
#[derive(Debug)]
pub(crate) struct AdmissionBusy;
impl std::fmt::Display for AdmissionBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("secure gateway temporarily busy")
    }
}
impl std::error::Error for AdmissionBusy {}
async fn retry_admission<T, F, Fut>(mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    for retry in 0..3 {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                let busy = error.chain().any(|cause| {
                    cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
                        io.get_ref()
                            .is_some_and(|inner| inner.is::<AdmissionBusy>())
                    })
                });
                if !busy || retry == 2 {
                    return Err(error);
                }
                let jitter = u64::from(uuid::Uuid::new_v4().as_bytes()[0]) % 126;
                tokio::time::sleep(Duration::from_millis((250_u64 << retry) + jitter)).await;
            }
        }
    }
    unreachable!("bounded admission attempts return")
}
async fn admit_best(
    host: &PairedHost,
    invite: Option<&PairingInvite>,
) -> Result<(Socket, Session, ReliableChannel)> {
    let mut endpoints = host
        .endpoints
        .iter()
        .filter(|e| e.kind != "relay")
        .filter_map(|e| direct_endpoint(&e.url).ok())
        .take(2)
        .collect::<Vec<_>>();
    let has_direct = !endpoints.is_empty();
    if !endpoints.contains(&host.websocket) {
        endpoints.push(host.websocket.clone());
    }
    let mut attempts = futures_util::stream::FuturesUnordered::new();
    for endpoint in endpoints {
        let cloud = endpoint == host.websocket && has_direct;
        let mut candidate = host.clone();
        candidate.websocket = endpoint;
        let invite = invite.cloned();
        attempts.push(async move {
            if cloud {
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            tokio::time::timeout(
                Duration::from_secs(15),
                retry_admission(|| admit(&candidate, invite.as_ref())),
            )
            .await
            .context("secure admission timed out")?
        });
    }
    let mut last = None;
    while let Some(result) = attempts.next().await {
        match result {
            Ok(value) => return Ok(value),
            Err(error) => last = Some(error),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no secure address")))
        .context("secure admission failed before request dispatch")
}
async fn admit(
    host: &PairedHost,
    invite: Option<&PairingInvite>,
) -> Result<(Socket, Session, ReliableChannel)> {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let identity = Identity::from_bytes(&decode_hex::<64>(&host.device_identity)?)?;
    let url = validate_websocket(&host.websocket)?;
    let (socket, _) = tokio_tungstenite::connect_async_with_config(
        url.as_str(),
        Some(
            tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
                .max_message_size(Some(24576))
                .max_frame_size(Some(24576)),
        ),
        false,
    )
    .await?;
    let mut socket = crate::object_socket::ObjectSocket::new(socket, host.namespace.clone(), false);
    let role = if invite.is_some() { "invite" } else { "device" };
    socket.send(Message::Text(serde_json::json!({"a":"hello","v":3,"role":role,"namespace":host.namespace,"hostKey":host.host_key}).to_string().into())).await?;
    let challenge = read_json(&mut socket).await?;
    if challenge["a"] != "challenge" {
        bail!("secure challenge required")
    }
    let expires = challenge["expiresAt"]
        .as_u64()
        .context("challenge expiration missing")?;
    if expires <= now() || expires > now() + 120 {
        bail!("secure challenge expiration invalid")
    }
    let bytes = join_bytes(
        &host.namespace,
        if invite.is_some() { 3 } else { 2 },
        challenge["connectionId"]
            .as_str()
            .context("missing connection ID")?,
        challenge["nonce"].as_str().context("missing nonce")?,
        &identity.signing_public(),
        expires,
    )
    .context("invalid secure relay challenge")?;
    let signature = if let Some(invite) = invite {
        let seed = decode_hex::<32>(&invite.invite_seed)?;
        let mut secret = zeroize::Zeroizing::new([0; 64]);
        secret[..32].copy_from_slice(&seed);
        Identity::from_bytes(&secret[..])?.sign(&bytes)
    } else {
        identity.sign(&bytes)
    };
    let mut join = serde_json::json!({"a":"join","v":3,"binding":identity.public_binding(),"signature":hex::encode(signature)});
    if let Some(invite) = invite {
        join["invite"] = serde_json::to_value(&invite.invite)?
    } else {
        join["grant"] = serde_json::to_value(&host.grant)?
    }
    socket.send(Message::Text(join.to_string().into())).await?;
    let mut ready = None;
    let mut accept = None;
    while ready.is_none() || accept.is_none() {
        let value = read_json(&mut socket).await?;
        match value["a"].as_str() {
            Some("ready") => ready = Some(value),
            Some("accept") => accept = Some(value),
            _ => bail!("secure admission refused"),
        }
    }
    let ready = ready.context("secure ready missing")?;
    let accept = accept.context("secure accept missing")?;
    let sid = ready["sid"]
        .as_str()
        .context("secure sid missing")?
        .to_string();
    let rid = ready["rid"]
        .as_str()
        .context("secure rid missing")?
        .to_string();
    if accept["sid"] != sid || accept["rid"] != rid {
        bail!("secure admission identity mismatch")
    }
    let mut session = Session::new(
        &identity,
        SessionConfig {
            initiator: true,
            namespace: host.namespace.clone(),
            host_key: host.host_key.clone(),
            device_key: hex::encode(identity.signing_public()),
            host_noise_key: Some(host.host_noise_key.clone()),
            device_noise_key: None,
            sid: sid.clone(),
            rid: rid.clone(),
        },
    )
    .context("invalid secure session configuration")?;
    let mut wire = ReliableChannel::new(sid, rid);
    wire.queue_packet(&session.handshake_write()?)?;
    while !session.established() {
        for frame in wire.outgoing(0)? {
            socket
                .send(Message::Text(serde_json::to_string(&frame)?.into()))
                .await?;
        }
        let frame: WireFrame = serde_json::from_value(read_json(&mut socket).await?)?;
        for packet in wire
            .receive(frame, 0)
            .context("invalid secure handshake delivery")?
        {
            session
                .handshake_read(&packet)
                .context("secure host handshake failed")?;
            if !session.established() {
                wire.queue_packet(&session.handshake_write()?)?;
            }
        }
    }
    for frame in wire.outgoing(0)? {
        socket
            .send(Message::Text(serde_json::to_string(&frame)?.into()))
            .await?;
    }
    Ok((socket, session, wire))
}
async fn drive(
    mut socket: Socket,
    mut session: Session,
    mut wire: ReliableChannel,
    mut commands: mpsc::Receiver<Command>,
    stop: CancellationToken,
) -> Result<()> {
    let mut streams: BTreeMap<u32, StreamState> = BTreeMap::new();
    let mut last_stream = 0;
    let start = std::time::Instant::now();
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    let mut heartbeat = tokio::time::interval(Duration::from_secs(15));
    let mut last_heartbeat = std::time::Instant::now();
    let result=async {loop {
  let now=start.elapsed().as_millis() as u64;socket.delivery_tracker().apply(&mut wire,now);for frame in wire.outgoing(now)?{tokio::time::timeout(Duration::from_secs(5),socket.send(Message::Text(serde_json::to_string(&frame)?.into()))).await??;}
  tokio::select!{
   _=stop.cancelled()=>return Ok(()),
   _=heartbeat.tick()=>{if last_heartbeat.elapsed()>Duration::from_secs(45){bail!("secure peer heartbeat timed out")}tokio::time::timeout(Duration::from_secs(5),socket.send(Message::Text(serde_json::json!({"a":"heartbeat","v":3}).to_string().into()))).await??;},
   _=tick.tick()=>{let cancelled=streams.iter().filter(|(_,s)|s.cancel.is_cancelled()||s.head.as_ref().is_some_and(|h|h.is_closed())).map(|(id,_)|*id).collect::<Vec<_>>();for id in cancelled{if wire.can_queue(){cancel_stream(&mut streams,id);wire.queue_packet(&session.seal(&Record::Cancel{stream_id:id,code:"cancelled".into()}.encode()?)?)?;}}},
   command=commands.recv(),if wire.can_queue()=>{match command.context("secure client released")? {
    Command::Start{open,id,state}=>{last_stream=id;if streams.len()>=MAX_STREAMS{state.fail("secure concurrent stream limit exceeded");continue}streams.insert(id,state);wire.queue_packet(&session.seal(&Record::Open{stream_id:id,request:open}.encode()?)?)?;},
    Command::Record(record)=>{if streams.contains_key(&record.stream_id()){wire.queue_packet(&session.seal(&record.encode()?)?)?;}},
    Command::Consumed(id,bytes)=>{if let Some(state)=streams.get_mut(&id){if let Some(bytes)=state.receive_credit.consume(bytes)?{wire.queue_packet(&session.seal(&Record::Credit{stream_id:id,bytes}.encode()?)?)?;}}},
   }},
   message=socket.next()=>{let message=message.context("secure socket closed")??;match message{
    Message::Text(text)=>{let value:serde_json::Value=serde_json::from_str(&text)?;if value["a"]=="heartbeat"&&value["v"]==3{last_heartbeat=std::time::Instant::now();continue}
if value["a"]=="cancel"{bail!("secure peer closed")}
     let frame:WireFrame=serde_json::from_value(value)?;for packet in wire.receive(frame,now)?{let record=Record::decode(&session.open(&packet)?)?;let id=record.stream_id();if !streams.contains_key(&id) && id<=last_stream {continue}match record{
      Record::Headers{response,..}=>{let state=streams.get_mut(&id).context("unknown secure response")?;if state.head_seen{bail!("duplicate response head")}state.head_seen=true;if let Some(head)=state.head.take(){let _=head.send(Ok(response));}},
      Record::Data{bytes,..}=>{let state=streams.get_mut(&id).context("unknown secure response")?;if !state.head_seen{bail!("body before response head")}state.receive_credit.receive(bytes.len() as u32)?;if state.body.try_send(Ok(bytes.into())).is_err(){state.cancel.cancel();}},
      Record::Credit{bytes,..}=>{if let Some(state)=streams.get(&id){if state.upload_credit.available_permits()+bytes as usize>STREAM_WINDOW as usize{bail!("upload credit exceeded")}state.upload_credit.add_permits(bytes as usize);}},
      Record::End{..}=>{let state=streams.remove(&id).context("unknown secure EOF")?;if !state.head_seen{bail!("EOF before head")}state.cancel.cancel();},
      Record::Cancel{..}=>cancel_stream(&mut streams,id),
      Record::Open{..}=>bail!("host opened an unauthorized request"),
     }}
    },Message::Ping(data)=>socket.send(Message::Pong(data)).await?,Message::Pong(_)=>{},_=>bail!("secure socket closed"),
   }}
  }
 }}.await;
    for (_, state) in streams {
        state.fail("secure connection lost; mutation outcome may be unknown");
    }
    result
}

/// Pair only from an operator-provided invitation. No cloud response supplies a pin.
pub async fn pair(invite: PairingInvite, client_name: &str) -> Result<PairedHost> {
    if ![2, 3].contains(&invite.protocol) {
        bail!("new protocol3 invitation required")
    }
    let host_key = decode_hex::<32>(&invite.host_key)?;
    if namespace(&host_key) != invite.namespace {
        bail!("invitation namespace mismatch")
    }
    Binding {
        signing_key: invite.host_key.clone(),
        noise_key: invite.host_noise_key.clone(),
        signature: invite.host_binding_signature.clone(),
    }
    .bytes()?;
    invite
        .invite
        .verify(&host_key, now(), invite.invite.epoch)?;
    let seed = decode_hex::<32>(&invite.invite_seed)?;
    let mut secret = zeroize::Zeroizing::new([0; 64]);
    secret[..32].copy_from_slice(&seed);
    let invitation_key = Identity::from_bytes(secret.as_ref())?;
    if hex::encode(invitation_key.signing_public()) != invite.invite.signing_key {
        bail!("invitation secret mismatch")
    }
    let identity = Identity::generate()?;
    let mut host = PairedHost {
        protocol: 3,
        origin: invite.origin.clone(),
        websocket: invite
            .websocket
            .clone()
            .or_else(|| {
                invite
                    .endpoints
                    .first()
                    .and_then(|e| direct_endpoint(&e.url).ok())
            })
            .context("invitation has no secure endpoint")?,
        endpoints: invite.endpoints.clone(),
        namespace: invite.namespace.clone(),
        host_key: invite.host_key.clone(),
        host_noise_key: invite.host_noise_key.clone(),
        host_binding_signature: invite.host_binding_signature.clone(),
        device_identity: hex::encode(identity.to_bytes()),
        api_key: String::new(),
        grant: DeviceGrant {
            id: String::new(),
            expires_at: 0,
            epoch: 0,
            signature: String::new(),
        },
    };
    let (socket, session, wire) = admit_best(&host, Some(&invite)).await?;
    let proof = invitation_key.sign(&claim_bytes(
        &host.namespace,
        &invite.invite.id,
        &session.handshake_hash()?,
        &identity.signing_public(),
        &identity.noise_public(),
    )?);
    let (tx, rx) = mpsc::channel(64);
    let connection = Arc::new(Connection {
        commands: tx,
        dispatch: Mutex::new(()),
        next: AtomicU32::new(1),
        alive: AtomicBool::new(true),
        cancel: CancellationToken::new(),
    });
    let weak = Arc::downgrade(&connection);
    let stop = connection.cancel.clone();
    tokio::spawn(async move {
        let _ = drive(socket, session, wire, rx, stop).await;
        if let Some(c) = weak.upgrade() {
            c.alive.store(false, Ordering::Release)
        }
    });
    let client = SecureClient {
        stop: CancellationToken::new(),
        host: Arc::new(std::sync::RwLock::new(host.clone())),
        renewal: Default::default(),
        connection: Arc::new(Mutex::new(Some(connection))),
    };
    let request=reqwest::Client::new().post(format!("{}/api/pairing/claim",host.origin)).json(&serde_json::json!({"proof":hex::encode(proof),"client_name":client_name,"client_kind":"cli"})).build()?;
    let response = client.send_inner(request).await?.error_for_status()?;
    let value = bounded_authority_json(response).await?;
    host.api_key = value["api_key"]
        .as_str()
        .context("pairing response missing device credential")?
        .to_string();
    host.grant = serde_json::from_value(value["grant"].clone())?;
    host.validate()?;
    Ok(host)
}

impl Drop for PairedHost {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.device_identity.zeroize();
        self.api_key.zeroize();
    }
}
impl Drop for PairingInvite {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.invite_seed.zeroize();
    }
}

async fn bounded_authority_json(mut response: reqwest::Response) -> Result<serde_json::Value> {
    const LIMIT: usize = 65536;
    tokio::time::timeout(Duration::from_secs(15), async move {
        if response.content_length().is_some_and(|n| n > LIMIT as u64) {
            bail!("secure authority response too large");
        }
        let mut bytes = zeroize::Zeroizing::new(Vec::new());
        while let Some(chunk) = response.chunk().await? {
            if bytes.len() + chunk.len() > LIMIT {
                bail!("secure authority response too large");
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).context("invalid secure authority response")
    })
    .await
    .context("secure authority response timed out")?
}

#[cfg(test)]
mod bounded_metadata_tests {
    use super::*;
    #[tokio::test]
    async fn authority_response_rejects_unbounded_or_stalled_body() {
        let response: reqwest::Response =
            http::Response::new(reqwest::Body::from(vec![b' '; 65537])).into();
        assert!(bounded_authority_json(response).await.is_err());
        let response: reqwest::Response =
            http::Response::new(reqwest::Body::from("{\"grant\":1}")).into();
        assert_eq!(bounded_authority_json(response).await.unwrap()["grant"], 1);
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    fn pending_stream(
        headers_pending: bool,
    ) -> (
        BTreeMap<u32, StreamState>,
        oneshot::Receiver<Result<Headers>>,
        crate::BodyReceiver,
    ) {
        let (head_tx, head_rx) = oneshot::channel();
        let (body_tx, body_rx) = crate::body_channel();
        let cancel = CancellationToken::new();
        // Force the exact state the timer observes after Forget cancels a
        // parent token, before the connection-stop select arm is polled.
        cancel.cancel();
        let state = StreamState {
            head: headers_pending.then_some(head_tx),
            body: body_tx,
            upload_credit: Arc::new(Semaphore::new(STREAM_WINDOW as usize)),
            receive_credit: ReceiveCredit::default(),
            cancel,
            head_seen: !headers_pending,
        };
        (BTreeMap::from([(1, state)]), head_rx, body_rx)
    }

    #[tokio::test]
    async fn cancelled_timer_cleanup_reports_body_error_instead_of_successful_eof() {
        let (mut streams, _, mut body) = pending_stream(false);
        cancel_stream(&mut streams, 1);
        assert!(streams.is_empty());
        let error = body
            .recv()
            .await
            .expect("a cancelled response must not look like successful EOF")
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("mutation outcome may be unknown"));
        assert!(body.recv().await.is_none());
    }

    #[tokio::test]
    async fn cancelled_timer_cleanup_reports_explicit_error_before_headers() {
        let (mut streams, head, _) = pending_stream(true);
        cancel_stream(&mut streams, 1);
        assert!(streams.is_empty());
        let error = head
            .await
            .expect("cancellation must report why the response was interrupted")
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("mutation outcome may be unknown"));
    }
}

#[cfg(test)]
mod admission_retry_tests {
    use super::*;
    #[tokio::test]
    async fn gateway_busy_retries_only_admission_and_dispatches_once() {
        let count = std::sync::atomic::AtomicUsize::new(0);
        let admitted = retry_admission(|| async {
            if count.fetch_add(1, Ordering::SeqCst) < 2 {
                Err(
                    tokio_tungstenite::tungstenite::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::WouldBlock,
                        AdmissionBusy,
                    ))
                    .into(),
                )
            } else {
                Ok(())
            }
        })
        .await;
        assert!(admitted.is_ok());
        let dispatched = usize::from(admitted.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 3);
        assert_eq!(dispatched, 1);
    }
    #[tokio::test]
    async fn busy_retry_is_bounded_and_protocol_refusals_never_retry() {
        let count = std::sync::atomic::AtomicUsize::new(0);
        let exhausted = retry_admission(|| async {
            count.fetch_add(1, Ordering::SeqCst);
            Err::<(), _>(
                tokio_tungstenite::tungstenite::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    AdmissionBusy,
                ))
                .into(),
            )
        })
        .await;
        assert!(exhausted.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 3);
        for refusal in [
            "secure admission refused",
            "secure challenge required",
            "invalid grant",
            "request dispatched; outcome unknown",
        ] {
            let count = std::sync::atomic::AtomicUsize::new(0);
            let failed = retry_admission(|| async {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(anyhow::anyhow!("{refusal}"))
            })
            .await;
            assert!(failed.is_err());
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
    }
}
