//! E2EE host endpoint. The cloud carries ciphertext; only the loopback API sees HTTP.
use crate::authenticated_target;
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use futures_util::StreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::{body::Frame, Request};
use hyper_util::rt::TokioIo;
use mold_secure::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    net::SocketAddr,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, watch, Semaphore};
use tokio_util::sync::CancellationToken;

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct HostPolicy {
    pub epoch: u64,
    pub active: Vec<String>,
}
#[derive(Clone)]
pub struct HostConfig {
    pub identity: Arc<Identity>,
    pub target: SocketAddr,
    pub policy: watch::Receiver<HostPolicy>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerAdmission {
    pub sid: String,
    pub rid: String,
    pub role: String,
    pub binding: Binding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant: Option<DeviceGrant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invite: Option<Invitation>,
}
impl PeerAdmission {
    pub fn validate(&self, config: &HostConfig) -> Result<()> {
        self.binding.bytes()?;
        let policy = config.policy.borrow();
        let host = config.identity.signing_public();
        match (&*self.role, &self.grant, &self.invite) {
            ("device", Some(g), None) => {
                g.verify(&host, &self.binding, unix_now(), policy.epoch)?;
                if !policy.active.contains(&g.id) {
                    bail!("secure device is revoked")
                }
            }
            ("invite", None, Some(i)) => {
                i.verify(&host, unix_now(), policy.epoch)?;
            }
            _ => bail!("invalid secure admission"),
        }
        Ok(())
    }
    fn current(&self, config: &HostConfig) -> Result<()> {
        let policy = config.policy.borrow();
        let allowed = match (&self.grant, &self.invite) {
            (Some(g), None) => {
                g.expires_at > unix_now()
                    && g.epoch == policy.epoch
                    && policy.active.contains(&g.id)
            }
            (None, Some(i)) => i.expires_at > unix_now() && i.epoch == policy.epoch,
            _ => false,
        };
        if !allowed {
            bail!("secure peer expired or revoked")
        }
        Ok(())
    }
    fn context(
        &self,
        hash: [u8; 32],
        id: u32,
        identity: &Identity,
        open: &Open,
    ) -> Result<RequestContext> {
        let mut context = RequestContext {
            binding: self.binding.clone(),
            grant: self.grant.clone(),
            invite: self.invite.clone(),
            handshake_hash: hex::encode(hash),
            stream_id: id,
            issued_at: unix_now(),
            request_nonce: uuid::Uuid::new_v4().simple().to_string(),
            signature: String::new(),
        };
        context.sign(identity, &open.method, &open.path)?;
        Ok(context)
    }
}
pub async fn read_policy(identity: &Identity, target: SocketAddr) -> Result<HostPolicy> {
    crate::validate_target(target)?;
    let mut context = ControlContext {
        issued_at: unix_now(),
        nonce: uuid::Uuid::new_v4().simple().to_string(),
        signature: String::new(),
    };
    context.sign(identity)?;
    let mut response = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_millis(500))
        .build()?
        .get(format!("http://{target}/api/pairing/secure/check"))
        .header(
            "x-mold-secure-control",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&context)?),
        )
        .send()
        .await?
        .error_for_status()?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len() + chunk.len() > 200_000 {
            bail!("secure policy bound exceeded")
        }
        bytes.extend_from_slice(&chunk);
    }
    let policy: HostPolicy = serde_json::from_slice(&bytes)?;
    if policy.active.len() > 4096 {
        bail!("secure policy bound exceeded")
    }
    Ok(policy)
}
pub async fn watch_policy(
    identity: Arc<Identity>,
    target: SocketAddr,
    shutdown: CancellationToken,
) -> Result<watch::Receiver<HostPolicy>> {
    let mut first = read_policy(&identity, target).await?;
    first.active.sort_unstable();
    let (tx, rx) = watch::channel(first);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(500));
        loop {
            tokio::select! {
                _ = shutdown.cancelled()=>break,
                _ = tx.closed()=>break,
                _ = tick.tick()=>{
                    let mut policy = match read_policy(&identity,target).await {
                        Ok(policy)=>policy,
                        Err(_)=>HostPolicy{epoch:tx.borrow().epoch,active:vec![]},
                    };
                    policy.active.sort_unstable();
                    tx.send_if_modified(|current| {
                        if *current == policy { return false; }
                        *current = policy;
                        true
                    });
                }
            }
        }
    });
    Ok(rx)
}
struct Active {
    body: Option<mold_secure_client::BodySender>,
    remaining: Option<u64>,
    uploaded: u64,
    receive_credit: ReceiveCredit,
    response_credit: Arc<Semaphore>,
    cancel: CancellationToken,
}
enum Event {
    Record(Record),
    Consumed(u32, u32),
    Finished(u32),
}
fn apply_event(
    event: Event,
    streams: &mut BTreeMap<u32, Active>,
    wire: &mut ReliableChannel,
    session: &mut Session,
    invitation_finished: &mut bool,
    role: &str,
) -> Result<()> {
    match event {
        Event::Consumed(id, n) => {
            if let Some(active) = streams.get_mut(&id) {
                if let Some(bytes) = active.receive_credit.consume(n)? {
                    wire.queue_packet(
                        &session.seal(
                            &Record::Credit {
                                stream_id: id,
                                bytes,
                            }
                            .encode()?,
                        )?,
                    )?;
                }
            }
        }
        Event::Record(record) => {
            wire.queue_packet(&session.seal(&record.encode()?)?)?;
        }
        Event::Finished(id) => {
            if let Some(active) = streams.remove(&id) {
                active.cancel.cancel();
            }
            if role == "invite" {
                *invitation_finished = true;
            }
        }
    }
    Ok(())
}
/// Drivers must keep both wire queues bounded. Any loss/corruption closes this
/// whole Noise session; an uncertain mutation is never replayed on reconnect.
pub async fn serve_peer(
    config: HostConfig,
    admission: PeerAdmission,
    input: mpsc::Receiver<WireFrame>,
    output: mpsc::Sender<WireFrame>,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_peer_with_delivery(config, admission, input, output, shutdown, None).await
}
async fn serve_peer_with_delivery(
    config: HostConfig,
    admission: PeerAdmission,
    mut input: mpsc::Receiver<WireFrame>,
    output: mpsc::Sender<WireFrame>,
    shutdown: CancellationToken,
    delivery: Option<mold_secure_client::object_socket::DeliveryTracker>,
) -> Result<()> {
    admission.validate(&config)?;
    let mut session = Session::new(
        &config.identity,
        SessionConfig {
            initiator: false,
            namespace: namespace(&config.identity.signing_public()),
            host_key: hex::encode(config.identity.signing_public()),
            device_key: admission.binding.signing_key.clone(),
            host_noise_key: Some(hex::encode(config.identity.noise_public())),
            device_noise_key: Some(admission.binding.noise_key.clone()),
            sid: admission.sid.clone(),
            rid: admission.rid.clone(),
        },
    )?;
    let mut wire = ReliableChannel::new(admission.sid.clone(), admission.rid.clone());
    let start = std::time::Instant::now();
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    let (events, mut incoming) = mpsc::channel::<Event>(16);
    let mut streams: BTreeMap<u32, Active> = BTreeMap::new();
    let mut last_stream = 0;
    let mut invitation_finished = false;
    let mut policy = config.policy.clone();
    let stop = shutdown.child_token();
    let _cancel_on_drop = stop.clone().drop_guard();
    // Revocation must interrupt a stalled network writer as well as the main
    // receive loop. The task exits with this peer's cancellation token.
    let revoke_stop = stop.clone();
    let revoke_admission = admission.clone();
    let revoke_config = config.clone();
    let mut revoke_policy = config.policy.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = revoke_stop.cancelled() => break,
                changed = revoke_policy.changed() => {
                    if changed.is_err() || revoke_admission.current(&revoke_config).is_err() {
                        revoke_stop.cancel();
                        break;
                    }
                }
            }
        }
    });
    let result=async {
 loop {
  let now=start.elapsed().as_millis() as u64;
  if session.established() {
   for _ in 0..16 {
    if !wire.can_queue(){break;}
    let Ok(event)=incoming.try_recv() else {break;};
    apply_event(event,&mut streams,&mut wire,&mut session,&mut invitation_finished,&admission.role)?;
   }
  }
  if let Some(delivery)=&delivery{delivery.apply(&mut wire,now);}
  for frame in wire.outgoing(now)? {tokio::select!{_ = stop.cancelled()=>return Ok(()),r=output.send(frame)=>{r.context("secure wire closed")?}}}
  if invitation_finished && wire.is_idle(){return Ok(())}
  tokio::select!{
   _=stop.cancelled()=>return Ok(()),
   _=tick.tick()=>{if !session.established()&&start.elapsed()>Duration::from_secs(15){bail!("secure handshake timed out")}admission.current(&config)?;},
   r=policy.changed()=>{r.context("secure policy closed")?;admission.current(&config)?;},
   event=incoming.recv(),if wire.can_queue()=>{let event=event.context("secure event queue closed")?;apply_event(event,&mut streams,&mut wire,&mut session,&mut invitation_finished,&admission.role)?},
   frame=input.recv()=>{
    for packet in wire.receive(frame.context("secure wire closed")?,now)? {
     if !session.established(){session.handshake_read(&packet)?;if !session.established(){wire.queue_packet(&session.handshake_write()?)?;}continue}
     let record=Record::decode(&session.open(&packet)?)?;let id=record.stream_id();
     match record {
      Record::Open{request,..}=>{
       if id<=last_stream||streams.len()>=MAX_STREAMS{bail!("secure stream limit or replay")}
       last_stream=id;admission.current(&config)?;
       if request.path.split('?').next()==Some("/metrics")||request.path.split('?').next() == Some("/api/pairing/secure/check"){bail!("secure private target forbidden")}
       if admission.role=="invite"&&(id!=1||request.method!="POST"||request.path!="/api/pairing/claim"){bail!("invitation permits only pairing")}
       let context=admission.context(session.handshake_hash()?,id,&config.identity,&request)?;
       let (body,receiver)=mold_secure_client::body_channel();let credit=Arc::new(Semaphore::new(STREAM_WINDOW as usize));let cancel=stop.child_token();
       streams.insert(id,Active{body:Some(body),remaining:request.length,uploaded:0,receive_credit:ReceiveCredit::default(),response_credit:credit.clone(),cancel:cancel.clone()});
       let target=config.target;let events=events.clone();
       tokio::spawn(async move{let result=tokio::select!{_ = cancel.cancelled()=>Ok(()),r=forward(target,id,request,context,receiver,credit,events.clone())=>r};if result.is_err(){let _=events.send(Event::Record(Record::Cancel{stream_id:id,code:"target_failed".into()})).await;}let _=events.send(Event::Finished(id)).await;});
      },
      Record::Data{bytes,..}=>{if let Some(active)=streams.get_mut(&id){let n=bytes.len() as u32;active.receive_credit.receive(n)?;active.uploaded+=u64::from(n);if active.uploaded>MAX_UPLOAD{bail!("upload limit exceeded")}
if let Some(remaining)=&mut active.remaining{*remaining=remaining.checked_sub(u64::from(n)).context("upload exceeds length")?;}active.body.as_ref().context("upload after end")?.try_send(Ok(bytes.into())).context("upload pressure exceeded")?;}else if id>last_stream{bail!("unknown secure stream")}},
      Record::End{..}=>{if let Some(active)=streams.get_mut(&id){if active.remaining.is_some_and(|n|n!=0){bail!("truncated upload")}active.body.take();}else if id>last_stream{bail!("unknown secure stream")}},
      Record::Cancel{..}=>{if let Some(active)=streams.remove(&id){active.cancel.cancel();}},
      Record::Credit{bytes,..}=>{if let Some(active)=streams.get(&id){if active.response_credit.available_permits()+bytes as usize>STREAM_WINDOW as usize{bail!("response credit exceeded")}active.response_credit.add_permits(bytes as usize);}},
      Record::Headers{..}=>bail!("client sent response headers"),
     }
    }
   }
  }
 }
 }.await;
    if let Err(error) = &result {
        eprintln!(
            "Mold secure peer ended ({})",
            safe_relay_error_category(error)
        );
    }
    stop.cancel();
    for active in streams.into_values() {
        active.cancel.cancel();
    }
    result
}
fn stripped_header(name: &str, connection_tokens: &BTreeSet<String>) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host"
            | "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
            | "x-mold-secure-peer"
            | "x-mold-secure-control"
            | "forwarded"
            | "x-forwarded-for"
            | "x-forwarded-host"
            | "x-forwarded-proto"
            | "x-real-ip"
    ) || connection_tokens.contains(&name.to_ascii_lowercase())
}
async fn forward(
    target: SocketAddr,
    id: u32,
    open: Open,
    context: RequestContext,
    body: mold_secure_client::BodyReceiver,
    credit: Arc<Semaphore>,
    events: mpsc::Sender<Event>,
) -> Result<()> {
    let socket = authenticated_target(target).await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(socket)).await?;
    let connection_task = tokio::spawn(connection);
    let guard = AbortOnDrop(connection_task);
    let eventbody = events.clone();
    let stream = futures_util::stream::unfold(
        (body, eventbody),
        move |(mut receiver, events)| async move {
            let data = match receiver.recv().await? {
                Ok(bytes) => bytes,
                Err(error) => return Some((Err(std::io::Error::other(error)), (receiver, events))),
            };
            let n = data.len() as u32;
            if events.send(Event::Consumed(id, n)).await.is_err() {
                return None;
            }
            Some((
                Ok::<_, std::io::Error>(Frame::data(data)),
                (receiver, events),
            ))
        },
    );
    let mut request = Request::builder()
        .method(open.method.as_str())
        .uri(open.path.as_str())
        .header("host", target.to_string());
    let connection_tokens = open
        .headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("connection"))
        .flat_map(|(_, v)| v.split(',').map(|s| s.trim().to_ascii_lowercase()))
        .collect();
    for (name, value) in &open.headers {
        if !stripped_header(name, &connection_tokens) {
            request = request.header(name, value);
        }
    }
    if let Some(length) = open.length {
        request = request.header("content-length", length);
    }
    request = request.header(
        "x-mold-secure-peer",
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&context)?),
    );
    let mut response = sender
        .send_request(request.body(StreamBody::new(Box::pin(stream)))?)
        .await?;
    let response_tokens = response
        .headers()
        .get_all("connection")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(',').map(|s| s.trim().to_ascii_lowercase()))
        .collect();
    let mut headers = vec![];
    for (name, value) in response.headers() {
        if !stripped_header(name.as_str(), &response_tokens)
            || name == hyper::header::CONTENT_LENGTH
        {
            headers.push((name.to_string(), value.to_str()?.into()));
        }
    }
    events
        .send(Event::Record(Record::Headers {
            stream_id: id,
            response: Headers {
                status: response.status().as_u16(),
                headers,
            },
        }))
        .await?;
    while let Some(frame) = response.body_mut().frame().await {
        if let Ok(data) = frame?.into_data() {
            for bytes in data.chunks(DATA_CHUNK) {
                let permit = credit.acquire_many(bytes.len() as u32).await?;
                permit.forget();
                events
                    .send(Event::Record(Record::Data {
                        stream_id: id,
                        bytes: bytes.to_vec(),
                    }))
                    .await?;
            }
        }
    }
    events
        .send(Event::Record(Record::End { stream_id: id }))
        .await?;
    drop(guard);
    Ok(())
}
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);
impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// LAN publishes only this WebSocket route, never the underlying HTTP router.
pub fn lan_router(config: HostConfig, shutdown: CancellationToken) -> axum::Router {
    use axum::{
        extract::{ws::WebSocketUpgrade, Query, State},
        response::IntoResponse,
        routing::get,
    };
    #[derive(Clone)]
    struct StateData {
        config: HostConfig,
        slots: Arc<Semaphore>,
        shutdown: CancellationToken,
    }
    async fn upgrade(
        State(state): State<StateData>,
        Query(query): Query<BTreeMap<String, String>>,
        ws: WebSocketUpgrade,
    ) -> axum::response::Response {
        if query.len() != 1 || query.get("v").map(String::as_str) != Some("3") {
            return (
                axum::http::StatusCode::UPGRADE_REQUIRED,
                "secure protocol 3 required",
            )
                .into_response();
        }
        let Ok(permit) = state.slots.try_acquire_owned() else {
            return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response();
        };
        ws.max_message_size(24 * 1024)
            .max_frame_size(24 * 1024)
            .on_upgrade(move |socket| async move {
                let _permit = permit;
                tokio::select! { _ = state.shutdown.cancelled() => {}, _ = lan_session(socket, state.config) => {} }
            })
    }
    axum::Router::new()
        .route("/_mold/secure", get(upgrade))
        .with_state(StateData {
            config,
            slots: Arc::new(Semaphore::new(32)),
            shutdown,
        })
}
async fn lan_session(mut socket: axum::extract::ws::WebSocket, config: HostConfig) -> Result<()> {
    use axum::extract::ws::Message;
    async fn read(socket: &mut axum::extract::ws::WebSocket) -> Result<serde_json::Value> {
        let frame = tokio::time::timeout(Duration::from_secs(10), socket.next())
            .await?
            .context("secure socket closed")??;
        let Message::Text(text) = frame else {
            bail!("secure text frame required")
        };
        if text.len() > 24 * 1024 {
            bail!("secure frame too large")
        }
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if value["v"] != 3 {
            bail!("secure version required")
        }
        Ok(value)
    }
    let ns = namespace(&config.identity.signing_public());
    let hello = read(&mut socket).await?;
    if hello["a"] != "hello"
        || hello["namespace"] != ns
        || hello["hostKey"] != hex::encode(config.identity.signing_public())
    {
        bail!("wrong secure host")
    }
    let role = hello["role"].as_str().context("missing role")?;
    let role_byte = match role {
        "device" => 2,
        "invite" => 3,
        _ => bail!("invalid LAN role"),
    };
    let rid = uuid::Uuid::new_v4().simple().to_string();
    let sid = uuid::Uuid::new_v4().simple().to_string();
    let nonce = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let expires_at = unix_now() + 60;
    socket.send(Message::Text(serde_json::json!({"a":"challenge","v":3,"connectionId":rid,"nonce":nonce,"expiresAt":expires_at}).to_string())).await?;
    let join = read(&mut socket).await?;
    if join["a"] != "join" || unix_now() > expires_at {
        bail!("secure admission expired")
    }
    let admission = PeerAdmission {
        sid: sid.clone(),
        rid: rid.clone(),
        role: role.into(),
        binding: serde_json::from_value(join["binding"].clone())?,
        grant: join
            .get("grant")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?,
        invite: join
            .get("invite")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?,
    };
    admission.current(&config)?;
    let signer = if let Some(invite) = &admission.invite {
        &invite.signing_key
    } else {
        &admission.binding.signing_key
    };
    if !verify_signature(
        &decode_hex::<32>(signer)?,
        &join_bytes(
            &ns,
            role_byte,
            &rid,
            &nonce,
            &decode_hex(&admission.binding.signing_key)?,
            expires_at,
        )?,
        &decode_hex::<64>(join["signature"].as_str().context("missing proof")?)?,
    ) {
        bail!("secure admission proof invalid")
    }
    for frame in [
        serde_json::json!({"a":"ready","v":3,"sid":sid,"rid":rid,"role":role,"epoch":config.policy.borrow().epoch}),
        serde_json::json!({"a":"accept","v":3,"sid":sid,"rid":rid}),
    ] {
        socket.send(Message::Text(frame.to_string())).await?;
    }
    let (input, rx) = mpsc::channel((WIRE_WINDOW * 2) as usize);
    let (tx, mut output) = mpsc::channel((WIRE_WINDOW * 2) as usize);
    let stop = CancellationToken::new();
    let _cancel_on_drop = stop.clone().drop_guard();
    let peerstop = stop.clone();
    let mut task = AbortOnDrop(tokio::spawn(serve_peer(
        config, admission, rx, tx, peerstop,
    )));
    let result=async {loop{tokio::select!{
  result=&mut task.0=>return result.context("secure peer task stopped")?,
  frame=output.recv()=>{let frame=frame.context("secure peer closed")?;tokio::time::timeout(Duration::from_secs(5),socket.send(Message::Text(serde_json::to_string(&frame)?))).await??;},
  message=socket.next()=>{match message.context("secure socket closed")??{Message::Text(text)=>{let value:serde_json::Value=serde_json::from_str(&text)?;if value["a"]=="heartbeat"&&value["v"]==3{socket.send(Message::Text(serde_json::json!({"a":"heartbeat","v":3,"sid":sid}).to_string())).await?;}else{input.try_send(serde_json::from_value(value)?).context("secure input pressure exceeded")?;}},Message::Ping(data)=>socket.send(Message::Pong(data)).await?,Message::Close(_)=>return Ok(()),_=>bail!("invalid secure socket message")}}
 }}}.await;
    stop.cancel();
    task.0.abort();
    result
}
/// Outbound cloud host connection. Reconnect establishes new peer Noise sessions
/// and never resends decrypted HTTP operations.
pub async fn connect(
    endpoint: &str,
    config: HostConfig,
    shutdown: CancellationToken,
    allow_loopback_ws: bool,
) -> Result<()> {
    let (status, _) = watch::channel(false);
    connect_with_status(endpoint, config, shutdown, allow_loopback_ws, status).await
}
pub async fn connect_with_status(
    endpoint: &str,
    config: HostConfig,
    shutdown: CancellationToken,
    allow_loopback_ws: bool,
    status: watch::Sender<bool>,
) -> Result<()> {
    crate::ensure_tls_provider();
    drop(authenticated_target(config.target).await?);
    let mut url = crate::aws::validate_aws_endpoint(endpoint, allow_loopback_ws)?;
    url.set_query(Some("v=3"));
    let mut delay = Duration::from_millis(250);
    loop {
        let began = std::time::Instant::now();
        status.send_replace(false);
        let result = tokio::select! { _ = shutdown.cancelled() => { status.send_replace(false); return Ok(()); }, result = cloud_session(url.as_str(), config.clone(), shutdown.child_token(), status.clone()) => result };
        status.send_replace(false);
        if shutdown.is_cancelled() {
            return Ok(());
        }
        if began.elapsed() > Duration::from_secs(60) {
            delay = Duration::from_millis(250)
        }
        if let Err(error) = &result {
            eprintln!(
                "Mold secure relay session interrupted ({}); reconnecting without replay",
                safe_relay_error_category(error)
            );
        }
        tokio::select! {_=shutdown.cancelled()=>return Ok(()),_=tokio::time::sleep(delay)=>{}}
        delay = (delay * 2).min(Duration::from_secs(15));
    }
}
fn safe_relay_error_category(error: &anyhow::Error) -> &'static str {
    // Emit only constants. Transport errors can contain response text or signed
    // URLs, so formatting the source error itself would expose credentials.
    for source in error.chain() {
        for category in [
            "secure relay heartbeat timed out",
            "invalid secure peer admission",
            "wrong secure epoch",
            "unexpected secure control",
            "secure writer closed",
            "secure socket closed",
            "wrong secure version",
            "secure policy stopped",
            "secure host admission failed",
            "secure stream limit or replay",
            "upload limit exceeded",
            "upload exceeds length",
            "upload after end",
            "upload pressure exceeded",
            "truncated upload",
            "unknown secure stream",
            "response credit exceeded",
            "client sent response headers",
            "secure wire closed",
        ] {
            if source.to_string() == category {
                return category;
            }
        }
        if let Some(error) = source.downcast_ref::<mold_secure::Error>() {
            return match error {
                mold_secure::Error::State => "invalid secure state",
                mold_secure::Error::Limit => "secure protocol limit",
                mold_secure::Error::Authentication => "secure authentication failure",
                _ => "invalid secure record",
            };
        }
        if source.is::<tokio::time::error::Elapsed>() {
            return "socket send timeout";
        }
        if source.is::<serde_json::Error>() {
            return "invalid relay frame";
        }
        if source.is::<tokio_tungstenite::tungstenite::Error>() {
            return "websocket error";
        }
    }
    "transport unavailable"
}
async fn cloud_session(
    endpoint: &str,
    config: HostConfig,
    shutdown: CancellationToken,
    status: watch::Sender<bool>,
) -> Result<()> {
    use futures_util::SinkExt;
    use tokio_tungstenite::tungstenite::Message;
    let (socket, _) = tokio_tungstenite::connect_async_with_config(
        endpoint,
        Some(
            tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
                .max_message_size(Some(24 * 1024))
                .max_frame_size(Some(24 * 1024)),
        ),
        false,
    )
    .await?;
    let ns = namespace(&config.identity.signing_public());
    let mut socket = mold_secure_client::object_socket::ObjectSocket::new(socket, ns.clone(), true);
    let delivery = socket.delivery_tracker();
    socket.send(Message::Text(serde_json::json!({"a":"hello","v":3,"namespace":ns,"hostKey":hex::encode(config.identity.signing_public()),"role":"host"}).to_string().into())).await?;
    async fn read<S>(socket: &mut S) -> Result<serde_json::Value>
    where
        S: futures_util::Stream<
                Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>,
            > + Unpin,
    {
        let msg = tokio::time::timeout(Duration::from_secs(15), socket.next())
            .await?
            .context("secure socket closed")??;
        let Message::Text(text) = msg else {
            bail!("secure text required")
        };
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if value["v"] != 3 {
            bail!("secure version required")
        }
        Ok(value)
    }
    let challenge = read(&mut socket).await?;
    if challenge["a"] != "challenge" {
        bail!("secure challenge missing")
    }
    let signed = join_bytes(
        &ns,
        1,
        challenge["connectionId"]
            .as_str()
            .context("missing connection ID")?,
        challenge["nonce"].as_str().context("missing nonce")?,
        &config.identity.signing_public(),
        challenge["expiresAt"]
            .as_u64()
            .context("missing challenge expiration")?,
    )?;
    socket.send(Message::Text(serde_json::json!({"a":"join","v":3,"binding":config.identity.public_binding(),"signature":hex::encode(config.identity.sign(&signed)),"epoch":config.policy.borrow().epoch}).to_string().into())).await?;
    let ready = read(&mut socket).await?;
    if ready["a"] != "ready" || ready["role"] != "host" {
        bail!("secure host admission failed")
    }
    status.send_replace(true);
    let sid = ready["sid"]
        .as_str()
        .context("missing host epoch")?
        .to_string();
    let stop = shutdown.child_token();
    let _cancel_on_drop = stop.clone().drop_guard();
    let mut peers: BTreeMap<String, (mpsc::Sender<WireFrame>, CancellationToken)> = BTreeMap::new();
    let (tx, mut outgoing) = mpsc::channel::<serde_json::Value>(64);
    let mut tick = tokio::time::interval(Duration::from_secs(15));
    let mut policy = config.policy.clone();
    let mut last_heartbeat = std::time::Instant::now();
    let result=async {loop{tokio::select!{
  _=shutdown.cancelled()=>return Ok(()),
  _=tick.tick()=>{if last_heartbeat.elapsed()>Duration::from_secs(45){bail!("secure relay heartbeat timed out")}tokio::time::timeout(Duration::from_secs(5),socket.send(Message::Text(serde_json::json!({"a":"heartbeat","v":3}).to_string().into()))).await??;},
  r=policy.changed()=>{r.context("secure policy stopped")?;socket.send(Message::Text(serde_json::json!({"a":"policy","v":3,"epoch":policy.borrow().epoch}).to_string().into())).await?;},
  frame=outgoing.recv()=>{let frame=frame.context("secure writer closed")?;if frame["a"]=="cancel"{if let Some(rid)=frame["rid"].as_str(){if let Some((_,cancel))=peers.remove(rid){cancel.cancel();}}}tokio::time::timeout(Duration::from_secs(5),socket.send(Message::Text(frame.to_string().into()))).await??;},
  msg=socket.next()=>{let msg=msg.context("secure socket closed")??;match msg {
   Message::Text(text)=>{let value:serde_json::Value=serde_json::from_str(&text)?;if value["v"]!=3{bail!("wrong secure version")}
    match value["a"].as_str().unwrap_or("") {
     "heartbeat"=>{if value["sid"]!=sid{bail!("secure host epoch changed")}last_heartbeat=std::time::Instant::now();},
     "open"=>{
      let admission:PeerAdmission=serde_json::from_value(value.clone())?;if admission.sid!=sid||peers.contains_key(&admission.rid)||peers.len()>=32{bail!("invalid secure peer admission")}let rid=admission.rid.clone();
      if admission.validate(&config).is_err(){socket.send(Message::Text(serde_json::json!({"a":"cancel","v":3,"sid":sid,"rid":rid}).to_string().into())).await?;continue}
      socket.send(Message::Text(serde_json::json!({"a":"accept","v":3,"sid":sid,"rid":rid}).to_string().into())).await?;
      let (input,rx)=mpsc::channel((WIRE_WINDOW * 2) as usize);let (frames,mut wire)=mpsc::channel((WIRE_WINDOW * 2) as usize);let cancel=stop.child_token();peers.insert(rid.clone(),(input,cancel.clone()));let outgoing=tx.clone();let peerconfig=config.clone();let sid=sid.clone();let delivery=delivery.clone();
      tokio::spawn(async move{let mut task=tokio::spawn(serve_peer_with_delivery(peerconfig,admission,rx,frames,cancel.clone(),Some(delivery)));loop{tokio::select!{_=cancel.cancelled()=>break,result=&mut task=>{if let Ok(Err(error))=result{eprintln!("Mold secure peer closed ({})",safe_relay_error_category(&error));}break},frame=wire.recv()=>{let Some(frame)=frame else{break};if outgoing.send(serde_json::to_value(frame).unwrap_or_default()).await.is_err(){break}}}}cancel.cancel();task.abort();let _=outgoing.send(serde_json::json!({"a":"cancel","v":3,"sid":sid,"rid":rid})).await;});

     },
     "data"|"ack"=>{let frame:WireFrame=serde_json::from_value(value)?;if frame.sid!=sid{bail!("wrong secure epoch")}
if let Some((input,cancel))=peers.get(&frame.rid){if input.try_send(frame).is_err(){cancel.cancel();}}},
     "cancel"=>{if value["sid"]!=sid{bail!("wrong secure epoch")}
if let Some((_,cancel))=value["rid"].as_str().and_then(|rid|peers.remove(rid)){cancel.cancel();}},
     _=>bail!("unexpected secure control"),
    }
   },Message::Ping(data)=>socket.send(Message::Pong(data)).await?,Message::Pong(_)=>{},Message::Close(_)=>return Ok(()),_=>bail!("unexpected secure binary message"),
  }}
 }}}.await;
    stop.cancel();
    result
}
