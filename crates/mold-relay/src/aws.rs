//! Lambda/API Gateway transport. TCP bytes are ordered and acknowledged per request.
use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};

const WINDOW: usize = 4;
const FRAME_LIMIT: usize = 24 * 1024;
const PAYLOAD: usize = 16 * 1024;
const GAP: Duration = Duration::from_secs(30);
const RETRY: Duration = Duration::from_secs(10);

#[derive(Clone, Serialize, Deserialize)]
struct Frame {
    a: String,
    v: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    sid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    d: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    credit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ack_seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
}
impl Frame {
    fn new(action: &str, sid: &str, rid: &str) -> Self {
        Self {
            a: action.into(),
            v: 2,
            sid: Some(sid.into()),
            rid: Some(rid.into()),
            seq: None,
            d: None,
            next: None,
            credit: None,
            ack_seq: None,
            role: None,
        }
    }
    fn text(&self) -> Result<Message> {
        let text = serde_json::to_string(self)?;
        if text.len() > FRAME_LIMIT {
            bail!("AWS relay frame exceeds limit");
        }
        Ok(Message::Text(text.into()))
    }
    fn parse(text: &str) -> Result<Self> {
        if text.len() > FRAME_LIMIT {
            bail!("AWS relay frame exceeds limit");
        }
        let frame: Self =
            serde_json::from_str(text).map_err(|_| anyhow::anyhow!("invalid AWS relay frame"))?;
        if frame.v != 2
            || frame
                .sid
                .as_ref()
                .is_some_and(|s| s.is_empty() || s.len() > 128)
            || frame
                .rid
                .as_ref()
                .is_some_and(|s| s.is_empty() || s.len() > 128)
        {
            bail!("invalid AWS relay identity");
        }
        Ok(frame)
    }
}
#[derive(Clone)]
struct Payload {
    bytes: Vec<u8>,
    eof: bool,
}
impl Payload {
    fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update([u8::from(self.eof)]);
        hash.update(&self.bytes);
        hash.finalize().into()
    }
}
struct Inbox {
    next: u64,
    held: BTreeMap<u64, Payload>,
    history: VecDeque<(u64, [u8; 32])>,
    eof: bool,
}
impl Inbox {
    fn new() -> Self {
        Self {
            next: 0,
            held: BTreeMap::new(),
            history: VecDeque::new(),
            eof: false,
        }
    }
    fn insert(&mut self, seq: u64, payload: Payload) -> Result<Vec<Payload>> {
        if payload.bytes.len() > PAYLOAD || seq > u32::MAX.into() {
            bail!("invalid AWS relay payload");
        }
        if seq < self.next {
            if self
                .history
                .iter()
                .any(|(n, d)| *n == seq && *d == payload.digest())
            {
                return Ok(Vec::new());
            }
            bail!("conflicting or expired AWS relay duplicate");
        }
        if self.eof || seq >= self.next + WINDOW as u64 {
            bail!("AWS relay receive window exceeded");
        }
        if let Some(previous) = self.held.get(&seq) {
            if previous.digest() != payload.digest() {
                bail!("conflicting AWS relay duplicate");
            }
            return Ok(Vec::new());
        }
        self.held.insert(seq, payload);
        let mut ready = Vec::new();
        while let Some(payload) = self.held.remove(&self.next) {
            if self.eof {
                bail!("AWS relay data after EOF");
            }
            self.history.push_back((self.next, payload.digest()));
            if self.history.len() > WINDOW * 2 {
                self.history.pop_front();
            }
            self.next += 1;
            self.eof = payload.eof;
            ready.push(payload);
        }
        Ok(ready)
    }
}

fn validate_request_header(bytes: &[u8]) -> Result<usize> {
    let text = std::str::from_utf8(bytes).context("invalid AWS relay HTTP header")?;
    let line = text
        .lines()
        .next()
        .context("missing AWS relay request line")?;
    let mut words = line.split_whitespace();
    let method = words.next().context("missing method")?;
    let path = words.next().context("missing path")?;
    if method == "CONNECT"
        || !path.starts_with('/')
        || path.starts_with("//")
        || path.split('?').next() == Some("/metrics")
    {
        bail!("AWS relay request forbidden");
    }
    if words.next() != Some("HTTP/1.1") || words.next().is_some() {
        bail!("invalid AWS relay HTTP version");
    }
    let mut length = None;
    let mut close = false;
    for line in text.lines().skip(1).filter(|line| !line.is_empty()) {
        if line.starts_with([' ', '\t']) {
            bail!("AWS relay folded header forbidden");
        }
        let (key, value) = line.split_once(':').context("invalid AWS relay header")?;
        let value = value.trim();
        if key.eq_ignore_ascii_case("upgrade") || key.eq_ignore_ascii_case("transfer-encoding") {
            bail!("AWS relay upgrade or transfer encoding forbidden");
        }
        if key.eq_ignore_ascii_case("content-length") {
            if length.is_some() || value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                bail!("ambiguous AWS relay body length");
            }
            let n: usize = value.parse().context("invalid AWS relay body length")?;
            if n > 64 * 1024 * 1024 {
                bail!("AWS relay body exceeds limit");
            }
            length = Some(n);
        }
        if key.eq_ignore_ascii_case("connection") {
            if close || !value.eq_ignore_ascii_case("close") {
                bail!("AWS relay requires connection close");
            }
            close = true;
        }
    }
    if !close {
        bail!("AWS relay requires connection close");
    }
    length.context("AWS relay requires explicit body length")
}

struct RequestBoundary {
    header: Vec<u8>,
    remaining: Option<usize>,
}
impl RequestBoundary {
    fn new() -> Self {
        Self {
            header: Vec::new(),
            remaining: None,
        }
    }
    fn push(&mut self, bytes: Vec<u8>) -> Result<Option<Vec<u8>>> {
        if let Some(remaining) = self.remaining.as_mut() {
            if bytes.len() > *remaining {
                bail!("AWS relay data exceeds request body");
            }
            *remaining -= bytes.len();
            return Ok((!bytes.is_empty()).then_some(bytes));
        }
        self.header.extend_from_slice(&bytes);
        let Some(end) = self.header.windows(4).position(|p| p == b"\r\n\r\n") else {
            if self.header.len() > 64 * 1024 {
                bail!("AWS relay HTTP header exceeds limit");
            }
            return Ok(None);
        };
        let end = end + 4;
        if end > 64 * 1024 {
            bail!("AWS relay HTTP header exceeds limit");
        }
        let length = validate_request_header(&self.header[..end])?;
        let body = self.header.len() - end;
        if body > length {
            bail!("AWS relay data exceeds request body");
        }
        self.remaining = Some(length - body);
        Ok(Some(std::mem::take(&mut self.header)))
    }
    fn finish(&self) -> Result<()> {
        if self.remaining != Some(0) {
            bail!("incomplete AWS relay HTTP request");
        }
        Ok(())
    }
}

struct ResponseAdmission {
    head: Vec<u8>,
    final_response: bool,
}
impl ResponseAdmission {
    fn new() -> Self {
        Self {
            head: Vec::new(),
            final_response: false,
        }
    }
    fn push(&mut self, bytes: &[u8]) -> bool {
        if self.final_response {
            return true;
        }
        if self.head.len() + bytes.len() > 64 * 1024 {
            self.final_response = true;
            return true;
        }
        self.head.extend_from_slice(bytes);
        loop {
            let Some(line_end) = self.head.windows(2).position(|p| p == b"\r\n") else {
                return false;
            };
            let status = std::str::from_utf8(&self.head[..line_end])
                .ok()
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u16>().ok());
            if status.is_some_and(|n| (200..=599).contains(&n)) {
                self.final_response = true;
                return true;
            }
            let Some(end) = self.head.windows(4).position(|p| p == b"\r\n\r\n") else {
                return false;
            };
            self.head.drain(..end + 4);
        }
    }
}
fn request_ack(sid: &str, rid: &str, inbox: &Inbox, queued: usize, serial: &mut u64) -> Frame {
    let mut ack = Frame::new("ack", sid, rid);
    ack.next = Some(inbox.next);
    ack.credit = Some(WINDOW.saturating_sub(inbox.held.len() + queued));
    ack.ack_seq = Some(*serial);
    *serial += 1;
    ack
}

/// AWS endpoints may include the API Gateway deployment stage path.
pub fn validate_aws_endpoint(endpoint: &str, allow_loopback_ws: bool) -> Result<url::Url> {
    let url =
        url::Url::parse(endpoint).map_err(|_| anyhow::anyhow!("invalid AWS relay endpoint"))?;
    let mut origin = url.clone();
    origin.set_path("/");
    validate_endpoint(origin.as_str(), allow_loopback_ws)?;
    if url.query().is_some() || url.fragment().is_some() {
        bail!("AWS relay endpoint must not contain query credentials");
    }
    Ok(url)
}
/// One authenticated outbound host session. Transport requests never replay on reconnect.
pub async fn connect(
    endpoint: &str,
    target: SocketAddr,
    token: String,
    allow_loopback_ws: bool,
    shutdown: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    connect_managed(
        endpoint,
        target,
        token,
        None,
        allow_loopback_ws,
        shutdown,
        options,
    )
    .await
}

/// Managed enrollment uses an isolated host namespace; legacy callers omit it.
pub fn validate_host_namespace(host_id: &str) -> Result<()> {
    if host_id.len() != 32
        || !host_id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        bail!("invalid relay host namespace");
    }
    Ok(())
}

/// One cancellable outbound session with an optional managed namespace.
#[allow(clippy::too_many_arguments)]
pub async fn connect_managed(
    endpoint: &str,
    target: SocketAddr,
    token: String,
    host_id: Option<String>,
    allow_loopback_ws: bool,
    shutdown: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    if let Some(host_id) = host_id.as_deref() {
        validate_host_namespace(host_id)?;
    }
    ensure_tls_provider();
    let endpoint = validate_aws_endpoint(endpoint, allow_loopback_ws)?;
    let token = validate_token(&token)?;
    validate_target(target)?;
    options.validate()?;
    drop(authenticated_target(target).await?);
    let mut backoff = Duration::from_secs(5);
    loop {
        diagnostic("connecting");
        let started = tokio::time::Instant::now();
        let result = tokio::select! {_ = shutdown.cancelled()=>return Ok(()),result=session(&endpoint,target,&token,host_id.as_deref(),shutdown.clone(),options)=>result};
        if result
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string() == "relay enrollment rejected")
        {
            return result;
        }
        backoff = reconnect_delay(backoff, started.elapsed());
        diagnostic("reconnecting");
        tokio::select! {_ = shutdown.cancelled()=>return Ok(()),_ = tokio::time::sleep(backoff)=>{}}
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}
async fn next_outgoing(
    cancel: &CancellationToken,
    heartbeat: &mut mpsc::Receiver<Frame>,
    controls: &mut mpsc::Receiver<Frame>,
    data: &mut mpsc::Receiver<Frame>,
) -> Option<Frame> {
    tokio::select! {biased;_ = cancel.cancelled()=>None,frame=heartbeat.recv()=>frame,frame=controls.recv()=>frame,frame=data.recv()=>frame}
}
fn writer_cadence() -> tokio::time::Interval {
    let mut cadence = tokio::time::interval(Duration::from_millis(5));
    cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    cadence
}
fn reconnect_delay(previous: Duration, session_age: Duration) -> Duration {
    if session_age >= Duration::from_secs(90) {
        Duration::from_secs(5)
    } else {
        previous
    }
}
fn stream_diagnostic(error: &anyhow::Error) {
    let message = error.to_string();
    let category = match message.as_str() {
        "AWS relay acknowledgement timed out" => "stream_ack_timeout",
        "AWS relay stream stalled" => "stream_idle_or_gap_timeout",
        "AWS relay data exceeds request body" => "stream_body_length_exceeded",
        "incomplete AWS relay HTTP request" => "stream_body_incomplete",
        "AWS relay receive window exceeded" => "stream_receive_window_exceeded",
        "AWS relay requires explicit body length" => "stream_length_missing",
        "AWS relay requires connection close" => "stream_connection_not_close",
        "AWS relay body exceeds limit" => "stream_body_limit",
        _ => "stream_other_failure",
    };
    diagnostic(category);
}
fn close_metadata(
    frame: Option<&tokio_tungstenite::tungstenite::protocol::CloseFrame>,
) -> (Option<u16>, usize) {
    frame
        .map(|frame| (Some(frame.code.into()), frame.reason.len()))
        .unwrap_or((None, 0))
}
fn diagnostic(reason: &'static str) {
    if std::env::var_os("MOLD_RELAY_DIAGNOSTICS").is_some() {
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        eprintln!("mold-relay: unix_s={time} {reason}");
    }
}

async fn next_text<S>(source: &mut S) -> Result<Option<String>>
where
    S: futures_util::Stream<
            Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>,
        > + Unpin,
{
    loop {
        match source.next().await {
            Some(Ok(Message::Text(text))) => return Ok(Some(text.to_string())),
            Some(Ok(Message::Ping(_))) => diagnostic("websocket_ping"),
            Some(Ok(Message::Pong(_))) => diagnostic("websocket_pong"),
            Some(Ok(Message::Close(frame))) => {
                if std::env::var_os("MOLD_RELAY_DIAGNOSTICS").is_some() {
                    let (code, length) = close_metadata(frame.as_ref());
                    eprintln!(
                        "mold-relay: websocket_close_code={} reason_bytes={length}",
                        code.unwrap_or(1005)
                    );
                }
                diagnostic("websocket_closed");
                return Ok(None);
            }
            None => {
                diagnostic("websocket_eof");
                return Ok(None);
            }
            Some(Ok(_)) => {
                diagnostic("unexpected_websocket_frame");
                bail!("invalid AWS relay WebSocket frame");
            }
            Some(Err(_)) => {
                diagnostic("websocket_read_failed");
                bail!("AWS relay WebSocket read failed");
            }
        }
    }
}

fn host_request(
    endpoint: &url::Url,
    token: &str,
    host_id: Option<&str>,
) -> Result<tokio_tungstenite::tungstenite::http::Request<()>> {
    let mut request = ws_request(endpoint, token)?;
    if let Some(host_id) = host_id {
        validate_host_namespace(host_id)?;
        request
            .headers_mut()
            .insert("x-mold-relay-host", host_id.parse()?);
    }
    request
        .headers_mut()
        .insert("x-mold-relay-role", "host".parse()?);
    Ok(request)
}

async fn session(
    endpoint: &url::Url,
    target: SocketAddr,
    token: &str,
    host_id: Option<&str>,
    shutdown: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    let request = host_request(endpoint, token, host_id)?;
    let (socket,_)=tokio::time::timeout(ATTACH,connect_async_with_config(request,Some(websocket_config()),false)).await.context("AWS relay connection timed out")?.map_err(|e| {
        if let tokio_tungstenite::tungstenite::Error::Http(response)=&e {
            if std::env::var_os("MOLD_RELAY_DIAGNOSTICS").is_some(){eprintln!("mold-relay: handshake_http_status={}", response.status().as_u16());}
        } else {diagnostic("handshake_transport_failed");}
        if matches!(&e,tokio_tungstenite::tungstenite::Error::Http(r) if r.status()==401||r.status()==403){anyhow::anyhow!("relay enrollment rejected")}else{anyhow::anyhow!("AWS relay unavailable")}
    })?;
    let (mut sink, mut source) = socket.split();
    let mut hello = Frame::new("hello", "", "");
    hello.sid = None;
    hello.rid = None;
    sink.send(hello.text()?)
        .await
        .map_err(|_| anyhow::anyhow!("AWS relay hello failed"))?;
    let text = tokio::time::timeout(ATTACH, next_text(&mut source))
        .await
        .context("AWS relay ready timed out")??
        .context("AWS relay ready unavailable")?;
    let ready = Frame::parse(&text)?;
    if ready.a != "ready" || ready.role.as_deref() != Some("host") {
        bail!("invalid AWS relay ready");
    }
    let sid = ready.sid.context("missing AWS relay epoch")?;
    diagnostic("host_ready");
    let cancel = shutdown.child_token();
    let (heartbeat_tx, mut heartbeat_rx) = mpsc::channel::<Frame>(1);
    let (control_tx, mut control_rx) = mpsc::channel::<Frame>(128);
    let (data_tx, mut data_rx) = mpsc::channel::<Frame>(128);
    let writer_cancel = cancel.clone();
    let writer = tokio::spawn(async move {
        // Priority controls, and a shared rate below API Gateway's route throttle.
        let mut cadence = writer_cadence();
        loop {
            let frame = next_outgoing(
                &writer_cancel,
                &mut heartbeat_rx,
                &mut control_rx,
                &mut data_rx,
            )
            .await;
            let Some(frame) = frame else {
                break;
            };
            tokio::select! {_ = writer_cancel.cancelled()=>break,_ = cadence.tick()=>{}}
            let Ok(message) = frame.text() else {
                break;
            };
            if !send_bounded(&mut sink, message, &writer_cancel).await {
                diagnostic("websocket_write_failed");
                break;
            }
        }
        writer_cancel.cancel();
    });
    let capacity = Arc::new(Semaphore::new(32));
    let mut requests: HashMap<String, mpsc::Sender<Frame>> = HashMap::new();
    let mut tasks = tokio::task::JoinSet::new();
    let mut heartbeat = tokio::time::interval(Duration::from_secs(30));
    let expiry = tokio::time::sleep(Duration::from_secs(110 * 60));
    tokio::pin!(expiry);
    let mut last = tokio::time::Instant::now();
    loop {
        tokio::select! {biased;
            _ = cancel.cancelled()=>{diagnostic("session_cancelled");break;},
            _ = heartbeat.tick()=>{
                if last.elapsed()>Duration::from_secs(90){diagnostic("heartbeat_timeout");break;}
                let mut ping=Frame::new("heartbeat","","");ping.sid=None;ping.rid=None;
                if heartbeat_tx.try_send(ping).is_err(){diagnostic("heartbeat_queue_full");break;}
            }
            _ = &mut expiry=>{diagnostic("session_rotation");break;},
            result=tasks.join_next(),if !tasks.is_empty()=>{if let Some(Ok(rid))=result{requests.remove(&rid);}},
            message=next_text(&mut source)=>{
                let text=match message { Ok(Some(text))=>text, _=>break };
                let frame=match Frame::parse(&text){Ok(frame)=>frame,Err(_)=>{
                    if serde_json::from_str::<serde_json::Value>(&text).ok().is_some_and(|value|value.get("message").is_some()) {diagnostic("api_gateway_error_envelope");} else {diagnostic("incoming_frame_invalid");} break;
                }};
                if frame.a=="heartbeat"{last=tokio::time::Instant::now();continue;}
                if frame.sid.as_deref()!=Some(&sid){diagnostic("epoch_mismatch");break;}
                last=tokio::time::Instant::now();
                let Some(rid)=frame.rid.clone() else {diagnostic("missing_stream_identity");break;};
                if frame.a=="open" {
                    if requests.contains_key(&rid){continue;}
                    let Ok(permit)=capacity.clone().try_acquire_owned() else {let _=control_tx.try_send(Frame::new("cancel",&sid,&rid));continue;};
                    let (tx,rx)=mpsc::channel(16);requests.insert(rid.clone(),tx);
                    let sid=sid.clone();let controls=control_tx.clone();let data=data_tx.clone();let stop=cancel.child_token();
                    tasks.spawn(async move {let _permit=permit;if let Err(error)=request_stream(target,(&sid,&rid),rx,controls.clone(),data,stop,options).await { stream_diagnostic(&error); let _=controls.try_send(Frame::new("cancel",&sid,&rid)); } rid});
                } else if let Some(tx)=requests.get(&rid) {
                    if tx.try_send(frame).is_err(){requests.remove(&rid);let _=control_tx.try_send(Frame::new("cancel",&sid,&rid));}
                }
            },

        }
    }
    cancel.cancel();
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    writer.await?;
    Ok(())
}
struct AckState {
    serial: Option<u64>,
    next: u64,
    credit: usize,
}
impl AckState {
    fn new() -> Self {
        Self {
            serial: None,
            next: 0,
            credit: WINDOW,
        }
    }
    fn apply(&mut self, serial: u64, next: u64, credit: usize, sent: u64) -> Result<bool> {
        if next > sent || credit > WINDOW || serial > 9_007_199_254_740_991 {
            bail!("invalid AWS relay acknowledgement");
        }
        if self.serial.is_some_and(|last| serial <= last) {
            return Ok(false);
        }
        if next < self.next {
            bail!("regressive AWS relay acknowledgement");
        }
        self.serial = Some(serial);
        self.next = next;
        self.credit = credit;
        Ok(true)
    }
}

struct Pending {
    frame: Frame,
    sent: tokio::time::Instant,
    retries: u8,
}
async fn request_stream(
    target: SocketAddr,
    identity: (&str, &str),
    mut input: mpsc::Receiver<Frame>,
    controls: mpsc::Sender<Frame>,
    output: mpsc::Sender<Frame>,
    cancel: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    let (sid, rid) = identity;
    let tcp = tokio::select! {_ = cancel.cancelled()=>return Ok(()),result=authenticated_target(target)=>result?};
    let (mut tcp_read, tcp_write) = tcp.into_split();
    controls
        .send(Frame::new("accept", sid, rid))
        .await
        .map_err(|_| anyhow::anyhow!("AWS relay closed"))?;
    let mut inbox = Inbox::new();
    let mut seq = 0;
    let mut acknowledgement = AckState::new();
    let mut ack_out = 0;
    let mut pending: BTreeMap<u64, Pending> = BTreeMap::new();
    let mut buffer = vec![0; PAYLOAD];
    let mut read_eof = false;
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut activity = tokio::time::Instant::now();
    let mut gap_since = None;
    let mut request = RequestBoundary::new();
    let mut response = ResponseAdmission::new();
    let mut writes: VecDeque<Vec<u8>> = VecDeque::new();
    let mut write_offset = 0;
    let mut write_failed = false;
    loop {
        tokio::select! {biased;
            _ = cancel.cancelled()=>return Ok(()),
            result=tcp_read.read(&mut buffer),if !read_eof && pending.len()<WINDOW && acknowledgement.credit>0=>{
                let n=match result {Ok(n)=>n,Err(_) if response.final_response=>0,Err(error)=>return Err(error.into())};
                if n>0 && response.push(&buffer[..n]) {writes.clear();write_offset=0;write_failed=true;}
                let mut frame=Frame::new(if n==0{"eof"}else{"data"},sid,rid);frame.seq=Some(seq);
                if n==0{read_eof=true;}else{frame.d=Some(STANDARD.encode(&buffer[..n]));activity=tokio::time::Instant::now();}
                output.send(frame.clone()).await.map_err(|_|anyhow::anyhow!("AWS relay closed"))?;
                pending.insert(seq,Pending{frame,sent:tokio::time::Instant::now(),retries:0});seq+=1;acknowledgement.credit-=1;
            },
            frame=input.recv()=>{
                let Some(frame)=frame else{return Ok(());};
                match frame.a.as_str() {
                    "cancel"=>return Ok(()),
                    "ack"=>{
                        let next=frame.next.context("missing AWS relay acknowledgement")?;
                        let available=frame.credit.context("missing AWS relay credit")?;
                        let serial=frame.ack_seq.context("missing AWS relay acknowledgement serial")?;
                        if acknowledgement.apply(serial,next,available,seq)? { pending.retain(|n,_|*n>=next); }
                    },
                    "data"|"eof"=>{
                        let n=frame.seq.context("missing AWS relay sequence")?;
                        let payload=if frame.a=="eof"{Payload{bytes:Vec::new(),eof:true}}else{Payload{bytes:STANDARD.decode(frame.d.context("missing AWS relay payload")?).map_err(|_|anyhow::anyhow!("invalid AWS relay base64"))?,eof:false}};
                        let ready=inbox.insert(n,payload)?;
                        for payload in ready {
                            if payload.eof { request.finish()?; }
                            if !payload.eof {
                                let Some(bytes) = request.push(payload.bytes)? else { continue; };
                                if !write_failed {anyhow::ensure!(writes.len()<WINDOW*2,"AWS relay target write queue exceeded");writes.push_back(bytes);}
                                activity=tokio::time::Instant::now();
                            }
                            // Request EOF is logical completion, never TCP FIN.
                        }
                        if inbox.held.is_empty(){gap_since=None;}else if gap_since.is_none(){gap_since=Some(tokio::time::Instant::now());}
                        let ack=request_ack(sid,rid,&inbox,writes.len(),&mut ack_out);
                        controls.send(ack).await.map_err(|_|anyhow::anyhow!("AWS relay closed"))?;
                    },
                    _=>bail!("invalid AWS relay request action"),
                }
            },
            readiness=tcp_write.writable(),if !write_failed && !writes.is_empty()=>{
                readiness?;
                let bytes=writes.front().unwrap();
                match tcp_write.try_write(&bytes[write_offset..]) {
                    Ok(0)=>{write_failed=true;writes.clear();write_offset=0;diagnostic("target_write_closed");},
                    Ok(n)=>{write_offset+=n;activity=tokio::time::Instant::now();if write_offset==bytes.len(){writes.pop_front();write_offset=0;let ack=request_ack(sid,rid,&inbox,writes.len(),&mut ack_out);controls.send(ack).await.map_err(|_|anyhow::anyhow!("AWS relay closed"))?;}},
                    Err(error) if error.kind()==std::io::ErrorKind::WouldBlock=>{},
                    Err(_)=>{write_failed=true;writes.clear();write_offset=0;diagnostic("target_write_failed");},
                }
            },
            _ = tick.tick()=>{
                if activity.elapsed()>options.idle_timeout || gap_since.is_some_and(|t:tokio::time::Instant|t.elapsed()>GAP){bail!("AWS relay stream stalled");}
                for item in pending.values_mut(){if item.sent.elapsed()>=RETRY{if item.retries>=3{bail!("AWS relay acknowledgement timed out");}output.send(item.frame.clone()).await.map_err(|_|anyhow::anyhow!("AWS relay closed"))?;item.retries+=1;item.sent=tokio::time::Instant::now();}}
            }
        }
        if read_eof && pending.is_empty() {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn host_namespace_is_only_attached_to_managed_handshakes() {
        let endpoint = url::Url::parse("wss://relay.example.com/ws").unwrap();
        let token = "a".repeat(64);
        let host = "b".repeat(32);
        let managed = host_request(&endpoint, &token, Some(&host)).unwrap();
        assert_eq!(managed.headers()["x-mold-relay-host"], host);
        assert_eq!(managed.headers()["x-mold-relay-role"], "host");
        assert!(host_request(&endpoint, &token, None)
            .unwrap()
            .headers()
            .get("x-mold-relay-host")
            .is_none());
    }
    #[test]
    fn managed_namespace_rejects_ambiguous_host_ids() {
        assert!(validate_host_namespace("0123456789abcdef0123456789abcdef").is_ok());
        for invalid in ["", "legacy", "0123456789ABCDEF0123456789ABCDEF", "a\r\nb"] {
            assert!(validate_host_namespace(invalid).is_err());
        }
    }

    use super::*;
    fn bytes(value: &[u8]) -> Payload {
        Payload {
            bytes: value.to_vec(),
            eof: false,
        }
    }
    async fn read_headers(tcp: &mut TcpStream) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut byte = [0];
        while !bytes.ends_with(b"\r\n\r\n") {
            if tcp.read_exact(&mut byte).await.is_err() {
                return Vec::new();
            }
            bytes.push(byte[0]);
        }
        bytes
    }
    #[tokio::test]
    async fn aws_early_response_survives_an_inflight_request_body() {
        let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = target.local_addr().unwrap();
        let fixture = tokio::spawn(async move {
            let (mut tcp, _) = target.accept().await.unwrap();
            assert!(read_headers(&mut tcp).await.starts_with(b"GET /api/status"));
            tcp.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            assert!(read_headers(&mut tcp).await.starts_with(b"PUT /api/upload"));
            tcp.write_all(b"HTTP/1.1 415 Unsupported Media Type\r\nContent-Length: 6\r\nConnection: close\r\n\r\nrefuse").await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        });
        let (tx, rx) = mpsc::channel(32);
        let (controls, mut control_rx) = mpsc::channel(32);
        let (output, mut output_rx) = mpsc::channel(32);
        let task = tokio::spawn(request_stream(
            address,
            ("epoch", "guest"),
            rx,
            controls,
            output,
            CancellationToken::new(),
            RelayOptions::default(),
        ));
        assert_eq!(control_rx.recv().await.unwrap().a, "accept");
        for seq in 0..4 {
            let mut frame = Frame::new("data", "epoch", "guest");
            frame.seq = Some(seq);
            let bytes = if seq == 0 {
                b"PUT /api/upload HTTP/1.1\r\nContent-Length: 65536\r\nConnection: close\r\n\r\n"
                    .to_vec()
            } else {
                vec![42; PAYLOAD]
            };
            frame.d = Some(STANDARD.encode(bytes));
            tx.send(frame).await.unwrap();
        }
        let mut bytes = Vec::new();
        let mut serial = 0;
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let frame = output_rx.recv().await.unwrap();
                if let Some(data) = frame.d {
                    bytes.extend(STANDARD.decode(data).unwrap());
                }
                let mut ack = Frame::new("ack", "epoch", "guest");
                ack.next = Some(frame.seq.unwrap() + 1);
                ack.credit = Some(WINDOW);
                ack.ack_seq = Some(serial);
                serial += 1;
                tx.send(ack).await.unwrap();
                if frame.a == "eof" {
                    break;
                }
            }
            task.await.unwrap().unwrap();
        })
        .await
        .unwrap();
        assert!(bytes.starts_with(b"HTTP/1.1 415"));
        assert!(bytes.ends_with(b"refuse"));
        fixture.await.unwrap();
    }
    #[tokio::test]
    async fn aws_request_reorders_without_replay_and_logical_eof_retains_http_response() {
        aws_forwarding_fixture(None).await;
    }
    #[tokio::test]
    async fn managed_aws_handshake_and_http_forwarding_keep_namespace() {
        aws_forwarding_fixture(Some("0123456789abcdef0123456789abcdef")).await;
    }
    async fn aws_forwarding_fixture(host_id: Option<&'static str>) {
        let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target_addr = target.local_addr().unwrap();
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        tokio::spawn(async move {
            loop {
                let incoming =
                    tokio::select! {_ = stop.cancelled()=>break,result=target.accept()=>result};
                let (mut tcp, _) = incoming.unwrap();
                tokio::spawn(async move {
                    let probe = read_headers(&mut tcp).await;
                    assert!(probe.starts_with(b"GET /api/status "));
                    tcp.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                        .await
                        .unwrap();
                    let request = read_headers(&mut tcp).await;
                    if request.is_empty() {
                        return;
                    }
                    assert!(request.starts_with(b"POST /echo "));
                    let mut body = [0; 7];
                    tcp.read_exact(&mut body).await.unwrap();
                    assert_eq!(&body, b"payload");
                    // No TCP half-close: a valid response is available before FIN.
                    tcp.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\npayload",
                    )
                    .await
                    .unwrap();
                });
            }
        });
        let gateway = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("ws://{}/production", gateway.local_addr().unwrap());
        let (done_tx, done_rx) = oneshot::channel();
        let done_tx = Arc::new(Mutex::new(Some(done_tx)));
        let app=Router::new().route("/production",get(move|headers:HeaderMap,ws:WebSocketUpgrade|{let done_tx=done_tx.clone();async move{assert_eq!(headers["x-mold-relay-role"],"host");assert_eq!(headers.get("x-mold-relay-host").and_then(|h|h.to_str().ok()),host_id);ws.on_upgrade(move|mut socket|async move{
            let Some(Ok(AxMessage::Text(text)))=socket.recv().await else{panic!("missing hello")};assert_eq!(Frame::parse(&text).unwrap().a,"hello");
            let mut ready=Frame::new("ready","epoch","host");ready.role=Some("host".into());
            // Send ready directly as text (the v2 envelope is not v1 Wire).
            socket.send(AxMessage::Ping(Vec::new().into())).await.unwrap();
            socket.send(AxMessage::Text(serde_json::to_string(&ready).unwrap().into())).await.unwrap();
            socket.send(AxMessage::Pong(Vec::new().into())).await.unwrap();
            socket.send(AxMessage::Text(serde_json::to_string(&Frame::new("open","epoch","guest")).unwrap().into())).await.unwrap();
            loop{let message=socket.recv().await.unwrap().unwrap();let AxMessage::Text(text)=message else {continue;};let frame=Frame::parse(&text).unwrap();if frame.a=="accept"{break;}}
            let request=b"POST /echo HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\nContent-Length: 7\r\n\r\npayload";
            for(seq,bytes)in[(1,&request[40..]),(0,&request[..40]),(0,&request[..40])]{let mut data=Frame::new("data","epoch","guest");data.seq=Some(seq);data.d=Some(STANDARD.encode(bytes));socket.send(AxMessage::Text(serde_json::to_string(&data).unwrap().into())).await.unwrap();}
            let mut eof=Frame::new("eof","epoch","guest");eof.seq=Some(2);socket.send(AxMessage::Text(serde_json::to_string(&eof).unwrap().into())).await.unwrap();
            let mut response=Vec::new();let mut next=0;
            loop{let Some(Ok(AxMessage::Text(text)))=socket.recv().await else{panic!("response transport closed")};let frame=Frame::parse(&text).unwrap();if frame.a=="data"||frame.a=="eof"{assert_eq!(frame.seq,Some(next));next+=1;if let Some(data)=frame.d{response.extend(STANDARD.decode(data).unwrap());}let mut ack=Frame::new("ack","epoch","guest");ack.next=Some(next);ack.credit=Some(4);ack.ack_seq=Some(next);socket.send(AxMessage::Text(serde_json::to_string(&ack).unwrap().into())).await.unwrap();if frame.a=="eof"{break;}}}
            assert!(response.ends_with(b"payload"));
            while let Ok(Some(Ok(AxMessage::Text(text)))) = tokio::time::timeout(Duration::from_millis(100), socket.recv()).await {
                assert_ne!(Frame::parse(&text).unwrap().a, "cancel", "successful EOF must not abort a frontend that is still consuming the response");
            }
            if let Some(tx)=done_tx.lock().await.take(){let _=tx.send(());}
        })}}));
        let stop = cancel.clone();
        tokio::spawn(async move {
            axum::serve(gateway, app)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        });
        let signal = cancel.clone();
        let connector = tokio::spawn(async move {
            connect_managed(
                &endpoint,
                target_addr,
                "0123456789012345678901234567890123456789".into(),
                host_id.map(str::to_owned),
                true,
                signal,
                RelayOptions::default(),
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(5), done_rx)
            .await
            .unwrap()
            .unwrap();
        cancel.cancel();
        connector.await.unwrap().unwrap();
    }
    #[tokio::test]
    async fn busy_control_queue_cannot_starve_heartbeat() {
        let cancel = CancellationToken::new();
        let (ht, mut hr) = mpsc::channel(1);
        let (ct, mut cr) = mpsc::channel(8);
        let (dt, mut dr) = mpsc::channel(8);
        dt.send(Frame::new("data", "epoch", "guest")).await.unwrap();
        let producer = tokio::spawn(async move {
            loop {
                if ct.send(Frame::new("ack", "epoch", "guest")).await.is_err() {
                    break;
                }
            }
        });
        let send = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            ht.send(Frame::new("heartbeat", "", "")).await.unwrap();
        });
        let mut cadence = writer_cadence();
        tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                let frame = next_outgoing(&cancel, &mut hr, &mut cr, &mut dr)
                    .await
                    .unwrap();
                cadence.tick().await;
                if frame.a == "heartbeat" {
                    break;
                }
            }
        })
        .await
        .unwrap();
        cancel.cancel();
        producer.abort();
        send.await.unwrap();
    }
    #[test]
    fn websocket_close_diagnostics_expose_only_code_and_reason_length() {
        let frame = tokio_tungstenite::tungstenite::protocol::CloseFrame {
            code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Policy,
            reason: "secret must not appear".into(),
        };
        assert_eq!(close_metadata(Some(&frame)), (Some(1008), 22));
        assert_eq!(close_metadata(None), (None, 0));
    }
    #[test]
    fn empty_validated_body_chunk_is_a_noop() {
        let mut request = RequestBoundary::new();
        assert!(request
            .push(
                b"PUT /api/upload HTTP/1.1\r\nContent-Length: 4\r\nConnection: close\r\n\r\n"
                    .to_vec()
            )
            .unwrap()
            .is_some());
        assert!(request.push(Vec::new()).unwrap().is_none());
        assert!(request.finish().is_err());
        assert_eq!(request.push(b"body".to_vec()).unwrap().unwrap(), b"body");
        request.finish().unwrap();
    }
    #[test]
    fn early_final_http_response_stops_request_body_forwarding() {
        let mut response = ResponseAdmission::new();
        assert!(!response.push(b"HTTP/1.1 100 Continue\r\n\r\n"));
        assert!(!response.push(b"HTTP/1.1 4"));
        assert!(response.push(b"15 Unsupported Media Type\r\nContent-Length: 0\r\n\r\n"));
        assert!(response.push(b""));
    }
    #[tokio::test]
    async fn writer_cadence_delays_and_healthy_sessions_reset_backoff() {
        assert_eq!(
            writer_cadence().missed_tick_behavior(),
            tokio::time::MissedTickBehavior::Delay
        );
        assert_eq!(
            reconnect_delay(Duration::from_secs(30), Duration::from_secs(90)),
            Duration::from_secs(5)
        );
        assert_eq!(
            reconnect_delay(Duration::from_secs(30), Duration::from_secs(1)),
            Duration::from_secs(30)
        );
    }
    #[test]
    fn stale_ack_credit_cannot_stall_a_newer_window() {
        let mut ack = AckState::new();
        assert!(ack.apply(1, 0, 4, 4).unwrap());
        assert!(!ack.apply(0, 0, 0, 4).unwrap());
        assert_eq!(ack.credit, 4);
        assert!(!ack.apply(1, 0, 0, 4).unwrap());
        assert!(ack.apply(2, 1, 3, 4).unwrap());
        assert_eq!((ack.next, ack.credit), (1, 3));
    }
    #[test]
    fn reorder_dedup_conflicts_and_window_are_bounded() {
        let mut inbox = Inbox::new();
        assert!(inbox.insert(1, bytes(b"b")).unwrap().is_empty());
        let result = inbox.insert(0, bytes(b"a")).unwrap();
        assert_eq!(
            result
                .iter()
                .flat_map(|p| p.bytes.clone())
                .collect::<Vec<_>>(),
            b"ab"
        );
        assert!(inbox.insert(0, bytes(b"a")).unwrap().is_empty());
        assert!(inbox.insert(0, bytes(b"conflict")).is_err());
        assert!(inbox.insert(6, bytes(b"future")).is_err());
    }
    #[test]
    fn ordered_eof_rejects_later_bytes() {
        let mut inbox = Inbox::new();
        inbox
            .insert(
                0,
                Payload {
                    bytes: vec![],
                    eof: true,
                },
            )
            .unwrap();
        assert!(inbox
            .insert(
                0,
                Payload {
                    bytes: vec![],
                    eof: true
                }
            )
            .unwrap()
            .is_empty());
        assert!(inbox.insert(1, bytes(b"late")).is_err());
    }
    #[test]
    fn request_body_boundary_refuses_pipelined_bytes_and_ambiguous_headers() {
        assert!(validate_request_header(
            b"POST /api/upload HTTP/1.1\r\nContent-Length: 4\r\nConnection: keep-alive\r\n\r\n"
        )
        .is_err());
        assert!(validate_request_header(b"POST /api/upload HTTP/1.1\r\nContent-Length: 4\r\nContent-Length: 4\r\nConnection: close\r\n\r\n").is_err());
        let mut request = RequestBoundary::new();
        assert!(request.push(b"POST /api/upload HTTP/1.1\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbodyGET /metrics HTTP/1.1\r\n\r\n".to_vec()).is_err());
        let mut request = RequestBoundary::new();
        assert!(request
            .push(
                b"POST /api/upload HTTP/1.1\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbody"
                    .to_vec()
            )
            .unwrap()
            .is_some());
        assert!(request.finish().is_ok());
        assert!(request.push(b"extra".to_vec()).is_err());
    }
    #[test]
    fn request_header_blocks_private_metrics_and_upgrades() {
        assert!(
            validate_request_header(b"GET /metrics?x=1 HTTP/1.1\r\nHost: local\r\n\r\n").is_err()
        );
        assert!(validate_request_header(b"CONNECT local:80 HTTP/1.1\r\n\r\n").is_err());
        assert!(
            validate_request_header(b"GET /api/events HTTP/1.1\r\nUpgrade: websocket\r\n\r\n")
                .is_err()
        );
        assert!(
            validate_request_header(b"GET /api/status HTTP/1.1\r\nHost: local\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").is_ok()
        );
    }
    #[test]
    fn aws_stage_url_and_text_size_contract() {
        assert!(validate_aws_endpoint("wss://example.com/production", false).is_ok());
        assert!(validate_aws_endpoint("wss://example.com/production?token=x", false).is_err());
        let mut frame = Frame::new("data", "epoch", "request");
        frame.seq = Some(0);
        frame.d = Some(STANDARD.encode(vec![0; PAYLOAD]));
        assert!(frame.text().is_ok());
        frame.d = Some(STANDARD.encode(vec![0; FRAME_LIMIT]));
        assert!(frame.text().is_err());
    }
}
