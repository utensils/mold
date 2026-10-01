//! Lambda/API Gateway transport. TCP bytes are ordered and acknowledged per request.
use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};

const WINDOW: usize = 4;
const FRAME_LIMIT: usize = 24 * 1024;
const PAYLOAD: usize = 16 * 1024;
const GAP: Duration = Duration::from_secs(10);
const RETRY: Duration = Duration::from_secs(2);

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

fn validate_request_header(bytes: &[u8]) -> Result<()> {
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
    if text.lines().skip(1).any(|line| {
        line.split(':')
            .next()
            .is_some_and(|key| key.eq_ignore_ascii_case("upgrade"))
    }) {
        bail!("AWS relay upgrade forbidden");
    }
    Ok(())
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
    let endpoint = validate_aws_endpoint(endpoint, allow_loopback_ws)?;
    let token = validate_token(&token)?;
    validate_target(target)?;
    options.validate()?;
    drop(authenticated_target(target).await?);
    let mut backoff = Duration::from_secs(5);
    loop {
        let result = tokio::select! {_ = shutdown.cancelled()=>return Ok(()),result=session(&endpoint,target,&token,shutdown.clone(),options)=>result};
        if result
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string() == "relay enrollment rejected")
        {
            return result;
        }
        tokio::select! {_ = shutdown.cancelled()=>return Ok(()),_ = tokio::time::sleep(backoff)=>{}}
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}
async fn session(
    endpoint: &url::Url,
    target: SocketAddr,
    token: &str,
    shutdown: CancellationToken,
    options: RelayOptions,
) -> Result<()> {
    let mut request = ws_request(endpoint, token)?;
    request
        .headers_mut()
        .insert("x-mold-relay-role", "host".parse()?);
    let (socket,_)=tokio::time::timeout(ATTACH,connect_async_with_config(request,Some(websocket_config()),false)).await.context("AWS relay connection timed out")?.map_err(|e| {
        if matches!(&e,tokio_tungstenite::tungstenite::Error::Http(r) if r.status()==401||r.status()==403){anyhow::anyhow!("relay enrollment rejected")}else{anyhow::anyhow!("AWS relay unavailable")}
    })?;
    let (mut sink, mut source) = socket.split();
    let mut hello = Frame::new("hello", "", "");
    hello.sid = None;
    hello.rid = None;
    sink.send(hello.text()?)
        .await
        .map_err(|_| anyhow::anyhow!("AWS relay hello failed"))?;
    let ready = tokio::time::timeout(ATTACH, source.next())
        .await
        .context("AWS relay ready timed out")?;
    let Some(Ok(Message::Text(text))) = ready else {
        bail!("AWS relay ready unavailable");
    };
    let ready = Frame::parse(&text)?;
    if ready.a != "ready" || ready.role.as_deref() != Some("host") {
        bail!("invalid AWS relay ready");
    }
    let sid = ready.sid.context("missing AWS relay epoch")?;
    let cancel = shutdown.child_token();
    let (control_tx, mut control_rx) = mpsc::channel::<Frame>(128);
    let (data_tx, mut data_rx) = mpsc::channel::<Frame>(128);
    let writer_cancel = cancel.clone();
    let writer = tokio::spawn(async move {
        // Priority controls, and a shared rate below API Gateway's route throttle.
        let mut cadence = tokio::time::interval(Duration::from_millis(20));
        loop {
            let frame = tokio::select! {biased;_ = writer_cancel.cancelled()=>break,frame=control_rx.recv()=>frame,frame=data_rx.recv()=>frame};
            let Some(frame) = frame else {
                break;
            };
            tokio::select! {_ = writer_cancel.cancelled()=>break,_ = cadence.tick()=>{}}
            let Ok(message) = frame.text() else {
                break;
            };
            if !send_bounded(&mut sink, message, &writer_cancel).await {
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
        tokio::select! {
            _ = cancel.cancelled()=>break,
            _ = &mut expiry=>break,
            result=tasks.join_next(),if !tasks.is_empty()=>{if let Some(Ok(rid))=result{requests.remove(&rid);}},
            message=source.next()=>{
                let Some(Ok(Message::Text(text)))=message else {break;};
                let frame=match Frame::parse(&text){Ok(frame)=>frame,Err(_)=>break};
                if frame.a=="heartbeat"{last=tokio::time::Instant::now();continue;}
                if frame.sid.as_deref()!=Some(&sid){break;}
                last=tokio::time::Instant::now();
                let Some(rid)=frame.rid.clone() else {break;};
                if frame.a=="open" {
                    if requests.contains_key(&rid){continue;}
                    let Ok(permit)=capacity.clone().try_acquire_owned() else {let _=control_tx.try_send(Frame::new("cancel",&sid,&rid));continue;};
                    let (tx,rx)=mpsc::channel(16);requests.insert(rid.clone(),tx);
                    let sid=sid.clone();let controls=control_tx.clone();let data=data_tx.clone();let stop=cancel.child_token();
                    tasks.spawn(async move {let _permit=permit;let _=request_stream(target,(&sid,&rid),rx,controls.clone(),data,stop,options).await;let _=controls.try_send(Frame::new("cancel",&sid,&rid));rid});
                } else if let Some(tx)=requests.get(&rid) {
                    if tx.try_send(frame).is_err(){requests.remove(&rid);let _=control_tx.try_send(Frame::new("cancel",&sid,&rid));}
                }
            },
            _ = heartbeat.tick()=>{
                if last.elapsed()>Duration::from_secs(90){break;}
                let mut ping=Frame::new("heartbeat","","");ping.sid=None;ping.rid=None;
                if control_tx.try_send(ping).is_err(){break;}
            }
        }
    }
    cancel.cancel();
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    writer.await?;
    Ok(())
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
    let mut tcp = tokio::select! {_ = cancel.cancelled()=>return Ok(()),result=authenticated_target(target)=>result?};
    controls
        .send(Frame::new("accept", sid, rid))
        .await
        .map_err(|_| anyhow::anyhow!("AWS relay closed"))?;
    let mut inbox = Inbox::new();
    let mut seq = 0;
    let mut acknowledged = 0;
    let mut credit = WINDOW;
    let mut pending: BTreeMap<u64, Pending> = BTreeMap::new();
    let mut buffer = vec![0; PAYLOAD];
    let mut read_eof = false;
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut activity = tokio::time::Instant::now();
    let mut gap_since = None;
    let mut header = Vec::new();
    let mut header_done = false;
    loop {
        tokio::select! {
            _ = cancel.cancelled()=>return Ok(()),
            frame=input.recv()=>{
                let Some(frame)=frame else{return Ok(());};
                match frame.a.as_str() {
                    "cancel"=>return Ok(()),
                    "ack"=>{
                        let next=frame.next.context("missing AWS relay acknowledgement")?;
                        let available=frame.credit.context("missing AWS relay credit")?;
                        if next>seq||available>WINDOW{bail!("invalid AWS relay acknowledgement");}
                        if next>=acknowledged{acknowledged=next;pending.retain(|n,_|*n>=next);credit=available;}
                    },
                    "data"|"eof"=>{
                        let n=frame.seq.context("missing AWS relay sequence")?;
                        let payload=if frame.a=="eof"{Payload{bytes:Vec::new(),eof:true}}else{Payload{bytes:STANDARD.decode(frame.d.context("missing AWS relay payload")?).map_err(|_|anyhow::anyhow!("invalid AWS relay base64"))?,eof:false}};
                        let ready=inbox.insert(n,payload)?;
                        for payload in ready {
                            if payload.eof && !header_done { bail!("incomplete AWS relay HTTP header"); }
                            if !payload.eof {
                                let bytes = if header_done { payload.bytes } else {
                                    header.extend_from_slice(&payload.bytes);
                                    if header.len() > 64 * 1024 { bail!("AWS relay HTTP header exceeds limit"); }
                                    let Some(end) = header.windows(4).position(|p| p == b"\r\n\r\n") else { continue; };
                                    validate_request_header(&header[..end + 4])?;
                                    header_done = true; std::mem::take(&mut header)
                                };
                                tokio::select!{_ = cancel.cancelled()=>return Ok(()),result=tokio::time::timeout(GAP,tcp.write_all(&bytes))=>{result.context("AWS relay target stalled")??;}}
                                if !bytes.is_empty(){activity=tokio::time::Instant::now();}
                            }
                            // Request EOF is logical completion, never TCP FIN.
                        }
                        if inbox.held.is_empty(){gap_since=None;}else if gap_since.is_none(){gap_since=Some(tokio::time::Instant::now());}
                        let mut ack=Frame::new("ack",sid,rid);ack.next=Some(inbox.next);ack.credit=Some(WINDOW-inbox.held.len());
                        controls.send(ack).await.map_err(|_|anyhow::anyhow!("AWS relay closed"))?;
                    },
                    _=>bail!("invalid AWS relay request action"),
                }
            },
            result=tcp.read(&mut buffer),if !read_eof && pending.len()<WINDOW && credit>0=>{
                let n=result?;let mut frame=Frame::new(if n==0{"eof"}else{"data"},sid,rid);frame.seq=Some(seq);
                if n==0{read_eof=true;}else{frame.d=Some(STANDARD.encode(&buffer[..n]));activity=tokio::time::Instant::now();}
                output.send(frame.clone()).await.map_err(|_|anyhow::anyhow!("AWS relay closed"))?;
                pending.insert(seq,Pending{frame,sent:tokio::time::Instant::now(),retries:0});seq+=1;credit-=1;
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
    async fn aws_request_reorders_without_replay_and_logical_eof_retains_http_response() {
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
        let app=Router::new().route("/production",get(move|headers:HeaderMap,ws:WebSocketUpgrade|{let done_tx=done_tx.clone();async move{assert_eq!(headers["x-mold-relay-role"],"host");ws.on_upgrade(move|mut socket|async move{
            let Some(Ok(AxMessage::Text(text)))=socket.recv().await else{panic!("missing hello")};assert_eq!(Frame::parse(&text).unwrap().a,"hello");
            let mut ready=Frame::new("ready","epoch","host");ready.role=Some("host".into());
            // Send ready directly as text (the v2 envelope is not v1 Wire).
            socket.send(AxMessage::Text(serde_json::to_string(&ready).unwrap().into())).await.unwrap();
            socket.send(AxMessage::Text(serde_json::to_string(&Frame::new("open","epoch","guest")).unwrap().into())).await.unwrap();
            loop{let Some(Ok(AxMessage::Text(text)))=socket.recv().await else{panic!("missing accept")};let frame=Frame::parse(&text).unwrap();if frame.a=="accept"{break;}}
            let request=b"POST /echo HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\nContent-Length: 7\r\n\r\npayload";
            for(seq,bytes)in[(1,&request[40..]),(0,&request[..40]),(0,&request[..40])]{let mut data=Frame::new("data","epoch","guest");data.seq=Some(seq);data.d=Some(STANDARD.encode(bytes));socket.send(AxMessage::Text(serde_json::to_string(&data).unwrap().into())).await.unwrap();}
            let mut eof=Frame::new("eof","epoch","guest");eof.seq=Some(2);socket.send(AxMessage::Text(serde_json::to_string(&eof).unwrap().into())).await.unwrap();
            let mut response=Vec::new();let mut next=0;
            loop{let Some(Ok(AxMessage::Text(text)))=socket.recv().await else{panic!("response transport closed")};let frame=Frame::parse(&text).unwrap();if frame.a=="data"||frame.a=="eof"{assert_eq!(frame.seq,Some(next));next+=1;if let Some(data)=frame.d{response.extend(STANDARD.decode(data).unwrap());}let mut ack=Frame::new("ack","epoch","guest");ack.next=Some(next);ack.credit=Some(4);socket.send(AxMessage::Text(serde_json::to_string(&ack).unwrap().into())).await.unwrap();if frame.a=="eof"{break;}}}
            assert!(response.ends_with(b"payload"));if let Some(tx)=done_tx.lock().await.take(){let _=tx.send(());}
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
            connect(
                &endpoint,
                target_addr,
                "0123456789012345678901234567890123456789".into(),
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
            validate_request_header(b"GET /api/status HTTP/1.1\r\nHost: local\r\n\r\n").is_ok()
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
