//! Optional storage carrier for opaque Noise ciphertext. The WebSocket reader
//! stays live during bounded object I/O; HTTP application requests never retry.
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use mold_secure::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context as TaskContext, Poll},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_tungstenite::{
    tungstenite::{Error as SocketError, Message},
    MaybeTlsStream, WebSocketStream,
};
use tokio_util::sync::{CancellationToken, PollSender};

type RawSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
const PRODUCTION_ORIGIN: &str =
    "https://mold-relay-042506291754-us-west-2.s3.us-west-2.amazonaws.com";
const FLUSH: Duration = Duration::from_millis(32);
const MIN_OBJECT_BYTES: usize = 65536;
const ALLOCATION_TIMEOUT: Duration = Duration::from_secs(5);
const OBJECT_IO_TIMEOUT: Duration = Duration::from_secs(20);
type Peer = (String, String);

#[derive(Clone, Default)]
pub struct DeliveryTracker(Arc<Mutex<DeliveryState>>);
#[derive(Default)]
struct DeliveryState {
    objects: std::collections::BTreeSet<Peer>,
    sent: BTreeMap<Peer, BTreeMap<u64, ()>>,
}
impl DeliveryTracker {
    pub fn apply(&self, wire: &mut ReliableChannel, now: u64) {
        let peer = (wire.peer().0.to_owned(), wire.peer().1.to_owned());
        let mut state = self.0.lock().unwrap();
        if state.objects.contains(&peer) {
            wire.defer_transmission();
        }
        if let Some(sent) = state.sent.remove(&peer) {
            for seq in sent.into_keys() {
                wire.mark_transmitted(seq, 1, now);
            }
        }
    }
    fn enable(&self, peer: Peer) {
        self.0.lock().unwrap().objects.insert(peer);
    }
    fn remove(&self, peer: &Peer) {
        let mut state = self.0.lock().unwrap();
        state.objects.remove(peer);
        state.sent.remove(peer);
    }
    fn record(&self, value: &Value) {
        if value["a"] != "data" {
            return;
        }
        let (Some(peer), Some(seq)) = (peer(value), value["seq"].as_u64()) else {
            return;
        };
        let mut state = self.0.lock().unwrap();
        if !state.objects.contains(&peer) {
            return;
        }
        let count = value["o"]["count"].as_u64().unwrap_or(1).min(16);
        let sent = state.sent.entry(peer).or_default();
        for n in seq..seq.saturating_add(count) {
            sent.insert(n, ());
        }
        while sent.len() > (WIRE_WINDOW * 2) as usize {
            sent.pop_first();
        }
    }
}
pub struct ObjectSocket {
    send: PollSender<Message>,
    receive: mpsc::Receiver<std::result::Result<Message, SocketError>>,
    stop: CancellationToken,
    delivery: DeliveryTracker,
}
impl ObjectSocket {
    pub fn delivery_tracker(&self) -> DeliveryTracker {
        self.delivery.clone()
    }
    pub fn new(socket: RawSocket, namespace: String, host: bool) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let (out, receive) = mpsc::channel(128);
        let stop = CancellationToken::new();
        let worker_stop = stop.clone();
        let delivery = DeliveryTracker::default();
        let worker_delivery = delivery.clone();
        tokio::spawn(async move {
            let result = run(
                socket,
                namespace,
                host,
                rx,
                out.clone(),
                worker_stop.clone(),
                worker_delivery,
            )
            .await;
            if result.is_err() && !worker_stop.is_cancelled() {
                if let Err(error) = &result {
                    eprintln!(
                        "Mold encrypted object carrier stopped ({})",
                        safe_category(error)
                    );
                }
                let _ = out.try_send(Err(SocketError::Io(std::io::Error::other(
                    "secure ciphertext carrier unavailable",
                ))));
            }
            worker_stop.cancel();
        });
        Self {
            send: PollSender::new(tx),
            receive,
            stop,
            delivery,
        }
    }
}
fn safe_category(error: &anyhow::Error) -> &'static str {
    for source in error.chain() {
        for category in [
            "object allocation refused",
            "object allocation correlation mismatch",
            "invalid object expiry",
            "invalid object write headers",
            "untrusted object address",
            "object fetch refused",
            "object fetch timed out",
            "object fetch unavailable",
            "object body unavailable",
            "object body exceeds bound",
            "object exceeds receive window",
            "object receive bound exceeded",
            "object peer did not negotiate storage",
            "unknown object peer",
            "object session mismatch",
            "conflicting object retry",
            "object carrier receive pressure exceeded",
            "object carrier socket closed",
        ] {
            if source.to_string() == category {
                return category;
            }
        }
        if source.is::<mold_secure::Error>() {
            return "invalid ciphertext object";
        }
    }
    "transport unavailable"
}
impl Drop for ObjectSocket {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}
impl Stream for ObjectSocket {
    type Item = std::result::Result<Message, SocketError>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.receive.poll_recv(cx)
    }
}
impl Sink<Message> for ObjectSocket {
    type Error = SocketError;
    fn poll_ready(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::result::Result<(), Self::Error>> {
        Pin::new(&mut self.send)
            .poll_ready(cx)
            .map_err(|_| SocketError::ConnectionClosed)
    }
    fn start_send(mut self: Pin<&mut Self>, item: Message) -> std::result::Result<(), Self::Error> {
        Pin::new(&mut self.send)
            .start_send(item)
            .map_err(|_| SocketError::ConnectionClosed)
    }
    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::result::Result<(), Self::Error>> {
        Pin::new(&mut self.send)
            .poll_flush(cx)
            .map_err(|_| SocketError::ConnectionClosed)
    }
    fn poll_close(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::result::Result<(), Self::Error>> {
        Pin::new(&mut self.send)
            .poll_close(cx)
            .map_err(|_| SocketError::ConnectionClosed)
    }
}

fn origin() -> String {
    std::env::var("MOLD_SECURE_OBJECT_ORIGIN").unwrap_or_else(|_| PRODUCTION_ORIGIN.into())
}
fn segment(value: &str) -> String {
    let mut result = String::new();
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            result.push(char::from(b));
        } else {
            result.push_str(&format!("%{b:02X}"));
        }
    }
    result
}
fn object_url(
    value: &str,
    origin: &str,
    namespace: &str,
    sid: &str,
    rid: &str,
    direction: &str,
    id: &str,
) -> Result<reqwest::Url> {
    decode_hex::<16>(namespace)?;
    decode_hex::<16>(id)?;
    if value.len() > 4096 {
        bail!("invalid object address");
    }
    let url = reqwest::Url::parse(value).map_err(|_| anyhow::anyhow!("invalid object address"))?;
    let path = format!(
        "/ciphertext/v3/{}/{}/{}/{}/{}",
        namespace,
        segment(sid),
        segment(rid),
        direction,
        id
    );
    if url.scheme() != "https"
        || url.origin().ascii_serialization() != origin
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.path() != path
    {
        bail!("untrusted object address");
    }
    Ok(url)
}
fn supports_objects(value: &Value, trusted: &str) -> bool {
    value["version"] == 1
        && value["origin"] == trusted
        && value["maxBytes"] == 262144
        && value["maxCount"] == 16
        && value["maxInFlightBytes"] == 1048576
        && value["maxInFlightObjects"] == 4
}
fn peer(value: &Value) -> Option<Peer> {
    Some((value["sid"].as_str()?.into(), value["rid"].as_str()?.into()))
}
async fn send(socket: &mut RawSocket, value: Value, delivery: &DeliveryTracker) -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(5),
        socket.send(Message::Text(value.to_string().into())),
    )
    .await
    .map_err(|_| anyhow::anyhow!("object carrier send timed out"))?
    .map_err(|_| anyhow::anyhow!("object carrier socket closed"))?;
    delivery.record(&value);
    Ok(())
}
fn publish(
    out: &mpsc::Sender<std::result::Result<Message, SocketError>>,
    value: Value,
) -> Result<()> {
    out.try_send(Ok(Message::Text(value.to_string().into())))
        .map_err(|_| anyhow::anyhow!("object carrier receive pressure exceeded"))
}
struct Staged {
    frames: Vec<WireFrame>,
    bytes: usize,
    since: Instant,
}
struct Allocation {
    reserved_id: Option<String>,
    frames: Vec<WireFrame>,
    body: Arc<Vec<u8>>,
    request: Value,
    began: Instant,
    retried: Instant,
    uploading: bool,
}
enum Completion {
    Uploaded {
        request_id: String,
        reference: ObjectFrame,
        result: Result<()>,
    },
    Fetched {
        reference: ObjectFrame,
        result: Result<Vec<WireFrame>>,
    },
}
async fn release(
    socket: &mut RawSocket,
    allocation: &Allocation,
    delivery: &DeliveryTracker,
) -> Result<()> {
    if let Some(id) = &allocation.reserved_id {
        send(socket,json!({"a":"object","v":3,"op":"release","requestId":uuid::Uuid::new_v4().simple().to_string(),"sid":allocation.request["sid"],"rid":allocation.request["rid"],"id":id}),delivery).await?;
    }
    Ok(())
}
fn cancel_peer(
    peer: &Peer,
    stops: &mut BTreeMap<Peer, CancellationToken>,
    staged: &mut BTreeMap<Peer, Staged>,
    allocations: &mut BTreeMap<String, Allocation>,
    uploaded: &mut BTreeMap<(Peer, u64), ObjectFrame>,
    downloads: &mut BTreeMap<(Peer, u64), ObjectFrame>,
) {
    if let Some(stop) = stops.remove(peer) {
        stop.cancel();
    }
    staged.remove(peer);
    allocations.retain(|_, a| a.request["sid"] != peer.0 || a.request["rid"] != peer.1);
    uploaded.retain(|(key, _), _| key != peer);
    downloads.retain(|(key, _), _| key != peer);
}

async fn upload(
    http: reqwest::Client,
    url: reqwest::Url,
    body: Arc<Vec<u8>>,
    headers: reqwest::header::HeaderMap,
    stop: CancellationToken,
) -> Result<()> {
    tokio::select! { _=stop.cancelled()=>bail!("object upload cancelled"), result=tokio::time::timeout(OBJECT_IO_TIMEOUT,async {
        let response=http.put(url).headers(headers).body(body.as_ref().clone()).send().await.map_err(|_|anyhow::anyhow!("object upload unavailable"))?;
        if !response.status().is_success() && response.status().as_u16()!=412 { bail!("object upload refused"); }
        Ok(())
    })=>result.map_err(|_|anyhow::anyhow!("object upload timed out"))? }
}
async fn fetch(
    http: reqwest::Client,
    url: reqwest::Url,
    reference: ObjectFrame,
    stop: CancellationToken,
) -> Result<Vec<WireFrame>> {
    tokio::select! { _=stop.cancelled()=>bail!("object fetch cancelled"), result=tokio::time::timeout(OBJECT_IO_TIMEOUT,async {
        let mut response=http.get(url).send().await.map_err(|_|anyhow::anyhow!("object fetch unavailable"))?;
        if !response.status().is_success() || response.content_length().is_some_and(|n|n!=u64::from(reference.o.bytes)) { bail!("object fetch refused"); }
        let mut body=Vec::with_capacity(reference.o.bytes as usize);
        while let Some(chunk)=response.chunk().await.map_err(|_|anyhow::anyhow!("object body unavailable"))? {
            if body.len()+chunk.len()>reference.o.bytes as usize { bail!("object body exceeds bound"); }
            body.extend_from_slice(&chunk);
        }
        Ok(unpack_ciphertext_frames(&reference,&body)?)
    })=>result.map_err(|_|anyhow::anyhow!("object fetch timed out"))? }
}

fn validate_created(
    value: &Value,
    allocation: &Allocation,
    trusted: &str,
    namespace: &str,
    direction: &str,
) -> Result<(ObjectFrame, reqwest::Url, reqwest::header::HeaderMap)> {
    for key in ["sid", "rid", "seq", "count", "bytes", "sha256"] {
        if value[key] != allocation.request[key] {
            bail!("object allocation correlation mismatch");
        }
    }
    let expiry = value["expiresAt"]
        .as_u64()
        .context("invalid object expiry")?;
    let now = crate::now();
    if expiry <= now || expiry > now + 330 {
        bail!("invalid object expiry");
    }
    let reference = ObjectFrame {
        a: "data".into(),
        v: 3,
        sid: value["sid"]
            .as_str()
            .context("missing object session")?
            .into(),
        rid: value["rid"].as_str().context("missing object peer")?.into(),
        seq: value["seq"].as_u64().context("invalid object sequence")?,
        o: CiphertextObject {
            id: value["id"].as_str().context("missing object id")?.into(),
            bytes: value["bytes"].as_u64().context("invalid object size")? as u32,
            sha256: value["sha256"]
                .as_str()
                .context("invalid object checksum")?
                .into(),
            url: value["getUrl"]
                .as_str()
                .context("missing object read address")?
                .into(),
            count: value["count"].as_u64().context("invalid object count")? as u8,
        },
    };
    reference.validate()?;
    object_url(
        &reference.o.url,
        trusted,
        namespace,
        &reference.sid,
        &reference.rid,
        direction,
        &reference.o.id,
    )?;
    let url = object_url(
        value["putUrl"]
            .as_str()
            .context("missing object write address")?,
        trusted,
        namespace,
        &reference.sid,
        &reference.rid,
        direction,
        &reference.o.id,
    )?;
    let expected = json!({"content-type":"application/octet-stream","x-amz-checksum-sha256":STANDARD.encode(decode_hex::<32>(&reference.o.sha256)?),"if-none-match":"*"});
    if value["putHeaders"] != expected {
        bail!("invalid object write headers");
    }
    let mut headers = reqwest::header::HeaderMap::new();
    for (key, value) in expected.as_object().unwrap() {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(key.as_bytes())?,
            value.as_str().unwrap().parse()?,
        );
    }
    Ok((reference, url, headers))
}

async fn publish_batch(
    socket: &mut RawSocket,
    batch: Staged,
    allocations: &mut BTreeMap<String, Allocation>,
    uploaded: &BTreeMap<(Peer, u64), ObjectFrame>,
    abandoned_count: usize,
    delivery: &DeliveryTracker,
) -> Result<()> {
    let p = (batch.frames[0].sid.clone(), batch.frames[0].rid.clone());
    if std::env::var_os("MOLD_SECURE_TRACE_CARRIER").is_some() {
        eprintln!(
            "Mold ciphertext batch frames={} bytes={} eligible={}",
            batch.frames.len(),
            batch.bytes,
            batch.bytes >= MIN_OBJECT_BYTES
        );
    }
    if batch.bytes < MIN_OBJECT_BYTES || batch.frames.len() < 2 {
        for frame in batch.frames {
            send(socket, serde_json::to_value(frame)?, delivery).await?;
        }
        return Ok(());
    }
    let body = Arc::new(pack_ciphertext_frames(&batch.frames)?);
    let live = allocations
        .values()
        .filter(|a| peer(&a.request).as_ref() == Some(&p))
        .count()
        + uploaded
            .values()
            .filter(|o| o.sid == p.0 && o.rid == p.1)
            .count();
    if live >= 4 || abandoned_count >= 128 {
        for frame in batch.frames {
            send(socket, serde_json::to_value(frame)?, delivery).await?;
        }
        return Ok(());
    }
    let request_id = uuid::Uuid::new_v4().simple().to_string();
    let request = json!({"a":"object","v":3,"op":"create","requestId":request_id,"sid":p.0,"rid":p.1,"seq":batch.frames[0].seq,"count":batch.frames.len(),"bytes":body.len(),"sha256":hex::encode(Sha256::digest(body.as_ref()))});
    send(socket, request.clone(), delivery).await?;
    allocations.insert(
        request_id,
        Allocation {
            reserved_id: None,
            frames: batch.frames,
            body,
            request,
            began: Instant::now(),
            retried: Instant::now(),
            uploading: false,
        },
    );
    Ok(())
}

async fn run(
    mut socket: RawSocket,
    namespace: String,
    host: bool,
    mut outgoing: mpsc::Receiver<Message>,
    incoming: mpsc::Sender<std::result::Result<Message, SocketError>>,
    stop: CancellationToken,
    delivery: DeliveryTracker,
) -> Result<()> {
    let _cancel_on_drop = stop.clone().drop_guard();
    let trusted = origin();
    let http = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut enabled = false;
    let mut object_peers = std::collections::BTreeSet::<Peer>::new();
    let mut sid = None::<String>;
    let direction = if host { "host" } else { "device" };
    let receive_direction = if host { "device" } else { "host" };
    let mut staged = BTreeMap::<Peer, Staged>::new();
    let mut allocations = BTreeMap::<String, Allocation>::new();
    let mut abandoned = BTreeMap::<String, (Value, Instant)>::new();
    let mut uploaded = BTreeMap::<(Peer, u64), ObjectFrame>::new();
    let mut resent = BTreeMap::<(Peer, u64), Instant>::new();
    let mut downloads = BTreeMap::<(Peer, u64), ObjectFrame>::new();
    let mut acknowledgements = BTreeMap::<Peer, Value>::new();
    let mut ack_dirty = BTreeMap::<Peer, Value>::new();
    let mut newest = BTreeMap::<Peer, u64>::new();
    let mut peer_stops = BTreeMap::<Peer, CancellationToken>::new();
    let (completed, mut completion) = mpsc::channel(128);
    let mut tick = tokio::time::interval(Duration::from_millis(8));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
                _=stop.cancelled()=>return Ok(()),
                _=tick.tick()=>{
            abandoned.retain(|_, (_, at)| at.elapsed() < Duration::from_secs(300));
            for (_, ack) in std::mem::take(&mut ack_dirty) {
                send(&mut socket, ack, &delivery).await?;
            }
            let flush = staged
                .iter()
                .filter(|(_, s)| s.since.elapsed() >= FLUSH)
                .map(|(p, _)| p.clone())
                .collect::<Vec<_>>();
            for p in flush {
                let batch = staged.remove(&p).unwrap();
                publish_batch(&mut socket, batch, &mut allocations, &uploaded, abandoned.len(), &delivery).await?;
            }
            let expired = allocations
                .iter()
                .filter(|(_, a)| a.began.elapsed() > ALLOCATION_TIMEOUT && !a.uploading)
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>();
            for id in expired {
                let allocation = allocations.remove(&id).unwrap();
                if allocation.reserved_id.is_none() {
                    abandoned.insert(id, (allocation.request.clone(), Instant::now()));
                }
                release(&mut socket, &allocation, &delivery).await?;
                for frame in allocation.frames {
                    send(&mut socket, serde_json::to_value(frame)?, &delivery).await?;
                }
            }
            let retries = allocations
                .values_mut()
                .filter(|a| !a.uploading && a.retried.elapsed() > Duration::from_millis(500))
                .map(|a| {
                    a.retried = Instant::now();
                    a.request.clone()
                })
                .collect::<Vec<_>>();
            for request in retries {
                send(&mut socket, request, &delivery).await?;
            }
        },
                done=completion.recv()=>{
            match done.context("object worker stopped")? {
                Completion::Uploaded {
                    request_id,
                    reference,
                    result,
                } => {
                    if let Some(allocation) = allocations.remove(&request_id) {
                        if result.is_err() {
                            release(&mut socket, &allocation, &delivery).await?;
                            for frame in allocation.frames {
                                send(&mut socket, serde_json::to_value(frame)?, &delivery).await?;
                            }
                            continue;
                        }
                        send(&mut socket, serde_json::to_value(&reference)?, &delivery).await?;
                        resent.insert(
                            (
                                (reference.sid.clone(), reference.rid.clone()),
                                reference.seq,
                            ),
                            Instant::now(),
                        );
                        uploaded.insert(
                            (
                                (reference.sid.clone(), reference.rid.clone()),
                                reference.seq,
                            ),
                            reference,
                        );
                    }
                }
                Completion::Fetched { reference, result } => {
                    if downloads
                        .remove(&(
                            (reference.sid.clone(), reference.rid.clone()),
                            reference.seq,
                        ))
                        .is_none()
                    {
                        continue;
                    }
                    let p = (reference.sid.clone(), reference.rid.clone());
                    if let Some(ack) = acknowledgements.get(&p).filter(|ack| {
                        ack["next"]
                            .as_u64()
                            .is_some_and(|next| next >= reference.seq + u64::from(reference.o.count))
                    }) {
                        send(&mut socket, ack.clone(), &delivery).await?;
                        continue;
                    }
                    match result {
                        Ok(frames) => {
                            for frame in frames {
                                publish(&incoming, serde_json::to_value(frame)?)?;
                            }
                        }
                        Err(error) => {
                            let p = (reference.sid.clone(), reference.rid.clone());
                            eprintln!(
                                "Mold encrypted object peer stopped ({})",
                                safe_category(&error)
                            );
                            cancel_peer(
                                &p,
                                &mut peer_stops,
                                &mut staged,
                                &mut allocations,
                                &mut uploaded,
                                &mut downloads,
                            );
                            object_peers.remove(&p);
                            delivery.remove(&p);
                            acknowledgements.remove(&p);
                            ack_dirty.remove(&p);
                            newest.remove(&p);
                            resent.retain(|(key, _), _| key != &p);
                            let cancel = json!({"a":"cancel","v":3,"sid":p.0,"rid":p.1});
                            send(&mut socket, cancel.clone(), &delivery).await?;
                            publish(&incoming, cancel)?;
                        }
                    }
                }
            }
        },
                message=outgoing.recv()=>{
            let Some(message) = message else {
                return Ok(());
            };
            let Message::Text(text) = &message else {
                socket.send(message).await?;
                continue;
            };
            let mut value: Value = serde_json::from_str(text)?;
            if value["a"] == "hello" && value["role"] != "invite" {
                value["objects"] = 1.into();
                send(&mut socket, value, &delivery).await?;
                continue;
            }
            let p = peer(&value);
            if value["a"] == "cancel" {
                if let Some(p) = p.clone() {
                    cancel_peer(
                        &p,
                        &mut peer_stops,
                        &mut staged,
                        &mut allocations,
                        &mut uploaded,
                        &mut downloads,
                    );
                    object_peers.remove(&p);
                    delivery.remove(&p);
                    acknowledgements.remove(&p);
                    ack_dirty.remove(&p);
                    newest.remove(&p);
                    resent.retain(|(key, _), _| key != &p);
                }
            }
            if value["a"] == "ack" {
                if let Some(p) = p.clone() {
                    acknowledgements.insert(p.clone(), value.clone());
                    if enabled {
                        ack_dirty.insert(p, value);
                        continue;
                    }
                }
            }
            if enabled
                && p.as_ref().is_some_and(|p| object_peers.contains(p))
                && value["a"] == "data"
                && value["d"].is_string()
            {
                let frame: WireFrame = serde_json::from_value(value.clone())?;
                let p = p.context("missing ciphertext peer")?;
                let seq = frame.seq.context("missing ciphertext sequence")?;
                if let Some((key, reference)) = uploaded.iter().find(|((peer, start), o)| {
                    peer == &p && seq >= *start && seq < *start + u64::from(o.o.count)
                }) {
                    if resent
                        .get(key)
                        .is_none_or(|t| t.elapsed() >= Duration::from_millis(100))
                    {
                        send(&mut socket, serde_json::to_value(reference)?, &delivery).await?;
                        resent.insert(key.clone(), Instant::now());
                    }
                    continue;
                }
                if allocations.values().any(|a| {
                    peer(&a.request).as_ref() == Some(&p)
                        && seq >= a.request["seq"].as_u64().unwrap_or(u64::MAX)
                        && seq < a.request["seq"].as_u64().unwrap_or(0) + a.frames.len() as u64
                }) {
                    continue;
                }
                let new = seq >= *newest.get(&p).unwrap_or(&0);
                newest
                    .entry(p.clone())
                    .and_modify(|v| *v = (*v).max(seq + 1))
                    .or_insert(seq + 1);
                let size = frame.d.as_ref().map(|d| d.len() / 4 * 3 + 4).unwrap_or(0);
                if new && (size >= 4096 || staged.contains_key(&p)) {
                    if staged.get(&p).is_some_and(|s| {
                        s.frames.last().and_then(|f| f.seq) != seq.checked_sub(1)
                            || s.frames.len() >= 16
                            || s.bytes + size > MAX_OBJECT_BYTES
                    }) {
                        let batch = staged.remove(&p).unwrap();
                        publish_batch(&mut socket, batch, &mut allocations, &uploaded, abandoned.len(), &delivery).await?;
                    }
                    let batch = staged.entry(p).or_insert_with(|| Staged {
                        frames: vec![],
                        bytes: 0,
                        since: Instant::now(),
                    });
                    batch.bytes += size;
                    batch.frames.push(frame);
                    continue;
                }
            }
            send(&mut socket, value, &delivery).await?;
        },
                message=socket.next()=>{
            let Some(message) = message else {
                return Ok(());
            };
            let message = message?;
            let Message::Text(text) = &message else {
                incoming
                    .try_send(Ok(message))
                    .map_err(|_| anyhow::anyhow!("object receive pressure"))?;
                continue;
            };
            let value: Value = serde_json::from_str(text)?;
            if value["a"] == "ready" {
                enabled = supports_objects(&value["objects"], &trusted);
                sid = value["sid"].as_str().map(str::to_string);
                if let Some(p) = peer(&value) {
                    if enabled {
                        object_peers.insert(p.clone());
                        delivery.enable(p.clone());
                    }
                    peer_stops.entry(p).or_insert_with(|| stop.child_token());
                }
            }
            if value["a"] == "open" {
                if let Some(p) = peer(&value) {
                    if enabled && supports_objects(&value["objects"], &trusted) {
                        object_peers.insert(p.clone());
                        delivery.enable(p.clone());
                    }
                    if peer_stops.len() >= 32 && !peer_stops.contains_key(&p) {
                        bail!("object peer bound exceeded");
                    }
                    peer_stops.entry(p).or_insert_with(|| stop.child_token());
                }
            }
            if value["a"] == "cancel" {
                if let Some(p) = peer(&value) {
                    cancel_peer(
                        &p,
                        &mut peer_stops,
                        &mut staged,
                        &mut allocations,
                        &mut uploaded,
                        &mut downloads,
                    );
                    object_peers.remove(&p);
                    delivery.remove(&p);
                    acknowledgements.remove(&p);
                    ack_dirty.remove(&p);
                    newest.remove(&p);
                    resent.retain(|(key, _), _| key != &p);
                }
            }
            if value["a"] == "object" {
                let request_id = value["requestId"]
                    .as_str()
                    .context("missing object correlation")?
                    .to_string();
                let Some(allocation) = allocations.get_mut(&request_id) else {
                    if value["op"] == "created" {
                        if let Some((request, _)) = abandoned.remove(&request_id) {
                            if ["sid", "rid", "seq", "count", "bytes", "sha256"]
                                .iter()
                                .all(|key| value[*key] == request[*key])
                                && value["id"]
                                    .as_str()
                                    .is_some_and(|id| decode_hex::<16>(id).is_ok())
                            {
                                send(&mut socket,json!({"a":"object","v":3,"op":"release","requestId":uuid::Uuid::new_v4().simple().to_string(),"sid":request["sid"],"rid":request["rid"],"id":value["id"]}),&delivery).await?;
                            }
                        }
                    }
                    continue;
                };
                if value["op"] == "error" {
                    let correlated = value["sid"] == allocation.request["sid"]
                        && value["rid"] == allocation.request["rid"];
                    let code = if correlated {
                        value["code"].as_str().unwrap_or("")
                    } else {
                        "conflict"
                    };
                    if code == "disabled" {
                        let allocation = allocations.remove(&request_id).unwrap();
                        for frame in allocation.frames {
                            send(&mut socket, serde_json::to_value(frame)?, &delivery).await?;
                        }
                        continue;
                    }
                    if ["capacity", "not_ready", "unavailable"].contains(&code)
                        && allocation.retried.elapsed() >= Duration::from_millis(100)
                    {
                        allocation.retried = Instant::now();
                        send(&mut socket, allocation.request.clone(), &delivery).await?;
                        continue;
                    }
                    if ["capacity", "not_ready", "unavailable"].contains(&code) {
                        continue;
                    }
                    let p = peer(&allocation.request).context("missing allocation peer")?;
                    eprintln!("Mold encrypted object peer stopped (object allocation refused)");
                    cancel_peer(
                        &p,
                        &mut peer_stops,
                        &mut staged,
                        &mut allocations,
                        &mut uploaded,
                        &mut downloads,
                    );
                    object_peers.remove(&p);
                    delivery.remove(&p);
                    acknowledgements.remove(&p);
                    ack_dirty.remove(&p);
                    newest.remove(&p);
                    resent.retain(|(key, _), _| key != &p);
                    let cancel = json!({"a":"cancel","v":3,"sid":p.0,"rid":p.1});
                    send(&mut socket, cancel.clone(), &delivery).await?;
                    publish(&incoming, cancel)?;
                    continue;
                }
                if value["op"] != "created" || allocation.uploading {
                    continue;
                }
                let (reference, url, headers) =
                    match validate_created(&value, allocation, &trusted, &namespace, direction) {
                        Ok(created) => created,
                        Err(_) => {
                            let p = peer(&allocation.request).context("missing allocation peer")?;
                            eprintln!("Mold encrypted object peer stopped (object allocation refused)");
                            cancel_peer(
                                &p,
                                &mut peer_stops,
                                &mut staged,
                                &mut allocations,
                                &mut uploaded,
                                &mut downloads,
                            );
                            object_peers.remove(&p);
                            delivery.remove(&p);
                            acknowledgements.remove(&p);
                            ack_dirty.remove(&p);
                            newest.remove(&p);
                            resent.retain(|(key, _), _| key != &p);
                            let cancel = json!({"a":"cancel","v":3,"sid":p.0,"rid":p.1});
                            send(&mut socket, cancel.clone(), &delivery).await?;
                            publish(&incoming, cancel)?;
                            continue;
                        }
                    };
                if std::env::var_os("MOLD_SECURE_TRACE_CARRIER").is_some() {
                    eprintln!("Mold ciphertext broker elapsed_ms={} bytes={}", allocation.began.elapsed().as_millis(), allocation.body.len());
                }
                allocation.reserved_id = Some(reference.o.id.clone());
                allocation.uploading = true;
                let body = allocation.body.clone();
                let done = completed.clone();
                let http = http.clone();
                let stop = peer_stops
                    .get(&(reference.sid.clone(), reference.rid.clone()))
                    .context("unknown object peer")?
                    .child_token();
                tokio::spawn(async move {
                    let began = Instant::now();
                    let result = upload(http, url, body, headers, stop).await;
                    if std::env::var_os("MOLD_SECURE_TRACE_CARRIER").is_some() { eprintln!("Mold ciphertext PUT elapsed_ms={} success={}", began.elapsed().as_millis(), result.is_ok()); }
                    let _ = done
                        .send(Completion::Uploaded {
                            request_id,
                            reference,
                            result,
                        })
                        .await;
                });
                continue;
            }
            if value["a"] == "ack" {
                if let Some(p) = peer(&value) {
                    let next = value["next"]
                        .as_u64()
                        .context("invalid object acknowledgement")?;
                    uploaded.retain(|(key, _), o| key != &p || o.seq + u64::from(o.o.count) > next);
                    resent.retain(|key, _| uploaded.contains_key(key));
                }
            }
            if value["a"] == "data" && value["o"].is_object() {
                let context_peer = peer(&value).filter(|p| peer_stops.contains_key(p));
                let checked: Result<Option<(ObjectFrame, reqwest::Url)>> = (|| {
                    if !enabled {
                        bail!("untrusted object transport");
                    }
                    let reference: ObjectFrame = serde_json::from_value(value.clone())?;
                    reference.validate()?;
                    if sid.as_deref() != Some(&reference.sid) {
                        bail!("object session mismatch");
                    }
                    let p = (reference.sid.clone(), reference.rid.clone());
                    if !object_peers.contains(&p) {
                        bail!("object peer did not negotiate storage");
                    }
                    let next = acknowledgements
                        .get(&p)
                        .and_then(|a| a["next"].as_u64())
                        .unwrap_or(0);
                    if reference.seq + u64::from(reference.o.count) <= next {
                        return Ok(None);
                    }
                    if reference.seq + u64::from(reference.o.count) > next + WIRE_WINDOW {
                        bail!("object exceeds receive window");
                    }
                    if let Some(old) = downloads.get(&(p.clone(), reference.seq)) {
                        if old != &reference {
                            bail!("conflicting object retry");
                        }
                        return Ok(None);
                    }
                    let pending = downloads
                        .values()
                        .filter(|o| o.sid == p.0 && o.rid == p.1)
                        .collect::<Vec<_>>();
                    if pending.len() >= 4
                        || pending.iter().map(|o| o.o.bytes as usize).sum::<usize>()
                            + reference.o.bytes as usize
                            > 1048576
                    {
                        bail!("object receive bound exceeded");
                    }
                    let url = object_url(
                        &reference.o.url,
                        &trusted,
                        &namespace,
                        &reference.sid,
                        &reference.rid,
                        receive_direction,
                        &reference.o.id,
                    )?;
                    Ok(Some((reference, url)))
                })();
                match checked {
                    Ok(Some((reference, url))) => {
                        let p = (reference.sid.clone(), reference.rid.clone());
                        downloads.insert((p.clone(), reference.seq), reference.clone());
                        let done = completed.clone();
                        let http = http.clone();
                        let stop = peer_stops
                            .get(&p)
                            .context("unknown object peer")?
                            .child_token();
                        tokio::spawn(async move {
                            let began = Instant::now();
                            let result = fetch(http, url, reference.clone(), stop).await;
                            if std::env::var_os("MOLD_SECURE_TRACE_CARRIER").is_some() { eprintln!("Mold ciphertext GET elapsed_ms={} bytes={} success={}", began.elapsed().as_millis(), reference.o.bytes, result.is_ok()); }
                            let _ = done.send(Completion::Fetched { reference, result }).await;
                        });
                    }
                    Ok(None) => {
                        if let Some(ack) = context_peer.as_ref().and_then(|p| acknowledgements.get(p)) {
                            send(&mut socket, ack.clone(), &delivery).await?;
                        }
                    }
                    Err(error) => {
                        let Some(p) = context_peer else {
                            return Err(error);
                        };
                        eprintln!(
                            "Mold encrypted object peer stopped ({})",
                            safe_category(&error)
                        );
                        cancel_peer(
                            &p,
                            &mut peer_stops,
                            &mut staged,
                            &mut allocations,
                            &mut uploaded,
                            &mut downloads,
                        );
                        object_peers.remove(&p);
                        delivery.remove(&p);
                        acknowledgements.remove(&p);
                        ack_dirty.remove(&p);
                        newest.remove(&p);
                        resent.retain(|(key, _), _| key != &p);
                        let cancel = json!({"a":"cancel","v":3,"sid":p.0,"rid":p.1});
                        send(&mut socket, cancel.clone(), &delivery).await?;
                        publish(&incoming, cancel)?;
                    }
                }
                continue;
            }
            publish(&incoming, value)?;
        }
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn object_address_is_exactly_bound_to_trusted_origin_peer_and_direction() {
        let ns = "11".repeat(16);
        let id = "22".repeat(16);
        let origin = "https://trusted.invalid";
        let path = format!("{origin}/ciphertext/v3/{ns}/host%3D/peer%3D/host/{id}?signed=opaque");
        assert!(object_url(&path, origin, &ns, "host=", "peer=", "host", &id).is_ok());
        for bad in [
            path.replace("trusted.invalid", "attacker.invalid"),
            path.replace("/host/", "/device/"),
            format!("{path}#fragment"),
            path.replace("peer%3D", "other"),
        ] {
            assert!(object_url(&bad, origin, &ns, "host=", "peer=", "host", &id).is_err());
        }
    }
    #[tokio::test]
    async fn broker_disabled_returns_the_identical_ciphertext_without_waiting_for_new_encryption() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            socket.send(Message::Text(json!({"a":"ready","v":3,"sid":"s","rid":"r","objects":{"version":1,"origin":PRODUCTION_ORIGIN,"maxBytes":262144,"maxCount":16,"maxInFlightBytes":1048576,"maxInFlightObjects":4}}).to_string().into())).await.unwrap();
            let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                panic!()
            };
            let request: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(request["op"], "create");
            assert!(request["count"].as_u64().unwrap() >= 2);
            socket.send(Message::Text(json!({"a":"object","v":3,"op":"error","requestId":request["requestId"],"sid":"s","rid":"r","code":"disabled"}).to_string().into())).await.unwrap();
            let mut result = vec![];
            while result.len() < 16 {
                let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                    panic!()
                };
                result.push(serde_json::from_str::<WireFrame>(&text).unwrap());
            }
            result
        });
        let (raw, _) = tokio_tungstenite::connect_async(format!("ws://{address}"))
            .await
            .unwrap();
        let mut socket = ObjectSocket::new(raw, "11".repeat(16), false);
        socket.next().await.unwrap().unwrap();
        let mut wire = ReliableChannel::new("s".into(), "r".into());
        let mut frames = vec![];
        for n in 0..16 {
            // HTTP reads contain small tail records; they must not split a
            // consecutive ciphertext batch into expensive inline fragments.
            wire.queue_packet(&vec![42; if n % 2 == 0 { 16320 } else { 256 }])
                .unwrap();
            frames.extend(wire.outgoing(0).unwrap());
        }
        for frame in &frames {
            socket
                .send(Message::Text(serde_json::to_string(frame).unwrap().into()))
                .await
                .unwrap();
        }
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), server)
                .await
                .unwrap()
                .unwrap(),
            frames
        );
    }
    #[tokio::test]
    async fn invalid_object_only_closes_its_peer_on_a_shared_host_socket() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (raw, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(raw).await.unwrap();
            let policy = json!({"version":1,"origin":PRODUCTION_ORIGIN,"maxBytes":262144,"maxCount":16,"maxInFlightBytes":1048576,"maxInFlightObjects":4});
            for frame in [
                json!({"a":"ready","v":3,"sid":"s","role":"host","objects":policy}),
                json!({"a":"open","v":3,"sid":"s","rid":"r1","objects":policy}),
                json!({"a":"open","v":3,"sid":"s","rid":"r2","objects":policy}),
            ] {
                socket
                    .send(Message::Text(frame.to_string().into()))
                    .await
                    .unwrap();
            }
            let invalid = json!({"a":"data","v":3,"sid":"s","rid":"r1","seq":1000,"o":{"id":"11".repeat(16),"bytes":5,"sha256":"22".repeat(32),"url":"https://trusted.invalid/opaque","count":1}});
            socket
                .send(Message::Text(invalid.to_string().into()))
                .await
                .unwrap();
            let Message::Text(cancel) = socket.next().await.unwrap().unwrap() else {
                panic!()
            };
            let cancel: Value = serde_json::from_str(&cancel).unwrap();
            assert_eq!(cancel["a"], "cancel");
            assert_eq!(cancel["rid"], "r1");
            socket
                .send(Message::Text(
                    json!({"a":"ack","v":3,"sid":"s","rid":"r2","next":0,"credit":16,"ack_seq":0})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let (raw, _) = tokio_tungstenite::connect_async(format!("ws://{address}"))
            .await
            .unwrap();
        let mut socket = ObjectSocket::new(raw, "11".repeat(16), true);
        for _ in 0..4 {
            socket.next().await.unwrap().unwrap();
        }
        let Message::Text(ack) = socket.next().await.unwrap().unwrap() else {
            panic!()
        };
        let ack: Value = serde_json::from_str(&ack).unwrap();
        assert_eq!(ack["rid"], "r2");
        assert_eq!(ack["a"], "ack");
        server.await.unwrap();
    }
}
