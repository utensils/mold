use axum::{
    body::{to_bytes, Body},
    http::{Request, Response, StatusCode},
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use mold_relay::secure::*;
use mold_secure::*;
use std::{sync::Arc, time::Duration};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

struct Client {
    crypto: Session,
    wire: ReliableChannel,
    input: mpsc::Sender<WireFrame>,
    output: mpsc::Receiver<WireFrame>,
    records: std::collections::VecDeque<Record>,
    start: std::time::Instant,
}
impl Client {
    async fn pump(&mut self) -> anyhow::Result<()> {
        let now = self.start.elapsed().as_millis() as u64;
        for f in self.wire.outgoing(now)? {
            self.input.send(f).await?;
        }
        if let Ok(Some(frame)) =
            tokio::time::timeout(Duration::from_millis(10), self.output.recv()).await
        {
            for packet in self.wire.receive(frame, now)? {
                if !self.crypto.established() {
                    self.crypto.handshake_read(&packet)?;
                    self.wire.queue_packet(&self.crypto.handshake_write()?)?;
                } else {
                    self.records
                        .push_back(Record::decode(&self.crypto.open(&packet)?)?);
                }
            }
        }
        Ok(())
    }
    async fn send(&mut self, record: Record) {
        while !self.wire.can_queue() {
            self.pump().await.unwrap();
        }
        self.wire
            .queue_packet(&self.crypto.seal(&record.encode().unwrap()).unwrap())
            .unwrap();
        self.pump().await.unwrap();
    }
    async fn receive(&mut self) -> Record {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(r) = self.records.pop_front() {
                    return r;
                }
                self.pump().await.unwrap();
            }
        })
        .await
        .unwrap()
    }
}
async fn fixture() -> (
    Client,
    watch::Sender<HostPolicy>,
    CancellationToken,
    tokio::task::JoinHandle<anyhow::Result<()>>,
    HostConfig,
    mold_secure_client::PairedHost,
) {
    fixture_with_capacity(32).await
}
async fn fixture_with_capacity(
    capacity: usize,
) -> (
    Client,
    watch::Sender<HostPolicy>,
    CancellationToken,
    tokio::task::JoinHandle<anyhow::Result<()>>,
    HostConfig,
    mold_secure_client::PairedHost,
) {
    let host = Arc::new(Identity::from_bytes(&[1; 64]).unwrap());
    let device = Identity::from_bytes(&[2; 64]).unwrap();
    let host_for_api = host.clone();
    async fn echo(req: Request<Body>, host: Arc<Identity>) -> Response<Body> {
        assert_eq!(req.headers()["x-api-key"], "private-fixture-key");
        let context: RequestContext = serde_json::from_slice(
            &URL_SAFE_NO_PAD
                .decode(req.headers()["x-mold-secure-peer"].to_str().unwrap())
                .unwrap(),
        )
        .unwrap();
        context
            .verify(
                &host.signing_public(),
                req.method().as_str(),
                req.uri().to_string().as_str(),
                unix_now(),
            )
            .unwrap();
        let bytes = to_bytes(req.into_body(), MAX_UPLOAD as usize)
            .await
            .unwrap();
        Response::builder()
            .status(206)
            .header("content-type", "application/octet-stream")
            .header(
                "content-range",
                format!("bytes 0-{}/{}", bytes.len() - 1, bytes.len()),
            )
            .body(Body::from(bytes))
            .unwrap()
    }
    let router = Router::new()
        .route(
            "/api/status",
            get(|| async { (StatusCode::UNAUTHORIZED, "auth required") }),
        )
        .route("/echo", post(move |req| echo(req, host_for_api)))
        .route(
            "/api/gallery/image/fixture.mp4",
            get(|req: Request<Body>| async move {
                let ranged = req.headers().get("range").is_some();
                Response::builder()
                    .status(if ranged { 206 } else { 200 })
                    .header("content-type", "video/mp4")
                    .header(
                        "content-range",
                        if ranged { "bytes 2-4/8" } else { "bytes 0-7/8" },
                    )
                    .header("content-length", if ranged { "3" } else { "8" })
                    .body(Body::from(if ranged {
                        vec![2, 3, 4]
                    } else {
                        vec![0, 1, 2, 3, 4, 5, 6, 7]
                    }))
                    .unwrap()
            }),
        )
        .route(
            "/sse",
            get(|| async {
                let stream = futures_util::stream::unfold(0, |n| async move {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Some((Ok::<_, std::io::Error>(format!("data: {n}\n\n")), n + 1))
                });
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = listener.local_addr().unwrap();
    let stop = CancellationToken::new();
    let stop_server = stop.clone();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(stop_server.cancelled_owned())
            .await
            .unwrap();
    });
    let ns = namespace(&host.signing_public());
    let mut grant = DeviceGrant {
        id: "11".repeat(16),
        expires_at: unix_now() + 30 * 86400,
        epoch: 0,
        signature: String::new(),
    };
    grant.signature = hex::encode(
        host.sign(
            &grant_bytes(
                &ns,
                &device.signing_public(),
                &device.noise_public(),
                &grant.id,
                grant.expires_at,
                0,
            )
            .unwrap(),
        ),
    );
    let (policy_tx, policy) = watch::channel(HostPolicy {
        epoch: 0,
        active: vec![grant.id.clone()],
    });
    let config = HostConfig {
        identity: host.clone(),
        target,
        policy,
    };
    let stored = mold_secure_client::PairedHost {
        endpoints: vec![],
        protocol: 3,
        origin: "http://localhost".into(),
        websocket: "ws://localhost/_mold/secure".into(),
        namespace: ns.clone(),
        host_key: hex::encode(host.signing_public()),
        host_noise_key: hex::encode(host.noise_public()),
        host_binding_signature: host.public_binding().signature,
        device_identity: hex::encode(device.to_bytes()),
        api_key: "private-fixture-key".into(),
        grant: grant.clone(),
        name: None,
        instance_id: None,
        hostname: None,
    };
    let admission = PeerAdmission {
        sid: "test-host".into(),
        rid: "test-client".into(),
        role: "device".into(),
        binding: device.public_binding(),
        grant: Some(grant),
        invite: None,
    };
    let mut crypto = Session::new(
        &device,
        SessionConfig {
            initiator: true,
            namespace: ns,
            host_key: hex::encode(host.signing_public()),
            device_key: hex::encode(device.signing_public()),
            host_noise_key: Some(hex::encode(host.noise_public())),
            device_noise_key: None,
            sid: admission.sid.clone(),
            rid: admission.rid.clone(),
        },
    )
    .unwrap();
    let mut wire = ReliableChannel::new(admission.sid.clone(), admission.rid.clone());
    wire.queue_packet(&crypto.handshake_write().unwrap())
        .unwrap();
    let (input, rx) = mpsc::channel(capacity);
    let (tx, output) = mpsc::channel(capacity);
    let task = tokio::spawn(serve_peer(config.clone(), admission, rx, tx, stop.clone()));
    let mut client = Client {
        crypto,
        wire,
        input,
        output,
        records: Default::default(),
        start: std::time::Instant::now(),
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        while !client.crypto.established() {
            client.pump().await.unwrap();
        }
        client.pump().await.unwrap();
    })
    .await
    .unwrap();
    (client, policy_tx, stop, task, config, stored)
}
#[tokio::test]
async fn encrypted_http_streams_uploads_and_range_response_with_credit() {
    let (mut client, _policy, stop, task, _, _) = fixture().await;
    let data = (0..196608).map(|v| (v % 251) as u8).collect::<Vec<_>>();
    client
        .send(Record::Open {
            stream_id: 1,
            request: Open {
                method: "POST".into(),
                path: "/echo".into(),
                headers: vec![
                    ("x-api-key".into(), "private-fixture-key".into()),
                    ("x-mold-secure-peer".into(), "attacker supplied".into()),
                ],
                length: None,
            },
        })
        .await;
    let mut credit = STREAM_WINDOW;
    let mut bytes = vec![];
    let mut heads = 0;
    let mut ended = false;
    for chunk in data.chunks(DATA_CHUNK) {
        while credit < chunk.len() as u32 {
            match client.receive().await {
                Record::Credit { bytes: n, .. } => credit += n,
                other => panic!("unexpected {other:?}"),
            }
        }
        credit -= chunk.len() as u32;
        client
            .send(Record::Data {
                stream_id: 1,
                bytes: chunk.to_vec(),
            })
            .await;
    }
    client.send(Record::End { stream_id: 1 }).await;
    while !ended {
        match client.receive().await {
            Record::Headers { response, .. } => {
                assert_eq!(response.status, 206);
                assert!(response.headers.iter().any(|(k, _)| k == "content-range"));
                heads += 1;
            }
            Record::Data { bytes: part, .. } => {
                client
                    .send(Record::Credit {
                        stream_id: 1,
                        bytes: part.len() as u32,
                    })
                    .await;
                bytes.extend(part);
            }
            Record::End { .. } => ended = true,
            Record::Credit { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(heads, 1);
    assert_eq!(bytes, data);
    stop.cancel();
    assert!(task.await.unwrap().is_ok());
}
#[tokio::test]
async fn revoked_device_closes_established_session_without_another_request() {
    let (_client, policy, stop, task, _, _) = fixture().await;
    policy
        .send(HostPolicy {
            epoch: 0,
            active: vec![],
        })
        .unwrap();
    let _result = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    // Either the policy error or its independent cancellation watcher may win.
    // Completing this task proves the entire established session was closed.
    stop.cancel();
}

#[tokio::test]
async fn native_client_uses_lan_protocol_and_remote_access_stop_closes_upgraded_socket() {
    let (old_client, _policy, stop, old_task, config, mut host) = fixture().await;
    drop(old_client);
    let _ = old_task.await;
    let lan_stop = CancellationToken::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = lan_router(config, lan_stop.clone());
    let server_stop = lan_stop.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(server_stop.cancelled_owned())
            .await
            .unwrap();
    });
    host.origin = format!("http://{address}");
    host.websocket = format!("ws://{address}/_mold/secure");
    let client = mold_secure_client::SecureClient::new(host.clone()).unwrap();
    let results = futures_util::future::join_all((0..8).map(|n| {
        let client = client.clone();
        let host = host.clone();
        async move {
            let body = vec![n; 196608];
            let request = reqwest::Client::new()
                .post(format!("{}/echo", host.origin))
                .body(body.clone())
                .build()
                .unwrap();
            let response = client.send(request).await.unwrap();
            assert_eq!(response.status(), 206);
            assert_eq!(response.bytes().await.unwrap(), body);
        }
    }))
    .await;
    assert_eq!(results.len(), 8);
    let bridge = mold_secure_client::MediaBridge::start(
        client.clone(),
        host.origin.clone(),
        "tauri://localhost".into(),
        vec!["/api/gallery/image/fixture.mp4".into()],
    )
    .await
    .unwrap();
    let media = bridge.url("/api/gallery/image/fixture.mp4").unwrap();
    let plain = reqwest::Client::new();
    let range = plain
        .get(media)
        .header("Range", "bytes=2-4")
        .header("Origin", "tauri://localhost")
        .send()
        .await
        .unwrap();
    assert_eq!(range.status(), 206);
    assert_eq!(range.bytes().await.unwrap(), vec![2, 3, 4]);
    let head = plain.head(media).send().await.unwrap();
    assert_eq!(head.status(), 200);
    assert_eq!(head.headers()["content-length"], "8");
    assert!(head.bytes().await.unwrap().is_empty());
    assert_eq!(
        plain
            .get(media)
            .header("Origin", "https://attacker.invalid")
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(plain.post(media).send().await.unwrap().status(), 405);
    let bad = media.replace("/0", "/100");
    assert_eq!(plain.get(bad).send().await.unwrap().status(), 404);
    let response = client
        .send(plain.get(format!("{}/sse", host.origin)).build().unwrap())
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    assert!(stream.next().await.unwrap().is_ok());
    lan_stop.cancel();
    assert!(tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    bridge.stop();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let request = reqwest::Client::new()
        .post(format!("{}/echo", host.origin))
        .body(vec![1])
        .build()
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), client.send(request))
            .await
            .unwrap()
            .is_err()
    );
    stop.cancel();
}

#[tokio::test]
async fn forgotten_native_pairing_closes_live_stream_and_cannot_reuse_cached_authority() {
    use futures_util::StreamExt;
    use mold_secure_client::RequestBuilderExt;
    let (_initial, _policy, stop, _task, config, mut host) = fixture().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let lan_stop = CancellationToken::new();
    let run_stop = lan_stop.clone();
    tokio::spawn(async move {
        axum::serve(listener, mold_relay::secure::lan_router(config, run_stop))
            .await
            .unwrap();
    });
    host.origin = format!("https://{}.invalid", uuid::Uuid::new_v4());
    host.websocket = format!("ws://{address}/_mold/secure");
    let owner = uuid::Uuid::new_v4().to_string();
    mold_secure_client::register_ephemeral(&owner, host.clone()).unwrap();
    let http = reqwest::Client::new();
    let response = http
        .get(format!("{}/sse", host.origin))
        .secure_send()
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    assert!(stream.next().await.unwrap().is_ok());
    mold_secure_client::forget_ephemeral(&owner, Some(&host.origin));
    assert!(tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    let error = http
        .get(format!("{}/sse", host.origin))
        .secure_send()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("No secure pairing"));
    lan_stop.cancel();
    stop.cancel();
}
#[tokio::test]
async fn revocation_stops_peer_even_when_its_network_writer_is_blocked() {
    let (mut client, policy, stop, mut task, _, _) = fixture_with_capacity(8).await;
    // Deliberately stop receiving. Small SSE events fill the bounded writer
    // channel while the remote consumer is gone or its socket is congested.
    client
        .send(Record::Open {
            stream_id: 1,
            request: Open {
                method: "GET".into(),
                path: "/sse".into(),
                headers: vec![("x-api-key".into(), "private-fixture-key".into())],
                length: Some(0),
            },
        })
        .await;
    client.send(Record::End { stream_id: 1 }).await;
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(!task.is_finished());
    policy
        .send(HostPolicy {
            epoch: 0,
            active: vec![],
        })
        .unwrap();
    tokio::time::timeout(Duration::from_millis(250), &mut task)
        .await
        .expect("revocation was blocked by the network writer")
        .unwrap()
        .ok();
    stop.cancel();
}

#[tokio::test]
async fn unchanged_standalone_policy_does_not_emit_idle_cloud_work() {
    let app = Router::new().route(
        "/api/pairing/secure/check",
        get(|| async { axum::Json(serde_json::json!({"epoch":0,"active":["11".repeat(16)]})) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let stop = CancellationToken::new();
    let mut policy = watch_policy(
        Arc::new(Identity::generate().unwrap()),
        address,
        stop.clone(),
    )
    .await
    .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(1100), policy.changed())
            .await
            .is_err(),
        "unchanged policy must not wake cloud writer"
    );
    stop.cancel();
    task.abort();
}

#[tokio::test]
async fn hung_standalone_policy_fails_closed_within_one_second() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let hang = Arc::new(AtomicBool::new(false));
    let handler = hang.clone();
    let app = Router::new().route(
        "/api/pairing/secure/check",
        get(move || {
            let hang = handler.clone();
            async move {
                if hang.load(Ordering::SeqCst) {
                    std::future::pending::<()>().await;
                }
                axum::Json(serde_json::json!({"epoch":0,"active":["11".repeat(16)]}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let stop = CancellationToken::new();
    let mut policy = mold_relay::secure::watch_policy(
        Arc::new(Identity::generate().unwrap()),
        address,
        stop.clone(),
    )
    .await
    .unwrap();
    assert_eq!(policy.borrow().active.len(), 1);
    hang.store(true, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !policy.borrow().active.is_empty() {
            policy.changed().await.unwrap();
        }
    })
    .await
    .expect("stale policy exceeded one second");
    stop.cancel();
    task.abort();
}
