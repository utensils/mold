use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{header, StatusCode},
    response::Response,
    routing::any,
    Router,
};
use futures_util::StreamExt;
use mold_relay::{authenticated_target, connect, serve};
use std::{net::SocketAddr, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Message},
};
use tokio_util::sync::CancellationToken;
const TOKEN: &str = "fixture-token-012345678901234567890123456789";

struct Fixture {
    data: SocketAddr,
    control: SocketAddr,
    target: SocketAddr,
    cancel: CancellationToken,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
async fn fixture() -> Fixture {
    let target_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = target_listener.local_addr().unwrap();
    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    let app = Router::new().fallback(any(|request: Request| async move {
        if request.headers().get(header::AUTHORIZATION).is_none() {
            return Response::builder()
                .status(401)
                .body(Body::from("unauthorized fixture"))
                .unwrap();
        }
        if request.uri().path() == "/sse" {
            let stream = futures_util::stream::unfold(0, |n| async move {
                if n == 2 {
                    return None;
                }
                if n > 0 {
                    tokio::time::sleep(Duration::from_millis(600)).await;
                }
                Some((
                    Ok::<_, std::io::Error>(Bytes::from(format!("data: {n}\n\n"))),
                    n + 1,
                ))
            });
            return Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap();
        }
        if request.headers().get(header::RANGE).is_some() {
            return Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_RANGE, "bytes 2-5/10")
                .body(Body::from("2345"))
                .unwrap();
        }
        let bytes = axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024)
            .await
            .unwrap();
        Response::new(Body::from(bytes))
    }));
    tokio::spawn(async move {
        axum::serve(target_listener, app)
            .with_graceful_shutdown(stop.cancelled_owned())
            .await
            .unwrap();
    });
    let data_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let control_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let data = data_listener.local_addr().unwrap();
    let control = control_listener.local_addr().unwrap();
    let stop = cancel.clone();
    tokio::spawn(async move {
        serve(data_listener, control_listener, TOKEN.into(), stop)
            .await
            .unwrap();
    });
    Fixture {
        data,
        control,
        target,
        cancel,
    }
}
async fn start_connector(f: &Fixture) {
    let endpoint = format!("ws://{}", f.control);
    let target = f.target;
    let cancel = f.cancel.clone();
    tokio::spawn(async move {
        connect(&endpoint, target, TOKEN.into(), true, cancel)
            .await
            .unwrap();
    });
    let client = reqwest::Client::new();
    for _ in 0..100 {
        let response = client
            .get(format!("http://{}/_mold/relay/health", f.control))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if response == "online\n" {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("connector never enrolled");
}
#[tokio::test]
async fn http_auth_large_binary_range_sse_concurrency() {
    let f = fixture().await;
    start_connector(&f).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let base = format!("http://{}", f.data);
    assert_eq!(client.get(&base).send().await.unwrap().status(), 401);
    let body = (0..2_000_000).map(|n| (n % 251) as u8).collect::<Vec<_>>();
    let response = client
        .post(&base)
        .bearer_auth("mold-key")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(response.bytes().await.unwrap().as_ref(), body);
    let response = client
        .get(&base)
        .bearer_auth("mold-key")
        .header("range", "bytes=2-5")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 206);
    assert_eq!(response.text().await.unwrap(), "2345");
    let response = client
        .head(&base)
        .bearer_auth("mold-key")
        .header("range", "bytes=2-5")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 206);
    assert!(response.bytes().await.unwrap().is_empty());
    let mut response = client
        .get(format!("{base}/sse"))
        .bearer_auth("mold-key")
        .send()
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_millis(300), response.chunk())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.as_ref(), b"data: 0\n\n");
    assert_eq!(
        response.chunk().await.unwrap().unwrap().as_ref(),
        b"data: 1\n\n"
    );
    let mut tasks = tokio::task::JoinSet::new();
    for n in 0..20 {
        let client = client.clone();
        let base = base.clone();
        tasks.spawn(async move {
            let body = format!("body-{n}");
            assert_eq!(
                client
                    .post(base)
                    .bearer_auth("mold-key")
                    .body(body.clone())
                    .send()
                    .await
                    .unwrap()
                    .text()
                    .await
                    .unwrap(),
                body
            );
        });
    }
    while let Some(result) = tasks.join_next().await {
        result.unwrap();
    }
}
#[tokio::test]
async fn offline_unauthorized_occupied_and_session_fencing() {
    let f = fixture().await;
    assert_eq!(
        reqwest::get(format!("http://{}", f.data))
            .await
            .unwrap()
            .status(),
        503
    );
    let endpoint = format!("ws://{}/_mold/relay/control", f.control);
    assert!(connect_async(&endpoint).await.is_err());
    let mut req = endpoint.into_client_request().unwrap();
    req.headers_mut()
        .insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    let (mut socket, _) = connect_async(req.clone()).await.unwrap();
    let session = match socket.next().await.unwrap().unwrap() {
        Message::Text(t) => t.to_string(),
        _ => panic!(),
    };
    assert!(connect_async(req).await.is_err());
    let mut request = format!(
        "ws://{}/_mold/relay/data/{session}/{}",
        f.control,
        uuid::Uuid::new_v4()
    )
    .into_client_request()
    .unwrap();
    request
        .headers_mut()
        .insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    assert!(connect_async(request).await.is_err());
    socket.close(None).await.unwrap();
    for _ in 0..100 {
        if reqwest::get(format!("http://{}/_mold/relay/health", f.control))
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
            == "offline\n"
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("session was not cleaned up");
}
#[tokio::test]
async fn auth_disabled_and_redirect_targets_fail_closed() {
    for status in [200, 302, 403] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target = listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        let app = Router::new().fallback(any(move || async move {
            Response::builder()
                .status(status)
                .header("location", "http://127.0.0.1:1/")
                .body(Body::empty())
                .unwrap()
        }));
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        });
        assert!(authenticated_target(target).await.is_err());
        cancel.cancel();
    }
}
async fn raw_fixture(upgrade: bool) -> Fixture {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut f = fixture().await;
    f.target = listener.local_addr().unwrap();
    let cancel = f.cancel.clone();
    tokio::spawn(async move {
        loop {
            let accepted = tokio::select! { _ = cancel.cancelled() => break, accepted = listener.accept() => accepted };
            let (mut socket, _) = accepted.unwrap();
            tokio::spawn(async move {
                loop {
                    let mut request = Vec::new();
                    let mut byte = [0];
                    while !request.ends_with(b"\r\n\r\n") {
                        if socket.read_exact(&mut byte).await.is_err() {
                            return;
                        }
                        request.push(byte[0]);
                    }
                    if request.starts_with(b"GET /api/status ") {
                        if socket
                            .write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    if upgrade {
                        socket.write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: fixture\r\nConnection: Upgrade\r\n\r\n").await.unwrap();
                        let mut buffer = [0; 8192];
                        loop {
                            let n = match socket.read(&mut buffer).await {
                                Ok(n) => n,
                                Err(_) => return,
                            };
                            if n == 0 {
                                return;
                            }
                            if socket.write_all(&buffer[..n]).await.is_err() {
                                return;
                            }
                        }
                    } else {
                        let mut body = Vec::new();
                        socket.read_to_end(&mut body).await.unwrap();
                        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello").await.unwrap();
                        return;
                    }
                }
            });
        }
    });
    f
}
#[tokio::test]
async fn tcp_half_close_retains_response() {
    // Target deliberately waits for directional EOF before responding.
    let f = raw_fixture(false).await;
    start_connector(&f).await;
    let mut tcp = TcpStream::connect(f.data).await.unwrap();
    tcp.write_all(b"POST / HTTP/1.1\r\nHost: fixture\r\nAuthorization: Bearer mold-key\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello").await.unwrap();
    tcp.shutdown().await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), tcp.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(
        response.ends_with(b"hello"),
        "{}",
        String::from_utf8_lossy(&response)
    );
}
#[tokio::test]
async fn one_use_streams_capacity_timeout_and_disconnect_cleanup() {
    let f = fixture().await;
    let mut req = format!("ws://{}/_mold/relay/control", f.control)
        .into_client_request()
        .unwrap();
    req.headers_mut()
        .insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    let (mut control, _) = connect_async(req.clone()).await.unwrap();
    let session = match control.next().await.unwrap().unwrap() {
        Message::Text(t) => t.to_string(),
        _ => panic!(),
    };
    let first = TcpStream::connect(f.data).await.unwrap();
    let stream = match control.next().await.unwrap().unwrap() {
        Message::Text(t) => t.to_string(),
        Message::Ping(_) => match control.next().await.unwrap().unwrap() {
            Message::Text(t) => t.to_string(),
            _ => panic!(),
        },
        _ => panic!(),
    };
    let mut data_req = format!("ws://{}/_mold/relay/data/{session}/{stream}", f.control)
        .into_client_request()
        .unwrap();
    data_req
        .headers_mut()
        .insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    let (attached, _) = connect_async(data_req.clone()).await.unwrap();
    assert!(
        connect_async(data_req.clone()).await.is_err(),
        "stream attachment is one-use"
    );
    let mut sockets = vec![first];
    for _ in 1..64 {
        sockets.push(TcpStream::connect(f.data).await.unwrap());
    }
    // Wait until the accept loop admits all connections; no data transport attaches.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut over_limit = TcpStream::connect(f.data).await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(2),
        over_limit.read_to_end(&mut response),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(response.starts_with(b"HTTP/1.1 503"));
    // Pending requests expire rather than keeping capacity forever.
    let mut pending = sockets.pop().unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(17), pending.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(response.starts_with(b"HTTP/1.1 503"));
    control.close(None).await.unwrap();
    drop(attached);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (_new_control, _) = connect_async(req).await.unwrap();
    assert!(
        connect_async(data_req).await.is_err(),
        "old session cannot attach after reconnect"
    );
}
#[tokio::test]
async fn http_upgrade_preserves_bidirectional_binary_and_disconnect() {
    let f = raw_fixture(true).await;
    start_connector(&f).await;
    let mut tcp = TcpStream::connect(f.data).await.unwrap();
    tcp.write_all(b"GET / HTTP/1.1\r\nHost: fixture\r\nAuthorization: Bearer fixture\r\nUpgrade: fixture\r\nConnection: Upgrade\r\n\r\n").await.unwrap();
    let mut headers = Vec::new();
    let mut byte = [0];
    while !headers.ends_with(b"\r\n\r\n") {
        tcp.read_exact(&mut byte).await.unwrap();
        headers.push(byte[0]);
    }
    assert!(headers.starts_with(b"HTTP/1.1 101"));
    let data = (0..100_000).map(|n| (n % 256) as u8).collect::<Vec<_>>();
    tcp.write_all(&data).await.unwrap();
    let mut echoed = vec![0; data.len()];
    tcp.read_exact(&mut echoed).await.unwrap();
    assert_eq!(echoed, data);
    f.cancel.cancel();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), tcp.read(&mut byte))
            .await
            .unwrap()
            .unwrap(),
        0
    );
}
#[tokio::test]
async fn authentication_probe_belongs_to_the_exact_tunneled_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = listener.local_addr().unwrap();
    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    tokio::spawn(async move {
        let mut first = true;
        loop {
            let accepted =
                tokio::select! {_ = stop.cancelled()=>break, result=listener.accept()=>result};
            let (mut tcp, _) = accepted.unwrap();
            let status = if first {
                first = false;
                "200 OK"
            } else {
                "401 Unauthorized"
            };
            tokio::spawn(async move {
                let mut buffer = [0; 4096];
                if tcp.read(&mut buffer).await.unwrap_or(0) > 0 {
                    let _ = tcp
                        .write_all(
                            format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").as_bytes(),
                        )
                        .await;
                }
            });
        }
    });
    assert!(
        authenticated_target(target).await.is_err(),
        "separate auth probe accepted an auth-disabled held connection"
    );
    cancel.cancel();
}
#[tokio::test]
async fn target_probe_rejects_ambiguous_or_nonpersistent_framing() {
    for headers in [
        "",
        "Content-Length: 0\r\nContent-Length: 0\r\n",
        "Content-Length: 0\r\nTransfer-Encoding: chunked\r\n",
        "Content-Length: 0\r\nConnection: close\r\n",
        "Content-Length: 65537\r\n",
        " Content-Length: 0\r\n",
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let _ = tcp.read(&mut request).await;
            let _ = tcp
                .write_all(format!("HTTP/1.1 401 Unauthorized\r\n{headers}\r\n").as_bytes())
                .await;
        });
        assert!(authenticated_target(target).await.is_err());
    }
}
