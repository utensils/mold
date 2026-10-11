use mold_relay::authenticated_target;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tokio_util::sync::CancellationToken;
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
