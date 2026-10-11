//! Scoped, read-only loopback adapter for native WebViews without ServiceWorker.
use crate::SecureClient;
use anyhow::{bail, Result};
use futures_util::StreamExt;
use http::{Request, Response, StatusCode};
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full, StreamBody};
use hyper::{
    body::{Bytes, Frame, Incoming},
    service::service_fn,
};
use hyper_util::rt::TokioIo;
use std::{collections::BTreeMap, convert::Infallible, sync::Arc};
use tokio_util::sync::CancellationToken;
type ResponseBody = UnsyncBoxBody<Bytes, anyhow::Error>;
struct State {
    client: SecureClient,
    server_origin: String,
    allowed_origin: String,
    host_header: String,
    paths: Vec<String>,
    capability: String,
}
/// Own this handle for exactly the WebView/media lifetime; dropping it cancels
/// listeners, active connections, and the underlying encrypted response bodies.
pub struct MediaBridge {
    urls: BTreeMap<String, String>,
    stop: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for MediaBridge {
    fn drop(&mut self) {
        self.stop.cancel();
        self.task.abort();
    }
}
impl MediaBridge {
    pub async fn start(
        client: SecureClient,
        server_origin: String,
        allowed_webview_origin: String,
        paths: Vec<String>,
    ) -> Result<Self> {
        if paths.is_empty()
            || paths.len() > 256
            || allowed_webview_origin.is_empty()
            || allowed_webview_origin.contains(['\r', '\n'])
        {
            bail!("invalid media bridge scope")
        }
        for path in &paths {
            if !path.starts_with("/api/gallery/")
                && !path.starts_with("/api/chain/")
                && !path.starts_with("/api/mesh/")
                && !path.starts_with("/api/source-media/")
                && !path.starts_with("/api/chain-jobs/")
            {
                bail!("media bridge requires an explicit media API path")
            }
            mold_secure::Record::Open {
                stream_id: 1,
                request: mold_secure::Open {
                    method: "GET".into(),
                    path: path.clone(),
                    headers: vec![],
                    length: Some(0),
                },
            }
            .validate()?;
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let capability = hex::encode(mold_secure::Identity::generate()?.signing_public());
        let urls = paths
            .iter()
            .enumerate()
            .map(|(n, path)| (path.clone(), format!("http://{address}/{capability}/{n}")))
            .collect();
        let state = Arc::new(State {
            client,
            server_origin,
            allowed_origin: allowed_webview_origin,
            host_header: address.to_string(),
            paths,
            capability,
        });
        let stop = CancellationToken::new();
        let server_stop = stop.clone();
        let task = tokio::spawn(async move {
            let _guard = server_stop.clone().drop_guard();
            let mut connections = tokio::task::JoinSet::new();
            let slots = Arc::new(tokio::sync::Semaphore::new(32));
            loop {
                tokio::select! {_=server_stop.cancelled()=>break,result=listener.accept()=>{let Ok((socket,peer))=result else{break};if !peer.ip().is_loopback(){continue}let Ok(permit)=slots.clone().try_acquire_owned()else{continue};let state=state.clone();let cancel=server_stop.clone();connections.spawn(async move{let _permit=permit;let handler=service_fn(move|request|serve(request,state.clone()));let connection=hyper::server::conn::http1::Builder::new().max_buf_size(32768).serve_connection(TokioIo::new(socket),handler);tokio::select!{_=cancel.cancelled()=>{},_=connection=>{}}});},Some(_)=connections.join_next(),if !connections.is_empty()=>{}}
            }
            connections.abort_all();
            while connections.join_next().await.is_some() {}
        });
        Ok(Self { urls, stop, task })
    }
    pub fn url(&self, path: &str) -> Option<&str> {
        self.urls.get(path).map(String::as_str)
    }
    pub fn stop(&self) {
        self.stop.cancel();
    }
}
fn failure(status: StatusCode) -> Response<ResponseBody> {
    Response::builder()
        .status(status)
        .header("cache-control", "no-store")
        .body(
            Full::new(Bytes::new())
                .map_err(|never: Infallible| match never {})
                .boxed_unsync(),
        )
        .unwrap()
}
async fn serve(
    request: Request<Incoming>,
    state: Arc<State>,
) -> Result<Response<ResponseBody>, Infallible> {
    if request.headers().get("host").and_then(|v| v.to_str().ok()) != Some(&state.host_header)
        || request.uri().query().is_some()
    {
        return Ok(failure(StatusCode::FORBIDDEN));
    }
    if request
        .headers()
        .get("origin")
        .is_some_and(|v| v.to_str().ok() != Some(&state.allowed_origin))
    {
        return Ok(failure(StatusCode::FORBIDDEN));
    }
    if request.method() != http::Method::GET && request.method() != http::Method::HEAD {
        return Ok(failure(StatusCode::METHOD_NOT_ALLOWED));
    }
    let expected = format!("/{}/", state.capability);
    let Some(index) = request.uri().path().strip_prefix(&expected).and_then(|s| {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            None
        } else {
            s.parse::<usize>().ok()
        }
    }) else {
        return Ok(failure(StatusCode::NOT_FOUND));
    };
    let Some(path) = state.paths.get(index) else {
        return Ok(failure(StatusCode::NOT_FOUND));
    };
    let mut upstream = reqwest::Client::new().request(
        request.method().clone(),
        format!("{}{path}", state.server_origin),
    );
    for name in ["range", "if-range", "if-none-match", "if-modified-since"] {
        if let Some(value) = request.headers().get(name) {
            upstream = upstream.header(name, value);
        }
    }
    let result = async {
        let response = state.client.send(upstream.build()?).await?;
        let mut builder = Response::builder().status(response.status());
        for (name, value) in response.headers() {
            if matches!(
                name.as_str(),
                "content-type"
                    | "content-length"
                    | "content-range"
                    | "accept-ranges"
                    | "etag"
                    | "last-modified"
            ) {
                builder = builder.header(name, value)
            }
        }
        builder = builder
            .header("access-control-allow-origin", &state.allowed_origin)
            .header(
                "access-control-expose-headers",
                "Content-Length, Content-Range, Accept-Ranges, ETag",
            )
            .header("vary", "Origin")
            .header("cache-control", "private, no-store");
        let stream = response
            .bytes_stream()
            .map(|r| r.map(Frame::data).map_err(anyhow::Error::from));
        Ok::<_, anyhow::Error>(builder.body(StreamBody::new(stream).boxed_unsync())?)
    }
    .await;
    Ok(result.unwrap_or_else(|_| failure(StatusCode::BAD_GATEWAY)))
}
