//! Authenticated HTTP facade for the optional AWS relay.
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn relay_hashes_small_bodies_and_rejects_unsafe_upload_grants() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/_mold/relay/info")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"protocol":2,"upload_threshold":2097152,"max_body_bytes":67108864}))).mount(&server).await;
        let hash = format!("{:x}", Sha256::digest(b"body"));
        Mock::given(method("POST"))
            .and(path("/api/test"))
            .and(header("x-amz-content-sha256", hash.as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"okay"))
            .expect(1)
            .mount(&server)
            .await;
        let client = RelayClient::new(Client::new());
        client.known.lock().unwrap().insert(server.uri(), true);
        assert_eq!(
            client
                .post(format!("{}/api/test", server.uri()))
                .body("body")
                .send()
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            &b"okay"[..]
        );
        Mock::given(method("POST")).and(path("/_mold/relay/uploads")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"id":"a","url":"https://evil.example/file","headers":{},"expires_at":9999999999u64}))).expect(2).mount(&server).await;
        for request in [
            client.put(format!("{}/api/upload", server.uri())),
            client.delete(format!("{}/api/upload", server.uri())),
        ] {
            assert!(request
                .body(vec![0; STAGE_THRESHOLD])
                .send()
                .await
                .unwrap_err()
                .to_string()
                .contains("untrusted relay upload URL"));
        }
        let methods: Vec<String> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter(|request| request.url.path() == "/_mold/relay/uploads")
            .map(|request| {
                serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["method"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(methods, ["PUT", "DELETE"]);
    }
    #[tokio::test]
    async fn object_download_omits_api_key_and_reconstructs_streamed_response() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/api/media")).and(header("x-api-key","private"))
            .respond_with(ResponseTemplate::new(200).insert_header("x-mold-relay-object","1").set_body_json(serde_json::json!({"url":format!("{}/_mold/objects/a?signature=x",server.uri()),"status":206,"headers":{"content-type":"video/mp4","x-mold-video-frames":"25"}}))).expect(1).mount(&server).await;
        Mock::given(method("GET"))
            .and(path("/_mold/objects/a"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"media"))
            .expect(1)
            .mount(&server)
            .await;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", "private".parse().unwrap());
        let client = RelayClient::new(Client::builder().default_headers(headers).build().unwrap());
        client.mark_relay(&server.uri());
        let response = client
            .get(format!("{}/api/media", server.uri()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 206);
        assert_eq!(response.headers()["x-mold-video-frames"], "25");
        assert_eq!(response.bytes().await.unwrap(), &b"media"[..]);
        let requests = server.received_requests().await.unwrap();
        let fetched = requests
            .iter()
            .find(|r| r.url.path() == "/_mold/objects/a")
            .unwrap();
        assert!(!fetched.headers.contains_key("x-api-key"));
        assert!(!fetched.headers.contains_key("x-amz-content-sha256"));
    }
    #[test]
    fn signed_object_handoff_stays_on_the_trusted_origin_and_prefix() {
        let origin = reqwest::Url::parse("https://mold-link.urandom.io/api/status").unwrap();
        assert!(validate_object_url(
            &origin,
            "https://mold-link.urandom.io/_mold/objects/a?sig=x"
        )
        .is_ok());
        assert!(validate_object_url(&origin, "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com/_mold/objects/a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=300&X-Amz-Signature=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef").is_ok());
        assert!(validate_object_url(&origin, "https://evil.example/_mold/objects/a").is_err());
        assert!(validate_object_url(&origin, "https://mold-link.urandom.io/api/status").is_err());
    }
}
use anyhow::{ensure, Context, Result};
use futures_util::TryStreamExt;
use http_body_util::BodyExt;
use reqwest::{Client, RequestBuilder, Response, Url};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const MAX_BODY: usize = 64 * 1024 * 1024;
const STAGE_THRESHOLD: usize = 2 * 1024 * 1024;

fn validate_object_url(origin: &Url, value: &str) -> Result<Url> {
    let url = Url::parse(value).context("invalid relay object URL")?;
    ensure!(
        (url.origin() == origin.origin() || signed_mold_s3(&url))
            && url.path().starts_with("/_mold/objects/")
            && url.username().is_empty()
            && url.password().is_none()
            && url.fragment().is_none(),
        "untrusted relay object URL"
    );
    Ok(url)
}

fn signed_mold_s3(url: &Url) -> bool {
    if url.scheme() != "https" || url.port_or_known_default() != Some(443) {
        return false;
    }
    let Some((bucket, service)) = url.host_str().and_then(|host| host.split_once(".s3.")) else {
        return false;
    };
    let service = service.strip_prefix("dualstack.").unwrap_or(service);
    let Some(region) = service.strip_suffix(".amazonaws.com") else {
        return false;
    };
    let Some(identity) = bucket.strip_prefix("mold-relay-") else {
        return false;
    };
    let Some((account, bucket_region)) = identity.split_once('-') else {
        return false;
    };
    if account.len() != 12
        || !account.bytes().all(|byte| byte.is_ascii_digit())
        || bucket_region != region
        || region.is_empty()
        || !region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return false;
    }
    let mut params = HashMap::new();
    for (key, value) in url.query_pairs() {
        if params.insert(key, value).is_some() {
            return false;
        }
    }
    params
        .get("X-Amz-Algorithm")
        .is_some_and(|v| v == "AWS4-HMAC-SHA256")
        && params
            .get("X-Amz-Expires")
            .and_then(|v| v.parse::<u16>().ok())
            .is_some_and(|v| (1..=900).contains(&v))
        && params
            .get("X-Amz-Signature")
            .is_some_and(|v| v.len() == 64 && v.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

#[derive(Clone)]
pub(crate) struct RelayClient {
    client: Client,
    unsigned: Client,
    known: Arc<Mutex<HashMap<String, bool>>>,
}
impl RelayClient {
    pub(crate) fn new(client: Client) -> Self {
        Self {
            client,
            unsigned: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("TLS client"),
            known: Default::default(),
        }
    }
    #[cfg(test)]
    pub(crate) fn mark_relay(&self, url: &str) {
        self.known.lock().unwrap().insert(url.to_string(), true);
    }
    pub(crate) fn get(&self, url: impl reqwest::IntoUrl) -> RelayRequest {
        self.wrap(self.client.get(url))
    }
    pub(crate) fn post(&self, url: impl reqwest::IntoUrl) -> RelayRequest {
        self.wrap(self.client.post(url))
    }
    pub(crate) fn put(&self, url: impl reqwest::IntoUrl) -> RelayRequest {
        self.wrap(self.client.put(url))
    }
    pub(crate) fn patch(&self, url: impl reqwest::IntoUrl) -> RelayRequest {
        self.wrap(self.client.patch(url))
    }
    pub(crate) fn delete(&self, url: impl reqwest::IntoUrl) -> RelayRequest {
        self.wrap(self.client.delete(url))
    }
    fn wrap(&self, request: RequestBuilder) -> RelayRequest {
        RelayRequest {
            request,
            client: self.clone(),
        }
    }
    pub(crate) async fn is_relay(&self, url: &Url) -> Result<bool> {
        let key = url.origin().ascii_serialization();
        if let Some(value) = self.known.lock().unwrap().get(&key).copied() {
            return Ok(value);
        }
        if url.scheme() != "https" {
            return Ok(false);
        }
        let mut info_url = url.clone();
        info_url.set_path("/_mold/relay/info");
        info_url.set_query(None);
        let value = match self
            .client
            .get(info_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let info: serde_json::Value =
                    response.json().await.context("invalid relay metadata")?;
                ensure!(
                    info["protocol"] == 2
                        && info["upload_threshold"] == STAGE_THRESHOLD
                        && info["max_body_bytes"] == MAX_BODY,
                    "unsupported relay metadata"
                );
                true
            }
            _ => false,
        };
        self.known.lock().unwrap().insert(key, value);
        Ok(value)
    }
}

pub(crate) struct RelayRequest {
    request: RequestBuilder,
    client: RelayClient,
}
impl RelayRequest {
    pub(crate) fn json<T: Serialize + ?Sized>(mut self, value: &T) -> Self {
        self.request = self.request.json(value);
        self
    }
    pub(crate) fn query<T: Serialize + ?Sized>(mut self, value: &T) -> Self {
        self.request = self.request.query(value);
        self
    }
    pub(crate) fn body(mut self, value: impl Into<reqwest::Body>) -> Self {
        self.request = self.request.body(value);
        self
    }
    pub(crate) fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        reqwest::header::HeaderName: TryFrom<K>,
        <reqwest::header::HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        reqwest::header::HeaderValue: TryFrom<V>,
        <reqwest::header::HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.request = self.request.header(key, value);
        self
    }
    pub(crate) async fn send(self) -> Result<Response> {
        let mut request = self.request.build()?;
        let origin = request.url().clone();
        if origin.scheme() == "https" {
            let target = format!(
                "{}{}",
                origin.path(),
                origin
                    .query()
                    .map(|query| format!("?{query}"))
                    .unwrap_or_default()
            );
            request
                .headers_mut()
                .insert("x-mold-request-target", target.parse()?);
        }
        if !self.client.is_relay(&origin).await? {
            return Ok(self.client.client.execute(request).await?);
        }
        if request.timeout().is_none() {
            *request.timeout_mut() = Some(std::time::Duration::from_secs(850));
        }
        let mut bytes = Vec::new();
        if let Some(mut body) = request.body_mut().take() {
            while let Some(frame) = body.frame().await {
                let frame = frame?;
                if let Ok(data) = frame.into_data() {
                    ensure!(
                        bytes.len() + data.len() <= MAX_BODY,
                        "relay request exceeds 64 MiB"
                    );
                    bytes.extend_from_slice(&data);
                }
            }
        }
        let hash = format!("{:x}", Sha256::digest(&bytes));
        request
            .headers_mut()
            .insert("x-amz-content-sha256", hash.parse()?);
        let response = if bytes.len() >= STAGE_THRESHOLD {
            let mut upload_url = origin.clone();
            upload_url.set_path("/_mold/relay/uploads");
            upload_url.set_query(None);
            let headers: HashMap<String, String> = request
                .headers()
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "authorization"
                            | "x-api-key"
                            | "host"
                            | "content-length"
                            | "x-amz-content-sha256"
                    )
                })
                .filter_map(|(key, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (key.to_string(), value.to_string()))
                })
                .collect();
            let path = format!(
                "{}{}",
                origin.path(),
                origin.query().map(|q| format!("?{q}")).unwrap_or_default()
            );
            let envelope = serde_json::to_vec(
                &serde_json::json!({"method":request.method().as_str(),"path":path,"headers":headers,"size":bytes.len(),"sha256":hash}),
            )?;
            let grant: serde_json::Value = signed_json(&self.client.client, upload_url, envelope)
                .await?
                .error_for_status()?
                .json()
                .await?;
            ensure!(
                grant["expires_at"].as_u64().is_some_and(|expiry| expiry
                    > std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()),
                "expired relay upload grant"
            );
            let put_url = Url::parse(grant["url"].as_str().context("missing relay upload URL")?)?;
            ensure!(
                signed_mold_s3(&put_url)
                    && put_url.username().is_empty()
                    && put_url.password().is_none(),
                "untrusted relay upload URL"
            );
            let mut upload = self.client.unsigned.put(put_url).body(bytes);
            if let Some(headers) = grant["headers"].as_object() {
                for (key, value) in headers {
                    ensure!(
                        !matches!(
                            key.to_ascii_lowercase().as_str(),
                            "authorization" | "x-api-key"
                        ),
                        "invalid relay upload headers"
                    );
                    upload =
                        upload.header(key, value.as_str().context("invalid relay upload header")?);
                }
            }
            upload
                .send()
                .await
                .map_err(|_| anyhow::anyhow!("relay upload failed"))?
                .error_for_status()
                .map_err(|_| anyhow::anyhow!("relay upload refused"))?;
            let id = grant["id"].as_str().context("missing relay upload ID")?;
            let mut dispatch_url = origin.clone();
            dispatch_url.set_path("/_mold/relay/request");
            dispatch_url.set_query(None);
            signed_json(
                &self.client.client,
                dispatch_url,
                serde_json::to_vec(&serde_json::json!({"id":id}))?,
            )
            .await?
        } else {
            *request.body_mut() = Some(bytes.into());
            self.client.client.execute(request).await?
        };
        if response
            .headers()
            .get("x-mold-relay-object")
            .is_none_or(|value| value != "1")
        {
            return Ok(response);
        }
        let object: serde_json::Value = response.json().await?;
        let url = validate_object_url(
            &origin,
            object["url"].as_str().context("missing relay object URL")?,
        )?;
        let fetched = self
            .client
            .unsigned
            .get(url)
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("relay object download failed"))?
            .error_for_status()
            .map_err(|_| anyhow::anyhow!("relay object download refused"))?;
        let status = object["status"]
            .as_u64()
            .context("invalid relay object status")?;
        let mut reconstructed = http::Response::builder().status(u16::try_from(status)?);
        if let Some(headers) = object["headers"].as_object() {
            for (key, value) in headers {
                if !matches!(
                    key.to_ascii_lowercase().as_str(),
                    "transfer-encoding" | "connection" | "content-length"
                ) {
                    reconstructed = reconstructed
                        .header(key, value.as_str().context("invalid relay object header")?);
                }
            }
        }
        Ok(reconstructed
            .body(reqwest::Body::wrap_stream(
                fetched.bytes_stream().map_err(reqwest::Error::without_url),
            ))?
            .into())
    }
}
async fn signed_json(client: &Client, url: Url, body: Vec<u8>) -> Result<Response> {
    let hash = format!("{:x}", Sha256::digest(&body));
    let target = format!(
        "{}{}",
        url.path(),
        url.query()
            .map(|query| format!("?{query}"))
            .unwrap_or_default()
    );
    Ok(client
        .post(url)
        .header("content-type", "application/json")
        .header("x-amz-content-sha256", hash)
        .header("x-mold-request-target", target)
        .body(body)
        .send()
        .await?)
}
