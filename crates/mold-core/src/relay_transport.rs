//! HTTP facade: local engine sockets or a pinned end-to-end encrypted session.
use anyhow::Result;
use reqwest::{Client, RequestBuilder, Response};
use serde::Serialize;
#[derive(Clone)]
pub(crate) struct RelayClient {
    client: Client,
    mode: Result<mold_secure_client::ConnectionMode, String>,
    api_key: Option<String>,
}
impl RelayClient {
    #[cfg(test)]
    pub(crate) fn new(client: Client) -> Self {
        Self {
            client,
            mode: Ok(mold_secure_client::ConnectionMode::Secure),
            api_key: None,
        }
    }
    pub(crate) fn for_origin(client: Client, origin: &str, mut api_key: Option<String>) -> Self {
        let mode = mold_secure_client::configured_connection_mode(origin)
            .and_then(|mode| {
                if mode == mold_secure_client::ConnectionMode::Regular
                    && std::env::var("MOLD_CONNECTION_MODE").as_deref() == Ok("regular")
                {
                    let key = api_key.clone().or_else(|| {
                        mold_secure_client::load_regular_host(origin)
                            .ok()
                            .and_then(|host| host.api_key)
                    });
                    mold_secure_client::register_regular_ephemeral(
                        "cli-env",
                        mold_secure_client::RegularHost {
                            origin: origin.into(),
                            api_key: key,
                        },
                    )?;
                }
                if mode == mold_secure_client::ConnectionMode::Regular {
                    let host = mold_secure_client::load_regular_host(origin)?;
                    host.validate()?;
                    if api_key.is_none() {
                        api_key = host.api_key;
                    }
                }
                Ok(mode)
            })
            .map_err(|error| error.to_string());
        Self {
            client,
            mode,
            api_key,
        }
    }
    pub(crate) fn has_paired_api_key(&self, origin: &str) -> bool {
        match self.mode {
            Ok(mold_secure_client::ConnectionMode::Regular) => {
                self.api_key.as_deref().is_some_and(|key| {
                    !key.is_empty() && reqwest::header::HeaderValue::from_str(key).is_ok()
                })
            }
            Ok(mold_secure_client::ConnectionMode::Secure) => {
                mold_secure_client::load_pairing(origin).is_ok_and(|host| {
                    !host.api_key.is_empty()
                        && reqwest::header::HeaderValue::from_str(&host.api_key).is_ok()
                })
            }
            Err(_) => false,
        }
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
        let request = match &self.api_key {
            Some(key) => request.header("x-api-key", key),
            None => request,
        };
        RelayRequest {
            request,
            client: self.clone(),
        }
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
        let mode = self
            .client
            .mode
            .as_ref()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        request
            .headers_mut()
            .insert("x-mold-connection-mode", mode.as_str().parse()?);
        mold_secure_client::execute_http(&self.client.client, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_numeric_loopback_and_localhost_can_use_private_engine_http() {
        for url in [
            "http://localhost:7680",
            "http://127.0.0.1:7680",
            "http://[::1]:7680",
        ] {
            assert!(mold_secure_client::is_private_engine(
                &reqwest::Url::parse(url).unwrap()
            ))
        }
        for url in [
            "http://192.168.1.5:7680",
            "https://abc.mold-link.urandom.io",
            "http://localhost.evil.invalid",
        ] {
            assert!(!mold_secure_client::is_private_engine(
                &reqwest::Url::parse(url).unwrap()
            ))
        }
    }
    #[tokio::test]
    async fn unpaired_remote_fails_before_any_http_credentials_leave() {
        let error = RelayClient::new(Client::new())
            .post("https://unpaired.invalid/api/generate")
            .header("x-api-key", "fixture-secret")
            .send()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("No secure pairing"));
        assert!(!crate::client::MoldClient::is_connection_error(&error));
    }
}
