use anyhow::Result;
use mold_core::{
    build_model_catalog, classify_server_error, Config, ModelInfoExtended, MoldClient,
    ServerAvailability,
};

use crate::ui::render_progress;

pub(crate) enum ModelCatalogSource {
    Remote(Vec<ModelInfoExtended>),
    Local(Vec<ModelInfoExtended>),
}

pub(crate) struct CliContext {
    client: MoldClient,
    config: Config,
}

impl CliContext {
    pub(crate) fn new(host: Option<&str>) -> Self {
        let client = client_for_host(host);
        let config = Config::load_or_default();
        Self { client, config }
    }

    pub(crate) fn client(&self) -> &MoldClient {
        &self.client
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) async fn list_models(&self) -> Result<ModelCatalogSource> {
        match self.client.list_models_extended().await {
            Ok(models) => Ok(ModelCatalogSource::Remote(models)),
            Err(err) => match classify_server_error(&err) {
                ServerAvailability::FallbackLocal => Ok(ModelCatalogSource::Local(
                    build_model_catalog(&self.config, None, false),
                )),
                ServerAvailability::SurfaceError => Err(err),
            },
        }
    }

    pub(crate) async fn stream_server_pull(&self, model: &str) -> Result<()> {
        stream_server_pull(&self.client, model).await
    }
}

/// Build the exact CLI target client without contacting the host.
///
/// H3 authoring uses this before opening reference media so a missing or
/// invalid API-key header fails before local bytes are read.
pub(crate) fn client_for_host(host: Option<&str>) -> MoldClient {
    match host {
        Some(host) => std::env::var("MOLD_API_KEY")
            .ok()
            .filter(|key| !key.is_empty())
            .map_or_else(
                || MoldClient::new(host),
                |key| MoldClient::with_api_key(host, key),
            ),
        None => MoldClient::from_env(),
    }
}

/// Whether an unavailable HTTP target names this machine.
///
/// Server-first commands may fall back to local, read-only facts only for a
/// loopback target. Falling back when `MOLD_HOST` names another machine would
/// silently answer a different question.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    reqwest::Url::parse(host)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        })
}

pub(crate) async fn stream_server_pull(client: &MoldClient, model: &str) -> Result<()> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let render = tokio::spawn(render_progress(rx));
    let result = client.pull_model_stream(model, tx).await;
    let _ = render.await;
    result
}

#[cfg(test)]
mod tests {
    use super::is_loopback_host;

    #[test]
    fn local_fallback_is_limited_to_loopback_targets() {
        for host in [
            "http://localhost:7680",
            "http://127.0.0.1:7680",
            "http://127.0.0.2:7680",
            "http://[::1]:7680",
        ] {
            assert!(is_loopback_host(host), "{host}");
        }
        for host in ["http://gpu-box:7680", "https://10.0.0.8:7680", "not a url"] {
            assert!(!is_loopback_host(host), "{host}");
        }
    }
}
