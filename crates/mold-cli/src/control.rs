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
        let client = match host {
            Some(host) => MoldClient::new(host),
            None => MoldClient::from_env(),
        };
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

pub(crate) async fn stream_server_pull(client: &MoldClient, model: &str) -> Result<()> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let render = tokio::spawn(render_progress(rx));
    let result = client.pull_model_stream(model, tx).await;
    let _ = render.await;
    result
}
