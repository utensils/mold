use anyhow::Result;
use mold_core::MoldClient;

use crate::ui::render_progress;

pub(crate) fn client_for_host(host: Option<&str>) -> MoldClient {
    match host {
        Some(host) => MoldClient::new(host),
        None => MoldClient::from_env(),
    }
}

pub(crate) async fn stream_server_pull(client: &MoldClient, model: &str) -> Result<()> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let render = tokio::spawn(render_progress(rx));
    let result = client.pull_model_stream(model, tx).await;
    let _ = render.await;
    result
}
