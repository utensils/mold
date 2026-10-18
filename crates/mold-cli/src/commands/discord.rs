use anyhow::Result;

pub async fn run(host: Option<&str>) -> Result<()> {
    mold_discord::run_with_host(host).await
}
