use anyhow::Result;

pub async fn run() -> Result<()> {
    mold_discord::run().await
}
