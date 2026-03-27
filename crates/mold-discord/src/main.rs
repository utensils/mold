#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing (when run as a standalone binary).
    // When invoked via `mold discord`, the CLI sets up tracing instead.
    let filter = std::env::var("MOLD_LOG").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&filter)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    mold_discord::run().await
}
