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

    // Route every `Config::load_or_default()` through the shared
    // DB-backed user-preference hook so this standalone binary sees the
    // same view as `mold discord` invoked from the main CLI.
    mold_db::config_sync::install_config_post_load_hook();

    mold_discord::run().await
}
