use anyhow::{Context as _, Result};
use mold_core::MoldClient;
use poise::serenity_prelude as serenity;
use tracing::info;

use mold_discord::commands;
use mold_discord::state::{BotConfig, BotState};

fn load_token() -> Result<String> {
    std::env::var("MOLD_DISCORD_TOKEN")
        .or_else(|_| std::env::var("DISCORD_TOKEN"))
        .context(
            "Discord bot token not found. Set MOLD_DISCORD_TOKEN or DISCORD_TOKEN environment variable.",
        )
}

fn load_cooldown() -> u64 {
    std::env::var("MOLD_DISCORD_COOLDOWN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let filter = std::env::var("MOLD_LOG").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&filter)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let token = load_token()?;
    let client = MoldClient::from_env();
    let config = BotConfig {
        cooldown_seconds: load_cooldown(),
    };

    info!(
        host = client.host(),
        cooldown = config.cooldown_seconds,
        "Starting mold Discord bot"
    );

    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![
                commands::generate::generate(),
                commands::models::models(),
                commands::status::status(),
            ],
            on_error: |error| {
                Box::pin(async move {
                    tracing::error!("Framework error: {:?}", error);
                })
            },
            ..Default::default()
        })
        .setup(|ctx, _ready, framework| {
            Box::pin(async move {
                info!("Bot connected, registering slash commands...");
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                info!("Slash commands registered");
                Ok(BotState::new(client, config))
            })
        })
        .build();

    let intents = serenity::GatewayIntents::empty();
    let mut serenity_client = serenity::ClientBuilder::new(token, intents)
        .framework(framework)
        .await
        .context("Failed to create Discord client")?;

    info!("Bot starting...");
    serenity_client.start().await.context("Bot crashed")?;

    Ok(())
}
