pub mod access;
pub mod checks;
pub mod commands;
pub mod cooldown;
pub mod format;
pub mod handler;
pub mod quota;
pub mod state;

use access::AllowedRoles;
use anyhow::{Context as _, Result};
use mold_core::MoldClient;
use poise::serenity_prelude as serenity;
use state::{BotConfig, BotState};
use tracing::info;

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

fn load_allowed_roles() -> AllowedRoles {
    AllowedRoles::parse(std::env::var("MOLD_DISCORD_ALLOWED_ROLES").ok().as_deref())
}

fn load_daily_quota() -> Option<u32> {
    std::env::var("MOLD_DISCORD_DAILY_QUOTA")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
}

/// Start the Discord bot.
///
/// Reads configuration from environment variables:
/// - `MOLD_DISCORD_TOKEN` or `DISCORD_TOKEN` — bot token (required)
/// - `MOLD_HOST` — mold server URL (default: `http://localhost:7680`)
/// - `MOLD_DISCORD_COOLDOWN` — per-user cooldown in seconds (default: 10)
/// - `MOLD_DISCORD_ALLOWED_ROLES` — comma-separated role names/IDs (default: unrestricted)
/// - `MOLD_DISCORD_DAILY_QUOTA` — max generations per user per day (default: unlimited)
/// - `MOLD_LOG` — log level (default: `info`)
pub async fn run() -> Result<()> {
    let token = load_token()?;
    let client = MoldClient::from_env();
    let allowed_roles = load_allowed_roles();
    let daily_quota = load_daily_quota();
    let config = BotConfig {
        cooldown_seconds: load_cooldown(),
        allowed_roles,
        daily_quota,
    };

    info!(
        host = client.host(),
        cooldown = config.cooldown_seconds,
        roles_restricted = !config.allowed_roles.unrestricted,
        daily_quota = ?config.daily_quota,
        "Starting mold Discord bot"
    );

    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![
                commands::generate::generate(),
                commands::expand::expand(),
                commands::models::models(),
                commands::status::status(),
                commands::quota::quota(),
                commands::admin::admin(),
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
                let state = BotState::new(client.clone(), config);
                // Keep the model cache warm so autocomplete never races Discord's
                // 3-second interaction budget against server latency.
                BotState::spawn_model_cache_refresher(state.model_cache.clone(), client);
                Ok(state)
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
