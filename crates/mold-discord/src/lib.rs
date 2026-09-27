pub mod access;
pub mod checks;
pub mod commands;
pub mod cooldown;
pub mod format;
mod h3_references;
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

/// Every slash command the bot registers, in registration order.
pub fn slash_commands() -> Vec<poise::Command<state::BotState, anyhow::Error>> {
    vec![
        commands::generate::generate(),
        commands::identity::identity(),
        commands::transparent::transparent(),
        commands::mesh::mesh(),
        commands::expand::expand(),
        commands::remix::remix(),
        commands::upscale::upscale(),
        commands::operations::search(),
        commands::models::models(),
        commands::status::status(),
        commands::quota::quota(),
        commands::operations::queue(),
        commands::operations::downloads(),
        commands::operations::video_upscale(),
        commands::operations::gallery(),
        commands::admin::admin(),
    ]
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
            commands: slash_commands(),
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

#[cfg(test)]
mod tests {
    /// Discord rejects registration of a chat-input command with more than
    /// 25 options, or a description (command or option) over 100
    /// characters — and it rejects the WHOLE global registration, so one
    /// oversized command would take every command offline.
    #[test]
    fn every_command_fits_discords_registration_limits() {
        fn check(command: &poise::Command<crate::state::BotState, anyhow::Error>, path: &str) {
            assert!(
                command.parameters.len() <= 25,
                "/{path} has {} options",
                command.parameters.len()
            );
            assert!(
                command.subcommands.len() <= 25,
                "/{path} has {} subcommands",
                command.subcommands.len()
            );
            let description = command.description.as_deref().unwrap_or_default();
            assert!(
                !description.is_empty() && description.chars().count() <= 100,
                "/{path} description is {} characters",
                description.chars().count()
            );
            for parameter in &command.parameters {
                let description = parameter.description.as_deref().unwrap_or_default();
                assert!(
                    !description.is_empty() && description.chars().count() <= 100,
                    "/{path} option {} description is {} characters",
                    parameter.name,
                    description.chars().count()
                );
            }
            for subcommand in &command.subcommands {
                check(subcommand, &format!("{path} {}", subcommand.name));
            }
        }

        let commands = super::slash_commands();
        assert!(commands.iter().any(|command| command.name == "transparent"));
        for command in &commands {
            check(command, &command.name);
        }
    }

    #[test]
    fn operator_commands_are_guild_only_and_require_manage_guild() {
        let commands = super::slash_commands();
        for name in ["queue", "downloads", "video-upscale", "gallery"] {
            let command = commands
                .iter()
                .find(|command| command.name == name)
                .unwrap();
            assert!(command.guild_only, "/{name} must be guild-only");
            assert!(
                command
                    .required_permissions
                    .contains(super::serenity::Permissions::MANAGE_GUILD),
                "/{name} must require Manage Server"
            );
            for subcommand in &command.subcommands {
                assert!(
                    subcommand.guild_only,
                    "/{name} {} must be guild-only",
                    subcommand.name
                );
                assert!(
                    subcommand
                        .required_permissions
                        .contains(super::serenity::Permissions::MANAGE_GUILD),
                    "/{name} {} must require Manage Server",
                    subcommand.name
                );
            }
        }
    }

    #[test]
    fn intentional_cli_only_workflows_are_not_registered() {
        let commands = super::slash_commands();
        for forbidden in ["sequence", "jobs", "mesh-workflow"] {
            assert!(commands.iter().all(|command| command.name != forbidden));
        }
    }
}
