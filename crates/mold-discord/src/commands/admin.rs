use crate::state::Context;
use anyhow::Result;

/// Admin commands for managing the bot.
#[poise::command(
    slash_command,
    subcommands("reset_quota", "block", "unblock"),
    required_permissions = "MANAGE_GUILD"
)]
pub async fn admin(_ctx: Context<'_>) -> Result<()> {
    // Parent command — Discord shows the subcommand picker instead.
    Ok(())
}

/// Reset a user's daily generation quota.
#[poise::command(
    slash_command,
    rename = "reset-quota",
    required_permissions = "MANAGE_GUILD"
)]
pub async fn reset_quota(
    ctx: Context<'_>,
    #[description = "User whose quota to reset"] user: poise::serenity_prelude::User,
) -> Result<()> {
    ctx.data().quotas.reset(user.id.get());
    ctx.send(
        poise::CreateReply::default()
            .content(format!("Quota reset for <@{}>.", user.id))
            .ephemeral(true),
    )
    .await?;
    Ok(())
}

/// Block a user from generating images (until bot restart).
#[poise::command(slash_command, required_permissions = "MANAGE_GUILD")]
pub async fn block(
    ctx: Context<'_>,
    #[description = "User to block"] user: poise::serenity_prelude::User,
) -> Result<()> {
    ctx.data().block_list.block(user.id.get());
    ctx.send(
        poise::CreateReply::default()
            .content(format!("<@{}> has been blocked from generating.", user.id))
            .ephemeral(true),
    )
    .await?;
    Ok(())
}

/// Unblock a previously blocked user.
#[poise::command(slash_command, required_permissions = "MANAGE_GUILD")]
pub async fn unblock(
    ctx: Context<'_>,
    #[description = "User to unblock"] user: poise::serenity_prelude::User,
) -> Result<()> {
    ctx.data().block_list.unblock(user.id.get());
    ctx.send(
        poise::CreateReply::default()
            .content(format!("<@{}> has been unblocked.", user.id))
            .ephemeral(true),
    )
    .await?;
    Ok(())
}
