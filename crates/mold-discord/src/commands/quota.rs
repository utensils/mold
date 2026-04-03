use crate::format;
use crate::state::Context;
use anyhow::Result;
use poise::serenity_prelude as serenity;

/// Check your remaining daily generation quota.
#[poise::command(slash_command)]
pub async fn quota(ctx: Context<'_>) -> Result<()> {
    let user_id = ctx.author().id.get();
    let data = ctx.data();

    let used = data.quotas.usage(user_id);
    let embed_data = format::format_quota(used, data.config.daily_quota);

    let embed = serenity::CreateEmbed::new()
        .title(&embed_data.title)
        .description(&embed_data.description)
        .color(embed_data.color);

    let embed = embed_data
        .fields
        .iter()
        .fold(embed, |e, (name, value, inline)| {
            e.field(name, value, *inline)
        });

    ctx.send(poise::CreateReply::default().embed(embed).ephemeral(true))
        .await?;
    Ok(())
}
