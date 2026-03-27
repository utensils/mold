use crate::format;
use crate::handler;
use crate::state::Context;
use anyhow::Result;

/// List available models and their status.
#[poise::command(slash_command)]
pub async fn models(ctx: Context<'_>) -> Result<()> {
    ctx.defer().await?;

    match ctx.data().client.list_models_extended().await {
        Ok(models) => {
            let embed_data = format::format_model_list(&models);
            let embed = handler::embed_data_to_create_embed(&embed_data);

            ctx.send(poise::CreateReply::default().embed(embed)).await?;
        }
        Err(e) => {
            let msg = if mold_core::MoldClient::is_connection_error(&e) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else {
                format!("Failed to list models: {e}")
            };
            handler::send_error(ctx, &msg).await?;
        }
    }

    Ok(())
}
