use crate::format;
use crate::handler;
use crate::state::Context;
use anyhow::Result;

/// Show mold server health and status.
#[poise::command(slash_command)]
pub async fn status(ctx: Context<'_>) -> Result<()> {
    ctx.defer().await?;

    let (status, devices, queue) = tokio::join!(
        ctx.data().client.server_status(),
        ctx.data().client.devices(),
        ctx.data().client.list_queue()
    );
    match status {
        Ok(server_status) => {
            let embed_data = format::format_server_status_with_devices(
                &server_status,
                devices.as_ref().ok(),
                queue.as_ref().ok(),
            );
            let embed = handler::embed_data_to_create_embed(&embed_data);

            ctx.send(poise::CreateReply::default().embed(embed)).await?;
        }
        Err(e) => {
            let msg = if mold_core::MoldClient::is_connection_error(&e) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else {
                format!("Failed to get server status: {e}")
            };
            handler::send_error(ctx, &msg).await?;
        }
    }

    Ok(())
}
