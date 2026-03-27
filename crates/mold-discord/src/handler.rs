use crate::format;
use crate::state::Context;
use anyhow::Result;
use mold_core::{GenerateRequest, MoldClient};
use poise::serenity_prelude::{CreateAttachment, CreateEmbed};
use std::time::Instant;

/// Minimum interval between Discord embed edits to avoid rate limits.
const EDIT_THROTTLE: std::time::Duration = std::time::Duration::from_secs(3);

/// Run image generation via SSE streaming, updating the deferred Discord reply
/// with progress events and attaching the final image.
pub async fn run_generation(ctx: Context<'_>, req: GenerateRequest) -> Result<()> {
    let prompt = req.prompt.clone();
    let client = &ctx.data().client;

    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

    // Spawn the streaming generation in a background task
    let client_clone = MoldClient::new(client.host());
    let req_clone = req.clone();
    let gen_handle =
        tokio::spawn(async move { client_clone.generate_stream(&req_clone, progress_tx).await });

    // Consume progress events and throttle embed updates
    let mut last_edit = Instant::now() - EDIT_THROTTLE;

    while let Some(event) = progress_rx.recv().await {
        let status_text = format::format_progress(&event);

        if last_edit.elapsed() >= EDIT_THROTTLE {
            let embed = CreateEmbed::new()
                .title("Generating...")
                .description(&status_text)
                .color(0x5865F2);
            let reply = poise::CreateReply::default().embed(embed);
            let _ = ctx.send(reply).await;
            last_edit = Instant::now();
        }
    }

    // Generation complete — get the result
    let result = gen_handle.await??;

    match result {
        Some(resp) => {
            send_result(ctx, &resp, &prompt).await?;
        }
        None => {
            // Server doesn't support SSE — fall back to non-streaming
            let resp = client.generate(req).await?;
            send_result(ctx, &resp, &prompt).await?;
        }
    }

    Ok(())
}

/// Send the final generation result with image attachment.
async fn send_result(
    ctx: Context<'_>,
    resp: &mold_core::GenerateResponse,
    prompt: &str,
) -> Result<()> {
    let embed_data = format::format_generation_result(resp, prompt);
    let mut embed = embed_data_to_create_embed(&embed_data);

    let mut reply = poise::CreateReply::default();

    if let Some(image) = resp.images.first() {
        let ext = match image.format {
            mold_core::OutputFormat::Png => "png",
            mold_core::OutputFormat::Jpeg => "jpeg",
        };
        let filename = format!("mold-{}.{ext}", resp.seed_used);
        let attachment = CreateAttachment::bytes(image.data.clone(), filename.clone());
        embed = embed.attachment(&filename);
        reply = reply.attachment(attachment);
    }

    reply = reply.embed(embed);
    ctx.send(reply).await?;

    Ok(())
}

/// Convert our format::EmbedData into a serenity CreateEmbed.
fn embed_data_to_create_embed(data: &format::EmbedData) -> CreateEmbed {
    let mut embed = CreateEmbed::new().title(&data.title).color(data.color);

    if !data.description.is_empty() {
        embed = embed.description(&data.description);
    }

    for (name, value, inline) in &data.fields {
        embed = embed.field(name, value, *inline);
    }

    embed
}

/// Send an error embed as the deferred response.
pub async fn send_error(ctx: Context<'_>, message: &str) -> Result<()> {
    let embed_data = format::format_error(message);
    let embed = embed_data_to_create_embed(&embed_data);
    ctx.send(poise::CreateReply::default().embed(embed)).await?;
    Ok(())
}
