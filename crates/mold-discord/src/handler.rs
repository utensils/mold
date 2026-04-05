use crate::format::{self, EmbedData};
use crate::state::Context;
use anyhow::Result;
use mold_core::{GenerateRequest, MoldClient};
use poise::serenity_prelude::{CreateAttachment, CreateEmbed};
use std::time::Instant;

/// Minimum interval between Discord embed edits to avoid rate limits.
const EDIT_THROTTLE: std::time::Duration = std::time::Duration::from_secs(3);

/// Convert our format::EmbedData into a serenity CreateEmbed.
pub fn embed_data_to_create_embed(data: &EmbedData) -> CreateEmbed {
    let mut embed = CreateEmbed::new().title(&data.title).color(data.color);

    if !data.description.is_empty() {
        embed = embed.description(&data.description);
    }

    for (name, value, inline) in &data.fields {
        embed = embed.field(name, value, *inline);
    }

    embed
}

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

    // Send initial progress message and capture the handle for edits
    let initial_embed = CreateEmbed::new()
        .title("Generating...")
        .description("Starting generation...")
        .color(0x5865F2);
    let reply_handle = ctx
        .send(poise::CreateReply::default().embed(initial_embed))
        .await?;

    // Consume progress events and throttle embed updates via edit
    let mut last_edit = Instant::now();

    while let Some(event) = progress_rx.recv().await {
        let status_text = format::format_progress(&event);

        if last_edit.elapsed() >= EDIT_THROTTLE {
            let embed = CreateEmbed::new()
                .title("Generating...")
                .description(&status_text)
                .color(0x5865F2);
            let _ = reply_handle
                .edit(ctx, poise::CreateReply::default().embed(embed))
                .await;
            last_edit = Instant::now();
        }
    }

    // Generation complete — get the result
    let result = gen_handle.await??;

    match result {
        Some(resp) => {
            send_result_edit(&reply_handle, ctx, &resp, &prompt).await?;
        }
        None => {
            // Server doesn't support SSE — fall back to non-streaming
            let resp = client.generate(req).await?;
            send_result_edit(&reply_handle, ctx, &resp, &prompt).await?;
        }
    }

    Ok(())
}

/// Edit the existing reply with the final generation result and image attachment.
async fn send_result_edit(
    handle: &poise::ReplyHandle<'_>,
    ctx: Context<'_>,
    resp: &mold_core::GenerateResponse,
    prompt: &str,
) -> Result<()> {
    let embed_data = format::format_generation_result(resp, prompt);
    let mut embed = embed_data_to_create_embed(&embed_data);

    let mut reply = poise::CreateReply::default();

    if let Some(image) = resp.images.first() {
        let ext = image.format.extension();
        let filename = format!("mold-{}.{ext}", resp.seed_used);
        let attachment = CreateAttachment::bytes(image.data.clone(), filename.clone());
        embed = embed.attachment(&filename);
        reply = reply.attachment(attachment);
    }

    reply = reply.embed(embed);
    handle.edit(ctx, reply).await?;

    Ok(())
}

/// Send an error embed as the deferred response.
pub async fn send_error(ctx: Context<'_>, message: &str) -> Result<()> {
    let embed_data = format::format_error(message);
    let embed = embed_data_to_create_embed(&embed_data);
    ctx.send(poise::CreateReply::default().embed(embed)).await?;
    Ok(())
}
