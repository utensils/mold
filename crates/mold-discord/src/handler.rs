use crate::format::{self, EmbedData};
use crate::state::Context;
use anyhow::Result;
use mold_core::{GenerateRequest, MoldClient, OutputFormat};
use poise::serenity_prelude::{CreateAttachment, CreateEmbed};
use std::time::Instant;

/// Minimum interval between Discord embed edits to avoid rate limits.
const EDIT_THROTTLE: std::time::Duration = std::time::Duration::from_secs(3);

/// Discord free-tier upload ceiling. We stay conservatively under the 25 MiB
/// boundary to leave headroom for multipart overhead. If the primary video
/// payload exceeds this we fall back to the always-generated gif_preview.
const MAX_ATTACHMENT_BYTES: usize = 24 * 1024 * 1024;

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

/// Pick the attachment bytes we want to send to Discord. Video responses take
/// precedence over image payloads; oversized MP4s fall back to the GIF preview
/// the server always bundles so users still see *something* in their channel.
pub fn select_attachment(resp: &mold_core::GenerateResponse, seed: u64) -> Option<DiscordPayload> {
    if let Some(video) = resp.video.as_ref() {
        let primary_too_big = video.data.len() > MAX_ATTACHMENT_BYTES;
        let has_preview = !video.gif_preview.is_empty();
        if primary_too_big && has_preview {
            return Some(DiscordPayload {
                filename: format!("mold-{seed}.gif"),
                data: video.gif_preview.clone(),
                note: Some(format!(
                    "Primary {} exceeded Discord's upload limit ({:.1} MiB); falling back to GIF preview.",
                    video.format.extension().to_ascii_uppercase(),
                    video.data.len() as f64 / (1024.0 * 1024.0)
                )),
            });
        }
        return Some(DiscordPayload {
            filename: format!("mold-{seed}.{}", video.format.extension()),
            data: video.data.clone(),
            note: None,
        });
    }

    resp.images.first().map(|image| {
        let ext = match image.format {
            OutputFormat::Png => "png",
            other => other.extension(),
        };
        DiscordPayload {
            filename: format!("mold-{seed}.{ext}"),
            data: image.data.clone(),
            note: None,
        }
    })
}

/// Bytes + filename destined for a Discord attachment.
#[derive(Debug, Clone)]
pub struct DiscordPayload {
    pub filename: String,
    pub data: Vec<u8>,
    /// Optional user-visible note (e.g. "primary output was too large, here's the preview").
    pub note: Option<String>,
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

    if let Some(payload) = select_attachment(resp, resp.seed_used) {
        if let Some(note) = &payload.note {
            embed = embed.footer(poise::serenity_prelude::CreateEmbedFooter::new(note));
        }
        let attachment = CreateAttachment::bytes(payload.data.clone(), payload.filename.clone());
        embed = embed.attachment(&payload.filename);
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

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GenerateResponse, ImageData, OutputFormat, VideoData};

    fn video_response(data: Vec<u8>, preview: Vec<u8>, format: OutputFormat) -> GenerateResponse {
        GenerateResponse {
            images: vec![],
            video: Some(VideoData {
                data,
                format,
                width: 768,
                height: 512,
                frames: 25,
                fps: 24,
                thumbnail: vec![],
                gif_preview: preview,
                has_audio: false,
                duration_ms: Some(1000),
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms: 1234,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            seed_used: 7,
            gpu: None,
        }
    }

    #[test]
    fn select_attachment_prefers_video_mp4() {
        let resp = video_response(vec![0u8; 1024], vec![], OutputFormat::Mp4);
        let payload = select_attachment(&resp, 7).expect("payload");
        assert_eq!(payload.filename, "mold-7.mp4");
        assert_eq!(payload.data.len(), 1024);
        assert!(payload.note.is_none());
    }

    #[test]
    fn select_attachment_uses_gif_for_gif_format() {
        let resp = video_response(vec![0u8; 512], vec![], OutputFormat::Gif);
        let payload = select_attachment(&resp, 42).expect("payload");
        assert_eq!(payload.filename, "mold-42.gif");
    }

    #[test]
    fn select_attachment_falls_back_when_video_too_large() {
        // Primary > MAX_ATTACHMENT_BYTES; preview has content.
        let huge = vec![0u8; MAX_ATTACHMENT_BYTES + 1];
        let preview = vec![0x47, 0x49, 0x46, 0x38, 0x39, 0x61]; // GIF89a stub
        let resp = video_response(huge, preview.clone(), OutputFormat::Mp4);
        let payload = select_attachment(&resp, 9).expect("payload");
        assert_eq!(payload.filename, "mold-9.gif");
        assert_eq!(payload.data, preview);
        assert!(payload.note.is_some());
        let note = payload.note.unwrap();
        assert!(note.contains("MP4"));
        assert!(note.contains("preview"));
    }

    #[test]
    fn select_attachment_keeps_large_video_when_no_preview() {
        // If there's no preview to fall back to, keep the big video — Discord
        // will reject it, which is better than silently dropping the output.
        let huge = vec![0u8; MAX_ATTACHMENT_BYTES + 1];
        let resp = video_response(huge.clone(), vec![], OutputFormat::Mp4);
        let payload = select_attachment(&resp, 1).expect("payload");
        assert_eq!(payload.filename, "mold-1.mp4");
        assert_eq!(payload.data.len(), huge.len());
    }

    #[test]
    fn select_attachment_falls_back_to_images_when_no_video() {
        let resp = GenerateResponse {
            images: vec![ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 1024,
                height: 1024,
                index: 0,
            }],
            video: None,
            generation_time_ms: 100,
            model: "flux-schnell:q8".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let payload = select_attachment(&resp, 5).expect("payload");
        assert_eq!(payload.filename, "mold-5.png");
        assert_eq!(payload.data, vec![1, 2, 3]);
    }

    #[test]
    fn select_attachment_returns_none_when_empty() {
        let resp = GenerateResponse {
            images: vec![],
            video: None,
            generation_time_ms: 10,
            model: "empty".to_string(),
            seed_used: 0,
            gpu: None,
        };
        assert!(select_attachment(&resp, 0).is_none());
    }
}
