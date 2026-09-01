use crate::format::{self, EmbedData};
use crate::state::Context;
use anyhow::Result;
use mold_core::{GenerateRequest, OutputFormat};
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
    // The response carries no metadata, so the identity note is derived from
    // the request the bot is about to send — the only place that knows which
    // photo went out.
    let identity_note = format::identity_note(&req);
    let client = &ctx.data().client;

    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

    // Spawn the streaming generation in a background task
    // Preserve the configured HTTP client (including its Authorization
    // header). Reconstructing from only the host breaks every bot generation
    // against an authenticated `mold serve`.
    let client_clone = client.clone();
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

    send_result_edit(
        &reply_handle,
        ctx,
        &result,
        &prompt,
        identity_note.as_deref(),
    )
    .await?;

    Ok(())
}

/// Pick the attachment bytes we want to send to Discord. Audio and video
/// responses take precedence over image payloads; oversized MP4s fall back to
/// the GIF preview the server always bundles so users still see *something* in
/// their channel.
pub fn select_attachment(resp: &mold_core::GenerateResponse, seed: u64) -> Option<DiscordPayload> {
    // A mesh is probed FIRST, for the same reason audio is probed before
    // video: a mesh response carries no images and no video, so any wider
    // probe would classify it as an empty response. Discord cannot render
    // glTF inline, so the poster the server rendered at save time is the
    // picture in the channel and the `.glb` rides beside it as a download.
    if let Some(mesh) = resp.mesh.as_ref() {
        let too_big = mesh.data.len() > MAX_ATTACHMENT_BYTES;
        let poster = (!mesh.poster.is_empty()).then(|| DiscordEmbedImage {
            filename: format!("mold-{seed}-poster.png"),
            data: mesh.poster.clone(),
        });
        if too_big {
            // The geometry cannot be posted; the poster still can, and the
            // note says where the mesh itself lives.
            let poster = poster?;
            return Some(DiscordPayload {
                filename: poster.filename.clone(),
                data: poster.data.clone(),
                note: Some(format!(
                    "Mesh {} exceeded Discord's upload limit ({:.1} MiB); showing the poster. Fetch the .glb from the gallery.",
                    mesh.format.extension().to_ascii_uppercase(),
                    mesh.data.len() as f64 / (1024.0 * 1024.0)
                )),
                embed_image: None,
            });
        }
        return Some(DiscordPayload {
            filename: format!("mold-{seed}.{}", mesh.format.extension()),
            data: mesh.data.clone(),
            note: None,
            embed_image: poster,
        });
    }
    if let Some(audio) = resp.audio.as_ref() {
        // Discord renders a `.wav` attachment with its own inline player, so
        // the waveform PNG stays out of the channel — it exists for gallery
        // tiles, and posting it would read as a second, silent output.
        return Some(DiscordPayload {
            filename: format!("mold-{seed}.{}", audio.format.extension()),
            data: audio.data.clone(),
            note: None,
            embed_image: None,
        });
    }
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
                embed_image: None,
            });
        }
        return Some(DiscordPayload {
            filename: format!("mold-{seed}.{}", video.format.extension()),
            data: video.data.clone(),
            note: None,
            embed_image: None,
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
            embed_image: None,
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
    /// A second, image-shaped attachment that takes the embed's image slot
    /// when the primary cannot (a mesh's poster beside its `.glb`). `None`
    /// means the primary itself is embedded, or is left as a bare
    /// attachment when it is an MP4.
    pub embed_image: Option<DiscordEmbedImage>,
}

/// The raster companion of a non-raster primary attachment.
#[derive(Debug, Clone)]
pub struct DiscordEmbedImage {
    pub filename: String,
    pub data: Vec<u8>,
}

/// Whether a Discord attachment filename is an MP4. Used to decide between
/// embedding the attachment inside the embed's `image` slot (PNG / JPEG / GIF
/// / APNG / WebP all render correctly there) and leaving the attachment
/// unreferenced so Discord renders its own inline video-player block — MP4
/// rendered as an embed image degrades to a static first-frame WebP preview
/// with no playback controls.
fn is_mp4_filename(filename: &str) -> bool {
    std::path::Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("mp4"))
}

/// Edit the existing reply with the final generation result and image attachment.
async fn send_result_edit(
    handle: &poise::ReplyHandle<'_>,
    ctx: Context<'_>,
    resp: &mold_core::GenerateResponse,
    prompt: &str,
    identity: Option<&str>,
) -> Result<()> {
    let embed_data = format::format_generation_result(resp, prompt, identity);
    let mut embed = embed_data_to_create_embed(&embed_data);

    let mut reply = poise::CreateReply::default();

    if let Some(payload) = select_attachment(resp, resp.seed_used) {
        if let Some(note) = &payload.note {
            embed = embed.footer(poise::serenity_prelude::CreateEmbedFooter::new(note));
        }
        let attachment = CreateAttachment::bytes(payload.data.clone(), payload.filename.clone());
        if let Some(poster) = payload.embed_image.as_ref() {
            // A non-raster primary (the `.glb`) rides as a plain download and
            // its poster takes the embed's image slot, so the channel shows a
            // picture and offers the geometry beside it.
            embed = embed.attachment(&poster.filename);
            reply = reply.attachment(CreateAttachment::bytes(
                poster.data.clone(),
                poster.filename.clone(),
            ));
        } else if !is_mp4_filename(&payload.filename) {
            // Only reference the attachment as the embed's image for
            // image-shaped formats. MP4 attachments get left off the embed so
            // Discord renders them as a separate inline video player block
            // below — setting `image.url = attachment://mold-*.mp4` on an
            // embed forces Discord's CDN to serve a WebP preview in the
            // embed's image slot, which shows up as a static first frame
            // instead of a playable video.
            embed = embed.attachment(&payload.filename);
        }
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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: Some(VideoData {
                video_only: None,
                attention_path: None,
                int8_arm: None,
                data,
                format,
                width: 768,
                height: 512,
                frames: 25,
                fps: 24,
                pipeline: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
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

    fn mesh_response(glb: Vec<u8>, poster: Vec<u8>) -> GenerateResponse {
        GenerateResponse {
            mesh: Some(mold_core::MeshData {
                data: glb,
                format: OutputFormat::Glb,
                vertex_count: 24_576,
                face_count: 49_152,
                bounds_min: [-0.5, -0.4, -0.3],
                bounds_max: [0.5, 0.4, 0.3],
                textured: false,
                poster,
                poster_width: 512,
                poster_height: 512,
            }),
            request_warnings: Vec::new(),
            audio: None,
            // A stray image beside the mesh must not win: the mesh is probed
            // first, as audio is probed before video.
            images: vec![ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 8,
                height: 8,
                index: 0,
            }],
            video: None,
            generation_time_ms: 4_000,
            model: mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string(),
            seed_used: 9,
            gpu: None,
        }
    }

    /// A mesh is probed before every other slot: the `.glb` is the primary
    /// attachment and its poster is the picture the embed shows.
    #[test]
    fn select_attachment_probes_mesh_first_and_pairs_the_poster() {
        let resp = mesh_response(b"glTF".to_vec(), b"\x89PNG".to_vec());
        let payload = select_attachment(&resp, 9).expect("payload");
        assert_eq!(payload.filename, "mold-9.glb");
        assert_eq!(payload.data, b"glTF");
        assert!(payload.note.is_none());
        let poster = payload.embed_image.expect("the poster is embedded");
        assert_eq!(poster.filename, "mold-9-poster.png");
        assert_eq!(poster.data, b"\x89PNG");

        // Without a poster the `.glb` still posts, un-embedded.
        let bare = select_attachment(&mesh_response(b"glTF".to_vec(), vec![]), 9).unwrap();
        assert_eq!(bare.filename, "mold-9.glb");
        assert!(bare.embed_image.is_none());
    }

    /// An oversized mesh posts its poster with a note instead of failing the
    /// upload; with no poster there is nothing safe to post.
    #[test]
    fn select_attachment_falls_back_to_the_poster_when_the_mesh_is_too_large() {
        let huge = vec![0u8; MAX_ATTACHMENT_BYTES + 1];
        let payload =
            select_attachment(&mesh_response(huge.clone(), b"\x89PNG".to_vec()), 9).unwrap();
        assert_eq!(payload.filename, "mold-9-poster.png");
        assert_eq!(payload.data, b"\x89PNG");
        assert!(payload.embed_image.is_none());
        assert!(payload
            .note
            .as_deref()
            .is_some_and(|note| note.contains("exceeded") && note.contains("gallery")));
        assert!(select_attachment(&mesh_response(huge, vec![]), 9).is_none());
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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: None,
            generation_time_ms: 10,
            model: "empty".to_string(),
            seed_used: 0,
            gpu: None,
        };
        assert!(select_attachment(&resp, 0).is_none());
    }

    #[test]
    fn is_mp4_filename_matches_mp4_case_insensitive() {
        assert!(is_mp4_filename("mold-7.mp4"));
        assert!(is_mp4_filename("mold-7.MP4"));
        assert!(is_mp4_filename("a.b.Mp4"));
    }

    #[test]
    fn is_mp4_filename_rejects_image_formats() {
        // These all render correctly inside an embed image, so we *want* to
        // keep `embed.attachment(...)` for them.
        assert!(!is_mp4_filename("mold-7.png"));
        assert!(!is_mp4_filename("mold-7.jpg"));
        assert!(!is_mp4_filename("mold-7.jpeg"));
        assert!(!is_mp4_filename("mold-7.gif"));
        assert!(!is_mp4_filename("mold-7.apng"));
        assert!(!is_mp4_filename("mold-7.webp"));
        assert!(!is_mp4_filename("mold-7"));
        assert!(!is_mp4_filename("mp4"));
    }
}
