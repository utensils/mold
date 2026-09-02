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

/// Decide what the final edit carries: at most one primary attachment, an
/// optional poster for the embed, and a user-visible note. Audio and video
/// responses take precedence over image payloads; oversized MP4s fall back to
/// the GIF preview the server always bundles so users still see *something*
/// in their channel.
///
/// The note lives on the DELIVERY, not the attachment, so it survives when
/// nothing can be attached at all: an oversized mesh with no poster used to
/// lose its explanation along with its file, leaving a "Mesh Generated"
/// embed with statistics and no hint of where the geometry went.
pub fn plan_delivery(resp: &mold_core::GenerateResponse, seed: u64) -> DiscordDelivery {
    // A mesh is probed FIRST, for the same reason audio is probed before
    // video: a mesh response carries no images and no video, so any wider
    // probe would classify it as an empty response. Discord cannot render
    // glTF inline, so the poster the server rendered at save time is the
    // picture in the channel and the `.glb` rides beside it as a download.
    if let Some(mesh) = resp.mesh.as_ref() {
        let poster = (!mesh.poster.is_empty()).then(|| DiscordEmbedImage {
            filename: format!("mold-{seed}-poster.png"),
            data: mesh.poster.clone(),
        });
        let mebibytes = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        if mesh.data.len() > MAX_ATTACHMENT_BYTES {
            // The geometry cannot be posted; the poster still can, and the
            // note says where the mesh itself lives — with or without a
            // poster to show beside it.
            let note = format!(
                "Mesh {} exceeded Discord's upload limit ({:.1} MiB); {}. Fetch the .glb from the gallery.",
                mesh.format.extension().to_ascii_uppercase(),
                mebibytes(mesh.data.len()),
                if poster.is_some() {
                    "showing the poster"
                } else {
                    "there is no poster to show"
                }
            );
            return DiscordDelivery {
                attachment: poster.map(|poster| DiscordPayload {
                    filename: poster.filename,
                    data: poster.data,
                    embed_image: None,
                }),
                note: Some(note),
            };
        }
        // Both files ride ONE edit, so the budget is their SUM: a 23.9 MiB
        // mesh beside a 900 KB poster fails the whole edit. The poster is
        // the one to drop — the embed keeps its statistics, and the mesh is
        // what the user asked for.
        let poster_len = poster.as_ref().map_or(0, |poster| poster.data.len());
        let (embed_image, note) = if mesh.data.len() + poster_len > MAX_ATTACHMENT_BYTES {
            (
                None,
                Some(format!(
                    "Mesh and poster together exceed Discord's upload limit ({:.1} MiB); attaching the .glb alone.",
                    mebibytes(mesh.data.len() + poster_len)
                )),
            )
        } else {
            (poster, None)
        };
        return DiscordDelivery {
            attachment: Some(DiscordPayload {
                filename: format!("mold-{seed}.{}", mesh.format.extension()),
                data: mesh.data.clone(),
                embed_image,
            }),
            note,
        };
    }
    if let Some(audio) = resp.audio.as_ref() {
        // Discord renders a `.wav` attachment with its own inline player, so
        // the waveform PNG stays out of the channel — it exists for gallery
        // tiles, and posting it would read as a second, silent output.
        return DiscordDelivery::attachment(DiscordPayload {
            filename: format!("mold-{seed}.{}", audio.format.extension()),
            data: audio.data.clone(),
            embed_image: None,
        });
    }
    if let Some(video) = resp.video.as_ref() {
        let primary_too_big = video.data.len() > MAX_ATTACHMENT_BYTES;
        let has_preview = !video.gif_preview.is_empty();
        if primary_too_big && has_preview {
            return DiscordDelivery {
                attachment: Some(DiscordPayload {
                    filename: format!("mold-{seed}.gif"),
                    data: video.gif_preview.clone(),
                    embed_image: None,
                }),
                note: Some(format!(
                    "Primary {} exceeded Discord's upload limit ({:.1} MiB); falling back to GIF preview.",
                    video.format.extension().to_ascii_uppercase(),
                    video.data.len() as f64 / (1024.0 * 1024.0)
                )),
            };
        }
        return DiscordDelivery::attachment(DiscordPayload {
            filename: format!("mold-{seed}.{}", video.format.extension()),
            data: video.data.clone(),
            embed_image: None,
        });
    }

    DiscordDelivery {
        attachment: resp.images.first().map(|image| {
            let ext = match image.format {
                OutputFormat::Png => "png",
                other => other.extension(),
            };
            DiscordPayload {
                filename: format!("mold-{seed}.{ext}"),
                data: image.data.clone(),
                embed_image: None,
            }
        }),
        note: None,
    }
}

/// Everything the final edit carries besides the embed itself.
#[derive(Debug, Clone, Default)]
pub struct DiscordDelivery {
    /// The primary attachment, when the response has one Discord can take.
    pub attachment: Option<DiscordPayload>,
    /// Optional user-visible note (e.g. "primary output was too large, here's
    /// the preview"). Present with or without an attachment.
    pub note: Option<String>,
}

impl DiscordDelivery {
    fn attachment(payload: DiscordPayload) -> Self {
        Self {
            attachment: Some(payload),
            note: None,
        }
    }
}

/// Bytes + filename destined for a Discord attachment.
#[derive(Debug, Clone)]
pub struct DiscordPayload {
    pub filename: String,
    pub data: Vec<u8>,
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

    let delivery = plan_delivery(resp, resp.seed_used);
    if let Some(note) = &delivery.note {
        embed = embed.footer(poise::serenity_prelude::CreateEmbedFooter::new(note));
    }
    if let Some(payload) = delivery.attachment {
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
    fn plan_delivery_probes_mesh_first_and_pairs_the_poster() {
        let resp = mesh_response(b"glTF".to_vec(), b"\x89PNG".to_vec());
        let delivery = plan_delivery(&resp, 9);
        assert!(delivery.note.is_none());
        let payload = delivery.attachment.expect("payload");
        assert_eq!(payload.filename, "mold-9.glb");
        assert_eq!(payload.data, b"glTF");
        let poster = payload.embed_image.expect("the poster is embedded");
        assert_eq!(poster.filename, "mold-9-poster.png");
        assert_eq!(poster.data, b"\x89PNG");

        // Without a poster the `.glb` still posts, un-embedded.
        let bare = plan_delivery(&mesh_response(b"glTF".to_vec(), vec![]), 9)
            .attachment
            .unwrap();
        assert_eq!(bare.filename, "mold-9.glb");
        assert!(bare.embed_image.is_none());
    }

    /// An oversized mesh posts its poster with a note instead of failing the
    /// upload; with no poster there is nothing safe to post, but the note
    /// still reaches the embed so the user learns where the mesh went.
    #[test]
    fn plan_delivery_falls_back_to_the_poster_when_the_mesh_is_too_large() {
        let huge = vec![0u8; MAX_ATTACHMENT_BYTES + 1];
        let delivery = plan_delivery(&mesh_response(huge.clone(), b"\x89PNG".to_vec()), 9);
        let payload = delivery.attachment.expect("the poster is attached");
        assert_eq!(payload.filename, "mold-9-poster.png");
        assert_eq!(payload.data, b"\x89PNG");
        assert!(payload.embed_image.is_none());
        assert!(delivery
            .note
            .as_deref()
            .is_some_and(|note| note.contains("exceeded")
                && note.contains("showing the poster")
                && note.contains("gallery")));

        let posterless = plan_delivery(&mesh_response(huge, vec![]), 9);
        assert!(posterless.attachment.is_none());
        let note = posterless
            .note
            .expect("the note survives without an attachment");
        assert!(
            note.contains("exceeded") && note.contains("gallery"),
            "{note}"
        );
        assert!(!note.contains("showing the poster"), "{note}");
    }

    /// The `.glb` and its poster ride ONE edit, so the budget is their sum:
    /// a mesh just under the limit with a poster that tips it over drops the
    /// poster, keeps the mesh, and says so.
    #[test]
    fn plan_delivery_drops_the_poster_when_mesh_and_poster_together_exceed_the_budget() {
        let mesh = vec![0u8; MAX_ATTACHMENT_BYTES - 1024];
        let poster = vec![0u8; 4096];
        let delivery = plan_delivery(&mesh_response(mesh.clone(), poster.clone()), 9);
        let payload = delivery.attachment.expect("the mesh is attached");
        assert_eq!(payload.filename, "mold-9.glb");
        assert_eq!(payload.data.len(), mesh.len());
        assert!(
            payload.embed_image.is_none(),
            "the poster is the one to drop"
        );
        assert!(delivery
            .note
            .as_deref()
            .is_some_and(|note| note.contains("together") && note.contains(".glb alone")));

        // Exactly at the budget both still ride.
        let fits = vec![0u8; MAX_ATTACHMENT_BYTES - poster.len()];
        let delivery = plan_delivery(&mesh_response(fits, poster), 9);
        assert!(delivery.note.is_none());
        assert!(delivery.attachment.unwrap().embed_image.is_some());
    }

    #[test]
    fn plan_delivery_prefers_video_mp4() {
        let resp = video_response(vec![0u8; 1024], vec![], OutputFormat::Mp4);
        let delivery = plan_delivery(&resp, 7);
        assert!(delivery.note.is_none());
        let payload = delivery.attachment.expect("payload");
        assert_eq!(payload.filename, "mold-7.mp4");
        assert_eq!(payload.data.len(), 1024);
    }

    #[test]
    fn plan_delivery_uses_gif_for_gif_format() {
        let resp = video_response(vec![0u8; 512], vec![], OutputFormat::Gif);
        let payload = plan_delivery(&resp, 42).attachment.expect("payload");
        assert_eq!(payload.filename, "mold-42.gif");
    }

    #[test]
    fn plan_delivery_falls_back_when_video_too_large() {
        // Primary > MAX_ATTACHMENT_BYTES; preview has content.
        let huge = vec![0u8; MAX_ATTACHMENT_BYTES + 1];
        let preview = vec![0x47, 0x49, 0x46, 0x38, 0x39, 0x61]; // GIF89a stub
        let resp = video_response(huge, preview.clone(), OutputFormat::Mp4);
        let delivery = plan_delivery(&resp, 9);
        let payload = delivery.attachment.expect("payload");
        assert_eq!(payload.filename, "mold-9.gif");
        assert_eq!(payload.data, preview);
        assert!(delivery.note.is_some());
        let note = delivery.note.unwrap();
        assert!(note.contains("MP4"));
        assert!(note.contains("preview"));
    }

    #[test]
    fn plan_delivery_keeps_large_video_when_no_preview() {
        // If there's no preview to fall back to, keep the big video — Discord
        // will reject it, which is better than silently dropping the output.
        let huge = vec![0u8; MAX_ATTACHMENT_BYTES + 1];
        let resp = video_response(huge.clone(), vec![], OutputFormat::Mp4);
        let payload = plan_delivery(&resp, 1).attachment.expect("payload");
        assert_eq!(payload.filename, "mold-1.mp4");
        assert_eq!(payload.data.len(), huge.len());
    }

    #[test]
    fn plan_delivery_falls_back_to_images_when_no_video() {
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
        let payload = plan_delivery(&resp, 5).attachment.expect("payload");
        assert_eq!(payload.filename, "mold-5.png");
        assert_eq!(payload.data, vec![1, 2, 3]);
    }

    #[test]
    fn plan_delivery_returns_nothing_when_empty() {
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
        let delivery = plan_delivery(&resp, 0);
        assert!(delivery.attachment.is_none());
        assert!(delivery.note.is_none());
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
