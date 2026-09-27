use crate::checks::{self, AuthResult};
use crate::state::Context;
use anyhow::Result;
use mold_core::catalog_wire::CatalogSearchQuery;
use mold_core::{CreateVideoUpscaleJobRequest, VideoUpscaleSource};
use poise::serenity_prelude as serenity;

const PAGE_ROWS: usize = 20;
const CONTENT_MAX: usize = 1900;

fn safe_text(value: &str) -> String {
    value
        .replace('@', "@\u{200b}")
        .chars()
        .flat_map(|ch| match ch {
            '\\' | '*' | '_' | '~' | '`' | '|' | '>' => vec!['\\', ch],
            _ if ch.is_control() => vec![' '],
            _ => vec![ch],
        })
        .collect()
}

fn bounded(lines: impl IntoIterator<Item = String>) -> String {
    let mut output = String::new();
    for line in lines {
        let separator = usize::from(!output.is_empty());
        if output.chars().count() + separator + line.chars().count() > CONTENT_MAX {
            output.push_str("\n…more results omitted");
            break;
        }
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&line);
    }
    if output.is_empty() {
        "No results.".into()
    } else {
        output
    }
}

async fn ephemeral_error(ctx: Context<'_>, error: impl std::fmt::Display) -> Result<()> {
    ctx.send(
        poise::CreateReply::default()
            .content(safe_text(&error.to_string()))
            .ephemeral(true),
    )
    .await?;
    Ok(())
}

/// Search the server's live model catalog without installing anything.
#[poise::command(slash_command)]
pub async fn search(
    ctx: Context<'_>,
    #[description = "Model, checkpoint, or adapter name"] query: String,
    #[description = "Optional server-advertised model family"] family: Option<String>,
) -> Result<()> {
    if let AuthResult::Denied(message) = checks::check_access_only(&ctx).await {
        return ephemeral_error(ctx, message).await;
    }
    ctx.defer_ephemeral().await?;
    match ctx
        .data()
        .client
        .search_catalog(&CatalogSearchQuery {
            q: Some(query),
            family,
            page: Some(1),
            page_size: Some(10),
            include_nsfw: Some(false),
            ..CatalogSearchQuery::default()
        })
        .await
    {
        Ok(page) => {
            let mut lines = page
                .entries
                .into_iter()
                .take(10)
                .map(|entry| {
                    format!(
                        "`{}` — **{}** · {} · {}{}",
                        safe_text(&entry.id),
                        safe_text(&entry.name),
                        safe_text(&entry.family),
                        safe_text(&entry.kind),
                        if entry.installed { " · installed" } else { "" }
                    )
                })
                .collect::<Vec<_>>();
            for warning in page.provider_errors.into_iter().take(2) {
                lines.push(format!(
                    "Warning: {}: {}",
                    safe_text(&warning.source),
                    safe_text(&warning.message)
                ));
            }
            ctx.send(
                poise::CreateReply::default()
                    .content(bounded(lines))
                    .ephemeral(true),
            )
            .await?;
        }
        Err(error) => ephemeral_error(ctx, format!("Catalog search failed: {error}")).await?,
    }
    Ok(())
}

/// Inspect and control the host-wide generation queue.
#[poise::command(
    slash_command,
    rename = "queue",
    subcommands(
        "queue_list",
        "queue_show",
        "queue_pause",
        "queue_resume",
        "queue_pause_item",
        "queue_resume_item",
        "queue_cancel",
        "queue_retry"
    ),
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn queue(_ctx: Context<'_>) -> Result<()> {
    Ok(())
}

/// List queue rows without exposing prompts or source media.
#[poise::command(
    slash_command,
    rename = "list",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn queue_list(ctx: Context<'_>) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.list_queue_all().await {
        Ok(listing) => {
            let lines = listing.entries.into_iter().take(PAGE_ROWS).map(|row| {
                format!(
                    "`{}` · {} · `{}` · #{}",
                    safe_text(&row.id),
                    safe_text(&row.state),
                    safe_text(&row.model),
                    row.position + 1
                )
            });
            ctx.send(
                poise::CreateReply::default()
                    .content(bounded(lines))
                    .ephemeral(true),
            )
            .await?;
        }
        Err(error) => ephemeral_error(ctx, format!("Could not read the queue: {error}")).await?,
    }
    Ok(())
}

/// Show one queue row by its opaque id.
#[poise::command(
    slash_command,
    rename = "show",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn queue_show(
    ctx: Context<'_>,
    #[description = "Exact queue job id"] job_id: String,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.queue_job(&job_id).await {
        Ok(Some(detail)) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!(
                        "`{}` · {} · `{}` · #{}{}",
                        safe_text(&detail.job.id),
                        safe_text(&detail.job.state),
                        safe_text(&detail.job.model),
                        detail.job.position + 1,
                        detail
                            .job
                            .held_reason
                            .as_deref()
                            .map(|r| format!("\n{}", safe_text(r)))
                            .unwrap_or_default()
                    ))
                    .ephemeral(true),
            )
            .await?
        }
        Ok(None) => {
            ctx.send(
                poise::CreateReply::default()
                    .content("That queue job was not found.")
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, format!("Could not read that queue job: {error}")).await?;
            return Ok(());
        }
    };
    Ok(())
}

/// Pause dispatch for the whole host.
#[poise::command(
    slash_command,
    rename = "pause",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn queue_pause(ctx: Context<'_>) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.pause_queue().await {
        Ok(_) => {
            ctx.send(
                poise::CreateReply::default()
                    .content("Host dispatch is paused.")
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

/// Resume dispatch for the whole host.
#[poise::command(
    slash_command,
    rename = "resume",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn queue_resume(ctx: Context<'_>) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.resume_queue().await {
        Ok(_) => {
            ctx.send(
                poise::CreateReply::default()
                    .content("Host dispatch is resumed.")
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "pause-item",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Pause one waiting queue item without pausing host dispatch.
pub async fn queue_pause_item(
    ctx: Context<'_>,
    #[description = "Exact queue job id"] job_id: String,
) -> Result<()> {
    queue_item_transition(ctx, job_id, "pause").await
}

#[poise::command(
    slash_command,
    rename = "resume-item",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Resume one paused queue item without changing host dispatch.
pub async fn queue_resume_item(
    ctx: Context<'_>,
    #[description = "Exact queue job id"] job_id: String,
) -> Result<()> {
    queue_item_transition(ctx, job_id, "resume").await
}

async fn queue_item_transition(ctx: Context<'_>, job_id: String, action: &str) -> Result<()> {
    ctx.defer_ephemeral().await?;
    let result = if action == "pause" {
        ctx.data().client.pause_queue_job(&job_id).await
    } else {
        ctx.data().client.resume_queue_job(&job_id).await
    };
    match result {
        Ok(()) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!(
                        "{}d `{}`.",
                        if action == "pause" { "Pause" } else { "Resume" },
                        safe_text(&job_id)
                    ))
                    .ephemeral(true),
            )
            .await?;
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
        }
    }
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "cancel",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Cancel one queue item by its exact id.
pub async fn queue_cancel(
    ctx: Context<'_>,
    #[description = "Exact queue job id"] job_id: String,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.cancel_queue_job(&job_id).await {
        Ok(()) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!("Cancelled `{}`.", safe_text(&job_id)))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "retry",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Retry one held durable queue item by its exact id.
pub async fn queue_retry(
    ctx: Context<'_>,
    #[description = "Exact held queue job id"] job_id: String,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    let result: Result<(), String> = async {
        let status = ctx
            .data()
            .client
            .server_status()
            .await
            .map_err(|e| e.to_string())?;
        let instance = status
            .instance_id
            .ok_or_else(|| "This server does not advertise durable retry identity.".to_string())?;
        let row = ctx
            .data()
            .client
            .find_queue_job(&job_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "That queue job was not found.".to_string())?;
        let request = row
            .retry_request(&instance)
            .ok_or_else(|| "That row has no durable batch authority to retry.".to_string())?;
        ctx.data()
            .client
            .retry_queue_job(&request)
            .await
            .map_err(|e| e.to_string())
    }
    .await;
    match result {
        Ok(()) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!("Retried `{}`.", safe_text(&job_id)))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

/// Inspect or cancel host-wide model downloads.
#[poise::command(
    slash_command,
    subcommands("downloads_list", "downloads_cancel"),
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn downloads(_ctx: Context<'_>) -> Result<()> {
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "list",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// List active, queued, and recent model downloads.
pub async fn downloads_list(ctx: Context<'_>) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.list_downloads().await {
        Ok(listing) => {
            let lines = listing
                .active_jobs
                .into_iter()
                .chain(listing.queued)
                .chain(listing.history)
                .take(PAGE_ROWS)
                .map(|job| {
                    format!(
                        "`{}` · {:?} · `{}` · {}/{} files",
                        safe_text(&job.id),
                        job.status,
                        safe_text(&job.model),
                        job.files_done,
                        job.files_total
                    )
                });
            ctx.send(
                poise::CreateReply::default()
                    .content(bounded(lines))
                    .ephemeral(true),
            )
            .await?;
        }
        Err(error) => ephemeral_error(ctx, error).await?,
    }
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "cancel",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Cancel one model download by its exact id.
pub async fn downloads_cancel(
    ctx: Context<'_>,
    #[description = "Exact download job id"] job_id: String,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.cancel_download(&job_id).await {
        Ok(()) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!("Cancelled download `{}`.", safe_text(&job_id)))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

/// Manage framewise upscale jobs whose sources are exact Library filenames.
#[poise::command(
    slash_command,
    rename = "video-upscale",
    subcommands(
        "video_upscale_create",
        "video_upscale_list",
        "video_upscale_status",
        "video_upscale_cancel",
        "video_upscale_resume"
    ),
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn video_upscale(_ctx: Context<'_>) -> Result<()> {
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "create",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Create a framewise upscale job from an exact Library filename.
pub async fn video_upscale_create(
    ctx: Context<'_>,
    #[description = "Exact video filename in this host's Library"] filename: String,
    #[description = "Frame upscaler model"] model: Option<String>,
    #[description = "Tile size"] tile_size: Option<u32>,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    let request = CreateVideoUpscaleJobRequest {
        source: VideoUpscaleSource::Library { filename },
        model: model.unwrap_or_else(|| "real-esrgan-x4plus:fp16".into()),
        tile_size,
    };
    match ctx.data().client.create_video_upscale_job(&request).await {
        Ok(job) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!(
                        "Created `{}` · {:?}.\n{}",
                        safe_text(&job.id),
                        job.state,
                        safe_text(&job.disclosure)
                    ))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "list",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// List framewise video upscale jobs.
pub async fn video_upscale_list(ctx: Context<'_>) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.list_video_upscale_jobs().await {
        Ok(jobs) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(bounded(jobs.into_iter().take(PAGE_ROWS).map(|job| {
                        format!(
                            "`{}` · {:?} · {}/{} frames · `{}`",
                            safe_text(&job.id),
                            job.state,
                            job.completed_frames,
                            job.total_frames,
                            safe_text(&job.model)
                        )
                    })))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "status",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Show one framewise video upscale job.
pub async fn video_upscale_status(
    ctx: Context<'_>,
    #[description = "Exact video upscale job id"] job_id: String,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.get_video_upscale_job(&job_id).await {
        Ok(job) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!(
                        "`{}` · {:?} · {}/{} frames{}\n{}",
                        safe_text(&job.id),
                        job.state,
                        job.completed_frames,
                        job.total_frames,
                        job.output_filename
                            .as_deref()
                            .map(|f| format!(" · `{}`", safe_text(f)))
                            .unwrap_or_default(),
                        safe_text(&job.disclosure)
                    ))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "cancel",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Cancel one framewise video upscale job.
pub async fn video_upscale_cancel(
    ctx: Context<'_>,
    #[description = "Exact video upscale job id"] job_id: String,
) -> Result<()> {
    video_transition(ctx, job_id, "cancel").await
}

#[poise::command(
    slash_command,
    rename = "resume",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Resume one paused framewise video upscale job.
pub async fn video_upscale_resume(
    ctx: Context<'_>,
    #[description = "Exact video upscale job id"] job_id: String,
) -> Result<()> {
    video_transition(ctx, job_id, "resume").await
}

async fn video_transition(ctx: Context<'_>, job_id: String, action: &'static str) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx
        .data()
        .client
        .transition_video_upscale_job(&job_id, action)
        .await
    {
        Ok(job) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(format!("`{}` · {:?}.", safe_text(&job.id), job.state))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

/// Browse host Library metadata and thumbnails. No mutations are exposed.
#[poise::command(
    slash_command,
    subcommands("gallery_list", "gallery_show"),
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn gallery(_ctx: Context<'_>) -> Result<()> {
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "list",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// List bounded metadata for this host's live Library.
pub async fn gallery_list(ctx: Context<'_>) -> Result<()> {
    ctx.defer_ephemeral().await?;
    match ctx.data().client.list_gallery().await {
        Ok(items) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(bounded(items.into_iter().take(PAGE_ROWS).map(|item| {
                        format!(
                            "`{}` · `{}` · {} bytes",
                            safe_text(&item.filename),
                            safe_text(&item.metadata.model),
                            item.size_bytes.unwrap_or(0)
                        )
                    })))
                    .ephemeral(true),
            )
            .await?
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    Ok(())
}

#[poise::command(
    slash_command,
    rename = "show",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
/// Show metadata and a bounded thumbnail for one exact Library filename.
pub async fn gallery_show(
    ctx: Context<'_>,
    #[description = "Exact opaque Library filename"] filename: String,
) -> Result<()> {
    ctx.defer_ephemeral().await?;
    let item = match ctx.data().client.gallery_item(&filename).await {
        Ok(Some(item)) => item,
        Ok(None) => {
            ctx.send(
                poise::CreateReply::default()
                    .content("That Library item was not found.")
                    .ephemeral(true),
            )
            .await?;
            return Ok(());
        }
        Err(error) => {
            ephemeral_error(ctx, error).await?;
            return Ok(());
        }
    };
    let mut reply = poise::CreateReply::default()
        .content(format!(
            "`{}`\nModel: `{}`\nSize: {} bytes",
            safe_text(&item.filename),
            safe_text(&item.metadata.model),
            item.size_bytes.unwrap_or(0)
        ))
        .ephemeral(true);
    if let Ok(bytes) = ctx.data().client.get_gallery_thumbnail(&filename).await {
        if bytes.len() <= 8 * 1024 * 1024 {
            reply = reply.attachment(serenity::CreateAttachment::bytes(bytes, "thumbnail.webp"));
        }
    }
    ctx.send(reply).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_catalog_text_cannot_ping_or_break_markdown() {
        assert_eq!(
            safe_text("@everyone **boom**"),
            "@\u{200b}everyone \\*\\*boom\\*\\*"
        );
    }

    #[test]
    fn discord_content_is_bounded() {
        let output = bounded((0..200).map(|i| format!("{i}: {}", "x".repeat(100))));
        assert!(output.chars().count() <= 1930);
        assert!(output.contains("omitted"));
    }
}
