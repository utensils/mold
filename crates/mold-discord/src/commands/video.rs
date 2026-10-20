//! Compact durable video authoring. No local GPU or transient reference uploads.

use crate::checks::{self, AuthResult};
use crate::state::Context;
use anyhow::{ensure, Context as _, Result};
use mold_core::chain_job::{ChainJobDetail, ChainJobState};
use mold_core::{
    ChainContinuation, ChainRequest, GenerationReference, GenerationReferenceAuthority,
    GenerationReferenceProvenance, ModelInfoExtended, SourceImageCapability,
};
use poise::serenity_prelude as serenity;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct VideoAuthoring {
    motion_tail_frames: u32,
    min_frames: u32,
    frame_step: u32,
    frame_offset: u32,
    #[serde(default)]
    source_fit_supported: bool,
}
#[derive(Deserialize)]
struct VideoLimits {
    model: String,
    supports_sequence: bool,
    frames_per_clip_cap: u32,
    frames_per_clip_recommended: u32,
    supports_audio: bool,
    authoring: Option<VideoAuthoring>,
}

/// Stable custom (v8) UUID derived from the Discord interaction. No in-memory ownership map
/// or successful reply is needed to identify an ambiguously accepted operation.
fn operation_id(interaction: u64) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.discord.video.operation.v1\0");
    hash.update(interaction.to_be_bytes());
    let digest = hash.finalize();
    let mut bytes: [u8; 16] = digest[..16].try_into().expect("SHA-256 has 32 bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

fn validate_job_id(id: &str) -> Result<()> {
    ensure!(id.len() <= 128, "Video job ID is too long.");
    ChainContinuation {
        job_id: id.into(),
        finalize_take: 1,
    }
    .validate()?;
    Ok(())
}

fn tail_for(model: &ModelInfoExtended, limits: &VideoLimits) -> u32 {
    limits
        .authoring
        .as_ref()
        .map(|value| value.motion_tail_frames)
        .unwrap_or_else(|| {
            mold_core::validation::chain_motion_tail_frames_for_model(
                &model.info.family,
                &model.info.name,
                model.source_image,
                mold_core::chain::DEFAULT_MOTION_TAIL_FRAMES,
            )
        })
}

fn clip_frames(
    model: &ModelInfoExtended,
    limits: &VideoLimits,
    fps: u32,
    length: Option<f64>,
    preferred: u32,
    tail: u32,
) -> Result<u32> {
    let authoring = limits.authoring.as_ref();
    let step = authoring
        .map(|value| value.frame_step)
        .or(model.defaults.frame_step)
        .or_else(|| mold_core::validation::frame_step_for_family(&model.info.family))
        .unwrap_or(1)
        .max(1);
    let offset = authoring
        .map(|value| value.frame_offset)
        .or(model.defaults.frame_offset)
        .unwrap_or(1);
    let minimum = authoring
        .map(|value| value.min_frames)
        .or(model.defaults.min_frames)
        .unwrap_or_else(|| {
            mold_core::validation::min_frames_for_model(&model.info.family, &model.info.name)
                .unwrap_or(1)
        })
        .max(tail.saturating_add(1));
    let seconds = length.unwrap_or(f64::from(preferred) / f64::from(fps));
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "Clip length must be a positive number of seconds."
    );
    let raw = seconds * f64::from(fps);
    ensure!(raw <= f64::from(u32::MAX), "Clip length is too large.");
    let snapped = ((raw - f64::from(offset)) / f64::from(step))
        .round()
        .max(0.0)
        * f64::from(step)
        + f64::from(offset);
    ensure!(
        snapped >= f64::from(minimum) && snapped <= f64::from(limits.frames_per_clip_cap),
        "This model needs each clip between {:.2} and {:.2} seconds on its frame grid.",
        f64::from(minimum) / f64::from(fps),
        f64::from(limits.frames_per_clip_cap) / f64::from(fps)
    );
    Ok(snapped as u32)
}

#[allow(clippy::too_many_arguments)]
fn make_request(
    model: &ModelInfoExtended,
    limits: &VideoLimits,
    prompt: &str,
    next_prompt: &str,
    length: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    image: Option<Vec<u8>>,
) -> Result<ChainRequest> {
    ensure!(
        limits.supports_sequence && limits.model == model.info.name,
        "Choose a model that supports video clips on this machine."
    );
    ensure!(
        model.runtime_available != Some(false),
        "This model is not runnable on this machine."
    );
    ensure!(
        !prompt.trim().is_empty() && !next_prompt.trim().is_empty(),
        "Write both clip prompts."
    );
    let fps = model.defaults.default_fps.unwrap_or(24).max(1);
    let tail = tail_for(model, limits);
    let frames = clip_frames(
        model,
        limits,
        fps,
        length,
        limits.frames_per_clip_recommended,
        tail,
    )?;
    let mut request: ChainRequest = serde_json::from_value(serde_json::json!({
        "model":model.info.name,"width":width.unwrap_or(model.defaults.default_width),
        "height":height.unwrap_or(model.defaults.default_height),"fps":fps,
        "steps":model.defaults.default_steps,"guidance":model.defaults.default_guidance,
        "motion_tail_frames":tail,"enable_audio":limits.supports_audio,
        "output_format":"mp4","output_mode":"sequence",
        "stages":[{"prompt":prompt,"frames":frames},{"prompt":next_prompt,"frames":frames}]
    }))?;
    let task = mold_core::minimax_h3::task_for_model(&model.info.name);
    if task == Some(mold_core::minimax_h3::Task::Ref2va) {
        let bytes =
            image.context("This reference model needs an image attachment, used in both clips.")?;
        let format = image::guess_format(&bytes)?;
        let (width, height) = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()?
            .into_dimensions()?;
        ensure!(
            matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg),
            "Use a PNG or JPEG reference image."
        );
        ensure!(
            bytes.len() <= 16 * 1024 * 1024,
            "The repeated image must fit within the 32 MiB part budget."
        );
        let reference = GenerationReference::Image {
            media: GenerationReferenceAuthority::Inline { data: bytes },
            provenance: GenerationReferenceProvenance::default(),
            mime_type: if format == image::ImageFormat::Png {
                "image/png"
            } else {
                "image/jpeg"
            }
            .into(),
            width,
            height,
        };
        for stage in &mut request.stages {
            stage.generation_references = Some(vec![reference.clone()]);
        }
    } else {
        let contract = mold_core::validation::source_image_capability_for_engine(
            Some(&model.info.family),
            model.source_image,
        );
        ensure!(
            !(contract == Some(SourceImageCapability::Unsupported) && image.is_some()),
            "This model does not accept an opening image."
        );
        ensure!(
            !(contract == Some(SourceImageCapability::Required) && image.is_none()),
            "This model needs an opening image."
        );
        request.stages[0].source_image = image;
    }
    Ok(request)
}

fn continue_request(
    source: &ChainJobDetail,
    model: &ModelInfoExtended,
    limits: &VideoLimits,
    prompt: &str,
    length: Option<f64>,
) -> Result<ChainRequest> {
    ensure!(
        source.summary.state == ChainJobState::Completed && !source.finalizes.is_empty(),
        "The original video must be completed."
    );
    ensure!(
        source
            .finalizes
            .last()
            .and_then(|value| value.gallery_filename.as_ref())
            .is_some(),
        "The source has no published Library video."
    );
    ensure!(
        source.script.schema == "mold.chain.v1" && source.script.chain.model == model.info.name,
        "The source model is unavailable on this machine."
    );
    ensure!(
        !prompt.trim().is_empty(),
        "Write a prompt for the next clip."
    );
    let previous = source
        .script
        .stages
        .last()
        .context("The source has no clips.")?;
    let settings = &source.script.chain;
    ensure!(
        settings.source_fit.is_none()
            || limits
                .authoring
                .as_ref()
                .is_some_and(|value| value.source_fit_supported),
        "Update this machine to preserve the original source fitting policy."
    );
    let frames = clip_frames(
        model,
        limits,
        settings.fps,
        length,
        previous.frames,
        settings.motion_tail_frames,
    )?;
    let mut request: ChainRequest = serde_json::from_value(serde_json::json!({
        "model":settings.model,"width":settings.width,"height":settings.height,"fps":settings.fps,
        "steps":settings.steps,"guidance":settings.guidance,"strength":settings.strength,
        "motion_tail_frames":settings.motion_tail_frames,"enable_audio":settings.enable_audio,
        "source_fit":settings.source_fit,"output_format":"mp4","output_mode":"sequence",
        "stages":[{"prompt":prompt,"frames":frames}]
    }))?;
    request.continue_from = Some(ChainContinuation {
        job_id: source.summary.id.clone(),
        finalize_take: u32::try_from(source.finalizes.len())?,
    });
    request.stages[0].negative_prompt = previous.negative_prompt.clone();
    request.stages[0].loras = previous.loras.clone();
    if let Some(references) = &previous.generation_references {
        let mut bytes = 0usize;
        for reference in references {
            let GenerationReferenceAuthority::Inline { data } = reference.media() else {
                anyhow::bail!("Restore original inline references before continuing this video.");
            };
            bytes = bytes
                .checked_add(data.len())
                .context("Reference size overflowed.")?;
        }
        ensure!(
            bytes <= 32 * 1024 * 1024
                && references.len() + settings.motion_tail_frames as usize <= 64,
            "The next part exceeds its bounded reference budget."
        );
        request.stages[0].generation_references = Some(references.clone());
    }
    Ok(request)
}

struct GenerationCharge<'a> {
    quotas: &'a crate::quota::QuotaTracker,
    user: u64,
    refundable: bool,
}
impl<'a> GenerationCharge<'a> {
    fn new(ctx: &'a crate::state::BotState, user: u64) -> Self {
        Self {
            quotas: &ctx.quotas,
            user,
            refundable: true,
        }
    }
    fn attempting_admission(&mut self) {
        self.refundable = false;
    }
}
impl Drop for GenerationCharge<'_> {
    fn drop(&mut self) {
        if self.refundable {
            self.quotas.refund(self.user);
        }
    }
}

async fn reply(ctx: Context<'_>, message: impl std::fmt::Display) -> Result<()> {
    let text: String = super::operations::safe_text(&message.to_string())
        .chars()
        .take(1900)
        .collect();
    ctx.send(poise::CreateReply::default().content(text).ephemeral(true))
        .await?;
    Ok(())
}
async fn live_model(ctx: Context<'_>, requested: Option<String>) -> Result<ModelInfoExtended> {
    let models = ctx.data().client.list_models_extended().await?;
    let requested = requested.map(|name| mold_core::manifest::resolve_model_name(&name));
    models.into_iter().find(|model| requested.as_ref().map_or(
        model.downloaded && model.runtime_available != Some(false) && model.supports_sequence == Some(true), |name| model.info.name == *name))
        .context("Choose an installed video model with the model option. Use /models to inspect this machine.")
}
async fn limits(ctx: Context<'_>, model: &str, fps: u32) -> Result<VideoLimits> {
    Ok(serde_json::from_value(
        ctx.data().client.chain_limits(model, Some(fps)).await?,
    )?)
}
async fn enqueue(
    ctx: Context<'_>,
    request: ChainRequest,
    mut charge: GenerationCharge<'_>,
) -> Result<()> {
    let plan = match ctx.data().client.validate_chain_request(&request).await {
        Ok(plan) => plan,
        Err(error) => return reply(ctx, format!("Video validation failed: {error}")).await,
    };
    let id = operation_id(ctx.id());
    charge.attempting_admission();
    match ctx.data().client.create_chain_job_with_operation_id(&request, &id).await {
        Ok(accepted) => {
            ensure!(accepted.job_id == id, "The host returned an unexpected video identity; reconcile {id}.");
            let mut message = format!("Video queued: {id}\n{} clips · {:.2} seconds delivered. Saved on this machine in Mold Library when complete.\nManage Server permission is needed for /video status job_id:{id} and continuing this video.", plan.stage_count, plan.estimated_duration_ms as f64 / 1000.0);
            for warning in accepted.request_warnings { message.push_str(&format!("\n{warning}")); }
            reply(ctx, message).await
        }
        Err(error) => reply(ctx, format!("Video job ID: {id}\nCould not confirm acceptance. Check /video status job_id:{id}; do not repeat make or continue while its outcome is unknown.\n{error}")).await,
    }
}

/// Make, continue and inspect videos saved in the server's Mold Library.
#[poise::command(
    slash_command,
    subcommands(
        "make",
        "continue_video",
        "video_status",
        "video_resume",
        "video_cancel"
    )
)]
pub async fn video(_ctx: Context<'_>) -> Result<()> {
    Ok(())
}

/// Join two prompted clips into one video, saved in Mold Library.
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn make(
    ctx: Context<'_>,
    #[description = "What happens in the first clip"] prompt: String,
    #[description = "What happens in the next clip"] next_prompt: String,
    #[description = "Installed video model"]
    #[autocomplete = "super::generate::autocomplete_model"]
    model: Option<String>,
    #[description = "Seconds per clip, snapped to the model's frame grid"] length: Option<f64>,
    #[description = "Video width in pixels"] width: Option<u32>,
    #[description = "Video height in pixels"] height: Option<u32>,
    #[description = "Opening image, or the reference used in every Ref2VA clip"] image: Option<
        serenity::Attachment,
    >,
) -> Result<()> {
    if let AuthResult::Denied(message) = checks::check_generate_auth(&ctx).await {
        return reply(ctx, message).await;
    }
    let charge = GenerationCharge::new(ctx.data(), ctx.author().id.get());
    ctx.defer_ephemeral().await?;
    let prepared: Result<ChainRequest> = async {
        let model = live_model(ctx, model).await?;
        let limits = limits(
            ctx,
            &model.info.name,
            model.defaults.default_fps.unwrap_or(24),
        )
        .await?;
        let image = match image {
            Some(image) => Some(
                super::generate::fetch_source_image(&image)
                    .await
                    .map_err(anyhow::Error::msg)?,
            ),
            None => None,
        };
        make_request(
            &model,
            &limits,
            &prompt,
            &next_prompt,
            length,
            width,
            height,
            image,
        )
    }
    .await;
    match prepared {
        Ok(request) => enqueue(ctx, request, charge).await,
        Err(error) => reply(ctx, error).await,
    }
}

/// Continue a completed video into its next separately saved part.
#[poise::command(
    slash_command,
    rename = "continue",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn continue_video(
    ctx: Context<'_>,
    #[description = "Completed source job ID on this machine"] job_id: String,
    #[description = "What happens in the next clip"] prompt: String,
    #[description = "Seconds in the next clip; default keeps its previous length"] length: Option<
        f64,
    >,
) -> Result<()> {
    if let Err(error) = validate_job_id(&job_id) {
        return reply(ctx, error).await;
    }
    if let AuthResult::Denied(message) = checks::check_generate_auth(&ctx).await {
        return reply(ctx, message).await;
    }
    let charge = GenerationCharge::new(ctx.data(), ctx.author().id.get());
    ctx.defer_ephemeral().await?;
    let prepared: Result<ChainRequest> = async {
        let source = ctx.data().client.get_chain_job_with_media(&job_id).await?;
        ensure!(
            ctx.data()
                .client
                .linked_chain_parts_supported(&source.summary.model)
                .await?
                == Some(true),
            "Update this machine before continuing video parts."
        );
        let model = live_model(ctx, Some(source.summary.model.clone())).await?;
        let limits = limits(ctx, &model.info.name, source.script.chain.fps).await?;
        continue_request(&source, &model, &limits, &prompt, length)
    }
    .await;
    match prepared {
        Ok(request) => enqueue(ctx, request, charge).await,
        Err(error) => reply(ctx, error).await,
    }
}

/// Read video progress and its exact published Library filename.
#[poise::command(
    slash_command,
    rename = "status",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn video_status(
    ctx: Context<'_>,
    #[description = "Video job ID"] job_id: String,
) -> Result<()> {
    if let Err(error) = validate_job_id(&job_id) {
        return reply(ctx, error).await;
    }
    if let AuthResult::Denied(message) = checks::check_access_only(&ctx).await {
        return reply(ctx, message).await;
    }
    ctx.defer_ephemeral().await?;
    match ctx.data().client.get_chain_job(&job_id).await {
        Ok(detail) => {
            let published = detail.finalizes.last().and_then(|value| value.gallery_filename.as_deref());
            let result = if let Some(filename) = published { format!("\nLibrary video: {filename}") }
                else if detail.summary.state == ChainJobState::Completed { "\nFinished without a published Library file. Check this machine's output settings.".into() }
                else { String::new() };
            reply(ctx, format!("{} · {} · clip {} of {}{}", detail.summary.id, detail.summary.state.as_str(), detail.summary.current_stage.saturating_add(1).min(detail.summary.stage_count), detail.summary.stage_count, result)).await
        }
        Err(error) => reply(ctx, format!("Video status unavailable: {error}. Keep this job ID; do not resubmit unknown work.")).await,
    }
}

/// Resume an interrupted or failed host-wide video job (Manage Server).
#[poise::command(
    slash_command,
    rename = "resume",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn video_resume(
    ctx: Context<'_>,
    #[description = "Video job ID"] job_id: String,
) -> Result<()> {
    if let Err(error) = validate_job_id(&job_id) {
        return reply(ctx, error).await;
    }
    ctx.defer_ephemeral().await?;
    match ctx.data().client.resume_chain_job(&job_id).await {
        Ok(_) => reply(ctx, format!("Video {job_id} queued to resume from its saved clips.")).await,
        Err(error) => reply(ctx, format!("Could not confirm resume for {job_id}: {error}. Check /video status before retrying.")).await,
    }
}

/// Cancel a host-wide video job, retaining its saved clips (Manage Server).
#[poise::command(
    slash_command,
    rename = "cancel",
    guild_only,
    required_permissions = "MANAGE_GUILD"
)]
pub async fn video_cancel(
    ctx: Context<'_>,
    #[description = "Video job ID"] job_id: String,
) -> Result<()> {
    if let Err(error) = validate_job_id(&job_id) {
        return reply(ctx, error).await;
    }
    ctx.defer_ephemeral().await?;
    match ctx.data().client.cancel_chain_job(&job_id).await {
        Ok(_) => {
            reply(
                ctx,
                format!(
                    "Cancel requested for video {job_id}; check /video status until it settles."
                ),
            )
            .await
        }
        Err(error) => {
            reply(
                ctx,
                format!("Could not confirm cancellation for {job_id}: {error}."),
            )
            .await
        }
    }
}

#[cfg(test)]
#[path = "video_tests.rs"]
mod tests;
