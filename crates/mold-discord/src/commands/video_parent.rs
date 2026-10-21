//! Single-prompt duration videos; only the server plans bounded clip parts.

use super::{operation_id, GenerationCharge};
use crate::state::Context;
use anyhow::{ensure, Context as _, Result};
use mold_core::video_job::{VideoJob, VideoJobRequest, VideoJobScene, VideoJobSource};
use mold_core::{ModelInfoExtended, OutputFormat, ServerCapabilities};

/// Receipts are copied back verbatim. Disable mentions at delivery instead of
/// injecting invisible characters into the installation-bound identifier.
fn receipt_message(message: &str) -> String {
    message
        .lines()
        .map(super::super::operations::safe_text)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("@\u{200b}", "@")
        .chars()
        .take(1900)
        .collect()
}
fn receipt_reply(message: &str) -> poise::CreateReply {
    poise::CreateReply::default()
        .content(receipt_message(message))
        .ephemeral(true)
        .allowed_mentions(
            poise::serenity_prelude::CreateAllowedMentions::new()
                .empty_users()
                .empty_roles()
                .everyone(false)
                .replied_user(false),
        )
}
pub(super) async fn reply(ctx: Context<'_>, message: impl std::fmt::Display) -> Result<()> {
    ctx.send(receipt_reply(&message.to_string())).await?;
    Ok(())
}

pub(super) fn require_parent_capability(caps: &ServerCapabilities, extending: bool) -> Result<()> {
    let caps = caps
        .video_jobs
        .as_ref()
        .context("Update this machine before making duration videos.")?;
    ensure!(
        caps.supported && caps.literal_prompts,
        "Update this machine before making duration videos."
    );
    ensure!(
        !extending || caps.library_extend,
        "Update this machine before extending Library videos."
    );
    Ok(())
}

pub(super) fn is_video_model(model: &ModelInfoExtended) -> bool {
    model
        .generation_profile
        .as_ref()
        .and_then(|p| p.default_recipe())
        .map_or(
            model.defaults.default_frames.is_some() || model.modality.as_deref() == Some("video"),
            |recipe| recipe.temporal.is_some(),
        )
}

fn parse_seconds(value: &str) -> Result<u64> {
    let value = value.trim();
    let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        !seconds.is_empty()
            && seconds.bytes().all(|b| b.is_ascii_digit())
            && (!value.contains('.') || !fraction.is_empty())
            && fraction.len() <= 3
            && fraction.bytes().all(|b| b.is_ascii_digit()),
        "Duration must be positive seconds with up to three decimal places."
    );
    let whole: u64 = seconds.parse().context("Duration is too large.")?;
    let fraction: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<u64>()? * 10u64.pow(3 - fraction.len() as u32)
    };
    let ms = whole
        .checked_mul(1000)
        .and_then(|n| n.checked_add(fraction))
        .context("Duration is too large.")?;
    ensure!(ms > 0, "Duration must be positive.");
    Ok(ms)
}
fn recipe_timing(model: &ModelInfoExtended) -> (Option<u32>, Option<u32>) {
    let recipe = model
        .generation_profile
        .as_ref()
        .and_then(|p| p.default_recipe());
    (
        recipe
            .and_then(|r| r.defaults.frames)
            .or(model.defaults.default_frames),
        recipe
            .and_then(|r| r.defaults.fps)
            .or(model.defaults.default_fps),
    )
}
fn default_duration(model: &ModelInfoExtended) -> Result<u64> {
    let (frames, fps) = recipe_timing(model);
    let frames = frames.context(
        "This recipe does not advertise a short video length. Supply duration in seconds.",
    )?;
    let fps = fps.context("This recipe does not advertise a frame rate.")?;
    ensure!(
        fps > 0 && frames > 0,
        "This recipe has invalid video timing."
    );
    Ok((u64::from(frames) * 1000 / u64::from(fps)).max(1))
}
fn requested_duration(model: &ModelInfoExtended, duration: Option<&str>) -> Result<u64> {
    match duration {
        Some(value) => parse_seconds(value),
        None => default_duration(model),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_parent_request(
    model: &ModelInfoExtended,
    prompt: &str,
    next: Option<&str>,
    duration: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    image: Option<Vec<u8>>,
    audio: Option<bool>,
) -> Result<VideoJobRequest> {
    ensure!(
        model.downloaded
            && model.runtime_available != Some(false)
            && model.runtime_ready != Some(false),
        "Choose an installed, runnable video model."
    );
    ensure!(
        is_video_model(model),
        "Choose a video recipe advertised by this machine."
    );
    ensure!(!prompt.trim().is_empty(), "Write a video prompt.");
    let duration_ms = requested_duration(model, duration)?;
    let (frames, fps) = recipe_timing(model);
    let mut generation =
        super::super::generate::build_generate_request(super::super::generate::BuildParams {
            prompt,
            model: &model.info.name,
            family: Some(&model.info.family),
            width,
            height,
            defaults: Some(&model.defaults),
            frames,
            fps,
            ..Default::default()
        });
    generation.frames = frames;
    generation.fps = fps;
    generation.output_format = Some(OutputFormat::Mp4);
    generation.batch_size = 1;
    generation.enable_audio = audio.or_else(|| {
        model
            .generation_profile
            .as_ref()
            .and_then(|p| p.default_recipe())
            .map(|r| r.capabilities.supports_audio)
            .or(model.supports_audio)
    });
    generation.save_to_gallery = Some(true);
    generation.embed_metadata = Some(true);
    if let Some(bytes) = image {
        ensure!(
            bytes.len() <= 16 * 1024 * 1024,
            "Image attachment must be at most 16 MiB."
        );
        if matches!(
            super::super::generate::route_references(
                Some(model),
                Some(&model.info.family),
                &model.info.name
            ),
            super::super::generate::ReferenceRoute::H3Ref2va
        ) {
            let format = image::guess_format(&bytes)?;
            ensure!(
                matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg),
                "Use a PNG or JPEG reference image."
            );
            let (width, height) = image::ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()?
                .into_dimensions()?;
            generation.references = Some(vec![mold_core::GenerationReference::Image {
                media: mold_core::GenerationReferenceAuthority::Inline { data: bytes },
                provenance: Default::default(),
                mime_type: if format == image::ImageFormat::Png {
                    "image/png"
                } else {
                    "image/jpeg"
                }
                .into(),
                width,
                height,
            }]);
        } else {
            generation.source_image = Some(bytes);
        }
    }
    let scenes = next
        .map(|next| -> Result<_> {
            ensure!(
                !next.trim().is_empty(),
                "Write the second scene prompt or omit it."
            );
            ensure!(
                duration_ms >= 2,
                "Two scenes need at least two milliseconds in total."
            );
            let first = duration_ms / 2;
            Ok(vec![
                VideoJobScene {
                    prompt: prompt.into(),
                    duration_ms: first,
                },
                VideoJobScene {
                    prompt: next.into(),
                    duration_ms: duration_ms - first,
                },
            ])
        })
        .transpose()?;
    let request = VideoJobRequest {
        generation,
        duration_ms,
        source: None,
        scenes,
        validation_revision: None,
    };
    request.validate_shape().map_err(anyhow::Error::msg)?;
    Ok(request)
}

fn uuid_shaped(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}
fn video_receipt(id: &str, instance: &str) -> Result<String> {
    ensure!(
        uuid_shaped(id) && uuid_shaped(instance),
        "The machine did not provide a valid video installation identity."
    );
    Ok(format!("{id}@{instance}"))
}
fn receipt_parts(value: &str) -> Result<(&str, &str)> {
    let (id, instance) = value
        .split_once('@')
        .context("Copy the whole Video ID, including @ and its installation identity.")?;
    ensure!(
        uuid_shaped(id) && uuid_shaped(instance),
        "Copy the whole Video ID: job UUID@installation UUID."
    );
    Ok((id, instance))
}
fn validate_receipt_owner<'a>(receipt: &'a str, current: &str) -> Result<(&'a str, &'a str)> {
    let pair = receipt_parts(receipt)?;
    ensure!(pair.1==current,"This machine has been replaced. Reconnect the original installation to control this Video ID.");
    Ok(pair)
}
pub(super) async fn current_instance(ctx: Context<'_>, extending: bool) -> Result<String> {
    require_parent_capability(&ctx.data().client.capabilities().await?, extending)?;
    let instance = ctx
        .data()
        .client
        .server_status()
        .await?
        .instance_id
        .context("Reconnect this machine before making videos.")?;
    ensure!(
        uuid_shaped(&instance),
        "The machine did not provide a valid installation identity."
    );
    Ok(instance)
}
pub(super) async fn receipt_owner(ctx: Context<'_>, receipt: &str) -> Result<(String, String)> {
    let instance = current_instance(ctx, false).await?;
    let (id, owner) = validate_receipt_owner(receipt, &instance)?;
    Ok((id.into(), owner.into()))
}
async fn admit_after_receipt<T, F, Fut>(
    receipt: impl std::future::Future<Output = Result<()>>,
    admit: F,
) -> Result<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    receipt.await?;
    admit().await
}
fn seconds(ms: u64) -> String {
    format!("{}.{:03}", ms / 1000, ms % 1000)
}
pub(super) fn status_message(receipt: &str, job: &VideoJob) -> String {
    let result = match job.output_filename.as_deref() {
        Some(file) => format!("\nLibrary video: {file}"),
        None if job.state == mold_core::video_job::VideoJobState::Completed => {
            "\nFinished without a published Library video. Check the server output settings.".into()
        }
        None => String::new(),
    };
    format!(
        "Video ID: {receipt}\n{:?} · {} of {} clips finished · {} seconds total{}{}",
        job.state,
        job.completed_clips,
        job.plan.clip_count,
        seconds(job.plan.total_duration_ms),
        result,
        job.error
            .as_ref()
            .map(|e| format!("\n{e}"))
            .unwrap_or_default()
    )
}
pub(super) async fn enqueue(
    ctx: Context<'_>,
    mut request: VideoJobRequest,
    instance: String,
    mut charge: GenerationCharge<'_>,
) -> Result<()> {
    let plan = match ctx
        .data()
        .client
        .validate_video_job(&request, &instance)
        .await
    {
        Ok(plan) => plan,
        Err(error) => return reply(ctx, format!("Video validation failed: {error}")).await,
    };
    ensure!(
        !plan.validation_revision.is_empty(),
        "The machine returned no validation proof."
    );
    request.validation_revision = Some(plan.validation_revision.clone());
    let id = operation_id(ctx.id());
    let receipt = video_receipt(&id, &instance)?;
    let clips = if plan.clip_count == 1 {
        "1 clip".into()
    } else {
        format!("{} clips joined into one video", plan.clip_count)
    };
    let warnings = if plan.warnings.is_empty() {
        String::new()
    } else {
        format!("\n{}", plan.warnings.join("\n"))
    };
    let notice=format!("Video ID: {receipt}\n{clips} · {} seconds total.\nKeep this ID for /video status, resume or cancel. Manage Server permission is needed to control host-wide work.{warnings}",seconds(plan.total_duration_ms));
    let result = admit_after_receipt(reply(ctx, notice), || async {
        charge.attempting_admission();
        ctx.data()
            .client
            .create_video_job(&request, &id, &instance)
            .await
    })
    .await;
    match result{
        Ok(job)=>reply(ctx,format!("Video queued. {}",status_message(&receipt,&job))).await,
        Err(error) if charge.refundable=>Err(error),
        Err(error)=>reply(ctx,format!("Video ID: {receipt}\nAcceptance is unknown. Check /video status job_id:{receipt}; do not repeat make or extend.\n{error}")).await
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) async fn prepare_make(
    ctx: Context<'_>,
    prompt: &str,
    next: Option<&str>,
    duration: Option<&str>,
    model: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    image: Option<poise::serenity_prelude::Attachment>,
    audio: Option<bool>,
) -> Result<(VideoJobRequest, String)> {
    let instance = current_instance(ctx, false).await?;
    let model = super::live_model(ctx, model).await?;
    let image = match image {
        Some(image) => Some(
            super::super::generate::fetch_source_image(&image)
                .await
                .map_err(anyhow::Error::msg)?,
        ),
        None => None,
    };
    Ok((
        make_parent_request(&model, prompt, next, duration, width, height, image, audio)?,
        instance,
    ))
}
pub(super) async fn prepare_extension(
    ctx: Context<'_>,
    filename: &str,
    prompt: &str,
    duration: Option<&str>,
) -> Result<(VideoJobRequest, String)> {
    let instance = current_instance(ctx, true).await?;
    let source = ctx
        .data()
        .client
        .gallery_video_source(filename)
        .await?
        .context("Choose an existing video from this machine's Library.")?;
    let model = super::live_model(ctx, Some(source.metadata.model.clone())).await?;
    Ok((
        extension_request(&source, filename, prompt, duration, &model)?,
        instance,
    ))
}
fn extension_request(
    source: &mold_core::GalleryImage,
    filename: &str,
    prompt: &str,
    duration: Option<&str>,
    model: &ModelInfoExtended,
) -> Result<VideoJobRequest> {
    ensure!(
        !prompt.trim().is_empty(),
        "Write a prompt for the added footage."
    );
    ensure!(
        source.filename == filename
            && source.trashed_at.is_none()
            && (source.format == Some(OutputFormat::Mp4)
                || (source.format.is_none() && filename.to_ascii_lowercase().ends_with(".mp4"))),
        "Choose an active MP4 from this machine's Library."
    );
    ensure!(!source.metadata_synthetic,"This older print has no recoverable recipe. Open Extend in an app to choose a video model.");
    let generation = mold_core::video_extension::extension_generation(&source.metadata, prompt)
        .map_err(anyhow::Error::msg)?;
    let duration_ms = match duration {
        Some(value) => parse_seconds(value)?,
        None => {
            let (frames, default_fps) = recipe_timing(model);
            let frames =
                frames.context("Supply the seconds to add; this recipe has no default length.")?;
            let fps = generation
                .fps
                .or(default_fps)
                .context("This recipe has no frame rate.")?;
            ensure!(
                frames > 0 && fps > 0,
                "This recipe has invalid video timing."
            );
            (u64::from(frames) * 1000 / u64::from(fps)).max(1)
        }
    };
    let request = VideoJobRequest {
        generation,
        duration_ms,
        source: Some(VideoJobSource {
            filename: filename.into(),
            expected_archive_identity_sha256: None,
            retained_upload: None,
        }),
        scenes: None,
        validation_revision: None,
    };
    request.validate_shape().map_err(anyhow::Error::msg)?;
    Ok(request)
}

#[cfg(test)]
#[path = "video_parent_tests.rs"]
mod tests;
