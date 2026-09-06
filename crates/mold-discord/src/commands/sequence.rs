use crate::checks::{self, AuthResult};
use crate::handler;
use crate::state::Context;
use anyhow::{Context as _, Result};
use mold_core::chain::{ChainRequest, ChainStage, TransitionMode, MAX_CHAIN_STAGES};
use mold_core::chain_job::{ChainJobDetail, ChainJobState};
use mold_core::{ModelDefaults, ModelInfoExtended, OutputFormat};
use poise::serenity_prelude::{
    self as serenity, CreateAttachment, CreateEmbed, CreateMessage, EditMessage,
};
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_secs(1);
const EDIT_THROTTLE: Duration = Duration::from_secs(3);
const MAX_ATTACHMENT_BYTES: u64 = 24 * 1024 * 1024;

async fn autocomplete_sequence_model(ctx: Context<'_>, partial: &str) -> Vec<String> {
    let partial = partial.to_lowercase();
    let models = ctx.data().cached_models().await;
    let mut names = if models.is_empty() {
        mold_core::manifest::visible_manifests()
            .filter(|manifest| mold_core::catalog::chain_capable_family(&manifest.family))
            .map(|manifest| manifest.name.clone())
            .collect::<Vec<_>>()
    } else {
        models
            .into_iter()
            .filter(|model| {
                mold_core::catalog::chain_capable_family(&model.info.family)
                    && model.downloaded
                    && model.supports_sequence != Some(false)
            })
            .map(|model| model.info.name)
            .collect::<Vec<_>>()
    };
    names.retain(|name| partial.is_empty() || name.to_lowercase().contains(&partial));
    names.truncate(25);
    names
}

fn parse_prompts(input: &str) -> Result<Vec<String>, String> {
    let prompts = input
        .split('|')
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if prompts.len() < 2 {
        return Err("Sequence requires at least two prompts separated by `|`.".to_string());
    }
    if prompts.len() > MAX_CHAIN_STAGES {
        return Err(format!(
            "Sequence supports at most {MAX_CHAIN_STAGES} prompts."
        ));
    }
    Ok(prompts)
}

fn resolve_default_sequence_model(models: &[ModelInfoExtended]) -> String {
    models
        .iter()
        .filter(|model| {
            mold_core::catalog::chain_capable_family(&model.info.family)
                && model.downloaded
                && model.supports_sequence != Some(false)
        })
        .min_by(|left, right| {
            right
                .info
                .is_loaded
                .cmp(&left.info.is_loaded)
                .then_with(|| {
                    left.info
                        .size_gb
                        .partial_cmp(&right.info.size_gb)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .map(|model| model.info.name.clone())
        .unwrap_or_else(|| "ltx-2-19b-distilled:fp8".to_string())
}

#[derive(Debug)]
struct SequenceParams<'a> {
    prompts: Vec<String>,
    model: String,
    frames_per_clip: u32,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    audio: Option<bool>,
    defaults: Option<&'a ModelDefaults>,
    /// Resolved family. The clip grid, the floor, and the seam are all its
    /// properties, never LTX-2 constants (#783).
    family: Option<&'a str>,
    /// The checkpoint's own conditioning contract. Wan's seam is last-frame
    /// image conditioning, so only an image-conditioned checkpoint carries
    /// anything across a boundary.
    source_image: Option<mold_core::SourceImageCapability>,
}

fn build_sequence_request(params: SequenceParams<'_>) -> Result<ChainRequest, String> {
    // The grid is the family's: wan's VAE compresses time by 4 where the LTX
    // families compress by 8. Hardcoding `8n+1` here rejected wan's own
    // 53-frame routing default before the request was ever submitted.
    let family = params.family.unwrap_or("ltx2");
    let step = mold_core::validation::frame_step_for_family(family).unwrap_or(8);
    let offset = mold_core::validation::frame_offset_for_family(family).unwrap_or(1);
    // LTX-2's 25 is three latent frames under its 8x stride; the equivalent
    // floor scales with the family's own stride rather than carrying over.
    let min_frames = step * 3 + offset;
    if params.frames_per_clip < min_frames || params.frames_per_clip % step != offset % step {
        return Err(format!(
            "Frames per clip must be {min_frames} or greater and satisfy {step}n+{offset}."
        ));
    }
    if let Some(cap) = mold_core::validation::max_frames_for_family_at_fps(
        family,
        params
            .fps
            .or_else(|| params.defaults.and_then(|defaults| defaults.default_fps))
            .unwrap_or(mold_core::validation::LTX2_DEFAULT_FPS),
    ) {
        if params.frames_per_clip > cap {
            return Err(format!(
                "Frames per clip must be {cap} or fewer for this model's family."
            ));
        }
    }
    let (default_width, default_height, default_fps, default_steps, default_guidance) =
        params.defaults.map_or((1216, 704, 24, 8, 3.0), |defaults| {
            (
                defaults.default_width,
                defaults.default_height,
                defaults.default_fps.unwrap_or(24),
                defaults.default_steps,
                defaults.default_guidance,
            )
        });
    let width = params.width.unwrap_or(default_width);
    let height = params.height.unwrap_or(default_height);
    let fps = params.fps.unwrap_or(default_fps);
    let steps = params.steps.unwrap_or(default_steps);
    let guidance = params.guidance.unwrap_or(default_guidance);
    let stages = params
        .prompts
        .into_iter()
        .map(|prompt| ChainStage {
            prompt,
            frames: params.frames_per_clip,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Smooth,
            fade_frames: None,
            model: None,
            loras: Vec::new(),
            references: Vec::new(),
        })
        .collect();
    // The seam a family and checkpoint can actually honour. Sending LTX-2's
    // 17 for wan meant the server silently repaired it, so the bot advertised
    // a boundary it was not submitting.
    let motion_tail_frames = mold_core::validation::chain_motion_tail_frames_for_family(
        family,
        params.source_image,
        mold_core::validation::DEFAULT_EXTEND_OVERLAP_FRAMES,
    );
    Ok(ChainRequest {
        offload: None,
        collection: None,
        tags: None,
        title: None,
        model: params.model,
        stages,
        motion_tail_frames,
        width,
        height,
        fps,
        seed: params.seed,
        steps,
        guidance,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        output_mode: Some(mold_core::GenerationOutputMode::Sequence),
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
        enable_audio: params.audio,
        ephemeral: false,
    })
}

fn progress_description(detail: &ChainJobDetail) -> String {
    match detail.summary.state {
        ChainJobState::Queued => format!(
            "Queued durable job `{}` with {} clips.",
            detail.summary.id, detail.summary.stage_count
        ),
        ChainJobState::Running => format!(
            "Rendering clip {} of {}…",
            detail
                .summary
                .current_stage
                .saturating_add(1)
                .min(detail.summary.stage_count),
            detail.summary.stage_count
        ),
        ChainJobState::Completed => "Finalizing Discord delivery…".to_string(),
        state => format!("Sequence {}.", state.as_str()),
    }
}

async fn send_completed_sequence(
    ctx: Context<'_>,
    message: &mut serenity::Message,
    job_id: &str,
    clip_count: usize,
) -> Result<()> {
    let client = &ctx.data().client;
    let gallery = client.list_gallery().await?;
    let output = gallery
        .into_iter()
        .filter(|item| item.metadata.chain_job_id.as_deref() == Some(job_id))
        .max_by_key(|item| item.timestamp)
        .with_context(|| format!("completed sequence {job_id} has no gallery output"))?;
    let library_url = format!("{}/library?print={}", client.host(), output.filename);
    let mut embed = CreateEmbed::new()
        .title("Sequence complete")
        .description(format!(
            "Rendered {clip_count} clips. [Open in the Mold Library]({library_url})"
        ))
        .color(0x57F287);
    let mut edit = EditMessage::new();
    if output
        .size_bytes
        .is_some_and(|size| size > MAX_ATTACHMENT_BYTES)
    {
        embed = embed.footer(poise::serenity_prelude::CreateEmbedFooter::new(
            "The MP4 exceeds Discord's upload limit; use the Library link.",
        ));
    } else {
        let bytes = client.get_gallery_image(&output.filename).await?;
        if bytes.len() as u64 <= MAX_ATTACHMENT_BYTES {
            edit = edit.new_attachment(CreateAttachment::bytes(bytes, output.filename));
        } else {
            embed = embed.footer(poise::serenity_prelude::CreateEmbedFooter::new(
                "The MP4 exceeds Discord's upload limit; use the Library link.",
            ));
        }
    }
    message
        .edit(ctx.serenity_context(), edit.embed(embed))
        .await?;
    Ok(())
}

/// Render a durable multi-prompt video sequence.
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn sequence(
    ctx: Context<'_>,
    #[description = "Clip prompts separated by | (2–16 prompts)"] prompts: String,
    #[description = "Sequence-capable model (LTX-2, LTX Video, or Wan Video)"]
    #[autocomplete = "autocomplete_sequence_model"]
    model: Option<String>,
    #[description = "Frames per clip; must sit on the model family's frame grid"]
    frames_per_clip: Option<u32>,
    #[description = "Output width"] width: Option<u32>,
    #[description = "Output height"] height: Option<u32>,
    #[description = "Output FPS (default 24)"] fps: Option<u32>,
    #[description = "Inference steps"] steps: Option<u32>,
    #[description = "Guidance scale"] guidance: Option<f64>,
    #[description = "Base seed"] seed: Option<u64>,
    #[description = "Synchronized audio — on by default where supported"] audio: Option<bool>,
) -> Result<()> {
    let prompts = match parse_prompts(&prompts) {
        Ok(prompts) => prompts,
        Err(message) => {
            ctx.send(
                poise::CreateReply::default()
                    .content(message)
                    .ephemeral(true),
            )
            .await?;
            return Ok(());
        }
    };
    let user_id = ctx.author().id.get();
    if let AuthResult::Denied(message) = checks::check_generate_auth(&ctx).await {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    // The interaction token expires after 15 minutes, while a durable
    // sequence can legitimately run longer. Use it only for a private handoff;
    // all long-lived progress and delivery use a normal bot-authored message.
    ctx.defer_ephemeral().await?;
    let models = ctx.data().cached_models().await;
    let model = model.unwrap_or_else(|| resolve_default_sequence_model(&models));
    let model_entry = models.iter().find(|entry| entry.info.name == model);
    if model_entry.is_some_and(|entry| {
        !mold_core::catalog::chain_capable_family(&entry.info.family)
            || entry.supports_sequence == Some(false)
    }) {
        ctx.data().quotas.refund(user_id);
        handler::send_error(
            ctx,
            "Sequence requires a model whose family renders sequence clips \
             (LTX-2, LTX Video, or Wan Video) with sequence support advertised.",
        )
        .await?;
        return Ok(());
    }
    let family = model_entry.map(|entry| entry.info.family.as_str());
    // A wan I2V checkpoint cannot render stage 0 without an image, and this
    // command has no attachment to give it. Refuse before spending the
    // model load rather than after (#783).
    if model_entry
        .is_some_and(|entry| entry.source_image == Some(mold_core::SourceImageCapability::Required))
    {
        ctx.data().quotas.refund(user_id);
        handler::send_error(
            ctx,
            "This checkpoint requires a source image for every clip, and `/sequence` \
             has no image input. Pick a text-to-video or optionally-conditioned model.",
        )
        .await?;
        return Ok(());
    }
    // The default clip length is the family's own, not LTX-2's 97: 97 is off
    // wan's `4k+1` grid only by luck of arithmetic, and its routing envelope
    // is smaller than its cap.
    let default_clip_frames = model_entry
        .and_then(|entry| entry.defaults.default_frames)
        .unwrap_or(97);
    let request = match build_sequence_request(SequenceParams {
        prompts,
        model,
        frames_per_clip: frames_per_clip.unwrap_or(default_clip_frames),
        width,
        height,
        fps,
        steps,
        guidance,
        seed,
        audio,
        defaults: model_entry.map(|entry| &entry.defaults),
        family,
        source_image: model_entry.and_then(|entry| entry.source_image),
    }) {
        Ok(request) => request,
        Err(message) => {
            ctx.data().quotas.refund(user_id);
            handler::send_error(ctx, &message).await?;
            return Ok(());
        }
    };
    let clip_count = request.stages.len();
    let created = match ctx.data().client.create_chain_job(&request).await {
        Ok(created) => created,
        Err(error) => {
            ctx.data().quotas.refund(user_id);
            handler::send_error(ctx, &format!("Sequence failed: {error:#}")).await?;
            return Ok(());
        }
    };
    let mut message = match ctx
        .channel_id()
        .send_message(
            ctx.serenity_context(),
            CreateMessage::new().embed(
                CreateEmbed::new()
                    .title("Rendering sequence…")
                    .description(format!("Queued durable job `{}`.", created.job_id))
                    .color(0x5865F2),
            ),
        )
        .await
    {
        Ok(message) => message,
        Err(error) => {
            ctx.data().cooldowns.record(user_id);
            handler::send_error(
                ctx,
                &format!(
                    "Sequence `{}` is queued, but I could not post its progress message: {error}",
                    created.job_id
                ),
            )
            .await?;
            return Ok(());
        }
    };
    ctx.send(
        poise::CreateReply::default()
            .content(format!(
                "Sequence `{}` is queued; progress will continue in the channel.",
                created.job_id
            ))
            .ephemeral(true),
    )
    .await?;
    let result = async {
        let mut last_edit = Instant::now() - EDIT_THROTTLE;
        let mut last_status = String::new();
        loop {
            let detail = ctx.data().client.get_chain_job(&created.job_id).await?;
            let status = progress_description(&detail);
            if status != last_status && last_edit.elapsed() >= EDIT_THROTTLE {
                let _ = message
                    .edit(
                        ctx.serenity_context(),
                        EditMessage::new().embed(
                            CreateEmbed::new()
                                .title("Rendering sequence…")
                                .description(&status)
                                .color(0x5865F2),
                        ),
                    )
                    .await;
                last_status = status;
                last_edit = Instant::now();
            }
            match detail.summary.state {
                ChainJobState::Completed => {
                    send_completed_sequence(ctx, &mut message, &created.job_id, clip_count).await?;
                    return Ok::<(), anyhow::Error>(());
                }
                ChainJobState::Failed | ChainJobState::Cancelled | ChainJobState::Interrupted => {
                    anyhow::bail!(
                        "sequence {}: {}",
                        detail.summary.state.as_str(),
                        detail
                            .summary
                            .error
                            .unwrap_or_else(|| "no server detail".to_string())
                    );
                }
                ChainJobState::Queued | ChainJobState::Paused | ChainJobState::Running => {}
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }
    .await;
    match result {
        Ok(()) => ctx.data().cooldowns.record(user_id),
        Err(error) => {
            ctx.data().quotas.refund(user_id);
            let _ = message
                .edit(
                    ctx.serenity_context(),
                    EditMessage::new().embed(
                        CreateEmbed::new()
                            .title("Sequence failed")
                            .description(format!("{error:#}"))
                            .color(0xED4245),
                    ),
                )
                .await;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_pipe_delimited_and_trimmed() {
        assert_eq!(
            parse_prompts(" first shot | second shot | third shot ").unwrap(),
            ["first shot", "second shot", "third shot"]
        );
        assert!(parse_prompts("one prompt").is_err());
    }

    /// Discord builds a wan sequence on wan's own grid and seam (#783).
    ///
    /// Eligibility was de-`ltx2`-ed in #936, but request construction stayed
    /// LTX-shaped: an `8n+1` grid with a minimum of 25, and a motion tail of
    /// 17. Wan is `4k+1`, so **53** — the A14B routing default the CLI itself
    /// picks — was rejected here before the request was ever submitted, while
    /// 81/97/121 passed only by coincidence. The tail was repaired silently
    /// by the server, so the bot advertised a seam it was not sending.
    #[test]
    fn a_wan_sequence_uses_wans_grid_and_seam() {
        let wan = |frames_per_clip: u32, source_image| {
            build_sequence_request(SequenceParams {
                prompts: vec!["a paper boat".into(), "it reaches the drain".into()],
                model: "wan22-ti2v-5b:q8".into(),
                frames_per_clip,
                width: None,
                height: None,
                fps: None,
                steps: None,
                guidance: None,
                seed: None,
                audio: None,
                defaults: None,
                family: Some("wan"),
                source_image,
            })
        };

        // The A14B routing default. Rejected outright before this.
        let request = wan(53, Some(mold_core::SourceImageCapability::Optional))
            .expect("53 is on wan's 4k+1 grid");
        assert_eq!(request.stages[0].frames, 53);
        assert_eq!(
            request.motion_tail_frames,
            mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES,
            "wan re-renders exactly the frame the continuation was seeded with"
        );

        // A text-to-video checkpoint cannot be seeded, so its seam carries
        // nothing rather than a value the server has to repair.
        let t2v = wan(53, Some(mold_core::SourceImageCapability::Unsupported)).unwrap();
        assert_eq!(t2v.motion_tail_frames, 0);

        // Off-grid for wan is still refused, and the message names wan's grid.
        let error = wan(50, Some(mold_core::SourceImageCapability::Optional)).unwrap_err();
        assert!(error.contains("4n+1"), "got: {error}");

        // Below wan's own floor, which is not LTX-2's 25.
        assert!(wan(1, Some(mold_core::SourceImageCapability::Optional)).is_err());

        // LTX-2 is unchanged: 8n+1, minimum 25, tail 17.
        let ltx2 = build_sequence_request(SequenceParams {
            prompts: vec!["one".into(), "two".into()],
            model: "ltx-2-19b-distilled:fp8".into(),
            frames_per_clip: 97,
            width: None,
            height: None,
            fps: None,
            steps: None,
            guidance: None,
            seed: None,
            audio: None,
            defaults: None,
            family: Some("ltx2"),
            source_image: None,
        })
        .unwrap();
        assert_eq!(ltx2.motion_tail_frames, 17);
        assert!(build_sequence_request(SequenceParams {
            prompts: vec!["one".into(), "two".into()],
            model: "ltx-2-19b-distilled:fp8".into(),
            frames_per_clip: 53,
            width: None,
            height: None,
            fps: None,
            steps: None,
            guidance: None,
            seed: None,
            audio: None,
            defaults: None,
            family: Some("ltx2"),
            source_image: None,
        })
        .is_err());
    }

    #[test]
    fn request_has_one_smooth_stage_per_prompt() {
        let request = build_sequence_request(SequenceParams {
            prompts: vec!["one".into(), "two".into()],
            model: "ltx-2-19b-distilled:fp8".into(),
            frames_per_clip: 97,
            width: None,
            height: None,
            fps: None,
            steps: None,
            guidance: None,
            seed: Some(42),
            audio: Some(true),
            defaults: None,
            family: Some("ltx2"),
            source_image: None,
        })
        .unwrap();
        assert_eq!(request.stages.len(), 2);
        assert!(request
            .stages
            .iter()
            .all(|stage| stage.frames == 97 && stage.transition == TransitionMode::Smooth));
        assert_eq!(request.motion_tail_frames, 17);
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.enable_audio, Some(true));
        assert_eq!(request.output_format, OutputFormat::Mp4);
    }

    #[test]
    fn rejects_invalid_clip_frame_grid() {
        let result = build_sequence_request(SequenceParams {
            prompts: vec!["one".into(), "two".into()],
            model: "ltx-2-19b-distilled:fp8".into(),
            frames_per_clip: 96,
            width: None,
            height: None,
            fps: None,
            steps: None,
            guidance: None,
            seed: None,
            audio: None,
            defaults: None,
            family: Some("ltx2"),
            source_image: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn sequence_stays_within_discords_option_limit() {
        let command = sequence();
        assert_eq!(command.parameters.len(), 10);
        assert!(command.parameters.len() <= 25);
    }
}
