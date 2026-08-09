use crate::checks::{self, AuthResult};
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::{
    GenerateRequest, KeyframeCondition, Ltx2PipelineMode, ModelInfoExtended, OutputFormat,
    TimeRange,
};
use poise::serenity_prelude as serenity;
use std::time::Duration;

/// Hard ceiling on how long we're willing to spend computing autocomplete
/// choices. Discord drops the interaction after 3 s and shows "Loading options
/// failed" — we stay well under that.
const AUTOCOMPLETE_BUDGET: Duration = Duration::from_millis(1500);

/// Pick the best set of model names to suggest for the given partial input.
///
/// Pure function so we can test the ranking logic without a live bot. The
/// autocomplete fn wraps this and adds the Discord-plumbing + timeout guard.
pub fn rank_model_suggestions(
    cached: &[ModelInfoExtended],
    fallback_names: &[&str],
    partial: &str,
) -> Vec<String> {
    let lower = partial.to_lowercase();
    let matches_partial =
        |name: &str| -> bool { lower.is_empty() || name.to_lowercase().contains(&lower) };

    // When we have a populated cache, prefer downloaded models and fall back
    // to undownloaded (they still work — the server auto-pulls on demand).
    if !cached.is_empty() {
        let mut downloaded: Vec<String> = cached
            .iter()
            .filter(|m| m.downloaded && matches_partial(&m.info.name))
            .map(|m| m.info.name.clone())
            .collect();
        if downloaded.len() < 25 {
            let room = 25 - downloaded.len();
            let also: Vec<String> = cached
                .iter()
                .filter(|m| !m.downloaded && matches_partial(&m.info.name))
                .take(room)
                .map(|m| m.info.name.clone())
                .collect();
            downloaded.extend(also);
        }
        downloaded.truncate(25);
        return downloaded;
    }

    // Cold cache (bot just started / server unreachable): offer manifest
    // names so the dropdown still renders instead of "Loading options failed".
    fallback_names
        .iter()
        .filter(|n| matches_partial(n))
        .take(25)
        .map(|n| n.to_string())
        .collect()
}

fn manifest_fallback_names() -> Vec<String> {
    mold_core::manifest::visible_manifests()
        .filter(|m| !mold_core::manifest::UTILITY_FAMILIES.contains(&m.family.as_str()))
        .map(|m| m.name.clone())
        .collect()
}

fn defaults_from_manifest(
    manifest: &mold_core::manifest::ModelManifest,
) -> mold_core::ModelDefaults {
    mold_core::ModelDefaults {
        default_steps: manifest.defaults.steps,
        default_guidance: manifest.defaults.guidance,
        default_width: manifest.defaults.width,
        default_height: manifest.defaults.height,
        description: manifest.description.clone(),
        default_frames: manifest.defaults.frames,
        default_fps: manifest.defaults.fps,
        max_frames: mold_core::validation::max_frames_for_family_at_fps(
            &manifest.family,
            manifest.defaults.fps.unwrap_or(24),
        ),
        max_runtime_seconds: mold_core::validation::max_runtime_seconds_for_family(
            &manifest.family,
        ),
        max_frames_absolute: mold_core::validation::max_frames_absolute_for_family(
            &manifest.family,
        ),
        frame_step: mold_core::validation::frame_step_for_family(&manifest.family),
        frame_offset: mold_core::validation::frame_offset_for_family(&manifest.family),
        // Same per-model grid `/api/models` would advertise, so the
        // cold-cache fallback fits sources on the checkpoint's real grid.
        dimension_alignment: Some(mold_core::dimension_alignment_for_model(
            &manifest.name,
            Some(&manifest.family),
        )),
        ..Default::default()
    }
}

/// Autocomplete function for model names. Reads the cached model list without
/// blocking on network I/O; the cache is refreshed by a background task so the
/// hot path here is always a quick `RwLock::read`. Falls back to the static
/// manifest if the cache is still cold so users never see "Loading options
/// failed" due to a slow first fetch.
async fn autocomplete_model(ctx: Context<'_>, partial: &str) -> Vec<String> {
    let partial = partial.to_string();
    let data = ctx.data();
    let work = async {
        let cached = data.cached_models().await;
        let fallback = manifest_fallback_names();
        let fallback_refs: Vec<&str> = fallback.iter().map(String::as_str).collect();
        rank_model_suggestions(&cached, &fallback_refs, &partial)
    };
    match tokio::time::timeout(AUTOCOMPLETE_BUDGET, work).await {
        Ok(v) => v,
        Err(_) => manifest_fallback_names().into_iter().take(25).collect(),
    }
}

/// Discord-selectable container for video generations. Videos are always
/// returned as either MP4 or animated GIF — nothing else is delivered through
/// the bot.
#[derive(Debug, Clone, Copy, poise::ChoiceParameter)]
pub enum VideoFormat {
    #[name = "MP4 (default)"]
    Mp4,
    #[name = "Animated GIF"]
    Gif,
}

impl VideoFormat {
    fn to_output_format(self) -> OutputFormat {
        match self {
            VideoFormat::Mp4 => OutputFormat::Mp4,
            VideoFormat::Gif => OutputFormat::Gif,
        }
    }
}

/// Slash-command facing selector for the ordinary LTX-2 pipelines. Specialized
/// attachment-driven modes (`a2-vid`, `retake`, and `keyframe`) are selected
/// automatically from their required inputs instead of appearing here.
#[derive(Debug, Clone, Copy, poise::ChoiceParameter)]
pub enum PipelineChoice {
    #[name = "one-stage"]
    OneStage,
    #[name = "two-stage"]
    TwoStage,
    #[name = "two-stage-hq"]
    TwoStageHq,
    #[name = "distilled"]
    Distilled,
}

impl PipelineChoice {
    fn to_mode(self) -> Ltx2PipelineMode {
        match self {
            PipelineChoice::OneStage => Ltx2PipelineMode::OneStage,
            PipelineChoice::TwoStage => Ltx2PipelineMode::TwoStage,
            PipelineChoice::TwoStageHq => Ltx2PipelineMode::TwoStageHq,
            PipelineChoice::Distilled => Ltx2PipelineMode::Distilled,
        }
    }
}

/// Classify a model family without making it discoverable or runnable.
///
/// H3 remains absent from the visible model catalog while its runtime and
/// compliance gates are closed. Keeping its timing/container semantics here
/// means an authorized future server can advertise it without Discord falling
/// back to still-image defaults.
pub fn video_family(family: &str) -> Option<&str> {
    if mold_core::minimax_h3::is_family(family) {
        Some(mold_core::minimax_h3::FAMILY)
    } else {
        match family {
            "ltx-video" | "ltx2" | "wan" => Some(family),
            _ => None,
        }
    }
}

/// Look up the family for a resolved model name from the bot's cached models.
pub fn family_for_model<'a>(models: &'a [ModelInfoExtended], name: &str) -> Option<&'a str> {
    models
        .iter()
        .find(|m| m.info.name == name)
        .map(|m| m.info.family.as_str())
}

/// Reject a request the selected checkpoint's source-image contract (#772)
/// already refuses, before it costs a queue slot, a UMT5 encode, and an
/// expert load. Delegates to the shared `mold-core` builder so the wording is
/// identical wherever the rejection lands — and family-aware, so a plain
/// LTX-Video refusal names LTX-Video, never Wan.
///
/// A `None` capability — an older server, or a checkpoint neither the server
/// nor the built-in manifests could classify — enforces nothing: the engine
/// remains the authority and its late error is no worse than today's
/// behavior.
///
/// `has_source` counts first/last-frame keyframes as well as a source image
/// (#779), matching admission: both carry source frames, so either satisfies
/// a required contract and either is refused by a T2V-only checkpoint.
fn source_image_contract_error(
    family: Option<&str>,
    model: &str,
    capability: Option<mold_core::SourceImageCapability>,
    has_source: bool,
) -> Option<String> {
    mold_core::validation::source_image_contract_violation(family, model, capability, has_source)
}

/// Resolve the checkpoint's source-image contract: the server's advertised
/// classification first, then the built-in manifest, which answers both while
/// the model cache is cold and against a server too old to advertise the
/// field at all. A manifest tier's contract is structural, so it is as true
/// of an old server's engine as of a new server's admission gate.
fn resolve_source_image_contract(
    model_entry: Option<&ModelInfoExtended>,
    model: &str,
) -> Option<mold_core::SourceImageCapability> {
    model_entry
        .and_then(|entry| entry.source_image)
        .or_else(|| {
            mold_core::manifest::find_manifest(model)
                .and_then(|manifest| manifest.defaults.source_image)
        })
}

/// Parameters collected from the Discord slash command before they are turned
/// into a concrete `GenerateRequest`. Keeps `build_generate_request` signature
/// small and makes test fixtures explicit.
#[derive(Debug, Clone, Default)]
pub struct BuildParams<'a> {
    pub prompt: &'a str,
    pub model: &'a str,
    pub family: Option<&'a str>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<&'a str>,
    pub defaults: Option<&'a mold_core::ModelDefaults>,
    pub source_image: Option<Vec<u8>>,
    /// Display-safe source attachment basename; never a Discord URL or path.
    pub source_image_name: Option<String>,
    pub references: Option<Vec<mold_core::GenerationReference>>,
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub strength: Option<f64>,
    pub video_format: Option<VideoFormat>,
    pub audio: Option<bool>,
    pub pipeline: Option<Ltx2PipelineMode>,
    pub audio_file: Option<Vec<u8>>,
    pub source_video: Option<Vec<u8>>,
    pub keyframes: Option<Vec<KeyframeCondition>>,
    pub retake_range: Option<TimeRange>,
}

/// Resolve Discord's human-readable duration sugar onto the selected model's
/// concrete frame grid. Explicit `frames` remains the advanced escape hatch;
/// it is deliberately not normalized here so the server can reject mistakes
/// rather than silently changing a precise request.
fn resolve_video_timing(
    family: Option<&str>,
    defaults: Option<&mold_core::ModelDefaults>,
    frames: Option<u32>,
    fps: Option<u32>,
    duration_seconds: Option<f64>,
) -> Result<(Option<u32>, Option<u32>), String> {
    if frames.is_some() && duration_seconds.is_some() {
        return Err("Use either duration or frames, not both.".to_string());
    }

    let Some(family) = family.and_then(video_family) else {
        if duration_seconds.is_some() {
            return Err("Duration is only available for a recognized video model.".to_string());
        }
        return Ok((frames, fps));
    };

    let is_h3 = mold_core::minimax_h3::is_family(family);
    if is_h3 && fps.is_some_and(|value| value != mold_core::minimax_h3::FIXED_FPS) {
        return Err(format!(
            "MiniMax H3 requires {} FPS.",
            mold_core::minimax_h3::FIXED_FPS
        ));
    }

    let resolved_fps = if is_h3 {
        mold_core::minimax_h3::FIXED_FPS
    } else {
        fps.or_else(|| defaults.and_then(|value| value.default_fps))
            .unwrap_or(24)
    };
    if resolved_fps == 0 {
        return Err("Video FPS must be at least 1.".to_string());
    }

    let Some(seconds) = duration_seconds else {
        let resolved_frames = frames
            .or_else(|| defaults.and_then(|value| value.default_frames))
            .unwrap_or(25);
        return Ok((Some(resolved_frames), Some(resolved_fps)));
    };
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("Duration must be a positive number of seconds.".to_string());
    }
    if is_h3 && seconds < f64::from(mold_core::minimax_h3::MIN_DURATION_SECONDS) {
        return Err(format!(
            "MiniMax H3 duration must be at least {} seconds.",
            mold_core::minimax_h3::MIN_DURATION_SECONDS
        ));
    }

    let max_runtime_seconds = defaults
        .and_then(|value| value.max_runtime_seconds)
        .or_else(|| mold_core::validation::max_runtime_seconds_for_family(family));
    if max_runtime_seconds.is_some_and(|maximum| seconds > f64::from(maximum)) {
        return Err(format!(
            "Duration for this model cannot exceed {} seconds.",
            max_runtime_seconds.unwrap_or_default()
        ));
    }

    let step = defaults
        .and_then(|value| value.frame_step)
        .or_else(|| mold_core::validation::frame_step_for_family(family))
        .unwrap_or(1)
        .max(1);
    let offset = defaults
        .and_then(|value| value.frame_offset)
        .or_else(|| mold_core::validation::frame_offset_for_family(family))
        .unwrap_or(1)
        .max(1);
    let requested_frames = seconds * f64::from(resolved_fps);
    if requested_frames > f64::from(u32::MAX) {
        return Err("Duration is too large.".to_string());
    }
    let frames_on_grid = if requested_frames <= f64::from(offset) {
        offset
    } else {
        (((requested_frames - f64::from(offset)) / f64::from(step)).round() as u32)
            .saturating_mul(step)
            .saturating_add(offset)
    };
    let min_frames = mold_core::validation::min_frames_for_family(family).unwrap_or(offset);
    if frames_on_grid < min_frames {
        return Err(format!(
            "Duration for this model must be at least {:.1} seconds.",
            f64::from(min_frames) / f64::from(resolved_fps)
        ));
    }

    let max_frames = if let Some(runtime_seconds) = max_runtime_seconds {
        let raw = runtime_seconds
            .saturating_mul(resolved_fps)
            .saturating_add(4);
        raw.min(
            defaults
                .and_then(|value| value.max_frames_absolute)
                .or_else(|| mold_core::validation::max_frames_absolute_for_family(family))
                .unwrap_or(u32::MAX),
        )
    } else {
        defaults
            .and_then(|value| value.max_frames)
            .or_else(|| mold_core::validation::max_frames_for_family_at_fps(family, resolved_fps))
            .unwrap_or(u32::MAX)
    };
    let max_frames_on_grid = if max_frames < offset {
        max_frames
    } else if max_frames == offset {
        offset
    } else {
        max_frames - ((max_frames - offset) % step)
    };
    if frames_on_grid > max_frames_on_grid {
        return Err(format!(
            "Duration is too long for this model at {resolved_fps} FPS (maximum {:.1} seconds).",
            f64::from(max_frames_on_grid) / f64::from(resolved_fps)
        ));
    }

    Ok((Some(frames_on_grid), Some(resolved_fps)))
}

/// Build a `GenerateRequest` from slash command parameters, honoring model
/// defaults and routing video families to an appropriate container.
pub fn build_generate_request(params: BuildParams<'_>) -> GenerateRequest {
    let (def_w, def_h, def_steps, def_guidance) = match params.defaults {
        Some(d) => (
            d.default_width,
            d.default_height,
            d.default_steps,
            d.default_guidance,
        ),
        None => (1024, 1024, 20, 3.5),
    };

    let is_video_family = params.family.and_then(video_family).is_some();
    let is_ltx2 = params.family == Some("ltx2");
    let is_h3 = params.family.is_some_and(mold_core::minimax_h3::is_family);

    // Video families always deliver MP4 or GIF. Image models stay PNG.
    let output_format = if is_h3 {
        // H3 always emits synchronized audio and its contract requires an MP4
        // container. This does not activate the hidden family; it only keeps
        // the request builder from manufacturing an invalid future request.
        OutputFormat::Mp4
    } else if is_video_family {
        params
            .video_format
            .map(VideoFormat::to_output_format)
            .unwrap_or(OutputFormat::Mp4)
    } else {
        OutputFormat::Png
    };

    // Match the model defaults advertised by `/api/models`, which are also the
    // defaults used by Studio. Older servers fall back to the legacy values.
    let frames = if is_video_family {
        Some(
            params
                .frames
                .or_else(|| params.defaults.and_then(|value| value.default_frames))
                .unwrap_or(if is_h3 {
                    mold_core::minimax_h3::MIN_FRAMES
                } else {
                    25
                }),
        )
    } else {
        params.frames
    };
    let fps = if is_h3 {
        // This is a final defense for callers that bypass the slash-command
        // timing resolver. H3's audio/video clock is fixed at 24 FPS.
        Some(mold_core::minimax_h3::FIXED_FPS)
    } else if is_video_family {
        Some(
            params
                .fps
                .or_else(|| params.defaults.and_then(|value| value.default_fps))
                .unwrap_or(24),
        )
    } else {
        params.fps
    };

    // Optional LTX-2 audio follows the user override; H3 audio is inherent.
    let enable_audio = if is_h3 {
        Some(true)
    } else if is_ltx2 {
        params.audio
    } else {
        None
    };

    GenerateRequest {
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: params.prompt.to_string(),
        negative_prompt: if is_h3 {
            None
        } else {
            params.negative_prompt.map(|s| s.to_string())
        },
        model: params.model.to_string(),
        width: params.width.unwrap_or(if is_h3 {
            mold_core::minimax_h3::DEFAULT_WIDTH
        } else {
            def_w
        }),
        height: params.height.unwrap_or(if is_h3 {
            mold_core::minimax_h3::DEFAULT_HEIGHT
        } else {
            def_h
        }),
        steps: params.steps.unwrap_or(if is_h3 {
            mold_core::minimax_h3::DEFAULT_STEPS
        } else {
            def_steps
        }),
        // H3 has no CFG path; an explicit Discord override cannot weaken the
        // model contract even when this builder is called independently.
        guidance: if is_h3 {
            0.0
        } else {
            params.guidance.unwrap_or(def_guidance)
        },
        seed: params.seed,
        batch_size: 1,
        output_format: Some(output_format),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        edit_images: None,
        references: if is_h3
            && mold_core::minimax_h3::task_for_model(params.model)
                == Some(mold_core::minimax_h3::Task::Ref2va)
        {
            params.references
        } else {
            None
        },
        source_image: params.source_image,
        source_image_name: params.source_image_name,
        strength: if is_h3 {
            1.0
        } else {
            params.strength.unwrap_or(0.75)
        },
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: None,
        frames,
        fps,
        upscale_model: None,
        // Always request a GIF preview for video jobs: if the primary MP4
        // exceeds Discord's upload ceiling we fall back to the preview so the
        // user still sees the generation.
        gif_preview: is_video_family,
        enable_audio,
        audio_file: params.audio_file,
        audio_file_path: None,
        source_video: params.source_video,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        keyframes: params.keyframes,
        pipeline: if is_ltx2 { params.pipeline } else { None },
        ic_lora_control: None,
        loras: None,
        retake_range: params.retake_range,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
    }
}

/// Resolve the default model name from the cached model list.
/// Prefers: loaded model > smallest downloaded model > "flux2-klein:q8" fallback.
fn resolve_default_model(models: &[mold_core::ModelInfoExtended]) -> String {
    // Prefer the currently loaded model
    if let Some(loaded) = models.iter().find(|m| m.info.is_loaded) {
        return loaded.info.name.clone();
    }
    // Fall back to the smallest downloaded generative model (avoids accidentally
    // picking a 23GB BF16 variant, ControlNet, or utility models like qwen3-expand).
    if let Some(downloaded) = models
        .iter()
        .filter(|m| {
            m.downloaded
                && m.info.family != "controlnet"
                && !mold_core::manifest::UTILITY_FAMILIES.contains(&m.info.family.as_str())
        })
        .min_by(|a, b| {
            a.info
                .size_gb
                .partial_cmp(&b.info.size_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        return downloaded.info.name.clone();
    }
    // Last resort
    "flux2-klein:q8".to_string()
}

/// Maximum inline attachment size we will accept for `source_image`. Discord
/// caps uploads to 25 MiB for free users (100 MiB with Nitro), but very large
/// images are almost always a mistake for img2img — resizing happens inside
/// the server. Keep the bar well below Discord's hard limit to avoid obvious
/// abuse.
const MAX_SOURCE_IMAGE_BYTES: u64 = 10 * 1024 * 1024;
// GenerateRequest embeds every attachment as base64 in one JSON body. Keep the
// aggregate raw payload below 47 MiB so its 4/3 expansion plus request metadata
// remains below mold-server's 64 MiB body limit.
const MAX_INLINE_MEDIA_BYTES: u64 = 47 * 1024 * 1024;

fn validate_inline_media_size<'a>(
    attachments: impl IntoIterator<Item = &'a serenity::Attachment>,
) -> Result<(), String> {
    validate_inline_media_lengths(
        attachments
            .into_iter()
            .map(|attachment| attachment.size as u64),
    )
}

fn validate_inline_media_lengths(sizes: impl IntoIterator<Item = u64>) -> Result<(), String> {
    let total = sizes
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
        .ok_or_else(|| "Combined attachment size is too large.".to_string())?;
    if total > MAX_INLINE_MEDIA_BYTES {
        return Err(format!(
            "Combined attachments are too large ({:.1} MiB). Keep their total under {} MiB.",
            total as f64 / (1024.0 * 1024.0),
            MAX_INLINE_MEDIA_BYTES / (1024 * 1024)
        ));
    }
    Ok(())
}

/// Derive img2img dimensions from a Discord attachment's reported
/// width/height when the user didn't supply explicit values — otherwise the
/// server would `resize_exact` a landscape photo into a square default.
/// Fits the source into the model's default canvas via the shared
/// aspect-preserving helper used by the CLI and server. H3 owns a stricter
/// 32px canvas contract and its official short-edge/area rounding authority;
/// other models use the grid advertised by the server (`dimension_alignment`
/// is 32 for `wan22-ti2v-5b`, 16 elsewhere, and omitted by older servers).
fn fit_attachment_dims(
    w: u32,
    h: u32,
    defaults: Option<&mold_core::ModelDefaults>,
    family: Option<&str>,
) -> (u32, u32) {
    if family.is_some_and(mold_core::minimax_h3::is_family) {
        return mold_core::minimax_h3::recommended_dimensions(w, h);
    }
    let (model_w, model_h) = defaults
        .map(|d| (d.default_width, d.default_height))
        .unwrap_or((1024, 1024));
    let align = defaults.and_then(|d| d.dimension_alignment).unwrap_or(16);
    mold_core::fit_to_model_dimensions_aligned(w, h, model_w, model_h, align)
}

/// Download an attachment and sanity-check that it looks like a PNG or JPEG
/// before we ship the bytes to the server (which will reject anything else
/// with a less friendly error).
async fn fetch_source_image(att: &serenity::Attachment) -> Result<Vec<u8>, String> {
    if att.size as u64 > MAX_SOURCE_IMAGE_BYTES {
        return Err(format!(
            "Source image is too large ({:.1} MiB). Keep it under {} MiB.",
            att.size as f64 / (1024.0 * 1024.0),
            MAX_SOURCE_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    if let Some(ct) = &att.content_type {
        if !(ct.starts_with("image/png") || ct.starts_with("image/jpeg")) {
            return Err(format!("Source image must be PNG or JPEG, got `{ct}`."));
        }
    }
    let bytes = att
        .download()
        .await
        .map_err(|e| format!("Failed to download source image: {e}"))?;
    if !looks_like_png_or_jpeg(&bytes) {
        return Err("Source image must be a valid PNG or JPEG file.".to_string());
    }
    Ok(bytes)
}

fn looks_like_png_or_jpeg(bytes: &[u8]) -> bool {
    let is_png = bytes.len() >= 4 && bytes[..4] == [0x89, 0x50, 0x4E, 0x47];
    let is_jpeg = bytes.len() >= 2 && bytes[..2] == [0xFF, 0xD8];
    is_png || is_jpeg
}

async fn fetch_conditioning_media(
    att: &serenity::Attachment,
    kind: &str,
    expected_content_prefix: &str,
) -> Result<Vec<u8>, String> {
    if att.size as u64 > MAX_INLINE_MEDIA_BYTES {
        return Err(format!(
            "{kind} is too large ({:.1} MiB). Keep it under {} MiB.",
            att.size as f64 / (1024.0 * 1024.0),
            MAX_INLINE_MEDIA_BYTES / (1024 * 1024)
        ));
    }
    if let Some(content_type) = &att.content_type {
        if !content_type.starts_with(expected_content_prefix) {
            return Err(format!(
                "{kind} must be a {expected_content_prefix} attachment, got `{content_type}`."
            ));
        }
    }
    let bytes = att
        .download()
        .await
        .map_err(|error| format!("Failed to download {kind}: {error}"))?;
    if bytes.is_empty() {
        return Err(format!("{kind} must not be empty."));
    }
    Ok(bytes)
}

fn specialized_pipeline(
    has_source_image: bool,
    has_audio: bool,
    has_video: bool,
    retake_start: Option<f64>,
    retake_end: Option<f64>,
    keyframe_count: usize,
) -> Result<Option<Ltx2PipelineMode>, String> {
    let has_retake_range = retake_start.is_some() || retake_end.is_some();
    if has_source_image && (has_video || keyframe_count > 0) {
        return Err(
            "Source image cannot be combined with retake or keyframe attachments.".to_string(),
        );
    }
    let groups = usize::from(has_audio)
        + usize::from(has_video || has_retake_range)
        + usize::from(keyframe_count > 0);
    if groups > 1 {
        return Err(
            "Use only one specialized mode at a time: audio-to-video, retake, or keyframes."
                .to_string(),
        );
    }
    if has_video || has_retake_range {
        if !has_video {
            return Err("Retake times require a source video attachment.".to_string());
        }
        let (Some(start), Some(end)) = (retake_start, retake_end) else {
            return Err("Retake requires both start and end times in seconds.".to_string());
        };
        if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
            return Err(
                "Retake end time must be greater than a non-negative start time.".to_string(),
            );
        }
        return Ok(Some(Ltx2PipelineMode::Retake));
    }
    if keyframe_count > 0 {
        if keyframe_count < 2 {
            return Err("Keyframe mode requires at least two keyframe images.".to_string());
        }
        return Ok(Some(Ltx2PipelineMode::Keyframe));
    }
    Ok(has_audio.then_some(Ltx2PipelineMode::A2Vid))
}

fn build_keyframes(images: Vec<Vec<u8>>, frames: u32) -> Vec<KeyframeCondition> {
    let last_frame = frames.saturating_sub(1);
    let denominator = u32::try_from(images.len().saturating_sub(1))
        .unwrap_or(1)
        .max(1);
    images
        .into_iter()
        .enumerate()
        .map(|(index, image)| KeyframeCondition {
            frame: u32::try_from(index).unwrap_or(0) * last_frame / denominator,
            image,
            name: None,
        })
        .collect()
}

/// Whether `/generate` must reject the interaction up front for a missing prompt.
///
/// The slash command runs this before deferring, so the model family isn't
/// resolved yet — only the attachment is known. A source image is enough to
/// let the run through unprompted (LTX-2 image-to-video conditions on the
/// frame); the server's family-aware validator is the backstop that still
/// rejects an empty prompt for image families.
fn prompt_missing_before_defer(prompt: Option<&str>, has_visual_conditioning: bool) -> bool {
    !has_visual_conditioning && prompt.is_none_or(|p| p.trim().is_empty())
}

/// Generate an image (PNG) or video (MP4 by default, GIF on request).
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn generate(
    ctx: Context<'_>,
    #[description = "Text prompt — optional for image-to-video when a source image is attached"]
    prompt: Option<String>,
    #[description = "Model to use (e.g. flux-schnell:q8, ltx-2-19b-distilled:fp8)"]
    #[autocomplete = "autocomplete_model"]
    model: Option<String>,
    #[description = "Source image for img2img / img-to-video (PNG or JPEG)"] source_image: Option<
        serenity::Attachment,
    >,
    #[description = "Video container (MP4 default, GIF for easier sharing) — video models only"]
    video_format: Option<VideoFormat>,
    #[description = "Number of video frames (video models only; LTX prefers 8n+1)"] frames: Option<
        u32,
    >,
    #[description = "Video duration in seconds (simple alternative to frames; video models only)"]
    duration: Option<f64>,
    #[description = "Video FPS (video models only; uses the model default)"] fps: Option<u32>,
    #[description = "Image width in pixels"] width: Option<u32>,
    #[description = "Image height in pixels"] height: Option<u32>,
    #[description = "Number of inference steps"] steps: Option<u32>,
    #[description = "Guidance scale (0.0 for schnell, ~3.5 for dev)"] guidance: Option<f64>,
    #[description = "Random seed for reproducibility"] seed: Option<u64>,
    #[description = "img2img strength (0.0 = preserve source, 1.0 = full noise)"] strength: Option<
        f64,
    >,
    #[description = "Enable synchronized audio (LTX-2 + MP4 only)"] audio: Option<bool>,
    #[description = "LTX-2 pipeline mode (advanced)"] pipeline: Option<PipelineChoice>,
    #[description = "Negative prompt — what to avoid (CFG models: SD1.5, SDXL, SD3)"]
    negative_prompt: Option<String>,
    #[description = "Audio attachment for LTX-2 audio-to-video"] audio_file: Option<
        serenity::Attachment,
    >,
    #[description = "MP4 source for LTX-2 retake"] source_video: Option<serenity::Attachment>,
    #[description = "Retake start time in seconds"] retake_start: Option<f64>,
    #[description = "Retake end time in seconds"] retake_end: Option<f64>,
    #[description = "First image for LTX-2 keyframe interpolation"] keyframe_1: Option<
        serenity::Attachment,
    >,
    #[description = "Second image for LTX-2 keyframe interpolation"] keyframe_2: Option<
        serenity::Attachment,
    >,
    #[description = "Optional third image for LTX-2 keyframe interpolation"] keyframe_3: Option<
        serenity::Attachment,
    >,
    #[description = "First ordered H3 reference (image, H.264 MP4, or WAV)"] reference_1: Option<
        serenity::Attachment,
    >,
    #[description = "Second ordered H3 reference (image, H.264 MP4, or WAV)"] reference_2: Option<
        serenity::Attachment,
    >,
) -> Result<()> {
    if reference_1.is_none() && reference_2.is_some() {
        ctx.send(
            poise::CreateReply::default()
                .content("Add reference_1 before reference_2 so H3 ordering is unambiguous.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    let keyframe_attachments = [
        keyframe_1.as_ref(),
        keyframe_2.as_ref(),
        keyframe_3.as_ref(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let reference_attachments = [reference_1.as_ref(), reference_2.as_ref()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if !reference_attachments.is_empty()
        && (source_image.is_some()
            || audio_file.is_some()
            || source_video.is_some()
            || !keyframe_attachments.is_empty()
            || retake_start.is_some()
            || retake_end.is_some()
            || pipeline.is_some())
    {
        ctx.send(
            poise::CreateReply::default()
                .content(
                    "Ordered H3 references cannot be combined with source, retake, keyframe, audio, or pipeline inputs.",
                )
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    let specialized = match specialized_pipeline(
        source_image.is_some(),
        audio_file.is_some(),
        source_video.is_some(),
        retake_start,
        retake_end,
        keyframe_attachments.len(),
    ) {
        Ok(mode) => mode,
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
    if specialized.is_some() && pipeline.is_some() {
        ctx.send(
            poise::CreateReply::default()
                .content("The attachment selects the LTX-2 pipeline; omit the pipeline option.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    let inline_attachments = source_image
        .iter()
        .chain(audio_file.iter())
        .chain(source_video.iter())
        .chain(keyframe_attachments.iter().copied())
        .chain(reference_attachments.iter().copied());
    if let Err(message) = validate_inline_media_size(inline_attachments) {
        ctx.send(
            poise::CreateReply::default()
                .content(message)
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    // Validate prompt before deferring (avoids wasting the interaction)
    if prompt_missing_before_defer(
        prompt.as_deref(),
        source_image.is_some() || source_video.is_some() || !keyframe_attachments.is_empty(),
    ) {
        ctx.send(
            poise::CreateReply::default()
                .content(
                    "Prompt cannot be empty. \
                     (It's optional only with visual conditioning: a source image, retake video, \
                     or keyframes.)",
                )
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }
    let prompt = prompt.unwrap_or_default();

    // Check authorization (block list, roles, cooldown, quota)
    let user_id = ctx.author().id.get();
    if let AuthResult::Denied(msg) = checks::check_generate_auth(&ctx).await {
        ctx.send(poise::CreateReply::default().content(msg).ephemeral(true))
            .await?;
        return Ok(());
    }

    // Defer the response (shows "Bot is thinking...")
    ctx.defer().await?;

    // Resolve model name — use server's loaded/downloaded model if none specified
    let models = ctx.data().cached_models().await;
    let model_name = model.unwrap_or_else(|| resolve_default_model(&models));

    // Look up model defaults + family from cache
    let model_entry = models.iter().find(|m| m.info.name == model_name);
    // Autocomplete deliberately falls back to built-in manifests while the
    // server cache is cold. Preserve duration/default behavior in that window
    // instead of presenting a model that the timing resolver cannot identify.
    let fallback_manifest = model_entry
        .is_none()
        .then(|| mold_core::manifest::find_manifest(&model_name))
        .flatten();
    let fallback_defaults = fallback_manifest.map(defaults_from_manifest);
    let model_defaults = model_entry
        .map(|m| &m.defaults)
        .or(fallback_defaults.as_ref());
    let family = model_entry
        .map(|m| m.info.family.as_str())
        .or_else(|| fallback_manifest.map(|manifest| manifest.family.as_str()))
        .or_else(|| {
            mold_core::minimax_h3::task_for_model(&model_name)
                .map(|_| mold_core::minimax_h3::FAMILY)
        });
    let h3_task = mold_core::minimax_h3::task_for_model(&model_name);
    if !reference_attachments.is_empty() && h3_task != Some(mold_core::minimax_h3::Task::Ref2va) {
        ctx.data().quotas.refund(user_id);
        handler::send_error(
            ctx,
            "Ordered references require an explicitly authorized MiniMax H3 Ref2VA model.",
        )
        .await?;
        return Ok(());
    }
    if h3_task == Some(mold_core::minimax_h3::Task::Ref2va) && reference_attachments.is_empty() {
        ctx.data().quotas.refund(user_id);
        handler::send_error(
            ctx,
            "MiniMax H3 Ref2VA requires at least one ordered reference.",
        )
        .await?;
        return Ok(());
    }
    // Preflight the source-image contract before the attachment download and
    // the enqueue, so an impossible pairing never takes a queue slot.
    if let Some(message) = source_image_contract_error(
        family,
        &model_name,
        resolve_source_image_contract(model_entry, &model_name),
        source_image.is_some() || !keyframe_attachments.is_empty(),
    ) {
        ctx.data().quotas.refund(user_id);
        handler::send_error(ctx, &message).await?;
        return Ok(());
    }

    let (resolved_frames, resolved_fps) =
        match resolve_video_timing(family, model_defaults, frames, fps, duration) {
            Ok(timing) => timing,
            Err(message) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(ctx, &message).await?;
                return Ok(());
            }
        };

    let prepared_references = if reference_attachments.is_empty() {
        None
    } else {
        match crate::h3_references::prepare_attachments(&ctx.data().client, &reference_attachments)
            .await
        {
            Ok(prepared) => Some(prepared),
            Err(error) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(
                    ctx,
                    &format!("MiniMax H3 reference preparation failed: {error}"),
                )
                .await?;
                return Ok(());
            }
        }
    };

    // Fetch source image bytes if an attachment was supplied. This also covers
    // the LTX-2 image-to-video path — the server forwards a source_image to
    // LTX-2 as the first keyframe automatically.
    let source_bytes = if let Some(att) = source_image.as_ref() {
        match fetch_source_image(att).await {
            Ok(bytes) => Some(bytes),
            Err(msg) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(ctx, &msg).await?;
                return Ok(());
            }
        }
    } else {
        None
    };
    let audio_bytes = if let Some(att) = audio_file.as_ref() {
        match fetch_conditioning_media(att, "Audio attachment", "audio/").await {
            Ok(bytes) => Some(bytes),
            Err(message) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(ctx, &message).await?;
                return Ok(());
            }
        }
    } else {
        None
    };
    let video_bytes = if let Some(att) = source_video.as_ref() {
        match fetch_conditioning_media(att, "Source video", "video/").await {
            Ok(bytes) => Some(bytes),
            Err(message) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(ctx, &message).await?;
                return Ok(());
            }
        }
    } else {
        None
    };
    let mut keyframe_images = Vec::with_capacity(keyframe_attachments.len());
    for attachment in keyframe_attachments {
        match fetch_source_image(attachment).await {
            Ok(bytes) => keyframe_images.push(bytes),
            Err(message) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(ctx, &message).await?;
                return Ok(());
            }
        }
    }

    // When a source image is attached and the user didn't pass explicit dims,
    // use the attachment's reported width/height (snapped to multiples of 16)
    // instead of falling through to the model's square default. Without this
    // landscape/portrait photos get `resize_exact`'d to 1024x1024.
    let (effective_width, effective_height) =
        if source_bytes.is_some() && width.is_none() && height.is_none() {
            match (
                source_image.as_ref().and_then(|a| a.width),
                source_image.as_ref().and_then(|a| a.height),
            ) {
                (Some(w), Some(h)) => {
                    let (sw, sh) = fit_attachment_dims(w, h, model_defaults, family);
                    (Some(sw), Some(sh))
                }
                _ => (width, height),
            }
        } else {
            (width, height)
        };

    let requested_frames = resolved_frames.unwrap_or(25);
    let keyframes =
        (!keyframe_images.is_empty()).then(|| build_keyframes(keyframe_images, requested_frames));
    let retake_range = match (retake_start, retake_end) {
        (Some(start_seconds), Some(end_seconds)) => Some(TimeRange {
            start_seconds: start_seconds as f32,
            end_seconds: end_seconds as f32,
        }),
        _ => None,
    };
    let mut req = build_generate_request(BuildParams {
        prompt: &prompt,
        model: &model_name,
        family,
        width: effective_width,
        height: effective_height,
        steps,
        guidance,
        seed,
        negative_prompt: negative_prompt.as_deref(),
        defaults: model_defaults,
        source_image: source_bytes,
        source_image_name: source_image
            .as_ref()
            .and_then(|attachment| crate::h3_references::safe_name(&attachment.filename)),
        references: prepared_references
            .as_ref()
            .map(|prepared| prepared.descriptors.clone()),
        frames: resolved_frames,
        fps: resolved_fps,
        strength,
        video_format,
        audio,
        pipeline: specialized.or_else(|| pipeline.map(PipelineChoice::to_mode)),
        audio_file: audio_bytes,
        source_video: video_bytes,
        keyframes,
        retake_range,
    });

    let mut reference_session = if let Some(prepared) = prepared_references {
        match crate::h3_references::bind_remote_references(&ctx.data().client, &mut req, prepared)
            .await
        {
            Ok(session) => Some(session),
            Err(error) => {
                ctx.data().quotas.refund(user_id);
                handler::send_error(
                    ctx,
                    &format!("MiniMax H3 reference authorization/upload failed: {error}"),
                )
                .await?;
                return Ok(());
            }
        }
    } else {
        None
    };

    match handler::run_generation(ctx, req).await {
        Ok(()) => {
            if let Some(lease) = reference_session.as_mut() {
                lease.mark_consumed();
            }
            // Quota slot was already consumed atomically in check_generate_auth
            ctx.data().cooldowns.record(user_id);
        }
        Err(e) => {
            if let Some(lease) = reference_session.as_mut() {
                let _ = lease.cancel().await;
            }
            // Refund the quota slot consumed during auth check
            ctx.data().quotas.refund(user_id);
            let msg = if h3_task.is_some() {
                format!(
                    "MiniMax H3 runtime/authorization is unavailable: {e} ({})",
                    mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED
                )
            } else if mold_core::MoldClient::is_connection_error(&e) {
                "Could not connect to the mold server. Is it running?".to_string()
            } else if mold_core::MoldClient::is_model_not_found(&e) {
                format!("Model '{model_name}' is not downloaded. Use `/models` to see available models.")
            } else {
                format!("Generation failed: {e}")
            };
            handler::send_error(ctx, &msg).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> mold_core::ModelDefaults {
        mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 0.0,
            default_width: 1024,
            default_height: 1024,
            description: "test".to_string(),
            ..Default::default()
        }
    }

    fn base_params<'a>(prompt: &'a str, model: &'a str) -> BuildParams<'a> {
        BuildParams {
            prompt,
            model,
            ..BuildParams::default()
        }
    }

    /// The pre-enqueue contract check. Identical table in `mold-ai` and
    /// `mold-ai-tui`; all three must agree with the server's admission gate,
    /// whose wording they reuse verbatim.
    #[test]
    fn source_image_contract_rejects_exactly_what_admission_rejects() {
        use mold_core::SourceImageCapability::{Optional, Required, Unsupported};

        for (capability, has_source, rejected) in [
            // Unknown contract — an older server, or a checkpoint it could
            // not classify — enforces nothing.
            (None, false, false),
            (None, true, false),
            (Some(Optional), false, false),
            (Some(Optional), true, false),
            (Some(Unsupported), false, false),
            (Some(Unsupported), true, true),
            (Some(Required), false, true),
            (Some(Required), true, false),
        ] {
            assert_eq!(
                source_image_contract_error(
                    Some("wan"),
                    "wan22-t2v-a14b:q8",
                    capability,
                    has_source
                )
                .is_some(),
                rejected,
                "{capability:?} with has_source={has_source}"
            );
        }

        assert!(source_image_contract_error(
            Some("wan"),
            "wan22-t2v-a14b:q8",
            Some(Unsupported),
            true
        )
        .is_some_and(|message| message.contains("text-to-video only")));
        assert!(source_image_contract_error(
            Some("wan"),
            "wan22-i2v-a14b:q8",
            Some(Required),
            false
        )
        .is_some_and(|message| message.contains("needs a source image")));
        // Family-aware: a plain LTX-Video refusal names the actual model, not Wan.
        let message = source_image_contract_error(
            Some("ltx-video"),
            "ltx-video-0.9.6:bf16",
            Some(Unsupported),
            true,
        )
        .expect("unsupported + source refuses");
        assert!(message.contains("ltx-video-0.9.6"), "{message}");
        assert!(!message.contains("Wan"), "{message}");
    }

    /// The served row is the first authority; the built-in manifest answers
    /// for a cold cache and for a server too old to advertise the field.
    #[test]
    fn advertised_contract_wins_over_the_manifest_fallback() {
        let mut advertised = mk_model("wan22-i2v-a14b:q5", "wan", true);
        advertised.source_image = Some(mold_core::SourceImageCapability::Optional);

        assert_eq!(
            resolve_source_image_contract(Some(&advertised), "wan22-i2v-a14b:q5"),
            Some(mold_core::SourceImageCapability::Optional),
            "a served row is the server's own classification of the checkpoint"
        );
        assert_eq!(
            resolve_source_image_contract(None, "wan22-i2v-a14b:q5"),
            Some(mold_core::SourceImageCapability::Required),
            "a cold cache falls back to the manifest tier"
        );
        assert_eq!(
            resolve_source_image_contract(
                Some(&mk_model("wan22-i2v-a14b:q5", "wan", true)),
                "wan22-i2v-a14b:q5"
            ),
            Some(mold_core::SourceImageCapability::Required),
            "a row from a server that predates the field is not evidence of absence"
        );
        // No manifest tier and nothing advertised: the contract stays unknown
        // and the engine remains the authority.
        assert_eq!(resolve_source_image_contract(None, "cv:12345"), None);
    }

    #[test]
    fn build_request_with_defaults() {
        let defs = defaults();
        let req = build_generate_request(BuildParams {
            defaults: Some(&defs),
            ..base_params("a cat", "flux2-klein:q8")
        });
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.model, "flux2-klein:q8");
        assert_eq!(req.width, 1024);
        assert_eq!(req.height, 1024);
        assert_eq!(req.steps, 4);
        assert_eq!(req.guidance, 0.0);
        assert!(req.seed.is_none());
        assert!(req.negative_prompt.is_none());
        assert_eq!(req.batch_size, 1);
        assert_eq!(req.output_format, Some(OutputFormat::Png));
        assert!(req.frames.is_none());
        assert!(req.fps.is_none());
        assert!(req.source_image.is_none());
    }

    #[test]
    fn build_request_overrides() {
        let defs = defaults();
        let req = build_generate_request(BuildParams {
            width: Some(768),
            height: Some(512),
            steps: Some(28),
            guidance: Some(7.5),
            seed: Some(42),
            defaults: Some(&defs),
            ..base_params("a dog", "flux-dev:q4")
        });
        assert_eq!(req.width, 768);
        assert_eq!(req.height, 512);
        assert_eq!(req.steps, 28);
        assert_eq!(req.guidance, 7.5);
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn build_request_no_defaults() {
        let req = build_generate_request(base_params("test", "unknown-model"));
        assert_eq!(req.width, 1024);
        assert_eq!(req.height, 1024);
        assert_eq!(req.steps, 20);
        assert_eq!(req.guidance, 3.5);
        assert_eq!(req.output_format, Some(OutputFormat::Png));
    }

    #[test]
    fn build_request_partial_overrides() {
        let defs = mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 0.0,
            default_width: 512,
            default_height: 512,
            description: "test".to_string(),
            ..Default::default()
        };
        let req = build_generate_request(BuildParams {
            width: Some(768),
            defaults: Some(&defs),
            ..base_params("test", "sd15:fp16")
        });
        assert_eq!(req.width, 768);
        assert_eq!(req.height, 512); // from defaults
        assert_eq!(req.steps, 4); // from defaults
    }

    #[test]
    fn build_request_with_negative_prompt() {
        let req = build_generate_request(BuildParams {
            negative_prompt: Some("blurry, low quality"),
            ..base_params("a cat", "sd15:fp16")
        });
        assert_eq!(req.negative_prompt.as_deref(), Some("blurry, low quality"));
    }

    // --- Video routing tests ---

    #[test]
    fn ltx_video_defaults_to_mp4_with_frames() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx-video"),
            ..base_params("a drone shot", "ltx-video-0.9.6-distilled:bf16")
        });
        assert_eq!(req.output_format, Some(OutputFormat::Mp4));
        assert_eq!(req.frames, Some(25));
        assert_eq!(req.fps, Some(24));
    }

    #[test]
    fn ltx2_defaults_to_mp4() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            ..base_params("panning over a beach", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.output_format, Some(OutputFormat::Mp4));
        assert_eq!(req.frames, Some(25));
    }

    #[test]
    fn video_uses_studio_model_defaults() {
        let defs = mold_core::ModelDefaults {
            default_frames: Some(97),
            default_fps: Some(24),
            ..defaults()
        };
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            defaults: Some(&defs),
            ..base_params("panning over a beach", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.frames, Some(97));
        assert_eq!(req.fps, Some(24));
    }

    #[test]
    fn duration_uses_model_fps_and_snaps_to_the_frame_grid() {
        let defs = mold_core::ModelDefaults {
            default_frames: Some(97),
            default_fps: Some(24),
            max_runtime_seconds: Some(20),
            max_frames_absolute: Some(604),
            frame_step: Some(8),
            ..defaults()
        };

        assert_eq!(
            resolve_video_timing(Some("ltx2"), Some(&defs), None, None, Some(10.0)).unwrap(),
            (Some(241), Some(24))
        );
        assert_eq!(
            resolve_video_timing(Some("ltx2"), Some(&defs), None, None, Some(20.0)).unwrap(),
            (Some(481), Some(24))
        );
        assert_eq!(
            resolve_video_timing(Some("ltx2"), Some(&defs), None, Some(30), Some(10.0)).unwrap(),
            (Some(297), Some(30))
        );
    }

    #[test]
    fn h3_duration_uses_its_five_offset_frame_grid() {
        let defs = mold_core::ModelDefaults {
            default_frames: Some(mold_core::minimax_h3::MIN_FRAMES),
            default_fps: Some(mold_core::minimax_h3::FIXED_FPS),
            max_runtime_seconds: Some(mold_core::minimax_h3::MAX_DURATION_SECONDS),
            max_frames_absolute: Some(mold_core::minimax_h3::MAX_FRAMES),
            frame_step: Some(mold_core::minimax_h3::FRAME_STEP),
            frame_offset: Some(mold_core::minimax_h3::FRAME_OFFSET),
            ..defaults()
        };

        assert_eq!(
            resolve_video_timing(
                Some(mold_core::minimax_h3::FAMILY),
                Some(&defs),
                None,
                None,
                Some(5.0),
            )
            .unwrap(),
            (Some(124), Some(24))
        );
        assert_eq!(
            resolve_video_timing(
                Some(mold_core::minimax_h3::FAMILY),
                Some(&defs),
                None,
                None,
                Some(15.0),
            )
            .unwrap(),
            (Some(362), Some(24))
        );
        assert_eq!(
            resolve_video_timing(
                Some(mold_core::minimax_h3::FAMILY),
                Some(&defs),
                None,
                None,
                Some(4.9),
            )
            .unwrap_err(),
            "MiniMax H3 duration must be at least 5 seconds."
        );
    }

    #[test]
    fn h3_rejects_explicit_non_fixed_fps_on_both_timing_paths() {
        let duration_error = resolve_video_timing(
            Some(mold_core::minimax_h3::FAMILY),
            None,
            None,
            Some(30),
            Some(5.0),
        )
        .unwrap_err();
        let frames_error = resolve_video_timing(
            Some("minimax_h3"),
            None,
            Some(mold_core::minimax_h3::MIN_FRAMES),
            Some(30),
            None,
        )
        .unwrap_err();

        assert_eq!(duration_error, "MiniMax H3 requires 24 FPS.");
        assert_eq!(frames_error, "MiniMax H3 requires 24 FPS.");
    }

    #[test]
    fn h3_request_builder_preserves_mandatory_av_contract() {
        let defs = mold_core::ModelDefaults {
            default_frames: Some(mold_core::minimax_h3::MIN_FRAMES),
            default_fps: Some(mold_core::minimax_h3::FIXED_FPS),
            default_width: mold_core::minimax_h3::DEFAULT_WIDTH,
            default_height: mold_core::minimax_h3::DEFAULT_HEIGHT,
            default_steps: mold_core::minimax_h3::DEFAULT_STEPS,
            default_guidance: 0.0,
            frame_step: Some(mold_core::minimax_h3::FRAME_STEP),
            frame_offset: Some(mold_core::minimax_h3::FRAME_OFFSET),
            ..defaults()
        };
        let req = build_generate_request(BuildParams {
            family: Some("minimax_h3"),
            defaults: Some(&defs),
            // Neither a user-selected GIF nor a false audio flag may weaken
            // H3's synchronized MP4 contract.
            video_format: Some(VideoFormat::Gif),
            audio: Some(false),
            strength: Some(0.25),
            fps: Some(30),
            guidance: Some(7.5),
            negative_prompt: Some("stale negative"),
            ..base_params("a quiet shoreline", mold_core::minimax_h3::FL2VA_COMFY)
        });

        assert_eq!(req.output_format, Some(OutputFormat::Mp4));
        assert_eq!(req.enable_audio, Some(true));
        assert_eq!(req.frames, Some(124));
        assert_eq!(req.fps, Some(24));
        assert_eq!(req.steps, 50);
        assert_eq!(req.guidance, 0.0);
        assert_eq!(req.strength, 1.0);
        assert_eq!(req.negative_prompt, None);

        let without_server_defaults = build_generate_request(BuildParams {
            family: Some(mold_core::minimax_h3::FAMILY),
            ..base_params("a quiet shoreline", mold_core::minimax_h3::FL2VA_COMFY)
        });
        assert_eq!(
            (
                without_server_defaults.width,
                without_server_defaults.height
            ),
            (1344, 768)
        );
        assert_eq!(without_server_defaults.frames, Some(124));
        assert_eq!(without_server_defaults.fps, Some(24));
    }

    #[test]
    fn h3_builder_preserves_ordered_descriptor_authority_for_session_binding() {
        let references = vec![mold_core::GenerationReference::Image {
            media: mold_core::GenerationReferenceAuthority::Descriptor,
            provenance: mold_core::GenerationReferenceProvenance {
                name: Some("anchor.png".to_string()),
                sha256: Some("0".repeat(64)),
            },
            mime_type: "image/png".to_string(),
            width: 24,
            height: 16,
        }];
        let req = build_generate_request(BuildParams {
            family: Some(mold_core::minimax_h3::FAMILY),
            references: Some(references),
            ..base_params("animate the anchor", mold_core::minimax_h3::REF2VA_COMFY)
        });
        assert!(matches!(
            req.references.as_deref(),
            Some([mold_core::GenerationReference::Image {
                provenance,
                width: 24,
                height: 16,
                ..
            }]) if provenance.name.as_deref() == Some("anchor.png")
        ));
    }

    #[test]
    fn h3_builder_preserves_only_sanitized_source_name() {
        let name = crate::h3_references::safe_name("../../closing-frame.png\nignored");
        assert!(name.is_none());
        let req = build_generate_request(BuildParams {
            family: Some(mold_core::minimax_h3::FAMILY),
            source_image: Some(vec![1, 2, 3]),
            source_image_name: crate::h3_references::safe_name("C:\\uploads\\opening-frame.png"),
            ..base_params("animate the frame", mold_core::minimax_h3::FL2VA_COMFY)
        });
        assert_eq!(req.source_image_name.as_deref(), Some("opening-frame.png"));
        assert!(!format!("{req:?}").contains("C:\\uploads"));
    }

    #[test]
    fn duration_rejects_ambiguous_or_unsupported_requests() {
        assert!(resolve_video_timing(Some("ltx2"), None, Some(97), None, Some(4.0)).is_err());
        assert!(resolve_video_timing(Some("ltx2"), None, None, None, Some(20.1)).is_err());
        assert!(resolve_video_timing(Some("flux"), None, None, None, Some(4.0)).is_err());
        assert!(resolve_video_timing(Some("ltx2"), None, None, None, Some(0.0)).is_err());
    }

    #[test]
    fn duration_respects_flat_family_frame_caps() {
        let defs = mold_core::ModelDefaults {
            default_fps: Some(30),
            max_frames: Some(257),
            frame_step: Some(8),
            ..defaults()
        };
        let message = resolve_video_timing(Some("ltx-video"), Some(&defs), None, None, Some(10.0))
            .unwrap_err();
        assert!(message.contains("maximum 8.6 seconds"));
    }

    #[test]
    fn ltx2_can_select_gif() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            video_format: Some(VideoFormat::Gif),
            ..base_params("waves", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.output_format, Some(OutputFormat::Gif));
    }

    #[test]
    fn video_frames_and_fps_pass_through() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            frames: Some(49),
            fps: Some(30),
            ..base_params("spinning cube", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.frames, Some(49));
        assert_eq!(req.fps, Some(30));
    }

    #[test]
    fn image_family_ignores_video_format() {
        // If someone passes video_format on a non-video model, we still return PNG
        // instead of silently producing an unsupported container.
        let req = build_generate_request(BuildParams {
            family: Some("flux"),
            video_format: Some(VideoFormat::Mp4),
            ..base_params("a cat", "flux-schnell:q8")
        });
        assert_eq!(req.output_format, Some(OutputFormat::Png));
        assert!(req.frames.is_none());
    }

    #[test]
    fn ltx2_audio_only_for_ltx2() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            audio: Some(true),
            ..base_params("storm at sea", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.enable_audio, Some(true));

        // LTX-Video doesn't support audio
        let req = build_generate_request(BuildParams {
            family: Some("ltx-video"),
            audio: Some(true),
            ..base_params("storm at sea", "ltx-video-0.9.6-distilled:bf16")
        });
        assert!(req.enable_audio.is_none());
    }

    #[test]
    fn ltx2_pipeline_plumbed_through() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            pipeline: Some(Ltx2PipelineMode::Distilled),
            ..base_params("a city timelapse", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.pipeline, Some(Ltx2PipelineMode::Distilled));

        // Pipeline is ignored for non-LTX-2 families (server validates this too)
        let req = build_generate_request(BuildParams {
            family: Some("ltx-video"),
            pipeline: Some(Ltx2PipelineMode::Distilled),
            ..base_params("a city timelapse", "ltx-video-0.9.6-distilled:bf16")
        });
        assert!(req.pipeline.is_none());
    }

    #[test]
    fn img2img_source_image_passes_through() {
        let bytes = vec![0x89, b'P', b'N', b'G', 0u8, 1, 2, 3];
        let req = build_generate_request(BuildParams {
            family: Some("flux"),
            source_image: Some(bytes.clone()),
            strength: Some(0.55),
            ..base_params("a cat", "flux-schnell:q8")
        });
        assert_eq!(req.source_image.as_ref(), Some(&bytes));
        assert!((req.strength - 0.55).abs() < 1e-9);
    }

    #[test]
    fn img2video_on_ltx2_routes_to_mp4() {
        // A source_image on an LTX-2 request is the I2V path (server treats it
        // as a first-frame keyframe).
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 1, 2, 3];
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            source_image: Some(bytes.clone()),
            ..base_params("dolly in", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.output_format, Some(OutputFormat::Mp4));
        assert_eq!(req.source_image.as_ref(), Some(&bytes));
        assert_eq!(req.frames, Some(25));
    }

    #[test]
    fn specialized_inputs_are_plumbed_to_the_request() {
        let audio = b"RIFFtestWAVE".to_vec();
        let video = b"fake-mp4".to_vec();
        let keyframes = vec![
            KeyframeCondition {
                frame: 0,
                image: vec![0x89, b'P', b'N', b'G'],
                name: None,
            },
            KeyframeCondition {
                frame: 24,
                image: vec![0xFF, 0xD8],
                name: None,
            },
        ];
        let range = TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.0,
        };
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            pipeline: Some(Ltx2PipelineMode::Retake),
            audio_file: Some(audio.clone()),
            source_video: Some(video.clone()),
            keyframes: Some(keyframes.clone()),
            retake_range: Some(range.clone()),
            ..base_params("replace this shot", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.audio_file, Some(audio));
        assert_eq!(req.source_video, Some(video));
        assert_eq!(req.keyframes.unwrap().len(), keyframes.len());
        assert_eq!(req.retake_range, Some(range));
        assert_eq!(req.pipeline, Some(Ltx2PipelineMode::Retake));
    }

    #[test]
    fn attachment_modes_are_unambiguous() {
        assert_eq!(
            specialized_pipeline(false, true, false, None, None, 0).unwrap(),
            Some(Ltx2PipelineMode::A2Vid)
        );
        assert_eq!(
            specialized_pipeline(false, false, true, Some(0.5), Some(1.5), 0).unwrap(),
            Some(Ltx2PipelineMode::Retake)
        );
        assert_eq!(
            specialized_pipeline(false, false, false, None, None, 2).unwrap(),
            Some(Ltx2PipelineMode::Keyframe)
        );
        assert!(specialized_pipeline(false, true, true, Some(0.0), Some(1.0), 0).is_err());
        assert!(specialized_pipeline(false, false, true, Some(0.0), None, 0).is_err());
        assert!(specialized_pipeline(false, false, false, None, None, 1).is_err());
        assert!(specialized_pipeline(true, false, true, Some(0.0), Some(1.0), 0).is_err());
        assert!(specialized_pipeline(true, false, false, None, None, 2).is_err());
        assert_eq!(
            specialized_pipeline(true, true, false, None, None, 0).unwrap(),
            Some(Ltx2PipelineMode::A2Vid),
            "upstream explicitly supports audio plus image conditioning"
        );
    }

    #[test]
    fn aggregate_media_cap_fits_the_server_json_limit_after_base64() {
        let encoded = MAX_INLINE_MEDIA_BYTES.div_ceil(3) * 4;
        assert!(encoded + 1024 * 1024 < 64 * 1024 * 1024);
        assert!(validate_inline_media_lengths([37 * 1024 * 1024, MAX_SOURCE_IMAGE_BYTES]).is_ok());
        assert!(validate_inline_media_lengths([47 * 1024 * 1024, MAX_SOURCE_IMAGE_BYTES]).is_err());
    }

    #[test]
    fn keyframes_are_spaced_across_the_requested_timeline() {
        let keyframes = build_keyframes(vec![vec![1], vec![2], vec![3]], 49);
        assert_eq!(
            keyframes.iter().map(|item| item.frame).collect::<Vec<_>>(),
            [0, 24, 48]
        );
    }

    #[test]
    fn generate_stays_within_discords_option_limit() {
        let command = generate();
        assert_eq!(command.parameters.len(), 25);
        assert!(command.parameters.len() <= 25);
        assert!(command
            .parameters
            .iter()
            .any(|parameter| parameter.name == "duration"));
        assert_eq!(command.parameters[23].name, "reference_1");
        assert_eq!(command.parameters[24].name, "reference_2");
    }

    #[test]
    fn video_family_requests_gif_preview() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            ..base_params("a cat", "ltx-2-19b-distilled:fp8")
        });
        assert!(req.gif_preview);

        let req = build_generate_request(BuildParams {
            family: Some("ltx-video"),
            ..base_params("a cat", "ltx-video-0.9.6-distilled:bf16")
        });
        assert!(req.gif_preview);
    }

    #[test]
    fn image_family_does_not_request_gif_preview() {
        let req = build_generate_request(base_params("a cat", "flux-schnell:q8"));
        assert!(!req.gif_preview);
    }

    #[test]
    fn pipeline_choice_maps_to_supported_modes_only() {
        // Round-trip every exposed variant and assert it maps to an
        // Ltx2PipelineMode the slash command can actually satisfy.
        assert_eq!(
            PipelineChoice::OneStage.to_mode(),
            Ltx2PipelineMode::OneStage
        );
        assert_eq!(
            PipelineChoice::TwoStage.to_mode(),
            Ltx2PipelineMode::TwoStage
        );
        assert_eq!(
            PipelineChoice::TwoStageHq.to_mode(),
            Ltx2PipelineMode::TwoStageHq
        );
        assert_eq!(
            PipelineChoice::Distilled.to_mode(),
            Ltx2PipelineMode::Distilled
        );
    }

    // --- attachment dims ---

    fn defaults_1024() -> mold_core::ModelDefaults {
        mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 3.5,
            default_width: 1024,
            default_height: 1024,
            description: String::new(),
            ..Default::default()
        }
    }

    #[test]
    fn fit_attachment_dims_preserves_aspect_ratio() {
        let d = defaults_1024();
        // Landscape photo is no longer squashed to a square.
        assert_eq!(
            fit_attachment_dims(1920, 1080, Some(&d), Some("flux")),
            (1024, 576)
        );
        // Portrait keeps its orientation.
        assert_eq!(
            fit_attachment_dims(1080, 1920, Some(&d), Some("flux")),
            (576, 1024)
        );
        // Square source fills the model's native square.
        assert_eq!(
            fit_attachment_dims(1000, 1000, Some(&d), Some("flux")),
            (1024, 1024)
        );
        // Missing defaults fall back to a 1024x1024 canvas.
        assert_eq!(
            fit_attachment_dims(1920, 1080, None, Some("flux")),
            (1024, 576)
        );
        // Output always lands on the 16px grid.
        let (w, h) = fit_attachment_dims(777, 513, Some(&d), Some("flux"));
        assert_eq!((w % 16, h % 16), (0, 0));
    }

    /// The bot fits on the grid `/api/models` advertises per model:
    /// `wan22-ti2v-5b` reports 32, so its I2V sources must land on /32
    /// instead of the family-wide /16 that queued canvases the engine then
    /// rejected after a 10 GB load.
    #[test]
    fn fit_attachment_dims_honors_advertised_32px_grid() {
        // 1617x1000 into a 1280x704 canvas is height-limited:
        // h=704, w=704*1.617=1138.4 — which floors differently per grid.
        let five_b = mold_core::ModelDefaults {
            default_width: 1280,
            default_height: 704,
            dimension_alignment: Some(32),
            ..defaults_1024()
        };
        assert_eq!(
            fit_attachment_dims(1617, 1000, Some(&five_b), Some("wan")),
            (1120, 704)
        );

        let fourteen_b = mold_core::ModelDefaults {
            default_width: 1280,
            default_height: 704,
            dimension_alignment: Some(16),
            ..defaults_1024()
        };
        assert_eq!(
            fit_attachment_dims(1617, 1000, Some(&fourteen_b), Some("wan")),
            (1136, 704)
        );

        // Older servers omit the field; the compatible 16px grid remains.
        let unadvertised = mold_core::ModelDefaults {
            default_width: 1280,
            default_height: 704,
            ..defaults_1024()
        };
        assert_eq!(
            fit_attachment_dims(1617, 1000, Some(&unadvertised), Some("wan")),
            (1136, 704)
        );
    }

    /// The cold-cache manifest fallback must carry the same per-model grid
    /// the live `/api/models` advertisement would.
    #[test]
    fn manifest_fallback_defaults_carry_the_models_grid() {
        let five_b = mold_core::manifest::find_manifest("wan22-ti2v-5b:fp16")
            .expect("wan22-ti2v-5b:fp16 is a manifest model");
        assert_eq!(defaults_from_manifest(five_b).dimension_alignment, Some(32));

        let one_three_b = mold_core::manifest::find_manifest("wan21-t2v-1.3b:bf16")
            .expect("wan21-t2v-1.3b:bf16 is a manifest model");
        assert_eq!(
            defaults_from_manifest(one_three_b).dimension_alignment,
            Some(16)
        );
    }

    #[test]
    fn h3_attachment_dims_use_the_official_32px_canvas_authority() {
        for (source_width, source_height) in [(1920, 1080), (1080, 1920)] {
            let (width, height) = fit_attachment_dims(
                source_width,
                source_height,
                Some(&defaults_1024()),
                Some("minimax_h3"),
            );
            assert_eq!((width % 32, height % 32), (0, 0));
            assert_eq!(
                (width, height),
                mold_core::minimax_h3::recommended_dimensions(source_width, source_height)
            );
        }
    }
    // --- autocomplete ranking ---

    fn mk_model(name: &str, family: &str, downloaded: bool) -> ModelInfoExtended {
        ModelInfoExtended {
            info: mold_core::ModelInfo {
                name: name.to_string(),
                family: family.to_string(),
                size_gb: 1.0,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: defaults(),
            downloaded,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
        }
    }

    #[test]
    fn rank_prefers_downloaded_then_undownloaded() {
        let models = vec![
            mk_model("flux-schnell:q8", "flux", true),
            mk_model("flux-dev:q4", "flux", false),
            mk_model("ltx-2-19b-distilled:fp8", "ltx2", true),
        ];
        let out = rank_model_suggestions(&models, &[], "");
        assert_eq!(
            out,
            vec![
                "flux-schnell:q8".to_string(),
                "ltx-2-19b-distilled:fp8".to_string(),
                "flux-dev:q4".to_string(),
            ]
        );
    }

    #[test]
    fn rank_filters_by_partial_case_insensitive() {
        let models = vec![
            mk_model("flux-schnell:q8", "flux", true),
            mk_model("flux-dev:q4", "flux", true),
            mk_model("ltx-2-19b-distilled:fp8", "ltx2", true),
        ];
        let out = rank_model_suggestions(&models, &[], "FLUX");
        assert!(out.iter().all(|n| n.to_lowercase().contains("flux")));
        assert!(out.contains(&"flux-schnell:q8".to_string()));
        assert!(!out.contains(&"ltx-2-19b-distilled:fp8".to_string()));
    }

    #[test]
    fn rank_caps_at_discord_limit() {
        let models: Vec<_> = (0..40)
            .map(|i| mk_model(&format!("model-{i}:q8"), "flux", true))
            .collect();
        let out = rank_model_suggestions(&models, &[], "");
        assert_eq!(out.len(), 25);
    }

    #[test]
    fn rank_falls_back_to_manifest_when_cache_cold() {
        // Simulate bot just started: cache empty, rely on static manifest.
        let fallback = vec!["flux-schnell:q8", "ltx-2-19b-distilled:fp8", "sd15:fp16"];
        let out = rank_model_suggestions(&[], &fallback, "ltx");
        assert_eq!(out, vec!["ltx-2-19b-distilled:fp8".to_string()]);
    }

    #[test]
    fn rank_empty_partial_returns_everything_within_limit() {
        let models: Vec<_> = (0..5)
            .map(|i| mk_model(&format!("m{i}:q8"), "flux", true))
            .collect();
        let out = rank_model_suggestions(&models, &[], "");
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn manifest_fallback_is_non_empty_and_excludes_utilities() {
        let names = manifest_fallback_names();
        assert!(!names.is_empty(), "manifest should supply fallback names");
        assert!(
            !names.iter().any(|n| n.starts_with("qwen3-expand")),
            "utility models should be excluded from suggestions: {names:?}"
        );
    }

    #[test]
    fn manifest_fallback_preserves_video_timing_contract() {
        let manifest = mold_core::manifest::find_manifest("ltx-2-19b-distilled:fp8").unwrap();
        let defaults = defaults_from_manifest(manifest);
        assert_eq!(manifest.family, "ltx2");
        assert_eq!(defaults.default_frames, Some(97));
        assert_eq!(defaults.default_fps, Some(24));
        assert_eq!(defaults.max_runtime_seconds, Some(20));
        assert_eq!(defaults.frame_step, Some(8));
    }

    // --- Header sniff ---

    #[test]
    fn looks_like_png_or_jpeg_sniffs_magic_bytes() {
        assert!(looks_like_png_or_jpeg(&[
            0x89, 0x50, 0x4E, 0x47, 0, 0, 0, 0
        ]));
        assert!(looks_like_png_or_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!looks_like_png_or_jpeg(&[]));
        assert!(!looks_like_png_or_jpeg(&[0x47, 0x49, 0x46, 0x38])); // GIF89a
        assert!(!looks_like_png_or_jpeg(&[0x00, 0x00, 0x00, 0x20])); // MP4-ish
    }

    // --- video_family / family_for_model ---

    #[test]
    fn video_family_classification() {
        assert_eq!(video_family("ltx-video"), Some("ltx-video"));
        assert_eq!(video_family("ltx2"), Some("ltx2"));
        assert_eq!(video_family("wan"), Some("wan"));
        for alias in mold_core::minimax_h3::FAMILY_ALIASES {
            assert_eq!(
                video_family(alias),
                Some(mold_core::minimax_h3::FAMILY),
                "alias {alias}"
            );
        }
        assert!(video_family("flux").is_none());
        assert!(video_family("sdxl").is_none());
    }

    /// This one predicate gates everything Discord does with a video model —
    /// the MP4 output format, the frame/fps defaults, the GIF preview, and
    /// whether `/generate duration:` is accepted at all. Without the wan arm
    /// the bot renders wan as a still image and rejects duration outright.
    #[test]
    fn wan_duration_resolves_onto_its_own_four_frame_grid() {
        // No model defaults: the resolver falls back to the shared Rust
        // helpers, which is the path a bot with a stale model cache takes.
        let (frames, fps) =
            resolve_video_timing(Some("wan"), None, None, Some(16), Some(3.0)).unwrap();
        let frames = frames.expect("wan is a video family");
        assert_eq!(fps, Some(16));
        // 3s x 16fps = 48 -> the nearest 4k+1 value. An 8k+1 grid would have
        // produced 49 here, which the server rejects for wan.
        assert_eq!(frames, 49);
        assert_eq!((frames - 1) % 4, 0);

        // Wan's ceiling is a flat frame count, not a seconds budget, so the
        // duration it permits is derived from 257 frames and therefore moves
        // with fps: ~16.1s at 16 fps, half that at 32. The refusal quotes the
        // derived figure rather than silently trimming the request.
        let error =
            resolve_video_timing(Some("wan"), None, None, Some(16), Some(600.0)).unwrap_err();
        assert!(error.contains("16.1 seconds"), "{error}");
        let error =
            resolve_video_timing(Some("wan"), None, None, Some(32), Some(600.0)).unwrap_err();
        assert!(error.contains("8.0 seconds"), "{error}");

        // An image family still refuses duration entirely.
        assert!(resolve_video_timing(Some("flux"), None, None, None, Some(3.0)).is_err());
    }

    #[test]
    fn family_for_model_lookup() {
        let models = vec![ModelInfoExtended {
            info: mold_core::ModelInfo {
                name: "ltx-2-19b-distilled:fp8".to_string(),
                family: "ltx2".to_string(),
                size_gb: 19.0,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: defaults(),
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
        }];
        assert_eq!(
            family_for_model(&models, "ltx-2-19b-distilled:fp8"),
            Some("ltx2")
        );
        assert!(family_for_model(&models, "unknown").is_none());
    }

    // --- resolve_default_model (unchanged behavior) ---

    #[test]
    fn resolve_default_prefers_loaded() {
        let models = vec![
            ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux-dev:q4".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.0,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: defaults(),
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            },
            ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux2-klein:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.5,
                    is_loaded: true,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: defaults(),
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            },
        ];
        assert_eq!(resolve_default_model(&models), "flux2-klein:q8");
    }

    #[test]
    fn resolve_default_falls_back_to_downloaded() {
        let models = vec![ModelInfoExtended {
            info: mold_core::ModelInfo {
                name: "flux-dev:q4".to_string(),
                family: "flux".to_string(),
                size_gb: 4.0,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: defaults(),
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
        }];
        assert_eq!(resolve_default_model(&models), "flux-dev:q4");
    }

    #[test]
    fn resolve_default_empty_list() {
        assert_eq!(resolve_default_model(&[]), "flux2-klein:q8");
    }

    #[test]
    fn resolve_default_skips_utility_models() {
        let models = vec![
            ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "qwen3-expand:q8".to_string(),
                    family: "qwen3-expand".to_string(),
                    size_gb: 1.8,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: defaults(),
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            },
            ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux2-klein:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.5,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: defaults(),
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            },
        ];
        assert_eq!(resolve_default_model(&models), "flux2-klein:q8");
    }

    #[test]
    fn resolve_default_skips_controlnet() {
        let models = vec![
            ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "controlnet-canny-sd15:fp16".to_string(),
                    family: "controlnet".to_string(),
                    size_gb: 0.7,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: defaults(),
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            },
            ModelInfoExtended {
                info: mold_core::ModelInfo {
                    name: "flux2-klein:q8".to_string(),
                    family: "flux".to_string(),
                    size_gb: 4.5,
                    is_loaded: false,
                    last_used: None,
                    hf_repo: "test/repo".to_string(),
                },
                defaults: defaults(),
                downloaded: true,
                disk_usage_bytes: None,
                remaining_download_bytes: None,
                display_name: None,
                kind: None,
                modality: None,
                nsfw: None,
                supports_audio: None,
                supports_extend: None,
                supports_sequence: None,
                extend_default_overlap_frames: None,
                guidance_capabilities: None,
                source_image: None,
            },
        ];
        assert_eq!(resolve_default_model(&models), "flux2-klein:q8");
    }

    #[test]
    fn prompt_rejected_up_front_only_without_a_source_image() {
        // No attachment: an absent or blank prompt is rejected before the
        // interaction is deferred.
        assert!(prompt_missing_before_defer(None, false));
        assert!(prompt_missing_before_defer(Some("   "), false));
        assert!(!prompt_missing_before_defer(Some("a cat"), false));

        // With a source image the run may be unprompted — LTX-2 image-to-video
        // conditions on the frame. Non-video families are still rejected, by
        // the server's family-aware validator.
        assert!(!prompt_missing_before_defer(None, true));
        assert!(!prompt_missing_before_defer(Some("  "), true));
        assert!(!prompt_missing_before_defer(Some("a cat"), true));
    }

    #[test]
    fn absent_prompt_builds_an_empty_prompt_request() {
        let req = build_generate_request(BuildParams {
            source_image: Some(vec![0x89, 0x50, 0x4E, 0x47]),
            ..base_params("", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.prompt, "");
    }
}
