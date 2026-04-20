use crate::checks::{self, AuthResult};
use crate::handler;
use crate::state::Context;
use anyhow::Result;
use mold_core::{GenerateRequest, Ltx2PipelineMode, ModelInfoExtended, OutputFormat};
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

/// Slash-command facing LTX-2 pipeline selector. Only pipelines that are
/// fully satisfiable from the Discord command surface are exposed — modes
/// like `a2vid` / `retake` / `ic-lora` / `keyframe` require extra inputs
/// (`audio_file`, `source_video`, LoRA stacks, ≥2 keyframes) that the slash
/// command doesn't collect, and server validation would reject them outright.
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

/// Classify a model family: returns `Some("ltx-video")`, `Some("ltx2")`, or
/// `None` for anything that isn't a recognized video-producing family.
pub fn video_family(family: &str) -> Option<&str> {
    match family {
        "ltx-video" | "ltx2" => Some(family),
        _ => None,
    }
}

/// Look up the family for a resolved model name from the bot's cached models.
pub fn family_for_model<'a>(models: &'a [ModelInfoExtended], name: &str) -> Option<&'a str> {
    models
        .iter()
        .find(|m| m.info.name == name)
        .map(|m| m.info.family.as_str())
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
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub strength: Option<f64>,
    pub video_format: Option<VideoFormat>,
    pub audio: Option<bool>,
    pub pipeline: Option<Ltx2PipelineMode>,
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

    // Video families always deliver MP4 or GIF. Image models stay PNG.
    let output_format = if is_video_family {
        params
            .video_format
            .map(VideoFormat::to_output_format)
            .unwrap_or(OutputFormat::Mp4)
    } else {
        OutputFormat::Png
    };

    // Default to a sensible frame count for video models when the user didn't
    // specify one: 25 frames (8n+1) ≈ 1 second at 24 FPS.
    let frames = if is_video_family {
        Some(params.frames.unwrap_or(25))
    } else {
        params.frames
    };
    let fps = if is_video_family {
        Some(params.fps.unwrap_or(24))
    } else {
        params.fps
    };

    // LTX-2 audio only flows through MP4 container.
    let enable_audio = if is_ltx2 { params.audio } else { None };

    GenerateRequest {
        prompt: params.prompt.to_string(),
        negative_prompt: params.negative_prompt.map(|s| s.to_string()),
        model: params.model.to_string(),
        width: params.width.unwrap_or(def_w),
        height: params.height.unwrap_or(def_h),
        steps: params.steps.unwrap_or(def_steps),
        guidance: params.guidance.unwrap_or(def_guidance),
        seed: params.seed,
        batch_size: 1,
        output_format,
        embed_metadata: None,
        scheduler: None,
        edit_images: None,
        source_image: params.source_image,
        strength: params.strength.unwrap_or(0.75),
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames,
        fps,
        upscale_model: None,
        // Always request a GIF preview for video jobs: if the primary MP4
        // exceeds Discord's upload ceiling we fall back to the preview so the
        // user still sees the generation.
        gif_preview: is_video_family,
        enable_audio,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: if is_ltx2 { params.pipeline } else { None },
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
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

/// Snap `(w, h)` to multiples of 16 (FLUX patch / LTX tile constraint) while
/// keeping the aspect ratio roughly intact and staying inside `[16, max_dim]`.
/// Used to derive img2img dimensions from a Discord attachment's reported
/// width/height when the user didn't supply explicit values — otherwise the
/// server would `resize_exact` a landscape photo into a square default.
pub fn snap_dims_to_multiple_of_16(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    let clamp = |v: u32| -> u32 {
        let capped = v.min(max_dim.max(16));
        let rounded = ((capped + 8) / 16) * 16;
        rounded.max(16)
    };
    (clamp(w), clamp(h))
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

/// Generate an image (PNG) or video (MP4 by default, GIF on request).
#[allow(clippy::too_many_arguments)]
#[poise::command(slash_command)]
pub async fn generate(
    ctx: Context<'_>,
    #[description = "Text prompt describing the image or video to generate"] prompt: String,
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
    #[description = "Video FPS (video models only, default 24)"] fps: Option<u32>,
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
) -> Result<()> {
    // Validate prompt before deferring (avoids wasting the interaction)
    if prompt.trim().is_empty() {
        ctx.send(
            poise::CreateReply::default()
                .content("Prompt cannot be empty.")
                .ephemeral(true),
        )
        .await?;
        return Ok(());
    }

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
    let model_defaults = model_entry.map(|m| &m.defaults);
    let family = model_entry.map(|m| m.info.family.as_str());

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
                    let max_dim = model_defaults
                        .map(|d| d.default_width.max(d.default_height))
                        .unwrap_or(1024);
                    let (sw, sh) = snap_dims_to_multiple_of_16(w, h, max_dim);
                    (Some(sw), Some(sh))
                }
                _ => (width, height),
            }
        } else {
            (width, height)
        };

    let req = build_generate_request(BuildParams {
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
        frames,
        fps,
        strength,
        video_format,
        audio,
        pipeline: pipeline.map(PipelineChoice::to_mode),
    });

    match handler::run_generation(ctx, req).await {
        Ok(()) => {
            // Quota slot was already consumed atomically in check_generate_auth
            ctx.data().cooldowns.record(user_id);
        }
        Err(e) => {
            // Refund the quota slot consumed during auth check
            ctx.data().quotas.refund(user_id);
            let msg = if mold_core::MoldClient::is_connection_error(&e) {
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
        }
    }

    fn base_params<'a>(prompt: &'a str, model: &'a str) -> BuildParams<'a> {
        BuildParams {
            prompt,
            model,
            ..BuildParams::default()
        }
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
        assert_eq!(req.output_format, OutputFormat::Png);
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
        assert_eq!(req.output_format, OutputFormat::Png);
    }

    #[test]
    fn build_request_partial_overrides() {
        let defs = mold_core::ModelDefaults {
            default_steps: 4,
            default_guidance: 0.0,
            default_width: 512,
            default_height: 512,
            description: "test".to_string(),
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
        assert_eq!(req.output_format, OutputFormat::Mp4);
        assert_eq!(req.frames, Some(25));
        assert_eq!(req.fps, Some(24));
    }

    #[test]
    fn ltx2_defaults_to_mp4() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            ..base_params("panning over a beach", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.output_format, OutputFormat::Mp4);
        assert_eq!(req.frames, Some(25));
    }

    #[test]
    fn ltx2_can_select_gif() {
        let req = build_generate_request(BuildParams {
            family: Some("ltx2"),
            video_format: Some(VideoFormat::Gif),
            ..base_params("waves", "ltx-2-19b-distilled:fp8")
        });
        assert_eq!(req.output_format, OutputFormat::Gif);
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
        assert_eq!(req.output_format, OutputFormat::Png);
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
        assert_eq!(req.output_format, OutputFormat::Mp4);
        assert_eq!(req.source_image.as_ref(), Some(&bytes));
        assert_eq!(req.frames, Some(25));
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

    // --- snap_dims ---

    #[test]
    fn snap_dims_rounds_to_multiples_of_16() {
        assert_eq!(snap_dims_to_multiple_of_16(1000, 1000, 1024), (1008, 1008));
        assert_eq!(snap_dims_to_multiple_of_16(1920, 1080, 1024), (1024, 1024));
        assert_eq!(snap_dims_to_multiple_of_16(777, 513, 1024), (784, 512));
        // Under the multiple-of-16 floor
        assert_eq!(snap_dims_to_multiple_of_16(3, 5, 1024), (16, 16));
        // Landscape aspect preserved roughly
        let (w, h) = snap_dims_to_multiple_of_16(800, 600, 1024);
        assert_eq!((w, h), (800, 608));
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
        assert!(video_family("flux").is_none());
        assert!(video_family("sdxl").is_none());
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
            },
        ];
        assert_eq!(resolve_default_model(&models), "flux2-klein:q8");
    }
}
