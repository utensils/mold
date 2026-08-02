use anyhow::Result;
#[cfg(any(feature = "preview", test))]
use base64::{engine::general_purpose, Engine as _};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{
    classify_generate_error, fit_to_model_dimensions, fit_to_target_area, manifest, Config,
    DevicePlacement, GenerateRequest, GenerateResponse, GenerateServerAction, ImageData,
    KeyframeCondition, LoraWeight, Ltx2GuidanceOverrides, Ltx2PipelineMode, Ltx2SpatialUpscale,
    Ltx2TemporalUpscale, MoldClient, OutputFormat, Scheduler, TimeRange,
};
use rand::Rng;
#[cfg(feature = "preview")]
use std::io::IsTerminal;
use std::io::Write;
use std::time::Duration;

use crate::control::{stream_server_pull, CliContext};
use crate::errors::RemoteInferenceError;
use crate::output::{is_piped, status};
use crate::theme;
use crate::ui::{print_server_pull_missing_model, print_using_local_inference, render_progress};

/// Tag an error as originating from the remote server so the top-level handler
/// can produce a remote-context diagnostic (e.g. distinguishing a server-side
/// CUDA OOM from a local Metal OOM on the client).
fn tag_remote(client: &MoldClient, e: anyhow::Error) -> anyhow::Error {
    RemoteInferenceError::wrap(client.host(), e)
}

/// Fit source image dimensions to the model's native resolution, preserving aspect ratio.
fn source_image_model_dimensions(bytes: &[u8], model_w: u32, model_h: u32) -> Result<(u32, u32)> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode source image: {e}"))?;
    let orig_w = img.width();
    let orig_h = img.height();
    let (w, h) = fit_to_model_dimensions(orig_w, orig_h, model_w, model_h);
    if w != orig_w || h != orig_h {
        let is_upscale = w > orig_w || h > orig_h;
        let icon = if is_upscale {
            theme::icon_info()
        } else {
            theme::icon_warn()
        };
        status!(
            "{} Source image {}x{} -> {}x{} (fit to {}x{} model bounds, 16px aligned)",
            icon,
            orig_w,
            orig_h,
            w,
            h,
            model_w,
            model_h
        );
    }
    Ok((w, h))
}

fn qwen_image_edit_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    const TARGET_AREA: u32 = 1024 * 1024;
    const ALIGN: u32 = 16;

    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode source image: {e}"))?;
    let orig_w = img.width().max(1);
    let orig_h = img.height().max(1);
    let (width, height) = fit_to_target_area(orig_w, orig_h, TARGET_AREA, ALIGN);

    if width != orig_w || height != orig_h {
        status!(
            "{} Edit image {}x{} -> {}x{} (target ~1024x1024 area, 16px aligned)",
            theme::icon_info(),
            orig_w,
            orig_h,
            width,
            height
        );
    }

    Ok((width, height))
}

fn resolve_family(model: &str, config: &Config) -> Option<String> {
    config
        .resolved_model_config(model)
        .family
        .or_else(|| manifest::find_manifest(model).map(|m| m.family.clone()))
}

/// Validate a request on the forced-local path.
///
/// Feeds the config/manifest-resolved family through as a hint so `cv:` / `hf:`
/// catalog IDs keep their family-gated features (audio, keyframes, retake,
/// optional prompts, …) instead of failing with `unknown model family`. This
/// mirrors what the HTTP server does with its catalog family hint.
#[cfg(any(feature = "cuda", feature = "metal", test))]
fn validate_local_request(req: &GenerateRequest, config: &Config) -> Result<()> {
    mold_core::validate_generate_request_with_family(
        req,
        resolve_family(&req.model, config).as_deref(),
    )
    .map_err(|e| anyhow::anyhow!(e))
}

fn effective_dimensions(
    config: &Config,
    model_cfg: &mold_core::ModelConfig,
    family: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    source_image: Option<&[u8]>,
    edit_images: Option<&[Vec<u8>]>,
) -> Result<(u32, u32)> {
    match (width, height, source_image) {
        (Some(width), Some(height), _) => Ok((width, height)),
        (Some(width), None, _) => Ok((width, model_cfg.effective_height(config))),
        (None, Some(height), _) => Ok((model_cfg.effective_width(config), height)),
        (None, None, _) if family == Some("qwen-image-edit") => {
            let first = edit_images
                .and_then(|images| images.first())
                .ok_or_else(|| {
                    anyhow::anyhow!("qwen-image-edit requires at least one input image")
                })?;
            qwen_image_edit_dimensions(first)
        }
        (None, None, Some(source_image)) => {
            let model_w = model_cfg.effective_width(config);
            let model_h = model_cfg.effective_height(config);
            source_image_model_dimensions(source_image, model_w, model_h)
        }
        (None, None, None) => Ok((
            model_cfg.effective_width(config),
            model_cfg.effective_height(config),
        )),
    }
}

fn effective_negative_prompt(
    family: Option<&str>,
    guidance: f64,
    negative_prompt: Option<String>,
) -> Option<String> {
    if family == Some("qwen-image-edit") && guidance > 1.0 && negative_prompt.is_none() {
        // Keep CFG's negative branch explicitly populated for Qwen edit models.
        Some(" ".to_string())
    } else {
        negative_prompt
    }
}

fn validate_cli_batch_for_family(family: Option<&str>, batch: u32) -> Result<()> {
    if family == Some("qwen-image-edit") && batch > 1 {
        anyhow::bail!(
            "qwen-image-edit only supports --batch 1. Run separate commands for multiple edits."
        );
    }
    Ok(())
}

fn validate_batch_stdout(batch: u32, piped: bool, output: &Option<String>) -> Result<()> {
    if batch > 1 {
        let stdout_output = (piped && output.is_none()) || output.as_deref() == Some("-");
        if stdout_output {
            anyhow::bail!(
                "--batch with more than 1 output is not supported with stdout output. \
                 Use --output <path> to save batch outputs to separate files."
            );
        }
    }
    Ok(())
}

/// Re-derive request defaults after an auto-pull refreshed the model
/// config. `cli_*` carry the user's explicit flags — `None` means the
/// field was defaulted at request-build time and must now track the
/// refreshed model defaults. Reuses `effective_dimensions`, so img2img
/// sources are re-fit to the refreshed canvas and qwen-image-edit sizes
/// from its first edit image.
#[cfg(any(feature = "cuda", feature = "metal", test))]
#[allow(clippy::too_many_arguments)]
fn refit_request_after_pull(
    req: &mut GenerateRequest,
    config: &Config,
    model_cfg: &mold_core::ModelConfig,
    family: Option<&str>,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<()> {
    if cli_width.is_none() && cli_height.is_none() {
        let (w, h) = effective_dimensions(
            config,
            model_cfg,
            family,
            None,
            None,
            req.source_image.as_deref(),
            req.edit_images.as_deref(),
        )?;
        req.width = w;
        req.height = h;
    } else {
        if cli_width.is_none() {
            req.width = model_cfg.effective_width(config);
        }
        if cli_height.is_none() {
            req.height = model_cfg.effective_height(config);
        }
    }
    if cli_steps.is_none() {
        req.steps = model_cfg.effective_steps(config);
    }
    if cli_guidance.is_none() {
        req.guidance = model_cfg.effective_guidance();
    }
    Ok(())
}

pub struct Ltx2Options {
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    /// Per-clip cap for chained rendering. `None` = use the model-family default
    /// (currently 97 for LTX-2 distilled). Only read when `frames > cap`.
    pub clip_frames: Option<u32>,
    /// Motion-tail overlap between chained clips (pixel frames).
    pub motion_tail: u32,
    pub enable_audio: Option<bool>,
    pub audio_file: Option<Vec<u8>>,
    pub source_video: Option<Vec<u8>>,
    /// Existing video to continue. Makes the request a continuation: the
    /// delivered output is this clip followed by the newly rendered frames.
    pub extend_video: Option<Vec<u8>>,
    /// Pixel-frame overlap the continuation conditions on. `None` uses the
    /// shared chain motion-tail default.
    pub extend_overlap_frames: Option<u32>,
    pub keyframes: Option<Vec<KeyframeCondition>>,
    pub pipeline: Option<Ltx2PipelineMode>,
    pub ic_lora_control: Option<String>,
    pub loras: Option<Vec<LoraWeight>>,
    pub retake_range: Option<TimeRange>,
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
    /// Advanced multimodal-guider overrides. `None` keeps every pipeline
    /// constant, which is what makes existing seeds reproducible.
    pub guidance_overrides: Option<Ltx2GuidanceOverrides>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    prompt: &str,
    model: &str,
    output: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    batch: u32,
    ltx2: Ltx2Options,
    host: Option<String>,
    format: OutputFormat,
    no_metadata: bool,
    preview: bool,
    local: bool,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    scheduler: Option<Scheduler>,
    cfg_plus: Option<bool>,
    eager: bool,
    offload: bool,
    placement: Option<DevicePlacement>,
    source_image: Option<Vec<u8>>,
    edit_images: Option<Vec<Vec<u8>>>,
    strength: f64,
    mask_image: Option<Vec<u8>>,
    control_image: Option<Vec<u8>>,
    control_model: Option<String>,
    control_scale: f64,
    negative_prompt: Option<String>,
    original_prompt: Option<String>,
    batch_prompts: Option<Vec<String>>,
    lora: Option<LoraWeight>,
    expand: Option<bool>,
) -> Result<()> {
    let Ltx2Options {
        frames,
        fps,
        clip_frames,
        motion_tail,
        enable_audio,
        audio_file,
        source_video,
        extend_video,
        extend_overlap_frames,
        keyframes,
        pipeline,
        ic_lora_control,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        guidance_overrides,
    } = ltx2;

    // Load config and pull model-specific defaults.
    let ctx = CliContext::new(host.as_deref());
    let mut config = ctx.config().clone();
    // Catalog bridge: when the user passed `cv:<id>` / `hf:<author>/<name>`,
    // resolve it (sidecar-first, then live) and inject the synthesized
    // `ModelConfig` into `config.models` so the downstream
    // `ModelPaths::resolve` and engine factory accept it. A no-op for
    // non-catalog model names. Catalog lookup failures now surface here
    // rather than being swallowed and re-reported as a generic "unknown
    // model" downstream.
    crate::catalog_bridge::ensure_catalog_model(&mut config, model).await?;
    // `--no-metadata` is an opt-out override, so we only pass `Some(false)` when set.
    // Otherwise we defer to env/config/default precedence inside Config.
    let embed_metadata = config.effective_embed_metadata(no_metadata.then_some(false));
    let model_cfg = config.resolved_model_config(model);
    let family = resolve_family(model, &config);
    let effective_frames = frames.or_else(|| model_cfg.effective_frames());
    let effective_fps = fps.or_else(|| model_cfg.effective_fps());
    let is_ltx2 = family.as_deref() == Some("ltx2");
    validate_cli_batch_for_family(family.as_deref(), batch)?;

    // Default video models to a sensible container unless the user explicitly picked one.
    let output_format = if format == OutputFormat::Png && effective_frames.is_some() {
        if is_ltx2 {
            OutputFormat::Mp4
        } else {
            OutputFormat::Apng
        }
    } else {
        format
    };

    // ── Chain routing ─────────────────────────────────────────────────────
    // When --frames exceeds the per-clip cap, auto-build a ChainRequest and
    // delegate to the chain helper. Only LTX-2 distilled is chainable in v1;
    // other video families error fast rather than silently over-producing.
    {
        use super::chain::{decide_chain_routing, warn_if_clamped, ChainRoutingDecision};
        let routing_fps = effective_fps.unwrap_or(mold_core::validation::LTX2_DEFAULT_FPS);
        let decision = decide_chain_routing(
            effective_frames,
            family.as_deref(),
            model,
            clip_frames,
            motion_tail,
            routing_fps,
        );
        match decision {
            ChainRoutingDecision::SingleClip => {
                // Fall through to the existing single-clip path below.
            }
            ChainRoutingDecision::Rejected { reason } => {
                anyhow::bail!(reason);
            }
            ChainRoutingDecision::Chain {
                clip_frames: cf,
                motion_tail: mt,
            } => {
                warn_if_clamped(
                    clip_frames,
                    family
                        .as_deref()
                        .and_then(|family| {
                            mold_core::validation::max_frames_for_family_at_fps(family, routing_fps)
                        })
                        .unwrap_or(super::chain::LTX2_DEFAULT_CLIP_FRAMES),
                );
                let (eff_w, eff_h) = effective_dimensions(
                    &config,
                    &model_cfg,
                    family.as_deref(),
                    width,
                    height,
                    source_image.as_deref(),
                    edit_images.as_deref(),
                )?;
                let eff_steps = steps.unwrap_or_else(|| model_cfg.effective_steps(&config));
                let eff_guidance = guidance.unwrap_or_else(|| model_cfg.effective_guidance());
                let eff_fps = effective_fps.unwrap_or(24);
                let total_frames = effective_frames
                    .expect("decide_chain_routing only returns Chain when frames is Some");

                // Chain path doesn't use batch/edit_images/mask/control/loras —
                // those are single-clip concepts. If the user set them, warn and
                // continue (we don't hard-error to keep the UX lenient).
                if batch > 1 {
                    status!(
                        "{} --batch has no effect in chain mode; rendering a single stitched video",
                        theme::icon_warn(),
                    );
                }

                let inputs = super::chain::ChainInputs {
                    prompt: prompt.to_string(),
                    model: model.to_string(),
                    width: eff_w,
                    height: eff_h,
                    steps: eff_steps,
                    guidance: eff_guidance,
                    strength,
                    seed,
                    fps: eff_fps,
                    output_format,
                    total_frames,
                    clip_frames: cf,
                    motion_tail: mt,
                    source_image: source_image.clone(),
                    placement: placement.clone(),
                    enable_audio,
                    original_prompt: original_prompt.clone(),
                    batch_id: None,
                    batch_index: None,
                    batch_count: None,
                };
                // Guidance overrides are a single-clip concept: chain stages
                // render through their own pipeline constants, so say so
                // rather than pretend the flags took effect.
                if guidance_overrides.is_some() {
                    status!(
                        "{} guidance overrides have no effect in chain mode; each clip keeps its pipeline defaults",
                        theme::icon_warn(),
                    );
                }
                // Consume otherwise-unused LTX-2 knobs that chain v1 ignores so
                // clippy doesn't fire `unused_variables` on the early return.
                let _ = (
                    &guidance_overrides,
                    &audio_file,
                    &source_video,
                    &keyframes,
                    &pipeline,
                    &loras,
                    &retake_range,
                    &spatial_upscale,
                    &temporal_upscale,
                    &mask_image,
                    &control_image,
                    &control_model,
                    control_scale,
                    &negative_prompt,
                    &original_prompt,
                    &batch_prompts,
                    &lora,
                    &scheduler,
                    expand,
                );
                // Expand the single-prompt inputs into a canonical
                // ChainRequest and normalise before handing off to run_chain.
                let chain_req = inputs.to_chain_request().normalise()?;
                return super::chain::run_chain(
                    chain_req,
                    host,
                    output,
                    no_metadata,
                    preview,
                    local,
                    gpus,
                    t5_variant,
                    qwen3_variant,
                    qwen2_variant,
                    qwen2_text_encoder_mode,
                    eager,
                    offload,
                )
                .await;
            }
        }
    }

    let piped = is_piped();

    validate_batch_stdout(batch, piped, &output)?;

    // Default to the source image size for img2img/inpainting when neither
    // dimension was provided. We still normalize to the validation envelope
    // (multiples of 16, megapixel clamp) to avoid invalid requests and OOMs.
    let (effective_width, effective_height) = effective_dimensions(
        &config,
        &model_cfg,
        family.as_deref(),
        width,
        height,
        source_image.as_deref(),
        edit_images.as_deref(),
    )?;
    let effective_steps = steps.unwrap_or_else(|| model_cfg.effective_steps(&config));
    let effective_guidance = guidance.unwrap_or_else(|| model_cfg.effective_guidance());
    let effective_negative_prompt = effective_negative_prompt(
        family.as_deref(),
        effective_guidance,
        negative_prompt.clone(),
    );

    let mut req = GenerateRequest {
        guidance_overrides,
        prompt: prompt.to_string(),
        negative_prompt: effective_negative_prompt.clone(),
        model: model.to_string(),
        width: effective_width,
        height: effective_height,
        steps: effective_steps,
        guidance: effective_guidance,
        seed,
        batch_size: batch,
        output_format: Some(output_format),
        embed_metadata: Some(embed_metadata),
        scheduler,
        cfg_plus,
        edit_images: edit_images.clone(),
        source_image: source_image.clone(),
        source_image_name: None,
        strength,
        mask_image: mask_image.clone(),
        control_image: control_image.clone(),
        control_model: control_model.clone(),
        control_scale,
        expand,
        original_prompt,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: lora.clone(),
        frames: effective_frames,
        fps: effective_fps,
        upscale_model: None,
        gif_preview: preview,
        enable_audio,
        audio_file,
        audio_file_path: None,
        source_video,
        source_video_path: None,
        extend_video,
        extend_video_path: None,
        extend_overlap_frames,
        keyframes,
        pipeline,
        ic_lora_control,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        placement,
    };
    if local {
        materialize_local_builtin_control(&mut req, &config).await?;
        materialize_local_builtin_camera_controls(&mut req, &config).await?;
    }

    // Warn if user-provided dimensions don't match model recommendations.
    // Only warn locally — in remote mode the server sends the warning via SSE/header.
    if local && (width.is_some() || height.is_some()) {
        if let Some(ref family) = family {
            if let Some(warning) =
                mold_core::dimension_warning(effective_width, effective_height, family)
            {
                status!("{} {}", theme::icon_warn(), warning);
            }
        }
    }

    if let Some(desc) = &model_cfg.description {
        status!(
            "{} {} — {}",
            theme::icon_ok(),
            model.bold(),
            crate::output::colorize_description(desc)
        );
    }
    // Show truncated prompt so the user can confirm their input
    let display_prompt = if prompt.chars().count() > 60 {
        let truncated: String = prompt.chars().take(57).collect();
        format!("{truncated}...")
    } else {
        prompt.to_string()
    };
    status!("{} \"{}\"", theme::icon_info(), display_prompt.dimmed());
    if let Some(ref neg) = effective_negative_prompt {
        let display_neg = if neg.chars().count() > 50 {
            let truncated: String = neg.chars().take(47).collect();
            format!("{truncated}...")
        } else {
            neg.clone()
        };
        status!(
            "{} Negative: \"{}\"",
            theme::icon_info(),
            display_neg.dimmed()
        );
    }
    if mask_image.is_some() {
        status!(
            "{} inpainting mode (strength: {:.2})",
            theme::icon_mode(),
            strength,
        );
    } else if let Some(ref images) = edit_images {
        status!(
            "{} image edit mode ({} input image{})",
            theme::icon_mode(),
            images.len(),
            if images.len() == 1 { "" } else { "s" }
        );
    } else if source_image.is_some() {
        status!(
            "{} img2img mode (strength: {:.2})",
            theme::icon_mode(),
            strength,
        );
    }
    if let Some(ref cm) = control_model {
        status!(
            "{} ControlNet: {} (scale: {:.2})",
            theme::icon_mode(),
            cm.bold(),
            control_scale
        );
    }
    let is_video = effective_frames.is_some();
    if let Some(f) = effective_frames {
        let effective_fps = effective_fps.unwrap_or(24);
        status!(
            "{} Video mode: {} frames @ {} fps",
            theme::icon_mode(),
            f,
            effective_fps,
        );
        if is_ltx2 {
            let audio_mode = if enable_audio == Some(false) {
                "silent"
            } else {
                "audio+video"
            };
            status!("{} LTX-2 pipeline: {}", theme::icon_mode(), audio_mode);
        }
    }
    status!(
        "{} Generating {}x{} ({} steps, guidance {:.1})",
        theme::icon_info(),
        effective_width,
        effective_height,
        effective_steps,
        effective_guidance,
    );
    status!("{}", "─".repeat(40).dimmed());

    let base_seed = req.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let response = if local {
        print_using_local_inference();
        generate_local_batch(
            &req,
            &config,
            gpus,
            t5_variant.clone(),
            qwen3_variant.clone(),
            qwen2_variant.clone(),
            qwen2_text_encoder_mode.clone(),
            eager,
            offload,
            width,
            height,
            steps,
            guidance,
            batch,
            base_seed,
            batch_prompts.as_deref(),
            &output,
            output_format,
            preview,
        )
        .await?
    } else {
        let mut all_images: Vec<ImageData> = Vec::with_capacity(batch as usize);
        let mut last_video: Option<mold_core::VideoData> = None;
        let mut total_time_ms: u64 = 0;
        let mut last_seed_used: u64 = base_seed;
        let mut last_model = String::new();

        for i in 0..batch {
            let mut iter_req = req.clone();
            iter_req.seed = Some(base_seed.wrapping_add(i as u64));
            iter_req.batch_size = 1;

            // Use per-batch expanded prompt if available
            if let Some(ref prompts) = batch_prompts {
                if let Some(p) = prompts.get(i as usize) {
                    iter_req.prompt = p.clone();
                }
            }

            if batch > 1 {
                status!(
                    "{} Generating image {}/{} (seed: {})",
                    theme::icon_info(),
                    i + 1,
                    batch,
                    iter_req.seed.unwrap(),
                );
            }

            let response = generate_remote(
                ctx.client(),
                &iter_req,
                &config,
                model,
                piped,
                effective_width,
                effective_height,
                effective_steps,
                gpus.clone(),
                t5_variant.clone(),
                qwen3_variant.clone(),
                qwen2_variant.clone(),
                qwen2_text_encoder_mode.clone(),
                eager,
                offload,
                width,
                height,
                steps,
                guidance,
            )
            .await?;

            total_time_ms += response.generation_time_ms;
            last_seed_used = response.seed_used;
            last_model = response.model.clone();

            // Persist every successful clip before the next batch item can
            // fail. The aggregate retains the last clip only for the common
            // response shape; it is not the durability boundary.
            if let Some(video) = response.video.as_ref() {
                if batch > 1 {
                    save_and_preview_video(
                        video,
                        &output,
                        model,
                        batch,
                        i,
                        preview,
                        Some(PersistArgs {
                            request: &iter_req,
                            seed_used: response.seed_used,
                            generation_time_ms: response.generation_time_ms,
                        }),
                    )?;
                }
                last_video = response.video;
            }

            for mut img in response.images {
                img.index = i;
                // Save and preview each image immediately during batch generation
                // (single-image mode is handled in the post-loop section)
                if batch > 1 {
                    save_and_preview_image(
                        &img,
                        &output,
                        model,
                        batch,
                        output_format,
                        preview,
                        Some(PersistArgs {
                            request: &iter_req,
                            seed_used: response.seed_used,
                            generation_time_ms: response.generation_time_ms,
                        }),
                    )?;
                }
                all_images.push(img);
            }
        }

        GenerateResponse {
            images: all_images,
            video: last_video,
            generation_time_ms: total_time_ms,
            model: last_model,
            seed_used: last_seed_used,
            gpu: None,
        }
    };

    // Output: video or image.
    if let Some(ref video) = response.video {
        // --- Video output ---
        if batch > 1 {
            // Batch clips were persisted per item before aggregation.
        } else if piped && output.is_none() {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&video.data)?;
            stdout.flush()?;
        } else {
            save_and_preview_video(
                video,
                &output,
                model,
                1,
                0,
                preview,
                Some(PersistArgs {
                    request: &req,
                    seed_used: response.seed_used,
                    generation_time_ms: response.generation_time_ms,
                }),
            )?;
        }
    } else {
        // --- Image output ---
        // For batch > 1, images were already saved inside the batch loop above.
        if piped && output.is_none() {
            let mut stdout = std::io::stdout().lock();
            for img in &response.images {
                stdout.write_all(&img.data)?;
            }
            stdout.flush()?;
        } else if batch == 1 {
            for img in &response.images {
                save_and_preview_image(
                    img,
                    &output,
                    model,
                    batch,
                    output_format,
                    preview,
                    Some(PersistArgs {
                        request: &req,
                        seed_used: response.seed_used,
                        generation_time_ms: response.generation_time_ms,
                    }),
                )?;
            }
        }
        // batch > 1: already saved+previewed inside the batch loop
    }

    let secs = response.generation_time_ms as f64 / 1000.0;
    if is_video {
        let frame_count = response
            .video
            .as_ref()
            .map(|v| v.frames)
            .unwrap_or_default();
        status!(
            "{} Done — {} in {:.1}s ({} frames, seed: {})",
            theme::icon_done(),
            model.bold(),
            secs,
            frame_count,
            response.seed_used,
        );
    } else if batch > 1 {
        status!(
            "{} Done — {} in {:.1}s ({} images, base seed: {})",
            theme::icon_done(),
            model.bold(),
            secs,
            batch,
            base_seed,
        );
    } else {
        status!(
            "{} Done — {} in {:.1}s (seed: {})",
            theme::icon_done(),
            model.bold(),
            secs,
            response.seed_used,
        );
    }

    // Remember the model for next time (best-effort, non-fatal)
    mold_db::settings::record_last_model(&response.model);

    Ok(())
}

async fn materialize_local_builtin_control(
    request: &mut GenerateRequest,
    config: &Config,
) -> Result<()> {
    let Some(control) = request.ic_lora_control.as_deref() else {
        return Ok(());
    };
    let model_config = config.resolved_model_config(&request.model);
    mold_core::validate_generate_request_with_family(request, model_config.family.as_deref())
        .map_err(anyhow::Error::msg)?;
    let profile = mold_core::ltx2_control::control_profile_for_model(&request.model, &model_config)
        .map_err(anyhow::Error::msg)?;
    let adapter = mold_core::ltx2_control::resolve_control_adapter(profile, control)
        .map_err(anyhow::Error::msg)?;
    let manifest = mold_core::manifest::find_manifest(adapter.download_model)
        .expect("control registry and hidden manifests must stay in sync");
    let file = manifest
        .files
        .first()
        .expect("control manifests contain one adapter file");
    let path = config
        .resolved_models_dir()
        .join(mold_core::manifest::storage_path(manifest, file));
    if !local_control_artifact_is_complete(adapter, &path) {
        mold_core::download::pull_and_configure(
            adapter.download_model,
            &mold_core::download::PullOptions::default(),
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to download IC-LoRA control '{}': {error}",
                adapter.id
            )
        })?;
    }
    if !local_control_artifact_is_complete(adapter, &path) {
        anyhow::bail!(
            "IC-LoRA control '{}' download completed without a verified {}",
            adapter.id,
            path.display()
        );
    }

    let mut ordered = vec![LoraWeight {
        path: path.to_string_lossy().into_owned(),
        scale: 1.0,
    }];
    if let Some(legacy) = request.lora.take() {
        ordered.push(legacy);
    }
    ordered.extend(request.loras.take().unwrap_or_default());
    request.loras = Some(ordered);
    Ok(())
}

/// Download any built-in `camera-control:<id>` adapter the request names.
///
/// The server does this in `materialize_builtin_ltx2_camera_controls` before
/// planning. Forced-local had no counterpart, and
/// `execution_plan::materialize_request` rewrites the alias to its manifest
/// path during planning — so by the time the engine's `resolve_loras` runs,
/// the `camera-control:` prefix is gone and its download never fires. The
/// render then failed on a missing file. It only ever *looked* fine because
/// the runtime refused the real path for any plan carrying a LoRA and quietly
/// emitted placeholder frames instead.
async fn materialize_local_builtin_camera_controls(
    request: &mut GenerateRequest,
    config: &Config,
) -> Result<()> {
    let aliases: Vec<String> = request
        .loras
        .iter()
        .flatten()
        .chain(request.lora.iter())
        .filter_map(|lora| {
            lora.path
                .strip_prefix("camera-control:")
                .map(str::to_string)
        })
        .collect();
    if aliases.is_empty() {
        return Ok(());
    }

    let model_config = config.resolved_model_config(&request.model);
    mold_core::ltx2_camera::camera_profile_for_model(&request.model, &model_config)
        .map_err(anyhow::Error::msg)?;

    for alias in aliases {
        let preset = mold_core::ltx2_camera::resolve_camera_control_preset(&alias)
            .map_err(anyhow::Error::msg)?;
        let manifest = mold_core::manifest::find_manifest(preset.download_model)
            .expect("camera-control registry and hidden manifests must stay in sync");
        let file = manifest
            .files
            .first()
            .expect("camera-control manifests contain one adapter file");
        let path = config
            .resolved_models_dir()
            .join(mold_core::manifest::storage_path(manifest, file));
        if local_camera_control_artifact_is_complete(preset, &path) {
            continue;
        }
        mold_core::download::pull_and_configure(
            preset.download_model,
            &mold_core::download::PullOptions::default(),
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!("failed to download camera-motion preset '{alias}': {error}")
        })?;
        if !local_camera_control_artifact_is_complete(preset, &path) {
            anyhow::bail!(
                "camera-motion preset '{alias}' download completed without a verified {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn local_camera_control_artifact_is_complete(
    preset: &mold_core::ltx2_camera::Ltx2CameraControlPreset,
    path: &std::path::Path,
) -> bool {
    mold_core::download::has_sha256_marker(path)
        || path
            .metadata()
            .is_ok_and(|metadata| metadata.len() == preset.size_bytes)
}

fn local_control_artifact_is_complete(
    adapter: &mold_core::ltx2_control::Ltx2ControlAdapter,
    path: &std::path::Path,
) -> bool {
    mold_core::download::has_sha256_marker(path)
        || path
            .metadata()
            .is_ok_and(|metadata| metadata.len() == adapter.size_bytes)
}

/// Remote generation: try SSE streaming first, fall back to blocking API.
#[allow(clippy::too_many_arguments)]
async fn generate_remote(
    client: &MoldClient,
    req: &GenerateRequest,
    config: &Config,
    model: &str,
    piped: bool,
    effective_width: u32,
    effective_height: u32,
    effective_steps: u32,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    // Try SSE streaming first
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let render = tokio::spawn(render_progress(rx));

    match client.generate_stream(req, tx).await {
        Ok(Some(response)) => {
            let _ = render.await;
            Ok(response)
        }
        Ok(None) => {
            // Server doesn't support SSE — fall back to blocking API with spinner
            let _ = render.await;
            generate_remote_blocking(
                client,
                req,
                config,
                model,
                piped,
                effective_width,
                effective_height,
                effective_steps,
                gpus,
                t5_variant,
                qwen3_variant,
                qwen2_variant,
                qwen2_text_encoder_mode,
                eager,
                offload,
                cli_width,
                cli_height,
                cli_steps,
                cli_guidance,
            )
            .await
        }
        Err(e) => {
            let _ = render.await;
            match classify_generate_error(&e) {
                GenerateServerAction::PullModelAndRetry => {
                    print_server_pull_missing_model(model);
                    stream_server_pull(client, model)
                        .await
                        .map_err(|e| tag_remote(client, e))?;

                    status!("{} Generating...", theme::icon_info());

                    let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
                    let render2 = tokio::spawn(render_progress(rx2));
                    match client.generate_stream(req, tx2).await {
                        Ok(Some(response)) => {
                            let _ = render2.await;
                            Ok(response)
                        }
                        // Either the server has no SSE endpoint (Ok(None)) or
                        // the SSE stream failed (Err) — some proxies/servers
                        // close the stream before the final `complete` event
                        // even though the blocking `/api/generate` path still
                        // works. Fall back to the blocking endpoint before
                        // giving up.
                        _ => {
                            let _ = render2.await;
                            client
                                .generate(req.clone())
                                .await
                                .map_err(|e| tag_remote(client, e))
                        }
                    }
                }
                GenerateServerAction::FallbackLocal => {
                    print_using_local_inference();
                    let mut local_request = req.clone();
                    materialize_local_builtin_control(&mut local_request, config).await?;
                    materialize_local_builtin_camera_controls(&mut local_request, config).await?;
                    generate_local(
                        &local_request,
                        config,
                        gpus,
                        t5_variant,
                        qwen3_variant,
                        qwen2_variant,
                        qwen2_text_encoder_mode,
                        eager,
                        offload,
                        cli_width,
                        cli_height,
                        cli_steps,
                        cli_guidance,
                    )
                    .await
                }
                GenerateServerAction::SurfaceError => Err(tag_remote(client, e)),
            }
        }
    }
}

/// Blocking remote generation with a simple spinner (fallback for servers without SSE).
#[allow(clippy::too_many_arguments)]
async fn generate_remote_blocking(
    client: &MoldClient,
    req: &GenerateRequest,
    config: &Config,
    model: &str,
    piped: bool,
    effective_width: u32,
    effective_height: u32,
    effective_steps: u32,
    gpus: Option<String>,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    let pb = ProgressBar::new_spinner();
    if piped {
        pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    }
    pb.set_style(
        ProgressStyle::default_spinner()
            .template(&format!("{{spinner:.{}}} {{msg}}", theme::SPINNER_STYLE))
            .unwrap(),
    );
    pb.set_message(format!(
        "Generating on server ({}x{}, {} steps)...",
        effective_width, effective_height, effective_steps
    ));
    pb.enable_steady_tick(Duration::from_millis(100));

    match client.generate(req.clone()).await {
        Ok(response) => {
            pb.finish_and_clear();
            Ok(response)
        }
        Err(e) => {
            pb.finish_and_clear();
            match classify_generate_error(&e) {
                GenerateServerAction::PullModelAndRetry => {
                    print_server_pull_missing_model(model);
                    stream_server_pull(client, model)
                        .await
                        .map_err(|e| tag_remote(client, e))?;
                    status!("{} Generating...", theme::icon_info());
                    client
                        .generate(req.clone())
                        .await
                        .map_err(|e| tag_remote(client, e))
                }
                GenerateServerAction::FallbackLocal => {
                    print_using_local_inference();
                    let mut local_request = req.clone();
                    materialize_local_builtin_control(&mut local_request, config).await?;
                    materialize_local_builtin_camera_controls(&mut local_request, config).await?;
                    generate_local(
                        &local_request,
                        config,
                        gpus,
                        t5_variant,
                        qwen3_variant,
                        qwen2_variant,
                        qwen2_text_encoder_mode,
                        eager,
                        offload,
                        cli_width,
                        cli_height,
                        cli_steps,
                        cli_guidance,
                    )
                    .await
                }
                GenerateServerAction::SurfaceError => Err(tag_remote(client, e)),
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn prepare_local_request(
    req: &GenerateRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<(
    GenerateRequest,
    mold_core::ModelPaths,
    Config,
    super::local_engine::EngineOverrides,
)> {
    use super::local_engine::{resolve_or_pull_model, EngineOverrides};

    let model_name = req.model.clone();
    let mut req = req.clone();

    let (paths, effective_config, pulled) = resolve_or_pull_model(&model_name, config).await?;
    if pulled {
        let model_cfg = effective_config.resolved_model_config(&model_name);
        let family = resolve_family(&model_name, &effective_config);
        refit_request_after_pull(
            &mut req,
            &effective_config,
            &model_cfg,
            family.as_deref(),
            cli_width,
            cli_height,
            cli_steps,
            cli_guidance,
        )?;
        status!(
            "{} Updated defaults: {}x{} ({} steps, guidance {:.1})",
            theme::icon_info(),
            req.width,
            req.height,
            req.steps,
            req.guidance,
        );
    }

    validate_local_request(&req, &effective_config)?;

    let overrides = EngineOverrides {
        gpus,
        t5_variant: t5_variant_override,
        qwen3_variant: qwen3_variant_override,
        qwen2_variant: qwen2_variant_override,
        qwen2_text_encoder_mode: qwen2_text_encoder_mode_override,
        eager,
        offload,
    };
    Ok((req, paths, effective_config, overrides))
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn generate_local(
    req: &GenerateRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    let base_seed = req.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let output_format = req.output_format.unwrap_or(OutputFormat::Png);
    generate_local_batch(
        req,
        config,
        gpus,
        t5_variant_override,
        qwen3_variant_override,
        qwen2_variant_override,
        qwen2_text_encoder_mode_override,
        eager,
        offload,
        cli_width,
        cli_height,
        cli_steps,
        cli_guidance,
        1,
        base_seed,
        None,
        &None,
        output_format,
        false,
    )
    .await
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
struct LocalOwnerPool {
    command_txs: std::collections::BTreeMap<
        usize,
        std::sync::mpsc::SyncSender<Option<(u32, GenerateRequest)>>,
    >,
    workers: tokio::task::JoinSet<()>,
    progress_tasks: Vec<tokio::task::JoinHandle<()>>,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl LocalOwnerPool {
    async fn shutdown_and_join(&mut self) -> Option<anyhow::Error> {
        // Closing every sender is the shutdown signal. A blocking owner
        // finishes its current request, observes Disconnected, clears the
        // progress callback, and drops the engine on that same thread.
        self.command_txs.clear();
        let mut first_error = None;
        while let Some(result) = self.workers.join_next().await {
            if let Err(error) = result {
                first_error.get_or_insert_with(|| error.into());
            }
        }
        for render in self.progress_tasks.drain(..) {
            if let Err(error) = render.await {
                first_error.get_or_insert_with(|| error.into());
            }
        }
        first_error
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl Drop for LocalOwnerPool {
    fn drop(&mut self) {
        // This is the panic/cancellation safety net. Normal and recoverable
        // error paths call `shutdown_and_join`; Drop still closes the owner
        // channels before aborting async wrappers. Tokio cannot synchronously
        // join from Drop, but spawn_blocking owners retain no sender and
        // therefore exit after their current request.
        self.command_txs.clear();
        self.workers.abort_all();
        for task in &self.progress_tasks {
            task.abort();
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn local_batch_requests(
    base: &GenerateRequest,
    batch: u32,
    base_seed: u64,
    prompts: Option<&[String]>,
) -> Vec<GenerateRequest> {
    (0..batch)
        .map(|index| {
            let mut request = base.clone();
            request.seed = Some(base_seed.wrapping_add(index as u64));
            request.batch_size = 1;
            if let Some(prompt) = prompts.and_then(|values| values.get(index as usize)) {
                request.prompt = prompt.clone();
            }
            request
        })
        .collect()
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
#[allow(clippy::too_many_arguments)]
fn finalize_local_batch_outputs(
    req: &GenerateRequest,
    batch_requests: &[GenerateRequest],
    mut completed: Vec<(u32, GenerateResponse)>,
    first_error: Option<anyhow::Error>,
    output: &Option<String>,
    output_format: OutputFormat,
    preview: bool,
    persist_metadata: bool,
    batch: u32,
    base_seed: u64,
) -> Result<GenerateResponse> {
    completed.sort_by_key(|(index, _)| *index);
    let successful_items = completed
        .iter()
        .map(|(index, _)| (index + 1).to_string())
        .collect::<Vec<_>>();

    let mut all_images: Vec<ImageData> = Vec::with_capacity(batch as usize);
    let mut last_video: Option<mold_core::VideoData> = None;
    let mut total_time_ms = 0;
    let mut last_seed_used = base_seed;
    let mut last_model = String::new();

    for (i, mut response) in completed {
        total_time_ms += response.generation_time_ms;
        last_seed_used = response.seed_used;
        last_model = response.model.clone();
        let item_request = batch_requests.get(i as usize).ok_or_else(|| {
            anyhow::anyhow!("completed local batch item {} is out of range", i + 1)
        })?;

        if let Some(video) = response.video.as_ref() {
            if batch > 1 {
                save_and_preview_video(
                    video,
                    output,
                    &req.model,
                    batch,
                    i,
                    preview,
                    persist_metadata.then_some(PersistArgs {
                        request: item_request,
                        seed_used: response.seed_used,
                        generation_time_ms: response.generation_time_ms,
                    }),
                )?;
            }
            last_video = response.video.take();
        }

        for mut img in response.images {
            img.index = i;
            if batch > 1 {
                save_and_preview_image(
                    &img,
                    output,
                    &req.model,
                    batch,
                    output_format,
                    preview,
                    persist_metadata.then_some(PersistArgs {
                        request: item_request,
                        seed_used: response.seed_used,
                        generation_time_ms: response.generation_time_ms,
                    }),
                )?;
            }
            all_images.push(img);
        }
    }

    if let Some(error) = first_error {
        return Err(error.context(format!(
            "local batch preserved {} successful output item(s) ({}) with exact per-item prompt/seed provenance",
            successful_items.len(),
            if successful_items.is_empty() {
                "none".to_string()
            } else {
                successful_items.join(", ")
            }
        )));
    }

    Ok(GenerateResponse {
        images: all_images,
        video: last_video,
        generation_time_ms: total_time_ms,
        model: last_model,
        seed_used: last_seed_used,
        gpu: None,
    })
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
async fn generate_local_batch(
    req: &GenerateRequest,
    config: &Config,
    gpus: Option<String>,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    qwen2_variant_override: Option<String>,
    qwen2_text_encoder_mode_override: Option<String>,
    eager: bool,
    offload: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
    batch: u32,
    base_seed: u64,
    batch_prompts: Option<&[String]>,
    output: &Option<String>,
    output_format: OutputFormat,
    preview: bool,
) -> Result<GenerateResponse> {
    use super::local_engine::{
        apply_local_engine_env_overrides, build_local_engine_from_plan, plan_local_batch,
        LocalBatchAdmission, LocalLease,
    };

    let (base_req, _paths, effective_config, overrides) = prepare_local_request(
        req,
        config,
        gpus,
        t5_variant_override,
        qwen3_variant_override,
        qwen2_variant_override,
        qwen2_text_encoder_mode_override,
        eager,
        offload,
        cli_width,
        cli_height,
        cli_steps,
        cli_guidance,
    )
    .await?;
    apply_local_engine_env_overrides(
        overrides.t5_variant.as_deref(),
        overrides.qwen3_variant.as_deref(),
        overrides.qwen2_variant.as_deref(),
        overrides.qwen2_text_encoder_mode.as_deref(),
    );
    let local_plan = plan_local_batch(&base_req, &effective_config, &overrides).await?;
    let mut admission = LocalBatchAdmission::new(
        &local_plan.candidates,
        batch as usize,
        local_plan.host_headroom_bytes,
    )?;
    let batch_requests = local_batch_requests(&base_req, batch, base_seed, batch_prompts);
    let mut planned_items = batch_requests
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, request)| Some((index as u32, request)))
        .collect::<Vec<_>>();

    enum LocalOwnerEvent {
        Ready(usize),
        Completed(usize, u32, GenerateResponse),
        Failed(usize, u32, anyhow::Error),
        Crashed(usize, anyhow::Error),
    }
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<LocalOwnerEvent>();
    let mut command_txs = std::collections::BTreeMap::new();
    let mut workers = tokio::task::JoinSet::new();
    let mut progress_tasks = Vec::new();
    for (ordinal, execution_plan) in &local_plan.execution_plans {
        let ordinal = *ordinal;
        let config = effective_config.clone();
        let execution_plan = execution_plan.clone();
        let prepared_execution_inputs = local_plan.prepared_execution_inputs.clone();
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<Option<(u32, GenerateRequest)>>(1);
        command_txs.insert(ordinal, command_tx);
        let event_tx = event_tx.clone();
        let crash_tx = event_tx.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<mold_core::SseProgressEvent>();
        let render = tokio::spawn(render_progress(rx));
        workers.spawn(async move {
            let joined = tokio::task::spawn_blocking(move || {
                // Engine construction, loading, generation, and drop all remain
                // on this device's one local owner thread.
                mold_inference::device::init_thread_gpu_ordinal(ordinal);
                let mut engine = None;
                let _ = event_tx.send(LocalOwnerEvent::Ready(ordinal));
                while let Ok(Some((index, mut request))) = command_rx.recv() {
                    mold_server::execution_plan::materialize_request(&execution_plan, &mut request);
                    let result = (|| -> Result<GenerateResponse> {
                        if engine.is_none() {
                            let mut created = build_local_engine_from_plan(
                                &request,
                                &config,
                                &execution_plan,
                                &prepared_execution_inputs,
                            )?;
                            created.set_on_progress(Box::new({
                                let tx = tx.clone();
                                move |event| {
                                    let _ = tx.send(event.into());
                                }
                            }));
                            created.load_for_request(&request)?;
                            engine = Some(created);
                        }
                        engine
                            .as_mut()
                            .expect("local engine initialized before generation")
                            .generate(&request)
                    })();
                    match result {
                        Ok(response) => {
                            let _ =
                                event_tx.send(LocalOwnerEvent::Completed(ordinal, index, response));
                            let _ = event_tx.send(LocalOwnerEvent::Ready(ordinal));
                        }
                        Err(error) => {
                            let _ = event_tx.send(LocalOwnerEvent::Failed(ordinal, index, error));
                            break;
                        }
                    }
                }
                if let Some(engine) = engine.as_mut() {
                    engine.clear_on_progress();
                }
            })
            .await;
            if let Err(error) = joined {
                let _ = crash_tx.send(LocalOwnerEvent::Crashed(ordinal, error.into()));
            }
        });
        progress_tasks.push(render);
    }
    drop(event_tx);
    let mut owners = LocalOwnerPool {
        command_txs,
        workers,
        progress_tasks,
    };

    let mut completed = Vec::with_capacity(batch as usize);
    let mut first_error = None;
    let mut terminal_items = 0_usize;
    let mut initial_ready = std::collections::BTreeSet::new();
    'events: while terminal_items < batch as usize {
        let Some(event) = event_rx.recv().await else {
            if first_error.is_none() {
                first_error = Some(anyhow::anyhow!(
                    "all local GPU owners exited with batch work unfinished"
                ));
            }
            break;
        };
        match event {
            LocalOwnerEvent::Ready(ordinal) => {
                if let Err(error) = admission.owner_ready(ordinal) {
                    first_error = Some(error.context(format!(
                        "local owner {ordinal} published an invalid ready transition"
                    )));
                    break 'events;
                }
                initial_ready.insert(ordinal);
            }
            LocalOwnerEvent::Completed(ordinal, index, response) => {
                if let Err(error) = admission.owner_completed(ordinal) {
                    first_error = Some(error.context(format!(
                        "local owner {ordinal} completed item {} outside its lease",
                        index + 1
                    )));
                    break 'events;
                }
                completed.push((index, response));
                terminal_items += 1;
            }
            LocalOwnerEvent::Failed(ordinal, index, error) => {
                let _ = admission.owner_failed(ordinal);
                terminal_items += 1;
                if first_error.is_none() {
                    first_error = Some(anyhow::anyhow!(
                        "local item {} failed on GPU {}: {error:#}",
                        index + 1,
                        ordinal
                    ));
                }
            }
            LocalOwnerEvent::Crashed(ordinal, error) => {
                initial_ready.insert(ordinal);
                if admission.owner_failed(ordinal).is_some() {
                    terminal_items += 1;
                }
                if first_error.is_none() {
                    first_error = Some(anyhow::anyhow!(
                        "local GPU owner {ordinal} crashed: {error:#}"
                    ));
                }
            }
        }

        // Wait for every owner to publish its initial Ready before the first
        // admission so batch=1 is selected from the complete device set.
        if initial_ready.len() < owners.command_txs.len() {
            continue;
        }
        loop {
            let leases = match admission.lease_ready() {
                Ok(leases) => leases,
                Err(error) => {
                    first_error = Some(error.context("local batch admission invariant failed"));
                    break 'events;
                }
            };
            if leases.is_empty() {
                break;
            }
            let mut retry_admission = false;
            for lease in leases {
                let Some((index, request)) = planned_items
                    .get_mut(lease.item_index)
                    .and_then(Option::take)
                else {
                    admission.lease_transport_failed(lease);
                    retry_admission = true;
                    if first_error.is_none() {
                        first_error = Some(anyhow::anyhow!(
                            "local scheduler leased an unavailable batch item"
                        ));
                    }
                    continue;
                };
                if batch > 1 {
                    status!(
                        "{} Starting local item {}/{} on GPU {} (seed: {})",
                        theme::icon_info(),
                        index + 1,
                        batch,
                        lease.ordinal,
                        request.seed.unwrap(),
                    );
                }
                let returned = match owners.command_txs.get(&lease.ordinal) {
                    Some(tx) => match tx.send(Some((index, request))) {
                        Ok(()) => None,
                        Err(error) => error.0,
                    },
                    None => Some((index, request)),
                };
                if let Some(returned) = returned {
                    planned_items[lease.item_index] = Some(returned);
                    admission.lease_transport_failed(LocalLease {
                        ordinal: lease.ordinal,
                        item_index: lease.item_index,
                    });
                    retry_admission = true;
                }
            }
            if !retry_admission {
                break;
            }
        }
        if admission.has_pending() && !admission.has_active() && !admission.has_owners() {
            if first_error.is_none() {
                first_error = Some(anyhow::anyhow!(
                    "no healthy local GPU owner remains for queued batch work"
                ));
            }
            break;
        }
    }
    if let Some(error) = owners.shutdown_and_join().await {
        first_error.get_or_insert(error);
    }
    finalize_local_batch_outputs(
        req,
        &batch_requests,
        completed,
        first_error,
        output,
        output_format,
        preview,
        true,
        batch,
        base_seed,
    )
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
async fn generate_local(
    _req: &GenerateRequest,
    _config: &Config,
    _gpus: Option<String>,
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
    _qwen2_variant: Option<String>,
    _qwen2_text_encoder_mode: Option<String>,
    _eager: bool,
    _offload: bool,
    _cli_width: Option<u32>,
    _cli_height: Option<u32>,
    _cli_steps: Option<u32>,
    _cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
    anyhow::bail!(
        "No mold server running and this binary was built without GPU support.\n\
         Either start a server with `mold serve` or rebuild with --features cuda"
    )
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
async fn generate_local_batch(
    _req: &GenerateRequest,
    _config: &Config,
    _gpus: Option<String>,
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
    _qwen2_variant: Option<String>,
    _qwen2_text_encoder_mode: Option<String>,
    _eager: bool,
    _offload: bool,
    _cli_width: Option<u32>,
    _cli_height: Option<u32>,
    _cli_steps: Option<u32>,
    _cli_guidance: Option<f64>,
    _batch: u32,
    _base_seed: u64,
    _batch_prompts: Option<&[String]>,
    _output: &Option<String>,
    _output_format: OutputFormat,
    _preview: bool,
) -> Result<GenerateResponse> {
    anyhow::bail!(
        "No mold server running and this binary was built without GPU support.\n\
         Either start a server with `mold serve` or rebuild with --features cuda"
    )
}

/// Optional metadata used by [`save_and_preview_image`] to persist a row
/// in the SQLite gallery DB after a successful save. Skipped silently when
/// `None` (e.g. tests, stdout output, when the DB is disabled).
struct PersistArgs<'a> {
    request: &'a GenerateRequest,
    seed_used: u64,
    generation_time_ms: u64,
}

fn save_and_preview_video(
    video: &mold_core::VideoData,
    output: &Option<String>,
    model: &str,
    batch: u32,
    index: u32,
    preview: bool,
    persist: Option<PersistArgs<'_>>,
) -> anyhow::Result<()> {
    let filename = match output {
        Some(path) if path == "-" => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&video.data)?;
            stdout.flush()?;
            return Ok(());
        }
        Some(path) if batch == 1 => path.clone(),
        Some(path) => {
            let path = std::path::Path::new(path);
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("output");
            let leaf = format!("{stem}-{index}.{}", video.format.extension());
            path.parent()
                .filter(|directory| !directory.as_os_str().is_empty())
                .map(|directory| directory.join(&leaf))
                .unwrap_or_else(|| std::path::PathBuf::from(&leaf))
                .to_string_lossy()
                .into_owned()
        }
        None => default_filename(
            model,
            mold_core::time::now_epoch_ms_u64(),
            video.format.extension(),
            batch,
            index,
        ),
    };

    if std::path::Path::new(&filename).exists() {
        status!("{} Overwriting: {}", theme::icon_alert(), filename);
    }
    std::fs::write(&filename, &video.data)?;
    status!(
        "{} Saved: {} ({} frames, {}x{}, {} fps)",
        theme::icon_done(),
        filename.bold(),
        video.frames,
        video.width,
        video.height,
        video.fps,
    );
    if let Some(persist) = persist {
        crate::metadata_db::record_local_save(
            std::path::Path::new(&filename),
            persist.request,
            persist.seed_used,
            persist.generation_time_ms,
            video.format,
            Some((video.width, video.height)),
        );
    }
    if preview {
        // Show first frame preview (viuer doesn't support animation).
        // Fallback to the video data itself for GIF/APNG/WebP when neither
        // precomputed preview is available.
        if !video.gif_preview.is_empty() {
            preview_image(&video.gif_preview);
        } else if !video.thumbnail.is_empty() {
            preview_image(&video.thumbnail);
        } else {
            preview_image(&video.data);
        }
    }
    Ok(())
}

/// Save a single image to disk and optionally preview it inline.
///
/// Resolves the output filename from the `--output` flag, batch index,
/// model name, and output format. Used both inside batch loops (for
/// immediate save+preview) and for single-image output. When `persist`
/// is `Some`, also records a metadata row in the SQLite gallery DB.
fn save_and_preview_image(
    img: &ImageData,
    output: &Option<String>,
    model: &str,
    batch: u32,
    output_format: OutputFormat,
    preview: bool,
    persist: Option<PersistArgs<'_>>,
) -> anyhow::Result<()> {
    let filename = match output {
        Some(path) if path == "-" => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&img.data)?;
            stdout.flush()?;
            return Ok(());
        }
        Some(path) if batch == 1 => path.clone(),
        Some(path) => {
            let p = std::path::Path::new(path);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let ext = output_format.to_string();
            let leaf = format!("{stem}-{}.{ext}", img.index);
            p.parent()
                .filter(|d| !d.as_os_str().is_empty())
                .map(|d| d.join(&leaf))
                .unwrap_or_else(|| std::path::PathBuf::from(&leaf))
                .to_string_lossy()
                .into_owned()
        }
        None => {
            let ext = output_format.to_string();
            default_filename(
                model,
                mold_core::time::now_epoch_ms_u64(),
                &ext,
                batch,
                img.index,
            )
        }
    };

    if std::path::Path::new(&filename).exists() {
        status!("{} Overwriting: {}", theme::icon_alert(), filename);
    }
    std::fs::write(&filename, &img.data)?;
    status!("{} Saved: {}", theme::icon_done(), filename.bold());
    if let Some(p) = persist {
        crate::metadata_db::record_local_save(
            std::path::Path::new(&filename),
            p.request,
            p.seed_used,
            p.generation_time_ms,
            output_format,
            Some((img.width, img.height)),
        );
    }
    if preview {
        preview_image(&img.data);
    }
    Ok(())
}

/// Display an image inline in the terminal using viuer.
/// Silently skipped if the `preview` feature is not compiled or if decoding fails.
#[cfg(any(feature = "preview", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewBackend {
    ViuerAuto,
    GhosttyKitty,
}

#[cfg(any(feature = "preview", test))]
fn preview_backend(term_program: Option<&str>, term: Option<&str>) -> PreviewBackend {
    if matches!(term_program, Some("ghostty")) || matches!(term, Some("xterm-ghostty")) {
        PreviewBackend::GhosttyKitty
    } else {
        PreviewBackend::ViuerAuto
    }
}

#[cfg(any(feature = "preview", test))]
fn fit_preview_cells(img_width: u32, img_height: u32, term_w: u16, term_h: u16) -> (u32, u32) {
    // Kitty is pixel-accurate, but the preview still occupies terminal cells.
    // We intentionally model each character row as roughly 2x taller than a
    // column is wide so Ghostty previews preserve image aspect instead of
    // stretching to the terminal's full row count.
    let bound_width = term_w as u32;
    let bound_height = 2 * term_h as u32;

    if img_width <= bound_width && img_height <= bound_height {
        let rows = std::cmp::max(1, img_height / 2 + img_height % 2);
        let rows = if rows == term_h as u32 && rows > 1 {
            rows - 1
        } else {
            rows
        };
        return (img_width, rows);
    }

    let ratio = img_width * bound_height;
    let nratio = bound_width * img_height;
    let use_width = nratio <= ratio;
    let intermediate = if use_width {
        img_height * bound_width / img_width
    } else {
        img_width * bound_height / img_height
    };

    let (cols, mut rows) = if use_width {
        (bound_width, std::cmp::max(1, intermediate / 2))
    } else {
        (intermediate, std::cmp::max(1, bound_height / 2))
    };

    if rows == term_h as u32 && rows > 1 {
        rows -= 1;
    }

    (cols, rows)
}

#[cfg(any(feature = "preview", test))]
fn ghostty_kitty_preview_chunks(encoded: &str, cell_w: u32, cell_h: u32) -> String {
    let mut out = String::new();
    let mut chunks = encoded.as_bytes().chunks(4096).peekable();
    if let Some(first) = chunks.next() {
        let more = if chunks.peek().is_some() { 1 } else { 0 };
        out.push_str(&format!(
            "\x1b_Gf=100,a=T,t=d,c={},r={},m={};{}\x1b\\",
            cell_w,
            cell_h,
            more,
            std::str::from_utf8(first).expect("base64 payload must be utf-8"),
        ));
    }

    while let Some(chunk) = chunks.next() {
        let more = if chunks.peek().is_some() { 1 } else { 0 };
        out.push_str(&format!(
            "\x1b_Gm={};{}\x1b\\",
            more,
            std::str::from_utf8(chunk).expect("base64 payload must be utf-8"),
        ));
    }

    out
}

#[cfg(any(feature = "preview", test))]
fn build_ghostty_kitty_preview_payload(
    img: &image::DynamicImage,
    term_w: u16,
    term_h: u16,
) -> std::io::Result<(String, u32, u32)> {
    let mut encoded_png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut encoded_png, image::ImageFormat::Png)
        .map_err(std::io::Error::other)?;
    let encoded = general_purpose::STANDARD.encode(encoded_png.into_inner());
    let (cell_w, cell_h) = fit_preview_cells(img.width(), img.height(), term_w, term_h);
    let payload = ghostty_kitty_preview_chunks(&encoded, cell_w, cell_h);
    Ok((payload, cell_w, cell_h))
}

#[cfg(feature = "preview")]
fn print_ghostty_kitty_preview(img: &image::DynamicImage) -> std::io::Result<()> {
    let (term_w, term_h) = viuer::terminal_size();
    let (payload, cell_w, _cell_h) = build_ghostty_kitty_preview_payload(img, term_w, term_h)?;

    let mut stdout = std::io::stdout();
    write!(stdout, "{payload}")?;
    if cell_w < term_w as u32 {
        writeln!(stdout)?;
    }
    stdout.flush()
}

#[cfg(feature = "preview")]
fn preview_config() -> viuer::Config {
    viuer::Config {
        absolute_offset: false,
        ..Default::default()
    }
}

#[cfg(feature = "preview")]
pub(crate) fn preview_image(data: &[u8]) {
    if !std::io::stdout().is_terminal() {
        return;
    }

    let term_program = std::env::var("TERM_PROGRAM").ok();
    let term = std::env::var("TERM").ok();
    let backend = preview_backend(term_program.as_deref(), term.as_deref());

    // Try to decode as an animated container first (GIF/APNG/animated WebP).
    if let Some(frames) = decode_preview_animation(data) {
        if frames.len() >= 2 {
            animate_preview(&frames, backend);
            return;
        }
    }

    let Ok(img) = image::load_from_memory(data) else {
        return;
    };
    render_preview_frame(&img, backend);
}

#[cfg(feature = "preview")]
fn render_preview_frame(img: &image::DynamicImage, backend: PreviewBackend) {
    match backend {
        PreviewBackend::GhosttyKitty => {
            let _ = print_ghostty_kitty_preview(img);
        }
        PreviewBackend::ViuerAuto => {
            let conf = preview_config();
            let _ = viuer::print(img, &conf);
        }
    }
}

/// Decoded frame for CLI preview animation.
#[cfg(feature = "preview")]
struct CliAnimatedFrame {
    image: image::DynamicImage,
    delay: Duration,
}

#[cfg(feature = "preview")]
fn decode_preview_animation(data: &[u8]) -> Option<Vec<CliAnimatedFrame>> {
    use image::AnimationDecoder;
    use std::io::Cursor;

    // Sniff the container before paying for a full decode.
    let is_gif = data.len() >= 6 && (data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a"));
    let is_png_sig =
        data.len() >= 8 && data[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let is_webp = data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP";

    let frames: Vec<_> = if is_gif {
        image::codecs::gif::GifDecoder::new(Cursor::new(data))
            .ok()?
            .into_frames()
            .collect_frames()
            .ok()?
    } else if is_png_sig {
        let decoder = image::codecs::png::PngDecoder::new(Cursor::new(data)).ok()?;
        decoder.apng().ok()?.into_frames().collect_frames().ok()?
    } else if is_webp {
        image::codecs::webp::WebPDecoder::new(Cursor::new(data))
            .ok()?
            .into_frames()
            .collect_frames()
            .ok()?
    } else {
        return None;
    };

    let mut out = Vec::with_capacity(frames.len());
    for frame in frames {
        let (num, den) = frame.delay().numer_denom_ms();
        // `numer_denom_ms()` is the delay in ms as the ratio num/den.
        let micros = if den == 0 {
            100_000
        } else {
            (u64::from(num) * 1000) / u64::from(den.max(1))
        };
        let delay = Duration::from_micros(micros).max(Duration::from_millis(20));
        let image = image::DynamicImage::ImageRgba8(frame.into_buffer());
        out.push(CliAnimatedFrame { image, delay });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Decide how many times to loop the animation for the CLI preview. A short
/// clip gets looped a couple of times so the motion is easy to see; longer
/// clips play once. Pure function for testability.
#[cfg(any(feature = "preview", test))]
fn preview_loop_count(total: Duration) -> u32 {
    if total < Duration::from_millis(1500) {
        3
    } else if total < Duration::from_secs(3) {
        2
    } else {
        1
    }
}

#[cfg(feature = "preview")]
fn animate_preview(frames: &[CliAnimatedFrame], backend: PreviewBackend) {
    let total: Duration = frames.iter().map(|f| f.delay).sum();
    let loops = preview_loop_count(total);

    // Save the cursor position once; restore before each frame so every
    // render lands at the same spot and overwrites the previous frame.
    // After the final frame, the backend naturally leaves the cursor
    // below the image — no further restore needed.
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(b"\x1b[s");
    let _ = stdout.flush();

    for _ in 0..loops {
        for frame in frames {
            let _ = stdout.write_all(b"\x1b[u");
            let _ = stdout.flush();
            render_preview_frame(&frame.image, backend);
            std::thread::sleep(frame.delay);
        }
    }
}

#[cfg(not(feature = "preview"))]
pub(crate) fn preview_image(_data: &[u8]) {
    use std::sync::OnceLock;
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        eprintln!(
            "warning: --preview has no effect — rebuild with `--features preview` to enable inline image display"
        );
    });
}

/// Build a default output filename, sanitizing colons from model names.
fn default_filename(model: &str, timestamp: u64, ext: &str, batch: u32, index: u32) -> String {
    mold_core::default_output_filename(model, timestamp, ext, batch, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::ModelConfig;

    #[tokio::test]
    async fn local_control_materialization_preserves_built_in_first_ordering() {
        let temp = tempfile::tempdir().unwrap();
        let config = Config {
            models_dir: temp.path().display().to_string(),
            ..Config::default()
        };
        let adapter = mold_core::ltx2_control::resolve_control_adapter(
            mold_core::ltx2_control::Ltx2ControlProfile::Ltx2_19bDistilled,
            "union",
        )
        .unwrap();
        let manifest = mold_core::manifest::find_manifest(adapter.download_model).unwrap();
        let adapter_path = temp.path().join(mold_core::manifest::storage_path(
            manifest,
            &manifest.files[0],
        ));
        std::fs::create_dir_all(adapter_path.parent().unwrap()).unwrap();
        std::fs::write(&adapter_path, b"installed").unwrap();
        mold_core::download::write_sha256_marker(&adapter_path, "test").unwrap();
        let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test",
            "model": "ltx-2-19b-distilled:fp8",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "output_format": "mp4",
            "source_video_path": "/guide.mp4",
            "pipeline": "ic-lora",
            "ic_lora_control": "union",
            "lora": { "path": "/loras/legacy.safetensors", "scale": 0.6 },
            "loras": [{ "path": "/loras/style.safetensors", "scale": 0.8 }]
        }))
        .unwrap();

        materialize_local_builtin_control(&mut request, &config)
            .await
            .unwrap();

        assert!(request.lora.is_none());
        let loras = request.loras.unwrap();
        assert_eq!(loras[0].path, adapter_path.to_string_lossy());
        assert_eq!(loras[0].scale, 1.0);
        assert_eq!(loras[1].path, "/loras/legacy.safetensors");
        assert_eq!(loras[2].path, "/loras/style.safetensors");
    }

    #[test]
    fn filename_sanitizes_colon() {
        let name = default_filename("flux-dev:q6", 1773609166, "png", 1, 0);
        assert_eq!(name, "mold-flux-dev-q6-1773609166.png");
        assert!(!name.contains(':'));
    }

    #[test]
    fn filename_no_colon_passthrough() {
        let name = default_filename("flux-schnell", 100, "png", 1, 0);
        assert_eq!(name, "mold-flux-schnell-100.png");
    }

    #[test]
    fn filename_batch_includes_index() {
        let name = default_filename("flux-dev:q4", 100, "jpeg", 3, 2);
        assert_eq!(name, "mold-flux-dev-q4-100-2.jpeg");
    }

    #[test]
    fn filename_single_batch_no_index() {
        let name = default_filename("flux-dev:q4", 100, "png", 1, 0);
        assert!(!name.contains("-0."));
    }

    #[test]
    fn local_batch_requests_freeze_per_item_prompt_and_seed_provenance() {
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"source","model":"flux-dev:q4","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let prompts = vec!["first".to_string(), "second".to_string()];
        let requests = local_batch_requests(&request, 2, 41, Some(&prompts));
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].prompt, "first");
        assert_eq!(requests[0].seed, Some(41));
        assert_eq!(requests[1].prompt, "second");
        assert_eq!(requests[1].seed, Some(42));
        assert_eq!(requests[0].batch_size, 1);
        assert_eq!(requests[1].batch_size, 1);
    }

    #[test]
    fn partial_local_video_batch_persists_success_before_returning_failure() {
        let temp = tempfile::TempDir::new().unwrap();
        let output = Some(temp.path().join("clip.mp4").to_string_lossy().into_owned());
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"source","model":"ltx-video:bf16","width":512,"height":512,"steps":4,"guidance":1.0}"#,
        )
        .unwrap();
        let prompts = vec!["first clip".to_string(), "second clip".to_string()];
        let batch_requests = local_batch_requests(&request, 2, 91, Some(&prompts));
        let response = GenerateResponse {
            images: Vec::new(),
            video: Some(mold_core::VideoData {
                data: b"successful-video".to_vec(),
                format: OutputFormat::Mp4,
                width: 512,
                height: 512,
                frames: 9,
                fps: 24,
                thumbnail: Vec::new(),
                gif_preview: Vec::new(),
                has_audio: false,
                duration_ms: None,
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms: 123,
            model: "ltx-video:bf16".to_string(),
            seed_used: 91,
            gpu: Some(0),
        };

        let error = finalize_local_batch_outputs(
            &request,
            &batch_requests,
            vec![(0, response)],
            Some(anyhow::anyhow!("local item 2 failed")),
            &output,
            OutputFormat::Mp4,
            false,
            false,
            2,
            91,
        )
        .unwrap_err();

        assert_eq!(
            std::fs::read(temp.path().join("clip-0.mp4")).unwrap(),
            b"successful-video"
        );
        let rendered = format!("{error:#}");
        assert!(rendered.contains("local item 2 failed"));
        assert!(rendered.contains("successful output item(s) (1)"));
        assert_eq!(batch_requests[0].prompt, "first clip");
        assert_eq!(batch_requests[0].seed, Some(91));
    }

    #[tokio::test]
    async fn local_owner_pool_shutdown_closes_and_joins_owner_threads() {
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<Option<(u32, GenerateRequest)>>(1);
        let exited = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let owner_exited = exited.clone();
        let mut workers = tokio::task::JoinSet::new();
        workers.spawn(async move {
            tokio::task::spawn_blocking(move || {
                while command_rx.recv().is_ok() {}
                owner_exited.store(true, std::sync::atomic::Ordering::SeqCst);
            })
            .await
            .unwrap();
        });
        let mut owners = LocalOwnerPool {
            command_txs: std::collections::BTreeMap::from([(0, command_tx)]),
            workers,
            progress_tasks: vec![tokio::spawn(async {})],
        };
        assert!(owners.shutdown_and_join().await.is_none());
        assert!(exited.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn output_dash_is_special() {
        // "--output -" triggers stdout output in both interactive and piped modes
        let path = "-";
        assert_eq!(path, "-");
    }

    #[test]
    fn batch_stdout_rejection_is_media_neutral_for_images_and_videos() {
        for (piped, output) in [(true, None), (false, Some("-".to_string()))] {
            let error = validate_batch_stdout(2, piped, &output).unwrap_err();
            let rendered = format!("{error:#}");
            assert!(rendered.contains("more than 1 output"));
            assert!(rendered.contains("batch outputs"));
            assert!(!rendered.contains("image"));
        }
        validate_batch_stdout(2, true, &Some("clip.mp4".to_string())).unwrap();
        validate_batch_stdout(1, true, &None).unwrap();
    }

    #[test]
    fn pipe_detection_available() {
        // Verify pipe detection is available (used at runtime to route output)
        let _piped = crate::output::is_piped();
    }

    #[test]
    fn test_default_filename_empty_model() {
        let name = default_filename("", 100, "png", 1, 0);
        assert_eq!(name, "mold--100.png");
        // Batch variant with empty model
        let batch_name = default_filename("", 100, "png", 2, 1);
        assert_eq!(batch_name, "mold--100-1.png");
    }

    #[test]
    fn test_default_filename_special_chars() {
        // Colons are sanitized to dashes
        let name = default_filename("model:tag:extra", 42, "png", 1, 0);
        assert_eq!(name, "mold-model-tag-extra-42.png");
        assert!(!name.contains(':'));

        // Other special characters pass through
        let name2 = default_filename("my_model.v2", 42, "png", 1, 0);
        assert_eq!(name2, "mold-my_model.v2-42.png");
    }

    #[test]
    fn test_default_filename_jpeg_extension() {
        // "jpeg" extension passes through as-is
        let name = default_filename("flux-dev:q4", 500, "jpeg", 1, 0);
        assert_eq!(name, "mold-flux-dev-q4-500.jpeg");
        assert!(name.ends_with(".jpeg"));

        // "jpg" extension also works
        let name2 = default_filename("flux-dev:q4", 500, "jpg", 1, 0);
        assert_eq!(name2, "mold-flux-dev-q4-500.jpg");
        assert!(name2.ends_with(".jpg"));
    }

    fn png_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |_, _| image::Rgb([255, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn effective_dimensions_uses_model_defaults_for_txt2img() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };

        assert_eq!(
            effective_dimensions(&config, &model_cfg, None, None, None, None, None).unwrap(),
            (1024, 1024)
        );
    }

    #[test]
    fn effective_dimensions_fits_wide_source_to_model_bounds() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        // 1280x704 is wider than 1:1 model -> width-limited: w=1024, h=1024*(704/1280)=563.2->560
        let source = png_with_dimensions(1280, 704);

        assert_eq!(
            effective_dimensions(&config, &model_cfg, None, None, None, Some(&source), None)
                .unwrap(),
            (1024, 560)
        );
    }

    #[test]
    fn effective_dimensions_fits_non_aligned_source_to_model_bounds() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        // 1001x777: src_ratio=1.288 > 1.0, width-limited: w=1024, h=1024/1.288=795.0->784
        let source = png_with_dimensions(1001, 777);

        assert_eq!(
            effective_dimensions(&config, &model_cfg, None, None, None, Some(&source), None)
                .unwrap(),
            (1024, 784)
        );
    }

    #[test]
    fn effective_dimensions_explicit_overrides_win_for_img2img() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1280, 704);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                None,
                Some(512),
                Some(768),
                Some(&source),
                None,
            )
            .unwrap(),
            (512, 768)
        );
    }

    #[test]
    fn effective_dimensions_downscales_to_sd15_model() {
        // 1024x1024 FLUX output -> 512x512 SD1.5 model
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(512),
            default_height: Some(512),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1024, 1024);

        assert_eq!(
            effective_dimensions(&config, &model_cfg, None, None, None, Some(&source), None)
                .unwrap(),
            (512, 512)
        );
    }

    #[test]
    fn effective_dimensions_wide_source_to_sd15() {
        // 1920x1080 -> 512x512 SD1.5: width-limited, h=512/1.778=287.9->288
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(512),
            default_height: Some(512),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1920, 1080);

        assert_eq!(
            effective_dimensions(&config, &model_cfg, None, None, None, Some(&source), None)
                .unwrap(),
            (512, 288)
        );
    }

    #[test]
    fn effective_dimensions_upscales_small_source_to_model_native() {
        // 512x512 source -> 1024x1024 FLUX model (upscale to model native)
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(512, 512);

        assert_eq!(
            effective_dimensions(&config, &model_cfg, None, None, None, Some(&source), None)
                .unwrap(),
            (1024, 1024)
        );
    }

    #[test]
    fn effective_dimensions_qwen_image_edit_uses_first_edit_image() {
        let config = Config::default();
        let model_cfg = ModelConfig::default();
        let first = png_with_dimensions(1600, 900);
        let second = png_with_dimensions(512, 512);

        assert_eq!(
            effective_dimensions(
                &config,
                &model_cfg,
                Some("qwen-image-edit"),
                None,
                None,
                None,
                Some(&[first, second]),
            )
            .unwrap(),
            (1360, 768)
        );
    }

    #[test]
    fn effective_dimensions_qwen_image_edit_requires_input_image() {
        let config = Config::default();
        let model_cfg = ModelConfig::default();
        let err = effective_dimensions(
            &config,
            &model_cfg,
            Some("qwen-image-edit"),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("requires at least one input image"));
    }

    fn refit_req(width: u32, height: u32, source_image: Option<Vec<u8>>) -> GenerateRequest {
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{
                "prompt":"p",
                "model":"flux-dev:q4",
                "width":0,
                "height":0,
                "steps":50,
                "guidance":1.0
            }"#,
        )
        .unwrap();
        req.width = width;
        req.height = height;
        req.source_image = source_image;
        req
    }

    /// After an auto-pull refreshes the model config, an img2img request
    /// whose dimensions were defaulted must be re-fit to the refreshed
    /// model canvas (this used to be two copy-pasted blocks inside
    /// prepare_local_engine).
    #[test]
    fn refit_request_after_pull_fits_source_after_autopull() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            default_steps: Some(8),
            default_guidance: Some(3.5),
            ..ModelConfig::default()
        };
        let source = png_with_dimensions(1280, 704);
        let mut req = refit_req(512, 512, Some(source));

        refit_request_after_pull(&mut req, &config, &model_cfg, None, None, None, None, None)
            .unwrap();
        // Same fit as effective_dimensions: width-limited 1024, 560.
        assert_eq!((req.width, req.height), (1024, 560));
        assert_eq!(req.steps, 8);
        assert_eq!(req.guidance, 3.5);
    }

    /// Explicit CLI flags always win over refreshed model defaults.
    #[test]
    fn refit_keeps_explicit_cli_dims_and_params() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1024),
            default_height: Some(1024),
            default_steps: Some(8),
            default_guidance: Some(3.5),
            ..ModelConfig::default()
        };
        let mut req = refit_req(640, 480, None);
        req.steps = 12;
        req.guidance = 7.0;

        refit_request_after_pull(
            &mut req,
            &config,
            &model_cfg,
            None,
            Some(640),
            Some(480),
            Some(12),
            Some(7.0),
        )
        .unwrap();
        assert_eq!((req.width, req.height), (640, 480));
        assert_eq!(req.steps, 12);
        assert_eq!(req.guidance, 7.0);
    }

    /// txt2img with defaulted dims re-derives from the refreshed model
    /// defaults (previously left at the pre-pull values).
    #[test]
    fn refit_rederives_txt2img_dims_from_refreshed_defaults() {
        let config = Config::default();
        let model_cfg = ModelConfig {
            default_width: Some(1216),
            default_height: Some(704),
            ..ModelConfig::default()
        };
        let mut req = refit_req(1024, 1024, None);
        refit_request_after_pull(&mut req, &config, &model_cfg, None, None, None, None, None)
            .unwrap();
        assert_eq!((req.width, req.height), (1216, 704));
    }

    #[test]
    fn effective_negative_prompt_injects_space_for_qwen_image_edit_cfg() {
        assert_eq!(
            effective_negative_prompt(Some("qwen-image-edit"), 4.0, None).as_deref(),
            Some(" ")
        );
    }

    #[test]
    fn effective_negative_prompt_preserves_explicit_or_non_edit_values() {
        assert_eq!(
            effective_negative_prompt(Some("qwen-image-edit"), 4.0, Some("keep".to_string()))
                .as_deref(),
            Some("keep")
        );
        assert_eq!(effective_negative_prompt(Some("flux"), 4.0, None), None);
        assert_eq!(
            effective_negative_prompt(Some("qwen-image-edit"), 1.0, None),
            None
        );
    }

    #[test]
    fn qwen_image_edit_rejects_cli_batch_above_one() {
        let err = validate_cli_batch_for_family(Some("qwen-image-edit"), 2).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("qwen-image-edit only supports --batch 1"));
    }

    #[test]
    fn cli_batch_guard_allows_single_edit_and_other_families() {
        validate_cli_batch_for_family(Some("qwen-image-edit"), 1).unwrap();
        validate_cli_batch_for_family(Some("flux"), 4).unwrap();
        validate_cli_batch_for_family(None, 4).unwrap();
    }

    #[test]
    fn ghostty_preview_uses_direct_kitty_backend() {
        assert_eq!(
            preview_backend(Some("ghostty"), Some("xterm-ghostty")),
            PreviewBackend::GhosttyKitty
        );
        assert_eq!(
            preview_backend(Some("ghostty"), Some("xterm-256color")),
            PreviewBackend::GhosttyKitty
        );
        assert_eq!(
            preview_backend(None, Some("xterm-ghostty")),
            PreviewBackend::GhosttyKitty
        );
    }

    #[test]
    fn non_ghostty_preview_uses_viuer_auto() {
        assert_eq!(
            preview_backend(Some("WezTerm"), Some("xterm-256color")),
            PreviewBackend::ViuerAuto
        );
        assert_eq!(
            preview_backend(None, Some("xterm-256color")),
            PreviewBackend::ViuerAuto
        );
    }

    #[test]
    fn fit_preview_cells_keeps_last_row_free_when_using_full_height() {
        assert_eq!(fit_preview_cells(80, 48, 80, 24), (80, 23));
        assert_eq!(fit_preview_cells(40, 20, 80, 24), (40, 10));
    }

    #[test]
    fn ghostty_kitty_preview_payload_uses_png_chunks() {
        let payload = ghostty_kitty_preview_chunks(&"A".repeat(5000), 40, 20);
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=40,r=20,m=1;"));
        assert!(payload.contains("\u{1b}_Gm=0;"));
    }

    #[test]
    fn ghostty_kitty_preview_payload_uses_single_chunk_when_small() {
        let payload = ghostty_kitty_preview_chunks("AAAA", 12, 6);
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=12,r=6,m=0;AAAA\u{1b}\\"));
        assert!(!payload.contains("\u{1b}_Gm=1;"));
    }

    #[test]
    fn build_ghostty_kitty_preview_payload_scales_wide_images_to_terminal_width() {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(200, 100));
        let (payload, cell_w, cell_h) =
            build_ghostty_kitty_preview_payload(&img, 80, 24).expect("payload should build");
        assert_eq!((cell_w, cell_h), (80, 20));
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=80,r=20,"));
    }

    #[test]
    fn build_ghostty_kitty_preview_payload_leaves_one_row_for_prompt_when_full_height() {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(80, 48));
        let (payload, cell_w, cell_h) =
            build_ghostty_kitty_preview_payload(&img, 80, 24).expect("payload should build");
        assert_eq!((cell_w, cell_h), (80, 23));
        assert!(payload.starts_with("\u{1b}_Gf=100,a=T,t=d,c=80,r=23,"));
    }

    #[test]
    fn preview_loop_count_replays_short_clips() {
        // ≤1.5s gets 3 plays, ≤3s gets 2, longer plays once.
        assert_eq!(preview_loop_count(Duration::from_millis(500)), 3);
        assert_eq!(preview_loop_count(Duration::from_millis(1499)), 3);
        assert_eq!(preview_loop_count(Duration::from_millis(1500)), 2);
        assert_eq!(preview_loop_count(Duration::from_millis(2999)), 2);
        assert_eq!(preview_loop_count(Duration::from_millis(3000)), 1);
        assert_eq!(preview_loop_count(Duration::from_secs(30)), 1);
    }

    #[test]
    fn forced_local_validation_passes_family_hint() {
        // `cv:` / `hf:` catalog IDs are absent from the static manifest, so the
        // forced-local path must feed the config-resolved family through or
        // every family-gated LTX-2 feature fails with "unknown model family".
        let mut config = Config::default();
        config.models.insert(
            "cv:2781713".to_string(),
            ModelConfig {
                family: Some("ltx2".to_string()),
                ..ModelConfig::default()
            },
        );
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a red apple",
            "model": "cv:2781713",
            "width": 960,
            "height": 576,
            "steps": 8,
            "guidance": 3.0,
            "batch_size": 1,
            "output_format": "mp4",
            "enable_audio": true
        }))
        .unwrap();

        // Sanity: without the hint the family gate rejects it.
        assert!(mold_core::validate_generate_request(&request).is_err());
        validate_local_request(&request, &config).unwrap();
    }

    #[test]
    fn forced_local_validation_still_rejects_invalid_requests() {
        let config = Config::default();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "",
            "model": "flux-dev:q4",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1
        }))
        .unwrap();

        assert!(validate_local_request(&request, &config)
            .unwrap_err()
            .to_string()
            .contains("prompt"));
    }

    /// Metadata must survive a zero-length prompt on every embedded surface.
    ///
    /// `mold-inference`'s `add_metadata_chunks` writes `mold:prompt` as an
    /// UNCONDITIONAL iTXt chunk (unlike the neighbouring `Option`-guarded
    /// fields), and the JPEG COM writer interpolates the prompt unconditionally
    /// too. This pins both encoders' tolerance of `""` so an optional-prompt
    /// image-to-video run can't silently lose its provenance. The JPEG XMP
    /// packet is not covered here — nothing in mold reads it back, and an empty
    /// `<mold:prompt></mold:prompt>` element is well-formed XML.
    #[test]
    fn empty_prompt_metadata_round_trips_through_png_and_jpeg() {
        use std::io::Write;

        // `OutputMetadata` has no `Default`; start from the synthesized shape
        // and fill in the fields the round-trip asserts on.
        let mut metadata = mold_db::metadata_io::synthesize_from_filename("mold-ltx2-1.png", 1);
        metadata.prompt = String::new();
        metadata.model = "ltx-2-19b-distilled:fp8".to_string();
        metadata.seed = 7;
        metadata.steps = 8;
        metadata.guidance = 3.0;
        metadata.width = 64;
        metadata.height = 64;
        metadata.version = "test".to_string();
        let json = serde_json::to_string(&metadata).unwrap();
        let dir = tempfile::tempdir().unwrap();

        // PNG: zero-length iTXt payloads must encode and decode.
        let png_path = dir.path().join("still.png");
        {
            let file = std::fs::File::create(&png_path).unwrap();
            let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .add_itxt_chunk("mold:prompt".to_string(), metadata.prompt.clone())
                .unwrap();
            encoder
                .add_itxt_chunk("mold:parameters".to_string(), json.clone())
                .unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[0u8, 0, 0]).unwrap();
            writer.finish().unwrap();
        }
        let recovered =
            mold_db::metadata_io::read_embedded(&png_path, mold_core::OutputFormat::Png)
                .expect("png metadata should decode with an empty prompt");
        assert_eq!(recovered.prompt, "");
        assert_eq!(recovered.model, "ltx-2-19b-distilled:fp8");

        // JPEG: the COM marker carries `mold:parameters {json}` verbatim, so an
        // empty prompt only shortens the JSON.
        let jpeg_path = dir.path().join("still.jpg");
        {
            let comment = format!("mold:parameters {json}");
            let mut out = vec![0xFF, 0xD8];
            out.push(0xFF);
            out.push(0xFE);
            let len = (comment.len() + 2) as u16;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(comment.as_bytes());
            out.extend_from_slice(&minimal_jpeg_body());
            std::fs::File::create(&jpeg_path)
                .unwrap()
                .write_all(&out)
                .unwrap();
        }
        let recovered =
            mold_db::metadata_io::read_embedded(&jpeg_path, mold_core::OutputFormat::Jpeg)
                .expect("jpeg metadata should decode with an empty prompt");
        assert_eq!(recovered.prompt, "");
        assert_eq!(recovered.seed, 7);
    }

    /// A 1x1 baseline JPEG minus its leading SOI marker, produced by the
    /// `image` crate so the COM-marker test has a real scan to append.
    fn minimal_jpeg_body() -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(1, 1)
            .write_with_encoder(image::codecs::jpeg::JpegEncoder::new(&mut buf))
            .unwrap();
        buf.into_inner()[2..].to_vec()
    }
}
