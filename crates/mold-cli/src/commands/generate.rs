use anyhow::Result;
#[cfg(any(feature = "preview", test))]
use base64::{engine::general_purpose, Engine as _};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{
    classify_generate_error, fit_to_model_dimensions, fit_to_target_area, manifest, Config,
    GenerateRequest, GenerateResponse, GenerateServerAction, ImageData, KeyframeCondition,
    LoraWeight, Ltx2PipelineMode, Ltx2SpatialUpscale, Ltx2TemporalUpscale, MoldClient,
    OutputFormat, Scheduler, TimeRange,
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

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn apply_local_engine_env_overrides(
    t5_variant_override: Option<&str>,
    qwen3_variant_override: Option<&str>,
    qwen2_variant_override: Option<&str>,
    qwen2_text_encoder_mode_override: Option<&str>,
) {
    if let Some(variant) = t5_variant_override {
        std::env::set_var("MOLD_T5_VARIANT", variant);
    }
    if let Some(variant) = qwen3_variant_override {
        std::env::set_var("MOLD_QWEN3_VARIANT", variant);
    }
    if let Some(variant) = qwen2_variant_override {
        std::env::set_var("MOLD_QWEN2_VARIANT", variant);
    }
    if let Some(mode) = qwen2_text_encoder_mode_override {
        std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", mode);
    }
}

pub struct Ltx2Options {
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub enable_audio: Option<bool>,
    pub audio_file: Option<Vec<u8>>,
    pub source_video: Option<Vec<u8>>,
    pub keyframes: Option<Vec<KeyframeCondition>>,
    pub pipeline: Option<Ltx2PipelineMode>,
    pub loras: Option<Vec<LoraWeight>>,
    pub retake_range: Option<TimeRange>,
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
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
    eager: bool,
    offload: bool,
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
        enable_audio,
        audio_file,
        source_video,
        keyframes,
        pipeline,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
    } = ltx2;

    // Load config and pull model-specific defaults.
    let ctx = CliContext::new(host.as_deref());
    let config = ctx.config().clone();
    // `--no-metadata` is an opt-out override, so we only pass `Some(false)` when set.
    // Otherwise we defer to env/config/default precedence inside Config.
    let embed_metadata = config.effective_embed_metadata(no_metadata.then_some(false));
    let model_cfg = config.resolved_model_config(model);
    let family = resolve_family(model, &config);
    let effective_frames = frames.or_else(|| model_cfg.effective_frames());
    let effective_fps = fps.or_else(|| model_cfg.effective_fps());
    let is_ltx2 = family.as_deref() == Some("ltx2");

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
    let piped = is_piped();

    // Reject batch > 1 when output goes to stdout (piped with no --output, or --output -)
    if batch > 1 {
        let stdout_output = (piped && output.is_none()) || output.as_deref() == Some("-");
        if stdout_output {
            anyhow::bail!(
                "--batch with more than 1 image is not supported with stdout output. \
                 Use --output <path> to save batch images to files."
            );
        }
    }

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

    let req = GenerateRequest {
        prompt: prompt.to_string(),
        negative_prompt: effective_negative_prompt.clone(),
        model: model.to_string(),
        width: effective_width,
        height: effective_height,
        steps: effective_steps,
        guidance: effective_guidance,
        seed,
        batch_size: batch,
        output_format,
        embed_metadata: Some(embed_metadata),
        scheduler,
        edit_images: edit_images.clone(),
        source_image: source_image.clone(),
        strength,
        mask_image: mask_image.clone(),
        control_image: control_image.clone(),
        control_model: control_model.clone(),
        control_scale,
        expand,
        original_prompt,
        lora: lora.clone(),
        frames: effective_frames,
        fps: effective_fps,
        upscale_model: None,
        gif_preview: preview,
        enable_audio,
        audio_file,
        source_video,
        keyframes,
        pipeline,
        loras,
        retake_range,
        spatial_upscale,
        temporal_upscale,
        placement: None,
    };

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

            // Capture video from the last response (video models produce one clip per run)
            if response.video.is_some() {
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
        if piped && output.is_none() {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&video.data)?;
            stdout.flush()?;
        } else {
            let filename = match output {
                Some(ref path) if path == "-" => {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&video.data)?;
                    stdout.flush()?;
                    None
                }
                Some(ref path) => Some(path.clone()),
                None => {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    Some(default_filename(
                        model,
                        timestamp,
                        video.format.extension(),
                        1,
                        0,
                    ))
                }
            };
            if let Some(ref filename) = filename {
                if std::path::Path::new(filename).exists() {
                    status!("{} Overwriting: {}", theme::icon_alert(), filename);
                }
                std::fs::write(filename, &video.data)?;
                status!(
                    "{} Saved: {} ({} frames, {}x{}, {} fps)",
                    theme::icon_done(),
                    filename.bold(),
                    video.frames,
                    video.width,
                    video.height,
                    video.fps,
                );
                crate::metadata_db::record_local_save(
                    std::path::Path::new(filename),
                    &req,
                    response.seed_used,
                    response.generation_time_ms,
                    video.format,
                );
            }
            if preview {
                // Show first frame preview (viuer doesn't support animation).
                // Fallback to the video data itself for GIF/APNG/WebP (decodable
                // as images) when thumbnail/gif_preview are absent (non-SSE path).
                if !video.gif_preview.is_empty() {
                    preview_image(&video.gif_preview);
                } else if !video.thumbnail.is_empty() {
                    preview_image(&video.thumbnail);
                } else {
                    preview_image(&video.data);
                }
            }
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
    Config::write_last_model(&response.model);

    Ok(())
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
                    generate_local(
                        req,
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
                    generate_local(
                        req,
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
async fn prepare_local_engine(
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
) -> Result<(GenerateRequest, Box<dyn mold_inference::InferenceEngine>)> {
    use mold_core::manifest::find_manifest;
    use mold_core::{validate_generate_request, ModelPaths};
    use mold_inference::LoadStrategy;

    let model_name = req.model.clone();
    let (paths, auto_config);
    let effective_config: &Config;
    let mut req = req.clone();
    if config.manifest_model_needs_download(&model_name) {
        status!(
            "{} Model '{}' is missing local assets, pulling repair...",
            theme::icon_info(),
            model_name.bold(),
        );
        let updated_config = super::pull::pull_and_configure(
            &model_name,
            &mold_core::download::PullOptions::default(),
        )
        .await?;
        paths = ModelPaths::resolve(&model_name, &updated_config).ok_or_else(|| {
            anyhow::anyhow!(
                "model '{}' was pulled but paths could not be resolved",
                model_name,
            )
        })?;
        auto_config = updated_config;
        effective_config = &auto_config;

        let model_cfg = effective_config.resolved_model_config(&model_name);
        let new_model_w = model_cfg.effective_width(effective_config);
        let new_model_h = model_cfg.effective_height(effective_config);
        if cli_width.is_none() && cli_height.is_none() {
            if let Some(src_bytes) = &req.source_image {
                // img2img with auto-pull: fit source to newly-discovered model defaults
                if let Ok(img) = image::load_from_memory(src_bytes) {
                    let (w, h) = fit_to_model_dimensions(
                        img.width(),
                        img.height(),
                        new_model_w,
                        new_model_h,
                    );
                    req.width = w;
                    req.height = h;
                }
            }
        } else {
            if cli_width.is_none() {
                req.width = new_model_w;
            }
            if cli_height.is_none() {
                req.height = new_model_h;
            }
        }
        if cli_steps.is_none() {
            req.steps = model_cfg.effective_steps(effective_config);
        }
        if cli_guidance.is_none() {
            req.guidance = model_cfg.effective_guidance();
        }
        status!(
            "{} Updated defaults: {}x{} ({} steps, guidance {:.1})",
            theme::icon_info(),
            req.width,
            req.height,
            req.steps,
            req.guidance,
        );
    } else if let Some(p) = ModelPaths::resolve(&model_name, config) {
        paths = p;
        effective_config = config;
    } else if find_manifest(&model_name).is_some() {
        status!(
            "{} Model '{}' not found locally, pulling...",
            theme::icon_info(),
            model_name.bold(),
        );
        let updated_config = super::pull::pull_and_configure(
            &model_name,
            &mold_core::download::PullOptions::default(),
        )
        .await?;
        paths = ModelPaths::resolve(&model_name, &updated_config).ok_or_else(|| {
            anyhow::anyhow!(
                "model '{}' was pulled but paths could not be resolved",
                model_name,
            )
        })?;
        auto_config = updated_config;
        effective_config = &auto_config;

        let model_cfg = effective_config.resolved_model_config(&model_name);
        let new_model_w = model_cfg.effective_width(effective_config);
        let new_model_h = model_cfg.effective_height(effective_config);
        if cli_width.is_none() && cli_height.is_none() {
            if let Some(src_bytes) = &req.source_image {
                // img2img with auto-pull: fit source to newly-discovered model defaults
                if let Ok(img) = image::load_from_memory(src_bytes) {
                    let (w, h) = fit_to_model_dimensions(
                        img.width(),
                        img.height(),
                        new_model_w,
                        new_model_h,
                    );
                    req.width = w;
                    req.height = h;
                }
            }
        } else {
            if cli_width.is_none() {
                req.width = new_model_w;
            }
            if cli_height.is_none() {
                req.height = new_model_h;
            }
        }
        if cli_steps.is_none() {
            req.steps = model_cfg.effective_steps(effective_config);
        }
        if cli_guidance.is_none() {
            req.guidance = model_cfg.effective_guidance();
        }
        status!(
            "{} Updated defaults: {}x{} ({} steps, guidance {:.1})",
            theme::icon_info(),
            req.width,
            req.height,
            req.steps,
            req.guidance,
        );
    } else {
        anyhow::bail!(
            "no model paths configured for '{}'. Add [models.{}] to ~/.mold/config.toml \
             or set MOLD_TRANSFORMER_PATH / MOLD_VAE_PATH / MOLD_T5_PATH / MOLD_CLIP_PATH \
             / MOLD_T5_TOKENIZER_PATH / MOLD_CLIP_TOKENIZER_PATH env vars.",
            model_name,
            model_name,
        );
    }

    validate_generate_request(&req).map_err(|e| anyhow::anyhow!(e))?;

    apply_local_engine_env_overrides(
        t5_variant_override.as_deref(),
        qwen3_variant_override.as_deref(),
        qwen2_variant_override.as_deref(),
        qwen2_text_encoder_mode_override.as_deref(),
    );
    let is_eager = eager || std::env::var("MOLD_EAGER").is_ok_and(|v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    if is_eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    let is_offload = offload || std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");

    // Select the best GPU from the allowed set (most free VRAM).
    let gpu_selection = match &gpus {
        Some(s) => mold_core::types::GpuSelection::parse(s)?,
        None => config.gpu_selection(),
    };
    let discovered = mold_inference::device::discover_gpus();
    let available = mold_inference::device::filter_gpus(&discovered, &gpu_selection);
    let gpu_ordinal = mold_inference::device::select_best_gpu(&available)
        .map(|g| g.ordinal)
        .unwrap_or(0);

    let engine = mold_inference::create_engine(
        model_name,
        paths,
        effective_config,
        load_strategy,
        gpu_ordinal,
        is_offload,
    )?;
    Ok((req, engine))
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
    let (req, mut engine) = prepare_local_engine(
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

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<mold_core::SseProgressEvent>();
    engine.set_on_progress(Box::new(move |event| {
        let _ = tx.send(event.into());
    }));

    let handle = tokio::task::spawn_blocking(move || {
        engine.load()?;
        engine.generate(&req)
    });

    let render = tokio::spawn(render_progress(rx));
    let result = handle.await?;
    let _ = render.await;
    result
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
    let (base_req, mut engine) = prepare_local_engine(
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

    engine = tokio::task::spawn_blocking(
        move || -> Result<Box<dyn mold_inference::InferenceEngine>> {
            let mut engine = engine;
            engine.load()?;
            Ok(engine)
        },
    )
    .await??;

    let mut all_images: Vec<ImageData> = Vec::with_capacity(batch as usize);
    let mut last_video: Option<mold_core::VideoData> = None;
    let mut total_time_ms = 0;
    let mut last_seed_used = base_seed;
    let mut last_model = String::new();

    for i in 0..batch {
        let mut iter_req = base_req.clone();
        iter_req.seed = Some(base_seed.wrapping_add(i as u64));
        iter_req.batch_size = 1;

        // Use per-batch expanded prompt if available
        if let Some(prompts) = batch_prompts {
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

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<mold_core::SseProgressEvent>();
        engine.set_on_progress(Box::new(move |event| {
            let _ = tx.send(event.into());
        }));

        let handle = tokio::task::spawn_blocking(
            move || -> Result<(Box<dyn mold_inference::InferenceEngine>, GenerateResponse)> {
                let mut engine = engine;
                let response = engine.generate(&iter_req)?;
                Ok((engine, response))
            },
        );
        let render = tokio::spawn(render_progress(rx));
        let (mut returned_engine, response) = handle.await??;
        returned_engine.clear_on_progress(); // drop tx so render_progress can drain and exit
        let _ = render.await;
        engine = returned_engine;

        total_time_ms += response.generation_time_ms;
        last_seed_used = response.seed_used;
        last_model = response.model.clone();

        // Capture video response if present
        if response.video.is_some() {
            last_video = response.video;
        }

        for mut img in response.images {
            img.index = i;
            // Save and preview each image immediately during batch generation
            // (single-image mode is handled by the caller's post-loop section)
            if batch > 1 {
                save_and_preview_image(
                    &img,
                    output,
                    &req.model,
                    batch,
                    output_format,
                    preview,
                    Some(PersistArgs {
                        request: req,
                        seed_used: response.seed_used,
                        generation_time_ms: response.generation_time_ms,
                    }),
                )?;
            }
            all_images.push(img);
        }
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
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let ext = output_format.to_string();
            default_filename(model, timestamp, &ext, batch, img.index)
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
    use crate::test_support::ENV_LOCK;
    use mold_core::ModelConfig;

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
    fn output_dash_is_special() {
        // "--output -" triggers stdout output in both interactive and piped modes
        let path = "-";
        assert_eq!(path, "-");
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
    fn apply_local_engine_env_overrides_sets_qwen2_overrides() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior_variant = std::env::var("MOLD_QWEN2_VARIANT").ok();
        let prior_mode = std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE").ok();

        std::env::remove_var("MOLD_QWEN2_VARIANT");
        std::env::remove_var("MOLD_QWEN2_TEXT_ENCODER_MODE");

        apply_local_engine_env_overrides(None, None, Some("q6"), Some("cpu-stage"));

        assert_eq!(
            std::env::var("MOLD_QWEN2_VARIANT").ok().as_deref(),
            Some("q6")
        );
        assert_eq!(
            std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE")
                .ok()
                .as_deref(),
            Some("cpu-stage")
        );

        match prior_variant {
            Some(value) => std::env::set_var("MOLD_QWEN2_VARIANT", value),
            None => std::env::remove_var("MOLD_QWEN2_VARIANT"),
        }
        match prior_mode {
            Some(value) => std::env::set_var("MOLD_QWEN2_TEXT_ENCODER_MODE", value),
            None => std::env::remove_var("MOLD_QWEN2_TEXT_ENCODER_MODE"),
        }
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
}
