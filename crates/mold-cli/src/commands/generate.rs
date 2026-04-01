use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{
    classify_generate_error, fit_to_model_dimensions, Config, GenerateRequest, GenerateResponse,
    GenerateServerAction, ImageData, LoraWeight, MoldClient, OutputFormat, Scheduler,
};
use rand::Rng;
use std::io::Write;
use std::time::Duration;

use crate::control::{stream_server_pull, CliContext};
use crate::output::{is_piped, status};
use crate::theme;
use crate::ui::{print_server_pull_missing_model, print_using_local_inference, render_progress};

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

fn effective_dimensions(
    config: &Config,
    model_cfg: &mold_core::ModelConfig,
    width: Option<u32>,
    height: Option<u32>,
    source_image: Option<&[u8]>,
) -> Result<(u32, u32)> {
    match (width, height, source_image) {
        (Some(width), Some(height), _) => Ok((width, height)),
        (Some(width), None, _) => Ok((width, model_cfg.effective_height(config))),
        (None, Some(height), _) => Ok((model_cfg.effective_width(config), height)),
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
    host: Option<String>,
    format: OutputFormat,
    no_metadata: bool,
    preview: bool,
    local: bool,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    scheduler: Option<Scheduler>,
    eager: bool,
    offload: bool,
    source_image: Option<Vec<u8>>,
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
    let output_format = format;
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

    // Load config and pull model-specific defaults.
    let ctx = CliContext::new(host.as_deref());
    let config = ctx.config().clone();
    // `--no-metadata` is an opt-out override, so we only pass `Some(false)` when set.
    // Otherwise we defer to env/config/default precedence inside Config.
    let embed_metadata = config.effective_embed_metadata(no_metadata.then_some(false));
    let model_cfg = config.resolved_model_config(model);

    // Default to the source image size for img2img/inpainting when neither
    // dimension was provided. We still normalize to the validation envelope
    // (multiples of 16, megapixel clamp) to avoid invalid requests and OOMs.
    let (effective_width, effective_height) =
        effective_dimensions(&config, &model_cfg, width, height, source_image.as_deref())?;
    let effective_steps = steps.unwrap_or_else(|| model_cfg.effective_steps(&config));
    let effective_guidance = guidance.unwrap_or_else(|| model_cfg.effective_guidance());

    let req = GenerateRequest {
        prompt: prompt.to_string(),
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
        source_image: source_image.clone(),
        strength,
        mask_image: mask_image.clone(),
        control_image: control_image.clone(),
        control_model: control_model.clone(),
        control_scale,
        negative_prompt: negative_prompt.clone(),
        expand,
        original_prompt,
        lora: lora.clone(),
    };

    // Warn if user-provided dimensions don't match model recommendations.
    // Only warn locally — in remote mode the server sends the warning via SSE/header.
    if local && (width.is_some() || height.is_some()) {
        if let Some(ref family) = model_cfg.family {
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
    if let Some(ref neg) = negative_prompt {
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
            t5_variant.clone(),
            qwen3_variant.clone(),
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
                t5_variant.clone(),
                qwen3_variant.clone(),
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

            for mut img in response.images {
                img.index = i;
                // Save and preview each image immediately during batch generation
                // (single-image mode is handled in the post-loop section)
                if batch > 1 && (!piped || output.is_some()) {
                    save_and_preview_image(&img, &output, model, batch, output_format, preview)?;
                }
                all_images.push(img);
            }
        }

        GenerateResponse {
            images: all_images,
            generation_time_ms: total_time_ms,
            model: last_model,
            seed_used: last_seed_used,
        }
    };

    // Output: pipe mode writes raw image bytes to stdout; interactive mode saves files.
    // For batch > 1, images were already saved inside the batch loop above.
    if piped && output.is_none() {
        // Pipe mode: write raw image bytes to stdout
        let mut stdout = std::io::stdout().lock();
        for img in &response.images {
            stdout.write_all(&img.data)?;
        }
        stdout.flush()?;
    } else if batch == 1 {
        // Single image: save and preview now
        for img in &response.images {
            save_and_preview_image(img, &output, model, batch, output_format, preview)?;
        }
    }
    // batch > 1: already saved+previewed inside the batch loop

    let secs = response.generation_time_ms as f64 / 1000.0;
    if batch > 1 {
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
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
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
                t5_variant,
                qwen3_variant,
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
                    stream_server_pull(client, model).await?;

                    status!("{} Generating...", theme::icon_info());

                    let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
                    let render2 = tokio::spawn(render_progress(rx2));
                    match client.generate_stream(req, tx2).await {
                        Ok(Some(response)) => {
                            let _ = render2.await;
                            Ok(response)
                        }
                        _ => {
                            let _ = render2.await;
                            Ok(client.generate(req.clone()).await?)
                        }
                    }
                }
                GenerateServerAction::FallbackLocal => {
                    print_using_local_inference();
                    generate_local(
                        req,
                        config,
                        t5_variant,
                        qwen3_variant,
                        eager,
                        offload,
                        cli_width,
                        cli_height,
                        cli_steps,
                        cli_guidance,
                    )
                    .await
                }
                GenerateServerAction::SurfaceError => Err(e),
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
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
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
                    stream_server_pull(client, model).await?;
                    status!("{} Generating...", theme::icon_info());
                    Ok(client.generate(req.clone()).await?)
                }
                GenerateServerAction::FallbackLocal => {
                    print_using_local_inference();
                    generate_local(
                        req,
                        config,
                        t5_variant,
                        qwen3_variant,
                        eager,
                        offload,
                        cli_width,
                        cli_height,
                        cli_steps,
                        cli_guidance,
                    )
                    .await
                }
                GenerateServerAction::SurfaceError => Err(e),
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
async fn prepare_local_engine(
    req: &GenerateRequest,
    config: &Config,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
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
    match ModelPaths::resolve(&model_name, config) {
        Some(p) => {
            paths = p;
            effective_config = config;
        }
        None => {
            // Auto-pull: if a manifest exists, download the model automatically
            if find_manifest(&model_name).is_some() {
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
        }
    }

    validate_generate_request(&req).map_err(|e| anyhow::anyhow!(e))?;

    // Apply CLI variant overrides via env vars (factory reads MOLD_T5_VARIANT / MOLD_QWEN3_VARIANT)
    if let Some(ref variant) = t5_variant_override {
        std::env::set_var("MOLD_T5_VARIANT", variant);
    }
    if let Some(ref variant) = qwen3_variant_override {
        std::env::set_var("MOLD_QWEN3_VARIANT", variant);
    }
    let is_eager = eager || std::env::var("MOLD_EAGER").map_or(false, |v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    if is_eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    let is_offload = offload || std::env::var("MOLD_OFFLOAD").map_or(false, |v| v == "1");
    let engine = mold_inference::create_engine(
        model_name,
        paths,
        effective_config,
        load_strategy,
        is_offload,
    )?;
    Ok((req, engine))
}

#[cfg(any(feature = "cuda", feature = "metal"))]
async fn generate_local(
    req: &GenerateRequest,
    config: &Config,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
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
        t5_variant_override,
        qwen3_variant_override,
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
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
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
        t5_variant_override,
        qwen3_variant_override,
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
    let mut total_time_ms = 0;
    let mut last_seed_used = base_seed;
    let mut last_model = String::new();

    for i in 0..batch {
        let mut iter_req = base_req.clone();
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

        for mut img in response.images {
            img.index = i;
            // Save and preview each image immediately during batch generation
            if batch > 1 {
                save_and_preview_image(&img, output, &req.model, batch, output_format, preview)?;
            }
            all_images.push(img);
        }
    }

    Ok(GenerateResponse {
        images: all_images,
        generation_time_ms: total_time_ms,
        model: last_model,
        seed_used: last_seed_used,
    })
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
async fn generate_local(
    _req: &GenerateRequest,
    _config: &Config,
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
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
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
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

/// Save a single image to disk and optionally preview it inline.
///
/// Resolves the output filename from the `--output` flag, batch index,
/// model name, and output format. Used both inside batch loops (for
/// immediate save+preview) and for single-image output.
fn save_and_preview_image(
    img: &ImageData,
    output: &Option<String>,
    model: &str,
    batch: u32,
    output_format: OutputFormat,
    preview: bool,
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
            let parent = p.parent().and_then(|d| {
                let s = d.to_string_lossy();
                if s.is_empty() {
                    None
                } else {
                    Some(format!("{}/", s))
                }
            });
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let ext = output_format.to_string();
            format!("{}{stem}-{}.{ext}", parent.unwrap_or_default(), img.index)
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
    if preview {
        preview_image(&img.data);
    }
    Ok(())
}

/// Display an image inline in the terminal using viuer.
/// Silently skipped if the `preview` feature is not compiled or if decoding fails.
#[cfg(feature = "preview")]
fn preview_image(data: &[u8]) {
    let Ok(img) = image::load_from_memory(data) else {
        return;
    };
    let conf = viuer::Config {
        absolute_offset: false,
        ..Default::default()
    };
    let _ = viuer::print(&img, &conf);
}

#[cfg(not(feature = "preview"))]
fn preview_image(_data: &[u8]) {
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
            effective_dimensions(&config, &model_cfg, None, None, None).unwrap(),
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
            effective_dimensions(&config, &model_cfg, None, None, Some(&source)).unwrap(),
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
            effective_dimensions(&config, &model_cfg, None, None, Some(&source)).unwrap(),
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
            effective_dimensions(&config, &model_cfg, Some(512), Some(768), Some(&source)).unwrap(),
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
            effective_dimensions(&config, &model_cfg, None, None, Some(&source)).unwrap(),
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
            effective_dimensions(&config, &model_cfg, None, None, Some(&source)).unwrap(),
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
            effective_dimensions(&config, &model_cfg, None, None, Some(&source)).unwrap(),
            (1024, 1024)
        );
    }
}
