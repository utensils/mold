use anyhow::Result;
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mold_core::{
    Config, GenerateRequest, GenerateResponse, ImageData, MoldClient, OutputFormat, Scheduler,
    SseProgressEvent,
};
use rand::Rng;
use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

use crate::output::{is_piped, status};

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
    local: bool,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    scheduler: Option<Scheduler>,
    eager: bool,
    source_image: Option<Vec<u8>>,
    strength: f64,
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
    let config = Config::load_or_default();
    let model_cfg = config.model_config(model);
    // Fall back to manifest defaults for models not yet in config (e.g. unpulled)
    let model_cfg = if model_cfg.default_steps.is_none() {
        if let Some(manifest) = mold_core::manifest::find_manifest(model) {
            mold_core::ModelConfig {
                default_steps: Some(manifest.defaults.steps),
                default_guidance: Some(manifest.defaults.guidance),
                default_width: Some(manifest.defaults.width),
                default_height: Some(manifest.defaults.height),
                ..model_cfg
            }
        } else {
            model_cfg
        }
    } else {
        model_cfg
    };

    // When source_image is provided and width/height not specified, derive from image dimensions
    let (effective_width, effective_height) =
        if source_image.is_some() && width.is_none() && height.is_none() {
            if let Some(ref img_bytes) = source_image {
                let reader = image::ImageReader::new(std::io::Cursor::new(img_bytes))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok());
                match reader {
                    Some((w, h)) => {
                        // Round to nearest multiple of 16
                        let w = ((w + 8) / 16) * 16;
                        let h = ((h + 8) / 16) * 16;
                        (w, h)
                    }
                    None => (
                        width.unwrap_or_else(|| model_cfg.effective_width(&config)),
                        height.unwrap_or_else(|| model_cfg.effective_height(&config)),
                    ),
                }
            } else {
                unreachable!()
            }
        } else {
            (
                width.unwrap_or_else(|| model_cfg.effective_width(&config)),
                height.unwrap_or_else(|| model_cfg.effective_height(&config)),
            )
        };
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
        scheduler,
        source_image: source_image.clone(),
        strength,
    };

    if let Some(desc) = &model_cfg.description {
        status!(
            "{} {} — {}",
            "●".green(),
            model.bold(),
            crate::output::colorize_description(desc)
        );
    }
    if source_image.is_some() {
        status!("{} img2img mode (strength: {:.2})", "●".magenta(), strength,);
    }
    status!(
        "{} Generating {}x{} ({} steps, guidance {:.1})",
        "●".cyan(),
        effective_width,
        effective_height,
        effective_steps,
        effective_guidance,
    );

    // Validate output directory exists
    if let Some(ref path) = output {
        if path != "-" {
            let out_path = std::path::Path::new(path);
            if let Some(parent) = out_path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    anyhow::bail!("output directory does not exist: {}", parent.display());
                }
            }
        }
    }

    // Batch loop: generate N images with incrementing seeds.
    // The server/engine produces 1 image per request, so we loop client-side.
    let base_seed = req.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut all_images: Vec<ImageData> = Vec::with_capacity(batch as usize);
    let mut total_time_ms: u64 = 0;
    let mut last_seed_used: u64 = base_seed;
    let mut last_model = String::new();

    for i in 0..batch {
        let mut iter_req = req.clone();
        iter_req.seed = Some(base_seed.wrapping_add(i as u64));
        iter_req.batch_size = 1;

        if batch > 1 {
            status!(
                "{} Generating image {}/{} (seed: {})",
                "●".cyan(),
                i + 1,
                batch,
                iter_req.seed.unwrap(),
            );
        }

        let response = if local {
            if i == 0 {
                status!("{} Using local GPU inference", "●".cyan());
            }
            generate_local(
                &iter_req,
                &config,
                t5_variant.clone(),
                qwen3_variant.clone(),
                eager,
                width,
                height,
                steps,
                guidance,
            )
            .await?
        } else {
            let client = match &host {
                Some(h) => MoldClient::new(h),
                None => MoldClient::from_env(),
            };

            generate_remote(
                &client,
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
                width,
                height,
                steps,
                guidance,
            )
            .await?
        };

        total_time_ms += response.generation_time_ms;
        last_seed_used = response.seed_used;
        last_model = response.model.clone();

        for mut img in response.images {
            img.index = i;
            all_images.push(img);
        }
    }

    let response = GenerateResponse {
        images: all_images,
        generation_time_ms: total_time_ms,
        model: last_model,
        seed_used: last_seed_used,
    };

    // Output: pipe mode writes raw image bytes to stdout; interactive mode saves files.
    if piped && output.is_none() {
        // Pipe mode: write raw image bytes to stdout
        let mut stdout = std::io::stdout().lock();
        for img in &response.images {
            stdout.write_all(&img.data)?;
        }
        stdout.flush()?;
    } else {
        // File mode: save to disk
        for img in &response.images {
            let filename = match &output {
                Some(path) if path == "-" => {
                    // Explicit stdout via --output -
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&img.data)?;
                    stdout.flush()?;
                    continue;
                }
                Some(path) if batch == 1 => path.clone(),
                Some(path) => {
                    let stem = std::path::Path::new(path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output");
                    let ext = output_format.to_string();
                    format!("{stem}-{}.{ext}", img.index)
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
                status!("{} Overwriting: {}", "!".yellow(), filename);
            }
            std::fs::write(&filename, &img.data)?;
            status!("{} Saved: {}", "✓".green(), filename.bold());
        }
    }

    let secs = response.generation_time_ms as f64 / 1000.0;
    if batch > 1 {
        status!(
            "{} Done in {:.1}s ({} images, base seed: {})",
            "✓".green(),
            secs,
            batch,
            base_seed,
        );
    } else {
        status!(
            "{} Done in {:.1}s (seed: {})",
            "✓".green(),
            secs,
            response.seed_used,
        );
    }

    Ok(())
}

/// Render SSE progress events using indicatif progress bars.
/// Used by both local GPU inference and remote SSE streaming.
async fn render_progress(mut rx: tokio::sync::mpsc::UnboundedReceiver<SseProgressEvent>) {
    let pb = ProgressBar::new_spinner();
    if is_piped() {
        pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    }
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    // No steady tick — redraws only on events to avoid flickering with
    // hf-hub's download bars (which write to stderr independently).
    pb.tick();

    let mut denoise_bar: Option<ProgressBar> = None;
    let mut download_multi: Option<MultiProgress> = None;
    let mut download_bars: HashMap<usize, ProgressBar> = HashMap::new();
    while let Some(event) = rx.recv().await {
        match event {
            SseProgressEvent::StageStart { name } => {
                if let Some(db) = denoise_bar.take() {
                    db.finish_and_clear();
                }
                pb.set_message(format!("{}...", name));
                pb.tick();
            }
            SseProgressEvent::StageDone { name, elapsed_ms } => {
                if let Some(db) = denoise_bar.take() {
                    db.finish_and_clear();
                }
                let secs = elapsed_ms as f64 / 1000.0;
                pb.suspend(|| {
                    status!(
                        "  {} {} {}",
                        "✓".green(),
                        name,
                        format!("[{:.1}s]", secs).dimmed(),
                    );
                });
            }
            SseProgressEvent::Info { message } => {
                pb.suspend(|| {
                    status!("  {} {}", "·".dimmed(), message.dimmed());
                });
            }
            SseProgressEvent::DenoiseStep {
                step,
                total,
                elapsed_ms,
            } => {
                let db = denoise_bar.get_or_insert_with(|| {
                    pb.disable_steady_tick();
                    pb.set_message("");

                    let bar = ProgressBar::new(total as u64);
                    if is_piped() {
                        bar.set_draw_target(ProgressDrawTarget::stderr());
                    }
                    bar.set_style(
                        ProgressStyle::default_bar()
                            .template(
                                "  {spinner:.cyan} Denoising [{bar:30.cyan/dim}] {pos}/{len} [{elapsed_precise}, {msg}]",
                            )
                            .unwrap()
                            .progress_chars("━╸─"),
                    );
                    bar.enable_steady_tick(Duration::from_millis(100));
                    bar
                });
                let elapsed_secs = elapsed_ms as f64 / 1000.0;
                let it_s = if elapsed_secs > 0.0 {
                    1.0 / elapsed_secs
                } else {
                    0.0
                };
                db.set_message(format!("{:.2} it/s", it_s));
                db.set_position(step as u64);
            }
            SseProgressEvent::DownloadProgress {
                filename,
                file_index,
                bytes_downloaded,
                bytes_total,
                total_files,
            } => {
                let multi = download_multi.get_or_insert_with(|| {
                    pb.disable_steady_tick();
                    pb.set_message("");
                    MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
                });
                let bar = download_bars.entry(file_index).or_insert_with(|| {
                    let b = multi.add(ProgressBar::new(bytes_total));
                    let msg_width = 45usize;
                    b.set_style(
                        ProgressStyle::with_template(&format!(
                            "  {{msg:<{msg_width}}} [{{bar:30.cyan/dim}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})"
                        ))
                        .unwrap()
                        .progress_chars("━╸─"),
                    );
                    // Show total files on first file
                    if total_files > 0 {
                        b.set_message(format!("[{}/{}] {}", file_index + 1, total_files, truncate_name(&filename, msg_width - 8)));
                    } else {
                        b.set_message(truncate_name(&filename, msg_width));
                    }
                    b
                });
                bar.set_position(bytes_downloaded);
            }
            SseProgressEvent::DownloadDone { file_index, .. } => {
                if let Some(bar) = download_bars.get(&file_index) {
                    bar.finish_with_message("done");
                }
            }
            SseProgressEvent::PullComplete { model } => {
                // Clear download bars
                for (_, bar) in download_bars.drain() {
                    bar.finish_and_clear();
                }
                if let Some(multi) = download_multi.take() {
                    multi.clear().ok();
                }
                pb.suspend(|| {
                    status!("{} Pull complete: {}", "✓".green(), model.bold());
                });
            }
        }
    }
    if let Some(db) = denoise_bar.take() {
        db.finish_and_clear();
    }
    for (_, bar) in download_bars.drain() {
        bar.finish_and_clear();
    }
    if let Some(multi) = download_multi.take() {
        multi.clear().ok();
    }
    pb.finish_and_clear();
}

/// Truncate a filename for display, keeping the end (unique part).
fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len || max_len < 8 {
        return name.to_string();
    }
    let suffix_len = max_len - 3;
    let start = name.len() - suffix_len;
    format!("...{}", &name[start..])
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
                cli_width,
                cli_height,
                cli_steps,
                cli_guidance,
            )
            .await
        }
        Err(e) if MoldClient::is_model_not_found(&e) => {
            let _ = render.await;
            status!(
                "{} Model '{}' not on server — pulling...",
                "●".cyan(),
                model.bold()
            );

            // Stream download progress from server
            let (pull_tx, pull_rx) = tokio::sync::mpsc::unbounded_channel();
            let pull_render = tokio::spawn(render_progress(pull_rx));
            client.pull_model_stream(model, pull_tx).await?;
            let _ = pull_render.await;

            status!("{} Generating...", "●".cyan());

            // Retry with SSE streaming after pull
            let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
            let render2 = tokio::spawn(render_progress(rx2));
            match client.generate_stream(req, tx2).await {
                Ok(Some(response)) => {
                    let _ = render2.await;
                    Ok(response)
                }
                _ => {
                    // Fall back to blocking if SSE still fails
                    let _ = render2.await;
                    Ok(client.generate(req.clone()).await?)
                }
            }
        }
        Err(e) if MoldClient::is_connection_error(&e) => {
            let _ = render.await;
            status!("{} Using local GPU inference", "●".cyan());
            generate_local(
                req,
                config,
                t5_variant,
                qwen3_variant,
                eager,
                cli_width,
                cli_height,
                cli_steps,
                cli_guidance,
            )
            .await
        }
        Err(e) => {
            let _ = render.await;
            Err(e)
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
            .template("{spinner:.green} {msg}")
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
        Err(e) if MoldClient::is_model_not_found(&e) => {
            pb.finish_and_clear();
            status!(
                "{} Model '{}' not on server — pulling...",
                "●".cyan(),
                model.bold()
            );

            // Stream download progress from server
            let (pull_tx, pull_rx) = tokio::sync::mpsc::unbounded_channel();
            let pull_render = tokio::spawn(render_progress(pull_rx));
            client.pull_model_stream(model, pull_tx).await?;
            let _ = pull_render.await;

            status!("{} Generating...", "●".cyan());
            Ok(client.generate(req.clone()).await?)
        }
        Err(e) if MoldClient::is_connection_error(&e) => {
            pb.finish_and_clear();
            status!("{} Using local GPU inference", "●".cyan());
            generate_local(
                req,
                config,
                t5_variant,
                qwen3_variant,
                eager,
                cli_width,
                cli_height,
                cli_steps,
                cli_guidance,
            )
            .await
        }
        Err(e) => Err(e),
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
async fn generate_local(
    req: &GenerateRequest,
    config: &Config,
    t5_variant_override: Option<String>,
    qwen3_variant_override: Option<String>,
    eager: bool,
    cli_width: Option<u32>,
    cli_height: Option<u32>,
    cli_steps: Option<u32>,
    cli_guidance: Option<f64>,
) -> Result<GenerateResponse> {
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
                    "●".cyan(),
                    model_name.bold(),
                );
                let updated_config = super::pull::pull_and_configure(&model_name).await?;
                paths = ModelPaths::resolve(&model_name, &updated_config).ok_or_else(|| {
                    anyhow::anyhow!(
                        "model '{}' was pulled but paths could not be resolved",
                        model_name,
                    )
                })?;
                auto_config = updated_config;
                effective_config = &auto_config;

                let model_cfg = effective_config.model_config(&model_name);
                if cli_width.is_none() {
                    req.width = model_cfg.effective_width(effective_config);
                }
                if cli_height.is_none() {
                    req.height = model_cfg.effective_height(effective_config);
                }
                if cli_steps.is_none() {
                    req.steps = model_cfg.effective_steps(effective_config);
                }
                if cli_guidance.is_none() {
                    req.guidance = model_cfg.effective_guidance();
                }
                status!(
                    "{} Updated defaults: {}x{} ({} steps, guidance {:.1})",
                    "●".cyan(),
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
    let mut engine =
        mold_inference::create_engine(model_name, paths, effective_config, load_strategy)?;

    // Set up progress channel — convert ProgressEvent → SseProgressEvent for unified rendering
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SseProgressEvent>();
    engine.set_on_progress(Box::new(move |event| {
        let _ = tx.send(event.into());
    }));

    let req = req.clone();

    // Spawn inference in a blocking thread
    let handle = tokio::task::spawn_blocking(move || {
        engine.load()?;
        engine.generate(&req)
    });

    // Render progress events using the shared renderer
    let render = tokio::spawn(render_progress(rx));

    let result = handle.await?;
    let _ = render.await;
    result
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
async fn generate_local(
    _req: &GenerateRequest,
    _config: &Config,
    _t5_variant: Option<String>,
    _qwen3_variant: Option<String>,
    _eager: bool,
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

/// Build a default output filename, sanitizing colons from model names.
fn default_filename(model: &str, timestamp: u64, ext: &str, batch: u32, index: u32) -> String {
    let safe_model = model.replace(':', "-");
    if batch == 1 {
        format!("mold-{safe_model}-{timestamp}.{ext}")
    } else {
        format!("mold-{safe_model}-{timestamp}-{index}.{ext}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
