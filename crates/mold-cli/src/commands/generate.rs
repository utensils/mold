use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{Config, GenerateRequest, GenerateResponse, MoldClient, OutputFormat};
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
    format: &str,
    local: bool,
    t5_variant: Option<String>,
    qwen3_variant: Option<String>,
    eager: bool,
) -> Result<()> {
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;
    let piped = is_piped();

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

    let effective_width = width.unwrap_or_else(|| model_cfg.effective_width(&config));
    let effective_height = height.unwrap_or_else(|| model_cfg.effective_height(&config));
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
    };

    if let Some(desc) = &model_cfg.description {
        status!(
            "{} {} — {}",
            "●".green(),
            model.bold(),
            crate::output::colorize_description(desc)
        );
    }
    status!(
        "{} Generating {}x{} ({} steps, guidance {:.1})",
        "●".cyan(),
        effective_width,
        effective_height,
        effective_steps,
        effective_guidance,
    );

    let response = if local {
        // --local: skip server, go straight to local inference
        status!("{} Using local GPU inference", "●".cyan());
        generate_local(
            &req,
            &config,
            t5_variant,
            qwen3_variant,
            eager,
            width,
            height,
            steps,
            guidance,
        )
        .await?
    } else {
        // Try remote server first
        let client = match &host {
            Some(h) => MoldClient::new(h),
            None => MoldClient::from_env(),
        };

        let pb = ProgressBar::new_spinner();
        if piped {
            // Don't render spinner to stdout when piped — it would corrupt binary output.
            // Draw to stderr instead.
            pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
        }
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Connecting to server...");
        pb.enable_steady_tick(Duration::from_millis(100));

        match client.generate(req.clone()).await {
            Ok(response) => {
                pb.finish_and_clear();
                response
            }
            Err(e) if MoldClient::is_model_not_found(&e) => {
                pb.finish_and_clear();
                status!(
                    "{} Model '{}' not on server — pulling...",
                    "●".cyan(),
                    model.bold()
                );
                super::pull::pull_and_configure(model).await?;
                // Retry after pull — the server can now find the model files
                let pb2 = ProgressBar::new_spinner();
                if piped {
                    pb2.set_draw_target(indicatif::ProgressDrawTarget::stderr());
                }
                pb2.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} {msg}")
                        .unwrap(),
                );
                pb2.set_message("Retrying generation...");
                pb2.enable_steady_tick(Duration::from_millis(100));
                let response = client.generate(req.clone()).await?;
                pb2.finish_and_clear();
                response
            }
            Err(e) if MoldClient::is_connection_error(&e) => {
                pb.finish_and_clear();
                status!("{} Using local GPU inference", "●".cyan());
                generate_local(
                    &req,
                    &config,
                    t5_variant,
                    qwen3_variant,
                    eager,
                    width,
                    height,
                    steps,
                    guidance,
                )
                .await?
            }
            Err(e) => return Err(e),
        }
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

            std::fs::write(&filename, &img.data)?;
            status!("{} Saved: {}", "✓".green(), filename.bold());
        }
    }

    let secs = response.generation_time_ms as f64 / 1000.0;
    status!(
        "{} Done in {:.1}s (seed: {})",
        "✓".green(),
        secs,
        response.seed_used,
    );

    Ok(())
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
    use mold_inference::{LoadStrategy, ProgressEvent};

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

                // Re-resolve model defaults now that config has the manifest values.
                // The request was built before the pull, so dimensions/guidance/steps
                // may be wrong (global defaults instead of model-specific ones).
                // Only override values the user didn't explicitly set via CLI flags.
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
    // Determine load strategy: --eager flag or MOLD_EAGER=1 env var → Eager, else Sequential
    let is_eager = eager || std::env::var("MOLD_EAGER").map_or(false, |v| v == "1");
    let load_strategy = if is_eager {
        LoadStrategy::Eager
    } else {
        LoadStrategy::Sequential
    };
    // Propagate eager flag so preflight_memory_check can see it
    if is_eager {
        std::env::set_var("MOLD_EAGER", "1");
    }
    let mut engine =
        mold_inference::create_engine(model_name, paths, effective_config, load_strategy)?;

    // Set up progress channel for UI updates from the blocking inference thread
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ProgressEvent>();
    engine.set_on_progress(Box::new(move |event| {
        let _ = tx.send(event);
    }));

    let req = req.clone();

    // Spawn inference in a blocking thread
    let handle = tokio::task::spawn_blocking(move || {
        engine.load()?;
        engine.generate(&req)
    });

    // Render progress events as they arrive (always to stderr when piped)
    let render = tokio::spawn(async move {
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
        while let Some(event) = rx.recv().await {
            match event {
                ProgressEvent::StageStart { name } => {
                    if let Some(db) = denoise_bar.take() {
                        db.finish_and_clear();
                    }
                    pb.set_message(format!("{}...", name));
                    pb.tick();
                }
                ProgressEvent::StageDone { name, elapsed } => {
                    if let Some(db) = denoise_bar.take() {
                        db.finish_and_clear();
                    }
                    pb.suspend(|| {
                        status!(
                            "  {} {} {}",
                            "✓".green(),
                            name,
                            format!("[{:.1}s]", elapsed.as_secs_f64()).dimmed(),
                        );
                    });
                }
                ProgressEvent::Info { message } => {
                    pb.suspend(|| {
                        status!("  {} {}", "·".dimmed(), message.dimmed());
                    });
                }
                ProgressEvent::DenoiseStep {
                    step,
                    total,
                    elapsed,
                } => {
                    let db = denoise_bar.get_or_insert_with(|| {
                        // Hide the spinner while denoise bar is active
                        pb.disable_steady_tick();
                        pb.set_message("");

                        let bar = ProgressBar::new(total as u64);
                        if is_piped() {
                            bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());
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
                    let it_s = if elapsed.as_secs_f64() > 0.0 {
                        1.0 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };
                    db.set_message(format!("{:.2} it/s", it_s));
                    db.set_position(step as u64);
                }
            }
        }
        if let Some(db) = denoise_bar.take() {
            db.finish_and_clear();
        }

        pb.finish_and_clear();
    });

    let result = handle.await?;
    // Wait for all progress events to be rendered
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
