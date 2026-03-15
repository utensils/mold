use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{Config, GenerateRequest, GenerateResponse, MoldClient, OutputFormat};
use std::time::Duration;

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
) -> Result<()> {
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    // Load config and pull model-specific defaults.
    let config = Config::load_or_default();
    let model_cfg = config.model_config(model);

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
        println!("{} {} — {}", "●".green(), model.bold(), desc.dimmed());
    }
    println!(
        "{} Generating {}x{} ({} steps, guidance {:.1})",
        "●".cyan(),
        effective_width,
        effective_height,
        effective_steps,
        effective_guidance,
    );

    let response = if local {
        // --local: skip server, go straight to local inference
        println!("{} Using local GPU inference", "●".cyan());
        generate_local(&req, &config).await?
    } else {
        // Try remote server first
        let client = match &host {
            Some(h) => MoldClient::new(h),
            None => MoldClient::from_env(),
        };

        let pb = ProgressBar::new_spinner();
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
            Err(e) if MoldClient::is_connection_error(&e) => {
                pb.finish_and_clear();
                println!("{} Using local GPU inference", "●".cyan());
                generate_local(&req, &config).await?
            }
            Err(e) => return Err(e),
        }
    };

    for img in &response.images {
        let filename = match &output {
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
        println!("{} Saved: {}", "✓".green(), filename.bold());
    }

    let secs = response.generation_time_ms as f64 / 1000.0;
    println!(
        "{} Done in {:.1}s (seed: {})",
        "✓".green(),
        secs,
        response.seed_used,
    );

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
async fn generate_local(req: &GenerateRequest, config: &Config) -> Result<GenerateResponse> {
    use mold_core::manifest::find_manifest;
    use mold_core::{validate_generate_request, ModelPaths};
    use mold_inference::{FluxEngine, InferenceEngine, ProgressEvent};

    validate_generate_request(req).map_err(|e| anyhow::anyhow!(e))?;

    let model_name = req.model.clone();
    let (paths, auto_config);
    let effective_config: &Config;
    match ModelPaths::resolve(&model_name, config) {
        Some(p) => {
            paths = p;
            effective_config = config;
        }
        None => {
            // Auto-pull: if a manifest exists, download the model automatically
            if find_manifest(&model_name).is_some() {
                println!(
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

    let is_schnell = effective_config.model_config(&model_name).is_schnell;
    let mut engine = FluxEngine::new(model_name, paths, is_schnell);

    // Set up progress channel for UI updates from the blocking inference thread
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ProgressEvent>();
    engine.set_on_progress(move |event| {
        let _ = tx.send(event);
    });

    let req = req.clone();

    // Spawn inference in a blocking thread
    let handle = tokio::task::spawn_blocking(move || {
        engine.load()?;
        engine.generate(&req)
    });

    // Render progress events as they arrive
    let render = tokio::spawn(async move {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        pb.enable_steady_tick(Duration::from_millis(100));

        while let Some(event) = rx.recv().await {
            match event {
                ProgressEvent::StageStart { name } => {
                    pb.set_message(format!("{}...", name));
                }
                ProgressEvent::StageDone { name, elapsed } => {
                    pb.suspend(|| {
                        println!(
                            "  {} {} {}",
                            "✓".green(),
                            name,
                            format!("[{:.1}s]", elapsed.as_secs_f64()).dimmed(),
                        );
                    });
                }
                ProgressEvent::Info { message } => {
                    pb.suspend(|| {
                        println!("  {} {}", "·".dimmed(), message.dimmed());
                    });
                }
            }
        }

        pb.finish_and_clear();
    });

    let result = handle.await?;
    // Wait for all progress events to be rendered
    let _ = render.await;
    result
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
async fn generate_local(_req: &GenerateRequest, _config: &Config) -> Result<GenerateResponse> {
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
}
