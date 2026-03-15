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
        "{} Generating {}x{} ({} steps, guidance {:.1})...",
        "●".cyan(),
        effective_width,
        effective_height,
        effective_steps,
        effective_guidance,
    );

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let response = if local {
        // --local: skip server, go straight to local inference
        pb.set_message("Loading model for local inference...");
        generate_local(&req, &config).await?
    } else {
        // Try remote server first, fall back to local on connection error
        let client = match &host {
            Some(h) => MoldClient::new(h),
            None => MoldClient::from_env(),
        };

        pb.set_message("Running inference...");
        match client.generate(req.clone()).await {
            Ok(response) => response,
            Err(e) if MoldClient::is_connection_error(&e) => {
                pb.set_message("No server found, falling back to local inference...");
                eprintln!(
                    "{} No mold server running, falling back to local inference",
                    "●".yellow(),
                );
                generate_local(&req, &config).await?
            }
            Err(e) => return Err(e),
        }
    };

    pb.finish_and_clear();

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
                if batch == 1 {
                    format!("mold-{model}-{timestamp}.{ext}")
                } else {
                    format!("mold-{model}-{timestamp}-{}.{ext}", img.index)
                }
            }
        };

        std::fs::write(&filename, &img.data)?;
        println!("{} Saved: {}", "✓".green(), filename.bold());
    }

    println!(
        "{} Done in {}ms (seed: {})",
        "✓".green(),
        response.generation_time_ms,
        response.seed_used,
    );

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
async fn generate_local(req: &GenerateRequest, config: &Config) -> Result<GenerateResponse> {
    use mold_core::{ModelPaths, validate_generate_request};
    use mold_inference::{FluxEngine, InferenceEngine};

    validate_generate_request(req).map_err(|e| anyhow::anyhow!(e))?;

    let model_name = req.model.clone();
    let paths = ModelPaths::resolve(&model_name, config).ok_or_else(|| {
        anyhow::anyhow!(
            "no model paths configured for '{}'. Add [models.{}] to ~/.mold/config.toml \
             or set MOLD_TRANSFORMER_PATH / MOLD_VAE_PATH / MOLD_T5_PATH / MOLD_CLIP_PATH \
             / MOLD_T5_TOKENIZER_PATH / MOLD_CLIP_TOKENIZER_PATH env vars.",
            model_name,
            model_name,
        )
    })?;

    let is_schnell = config.model_config(&model_name).is_schnell;
    let mut engine = FluxEngine::new(model_name, paths, is_schnell);

    let req = req.clone();
    tokio::task::spawn_blocking(move || {
        engine.load()?;
        engine.generate(&req)
    })
    .await?
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
async fn generate_local(_req: &GenerateRequest, _config: &Config) -> Result<GenerateResponse> {
    anyhow::bail!(
        "No mold server running and this binary was built without GPU support.\n\
         Either start a server with `mold serve` or rebuild with --features cuda"
    )
}
