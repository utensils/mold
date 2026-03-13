use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{Config, GenerateRequest, MoldClient, OutputFormat};
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
) -> Result<()> {
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    // Load config and pull model-specific defaults.
    let config = Config::load_or_default();
    let model_cfg = config.model_config(model);

    let effective_width = width.unwrap_or_else(|| model_cfg.effective_width(&config));
    let effective_height = height.unwrap_or_else(|| model_cfg.effective_height(&config));
    let effective_steps = steps.unwrap_or_else(|| model_cfg.effective_steps(&config));
    let effective_guidance = guidance.unwrap_or_else(|| model_cfg.effective_guidance());

    let client = match &host {
        Some(h) => MoldClient::new(h),
        None => MoldClient::from_env(),
    };

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
    pb.set_message("Running inference...");
    pb.enable_steady_tick(Duration::from_millis(100));

    let response = client.generate(req).await?;
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
