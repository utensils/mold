use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use mold_core::{GenerateRequest, MoldClient, OutputFormat};
use std::time::Duration;

#[allow(clippy::too_many_arguments)]
pub async fn run(
    prompt: &str,
    model: &str,
    output: Option<String>,
    width: u32,
    height: u32,
    steps: Option<u32>,
    seed: Option<u64>,
    batch: u32,
    host: Option<String>,
    format: &str,
) -> Result<()> {
    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    let default_steps = match model {
        "flux-dev" => 25,
        _ => 4,
    };

    let client = match &host {
        Some(h) => MoldClient::new(h),
        None => MoldClient::from_env(),
    };

    let req = GenerateRequest {
        prompt: prompt.to_string(),
        model: model.to_string(),
        width,
        height,
        steps: steps.unwrap_or(default_steps),
        seed,
        batch_size: batch,
        output_format,
    };

    println!(
        "{} Generating with {} ({}x{}, {} steps)...",
        "●".green(),
        model.bold(),
        width,
        height,
        req.steps,
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
                    format!("mold-output-{timestamp}.{ext}")
                } else {
                    format!("mold-output-{timestamp}-{}.{ext}", img.index)
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
