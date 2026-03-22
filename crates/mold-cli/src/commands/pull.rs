use anyhow::Result;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::download::DownloadError;
use mold_core::manifest::{find_manifest, known_manifests, resolve_model_name};

use crate::output::status;
use crate::AlreadyReported;

/// Download a model and write its config. Returns the updated Config.
pub async fn pull_and_configure(model: &str) -> Result<Config> {
    let canonical = resolve_model_name(model);

    // Pre-flight: print status and validate manifest exists (for CLI-specific error formatting)
    let manifest = match find_manifest(&canonical) {
        Some(m) => m,
        None => {
            print_unknown_model_error(model);
            return Err(AlreadyReported.into());
        }
    };

    let (total_bytes, remaining_bytes) = mold_core::manifest::compute_download_size(manifest);
    let total_gb = total_bytes as f64 / 1_073_741_824.0;
    let remaining_gb = remaining_bytes as f64 / 1_073_741_824.0;
    let cached_gb = total_gb - remaining_gb;
    if cached_gb > 0.1 {
        status!(
            "{} Pulling {} ({:.1}GB to download, {:.1}GB already cached)",
            "●".cyan(),
            manifest.name.bold(),
            remaining_gb,
            cached_gb,
        );
    } else {
        status!(
            "{} Pulling {} ({:.1}GB to download)",
            "●".cyan(),
            manifest.name.bold(),
            total_gb,
        );
    }
    status!(
        "  {}",
        crate::output::colorize_description(&manifest.description)
    );
    status!("");

    // Delegate to core pull_and_configure
    let (config, _paths) = mold_core::download::pull_and_configure(model)
        .await
        .map_err(|e| -> anyhow::Error {
            match e {
                DownloadError::UnknownModel { .. } => {
                    print_unknown_model_error(model);
                }
                DownloadError::Unauthorized { repo, .. } => {
                    eprintln!();
                    eprintln!("{} Authentication required for {repo}", "✗".red().bold());
                    eprintln!();
                    eprintln!("  1. Create a token at: https://huggingface.co/settings/tokens");
                    eprintln!("     (select at least \"Read\" access)");
                    eprintln!("  2. Set: export HF_TOKEN=hf_...");
                    eprintln!("     Or run: huggingface-cli login");
                    eprintln!("  3. Retry: mold pull {}", canonical);
                    if std::env::var("HF_TOKEN").is_ok() {
                        eprintln!();
                        eprintln!(
                            "  {} HF_TOKEN is set but was rejected — it may be invalid or expired.",
                            "!".yellow().bold()
                        );
                    }
                }
                DownloadError::GatedModel { .. } => {
                    eprintln!();
                    eprintln!(
                        "{} This model requires access approval on HuggingFace.",
                        "✗".red().bold()
                    );
                    eprintln!();

                    let gated_repo = manifest
                        .files
                        .iter()
                        .find(|f| f.gated)
                        .map(|f| f.hf_repo.as_str())
                        .unwrap_or("the model repository");

                    eprintln!("  1. Visit: https://huggingface.co/{gated_repo}");
                    eprintln!("  2. Accept the license agreement");
                    eprintln!("  3. Create a token at: https://huggingface.co/settings/tokens");
                    eprintln!("  4. Set: export HF_TOKEN=hf_...");
                    eprintln!("  5. Retry: mold pull {}", canonical);
                }
                other => {
                    eprintln!();
                    eprintln!("{} Download failed: {other}", "✗".red().bold());
                }
            }
            AlreadyReported.into()
        })?;

    status!("");
    status!("{} {} is ready!", "✓".green().bold(), canonical.bold());

    Ok(config)
}

fn print_unknown_model_error(model: &str) {
    eprintln!("{} Unknown model: {}", "✗".red().bold(), model.bold());
    eprintln!();
    eprintln!("Available models:");
    let all = known_manifests();
    let nw = all.iter().map(|m| m.name.len()).max().unwrap_or(4) + 2;
    for m in all {
        let total_bytes = mold_core::manifest::total_download_size(m);
        let total_gb = total_bytes as f64 / 1_073_741_824.0;
        eprintln!(
            "  {:<nw$} {:>5.1}GB  {}",
            m.name.bold(),
            total_gb,
            crate::output::colorize_description(&m.description),
            nw = nw,
        );
    }
    eprintln!();
    eprintln!("Usage: mold pull <model>");
}

pub async fn run(model: &str) -> Result<()> {
    pull_and_configure(model).await?;
    status!("  mold run \"your prompt\"");
    Ok(())
}
