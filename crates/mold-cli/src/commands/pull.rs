use anyhow::Result;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::download::{pull_model, DownloadError};
use mold_core::manifest::{find_manifest, known_manifests, resolve_model_name};

use crate::output::status;
use crate::AlreadyReported;

/// Download a model and write its config. Returns the updated Config.
pub async fn pull_and_configure(model: &str) -> Result<Config> {
    let canonical = resolve_model_name(model);

    let manifest = match find_manifest(&canonical) {
        Some(m) => m,
        None => {
            eprintln!("{} Unknown model: {}", "✗".red().bold(), model.bold());
            eprintln!();
            eprintln!("Available models:");
            let all = known_manifests();
            let nw = all.iter().map(|m| m.name.len()).max().unwrap_or(4) + 2;
            for m in &all {
                eprintln!(
                    "  {:<nw$} {:>5.1}GB  {}",
                    m.name.bold(),
                    m.size_gb,
                    m.description.dimmed(),
                    nw = nw,
                );
            }
            eprintln!();
            eprintln!(
                "{}",
                format!(
                    "Sizes are transformer only. First pull also downloads {:.1}GB of shared components (T5, CLIP, VAE).",
                    mold_core::manifest::SHARED_COMPONENTS_GB
                ).dimmed(),
            );
            eprintln!("Usage: mold pull <model>");
            return Err(AlreadyReported.into());
        }
    };

    let shared_gb = match manifest.family.as_str() {
        "sd15" => mold_core::manifest::SHARED_SD15_COMPONENTS_GB,
        "sdxl" => mold_core::manifest::SHARED_SDXL_COMPONENTS_GB,
        "z-image" => mold_core::manifest::SHARED_ZIMAGE_COMPONENTS_GB,
        _ => mold_core::manifest::SHARED_COMPONENTS_GB,
    };
    let total_gb = manifest.size_gb + shared_gb;
    status!(
        "{} Pulling {} ({:.1}GB transformer, {:.1}GB total with shared components)",
        "●".cyan(),
        manifest.name.bold(),
        manifest.size_gb,
        total_gb,
    );
    status!("  {}", manifest.description.dimmed());
    status!("");

    let paths = match pull_model(&manifest).await {
        Ok(paths) => paths,
        Err(DownloadError::Unauthorized { repo, .. }) => {
            eprintln!();
            eprintln!("{} Authentication required for {repo}", "✗".red().bold());
            eprintln!();
            eprintln!("  1. Create a token at: https://huggingface.co/settings/tokens");
            eprintln!("     (select at least \"Read\" access)");
            eprintln!("  2. Set: export HF_TOKEN=hf_...");
            eprintln!("     Or run: huggingface-cli login");
            eprintln!("  3. Retry: mold pull {}", manifest.name);
            if std::env::var("HF_TOKEN").is_ok() {
                eprintln!();
                eprintln!(
                    "  {} HF_TOKEN is set but was rejected — it may be invalid or expired.",
                    "!".yellow().bold()
                );
            }
            return Err(AlreadyReported.into());
        }
        Err(DownloadError::GatedModel { .. }) => {
            eprintln!();
            eprintln!(
                "{} This model requires access approval on HuggingFace.",
                "✗".red().bold()
            );
            eprintln!();

            // Find the gated repo for a helpful message
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
            eprintln!("  5. Retry: mold pull {}", manifest.name);
            return Err(AlreadyReported.into());
        }
        Err(e) => {
            eprintln!();
            eprintln!("{} Download failed: {e}", "✗".red().bold());
            return Err(AlreadyReported.into());
        }
    };

    // Save to config
    let mut config = Config::load_or_default();
    let model_config = manifest.to_model_config(&paths);

    // Auto-set default_model if no config existed before
    if !Config::exists_on_disk() {
        config.default_model = manifest.name.clone();
    }

    config.upsert_model(manifest.name.clone(), model_config);
    config.save()?;

    status!("");
    status!("{} {} is ready!", "✓".green().bold(), manifest.name.bold());

    Ok(config)
}

pub async fn run(model: &str) -> Result<()> {
    pull_and_configure(model).await?;
    status!("  mold run \"your prompt\"");
    Ok(())
}
