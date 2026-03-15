use anyhow::{bail, Result};
use colored::Colorize;
use mold_core::config::Config;
use mold_core::download::{pull_model, DownloadError};
use mold_core::manifest::{find_manifest, known_manifests, resolve_model_name};

/// Download a model and write its config. Returns the updated Config.
pub async fn pull_and_configure(model: &str) -> Result<Config> {
    let canonical = resolve_model_name(model);

    let manifest = match find_manifest(&canonical) {
        Some(m) => m,
        None => {
            eprintln!("{} Unknown model: {}", "✗".red().bold(), model.bold());
            eprintln!();
            eprintln!("Available models:");
            for m in known_manifests() {
                eprintln!(
                    "  {:<20} {:>5.1}GB  {}",
                    m.name.bold(),
                    m.size_gb,
                    m.description.dimmed(),
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
            bail!("unknown model: {model}");
        }
    };

    let total_gb = manifest.size_gb + mold_core::manifest::SHARED_COMPONENTS_GB;
    println!(
        "{} Pulling {} ({:.1}GB transformer, {:.1}GB total with shared components)",
        "●".cyan(),
        manifest.name.bold(),
        manifest.size_gb,
        total_gb,
    );
    println!("  {}", manifest.description.dimmed());
    println!();

    let paths = match pull_model(&manifest).await {
        Ok(paths) => paths,
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
            bail!("gated model requires HuggingFace access approval");
        }
        Err(e) => {
            eprintln!();
            eprintln!("{} Download failed: {e}", "✗".red().bold());
            bail!(e);
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

    println!();
    println!("{} {} is ready!", "✓".green().bold(), manifest.name.bold());

    Ok(config)
}

pub async fn run(model: &str) -> Result<()> {
    pull_and_configure(model).await?;
    println!("  mold run \"your prompt\"");
    Ok(())
}
