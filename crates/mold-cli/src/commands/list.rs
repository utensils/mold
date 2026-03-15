use anyhow::Result;
use colored::Colorize;
use mold_core::{Config, MoldClient};

pub async fn run() -> Result<()> {
    let client = MoldClient::from_env();

    match client.list_models_extended().await {
        Ok(models) => {
            println!(
                "{:<18} {:<10} {:<7} {:<9} {:<8} {:<7} {}",
                "NAME".bold(),
                "FAMILY".bold(),
                "STEPS".bold(),
                "GUIDANCE".bold(),
                "WIDTH".bold(),
                "HEIGHT".bold(),
                "DESCRIPTION".bold(),
            );
            println!("{}", "─".repeat(90).dimmed());

            for model in &models {
                let name = if model.is_loaded {
                    format!("{} ●", model.name).green().to_string()
                } else {
                    model.name.clone()
                };
                println!(
                    "{:<18} {:<10} {:<7} {:<9} {:<8} {:<7} {}",
                    name,
                    model.family,
                    model.defaults.default_steps,
                    format!("{:.1}", model.defaults.default_guidance),
                    model.defaults.default_width,
                    model.defaults.default_height,
                    model.defaults.description.dimmed(),
                );
            }

            if models.is_empty() {
                println!("{}", "No models configured.".dimmed());
            }
        }
        Err(_) => {
            let config = Config::load_or_default();

            if config.models.is_empty() {
                println!("{} No models configured.", "●".dimmed());
            } else {
                println!(
                    "{:<18} {:<7} {:<9} {:<8} {:<7} {}",
                    "NAME".bold(),
                    "STEPS".bold(),
                    "GUIDANCE".bold(),
                    "WIDTH".bold(),
                    "HEIGHT".bold(),
                    "DESCRIPTION".bold(),
                );
                println!("{}", "─".repeat(75).dimmed());
                for (name, mcfg) in &config.models {
                    println!(
                        "{:<18} {:<7} {:<9} {:<8} {:<7} {}",
                        name,
                        mcfg.effective_steps(&config),
                        format!("{:.1}", mcfg.effective_guidance()),
                        mcfg.effective_width(&config),
                        mcfg.effective_height(&config),
                        mcfg.description.as_deref().unwrap_or("").dimmed(),
                    );
                }
            }

            // Show available-to-pull models
            let manifests = mold_core::manifest::known_manifests();
            let available: Vec<_> = manifests
                .iter()
                .filter(|m| !config.models.contains_key(&m.name))
                .collect();

            if !available.is_empty() {
                println!();
                println!("Available to pull:");
                for m in &available {
                    println!(
                        "  {:<20} {:>5.1}GB  {}",
                        m.name.bold(),
                        m.size_gb,
                        m.description.dimmed(),
                    );
                }
                println!();
                println!("Use {} to download.", "mold pull <model>".bold(),);
            }
        }
    }

    Ok(())
}
