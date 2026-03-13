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
            // Server not running — show config-defined models locally.
            let config = Config::load_or_default();
            println!(
                "{} Server not reachable — showing local config:",
                "!".yellow()
            );
            println!();

            if config.models.is_empty() {
                println!("{} No models in ~/.mold/config.toml", "●".dimmed());
                println!("Add [models.<name>] sections to configure models.");
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
        }
    }

    Ok(())
}
