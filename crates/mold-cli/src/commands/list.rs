use anyhow::Result;
use colored::Colorize;
use mold_core::MoldClient;

pub async fn run() -> Result<()> {
    let client = MoldClient::from_env();

    match client.list_models().await {
        Ok(models) => {
            println!(
                "{:<20} {:<10} {:<10} {:<10}",
                "NAME".bold(),
                "FAMILY".bold(),
                "SIZE".bold(),
                "LOADED".bold(),
            );

            for model in &models {
                let loaded = if model.is_loaded {
                    "yes".green().to_string()
                } else {
                    "no".dimmed().to_string()
                };

                println!(
                    "{:<20} {:<10} {:<10} {:<10}",
                    model.name,
                    model.family,
                    format!("{:.1} GB", model.size_gb),
                    loaded,
                );
            }

            if models.is_empty() {
                println!("{}", "No models found.".dimmed());
            }
        }
        Err(_) => {
            // Server not running — show known models from registry
            println!(
                "{} Server not reachable, showing known models:",
                "!".yellow()
            );
            println!();
            println!(
                "{:<20} {:<10} {:<10}",
                "NAME".bold(),
                "FAMILY".bold(),
                "SIZE".bold(),
            );
            println!("{:<20} {:<10} {:<10}", "flux-schnell", "flux", "23.8 GB");
            println!("{:<20} {:<10} {:<10}", "flux-dev", "flux", "23.8 GB");
        }
    }

    Ok(())
}
