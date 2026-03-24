use anyhow::Result;
use colored::Colorize;

use crate::control::CliContext;
use crate::ui::print_server_unavailable;

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);

    match ctx.client().server_status().await {
        Ok(status) => {
            println!("{} mold server v{}", "●".green(), status.version);
            println!("{} Uptime: {}s", "●".green(), status.uptime_secs,);

            if let Some(gpu) = &status.gpu_info {
                println!(
                    "{} GPU: {} ({}/{} MB VRAM)",
                    "●".green(),
                    gpu.name,
                    gpu.vram_used_mb,
                    gpu.vram_total_mb,
                );
            } else {
                println!("{} GPU: {}", "●".green(), "not detected".dimmed());
            }

            println!();
            if status.models_loaded.is_empty() {
                println!("{}", "No models loaded.".dimmed());
            } else {
                println!("{}", "Loaded models:".bold());
                for model in &status.models_loaded {
                    println!("  - {}", model.green());
                }
            }
        }
        Err(e) => {
            print_server_unavailable(ctx.client().host(), &e);
        }
    }

    Ok(())
}
