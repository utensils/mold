use anyhow::Result;
use colored::Colorize;

use crate::control::CliContext;
use crate::procinfo;
use crate::theme;

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);

    match ctx.client().server_status().await {
        Ok(status) => {
            println!("{} mold server v{}", theme::icon_ok(), status.version);
            println!("{} Uptime: {}s", theme::icon_ok(), status.uptime_secs,);

            if let Some(gpu) = &status.gpu_info {
                println!(
                    "{} GPU: {} ({}/{} MB VRAM)",
                    theme::icon_ok(),
                    gpu.name,
                    gpu.vram_used_mb,
                    gpu.vram_total_mb,
                );
            } else {
                println!("{} GPU: {}", theme::icon_ok(), "not detected".dimmed());
            }

            println!(
                "{} Busy: {}",
                theme::icon_ok(),
                if status.busy {
                    "yes".yellow()
                } else {
                    "no".dimmed()
                }
            );

            if let Some(job) = &status.current_generation {
                println!("{} Active model: {}", theme::icon_ok(), job.model);
                println!(
                    "{} Active prompt SHA-256: {}",
                    theme::icon_ok(),
                    job.prompt_sha256.dimmed()
                );
                println!(
                    "{} Active for: {:.1}s",
                    theme::icon_ok(),
                    job.elapsed_ms as f64 / 1000.0
                );
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
        Err(_) => {
            let procs = procinfo::find_mold_processes();
            if procs.is_empty() {
                println!(
                    "{} No mold server or processes running.",
                    theme::icon_bullet()
                );
                println!(
                    "  {} start the server with {}",
                    theme::prefix_hint(),
                    "mold serve".bold()
                );
            } else {
                println!("{}", "Running mold processes:".bold());
                for p in &procs {
                    let args_display = if p.args.is_empty() {
                        String::new()
                    } else {
                        let joined = p.args.join(" ");
                        if joined.len() > 60 {
                            let truncated: String = joined.chars().take(57).collect();
                            format!(" {truncated}...")
                        } else {
                            format!(" {joined}")
                        }
                    };
                    println!(
                        "  {} {} {}{} {} ({})",
                        theme::icon_bullet(),
                        format!("[{}]", p.pid).dimmed(),
                        p.subcommand.green(),
                        args_display.dimmed(),
                        procinfo::format_duration(p.run_time_secs).dimmed(),
                        procinfo::format_memory_mb(p.memory_bytes).dimmed(),
                    );
                }
                println!();
                println!(
                    "  {} no mold server running — start with {}",
                    theme::prefix_hint(),
                    "mold serve".bold()
                );
            }
        }
    }

    Ok(())
}
