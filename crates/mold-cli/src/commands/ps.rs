use anyhow::Result;
use colored::Colorize;
use mold_core::types::{DeviceAdminState, DeviceHealth, GpuWorkerState};

use crate::control::CliContext;
use crate::procinfo;
use crate::theme;

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);

    match ctx.client().server_status().await {
        Ok(status) => {
            println!("{} mold server v{}", theme::icon_ok(), status.version);
            println!("{} Uptime: {}s", theme::icon_ok(), status.uptime_secs,);

            // Prefer the stable-ID inventory; mixed-version hosts retain the
            // legacy status fallback below.
            if let Ok(devices) = ctx.client().devices().await {
                println!();
                for device in devices.devices {
                    let model = device
                        .loaded_models
                        .first()
                        .map_or("(none)", String::as_str);
                    let state_str = match (device.admin_state, device.health) {
                        (DeviceAdminState::Draining, _) => {
                            "[finishing current work]".yellow().to_string()
                        }
                        (_, DeviceHealth::Unavailable | DeviceHealth::Poisoned) => {
                            format!("[{:?}]", device.health)
                                .to_lowercase()
                                .red()
                                .to_string()
                        }
                        _ if device.schedulable => "[enabled]".green().to_string(),
                        _ => format!("[{:?}]", device.admin_state)
                            .to_lowercase()
                            .dimmed()
                            .to_string(),
                    };
                    let vram_used_gb =
                        device.memory.used_bytes.unwrap_or(0) as f64 / 1_073_741_824.0;
                    let vram_total_gb =
                        device.memory.total_bytes.unwrap_or(0) as f64 / 1_073_741_824.0;
                    println!(
                        "{} · GPU {} ({}, {:.0}GB):  {:<20} {}  VRAM: {:.1}/{:.1} GB",
                        device.id,
                        device
                            .ordinal
                            .map_or_else(|| "—".into(), |value| value.to_string()),
                        device.name,
                        vram_total_gb,
                        model.green(),
                        state_str,
                        vram_used_gb,
                        vram_total_gb,
                    );
                }
                if let (Some(depth), Some(capacity)) = (status.queue_depth, status.queue_capacity) {
                    println!("Queue: {}/{}", depth, capacity);
                }
            } else if let Some(gpus) = &status.gpus {
                println!();
                for gpu in gpus {
                    let model = gpu.loaded_model.as_deref().unwrap_or("(none)");
                    let state_str = match gpu.state {
                        GpuWorkerState::Generating => "[generating]".yellow().to_string(),
                        GpuWorkerState::Idle => "[idle]".dimmed().to_string(),
                        GpuWorkerState::Loading => "[loading]".cyan().to_string(),
                        GpuWorkerState::Degraded => "[degraded]".red().to_string(),
                    };
                    println!("GPU {}: {:<20} {}", gpu.ordinal, model.green(), state_str);
                }
            } else {
                // Single-GPU fallback display.
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
            }

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
            println!(
                "  {} no mold server running — start with {}",
                theme::prefix_hint(),
                "mold serve".bold()
            );
        }
    }

    // Always scan for running mold processes.
    let procs = procinfo::find_mold_processes();
    if !procs.is_empty() {
        println!();
        println!("{}", "Running mold processes:".bold());
        for p in &procs {
            let args_display = if p.args.is_empty() {
                String::new()
            } else {
                let joined = p.args.join(" ");
                if joined.chars().count() > 60 {
                    let truncated: String = joined.chars().take(57).collect();
                    format!(" {truncated}...")
                } else {
                    format!(" {joined}")
                }
            };
            let threads = if p.thread_count > 1 {
                format!(", {} threads", p.thread_count)
            } else {
                String::new()
            };
            println!(
                "  {} {} {}{} {} ({}{})",
                theme::icon_bullet(),
                format!("[{}]", p.pid).dimmed(),
                p.subcommand.green(),
                args_display.dimmed(),
                procinfo::format_duration(p.run_time_secs).dimmed(),
                procinfo::format_memory_mb(p.memory_bytes).dimmed(),
                threads.dimmed(),
            );
        }
    }

    Ok(())
}
